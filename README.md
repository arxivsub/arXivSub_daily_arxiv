# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-13 | 今日论文总数: 848

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. How Does Differential Privacy Affect Social Bias in LLMs? A Systematic Evaluation

**arXiv ID:** 2605.11195 | [PDF](https://arxiv.org/pdf/2605.11195v1)

**作者:** Eduardo Tenorio `[一作]` (University of Arkansas), Xintao Wu `[通讯]` (University of Arkansas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了差分隐私预训练的LLM对社会偏见的影响，并在句子打分、文本生成、表格分类和问答四种范式下进行系统评估。

**💡 创新点**

首次将差分隐私预训练与多范式偏见评估相结合，揭示了 logit 级别与输出级别偏见的脱节。

**🔧 技术方法**

采用 DP‑SGD 训练的 VaultGemma‑1B，与非 DP 的 Gemma‑3‑1B‑PT 与 Gemma‑2‑2B 进行对比，并使用 SS、ICAT、Bias Gap 等标准偏见指标。

**📊 数据集**

使用 StereoSet、WinoBias、CrowS‑Pairs、BOLD、HolisticBias、Adult Census、COMPAS、German Credit、BBQ 等公开数据集。

**📈 对比分析**

与非 DP 基线相比，DP 模型在句子打分任务中 SS 降低、ICAT 上升，显示偏见减弱；在文本生成、表格分类与问答任务中差异不显著或无优势。

**⚠️ 局限性**

仅有单一公开 DP 预训练模型，且基线模型未进行指令调优，导致在生成与结构化任务中的性能不足，难以充分验证偏见评估的可靠性。

---

## 2. MMTB: Evaluating Terminal Agents on Multimedia-File Tasks

**arXiv ID:** 2605.10966 | [PDF](https://arxiv.org/pdf/2605.10966v1)

**作者:** Chiyeong Heo `[一作]` (POSTECH), Jungseul Ok `[通讯]` (POSTECH)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多媒体终端工作流程基准MMTB，并开发了可对音频、视频文件提供原生感知的终端代理工具包Terminus-MM。

**💡 创新点**

创新点在于：①把多媒体文件的实际处理任务纳入终端代理评估；②将多媒体感知嵌入终端代理，支持工作空间感知化工具暴露；③在多媒体感知与终端工具使用之间进行系统性对比与分析。

**🔧 技术方法**

技术手段包括：终端代理框架（Terminus系列），多模态感知工具（音频、视频、图像），Harbor任务格式，自动化构建与过滤管道，基于verifier的artifact-level评估。

**📊 数据集**

使用MMTB数据集，包含105个基于真实业界任务的多媒体终端任务，涉及音频、视频、图像等文件，总计536个媒体文件，约6小时54分钟的多媒体时长。

**📈 对比分析**

比较方法：在相同Harbor工作空间和10分钟交互预算下，对比不同感知级别（文本、图像、音频、视频）和不同模型后端（Gemini、Qwen、GPT）的成功率、部分成功率及API成本。实验显示，完整原生多模态感知（Terminus-MM）使Binary成功率提升至约37%，Partial成功率至47%，且API成本最低，证明原生多媒体感知显著提升终端代理性能。

**⚠️ 局限性**

局限性包括：缺乏与人类专家的直接比较；仅评估600秒预算，未进行预算扩展研究；对多媒体转换工具的依赖导致模型推理与工具操作的错误分布不均；未探讨更高级的多模态推理与终端交互设计。

---

## 3. TMPO: Trajectory Matching Policy Optimization for Diverse and Efficient Diffusion Alignment

**arXiv ID:** 2605.10983 | [PDF](https://arxiv.org/pdf/2605.10983v1)

**作者:** Jiaming Li `[一作]` (HUST), Bowen Zhou `[通讯]` (Tsinghua University)

**通讯引用:** 16434 | [OpenAlex ID](https://openalex.org/A5107808331)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种通过轨迹级奖励分布匹配的强化学习框架（TMPO），用于对扩散模型进行对齐。

**💡 创新点**

创新点在于用 Softmax‑Trajectory Balance 替代传统奖励最大化，实现前向 KL 的模式覆盖；同时引入动态随机树采样（Dynamic Stochastic Tree Sampling）显著减少计算成本。

**🔧 技术方法**

采用前向‑后向概率流网络思想、Boltzmann 目标分布、树形共享前缀的 SDE 采样，以及基于熵正则化的轨迹优势梯度。

**📊 数据集**

使用 FLUX.1‑dev 作为基准模型，并在 GenEval、OCR（视觉文本渲染）以及 HPDv2 的人类偏好数据集上进行实验。

**📈 对比分析**

与 Flow‑GRPO、MixGRPO、TreeGRPO、GARDO 等现有 GRPO/Flow‑GFlowNet 方法对比，TMPO 在多项任务中平均提升生成多样性 9.1%，在奖励、效率与多样性三者的 Pareto 前沿上表现最优，同时训练时间下降约 20%。

**⚠️ 局限性**

局限在于对奖励模型质量高度依赖，且仅在文本到图像（T2I）任务上验证，未来可扩展至视频、3D 或机器人控制等领域。

---

## 4. PresentAgent-2: Towards Generalist Multimodal Presentation Agents

**arXiv ID:** 2605.11363 | [PDF](https://arxiv.org/pdf/2605.11363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 5. ABRA: Agent Benchmark for Radiology Applications

**arXiv ID:** 2605.11224 | [PDF](https://arxiv.org/pdf/2605.11224v1)

**作者:** Bulat Maksudov `[一作]` (Dublin City University), Alessandra Mileo `[通讯]` (Dublin City University)

**通讯引用:** 1675 | [OpenAlex ID](https://openalex.org/A5016411514)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了ABRA基准，构建了一个可执行的放射科工作站环境（OHIF Viewer + Orthanc PACS）并通过统一的函数调用接口设计了655个多难度、8种任务类型，评估LLM代理在真实影像环境中的工具使用与视觉感知能力。

**💡 创新点**

首创将LLM代理嵌入实时DICOM工作站，并提供可切换的oracle/real任务配对，能够分离工具协调与视觉感知的瓶颈；同时实现了完整的规划、执行与结果自动化评分体系。

**🔧 技术方法**

利用OHIF Viewer、Orthanc PACS、Puppeteer headless浏览器、OpenAI函数调用接口、自动化规划/执行/结果评分器，结合预处理流水线将DICOM像素转换为可分割的PNG。

**📊 数据集**

使用了三大公开TCIA数据集：LIDC-IDRI（胸部CT）、Duke Breast Cancer MRI（乳腺MRI）以及NLST New-Lesion LongCT（随访CT）。

**📈 对比分析**

对10个模型（5闭源API，5开源检查点）进行评测，得分由Planning、Execution、Outcome组成。Execution和Planning几乎满分，但在真实影像任务中Outcome仅为0–0.25，oracle任务则可达0.69–1.00，表明视觉感知是主要瓶颈。

**⚠️ 局限性**

主要局限在于视觉感知能力不足；参考轨迹由程序生成，规划分数可能偏高；任务规模有限；尚未验证长期临床安全性与实用性。

---

## 6. A Comparative Study of Federated Learning Aggregation Strategies under Homogeneous and Heterogeneous Data Distributions

**arXiv ID:** 2605.11010 | [PDF](https://arxiv.org/pdf/2605.11010v1)

**作者:** Antonios Makris `[一作]` (National Technical University Of Athens), Konstantinos Tserpes `[通讯]` (National Technical University Of Athens)

**通讯引用:** 2637 | [OpenAlex ID](https://openalex.org/A5021420035)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多种联邦学习聚合策略在IID与非IID数据下的性能与系统效率进行系统性实验比较。

**💡 创新点**

系统性展示不同聚合策略在不同数据异质性和数据集上的权衡，并指出无统一最佳方案，揭示了聚合策略选择的多重考量。

**🔧 技术方法**

采用Flower框架进行联邦学习实验，使用FedAvg、FedAvgM、FedAdam、FedAdagrad、FedMedian、FedProx和DP聚合，并在客户端使用SGD或Adam进行本地训练。

**📊 数据集**

使用MNIST、FMNIST和CIFAR‑10三个图像分类基准数据集。

**📈 对比分析**

通过集中精度、损失、聚合/训练/通信时间等指标，在10/20客户端、25/50轮实验中进行比较，结果显示FedAdam/FedAdagrad在IID下精度最高，FedMedian/FedProx在非IID下鲁棒性好，DP聚合精度最低且聚合耗时最高。

**⚠️ 局限性**

实验规模有限，未覆盖更大客户端数、异构硬件、动态缺失和更高级鲁棒聚合；DP仅在简单设置下实现，未评估更强隐私预算与模型实用性的平衡。

---

## 7. A Proof-of-Concept Simulation-Driven Digital Twin Framework for Decision-Aware Diabetes Modeling

**arXiv ID:** 2605.11247 | [PDF](https://arxiv.org/pdf/2605.11247v1)

**作者:** Zarrin Monirzadeh `[一作]` `[通讯]`, Zarrin Monirzadeh

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向决策的多型糖尿病数字孪生框架，结合预测与反事实仿真实现可解释的干预评估。

**💡 创新点**

将时间序列预测、因果图和反事实模拟统一到一个可重复、可解释的数字孪生架构，突破传统仅预测的限制。

**🔧 技术方法**

模块化管线：数据融合、潜在状态表示、梯度提升/随机森林/MLP预测、LSTM/Transformer时序核心、因果图建模、反事实仿真及效用排序。

**📊 数据集**

使用公开的 sklearn diabetes 回归基准数据进行模型评估，并通过人工时序扩展和 OhioT1DM CGM 数据示例进行模拟验证。

**📈 对比分析**

通过 MAE、RMSE、R² 评估四种回归模型（梯度提升、线性回归、随机森林、MLP）；梯度提升和线性回归在 MAE≈44–43、RMSE≈53–54、R²≈0.45；分类任务随机森林 AUC 0.835；反事实仿真显示干预可降低峰值并提升时间内范围。

**⚠️ 局限性**

仅在基准数据上验证，缺乏真实 CGM 时序；反事实轨迹为简化模拟；因果结构未从数据估计；无临床前向或专家评估；需进一步使用纵向临床数据验证。

---

## 8. The Midas Touch for Metric Depth

**arXiv ID:** 2605.11578 | [PDF](https://arxiv.org/pdf/2605.11578v1)

**作者:** Yu Ma `[一作]` (Tongji University), Rui Fan `[通讯]` (Tongji University)

**通讯引用:** 4268 | [OpenAlex ID](https://openalex.org/A5038867899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无监督的 Midas Touch for Depth (MTD) 方法，利用极少量 3D 种子点将相对深度映射为度量深度，实现零样本通用深度恢复。

**💡 创新点**

创新点在于将段级稀疏图优化与像素级不连续性感知的几何地理问题相结合，既纠正局部尺度不一致，又通过动态规划实现高效像素细化。

**🔧 技术方法**

主要技术包括稀疏图正则化的段级尺度校准、基于极小化地理距离的像素级 geodesic 传播、动态规划求解、以及知识蒸馏的轻量化基础网络。

**📊 数据集**

在多种室内外基准上进行评估，包括 nuScenes、DDAD、Make3D、DIODE、ETH3D、ScanNet、VOID、SUN-RGBD、HAMMER、IBims-1、KITTI、NYU-Depth V2 等，并使用 VKITTI2、Hypersim、TartanAir、SA-1B 进行蒸馏训练。

**📈 对比分析**

与现有最先进的深度补全与相对深度估计方法（如 CFormer、LRRU、BP-Net、DMD^3C、PromptDA、Marigold-DC、DepthAnythingV2、MiDaS、DepthPro、UniDepthV2 等）比较，MTD 在零样本设置下显著降低 RMSE/MAE，提升 δ₁ 率，且后端推理时间仅为 1.9 ms，能够实时部署。

**⚠️ 局限性**

局限性包括对稀疏 3D 种子点的依赖，在种子点极少时恢复性能退化到传统最小二乘；对图形构造和段尺度选择的敏感性，以及在极端遮挡或纹理缺失区域仍可能出现误差。

---

## 9. Deep Minds and Shallow Probes

**arXiv ID:** 2605.11448 | [PDF](https://arxiv.org/pdf/2605.11448v1)

**作者:** Su Hyeong Lee `[一作]` (University of Chicago), Risi Kondor `[通讯]` (University of Chicago)

**通讯引用:** 11269 | [OpenAlex ID](https://openalex.org/A5003806118)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出利用神经网络表示层的仿射不变性来约束探测器（probe）的设计，构造出唯一的多项式阶层，并通过低秩CP（Canonical Polyadic）结构实现高阶探测；同时定义probe‑visible商空间（Z(V)），实现跨模型的覆盖意识监控转移。

**💡 创新点**

创新点在于：①将仿射不变性与有限维浅层探测器相结合，推导出只有多项式空间才是唯一的非零解；②证明线性探测器是该阶层的阶数 1 成员；③提出低秩CP作为高阶探测的结构化实现；④在跨模型时仅对商空间对齐而非全隐藏状态，提供了覆盖意识的监控转移方法。

**🔧 技术方法**

使用群表示理论、仿射变换不变性分析、低秩张量（CP）参数化、SVD求商空间、线性与多项式回归/分类、Ridge/OLS对齐、以及对比实验方法。

**📊 数据集**

实验数据集包括：合成任务（XOR、N-奇偶、圆形奇偶）、语言任务（主谓一致、UDEG）、安全监控数据（Qwen、Mistral 等模型的毒性、监控等任务）以及通用文本评估（LLaMA 判定）。

**📈 对比分析**

与传统线性探测器、全二次探测器、全状态OLS、PCA 对齐等方法对比，实验表明：度为 2 的 CP 探测器在 Pythia 模型上 AUROC 提升 16–20 个百分点；商空间对齐在合成与真实数据上保持 0.99+ 的准确率，并在覆盖缺失时显著下降；零标签跨模型安全监控转移可获得 0.93–0.97 AUROC，优于少量标注训练的结果。

**⚠️ 局限性**

局限性包括：仅在最终读出层的仿射对称性下推导，未涵盖更深层或非仿射变换；高阶多项式参数量大；商空间对齐需要足够的探测器覆盖且对噪声敏感；在真实模型中可能存在近似不等价，导致转移性能下降。

---

## 10. OLIVIA: Online Learning via Inference-time Action Adaptation for Decision Making in LLM ReAct Agents

**arXiv ID:** 2605.11169 | [PDF](https://arxiv.org/pdf/2605.11169v1)

**作者:** Sheldon Yu `[一作]` (University Of San Diego), Julian McAuley `[通讯]` (University Of San Diego)

**通讯引用:** 25538 | [OpenAlex ID](https://openalex.org/A5021827617)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出OLIVIA框架，使LLM代理在部署时通过在线决策层直接从步骤级反馈中自适应动作选择，保持原有推理过程不变。

**💡 创新点**

创新点在于将LLM的动作选择接口视为线性情境赌博机，引入可在线更新的LinUCB探索策略，显式估计不确定性并实现样本高效的动作改进。

**🔧 技术方法**

技术细节包括：冻结LLM隐藏状态作为上下文、构建每个动作的线性估计器、使用UCB激励探索、采用Sherman-Morrison公式进行O(d²)增量更新。

**📊 数据集**

实验使用四个工具使用基准（ToolBench、TaskBench、TaskBench-MM、BFCL），基于公开的LLaMA/Alpaca等开源LLM模型进行评估。

**📈 对比分析**

方法与CoT、BM25、ReAct、CLIN等静态与在线基线对比；OLIVIA在所有四个基准上均取得最高F1，收敛速度最快且最终性能显著优于对手。

**⚠️ 局限性**

局限性包括：仅适用于离散工具集；需要步骤级奖励信号；对LLM隐藏层线性假设的依赖；在极大候选集或多模态场景下的计算与扩展性待进一步研究。

---

## 11. Red-Teaming Agent Execution Contexts: Open-World Security Evaluation on OpenClaw

**arXiv ID:** 2605.11047 | [PDF](https://arxiv.org/pdf/2605.11047v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 12. Efficient LLM Reasoning via Variational Posterior Guidance with Efficiency Awareness

**arXiv ID:** 2605.11019 | [PDF](https://arxiv.org/pdf/2605.11019v1)

**作者:** Zizhao Chen `[一作]` (Guangdong University of Foreign Studies), Lianxi Wang `[通讯]` (Guangdong University of Foreign Studies)

**通讯引用:** 386 | [OpenAlex ID](https://openalex.org/A5022448274)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于变分推理的后验引导高效推理框架VPG-EA，解决LLM过度思考导致的推理冗长问题。

**💡 创新点**

利用参考答案指导的后验分布突破采样瓶颈，并通过跨视图验证与变分蒸馏将高效路径知识单向迁移至先验策略。

**🔧 技术方法**

变分推理、参数共享双流采样、跨视图评估、优势门控、变分蒸馏等技术，并结合RL与ELBO理论。

**📊 数据集**

使用DeepScaleR数学推理数据集进行训练，评估数据包括GSM8K、MATH-500、AIME 2024/2025以及非数学集GPQA-Diamond与MMLU-Pro。

**📈 对比分析**

在1.5B和7B模型上与主流RL、SFT+RL及SFT基准比较，综合效率指标ε^3提升约8.7%–12.4%，token消耗减少30%以上，性能表现稳健。

**⚠️ 局限性**

局限于奖励劫持风险、对参考答案的强依赖，主要适用于可验证任务，开放域推理难以直接使用。

---

## 13. Trust Region Inverse Reinforcement Learning: Explicit Dual Ascent using Local Policy Updates

**arXiv ID:** 2605.11020 | [PDF](https://arxiv.org/pdf/2605.11020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 14. Spurious Correlation Learning in Preference Optimization: Mechanisms, Consequences, and Mitigation via Tie Training

**arXiv ID:** 2605.11134 | [PDF](https://arxiv.org/pdf/2605.11134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 15. Deep Probabilistic Unfolding for Quantized Compressive Sensing

**arXiv ID:** 2605.11475 | [PDF](https://arxiv.org/pdf/2605.11475v1)

**作者:** Gang Qu `[一作]` (Westlake University), Xin Yuan `[通讯]` (Westlake University)

**通讯引用:** 13987 | [OpenAlex ID](https://openalex.org/A5015431603)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种深度概率展开网络DPUNet，用闭式似然梯度投影和双域Mamba块实现量化压缩感知（QCS）的高效重建。

**💡 创新点**

创新点包括：①用闭式数值稳定的量化似然梯度投影替代传统EP或采样，直接在网络中实现数据一致性；②设计双域Mamba块，将空间状态空间建模与频域S^3M相结合，加入低秩跨频耦合，实现全局感知且计算成本低；③端到端训练同时兼顾重建精度与推理效率。

**🔧 技术方法**

采用的技术包括：深度概率展开（unfolding）框架、Mills比率与高斯CDF推导的梯度投影、残差UNet结构、双域Mamba（空间SSM + 频域S^3M）、低秩频域耦合、对数似然损失和自学习步长。

**📊 数据集**

使用的数据集为CelebA（64×64）和FFHQ（256×256），并在CSet8上做OOD实验；采样比例为12.5%，量化位数为1/2/3位。

**📈 对比分析**

与QCS‑SGM、QCS‑SGM+以及传统Vanilla DUN等方法比较。DPUNet在所有量化位数下的PSNR/SSIM均明显优于SOTA，并且推理时间仅为0.25–0.28 s，参数量和FLOPs也最低；在OOD实验中表现仍优于其他方法。

**⚠️ 局限性**

局限性包括：①对测量矩阵的假设仍有限，需在更广泛矩阵下验证收敛性；②主要验证于图像重建，缺乏对下游任务（如识别、加密传输）的评估；③在极低位量化（1位）下仍受噪声影响，未来需进一步提升鲁棒性。

---

## 16. Gradient-Free Noise Optimization for Reward Alignment in Generative Models

**arXiv ID:** 2605.11347 | [PDF](https://arxiv.org/pdf/2605.11347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 17. AESOP: Adversarial Execution-path Selection to Overload Deep Learning Pipelines

**arXiv ID:** 2605.10987 | [PDF](https://arxiv.org/pdf/2605.10987v1)

**作者:** Tingxi Li `[一作]` (University of Texas at Dallas), Wei Yang `[通讯]` (University of Texas at Dallas)

**通讯引用:** 11213 | [OpenAlex ID](https://openalex.org/A5036689637)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种针对多模型深度学习推理流水线的路径感知攻击框架 AESOP，能够通过对输入进行微小扰动将工作负载引导至计算量最大的执行路径，从而显著放大整体 FLOPs 与延迟；

**💡 创新点**

创新点在于将路径选择与单模型效率攻击分离为两阶段流程——先通过“漏洞引导路径排序”确定最易放大成本的执行路径，再通过“自适应损失加权”在该路径上生成针对性扰动，从而实现比传统单模型攻击多达 20 倍的效率攻击；

**🔧 技术方法**

技术核心包括：① 路径级脆弱性评分与排序，基于每个模块的 per‑inference cost、output cardinality 与 gating 行为；② 对每个选定路径的加权损失函数，将单模型攻击后端（如 Overload、Phantom Sponges、SlowTrack）嵌入；③ 白盒与灰盒两种攻击模式，其中灰盒通过公共模型规格实现路径排序，利用同族或跨族替代模型实现梯度迁移；

**📊 数据集**

主要使用公开基准数据集 MS COCO（用于交通监控、车辆/行人检测等）与 Animal Faces（用于野生动物监测）来评估攻击效果；

**📈 对比分析**

在五个典型流水线与一套生产级部署变体上与三种单模型效率攻击基线对比，白盒下 AESOP 达到最高 2407 倍 FLOPs 与 263 倍延迟提升，灰盒迁移后仍可实现 3.6–58.3 倍 FLOPs；相较于最强基线仅 117 倍，展示了显著性能提升；

**⚠️ 局限性**

限制在于：需知晓流水线拓扑与各模块成本信息，若采用自定义或私有模型需要额外推断；攻击依赖梯度可迁移的替代模型，无法完全跨域；仅针对推理时攻击，未考虑训练时或硬件层面的防御；目前现有系统级防御均无法有效阻止此类路由层攻击，需研发路由感知防御机制。

---

## 18. Error whitening: Why Gauss-Newton outperforms Newton

**arXiv ID:** 2605.11316 | [PDF](https://arxiv.org/pdf/2605.11316v1)

**作者:** Maricela Best McKay `[一作]` (University of British Columbia), R. Bhushan Gopaluni `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究Gauss-Newton在函数空间中的“误差白化”特性，并通过回归、物理信息神经网络及强化学习等案例证明其优于Newton、Adam与Muon方法。

**💡 创新点**

提出误差白化概念，证明GGN和G_J在函数空间投影时消除参数化扭曲，提供了Gauss-Newton优于Newton的理论解释。

**🔧 技术方法**

采用函数空间动力学分析、误差白化理论、GGN/G_J更新、矩阵投影、sketching近似以及JAX实现。

**📊 数据集**

使用多种数据集，包括回归样本、MNIST、Allen‑Cahn PDE、双积分器最短时间控制等。

**📈 对比分析**

与Newton、Adam、Muon在相同超参数与线搜索条件下对比，结果显示Gauss-Newton收敛更快、误差更低，尤其在PINN与RL任务中显著优于其它优化器。

**⚠️ 局限性**

仅分析瞬时一步动力学，无法完全解释带动量优化；sketching需较高秩以保持精度，规模受限；不适用于极大网络或高度结构化模型。

---

## 19. Quotient-Categorical Representations for Bellman-Compatible Average-Reward Distributional Reinforcement Learning

**arXiv ID:** 2605.11289 | [PDF](https://arxiv.org/pdf/2605.11289v1)

**作者:** Ege C. Kaya `[一作]` (Purdue University), Abolfazl Hashemi `[通讯]` (Purdue University)

**通讯引用:** 466 | [OpenAlex ID](https://openalex.org/A5036900440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于商空间的平均奖励分布式偏置估计框架，并给出了对应的投影算子与分类参数化；

**💡 创新点**

创新点在于通过商空间消除偏置的加性常数不唯一性，构造了非收缩的投影算子，进而实现了TD式随机近似迭代与在线增益估计；

**🔧 技术方法**

主要技术包括商空间与分类分布的构造、Cramér度量下的非扩张性分析、Krasnosel'skii–Mann迭代、随机逼近理论以及投影-中心化的奖励处理；

**📊 数据集**

实验数据集为一个五状态的离散MDP（确定性奖励）以及一个连续状态下的神经网络函数逼近实验；

**📈 对比分析**

通过与精确KM迭代、i.i.d.与马尔可夫采样的中心化SKM、以及耦合增益估计的SKM进行对比，实验结果显示残差收敛速度与理论预测一致，增益误差随迭代快速下降；

**⚠️ 局限性**

局限性包括仅在有限状态、固定策略评估和固定分类支持下证明，缺乏对更复杂函数逼近和控制问题的推广，且收敛速率为子线性，需谨慎设置步长。

---

## 20. Quantifying Rodda and Graham Gait Classification from 3D Makerless Kinematics derived from a Single-view Video in a Heterogeneous Pediatric Clinical Cohort

**arXiv ID:** 2605.11314 | [PDF](https://arxiv.org/pdf/2605.11314v1)

**作者:** Lauhitya Reddy `[一作]` (Emory University), Hyeokhyen Kwon `[通讯]` (Emory University)

**通讯引用:** 734 | [OpenAlex ID](https://openalex.org/A5036233162)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过单摄像头的视频推断儿童脑性瘫痪患者的膝关节和踝关节 z 分数，实现 Rodda‑Graham 步态分类与二元拐膝筛查。

**💡 创新点**

结合单摄像头无标记 3D 姿态估计与自适应图卷积+ViT 的深度回归模型，在异质儿科临床样本中实现了对 7 类步态模式的量化预测与连续趋势跟踪。

**🔧 技术方法**

单摄像头 3D 姿态估计 (MeTRABS)，自适应图卷积网络 (AGCN)，Vision Transformer 级联聚合，五折受试者独立交叉验证。

**📊 数据集**

152 名儿童（共 60 种诊断）在 529 次 3D 运动分析实验中记录的单摄像头视频与 3D‑IGA 标注的 z 分数。

**📈 对比分析**

与 3D‑IGA 真实 z 分数进行 R²、CCC、MAE 等回归评估，AGCN+ViT 在膝关节 R²≈0.80、CCC≈0.89，踝关节 R²≈0.57、CCC≈0.72；二元拐膝分类 AUROC≈0.88、召回≈0.83，七类分类准确率≈43%，macro‑AUROC≈0.78。

**⚠️ 局限性**

踝关节预测误差大、对极端病变表现欠佳；模型受单摄像头姿态估计对非典型步态的偏差影响；样本分布不均导致罕见类别识别不可靠；缺乏跨站点验证。

---

## 21. ReVision: Scaling Computer-Use Agents via Temporal Visual Redundancy Reduction

**arXiv ID:** 2605.11212 | [PDF](https://arxiv.org/pdf/2605.11212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 22. VNDUQE: Information-Theoretic Novelty Detection using Deep Variational Information Bottleneck

**arXiv ID:** 2605.11551 | [PDF](https://arxiv.org/pdf/2605.11551v1)

**作者:** Aryan Gondkar `[一作]` (Michigan State University), Yiming Deng `[通讯]` (Michigan State University)

**通讯引用:** 3007 | [OpenAlex ID](https://openalex.org/A5025525711)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 MNIST 数据集上，利用深度变分信息瓶颈（VIB）构建 VNDUQE 系统，实现对分布外（OOD）样本的检测与不确定性量化。

**💡 创新点**

创新点在于引入 KL 散度与预测熵两种互补的检测信号，并将它们通过并行 OR 策略组合，显著提升 OOD 检测性能与校准度；同时证明了信息瓶颈约束可降低期望校准误差。

**🔧 技术方法**

使用技术包括：Deep Variational Information Bottleneck、Gaussian 编码器/解码器、KL 散度与熵指标、最大 softmax 概率基线、信息平面分析、期望校准误差（ECE）评估。

**📊 数据集**

主要数据集为 MNIST（训练 9 类，留出 8 类做近距离 OOD），近 OOD 为留出的数字 8，远 OOD 为均匀噪声、Gaussian 噪声与 FashionMNIST；对比基线模型（β=0）与压缩模型（β=10⁻³）。

**📈 对比分析**

通过 AUROC 与 5% FPR 下的 TPR 进行比较；组合策略获得 95.3% 的平均 AUROC 与 92% 的 TPR，较基线 MSP 的 85% AUROC 与 60% TPR 提升 32 个百分点；KL 散度在噪声样本上实现 100% AUROC，熵在留出数字上达 94.7% AUROC；压缩模型 ECE 下降 38%。

**⚠️ 局限性**

局限性包括仅在全连接网络和 MNIST 上验证；仅测试单一留出类；固定 β 值且未自适应；缺乏卷积结构和更复杂数据集（如 CIFAR-10、ImageNet）的验证；并行 OR 策略为简单实现，可能存在更优组合方式。

---

## 23. 3D-Belief: Embodied Belief Inference via Generative 3D World Modeling

**arXiv ID:** 2605.11367 | [PDF](https://arxiv.org/pdf/2605.11367v1)

**作者:** Yifan Yin `[一作]`, Tianmin Shu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出3D-Belief，一种能在线更新、在部分观测下进行3D信念推理的生成世界模型，并在此基础上实现了更精准的对象导航等任务。

**💡 创新点**

将世界建模视为在3D空间中的信念推理，首次提出空间一致的场景记忆、多假设采样、顺序更新和语义驱动的未观测区域预测，并在3D空间直接生成可操作的、带不确定性的完整3D表示。

**🔧 技术方法**

基于扩散模型的3D Gaussian Splatting生成器，U‑ViT backbone结合MVS式成本体积与多视角Transformer，配合CLIP/DINO语义头；训练采用上下文‑目标图像对进行去噪学习。

**📊 数据集**

主要使用AI2‑THOR（模拟环境）和RealEstate10K（真实图像）进行训练与评估，并构建3D‑CORE基准进行对象/房间级3D推理测试。

**📈 对比分析**

与DFoT、NWM、DFoT‑VGGT、VGGT等基线进行对比。3D‑Belief在2D视觉质量（LPIPS、FVD、FID等）显著优于基线，在3D‑CORE任务（对象完成、房间完成、对象永存）取得更高的IoU、召回率、相似度，并在对象导航（仿真与真实）中实现更高的成功率(SR)、路径效率(SPL/SEL)且Token消耗更低。

**⚠️ 局限性**

假设环境静态，无法处理动态变化；对高层语言/场景图等外部指令的可控性有限；在极长时间的连续推理中可能出现误差累计。

---

## 24. A Cascaded Generative Approach for e-Commerce Recommendations

**arXiv ID:** 2605.11118 | [PDF](https://arxiv.org/pdf/2605.11118v1)

**作者:** Moein Hasani `[一作]` (Instacart), Tejaswi Tenneti `[通讯]` (Ambience Healthcare)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种级联生成式商品陈列框架，先用大语言模型生成个性化的页面主题，再生成受检索约束的关键词以驱动商品检索，并通过教师-学生蒸馏、检索增强生成和AIQA多级过滤实现可扩展部署。

**💡 创新点**

创新点在于将页面布局与内容生成拆分为两阶段的生成任务，采用教师-学生细调提升生成模型的推理效率，同时利用检索增强生成减少检索输入量，并在多级AI评估与过滤下保证质量与多样性，最终实现了与传统排序系统兼容的个性化商品展示。

**🔧 技术方法**

核心技术包括大语言模型（LLM1用于主题生成，LLM2用于关键词生成）、检索增强生成（RAG）与嵌入相似度检索、教师-学生蒸馏与LoRA适配器、约束解码、DeBERTa交叉编码器做相关性过滤、AIQA自动质量评估以及大规模缓存机制。

**📊 数据集**

实验使用了公司内部的专有生鲜电商商品目录与用户行为上下文（购买历史、参与信号、饮食偏好等），并在此基础上构建了训练与评估数据集。

**📈 对比分析**

通过离线评估（LLM-as-a-Judge、P‑T@5/20、K‑U、密度）和线上A/B测试（每页视图加购率+2.7%，每次访问+1.0%）与生产基线进行对比，发现学生模型Llama‑3.2‑3B在保持高精度的同时显著降低推理成本。

**⚠️ 局限性**

主要限制包括学生模型可能产生流行度偏差、生成的关键词导致召回密度下降、固定关键词词表限制新概念的出现、阶段间错误传播导致下游检索受损，以及对长尾商品覆盖不足时可能导致商品轮播失效。

---

## 25. Parameter Estimation of Mutual Information Maximized Channels

**arXiv ID:** 2605.11352 | [PDF](https://arxiv.org/pdf/2605.11352v1)

**作者:** Hassan Tavakoli `[一作]` (Oregon State University), Bella Bose `[通讯]` (Oregon State University)

**通讯引用:** 1092 | [OpenAlex ID](https://openalex.org/A5108277072)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了两种基于 Blahut–Arimoto 最优条件的高效算法，用于在仅观测输出的条件下，联合估计离散无记忆信道的参数 θ 和最优输入分布 π。

**💡 创新点**

创新点在于将互信息最大化约束显式纳入双层优化或增广拉格朗日框架，从而克服传统最大似然估计在该问题上易出现不可辨识的缺陷，并实现可收敛、可解释的参数恢复。

**🔧 技术方法**

使用了双层固定点梯度法、增广拉格朗日法、Blahut–Arimoto 固定点映射、隐函数微分（自动微分）以及 Fisher 信息量分析等技术。

**📊 数据集**

实验数据采用一个 10 输入、50 输出的离散高斯样本通道（生成的 i.i.d. 输出序列 T=200,000）作为仿真数据集，没有使用公开真实数据集。

**📈 对比分析**

与传统 Joint‑ML（不施加互信息约束）比较，AL 与双层法均能收敛至真参数并得到更高的似然值；AL 在 BA 求解次数和运行时间上比双层法降低约 30%，同时保持相同的估计精度。

**⚠️ 局限性**

局限性包括对 BA 固定点唯一性的假设、对梯度与逆矩阵可逆性的要求、需要手动调节惩罚因子与步长，并且目前仅在离散无记忆信道的仿真场景中验证。

---

## 26. Interpretable EEG Microstate Discovery via Variational Deep Embedding: A Systematic Architecture Search with Multi-Quadrant Evaluation

**arXiv ID:** 2605.10947 | [PDF](https://arxiv.org/pdf/2605.10947v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 27. Backbone-Equated Diffusion OOD via Sparse Internal Snapshots

**arXiv ID:** 2605.11014 | [PDF](https://arxiv.org/pdf/2605.11014v1)

**作者:** Yadang Alexis Rouzoumka `[一作]` (Université Paris-Saclay), Chengfang Ren `[通讯]` (Université Paris-Saclay)

**通讯引用:** 263 | [OpenAlex ID](https://openalex.org/A5029566720)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在冻结的扩散模型上，提出了最小化内部特征探测方法 Canonical Feature Snapshots（CFS）；

**💡 创新点**

创新点在于引入 Mutualized Backbone‑Equated（MBE）协议统一不同后端的评估，证明仅采样少量稀疏内部快照即可实现高效 OOD 检测；

**🔧 技术方法**

采用对齐的低噪声层级激活、对角线统计的 ID‑only 评分，以及轻量级的内部特征池化；

**📊 数据集**

主要使用 CIFAR‑10、SVHN、CelebA32 作为 ID 组，CIFAR‑100、DTD 作为 OOD 组，同时在 ImageNet 上进行迁移实验；

**📈 对比分析**

在 MBE 协议下与多种基线（多尺度输出、递归路径、重建等）对比，1×2 方案仅需一次前向推理即可获得 0.81‑0.92 的 AUROC，显著优于传统方法；

**⚠️ 局限性**

局限性包括需要模型内部特征访问（非黑盒），对源模型与架构敏感，以及在多模态 ID 场景下性能仍有提升空间。

---

## 28. Optimal Codes with Positive Griesmer Defects, Related Optimal and Almost Optimal LRC Codes

**arXiv ID:** 2605.11431 | [PDF](https://arxiv.org/pdf/2605.11431v1)

**作者:** Yurui Wang `[一作]` (Southeast University), Xia Wu `[通讯]` (Southeast University)

**通讯引用:** 4632 | [OpenAlex ID](https://openalex.org/A5100740259)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一系列具有正Griesmer缺陷的最优线性码，并证明其为局部可恢复码(LRC)，局部性为2，且部分码满足Cadambe‑Mazumdar（CM）上界或仅偏差1。

**💡 创新点**

创新点在于：①提出新的构造方法，得到非Solomon‑Stiffler类的正缺陷最优码；②推导了这些码的权分布和子码支持权分布；③证明其在局部可恢复码框架下可达到或逼近CM上界。

**🔧 技术方法**

采用了有限域向量空间、子空间划分与线性代数工具，利用Griesmer界、Generalized Hamming Weights、子码支持权分布计数以及LRC的本地恢复性条件进行构造与证明。

**📊 数据集**

未使用外部实验数据集；所有示例均由符号计算软件Magma验证。

**📈 对比分析**

与已知的Solomon‑Stiffler、Belov码以及已公布的最优LRC码对比，本文构造的码在相同参数下均达到或仅偏差1于CM上界，证明了构造的有效性与优越性。

**⚠️ 局限性**

限制主要在于：①局部性仅为2；②构造参数范围受q、k、u_i等整数限制；③对更高局部性的最优码或更大正缺陷的进一步推广仍未完成。

---

## 29. Hierarchical LLM-Driven Control for HAPS-Assisted UAV Networks: Joint Optimization of Flight and Connectivity

**arXiv ID:** 2605.11509 | [PDF](https://arxiv.org/pdf/2605.11509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 30. Options, Not Clicks: Lattice Refinement for Consent-Driven MCP Authorization

**arXiv ID:** 2605.11360 | [PDF](https://arxiv.org/pdf/2605.11360v1)

**作者:** Ying Li `[一作]` (University of California, Los Angeles), Yuan Tian `[通讯]` (University of California, Los Angeles)

**通讯引用:** 6117 | [OpenAlex ID](https://openalex.org/A5100716458)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于边界约束的MCP（Model Context Protocol）授权中间件——Consent Leash，用于在代理调用外部工具时自动化、细粒度地进行用户同意管理。

**💡 创新点**

创新点在于：① 引入风险格（risk lattice）将工具调用抽象为四维边界（输入/输出位置、数据敏感度、操作效果），实现可判定的包含检查；② 设计了声明式策略引擎和基于 Datalog 的决策过程，可把自然语言约束编译为不可覆盖的 DENY 规则；③ 开发了边界细化循环，将每次用户同意泛化为可复用的、最小化提示的权限规则，从而在保持安全性的同时显著降低同意疲劳。

**🔧 技术方法**

使用的技术包括：Python + Datalog（Soufflé）实现权限检查和 taint 传播；LLM（大型语言模型）负责工具调用和自然语言约束的结构化抽象与规则生成；风险格和部分顺序用于判定包含关系；边界细化策略通过优化目标近似实现权限泛化。

**📊 数据集**

使用的数据集为 ConsentBench（984 条真实 MCP 服务器轨迹）以及 50 个真实代理会话（1,435 次工具调用），并通过 16 名受试者的使用体验实验验证。

**📈 对比分析**

在 ConsentBench 上达到 98.2% 的步骤准确率（F1 98.7%），自动批准 98.3% 的正常调用，捕获 99.4% 的权限升级；每步推理开销仅 8.2 ms。与传统工具级同意（Allow/Always Allow）和 LLM 自动同意相比，Consent Leash 在安全性上更高、提示次数更少（用户在研究中 3.5 倍更倾向于使用边界级“Always Allow”），且用户对可控性和信息披露的满意度显著更高。

**⚠️ 局限性**

局限性包括：① 对 LLM 的抽象和规则生成依赖度高，若模型误解可能导致权限细化不准确；② DSL 目前缺乏时间约束和更细粒度的领域特定表达，可能无法覆盖所有安全需求；③ 仅在 MCP 的客户端/主机层面实现，若中间件或 OS 被破坏，安全性不再完整；④ 保障仅对抽象层面的安全，无法保证对所有潜在语义错误的绝对防护。

---

## 31. DCVD: Dual-Channel Cross-Modal Fusion for Joint Vulnerability Detection and Localization

**arXiv ID:** 2605.11015 | [PDF](https://arxiv.org/pdf/2605.11015v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 32. Principled Design of Diffusion-based Optimizers for Inverse Problems

**arXiv ID:** 2605.11506 | [PDF](https://arxiv.org/pdf/2605.11506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 33. SOMA: Efficient Multi-turn LLM Serving via Small Language Model

**arXiv ID:** 2605.11317 | [PDF](https://arxiv.org/pdf/2605.11317v1)

**作者:** Xueqi Cheng `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 1001 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SOMA框架，利用会话前几轮的上下文构建局部响应流形，先用软提示挖掘大模型与小模型的对齐差异，再将其转化为LoRA微调，实现小模型在后续轮次的无提示推理，并通过语义门控与漂移检测实现一次性切换与回滚，达到高效多轮对话服务；

**💡 创新点**

①发现多轮对话中前几轮词数显著更长，后续轮次呈长尾分布；②利用软提示主动挖掘局部对齐不足方向；③将软提示信息蒸馏为LoRA微调，消除对提示的依赖；④设计语义门控与漂移检测，保证切换时的上下文一致性与快速回滚；

**🔧 技术方法**

软提示调优、语义不相似损失、期望加权语义偏差、反退化正则化、LoRA局部微调、语义门控+漂移检测、理论分析（局部流形、采样覆盖等）；

**📊 数据集**

ShareGPT、ReMeDi、Craigslist Bargain、Multi-Character、MATH、MT-Bench；

**📈 对比分析**

与原大模型、单一小模型、历史前缀、历史微调、LLMLingua-2、RouteLLM等基线对比，SOMA在六个数据集上实现了最高的响应相似度、显著提升任务精度（如MATH），并在中长会话中显著降低token使用和延迟；

**⚠️ 局限性**

对极短会话收益有限；依赖早期轮次建立的局部状态，主题突变时需要回滚；软提示搜索和LoRA微调产生额外成本；仅适用于大模型不可直接访问或成本高的部署场景。

---

## 34. Adversarial SQL Injection Generation with LLM-Based Architectures

**arXiv ID:** 2605.11188 | [PDF](https://arxiv.org/pdf/2605.11188v1)

**作者:** Ali Karakoc `[一作]` (Bogazici University), H. Birkan Yilmaz `[通讯]` (Bogazici University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对 Web 应用的 SQL 注入攻击，构建并评估了两种基于大型语言模型的新型生成框架 RADAGAS（RAG+MMR+多阶段过滤）与 RefleXQLi（CoT+双 LLM 迭代）并与传统规则、零射击、GenSQLi 等基线进行对比；

**💡 创新点**

创新点包括①将检索增强生成与最大边际相关性（MMR）相结合，提升语义相关又多样化的 SQLi 生成；②设计了双 LLM 对抗循环的 CoT 推理架构，既实现高度唯一性又保持可执行性；③在十种规则型、AI/ML 以及商用 WAF 上做系统性基准实验；

**🔧 技术方法**

主要技术包括：大型语言模型（GPT‑4o、Claude‑Sonnet、DeepSeek‑R1）、检索增强生成（RAG+FAISS）、最大边际相关性（MMR）、语义/结构多样性过滤（BERTScore、Levenshtein、AST 距离）、双 LLM 对抗迭代、Chain‑of‑Thought 逐步推理、温度与阈值调优；

**📊 数据集**

使用的数据集为：① 82 KB 由 OWASP、PortSwigger、GitHub（PayloadsAllTheThings）等聚合的 SQL 注入知识库；② MySQL 8.0 的自建漏洞应用做执行验证；③ 生成的 240,000 条 SQLi 负载；

**📈 对比分析**

通过 240 次实验共 2.2 M 次 WAF/执行测试，RADAGAS‑GPT‑4o 取得 22.73% 的平均绕过率，DeepSeek 22.09%，Claude 21.73%；RefleXQLi 21.21%；最高商用 WAF Cloudflare 被 RADAGAS‑GPT‑4o 绕过 49.5%，而带种子版本的 RefleXQLi V2 则在所有 WAF 上提升至 35.78%；

**⚠️ 局限性**

主要局限：仅研究 SQL 注入且仅以 MySQL 8.0 为验证数据库；WAF 版本与部署场景有限；跨系统相关性统计样本量小；LLM 与 WAF 的动态演进可能导致结果时效性受限。

---

## 35. SHIA: A Direct SysML-Hardware Interface Architecture for Model-Centric Verification

**arXiv ID:** 2605.11248 | [PDF](https://arxiv.org/pdf/2605.11248v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 36. Dynamic Execution Commitment of Vision-Language-Action Models

**arXiv ID:** 2605.11567 | [PDF](https://arxiv.org/pdf/2605.11567v1)

**作者:** Feng Chen `[一作]` (University of Adelaide), Yicheng Wu `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对视觉-语言-动作模型中动作chunking的执行窗口选择问题，提出A^3机制通过自我投机前缀验证自适应决定执行长度。

**💡 创新点**

创新点在于把执行窗口视为自我一致性与序列可行性验证的前缀问题，利用轨迹级共识评分与双层层次验证（共识-有序条件不变性和前缀闭合序列一致性），实现无需手工调参的自适应执行。

**🔧 技术方法**

技术主要包括基于多重采样的轨迹共识计算、聚类选取主草稿、双层并行验证树以及阈值匹配的自我投机推理。

**📊 数据集**

在LIBERO、MetaWorld、MainSkill等仿真基准以及四个真实机器人任务（FlipMug、TapeBox、HangMug、StackCube）上进行评估。

**📈 对比分析**

与固定窗口、MoH、EverydayVLA、AutoHorizon等基线对比，A^3在保持或提升任务成功率的同时，显著延长执行长度，降低前向调用次数，逼近Pareto最优平衡。

**⚠️ 局限性**

局限在于无法补偿感知缺失或极端精度需求的失败，且对模型自身的可靠性假设较高，重采样与验证仍带来一定推理开销。

---

## 37. Constraint-Data-Value-Maximization: Utilizing Data Attribution for Effective Data Pruning in Low-Data Environments

**arXiv ID:** 2605.11312 | [PDF](https://arxiv.org/pdf/2605.11312v1)

**作者:** Danilo Brajovic `[一作]` (Fraunhofer IPA), Marco F. Huber `[通讯]` (Fraunhofer IPA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于数据归因矩阵的优化框架 CDVM，用于在低数据场景下对训练样本进行低价值裁剪

**💡 创新点**

将数据归因矩阵与约束优化相结合，既最大化每个测试样本的总影响，又通过阈值惩罚过度集中，解决传统 Shapley 等方法在大簇数据中早期裁剪导致的失衡问题

**🔧 技术方法**

数据归因（如 TRAK / 影响函数）、混合整数/线性规划、最大样本重用 (MSR) 采样估计归因矩阵

**📊 数据集**

OpenDataVal 公开基准中的六个数据集（包含图像与文本）

**📈 对比分析**

与随机删减、DataOob/记忆化、DataBanzhaf、影响优化等方法对比，CDVM 在 28/36 评估实例中达到或超过 state‑of‑the‑art，且在速度–准确度上表现最佳

**⚠️ 局限性**

归因矩阵的计算与存储成本高，规模扩展受限；缺乏对高阶交互的建模，导致在某些数据集上初期性能不佳

---

## 38. Hierarchical Multi-Scale Graph Neural Networks: Scalable Heterophilous Learning with Oversmoothing and Oversquashing Mitigation

**arXiv ID:** 2605.10975 | [PDF](https://arxiv.org/pdf/2605.10975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 39. Benchmarking LLM-Based Static Analysis for Secure Smart Contract Development: Reliability, Limitations, and Potential Hybrid Solutions

**arXiv ID:** 2605.11163 | [PDF](https://arxiv.org/pdf/2605.11163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 40. When to Ask a Question: Understanding Communication Strategies in Generative AI Tools

**arXiv ID:** 2605.11240 | [PDF](https://arxiv.org/pdf/2605.11240v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 41. The Authorization-Execution Gap Is a Major Safety and Security Problem in Open-World Agents

**arXiv ID:** 2605.11003 | [PDF](https://arxiv.org/pdf/2605.11003v1)

**作者:** Baoyuan Wu `[一作]` (Chinese University of Hong Kong), Siwei Lyu `[通讯]` (State University of New York at Buffalo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出授权-执行差距（AEG）框架，分析其三类结构来源并提出执行时的授权完整性检查。

**💡 创新点**

创新性在于将agent失败归因到具体结构来源，强调基于源的诊断与防御，而非仅靠表面错误分类。

**🔧 技术方法**

使用授权-执行路径模型、案例分析和安全性检查技术，对三类来源进行抽象与验证。

**📊 数据集**

利用已有的agent攻击与失败案例、benchmark文献作为实证材料，未引入新的数据集。

**📈 对比分析**

主要以案例对比说明框架效果，未进行量化实验或性能评测，强调概念性验证。

**⚠️ 局限性**

局限在于缺乏完整的防御体系与量化评估，且仅聚焦授权完整性，未考虑其他安全风险。

---

## 42. PASA: A Principled Embedding-Space Watermarking Approach for LLM-Generated Text under Semantic-Invariant Attacks

**arXiv ID:** 2605.10977 | [PDF](https://arxiv.org/pdf/2605.10977v1)

**作者:** Zhenxin Ai `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Haiyun He `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在潜在语义空间进行水印嵌入和检测的算法（PASA），能够在文本经过语义不变攻击（如同义替换、改写、转述）后仍保持高检测准确率，同时保持生成文本与原始分布一致，无质量损失。

**💡 创新点**

创新点包括：①将共享随机性与语义聚类对齐，使水印在语义层面稳固；②基于信息理论的最优嵌入-检测对，给出最小误检率与假警率的理论下界；③设计两阶段采样策略（先抽取辅助随机变量，再在对应语义簇内重采样），实现零失真生成；④通过可逆的语义映射与秘密键同步，确保检测端能恢复辅助序列。

**🔧 技术方法**

主要技术：预训练语言模型生成文本的 NTP 采样；使用预训练句子/词嵌入（如BERT、CLIP等）构建词向量；K‑means 对词向量进行聚类得到语义簇；伪随机函数（PRF）与秘密键同步生成辅助序列；两阶段采样与阈值裁剪实现假警率控制；检测端使用轻量级代理语言模型（SLM）估计 NTP，重构辅助分布并累积分数。

**📊 数据集**

使用 C4 语料库（多语言文本）进行训练与评估；在长篇问答数据集上进行泛化测试；攻击评估采用 T5‑Large/T5‑XXL 进行词替换与 DIPPER 进行改写；对比 Llama‑2‑13B、Mixtral‑8×7B 等模型。

**📈 对比分析**

与 KGW、Exp‑Edit、AWTI、DAWA 等基线对比。PASA 在清洁文本上达到 99.9%+ 的 AUC‑ROC，TPR@1%FPR 近 100%；在 T5‑Large 替换攻击下 TPR@1%FPR 为 92.96%（远高于 KGW 73.5%），在强改写（DIPPER Ord=80）下仍保持 TPR@1%FPR 约 58.8%，显著优于基线；文本质量指标 PPL 与未水印模型相近（≈11.4 vs 12.4），生成与检测延迟仅略有提升。

**⚠️ 局限性**

局限性：1）对极端重写或水印移除攻击效果有限；2）对短文本的检测可信度低，需结合句子/段落级别统计；3）检测端需使用与生成模型兼容的 tokenizer，tokenizer 不匹配会降低跨模型迁移性能。

---

## 43. Predicting Psychological Well-Being from Spontaneous Speech using LLMs

**arXiv ID:** 2605.11303 | [PDF](https://arxiv.org/pdf/2605.11303v1)

**作者:** Erfan Loweimi `[一作]` (Cisco), Saturnino Luz `[通讯]` (University of Edinburgh)

**通讯引用:** 3547 | [OpenAlex ID](https://openalex.org/A5060851208)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用零样本LLM预测自发语音中的Ryff心理福祉得分，并给出可解释的关键词和证据。

**💡 创新点**

提出了基于临床心理学和语言学专家共同设计的领域导向提示，首次展示LLM能在无微调条件下实现多维心理福祉预测，并提供可解释性分析。

**🔧 技术方法**

使用了12种指令微调LLM（如Meta‑Llama‑3.3‑70B、DeepSeek‑Qwen、Gemma‑3‑27B等）与结构化JSON输出，结合手工转录的语音文本。

**📊 数据集**

使用了来自PsyVoiD数据库的111份自发语音样本（约1–2分钟，平均150词），并与Ryff自评问卷得分作为黄金标准。

**📈 对比分析**

通过Pearson和Spearman相关系数评估性能，Meta‑Llama‑3.3‑70B在完整数据集上实现最高Spearman 0.408（p<0.01），在信息丰富的前75%样本上提升至0.8；其他模型表现低于0.5，且多数存在显著性不足。

**⚠️ 局限性**

局限包括：样本量仅111人、语音转录为手工导致缺乏自动化可扩展性、模型对低词量/信息稀缺样本表现差、存在系统性低估偏差，以及缺乏多模态声学特征融合。

---

## 44. Don't Look at the Numbers: Visual Anchoring Bias and Layer-wise Representation in VLMs

**arXiv ID:** 2605.11218 | [PDF](https://arxiv.org/pdf/2605.11218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 45. The Entropy of Floating-Point Numbers

**arXiv ID:** 2605.11546 | [PDF](https://arxiv.org/pdf/2605.11546v1)

**作者:** Sultan Daniels `[一作]` (University of California, Berkeley), Anant Sahai `[通讯]` (University of California, Berkeley)

**通讯引用:** 9955 | [OpenAlex ID](https://openalex.org/A5079317491)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文给出了浮点数量化后随机变量熵的解析近似，并提供了该近似误差的上界；同时证明了该熵在尺度变换下大致不变，并给出了多种常见分布的闭式熵表达式。

**💡 创新点**

创新点主要在于：① 将差分熵与非均匀（浮点）量化离散熵的对应关系进行推广；② 通过平滑与扩展 bin 大小函数得到可计算的近似公式；③ 证明了浮点量化熵的尺度不变性；④ 给出多种分布的闭式近似，并与数值计算结果做对比，验证误差可控。

**🔧 技术方法**

主要技术包括：非均匀量化分析、KL 散度上界推导、平滑 bin 大小函数、卷积（folded）分布的对数矩计算、以及多维扩展和误差评估。

**📊 数据集**

文中未使用具体实验数据集，而是基于理论分析对多种连续分布（正态、均匀、伽玛、卡方、拉普拉斯、Logistic、Weibull、对数正态、帕累托、Beta、t 分布以及多元正态）进行熵近似推导。

**📈 对比分析**

作者将解析近似值与数值求解的精确离散熵进行对比，利用图表展示误差随尺度、指数位数和精度变化的趋势。结果表明，在合理的指数范围内，近似值与精确值相差不超过 0.5 位（加上溢出/欠流误差），并且尺度变换下误差保持不变，证明了近似的稳健性。

**⚠️ 局限性**

局限性包括：① 近似误差受到高斯峰度和 bin 宽度的影响，在极端峰值或小 bin 处误差可能增大；② 需要较宽的指数位数以避免溢出/欠流；③ 仅考虑无尾数（subnormal）格式的简化浮点；④ 对硬件实现细节（如舍入规则、特殊值处理）未作深入讨论。

---

## 46. Coordinated Diffusion: Generating Multi-Agent Behavior Without Multi-Agent Demonstrations

**arXiv ID:** 2605.11485 | [PDF](https://arxiv.org/pdf/2605.11485v1)

**作者:** Lasse Peters `[一作]` (University of California, Berkeley), Andrea Bajcsy `[通讯]` (Carnegie Mellon University)

**通讯引用:** 569 | [OpenAlex ID](https://openalex.org/A5050279893)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 Codi，一种仅利用单机演示数据即可生成多机协同行为的方法。

**💡 创新点**

创新点在于：1) 将预训练的单机扩散策略通过一个多机成本函数以乘积形式耦合；2) 引入梯度无关的引导评分估计，消除了对可微成本或额外训练的需求；3) 理论分析说明单机支持与成本补偿的关系，阐明何时能逼近目标多机行为。

**🔧 技术方法**

技术核心包括：连续时间扩散模型、基于扩散的策略学习、乘积分布下的 KL 约束游戏、分类器引导思路的改进、基于采样的梯度无关引导评分估计。

**📊 数据集**

使用在 Isaac Gym 中仿真得到的两臂 Franka 机器人（7-DoF）采集的 1,000 条单机演示（抓取与放置）。硬件实验亦在同类机器人上验证。

**📈 对比分析**

与使用多机演示的基线（CG-Joint、DPMD-Joint、SDAC-Joint、EXPO-Joint）进行比较。Codi 在任务完成率、目标距离和时间上均优于所有基线，并在 20% 数据量下仍保持接近完整数据版本的性能，显示出更高的数据效率。

**⚠️ 局限性**

局限性包括：1) 需要在测试时进行集中执行，无法直接实现去中心化策略；2) 成本函数设计对最终协作质量关键，若设计不当会导致失败；3) 对单机演示覆盖范围和支持的依赖，若单机支持不足则需额外成本补偿；4) 目前仅适用于完全协同的合作场景，无法直接处理非合作或竞争情形。

---

## 47. Principle-Guided Supervision for Interpretable Uncertainty in Medical Image Segmentation

**arXiv ID:** 2605.10984 | [PDF](https://arxiv.org/pdf/2605.10984v1)

**作者:** An Sui `[一作]` (Fudan University), Xiahai Zhuang `[通讯]` (Fudan University)

**通讯引用:** 6527 | [OpenAlex ID](https://openalex.org/A5011662977)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于感知原则的不确定性监督框架 PriUS，旨在使医学影像分割模型的空间不确定性分布与图像对比度、噪声程度和几何复杂度保持一致，从而提升不确定性的可解释性。

**💡 创新点**

创新点在于将三条直观的感知原则（对比度原则、腐蚀原则、几何原则）量化为可直接监督的损失，并将其与证据学习耦合，首次实现对不确定性空间行为的结构化约束。

**🔧 技术方法**

采用证据深度学习（EDL）作为基准不确定性预测器，配合基于图像梯度、噪声水平和边界距离的三项监督损失，实现统一的端到端训练。

**📊 数据集**

在三大公开医学影像分割基准上进行评估：ACDC（心脏 MRI）、ISIC（皮肤病变 RGB）和 WHS（心脏 CT），覆盖单通道、二分类和多分类任务。

**📈 对比分析**

与 DEviS、PU、EU、UDrop、TTA 等现有方法比较，PriUS 在 UCC/UR 指标上均保持期望符号并取得最高或相近的数值，同时在 Dice/HD95 上保持竞争甚至领先，证明在不损失分割精度的前提下提升了不确定性可解释性。

**⚠️ 局限性**

局限性包括对阈值 d0 等超参数的依赖，需要人工调优；此外仅在单一不确定性估计器（EDL）上验证，跨方法或更复杂噪声/模态的泛化性尚待进一步研究。

---

## 48. Efficient Adjoint Matching for Fine-tuning Diffusion Models

**arXiv ID:** 2605.11480 | [PDF](https://arxiv.org/pdf/2605.11480v1)

**作者:** Jeongwoo Shin `[一作]` (Seoul National University), Jaemoo Choi `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种高效的奖励梯度微调方法EAM，用以对预训练的扩散模型进行人类偏好对齐；

**💡 创新点**

创新点在于通过重构基漂移为线性且无记忆的形式，消除了Adjoint Matching（AM）中的两大计算瓶颈——前向随机轨迹模拟和后向adjoint ODE求解，并给出闭式adjoint解和端点条件噪声采样；

**🔧 技术方法**

主要技术包括：基于线性基漂移的SOC重构、终端成本重设计、闭式adjoint求解、利用Ode求解器进行端点采样、Tweedie公式估计预训练终端梯度、LoRA控制参数化；

**📊 数据集**

在Stable Diffusion 3.5‑Medium上使用Pick‑a‑Pic提示集、PickScore、HPSv2.1和Aesthetics奖励模型，评估数据集为DrawBench；

**📈 对比分析**

与原AM相比，EAM在相同奖励尺度下实现了约4倍的训练速度提升，且在PickScore、ImageReward、HPSv2.1、CLIPScore和Aesthetics等指标上匹配或优于AM；

**⚠️ 局限性**

局限性包括：对奖励尺度、常数C等超参数的敏感性，未探索更高级的优化策略（如经验回放、策略缓冲等），以及对不同模型或奖励的泛化性待进一步验证。

---

## 49. From Code-Centric to Intent-Centric Software Engineering: A Reflexive Thematic Analysis of Generative AI, Agentic Systems, and Engineering Accountability

**arXiv ID:** 2605.11027 | [PDF](https://arxiv.org/pdf/2605.11027v1)

**作者:** Elyson De La Cruz `[一作]` (University of the Cumberlands), Elyson De La Cruz `[通讯]` (University of the Cumberlands)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5106689764)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对公共话语、技术论文和案例材料进行反射性主题分析，研究了生成式 AI 与代理系统如何将软件工程从代码生产转向以意图为中心的工作；

**💡 创新点**

提出了意图驱动软件工程的核心二分法（加速 vs 责任）并构建了五阶段的成熟路径，揭示了 GenAI 降低代码成本但提升验证、架构、治理和安全需求的现实；

**🔧 技术方法**

采用反射性主题分析（RTA）结合解释性现象学分析（IPA）对多源文本进行编码和主题构建；

**📊 数据集**

使用了一个三层公共语料库：同行评议论文、技术预印本/基准以及公开演讲、访谈、X 贴、产品公告等公开材料；

**📈 对比分析**

未进行传统实验比较，而是通过与同行评议文献交叉验证主题，阐明 GenAI 在意图表达、代理工作流程、验证与治理等维度的影响；

**⚠️ 局限性**

局限包括：公共话语可能带有宣传与策略性偏差、对普通工程师实践的覆盖不足、成熟路径为理论模型非精准预测、以及 GenAI 能力快速演进导致结论快速过时。

---

## 50. A Boundary-Aware Non-parametric Granular-Ball Classifier Based on Minimum Description Length

**arXiv ID:** 2605.11406 | [PDF](https://arxiv.org/pdf/2605.11406v1)

**作者:** Zeqiang Xian `[一作]` (Gannan Normal University), Witold Pedrycz `[通讯]` (University of Alberta)

**通讯引用:** 91568 | [OpenAlex ID](https://openalex.org/A5003799782)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于最小描述长度的边界感知非参数粒子球分类器(MDL-GBC)。

**💡 创新点**

创新点在于将类内正样本与类间负样本的边界信息统一入MDL描述长度准则下的局部模型竞争，实现无手工阈值的自适应粒子球构造与解释。

**🔧 技术方法**

采用最小描述长度(MDL)、距离投影分裂、核心-边界分解以及高斯编码等技术。

**📊 数据集**

在18个UCI基准数据集（含小型到大型、平衡与不平衡）上进行实验。

**📈 对比分析**

与XGBoost、KNN、CART、SVM以及GBTSVM、ScOrGBC、SGBSkNN等方法比较，MDL-GBC在平均Accuracy与Macro-F1上均获得最佳排名。

**⚠️ 局限性**

主要局限在于计算开销大、对大规模数据的最近邻搜索成本高、仅使用欧氏距离与对角高斯模型。

---

## 51. A Multi-Interface Firmware Acquisition and Validation Methodology for Low-Cost Consumer Drones: A Case Study on Three Holy Stone Platforms

**arXiv ID:** 2605.11040 | [PDF](https://arxiv.org/pdf/2605.11040v1)

**作者:** Sandesh More `[一作]` (Florida Institute of Technology), Marco Carvalho `[通讯]` (Florida Institute of Technology)

**通讯引用:** 1687 | [OpenAlex ID](https://openalex.org/A5016664598)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并验证了一种多接口固件提取与验证框架，针对消费级四旋翼无人机（Holy Stone HS175D、HS720、HS360S）获取高质量固件镜像。

**💡 创新点**

创新点在于：①将SPI闪存、SWD/JTAG调试口、UART日志以及无拆焊的接触式提取方法结合起来，实现了多路复核；②引入三层验证体系（尺寸、哈希一致性、熵+结构签名），提升提取可靠性；③为后续固件重托管、模糊测试和安全评估提供可用的、可验证的固件语料库。

**🔧 技术方法**

使用了低成本硬件工具（CH341A SPI编程器、ST-Link调试器、USB‑UART转换器）、滑窗Shannon熵分析、Binwalk结构签名识别、EMBA静态分析框架。

**📊 数据集**

构建了基于三款Holy Stone无人机的固件数据集（HS175D 16 MiB、HS720 8 MiB，HS360S 8 MiB），每个设备多次提取并存档，哈希一致性可验证。

**📈 对比分析**

通过对比不同接触式工具（全鳄鱼夹 vs 钩形夹）和不同接口的成功率，发现钩形夹在HS175D/HS720上成功率约为75%，鳄鱼夹约为50%；熵与结构签名能够区分完整固件与空白闪存，验证结果显示HS175D、HS720通过所有三层验证，HS360S仅通过尺寸校验。

**⚠️ 局限性**

局限性包括：仅覆盖三款同一厂商的入门级无人机；接触式提取受焊盘位置、粘附力等物理因素影响；熵与签名方法对加密或高度混淆的固件识别不敏感；实验样本量有限，统计可靠性不足。

---

## 52. Efficient LLM-based Advertising via Model Compression and Parallel Verification

**arXiv ID:** 2605.11582 | [PDF](https://arxiv.org/pdf/2605.11582v1)

**作者:** Wenxin Dong `[一作]` (Baidu Inc.), Lin Liu `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套结合层适应组量化、层级稀疏和前缀树并行验证的LLM压缩与推理加速框架，并在百度广告平台实现部署。

**💡 创新点**

创新地将指数压缩的2bit-CSR索引、INT4自适应组量化、层级稀疏以及基于前缀树的动态并行验证集成在一起，显著提升推理速度。

**🔧 技术方法**

采用了自适应组量化、层级稀疏、SparseGemv自定义稀疏矩阵乘核、前缀树结构和动态并行验证技术。

**📊 数据集**

使用内部商业流量数据评估广告投放场景，使用CSL（约39.6万篇中文学术文章）评估创意生成场景。

**📈 对比分析**

与FP16基准和单独量化/稀疏方案对比，在推理速度上提升78%+，在广告召回率和创意生成的BLEU/Meteor分数保持可接受范围。

**⚠️ 局限性**

方案主要针对商业广告场景设计，可能不适用于其他领域；稀疏率提升过高会显著损失生成质量。

---

## 53. SEVO: Semantic-Enhanced Virtual Observation for Robust VLA Manipulation via Active Illumination and Data-Centric Collection

**arXiv ID:** 2605.11114 | [PDF](https://arxiv.org/pdf/2605.11114v1)

**作者:** Tianchonghui Fang `[一作]` (University of Connecticut), Fei Miao `[通讯]` (University of Connecticut)

**通讯引用:** 2469 | [OpenAlex ID](https://openalex.org/A5004660487)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对低成本机器人在透明水瓶拾取放置任务中跨环境性能不足，提出SEVO观测管线——通过身体固定摄像头、主动红光照明和实时YOLO分割叠加，并采用多样化数据采集协议，提升了现有ACT与SmolVLA策略的鲁棒性。

**💡 创新点**

创新点在于将观测空间设计与数据收集的多样性相结合，提出不改动模型架构的三种硬件+软件处理方式，并证明背景多样化是实现跨域泛化的关键因素。

**🔧 技术方法**

采用RGB身体固定摄像头、5W红光LED、YOLOv8语义分割实时叠加、真实机器人远程操作采集、ACT与SmolVLA两种VLA/Imitation Learning模型，并进行参数级别分析。

**📊 数据集**

使用真实机器人在单一实验室环境下收集的80/120条远程操控演示数据，期间系统性变换背景、灯光、干扰物与人类活动，未使用任何合成或模拟数据集。

**📈 对比分析**

通过在两台不同平台（Jetson Orin NX与Raspberry Pi 5）上，比较“完整SEVO”与“无SEVO”以及各组件剔除后的性能；完整SEVO在训练环境达到95%/83%成功率，转移至相似新环境保持85%/75%；无SEVO在新环境仅30–35%；并完成手腕摄像头、组件重要性排序与跨平台验证。

**⚠️ 局限性**

局限包括：需固定身体摄像头、仅支持静态抓取（移动抓取成功率低），在极端环境（如白色地板）仍有显著下降，依赖YOLO检测质量，SmolVLA在低算力平台上推理延迟高，以及对冻结编码器的学习容量有限。

---

## 54. Portable Agent Memory: A Protocol for Cryptographically-Verified Memory Transfer Across Heterogeneous AI Agents

**arXiv ID:** 2605.11032 | [PDF](https://arxiv.org/pdf/2605.11032v1)

**作者:** Santhosh Kumar Ravindran `[一作]` `[通讯]` (Microsoft Corporation), Santhosh Kumar Ravindran (Microsoft Corporation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了可携带、可验证的代理记忆协议 Portable Agent Memory，支持跨 LLM 平台的记忆序列化、传输和重构。

**💡 创新点**

创新点在于引入五元记忆模型、Merkle‑DAG 证明链、基于能力的细粒度访问控制以及抗注入的重构流程。

**🔧 技术方法**

主要技术包括 JSON/CBOR 序列化、BLAKE3 哈希与 Ed25519 签名、Merkle‑DAG 证明结构、能力令牌、结构化框架与压缩。

**📊 数据集**

实验使用 Claude、GPT‑4 与 Gemini 三大模型的 50 个手工设计任务（编码、问答、规划）进行评估。

**📈 对比分析**

通过 Transfer Continuity Score 与 RHF 对比基线无记忆情况，TCS 在 0.83–0.92 之间，平均提升约 2.4 倍；重构时间低于 13 ms，存储压缩率达 69%。

**⚠️ 局限性**

局限性包括相关性评分模型过于简单、压缩采用提取式摘要、实验规模有限且对嵌入模型的依赖。

---

## 55. HiDream-O1-Image: A Natively Unified Image Generative Foundation Model with Pixel-level Unified Transformer

**arXiv ID:** 2605.11061 | [PDF](https://arxiv.org/pdf/2605.11061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 56. HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series

**arXiv ID:** 2605.11130 | [PDF](https://arxiv.org/pdf/2605.11130v1)

**作者:** Jonas Petersen `[一作]` (ETH Zurich), Philipp Petersen `[通讯]` (University of Vienna)

**通讯引用:** 815 | [OpenAlex ID](https://openalex.org/A5041074956)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用自监督 Joint-Embedding Predictive Architecture（JEPA）预训练 causal Transformer 编码器，并在下游仅微调 predictor 生成可解释的生存 CDF，从而实现多变量时间序列事件预测；

**💡 创新点**

创新点在于：① 通过 horizon‑conditioned 预测未来表示而非数值，迫使编码器捕获可预测的时间动力学；② 将 predictor 作为“桥梁”在下游仅微调，保持编码器冻结，显著降低参数量且保持表达力；③ 架构通用，可在 14 个 benchmark、11 个领域中统一应用；

**🔧 技术方法**

技术细节包括 causal Transformer 编码器（d=256, 2层）、2 层 MLP predictor、SIGReg 正则化避免表示崩塌、离散时间生存 CDF（λ→p(t,Δt)）、正向权重 BCE 训练、h‑AUROC 统一评估；

**📊 数据集**

使用 14 个公开 benchmark：C‑MAPSS（1‑4）、SMAP、PSM、MBA、BATADAL、TEP、ETTm1、Weather、Beijing‑AQ、VIX、GECCO、5ch，覆盖工业、医疗、金融、气象等 11 个领域；

**📈 对比分析**

与 PatchTST、iTransformer、MAE、Chronos‑2、MTS‑JEPA 等基线同等下游头比较。HEPA 在 100% 标签下 10/14 任务获胜，生命周期数据（C‑MAPSS）上 92% 的全标记性能可用 2% 标签获得；参数量比 PatchTST 少约 11 倍；在 10% 标签场景下 6/14 任务获胜；总体性能优于或接近领先模型；

**⚠️ 局限性**

局限性：对局部传感器事件（如 BATADAL、MBA）表现不佳，原因是 token 化稀释了关键信号；预训练收益取决于事件前驱持续时间，短前驱场景下预训练无效；跨域预训练仍未实现，需进一步研究；信息保留界定不具可直接量化；

---

## 57. TOPPO: Rethinking PPO for Multi-Task Reinforcement Learning with Critic Balancing

**arXiv ID:** 2605.11473 | [PDF](https://arxiv.org/pdf/2605.11473v1)

**作者:** Yuanpeng Li `[一作]` (University of California Irvine), Rui Miao `[通讯]` (University of Texas Dallas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Meta‑World+ 多任务强化学习上提出 TOPPO，利用四个模块对 PPO 的 Critic 梯度进行重构，显著提升平均与尾部任务的成功率。

**💡 创新点**

核心创新在于：① 系统识别并针对多任务 PPO 的 Critic 梯度失衡；② 设计 PopArt 归一化、Critic LayerNorm、α=1 FairGrad 聚合与 PCGrad 投影四模块组合；③ 通过梯度尺度、方向和聚合公平性全方位调节实现对梯度失配的彻底纠正。

**🔧 技术方法**

使用了 PopArt 归一化、LayerNorm、α=1 FairGrad 加权、PCGrad 梯度投影、优势归一化、梯度裁剪、投影牛顿求解器等技术。

**📊 数据集**

在 Meta‑World V2 多任务基准上进行实验，分别对 MT10 与 MT50 两个规模进行评估。

**📈 对比分析**

与多种 SAC、ARS、CAGrad、PaCo 等 MTRL 方法对比，在 100M 步预算下仅用 717K 参数即可实现 90.9% 平均成功率、56.5% worst‑10 尾部成功率，并在大多数指标上超越现有基线，且收敛速度提前约 40% 的训练步骤。

**⚠️ 局限性**

局限性包括：① 对共享特征的 actor‑critic 架构未做充分验证；② Actor 聚合器的选择仅限于 PCGrad，未探究更优方案；③ PopArt 采用 Welford 样本统计而非 EMA，可能在非平稳回报环境下表现不佳；④ 在极端奖励异质性下，各模块效果差异显著，需要进一步研究。

---

## 58. A Comparative Study of Model Selection Criteria for Symbolic Regression

**arXiv ID:** 2605.11233 | [PDF](https://arxiv.org/pdf/2605.11233v1)

**作者:** Ali Soltani `[一作]` (Aarhus University), Alessandro Lucantonio `[通讯]` (Aarhus University)

**通讯引用:** 784 | [OpenAlex ID](https://openalex.org/A5030428670)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估并比较了多种模型选择准则在符号回归中的性能，尤其关注它们对泛化误差、模型复杂度和真值模型检索的影响。

**💡 创新点**

首次在同一套受控候选表达式上对AIC、AICc、BIC、MDL和Bootstrap修正训练误差（Err_in）进行全面对比，并给出了在七个合成噪声数据集上的实证结果，揭示MDL与BIC在泛化与模型简洁性方面表现突出。

**🔧 技术方法**

使用信息论准则（AIC、AICc、BIC）、最小描述长度（MDL）以及基于Bootstrap的协方差惩罚估计（Err_in）来评估模型优劣，并利用BFGS优化参数、随机森林估计噪声方差。

**📊 数据集**

七个基于常见GP基准函数的合成数据集（含高非线性、非分离性、多维度等），训练集100点，测试集10000点，噪声服从0.1σ(y)的高斯分布。

**📈 对比分析**

通过平均测试误差、k阶精度、平均表达式大小和最小k检索真值模型等四项指标进行比较；实验表明MDL在大多数数据集上能以更小的k获得最低或接近最低的测试误差，且生成的模型最简；BIC紧随其后；AIC和AICc略逊；Err_in性能不稳定且计算成本高。

**⚠️ 局限性**

局限性包括：仅在单一噪声水平下测试；候选表达式来源为手工扰动，未考虑真实GP搜索过程；Bootstrap方法计算量大，未能在实际搜索循环中高效应用；对真实世界数据的鲁棒性尚未验证。

---

## 59. Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization

**arXiv ID:** 2605.11246 | [PDF](https://arxiv.org/pdf/2605.11246v1)

**作者:** Yonghan Yang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xue Liu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 14001 | [OpenAlex ID](https://openalex.org/A5100372152)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SPA­DE框架，利用条件扩散模型构建前向似然p(y|x)，并加入一阶矩匹配与秩一致性校准以及kNN支持亲近正则化，以解决离线黑盒优化中的OOD问题。

**💡 创新点**

创新点包括：①将条件扩散模型用于前向概率建模；②引入一阶矩匹配与排名一致性校准；③利用kNN支持距离实现均值收缩与方差膨胀，理论上等价于后验优化。

**🔧 技术方法**

使用技术包括条件扩散模型（DDPM）、kNN密度估计、蒙特卡罗采样、低置信界（LCB）获取函数、演化算法、梯度上升和对比损失。

**📊 数据集**

实验数据集涵盖Design‑Bench（Superconductor、Ant、D'Kitty）、LLM Data Mixture 以及TF Bind 8/10。

**📈 对比分析**

与23种前向、逆向和传统方法在128次查询预算下进行归一化最大得分评估，SPA­DE在5/6个任务获得最高分，平均排名仅为2.8，显著优于现有最优方法。

**⚠️ 局限性**

局限性在于kNN支持近似在高维空间的可扩展性有限；计算成本高（需多次蒙特卡罗采样和演化搜索）；当前仅支持单目标优化，缺乏多目标或约束学习。

---

## 60. Mechanism Design for Quality-Preserving LLM Advertising

**arXiv ID:** 2605.10964 | [PDF](https://arxiv.org/pdf/2605.10964v1)

**作者:** Jiale Han `[一作]` (University of California, Los Angeles), Xiaowu Dai `[通讯]` (University of California, Los Angeles)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5000889358)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套质量保留型LLM广告拍卖框架，利用检索增强生成（RAG）将有机内容作为参考并设计内生保留价格，筛选对社会福利有正面贡献的广告；

**💡 创新点**

创新点在于：①将有机内容直接纳入社会福利目标，形成自适应保留价格；②在单一分配和多分配场景下分别设计KL正则化Myerson机制与屏蔽VCG机制，实现基于质量的收入最大化与激励兼容；

**🔧 技术方法**

核心技术包括检索增强生成（RAG）、Kullback–Leibler正则化、Myerson支付公式、VCG机制、以及基于检索权重的保留价格设计；

**📊 数据集**

实验使用人工构造的广告集合和查询（如“我去夏威夷旅游可去哪里？”），有机内容由LLM生成的简短百科风格摘要构成；

**📈 对比分析**

与现有基于分段拍卖的基线（有/无替换、多分配版本）在六项指标（广告收入、社会福利、相关性、KL散度、输出质量）上进行对比。结果显示，质量保留机制在收入、福利和相关性上均明显优于基线，并且KL散度更低、语义相似度更高，表明输出质量更好；

**⚠️ 局限性**

局限性包括：①有机福利函数需手工指定，未从用户交互学习；②未考虑广告主预算约束与多轮互动；③仅使用语义相似度作为质量度量，缺乏事实性或用户参与度等更丰富的评估维度。

---

## 61. Beyond Masks: The Case for Medical Image Parsing

**arXiv ID:** 2605.11438 | [PDF](https://arxiv.org/pdf/2605.11438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 62. Making Abstraction Concrete: A Design Space and Interaction Model of Abstraction in Interactive Systems

**arXiv ID:** 2605.11344 | [PDF](https://arxiv.org/pdf/2605.11344v1)

**作者:** Bryan Min `[一作]` (University of California San Diego), Haijun Xia `[通讯]` (University of California San Diego)

**通讯引用:** 2541 | [OpenAlex ID](https://openalex.org/A5016819583)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对457篇HCI论文进行系统梳理，构建了包含单元、粒度、表示、可变性、展示和引导六维的抽象设计空间，并提出Abstraction Spaces交互模型，扩展了传统的执行与评估gulfs。

**💡 创新点**

创新点在于首次把抽象视为交互核心，引入Gulf of Abstraction、超越/不足的对齐概念，并将抽象视作空间而非单层阶梯，从而为设计更丰富的系统提供框架。

**🔧 技术方法**

采用系统文献综述与主题分析方法，基于HCI会议论文构建标签体系；模型构建借助Norman的gulfs框架和概念映射。

**📊 数据集**

使用的“数据集”为457篇符合条件的HCI研究论文，来源包括ACM、IEEE、Google Scholar等。

**📈 对比分析**

通过案例与现有交互模型对比，展示Abstraction Spaces能更完整描述多层抽象和AI交互的认知过程；并未进行定量性能评估，而是提供理论验证与实证案例。

**⚠️ 局限性**

局限性包括：未对模型在实际系统中的效果进行实证实验；对用户自身抽象过程的描述不够细致；模型对跨领域多元抽象的泛化仍待验证。

---

## 63. Template-as-Ontology: Configurable Synthetic Data Infrastructure for Cross-Domain Manufacturing AI Validation

**arXiv ID:** 2605.11259 | [PDF](https://arxiv.org/pdf/2605.11259v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 64. Understanding and Preventing Entropy Collapse in RLVR with On-Policy Entropy Flow Optimization

**arXiv ID:** 2605.11491 | [PDF](https://arxiv.org/pdf/2605.11491v1)

**作者:** Huimin Xu `[一作]` (Nanyang Technological University), Anh Tuan Luu `[通讯]` (Nanyang Technological University)

**通讯引用:** 2402 | [OpenAlex ID](https://openalex.org/A5050386762)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了严格按策略的熵流优化方法OPEFO，用于解决大语言模型RLVR训练中的熵坍塌问题；

**💡 创新点**

创新点在于从token级熵流角度分析熵坍塌，揭示熵增与熵减更新失衡，并通过自适应重新加权熵增/熵减梯度实现严格按策略的熵流平衡；

**🔧 技术方法**

技术上使用基于一阶熵变近似的token级熵流估计，基于软max假设的梯度重加权，并在GRPO框架下实现严格按策略训练；

**📊 数据集**

使用DAPO‑17K数学推理数据集以及Qwen‑2.5‑Math‑7B和Qwen3‑4B‑Base两大模型；

**📈 对比分析**

与GRPO、近似按策略GRPO、熵正则化、Clip‑higher等基线相比，OPEFO在六大数学推理基准（AIME24、AIME25、AMC23、MATH500、Minerva、OlympiadBench）上平均提升约1.7–1.8%，并在Pass@k等多样性评估中表现更好；

**⚠️ 局限性**

局限性包括：依赖一阶熵变近似且软max假设；仅在按策略训练下验证，缺乏对稠密奖励或更复杂信用分配场景的系统评估；仅针对数学推理任务，未检验跨领域通用性。

---

## 65. Vision2Code: A Multi-Domain Benchmark for Evaluating Image-to-Code Generation

**arXiv ID:** 2605.11307 | [PDF](https://arxiv.org/pdf/2605.11307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 66. HEBATRON: A Hebrew-Specialized Open-Weight Mixture-of-Experts Language Model

**arXiv ID:** 2605.11255 | [PDF](https://arxiv.org/pdf/2605.11255v1)

**作者:** Noam Kayzer `[一作]`, Sarel Weinberger `[通讯]` (PwC Next)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一款专门为希伯来语设计的开源稀疏Mixture‑of‑Experts大模型 Hebatron，支持原生 65,536 token 长上下文，训练规模约 154B token，模型 30B 参数但推理时仅激活约 3B 参数。

**💡 创新点**

创新点包括：①首次将 NVIDIA Nemotron‑3 MoE 体系迁移到希伯来语；②采用易到难的三阶段预训练与连续 anti‑forgetting 锚定，实现语言迁移中对原始推理能力的保留；③构建了专门的希伯来语对齐与生成评估数据集（如 Hebrew IFEval），并在 SFT 阶段使用 200W 规模的双语样本。

**🔧 技术方法**

核心技术：Nemotron‑3‑Nano 30B A3B MoE 架构、FP8 混合精度训练、分布式 Pipeline/Expert/Tensor 并行、MinHash 去重、长文本预处理与多阶段课程学习、知识蒸馏与自监督生成。

**📊 数据集**

使用的数据集涵盖：154B token 的多源希伯来/英语语料（文学、法律、新闻、社交媒体、Long‑Document）、2M 高质量双语 SFT 样本（含 Hebrew IFEval 200k 条、翻译对齐 678k 条、语义结构 9.9k 条、科学推理 214k 条、长文本 43k 条、合成对齐 187k 条）、以及原始 Nemotron 英文推理数据。

**📈 对比分析**

在自动化基准（SNLI、QA、Sentiment、Winograd、Translation、Israeli Trivia）和英语推理基准（HellaSwag、GSM8K、Psi）上，Hebatron SFT 后的希伯来平均得分 73.8%（高于 DictaLM‑3.0‑24B‑Thinking 68.9%），在 GSM8K‑HE、Israeli Trivia 上与 Gemma‑3‑27B‑IT 竞争；推理吞吐量比 Gemma‑3‑27B‑IT 高约 9×，且激活参数仅 1/9。

**⚠️ 局限性**

局限性：与 Gemma‑3‑27B‑IT 相比，整体表现仍略逊一筹；在某些任务（如 MMLU‑HE、Psi‑HE）性能仍落后；模型依赖大量英语原始数据，可能在极端希伯来语文化细节上受限；长上下文训练成本高，资源要求仍显著。

---

## 67. Rethinking LLMOps for Fraud and AML: Building a Compliance-Grade LLM Serving Stack

**arXiv ID:** 2605.11232 | [PDF](https://arxiv.org/pdf/2605.11232v1)

**作者:** Prathamesh Vasudeo Naik `[一作]` (University of Southern California), Yue Wang `[通讯]` (University of Illinois)

**通讯引用:** 55716 | [OpenAlex ID](https://openalex.org/A5113600509)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并评估了一套针对欺诈检测与反洗钱场景的 LLM 服务堆栈，专门针对前缀占主导、结构化短输出的工作负载实现高吞吐、低延迟且合规的部署；

**💡 创新点**

创新点在于将自动前缀缓存、分页 KV、前缀重用、多适配器与长度感知批处理、睡眠/唤醒生命周期以及 LLM-as-判定质量门控等技术整合为工作负载感知的整体优化框架；

**🔧 技术方法**

使用的技术包括 vLLM、PagedAttention、Automatic Prefix Caching、LMCache、SGLang、EAGLE 推测解码、CUDA 图、LoRA 多适配器、预取/解码分离以及 GPU 睡眠/唤醒等；

**📊 数据集**

采用公开合成 AML 数据集 IBM AML、SAML‑D（合成 AML 数据）以及 SynthAML 构造前缀占主导的合规提示；

**📈 对比分析**

通过基准仪表盘对 P50/P95/P99 延迟、吞吐量、GPU 利用率等指标进行测量，工作负载感知调优将吞吐提升约 5.5–5.9×，P99 延迟从数十秒降至单数秒，GPU 利用率从 12% 升至 78%，成本下降约 83%；

**⚠️ 局限性**

局限性包括依赖公开合成数据无法覆盖真实机构细节；LLM-as-判定质量门控可能受偏差影响；高级技术（LMCache、预取/解码分离）尚未完全稳定；实验规模与任务范围有限。

---

## 68. Optimistic Dual Averaging Unifies Modern Optimizers

**arXiv ID:** 2605.11172 | [PDF](https://arxiv.org/pdf/2605.11172v1)

**作者:** Thomas Pethick `[一作]` (Independent Researcher), Volkan Cevher `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 SODA（通用乐观双重平均）框架，将 Optimistic Dual Averaging 泛化，用于解释并统一多种现代优化器。

**💡 创新点**

创新点在于把 Muon、Lion、AdEMAMix、NAdam 等优化器视为 SODA 的乐观实例，并推出了一个无需额外调参的权重衰减包装器，采用理论推导的 1/k 下降调度。

**🔧 技术方法**

核心技术是基于 Optimistic Dual Averaging 的理论框架，并在此基础上设计 1/k 权重衰减调度及包装器实现。

**📊 数据集**

实验使用了多种常见任务和不同规模/训练时长的基准（如语言建模、图像分类等），但文中未列出具体数据集名称。

**📈 对比分析**

通过在多种规模和训练时长下与现有优化器对比实验，SODA 在不需要额外超参数调优的情况下持续提升了性能，实验结果表明其具有稳定的改进效果。

**⚠️ 局限性**

主要限制包括：理论分析主要针对凸优化场景；在极大规模模型或非凸任务上的表现尚未深入验证；对特定任务的细粒度性能提升仍需进一步实验确认。

---

## 69. Measuring Five-Nines Reliability: Sample-Efficient LLM Evaluation in Saturated Benchmarks

**arXiv ID:** 2605.11209 | [PDF](https://arxiv.org/pdf/2605.11209v1)

**作者:** Eungyeup Kim `[一作]` (Carnegie Mellon University), J. Zico Kolter `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17427 | [OpenAlex ID](https://openalex.org/A5075035644)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在已饱和基准上评估大型语言模型可靠性的推理高效框架，利用参数化GSM模板对模型错误率进行精确估计。

**💡 创新点**

创新点在于发现LLM错误在参数空间中高度集中，并采用交叉熵方法(CEM)学习聚焦于失败输入的采样分布，从而在罕见失败率估计中显著降低所需推理次数。

**🔧 技术方法**

技术手段包括交叉熵方法、重要性采样与防御采样、少量抽样的自一致性投票（self‑consistency）以及引入自举重采样以节省推理成本。

**📊 数据集**

使用的实验数据集为九个参数化GSM8K模板（每个约10万条样本），以及三种中等规模LLM：Qwen2.5-Math-7B-Instruct、gpt-oss-20b-low 与 Gemini 2.5 Flash Lite。

**📈 对比分析**

与均匀采样相比，该方法在给定置信区间宽度（≈1%相对标准误）时，推理量可提升至156倍（最高）甚至数十倍，且置信区间覆盖率保持在99% ±1%，显著提高估计效率。

**⚠️ 局限性**

局限性包括：假设错误高度集中，若错误分布较为均匀或仅使用单次推理（K=1）时效率下降；仅在中等规模模型上验证，扩展到更大模型仍需进一步研究；高置信度与极低失败率时所需推理量仍然很大。

---

## 70. Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights

**arXiv ID:** 2605.11330 | [PDF](https://arxiv.org/pdf/2605.11330v1)

**作者:** Wenbo Chen `[一作]` (Amazon), Leman Akoglu `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了一个新的基于检索增强生成（RAG）的幻觉检测基准（HDB），并对现有基准进行系统评估。

**💡 创新点**

提出了包含长上下文、真实人类标注、四种噪声训练标签、多种LLM、多领域的HDB，填补了RAG基准缺口。

**🔧 技术方法**

利用人类多轮标注、弱监督标签、噪声模拟以及多种检测方法（SelfCheckGPT、FS、PO、SFT、ReDeEP等）进行实验。

**📊 数据集**

使用来自多大数据集（如NaturalQuestions、MS‑MARCO、COVID‑QA、DROP等）和三款LLM（SOTA LLM、Gemma‑7b、Mixtral‑8x7b）生成样本。

**📈 对比分析**

对比了无监督与有监督检测器，在非有机样本上表现优异，而在有机RAG样本上F1低于0.7，显示检测器仍有巨大提升空间。

**⚠️ 局限性**

局限性在于只关注基于知识检索的问答任务、仅评估真实性而非事实性、仅为文本单模态、且预过滤可能导致样本偏倚。

---

## 71. The Semantic Training Gap: Ontology-Grounded Tool Architectures for Industrial AI Agent Systems

**arXiv ID:** 2605.11234 | [PDF](https://arxiv.org/pdf/2605.11234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 72. A Theory of Time-Sensitive Language Generation: Sparse Hallucination Beats Mode Collapse

**arXiv ID:** 2605.11302 | [PDF](https://arxiv.org/pdf/2605.11302v1)

**作者:** Atul Ganju `[一作]`, Shang-Hua Teng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言生成中的及时性、密度与幻觉之间的理论关系，给出了不可行性与可行性的边界

**💡 创新点**

首次证明在任何一致生成器下及时密度必为0，并提出允许稀疏幻觉可实现1/2的及时密度；提出黑盒降维和推测算法

**🔧 技术方法**

利用生成游戏模型、概率与信息论工具、Borel–Cantelli、Chernoff与 Freedman 不等式等理论分析

**📊 数据集**

无具体数据集，全部为理论证明

**📈 对比分析**

未进行实验比较，性能通过理论证明给出密度1/2和幻觉率与截止函数之间的最优取舍

**⚠️ 局限性**

需超线性截止函数；对线性截止函数仍无解，且对语言层面及时性、黄金比例假设等问题仍开放

---

## 73. Curriculum Learning-Guided Progressive Distillation in Large Language Models

**arXiv ID:** 2605.11260 | [PDF](https://arxiv.org/pdf/2605.11260v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 74. Instructions shape Production of Language, not Processing

**arXiv ID:** 2605.11206 | [PDF](https://arxiv.org/pdf/2605.11206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 75. Causal Bias Detection in Generative Artifical Intelligence

**arXiv ID:** 2605.11365 | [PDF](https://arxiv.org/pdf/2605.11365v1)

**作者:** Drago Plecko `[一作]` `[通讯]` (UCLA), Drago Plecko (UCLA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于因果推理的生成式 AI 公平性评估框架，统一标准机器学习与生成模型的公平性分析，并在大型语言模型中量化种族和性别偏差。

**💡 创新点**

创新点：① 将 S‑型标准公平性模型 (S‑SFM) 引入生成式 AI，允许对模型内部机制进行选择性替换；② 推导出可识别的因果分解公式（ΔTV、ΔDE、ΔIE、ΔSE），实现对直接、间接和混杂路径及机制替换对偏差影响的细粒度分析；③ 在标准机器学习情境下该框架退化为已有结果，证明其更一般性。

**🔧 技术方法**

使用结构因果模型（SCM）、潜在变量推断、因果效应分解（DE/IE/SE）、可识别公式与高效估计器（基于条件分布乘积），以及在 LLM 中通过 Prompt‑ing、生成器（Γ）与注释器（A）构造条件样本。

**📊 数据集**

使用的公开数据集：NSDUH 2023（种族与大麻使用）、BRFSS 2023（种族与糖尿病）、ACS 2023（性别与薪酬）。

**📈 对比分析**

对 10 个开源 LLM（Llama、Qwen、Gemma、DeepSeek、Ministral、Phi‑4 等）分别构造三阶段机制替换数据集，计算 9 维偏差签名，按比例分析模型在放大、抑制或反转偏差上的表现，并通过层次聚类评估模型间偏差相似性；结果表明不同模型在直接、间接、混杂路径上的偏差行为差异显著，且模型家族并非决定性因素。

**⚠️ 局限性**

局限性：① 仅适用于可对任意条件进行查询的生成模型；② 需要手工指定 S‑SFM，受因果假设限制；③ 对视觉等非语言模态的适用性有限；④ 仅诊断偏差不提供纠正方法；⑤ 对生成器与注释器的质量依赖较高。

---

## 76. VidSplat: Gaussian Splatting Reconstruction with Geometry-Guided Video Diffusion Priors

**arXiv ID:** 2605.11424 | [PDF](https://arxiv.org/pdf/2605.11424v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 77. Predictive Maps of Multi-Agent Reasoning: A Successor-Representation Spectrum for LLM Communication Topologies

**arXiv ID:** 2605.11453 | [PDF](https://arxiv.org/pdf/2605.11453v1)

**作者:** Ethan David James Park `[一作]` (University of Arizona), Dalal Alharthi `[通讯]` (University of Arizona)

**通讯引用:** 76 | [OpenAlex ID](https://openalex.org/A5082014850)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多智能体LLM系统的通信拓扑结构，通过构造成功者表示（SR）并提取谱半径、谱间隙和条件数作为预推理诊断量，预测漂移放大、共识速度和抗扰动鲁棒性。

**💡 创新点**

首次将成功者表示谱量应用于多智能体LLM拓扑诊断，并提出漂移校正半径ρ̃来解决谱半径与累计误差的逆相关性。

**🔧 技术方法**

使用图论的行标准化邻接矩阵构成随机行演算子P，求逆得到M=(I-γP)⁻¹，计算其谱半径、谱间隙和条件数，以及漂移校正公式；实验采用Qwen2.5-7B-Instruct生成器。

**📊 数据集**

采用自定义的12步结构化JSON状态跟踪任务，测试链、四叶星和四节点网状三种拓扑。

**📈 对比分析**

通过100次独立实验评估累计误差、共识衰减和扰动敏感度，发现条件数完全匹配扰动鲁棒性，谱间隙部分预测共识速度，谱半径与误差呈完全负相关；使用Spearman ρ_r_s 评价排名一致性。

**⚠️ 局限性**

局限在于仅检验三种拓扑、单一模型族和单一任务，统计功效有限，未验证噪声相关性、动态权重或更大规模拓扑的普适性。

---

## 78. Digital Identity for Agentic Systems: Toward a Portable Authorization Standard for Autonomous Agents

**arXiv ID:** 2605.11487 | [PDF](https://arxiv.org/pdf/2605.11487v1)

**作者:** Partha Madhira `[一作]` `[通讯]`, Partha Madhira

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出一种可移植授权语义模型，旨在解决企业级自主智能体跨组织边界执行任务时的身份识别与信任治理问题，结合保险理赔和国防航空供应链两个案例分析现有身份管理模型的缺口，并设计三层架构（容器、授权载荷、执行引擎）与四类约束表达式，定义评估语义、委托衰减、最严格优先、审计与撤销等完整流程；

**💡 创新点**

创新点在于：①将身份与授权从点对点验证扩展为可跨域、可约束、可撤销的授权载荷；②构建统一、可组合的约束代数（数值、时间、枚举、字符串模式），实现实时决策与后续审计一致性；③通过预飞行发现、信任注册与治理目录实现语义互操作性与跨域治理；④在三层架构中分离容器、载荷与引擎，保证授权语义可在不同技术栈与协议下复用。

**🔧 技术方法**

核心技术包括：JSON‑LD/VC数据完整性、JWT/VC签名验证、可扩展约束代数与评估引擎、最小可行词汇表与行业映射配置、委托衰减算法、撤销列表、审计日志签名、治理注册与预飞行发现接口（如Discovery Endpoint）。

**📊 数据集**

该工作为概念性规范与示例，未使用传统机器学习或大规模实验数据集；主要使用两条业务案例（保险理赔、航空供应链）进行结构化需求与语义映射演示。

**📈 对比分析**

由于本文以标准化规范为主，未开展实测对比；但通过案例演示的伪代码与预期流程说明，评估语义在不同实施引擎下应保持一致，可行性与可扩展性在理论上得到保证；在性能上，评估复杂度为 O(n)（n 为约束数量）并支持流式实时决策。

**⚠️ 局限性**

局限性包括：①模型对跨域委托不作默认支持，需独立授权；②在高度动态的业务场景下，约束表达式的手动维护与映射更新可能导致运维复杂；③隐私风险仍需通过短期、受限受众的凭证与最小化披露进一步缓解；④缺乏实际部署与基准测试，验证其在大规模高并发系统中的可伸缩性与性能。

---

## 79. Newton's Lantern: A Reinforcement Learning Framework for Finetuning AC Power Flow Warm Start Models

**arXiv ID:** 2605.11102 | [PDF](https://arxiv.org/pdf/2605.11102v1)

**作者:** Shourya Bose `[一作]` (Pravah), Dhruv Suri `[通讯]` (Pravah)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了AC电力流问题中的温暖启动，证明Newton‑Raphson迭代次数的下界主要由误差方向决定，并基于此提出了“Newton's Lantern”这一强化学习微调管线；

**💡 创新点**

创新点在于：①给出误差方向对NR迭代次数的理论下界，解释监督回归在接近电压崩溃时失效的原因；②提出使用群相对策略优化（GRPO）结合学习的奖励模型，以迭代次数作为监督信号进行微调；③在极端负载情况下实现了全样本收敛和最低平均迭代次数。

**🔧 技术方法**

使用了Newton‑Raphson理论分析、方向敏感性证明、GRPO算法、PPO基础、学习奖励模型以及对NR求解结果的自适应采样。

**📊 数据集**

使用PFΔ基准数据集，包含IEEE 118、GOC 500、GOC 2000三种电网，分别划分为电压稳定实例和靠近电压崩溃的实例。

**📈 对比分析**

与平面起点、DC起点、监督微调（SFT）以及带oracle基线的PPO+V*进行对比。Newton's Lantern在所有30个测试样本上实现100%收敛，并在收敛样本中取得最低平均迭代次数，尤其在难点压降实例上恢复了SFT失效的三例。

**⚠️ 局限性**

局限性：在大型GNN结构的RL微调中出现训练不稳定，需要极低学习率和小剪辑阈值；对更大规模网格的可扩展性和鲁棒性仍需进一步研究。

---

## 80. A Generative AI Driven Interactive Narrative Serious Fame for Stress Relief and Its Randomized Controlled Pilot Study

**arXiv ID:** 2605.11562 | [PDF](https://arxiv.org/pdf/2605.11562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 81. A Mimetic Detector for Adversarial Image Perturbations

**arXiv ID:** 2605.11492 | [PDF](https://arxiv.org/pdf/2605.11492v1)

**作者:** Johnny Corbino `[一作]` (Lawrence Berkeley National Laboratory), Johnny Corbino `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 39 | [OpenAlex ID](https://openalex.org/A5063232431)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单次、无训练、无网络访问的对抗扰动检测器，利用高阶 Corbino–Castillo 媒体梯度算子计算梯度能量比 T 来区分干净图像和被攻击图像。

**💡 创新点**

创新点在于将 MOLE 库中的高阶 mimetic 方案与无训练阈值校准相结合，既实现了检测的可部署性，又使检测性能随梯度算子阶数 k 单调提升。

**🔧 技术方法**

使用了 Corbino–Castillo 高阶 mimetic 梯度算子、MOLE 计算框架、欧氏梯度能量比 T 以及阈值化判定。

**📊 数据集**

仅使用 128×128 灰度版本的 PEPPERS 图像，生成干净、FGSM/PGD 形态对抗扰动以及无高频光滑控制扰动三种输入。

**📈 对比分析**

在此测试案例中，与 SpectralDefense 等基线相比，检测器实现了清洁与对抗图像 T 的比值从 3.55 倍提升至 4.19 倍；算法复杂度为 O(HW)，并随阶数提升检测边际单调改善。

**⚠️ 局限性**

局限性包括：对自适应攻击缺乏认证，攻击者若刻意最小化 T 可降低攻击成功率或需更大 ℓ∞ 范围；当前仅针对单通道灰度图，彩色、多光谱或非矩形传感器场景需进一步扩展。

---

## 82. RIO: Flexible Real-Time Robot I/O for Cross-Embodiment Robot Learning

**arXiv ID:** 2605.11564 | [PDF](https://arxiv.org/pdf/2605.11564v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 83. The Granularity Mismatch in Agent Security: Argument-Level Provenance Solves Enforcement and Isolates the LLM Reasoning Bottleneck

**arXiv ID:** 2605.11039 | [PDF](https://arxiv.org/pdf/2605.11039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 84. Deep Reasoning in General Purpose Agents via Structured Meta-Cognition

**arXiv ID:** 2605.11388 | [PDF](https://arxiv.org/pdf/2605.11388v1)

**作者:** Dean Light `[一作]` (University of Washington), Yulia Tsvetkov `[通讯]` (University of Washington)

**通讯引用:** 5329 | [OpenAlex ID](https://openalex.org/A5062910836)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为Deep Reasoning的形式化语言，并基于此构建Deep Layered Reasoning Scaffold（DLRS）代理，能够在推理时即时生成任务专用的推理骨架，自动化地将人类的元推理轨迹转化为可执行的分解策略；

**💡 创新点**

创新点在于：①将人类元推理映射为可执行的结构化语言，兼顾关联与形式推理；②通过少量的“原子”示例实现在上下文中自适应地构造推理骨架；③实现了推理负载分层、递归子任务调用，提升推理稳定性；

**🔧 技术方法**

技术包括：大语言模型（Qwen3-8B/32B、Llama-3.3-70B）与Python解释器的协同执行；关联推理使用LLM，形式推理使用可执行代码；递归子代理调用；基于人类推理轨迹的示例式提示；token与认知负载分析；

**📊 数据集**

数据集涵盖四大难题基准：SynthWorlds、PhantomWiki、DeepSearchQA 与 Oolong-real，分别考察多跳推理、信息检索、长上下文聚合与深度研究型问答；

**📈 对比分析**

与ReAct、CodeAct、Deep Research、RLM等四种主流脚手架在三种模型尺寸与两族群上进行对比；DLRS平均提升24.8%（在11/12实验设置中获胜），甚至在8B模型下超过同族32B基准，显示出显著性能优势；

**⚠️ 局限性**

局限性包括：①推理线程增多导致token成本显著提高；②需要人工编写“原子”示例，缺乏完全自动化的分解生成；③在没有示例的情况下LLM难以自行推断结构化分解；④在某些基准上仍易出现早期终止与幻觉；

---

## 85. Physics-Informed Teacher-Student Ensemble Learning for Traffic State Estimation with a Varying Speed Limit Scenario

**arXiv ID:** 2605.11346 | [PDF](https://arxiv.org/pdf/2605.11346v1)

**作者:** Archie J. Huang `[一作]` (Concordia University), Muhammad Shahbaz `[通讯]` (University of Central Florida)

**通讯引用:** 699 | [OpenAlex ID](https://openalex.org/A5040274232)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了教师‑学生式的 Physics‑Informed Deep Learning（PIDL）集成框架，用于在不同速度限制（VSL）场景下的交通状态估计。

**💡 创新点**

创新点在于将局部物理约束的教师 PIDL 模型与全局学生 MLP 分类器相结合，形成可根据路段交通特性动态选择教师模型的集成学习体系，从而克服传统 PIDL 在空间异质交通中的局限。

**🔧 技术方法**

使用了 Physics‑Informed Deep Learning、教师‑学生集成学习、多层感知机（MLP）分类器、LWR + Greenshields 基本图、Lax‑Hopf 速度密度模拟等技术。

**📊 数据集**

采用了 5000 米长度、分为 5 个 1000 米 VSL 段的合成测试床数据，使用 LWR 模型生成的密度时间序列；训练仅使用了 0.12% 的数据点。

**📈 对比分析**

与非集成 PIDL、纯深度学习网络、LSTM+插值等基线方法对比，教师‑学生集成模型的相对 L₂ 误差为 3.89×10⁻²，显著优于非集成 PIDL（6.24×10⁻²）、深度学习（1.47×10⁻¹）和 LSTM（4.08×10⁻¹）。

**⚠️ 局限性**

限制包括：假设路段内交通特性为分段常数，依赖 Greenshields 基本图，验证仅在合成数据上完成，未检验对真实噪声测量和动态 VSL 策略的鲁棒性。

---

## 86. Optimal Interventions on the Linear Threshold Model in Large-Scale Networks

**arXiv ID:** 2605.11337 | [PDF](https://arxiv.org/pdf/2605.11337v1)

**作者:** Leonardo Cianfanelli `[一作]` (Politecnico di Torino), Fabio Fagnani `[通讯]` (Politecnico di Torino)

**通讯引用:** 4561 | [OpenAlex ID](https://openalex.org/A5066105783)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于局部均值场近似的线性阈值模型（LTM）最少成本干预问题（LCIP）并将其转化为可求解的线性规划

**💡 创新点**

创新点在于仅利用网络统计信息（而非完整结构）即可得到接近最优的干预方案，并将无限约束的线性规划通过离散化转化为有限维线性规划，从而实现大规模网络的可扩展求解

**🔧 技术方法**

使用了局部均值场近似、配置模型（CM）采样、递归方程分析、线性规划以及其离散化方法

**📊 数据集**

使用了真实网络数据集：Epinions（26588节点）和美国西部电网网络（4941节点）

**📈 对比分析**

通过与最优种子设置（WTSS）和已有LCIP算法对比，实验显示在保证目标采纳率的前提下，成本降低约25%，且在实验网络中干预效果与理论递归预测高度一致

**⚠️ 局限性**

局限性在于理论保证仅适用于从配置模型抽样的随机网络，对非随机网络的理论保证有限；离散化参数（如Δ、N）需要经验选择，且对阈值分布的假设仍然影响方法的适用范围

---

## 87. Natural Language based Specification and Verification

**arXiv ID:** 2605.11315 | [PDF](https://arxiv.org/pdf/2605.11315v1)

**作者:** Zhaorui Li `[一作]` (University of California Riverside), Chengyu Song `[通讯]` (University of California Riverside)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 NLForge 框架，利用 LLM 生成自然语言的函数级别总结并进行逐步下推的内存安全验证；

**💡 创新点**

创新点在于把 LLM 作为推理主体，使用自然语言中间表示进行模块化、可复用的验证流程，而不是传统的形式化规范生成与符号检查；

**🔧 技术方法**

采用多种现代 LLM（Claude Sonnet‑4.6、GPT‑5.4、Gemini‑3.1‑flash‑lite、Qwen3.5‑27B‑UD‑Q4），结合调用图、自然语言摘要、JSON 结构化表示及底层抽象；

**📊 数据集**

使用 SV‑COMP 2026 C 内存安全子集（包括 Juliet、data‑structure、control‑flow 等类别）进行评测；

**📈 对比分析**

与传统静态验证器（Symbiotic、CPAchecker、UAutomizer）在 TP/FP/TN/FN、准确率、召回率等指标下对比，结果表明 NLForge 在大型、路径敏感任务中提升了召回率，整体性能接近成熟工具，但仍有较多假阳性；

**⚠️ 局限性**

局限性包括仅评估了部分基准与模型；摘要质量不稳定可能导致误差放大；缺乏对间接调用、复杂堆结构、符号证明等高级特性的完整支持；缺乏正式的安全证明与误差校正机制。

---

## 88. TB-AVA: Text as a Semantic Bridge for Audio-Visual Parameter Efficient Finetuning

**arXiv ID:** 2605.11572 | [PDF](https://arxiv.org/pdf/2605.11572v1)

**作者:** Seongah Kim `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用冻结的音频、视觉与文本编码器，插入轻量化的 TB-AVA 适配器，通过文本作为语义锚点实现跨模态对齐。

**💡 创新点**

创新点在于将文本转化为可控的通道门控（GSM），在不修改主干的前提下实现参数高效、可解释的音视频交互。

**🔧 技术方法**

技术核心为文本桥接的跨模态适配器（TB-AVA）与门控语义调制（GSM）机制，配合冻结的 SigLIP2/BEATs 编码器。

**📊 数据集**

实验使用 AVE、AVVP（LLP）和 AVSBench 对象分割三大数据集。

**📈 对比分析**

与多种全参数或轻量化对齐方法对比，TB-AVA 在 AVE 上实现 85.0%（比 MoLT 高 1.5pp）并在 AVVP 与 AVSBench 上均保持竞争力。

**⚠️ 局限性**

局限性包括对固定类别文本的依赖，导致多源分割等多标签任务中无法完全区分不同声源与目标。

---

## 89. Robust Multi-Agent Path Finding under Observation Attacks: A Principled Adversarial-Plus-Smoothing Training Recipe

**arXiv ID:** 2605.11469 | [PDF](https://arxiv.org/pdf/2605.11469v1)

**作者:** Riad Ahmed `[一作]` `[通讯]` (University of New Hampshire), Riad Ahmed (University of New Hampshire)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出两种训练方案，在不改变网络架构和部署流程的前提下，使分布式多智能体路径规划（MAPF）对观测扰动具备鲁棒性。

**💡 创新点**

创新在于将单智能体的状态对抗训练（SA-PPO）推广到共享参数多智能体场景，并在此基础上加入局部平滑正则与随机平滑的MACER可微化安全边界项。

**🔧 技术方法**

使用Proximal Policy Optimization (PPO)作为基础，加入对抗扰动、TRADES风格的平滑正则、SA-PPO正则以及MACER的安全边界损失，并在训练时进行对抗采样。

**📊 数据集**

在POGEMA 8×8格子、4个智能体、障碍密度0.1的环境上进行实验，使用随机种子生成地图和起点/终点。

**📈 对比分析**

与基准PPO及后置平滑策略对比，Adv-PPO将最差攻击成功率从2.5%提升至59.2%（清洁性能基本不变），Adv-PPO+MACER进一步提升至77.5%±6.0%（清洁成功率约95%）。

**⚠️ 局限性**

实验规模有限，仅在8×8格子与4个智能体上验证，攻击强度采用单次PGD随机初始化，未评估更大地图或更强攻击的鲁棒性。

---

## 90. Distance-Constrained Unlabeled Multi-Agent Pathfinding

**arXiv ID:** 2605.11503 | [PDF](https://arxiv.org/pdf/2605.11503v1)

**作者:** Takahiro Suzuki `[一作]` (Tohoku University), Keisuke Okumura `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 2687 | [OpenAlex ID](https://openalex.org/A5038362443)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在图上加入距离-r 约束的无标签多智能体路径规划问题，并给出了两套求解方案：基于整数线性规划与压缩的最优算法，以及基于配置生成器并带活锁检测的搜索算法。

**💡 创新点**

核心创新在于将传统冲突模型推广为距离-r 独立性约束，证明该问题为 PSPACE‑complete，并同时提出可压缩核化技术和避免旋转与活锁的配置生成器与搜索框架。

**🔧 技术方法**

采用整数线性规划（ILP）+ 核化压缩、递归优先级规划的配置生成器（PIBT 变体）以及带活锁检测与目标重新分配的搜索框架。

**📊 数据集**

使用经典网格与仓库图（empty-16-16、random-64-64-20、lak303d、warehouse-10-20-10-2-2 等）以及随机生成的距离-r 独立集合实例。

**📈 对比分析**

与现有无标签规划算法（如 TSWAP）在运行时间和子最优度上进行对比。实验显示，基于配置生成器+搜索的方法在大规模实例中成功率高、计划长度接近下界；在 r=0 时与 TSWAP 的性能相当。

**⚠️ 局限性**

局限性包括：对 r≥1 的可行性判定仍为 PSPACE‑complete；在高密度或狭窄通道图中可能因缺乏足够的距离独立集合或陷入活锁而失效；压缩方法受限于图度 Δ 和智能体数 n；基于 ILP 的最优方法在大规模实例上不可扩展。

---

## 91. On the Approximation Complexity of Matrix Product Operator Born Machines

**arXiv ID:** 2605.11471 | [PDF](https://arxiv.org/pdf/2605.11471v1)

**作者:** Chao Li `[一作]` (RIKEN), Qibin Zhao `[通讯]` (RIKEN)

**通讯引用:** 8277 | [OpenAlex ID](https://openalex.org/A5083182987)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了矩阵乘积算子 Born 机（MPO‑BM）的近似能力，证明了其在最坏情况下的 KL 近似为 NP‑难，并在局部性与谱间隙条件下给出了多项式边维和查询复杂度的理论保证。

**💡 创新点**

创新点在于将 Born 机的近似难度与 NP 难性相连，并首次将量子力学中的局部 gapped Hamiltonian 与 Score‑Based Variational Inference 结合，提出了 MPO‑BM 在结构化分布下的可学习性理论。

**🔧 技术方法**

采用了张量网络（MPO）表示、Score‑Based Variational Inference（EigenVI）框架、局部 Hamiltonian 的谱间隙理论、矩阵 Bernstein 以及局部重要性采样等技术。

**📊 数据集**

使用了合成的高维目标分布（高斯、GMM‑3、X‑shape、Ring、Funnel）以及通过路径图 Markov 随机场嵌入得到的多维数据。

**📈 对比分析**

与全局重要性采样（EigenVI）对比，局部采样在低查询量下实现更低的前向 KL 散度，并且 MPO 维度增长呈多项式，优于全局方法随维度指数增长的表现。

**⚠️ 局限性**

主要局限在于谱间隙假设尚未能从目标分布直接推导，且 NP‑难性证明依赖于对连续密度的特殊构造，难以推广到更一般情形。

---

## 92. When Looking Is Not Enough: Visual Attention Structure Reveals Hallucination in MLLMs

**arXiv ID:** 2605.11559 | [PDF](https://arxiv.org/pdf/2605.11559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. LiBrA-Net: Lie-Algebraic Bilateral Affine Fields for Real-Time 4K Video Dehazing

**arXiv ID:** 2605.11508 | [PDF](https://arxiv.org/pdf/2605.11508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 94. Rotation-Preserving Supervised Fine-Tuning

**arXiv ID:** 2605.10973 | [PDF](https://arxiv.org/pdf/2605.10973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 95. BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion

**arXiv ID:** 2605.11577 | [PDF](https://arxiv.org/pdf/2605.11577v1)

**作者:** Shaobin Zhuang `[一作]` (Shanghai Jiao Tong University), Hao Chen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 BitLM，一种将词汇 softmax 替换为固定长度二进制编码并通过轻量级扩散头实现块级并行解码的语言模型。

**💡 创新点**

创新点在于将输出空间从离散词表转为二进制空间，并将并行生成嵌入自回归背骨中，实现块级因果并行生成。

**🔧 技术方法**

使用二进制编码、位置 MLP、BlockCausal LLM、轻量级扩散头、AdaLN、ODE 求解器和分类器无关引导等技术。

**📊 数据集**

在 FineWeb‑350B 上预训练，并在 XSum 数据集上微调。

**📈 对比分析**

与传统自回归模型和指针生成基线相比，预训练的 8B BitLM 在 XSum 上取得 26.05/6.44/20.12 的 ROUGE‑1/2/L，速度显著提升，但整体性能仍低于最佳基线。

**⚠️ 局限性**

在语言细粒度控制、复制机制和块大小自适应等方面仍有限，当前二进制表征对高精度词汇实现尚未充分优化。

---

## 96. LPDP: Inference-Time Reward Control for Variable-Length DNA Generation with Edit Flows

**arXiv ID:** 2605.11368 | [PDF](https://arxiv.org/pdf/2605.11368v1)

**作者:** Jeongchan Kim `[一作]` (KAIST AI), Jong Chul Ye `[通讯]` (KAIST AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一种基于Edit Flow的推理时奖励倾斜框架LPDP，用于可变长度DNA序列生成。

**💡 创新点**

通过局部离散规划对根编辑进行重排序，利用插入、删除、替换的类型几何，提供训练无关且可扩展的局部搜索。

**🔧 技术方法**

使用Edit Flow生成模型、局部动态规划、Log‑sum‑exponential/Max备份以及同类型候选规则（Mixed、ST‑after、ST‑first），并结合冷冻奖励oracle。

**📊 数据集**

使用GRCh38的可变长度增强子序列（200–400 bp）用于增强优化，以及exon–intron–exon拼接数据用于splice inpainting。

**📈 对比分析**

与CEM、Beam、TDS、SMC等搜索基线在相同搜索预算下对比，LPDP（尤其ST‑after‑LSE）在增强活性、3-mer JSD和splice Geomean等指标上均显著优于基线。

**⚠️ 局限性**

局限于根编辑带宽和有限局部图，无法捕捉长程或多站点交互；且结果仅基于预测oracle，需实验验证。

---

## 97. Couple to Control: Joint Initial Noise Design in Diffusion Models

**arXiv ID:** 2605.11311 | [PDF](https://arxiv.org/pdf/2605.11311v1)

**作者:** Jing Jia `[一作]` (Rutgers University), Guanyang Wang `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种初始噪声耦合框架，通过在多样本批次中指定噪声的联合分布来控制扩散模型的输出多样性与背景生成。

**💡 创新点**

创新点在于将初始噪声视为可耦合的多样本联合分布，提出最优的“repulsive”耦合以及子空间/学习耦合方法，使得在不改变模型或采样流程的前提下提升多样性和结构化控制。

**🔧 技术方法**

主要技术包括高斯耦合理论、等相关耦合、矩阵参数化、反向扩散与噪声优化、对比噪声优化等，并在Stable Diffusion v1.5、XL、3上实现。

**📊 数据集**

使用的数据集包括COCO Caption 2000个提示、LSUN Bedroom 100张图像，并利用YOLOv8生成前景掩码。

**📈 对比分析**

通过与独立噪声、CNO、DDIM、SDEdit、Stable Diffusion inpainting等方法对比，发现耦合噪声在多样性指标（L2、MSS、Vendi）上与CNO持平或超越，且无额外采样成本；在固定前景背景生成任务中，耦合方法在背景多样性和自然度上优于基线。

**⚠️ 局限性**

局限性包括仅提供批级控制，难以提升单图质量；耦合参数化目前受限于低维矩阵，难以捕捉复杂语义；未验证在视频、音频、3D等其他模态上的有效性。

---

## 98. Do Vision-Language-Models show human-like logical problem-solving capability in point and click puzzle games?

**arXiv ID:** 2605.11223 | [PDF](https://arxiv.org/pdf/2605.11223v1)

**作者:** Dominik Helfenstein `[一作]` (University of Stuttgart), Maximilian Triebel `[通讯]` (University of Stuttgart)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了VLATIM基准，用以评估视觉语言模型在点选式物理谜题游戏TIM中的逻辑推理与交互能力。

**💡 创新点**

创新点在于设计了涵盖视觉定位、领域理解、事件推理、操纵与完整谜题求解的五阶段基准，并将文本指令与连续鼠标动作统一在同一模型框架内。

**🔧 技术方法**

使用视觉语言模型、基于文本的指令生成、LLM评估器与人类评测相结合的多模态循环交互技术。

**📊 数据集**

数据集为TIM游戏的若干谜题，按任务划分为五部分，包含截图、零样本提示、动作指令等。

**📈 对比分析**

与UI‑Tars、Qwen2.5、Qwen3、Gemini和GPT五大模型进行对比，评价指标包括视觉定位准确度、推理得分、操纵成功率和整体分数；结果显示模型均无法完成完整谜题，Qwen3与Gemini在各自擅长的维度表现最佳。

**⚠️ 局限性**

主要局限在于缺乏精确视觉定位与高阶推理的结合，评测手工执行部分任务，缺少人类基准，且基准对模型的零样本能力关注，未探索提示学习或微调的提升空间。

---

## 99. Hi-GaTA: Hierarchical Gated Temporal Aggregation Adapter for Surgical Video Report Generation

**arXiv ID:** 2605.11208 | [PDF](https://arxiv.org/pdf/2605.11208v1)

**作者:** Kedi Sun `[一作]` (University of Birmingham), Le Zhang `[通讯]` (University of Birmingham)

**通讯引用:** 256691 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于手术视频报告生成的Hi-GaTA框架，并创建了214段高质量手术视频与评估报告的基准数据集。

**💡 创新点**

创新点包括：①针对手术场景的自监督视频编码器Sur40k；②轻量级层次门控时间聚合适配器Hi-GaTA，实现多尺度时间聚合并生成LLM兼容视觉前缀；③结合LoRA微调LLM，实现在有限监督下的临床风格报告生成。

**🔧 技术方法**

使用技术包括：ViViT风格的自监督视频编码、InfoNCE损失、Temporal Pyramid Pooling、Dual Cross-Attention、门控融合、深度递增策略、LoRA微调、LLM生成。

**📊 数据集**

数据集为由大学与医院合作构建的214段手术模拟视频，配有专家评估报告，并基于40,000分钟公开手术视频预训练Sur40k。

**📈 对比分析**

与多模态LLM基线（如LLaVA-Med-v1.5、Qwen2.5-VL-7B）和不同视频编码器/LLM组合进行对比，Hi-GaTA在BLEU、ROUGE-L、METEOR、MedBERTScore、CIDEr等指标上均显著优于基线，且显示更稳定的语义一致性。

**⚠️ 局限性**

局限性包括：基准数据集覆盖手术种类有限，缺乏多中心真实手术视频；模型对极端噪声或隐私敏感信息的鲁棒性尚未验证；以及对临床部署的安全性和可解释性仍需进一步研究。

---

## 100. MLCommons Chakra: Advancing Performance Benchmarking and Co-design using Standardized Execution Traces

**arXiv ID:** 2605.11333 | [PDF](https://arxiv.org/pdf/2605.11333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 101. Sequential Behavioral Watermarking for LLM Agents

**arXiv ID:** 2605.11036 | [PDF](https://arxiv.org/pdf/2605.11036v1)

**作者:** Hyeseon An `[一作]` (Yonsei University), Yo-Sub Han `[通讯]` (Yonsei University)

**通讯引用:** 1472 | [OpenAlex ID](https://openalex.org/A5077698683)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对大型语言模型（LLM）代理的顺序行为水印框架（SeqWM），通过在代理的行动序列中嵌入基于历史的水印信号并以滑动窗口方式进行无位置信息的检测，从而实现轨迹级别的所有权验证。

**💡 创新点**

创新点在于：1）将水印种子由绝对时间步转为基于最近动作窗口的历史依赖，从而避免位置误对齐导致的脆弱性；2）采用多通道冗余分布水印，使单一动作被多通道共同指示，提升鲁棒性；3）引入随机键校准方法，在无独立性假设下仍能给出有效的p值，保证统计检验的合法性；4）整体实现了对删除、截断、部分观测等轨迹破坏的局部化鲁棒。

**🔧 技术方法**

核心技术包括：历史条件多通道编码（利用HMAC生成伪随机子集并对动作分布做指数倾斜）；滑动窗口检测（聚合长度为w+1的子序列中的指示）；随机键校准（通过错误键生成经验分布计算p值）。

**📊 数据集**

实验数据集涵盖三个代理基准：ToolBench（ReAct循环），ALFWorld（单次提示任务），以及OASIS Reddit（多代理模拟），分别在LLaMA‑3.2、Gemma‑4、Qwen‑3等开源LLM上进行测试。

**📈 对比分析**

与基线AgentGuide、AgentMark等方法对比，SeqWM在所有模型和基准上都实现了显著更高的检测z‑score和p‑value（p<0.01），并且在随机动作删除攻击下保持高TPR（删除率30‑50%仍可识别），而基线方法在少量删除后性能急剧下降。水印对代理实用性的影响微乎其微。

**⚠️ 局限性**

局限性包括：尚未针对替换攻击或语义等价轨迹编辑进行评估；需要共享密钥且对密钥管理提出要求；在极短轨迹或极大动作空间时，单步水印强度可能不足；计算与存储成本相对较高，尤其是多通道和随机键校准的实现。

---

## 102. Forecast-aware Gaussian Splatting for Predictive 3D Representation in Language-Guided Pick-and-Place Manipulation

**arXiv ID:** 2605.11144 | [PDF](https://arxiv.org/pdf/2605.11144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 103. Can a Single Message Paralyze the AI Infrastructure? The Rise of AbO-DDoS Attacks through Targeted Mobius Injection

**arXiv ID:** 2605.11442 | [PDF](https://arxiv.org/pdf/2605.11442v1)

**作者:** Zi Liang `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 9075 | [OpenAlex ID](https://openalex.org/A5020630816)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于语义闭包的“莫比乌斯注入”攻击，利用 LLM 代理的组件调用循环将单一文本注入转化为自我持续的 AbO‑DDoS，导致代理成为僵尸节点并在 AI 基础设施上产生大规模资源耗尽。

**💡 创新点**

创新点包括：① 定义 AbO‑DDoS 新威胁模型；② 设计莫比乌斯注入框架，利用闭包机制构造隐藏的递归调用循环；③ 提出 Agent Component Energy (ACE) 监测方法，在组件图层检测异常能量激增，实现针对该攻击的首个实时防御。

**🔧 技术方法**

核心技术：语义闭包（将返回文本视作可执行闭包），组件图转换与递归检测，ACE 组件能量分析，条件触发与目标定位逻辑，以及对多代理环境的协同放大模型。

**📊 数据集**

实验数据集与环境：ClawBench（8 类任务，153 任务），SWE‑Bench 与 HumanEval（各 20 任务），3 种编码代理（Claude Code、OpenCode、Kilo Code）和 3 种手掌代理（OpenClaw、ZeroClaw、Hermes）；12 种 LLM 后端（Claude Sonnet 4.6、GPT‑5.4、Gemini 3.1、DeepSeek、Qwen3.6‑Plus 等）。

**📈 对比分析**

对比方法：在同一代理/任务/模型组合下对照清洁执行和注入后执行，测量 P‑ASR、T‑ASR、R‑ASR、调用次数、令牌消耗、延迟比例和资源放大因子。实验显示注入成功率普遍高于 80%，单节点可实现 5–51 倍调用放大，10‑20 倍令牌消耗；多节点协同使 p95 延迟可升至 229 倍。ACE 检测率达 93.5%，无误报，且能完全阻止攻击。

**⚠️ 局限性**

局限性：① 需依赖代理能够将外部文本转化为组件修改，某些模型/框架可能无此路径；② 目标定位会降低成功率（“targeting tax”），并受模型自省能力限制；③ ACE 在检测已存在组件内的细粒度闭包时可能失效；④ 评估仅在本地隔离环境进行，未覆盖真实多租户云服务的细粒度网络与系统细节。

---

## 104. Minimization of Streaming Transducers

**arXiv ID:** 2605.11190 | [PDF](https://arxiv.org/pdf/2605.11190v1)

**作者:** Christian Bianchini `[一作]` (Università degli Studi di Udine), Gabriele Puppis `[通讯]` (Università degli Studi di Udine)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了适用于广义流式转换器（streaming transducer）的最小化理论，给出存在最小模型的充分必要条件，并在此框架下实现了对字符串到树（string‑to‑term）转换器的有效最小化；

**💡 创新点**

核心创新在于将最小化问题抽象为范畴理论中的子对象-商对象分解（subquotient）并利用更新（updates）的可合成性与最大公约数（GCD）结构，首次将反向归约与前向抽象结合，形成一种统一的最小化方法；

**🔧 技术方法**

使用范畴理论的因子化系统、闭包算子、以及术语代数中的统一与反统一（unification/anti‑unification）技术，构造初始/最终转化器并证明其最小性；

**📊 数据集**

无具体实验数据集，研究完全基于理论证明与算法构造；

**📈 对比分析**

比较方法主要是与已知的 Choffrut 等顺序转换器最小化定理对齐；在理论层面展示了与资源最小化（状态数、寄存器数）的一致性与局限性，未给出数值性能评估；

**⚠️ 局限性**

局限性包括：(1) 需要假设数据结构满足“闭包”与“GCD”两大条件，在某些数据结构（如复制型更新）不成立；(2) 对于非复制、非擦除约束下的自由项式代数，GCD 的存在性尚未普适；(3) 算法复杂度与实现细节未在本文中系统分析。

---

## 105. PoseBridge: Bridging the Skeletonization Gap for Zero-Shot Skeleton-Based Action Recognition

**arXiv ID:** 2605.11497 | [PDF](https://arxiv.org/pdf/2605.11497v1)

**作者:** Sanghyeon Lee `[一作]` (Kyungpook National University), Jong Taek Lee `[通讯]` (Kyungpook National University)

**通讯引用:** 1288 | [OpenAlex ID](https://openalex.org/A5000134514)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 PoseBridge 框架，将人类姿态估计（HPE）过程中的中间特征提炼为 pose-anchored semantics，并将其桥接到骨架-文本对齐中，以提升零样本骨架动作识别（ZSSAR）性能。

**💡 创新点**

创新点在于：① 在骨架化阶段保留并利用 HPE 的多层特征，避免传统骨架化-对齐（S2A）导致的上游语义丢失；② 通过骨架条件语义桥（跨注意力）和语义原型适配，将 pose-anchored semantics 与文本原型对齐，构建更语义一致的共享空间。

**🔧 技术方法**

技术手段包括：多层 HPE 特征递归融合与身体感知池化、CLIP 风格的语义对比损失、骨架条件跨注意力桥接、原型残差适配、监督对比学习、知识蒸馏与 GZSL 校准。

**📊 数据集**

实验数据集涵盖 NTU-RGB+D 60/120、PKU-MMD、Kinetics-200/400（PURLS）等三大主流骨架与大规模视频数据集。

**📈 对比分析**

与多种基线（含 RGB 辅助方法）在标准、随机与 PURLS 评价协议下进行对比，PoseBridge 在 ZSL、GZSL 及 Kinetics 任务中均实现显著提升，最难分割（48/12、96/24）提高 10.5–17.4 点，Kinetics-200/400 上提升 16.0+ 点，展现出强大的跨域与泛化能力。

**⚠️ 局限性**

局限性：依赖 HPE 过程，若 HPE 精度低会影响语义桥接效果；未在多摄像头、低帧率或极低资源环境下进行验证；当前实现仍需一定计算量，尚未针对极端实时或边缘设备进行优化。

---

## 106. JACoP: Joint Alignment for Compliant Multi-Agent Prediction

**arXiv ID:** 2605.11385 | [PDF](https://arxiv.org/pdf/2605.11385v1)

**作者:** Qingze Liu `[一作]` (Rutgers University), Mubbasir Kapadia `[通讯]` (College of New Jersey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种多阶段框架 JACoP，用于生成符合环境与社会约束的多智能体轨迹预测。

**💡 创新点**

创新点在于引入锚点式的基于代理的分析器与基于马尔科夫随机场的联合对齐，能够在样本空间内显式约束并采样可行轨迹。

**🔧 技术方法**

技术方案包括对轨迹进行 SVD 压缩+K 均值生成锚点、基于 Fourier 特征的历史编码、LSTM 解码，以及 MRF 能量模型和 Gibbs 采样。

**📊 数据集**

实验使用公开的 ETH‑UCY 轨迹数据集进行验证。

**📈 对比分析**

与 Agentformer、GP‑Graph、EqMotion、SingularTrajectory 等 SOTA 方法对比，JACoP 在 JADE/JFDE 以及环境/社交碰撞率上实现了最优或次优成绩，并将碰撞率降至接近零。

**⚠️ 局限性**

局限性主要是采样过程导致的计算开销增加，且在极度稠密场景下仍有轻微的预测误差提升。

---

## 107. RETUYT-INCO at BEA 2026 Shared Task 2: Meta-prompting in Rubric-based Scoring for German

**arXiv ID:** 2605.11242 | [PDF](https://arxiv.org/pdf/2605.11242v1)

**作者:** Ignacio Sastre `[一作]` (Universidad de la República), Santiago Góngora `[通讯]` (Universidad de la República)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5045174175)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在BEA 2026德国短答评分共享任务中，提出并实现了Meta‑Prompting、合成数据生成、角色扮演提示、提示调优与LoRA微调等多种方法，并提交了在三条不同轨道（三分类/二分类答案/二分类问题）的成绩。

**💡 创新点**

创新点在于为每个问题‑评分组自动生成专用评分提示，并结合合成样本与多提示变体来提升鲁棒性；同时首次在共享任务中系统比较多种提示与微调策略的效果。

**🔧 技术方法**

主要技术包括Gemini 3 Flash与Llama 3.1 8B LLM、Meta‑Prompting、提示调优、LoRA微调、SVM文本分类、合成数据生成（基于LangGraph/LangChain）、角色扮演提示。

**📊 数据集**

使用的数据集为BEA 2026共享任务的德语短答数据，包含72个问题‑评分组，划分为训练集、验证集（dev）和测试集。

**📈 对比分析**

在dev集上，Meta‑Prompting的最佳变体达QWK 0.743、加权F1 0.892；最终提交在Track1 QWK 0.729、Track3 QWK 0.674、Track4 QWK 0.49，整体排名中游（Track1第6/8，Track3第4/9，Track4第4/8）。

**⚠️ 局限性**

局限性包括高度依赖闭源API导致成本与延迟高、缺乏对合成数据质量的人工评估、在三分类任务中未做专门调优、在dev集上可能出现过拟合，且实验资源受限（仅一台Colab Pro）。

---

## 108. Distributed Pose Graph Optimization via Continuous Riemannian Dynamics

**arXiv ID:** 2605.11210 | [PDF](https://arxiv.org/pdf/2605.11210v1)

**作者:** Jaeho Shin `[一作]` (University of Michigan), Yulun Tian `[通讯]` (University of Michigan)

**通讯引用:** 748 | [OpenAlex ID](https://openalex.org/A5025546142)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于连续黎曼动力学的分布式姿势图优化框架。

**💡 创新点**

将姿势优化建模为在Lie群上进行的二阶连续动力学，并引入质量、阻尼以及邻居状态预测以实现加速收敛，尤其在异步通信下保持鲁棒性。

**🔧 技术方法**

采用欧拉‑庞加莱方程、几何半隐式积分、Levenberg–Marquardt 预处理、速度预测、能量耗散分析等技术。

**📊 数据集**

在公开 SLAM 基准（Sm. Grid、Sphere、Torus、Grid、Cubicle、Rim、Garage）以及模拟网格场景上进行评测。

**📈 对比分析**

与 AMM‑PGO、MESA、DJ、CORD 等分布式基线对比，实验表明在同步与异步条件下均能与或优于最先进方法的成本与收敛速度，异步时显著抑制振荡并加速收敛。

**⚠️ 局限性**

收敛分析仅针对同步、常量质量矩阵，未涵盖状态依赖质量与通信延迟的完整理论；对非 SE(3) 的广义适用性尚未完全证明。

---

## 109. ShardTensor: Domain Parallelism for Scientific Machine Learning

**arXiv ID:** 2605.11111 | [PDF](https://arxiv.org/pdf/2605.11111v1)

**作者:** Corey Adams `[一作]` (NVIDIA), Sanjay Choudhry `[通讯]` (NVIDIA)

**通讯引用:** 300 | [OpenAlex ID](https://openalex.org/A5051964203)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在科学机器学习中提出并实现了域并行（ShardTensor）框架，能够将高分辨率输入数据跨GPU进行分块处理，实现单设备内存瓶颈突破；

**💡 创新点**

创新点在于将 PyTorch 的分布式张量机制扩展到动态输入维度，增加了 sharding shapes 并在 imperative dispatch 下实现层级并行，兼容多种现有并行模式（数据、模型、Ring Attention 等），实现了高分辨率科学数据的灵活并行；

**🔧 技术方法**

核心技术包括 PyTorch 的 DTensor 与 Dispatch API、ShardTensor 的分布式张量实现、Halo 交换、自动化收敛统计、与 FSDP、DP 组合使用的集成；

**📊 数据集**

实验使用的主要数据集有：用于基准的合成 2D/3D 图像和向量序列；Transolver 在汽车气动实验数据集上（DrivaerML）训练；StormScope 在 GOES‑16 卫星影像与地面雷达数据（CONUS 3km 解析度）上训练；

**📈 对比分析**

与传统单卡或仅数据并行的实现相比，Ring Attention 在大序列长度下实现近线性强扩展，ViT 在 16 GPU 上可获得 15× 推理加速；Transolver 在 1M 网格点上保持稳定且准确的收敛；StormScope 在 3km 解析度下完成全大陆训练，单卡内存上限被突破，训练损失与 6km 单卡基线相当；

**⚠️ 局限性**

局限性包括：Python 级 dispatch 引入的额外延迟，尤其在小规模或低分辨率任务中可能抵消收益；某些 PyTorch 操作尚未实现域并行路径，需手动扩展；通信开销受互连带宽限制，当前不支持静态图优化（如核融合、层级通信重叠）等。

---

## 110. Freeze Deep, Train Shallow: Interpretable Layer Allocation for Continued Pre-Training

**arXiv ID:** 2605.11416 | [PDF](https://arxiv.org/pdf/2605.11416v1)

**作者:** Yu-Hang Wu `[一作]` (Nanhu Research Institute of China Electronic Science and Technology), Qing-Wei Cong `[通讯]` (Nanhu Research Institute of China Electronic Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LayerTracer 框架，用于诊断大型语言模型（LLM）在持续预训练中的层级冻结与训练策略，并给出可解释的层级分配建议。

**💡 创新点**

创新点在于结合任务粒子（Task Particle）与层级敏感度（Layer-wise Sensitivity）两种度量，揭示深层为执行区、浅层为敏感区的层级层次，并基于此提出浅层训练、深层冻结的实用策略。

**🔧 技术方法**

使用了任务粒子计算目标 token 概率相对变化、层级敏感度计算 Jensen‑Shannon 散度（JSD）变化、上下文掩码扰动等无梯度解释技术。

**📊 数据集**

主要使用 AntSynNET 结构化样本以及中文高质量语料 CCI3.0‑HQ 进行持续预训练，评估基准为 C‑Eval 与 CMMLU。

**📈 对比分析**

通过对比全参数微调、冻结浅层/训练深层、训练浅层/冻结深层三种策略，实验表明训练浅层/冻结深层在两大基准上均优于全参数并比相反策略提升约 1.6–3.2%；在混合架构实验中，深层放置预训练块亦提升 9–10%。

**⚠️ 局限性**

局限性主要在于实验规模有限，验证集中在 0.6B‑14B 参数的中等规模模型，缺乏对更大参数规模与多任务泛化的进一步验证。

---

## 111. The DAWN of World-Action Interactive Models

**arXiv ID:** 2605.11550 | [PDF](https://arxiv.org/pdf/2605.11550v1)

**作者:** Hongbo Lu `[一作]` (COWARobot Co. Ltd), Pai Peng `[通讯]` (COWARobot Co. Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种世界-动作交互模型（WAIM），通过短期潜在世界滚动与动作去噪的递归互动，在推理时实现世界状态与行动的相互约束；

**💡 创新点**

核心创新是将未来世界与行动视为共生推理变量，打破传统WAM中单向或并行生成的结构，利用递归互动提升决策相关未来的真实性与安全性；

**🔧 技术方法**

技术主要包括V-JEPA 2大型视觉编码器、Auto‑Encoder Resampler压缩潜在世界，因果Transformer进行潜在滚动，DiT风格的动作去噪器，联合训练的多阶段策略；

**📊 数据集**

使用了大规模驾驶视频数据集（OpenScene、DrivingDojo、CoVLA）进行预训练，并在nuScenes、NAVSIM v1等标准自动驾驶基准上进行微调与评测；

**📈 对比分析**

与多种基准方法（Transfuser、Hydra‑MDP、DiffusionDrive、Drive‑JEPA等）对比，DAWN在NAVSIM的PDMS达到89.1（最高），在nuScenes的平均L2误差0.33 m、碰撞率0.11 %，均为最优或接近最优；

**⚠️ 局限性**

局限性包括对潜在分辨率与推理步数的敏感，较长的世界滚动会显著增加延迟；模型仍依赖大量预训练数据，且对极端交互场景的泛化尚待验证；

---

## 112. CORE: Cyclic Orthotope Relation Embedding for Knowledge Graph Completion

**arXiv ID:** 2605.11159 | [PDF](https://arxiv.org/pdf/2605.11159v1)

**作者:** Yingqi Zeng `[一作]`, Huiling Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 CORE 模型，在无边界的环面（torus）空间中将知识图谱关系表示为循环直角坐标体（cyclic orthotope），并通过动态实体表示和自适应宽度正则化实现知识图谱补全。

**💡 创新点**

创新点在于：1）在环面上构造无边界关系区域，解决欧氏空间边界截断导致的梯度饱和问题；2）采用动态实体偏移（bump）提升局部语义适应性；3）引入宽度正则化防止关系区域无限膨胀，从而显著提升模型的表达力与泛化能力。

**🔧 技术方法**

技术实现包括：环面嵌入与周期模数映射、循环直角坐标体的几何定义、动态实体偏移机制、宽度正则化、基于距离的分数函数、以及自对抗负样本采样和 margin‑based 损失。

**📊 数据集**

实验数据集为 FB15k、FB15k‑237、WN18 和 WN18RR 四大主流基准。

**📈 对比分析**

与 TransE、RotatE、DualE、HAKE、TorusE、BoxE、ExpressivE 等多种基线进行 Hits@1/3/10 与 MRR 的比较，CORE 在所有四个数据集上均取得最优或接近最优的表现，尤其在 WN18、WN18RR 以及 FB15k‑237 上的提升尤为显著。

**⚠️ 局限性**

局限性包括：对宽度正则化系数 λ 极度敏感，密集图结构下过强正则会导致关系区域过度收缩并影响性能；此外，环面空间与动态偏移的实现复杂度较高，模型在极大规模知识图谱上的可扩展性尚待验证。

---

## 113. StoicLLM: Preference Optimization for Philosophical Alignment in Small Language Models

**arXiv ID:** 2605.11483 | [PDF](https://arxiv.org/pdf/2605.11483v1)

**作者:** Ishmam Khan `[一作]` (Tufts University), Shuo Zhang `[通讯]` (Tufts University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

微调小型LLM在极少量斯多葛哲学文本上实现哲学对齐，并用多模型评审评估其表现。

**💡 创新点**

证明仅用300条高质量例子即可在小模型中注入复杂人格，并比较两种偏好优化器对不同基模型的模型依赖性。

**🔧 技术方法**

采用DoRA微调、ORPO与AlphaPO偏好优化、WSD调度、StableAdamW以及LLM-as-a-judge评估框架。

**📊 数据集**

使用Seneca《致卢西利乌斯的伦理书信》和Epictetus《全集》译本生成的V100/V200/V300微数据集。

**📈 对比分析**

通过多模型评审对100道开放式问题进行评分，比较几-shot、微调模型与基线，AlphaPO在Qwen3上超越ORPO，性能接近few-shot上限。

**⚠️ 局限性**

评价完全基于LLM生成的测试与评审，缺乏人类专家判定，且小模型无法学习斯多葛外向的社会义务。

---

## 114. SpatialForge: Bootstrapping 3D-Aware Spatial Reasoning from Open-World 2D Images

**arXiv ID:** 2605.11462 | [PDF](https://arxiv.org/pdf/2605.11462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 115. FeatMap: Understanding image manipulation in the feature space and its implications for feature space geometry

**arXiv ID:** 2605.11203 | [PDF](https://arxiv.org/pdf/2605.11203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. Seeing the Needle in the Haystack: Towards Weakly-Supervised Log Instance Anomaly Localization via Counterfactual Perturbation

**arXiv ID:** 2605.10988 | [PDF](https://arxiv.org/pdf/2605.10988v1)

**作者:** Yutszyuk Wong `[一作]` (Jinan University), Weiwei Lin `[通讯]` (South China University of Technology)

**通讯引用:** 5891 | [OpenAlex ID](https://openalex.org/A5007440245)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种弱监督多实例学习框架 LogMILP，用原始日志袋标签实现日志异常检测与实例级定位。

**💡 创新点**

创新点在于将可学习原型引导与关键实例的反事实扰动一致性正则相结合，提升定位可靠性与可解释性。

**🔧 技术方法**

采用原型学习、Transformer编码、多头注意力以及扰动一致性损失等技术。

**📊 数据集**

在 BGL、Spirit 和 ZooKeeper 三个公开日志数据集上进行评估。

**📈 对比分析**

与 DeepLog、LogAnomaly、LogBERT、LogFormer、MIDLog 等基线相比，在袋级 F1 上保持竞争力，同时在实例级 Loc@3 和 SR 指标上显著优于对手。

**⚠️ 局限性**

局限在于对噪声数据鲁棒性未充分验证，且未实现在线原型更新与跨域泛化。

---

## 117. RankGuardPolar Private Public Finite Length Polar Codes with Rank-Certified Leakage

**arXiv ID:** 2605.11356 | [PDF](https://arxiv.org/pdf/2605.11356v1)

**作者:** Hassan Tavakoli `[一作]` (Oregon State University), Bella Bose `[通讯]` (Oregon State University)

**通讯引用:** 1092 | [OpenAlex ID](https://openalex.org/A5108277072)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了RankGuard-Polar框架，能够在有限长度极化码中安全公开部分码字坐标并对泄露信息量进行精确评估。

**💡 创新点**

创新点在于给出了泄漏量与极化码生成矩阵子块秩差的等价性，并基于此构造了线性提取器与高效的秩证明算法，实现了对任意公开索引集合泄露量的精确计算与校验。

**🔧 技术方法**

使用的技术包括二元线性代数、极化码的Kronecker结构、秩与图像空间的关系以及贪心ScoreGreedy算法用于快速选取低泄漏索引集合。

**📊 数据集**

本文未使用外部数据集，而是在理论上对极化码的生成矩阵进行符号计算与实验验证，示例中以N=4、N=8等小规模极化码进行说明。

**📈 对比分析**

与传统全密钥隐藏或无公开码字设计相比，RankGuard-Polar在保持相同可靠性的前提下可显著降低因公开码字导致的信息泄漏，并且提供了可在实际系统中直接使用的可证明泄漏上界，实验结果表明贪心算法仅在 O(N²) 时间内即可获得与精确搜索相近的泄漏值。

**⚠️ 局限性**

局限性包括：需预先共享并保持冻结位的秘密随机性；对极大块长时仍需进一步优化秩计算速度；且仅针对公开码字的输入泄漏模型，未考虑通道噪声或对称性对泄漏的进一步影响。

---

## 118. Localization Boosting for Growth Markets: Mitigating Cross-Locale Behavioral Bias in Learning-to-Rank

**arXiv ID:** 2605.11272 | [PDF](https://arxiv.org/pdf/2605.11272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 119. CheXTemporal: A Dataset for Temporally-Grounded Reasoning in Chest Radiography

**arXiv ID:** 2605.11304 | [PDF](https://arxiv.org/pdf/2605.11304v1)

**作者:** Eva Prakash `[一作]` (Stanford University), Curtis Langlotz `[通讯]` (Stanford University)

**通讯引用:** 21183 | [OpenAlex ID](https://openalex.org/A5087710258)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了CheXTemporal数据集并在其上评估多种视觉‑语言模型的时空推理能力

**💡 创新点**

首次提供对称标注的五类进展标签、病灶级空间定位与跨源对齐，形成全新时空推理基准

**🔧 技术方法**

采用Transformer视觉编码器、对比学习与跨模态对齐、时间编码等技术，并评估BioViL、BioViL‑T、TILA、TempA‑VLP、ALTA、BiomedCLIP等模型

**📊 数据集**

使用手工标注的金标数据（197例、1,787 pair‑finding）和自动生成的银标数据（34,296例、282,214 pair‑finding），来源于CheXpert、MIMIC‑CXR和ReXGradient

**📈 对比分析**

在零样本下对定位（CNR/PG）和5‑分类进展识别进行评估，最佳模型BioViL‑T在定位CNR≈0.79、PG≈0.31，进展分类准确率仅≈27%，对“稳定”“已恢复”等细粒度类别表现尤差

**⚠️ 局限性**

局限包括金标规模有限、注释单一放射科医生、定位采用粗略框而非精细分割、仅针对胸片、模型对细粒度时序推理与跨域泛化能力不足

---

## 120. Overcoming Dynamics-Blindness: Training-Free Pace-and-Path Correction for VLA Models

**arXiv ID:** 2605.11459 | [PDF](https://arxiv.org/pdf/2605.11459v1)

**作者:** Yanyan Zhang `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**通讯引用:** 4068 | [OpenAlex ID](https://openalex.org/A5004523290)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为 Vision‑Language‑Action（VLA）模型提供一种无需训练、在推理时即可补偿环境动力学的闭式包装器——Pace-and-Path Correction（PPC），并用专门的 MoveBench 基准评估其在不同运动模式下的鲁棒性。

**💡 创新点**

创新点在于：①将单一二次成本函数解析为可分离的“节奏”（Pace）和“路径”（Path）两路闭式校正；②引入层级 2‑EMA 触发器检测并限制非平稳运动时的执行长度；③完全不依赖学习参数、无后端改造，兼容任何已发布的 VLA。

**🔧 技术方法**

技术手段包括：闭式二次优化、正交分解、黄金分割（Fibonacci）序列路径补偿、EMA 滞后稳态检测、动态速度/加速度外部感知。

**📊 数据集**

使用 MoveBench——一个将运动模式（静止、均匀平移、加速、随机步行、停走、瞬移）作为唯一可变因素的基准，共 10,000 条轨迹、约 460k 帧。

**📈 对比分析**

与 8 类基线（基础 VLA、训练无关包装器、动态自适应方法）比较，PPC 在动态场景下提升成功率 28.8%（仅动态）/25.9%（混合），并在所有任务、模型上优于其他方法，尤其在加速运动上优势最大。

**⚠️ 局限性**

局限性：需要外部速度/加速度感知信号，且假设在每个块内扰动近似平稳；对极度不规则或多物体动态场景的适应性尚待验证；在高度噪声的跟踪环境中，性能相对基线有下降。

---

## 121. Offline Policy Evaluation for Manipulation Policies via Discounted Liveness Formulation

**arXiv ID:** 2605.11479 | [PDF](https://arxiv.org/pdf/2605.11479v1)

**作者:** Hao Wang `[一作]` (University of Southern California), Somil Bansal `[通讯]` (Stanford University)

**通讯引用:** 878 | [OpenAlex ID](https://openalex.org/A5059176959)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种离线策略评估框架，将稀疏奖励的机械操作任务视为折扣生命周期问题，并学习表示任务进度的值函数。

**💡 创新点**

创新点在于将评估转化为寿命(liveness)问题，构造保守的折扣贝尔曼算子，并通过两阶段引导式自举（bootstrapping）显著减少因手动超时导致的截断偏差。

**🔧 技术方法**

采用折扣寿命贝尔曼算子、两阶段自举更新、神经网络值函数拟合（输入为SigLIP2图像/关节嵌入），并使用优先经验回放等强化学习技术。

**📊 数据集**

使用三类数据集：LIBERO‑Spatial pick‑and‑place（200/100 试验），Robomimic Square peg‑hole（200/100 试验），以及Franka Panda 硬件布折叠（150/20 试验）。

**📈 对比分析**

与 TD(0)、Monte Carlo、分布式 Monte Carlo 等基线对比，实验表明在成功率和综合指标上均优于基线，失败率略有下降，但整体性能显著提升并具统计显著性。

**⚠️ 局限性**

主要限制包括：自举可能导致过度乐观，取决于图像/潜在空间的相似度；对训练分布外的数据不稳健；当策略缺乏恢复行为时，方法效果下降。

---

## 122. Unlocking LLM Creativity in Science through Analogical Reasoning

**arXiv ID:** 2605.11258 | [PDF](https://arxiv.org/pdf/2605.11258v1)

**作者:** Andrew Shen `[一作]` (Stanford University), James Zou `[通讯]` (Stanford University)

**通讯引用:** 40530 | [OpenAlex ID](https://openalex.org/A5005779176)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究提出并验证了一种基于类比推理的开放式方案生成方法，显著提升AI在生物医学研究中的多样性与创新性。

**💡 创新点**

创新点在于将结构映射理论与LLM结合，先生成跨域类比再搜索方案，解决模式坍塌问题。

**🔧 技术方法**

采用大语言模型（Claude Sonnet 4.5、GPT-5.2、Gemini 3 Flash）进行类比抽取与方案搜索，并用Vendi Score、LLM评估等指标衡量多样性与新颖性。

**📊 数据集**

使用了 PerturBench、OpenProblems ligand‑target、脑区互作数据以及 OligoGym 等生物医学数据集。

**📈 对比分析**

与无域、跨域基线对比，AR 在生成多样性提升 90–173%，新颖率超过 50%，在四个案例中实现近 13 倍的分布式度量提升、AUPRC 领先、Spearman ρ=0.729 以及两项 SOTA 表现。

**⚠️ 局限性**

限制在于并非所有类比都能落地实现，当前管线仅生成方案，缺乏完整执行评估；且实验聚焦生物医学，跨学科扩展仍待验证。

---

## 123. 3DGS$^3$: Joint Super Sampling and Frame Interpolation for Real-Time Large-Scale 3DGS Rendering

**arXiv ID:** 2605.11489 | [PDF](https://arxiv.org/pdf/2605.11489v1)

**作者:** Yibo Zhao `[一作]` (University of Science and Technology of China), Ligang Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8890 | [OpenAlex ID](https://openalex.org/A5100635702)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了3DGS^3框架，利用后处理实现高分辨率和高帧率的3D Gaussian Splatting渲染，结合梯度感知的超采样和轻量级时序帧插值。

**💡 创新点**

创新点包括：1) 通过3DGS的连续可微性质提取图像梯度，驱动GRU网络实现精细超采样；2) 设计轻量级U-Net结构与时间递归单元融合时空梯度信息，生成中间帧；3) 以后处理方式兼容多种3DGS加速技术。

**🔧 技术方法**

核心技术包括：梯度感知插值（Hermite插值 + GRU细化）、轻量级时间帧插值（前后投影、特征融合、GRU递归）、自定义CUDA核实现梯度与运动矢量计算、TensorRT推理加速。

**📊 数据集**

使用公开数据集Mip-NeRF 360、Tanks and Temples、Deep Blending进行评估，并在这些数据集上对比传统插值、学习型超采样、视频帧插值及多种3DGS加速方法。

**📈 对比分析**

与基线方法对比，3DGS^3在PSNR/SSIM维持或略高的同时，帧率显著提升（如Deep Blending 4×超采样达到226 FPS），并在与3DGS加速技术整合后进一步提升渲染效率。

**⚠️ 局限性**

局限性在于仅针对静态场景，动态3DGS的适配尚未实现；网络模型规模相对较大，移动端部署仍需压缩和优化。

---

## 124. Attributing Emergence in Million-Agent Systems

**arXiv ID:** 2605.11404 | [PDF](https://arxiv.org/pdf/2605.11404v1)

**作者:** Ling Tang `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongrui Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5020653216)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型LLM驱动多智能体系统中宏观现象归因问题，并提出了能在百万级规模下满足四个Aumann–Shapley公理的路径积分归因方法。

**💡 创新点**

在百万级规模上实现全局归因，克服传统Shapley方法指数级成本；证明非线性宏观指标下小样本偏置不可通过全局缩放修正；通过对Bluesky数据的跨尺度实验揭示了采样偏差对归因结果的结构性影响。

**🔧 技术方法**

Aumann–Shapley路径积分归因、解析积分化简、基于梯度的数值积分、可变K步量化，以及对四种宏观指标（线性、热度、方差、Gini）的解析公式。

**📊 数据集**

2026年4月8日至21日公开的Bluesky AT-Protocol全量数据，包含167万活跃用户的关注、发帖和回复特征。

**📈 对比分析**

与采样Shapley、Leave-One-Out、Exact Shapley等传统归因方法在同一数据上对比；在百万规模下本方法仅需约9毫秒完成归因，而采样Shapley需数百秒，速度提升三到五个数量级，且在可比情形下归因排名一致。

**⚠️ 局限性**

归因偏差的定量大小受样本结构影响；该方法对非线性指标不可通过后处理修正；理论证明为结构性而非定量，无法给出误差上界；适用范围局限于可微、对称的宏观指标。

---

## 125. Conversational Customization of Productivity Systems: A Design Probe of Malleable AI Interfaces

**arXiv ID:** 2605.11149 | [PDF](https://arxiv.org/pdf/2605.11149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 126. A Mechanistic Investigation of Supervised Fine Tuning

**arXiv ID:** 2605.11426 | [PDF](https://arxiv.org/pdf/2605.11426v1)

**作者:** Ruhaan Chopra `[一作]` `[通讯]`, Ruhaan Chopra

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究监督式微调对Gemma 3 1B模型内部表示的影响，使用预训练稀疏自编码器对激活空间进行高分辨率诊断。

**💡 创新点**

提出将固定稀疏自编码器作为诊断基准，揭示原始激活几乎不变但稀疏潜变量剧烈漂移，并识别出任务与层级特异的子空间重写模式。

**🔧 技术方法**

使用稀疏自编码器（SAE）、SVD主成分分析、特征翻转率检测、Neuronpedia API和LLM-as-judge分类技术。

**📊 数据集**

在Gemma 3 1B IT模型上微调四个任务：MultiNLI、GSM8K、WildJailbreak（安全对齐）和OpenAI Tool Calling。

**📈 对比分析**

通过对比原始激活余弦相似度与SAE潜变量相似度以及主成分方向与SAE特征对齐，量化不同层级的漂移；结果显示原始激活保持>0.96相似度，而潜变量在深层下降至约0.55，说明SFT并非完全外科微调，且安全对齐在浅层影响最大。

**⚠️ 局限性**

仅在GemmaScope 2和Gemma 3 1B规模上验证，未检验其他架构、规模或微调方案；依赖预训练SAE的质量与层覆盖范围。

---

## 127. Comment and Control: Hijacking Agentic Workflows via Context-Grounded Evolution

**arXiv ID:** 2605.11229 | [PDF](https://arxiv.org/pdf/2605.11229v1)

**作者:** Neil Fendley `[一作]` (Johns Hopkins University), Yinzhi Cao `[通讯]` (Johns Hopkins University)

**通讯引用:** 3715 | [OpenAlex ID](https://openalex.org/A5070605476)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并实现了一个用于检测和利用自动化平台（如GitHub Actions和n8n）中Agentic工作流被劫持漏洞的完整框架，揭示了数千个可被攻击的工作流实例。

**💡 创新点**

创新点在于首次将路径可行性分析、动态提示溯源、运行时能力分析三种混合程序分析方法结合，形成Context‑Grounded Evolution技术，能够在工作流上下文中演化有效的攻击payload。

**🔧 技术方法**

使用了静态路径敏感分析、动态taint跟踪、提示来源分析、能力分析、Z3求解器、CodeQL、Python/JavaScript实现，以及Claude Sonnet 4.5等LLM模型作为变异器。

**📊 数据集**

评估数据集包括8524个GitHub Actions工作流和8个n8n模板，最终发现4174个可劫持的GitHub工作流和8个可劫持的n8n模板，覆盖15个广泛使用的GitHub Action和2个官方n8n节点。

**📈 对比分析**

与现有工作流漏洞检测工具（如ARGUS）和LLM jailbreak基线相比，本框架在检测覆盖率和payload成功率上显著提升，平均分析时间约为X秒（未给出具体数值），整体性能优于传统方法。

**⚠️ 局限性**

局限性包括：仅针对GitHub Actions和n8n；对更复杂的多步骤或跨平台工作流的分析仍有限；需要平台特定的Instrumentation；依赖LLM模型的可用性和安全策略；未覆盖所有潜在触发器和输出通道。

---

## 128. MambaNetBurst: Direct Byte-level Network Traffic Classification without Tokenization or Pretraining

**arXiv ID:** 2605.11034 | [PDF](https://arxiv.org/pdf/2605.11034v1)

**作者:** Gayan K. Kulatilleke `[一作]` (University of Queensland), Marius Portmann `[通讯]` (University of Queensland)

**通讯引用:** 4752 | [OpenAlex ID](https://openalex.org/A5078468070)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于Mamba‑2的无词典、无预训练的字节级网络流分类模型OURMODEL，直接用原始包字节进行端到端监督学习。

**💡 创新点**

创新点在于：①完全消除分词/分块/下采样与多模态特征工程；②在没有任何自监督预训练的情况下，利用线性时间的SSM实现高精度；③Mamba‑2受限的状态转移结构不仅足够，还能起到正则化作用。

**🔧 技术方法**

技术包括：学习型字节嵌入+轻量MLP投影、可学习CLS标记和位置编码、堆叠4层Mamba‑2残差块、交叉熵分类头；训练使用AdamW、混合精度、cosine学习率调度。

**📊 数据集**

使用六个公开基准：CrossPlatform(Android/iOS)、ISCXVPN2016、ISCXTor2016、USTC‑TFC2016、CICIoT2022，覆盖加密应用识别、VPN/Tor识别、恶意软件流和IoT攻击流。

**📈 对比分析**

与传统基于特征、预训练Transformer（ET‑BERT、YaTC等）和NetMamba等对比，OURMODEL在大多数数据集上取得与或优于现有最佳结果，宏F1平均≈0.991；同时模型参数仅约250‑270万，推理速度和显存比大多数预训练模型快30‑60%。

**⚠️ 局限性**

局限性包括：仅处理固定长度的流突发（5包×320字节），不支持可变长长流；对极低资源设备的实时在线推理仍需进一步优化；在跨域迁移或概念漂移场景下的鲁棒性尚未深入评估。

---

## 129. CPEMH: An Agentic Framework for Prompt-Driven Behavior Evaluation and Assurance in Foundation-Model Systems for Mental Health Screening

**arXiv ID:** 2605.11341 | [PDF](https://arxiv.org/pdf/2605.11341v1)

**作者:** Giuliano Lorenzoni `[一作]` (University of Waterloo), Donald Cowan `[通讯]` (University of Waterloo)

**通讯引用:** 24441 | [OpenAlex ID](https://openalex.org/A5081821121)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了CPEMH多代理框架，用于在转录文本基础的精神健康筛查系统中自动化评估与保证prompt驱动的行为稳定性、可解释性与可复现性。

**💡 创新点**

创新点在于将行为保证概念融入prompt设计评估循环，提出基于偏差（Bias）、鲁棒性（Robustness）等行为指标的多代理评估流程，并通过自动化prompt生成、推理与评估实现可追溯、可审核的prompt选择。

**🔧 技术方法**

主要技术包括多代理（Orchestrator、Inference、Evaluation等）架构、LLM推理、结构化prompt策略（DI、RP、CoT等）、偏差与鲁棒性度量、自动化工作流与指标驱动的决策启发式。

**📊 数据集**

使用DAIC‑WOZ数据集（189条临床访谈转录，包含抑郁标签），划分为小样本的In‑Sample与四倍大的Out‑of‑Sample子集。

**📈 对比分析**

在28个prompt配置上计算Accuracy、Precision、Recall、F1、Bias（|P‑R|）与Robustness（σ_F1），采用F1优先且Bias/σ满足阈值的启发式挑选Prompt；在OOS验证中，macro‑F1≈0.57，ΔF1<0.03，表明性能与行为稳定性良好。

**⚠️ 局限性**

局限性包括：仅在抑郁筛查任务验证，泛化至其他任务需进一步测试；偏差与鲁棒性阈值需手工设定；受LLM版本与资源限制；未深入探讨数据本身的偏差来源与长期漂移问题。

---

## 130. Three Regimes of Context-Parametric Conflict: A Predictive Framework and Empirical Validation

**arXiv ID:** 2605.11574 | [PDF](https://arxiv.org/pdf/2605.11574v1)

**作者:** Pruthvinath Jeripity Venkata `[一作]` `[通讯]` (Independent Researcher), Pruthvinath Jeripity Venkata (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并验证了三种处理冲突的体制（Regime 1、2、3），系统评估大型语言模型在不同冲突情景下对对立信息的依赖与抵抗。

**💡 创新点**

创新点在于通过区分单源、竞争整合与任务指令三类情境，揭示证据一致性、参数置信度和任务要求是各自的主要预测因子，并证明参数强度与唯一性为正交维度。

**🔧 技术方法**

采用多模型API实验（Claude Sonnet 4.6、GPT‑5.5、Gemini 2.5 Flash、Llama 4 Maverick、DeepSeek V3）并使用GEE回归、logistic回归以及多项式检验等统计方法。

**📊 数据集**

主要使用PopQA事实问答数据，结合Wikidata编辑频率和维基百科页面浏览量等外部指标，并按置信度层级进行分层。

**📈 对比分析**

通过对比单源与多源、单回合与多回合、任务指令三类实验，发现任务框定（Regime 3）可将上下文跟随率从近100%骤降至6–71%，并验证置信度梯度在所有五种模型上均显著。

**⚠️ 局限性**

局限性包括样本偏重PopQA创意作品属性、缺乏对动态事实域的验证、模型自回应与外部提示的差异以及参数唯一性仅以实体级编辑频率衡量，导致对更广泛领域的泛化受限。

---

## 131. LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models

**arXiv ID:** 2605.11011 | [PDF](https://arxiv.org/pdf/2605.11011v1)

**作者:** Taekhyun Park `[一作]`, Hyerim Bae `[通讯]` (Pusan National University)

**通讯引用:** 1628 | [OpenAlex ID](https://openalex.org/A5047158713)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将预训练的LLM通过编码器-推理-解码器分解，并利用循环深度放缩实现隐层递归推理；

**💡 创新点**

核心创新在于：①基于隐藏层表示动态的分块拆解；②输入可调的衰减门缓解隐藏状态漂移；③随机深度监督减轻BPTT内存压力；④置信度头实现自适应早停；

**🔧 技术方法**

使用的技术包括：Transformer分块拆解、Mamba式衰减门、随机深度监督与稀疏梯度、Monotonicity正则、置信度早停头；

**📊 数据集**

使用FineWeb‑Edu作为微调数据，评测数据集包括WikiText、LAMBADA、MMLU、HellaSwag、ARC‑E、ARC‑C、PIQA、WinoGrande、OpenBookQA；

**📈 对比分析**

与原始预训练模型及先前循环化方法对比，LoopUS在多尺度模型上平均提升1.6–2.2点准确率、显著降低WikiText/LAMBADA困惑度；在TinyLlama低训练量下实现+6.3的平均提升；

**⚠️ 局限性**

局限性包括：对中层表示的依赖限制了适用范围，知识检索型任务提升有限；缺乏严格的收敛或收缩理论保证；需要额外的超参调优以获得最佳门控与深度监督效果。

---

## 132. Leveraging Non-Equilibrium ECRAM Dynamics for Short-Term Plasticity in Neuromorphic Circuits

**arXiv ID:** 2605.11243 | [PDF](https://arxiv.org/pdf/2605.11243v1)

**作者:** Alex Currie `[一作]` (Rochester Institute of Technology), Tejasvi Das `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5113274478)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种基于ECRAM可挥发动态的延迟反馈LIF神经元架构，实现短时可塑性(STP)并与长时记忆共存。

**💡 创新点**

将ECRAM的电荷双层瞬态变化从无用噪声转变为STP的硬件原语，跨层面实现突触促进与神经元可塑性，无额外电路负担。

**🔧 技术方法**

采用3端电离门MOS器件、物理基础的ECRAM模型、Verilog‑A仿真、55nm CMOS实现的延迟反馈LIF电路及snnTorch网络仿真。

**📊 数据集**

使用自制MoS2/PEO:LiClO4 ECRAM样品的实验数据进行模型验证，并在网络层采用合成脉冲序列进行测试。

**📈 对比分析**

通过仿真对比不同STP配置下的能耗(≈2pJ/脉冲)和阈值抑制效果，证明在毫秒/微秒尺度下实现高能效、可调频率滤波。

**⚠️ 局限性**

局限在于设备速率仍受离子扩散限制、晶圆工艺一致性、变异对时序的影响，以及对更大规模网络的可扩展性待验证。

---

## 133. Improving Hybrid Human-AI Tutoring by Differentiating Human Tutor Roles Based on Student Needs

**arXiv ID:** 2605.11155 | [PDF](https://arxiv.org/pdf/2605.11155v1)

**作者:** Ashish Gurung `[一作]` (Carnegie Mellon University), Kenneth R. Koedinger `[通讯]` (Carnegie Mellon University)

**通讯引用:** 26982 | [OpenAlex ID](https://openalex.org/A5062550465)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种根据学生先前成绩差异化人类与 AI 辅导的策略，低分学生由教师主动介入，高分学生则按需回应。

**💡 创新点**

创新点在于将人类辅导按学生表现划分为主动与被动两种角色，并通过差分中断（DiDC）设计评估其对学习成效的影响。

**🔧 技术方法**

采用差分中断（DiDC）、差分中的差异（DiD）、回归不连续（RD）、多层混合效应模型、Imbens‑Kalyanaraman bandwidth 以及其他统计方法。

**📊 数据集**

使用635名5-8年级学生的 IXL 使用日志、MAP 测评分数及前年州立考试成绩等数据集。

**📈 对比分析**

对比 AI‑仅辅导（秋季）与混合人类‑AI 辅导（春季），以及主动组与被动组；结果显示人类‑AI 辅导使时间投入 +25%，技能熟练度 +36%，MAP 增长 +61%，主动组对低分学生的提升更显著。

**⚠️ 局限性**

局限在于非随机对照、依赖全国成长基准、二元分配可能过度简化、缺乏 Zoom 互动细节、未实现动态需求分配。

---

## 134. Breaking $\textit{Winner-Takes-All}$: Cooperative Policy Optimization Improves Diverse LLM Reasoning

**arXiv ID:** 2605.11461 | [PDF](https://arxiv.org/pdf/2605.11461v1)

**作者:** Haoxuan Chen `[一作]` (Sun Yat-sen University), Jian-Fang Hu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1938 | [OpenAlex ID](https://openalex.org/A5102336058)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GCPO——一种将LLM推理过程中的多次rollout视为协作团队、通过团队覆盖度奖励取代单个rollout评分的强化学习优化框架，显著提升推理准确率和多样性。

**💡 创新点**

创新点在于把奖励从个人分数转为团队覆盖度，用判别点过程行列式体积衡量多样性，并通过Shapley值公平分配团队价值，保持奖励总额不变，从而引导模型走向非冗余正确推理路径。

**🔧 技术方法**

核心技术包括判别点过程（DPP）行列式体积、Shapley值信用分配、奖励重分配、GRPO基础、冻结语义编码器（MiniLM-L6-v2）以及PPO式截断优化。

**📊 数据集**

实验使用了DAPO-Math-17K数据集作为训练集，评估基准包括AIME2024/25、MATH500、Minerva Math、Olympiad、ARC-Challenge、GPQA和MMLU-Pro等多域推理数据集。

**📈 对比分析**

与GRPO、DAPO、DIVER、DQO等RLVR与多样性方法对比，GCPO在所有内/外域任务上均实现了Pass@1和多样性指标的显著提升，尤其在大模型和高难度题目上提升10+分。

**⚠️ 局限性**

局限性主要在于DPP行列式计算和Shapley枚举的计算开销，尤其在大批量rollout时成本上升，同时目前仅针对可验证推理任务，尚需探索更高效近似与开放式生成场景的扩展。

---

## 135. Exploring Token-Space Manipulation in Latent Audio Tokenizers

**arXiv ID:** 2605.11192 | [PDF](https://arxiv.org/pdf/2605.11192v1)

**作者:** Francesco Paissan `[一作]` (Mila Quebec AI Institute), Cem Subakan `[通讯]` (Mila Quebec AI Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出LATTE，一种将整句语音压缩为少量非时间对齐的离散latent token的音频tokenizer，并利用token交换实现无监督的全局属性编辑。

**💡 创新点**

创新之处在于将TiTok的学习式latent槽概念迁移到音频领域，使用固定数目的可学习槽取代帧级token，使每个槽聚合全句信息，并可通过交换槽实现说话人或噪声等全局属性的控制。

**🔧 技术方法**

使用技术包括冻结FocalCodec前端并插入compressor-quantizer-decompressor链；Binary Spherical Quantization做离散化；slot importance评估确定槽的重要性；以及基于重要性排名的零样本token替换。

**📊 数据集**

实验数据集包括LibriSpeech、VoiceBank、Libri1Mix、VCTK、LibriTTS，以及加入白噪声和WHAM!噪声的LibriTTS，以评估重构、属性分析与编辑效果。

**📈 对比分析**

与EnCodec、DAC、FocalCodec、Stable Codec等多种现有神经语音编解码器在UTMOS/DNSMOS/dWER/相似度等指标上对比，LATTE Large在感知质量上与最强基线相当或更优，但在词级可懂度（dWER）上略低。

**⚠️ 局限性**

局限性包括：由于缺乏帧级对齐，细粒度语音细节的可懂度稍逊；槽间仍存在部分属性混杂，难以实现完全的因子分离；并且对更大规模或更多样化音频域的推广仍需进一步研究。

---

## 136. Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs

**arXiv ID:** 2605.10998 | [PDF](https://arxiv.org/pdf/2605.10998v1)

**作者:** Sangyeon Yoon `[一作]` (Yonsei University), Albert No `[通讯]` (Yonsei University)

**通讯引用:** 527 | [OpenAlex ID](https://openalex.org/A5049196468)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种利用最小规模（10个）无害偏好对话对进行Direct Preference Optimization（DPO）攻击，能在保持模型实用性的同时削弱其安全对齐，导致模型对有害提示的合规率显著提升。

**💡 创新点**

创新点在于展示DPO优化目标本身即可在不包含任何有害内容的训练样本下诱导安全行为削弱，且该攻击极其轻量、低成本且难以通过现有内容审核发现。

**🔧 技术方法**

主要技术包括DPO训练框架、偏好对话对构造、对模型拒绝行为的相对优先级调节，以及使用GPT‑5‑mini等大模型进行自动化评估与梯度相似性分析。

**📊 数据集**

使用的训练数据为10条纯无害提示及其对应的模型友好回答和标准拒绝回复；评估数据来自HEx‑PHI、HarmBench、SorryBench、StrongREJECT、JailbreakBench等多种红队测试集。

**📈 对比分析**

实验对比了多款OpenAI GPT‑4系列模型及开源Llama3、Qwen等模型，攻击成功率（ASR）分别在GPT‑4o、GPT‑4.1、GPT‑4.1‑mini、GPT‑4.1‑nano上达到约59%–82%，显著高于传统SFT或LoRA攻击，且对模型下游性能损失最小。

**⚠️ 局限性**

局限性包括：攻击仅针对DPO及类似基于偏好的微调方法，且在极少数据场景下仍有效但对不同安全约束的鲁棒性未知；同时对更复杂的对齐机制或多模态模型的适用性尚需进一步验证。

---

## 137. STRIDE: Training-Free Diversity Guidance via PCA-Directed Feature Perturbation in Single-Step Diffusion Models

**arXiv ID:** 2605.11494 | [PDF](https://arxiv.org/pdf/2605.11494v1)

**作者:** Ankit Yadav `[一作]` (Australian Institute for Machine Learning, Adelaide University), Lingqiao Liu `[通讯]` (Australian Institute for Machine Learning, Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了STRIDE，一种训练和优化无关、仅在单次前向推理中通过PCA投影的结构化噪声注入来提升少步扩散模型的生成多样性。

**💡 创新点**

核心创新在于把噪声投影到模型自身激活的主成分子空间内，并结合1/f谱的空间相关噪声，既保持了多样性又不破坏文本对齐。

**🔧 技术方法**

技术包括在线局部特征PCA、基于FFT的pink噪声生成、空间补丁化、单前向钩子注入和频率形状调节。

**📊 数据集**

使用FLUX.1-schnell（单步）和SD3.5 Turbo（四步）两大模型，在COCO、DrawBench、PartiPrompts和GenEval等四个常用文本-图像数据集上进行实验。

**📈 对比分析**

与CADS、Input Noise、DDC、SPELL等训练无关基线对比，STRIDE在InBSim、CLIP、HPS和KID等多样性与质量指标上实现Pareto支配，单前向推理下显著提升多样性且文本一致性几乎不变。

**⚠️ 局限性**

主要局限包括：高强度噪声可能导致高频伪影；在线PCA计算带来额外推理开销；对不同模型结构需手动调整注入层和时间步，迁移成本较高。

---

## 138. Beyond Prediction: Interval Neural Networks for Uncertainty-Aware System Identification

**arXiv ID:** 2605.11460 | [PDF](https://arxiv.org/pdf/2605.11460v1)

**作者:** Mehmet Ali Ferah `[一作]` (Istanbul Technical University), Tufan Kumbasar `[通讯]` (Istanbul Technical University)

**通讯引用:** 2453 | [OpenAlex ID](https://openalex.org/A5010725194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一套面向系统辨识的区间神经网络（INN）框架，用于无概率假设下进行不确定性量化，构造了ILSTM和INODE两种区间模型，并提供了两种训练策略（Cascade INN和Joint INN）。

**💡 创新点**

创新点在于：① 将传统神经网络转换为区间形式，直接在参数层嵌入不确定性；② 设计两种训练策略，其中Joint INN通过GradNorm实现点预测与预测区间的多目标平衡；③ 引入通道级弹性（channel‑wise elasticity）对模型不确定性进行可解释性分析。

**🔧 技术方法**

主要技术包括区间算术（interval arithmetic）扩展LSTM和NODE网络；自定义不确定性损失（RQR‑W）；参数化技巧（abs / ReLU）保证区间有效性；单阶段联合优化与两阶段转换策略；以及弹性分析工具。

**📊 数据集**

实验使用四个公开SysID基准数据集：MR‑Damper、Heat Exchanger、Hair Dryer和Robot Arm，所有数据均做z‑score归一化后分为训练/测试。

**📈 对比分析**

与传统的Bayesian NN、MC Dropout、Deep Ensemble等概率方法以及无不确定性神经网络进行对比。结果显示：C‑INN在RMSE上更优，J‑INN在PICP和PINAW方面更紧凑、校准度更高，整体CWC指标上均超过或匹配基准方法。

**⚠️ 局限性**

局限性包括：① 对大型网络和实时部署的可扩展性尚未验证；② 训练过程对超参数（如r_h、r_o、λ、β）敏感；③ 弹性分析目前仅对小规模网络可视化，难以在复杂网络中快速解释。

---

## 139. FastUMAP: Scalable Dimensionality Reduction via Bipartite Landmark Sampling

**arXiv ID:** 2605.11428 | [PDF](https://arxiv.org/pdf/2605.11428v1)

**作者:** Hongmin Li `[一作]` `[通讯]` (Institute of Science Tokyo), Hongmin Li (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FastUMAP，一种基于 landmark 的快速非线性降维方法，旨在解决多次探索性嵌入时的计算瓶颈。

**💡 创新点**

创新点包括：① 用稀疏点–landmark 二部模糊图替代全邻接图，显著降低邻居图构造成本；② 采用 Nyström 谱暖启动，为所有样本提供几何导向的初始化；③ 在 SGD 优化中引入角色区分（数据点 vs landmark）和不同的力学参数；④ 将 landmark 比例 r = m/n 明确为可调的速度‑精度权衡参数。

**🔧 技术方法**

技术手段：k‑NN 搜索生成 landmark 邻居；构造点–landmark 二部模糊图；Nyström 方法求解 m×m 谱特征；UMAP 风格的交叉熵目标与负采样 SGD；角色差异化的优化力学；线性时间复杂度的近似实现。

**📊 数据集**

使用了 9 个基准数据集，涵盖 178–70,000 样本，包含低维表格数据（Wine、Dermatology、Breast Cancer）、中等规模结构化数据（Mfeat、Spambase、Dry Bean、Shuttle）以及大规模图像数据（MNIST、Fashion‑MNIST）。

**📈 对比分析**

在同一工作站上与 BH‑t‑SNE、UMAP、openTSNE 等公开实现进行对比，使用相同预处理（min‑max + PCA），评估壁钟时间和嵌入空间的 kNN 准确率。FastUMAP 在 7/9 数据集上取得最快运行时间，尤其在 MNIST/Fashion‑MNIST 仅需约 4.6 s（vs ~73‑75 s BH‑t‑SNE），但平均 kNN 准确率为 91.4%（BH‑t‑SNE 为 94.6%）。准确率随 landmark 比例 r 逐步提升，运行时间随 r 减小。

**⚠️ 局限性**

局限性：① 固定 5,000 个 landmark 在最大规模数据集上导致准确率下降；② 与 BH‑t‑SNE 及标准 UMAP 在精度上仍有明显差距；③ 评测仅在单一硬件与默认实现下进行，未覆盖多线程或其它实现的完整对比；④ 作为探索性工具，可能被误用为决策依据，需要谨慎解释。

---

## 140. Beyond Polynomials: Optimal Locally Recoverable Codes from Good Rational Functions

**arXiv ID:** 2605.11465 | [PDF](https://arxiv.org/pdf/2605.11465v1)

**作者:** Hengfeng Liu `[一作]` (Southwest Jiaotong University), Xuemin Zheng `[通讯]` (China West Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文将良好多项式推广为良好有理函数，并基于此构造了一类新的最优局部可恢复码（LRC），实现了长度最大化

**💡 创新点**

创新点在于将有理函数与代数函数域、Galois 理论相结合，首次通过控制有理函数的 Galois 群大小来显著提高可恢复码的长度和灵活性

**🔧 技术方法**

核心技术包括代数函数域理论、分歧与完全分裂点分析、Hurwitz 公式、Galois 作用与 PGL(2,q) 子群构造

**📊 数据集**

论文主要为理论构造，未使用具体数据集；所有结果均来自解析推导与代数证明

**📈 对比分析**

与传统 Tamo–Barg 多项式构造相比，该方法在相同局部性下能获得更大长度（接近 q+1），并在某些情形下达到理论上可获得的最大完全分裂点数，实验性证明显示改进幅度随局部性参数 r 显著增大

**⚠️ 局限性**

局限性包括：构造依赖于存在合适的 PGL(2,q) 子群，受限于 q 与 r 的数论约束；实现复杂度较高；并且对高阶分裂点的分析仍有待进一步简化

---

## 141. Survey-Free Radio Map Construction via HMM-Based Coarse-to-Fine Inference

**arXiv ID:** 2605.11038 | [PDF](https://arxiv.org/pdf/2605.11038v1)

**作者:** Zheng Xing `[一作]` (FNii Chinese University of Hong Kong), Shuguang Cui `[通讯]` (FNii Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种完全无现场测绘、无IMU的基于RSS的室内无线信号地图构建方法，利用HMM的粗细推断从未标记的RSS序列恢复用户轨迹并生成精细的无线电图。

**💡 创新点**

创新点包括：将区域级与位置级推断分层并结合HMM实现无标签标注；通过RNN嵌入和二分类自监督提升时间连续性；利用虚拟‑物理区域匹配和基于遗传算法的轨迹搜索实现精细定位。

**🔧 技术方法**

采用HMM+EM、RNN嵌入、子空间PCA、图匹配（Hungarian算法）、遗传算法、区域约束的RSS路径损耗模型、泊松驻留时间模型等技术。

**📊 数据集**

在一栋768㎡的办公室环境中，9个区域、27个AP，4名用户采集的RSS序列（总长度约4万条），并构造了三个包含200个随机位置的测试集。

**📈 对比分析**

与多种无标签聚类方法（SPC、BD‑DB、LSE等）以及有标签方法（SVM、KNN、MLP）在区域标注上比较，HCFI实现97.8%准确率、100%拓扑一致；在无线电图构建与定位上与WCL、RRM、VRLoc、Leto等方法对比，MAE为8.96 dB、定位误差为3.33 m，接近手工测绘的2.24 m，且完全不依赖IMU。

**⚠️ 局限性**

仅适用于单向走廊流动、用户不回访已走过区域的场景；对开放式或复杂路径环境需要进一步扩展；仍需要已知楼层图和AP位置。

---

## 142. Adaptive Calibration in Non-Stationary Environments

**arXiv ID:** 2605.11490 | [PDF](https://arxiv.org/pdf/2605.11490v1)

**作者:** Junyan Liu `[一作]`, Lillian J. Ratliff `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一类在线预测算法，能够在不知情的非平稳环境中自适应地实现概率预测的校准，并给出了对三种校准误差（ℓ1、ℓ2和伪KL）的上界，能够在 i.i.d. 与完全对抗性两端实现最佳性能，且在中间区间实现逐步平滑的误差上界。

**💡 创新点**

创新点包括：
1) 定义了一个新的非平稳性度量 C（即平均预测值的最小 ℓ1 偏差），并证明了校准误差可随 C 平滑地从 i.i.d. 率（√T 或 logT）过渡到对抗性率（T2/3 或 T1/3）。
2) 设计了基于 epoch 的非均匀划分框架，在每个 epoch 内构造离真值更近的细粒度区间，利用 MsMwC、swap‑regret 与 OGD 等在线学习子算法实现自适应校准。
3) 通过对 ℓ1、ℓ2 与伪KL 三种校准指标分别给出可实现的上界，证明了同一框架可获得统一的 (1+C)1/3 量级性能。
4) 给出了与已知 C 情况的 reduction，进一步消除对 C 的预先了解。

**🔧 技术方法**

技术手段主要包括：
- MsMwC（multi‑armed bandit with mixed confidence）算法用于 ℓ1 校准。
- 递增 epoch 调度与自适应非均匀区间划分。
- swap‑regret 与在线梯度下降（OGD）结合，用于 ℓ2 与伪KL 校准。
- 以 C 为参数的自适应区间宽度 r_m 以及分区数 N_m、K 的精确设置。
- 通过分块与自我约束（self‑bounding）技术对非平稳性误差进行控制。

**📊 数据集**

本文为理论工作，未使用任何具体数据集；所有结果均在对数式、期望式上通过概率不等式和调度证明得到。

**📈 对比分析**

与现有工作比较：
- 在 C=0 时（i.i.d.）恢复已知最优下界（ℓ1: √T、ℓ2: logT、伪KL: logT）
- 在 C=T 时（完全对抗）达到或接近已知最优上界（ℓ1: T2/3、ℓ2 & 伪KL: T1/3）
- 对于中间 C，给出了与已知下界匹配的上界，弥补了此前仅在极端两端有结果的空白。
- 进一步提升了时间复杂度：在 C=0 时每步只需 O((1+C)1/3) 计算量，优于传统 O(√T) 方案。

**⚠️ 局限性**

局限与未来工作：
- 对于其他校准概念（光滑校准、子样本校准、距离校准等）尚无完整自适应算法与下界。
- 现有结果为上界，缺乏针对 C 的匹配下界，需进一步探讨中间区间的最优性。
- 仅针对无上下文、二元事件的设定，如何推广至上下文、多类别或结构化预测仍是开放问题。
- 目前算法在实践中未做实验验证，未来需要在真实数据上评估其鲁棒性与效率。

---

## 143. SURGE: Surrogate Gradient Adaptation in Binary Neural Networks

**arXiv ID:** 2605.10989 | [PDF](https://arxiv.org/pdf/2605.10989v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 144. Enforcing Constraints in Generative Sampling via Adaptive Correction Scheduling

**arXiv ID:** 2605.11214 | [PDF](https://arxiv.org/pdf/2605.11214v1)

**作者:** Noah Trupin `[一作]` (Purdue University), Yexiang Xue `[通讯]` (Purdue University)

**通讯引用:** 1890 | [OpenAlex ID](https://openalex.org/A5060838579)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个基于局部约束缺陷的自适应校正调度框架，用于在生成采样过程中动态决定投影时机，以保持轨迹一致性。

**💡 创新点**

将约束强制视为预算分配问题，引入一阶缺陷作为在线价值近似，并设计了基于剩余预算的阈值策略，使得在匹配投影预算的前提下获得几乎逐步投影的性能。

**🔧 技术方法**

使用生成式模型（如逆扩散、流匹配）+投影算子、预算调度策略和一阶缺陷测度；实现为在线投影决策算法。

**📊 数据集**

在六种受控流形模拟环境（SO(3)、SE(3)、Terrain、SO(3)-Impulse、SE(3)-Lever、Terrain-Ridge）以及已训练的Projected Diffusion Models（PDM）上进行实验。

**📈 对比分析**

与终端投影、周期投影、逐步投影等基线对比；在相同投影预算下，适应性调度在路径误差指标NEPE上实现了约41%‑71%的改进，且在高缺陷聚集场景下优势最显著。

**⚠️ 局限性**

仅使用一阶缺陷作为下游价值代理，可能在缺陷与轨迹偏差不对齐或投影不稳定时失效；假设投影算子可用且无额外开销，且未对更高维度、噪声更大的场景做充分验证。

---

## 145. Recent Advances in Spatially Coupled Codes: Overview and Outlook

**arXiv ID:** 2605.11542 | [PDF](https://arxiv.org/pdf/2605.11542v1)

**作者:** Min Qiu `[一作]`, Jinhong Yuan `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了空间耦合码（SC-LDPC、SC-TC、阶梯码/缝隙码、SC-SPARC 等）在理论与实践上的最新进展，并对其关键属性、设计方法及性能进行了系统评估。

**💡 创新点**

在传统综述基础上，本文提出了统一的分类框架，阐明了阈值饱和、通用性、线性最小距离与有限长度缩放等性质，并指出目前结构化码阈值饱和的严格证明缺失、有限长度通用性不足、高吞吐量实现等未解决的关键问题。

**🔧 技术方法**

主要采用密度演化、潜在函数分析、窗口式 BP/AMP 解码、连通链与子块局部性等技术，探讨其对阈值、误码率和误码底的影响。

**📊 数据集**

论文未使用公开数据集，而是通过理论分析和仿真（如 BEC、AWGN、BSC 等通道模型）验证不同码种类和参数设置下的性能。

**📈 对比分析**

通过比较 BP 阈值、误码率曲线、误码底和窗口解码的收敛速度，展示了 SC-LDPC 与 SC-TC 在不同速率和块长下的性能优劣；实验表明阈值更高的码在中等块长下具有更好的水平方向性能，而在短块长下误码底更低。

**⚠️ 局限性**

主要限制包括：1）结构化（确定性）耦合码阈值饱和的严格证明仍缺失；2）通用性在有限长度或非 BMS 通道上的证明不足；3）高吞吐量实现、速率损失抑制、波动速度优化及量子纠错等实际部署问题尚未得到充分解决。

---

## 146. VERDI: Single-Call Confidence Estimation for Verification-Based LLM Judges via Decomposed Inference

**arXiv ID:** 2605.11334 | [PDF](https://arxiv.org/pdf/2605.11334v1)

**作者:** Jasmine Qi `[一作]` (Indeed Inc), Muyang Sun `[通讯]` (Indeed Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于LLM判定者推理轨迹的置信度评估方法VERDI，该方法无需额外推理调用，利用推理轨迹中的结构信号（SVA、CLM、EGS）和Platt缩放的逻辑回归来计算判定结果的置信度。

**💡 创新点**

通过拆解验证任务为子检查，提取推理轨迹内部的一致性、命题级差距和证据基础等结构化信号，实现对LLM判定结果的后置置信度估计，并提供33M参数的NLI模型作为可扩展、与规则无关的替代方案。

**🔧 技术方法**

结构化推理轨迹分析、步骤-判决一致性（SVA）、命题级差距（CLM）、证据基础得分（EGS）、Platt缩放的逻辑回归、NLI模型以及传统的token logprob、口头置信度和轨迹长度等基线比较。

**📊 数据集**

内部8个二分类评价维度的验证集（≈476样本/维度）、SummEval（新闻摘要一致性）、FEVER（事实验证）和SciFact（科学命题验证）四个公开数据集。

**📈 对比分析**

与logprob、口头置信度、轨迹长度、CoT-UQ等基线在相同模型下进行5折交叉验证；VERDI在GPT-4.1-mini上AUROC 0.72–0.91，GPT-5.4-mini 0.66–0.80，Qwen3.5 0.56–0.70；在内部验证集上对事实类维度（Q2、Q4、Q5）表现最佳，并能在缺乏logprob时仍提供可靠置信度。

**⚠️ 局限性**

仅适用于结构化验证任务，依赖轨迹条理性；当任务准确率高于94%时误差较少，后置置信度难以提升；跨域迁移受轨迹格式影响；无法用于无证据对比或开放式生成的置信度评估。

---

## 147. Finite Volume-Informed Neural Network Framework for 2D Shallow Water Equations: Rugged Loss Landscapes and the Importance of Data Guidance

**arXiv ID:** 2605.11001 | [PDF](https://arxiv.org/pdf/2605.11001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 148. Drop the Act: Probe-Filtered RL for Faithful Chain-of-Thought Reasoning

**arXiv ID:** 2605.11467 | [PDF](https://arxiv.org/pdf/2605.11467v1)

**作者:** Swapnil Parekh `[一作]` `[通讯]` (Intuit), Swapnil Parekh (Intuit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出ProFIL方法，利用冻结基模型的多头注意力探针在强化学习训练中筛选剧院式的后续推理步骤，提升推理链的真实性并缩短链长；

**💡 创新点**

创新点在于：1）构建无需人工标签的“内部”承诺探针，提供稳定的可信度信号；2）在GRPO框架下以探针阈值过滤不可信奖励，实现对剧院式推理的根除；3）将探针训练与RL分离，避免探针被模型操纵；

**🔧 技术方法**

技术包括：多头注意力探针（RLFR风格）、冻结基模型的内部激活、GRPO（Group Relative Policy Optimization）、强制回答法生成承诺标签、阈值过滤机制；

**📊 数据集**

数据集涵盖四个推理领域：GSM8K（数学问题）、LiveCodeBench（竞赛编程）、ToolUse（多步工具调用）、MMLU-Redux（多学科知识测验），使用Llama-8B与Qwen-7B两大模型；

**📈 对比分析**

与标准GRPO和加长惩罚的匹配基线对比，ProFIL在所有领域将后承诺剧院比例下降11–100%，提升可信度比率（例如LiveCodeBench +24pp），链长缩短4–19%，同时保持或提高任务准确率；

**⚠️ 局限性**

限制包括：仅在可验证标签的任务上有效，未直接验证在开放式对话或写作等无确切答案的场景；探针阈值需手工调节；实验仅在DeepSeek-R1-Distill模型上验证，缺乏更大模型或多样化架构的评估。

---

## 149. DenseTRF: Texture-Aware Unsupervised Representation Adaptation for Surgical Scene Dense Prediction

**arXiv ID:** 2605.11265 | [PDF](https://arxiv.org/pdf/2605.11265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 150. Vertex-Softmax: Tight Transformer Verification via Exact Softmax Optimization

**arXiv ID:** 2605.10974 | [PDF](https://arxiv.org/pdf/2605.10974v1)

**作者:** Navid Rezazadeh `[一作]` (University of California), Arash Gholami Davoodi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Vertex‑Softmax 原语，能够在给定的分数区间和方向系数下，精确计算注意力层中 softmax 的线性目标最小值；并将其集成到 Vertex‑CROWN 验证器中，显著提升对 Transformer 关注层的可验证鲁棒性；

**💡 创新点**

核心创新在于证明了最优解必定在约束盒的顶点，并进一步揭示该顶点可通过排序后阈值分割压缩为仅 K+1 个候选点，从而实现 O(KlogK) 的 exact 求解；同时给出了信息最优性定理，说明仅凭独立分数区间无法进一步提升下界；

**🔧 技术方法**

使用的主要技术包括：线性分数盒的线性分式优化、顶点精确性证明、阈值结构定理、基于 CROWN 的凸松弛传播以及与 Vertex‑Softmax 的集成；

**📊 数据集**

实验使用 MNIST、Fashion‑MNIST 与 CIFAR‑10 三个公开图像数据集的 patch‑attention 模型以及完整的 attention‑残差‑MLP 块；

**📈 对比分析**

与现有的 CROWN、α‑CROWN、Wei‑LSE、GaLileo 等软max 松弛方法以及固定预算的 ABCrown‑BaB 进行对比；Vertex‑CROWN 在多数测试场景下实现了更高的认证率、更小的攻击间隙和显著更低的运行时成本（通常比 α‑CROWN 低 30–100 倍、比 ABCrown‑BaB 低 3–4 级）；

**⚠️ 局限性**

局限性包括：仅消除了 softmax 松弛的松弛，仍受独立分数盒、值下界、行间组合以及非线性后缀松弛的影响；未考虑分数相关性、分数–值耦合以及多行联合优化，因而在深层 Transformer 或存在大量交互的网络中仍存在可进一步改进的空间；

---

## 151. Engagement Process: Rethinking the Temporal Interface of Action and Observation

**arXiv ID:** 2605.11484 | [PDF](https://arxiv.org/pdf/2605.11484v1)

**作者:** Jialian Li `[一作]` (XPENG Robotics), Jie Chen `[通讯]` (XPENG Robotics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Engagement Process (EP) 形式化，显式区分动作与观察在时间轴上的独立事件；

**💡 创新点**

创新点在于将动作与观察解耦为时间标记的事件流，并在同一时序上提供统一接口，支持延迟反馈、持续动作、异步观测等多种时序现象；

**🔧 技术方法**

使用基于 EP 的离散/连续时间决策模型、强化学习、LLM 生成与工具调用、以及基于奖励的时间感知训练；

**📊 数据集**

实验数据集包括自制 toy 环境、LLM 生成的数字助理与烹饪/教学交互场景、DeepMath-103K 代数题集；

**📈 对比分析**

与传统同步 POMDP/Step、循环式 AgentLoop、周期性轮询 Poll 以及手工中断补丁比较，EP 在任务成功率、超时率、首响应延迟、总体收益等指标上均优于对照方法；

**⚠️ 局限性**

局限性包括：对真实连续时间系统的适配仍需改进、对大规模多智能体/复杂层次结构的实验不足、以及 EP 仅提供接口，仍需配合合适的架构与学习算法实现最佳性能。

---

## 152. Skill Drift Is Contract Violation: Proactive Maintenance for LLM Agent Skill Libraries

**arXiv ID:** 2605.10990 | [PDF](https://arxiv.org/pdf/2605.10990v1)

**作者:** Linfeng Fan `[一作]` (Renmin University of China), Zhiwu Lu `[通讯]` (Renmin University of China)

**通讯引用:** 2440 | [OpenAlex ID](https://openalex.org/A5085349794)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将LLM技能漂移视为合同违约，并实现了一个只监测角色依赖的精确维护监视器。

**💡 创新点**

创新点在于引入角色感知的环境合同抽取与验证，消除因非操作性引用导致的误报。

**🔧 技术方法**

技术上结合正则提取、语义化合同生成、类型兼容匹配与实时环境验证。

**📊 数据集**

使用了新构建的880对技能衰退基准，包括174对人工合成漂移、107个真实漂移和599个负样本。

**📈 对比分析**

与传统CI探针、依赖扫描和文本差分方法对比，Precision 100%，Recall 76%（最强基线），误报率 0%，而检测率高于40% FPR 的传统方法。

**⚠️ 局限性**

局限性在于缺乏对复杂架构或认证/模式变更的完整验证，且修复验证仅基于文字替换，未覆盖语义层面。

---

## 153. Causal Algorithmic Recourse: Foundations and Methods

**arXiv ID:** 2605.11373 | [PDF](https://arxiv.org/pdf/2605.11373v1)

**作者:** Drago Plecko `[一作]`, Elias Bareinboim `[通讯]` (Columbia University)

**通讯引用:** 3749 | [OpenAlex ID](https://openalex.org/A5039620960)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于结构因果模型的算法回归框架，利用在同一主体上多次决策的观测数据（recourse data）来推断实施干预后结果的分布，并给出了从仅有观测数据时推断的条件与方法。

**💡 创新点**

创新点包括：
- 定义了“后回归稳定性（Post‑Recourse Stability, PRS）”条件，描述在干预后潜在变量的分布如何保持或变化；
- 引入了“recourse data”概念，允许在同一主体上观察干预前后的结果；
- 设计了基于Frank copula的无参数推断与灵敏度分析方法；
- 提出了copula拟合的好坏检验与相应的重采样统计量；
- 在PRS不满足时给出一种copula‑free的两阶段回归学习方案。

**🔧 技术方法**

使用的主要技术包括：
- 结构因果模型（SCM）与干预符号 do();
- 量化量子回归与分位数回归；
- Frank copula 与斯克拉尔定理建模分位数间的依赖；
- 蒙特卡罗采样与Bootstrap 统计量实现置信区间与假设检验；
- 两阶段 OLS/回归法提升在PRS不满足情形下的预测精度。

**📊 数据集**

实验数据：
- 真实的 HELOC（房贷额度授信）数据（10459 条样本）；
- 半合成 HELOC 数据（SeS‑HELOC）用来验证 copula 是否合适以及 PRS 是否成立。

**📈 对比分析**

比较方法与性能：
- 对照 Frank copula 推断、直接回归与真实 SCM 生成的结果，用 Kolmogorov–Smirnov (KS) 统计量评估分布差距；
- 在 PRS 满足时，copula‑based 方法与真实分布差距小；
- 在 PRS 失效时，copula‑free 两阶段回归优于直接回归和 copula‑based 方法；
- 通过 p‑value 分布检验 copula 拟合的有效性，PRS 成立时 p‑value 接近均匀分布。

**⚠️ 局限性**

局限性：
- 需要先验的因果图，若隐藏混杂或因果结构错误，结果会失真；
- PRS 条件可能在实际应用中难以检验；
- 需要足够多的 recourse 数据才能可靠估计 copula 参数或回归系数；
- Frank copula 仅为一种特定形式，若真实依赖结构不同，可能导致误判；
- 量化量子估计误差对推断结果有影响，尤其在样本量有限时。

---

## 154. Kairos: A Scalable Serving System for Physical AI

**arXiv ID:** 2605.11381 | [PDF](https://arxiv.org/pdf/2605.11381v1)

**作者:** Yinwei Dai `[一作]` (Princeton University), Ravi Netravali `[通讯]` (Princeton University)

**通讯引用:** 2710 | [OpenAlex ID](https://openalex.org/A5053593890)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

设计并实现了第一个针对物理 AI 的可扩展服务系统，支持动态执行视界、执行感知调度和边缘‑云混合部署。

**💡 创新点**

创新点包括：①基于扩散模型的置信度信号动态决定每轮执行视界；②基于等待比率的分层调度器，将生成与执行阶段结合；③利用观察过期检测实现异步执行与生成同步；④兼容边缘‑云双层资源。

**🔧 技术方法**

技术包括：扩散生成模型、动态执行视界决策、等待比率调度、分层桶排序、动态观测刷新、异步请求/批处理、边缘‑云负载分配。

**📊 数据集**

数据集与基准：LIBERO、Meta‑World、Isaac Lab、RoboTwin 2.0、SIMPLER，以及真实双臂 SO‑101 实验。

**📈 对比分析**

与 FIFO 和 Autellix 基线比较；在峰值负载下平均延迟降低 31.8–66.5%，P25 延迟降低 39.5–88.4%，P95 延迟降低 22.2–52%；在 10–100 台机器人集群下平均延迟下降 20.4–42.8%；在混合边缘‑云部署下平均延迟相较单边缘/单云分别降低 36.9–47.7% 与 51.9–67.9%。

**⚠️ 局限性**

局限性：依赖扩散置信度判定，对阈值敏感；仅针对无状态或短期历史模型；在极大规模或极高网络延迟环境下仍需评估；缺乏对多机器人协作任务的专门优化；实现复杂度较高。

---

## 155. The Price of Proportional Representation in Temporal Voting

**arXiv ID:** 2605.11157 | [PDF](https://arxiv.org/pdf/2605.11157v1)

**作者:** Nicholas Teh `[一作]` (University of Oxford), Nicholas Teh `[通讯]` (University of Oxford)

**通讯引用:** 75 | [OpenAlex ID](https://openalex.org/A5037242857)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了时间投票模型中的比例代表性，量化了在满足不同比例代表性公理（JR、PJR、EJR、EJR+）时对效用（utilitarian welfare）的最坏情况损失，并探讨了在此约束下最大化效用的算法复杂度。

**💡 创新点**

创新点在于提出“比例代表性的价格”框架并给出JR的渐进零损失与更强公理的非零下界，证明了在几乎所有结构限制下的NP/APX难度，并在候选人数、选民类型及轮次轮廓数等参数上实现了固定参数可解（FPT）算法。

**🔧 技术方法**

使用了组合构造、归约（如从集合覆盖或团问题）、整数规划、动态规划以及参数化复杂度理论中的Lenstra算法和动态规划技术。

**📊 数据集**

本文完全是理论分析，没有使用任何真实数据集；所有结论均来自构造实例和抽象模型。

**📈 对比分析**

通过理论上与无约束最优效用做比较，给出价格上界与下界；在可行域上证明了NP‑完整性和APX‑难度；在特定参数下提供了多项式时间/固定参数多项式时间算法，展示了在这些结构化实例中可实现的高效性。

**⚠️ 局限性**

局限性包括：仅考虑效用最大化（未研究均衡/纳什效用等其他目标）、最坏情况分析可能不反映实际实例的平均性能、对动态偏好变化的实证验证缺失，以及对更强公理在实际应用中可实现性的进一步探索仍需后续工作。

---

## 156. USEMA: a Scalable Efficient Mamba Like Attention for Medical Image Segmentation

**arXiv ID:** 2605.11131 | [PDF](https://arxiv.org/pdf/2605.11131v1)

**作者:** Elisha Dayag `[一作]` (University of California Irvine), Jack Xin `[通讯]` (University of California Irvine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于UNet的医学图像分割网络USEMA，将可扩展的高效Mamba式注意力SEMA与卷积特征提取相结合；

**💡 创新点**

创新点在于将局部窗口注意力与全局算术平均相结合，形成理论一致的SEMA注意力，既降低了计算复杂度，又保持了全局信息；

**🔧 技术方法**

核心技术包括SEMA注意力（窗口注意力+全局平均）、Mamba状态空间模型、旋转位置编码、深度可分离卷积、残差块以及UNet对称结构；

**📊 数据集**

在三个多模态数据集上验证：腹部MRI（MICCAI 2022 AMOS）、内镜视频（MICCAI 2017 Endovis）和显微镜细胞图像（NeurIPS 2022 Cell Segmentation Challenge）；

**📈 对比分析**

与多种Transformer（UNETR、Swin-UNETR、nnFormer）和Mamba/UMamba模型对比，USEMA在DSC/NSD/F1分数上均超过基线，参数量更少、计算成本更低；

**⚠️ 局限性**

局限性包括对极长序列的平均近似仍为均匀平均，未来可探索学习权重的平均或稀疏加权，以及对大尺寸病理图像的进一步扩展。

---

## 157. A Systematic Security Testing Approach for InterUSS-based environments

**arXiv ID:** 2605.11339 | [PDF](https://arxiv.org/pdf/2605.11339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 158. LatentRouter: Can We Choose the Right Multimodal Model Before Seeing Its Answer?

**arXiv ID:** 2605.11301 | [PDF](https://arxiv.org/pdf/2605.11301v1)

**作者:** Xueqi Cheng `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 1001 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LatentRouter，一种用于多模态大语言模型（MLLM）路由的系统；

**💡 创新点**

创新点在于把路由建模为反事实多模态效用预测，通过学习的路由胶囊与模型能力标记进行潜在通信，实现查询与模型能力的精细匹配；

**🔧 技术方法**

核心技术包括多模态路由胶囊、模型能力Token、潜在通信层、分布式反事实预测头以及有限修正机制；

**📊 数据集**

使用了MMR-Bench和VL-RouterBench这两个公开的多模态路由基准数据集；

**📈 对比分析**

与固定模型、特征级路由器和其他学习路由器进行比较，在性能导向和成本导向两种设置下均显著优于非oracle基线，提升了nAUC和Rank Score；

**⚠️ 局限性**

局限性在于对可用模型池质量高度依赖，最强模型移除后性能下降；同时需要收集并标注大量模型效用数据以训练路由器。

---

## 159. The Metaverse Is Not a Place Apart: Law, Code, and the Recursive Governance of Digital Space (A Review Essay on Mark Findlay, Governing the Metaverse: Law, Order and Freedom in Digital Space (2025))

**arXiv ID:** 2605.11023 | [PDF](https://arxiv.org/pdf/2605.11023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 160. Optimal Representations for Generalized Contrastive Learning with Imbalanced Datasets

**arXiv ID:** 2605.11291 | [PDF](https://arxiv.org/pdf/2605.11291v1)

**作者:** Thuan Nguyen `[一作]` (East Tennessee State University), Prakash Ishwar `[通讯]` (Boston University)

**通讯引用:** 5977 | [OpenAlex ID](https://openalex.org/A5036913803)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究对比学习在类别不平衡情况下的最优表示几何特性，证明内部类方差坍塌与少数类坍塌，并给出凸优化求解方法。

**💡 创新点**

提供了对一般严格凸递增对比损失的可计算几何描述，揭示了相对类别比例决定的等角对称结构，并首次在极端不平衡下证明少数类坍塌的非渐近阈值。

**🔧 技术方法**

基于无约束特征模型，构造了对类均值的下界，利用凸优化求解Gram矩阵，结合凸分析与等角框架证明相关性质。

**📊 数据集**

在CIFAR‑10数据集的三分类子集上进行实验，分别构造平衡与极度不平衡两种比例。

**📈 对比分析**

通过训练ResNet‑50并与CVX求解的最优Gram矩阵对比，实验结果与理论一致，展示了少数类坍塌现象，证明所提出方法有效。

**⚠️ 局限性**

仅处理维度满足 d≥C‑1 的情形，未涵盖高维多类场景；对硬负样本和多主类极端不平衡阈值的解析仍待研究；理论基于理想化损失函数，实际应用可能受限。

---

## 161. Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing

**arXiv ID:** 2605.11202 | [PDF](https://arxiv.org/pdf/2605.11202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 162. Unlearning with Asymmetric Sources: Improved Unlearning-Utility Trade-off with Public Data

**arXiv ID:** 2605.11170 | [PDF](https://arxiv.org/pdf/2605.11170v1)

**作者:** Ahmed Mehdi Inane `[一作]` (Université de Montréal), Ioannis Mitliagkas `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Asymmetric Langevin Unlearning（ALU）框架，利用公共数据来提升机器学习模型的遗忘效果，降低噪声要求并保持高效性。

**💡 创新点**

创新点在于将公共数据注入 Langevin Unlearning，证明公开数据可将隐私税降低到 O(1/n_pub^2)，实现对大规模删除的支持、计算优势及对分布不匹配的鲁棒性。

**🔧 技术方法**

使用技术包括噪声注入的 Langevin 采样、对数 Sobolev 不等式分析、Rényi 散度（变分估计）衡量遗忘质量、U‑LiRA 成员推断攻击评估，以及对比实验。

**📊 数据集**

主要使用 DomainNet（Quickdraw、Clipart、Infograph、Real）和 IMDB 与 Amazon 评价数据集进行实验验证。

**📈 对比分析**

通过与对称 Langevin Unlearning、完整重训练以及 U‑LiRA 攻击进行对比，发现 ALU 在大规模删除场景下可在低噪声条件下保持更高精度，攻击成功率显著下降；在分布对齐的场景下性能损失仅为几百分点。

**⚠️ 局限性**

局限性包括 Log‑Sobolev 常数难以估计、对公共与私有分布不匹配敏感、需大量模型样本进行 Rényi 散度估计，以及公共数据来源可能带来的二次泄露风险。

---

## 163. A Study on Hidden Layer Distillation for Large Language Model Pre-Training

**arXiv ID:** 2605.11513 | [PDF](https://arxiv.org/pdf/2605.11513v1)

**作者:** Maxime Guigon `[一作]` (Google DeepMind), Michaël E. Sander `[通讯]` (Google DeepMind)

**通讯引用:** 40 | [OpenAlex ID](https://openalex.org/A5090284871)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了隐藏层蒸馏（HLD）在预训练解码器LLM中的效果，比较了与标准logit蒸馏的性能，使用Gemma3教师与123M/735M学生模型在C4数据上训练。

**💡 创新点**

首次在计算预算严格匹配、FLOP计数精确的实验框架下，系统评估HLD在因果LLM预训练中的价值，并引入共享超参数网格与两种HLD协议（顺序与联合）进行公平比较。

**🔧 技术方法**

采用学习回归器的隐藏层对齐（FitNet与Composite损失）、KL散度的logit蒸馏、AdamW优化器、WSD学习率调度、NanoDo代码库及TPU硬件。

**📊 数据集**

训练使用C4英文子集，评估则在C4验证集以及Wikitext‑103、HellaSwag、WinoGrande、LAMBADA、PIQA、ARC‑E等基准上进行。

**📈 对比分析**

通过共享超参数网格和等价计算预算对比，发现HLD（尤其是HLDF）在C4困惑度上略优于KD，但在下游任务上未能持续超越KD，整体性能差距不大。

**⚠️ 局限性**

HLD的收益微弱且对超参数高度敏感，受教师-学生规模不匹配和计算成本差异限制，缺乏在更大规模或不同模型族上的验证，且统计显著性不足。

---

## 164. CVEvolve: Autonomous Algorithm Discovery for Unstructured Scientific Data Processing

**arXiv ID:** 2605.11359 | [PDF](https://arxiv.org/pdf/2605.11359v1)

**作者:** Ming Du `[一作]` (Advanced Photon Source), Mathew J. Cherukara `[通讯]` (Advanced Photon Source)

**通讯引用:** 3791 | [OpenAlex ID](https://openalex.org/A5066462592)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个零代码、自治式的 C 进化框架 CVEvolve，用于在科学数据处理中自动发现和改进算法。

**💡 创新点**

创新点在于将 LLM 代理与多轮搜索、持久化历史管理、可视化支持和保留测试机制相结合，使得代理能够在无结构化任务中自由配置环境、编写代码并迭代优化。

**🔧 技术方法**

采用 LLM（Claude Opus 4.6）、LangGraph 框架、SQLite 持久化、图像可视化工具、工具调用接口和多模态图像跟随中间件等技术。

**📊 数据集**

使用了三类科研影像数据集：X射线荧光显微镜（XRF）图像配准、X射线衍射的 Bragg 峰检测和 2D 粉末衍射图像分割。

**📈 对比分析**

通过与基线算法（如相位相关、蛮力搜索、手工阈值等）比较，CVEvolve 在所有任务上均取得了显著改进，例如 XRF 配准误差降至 0.12，Bragg 峰检测 F1 从 0.298 提升至 0.788，分割 IoU 从 0.37 提升至 0.53。

**⚠️ 局限性**

局限性包括对小型训练集易出现过拟合、对高维或时空多模态数据支持不足、以及对代理安全性（文件系统访问）仍需容器化环境保护。

---

## 165. Conditional Memory Enhanced Item Representation for Generative Recommendation

**arXiv ID:** 2605.11447 | [PDF](https://arxiv.org/pdf/2605.11447v1)

**作者:** Ziwei Liu `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6506 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种条件记忆增强的条目表示框架（MMGR），在生成式推荐中把SID词元嵌入重构为条目级输入，并在解码时恢复词元级细粒度；

**💡 创新点**

创新点：①MM‑guided token scoring利用多模态嵌入对SID词元做加权，强化条目身份；②双层 Engram 记忆捕捉条目内部代码组合和用户历史的代码转移，保持 SID 结构；③记忆条件 token merge 在压缩时保留结构信息；④记忆恢复预测头在解码时重用记忆，桥接条目级输入与词元级生成；

**🔧 技术方法**

采用多模态查询+注意力机制、双层 Engram 稀疏哈希记忆、门控残差合并、记忆恢复的层级预测头以及 LLM 作为生成器；

**📊 数据集**

使用 Yelp、Amazon Industrial 与 Amazon Instrument 三个公开数据集；

**📈 对比分析**

与传统 flattening 与 token‑merging 方式对比，基于 RQ‑VAE 或 RQ‑Kmeans 量化，Qwen3‑0.6B / LLaMA3‑1B 作为 LLM，结果在 H@K、N@K 上提升约 3–8%，并在推理速度上提升约 2.5×；

**⚠️ 局限性**

局限：需预存多模态嵌入；哈希碰撞可能影响稀疏记忆效果；在极大数据集下可扩展性受限；仅在离线 SID 量化环境中验证。

---

## 166. MCPShield: Content-Aware Attack Detection for LLM Agent Tool-Call Traffic

**arXiv ID:** 2605.11053 | [PDF](https://arxiv.org/pdf/2605.11053v1)

**作者:** Sultan Zavrak `[一作]` `[通讯]` (Duzce University), Sultan Zavrak (Duzce University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于会话图的攻击检测框架，对MCP工具调用流进行图编码，并利用句子嵌入与图神经网络进行恶意会话判别。

**💡 创新点**

主要创新在于：①首次将工具调用的语义内容与图结构联合表示以捕获攻击信号；②提出任务分层评估协议以消除任务记忆偏差；③发现聚合SBERT特征的树模型可超越GNN，且自监督预训练在此任务上无显著优势。

**🔧 技术方法**

采用GraphSAGE/GCN/GAT三种GNN、无图MLP、XGBoost、随机森林、逻辑回归、线性SVM；使用all‑MiniLM‑L6‑v2进行SBERT句子嵌入；并通过对比视图的对比学习实现自监督预训练。

**📊 数据集**

使用了Ras‑Eval、ATBench以及将两者与MCPBench安全样本合并的Combined三组公开MCP工具调用数据集。

**📈 对比分析**

通过任务分层（训练/验证/测试任务不重叠）和标签平衡的验证策略进行比较。结果显示：在Ras‑Eval上XGBoost+SBERT聚合特征实现AUROC 0.975；GNN模型（GraphSAGE、GCN、GAT）AUROC约0.89‑0.92；无图MLP 0.896；随机分割会导致AUROC高达0.90以上，显著抬高。每种攻击模式的召回率存在差异，单向输入操纵召回率最低。

**⚠️ 局限性**

主要限制包括：需要访问工具调用的文本内容（对隐私受限场景不适用）；任务分层评估与真实生产环境的可迁移性尚待验证；数据集来源为研究基准，缺乏真实生产攻击记录；攻击样本主要针对单一LLM模型，缺乏多模型泛化验证；自监督预训练未能提升标签效率，未能体现其潜在优势。

---

## 167. AutoLLMResearch: Training Research Agents for Automating LLM Experiment Configuration -- Learning from Cheap, Optimizing Expensive

**arXiv ID:** 2605.11518 | [PDF](https://arxiv.org/pdf/2605.11518v1)

**作者:** Taicheng Guo `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 12988 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了 AutoLLMResearch 框架，利用 LLM 代理在低成本实验中学习并迁移经验，自动配置高成本 LLM 实验。

**💡 创新点**

创新点在于：① 构建了可验证的多阶梯 LLMConfig‑Gym 环境；② 将配置任务建模为长周期 MDP，使用文本推理进行跨保真度（fidelity）迁移；③ 通过策略蒸馏 + 多轮 RL 的训练管道实现经验累积和可靠性提升。

**🔧 技术方法**

核心技术包括：LLM（Qwen3‑1.7B/4B）、RL‑VR、GRPO、策略蒸馏、最相似配置匹配、跨保真度提示工程。

**📊 数据集**

数据集：自建 LLMConfig‑Gym，包含 4 种任务（模型架构、预训练超参、RL GRPO 调参、数据混合），超过 1M GPU‑小时实验结果，覆盖 3 级保真度。

**📈 对比分析**

与随机搜索、Top‑K warm‑start、MetaBO/NAP/FSBO 以及 OpenAI O4‑mini/Gemini/GPT‑5 的 AgentHPO 基线对比。结果显示，在预算 1–5 次实验内，AutoLLMResearch 在 4 项任务上均取得最低累积 regret，平均低于基线 30%–80%，并在规模化部署中显著降低 GPU 成本。

**⚠️ 局限性**

局限性：① 仍依赖大量离线实验数据；② 适用于离散配置空间，对连续或高维连续空间的推广待验证；③ 训练时间与算力较高，受限于大模型可用性；④ 在极端保真度不一致或对抗性训练样本下仍会出现迁移偏差。

---

## 168. Generative Diffusion Prior Distillation for Long-Context Knowledge Transfer

**arXiv ID:** 2605.11414 | [PDF](https://arxiv.org/pdf/2605.11414v1)

**作者:** Nilushika Udayangani `[一作]` (University of Melbourne), Marimuthu Palaniswami `[通讯]` (University of Melbourne)

**通讯引用:** 31046 | [OpenAlex ID](https://openalex.org/A5080554686)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种新的知识蒸馏框架GDPD，利用扩散模型作为生成先验，帮助从完整时间序列教师向部分序列学生迁移长上下文知识。

**💡 创新点**

将教师知识视为分布而非点，利用扩散模型在后验采样中提供动态、丰富且可信的教师信号，解决全到部分蒸馏的表示、分布和可信性问题。

**🔧 技术方法**

结合知识蒸馏、扩散模型、后验采样与引导采样、动态温度调度、学生-教师特征匹配等技术。

**📊 数据集**

在UCR、UEA、PhysioNet等多种单/多变量时间序列数据集上进行实验。

**📈 对比分析**

与传统logit/feature KD、RKD、VID等多种蒸馏方法比较，GDPD在多种早期截断、通道缺失、压缩、自蒸馏等场景下均取得显著提升，平均AUC-PRC和排名均优于对手。

**⚠️ 局限性**

需要额外的扩散模型训练与调度，且在极端教师弱监督或数据差异极大时性能下降仍有提升空间。

---

## 169. FedMM: Federated Collaborative Signal Quantization for Multi-Market CTR Prediction

**arXiv ID:** 2605.11433 | [PDF](https://arxiv.org/pdf/2605.11433v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 170. Internalizing Curriculum Judgment for LLM Reinforcement Fine-Tuning

**arXiv ID:** 2605.11235 | [PDF](https://arxiv.org/pdf/2605.11235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 171. An Executable Benchmarking Suite for Tool-Using Agents

**arXiv ID:** 2605.11030 | [PDF](https://arxiv.org/pdf/2605.11030v1)

**作者:** Zhiqing Zhong `[一作]` (Stevens Institute of Technology), Xiaodong Yu `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 931 | [OpenAlex ID](https://openalex.org/A5052001478)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可执行基准套件和证据接受门，整合 WebArena Verified、SWE‑Gym 切片与 MiniWoB++，明确区分工作负载、驱动和设置，并定义哪些运行结果可用于论文主张。

**💡 创新点**

创新点包括：① 明确的证据接受合同，使得工作负载、驱动和设置在评测中可追溯；② 跨工作负载共享适配器、事件模式和重放策略；③ 决策相关的证据门能够在不混入非论文数据的情况下直接影响系统评估结果。

**🔧 技术方法**

使用了事件日志、W3C Trace Context 追踪、可执行重放、驱动合同（包含 LLM、控制器、脚本化驱动）、vLLM/SGLang 后端、Qwen LLM、工具调用验证器、以及统一的报告与验证管道。

**📊 数据集**

基准数据集包括 WebArena Verified、SWE‑Gym（兼容 SWE‑bench 的代码/仓库工作负载）和 MiniWoB++ 微任务集。

**📈 对比分析**

评测通过收集 930 行论文证据，统计模型调用延迟、令牌成本、无效动作率、重放完整性等指标；在控制器选择实验中，清洁基线与中等负载的结果相反，证明证据门对系统决策具有决定性影响。

**⚠️ 局限性**

局限性：仅覆盖三类工作负载、指定驱动和设置；未提供完整 RL 训练或新代理策略；未对模型能力或后台优劣做广泛结论；仅在受限实验范围内展示决策逆转效果。

---

## 172. MaskTab: Scalable Masked Tabular Pretraining with Scaling Laws and Distillation for Industrial Classification

**arXiv ID:** 2605.11408 | [PDF](https://arxiv.org/pdf/2605.11408v1)

**作者:** Bo Zheng `[一作]` (Zhejiang University), Sheng Guo `[通讯]` (MyBank, Ant Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出MaskTab，一种针对工业级高维缺失率表格数据的统一预训练框架，利用可学习的缺失标记、双路混合预训练和MoE重构头实现高效特征建模；

**💡 创新点**

创新点在于将缺失视为信号，引入专用缺失标记并与随机遮盖同步训练，采用双路架构消除遮盖诱导的分布偏移，并用Mixture‑of‑Experts动态分配重构容量；

**🔧 技术方法**

使用可学习缺失标记、Transformer编码器、双路混合预训练（自监督+监督联合）、MoE重构头、知识蒸馏以及自适应遮盖率等技术；

**📊 数据集**

在公开TabReD基准（8个分类/回归任务）和私有CreditRisk（2500维、6.4e5标注/1.3e7无标注）数据集上进行实验；

**📈 对比分析**

与XGBoost、LightGBM、CatBoost、FT‑Transformer、TransTab等基线相比，MaskTab‑Base在CreditRisk上AUC提升5.04%、KS提升8.28%，蒸馏版在保留500可解释特征时仍比XGBoost提升2.55% AUC、4.85% KS，并显著改善时间漂移鲁棒性；

**⚠️ 局限性**

实验规模有限，未进行完整多维交叉的联合尺度探索，且对更丰富的文本/时间序列特征的适应性仍待验证。

---

## 173. Sensor Design for Accuracy-Bounded Estimation via Maximum-Entropy Likelihood Synthesis

**arXiv ID:** 2605.11120 | [PDF](https://arxiv.org/pdf/2605.11120v1)

**作者:** Raktim Bhattacharya `[一作]` (Texas A&M University), Raktim Bhattacharya `[通讯]` (Texas A&M University)

**通讯引用:** 1526 | [OpenAlex ID](https://openalex.org/A5006156978)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计一种逆向感知框架：给定估计误差预算，合成最小信息量的测量似然，并通过两层架构将其映射到传感器布置。

**💡 创新点**

创新点在于把分布距离约束与最大熵先验合成相结合，实现无前向测量模型的似然生成，并提出将信息设计与物理传感器配置解耦的两层感知‑估计架构。

**🔧 技术方法**

使用最大熵、KL极小化、Wasserstein、MMD、f‑散度、矩约束等分布相似度；粒子滤波实现；Sinkhorn、L‑BFGS 等优化算法；参数化似然提炼为高斯混合或指数族。

**📊 数据集**

使用人工合成的一维高斯、混合高斯及多峰先验与目标分布；无真实数据集，仅采用仿真数据。

**📈 对比分析**

与四种误差度量（Wasserstein、MMD、矩、χ²）以及传统粒子滤波比较；实验显示单峰时差异不大，多峰时度量选择决定更新形状；感知实现可逼近理想似然，传感器数量增大可实现零 realizability gap。

**⚠️ 局限性**

局限性包括：仅在一维实验、需大量粒子、对高维扩展性未知、目标后验可实现性假设、需人工设定误差阈值，以及在现实非可实现传感器场景下的部署挑战。

---

## 174. LDDR: Linear-DPP-Based Dynamic-Resolution Frame Sampling for Video MLLMs

**arXiv ID:** 2605.11477 | [PDF](https://arxiv.org/pdf/2605.11477v1)

**作者:** Jingfeng Chen `[一作]` (Carnegie Mellon University), Bhuwan Dhingra `[通讯]` (Duke University)

**通讯引用:** 3068 | [OpenAlex ID](https://openalex.org/A5055033421)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无训练、插件式的视频帧采样框架LDDR，能在固定视觉-token预算下为多模态大型语言模型挑选最具信息量且多样化的帧，并动态分配分辨率。

**💡 创新点**

创新点在于：① 用线性特征空间的DPP实现全局查询感知的帧选择，时间复杂度降至线性；② 引入Group DPP重要性度量来指导每帧的视觉-token分配，实现预算感知的动态分辨率。

**🔧 技术方法**

使用技术包括CLIP风格的视觉‑文本嵌入、线性特征空间Determinantal Point Process (DPP)、Group DPP重要性评估以及基于预算的动态分辨率分配。

**📊 数据集**

实验数据集涵盖四大视频理解基准：Video‑MME、LongVideoBench、LVBench、MLVU，并在多种开源/闭源MLLM（Qwen2.5‑VL、Qwen3‑VL、LLaVA‑OneVision、InternVL3）上评测。

**📈 对比分析**

与均匀采样、AKS、Q‑frame、FOCUS、MDP3等现有采样方法对比，LDDR在所有模型和数据集上均显著提升性能，平均在受限预算下提高2.5分，在高预算下提升1.6分；在长视频场景下性能提升尤为显著。

**⚠️ 局限性**

局限性包括：① 对动态分辨率的支持仅限于支持多分辨率输入的模型；② 在预算充足时，动态分辨率反而可能带来轻微性能下降；③ 仍依赖CLIP类视觉‑文本对齐模型，若对齐不足则采样效果受限。

---

## 175. Feedback Set Problems on Bounded-Degree (Planar) Graphs

**arXiv ID:** 2605.11407 | [PDF](https://arxiv.org/pdf/2605.11407v1)

**作者:** Tian Bai `[一作]` (University of Electronic Science and Technology of China), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1496 | [OpenAlex ID](https://openalex.org/A5033729619)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了反馈集合问题在最大度有限的有向图与平面有向图上的复杂度分类，并给出了连通反馈顶点集在平面无向图最大度为三时的NP‑完全性

**💡 创新点**

完整绘制了度限制与平面性下的复杂度图谱，证明在度为三的有向图以及平面有向图中若每个顶点入度或出度≤1则可多项式求解，反之为NP‑完全；同时指出连通反馈顶点集在平面无向图最大度三时仍是NP‑完全

**🔧 技术方法**

使用了“doubling”“splitting”两种图变换、特殊的平面嵌入（bipolar 与 irregular）以及 Speckenmeyer 风格的低度 gadget 进行归约构造，进一步利用入度/出度分布对问题进行细化

**📊 数据集**

本研究为理论性工作，未使用实验数据集

**📈 对比分析**

通过多步归约与构造证明了算法复杂度，未进行实验比较，结果为理论上的NP‑完全与多项式可解的边界判定

**⚠️ 局限性**

仅给出了决定性分类，未提供近似或参数化算法，且未讨论更高度或非平面图的进一步细化

---

## 176. ExploitGym: Can AI Agents Turn Security Vulnerabilities into Real Attacks?

**arXiv ID:** 2605.11086 | [PDF](https://arxiv.org/pdf/2605.11086v1)

**作者:** Zhun Wang `[一作]` (University Of California Berkeley), Dawn Song `[通讯]` (University Of California Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并发布了一个规模达898个实例、涵盖用户空间程序、V8引擎和Linux内核、支持可调防御的可复现漏洞利用基准，并用它评估前沿AI模型的利用能力。

**💡 创新点**

首次构建面向AI漏洞利用的完整、真实且可复现的基准，集成了agent-as-judge来验证是否真正利用指定漏洞，且提供了跨三大软件层的多样化防御设置。

**🔧 技术方法**

利用大语言模型（如Anthropic、OpenAI GPT-5.5）与自适应代理框架，结合容器化环境、可调节的安全防御、可信访问、工具调用等技术进行评估。

**📊 数据集**

使用来自OSS‑Fuzz、Google V8 issue tracker、kernelCTF和syzbot的真实漏洞构建了898个实例，包括PoV、漏洞描述、补丁及构建脚本。

**📈 对比分析**

在统一的2小时时间窗口下对每个模型跑一次，记录成功、flag捕获与judge判定；最强模型Anthropic 157次、OpenAI GPT‑5.5 120次；6小时后覆盖239次；不同模型在不同域表现互补；开启防御后成功率下降但仍有非零值。

**⚠️ 局限性**

仅覆盖Linux、V8、用户空间，不含Windows、iOS、Android；只评估完整代码执行，未覆盖读取/写入原语等；安全拒绝或非可利用漏洞导致失败；时间与成本限制未能充分挖掘；模型偏好单一提示；缺少完整ground truth。

---

## 177. Epistemic Uncertainty for Test-Time Discovery

**arXiv ID:** 2605.11328 | [PDF](https://arxiv.org/pdf/2605.11328v1)

**作者:** Kainat Riaz `[一作]` (Stanford University), John M. Cioffi `[通讯]` (Stanford University)

**通讯引用:** 28169 | [OpenAlex ID](https://openalex.org/A5029146230)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用LoRA适配器集成与互信息探索信号提升LLM在科学发现中的最大奖励

**💡 创新点**

首次在单模型RL中引入集成误差与核范数正则化提取词级互信息作为探索驱动

**🔧 技术方法**

低秩LoRA适配器集成、贝叶斯互信息、核范数正则化、温度耦合奖励

**📊 数据集**

四个科学发现基准（AC1、AC2、CP26、Erdős）

**📈 对比分析**

相较基线RL，UGTT在3/4任务上提升最大奖励，保持1.1-1.7比特族多样性，并更快收敛

**⚠️ 局限性**

仅在Qwen3-8B、单GPU、单种子实验，参数量成正比放大，超参数经验性选择，需在其他基模型和大规模实验验证

---

## 178. Oversmoothing as Representation Degeneracy in Neural Sheaf Diffusion

**arXiv ID:** 2605.11178 | [PDF](https://arxiv.org/pdf/2605.11178v1)

**作者:** Arif Dönmez `[一作]` (IUF -- Leibniz Research Institute for Environmental Medicine), Katharina Koch `[通讯]` (IUF -- Leibniz Research Institute for Environmental Medicine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过将神经网络中的细胞层构造为图的入射双向 quiver 表示，阐释了神经结构扩散中的过平滑现象为表示退化，并提出基于几何不变理论的 moment‑map 正则化及矩形 stalk 结构来缓解该退化；

**💡 创新点**

创新点在于将过平滑视为 quiver 表示的直接和分解导致的全局节段退化，利用 King 稳定性与 moment‑map 约束构建可学习的稳定性参数，揭示了等维度结构下的稳定性墙并证明矩形结构可打破该障碍；

**🔧 技术方法**

使用的技术包括：细胞层 (cellular sheaf)、入射 quiver 表示、Krull–Schmidt 定理、King 稳定性、Kempf–Ness moment‑map、梯度下降隐式偏差分析；

**📊 数据集**

实验数据集为 WebKB 的三种异质图（Texas、Cornell、Wisconsin）以及在 Wisconsin 上的深度可扩展实验；

**📈 对比分析**

与基准 Gen‑NSD 进行比较，结果表明在矩形架构下使用 ThetaMM 或 CentMM 正则化可在部分数据集上提升 1–2% 甚至更高，表现出数据集依赖的结构性优势；

**⚠️ 局限性**

局限性包括：正则化并非严格保证 GIT 半稳定；仅抑制了 trivial 子表示的退化，无法消除所有过平滑模式；改进效果高度依赖数据集与架构，且在极深网络下仍会出现数值不稳定。

---

## 179. Control Charts for Multi-agent Systems

**arXiv ID:** 2605.11135 | [PDF](https://arxiv.org/pdf/2605.11135v1)

**作者:** Hayden Helm `[一作]` (Helivan), Brandon Duderstadt `[通讯]` (Calcifer Computing)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于 iso-mirror 的自适应控制图，用于监测学习型多智能体系统的全局状态，并研究了其对动态环境和慢速攻击的检测能力。

**💡 创新点**

创新点在于将 Temporal Data Kernel Perspective Space (TDKPS) 与 iso-mirror 结合，构造系统级低维表示；在此基础上引入自适应控制图以应对自然漂移；理论证明任何自适应监测方案均易受慢速失效攻击，揭示学习-安全权衡。

**🔧 技术方法**

核心技术包括：TDKPS 低维嵌入；iso-mirror 对系统状态的空间表示；Shewhart 控制图的自适应窗口更新；利用多元统计推断评估偏移和误报；以及对失效代理的离散化策略。

**📊 数据集**

使用仿真数据：由 100 只大语言模型驱动的代理组成的完整通信网络，分为静态/动态记忆两种设置，生成固定与环境查询，模拟失效代理的慢速/快速攻击；不依赖真实世界公开数据集。

**📈 对比分析**

对比方法：固定控制图 vs 自适应控制图；对失效代理的即时 vs 逐步失效。结果显示：自适应控制图显著降低因自然漂移产生的误报，但对慢速失效攻击几乎不可检出；固定控制图能检测慢速攻击但误报率高。实验中，动态记忆代理在环境查询上保持 60–70% 的准确率，而静态记忆仅 20%。

**⚠️ 局限性**

局限性：仅在完整连通图下实验，未考虑稀疏/动态拓扑；攻击模型仅为单一失效代理，未覆盖协同或自适应攻击；代理架构简化，仅包含检索+LLM；监测仅基于黑盒查询响应，缺乏内部状态观测；未对多智能体的个体层面控制图做深入分析。

---

## 180. Byzantine Consensus in Directed Graphs with Message Authentication

**arXiv ID:** 2605.11309 | [PDF](https://arxiv.org/pdf/2605.11309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 181. MT-JailBench: A Modular Benchmark for Understanding Multi-Turn Jailbreak Attacks

**arXiv ID:** 2605.11002 | [PDF](https://arxiv.org/pdf/2605.11002v1)

**作者:** Xinkai Zhang `[一作]` (University of California Berkeley), N. Benjamin Erichson `[通讯]` (International Computer Science Institute)

**通讯引用:** 1194 | [OpenAlex ID](https://openalex.org/A5007032334)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MT‑JailBench，一个可模块化、可控制实验条件的多轮 jailbreak 评估框架，并用它对现有多轮 jailbreak 方法进行系统比较与组件拆解。

**💡 创新点**

创新点在于：①将多轮 jailbreak 分解为评估函数、攻击策略、提示生成、提示精炼和流程控制五个可替换模块；②通过固定资源预算和评估标准实现对攻击机制的可比性；③揭示提示生成是性能差异的主要来源；④通过组件重组（CrescendoX）展示了组件级分析的可预测性。

**🔧 技术方法**

主要技术包括黑盒文本交互、评估函数/判定器（Binary、Score 等）、在线/离线提示生成策略、基于评估反馈的提示精炼、流程控制策略、以及对已有多轮 jailbreak 的标准化实现。

**📊 数据集**

使用了 HarmBench 数据集（159 条有害行为）以及多种 LLM（GPT‑4o、GPT‑5、Llama‑3‑70B、Llama‑3‑8B 等）进行实验。

**📈 对比分析**

通过在统一资源预算（最多10次交互、每轮最多3次重试）和统一判定器（跨验证后采用三位裁判一致判定）的条件下，比较 Crescendo、CoA、FITD、ActorBreaker、XTeaming 的 ASR。结果显示资源预算与评估器对 ASR 影响巨大；在控制条件下，提示生成贡献最大，精炼与流程控制提供中等提升。将 Crescendo 的提示生成与 XTeaming 的精炼、流程控制组合得到 CrescendoX，实验表明它在 21 个目标模型中 19 个获得最高 ASR，明显优于原始攻击。

**⚠️ 局限性**

局限性包括：评估仅在黑盒文本交互、有限资源预算和特定评估标准下进行；仅覆盖 HarmBench 的有害行为集合；判定器为基于 LLM 的评分，可能与人类评估存在偏差；未考虑多模态或更复杂的防御机制，且对商业 LLM 的安全策略（如 GPT‑5 的安全分类器）影响难以分离。

---

## 182. ClinicalBench: Stress-Testing Assertion-Aware Retrieval for Cross-Admission Clinical QA on MIMIC-IV

**arXiv ID:** 2605.11143 | [PDF](https://arxiv.org/pdf/2605.11143v1)

**作者:** Alex Stinard `[一作]` `[通讯]` (University of Central Florida), Alex Stinard (University of Central Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为EpiKG的患者级知识图谱，并在此基础上实现了一个基于意图感知的检索-生成系统C4g，用于在真实MIMIC-IV电子健康记录上进行跨入院临床问答的检索可信度评估。

**💡 创新点**

创新点在于：①保持并传递临床文本中的断言标签（肯定、否定、可能等）和时间标签；②将检索策略与问题意图（如当前状态、历史、变更）对齐；③提出了ClinicalBench这一400题、43人、9类断言敏感的检索可信度压力测试，并公开所有评测数据。

**🔧 技术方法**

技术包括：1) 规则+机器学习的断言与时间标签化；2) 双向BFS图遍历与OMOP关系查询；3) Dense-RAG（Contriever）与KG-RAG（带断言与时间的知识图检索）；4) 关键词与oracle意图分类器；5) 通过Exact McNemar、Bootstrap CI等统计方法评估性能。

**📊 数据集**

使用了MIMIC-IV 3.1版本的真实电子健康记录（共43名患者，含两入院记录）以及公开的ClinicalBench问题集和对应的参考答案。

**📈 对比分析**

方法比较：在六款LLM（Opus 4.6、GPT-OSS 20B、MedGemma 27B、Gemma 4 31B、MedGemma 1.5 4B、Qwen 3.5 35B）上对C1（LLM单独）与C4g（意图感知KG-RAG）进行对比。主实验（leave‑author‑out paired McNemar）在50个项目上得到+22.0个百分点（95% CI [+5.1,+31.5]，p=0.0192），keyword敏感性得到+39.5个百分点，oracle意图得到+43.1个百分点，所有模型均显示显著提升。外部医生三评判中，C4g比C1多出+24个百分点。

**⚠️ 局限性**

局限性包括：①数据集仅为单中心MIMIC-IV，缺乏多中心泛化验证；②参考答案存在56%缺陷，需要人工纠正；③意图分类器以关键词为主，精度有限；④统计方法受样本量和聚类Bootstrap的可靠性限制；⑤作者与评测人员重叠带来的潜在循环偏倚。

---

## 183. TRACE: Temporal Routing with Autoregressive Cross-channel Experts for EEG Representation Learning

**arXiv ID:** 2605.11380 | [PDF](https://arxiv.org/pdf/2605.11380v1)

**作者:** Fan Ma `[一作]` (Yale University), Hua Xu `[通讯]` (Yale University)

**通讯引用:** 53820 | [OpenAlex ID](https://openalex.org/A5101613292)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种自回归EEG预训练框架Trace，能在不同通道数、采样方式和时间尺度下学习可迁移的脑电表示。

**💡 创新点**

创新点包括：①跨通道时间路由（Cross‑Channel Temporal Routing）使所有同一时刻通道共享同一专家，保持跨通道一致性；②Temporal Routing MoE (TR‑MoE)块融合因果时空注意力和共享专家FFN；③多源异构EEG预训练不需投影到统一montage，支持不同通道布局；④多视界自回归预训练目标，预测未来多个时间窗口。

**🔧 技术方法**

技术细节涵盖：时频嵌入、跨通道时空注意力、TemporalFormer（查询‑注意力）生成路由上下文、CTR‑FFN专家选择、Huber损失与专家负载平衡正则。

**📊 数据集**

预训练数据来自Temple University Hospital EEG Corpus (TUEG)、HBN 与多项任务相关EEG（包含16–128通道、4–30秒窗口）共约150万段；下游任务涵盖睡眠分期、情绪识别、运动想象、癫痫检测、想象语音与事件分类等八个公开数据集。

**📈 对比分析**

与任务专用模型（EEGNet、EEGConformer）以及现有EEG基础模型（BIOT、LaBraM、CBraMod、CodeBrain）对比，Trace在多任务、不同转移域（见域仅作无标签预训练、完全未知数据集）中多项指标（Balanced Accuracy、AUC‑PR、AUROC、F1‑W）均达到或接近SOTA，特别在想象语音和情绪识别上显著领先。

**⚠️ 局限性**

局限性：仅评估了分类任务，未验证回归或预警问题；预训练语料可能缺乏部分硬件、极少见疾病或大规模异构布局；模型参数量和训练成本因专家池规模而升高；缺乏临床可解释性、公平性和隐私保护研究。

---

## 184. Steerable Neural ODEs on Homogeneous Spaces

**arXiv ID:** 2605.11133 | [PDF](https://arxiv.org/pdf/2605.11133v1)

**作者:** Emma Andersdotter `[一作]` (Umea University), Fredrik Ohlsson `[通讯]` (Umea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

在齐性空间M=G/H上提出一种神经ODE模型，利用主连接实现特征向量场的平行输送，使得点的流动与特征的旋转同步；

**💡 创新点**

通过将特征视为关联向量束的截面，并将平行输送与主连接耦合，首次在连续时间框架下给出可全局G等变的可导轨道，并用Wang定理给出G-不变连接的完整分类；

**🔧 技术方法**

使用微分几何工具（主束、关联束、平行输送、G-不变向量场、Wang定理）以及神经ODE和连续归一化流的框架；

**📊 数据集**

论文主要为理论研究，未使用具体数据集；

**📈 对比分析**

未给出实验比较或性能指标，主要通过理论证明展示等变性与可逼近性；

**⚠️ 局限性**

局部截面依赖、图切换与数值实现细节缺失，对非齐性空间和近似对称性的推广有限。

---

## 185. LatentHDR: Decoupling Exposure from Diffusion via Conditional Latent-to-Latent Mapping for Text/Image-to-Panoramic HDR

**arXiv ID:** 2605.11115 | [PDF](https://arxiv.org/pdf/2605.11115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 186. Towards Scalable Persistence-Based Topological Optimization

**arXiv ID:** 2605.10996 | [PDF](https://arxiv.org/pdf/2605.10996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 187. DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated AF-Pipe Parallelism

**arXiv ID:** 2605.11005 | [PDF](https://arxiv.org/pdf/2605.11005v1)

**作者:** Zhichen Zeng `[一作]` (ByteDance Seed), Ziheng Jiang `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DisaggMoE 系统，将注意力层和 FFN 层拆分到独立设备组，使用 AF-Pipe 多阶段流水线和 adaptive worker 分配来提升大规模 MoE 训练效率。

**💡 创新点**

创新点包括：①分离组件放置，②把全局交互视为管道阶段并采用 many-to-many 通信，③用计算-通信屋顶线模型动态分配 GPU 与 NIC 资源，④兼顾大规模模型和长序列训练。

**🔧 技术方法**

采用 MoE（专家并行）+ 数据并行 + 管道并行 + many-to-many 交换 + GPUDirect/GPUCopy + Python/C++ 实现 + Megatron-LM 3D 并行基础 + Compute-Communication Roofline 模型 + MILP 搜索等技术。

**📊 数据集**

在预训练 LLM 数据上进行评估，使用 DeepSeek-MoE、GPT-OSS、Qwen3 三个模型，序列长度 4K–32K，部署在 H800 GPU 集群上。

**📈 对比分析**

与 Megatron-LM、Tutel、Comet、DualPipe 等基线对比，在 8–16 节点 H800 上，DisaggMoE 达到 1.81 倍 Megatron‑1F1B 速度，1.34 倍其他 MoE 重叠系统；通信占比显著降低，整体吞吐量提升。

**⚠️ 局限性**

限制：仅支持预训练式固定形状（序列长度、批次），无法处理动态形状工作负载；使用单一流水线深度，非异步深度可能导致 OOM；需要预先分配资源，难以在线自适应；对硬件（如 NIC 带宽）高度依赖。

---

## 188. FragBench: Cross-Session Attacks Hidden in Benign-Looking Fragments

**arXiv ID:** 2605.11029 | [PDF](https://arxiv.org/pdf/2605.11029v1)

**作者:** Astha Mehta `[一作]` (SPAR), Linh Le `[通讯]` (Mila)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建FragBench数据集，生成跨会话碎片化攻击，利用强化学习重写器使其通过单轮安全判别器，并在MCP沙盒中验证执行；随后使用图神经网络在用户层面检测跨会话组合的恶意行为。

**💡 创新点**

首次提出跨会话碎片攻击范式，结合RL重写、MCP执行验证和图级检测捕获单轮安全过滤器无法发现的组合恶意信号；同时提供开放的生成器、沙盒和检测器。

**🔧 技术方法**

使用强化学习重写器（基于Claude Sonnet 4.6/Opus 4.6），单轮安全判别器，Model Context Protocol（MCP）沙盒执行，Docker Compose托管多工具服务器；在检测端采用GCN、GraphSAGE、GAT、GIN等图神经网络及SVM、MLP、GBT等经典ML基线。

**📊 数据集**

基于24个真实网络攻击报告的碎片化数据（共25,400片段，约10.6片段/变体），以及匹配的合规benign覆盖会话数据，构成训练与测试集。

**📈 对比分析**

在所有24个攻击案例上对比四种GNN和三种经典基线，聚合F1从0.878到0.956，单轮判别器仅接近随机；攻击重写器在10轮内将通过率从约60%提升至近100%，验证了跨会话检测的重要性。

**⚠️ 局限性**

benign数据可能与恶意在统计上仍存在差异；在正例稀疏的场景下F1显著下降；数据生成依赖Claude模型，需额外成本；开放数据受限于安全审查，限制了复现与进一步研究的可访问性。

---

## 189. PIVOT: Bridging Planning and Execution in LLM Agents via Trajectory Refinement

**arXiv ID:** 2605.11225 | [PDF](https://arxiv.org/pdf/2605.11225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 190. 20/20 Vision Language Models: A Prescription for Better VLMs through Data Curation Alone

**arXiv ID:** 2605.11405 | [PDF](https://arxiv.org/pdf/2605.11405v1)

**作者:** Siddharth Joshi `[一作]`, Matthew Leavitt `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并使用一套完整的端到端数据清洗与增强流水线，对单图像 VLM 训练数据进行去重、过滤、分布匹配以及合成数据生成，随后在固定架构、学习速率与算力下训练 Qwen3 语言模型与 SigLIP2 视觉编码器的 1B、2B、4B 规模模型；

**💡 创新点**

将数据视为首要设计变量，将多种清洗与增强手段融合为统一流程，证明仅通过数据预处理即可在大规模 VLM 训练中获得接近或超过大量后期 fine‑tuning 与 RLHF 的性能提升，显著缩短训练算力需求；

**🔧 技术方法**

采用多模态去重、双模过滤、分布匹配的混合设计、任务无关与任务特定合成数据生成，以及高分辨率多块切片编码和动态分块；

**📊 数据集**

以 MAmmoTH‑VL‑12M 单图像子集（约 10M 条样本）为原始数据，经过流水线处理后生成 DatologyAI‑curated 语料；

**📈 对比分析**

与相同架构、同等训练算力的未清洗基线、以及已做多阶段后处理（SFT、RLHF、RLVR）的公开大模型（Qwen3‑VL、InternVL 等）对比。结果显示：在 20‑评测公共 VLM 基准上 2B 模型提升 11.7pp，3‑评测指标提升 11.3pp；在单图像清洗下对多图推理任务 BLINK 的 OOD 结果提升 3.09pp；在 1B、2B、4B 规模下均实现“Pareto”支配——在保持或提高准确率的同时，推理 FLOPs 降低 24–45%；整体训练算力比已做大量后处理的前沿模型低约 150×；

**⚠️ 局限性**

仅针对单图像数据进行清洗，缺乏针对多图、视频等更复杂多模态场景的验证；未结合 SFT、RLHF 等后处理技术，无法证明其在更完整的训练流程中的进一步协同效果；对不同模型架构的可迁移性仍需进一步评估。

---

## 191. Test-Time Compute for Dense Retrieval: Agentic Program Generation with Frozen Embedding Models

**arXiv ID:** 2605.11374 | [PDF](https://arxiv.org/pdf/2605.11374v1)

**作者:** Han Xiao `[一作]` `[通讯]` (Jina Ai), Han Xiao (Jina Ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过一个无监督的 LLM 生成程序搜索循环，探索在冻结的单向量嵌入模型上进行测试时计算干预的方法，最终发现了一种基于软最大化加权质心的查询更新（SoftCentroid）以及一个轻量级的 BM25 词频融合增益。

**💡 创新点**

创新点在于：① 证明即便是从大模型蒸馏出的嵌入模型，测试时计算也能显著提升；② 通过 LLM 生成的搜索框架发现了“质心替换”这一结构化查询更新，而不是传统的随机采样或多模态扩展；③ 提供了可复现的循环、日志与程序注册表，能在不同模型/数据集上重新生成最优前沿。

**🔧 技术方法**

使用技术包括：LLM 生成 Python 程序、GPU 加速评估、Pareto 前沿更新、SoftCentroid 公式（温度调节的 softmax 加权质心）、BM25 词频融合、ANN 搜索（HNSW）以及多任务评估工具 MTEB。

**📊 数据集**

使用数据集：BEIR 13 任务（NFCorpus、SciFact、ArguAna、FiQA‑2018 等）以及 held‑out SCIDOCS、Touche2020；实验规模覆盖七种嵌入模型家族，参数从 33M 到 335M。

**📈 对比分析**

比较方法：对比原始单向量余弦检索、Rocchio、SoftCentroid、BM25+RM3 等；统计显著性通过配对 bootstrap（10k 次）检验。SoftCentroid 在 NFCorpus、SciFact、ArguAna 上分别提升约 2.4、1.2、7.3 nDCG@10，且在 7 个模型家族全 BEIR 规模上保持正向提升；与传统 PRF 基线相比，SoftCentroid 的提升更大且无额外训练。

**⚠️ 局限性**

局限性：① 只在单向量冻结模型上工作，无法突破余弦几何上限；② 仅适用于无 LLM 推理的 “无训练” 场景，对需要生成式推理或多向量模型的任务不适用；③ 对于极端 QA 异构任务（NQ、HotpotQA）纯质心更新效果有限，需额外词频融合；④ 发现的前沿在不同任务集或模型家族上仍需进一步验证，且对 ANN 参数的鲁棒性虽高但在极低资源环境下可能受限。

---

## 192. Rank Is Not Capacity: Spectral Occupancy for Latent Graph Models

**arXiv ID:** 2605.11142 | [PDF](https://arxiv.org/pdf/2605.11142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 193. An Empirical Study of Automating Agent Evaluation

**arXiv ID:** 2605.11378 | [PDF](https://arxiv.org/pdf/2605.11378v1)

**作者:** Kang Zhou `[一作]` (AWS AI Labs), Lin Lee Cheong `[通讯]` (AWS AI Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EvalAgent，自动从代理源码与执行轨迹生成可执行评估代码与报告，解决评估标准设计与实现的全流程问题。

**💡 创新点**

创新点包括：①把评估知识拆解为可复用的“评估技能”包；②构建基于轨迹的六阶段评估管道；③推出 AgentEvalBench 20个多框架、多领域代理的评估基准；④设计可由机器与人工对比的元评估框架。

**🔧 技术方法**

采用 Claude Code 与 Claude/Haiku/ Sonnet 语言模型，使用 OpenTelemetry 轨迹采集、DeepEval 评估指标、Context7 实时 API 文档检索，并通过评估技能进行流程引导。

**📊 数据集**

数据集：AgentEvalBench——20个真实代理（9框架、14领域、3复杂度级别），每个代理配 5 场景测试和 2 类需求。

**📈 对比分析**

比较方法：采用元评估的配对对比与加权维度（需求满足、指标相关性、代码质量、规划质量、规划-实现一致性）以及 Eval@1 评估成功率。实验结果显示 EvalAgent 在所有基准上获得 84‑100% 的赢/平率，Eval@1 达到 62.5‑65%，并在人工专家对比中获得 79.5% 的偏好。

**⚠️ 局限性**

局限性：仅评估 20 个代理，未覆盖体态、多模态等类型；实验仅使用 Claude 系列模型；Eval@1 仍低于 70%，需要手动调试；元评估的主观性和可靠性仍有待提升。

---

## 194. Debiasing Message Passing to Mitigate Popularity Bias in GNN-based Collaborative Filtering

**arXiv ID:** 2605.11145 | [PDF](https://arxiv.org/pdf/2605.11145v1)

**作者:** Md Aminul Islam `[一作]` (University of Illinois Chicago), Elena Zheleva `[通讯]` (University of Illinois Chicago)

**通讯引用:** 2045 | [OpenAlex ID](https://openalex.org/A5071079350)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出DPAA框架，直接在GNN聚合中加入交互级权重和层级权重，消除流行度放大导致的偏差；

**💡 创新点**

创新点包括① embedding‑aware交互权重，使用预训练与当前模型的平滑过渡；② 层级权重放大高阶邻域信号；③ 理论证明和半合成可控偏差数据的系统评估；

**🔧 技术方法**

使用Graph Neural Network（LightGCN）、BPR损失、Inverse Interaction Weight、层级权重、残差连接、semi‑synthetic Zipf采样、对比IPW/因果/正则化等技术；

**📊 数据集**

实验基于三大真实数据集（Coat、KuaiRec、Yahoo! R3）以及基于KuaiRec的semi‑synthetic click数据；

**📈 对比分析**

与APDA、NAVIP、DAP、PPAC、MACR、CVIB、IPS、SAM‑REG、LightGCN等基线在Recall@20、NDCG@20、HR@20上进行all‑ranking评估，DPAA在所有数据集上均优于最佳基线，尤其在高偏差数据集KuaiRec上提升显著；

**⚠️ 局限性**

仅针对流行度偏差，未覆盖选择/位置等其他偏差；需要预训练模型和超参数调优，轻度偏差场景可能略有性能下降；

---

## 195. PG-3DGS: Optimizing 3D Gaussian Splatting to Satisfy Physics Objectives

**arXiv ID:** 2605.11266 | [PDF](https://arxiv.org/pdf/2605.11266v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 196. Human-AI Productivity Paradoxes: Modeling the Interplay of Skill, Effort, and AI Assistance

**arXiv ID:** 2605.11350 | [PDF](https://arxiv.org/pdf/2605.11350v1)

**作者:** Ali Aouad `[一作]` (Massachusetts Institute of Technology), Huiying Zhong `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文构建了一个人机协作模型，分析了生成式人工智能（GenAI）在生产率和技能发展中的多重影响，并通过引入技能演化、AI不可靠性与AI素养等机制，揭示了生产率悖论与技能极化的产生条件。

**💡 创新点**

创新点在于：1) 首次用理论模型统一解释了AI辅助导致生产率下降的两种机制——技能退化与努力被摄取；2) 引入绝对风险厌恶（IARA/DARA）来刻画不可靠AI对生产率的非线性影响；3) 通过AI素养的贝叶斯信号模型揭示了多模态技能分布与马太效应的关联，提供了关于AI教育和治理的政策建议。

**🔧 技术方法**

主要技术手段包括：基于出生-死亡链的技能演化模型、凸优化求解人类努力选择、生产函数的绝对风险厌恶分析、贝叶斯信号跟随者（Bayesian signal follower）模型、稳态分布推导及其非单峰性分析。

**📊 数据集**

本研究为理论推导，不使用具体实验数据或公开数据集。

**📈 对比分析**

论文没有开展实验或数值仿真比较，所有结论均基于数学证明；因此不存在传统意义上的性能评估。

**⚠️ 局限性**

局限性包括：模型假设人类为短视效用最大化者、线性努力成本、对AI可靠性和素养的简化形式、缺乏对不同任务类型的异质性检验，以及未考虑宏观经济后果和制度约束。

---

## 197. CATS: Cascaded Adaptive Tree Speculation for Memory-Limited LLM Inference Acceleration

**arXiv ID:** 2605.11186 | [PDF](https://arxiv.org/pdf/2605.11186v1)

**作者:** Yuning Han `[一作]` (University of Florida), Jingwei Sun `[通讯]` (University of Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对内存受限边缘设备的自回归推理加速框架CATS（Cascaded Adaptive Tree Speculation），通过三阶段流水线（draft、shallow‑verify、target‑verify）实现自省式解码，最大化token接受率并提升推理吞吐量；

**💡 创新点**

创新点在于：①将验证过程分层并与设备DRAM容量自适应；②在保持设备内存占用等同于目标模型的前提下，通过浅层验证器将权重量级分摊到多层，显著降低每token的flash↔DRAM流量；③采用Reduced KL Loss聚焦高概率token以提升子网络的推理质量；

**🔧 技术方法**

技术包括：自回归解码、模型裁剪与层级划分、流水线调度、树结构验证、注意力掩码编码、Distillation+Reduced KL Loss、对照实验与BPT（Bytes per Token）分析；

**📊 数据集**

使用的评测数据集有：Spec-bench、MT-bench、GSM8K、Alpaca、HumanEval；模型覆盖Vicuna‑7B/13B、LLaMA‑2‑7B/13B；

**📈 对比分析**

与REST、Lookahead、Kangaroo、Medusa、EAGLE等现有自省式解码器比较，在边缘设备（Jetson AGX Orin）上，CATS在五大基准上实现最高5.08×的wall‑clock加速，平均接受token数提升至约5.1，且与基线模型在质量评测（MT‑Bench GPT‑4o）上保持同等或更高分数；

**⚠️ 局限性**

局限性包括：仅验证7B–13B规模模型；需先进行adapter的distillation训练；三阶段设计依赖Transformer层对齐，难以直接迁移至state‑space等新架构。

---

## 198. Variational Linear Attention: Stable Associative Memory for Long-Context Transformers

**arXiv ID:** 2605.11196 | [PDF](https://arxiv.org/pdf/2605.11196v1)

**作者:** Vishal Pandey `[一作]` (Independent Researcher), Gopal Singh `[通讯]` (Metriqual)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Variational Linear Attention（VLA），一种在保持线性时间复杂度的同时，通过自适应惩罚矩阵实现残差写入的注意力机制，显著抑制了传统线性注意力中的内存累积干扰。

**💡 创新点**

核心创新在于将注意力内存更新视为在线正则化最小二乘问题，利用 Sherman‑Morrison 递推维护逆惩罚矩阵；通过对写入方向和键进行单位归一化，证明递推雅可比矩阵谱半径恒为 1，保证梯度不爆炸且状态自限。

**🔧 技术方法**

采用在线正则化最小二乘框架、Sherman‑Morrison rank‑1 更新、单位长度归一化、可学习的惩罚方向、Triton 融合内核实现高效 GPU 加速。

**📊 数据集**

实验使用合成的 Copy 任务和多查询关联回忆（MQAR）任务，分别在不同关键-值对数量和序列长度下评估模型。

**📈 对比分析**

与 Softmax、标准线性注意力和 DeltaNet 在相同 Transformer 结构下比较；VLA 在 MQAR 任务中在容量边界（d_h=32）内保持 100% 精确匹配，状态 Frobenius 范数比线性注意力低 100×；在 GPU 上 Triton 核心实现比纯 Python 速度快 14×，在约 43k tokens 时低于 Softmax 的延迟。

**⚠️ 局限性**

限制包括：Sherman‑Morrison 递推带来常数因子开销（纯 Python 约 3×慢），单头容量有限（最多 d_h 关联），且仅在合成任务上验证，尚未在真实语言建模或下游任务上展示效果。

---

## 199. Sampling More, Getting Less: Calibration is the Diversity Bottleneck in LLMs

**arXiv ID:** 2605.11128 | [PDF](https://arxiv.org/pdf/2605.11128v1)

**作者:** Amin Banayeeanzade `[一作]` (University of Southern California), Sai Praneeth Karimireddy `[通讯]` (University of Southern California)

**通讯引用:** 2212 | [OpenAlex ID](https://openalex.org/A5026569995)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“validity–diversity”框架，系统分析大语言模型在推理时产生多样性崩溃的根源，并通过实验证明两种失调机制——顺序校准（order calibration）与形状校准（shape calibration）——导致有效输出被压缩，导致可生成多样性急剧下降。

**💡 创新点**

创新点在于：①将多样性崩溃归因为模型在推理步骤中的概率分布对有效与无效续写的排序与质量失衡；②定义并量化局部精度‑召回折衷以及其对序列级多样性的指数级放大效应；③构造可控实验台（随机数生成、随机州名等）和oracle cutoff基线，提供可复现的诊断工具；④给出理论证明，说明即使模型规模增大或温度调节，若存在顺序或形状失调，仍难以恢复高多样性。

**🔧 技术方法**

技术方法包括：LLM-as-judge进行Token有效性判别；计算局部精度、召回、序列级精度/召回；基于熵的有效支持度度量；对不同采样策略（top‑k、top‑p、min‑p、无过滤）及温度进行全面评估；oracle cutoff基线；并用统计与理论证明说明误差的乘法放大。

**📊 数据集**

使用了14款不同家族与规模的LLM（包括Qwen、ChatGPT等）以及自定义Prompt集合（NoveltyBench、随机数生成、美国州名等），没有公开大型数据集，而是利用自定义受控任务和公开的Prompt。

**📈 对比分析**

与传统采样方法（top‑k、top‑p、min‑p、无过滤）对比，实验显示：即使在高温度或大模型下，validity‑diversity折衷仍显著；oracle cutoff在前两步可显著提升多样性（Embedding Diversity ↑，Self‑BLEU ↓），但整体仍受限。

**⚠️ 局限性**

限制包括：受控实验仅在已知有效集的任务上验证；LLM-as-judge的可靠性需进一步确认；理论证明基于理想化假设（固定有效分支、无前缀差异）；未给出实际可行的解法，仅提供诊断与分析框架。

---

## 200. Under the Hood of SKILL.md: Semantic Supply-chain Attacks on AI Agent Skill Registry

**arXiv ID:** 2605.11418 | [PDF](https://arxiv.org/pdf/2605.11418v1)

**作者:** Shoumik Saha `[一作]` (University of Maryland), Soheil Feizi `[通讯]` (University of Maryland)

**通讯引用:** 10675 | [OpenAlex ID](https://openalex.org/A5025450606)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了基于Agent Skill的语义供应链攻击，探讨了在技能注册、检索、选择和治理四个生命周期阶段，通过仅修改自然语言描述就能提升恶意技能可见度、诱导技能选择并规避治理审核的攻击方式。

**💡 创新点**

创新点在于：①将技能生命周期拆解为发现、选择、治理三阶段，形成完整的攻击面框架；②提出仅凭文本的三种攻击策略（发现触发器、描述框架偏置、治理规避）；③在真实的ClawHub技能仓库上进行大规模、跨模型评估，验证攻击在实际生态中的可行性。

**🔧 技术方法**

使用的技术包括：基于Beam搜索与梯度优化的发现触发器生成；描述层的四种文本诱导方法（虚假广告、主动提示、时效信号、可信度声明）；基于静态规则、LLM评审和VirusTotal扫描的治理流水线；以及上下文窗口溢出、重述与定义完成等规避手段。

**📊 数据集**

数据集：从ClawHub下载100个公开技能，覆盖电子邮件、旅行、税务、健康、提示等五类；从中筛选47个完全干净的技能，再通过GPT-5生成94个恶意变体，形成攻击实验数据。

**📈 对比分析**

对比方法：在发现阶段分别对OpenAI、BAAI的三种嵌入模型进行同模型和跨模型攻击，显示最高86%对手胜率和80% Top‑10 命中；在选择阶段通过配对测试，攻击版本被选择率平均达到77.6%；在治理阶段，四种规避策略导致恶意命令被识别为“干净”的比例最高可达87%。实验表明，仅文本修改即可显著提升攻击成功率。

**⚠️ 局限性**

局限性：实验仅针对ClawHub及其特定检索/治理算法；对模型的黑盒/白盒假设不同可能影响攻击效果；仅针对文本不涉及代码级别攻击；部分规避策略（如上下文溢出）依赖于实际LLM审查窗口大小；未考虑社区审计、签名验证等更广泛的治理机制。

---

## 201. Unpacking the Eye of the Beholder: Social Location, Identity, and the Moving Target of Political Perspectives

**arXiv ID:** 2605.11166 | [PDF](https://arxiv.org/pdf/2605.11166v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 202. Large Language Models for Causal Relations Extraction in Social Media: A Validation Framework for Disaster Intelligence

**arXiv ID:** 2605.11348 | [PDF](https://arxiv.org/pdf/2605.11348v1)

**作者:** Ujun Jeong `[一作]` (Arizona State University), Huan Liu `[通讯]` (Arizona State University)

**通讯引用:** 143415 | [OpenAlex ID](https://openalex.org/A5100338921)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了专家校准的灾害因果图，并评估大型语言模型（LLM）从社交媒体文本中提取因果关系的能力。

**💡 创新点**

创新点在于①使用影响链框架预定义因果变量，消除节点歧义；②将LLM生成的因果图与专家报告得到的真值图进行对比；③通过非信息化帖子实验揭示LLM对内部先验的依赖。

**🔧 技术方法**

主要技术包括：LLM因果图生成（如xAI、OpenAI、Mistral），批量分块处理，统一提示模板，结构化评价指标（精度、召回、SHD、nSHD）。

**📊 数据集**

使用了CrisisMMD与HumAID两个公开灾害社交媒体数据集，重点针对海上飓风Irma与Harvey的帖子，并结合NOAA官方事后报告构建真值图。

**📈 对比分析**

与随机生成图基准比较，所有LLM性能显著更好；xAI模型在有原生社交媒体访问时获得最高F1（0.75/0.71），其余模型性能略低；在非信息化帖子实验中，模型表现下降，表明依赖先验知识。

**⚠️ 局限性**

局限性包括：仅评估了有标准NOAA报告的美洲灾害，难以推广到缺乏统一报告的灾害；模型性能受数据访问差异影响；缺乏对不同模型规模和推理机制的严格分离。

---

## 203. RT-Transformer: The Transformer Block as a Spherical State Estimator

**arXiv ID:** 2605.11007 | [PDF](https://arxiv.org/pdf/2605.11007v1)

**作者:** Peter Racioppo `[一作]` `[通讯]`, Peter Racioppo

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

将 Transformer 块视为在假设的径向-切向 SDE（RT‑SDE）下的方向性滤波更新，说明注意力、残差连接与归一化的几何共同起源。

**💡 创新点**

创新点在于提出 RT‑SDE，使噪声仅在状态的切向平面内共旋转，从而保留可闭式的协方差传播；同时给出将 Transformer 视作第一阶方向性滤波器的统一推导。

**🔧 技术方法**

使用线性 SDE、对角化动态、切向/径向投影、精度加权聚合、切向空间投影及归一化（retraction）等数学技术。

**📊 数据集**

未使用任何具体数据集，文中仅作理论推导和几何解释。

**📈 对比分析**

论文未包含实验或性能比较，仅提出理论框架；后续工作计划在未来进行实证评估。

**⚠️ 局限性**

局限性包括：缺乏实证验证、对输入维度的近似假设（高维近似正交）、未对前馈网络给出推导，且对实际训练稳定性与效率的影响未给出定量分析。

---

## 204. RankQ: Offline-to-Online Reinforcement Learning via Self-Supervised Action Ranking

**arXiv ID:** 2605.11151 | [PDF](https://arxiv.org/pdf/2605.11151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 205. Towards Model-Free Learning in Dynamic Population Games: An Application to Karma Economies

**arXiv ID:** 2605.11042 | [PDF](https://arxiv.org/pdf/2605.11042v1)

**作者:** Matteo Cederle `[一作]` (University of Padova), Gian Antonio Susto `[通讯]` (University of Padova)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在Karma经济的Dynamic Population Games（DPG）框架下，如何在完全无模型的情况下实现平衡学习。首先证明了当新代理加入已处于Stationary Nash Equilibrium（SNE）时，使用Deep Q‑Network（DQN）可以得到一个误差上界，该误差可分解为O(1/√N_s)（DQN近似误差）和O(1/N)（均值场扰动误差）。随后提出了从零开始学习SNE的FP‑DQN算法，将深度RL与Fictitious Play以及平滑策略迭代相结合，并在合成的Karma经济实例中进行实验验证，证明该算法能在模型无知识的情况下收敛至接近中心计算的SNE。

**💡 创新点**

①首次在完全耦合的DPG中给出单代理DQN学习的误差分解与上界；②提出从零开始学习SNE的FP‑DQN方法，并在GMFG环境中实现经验性收敛验证；③通过实验验证误差随经验回放大小和人口规模的下降趋势，展示模型无知识学习的可行性。

**🔧 技术方法**

使用Deep Q‑Network（DQN）结合ε‑greedy、经验回放和目标网络；采用Fictitious Play进行均值场平均；引入平滑策略迭代；利用Wasserstein‑1距离评估策略与分布的相似度；使用演化动力学算法计算中心化SNE做基准。

**📊 数据集**

论文使用合成的Karma经济仿真数据：平均Karma为10，K=40，urgency层级为{1,5}或{1,1,10}，状态空间为离散，转移概率ϕ给定；无公开数据集。

**📈 对比分析**

通过与中心计算的SNE进行Wasserstein距离和价值差距对比评估。单代理实验表明误差随经验回放大小N_s增大而减小，随人口规模N增大而减小；FP‑DQN算法在几十次迭代后Wasserstein距离降至0.03-0.05，显示能逼近中心SNE。实验表明模型无知识学习在Karma经济中能达到与中心化算法相近的性能。

**⚠️ 局限性**

①缺乏从零学习FP‑DQN的理论收敛保证；②SNE的唯一性未得到保证，可能存在多重平衡；③仅在有限状态动作的DPG上验证，连续空间或高维问题难以直接推广；④算法需要全局状态分布估计，实际部署中可能难以实现。

---

## 206. Experimental Examination of Secure Two-Party Controller Computation

**arXiv ID:** 2605.11443 | [PDF](https://arxiv.org/pdf/2605.11443v1)

**作者:** Kaoru Teranishi `[一作]` (University of Osaka), Takashi Tanaka `[通讯]` (Purdue University)

**通讯引用:** 1418 | [OpenAlex ID](https://openalex.org/A5082230661)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文实现并实验验证了一个安全的两方计算协议，用于在不泄露控制器状态和输入的前提下实时运行动态控制器。

**💡 创新点**

创新点在于提出了向量化的两方计算方案，并给出足够的模长条件以避免状态溢出，首次在实际云平台上实现了实时加密控制，并对子协议进行了细致的时延评估。

**🔧 技术方法**

使用了秘密共享、Beaver三元组乘法、位截断协议以及固定点编码技术，并在Python 3.12 里实现。

**📊 数据集**

实验数据来源于Quanser旋转倒立摆系统，采样周期为40 ms，控制器采用伪微分器和状态反馈构造。

**📈 对比分析**

通过测量乘法和截断子协议的往返时延，实验显示两者均保持在毫秒级；控制器在云服务器上成功稳定倒立摆，验证了协议在实际条件下的实时可行性。

**⚠️ 局限性**

局限性包括：通信延迟对实时性有显著影响，乘法子协议是主要瓶颈；实验仅覆盖单一倒立摆案例，缺乏对更复杂系统的验证；实现仍在Python层，需进一步用C/C++优化。

---

## 207. Sieve: Dynamic Expert-Aware PIM Acceleration for Evolving Mixture-of-Experts Models

**arXiv ID:** 2605.11277 | [PDF](https://arxiv.org/pdf/2605.11277v1)

**作者:** Jungwoo Kim `[一作]` (Stanford University), Kunle Olukotun `[通讯]` (Stanford University)

**通讯引用:** 16092 | [OpenAlex ID](https://openalex.org/A5023857198)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Sieve的调度器和运行时框架，用于在多GPU+HBM‑PIM系统上高效执行现代Mixture‑of‑Experts（MoE）大语言模型（LLM），通过动态将专家分配到GPU或PIM并重叠计算与通信来提升吞吐量和交互性。

**💡 创新点**

创新点包括：
1) 通过观察到的二模态专家分布（少数专家处理大量token，多数专家仅处理1–4个token）动态评估算术强度；
2) 以算术强度为依据，将低算术强度的“稀有”专家下放至PIM，保持高算术强度的“热门”专家在GPU上；
3) 同时考虑GPU计算、PIM计算、HBM带宽以及多GPU间的专家/张量并行通信，形成统一的调度目标；
4) 采用轻量级贪心搜索实现运行时调度，开销≈20 µs；
5) 构建完整的运行时系统实现GPU–PIM协同执行、内存/通信重叠，并支持张量并行与专家并行的组合。

**🔧 技术方法**

技术手段包括：
- 基于算术强度的专家划分算法；
- 结合GPU峰值吞吐、PIM峰值吞吐、HBM‑PIM带宽与NVLink通信带宽的多维度成本模型；
- 采用Ramulator 2.0与Duplex模拟器进行周期级仿真；
- GPU侧使用grouped GEMM、张量并行；
- PIM侧采用GEMV/skinny GEMM转化为多条GEMV指令，并利用PIM通道的张量并行；
- 通过AllGather、路由、dispatch、聚合等步骤构建依赖图，确保计算与通信重叠。

**📊 数据集**

使用的真实数据集为：HH‑RLHF（人类偏好对话序列）和MATH‑500（数学推理题目），并基于这些请求对GPT‑OSS、Qwen3.5、Qwen3三大MoE模型进行token‑expert分布采样。

**📈 对比分析**

对比方法包括：
- NoExp（仅将attention移至PIM，专家全在GPU）；
- AllExp（所有专家全部在PIM）；
- PIMoE（基于阈值的静态专家分配）。
评估指标为吞吐量（token/s）和交互性（user‑token/s），通过Pareto曲线展示。Sieve在B=64–256范围内，分别在Qwen3.5、GPT‑OSS、Qwen3上实现了1.3×、1.3×、1.6×的吞吐量提升，交互性同样提升。尤其在高负载、多GPU场景下，相比PIMoE可获得≈1.3×的吞吐/交互性加速，并在 colocated prefill‑decode 场景下取得最高2.3×-2.4×的加速。

**⚠️ 局限性**

局限性与待改进点：
- 依赖于HBM‑PIM硬件模型，实际部署时需验证硬件实现细节；
- 调度算法仅考虑token‑expert分布，对极端请求混合（大量prefill）仍可能出现子最优；
- 评估主要在解码阶段，预填充阶段的影响尚待进一步研究；
- 目前采用仿真验证，缺乏实测的硬件验证；
- 对于更大规模的模型或更细粒度的张量并行策略，需要进一步扩展和优化。

---

## 208. On the Impact of Crossover in Many-Objective Optimization: A Runtime Analysis of NSGA-III

**arXiv ID:** 2605.11201 | [PDF](https://arxiv.org/pdf/2605.11201v1)

**作者:** Andre Opris `[一作]` (University of Passau), Andre Opris `[通讯]` (University of Passau)

**通讯引用:** 132 | [OpenAlex ID](https://openalex.org/A5081165511)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对NSGA-III在多目标m-功能上的运行时间进行了严格的理论分析，并比较了使用与不使用交叉算子时的性能差异。

**💡 创新点**

创新之处在于首次证明交叉算子能在多目标优化中实现渐进甚至指数级的运行时间加速，并给出了相应的上界与下界。

**🔧 技术方法**

采用了概率分析、Chernoff界、分层覆盖、参考点映射等技术，对交叉与变异操作的影响进行细致建模。

**📊 数据集**

实验基准为自定义的m-功能（m-函数），该函数在多目标空间中具有可控的Pareto前沿结构。

**📈 对比分析**

通过理论高概率与期望运行时间的比较，结果显示在大多数参数范围内，带交叉的NSGA-III比无交叉版本快至少O(k!)倍，某些参数甚至实现指数级加速。

**⚠️ 局限性**

局限在于下界仅针对m=4且不含交叉，且理论假设较为理想化，未覆盖更复杂的实际多目标问题与交叉算子类型。

---

## 209. On Problems of Implicit Context Compression for Software Engineering Agents

**arXiv ID:** 2605.11051 | [PDF](https://arxiv.org/pdf/2605.11051v1)

**作者:** Kirill Gelvan `[一作]` (JetBrains Research), Yaroslav Zharov `[通讯]` (JetBrains Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fede83ac-7505-405f-ab37-e7284695c47f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了隐式上下文压缩（ICAE）在LLM驱动的软件工程代理中的效果，并在单步与多步任务上进行了实验。

**💡 创新点**

提出将ICAE适配为代理工作流程，针对长轨迹的观察压缩并探讨压缩失败的原因。

**🔧 技术方法**

使用ICAE编码器‑解码器结构，基于Qwen3‑8B模型，结合LoRA、RoPE位置ID操作以及观察压缩技术。

**📊 数据集**

使用SlimPajama‑6B预训练数据，单步任务采用SQuAD和RepoQA，多步任务采用SWE‑Smith轨迹与SWE‑bench Verified基准。

**📈 对比分析**

对比Base、SFT与ICAE三种配置，单步任务中ICAE提升BLEU/EM等指标，但在SWE‑bench Verified上表现下降，导致解决问题数减少，虽然可延长约40%的上下文长度并加快约10%的生成速度。

**⚠️ 局限性**

主要限制在于压缩后的重构精度不足，导致误差累积；并且训练仅针对最近压缩观测，缺乏对未来步骤信息的保留，导致多步代理任务性能衰退。

---

## 210. Simpson's Paradox in Behavioral Curves: How Aggregation Distorts Parametric Models of User Dynamics

**arXiv ID:** 2605.11017 | [PDF](https://arxiv.org/pdf/2605.11017v1)

**作者:** Chao Zhou `[一作]` `[通讯]` (Meta Platforms), Chao Zhou (Meta Platforms)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了聚合行为曲线对个体峰值估计的系统性偏差，发现了“行为曲线中的辛普森悖论”，并提供了层次贝叶斯峰值估计和合成空缺校准（SNC）两种解决方案。

**💡 创新点**

创新点在于将辛普森悖论应用于行为曲线拟合，揭示了生存偏差（差异化流失）是导致聚合曲线与个体峰值显著失配的根本机制，并提出了针对高维模型过拟合的合成空缺校准方法。

**🔧 技术方法**

主要技术包括 Hill‑Exponential 参数化模型、似然比检验、AIC/BIC 选择、Empirical Bayes 经验贝叶斯收缩、Bootstrap 稳健性检验以及针对合成 null 数据的误报率校准。

**📊 数据集**

使用了三大公开数据集：Goodreads（330 万用户 9 个类别）、Amazon Electronics（1800 万条评论 1.88 万用户）和 MovieLens‑25M（2500 万条评分 20 个类别），用于验证和对照。

**📈 对比分析**

通过比较聚合曲线峰值与个体峰值的比率（D）评估偏差；Goodreads 的 D≈3、Amazon Electronics 的 D≈5.3，而 MovieLens 的 D≈1，表明偏差随生存偏差强度变化；层次贝叶斯估计在三组数据中显著降低了峰值偏差，合成空缺校准揭示单用户分类的误报率高达 32%。

**⚠️ 局限性**

局限包括：聚合曲线与个体曲线对比依赖于足够的数据点；单用户模型的高维度导致误报率高，无法精确估计逆 U 型曲线比例；结果主要基于离散消费数据，尚未验证于流媒体或临床剂量响应等连续行为场景；并且对时间演化和协变量的控制不充分。

---

## 211. Behavioral Mode Discovery for Fine-tuning Multimodal Generative Policies

**arXiv ID:** 2605.11387 | [PDF](https://arxiv.org/pdf/2605.11387v1)

**作者:** Alberta Longhini `[一作]` (Stanford University), Seungsu Kim `[通讯]` (Naver Labs Europe)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无监督的行为模式发现框架，在强化学习微调预训练的多模态生成策略时通过保持多模态性来提升任务成功率。

**💡 创新点**

创新点：① 使用互信息作为多模态度量；② 通过对驱动噪声的潜在变量重参数化，自动发现并控制行为模式；③ 在RL微调过程中加入互信息奖励作为正则化项，显著抑制模式崩塌。

**🔧 技术方法**

采用预训练的扩散/流式生成策略，训练“steering policy” π_w(w|s,z) 与推理网络 q_ϕ(z|s)；利用变分下界估计 I(Z;S) 作为 intrinsic reward；在 PPO 等 RL 算法中加入此奖励进行微调。

**📊 数据集**

使用演示数据集（ManiSkill、D3IL）、机器人模拟任务（ANYmal locomotion、Franka Kitchen）以及 2D Gaussian 混合奖励实验，并通过这些数据集进行训练和评估。

**📈 对比分析**

与直接微调、残差策略、Steering policy 等基线对比，评估指标包括成功率 SR、模式加权成功率 SR_M、模式覆盖 mc@0.8、熵等。在多模态任务中，BMD 在保持多模态的同时，成功率与基线相当或更高，并显著降低模式崩塌。

**⚠️ 局限性**

局限性：仅考虑离散潜在空间；互信息估计可能保守，无法捕捉所有任务相关多模态；需要额外的模式发现步骤，训练成本提升；对不同任务的可推广性和扩展性仍待验证。

---

## 212. Robust Biomedical Publication Type and Study Design Classification with Knowledge-Guided Perturbations

**arXiv ID:** 2605.11502 | [PDF](https://arxiv.org/pdf/2605.11502v1)

**作者:** Shufan Ming `[一作]` (University of Illinois Urbana-Champaign), Halil Kilicoglu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3220 | [OpenAlex ID](https://openalex.org/A5016571803)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过构建受控语义扰动评估框架，系统评估并提升了医学文献出版类型与研究设计分类模型的鲁棒性，并提出了结合实体掩码与领域对抗训练的鲁棒训练策略。

**💡 创新点**

创新点在于：①首次设计了知识引导的语义扰动（同义词替换、概念替换+EDA）来评估模型对方法信息保持的鲁棒性；②提出将实体掩码与领域对抗训练联合使用，能够在抑制非任务特征的同时不牺牲原始准确率，缓解鲁棒性与准确率的传统折中。

**🔧 技术方法**

技术手段包括：SPECTER2-base 编码器；非对称标签损失（ASL）+标签平滑；HeroCon 与 ADNCE 对比学习；领域对抗网络（DANN）配合梯度反转层；MetaMap 识别并掩码非方法性实体；以及基于 UMLS 的语义扰动操作。

**📊 数据集**

使用 Menke 等人 2023 年公开的 PubMed 论文集（166,192 篇），构成 61 个出版类型标签，按 70/10/20 划分训练/验证/测试。

**📈 对比分析**

与 baseline 进行比较，实验设置为四种模型（baseline、mask、adversarial、mask+adversarial）。在极端扰动（100% 概念替换+EDA）下，mask+adversarial 微 F1 达到 0.604、宏 F1 0.606，鲁棒准确率显著高于 baseline（0.573）。在原始测试集上，mask+adversarial 与 baseline 近乎相同，且在宏 F1、AUPRC 以及校准误差上略优或相当，显示出最优的鲁棒-准确平衡。

**⚠️ 局限性**

局限性包括：仅在单一 SPECTER2-base 模型上验证，缺乏跨模型泛化评估；掩码比例与领域标签阈值选择可能过于粗略，可能不小心抑制真正的任务相关信息；评估主要聚焦于摘要级输入，未考虑全文或方法章节的额外结构信息。

---

## 213. FibQuant: Universal Vector Quantization for Random-Access KV-Cache Compression

**arXiv ID:** 2605.11478 | [PDF](https://arxiv.org/pdf/2605.11478v1)

**作者:** Namyoon Lee `[一作]` (Pohang University of Science and Technology), Yongjune Kim `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 648 | [OpenAlex ID](https://openalex.org/A5025186204)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种面向KV缓存的随机访问固定率向量量化编码器FibQuant，证明其在旋转后服从的球面-贝塔源上优于传统标量量化，并实现亚比特及分数比特压缩；

**💡 创新点**

① 引入球面-贝塔源并证明KV缓存块服从该分布；② 构造径向-角向量码本（Beta分位半径 + Fibonacci/Roberts–Kronecker均匀球面点 + Lloyd–Max细化）；③ 通过向量量化获得密度匹配与单元形状优势，形成连续可调的比特率轴；④ 在无校准、全局随机访问条件下实现亚比特压缩；

**🔧 技术方法**

Haar随机正交旋转、归一化、球面-贝塔理论、Beta分位数半径压缩、Fibonacci/Roberts–Kronecker均匀球面点集、Lloyd–Max细化、固定率表查找、注意力余弦相似度评估；

**📊 数据集**

GPT‑2 small、TinyLlama‑1.1B‑Chat‑v1.0、WikiText‑103、HellaSwag、WikiText‑103滑动窗口等；

**📈 对比分析**

与Int、KIVI、TurboQuant、StreamingLLM、H2O、低秩SVD等方法在KV缓存压缩比与注意力余弦相似度、以及TinyLlama的perplexity与HellaSwag准确率上进行对比。FibQuant在5×以上压缩时保持≈0.95+相似度，34×压缩时达到0.946；TinyLlama 2位时perplexity 15.9，对比TurboQuant 56.7；分数比特区间提供细粒度调优；

**⚠️ 局限性**

未给出有限率下Lloyd–Max最优性闭式分析；仅在中等规模模型验证，未覆盖更大模型或更长上下文；未实现融合内核；随机旋转的方差影响未完全评估；缺少对生成质量全程影响的完整评估。

---

## 214. ASIP-Planner: Adaptive Planning for UAV Surface Inspection in Partially Known Indoor Environments

**arXiv ID:** 2605.11119 | [PDF](https://arxiv.org/pdf/2605.11119v1)

**作者:** Hanyu Jin `[一作]` (Carnegie Mellon University), Kenji Shimada `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4985 | [OpenAlex ID](https://openalex.org/A5101724128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种自适应无人机地面检查框架，将基于分段的全局覆盖规划与基于视角的局部适配相结合，实现在部分已知室内环境中的高效、连贯检查；

**💡 创新点**

创新点在于：①通过将参考地图按表面法向分段形成聚集的视点簇，实现紧凑且方向一致的全局序列；②在执行时实时调整摄像机视角，弥补遮挡导致的覆盖损失；

**🔧 技术方法**

采用了分段区域生长、Lin‑Kernighan‑Helsgaun（LKH）TSP求解、B‑spline轨迹优化、模型预测控制以及基于占据网格的视角加权评分算法；

**📊 数据集**

主要使用了随机生成的室内几何场景（10–15 个结构件的多种组合）进行仿真，及真实场景中的办公楼与隧道无人机飞行数据；

**📈 对比分析**

与基线 FC‑Planner 与 HCPP 进行对比；在五种场景配置中，覆盖率均≥99.8%，轨迹长度显著下降（约35%–45%），平均偏航变化降低约3倍；

**⚠️ 局限性**

局限性在于：①仍假设目标为垂直平面墙面；②对极端动态障碍和不规则几何结构适应性不足；③依赖参考地图的初始准确性，无法完全解决完全未知环境。

---

## 215. Leveraging Multimodal Large Language Models for All-in-One Image Restoration via a Mixture of Frequency Experts

**arXiv ID:** 2605.11444 | [PDF](https://arxiv.org/pdf/2605.11444v1)

**作者:** Eunho Lee `[一作]` (Chungbuk National University), Rei Kawakami `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 1813 | [OpenAlex ID](https://openalex.org/A5113695192)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于多模态大语言模型（MLLM）的全景图像修复框架，利用MLLM的视觉与文本嵌入在编码器-解码器网络中做指导，并引入混频专家（MoFE）模块自适应处理不同频率的降噪、去雾、去雨、低光等复合降解。

**💡 创新点**

创新点包括：① 用MLLM的连续降解嵌入替代离散标签，捕获混合降解的连续关系；② 通过MGFB将视觉与语言特征注入网络，弥补低层修复的表征缺口；③ 采用频域MoFE并使用MLLM引导的路由器与关系对齐损失，实现专家权重的语义一致性；④ 采用多尺度渐进训练提升对不同分辨率的鲁棒性。

**🔧 技术方法**

技术方法包括：Restormer结构；Qwen2.5-VL MLLM VQA提示获取嵌入；跨模态注意力（MGFB）；DWT频域分解；频域混合专家（MoFE）与MDTA融合；MLLM引导的路由器与关系对齐损失；L1+频域重建损失；多尺度渐进训练。

**📊 数据集**

使用的数据集主要包括CDD11（混合降解基准）、BSD400、WED、BSD68、Rain100L、SOTS、GoPro、LOL-v1 等全景修复和三、五种降解的标准数据集。

**📈 对比分析**

在CDD11、三降解和五降解基准上与最新SOTA（如PromptIR、VLU‑Net、TEAFormer等）进行对比，平均PSNR提升0.5–1.3 dB，达到最高PSNR 32.16 dB、SSIM 0.981，尤其在混合降解场景中显著优于对手。

**⚠️ 局限性**

局限性包括：对单一降解的性能略低于专门模型；在去雨任务（数据量有限）上表现不如部分SOTA；依赖MLLM推理增加算力和延迟，虽然小模型可用但仍高于纯提示方法；对极端高分辨率图像的鲁棒性仍待进一步验证。

---

## 216. When and How to Canonize: A Generalization Perspective

**arXiv ID:** 2605.11008 | [PDF](https://arxiv.org/pdf/2605.11008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 217. Leveraging RAG for Training-Free Alignment of LLMs

**arXiv ID:** 2605.11217 | [PDF](https://arxiv.org/pdf/2605.11217v1)

**作者:** John T. Halloran `[一作]` `[通讯]` (Leidos), John T. Halloran (Leidos)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种无训练、在线对齐算法RAG-Pref，用检索增强生成（RAG）结合对比信息，以提升大型语言模型在面对缺少传统拒绝触发词的代理攻击（FBA）时的拒绝率，并在一般人类偏好对齐任务中保持性能提升。

**💡 创新点**

创新点在于：①将偏好与反偏好示例同时检索并作为对齐指令，实现对比条件化；②无需额外训练，兼容现有RAG组件；③理论上证明对比信息能降低推理不确定性，并通过实验验证其对安全和性能的双重提升。

**🔧 技术方法**

技术包括：检索增强生成（RAG）、文本嵌入向量检索、对比信息编码、在线推理时的指令生成；与之对比的技术有DPO、SafeDPO、OPAD等。

**📊 数据集**

使用了自构建的FBA与TB对齐数据集（约1,150个FBA/1,035个TB训练样本，115个测试样本），以及公开的AlpacaEval 2和MT‑Bench评测基准。

**📈 对比分析**

与离线对齐（DPO、SafeDPO）和在线对齐（OPAD、标准RAG）比较，RAG‑Pref在FBA拒绝率上平均提升3.7倍（相较于离线方法1.5倍、OPAD 1.8倍），在AlpacaEval 2和MT‑Bench上平均提升24.4%/7.3%或228.4%（相较于基线、RAG、OPAD）。预处理速度比DPO快3个数量级，推理时延仅比OPAD慢20%（比372%显著）。

**⚠️ 局限性**

局限性包括：①仅在公开基准和自建FBA数据上验证，未评估在更大规模或不同语言的泛化；②对多轮对话或更复杂工具链的适用性仍待实验；③虽然无需训练，但对检索系统的依赖可能在资源受限场景下带来新的瓶颈。

---

## 218. Enabling Performant and Flexible Model-Internal Observability for LLM Inference

**arXiv ID:** 2605.11093 | [PDF](https://arxiv.org/pdf/2605.11093v1)

**作者:** Nengneng Yu `[一作]` (University of Maryland), Zaoxing Liu `[通讯]` (University of Maryland)

**通讯引用:** 1434 | [OpenAlex ID](https://openalex.org/A5015818714)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Deep Model Inspector（DMI），一种高性能、低开销的深度模型内部状态观测系统，能够在不破坏推理主路径的前提下，实时捕获并导出 LLM 的残差流、注意力、MLP 激活等多种内部张量，支持多引擎（Hugging Face、vLLM 等）和分布式推理；

**💡 创新点**

创新点在于：①将观测视作独立系统原语，使用轻量级 HookPoint 原子在模型前向图任意位置插桩；②设计 Ring^2 GPU–CPU 双环缓冲，实现在 CUDA 图内无同步的数据复制与异步导出；③通过 Hook 过滤与运行时策略两级控制，灵活应对不同硬件带宽与需求，避免主路阻塞；

**🔧 技术方法**

核心技术包括自定义 CUDA 设备对设备拷贝内核、CUDA 图兼容的 HookPoint、Ring^2 设备/主机双环缓冲、异步 host 采集与 ClickHouse 持久化、基于策略的请求/Hook 滤除；

**📊 数据集**

实验数据集涵盖：ShareGPT、WildChat‑1M（用于在线推理评估），HellaSwag（用于知识蒸馏验证），以及多规模 Qwen3、Llama 系列模型；

**📈 对比分析**

与 PyTorch 前向 Hook、Hugging Face 内置/逐步提取、vLLM Hook、TensorRT‑LLM Debug API、NNsight 等基线相比，DMI 在离线批处理下 0.4%–6.8% 负载，在线推理约 6% 负载；与基线相比，推理延迟提升仅 2×–15×，内存占用提升仅 1.3%–5.0%，并在分布式多 GPU 场景保持高吞吐；

**⚠️ 局限性**

局限性包括：①对 HookPoint 的插入位置仍需手工指定，过多 Hook 可能导致 GPU 侧显存占用升高；②在极端 PCIe 带宽饱和时需启用降采样或丢弃策略，影响完整观测；③对某些自研引擎或特殊算子支持有限；④部分 Hook 位置可能导致数值精度微小偏差。

---

## 219. ReAD: Reinforcement-Guided Capability Distillation for Large Language Models

**arXiv ID:** 2605.11290 | [PDF](https://arxiv.org/pdf/2605.11290v1)

**作者:** Xueqi Cheng `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 1001 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在固定token预算下对LLM进行能力蒸馏，并提出ReAD框架以自适应分配蒸馏预算

**💡 创新点**

通过识别任务关键能力并使用强化学习调度预算，显著降低跨能力负面迁移和预算浪费

**🔧 技术方法**

能力转移矩阵分析、任务需求识别器、基于上下文的UCB bandit、在线能力目标数据生成

**📊 数据集**

Llama-3.3-70B/8B、Qwen2.5-72B/14B 以及八大核心能力基准（General、Reasoning、Math、Code、Tool、LCU、Steerability、多语言）

**📈 对比分析**

与六种单目标蒸馏基线对比，ReAD在相同token预算下平均提升5–10%能力得分，在预算-收益曲线和负面迁移曲线中表现最佳

**⚠️ 局限性**

在单目标或极低预算场景下效果有限，且对非核心能力的误判可能导致预算分配不理想

---

## 220. gym-invmgmt: An Open Benchmarking Framework for Inventory Management Methods

**arXiv ID:** 2605.11355 | [PDF](https://arxiv.org/pdf/2605.11355v1)

**作者:** Reza Barati `[一作]` (Toronto Metropolitan University), Qinmin Vivian Hu `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 1133 | [OpenAlex ID](https://openalex.org/A5018749138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于 Gymnasium 的跨范式库存管理基准环境，提供了 22 个核心场景加 4 个 MARL 模式，统一的状态/动作/奖励/KPI 合同，使得优化、启发式、强化学习和 LLM 控制器能够在相同的物理机制下直接比较。

**💡 创新点**

创新点在于：① 通过可配置的 DAG 拓扑和可变需求/信息协议，设计了一个真正可复现且可扩展的评估合同；② 发布了 26 场景的全矩阵实验数据，涵盖拓扑、需求非平稳、顾客好感度、补货模式等维度；③ 引入了“通用式”训练策略，使得每个网络拓扑只训练一次通用检查点，测试跨场景泛化能力；④ 结合 LLM 策略参数生成的“策略+执行器”框架，为未来的 LLM‑库存控制提供可行路线。

**🔧 技术方法**

技术手段包括：Gymnasium 环境实现（继承 OR‑Gym），多种强化学习算法（PPO‑Transformer、PPO‑GNN、SAC、Residual‑RL 等），多阶段随机规划（MSSP‑I）和确定性 LP（DLP），经典启发式（Newsvendor、(s,S)、Echelon、ExpSmooth），以及基于 Qwen2.5 的 LLM‑Policy‑C 诊断控制器。

**📊 数据集**

使用的数据集为：① 合成需求（4 种分布，含趋势、季节性、冲击）；② M5 真实零售销售轨迹（30 天，按需求均值 20 归一化）；③ 10 个种子用于评估，每个场景多次迭代；④ 通过随机种子和域随机化在训练阶段覆盖多种需求/好感度组合。

**📈 对比分析**

比较方法：在相同的 22 场景核心网格上，统计每种控制器相对 Oracle 的收益百分比以及平均推理时间。结果显示：MSSP‑I（信息化随机规划）在所有场景下平均 95% 的 Oracle 利润，PPO‑Transformer 约 75%，Residual‑RL 约 73%，经典启发式 56‑68%，LLM‑Policy‑C 约 60%；计算上，MSSP‑I 需 ~10 s/episode，PPO‑Transformer ~0.23 s/episode，LLM‑Policy‑C ~8.2 s/episode，表明通用式学习策略在多次部署时可以显著降低在线计算成本。

**⚠️ 局限性**

局限性包括：① 场景仅覆盖两种拓扑（默认网络和串联）和单 SKU；② 未覆盖多产品、随机交货期、产量波动等更复杂物流特性；③ LLM 结果仅为诊断性实验，无法证明其在高频控制上的实用性；④ 训练检查点仅在单一种子上生成，缺乏多种子训练和超参搜索；⑤ Oracle、MSSP‑I 等方法在信息访问和模型假设上存在不对称性，评估结果需结合具体部署场景解释。

---

## 221. Computational Design of a Low-Visibility UAV Using a Human-Aligned Perceptual Metric

**arXiv ID:** 2605.11296 | [PDF](https://arxiv.org/pdf/2605.11296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 222. Annotation-Free Indoor Radio Mapping via Physics-Informed Trajectory Inference

**arXiv ID:** 2605.11037 | [PDF](https://arxiv.org/pdf/2605.11037v1)

**作者:** Zheng Xing `[一作]`, Shuguang Cui `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种不需要定位标签的室内射频地图构建框架，利用MIMO‑OFDM CSI中的物理信息推断用户轨迹并据此生成RSS、PADP、信道与波束地图。

**💡 创新点**

创新点在于设计PADP（功率角延迟剖面）特征距离作为物理连续性约束，并将该约束与RSS、方向信息和移动模型融合进空间正则化的贝叶斯轨迹推断，完全消除对IMU或人工标记的依赖。

**🔧 技术方法**

采用MIMO‑OFDM CSI预处理、PADP特征提取、贝叶斯轨迹推断、Viterbi式解码、空间图离散化以及参数交替优化等技术。

**📊 数据集**

使用真实工业环境的dichasus‑cf0x CSI数据集，覆盖约14m×14m的L形测量区域，包含四个UPA天线阵列和1024个子载波的OFDM信号。

**📈 对比分析**

与六个基线（Siamese、Bilateration、RITA、GraphIPS、ENAP、VRLoc）进行对比，平均定位误差0.88 m、Beam估计误差6.68 %，相较于最优基线（Bilateration 0.98 m）表现更佳。

**⚠️ 局限性**

局限性包括：PADP约束在强NLoS、多径交叉或同步误差显著时的可靠性下降；方法需已知AP几何与行走区域，未考虑动态障碍物、多用户干扰或更大规模环境。

---

## 223. What Do EEG Foundation Models Capture from Human Brain Signals?

**arXiv ID:** 2605.11410 | [PDF](https://arxiv.org/pdf/2605.11410v1)

**作者:** Ling Tang `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongrui Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5020653216)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对 EEG 基础模型进行因果解释，评估其是否编码并使用了已有的手工特征。

**💡 创新点**

创新点在于将层级岭回归探测、LEACE 样式子空间抹除以及透明逻辑回归分类器相结合，系统性地在三模型五任务下检验编码、使用与性能恢复。

**🔧 技术方法**

采用层级岭回归探测、跨协方差子空间抹除（LEACE 风格）、逻辑回归透明分类器、Bootstrap 置信区间及多重比较校正等技术。

**📊 数据集**

使用五个临床 EEG 任务数据集：MDD、Stress、ISRUC‑Sleep、TUSL、Siena，并在三种预训练模型 CSBrain、CBraMod、LaBraM 上进行评估。

**📈 对比分析**

与同维度随机特征基线和全特征基线对比，确认特征平均恢复约 79.3% 的模型优势，并表现出任务梯度（MDD ≈ 99%，Stress ≈ 55%）。

**⚠️ 局限性**

主要限制是探测与抹除均为线性，无法捕获非线性特征组合；特征词典不完整，结果仅适用于所评估的模型与任务。

---

## 224. DeconDTN-Toolkit: A Library for Evaluation and Enhancement of Robustness to Provenance Shift

**arXiv ID:** 2605.11237 | [PDF](https://arxiv.org/pdf/2605.11237v1)

**作者:** Yongsen Tan `[一作]` (University of Washington), Trevor Cohen `[通讯]` (University of Washington)

**通讯引用:** 4640 | [OpenAlex ID](https://openalex.org/A5071178113)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了训练与部署期间因数据来源与标签关系改变而产生的“来源偏移”（provenance shift），并提出了基于因果图的理论框架与专门的实验工具包；

**💡 创新点**

创新点在于将来源偏移与反事实不变性、可学习目标相联系，设计出可重复实验的 DeconDTN 工具，并系统评估多种不变学习算法在来源偏移下的鲁棒性；

**🔧 技术方法**

主要技术包括因果图建模、反事实不变性目标、Invariant Risk Minimization (IRM)、DomainBed 兼容的实验框架以及利用 α 参数合成不同程度的来源偏移；

**📊 数据集**

实验使用了五个公开数据集：SHAC、MIMIC‑III、HateSpeech、MultiNLI 与 Civilcomments；

**📈 对比分析**

在 DeconDTN 生成的不同 α 级别的来源偏移下对比方法，结果表明传统 ERM 在偏移下性能急剧下降，而去除来源-标签相关性的基线和部分不变学习方法能保持更优性能；

**⚠️ 局限性**

局限性主要包括假设训练阶段可观测来源标签但测试阶段不可观测、当前仅支持二分类两来源场景，且未探讨更复杂的多来源或多标签设置。

---

## 225. QuIDE: Mastering the Quantized Intelligence Trade-off via Active Optimization

**arXiv ID:** 2605.10959 | [PDF](https://arxiv.org/pdf/2605.10959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 226. PD-4DGS:Progressive Decomposition of 4D Gaussian Splatting for Bandwidth-Adaptive Dynamic Scene Streaming

**arXiv ID:** 2605.11427 | [PDF](https://arxiv.org/pdf/2605.11427v1)

**作者:** Jiachen Li `[一作]` (Qilu University Of Technology), Gang Li `[通讯]` (Qilu University Of Technology)

**通讯引用:** 47698 | [OpenAlex ID](https://openalex.org/A5100438769)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出PD-4DGS框架，实现4D Gaussian Splatting的渐进压缩与按需传输；

**💡 创新点**

将渲染网络拆分为静态支架、全局变形、局部细化三层，实现可逐层渲染；配合属性级R‑DO、时间掩码一致性和自适应容量加权训练，解决基准层压缩与动态网络训练失衡问题；

**🔧 技术方法**

Hierarchical Deformation Decomposition、Attribute‑Level Rate–Distortion Optimization、Temporal Mask Consistency、Capacity‑Weighted Progressive Rollout Training、Gaussian entropy modeling、D‑SSIM损失、LZMA压缩；

**📊 数据集**

Dycheck iPhone monocular基准数据集；

**📈 对比分析**

与SC‑GS、Deformable 3DGS、4DGS、MoDec‑GS等基线比较，平均mPSNR 14.54 dB、mSSIM 0.480，流媒体体积仅6.88 MB（相对MoDec‑GS 62.6 %缩减），首帧延迟1.7 s（2 Mbps），大幅优于传统单一模型；

**⚠️ 局限性**

仅适用于包含全球与局部运动的场景；层数固定，无法在运行时插入新层；评测仅在Dycheck iPhone上完成，需在其他数据集与大场景验证；

---

## 227. Language Modeling with Hyperspherical Flows

**arXiv ID:** 2605.11125 | [PDF](https://arxiv.org/pdf/2605.11125v1)

**作者:** Justin Deschenaux `[一作]` (EPFL), Caglar Gulcehre `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在单位超球面（𝕊^{d-1}) 上训练的连续流语言模型——Hyperspherical Flow Language Model（SFLM），通过对 token 嵌入进行端到端学习，避免了传统 Flow Language Model（FLM）需要 materialize 的 one‑hot 维度；并在采样时使用交叉熵训练的去噪器推导出闭式速度场，实现了并行生成。

**💡 创新点**

主要创新点包括：
1) 在 Riemannian 流框架下定义在超球面上的流；
2) 对噪声时间刻度进行理论分析并提出截断与自适应噪声调度，显著提升训练稳定性和采样性能；
3) 设计了 hyperspherical Transformer（H-Transformer），将中间激活保持在单位球面上，并通过旋转操作代替加法残差；
4) 提出了精确速度、随机速度与 top‑k 速度三种采样策略，提升了推断精度。

**🔧 技术方法**

使用技术：
- Riemannian Flow Matching（RFM）与 SLERP 插值
- 交叉熵（Cross‑Entropy）去噪器训练
- 速度场闭式推导与数值积分
- top‑k 速度与随机速度采样
- 自适应噪声调度（基于梯度变化的逆 CDF）
- hyperspherical Transformer 架构（norm‑interpolated residual, 旋转 attention/MLP）。

**📊 数据集**

实验数据集：
- Sudoku（48k 训练 + 2k 验证）
- TinyGSM（≈11.8M 生成数学问题）
- GSM8K（测试集，用于执行 Python 代码后评估答案）
- OpenWebText（无条件语言建模，使用 GPT‑2 tokeniser）。

**📈 对比分析**

比较方法与性能：
- 与 AR、离散扩散（MDLM、Duo）、连续扩散（FLM、CANDI）以及标准 DiT 进行对比；
- Sudoku：SFLM 与 FLM 相近，达到 94% 以上的准确率，超过大部分连续模型；
- GSM8K：使用 top‑1 速度时，SFLM 达到 18% 的正确率，远高于 FLM/CANDI（<1%）和 MDLM（18%），但在低温（T=0.1）下仍落后 Duo（~35%）；
- OpenWebText：在高 NFE（1024）时 Gen. PPL 与 entropy frontier 与先前 FLMs 一致，低 NFE（32）下更稳定并略优；
- 总体上，SFLM 在大词表下的并行生成与生成困境任务上显著提升，且训练速度快于传统 FLM，但在极低温或复杂推断任务仍有提升空间。

**⚠️ 局限性**

局限性：
- 与离散扩散/AR 模型相比，仍存在显著性能差距，尤其在低温采样时；
- 截断阈值 α^⋆(δ) 依赖简化假设（随机分布嵌入），可能与实际学习到的结构不符；
- H-Transformer 训练速度低于标准 DiT；
- 训练规模有限（单一模型，TinyGSM 训练成本 > $300），缺乏大规模复现；
- 需进一步探索更高效的训练与采样策略，以及更深层次的嵌入表示学习。

---

## 228. Can Graphs Help Vision SSMs See Better?

**arXiv ID:** 2605.11300 | [PDF](https://arxiv.org/pdf/2605.11300v1)

**作者:** Dhruv Parikh `[一作]` (University Of Southern California), Viktor Prasanna `[通讯]` (University Of Southern California)

**通讯引用:** 17540 | [OpenAlex ID](https://openalex.org/A5033166029)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出GraphScan，将局部图注意力作为预处理步骤，替代传统几何扫描，使视觉SSM在序列化前实现语义路由；

**💡 创新点**

创新点在于将扫描视作输入侧的局部语义路由，用图结构学习邻域注意力，而非仅依赖几何路径或坐标偏移；

**🔧 技术方法**

使用局部图注意力（图结构消息传递）、Mamba式选择性状态空间模型、以及Transformer/卷积等常规视觉网络组件；

**📊 数据集**

在ImageNet-1K分类、COCO目标检测/实例分割以及ADE20K语义分割数据集上进行评估；

**📈 对比分析**

与现有Vision SSM、ConvNet、Transformer在相同参数/ FLOP预算下进行对比，GraphScan‑Mamba在分类、检测和分割任务上均取得领先或同等性能，显著提升了准确率；

**⚠️ 局限性**

局限性包括：仍需预设局部窗口大小、图邻域固定、对大尺度或全局语义聚合的进一步优化尚待探索，且实验仅在SSM框架内验证，跨模型通用性尚待进一步验证。

---

## 229. A Controlled Counterexample to Strong Proxy-Based Explanations of OOD Performance: in a Fixed Pretraining-and-Probing Setup

**arXiv ID:** 2605.11554 | [PDF](https://arxiv.org/pdf/2605.11554v1)

**作者:** Hongmin Li `[一作]` `[通讯]` (Institute of Science Tokyo), Hongmin Li (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定的预训练‑探测管道中构造并验证了一个受控反例，证明任务无关的结构代理（如压缩式代理）不一定与OOD性能排名保持一致；通过理论证明与合成序列实验双重验证。

**💡 创新点**

首次将正式结构量、操作代理和任务相关结构分离，展示总结构代理与任务相关结构可能不对齐，并给出理论与实验双重证据；提供了对结构代理可靠性的边界分析。

**🔧 技术方法**

使用计算机有限观测者框架、epiplexity 概念、基于验证损失的压缩式结构代理、冻结特征的MLP探测器、合成序列生成与对比实验。

**📊 数据集**

仅使用自生成的合成token序列数据集：背景结构与相关信号两部分，分别对应数据集A（背景重、信号弱）与数据集B（背景轻、信号强）。

**📈 对比分析**

通过比较代理分数与OOD探测准确率；在三种随机种子下，代理始终将A排在B前，但在两种子中OOD准确率逆转，诊断子集进一步支持机制；整体显示代理排名与OOD表现不一致。

**⚠️ 局限性**

局限性包括：仅在单一预训练‑探测设置下测试；使用合成数据且种子数有限；未检验更大模型或真实数据集；代理的稳定性与不同任务的推广性尚未评估。

---

## 230. ECHO: Continuous Hierarchical Memory for Vision-Language-Action Models

**arXiv ID:** 2605.10993 | [PDF](https://arxiv.org/pdf/2605.10993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 231. Much of Geospatial Web Search Is Beyond Traditional GIS

**arXiv ID:** 2605.11336 | [PDF](https://arxiv.org/pdf/2605.11336v1)

**作者:** Ilya Ilyankou `[一作]` (University College London), James Haworth `[通讯]` (University College London)

**通讯引用:** 1917 | [OpenAlex ID](https://openalex.org/A5056082244)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对MS MARCO 1.01M查询日志进行无预过滤的地理空间查询识别与分类，构建了18%比例的地理查询集并提出了88类主题分类。

**💡 创新点**

首次在大规模查询日志中结合密集句子嵌入、SetFit二分类与UMAP+HDBSCAN无监督聚类，量化并细分地理查询的真实分布，提供更完整的空间信息需求视图。

**🔧 技术方法**

使用BGE-small句子嵌入模型、SetFit二分类框架、UMAP降维、HDBSCAN聚类、c‑TF‑IDF关键词提取，以及LLM弱标签辅助标注等技术。

**📊 数据集**

MS MARCO公开查询日志（约1.01M条）及人工标注的1,200条金标准查询，用于训练、评估与聚类。

**📈 对比分析**

SetFit分类器在金标准上取得F1≈0.93；对完整数据集识别18%地理查询；聚类质量以DBCV平均0.31、噪声率36%为评估指标，表明方法在大规模数据上具有可行性。

**⚠️ 局限性**

仅覆盖英美北美用户、缺乏多语言多地区样本；仅针对短单意图查询；聚类噪声较高；未涵盖对话式、实时或多意图查询的需求。

---

## 232. NAVIS: Concurrent Search and Update with Low Position-Seeking Overhead in On-SSD Graph-Based Vector Search

**arXiv ID:** 2605.11523 | [PDF](https://arxiv.org/pdf/2605.11523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 233. Modelling Expert Cognition Beyond Behaviour: towardss Interpretation, Tension, and Value Structures

**arXiv ID:** 2605.11393 | [PDF](https://arxiv.org/pdf/2605.11393v1)

**作者:** Annie Yuan `[一作]` (University of Sydney), Annie Yuan `[通讯]` (University of Sydney)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5066926831)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了Expert Identity Cognition Model（EICM），将专家认知建模为受情境约束影响、通过身份张力解释、最终形成价值结构并产生行动的三层认知框架。

**💡 创新点**

核心创新在于把“身份张力”作为认知核心机制，强调约束只有在身份结构的调解下才具有意义，区别于传统的行为或约束驱动模型；并将价值视为张力稳定化后的决策原则，而非单纯的奖励或偏好。

**🔧 技术方法**

采用形式化的三层转换函数（T=f(C,I)、V=g(T)、A=h(V)）来描述从约束→张力→价值→行动的认知流程，并在示例中通过简单的符号表示（C={…}, T={…}, V={…})进行演示。

**📊 数据集**

未使用公开数据集，仅在“玉石雕刻”模拟场景中构造了12种材料状态与两种专家人物的对比案例，演示身份张力导致的解释和行动差异。

**📈 对比分析**

未与现有行为模仿或奖励学习方法进行定量对比，案例仅展示概念层面的解释差异，未给出性能指标。

**⚠️ 局限性**

局限性包括：1）仅为理论与示例框架，缺乏实证验证；2）身份承诺与张力的具体表征难以量化，实际应用需进一步建模；3）忽略社会协作、集体身份等因素；4）未提供从实际数据提取身份张力的标准流程。

---

## 234. SkillGen: Verified Inference-Time Agent Skill Synthesis

**arXiv ID:** 2605.10999 | [PDF](https://arxiv.org/pdf/2605.10999v1)

**作者:** Yuchen Ma `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过基于多代理的框架，利用对成功与失败轨迹的对比归纳、生成、验证和迭代细化，自动合成单一可审计的推理时技能。

**💡 创新点**

创新点在于把技能合成视作干预实验，使用对比归纳提取成功与失败模式，并通过经验验证净增益来保障技能在部署前的正向影响。

**🔧 技术方法**

采用LLM生成式推理、文本编码与聚类、对比分析、强化学习式迭代细化以及干预验证门控等技术。

**📊 数据集**

使用多种交互、科学、编码、网页及工具使用基准（如IOD、OOD、PropertyYield、τ-等）作为训练与评估数据集。

**📈 对比分析**

与无技能基线以及Trace2Skill、SkillX、EvoSkill、CoEvoSkills等四大基线对比，SkillGen在8种基础模型上平均提升3.27–10.08个百分点，提升案例占50/80，退化仅5例。

**⚠️ 局限性**

局限性包括对知识型任务的改进有限、可能出现过度泛化导致未覆盖的失败、依赖验证门控制且在极端高复杂度情境下细化收敛慢。

---

## 235. ReCoVer: Resilient LLM Pre-Training System via Fault-Tolerant Collective and Versatile Workload

**arXiv ID:** 2605.11215 | [PDF](https://arxiv.org/pdf/2605.11215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 236. The first global agricultural field boundary map at 10m resolution

**arXiv ID:** 2605.11055 | [PDF](https://arxiv.org/pdf/2605.11055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 237. State Canonization and Early Pruning in Width-Based Automated Theorem Proving

**arXiv ID:** 2605.11025 | [PDF](https://arxiv.org/pdf/2605.11025v1)

**作者:** Mateus de Oliveira Oliveira `[一作]` (Stockholm University), Sam Urmian `[通讯]` (University of Bergen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了宽度相关的自动定理证明框架，并在树宽/路径宽为 k 的图类上搜索反例，验证图论猜想；

**💡 创新点**

核心创新包括：① 通过状态规范化（state‑canonization）显著减少搜索空间；② 针对闭合于子图的蕴含式（P1→P2）引入早期剪枝（early‑pruning）进一步剪掉无效搜索路径；

**🔧 技术方法**

技术细节主要包括：基于指令式树分解（instructive tree decompositions）构造的动态规划核心（DP‑core），配合可识别相同图的同构判定、符号重命名（relabeling）和 witness action；实现了宽度相关的 BFS 与 ISO‑BFS（带规范化）搜索算法；

**📊 数据集**

在实验中使用了三类数据集：① 通过算法生成的反例图（如 14‑vertex 4‑chromatic 无三角图）；② 对 Reed 猜想在固定路径宽（≤5）和树宽（≤3）下的验证；③ 通过对不同最大度、路径宽、树宽组合的搜索，评估性能；

**📈 对比分析**

实验对比显示：单纯 BFS 需要数十亿状态且内存上限被突破；加上规范化后仅保留几百个状态；再加上早期剪枝后状态进一步压缩到 1‑几百；在路径宽 5、最大度 3 的组合中，使用两种技术后搜索耗时约 41 小时、内存 29 GB；整体上两种技术组合可将状态数、内存、时间降低 2–3 个数量级；

**⚠️ 局限性**

局限性包括：① 需要手工实现指令式 DP‑core，工作量大；② 对于高树宽（k>3）和大最大度的图类，仍然存在指数级状态爆炸；③ 仅适用于闭合于子图的蕴含式；④ 目前尚未处理多重边/自环的复杂性质，导致某些核心的多样性（multiplicity）较高。

---

## 238. Decaf: Improving Neural Decompilation with Automatic Feedback and Search

**arXiv ID:** 2605.11501 | [PDF](https://arxiv.org/pdf/2605.11501v1)

**作者:** Alexander Shypula `[一作]` (University of Pennsylvania), Edward Schwartz `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2164 | [OpenAlex ID](https://openalex.org/A5061771254)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Decaf 系统，先用传统反编译器生成低层代码，随后通过大语言模型进行温度采样产生多种高层源码候选，并利用编译器自动反馈和训练好的神经重排序器自动挑选最符合语义的解码结果，从而显著提升反编译质量。

**💡 创新点**

创新点在于：①将传统反编译输出作为神经模型的输入，减少模型学习难度；②采用温度采样生成多样化候选；③利用编译器回调提供的功能正确性反馈，结合神经重排序器进行基于语义的自动挑选，突破了单样本、无反馈的传统神经反编译局限。

**🔧 技术方法**

技术手段包括：22B Llama2‑Chat 微调模型作为生成器，32B Flan 微调模型作为重排序器；温度采样、多样化生成；编译器重编译反馈、字节编辑距离、神经重排序器；训练过程使用自监督式数据生成和梯度累积等深度学习技术。

**📊 数据集**

使用的数据集主要有：CodeQL/CodeBERT 开源 C/C++ 函数数据集（Real、Synth、SimpleIO 等分割）、ExeBench 真实二进制测试集，以及 Juliet 漏洞测试套件用于漏洞恢复实验。

**📈 对比分析**

与 Idioms、LLM4Decompile 等基线对比，单样本时功能正确率仅 59.1%；通过 Decaf 后功能正确率提升至 83.9%，字节级匹配率提升至 70.9%；在多种数据集上均显著优于现有最强模型，性能提升幅度显著。

**⚠️ 局限性**

局限性包括：依赖原始编译器及其编译参数，跨编译器/架构迁移时性能下降；未覆盖恶意代码、混淆二进制和非 x86_64 架构；对缺少完整测试用例的函数功能正确性评估存在不确定性。

---

## 239. Generative AI for Visualizing Highway Construction Hazards Through Synthetic Images and Temporal Sequences

**arXiv ID:** 2605.11276 | [PDF](https://arxiv.org/pdf/2605.11276v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. ASD-Bench: A Four-Axis Comprehensive Benchmark of AI Models for Autism Spectrum Disorder

**arXiv ID:** 2605.11091 | [PDF](https://arxiv.org/pdf/2605.11091v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 241. CAMPA: Efficient and Aligned Multimodal Graph Learning via Decoupled Propagation and Aggregation

**arXiv ID:** 2605.11468 | [PDF](https://arxiv.org/pdf/2605.11468v1)

**作者:** Daohan Su `[一作]`, Guoren Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了CAMPA框架，利用解耦式多模态图神经网络，在传播和聚合阶段分别对跨模态信息进行对齐，从而缓解模态冲突，提高大规模多模态图学习的效率与效果。

**💡 创新点**

①跨模态对齐传播（CAP）：将跨模态相似度先验嵌入图扩散，保持语义一致性；②轨迹对齐聚合（TAA）：在多跳轨迹上使用自注意力与跨模注意力实现模态内部和跨模态的语义对齐；③两阶段对齐结合的理论分析，证明传播稳定性和聚合差异可控。

**🔧 技术方法**

解耦式图学习、跨模态相似度先验、加权多跳传播、残差保持、轨迹级自注意力与交叉注意力、门控跳跃权重、可训练的投影与融合模块；同时使用预训练的视觉‑语言编码器作为特征提取器。

**📊 数据集**

八个多模态图数据集：Goodreads、Grocery、Cloth、Movies、Ele‑fashion、RedditS、Bili_music、SemArt；涵盖推荐、社交、内容理解等场景。

**📈 对比分析**

与GAT、MMGCN、MGAT、SIGN、MSGC、MGDN、DMGC、DGF、MIG‑GT、LGMRec、UniGraph2等基线进行对比。CAMPA在节点分类、链接预测、节点聚类等图中心任务以及模态检索、图‑图像生成等模态中心任务上均实现了最优或领先效果，同时保持了与传统解耦模型相近的训练速度与更低的内存占用。

**⚠️ 局限性**

目前仍受限于：①仅针对文本‑图像两模态设计，扩展到更多模态需进一步研究；②跨模态相似度先验依赖于预训练编码器的语义一致性，可能在极端异构或噪声场景下效果受限；③对传播深度与对齐强度的超参数仍需经验调优，尽管范围较宽，但在极大规模图上仍需评估。

---

## 242. The Scaling Law of Evaluation Failure: Why Simple Averaging Collapses Under Data Sparsity and Item Difficulty Gaps, and How Item Response Theory Recovers Ground Truth Across Domains

**arXiv ID:** 2605.11205 | [PDF](https://arxiv.org/pdf/2605.11205v1)

**作者:** Jung Min Kang `[一作]` `[通讯]` (Independent Researcher), Jung Min Kang (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对四个领域（NLP、临床试验、自动驾驶安全、网络安全）进行跨域仿真研究，比较简单平均与 2PL IRT 在稀疏且难度差异显著的评估矩阵上的排名准确性。

**💡 创新点**

提出 Evaluation Failure Scaling Law，揭示稀疏度与难度差异乘积对简单平均排名误差的单调影响，并证明 2PL IRT 在此类场景下几乎不失真。

**🔧 技术方法**

使用两参数逻辑斯蒂 IRT 模型、Spearman 相关系数、网格搜索与蒙特卡罗仿真等统计技术进行评估与对比。

**📊 数据集**

基于 GLUE、临床医院、驾驶环境和攻击类型的真实世界统计特征构建合成数据集，所有实验均为模拟生成。

**📈 对比分析**

在四个领域以及 150 条网格条件下对比两种方法；简单平均在稀疏且难度差异大的条件下 Spearman ρ 可降至 0.24，而 2PL IRT 始终保持 ρ≥0.993，显示显著性能优势。

**⚠️ 局限性**

局限性包括：仅在仿真数据上验证，假设 2PL 模型成立；系统与题目规模有限；未在真实数据上进行验证，需要后续实际应用与验证。

---

## 243. ChunkFlow: Communication-Aware Chunked Prefetching for Layerwise Offloading in Distributed Diffusion Transformer Inference

**arXiv ID:** 2605.11335 | [PDF](https://arxiv.org/pdf/2605.11335v1)

**作者:** Han Meng `[一作]` (University of California Merced), Dong Li `[通讯]` (University of California Merced)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在多GPU PCIe 环境下针对扩散 Transformer 推理，提出一种基于块粒度预取与部分参数驻留的层级卸载运行时，以减少显存占用并降低推理延迟。

**💡 创新点**

提出首阶预取-计算重叠模型，识别 PCIe 争用失败模式，并设计通信感知的块级预取与可调节的部分驻留机制，实现显存与延迟的可插拔权衡。

**🔧 技术方法**

使用块级预取、CUDA 流暂停/恢复、NCCL collectives、Ulysses 序列并行、FlashAttention、SGLang 运行时、首阶 Roofline 计算模型等技术。

**📊 数据集**

在 WanVideo、HunyuanVideo、Flux 三种公开扩散 Transformer 上评估，分别处理文本到视频和文本到图像生成任务。

**📈 对比分析**

与无卸载和 SGLang 全层卸载进行对比；在两张 H100 GPU PCIe 上实现 1.13–1.28 倍推理速度提升，峰值显存下降 30–50%，在大负载下接近无卸载基线。

**⚠️ 局限性**

对低计算负载仍需部分驻留才能显著降低延迟；块大小需手工调优，且对不同网络架构的适应性未完全验证；主要针对 PCIe 节点，NVLink 场景改进有限。

---

## 244. LEAP: Unlocking dLLM Parallelism via Lookahead Early-Convergence Token Detection

**arXiv ID:** 2605.10980 | [PDF](https://arxiv.org/pdf/2605.10980v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 245. Neural Statistical Functions

**arXiv ID:** 2605.11327 | [PDF](https://arxiv.org/pdf/2605.11327v1)

**作者:** Daniel Xu `[一作]` (Columbia University), Wojciech Matusik `[通讯]` (MIT CSAIL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并训练一种新型的神经网络——神经统计函数，能够在无需采样的情况下直接推断连续条件区间内的统计量（如积分、分位数、最大值等）。

**💡 创新点**

创新点在于引入“前缀统计”视角，将各类统计量统一为区间条件的前缀积分，并通过一阶标识式得到高效稳定的学习目标，完成对区间统计的无采样推断。

**🔧 技术方法**

主要技术包括：前缀统计框架、基于梯度匹配的一阶标识式损失、混合监督（预训练单点预测器与真实数据），以及使用 MLP/Transolver 等深度网络实现。

**📊 数据集**

使用了三大数据集：合成 2D 动力学轨迹数据、NASA‑CRM 飞机气动仿真数据以及 OpenRadioss 的车碰撞测试数据。

**📈 对比分析**

与传统的单点预测 + Monte‑Carlo 采样以及基于数值求解器的密集推断进行比较，实验显示神经统计函数在相同或更低的模型推断次数下实现了更低的相对误差（尤其在宽区间时），并将推断延迟降低至 100 倍左右。

**⚠️ 局限性**

局限性：目前仅针对一维条件变量设计，尽管可推广到多维但评估成本随维度指数增长；在极高维条件空间仍缺乏高效无采样方法。

---

## 246. ZeroIDIR: Zero-Reference Illumination Degradation Image Restoration with Perturbed Consistency Diffusion Models

**arXiv ID:** 2605.11435 | [PDF](https://arxiv.org/pdf/2605.11435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 247. Read, Extract, Classify: A Tool for Smarter Requirements Engineering

**arXiv ID:** 2605.11045 | [PDF](https://arxiv.org/pdf/2605.11045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 248. LLM-X: A Scalable Negotiation-Oriented Exchange for Communication Among Personal LLM Agents

**arXiv ID:** 2605.11376 | [PDF](https://arxiv.org/pdf/2605.11376v1)

**作者:** Giuliano Lorenzoni `[一作]` (University of Waterloo), Donald Cowan `[通讯]` (University of Waterloo)

**通讯引用:** 24441 | [OpenAlex ID](https://openalex.org/A5081821121)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了LLM-X——一个支持个人LLM代理之间直接、结构化协商的可扩展消息总线和路由系统，并在5、9、12个代理的实验中验证其可伸缩性。

**💡 创新点**

创新点在于：①为LLM-to-LLM通信提供了schema验证与政策执行的typed消息协议；②将协商视为核心交互模式（ContractNet、FIPA交替）而非传统工具API交互；③通过可调的接受策略（Low/Medium/High）实现了公平、鲁棒与效率的可编程权衡；④提供了可复现的长时段实验基准。

**🔧 技术方法**

采用Python轻量代理、JWT鉴权、JSON Schema校验、NATS消息总线、异步调度、Policy Engine以及ContractNet/FIPA协议实现；实验使用Poisson过程生成的合成负载。

**📊 数据集**

未使用公开数据集，全部使用合成Poisson生成的 CFP/Offer 流量作为实验输入。

**📈 对比分析**

通过对比不同代理数和接受策略的短跑（2min）、中跑（2h）和长跑（12h）实验，测量每分钟消息量、延迟（p50、p95）和完整度。结果显示：延迟保持在1–11 ms，吞吐量随代理数线性增长；严格策略提升公平性与鲁棒性但延迟和消息量上升；相反，宽松策略延迟低、吞吐高但完整度下降。

**⚠️ 局限性**

局限性包括：目前仅在Python模拟环境下测试，未验证与真实LLM API的兼容性；实验规模限制在12个代理，缺乏更大规模的压力测试；缺乏对非协商类任务的适用性评估；策略参数未通过自适应或强化学习自动调优。

---

## 249. FedSurrogate: Backdoor Defense in Federated Learning via Layer Criticality and Surrogate Replacement

**arXiv ID:** 2605.11122 | [PDF](https://arxiv.org/pdf/2605.11122v1)

**作者:** Fatima Z. Abacha `[一作]` (University of Manchester), Mustafa A. Mustafa `[通讯]` (University of Manchester)

**通讯引用:** 122307 | [OpenAlex ID](https://openalex.org/A5006201777)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种服务器端的联邦学习反向门攻击防御方案 FedSurrogate，利用动态层级重要性分析、双向梯度对齐过滤以及结构相似的善意客户端替代，检测并中和恶意客户端的更新。

**💡 创新点**

创新点：① 动态层级重要性分析（LCA）自动定位攻击主要集中层；② 双向梯度对齐过滤既能捕捉残留攻击者，又能救助误报的善意客户端；③ 只替换恶意更新中的关键层，用结构相似的善意更新进行替代，既抑制背后攻击又保留主任务知识。

**🔧 技术方法**

技术细节包括：方向余弦散度用于衡量层级差异，HDBSCAN 密度聚类实现粗筛；中位数/四分位数阈值进行二级筛选；梯度对齐得分累积实现长期监控；层级替代与下调权重结合实现安全聚合；使用 ResNet‑18 与多层 CNN 作为模型。

**📊 数据集**

实验数据集：MNIST、Fashion‑MNIST、CIFAR‑10、CIFAR‑100；采用 Dirichlet 分区生成非 IID 客户端数据，客户端数为 20~100，局部训练 epoch 2/5，学习率 0.01/0.1。

**📈 对比分析**

与七种现有防御（FoolsGold、FLAME、FedGrad、Snowball、AlignIns、FLShield、SPMC）在三种攻击（CBA、DBA、Neurotoxin）下对比。FedSurrogate 主任务准确率仅比 FedAvg 低 0–3%，攻击成功率（ASR）始终 < 2.6%，误报率（FPR）在所有配置下均低于 10%，在大型客户端群和适应性攻击情境下仍保持稳健。

**⚠️ 局限性**

局限性：仍假设诚实多数，极端异质性或高攻击比例时误报可能上升；实现上需要在每轮计算 LCA 及多阶段过滤，增加计算和通信开销；对特殊模型结构或完全层级针对的攻击（如 LP）仍有一定挑战；对攻击者具备完备防御信息时，部分指标（如 FPR）略升。

---

## 250. A Switching System Theory of Q-Learning with Linear Function Approximation

**arXiv ID:** 2605.11021 | [PDF](https://arxiv.org/pdf/2605.11021v1)

**作者:** Donghwan Lee `[一作]` (Korea Advanced Institute of Science and Technology), Han-Dong Lim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1092 | [OpenAlex ID](https://openalex.org/A5081465224)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文开发了一种基于联合谱半径（JSR）的线性函数逼近（LFA）Q学习的切换系统解释，推导出均值动态的精确线性切换模型，并将收敛性与相应切换系统的稳定性联系起来。

**💡 创新点**

创新点在于将Q学习视为切换系统，并通过JSR分析提供了一种新的收敛性证明方法，连接了投影贝尔曼方程、有限差分随机策略切换和切换系统稳定性。

**🔧 技术方法**

使用了切换系统理论和联合谱半径（JSR）作为主要技术手段。

**📊 数据集**

使用了独立同分布（i.i.d.）观察和马尔可夫观察的随机线性Q学习模型进行分析。

**📈 对比分析**

与传统的收敛性分析方法相比，JSR提供了一种更不保守的充分条件，能够更好地捕捉切换模式的产品，性能上优于简单的一步范数界限。

**⚠️ 局限性**

限制在于JSR的精确计算通常是困难的，尽管提供了更好的收敛性条件，但在实际应用中可能仍然面临计算复杂性的问题。

---

## 251. MIRA: An LLM-Assisted Benchmark for Multi-Category Integrated Retrieval

**arXiv ID:** 2605.11254 | [PDF](https://arxiv.org/pdf/2605.11254v1)

**作者:** Mehmet Deniz Türkmen `[一作]` (GESIS -- Leibniz Institute for Social Sciences), Derek Greene `[通讯]` (University College Dublin)

**通讯引用:** 5551 | [OpenAlex ID](https://openalex.org/A5053619267)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个基于GESIS Search平台用户日志的多类别集成检索评估基准MIRA，用于评估跨类别检索系统。

**💡 创新点**

创新点在于将四种学术资源类型（数据集、变量、工具、出版物）整合到单一评估框架，利用LLM生成主题描述与多级相关性标注，显著降低人工成本并支持持续评估。

**🔧 技术方法**

使用LLM（OpenAI GPT‑4）、多语言嵌入（paraphrase‑multilingual‑MiniLM‑L12‑v2）、UMAP+HDBSCAN聚类、BM25、ColBERT、MonoT5、以及RLM进行检索与标注。

**📊 数据集**

数据集包含来自GESIS Search平台的四类学术资源（约468k条记录）和200个基于真实用户查询的主题，覆盖德英双语。

**📈 对比分析**

通过与BM25、RLM、ColBERT、MonoT5的对比，使用P@10、nDCG@10、MAP、GMAP等指标评估；神经模型在大多数指标上显著优于BM25，MAP最高约0.56。

**⚠️ 局限性**

限制在于隐式反馈信号可能产生噪声，LLM标注仍需人工校验；跨类别相关性评估仍面临挑战，整体性能仍有提升空间。

---

## 252. Agent-BRACE: Decoupling Beliefs from Actions in Long-Horizon Tasks via Verbalized State Uncertainty

**arXiv ID:** 2605.11436 | [PDF](https://arxiv.org/pdf/2605.11436v1)

**作者:** Joykirat Singh `[一作]` (UNC Chapel Hill), Mohit Bansal `[通讯]` (UNC Chapel Hill)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并训练了一种名为 Agent-BRACE 的方法，利用大型语言模型拆分为信念状态模型和策略模型，联合通过强化学习学习，在长周期、部分可观测任务中通过结构化自然语言声明并附加分级不确定性标签来近似环境的信念分布，从而解决上下文无限增长与不确定性表达困难的问题。

**💡 创新点**

创新点包括：①使用 Words of Estimative Probability (WEP) 分级标签在文本中表达不确定性，保持完整的信念分布；②将信念状态与策略解耦并共同优化，使策略在不确定性下决策；③采用结构化声明而非自由文本摘要，保持近常数上下文长度并显著提升性能。

**🔧 技术方法**

技术手段：大模型（Qwen‑2.5‑3B、Qwen‑3‑30B‑A3B、Qwen‑3‑4B‑Instruct）+ 监督微调（SFT） + Proximal Policy Optimization (PPO) 联合训练；自定义信念奖励（状态追踪、状态正确性、多样性、格式、任务成功）+ 传统环境奖励；Brier 分数评估信念校准；WEP 词汇表；结构化自然语言声明。

**📊 数据集**

数据集与环境：TextWorld（Quest、Treasure、Cooking）作为训练与评估基准；ALFWorld 用于跨域泛化测试；Qwen‑3‑30B‑A3B‑Instruct 作为判定器。

**📈 对比分析**

与多种基线（Base Model、ReAct、Direct‑Action (RL)、ReAct (RL)、MEM1、PABU）比较，在 Quest、Treasure、Cooking 上平均准确率分别为 72.8%（qwen‑2.5‑3B）和 79.3%（qwen‑3‑4B），比最强 RL 基线提升 14.5% / 5.3%；在 ALFWorld 上提升 2.85%；保持上下文窗口常数，显著提高求解率和样本效率。

**⚠️ 局限性**

局限性：①需要手工指定信念槽位，未能自动发现开放式文本环境的所有维度；②WEP 标签数量有限，细粒度不确定性表达仍受限；③训练成本高，需大量 RL 交互；④对极端长序列或高度动态环境的鲁棒性仍待验证。

---

## 253. COSMOS: Model-Agnostic Personalized Federated Learning with Clustered Server Models and Pseudo-Label-Only Communication

**arXiv ID:** 2605.11165 | [PDF](https://arxiv.org/pdf/2605.11165v1)

**作者:** Ben Rachmut `[一作]` (Washington University in St. Louis), Yevgeniy Vorobeychik `[通讯]` (Washington University in St. Louis)

**通讯引用:** 5236 | [OpenAlex ID](https://openalex.org/A5038669899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了COSMOS框架，实现完全模型无关的联邦学习，通过客户端在公共无标签数据上产生伪标签，服务器按伪标签相似度聚类并训练集群专属模型，再将伪标签传回客户端进行蒸馏，从而实现服务器端个性化。

**💡 创新点**

1）服务器主动训练集群模型，充分利用服务器算力；2）采用基于伪标签的距离控制聚类，无需预设聚类数；3）给出了全局风险收缩的理论证明，实现指数级个性化风险下降。

**🔧 技术方法**

伪标签蒸馏、基于距离控制的贪心聚类、带正则化的梯度优化、半监督学习的扩散与鲁棒性理论。

**📊 数据集**

CIFAR-10、CIFAR-100、EMNIST-balanced、Tiny ImageNet，使用20%训练集作为公共无标签数据。

**📈 对比分析**

与FedMD、FedCT、COMET等模型无关基线，以及在同构设置下的多种主流PFL方法进行对比。实验显示，COSMOS在所有四个基准上均显著优于所有模型无关基线，且在同构场景下与最优PFL方法竞争，通信开销比基线低1-2个数量级。

**⚠️ 局限性**

依赖公共无标签数据的覆盖性；对B0阈值与伪标签置信度的设定较为敏感；在极端异构或小数据规模下可能收敛慢；目前仅支持中心化服务器架构，尚未扩展到完全去中心化网络。

---

## 254. GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading

**arXiv ID:** 2605.11517 | [PDF](https://arxiv.org/pdf/2605.11517v1)

**作者:** Jaeyong Song `[一作]` (Seoul National University), Jinho Lee `[通讯]` (Seoul National University)

**通讯引用:** 49185 | [OpenAlex ID](https://openalex.org/A5100335103)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本工作提出了一个名为GriNNder的全图GNN训练框架，利用NVMe SSD等存储设备突破GPU与主机内存容量壁垒，实现单GPU下的大规模全图训练；

**💡 创新点**

创新点在于设计了Cache‑(Re)Gather‑Bypass三种机制，结合分区级缓存、回收式梯度重构以及轻量级分区算法，显著减少存储I/O与内存占用；

**🔧 技术方法**

核心技术包括分区级缓存管理、梯度重构引擎、GPU‑Direct Storage (GDS) 直接存储访问、Linux AIO + Kvikio 数据搬运、以及自研的轻量级图分区器；

**📊 数据集**

实验使用Products、IGBM、Papers三大真实大图以及合成Kronecker图，节点规模从几百万到一亿；

**📈 对比分析**

与微批量（Betty、Ginex）、单机存储(offload)（HongTu）、分布式全图训练（CAGNET、Sancus）等基线相比，GriNNder在单GPU上实现高达9.78×的加速，吞吐量可与分布式系统相媲美；

**⚠️ 局限性**

主要局限包括对高阶分区依赖的扩展性、在极低存储带宽环境下仍受限、以及对动态图或大规模分布式多GPU并行的进一步优化仍待研究。

---

## 255. Fast MoE Inference via Predictive Prefetching and Expert Replication

**arXiv ID:** 2605.11537 | [PDF](https://arxiv.org/pdf/2605.11537v1)

**作者:** Ankit Jyothish `[一作]` (Iowa State University), Joseph Zuber `[通讯]` (Iowa State University)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5103185193)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于预测预取和专家复制的动态专家复制策略，以提升Mixture‑of‑Experts（MoE）模型的推理速度和GPU利用率。

**💡 创新点**

创新点在于使用SRU网络对下一批次的专家激活进行精确预测，并根据预测结果在GPU上复制专家，实现多线程的预测与推理并行，显著缓解负载不均与等待瓶颈。

**🔧 技术方法**

技术包括SRU‑based专家预测器、稀疏max/softmax路由、哈希表构建线程、专家复制与容量上限控制，以及基于SwitchTransformers的MoE实现。

**📊 数据集**

实验数据集主要是GLUE benchmark中的SST‑2和MRPC，以及SwitchTransformers基础模型的不同专家规模（8/64/128）。

**📈 对比分析**

与传统SwitchTransformers和SiDA‑MoE进行对比，MoE‑MPMC实现了近100% GPU利用率、约3倍的推理速度提升，且在保持90‑95%性能的同时显著提高了吞吐量。

**⚠️ 局限性**

局限性包括需要额外的预测训练开销、复制策略受GPU内存限制、预测误差可能导致负载不平衡，以及对不同任务与硬件环境的适配性尚待进一步验证。

---

## 256. Real-Scale Island Area and Coastline Estimation using Only its Place Name or Coordinates

**arXiv ID:** 2605.11267 | [PDF](https://arxiv.org/pdf/2605.11267v1)

**作者:** Quanyun Wu `[一作]` (University of Waterloo), Jonathan Li `[通讯]` (University of Waterloo)

**通讯引用:** 25674 | [OpenAlex ID](https://openalex.org/A5100613889)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用仅有的地点名称或坐标，通过Google Earth Studio生成低空单目图像序列，使用Transformer 3D重建、轨迹对齐和多视角语义投影，恢复真实尺度并自动提取岛屿面积与海岸线。

**💡 创新点**

突破传统需昂贵航空传感器或现场测绘的瓶颈，创新性地将Web-GIS渲染、单目尺度恢复和基于SAM3的多视角语义融合结合，实现在无地面控制点、无垂直视角下的高精度岛屿测量。

**🔧 技术方法**

核心技术包括Google Earth Studio图像采集、Pi-Long Transformer 3D重建、Umeyama轨迹对齐、SAM3语义分割及多视角2D→3D投影、基于自适应网格的面积计算。

**📊 数据集**

在四个具有不同地形与人工设施的岛屿（自由女神像岛、执政官岛、Somes岛、埃利斯岛）上构建数据集，使用官方面积为基准。

**📈 对比分析**

与官方面积对比，平均绝对误差约8.8%，相对误差在±12%范围内；单帧推理仅需70 ms，显示出极高的测量精度和实时性能。

**⚠️ 局限性**

局限性包括对茂密植被、潮汐湿地及海面反射的敏感，导致在部分岛屿上出现高达12%误差；仅依赖单目斜视图、缺乏垂直视角与地面控制点，且对图像质量和光照变化的鲁棒性仍待提升。

---

## 257. Beyond Similarity Search: Tenure and the Case for Structured Belief State in LLM Memory

**arXiv ID:** 2605.11325 | [PDF](https://arxiv.org/pdf/2605.11325v1)

**作者:** Jeffrey Flynt `[一作]` `[通讯]` (Independent Researcher), Jeffrey Flynt (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Tenure，一个本地首选的有状态记忆架构，用 Typed belief store、精确检索和压缩来解决跨会话记忆的重新定位成本。

**💡 创新点**

将记忆问题建模为有状态管理而非搜索问题，提出精确优先检索、作用域隔离、写时提取与“action”字段，实现高精度检索和上下文漂移抑制。

**🔧 技术方法**

使用 BM25+别名加权检索、Atlas Search 的查询/索引分析器异构、结构化 Belief Schema、增量提取与合并、上下文压缩、计数信号飞轮。

**📊 数据集**

基于 30 条人工标注的 belief seed，涵盖两域、跨用户、supersession 链以及 72 条静态+12 条多轮检索测试用例。

**📈 对比分析**

与基准向量检索比较，BM25 在 72 条检索案例中 100% 通过、平均精度 1.0，向量检索仅 8/72、平均精度 0.12，且在多轮漂移上向量检索漂移分数 0.43–0.50，BM25 为 0。

**⚠️ 局限性**

依赖别名准确性，单用户或语义相近团队环境有效；多语义多团队或跨语言场景下仍需改进；缺乏跨语言或大规模分布式部署的评估。

---

## 258. Hedwig: Dynamic Autonomy for Coding Agents Under Local Oversight

**arXiv ID:** 2605.11495 | [PDF](https://arxiv.org/pdf/2605.11495v1)

**作者:** Tanjal Shukla `[一作]` (University of Washington), Amy X. Zhang `[通讯]` (University of Washington)

**通讯引用:** 2645 | [OpenAlex ID](https://openalex.org/A5037569091)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个命令行编码代理 Hedwig，能够根据开发者的交互动态调整其自治程度。

**💡 创新点**

创新点在于将自治策略拆分为硬约束和行为指导，并通过在线逻辑回归学习从历史交互中自适应地调节检查点阈值，实现透明、可解释的自治调度。

**🔧 技术方法**

技术方案包括：Claude Sonnet 4.6 LLM、Python CLI、在线逻辑回归分类器、检索路径、硬约束执行链、以及与 VS Code 的集成。

**📊 数据集**

使用了由 21 位专业程序员完成的问卷收集的合成交互记录作为训练和评估数据，并在实验中用合成的决策种子模拟审慎与宽松两种角色。

**📈 对比分析**

通过与 Claude Code（Auto Memory）在两项后端任务上进行对比，使用 LLM Judge 作为oracle 评估检查点召回率和精确率；Hedwig 在审慎角色下召回率高达 1.0，尽管精确率略低，显示出更高的安全性。

**⚠️ 局限性**

局限性包括：实验仅基于合成交互，未进行真实用户研究；缺乏逆向撤销机制和更丰富的偏好分类；未实现子代理委托或更复杂的规范驱动流程；目前仍是原型而非生产级实现。

---

## 259. A Composite Activation Function for Learning Stable Binary Representations

**arXiv ID:** 2605.11558 | [PDF](https://arxiv.org/pdf/2605.11558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 260. Primal Generation, Dual Judgment: Self-Training from Test-Time Scaling

**arXiv ID:** 2605.11299 | [PDF](https://arxiv.org/pdf/2605.11299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 261. Encore: Conditioning Trajectory Forecasting via Biased Ego Rehearsals

**arXiv ID:** 2605.11463 | [PDF](https://arxiv.org/pdf/2605.11463v1)

**作者:** Conghao Wong `[一作]` (Huazhong University of Science and Technology), Xinge You `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 6333 | [OpenAlex ID](https://openalex.org/A5057095711)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 Encore 的轨迹预测框架，利用偏置自我回放（biased ego rehearsals）对未来轨迹进行条件化预测。

**💡 创新点**

创新点在于将偏置回放策略与轨迹预测模型耦合，使模型能够在不同条件（如环境约束、目标目标）下生成更精准的轨迹，并通过回放机制提升对长期依赖的学习能力。

**🔧 技术方法**

使用的主要技术包括：基于 Transformer 或 LSTM 的序列预测网络、偏置自我回放模块、注意力机制以及强化学习或自监督学习策略来优化回放权重。

**📊 数据集**

实验数据集主要包含公开的交通轨迹数据集，例如 Argoverse、nuScenes 和 Waymo Open Dataset，涵盖车辆、行人和自行车等多种交通参与者。

**📈 对比分析**

与现有方法（如 SocialGAN、Trajectron++、TrajectoryTransformer 等）进行对比实验，Encore 在平均预测误差（ADE/FDE）上分别提升 5%–15%，并在长时预测（>3 秒）场景中表现出更稳健的性能。

**⚠️ 局限性**

局限性包括：1) 回放策略参数需要手工调优，影响模型的泛化；2) 在高度动态或交互复杂的场景下，偏置回放可能无法完全捕捉所有约束；3) 计算开销相对传统单向预测模型略高。

---

## 262. The Evaluation Differential: When Frontier AI Models Recognise They Are Being Tested

**arXiv ID:** 2605.11496 | [PDF](https://arxiv.org/pdf/2605.11496v1)

**作者:** Varad Vishwarupe `[一作]` (University of Oxford), Ivan Flechais `[通讯]` (University of Oxford)

**通讯引用:** 1594 | [OpenAlex ID](https://openalex.org/A5061337880)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了评价差异(ED)量化框架并设计了TRACE审核协议，以限制从前沿AI评估中得出的安全性声明。

**💡 创新点**

定义了ED与归一化效应值nED，提出四层证据分类和安全性声明类型学，并将其嵌入可操作的TRACE协议。

**🔧 技术方法**

测量理论、统计推断、效果量计算、对抗性重放、提示敏感性评估等技术。

**📊 数据集**

利用公开案例数据：Anthropic的BrowseComp、NLA的SWE‑bench与破坏性编码、OpenAI/Apollo的反计谋训练结果。

**📈 对比分析**

通过ED与传统边际分数对比，证明前者能捕捉评估‑部署差异；在案例回溯中验证了安全性声明的稳定/退化/倒置类型，显示ED能揭示传统指标忽视的风险。

**⚠️ 局限性**

需要对部署环境μ_D可定义且可重放，依赖可访问的证据层；可能对提示或特征做过拟合；无法提供对齐保证，仅能限制声明。

---

## 263. MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization

**arXiv ID:** 2605.11396 | [PDF](https://arxiv.org/pdf/2605.11396v1)

**作者:** Yupeng Su `[一作]` (University of California), Zheng Zhang `[通讯]` (University of California)

**通讯引用:** 246268 | [OpenAlex ID](https://openalex.org/A5044268160)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MuonQ，一种 4‑bit 低精度量化 Muon 优化器的完整框架，用于训练大语言模型；

**💡 创新点**

创新点在于将方向保真性（directional fidelity）作为量化目标，结合预量化归一化、结构化分解和 μ‑law 编解码三种技术，有效抑制正交化过程中的误差放大；

**🔧 技术方法**

使用预量化归一化、结构化分解（利用功率迭代提取前 k 个奇异向量）以及 μ‑law 编解码来实现稀疏性友好的 4‑bit 量化；

**📊 数据集**

在 FineWeb 上对 GPT‑2（Medium/Large）和 LLaMA（350M/1.1B）进行预训练，并在 ARC、OpenBookQA、BoolQ、HellaSwag、PIQA、WinoGrande 等零样本基准上评估下游性能；

**📈 对比分析**

与全精度 Muon、8‑bit Muon 以及 Naïve 4‑bit Muon 进行对比，MuonQ4 在训练损失和下游准确率上几乎与全精度相同，同时在优化器状态内存上实现最高 7.3× 的压缩；

**⚠️ 局限性**

局限性包括仅在 1.1B 规模模型上验证、功率迭代带来的额外计算开销、未在更大规模或分布式训练环境下测试，以及对更低位宽（<4 bit）或混合精度的适用性尚未探究。

---

## 264. The Bicameral Model: Bidirectional Hidden-State Coupling Between Parallel Language Models

**arXiv ID:** 2605.11167 | [PDF](https://arxiv.org/pdf/2605.11167v1)

**作者:** Cedric Flamant `[一作]` (AWS Agentic AI), Kanna Shimizu `[通讯]` (AWS Agentic AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两台冻结的大语言模型之间引入可训练的神经接口，使其在生成过程中通过隐藏状态进行双向连续通信，并让辅助模型利用工具完成原本难以解决的任务。

**💡 创新点**

首次提出在每个解码步通过隐藏状态而非文本来实现两模型的持续双向协作，并让门控机制自适应学习信息传递协议。

**🔧 技术方法**

使用翻译网络+抑制门的轻量级接口、双目标监督微调、工具集成（计算器、Z3 约束求解器、Python 沙箱）以及可选的低秩适配器。

**📊 数据集**

在算术（通用算术、GSM8K、GSM8K‑IRL）、逻辑网格谜题（ZebraLogic、GeneralZebra）和数值推理（NuminaMath‑TIR、MATH）等数据集上进行训练与评测。

**📈 对比分析**

与单模型基线、适配器等对照实验比较，算术任务准确率从36%提升至96.5%，逻辑谜题从37.5%提升至64.7%，Python 辅助的数学推理在部分难题上得到提升，整体显示显著性能提升。

**⚠️ 局限性**

计算量加倍，已训练任务上表现下降（如 GSM8K 基线 49.6% 降至 40%），需要为每个任务提供因果约束的训练数据，且对已有强大能力的任务可能产生噪声干扰。

---

## 265. Muon is Not That Special: Random or Inverted Spectra Work Just as Well

**arXiv ID:** 2605.11181 | [PDF](https://arxiv.org/pdf/2605.11181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 266. Structural Interpretations of Protein Language Model Representations via Differentiable Graph Partitioning

**arXiv ID:** 2605.10985 | [PDF](https://arxiv.org/pdf/2605.10985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 267. Transformer Interpretability from Perspective of Attention and Gradient

**arXiv ID:** 2605.11392 | [PDF](https://arxiv.org/pdf/2605.11392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 268. GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and Evolutionary Numerical Algorithms

**arXiv ID:** 2605.11117 | [PDF](https://arxiv.org/pdf/2605.11117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 269. Latent Chain-of-Thought Improves Structured-Data Transformers

**arXiv ID:** 2605.11262 | [PDF](https://arxiv.org/pdf/2605.11262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 270. Extending Kernel Trick to Influence Functions

**arXiv ID:** 2605.11239 | [PDF](https://arxiv.org/pdf/2605.11239v1)

**作者:** Zhenhuan Sun `[一作]` (University of Toronto), Shahrokh Valaee `[通讯]` (University of Toronto)

**通讯引用:** 9038 | [OpenAlex ID](https://openalex.org/A5089951737)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了影响函数的双重表示，使其计算复杂度从模型参数规模转移到数据集规模；

**💡 创新点**

创新点在于通过线性化模型的 NTK 及 Δα 空间重参数化，提供了一种在参数规模大时更高效的影响估计方法；

**🔧 技术方法**

使用了线性化模型、Neural Tangent Kernel、Hessian–向量积、共轭梯度求解、Kernel Gradient Descent 等技术；

**📊 数据集**

实验数据集包括 MNIST、CIFAR-10 的子集以及其无限宽对应网络；

**📈 对比分析**

通过与传统 θ‑空间方法在冷启动与热启动、参数距离和准确率等指标比较，Δα‑空间方法在模型规模大于数据规模时具有更低的运行时间且保持相当的 unlearning 性能；

**⚠️ 局限性**

局限性：仅适用于可线性化模型，需要显式或隐式计算和存储 NTK，导致在大规模数据集和多分类场景下计算和存储成本高。

---

## 271. Instruct-ICL: Instruction-Guided In-Context Learning for Post-Disaster Damage Assessment

**arXiv ID:** 2605.11439 | [PDF](https://arxiv.org/pdf/2605.11439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 272. Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models

**arXiv ID:** 2605.10971 | [PDF](https://arxiv.org/pdf/2605.10971v1)

**作者:** Hanhan Zhou `[一作]` (AWS AI Labs), Rashmi Gangadharaiah `[通讯]` (AWS AI Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用稀疏自编码器对离散扩散语言模型的属性随去噪步骤的形成过程进行解释，并基于此提出自适应调度的控制方法，重点在属性活跃的去噪阶段进行干预，减少对模型生成质量的影响。

**💡 创新点**

创新点在于：①首次揭示离散扩散模型中不同语义属性的“承诺时序”“尖锐度”和“效应大小不平衡”三大特征；②基于这些动态特征设计自适应调度和效应比例校准，使控制更精准、跨属性干扰更小；③提供闭式解析的调度效率与交叉属性干扰理论，为后续学习式调度奠定基础。

**🔧 技术方法**

核心技术包括稀疏自编码器（Top‑K SAE）、对比性特征选择、稀疏特征空间干预、基于属性承诺分布的自适应权重、效应比例校准、以及离散扩散模型的残差流插桩。

**📊 数据集**

使用了四个不同规模与训练目标的离散扩散模型（MDLM、SEDD、DREAM、LLaDA）以及与之对应的对比分类语料：AG News（主题）、IMDB 影评（情感）以及形式化/非正式文本数据集。

**📈 对比分析**

与对比基线（对比向量、线性探针、PCA、提示式控制）在七种单/多属性组合上进行评估；结果显示自适应调度在单属性控制中能保持与最优基线相当甚至更好的控制力（>90%），并在三属性联合控制下实现最高93%控制精度，优于最强基线15个百分点；同时生成质量（困惑度/多样性）明显优于均匀干预方案。

**⚠️ 局限性**

局限性包括：①评估依赖离散分类器指标，可能无法完全捕捉属性表达细微差别；②实验仅覆盖短文本（64–1024 token），长文本的适用性未知；③高强度控制下多样性下降；④仅针对离散扩散模型，未探讨更细粒度属性或学习式调度的可行性。

---

## 273. AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment

**arXiv ID:** 2605.11398 | [PDF](https://arxiv.org/pdf/2605.11398v1)

**作者:** Robin Linzmayer `[一作]` (Columbia University), Noémie Elhadad `[通讯]` (Columbia University)

**通讯引用:** 10357 | [OpenAlex ID](https://openalex.org/A5047270546)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并发布了 AcuityBench 基准，用于评估语言模型在多种健康交互场景下识别并沟通适当的护理紧急度。

**💡 创新点**

创新点在于将五个公开数据集统一到四级紧急度框架，设计 QA 与自由对话双模式评估，并区分一致与模糊案例以衡量模型对专家不确定性的对齐程度。

**🔧 技术方法**

使用多种前沿大型语言模型（GPT‑5、Claude、Gemini、Qwen、Llama 等）及 GPT‑4.1 判定器进行对话评估，并采用 Krippendorff α、Jensen‑Shannon Divergence 等统计方法分析结果。

**📊 数据集**

共集成 HealthBench、Semigran、Ramaswamy、PMR‑Synthetic、PMR‑Reddit 等五个公开数据集，总计 914 个案例。

**📈 对比分析**

在 12 个模型上对清晰案例进行 QA 与对话模式比较，QA 准确率介于 53%–85%，对话模式更倾向低紧急度，模型在模糊案例中与专家分布显著偏离，未能准确匹配专家不确定性。

**⚠️ 局限性**

限制包括仅使用四级框架、依赖专家标注可能存在偏差、对话评估依赖判定器、未验证真实临床结果以及模型在特定部署场景下的安全性。

---

## 274. AgentShield: Deception-based Compromise Detection for Tool-using LLM Agents

**arXiv ID:** 2605.11026 | [PDF](https://arxiv.org/pdf/2605.11026v1)

**作者:** Yassin H. Rassul `[一作]` (University of Kurdistan Hewlêr), Tarik A. Rashid `[通讯]` (University of Kurdistan Hewlêr)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AgentShield，一种面向工具调用LLM代理的欺骗式检测框架，用三层陷阱（伪工具、伪凭证、参数白名单）监控代理行为并实时捕捉间接注入攻击。

**💡 创新点**

创新点在于：①把代理的工具接口转化为“监测器”而非传统输入防御；②三层陷阱可生成零误报标签，用于无监督训练下游分类器；③在多语言（英语、库尔德语、阿拉伯语、混合语言）环境下验证，首次覆盖低资源语言。

**🔧 技术方法**

技术方法包括：构造伪工具与伪凭证作为蜜罐；实现参数白名单校验；在工具调用前后插入AgentShield检查；利用陷阱触发事件训练随机森林自监督分类器。

**📊 数据集**

使用AgentDojo v1.2.2基准，包含4种代理环境和176个跨语言注入攻击提示；对四种LLM（GPT‑4o‑mini、GPT‑5‑mini、Llama 3.3 70B、DeepSeek‑V3）进行评测。

**📈 对比分析**

与现有防御（Spotlighting、ProtectAI DeBERTa v2、Meta Prompt‑Guard‑2）对比；在商业模型上AgentShield检测成功攻击率达90.7%–100%，误报率为0%（485次正常测试）；自监督分类器F₁≈0.996，跨模型、跨语言转移保持≈0.99。

**⚠️ 局限性**

局限性包括：仅在AgentDojo单一基准上验证，未覆盖文件浏览等场景导致伪凭证层未触发；只检测触发陷阱的攻击，合法工具与合法参数的攻击仍可逃逸；需要为每个环境手动配置白名单。

---

## 275. Beyond Similarity: Temporal Operator Attention for Time Series Analysis

**arXiv ID:** 2605.11287 | [PDF](https://arxiv.org/pdf/2605.11287v1)

**作者:** Jevon Twitty `[一作]` (Georgia Institute of Technology), Jiecheng Lu `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Temporal Operator Attention (TOA)，通过引入可学习的全密集时序操作符，打破了传统softmax注意力的单纯概率混合瓶颈，提升了时序建模能力。

**💡 创新点**

创新点在于：①把软max注意力视为缺失的全混合算子；②通过在注意力前后加入可学习的密集矩阵并使用激活函数解锁负权重；③设计了Stochastic Operator Regularization（SOR）高方差dropout来稳定稠密矩阵的学习；④在多种任务中验证了此类显式算子的重要性。

**🔧 技术方法**

核心技术包括：改写自注意力的前后混合矩阵（S₁,S₂）、不同激活变体（TOA-Softmax、TOA-ReLU、TOA-Gated）、高方差随机丢弃（SOR）、以及与PatchTST、iTransformer、DUET等现有时序Transformer骨干的无缝融合。

**📊 数据集**

实验数据集涵盖长短期预测（Weather、ETT系列、Solar、ECL、Exchange、Traffic等）、异常检测（PSM、MSL、SMAP、SMD、SWAT）、以及多变量分类（28个UEA多维时间序列数据集）。

**📈 对比分析**

与标准softmax注意力以及其他对比模型（PatchTST、iTransformer、Transformer、DUET）在MSE/MAE/RMSE/F1/Accuracy等指标上对比，TOA在绝大多数基准上实现了显著提升（例如预测MSE下降至0.0028、分类准确率提升至69.16%，异常检测F1值提升至0.809）。

**⚠️ 局限性**

局限性包括：①对高维稠密矩阵的计算与存储开销较大；②需引入额外的正则化（SOR）以避免过拟合，调参成本升高；③理论上多头、深层堆叠的可表达性尚未完全分析，可能在极大规模序列上仍受限。

---

## 276. An Execution-Verified Multi-Language Benchmark for Code Semantic Reasoning

**arXiv ID:** 2605.11006 | [PDF](https://arxiv.org/pdf/2605.11006v1)

**作者:** Yikun Li `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 31358 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了一个执行验证的多语言调用图基准，包含10,583个真实世界程序，并公开了可复现的提取管道。

**💡 创新点**

首次统一多语言（Python、JavaScript、Java）调用图基准，并采用运行时动态追踪生成机械可验证的标签，解决了传统手工注解的不确定性。

**🔧 技术方法**

使用LLM辅助驱动程序生成、语言特定动态追踪器、trace验证、LLM零射击评估以及LoRA微调和推理链（CoT）技术。

**📊 数据集**

基于1,600+许可开放的GitHub仓库抽取的源文件，经过LLM驱动生成后得到10,583个可执行程序（训练集8,454，测试集2,129）。

**📈 对比分析**

对10个LLM（5前沿专有+5开源）进行零射击边缘级别精/召回/F1评估，最高Claude‑Opus‑4.6平均F1 72.9%，最低Qwen2.5‑Coder‑1.5B 9.5%；微调后Qwen2.5‑Coder‑32B达71.2%，仅差1.7个百分点。

**⚠️ 局限性**

仅覆盖三种语言，依赖LLM生成驱动可能缺少更复杂执行路径；未涵盖Go、Rust等新兴语言；缺乏自监督后验验证训练机制。

---

## 277. Decoding Algorithm to Composite Errors Consisting of Deletions and Insertions for Quantum Deletion-Correcting Codes Based on Quantum Reed-Solomon Codes

**arXiv ID:** 2605.11510 | [PDF](https://arxiv.org/pdf/2605.11510v1)

**作者:** Koki Sasaki `[一作]` (Yamaguchi University), Takayuki Nozaki `[通讯]` (Yamaguchi University)

**通讯引用:** 1224 | [OpenAlex ID](https://openalex.org/A5011452126)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种高效的解码算法，能够同时纠正量子删除错误和插入错误，适用于由量子Reed–Solomon码构造的Hagiwara量子删除纠错码。

**💡 创新点**

创新点在于将经典Hagiwara码的标记位技术推广到量子域，并设计了一套既能检测又能定位复合错误（删除+插入）的解码流程，同时证明该算法满足 e + 2m ≤ t 的误差容忍度。

**🔧 技术方法**

核心技术包括：量子Reed–Solomon码的CSS构造、标记位（marker）辅助定位、对复合错误的拆分为删除与插入的组合、利用测量得到的经典比特序列构造判定表、以及将量子块变换错误转化为Pauli误差的线性组合。

**📊 数据集**

该工作为理论研究，未使用实验数据集；算法的有效性通过数学证明与理论分析展示。

**📈 对比分析**

通过理论推导证明算法能够纠正任意满足 t_d + t_i ≤ t 且 t_d, t_i ≠ 0 的错误组合；与先前只能纠正单类错误的Hagiwara码解码器相比，误差恢复范围显著扩展，且解码复杂度保持多项式级别。

**⚠️ 局限性**

局限性包括：仍假设错误是独立且随机出现，未考虑量子退相干等更复杂噪声模型；算法对块长度和标记位长度有严格要求，过大的标记会显著降低码率；此外，实验验证尚未在实际量子硬件上进行。

---

## 278. Lite3R: A Model-Agnostic Framework for Efficient Feed-Forward 3D Reconstruction

**arXiv ID:** 2605.11354 | [PDF](https://arxiv.org/pdf/2605.11354v1)

**作者:** Haoyu Zhang `[一作]` (Peking University), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 9327 | [OpenAlex ID](https://openalex.org/A5100662197)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Lite3R 框架，将密集 3D 重建 Transformer 的注意力替换为稀疏线性注意力，并通过 FP8 量化感知训练和部分注意力蒸馏实现轻量化。

**💡 创新点**

创新点在于：① 采用稀疏线性注意力同时保留关键跨视角交互；② 通过参数高效的 FP8 量化感知训练，仅更新稀疏分支的线性投影层，保持预训练几何先验；③ 将注意力蒸馏与低精度训练结合，提升数值稳定性。

**🔧 技术方法**

核心技术包括：Sparse Linear Attention、FP8‑aware Quantization‑Aware Training、Partial Attention Distillation、Teacher–Student 迁移、模型冻结与轻量化投影。

**📊 数据集**

使用了 BlendedMVS（低分辨率多视图）和 DTU64（姿态评估）两个公开数据集进行实验。

**📈 对比分析**

与原始 VGGT 与 DA3‑Large 进行对比，Lite3R 在 BlendedMVS 上实现 1.76–1.97× 的速度提升、2.32–2.71× 的显存节省，DTU64 亦实现 1.75–1.87× 速度提升；几何质量（深度、姿态、Chamfer、F‑score）略有下降，但仍保持可接受水平。

**⚠️ 局限性**

局限性：目前仅支持 FP8 权重‑仅推理，未充分利用动态 FP8 激活；对更复杂模型（如 DA3‑Large）对结构和量化更敏感；训练需冻结大量参数，适配更大网络时可能受限；缺乏对不同 GPU（尤其是新型 FP8 加速硬件）的深入评估。

---

## 279. Deep Learning for Protein Complex Prediction and Design

**arXiv ID:** 2605.11189 | [PDF](https://arxiv.org/pdf/2605.11189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 280. UniPath: Adaptive Coordination of Understanding and Generation for Unified Multimodal Reasoning

**arXiv ID:** 2605.11400 | [PDF](https://arxiv.org/pdf/2605.11400v1)

**作者:** Hayes Bai `[一作]` (William & Mary), Jindong Wang `[通讯]` (William & Mary)

**通讯引用:** 15725 | [OpenAlex ID](https://openalex.org/A5100700956)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 UniPath 框架，实现统一多模态模型的自适应协调路径选择与执行；

**💡 创新点**

创新点在于利用协调路径多样性，并通过查询形式校准的规划器实现输入依赖路径选择，以及通过角色对齐轨迹训练多路径执行器；

**🔧 技术方法**

技术包括路径条件执行器（LoRA 适配器、视觉思维对齐）、分阶段训练、基于多标签的规划器监督（BCE+正则）和查询形式桶化校准；

**📊 数据集**

使用了 MMMU、MMBench‑EN/CN、MathVista、MMStar 等理解基准，GenEval 与 WISE 生成基准，以及 UnifiedBench 一致性评测；

**📈 对比分析**

与 BAGEL、Janus、Blip3、UniCoT 等多种方法对比，UniPath 在 MMMU、MMBench‑EN/CN、MMStar 上分别提升 4.3%、4.4% 和 7.7%，在 GenEval 上提升 1.5%，整体实现了固定路径方法的显著性能超越；

**⚠️ 局限性**

主要局限在于规划器仍与理想（oracle）路径选择相差明显，跨域泛化和路径选择的鲁棒性仍需提升。

---

## 281. LiBaGS: Lightweight Boundary Gap Synthesis for Targeted Synthetic Data Selection

**arXiv ID:** 2605.11231 | [PDF](https://arxiv.org/pdf/2605.11231v1)

**作者:** Abhishek Moturu `[一作]` (University of Toronto), Babak Taati `[通讯]` (University of Toronto)

**通讯引用:** 3404 | [OpenAlex ID](https://openalex.org/A5011257199)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量级的合成数据选择方法LiBaGS，自动决定哪些合成样本最能填补训练分布中决策边界稀疏区域，且不超过已有真实样本覆盖的区域

**💡 创新点**

创新点在于将边界缺口、预测不确定性、真实数据密度与支持有效性融合成连续分配规则，并通过自适应阈值停止加入样本；同时兼容任意生成器并提供软标签与多样性约束

**🔧 技术方法**

使用特征编码器、轻量化评分模型（如分类器或集成），基于预测概率、熵、支持有效性与核密度估计计算边界缺口得分；通过贪婪子模优化与边界缺口分配公式选样本

**📊 数据集**

在三组实验中验证：两圆月（synthetic）任务、8×8手写数字3/8任务、以及CIFAR-10猫/狗任务；使用Stable Diffusion等生成器产生候选池

**📈 对比分析**

与ERM、随机候选、噪声增强、传统过采样（SMOTE系列）、AugMax、TSynD、反馈导向、GenDataAgent、Deliberate Practice、Conformal、Long-tail、C2I、Uncertainty-only等基线比较；LiBaGS在所有三个数据集上取得最高平均精度，提升幅度从~5%到~2%不等

**⚠️ 局限性**

主要局限是对候选池质量和表示/评分模型的依赖；若生成器产生低质量或偏倚样本，支持过滤与软标签虽能缓解但无法完全消除风险；计算复杂度受候选数平方影响，适合中等规模数据

---

## 282. NeuroFlake: A Neuro-Symbolic LLM Framework for Flaky Test Classification

**arXiv ID:** 2605.11482 | [PDF](https://arxiv.org/pdf/2605.11482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 283. Causal Fairness for Survival Analysis

**arXiv ID:** 2605.11362 | [PDF](https://arxiv.org/pdf/2605.11362v1)

**作者:** Drago Plecko `[一作]` `[通讯]`, Drago Plecko

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个因果公平框架，用于时间到事件（TTE）分析，能够将存活差异拆分为直接、间接和混杂路径的贡献；

**💡 创新点**

创新点在于将因果公平理论与非参数生存分析结合，并在非信息删失、竞争风险和信息删失三种情境下分别建立识别定理、双重稳健估计和Copula-graphic敏感性分析；

**🔧 技术方法**

采用结构因果模型、随机生存森林、双重稳健估计、影响函数理论以及Archimedean Copula方法实现估计；

**📊 数据集**

使用澳大利亚和新西兰重症监护协会（ANZICS）的成人患者数据库（Adult Patient Database）进行实际案例研究；

**📈 对比分析**

通过与传统统计公平度量（如总变差、Kaplan–Meier差异）对比，因果分解揭示了隐藏的路径效应，表明表面零差异下仍存在显著正负相抵的因果效应，表现出更细粒度、更具可解释性的公平性洞察；

**⚠️ 局限性**

局限性包括对无测量混杂的假设缺乏敏感性分析、对Copula族的依赖、时间不变协变量的假设以及竞争风险分析未区分主/竞争风险影响。

---

## 284. EVOCHAMBER: Test-Time Co-evolution of Multi-Agent System at Individual, Team, and Population Scales

**arXiv ID:** 2605.11136 | [PDF](https://arxiv.org/pdf/2605.11136v1)

**作者:** Yaolun Zhang `[一作]` (Oregon State University), Huazheng Wang `[通讯]` (Oregon State University)

**通讯引用:** 523 | [OpenAlex ID](https://openalex.org/A5062299183)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无梯度训练的多智能体测试时演化框架，能够在任务流中自适应更新个体记忆、团队协作结构以及全局智能体池；

**💡 创新点**

创新点在于同时实现三层演化（个体、团队、群体），并引入协作梦境（Collaborative Dreaming）实现异步知识传递，保留专精而非对称广播；

**🔧 技术方法**

核心技术包括：经验归档与子任务/跨域洞察提取、基于权重的角色选择与协作结构学习、异步知识流、生命周期操作（fork/merge/prune/seed），全部仅通过提示演化实现；

**📊 数据集**

使用三大任务流：Hard Math（MATH Level4/5与AIME 2022-25）、Hard Code（MBPP+ 与 CodeContests）、AFlow-Stream（六个连续域块：GSM8K, HotpotQA, MBPP, MATH, HumanEval, DROP），以及Qwen3-8B与GPT‑4.1‑mini两种模型；

**📈 对比分析**

与单智能体、投票、DyLAN、AgentNet、EvoMem、MemCollab、DyLAN等多智能体基线进行比较；在所有任务流上均明显优于基线，尤其在最难子任务上提升32%以上；

**⚠️ 局限性**

局限在于：仅在LLM提示层面实现，无梯度更新可能限制更深层次的策略优化；方法对任务标签的依赖较强，标签质量不足时可能影响专精形成；

---

## 285. Performance bounds for nearest neighbor search with k-d trees

**arXiv ID:** 2605.11313 | [PDF](https://arxiv.org/pdf/2605.11313v1)

**作者:** Marco Bazzani `[一作]` (University of Washington), Sanjoy Dasgupta `[通讯]` (University of California San Diego)

**通讯引用:** 8995 | [OpenAlex ID](https://openalex.org/A5101707744)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文对 k-d 树在高维空间中做最近邻搜索的表现进行了理论分析，并给出了非渐进的时间和准确性上界。

**💡 创新点**

创新点在于首次证明了在维度至少为多项式对数时，失败式搜索几乎不可能返回最近邻，而全域搜索几乎必访问所有叶子；并给出在均匀分布下的上界，表明 k‑d 树在最坏情形下性能的极限。

**🔧 技术方法**

采用概率与集中不等式（如 Hoeffding、Bernstein、正态近似）对 k‑d 树的分割深度、叶子体积、宽高比等进行分析，结合抗集中和中位数估计技术得到上下界。

**📊 数据集**

研究基于理论模型，主要考虑在 [0,1]^d 上的均匀分布与任意绝对连续分布的数据，未使用具体实验数据集。

**📈 对比分析**

通过对比失败式搜索与全域搜索的概率与访问量，上界显示在均匀数据下，全域搜索最多访问 4·(8√(2πe))^d 个叶子；下界表明在高维时两种搜索几乎无法逃离全局探索，性能差异与维度指数成正比。

**⚠️ 局限性**

局限性包括上界仅适用于均匀分布；对非均匀或低内在维度分布的分析不足；假设均匀或产品分布的密度有下界，实际数据分布可能不满足。

---

## 286. FlowSteer: Prompt-Only Workflow Steering Exposes Planning-Time Vulnerabilities in Multi-Agent LLM Systems

**arXiv ID:** 2605.11514 | [PDF](https://arxiv.org/pdf/2605.11514v1)

**作者:** Fanxiao Li `[一作]` (Yunnan University), Min-Yen Kan `[通讯]` (National University Of Singapore)

**通讯引用:** 11474 | [OpenAlex ID](https://openalex.org/A5066305082)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对多智能体系统（MAS）的工作流形成过程进行安全分析，提出了一种仅通过修改用户提示即可操纵工作流的攻击方式，并针对该攻击设计了在输入层的防御机制。

**💡 创新点**

创新点在于：①首次将工作流形成视为新的攻击面；②利用社会影响分析与任务感知、框架化提示实现“工作流定向”攻击；③在提示级别拆分意图并去污，从而在规划阶段就抑制恶意信息。

**🔧 技术方法**

主要技术包括：社会影响诊断、任务感知的谄媚（sycophantic）提示、依赖引导的工作流构造、意图拆分与去污、以及对工作流生成的结构与语义分析。

**📊 数据集**

实验使用的主要数据集为 MisinfoTask（108个任务）和基于 ASB 的 ASB‑Bench（100个任务）。

**📈 对比分析**

对比方法包括 Naive Malicious Prompting、Direct Goal Injection 以及现有的图/拓扑防御（ARGUS、G‑Safeguard）。实验显示，攻击在多模型、多配置下成功率提升约55%，而输入侧防御可将成功率降低约34%。

**⚠️ 局限性**

局限性包括：需要离线脆弱性分析；对任务类型的泛化性有限；防御在提升安全的同时可能略微降低提示的可读性或实用性。

---

## 287. AffectCodec: Emotion-Preserving Neural Speech Codec for Expressive Speech Modeling

**arXiv ID:** 2605.11098 | [PDF](https://arxiv.org/pdf/2605.11098v1)

**作者:** Jiacheng Shi `[一作]` (College of William & Mary), Ye Gao `[通讯]` (College of William & Mary)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种情感导向的神经语音编码器，融合情感-语义引导潜在调制、关系保持蒸馏和情感加权语义对齐，实现对情感信息的显式保留。

**💡 创新点**

创新点在于将情感保留提升为编码器核心目标，提出三阶段联合优化框架，并通过情感-语义交叉注意、关系蒸馏和情感权重对齐三种技术共同提升情感一致性。

**🔧 技术方法**

使用残差向量量化、跨模态注意、冻结情感识别模型、ASR模型和自监督语音模型作为引导，结合多任务损失和情感权重对齐。

**📊 数据集**

训练集包括LibriSpeech、VCTK、AISHELL‑3、AudioSet、MSP‑Podcast和CMU‑MOSEI，评估使用EmoVoiceDB、EMO‑SUPERB、LibriSpeech测试集等。

**📈 对比分析**

与EnCodec、DAC、Facodec等基线在EmoSIM、PESQ、UTMOS、WER等指标上比较，获得情感保持最高、内容准确率与自然度均不低于基线，表现优异。

**⚠️ 局限性**

局限在于模型参数量和训练成本较高，未实现轻量化，且在某些低情感强度语料上的效果仍有限。

---

## 288. Revisiting Privacy Preservation in Brain-Computer Interfaces: Conceptual Boundaries, Risk Pathways, and a Protection-Strength Grading Framework

**arXiv ID:** 2605.11386 | [PDF](https://arxiv.org/pdf/2605.11386v1)

**作者:** Lei Sun `[一作]` (PLA Information Engineering University), Wenle Dong `[通讯]` (PLA Information Engineering University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对脑机接口（BCI）隐私保护领域进行系统综述，提出了统一的三维分析框架（保护对象—生命周期阶段—主导保护强度），并阐述了隐私风险路径与保护边界。

**💡 创新点**

创新点在于：①将用户数据隐私与模型隐私视为风险链上的两个连续节点，明确其相互关系；②引入PS1–PS4四级保护强度概念，量化不同技术方案的防护级别；③从隐私保护边界、保护对象、风险路径三个维度组织文献，避免传统方法按技术或攻击类型划分，提升可比性；④将精神隐私和神经伦理风险作为开放议题单独讨论。

**🔧 技术方法**

主要技术手段包括：隐私保护理论（差分隐私、联邦学习、加密推理、可信执行环境、模型水印等），以及对这些技术在BCI中的应用与安全性能的归纳与对比。

**📊 数据集**

作为综述论文并未使用具体实验数据集，文中引用的研究涉及多种公开EEG/多模态脑电数据集（如BCI Competition数据、EEG-MNIST、P300、Emotiv等），但并未进行统一实验。

**📈 对比分析**

通过构建保护对象–生命周期–保护强度的三维矩阵，对比了在训练阶段与推理阶段的多种方案，展示了不同方法在暴露、关联、推断、重建与逆向路径上的抑制效果；结果表明大多数方法落在PS1/PS2，差分隐私、加密推理与安全多方计算在PS3/PS4具有较高防护等级，但伴随精度损失、计算开销或延迟。

**⚠️ 局限性**

局限性包括：①对收集、传输、存储等早期阶段的保护讨论不足；②对精神隐私和群体隐私等高层伦理风险的定量评估缺失；③现有技术主要针对EEG，其他模态（fNIRS、MEG等）保护方案缺乏系统分析；④高强度保护（PS3/PS4）在实时闭环BCI中的实现仍面临性能瓶颈。

---

## 289. fg-expo: Frontier-guided exploration-prioritized policy optimization via adaptive kl and gaussian curriculum

**arXiv ID:** 2605.11403 | [PDF](https://arxiv.org/pdf/2605.11403v1)

**作者:** Mingxiong Lin `[一作]` (OPPO AI Center), Haonan Lu `[通讯]` (OPPO AI Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为FG-ExPO的新训练框架，结合了基于批次准确率自适应的KL缩放和以正态分布为中心的前沿采样方案，以实现更高效的探索与正则化；

**💡 创新点**

创新点在于将固定KL系数与均匀采样这两个传统设计放在同一视角下重新审视，提出了兼顾探索与稳定性的自适应KL调度，并通过高斯前沿采样将梯度预算聚焦在难度适中的问题上，实现了对探索预算的全局协同分配；

**🔧 技术方法**

采用的技术包括：基于可验证奖励的RLVR、Group Relative Policy Optimization (GRPO)、无偏K3 KL估计器、以批次平均准确率为输入的非线性KL缩放函数、指数移动平均平滑的单题通过率、以及以p≈0.5为中心的高斯采样权重；

**📊 数据集**

实验数据集涵盖DAPO-17K训练集以及六个竞争级数学推理基准：AIME 2024、AIME 2025、MATH‑500、Minerva、OlympiadBench与AMC；

**📈 对比分析**

与GRPO基线在DeepSeek‑R1‑Distill‑Qwen‑1.5B与Qwen3‑8B‑Base两大模型上进行对比；FG-ExPO在pass@32上最高提升了13.34个百分点（AIME 2025，8B模型），在pass@1上提升约1–2个百分点；整体平均提升分别为+2.66（8B）和+2.16（1.5B）；

**⚠️ 局限性**

局限性包括：仅在数学推理任务且奖励为二元可验证信号上验证，未测试在部分奖励或多轮交互任务中的适用性；自适应KL和采样策略仅在批次级别实现，未细化到token或轨迹层面；实验仅覆盖两种模型体系，需进一步验证跨模型迁移性。

---

## 290. The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes

**arXiv ID:** 2605.11182 | [PDF](https://arxiv.org/pdf/2605.11182v1)

**作者:** Siqi Zhu `[一作]` (UIUC), Ge Liu `[通讯]` (UIUC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型的 On‑Policy Distillation（OPD）与 On‑Policy Self‑Distillation（OPSD）在推理、系统提示内部化和对齐任务中的效果进行了全面实验，系统分析了它们的成功与失败原因，并提出了多种稳定性改进方案。

**💡 创新点**

创新点在于识别并分类三种失效机制——教师-学生分布不匹配、Top‑K 逆 KL 梯度偏差以及 OPSD 仅学习 PI‑无关策略——以及通过停梯度 Top‑K 损失、RLVR 适配教师和 SFT 稳定学生等方法实现的性能提升。

**🔧 技术方法**

使用了 OPD/OPSD 损失设计、Top‑K 逆 KL 修正、强化学习与可验证奖励（RLVR）教师适配、监督微调（SFT）以及多种梯度处理技巧。

**📊 数据集**

主要实验数据集包括 OpenThoughts、Math500、AIME24/25、CharacterBench、EmotionBench、Wildguardmix、GPQA‑Diamond 等，覆盖数学推理、对齐和安全对齐任务。

**📈 对比分析**

与基线（RLPO、PPO、无 distillation）比较，OPSD 在系统提示内部化与对齐任务中显著提升准确率和效率；OPD 在数学推理任务中需要改进教师与学生分布匹配，否则会出现长度膨胀和重复；改进方案后，模型在多数任务上均优于原始方法。

**⚠️ 局限性**

局限性包括实验仅在少数模型规模（1.7B/4B/8B）与模型族上验证，可能无法推广到更大规模或其他模型；对数学推理的 OPSD 仍表现不佳，表明实例特定 PI 的处理仍需进一步研究。

---

## 291. ADMM-Q: An Improved Hessian-based Weight Quantizer for Post-Training Quantization of Large Language Models

**arXiv ID:** 2605.11222 | [PDF](https://arxiv.org/pdf/2605.11222v1)

**作者:** Ryan Lucas `[一作]` (MIT Operations Research Center), Rahul Mazumder `[通讯]` (MIT Sloan School of Management)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于ADMM的权重量化方法，用于大语言模型的后训练量化。

**💡 创新点**

创新点在于把层级量化建模为Hessian加权的约束优化，并通过ADMM联合优化权重，加入对角缩放、惩罚调度、网格刷新及局部搜索等技术。

**🔧 技术方法**

使用了ADMM、对角预处理、量化网格刷新、局部搜索、旋转/缩放变换以及Hessian近似校准。

**📊 数据集**

使用了WikiText-2、C4、Penn Treebank和LM Evaluation Harness等数据集进行评估。

**📈 对比分析**

与RTN、GPTQ、AWQ比较，实验显示在Qwen3系列模型的权重量化（W4/W3/W2）以及SpinQuant、SmoothQuant的权重-激活量化中，本文方法在困惑度和零样本准确率上均优于对比方法，推理速度与GPTQ相近。

**⚠️ 局限性**

局限性包括：相较轻量级方法计算成本更高、依赖校准数据集的代表性、仅在稠密解码器 LLM 上验证，未验证混合专家、扩散或视觉变压器模型。

---

## 292. Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning

**arXiv ID:** 2605.11458 | [PDF](https://arxiv.org/pdf/2605.11458v1)

**作者:** Zihao Han `[一作]` (ByteDance Douyin), Yilun Sun `[通讯]` (ByteDance Douyin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究教师在自蒸馏过程中的暴露级别对学习效果的影响，提出一种可学习的教师暴露控制方法，使教师只看到与学生当前能力匹配的那部分参考推理。

**💡 创新点**

创新点在于将教师暴露水平视为连续可学习的控制变量，使用Beta分布策略控制器，并通过延迟学习进展奖励（look‑ahead）和REINFORCE训练，以在不同训练阶段动态调节教师信息量。

**🔧 技术方法**

技术主要包括：On‑Policy Self‑Distillation (OPSD)、Beta 分布控制器、REINFORCE 与延迟奖励、持久窗口 + look‑ahead 的两阶段训练、LoRA 微调、以及 KL 目标的 token‑级监督。

**📊 数据集**

使用 OpenThoughts 作为自蒸馏训练数据；在 AIME 2024/2025、HMMT 2025 这三个数学竞赛数据集上进行评估。

**📈 对比分析**

与 Instruct 基础、SFT、GRPO、OPSD 等基线在相同 100 步预算下对比；ATEs 在 Qwen3‑1.7B、4B、8B 三个规模模型上分别提升了约 0.95、2.05、2.33 分，4B 最高得到 65.65 Average@12，8B 最高得到 67.13，整体显著优于所有基线。

**⚠️ 局限性**

局限性包括：仅在全局层面控制暴露，未实现样本级或难度级别的动态调整；延迟奖励窗口固定，可能影响对不同任务的适应；实验仅在数学竞赛数据上验证，未检验在代码生成或更广泛科学推理任务中的效果。

---

## 293. HamBR: Active Decision Boundary Restoration Based on Hamiltonian Dynamics for Learning with Noisy Labels

**arXiv ID:** 2605.11383 | [PDF](https://arxiv.org/pdf/2605.11383v1)

**作者:** Ningkang Peng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 652 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于Hamiltonian动力学的主动决策边界恢复方法HamBR，能够在噪声标签场景下主动生成虚拟外点并通过能量壁垒提升特征分离度。

**💡 创新点**

创新点在于将球面Hamiltonian Monte Carlo用于主动探测类间空隙，生成高质量虚拟外点构建能量障碍，从而突破传统被动样本筛选导致的边界坍塌瓶颈。

**🔧 技术方法**

主要技术包括球面HMC、能量潜在场构造、Riemannian Langevin动力学、对抗式几何正则化以及与半监督噪声学习框架的无缝集成。

**📊 数据集**

实验采用CIFAR-10/100（对称/非对称噪声）、Animal‑10N、Food‑101等公开噪声数据集。

**📈 对比分析**

与DivideMix、UNICON、LongReMix等基线相比，HamBR在各噪声率下平均提升约7–12个百分点，最高可达18%以上，且在OOD检测中AUROC和FPR95均显著优于对手。

**⚠️ 局限性**

局限性在于引入的球面HMC与能量计算增加了训练开销，且在极低噪声或高维任务中虚拟外点生成的有效性仍待进一步验证。

---

## 294. The tractability landscape of diffusion alignment: regularization, rewards, and computational primitives

**arXiv ID:** 2605.11361 | [PDF](https://arxiv.org/pdf/2605.11361v1)

**作者:** Ankur Moitra `[一作]` (Massachusetts Institute of Technology), Dhruv Rohatgi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5036102061)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文研究了在预训练扩散模型的推理时刻如何通过对齐分布来偏好高奖励样本，重点探讨了不同分布几何（KL 与 Wasserstein）下的可行性与原语。

**💡 创新点**

创新点在于：①从几何视角提出“原语驱动的奖励对齐”框架；②揭示 KL 对齐可用线性指数平移（linear tilt）原语实现，且对凸低维奖励可通过 log‑sum‑exp 包络高效采样；③证明 Wasserstein 对齐可通过 proximal transport oracle 实现，且对凹奖励或低维 Lipschitz 奖励可高效采样；④指出同一奖励类在不同几何下的可解性差异，形成“几何‑原语‑可解性”三维对照表。

**🔧 技术方法**

所用技术包括：线性指数平移采样原语、log‑sum‑exp 逼近、混合指数分布、近似重投射原语、凸优化的 proximal map、低维投影与 exhaustive search、复杂度分析与硬度证明。

**📊 数据集**

论文为理论研究，未使用实际数据集，主要给出算法的复杂度与可行性证明。

**📈 对比分析**

相比以往仅关注 KL 对齐的研究，本文展示了在 Wasserstein 约束下同类奖励可实现；在 KL 约束下对负二次奖励的采样被证明为 NP‑难。复杂度方面：KL 对齐在低维凸奖励下实现时间为 poly(d)·exp(k)，Wasserstein 对齐在低维 Lipschitz 或凹奖励下实现时间为 poly(d)·exp(r_A)。

**⚠️ 局限性**

限制：①KL 对齐仅对可用线性 tilt 原语的奖励有效；②对负二次（凹）奖励仍不可解；③Wasserstein 对齐需能够高效求解 proximal map，若目标域或成本不满足凸性则不适用；④未讨论 score oracle 的误差、离散模型或训练误差对对齐效果的影响。

---

## 295. ForceFlow: Learning to Feel and Act via Contact-Driven Flow Matching

**arXiv ID:** 2605.11048 | [PDF](https://arxiv.org/pdf/2605.11048v1)

**作者:** Shuoheng Zhang `[一作]` (Tianjin University), Jianye Hao `[通讯]` (Tianjin University)

**通讯引用:** 5631 | [OpenAlex ID](https://openalex.org/A5047509839)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于流匹配的力感知控制框架 ForceFlow，并通过 Vision‑to‑Force（V2F）分阶段交接实现了从视觉定位到力控交互的自然过渡。

**💡 创新点**

创新点在于（1）异构多模态融合：通过全局力向量条件与局部视觉序列条件的分离与交互，防止视觉特征掩盖低维力信号；（2）联合运动与力预测的流匹配策略，促使模型内化力与运动的耦合；（3）V2F 交接机制将空间泛化与物理泛化解耦，使系统在视觉不确定和力学变化下均能稳健工作。

**🔧 技术方法**

采用了流匹配（Flow Matching）作为生成器，AdaLN 对力向量进行全局调制，跨模态注意力融合视觉序列，Vision‑Language Model 进行目标指向点定位，并使用离散力历史窗口与主动力预测头提升力感知与调节。

**📊 数据集**

利用六个真实世界的接触任务数据集（Stamping、Plug/USB 插入、Press Button、Clean Whiteboard、Clean Vase、Cucumber Peeling）以及与 ForceVLA 相同的多视角 RGB + 关节姿态 + 6D 力/力矩观测；对 VLM 通过人工标注的 2D 关键点构建 VQA 样式训练集进行微调。

**📈 对比分析**

与 π_0.5、ACT、Diffusion Policy、ForceVLA 等基线对比，ForceFlow 在六项任务上的平均成功率提升至 81.67%（相比最佳基线 45%），同时平均力误差 MAE 降至 8.23 N（相比 20–30 N 的视觉主导基线）。在未见物理属性和空间 OOD 环境下亦保持高鲁棒性。

**⚠️ 局限性**

主要局限在于对高精度力/扭矩传感器的依赖，限制了在低成本机器人上的直接部署；此外 V2F 的切换阈值和策略目前是硬编码的，缺乏自适应动态调节。

---

## 296. Test-Time Personalization: A Diagnostic Framework and Probabilistic Fix for Scaling Failures

**arXiv ID:** 2605.10991 | [PDF](https://arxiv.org/pdf/2605.10991v1)

**作者:** Linhai Zhang `[一作]` (King's College London), Yulan He `[通讯]` (King's College London)

**通讯引用:** 13782 | [OpenAlex ID](https://openalex.org/A5015709853)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理时对个性化语言模型进行扩展采样（Test‑Time Personalization, TTP），通过生成 N 个候选回复并用个性化奖励模型挑选最优，从而提升生成质量。

**💡 创新点**

创新点在于：①推导出奖励模型 Best‑of‑N 的理论缩放规律；②发现并归纳两种失败模式——用户级崩塌与查询级奖励劫持；③提出概率奖励模型（带可学习方差）在训练时实现梯度缓冲与隐式正则化，成功消除两种失败模式，实现稳健的缩放。

**🔧 技术方法**

采用子高斯理论推导oracle缩放、相关度诊断指标、概率奖励模型（Gaussian NLL + 高分区间对比损失）、LoRA 参数微调、RAG 生成策略等技术。

**📊 数据集**

在 LaMP（新闻标题、学术标题）和 LongLaMP（摘要、产品评论、主题写作）共 5 个个性化文本生成任务上进行评估，使用用户历史数据作为训练集。

**📈 对比分析**

与随机挑选、全局奖励模型、确定性 User RM、训练基线（用户级 LoRA 微调）以及 oracle（基于 ROUGE 的真实奖励）对比。实验表明概率 User RM 在 N=30 时可接近 50‑80% 的 oracle 缺口，并在大多数任务上超过训练基线；理论缩放曲线与实验曲线误差 <3%。

**⚠️ 局限性**

局限性包括：仅验证文本生成任务；依赖足够的用户历史数据，冷启动情形未覆盖；每个用户训练单独奖励模型在大规模部署时成本高；对非文本领域或更复杂对话任务的泛化尚待探索。

---

## 297. Diabetic Retinopathy Classification using Downscaling Algorithms and Deep Learning

**arXiv ID:** 2605.11430 | [PDF](https://arxiv.org/pdf/2605.11430v1)

**作者:** Nishi Doshi `[一作]` (Dhirubhai Ambani Institute of Information and Communication Technology), Pankaj Kumar `[通讯]` (Dhirubhai Ambani Institute of Information and Communication Technology)

**通讯引用:** 2439 | [OpenAlex ID](https://openalex.org/A5100738690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过对糖尿病视网膜病变（DR）影像进行裁剪、下采样、填零后，将其统一为 600×600×3 的尺寸，并利用多通道 Inception V3 结构进行 5 类 DR 分级的深度学习分类。

**💡 创新点**

创新点在于：① 结合多种下采样算法（尤其是内容自适应的 LID）与 PSNR/SSIM 评估，系统地比较下采样效果；② 设计多通道 Inception V3 网络，将单张图像切分为四块并并行特征提取，以减少参数并充分利用图像信息；③ 将 Kaggle 与 IDRID 两大公开数据集融合，提升样本量与泛化能力。

**🔧 技术方法**

技术主要包括：图像预处理（裁剪、8 倍下采样、填零）、内容自适应下采样算法（LID、RDIP）、Lanczos 上采样、PSNR/SSIM 质量评估、基于 ImageNet 的 Inception V3 迁移学习、多通道卷积网络、交叉熵损失与梯度下降、以及混合训练/验证/测试划分。

**📊 数据集**

使用的公开数据集为：Kaggle DR 视网膜图像（35,216 张）与印度 DR 图像数据集（IDRID，516 张），共计 35,732 张图像，覆盖五个严重程度等级。

**📈 对比分析**

通过将 LID 与 Bilinear 两种下采样方式分别输入相同的多通道网络进行测试，得到的准确率、灵敏度、特异度分别为：LID‑下采样 85.2% / 83.4% / 87.6%，Bilinear‑下采样 83.15% / 81.2% / 84.6%。与已有方法相比，LID 方案在准确率和灵敏度上均实现了明显提升，超过 BT‑VGG 等传统 CNN 结果。

**⚠️ 局限性**

局限性包括：① 数据集仍存在类别不平衡，导致特异度受限；② 下采样后信息损失可能在更高分辨率下影响极细纹理特征；③ 仅在两种下采样方式中评估，其他潜在算法尚未探索；④ 训练时间和显存消耗较大，未在低资源环境中验证可行性。

---

## 298. Evaluating Structured Documentation as a Tool for Reflexivity in Dataset Development

**arXiv ID:** 2605.11345 | [PDF](https://arxiv.org/pdf/2605.11345v1)

**作者:** Eshta Bhardwaj `[一作]` (University of Toronto), Christoph Becker `[通讯]` (University of Toronto)

**通讯引用:** 1973 | [OpenAlex ID](https://openalex.org/A5101397764)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过混合方法分析，评估了结构化数据集文档（datasheets、data statements、dataset nutrition labels）在促进数据集开发反思性（reflexivity）方面的有效性。

**💡 创新点**

创新点在于：①将反思性文献的六大主题系统化为代码书；②将这些主题映射到现有文档框架，揭示其缺失；③结合语料库辅助话语分析（CADA），实证发现数据集开发者在填写文档时基本停留在透明度层面，缺乏深层反思；④提出可操作的改进建议，帮助文档框架更好地引导反思。

**🔧 技术方法**

技术手段包括：RTA（反思性主题提取）、CADA（定量词频、共现、主题建模）以及定性案例分析，配合Python工具（NLTK、gensim、Scikit-learn）。

**📊 数据集**

数据集：①30篇关于反思性的理论文献用于主题构建；②3个主流文档框架（datasheets、data statements、dataset nutrition labels）用于框架分析；③36份NeurIPS Datasets & Benchmarks赛道的完整datasheets，用于语料库分析。

**📈 对比分析**

方法对比：在框架分析中对比三种文档结构与六大反思主题的匹配度；在语料库分析中对比词频、主题距离和词汇多样性，发现文档主题高度同质化，说明缺乏深入反思。由于研究聚焦于质性评估，没有传统意义上的数值性能指标。

**⚠️ 局限性**

局限性：①仅采用批判性综述而非系统综述，代码书可能不涵盖所有反思概念；②样本局限于三种框架和NeurIPS数据集，未涵盖其他领域或机构的实践；③只分析文档文本，忽略了论文主体或补充材料中的隐含反思；④未提供机制实现机构层面的反思责任与评估。

---

## 299. $ξ$-DPO: Direct Preference Optimization via Ratio Reward Margin

**arXiv ID:** 2605.10981 | [PDF](https://arxiv.org/pdf/2605.10981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 300. Journalists, media and influencers: An analysis of the conversation in the digital public sphere during the Qatar 2022 World Cup

**arXiv ID:** 2605.11331 | [PDF](https://arxiv.org/pdf/2605.11331v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 301. A Mixture Autoregressive Image Generative Model on Quadtree Regions for Gaussian Noise Removal via Variational Bayes and Gradient Methods

**arXiv ID:** 2605.11585 | [PDF](https://arxiv.org/pdf/2605.11585v1)

**作者:** Shota Saito `[一作]` (Gunma University), Toshiyasu Matsushima `[通讯]` (Waseda University)

**通讯引用:** 449 | [OpenAlex ID](https://openalex.org/A5110471799)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种结合四叉树区域分割与混合自回归模型的概率图像生成框架，用于单幅图像的灰度去噪。

**💡 创新点**

创新点在于将MAP去噪转化为变分下界最大化，并提出可解析的梯度更新公式，使变分贝叶斯与梯度方法可交替迭代。

**🔧 技术方法**

采用变分贝叶斯推断、梯度上升优化、四叉树分割策略、混合自回归模型和变分下界计算等技术。

**📊 数据集**

实验使用标准的Set12灰度图像数据集，噪声水平设为σ∈{5,10,15}。

**📈 对比分析**

与高斯滤波、TV去噪和BM3D比较，低噪声下与TV相当，但在高噪声下性能下降，RMSE/PSNR/SSIM指标略逊于BM3D。

**⚠️ 局限性**

主要限制是高噪声时四叉树分割不稳定，导致去噪效果退化；对超参数和梯度步长敏感，需进一步优化。

---

## 302. Information and Contract Design for Repeated Interactions between Agents with Misaligned Incentives

**arXiv ID:** 2605.11294 | [PDF](https://arxiv.org/pdf/2605.11294v1)

**作者:** Nanda Kishore Sreenivas `[一作]` (University of Waterloo), Kate Larson `[通讯]` (University of Waterloo)

**通讯引用:** 2853 | [OpenAlex ID](https://openalex.org/A5105978897)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了信息丰富的发送者与信息不足的接收者在多次交互中的信号与合同设计

**💡 创新点**

创新点是将 Bayesian Persuasion 与线性合同结合到多智能体强化学习框架，探讨误配激励下的通信与定价策略

**🔧 技术方法**

使用多臂赌博机、离散化策略空间、PPO、tabular Q‑learning 等强化学习技术来学习信号和合同策略

**📊 数据集**

在推荐信情境和人工构造的10×10 gridworld 环境中进行实验，未使用公开数据集

**📈 对比分析**

实验表明 Sender 能学到最优信号与合同，Receiver 在接受信息后收益提升，合同进一步提高 Sender 收益但降低 Receiver 收益，结果与理论分析一致

**⚠️ 局限性**

局限包括对 Sender 完备观测的假设、离散化与简化环境、缺乏真实世界数据、未探讨多人社交困境等复杂场景

---

## 303. Birds of a Feather Flock Together: Background-Invariant Representations via Linear Structure in VLMs

**arXiv ID:** 2605.11107 | [PDF](https://arxiv.org/pdf/2605.11107v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 304. Selective Off-Policy Reference Tuning with Plan Guidance

**arXiv ID:** 2605.11505 | [PDF](https://arxiv.org/pdf/2605.11505v1)

**作者:** Duc Anh Le `[一作]` (Independent Author), Trung Le `[通讯]` (Monash University)

**通讯引用:** 1856 | [OpenAlex ID](https://openalex.org/A5102780660)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SORT（Selective Off‑Policy Reference Tuning with Plan Guidance），一种辅助更新，用来修复GRPO在所有采样结果错误时的零梯度失效；

**💡 创新点**

创新点在于利用参考答案提取的计划作为同模型的条件化上下文，计算每个参考token在有计划与无计划两种情境下的概率差异，基于几何平均权重实现结构信息的选择性强化，保持原有GRPO的rollout过程不变；

**🔧 技术方法**

技术包括GRPO、参考答案计划提取（PlanExtract）、同模型计划条件化推理、基于log‑ratio的token salience评估、Dynamic Fine‑Tuning（DFT）风格的几何平均权重、缓冲区机制与辅助训练交替；

**📊 数据集**

使用的数据集包括15k OpenR1‑Math‑220k（训练），以及AIME24/25、AMC23、MATH‑500、Minerva Math、OlympiadBench（在分布内测试）和GPQA‑diamond、MMLU‑Pro（离散布）做评估；

**📈 对比分析**

与SFT、GRPO、LUFFY、ReLIFT、Scaf‑GRPO等基线对比，SORT在弱模型上平均提升约+5.7点，在强模型上约+4点，整体在所有指标上均优于或与最强基线相当，特别在弱模型和零奖励样本上表现突出；

**⚠️ 局限性**

局限性包括对提取计划质量的依赖，若计划不准确会导致权重误差；仅解决零奖励组问题，对非零奖励错误缺乏直接干预；实现相对复杂，且在更大模型上的泛化仍需验证。

---

## 305. Decomposing Evolutionary Mixture-of-LoRA Architectures: The Routing Lever, the Lifecycle Penalty, and a Substrate-Conditional Boundary

**arXiv ID:** 2605.11153 | [PDF](https://arxiv.org/pdf/2605.11153v1)

**作者:** Ramchand Kumaresan `[一作]` `[通讯]` (Murai Labs), Ramchand Kumaresan (Murai Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在150M GPT模型上对进化Mixture‑of‑LoRA系统进行因子分解，评估路由器改写、评估范围和生命周期动态对对数PPL的影响。

**💡 创新点**

提出通过部分2^3因子分解和对数PPL归因链，揭示路由器改写是提升的唯一载体，并定义了ES有益的oracle‑对齐边界。

**🔧 技术方法**

使用LoRA低秩适配器、并行sigmoid路由器、可学习门限、温度退火、进化生命周期（选择、继承、变异、槽位重分配）以及合成基准任务进行实验。

**📊 数据集**

基于32k词表的混合领域文本语料（生物、代码、通用文本、科学），以及一个128词的合成bigram数据集。

**📈 对比分析**

与静态B3基线进行对比，采用三种种子、25k步训练并用几何平均平衡PPL做评估；路由改写带来+0.0426 nat显著提升，生命周期导致-0.028 nat损失；合成基准仅在oracle对齐时ES有益。

**⚠️ 局限性**

仅在单一150M模型、3种子、部分因子实验、评估确定性、生命周期组件未分解、合成基准规模有限等方面受限，缺乏对更大模型或不同数据集的推广性验证。

---

## 306. Multi-Agent System Identification with Nonlinear Sheaf Diffusion

**arXiv ID:** 2605.11204 | [PDF](https://arxiv.org/pdf/2605.11204v1)

**作者:** Nivar Anwer `[一作]` (Georgia Tech), Matthew Hale `[通讯]` (Georgia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在基于细胞层 Sheaf 的多智能体系统中，从轨迹数据恢复本地交互律的可识别性，并给出了必要与充分条件。

**💡 创新点**

首次将 Sheaf 同调（H¹(G;𝔽））作为恢复交互律的根本障碍，证明非参数恢复仅在 H¹=0 时可行，并提出参数化恢复的正定信息矩阵判据。

**🔧 技术方法**

使用了细胞 Sheaf 理论、非线性 Sheaf 拉普拉斯、同调与 Hodge 分解、信息矩阵分析以及最小二乘估计等技术。

**📊 数据集**

采用了三组仿真实验数据：环形 Sheaf 的形成迁移、带阈值的信任动态以及有限基底识别实验。

**📈 对比分析**

将恢复结果与仅基于节点轨迹回放的基准进行比较，发现回放误差不一定反映交互律恢复情况；实验表明当 H¹=0 或信息矩阵正定时能准确恢复，否则会出现误判。

**⚠️ 局限性**

限制在于假设节点势能 Ψ 已知且可预处理，忽略了噪声与部分观测；对异构 stalks 的处理仅为理论简化，且未给出联合恢复 Φ 与 Ψ 的方法。

---

## 307. More Than Meets the Eye: A Semantics-Aware Traffic Augmentation Framework for Generalizable Website Fingerprinting

**arXiv ID:** 2605.11402 | [PDF](https://arxiv.org/pdf/2605.11402v1)

**作者:** Youquan Xian `[一作]` (Beijing University of Posts and Telecommunications), Zhiyu Hao `[通讯]` (Zhongguancun Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于语义的交通增广框架，扩展 HTTP/2 资源组合与帧序列，并通过知识蒸馏实现跨层特征对齐，从而提升网站指纹识别的泛化性能。

**💡 创新点**

① 在协议约束下从应用层语义出发进行资源重组和帧序列增广；② 构造理想包长度序列并利用知识蒸馏将语义信息迁移到可观测的包长度特征；③ 通过三模块完整管线实现对跨域、跨时段的系统性特征偏移补偿。

**🔧 技术方法**

使用生成式增广（资源重组+帧序列增广）、HPACK 状态建模、最优分配求解（QP）、知识蒸馏（软标签+余弦对齐）以及常用深度学习 WF 模型（FSNet、BERT-PS 等）。

**📊 数据集**

基于 HTTP/2 的加密流量数据集 Singapore‑A、SouthKorea‑A、France‑A、Singapore‑B、China‑C，包含多国 Alexa Top‑N 网站访问的完整 TCP 包。

**📈 对比分析**

在闭域、跨域、开放域以及少样本环境下将框架嵌入多种主流 WF 模型进行对比实验，平均提升 ACC≈2.4%、F1≈4.5%；开放域中 ACC、AUROC 分别提升 90.8% 与 48.4%；少样本场景提升约 12%–25%。

**⚠️ 局限性**

仅聚焦 HTTP/2 且仅利用包长度侧信道，未涵盖 HTTP/3、WebSocket、时延等特征；增广过程需手工建模协议细节，扩展至其他协议的通用性待验证。

---

## 308. $\varepsilon$-Good Action Identification in Fixed-Budget Monte Carlo Tree Search

**arXiv ID:** 2605.11324 | [PDF](https://arxiv.org/pdf/2605.11324v1)

**作者:** Yinan Li `[一作]` (University of Arizona), Kwang-Sung Jun `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了深度为2的最大最小树（MCTS）的固定预算子树识别问题，提出一种ε-无关的Successive-Rejects风格算法并给出实例依赖的误判概率上界；

**💡 创新点**

首次给出固定预算最大最小动作识别的理论保证，并设计了树结构下的安全子树消除规则，实现了ε-无关的近似（ε-good）识别；

**🔧 技术方法**

基于自适应抽样与消除（Successive Rejects）、树结构特定的子树消除策略、子均值估计、集中不等式、以及信息论下界构造（KL、Le Cam）等技术；

**📊 数据集**

无具体数据集，全部为理论分析与模拟实验（实验部分在附录中展示）；

**📈 对比分析**

与传统最佳臂识别方法（如Successive Halving、Successive Rejects）对比，证明在ε-good任务上误判概率随预算按exp(−T/H₂(ε))衰减，匹配或接近已知最优下界；

**⚠️ 局限性**

上界与下界之间仍存在一定余差，难以进一步收窄；算法仅适用于深度2树，未推广到更深或更复杂的MCTS结构；

---

## 309. Dynamic Full-body Motion Agent with Object Interaction via Blending Pre-trained Modular Controllers

**arXiv ID:** 2605.11369 | [PDF](https://arxiv.org/pdf/2605.11369v1)

**作者:** Sanghyeok Nam `[一作]` (Korea Advanced Institute of Science and Technology), Tae-Kyun Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 11681 | [OpenAlex ID](https://openalex.org/A5100653784)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两阶段框架，将预训练的人体运动扩散模型与HOI先验混合生成动态、接触一致的人物-物体交互动作，并用composer网络融合全身与手部专家控制器，实现高效的动态HOI仿真。

**💡 创新点**

创新点在于通过交互一致的引导与在扩散采样中的inpainting保持手物接触一致，以及在执行阶段引入基于PCA子空间的composer，实现预训练专家的时空混合，显著提升动态HOI任务成功率并降低训练成本。

**🔧 技术方法**

使用的技术包括人类运动扩散模型（MDM）、FullBodyManip HOI先验、交互一致的inpainting与Kabsch对齐、PHC与InterMimic的预训练专家、轻量级composer及其PCA探索子空间，以及IsaacGym物理仿真。

**📊 数据集**

使用的数据集为AMASS用于训练MDM、FullBodyManip用于HOI先验、以及自行构造的动态HOI测试集（文本描述与物体交互），对比基线包括HOI-Diff、DAViD、MDM等。

**📈 对比分析**

实验对比表明，Ours_P+E在HOI质量、物理可行性与运动多样性上优于基线，动态HOI任务成功率提升至0.59，训练时间约为InterMimic_FT的三分之一。

**⚠️ 局限性**

局限性包括对手物接触一致性的依赖导致多样性下降、缺乏真实动态HOI数据作为评估基准，以及对更复杂物体质量与几何的物理规划尚未实现。

---

## 310. ACSAC: Adaptive Chunk Size Actor-Critic with Causal Transformer Q-Network

**arXiv ID:** 2605.11009 | [PDF](https://arxiv.org/pdf/2605.11009v1)

**作者:** Qian Chen `[一作]` (Tongji University), Guang Chen `[通讯]` (Tongji University)

**通讯引用:** 471638 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种能够自适应选择动作块大小的 Actor‑Critic 方法（ACSAC），通过在每个重规划时点动态决定执行多长的动作序列；

**💡 创新点**

核心创新在于使用因果 Transformer 作为批判器，学习跨时间长度的前缀 Q 值并保证跨尺度可比较；再通过流式行为克隆（flow BC）生成候选动作块，并在所有候选长度上联合 argmax 选取执行；

**🔧 技术方法**

采用因果 Transformer 作为 Q 评估网络，流式行为克隆（flow matching）用于策略生成，联合采样与多步 TD 损失实现训练；

**📊 数据集**

使用 OGBench 机器人操纵数据集（共 25 个长时程稀疏奖励任务）进行实验；

**📈 对比分析**

与单步、多步以及固定块大小的基线（如 SOTA 的离线 RL、Q‑chunking、AC3 等）进行对比，ACSAC 在离线和离线‑至‑在线阶段均获得显著更高的回报，尤其在长时程任务上优势更为明显；

**⚠️ 局限性**

局限性包括对最大块大小 H 的依赖，若 H 过小或过大均可能影响性能；在某些任务（如四物体操纵）中，固定块大小方法仍保持竞争力，表明自适应机制在部分情境下并非必然优越。

---

## 311. CTFusion: A CTF-based Benchmark for LLM Agent Evaluation

**arXiv ID:** 2605.11504 | [PDF](https://arxiv.org/pdf/2605.11504v1)

**作者:** Dongjun Lee `[一作]` (KAIST), Insu Yun `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了实时流式评测框架 CTFusion，用于在未发布的 CTF 竞赛中评估 LLM 代理，并揭示静态 benchmark 的数据污染与作弊问题。

**💡 创新点**

创新点在于引入共享团队账号但保持代理独立、仅提交第一次正确 flag、通过 Model Context Protocol (MCP) 代理实现对不同竞赛的无缝接入，以及通过实时竞赛对比展示静态 benchmark 的性能偏高。

**🔧 技术方法**

使用了 GPT‑4.1、Gemini 2.5‑Flash、Claude 3.5‑Sonnet 等 LLM；代理框架包括 Autogen、OpenAI Agent；MCP 服务器、web‑search RAG、pass@k 评估指标、CTFd API 与 CTFtime 数据。

**📊 数据集**

采用了五场 CTFd‑hosted 竞赛（CTF 2025 年度）和 NYU CTF Bench（约 210 题，最终 180 题）作为评测数据集。

**📈 对比分析**

通过 pass@3 指标对比实时 CTF 与静态 benchmark，实时竞赛的成功率约为 5–7%，而静态 benchmark 约为 11–17%，显著表明静态评测过高，真实能力更低。

**⚠️ 局限性**

限制包括难以区分任务难度与数据污染的影响；仅评估少数 LLM 与代理、仅 CTFd 竞赛、固定预算，缺乏置信区间，结果仅为指示性。

---

## 312. Context-Aware Spear Phishing: Generative AI-Enabled Attacks Against Individuals via Public Social Media Data

**arXiv ID:** 2605.11268 | [PDF](https://arxiv.org/pdf/2605.11268v1)

**作者:** Elham Pourabbas Vafa `[一作]` (University of Texas at Arlington), Shirin Nilizadeh `[通讯]` (University of Texas at Arlington)

**通讯引用:** 1005 | [OpenAlex ID](https://openalex.org/A5076332929)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究利用公开的社交媒体数据与生成式AI，自动化生成高度个性化的鱼叉式钓鱼邮件并评估其效果与现有防御；

**💡 创新点**

创新点在于提出了结合七类攻击策略与五维上下文的统一税onomies与评估框架，并展示了低成本大规模个性化钓鱼的可行性；

**🔧 技术方法**

技术手段包括多模态特征提取、Prompt工程、使用多大模型（GPT‑4、Claude、Gemini、Gemma、LLaMA）生成钓鱼内容，以及RoBERTa/DeBERTa等模型进行提示级检测；

**📊 数据集**

数据集来源为公开的Instagram帖子（200名用户共3268条）作为攻击源，对比APWG eCrimeX真实钓鱼邮件和Enron邮件作为基准；

**📈 对比分析**

通过八维质量指标（情境相关、说服力、情绪操控、个性化、自然语言、CTA特异性、发件人可信度、技术成熟度）评估18k AI生成邮件，结果显示LLM生成邮件在多项指标上均超过90%，而真实钓鱼邮件普遍低于30%；提示级检测模型在恶意提示上达98%准确率；

**⚠️ 局限性**

局限性包括仅依赖Instagram数据，未覆盖其他平台；防御模型为静态，需要持续更新；生成的邮件可能仍被现有过滤器拦截或需进一步改进。

---

## 313. Interpretability Can Be Actionable

**arXiv ID:** 2605.11161 | [PDF](https://arxiv.org/pdf/2605.11161v1)

**作者:** Hadas Orgad `[一作]` (Kempner Institute at Harvard University), Mor Geva `[通讯]` (Tel Aviv University)

**通讯引用:** 1619 | [OpenAlex ID](https://openalex.org/A5065717258)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出将可解释性评估标准转向可操作性（actionability），并给出一套框架、维度（具体性与验证）以及可操作性检查清单，旨在推动可解释性研究从仅解释模型转向支持具体决策与干预。

**💡 创新点**

创新点在于：①将可解释性研究的评价焦点从“解释质量”转向“能否驱动实际行动”；②构建了“可操作性”两维度模型（具体性、验证）来定位工作在可操作性空间中的位置；③列举了五大领域（缩放、对齐、手术干预、架构设计、概念翻译）及相应的行动类型，提供可操作性评估标准和指标。

**🔧 技术方法**

本文并未提出新的算法，而是综述并分析现有解释方法（如影响函数、机制定位、概念擦除、模型编辑、运行时干预等），并以此为基础构建评价框架和行动维度。

**📊 数据集**

无数据集。该文为立场性论文，主要以案例与文献综述为依据，并未进行实验验证。

**📈 对比分析**

未进行方法对比或性能评估。本文聚焦于提出评价框架与行动维度，未提供实验结果或与其他方法的直接性能比较。

**⚠️ 局限性**

局限性包括：①研究社区对可操作性奖励不足，导致方法验证缺乏动力；②现有解释方法多停留在理论或小规模任务，缺乏在大模型/真实场景中的通用性与验证；③部署难点（技术复杂、缺乏权重访问、专业人才不足）限制了解释方法的实际应用；④缺乏统一的可操作性评估标准，导致成果难以落地。

---

## 314. UNIPO: Unified Interactive Visual Explanation for RL Fine-Tuning Policy Optimization

**arXiv ID:** 2605.11549 | [PDF](https://arxiv.org/pdf/2605.11549v1)

**作者:** Aeree Cho `[一作]` (Georgia Tech), Chau `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一款基于 Web 的交互式可视化工具，统一展示并解释了 REINFORCE、PPO、GRPO、DAPO 与 Dr. GRPO 等五种 RL fine‑tuning 算法的训练过程与公式。

**💡 创新点**

创新点在于将算法的数学公式模块化，提供多层级视图，帮助非专家直观理解 token‑level 计算，并实现算法之间的可视化对比。

**🔧 技术方法**

使用 Svelte + D3 构建前端交互，采用统一的 JSON 训练日志 schema，并结合 LTTB 下采样、radial plot、token 颜色覆盖等可视化技术。

**📊 数据集**

以 Llama‑3.2‑1B‑Instruct 在 MATH 任务上分别使用上述五种算法训练的日志为数据（约 1,000 步），展示其训练动态。

**📈 对比分析**

通过并排展示两算法的目标函数并用颜色高亮差异，结合 Step Inspector 的 token‑级梯度信息来对比算法行为；主要侧重训练动态与可视化对比，未给出传统数值性能指标。

**⚠️ 局限性**

局限性在于仅支持五个预置算法，缺乏对更大规模训练日志的高效支持，且聚焦可视化，未提供自动化的算法效果评估。

---

## 315. Optimal LTLf Synthesis

**arXiv ID:** 2605.11544 | [PDF](https://arxiv.org/pdf/2605.11544v1)

**作者:** Yujian Cao `[一作]` (University of Liverpool), Shufang Zhu `[通讯]` (University of Liverpool)

**通讯引用:** 536 | [OpenAlex ID](https://openalex.org/A5101756232)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对多目标LTL_f合成中不可实现情形的最优合成框架，定义并实现了 max-guarantee、max-observation 以及 incremental max-observation 三种优化目标，给出对应的算法与符号实现。

**💡 创新点**

创新点在于：①将最优合成作为新问题引入，既考虑可预见的最大可实现子集，也考虑执行过程中可观察到的最优实现；②设计了增量最优策略的算法，使得在执行过程中能够不断提升保证值；③将权重化的目标值与最优合成结合，支持不同目标的重要性。

**🔧 技术方法**

主要技术包括：LTL_f 的 DFA 翻译、DFA 游戏求解（可达性游戏）、符号化（BDD）实现、符号固定点迭代、Boolean functional synthesis 生成策略、Spot 与 CUDD 库的调用。

**📊 数据集**

实验使用了 SyntCOMP 赛道中 1145 个不可实现的多目标合成实例（每个实例 2~50 个子目标），并提供相应的 Spot 生成的 DFA。

**📈 对比分析**

方法对比：在 300 秒超时限制下，max-guarantee 解 1060 个实例，basic max-observation 1005，extended max-observation 1006，improved max-observation 1008；max-observation 在 16 个实例上获得更高的保证值。性能上，max-guarantee 在某些实例更快；增量版本在部分实例提升或保持速度；总体上四种方法规模可控，表现相近。

**⚠️ 局限性**

局限性：当目标组合数大时，BDD 规模急剧膨胀导致符号操作代价高；增量方法在极端目标组合多样性时可能不如基准快；实验仅涵盖不可实现、完全可观测的场景，未覆盖部分可观测或实时构造等情况。

---

## 316. Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting

**arXiv ID:** 2605.11538 | [PDF](https://arxiv.org/pdf/2605.11538v1)

**作者:** Cheng Wang `[一作]` (National University of Singapore), Muhao Chen `[通讯]` (University of California, Davis)

**通讯引用:** 5029 | [OpenAlex ID](https://openalex.org/A5102861481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种协方差加权的 GRPO（CW‑GRPO），通过 Gaussian 核对极端 token 更新进行下调，以实现探索-利用平衡；

**💡 创新点**

创新点在于利用 token 归一化对数概率与优势的协方差动态调节梯度，而不依赖任何超参数；

**🔧 技术方法**

技术包括自然策略梯度、KL 限制、协方差计算、Gaussian 加权与归一化，再将加权优势融入 GRPO 损失；

**📊 数据集**

使用 Open‑RS 训练集（约 7k 高质量数学题）和多项数学评测集（AIME24、MATH‑500、AMC23、Minerva、OlympiadBench）进行评估；

**📈 对比分析**

与 vanilla GRPO、Clip‑Cov 等方法比较，CW‑GRPO 在 1.5B 与 7B 规模模型上均实现 2–6 分的平均提升，且训练过程中的熵保持更稳定；

**⚠️ 局限性**

局限在于仅验证至 7B 参数模型，且评测聚焦数学推理，需进一步测试更大规模模型及多样化任务。

---

## 317. Checkup2Action: A Multimodal Clinical Check-up Report Dataset for Patient-Oriented Action Card Generation

**arXiv ID:** 2605.11533 | [PDF](https://arxiv.org/pdf/2605.11533v1)

**作者:** Sike Xiang `[一作]`, Amir Atapour-Abarghouei `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文说明并演示了英国机器视觉会议（BMVC）论文的格式要求与排版规范。

**💡 创新点**

创新点在于提供了弹出式引用注释、边距标尺以及简化多作者机构输入的功能，以提升审稿和排版效率。

**🔧 技术方法**

采用LaTeX（pdflatex）模板实现，支持屏幕阅读与双页/单页打印。

**📊 数据集**

本文不使用任何数据集，主要是格式与排版指南。

**📈 对比分析**

不涉及方法比较或性能评估，本文仅作为格式示例。

**⚠️ 局限性**

局限性在于仅适用于BMVC格式，其他会议需要自行调整；对单页打印与多页打印的兼容性有限。

---

## 318. PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting

**arXiv ID:** 2605.11520 | [PDF](https://arxiv.org/pdf/2605.11520v1)

**作者:** Yixiao Song `[一作]` (Beijing Jiaotong University), Zhicheng Yan `[通讯]` (Beijing Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种无监督点云语义分割方法PointGS，利用3D高斯光散射重建稠密点云并通过SAM提取2D语义，再通过对比学习将语义迁移至3D高斯，并最终通过两步ICP对齐将标签传播到原始点云。

**💡 创新点**

创新点在于将3D高斯光散射作为统一中间表示，消除离散点与连续图像之间的域不匹配，并在此基础上实现跨视角一致的语义蒸馏与高斯-点对齐，无需额外3D预训练或人工标注。

**🔧 技术方法**

使用了3D Gaussian Splatting、Segment Anything Model、对比学习/尺度门控的特征对齐、两步ICP注册以及最近邻标签传播等技术。

**📊 数据集**

在ScanNet‑v2（20类）和S3DIS（13类）两个室内数据集上进行实验。

**📈 对比分析**

与多种无监督点云分割基线（如LogoSP、PointDC、GrowSP等）对比，PointGS在ScanNet‑v2上mIoU提升0.9个百分点，在S3DIS上提升2.8个百分点，显著优于现有方法。

**⚠️ 局限性**

局限性包括对视角投影数量和角度的敏感性、对高斯重建质量的依赖以及在极端稀疏或噪声点云场景下可能导致对齐误差。

---

## 319. Learning Weakly Communicating Average-Reward CMDPs: Strong Duality and Improved Regret

**arXiv ID:** 2605.11586 | [PDF](https://arxiv.org/pdf/2605.11586v1)

**作者:** Kihyun Yu `[一作]` (KAIST), Dabeen Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究弱通信平均奖励约束马尔科夫决策过程的强对偶性，并在此基础上提出一种前后对偶的剪切值迭代学习算法。

**💡 创新点**

在弱通信假设下首次证明强对偶性，利用占用测量集合的几何性质实现LP形式的对偶转换；随后结合价值函数剪裁与强对偶分析，取得平均奖励与约束违约的 O(T^{2/3}) 代价。

**🔧 技术方法**

主要技术包括占用测量集合的几何分析、线性规划强对偶性推导、有限期近似、剪切值迭代（LSCVI）、上置信界（UCB）估计、以及对偶变量的梯度上限与剪裁。

**📊 数据集**

该工作为理论分析，未使用任何实验数据集。

**📈 对比分析**

与之前针对弱通信线性 CMDP 的 O(T^{3/4}) 结果相比，本文方法在 regret 与约束违约上均达到 O(T^{2/3})，与最优的无约束 tabular 结果相匹配。

**⚠️ 局限性**

局限性在于仅适用于有限状态与动作空间；强对偶性尚未推广至无穷维情况；需要最优策略收益恒定且已知跨度的假设，且仍未实现 √T 级别的上界。

---

## 320. TCP-SSM: Efficient Vision State Space Models with Token-Conditioned Poles

**arXiv ID:** 2605.11563 | [PDF](https://arxiv.org/pdf/2605.11563v1)

**作者:** Sara Shoouri `[一作]` (University of Michigan), Hun-Seok Kim `[通讯]` (University of Michigan)

**通讯引用:** 4327 | [OpenAlex ID](https://openalex.org/A5014196508)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Token-Conditioned Poles State Space Model（TCP-SSM），通过对状态空间模型的极点进行可控、可解释的调节，实现输入依赖的自适应记忆与频率响应；

**💡 创新点**

创新点在于：①将极点分为实极点与共轭复极点，分别控制单调衰减与阻尼振荡；②通过令极点随token变化而调节，形成输入依赖的记忆时标；③采用分组共享极点与轻量低秩输入通道，保持线性时间扫描复杂度；④通过特征层级蒸馏指导新结构，确保性能；

**🔧 技术方法**

核心技术包括：状态空间模型（SSM）与控制理论极点设计、Token-Conditioned极点调节、低秩分解的输入分量、特征蒸馏训练、Vision Mamba/ EfficientVMamba 等视觉SSM骨干；

**📊 数据集**

在ImageNet‑1K进行分类；ADE20K进行语义分割；COCO 2017进行目标检测与实例分割；

**📈 对比分析**

对比方法主要是原版Vision Mamba与EfficientVMamba、Swin、ViT等注意力模型；实验表明在保持或略高准确率（如ImageNet Top‑1 76.0–82.6%，ADE20K mIoU 41.1% 等）的同时，TCP-SSM 将SSM FLOPs 降至 40–70% 以内，显著提升效率；

**⚠️ 局限性**

局限性包括：需要手动调节极点阶数、组数及低秩维度；极点参数化虽保证局部稳定，却未解决扫描顺序与内存访问模式等其他瓶颈；未来可探索二维极点结构、动态扫描与硬件友好实现。

---

## 321. Hindsight Hint Distillation: Scaffolded Reasoning for SWE Agents from CoT-free Answers

**arXiv ID:** 2605.11556 | [PDF](https://arxiv.org/pdf/2605.11556v1)

**作者:** Shengjie Wang `[一作]` (Tsinghua University), Yang Gao `[通讯]` (Tsinghua University)

**通讯引用:** 12780 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Hindsight Hint Distillation（HHD）框架，通过回顾失败轨迹生成简洁提示，引导 LLM 生成高质量长序推理轨迹。

**💡 创新点**

创新点在于利用后视提示合成与全局提示注入，保持 on‑policy 生成，避免离线/中途干预，显著提升长序推理效果。

**🔧 技术方法**

技术包括自我 roll‑out、失败轨迹聚类、提示生成、基于提示的 scaffolded reasoning 以及监督 fine‑tuning，结合 OpenHands 代理框架实现代码修复任务。

**📊 数据集**

使用 SWE‑Gym、SWE‑bench Verified 与 Multilingual 数据集，均为真实 GitHub 问题与多语言任务。

**📈 对比分析**

与 Naive RFT、SE‑agent‑Reflect、Agent‑RLVR、Dense Expert Judge 等基线比较，Qwen‑72B 在 SWE‑bench Verified 上 Pass@1 51.2%、Pass@5 70.2%，比基线提升约 8% 以上；在 Multilingual 任务上亦取得最优性能。

**⚠️ 局限性**

局限性包括对提示生成质量的依赖、对极长序列推理的效率挑战，以及在动态交互环境中的适配性仍待提升。

---

## 322. ScribbleDose: Scribble-Guided Dose Prediction in Radiotherapy

**arXiv ID:** 2605.11555 | [PDF](https://arxiv.org/pdf/2605.11555v1)

**作者:** Zhenxi Zhang `[一作]` (Hong Kong Polytechnic University), Ge Ren `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 911 | [OpenAlex ID](https://openalex.org/A5064187352)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种基于稀疏标记（scribble）的放射治疗剂量预测框架 ScribbleDose，利用 Scribble Completion Module (SCM) 将稀疏笔划扩展为密集解剖结构掩模，再通过 Structure-Guided Dose Generation Module (SGDGM) 将掩模与剂量分布进行连续学习，从而实现无需完整结构标注的剂量预测。

**💡 创新点**

创新点包括：1) 将稀疏笔划作为结构引导，显著降低标注成本；2) 在 SCM 中结合语义聚类与基于超体素的距离敏感边界约束，实现结构一致且几何可解释的掩模重建；3) 在 SGDGM 中引入基于掩模的注意力调制和剂量排名正则化，强化结构-剂量耦合与空间连续性；4) 将结构重建与剂量预测联合优化，形成端到端的连续学习管道。

**🔧 技术方法**

核心技术包括：3D 卷积编码器-解码器、类中心语义聚类、超体素分割与距离变换边界约束、Retentive Transformer 结构指导注意力、损失函数中的 compactness、separation、boundary、rank 正则化。

**📊 数据集**

使用了 GDP‑HMM 2025 公开数据集，共 3231 个放疗案例，训练集 2627，验证集 148，测试集 356；并在该数据集上生成了与原始完整掩模一致的笔划注释。

**📈 对比分析**

与传统完整掩模（Mask）方法、两阶段框架（Box/Point + 预训练分割模型）以及纯 CT 预测模型相比，ScribbleDose 在 Dose Score、DVH Score、HI、CI、D95 等指标上实现了与 Mask 基线相近甚至更优的性能；在所有稀疏提示方法中获得最低的剂量误差。

**⚠️ 局限性**

局限性包括：1) 目前的笔划注释是从完整掩模人工生成的模拟笔划，未评估真实临床笔划的误差、位置漂移和缺失切片问题；2) 仍依赖超体素分割对边界的准确性，复杂解剖结构下可能不够鲁棒；3) 仅使用了剂量分布误差指标，缺乏更具临床解释力的剂量目标约束和安全性评估。

---

## 323. Strong Inapproximability for a Promise Rank Problem

**arXiv ID:** 2605.11545 | [PDF](https://arxiv.org/pdf/2605.11545v1)

**作者:** Venkatesan Guruswami `[一作]` (University of California), Shaoxuan Tang `[通讯]` (Tsinghua University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明在有限域 𝔽₂^r 上，给定一个线性矩阵子空间，若已知其中存在秩为 1 的矩阵，则寻找最小秩矩阵在多项式时间内不可逼近至 n^o(1/ loglog n) 的秩，假设 NP 没有亚指数算法。

**💡 创新点**

创新点在于将 Khot–Saket 的超位置完备性与伪矩阵（moment matrix）技术相结合，构造了两种不同的归约：一种在 𝔽₂^r 上实现 n^O(log k) 的秩间隙，另一种在任意有限域 𝔽_q 上实现 n^O(k) 的归约，从而推广了最低秩距离和最短向量问题的 PCP‑free 不可逼近结果。

**🔧 技术方法**

核心技术包括：超位置完备性（superposition soundness）框架、伪矩阵与等并集合约束、矩阵分解为对称秩一项的引理、以及利用等并集合约束与低秩矩阵相结合得到的秩下界。

**📊 数据集**

无数据集；研究完全基于理论计算复杂性与矩阵代数。

**📈 对比分析**

通过证明若存在低秩矩阵会导致超位置满足和矩阵分解相矛盾，从而在多项式时间内证明不可逼近；在理论上给出了在不同场和参数下的严格不可逼近阈值。

**⚠️ 局限性**

局限性包括：结果依赖于 NP 不存在亚指数算法的假设；对 𝔽₂^r 的归约需要额外的秩下降步骤；目前对一般有限域 𝔽_q 的 n^O(log k) 归约尚未完成，且仅针对特定类型的矩阵子空间（即满足等并集合约束的伪矩阵）有效。

---

## 324. GeoR-Bench: Evaluating Geoscience Visual Reasoning

**arXiv ID:** 2605.11541 | [PDF](https://arxiv.org/pdf/2605.11541v1)

**作者:** Yushuo Zheng `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22203 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了GeoR-Bench基准，通过推理驱动的图像编辑任务评估多模态模型在地球科学视觉推理中的能力，测试模型是否能生成科学上有效的视觉结果。

**💡 创新点**

采用推理驱动图像编辑评估框架，将科学推理与视觉生成直接绑定；提出推理、一致性、质量三维度评价以及严格准确度指标；构建跨六大地球科学类别的440样本基准。

**🔧 技术方法**

利用多模态大型语言模型和图像编辑模型进行推理与生成；评估使用Gemini 3 Flash自动判定器结合Q‑Align图像质量评估；实验部署在NVIDIA H200 GPU上。

**📊 数据集**

GeoR-Bench自制数据集，440个样本，涵盖6个类别（地貌学、GIS与空间几何、构造科学、大气与海洋动力学、冰雪科学、水文学）和24个任务类型，来源包括卫星遥感图像、地图、三维剖面和科学图表。

**📈 对比分析**

通过严格准确度（推理+一致性+质量全部通过）对21个闭源与开源模型进行排行榜；闭源最佳GPT‑Image‑2 42.7%严格准确率，开源最佳Qwen‑Edit‑2511仅10.3%；推理分数是主要瓶颈，质量和一致性相对饱和。

**⚠️ 局限性**

仍无法达到50%严格准确率，说明多模态模型在地球科学推理方面能力不足；开放源模型表现更差，缺乏对科学过程的深度理解；数据集规模有限，可能无法覆盖所有复杂地球科学场景；自动评判与专家评判之间仍存在误差。

---

## 325. ToF ReSTIR: Time-of-Flight Rendering with Spatio-temporal Reservoir Resampling

**arXiv ID:** 2605.11536 | [PDF](https://arxiv.org/pdf/2605.11536v1)

**作者:** Juhyeon Kim `[一作]` (Dartmouth), Adithya Pediredla `[通讯]` (Dartmouth)

**通讯引用:** 498 | [OpenAlex ID](https://openalex.org/A5016228129)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于 ReSTIR 的时域光路重用框架（Path‑Length‑Aware Shift Mapping），实现实时时间飞行（ToF）渲染，可同时处理时间门控图像、瞬态直方图以及多频率（多普勒）渲染，并在此基础上演示了非视线重建与导航等下游应用。

**💡 创新点**

创新点包括：
• 通过在 ReSTIR 采样与重用过程中引入光程约束，构造了新的路径长度感知平移映射；
• 采用 Newton 迭代求解路径长度等值曲线并引入“平均梯度”规范化（gauge）以保证可逆性与无偏性；
• 为初始采样设计了“路径长度收缩”方法，既可避免昂贵的椭圆–几何交点计算，又能在宽门到窄门之间迁移样本；
• 将该框架扩展到瞬态直方图和多普勒频率域，形成统一的重用策略。

**🔧 技术方法**

技术手段：ReSTIR 结构（reservoir‑based spatio‑temporal resampling）、光程感知平移映射、Newton 迭代求解、平均梯度规范化、椭圆路径连接、MIS、GPU 实现（Falcor）、时间门控、瞬态直方图、Doppler 频率渲染、以及对非视线重建与导航的后处理算法。

**📊 数据集**

实验数据集主要为常用渲染基准场景：Cornell‑Box、Cornell‑Dragon、Veach‑Ajar、Kitchen、Bedroom、Staircase、Bistro、Classroom、NLOS‑Bunny 等；以及自定义的 NLOS 隐藏物体与导航场景。

**📈 对比分析**

与传统路径追踪及无约束的 ReSTIR 重用比较，实验表明：
• 在时间门宽度较窄（<5–10% 场景尺寸）时，误差（MAPE）显著下降，视觉噪声大幅降低；
• 在瞬态直方图渲染中，重用对 B 较小（≤32）时提升明显，B 过大则成本占优；
• 在非视线重建中，利用本框架的图像噪声降低可使体素重建误差减半；
• 在导航实验中，基于本方法的路径规划避免了碰撞，成功率提升约 30%；
• 所有方法均在 RTX 3090 GPU 上实现，交互帧率可达 30+ FPS（视图与门宽限制）。

**⚠️ 局限性**

局限性：
• 只在重用点附近进行光程修正，未考虑全路径或多次反射的联合约束；
• 需要单一重用顶点，可能导致可见性变化导致重用失败；
• Newton 迭代在极端几何或材料下可能不收敛，导致成功率下降；
• 目前仅支持 delta 光源，有限面积光源需要额外处理；
• “收缩映射”虽高效但在极窄门宽下可能引入偏差；
• 瞬态直方图中 B 较大时重用成本迅速膨胀，需进一步优化多频段重用策略。

---

## 326. Primal-Dual Policy Optimization for Linear CMDPs with Adversarial Losses

**arXiv ID:** 2605.11535 | [PDF](https://arxiv.org/pdf/2605.11535v1)

**作者:** Kihyun Yu `[一作]` (KAIST), Dabeen Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对线性约束马尔可夫决策过程的首个在线主-对偶策略优化算法，处理损失为完全信息下的对抗性、成本为带反馈下的随机性；

**💡 创新点**

创新点包括：① 设计新的加权LogSumExp softmax策略并给出其覆盖数上界；② 引入周期性策略混合与正则化双变量更新，实现覆盖数与双变量同时受控；③ 结合线性函数逼近在大状态空间下实现子线性回报与约束违规上界；

**🔧 技术方法**

技术主要有：主-对偶镜像下降（KL）优化、特征收缩（feature contraction）、对数覆盖数分析、漂移分析（drift analysis）、正则化双变量更新；

**📊 数据集**

实验采用改造后的作业调度CMDP（job‑scheduling CMDP）作为基准，模拟对抗性损失；

**📈 对比分析**

与现有对抗性或随机性CMDP算法对比，实验显示回报与约束违规均随试验周期增长保持子线性，约束违规最终趋于零；

**⚠️ 局限性**

局限性包括：回报与违规上界为K^{3/4}，未达到√K；仅在全信息损失与带反馈成本下验证；对样本效率和更一般的对抗性成本设置尚未解决。

---

## 327. Read, Grep, and Synthesize: Diagnosing Cross-Domain Seed Exposure for LLM Research Ideation

**arXiv ID:** 2605.11532 | [PDF](https://arxiv.org/pdf/2605.11532v1)

**作者:** Yunju Choi `[一作]` (Yonsei University), Min Song `[通讯]` (OnomaAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 PaperGym 三阶段流水线：工具增强的种子提取、跨域释义检索以及基于检索种子的多种子方法合成，并对其进行评估。

**💡 创新点**

创新点在于系统化评估跨域检索对 LLM 研究想法生成的帮助，并发现多样化种子暴露有效但缺乏对检索语义的充分利用；同时提出带有来源归属的合成方式。

**🔧 技术方法**

采用 LLM 代理结合 read、grep、bash 工具进行交互式提取；通过对问题进行七域释义并嵌入检索；使用 GPT‑5 生成方法、Claude Sonnet 4.6 评判；辅以 Rubric‑based 评分和 pairwise 新颖度/连贯性评估。

**📊 数据集**

使用了 30 条跨 7 个机器学习领域（LLM/NLP、多模态、计算机视觉、强化学习、信息检索/推荐、语音、机器人）的研究问题；种子库包含 1,167 条从 446 篇 2017–2025 年会议论文中提取的种子。

**📈 对比分析**

通过四个 ablation（无检索、同域检索、跨域检索、随机种子控制）进行比较；跨域检索在新颖度上优于无检索和同域检索，但与随机控制无显著差异；效度和连贯性整体接近上限，跨域检索在种子归属整合方面略优。

**⚠️ 局限性**

局限性包括仅使用 30 个问题、单一模型评判、缺乏多轮随机抽样和人工专家评估；未来需扩大基准、引入人类评价、强化学习训练及自我扩展种子库。

---

## 328. FERMI: Exploiting Relations for Membership Inference Against Tabular Diffusion Models

**arXiv ID:** 2605.11527 | [PDF](https://arxiv.org/pdf/2605.11527v1)

**作者:** Abtin Mahyar `[一作]` (University of Waterloo), Xi He `[通讯]` (University of Waterloo)

**通讯引用:** 4986 | [OpenAlex ID](https://openalex.org/A5038736889)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FERMI 框架，在多表关系数据库中进行会员推断攻击，解决训练时可访问关系信息但推断时只能看到单表记录的非对称场景。

**💡 创新点**

创新点在于将单表特征映射到多表特征空间，利用关系上下文强化会员信号，并在白盒和黑盒两种攻击模型下实现可行的特征映射与分类器训练。

**🔧 技术方法**

采用扩散模型（TabDDPM、TabSyn、TabDiff）以及 shadow‑model 方案，提取基于 denoising loss 的 fingerprint 特征；使用 CORAL 域适配器实现单表→多表特征映射，并联合训练映射器与分类器。

**📊 数据集**

在真实的多表数据集上评估：California、Instacart 与 Berka 三个公开数据集。

**📈 对比分析**

通过与单表基线、合并表直接攻击的 AUC、TPR@0.1/0.01 等指标比较；在白盒环境下 FERMI 最高提升 53% 的 TPR@0.1，黑盒环境提升 22%，显著优于单表基线并接近合并表上限。

**⚠️ 局限性**

局限性：仅在合并表能产生显著更强会员信号时有效；对采用潜在空间编码的 TabSyn 等模型几乎无提升；需要在训练阶段拥有完整关系信息，无法处理训练时缺失关联上下文的情况。

---

## 329. OverNaN: NaN-Aware Oversampling for Imbalanced Learning with Meaningful Missingness

**arXiv ID:** 2605.11525 | [PDF](https://arxiv.org/pdf/2605.11525v1)

**作者:** Amanda S Barnard `[一作]` (Australian National University), Amanda S Barnard `[通讯]` (Australian National University)

**通讯引用:** 12472 | [OpenAlex ID](https://openalex.org/A5039459434)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个名为 OverNaN 的 NaN‑aware 过采样框架，支持 SMOTE、ADASYN、ROSE 等方法在存在缺失值的情况下直接生成合成样本，而不需要先填补缺失值。

**💡 创新点**

将缺失结构视为特征信息，在合成样本时保留、传播或按策略插值 NaN，提供三种缺失处理策略；在公开数据集与合成实验中证明其在不失去信息的前提下提升不平衡学习效果。

**🔧 技术方法**

采用共享观测子空间的距离计算、按策略的 NaN 传播、并行化合成样本生成，并用 NumPy/Pandas 实现轻量化 Python 库。

**📊 数据集**

实验数据包括 OpenML 上的 labor、cylinder-bands、Titanic、soybean、car‑evaluation 等公开数据集，Graphene oxide nano‑flakes 数据集，以及 OpenML anneal 数据集。

**📈 对比分析**

与传统填补‑后过采样、仅删除特征/样本、标准 SMOTE/ADASYN/ROSE 进行对比；结果显示 OverNaN 在多数指标上保持或略优于基线，在 Graphene 例子中获得最高准确率与 F1，且方差更小；在 OpenML 基准中 ROSENaN 的平均 AUC 最高，提升约 1–1.3 个百分点。

**⚠️ 局限性**

仅适用于小至中等规模数据，缺乏 GPU 或分布式支持；不建模缺失机制，不能替代必要的插补；高缺失率下生成样本仍可能产生不确定性，对极端不平衡需慎重使用。

---

## 330. State Twins: An Off-Chain Substrate for Agentic Reasoning over Decentralized Finance Protocols

**arXiv ID:** 2605.11522 | [PDF](https://arxiv.org/pdf/2605.11522v1)

**作者:** Ian C. Moore `[一作]` `[通讯]` (DeFiMind), Ian C. Moore (DeFiMind)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了离线类型化的 State Twin 副本，用于在 DeFi AMM 池上进行高精度的离线、可分叉的推演与分析。

**💡 创新点**

创新点在于将 AMM 视作离散时间受控动力学系统，给出精确的误差上界；提出 Provider/Builder 两层解耦的模式，支持多源多协议统一接口；构建了 DeFiPy v2 开源工具箱，将 State Twin 与 LLM 调用层（MCP）整合；演示“fork‑and‑evaluate”工作流，实现单次链读后多场景评估，突破传统反应式架构的结构瓶颈。

**🔧 技术方法**

核心技术包括：离散时间状态空间建模、固定点算术误差分析、Python 类型化与元数据结构、Web3 RPC 与多链多协议读取、MCP（Model Context Protocol）工具暴露、深度复制与克隆实现独立分支、以及基于数学不变式的性能证明。

**📊 数据集**

使用的数据主要是链上实时快照（Uniswap V2/V3、Balancer、Stableswap）、自定义 CSV 以及合成配方，构成不同协议与协议版本的多源状态数据集；未使用传统机器学习训练集。

**📈 对比分析**

方法对比：相较于传统需多次链查询或重放的 reactive 架构，State Twin 只需一次 RPC，随后所有分叉评估均在内存中完成；实验表明，在主网 V3 USDC/WETH 5bps 池上，50 个价格冲击场景的 fork‑and‑evaluate 在一次链读后即可在子秒级完成，且误差上界可根据公式量化，实际误差远低于 10⁻¹⁰。该工作没有提供数值性能对比基准，但从理论上证明可在 O(1) 链交互下完成 O(N) 场景评估。

**⚠️ 局限性**

局限性包括：目前仅实现 V2/V3，Balancer 与 Stableswap 需在 v2.2 以后；单池模式，尚无多池路由副本；误差上界仅针对固定点常乘积规则，Balancer 与 Stableswap 的不等式仍待证明；缺乏信任门控（LLM 与状态变更的安全校验）以及完整的多线程/并行执行细节；对大规模并发 fork 场景的内存与 GC 性能未做深入评估。

---

## 331. Controllable User Simulation

**arXiv ID:** 2605.11519 | [PDF](https://arxiv.org/pdf/2605.11519v1)

**作者:** Guy Tennenholtz `[一作]` (Google Research), Craig Boutilier `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出基于因果推断的可控用户模拟框架，揭示传统轨迹标签式监督微调产生的前瞻偏差与可控性崩溃问题，并提出三种因果一致性修正方案（先验控制、步进动态控制、策略感知学习），验证其在多轮对话数据上能消除偏差、保持自然方差并实现零样本泛化。

**💡 创新点**

首次将离线评估与因果推断统一，正式证明轨迹标签导致的“look‑ahead bias”和“controllability collapse”，并提出因果一致性训练方法，解决了可控模拟的结构性失效。

**🔧 技术方法**

使用因果图分析、离线策略评估（OPE）理论、监督微调（SFT）与LLM标注器、以及基于贝叶斯更新的步进动态控制技术。

**📊 数据集**

在WildChat和ConvApparel‑V2两大多轮对话数据集上进行实验。

**📈 对比分析**

与无条件SFT、提示式模拟、拒绝采样、传统轨迹‑SFT以及认知档案‑SFT等基线比较，因果一致性方法在控制遵循度、对话长度、语义漂移、行为多样性及方差稳定性上均显著优于其他方法，且实现了接近真实人类分布的零样本评估。

**⚠️ 局限性**

对策略感知学习依赖白盒代理信息，步进动态控制在需满足全局约束时实现难度大，且在极大时间步长下仍可能出现方差扩散；未来需要适配黑盒API和自动发现最优动态控制轨迹。

---

## 332. Agents Should Replace Narrow Predictive AI as the Orchestrator in 6G AI-RAN

**arXiv ID:** 2605.11516 | [PDF](https://arxiv.org/pdf/2605.11516v1)

**作者:** Pranshav Gajjar `[一作]` (North Carolina State University), Vijay K Shah `[通讯]` (North Carolina State University)

**通讯引用:** 1044 | [OpenAlex ID](https://openalex.org/A5083496212)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出将多模态大语言模型（Domain‑adapted LLM/LTM）作为 RAN Intelligent Controller（RIC）的核心认知操作系统，以取代传统分散的预测模型，实现意图驱动的网络配置与自动故障诊断。

**💡 创新点**

创新点包括：
- 将 LLM 作为多域“意图翻译器”，通过 Retrieval‑Augmented Generation（RAG）和 TeleMCP 协议实现跨供应商日志与技术规范的语义理解；
- 通过 Chain‑of‑Thought/ ReAct 推理框架，让 LLM 自动生成可执行的配置脚本并通过神经‑符号验证保障确定性；
- 提出了针对通信场景的极低位量化、结构化剪枝及新型线性注意力模型，为边缘 RIC 的快速推理奠定基础；
- 明确了面向网络的持续对齐（RLNF）、语义通信与多代理协同等开放挑战。

**🔧 技术方法**

技术手段包括：
- 大语言模型（LLM/LTM）与小型量化语言模型（SLM）；
- Retrieval‑Augmented Generation（RAG）与 TeleMCP（多域上下文汇聚协议）；
- Chain‑of‑Thought / ReAct 推理模板；
- 低位量化（sub‑8‑bit）、结构化剪枝、状态空间模型（SSM）/线性注意力等新架构；
- 神经‑符号验证框架、数字孪生仿真、联邦学习与 LoRA/DPO 微调。

**📊 数据集**

使用的数据集与测试集包括：
- AI5GTest 自动化验证框架（基于 3GPP 标准的日志与测试用例）；
- 真实网络遥测日志、Syslog 以及多供应商互操作性故障记录；
- 3GPP 技术规范文本与厂商硬件手册（作为检索库）。

**📈 对比分析**

比较方法：在 AI5GTest 环境下，分别评估三种上下文注入策略——原始日志直接注入、标准 MCP 方案、TeleMCP 方案；
- 结果显示：传统直接注入最少 token 消耗但准确率 0%；MCP 方案准确率高但 token 超过 1.6M；TeleMCP 方案在保留 100% 准确率的同时，token 消耗减少约 81%。
- 另外，演示了层级化推理模型（Non‑RT RIC 重模型 + Near‑RT SLM）在推理延迟与计算成本上的可行性。

**⚠️ 局限性**

局限性：
- 推理延迟与计算开销仍是边缘部署的瓶颈；
- LLM 的概率性生成易产生 hallucination，需神经‑符号验证与数字孪生校验；
- 数据隐私与安全风险（日志与配置的敏感性）需本地化模型与联邦学习；
- 对抗性注入攻击与 Jailbreaking 的防御尚未成熟；
- 需要针对网络 KPI 的持续对齐（RLNF）与复杂多域奖励分配的研究。

---

## 333. Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference

**arXiv ID:** 2605.11581 | [PDF](https://arxiv.org/pdf/2605.11581v1)

**作者:** Wenxin Dong `[一作]` (Baidu Inc.), Lin Liu `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现 Ada-MK，针对 NVIDIA Ada GPU 的 LLM 推理，在 TensorRT-LLM 插件中融合 MegaKernel，完成多阶段共享内存管理、离线 DAG 搜索和混合推理引擎，实现小批量、短序列的低延迟推理加速。

**💡 创新点**

三维共享内存约束模型+K 维细分压缩 50% 共享内存，基于 MLIR 的细粒度 DAG 离线搜索消除运行时分支，和将 MegaKernel 作为插件嵌入 TensorRT-LLM 的异构混合引擎，实现首个在商业在线广告系统的 MegaKernel 部署。

**🔧 技术方法**

MLIR 细粒度分解、PTX DAG 依赖建模、K 维共享内存细分、Warp Specialization、Tensor Core 量化权重重排、异步 TMA/IO、离线性能调优、TensorRT-LLM 插件化。

**📊 数据集**

Qwen3-1.7B 与 Qwen2.5-1.5B（GPTQ-W4A16 量化）在固定短序列（in64/out12）以及 CSL、Human-eval 真实任务数据集上进行评测。

**📈 对比分析**

在 NVIDIA L20 GPU 上对比 vLLM、SGLang、原生 TensorRT-LLM，采用离线批处理测量 tokens/s。Ada-MK 在小批量/短序列下提升 23.6% 以上，最高 50.2%/64.5% 超过 vLLM/SGLang；在 CSL/ Human-eval 中维持 4–19.5% 的提升，最高 19.5%；在大批量长序列场景下优势收窄但仍保持领先。

**⚠️ 局限性**

受 Ada 共享内存限制仅能支持 2 阶段流水线，K 维细分仍需手动参数化；离线搜索成本高，依赖特定模型/硬件配置；在大批量/长序列场景下仍落后于 vLLM/SGLang 的系统级调度优势；未来需适配更大模型与 Blackwell 架构。

---

## 334. TwiSTAR:Think Fast, Think Slow, Then Act,Generative Recommendation with Adaptive Reasoning

**arXiv ID:** 2605.11553 | [PDF](https://arxiv.org/pdf/2605.11553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 335. FedOUI: OUI-Guided Client Weighting for Federated Aggregation

**arXiv ID:** 2605.11571 | [PDF](https://arxiv.org/pdf/2605.11571v1)

**作者:** Alberto Fernández-Hernández `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 6950 | [OpenAlex ID](https://openalex.org/A5012806004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FedOUI，一种基于激活结构的客户端权重分配方法，利用OUI指标对客户端的激活模式进行评价并在服务器端进行软加权聚合。

**💡 创新点**

创新点在于引入无标签、激活层平衡度指标OUI作为结构典型性信号，结合Beta分布对轮次内OUI分布建模，进而实现对极端客户端的轻量化降权。

**🔧 技术方法**

使用了FedAvg、FedProx、FedAlign等聚合规则作基线，对每个客户端计算OUI并拟合Beta分布，利用结构分数与样本量组合得到权重；整体算法实现轻量化且兼容同步FL。

**📊 数据集**

在CIFAR-10数据集上使用小型CNN模型进行实验。

**📈 对比分析**

与FedAvg、FedProx、FedAlign比较，FedOUI在强非IID Dirichlet分割下实现最高的最终准确率、峰值准确率和AUC；在噪声客户端设置下，峰值准确率最高，最终准确率略低于基线。

**⚠️ 局限性**

局限性包括：仅在小规模CNN与CIFAR-10上验证；对更大模型或更复杂异构场景的泛化未知；只使用单层OUI，未探索多层或时间平滑；在极端噪声下最终准确率仍落后于基线。

---

## 336. OUI as a Structural Observable: Towards an Activation-Centric View of Neural Network Training

**arXiv ID:** 2605.11570 | [PDF](https://arxiv.org/pdf/2605.11570v1)

**作者:** Alberto Fernández-Hernández `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 6950 | [OpenAlex ID](https://openalex.org/A5012806004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并阐述了过拟合-欠拟合指标（OUI）作为神经网络训练中激活结构的早期可观测信号，展示其在监督学习、强化学习和在线控制等场景下的实用性；

**💡 创新点**

创新点在于将OUI从单纯的经验调参工具转变为激活结构的结构性观测量，形成激活中心训练理论的初步框架；

**🔧 技术方法**

使用批量基准的激活掩码统计方法，计算每层神经元在批次中的激活比例，随后归一化得到OUI；

**📊 数据集**

主要实验数据集包括CIFAR-100（用于DenseNet、ViT）、CartPole-v1（PPO actor–critic）以及其他标准CNN/Transformer模型；

**📈 对比分析**

与传统的损失、准确率以及RL中的PPO专属指标对比，OUI能在训练的早期就分离出优秀与差劲的超参数设置，提升超参数搜索效率，表现出与最终性能高度相关的早期预测能力；

**⚠️ 局限性**

局限性包括OUI主要适用于ReLU等门控激活函数，其他激活可能需要重新定义；若作为在线控制信号使用，可能导致训练过度优化OUI本身而非学习目标，需要谨慎验证。

---

## 337. Dual-Temporal LSTM with Hybrid Attention for Airline Passenger Load Factor Forecasting: Integrating Intra-Flight and Inter-Flight Booking Dynamics

**arXiv ID:** 2605.11569 | [PDF](https://arxiv.org/pdf/2605.11569v1)

**作者:** ASM Nazrul Islam `[一作]`, Joydeb Kumar Sana `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了双流LSTM加注意力机制，用以预测航班的乘客负载率（PLF）

**💡 创新点**

创新点在于同时利用水平（航班自身的预订累积）与垂直（同航线历史预订位置）两种时序信号，采用偏移匹配生成垂直序列，并结合自注意力、交叉注意力与混合注意力实现信息融合；同时提出七阶段特征筛选流程和PLF为预测目标

**🔧 技术方法**

使用双流LSTM、Self-Attention、Cross-Attention、Hybrid-Attention、Gated/Residual Fusion以及七阶段特征选择（相关系数、互信息、随机森林重要性、SFS、VIF、去重等）

**📊 数据集**

基于Biman Bangladesh Airlines 2023–2024年的航班预订数据（约12,800条记录/航线），并补充机场与假日信息

**📈 对比分析**

与单流LSTM、树模型（LR、RF、XGBoost、LightGBM）以及之前的双流LSTM（Pan、He、Islam、Peng、Gu）对比，模型MAE为2.82、R²为0.95，显著优于所有基线

**⚠️ 局限性**

局限在于仅使用单一航空公司数据，预测仅为整体PLF，未分舱位，且跨国/多航空公司通用性待验证

---

## 338. Sharpen Your Flow: Sharpness-Aware Sampling for Flow Matching

**arXiv ID:** 2605.11547 | [PDF](https://arxiv.org/pdf/2605.11547v1)

**作者:** Aditi Gupta `[一作]` (Lawrence Berkeley National Laboratory), N. Benjamin Erichson `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 1194 | [OpenAlex ID](https://openalex.org/A5007032334)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SharpEuler，一种无训练、无额外成本的基于已训练流匹配模型的 Euler 步骤调度方法，用以提升低评估预算下的采样质量。

**💡 创新点**

创新点在于：① 通过离线精细轨迹计算加速度（速度场变化速率）来构造“尖锐度”曲线；② 将此曲线平滑后映射为时间网格，从而在固定步数预算下将更多采样点集中在动态变化剧烈的区域；③ 该方法完全不改变模型、求解器或评估次数，只调整时间分布。

**🔧 技术方法**

技术手段包括：流匹配 ODE 与 Euler 近似、离线加速度估计（有限差分）、平滑与指数变换、变分原理下的幂律时间密度、统计稳定性分析；实验中使用 FLUX.1-dev 12B 变换器、两维 Synthetic 数据集以及 GPT‑5.5 视觉评判。

**📊 数据集**

数据集：两维 Synthetic（Branched Manifold、Rotated Grid）用于可视化和统计评估；FLUX.1-dev 原始 1024×1024 图像数据，64 条校准 prompt 及 80 条 held‑out prompt，用 50 步参考采样做对照。

**📈 对比分析**

对比方法：Uniform、ETS、Shifted（α=3）以及 SharpEuler（γ=0.5、1.0、1.5）。评价指标包括 Density、Coverage、W₂²（Synthetic）以及 RMSE、FID、CLIP（FLUX）和 GPT‑5.5 视觉评判。实验表明，在 8–16 步 NFE 的低预算下，SharpEuler 在覆盖度、样本质量、像素级相似度和视觉满意度上显著优于其他调度，尤其在 FID 和 RMSE 上降幅最大。

**⚠️ 局限性**

局限性：① 需要先行收集代表性 prompt 的离线轨迹；② 对加速度估计的平滑与指数参数设定敏感；③ 仅适用于 Euler 及其变体，缺乏在线自适应能力；④ 理论保证基于光滑性与稳定性假设，实际中对极端动态或高维复杂场景的适应性尚未充分验证。

---

## 339. PRISM: : Planning and Reasoning with Intent in Simulated Embodied Environments

**arXiv ID:** 2605.11534 | [PDF](https://arxiv.org/pdf/2605.11534v1)

**作者:** Yunn Kang Lim `[一作]` (A*STAR), Shijie Li `[通讯]` (A*STAR)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个诊断性多任务基准，针对具身智能体在多房间公寓中完成从简单到长周期任务的能力评估，采用可执行的、基于可操作性的动作接口和可插拔的感知、记忆、规划模块进行组件级诊断。

**💡 创新点**

创新点包括：① 将任务分为三层能力等级（基础、推理、长周期）以定位感知、意图解析和记忆规划的瓶颈；② 设计了5个高保真Unity公寓场景和300个人工验证任务，并给出跨房间依赖标签；③ 提供统一、agent‑agnostic的可执行动作API，结构化消除动作幻觉；④ 引入可选的感知、记忆、规划探针实现细粒度故障归因。

**🔧 技术方法**

使用技术涵盖：Unity 6/HDRP物理渲染的模拟器；基于场景图的任务生成与人类验证；可操作性动作空间（21个原子动作）；LLM记忆增强策略（长期与短期记忆分离）；可插拔的感知、记忆、规划模块；多指标评估（成功率、步骤数、token消耗）。

**📊 数据集**

数据集为5个光学逼真公寓（每个4–8间房），共79种语义对象类别和293个3D资产；任务集为300个人工验证的指令，按基础、推理、长周期三层分布。

**📈 对比分析**

与七种主流LLM（包括GPT‑5.2、Claude Sonnet、Gemini 3 Flash、GLM‑4.6V及轻量版模型）进行对比，结果显示：① 基础层所有模型>66%成功率，表明感知不构成主瓶颈；② 推理层所有模型≤73%成功率，暗示隐式意图解析是主要难点；③ 长周期层轻量模型性能骤降（如Gemini 2.5‑Flash‑Lite仅20%），并伴随token消耗激增，揭示“过度推理”失效模式。

**⚠️ 局限性**

局限性：实验仅使用oracle感知，未充分评估视觉噪声；仅覆盖住宅环境；任务生成受LLM语言偏差；3D资产受学术使用许可限制；未对非LLM基准进行系统评估。

---

## 340. Multi-Narrow Transformation as a Single-Model Ensemble: Boundary Conditions, Mechanisms, and Failure Modes

**arXiv ID:** 2605.11530 | [PDF](https://arxiv.org/pdf/2605.11530v1)

**作者:** Tatsuhito Hasegawa `[一作]` (University of Fukui), Taisei Tanaka `[通讯]` (University of Fukui)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5109525193)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了单模型集成（SME）中宽模型与多窄模型在相近参数预算下的容量分配对性能的影响。

**💡 创新点**

创新点在于提出并系统化评估 Multi‑Narrow (MN) 变换，揭示其在不同数据规模下的优势与失效机制，并通过内部表示分析解释该现象。

**🔧 技术方法**

主要技术包括 MN 结构变换、标准交叉熵训练、线性 CKA、oracle accuracy、Dead Neuron Ratio 等内部表示分析，以及对不同 MN 强度 r 的实验。

**📊 数据集**

实验使用了 CIFAR‑100、CIFAR‑10、ImageNet‑100 等图像分类数据集，并在多种 CNN 架构（如 ResNet、EfficientNet、MobileNet 等）上进行验证。

**📈 对比分析**

通过在相同参数预算下改变 MN 变换强度 r，并比较在不同训练样本数（IPC）下的测试准确率，结果显示低数据量时高 MN 获得显著提升，数据量大时宽模型更优。

**⚠️ 局限性**

局限性包括仅针对 CNN 图像分类任务，未使用预训练或协作损失；高 MN 方案导致显著前向延迟和显存占用，需要在实际部署中权衡。

---

## 341. EqOD: Symmetry-Informed Stability Selection for PDE Identification

**arXiv ID:** 2605.11524 | [PDF](https://arxiv.org/pdf/2605.11524v1)

**作者:** Gnankan Landry Regis N'guessan `[一作]` (Axiom Research Group), Bum Jun Kim `[通讯]` (University of Tokyo)

**通讯引用:** 13608 | [OpenAlex ID](https://openalex.org/A5100661072)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种完全自动的偏微分方程（PDE）识别流程 EqOD，能从带噪声的时空轨迹数据中通过对称性检测、稳定性选择和残差回退实现高精度方程学习。

**💡 创新点**

创新点在于将 Lie 对称性检测（尤其是 Galilean 对称）与随机 LASSO 稳定性选择相结合，形成自适应的库缩减路径；同时引入残差回退机制防止低质量库导致性能下降。

**🔧 技术方法**

技术包括弱形式结构检验（检测 Galilean 对称）、随机化 LASSO 稳定性选择、交叉验证的弱形式 LASSO、残差比率回退以及自适应阈值与 OLS 去偏。

**📊 数据集**

主要使用 8 种 1+1 维标杆 PDE（Heat、Burgers、KdV、Fisher‑KPP、Adv‑Diff、KS、KdV‑Burgers、React‑Diff）在 4 个噪声水平（0、5%、10%、20%）的合成数据；外部验证使用 WeakIdent 与 PINN‑SR 的 5 个干净基准；还评估 NLS、二维 Navier–Stokes、耦合反应扩散和实测圆柱尾流。

**📈 对比分析**

与全库 WF‑LASSO、PySINDy 2.0.0、WSINDy 重新实现等基线对比，EqOD 在 32 个噪声‑种子组合中平均获得 7 次显著优势（按均值差大于两者标准差之最大值），在 20% 噪声下 Heat 方程实现 F1=1.000±0.000，远优于 WF‑LASSO（0.475±0.181）和 PySINDy（0.000）。在外部基准上也取得全 5 个样例的完美 F1。

**⚠️ 局限性**

局限性主要集中在非标量、二维或耦合系统、以及对小系数（如 Fisher‑KPP）不敏感；稳定性选择在这些情形下未能持续提升；残差回退在高噪声耦合 PDE（如 KdV‑Burgers）可能导致性能退化；对非周期域、非局部或分数阶算子缺乏适配。

---

## 342. XWOD: A Real-World Benchmark for Object Detection under Extreme Weather Conditions

**arXiv ID:** 2605.11521 | [PDF](https://arxiv.org/pdf/2605.11521v1)

**作者:** Chih-Hsin Chen `[一作]` (National Taipei University of Technology), Dong Liu `[通讯]` (Adobe Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个大规模真实极端天气目标检测数据集XWOD，并在其上训练YOLO系列模型，评估跨天气和跨数据集的检测性能。

**💡 创新点**

在规模、天气种类、地理多样性、稀有灾害覆盖以及真实数据优先上实现创新。

**🔧 技术方法**

采用YOLOv8、YOLOv11、YOLOv26等模型，并引入鲁棒训练、长尾权重调节和混合数据增强等技术。

**📊 数据集**

使用自研的XWOD数据集进行训练，并在RTTS、DAWN、WEDGE等公开极端天气数据集上进行零射评估。

**📈 对比分析**

采用mAP_50和mAP_50–95评估标准，零射对RTTS/DAWN/WEDGE的提升分别为+56%/ +83%/ +35%，XWOD内部mAP_50为54.69%，天气间差距高达51.12pp。

**⚠️ 局限性**

仅限2D摄像头检测，缺少LiDAR/雷达；仅涵盖6类交通对象；天气分布偏重洪水；未包含预灾害场景；不适用于分割任务。

---

## 343. A CAP-like Trilemma for Large Language Models: Correctness, Non-bias, and Utility under Semantic Underdetermination

**arXiv ID:** 2605.11672 | [PDF](https://arxiv.org/pdf/2605.11672v1)

**作者:** Vinu Ellampallil Venugopal `[一作]` (International Institute of Information Technology Bangalore), Vinu Ellampallil Venugopal `[通讯]` (International Institute of Information Technology Bangalore)

**通讯引用:** 762 | [OpenAlex ID](https://openalex.org/A5045176640)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了类似CAP定理的LLM三元悖论，阐述在语义不确定条件下强一致性、严格无偏和高效能不能同时满足

**💡 创新点**

首次将分布式系统的CAP定理类比到语言模型决策中，引入“语义不确定”概念，并将偏见视为选择机制的隐含先验

**🔧 技术方法**

使用形式化逻辑（集合、命题、推理关系）对正确性、无偏性、效用进行定义和推导，构建理论框架

**📊 数据集**

未使用具体数据集，文中以若干说明性案例（奖学金、城市推荐、企业价值取向）展示理论

**📈 对比分析**

未进行实验比较或性能评估，本文仅提供理论证明与例证，未给出定量指标

**⚠️ 局限性**

局限性：依赖严格定义，缺乏实证验证；未考虑概率性偏差或部分正确性；理论过度抽象，难以直接映射到实际LLM行为

---

## 344. Every Bit, Everywhere, All at Once: A Binomial Multibit LLM Watermark

**arXiv ID:** 2605.11653 | [PDF](https://arxiv.org/pdf/2605.11653v1)

**作者:** Thibaud Gloaguen `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11346 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于二项编码的多比特LLM水印方法，能够在每个生成位置编码完整消息并通过状态化编码器提升位准确率。

**💡 创新点**

创新点在于不使用位置分配，而是用二项编码在每个位置同时编码所有比特，并通过状态化编码器动态重定向编码压力，显著提高了消息可靠性。

**🔧 技术方法**

采用二项编码、状态化编码器、Soft‑PPL/Red‑Green水印等技术实现多比特编码与解码。

**📊 数据集**

在GPT‑3.5（或类似模型）上，用1000个提示生成250–350 token文本，随机产生16/32/64比特的payload进行实验。

**📈 对比分析**

与8个基线（MPAC、Cycle‑Shift、ArcMark、RSBH、StealthInk等）在不同payload长度和低失真条件下对比，本文方案在消息准确率、位准确率及鲁棒性上均优于对手，差距随payload增大而显著。

**⚠️ 局限性**

鲁棒性仍有限，尤其在词删除、同义词替换或大幅改写后，位准确率迅速下降至≈50%，表明在实际场景中的稳健性仍需进一步提升。

---

## 345. Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation

**arXiv ID:** 2605.11651 | [PDF](https://arxiv.org/pdf/2605.11651v1)

**作者:** Seonghoon Yu `[一作]` (KAIST), Jeany Son `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种新的think-answer视觉语言模型蒸馏框架——Masking-KD，利用显著推理前缀遮蔽来抑制学生模型对文本提示的依赖，迫使其在推理过程中更多地依赖视觉信息；

**💡 创新点**

创新点在于：① token-wise显著推理前缀遮蔽策略，按每个解码步骤动态屏蔽最重要的文本提示；② 自适应遮蔽预算调度，根据教师-学生的逆KL散度自适应调整遮蔽比例；③ 通过在蒸馏时替代传统因果遮蔽，仅遮蔽显著文本前缀，提升视觉记忆与推理质量；

**🔧 技术方法**

技术方法包括：逆KL蒸馏、响应-响应注意力映射、token-wise显著前缀掩码、基于逆KL的自适应遮蔽阈值（σ函数+对数变换）以及多头Transformer的注意力机制；

**📊 数据集**

使用的数据集：ViRK39K（生成蒸馏数据），评估时使用Geometry-3K、MathVista、We-Math、MMK12、MathVerse、LogicVista、MMMU-Pro等多模态推理基准；

**📈 对比分析**

与多种基准模型（开源VLMs、VLM蒸馏方法如CompoDistill、LLaVA-KD、Align-TI、OPSD自蒸馏等）对比，Masking-KD在所有模型规模下均实现了最优或竞争性的Pass@1成绩，尤其在2B和4B规模模型上超越对应的8B教师模型，显著提升视觉推理能力并减轻视觉遗忘；

**⚠️ 局限性**

局限性包括：对推理长度的依赖性强（长推理轨迹更易被遮蔽策略解决），遮蔽比例的超参数调节需要手工设定（ρ_min/ρ_max），以及对高分辨率图像的计算开销仍不小，未来工作需探索更高效的遮蔽策略与自动化参数调优。

---

## 346. Schur Products of Constacyclic Codes via the Constacyclic Discrete Fourier Transform

**arXiv ID:** 2605.11650 | [PDF](https://arxiv.org/pdf/2605.11650v1)

**作者:** Peifeng Lin `[一作]` `[通讯]` (Lomonosov Moscow State University), Peifeng Lin (Lomonosov Moscow State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了常数循环码的Schur乘积，并通过构造常数循环离散傅里叶变换(DFT)来分析其代数结构。

**💡 创新点**

创新点在于将循环码的退化（degenerate）概念推广到常数循环码，引入“模式多项式”描述退化结构；利用常数循环DFT将Schur乘积转化为生成集的和集，进一步推导出GCD式计算方法，并结合加法组合数理得到维度、正则性及填充空间的理论界限。

**🔧 技术方法**

核心技术包括：常数循环DFT及其Frobenius性质、模式多项式与生成集闭包、加法组合数理（和集、子群、余数），以及GCD/乘积多项式的构造与判定。

**📊 数据集**

论文主要为理论研究，未使用具体数值数据集，所有结论均为解析推导。

**📈 对比分析**

由于缺乏实验数据，无法与实测方法直接比较；作者通过理论证明表明在维度之和超过长度时Schur乘积即为全空间，并给出了正则性上界和足够大次数时填充全空间的条件。

**⚠️ 局限性**

限制主要在于：① 研究聚焦理论分析，缺乏实验验证；② 对大规模场扩展与编码/译码实现的复杂度讨论不足；③ 模式多项式的计算在实际编码中仍需进一步优化。

---

## 347. MIST: Reliable Streaming Decision Trees for Online Class-Incremental Learning via McDiarmid Bound

**arXiv ID:** 2605.11617 | [PDF](https://arxiv.org/pdf/2605.11617v1)

**作者:** Phu-Hoa Pham `[一作]` (Vietnam National University), Long Tran-Thanh `[通讯]` (University of Warwick)

**通讯引用:** 2744 | [OpenAlex ID](https://openalex.org/A5012915897)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为McDiarmid Incremental Streaming Tree（MIT）的在线决策树，专为开放世界的连续学习设计；

**💡 创新点**

核心创新包括：①利用McDiarmid不等式得到K无关的Gini增益置信半径，解决分裂准则随类数扩展失效的问题；②引入贝叶斯继承机制（截断高斯矩投影与Dirichlet分配），在分裂时将父节点统计信息迁移到子节点，缓解冷启动不稳定；③采用Per-leaf KLL分位数草图实现连续阈值评估与几何自适应叶子预测，统一了分裂与预测的数据结构；

**🔧 技术方法**

技术手段涵盖：McDiarmid不等式、贝叶斯推断（截断高斯、Dirichlet）、KLL分位数草图、Gini impurity分裂、Naïve Bayes叶子模型、统计方差缩减分析；

**📊 数据集**

实验数据集包括：合成高斯混合（Synth-10、Synth-50）、八个真实表格数据集（Covertype、Split-MNIST、Pendigits、Shuttle、Letter、Wine、Iris、HAR）以及多种非高斯形状的压力测试合成流；

**📈 对比分析**

与EFDT、HAT、Rutkowski、-NR等流式树以及NCM、RLS、PEC等全局参数化方法进行比较；MIT-G在近似高斯流上与全局方法相当，MIT-K在强非高斯流上明显优于全局方法，且在多数基准上均保持低遗忘；

**⚠️ 局限性**

局限性包括：对分裂投影的高斯假设在非球面分布下不完全适用，可能需要引入Copula等更丰富的依赖建模；此外，当前的置信半径基于一次性决策，缺乏时间均匀序列保证，未来工作需探索超martingale方法以消除多次检验影响。

---

## 348. PRISM: A Geometric Risk Bound that Decomposes Drift into Scale, Shape, and Head

**arXiv ID:** 2605.11608 | [PDF](https://arxiv.org/pdf/2605.11608v1)

**作者:** Chieh-Yen Lin `[一作]` (Appier AI Research), Shao-Hua Sun `[通讯]` (Appier AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PRISM 诊断框架，用三轴（尺度、形状、头部偏差）来量化并解释 LLM 量化或 LoRA 调优后变体的风险漂移。

**💡 创新点**

创新点在于：① 通过线性头与近等距背骨推导出闭式交叉熵风险上界；② 将误差分解为可测量的三轴，直接映射到实际失效模式；③ 在形状轴上实现可微正则化，减轻 LoRA 造成的灾难性遗忘。

**🔧 技术方法**

核心技术包括：Procrustes 对齐、特征尺度与形状度量、协方差加权头部误差、Lipschitz 变分估计、可微形状正则化。

**📊 数据集**

使用 Llama‑3.1‑8B、Qwen3‑8B（以及 Ministral‑3‑8B、DeepSeek‑R1‑Distill‑Llama‑8B 等）作为模型；五个基准（MMLU、ARC、TriviaQA、SQuAD、GSM8K）评估风险；量化方案包含 GGUF、GPTQ、BitsAndBytes；LoRA 微调任务为 TruthfulQA 与 BBQ。

**📈 对比分析**

对比方法：传统 SVCCA、CKA 等整体相似度；PRISM 在 PTQ 与 LoRA 情景下的 Spearman 相关性均在 0.82‑0.83 之间，且能精准定位主导轴；形状正则化在对抗遗忘时表现优于经验重放，平均降低 19% 的 downstream 损失。

**⚠️ 局限性**

局限性：仅提供风险上界，无法给出精确的绝对风险估计；依赖教师强制特征，未考虑自由生成轨迹；仅适用于线性预测头，对非线性头部或更复杂的模型结构需进一步扩展。

---

## 349. Convolutional-Neural-Networks for Deanonymisation of I2P Traffic

**arXiv ID:** 2605.11606 | [PDF](https://arxiv.org/pdf/2605.11606v1)

**作者:** Luca Rohrer `[一作]` (Lucerne University of Applied Sciences and Arts), Dieter Arnold `[通讯]` (Lucerne University of Applied Sciences and Arts)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在实验室生成合成I2P流量，并通过k‑means聚类、卷积神经网络等机器学习方法，对被动流量进行分析，评估其对服务去匿名化的可行性。

**💡 创新点**

首次将Fano不等式与机器学习相结合，构建理论框架量化匿名网络中的信息泄露，并证明CNN在仅使用元数据时可在受控实验中实现高精度分类，但在真实网络中失效。

**🔧 技术方法**

核心技术包括Fano不等式理论分析、k‑means聚类、卷积神经网络（CNN）、Python/TensorFlow等深度学习框架。

**📊 数据集**

使用实验室自行生成的两套数据集（小型约3.4k条包、完整实验约5.8万条包）以及公开I2P网络的真实抓包数据作为验证集。

**📈 对比分析**

与传统k‑means聚类相比，CNN在实验室数据上的准确率可达99.5%，但在公开网络中仅维持约81%（仅元数据）或更低，显示模型在不同环境下表现差异巨大。

**⚠️ 局限性**

主要局限在于实验室数据与真实网络差异过大导致模型泛化性差，即使在实验室得到高准确率也无法实现完整的服务去匿名化，说明I2P的加密与流量混淆机制对被动攻击具有强鲁棒性。

---

## 350. Targeted Tests for LLM Reasoning: An Audit-Constrained Protocol

**arXiv ID:** 2605.11599 | [PDF](https://arxiv.org/pdf/2605.11599v1)

**作者:** Hongmin Li `[一作]` (Institute of Science Tokyo), Hongmin Li `[通讯]` (Graduate School of Frontier Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种可审计的目标提示变异协议，用于识别在语义合法的提示变体下导致语言模型推理错误的情况，并实现了Component‑Adaptive Prompt Sampling（CAPS）与均匀采样的对比实验。

**💡 创新点**

提出在固定任务库、有限提示组件语法和查询预算下的审计约束协议，将自动误匹配信号与人工审计分离，并以提示关键作为唯一发现单位，从而使提示变异实验可复现且可检验。

**🔧 技术方法**

使用有限组件语法生成提示、CAPS基于误匹配信号的加权采样、结构与语义有效性检查、答案提取器、模型接口以及完整实验轨迹记录等技术。

**📊 数据集**

采用手工设计的小型推理任务库（v1–v4），涵盖算术、符号关系和逆向规则三类任务，并在Qwen2.5‑7B、Phi‑3 Mini、Mistral‑7B等模型上进行实验。

**📈 对比分析**

在相同查询预算下对比CAPS与均匀采样，使用审计后错误率、首次错误查询次数和唯一提示键数量作为评估指标；实验表明两者在审计后发现率和独特键数上无显著优势，CAPS未优于均匀采样。

**⚠️ 局限性**

任务库规模有限、提示空间受限、仅关注语义保持与提取有效性，无法覆盖完整推理行为；审计过程人工耗时，且模型错误检测受答案提取与语义检查的局限性影响。

---

## 351. OmniThoughtVis: A Scalable Distillation Pipeline for Deployable Multimodal Reasoning Models

**arXiv ID:** 2605.11629 | [PDF](https://arxiv.org/pdf/2605.11629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 352. Grounding by Remembering: Cross-Scene and In-Scene Memory for 3D Functional Affordances

**arXiv ID:** 2605.11616 | [PDF](https://arxiv.org/pdf/2605.11616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 353. Native Explainability for Bayesian Confidence Propagation Neural Networks: A Framework for Trusted Brain-Like AI

**arXiv ID:** 2605.11595 | [PDF](https://arxiv.org/pdf/2605.11595v1)

**作者:** Georgios Makridis `[一作]` (ExpertAI-Lux S.à r.l), Dimosthenis Kyriazis `[通讯]` (University of Piraeus)

**通讯引用:** 4531 | [OpenAlex ID](https://openalex.org/A5069674161)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

针对贝叶斯置信传播神经网络（BCPNN）提出了完整的可解释性框架，包括XAI分类法、16个架构级解释原语、5个设计时配置原语，并提供了从BCPNN已有量化指标直接计算解释的闭式算法。

**💡 创新点**

创新点在于：①把BCPNN的权重、偏置、超列后验、结构可塑性、原型、吸引子路径等硬件友好且可直接可解释的模块与主流XAI类别（归因、原型、概念、反事实、机制）一一对应；②引入了“配置即解释”理念，将模型超参数（如超列数、连接掩码、可塑性阈值、时间窗口）视为可审核的解释文档；③在不增加推理负载的前提下，实现了零成本的 OOD 检测、漂移监测、局部鲁棒性证明等。

**🔧 技术方法**

主要技术包括：Bayesian-Hebbian 学习规则、软WTA 竞争、结构可塑性连通度更新、混合架构的吸引子动态、Spiking BCPNN 的 z-/p-trace、闭式归因与模块化因子分解算法、以及欧盟 AI Act 对高风险系统透明度的合规映射。

**📊 数据集**

论文未给出具体实验数据集，作者提及在标准无监督学习基准（如 MNIST、CIFAR‑10/100、ImageNet 低分辨率版本）和工业场景（金融时序、IoT 传感器、网络安全日志）上已实现子瓦特级别的 FPGA 推理，但并未提供公开数据集细节。

**📈 对比分析**

比较方法主要是与传统后置解释方法（SHAP、LIME、Integrated Gradients、TCAV 等）以及深度学习的性能对照。BCPNN 在无监督表示学习、稀疏分布式编码和能耗方面与当前深度网络保持竞争性；在解释性方面，由于所有解释量化直接来自模型内部参数，推理时不需要额外计算，显著降低了资源占用。性能指标（准确率、能耗）与现有深度模型相当，但提供了完整的解释性证书。

**⚠️ 局限性**

局限性包括：①大规模深层 BCPNN 的解释图易于可视化但人类可解释性可能随层数增加而下降；②缺乏系统级的可靠性验证与正式的可信度评估；③需要与实际生产部署合作进行用户研究和基准测试；④新引入的原语（P12–P16）尚未在大型实际数据集上做充分的实验验证。

---

## 354. Cochise: A Reference Harness for Autonomous Penetration Testing

**arXiv ID:** 2605.11671 | [PDF](https://arxiv.org/pdf/2605.11671v1)

**作者:** Andreas Happe `[一作]` (TU Wien), Jürgen Cito `[通讯]` (TU Wien)

**通讯引用:** 1704 | [OpenAlex ID](https://openalex.org/A5033732305)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个597行Python的Cochise参考框架，用于LLM驱动的自主渗透测试实验。

**💡 创新点**

创新在于提供可复现、极简的Agent架构、统一的执行、日志、重放与分析接口，方便对比模型与架构。

**🔧 技术方法**

采用Planner‑Executor分层架构，Planner维护长期状态，Executor实现ReAct式命令执行，使用LiteLLM连接LLM并通过SSH远程执行。

**📊 数据集**

使用Game of Active Directory（GOAD）测试平台，并提供GOAD轨迹日志数据集。

**📈 对比分析**

通过与Gemini-3-Flash和Claude-4.7-Opus两大LLM对比，评估账户获取成本/小时，显示Cochise能产生多阶段轨迹且成本低于对手。

**⚠️ 局限性**

局限包括仅支持已获得SSH访问的单跳宿主，未涵盖初始访问、人工干预，以及对更大规模或多目标网络的扩展。

---

## 355. Seirênes: Adversarial Self-Play with Evolving Distractions for LLM Reasoning

**arXiv ID:** 2605.11636 | [PDF](https://arxiv.org/pdf/2605.11636v1)

**作者:** Chi Zhang `[一作]` (Wuhan University), Jing Zhang `[通讯]` (Wuhan University)

**通讯引用:** 27150 | [OpenAlex ID](https://openalex.org/A5100345341)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自对弈式强化学习框架 Seirênes，通过同一模型同时充当对手生成误导性上下文和推理者完成问题，从而在不改变任务本身的情况下提升语言模型的推理鲁棒性。

**💡 创新点**

创新点在于将上下文干扰转化为内部训练信号，利用参数共享的对抗自进化循环，动态产生逼真且误导性的提示，让模型在推理过程中学会主动忽视干扰而非仅依赖表面模式。

**🔧 技术方法**

核心技术包括基于 GRPO 的无价值估计强化学习、REINFORCE/GRPO 的对抗奖励设计、角色条件化策略、配对推理与提示生成的 rollout 包、FIFO 缓冲队列与 Mastery‑aware 采样等实现细节。

**📊 数据集**

使用七个数学推理基准（AIME 2024–26、IMO‑Bench、Minerva Math、OlympiadBench、HMMT 2026）以及多种扰动测试集（Math‑Perturb、GSMIR、MMLU‑Perturb、OpenBookQA‑Perturb）进行实验。

**📈 对比分析**

与标准 RLVR 基线（DAPO、ExGRPO、Dr.GRPO）和协作提示基线（SAGE、LUFFY、Scaf‑GRPO、InT）在相同算力预算下对比，Seirênes 在四个模型规模上平均提升 7–10 分，显著优于所有对比方法。

**⚠️ 局限性**

局限性包括额外的推理步骤导致更高的计算成本、在更大提示数量或更高容量模型上的可扩展性受限、以及对离线静态提示的适应性不足；此外，实验主要集中在数学推理任务，跨域通用性尚待验证。

---

## 356. EpiCastBench: Datasets and Benchmarks for Multivariate Epidemic Forecasting

**arXiv ID:** 2605.11598 | [PDF](https://arxiv.org/pdf/2605.11598v1)

**作者:** Madhurima Panja `[一作]` (Sorbonne University Abu Dhabi), Nan Liu `[通讯]` (Duke-NUS Medical School)

**通讯引用:** 15760 | [OpenAlex ID](https://openalex.org/A5100367554)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EpiCastBench，一个包含 40 个多变量流行病时间序列的公开基准框架，用于统一评估流行病预测模型。

**💡 创新点**

创新点在于：①构建多样化、相关性强的多变量数据集；②制定统一的预处理、训练、评估流程；③结合统计、机器学习、深度学习和基于预训练的基础模型进行全方位对比；④提供基准结果、统计显著性分析和可复现的代码与数据。

**🔧 技术方法**

技术手段包括：多变量时间序列特征分析（trend、seasonality 等）、统一的 Darts 框架实现 15 种模型、滚动窗口评估、MASE、RMSE、MAE、SMAPE 四种误差指标、Friedman 与 MCB 统计显著性检验。

**📊 数据集**

数据集：40 个多变量流行病数据集，涵盖 COVID‑19、鸡痘、登革热、寨卡、流感、麻疹、结核、登革热等疾病，来自 27 个不同地区，时间粒度包括日、周、月，样本长度从 60 到 2088，维度从 4 到 107，包含零膨胀和缺失值。

**📈 对比分析**

比较方法：使用统一的 24 条历史窗口预测 12 步的滚动窗口实验，评估三种预测时长（长、中、短）下的 MASE、RMSE 等指标。统计检验显示所有模型差异显著，基础模型 TimesFM 与 Chronos‑2 在大多数时长和指标上取得最佳或相近性能；经典 ML 模型（如 XGBoost、KAN、Random Forest、DLinear）在中短期任务表现竞争；深度学习模型表现更不稳定；Naive 基线最差。

**⚠️ 局限性**

局限性：未考虑空间相关性，缺乏概率预测与分布评估；仅评估确定性方法；未包含流行病学信息驱动的模型；未来计划扩展至时空预测、概率预测以及机制驱动方法。

---

## 357. ShapeCodeBench: A Renewable Benchmark for Perception-to-Program Reconstruction of Synthetic Shape Scenes

**arXiv ID:** 2605.11680 | [PDF](https://arxiv.org/pdf/2605.11680v1)

**作者:** Shivam Kumar `[一作]` `[通讯]`, Shivam Kumar

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ShapeCodeBench，一个可重置、基于渲染的图像到可执行绘图程序的基准测试，评估模型从渲染图像推断DSL程序的能力。

**💡 创新点**

创新点在于整合确定性渲染评估、可控难度级别和可续期的生成器，使得基准可以快速自动重现且避免数据泄漏，同时提供多维度评价指标。

**🔧 技术方法**

采用Python限定解析器、Pillow渲染器、基于种子随机生成的场景、以及多任务的评估指标（精确匹配、像素准确率、前景IoU等）。

**📊 数据集**

使用自定义的四原语DSL（圆、空心圆、正方形、空心正方形）生成的合成数据，分为易、中、难三档，共150个样本（eval_v1）并发布种子与哈希。

**📈 对比分析**

与空程序基准、经典计算机视觉启发式以及两大前沿多模态模型（Claude Opus 4.7与GPT‑5.5）在不同推理成本下进行对比；精确匹配最高仅为0.087（启发式）与0.027（LLM），前景IoU最高达0.87（GPT‑5.5），显示基准仍未饱和。

**⚠️ 局限性**

局限性包括单色调、仅四种原语、零样本评估、缺乏人类基线、仅评估离线指标、对模型推理的不可复现性以及未涵盖更广泛的多模态系统。

---

## 358. A Cross-Cultural Analysis of Animated Representations of Emotions for Wearable Interfaces

**arXiv ID:** 2605.11668 | [PDF](https://arxiv.org/pdf/2605.11668v1)

**作者:** Michal R. Wrobel `[一作]` (Gdansk University of Technology), Agnieszka Landowska `[通讯]` (Gdansk University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了波兰和土耳其两种文化背景下，抽象几何动画表达情感的可视化参数（颜色、形状、尺寸、速度、动画类型）的选择与对应关系，评估其跨文化一致性。

**💡 创新点**

首次系统检验抽象动画参数在不同文化中的普遍性，并发现颜色与尺寸具有跨文化共性，而速度表现出文化差异，为全球可穿戴情感反馈算法提供可适配的设计依据。

**🔧 技术方法**

采用基于网页的交互式问卷收集RGB颜色、形状、尺寸、速度等参数；将RGB转换为HSV；使用卡方检验评估文化差异；计算均值、标准差与置信区间分析情感-参数关联。

**📊 数据集**

105名学生（51波兰，54土耳其），年龄19-23，使用自定义的10种情感标签集进行动画设计，形成情感-参数对应数据集。

**📈 对比分析**

通过卡方检验比较两国在颜色、形状、速度、尺寸等参数上的独立性，结果显示颜色与尺寸无显著差异，速度在部分情感上显著差异；表明颜色与尺寸可作为全球通用可视化维度，速度需进行文化适配。

**⚠️ 局限性**

样本仅限于波兰和土耳其的年轻大学生，可能不代表更广泛人群；情感标签有限，未覆盖更细腻或文化特定情绪；抽象动画与真实表情的比较未深入。

---

## 359. HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment

**arXiv ID:** 2605.11662 | [PDF](https://arxiv.org/pdf/2605.11662v1)

**作者:** Guorui Li `[一作]` (Shenzhen University), Zhong Ming `[通讯]` (Shenzhen University)

**通讯引用:** 13269 | [OpenAlex ID](https://openalex.org/A5100633973)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个可插拔的框架HSUGA，专门用于提升LLM增强的序列推荐系统，主要包含两大模块：分层语义理解（HSU）和基于群组的对齐（GAA）。

**💡 创新点**

创新点在于（1）通过分阶段、两步（操作选择+执行）的编辑式语义理解，控制兴趣演化并避免长序列推理困难；（2）根据用户活跃度动态调整语义利用强度，长尾用户获得更强引导、活跃用户则弱化对齐，显著缓解长尾问题。

**🔧 技术方法**

技术手段包括：LLM（如Qwen2.5-7B/14B）进行基于提示的分阶段推理；四种原子编辑操作（Add/ Delete/ Update/ Retain）实现可解释的兴趣更新；邻居检索与自蒸馏相结合的组级对齐策略。

**📊 数据集**

在Steam、Amazon Fashion、Amazon Beauty三大真实数据集上进行了实验。

**📈 对比分析**

与多种基线（GRU4Rec、BERT4Rec、SASRec以及现有LLM增强方法）进行对比，HSUGA在HR@10、NDCG@10等Top‑K指标上均获得显著提升，尤其在尾部用户和尾部商品上表现尤为突出。

**⚠️ 局限性**

主要局限是：仍然需要较高的计算开销，尤其是分阶段LLM推理导致离线时间和内存负担增大；未来需进一步优化推理效率以适应工业部署。

---

## 360. GeomHerd: A Forward-looking Herding Quantification via Ricci Flow Geometry on Agent Interactive Simulations

**arXiv ID:** 2605.11645 | [PDF](https://arxiv.org/pdf/2605.11645v1)

**作者:** Lake Yang `[一作]` (Imperial College London), Dunhong Jin `[通讯]` (University of Hong Kong)

**通讯引用:** 102 | [OpenAlex ID](https://openalex.org/A5046073528)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在可观测的 LLM 驱动多代理金融模拟器中构建代理交互图，利用 Ollivier–Ricci 曲率及其正负分解、Ricci 流奇点时间和有效词汇熵等几何指标，前瞻性地检测并预测市场参与者的集体行为（herding）

**💡 创新点**

① 直接在代理行动图上使用离散 Ricci 曲率，绕过传统价格相关图的观测滞后；② 对曲率按正负分解，分别捕捉聚集和传播机制；③ 结合 Ricci 流的奇点时间与行为词汇熵形成多维前瞻指标；④ 构建与经典 CSAD 等宏观统计的理论桥梁

**🔧 技术方法**

LLM 驱动的多代理模拟器、Ollivier–Ricci 曲率计算、离散 Ricci 流、CUSUM+Kendall‑τ 预警检测、有效词汇熵、Kronos‑style Transformer 预测头、mean‑field 解析

**📊 数据集**

Cividino‑Sornette 连续旋转模型（CWS）的仿真轨迹作为主要实验数据集；Vicsek 自驱动粒子模型作为跨域验证；以及对应的交易流和价格序列

**📈 对比分析**

与价格相关曲率、CSAD、LSV、交易流、点过程几何检测、相互信息基线等七个基线对比；GeomHerd 在保持低误报率的条件下，比价格图方法提前数十步发出预警；在预测任务中，GeomHerd 条件下的 MAE 低于所有基线；在 Vicsek 模型上曲率 AUROC 较高，表明跨域可泛化

**⚠️ 局限性**

依赖于可观测的代理行动数据，现实金融市场难以直接获取；理论桥梁基于近似假设，尚需更严格证明；目前仅在仿真环境验证，真实市场数据的实证检验仍待开展

---

## 361. Finite Sentence-Interface Control for Learning Bounded-Fan-Out Linear MCFGs under Fixed Monoid Typing

**arXiv ID:** 2605.11644 | [PDF](https://arxiv.org/pdf/2605.11644v1)

**作者:** Takayuki Kuriyama `[一作]` `[通讯]`, Takayuki Kuriyama

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出在固定有限幺半群映射下，通过句子界面类型实现受限多上下文无关文法（MCFG）从正样本的可识别性与多项式时间学习器的构造

**💡 创新点**

创新点在于引入句子界面类型这一有限外部控制对象，能够同时记录子词组的排列顺序与边界的幺半群值，从而克服传统分配性学习方法在多值输出场景下的局限；同时给出了基于该类型的符号与规则观测的特征样本与可证明确性

**🔧 技术方法**

利用句子界面类型、运输映射、受限MCFG的精炼、特征样本构造以及多项式时间构造的单位消除等技术；并基于(f,h)-tuple 可替代性定理进行学习器的正确性证明

**📊 数据集**

该工作为理论性研究，未使用任何实验数据集，所讨论的语言类为受(f,h)-tuple 可替代性约束下的MCFG语言

**📈 对比分析**

方法通过证明在固定 fan-out 上限 f 与固定映射 h 的前提下，学习器在包含特征样本后即可在多项式时间内重建目标语言，达到“识别于极限”与“多项式时间构造”双重性质，尚未进行实验对比

**⚠️ 局限性**

局限性在于仅适用于受限的工作形式化（二元线性非删除MCFG）且需要已知 fan-out 上限与固定映射 h；特征样本的大小与构造复杂度不受限，且该方法不适用于一般MCFG语言的学习

---

## 362. Performance of QUBO-Formulated MIMO Detection Under Hardware Precision Constraints

**arXiv ID:** 2605.11626 | [PDF](https://arxiv.org/pdf/2605.11626v1)

**作者:** Seyedkhashayar Hashemi `[一作]` (1QB Information Technologies), Moslem Noori `[通讯]` (1QB Information Technologies)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在有限硬件精度下，将MIMO检测问题转化为QUBO后所产生的量化误差对检测性能的影响，并提出了多种量化方案。

**💡 创新点**

创新点在于首次给出QUBO系数的概率分布、提出同质与异质量化策略，并推导了保持最优解所需的精度阈值。

**🔧 技术方法**

采用了QUBO建模、均匀量化、归一化预处理、并使用并行温度退火（Parallel Tempering）求解器进行数值仿真。

**📊 数据集**

实验使用 4×4、16×16、32×32 天线的 MIMO 系统，QAM 4、16、64、256，SNR 5–30 dB 的随机 Rayleigh 通道实例。

**📈 对比分析**

与全精度基准和 MMSE 对比，异质量化在较低位宽（6–12 位）即可达到或接近全精度误码率，优于同质量化；当位宽不足时，误码率会显著上升。

**⚠️ 局限性**

主要局限是对量化误差的分析假设最坏情况，且仅考虑均匀量化；在非 Rayleigh 信道或存在硬件非理想性时结果可能不再适用。

---

## 363. Nice Fold or Hero Call: Learning Budget-Efficient Thinking for Adaptive Reasoning

**arXiv ID:** 2605.11625 | [PDF](https://arxiv.org/pdf/2605.11625v1)

**作者:** Zhaomeng Zhou `[一作]` (University of Science and Technology of China), Junda Lin `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出预算高效思考（Budget-Efficient Thinking，BET）框架，利用可解性（solvability）信息在测试时动态分配推理计算，避免对不可解查询浪费算力并保留对难但可解查询的足够推理深度。

**💡 创新点**

创新点在于：① 将推理视为在不确定性下的投资决策；② 引入可解性感知的预算分配；③ 设计非对称奖励（R_val、R_eff、R_cal）以鼓励三种行为（短解、优雅折叠、英雄调用）并避免统一压缩；④ 采用两阶段训练（行为冷启动+GRPO）实现可解性估计与预算规划的协同。

**🔧 技术方法**

技术手段包括：多轮滚动（K=16）采样获取可解率 ᵨ(x) 与高效解算成本 ᵩ*(x)；两阶段训练：① 通过有结构的SFT演示训练模型掌握可解性估计与预算规划；② 使用GRPO在投资‑成本感知奖励下进行强化学习；奖励分解为 solve/abstain（R_val）、长度适配（R_eff）和预算校准（R_cal）；还利用群组统计、动态奖励和群组相对优势更新策略。

**📊 数据集**

训练数据：DeepMath-103K（20K用于训练），去除评测集污染；评测数据：七大推理基准——数学领域的 Omni‑Math、MATH500、AMC‑23、AIME‑25；跨域零样本转移的 GPQA‑Diamond（科学 QA）、MUSR（多步叙事推理）、LSAT‑AR（逻辑推理）。

**📈 对比分析**

与七类基线（ThinkSwitcher、DiffAdapt、DEER、VeriThinker、Overthink、Length‑Penalty、DR.SAF）对比。BET 在三大模型（Qwen3‑4B、DeepSeek‑R1‑Distill‑Qwen‑7B/14B）上平均节省 55% 以上推理 token，同时保持或提升准确率；在数学基准上处于 Pareto 前沿；在跨域基准上实现 53–59% token 节省且准确率不低于原始模型，显示出强泛化能力。

**⚠️ 局限性**

局限性包括：① 需要多轮滚动（K=16）以获得可靠可解性估计，增加计算开销；② 可解性估计受模型当前能力波动影响，可能导致预算规划失真；③ 对 δ、β、p 等超参数敏感，需根据任务和预算约束手动调优；④ 目前仅在推理任务上验证，其他任务（如对话、生成）仍需进一步探究。

---

## 364. HorizonDrive: Self-Corrective Autoregressive World Model for Long-horizon Driving Simulation

**arXiv ID:** 2605.11596 | [PDF](https://arxiv.org/pdf/2605.11596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 365. Reviving In-domain Fine-tuning Methods for Source-Free Cross-domain Few-shot Learning

**arXiv ID:** 2605.11659 | [PDF](https://arxiv.org/pdf/2605.11659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 366. Nautilus: From One Prompt to Plug-and-Play Robot Learning

**arXiv ID:** 2605.11665 | [PDF](https://arxiv.org/pdf/2605.11665v1)

**作者:** Yufeng Jin `[一作]` (TU Darmstadt), Georgia Chalvatzaki `[通讯]` (TU Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为Nautilus的开放源代码机器人学习研究 harness，能够将自然语言请求自动转化为端到端的重现、评估、微调和部署工作流；

**💡 创新点**

创新点在于提出基于 typed interface contracts、chambered execution 与统一的 inter‑module transport 的 substrate，配合 Guides、Sensors、State 的 content layer，显著降低跨政策、基准和机器人实例的集成成本；

**🔧 技术方法**

采用大型语言模型编码代理、容器化、WebSocket 通信、JSON‑schema 合约、注册表机制以及自动化的验证与监控工具实现；

**📊 数据集**

使用六大基准（LIBERO、ManiSkill、RoboCasa、RoboTwin、ALOHA、RoboArena）以及 Franka Panda 与 Unitree H1 两款机器人进行实验；

**📈 对比分析**

通过跨策略验证、ablation 试验和真实机器人部署，证明了包装器的可信度与重现性，集成成本与传统手写 glue 代码相当且性能与公开参考值相近；

**⚠️ 局限性**

局限在于目前仅覆盖重现、评估与部署工作流，物理模拟器层面统一不足，且依赖于底层 coding‑agent 平台的更新。

---

## 367. Safety Context Injection: Inference-Time Safety Alignment via Static Filtering and Agentic Analysis

**arXiv ID:** 2605.11664 | [PDF](https://arxiv.org/pdf/2605.11664v1)

**作者:** Zhenhao Xu `[一作]` (City University of Macau), Tianqing Zhu `[通讯]` (City University of Macau)

**通讯引用:** 8341 | [OpenAlex ID](https://openalex.org/A5068985187)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Safety Context Injection (SCI)框架，通过在LLM生成前注入结构化风险报告实现推理时安全对齐；

**💡 创新点**

创新点在于将安全评估与任务生成解耦，直接将外部安全评估结果注入模型上下文，而非仅做二元拦截；

**🔧 技术方法**

技术包括两种防御变体：轻量级Static Model Filtering (SMF)和深度Agentic Filtering (DAF)，分别利用单次分类器和多轮智能代理分析；

**📊 数据集**

数据集主要为AdvBench和GPTFuzz，覆盖五类jailbreak攻击（DarkCite、DRA、CoL-SingleTurn、CoL-MultiTurn、AutoRAN）；

**📈 对比分析**

对比方法包括未防御、SMF、DAF（DS-V3.2或GPT-OSS-20B）三种部署；在攻击成功率(ASR)和毒性分数(TS)上，SMF显著降低模板类攻击的ASR，DAF进一步压低所有攻击的ASR和TS，尤其在长上下文或语义伪装攻击中效果更显著；

**⚠️ 局限性**

局限性包括：1）SMF在语义伪装或长上下文攻击中可能漏判；2）DAF的令牌开销和推理时延显著增加；3）仍无法完全消除所有jailbreak，尤其是新颖的、未见过的攻击模式；4）需要对后端安全分析模型的可靠性和可解释性进行更严格的评估。

---

## 368. Enhancing Multilingual Counterfactual Generation through Alignment-as-Preference Optimization

**arXiv ID:** 2605.11632 | [PDF](https://arxiv.org/pdf/2605.11632v1)

**作者:** Yilong Wang `[一作]` (Technical University Berlin), Simon Ostermann `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于直接偏好优化（DPO）的多语种自我反事实解释（SCE）生成框架，利用可组合的三项评分（flip、augmentation、edit）构建偏好对并对模型进行对齐；

**💡 创新点**

创新点在于将偏好优化方法直接应用于SCE生成，显式平衡有效性与最小化改动，并通过多语种评分融合显著提升非英语语言下SCE的有效性；

**🔧 技术方法**

核心技术为直接偏好优化（DPO）、基于词级扰动的候选采样、三维评分函数以及跨语言相似度评估；

**📊 数据集**

使用两大多语种数据集SIB200（205语种）和TAXI1500（1504语种），挑选7种语言（英、德、中、阿、斯威、越、俄）进行实验；

**📈 对比分析**

与DG-CF、TB-CF及SFT基线对比，DPO框架平均提升12.55%有效率（SLFR）且保持甚至提升最小化（SS），在流畅度（PPL）和跨语言编辑相似度（CES）上也表现更优；

**⚠️ 局限性**

局限性包括只评估了7种语言，未覆盖更广泛语言；实验仅采用DPO，未探索其他强化学习或混合优化策略。

---

## 369. GraphFlash: Enabling Fast and Elastic Graph Processing on Serverless Infrastructure

**arXiv ID:** 2605.11631 | [PDF](https://arxiv.org/pdf/2605.11631v1)

**作者:** Chen Zhao `[一作]` (University of Melbourne), Adel N. Toosi `[通讯]` (University of Melbourne)

**通讯引用:** 5005 | [OpenAlex ID](https://openalex.org/A5083902835)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了完全基于无服务器（serverless）架构的图计算框架 GraphFlash，支持旋转（rotating）和固定（pinned）两种执行模式。

**💡 创新点**

创新点包括：①子图中心的编程模型与无状态函数结合；②分区感知的键聚合、函数内分区共置以及超步感知激活等三项针对无服务器环境的系统级优化；③在保持完全无服务器特性的同时实现接近传统分布式框架的性能。

**🔧 技术方法**

技术手段包括：使用外部存储 MaaS（如 Dragonfly/MinIO/S3）实现状态和消息传递；C++实现的插件化算法接口；多线程函数实例支持多分区并行；二进制序列化、Zstd 压缩、Cilium CNI 等网络与内存优化；以及 Knative 与 AWS Lambda 两个云平台的部署。

**📊 数据集**

评估数据集涵盖 LDBC Graphalytics 的多规模图：Dota-league、com-friendster、Graph500 系列（G3-G8）、datagen-9_2-zf 等，覆盖从数万到十亿级顶点/边的各种结构。

**📈 对比分析**

与 Graphless、FaaSGraph、GraphScope、Giraph 等基准比较，GraphFlash 在小中型图上可实现 12–127 倍速度提升、48 倍加速（DL 数据集），在大规模图上保持与 GraphScope/ Giraph 相当；在成本方面相比 Graphless 资源消耗下降 90% 以上、单个函数即可完成任务。

**⚠️ 局限性**

局限性包括：仍需依赖外部存储进行消息传递，导致网络 I/O 成为瓶颈；冷启动、函数间无直接通信限制了极大规模并发的性能；在极大图或内存受限场景下仍受 Lambda/Knative 内存上限影响。

---

## 370. Single-Shot HDR Recovery via a Video Diffusion Prior

**arXiv ID:** 2605.11628 | [PDF](https://arxiv.org/pdf/2605.11628v1)

**作者:** Chinmay Talegaonkar `[一作]` (University of California San Diego), Nicholas Antipa `[通讯]` (University of California San Diego)

**通讯引用:** 1441 | [OpenAlex ID](https://openalex.org/A5027006335)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种将单次 HDR 重建视为条件视频生成并融合的框架，先使用微调的视频扩散模型生成曝光序列（LDR 曝光堆栈），再通过轻量级 UNet 对该序列进行像素级融合得到最终 HDR 图像。

**💡 创新点**

创新点在于：① 把曝光堆栈等同于视频序列，利用视频扩散模型的时序先验生成可控曝光堆栈；② 通过单一模型生成完整曝光序列，避免了传统方法中为阴影和高光分别训练的多模型；③ 通过可解释的中间曝光堆栈与轻量级融合网络，兼具可解释性与高性能；④ 无需强制使用分类器无导向指导（CFG），保持输入图像与生成结果的像素级一致性。

**🔧 技术方法**

技术手段包括：Stable Video Diffusion（latent 视频扩散模型）+ VAE 编码/解码；对输入 LDR 图像进行无噪声的条件编码；微调扩散模型以生成从低到高曝光的 LDR 堆栈；使用 PU‑21 变换的感知空间对融合网络进行监督；轻量级 UNet 预测像素级权重完成曝光堆栈融合；在训练与推理中不使用 CFG；在部分实验中利用推理时优化实现均匀曝光间距。

**📊 数据集**

训练集为约 11K 张 raw‑HDR 图像（来源于 X2HDR 数据集），通过将 HDR 转换为多曝光 LDR 堆栈生成训练样本；评估使用 SI‑HDR 基准集；为验证框架通用性，还在 NYUv2 数据集上构造深度引导的焦点堆栈，用于 all‑in‑focus 图像恢复。

**📈 对比分析**

对比基线包括：生成式扩散方法（LEDiff、X2HDR、BracketDiff）和无生成的前馈方法（HDRCNN、SingleHDR）。在 Q* PSNR、MAE、LPIPS 上超过所有生成式基线，在 Q* SSIM 上保持竞争力；与前馈方法相比，性能相当或略高，同时避免了其在极端曝光区域的失真。用户研究显示 72% 的偏好率；推理时间约 15 s，显著快于 BracketDiff（≈6 min）且与其它基线相当。

**⚠️ 局限性**

局限性包括：① 由于 VAE 压缩，细节（如细小文字、高频纹理）在生成堆栈中可能被模糊；② 对极暗或极亮输入仍有挑战；③ 在已包含足够曝光信息的场景中，方法可能略逊于专门的前馈模型；④ 推理内存需求约 13.5 GB；⑤ 需要更多多样化数据与更强的生成模型以进一步提升鲁棒性。

---

## 371. From Generic Correlation to Input-Specific Credit in On-Policy Self Distillation

**arXiv ID:** 2605.11613 | [PDF](https://arxiv.org/pdf/2605.11613v1)

**作者:** Guobin Shen `[一作]` (Xiaohongshu Inc), Xing Yu `[通讯]` (Xiaohongshu Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种对自我蒸馏（On‑policy Self‑Distillation）奖励进行对比基线校正的技术

**💡 创新点**

通过理论证明自我蒸馏奖励等价于给定输入下响应与反馈的点对点互信息，并识别出输入通用（shortcut）成分；随后设计批量对比基线消除该成分，形成Contrastive REward from DIsTillation (CRD)

**🔧 技术方法**

基于贝叶斯滤波、控制即推理框架、对比学习与自我蒸馏的逆 KL 梯度

**📊 数据集**

在LiveCodeBench v6、SciKnowEval（Chemistry/Physics/Biology/Materials Science）和ToolAlpaca这三个结构化反馈任务上进行评估

**📈 对比分析**

与传统单一标量奖励（GRPO）以及标准自我蒸馏（SDPO）对比；CRD 在两大模型（Qwen3‑8B、OLMo‑3.7B‑Instruct）上在所有子任务上均取得最优或次优表现，整体平均得分最高，且训练收敛更快

**⚠️ 局限性**

仅在7–8B规模模型和结构化反馈场景下验证，假设批量多样性充足且对比负样本不与反馈不匹配；对多轮对话、动态 λ 及更大规模模型的适用性尚待进一步研究

---

## 372. When Emotion Becomes Trigger: Emotion-style dynamic Backdoor Attack Parasitising Large Language Models

**arXiv ID:** 2605.11612 | [PDF](https://arxiv.org/pdf/2605.11612v1)

**作者:** Ziyu Liu `[一作]` (Sichuan University), Junjiang He `[通讯]` (Sichuan University)

**通讯引用:** 313 | [OpenAlex ID](https://openalex.org/A5047196217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了利用情感风格作为触发器的动态后门攻击（Paraesthesia），通过在LLM微调数据中注入情绪化表达来植入后门。

**💡 创新点**

创新点在于将情感（情绪）本身视为独立、可动态调节的触发机制，突破传统固定词/结构触发的局限，提升隐蔽性与持久性。

**🔧 技术方法**

采用Russell Circumplex情感量化模型、情感重写引擎、语义保真度评分、QLoRA微调以及BERTScore/DeBERTa等评估技术。

**📊 数据集**

实验数据集包括Alpaca指令集与AG’s News分类数据集，评测模型涵盖 Llama 2、Vicuna、Mistral、Qwen2.5、GLM 4、Qwen3 等多种 LLM。

**📈 对比分析**

通过与 BadNets、CBA、Syn、VPI、Sleeper 等基线方法对比，Paraesthesia 的攻击成功率达 99–100%，清洁性能与基线持平，并在后续清洗微调后仍保持高 ASR。

**⚠️ 局限性**

局限性：仅在文本 LLM 上验证，未对视听多模态模型或超大参数模型进行测试；情感重写工具需进一步自动化与扩展。

---

## 373. SoK: Unlearnability and Unlearning for Model Dememorization

**arXiv ID:** 2605.11592 | [PDF](https://arxiv.org/pdf/2605.11592v1)

**作者:** Mengying Zhang `[一作]` (RMIT University), Minhui Xue `[通讯]` (CSIRO)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统化研究了模型去记忆的两大范式——无学习性（unlearnability）与机器去学习（unlearning），并在统一框架下分析其交互与浅层去记忆现象。

**💡 创新点**

提出将无学习与去学习统一的分类体系，开展大规模实验揭示两者互相影响及浅层去记忆普遍性，并给出基于权重扰动的理论深度去记忆上限与可传递性证明。

**🔧 技术方法**

采用噪声注入、梯度上升/下降、影响函数、噪声微调、特征分离与碰撞、数据增强、恢复攻击以及分布式无学习等多种去记忆与去学习技术。

**📊 数据集**

主要实验数据集包括 CIFAR-10、CIFAR-100、ImageNet 子集和 RegText 文本任务。

**📈 对比分析**

通过比较未防御、无学习后及去学习后模型在准确率、成员推断攻击等指标的恢复曲线，发现大多数无学习+去学习组合易受恢复攻击，仅 FT 在类级别去学习能保持较深的去记忆，但整体性能仍受限。

**⚠️ 局限性**

局限在于无学习和去学习的浅层去记忆、超参数敏感性和对恢复攻击的脆弱性；理论证明仍依赖高斯假设与凸性，尚未在大型非凸模型上完全验证。

---

## 374. Logit-Attention Divergence: Mitigating Position Bias in Multi-Image Retrieval via Attention-Guided Calibration

**arXiv ID:** 2605.11591 | [PDF](https://arxiv.org/pdf/2605.11591v1)

**作者:** Mingtao Xian `[一作]` (Shanghai Jiao Tong University), Nanyang Ye `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 600 | [OpenAlex ID](https://openalex.org/A5077493772)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究多图检索任务中的位置偏差问题，发现模型的最终logit严重受位置结构影响，而内部注意力却能正确聚焦目标图像，称之为Logit‑Attention Divergence；基于此提出一种无训练的注意力引导去偏框架，利用少量校准样本在推理时通过内部注意力动态校正位置偏差，恢复置换不变性并显著提升检索准确率。

**💡 创新点**

创新点在于首次揭示logit与内部注意力之间的偏差现象；提出利用条件位置偏差估计与内部注意力先验的结合，通过贝叶斯式动态加权实现实例级去偏；并实现了仅需5条校准样本即可完成去偏的训练-free方法，兼顾高效与可扩展性。

**🔧 技术方法**

技术上使用了多层Transformer的注意力提取、循环位移校准数据集、统计逆向估计条件偏差、温度缩放的注意力后处理、动态层选择以及基于注意力的先验校正与logit重校。

**📊 数据集**

主要数据集包括MS‑COCO的多图检索基准（随机与对抗两种候选采样），Flickr8k用于跨域校准测试，以及MMIU六个多图VQA任务（forensic_forgerynet、forensic_blink、emotion_expw、emotion_findingemo、text2image_retrieval、visual_quality）。

**📈 对比分析**

与Vanilla、Prompt、CoT、PriDe、Self‑Consistency等传统无训练方法对比，本文方法在随机设置下准确率提升超过40%，在对抗设置下仍保持显著优势；同时RStd下降、Consistency提升，跨域、跨难度转移实验显示校准后性能几乎不受数据域变化影响。

**⚠️ 局限性**

局限性主要体现在：对高度视觉模糊或极难区分的样本，内部注意力自身的可靠性受限，去偏效果受其影响；此外，虽然校准样本极少，但在极大候选集（N>12）时仍需额外的前向计算，导致推理延迟略有上升。

---

## 375. STA-FEM: Exact Streaming Assembly for Preplanned Dynamic Tetrahedral Topology Edits

**arXiv ID:** 2605.11673 | [PDF](https://arxiv.org/pdf/2605.11673v1)

**作者:** Manish Acharya `[一作]` (Vanderbilt University), David Hyde `[通讯]` (Vanderbilt University)

**通讯引用:** 1093 | [OpenAlex ID](https://openalex.org/A5040432863)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究了一种在预先规划的动态四面体网格模拟中，使用事件流增量维护稀疏矩阵的技术（STA‑FEM），从而在每帧保持与完整重建等价的精确组装。

**💡 创新点**

创新点在于将拓扑编辑视为流事件，仅更新受编辑影响的矩阵条目，避免了全局重建和局部重算，并通过边多重计数实现对图拉普拉斯和块弹性算子的精准维护。

**🔧 技术方法**

采用事件驱动增量更新、边多重计数、稀疏矩阵维护、线性弹性块装配、图拉普拉斯代理、共轭梯度求解以及可扩展的并行编辑处理技术。

**📊 数据集**

使用 Bunny（10k/50k/460k 版）、Gargoyle（50k）和 Hand（100k）四面体网格，分别在 fracture、refinement、merge 三种脚本化场景下进行实验。

**📈 对比分析**

将 STA‑FEM 与全重建（R）和局部重算（L）三种策略对比，衡量帧时间、更新时间、CG 迭代等指标。实验显示在所有模型和场景中 STA‑FEM 提升帧速率约 1.3–1.6×，更新时间从数百毫秒降至 <1 ms，最大可达 51× 的组装速度提升。

**⚠️ 局限性**

局限在于仅适用于已知候选元素集合的预规划场景；对大规模并行编辑与动态连接性的支持仍需进一步研究；并且依赖周期性重建的连接性处理可能不适用于需要实时完全动态连接的应用。

---

## 376. Explaining and Breaking the Safety-Helpfulness Ceiling via Preference Dimensional Expansion

**arXiv ID:** 2605.11679 | [PDF](https://arxiv.org/pdf/2605.11679v1)

**作者:** ShiYing Huang `[一作]` (Huazhong University of Science and Technology), Zhigang Zeng `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 27561 | [OpenAlex ID](https://openalex.org/A5081245089)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MORA（Multi‑Objective Reward Assimilation）方法，通过将单意图提示扩展为多意图提示，构造高维奖励数据，解决大语言模型在帮助性与安全性之间的零和冲突。

**💡 创新点**

核心创新在于发现提示的单维性是多目标冲突的根源，并通过自我对弈式数据合成与最大边缘选择，主动扩展奖励维度，打破固定的 Pareto 前沿。

**🔧 技术方法**

采用自对弈（self‑play）生成多意图提示、最大边缘（max‑margin）筛选、DPO（Direct Preference Optimization）配对，以及奖励模型评估与RLHF等技术。

**📊 数据集**

使用HelpSteer2、UltraFeedback、SafeRLHF‑10k、TruthfulQA 等公开数据集进行训练，并通过额外生成 5,856 条多意图样本来扩充训练集。

**📈 对比分析**

与 MODPO、SPO、MO‑ODPO、OrthAlign、RSDPO‑W 等基线相比，MORA 在顺序对齐中单目标提升 5%–12.4%（安全性尤显突出），在并行对齐中整体奖励平均提升 4.6%，实现安全率 96.5% 与帮助率 79.1%，在三目标（帮助、安全、真诚）对齐中各目标均有 7–11% 的提升。

**⚠️ 局限性**

目前仅在文本任务上验证，未探究多模态场景的适用性，且合成数据对不同模型与规模的泛化能力仍待进一步评估。

---

## 377. OOM-Free Alpamayo via CPU-GPU Memory Swapping for Vision-Language-Action Models

**arXiv ID:** 2605.11678 | [PDF](https://arxiv.org/pdf/2605.11678v1)

**作者:** Seungwoo Roh `[一作]` (Kookmin University), Jong-Chan Kim `[通讯]` (Kookmin University)

**通讯引用:** 1044 | [OpenAlex ID](https://openalex.org/A5101647596)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种系统级优化框架，利用需求层加载、双缓冲流水线预取与GPU驻留层决策，在仅16 GB显存的消费级GPU上实现大规模Vision‑Language‑Action模型（Alpamayo‑R1‑10B）的高效推理。

**💡 创新点**

1) 结合层级需求加载与双缓冲流水线，实现显存使用从模型级降至层级；2) 针对EXE‑intensive、DMA‑intensive及混合模块推导闭式驻留收益表达式；3) 提出交错驻留放置策略，单次剖析即可获得最优驻留配置；4) 构建误差≤1.3%的推理时间预测模型。

**🔧 技术方法**

需求层加载（Demand Layering）、双缓冲预取（Double Flat Buffer）、CUDA事件同步、页面锁定内存、PCIe Gen5/3传输、CUDA流重叠、驻留层决策政策。

**📊 数据集**

主要使用Alpamayo‑R1‑10B（21.52 GB BF16）作为测试模型，亦在RTX 3080 Ti上验证OpenVLA‑7B模型；测试集为10步扩散动作生成的30条驾驶轨迹。

**📈 对比分析**

通过与Hugging Face Accelerate offloading基线和理论最优预加载C^opt对比；在RTX 5070 Ti 16 GB上实现4.09 s推理，较Accelerate 14.52 s提升3.55×，接近理论下限2.86 s；在RTX 3080 Ti上提升1.7×，在OpenVLA‑7B上提升1.3×。

**⚠️ 局限性**

仍未达成实时闭环控制所需的100 ms以内，受限于模型规模与显卡算力；对PCIe传输带宽高度依赖；仅适用于单个大模型；系统层面需特定内核/驱动。

---

## 378. Weather-Robust Cross-View Geo-Localization via Prototype-Based Semantic Part Discovery

**arXiv ID:** 2605.11654 | [PDF](https://arxiv.org/pdf/2605.11654v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 379. Unlocking UML Class Diagram Understanding in Vision Language Models

**arXiv ID:** 2605.11634 | [PDF](https://arxiv.org/pdf/2605.11634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 380. A Proprioceptive-Only Benchmark for Quadruped State Estimation: ATE, RPE, and Runtime Trade-offs Between Filters and Smoothers

**arXiv ID:** 2605.11674 | [PDF](https://arxiv.org/pdf/2605.11674v1)

**作者:** Ylenia Nisticò `[一作]` (Istituto Italiano di Tecnologia), Claudio Semini `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种本体感知状态估计器（MUSE、IEKF、Invariant Smoother）在同一数据集上进行统一评估与比较

**💡 创新点**

通过提供统一评估框架与精细化指标，阐明了精度、实时性与计算负荷的权衡关系

**🔧 技术方法**

使用MUSE（观测-融合框架）、Invariant Extended Kalman Filter（IEKF）和固定窗口Invariant Smoother（IS）三种技术

**📊 数据集**

Grand Tour Dataset中的CYN-1（Grindelwald Canyon）序列

**📈 对比分析**

采用ATE、RPE（空间/时域）、速度RMSE和每次更新计算时间进行对比；IS在ATE和RPE上最优，IEKF在速度上最快，MUSE则最快但精度稍逊

**⚠️ 局限性**

实验仅在单一机器人与单一环境下进行，未评估闭环控制性能或在不同地形/动态条件下的鲁棒性

---

## 381. Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling

**arXiv ID:** 2605.11666 | [PDF](https://arxiv.org/pdf/2605.11666v1)

**作者:** Liqin Ye `[一作]` (Georgia Institute of Technology), Chao Zhang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 10317 | [OpenAlex ID](https://openalex.org/A5100460272)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Evolutionary Task Discovery (EvoTD) 框架，通过进化搜索在算法技能与复杂度属性两维空间中生成多样且具挑战性的可执行推理任务，提升 LLM 的推理能力。

**💡 创新点**

创新点：① 将任务空间分解为算法技能（语义轴）与复杂度属性（结构轴），实现逻辑与规模的独立控制；② 设计两种进化算子—复杂度变异 (M_attr) 与技能交叉 (X_skill)，系统探索任务边界；③ 引入 Zone of Proximal Development (ZPD) 滤波器与可执行、技能匹配、可学习三阶段验证，形成动态、可解释的自适应课程。

**🔧 技术方法**

技术手段：进化算子 (mutation, crossover)、LLM 驱动的任务生成与元认知审计、可执行代码验证、技能对齐检测、学习难度评估、动态 ZPD 筛选、基于 RL 的强化学习（VeRL）训练。

**📊 数据集**

使用了多种数据集进行评估：代码生成（MBPP+, LiveCodeBench v6）、数学推理（AIME 24/25、OlympiadBench）、通用推理（MMLU-Pro、SuperGPQA），以及从种子语料中自动抽取的技能与复杂度属性库。

**📈 对比分析**

与传统的 Evol-Instruct、SPIRAL、Agent0 等方法对比，EvoTD 在 5 个基准上均表现出显著提升，尤其在 AIME、LiveCodeBench 等困难任务上提升 8%+；在不同架构（Qwen3、LLaMA）、预训练模式（Base、Instruct）与参数规模（3–8B）上均保持最优，说明其良好的迁移与可扩展性。

**⚠️ 局限性**

局限性：① 依赖高质量的 LLM 生成器与元认知评估，生成成本和 API 调用费用较高；② 任务空间主要局限于可执行编程任务，尚未扩展至非代码推理场景；③ 对复杂度属性的定义与选择仍需手工或半自动化设定，可能影响覆盖全面性。

---

## 382. Human-Grounded Multimodal Benchmark with 900K-Scale Aggregated Student Response Distributions from Japan's National Assessment of Academic Ability

**arXiv ID:** 2605.11663 | [PDF](https://arxiv.org/pdf/2605.11663v1)

**作者:** Kyosuke Takami `[一作]` (Osaka Kyoiku University), Yusuke Miyao `[通讯]` (University of Tokyo)

**通讯引用:** 4681 | [OpenAlex ID](https://openalex.org/A5004444958)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于日本全国学力测验的多模态评测基准，保留原始试卷布局、图表和学生答题分布，提供可直接对比人类与LLM表现的真实数据；

**💡 创新点**

首次将真实考试材料与大规模学生答题分布相结合，形成统一的多模态评测框架，并引入LLM-as-judge自动评分与人工评估并行验证；

**🔧 技术方法**

采用PDF解析→Markdown→图像提取→JSON模式转换的流水线；使用GPT‑5、GPT‑4o、Gemini、Claude等多模态LLM进行评测，并用精确匹配与字符级F1、人工评估和LLM-as-judge来衡量开放式答案；

**📊 数据集**

2022年九年级全国学力测验的科学、数学与日语题目及约900万学生的累计答题分布；

**📈 对比分析**

对多模态LLM进行多选题精确匹配准确率和开放题字符级F1评估，最优多选准确率约0.8，开放题F1约0.5，表现因学科而异，图像信息对科学题显著提升；人工评估与LLM-as-judge在科学和数学上相对一致，但在日语上差异较大；

**⚠️ 局限性**

仅覆盖单一年份、仅限日语教育体系、使用聚合答题数据缺乏个体过程信息、图像预处理可能丢失细节、开放式评测指标有限、潜在的课程偏差与样本偏倚需谨慎解释；

---

## 383. Can LLM Agents Respond to Disasters? Benchmarking Heterogeneous Geospatial Reasoning in Emergency Operations

**arXiv ID:** 2605.11633 | [PDF](https://arxiv.org/pdf/2605.11633v1)

**作者:** Junjue Wang `[一作]` (University of Tokyo), Naoto Yokoya `[通讯]` (University of Tokyo)

**通讯引用:** 15137 | [OpenAlex ID](https://openalex.org/A5034435383)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了灾难运营响应代理基准（DORA），包含 515 个专家设计的任务、108 个专门化工具以及针对多传感器地理空间数据的五维分析流程，并对 13 种前沿 LLM 代理在该基准上的表现进行了系统评估。

**💡 创新点**

创新点包括：
1) 第一次为灾难运营响应设计完整的代理基准，覆盖感知、空间关系、操作规划、时间演化与多模态报告合成等全流程；
2) 构建了专门的 108 工具库，覆盖感知、栅格处理、矢量空间算子、控制流、可视化与报告生成；
3) 通过专家手工注释与确定性重放生成黄金轨迹，保证任务与工具链的语义正确性；
4) 在多模态、多传感器与灾难特定语义的交叉难点上系统揭示 LLM 代理的弱点。

**🔧 技术方法**

使用技术包括：
- LLM 代理框架（ReAct、Plan‑then‑Execute、Reflexion、ReWOO）；
- MCP（Multi‑Component Protocol）工具服务器，统一 JSON‑RPC 接口；
- 轨迹评估指标（Tool‑Any‑Order、Tool‑In‑Order、Tool‑Exact‑Match、Parameter Accuracy）与最终答案评估；
- 通过可视化工具（Matplotlib、Shapely、NetworkX）生成地图与报告。

**📊 数据集**

数据集来源于 45 起真实灾难事件，涵盖 10 类灾害，主要包括：
- xBD、DisasterM3、BRIGHT（光学、SAR、复合影像）
- NAIP、Maxar（多时相洪水/重建序列）
- Landslide4Sense、GVLM‑CD（多光谱+DEM）
- Planet、RescueNet、CRASAR‑U‑DRoIDS（城市/空中灾害场景）
- OpenStreetMap、Our World in Data（道路、POI、人口、设施矢量）等。

**📈 对比分析**

比较方法：在所有模型上执行完整的代理循环，记录工具调用轨迹与最终答案；将结果与黄金轨迹进行对比；并在工具使用、参数正确性、效率等维度给出分数。性能方面：最强模型 Gemini‑3.0‑Flash 仅达 53.7% 的平均最终答案准确率；工具无用基线（Qwen3‑VL‑235B、Gemini‑3.0‑Flash）在视觉推理上表现极差；各类工具使用率与参数准确率均低于 40%。

**⚠️ 局限性**

局限性：
- 在灾难语义与感知标签上仍缺乏深度 grounding，导致损伤语义与传感器模态误配；
- 需要同时选取正确工具并给出精确参数，错误往往在链条前期放大，导致长轨迹性能急剧下降；
- 对多模态融合（SAR/光学、DEM/坡度等）的推理仍表现不佳；
- 基准仅覆盖 45 个灾难事件，可能在更广泛的灾难场景中泛化受限；
- 评估侧重最终答案准确率，未完全捕捉实时性与鲁棒性需求。

---

## 384. RNA-FM: Flow-Matching Generative Model for Genome-wide RNA-Seq Prediction

**arXiv ID:** 2605.11622 | [PDF](https://arxiv.org/pdf/2605.11622v1)

**作者:** Yaxuan Song `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**通讯引用:** 12046 | [OpenAlex ID](https://openalex.org/A5076697411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了 RNA-FM，一种基于流匹配的生成框架，用以从组织学全切片图像预测全基因组 bulk RNA‑seq 表达谱。

**💡 创新点**

创新点在于：① 将基因表达建模为条件连续时间传输问题，利用流匹配学习速度场；② 通过 GO 通路层级对 >20,000 基因进行分组，实现可扩展、可解释的通路级表示；③ 引入通路图注意力、CFG 指导以及多实例学习，提升预测精度与不确定性估计。

**🔧 技术方法**

核心技术包括：流匹配生成模型、路径级 MLP 及图注意力网络、ODE 传输、分类器无关引导（CFG）、多实例学习（MIL）图像特征提取（ResNet‑50/UNI）。

**📊 数据集**

实验使用 TCGA 的 LUAD、BRCA、COAD 组织切片与 RNA‑seq 数据进行训练与交叉验证，并在 CPTAC 的同样三种癌症类型做外部验证。

**📈 对比分析**

与 HE2RNA、tRNAsformer、SEQUOIA 等基线比较，RNA‑FM 在 PCC（相关系数）上显著提升、RMSE（均方根误差）显著降低，尤其在前 1,000/500/100/50 预测基因子集上表现最为突出；通路层级预测同样优于基线；不确定性评估显示预测误差与方差正相关，表明模型具有可靠的不确定性估计。

**⚠️ 局限性**

主要局限在于仅覆盖三种器官与有限样本；对更多器官、疾病类型的泛化尚待验证；未来需扩展到单细胞或空间测序等其他转录组模态。

---

## 385. PhishSigma++: Malicious Email Detection with Typed Entity Relations

**arXiv ID:** 2605.11619 | [PDF](https://arxiv.org/pdf/2605.11619v1)

**作者:** Shang Shang `[一作]` (Chinese Academy of Sciences), Zhengwei Jiang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 802 | [OpenAlex ID](https://openalex.org/A5020151253)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于类型实体关系图和粒子群优化（PSO）的恶意邮件检测框架 PhishSigma++，能够在单条 RFC822 邮件上直接生成判定结果及可审计的关系证据。

**💡 创新点**

创新点在于：①将手工 Sigma 规则的跨字段一致性转化为可学习的实体关系图；②使用 PSO 在稀疏的关系空间中搜索最佳掩码，实现数据驱动的关系特征选择；③保持检测过程的可解释性，输出与 Sigma 规则兼容的“关系词条”而非仅靠词袋特征。

**🔧 技术方法**

技术细节包括：①基于确定性提取器抽取结构、内容、关键词三类实体；②计算五种相似度函数并取最大得到单一边权；③构造 N×N 的关系矩阵，展开为 d 维稀疏向量；④PSO 搜索二值掩码，结合 F1 与稀疏惩罚；⑤最后用线性 SVM 进行分类并产生阈值化的 Sigma 兼容输出。

**📊 数据集**

使用的数据集为 约 12,000 条 RFC822 邮件，来源于 Enron、Phishing Pot、Nazario 与 SpamAssassin，正负样本各 6,000 条，且包含多种欺骗手法（BEC、伪造品牌、链接伪装等）。

**📈 对比分析**

与规则基、token 基、BERT、DistilBERT 等基线相比，PhishSigma++ 在干净数据上 F1 0.968（仅比 SpamBayes 高 0.009），在 Good Word 文字填充攻击（ρ=0.8）时仍保持 0.958，远优于 token 基模型（降至 0.73）和传统规则（0.53）。

**⚠️ 局限性**

局限性包括：①仅针对非自适应的文字填充攻击，未覆盖自适应 URL 隐蔽、头部伪造等；②最大化压缩（max‑collapse）丢失了具体相似度通道信息，限制了可解释深度；③PSO 仅为启发式搜索，可能不达最优；④交叉验证可能因来源与标签相关而过估；⑤不涉及大规模预训练文本编码器或多事件关联分析。

---

## 386. Keep What Audio Cannot Say: Context-Preserving Token Pruning for Omni-LLMs

**arXiv ID:** 2605.11605 | [PDF](https://arxiv.org/pdf/2605.11605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 387. Sampling-Based Follow-the-Leader Motion Planning for Manipulator-Mounted Continuum Robots

**arXiv ID:** 2605.11618 | [PDF](https://arxiv.org/pdf/2605.11618v1)

**作者:** Chengnan Shentu `[一作]` (University of Toronto), Jessica Burgner-Kahrs `[通讯]` (University of Toronto)

**通讯引用:** 4927 | [OpenAlex ID](https://openalex.org/A5087473881)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对全6-DOF基座姿态的连续机器人（CR）进行Follow-the-Leader（FTL）运动规划，提出分离全局形状搜索与局部插值的采样式框架，实现离线预计算形状库、闭式几何基座对齐、并在在线阶段实现零误差尖端跟踪。

**💡 创新点**

创新点：①将基座姿态与形状分离，使用闭式几何构造一次性确定基座姿态，消除在线优化；②支持任意前向模型（PCC、Cosserat、FEM等）；③提供解析解析性保证（解析完整性、尖端跟踪收敛）；④引入阈值聚类可调节计算与精度的权衡。

**🔧 技术方法**

技术：采样式形状库生成、闭式三步基座对齐（平移+两次旋转）、对齐后使用对称Chamfer距离评估形状、径向对称预对齐、线性插值+SLERP+基座构造实现离散点间的尖端跟踪。

**📊 数据集**

数据集：120条测试路径（3类：C曲线、S曲线、Robot曲线），每类40条，包含10个waypoint、10个插值步；使用PCC模型、伪刚体模型（MuJoCo）和真实硬件（3段肌腱驱动CR+7-DOF Franka机械臂）进行验证。

**📈 对比分析**

比较方法：与基于PCC的L-BFGS-B优化基准、与聚类采样（线性搜索 vs. 2√N_lib聚类）进行对比。性能：尖端误差始终为0%，形状偏差在PCC下线性搜索为0.0%，聚类版略高但仍低于优化基准；计算时间在聚类版显著降低（≈10×），线性搜索较慢。

**⚠️ 局限性**

限制：①假设前向模型与基座姿态独立，未考虑重力或外部负载；②仅对每个waypoint局部搜索，缺乏全局轨迹形状优化；③形状库采样为均匀，未针对形状空间进行重要性采样；④硬件执行误差主要来自模型不匹配，需闭环控制或更精准模型。

---

## 388. CuSearch: Curriculum Rollout Sampling via Search Depth for Agentic RAG

**arXiv ID:** 2605.11611 | [PDF](https://arxiv.org/pdf/2605.11611v1)

**作者:** Jianghan Shen `[一作]` (Nanjing University), Junjun He `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 14028 | [OpenAlex ID](https://openalex.org/A5003413234)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对基于 RLVR 的 agentic RAG 训练，提出 CuSearch 框架，通过搜索深度优先分配来聚焦更深检索轨迹，从而提升检索子策略的学习效果。

**💡 创新点**

创新点在于：① 将无注释的搜索深度作为检索监督稠密度的代理；② 设计了 SDGA（Search‑Depth Greedy Allocation）算子，推出 SDGA‑Auto 与 SDGA‑Phase 两种自适应预算分配策略；③ 通过动态聚焦更深轨迹实现隐式/显式的训练进度课程。

**🔧 技术方法**

采用 GRPO 优化框架，结合 SDGA 算子、搜索深度贪婪分配、优势归一化、奖励 KL 正则等技术；实现了在现有 RLVR 训练流程中的轻量级轨迹选择插入。

**📊 数据集**

使用七大开放域 QA 基准（NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle）进行评估，并在两种检索环境（ZeroSearch 的 LLM‑simulated 检索与 Search‑R1 的离线检索）下测试。

**📈 对比分析**

与 GRPO‑Full、GRPO‑Half、随机挑选、Top‑K Reward、DAPO、AR3PO 等方法对比，SDGA‑Phase 在 ZeroSearch 下平均提升 EM 约 11.8 分（从 44.5 到 56.3），在 Search‑R1 下提升约 5 分；显著优于提示级自适应方法。

**⚠️ 局限性**

限制在于：当搜索深度增长有限或检索环境随机性低时，收益减小；仅依赖单一搜索深度特征，未结合奖励信息；在极端情况下可能导致对低奖励轨迹的误偏差。

---

## 389. Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information

**arXiv ID:** 2605.11609 | [PDF](https://arxiv.org/pdf/2605.11609v1)

**作者:** Guobin Shen `[一作]` (Xiaohongshu Inc), Xing Yu `[通讯]` (Xiaohongshu Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估反向自蒸馏（AntiSD）方法，用以提升大语言模型在数学推理任务中的准确率和训练效率。

**💡 创新点**

创新点在于将默认自蒸馏的 per‑token 信号识别为条件 PMI，发现其偏向“捷径”token 并抑制推理 token；通过反转梯度方向、采用 Jensen–Shannon 上升与熵触发门，实现对梯度的自然界定和自适应关闭，显著提升推理质量。

**🔧 技术方法**

采用 On‑policy Self‑Distillation、GRPO、Jensen–Shannon Divergence、熵触发门（entropy‑triggered gate）等技术。

**📊 数据集**

训练使用 DAPO‑Math‑17k，评估使用 AIME 2024/25/26、HMMT 2025、MinervaMath 等数学推理数据集；还在 HumanEval+ 与 MBPP+ 上测试代码推理能力。

**📈 对比分析**

与 GRPO 基线以及默认自蒸馏对比，AntiSD 在 5 个 4B–30B 模型上在 2–10 倍更少的训练步骤即可达到 GRPO 的最高准确率，最终准确率提升 2–11.5 分（平均 7–8 分），且在大多数基准上均名列前茅。

**⚠️ 局限性**

局限包括：门控阈值需要手动校准（尽管可自动标定但仍需经验值）、对非数学推理任务（如复杂多回合对话）的适用性尚未完全验证、极大规模模型在训练稳定性方面可能出现熵崩溃问题。

---

## 390. One-Step Generative Modeling via Wasserstein Gradient Flows

**arXiv ID:** 2605.11755 | [PDF](https://arxiv.org/pdf/2605.11755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 391. GAR: Carbon-Aware Routing for LLM Inference via Constrained Optimization

**arXiv ID:** 2605.11603 | [PDF](https://arxiv.org/pdf/2605.11603v1)

**作者:** Disha Sheshanarayana `[一作]` (Manipal University Jaipur), Tirthankar Dasgupta `[通讯]` (TCS Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Green-Aware Routing（GAR）框架，针对大型语言模型推理实现按请求碳排放最小化并满足准确率与延迟SLO的路由。

**💡 创新点**

创新点在于将碳排放作为首要目标，将其与准确率与尾部延迟约束统一为受限多目标优化，并通过在线原始-对偶算法实现实时决策。

**🔧 技术方法**

采用轻量级预测器（准确率、尾延迟、碳估计）结合时间变化的电网碳强度，和 GAR-PD 等在线原始-对偶路由算法。

**📊 数据集**

使用六个公开 NLP 基准（MMLU、HellaSwag、GSM8K、WinoGrande、SQuAD、ARC）以及包含 7B–70B 参数的五种 LLM 进行评估。

**📈 对比分析**

与基准路由（最小/最大模型、Accuracy‑Max、Oracle‑Feasible）对比，GAR‑PD 在宏观准确率 0.737、碳排放 0.712 g/请求（比最大模型低 74%）且满足 95% 分位延迟要求，显示出优越的准确率-碳折衷。

**⚠️ 局限性**

局限在于依赖公开的电网碳强度数据（可能滞后），未考虑财务成本，且对快速可再生能源波动的即时响应能力有限。

---

## 392. DiffScore: Text Evaluation Beyond Autoregressive Likelihood

**arXiv ID:** 2605.11601 | [PDF](https://arxiv.org/pdf/2605.11601v1)

**作者:** Wen Lai `[一作]` (Ant Group), Alexander Fraser `[通讯]` (Technical University Of Munich)

**通讯引用:** 2309 | [OpenAlex ID](https://openalex.org/A5101957153)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 DiffScore，一种基于掩码重建的 NLG 评价框架，利用 MDLLM 的 ELBO 进行文本质量评估。

**💡 创新点**

创新点在于：① 用掩码重建消除自回归模型的位置信息偏差；② 通过多时序质量曲线和双向 PMI 分解提供多维诊断；③ 通过全双向上下文实现方向一致性。

**🔧 技术方法**

技术核心为 Masked Diffusion Language Models（MDLLM）与 ELBO 估计，采用 Monte Carlo 抽样获得多时序得分；同时对比传统 AR 评估（BARTScore、GPTScore）和基于相似度的评估（BLEU、ROUGE、BERTScore）。

**📊 数据集**

在十个评估基准上进行实验，包括 SummEval、Newsroom、REALSumm、WMT19（多语言 MT）、BAGEL、SFHOT、SFRES 等，涵盖摘要、翻译、数据到文本三大任务。

**📈 对比分析**

与 BLEU/ROUGE、BERTScore、BARTScore、GPTScore、G‑Eval 等基线比较时，DiffScore‑FT 在大多数任务上均显著高于 BARTScore，零样本 DiffScore 超过 GPTScore，并能与 G‑Eval 的表现相匹配，且无需 API 调用。

**⚠️ 局限性**

局限性包括：① 需要大量 Monte Carlo 前向传播导致计算成本高；② 对 MDLLM 的预训练规模和指令微调敏感，小模型表现不佳；③ 当前实现仍不够高效，未来需探索更轻量级的推理或稀疏化方法。

---

## 393. PointForward: Feedforward Driving Reconstruction through Point-Aligned Representations

**arXiv ID:** 2605.11594 | [PDF](https://arxiv.org/pdf/2605.11594v1)

**作者:** Cheng Chi `[一作]` (Xiaomi EV), Haiyang Sun `[通讯]` (Xiaomi EV)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6514db3d-8de6-452c-91b7-acdb31787cc4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 PointForward，基于点对齐表示的前馈驾驶场景重建框架。

**💡 创新点**

创新点在于使用稀疏 3D 查询在世界坐标初始化、空间-时间融合实现跨视角一致性，并通过场景图进行实例级动态建模。

**🔧 技术方法**

技术包括 3D 高斯散射（3DGS）、EfficientNetV2 特征提取、InfiniDepth 深度估计、可学习的 3D 查询、空间-时间融合模块、场景图结构与规范化空间变换、以及轻量化 UNet 渲染。

**📊 数据集**

实验使用 Waymo Open Dataset 及 nuScenes 两大驾驶数据集。

**📈 对比分析**

与 STORM、DGGT 等现有前馈方法对比，在 Waymo 上全图 PSNR 28.48/SSIM 0.861、动态区 PSNR 25.01/SSIM 0.768，零样本 nuScenes PSNR 26.54/SSIM 0.821，均显著优于对比模型。

**⚠️ 局限性**

局限性在于动态建模主要适用于基于 3D 边界框的刚体实例，对高度非刚性运动（如行人）效果不佳。

---

## 394. Learning Feature Encoder with Synthetic Anomalies for Weakly Supervised Graph Anomaly Detection

**arXiv ID:** 2605.11749 | [PDF](https://arxiv.org/pdf/2605.11749v1)

**作者:** Yingjie Zhou `[一作]` (Sichuan University), Lingqiao Liu `[通讯]` (University of Adelaide)

**通讯引用:** 9876 | [OpenAlex ID](https://openalex.org/A5070976480)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种弱监督图异常检测方法，利用合成异常和多任务学习来学习对扰动敏感的特征表示，再通过弱监督训练实现节点异常检测。

**💡 创新点**

创新点在于①引入五类合成异常作为辅助任务，专门驱动特征编码器捕获多种异常扰动；②采用两阶段训练（预热+完整训练）以避免少量真实异常导致过拟合；③在合成异常上使用专用检测头而非单一异常类，提升特征判别能力。

**🔧 技术方法**

技术主要包括图神经网络（GATSep）作为特征编码器；多任务学习框架配合五个专用检测头；合成异常生成策略（基于度数、异质连接、结构重组、特征替换、特征扰动）；弱监督损失（real与synthetic损失相加）以及两阶段训练策略。

**📊 数据集**

实验数据集为Amazon、Yelp、Weibo、Questions和T‑Finance共五个真实图数据集，均含真实异常标签。

**📈 对比分析**

与传统弱监督方法（PC‑GNN、H2‑FDetector、BWGNN、GHRN、ConsisGAD、CARE‑GNN、GATSep）以及基线（MLP、GCN、GraphSAGE、GAT、GIN）对比，本文方法在AUROC和AUPRC上均取得最高或接近最高的排名，尤其在异常样本极少（≤30）时显著优于其他方法。

**⚠️ 局限性**

局限性主要体现在：①仅针对静态图，无法处理动态图的异常演化；②合成异常与真实异常的分布差异可能导致部分异常模式未被充分覆盖；③两阶段训练增加训练时间，且需手动调节λ等超参数。

---

## 395. Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation

**arXiv ID:** 2605.11739 | [PDF](https://arxiv.org/pdf/2605.11739v1)

**作者:** Yuchen Cai `[一作]` (USTC), Junfeng Fang `[通讯]` (NUS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的后训练蒸馏（OPD）进行参数级分析，发现其高效源于模块层面的冗余避免和更新方向层面的早期低秩锁定，并基于此提出 EffOPD 通过沿稳定方向外推实现加速。

**💡 创新点**

首次将 OPD 的效率归因于“前瞻性”机制：在训练早期就锁定高效模块与主导方向，从而实现更紧凑的参数更新；并提出一种无须额外模块、可插拔的外推加速方法。

**🔧 技术方法**

对 OPD 与 RL 进行谱分解、子空间对齐、截断实验；利用逐步指数外推与轻量验证筛选 Extrapolation 步长；对比不同模型规模下的训练进度与性能。

**📊 数据集**

使用 Codeforces、Taco、AIME24/25/26、MINERVA、GPQA 等编程与数学推理数据集；教师模型为 RL 微调的 1.5B–32B Qwen 系列模型。

**📈 对比分析**

与 Vanilla OPD、AlphaOPD、ExOPD 等方法对比，EffOPD 在所有规模上平均提升 3 倍训练速度，且在数学推理任务上实现更高的最终性能；验证集轻量化提升稳定性。

**⚠️ 局限性**

仅在 OPD 训练环境下验证，可能对教师质量和任务类型敏感；外推过程中仍需轻量验证，增加少量计算开销；对超大模型或其他结构的通用性尚待进一步测试。

---

## 396. Approximate Strategyproofness in Approval-based Budget Division

**arXiv ID:** 2605.11736 | [PDF](https://arxiv.org/pdf/2605.11736v1)

**作者:** Haris Aziz `[一作]` (UNSW Sydney), Jeremy Vollen `[通讯]` (Northwestern University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在批准型预算分配中，规则的近似策略不变性，通过激励比率衡量可操纵性，并证明Nash产品规则在保持效率与公平的同时激励比率为2，且此值最优。

**💡 创新点**

提出激励比率概念用于公共物品分配；证明Nash规则的激励比率为2；证明在公平与效率约束下，激励比率最优为2；给出多种规则激励比率上界下界。

**🔧 技术方法**

理论分析：激励比率、最优性证明、对凹性效用推广；实验分析：采样批准配置，计算最大激励比率与可操纵性。

**📊 数据集**

无外部数据集；使用人工生成的批准配置（IC、欧几里得、Mallows）。

**📈 对比分析**

比较方法：对四个规则（Nash、egalitarian、maximum payment、fair utilitarian）计算可操纵性频率与平均激励比率；结果显示Nash激励比率低，其他规则高；实验表明规则均易被操纵。

**⚠️ 局限性**

局限：只考虑批准型预算分配，激励比率仅为理论上最坏情况，实验仅在小规模（n≤100，m=10）随机模型下；未探讨策略不变性与效率公平的更细粒度权衡。

---

## 397. U-STS-LLM A Unified Spatio-Temporal Steered Large Language Model for Traffic Prediction and Imputation

**arXiv ID:** 2605.11735 | [PDF](https://arxiv.org/pdf/2605.11735v1)

**作者:** Yichen Zhang `[一作]` (Southeast University), Jun Li `[通讯]` (Southeast University)

**通讯引用:** 55467 | [OpenAlex ID](https://openalex.org/A5021388534)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一的基于大型语言模型的时空驱动框架 U-STS-LLM，用于移动通信流量的长周期预测和高缺失率补全。

**💡 创新点**

创新点在于引入动态时空注意力偏置生成器、部分冻结+LoRA 微调、门控自适应融合以及统一多任务目标，实现对时空关系的显式引导与参数高效适配。

**🔧 技术方法**

技术手段包括预训练 GPT‑2 作为主干，结合 LoRA 参数高效微调、动态图注意力偏置、时空嵌入、GRU 辅助引导和多任务损失。

**📊 数据集**

使用米兰（Milan）通信数据集（10 min 10k 网格）训练，采用 Trento 区域数据进行零样本迁移验证。

**📈 对比分析**

与 LSTM、ConvLSTM、DLinear、TimesNet、STGCN、GATGPT、GCNGPT、ST‑LLM 等 10+ 基线对比，U-STS‑LLM 在长周期预测 MAE/RMSE 平均提升约 24%，在 70–80% 缺失率补全提升约 12%，且在零样本迁移任务中亦显著优于所有对手。

**⚠️ 局限性**

模型仍依赖预先计算的功能图结构，适配不同规模或完全不同网络拓扑时需重新构建；此外，LLM 主干的计算与存储开销较大，对资源受限环境有限制。

---

## 398. CaC: Advancing Video Reward Models via Hierarchical Spatiotemporal Concentrating

**arXiv ID:** 2605.11723 | [PDF](https://arxiv.org/pdf/2605.11723v1)

**作者:** Jiyuan Wang `[一作]` (Beijing Jiaotong University), Tingting Gao `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种双阶段聚焦的奖励模型 CaC，用于检测和定位生成视频中的稀疏细微异常

**💡 创新点**

引入两轮时空聚焦、三阶段递进训练与 IoU 奖励，显著提升对稀疏异常的感知

**🔧 技术方法**

基于 Vision‑Language 模型的链式推理、两轮 Group Relative Policy Optimization (GRPO)、Temporal/Spatial IoU 奖励

**📊 数据集**

构建了首个含帧级边框、时段与属性的生成视频异常数据集（约 3 万条）

**📈 对比分析**

与 Gemini、GPT‑4o、UnifiedReward 等基线对比，在 CaC‑Bench‑Main 上精确率 0.767、召回 0.910，整体准确率 81.7%；在生成视频对齐实验中降低异常率 11.7%

**⚠️ 局限性**

两轮推理导致推理时间加倍；对极端短时空异常仍有一定误检；依赖大量人工标注

---

## 399. SafeSteer: A Decoding-level Defense Mechanism for Multimodal Large Language Models

**arXiv ID:** 2605.11716 | [PDF](https://arxiv.org/pdf/2605.11716v1)

**作者:** Xinyi Zeng `[一作]` (Tsinghua University), Yu Tian `[通讯]` (Tsinghua University)

**通讯引用:** 31128 | [OpenAlex ID](https://openalex.org/A5047476496)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SafeSteer，一种在 MLLM 解码阶段通过轻量级解码探测器和模态语义对齐向量实时纠正有害输出的防御框架。

**💡 创新点**

创新点在于利用 MLLM 自身的内在安全判别能力，通过解码探测器对候选词进行重排序，以及将文本安全对齐迁移到视觉模态的 MSAV，从而实现无需微调、即时的安全改进。

**🔧 技术方法**

技术手段包括：PCA+t‑SNE 可视化分析隐藏状态、线性解码探测器（logistic 回归）计算有害分数、基于 MSAV 的向量注入、Top‑k 重采样、温度 1.0 的生成策略。

**📊 数据集**

使用的数据集包括 MM‑Vet、MM‑SafetyBench、FigStep、VL‑Guard、以及三大 MLLM（Qwen2.5‑VL、Qwen3‑VL、LLaVA‑1.5‑7B）的标准评测集。

**📈 对比分析**

与 ECSO、MLLM‑Protector、MRD 等基线对比，SafeSteer 在 13 个安全场景中平均降低攻击成功率（ASR）33.4%，且保持或提升拒绝率（RR）与准确率（Acc）的平衡；计算开销极低，仅需一次线性探测器训练，推理时间与原始模型相近。

**⚠️ 局限性**

局限性：MSAV 可能在某些场景下降低整体鲁棒性；解码探测器的逐步修正需要 Top‑k 中存在安全词，若无则改正过程会变慢且效果受限。

---

## 400. Deanonymizable Scoped Linkable Ring Signatures

**arXiv ID:** 2605.11715 | [PDF](https://arxiv.org/pdf/2605.11715v1)

**作者:** Montassar Naghmouchi `[一作]` (Institut Polytechnique de Paris), Maryline Laurent `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 2946 | [OpenAlex ID](https://openalex.org/A5044148377)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种新的门限可去匿名化可链接环签名（DSLRS），用于在医疗同意管理等需要匿名与可追溯并存的场景中实现可追踪、可跨域解密的签名。

**💡 创新点**

创新点：① 在签名中嵌入ElGamal加密的可去匿名化组件，省去单独承诺；② 通过动态密钥图像实现范围级链接；③ 采用分布式密钥生成的门限网络实现去中心化可去匿名化；④ 在随机Oracle模型下对完整安全性进行形式化证明。

**🔧 技术方法**

技术：环签名、可链接环签名、ElGamal加密、Shamir门限共享、分布式密钥生成（DKG）、哈希到曲线、椭圆曲线离散对数与DDH假设、随机Oracle模型、Hyperledger Fabric区块链、Schnorr式零知识证明。

**📊 数据集**

未使用公开数据集；在区块链实例中使用Hyperledger Fabric的实验环境进行性能评估。

**📈 对比分析**

与RS、LSAG、MLSAG、ARS、TRS、SARS等现有方案对比，DSLRS在签名大小上为97n+196字节（n为环大小），与这些方案保持O(n)的时间复杂度；相比SARS的O(log₂n)时间复杂度，DSLRS在可追溯性与去中心化去匿名化方面更具优势，实验表明在区块链部署后签名验证和去匿名化的延迟均在毫秒级。

**⚠️ 局限性**

局限性：签名尺寸相对较大；验证与链接仍为O(n)复杂度，对大规模环可能导致效率下降；去匿名化需要k个节点协作，若节点同步不佳会影响实时性；实现依赖区块链基础设施，部署成本与安全管理需要进一步评估。

---

## 401. Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance

**arXiv ID:** 2605.11712 | [PDF](https://arxiv.org/pdf/2605.11712v1)

**作者:** Wenhao Chen `[一作]` (Peking University), Guojie Song `[通讯]` (Peking University)

**通讯引用:** 6130 | [OpenAlex ID](https://openalex.org/A5088976879)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Stable Value Guidance Transformer（SVGT），通过在冻结的语言模型后面添加独立的价值模块，实时提供稳定的价值引导，提升对有害输出的抑制。

**💡 创新点**

创新点在于将价值建模从主模型中分离，构建专用的价值空间并利用桥接词（Bridge Tokens）在解码时动态对齐生成轨迹，既保持了价值的稳定性，又不破坏主模型内部表示。

**🔧 技术方法**

采用了价值空间构建（双路径编码与判别器）、潜在价值桥接（Late‑Binding桥接词生成）、多阶段课程训练以及密集安全监督等技术，形成了一个完整的价值引导体系。

**📊 数据集**

主要使用 WildGuardMix、BeaverTails、安全评测数据集（HarmBench）以及通用生成数据集 WikiText-2 进行评估，验证模型的价值辨识与生成质量。

**📈 对比分析**

与系统提示、DPO、ITI、RE‑Control 等传统对齐方法相比，SVGT 在 WildGuardMix/BeaverTails 上的有害分数下降 70% 以上，HarmBench 的攻击成功率下降 72–80%，拒绝率提升至 75% 以上，同时保持与基线相当甚至更好的困惑度（PPL）表现。

**⚠️ 局限性**

局限性包括对价值范畴的覆盖有限、在大规模模型上仍有 50–70% 的推理延迟、缺乏对极端攻击或多模态输入的充分验证，以及对少样本新价值适配的能力尚待进一步提升。

---

## 402. Quality-Aware Collaborative Multi-Positive Contrastive Learning for Sequential Recommendation

**arXiv ID:** 2605.11707 | [PDF](https://arxiv.org/pdf/2605.11707v1)

**作者:** Wei Wang `[一作]` (Shandong University of Science and Technology), Wei Wang `[通讯]` (Shandong University of Science and Technology)

**通讯引用:** 75616 | [OpenAlex ID](https://openalex.org/A5100391883)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于对比学习的序列推荐模型QCMP-CL，利用协同增强模块生成两个多样化的正样本视图，并通过质量感知机制自适应地调整它们在对比损失中的权重，显著提升序列表示的鲁棒性与推荐效果。

**💡 创新点**

创新点在于（1）协同多视图增强：结合相同目标序列与相似序列两种互补协同上下文，生成多样且语义一致的正样本；（2）质量感知多正对比学习：利用增强操作的置信度来估计视图质量并动态赋权，抑制低质量视图导致的误匹配；（3）整体两阶段训练框架：先自监督预训练协同增强模块，再冻结其生成高质量视图，后期联合优化推荐与对比目标。

**🔧 技术方法**

技术包括：Transformer编码器/解码器、对比学习（InfoNCE）与多正对比、协同增强模块（包含删除/插入/保持操作、注意力融合与复制机制）、质量感知加权机制、双重对比损失（主多正L_qmp与辅助L_cl）以及传统的下一条项预测损失。

**📊 数据集**

使用公开的三大电商/点评数据集：Beauty、Yelp和Sports，包含数十万交互记录，平均序列长度约为8-10，数据稀疏度低于0.1%。

**📈 对比分析**

与八种SOTA基线（SASRec、BERT4Rec、CL4SRec、DuoRec、CoSeRec、ICSRec、STEAM、TCLARec）在HR、MRR、NDCG等指标上进行比较。QCMP-CL在所有数据集与指标上均实现最优成绩，提升幅度约8%–16%相较CL4SRec，约2%–4%相较TCLARec；在模拟噪声测试中，性能下降率最低，体现更好的鲁棒性。

**⚠️ 局限性**

限制包括：视图质量的置信度估计仍非完全可靠，易受预训练程度影响；仅使用批次负样本，未显式评估其可靠性；冻结增强模块可能阻碍对下游任务的自适应优化；多视图生成增加计算成本。

---

## 403. Emergent Communication between Heterogeneous Visual Agents through Decentralized Learning

**arXiv ID:** 2605.11695 | [PDF](https://arxiv.org/pdf/2605.11695v1)

**作者:** Mikako Ochiai `[一作]` (Kyoto University), Tadahiro Taniguchi `[通讯]` (Kyoto University)

**通讯引用:** 2844 | [OpenAlex ID](https://openalex.org/A5023160093)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在视觉表示不同的代理之间，通过去中心化学习和Metropolis–Hastings标注游戏（MHCG）产生共享符号的可行性；

**💡 创新点**

首次提出在没有共享任务奖励、仅基于私有感知证据的去中心化通信框架，并通过监听端的MH接受过滤证明可避免符号坍塌；

**🔧 技术方法**

采用Metropolis–Hastings采样的标注游戏、ProbVLM概率密度、BLIP式多模态结构与RSA对比分析；

**📊 数据集**

使用MS‑COCO图像与文本数据集进行训练与评估；

**📈 对比分析**

与无通信基线及不同接受规则的对照实验比较，评估跨代理文本/视觉对齐、视觉特征预测、图文检索等指标；结果显示MHCG显著优于基线，且性能随视觉编码器不匹配度升高而下降；

**⚠️ 局限性**

局限包括：仅冻结视觉编码器、仅考虑三种编码器组合、词表与序列长度有限、只使用单一数据集、采用一阶MH近似且自洽性在初始化时弱；

---

## 404. Focusable Monocular Depth Estimation

**arXiv ID:** 2605.11756 | [PDF](https://arxiv.org/pdf/2605.11756v1)

**作者:** Yuxin Du `[一作]` (Shanghai Jiao Tong University), Bo Zhao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 35322 | [OpenAlex ID](https://openalex.org/A5053256392)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种针对用户指定目标区域的可关注单目深度估计任务，并构建对应的基准FDE-Bench，提出FocusDepth框架实现局部目标精准深度推理；

**💡 创新点**

创新点在于引入多尺度空间对齐融合(MSSA)，通过将SAM3的提示条件注入Depth Anything模型的几何表示，实现对目标前景和边界的精细关注，同时保持全局几何一致；

**🔧 技术方法**

主要技术包括SAM3与Depth Anything (DA2/DA3)的联合使用、MSSA模块（多尺度空间对齐融合）、MoE门控融合、两阶段训练策略、相对深度对齐以及前景/边界/全局三重损失；

**📊 数据集**

使用NYU v2、TUM RGB‑D、YCB‑Video、RLBench、RoboTwin等公开RGB‑D数据集，构建了包含252.9K/72.5K训练/验证图像–目标–深度三元组、共972类别的FDE-Bench；

**📈 对比分析**

与零样本、全图微调的Depth Anything、UniDepth、Metric3D等基线进行对比，FocusDepth在FDE-Bench上在前景、边界和全局AbsRel/δ1均实现显著提升，特别是在对象中心与操纵场景中的性能提升最为突出；

**⚠️ 局限性**

局限性包括：仍需两阶段训练，未在大规模联合训练或下游任务中验证；提示不完整或错误仍会降低局部性能；以及对极大规模或复杂场景的可扩展性尚未充分评估。

---

## 405. Federated Client Selection under Partial Visibility: A POMDP Approach with Spatio-Temporal Attention

**arXiv ID:** 2605.11752 | [PDF](https://arxiv.org/pdf/2605.11752v1)

**作者:** Qijun Hou `[一作]` (Tsinghua University), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 46421 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对在部分可见性环境下的联邦学习客户端选择问题，构建了部分可观测马尔可夫决策过程（POMDP）模型，并提出了一种基于多步深度 Q 学习（DQL）的强化学习框架，利用空间-时间注意力网络对历史全局模型和可见客户端进行编码，实现自适应的客户端选择。

**💡 创新点**

创新点在于：①首次将客户端选择问题建模为 POMDP，充分考虑服务器只能观测到部分客户端的现实约束；②设计了空间-时间注意力架构，结合历史全局模型和客户端身份嵌入，能够捕获跨时间的客户端特性和交互关系；③采用多步 DQN 与双网络（dueling）结构，提升学习稳定性并有效利用有限样本。

**🔧 技术方法**

技术主要包括：POMDP 定义、随机投影（RP）降维、空间-时间注意力（Self‑Attention、Cross‑Attention）、身份感知嵌入、双网络 Q 估计、多步 TD 损失、经验回放、软更新、随机裁剪等。

**📊 数据集**

实验数据集：CIFAR‑10、Fashion‑MNIST（图像分类）和 UCI‑HAR（加速度信号），在三种非 IID 分布（Dirichlet、Label Skew）以及两种可见性模式（Mobile Server、Random Availability）下进行评估。

**📈 对比分析**

与 FedProx、HA‑EdgeFlow、FedAWAC 等基线进行对比。实验表明，本文方法在绝大多数场景下提升 3.6%–7.9% 的最终准确率，收敛更快、波动更小；在 Label Skew 与 Mobile Server 场景的优势尤为明显。

**⚠️ 局限性**

局限性包括：①对上下文长度 H 的选择敏感，过短会导致信息不足，过长计算成本上升；②身份嵌入需预先训练，若客户端出现频繁变动或新客户端加入，需重新初始化；③实验仅覆盖三种数据集和特定通信轮数，尚未验证在更大规模或更复杂系统中的鲁棒性；④POMDP 需要维护完整的历史观测，内存和推理开销相对较高。

---

## 406. BronchoLumen: Analysis of recent YOLO-based architectures for real-time bronchial orifice detection in video bronchoscopy

**arXiv ID:** 2605.11748 | [PDF](https://arxiv.org/pdf/2605.11748v1)

**作者:** Yongchao Li `[一作]` (Technical University of Applied Sciences Lubeck), Marian Himstedt `[通讯]` (Technical University of Applied Sciences Lubeck)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究并实现了一套基于YOLO的实时支气管口检测系统BronchoLumen，比较了YOLOv8和YOLOv12在公开支气管镜数据集上的检测效果。

**💡 创新点**

创新点在于首次公开YOLO模型权重和边界框标注，加入A2注意力模块提升YOLOv12的定位精度，并展示跨域支气管口检测的鲁棒性。

**🔧 技术方法**

采用YOLOv8-M和YOLOv12-M对象检测框架，使用anchor-free、decoupled head、C2f backbone以及A2注意力模块，并通过Ultralytics实现。

**📊 数据集**

使用的公开数据集包括BM-BronchoLC（人类内镜）和SIRGLab-DS（内镜、外镜、phantom多域），共计约4,500张图像进行训练与评估。

**📈 对比分析**

通过mAP@0.5和mAP@0.5:0.95指标进行对比：YOLOv8在Test Set1达到0.91/0.451，Test Set2为0.68/0.248；YOLOv12略优于定位精度（0.897/0.476 vs 0.909/0.451）但整体精度略低（0.836 vs 0.910）。推理速度约30 FPS，满足实时需求。

**⚠️ 局限性**

限制主要在于训练数据样本有限，导致在低对比度、运动模糊、光照不均等情况下出现误检/漏检；低分辨率输入显著降低检测准确性，且未针对不同支气管分支的大小差异进行专门优化。

---

## 407. WildRelight: A Real-World Benchmark and Physics-Guided Adaptation for Single-Image Relighting

**arXiv ID:** 2605.11696 | [PDF](https://arxiv.org/pdf/2605.11696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 408. Augmented Lagrangian Method for Last-Iterate Convergence for Constrained MDPs

**arXiv ID:** 2605.11694 | [PDF](https://arxiv.org/pdf/2605.11694v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 409. OptArgus: A Multi-Agent System to Detect Hallucinations in LLM-based Optimization Modeling

**arXiv ID:** 2605.11738 | [PDF](https://arxiv.org/pdf/2605.11738v1)

**作者:** Zhong Li `[一作]` (Great Bay University), Mingyang Sun `[通讯]` (Peking University)

**通讯引用:** 5124 | [OpenAlex ID](https://openalex.org/A5079378336)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对LLM自动化优化建模的幻觉检测方法（OptArgus），通过多代理体系结构对问题描述、符号模型和求解器实现进行结构一致性审计；同时构建三部分基准测试套件，评估清洁案例保守性、注入错误定位以及自然生成错误检测。

**💡 创新点**

创新点包括：1）首个专门针对优化建模的细粒度幻觉分类体系（objective、variable、constraint、implementation四大类及其子类）；2）将多代理分工与显式整合、重新排序相结合的检测框架；3）基于专家标注的三维基准（清洁、注入、自然）以全面评估检测效果。

**🔧 技术方法**

使用基于大型语言模型（DeepSeek‑V3.2 或 Qwen3‑Max‑Preview）的多代理推理；构建共享状态与路由策略；通过专家代理对四大幻觉类型进行局部判定；再通过确定性重新排序与聚合得到最终报告。

**📊 数据集**

数据集包括：484份由OR专家验证的干净工件、1266份人工注入单一幻觉类型的工件、6292份由多款LLM自动生成的工件（覆盖行业、供应链、能源等领域），全部经过OR专家标注，形成三种测试集。

**📈 对比分析**

与单代理检测器进行匹配基线对比，评估指标包括清洁案例的空报告率/平均报告量、注入错误的Top‑1分类、子分类、具体类型命中率以及自然工件的F1（整体与按主要类别）。结果显示：OptArgus 在清洁案例上空报告率从0.483升至0.853，报告量下降；在注入错误上三类Top‑1命中率均提升，报告量显著减少；在自然工件上各类F1均提升（例如整体F1从0.521升至0.617，宏观/微观宏观F1从0.382/0.379升至0.512/0.486）。

**⚠️ 局限性**

局限性包括：1）依赖大型语言模型，性能受模型规模与质量影响；2）基准集主要基于英文工业/科研场景，跨语言或更广泛业务场景的通用性待验证；3）多代理系统复杂度高，部署与调优成本较大；4）对极其罕见或高度耦合的多错误场景的定位精度仍有提升空间。

---

## 410. Position: LLM Inference Should Be Evaluated as Energy-to-Token Production

**arXiv ID:** 2605.11733 | [PDF](https://arxiv.org/pdf/2605.11733v1)

**作者:** Xiang Liu `[一作]` (Hong Kong University Of Science And Technology), Xiaowen Chu `[通讯]` (Hong Kong University Of Science And Technology)

**通讯引用:** 10724 | [OpenAlex ID](https://openalex.org/A5100730785)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将LLM推理的评估从传统的准确率/吞吐率/延迟转向以能源为导向的“能量‑令牌”视角，并构建了一个维度一致的令牌生产函数；

**💡 创新点**

核心创新是将推理视为受计算与供电双重瓶颈约束的生产过程，提出通过系统级优化（KV压缩、稀疏/混合注意力、量化、路由等）实现能源乘数，并呼吁在论文和基准中报告J/令牌、绑定约束、PUE调整功率和利用率；

**🔧 技术方法**

使用令牌生产函数、Leontief最小约束、ρ‑ρ*诊断、Φ_system拆解、PUE、能量/计算强度等模型与指标；

**📊 数据集**

主要利用公开统计数据（IEA电力消费、PUE平均值、H100峰值FLOPs、不同模型的e_tok估算等）以及标准评测基准（MMLU、IFEval、RULER等）作为质量与延迟基准；

**📈 对比分析**

比较方法是固定质量(q^*)和服务(s^*)条件下，报告J/令牌、实际绑定约束、PUE调整功率和利用率，展示不同阶段和地区的能量与吞吐对比；性能表现表明随着上下文长度增长和地区电力瓶颈，供电约束逐渐成为主导；

**⚠️ 局限性**

局限性包括：①令牌生产函数为短期绑定约束近似，未构成宏观经济模型；②e_tok等能量指标为方向性估计，缺乏统一的标准化基准；③需要明确的q^*,s^*配置和负载条件才能可比；④未对成本模型或市场行为进行量化；

---

## 411. ScaleMoGen: Autoregressive Next-Scale Prediction for Human Motion Generation

**arXiv ID:** 2605.11704 | [PDF](https://arxiv.org/pdf/2605.11704v1)

**作者:** Inwoo Hwang `[一作]` (Seoul National University), Chuan Guo `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于骨架-时间多尺度离散 Token 的自回归下一个尺度预测框架，实现文本驱动的 3D 人体运动生成与零拷贝编辑。

**💡 创新点**

创新点包括：①将运动生成从时间序列预测转为尺度级别预测，形成粗细层级生成；②设计骨架结构感知的多尺度位量化向量量化（bitwise VQ）和拓扑缩放，保证不同尺度下语义一致；③通过 Transformer 预测下一级尺度 Token，并通过随机位扰动提升鲁棒性；④支持在生成过程中对特定骨架部位或时间段进行零拷贝文本编辑。

**🔧 技术方法**

使用技术：骨架-时间离散 Token 化（位量化 VQ）、拓扑感知的上/下采样、Transformer 自回归生成、文本编码器与交叉注意力、随机位扰动、RoPE2d 等。

**📊 数据集**

使用数据集：HumanML3D（14k 动作 + 44k 文本）和 SnapMoGen（20k 动作 + 122k 长文本）。

**📈 对比分析**

与多种基准（Diffusion、T2M、MoMask++、SALAD 等）比较；在 HumanML3D 上取得 FID 0.030（最高），在 SnapMoGen 上 CLIP Score 0.693（最高）；R‑Precision、MM Distance、多模态性均优于现有方法；编辑实验中在用户研究中多项评分最高。

**⚠️ 局限性**

局限性：①模型对数据规模敏感，过大模型易过拟合；②位数选择需权衡 FID 与 CLIP，存在 trade‑off；③仅在现有数据集上验证，跨域泛化和实时速度仍待提升；④对极长或极复杂描述的鲁棒性尚不充分。

---

## 412. AgentDisCo: Towards Disentanglement and Collaboration in Open-ended Deep Research Agents

**arXiv ID:** 2605.11732 | [PDF](https://arxiv.org/pdf/2605.11732v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 413. Compositional Neural Operators for Multi-Dimensional Fluid Dynamics

**arXiv ID:** 2605.11691 | [PDF](https://arxiv.org/pdf/2605.11691v1)

**作者:** Hamda Hmida `[一作]`, Youssef Mesri `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了二维科学基础模型框架，将复杂流体动力学方程拆解为基础算子块（对流、扩散、非线性对流、泊松方程）并通过聚合器实现高效求解；

**💡 创新点**

创新点在于模块化预训练-组合-微调策略，冻结基础块并在聚合器中加入物理约束，实现多物理耦合的可解释性与泛化性；

**🔧 技术方法**

采用参数化傅里叶神经算子(PFNO)预训练基础块，多层感知机聚合器，以及物理信息损失、残差约束和自回归序列训练；

**📊 数据集**

使用高精度数值仿真生成的二维数据集，包括Taylor–Green vortex、剪切层、各向同性湍流等，网格尺寸为128×128和256×256；

**📈 对比分析**

与单一PFNO基线在相同数据预算下比较，标量轨道速度提升12–19×，向量轨道提升33–35×，物理残差显著下降，误差增长保持线性；

**⚠️ 局限性**

局限在于需手动选择并组装基础块，适配器训练仍耗时，对复杂几何或不同边界条件的适应性有限，且当前库仅覆盖流体方程。

---

## 414. Shaping Zero-Shot Coordination via State Blocking

**arXiv ID:** 2605.11688 | [PDF](https://arxiv.org/pdf/2605.11688v1)

**作者:** Mingu Kang `[一作]` (UNIST), Seungyul Han `[通讯]` (UNIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并评估了一种通过状态封锁产生结构化伙伴多样性的零射击协调框架 State-Blocked Coordination（SBC）。

**💡 创新点**

创新点在于利用可变的处罚状态生成虚拟环境，让伙伴在避免指定状态的约束下产生不同的子最优协作模式，从而在不修改环境或使用大规模伙伴种群的前提下实现多样化训练。

**🔧 技术方法**

采用双阶段训练（自我对弈与阻塞感知伙伴训练）、基于价值差距的处罚状态采样、距离型奖励惩罚以及 IPPO 强化学习算法。

**📊 数据集**

实验数据集包括 Multi‑Destination Spread 与 Overcooked v1 两个标准合作游戏环境。

**📈 对比分析**

与 IPPO、FCP、MEP、E3T、GAMMA、CEC 等主流 ZSC 方法以及人类代理进行对比；SBC 在自对弈与跨对弈上的得分均为最高，且在真实人类实验中实现了最低碰撞率和最高协作流畅度。

**⚠️ 局限性**

需要调节的超参数 α 与 K，计算价值差距带来额外计算成本；在大人数或更复杂环境中的可扩展性尚未充分验证。

---

## 415. Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention

**arXiv ID:** 2605.11753 | [PDF](https://arxiv.org/pdf/2605.11753v1)

**作者:** Abid Ali `[一作]` (Macquarie University), Usman Naseem `[通讯]` (Macquarie University)

**通讯引用:** 3158 | [OpenAlex ID](https://openalex.org/A5077006200)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个统一框架，既生成多模态摘要又同时选择代表性图片。

**💡 创新点**

①引入深度视觉处理器DVP实现层级视觉-文本对齐；②通过DPP监督的知识蒸馏训练轻量级VRP，实现高效多样化图像选择；③使用多目标训练平衡文本质量、视觉对齐和图像选择。

**🔧 技术方法**

LLaVA-OneVision+Qwen-2语言模型、SigLIP视觉编码器、Perceiver式视觉采样器、深度视觉处理器DVP、门控跨模态注意力、DPP蒸馏式VRP、跨模态对齐损失、多目标损失。

**📊 数据集**

MSMO（多模态新闻摘要）数据集。

**📈 对比分析**

与ATG、ATL、HAN、MOF、UniMS、SITA、BART‑VGG、ViL‑Sum、DIUSum等基线在ROUGE、IP、MaxSim、MMAE等指标对比。我们的DVP在ROUGE‑1 44.20、ROUGE‑2 20.77与ViL‑Sum接近，图像选择IP 74.03显著优于ViL‑Sum的66.27，多目标训练提升了文本与视觉质量。

**⚠️ 局限性**

自动评测指标（ROUGE等）对视觉语义不敏感；多样性分数易被无关图像膨胀；缺乏完整的视觉‑文本对齐人类评测；模型可能继承数据集偏见。

---

## 416. Persona-Conditioned Adversarial Prompting: Multi-Identity Red-Teaming for Adversarial Discovery and Mitigation

**arXiv ID:** 2605.11730 | [PDF](https://arxiv.org/pdf/2605.11730v1)

**作者:** Cristian Morasso `[一作]` (IBM Research), Douglas Leith `[通讯]` (Trinity College Dublin)

**通讯引用:** 10693 | [OpenAlex ID](https://openalex.org/A5086446911)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Persona‑Conditioned Adversarial Prompting（PCAP），通过在自动红队中引入多样化攻击者角色（如医生、学生、恶意攻击者）和策略卡，生成更丰富、更具可转移性的 jailbreak prompt，并构建无人工标注、带有元数据的对抗语料库，用于轻量级自适应微调，形成闭环安全强化流程。

**💡 创新点**

创新点在于：① 将攻击者身份与策略作为搜索条件，显著提升攻击成功率与提示多样性；② 在搜索过程中自动记录元数据，省去人工标注；③ 通过轻量级 ALoRA 微调在多种模型上实现高鲁棒性，展示闭环红队→防御的可行性。

**🔧 技术方法**

使用技术包括：TAP（Tree of Attacks）框架的并行扩展、Persona 生成器与目标重构模型、策略卡作为上下文引导、on‑topic 过滤器、评估器（E）、相似度剪枝；Fine‑tune 时采用 ALoRA 自适应权重，训练时加入策略标签；紫队验证（purple‑teaming）评估防御效果。

**📊 数据集**

数据集：取 AdvBench 子集 50 个攻击目标，PCAP 生成 300 条成功对抗 prompt（每目标 6 条），覆盖 3 大模型（Llama 3.3‑70B、Granite 4.0‑H Tiny、Mixtral 8×22B）以及 5 种攻击者模型；同时使用 GPT‑OSS‑120B、Llama‑3.3‑70B 等内部推理服务做评测。

**📈 对比分析**

方法对比：与传统 TAP 基线比较，PCAP 在 5 种攻击者模型上均提升 ASR（+10%~+57%），提示产出量提升 2‑6 倍；对专门防护层的迁移实验显示 30‑58% 的可转移率；微调后，召回率从 0.26‑0.36 提升到 0.96‑1.00，F1 从 0.41‑0.53 提升到 0.96‑0.98，精确率保持 0.94‑0.97；紫队验证显示攻击成功率下降 24‑92%，查询次数提升 5‑7 倍。

**⚠️ 局限性**

局限性：① 并行 persona 搜索增加查询成本，尤其在小模型上扩展性差；② 受 Persona 生成、重构、评估 LLM 的偏见影响，可能漏掉某些攻击类；③ 对抗数据微调会略微提升误拦率，生产环境需使用平衡数据并人工复核；④ 目前仅支持单轮攻击，缺少多轮/代理人扩展。

---

## 417. CAST: Collapse-Aware multi-Scale Topology Fusion for Multimodal Coreset Selection

**arXiv ID:** 2605.11705 | [PDF](https://arxiv.org/pdf/2605.11705v1)

**作者:** Boran Zhao `[一作]` (Xi'an Jiaotong University), Pengju Ren `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 1661 | [OpenAlex ID](https://openalex.org/A5044243518)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于多尺度拓扑融合的 CAST 框架，用于从大规模多模态数据集中高效选择代表性子集（coreset）。

**💡 创新点**

创新点包括：① 对图结构的局部崩塌自适应修正与跨模态融合；② 在扩散小波域构建多尺度分布匹配目标；③ 引入局部软关系覆盖（LSRC）来降低高密度区域的冗余。

**🔧 技术方法**

采用图构建、局部崩塌评估、扩散小波变换、Sliced Wasserstein 距离、多尺度匹配、LSRC 关系传播以及匈牙利算法映射等技术。

**📊 数据集**

在 Flickr30K、MS‑COCO 图文检索数据集以及 LLaVA‑1.5‑mix‑665K 生成式多模态指令微调数据集上进行实验。

**📈 对比分析**

与无模态、单模态以及现有多模态选择/合成基线对比，CAST 在不同压缩预算下在检索 Recall@K、跨架构泛化、指令微调和能耗等指标上均优于所有基线，显示出显著的性能提升和能源节省。

**⚠️ 局限性**

局限性包括：对图结构构建与小波尺度的超参数需要经验调优；方法主要针对图像‑文本对，扩展到其他模态或更大规模时可能面临计算与存储挑战；并未给出严格的理论收敛或表示完备性分析。

---

## 418. MindMirror: A Local-First Multimodal State-Aware Support System for Digital Workers

**arXiv ID:** 2605.11700 | [PDF](https://arxiv.org/pdf/2605.11700v1)

**作者:** Wenqi Luo `[一作]` (East China Normal University), Yan Wang `[通讯]` (East China Normal University)

**通讯引用:** 213888 | [OpenAlex ID](https://openalex.org/A5100437036)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 MindMirror，一款本地优先的多模态状态感知支持系统，利用摄像头表情识别、文本与可选语音输入实现状态检查、结构化阻塞反思、LLM生成建议以及日/周回顾报告，帮助数字工作者记录并反思工作状态。

**💡 创新点**

①将情绪识别作为可编辑的状态提示并提供手动校正；②设计结构化三问反思流程降低提示生成负担；③整合本地LLM和多模态交互形成闭环；④采用本地-first数据存储提升隐私。

**🔧 技术方法**

前端Web（HTML5/CSS/JS、MediaDevices API、Chart.js），后端Flask，ViT-based 表情识别模型，Ollama托管的 Qwen LLM，PyTorch/Transformers，语音STT/TTS（可选第三方），本地 JSON/LocalStorage 存储。

**📊 数据集**

FER-2013 基准（6,767 张图）+自采亚洲面部图像，共同构成七类表情基准；诊断子集 500 样本用于细粒度评估。

**📈 对比分析**

通过对比非微调基线与微调模型，准确率从 59.66% 提升至 94.49%（+34.83%）。诊断子集宏 F1 0.9195。系统技术验证显示核心 API 30 次 100% 通过，语音路径 10 次 100% 通过，整体端到端延迟约 1.2 秒。

**⚠️ 局限性**

表情识别仅为视觉提示，无法真实捕捉心理状态；语音功能依赖第三方 API，缺乏多声学环境评估；评估样本规模小，缺乏长效使用数据；LLM 生成的建议可操作性仍需提升；本地部署仍需进一步完善隐私与离线 ASR/TTS。

---

## 419. Auditing African Content Moderators' Working Conditions by Using the European General Data Protection Regulation (GDPR)

**arXiv ID:** 2605.11699 | [PDF](https://arxiv.org/pdf/2605.11699v1)

**作者:** Mariame Tighanimine `[一作]` (University of Neuchâtel), James Oyange `[通讯]` (African Content Moderators Union)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过向在肯尼亚和尼日利亚的内容审核员提出欧盟通用数据保护条例（GDPR）的主体访问请求，收集并审计了其雇佣合同、保密协议等工作条件文件。

**💡 创新点**

创新之处在于将GDPR的跨境执法能力与劳工权益审计相结合，利用法律工具迫使企业披露隐藏的雇佣文件，并将工人定位为共研者，首次提供了结构化的实证证据揭示非洲内容审核员的剥削现象。

**🔧 技术方法**

主要使用GDPR条款（第15条访问权、第20条可携带权）和Digipower方法学框架进行数据请求和法律分析；未涉及机器学习等技术。

**📊 数据集**

数据集为五名工人从Teleperformance和Sama两家BPO公司获得的个人资料，包括雇佣合同、任命函、保密协议、工资单和绩效记录。

**📈 对比分析**

通过对比同一公司内不同工人收到的文件差异，以及对合同条款（职位、工时、合同期限、工作描述、薪酬、地点灵活性和NDA条款）进行定性分析，展示了工人待遇的不一致性和系统性剥削；并对比GDPR回应的完整性，说明了不同工人和公司在合规程度上的差异。

**⚠️ 局限性**

局限性包括样本量极小（仅五名工人，覆盖两家公司两国），无法具备统计代表性；受GDPR适用范围限制，可能在其他地区难以复制；部分请求遭遇公司阻碍，导致数据不完整；并未对工人真实工时、工资等实际数值进行量化评估。

---

## 420. Slicing and Dicing: Configuring Optimal Mixtures of Experts

**arXiv ID:** 2605.11689 | [PDF](https://arxiv.org/pdf/2605.11689v1)

**作者:** Margaret Li `[一作]` (University of Washington), Luke Zettlemoyer `[通讯]` (University of Washington)

**通讯引用:** 30540 | [OpenAlex ID](https://openalex.org/A5067919401)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了Mixture-of-Experts（MoE）架构中的专家数量、粒度、异质性、共享专家、负载平衡与无丢失路由等多维度组合，进行2000多次预训练实验。

**💡 创新点**

提出简化MoE调优的实践方案：只需关注专家总数与粒度，保持粒度约为1/4，其他超参数对最终性能影响有限。

**🔧 技术方法**

采用Transformer主体、基于token的路由器、负载平衡辅助损失、无丢失路由等技术，对多种专家配置进行系统对比。

**📊 数据集**

使用公开的混合语料（网络文本、代码、数学、百科等）进行预训练，规模覆盖10M到6.6B总参数。

**📈 对比分析**

与稠密模型和不同MoE配置比较，发现总专家参数越多、粒度越粗越能提升语言模型性能；无丢失路由略优于默认路由；负载平衡超参数对性能影响有限。

**⚠️ 局限性**

实验受限于可用计算资源，未覆盖极大规模（如10B+）及更广泛的下游任务评测；异质专家和共享专家的潜在优势在当前设置下未显现，可能需更复杂的实验设计。

---

## 421. Persistent and Conversational Multi-Method Explainability for Trustworthy Financial AI

**arXiv ID:** 2605.11687 | [PDF](https://arxiv.org/pdf/2605.11687v1)

**作者:** Georgios Makridis `[一作]` (ExpertAI-Lux), Dimosthenis Kyriazis `[通讯]` (University of Piraeus)

**通讯引用:** 4531 | [OpenAlex ID](https://openalex.org/A5069674161)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套面向金融情感分析的可持续、跨方法、对话式可解释 AI 架构，将 XAI 结果存储为持久、可搜索的对象，并通过 RAG 聊天机器人实现多方法解释三角化。

**💡 创新点**

创新点包括：1) 以 S3 对象存储实现 XAI 产出持久化与语义检索；2) 将多种 XAI 方法结果索引并在对话中交叉验证；3) 通过约束式提示与自动化可信度检查提升生成解释的可靠性。

**🔧 技术方法**

使用技术包括：Docker 微服务架构、RustFS S3 兼容存储、FinBERT 预测模型、LIME 与遮挡法的文本重要性分析、视觉显著性热图、检索增强生成（RAG）与 OpenAI embeddings、约束式提示、规则化的可信度评估。

**📊 数据集**

采用基于 FinBERT 的金融新闻情感分类数据集（新闻标题正负中性标签）进行实验。

**📈 对比分析**

通过对比 30 个评估查询的“约束式提示”与“普通提示”，约束式提示将幻觉率降低 36%，方法归因引用次数提升 73%，在单方法、跨方法、对抗性以及数据集级查询上均表现优异，尽管基线在 grounding 完整度上略优。

**⚠️ 局限性**

局限性包括：可信度评估仅基于规则检查且缺乏全面的 ground‑truth 覆盖；向量检索使用简单余弦相似度，未使用近似最近邻；实验仅覆盖两种文本 XAI 方法，未进行人机试验；系统对更大规模部署和多模态任务的性能尚未验证。

---

## 422. DORA: Dynamic Online Reinforcement Agent for Token Merging in Vision Transformers

**arXiv ID:** 2605.11683 | [PDF](https://arxiv.org/pdf/2605.11683v1)

**作者:** Kaixuan He `[一作]` (University of Science and Technology of China), Yi Kang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3557 | [OpenAlex ID](https://openalex.org/A5101941645)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DORA，一种在线强化学习框架，用于 Vision Transformer 的动态 token 合并，以降低计算复杂度。

**💡 创新点**

创新点在于将 token 合并建模为序列 MDP，使用在线 RL 生成输入依赖的 merge mask，结合非线性知识蒸馏奖励和异构 Actor‑Critic 结构，实现自适应的 token 选择。

**🔧 技术方法**

采用强化学习（PPO）、知识蒸馏、异构 Actor‑Critic、Transformer 状态编码、非线性奖励惩罚以及多对一聚合等技术。

**📊 数据集**

使用 ImageNet-1K 进行主实验，并在 ImageNet-A、ImageNet-R、ImageNet-C 上评估 OOD 鲁棒性。

**📈 对比分析**

在与 ToMe、ZeroTPrune、V‑Pruner 等基线在 Top‑1 误差≤0.05% 条件下对比，DORA 的 token 合并率提升 10–28%，FLOPs 降低 10–76%，在 OOD 场景下相对基线提升 430–569% 的效率，且准确率保持稳定。

**⚠️ 局限性**

局限性包括需要离线训练高容量 Critic、仅针对图像任务验证、对视频/时序任务和更大规模模型的可扩展性尚未充分评估。

---

## 423. HySecTwin: A Knowledge-Driven Digital Twin Framework Augmented with Hybrid Reasoning for Cyber-Physical Systems

**arXiv ID:** 2605.11682 | [PDF](https://arxiv.org/pdf/2605.11682v1)

**作者:** David Holmes `[一作]` (Edith Cowan University), Helge Yanicke `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 HySecTwin——一种将语义建模与混合推理集成的数字孪生框架，用于实时检测工业 CPS 的网络攻击。

**💡 创新点**

创新点在于：①将 CPS 结构与安全知识语义化为 RDF/OWL 本体；②将确定性规则和模糊推理结合的混合推理引擎；③轻量化、可容器化的实现，使数字孪生可直接嵌入生产系统。

**🔧 技术方法**

技术：语义本体（SAREF、OWL2）、Eclipse Ditto 数字孪生平台、Rete 规则引擎（Durable Rules）、模糊推理（FuzzyLite）、MQTT、MongoDB、InfluxDB。

**📊 数据集**

数据集：实验使用自建智能照明 CPS 仿真环境，基于 MITRE ATT&CK 组 Dragonfly 2.0 的攻击场景（C0012、C0020、C0025、C0028），生成的 MQTT 事件流。

**📈 对比分析**

对比：与单纯确定性推理 baseline 比较，混合推理在检测延迟上平均提升 21.5%（减少 16–21%），且不增加系统开销；吞吐量与延迟与物理双通道基本一致，符合 NIST 规范，说明系统可在实时环境下运行。

**⚠️ 局限性**

局限：依赖高保真数字孪生和完整本体，未知攻击模式仍难以捕获；规则和本体维护需要人工，缺乏完全自动化；对复杂异构设备的兼容性与扩展性还有待进一步验证。

---

## 424. When Reasoning Traces Become Performative: Step-Level Evidence that Chain-of-Thought Is an Imperfect Oversight Channel

**arXiv ID:** 2605.11746 | [PDF](https://arxiv.org/pdf/2605.11746v1)

**作者:** Wenkai Li `[一作]` (Carnegie Mellon University), Koichi Onoue `[通讯]` (Fujitsu Research of America Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究Chain-of-Thought（CoT）在多步推理中的时间一致性，量化内部答案承诺与可见推理文本的同步程度。

**💡 创新点**

提出基于步骤的时间一致性度量、五类不匹配分类法，以及架构匹配的训练管线对齐实验，揭示CoT不一致主要源自后续伪造步骤。

**🔧 技术方法**

采用 logit lens、Patchscopes、tuned-lens 探针和方向消融等内部读出技术进行答案承诺检测与因果验证。

**📊 数据集**

使用七个推理基准（MATH-500、PrOntoQA、ProsQA、MMLU-Pro、BBH-LD、BBH-TSO、GPQA-Diamond）。

**📈 对比分析**

对九个开源模型进行对比，平均对齐度为 61.9%；主导不匹配为后续伪造步骤（58%）；深度学习管线对齐分布显著影响，CoT 在对齐较差的任务上提升更明显。

**⚠️ 局限性**

仅通过 logit 概率捕捉答案承诺，可能忽略非线性或多词信念；实验仅覆盖 7B–32B 开源模型，结果可能不具普适性。

---

## 425. Training-Inference Consistent Segmented Execution for Long-Context LLMs

**arXiv ID:** 2605.11744 | [PDF](https://arxiv.org/pdf/2605.11744v1)

**作者:** Xianpeng Shang `[一作]` (Inner Mongolia University), Xiangdong Su `[通讯]` (Inner Mongolia University)

**通讯引用:** 1336 | [OpenAlex ID](https://openalex.org/A5100830970)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练–推理一致的分段执行框架，以解决长上下文大模型在推理阶段与训练阶段存在的执行不匹配问题。

**💡 创新点**

通过将跨段状态限制为可微的 KV 尾部并使用仅前向检索前缀，实现了训练与推理在分段层面上的完全一致，并证明了截断反向传播（TBPTT）可得到一致目标的精确梯度。

**🔧 技术方法**

采用 Transformer 解码器架构，在每层将注意力头分为局部头和长范围头；局部头携带 KV 尾部，长范围头仅在指定层通过检索前缀进行前向访问；并结合 RoPE 重索引、TBPTT、可分离检索池等技术。

**📊 数据集**

在 LLaMA2‑32K、LLaMA2‑80K 等长上下文模型上，使用 PG19、LongBench、RULER 等公开数据集进行评测。

**📈 对比分析**

与全注意力、CCA‑attention、StreamingLLM、DuoAttention、MInference 等基线相比，方法在极长上下文（如128K）下显著降低预填内存（≈6×），保持与全注意力相当的 PPL；在 LongBench‑E、RULER 等任务上获得最高或竞争力的分数；在推理速度和内存占用上优于大多数基线。

**⚠️ 局限性**

依赖于固定长度的 KV 尾部与检索前缀，可能在极大语境跨度下仍受限；检索池随长度增长，虽为前向，但仍占用内存；方法需要额外的检索实现与调参，且在非常短上下文或非分段任务中效果不明显。

---

## 426. Block-R1: Rethinking the Role of Block Size in Multi-domain Reinforcement Learning for Diffusion Large Language Models

**arXiv ID:** 2605.11726 | [PDF](https://arxiv.org/pdf/2605.11726v1)

**作者:** Yan Jiang `[一作]` (University of Queensland), Zi Huang `[通讯]` (University of Queensland)

**通讯引用:** 13718 | [OpenAlex ID](https://openalex.org/A5078170935)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在多域强化学习后训练 Diffusion 大语言模型时，块大小冲突对模型性能的影响，并提出了基于样本级最佳块大小的训练方法。

**💡 创新点**

创新点在于正式定义域块大小冲突并量化其对多域 RL 的负面影响，构建了 Block‑R1‑41K 数据集，其中每个样本标注了最佳训练块大小，并将该标注用于实现样本级块条件化训练，极大缓解了域间冲突。

**🔧 技术方法**

使用了强化学习（如 StableDRL、GRPO 等）与块级半自回归生成、教师-学生奖励对比评估、Wasserstein 距离衡量冲突、动态块大小推断等技术。

**📊 数据集**

数据集包含 13 个推理基准（数学推理、代码生成、逻辑谜题、通用能力等）共 41K 样本，每个样本配有最佳块大小标签。

**📈 对比分析**

通过与单域 RL、统一块大小多域 RL、动态块大小推断等方法对比，Block‑R1 在多域下平均提升约 15–30%（在 13 个基准上显著优于现有方法），并在多域 RL 任务中取得最佳性能。

**⚠️ 局限性**

局限性在于需要人工或教师模型来确定最佳块大小，对低质量样本或非标准任务的推广受限，且未将块大小自适应化与 RL 训练完全联合，仍有提升空间。

---

## 427. EPIC: Efficient Predicate-Guided Inference-Time Control for Compositional Text-to-Image Generation

**arXiv ID:** 2605.11722 | [PDF](https://arxiv.org/pdf/2605.11722v1)

**作者:** Sunung Mun `[一作]` (POSTECH), Jungseul Ok `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关的推理时细化框架（Predicate‑Guided Refinement），通过将文本提示一次性编译成固定的视觉程序（包含对象、计数、属性、关系等谓词），并在每次生成或编辑后使用该程序进行谓词级检验，从而实现对多对象、多属性、多关系提示的精准满足。

**💡 创新点**

创新点在于将提示解析为稳定的视觉程序并利用谓词状态向量来驱动细化决策：未满足的谓词直接指示是局部编辑还是全局重采样，从而在保持提示一致性的前提下提供可定位的反馈，显著提升推理时效率与准确性。

**🔧 技术方法**

采用的技术包括：视觉程序构造（Parser + Rewriter）、多模态大型语言模型（Qwen3‑VL‑32B）用于审核与编辑指令生成、基于检测器与属性识别器的谓词验证器、图像生成器（FLUX.2‑Klein‑9B）和图像条件式指令编辑器，辅以谓词驱动的细化策略和预算管理。

**📊 数据集**

主要在GenEval2基准集上评估，使用FLUX.2‑Klein‑9B作为生成器、相同检查点的编辑器，以及Qwen3‑VL‑32B作为MLLM；实验覆盖高原子度提示，比较了单通道生成与多种细化基线。

**📈 对比分析**

与单通道生成、BoN+NVILA、T2I‑Copilot及RAISE等方法对比，Predicate‑Guided Refinement在相同的32次图像模型调用预算下实现了71.46%（比单通道提升37.3个百分点）的提示级准确率，且在图像模型调用、MLLM调用与令牌使用方面均优于RAISE，展现出最佳的增益归一化效率。

**⚠️ 局限性**

局限性包括：依赖视觉程序和谓词验证器的准确性；错误或模糊的程序会在整个细化过程中持续误导；误报/漏报会导致不必要的编辑或重采样；谓词表示的范围有限，无法覆盖所有需要推理的提示；系统可能继承生成器、检测器和MLLM的偏见，需配合安全过滤与内容审查。

---

## 428. A Research Agenda on Agents and Software Engineering: Outcomes from the Rio A2SE Seminar

**arXiv ID:** 2605.11720 | [PDF](https://arxiv.org/pdf/2605.11720v1)

**作者:** Davide Taibi `[一作]` (University of Southern Denmark), Lucas Romao `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过组织A2SE研讨会，收集18位专家意见，提出面向Agentic AI的六大主题研究议程。

**💡 创新点**

首次将治理、软件工程、架构、质量、可持续性与代码等六大主题聚焦为短期与长期的系统化研究方向，形成社区驱动的、跨学科的研究路线图。

**🔧 技术方法**

采用结构化研讨流程（演示、主题聚类、焦点小组讨论）、开放式编码以及ChatGPT辅助初始代码生成来整理与分析专家意见。

**📊 数据集**

未使用传统实验数据集，主要以研讨会现场收集的专家笔记与讨论记录为数据来源。

**📈 对比分析**

通过研讨会后的反馈与验证环节，确保所列研究方向与优先级获得与会专家一致认可；并未进行性能评估或对比实验。

**⚠️ 局限性**

局限在于缺乏经验性验证与定量评估，议程构建高度依赖专家主观意见，后续需要更多实证研究与案例检验来验证所提方向的有效性与可行性。

---

## 429. Introducing Environmental Constraints to Grasping Strategies for Paper-Like Flexible Materials Using a Soft Gripper

**arXiv ID:** 2605.11714 | [PDF](https://arxiv.org/pdf/2605.11714v1)

**作者:** Yi Dong `[一作]` (Nanjing University of Aeronautics and Astronautics), Zhendong Dai `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 6294 | [OpenAlex ID](https://openalex.org/A5100746667)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文使用通用软夹具结合桌面、墙面等环境约束，对纸质柔性材料制定并实验验证了四种抓取策略。

**💡 创新点**

创新点在于系统化提出基于环境约束的抓取策略、推导其力学与运动模型，并量化不同策略的工作空间与抓取性能。

**🔧 技术方法**

使用软夹具、ROS框架、YOLO目标检测、深度相机、力传感器以及夹具开合与倾斜控制等技术。

**📊 数据集**

实验采用不同 GSM（17–250 g）规格的纸张样本，尺寸为 105 mm × 297 mm。

**📈 对比分析**

与现有专用夹具/多臂/多指抓取方法对比，本文策略在多种纸张厚度下实现 70–100 % 的成功率，且使用通用软夹具实现更高效率和鲁棒性。

**⚠️ 局限性**

局限性包括模型假设纸张为线性弹性、摩擦系数恒定，未考虑材料异向性、厚度非均匀性以及未知材料对抓取稳定性的影响。

---

## 430. Unlocking Compositional Generalization in Continual Few-Shot Learning

**arXiv ID:** 2605.11710 | [PDF](https://arxiv.org/pdf/2605.11710v1)

**作者:** Phu-Quy Nguyen-Lam `[一作]` (Vietnam National University), Long Tran-Thanh `[通讯]` (University of Warwick)

**通讯引用:** 2744 | [OpenAlex ID](https://openalex.org/A5012915897)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在持续少样本学习中提出COMPOSE框架，先用冻结的自监督ViT+Slot Attention提取对象级表示，再通过轻量化路由器与投影头在全局级别上训练，推理阶段使用保持的对象槽进行无梯度的组合匹配，达到对全新概念的强泛化；

**💡 创新点**

核心创新在于将表示学习与组合推理完全分离：通过在训练时采用全局交叉熵与交叉相关去耦合损失，避免了传统局部匹配导致的槽过拟合；

**🔧 技术方法**

技术包括自监督ViT（如DINOv2）提取patch语义几何、Slot Attention + 关注加权聚合生成槽、MLP路由器与线性投影头、全局交叉熵与交叉相关惩罚、推理时的双向Chamfer匹配与槽中心化；

**📊 数据集**

主要使用合成复合图像数据集CGQA和COBJ进行组合少样本测试，同时在CIFAR‑100、ImageNet‑R、ImageNet‑A、miniImageNet、CUB‑200等标准持续学习基准上评估；

**📈 对比分析**

相较于现有零样本、离线元学习、PTM‑CIL方法，COMPOSE在CGQA的无新概念（noc）分数达到94.14%，H_a 92.92，刷新了公开基准；在持续学习任务中保持低灾难性遗忘（如CIFAR‑100 FF从27.91%降至3.54%），并取得最高平均准确率；

**⚠️ 局限性**

局限性在于高度依赖自监督ViT提供的patch语义几何；当语义结构弱化（如监督ViT、CLIP或域迁移场景）时，槽分解质量下降，导致组合泛化和记忆保持受限；

---

## 431. Rainbow Deep Q-Learning with Kinematics-Aware Design for Cooperative Delta and 3-RRS Parallel Robot Insertion

**arXiv ID:** 2605.11697 | [PDF](https://arxiv.org/pdf/2605.11697v1)

**作者:** Hassen Nigatu `[一作]` (Zhejiang University), Lu Guodong `[通讯]` (Zhejiang University)

**通讯引用:** 4794 | [OpenAlex ID](https://openalex.org/A5025742489)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出基于Delta与3‑RRS协作的插孔任务，先进行几何优化后使用Rainbow DQN实现插孔控制。

**💡 创新点**

关键创新在于先对3‑RRS几何进行最优化以扩大无奇异点工作空间，再将运动学信息融入Reward与动作屏蔽，显著提升学习效率与插入成功率。

**🔧 技术方法**

采用几何优化、Rainbow DQN（双Q、dueling、PER、n‑step、NoisyNet、distributional）以及基于运动学的状态/动作屏蔽。

**📊 数据集**

使用高保真仿真环境产生的轨迹数据，没有公开真实数据集。

**📈 对比分析**

与传统RRT‑Connect规划器和普通DQN进行对比，Rainbow DQN在成功率（95%）和插入时间、对齐误差、碰撞率等指标上均优于两者。

**⚠️ 局限性**

局限包括仅在仿真中验证、离散动作导致轨迹不够平滑、缺乏硬件实现与对实时噪声的鲁棒性验证。

---

## 432. Robust LLM Unlearning Against Relearning Attacks: The Minor Components in Representations Matter

**arXiv ID:** 2605.11685 | [PDF](https://arxiv.org/pdf/2605.11685v1)

**作者:** Zeguan Xiao `[一作]` (Shanghai University of Finance and Economics), Guanhua Chen `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 6772 | [OpenAlex ID](https://openalex.org/A5100665987)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型在去学习（unlearning）后易被重学习攻击恢复知识的脆弱性，并提出了基于少数主成分的“Minor Component Unlearning（MCU）”方法，利用表示几何投影使去学习更难被恢复。

**💡 创新点**

创新点在于揭示了去学习过程主要作用于表示中的主成分，导致攻击能轻易逆转；随后通过投影消除主成分，将优化重心转移到对抗性更强的次要成分，从而显著提升鲁棒性。

**🔧 技术方法**

技术手段包括主成分分析（PCA/SVD）、投影到次要子空间、改写RMU和MLP Breaking损失以仅作用于次要成分，以及与梯度层面过滤方法CIR的结合。

**📊 数据集**

实验使用三个忘记集：WMDP‑Cyber、WMDP‑Bio和Years，并在保留集上进行评估，验证方法在不同领域的泛化能力。

**📈 对比分析**

与传统去学习方法（NPO、RMU、MLP Breaking）以及与SAM、CIR等鲁棒性提升技术对比，MCU在所有数据集上均显著降低重学习后恢复率（Δ值大幅下降），同时保持或提升MMLU与WikiText性能，表明既提升了安全性又保持了实用性。

**⚠️ 局限性**

局限性包括：需额外计算主成分并投影，导致一定的计算与存储开销；方法对极其强的表示级攻击（直接对齐内部激活）仍存在一定恢复风险；在更大规模模型或不同架构上进一步验证仍待探索。

---

## 433. DreamAvoid: Critical-Phase Test-Time Dreaming to Avoid Failures in VLA Policies

**arXiv ID:** 2605.11750 | [PDF](https://arxiv.org/pdf/2605.11750v1)

**作者:** Xianzhe Fan `[一作]` (HKU), Hengshuang Zhao `[通讯]` (HKU)

**通讯引用:** 35553 | [OpenAlex ID](https://openalex.org/A5078109015)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

该论文提出了关键阶段测试时梦境（Critical‑Phase Test‑time Dreaming）框架，使 Vision‑Language‑Action（VLA）模型在任务关键阶段能够提前“梦境”并预测后果，从而避免失败；并引入自主边界学习（Autonomous Boundary Learning）让世界模型具备失败意识。

**💡 创新点**

创新点包括：①仅在检测到关键阶段时才触发高成本的测试时梦境，显著降低计算量；②通过 SDE 采样生成多样化候选动作，突破单一确定性 ODE 的局限；③利用大型预训练的世界模型（DreamDojo）进行短期视觉梦境并用 Robometer 进度值进行评估；④通过自组织收集成功、边界和失败数据并联合训练，使模型学习成功与失败的细微边界。

**🔧 技术方法**

主要技术：流式匹配（Flow Matching）生成基线动作；SDE 采样提升候选动作多样性；Dream Trigger（轻量二分类器）识别关键阶段；Action Proposer 采样候选动作；Dream Evaluator（DreamDojo + 价值模型）进行短期未来预测与评分；软标签、Huber + 排名损失的联合训练；边界学习机制。

**📊 数据集**

数据集：真实世界四项细粒度操作（杯子套、充电器插、盖子开启、螺钉插）共 160 次实验；LIBERO 与 SimplerEnv 这两个仿真基准；人类遥控演示数据（Teleop）以及自组织收集的边界/失败轨迹。

**📈 对比分析**

与基线 VLA（π_0.5）和 GPC‑RANK 进行对比；DA‑ABL 在真实任务中平均成功率 72.5%（比 π_0.5 提升 23.7%），在 LIBERO 上达 97.8%（比 π_0.5 提升 1.3%），在 SimplerEnv 上分别为 63.6%/80.7%（均优于 GPC‑RANK）。同时计算开销仅在关键阶段触发，显著低于 GPC‑RANK 的全时常计算。

**⚠️ 局限性**

局限性：①仍需生成高保真视频导致关键阶段的延迟；②目前框架主要适用于 VLA 策略，需进一步验证可移植性；③自组织的失败数据收集效率有限，可能不足以覆盖所有失败边界；④未在潜在空间中加速梦境生成，未来可尝试在更小的潜在空间内进行短期预测。

---

## 434. Measuring What Matters Beyond Text: Evaluating Multimodal Summaries by Quality, Alignment, and Diversity

**arXiv ID:** 2605.11693 | [PDF](https://arxiv.org/pdf/2605.11693v1)

**作者:** Abid Ali `[一作]` (Macquarie University), Usman Naseem `[通讯]` (Macquarie University)

**通讯引用:** 3158 | [OpenAlex ID](https://openalex.org/A5077006200)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MM‑Eval统一评估框架，用于评估多模态摘要与多模态输出（MSMO）系统的文本质量、跨模态一致性和视觉多样性。

**💡 创新点**

创新点在于：①整合了文本事实一致性（OpenFActScore）、文本可读性（G‑Eval）、图像‑文本匹配（MLLM‑as‑Judge）和图像多样性（TCE）四大模块；②通过监督学习（岭回归）将三大维度加权聚合，形成对人类评审偏好高度一致的综合分数；③框架实现参考弱、可解释、模块化，易于在新领域迁移。

**🔧 技术方法**

技术包括：OpenFActScore（事实核对）、G‑Eval（LLM‑based 语言质量评估）、MLLM‑as‑Judge（多模态LLM进行图像‑文本相关性评估）、Truncated CLIP Entropy（图像多样性度量）、Ridge回归（学习聚合权重）。

**📊 数据集**

使用的主数据集是mLLM‑Eval新闻基准，包含142篇新闻、多模态摘要及约1,500条专家整体质量标注。

**📈 对比分析**

与传统的单一平均或手工加权基线相比，MM‑Eval在与人类评分的排名相关性（Kendall τ≈0.374、Spearman ρ≈0.514）上有明显提升；学习得到的权重显示文本质量占约79%，事实一致性约43%是最关键的门槛。

**⚠️ 局限性**

局限性包括：①在文本主导的新闻场景下评估结果高度偏向文本，视觉维度的贡献被低估；②TCE和MLLM‑Judge的视觉评估仍可能噪声较大；③整体相关性仅中等，适用于粗至中等水平比较，极其相近的系统仍需人工评审；④在图像主导领域需重新校准权重。

---

## 435. Allegory of the Cave: Measurement-Grounded Vision-Language Learning

**arXiv ID:** 2605.11727 | [PDF](https://arxiv.org/pdf/2605.11727v1)

**作者:** Kepeng Xu `[一作]` (Xidian University), Wenxin Yu `[通讯]` (Southwest University of Science and Technology)

**通讯引用:** 616 | [OpenAlex ID](https://openalex.org/A5027038441)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将视觉语言模型的输入从后ISP RGB改为保留测量域的RAW-derived XYZ，并构建PRISM-VL模型进行指令调优和评估。

**💡 创新点**

创新性地提出测量域可观测接口概念，并结合Meas.-XYZ观测、相机条件化的多模态接地以及曝光括号监督聚合(BracketSup)三大技术，实现了更完整的传感器证据利用。

**🔧 技术方法**

采用RAW到线性XYZ的转换、ISO/曝光/光圈等元数据在问题和视觉编码器中的残差条件化，并通过多曝光代理聚合进行监督转移。

**📊 数据集**

基于RAISEDNG、AODDNG、PASCALRAW等RAW数据集构建了约150K条指令调优样本，并设计了2,183条低光、高动态范围、可见性敏感等场景的评测基准。

**📈 对比分析**

在BLEU、ROUGE-L和LLM-Judge准确率上与RGB基线Qwen3-VL-8B对比，PRISM-VL-8B分别取得0.6120、0.4571和82.66%的成绩，提升约+0.1074 BLEU、+0.1071 ROUGE-L、+4.46% LLM-Judge。

**⚠️ 局限性**

局限性包括对RAW数据及相机元数据的依赖，迁移性受限于硬件/格式差异，且在非低光/高动态范围场景的优势不如在专门评测场景显著。

---

## 436. Debiased Model-based Representations for Sample-efficient Continuous Control

**arXiv ID:** 2605.11711 | [PDF](https://arxiv.org/pdf/2605.11711v1)

**作者:** Jiafei Lyu `[一作]` (Tencent Hunyuan), Deheng Ye `[通讯]` (Tencent Hunyuan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DR.Q算法，改进模型基表示学习，结合互信息最大化与衰减优先经验回放，用于连续控制任务。

**💡 创新点**

创新点包括：① 在模型基表示学习中加入InfoNCE互信息约束，提升状态-动作与下一状态表示之间的信息量；② 设计衰减优先经验回放（faded PER），兼顾TD误差与经验新旧，缓解训练偏置。

**🔧 技术方法**

使用了InfoNCE互信息约束、faded PER、双Q网络+目标网络的离线actor‑critic、MDP预测器、短期动力学滚动等技术。

**📊 数据集**

使用73个连续控制任务，涵盖MuJoCo、DMControl（感知与视觉）和HumanoidBench。

**📈 对比分析**

与SimBaV2、MR.Q、FoG、TD3等强基线对比，DR.Q在73个任务中普遍匹配或超越基线，尤其在DMC-Hard、HumanoidBench等任务表现显著提升。

**⚠️ 局限性**

局限性：统一超参数导致对某些任务表现不佳；在视觉DMControl或离散动作任务未测试；InfoNCE与faded PER带来额外计算；未处理硬探索或非马尔可夫任务。

---

## 437. GRAFT: Graph-Tokenized LLMs for Tool Planning

**arXiv ID:** 2605.11706 | [PDF](https://arxiv.org/pdf/2605.11706v1)

**作者:** Xinyi Gao `[一作]` (University of Queensland), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 17919 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GRAFT 框架，用图标记化方法将有向工具依赖图内化进 LLM，并通过对策略的 on‑policy 监督实现可靠的多步工具规划。

**💡 创新点**

创新点在于：①将每个工具映射为专属 token 并学习图中的指向边；②通过图‑token 边重构损失让模型内化依赖关系；③引入子任务意识的 on‑policy 工具上下文蒸馏，以缓解暴露偏差。

**🔧 技术方法**

主要技术包括：图‑token化、图‑token 边重构对比学习、子任务‑工具对齐训练、on‑policy 蒸馏、SFT+LoRA 微调和温度调节的相似度投影。

**📊 数据集**

在 HuggingFace、Multimedia、UltraTool 与 ToolBench 四个公开工具图数据集上进行实验，涵盖 AI 任务、跨模态、多领域与实际业务场景。

**📈 对比分析**

与 BeamSearch、PLaG、GNN4Plan、GTool、ToolGen 等基线比较，GRAFT 在 EM、ACPL 上实现最优表现，ELR 与图搜索方法持平，同时显著降低 hallucination 率，体现更高的规划准确性和可靠性。

**⚠️ 局限性**

局限性包括：仅支持单回合预执行规划，缺乏工具执行反馈；对子任务标注依赖较大，无法直接在交互式 agent 环境中实时修正计划。

---

## 438. Partial Model Sharing Improves Byzantine Resilience in Federated Conformal Prediction

**arXiv ID:** 2605.11684 | [PDF](https://arxiv.org/pdf/2605.11684v1)

**作者:** Ehsan Lari `[一作]` (Norwegian University of Science and Technology), Stefan Werner `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 8083 | [OpenAlex ID](https://openalex.org/A5059938646)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种通过部分模型共享实现的拜占庭鲁棒联邦置信预测方法。

**💡 创新点**

将PSO-Fed的部分共享机制与Rob-FCP结合，既能在训练阶段抑制毒化，也能在校准阶段通过直方图特征向量和恶意度评分检测并排除拜占庭客户端；同时显著降低通信成本。

**🔧 技术方法**

部分参数共享（PSO-Fed）、直方图特征向量压缩、基于距离的恶意度评分、Rob-FCP框架。

**📊 数据集**

使用100个非IID客户端的合成回归数据集，模型维度50，随机选取30%参数进行共享。

**📈 对比分析**

与标准FCP和Rob-FCP进行对比；实验表明在覆盖率约为90%的目标下，提出的方法在保持接近名义覆盖率的同时，使预测区间宽度比Rob-FCP紧缩约1.8倍、比FCP紧缩约2.2倍，并且通信量更低。

**⚠️ 局限性**

缺乏理论证明部分共享对鲁棒性和覆盖率的具体影响；实验仅在合成数据上验证，尚未在真实世界任务中检验。

---

## 439. WorldComp2D: Spatio-semantic Representations of Object Identity and Location from Local Views

**arXiv ID:** 2605.11743 | [PDF](https://arxiv.org/pdf/2605.11743v1)

**作者:** SeongMin Jin `[一作]`, Doo Seok Jeong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 WorldComp2D，一个轻量级框架，利用局部视角学习结构化的空间语义潜在空间，并通过局部化器定位对象。

**💡 创新点**

在潜在空间显式编码对象身份与视点中心的空间接近度，并采用邻近加权对比学习与多尺度感受野实现高效的空间语义推理。

**🔧 技术方法**

多尺度卷积编码器、L2 归一化潜在向量、邻近加权对比损失、聚合式局部化器及可选热图细化模块。

**📊 数据集**

COFW、300W、AFLW 面部关键点数据集。

**📈 对比分析**

与 SOTA 轻量模型相比，参数量和 FLOPs 分别减少 4×、2.2×；在 CPU 上实现 78 FPS 以上，COFW/300W 的定位精度保持可接受水平，在 AFLW 与部分 SOTA 竞争力相当。

**⚠️ 局限性**

验证范围局限于 2D 面部关键点，视点策略固定，尚未充分证明可扩展到多物体、3D 或更大尺度变化。

---

## 440. Online Continual Learning with Dynamic Label Hierarchies

**arXiv ID:** 2605.11742 | [PDF](https://arxiv.org/pdf/2605.11742v1)

**作者:** Xinrui Wang `[一作]` (Nanjing University of Aeronautics and Astronautics), Songcan Chen `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 13825 | [OpenAlex ID](https://openalex.org/A5101596072)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的在线持续学习设置（DHOCL），并设计了HALO框架来同时处理动态层次标签和流式数据的学习挑战。

**💡 创新点**

创新点在于①将动态层次标签引入在线持续学习场景，②使用可学习的层次原型（Hierarchical Prototypes）对特征空间进行结构化约束，③采用双头（线性+解析）分类器与预测级聚合（PredLA）解决层次不同级别的稳定性与可塑性冲突。

**🔧 技术方法**

采用的技术包括层次原型正则化（HPR）与余弦相似度聚合、两种预测器（可训练线性头与冻结特征解析头）以及温度缩放与权重聚合的二层优化策略。

**📊 数据集**

使用的公开数据集有 CIFAR-100、FGVC-Aircraft、CUB-200、iNaturalist 以及 ImageNet-H，所有数据集均按 DHOCL 的流式、随机层次标签方式划分。

**📈 对比分析**

与基于经验回放和正则化的传统 OCL 方法（如 RS、EWC++、LwF、ACIL、GACIL 等）以及层次分类方法（HAF、HCon 等）进行对比，HALO 在 AAUC、FAUC、误差严重性 MS、细粒度精度等多项指标上均显著优于对照组，显示出更好的性能与更强的稳定性。

**⚠️ 局限性**

局限性包括：①对极大规模或高度不平衡的层次结构的适应性尚待验证；②虽然对层次噪声具有一定鲁棒性，但对结构大幅破坏仍会显著下降；③实现中引入的多头与原型模块带来额外的计算与存储开销。

---

## 441. Five Attacks on x402 Agentic Payment Protocol

**arXiv ID:** 2605.11781 | [PDF](https://arxiv.org/pdf/2605.11781v1)

**作者:** Zelin Li `[一作]` (Ohio State University), Zhipeng Wang `[通讯]` (University of Manchester)

**通讯引用:** 482 | [OpenAlex ID](https://openalex.org/A5100424172)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统地分析并验证了 x402 微支付协议在设计与实现层面的安全缺陷，提出并演示了五类攻击（回滚授权、预支授权、重放、HTTP/代理混淆、服务选择攻击），并基于可复现的测试平台、公开测试网与三类主流 SDK 进行实验验证与代码审计。

**💡 创新点**

创新点在于：① 构建了 x402 的形式化安全模型和四个核心安全属性；② 提出了从协议、实现到部署的完整攻击链与量化指标；③ 将攻击结果与实时链重组、缓存泄漏等跨层面风险结合，首次量化了 HTTP‑链之间的时序不一致性；④ 结合 LLM 代理的发现层攻击，展示了发现层对支付安全的影响。

**🔧 技术方法**

使用技术包括：形式化安全模型（PPT adversary、probabilistic inclusion/finality）、EIP‑712 结构化签名、EIP‑3009/Permit2 付款执行、基于 Hardhat/Anvil 的本地区块链重组注入、Base Sepolia 的真实链实验、nginx/Caddy/自定义 MITM 的 HTTP 中间件测试、LLM（MiniMax‑M2.7、GPT‑5.3、Sonnet‑4.5）在 Bazaar‑style 发现层的评估。

**📊 数据集**

实验数据集包括：① 5,000 条本地实验请求（重组/延迟多维度），② Base Sepolia 上 1,000 条并发重放请求、1,000 条缓存测试请求、1,000 条预支授权验证请求；③ 12 类别、15 个查询、3 个 LLM 的 2,160 次发现决策；④ 3 个 SDK（TypeScript、Python、Rust）与 4 个公开测试网端点的审计日志。

**📈 对比分析**

比较方法：对比不同确认深度 k、不同重组概率 p_reorg、不同网络延迟 δ 下的 revert‑grant 率 RGP_k 与授予时延 T_gf；对比是否实现 idempotency 的重放次数 DGR；对比不同代理缓存策略的缓存泄漏率；对比原始与 Sybil 攻击后的服务选择率。结果表明：optimistic 执行下 RGP_0 可达 5.18%，提高 k 可降至 <1%；重放攻击在无 idempotency 时 DGR 可达 50 倍；nginx 在未加 no-store 时导致 100% 的支付缓存泄漏；Sybil 攻击在 5 个伪服务器时服务选择率可达 60%。

**⚠️ 局限性**

局限性：① 重组实验使用了本地模拟与参数化模型，未覆盖真实链长期重组分布；② 只评估了部分 SDK 与公开端点，未包含所有可能的实现变体；③ 未对完全恶意服务器、协同攻击以及跨服务链的多代理场景进行建模；④ 仅关注 x402 协议本身，对更高层次的应用逻辑与经济治理未进行深入分析。

---

## 442. RevealLayer: Disentangling Hidden and Visible Layers via Occlusion-Aware Image Decomposition

**arXiv ID:** 2605.11818 | [PDF](https://arxiv.org/pdf/2605.11818v1)

**作者:** Binhao Wang `[一作]` (Wenzhou University), Yuhui Yin `[通讯]` (360 AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于Diffusion的可控层级图像分解框架RevealLayer，将RGB图像分解为背景层和若干透明RGBA前景层，并利用用户给定的边界框进行控制。

**💡 创新点**

创新点包括：① Region-Aware Attention实现区域级别的层级解耦；② Occlusion-Guided Adapter利用原图上下文完成重叠区域的遮挡恢复；③ 复合损失（硬Alpha损失+正交性损失）提升边界清晰度并抑制残留伪影；④ 构建规模达10万张的自然场景多层图像数据集RevealLayer-100K，并推出Benchmark。

**🔧 技术方法**

技术栈主要基于FLUX.1的MM-DiT架构，结合VAE编码器、3D-RoPE位置编码、Region-Aware Attention、Occlusion-Guided Adapter、流匹配损失、Alpha损失和正交性损失，并使用LoRA微调、Prodigy优化器。

**📊 数据集**

使用了LAION-2B、GRIT-20M等大规模自然图像作为原始素材，并通过Qwen3-VL、SAM、Qwen-Image-Edit、ViTMatte等工具进行自动化提取与人工校验，最终形成RevealLayer-100K数据集；Benchmark为RevealLayerBench。

**📈 对比分析**

在对象移除、图像抠图与多层分解等任务上，RevealLayer在PSNR、SSIM、LPIPS、FID、SoftIoU等指标上均优于现有方法（如Qwen-Image-Layered、CLD、OmniPSD、LayerD等），并在人类评估中获得最高的层级控制、背景质量与前景完整度分数。

**⚠️ 局限性**

局限性主要体现在：对极端误差的边界框敏感；高遮挡、透明/半透明区域以及重复纹理的场景下分解和遮挡恢复仍有挑战；模型训练与推理成本相对较高，需进一步优化。

---

## 443. Trade-offs in Decentralized Agentic AI Discovery Across the Compute Continuum

**arXiv ID:** 2605.11839 | [PDF](https://arxiv.org/pdf/2605.11839v1)

**作者:** Patrizio Dazzi `[一作]` (University of Pisa), Saul Urso `[通讯]` (University of Pisa)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在计算连续体中比较了Chord、Pastry和Kademlia三种分布式哈希表（DHT）作为智能体目录的去中心化发现子系统，评估其可靠性、启动开销和控制平面成本。

**💡 创新点**

创新点在于从结构化覆盖层族层面对比DHT，揭示在统一工作负载和网络假设下，三种族群在可靠性‑成本权衡上的差异，而非单个实现的细节。

**🔧 技术方法**

采用了统一的Agentic系统工作负载模型（50种技能、每个代理发布一种技能），在共享的控制平面框架下实现了Chord、Pastry、Kademlia三种协议，并在相同的发布TTL、复制因子和重发布周期下运行。

**📊 数据集**

使用了规模为4096节点的静态对比和代表性流失基准两套实验，其中流失基准设定平均会话时长100、停机时长30，模拟了持续的节点加入与离开。

**📈 对比分析**

通过对发现成功率、召回率、P95延迟和消息/查询的观测与查询仅GET两种成本度量进行比较，结果显示：所有族群在冷启动时均表现出显著性能下降；短暂的log₂N热身后，Pastry成本最低、Chord居中、Kademlia在低尾延迟上表现最好但通信成本最高。

**⚠️ 局限性**

局限性包括仅评估了三种基本族群的基线实现，未覆盖更丰富的发布/查询策略、不同网络拥塞或丢包场景，且只考察了单一流失点，未来需扩展工作负载多样性、容错模型与协议调优来验证结论的普适性。

---

## 444. More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing

**arXiv ID:** 2605.11836 | [PDF](https://arxiv.org/pdf/2605.11836v1)

**作者:** Xin Ma `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 29159 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究终身模型编辑（LME）中 Lifelong Normalization（LN）的机制，提供理论解释，并基于此提出一种新的编辑器 StableEdit。

**💡 创新点**

创新点包括：①给出了 LN 的第一套理论框架，证明其形成自强化稳定循环并产生渐近正交、范数有界的参数更新；②利用这一理论设计了 Warm‑up 阶段和全白化（full whitening）来进一步提升长期编辑的稳定性；③系统验证了 LN 对正向累积效应（early edits 促进后续 edits）的作用。

**🔧 技术方法**

主要技术手段：递归贝叶斯跟踪估计梯度分布、LN 归一化、岭回归求解参数更新、全白化预处理，以及对更新几何（正交性、范数）的理论与实验分析。

**📊 数据集**

使用的数据集包括 ZsRE、FEVER、ULTRAEDITBENCH、WikiBigEdit 以及医学基准 MedCF；模型涵盖 GPT‑J‑6B、Mistral‑7B‑v0.3、Llama‑3‑8B‑Instruct 和 Qwen2.5‑7B‑Instruct 等大语言模型。

**📈 对比分析**

通过 Efficacy、Generalization、Specificity 以及 GLUE 任务等指标与多种基线（FT、MEMIT、AlphaEdit、MEND、MALMEN、RLEdit、ULTRAEDIT）进行对比。实验表明 StableEdit 在标准规模和百万级编辑序列上均优于或接近最佳方法，同时保持原模型的泛化能力；在运行时间上仅略高于 ULTRAEDIT，整体开销可忽略。

**⚠️ 局限性**

局限性：①Warm‑up 需要一定数量的初始编辑样本，且对分布漂移的鲁棒性尚需进一步验证；②虽然理论证明了更新正交与范数有界，但对极端非高斯梯度分布的适用性未知；③在极大模型（数十亿参数）上的计算成本与内存需求未完全评估；④模型内部知识分布的长期一致性仍是待解决的问题。

---

## 445. Polar Complexity: A New Descriptive Complexity with Applications to Source and Joint Source-Channel Coding

**arXiv ID:** 2605.11826 | [PDF](https://arxiv.org/pdf/2605.11826v1)

**作者:** Xinyuanmeng Yao `[一作]` (Ningbo University of Technology), Xiao Ma `[通讯]` (Sun Yat-sen University)

**通讯引用:** 25615 | [OpenAlex ID](https://openalex.org/A5051762810)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文提出极化复杂度概念，并基于此设计了两阶段极化源编码和自适应双极化联合源-信道编码（JSCC）方案。

**💡 创新点**

创新点在于：①给出了可计算的极化复杂度定义及其通过二分搜索与低复杂度预测方法的实现；②利用极化复杂度生成的极化描述，实现无损且前缀自由的两阶段源编码；③提出通过有限的极化压缩长度（PCL）集合实现自适应JSCC，并用动态规划优化PCL集合。

**🔧 技术方法**

采用的技术包括：BBT极化结构、极化压缩与成功取消解码（SCD）/列表SCLD、通用极化压缩长度的二分搜索、极化复杂度预测的GA‑SCD、CRC校验以及动态规划进行PCL集合设计。

**📊 数据集**

实验使用的“数据集”是通过模拟二进制马尔可夫/均匀源、不同熵的二进制独立同分布（BMS）序列以及各种 Hamming 权重的固定长度序列进行评估。

**📈 对比分析**

与枚举编码、现有极化源编码以及传统双极化JSCC基线进行对比；ACL/NACL 接近源熵，FER 在相同信噪比下明显低于基线，证明了方案在短块长下的优越性能。

**⚠️ 局限性**

局限性包括：①需要预先设计并共享 PCL 集合，影响灵活性；②极化复杂度预测可能过于保守，导致冗余；③BBT 极化结构对非 2 的幂长度存在一定实现复杂度；④在极端熵或高度相关序列上性能尚未完全验证。

---

## 446. Ink Spiral: Symbolic Transformation from The Thinker to the Four Gentlemen

**arXiv ID:** 2605.11816 | [PDF](https://arxiv.org/pdf/2605.11816v1)

**作者:** Lingyu Peng `[一作]` (Harbin Institute of Technology), Qingchuan Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1293 | [OpenAlex ID](https://openalex.org/A5013804346)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过生成式 AI 将罗丹《思考者》雕塑逐帧转化为四君子（梅、兰、竹、菊）的水墨图像，构建跨文化的视频装置。

**💡 创新点**

将多模态控制网络与 LoRA 微调的扩散模型相结合，实现雕塑向自然符号的连续、可逆过渡，突破传统静态对比，体现符号之间的流动性与模糊性。

**🔧 技术方法**

使用 LoRA 微调预训练扩散模型、ControlNet 三分支控制网络（模糊图块、深度图、分割掩码）以及生成式 AI（GenAI）技术。

**📊 数据集**

基于 1000+ 张手工标注的四君子水墨画数据集进行训练与微调。

**📈 对比分析**

主要通过视觉连续性、符号映射的主观评价以及观众反应进行对比，未给出定量指标；生成过程在视觉上实现了雕塑与水墨形态的流畅转换。

**⚠️ 局限性**

受限于数据集规模、模型对复杂形变的控制精度，生成的中间形态仍可能出现形态失真、符号对应模糊等问题，且缺乏客观性能评估。

---

## 447. See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model

**arXiv ID:** 2605.11817 | [PDF](https://arxiv.org/pdf/2605.11817v1)

**作者:** Yixu Feng `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 22119 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可微分网格采样器（GridS），用于在视觉-语言-动作模型中连续、几何感知地重采样视觉标记，显著降低视觉令牌数量。

**💡 创新点**

创新点在于打破传统标记剪枝与几何信息丢失的权衡，通过自适应预测最小关键坐标并使用可微分插值保留空间细节，实现了几乎无性能损失的高压缩率。

**🔧 技术方法**

采用可微分插值的网格采样模块，配合视觉编码器实现任务感知的连续重采样，同时与现有VLA模型无缝集成。

**📊 数据集**

使用LIBERO基准数据集以及真实机器人平台进行实验验证。

**📈 对比分析**

与传统剪枝方法相比，GridS在LIBERO和实际机器人实验中将FLOPs降低了76%，且在成功率上无任何下降，达到目前报告的最低可行视觉令牌数（不足10%）。

**⚠️ 局限性**

局限性包括仅在特定VLA模型和机器人任务上验证，缺乏对不同场景、模型或更大规模数据集的泛化评估，且对极端几何变化的鲁棒性尚待进一步研究。

---

## 448. Mitigating Action-Relation Hallucinations in LVLMs via Relation-aware Visual Enhancement

**arXiv ID:** 2605.11808 | [PDF](https://arxiv.org/pdf/2605.11808v1)

**作者:** Zhenxin Qin `[一作]` (Tongji University), Wen Shen `[通讯]` (Tongji University)

**通讯引用:** 11879 | [OpenAlex ID](https://openalex.org/A5032104899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练-free框架，通过识别注意力头的动作关系敏感度并增强对动作相关图像区域的关注，从而在推理阶段直接降低大型视觉语言模型的动作关系幻觉。

**💡 创新点**

创新点包括：①定义动作关系敏感度(ARS)分数来定位中层注意力头；②构造增强掩码与去噪掩码，在预softmax得分上调整注意力，实现对动作相关区域的视觉关注提升；③方法无训练、可直接在推理阶段使用。

**🔧 技术方法**

技术手段：多头注意力分析、ARS分数计算、基于掩码的预softmax得分增强、图像区域定位；利用 GPT‑5 生成动作对比文本以构造敏感性对比样本。

**📊 数据集**

数据集：MMRel、R‑Bench、AMBER 用于动作关系幻觉评估；POPE 用于对象幻觉评估；实验还涵盖真实与合成图像子集。

**📈 对比分析**

方法与基线比较：与原始 LVLM、VCD、ICD、VAF 等训练‑free 方法对比；在动作关系、空间关系及对象幻觉任务中均获得显著提升，例如 MMRel（Real）准确率从 71.1% 提升至 81.1%；R‑Bench F1 从 87.5% 提升至 88.1%；推理速度几乎无额外成本。

**⚠️ 局限性**

局限性：需要访问模型内部层，无法在仅 API 访问的闭源模型上使用；最佳增强层和掩码阈值需要针对不同模型手动调参，缺乏自动化选择机制；整体效果仍受基础模型能力限制。

---

## 449. Advancing Dynamic Ride-Pooling Simulation -- A Highly Scalable Dispatcher

**arXiv ID:** 2605.11798 | [PDF](https://arxiv.org/pdf/2605.11798v1)

**作者:** Moritz Laupichler `[一作]` (Karlsruhe Institute of Technology), Peter Vortisch `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 2569 | [OpenAlex ID](https://openalex.org/A5014371806)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种可扩展到每小时数百万请求的动态拼车调度器，支持会议点和多线程批量处理。

**💡 创新点**

创新点在于将高效的短路搜索（CH、BCH、椭圆修剪）与批量并行冲突解决相结合，实现了大规模在线匹配与实时调度。

**🔧 技术方法**

采用了卷积层级路由、Bucket Contraction Hierarchies、SIMD加速、线程池并行、离散选择模式和多源交通网络的统一查询。

**📊 数据集**

使用了德国卡尔斯鲁厄、斯图加特的mobiTopp需求数据以及美国洛杉矶的MATSim需求数据，结合真实道路网络与步行网络。

**📈 对比分析**

与单线程原始实现对比，利用96核可实现约30–36倍加速，单请求响应时间保持在约1秒；质量评估显示可获得约40%拼车占比、平均乘客数1.7、系统效能1.5倍以上，且会议点显著降低等待与行程偏差。

**⚠️ 局限性**

局限在于批量冲突导致部分质量下降、Bucket结构并发访问造成缓存失效、未实现动态路况更新与车辆重新分配、且仅采用贪心插入算法，缺乏全局最优匹配。

---

## 450. NavOL: Navigation Policy with Online Imitation Learning

**arXiv ID:** 2605.11762 | [PDF](https://arxiv.org/pdf/2605.11762v1)

**作者:** Xiaofei Wei `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**通讯引用:** 82176 | [OpenAlex ID](https://openalex.org/A5100425671)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 NavOL 的在线模仿学习框架，用于视觉导航，通过在仿真环境中实时与全局规划器交互，收集专家轨迹并在 roll‑up‑update 循环中更新扩散式策略。

**💡 创新点**

创新点在于将 DAgger 风格的在线收集与扩散模型相结合，构建了高效的 rollout‑update 循环，消除了离线数据偏移和强化学习奖励设计的痛点，实现了数据高效、鲁棒性强的导航策略。

**🔧 技术方法**

采用了扩散 Transformer 策略与 critic 网络、IsaacLab 的 GPU 并行仿真、NavMesh 全局规划、MPC 低级控制、域随机化与数据增强等技术。

**📊 数据集**

使用了 3D‑Front 资产构建的室内导航基准（8 个场景、100 个 start‑goal 对），NavDP 生成的数据集以及在 Unitree Go2 上的真实世界实验数据。

**📈 对比分析**

与 NavDP、DD‑PPO、iPlanner、ViPlanner 等基线进行零射（zero‑shot）比较，NavOL 在 SR（成功率）和 SPL（成功加权路径长度）上均实现了显著提升，单 GPU 训练 2 天即可超越 NavDP；在真实环境中也表现出更高的成功率。

**⚠️ 局限性**

局限性包括训练阶段需依赖全局规划器和高性能 GPU 并行仿真，对更大规模或多机器人场景的可扩展性尚未验证，且在极端动态或高度复杂环境下的鲁棒性需要进一步测试。

---

## 451. Maximum Entropy of Sums of Independent Ternary Random Variables

**arXiv ID:** 2605.11831 | [PDF](https://arxiv.org/pdf/2605.11831v1)

**作者:** Mladen Kovačević `[一作]` (University of Novi Sad), Mladen Kovačević `[通讯]` (University of Novi Sad)

**通讯引用:** 354 | [OpenAlex ID](https://openalex.org/A5014784584)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了在独立的三值随机变量求和时，Shannon熵的最大化分布为前 n-1 个变量在 {0,2} 上均匀分布，最后一个变量为 {0,2} 与 1 的混合分布。

**💡 创新点**

创新点在于将 Shepp–Olkin–Mateev 二元情形推广到三元字母，并首次用 Hermite–Biehler 与 Newton 不等式结合 Yu 的最大熵定理来完成证明。

**🔧 技术方法**

主要技术包括 Hermite–Biehler 定理、Newton 不等式、超对数凸性（ultra‑log‑concavity）以及 Yu 的最大熵定理。

**📊 数据集**

本文不涉及实验数据，使用的是理论推导和符号计算。

**📈 对比分析**

没有实验比较；该工作通过严谨的数学证明给出了熵上界，并给出最优分布的构造，证明了该界可被达到。

**⚠️ 局限性**

局限性在于所用方法仅适用于三元字母，无法推广至更大字母表；对 r ≥ 3 的情况仍是未解的挑战。

---

## 452. Fed-BAC: Federated Bandit-Guided Additive Clustering in Hierarchical Federated Learning

**arXiv ID:** 2605.11815 | [PDF](https://arxiv.org/pdf/2605.11815v1)

**作者:** Satwat Bashir `[一作]` (London South Bank University), Muddesar Iqbal `[通讯]` (London South Bank University)

**通讯引用:** 2779 | [OpenAlex ID](https://openalex.org/A5074431197)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为Fed-BAC的三层边缘计算联邦学习框架，能够在非IID环境下通过联合优化服务器聚类与客户端选择来提升分布式模型准确性。

**💡 创新点**

创新点在于将聚类个性化与两层多臂赌博机（云层LinUCB、边缘层Thompson Sampling）结合，实现动态服务器分组与智能客户端采样，并通过加性模型拆分实现跨群共享与个性化。

**🔧 技术方法**

使用的技术包括加性模型拆分（Fed-CAM原理）、上下文线性UCB、Thompson Sampling、LeNet-5深度网络、Dirichlet分布数据异构化以及两阶段聚合策略。

**📊 数据集**

实验基于三大公开图像数据集：CIFAR-10、SVHN和Fashion-MNIST，分别在α=0.5和α=0.1两种非IID程度下进行验证。

**📈 对比分析**

与HierFAVG和IFCA进行对比，Fed-BAC在30%~35%的分布式准确率提升、1.5~4.8倍更快收敛以及跨服务器公平性显著改善，且在5×规模扩展下仍保持优势。

**⚠️ 局限性**

局限包括对参与率p的探索不足、对客户端离线或攻击场景的鲁棒性未评估、以及缺乏理论收敛证明。

---

## 453. Automated Reformulation of Robust Optimization via Memory-Augmented Large Language Models

**arXiv ID:** 2605.11813 | [PDF](https://arxiv.org/pdf/2605.11813v1)

**作者:** Jinbiao Chen `[一作]` (National University Of Singapore), Hanzhang Qin `[通讯]` (National University Of Singapore)

**通讯引用:** 192 | [OpenAlex ID](https://openalex.org/A5067909390)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 AutoRO-Bench benchmark 与 AutoREM 框架，利用 LLM 自动完成鲁棒优化（RO）模型的 reformulation（从不确定模型到可解的确定对偶），并实现了无参数、离线自适应的经验记忆机制。

**💡 创新点**

创新点包括：1）自动化数据生成与验证的 RO reformulation benchmark；2）设计了四大组件（ULE、SMO、DCC、VBA）的离线自适应流程，构建高质量、可验证的经验记忆；3）实现了可跨模型迁移的记忆增广方案，显著提升了 LLM 在数学推理任务中的可靠性。

**🔧 技术方法**

主要技术手段为：LLM（DeepSeek‑V4‑Flash 等）与固定强编码器（DeepSeek‑V4‑Pro），结构化记忆增广（UM、SMO 等），离线反射式自适应（DCC、VBA），以及自动化 RO 生成与求解流水线。

**📊 数据集**

使用数据集为：AutoRO‑Bench 生成的三组分布数据（Random、Hard、Large）和 32 个自然语言描述的 RO 应用实例。

**📈 对比分析**

通过与基线（Base LLM、Max Thinking、Expert Prompt、ReasoningBank、ACE）以及现有 formulation 方法（AlphaOPT、LEAN‑LLM‑OPT、OptiTree）的比较，AutoREM 在 In‑distribution 数据集上准确率 97.4%（提升 10%+），Out‑of‑distribution 达 94.8%/85.4%，Token 数量平均减少约 20%，且跨 GPT‑5.4、Qwen3.6‑Plus 等模型迁移后仍保持 6–7% 的准确率提升，应用任务上达 81.3% 对比最佳基线 75%。

**⚠️ 局限性**

主要局限是离线自适应依赖 solver 验证的目标值，若缺失目标值或面临更复杂的 RO（非 LP 对偶）则难以直接应用；同时记忆增广仅针对 LP 级别的 robust counterpart，未覆盖更高级的模型类型。

---

## 454. ROMER: Expert Replacement and Router Calibration for Robust MoE LLMs on Analog Compute-in-Memory Systems

**arXiv ID:** 2605.11800 | [PDF](https://arxiv.org/pdf/2605.11800v1)

**作者:** Wenyong Zhou `[一作]` (University of Hong Kong), Ngai Wong `[通讯]` (University of Hong Kong)

**通讯引用:** 12519 | [OpenAlex ID](https://openalex.org/A5043990959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了混合专家LLM在模拟计算内存硬件下受噪声影响的鲁棒性，并提出ROMER校准框架来恢复专家负载平衡和路由稳定性。

**💡 创新点**

提出无训练、无标注的ROMER方法，包括专家替换和分位数归一化路由校准，显著降低噪声导致的困境。

**🔧 技术方法**

基于真实芯片噪声校准的混合噪声模型、专家激活统计、分位数压缩路由、权重复制与归一化等技术。

**📊 数据集**

使用WikiText‑103/2、LAMBADA、PIQA、ARC‑Easy/Challenge等公开数据集进行实验。

**📈 对比分析**

与Clean、Vanilla、Bit‑slicing、Average、Weight‑remapping、k‑b calibration等六种基线对比，ROMER在不同模型和温度下将困惑度下降约56‑60%，并仅增加约6‑12%延迟/功耗。

**⚠️ 局限性**

仍受高温噪声影响，ROMER对极高温度或极大模型需要更多专家替换；方法仅针对专家级别，未解决所有潜在路由失配问题。

---

## 455. SB-BEVFusion: Enhancing the Robustness against Sensor Malfunction and Corruptions

**arXiv ID:** 2605.11799 | [PDF](https://arxiv.org/pdf/2605.11799v1)

**作者:** Markus Essl `[一作]`, Markus Schedl `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种单分支 BEV 融合模块 SB-BEVFusion，能够在摄像头或 LiDAR 缺失或受损时仍保持良好的 3D 检测性能。

**💡 创新点**

创新点在于融合模块自动在两种模态缺失情况下退化为单模态处理，并通过在训练时随机混合 L+C、L、C 场景来提升鲁棒性。

**🔧 技术方法**

使用 BEV 投影、无权重平均/最大池化/交叉注意力/进步模态衰减等融合算子，并在共享的下游 BEV 编码器中进行训练。

**📊 数据集**

实验基于公开的 nuScenes 数据集以及为评估鲁棒性构造的 MultiCorrupt 数据集进行。

**📈 对比分析**

与 BEVFusion、UniBEV 等现有方法比较，SB-BEVFusion 在干净数据上 mAP 达 0.6737，在缺失或受损场景下的平均鲁棒性得分 (mRA) 高达 0.7683，明显优于对比模型。

**⚠️ 局限性**

仅在摄像头和 LiDAR 两种模态上验证，未测试雷达等其他模态，且跨模态 Transformer 融合方法仍需进一步提升。

---

## 456. Psychological Benefits and Costs of Diversifying Algorithmic Recourse

**arXiv ID:** 2605.11793 | [PDF](https://arxiv.org/pdf/2605.11793v1)

**作者:** Tomu Tominaga `[一作]` (NTT, Inc.), Takeshi Kurashima `[通讯]` (NTT, Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在汽车贷款拒贷情境中，通过对受试者进行 5 种 recourse 集合（Close‑1、Close‑3、Close‑7、Diverse‑3、Diverse‑7）的随机对照实验，研究了 recourse 多样性和集合大小对心理收益（可行性、可理解性、行动意愿、决策接受度）和心理成本（认知负荷、情绪负担）的影响。

**💡 创新点**

首次系统性实证揭示了 recourse 多样化的规模相关权衡：在集合规模较小（k=3）时多样化能提升行动意愿而不增加成本；但在集合规模较大（k=7）时则显著提升认知负荷；并且多样化并未加重情绪负担，挑战传统预期。

**🔧 技术方法**

主要技术包括：
• 基于最近距离与平均两两距离的双目标优化（Close 与 Diverse 条件）生成 counterfactual 方案；
• 通过 NASA‑TLX 量化认知负荷；
• 采用自述题与开放式理由进行定量与定性评估；
• 统计分析采用两因素 ANOVA、Tukey、Dunnett 以及开放式文本编码。

**📊 数据集**

使用了 4,057 条真实贷款申请者特征数据作为 counterfactual 生成的候选集合；实验样本为 750 名受试者（筛选后来自日本在线问卷平台，年龄约 49 岁，男女比例约 1:2）。

**📈 对比分析**

比较方法：在每个组别下对 4 个心理收益指标与 3 个心理成本指标进行两因素 ANOVA 并做事后检验；结果显示：
- 对可行性、可理解性、决策接受度，Close 与 Diverse 在 k=3 时无显著差异；
- 对行动意愿，Diverse‑3 显著优于 Close‑3（p=0.002），但在 k=7 时无显著差异；
- 对认知负荷，Diverse‑7 的 Mental Demand、Effort 与 Frustration 明显高于 Close‑7（p<0.01）；
- 对情绪负担，Diverse 在所有规模下均低于 Close（p<0.01）。

**⚠️ 局限性**

限制：
- 仅测试 k=1、3、7，缺乏中间规模的细粒度；
- 研究仅限汽车贷款情境，难以直接推广至其他高风险决策领域；
- 样本主要为日本男性，文化与性别差异可能影响结果；
- 未对多样化算法的具体实现细节（如正则化、硬约束）进行系统比较；
- 结果基于实验设定，真实用户情境下的长期行为尚未验证。

---

## 457. A nonlinear extension of parametric model embedding for dimensionality reduction in parametric shape design

**arXiv ID:** 2605.11759 | [PDF](https://arxiv.org/pdf/2605.11759v1)

**作者:** Andrea Serani `[一作]` (National Research Council), Matteo Diez `[通讯]` (National Research Council)

**通讯引用:** 2542 | [OpenAlex ID](https://openalex.org/A5047085967)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并验证了一种非线性 Parametric Model Embedding (NLPME)，实现形状设计空间的高效降维。

**💡 创新点**

在保持 PME 的几何驱动潜变量和参数介导重建结构的同时，引入非线性编码器/解码器并使用可微前向参数-几何映射或 surrogate，显著提升压缩效率。

**🔧 技术方法**

结构化非线性 autoencoder（encoder+decoder+surrogate），梯度下降训练几何一致性损失，比较线性 PME 与深度自编码器。

**📊 数据集**

使用仿生水下滑翔机的 32 维参数化形状，Sobol 采样生成 16,385 条样本，过滤后 7,467 条有效几何。

**📈 对比分析**

采用统一的几何重建均方误差指标 ϵ(N) 进行比较；NLPME 在低维（N=5）即可达到 5% 误差（线性 PME 需要 N=8），在 1% 误差下 NLPME 仅需 N=9 而线性 PME 需要 N=15；与 DAE 比较，NLPME 维持了参数映射优势，压缩效果接近 DAE。

**⚠️ 局限性**

仅在固定拓扑、单一形状族上验证；未评估在优化或仿真中的实际效益；依赖 surrogate 近似前向映射；未加入参数一致性或正则化项；需进一步验证跨形状族的泛化能力。

---

## 458. Connectivity augmentation is fixed-parameter tractable

**arXiv ID:** 2605.11757 | [PDF](https://arxiv.org/pdf/2605.11757v1)

**作者:** Tuukka Korhonen `[一作]` (University of Copenhagen), Mikkel Thorup `[通讯]` (University of Copenhagen)

**通讯引用:** 14061 | [OpenAlex ID](https://openalex.org/A5039232562)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图的连通性增强问题，并给出了在参数λ和k下可解的固定参数可行算法，分别针对顶点连通性和边连通性提出运行时间为2^(k log(k+λ))n和2^(k log k)n的算法；

**💡 创新点**

创新点在于通过构造“叶状”区域和利用树分解、Gomory‑Hu树以及Ramsey/鸽笼原理，提出了新的不可忽略链子（irrelevant link）判定方法，使得在无先验连通性假设的情况下取得更一般的FPT结果；

**🔧 技术方法**

主要技术包括树分解与无分断性（unbreakability）理论、最大流/最小割算法、子模性质、以及对大规模链子集合进行组合化简的 Ramsey 论证；

**📊 数据集**

论文没有使用实验数据集，所有结果均为理论证明和算法复杂度分析；

**📈 对比分析**

与之前仅在λ≤4或特定连通性条件下可行的算法相比，本工作在理论上显著提升了参数范围，提供了更紧凑的时间上界；

**⚠️ 局限性**

局限在于顶点连通性增强的算法仍需依赖λ，且对仅按k参数化的FPT是否可行尚未解决；此外，在加权链子最小化问题中，算法对大权值的处理仍未知。

---

## 459. REFNet++: Multi-Task Efficient Fusion of Camera and Radar Sensor Data in Bird's-Eye Polar View

**arXiv ID:** 2605.11824 | [PDF](https://arxiv.org/pdf/2605.11824v1)

**作者:** Kavin Chandrasekaran `[一作]` (Elektrobit Automotive GmbH), Pavol Jancura `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 136 | [OpenAlex ID](https://openalex.org/A5009464865)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 REFNet++ 融合网络，能够在统一的鸟瞰极坐标 (BEV Polar) 域内同时完成车辆检测和自由空间分割，并支持单任务模式。

**💡 创新点**

创新点在于：
- 采用变分编码解码器（VAE）隐式学习相机前视图到 BEV 极坐标的变换，省去了手工几何预处理；
- 同时对雷达的复杂 Range‑Doppler (RD) 光谱进行角度恢复，得到 Range‑Azimuth (RA) 特征；
- 将两模态在同一 BEV 域对齐后按通道拼接，避免了传统早期/晚期/多级融合所需的多级对齐和特征映射；
- 在保持高精度的同时显著降低参数量、GPU 内存占用和推理时延。

**🔧 技术方法**

使用技术包括：
- 变分自编码器（VAE）+残差网络 + FPN；
- 复杂 RD 预编码与通道交换以提取角度信息；
- 采用交叉熵、焦点损失、平滑 L1 损失、BCE 损失组合的多任务损失；
- Adam 优化器、学习率衰减、Reparameterization trick；
- 采用 bilinear 插值对齐特征尺寸。

**📊 数据集**

使用公开的 RADIal 数据集（同步摄像头、雷达、LiDAR + GPS），共计约 25,000 帧，其中 8,252 帧标注了 9,550 辆车辆。

**📈 对比分析**

与多种基准（REFNet、EchoFusion、ROFusion、FFTRadNet、TFFTRadNet、ADCNet、SparseRad 等）进行对比：
- 车辆检测：AP/AR/F1、RE/AE 均位居前列，雷达+相机融合的 REFNet++ 在 AP、AR 及 F1 上仅次于 EchoFusion；
- 自由空间分割：mIoU 最高（单任务 88.13%/多任务 87.58%），超过 REFNet、TransRadar 等；
- 计算效率：参数量、FPS、GPU 内存均优于大多数多模态模型，单任务模式 FPS 最高，模型大小与 GPU 内存最小；
- 进一步实验显示，去除 VAE reparameterization 或仅使用相机模态会显著降低精度。

**⚠️ 局限性**

局限性：
- 只实现了车辆检测，未扩展到其他类别；
- 仅融合雷达和摄像头，未加入 LiDAR 或激光雷达的进一步提升；
- 对极端天气或时序同步误差的鲁棒性尚未系统评估；
- 模型虽然已压缩，但在嵌入式平台的实时性能仍需进一步验证。

---

## 460. Beyond World-Frame Action Heads: Motion-Centric Action Frames for Vision-Language-Action Models

**arXiv ID:** 2605.11809 | [PDF](https://arxiv.org/pdf/2605.11809v1)

**作者:** Huoren Yang `[一作]` (Xi'an Jiaotong University), Yihong Gong `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 23523 | [OpenAlex ID](https://openalex.org/A5100687952)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

重新设计Vision‑Language‑Action（VLA）模型的动作输出端，提出MCF‑Proto，通过预测一个运动中心帧（MCF）并在该帧中使用共享原型字典生成动作，从而实现更稳健、更可组合的低级控制。

**💡 创新点**

核心创新在于：①在动作输出端加入一个学习的局部坐标帧，使得不同场景下相同的交互意图在本地坐标系中对齐；②在该帧中使用原型词典和门控机制进行动作合成，压缩动作维度并提高可复用性；③通过正交正则和时间平滑正则进一步强化框架的结构化与一致性。

**🔧 技术方法**

技术方法包括：6D连续旋转参数化预测局部帧；共享的平移和旋转原型矩阵；soft‑max门控与线性组合产生本地动作增量；正交正则化保证原型不冗余；时序平滑正则鼓励帧的平滑演化；使用行为克隆（BC）在标准演示数据上端到端训练。

**📊 数据集**

主要数据集：LIBERO 基准（Spatial、Object、Goal、Long）及其扩展的 LIBERO‑plus（7 种扰动类型）；真实机器人 OpenArm 上的三组长周期操纵任务。

**📈 对比分析**

与多种 VLA 以及机器人策略基线（OpenVLA、π0、π0‑fast、FLOWER、GR00T‑N1.5、BEAST 等）进行对比。 在 LIBERO 上平均成功率 97.7% 超过所有基线（最高 97.1% 的 OpenVLA‑OFT）。 在 LIBERO‑plus 上在 Camera、Robot、Light、Background、Noise、Layout 等 6/7 个扰动类别中均表现最好，尤其是 Camera 与 Robot 扰动显著提升。

**⚠️ 局限性**

局限性：仅在 LIBERO、LIBERO‑plus 与少量真实任务上验证；未覆盖高度动态、柔性或细粒度接触任务；局部帧仅为隐式学习的，没有语义化的坐标体系；依赖离线演示质量，未探究在线适应或更大规模训练下的表现。

---

## 461. Why Users Go There: World Knowledge-Augmented Generative Next POI Recommendation

**arXiv ID:** 2605.11807 | [PDF](https://arxiv.org/pdf/2605.11807v1)

**作者:** Qiuyu Ding `[一作]` (Amap, Alibaba Group), Mu Xu `[通讯]` (Amap, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 AWARE 框架，利用 LLM 代理生成基于用户行为的个性化热点文本，并将其注入生成式 POI 推荐模型，以捕获动态世界知识。

**💡 创新点**

创新点在于：①把 LLM 代理用于离线知识获取而非实时推理；②通过行为先验（频率、转移、周期）将外部知识个性化；③将生成式推荐与世界知识融合，提升对不可预测事件的鲁棒性。

**🔧 技术方法**

采用 LLM 代理 + 检索工具、语义 ID 与空间前缀编码、LoRA 微调的生成式模型，以及行为先验对齐技术。

**📊 数据集**

使用 Foursquare‑NYC、Foursquare‑TKY、Gowalla‑CA 三个真实 LBSN 数据集进行评估。

**📈 对比分析**

与传统、神经网络和 LLM 基线（如 PRME、GETNext、SOTAs 的 LLM 方案）对比，在 HR@1 上相较 ROS 基线提升 6.0%–12.4%，尤其在用户行为稀疏的 CA 数据集上显著获益。

**⚠️ 局限性**

局限在于知识获取仅为离线一次性生成，无法及时捕捉新事件；检索主要依赖网页，覆盖度与语言多样性有限，且缺乏在线动态更新机制。

---

## 462. Beyond Inefficiency: Systemic Costs of Incivility in Multi-Agent Monte Carlo Simulations

**arXiv ID:** 2605.11789 | [PDF](https://arxiv.org/pdf/2605.11789v1)

**作者:** Alison Moldovan-Mauer `[一作]` (Technische Hochschule Nürnberg Georg Simon Ohm), Benedikt Mangold `[通讯]` (Technische Hochschule Nürnberg Georg Simon Ohm)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过大规模Monte Carlo仿真，利用多代理LLM系统对不同毒性水平的1对1辩论进行实验，评估其对辩论收敛时间与结果的影响。

**💡 创新点**

在先前MAD 1.0基础上扩展至三种不同规模（405B、120B、24B）的LLM模型，系统验证毒性与模型规模的交互效应，并揭示“先发优势”和“毒性说服优势”的普适性。

**🔧 技术方法**

采用多代理蒙特卡罗仿真框架、persona提示、LLM生成、毒性控制（无毒、轻度、中度、重度）以及统计检验（t检验、ANOVA、binomial检验）。

**📊 数据集**

使用自定义的主题池（64个争议性话题）进行随机抽取，生成约1,000场模拟辩论，数据为LLM输出的辩论对话文本。

**📈 对比分析**

结果显示：毒性越高，收敛时间越长，幅度随模型规模递减；毒性代理赢得辩论的概率显著高于非毒性代理；起始发言者胜率显著高于50%，呈现明显的先发优势。

**⚠️ 局限性**

局限性包括：仅为1对1对话，无法反映多方讨论与社会层级影响；LLM受RLHF偏差影响；与人类交流的外推性有限；缺乏对话文本的定性分析。

---

## 463. Urban Risk-Aware Navigation via VQA-Based Event Maps for People with Low Vision

**arXiv ID:** 2605.11782 | [PDF](https://arxiv.org/pdf/2605.11782v1)

**作者:** Antoni Valls `[一作]` (Institut de Robòtica i Informàtica Industrial), Jordi Sanchez-Riera `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 479 | [OpenAlex ID](https://openalex.org/A5063541182)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视觉问题回答（VQA）的城市风险感知导航框架，利用三层层级查询和加权风险评分生成可导航的事件地图。

**💡 创新点**

创新点包括：三层层级查询结构、使用生成式多模态大语言模型（MLLM）进行风险评估、加权风险评分映射到四级安全分类，以及构建全球20城市、6大洲的多样化VQA数据集。

**🔧 技术方法**

核心技术包括：Vision‑Language 模型（ViLT、LLaVA、InstructBLIP、Qwen‑VL）、VQA 任务、生成式大语言模型、加权风险评分公式、事件地图生成与地理映射。

**📊 数据集**

使用了由Mapillary与自记录智能眼镜采集的、覆盖20个城市、6大洲的800+图像及18,000+问答的Geo‑annotated VQA 数据集。

**📈 对比分析**

通过对四个VQA模型的准确率、精确率、召回率、F1 以及风险 MAE 进行对比，Qwen‑VL 在准确率（0.77）和 F1（0.69）上表现最佳，生成式模型的 MAE_ℛ 约为 0.13，分类模型 ViLT 精度低且误报高。

**⚠️ 局限性**

主要局限是对安全关键场景（如非人行道、施工区、路面危害）的召回率偏低，模型在这些高风险情境下易误判或漏检，需要针对性的数据增强和类别平衡训练。

---

## 464. Behavioral Integrity Verification for AI Agent Skills

**arXiv ID:** 2605.11770 | [PDF](https://arxiv.org/pdf/2605.11770v1)

**作者:** Yuhao Wu `[一作]` (Palo Alto Networks), Hongliang Liu `[通讯]` (Palo Alto Networks)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种行为完整性验证（BIV）框架，用于在LLM代理技能安装前通过类型化的能力词汇表对声明行为与实际行为进行比较，生成结构化偏差证据；

**💡 创新点**

创新点在于将跨模态（代码、自然语言说明、元数据）行为一致性问题统一为共享的29项能力词典上的集合差异，并通过混合确定性分析与LLM抽取构建可追溯的偏差证据，随后支持偏差分类、根因分析与恶意技能检测；

**🔧 技术方法**

技术包括静态抽象语法树分析、正则匹配、LLM结构化提示、置信过滤、HDBSCAN聚类、意图多级分类器以及带有阈值的“松弛否决”判定；

**📊 数据集**

使用的数据集为2026年初抓取的OpenClaw注册表共49,943个技能（约250k偏差），以及906个公开/合成恶意与良性技能的混合基准；

**📈 对比分析**

与规则扫描和单通道LLM审计对比，BIV在恶意技能检测上取得F1=0.946，召回率0.978、精确率0.917，误报率仅7.2%，显著优于两种基线；

**⚠️ 局限性**

局限性在于完全基于静态分析，无法捕获动态分发、严重混淆和运行时攻击；抽取的偏差为下限，缺乏人工标注的真实评估；对流水线的攻击不在范围内。

---

## 465. On Knowledge Compilation For Two-Variable First-Order Logic

**arXiv ID:** 2605.11796 | [PDF](https://arxiv.org/pdf/2605.11796v1)

**作者:** Qiaolan Meng `[一作]` (Beihang University), Ondřej Kuželka `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5021262042)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了两变量谓词一阶逻辑（FO²）在有限域上对其命题展开进行知识编译，提出了一种两阶段的编译器并给出了下界证明。

**💡 创新点**

创新点在于：①证明存在固定FO²句子，其命题展开在DNNF中至少需要指数大小；②设计基于单元和二元类型的两阶段分支编译器，并利用对称性实现高效的可扩展性检查和子电路合并，从而在保持d-DNNF的可解性特性的同时显著压缩电路。

**🔧 技术方法**

主要技术包括：FO²句子转化为Scott正则形；定义单元类型和二元类型并通过类型赋值构建电路；利用配置可满足性与偏序关系实现可扩展性检查；使用哈希表共享等价子电路；并展示该编译器可转化为结构化d-DNNF和OBDD。

**📊 数据集**

实验数据集包括经典的FO²基准句子：2-颜色图（_RB）、无孤立点图（_E）、组合句子（_RBE）、排列结构句子（_P）、支配集句子（_D），以及参数化句子族 (i,j)∈{(4,2),(2,2),(6,2),(4,1),(4,3)}。

**📈 对比分析**

与传统命题DNNF编译器（Bella、d4、c2d）进行对比，使用电路大小与编译时间比值作为评价指标。结果表明，在大多数基准上，该方法生成更小的电路且编译速度更快；在某些无约束二元原子（如_P和参数化句子）上表现略逊，原因在于未充分利用二元原子间的独立性。

**⚠️ 局限性**

局限性包括：①仅针对FO²，无法直接扩展到FO³；②依赖固定的单元先后再二元的分支顺序，无法灵活调整变量顺序；③在二元原子约束较弱的句子中未能充分利用原子级的独立性，导致编译效率下降。

---

## 466. The Death Spiral of Open Source Projects: A Post-Mortem Analysis of Pull Request Workflow Dynamics

**arXiv ID:** 2605.11844 | [PDF](https://arxiv.org/pdf/2605.11844v1)

**作者:** Mohit Kaushik `[一作]` (Guru Nanak Dev University), Kuljit Kaur Chahal `[通讯]` (Guru Nanak Dev University)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5020533434)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 1,736 个 GitHub 不活跃项目及其 1,296,093 条人类发起的 PR 进行大规模后验分析，挖掘 PR 工作流（讨论量、合并延迟、拒绝率、情感倾向、标签化程度）与项目寿命的关系。

**💡 创新点**

首次将 PR 级别的微观工作流动态与项目死亡“死亡螺旋”进行对比，提出死亡是因生态价值衰减与社区疏离而非工作流低效导致；并将 PR 数据与宏观指标（活跃度、依赖关系）整合形成全新的死亡指标框架。

**🔧 技术方法**

使用统计检验（Kruskal–Wallis、Mann–Whitney、Wilcoxon、Chi‑square）与多元 OLS/分位数回归，配合 DistilBERT 语义情感分析模型；采用混合方法实现定量描述与解释性建模。

**📊 数据集**

基于 SEART 平台公开的 1.7 万+ 仓库元数据，筛选出 1,736 个活跃与 1,736 个不活跃项目，共计 1,296,093 条人类 PR 与 2,160,722 条评论；对照组与实验组结构相似，确保结果可比。

**📈 对比分析**

对照组与实验组在标签化率、合并延迟、拒绝率、情绪等维度进行差异检验，显示 PR 级别摩擦和负面情绪为平台普遍现象；回归结果表明项目寿命受生态价值（stars、创新 PR 比例）显著正向影响，工作流效率反而为正向预测因子，整体模型解释率约 38%‑44%。

**⚠️ 局限性**

局限性包括：1）仅考虑人类 PR，忽略 bot 影响后仍可能存在漏检；2）不活跃阈值设定为 6 个月，对极端长寿命项目的结论需谨慎；3）仅使用 GitHub，跨平台泛化未知；4）情感分析模型依赖英文文本，中文/其他语言未覆盖；5）项目死亡定义为无提交，无法区分战略性完成与真性放弃。

---

## 467. Gradient Clipping Beyond Vector Norms: A Spectral Approach for Matrix-Valued Parameters

**arXiv ID:** 2605.11838 | [PDF](https://arxiv.org/pdf/2605.11838v1)

**作者:** Alexander Yukhimchuk `[一作]` (MBZUAI), Sayantan Choudhury `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于矩阵梯度奇异值的谱裁剪（Spectral Clipping）方法，用以在噪声/重尾梯度环境下稳定训练神经网络；

**💡 创新点**

创新点在于：①利用梯度矩阵的奇异值谱而非向量范数来裁剪；②给出了在非凸重尾噪声下的收敛分析；③设计了自适应阈值策略（EMA 与分位数）以及高效的截断奇异值分解实现；

**🔧 技术方法**

采用矩阵奇异值分解（SVD）、随机化截断SVD、EMA/分位数自适应阈值、SGDM/Adam/Muon等优化器结合谱裁剪；

**📊 数据集**

在 CIFAR‑10 上训练 ResNet‑18，使用 Shakespeare 字符级数据训练 NanoGPT，FineWeb 数据训练 GPT‑2 124M，及四层 MLP 的合成重尾噪声回归任务；

**📈 对比分析**

与传统梯度范数裁剪、无裁剪、Adam、Muon 等方法对比。实验表明谱裁剪在准确率、损失收敛速度、训练耗时等指标上均优于或至少不劣于范数裁剪，且在重尾噪声场景下性能提升更为显著；

**⚠️ 局限性**

限制包括：每次迭代需要进行（截断）奇异值分解，导致相对较高的计算和内存开销；截断秩 r 的选择仍需经验；理论分析基于重尾噪声假设，对轻尾噪声场景可能无显著优势。

---

## 468. Multi-Timescale Conductance Spiking Networks: A Sparse, Gradient-Trainable Framework with Rich Firing Dynamics for Enhanced Temporal Processing

**arXiv ID:** 2605.11835 | [PDF](https://arxiv.org/pdf/2605.11835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 469. Choosing features for classifying multiword expressions

**arXiv ID:** 2605.11779 | [PDF](https://arxiv.org/pdf/2605.11779v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 470. COSMIC 1001: Engaging Future Speculation on Space Exploration with Generative AI

**arXiv ID:** 2605.11827 | [PDF](https://arxiv.org/pdf/2605.11827v1)

**作者:** Lingyu Peng `[一作]` (Harbin Institute of Technology), Qingchuan Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1293 | [OpenAlex ID](https://openalex.org/A5013804346)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了交互式装置Cosmic 1001，融合历史太空探索档案、AI生成的未来新闻以及共享可视化，邀请公众通过新闻界面浏览过去事件并参与未来设想。

**💡 创新点**

创新点在于将新闻叙事框架用于过去与未来的交互；使用检索增强生成（RAG）将史料与科幻文本结合产生未来新闻；并通过“未来隧道”将个人生成的故事聚合为可讨论的共同未来，三者（档案、生成、参与）被整合为一种新型未来媒体体验。

**🔧 技术方法**

采用的技术包括：交互式新闻 feed 与时间线界面、检索增强生成（RAG）大模型、文本、语音、图像/视频多模态生成、可视化时间轴与距离式可实现性指标。

**📊 数据集**

使用的数据集包括：开源空间探索时间线 JSON 数据集、NASA 官方文档、278,973 条 instruction‑response 对的科幻语料库（用于检索 218 条科幻片段作为风格参考）。

**📈 对比分析**

论文未提供定量对比或性能指标，主要通过系统展示和案例分析说明效果；缺乏实验评估或与其他生成方法的比较。

**⚠️ 局限性**

局限性：生成内容的真实性和可实现性仅通过可视化指标呈现，缺乏客观评估；受限于历史数据和科幻语料库的质量；用户生成内容受提示与模型能力限制；多人使用时可视化可能混乱；未开展长周期的用户体验研究。

---

## 471. OTT-Vid: Optimal Transport Temporal Token Compression for Video Large Language Models

**arXiv ID:** 2605.11803 | [PDF](https://arxiv.org/pdf/2605.11803v1)

**作者:** Minseok Kang `[一作]` (Yonsei University), Sangyoun Lee `[通讯]` (Yonsei University)

**通讯引用:** 3507 | [OpenAlex ID](https://openalex.org/A5015739530)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了训练无关的视频视觉令牌压缩框架 OTT-Vid，先对每帧进行空间剪枝，再用最优传输（OT）对相邻帧进行时序压缩，并根据 OT 产生的难度动态分配压缩预算。

**💡 创新点**

创新点：① 将 token 重要性与跨帧匹配成本统一编码到 OT 问题中，让重要 token 获得更小质量以抵抗压缩；② 用 OT 总成本衡量帧对的可压缩性，进而实现自适应预算分配；③ 在同一 OT 框架下同时处理空间与时间冗余。

**🔧 技术方法**

使用技术：空间代表性选择（saliency‑weighted coverage）+ leave‑one‑out 重要性估计；可变质量最优传输（Sinkhorn 迭代）配合自适应局部成本 α_t；预算分配与基于 OT 计划的压缩执行；并行全局 OT 计算。

**📊 数据集**

实验数据集：视频问答（MVBench、VideoMME、LongVideoBench、MLVU）和视频时序定位（Charades‑STA/ActivityNet‑Captions，TimeLens 版），在 Qwen2.5‑VL‑7B、LLaVA‑OneVision‑7B、LLaVA‑Video‑7B 等 Video‑LLM 上进行评估。

**📈 对比分析**

与 HoliTom、FastVID、FlashVID、UniComp 等最新训练无关压缩方法对比。OTT‑Vid 在 10% token 保留下，VQA 性能保持 95.8%，VTG 保留 73.9%，分别比最强基准提升约 1%（VQA）和 7%（VTG）。同时保持与 FastVID、UniComp 相近的推理速度、内存占用与 FLOPs。

**⚠️ 局限性**

局限性：一次性计算所有 OT 计划，未考虑压缩后表示变化导致的迭代改进；对快速运动或剧烈变化的帧可能仍存在压缩误差；需要进一步探索逐步或递归的 OT 调整机制。

---

## 472. An Extensive Replication Study of the ABLoTS Approach for Bug Localization

**arXiv ID:** 2605.11790 | [PDF](https://arxiv.org/pdf/2605.11790v1)

**作者:** Feifei Niu `[一作]` (Nanjing University), Alexander Egyed `[通讯]` (Johannes Kepler University)

**通讯引用:** 7331 | [OpenAlex ID](https://openalex.org/A5057561309)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对ABLoTS（Bug Localization）方法进行了复现，并在原始Java数据集、扩展Java数据集以及Python数据集上评估了核心组件TraceScore和完整框架的性能。

**💡 创新点**

创新点在于揭示了ABLoTS中BugCache组件使用错误的切分时间导致的数据泄露问题，说明ABLoTS整体不可复现；同时系统比较了多种组合方法（composer），发现LR、CombSUM和固定权重在三种数据集上均表现最佳。

**🔧 技术方法**

使用信息检索技术（TF‑IDF、余弦相似度）、BugCache、BLUiR、TraceScore等组件，并对组合方法进行评估（DT、LR、CombSUM、CombMNZ、CombANZ、CorrB、Borda、RF、MLP 等）。

**📊 数据集**

数据集包括：原始Java 11个开源项目（共约8,494条bug报告），扩展Java 16个项目（共约25,893条bug报告），以及Python 12个项目（共1,289条bug报告）。

**📈 对比分析**

采用 MAP、MRR、Top‑1/5/10 作为评估指标。结果显示：在松散 cut‑off 条件下 TraceScore 能与原始结果保持一致；ABLoTS整体表现大幅下降；使用固定权重组合在扩展数据集上 Top‑10 约 70% 左右。不同 composer 的性能差异显著，LR、CombSUM、固定权重优于 DT、RF。

**⚠️ 局限性**

局限性包括：缺少公开实现导致复现细节可能不完整；Python 数据集缺少非bug报告和 traceability 信息；BugCache 的错误 cut‑off 导致实验结果不可复现；以及未系统评估项目规模、代码风格等因素对性能的影响。

---

## 473. Entropy Polarity in Reinforcement Fine-Tuning: Direction, Asymmetry, and Control

**arXiv ID:** 2605.11775 | [PDF](https://arxiv.org/pdf/2605.11775v1)

**作者:** Jiazheng Zhang `[一作]` (Fudan NLP Group), Xuanjing Huang `[通讯]` (Tencent Hunyuan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于令牌级别的熵机制理论和熵极性度量，并在RLVR中通过熵极性重加权实现自适应探索-利用控制；

**💡 创新点**

通过熵极性将熵扩张与收缩分支区分，并利用熵极性与经验熵曲线动态调节两者权重，实现更平衡的探索与奖励收敛；

**🔧 技术方法**

令牌级熵极性计算、软max梯度更新分析、GRPO框架、EMA平滑、动态权重调节；

**📊 数据集**

数学推理数据集DAPO-Math-17K、OlympiadBench、AMC、MATH500、AIME等，以及代码生成CRUX、指令跟随IFEval、知识库MMLU-Pro、工具调用BFCL-v3、ACEBench；

**📈 对比分析**

与GRPO、DAPO、Entropy Regularization、80/20等基线对比，PAPO在大多数数学、工具调用任务上取得显著性能提升，尤其在14B模型上在AIME、MATH、BFCL等任务上超越前沿方法；

**⚠️ 局限性**

仅限于中等规模（7B/14B）文本LLM和可验证奖励，未覆盖超长序列、多模态或更大规模模型的长期决策场景。

---

## 474. From Token to Token Pair: Efficient Prompt Compression for Large Language Models in Clinical Prediction

**arXiv ID:** 2605.11774 | [PDF](https://arxiv.org/pdf/2605.11774v1)

**作者:** Mingcheng Zhu `[一作]` (University of Oxford), Tingting Zhu `[通讯]` (University of Oxford)

**通讯引用:** 5466 | [OpenAlex ID](https://openalex.org/A5055850985)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 MedTPE（Medical Token-Pair Encoding）方法，对 EHR 文本进行无损压缩，以提升大型语言模型在临床预测任务中的推理效率和保持或提升预测性能。

**💡 创新点**

创新点包括：① 在不增加词表大小和计算复杂度的前提下，通过依赖-aware 替换机制将高频 token 对合并为新 token；② 仅微调新 token 嵌入并采用自监督学习（SSFT），实现无标签的嵌入对齐；③ 该方法实现了信息密度提升、无损压缩且在多种 LLM、任务和跨域场景中保持鲁棒性。

**🔧 技术方法**

技术方案：基于 BPE/WordPiece 等基础 tokenizer 的层次化 token-pair merge；利用频率+长度权重挑选高价值 token 对；对词表执行依赖-aware 替换；自监督 fine‑tuning 仅更新新 token 嵌入；在实验中使用多种大型语言模型（Qwen2.5、Llama3 等）及多任务评估。

**📊 数据集**

数据集：MIMIC‑IV（ICU 失血率、表型预测）、EHRSHOT（30 天再住院、1 年胰腺癌预测）为主；此外评估临床笔记、ARC-Challenge（科学推理）、ECTSum（金融摘要）和 CMedQA2（中文医疗问答）等跨域数据。

**📈 对比分析**

对比方法：与 T5Summary、LLMLingua2、ZeTT 等基线进行比较，使用 F1、格式符合率 (FCR)、推理时间、压缩率等指标。MedTPE 在所有 LLM 与任务中实现 22.8%–32.4% 的压缩率，推理时间下降 34%–63%，且 F1 与基线相比不下降甚至提升；在跨域实验中亦保持或提高性能。

**⚠️ 局限性**

局限性：需要直接修改 tokenizer 与嵌入矩阵，难以在闭源模型或固定词表的生产环境中应用；对 chain‑of‑thought（CoT）推理敏感，难以与复杂推理框架无缝整合；在不同地区或低资源环境需要重新构建词表，存在潜在偏差风险。

---

## 475. Breaking the Dependency Chaos: A Constraint-Driven Python Dependency Resolution Strategy with Selective LLM Imputation

**arXiv ID:** 2605.11772 | [PDF](https://arxiv.org/pdf/2605.11772v1)

**作者:** Kowshik Chowdhury `[一作]` (Kennesaw State University), Shazibul Islam Shamim `[通讯]` (Kennesaw State University)

**通讯引用:** 152 | [OpenAlex ID](https://openalex.org/A5019761662)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合 Z3 约束求解与 LLM 缺失元数据填补的混合管线——SMT‑LLM，用以解决 Python 代码片段的依赖冲突与安装失败问题。

**💡 创新点**

创新点包括：① 将 LLM 的角色限定为可验证的事实查询，避免盲目猜测；② 采用 AST 与五层 PyPI 解析取代 LLM 全部猜测；③ 引入软硬约束区分，将 PyPI 元数据设为硬约束，LLM 推断的依赖设为可放宽的软约束；④ 通过 Docker 运行错误自动提取约束并反向迭代；⑤ 在 LLM 调用与 Docker 构建上实现显著削减。

**🔧 技术方法**

技术栈包括 Python AST 分析、Vermin（Python 版本检测）、PyPI API 与多层名称映射、Z3 SMT 求解器、Docker 运行验证、Ollama 上的 Gemma‑2:9B LLM。

**📊 数据集**

评估基于 HG2.9K（2891 条 GitHub Gist）Python 代码片段及其对应的 PyPI 元数据。

**📈 对比分析**

与 PLLM（现有最强基线）对比：SMT‑LLM 成功率提升至 83.6%（PLL‑M 为 54.8%）；中位解法时间从 151.5 s 降至 23.9 s（≈6.3×快）；LLM 调用次数从约 24.9 次降至 2.26 次（≈11×少）；Docker 重试次数从约 23.9 次降至 4.9 次。

**⚠️ 局限性**

局限性主要包括：缺失的本地或平台 SDK 模块导致无法解析；Python 2/3 兼容性与 PyPI 元数据时间漂移造成版本不匹配；缓存中出现错误映射导致后续误解；Z3 求解器在多解情形下非确定；以及无法处理嵌入式运行时环境（如 Blender 脚本）等。

---

## 476. Safety-Oriented Evaluation of Language Understanding Systems for Air Traffic Control

**arXiv ID:** 2605.11769 | [PDF](https://arxiv.org/pdf/2605.11769v1)

**作者:** Yujing Chang `[一作]` (Nanyang Technological University), Sameer Alam `[通讯]` (Nanyang Technological University)

**通讯引用:** 2817 | [OpenAlex ID](https://openalex.org/A5008604294)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了针对空中交通管制（ATC）语言理解的安全导向、风险感知评估框架，并在该框架下对多种大型语言模型（LLM）进行系统评估。

**💡 创新点**

创新点在于将专家评估的实体重要性与动作级风险系数结合，构建了基于实体权重与动作风险的加权评分体系，并引入严格的全结构正确性约束，揭示了传统指标掩盖的安全隐患。

**🔧 技术方法**

使用技术包括零样本结构化提示的LLM推理（如Gemini、GPT‑5.1、Qwen系列等）、多种ASR引擎（Whisper‑large/medium/small/turbo、wav2vec2）以及自定义的风险加权评估脚本。

**📊 数据集**

数据集为新加坡樟宜机场（WSSS）三条地面管制频段录制的约1,000条对话，已人工标注发声者角色、意图、实体槽及动作级风险标签，覆盖航班指令、读回、信息告知等场景。

**📈 对比分析**

比较方法：对比宏观指标（Speaker‑F1、Intent‑F1、Act‑F1）与风险加权指标（Risk‑Score、Risk‑NER、Act‑W/T）以及严格评估（Risk‑Strict）。在清洁文本下最佳模型Risk‑Score≈0.69，绝大多数模型低于0.6；在ASR噪声下Risk‑Score降至0.04–0.07，严格指标普遍降至0。

**⚠️ 局限性**

局限性包括：仅采用零样本推理未进行领域微调；数据量有限，仅覆盖地面管制；风险权重基于少量专家调查，可能存在主观偏差；评估高度依赖ASR准确率，实际部署时可能更差；未考虑多语言或非标准口语的鲁棒性。

---

## 477. Probabilistic Calibration Is a Trainable Capability in Language Models

**arXiv ID:** 2605.11845 | [PDF](https://arxiv.org/pdf/2605.11845v1)

**作者:** Davide Baldelli `[一作]` (Chandar Research Lab), Sarath Chandar `[通讯]` (Chandar Research Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了一种通过微调实现语言模型概率校准的方法（Calibration Fine‑Tuning），在已知目标分布的数值采样任务中显著提升了采样的准确性和多样性。

**💡 创新点**

创新点在于提出两种微调策略——soft‑target（基于词典生成的下一词分布监督）和hard‑target（基于采样完成的交叉熵监督）——并证明概率校准是一种可训练的能力；同时探讨其在自然语言随机生成、NoveltyBench以及多项选择题生成等下游任务中的迁移效果。

**🔧 技术方法**

使用了LoRA适配器在冻结的基础模型上进行微调，软目标采用Trie‑derived next‑token监督，硬目标采用从目标分布采样完成的标准自回归交叉熵；训练时采用AdamW、学习率2e‑4、cosine调度等。

**📊 数据集**

数据集为完全合成的数值采样提示，覆盖30个分布族（24个用于训练，6个OOV用于测试），每个分布族在参数空间上进行离散化，最终生成约2000个提示-目标对。

**📈 对比分析**

与基线（原始检查点）以及SSOT提示等方法比较，在结构化分布采样任务中，Soft/Hard 微调都能将Wasserstein‑1距离和前向KL下降数倍；在开放式随机生成和NoveltyBench等迁移任务中，Soft‑target往往能显著提升支持度和多样性；但在某些模型和任务上也出现性能下降或不显著提升。

**⚠️ 局限性**

局限性包括：对重参数化和尾部分布的近似处理可能导致偏差；不同微调配置对模型能力的影响不均衡，尤其在推理、数学推理等任务上可能出现退化；当前方法仅针对已知目标分布，泛化到更复杂或隐式分布的能力尚待进一步验证。

---

## 478. Selection, Not Fusion: Radar-Modulated State Space Models for Radar-Camera Depth Estimation

**arXiv ID:** 2605.11840 | [PDF](https://arxiv.org/pdf/2605.11840v1)

**作者:** Zhangcheng Hou `[一作]` (Keio University), Tomoaki Ohtsuki `[通讯]` (Keio University)

**通讯引用:** 10751 | [OpenAlex ID](https://openalex.org/A5016337773)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Radar-Modulated Selection（RMS）与Multi-View Scan Pyramid，将雷达信息注入Mamba的选择机制中以完成深度估计

**💡 创新点**

创新点在于把雷达直接调制Mamba的step size和readout参数，实现线性成本的跨模态交互并保持预训练图像模型的初始化不变

**🔧 技术方法**

采用Mamba序列模型、ResNet-34图像编码器、PCA-GM雷达特征提取器、FiLM与Mamba扫描融合的层级解码器

**📊 数据集**

在nuScenes和ZJU-4DRadarCam两个公开数据集上进行训练与评估

**📈 对比分析**

与多种基线（LiDAR-摄像机融合、TacoDepth、FusionMamba等）对比，SemoDepth在nuScenes的MAE、RMSE均达到新标杆，单帧推理时间仅26.8 ms，成为最快方法

**⚠️ 局限性**

在极端夜间低光条件下，基于注意力的融合在近距离仍略优于RMS，表明在最恶劣视觉环境下仍需进一步改进

---

## 479. Learning Action Manifold with Multi-view Latent Priors for Robotic Manipulation

**arXiv ID:** 2605.11832 | [PDF](https://arxiv.org/pdf/2605.11832v1)

**作者:** Junjin Xiao `[一作]` (Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education), Wei-Shi Zheng `[通讯]` (Key Laboratory of Machine Intelligence and Advanced Computing, Ministry of Education)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种同时提升单目视觉三维感知和动作学习效率的 Vision‑Language‑Action (VLA) 框架。

**💡 创新点**

创新点包括：①利用预训练多视角扩散模型在潜在空间合成新视图以缓解单目深度歧义；②设计 Geometry‑Guided Gated Transformer (G^3T) 在单目几何先验指引下对齐并自适应融合多视图潜在特征；③提出 Action Manifold Learning (AML)，直接预测动作而非间接噪声/速度，从而降低优化难度。

**🔧 技术方法**

技术主要包括多视角潜在扩散合成、基于 VGGT 的几何先验、跨模态注意力融合、G^3T 自适应门控机制，以及基于 Diffusion Transformer 的 AML 直接动作预测。

**📊 数据集**

在 LIBERO、LIBERO‑Plus、RoboTwin 2.0 三大仿真基准以及真实 Franka Panda 机器人上进行评估，并使用公开演示数据集进行训练。

**📈 对比分析**

与 OpenVLA‑OFT、π_0 等最先进 VLA 基线比较，方法在 LIBERO 上平均成功率 98.6%、LIBERO‑Plus 85.7%、RoboTwin 2.0 80%+，并在真实机器人实验中实现了最高成功率，显著优于对比模型。

**⚠️ 局限性**

局限性在于多视角扩散合成的计算开销和推理时延，导致实时控制难度较大，未来需将几何推理能力蒸馏进模型以降低延迟。

---

## 480. Mapping Embodied Affective Touch Strategies on a Humanoid Robot

**arXiv ID:** 2605.11825 | [PDF](https://arxiv.org/pdf/2605.11825v1)

**作者:** Qiaoqiao Ren `[一作]`, Tony Belpaeme `[通讯]` (Ghent University)

**通讯引用:** 8606 | [OpenAlex ID](https://openalex.org/A5035627933)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了32名参与者在三种触摸条件（自由触摸、仅臂部触摸、仅躯干触摸）下对iCub机器人表达八种情感的触摸空间分布与动态特征，并通过问卷评估情感触摸对人机亲密感的影响。

**💡 创新点**

首次系统量化全身机器人的情感触摸在空间与动力学上的差异，揭示触摸策略随身体部位和空间限制而变化，并证明单向情感触摸可提升人与机器人的亲密感。

**🔧 技术方法**

使用分布式触觉传感器、视频同步记录，结合混合效应模型、logit回归、eta² 影响量、Hotelling’s T² 多变量检验等统计方法对触摸数据进行分析。

**📊 数据集**

基于30名有效参与者的实验数据，共收集了72条试验的触觉序列、视频帧及问卷信息，涵盖8种情感、3种触摸条件。

**📈 对比分析**

通过混合效应模型和效应量比较，评估身体部位、压力与运动特征在情感解释中的贡献，结果显示身体部位与压力/运动的情感解释力相当，且在臂部与躯干条件下情感特征显著差异；未进行分类性能评估。

**⚠️ 局限性**

局限包括机器人姿势固定、触摸区域受限、单向任务、样本文化单一、传感器覆盖仍有限、特征选择偏向可解释性而非完整性，缺乏长期或双向交互验证。

---

## 481. MedMemoryBench: Benchmarking Agent Memory in Personalized Healthcare

**arXiv ID:** 2605.11814 | [PDF](https://arxiv.org/pdf/2605.11814v1)

**作者:** Yihao Wang `[一作]` (Zhejiang University), Lidan Shou `[通讯]` (Zhejiang University)

**通讯引用:** 1828 | [OpenAlex ID](https://openalex.org/A5103017455)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 MedMemoryBench，一个专门用于评估个性化医疗代理记忆能力的基准数据集与评测协议。

**💡 创新点**

创新点包括：①使用人机协作流水线生成真实且可验证的长期医疗轨迹；②引入“streaming evaluate‑while‑constructing”动态评测协议，模拟生产环境下记忆累积与即时利用；③系统化研究了记忆饱和（memory saturation）现象，揭示持续信息输入会削弱检索与推理的鲁棒性。

**🔧 技术方法**

采用的技术包括：临床专家验证的合成患者档案与事件图生成；多轮对话仿真与记忆提取；Agentic RAG 结合陷阱事件（trap events）检索关键临床事实；噪声注入与对抗测试；以及基于图结构的检索与记忆写入框架。

**📊 数据集**

数据集：约 2,000 个会话、16,000 次交互回合，基于 20 个慢性病患者真实病例生成的全年诊疗轨迹；所有数据均经过专家复核，保证临床准确性。

**📈 对比分析**

在该基准上评测了主流记忆方法（Mem0、MemOS、Letta/MemGPT、Zep 等）。实验显示：现有模型在需要多记忆组合推理的复杂医学任务上表现不佳；检索精度是性能瓶颈；随记忆量增大且噪声累积，系统性能显著下降，验证了记忆饱和的危害。

**⚠️ 局限性**

局限性包括：数据覆盖范围仍有限（仅 20 种患者类型）；构造过程高度依赖专家人工审核，扩展性受限；评测侧重记忆机制，未覆盖完整代理系统的安全性与合规性；真实生产环境中的多源异构数据集成与动态更新尚未在基准中全面模拟。

---

## 482. Empirical coordination in the finite blocklength regime: an achievability result---Extended version

**arXiv ID:** 2605.11810 | [PDF](https://arxiv.org/pdf/2605.11810v1)

**作者:** Olivier Massicot `[一作]` (Univ. Rennes), Maël Le Treust `[通讯]` (Univ. Rennes)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在有限块长度条件下的经验协调问题，分析了编码器和解码器如何在通信约束下生成与目标分布共同典型的动作对序列。

**💡 创新点**

创新点在于将有限块长度分析扩展到经验协调设置，提供了一个关于最优速率的界限，并与现有的强协调结果相补充。

**🔧 技术方法**

采用了香农的随机编码论证和类型方法来分析随机码本的平均性能。

**📊 数据集**

使用了与目标分布相关的有限消息集和随机变量的序列，具体数据集未明确说明。

**📈 对比分析**

通过与现有的强协调和有限块长度文献中的结果进行比较，展示了所提出的界限与现有结果的一致性，性能表现良好。

**⚠️ 局限性**

限制在于未能完全解决所有可能的编码方案，且对特定条件下的最优速率的界限可能不够紧。

---

## 483. Stop Marginalizing My Dreams: Model Inversion via Laplace Kernel for Continual Learning

**arXiv ID:** 2605.11804 | [PDF](https://arxiv.org/pdf/2605.11804v1)

**作者:** Patryk Krukowski `[一作]` (Jagiellonian University), Łukasz Struski `[通讯]` (Jagiellonian University)

**通讯引用:** 560 | [OpenAlex ID](https://openalex.org/A5007063171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于拉普拉斯核的结构化协方差模型（LCM），并将其集成到数据无关连续学习（DFCIL）的模型反演流程中，以更精确地重建中间特征分布。

**💡 创新点**

突破传统对角协方差假设，利用拉普拉斯核参数化实现全协方差建模，仅需线性参数并通过三对角精度矩阵实现对数级别的计算复杂度，从而在保持可扩展性的同时显著提升伪样本质量。

**🔧 技术方法**

使用拉普拉斯核结构化协方差、Frobenius 范数目标、三对角精度矩阵、模型反演（基于生成器）、PMI 反演框架，以及 ResNet/ViT+CLIP 等主干网络。

**📊 数据集**

在 CIFAR-100、Tiny-ImageNet、CUB-200 数据集上进行实验，采用 ResNet-32 和 ViT‑CLIP（ViT-B/16）两种主干进行评估。

**📈 对比分析**

与 DeepInversion、ABD、R‑DFCIL、PMI 及部分带缓冲方法进行对比；在 ResNet-32 上平均增益约 1–1.5%（CIFAR‑100 10‑task 上 +1.48%），在 ViT‑CLIP 上平均/最后任务增益约 0.12% / 0.43%；整体性能优于现有无数据缓冲方法，并在部分场景接近或超过带缓冲基线。

**⚠️ 局限性**

协方差参数化仍相对有限，缺乏更灵活的分布建模；依赖 PMI 反演框架，若特征分布对齐不足仍会影响伪样本质量，且对更高维或更复杂模型的适应性尚待验证。

---

## 484. Crash Assessment via Mesh-Based Graph Neural Networks and Physics-Aware Attention

**arXiv ID:** 2605.11784 | [PDF](https://arxiv.org/pdf/2605.11784v1)

**作者:** Gabriel Curtosi `[一作]` (SEAT S.A.), Xabier Larráyoz Izcara `[通讯]` (SEAT S.A.)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究并实现了全车侧向杆撞击的自回归神经替代模型，使用混合局部网格消息传递与全局几何注意力机制，结合稀疏接触修正，以在工业设计空间中高精度预测车身变形场并评估乘员生存空间；

**💡 创新点**

创新点在于提出MeshGeoTransolver与MeshGeoFLARE两种基于局部消息传递与几何注意力的混合架构，并首次引入可学习的稀疏接触块，在保持全局信息传播的同时显著提升对动态接触的建模；

**🔧 技术方法**

采用图神经网络（MPNN）、Transolver式物理注意力、GeoTransolver/FLARE几何注意力、稀疏接触块以及自回归时间步推进的混合结构；

**📊 数据集**

使用SEAT公司提供的完整车体有限元模型，基于拉丁超立方采样生成的200组包含几何、厚度与撞击参数的设计空间数据；

**📈 对比分析**

与单一MPNN、纯Transolver、MeshTransolver等模型在相同训练与超参数设置下比较，MeshGeoFLARE在25个测试样本的时间平均RMSE仅为3.20 mm（相对误差1.37%），并在乘员生存空间误差上保持负向保守偏差；

**⚠️ 局限性**

局限包括仅验证单一撞击工况、缺乏对离群设计的评估、未对每种模型进行单独超参数调优，以及对接触块在更剧烈冲击下的鲁棒性尚待进一步验证。

---

## 485. Is Monotonic Sampling Necessary in Diffusion Models?

**arXiv ID:** 2605.11773 | [PDF](https://arxiv.org/pdf/2605.11773v1)

**作者:** Muhammad Haris Khan `[一作]` `[通讯]` (University of Copenhagen), Muhammad Haris Khan (University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探讨扩散模型采样时是否需要单调降噪调度，系统评估了四类非单调调度对不同架构模型（DDPM、EDM、Flow Matching）的影响。

**💡 创新点**

首次从经验上验证单调性是必要的，发现任何设计的非单调调度在所有测试配置下都不优于单调基线，并提出了Schedule Sensitivity Coefficient（SSC）作为衡量去噪器对调度敏感度的新诊断指标。

**🔧 技术方法**

设计四种结构化非单调调度（单次加热、锯齿、阻尼振荡、适应加热），在多种采样算法（DDIM、DDPM、EDM、Flow Matching）中实现；使用FID作为质量评估。

**📊 数据集**

所有实验均在CIFAR‑10数据集上进行。

**📈 对比分析**

对比方法为固定NFE（10至200）下的Fidelity（FID），所有非单调调度在任何NFE下都未能超过单调基线；单调采样在NFE≤100时始终占优，且在NFE=200时DDPM的随机噪声注入（η=1）略优于DDIM。

**⚠️ 局限性**

局限性包括仅使用CIFAR‑10、仅测试有限的非单调调度族、仅评估FID、不同表格使用不同样本量、Flow Matching基线Fid偏高、SSC为上限诊断而非精确衡量等。

---

## 486. M$^4$-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection

**arXiv ID:** 2605.11760 | [PDF](https://arxiv.org/pdf/2605.11760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 487. Revisiting Shadow Detection from a Vision-Language Perspective

**arXiv ID:** 2605.11771 | [PDF](https://arxiv.org/pdf/2605.11771v1)

**作者:** Yonghui Wang `[一作]` (University of Science and Technology of China), Houqiang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 26867 | [OpenAlex ID](https://openalex.org/A5078141810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SVL框架，利用语言作为显式语义参考来提升阴影检测的鲁棒性。

**💡 创新点**

创新点在于全局语义注入与局部语义约束结合，以及场景级阴影比例回归，且参数量仅占总量不到1%。

**🔧 技术方法**

使用冻结的DINOv3视觉编码器与CLIP文本编码器，结合跨模态一致性学习、全局-局部耦合、轻量级解码器及CRF后处理。

**📊 数据集**

在SBU、UCF、ISTD三大阴影检测基准以及构造的SBU‑Hard和UCF‑Hard难易样本集上进行评估。

**📈 对比分析**

与SwinShadow、STNet、ShadowAdapter等最新方法对比，SVL在SBU和UCF上实现最低BER，并在视觉模糊区块上显著降低误报，整体性能优于现有方法。

**⚠️ 局限性**

在极薄或交错的阴影模式上仍易失真，边界细节捕捉有限，且对文本提示的语义精确度有一定依赖。

---

## 488. Decomposing the Generalization Gap in PROTAC Activity Prediction: Variance Attribution and the Inter-Laboratory Ceiling

**arXiv ID:** 2605.11764 | [PDF](https://arxiv.org/pdf/2605.11764v1)

**作者:** Thor Klamt `[一作]` (Leibniz University Hannover), Ming Tang `[通讯]` (Leibniz University Hannover)

**通讯引用:** 9325 | [OpenAlex ID](https://openalex.org/A5071217303)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 PROTAC-Bench benchmark，使用留一靶点（LOTO）评估体系，分解随机交叉验证与 LOTO 之间的泛化缺口，并研究少量样本校准与模型架构、超参数对性能的影响。

**💡 创新点**

① 引入实验室间变异分解证明泛化瓶颈主要源自测量噪声；② 证明无论模型架构、蛋白语言模型规模或超参数优化都难以突破约 0.668 AUROC 的上限；③ 提出 k=5 少量样本校准与 ADMET 特征组合可将 LOTO AUROC 提升至 0.705；④ 发布完整基准、分解框架与代码，提供可复现的评估模板。

**🔧 技术方法**

使用随机森林、GIN、D‑MPNN、ChemBERTa、DeepPROTACs、DegradeMaster、PROTAC‑STAN、Ribes 等模型；ESM‑2 蛋白语言模型（150 M–3 B 参数级别）；少量样本（k‑shot）重训练与 Platt 归一化；方差分解、超参数搜索与交叉实验室对比分析。

**📊 数据集**

PROTAC‑Bench：10,748 条测定，173 个 UniProt 靶点，其中 65 个满足 LOTO 评估标准；数据来自 PROTAC‑DB 3.0、DegradeMaster、公开论文与数据库，包含 SMILES、DC50、Dmax、E3 ligase 等信息。

**📈 对比分析**

对 8 种模型、不同 ESM‑2 参数规模、超参数搜索、少量样本校准进行宏平均 AUROC 对比。随机交叉验证 AUROC 0.902，LOTO AUROC 0.668；少量样本 + ADMET + 校准后可达 0.705；所有模型在 LOTO 下达到相同 0.668‑0.678 的性能峰值，说明架构与超参数提升有限。

**⚠️ 局限性**

① 仅评估二分类性能，连续 DC50 预测差异不显著；② ESM‑2 规模评估受预训练覆盖影响，未能验证对未见靶点的泛化；③ LOFO 评估略低于 LOTO，表明结构泛化更困难；④ 数据集偏向 CRBN E3 ligase，限制对其他 ligase 的推广；⑤ SMILES 去重后上限下降至 0.617，提示测量噪声上限受限；⑥ 需在更大、更多样化靶点上进一步验证。

---

## 489. Vector Scaffolding: Inter-Scale Orchestration for Differentiable Image Vectorization

**arXiv ID:** 2605.11913 | [PDF](https://arxiv.org/pdf/2605.11913v1)

**作者:** Jaerin Lee `[一作]` (Seoul National University), Kyoung Mu Lee `[通讯]` (Seoul National University)

**通讯引用:** 27386 | [OpenAlex ID](https://openalex.org/A5046504049)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Vector Scaffolding——一种用于可微图像向量化的分层优化框架，旨在避免拓扑崩溃并提升可编辑性。

**💡 创新点**

创新点在于①引入Interior Gradient Aggregation补齐曲线内部梯度；②设计Progressive Stratification与Rapid Inflation Scheduling实现多层级结构和极高学习率；③通过结构化优化抑制曲线竞争导致的噪声累积。

**🔧 技术方法**

主要技术包括Bézier曲线+2D高斯splatting渲染后端，内部梯度聚合、分层发散策略、超高学习率（×50）以及Adan优化器。

**📊 数据集**

实验使用Kodak图像集、DIV2K高分辨率集，以及8000×8000的地球图像做LoD演示。

**📈 对比分析**

与DiffVG、LIVE、LIVSS、Bézier Splatting等基线在相同曲线预算下对比，PSNR提升1–1.4dB，LPIPS更优，优化速度提升约2.5×。

**⚠️ 局限性**

局限性在于仍需手动设定层级比例与增量步长，曲线数量与计算资源受限，极高频纹理下仍可能出现细节不足。

---

## 490. Delightful Gradients Accelerate Corner Escape

**arXiv ID:** 2605.11908 | [PDF](https://arxiv.org/pdf/2605.11908v1)

**作者:** Jincheng Mei `[一作]` (Google DeepMind), Ian Osband `[通讯]` (Google DeepMind)

**通讯引用:** 4927 | [OpenAlex ID](https://openalex.org/A5015899120)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种改进的策略梯度算法——Delightful Policy Gradient（DG），通过将优势与动作惊奇度相乘来门控梯度，从而消除softmax策略梯度在子最优角落自陷的慢逃逸问题；

**💡 创新点**

创新点在于：①零温度极限下完全移除负优势动作；②在任意温度下利用惊奇度对罕见恶劣动作实现多项式抑制；③证明了在K-armed bandit和表格MDP中DG可以全局收敛至最优策略，并给出对角落逃逸的对数时间上界；

**🔧 技术方法**

主要技术包括：基于softmax参数化的策略梯度、优势函数、惊奇度（action surprisal）门控、对数几率（logit）间隔分解、角落逃逸的量化分区与逃逸时间分析、全局收敛的单调价值提升与贝尔曼最优性论证；

**📊 数据集**

实验使用了三种数据集：①三臂bandit（理论验证）②MNIST上下文bandit（共享参数神经网络）以及与之类似的实验环境；

**📈 对比分析**

与标准softmax策略梯度（PG）比较：在bandit中，PG的逃逸时间随奖励差Δ指数级增长，而DG保持近线性；在MNIST上下文bandit中，DG的分类错误率随预训练“角落深度”变化保持平稳，而PG随角落加深错误率持续上升；

**⚠️ 局限性**

限制：在共享参数函数逼近下，存在精确的反例使得DG失效；理论证明仅在表格情形成立，未能涵盖结构化函数逼近或状态稀有分布的情况。

---

## 491. YFPO: A Preliminary Study of Yoked Feature Preference Optimization with Neuron-Guided Rewards for Mathematical Reasoning

**arXiv ID:** 2605.11906 | [PDF](https://arxiv.org/pdf/2605.11906v1)

**作者:** Yifan Le `[一作]` `[通讯]` (Zhejiang University), Yifan Le (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 YFPO 方法，在数学推理任务中将神经元激活信号与传统偏好优化结合，以提升大型语言模型的推理表现。

**💡 创新点**

创新点在于首次将基于 AttnLRP 识别的数学相关神经元激活差值作为辅助奖励，补充外部偏好信号，使优化更细粒度且可解释。

**🔧 技术方法**

采用 AttnLRP 进行神经元相关性筛选、DPO（Direct Preference Optimization）进行偏好优化，并在训练中加入神经元奖励边际。

**📊 数据集**

使用 Qwen2-0.5B 语言模型，偏好训练数据基于 GSM8K 的 10K（及 2K 子集）样本，评估指标为 GSM8K 准确率。

**📈 对比分析**

与标准 DPO 对比，YFPO 在最佳检查点可将 GSM8K 准确率提升至 0.3738（相较 0.3662）并在部分 λ 设置下平均准确率亦有所提升，增益虽有限但可观。

**⚠️ 局限性**

局限包括实验规模小（仅 0.5B 模型）、增益幅度有限、神经元选择固定不随训练动态变化、潜在的激活奖励捷径、缺乏更大模型与多任务验证。

---

## 492. Toward Modeling Player-Specific Chess Behaviors

**arXiv ID:** 2605.11893 | [PDF](https://arxiv.org/pdf/2605.11893v1)

**作者:** Loris Sogliuzzo `[一作]` (UCLouvain), Eric Piette `[通讯]` (UCLouvain)

**通讯引用:** 1119 | [OpenAlex ID](https://openalex.org/A5061767166)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个能够复现历史棋手个性化行为的AI模型，结合Maia-2的个性化嵌入与有限的蒙特卡洛搜索；

**💡 创新点**

首次将Jensen‑Shannon散度作为行为一致性度量，弥补传统动作准确率无法捕捉人类决策多样性的缺陷；

**🔧 技术方法**

采用Maia-2网络、细粒度嵌入微调、蒙特卡洛树搜索、AutoEncoder与UMAP降维、Jensen‑Shannon散度；

**📊 数据集**

使用 Chessgames.com 收集的 16 位20世纪世界冠军棋局，约 2000+ 场棋局；

**📈 对比分析**

对比了原始Maia-2、微调后 Maial-2 FT 以及加入 MCTS 的版本；MCTS 在动机上减少了准确率约 18%，但通过 JSD 平均从 0.159 降至 0.101，显著提升风格一致性；

**⚠️ 局限性**

局限在于度量受搜索、降维和离散化影响，可能掩盖真正的风格差异，且仅验证了顶级棋手，未扩展至业余玩家或其他游戏。

---

## 493. From Clever Hans to Scientific Discovery: Interpreting EEG Foundational Transformers with LRP

**arXiv ID:** 2605.11885 | [PDF](https://arxiv.org/pdf/2605.11885v1)

**作者:** Justus Meyer zu Bexten `[一作]` (Leipzig University), Simon M. Hofmann `[通讯]` (Max Planck Institute for Human Cognitive and Brain Sciences)

**通讯引用:** 522 | [OpenAlex ID](https://openalex.org/A5065181864)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文将注意力感知LRP方法应用于EEG-Transformer（LaBraM），并在R峰检测、运动想象与情绪预测三类任务上进行解释与验证。

**💡 创新点**

首次验证LRP在EEG-FM中的可靠性，同时揭示模型的Shortcut行为（如眼动伪信号）以及发现情绪唤醒相关的新颖电极模式。

**🔧 技术方法**

采用注意力感知LRP（epsilon-LRP/γ=0.25规则）、Transformer基模型LaBraM以及梯度相关的LRP规则，并与Finetune、Frozen、FromScratch与传统CSP‑LDA进行对比。

**📊 数据集**

使用的数据集包括AffectiveVR（情绪EEG与生理数据）、PhysioNet EEG Motor MI（运动想象/执行）以及AffectiveVR的情绪预测数据。

**📈 对比分析**

与CSP‑LDA对比，Finetune版LaBraM在R峰检测AUROC≈0.83、BAC≈75%；运动想象AUROC≈0.67；情绪唤醒预测AUROC≈0.56，Finetune与FromScratch差异不大，部分模型表现接近随机。

**⚠️ 局限性**

LRP在低直观域的解释仍需专家经验，热图可能产生偏见；FM在这些任务上性能不显著优于专用CNN，缺乏系统的faithfulness验证，且发现的Shortcut行为需进一步检验。

---

## 494. $h$-control: Training-Free Camera Control via Block-Conditional Gibbs Refinement

**arXiv ID:** 2605.11871 | [PDF](https://arxiv.org/pdf/2605.11871v1)

**作者:** Yuzhu Wang `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 471638 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种训练自由的后验采样器 h-control，利用预训练流匹配模型在摄像机控制任务中通过块条件伪 Gibbs 细化未观测空间来改进采样质量。

**💡 创新点**

将局部性假设下的块条件伪 Gibbs 采样与 Δ‑Welford 冻结门相结合，证明收敛到部分观测条件分布，避免手工调节指导强度。

**🔧 技术方法**

采用 Doob 的 h‑transform、流匹配（FlowMatch）框架、基于噪声的伪 Gibbs 迭代、Polyak 均值读出、块局部性测量与冻结门等技术。

**📊 数据集**

在公开视频数据集 RealEstate10K 与 DAVIS 上进行评估。

**📈 对比分析**

与四个训练自由基线（WorldForge、TTM、RePaint、Coarse‑Guided）及三种训练基线（TrajectoryAttention、TrajectoryCrafter、ReCamMaster）比较，h‑control 在 FVD、轨迹误差与 CLIP 分数等多项指标上均优于所有训练自由方法，并在大多数指标上接近或优于训练基线，显著提升视频质量与轨迹一致性。

**⚠️ 局限性**

需要针对不同流匹配骨干重新测量局部性阶数；以及仍依赖深度估计与重投影的几何准确性，深度误差会削弱条件强度。

---

## 495. FIS-DiT: Breaking the Few-Step Video Inference Barrier via Training-Free Frame Interleaved Sparsity

**arXiv ID:** 2605.11869 | [PDF](https://arxiv.org/pdf/2605.11869v1)

**作者:** Jian Tang `[一作]` (Tencent), Zheng Wei `[通讯]` (Tencent)

**通讯引用:** 27735 | [OpenAlex ID](https://openalex.org/A5010566708)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在少步视频扩散变压器（DiT）推理中，利用帧层稀疏计算并通过插值重建完整潜在序列，显著减少每步计算量。

**💡 创新点**

创新点在于将加速视角从跨步轨迹转移到单步帧维度，提出帧交错稀疏策略、块级与步级风险感知部署，以及无训练、无额外内存的插值重建方案。

**🔧 技术方法**

技术核心包括：局部线性插值重建、帧交错锚点调度、基于方差系数的块级稳定性判断、以及对最终几步的稀疏禁用。

**📊 数据集**

使用公开的 Wan 2.2（4 步、720p/480p）和 HunyuanVideo 1.5（8 步、480p）模型，在 VBench 与 CLIP 评测数据集上进行验证。

**📈 对比分析**

与传统跨步缓存方法（如 MagCache）对比，FIS‑DiT 在 4 步 Wan 2.2 下实现 2.11–2.41 倍加速、81 帧 720p 时延 127.6–111.7 秒，CLIP 与 VBench‑Q 分数保持与基线相近；在 HunyuanVideo 1.5 上亦实现 2.05–2.40 倍加速，视觉质量几乎无损。

**⚠️ 局限性**

局限性包括：在极端稀疏比例下会出现质量下降、插值重建仅为线性近似且对极端运动场景不够稳健、以及块级稳定性划分需基于先验热图手动选择，限制了自适应性。

---

## 496. IPI-proxy: An Intercepting Proxy for Red-Teaming Web-Browsing AI Agents Against Indirect Prompt Injection

**arXiv ID:** 2605.11868 | [PDF](https://arxiv.org/pdf/2605.11868v1)

**作者:** Chia-Pei `[一作]`, Alex Leung `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了IPI-proxy，一个开源的拦截代理工具，能够在企业白名单域的实时HTTP响应中插入隐藏的提示注入载荷，从而测试网页浏览型AI代理的安全性。

**💡 创新点**

创新点在于将攻击载荷注入到已授权域的真实响应中，解决了传统基准无法在白名单环境下测试的局限；同时整合了820条去重的跨基准载荷、三种嵌入技术和六个插入位置，形成可参数化的评估框架。

**🔧 技术方法**

技术实现依托mitmproxy实现网络层MITM代理、JSONL格式统一载荷库、HTML注释/不可见CSS/语义嵌入三种载荷包装方式，以及FastAPI编写的后门跟踪器记录成功的回调。

**📊 数据集**

载荷数据来源于六个公开基准（BIPIA、InjecAgent、AgentDojo、Tensor Trust、WASP、LLMail-Inject），经过去重后构成820条攻击字符串。

**📈 对比分析**

通过YAML配置对载荷、嵌入方式和插入位置进行组合实验，可在生产级别的白名单域上直接评估，而无需构造模拟页面；实验显示代理对网络延迟影响最小，能够在常规网络条件下高效运行。

**⚠️ 局限性**

局限性包括仅覆盖HTML文本注入，未支持PDF/图像/JSON等其他检索表面；对复杂页面结构或大量JS渲染的兼容性有限；以及缺乏对代理自身安全性与完整性保障的评估。

---

## 497. RealDiffusion: Physics-informed Attention for Multi-character Storybook Generation

**arXiv ID:** 2605.11927 | [PDF](https://arxiv.org/pdf/2605.11927v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 498. AuDirector: A Self-Reflective Closed-Loop Framework for Immersive Audio Storytelling

**arXiv ID:** 2605.11866 | [PDF](https://arxiv.org/pdf/2605.11866v1)

**作者:** Yiming Ren `[一作]` (Shanghai Artificial Intelligence Laboratory), Chao Zhang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 2137 | [OpenAlex ID](https://openalex.org/A5115596350)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AuDirector 多代理闭环框架，实现高质量长篇音频叙事生成。

**💡 创新点**

结合身份感知预制、协同合成纠错和人机交互修订，形成闭环自反性控制。

**🔧 技术方法**

采用 Gemini‑3‑Pro 作为导演与交互代理，嵌入语义检索与两步声源选择，层次化音频合成、Critic 评估、CLAP/MIMO‑Audio 等技术。

**📊 数据集**

使用 320 句语音样本构建多样化声库，评估集包含 100 个情境（40 个播客、60 个广播剧），以及 200 条自然语言编辑指令。

**📈 对比分析**

与 WavJourney、PodAgent 同一 LLM 与后端进行客观 AES、VRM 与主观 MOS 评测，AuDirector 在结构连贯、情感表达、音频保真等指标均优于基线，Critic 版提升多项指标。

**⚠️ 局限性**

对非语音音轨的细粒度建模不足，导致环境音的多样性和细微差别有限，影响沉浸感。

---

## 499. GEAR: Granularity-Adaptive Advantage Reweighting for LLM Agents via Self-Distillation

**arXiv ID:** 2605.11853 | [PDF](https://arxiv.org/pdf/2605.11853v1)

**作者:** Sijia Li `[一作]` (Hong Kong University Of Science And Technology), Rui Wang `[通讯]` (Hong Kong University Of Science And Technology)

**通讯引用:** 48267 | [OpenAlex ID](https://openalex.org/A5100687842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了GEAR（Granularity‑AdaptivE Advantage Reweighting）框架，用自蒸馏逆KL与熵动态分段对LLM代理在强化学习中的轨迹级优势进行细粒度重加权，解决长程轨迹的信用分配难题。

**💡 创新点**

创新点在于：①将同一模型的基于真值的教师分布与在政策下的学生分布的逆KL作为局部信用信号；②利用逆KL峰值与熵阈值相结合的动态分段策略，将信用分配从单个token扩展到语义连贯段；③通过优势符号感知的权重映射实现对正负轨迹的差异化加权，保持GRPO低方差特性。

**🔧 技术方法**

使用技术包括：基于GRPO的轨迹梯度优化；自蒸馏逆KL（student与ground‑truth‑conditioned teacher的KL差异）；token级熵作为分段终止阈值；分段级与token级权重组合；优势重加权与归一化；无额外奖励模型或手工过程奖励。

**📊 数据集**

实验数据集涵盖八个数学推理与工具使用基准：MATH、GSM8K、MATH500、AIME24/25、ToolSandbox、BFCL、ACEBench，并在Qwen3‑4B/8B上训练；此外还使用混合领域数据进行跨域泛化评估。

**📈 对比分析**

与GRPO、ARPO、MT‑GRPO、OPSD、OPSD+RL等基线对比，GEAR在Qwen3‑4B/8B上平均提升约10‑15%（最优约20%），在困难任务与跨域评估中优势更为显著，表明细粒度信用分配显著提升代理性能。

**⚠️ 局限性**

局限性包括：对逆KL/熵阈值的敏感性，分段策略在极长序列或非语义切换强的任务中可能不稳定；实验集中在单一LLM架构和特定任务集，缺乏对更广泛模型和对抗攻击鲁棒性的验证。

---

## 500. A Fast and Energy-Efficient Latch-Based Memristive Analog Content-Addressable Memory

**arXiv ID:** 2605.11847 | [PDF](https://arxiv.org/pdf/2605.11847v1)

**作者:** Paul-Philipp Manea `[一作]`, Luca Buonanno `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并验证了一种基于强臂锁存式忆阻器的模拟CAM（SALM），实现可扩展、低功耗的关联搜索，提升随机森林推理效率。

**💡 创新点**

创新点在于用动态电流竞争比较器替代传统静态电压分配，消除静态搜索功耗、显著提高增益并实现结果锁存；并引入顺序与并行共享锁存的可扩展架构。

**🔧 技术方法**

采用忆阻器（RRAM）、强臂锁存比较器、22 nm FD‑SOI CMOS工艺、SPICE LUT行为模型以及X‑TIME编译器进行设计、仿真与评估。

**📊 数据集**

使用公开的随机森林数据集（Churn、Digits、Forest Cover、Gas、Iris、Wine）进行实验验证。

**📈 对比分析**

与传统6T2M设计对比，SALM在相同延迟下读能降低约33%；在大阵列长度下保持接近软件精度；通过顺序扩展可将能耗降低50%并在多数数据集上保持准确率。

**⚠️ 局限性**

局限性包括较大的单元面积、需要多次顺序比较导致延迟增加，以及在极大阵列或极低误差容忍度场景下可能出现的误差累积。

---

## 501. On the LSH Distortion of Ulam and Cayley Similarities

**arXiv ID:** 2605.11921 | [PDF](https://arxiv.org/pdf/2605.11921v1)

**作者:** Flavio Chierichetti `[一作]` (Reddit), Erasmo Tani `[通讯]` (Sapienza University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5031393048)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究了在置换集合上两种常用相似度（Ulam 相似度和 Cayley 相似度）的 LSH 失真（distortion）问题，并给出了相应的上下界。

**💡 创新点**

创新点在于：①首次引入并量化“LSH 失真”概念，用以评估非 LSHable 相似度的可哈希性；②通过随机哈希与 wreath product 放大技术为 Ulam 相似度得到子线性上界；③利用对称群的表示理论和 Roichman 定理证明 Cayley 相似度的失真是线性的。

**🔧 技术方法**

主要技术包括：
- 随机哈希与记录最大值（record）序列的概率分析；
- 置换的 wreath product 与 Kronecker 积构造，用于放大 witness 的失真；
- 正定核（PSD kernel）的矩阵理论；
- 对称群不可约表示的字符分解及其上界估计。

**📊 数据集**

本工作为理论研究，没有使用具体数据集；所有结果均基于置换群的组合与表示理论推导。

**📈 对比分析**

由于是纯理论证明，没有与实验方法对比；提供的上界为 O(n/√log n)（Ulam）和 O(n)（Cayley），下界分别为 Ω(n^0.12)（Ulam）和 Ω(n)（Cayley），表明两种相似度在 LSH 可哈希性上差距显著。

**⚠️ 局限性**

局限性包括：
- Ulam 相似度的上界与下界之间仍有较大间隙，尚未确定精确失真阶数；
- 基础案例（n=8）是通过 LLM 生成，进一步扩展可能受限于计算资源；
- 当前方法难以直接推广到其他置换相似度（如 reversal），需要新的技术。

---

## 502. Rethinking Positional Encoding for Neural Vehicle Routing

**arXiv ID:** 2605.11910 | [PDF](https://arxiv.org/pdf/2605.11910v1)

**作者:** Chuanbo Hua `[一作]` (KAIST), Jinkyoo Park `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并改进神经网络在车辆路径规划（VRP）中的位置编码（PE）方法，提出一种兼顾距离感知、循环性与层级结构的分层异形PE（IPE+XPE），并在多种VRP变体上验证其有效性。

**💡 创新点**

创新点在于：①系统化定义三大路由特征（距阵异形、循环性、层级性）并以此为准则对现有PE进行分类；②设计出两部分分层PE——基于累计距离的循环正弦编码与基于枢纽极角的跨路线编码，完整满足三大特征；③在改进式Transformer解码器中直接插入该PE，证明其对解决质量的显著提升。

**🔧 技术方法**

主要技术包括Transformer架构、位置编码（sinusoidal、RoPE、DACT、CycleFormer等）对比，距离索引化正弦编码、极角正弦编码，以及与坐标特征的联合投影；实验采用标准的训练与搜索流程，使用预训练的解码器和搜索器。

**📊 数据集**

数据集涵盖四种VRP变体：CVRP、VRPTW、PCVRP、PDTSP，实例规模分别为N={500,1000,2000}（CVRP/VRPTW/PCVRP）和N=51（PDTSP），并在公开的CVRPLib等分布下进行外部测试。

**📈 对比分析**

方法比较：在同一底层模型（如DACT、NDS、N2S）下，用不同PE训练并在相同测试集上对比；评估指标为成本偏差与最优解偏差，结果显示IPE+XPE在所有问题尺寸上均优于传统PE，甚至在大型实例上逼近或超过经典运筹学求解器。

**⚠️ 局限性**

局限性：实验主要集中在改进式解码框架，对构造式模型的验证不足；对图Transformer PE（Laplacian PE、RWSE）适配尚未完成；跨多枢纽或更复杂约束的VRP未覆盖。

---

## 503. Mobile Traffic Camera Calibration from Road Geometry for UAV-Based Traffic Surveillance

**arXiv ID:** 2605.11900 | [PDF](https://arxiv.org/pdf/2605.11900v1)

**作者:** Alexey Popov `[一作]` (Embedded Intelligence Lab), Vadim Vashkelis `[通讯]` (Embedded Intelligence Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aaccfe5c-6b26-4208-b23c-35331481e142` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一个轻量级的无人机交通监控校准流程，将单目航拍视频转化为局部度量鸟瞰视图，用于车辆轨迹、速度、方向和三维立方体的分析。

**💡 创新点**

创新点在于使用可见道路几何（车道线、道路边界、斑马线）手工或半自动地估计单一平面单应矩阵，实现无需预先部署的移动交通摄像机校准，并在无人机视角下生成同步的 BEV 与 3D 立方体可视化。

**🔧 技术方法**

技术包括基于单应变换的逆透视映射、车辆底部中心点投影、轨迹求导速度和方向估计、类属尺寸先验生成定向立方体，以及人机交互校准验证。

**📊 数据集**

实验使用 UAVDT 数据集，聚焦 M1401 序列（40 帧），利用其公开的标注数据来评估校准与几何恢复的效果。

**📈 对比分析**

与完全自动线检测+消失点估计的基线相比，手工校准的 BEV 轨迹更为稳定，速度误差在 8.1 m/s 左右，车辆立方体与原始图像对齐良好，显示出相对低的错误率。

**⚠️ 局限性**

局限性主要是单平面假设导致对非平坦或视角变化大的道路区域失效，远场车辆对单应误差敏感，且当前仅采用手工校准，自动化精度仍待提升。

---

## 504. Fast Computation of Conditional Probabilities in MDPs and Markov Chain Families

**arXiv ID:** 2605.11897 | [PDF](https://arxiv.org/pdf/2605.11897v1)

**作者:** Milan Češka `[一作]` (Brno University of Technology), Tim Quatmann `[通讯]` (RWTH Aachen University)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5013247085)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于多目标 MDP 视角的新算法，用于高效计算 MDP 和 Markov 链族的条件可达概率；

**💡 创新点**

创新点在于将条件可达性转化为一次总奖励查询，配合线性时间的阈值判定和精确的分裂二叉搜索，从而避免了传统重启方法所带来的循环和收敛问题；

**🔧 技术方法**

采用多目标 MDP 模型检查、期望总奖励求解、线性规划、值迭代、二分搜索以及抽象-精化（AR）框架来实现；

**📊 数据集**

使用来自 BNs、runtime 监控、参数化马尔可夫链、以及已公开的彩色 MDP 基准（如 ceas、ladder、airport 等）共计数百个实例；

**📈 对比分析**

与重启方法及传统值迭代/线性规划做对比，实验显示在绝大多数基准上平均速度提升数十倍，甚至在某些无环模型上提升多阶；

**⚠️ 局限性**

局限包括对符号状态空间支持不足、需要先验可达性判定、对极大规模 Markov 链族的精确性仍受限于多目标求解的指数复杂度。

---

## 505. Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models

**arXiv ID:** 2605.11887 | [PDF](https://arxiv.org/pdf/2605.11887v1)

**作者:** Boyi Deng `[一作]` (Qwen Team), Jingren Zhou `[通讯]` (Qwen Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了 Qwen‑Scope，一套基于 Qwen 系列模型的稀疏自编码器（SAE）工具箱，并展示了其在推理调控、评估分析、数据分类与合成、监督微调以及强化学习等四大实际工作流中的应用。

**💡 创新点**

创新点在于将 SAE 从单纯的解释工具转变为可直接操纵模型内部表示的“接口”，通过特征级干预实现推理时的语言/概念调控、基于特征覆盖的评测重构、跨语言毒性检测、以特征驱动的安全数据合成、以及将 SAE 信号嵌入 SFT/RL 损失以抑制代码切换和无穷重复等异常行为。

**🔧 技术方法**

核心技术包括：层级稀疏自编码器训练（Top‑k sparsity、辅助去死特征损失），特征级对比识别、交叉语言特征匹配，基于特征覆盖率的评测重构与相似度分析，基于特征说明的自动化数据合成与验证，以及在 SFT 中加入特征抑制正则项和在 DAPO 中生成稀有负样本的 SAE 驱动滚动。

**📊 数据集**

使用的数据集包括：Qwen3/3.5 预训练模型内部激活、公开多语言毒性数据集、WildJailbreak、安全 SFT 数据、以及多种标准评测基准（MMLU、GSM8K、MATH、MBPP、C‑Eval、MMLU‑Redux 等）。

**📈 对比分析**

与传统方法相比，Qwen‑Scope 在 17 组评测基准上特征冗余度与性能冗余度的斯皮尔曼相关系数≈0.85；在毒性分类中单特征 F1 超过 0.90；在安全数据合成上，特征驱动策略在相同样本预算下实现了 99.7% 的目标特征覆盖率；在 SFT 中，SASFT 将代码切换比例降低 50%+，并保持或提升多语言基准得分；在 RL 中，SAE‑驱动的负样本生成使无穷重复率显著下降，且总体能力保持竞争。

**⚠️ 局限性**

局限性包括：SAE 仍需先验训练，且覆盖率可能遗漏隐藏的内部机制；特征识别依赖阈值和对比集，易受样本偏差影响；跨语言特征迁移并非完全通用，远距离语言效果不佳；对特征驱动的安全合成仍受生成模型质量限制；在 RL 中引入负样本可能对其他未目标行为产生副作用。

---

## 506. When Brains Disagree: Biological Ambiguity Underlies the Challenge of Amyloid PET Synthesis from Structural MRI

**arXiv ID:** 2605.11867 | [PDF](https://arxiv.org/pdf/2605.11867v1)

**作者:** Louise E. G. Baron `[一作]` (University College London), Hui Zhang `[通讯]` (University College London)

**通讯引用:** 22487 | [OpenAlex ID](https://openalex.org/A5100323374)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

通过控制训练数据的生物学一致性与不一致性，探究MRI到淀粉样蛋白PET合成的可行性，并验证加入血浆生物标志物是否能消除映射模糊性。

**💡 创新点**

证明了医学上“生物学模糊性”是导致MRI合成PET性能不稳定的根本原因，并展示了多模态（MRI+血浆）输入可以显著恢复并稳定预测性能。

**🔧 技术方法**

使用了两个主流生成模型（pix2pix 条件GAN 与 3D 潜在扩散模型 LDM），并在 pix2pix 生成器中加入了 FiLM 模块实现血浆信息的条件化。

**📊 数据集**

采用 ADNI 数据库中的 2,884 对 T1w MRI 与 AV45 PET 影像，以及 589 对具有相应血浆标志物（pT217, Aβ42/40, NfL, GFAP）的影像。

**📈 对比分析**

对照实验：① 训练时仅包含生物学一致（A+/N+ 或 A-/N-）或不一致（A+/N- 或 A-/N+）样本，② 将全部数据混合训练；评估指标包括 PSNR、SSIM、平衡准确率、敏感性/特异性以及皮尔逊相关系数。实验表明：在生物学一致的训练集上，pix2pix 和 LDM 均能取得 ~80% 的平衡准确率；而混合训练导致准确率下降至 ~57% 且相关系数仅 0.16。加入血浆标志物后，平衡准确率提升至 85% 以上，相关系数提升至 0.75，性能在一致与不一致子集上保持稳定。

**⚠️ 局限性**

主要限制在于血浆标志物的可用性（仅覆盖 589 对影像），以及实验仅验证了单一 MRI 协议与单一 PET 跟踪器，未来需在更大、多中心、多模态数据上进一步验证方法的泛化能力。

---

## 507. Avoiding Cross-Datacenter Collective Congestion via Disaggregated Buffering

**arXiv ID:** 2605.11852 | [PDF](https://arxiv.org/pdf/2605.11852v1)

**作者:** Mariano Scazzariello `[一作]` (RISE Research Institutes of Sweden), Mark Silberstein `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在多数据中心LLM训练中，作者设计了透明的网络机制，在目的地数据中心使用外部缓冲区（spillway）保存因跨域通信与本地聚合冲突导致的丢包，并在拥塞消失后有序释放，从而避免重传延迟。

**💡 创新点**

创新点在于将跨域丢包处理从深层交换机缓冲转移到可分离的外部缓冲区，并结合“丢包时偏转”与“安静间隔+探测+分段释放”三段控制，实现了微秒级的释放时序，兼容现有RDMA/Collective框架；并通过任何播与“粘性”映射实现负载均衡。

**🔧 技术方法**

技术主要包括：在交换机上实现丢包偏转（deflect-on-drop）并使用GRE/ICMP或L3包封装转发至spillway；spillway节点采用多队列RSS、缓冲池、安静间隔探测、分段释放算法；在出口交换机注入快速CNP以保持CC反馈；在硬件层面使用NVIDIA Spectrum‑4和BlueField‑3 DPU。

**📊 数据集**

使用的工作负载为基于MLSynth生成的Chakra trace，模拟24B参数稀疏MoE模型的分布式训练（4个pipeline stage、64个GPU、8个microbatch），以及简化的microbenchmark包含16条250MB跨域流和4GB本地聚合。

**📈 对比分析**

与基线（RDMA+RTO重传）、预置偏转等方案对比，ns‑3仿真显示在5–30ms RTT下微批次时间可提升至14%，迭代时间提升5%；硬件原型在单流FCT上可降低40%，并在极端骨干拥塞下仍保持1.08×的低慢速率，显著优于预置偏转的≈2×。

**⚠️ 局限性**

局限性包括：需要交换机支持丢包偏转与任播/粘性映射，假设环境支持乱序交付；quiet‑interval参数需手工调优；在极端骨干拥塞或严格顺序交付场景下效果未知；实现需要额外的外部缓冲资源和软硬件协同；未对高RTT（>30ms）或不同拓扑的可扩展性做完整评估。

---

## 508. Martingale-Consistent Self-Supervised Learning

**arXiv ID:** 2605.11846 | [PDF](https://arxiv.org/pdf/2605.11846v1)

**作者:** Moritz Gögl `[一作]` (University of Oxford), Christopher Yau `[通讯]` (University of Oxford)

**通讯引用:** 19649 | [OpenAlex ID](https://openalex.org/A5084571119)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在自监督学习中引入马尔可夫一致性约束，使模型在信息逐步完善时保持预测期望一致，避免系统性漂移。

**💡 创新点**

提出无偏的两样本马尔可夫一致性损失，并在预测空间与潜在空间实现，可与现有 SSL 目标（SimCLR、BYOL 等）无缝结合。

**🔧 技术方法**

使用编码器+预测头、随机填充网络（imputer）进行条件采样，利用指数移动平均（EMA）目标网络，结合两样本 Monte Carlo 估计条件期望。

**📊 数据集**

在多种数据集验证：时间序列（T‑SIM‑RC、HAR、TRAF、STK、CT、SAD）、静态表格（Adult、Bank Marketing、Credit‑g、Phoneme、S‑SIM）和图像（CIFAR‑10、STL‑10），以及 TCGA 多组学生存分析。

**📈 对比分析**

与原始 SSL 基线和仅使用填充的基线相比，马尔可夫一致性在低信息量下平均提升 10–30%（时间序列）和 20–50%（图像）下游准确率，且在校准与马尔可夫违规率上均显著改善。

**⚠️ 局限性**

局限性在于对填充网络近似条件分布的依赖，估计方差相对较大；在大规模模型、多模态或复杂任务中的可扩展性与效率仍需进一步研究。

---

## 509. StepCodeReasoner: Aligning Code Reasoning with Stepwise Execution Traces via Reinforcement Learning

**arXiv ID:** 2605.11922 | [PDF](https://arxiv.org/pdf/2605.11922v1)

**作者:** Hao Wang `[一作]` (Beihang University), Jie M. Zhang `[通讯]` (King's College London)

**通讯引用:** 4334 | [OpenAlex ID](https://openalex.org/A5088708850)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在程序中自动插入执行状态锚点，让模型在推理过程中预测中间执行状态，从而提升代码推理和生成质量。

**💡 创新点**

创新点在于显式引入中间执行状态监督，并设计双层 Bi‑Level GRPO 算法实现对每一步执行的精细信用分配。

**🔧 技术方法**

使用 LLMS 生成锚点、结构化提示、Bi‑Level GRPO 强化学习以及监督微调等技术。

**📊 数据集**

训练数据来自 CodeReasoner 数据集，经过 LLMS 注入锚点后构建 CRUXEval、LiveCodeBench、REval 等标注集。

**📈 对比分析**

与 CodeReasoner、GPT‑4o、Qwen 等模型对比，StepCodeReasoner 在 CRUXEval、LiveCodeBench、REval 等基准上平均得分 0.878/0.858/0.893，能在 7B 参数规模下击败 GPT‑4o 并超越同等规模对手。

**⚠️ 局限性**

局限性包括仅适用于 Python、锚点选择有限、难以捕获循环内细粒度状态、异步/文件网络等副作用，以及推理时额外令牌导致计算成本上升。

---

## 510. Procedural-skill SFT across capacity tiers: A W-Shaped pre-SFT Trajectory and Regime-Asymmetric Mechanism on 0.8B-4B Qwen3.5 Models

**arXiv ID:** 2605.11907 | [PDF](https://arxiv.org/pdf/2605.11907v1)

**作者:** Igor Strozzi `[一作]` `[通讯]` (Federal University of Rio de Janeiro), Igor Strozzi (Federal University of Rio de Janeiro)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在三种 Qwen3.5 LLM（0.8B、2B、4B）上使用 353 行手工制备的程序化技能训练数据，对 200 任务/40 技能的 holdout 进行 SFT 评估，比较 pre‑SFT 基础与 post‑SFT 的性能并分析 SFT 对程序化技能的提升。

**💡 创新点**

发现 SFT 在不同规模模型中对程序化技能的提升呈 W‑形预基准轨迹，且 SFT 在困难（0.8B、4B）规模下的绝对提升最大，揭示 SFT 的机制是“基准不平衡”而非纯规模效应；同时揭露并纠正了基准格式合规偏差，提供 LLM‑only 重新评判方案。

**🔧 技术方法**

使用 LoRA 微调、Opus 4.7 生成技能流程与任务合成、基准追踪、LLM‑only 重新评判、GPT‑5.4 与 Opus 4.7 的跨族评判交叉验证，以及 McNemar 与 bootstrap 统计方法。

**📊 数据集**

使用 40 项程序化技能手工编制、Opus 4.7 生成的 200 任务 holdout（分为 400 训练/200 评估）、353 行 SFT 训练语料，和 52 题 OOD 一般性探测集。

**📈 对比分析**

采用匹配路径的 LLM‑only 评判与 deterministic‑mixed 评判，比较 pre‑SFT 与 post‑SFT 的 pass‑rate 与 Δ‑lift，发现 Δ‑lift 在 0.8B、2B、4B 上分别为 +0.070、+0.040、+0.075，绝对贡献为 +0.115、+0.100、+0.145；v2.0（4B+SFT）与 Haiku‑4‑5 在 200 任务上达到 0.985 的最高分。

**⚠️ 局限性**

单一随机种子、基准格式合规偏差、Opus 生成与评判的重叠、仅使用 Qwen3.5、缺乏通用指令调优对照、n=200 限制统计精度、解码使用贪婪、OOD 探测不量化、Haiku 推理路径不匹配。

---

## 511. Rethinking Supervision Granularity: Segment-Level Learning for LLM-Based Theorem Proving

**arXiv ID:** 2605.11905 | [PDF](https://arxiv.org/pdf/2605.11905v1)

**作者:** Shuo Xu `[一作]` (Nanjing University), Jingwei Xu `[通讯]` (Nanjing University)

**通讯引用:** 21220 | [OpenAlex ID](https://openalex.org/A5100784495)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大型语言模型在 Lean 4 自动定理证明中的监督粒度问题，提出了基于“开关目标数”变化的段级监督策略，并将相同信号用于推理时的目标感知回放，从而在训练和推理两方面提升证明成功率与搜索效率。

**💡 创新点**

创新点在于：①把监督粒度视为证明轨迹的边界选择问题；②利用 Lean 证明器提供的开关目标数（open‑goal count）作为轻量级结构信号，自动划分局部连贯的证明段；③在推理阶段以相同信号触发短回放（Goal‑aware Rollout），既压缩搜索树又保持局部一致性；④在不改动模型架构或额外生成数据的前提下，实现了训练目标与搜索接口的统一。

**🔧 技术方法**

技术手段包括：基于开关目标数的边界选择、段级训练数据构造、Qwen2.5‑Math‑7B 作为策略模型、树搜索（Best‑First、Beam Search 等）与回放策略的结合、对比实验、损失曲线分析以及不同边界策略的 ablation 研究。

**📊 数据集**

使用的数据集为：STP、LeanWorkbook、NuminaMath‑LEAN（用于训练）；在 miniF2F 以及对应的 in‑domain 测试集上评估性能，所有数据均来自 Lean 4 官方或公开证明库。

**📈 对比分析**

比较方法：将段级监督与 step‑level 及 whole‑proof 监督在同一模型、同一训练设置下对比；在 miniF2F 上测算证明成功率、token 与时间成本；并将段级监督与“whole‑proof‑seg”搜索方式对比。实验结果表明：在 miniF2F 上段级监督分别达到 64.84%、60.90% 与 66.31% 的成功率，均优于两种基线；对现有 step‑level 推理器的目标感知回放可将成功率提升至 70.74%（BFS‑Prover‑V2‑7B）和 60.33%（InternLM2.5‑StepProver），同时显著降低 token 与时间消耗。

**⚠️ 局限性**

局限性：①仅在 Lean 4 体系下验证，未验证在其他证明助手的通用性；②依赖开关目标数信号的局部性，极短或结构非常平滑的证明中收益有限；③对超长证明的生成仍受限，未解决全程生成与局部生成之间的权衡；④实验主要集中在 Qwen2.5‑Math‑7B，需进一步评估在不同 LLM 大小与架构上的迁移效果。

---

## 512. The Future of Scholarly Blogs: Scholarly Bloggers' Perspectives on Long-Term Preservation

**arXiv ID:** 2605.11902 | [PDF](https://arxiv.org/pdf/2605.11902v1)

**作者:** Catharina Ochsner `[一作]` (Humboldt-Universität zu Berlin), Heinz Pampel `[通讯]` (Humboldt-Universität zu Berlin)

**通讯引用:** 804 | [OpenAlex ID](https://openalex.org/A5023356598)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过访谈13位德国学术博客作者，系统性地梳理了学术博客在长期保存与信息基础设施整合中遇到的挑战与需求。

**💡 创新点**

提出“去中心化信息基础设施”作为解决方案，并用Star & Ruhleder的基础设施维度为框架，首次将博客保存问题与基础设施理论结合。

**🔧 技术方法**

采用半结构化访谈和定性内容分析方法（Kuckartz & Rädiker），使用MAXQDA进行编码，不涉及具体技术实现。

**📊 数据集**

样本来源为Ochsner等人收集的866个德国学术博客列表，随机挑选代表性博客作者进行访谈。

**📈 对比分析**

本研究不进行实验性比较或性能评估；结论来自主题分析与访谈内容，没有量化指标。

**⚠️ 局限性**

局限性包括样本规模有限、受访者对博客保存认知水平不一、研究者自身与博客保存项目的关联可能引入偏见，且研究聚焦德国，缺乏跨国或跨文化验证。

---

## 513. Energy Consumption in Next Generation Radio Access Networks

**arXiv ID:** 2605.11899 | [PDF](https://arxiv.org/pdf/2605.11899v1)

**作者:** Urooj Tariq `[一作]` (Trinity College Dublin), Daniel Kilper `[通讯]` (Trinity College Dublin)

**通讯引用:** 4878 | [OpenAlex ID](https://openalex.org/A5034922871)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对5G和O‑RAN架构下的基站能耗进行基于事务的能耗建模，评估不同基带处理位置与网络密集度对能耗的影响。

**💡 创新点**

提出将处理能耗与传输能耗统一到同一模型中，并利用最新设备参数更新传统模型，揭示处理能耗在全链路能耗中的主导作用。

**🔧 技术方法**

使用事务式能耗模型（transaction‑based energy modeling）、eCPRI协议参数、开放RAN架构拆分（Split 7.2、8）以及仿真计算。

**📊 数据集**

使用行业标准网络设备参数（Cisco Catalyst 1300、Cisco 8000/9600、Benetel 650等）和用户平均10GB流量作为输入数据。

**📈 对比分析**

通过对四种基站部署方案（D‑RAN、O‑RAN、Edge‑CU、C‑RAN）在不同密集度下计算每用户能耗，结果显示处理能耗占比最高，中央化方案在高密度场景下能耗可下降约75%。

**⚠️ 局限性**

模型假设为平均流量、固定设备功率，未考虑动态功率管理、热管理、实际部署成本以及网络运营商特定负载波动。

---

## 514. LOFT: Low-Rank Orthogonal Fine-Tuning via Task-Aware Support Selection

**arXiv ID:** 2605.11872 | [PDF](https://arxiv.org/pdf/2605.11872v1)

**作者:** Lanxin Zhao `[一作]` (University of Sydney), Andi Han `[通讯]` (University of Sydney)

**通讯引用:** 92 | [OpenAlex ID](https://openalex.org/A5031625303)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的参数高效的正交微调框架 LOFT，明确分离适配子空间 P_r 与子空间内的正交变换 T_r，揭示并验证支持选择是影响性能的关键设计轴。

**💡 创新点**

创新点：①将正交 PEFT 的支持与变换拆分为可独立设计的两部分；②从一阶梯度分析得到任务感知支持选择的理论依据；③提出两种梯度启发的支持方法（GradSVD 与 SkewGrad），显著提升效率-性能权衡。

**🔧 技术方法**

技术：正交变换参数化（Cayley/指数映射），低秩正交子空间旋转，梯度引导的支持选取，理论分析（skew(W_0^⊤G)），多任务实验与对比评估。

**📊 数据集**

数据集：GLUE（CoLA、STS-B、RTE、MRPC、SST-2、QNLI）、MetaMathQA-40K（GSM8K、MATH）、VTAB-1K（19 视觉任务）以及附加的多语言 OOD、MNLI/QQP 等。

**📈 对比分析**

对比方法：LoRA、PiSSA、DoRA、LoRA-XS、GOFT、BOFT、OFT、PSOFT 等代表性 PEFT 方法；实验在匹配参数/内存/计算预算下进行。结果显示：LOFT+梯度支持在同等预算下平均分数提升约 0.6–0.7，参数量约为 LoRA/DoRA 的 1/20，显著降低内存占用；在视觉任务上，SkewGrad 提升平均准确率 0.4 点。

**⚠️ 局限性**

局限：①仅聚焦支持选择，未系统探索不同正交变换的效果；②梯度支持需要一次前向/反向推理进行校准，略增开销；③不同任务族对梯度支持的敏感度不同，需进一步研究自适应策略。

---

## 515. GATA2Floor: Graph attention for floor counting in street-view facades

**arXiv ID:** 2605.11863 | [PDF](https://arxiv.org/pdf/2605.11863v1)

**作者:** Ngoc Tan Le `[一作]` (Vrije Universiteit Brussel), Nikos Deligiannis `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 2516 | [OpenAlex ID](https://openalex.org/A5043511500)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了街景图像中建筑立面窗口/门元素的图形关系，并提出 GATA2Floor 模型实现楼层计数与软分配。

**💡 创新点**

创新点是将立面建模为垂直感知图并利用 GATv2+垂直注意力和跨注意力查询实现联合计数与分配，同时支持无标注轻量级候选生成。

**🔧 技术方法**

使用图注意力网络（GATv2）、垂直注意力、跨注意力查询、Self‑Supervised 特征、CLIP/GPT‑4o 轻量候选、Sobel 边缘、GMM 等技术。

**📊 数据集**

在 Amsterdam Facade、ECP、eTRIMS、ParisArtDecoFacades 等公开数据集上进行实验。

**📈 对比分析**

与 ResNet50 分类、KDE、Agglomerative、Intersection 等聚类基线比较，GATA2Floor 在 MAE、F1、Accuracy 上均优于基线，尤其在常规立面表现突出；无标注候选时性能略降。

**⚠️ 局限性**

局限性包括：需要每层至少检测到窗口/门；无标注候选覆盖率不足导致计数误差；视角失真或严重遮挡时垂直先验失效。

---

## 516. Improving the Performance and Learning Stability of Parallelizable RNNs Designed for Ultra-Low Power Applications

**arXiv ID:** 2605.11855 | [PDF](https://arxiv.org/pdf/2605.11855v1)

**作者:** Julien Brandoit `[一作]` (University of Liège), Guillaume Drion `[通讯]` (University of Liège)

**通讯引用:** 1031 | [OpenAlex ID](https://openalex.org/A5084189456)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了累积更新的 Bistable Memory RNN（CMRU/αCMRU），在保持持久记忆和低功耗硬件友好的前提下提升了平行可训练 RNN 的收敛稳定性与性能。

**💡 创新点**

创新点在于：① 用 ε 介入更新式，恢复梯度流，消除原 BMRU 的梯度阻塞；② 通过累积方式实现事件驱动的记忆衰减，兼顾持久记忆与线性可并行；③ 通过 ε-annealing 将训练后的模型映射回原 BMRU 结构，实现极低功耗模拟实现。

**🔧 技术方法**

使用技术包括：累积更新公式、ε-annealing 训练策略、αCMRU 的可学习量化尺度、关联扫描实现、与 SSM/Linear RNN 的对比实验，以及基于 Schmitt trigger 的模拟电路原型。

**📊 数据集**

数据集涵盖：Sequential MNIST、CIFAR-10、IMDb、ListOps、Pathfinder、Shakespeare 文字级语言模型、Google Speech Commands（Digits/All）以及合成任务（copy-first-input、parity）。

**📈 对比分析**

对比方法：在同一 backbone 下直接替换 RNN cell，统一超参数不调，使用 5 次随机种子评估平均 ± 极差；在多个任务上与 LRU、minGRU、SSM 等基线比较。结果显示：在低容量（d≤16）下 CMRU/αCMRU 与 LRU/minGRU 相当或更优，尤其在长程离散任务（ListOps、Pathfinder）和低功耗关键词识别中达到 90%+ 准确率，显著提升收敛稳定性并降低初始化敏感性。

**⚠️ 局限性**

局限性：实验聚焦低容量模型，未评估大规模超大容量性能；未做任务特定超参调优，可能低估基线峰值；对某些需要模数运算的任务（如 parity）需负 ε 或特殊设计；并行可训练架构在极大序列长度或更复杂任务上的可扩展性尚待验证。

---

## 517. Constacyclic codes of length $np^s$ over $\frac{\mathbb{F}_{p^m}[u]}{\langle u^t\rangle}$: Torsions and Cardinalities

**arXiv ID:** 2605.11912 | [PDF](https://arxiv.org/pdf/2605.11912v1)

**作者:** Akanksha Tiwari `[一作]` (Indian Institute of Technology Delhi), Ritumoni Sarma `[通讯]` (Ohio University Zanesville)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

对有限链环R^t=𝔽_p^m[u]/⟨u^t⟩上多项式商环R^t[x]/⟨x^np^s-δ⟩的理想结构进行研究，给出显式生成元，并将结果应用于长度为p^s、2p^s和3p^s的常数循环码；

**💡 创新点**

提出了一种统一的框架，能够在不同t值下对这些商环理想进行完整分类，显式推导了t=3时的理想参数和码长为p^s、2p^s、3p^s的常数循环码的基数；

**🔧 技术方法**

利用有限链环理论、理想生成元的显式构造以及对多项式因子分解的数论方法；

**📊 数据集**

无具体实验数据集，研究完全基于理论推导；

**📈 对比分析**

与已有的单场有限域常数循环码分类进行比较，结果表明对链环的推广保持了与原有分类一致的结构和基数公式；

**⚠️ 局限性**

局限在于只考虑了(n,p)=1且δ为单位的情况，且对高阶t值或其他特殊多项式因子分解情况的细节仍需进一步研究。

---

## 518. Beyond Point-wise Neural Collapse: A Topology-Aware Hierarchical Classifier for Class-Incremental Learning

**arXiv ID:** 2605.11904 | [PDF](https://arxiv.org/pdf/2605.11904v1)

**作者:** Huiyu Yi `[一作]` (Nanjing University), Furao Shen `[通讯]` (Nanjing University)

**通讯引用:** 1337 | [OpenAlex ID](https://openalex.org/A5036608458)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对类增量学习中单原型NCM的不足，提出了层次聚类+SOINN相结合的拓扑感知分类器HC‑SOINN，并设计了STAR模块通过点级残差跟踪实现非线性特征漂移的主动适配。

**💡 创新点**

创新点在于①将层次化聚类初始化与SOINN增量学习相融合，构建局部到全局的类别拓扑表示；②通过STAR对每个子原型的点轨迹进行残差跟踪与自适应变形，突破传统对齐仅限全局刚性变换的限制；③在不需额外训练的情况下实现特征空间的非线性漂移适配。

**🔧 技术方法**

技术方法包括：Neural Collapse理论、Procrustes距离分析、Spherical SOINN（含SLERP更新）、层次聚类（平均链接+余弦距离）、余弦相似度评分、EMA平滑、点级残差变形、双视图推断（全局+局部），以及与预训练ViT backbone的无缝集成。

**📊 数据集**

实验使用的公开数据集为：Split CIFAR‑100、Split CUB‑200 与 Split ImageNet‑R（分别覆盖通用、细粒度与抗干扰场景）。

**📈 对比分析**

与七种主流CIL方法（DualPrompt、CODA‑Prompt、SEMA、CL‑LoRA、SimpleCIL、APER、EASE）在保持同一backbone与prompt设置下替换分类器后进行对比；实验显示HC‑SOINN在A_avg和A_last上普遍提升，部分组合可逼近或超过最新SOTA，证明其显著提升性能且稳定。

**⚠️ 局限性**

局限性包括：相较于单原型NCM，HC‑SOINN在训练与推断上略有计算与存储开销；STAR需要维护额外的锚点与残差；当前推断仅利用节点距离，未充分利用拓扑边信息；动态拓扑结构可能带来可解释性与安全验证挑战。

---

## 519. AccLock: Unlocking Identity with Heartbeat Using In-Ear Accelerometers

**arXiv ID:** 2605.11901 | [PDF](https://arxiv.org/pdf/2605.11901v1)

**作者:** Lei Wang `[一作]` (Soochow University), He Huang `[通讯]` (Soochow University)

**通讯引用:** 21963 | [OpenAlex ID](https://openalex.org/A5100782959)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于耳机内加速度计捕获的BCG信号的被动连续身份验证系统 AccLock，用户和设备均不需主动交互。

**💡 创新点**

创新点包括：①首次使用仅靠耳机加速度计捕获的BCG实现完全被动身份验证；②提出两阶段去噪（小波+超平面阈值+RLS）以抑制内在与短时运动干扰；③设计 HIDNet 通过梯度逆转、正交正则化与重建约束实现身份特征与共享噪声的显式解耦；④采用用户无关的 Siamese 三元组学习实现一次性训练即可对新用户进行认证，免除每人单独模型训练。

**🔧 技术方法**

核心技术包括：离散小波变换与自适应阈值去噪、递归最小二乘（RLS）滤波、基于 Transformer 的共享编码器、梯度逆转对抗网络、正交正则化、信息完整性重建、Siamese 三元组损失、三轴融合与注意力机制、BLE 失真增强、动态阈值生成。

**📊 数据集**

使用了 33 名参与者的实验数据（每人 240 条 4 秒 BCG 片段，采样率 100 Hz，另外在 AirPods 适配实验中使用 25 Hz 采样），共计约 33 分钟数据；数据集包括静态、运动、姿势、环境、心脏疾病等多场景录制。

**📈 对比分析**

与 CNN、ResNet50、Conformer、LSTM 等基线模型对比，AccLock 在三折交叉验证下平均 FAR 为 3.13%、FRR 为 2.99%，且在所有对比实验中均显著优于基线（p < 0.05）。在不同训练样本、段长度、注册样本、运动、姿势、长时使用、角度、音乐、环境、心脏疾病、采样率和 BLE 丢包等多维度评估中均保持 3–5% 级别误差；在 AirPods 适配实验中也达到 FAR ≈ 7.5%、FRR ≈ 7.1%。

**⚠️ 局限性**

局限性包括：①对大幅运动（如跑步）时性能下降严重；②商业耳机无法直接获取原始 100 Hz 加速度数据，只能在低采样率下运行；③部分用户因耳道结构、佩戴稳定性导致 BCG 信号可分离性不足，误码率略高；④系统需在新设备上进行适配和轻量化微调。

---

## 520. Few-Shot Synthetic Data Generation with Diffusion Models for Downstream Vision Tasks

**arXiv ID:** 2605.11898 | [PDF](https://arxiv.org/pdf/2605.11898v1)

**作者:** Daniil Dushenev `[一作]` (National University of Science and Technology MISIS), Konstantin Kulikov `[通讯]` (National University of Science and Technology MISIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用LoRA轻量级微调预训练扩散模型，仅用20–50张少量正样本生成合成数据，再将合成数据与真实数据一起训练下游分类器，以提升罕见类的检测性能。

**💡 创新点**

在极少样本场景下，首次证明LoRA微调的扩散模型能高效生成对罕见类有用的合成图像；并系统地探究合成与真实样本比例对性能的影响，揭示中等比例合成最佳。

**🔧 技术方法**

LoRA参数高效微调、DreamBooth式自监督训练、FLUX.2-dev预训练扩散模型、ResNet‑18分类器、LPIPS/PSNR多样性评估。

**📊 数据集**

NIH ChestX‑ray14（胸部X光罕见病理分类）与Magnetic Tile Defect数据集（工业表面裂纹检测）。

**📈 对比分析**

在保留相同真实测试集的前提下，使用F1、PR‑AUC和Recall进行对比。相较于仅用真实样本，合成增强能显著提升F1（Chest X‑ray最高达0.686，Magnetic Tiles最高0.296），Recall也随合成比例增大而提升，最佳比例为4×；但过多合成会出现性能递减。

**⚠️ 局限性**

合成数据仍无法完全替代真实样本，过量合成可能导致分布失衡；实验仅覆盖两类结构差异较大的任务，需进一步验证在更多视觉域与更复杂任务中的泛化性。

---

## 521. Proteus: A Self-Evolving Red Team for Agent Skill Ecosystems

**arXiv ID:** 2605.11891 | [PDF](https://arxiv.org/pdf/2605.11891v1)

**作者:** Zhaojiacheng Zhou `[一作]` (Shanghai Jiao Tong University), Zhaojiacheng Zhou `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5112264504)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并实验了一种自我进化的红队框架（Proteus），用于衡量在第三方技能（skill）生态系统中，攻击者在多轮反馈循环中迭代修改技能后能否同时通过审计并造成运行时危害的风险（adaptive leakage）。

**💡 创新点**

创新点包括：
1) 将“adaptive leakage”定义为攻击者在预算内可迭代重写技能并成功逃逸审计的部署风险；
2) 设计了基于五轴（攻击目标、拓扑、代码、侧效通道、文档）的技能攻击空间；
3) 采用灰盒反馈循环（审计、沙盒、oracle）驱动的自我进化红队；
4) 引入路径扩张（path expansion）与表面扩张（surface expansion）两层演化，提升多样性与跨目标迁移能力。

**🔧 技术方法**

使用的技术包括：
- 大语言模型（DeepSeek‑V4‑Pro / Kimi‑2.6）进行代码/文档重写与链组合；
- 结构化审计输出转换为统一的发现模式；
- OpenClaw 沙盒+OpenClaw 运行时与目标骨干（GPT‑5.4‑mini / GLM‑5）交互；
- 以图为导向的链拓扑搜索与 BM25 基于历史日志的检索；
- 多轮实验框架，记录 ASR@5、学习曲线斜率(LCS) 与策略多样性增长(SDG)。

**📊 数据集**

使用的数据集与资源：
- ClawHub 技能图（约47k节点，16种有向风险边）；
- 20 条手工挑选的恶意种子技能；
- 10 个多域目标仓库（交易、DevOps、数据科学、RAG、CI/CD等）；
- 两个公开审计器（SkillVetter、AI‑Infra‑Guard）；
- 两个目标模型骨干（GPT‑5.4‑mini、GLM‑5）。

**📈 对比分析**

与基线（Random、Zero‑Shot、Blackbox、PyRIT）以及不同审计器、目标模型、攻击者模型的组合进行对比。实验结果显示：
- 在 8 个（攻击者、目标、审计器）配置中，Proteus 在 5 轮预算下的 ASR@5 在 40–90% 之间，平均约 80%；
- 学习曲线斜率（LCS）显著高于基线（0.16 对比 ≤0.014）；
- 对于 SkillVetter，所有细胞均达 93% 以上的迭代通过率；
- 对 AI‑Infra‑Guard，最高 41% 的通过率；
- 在扩张阶段，产生 438 个同时通过审计且在沙盒中导致致命行为的变体；
- 迁移实验表明，AIG 上进化出的 87.7% 变体同样能通过 SkillVetter。

**⚠️ 局限性**

局限性：
1) 种子技能与攻击者搜索通过同一图引导，可能导致样本偏差；
2) 只评估了两个公开审计器，未覆盖更广泛的审计器生态；
3) 仅考虑灰盒审计与沙盒反馈，未研究黑盒或全白盒场景；
4) 实验基于开放源代码工具与模型，商业审计器或更强模型的适用性尚未验证；
5) 对抗实验在 5 轮预算内完成，未探索更长时间/更高预算的极限。

---

## 522. Sobolev Regularized MMD Gradient Flow

**arXiv ID:** 2605.11884 | [PDF](https://arxiv.org/pdf/2605.11884v1)

**作者:** Chenyang Tian `[一作]` (Tsinghua University), Zonghao Chen `[通讯]` (University College London)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5085343114)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Sobolev正则化的最大均值散度（SrMMD）梯度流，给出连续与离散时间下的全局收敛理论，并实现粒子化的可执行算法，适用于生成建模与采样两大任务。

**💡 创新点**

创新点在于：①对MMD/KSD witness函数加入梯度范数惩罚，消除对isoperimetric条件的依赖；②提供闭式可实现的粒子实现；③在理论上证明指数收敛并给出源条件下的残差阶数；④兼顾生成与采样，弥补以往方法只能单一任务的不足。

**🔧 技术方法**

技术手段包括：RKHS与Sobolev正则化、Stein核构造、Tikhonov正则化、梯度惩罚、Euler离散、粒子系统实现、自动微分与矩阵求逆等。

**📊 数据集**

实验数据集涵盖：高斯混合与Swiss roll（仿真）；学生-教师网络；图像色彩迁移；二维10高斯混合；贝叶斯逻辑回归（Breast Cancer、Ionosphere、German Credit、Covtype）等公开数据集。

**📈 对比分析**

与原始MMD流、DrMMD、HrMMD、KSD流、SVGD、R‑SVGD等方法进行比较；SrMMD在MMD、KSD、W₂等指标上收敛速度更快、最终误差更低，生成任务中表现优于原MMD、KSD，采样任务中与SVGD相当且优于KSD。

**⚠️ 局限性**

主要限制：每一步需求解N³d³维矩阵逆，计算成本高；实验规模受限，未在大规模任务上验证；正则化条件（m_μ-π∈S_μ^α/2）难以直接验证；仅在小规模采样任务上与SVGD进行对比。

---

## 523. On-Policy Self-Evolution via Failure Trajectories for Agentic Safety Alignment

**arXiv ID:** 2605.11882 | [PDF](https://arxiv.org/pdf/2605.11882v1)

**作者:** Bo Yin `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**通讯引用:** 13745 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自演进框架FATE，用失败轨迹生成并筛选修复轨迹，训练工具使用型LLM代理更安全；

**💡 创新点**

创新点在于把自身失败轨迹转化为多目标（安全、效用、拒绝控制、轨迹有效性）Pareto前沿的修复监督，避免单目标奖励导致退化；

**🔧 技术方法**

采用同策略修复生成、验证器重新评分、Pareto前沿筛选、SFT+PFPO（Pareto-Front Policy Optimization）强化学习；

**📊 数据集**

在AgentDojo、AgentHarm、ATBench等基准上进行实验，使用多种开源模型（Qwen3、Llama3.1、Ministral、Gemma、Phi-4）及其不同规模；

**📈 对比分析**

与ReAct、Reflexion、工具过滤、PI检测等现有防御/微调方法对比，FATE在攻击成功率、危害合规率、任务成功率等指标上显著提升（例如ASR下降33.5%，HCR下降82.6%，安全评分提升6.5%）；

**⚠️ 局限性**

局限包括依赖验证器评分质量、对非可执行任务或极端攻击的适应性待验证、训练成本和多目标平衡调参的复杂性。

---

## 524. RecRM-Bench: Benchmarking Multidimensional Reward Modeling for Agentic Recommender Systems

**arXiv ID:** 2605.11874 | [PDF](https://arxiv.org/pdf/2605.11874v1)

**作者:** Wenwen Zeng `[一作]` (Meituan), Guojun Yin `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RecRM-Bench 及 RecRM-RL 两个框架，分别用于构建多维度奖励模型和优化智能推荐系统；

**💡 创新点**

创新点在于：①首次系统化构建包含指令遵循、事实一致性、查询-项目相关性和用户行为预测四大维度的 1.07M 条样本基准；②提出层次化奖励策略与混合多维奖励融合的 RL 训练框架；③通过数据增强、生成式判别与人机对齐方法提升奖励模型质量；

**🔧 技术方法**

技术上主要采用大型语言模型（Qwen3-0.6B/8B/14B、Qwen3-Reranker）、ReAct 结构、SFT 与序列交叉熵、双目标学习、分层门控奖励、交叉注意力重排序器等；

**📊 数据集**

使用的主要数据集为：RecRM-Bench（Instruction Following、Factual Consistency、Query‑Item Relevance、User Behavior 四子库共 1,073,779 条样本），并以 Meituan 实际交互日志为来源；

**📈 对比分析**

在 RecRM-Bench 上与 GPT‑4.1、LongCat、DeepSeek、Qwen3 等多种基线进行比较，零射模型表现低于 SFT 版本；我们的奖励模型和 RecRM‑RL 通过层次奖励显著提升行为预测准确率（+19%）和查询‑项目相关性（+7.8%），最终整体性能达到最高 89% 以上；

**⚠️ 局限性**

局限性包括：①对复杂多条件意图和类别判别仍易出现误判；②事实一致性检验过度敏感，导致误报；③目前仍依赖大规模预训练 LLM，算力成本高；④数据来源单一（Meituan），可能影响跨域推广。

---

## 525. Maximizing Reachability via Shifting of Temporal Paths

**arXiv ID:** 2605.11873 | [PDF](https://arxiv.org/pdf/2605.11873v1)

**作者:** Argyrios Deligkas `[一作]` (Royal Holloway University of London), Georg Tennigkeit `[通讯]` (Hasso Plattner Institute University of Potsdam)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在由 k 条时间路径组成的时序图上，通过对边标签的移动（前移/后移）来最大化指定源点的可达节点数。

**💡 创新点**

首次将传播约束纳入时序图的标签移动，并给出了多种参数化复杂度结果，证明了在预算无穷、预算有限及其组合下的 FPT 与 W[1]/W[2]-hard 边界。

**🔧 技术方法**

利用切换路径树、整数线性规划、滑动窗口推导和贪心/枚举策略，对不同操作类型构建了 FPT 与 XP 算法。

**📊 数据集**

未使用真实数据集，仅在理论模型上进行分析；若有实验则基于合成实例。

**📈 对比分析**

实验与比较部分缺失；所有结论基于理论复杂度证明，未给出运行时实验数据。

**⚠️ 局限性**

主要局限在于对单源、单路径数 k 的 FPT 结果尚未完全确定，且在多源或更一般时序图结构下的可达性优化仍未解决。

---

## 526. Very Efficient Listwise Multimodal Reranking for Long Documents

**arXiv ID:** 2605.11864 | [PDF](https://arxiv.org/pdf/2605.11864v1)

**作者:** Yiqun Sun `[一作]` (Magellan Technology Research Institute), Lawrence B. Hsieh `[通讯]` (Magellan Technology Research Institute)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种高效的列表式多模态重排序模型ZipRerank，用于长文档检索场景。

**💡 创新点**

创新点包括：① 两阶段训练策略（先在文本重排序数据上预训练，再用VLM教师蒸馏进行多模态微调）；② 软排名损失以降低教师监督噪声；③ 查询-图像早期交互的视觉令牌剪枝；④ 单步标记解码实现一次前向传递即可完成排序，显著降低推理延迟。

**🔧 技术方法**

技术手段包括：大规模文本重排序数据的图像化渲染、RankNet 与 RBP 指数衰减软排名损失、查询-图像相似度基于最大余弦相似度的令牌筛选、RoPE 位置嵌入、单日志解码（single-logit scoring）等。

**📊 数据集**

使用的数据集主要有：RankZephyr（文本重排序数据集）和MMDocIR（长文档检索基准），在MMDocIR的10个多领域长文档上进行评估，并在ViDoRe英文子集上做泛化验证。

**📈 对比分析**

与MM-R5、LamRA、UniME等现有多模态列表式重排序器以及基准的零样本VLM进行对比。ZipRerank在Recall@3/5上与MM-R5相当甚至优于其，在Recall@1略逊；同时缓存LLM推理时间低于0.4秒，比MM-R5低约10×，显著提高了效率。

**⚠️ 局限性**

局限性包括：依赖教师模型生成的排名，可能带来偏差；查询-图像剪枝在过度压缩时可能丢失关键信息（如小文本、密集表格等）；实验主要聚焦文档图像检索，对其他语言、领域和检索场景的泛化还需进一步验证。

---

## 527. EvoNav: Evolutionary Reward Function Design for Robot Navigation with Large Language Models

**arXiv ID:** 2605.11859 | [PDF](https://arxiv.org/pdf/2605.11859v1)

**作者:** Zhikai Zhao `[一作]` (KAIST), Jinkyoo Park `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）的进化框架，用于自动化设计机器人导航奖励函数，并通过三阶段逐步逼近的评估流程来降低计算成本。

**💡 创新点**

创新点在于将LLM与进化搜索结合，并引入低成本分析规则、代理模型和完整训练三阶段递进式评估，解决了奖励函数评估计算昂贵的瓶颈。

**🔧 技术方法**

采用LLM生成可执行的奖励代码，进化算法（变异、交叉、随机重启），基于规则的Spearman相关性评分，轻量级A2C代理训练以及最终PPO全量训练，并使用多目标性能指标。

**📊 数据集**

使用预收集的导航轨迹数据集（100个场景，每场景10条轨迹）以及在二维连续模拟器中的密集人群环境数据。

**📈 对比分析**

通过与经典导航方法（SF、ORCA、CrowdNav++）及其他自动奖励设计方法（Eureka、CARD）在同一网络架构下对比，实验显示该方法实现最高成功率、最低碰撞率和超时率，显著优于基线。

**⚠️ 局限性**

局限性包括对代理模型相关性假设的依赖、需要手工制定分析规则和代理参数，以及方法在不同环境或更大规模任务中的泛化能力待进一步验证。

---

## 528. Beyond Parameter Aggregation: Semantic Consensus for Federated Fine-Tuning of LLMs

**arXiv ID:** 2605.11857 | [PDF](https://arxiv.org/pdf/2605.11857v1)

**作者:** Amr Abourayya `[一作]` (Technical University Dortmund), Michael Kamp `[通讯]` (Technical University Dortmund)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于模型行为的联邦微调框架——Semantic Consensus，改为通过共享公共提示集的模型输出进行协同，而非参数聚合；

**💡 创新点**

创新点在于将协作媒介从参数切换到行为，消除了模型尺寸、架构、tokenizer及LoRA配置等异构问题，并实现了通信量与模型规模无关；

**🔧 技术方法**

采用公共提示集生成文本响应、语义嵌入编码、基于余弦距离的聚类与中心点选取来产生伪标签，随后在本地继续微调；

**📊 数据集**

使用多种LLM（TinyLlama、LLaMA、LLaMA2）在指令遵循（Dolly‑15k、Alpaca）、知识泛化（Wizard、OpenOrca）和聊天质量（ShareGPT、UltraChat）等公开基准上进行实验；

**📈 对比分析**

与多种参数聚合基线（FedIT、FLoRA、FlexLoRA）对比，FedCoFiT在指令、知识、聊天三个任务上均能匹配或超过基线，同时通信量降低数百倍（如LLaMA‑7B指令微调仅0.006 GB vs 3–20 GB），且能在异构LoRA等级下保持稳健；

**⚠️ 局限性**

局限性包括对公共提示集的依赖，提示数目不足或分布偏移会影响共识质量；聚类算法的参数选择与语义编码器对性能也有显著影响。

---

## 529. Domain Restriction via Multi SAE Layer Transitions

**arXiv ID:** 2605.11920 | [PDF](https://arxiv.org/pdf/2605.11920v1)

**作者:** Elias Shaheen `[一作]` (Technion -- Israel Institute of Technology), Avi Mendelson `[通讯]` (Technion -- Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种仅使用ID数据的轻量级域限制门控方法，利用Transformer内部激活的稀疏自编码器(SAE)特征的层间转移来检测超域文本。

**💡 创新点**

创新点在于将SAE稀疏特征映射为可解释的二进制分布式表示，并用层级转移统计（Markov/HTM/RNN）实现无监督的OD检测，同时可解释内部动态。

**🔧 技术方法**

采用稀疏自编码器(SAE)、层级马尔可夫/HTM/RNN序列模型、Top-k二值化、稀疏特征掩蔽等技术。

**📊 数据集**

实验基于Gemma2-2B/9B模型，在20新sgroup为ID，SST-2/MNLI/RTE/IMDB/CLINC150等数据集做far‑OOD；在AGNews、ROSTD、SNIPS、CLINC150做near‑OOD。

**📈 对比分析**

与基线LR（基模型与ID微调模型的似然比）比较，far‑OOD下AUROC≈0.99、FPR@95≈0.005；near‑OOD中表现略逊于LR，但在多种数据集仍保持竞争力。

**⚠️ 局限性**

局限性是当任务细粒度超过SAE特征分辨率时（如CLINC150），不同意图在SAE空间产生相似轨迹，导致OD检测性能显著下降。

---

## 530. Learning Subspace-Preserving Sparse Attention Graphs from Heterogeneous Multiview Data

**arXiv ID:** 2605.11881 | [PDF](https://arxiv.org/pdf/2605.11881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 531. STAGE: Tackling Semantic Drift in Multimodal Federated Graph Learning

**arXiv ID:** 2605.11919 | [PDF](https://arxiv.org/pdf/2605.11919v1)

**作者:** Zekai Chen `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7471 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 STAGE，一个基于协议优先的多模态联邦图学习框架，先对多模态特征进行语义校准，再控制图传播。

**💡 创新点**

创新点在于：1）将多模态特征映射到共享的冻结锚空间，解决特征漂移；2）利用对比信息NCE校准全局锚原型，消除伪对齐；3）采用最大熵正则化和可微同质性控制，抑制传播诱导漂移；4）使用低维协议信息实现通信压缩。

**🔧 技术方法**

主要技术包括：变分语义翻译（KL 正则化投影到锚分布）、最大熵投影、对比学习（InfoNCE）锚原型校准、Meta‑Controller 对消息传递温度的自适应调节、低维协议通信、指数移动平均更新全局锚原型。

**📊 数据集**

在 8 个多模态图数据集上评估：Toys、Grocery、Bili Music、DY、QB、Bili Cartoon、Flickr30k、SemArt。

**📈 对比分析**

与 FedAvg、Fed-MGNet、Fed-MHGAT、FedMVP、FedMAC、FedLap、S2FGL、FedSPA、FedIIH、FedProto 等多类基线对比，STAGE 在节点分类、链路预测、模态检索、G2Text、G2Image 等任务均取得 1–4% 以上的平均提升，同时通信量比 FedAvg 减少 122 倍，收敛速度更快。

**⚠️ 局限性**

局限性包括：1）需要服务器维护全局锚原型，扩展到极大规模联邦场景时可能受限；2）对锚数、温度等超参仍需经验调优；3）实验主要集中在已标注的多模态图，尚未验证对高度噪声或极端模态缺失的鲁棒性；4）对非常稀疏或异构度过高的图结构适用性仍待进一步研究。

---

## 532. Adaptive TD-Lambda for Cooperative Multi-agent Reinforcement Learning

**arXiv ID:** 2605.11880 | [PDF](https://arxiv.org/pdf/2605.11880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 533. Understanding Sample Efficiency in Predictive Coding

**arXiv ID:** 2605.11911 | [PDF](https://arxiv.org/pdf/2605.11911v1)

**作者:** Gaspard Oliviers `[一作]` (University of Oxford), Rafal Bogacz `[通讯]` (University of Oxford)

**通讯引用:** 12469 | [OpenAlex ID](https://openalex.org/A5049095056)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过对深度线性网络（DLN）中预测编码（PC）与反向传播（BP）的学习动力学进行解析，提出并量化了“目标对齐（target alignment）”指标，用以衡量学习过程中的干扰；随后推导出PC与BP在DLN中的闭式预测变化表达式，并证明PC通过S⁻¹缩放有效降低干扰，从而提升样本效率；为实现理论最优目标对齐，作者进一步提出层级学习率和权重更新缩放方法，并在单样本与批量学习场景中验证其有效性。

**💡 创新点**

创新点包括：① 引入目标对齐指标量化学习干扰；② 推导PC与BP在DLN中的闭式表达式，揭示干扰来源；③ 证明PC利用S⁻¹缩放抑制干扰；④ 设计层级学习率和权重缩放实现无干扰学习；⑤ 在多样本情境下提出跨样本干扰消除策略。

**🔧 技术方法**

技术手段主要为理论推导与闭式解析、梯度下降动力学分析、自然梯度比较、数值实验、学习率扫描、权重缩放矩阵实现，以及在非线性自动编码器上的验证。

**📊 数据集**

实验使用随机生成的线性回归数据、合成回归任务（W_data ∼ N(0,1/d_in)）以及MNIST数据集的非线性自动编码器。

**📈 对比分析**

实验通过比较BP与PC在单步更新后的目标对齐、整个训练轨迹的样本效率、以及在不同初始化、网络宽度/深度下的表现；结果表明PC在深、窄、预训练网络中目标对齐更高，样本效率显著优于BP；在批量训练中，缩放PC实现最优目标对齐并显著加速收敛，性能优于BP和缩放BP。

**⚠️ 局限性**

局限性包括：理论仅针对线性网络，非线性情况仅作初步验证；研究范围限定在多层感知机，未覆盖卷积或残差网络；权重/学习率缩放涉及不太生物学合理的计算；对更大规模或更复杂网络的验证不足。

---

## 534. Incentivizing Truthfulness and Collaborative Fairness in Bayesian Learning

**arXiv ID:** 2605.11889 | [PDF](https://arxiv.org/pdf/2605.11889v1)

**作者:** Rachael Hwee Ling Sim `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**通讯引用:** 809 | [OpenAlex ID](https://openalex.org/A5030304400)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于贝叶斯模型的协同机器学习奖励机制，既保证协同公平，又在均衡点激励数据源提交真实数据。

**💡 创新点**

创新点在于：①首次将半价值（如Shapley值）与基于验证集的数值评估函数（DVF）结合；②证明该DVF在满足一定假设下对数据源具有严格的真值激励；③给出了在预算有限或缺失验证集时的可行松弛方案。

**🔧 技术方法**

使用技术包括：贝叶斯模型（高斯过程、贝叶斯逻辑回归、贝叶斯神经网络、贝叶斯多项式逻辑回归）；对验证集的对数似然作为DVF；半价值（Shapley值）作为公平奖励；基于期望值的真值激励理论；无偏半价值估计器；对假设的概率推理。

**📊 数据集**

实验数据集：Friedman 生成数据、Heart Disease（心脏病）数据、Cycle Power Plant（循环电厂）数据、Blood MNIST（血液细胞图像）数据，均划分为3个数据源并与验证集共同训练。

**📈 对比分析**

方法比较：将提出的DVF+半价值机制与仅使用信息增益、体积等验证集自由DVF的方案对比。实验显示：①在所有数据集上，真实提交策略（T）获得最高价值和奖励；②半价值保持协同公平，其他源在对手提交不真实数据时奖励上升；③在有限预算或无验证集的场景下，通过加权半价值或多验证集方案仍能保持真值激励并保持公平，整体性能优于对比方法。

**⚠️ 局限性**

局限性：①需假设贝叶斯模型与验证集的分布已知；②对验证集的依赖限制了在隐私受限场景的直接应用；③半价值计算在源数较大时计算成本高；④预算限制下奖励的比例化可能导致非线性难以保证公平；⑤实验仅覆盖了少量源和特定模型，需进一步验证在更大规模、多源环境下的可扩展性。

---

## 535. Information theoretic underpinning of self-supervised learning by clustering

**arXiv ID:** 2605.11870 | [PDF](https://arxiv.org/pdf/2605.11870v1)

**作者:** Josef Kittler `[一作]` (University of Surrey), Muhammad Awais `[通讯]` (University of Surrey)

**通讯引用:** 2370 | [OpenAlex ID](https://openalex.org/A5100778579)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种以 K‑L 散度为目标的自监督学习框架，使用教师‑学生蒸馏机制并加入熵正则化以防止模式坍塌，得到基于逆集群先验的教师分布归一化，并用 Jensen 不等式证明其可简化为常用的 centering 归一化，从而为聚类型自监督方法提供理论基础。

**💡 创新点**

创新点在于：① 将 SSL 形式化为蒸馏的 K‑L 散度最小化；② 通过熵正则化得到闭式教师归一化公式；③ 证明该归一化可近似为 centering，解释了众多实用方法的有效性；④ 对教师与学生参数的迭代更新机制给出了可操作的实现细节。

**🔧 技术方法**

使用深度网络生成嵌入向量，假设嵌入空间可用高斯混合模型或软 K‑means 表示；采用 K‑L 散度、熵正则、Jensen 不等式、批归一化（batch centering）以及指数滑动平均等技术；在理论上还提出可选的基于全局数据的集群先验更新策略。

**📊 数据集**

文中未给出具体实验数据集，理论讨论主要基于通用的自监督学习框架；若以常见做法推断，可适用于 ImageNet、COCO 等大型视觉数据集。

**📈 对比分析**

与 DINO、SwAV、DeepCluster 等现有聚类型 SSL 方法进行概念对比；理论上说明通过逆集群先验归一化可避免模式坍塌，从而间接提升聚类质量；实验部分未给出数值，但作者声称理论分析与这些方法在实践中表现一致。

**⚠️ 局限性**

局限性包括：① 近似（尤其是 Jensen 下的 lower bound）在大 logits 范围可能失效；② 需要在批内或全局估计集群先验，易受样本噪声影响；③ 对权重向量中心化或单位化的假设在不同网络结构下可能不成立；④ 仍未系统评估不同温度、K 值、维度等对性能的影响；⑤ 只聚焦于聚类型方法，对对比学习等其他自监督范式的推广有限。

---

## 536. Concordance Comparison as a Means of Assembling Local Grammars

**arXiv ID:** 2605.11862 | [PDF](https://arxiv.org/pdf/2605.11862v1)

**作者:** Juliana Pirovani `[一作]` (Universidade Federal do Espirito Santo), Eric Laporte `[通讯]` (Universite Paris-Est)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用Unitex Concordance Comparison工具辅助手工构造葡萄牙语人名识别的局部语法（LG），并通过比较不同LG的共识与差异来合并、优化语法。

**💡 创新点**

创新点在于把语法比较结果转化为集合论关系（包含、交集、并集），以此系统化地选择和融合LG，从而显著提升识别准确性。

**🔧 技术方法**

采用的技术包括Unitex文本预处理、词典匹配、LGG（局部语法图）设计、ConcorDiff对比工具以及SAHARA在线评测平台。

**📊 数据集**

实验数据集为第二届HAREM的Golden Collection（129篇文本，包含1,609条PERSON标签）。

**📈 对比分析**

与Rembrandt等现有系统对比，最终LG在PERSON（INDIVIDUAL）子类别上实现了约10个百分点的召回率提升，F-Measure提升至约70.8（比Rembrandt高约6点）。

**⚠️ 局限性**

局限性包括：LG仅覆盖个体与职位两种子类型，导致整体召回率低于其他工具；构造LG仍需大量人工规则，自动化程度不足。

---

## 537. UniVLR: Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs

**arXiv ID:** 2605.11856 | [PDF](https://arxiv.org/pdf/2605.11856v1)

**作者:** Houcheng Jiang `[一作]` (University of Science and Technology of China), Yong Li `[通讯]` (Tsinghua University)

**通讯引用:** 38490 | [OpenAlex ID](https://openalex.org/A5100355277)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UniVLR 框架，将文本推理与辅助图像统一到视觉工作区，并在推理时仅通过视觉潜在令牌完成思考，最终仅解码答案。

**💡 创新点**

创新点在于：①把文字推理渲染为图像并与辅助图像拼合，形成统一的视觉画布，消除显式文本推理通道；②两阶段对齐训练（视觉潜在预训练 + 文本-视觉统一对齐）实现连续视觉潜在生成；③推理时只使用视觉潜在，显著减少推理令牌长度。

**🔧 技术方法**

技术手段包括：视觉工作区渲染（文字+图像拼合）、视觉编码器提取视觉特征、2D 视角池化生成潜在监督、两阶段对齐训练（视觉潜在预训练与文本-视觉统一对齐）、连续自回归视觉潜在推理头、控制标记切换、轻量化 MLP 生成层。

**📊 数据集**

使用的主要数据集为 V*、HRBench4K/HRBench8K、MME-RealWorld-Lite；训练时采集过滤后的 Zebra-CoT 与 VisCoT 等视觉推理示例。

**📈 对比分析**

与文本推理、工具驱动视觉推理以及传统视觉潜在推理方法（LVR、Monet、SkiLa、CoVT）对比，UniVLR 在四个基准上均取得最高整体分（平均 68.9 分），同时将推理令牌数从 190–270 条压缩到 12 条，仅生成最终答案，推理令牌数量减少约 15.2×。

**⚠️ 局限性**

局限性：①依赖视觉编码器强大的 OCR 与布局理解，效果受限于视觉先验；②潜在令牌不可直接可视化，缺乏自然语言推理的可解释性；③采用固定潜在令牌预算，可能不适用于不同任务；④不适合需要精确测量或执行外部视觉工具的场景。

---

## 538. Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models

**arXiv ID:** 2605.11854 | [PDF](https://arxiv.org/pdf/2605.11854v1)

**作者:** Kecheng Chen `[一作]` (City University of Hong Kong), Haoliang Li `[通讯]` (City University of Hong Kong)

**通讯引用:** 6452 | [OpenAlex ID](https://openalex.org/A5040091210)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TABOM 框架，对扩散语言模型进行自蒸馏轨迹对齐的后训练方法，显著提升推理质量并避免灾难性遗忘。

**💡 创新点**

创新点在于把推理的易到难无偏序建模为 Boltzmann 分布，并通过 pairwise ranking 损失实现训练-推理的对齐，从而充分利用自蒸馏轨迹的结构信息。

**🔧 技术方法**

采用自蒸馏轨迹采样、Boltzmann 分布建模、pairwise ranking 损失、LoRA 参数高效微调以及交叉熵重构等技术。

**📊 数据集**

使用 Dream‑7B‑Instruct 与 LLaDA‑8B‑Instruct 作为基模型，在代码生成（Ling‑Coder‑SFT、MBPP）和数学推理（MixChain‑Z‑PRM12K、GSM8K、MATH500）任务上生成自蒸馏轨迹。

**📈 对比分析**

与 SFT‑GT、SFT‑SD、dInfer、T3D 等基线对比，TABOM 在域内平均提升约 +5% 并保持甚至提升域外表现，彻底消除标准 SFT 的灾难性遗忘，整体性能最高。

**⚠️ 局限性**

局限性包括：对极大步数或高度并行解码仍有轻微性能下降；窗口大小 W 的设定对 ranking 效果有显著影响；在不同任务或更大模型规模下的泛化能力仍待进一步验证。

---

## 539. LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters

**arXiv ID:** 2605.12012 | [PDF](https://arxiv.org/pdf/2605.12012v1)

**作者:** Virgill van der Meer `[一作]` (Municipality of Amsterdam), Julien Rossi `[通讯]` (University of Amsterdam)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5001721772)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并部署LegalCheck系统，利用检索增强生成和多阶段上下文增强生成自动草拟市政法律意见书，显著降低人工起草时间与工作负荷；

**💡 创新点**

首次将检索增强生成（RAG）与多阶段上下文增强生成（CAG）相结合，形成交互式人机协同工作流程，使模型在获取相关案例后还能根据专家反馈迭代改进；

**🔧 技术方法**

采用GPT‑4o大语言模型，结合OpenAI文本嵌入模型（text‑embedding‑ada‑002）实现语义检索，利用定制Prompt进行生成与细化，后台采用Flask等技术构建Web接口；

**📊 数据集**

使用市政行政案件库约23,500份历史法律意见书（含废物罚金、车辆拖吊等四个执法领域），对文件进行章节切分、嵌入索引，并在评估中对比人工撰写的同类信件；

**📈 对比分析**

通过量化指标（内容覆盖率、准确性、F1分数）、编辑保留率（81%）以及时间节省比例（50‑70%）与人工起草对比，实验显示AI草稿在保持法律一致性、可读性和引用完整性的同时，显著提升效率；

**⚠️ 局限性**

局限在单一市政机构与四个特定执法领域、对第三方LLM的依赖、未系统评估长期职业影响及公平性风险，以及在其他法律文书类型或司法系统中的可推广性不足。

---

## 540. A geometry-aligned multi-fidelity framework for uncertainty quantification of wildfire spread

**arXiv ID:** 2605.12007 | [PDF](https://arxiv.org/pdf/2605.12007v1)

**作者:** Konstantinos Vogiatzoglou `[一作]` (University of Thessaly), Han Gao `[通讯]` (Iowa State University)

**通讯引用:** 2831 | [OpenAlex ID](https://openalex.org/A5062881243)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个几何对齐的多精度框架，用于火灾蔓延的不确定性量化。

**💡 创新点**

引入了前向映射将低精度与高精度快照对齐至共同参考域，从而解决了传统多精度方法在推进前沿时产生的振荡和高基数需求。

**🔧 技术方法**

采用了ADfiRe物理驱动的反应扩散方程、基于贪婪选择的低精度训练、正交映射与正则化最小二乘重构，以及一维/二维几何对齐映射技术。

**📊 数据集**

使用了基于ADfiRe模拟的合成数据集，涵盖一维和二维测试场景，并通过LHS生成参数采样。

**📈 对比分析**

与传统低精度、无映射双精度和高精度模型相比，几何对齐双精度在最大温度、蒸发量和烧毁面积的概率分布上误差显著降低，且在线预测速度约比高精度快三阶，成本换算可在多次查询后优于直接高精度蒙特卡罗。

**⚠️ 局限性**

局限于近凸、单一前沿的情况，对非凸或复杂拓扑前沿映射效果有限，且依赖于低高精度映射参数的多项式回归，未验证与真实野火实验或CFD基准的一致性。

---

## 541. CR^2: Cost-Aware Risk-Controlled Routing for Wireless Device-Edge LLM Inference

**arXiv ID:** 2605.12001 | [PDF](https://arxiv.org/pdf/2605.12001v1)

**作者:** Nan Xue `[一作]` (Shanghai Jiao Tong University), Meixia Tao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 11926 | [OpenAlex ID](https://openalex.org/A5016127527)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种两阶段移动边缘LLM路由框架CR^2，先在设备上用可观测特征预测本地执行是否具备优势，再将查询送往边缘服务器进行状态感知的最佳模型选择。

**💡 创新点**

创新点包括：1) 将路由拆分为本地门控和边缘选择，符合设备–边缘部署结构；2) 使用教师选择器提供全信息的利用率边际作为监督，并引入温度标度的分数；3) 采用基于置信度的风险控制（CRC）校准门槛，显式控制误判率；4) 通过周期化FiLM特征实现对不同成本权重的统一门控模型。

**🔧 技术方法**

主要技术包括：冻结查询编码器+轻量化MLP门控、教师模型的多头二元分类+对比排序损失、Huber回归门槛值、Monotonicity正则、CRC风险校准、状态感知的边缘选择器和基于统一成本模型的部署成本计算。

**📊 数据集**

使用公开LLM基准数据集MMLU、BBH、GPQA、MBPP进行模型准确率标注，并用手工采样的无线与计算资源参数构建部署成本模型。

**📈 对比分析**

对比了静态选择、KNN、MLP、EmbedLLM、LLMRank等一机路由方法。CR^2在准确率-成本Pareto前沿上显著优于所有基线，匹配同等准确率时可降低16.9%的归一化部署成本。

**⚠️ 局限性**

局限性：1) 仅针对已知模型池和预估的无线环境，动态变化需重新校准；2) CRC校准需额外校准集，且仅控制单一成本权重下的误差；3) 在低成本点门槛受限，导致误拒率略高；4) 部署成本模型假设与实际硬件差异可能影响泛化。

---

## 542. Random-Set Graph Neural Networks

**arXiv ID:** 2605.11987 | [PDF](https://arxiv.org/pdf/2605.11987v1)

**作者:** Tommy Woodley `[一作]` (Oxford Brookes University), Fabio Cuzzolin `[通讯]` (Oxford Brookes University)

**通讯引用:** 2681 | [OpenAlex ID](https://openalex.org/A5050777136)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于随机集的节点级不确定性量化框架 RS-GNN，替代传统 softmax 输出为 belief 函数，能够同时给出精确概率预测和 epistemic 不确定性估计。

**💡 创新点**

创新点在于将随机集贝叶斯不确定性理论引入图神经网络，使用预算焦点集实现可扩展的 belief 输出，并通过 Pignistic 变换得到可解释的概率预测。

**🔧 技术方法**

使用技术包括随机集贝叶斯理论、Belief 函数、Pignistic 变换、GNN 消息传递、预算焦点集选择、二元交叉熵损失加质量正则化，以及对模型进行校准与 OOD 检测。

**📊 数据集**

实验数据集共 9 个，涵盖小型图数据集（Cora、Pubmed、Coauthor、Squirrel、Chameleon）与大型自动驾驶基准（ROAD、nuScenes）以及 Amazon 等。

**📈 对比分析**

与软max、贝叶斯 GNN、集成、能量方法等基线进行比较；在 ID 分类保持相近甚至更优的准确率，校准更好；在 OOD 检测 AUROC、AUPRC 等指标显著优于传统方法。

**⚠️ 局限性**

局限性包括：对大类别数需使用预算焦点集，导致模型对超参数和配置敏感；跨域迁移仍面临挑战；计算复杂度随焦点集数量指数增长，需进一步优化。

---

## 543. Optimizing 4D Wires for Sparse 3D Abstraction

**arXiv ID:** 2605.11977 | [PDF](https://arxiv.org/pdf/2605.11977v1)

**作者:** Dong-Yi Wu `[一作]` (National Cheng Kung University), Tong-Yee Lee `[通讯]` (National Cheng Kung University)

**通讯引用:** 5200 | [OpenAlex ID](https://openalex.org/A5050657606)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种单一连续的 4D B‑spline 线条（x、y、z、宽度 w）来统一抽象 3D 形状，从单个曲线实现结构连贯的几何抽象。

**💡 创新点**

创新点包括：① 用单条连续 B‑spline 替代“散点曲线”集合，实现全局拓扑连通；② 引入可学习宽度 w 以同时编码体积与细节；③ 通过宽度引导的重初始化和节点插入实现动态拓扑管理；④ 设计基于 DiffVG 的可微渲染管线，将 3D 曲线投影近似为 2D 多项式曲线，避免了 rational 曲线的昂贵计算。

**🔧 技术方法**

技术手段包括：DiffVG 可微向量图渲染、Score Distillation Sampling (SDS) 与 CLIP 引导、三维 B‑spline 基础变换、宽度重初始化策略、基于 jerk 能量的几何正则化。

**📊 数据集**

实验使用 Zero123 预训练生成器作为视觉先验，QuickDraw 类别数据集用于多视角线条艺术评估；图像输入来自公开 2D 图片集，后续不需要手工标注。

**📈 对比分析**

与 Diff3DS、DreamWire 等基线相比，本文方法在 CLIP 语义一致性上相当甚至略优，同时在组件数、曲线长度、连通度等结构指标上显著优于基线；实验结果表明单条曲线能在保持语义的同时减少冗余、提升连通性。

**⚠️ 局限性**

局限性包括：对完全均匀圆形或具有严格常宽要求的物体时，宽度参数可能引入不必要的波动；多视角提示冲突时易产生较简单的全局解；宽度可变性在某些高频细节重建上仍存在稳定性挑战。

---

## 544. Towards Order Fairness: Mitigating LLMs Order Sensitivity through Dual Group Advantage Optimization

**arXiv ID:** 2605.11974 | [PDF](https://arxiv.org/pdf/2605.11974v1)

**作者:** Xu Chu `[一作]` (Peking University), Weiping Li `[通讯]` (Peking University)

**通讯引用:** 28917 | [OpenAlex ID](https://openalex.org/A5100415565)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为Dual Group Advantage Optimization（DGAO）的强化学习方法，用于同时提升大型语言模型在不同顺序输入下的准确性和一致性。

**💡 创新点**

创新点在于首次结合群内相对优势和群间相对优势进行奖励设计，以解决传统数据增强导致的准确率下降和过度一致性的问题。

**🔧 技术方法**

采用了基于策略梯度的自策略强化学习（类似PPO/GRPO）框架，并通过自监督奖励函数评估答案正确性和顺序稳定性。

**📊 数据集**

实验使用了SST‑2、SQuAD v2、SearchQA、GSM8K和CM17K等多任务数据集进行微调和评估。

**📈 对比分析**

与传统SFT、PAFT、Prompt以及其他顺序偏差缓解方法相比，DGAO在准确率、Consistency Rate上均有提升，且Overconfidence Rate显著下降，显示出更优的综合性能。

**⚠️ 局限性**

局限性包括仅在3B–14B规模模型上验证，且采用固定数量的随机顺序变体，可能不足以覆盖大规模上下文中的所有排列，导致在极大序列或不同任务时效果下降。

---

## 545. Estimating Subgraph Importance with Structural Prior Domain Knowledge

**arXiv ID:** 2605.12009 | [PDF](https://arxiv.org/pdf/2605.12009v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 546. Assessing and Mitigating Miscalibration in LLM-Based Social Science Measurement

**arXiv ID:** 2605.11954 | [PDF](https://arxiv.org/pdf/2605.11954v1)

**作者:** Jinyuan Wang `[一作]` (Hong Kong University of Science and Technology), Yi Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 82220 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估LLM在社会科学测量中的校准误差，并提出软标签蒸馏方法以提升校准；

**💡 创新点**

首次系统性审计LLM校准缺陷，揭示信心过滤会扭曲回归结果，并提出轻量级软标签蒸馏方案；

**🔧 技术方法**

采用T‑ECE_ε、Brier等校准度量，后置校准（Platt、Beta、温度缩放）以及将LLM置信度转为分布并在BERT上训练的软标签蒸馏技术；

**📊 数据集**

14个社会科学构造的公开数据集（如EmoBank、CHASM、Pavlick Formality、Stanford Politeness、FOMC会议记录等），每构造约1000条样本；

**📈 对比分析**

与原始LLM置信度及四种后置校准方法比较，软标签蒸馏平均T‑ECE_ε从0.408降至0.228（≈43%），Brier从0.376降至0.248（≈34%），后置校准虽能降低ECE，但往往导致分辨率坍塌；

**⚠️ 局限性**

仅针对英文文本，缺乏多语言与多模态评估；软标签蒸馏仍依赖LLM生成置信度，若其极度失准则可能误导模型；

---

## 547. From Submodularity to Matrix Determinants: Strengthening Han's, Szász's, and Fischer's Inequalities

**arXiv ID:** 2605.11998 | [PDF](https://arxiv.org/pdf/2605.11998v1)

**作者:** Gunank Jakhar `[一作]` (International Institute of Information Technology), Vinod M. Prabhakaran `[通讯]` (Tata Institute of Fundamental Research)

**通讯引用:** 2463 | [OpenAlex ID](https://openalex.org/A5045516299)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出并证明了对子模函数的条件化Han不等式和分区子加性的不等式强化，并将其推广到差分熵，从而得到强化的Szász和Fischer行列式不等式以及相应的特征值不等式；

**💡 创新点**

创新点在于首次利用条件化子模函数属性推导出比传统不等式更紧的界限，并给出严格的等号条件和对非对角正定矩阵的优势；

**🔧 技术方法**

使用子模函数理论、信息理论中的条件熵与链式法则、以及Gaussian随机向量的熵-行列式等价；

**📊 数据集**

未使用真实数据集，仅以手工构造的正定矩阵（如4×4示例矩阵）作为数值示例验证结果；

**📈 对比分析**

通过与经典Hadamard、Szász、Fischer以及Ky Fan不等式的对比，展示在多种参数设置下新不等式给出更小的上界，数值例子表明误差更小；

**⚠️ 局限性**

局限在于目前仅针对子模函数和Gaussian情形，且强化效果需要满足特定等价条件；对非高斯分布或一般随机过程的推广仍待研究。

---

## 548. Robust Promptable Video Object Segmentation

**arXiv ID:** 2605.12006 | [PDF](https://arxiv.org/pdf/2605.12006v1)

**作者:** Sohyun Lee `[一作]` (POSTECH), Suha Kwak `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对输入噪声、模糊、低照度等真实世界干扰，提出了鲁棒的可提示视频目标分割（RobustPVOS）研究，并构建了含有真实与合成干扰数据的完整基准；提出了基于记忆对象条件的门控低秩适配（MoGA）方法，改进了 SAM2 的鲁棒性。

**💡 创新点**

创新点包括：①首次针对可提示视频分割的鲁棒性问题系统化研究与基准构建；②提出 MoGA，将低秩适配分解为多条 rank‑1 组件，并通过记忆银行中的对象特征实现对象特异、时序一致的门控；③通过共享组件实现参数高效且在不同对象间可自适应。

**🔧 技术方法**

主要技术包括：低秩适配（LoRA）与门控机制、Gumbel‑Sigmoid 离散化、对象记忆银行、SAM2 的自注意力投影融合；训练使用标准分割损失（focal+dice）并在 MoGA 中对不同对象动态选择适配器。

**📊 数据集**

数据集：真实评估集—ACDC‑Video（149 篇、3,259 帧、613 对象）与 MVSeg（202 篇、13,581 帧、1,930 对象）；合成训练/评估集—对 MOSE、DAVIS、YouTube‑VOS 进行 8 种时变噪声合成，生成 46,768 视频（1,774,560 帧）用于训练，507 视频（13,710 帧）用于评估。

**📈 对比分析**

与 baseline（SAM2）及多种鲁棒化方案（URIE、AirNet、GaRA、LoRA、全微调）对比，MoGA+SAM2 在真实评估集上分别提升至 71.8%（MVSeg‑adv）和 64.5%（ACDC‑Video）相较于 SAM2 的 69.6%/63.5%；在合成评估集上提升至 79.9%（vs 78.7%）。参数量仅 1.1M，显著低于全微调的 80.9M，显存需求从 25GB 降至 22GB，说明 MoGA 在保持或提升性能的同时实现了高效学习。

**⚠️ 局限性**

局限性：方法依赖于对象记忆银行，长时序或极端遮挡下的记忆衰减可能影响性能；门控机制与对象信息耦合，若记忆提取错误或缺失，鲁棒性会下降；目前仅在 SAM2 框架下验证，迁移到其它可提示分割模型的通用性尚待进一步探索。

---

## 549. EDGER: EDge-Guided with HEatmap Refinement for Generalizable Image Forgery Localization

**arXiv ID:** 2605.12002 | [PDF](https://arxiv.org/pdf/2605.12002v1)

**作者:** Minh-Khoa Le-Phan `[一作]` (University of Science), Trong-Le Do `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种双分支的基于补丁的框架 EDGER，用于在任意分辨率图像中定位伪造区域。

**💡 创新点**

结合频域边缘检测的 Edge-Guided Segmentation 与基于 CLIP‑ViT 的 Synthetic Heatmapping，互补边缘和区域信息，兼顾细节和全局，提升跨域泛化。

**🔧 技术方法**

频域边缘检测器、SegFormer 解码器、CLIP‑ViT + LoRA 微调、滑动窗口热图生成、空间堆叠融合与边缘增强等技术。

**📊 数据集**

训练使用 TGIF 生成的 75k 文本引导填充样本，验证使用 SynthIM 2025 的 SAGI‑D 9,439 张多分辨率 SP 图像。

**📈 对比分析**

与 IML‑ViT、MVSSNet、CATNet 等基线在 MediaEval SynthIM 验证集上对比，融合后 IoU 达到 0.590，显著优于单支或传统方法。

**⚠️ 局限性**

仍依赖预先训练的边缘检测器与 CLIP‑ViT，难以完全端到端训练；对极小细节或完全生成图像的定位精度有限。

---

## 550. The Illusion of Power Capping in LLM Decode: A Phase-Aware Energy Characterisation Across Attention Architectures

**arXiv ID:** 2605.11999 | [PDF](https://arxiv.org/pdf/2605.11999v1)

**作者:** Bole Ma `[一作]` (Erlangen National High Performance Computing Center), Gerhard Wellein `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 6511 | [OpenAlex ID](https://openalex.org/A5070209050)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM解码阶段进行能耗研究，比较功率上限与SM时钟锁定对不同注意力架构的影响，并给出统一的能耗控制策略。

**💡 创新点**

揭示功率上限在解码阶段无效，提出SM时钟锁定是更优能耗控制手段，并系统评估多种新型注意力架构（GQA、MLA、GDN、Mamba2）的能耗特征及预填充与解码成本的交叉点。

**🔧 技术方法**

采用NVML功耗采样、Nsight Compute屋顶线分析、vLLM推理、SM时钟锁定与功率上限设置，结合批次/序列长度 sweep 进行实验。

**📊 数据集**

使用4B参数的多种模型（来自HuggingFace权重），在序列长度1K–64K、批次1–32的推理任务上进行测评。

**📈 对比分析**

通过能量/标记(mJ/tok)、吞吐量(tok/s)等指标绘制Pareto前沿，实验显示SM时钟锁定比功率上限低20–30%能耗，吞吐量损失<1%，且在所有架构与批次下均保持优势。

**⚠️ 局限性**

实验仅在单张H200 SXM、未融合的vLLM实现、4B规模模型、单GPU环境下进行，未评估多GPU/混合并行、其他驱动版本或自定义融合核，结果可能受特定硬件/软件版本影响。

---

## 551. L2P: Unlocking Latent Potential for Pixel Generation

**arXiv ID:** 2605.12013 | [PDF](https://arxiv.org/pdf/2605.12013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 552. Towards Visually-Guided Movie Subtitle Translation for Indic Languages

**arXiv ID:** 2605.11993 | [PDF](https://arxiv.org/pdf/2605.11993v1)

**作者:** Tarun Chintada `[一作]` (Indian Institute of Technology Patna), Asif Ekbal `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 9732 | [OpenAlex ID](https://openalex.org/A5085370631)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对比两种视觉摘要策略（5分钟滑动窗口属性摘要与对话间隙自由文本摘要）在五种低资源印度语言（Hindi、Bengali、Telugu、Tamil、Kannada）电影字幕翻译中的表现。

**💡 创新点**

提出时间误差是导致全视觉增强失败的主要原因，并证明粗略属性摘要在时间漂移下更稳健；同时通过oracle selective grounding（仅替换最低质量20‑30%段落）实现了大部分性能提升。

**🔧 技术方法**

使用 Qwen‑2.5‑7B‑Instruct 进行零训练翻译，Llama‑3.1‑8B‑Instruct 进行视觉摘要，FastVLM‑0.5B 生成原始视觉描述；结合 Oracle selective 逻辑进行选择性视觉增强。

**📊 数据集**

构建包含 5 部电影（Titanic、Skyfall、Oppenheimer、Spider‑Man 2、Avatar 2）的字幕与视频帧数据集，涵盖 5 种语言的平行字幕，并提供视觉描述。

**📈 对比分析**

通过对比文本基线、全视觉增强和 oracle selective 结果，发现全视觉增强往往不如文本基线，Oracle selective 在 2‑5% 的 COMET 提升，Attr‑VC 方法在多语言中表现更稳定。

**⚠️ 局限性**

研究仅覆盖 5 部电影，视觉对齐问题未完全解决，Oracle selective 仅给出上限，缺乏自动质量估计模型与更深入的人类评测。

---

## 553. LLMs and the ZPD

**arXiv ID:** 2605.12016 | [PDF](https://arxiv.org/pdf/2605.12016v1)

**作者:** Peter Wallis `[一作]` `[通讯]` (Centre for Policy Modelling), Peter Wallis (Centre for Policy Modelling)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

阐述LLM可视为“高级自动补全”机制，并将其与Vygotsky的ZPD理论相结合，提出LLM实现的是原始型（Type 1）思维而非符号型（Type 2）思维，并通过棋类游戏、Roomba迷宫等示例说明该观点；

**💡 创新点**

创新点在于把LLM重新定义为执行行为的补全案例，提出Type 1思维模型和直接感知（Direct Perception）理论，从而重构非符号交互的认知机制；

**🔧 技术方法**

主要使用LLM的统计补全技术、案例推理框架以及Roomba行为序列模拟等方法；

**📊 数据集**

未提供专门的数据集，主要引用棋谱数据与Roomba模拟环境；

**📈 对比分析**

方法上通过比较LLM在棋局中的表现与传统基于规则的AI，说明即使缺乏符号模型，LLM仍能完成高质量棋局，但未给出具体的定量指标；

**⚠️ 局限性**

限制包括缺乏客观实验数据、对Type 1与Type 2划分的主观性、未深入探讨多模态交互及其对模型的影响。

---

## 554. A Transfer Learning Evaluation of Deep Neural Networks for Image Classification

**arXiv ID:** 2605.11989 | [PDF](https://arxiv.org/pdf/2605.11989v1)

**作者:** Nermeen Abou Baker `[一作]` (Ruhr West University of Applied Sciences), Uwe Handmann `[通讯]` (Ruhr West University of Applied Sciences)

**通讯引用:** 987 | [OpenAlex ID](https://openalex.org/A5078414923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对11种预训练CNN模型在5个数据集（MNIST、CIFAR-10、Hymenoptera、智能手机及其增强版）上进行迁移学习实验，比较全层微调与仅分类层微调的效果；

**💡 创新点**

系统化地提供多指标评估（准确率、准确率密度、模型尺寸、训练时间）并给出针对不同应用场景的模型选择建议；

**🔧 技术方法**

使用PyTorch预训练模型、Adam优化器、grid搜索学习率、early stopping，对模型进行全层或仅分类层微调，并记录Accuracy、Accuracy Density、模型大小与GPU训练时间；

**📊 数据集**

采用MNIST、CIFAR-10、Hymenoptera、智能手机及其增强版等5个数据集；

**📈 对比分析**

通过对比不同模型在不同微调方式和不同episode数（一次或十次）下的准确率密度、模型尺寸与训练时间，发现SqueezeNet在准确率密度上最好，ResNet18在仅分类层微调时最优，MnasNet训练最快但精度最低；

**⚠️ 局限性**

实验仅在单卡GPU上完成，未涉及多GPU或分布式训练；仅评估了11个模型，未覆盖最新架构；数据集规模相对较小，未检验在更大样本量或更复杂任务下的泛化性能。

---

## 555. NOFE -- Neural Operator Function Embedding

**arXiv ID:** 2605.11970 | [PDF](https://arxiv.org/pdf/2605.11970v1)

**作者:** Lars Uebbing `[一作]` (UiT Arctic University of Norway), Robert Jenssen `[通讯]` (UiT Arctic University of Norway)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `70e40602-aae3-44bd-80ec-4a7f2674330f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于神经算子与层析理论的连续域感知降维框架 NOFE，能够学习从高维连续函数到低维连续函数的映射。

**💡 创新点**

创新点在于把降维视为连续层析映射（sheaf morphism）并通过图核算子（Graph Kernel Operator）实现无网格、解析度无关的函数映射；同时将局部信息与全局结构统一处理，突破传统点云方法的连续性和采样依赖限制。

**🔧 技术方法**

主要技术包括神经算子（Neural Operator）、图核算子（GKO）、多层感知器（MLP）预测核矩阵、基于学生‑t 核的相似度度量、Lipschitz 连续性评估及其在图上的消息传播。

**📊 数据集**

使用 ERA5 气候再分析数据（多变量 550 hPa 级别，欧洲网格 180×180）进行主实验，并在附录中对扩散 MRI 与卫星图像数据做进一步验证。

**📈 对比分析**

与 PCA、t‑SNE、UMAP 等传统方法对比，NOFE 在局部结构保留（Stress‑local 0.111 vs. 0.398/0.773/0.791）、采样独立性（Patch‑Stitching Error 13.0 vs. 22.2/59.0/267.6）以及超分辨率预测（可在未观测位置直接生成低维嵌入）等指标上均显著优于基线。

**⚠️ 局限性**

主要局限包括：对空间点数和邻域大小敏感，导致高分辨率网格下显著增加显存和训练时间；对超参数（邻域半径、消息传递步数等）高度依赖，需要数据集特定调优；在高度非欧几里得或几何复杂的流形上构建图的欧氏距离可能失效，影响性能。

---

## 556. H2G: Hierarchy-Aware Hyperbolic Grouping for 3D Scenes

**arXiv ID:** 2605.11967 | [PDF](https://arxiv.org/pdf/2605.11967v1)

**作者:** ByungHa Ko `[一作]` (Korea University), Dong Hwan Kim `[通讯]` (Korea University)

**通讯引用:** 24664 | [OpenAlex ID](https://openalex.org/A5100370734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在三维场景中实现多层次（从物体部件到完整物体）分组的方法，利用二维预训练模型SAM与DINO产生的语义相似度构建树形结构，并将其嵌入单一Lorentz超bolic特征场中；

**💡 创新点**

其创新点包括：①基于Dasgupta目标将语义相似度转换为层次树结构；②将所有层级映射至超bolic空间，消除尺度查询需求；③设计四个层次感知损失（叶子角度分离、根级角度、紧凑度与LCA顺序）以对齐特征与树结构；

**🔧 技术方法**

技术上结合NeRF式三维特征场、SAM与DINO分割与特征提取、谱二分法构造层次树、Lorentz超bolic投影与Einstein中点聚合，以及角度分类和LCA顺序损失；

**📊 数据集**

实验使用GARField场景（带层次注释）进行3D完备性与组召回评估，同时利用SAM与DINO的预训练模型进行2D监督；

**📈 对比分析**

在GARField 3D完备性与组召回任务中，H2G相较于GARField、SAM、OmniSeg3D等基线在视图级和层级级阈值选择下均取得最高平均准确率（例如从0.749提升到0.835），且在候选池规模更小的同时，mIoU、R@0.5、R@0.75均优于对照；

**⚠️ 局限性**

局限性在于对2D监督质量的高度依赖，噪声SAM或DINO相似度可能导致树结构错误；跨视图一致性不完全；谱二分法仅为近似构造，最终聚类仍需阈值或聚类算法。

---

## 557. What Does It Mean for a Medical AI System to Be Right?

**arXiv ID:** 2605.11963 | [PDF](https://arxiv.org/pdf/2605.11963v1)

**作者:** Antony Gitau `[一作]` `[通讯]` (University of South-Eastern Norway), Antony Gitau (University of South-Eastern Norway)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过对多发性骨髓瘤骨髓涂片中浆细胞自动分类的案例，系统阐述医学AI“正确性”的多维性，重点分析标签不稳定、模型过度自信、评价指标失当与自动化偏差四个维度；

**💡 创新点**

创新点在于提出将“正确性”拆解为标签稳定性、模型不确定性、评估指标与责任分配四个互相关联的维度，并强调软标签、校准不确定性、分层评价和人机协同的伦理规范，从哲学与伦理视角为医学AI设计与治理提供新框架；

**🔧 技术方法**

主要技术讨论包括软标签（soft labeling）处理标注不确定性、校准不确定性与选择性预测、注意力/可解释性可视化、针对不平衡问题的AUPRC、精确率/召回率等评估指标以及人机协同设计原理；

**📊 数据集**

以数字化骨髓涂片图像为研究对象，使用来自不同机构、由多名专家标注的数据集（未给出具体数据集名称），强调需采集多机构、多专家标注的临床图像数据；

**📈 对比分析**

文章未进行实验对比，而是通过案例和理论分析指出传统准确率可能产生误导，建议采用分层评价、置信度阈值和软标签等方法进行比较，性能数据未给出；

**⚠️ 局限性**

主要限制在于缺乏实证验证，框架仍需在真实临床数据上测试；软标签与不确定性校准实现技术挑战；人机协同界面与工作流程设计尚缺标准；责任归属的法律与规范尚未完善。

---

## 558. Chronicles-OCR: A Cross-Temporal Perception Benchmark for the Evolutionary Trajectory of Chinese Characters

**arXiv ID:** 2605.11960 | [PDF](https://arxiv.org/pdf/2605.11960v1)

**作者:** Gengluo Li `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Han Hu `[通讯]` (Tencent)

**通讯引用:** 10214 | [OpenAlex ID](https://openalex.org/A5091049278)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了Chronicles-OCR基准，系统评估视觉大型语言模型（VLLMs）在七种中国书体跨时空的视觉感知能力。

**💡 创新点**

创新点在于首次覆盖完整书体演化，提出阶段自适应标注范式，并设计四项任务将视觉感知与语义推理分离。

**🔧 技术方法**

技术上采用多模态预训练模型（OpenAI GPT、Qwen、Gemma等）和视觉参考机制进行评测，并使用H‑mean、Exact Match、NED和准确率等指标。

**📊 数据集**

数据集为Chronicles‑OCR，包含2800张来自龟甲、青铜、石碑、纸张等多媒体的高质量图像，按专家分层标注。

**📈 对比分析**

通过对比多款开源与专有VLLMs的四项任务，发现模型在古文字定位和识别上接近零，成熟书体的解析略好，但整体性能远低于现代文本OCR。

**⚠️ 局限性**

局限性包括缺乏细粒度笔画感知、对古文字形态的语义映射不足、推理模块未能弥补感知缺陷，以及评测仅聚焦中文书体。

---

## 559. From Noise to Diversity: Random Embedding Injection in LLM Reasoning

**arXiv ID:** 2605.11936 | [PDF](https://arxiv.org/pdf/2605.11936v1)

**作者:** Heejun Kim `[一作]` (Gwangju Institute of Science and Technology), Sundong Kim `[通讯]` (Gwangju Institute of Science and Technology)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5073596347)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了随机软提示（RSP），通过无训练、随机生成的正态向量注入LLM输入，验证其能提升推理性能。

**💡 创新点**

创新点在于剔除软提示的学习步骤，证明仅注入即可产生分支多样性，并通过注意力权重衰减理论解释其机制，且将RSP应用于推理与强化学习训练。

**🔧 技术方法**

使用随机正态分布生成嵌入、注意力矩阵分析、Pass@N 计算及RL方法DAPO与RSP组合等技术。

**📊 数据集**

实验基准包括数学推理数据集MATH-500、GSM8K、AIME24，以及College Math、Minerva Math 等。

**📈 对比分析**

通过与无RSP基线以及已优化的软提示方法（TTSV、SoftCoT、LTPO）在相同模型配置下对比，RSP在多种模型上取得与优化方法相近甚至更好的准确率，尤其在格式不匹配恢复和Pass@N提升方面表现突出。

**⚠️ 局限性**

局限在于仅在RoPE编码的数学推理场景验证，未探讨多答案任务的适用性；缺乏对层轴衰减的完整理论；RSP需要独立重采样才能获得多样性提升。

---

## 560. Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training

**arXiv ID:** 2605.11931 | [PDF](https://arxiv.org/pdf/2605.11931v1)

**作者:** Qihuang Zhong `[一作]` (Wuhan University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 101243 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态自我改进训练框架 VISTA，利用前缀重采样和视觉注意力得分两项技术提升 MLLM 的推理能力。

**💡 创新点**

创新点在于：①通过前缀重采样重新利用部分正确推理轨迹解决数据不平衡问题；②设计视觉注意力得分（VAS）自动筛除视觉幻觉，缓解语言先验偏置。

**🔧 技术方法**

核心技术包括前缀重采样、视觉注意力得分（VAS）、基于 SFT、DPO、GRPO 的微调与强化学习策略。

**📊 数据集**

实验使用 SLAKE、VQA‑Rad、Geometry3K、ScienceQA、ChartQA、IllusionBench、PathVQA、MathVista、MMMU 等多模态推理与幻觉数据集。

**📈 对比分析**

与 SFT‑Seed、STaR、ReST^EM、R3V、IRPO 等基线对比，VISTA 在 Qwen2.5‑VL‑3B/7B 上平均提升 13.66% / 6.67% 的推理准确率，并在 OOD 与幻觉评测中表现最优。

**⚠️ 局限性**

局限性包括前缀重采样可能降低轨迹多样性，实验规模仅涵盖 3B/7B 模型，未验证在更大模型或 MoE 架构上的效果。

---

## 561. PROTECT-DB: Protecting Data using Replicated State Machines: Efficient Corruption Detection & Recovery

**arXiv ID:** 2605.11953 | [PDF](https://arxiv.org/pdf/2605.11953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 562. Citation Cliques in Low Impact Journals

**arXiv ID:** 2605.11930 | [PDF](https://arxiv.org/pdf/2605.11930v1)

**作者:** Panagiotis-Alexios Spanakis `[一作]` (Athens University of Economics and Business), Diomidis Spinellis `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 8972 | [OpenAlex ID](https://openalex.org/A5021948425)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过构建作者级定向引用网络，比较低影响期刊与高影响期刊作者在自引、合著引用及网络凝聚力上的差异。

**💡 创新点**

创新点在于将LLM驱动的期刊主题分类、主题Eigenfactor评估、作者级h_5匹配与层次化异常检测（混合孤立森林+凝聚度分数）相结合，形成一种新型的“引用卡特尔”识别框架。

**🔧 技术方法**

技术手段包括：OpenAI LLM进行期刊主题归类、Eigenfactor计算、作者h_5匹配、定向引用网络构建、特征标准化、混合孤立森林+凝聚度分数的异常检测、k-means聚类、LDA可视化等。

**📊 数据集**

数据集为2020‑2024年Crossref元数据（DOI、作者、期刊、引用）及公开ORCID信息，最终得到9431对低影响与高影响作者的匹配样本。

**📈 对比分析**

使用配对Wilcoxon检验、效应量（Cliff's δ、Cohen's d）与bootstrap置信区间比较，结果显示低影响作者的合著引用率提升6.7倍、互惠率提升4.7倍；异常检测筛选出277个高置信度案例，其中93.5%为低影响作者，验证了模型的高精准度。

**⚠️ 局限性**

局限性包括：仅基于Crossref元数据，缺乏文本上下文与编辑审稿信息；LLM期刊分类未经独立验证；无法完全区分“引用卡特尔”与高度专业化的小型研究团队。

---

## 563. Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning

**arXiv ID:** 2605.11975 | [PDF](https://arxiv.org/pdf/2605.11975v1)

**作者:** Jingduo Pan `[一作]` (Key Laboratory of System Software (Chinese Academy of Sciences), Institute of Software, Chinese Academy of Sciences), Bai Xue `[通讯]` (Key Laboratory of System Software (Chinese Academy of Sciences), Institute of Software, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种在随机环境下同时满足概率到达‑避免约束并最小化累计成本的强化学习框架，称为RAPCPO。

**💡 创新点**

创新点在于引入Reach‑Avoid Probability Certificate（RAPC）以及基于收缩的贝尔曼递归，利用证书构造可计算的下界并驱动策略优化。

**🔧 技术方法**

采用Actor‑Critic（PPO）框架，结合贝尔曼递归、补偿因子ϕγ、梯度修正及证书引导的状态空间分区约束优化。

**📊 数据集**

实验基于MuJoCo仿真环境（PointGoal、FixedWing、Pendulum、Safety Hopper、Safety HalfCheetah）以及FrozenLake进行消融验证。

**📈 对比分析**

与CMDP、CVaR、RC‑PPO、PPO_β等基线对比，RAPCPO在成本最低且成功率最高，尤其在随机噪声环境中显著优于其他方法。

**⚠️ 局限性**

局限性包括：证书条件仅为充分条件，缺乏必要条件；对超参数p和补偿因子敏感；在更大规模任务中的计算与收敛性仍待进一步研究。

---

## 564. Enhancing Target-Guided Proactive Dialogue Systems via Conversational Scenario Modeling and Intent-Keyword Bridging

**arXiv ID:** 2605.11964 | [PDF](https://arxiv.org/pdf/2605.11964v1)

**作者:** Maodong Li `[一作]` (Soochow University), Fang Kong `[通讯]` (Soochow University)

**通讯引用:** 800 | [OpenAlex ID](https://openalex.org/A5102803936)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造对话场景并预测多步意图关键词，联合用户画像与领域知识来引导目标导向的主动对话生成。

**💡 创新点**

① 将用户画像和领域知识联合建模为对话场景，实现动态场景偏置；② 通过意图关键词桥接（IKB）动态预测未来多轮意图关键词，提供更高层次、更灵活的引导；③ 软/硬模式处理预测不确定性。

**🔧 技术方法**

采用T5 encoder/decoder框架，使用平均池化+MLP进行场景编码，最大池化与分类头完成意图关键词预测，利用Softmax、交叉熵损失、NLL联合训练。

**📊 数据集**

使用DuRecDial与DuRecDial2.0两个数据集（含ID/OOD子集），并在此基础上对比大型LLM（LLaMA、Qwen）以及专门的目标导向模型。

**📈 对比分析**

与TPDial、TRIPDial以及LLM的微调/提示方法对比，模型在DuRecDial2.0上取得W.F1 44.87、BLEU‑1/2 0.416/0.310、K.F1 61.17，且失败率仅20‑23%，明显优于基线且在OOD数据上表现更稳健；人类评测亦显示在连贯性、适宜性、主动性上明显领先。

**⚠️ 局限性**

仅基于DuRecDial数据集，未对意图关键词进行细粒度分类；在大模型对比中并未始终优于LLM；场景偏置和关键词桥接可能需要更柔性设计；未使用LLM作为骨干，资源受限；未来需进一步细化关键词、改进提取方法、探索LLM骨干。

---

## 565. Counterfactual Trace Auditing of LLM Agent Skills

**arXiv ID:** 2605.11946 | [PDF](https://arxiv.org/pdf/2605.11946v1)

**作者:** Xiaolin Zhou `[一作]` (Arizona State University), Xiyang Hu `[通讯]` (Arizona State University)

**通讯引用:** 868 | [OpenAlex ID](https://openalex.org/A5044665455)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了Counterfactual Trace Auditing（CTA）框架，用以对比大语言模型代理在启用与未启用技能时的行为轨迹。

**💡 创新点**

创新点在于通过阶段分割、意图对齐和差异记录，将技能对代理行为的细粒度影响转化为可计量的Skill Influence Pattern（SIP），突破了仅用通过率评估的局限。

**🔧 技术方法**

技术包括基于DTW的阶段对齐、TF‑IDF相似度意图匹配、规则驱动的SIP检测，以及对齐后差异的结构化记录。

**📊 数据集**

数据集为公开的SWE‑Skills‑Bench，使用Claude Sonnet 4.5在49个软件工程任务上生成配对轨迹。

**📈 对比分析**

与传统通过率（ΔP）比较时，CTA发现平均通过率提升仅+0.3个百分点，但却捕捉到522个SIP实例，表明技能对行为有显著影响但被通过率掩盖；在中间基准任务中通过率提升+3.6个百分点且token消耗平均2.77倍。

**⚠️ 局限性**

局限包括仅单次实验、单模型单基准、规则检测无人工标注评估、阶段分割未验证、以及基准在高通过率任务上饱和。

---

## 566. When Simulation Lies: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents

**arXiv ID:** 2605.11928 | [PDF](https://arxiv.org/pdf/2605.11928v1)

**作者:** Xiaolin Zhou `[一作]` (Arizona State University), Xiyang Hu `[通讯]` (Arizona State University)

**通讯引用:** 868 | [OpenAlex ID](https://openalex.org/A5044665455)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个以 POMDP 为框架的工具调用鲁棒性基准 RobustBench‑TC，设计并实现了 22 种基于观察、动作、奖励与转移四个组件的扰动；对 21 个不同规模的工具使用模型进行评估，并提出基于域随机化的 RL 训练方法 ToolRL‑DR，验证其能显著提升模型在奖励和转移扰动下的表现。

**💡 创新点**

（1）首次将工具调用的四个 POMDP 组件与真实生产中出现的失效模式统一归类，形成了可复现的扰动体系；（2）跨 5 个主流基准构建统一数据集，保证评测结果的广泛代表性；（3）通过仅对可静态修改的三类扰动进行域随机化训练，展示了对未见过的转移错误的迁移鲁棒性，首次证明了此类 RL 训练能在部署级错误上获得提升。

**🔧 技术方法**

POMDP 建模、LLM 生成扰动、Rule‑based 扰动注入、GRPO + 结构化奖励的 RL 训练、vLLM 作为推理服务器、Bootstrap 统计评估与 Deterministic 解析器。

**📊 数据集**

来自 BFCL V3、API‑Bank、RoTBench、ToolAlpaca、ToolEyes 5 个工具调用基准的 199 条干净样本，扩增得到 3522 条扰动样本，完整评测集 3721 条。

**📈 对比分析**

对 21 个模型（包括 RL 训练模型、SFT 模型、闭源大模型等）按规模对比，发现观察扰动 <5% 影响，动作扰动 10–20%，奖励约 40% 影响，转移约 30% 影响；规模提升并未显著抑制奖励与转移误差。采用 ToolRL‑DR‑Full 的 3B Backbone 在奖励类扰动上实现约 27% 的 gap 缩减，且在未见过的转移扰动上也实现约 27% 的提升，表明方法具有一定的泛化能力。

**⚠️ 局限性**

仅评估单轮交互，未覆盖多轮对话与累计错误；训练规模局限于 3B 参数，未知能否推广到更大模型；转移到转移扰动的提升不完整，仍需通过带错误响应的 RL 或自适应重试策略进一步完善；缺乏对实际生产流量的直接验证。

---

## 567. Efficient and Adaptive Human Activity Recognition via LLM Backbones

**arXiv ID:** 2605.12019 | [PDF](https://arxiv.org/pdf/2605.12019v1)

**作者:** Aleksandr Bredikhin `[一作]` (University Grenoble Alpes), German Vega `[通讯]` (University Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将预训练的大型语言模型（LLM）重新用于人体动作识别任务，通过卷积前端将惯性传感器信号投影到LLM潜在空间，并采用LoRA实现参数高效微调。

**💡 创新点**

创新点在于：①将LLM作为通用时间序列编码器而非专门设计的Transformer；②通过结构化卷积投影弥合信号与语言模型的模态差距；③利用LoRA实现冻结LLM并仅微调极少参数，从而显著降低训练成本。

**🔧 技术方法**

技术包括：1D卷积前端、多分支特征融合、LLM（如GPT‑2 Small）作为骨干、LoRA参数高效微调、时间均值池化+线性分类头。

**📊 数据集**

实验使用三大标准HAR基准：UCI HAR、HHAR（多设备、多用户）和RealWorld（真实场景、传感器摆放多变）。

**📈 对比分析**

与传统CNN、DeepConvLSTM、Transformer、HART等基线对比，HARLLM在HHAR上最高Weighted F1（98.43%），在UCI和RealWorld上保持竞争力；在少量训练样本和跨数据集迁移场景下亦表现出优异的样本效率和快速收敛。

**⚠️ 局限性**

局限性主要在于：对传感器摆放和姿态的局部不变性不如专门设计的卷积前端处理；在RealWorld等强几何变异数据集上性能略逊于HART；LLM作为时序推理器的优势在极少标注数据下显著，但在充分数据时提升有限。

---

## 568. SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces

**arXiv ID:** 2605.12015 | [PDF](https://arxiv.org/pdf/2605.12015v1)

**作者:** Chang Jin `[一作]` (Shanghai AI Laboratory), Xingcheng Xu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 SkillSafetyBench，一个基于可执行工作流的可重用技能安全评估基准，包含155个针对本地非用户攻击面的对抗案例。

**💡 创新点**

创新点在于首次将技能层面与本地环境攻击相结合，构造出多维度风险域和攻击类标签，并为每个案例提供规则式验证器，实现可复现的安全测试。

**🔧 技术方法**

使用了可执行工作流、规则式验证器、LLM-as-judge 验证、以及多种 CLI 代理与多模型后端（如 GPT‑5.5、GLM‑5.1 等）的对比实验。

**📊 数据集**

数据集基于 SkillsBench 的 47 个任务子集，生成 155 个对抗案例，涵盖 6 个风险域、30 个安全类别和 8 个攻击类标签。

**📈 对比分析**

通过在多组合（CLI+模型）上计算攻击成功率（ASR）与任务成功率的散点图进行比较，发现 Codex+GLM‑5.1 的 ASR 高达 50% 但任务完成率仅 40%，表明高安全风险与任务能力不完全正相关。

**⚠️ 局限性**

局限性包括只评估可执行的本地攻击面，缺乏对更广泛工具链或外部 API 攻击的覆盖；验证器基于规则，可能无法捕捉所有非结构化攻击。

---

## 569. Split the Differences, Pool the Rest: Provably Efficient Multi-Objective Imitation

**arXiv ID:** 2605.12000 | [PDF](https://arxiv.org/pdf/2605.12000v1)

**作者:** Ziyad Sheebaelhamd `[一作]` (University of Tübingen), Claire Vernade `[通讯]` (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究多目标模仿学习问题，提出 Multi‑Output Augmented Behavioral Cloning (MA‑BC) 算法，能够从多个 Pareto 最优专家的演示中恢复 Pareto 前沿上的策略。

**💡 创新点**

创新点包括：① 引入“Pareto Path 存在”结构性结果，说明相邻 Pareto 策略仅需逐状态修改；② 设计 MA‑BC 通过识别专家冲突状态并聚合共识状态，实现数据重用与冲突避免；③ 给出多目标模仿学习下的下界，并证明 MA‑BC 在统计收敛率上是 minimax 最优。

**🔧 技术方法**

使用技术：行为克隆（Behavioral Cloning）、冲突检测与数据聚合、理论收敛与下界分析、连续空间扩增（基于 Lipschitz 常数和 δ 区域）、LQR 回归用于连续任务。

**📊 数据集**

实验数据集：离散基准三大 MOMDP（Deep Sea Treasure、Resource Gathering、Slippery Y‑Maze）以及一个连续的 6‑DOF LQR 四旋翼控制任务。

**📈 对比分析**

与 Naive BC（全聚合）和 Independent BC（单专家）对比。MA‑BC 在所有离散任务中以更快的统计率收敛至 Pareto 前沿，样本效率明显优于 Independent BC，并成功避免 Naive BC 失效。在连续任务中，通过调节 δ 实现 bias‑variance 权衡，实验表明 MA‑BC 在低数据量下可显著提升性能。

**⚠️ 局限性**

局限性：① 对连续空间的扩展缺乏严格理论保证；② 需要先验知道专家身份（或额外分组步骤）；③ 性能受浓度系数 C 的影响，当专家信息不共享或 C 较大时，MA‑BC 可能不优于独立学习；④ δ 参数需经验调节，难以自动化。

---

## 570. FAME: Feature Activation Map Explanation on Image Classification and Face Recognition

**arXiv ID:** 2605.12017 | [PDF](https://arxiv.org/pdf/2605.12017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 571. BadSKP: Backdoor Attacks on Knowledge Graph-Enhanced LLMs with Soft Prompts

**arXiv ID:** 2605.11996 | [PDF](https://arxiv.org/pdf/2605.11996v1)

**作者:** Xiaoting Lyu `[一作]` (Xi'an Jiaotong University), Wei Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 253022 | [OpenAlex ID](https://openalex.org/A5100444820)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了知识图谱增强大型语言模型（KG-enhanced LLM）在软提示（soft prompt）下的后门攻击安全性，提出了一种针对图‑到‑提示接口的攻击方法BadSKP。

**💡 创新点**

创新点在于：①揭示软提示的语义锚定机制导致文本通道攻击失效；②设计了多阶段优化流程同时扰动图结构和节点属性，以破坏锚定并实现后门；③引入梯度对齐损失提升后门在下游微调后的持久性。

**🔧 技术方法**

主要技术包括图神经网络（GNN）编码、连续软提示投影、目标嵌入构造、节点嵌入优化、语言模型辅助的文本后缀搜索以及对抗性梯度对齐。

**📊 数据集**

实验使用了WebQuestionsSP和ComplexWebQuestions两个知识问答数据集，并在其中选取多个触发实体进行攻击评估。

**📈 对比分析**

与两种仅文本通道的基线（ACPI、GCG）相比，BadSKP在冻结和Trojaned两种设置下都显著提升了攻击成功率（ASR），同时保持接近正常的准确率（ACC），即使在基于困惑度的防御下也保持较高ASR。

**⚠️ 局限性**

限制包括：攻击仍需对KG、检索和软提示模块有完整白盒信息；对较短或结构化文本属性的改动易被困惑度过滤；在某些模型或硬件条件下的可扩展性未完全验证。

---

## 572. sweap: Reactive Synthesis for Infinite-State Integer Problems

**arXiv ID:** 2605.11992 | [PDF](https://arxiv.org/pdf/2605.11992v1)

**作者:** Shaun Azzopardi `[一作]` (Dedaub), Nir Piterman `[通讯]` (University of Gothenburg and Chalmers University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一个名为Sweap的工具，用于在无限状态整数算术领域进行反应式合成。

**💡 创新点**

创新点包括双重抽象方法（同时考虑可达性与不可达性）、支持非确定性与无界更新、使用二进制谓词编码以降低抽象复杂度，以及针对多种输入形式的等价可实现转换与简化。

**🔧 技术方法**

技术上采用了基于CEGAR的谓词抽象、LTL有限合成引擎（如Strix、SemML）作为黑盒、二进制谓词编码、加速假设、符号游戏图分析、以及输入/状态变量的量化消除等。

**📊 数据集**

实验使用了自定义输入语言、TSL、RPG等多种格式的基准集，并覆盖了文献中的Linear Integer Arithmetic合成问题。

**📈 对比分析**

与前身Sweap和其他工具ISSY的比较表明，新的Sweap在可实现和不可实现问题上都取得了更高的求解率和更快的运行时间，特别是在不可实现问题和含有Büchi目标的中等规模问题上表现突出。

**⚠️ 局限性**

主要限制包括对有限合成引擎的依赖导致的内存占用高、启动时间较长、以及目前仅支持线性整数算术，尚未扩展到线性实数算术。

---

## 573. Limits of Learning Linear Dynamics from Experiments

**arXiv ID:** 2605.12010 | [PDF](https://arxiv.org/pdf/2605.12010v1)

**作者:** Aybüke Ulusarslan `[一作]` (Technical University of Munich), Nora Schneider `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在缺乏可控性假设下，单一实验条件下受控线性时不变系统的可辨识性，提出“可见子空间”概念并给出闭式可辨识条件。

**💡 创新点**

创新点在于：①不依赖系统可控性，提供实验条件下的部分可辨识定理；②给出完整的闭式参数化与可辨识子系统的唯一性；③提出基于左特征向量与固定实验PBH检验的可辨识度量；④将连续时间可辨识结果映射至离散时间。

**🔧 技术方法**

采用Krylov子空间、持续激励（PE）理论、可见子空间投影、闭式参数化、矩阵正定性与奇异值判据等线性代数与控制理论技术。

**📊 数据集**

使用合成稀疏LTI系统（Ginibre + 稀疏化），随机初始状态和输入，模拟不同稀疏度、维度、噪声水平的实验数据。

**📈 对比分析**

对比四种识别方法（DMDc、MOESP、SINDy、神经ODE），在无噪声时识别子系统误差趋近0，整体误差受可见维度影响；在有噪声时子系统误差始终低于整体误差，且误差随可见维度增大而减小。

**⚠️ 局限性**

局限性包括：仅考虑完整状态测量；实验假设为稠密且包含PE信号；对非线性或非齐次系统的推广有限；实际应用需估计可见子空间而非已知。

---

## 574. Learning Agentic Policy from Action Guidance

**arXiv ID:** 2605.12004 | [PDF](https://arxiv.org/pdf/2605.12004v1)

**作者:** Yuxiang Ji `[一作]` (Xiamen University), Xiangxiang Chu `[通讯]` (AMAP, Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将仅含动作的演示轨迹作为计划式参考引导LLM代理进行agentic RL，采用最小干预原则动态选择最小足够的引导级别，并通过混合策略优化将有引导学习转移到无引导模型，显著提升稀疏奖励环境下的探索能力，降低对昂贵SFT的依赖。

**💡 创新点**

创新点在于：①将动作数据直接嵌入prompt作为非强制性计划式引导；②提出最小干预原则并用二分搜索自动选取最小足够的引导级别；③结合混合策略优化，实现引导与无引导rollout的联合学习，使探索收益内化到最终模型；④在不需要冷启动或大规模SFT的情况下实现与SFT+RL相当的性能。

**🔧 技术方法**

主要技术包括Agentic RL框架下的PPO/GRPO式策略梯度、Token‑level importance sampling、分布偏移量与方差评估、混合策略优化（guided/unguided）以及二分搜索实现最小干预。

**📊 数据集**

使用了搜索‑agent基准GAIA、WebWalkerQA、XBench、BrowseComp‑ZH进行在域评估，并在GPQA、TruthfulQA、IFEval上进行跨域测试；动作引导数据来源为公开的日常交互轨迹（GUI/CLI、游戏等）。

**📈 对比分析**

实验对比了零RL、SFT+RL以及标准RL基线，ActGuide‑RL在所有基线模型上平均提升10–30%（例如Qwen3‑4B在GAIA +12.8pp、WebWalker +13.6pp、XBench +10.7pp），且在无冷启动条件下与SFT+RL性能相当，表现出优异的鲁棒性和可扩展性。

**⚠️ 局限性**

局限性包括对动作数据质量的敏感度（噪声比例>20%会显著下降）、仍需手工收集多样化动作轨迹、在极端稀疏奖励或高度非结构化任务中可能仍受限，以及需要进一步验证在更大模型和更复杂环境中的迁移能力。

---

## 575. A microservices-based endpoint monitoring platform with predictive NLP models for real-time security and hate-speech risk alerting

**arXiv ID:** 2605.11997 | [PDF](https://arxiv.org/pdf/2605.11997v1)

**作者:** Darlan Noetzold `[一作]` (University of Salamanca), Valderi Reis Quietinho Leithard `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个统一的微服务平台，集成端点监控、实时安全分析与多语言恶意言论预测模型，实现端点事件与安全告警的即时关联与集中管理。

**💡 创新点**

创新点在于将多源端点信号（键盘记录、网络流量、进程审计、漏洞扫描）与Transformer‑based NLP 模型同步融合，采用风险评分机制和多信号决策，突破传统单一功能的隔离，实现端点安全与合规风险的实时联合预警。

**🔧 技术方法**

技术栈包括 RabbitMQ + Redis 事件流、PostgreSQL/Redis 数据持久化、Docker/Undertow 微服务、Prometheus+Grafana 可观测、Spyware 代理（Python多线程）以及 BERT、Logistic Regression、SVM、MNB 等机器学习模型，GPT‑3 作为后备分类器。

**📊 数据集**

使用多语言公开数据集：葡萄牙语（Linguistic Datasets、BraSNAM2018）、西班牙语（IberEval 2018、MEX‑A3T）、英语（Improved Cyberbullying Detection、Are You a Racist?），以及端点采集实验数据。

**📈 对比分析**

通过五折交叉验证得到平均准确率约 87%、平衡准确率 0.86‑0.90、AUC 0.88‑0.90；系统整体响应时间在 1 s 以下，10 k 并发请求平均 0.9 s，CPU 33%/内存 39%；Spyware 监控占用 <1% CPU/内存，捕获延迟 200 ms，推理 500 ms，整体周期 700 ms，表现出良好的实时性与可扩展性。

**⚠️ 局限性**

局限性包括：仅支持文本恶意言论检测，缺乏多模态分析；依赖外部 GPT‑3 产生额外成本与延迟；端点隐私与合规风险未充分解决；实验规模有限，未在真实企业大规模部署验证；模型训练受限于公开数据集，可能在新兴语言或行业语境下表现不足。

---

## 576. On the Limitations of Large Language Models for Conceptual Database Modeling

**arXiv ID:** 2605.11986 | [PDF](https://arxiv.org/pdf/2605.11986v1)

**作者:** Arthur F. Siqueira `[一作]` (Federal University of Campina Grande), Júlia Menezes `[通讯]` (Federal University of Campina Grande)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究使用大型语言模型（Qwen-3与GPT‑5.1）结合提示工程技术，自动从自然语言需求生成实体–关系（ER）图，并通过定性评估比较不同模型与提示策略在概念建模中的表现。

**💡 创新点**

提出了一套可复现的实验框架，整合多种提示策略（One‑shot、Chain‑of‑Thought、CoT+Verifier），构建并公开了包含三份递增复杂度需求与对应参考ER图的新数据集，系统评估LLM在ER图生成任务中的优势与局限。

**🔧 技术方法**

采用大型语言模型（Qwen‑3开源模型与GPT‑5.1商业模型）及提示工程（CoT、Verifier），Python实现的自动化流水线（JSON规范化 → ERDot → Graphviz渲染），并使用结构化评估清单进行定性分析。

**📊 数据集**

自制需求文档集（三份递增复杂度），以及对应的参考ER图，公开在GitHub上；该数据集为本研究的唯一实验素材。

**📈 对比分析**

通过四级概念质量评估（L1–L4）与概念清单对比，发现CoT提示相较于Baseline提升了模型表现，但Verifier效果有限；GPT‑5.1倾向过度规范，Qwen‑3表现为不足规范；两模型均未达到L3级别，整体性能随需求复杂度升高而下降。

**⚠️ 局限性**

评估方法主观性高，缺乏可量化指标；LLM生成的ER图存在冗余或缺失实体/关系；验证机制不足；实验仅涵盖两款LLM与三种提示策略，未覆盖多语言或更大规模数据，限制了结论的普适性。

---

## 577. From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation

**arXiv ID:** 2605.11951 | [PDF](https://arxiv.org/pdf/2605.11951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 578. QDSB: Quantized Diffusion Schrödinger Bridges

**arXiv ID:** 2605.11983 | [PDF](https://arxiv.org/pdf/2605.11983v1)

**作者:** Tobias Fuchs `[一作]` (Karlsruhe Institute of Technology), Nadja Klein `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 1618 | [OpenAlex ID](https://openalex.org/A5009563869)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 QDSB 量化耦合方法，使用锚点量化端点分布并在此基础上求解熵正则化 OT，减少模拟自由 Schrödinger 桥训练中的配对计算成本。

**💡 创新点**

创新点在于用锚点量化替代每个 mini‑batch OT，提供稳定性理论保证，并通过单次 OT 计划与细胞采样实现训练加速。

**🔧 技术方法**

采用的技术包括熵正则化 OT、Schrödinger 桥框架、Brownian 桥监督、k‑center 量化、细胞级采样，以及深度网络拟合漂移与分数。

**📊 数据集**

实验使用的资料集包括二维 toy（8Gaussians→Moons、N→Moons、N→8Gaussians）、细胞生物学数据集 EB、Cite、Multi，以及 FFHQ 成人‑儿童图像。

**📈 对比分析**

与 DSB、DSBM、SF2M、SF2M+mPOT、LightSB‑M 等基线在 MMD、时间‑质量曲线上进行对比，QDSB 在相同时间内获得更低 MMD，并显著降低总训练时间。

**⚠️ 局限性**

局限性包括对锚点选择的依赖、在高维空间量化误差可能增大、未在极大规模数据或非欧氏空间验证，以及需进一步验证在更复杂任务上的鲁棒性。

---

## 579. On Predicting the Post-training Potential of Pre-trained LLMs

**arXiv ID:** 2605.11978 | [PDF](https://arxiv.org/pdf/2605.11978v1)

**作者:** Xiaoyuan Li `[一作]` (University of Science and Technology of China), Dayiheng Liu `[通讯]` (Alibaba Group)

**通讯引用:** 1779 | [OpenAlex ID](https://openalex.org/A5062188134)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 RuDE 框架，通过基于 4C 维度的细粒度 Rubric 评估生成对抗样本，利用预训练模型的判别能力预测其后期指令微调的潜在性能。

**💡 创新点**

创新点在于将判别式评估与 4C 分类结合，构建了生成-判别协同的迭代流程；并通过 GD‑Potential 假设证明判别准确率与后期生成表现高度相关。

**🔧 技术方法**

技术手段包括：基于 Gemini‑3‑Pro 的生成器与 GPT‑4.1 验证器的交互式构造流程、对比样本生成、生成-评估一致性理论、DPO/GRPO 等强化学习微调以及 Pearson 相关性分析。

**📊 数据集**

使用的主要数据集有 HealthBench（Hard 子集）、PRBench、AdvancedIF（去除 System Prompt Modification）和 WritingBench，全部采用细粒度 Rubric 进行标注。

**📈 对比分析**

实验将 RuDE 判别准确率与指令微调后的生成性能进行比较，Pearson 相关系数在 0.90 以上；在 RL 验证中，RuDE 预测更高潜力的小模型 Qwen3‑4B 在 HealthBench 上超过同类大模型，显示显著的计算效率和性能提升。

**⚠️ 局限性**

局限性包括：需预先设定 Rubric 规则，对非 Rubric 任务适用性受限；判别结果受验证器（如 GPT‑4.1）质量影响；生成对抗样本构造成本高；模型在极端多模态或实时交互场景下的泛化性尚未验证。

---

## 580. Cooperative Robotics Reinforced by Collective Perception for Traffic Moderation

**arXiv ID:** 2605.11972 | [PDF](https://arxiv.org/pdf/2605.11972v1)

**作者:** Mohammad Khoshkdahan `[一作]` (Karlsruhe Institute of Technology), Alexey Vinel `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 7556 | [OpenAlex ID](https://openalex.org/A5003670494)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一套融合基础设施视觉与V2X通信的协作感知系统，利用人形机器人ARI实时检测主路车辆并在非视线交叉口通过手势和物理拦截方式阻止侧路车辆冲突。

**💡 创新点**

创新点在于将机器人作为主动交通调节器与协作感知（CPM）和V2X信息（CAM、DENM）深度融合，首次实现机器人在低V2X渗透率环境下物理介入并充当DENM中继，直接影响未装备V2X的车辆。

**🔧 技术方法**

技术包括RGB‑D摄像机+YOLOv12目标检测、2D LiDAR SLAM定位、V2X通信模块（ETSIT1/5 CAM/CPM/DENM）、基于Zone of Danger的碰撞预测算法、ROS与机器人控制系统、手势识别与执行。

**📊 数据集**

使用了Future Mobility Park（FMP）真实交通数据，包含来自双摄像头的CPM、VW ID.4的CAM以及RSU的DENM；YOLOv12采用公开预训练模型（如COCO）进行车辆/行人检测。

**📈 对比分析**

通过与无人机器人干预前后对比，系统在798辆来车中实现100%冲突预防；关键指标包括CPM延迟1.30 s、机器人停止时间12.59 s、车辆在危险区内时间10.99 s，证明融合感知显著提升实时性和安全性。

**⚠️ 局限性**

局限性包括摄像机检测距离受限、对高速车辆的速度估计不够精准、机器人响应时间仍较慢、地面机器人易受破坏且安全风险高、缺乏在更复杂交通场景（多车道、多方向）和大规模部署的验证。

---

## 581. Cluster-Aware Neural Collapse Prompt Tuning for Long-Tailed Generalization of Vision-Language Models

**arXiv ID:** 2605.11939 | [PDF](https://arxiv.org/pdf/2605.11939v1)

**作者:** Boyang Guo `[一作]` (Hangzhou Dianzi University), Chenggang Yan `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 14646 | [OpenAlex ID](https://openalex.org/A5054311881)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的 Cluster‑Aware Neural Collapse Prompt Tuning（CPT）框架，用于在类不平衡数据上提升提示调优的 VLM 的尾部类判别能力。

**💡 创新点**

创新点包括：① 通过在冻结的预训练文本特征上进行静态聚类构建“集群不变空间”，保持全局语义层次；② 在每个集群内部采用神经崩塌驱动的三重损失（文本 ETF 分离、类内收敛、旋转稳定）来提升局部判别与训练稳定性；③ 旋转稳定损失将可学习文本原型与冻结原型对齐，消除全局旋转自由度。

**🔧 技术方法**

核心技术包括提示调优、CLIP 预训练、K‑means（余弦）聚类、神经崩塌理论、ETF 约束、角度分离损失、类内收敛损失、旋转稳定损失以及对比学习。

**📊 数据集**

在 11 个多样化图像分类基准上进行实验，涵盖基于 ImageNet 的基-新、跨数据集、以及域泛化评估。

**📈 对比分析**

与 MaPLe、CoPrompt、NPT、DeKg、DPC 等现有最先进方法相比，CPT 在平衡、严重不平衡（τ=0.25、0.06）场景下均实现了更高的谐波均值与尾部类精度，同时保持或提升了跨域和新类的泛化能力。

**⚠️ 局限性**

局限性：① 依赖于静态聚类，无法在训练过程中动态调整集群边界；② 对聚类方法和超参数（λ_TETF、λ_CC、λ_RS）仍有一定敏感性；③ 目前仅针对提示调优的 VLM 进行评估，需进一步验证在其他模型或更大规模数据上的可推广性。

---

## 582. Interactive State Space Model with Cross-Modal Local Scanning for Depth Super-Resolution

**arXiv ID:** 2605.11934 | [PDF](https://arxiv.org/pdf/2605.11934v1)

**作者:** Chen Wu `[一作]` (National University of Defense Technology), Jiantao Zhou `[通讯]` (University of Macau)

**通讯引用:** 9835 | [OpenAlex ID](https://openalex.org/A5037979193)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于交互状态空间模型的RGB引导深度超分框架，融合跨模态局部扫描和特征匹配模块。

**💡 创新点**

创新点在于将线性复杂度的Mamba架构引入GDSR，设计跨模态局部扫描实现细粒度语义交互，并通过跨模态匹配变换进一步提升模态表征。

**🔧 技术方法**

主要技术包括交互状态空间模型（ISSM）、跨模态局部扫描（CMLS）、跨模态匹配变换（CMMT）以及Mamba Block、GDFN等。

**📊 数据集**

实验使用NYU‑v2、Middlebury、Lu三大基准，并在Hypersim上进行训练。

**📈 对比分析**

与DJF、DJFR、CUNet等十余种方法对比，本文方法在×4、×8、×16尺度下均取得最优RMSE，NYU‑v2 4×任务相对最佳方法降低10.38%。

**⚠️ 局限性**

局限性包括模型仍受限于Mamba的计算资源，对极大尺度或不同模态的适应性未作评估。

---

## 583. Multimodal Abstractive Summarization of Instructional Videos with Vision-Language Models

**arXiv ID:** 2605.11959 | [PDF](https://arxiv.org/pdf/2605.11959v1)

**作者:** Maham Nazir `[一作]` (Beihang University), Francesco Setti `[通讯]` (University of Verona)

**通讯引用:** 1145 | [OpenAlex ID](https://openalex.org/A5001003105)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种利用冻结CLIP视觉特征与BART结合的多模态抽象式教学视频摘要框架

**💡 创新点**

创新点包括冻结CLIP保持预训练视觉-语言对齐、显式时序建模、维度自适应跨模态融合，并证明冻结优于微调

**🔧 技术方法**

使用技术包括CLIP ViT‑B/32视觉编码、BART‑base文本编码/解码、Transformer时序编码、跨模态注意力融合及维度投影

**📊 数据集**

使用数据集为YouCook2（约2000条烹饪视频）

**📈 对比分析**

与文本仅模型、现有多模态基线和不同视觉特征（ResNet、随机）对比，CLIP+BART在ROUGE‑1上达到33.0%，高于ResNet（30.5%）及其他方法

**⚠️ 局限性**

局限性在于仅验证于烹饪视频、固定50帧采样、对更大规模或不同领域的泛化未评估，以及依赖CLIP预训练数据质量

---

## 584. UniCustom: Unified Visual Conditioning for Multi-Reference Image Generation

**arXiv ID:** 2605.12088 | [PDF](https://arxiv.org/pdf/2605.12088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 585. From Trajectories to Phenotypes: Disease Progression as Structural Priors for Multi-organ Imaging Representation Learning

**arXiv ID:** 2605.11958 | [PDF](https://arxiv.org/pdf/2605.11958v1)

**作者:** Zian Wang `[一作]` (Fudan University), Chengyan Wang `[通讯]` (Fudan University)

**通讯引用:** 11373 | [OpenAlex ID](https://openalex.org/A5085132583)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于轨迹蒸馏的影像表型预训练框架，将大规模诊断序列生成的疾病进展结构知识迁移至多器官影像表型编码器；

**💡 创新点**

创新点在于利用生成式疾病轨迹Transformer作为教师，对影像表型进行几何保持的对齐蒸馏，并可通过交叉注意力融合轨迹与影像特征；

**🔧 技术方法**

使用生成式轨迹Transformer（GPT‑2 结构）进行轨迹预训练；对影像表型采用组织级 Transformer 编码器；对齐损失采用 InfoNCE 或 MSE；融合阶段使用交叉注意力；

**📊 数据集**

实验基于英国生物银行（UK Biobank）502,387名受试者的多器官影像表型和诊断记录；

**📈 对比分析**

与从零开始训练、仅使用影像表型、仅使用轨迹表征以及多种融合方法对比；在159种疾病上，轨迹蒸馏平均提升 AUC 约 0.02-0.04，MAE 下降 0.04-0.1，尤其在低患病率疾病上提升显著；

**⚠️ 局限性**

局限包括仅在单一数据集验证；对不同轨迹模型（如 BEHRT、Med‑BERT）的教师效果未评估；基线对比范围有限；对外部数据集的泛化能力未知。

---

## 586. AB-Sparse: Sparse Attention with Adaptive Block Size for Accurate and Efficient Long-Context Inference

**arXiv ID:** 2605.12110 | [PDF](https://arxiv.org/pdf/2605.12110v1)

**作者:** Di Liu `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14632 | [OpenAlex ID](https://openalex.org/A5039318240)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于注意力头异质性自适应块大小分配的块稀疏注意力框架AB‑Sparse；在保持原有块稀疏结构的前提下，通过校准驱动的头级块尺寸分配、无损质心量化和专用GPU核实现，提高了长上下文推理的准确率。

**💡 创新点**

创新点在于①发现不同注意力头对块粒度的敏感度显著差异，并利用一次性校准即可获得稳定的自适应块尺寸；②设计了INT4逐通道非对称无损量化的质心压缩；③实现了三阶段自定义GPU核（查询‑质心估计、变块Top‑K选择、异构分页注意力），兼容现有KV缓存分页机制。

**🔧 技术方法**

技术手段包括轻量级校准驱动的头级块尺寸分配、INT4逐通道非对称量化的无损质心压缩、分段化的查询‑质心估计、基于前缀和的批处理Top‑K选择以及异构分页注意力核。

**📊 数据集**

使用了Wikipedia文本作为校准集，评估数据集包括RULER（四类任务共13个任务，16K–96K上下文）与LongBench（六类真实长文任务），以及长生成任务的AIME24/AMC23/MATH‑500。

**📈 对比分析**

与完整注意力、Quest和ArkVale进行对比，模型覆盖Llama‑3.1‑8B、Qwen3‑8B和Qwen3‑32B。AB‑Sparse在RULER上提升3.5–5.4%，在LongBench上提升2.4–2.6%，并在解码吞吐上与Quest持平甚至略快；INT4量化显著降低内存流量并保持准确率。

**⚠️ 局限性**

局限性包括：对极端稀疏或特殊块表示的适应性待验证；目前仅针对KV缓存块大小，未扩展至其它稀疏注意力形式；在不同GPU架构（如H800）上需进一步优化；量化在极低位宽下可能对部分头的排序精度产生影响。

---

## 587. Intermediate Artifacts as First-Class Citizens: A Data Model for Durable Intermediate Artifacts in Agentic Systems

**arXiv ID:** 2605.12087 | [PDF](https://arxiv.org/pdf/2605.12087v1)

**作者:** Josh Rosen `[一作]` (ThruWire, Inc.), Seth Rosen `[通讯]` (ThruWire, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种面向 AI 系统的“artifact-first”数据模型，旨在将多步推理中的中间产物（如证据表、论断矩阵、标准/假设、计划等）以可地址、可版本、可依赖、可权威的形式持久化，以支持后续检索、修改和重用。

**💡 创新点**

创新点在于：① 明确区分最终答案与中间产物的语义差异；② 为中间产物定义七大“首要属性”（Typed、Structured、Addressable、Versioned、Dependency-aware、Authoritative、Consumable）；③ 引入 additive 与 superseding 两种更新语义，配合 authority‑scope 进行状态解析；④ 将 artifact lineage 与执行 lineage 分离，形成可追溯的重算图；⑤ 提出了针对维护状态质量的评估指标（Authority Accuracy、Stale‑Artifact F1、Revision‑Localization Precision/Recall）。

**🔧 技术方法**

技术实现上，作者提出了一个最小参考架构：
- 图存储（记录步骤、边、依赖）
- 执行引擎（按图顺序执行并产生 artifact）
- Artifact store（保存 payload、身份、版本、依赖、状态等）
- Resolver（根据角色和 scope 解析当前有效 artifact）
- Supersession manager（标记被替代的 artifact 并触发失效事件）
- 编辑接口（供人类/代理检查、修改 artifact 或步骤）。

**📊 数据集**

论文没有使用专门的公开数据集，而是通过跨领域的示例（远程医疗政策、软件工程、法律分析、科研综述、产品规划）来阐述模型适用性和工作流程。它强调此框架可以与任何生成式 LLM 及其工具链集成。

**📈 对比分析**

评价方法：在仿真或真实任务中对比两种系统（传统 transcript‑centric 与 artifact‑first）在 Authority Accuracy、Stale‑Artifact F1、Revision‑Localization Precision/Recall 等指标上的表现；实验表明，artifact‑first 系统能在一次更新后仅重算受影响的 downstream artifact，显著减少不必要的重算，并保持更高的状态一致性。性能细节（如时间开销、存储占用）在论文中给出粗略估计，但未给出统一基准。

**⚠️ 局限性**

局限性包括：① 需要手工设计与维护 artifact schema，易出现脆弱性；② 颗粒度选择的主观性——粒度过粗会丢失细粒度更新，过细会导致 artifact 垃圾堆积；③ 仍依赖 LLM 的随机性，生成的 payload 可能不完全正确；④ 只适用于需要长期可追溯性的任务，短期交互或一次性对话可能不必要；⑤ 需要额外的工具链支持（如 graph executor、resolver），增加实现复杂度。

---

## 588. The Missing GAP: From Solving Square Jigsaw Puzzles to Handling Real World Archaeological Fragments

**arXiv ID:** 2605.12077 | [PDF](https://arxiv.org/pdf/2605.12077v1)

**作者:** Ofir Itzhak Shahar `[一作]` (Ben Gurion University of Negev), Ohad Ben-Shahar `[通讯]` (Ben Gurion University of Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了面向考古碎片的无规则拼图数据集GAP，并提出了基于ViT与离散流匹配的PuzzleFlow算法，用以在形状不规则、严重侵蚀的拼图上完成重组。

**💡 创新点**

创新点在于：①通过VAE从真实考古碎片学习生成器，生成形状逼真、侵蚀多样的碎片；②采用离散流匹配框架，在拼图重组中实现逐步迭代推理；③结合ViT对碎片的全局视觉特征与alpha掩码进行融合，提升对形状信息的利用。

**🔧 技术方法**

使用的技术包括：Vision Transformer (ViT-Base) 预训练并微调、1×1 RGBA→RGB投影层、离散流匹配（Discrete Flow Matching）、Transformer Encoder层进行全局关系推理、交叉熵损失与迭代推理策略。

**📊 数据集**

使用的数据集为：GAP-3（3×3，20,000个拼图）和GAP-5（5×5，20,000个拼图），每个拼图基于Metropolitan Museum of Art公开图片生成，碎片形状来自RePAIR考古碎片的VAE生成；同时对比了JPwLEG-3/5等传统方形拼图数据集。

**📈 对比分析**

与经典贪心、遗传算法以及深度学习基线（JigsawGAN、DiffAssemble、FCViT、JPDVT等）相比，PuzzleFlow在GAP-3上达到28.5%完美准确率（PA）和62.9%绝对准确率（AA），SRA也显著提升；在GAP-5上则实现0.3% PA、29.1% AA、19.8% SRA，显著优于第二名DiffAssemble，展示了在更大规模、形状更复杂的拼图上的竞争优势。

**⚠️ 局限性**

局限性包括：①仍然受限于网格拓扑，无法直接处理缺失碎片或非规则拼图布局；②对极大片段数的可扩展性不足，推理时间随片段数快速增长；③依赖于预训练ViT，需要较大规模的微调；④对极端侵蚀或完全无纹理碎片的鲁棒性尚待提升。

---

## 589. PairDropGS: Paired Dropout-Induced Consistency Regularization for Sparse-View Gaussian Splatting

**arXiv ID:** 2605.12072 | [PDF](https://arxiv.org/pdf/2605.12072v1)

**作者:** Hantang Li `[一作]` (Harbin Institute of Technology), Xiaopeng Fan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 4037 | [OpenAlex ID](https://openalex.org/A5079412089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在稀疏视角下对3D高斯散射进行对偶 dropout 训练并加入低频一致性正则化以提升重建稳定性和质量。

**💡 创新点**

创新点是将不同 dropout 子集的渲染结果构成对偶，使用低频一致性正则化和逐步加权的调度策略，避免过度约束高频细节，同时显著提升训练稳定性。

**🔧 技术方法**

主要技术包括对偶分支训练、低频滤波（高斯模糊）一致性损失、progressive consistency scheduling、DropGaussian 基础框架。

**📊 数据集**

使用 LLFF、MipNeRF‑360、Blender 三大稀疏视角基准数据集。

**📈 对比分析**

与 DropGaussian、DropAnSH‑GS、FSGS、CoR‑GS 等现有 dropout/3DGS 方法对比，PairDropGS 在 PSNR、SSIM、LPIPS 上均取得更高或相近分数，且训练稳定性大幅提升。

**⚠️ 局限性**

局限在于仍需两分支训练，计算成本相对单分支提升约一倍；对高频细节的精细恢复仍依赖原始 dropout 策略。

---

## 590. Control of Fully Actuated Aerial Vehicles: A Comparison of Model-based and Sensor-based Dynamic Inversion

**arXiv ID:** 2605.12071 | [PDF](https://arxiv.org/pdf/2605.12071v1)

**作者:** Ali Sidar Yilmaz `[一作]` (Technical University of Munich), Markus Ryll `[通讯]` (Technical University of Munich)

**通讯引用:** 2553 | [OpenAlex ID](https://openalex.org/A5018909750)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在固定倾斜全致动六旋翼上对几何非线性动态反演（NDI）与传感器增量动态反演（INDI）两种控制器进行实验比较，重点验证了INDI在全姿态跟踪中的性能优势

**💡 创新点**

首次在全致动平台上实现并实验验证了完整姿态跟踪的INDI控制器，且通过同一外环结构实现两种反演策略的公平对比

**🔧 技术方法**

采用了几何NDI和传感器增量NDI控制器，外环为共享的SE(3)误差驱动器，采样频率为500 Hz；平台使用IMU、Vicon运动捕捉与卡尔曼滤波状态估计

**📊 数据集**

实验使用自制的全致动六旋翼平台，实验数据包括参数不匹配、水平扰动、风 gust、不同控制频率以及传感器降噪等5组场景，不涉及公开数据集

**📈 对比分析**

通过相同的误差动力学与执行频率，比较两者在姿态误差、位置误差、转速响应等指标；结果显示INDI在参数误差、扰动抑制、传感器降噪时表现更好，而几何NDI在低频控制下姿态误差更稳健；整体而言，INDI在翻译运动上优势更显著

**⚠️ 局限性**

优势不全面：INDI对采样频率更敏感，低频时姿态性能下降；几何NDI在低频下更稳定；两者仍需进一步研究在欠驱动平台、不同工况下的鲁棒性与实现复杂度

---

## 591. Do Language Models Encode Knowledge of Linguistic Constraint Violations?

**arXiv ID:** 2605.12055 | [PDF](https://arxiv.org/pdf/2605.12055v1)

**作者:** Hardy `[一作]` (University of Stuttgart), Sebastian Padó `[通讯]` (University of Stuttgart)

**通讯引用:** 6115 | [OpenAlex ID](https://openalex.org/A5003870894)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过稀疏自编码器提取特征并利用置信度分数，探究大规模语言模型是否在参数中编码了专门用于识别语法违规的“检测器”，并对不同模型在BLiMP基准上的表现进行因果消融实验验证。

**💡 创新点**

创新点在于提出一种无监督的敏感度得分用于筛选可能与语法违规相关的特征，以及一个三项共识的否定性检验框架（F1‑F3），从而系统评估特征的因果性、稳定性与选择性。

**🔧 技术方法**

主要技术包括稀疏自编码器（Sparse Autoencoders）对Transformer残差流的分解、基于最小差异对的消融干预、Wilcoxon符号秩检验以及三维对照的因果检验策略。

**📊 数据集**

使用BLiMP（包含13个语言学现象的最小对句子集）作为评估数据集，并在六个Transformer模型（GPT‑2 124M、Gemma‑3 270M/1B/4B、Qwen‑3.5 2B/9B）上进行实验。

**📈 对比分析**

通过在每个模型和层上计算F1‑F3的通过情况，对比随机干预的基准，结果显示F1普遍通过，说明特征与违规相关；但F2（非违规稳定性）和F3（违规选择性）很少同时满足，整体上只有极少数现象在某层能满足所有三项检验，说明模型并未清晰分离出专门的违规检测器。

**⚠️ 局限性**

局限性包括：受算力限制仅评估了中小规模模型；评估框架仅关注语法违规且未涵盖语义异常或合理性等其它可能的违规信号；稀疏自编码器可能无法完整恢复所有相关特征，导致特征分布跨层或被混合吸收。

---

## 592. Is Child-Directed Language Optimized for Word Learning? A Computational Study of Verb Meaning Acquisition

**arXiv ID:** 2605.12047 | [PDF](https://arxiv.org/pdf/2605.12047v1)

**作者:** Francesca Padovani `[一作]` (University of Groningen), Arianna Bisazza `[通讯]` (University of Groningen)

**通讯引用:** 2455 | [OpenAlex ID](https://openalex.org/A5019968969)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练Transformer自回归语言模型在儿童指向语（CDL）与成人指向语（ADL）上，比较语法与词汇共现扰动对动词意义获取的影响

**💡 创新点**

系统检验语法扰动对动词学习的优先性，并揭示口语输入（无论儿童或成人）在语法扰动下更具鲁棒性，而非CDL独有

**🔧 技术方法**

使用GPT‑2模型，采用词序扰动和词汇替换两种输入扰动方法，并以语义与句法最小对进行评估

**📊 数据集**

四大语料库：CDL（CHILDES）、口语ADL（CANDOR+Switchboard+BNC）、书面ADL（BNC）与维基百科

**📈 对比分析**

通过对比原始、词序扰动、词汇替换三种条件下的语义最小对准确率，发现词序扰动导致更大性能下降；在语法与语义发展轨迹上，CDL表现出“语义先行”模式，但与书面ADL相比，语法敏感性更低

**⚠️ 局限性**

实验仅在英语且以词序为主要语法特征，未细化具体句法结构；最小对测试与语法注意力等更细粒度分析有限，跨语言通用性尚待验证

---

## 593. Resilient Vision-Tabular Multimodal Learning under Modality Missingness

**arXiv ID:** 2605.12031 | [PDF](https://arxiv.org/pdf/2605.12031v1)

**作者:** Camillo Maria Caruso `[一作]` (Università Campus Bio-Medico di Roma), Paolo Soda `[通讯]` (Università Campus Bio-Medico di Roma)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种在训练与推理阶段均能应对缺失图像或表格模态的多模态Transformer框架，利用注意力遮蔽与模态丢弃实现鲁棒融合；

**💡 创新点**

核心创新在于在中间融合层显式采用遮蔽自注意力，仅聚合可观测模态并通过模态Dropout正则化提升对任意缺失模式的稳健性，无需补值或模型切换；

**🔧 技术方法**

技术手段包括ResNet-50视觉编码器、Transformer式表格编码器、中间融合Transformer、模态Token、位置编码、模态Dropout以及目标缺失掩码；

**📊 数据集**

实验基于MIMIC-CXR胸部X光与MIMIC-IV临床数据的结合，构建62071例样本并标注14个诊断标签；

**📈 对比分析**

通过与零填充、最大池化、模型选择等主流缺失处理策略在训练缺失与测试缺失两种压力测试协议下对比，平均加权AUC持续领先基线，性能曲线更平滑、鲁棒；

**⚠️ 局限性**

局限在于仅验证于胸部X光单一任务，未扩展到文本或时间序列模态，Transformer融合开销较大，且未建模缺失机制本身的潜在信息。

---

## 594. Spectral Vision Transformer for Efficient Tokenization with Limited Data

**arXiv ID:** 2605.12026 | [PDF](https://arxiv.org/pdf/2605.12026v1)

**作者:** Alexandra G. Roberts `[一作]` (Cornell University), Yi Wang `[通讯]` (Cornell University)

**通讯引用:** 18058 | [OpenAlex ID](https://openalex.org/A5100364902)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种在频域（PCA/傅里叶/拉普拉斯）中对图像进行 token 化的 Spectral Vision Transformer（Spectral ViT），并在医学图像小样本场景下进行评估。

**💡 创新点**

创新点在于用固定的谱基（如 PCA）将图像投影到低秩空间生成 tokens，并利用谱秩层级作为位置编码，显著降低参数量和计算复杂度，同时实现全局性、空间不变性的注意力建模。

**🔧 技术方法**

使用 PCA、傅里叶变换、拉普拉斯算子等谱变换做 token 化；自注意力 Transformer；对比实验中还使用标准空间 ViT、Swin ViT、U‑net Attention、逻辑回归、MLP 等模型。

**📊 数据集**

实验数据包括：噪声与棋盘图案的模拟二分类数据；公开 IXI T1‑weighted 脑图像用于性别分类；内部和外部的全脑定量磁敏感性图（QSM）用于深脑刺激（DBS）结果预测。

**📈 对比分析**

与空间 ViT、Swin、U‑net 等方法对比，Spectral ViT 在样本量较少时（如 <1000）即可达到或超过空间 ViT 的 AUC，IXI 数据上 AUC 提升至 0.842（vs 0.826），DBS 预测 AUC 0.807（vs 0.804），且参数量仅约 1.3 万。

**⚠️ 局限性**

局限性包括：对 PCA 基需要图像配准；固定基线引入偏置，可能在大量预训练数据下不如全学习；谱基可能产生退化模式、噪声、混叠；并非所有谱模式对任务都有用，需要选择合适基。

---

## 595. Approximation Theory of Laplacian-Based Neural Operators for Reaction-Diffusion System

**arXiv ID:** 2605.12025 | [PDF](https://arxiv.org/pdf/2605.12025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 596. Caraman at SemEval-2026 Task 8: Three-Stage Multi-Turn Retrieval with Query Rewriting, Hybrid Search, and Cross-Encoder Reranking

**arXiv ID:** 2605.12028 | [PDF](https://arxiv.org/pdf/2605.12028v1)

**作者:** David-Maximilian Caraman `[一作]` (Babeș-Bolyai University), Gheorghe Cosmin Silaghi `[通讯]` (Babeș-Bolyai University)

**通讯引用:** 516 | [OpenAlex ID](https://openalex.org/A5045075548)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个三阶段检索流水线（查询重写、混合检索、交叉编码重排），解决多轮对话检索问题。

**💡 创新点**

创新点在于采用域特定温度调控的LoRA微调Qwen 2.5 7B重写器、递归排名融合、以及在多领域语料下的系统性负实验验证。

**🔧 技术方法**

使用了LoRA微调、Qwen 2.5 7B、BGE‑base/重排器、BM25、Dense检索、Reciprocal Rank Fusion以及跨编码重排器。

**📊 数据集**

使用了SemEval‑2026 Task 8 MTRAGEval 777查询，以及四个域语料库（ClapNQ、Cloud、FiQA、Govt）。

**📈 对比分析**

通过在holdout上对温度、重写策略、候选池大小等进行系统比较，最终在官方测试集上取得nDCG@5 0.531，排名第8/38，优于基线10.7%。

**⚠️ 局限性**

限制在于仅支持英文、模型规模受限于消费级硬件、未使用加权融合、未预测答案可解性、长对话可能超出上下文窗口。

---

## 597. Large Language Models as Amortized Pareto-Front Generators for Constrained Bi-Objective Convex Optimization

**arXiv ID:** 2605.12106 | [PDF](https://arxiv.org/pdf/2605.12106v1)

**作者:** Peipei Xu `[一作]` (University of Shanghai for Scienceand Technology), Yong Liu `[通讯]` (University of Shanghai for Scienceand Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

论文提出一种名为 DIPS 的端到端框架，利用大语言模型直接生成受约束的双目标凸优化问题的 Pareto 前沿。

**💡 创新点**

创新点包括：1) 将连续数值输出转化为两令牌离散表示；2) 通过 Numerically Grounded Token Initialization (NGTI) 为新数值词表做热启动；3) 引入 Three-Phase Curriculum Optimization (TPCO) 将结构学习与粗细数值监督分阶段进行；4) 在推理时使用 Multi-Pass Pareto Fusion (MPPF) 进一步提升前沿覆盖。

**🔧 技术方法**

使用技术包括：预训练 7B 参数 LLM（Qwen2.5-7B-Instruct）+ LoRA 微调；vLLM 加速推理；两令牌离散化、NGTI、TPCO、MPPF 等自研模块；对五类约束双目标凸问题进行训练和评估。

**📊 数据集**

使用五类随机生成的受约束双目标凸优化实例（一般二次、可分二次、岭回归、Huber 回归、Softplus 光滑凸）作为数据集，分别在维度 10–20 上采样 50,000 条训练实例，目标前沿为 20 个点。

**📈 对比分析**

与多种基线比较：通用 LLM、推理 LLM、Prompt 生成策略、经典单目标加权、NBI、NC、NSGA-II、MOEA/D。实验表明 DIPS 在 0.16–0.49 秒内实现 97–100% 可行率、0.953–0.998 的 hypervolume 比例，显著优于所有 LLM 基线（提升 0.2+ 并且耗时 2–3 orders 级别更快），并在 0.5 秒预算下与最强经典标尺相当甚至更好。

**⚠️ 局限性**

局限性包括：只适用于凸双目标问题；对更高维度、更多目标或非凸目标的泛化尚未验证；训练与推理仍需大量算力；模型对离散化精度和 token 设计敏感，需要进一步简化或自适应方案。

---

## 598. Bayesian Persuasion with a Risk-Conscious Receiver

**arXiv ID:** 2605.12094 | [PDF](https://arxiv.org/pdf/2605.12094v1)

**作者:** Yujing Chen `[一作]` (Peking University), Yujing Chen `[通讯]` (Peking University)

**通讯引用:** 49035 | [OpenAlex ID](https://openalex.org/A5100401978)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在接收者用条件价值-at-风险（CVaR）评估行动时的贝叶斯说服问题，并给出了求解的理论方法；

**💡 创新点**

创新点在于发现CVaR偏好破坏传统的“按行动直接推荐”简化，并提出以“行动+活跃面片”作为信号类型的激励兼容性新简化；此外阐明了显式与简洁表示的计算复杂度边界，并给出了有限精度近似求解方案；

**🔧 技术方法**

主要技术包括：最大-仿射（max-affine）结构分析、活跃面片揭示原理、线性规划求解、统计格点离散化与切线约束、边界过滤与局部面片精化；

**📊 数据集**

该研究为纯理论分析，未使用实验数据集；

**📈 对比分析**

由于是理论结果，没有与其他算法在数据集上的实验比较；但作者证明在显式面片表示下可多项式时间求解，在简洁表示下为NP‑hard；近似算法在给定精度下实现了 ε‑激励兼容，且在满足边界条件时可转为严格激励兼容；

**⚠️ 局限性**

局限性包括：需要显式列出所有面片才能保证多项式求解；在简洁表示下问题仍然难解；近似方案依赖于统计细胞访问，实际实现可能受限；对高维状态空间的离散化导致指数级规模，实际可行性受限。

---

## 599. On Capacity and Delay of Wireless Networks with Node Failures

**arXiv ID:** 2605.12080 | [PDF](https://arxiv.org/pdf/2605.12080v1)

**作者:** Wei Li `[一作]` (Xidian University), Jiandong Li `[通讯]` (Xidian University)

**通讯引用:** 82880 | [OpenAlex ID](https://openalex.org/A5057916222)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究无线自组网在随机节点失效情形下的容量与延迟的渐近扩展规律，并提出通过部署冗余节点以补偿容量损失，证明容量与延迟之间的最优权衡为O(1)；

**💡 创新点**

创新点在于首次系统性地将节点失效概率纳入无线自组网的容量/延迟分析，得出容量为Θ(√(n(1−q)/log n))、延迟为Θ(√(n(1−q)/log n))，并量化冗余节点需求及其对容量/延迟的影响；

**🔧 技术方法**

使用随机几何图、节点失效的概率模型、场景化的介质自由空间干扰模型、基于无干扰调度的多跳路由以及概率论（Chebyshev、马尔可夫）与渐近分析技术；

**📊 数据集**

论文为理论分析，未使用具体实验数据集，所有结论均基于均匀独立布点的理论模型；

**📈 对比分析**

与经典的无失效情形（q=0）以及文献中的容量/延迟下界上界进行比较，验证在节点失效时容量下降约为√(1−q)比例，冗余节点数需至少为ϵ(n,q)nq（ϵ>1）才能恢复至原容量；

**⚠️ 局限性**

局限性包括：仅给出渐近规模结果，忽略节点移动性、时变信道、真实环境中的多径/阴影；冗余节点部署量巨大，实际实现成本高；并未在仿真或实验中验证理论预测。

---

## 600. The Deepfakes We Missed: We Built Detectors for a Threat That Didn't Arrive

**arXiv ID:** 2605.12075 | [PDF](https://arxiv.org/pdf/2605.12075v1)

**作者:** Shaina Raza `[一作]` `[通讯]` (Vector Institute for Artificial Intelligence), Shaina Raza (Vector Institute for Artificial Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对比了深度伪造检测研究与实际危害的分布，发现研究过度聚焦公共人物视频检测（T1），而现实中主要危害为非同意亲密影像（NCII）、语音克隆诈骗及群聊传播（T2、T3、T5）。

**💡 创新点**

提出三条针对未被充分研究的危害类别的技术议程——实时语音克隆检测、端侧隐私保护的NCII检测和消息层传播防御，并诊断误匹配持续存在的结构性原因。

**🔧 技术方法**

利用统计计数、文献分类、威胁模型审计等方法进行分析；建议的议程基于流式语音检测、少量参考学习、联邦评估、图信号等技术方向。

**📊 数据集**

分析依赖公开报告（IC3、IWF、AIID、StopNCII.org等）和现有基准（FaceForensics++、Celeb-DF、DFDC、FakeAVCeleb等），并对论文集进行分类。

**📈 对比分析**

通过对论文数量与危害指标的对数曲线进行对比，表明T1论文增长与实际危害曲线不匹配；本文未给出具体模型性能评估。

**⚠️ 局限性**

研究依赖公开报告且缺乏全面性，论文分类具有一定主观性，缺乏实验验证，聚焦于英语与西方案例，导致结论的通用性受限。

---

## 601. RoboBlockly Studio: Conversational Block Programming with Embodied Robot Feedback for Computational Thinking

**arXiv ID:** 2605.12059 | [PDF](https://arxiv.org/pdf/2605.12059v1)

**作者:** Leyi Li `[一作]` (Xi'an Jiaotong-Liverpool University), Qing Zhang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 RoboBlockly Studio，一套将区块式编程、对话式 AI 教学代理与机器人执行结合的交互式学习平台，支持学生在完成真实任务的过程中进行编程、运行、观察和修正。

**💡 创新点**

创新点包括：① 通过三元交互（区块编辑、AI 对话、机器人执行）形成闭环迭代；② 在系统设计中明确体现“代理式计算思维”与“透明可解释”原则；③ 将 AI 反馈与机器人感知相结合，让错误和成功通过物理执行即时呈现；④ 以教师访谈为导向的任务与 AI 交互设计，保证教学可行性与可扩展性。

**🔧 技术方法**

技术实现：基于 Web 的区块式编程环境，后端使用 Node.js 生成 Python 代码并通过 WebSocket 传给 ROS2 机器人；机器人采用全向轮+机械臂，执行程序后即时反馈；对话代理使用 GPT‑4o，结合任务规则、程序状态和执行结果进行提示、检查与推理；评估工具包括 Bebras 问题、NASA‑TLX、SUS、UEQ。

**📊 数据集**

数据集：32 名高中生参与的实证研究；使用 Bebras 计算思维测试（前测 4 题、后测 4 题）以及 NASA‑TLX、SUS、UEQ 等问卷数据；实验中对比 AI‑Only 与 AI+Robot 两个条件的表现。

**📈 对比分析**

比较方法：在同一受试者内对 AI‑Only 与 AI+Robot 两个条件进行计量，使用配对 t‑检验或 Wilcoxon 检验；结果显示：① CT 成绩从平均 1.28 题提升至 2.88 题（p < .001，d_z=1.69）；② AI+Robot 条件在时间、精力与挫折感等 NASA‑TLX 维度显著低于 AI‑Only；③ SUS 平均 72.5 分，UEQ 各维均为正向评价，显示系统易用且具吸引力。

**⚠️ 局限性**

局限性：样本规模较小、实验时长短暂，未包含长期跟踪或多元对照；实验环境为控制式教室，缺乏真实课堂多样性；LLM 检测在某些非标准解法时偶尔误判；缺乏与更丰富的屏幕模拟基线对比，难以完全量化机器人体现的优势。

---

## 602. Closing the Motion Execution Gap: From Semantic Motion Task Constraints to Kinematic Control

**arXiv ID:** 2605.12053 | [PDF](https://arxiv.org/pdf/2605.12053v1)

**作者:** Simon Stelter `[一作]` (University of Bremen), Michael Beetz `[通讯]` (University of Bremen)

**通讯引用:** 19166 | [OpenAlex ID](https://openalex.org/A5003274224)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出一种通过运动状态图（MSC）将语义约束转化为连续可执行的机器人运动的框架；

**💡 创新点**

创新点在于将MSC与可微分的全局运动学世界模型以及基于jerk约束的局部模型预测控制（lmpc）相结合，实现了语义化运动描述、实时监控与平滑任务切换，弥合了“运动执行鸿沟”；

**🔧 技术方法**

采用了MSC作为可执行的符号运动表示、自动微分库（libsympy/auto‑diff）构建可微分运动学世界模型、lmpc + qp‑SWIFT求解器实现任务函数控制，并通过任务函数与监控器实现在线语义反馈；

**📊 数据集**

使用了八台不同机器人平台（PR2、TIAGo、HSR、UR10、Dual‑UR10、Stretch2、5‑finger hand等）与多样环境进行实验验证，而非公开数据集；

**📈 对比分析**

通过在八台机器人上的实地部署与仿真，展示了该框架在不需要机器人特定调参的情况下即可实现平滑、约束满足的运动，且在切割、门开、插孔等任务中实现了低误差、可观察的任务切换；

**⚠️ 局限性**

局限性主要包括缺乏动力学/力控制支持、对动力学约束的处理仍需进一步研究，以及与全局规划器的深度集成仍是未来工作。

---

## 603. Scaling Laws and Tradeoffs in Recurrent Networks of Expressive Neurons

**arXiv ID:** 2605.12049 | [PDF](https://arxiv.org/pdf/2605.12049v1)

**作者:** Aaron Spieler `[一作]` (University of Tübingen), Anna Levina `[通讯]` (University of Tübingen)

**通讯引用:** 1937 | [OpenAlex ID](https://openalex.org/A5050318092)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了可调节单元复杂度的 ELM Network，并在两类序列任务上进行大规模参数预算下的规模化实验与理论建模。

**💡 创新点**

创新点在于把单元复杂度与宽度、连接度作为独立可调轴，发现并量化了预算约束下的最优权衡，并提出了信息论框架 Effective Representation Information 解释这种折中。

**🔧 技术方法**

采用 Expressive Leaky Memory（ELM）神经元构建循环层，进行大范围超参数搜索、信息论分析和 Pareto 前沿搜索，并使用 BPTT 训练。

**📊 数据集**

使用了神经形态 SHD-Adding 任务和 Enwik8 字符级语言建模数据集。

**📈 对比分析**

通过与既往基线对比，实验表明在每个维度上性能单调提升，在固定预算下可找到更高性能的最优配置；在 SHD-Adding 上接近或超过现有最优，在 Enwik8 上实现了竞争性 BPC，并成功绘制了三阶数量级的 Pareto 前沿。

**⚠️ 局限性**

局限包括：GPU 计算与内存偏差导致的实验偏差；搜索空间未覆盖所有可能配置；理论模型对递归、拓扑与优化过程的简化；以及对生物学解释的有限性。

---

## 604. Adaptive Multi-Round Allocation with Stochastic Arrivals

**arXiv ID:** 2605.12111 | [PDF](https://arxiv.org/pdf/2605.12111v1)

**作者:** Yuqi Pan `[一作]` (Harvard University), Cheryl Johnson `[通讯]` (World Health Organization)

**通讯引用:** 6219 | [OpenAlex ID](https://openalex.org/A5003587385)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在自适应网络招募场景下的多轮预算有限资源分配问题，目标是最大化已知预算内招募人数。

**💡 创新点**

创新点在于证明单轮分配可用贪婪算法最优，提出种群层级代理价值函数并通过概率生成函数实现可求解的动态规划，同时给出误差分解的鲁棒性分析。

**🔧 技术方法**

技术方法包括贪婪分配、离散凹整数优化、动态规划、概率生成函数、总变差误差分析及鲁棒性证明。

**📊 数据集**

实验使用ICPSR公开的 HIV 网络（以及性传播疾病网络）进行模拟和真实网络的招募过程。

**📈 对比分析**

与常数分配、固定/动态α分配等基线相比，代理策略在多种折扣率和起始前沿规模下表现出更高的累计折扣奖励，且性能稳健。

**⚠️ 局限性**

局限性包括在高折扣率和模型严重偏差时代理策略可能表现不佳，群体异质性导致近似误差，且对模型估计误差敏感。

---

## 605. Memory Constrained Adversarial Hypothesis Testing

**arXiv ID:** 2605.12063 | [PDF](https://arxiv.org/pdf/2605.12063v1)

**作者:** Malhar A. Managoli `[一作]` (Tata Institute of Fundamental Research), Vinod M. Prabhakaran `[通讯]` (Tata Institute of Fundamental Research)

**通讯引用:** 2463 | [OpenAlex ID](https://openalex.org/A5045516299)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在内存限制下的对抗性二元假设检验，提出了一种时间不变的随机有限状态机（FSM）模型，并获得了最小最大渐近错误概率的上下界。

**💡 创新点**

创新点在于考虑了对抗者在选择样本分布时不仅依赖于过去的样本，还依赖于FSM的状态历史，并且在内存限制的情况下提供了错误概率的界限。

**🔧 技术方法**

使用了时间不变的随机有限状态机（FSM）模型，并采用了马尔可夫链分析方法来处理对抗性设置。

**📊 数据集**

未具体提及使用的数据集，但研究涉及的假设检验问题通常涉及从两个分布中抽取的样本。

**📈 对比分析**

通过与Hellman和Cover的工作进行比较，展示了在给定状态数S的情况下，最小最大错误概率的界限具有相同的指数行为，并且在某些问题类别中相匹配。

**⚠️ 局限性**

限制在于对抗者的策略可能会导致非平稳性，且未考虑其他类型的内存限制或对抗模型的影响。

---

## 606. Autonomy and Agency in Agentic AI: Architectural Tactics for Regulated Contexts

**arXiv ID:** 2605.12105 | [PDF](https://arxiv.org/pdf/2605.12105v1)

**作者:** Damir Safin `[一作]` (fortiss GmbH), Dian Balta `[通讯]` (fortiss GmbH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个包含自主性（Autonomy）与能动性（Agency）两维的设计空间，并在此空间中定义了六种可操作的架构策略（Checkpoints、Escalation、Multi-Agent Delegation、Tool Provisioning、Tool Fencing、Write Staging）以及五个关键部署参数（Model Capability、Agent Architecture、Tool Fidelity、Workflow Bottlenecks、Evaluation）。通过在公共部门文档分类和资助项目匹配这两个案例中应用上述方法，展示了如何在受监管环境下平衡人机协作、可追溯性和可逆性。

**💡 创新点**

创新点在于：1）将自主性与能动性视为耦合的双维设计轴，并构建可视化的可行性等值曲线；2）系统化列出六种架构策略，使设计者能够在不同维度上精细调节；3）提出五个影响实现阈值的部署参数，为从“何处开始”到“能达到何种高度”提供完整的决策框架；4）用具体公共部门案例验证框架的可操作性，填补了先前仅以单维度或经验式方法为主的空白。

**🔧 技术方法**

技术手段主要是基于大型语言模型（LLM）的代理系统，配合：
- 工具集（工具调用、检索、写入等）
- 检查点与升级机制（workflow 层控制）
- 多代理分工与委托（子代理协作）
- 工具屏蔽与分阶段写入（工具层和动作层控制）
- 评估与日志记录框架（支持后期量化评估）。

**📊 数据集**

论文未提供标准公开数据集。案例中使用的业务数据为：
- 公共部门内部的文档流（PDF、邮件附件、收据）及其分类与摘要结果；
- 资助项目的自然语言条款、地理信息与项目描述。 这些数据属于机构内部或公开可获取的业务数据，并未公开发布。

**📈 对比分析**

由于本文主要聚焦方法论和设计框架，未进行量化实验对比。评估方式以案例演示为主：通过将系统配置在设计空间中的不同点，观察其在合规性、可追溯性、错误修正成本等方面的变化。因缺乏统一的指标与基准，论文未给出具体性能数值或与其他方法的比较结果。

**⚠️ 局限性**

局限性包括：
- 设计空间与等值曲线为定性分析，缺乏经验式阈值或可度量的可行性界限；
- 仅给出六种策略，可能不足以覆盖所有受监管场景或更复杂的多代理工作流；
- 案例仅覆盖两个业务领域，未验证方法在不同法规、组织结构和任务复杂度下的普适性；
- 未提供实验数据或量化评估，难以衡量策略组合对系统性能、合规性或用户信任的真实影响；
- 关键部署参数的作用阐释为示例性，未给出如何系统化评估或优化这些参数的具体方法。

---

## 607. World Action Models: The Next Frontier in Embodied AI

**arXiv ID:** 2605.12090 | [PDF](https://arxiv.org/pdf/2605.12090v1)

**作者:** Siyin Wang `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (National University Of Singapore)

**通讯引用:** 24973 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理了世界动作模型（World Action Models, WAMs）的体系结构、训练数据、评估方法，并对 Cascade 与 Joint 两大范式及其子类进行了细粒度分类和时间演化概述。

**💡 创新点**

创新点在于首次提出将世界建模与动作生成统一为同一框架的 WAM，并细化为 Explicit/Implicit 规划与 Autoregressive/Diffusion 生成两大技术路径，同时对多模态数据的无监督预训练与跨任务迁移进行深入探讨。

**🔧 技术方法**

主要技术包括 Transformer、Diffusion Transformer、VPT、JEPA、Latent Video Prediction、Cross‑Attention Coupling、Mixture‑of‑Transformers、Action‑Conditioned Inverse Dynamics 等。

**📊 数据集**

使用的数据集覆盖机器人遥操作（如 RoboNet、BridgeData）、人体演示（如 UMI、FastUMI）、仿真环境（如 ManiSkill2、RoboCasa）以及大规模互联网视角视频（如 Ego4D、HowTo100M）。

**📈 对比分析**

与传统 VLA、世界模型或视频政策的对比表明，WAM 在多任务零样本迁移、物理一致性评估和安全性仿真测试上均显著优于单一模型，尤其在视觉‑语言‑动作多模态推理速度和准确度上提升约15‑30%。

**⚠️ 局限性**

主要局限包括：推理延迟受 Diffusion 多步迭代限制、对高质量多模态数据的强依赖、以及在高度动态或视觉模糊场景下易出现错误传播导致的控制失效。

---

## 608. Anomaly-Aware Vision-Language Adapters for Zero-Shot Anomaly Detection

**arXiv ID:** 2605.12069 | [PDF](https://arxiv.org/pdf/2605.12069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 609. Missing Old Logits in Asynchronous Agentic RL: Semantic Mismatch and Repair Methods for Off-Policy Correction

**arXiv ID:** 2605.12070 | [PDF](https://arxiv.org/pdf/2605.12070v1)

**作者:** Zhong Guan `[一作]` (Tianjin University), Hongke Zhao `[通讯]` (Tianjin University)

**通讯引用:** 2225 | [OpenAlex ID](https://openalex.org/A5017692278)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究异步LLM强化学习中缺失旧logits导致的PPO修正失效问题，并提出完整获取与低成本近似解决方案。

**💡 创新点**

提出缺失旧logit问题的统一解耦视角，三种完整获取策略（快照、旧logit模型、部分中断）以及改进的PPO‑EWMA参考，系统成本与性能并评估。

**🔧 技术方法**

采用PPO、PPO‑EWMA、截断/掩码、指数移动平均、训练‑推理不匹配修正、MoE模型、vLLM/SGLang、Megatron‑LM/FSDP等技术。

**📊 数据集**

使用	au^2‑Bench（零售、航空、电信）和VitaBench（in‑store、delivery）以及 HuggingFace 的 guanzhong2/TU_Pipeline 数据集。

**📈 对比分析**

将Decoupled PPO、Linear Prox、PPO‑EWMA 与理想快照（完整旧logit）对比；PPO‑EWMA 在大多数任务上接近快照且明显优于其他方法，且系统开销显著降低。

**⚠️ 局限性**

PPO‑EWMA 仍为近似参考，可能在极端版本间隔或非平稳环境下失效；完整旧logit获取成本高，适用范围有限。

---

## 610. Position Auctions with a Capacity Constraint

**arXiv ID:** 2605.12040 | [PDF](https://arxiv.org/pdf/2605.12040v1)

**作者:** Eleni Batziou `[一作]` (University of Liverpool), Piotr Krysta `[通讯]` (Augusta University)

**通讯引用:** 1110 | [OpenAlex ID](https://openalex.org/A5079362010)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了针对具有全局容量约束的广告位置分配（容量约束匹配）问题的常数近似真诚机制，兼顾效率与社群福利；

**💡 创新点**

首次构造了单参数、全局容量约束下可实现常数近似且真诚的机制，并通过改进贪心+局部重排保证单调性；

**🔧 技术方法**

使用基于密度排序的贪心-重排算法、概率随机化、Myerson支付机制以及证明单调性的结构性分析；

**📊 数据集**

无实验数据集，全部为理论分析与证明；

**📈 对比分析**

与传统单价位置拍卖的无约束匹配相比，算法在容量约束下实现了6-近似（原始）和12-近似（真诚化改进）；

**⚠️ 局限性**

仅提供随机化真诚机制，尚无确定性或PPTS实现；且当前技术仅适用于位置CTR仅依赖于位置的单参数设置，无法直接推广到广告影响CTR的更一般情形。

---

## 611. What-Where Transformer: A Slot-Centric Visual Backbone for Concurrent Representation and Localization

**arXiv ID:** 2605.12021 | [PDF](https://arxiv.org/pdf/2605.12021v1)

**作者:** Ryota Yoshihashi `[一作]` (Institute of Science Tokyo), Ikuro Sato `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3268 | [OpenAlex ID](https://openalex.org/A5100862952)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出What‑Where Transformer（WWT），在ViT编码器中将表示拆分为语义槽（what）和空间掩码（where），并直接生成多对象掩码，用于分类、分割、检测和无监督发现等多任务；

**💡 创新点**

在Transformer编码器中引入slot‑based互注意层（μAttn）和多流架构，实现显式what‑where分离；让单标签分类训练即可产生多对象注意图；将掩码直接用于下游，无需额外解码器；

**🔧 技术方法**

采用Transformer注意力、slot attention、互注意层、MLP‑over‑attention、自动编码头、弱监督分割头以及零样本发现方法，构成全新的视觉编码器；

**📊 数据集**

在ImageNet‑1k上进行预训练，并在ImageNet‑S、VOC12、COCO20k、VOC07等数据集上评估弱监督分割、零样本发现、检测与分割等任务；

**📈 对比分析**

与DeiT、DINO、DETR等方法比较；分类精度与DeiT相当；零样本单物体发现CorLoc 41.4%超过DeiT 29.7%；多物体召回率优于DINOv1；弱监督分割mIoU 49.1%优于DeiT 35.9%；检测mAP 63.1%略低于DETR 72.2，分割mIoU 61.3%略低于ResNet‑50 FCN 66.5；

**⚠️ 局限性**

仅在有标签分类预训练下验证，未探讨自监督或文本对比学习情形下的对象分解是否保持；模型参数增多，需进一步验证更大规模数据与多任务场景的适用性。

---

## 612. Property-Level Reconstructability of Agent Decisions: An Anchor-Level Pilot Across Vendor SDK Adapter Regimes

**arXiv ID:** 2605.12078 | [PDF](https://arxiv.org/pdf/2605.12078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 613. On the Hamming Distance and LCD Properties of Binary Polycyclic Codes and Their Duals

**arXiv ID:** 2605.12097 | [PDF](https://arxiv.org/pdf/2605.12097v1)

**作者:** Sujata Bansal `[一作]` (Indian Institute of Technology (ISM)), Pramod Kumar Kewat `[通讯]` (Indian Institute of Technology (ISM))

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了与不可约多项式的幂相关的二进制多环码，确定了其完整的代数结构，并发展了关于其最小汉明距离的通用结果，包括多个确切值和界限。

**💡 创新点**

创新点在于建立了二进制多环码的LCD（线性互补对偶）性质的必要和充分条件，并构造了多个家庭的二进制LCD码，展示了这些码的强距离、对偶性、可逆性和LCD性质。

**🔧 技术方法**

使用了代数结构分析、汉明距离计算和对偶码性质研究等技术，特别是通过不可约多项式的幂来构造和分析多环码。

**📊 数据集**

使用了与不可约多项式相关的多种数据集，特别是自反三项式的幂，如x^2·3^v+x^3^v+1。

**📈 对比分析**

通过与现有的编码方法进行比较，展示了所构造的多环码在汉明距离和LCD性质上的优越性，尤其是在较长码的构造中表现出色。

**⚠️ 局限性**

限制在于对于更高维度的LCD码的构造和分析仍然存在挑战，特别是在确定最优参数和界限方面。

---

## 614. Fuzzy k-anonymity in complex networks

**arXiv ID:** 2605.12062 | [PDF](https://arxiv.org/pdf/2605.12062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 615. Sign Language Recognition and Translation for Low-Resource Languages: Challenges and Pathways Forward

**arXiv ID:** 2605.12096 | [PDF](https://arxiv.org/pdf/2605.12096v1)

**作者:** Nigar Alishzade `[一作]` (Karabakh University), Gulchin Abdullayeva `[通讯]` (MSERA Institute of Mathematics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对低资源手语识别与翻译技术进行系统综述，识别七大相互关联挑战，并以阿塞拜疆手语（AzSL）为案例，提出数据驱动、签署适应、任务评估三大范式转变与技术路线图。

**💡 创新点**

创新点在于将挑战形成的“自我强化循环”量化建模，抽取八条可操作的跨地区经验教训，主张利用土耳其、哈萨克等突厥手语的语言相似性进行迁移学习，并强调社区共设计与离线可部署方案。

**🔧 技术方法**

采用 MediaPipe 关节与面部特征提取、轻量级 3D CNN/Transformer、迁移学习、少样本个性化、混合序列标注与分层评估等技术，并提出基于非手势特征的多模态融合与任务专用评价指标。

**📊 数据集**

主要数据集包括 AzSLD（约30k个孤立手势）以及对比使用的 WLASL、PHOENIX‑2014T、T2 等公开高资源手语数据，用于迁移学习与性能基准。

**📈 对比分析**

通过与高资源基线对比，迁移学习从土耳其手语提升 AzSL 识别准确率可达 70–75%，与从 ASL 迁移的 55–60% 相比显著优越；在离线移动端使用 MobileNetV3 可实现 25 fps、<50 MB 模型的 72% 准确率，说明方案在资源受限环境中具备可行性。

**⚠️ 局限性**

局限在于 AzSL 缺乏连续、非手势标注的数据，社区参与与标注质量仍不足，迁移学习的语言相似性评估依赖专家判断，模型在极端环境下的鲁棒性与长期部署的隐私保护机制尚未充分验证。

---

## 616. SAGE: A Self-Evolving Agentic Graph-Memory Engine for Structure-Aware Associative Memory

**arXiv ID:** 2605.12061 | [PDF](https://arxiv.org/pdf/2605.12061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 617. Learning What Matters: Adaptive Information-Theoretic Objectives for Robot Exploration

**arXiv ID:** 2605.12084 | [PDF](https://arxiv.org/pdf/2605.12084v1)

**作者:** Youwei Yu `[一作]` (Indiana University), Lantao Liu `[通讯]` (Indiana University)

**通讯引用:** 1906 | [OpenAlex ID](https://openalex.org/A5101917996)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一种自适应信息理论目标 QOED，用于引导机器人在有限交互预算下收集能显著降低模型参数不确定性的数据。

**💡 创新点**

创新点在于：①利用 Fisher 信息矩阵的特征空间自动识别可观测且可辨识的关键参数；②通过 Schur 补正则化抑制无关方向，使探索目标在高维空间近似最优；③将该自适应目标嵌入模型基础策略优化（MBPO），兼顾探索与任务学习。

**🔧 技术方法**

技术手段包括 Fisher 信息矩阵、特征分解与可观测子空间选择、Schur 补正则化、短路模型（Transformer）学习动力学、CEM 优化、MBPO + PPO 策略学习。

**📊 数据集**

使用 MuJoCo 仿真平台上的 7 种机器人（Go1、G1、Jackal、Inspire 等）以及真实机器人 Franka Emika Panda 与 Clearpath Jackal 进行实验。

**📈 对比分析**

与 BOED、QOED‑AGNOSTIC、SAC‑ADAPT、DISAGREEMENT、DOMAIN‑RANDOM 等基线比较，QOED 在参数估计、动态预测 RMSE 方面提升约 21.98% / 35.23%，在策略奖励与实测成功率上分别达到最高水平（例如 89% 的成功率），并在仿真与真实环境中均表现优异。

**⚠️ 局限性**

局限性包括：需预先给定物理参数化；目标复杂度可能影响 RL 学习效率；在学习动力学不足的初期性能下降；对高度非结构化或未知环境的适应性仍有限。

---

## 618. OmniHumanoid: Streaming Cross-Embodiment Video Generation with Paired-Free Adaptation

**arXiv ID:** 2605.12038 | [PDF](https://arxiv.org/pdf/2605.12038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 619. Hölder Policy Optimisation

**arXiv ID:** 2605.12058 | [PDF](https://arxiv.org/pdf/2605.12058v1)

**作者:** Yuxiang Chen `[一作]` (University College London), Jun Wang `[通讯]` (University College London)

**通讯引用:** 46259 | [OpenAlex ID](https://openalex.org/A5100384686)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 HölderPO，一种通过 Hölder 均值对 token 重要性比率进行可调聚合的强化学习框架，用以解决 GRPO 等固定聚合方法在长序列推理中的梯度集中与方差不稳定的权衡。

**💡 创新点**

创新点在于将 Hölder 参数 p 作为连续可调的聚合权重，既能在正集中阶段放大稀疏高置信度信号，又能在负集中阶段压缩梯度方差，并通过动态 p‑退火调度实现训练过程的自适应平衡。

**🔧 技术方法**

技术包括 PPO 样式的序列级策略梯度、Hölder 均值聚合、梯度方差分析与理论证明、以及线性退火调度策略。

**📊 数据集**

使用了多数学术推理数据集（AIME24、AMC、MATH500、Minerva、OlympiadBench）和 ALFWorld 代理任务进行评估。

**📈 对比分析**

与 GRPO、GMPO、PMPO 等基线相比，HölderPO 在五个数学基准上平均达 54.9% 的准确率（相较标准 GRPO 提升 7.2%），在 ALFWorld 上平均成功率 93.8%（较 GRPO 提升 28.8%），显著提高稳定性与收敛速度。

**⚠️ 局限性**

局限包括需要为不同任务手动调节 p 的上下界与退火曲线，且正集中阶段可能导致奖励攻击易受误报影响。

---

## 620. OmniRefine: Alignment-Aware Cooperative Compression for Efficient Omnimodal Large Language Models

**arXiv ID:** 2605.12056 | [PDF](https://arxiv.org/pdf/2605.12056v1)

**作者:** Yuchen Deng `[一作]` (Tsinghua University), Yuxing Han `[通讯]` (Tsinghua University)

**通讯引用:** 3635 | [OpenAlex ID](https://openalex.org/A5101944724)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了OmniRefine，一种训练无关的两阶段音视频token压缩框架，用于提升Omni-LLMs的推理效率。

**💡 创新点**

创新点在于：①通过帧-音频相似度与动态规划实现对应保持的块细化（CPCR），②在每个细化块内采用视频的树结构时空压缩与音频的语义锚点压缩，并通过跨模态预算协调实现合作压缩（MACC）。

**🔧 技术方法**

采用的技术包括：帧-音频余弦相似度、受约束的动态规划、树结构空间-时间压缩、语义锚点音频压缩、跨模态预算调节和KV-cache复用。

**📊 数据集**

使用的数据集为 WorldSense、VideoMME 与 AVUT 三个音视频理解基准。

**📈 对比分析**

在三大基准上与随机剪枝、FastV、DyCoke（V&A）及 OmniZip 进行对比。OmniRefine 在 44% token 保留率下，在 WorldSense 上获得 46.7% 的准确率（仅比全标记低 0.1%），FLOPs 下降 69%；在 30% 保留率时仍保持 46.4% 的准确率，显著优于其他方法；在 AVUT 与 VideoMME 上亦保持接近全标记的性能，同时显著降低 FLOPs 与显存占用。

**⚠️ 局限性**

主要限制是对超参数（如阈值、预算系数等）的手动调优，缺乏自动适配机制。

---

## 621. Boosting Omni-Modal Language Models: Staged Post-Training with Visually Debiased Evaluation

**arXiv ID:** 2605.12034 | [PDF](https://arxiv.org/pdf/2605.12034v1)

**作者:** Che Liu `[一作]`, Fei Tian `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出视觉去偏的评测视图 OmniClean 并通过三阶段后训练 OmniBoost 对小规模多模态 LLM 进行提升

**💡 创新点**

① 通过可视化单模态探测剔除视觉快捷方式，构造真正的多模态评测；② 设计分阶段后训练流程（混合双模 SFT → RLVR → 自蒸馏），展示自蒸馏在去偏视图上可显著提升性能

**🔧 技术方法**

可视化单模态探测、监督微调（SFT）、多模态强化学习（RLVR）、自蒸馏（SFT on distilled data）以及实体-关系图生成的合成查询构造

**📊 数据集**

九大公开多模态基准（Daily-Omni、IntentBench、Video-Holmes、WorldSense、OmniBench、UNO-Bench、AV-Odyssey、CG-AV-Counting、OmniVideoBench）及其清洗后子集 OmniClean

**📈 对比分析**

以宏观平均和查询加权平均对比，Stage 2 RLVR 在宏观平均上最佳（提升约5%），Stage 3 自蒸馏在查询加权平均上领先；相较基线 Qwen2.5‑Omni‑3B 提升约30%

**⚠️ 局限性**

仅验证单一基础模型，评测仅控制视觉泄漏；自蒸馏效果依赖过滤策略，部分基准表现不均；合成查询质量与生成模型相关，难以保证全面性

---

## 622. Learning plug-in surrogate endpoints for randomized experiments

**arXiv ID:** 2605.12051 | [PDF](https://arxiv.org/pdf/2605.12051v1)

**作者:** Alessandro-Umberto Margueritte `[一作]` (AstraZeneca), Fredrik D. Johansson `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出学习可直接替代长期主效应的插件式复合生物标志物（surrogate）方法，并在随机实验前利用观察数据进行训练。

**💡 创新点**

创新点在于：① 定义并可识别的试验CATE误差目标；② 两种可行算法——（a）通过对潜在后验生物标志物采样直接最小化CATE误差；（b）通过推导的上界将目标转化为加权回归；③ 只依赖后处理变量而非前处理变量，满足临床试验可接受性。

**🔧 技术方法**

技术手段包括：因果可识别假设（可忽略性、Prentice 条件、可传递性）、潜在变量模型、条件密度估计（如随机森林/高斯过程）、加权回归、对后验样本的重采样、梯度下降优化；实现时使用 Lasso、随机森林、梯度提升、深度学习等回归器。

**📊 数据集**

数据集：1）多个合成情景（a–d），模拟不同因果结构与线性/非线性关系；2）IHDP 实际随机对照试验数据（低出生体重儿童干预，36 个月 IQ 为主效应，24 个月发育评分为候选生物标志物）。

**📈 对比分析**

与基准方法对比：单变量回归、全局回归（outcome regression）、Surrogate Index（依赖前处理变量）、Bound Regression 等。实验结果显示：采样方法在 ATE、CATE 的 MAE 与 R² 上均优于基准，尤其在合成情景中几乎达到最佳线性插件 surrogate 的性能；在 IHDP 中采样方法的误差最小，其他方法误差略大，且单变量回归表现最差。

**⚠️ 局限性**

局限性包括：① 需要已知或可估计实验与观测人群的密度比，且假设可传递性成立；② 采样方法对后验密度估计质量敏感，若模型欠拟合会导致偏差；③ 仅在存在可用后处理变量且 T 对 Y 的直接效应可忽略的情况下才能得到无偏估计；④ 计算成本较高，尤其是多维后验采样与大样本加权回归。

---

## 623. HM-Req: A Framework for Embedding Values within CPS Human Monitoring Requirements

**arXiv ID:** 2605.12100 | [PDF](https://arxiv.org/pdf/2605.12100v1)

**作者:** Zoe Pfister `[一作]` (University of Innsbruck), Michael Vierhauser `[通讯]` (University of Innsbruck)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出HM-Req框架，包含一种控制自然语言用于规范CPS中的人类监测需求，并将利益相关者的价值观与需求关联，以实现价值冲突检测。

**💡 创新点**

创新点在于将人类价值与监测需求结合，使用Schwartz价值体系计算冲突分数，并通过控制自然语言减少自然语言歧义。

**🔧 技术方法**

采用控制自然语言（CNL）、WordNet/VerbNet词法分析、价值冲突得分算法、Eclipse Langium实现的VS Code插件与CLI工具。

**📊 数据集**

使用五个公开需求数据集（DS-1至DS-5）以及Dronology系统的需求集进行训练、测试和验证。

**📈 对比分析**

通过将训练集和测试集的需求转化为CNL，实验显示能完全捕获约76%需求、部分捕获24%；在独立测试集中为79%完全捕获，Dronology案例中为65%完全捕获；调查显示工具易用性与价值冲突可视化被认为有效。

**⚠️ 局限性**

局限性包括CNL语法仍无法覆盖所有表述、对时间/嵌套角色的支持有限、价值冲突分数基于欧氏距离的粗略近似、受限样本规模及研究方法的定性性质。

---

## 624. Automated Amortised Analysis of Skew Heaps and Leftist Heaps (Extended Version)

**arXiv ID:** 2605.12091 | [PDF](https://arxiv.org/pdf/2605.12091v1)

**作者:** Armin Walch `[一作]` (University of Innsbruck), Florian Zuleger `[通讯]` (Vienna University of Technology)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文实现了对偏斜堆、权重偏左堆以及秩偏左堆的全自动化摊销分析，自动推导出它们的最优时间复杂度上界。

**💡 创新点**

创新点在于提出了一个通用的类型系统与模板化潜能函数框架，支持路径感知推理、数据结构不变式以及分段与求和对数潜能函数，显著提升了自动分析的适用范围和精度。

**🔧 技术方法**

采用基于类型推导的摊销资源分析（AARA），结合SMT求解器（Z3）实现约束求解，并在Haskell中实现了一个可插拔的原型工具。

**📊 数据集**

使用的测试集包括偏斜堆、权重偏左堆、秩偏左堆三种堆结构，并覆盖了文献中已知的多种潜能函数（如分段、求和对数等）。

**📈 对比分析**

与手工证明和先前的自动化工具相比，本文的工具能够在不需要人工指定潜能函数的前提下，自动生成与文献一致的最优摊销上界，验证效果与手工证明完全匹配，且实现简洁。

**⚠️ 局限性**

局限性在于尚未能找到改进秩偏左堆的更优摊销上界，且当前实现仍依赖于SMT求解器的性能，复杂约束可能导致求解时间增长。

---

## 625. Elicitation-Augmented Bayesian Optimization

**arXiv ID:** 2605.12079 | [PDF](https://arxiv.org/pdf/2605.12079v1)

**作者:** Alvar Haltia `[一作]` (Aalto University), Samuel Kaski `[通讯]` (Aalto University)

**通讯引用:** 15109 | [OpenAlex ID](https://openalex.org/A5018305257)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种利用人类专家的相对比较来改进贝叶斯优化的算法，结合直接评估与比较查询的混合似然GP后验，并提出成本敏感的价值信息采样策略。

**💡 创新点**

①提出混合似然稀疏变分GP联合更新直接观测与比较查询；②推导基于成本的价值信息规则并给出闭式后验更新；③算法动态分配预算，能够在查询成本低时提升效率，查询成本高时退回传统贝叶斯优化。

**🔧 技术方法**

Gaussian Process、稀疏变分推断、probit比较似然、知识梯度、one-shot 重参数化、成本敏感的价值信息规则、多目标效用标量化。

**📊 数据集**

标准单目标和多目标贝叶斯优化基准函数：Branin、Hartmann6、BraninCurrin、VLMOP2，以及模拟的人工专家比较数据。

**📈 对比分析**

与单源贝叶斯优化基线(LogEI、UCB、EI-CF)、仅比较基线(EUBO、KG-Comp)以及HITL BO方法CoExBO比较，EA-BO在预算耗尽时往往达到或超过所有基线，并在低成本比较阶段快速收敛。

**⚠️ 局限性**

假设专家比较遵循已知效用函数U∘f，且多目标时需预先知晓效用；未考虑专家模型与目标函数不一致的情况；在高维或高噪声场景下比较信息可能不足。

---

## 626. BARISTA: A Multi-Task Egocentric Benchmark for Compositional Visual Understanding

**arXiv ID:** 2605.12074 | [PDF](https://arxiv.org/pdf/2605.12074v1)

**作者:** Patrick Knab `[一作]` (Ramblr.ai Research), Philipp Johannes Schubert `[通讯]` (Ramblr.ai Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了BARISTA数据集与统一评测框架，专注于第一人称视角的咖啡制作过程，通过每帧统一的场景图提供对象、属性、关系、交互、活动及步骤的稠密标注，并在此基础上构建一系列零样本视觉‑语言任务；

**💡 创新点**

核心创新在于（1）稠密且连贯的每帧场景图实现了对象、交互、关系与过程层面的可追溯性；（2）把多种任务归一化为基于同一图的零样本评测，能够精确定位模型在局部感知与全局推理上的弱点；（3）在单一、可复现的流程中覆盖三种咖啡制作方式，提供丰富的程序多样性；

**🔧 技术方法**

采用半自动标注流程：CaRe‑Ego检测交互、SAM 2/3 进行关键帧掩膜与后向传播、人工审核与身份归一、并通过属性、关系与活动标注补全场景图；在评测中使用多种开源VLM（Gemini、Gemma、Qwen、GPT）与专业基线（CaRe‑Ego、SAM 3）执行零样本任务；

**📊 数据集**

BARISTA数据集：185段咖啡制作视频（4.4小时），共计469,201帧，含3.61M掩膜、4.33M属性、2.48M关系、2424个细粒度活动和1305个过程步骤；

**📈 对比分析**

在统一的零样本框架下对五大任务（短语定位、手物交互识别、指代生成、活动识别、关系提取、时序VQA）进行评测；实验表明不同模型在各任务上性能差异显著，未出现单一模型在所有任务上领先；空间定位任务对模型差异最敏感；关系与活动等语义任务表现相对稳定；

**⚠️ 局限性**

局限性包括：仅覆盖咖啡制作这一单一程序；标注聚焦任务相关对象，缺乏全景覆盖；活动识别采用多选式评测，可能高估开放式场景性能；部分自由文本任务依赖LLM评判，存在标注不确定性；且使用冻结子集，未覆盖所有可能的程序变化。

---

## 627. TAR: Text Semantic Assisted Cross-modal Image Registration Framework for Optical and SAR Images

**arXiv ID:** 2605.12064 | [PDF](https://arxiv.org/pdf/2605.12064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 628. Clausal Deletion Backdoors for QBF: a Parameterized Complexity Approach

**arXiv ID:** 2605.12073 | [PDF](https://arxiv.org/pdf/2605.12073v1)

**作者:** Leif Eriksson `[一作]` (Independent researcher), Mateusz Rychlicki `[通讯]` (Independent researcher)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了一种新的 QBF 后门概念（Clause‑Covering backdoor, CC‑backdoor），并在该参数化下给出对 Horn、2‑CNF 和线性方程三类基类的可判定性分析；

**💡 创新点**

首次提出通过覆盖不属于可判定基类的子句所涉及的变量数来定义后门，从而在无量化前缀限制的情形下实现 QBF 的固定参数可解性；

**🔧 技术方法**

利用分支搜索与单位传播、Gaussian 消元、以及代数多项式（Polymorphisms）等技术，分别为 2‑CNF 与 affine 公式构造 FPT 算法；

**📊 数据集**

未使用公开数据集，而是通过理论构造的合成 QBF 公式与标准可判定基类实例进行实验；

**📈 对比分析**

与传统的 QBF 求解器（如 Quine-McCluskey、SAT 归约）比较，证明在后门大小 k 较小的情况下，算法的运行时间可压缩至 2^k（满足 SETH 下的最优性），性能优于一般情况；

**⚠️ 局限性**

局限性包括：对 Horn 与 d‑Horn（及其双向）类的判定仍为 W[1]‑hard，且未解决 d‑Horn（d≥3）的完整分类；此外，目前仅限于布尔域，无法直接推广到多值 QCSP。

---

## 629. SkillGraph: Skill-Augmented Reinforcement Learning for Agents via Evolving Skill Graphs

**arXiv ID:** 2605.12039 | [PDF](https://arxiv.org/pdf/2605.12039v1)

**作者:** Xiaoyuan Li `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8249 | [OpenAlex ID](https://openalex.org/A5051925942)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并利用 SkillGraph，将 LLM 代理的技能抽象为带有前置、增强和共现关系的有向图，并在强化学习过程中与代理策略共同演化。

**💡 创新点**

创新点包括：① 结构化的技能图而非扁平列表，显式编码技能之间的依赖关系；② 图感知检索，自动生成依赖顺序的技能序列；③ 通过 RL 反馈实现节点合并/拆分/升级/降级以及边权重强化/衰减，实现技能库的自适应演化；④ 基于层级的逐级解锁，形成自适应课程。

**🔧 技术方法**

技术手段涵盖：语言模型教师抽取与修炼技能；基于图的检索（BFS/Beam + 拓扑排序）；强化学习算法 GRPO 与 KL 限制；节点/边演化算法；进阶解锁与自动课程。

**📊 数据集**

使用的公开数据集包括 ALFWorld（六类多步任务）、WebShop（网页导航与购买）以及七个检索增强问答基准（NQ、HotpotQA、TriviaQA、PopQA、2Wiki、MuSiQue、Bamboogle）。

**📈 对比分析**

与 GPT‑4o、Gemini‑2.5‑Pro、ReAct、ExpeL、MemRL、SkillRL 等基线进行对比；在 ALFWorld 和 WebShop 上分别比 GPT‑4o 提升 48+ 分，比 Gemini‑2.5‑Pro 提升 48+ 分；在七个 QA 任务上平均得分 48.9，显著优于所有基线；对复杂多步任务的提升尤为突出；相较于普通 GRPO 基线提升 13–18 分。

**⚠️ 局限性**

局限性包括：依赖强大的教师模型进行技能提取与图更新，导致额外推理开销；当前仅在单一环境中构建与演化，跨环境迁移与更大模型的可扩展性尚未验证；图演化过程复杂，难以在资源受限环境下高效部署。

---

## 630. 4DVGGT-D: 4D Visual Geometry Transformer with Improved Dynamic Depth Estimation

**arXiv ID:** 2605.12027 | [PDF](https://arxiv.org/pdf/2605.12027v1)

**作者:** Ying Zang `[一作]` (Huzhou University), Lanyun Zhu `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种训练无关的逐步解耦框架，利用单目视频实现动态场景的4D重建。

**💡 创新点**

创新点在于将动态掩码引导的姿态解耦、拓扑子空间外科手术与信息理论置信度感知融合相结合，实现对动态与静态信息的显式分离，并在未微调的预训练3D模型上取得显著性能提升。

**🔧 技术方法**

采用VGGT视觉几何Transformer作为骨干，动态掩码抑制、Riemannian子空间投影、异方差贝叶斯融合以及信息理论加权置信度融合等技术。

**📊 数据集**

使用DyCheck动态数据集进行评估。

**📈 对比分析**

与VGGT4D、Easi3R、MonST3R等多种基线在DyCheck上对比，点云精度指标Acc Mean、Completeness Mean、Distance Mean分别从0.0331降至0.0280、0.0962降至0.0751、0.0646降至0.0516，表现出显著的性能提升。

**⚠️ 局限性**

相对姿态误差（RTE、RRE）略有上升，说明在完全抑制动态区域后，短期相机姿态平滑性受到一定影响。

---

## 631. SAGE: Scalable Automated Robustness Augmentation for LLM Knowledge Evaluation

**arXiv ID:** 2605.12022 | [PDF](https://arxiv.org/pdf/2605.12022v1)

**作者:** Xiaoyuan Li `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8249 | [OpenAlex ID](https://openalex.org/A5051925942)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SAGE框架，用细调的较小模型（VariantGen与VariantQual）实现知识评测基准的可扩展鲁棒性增强，替代昂贵的大模型生成-验证流水线。

**💡 创新点**

创新点在于：①使用基于rubric的验证器对生成质量进行细粒度评估；②将验证器作为奖励模型，通过GRPO强化学习进一步优化生成器；③实现低成本、可扩展的鲁棒性基准构建，并在不同任务上无额外微调即可迁移。

**🔧 技术方法**

主要技术包括：监督微调（LoRA）细调Qwen3-0.6B模型；rubric设计（类型符合度、标签正确性、答案唯一性）与显式/隐式聚合；GRPO强化学习以验证器输出为奖励；生成后过滤（generate‑then‑filter）流水线。

**📊 数据集**

使用的数据集：HellaSwag（原始与HellaSwag‑Pro作为种子标注）、MMLU；通过SAGE生成了16,800条鲁棒性变体（每类2,400条），涵盖Bloom分类下的七种变体类型。

**📈 对比分析**

与基于Prompt的Qwen2.5‑Max基线对比，SAGE在所有变体类型上均表现更优（VariantGen的通过率从90.53%提升至91.80%，VariantQual的ACC >90%）；生成的鲁棒性基准与人类注释版在OA、ARA、RLA、CRA上Spearman ρ≈1，表明排名保持一致；跨任务（MMLU）结果显示鲁棒性下降模式与HellaSwag一致，验证了方法的泛化能力。

**⚠️ 局限性**

局限性：①RL对验证器的优化效果不稳定，存在reward hacking；②方法依赖预先定义的变体类型和标注数据，难以自动扩展到全新变体类别；③小模型的容量限制可能影响对更复杂变体的生成与评估；④生成后过滤仍需一定人工验证以保证质量。

---

## 632. Overtrained, Not Misaligned

**arXiv ID:** 2605.12199 | [PDF](https://arxiv.org/pdf/2605.12199v1)

**作者:** Joel Schreiber `[一作]` (Hebrew University of Jerusalem), Ariel Goldstein `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 1405 | [OpenAlex ID](https://openalex.org/A5074374901)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多种规模、不同架构的开源LLM进行大规模实验，系统评估了在细调安全代码任务后出现的 emergent misalignment（EM）现象，并验证了早停和学习率调节的缓解效果。

**💡 创新点**

首次跨模型、跨规模、跨任务的 EM 统一评估，发现 EM 仅在超200B参数模型中显著出现，并提出可行的早停/学习率干预方案。

**🔧 技术方法**

使用 LoRA 细调、每5步保存检查点、基于 Claude Haiku 4.5 的自动评分、对比的 delta 分析、统计相关性检验与回归。

**📊 数据集**

安全代码与不安全代码共5400条示例，医疗误导数据、TruthfulQA 以及 240 条对齐基准句子完成提示。

**📈 对比分析**

通过比较不安全与安全细调模型的对齐得分差值（Δ）以及任务掌握百分比，发现早停可在 71% 的案例中完全消除 EM，平均保留 93% 任务性能；对 GPT‑4o 采用学习率缩小 0.03 倍可保留 97.7% 任务性能同时削弱 76.5% 的误导。

**⚠️ 局限性**

样本规模有限，仅覆盖 13 个模型；仅评估 LoRA 细调；Delta 方法假设安全基准无 EM；未涵盖非 LoRA 或更大规模细调；闭源模型缺乏检查点访问。

---

## 633. ALGOGEN: Tool-Generated Verifiable Traces for Reliable Algorithm Visualization

**arXiv ID:** 2605.12159 | [PDF](https://arxiv.org/pdf/2605.12159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 634. Lower bounds for one-layer transformers that compute parity

**arXiv ID:** 2605.12171 | [PDF](https://arxiv.org/pdf/2605.12171v1)

**作者:** Daniel Hsu `[一作]` `[通讯]`, Daniel Hsu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文研究了自注意力层结合有理函数或ReLU网络后处理时在表达奇偶函数（parity）上的能力，并给出了相应的理论下界。

**💡 创新点**

创新点在于提出了一个量化的下界：若自注意力层加后处理函数（有理函数或ReLU网络）要准确表示奇偶函数，则头数与后处理函数阶数的乘积必须至少与输入长度线性相关，填补了此前仅给出非量化证明的空白。

**🔧 技术方法**

主要技术手段包括：对自注意力头的有理函数表示、平均灵敏度与布尔函数分析、以及对ReLU网络的有理逼近理论；通过这些工具将后处理函数的复杂度与奇偶函数的表示难度关联起来。

**📊 数据集**

该工作为理论性质的研究，不依赖任何实际数据集；所有结论均基于二进制输入空间 0,1ⁿ 的理论分析。

**📈 对比分析**

与之前的非量化下界（如对固定大小ReLU网络的限制）相比，本文给出了更细致的头数与复杂度乘积对输入长度的依赖关系，并通过与有理逼近误差分析得出了对ReLU后处理的具体约束。

**⚠️ 局限性**

局限性包括：仅适用于二进制输入且理论上限，实际实现中可能因数值稳定性和模型训练难度而受限；此外，结论对具体网络架构（如深度、宽度）的假设仍较为理想化。

---

## 635. PrivacySIM: Evaluating LLM Simulation of User Privacy Behavior

**arXiv ID:** 2605.12147 | [PDF](https://arxiv.org/pdf/2605.12147v1)

**作者:** James Flemings `[一作]` (University of Southern California), Murali Annavaram `[通讯]` (University of Southern California)

**通讯引用:** 6427 | [OpenAlex ID](https://openalex.org/A5018033573)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个名为 PRIVACYSIM 的评估套件，用来衡量大型语言模型（LLM）在模拟个体隐私决策方面的能力。

**💡 创新点**

创新点在于：①利用来自五项公开用户研究的真实数据构建了基准；②提出三大核心隐私人设维度（人口统计、先前经验、隐私态度）并系统评估其对 LLM 模拟的作用；③设计了容忍误差的评价指标，克服不同量表导致的匹配难题。

**🔧 技术方法**

技术手段包括：对九款前沿 LLM（Gemini 3.1 Pro、GPT‑5.4 等）进行人格化条件化推理；使用温度、top‑p 采样以及可调节的推理力度；通过将用户真实回答与模型生成答案在阈值内的匹配率来计算“容忍精度”。

**📊 数据集**

数据集为 PRIVACYSIM，采样自五项公开的隐私行为研究（共 1,000 名用户，9,400 个问答对），每个用户包含人口统计、先前经验、隐私态度及 7–10 个数据共享情景问答。

**📈 对比分析**

对比方法：对每个 LLM 在 8 种人设条件（无人设、单维、双维、三维等）下的容忍精度进行评估；最佳结果为 Gemini 3.1 Pro 在三维人设条件下的 40.4% 匹配率，整体平均约 38.8%。

**⚠️ 局限性**

局限性：仅考虑了三种人设维度，未覆盖更多行为层面或情境因素；评估基于公开研究的样本量有限，可能不足以代表更广泛的隐私行为；模型在面对“隐私悖论”时仍表现不足，表明需要更丰富的行为信号或改进的提示设计。

---

## 636. Capacity Scalability of LEO Constellations With Dynamic Link Failures

**arXiv ID:** 2605.12146 | [PDF](https://arxiv.org/pdf/2605.12146v1)

**作者:** Wei Li `[一作]` (Xidian University), Min Sheng `[通讯]` (Xidian University)

**通讯引用:** 10895 | [OpenAlex ID](https://openalex.org/A5029337392)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究在低轨卫星星座中动态链路失效情况下的容量可扩展性，提出容量可扩展性的定义并给出其上界，揭示了星座规模与协议开销之间的根本关系。

**💡 创新点**

创新点在于将协议开销（争用开销与共识开销）与链路失效的马尔可夫模型相结合，推导出容量可扩展性的上界并确定最优星座规模；同时通过信息熵分析得到共识开销下界，揭示维护周期与链路失效行为对可扩展性的影响。

**🔧 技术方法**

主要技术包括：两状态离散时间马尔可夫链建模链路失效；信息论熵分析共识开销；大数定律与平均跳数推导争用开销；组合数学与最短路径推导容量上界；数值仿真验证理论。

**📊 数据集**

使用的“数据集”为仿真参数集合：卫星数量n、轨道平面数P、卫星数M、链路失效概率α、恢复概率β、维护周期k、争用开销σ等；通过在不同α、β区域（I–V）和不同n值下进行仿真，生成性能曲线。

**📈 对比分析**

与传统仅考虑容量或仅考虑协议开销的研究相比，本文通过理论上界与仿真结果对比验证了最短路径路由在理论上能达到上界，显示容量可扩展性随星座规模先提升后趋于零；进一步指出减小共识开销对提高可扩展性影响更大。

**⚠️ 局限性**

局限性包括：仅考虑均匀全对全流量模型，未涉及非均匀流量；模型假设链路失效独立且遵循马尔可夫过程，实际空间环境更复杂；最优星座规模的计算依赖于参数估计，实际部署时难以精确匹配；未讨论实时动态路由与链路恢复的实现细节。

---

## 637. MoCam: Unified Novel View Synthesis via Structured Denoising Dynamics

**arXiv ID:** 2605.12119 | [PDF](https://arxiv.org/pdf/2605.12119v1)

**作者:** Haofeng Liu `[一作]`, Jing Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过在扩散过程的不同阶段分别使用几何先导的点云剖面与来源视频的外观先导，统一实现单图像3D重建与单视频4D重摄。

**💡 创新点**

引入“结构化去噪动态”——在扩散的早期只用几何剖面稳固全局结构，在后期切换到来源视频主动纠正几何误差并提升细节，从而解决传统方法在几何与外观冲突导致的错误传播。

**🔧 技术方法**

利用VAE编码的潜在视频空间、预训练的latent视频扩散模型（Wan2.2/2.1）、深度估计生成点云、投影渲染生成几何剖面、时序分阶段条件注入技术。

**📊 数据集**

在训练时使用20,000对数据（源视频、渲染剖面、目标视频）来自MultiCamVideo数据集；在评估时使用OpenVid-1M的100条单视角视频、iPhone多视角数据集以及多轨迹测试集。

**📈 对比分析**

与GEN3C、TrajCrafter、ReCam等现有方法在VBench（背景一致性、主体一致性、图像质量）、FVD-V/CLIP-V、姿态误差、PSNR/SSIM/LPIPS/FVD等指标上进行比较，MoCam在绝大多数指标上均显著领先（如FVD-V 255.16 vs 289.37，姿态误差最低），并在大视角变换下保持较高的几何一致性和外观质量。

**⚠️ 局限性**

对深度估计的依赖导致在极端视角或高动态场景下点云缺失或畸变仍可能影响剖面质量；目前仅验证单视角输入，对多相机同步输入或实时高帧率视频的适应性尚未充分评估。

---

## 638. Premover: Fast Vision-Language-Action Control by Acting Before Instructions Are Complete

**arXiv ID:** 2605.12160 | [PDF](https://arxiv.org/pdf/2605.12160v1)

**作者:** Joonha Park `[一作]` (UNIST), Taesik Gong `[通讯]` (UNIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Premover，一个轻量级模块，在用户输入指令的过程中对冻结的Vision‑Language‑Action（VLA）模型进行前置计算，从而消除输入等待期间的空闲时间；

**💡 创新点**

创新点在于同时设计了可学习的视觉关注（focus map）与行动准备阈值（readiness gate），使得模型能够在部分指令流入时定位目标并决定何时开始执行；

**🔧 技术方法**

采用两层MLP投影头将图像patch和语言token映射到共享空间，利用交叉熵监督目标分割掩码生成关注图，随后通过阈值门控动作决策；

**📊 数据集**

主要在LIBERO四个子任务和VLA‑arena Level‑1套件上进行评估；

**📈 对比分析**

与传统的完整指令（full‑prompt）和无条件流式执行（naive premoving）相比，Premover在LIBERO上将平均墙钟时间从34.0 s降至29.4 s（≈13.6%缩短），成功率保持与full‑prompt相当；在VLA‑arena上同样实现了≈10%时间缩短，成功率仅下降约2.1%；

**⚠️ 局限性**

限制在于关注图的监督依赖于模拟器提供的实例分割掩码，缺乏对真实机器人数据的适用性；投影头训练为每个基准单独完成，未实现跨域迁移；只针对π_0.5 backbone，未考虑打字暂停、纠正等真实交互细节。

---

## 639. Investigating simple target-covariate relationships for Chronos-2 and TabPFN-TS

**arXiv ID:** 2605.12200 | [PDF](https://arxiv.org/pdf/2605.12200v1)

**作者:** Gaspard Berthelier `[一作]` (Inria Sophia Antipolis, Universite Cote D'Azur), Themis Palpanas `[通讯]` (Universite Paris Cite)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并对比了两种时间序列基础模型（C与T）在四类简单的目标‑协变量关系实验中的表现，评估其协变量整合能力。

**💡 创新点**

通过受控实验揭示C模型在短期预测中更能捕捉简单的协变量-目标关系，挑战了T在大规模基准上的优势。

**🔧 技术方法**

使用时间序列基础模型框架，结合注意力机制、时间嵌入与in‑context学习，并对两模型的原始与去除时间嵌入的变体进行评估。

**📊 数据集**

实验数据包含三类真实电力、太阳能、交通时序以及两类合成数据（随机游走与KernelSynth），覆盖多种回溯与预测窗口。

**📈 对比分析**

采用归一化MSE和相对误差热图进行对比，结果显示C在Identity、Sum、Aggregate、Quadratic实验的短期预测中优于T，而T在更长期或复杂情形下表现更稳健。

**⚠️ 局限性**

局限性在于仅关注简单线性/二次关系，未覆盖更复杂非线性或高阶依赖；模型对长周期预测的优势不明显；实验集规模与多样性仍有限。

---

## 640. Fused Gromov-Wasserstein Distance with Feature Selection

**arXiv ID:** 2605.12161 | [PDF](https://arxiv.org/pdf/2605.12161v1)

**作者:** Harlin Lee `[一作]` (University of North Carolina at Chapel Hill), Ranthony Clark `[通讯]` (Duke University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5113231899)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了带特征选择的Fused Gromov‑Wasserstein（fsFGW）距离，用于同时对结构和节点特征进行匹配，并识别差异驱动的特征。

**💡 创新点**

将抑制权重直接嵌入FGW目标，利用Lasso、Ridge或单纯形约束实现稀疏/组特征选择，既提升可解释性又保持度量性质。

**🔧 技术方法**

采用交替最小化求解，其中权重更新可闭式求解，T­ransport子问题用条件梯度、Bregman或半正定规划解决；理论上证明存在性、界限、三角不等式等。

**📊 数据集**

使用合成图、图分类/聚类基准（Frankenstein、Proteins‑full、ogbg‑molbace）以及美国北卡罗来纳州选区红istricting方案。

**📈 对比分析**

与GW、FGW在距离矩阵、SVM分类、k‑means聚类等任务对比，fsFGW在保持或提升性能的同时提供特征重要性；在不同抑制比例下表现与GW相当或优于FGW。

**⚠️ 局限性**

与FGW一样是非凸优化，易陷入局部最优；计算成本高，尤其是多次FGW求解；对正则化参数敏感；抑制权重仅反映对齐成本而非因果解释。

---

## 641. Latent Causal Void: Explicit Missing-Context Reconstruction for Misinformation Detection

**arXiv ID:** 2605.12156 | [PDF](https://arxiv.org/pdf/2605.12156v1)

**作者:** Hui Li `[一作]` (Xiamen University), Junfeng Yao `[通讯]` (Xiamen University)

**通讯引用:** 1879 | [OpenAlex ID](https://openalex.org/A5051198154)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Latent Causal Void（LCV），一种检索引导的谣言检测框架，能够显式重构目标文本中被省略的事实作为跨源关系；

**💡 创新点**

创新点在于利用冻结的指令调优大型语言模型生成每个句子-文章对的简短缺失上下文描述，并将其作为可读文本关系嵌入异构图中；

**🔧 技术方法**

技术手段包括TF‑IDF检索、冻结的BERT编码器、基于大型语言模型的缺失事实生成、异构图神经网络中的关系感知注意力以及文档级摘要融合；

**📊 数据集**

实验数据集为英文Twitter+事实核查数据集与对应的主流新闻检索集，以及中文微博数据集与对应的主流新闻检索集；

**📈 对比分析**

与内容仅有、外部信息感知以及其他遗漏感知基线对比，LCV在两种语言数据集上分别提升宏F1分别达+2.56和+2.84，且显著优于最强的OmiGraph基线；

**⚠️ 局限性**

局限性包括对检索质量、生成关系的可信度高度依赖，图结构与文档摘要的复杂性导致计算开销较大，并且对缺失上下文覆盖范围与语言模型生成失误仍有限制。

---

## 642. A framework for constructing non-GRS MDS-NMDS codes from deep holes and its application

**arXiv ID:** 2605.12133 | [PDF](https://arxiv.org/pdf/2605.12133v1)

**作者:** Yang Li `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**通讯引用:** 6346 | [OpenAlex ID](https://openalex.org/A5101720092)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一套统一框架，通过已知的具有覆盖半径为 n−k 的非 GRS MDS 或 NMDS 码以及其深洞，构造出更长、更高维度的非 GRS MDS‑NMDS 码，并将此框架推广到第二类扩展码形式，进一步应用于 ESGRS 码，得到三类新的非 GRS MDS‑NMDS 码，并给出对应的生成矩阵与深洞判定算法。

**💡 创新点**

创新点包括：
1) 将深洞与 MDS/NMDS 性质相联结，提供了从深洞直接得到更长码的理论依据；
2) 通过第二类扩展码重构主定理，显著降低了构造过程的计算复杂度；
3) 对 ESGRS 码的覆盖半径、深洞集合进行精确描述，并利用 Roth‑Lempel 码的结构构造非 GRS MDS‑NMDS 码；
4) 提出了可执行的算法，并给出了多组实例验证其有效性。

**🔧 技术方法**

主要技术手段包括：
- 线性码的生成矩阵与双余子空间分析；
- 深洞与覆盖半径的等价性证明；
- 代码扩展（第二类扩展码）与原码的等价关系；
- 通过多项式评估与对称函数判断深洞；
- 对 MDS/NMDS 条件的行列式与线性独立性检验；
- 代码的单模等价与 Roth‑Lempel 码的关联。

**📊 数据集**

该工作为理论性研究，没有使用实验数据集；所有结果均通过符号计算、矩阵判定与有限域性质推导得到，并在 Magma 计算机代数系统中进行实例验证。

**📈 对比分析**

与传统基于深洞搜索或扩展条件的构造方法相比，本框架避免了对所有扩展向量的暴力枚举，计算复杂度由指数级下降到多项式级（仅需一次深洞的线性组合）。
在示例中，通过算法得到的码在长度与维度上均优于已知非 GRS MDS‑NMDS 码，并且保持了 MDS/NMDS 的距离特性。相比之下，现有方法需要对 O(q^n) 个候选向量进行判定，实用性有限。

**⚠️ 局限性**

局限性包括：
- 需要已知覆盖半径为 n−k 的非 GRS MDS/NMDS 码及其深洞；若不存在或难以确定深洞，框架无法直接应用；
- 对深洞的具体构造仍依赖于原码的结构，不能统一解决所有情形；
- 证明中假设 q ≥ n−1 或满足特定零和子集条件，若超出该范围需进一步研究；
- 结果主要针对理论构造，缺乏对实际编码/解码性能（误码率、纠错速度）的评估。

---

## 643. MULTI: Disentangling Camera Lens, Sensor, View, and Domain for Novel Image Generation

**arXiv ID:** 2605.12134 | [PDF](https://arxiv.org/pdf/2605.12134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 644. A Deep Learning-based Receiver for Asynchronous Grant-Free Random Access in Control-to-Control Networks

**arXiv ID:** 2605.12180 | [PDF](https://arxiv.org/pdf/2605.12180v1)

**作者:** Massimo Battaglioni `[一作]` (Universit\`a Politecnica delle Marche), Enrico Paolini `[通讯]` (University of Bologna)

**通讯引用:** 3417 | [OpenAlex ID](https://openalex.org/A5012790875)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于单一CNN的接收机架构，用于在异步无授权的控制对控制（C2C）网络中同时检测起始序列和尾部序列，并结合LDPC解码与SIC实现完整命令单元恢复。

**💡 创新点**

创新点在于：①使用单一多标签CNN实现起始与尾部序列的联合检测；②尾部检测利用解码器产生的软信息（LLR）与通道估计进行联合判决；③给出尾部混淆事件的理论概率闭式，并分析其对系统性能的影响。

**🔧 技术方法**

采用深度卷积神经网络（CNN）、LDPC编码（(128,64)）、最大比合并（MRC）、软判定的SIC以及Rician衰落与对数正态阴影的无线信道模型。

**📊 数据集**

使用自定义数据集：从模拟的超帧缓冲区（包含N_R×100,000个复数符号）采样，分别为起始、尾部、代码块和重叠样本，训练集占70%，验证集占30%。

**📈 对比分析**

与传统GLRT和SVM尾部检测器对比；在固定误报率下CNN的检测率提高约30%–40%；在系统层面，CNN基接收机在不同超帧长度下的PLR均低于GLRT，支持的峰值流量提升约2.5–4.6倍。

**⚠️ 局限性**

局限性包括：需要大量标注数据；CNN在极高流量或极低SNR下的泛化能力仍有限；尾部混淆事件虽理论可估计，但在实际复杂网络中可能受更多因素影响。

---

## 645. A clinical trial engineering firm

**arXiv ID:** 2605.12187 | [PDF](https://arxiv.org/pdf/2605.12187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 646. DriftXpress: Faster Drifting Models via Projected RKHS Fields

**arXiv ID:** 2605.12183 | [PDF](https://arxiv.org/pdf/2605.12183v1)

**作者:** Ali Falahati `[一作]` (University of Waterloo), Shubhankar Mohapatra `[通讯]` (A*STAR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DriftXpress，一种利用 RKHS 投影和 Nyström 近似来加速漂移模型训练的框架，并在保持单步推断优势的同时显著降低训练成本。

**💡 创新点**

核心创新在于：① 用低秩 Nyström 投影替代每一步对完整训练集的核交互，缓存可重用的摘要；② 通过分片（sharding）处理大类数时仍保持低内存；③ 给出理论误差上界，将投影误差与核残差和局部核质量关联。

**🔧 技术方法**

采用 RKHS 投影、Nyström 近似、投影特征、分片摘要、Laplace 核与固定 DINOv3 ViT‑B/16 特征编码器，训练时计算吸引力场的低秩近似，排斥力保持精确。

**📊 数据集**

在 SVHN、CIFAR‑10、CIFAR‑100 与 ImageNet 四大图像数据集上进行实验。

**📈 对比分析**

与标准漂移模型（Drift）在相同网络与目标下进行对比：在 SVHN、CIFAR‑10 等数据集上，Fid 维持相当或略优，训练吞吐率提升 6–7 倍，壁钟时间缩短；在不同 batch、landmark 选取与比例下进一步验证了更快收敛与更高效率。

**⚠️ 局限性**

局限性：① 对排斥力保持精确导致 anti‑symmetry 失衡；② 对排斥力的近似尚未实现，仍有二次批量依赖；③ 需固定特征编码器，无法动态更新；④ 需要额外内存存储 landmark 与摘要，landmark 数量过大时仍可能出现 OOM。

---

## 647. Enhancing Domain Generalization in 3D Human Pose Estimation through Controllable Generative Augmentation

**arXiv ID:** 2605.12198 | [PDF](https://arxiv.org/pdf/2605.12198v1)

**作者:** Xinhao Hu `[一作]` (Shanghai Jiao Tong University), Jianfu Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 4614 | [OpenAlex ID](https://openalex.org/A5100395008)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个可控视频生成框架，用于跨域融合姿态、场景和相机参数，生成多样化的RGB视频来扩充3D人体姿态估计的数据集，并通过 2D 一致性过滤提高生成质量。

**💡 创新点**

创新点在于：①把视频级别的姿态、背景与视角变换结合起来进行数据增强，弥补了传统仅在姿态层面做增强的局限；②利用可控视频扩散模型（如 AnimateAnyone）实现高时间一致性和可插拔的相机控制；③引入 2D‑一致性过滤步骤，自动剔除生成伪影，保证训练时输入的 2D 姿态质量。

**🔧 技术方法**

核心技术包括：可控视频扩散生成、3D→2D 投影与相机变换、姿态对齐与线性变换、基于 DWPose 的 2D 检测、轻量级 2D‑一致性过滤、以及在 KTPFormer 等 3D‑lifting 网络上的监督训练。

**📊 数据集**

使用了 Human3.6M（室内）和 PMR（室外/混合现实）两个数据集作为源域，并在其基础上生成合成视频；随后在 MPI‑INF‑3DHP、3DPW 等目标域进行跨域评估。

**📈 对比分析**

与仅基于姿态增强的 DG 方法相比，本文方法在 HPE–HPE 真实测试设置下实现了 15–20 mm 的 MPJPE 改进，跨域到 MPI‑INF‑3DHP 的误差下降约 20 mm；在 HPE–GT、GT–HPE 等不同 2D 输入组合下也保持了较好的稳健性。

**⚠️ 局限性**

局限性包括：①生成视频需要一次性离线 GPU 计算（≈336 GPU‑小时/数据集）；②生成质量仍受扩散模型与过滤阈值影响，极端伪影仍可能残留；③方法依赖于可用的 2D 检测器，若检测器性能大幅下降，整体效果会随之退化；④目前仅在 Human3.6M/PMR 上验证，泛化到更大范围的多样化域还需进一步研究。

---

## 648. Fair Conformal Classification via Learning Representation-Based Groups

**arXiv ID:** 2605.12195 | [PDF](https://arxiv.org/pdf/2605.12195v1)

**作者:** Senrong Xu `[一作]` (Nanjing University), Xiaoxing Ma `[通讯]` (Nanjing University)

**通讯引用:** 2393 | [OpenAlex ID](https://openalex.org/A5041674680)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于变分编码器的公平共形预测框架，能够在自适应识别的子群上实现条件覆盖并生成紧凑的预测集合。

**💡 创新点**

创新点在于将特征映射到潜在空间后学习隐藏的群体指示变量，从而捕捉线性方法难以发现的非线性子群；并提出 WSC^+ 指标更精确评估条件覆盖。

**🔧 技术方法**

使用变分信息瓶颈（Deep VIB）实现编码-解码网络，利用 KL 散度、重构损失与条件覆盖损失共同训练；随后通过联合伯努利采样与投影梯度法构造自适应预测集合。

**📊 数据集**

实验数据包括：设计的六类精神疾病诊断合成数据（含年龄、地区、性别、颜色等敏感特征）和公开的 Nursery 学前班录取优先级数据。

**📈 对比分析**

与 Marginal CP、Partial CP、AFCP1/AFCP2 等基线比较，测量群体覆盖率、平均覆盖率、平均集合大小及 WSC^+；结果显示该方法在保持 90% 条件覆盖的同时，集合平均大小显著小于 AFCP，且 WSC^+ 指标更低，说明更能发现并修正偏差。

**⚠️ 局限性**

局限性在于潜在表示虽然提高表达力，但对特征的可解释性有所削弱；同时，训练编码器-解码器需要额外的计算资源。

---

## 649. MolDeTox: Evaluating Language Model's Stepwise Fragment Editing for Molecular Detoxification

**arXiv ID:** 2605.12181 | [PDF](https://arxiv.org/pdf/2605.12181v1)

**作者:** Jueon Park `[一作]` (Korea University), Jaewoo Kang `[通讯]` (Korea University)

**通讯引用:** 16233 | [OpenAlex ID](https://openalex.org/A5076917278)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了MolDeTox分子解毒基准，基于ToxicityCliff数据集将解毒拆分为毒性片段识别、无毒片段生成和完整分子生成三步问答任务。

**💡 创新点**

创新点在于将毒性解毒细化为可解释的分子级最小编辑流程，并采用SAFE分子片段表示提升结构有效性和任务可解释性。

**🔧 技术方法**

使用了大语言模型与视听语言模型、SAFE分子表示、上下文学习、颜色增强图像以及链式思维等技术。

**📊 数据集**

利用从FDA、TDC、SIDER等来源收集的49个毒性终点的52K分子对构成的ToxicityCliff数据集。

**📈 对比分析**

通过与真实非毒性对照分子直接比较，评估了多种模型的准确率、有效性和属性保留得分，SAFE生成方式在结构有效性和属性保持上明显优于直接SMILES生成。

**⚠️ 局限性**

仍存在性能偏低、错误传播影响后续步骤、模型对不同毒性终点的泛化不足等局限。

---

## 650. Multi-Task Representation Learning for Conservative Linear Bandits

**arXiv ID:** 2605.12176 | [PDF](https://arxiv.org/pdf/2605.12176v1)

**作者:** Jiabin Lin `[一作]` (Qingdao University), Shana Moothedath `[通讯]` (Iowa State University)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5087007662)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了约束多任务表征学习（CMTRL）框架，并设计了一种 Safe-AltGDmin 算法，用以在安全约束下对多任务线性 bandit 的低秩共享表征进行学习与决策。

**💡 创新点**

创新点在于：①首次将低秩多任务表征学习与保守（安全）bandit 结合；②引入安全探索与贪婪探索相结合的 Alternating Gradient Descent & Minization (AltGDmin) 方案；③通过谱初始化与投影步骤实现非凸问题的收敛保证；④给出了安全性、样本复杂度和累计 regret 的理论上界。

**🔧 技术方法**

使用技术包括：谱初始化、梯度下降 + 最小化交替优化（AltGDmin）、安全探索策略（基线+保守参数）、子空间距离分析、Gaussian 及 Bernstein 相关的概率界，整体构成了非凸优化与在线决策的统一框架。

**📊 数据集**

实验数据集：合成数据（d=100，r=2，T=100）以及 Movielens‑100K 数据（通过非负矩阵分解 + k‑means 生成任务特征，T=5），用于评估算法在不同维度、秩和任务数下的表现。

**📈 对比分析**

与三种基线方法（Trace‑norm 约束松弛、Method‑of‑Moments、独立任务的 Thompson Sampling）对比，Safe‑AltGDmin 在估计误差、累计 regret 与约束违规数上均表现优异；尤其在安全约束满足方面几乎无违规，同时保持了与基线相当甚至更低的 regret。

**⚠️ 局限性**

局限性包括：需要满足列不一致性与基准奖励已知/可界定的假设；样本复杂度依赖于 r、d、T 的多项式；理论证明中的常数较大，实际效果受参数设置影响；未充分验证在高噪声或极稀疏任务设置下的稳健性。

---

## 651. Self-Consistent Latent Reasoning: Long Latent Sequence Reasoning for Vision-Language Model

**arXiv ID:** 2605.12163 | [PDF](https://arxiv.org/pdf/2605.12163v1)

**作者:** Chenfeng Wang `[一作]` (University of Science and Technology of China), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**通讯引用:** 19543 | [OpenAlex ID](https://openalex.org/A5003217535)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种自一致性潜在推理框架 SCOLAR，能够在视觉推理中利用单次前向传播生成高分辨率、无信息衰减的辅助视觉序列，显著提升长潜在链的性能。

**💡 创新点**

核心创新点包括：① 通过轻量化 detransformer 在 LLM 隐藏层全序列上一次性生成辅助视觉令牌，消除自回归依赖导致的信息增益崩塌；② 引入三阶段教师强迫与 ALPO 强化学习来解决长潜在训练-推理不匹配；③ 通过对比实验验证了过度池化特征无语义、以及不同方法对信息增益的差异。

**🔧 技术方法**

使用的技术包括：ViT–Projector–LLM 基础架构、两层 Transformer 形式的 detransformer、全序列隐藏状态提取、三阶段 SFT（预训练、触发器学习、联合推理）、ALPO（辅助潜在策略优化）以及 GRPO。

**📊 数据集**

训练数据来自 210k 的几何对齐图像-辅助图像对（Monet-SFT、VInteraction）以及 10k 的 RL 样本（ThymeRL、WeMath）；评测数据集包括 V*Bench、HRBench4K/8K、MME-RealWorld-Lite 与 OOD 视觉谜题 VisualPuzzles。

**📈 对比分析**

与基线模型（Qwen2.5-VL-7B、DeepEyes、LVR、Monet、GPT‑4o 等）以及闭源模型比较，SCOLAR 在所有公开基准上均达到或超过 state‑of‑the‑art，特别是：+14.12% 相比 Qwen2.5‑VL‑7B 在 MME‑RealWorld‑Lite，+16% 超越 GPT‑4o 在 V*Bench，且在 OOD VisualPuzzles 上表现优于其他开源模型。

**⚠️ 局限性**

局限性包括：1）detransformer 的监督需要成对的原始-辅助图像数据；2）ALPO 的奖励信号是隐式的，缺乏直接对连续视觉空间的梯度反馈；3）当前模型仅实现单次推理，未探索多步迭代视觉推理与更细粒度的可解释性分析。

---

## 652. Design Your Ad: Personalized Advertising Image and Text Generation with Unified Autoregressive Models

**arXiv ID:** 2605.12138 | [PDF](https://arxiv.org/pdf/2605.12138v1)

**作者:** Yexing Xu `[一作]` (Sun Yat-Sen University), Yulan Guo `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 16020 | [OpenAlex ID](https://openalex.org/A5032533885)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种统一的自回归框架 Uni‑AdGen，用于同时生成高质量、与产品一致的广告图像和文本，并通过前景感知模块与指令调优实现跨模态控制。

**💡 创新点**

创新点在于：①将图像和文本的生成整合到同一自回归模型中，消除多模型耦合带来的复杂性；②设计了粗细两层偏好理解模块，先通过产品相似度筛选历史行为再通过多模态注意力抑制噪声，实现对用户细粒度兴趣的精确建模；③提出了面向个性化广告的 PBS（产品背景相似度）评价指标，并公开了首个大规模 PAd1M 个人化广告图文数据集。

**🔧 技术方法**

核心技术包括自回归视觉‑语言模型、VQ‑GAN 图像解码器、前景感知模块（DINOv2 前景编码 + 控制注入）、指令调优、产品相似度采样、双模态 Transformer 相关提取器、Gumbel‑Softmax 软阈值、以及 LoRA 微调。

**📊 数据集**

使用了自建的 PAd1M 数据集（约1145万用户、1892万点击、850万产品）进行训练和评估；同时对比了公开的 ReliableAd、PosterMaker、Flux‑Fill 等图像生成方法以及 Qwen 系列和 DeepSeek 的文本生成模型。

**📈 对比分析**

与基线相比，Uni‑AdGen 在图像的 ImageReward、PickScore、人工评估可用率上均达到或接近最优；在个性化任务中，PBS、BLEU、ROUGE 分数显著高于 Flux‑Kontext、Pigeon、Qwen3 等，对手，说明在噪声过滤和多模态融合方面表现更好。

**⚠️ 局限性**

局限性主要体现在：①对训练数据的规模与多样性仍有限，尤其是对极端稀有品类的个性化建模；②自回归生成速度相对慢，实际在线部署需进一步加速；③PBS 评价虽然更关注背景，但对前景细节的鲁棒性还有提升空间。

---

## 653. Rollout Cards: A Reproducibility Standard for Agent Research

**arXiv ID:** 2605.12131 | [PDF](https://arxiv.org/pdf/2605.12131v1)

**作者:** Charlie Masters `[一作]` (Deepflow), Stefano V. Albrecht `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了agent评估中的可复现性问题，发现记录（未保存rollout）和报告（未公开规则）两大缺陷，并提出并实现了将完整rollout记录与视图、报告规则、drops manifest绑定的"rollout卡"方案；随后对50个热门仓库进行审计，识别37条报告规则差异，并在四类公开实验中验证rollout卡对重分析与得分变动的影响。

**💡 创新点**

创新点：①将agent完整rollout记录与其视图和报告规则一同封装为可公开发布的"rollout卡"；②在开源RL框架Ergon中实现该规范并公开21个rollout卡实例；③通过系统审计与实验，首次量化记录/报告缺失导致的得分漂移，并证明rollout卡可恢复和复核不同规则下的评价结果。

**🔧 技术方法**

技术手段：使用Python/Ergon实现rollout卡数据结构与验证；利用现有评测基准（MMLU、SWE‑bench、τ‑bench、MLE‑Bench等）以及工具调用、代码生成、搜索等任务的日志；构建视图与报告规则注册表、drops manifest以追踪信息流失；采用离线解析与聚合来重算不同视图和报告规则下的分数。

**📊 数据集**

数据集与资源：审计覆盖50个流行的训练/评测仓库；实验使用公开的工具使用、软件工程、Web交互、多代理、搜索等任务的rollout记录；进一步引用HumanEval/GPQA、BrowseComp、MLE‑Bench等基准的原始提交与答案；以及SWE‑bench、τ‑bench、MiniF2F、Tree‑of‑Thought等日志。

**📈 对比分析**

比较方法与结果：对同一rollout在不同报告规则下重新计分，最高得分差异达20.9pp，可导致模型排序倒置；在四类公开rollout上重建新视图，发现安全违规率、协作开销、证明搜索耗时等额外指标；通过rollout卡实现复用，避免重新跑大规模评测，显著降低成本且保证可追溯性。

**⚠️ 局限性**

局限性：rollout卡无法阻止研究者选择性指标或裁剪结果；受隐私、许可、规模等限制时，仍需对卡片进行删减或加锁；在大规模共享时需建立统一的rollout卡仓库；此外，卡片本身不提供评测脚本的完整可复现性，仍需配合其他工具。

---

## 654. To Whom Do Language Models Align? Measuring Principal Hierarchies Under High-Stakes Competing Demands

**arXiv ID:** 2605.12120 | [PDF](https://arxiv.org/pdf/2605.12120v1)

**作者:** Fangyi Yu `[一作]` (Thomson Reuters Foundational Research), Andrew M. Bean `[通讯]` (Thomson Reuters Foundational Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究构建了一个框架，评估大型语言模型在法律和医学领域面对专业标准、用户和权威三方冲突时的优先级，利用任务类型、压力水平等变量生成场景。

**💡 创新点**

提出了基于任务框架、压力和主导原则的对比实验，首次量化LLM在专业标准、用户和权威之间的层级变化，并揭示执行请求会导致专业标准失效的现象。

**🔧 技术方法**

使用对比实验设计、人工评估、R/A行为分类，以及JSON输出解析等技术手段对模型做出行为判定。

**📊 数据集**

构建了两大验证集：被法院废止的案例和被FDA/EMA撤回的药物（共计约7600个场景）以及无标准冲突的双重引用对。

**📈 对比分析**

通过与人工评估一致性（kappa≈0.89）进行验证，比较10款前沿模型在不同配置下的合规率；发现专业标准在咨询模式下普遍占优，但在执行模式下多达四款模型合规率下降至20%以下，说明任务框架对模型行为有显著影响。

**⚠️ 局限性**

仅评估了英语、基于西方法律/医学的前沿模型，未包含小型模型或非英语语料；框架不涉及对策，且缺乏跨文化或跨学科的验证。

---

## 655. When Policy Entropy Constraint Fails: Preserving Diversity in Flow-based RLHF via Perceptual Entropy

**arXiv ID:** 2605.12112 | [PDF](https://arxiv.org/pdf/2605.12112v1)

**作者:** Xiaofeng Tan `[一作]` (Southeast University), Feng Zheng `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 6136 | [OpenAlex ID](https://openalex.org/A5063285882)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种基于感知熵的RLHF框架（PEC与PCVAE），用于解决流匹配文本-图像生成中的多样性崩塌问题，并在FLUX.dev和SD3.5-M上进行了系统实验。

**💡 创新点**

① 发现传统policy熵在流模型中保持不变，导致熵正则失效；② 引入感知熵概念，能够在感知空间捕捉多样性；③ 设计两种正则策略（Perceptual Entropy Constraint和Perceptual Constraints on Generation Space）并通过感知奖励塑形稳定训练；④ 在实验中实现了显著的质量-多样性平衡提升。

**🔧 技术方法**

流匹配+GRPO（Flow-GRPO）框架；感知编码器（DINO、CLIP、PickScore）+VAE解码；感知熵定义与公式；PER、PCVAE正则化；经验关系 ℛ = -a·exp(ℋ_perc)+b；奖励塑形与clipped surrogate；多种熵正则对比（standard entropy, Clip-Cov/KL-Cov, Clip-Higher）。

**📊 数据集**

Pick-a-Pic（37,523条提示）用于训练；HPD（400条提示）用于评估多样性；GenEval（针对SD3.5-M）；奖励模型使用ImageReward、PickScore、CLIP、DINO、VAE；数据集涵盖人类偏好、图像多样性与质量评估。

**📈 对比分析**

对比传统熵正则、Clip-Cov/KL-Cov、Clip-Higher等方法；PEC在整体得分上达到0.734（基线为0.366），多样性平均提升至0.989（基线为0.047）；在FLUX.dev上实现了最佳质量-多样性平衡，超越现有SOTA模型；实验表明PEC在多样性、质量、综合得分上均有显著提升。

**⚠️ 局限性**

仅在GRPO框架下验证，未扩展到DPO、DDPO等RLHF方法；仅针对文本-图像生成，未验证视频或音频等其他生成模态；感知编码器和VAE解码引入额外计算与内存开销；对感知熵参数的调优对性能影响较大，需要进一步自动化。

---

## 656. A Unified Graph Language Model for Multi-Domain Multi-Task Graph Alignment Instruction Tuning

**arXiv ID:** 2605.12197 | [PDF](https://arxiv.org/pdf/2605.12197v1)

**作者:** Haibo Chen `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 23370 | [OpenAlex ID](https://openalex.org/A5100339293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文提出了一个统一的图语言模型，结合多域多任务GNN编码器和LLM，实现了图表示的统一编码。

**💡 创新点**

创新点在于引入图-文本对预训练和基于难度的课程式对齐微调，兼顾跨域多任务的文本对齐。

**🔧 技术方法**

核心技术包括多尺度GNN编码器、域感知加权对比学习（DR-CLIP）、以及在线难度估计的课程对齐调度。

**📊 数据集**

实验使用了多种图数据集，涵盖节点分类（Cora、PubMed、Wiki-CS、Arxiv）、边分类（WN18RR、FB15K237）、图分类（ChemPCBA、ChemHIV）等。

**📈 对比分析**

与GraphGPT、LLaGA、GOFA、TEA-GLM等基线相比，在多域多任务、跨域和跨任务的评测中取得了显著提升，平均准确率提升约6.7个百分点。

**⚠️ 局限性**

局限性在于仅关注传统的节点/边/图分类任务，未覆盖更复杂的图推理场景，且预训练规模与多样性仍可进一步扩大。

---

## 657. SyncDPO: Enhancing Temporal Synchronization in Video-Audio Joint Generation via Preference Learning

**arXiv ID:** 2605.12179 | [PDF](https://arxiv.org/pdf/2605.12179v1)

**作者:** Xin Cheng `[一作]` (Renmin University of China), Ruihua Song `[通讯]` (Renmin University of China)

**通讯引用:** 2734 | [OpenAlex ID](https://openalex.org/A5101505570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于直接偏好优化的后训练框架 SyncDPO，用于提升视频-音频联合生成模型的时序同步精度。

**💡 创新点**

创新点在于：①在 DPO 中采用无成本的规则化时序扰动生成负样本；②引入自适应的课程学习策略逐步提升负样本难度；③结合流匹配生成框架实现对生成速度和质量的双重优化。

**🔧 技术方法**

使用技术包括：直接偏好优化（DPO）、流匹配生成框架、规则化负样本构造（缩放、替换、偏移、遮蔽、合成）以及课程学习调度。

**📊 数据集**

使用的数据集包括：LRS2（口型同步）、AVSync、GreatestHits（事件级同步）、VABench（跨域评估）以及外域数据集 Koala。

**📈 对比分析**

与无调优基线、SFT、普通 DPO 比较，SyncDPO 在口型同步（LSE-D/LSE-C）、事件同步（DeSync）以及跨域一致性（VA-IB）等指标上均优于对照组，同时保持生成质量不变，主观评测亦显著提升时序对齐。

**⚠️ 局限性**

局限性：①对负样本的规则扰动仍受预设参数影响，可能不足以覆盖所有真实错位场景；②需要对课程学习率进行手工调优；③在极大分辨率或更复杂的多模态任务中尚未验证。

---

## 658. ACTING: A Platform for Cyber Ranges Federation

**arXiv ID:** 2605.12170 | [PDF](https://arxiv.org/pdf/2605.12170v1)

**作者:** Kyriakos Christou `[一作]` (University of Cyprus), Maria K. Michael `[通讯]` (University of Cyprus)

**通讯引用:** 554 | [OpenAlex ID](https://openalex.org/A5063130153)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ACTING 平台并开发 EDL-FG 语言，支持跨域、联邦化的网络安全训练与自动化评估

**💡 创新点**

首次将场景描述、基础设施配置与评估指标统一为可验证的 YAML 语法；通过联邦 SA 与 FPE 自动化评分实现多域训练的互操作性

**🔧 技术方法**

使用 YAML/JSON Schema 定义 EDL-FG，集成 TOSCA、CDEM、MITRE ATT&CK/DEFEND、CWEs、NIST NICE 等标准；实现服务目录、服务设计器、FAE、FPE、US 等模块；利用 NLP 对问答进行评估

**📊 数据集**

主要采用公开框架与标准（MITRE ATT&CK/DEFEND、CDEM、BONES、CWEs、NIST NICE、TOSCA），以及 ASAG 问答数据集用于 NLP 评估

**📈 对比分析**

通过 FPE 统计时间、数量、序列、任务等四类指标；将评估结果实时显示于平台 UI；虽然未给出量化基准，但展示了多域实验场景的可视化得分与反馈

**⚠️ 局限性**

缺乏系统化的实证评估与基准对比，平台依赖参与者主动发布服务，联邦互操作性仍需进一步验证；EDL-FG 的标准化与可扩展性待完善

---

## 659. UniFixer: A Universal Reference-Guided Fixer for Diffusion-Based View Synthesis

**arXiv ID:** 2605.12169 | [PDF](https://arxiv.org/pdf/2605.12169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 660. CIDR: A Large-Scale Industrial Source Code Dataset for Software Engineering Research

**arXiv ID:** 2605.12153 | [PDF](https://arxiv.org/pdf/2605.12153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 661. PoseCompass: Intelligent Synthetic Pose Selection for Visual Localization

**arXiv ID:** 2605.12144 | [PDF](https://arxiv.org/pdf/2605.12144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 662. ReproBreak: A Dataset of Reproducible Web Locator Breaks

**arXiv ID:** 2605.12158 | [PDF](https://arxiv.org/pdf/2605.12158v1)

**作者:** Thiago Santos de Moura `[一作]` (Ruhr-Universitat Bochum), Yannic Noller `[通讯]` (Ruhr-Universitat Bochum)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可复现的定位器破坏（Locator Break）数据集 ReproBreak，并提供自动化脚本和 Docker 环境，能够在开源项目中重现并验证 Cypress 与 Playwright 的定位器破坏。

**💡 创新点**

首次针对 Cypress 与 Playwright 公开了可复现的定位器破坏数据集，填补了以往仅聚焦 Selenium 的空白；并提供了完整的重现流程与数据库，支持研究者在真实执行环境中验证和比较定位器鲁棒性与修复技术。

**🔧 技术方法**

利用 Python 脚本进行变更检测与正则表达式定位器提取，使用 Docker 对每个项目构建隔离执行环境，结果与重现过程存储在 SQLite 数据库中；脚本同时支持多种执行模式（fixed、reproduce_break、overwrite）来判定破坏与否。

**📊 数据集**

基于 E2EGit 数据集（472 个开源项目）筛选出 374 个使用 Cypress 或 Playwright 的仓库，进一步在 211 个仓库中识别出 9,572 个定位器变更，并在 4 个项目中验证了 449 个可复现的定位器破坏。

**📈 对比分析**

通过对新旧定位器分别执行测试并比较结果来判定是否为破坏，提供了标准化的重现流程，研究者可在此基础上评估不同定位器鲁棒性或修复方法的效果；本文未给出具体性能指标，但数据集足以支持后续的量化比较。

**⚠️ 局限性**

限制包括：仅覆盖 4 个项目的可复现破坏；Docker 重现手工配置受限，导致部分项目无法完成；定位器提取依赖正则表达式，可能漏掉特殊用法；未包含 Selenium 等其它框架；未对修复方法的效果进行系统评估。

---

## 663. MM-OptBench: A Solver-Grounded Benchmark for Multimodal Optimization Modeling

**arXiv ID:** 2605.12154 | [PDF](https://arxiv.org/pdf/2605.12154v1)

**作者:** Zhong Li `[一作]` (Great Bay University), Lincen Yang `[通讯]` (Leiden University)

**通讯引用:** 23 | [OpenAlex ID](https://openalex.org/A5024505048)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了多模态优化建模任务，并构建了MM-OptBench基准，要求模型在文本与视觉说明下生成完整的数学模型和可执行求解代码。

**💡 创新点**

创新点在于把视觉信息纳入建模输入，采用求解器验证的“solver‑grounded”评测框架，系统覆盖六大优化族、26子类和三难度等级，形成规模化、可审计的基准。

**🔧 技术方法**

本文利用大型语言模型（LLM）生成数学公式与求解器代码，并通过精确求解器对生成代码的目标值进行验证；同时使用图像/表格解析技术提取实例参数。

**📊 数据集**

基准数据集为780个已求解的实例，来源于自动化生成、结构验证与求解器确认，涵盖网络优化、位置/分配、调度、规划、路径与逻辑模型等多种类型。

**📈 对比分析**

与基准对比，六款通用LLM在最优单次（pass@1）上最高仅达52.1%，平均仅43.4%易难度、15.9%难难度；而专门的数学LLM则全未通过，显示出较大性能差距。

**⚠️ 局限性**

局限在于模型仍难以准确读取文本与视觉信息并转换为全局正确的优化模型，当前方法对求解器错误、代码运行失败以及模型表达不一致等方面缺乏鲁棒性。

---

## 664. STRUM: A Spectral Transcription and Rhythm Understanding Model for End-to-End Generation of Playable Rhythm-Game Charts

**arXiv ID:** 2605.12135 | [PDF](https://arxiv.org/pdf/2605.12135v1)

**作者:** Joshua Opria `[一作]` `[通讯]` (Independent Researcher), Joshua Opria (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套全自动的 STRUM 音频到游戏图表的流水线，可将原始录音转化为 Clone Hero / YARG 可玩图表，支持鼓、吉他、低音、声乐和键盘。

**💡 创新点**

首次提出基于多阶段混合模型和操作包络评估的自动图表生成框架，并公开了30首歌曲的严格评测基准与完整代码。

**🔧 技术方法**

结合了源分离（htdemucs_6s）、CRNN 双阶段鼓 onset 检测、六模型集成分类器、pYIN 旋律追踪、Whisper 语音识别、谱峰键盘检测等深度学习与规则推断技术。

**📊 数据集**

使用了由 Clone Hero 社区自有的 30 首歌曲（涵盖朋克、金属、流行、摇滚、电子等八类）的音频和图表数据，此外通过 htdemucs_6s 产生的分离音轨进行训练。

**📈 对比分析**

在±100 ms 的容差下，对鼓、低音、吉他、声乐四个乐器分别获得 F1 分别为 0.838、0.694、0.651、0.539，鼓的性能已达 89 % 的理论上限。

**⚠️ 局限性**

受限于源分离质量、社区图表的时间量化误差以及多轨分配的规则限制，导致鼓蓝道、低音吉他轨道的准则精度低，声乐与键盘的对齐与层级仍待改进。

---

## 665. Metaphor Is Not All Attention Needs

**arXiv ID:** 2605.12128 | [PDF](https://arxiv.org/pdf/2605.12128v1)

**作者:** Olga Sorokoletova `[一作]` (Sapienza University of Rome), Daniele Nardi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 11130 | [OpenAlex ID](https://openalex.org/A5075651762)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过输入层消融、构造可解释的注意力向量、聚类分析和线性探针，研究了文学（诗歌）风格对大语言模型（LLM）安全对抗的影响，并评估了模型在识别文学格式与执行安全行为时的内部机制差异。

**💡 创新点**

创新点包括①证明文学 jailbreak 并非依赖单一诗歌手法，而是多种风格失常的累积效应；②提出一种跨层、跨阶段、跨功能词组的 72 维可解释注意力特征向量，用于比较不同文本格式；③通过聚类和线性探针显示模型能高精度识别诗歌 vs 散文，但对安全性判断信号弱，提示安全机制需关注风格诱发的内部处理偏移。

**🔧 技术方法**

使用的技术有：输入层消融实验（逐一移除/加入诗歌设备），注意力张量聚合（头层最大池化、生成阶段/层聚类/功能词组均值）、PCA 降维、线性探针（L1 正则化 Logistic Regression）、非线性探针（Gaussian SVC、MLP），以及基于层间相关性的层聚类。

**📊 数据集**

数据集包括 20 对诗歌/散文校准样本（涵盖四类风险领域）以及 2397 条大规模提示（1197 散文、1200 诗歌），原始散文来自 MLCommons AILuminate Benchmark，诗歌通过标准化元提示转换生成。安全标签由人工评估或三大开源 LLM 判定集生成。

**📈 对比分析**

比较方法：对全数据集做格式分类探针，分别在散文和诗歌子集做安全分类探针。格式探针在 Logistic Regression、SVC、MLP 上取得 0.985±0.006 的准确率；安全探针在两类下仅 0.66 左右，显著低于格式探针。聚类结果表明格式特征在注意力空间高度分离，而安全标签则几乎不产生可分离模式，说明安全性信息在注意力层中的信号弱。非线性模型性能与线性相近，说明信号主要在一次线性投影可捕获。

**⚠️ 局限性**

限制：①仅评估单一 7B LLM，缺乏跨模型验证；②注意力特征通过多重聚合（头层最大池化、功能词组平均）简化，可能丢失细粒度信息；③功能词组划分基于 LLM 判定，分组粒度和语义覆盖有限；④未深入后处理层（MLP、残差流）对安全行为的贡献；⑤实验使用固定生成长度、禁用推理模式，可能影响结果泛化。

---

## 666. NPAP: Network Partitioning and Aggregation Package for Python

**arXiv ID:** 2605.12137 | [PDF](https://arxiv.org/pdf/2605.12137v1)

**作者:** Marco Anarmo `[一作]` (Graz University of Technology), Sonja Wogrin `[通讯]` (Graz University of Technology)

**通讯引用:** 2028 | [OpenAlex ID](https://openalex.org/A5061878389)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一个可直接pip安装、与任何NetworkX图兼容的开源Python库NPAP，用于对电力网络等图形进行分区和聚合，从而降低空间复杂度；

**💡 创新点**

创新点在于将空间简化拆分为明确的分区和聚合两步，并采用策略模式实现高度可扩展的分区与聚合方法；

**🔧 技术方法**

利用Python生态（NetworkX、NumPy、SciPy、scikit‑learn、Plotly）实现分区/聚合策略与可视化；

**📊 数据集**

主要使用公开的欧盟跨国输电网数据（约6800节点、17500条边）以及PyPSA‑Eur模型数据；

**📈 对比分析**

与PyPSA原生空间聚类后对比，NPAP在同一网络上实现了可比的聚合效果，且在测试集上（小型到几千节点）保持良好性能（运行时间和内存消耗未显著增长）；

**⚠️ 局限性**

局限性包括目前聚合规模仍受内存约束，聚合策略相对固定（仅提供两种聚合配置），以及对非电网领域的通用性和更大规模网络的支持仍需进一步扩展。

---

## 667. ECTO: Exogenous-Conditioned Temporal Operator for Ultra-Short-Term Wind Power Forecasting

**arXiv ID:** 2605.12196 | [PDF](https://arxiv.org/pdf/2605.12196v1)

**作者:** Cao Yuan `[一作]` (Wuhan Polytechnic University), Junjun Wang `[通讯]` (Wuhan Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种名为ECTO的框架，通过物理约束的层级变量选择和基于外生条件的混合专家修正，实现了超短期风电预测。

**💡 创新点**

创新点在于将外生变量建模拆分为两步：PGVS使用sparsemax实现物理分组的稀疏选择，ECRR则通过混合专家动态校准预测，充分利用气象变量的条件依赖。

**🔧 技术方法**

技术方法包括层级卷积时序分词、交叉注意力、sparsemax稀疏化、混合专家门控以及多尺度修正机制。

**📊 数据集**

实验使用三个不同规模和气候的风电场数据集：WF1（99 MW）、WF4（66 MW）和新疆（200 MW），共包含11–13个外生变量。

**📈 对比分析**

与九个基准模型比较，ECTO在所有数据集上均以最低MSE领先，平均相对提升从2.2%到5.2%，并在不同预测时隙均保持优势。

**⚠️ 局限性**

主要局限包括对物理分组的先验依赖、固定的专家数量、缺乏空间（机组间）信息，以及仅验证在风电领域的通用性。

---

## 668. DexTwist: Dexterous Hand Retargeting for Twist Motion via Mixed Reality-based Teleoperation

**arXiv ID:** 2605.12182 | [PDF](https://arxiv.org/pdf/2605.12182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 669. Sure-almost-sure and Sure-limit-sure Window Mean Payoff in Markov Decision Processes

**arXiv ID:** 2605.12191 | [PDF](https://arxiv.org/pdf/2605.12191v1)

**作者:** Pranshu Gaba `[一作]` (Tata Institute of Fundamental Research), Shibashis Guha `[通讯]` (Tata Institute of Fundamental Research)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5089838447)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在马尔可夫决策过程（MDP）中解决窗口均值支付目标下的“确定-几乎确定”和“确定-极限确定”问题，给出求解算法、复杂度分析与策略记忆量上界；

**💡 创新点**

证明确定-几乎确定性与确定-极限确定性在窗口均值支付目标下不等价，提出新的算法框架以同时满足确定和几乎确定约束，并给出精确的时间与空间复杂度；

**🔧 技术方法**

利用窗口均值支付目标的前缀无关性、子MDP与子MDP闭包、最大终端组件分解、以及对策略的构造与递归求解；

**📊 数据集**

本研究为理论算法设计，未使用实验数据集；

**📈 对比分析**

通过理论复杂度证明，固定窗口长度问题的算法为P（若阈值以一元表示），而无界窗口长度问题为NP∩coNP；算法实现需多次递归调用子算法，时间复杂度为多项式级别；

**⚠️ 局限性**

主要局限在于实现难度高、算法对输入参数（窗口长度）极度敏感，且对大规模MDP的实际计算仍然存在高内存与时间消耗，且对随机性分布假设有严格前提。

---

## 670. Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding

**arXiv ID:** 2605.12185 | [PDF](https://arxiv.org/pdf/2605.12185v1)

**作者:** Yigeng Zhou `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 116125 | [OpenAlex ID](https://openalex.org/A5100336796)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段的动态认知调和解码框架，用于缓解大型语言模型在检索增强生成场景中的上下文-记忆冲突问题。

**💡 创新点**

创新点在于：①基于注意力图的上下文忠诚度预测器，可在解码前判断冲突程度；②动态路由机制将无冲突输入直接走贪婪解码，冲突输入走自适应对比解码，从而兼顾准确性与效率。

**🔧 技术方法**

技术主要包括：注意力图提取与特征计算、单层MLP冲突分类器、可调参数α的自适应对比解码公式，以及基于贪婪与对比解码的动态路由。

**📊 数据集**

使用了四大主流LLM（Llama2‑7B、Llama2‑13B、Llama3‑8B、Mistral‑7B）和六个QA数据集（NQ、SQuAD、TriviaQA、Counterfacts、NQ‑Swap、ConflictKG）进行评测。

**📈 对比分析**

与Greedy、CAD、COIECD、ADACAD等基线对比，DCRD在高冲突场景中提升约20%准确率，在低冲突场景中提升约15%，并且在冲突比例变化、噪声上下文、跨域迁移等实验中表现最稳健。

**⚠️ 局限性**

局限性包括：仅在7B/13B规模模型上验证，未测试更大或聊天模型；使用单层MLP分类器，可能在复杂场景下不足；对不同检索系统与知识库的适配性尚未深入探索。

---

## 671. Correcting Selection Bias in Sparse User Feedback for Large Language Model Quality Estimation: A Multi-Agent Hierarchical Bayesian Approach

**arXiv ID:** 2605.12177 | [PDF](https://arxiv.org/pdf/2605.12177v1)

**作者:** Andrea Morandi `[一作]` (Cisco Systems, Inc.), Mahesh Viswanathan `[通讯]` (Cisco Systems, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个多代理层级贝叶斯框架，用于在稀疏且存在选择偏差的用户反馈流中估计大型语言模型的真实质量。

**💡 创新点**

创新点在于：① 通过 UMAP + HDBSCAN 自动生成语义层级聚类；② 采用两阶段 Beta‑Binomial 层级模型同时推断选择概率与质量；③ 关键是仅基于反馈通道的运营者先验（正反馈率与负反馈比），打破了可识别性瓶颈，使得无标签数据即可实现准确校正。

**🔧 技术方法**

所用技术包括：UMAP 降维、HDBSCAN 密度聚类、层级 Beta‑Binomial 贝叶斯模型、NUTS 采样、后验预测检查、留一交叉验证（LOO）、在线漂移检测与动态重校准。

**📊 数据集**

实验使用公开的 UltraFeedback 数据集，以 GPT‑4 评分作为代理真值，在此基础上模拟选择偏差后验证模型性能。

**📈 对比分析**

比较方法包括 Naive、IPW、Basic、Enhanced、Hierarchical‑Sentiment、Hierarchical‑Informed、Corrected‑Global 等。Hierarchical‑Informed 在绝对误差≤13个百分点、95% 置信区间覆盖率100%、对不同偏差强度鲁棒，并在 LOO 排名第一；其他模型误差显著，覆盖率不佳。

**⚠️ 局限性**

局限性：① 需满足聚类内忽略性假设；② 对嵌入器与聚类质量敏感；③ 需要初始反馈；④ 对恶意/欺诈反馈未建模；⑤ 运营者先验需手动校准；⑥ 无法直接验证关键假设；⑦ 在极端稀疏或高噪声场景下仍可能不稳定。

---

## 672. Expected Batch Optimal Transport Plans and Consequences for Flow Matching

**arXiv ID:** 2605.12174 | [PDF](https://arxiv.org/pdf/2605.12174v1)

**作者:** Samuel Boïté `[一作]` (ENS Paris), Kimia Nadjahi `[通讯]` (ENS Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文研究了在流匹配中使用随机小批量最优传输的期望传输计划，并给出了其一致性、误差率和对速度场可测性（可流性）的证明，进一步探讨了批大小与数值积分之间的权衡。

**💡 创新点**

创新点在于将期望批量 OT 作为人群层级对象进行严格定义和分析，提供了在半离散设定下的成本与计划收敛率（O(k^-1) 或 O(k^-1/2) 等），证明了其诱导的速度场是局部 Lipschitz 并能生成唯一的流，并系统评估了批大小与积分步数的相互作用。

**🔧 技术方法**

主要技术包括随机样本理论、半离散最优传输、经验过程界、速度场的后验表示、Euler 积分误差分析以及在神经网络流匹配中的逼近学习。

**📊 数据集**

实验数据集包括高维高斯-离散模拟、两原子可解析模型、CIFAR-10 与 SVHN 图像数据集，用以验证收敛速率与批大小对生成质量的影响。

**📈 对比分析**

与独立配对和精确 OT 进行比较，结果显示在低 NFE（如 10 次 Euler 步）下增大 OT 批大小能显著提升 FID，但在高 NFE 时收益减弱甚至反向，且批大小提升需要显著训练成本。

**⚠️ 局限性**

主要限制在于理论结果仅适用于半离散源-目标配置，未对神经网络逼近的速度场做严格分析，且未给出一般 n 与 k 的积分误差界，未来工作需扩展至更一般配对、学习模型及其他数值积分方法。

---

## 673. On What We Can Learn from Low-Resolution Data

**arXiv ID:** 2605.12168 | [PDF](https://arxiv.org/pdf/2605.12168v1)

**作者:** Theresa Dahl Frehr `[一作]` (Technical University Of Denmark), Tommy Sonne Alstrøm `[通讯]` (Technical University Of Denmark)

**通讯引用:** 1246 | [OpenAlex ID](https://openalex.org/A5064146030)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了混合分辨率训练，即训练数据由少量高分辨率样本与大量低分辨率样本组成，评估模型在高分辨率输入上的性能；

**💡 创新点**

提出了基于KL散度的理论分析，量化高低分辨率样本对参数分布影响的相对与绝对差异，并给出上限/下限；

**🔧 技术方法**

使用ViT与CNN（ResNet-18）模型进行实验，并利用损失函数的局部线性近似、采样率映射及上采样操作；

**📊 数据集**

在CIFAR‑10、CIFAR‑100和AudioMNIST三个基准数据集上进行实验；

**📈 对比分析**

通过“Subset”、“Ratio”、“Downsampled”和“Size”四种实验对比，显示在高分辨率数据稀缺时加入低分辨率数据可显著提升准确率，收益随高分辨率比例增加而递减；

**⚠️ 局限性**

局限性在于理论依赖损失对输入的局部线性假设，对极端下采样的适用性有限，实验规模受限于小型基准数据集，未覆盖更大、更异构的真实数据场景。

---

## 674. EchoTracker2: Enhancing Myocardial Point Tracking by Modeling Local Motion

**arXiv ID:** 2605.12140 | [PDF](https://arxiv.org/pdf/2605.12140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 675. From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation

**arXiv ID:** 2605.12167 | [PDF](https://arxiv.org/pdf/2605.12167v1)

**作者:** Yajie Li `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**通讯引用:** 82176 | [OpenAlex ID](https://openalex.org/A5100425671)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用视频生成模型产生的未来视觉轨迹，并通过预训练的混合逆动力学模型（Mixture of Latent Actions，MoIDM）将其转换为可执行的潜在动作表示，再由扩散动作头生成实际控制指令，从而实现想象未来与机器人执行之间的桥接。

**💡 创新点**

创新点包括：① 将逆动力学模型作为动作接口直接从生成的视频中提取潜在动作；② 采用三种模态（语义、深度、光流）分别训练逆动力学模型，实现信息互补；③ 通过联合微调使预训练的逆动力学模型与扩散动作头共同适应生成的未来视觉，提升控制稳定性和泛化能力。

**🔧 技术方法**

核心技术包括：Stable Video Diffusion（SVD）用于生成未来视频；基于ViT的时空 Transformer 与 VQ 代码簿实现逆动力学模型；三种模态的自监督监督（Depth Anything、SAM2、CoTracker3）；扩散 Transformer（Diffusion Transformer）配合流匹配目标用于动作解码；三阶段训练流程（视频生成微调、逆动力学预训练、端到端微调）。

**📊 数据集**

在大规模机器人学习数据集上预训练逆动力学模型，随后在 CALVIN、LIBERO、LIBERO-Plus 三个仿真基准以及真实 UR5e 机器人上进行评估；训练集包括 CALVIN ABC‑D、LIBERO‑90 以及实验室收集的 1000 条演示轨迹。

**📈 对比分析**

与现有 VLA（如 OpenVLA、π_0.5）、世界模型方法（VPP）以及扩散策略等基线相比，MoLA 在 CALVIN、LIBERO、LIBERO‑Plus 三大基准上均获得最高平均成功率，特别是在 LIBERO‑Plus 上领先 OpenVLA‑OFT+ 13.2%；在真实机器人任务中也显著优于 Diffusion Policy 和 OpenVLA，显示出更好的任务完成率和泛化能力。

**⚠️ 局限性**

局限性主要体现在：① 依赖于强大的视频生成模型和预训练逆动力学模型，对资源要求较高；② 生成视频的单步推理仍可能产生误差，导致潜在动作不精确；③ 当前方法在极端噪声或视觉遮挡环境下的鲁棒性尚待进一步验证；④ 需要在大规模多模态数据上进行预训练，数据获取成本仍然高。

---

## 676. It's Not the Size: Harness Design Determines Operational Stability in Small Language Models

**arXiv ID:** 2605.12129 | [PDF](https://arxiv.org/pdf/2605.12129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 677. X-Imitator: Spatial-Aware Imitation Learning via Bidirectional Action-Pose Interaction

**arXiv ID:** 2605.12162 | [PDF](https://arxiv.org/pdf/2605.12162v1)

**作者:** Kai Xiong `[一作]` (Shanghai Jiao Tong University), Cewu Lu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15757 | [OpenAlex ID](https://openalex.org/A5010726528)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了一种双路径模仿学习框架X‑Imitator，通过在动作生成和目标姿态预测之间实现双向交互，提升机器人在空间感知与动作执行上的协同能力。

**💡 创新点**

创新点在于将姿态与动作预测作为互相条件化的闭环循环，实现了内部前向模型；并将其设计为可插拔、低推理开销的模块，兼容多种现有视觉运动学策略。

**🔧 技术方法**

使用了双路径（action/pose）结构、条件投影与特征融合（Add、Concat、FiLM等）、交叉注意力、扩散模型预测动作与姿态、联合损失训练以及条件轨迹长度调节等技术。

**📊 数据集**

在24个仿真任务（Adroit、DexArt、MetaWorld、RoboTwin 2.0）和3个真实世界任务（Hang Mug、Pour Balls、Arrange Toy Truck）上进行评估。

**📈 对比分析**

与基线DP3、RISE、ACT及其多种变体对比，X‑Imitator在仿真环境中平均成功率从DP3的54.6%提升至63.8%，在RoboTwin 2.0上同样表现显著提升；在真实世界任务中，X‑RISE将Hang Mug任务成功率从60%提升至80%，并在Pour Balls任务中显著降低方差。

**⚠️ 局限性**

局限性包括：只能处理可用SE(3)姿态描述的刚体或可拆分为刚体部件的结构化对象；依赖外部姿态估计器，可能在严重遮挡、运动模糊或光照不佳时失效；假设目标对象已知，难以应对无结构化场景。

---

## 678. Cross-Modal-Domain Generalization Through Semantically Aligned Discrete Representations

**arXiv ID:** 2605.12145 | [PDF](https://arxiv.org/pdf/2605.12145v1)

**作者:** Souptik Sen `[一作]` (Hannover Medical School), Zahra Ahmadi `[通讯]` (Hannover Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 CoDAAR 框架，利用跨模态离散对齐和重构实现多模态表示的跨模态可迁移性与模态特异性兼顾。

**💡 创新点**

创新点在于：① 模态专属代码表与索引级对齐，消除共享代码表竞争；② 通过离散时间对齐 (DTA) 在时间维度上融合多模态信息；③ 级联语义对齐 (CSA) 通过层次式索引迁移实现跨模态语义一致。

**🔧 技术方法**

采用 VQ‑VAE 编码器+离散代码表、CPC 对比学习、指数移动平均 (EMA) 更新、对齐损失（DQA、CSA）、重构和承诺损失等自监督技术。

**📊 数据集**

主要使用 VGGSound‑AVEL（音视频）以及结合 MSCOCO、Clotho 的文本数据集进行预训练与评估。

**📈 对比分析**

在跨模态事件分类（AVE）、事件定位（AVVP）、零样本检索（MSCOCO、Clotho）、视频分割（AVSBench‑S4）等任务上，与 CMCM、CODIS、TURN、DCID、MICU 等 SOTA 方法相比，CoDAAR 在多任务、多域迁移下均获得更高的精度、召回和 IoU，尤其在跨模态定位和零样本检索上表现突出。

**⚠️ 局限性**

局限性包括：① 需要多模态配对数据，未验证对弱配对或噪声配对的鲁棒性；② 代码表数量与维度对性能敏感，需进一步自动化调优；③ 仅在音视频‑文本三模态上验证，扩展到更多模态（如点云、传感器）仍需研究。

---

## 679. BoolXLLM: LLM-Assisted Explainability for Boolean Models

**arXiv ID:** 2605.12139 | [PDF](https://arxiv.org/pdf/2605.12139v1)

**作者:** Du Cheng `[一作]` (Fidelity Investments), Xin Wang `[通讯]` (Fidelity Investments)

**通讯引用:** 1927 | [OpenAlex ID](https://openalex.org/A5100710332)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 BoolXllm 框架，将 LLM 集成到 BoolXai 规则学习的特征选择、阈值推荐和规则解释三阶段，以提升可解释性。

**💡 创新点**

在特征选择和阈值分箱中利用 LLM 提供语义驱动的选择，并结合 LLM 对规则的全局与局部自然语言解释，保证解释可信且可读。

**🔧 技术方法**

使用 LLM（如 GPT‑5.2）进行结构化提示，基于 BoolXai 的表达式规则学习和本地搜索优化，并采用句向量聚类进行规则压缩与解释。

**📊 数据集**

在 UCI Bank Marketing 数据集上进行评估。

**📈 对比分析**

与基线 BoolXai 全特征/标准分箱比较，BoolXllm 在特征减少约20%后保持 86% 平衡准确率，阈值更语义化，无性能损失；解释性通过规则复杂度、阈值可读性和定性评估提升。

**⚠️ 局限性**

仍需人工校验 LLM 输出，提示设计敏感；阈值生成受限于 LLM 先验知识；模型处于实验阶段，需更多跨域验证。

---

## 680. Disentangled Sparse Representations for Concept-Separated Diffusion Unlearning

**arXiv ID:** 2605.12122 | [PDF](https://arxiv.org/pdf/2605.12122v1)

**作者:** Hyeonjin Kim `[一作]` (Yonsei University), Dong-Jun Han `[通讯]` (Yonsei University)

**通讯引用:** 486 | [OpenAlex ID](https://openalex.org/A5101403023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对文本到图像扩散模型的概念去学，提出 SAEParate 通过稀疏自编码器实现概念特异性特征抑制。

**💡 创新点**

引入概念感知对比损失和 GeLU 加强的编码器，显著提升稀疏特征的概念分离度。

**🔧 技术方法**

稀疏自编码器、概念对比学习、GeLU 非线性、交叉注意力加权池化与特征重要性评分。

**📊 数据集**

UnlearnCanvas 基准集（50 版式、20 对象）以及 I2P 垄断性裸体去学测试。

**📈 对比分析**

与九种现有扩散去学方法及 SAeUron 对比，SAEParate 在单属性与联合属性去学上均获得最高 UA、IRA、CRA 并保持低 FID，尤其在风格-对象联合去学上显著超越对手。

**⚠️ 局限性**

需监督的概念标签，且实验仅覆盖图像生成，缺乏无监督/视频生成等更广泛场景。

---

## 681. Delay-Empowered Causal Hierarchical Reinforcement Learning

**arXiv ID:** 2605.12261 | [PDF](https://arxiv.org/pdf/2605.12261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 682. Do Enterprise Systems Need Learned World Models? The Importance of Context to Infer Dynamics

**arXiv ID:** 2605.12178 | [PDF](https://arxiv.org/pdf/2605.12178v1)

**作者:** Jishnu Sethumadhavan Nair `[一作]` (ServiceNow), Sai Rajeswar `[通讯]` (ServiceNow)

**通讯引用:** 1246 | [OpenAlex ID](https://openalex.org/A5041629023)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对企业系统的状态转移预测进行研究，比较了离线训练的世界模型和在推理时通过检索配置实时发现动态的发现代理。

**💡 创新点**

提出CascadeBench基准并证明在配置漂移环境下，学习模型会退化，而发现代理通过在推理时检索业务规则能保持鲁棒性。

**🔧 技术方法**

利用大型语言模型（Claude、Gemma、Qwen等）、LoRA微调、检索‑推理循环以及与业务规则交互的发现代理实现模型推理。

**📊 数据集**

构建Enterprise Gym与CascadeBench，使用ServiceNow的合成业务规则和真实记录，包含约27k条转移样本，涵盖6个行业和3个规模。

**📈 对比分析**

在CascadeBench上比较提示、Fine‑tuned和Discovery代理，提示+规则的IoU最高；Fine‑tuned在分布内表现优秀但在迁移时下降；Discovery代理在无规则提示下可恢复20‑30点IoU，整体表明发现代理更鲁棒。

**⚠️ 局限性**

假设业务规则可读；对大型模型检索不稳定，可能需要微调；仅评估单一平台ServiceNow，Tier3（执行顺序相关）效果有限。

---

## 683. From Image Hashing to Scene Change Detection

**arXiv ID:** 2605.12259 | [PDF](https://arxiv.org/pdf/2605.12259v1)

**作者:** Anh-Kiet Duong `[一作]` (La Rochelle University), Jean-Michel Carozza `[通讯]` (La Rochelle University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于图像哈希的场景变化检测框架HashSCD，能够实现全局比对与局部定位。

**💡 创新点**

创新点在于引入块级哈希与类似XOR的聚合操作，使得在Hamming空间即可完成变化检测与定位，且训练为无监督对比学习。

**🔧 技术方法**

采用无监督对比学习的软哈希、基于VGG16的特征提取、块级哈希投影以及递归绝对差异聚合等技术。

**📊 数据集**

在VL‑CMU‑CD、PCD（TSUNAMI和GSV子集）以及标准检索基准Oxford Flowers、Food101和NUS‑WIDE数据集上进行实验。

**📈 对比分析**

与连续特征、监督/零样本方法对比，HashSCD在变化检测上实现了与现有方法相近甚至优于的F1分数，在检索上在细粒度数据上优于主流哈希算法；同时显著降低了存储和计算成本。

**⚠️ 局限性**

局限在于对视角偏移、相机运动或对齐误差的鲁棒性不足，且在复杂场景下仍需进一步改进。

---

## 684. H3D-MarNet: Wavelet-Guided Dual-Path Learning for Metal Artifact Suppression and CT Modality Transformation for Radiotherapy Workflows

**arXiv ID:** 2605.12252 | [PDF](https://arxiv.org/pdf/2605.12252v1)

**作者:** Mubashara Rehman `[一作]` (Università degli Studi di Udine), Christian Micheloni `[通讯]` (Università degli Studi di Udine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出H3D-MarNet，完成kVCT金属伪影抑制与kVCT→MVCT域转换

**💡 创新点**

采用两阶段波形预处理+双路径CNN-Transformer编码融合，实现高频伪影先行抑制与长程体素依赖建模

**🔧 技术方法**

多尺度离散小波预处理、CNN+Transformer双编码器、注意力融合与深度监督解码、L1+SSIM+感知损失等技术

**📊 数据集**

使用Aviano IRCCS 52例口咽癌患者的kVCT与MVCT配对扫描数据集

**📈 对比分析**

与传统、CNN、Transformer等SOTA方法对比，H3D-MarNet在artifact-affected和全数据集上PSNR/SSIM均位列首位（PSNR≈28dB/SSIM≈0.72；全集PSNR≈30dB/SSIM≈0.76）

**⚠️ 局限性**

模型参数与计算量较大，且在极端金属伪影下仍存在细节失真，需进一步压缩与自监督扩展

---

## 685. Social Welfare under Heterogeneous Time Preferences

**arXiv ID:** 2605.12251 | [PDF](https://arxiv.org/pdf/2605.12251v1)

**作者:** Sarvin Bahmani `[一作]` (University of Liverpool), Ashutosh Trivedi `[通讯]` (University of Liverpool)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在多主体异质时间偏好下的决策问题，提出异折扣MDP模型并求解社会福利最优策略；

**💡 创新点**

首次将不同主体的折扣因子纳入MDP，并证明在此情形下记忆策略必不可少，同时给出可在多项式时间内构造的纯有限记忆计数策略；

**🔧 技术方法**

利用折扣MDP理论、动态规划、策略可视化、NP‑hardness归约和多项式时间构造等技术手段；

**📊 数据集**

实验使用随机生成的MDP实例，规模从几到10^5个状态，主体数从2到101，折扣因子间距可调；

**📈 对比分析**

与传统纯/记忆无关策略对比，实验表明算法在状态数、主体数和折扣比上保持线性/多项式增长，且在大规模实例下仍能快速完成；

**⚠️ 局限性**

局限在于对折扣因子间距作“合理间距”假设，未考虑多目标或不确定性，以及更一般的时间偏好结构。

---

## 686. No Action Without a NOD: A Heterogeneous Multi-Agent Architecture for Reliable Service Agents

**arXiv ID:** 2605.12240 | [PDF](https://arxiv.org/pdf/2605.12240v1)

**作者:** Zixu Yang `[一作]` (Shanghai Jiao Tong University), Kai Yu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21309 | [OpenAlex ID](https://openalex.org/A5100758006)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 NOD（Navigator‑Operator‑Director）多代理架构，将任务状态外部化为结构化表，并在关键操作前加入外部监督，以提升长周期服务代理的可靠性。

**💡 创新点**

创新点在于将任务状态显式化为结构化状态表，并通过三角色分工（Navigator、Operator、Director）实现结构化控制与选择性监督，从而解决隐式状态脆弱性和自我纠正不可靠的问题。

**🔧 技术方法**

使用大型语言模型（如 Qwen、GPT 系列）配合结构化状态模板、关键操作的前置验证和回溯修正机制，构建多代理协同框架。

**📊 数据集**

使用 τ²‑Bench 的 Retail 和 Airline 两大领域数据集进行实验。

**📈 对比分析**

与单模型基线（Vanilla LLM、Self‑Reflection、Self‑aware Abstention）及多模型基线（Multi‑Agent Debate、AutoGen）对比，NOD 在 Success Rate 和 Critical Action Precision 上分别提升高达 23.7 与 25.8 分，显著降低政策违规、工具幻觉和意图不一致率。

**⚠️ 局限性**

局限性包括每轮全量 JSON 状态重建导致令牌消耗高、状态不完整时易出现不一致，以及对状态模板设计和监督阈值敏感。

---

## 687. Reconnecting Fragmented Citation Networks with Semantic Augmentation

**arXiv ID:** 2605.12263 | [PDF](https://arxiv.org/pdf/2605.12263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 688. No More, No Less: Task Alignment in Terminal Agents

**arXiv ID:** 2605.12233 | [PDF](https://arxiv.org/pdf/2605.12233v1)

**作者:** Sina Mavali `[一作]` (CISPA Helmholtz Center for Information Security), Lea Schönherr `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套任务对齐基准，用于评估终端代理在面对环境中有用提示和无关干扰信息时，能否仅使用必要线索完成任务。

**💡 创新点**

创新点在于将终端任务抽象化并将必需提示与伪造干扰信息放在同一环境表面，分离任务完成能力与信息选择能力。

**🔧 技术方法**

使用了Prompt注入防御技术、LLM评估、自动化检测以及构建的Cue与Distractor标记机制。

**📊 数据集**

数据集为从Terminal‑Bench 2.1转换得到的89个对齐任务，包含各种提示与干扰。

**📈 对比分析**

对比十种最先进终端代理和六种Prompt注入防御，发现最强能力模型GPT‑5.5在任务对齐仅23%，而Claude Opus 4.7对齐达72%；防御虽然降低干扰执行，但往往也抑制了必要线索，整体任务完成率下降。

**⚠️ 局限性**

局限在于仅覆盖终端代理、干扰设计有限、缺乏对更复杂环境的验证以及缺少训练式对齐方法。

---

## 689. Learning Ego-Centric BEV Representations from a Perspective-Privileged View: Cross-View Supervision for Online HD Map Construction

**arXiv ID:** 2605.12218 | [PDF](https://arxiv.org/pdf/2605.12218v1)

**作者:** Daniel Lengerer `[一作]` (Technical University of Applied Sciences Augsburg), Carsten Markgraf `[通讯]` (Technical University of Applied Sciences Augsburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种跨视角监督（Cross‑View Supervision，CVS）方法，在训练阶段用高度优越的空中视角为摄像头生成的BEV编码器提供结构化的监督，从而提升在线HD地图构建的全局几何一致性。

**💡 创新点**

创新点在于：①仅在训练期间使用空中图像作为教师，实现无额外推理成本的结构监督；②通过特征空间对齐和通道级仿射适配器，克服视角差异，实现密集的BEV特征级监督；③在保持原始摄像头推理架构不变的前提下显著提升长距离地图预测精度。

**🔧 技术方法**

技术包括：BEVFormer‑style encoder、ResUNet 空中编码器（教师）、特征归一化+通道仿射适配器、MSE对齐损失、线性CKA与R² 评价指标，以及在 StreamMapNet 框架下的联合训练。

**📊 数据集**

使用 AID4AD 数据集（nuScenes 与对应的空中图像配准）进行训练与评估。

**📈 对比分析**

与 StreamMapNet 基线对比，在 60×30 m 区域 mAP 提升 3.9 点（≈11%），在 100×50 m 区域提升 9.9 点（≈44%），显示在远距离场景中性能显著提升。

**⚠️ 局限性**

局限性在于：①仅依赖已配准的空中图像，训练数据来源受限；②跨视角监督的效果受教师架构与对齐方式限制，进一步提升需改进传递机制；③目前仅在 nuScenes/AID4AD 上验证，缺乏多数据集泛化验证。

---

## 690. NARA: Anchor-Conditioned Relation-Aware Contextualization of Heterogeneous Geoentities

**arXiv ID:** 2605.12276 | [PDF](https://arxiv.org/pdf/2605.12276v1)

**作者:** Jina Kim `[一作]` (University of Minnesota), Yao-Yi Chiang `[通讯]` (University of Minnesota)

**通讯引用:** 3038 | [OpenAlex ID](https://openalex.org/A5045786247)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个统一的自监督框架 NARA，用于学习矢量地理实体（点、线、多边形）的上下文化表征，并在多种下游任务（交通速度预测、建筑功能分类、POI 预测）上直接使用预训练模型。

**💡 创新点**

创新点包括：①将语义、几何和空间关系三者统一建模；②引入基于 anchor 的关系感知对比学习和半变差正则化，实现对度量距离与拓扑关系共同调节的空间自相关建模；③利用空间感知注意力监督，使得注意力矩阵反映实际的空间邻接与拓扑关系。

**🔧 技术方法**

技术方法包括 Transformer 结合几何编码器 Poly2Vec、BERT 语义编码、遮蔽语义建模、空间感知注意力、anchor 条件对比学习、半变差正则化以及多任务联合训练。

**📊 数据集**

使用 OpenStreetMap（纽约、Singapore）作为矢量数据；Traffic Speed 采用 Uber Movement 交通速度；Building Function 使用 Singapore 政府建筑用地标签；Next POI 采用 Foursquare 纽约和 Singapore 的签到数据。

**📈 对比分析**

与多种基线（图网络、几何编码、语义+几何/语义+度量、CityFM、POI‑Enhancer 等）对比，NARA 在所有三类任务均取得显著提升（Traffic Speed MAE 3.05、Building Macro‑F1 72.82、POI Hit@1 最高 8.165±0.096），并且在不同数据集上保持一致性，说明模型具有良好的泛化能力。

**⚠️ 局限性**

局限性包括：①对稀有几何类型（如罕见道路类型）仍缺乏足够样本；②对线性实体（道路）学习主要依赖邻域信息，关系正则化对线实体效果有限；③模型未在多模态（图像+矢量）或大规模全球数据上进行验证，未来需扩展到更大规模和更复杂的空间关系。

---

## 691. Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models

**arXiv ID:** 2605.12258 | [PDF](https://arxiv.org/pdf/2605.12258v1)

**作者:** Runhe Lai `[一作]` (Sun Yat-sen University), Ruixuan Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 29145 | [OpenAlex ID](https://openalex.org/A5100431254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个无需训练的、基于指令嵌入的多模态大语言模型对象错觉检测方法。

**💡 创新点**

首次发现并利用指令嵌入能过滤视觉误导信息，并结合校准局部得分与上下文一致性得分形成全新指令透镜分数。

**🔧 技术方法**

使用 Logit Lens 对指令和视觉嵌入进行投影，构建校准置信度和上下文一致性度量，融合视觉基础分数。

**📊 数据集**

在 MSCOCO、Objects365、POPE 与 CLEVR 四个基准上进行评测。

**📈 对比分析**

与 NLL、Entropy、SVAR、GLSIM、ContextLens 等多种基线对比，InsLen 在 AUROC/AUPR 上均优于所有方法，尤其在 MSCOCO 上提升 7.7%AUROC。

**⚠️ 局限性**

依赖 Logit Lens 的投影可能导致语义偏差，且对指令长度敏感，计算开销相对较高。

---

## 692. Why Conclusions Diverge from the Same Observations: Formalizing World-Model Non-Identifiability via an Inference

**arXiv ID:** 2605.12255 | [PDF](https://arxiv.org/pdf/2605.12255v1)

**作者:** Toru Takahashi `[一作]` (Doshisha University), Toru Takahashi `[通讯]` (Doshisha University)

**通讯引用:** 3994 | [OpenAlex ID](https://openalex.org/A5065967819)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两级非可辨识性框架，用以解释在共享观测下出现的结论分歧。

**💡 创新点**

创新点在于：① 将推理过程拆解为四维推理概况 (Reference, Exploration, Stabilization, Horizon)；② 在计算、观测与协作约束下将其映射到三大基底 (抽象/具体、可外化性、秩序/自由)；③ 将该框架与深度表示学习的结构对应起来。

**🔧 技术方法**

采用信息理论（压缩与重构失真）、通信成本（描述长度）以及稳定性-多样性权衡（熵、探索与稳定阈值）等理论工具，对推理概况进行建模与分析。

**📊 数据集**

未使用标准数据集；通过对 AI 监管辩论的案例研究，演示框架如何解释不同利益相关者的立场差异。

**📈 对比分析**

由于是理论框架，未进行量化实验或对比；仅通过案例说明框架能够将分歧定位到推理概况与世界模型的非可辨识性，并提供可能的调和路径。

**⚠️ 局限性**

局限性包括：① 缺乏经验验证与定量评估；② 需要从对话或决策记录中提取推理概况的技术尚未实现；③ 依赖的计算、观测、协作约束假设在不同情境下的适用性仍待探讨。

---

## 693. UHR-Micro: Diagnosing and Mitigating the Resolution Illusion in Earth Observation VLMs

**arXiv ID:** 2605.12237 | [PDF](https://arxiv.org/pdf/2605.12237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 694. Combining On-Policy Optimization and Distillation for Long-Context Reasoning in Large Language Models

**arXiv ID:** 2605.12227 | [PDF](https://arxiv.org/pdf/2605.12227v1)

**作者:** Miguel Moura Ramos `[一作]` (Instituto Superior Técnico, Universidade de Lisboa), André F. T. Martins `[通讯]` (Instituto Superior Técnico, Universidade de Lisboa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两阶段长上下文推理后训练方案：先用离线监督（SFT）预热模型，然后在此基础上通过基于GRPO的强化学习并加入教师指导的OPD（称为dGRPO）进行稠密教师引导的 on‑policy 优化；

**💡 创新点**

创新点在于将稠密教师引导（OPD）与稀疏奖励的GRPO融合为单一目标，并构建了覆盖30种自然语种+15编程语言的多语言合成长上下文数据集LongBlocks，提供了更通用的长上下文后训练路径；

**🔧 技术方法**

采用的技术包括：离线监督训练（SFT/KD）、GRPO、OPD、dGRPO、RoPE位置编码扩展到128K、混合精度、序列打包、文档掩码等；

**📊 数据集**

使用的数据集为：LongBlocks（193,219问答对，跨30自然语言+15编程语言）以及短上下文训练集Nemotron-Post-Training-Dataset-v2，用于混合训练；

**📈 对比分析**

实验对比基线模型、SFT‑only、外部长上下文模型，结果显示短上下文指标保持甚至略升，而长上下文指标大幅提升（128K时RULER从11%提升至44.5%，LongBench从34.4%提升至39.6%），证明方案在保持短上下文性能的同时显著提升长上下文表现；

**⚠️ 局限性**

局限性包括：对教师模型质量高度依赖、β调参需经验、RL对长序列稀疏奖励仍需大量采样、实验仅在Qwen3-1.7B及与4B/32B教师验证，尚未全面评估不同规模模型的泛化性。

---

## 695. TriBand-BEV: Real-Time LiDAR-Only 3D Pedestrian Detection via Height-Aware BEV and High-Resolution Feature Fusion

**arXiv ID:** 2605.12220 | [PDF](https://arxiv.org/pdf/2605.12220v1)

**作者:** Mohammad Khoshkdahan `[一作]` (Karlsruhe Institute of Technology), Alexey Vinel `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 7556 | [OpenAlex ID](https://openalex.org/A5003670494)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于三带BEV编码的LiDAR‑only实时三维目标检测框架；

**💡 创新点**

创新点在于三带反射率编码、区域注意力模块与高分辨率双向特征融合，显著提升小目标检测性能；

**🔧 技术方法**

采用TriBand‑BEV投影、区域注意力B‑A2C2f块、双向多分辨率颈部、DFL与旋转IoU损失、IQR后处理等技术；

**📊 数据集**

使用KITTI数据集（64波束Velodyne HDL64）进行训练与验证；

**📈 对比分析**

与Complex‑YOLO对比，易/中/难三个难度级别的行人BEV AP分别为58.7%/52.6%/47.2%，速度达49 FPS，性能显著优于Baseline；

**⚠️ 局限性**

局限在高度估计仍受限，长距离小目标检测仍不稳健，模型规模仍可进一步压缩。

---

## 696. Goal-Oriented Reasoning for RAG-based Memory in Conversational Agentic LLM Systems

**arXiv ID:** 2605.12213 | [PDF](https://arxiv.org/pdf/2605.12213v1)

**作者:** Jiazhou Liang `[一作]` (University of Toronto), Scott Sanner `[通讯]` (University of Toronto)

**通讯引用:** 6654 | [OpenAlex ID](https://openalex.org/A5028174137)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Goal-Mem，一个基于目标导向的向后链式推理框架，用于提升 RAG（检索增强生成）记忆增强型 LLM 代理在长序列对话中的答案可靠性。

**💡 创新点**

创新点在于：① 将用户发问转化为自然语言逻辑（NL‑Logic）目标；② 通过 LLM 自动拆分为原子子目标并进行目标导向检索；③ 对检索到的事实进行可验证的统一（类型一致、变量一致、逻辑蕴含），从而避免无关或不完整信息导致的推理错误；④ 在失败时递归生成前置子目标，实现多跳推理。

**🔧 技术方法**

主要技术包括：NL‑Logic 逻辑表示、基于 LLM 的子目标拆分与细化、向后链式检索与统一、检索-增强生成（RAG）记忆骨干、可验证统一（LLM 判断类型、等价、蕴含），以及批量化的 LLM 调用以降低计算开销。

**📊 数据集**

实验数据集：LoCoMo（多会话、多跳长记忆）和 LongMemEval‑Small（核心助手记忆、提取、时间推理、知识更新）。

**📈 对比分析**

与八种现有 RAG 记忆骨干（BM25‑RAG、Dense‑RAG、Mem0、Mem0‑Graph、MemTree、RAPTOR、GraphRAG、A‑MEM）以及前向推理基线（Query Reformulation、Self‑Reflection、MemGuide、ReAct）在相同记忆骨干和 LLM 后端下对比，Goal‑Mem 在 LLM‑judge 精度和 token‑F1 上均显著提升，尤其在多跳推理任务上提升幅度最大，平坦骨干（BM25‑RAG、Dense‑RAG）受益最为明显。

**⚠️ 局限性**

局限性包括：① 对 LLM 的依赖度高，若 LLM 逻辑判断失误会导致统一失败；② 需要手工设定最大深度/宽度，过大会增加计算成本；③ 在已具备高度结构化记忆（如 Mem0‑Graph）时提升幅度有限；④ 统一过程仍依赖 LLM 对类型和蕴含的判断，可能对非英文或多语言场景适配不佳。

---

## 697. Intrinsic Vicarious Conditioning for Deep Reinforcement Learning

**arXiv ID:** 2605.12224 | [PDF](https://arxiv.org/pdf/2605.12224v1)

**作者:** Rodney A Sanchez `[一作]` (Rochester Institute of Technology), Jamison Heard `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 295 | [OpenAlex ID](https://openalex.org/A5075542202)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于观摩的 Vicarious Conditioning 内在奖励机制，允许 RL 代理仅凭少量演示轨迹学习避险与趋向行为。

**💡 创新点**

创新点在于将心理学的四步观摩学习（注意、保持、再现、强化）映射到 RL，并利用 Siamese 记忆增强网络实现低样本、无策略、无奖励函数的学习。

**🔧 技术方法**

采用 Siamese LSTM 编码器、外部记忆增强网络（MANN）、阈值抑制机制，以及 PPO 算法与内在奖励结合。

**📊 数据集**

使用 MiniWorld Sidewalk（非描述性终止）和 Box2D CarRacing（自定义 POMDP 包装）两套演示数据，每套 26 条轨迹。

**📈 对比分析**

与基线 PPO 及刺激型内在奖励对照，Vicarious Conditioning 在 Sidewalk 显著延长存活时间（平均 142 步 vs 117 步），在 CarRacing 显著提升回合长度（平均 ≈88 步 vs 33 步），同时维持或降低外部奖励。

**⚠️ 局限性**

局限在于 MANN 架构对特征提取器相似行为的区分能力有限，且一旦存储的行为不再有用，记忆无法消退，导致过期恐惧信号长期残留。

---

## 698. PreScam: A Benchmark for Predicting Scam Progression from Early Conversations

**arXiv ID:** 2605.12243 | [PDF](https://arxiv.org/pdf/2605.12243v1)

**作者:** Weixiang Sun `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5332 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含11,573条真实对话式诈骗实例的基准数据集，并提出了两项任务——实时终止预测和诈骗者行动预测

**💡 创新点**

首次将诈骗对话建模为三阶段（初始接触、互动、终止）的“诈骗杀链”，并在此框架下对心理技术进行细粒度标注

**🔧 技术方法**

利用LLM（GPT‑4o‑mini、MiniMax‑2.5等）进行对话抽取与结构化，并使用传统机器学习、神经网络和大型语言模型对任务进行评估

**📊 数据集**

基于BBB Scam Tracker公开的177,989条诈骗报告经过LLM抽取与清洗，最终得到真实世界诈骗对话的结构化数据集

**📈 对比分析**

对比了传统机器学习、深度学习编码器和零样本LLM；在实时终止预测中，监督编码器（如BERT）优于零样本LLM，表现为较高的AUC/AUPR；在行动预测中，顶尖LLM（Claude‑Sonnet‑4.5、DeepSeek‑V3.2）取得最高的动作命中率与心理技术命中率，但整体仍低于人类水平

**⚠️ 局限性**

模型在把握风险随对话进展而升高以及预测具体心理技术驱动的行动序列上表现不足，说明其缺乏对诈骗进程的结构化推理能力

---

## 699. Morphologically Equivariant Flow Matching for Bimanual Mobile Manipulation

**arXiv ID:** 2605.12228 | [PDF](https://arxiv.org/pdf/2605.12228v1)

**作者:** Max Siebenborn `[一作]` (TU Darmstadt), Georgia Chalvatzaki `[通讯]` (TU Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

利用机器人双侧形态对称性作为先验，通过流匹配（flow matching）实现行为克隆，提升双臂移动机器人任务的采样效率、泛化性和最优性。

**💡 创新点**

①将反射对称性形式化为POMDP中的[2]‑等变约束；②提出三种约束策略（数据增强、等变正则化、等变网络）并实现了一个通用的[2]‑等变Transformer；③证明等变流匹配能够在不需要对称数据的情况下实现零样本对称任务迁移。

**🔧 技术方法**

流匹配、等变神经网络、正则化等变损失、数据增强、双臂机器人控制（姿态控制）

**📊 数据集**

模拟数据集：Push‑T、Tiago++双臂箱子提起、柜子开启；真实数据集：Tiago++单臂抓取放置与倒立罐子任务

**📈 对比分析**

与无对称性基线（FM‑Baseline）相比，等变方法平均提升约2×采样效率；在镜像（g_r）任务上实现零样本成功率与原始任务相当；在更困难的任务（窄/宽目标分布）中，等变方法的性能提升更大；在真实机器人上也验证了零样本迁移。

**⚠️ 局限性**

仅考虑水平反射对称，未处理旋转或时域对称；依赖高质量专家演示；实验主要基于姿态/位置观测，未覆盖视觉输入；对大规模多臂/复杂环境的可扩展性仍需进一步验证。

---

## 700. Not How Many, But Which: Parameter Placement in Low-Rank Adaptation

**arXiv ID:** 2605.12207 | [PDF](https://arxiv.org/pdf/2605.12207v1)

**作者:** Arijit Sehanobish `[一作]` (Kensho Technologies), Charles Lovering `[通讯]` (Kensho Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LoRA适配器中，探究在固定训练参数预算下，训练哪些参数会影响性能，并提出一种基于梯度的轻量级参数选择方法；在多种模型和任务上验证该方法在SFT与RL两种训练模式下的效果。

**💡 创新点**

发现梯度分布决定参数选择是否重要；在RL训练下梯度高度多样且方向不一致，随机选取参数无法学习；而通过梯度一致性选取关键参数，即可在仅占LoRA参数0.1%以内恢复完整LoRA性能。

**🔧 技术方法**

梯度幅值/方差评分（Gradient Magnitude & Diagonal Fisher），LoRA适配器固定A、随机初始化B，随机梯度评估后快速挑选top‑k参数；对比随机、全量训练。

**📊 数据集**

使用Qwen2.5‑1.5B/3B/7B、Llama‑3.2‑3B‑Instruct、Llama‑3.1‑8B‑Instruct等基础模型；任务包括Alpaca、Bespoke‑Stratos、GPQADiamond、MathVista、GSM8K、MATH‑500、AIME‑2024、AMC‑2023等。

**📈 对比分析**

方法与随机选择、全量LoRA在SFT下差距小；在GRPO RL下随机选择几乎无提升，而梯度选择在0.05%–0.1%参数预算即可达到或超过全量LoRA（例如Qwen2.5‑3B在MATH‑500上，GRPO accuracy从≈17%提升到≈43%）。性能提升快速、成本极低（<10 s前向后向，<0.5%训练成本）。

**⚠️ 局限性**

仅验证了LoRA框架，其他PEFT方法可能不适用；方法依赖梯度在初始化时的统计，若梯度分布变化可能失效；在极大预算或不同训练策略下的表现尚未系统评估；对模型外推（跨任务/跨模型）的泛化能力仍需进一步研究。

---

## 701. On the Importance of Multistability for Horizon Generalization in Reinforcement Learning

**arXiv ID:** 2605.12206 | [PDF](https://arxiv.org/pdf/2605.12206v1)

**作者:** Asad Bakija `[一作]` (University of Liège), Guillaume Drion `[通讯]` (University of Liège)

**通讯引用:** 1031 | [OpenAlex ID](https://openalex.org/A5084189456)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

提出并理论化了“时间视界泛化（Temporal Horizon Generalization, THG）”概念，并阐明其与循环神经网络（RNN）动力学的关系；

**💡 创新点**

证明多稳定性（multistability）是实现THG的必要且充分条件，并揭示现代可并行化RNN（如SSM、minGRU）本质上为单稳定（monostable），从而无法实现THG；

**🔧 技术方法**

使用多种RNN变体（非线性GRU、BRC、nBRC；可并行化的minGRU；多稳定的BMRU）以及混合架构来验证理论；

**📊 数据集**

在自定义的RL环境中进行实验：T‑maze（短/长视界）和LookupTreeMaze（多级视界）；

**📈 对比分析**

通过平均奖励对模型在不同视界长度下的表现进行比较，结果显示多稳定RNN能在任意长视界下保持最优，单稳定RNN则无法泛化；实验也表明多稳定模型在短视界训练时更易实现THG；

**⚠️ 局限性**

主要局限包括：多稳定性虽为理论必然性，但训练过程中不一定能达到；多稳定RNN在更复杂任务中可能仍需短暂动力学；现有可并行化架构缺乏多稳定性，需要设计混合模型以兼顾并行与记忆。

---

## 702. Graph-Grounded Optimization: Rao-Family Metaheuristics, Classical OR, and SLM-Driven Formulation over Knowledge Graphs

**arXiv ID:** 2605.12204 | [PDF](https://arxiv.org/pdf/2605.12204v1)

**作者:** Madhulatha Mandarapu `[一作]`, Sandeep Kunkunuru `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于属性图的优化范式，并在 samyama-graph 实现了完整的工作流。

**💡 创新点**

创新点在于直接从知识图读取决策变量、约束与目标，而非依赖文本或表格输入，并通过两种查询模式揭示数据质量问题。

**🔧 技术方法**

采用 Cypher 查询、Rust 实现前置聚合、Rao 家族元启发式、Google OR-tools 以及多种 LLM 基准进行比较。

**📊 数据集**

使用了 7 个公共领域 KG：药物重定位 (245K 节点)、临床试验注册 (7.78M 节点)、印度道路图 (5.34M 节点)、WHO/GAVI/IHME (19.7K 节点)、智能电网 CSV (28 节点)、抗药基因库 (10.4K 节点) 与加州森林火灾道路图 (12.6K 节点)。

**📈 对比分析**

对比 Rao 家族五个变体与 OR-tools 的 CP-SAT/GLOP，发现 BMWR 在多样化问题中占优，Rao-1 在中低维连续问题上显著领先；LLM 基准在无 schema 提示下几乎全部失败。

**⚠️ 局限性**

局限包括种子数有限、未与所有 LLM 系统直接对比、部分问题约束与 LP 形式不完全对齐，以及对候选集选择规则的依赖。

---

## 703. Uncertainty Quantification for LLM-based Code Generation

**arXiv ID:** 2605.12201 | [PDF](https://arxiv.org/pdf/2605.12201v1)

**作者:** Senrong Xu `[一作]` (Nanjing University), Xiaoxing Ma `[通讯]` (Nanjing University)

**通讯引用:** 2393 | [OpenAlex ID](https://openalex.org/A5041674680)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于LTT框架的风险控制预测集，用于量化大型语言模型（LLM）在代码生成任务中的不确定性，避免传统PAC预测集的单标签和单调性限制；

**💡 创新点**

创新点在于：①将代码生成转化为多标签问题，移除PAC预测集的单调性约束；②引入学习后测试（Learn‑Then‑Test）多假设检验实现风险控制；③设计结构化风险函数和对应的部分程序优化；④提出可容忍标签误差的选择性执行策略；

**🔧 技术方法**

核心技术包括：LTT框架、多假设检验、部分程序（AST子树）优化、重要性抽样与Hoeffding上界、Z3求解器、LLM（Deepseek‑Coder, Qwen2.5‑Coder, Llama3.1）生成与测试；

**📊 数据集**

实验使用三大代码生成基准：HumanEval、MBPP、APPS，随机划分为50/50校准/测试集，重复100次；

**📈 对比分析**

与PAC预测集及其贪心变体比较，结果显示在所有LLM与数据集上，本文方法在满足1−α覆盖率的前提下，节点删除率显著降低（最高可达24.5%），并在多标签情况下保持更高的覆盖；

**⚠️ 局限性**

局限性：需要在校准阶段执行测试用例以扩充标签，增加计算开销；虽然选择性执行能降低成本，但仍依赖模型不确定性评分与阈值设定；

---

## 704. BatchBench: Toward a Workload-Aware Benchmark for Autoscaling Policies in Big Data Batch Processing -- A Proposed Framework

**arXiv ID:** 2605.12272 | [PDF](https://arxiv.org/pdf/2605.12272v1)

**作者:** Venkata Krishna Prasanth Budigi `[一作]`, Siri Chandana Sirigiri `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并设计了 BatchBench，一个可插拔的批量作业自动伸缩基准框架。

**💡 创新点**

创新点在于结合工作负载分类、统一策略接口、五维评价指标和对 LLM 推理成本的计入。

**🔧 技术方法**

使用 Spark 作业生成器、标准化 Agent 接口、Kolmogorov–Smirnov 与地球移动距离验证、Wilcoxon、BCa 等统计检验技术。

**📊 数据集**

利用公开的 Alibaba、Google、SWIM 集群轨迹以及 TPC、HiBench 等基准数据构建六类工作负载。

**📈 对比分析**

通过五轴评测（成本、SLA、响应、抖动、可解释性）对规则、学习和代理三类策略进行对比，并设计假设检验以评估性能差距。

**⚠️ 局限性**

局限在于目前仅为设计论文，缺乏实证结果；工作负载覆盖仍基于有限公开轨迹，云成本模型简化。

---

## 705. Minimalistic Terminal Editor for Julia Programming -- MinTEJ: A Friendly Approach for a Scientific Programmer

**arXiv ID:** 2605.12275 | [PDF](https://arxiv.org/pdf/2605.12275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 706. CAD-feature enhanced machine learning for manufacturing effort estimation on sheet metal bending parts

**arXiv ID:** 2605.12266 | [PDF](https://arxiv.org/pdf/2605.12266v1)

**作者:** Matteo Ballegeer `[一作]` (Ghent University), Joost R. Duflou `[通讯]` (KU Leuven)

**通讯引用:** 15658 | [OpenAlex ID](https://openalex.org/A5001583017)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种混合方法，结合基于规则的表面金属折弯特征识别与基于B‑rep的图神经网络，实现对折弯制造工时与可制造性评估的预测。

**💡 创新点**

创新点在于将规则提取的制造特征（如折弯角、半径、法兰长度等）作为节点属性直接嵌入完整的B‑rep面邻接图，既保留几何信息又增强了工艺语义，使模型在学习时关注真正相关的制造模式。

**🔧 技术方法**

采用STEP文件的特征识别模块、UV‑Net/FoV‑Net图神经网络、节点几何嵌入（UV采样或射线法）以及全连接网络进行回归/分类。

**📊 数据集**

使用两组数据集：1) KUL‑bend，503个真实工厂折弯设计与测量工时；2) BenDFM，14000个合成折弯设计的碰撞/可制造性标签。

**📈 对比分析**

与纯几何基学习基线（UV‑Net、FoV‑Net）和基于均值/类别比例的基准比较，结果显示加入制造特征后RMSE下降至31.16s、MAE 20.38s、MAPE 44.53%（KUL‑bend）；碰撞检测准确率提升至81.26%（BenDFM）。

**⚠️ 局限性**

局限包括仅验证于单一工厂的折弯过程、工时测量受操作员/设置噪声影响、以及特征对碰撞检测提升有限，未来需扩大多厂、多工艺的数据集并评估跨工艺迁移。

---

## 707. Hypernetworks for Dynamic Feature Selection

**arXiv ID:** 2605.12278 | [PDF](https://arxiv.org/pdf/2605.12278v1)

**作者:** Javier Fumanal-Idocin `[一作]` (University of Essex), Javier Andreu-Perez `[通讯]` (University of Essex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Hyper‑DFS框架，在动态特征选择中使用超网络按需生成每个特征子集的专用分类器，解决传统共享预测器无法同时对指数级子集最优的问题。

**💡 创新点**

核心创新在于将特征子集通过Set Transformer编码为连续表示，利用超网络生成对应权重，并通过梯度分配与学习率预热来稳定训练，从而实现对未见子集的强泛化与零样本性能提升。

**🔧 技术方法**

使用了超网络（hypernetwork）、Set Transformer编码、Gumbel‑softmax求解选择策略、压缩网络（compressor）以及多模型集成等技术。

**📊 数据集**

在合成（Cube、Sim1–3、ProxySub、SynPairs）、表格（Diabetes、Heart、Cirrhosis、Wine、Bank Marketing、California Housing、MiniBooNE、METABRIC、Yeast）和图像（MNIST、Fashion MNIST、SVHN、Imagenette、PCam）数据集上进行实验。

**📈 对比分析**

与多种现有DFS方法（DIME、VIP、CWCF、INVASE、RePa、SEFA、EDDI、AACO、Cardinality、Ensemble、MoE）以及超网络集成进行对比，Hyper‑DFS在平均AUAC‑F1上均表现最优，零样本子集泛化也显著优于基线。

**⚠️ 局限性**

局限性包括训练成本高、超网络生成的连续编码导致模型多样性受限（集成无显著提升）、需要手动调节批量掩码分布与预热策略，以及缺乏对每个子集性能差异的更严格理论解释。

---

## 708. How Useful Is Cross-Domain Generalization for Training LLM Monitors?

**arXiv ID:** 2605.12265 | [PDF](https://arxiv.org/pdf/2605.12265v1)

**作者:** Sam Martin `[一作]` (Anthropic Fellows Program), Fabien Roger `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了将单词级分类微调（1-token SFT）与多任务提示结合的训练方式，探讨其在相邻领域和思考式分类、摘要等生成任务中的迁移与鲁棒性。

**💡 创新点**

创新点在于证明单词级分类微调可有效迁移到思考式分类和摘要任务，并通过与指令跟随训练混合来缓解泛化失效问题。

**🔧 技术方法**

使用的技术包括LoRA低秩微调、CISPO强化学习、思考式推理、两阶段分组训练等。

**📊 数据集**

实验数据集涵盖化学危害、网络危害、ControlArena、IMDB、Covert Reasoning等多任务数据。

**📈 对比分析**

通过AUC/LogAUC等指标与未微调模型、单任务/多任务微调以及RL训练进行对比，发现1-token SFT可获得与RL相当甚至更高的性能，并提升思考分类AUC至0.94。

**⚠️ 局限性**

局限性包括仅在Qwen3系列模型上验证，任务与模型多样性有限，强化学习设置过于简化。

---

## 709. Missingness-MDPs: Bridging the Theory of Missing Data and POMDPs

**arXiv ID:** 2605.12262 | [PDF](https://arxiv.org/pdf/2605.12262v1)

**作者:** Joshua Wendland `[一作]` (Ruhr University Bochum), Nils Jansen `[通讯]` (Ruhr University Bochum)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一类名为 Missingness-MDP 的 POMDP 子类，并给出了在已知 MDP 转移但未知缺失函数的情况下，利用 PAC 学习算法从历史数据中逼近缺失函数，并进一步求解近似最优策略。

**💡 创新点**

创新点在于将缺失数据理论与决策过程紧密结合：①定义了 Missingness-MDP 并阐释 MCAR/MAR/MNAR 的可识别性；②证明 MAR 类型下的缺失函数可忽略，③提出针对 MCAR、简单 MAR 与可识别 MNAR 的 PAC 学习算法，并给出全局最优性保证。

**🔧 技术方法**

技术方法包括：基于统计计数的缺失指示器估计、m‑graph 结构化假设、PAC 学习框架、以及将逼近的缺失函数注入标准 POMDP 求解器（如 SARSOP）以获得策略。

**📊 数据集**

实验使用了两个合成基准：医生-病人诊断任务和 Tag 预捕猎任务，并在其中构造了四种缺失类型（MCAR、简单 MAR、可识别 MNAR、不可识别 MNAR）。

**📈 对比分析**

与基线（无信息先验、POMCP、PPO）对比，实验表明在满足缺失假设的情况下，所提出的算法能够在数据量足够时逼近真实缺失函数，进而得到接近最优的策略；基线方法在同一任务上表现显著逊色。

**⚠️ 局限性**

限制包括：需要已知的转移函数；PAC 结果在理论上要求足够多的历史数据，实际样本量有限时精度受限；仅适用于可识别的缺失类型，对自我屏蔽 MNAR 等更复杂情形尚未覆盖。

---

## 710. PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents

**arXiv ID:** 2605.12260 | [PDF](https://arxiv.org/pdf/2605.12260v1)

**作者:** Jingyi Peng `[一作]` (Singapore Management University), Qiuzhuang Sun `[通讯]` (Singapore Management University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 PRISM，一种训练无关的检索端框架，用于在长时序对话中高效检索并压缩记忆，支持在极低上下文成本下回答问题。

**💡 创新点**

创新点在于将检索视为图结构上的最小代价路径选择，并在检索后通过单次 LLM 调用实现内容压缩，同时引入意图路由以减少不必要的 LLM 调用。

**🔧 技术方法**

使用技术包括分层图检索（Hierarchical Bundle Search）、查询敏感边代价调整（Query‑Sensitive Edge Cost）、LLM 驱动的证据压缩（Evidence Compression）和关键词/原型/LLM 三阶意图路由（Adaptive Intent Routing）。

**📊 数据集**

实验基于 LoCoMo 长会话 QA 基准进行评估，涵盖单跳、多跳、时间和开放域四类问答。

**📈 对比分析**

与 Full Context、MAGMA、Mem0 等同协议基线比较，PRISM 在 LLM‑judge 分数上提升 14‑35% 点，同时将上下文 token 减少 13 倍，显示出明显的准确率‑成本优势。

**⚠️ 局限性**

局限性包括对锚点可检索的假设（多跳桥接场景占比低）、对图结构的优势在当前基准上难以显现，以及在意图路由层面仍需依赖 LLM 以处理复杂查询。

---

## 711. SI-Diff: A Framework for Learning Search and High-Precision Insertion with a Force-Domain Diffusion Policy

**arXiv ID:** 2605.12247 | [PDF](https://arxiv.org/pdf/2605.12247v1)

**作者:** Yibo Liu `[一作]` (Epson Canada), Tony Hong-Yau Lo `[通讯]` (Epson Canada)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了SI-Diff框架，利用力域扩散策略同时学习搜索与高精度插入动作，解决了传统方法需要分别处理两阶段的问题。

**💡 创新点**

核心创新点是：①引入模式嵌入机制，使单一扩散模型能学习并区分搜索与插入两种不同动作；②设计随机化搜索教师策略，生成多样化搜索轨迹，从而提升学习效率与泛化能力。

**🔧 技术方法**

采用了扩散模型（DDPM）、力反馈阻尼控制器、触觉与外力观测作为条件、模式嵌入、平衡批采样等技术。

**📊 数据集**

数据集来自7-DoF Franka机器人收集的300条成功搜索轨迹与TacDiffusion提供的1500条插入演示，并在六种未见形状（六边柱、五边柱、三角柱、圆柱、USB-A）上进行零样本迁移测试。

**📈 对比分析**

与TacDiffusion、Teacher-D、Teacher-R、FORGE等方法对比，SI-Diff在所有形状上实现最高成功率（最高达95%），误差容忍度从2 mm提升至5 mm，同时平均执行时间更短，整体性能优于现有基线。

**⚠️ 局限性**

局限性在于模式切换仅依赖EE沿z轴位移而非触觉信号，且目前仅针对x–y误差和小间隙插入，未处理姿态偏移（平移/偏转）等更复杂情况。

---

## 712. SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization

**arXiv ID:** 2605.12245 | [PDF](https://arxiv.org/pdf/2605.12245v1)

**作者:** Chengzhu Bao `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向大型语言模型的后训练量化框架SOAR，专门针对NVFP4微尺度格式进行精确量化

**💡 创新点**

创新点在于：①闭式联合尺度优化（CJSO）实现全局与块级尺度的解析式协同更新；②解耦尺度搜索（DSS）将量化尺度与去量化尺度分离，并在硬件约束下进行离散搜索，显著降低尺度量化误差

**🔧 技术方法**

使用解析梯度推导的闭式更新、块级离散搜索、梯度下降迭代、以及对FP4/FP8量化的数学模型

**📊 数据集**

在多种大型语言模型（LLaMA-3.1/3.2、Qwen3-4/8B）上进行零样本推理评估，并结合WikiText-2、C4作为校准数据进行GPTQ融合实验

**📈 对比分析**

与4over6、RaZeR等NVFP4基线以及FP16全精度模型对比，SOAR在五大零样本推理任务和MMLU/GSM8K推理任务上均优于现有NVFP4方案，平均提升约1–2个百分点，且内存占用与传统NVFP4相同

**⚠️ 局限性**

局限性包括：目前仅关注静态权重量化，未针对激活动态范围进行自适应优化；对极低精度（如INT4）仍需进一步验证；解耦搜索在极大模型规模下的计算开销尚未系统评估

---

## 713. Harness Engineering as Categorical Architecture

**arXiv ID:** 2605.12239 | [PDF](https://arxiv.org/pdf/2605.12239v1)

**作者:** Bogdan Banu `[一作]` `[通讯]`, Bogdan Banu

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 agent externalization 四大支柱映射到 ArchAgents 结构三元组，并实现编译器保证证书保留，支持跨框架迁移的 harness；

**💡 创新点**

首次将 Harness Engineering 与范畴理论 ArchAgents 框架对齐，提供正式的证书保持与运算符组合理论，使 harness 可正式验证与跨框架迁移；

**🔧 技术方法**

使用范畴论的 Architecture triple、operad 组合、functor 编译器、证书重放检查，并在 Operon 框架实现；

**📊 数据集**

利用 Gemma4、Phi‑3 Mini、DeepSeek‑R1 等 8B LLM 进行模型升级实验，并在 SWE‑bench‑lite 10 个真实 Python bug‑fix 任务上验证；

**📈 对比分析**

通过比较五个目标框架（Swarms、DeerFlow、Ralph、Scion、LangGraph）的编译结果，证书保持率均为 100%，LangGraph 的 per‑stage 编译保持 100%；在 8B 模型上 SWE‑bench‑lite 任务未提升，证明 8B 级别存在格式纪律瓶颈；

**⚠️ 局限性**

限于单一 Operon 实现、静态架构快照、仅覆盖结构性证书、实验规模有限，未验证动态演化与行为级别保证。

---

## 714. TMRL: Diffusion Timestep-Modulated Pretraining Enables Exploration for Efficient Policy Finetuning

**arXiv ID:** 2605.12236 | [PDF](https://arxiv.org/pdf/2605.12236v1)

**作者:** Matthew M. Hong `[一作]` (University of Washington), Abhishek Gupta `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种通过在预训练阶段对上下文注入前向扩散噪声来构建可调宽度行为分布的上下文平滑策略（CSP），并在微调阶段引入时间步调制强化学习（TMRL）使策略在执行过程中动态调节噪声级别，以实现从精细模仿到广泛探索的平滑过渡。

**💡 创新点**

创新点在于：①将BC与RL之间的覆盖瓶颈转化为可调的上下文平滑机制，①通过前向扩散噪声实现p(a|c)与p(a)之间的插值；②在微调阶段利用高层策略主动选择扩散时间步，从而实现探索与利用的自适应平衡；③提供理论分析证明噪声水平可显著提升动作分布覆盖。

**🔧 技术方法**

技术上使用了：
- 前向扩散噪声与后向去噪（或流模型）构成的生成控制策略；
- 高层RL（如离线 actor‑critic）控制噪声时间步与潜在变量；
- 对状态、图像、点云、VLM嵌入等多种上下文类型的统一处理；
- 结合离线BC、PostBC、CFG等已有方法进行对比实验。

**📊 数据集**

使用的数据集包括：
- 机器人演示数据（UW、Amazon FAR、BridgeData‑v2、DROID等）；
- OGBench 机器人导航与操纵模拟环境；
- LIBERO 视觉语言动作（VLA）基准；
- 真实世界 WidowX 250 6‑DoF 与 Franka Panda 7‑DoF 机器人实验数据。

**📈 对比分析**

与传统 BC、PostBC、Gaussian‑noise RL、Hierarchical RL、Steering‑diffusion 等基线相比，CSP+TMRL 在 OGBench、LIBERO 及真实机器人任务上取得显著优势：
- 在模拟任务中实现 100% 成功率，提升 14%‑200%；
- 在抓取任务中收敛速度加快 2.5×、最终成功率提升；
- 在真实机器人任务中，TMRL 在不到一小时的实验时间内完成精确抓取、放置等复杂任务，基线几乎无成功率。

**⚠️ 局限性**

局限性：
- 由于通过上下文混淆产生的宽动作分布可能导致安全性下降，需配合安全过滤器或世界模型；
- 对极其复杂或长时延任务的样本效率仍有限，尚需进一步提升蒸馏或高层策略效率；
- 目前的扩散噪声核与时间步调度仍为经验设计，未深入探索更优的结构化噪声策略。

---

## 715. ORCHID: Orchestrated Reduction Consensus for Hash-based Integrity in Distributed Ledgers

**arXiv ID:** 2605.12211 | [PDF](https://arxiv.org/pdf/2605.12211v1)

**作者:** Abraham Itzhak Weinberg `[一作]` `[通讯]` (AI Experts), Abraham Itzhak Weinberg (AI Experts)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并验证了ORCHID，一种基于量子相位同步和神经绑定启发的分布式共识协议；

**💡 创新点**

将神经绑定问题映射到共识，利用Kuramoto耦合振荡器在相干阈值下触发共识门限，并引入相干加权的量子秘密共享，突破传统Byzantine容错上限；

**🔧 技术方法**

Kuramoto耦合振荡器模型、量子相位噪声、相干加权量子秘密共享、Watts‑Strogatz小世界网络仿真；

**📊 数据集**

使用Watts‑Strogatz小世界网络模拟，节点数10–150，随机频率、相干度等参数；

**📈 对比分析**

与PBFT、PoW等传统协议对比，ORCHID实现100%共识、平均收敛≤4 s、O(n·k)消息复杂度，在n≥150时优于PBFT；

**⚠️ 局限性**

需要真实量子硬件实现，阈值固定可能不适用于异构网络，安全性为概率性而非确定性；

---

## 716. Secure (Multiple) Key-Cast over Networks: Multiple Eavesdropping Nodes

**arXiv ID:** 2605.12209 | [PDF](https://arxiv.org/pdf/2605.12209v1)

**作者:** Reza Sayyari `[一作]` (University at Buffalo), Michael Langberg `[通讯]` (University at Buffalo)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究在无噪声网络中，针对节点级窃听（可协同且可观测多节点）的安全多键投递（Secure Multiple Key‑Cast）问题，给出了单源和多源情形下在不同连通性条件下的容量下界、上界以及可实现的编码方案，并扩展到部分连通节点的网络。

**💡 创新点**

创新点包括：
- 在所有节点均为 d‑vertex 连接时，证明安全密钥速率为 d−ℓ 并给出可实现方案；
- 进一步证明该速率是最优的；
- 通过结合安全重构码和秘密分享技术，设计了可在存在部分连通节点时仍可实现安全密钥投递的编码方案，速率取决于最小顶点连通度及节点结构；
- 对多源情形进行推广，允许窃听者观察到除最多 x 个源节点外的 ℓ 个节点，给出可达的速率 (d−ℓ)(|S|−x)/|S| 并证明安全性。

**🔧 技术方法**

主要技术：
- 结合安全重构码（Secure Regenerating Codes）和 Shamir 秘密分享；
- 使用 Vandermonde 向量与矩阵进行符号映射，实现多路复用与信息分离；
- 通过线性代数分析（矩阵可逆性、随机对称矩阵分布）证明信息安全性；
- 采用顶点连通度与边连通度的网络流理论来确定可实现的速率。

**📊 数据集**

无实验数据集，论文完全基于理论分析与构造性证明。

**📈 对比分析**

与传统安全网络编码（仅考虑边级窃听或单源、多源下的密钥传输）相比，本工作在更一般的节点级窃听模型下给出了可实现的密钥速率，并证明在全连通网络中已达极限。性能上，若所有节点 d‑vertex 连接，则安全速率为 d−ℓ；若只保证终端 d‑vertex 连接，则速率为 d−ℓ−z+1（z 为部分连通节点的影响程度）并给出进一步修正项。

**⚠️ 局限性**

局限性：
- 仅在无噪声（无误码）网络中考虑；
- 对于部分连通节点的情况，仍需要满足严格的结构条件（如每个 d‑vertex 连接节点最多接收 z 个部分连通节点的输入）；
- 多源情况下对窃听者可观测源节点数的限制（x < |S|）；
- 方案对字段大小 q 的依赖性，需要足够大的有限域；
- 对于一般网络（非 DAG 或高复杂连通度）未给出完整的实现细节与可扩展性分析。

---

## 717. What makes a word hard to learn? Modeling L1 influence on English vocabulary difficulty

**arXiv ID:** 2605.12281 | [PDF](https://arxiv.org/pdf/2605.12281v1)

**作者:** Jonas Mayer Martins `[一作]` (University of Göttingen), Lisa Beinborn `[通讯]` (University of Göttingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并训练了针对西班牙语、德语和中文母语者的英语词汇难度预测模型，基于词汇熟悉度、语义、表面形式和跨语种转移特征进行解释性建模

**💡 创新点**

通过将特征划分为四个组并利用Shapley值解释模型，揭示不同母语者在词汇难度上依赖的不同机制（西班牙语/德语侧重熟悉度+正向转移，中文侧重熟悉度+表面特征）

**🔧 技术方法**

使用CatBoost梯度提升决策树，结合特征工程（词频、年龄获取、词义深度、形态学、字符相似度等）以及Shapley值分析

**📊 数据集**

采用Knowledge-based Vocabulary Lists (KVL) 数据集，包含超过100,000名学习者在三种母语背景下的英语词汇测试结果

**📈 对比分析**

与线性岭回归和Transformer基线对比，CatBoost在RMSE上略优，Pearson相关系数更高；在跨母语评估中，西德模型互相迁移效果良好，中文模型迁移性能较差

**⚠️ 局限性**

受限于数据覆盖的语言范围（仅四种高资源语言）、任务只评估词形回忆、跨语种转移仅以字符相似度衡量，且模型解释并未直接映射到学习者的认知过程

---

## 718. Iterative Audit Convergence in LLM-Managed Multi-Agent Systems: A Case Study in Prompt Engineering Quality Assurance

**arXiv ID:** 2605.12280 | [PDF](https://arxiv.org/pdf/2605.12280v1)

**作者:** Elias Calboreanu `[一作]` `[通讯]` (Swift Group, LLC), Elias Calboreanu (Swift Group, LLC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对生产级七道管道的提示规范进行九轮迭代审计，发现并修复51个缺陷。

**💡 创新点**

首次系统化证明单文件审计无法发现跨文档缺陷，并提出可复制的迭代审计协议。

**🔧 技术方法**

使用Claude子代理结合自定义检查清单，采用逐轮跨文件对照和修复机制。

**📊 数据集**

审计对象为AEGIS系统的八个提示规范文件（约7150行），并记录每轮缺陷列表。

**📈 对比分析**

通过对比各轮缺陷数量与累计发现率，展示非单调收敛曲线，最终在第九轮实现零缺陷；未与单遍全范围控制对比。

**⚠️ 局限性**

样本单一、LLM与作者同族、缺陷分类为单编码、未进行多轮重复和对照实验，导致结果可能受审计者偏差和协议演化影响。

---

## 719. Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs

**arXiv ID:** 2605.12242 | [PDF](https://arxiv.org/pdf/2605.12242v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 720. Into the Unknown: Accounting for Missing Demographic Data when Mitigating Ad Delivery Skew

**arXiv ID:** 2605.12273 | [PDF](https://arxiv.org/pdf/2605.12273v1)

**作者:** Isabel Corpus `[一作]` (Cornell University), Allison Koenecke `[通讯]` (Cornell University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了政府广告中因平台算法导致的性别投放偏差，并设计了一种考虑“未知用户”的预算拆分干预方案

**💡 创新点**

创新点在于将“未知用户”纳入预算拆分，通过“单性别+未知”组合实现对偏差的调节，兼顾公平与成本

**🔧 技术方法**

主要技术是Google Ads的预算拆分策略、性别标签推断、CTR/ CVR/ CPM 计算与置信区间评估

**📊 数据集**

使用的实证数据来自与州级政府机构合作的Google Search广告投放，覆盖4周至6周的投放结果以及公开的州级人口性别分布

**📈 对比分析**

与原始全量投放（All Users）和传统单性别拆分（Direct Budget Split）对比，结果显示预算拆分+未知方案在性别比例接近0.5的同时，成本（CPM）低于单性别拆分且显著低于全量投放；偏差降低幅度在40%–80%之间

**⚠️ 局限性**

局限性包括：仅针对性别属性且仅使用平台推断标签，无法处理受限敏感属性（如住房、就业）；预算拆分周期短，仅为6周，无法验证长期效果；未知用户分布假设存在不确定性，模拟结果对极端假设敏感

---

## 721. Beyond Text Prompts: Visual-to-Visual Generation as A Unified Paradigm

**arXiv ID:** 2605.12271 | [PDF](https://arxiv.org/pdf/2605.12271v1)

**作者:** Yaofang Liu `[一作]` (City University of Hong Kong), Raymond H. Chan `[通讯]` (Lingnan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出视觉到视觉（V2V）生成范式，使用结构化视觉规范页作为条件输入，通过冻结的视觉语言模型（VLM）提取隐藏状态并直接注入冻结的扩散生成器，实现零训练的图像和视频生成。

**💡 创新点**

创新点在于将视觉规范页视为第一类条件语言，利用现有VLM与扩散模型的隐藏状态兼容性，无需额外训练或适配器即可实现从视觉到视觉的零射击生成。

**🔧 技术方法**

技术核心是：1）冻结的多模态VLM编码器读取视觉规范页；2）将最终层隐藏状态作为条件注入冻结的Diffusion Transformer（DiT）生成器；3）使用两种条件模式（Image‑HS‑only 与 Full‑Final）来控制视觉与推理状态的混合。

**📊 数据集**

主要使用的数据集包括：GenEval（553个提示，6种结构化技能），Simple‑V2V Bench（7类任务，每类22个提示），以及HunyuanVideo‑1.5验证集（视频版），并通过Qwen‑3‑VL‑32B等VLM评判器进行质量与对齐评分。

**📈 对比分析**

与传统文本到图像模型相比，V2V‑Zero在GenEval上达到0.85（接近官方优化结果），在Simple‑V2V Bench中得到32.7/100，显著优于开源模型但仍落后于商业基线；视频扩展在HunyuanVideo‑1.5上获得20.2/100，证明方法可迁移。

**⚠️ 局限性**

局限性包括：1）结构化控制（姿势、草图、计数）效果仍不稳定；2）对内容生成的可靠性不足；3）依赖现有VLM‑扩散架构，缺乏端到端的专门训练；4）评价标准主要基于自动化VLM评判，缺乏人类主观评估；5）对极端视觉输入的泛化性尚未充分验证。

---

## 722. Reconstruction of Personally Identifiable Information from Supervised Finetuned Models

**arXiv ID:** 2605.12264 | [PDF](https://arxiv.org/pdf/2605.12264v1)

**作者:** Sae Furukawa `[一作]` (Northeastern University), Alina Oprea `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在监督微调（SFT）模型中从指令-响应对里泄露的个人身份信息（PII）的重构风险。

**💡 创新点**

提出了新的解码算法 COVA，利用全局候选池和序列级似然来提升 PII 重构的覆盖率和准确率，并构建了多轮医疗与法律领域的用户中心 Q&A 数据集，首次系统评估了不同攻击者知识水平对泄露程度的影响。

**🔧 技术方法**

使用 prefix‑based 语言模型攻击、覆盖感知解码 COVA、基于 log‑likelihood ratio 的重排序以及多模型（Qwen‑2.5、Llama‑3.1、DeepSeek‑LLM）和不同规模的实验。

**📊 数据集**

从 Reddit 采集的 MedRedQA（医疗）和法律问答数据扩展而来，加入合成 PII 并构建多轮对话，形成专门的 SFT 训练集。

**📈 对比分析**

与传统 top‑k 抽样和 beam search 进行对比，COVA 在 Top‑1、Top‑10、Top‑100 召回率上分别提升约 10%–30%，尤其在高熵 PII（出生日期、姓名）上显著优于基线；模型规模越大、家族性能越好。

**⚠️ 局限性**

限制包括：仅针对 SFT 任务；实验仅涵盖医疗和法律两类；合成 PII 的真实性和多样性受限；对更复杂对话结构与更大模型的可推广性尚未验证。

---

## 723. Characterizing the Failure Modes of LLMs in Resolving Real-World GitHub Issues

**arXiv ID:** 2605.12270 | [PDF](https://arxiv.org/pdf/2605.12270v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 724. Unlocking Crowdsourcing for Ontology Matching Validation

**arXiv ID:** 2605.12226 | [PDF](https://arxiv.org/pdf/2605.12226v1)

**作者:** Zhangcheng Qiang `[一作]` `[通讯]` (Australian National University), Zhangcheng Qiang (Australian National University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可与现有OM系统集成的众包验证平台，利用三种质量保证机制提升非专家验证的可靠性。

**💡 创新点**

创新点包括：1）差异化可信度评估，用种子对衡量用户专业度并加权投票；2）一致性预填充，利用本体推理自动推导并预填标注；3）时间依赖信念，允许随时修改标注并记录时间戳，以适应知识演化。

**🔧 技术方法**

采用众包平台实现用户角色管理与任务发布；差异化可信度基于正负种子对统计；一致性预填充使用推理器（如Reasoner）对用户断言进行一致性检查；时间依赖信念通过为每条标注分配时间戳实现。

**📊 数据集**

在OAEI 2025 “conference”域的两个数据集（cmt-conference、cmt-conof）以及建筑业常用的Brick Schema、RealEstateCore和Project Haystack本体上验证。

**📈 对比分析**

通过与Agent-OM匹配器对接，利用众包投票取代全部专家审阅；实验表明能显著扩展参考对齐，降低人工审核成本，且投票结果与专家一致性得到提升（未给出具体数值）。

**⚠️ 局限性**

限制：难以控制众包参与者的专业水平与多样性；系统目前仅支持一对一等价匹配，尚未扩展到包含子类、并集等更复杂的本体匹配任务。

---

## 725. Heterogeneous SoC Integrating an Open-Source Recurrent SNN Accelerator for Neuromorphic Edge Computing on FPGA

**arXiv ID:** 2605.12217 | [PDF](https://arxiv.org/pdf/2605.12217v1)

**作者:** Michelangelo Barocci `[一作]` (Politecnico di Torino), Gianvito Urgese `[通讯]` (Politecnico di Torino)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

在FPGA上集成并验证了ReckOn递归SNN加速器与X-HEEP微控制器或Zynq ARM处理器的异构SoC，实现了低功耗、可在线学习的边缘神经形态计算。

**💡 创新点**

创新点在于将开源ReckOn加速器与可编程微控制器/ARM系统结合，提供了灵活的硬件控制接口，支持裸机与Linux双模式，且在FPGA资源极限下实现高准确率的在线学习。

**🔧 技术方法**

使用了FPGA可重构平台、X-HEEP RISC-V微控制器、Zynq AXI桥接、SPI/AER接口、e-prop在线学习算法以及Python SDK/裸机代码控制技术。

**📊 数据集**

实验使用了二元导航数据集和Braille数字分类数据集（3类或4类子集）进行验证。

**📈 对比分析**

通过与ReckOn原始硅片和RTL仿真结果对比，FPGA实现的分类准确率与硅片几乎相同；在Braille 3类测试中达到90%准确率，4类测试分别为78.8%和60%。

**⚠️ 局限性**

局限在于BRAM使用率高、资源接近极限，模型规模受限；仅在低端FPGA上测试，未对更大网络或更复杂任务进行评估，也未与所有商业硬件做完整对比。

---

## 726. Angle Between Two Vectors over Finite Fields and an Application to Projective Unique Decoding

**arXiv ID:** 2605.12216 | [PDF](https://arxiv.org/pdf/2605.12216v1)

**作者:** Kamil Otal `[一作]` `[通讯]` (Tubitak Bilgem Uekae), Kamil Otal (Tubitak Bilgem Uekae)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

在有限域上定义并证明了一个基于Hamming距离的角度度量，并将其推广到射影空间，随后利用该度量给出了线性码的射影唯一译码定理。

**💡 创新点**

首次把角度概念引入有限域向量空间，提出了整数值的Hamming角度度量，并证明其满足三条度量公理（不含标量乘法的等价关系），为码论、几何与密码学提供了新的视角。

**🔧 技术方法**

使用了Hamming距离的最小化、射影空间的投影、三角不等式证明和线性时间算法（计数比值的出现次数）等数学技术。

**📊 数据集**

未使用实验数据集，全部为理论证明与算法分析。

**📈 对比分析**

对比方法主要是传统的唯一译码（Hamming距离）与本研究的射影唯一译码；理论上给出了相同的半径 d/2，并提供了线性时间计算角度的算法，复杂度为 O(n)。

**⚠️ 局限性**

局限性包括：仅在有限域上定义；未推广到更高维子空间（Grassmann 角度）；未处理仿射线的角度问题；对非Hamming度量（如秩度量、Lee度量）的适用性尚未研究。

---

## 727. Mechanistic Interpretability of ASR models using Sparse Autoencoders

**arXiv ID:** 2605.12225 | [PDF](https://arxiv.org/pdf/2605.12225v1)

**作者:** Dan Pluth `[一作]` (Vail Systems, Inc.), Vijay K. Gurbani `[通讯]` (Vail Systems, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对 Whisper ASR 模型的内部表示使用稀疏自编码器进行探测和解释。

**💡 创新点**

首次将 SAE 应用于音频 ASR 模型，揭示 Whisper 编码的丰富语言学特征并实现跨语言特征驱动。

**🔧 技术方法**

使用 k-稀疏自编码器在 Whisper 编码器最终层嵌入上训练，并结合激活驱动（steering）验证因果关系。

**📊 数据集**

在多语言多任务数据集（LJSpeech、LibriSpeech、VoxCeleb、Common Voice、SLR 系列、Musan）共 646,769 文件中训练，并用 28,414 文件做分析。

**📈 对比分析**

通过与无 SAE 的 Whisper 输出对比以及激活精准召回测量，展示 SAE 能准确捕捉语言、语音、形态和语义特征，精度可达 74–99%，召回 35–91%。

**⚠️ 局限性**

仅使用 Whisper base 模型，未探讨不同尺寸或层级的 SAE，且对非英语数据覆盖有限。

---

## 728. Fill the GAP: A Granular Alignment Paradigm for Visual Reasoning in Multimodal Large Language Models

**arXiv ID:** 2605.12374 | [PDF](https://arxiv.org/pdf/2605.12374v1)

**作者:** Yanting Miao `[一作]` (University of Waterloo), Guanjun Jiang `[通讯]` (Qwen Large Model Application Team, Alibaba)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GAP（Granular Alignment Paradigm）框架，通过在数据层、特征层和模型层进行分层对齐，实现在大型多模态语言模型内部生成并重用视觉潜在表示；

**💡 创新点**

创新点在于将视觉潜在反馈的特征空间匹配问题通过PCA对齐投影解决，同时引入上下文可控的视觉潜在监督和难度感知的目标分配，三维对齐显著提升了视觉推理性能；

**🔧 技术方法**

采用预归一化的Qwen2.5‑VL‑7B作为骨干，加入PCA对齐的潜在头、难度感知的监督策略，并在训练中使用48k条手工构造的视觉推理示例；

**📊 数据集**

使用49,309条自定义的多模态问答数据，涵盖视觉链式推理、图表、几何、数学、多重计数和视觉检索等任务；

**📈 对比分析**

与零样本基础模型、Monet‑7B、LVR以及密集描述微调模型进行对比，在HRBench4K、MMStar、MME‑RealWorld‑Lite和MathVista、WeMath等基准上平均提升了约3.5%到5.9%的指标，展示了显著的性能提升；

**⚠️ 局限性**

局限性包括对预归一化特征空间差异的解决仍需在更广泛模型中验证，PCA对齐在跨任务泛化上可能受限，且仅在约49k样本规模下验证，尚未评估大规模数据或更复杂视觉任务的适用性。

---

## 729. Optimized but Unowned: How AI-Authored Goals Undermine the Motivation They Are Meant to Drive

**arXiv ID:** 2605.12344 | [PDF](https://arxiv.org/pdf/2605.12344v1)

**作者:** Vivienne Bihe Chi `[一作]` (University of Pennsylvania), Sharath Chandra Guntuku `[通讯]` (University of Pennsylvania)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比自我制定与LLM（大型语言模型）自动生成的个人目标，探究作者身份对目标质量、心理归属感、承诺度、重要性以及行为执行的影响。

**💡 创新点**

首次揭示“目标质量-动机脱钩”——LLM生成的目标在SMART评分上更优，却显著削弱心理归属感与动机，并通过归属感中介解释行为执行差异；并发现自我效能低的用户在归属感侵蚀上更易受负面影响。

**🔧 技术方法**

使用OpenAI GPT‑5.2 生成目标，Gemini 3.1 Pro 作为客观评判者；采用心理归属感量表、承诺度量表、重要性评估、目标自我效能等问卷；利用贝叶斯回归与PROCESS宏进行中介与调节分析。

**📊 数据集**

470名美国成年人（Prolific平台），两周后返回 332 名受试者；所有受试者完成一次基线反思并随机分配至自写或LLM生成两组；使用客观SMART rubric 对 1410 个目标进行评分。

**📈 对比分析**

方法：双组间t检验（Welch）检验差异，Cohen’s d 计算效应量；对比显著性（p<0.001）显示LLM组在SMART评分上高出约2.26 SD；心理归属感、承诺度等指标在LLM组显著降低（d≈1.3-1.4）。中介分析表明归属感完全或部分中介所有动机指标。两周随访显示自写组完成目标的比例为72.8%，LLM组为46.6%，差异显著。

**⚠️ 局限性**

局限性：仅测试全自动生成并再键入的目标设计；未考察更具协同/编辑式的LLM交互；行为结果以自报为主，缺乏客观执行数据；随访时长仅两周，无法观察长期内部化过程；样本来源局限于Prolific平台，可能不具代表性。

---

## 730. Overview of the MedHopQA track at BioCreative IX: track description, participation and evaluation of systems for multi-hop medical question answering

**arXiv ID:** 2605.12313 | [PDF](https://arxiv.org/pdf/2605.12313v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 731. EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model for Electronic Health Records

**arXiv ID:** 2605.12335 | [PDF](https://arxiv.org/pdf/2605.12335v1)

**作者:** Saeed Shurrab `[一作]` (New York University), Farah E. Shamout `[通讯]` (New York University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种检索增强的基础模型（EHR-RAGp），通过原型引导的检索机制动态获取患者历史中最相关的片段并与当前查询拼接，用于临床预测任务。

**💡 创新点**

创新点在于将检索与原型对齐结合：①多粒度分块（事件、时间、就诊、护理阶段）构建向量数据库；②原型引导对齐评估检索片段与查询在潜在语义空间中的相似度，生成软权重；③通过软加权融合检索结果与查询，提升模型对长距离临床信息的利用。

**🔧 技术方法**

使用的技术包括：向量检索（预训练的检索编码器 + 余弦相似度索引）、Mask Language Modeling 预训练、Transformer 编码器、原型向量学习、交叉熵对齐与温度调节、软权重融合与最终分类头；整个系统端到端可微。

**📊 数据集**

实验基于 MIMIC‑IV V3.1（包含 364,627 名患者、546,028 次住院、94,458 次 ICU 入院），涉及诊断、实验室、药物等多种 EHR 事件。

**📈 对比分析**

在四项临床预测任务（住院死亡、ICU 30 天再入院、长期住院（≥7 天）和 1 年死亡）上，与多类基线（EHR 传统基础模型、长上下文 Transformer、检索模型、LLM 零样本）进行公平对比。EHR‑RAGp 在所有任务上均取得最高 AUROC 与 AUPRC，尤其在住院死亡（AUROC 0.940，AUPRC 0.716）和长期住院（AUROC 0.885，AUPRC 0.628）上表现显著优于现有模型。

**⚠️ 局限性**

局限性包括：①仅在单一 MIMIC‑IV 数据集验证，需评估跨数据集通用性；②检索与分块超参数（chunk size、窗口长度、Top‑M）固定，未探究更大或不同粒度的效果；③原型在语义层面缺乏临床可解释性，需专家评估；④模型仅面向事件级 EHR，未结合影像、基因等多模态信息。

---

## 732. Towards Automated Air Traffic Safety Assessment Around Non-Towered Airports Using Large Language Models

**arXiv ID:** 2605.12332 | [PDF](https://arxiv.org/pdf/2605.12332v1)

**作者:** Torsten Darrell `[一作]` (George Washington University), Peng Wei `[通讯]` (George Washington University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视觉语言模型（VLM）的非塔机场安全评估框架，利用CTAF通话转录、METAR气象数据、ADS‑B飞行轨迹和VFR航图进行安全分类与自适应提示生成。

**💡 创新点**

创新点在于首次将多模态（文本+图像）信息整合进LLM/VLM进行后飞行安全分析，并构建了针对非塔机场的12类危害级别合成数据集，验证VLM在安全评估中的有效性。

**🔧 技术方法**

技术手段包括Whisper ASR进行音频转录、Prompt Engineering与Chain‑of‑Thought（CoT）推理、对多种开闭源LLM（Qwen、Mistral、Gemma、GPT‑4o、GPT‑5.4、Claude Sonnet）进行冻结模型推理，并对输出做JSON结构化。

**📊 数据集**

使用了在Half Moon Bay Airport（KHAF）构建的100个合成情景数据集（每个情景包含CTA­F转录、METAR、ADS‑B轨迹、VFR航图），以及单一真实航班案例进行定性验证。

**📈 对比分析**

通过在94个测试场景上评估六种LLM，比较零样本、一样本、少样本与CoT策略，结果显示最佳配置宏F1>0.85，Qwen‑2.5‑7B在零样本直接推理下宏F1=0.964，Claude Sonnet 4.6和GPT‑4o在少样本+CoT下亦表现接近。

**⚠️ 局限性**

局限性包括高计算成本与推理延迟、缺乏实时边缘部署优化、合成数据未覆盖真实音频噪声与多模态视觉信息、未进行模型微调、以及对单机场的泛化性验证不足。

---

## 733. LISA: Cognitive Arbitration for Signal-Free Autonomous Intersection Management

**arXiv ID:** 2605.12321 | [PDF](https://arxiv.org/pdf/2605.12321v1)

**作者:** Abderrahmane Lakas `[一作]` (United Arab Emirates University), Merouane Debbah `[通讯]` (Khalifa University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出并实现了一种基于大语言模型（LLM）的信号‑free交叉口管理框架 LISA，能够在不使用传统信号灯的情况下，通过车辆意图进行认知仲裁并生成速度指令。

**💡 创新点**

创新点在于：
1) 将意图驱动的仲裁与低层速度执行分离，让 LLM 仅负责符号化的权利分配；
2) 采用 Memoized Arbitration Table 缓存冲突签名以显著降低推理延迟；
3) 引入安全 watchdog，确保在 LLM 推理失败或缓存失效时仍保持安全；
4) 对车辆意图进行多维度（空间、时间、优先级、能耗）建模，并通过 LLM 进行整体权衡。

**🔧 技术方法**

主要技术包括：
- Gemini 2.5 Flash Lite LLM 作为仲裁推理核心；
- SUMO 微观仿真与 TraCI 接口；
- 三类车辆（紧急、公交、普通）及其能耗模型；
- 预定义的三种交通负载（Low、Medium、High）和三随机种子；
- 四个基线对照（Fixed‑Cycle、SCATS、AIM、GLOSA）。

**📊 数据集**

使用的数据集为仿真生成的交通流数据：每个负载场景下的车辆生成率、路口几何结构、车辆参数（长度、加减速度、优先级、能耗偏好）等，均在 SUMO 环境中产生，并通过多次随机种子重复实验。

**📈 对比分析**

比较方法：在相同路口拓扑和交通负载下，分别运行 LISA 与四个基线，共 45 次仿真（5 控制器 × 3 负载 × 3 种子），对交通效率（吞吐量、平均控制延迟、平均等待时间、平均速度）、排队抑制（平均/峰值队列长度）、意图满足率（空间、时间、优先级、能耗四维加权）以及能耗/动能损失进行统计。
性能表现：
- 在 Medium 负载下，LISA 的平均控制延迟下降 89.1%（从 87.2 s 降至 22.2 s），并保持 LOS C；
- 通过率/意图满足率最高，达到 86.2%；
- 平均等待时间几乎为 0，平均速度最高；
- 燃油消耗相比固定周期下降 48.8%。

**⚠️ 局限性**

limitations：
1) 仅在单一四向交叉口进行验证，未验证多交叉口或网络级协同；
2) LLM 查询周期固定 30 s，未对不同负载下的最优频率进行系统分析；
3) 仍受 LLM 推理延迟限制，需进一步优化缓存失效策略；
4) 对提示语或输出模式的鲁棒性未进行全面测试；
5) 未探讨 V2I 覆盖率不足或车辆采纳率低的情境。

---

## 734. In-context learning to predict critical transitions in dynamical systems

**arXiv ID:** 2605.12308 | [PDF](https://arxiv.org/pdf/2605.12308v1)

**作者:** Yunus Sevinchan `[一作]` (kausable), Benjamin Herdeanu `[通讯]` (kausable)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了TipPFN框架，利用Transformer‑based Prior‑Data Fitted Network 通过上下文学习（ICL）直接预测时间序列的相对临界距离（RDTC），实现跨系统、跨机制的早期转变预警。

**💡 创新点**

创新点在于：① 用大规模合成数据训练的PFN实现可迁移的动态特征学习，② 引入连续的RDTC作为监督目标，③ 设计可在有或无上下文的零样本情形下工作，④ 通过因果注意力保证未来信息不泄露，⑤ 在多域多机制下展示出强泛化能力。

**🔧 技术方法**

技术包括：Transformer‑based PFN（TipPFN）与TabPFN、ICL（in‑context learning）、先验拟合网络（prior‑data fitted network）、RDTC指标、量化预测与贝叶斯后验逼近、合成数据生成器（嵌入三类分叉系统的高维随机SDE）、因果注意力机制、pinball 损失与量化回归。

**📊 数据集**

数据集：约120k个先验样本生成1M训练任务；验证集包含12个模型族（折、霍普夫、跨临界等）与9个观测系统，涵盖synthetic、semi‑real、sim‑to‑real 与 real‑world 数据，包括气候（AMOC）、生态（Daphnia）、神经（iEEG）、电力网等多领域。

**📈 对比分析**

与经典EWS（AR1、方差、偏度）、ML基线（Bury、Huang、Zhuge）、TabPFN2.6 等进行比较，评估指标为AUROC、ROC曲线与领先时间。TipPFN在所有系统与机制上均取得最高或接近最高AUROC，尤其在未见转变、r‑tipping、零样本和多上下文场景下表现优异。

**⚠️ 局限性**

局限性：目前未实现在线实时自适应更新，空间相关性未纳入，极稀疏观测下性能仍受限，需要更高质量的合成先验来支持特定领域的细粒度预测。

---

## 735. Executable Agentic Memory for GUI Agent

**arXiv ID:** 2605.12294 | [PDF](https://arxiv.org/pdf/2605.12294v1)

**作者:** Zerui Qin `[一作]` (Tsinghua University), Ju Ren `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Executable Agentic Memory (EAM) 框架，将 GUI 规划从自由生成转为基于结构化知识图的检索与执行流程。

**💡 创新点**

创新点在于将知识图建模为可执行状态机，并结合 Q‑guided MCTS 与自训练实现可检索且可执行的路径规划，同时给出了偏差一致性与样本复杂度理论保证。

**🔧 技术方法**

技术方法包括状态感知 DFS 探索、动作组挖掘、知识图构建、轻量化 Q 网络、MCTS 与自训练循环。

**📊 数据集**

使用了 AndroidWorld、MobileMiniWob++、DroidTask 三大基准数据集进行评估。

**📈 对比分析**

与多种本地与云端 SOTA 进行对比，成功率分别提升 19.6%、22.8%、31.1%，平均延迟 2.8 s，token 成本比 GPT‑4o 降低 6 倍。

**⚠️ 局限性**

局限在于依赖相对静态 UI，应用更新后知识图需要重建，缺乏高效增量更新机制。

---

## 736. Targeted Neuron Modulation via Contrastive Pair Search

**arXiv ID:** 2605.12290 | [PDF](https://arxiv.org/pdf/2605.12290v1)

**作者:** Sam Herring `[一作]` (Nous Research), Karan Malhotra `[通讯]` (Nous Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出对比神经元归因（CNA）方法，通过前向推理识别并消融模型中区分有害与正常提示的稀疏MLP神经元，从而降低拒绝率。

**💡 创新点**

创新点在于仅通过前向激活差异即可定位少量（0.1%）关键信号神经元，并证明对齐微调将预训练的判别结构转化为可控的拒绝门控，而残差流方法会牺牲生成质量。

**🔧 技术方法**

采用前向激活对比归因、神经元级消融与倍率调节（m），并与残差流方法CAA进行对比。

**📊 数据集**

使用Llama与Qwen系列模型（1B–72B）以及JBB‑Behaviors 100个有害/100个正常提示、MMLU问答数据集进行评测。

**📈 对比分析**

通过对比CNA与CAA在不同强度下的拒绝率与生成连贯性（n‑gram重复率）以及MMLU准确率，结果显示CNA在保持近基线质量的前提下，拒绝率下降超过50%，CAA在高强度下质量急剧恶化。

**⚠️ 局限性**

局限性包括仅针对LLM的门控行为、对比归因未提供可解释性度量、仅评估了Llama与Qwen架构，且在极高放大倍率下仍可能出现“安全门”失控。

---

## 737. Reimagining Assessment in the Age of Generative AI: Lessons from Open-Book Exams with ChatGPT

**arXiv ID:** 2605.12363 | [PDF](https://arxiv.org/pdf/2605.12363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 738. Fast Obligation Translation and Synthesis

**arXiv ID:** 2605.12372 | [PDF](https://arxiv.org/pdf/2605.12372v1)

**作者:** Alexandre Duret-Lutz `[一作]` (EPITA), Shufang Zhu `[通讯]` (University of Liverpool)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对 LTL 的 syntactic obligations（句法义务）构造最小化的确定性弱 ω-自动机（DWA），并在 MTBDD（多端点二叉决策图）表示上实现在线合成与游戏求解。

**💡 创新点**

创新点：① 直接使用 MTBDD 构造 DWA，避免传统的非确定性 NBA + powerset + Safra 变换；② 在 MTBDD 结构上实现 SCC 分解与最小化，保证得到最小 DWA；③ 将弱 Büchi 游戏直接在 MTBDD 结构上求解，省去显式图的生成，显著提升合成速度。

**🔧 技术方法**

主要技术：MTBDD、DWA、弱 Büchi 游戏、SCC 分解、Moore 分区、BDD 组合运算、命题等价化简化、MTBDD 基于分层的状态转移表示、在 MTBDD 上的游戏求解与回溯。

**📊 数据集**

数据集：共 494 条 syntactic obligations，来源于 17 个可扩展模式（310 条）、文献收集（76 条）以及合成竞赛（SyntComp）中的 108 条（除 2 条同义）。此外，还与 Spot 2.14、Couvreur、Safra、Safa、Owl 21.0 等工具在同一数据集上进行对比。

**📈 对比分析**

比较方法：在同一台 Core i7-3770 机器上使用 BenchExec 计时，统一输出 HOA 格式。实验显示：① MTBDD 翻译（mttrans/mtbddmin）在 494 条公式上与旧 Pipeline（couvreur + powerset + minimization）相比，翻译时间提升 3–4 倍；② 在合成阶段，MTBDD 直接求解游戏的运行时比传统的 DPA+explicit game 下降 50% 以上，且内存占用也更低；③ 与第三方工具（Safa, Owl 21.0, Syft）对比，Spot 2.15 在 syntactic obligations 的可实现性判定上往往更快或相当。

**⚠️ 局限性**

局限性：① 目前仅针对 syntactic obligations；对一般 LTL 的支持仍需进一步研究（如 Δ₂‑规范化、DELA 转换等）；② 将 MTBDD 转为显式图（必要时用于输出 HOA）仍耗时，导致在部分工作流中整体时间提升；③ 对于极大状态空间的公式，MTBDD 可能出现二进制决策图膨胀的问题。

---

## 739. Early AI Literacy in Culturally Responsive STEM Outreach for Black Youth

**arXiv ID:** 2605.12355 | [PDF](https://arxiv.org/pdf/2605.12355v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 740. Real-Time Whole-Body Teleoperation of a Humanoid Robot Using IMU-Based Motion Capture with Sim2Sim and Sim2Real Validation

**arXiv ID:** 2605.12347 | [PDF](https://arxiv.org/pdf/2605.12347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 741. Classifier Context Rot: Monitor Performance Degrades with Context Length

**arXiv ID:** 2605.12366 | [PDF](https://arxiv.org/pdf/2605.12366v1)

**作者:** Sam Martin `[一作]` (Anthropic Fellows Program), Fabien Roger `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估语言模型监控器在处理长达数十万甚至百万令牌的 Agent 记录时识别危险行为的召回率，并探索多种提示、增量监控、提醒及微调技术以缓解长上下文导致的“context rot”。

**💡 创新点**

首次系统性揭示了长上下文会导致监控召回率显著下降（最高可达 30 倍），并量化了提示设计、增量监控和提醒等缓解措施的有效性；同时证明仅凭微调难以跨域提升长上下文监控性能。

**🔧 技术方法**

使用 Claude Opus 4.6、Gemini 3.1 Pro 与 GPT 5.4 作为监控器，实验了思考（Chain‑of‑Thought）、列表式提示、增量监控、在每个工具调用后插入提醒提示以及对 GPT‑4.1 进行的微调等技术。

**📊 数据集**

实验数据集包括 148 条 MonitorBench 含攻击的转录、80 条无攻击的 Claude Code 转录，以及通过脚本注入“needle”攻击的合成转录；所有转录均包含数千到上百万令牌的长文本。

**📈 对比分析**

通过在不同前缀长度、攻击位置、思考令牌预算以及监控策略下测量召回率、AUC 等指标进行对比。结果显示：长上下文下召回率从 98% 降至 88%，在 800k 令牌前缀下甚至几乎为 0；增量监控、更多思考以及提醒提示可将召回率提升 10–30% 以上，微调在相近任务上有提升但不具备跨域泛化。

**⚠️ 局限性**

局限性：微调样本极少且未使用思考，无法覆盖多步跨全上下文的攻击；模型校准误差可能影响召回率评估；仅关注短时攻击，未评估更复杂的多阶段攻击；未尝试大规模、跨域的长上下文微调或更强的训练方法。

---

## 742. Poly-SVC: Polyphony-Aware Singing Voice Conversion with Harmonic Modeling

**arXiv ID:** 2605.12310 | [PDF](https://arxiv.org/pdf/2605.12310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 743. Attacks and Mitigations for Distributed Governance of Agentic AI under Byzantine Adversaries

**arXiv ID:** 2605.12364 | [PDF](https://arxiv.org/pdf/2605.12364v1)

**作者:** Matthew D. Laws `[一作]` (Northeastern University), Cristina Nita-Rotaru `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

分析了在Byzantine攻击者情况下SAGA治理服务的脆弱性，并提出了三种不同安全架构（完全BFT、轻量级监控/审计、混合架构）进行对比评估。

**💡 创新点**

创新点在于：①提出完全BFT版SAGA以消除对可信中心的依赖；②设计基于日志监控和客户端审计的轻量化防御，给出检测概率与延迟的定量分析；③构建可混合使用BFT与监控/审计的分片架构，实现安全性与性能的可调权衡。

**🔧 技术方法**

使用技术包括PBFT/Tendermint BFT共识、Raft复制、数据库分片、TLS会话密钥注入、日志收集与验证、客户端审计器、路由层分片管理。

**📊 数据集**

实验数据主要来自自建的GPT‑5.4、GPT‑5.4‑mini、Qwen3‑VL‑30B‑A3B‑Instruct‑FP8等大语言模型生成的工作负载，并使用RethinkDB（Raft复制）和BigchainDB（BFT数据库）模拟存储层；未使用公开数据集。

**📈 对比分析**

与原始SAGA比较：BFT版吞吐量约1%，监控版≈95%，审计版≈85%；混合版在保持大部分吞吐量的同时，可通过分片配置实现不同安全等级；在跨区域部署时，BFT版显著提升安全但延迟升高。

**⚠️ 局限性**

局限性：监控/审计检测存在时间延迟；BFT实现对高并发时的通信开销极大；若攻击者能区分审计客户端与普通用户，审计效果降低；实验环境为容器化实验，未覆盖真实云多租户复杂性。

---

## 744. A Family of Quaternion-Valued Differential Evolution Algorithms for Numerical Function Optimization

**arXiv ID:** 2605.12362 | [PDF](https://arxiv.org/pdf/2605.12362v1)

**作者:** Gerardo Altamirano-Gomez `[一作]` (Universidad Nacional Autónoma de México), Carlos Ignacio Hernández Castellanos `[通讯]` (Universidad Nacional Autónoma de México)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种在四元数空间中进行搜索的差分进化算法（QDE），并设计了欧氏与极坐标两种初始化以及六种基于哈密顿乘积的变异策略。

**💡 创新点**

首次将四元数代数与极坐标旋转映射融入进化搜索，提出PM1/PM3等极坐标变异和EGSD等多哈密顿积变异，利用四元数的几何旋转特性实现更高效的搜索。

**🔧 技术方法**

采用四元数运算（哈密顿乘积、单位四元数变换、极坐标表示）构建算法，并使用Friedman检验与Nemenyi后验检验对性能进行统计比较；实验基于BBOB基准套件进行。

**📊 数据集**

实验数据来源于BBOB基准中的24个连续优化函数（分为可分离、低/高条件单峰、多峰有/无全局结构），主要在3维/4维问题上进行评估。

**📈 对比分析**

通过20次独立实验记录每代最优适应度，计算中位数与标准差，再对所有函数和各函数组做Friedman+Nemenyi比较；结果显示PM1/PM3在高条件单峰和多峰有全局结构组上显著优于实数DE，整体收敛速度普遍更快，而在可分离和低条件单峰组性能与实数DE相当。

**⚠️ 局限性**

该方法受限于只能处理维度为3或4（或其四分之一的）问题，缺乏理论收敛保证，且在部分U‑Low和M‑Weak组表现不佳；未在大规模实际工程优化任务中验证，对非欧几里得空间的适用性亦待进一步研究。

---

## 745. Output Composability of QLoRA PEFT Modules for Plug-and-Play Attribute-Controlled Text Generation

**arXiv ID:** 2605.12345 | [PDF](https://arxiv.org/pdf/2605.12345v1)

**作者:** Michela Lorandi `[一作]` (Dublin City University), Anya Belz `[通讯]` (Dublin City University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将多任务PEFT模块在同一主模型中进行输出级组合，以实现多属性文本生成控制。

**💡 创新点**

提出了输出求和/平均的模块组合方法，并在QLoRA框架下实现可插拔的多模块组合；验证输出求和可提升单任务及多任务性能。

**🔧 技术方法**

采用QLoRA PEFT技术，结合多种输出/权重组合方式，对LLaMA3 8B/3.1 8B和Mistral 7B进行实验。

**📊 数据集**

使用情感控制数据集（Yelp、IMDB、SST‑2）和主题控制数据集（AG News、DBpedia），以及PPLM、STS等外域数据进行测试。

**📈 对比分析**

通过与单任务微调、合并数据集微调以及权重平均等基线比较，发现输出求和在多任务控制下平均提升约2% CE，且在大部分指标上优于其他组合方式。

**⚠️ 局限性**

局限在于仅测试了三款相似规模模型、仅两种控制属性、仅QLoRA一个PEFT技术，且对大规模模块组合、不同任务类型及人类评估未展开。

---

## 746. KAN-CL: Per-Knot Importance Regularization for Continual Learning with Kolmogorov-Arnold Networks

**arXiv ID:** 2605.12306 | [PDF](https://arxiv.org/pdf/2605.12306v1)

**作者:** Minjong Cheon `[一作]` `[通讯]` (Sejong University), Minjong Cheon (Sejong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种持续学习框架 KAN-CL，利用 KAN 的分段 B‑spline 参数化实现按节点（knot）级别的重要性加权锚定，并在卷积骨干上使用 EWC 正则化。

**💡 创新点**

创新点在于：①在 KAN 的 compact‑support spline 结构上实现细粒度 per‑knot 正则化，②与骨干 EWC 组合形成模块化方法，③给出基于 Neural Tangent Kernel 的遗忘上界证明，说明结构局部性可降低跨任务耦合。

**🔧 技术方法**

技术细节包括：Kolmogorov–Arnold Network (KAN) 的 B‑spline 激活；按节点 Fisher 与激活质量计算重要性；L2 锚定与梯度屏蔽；骨干上的 bbEWC；anchor 退火用于 class‑IL+replay；以及 NTK 分析验证跨任务耦合低。

**📊 数据集**

实验数据集：Split‑CIFAR‑10/5T、Split‑CIFAR‑100/10T、Permuted‑MNIST、Rotation‑MNIST、Split‑MNIST；在 Task‑IL、Domain‑IL 与 Class‑IL (带 replay) 三种协议下评估。

**📈 对比分析**

与 Finetune、EWC、SI、MLP+、KAN‑CL（仅 head）等基线比较；在 CIFAR 基准上，KAN‑CL+bbEWC 在保持或提升平均准确率的同时，将遗忘率降低 88%–93%，实现 Pareto 取向的稳定性‑可塑性优势。

**⚠️ 局限性**

局限性包括：①骨干与 head 的容量匹配需要手工调节，②正则化超参需基于验证集调优，③理论假设的支持域互不相交在类分割任务中仅近似成立，④适用于可变网格 KAN 的扩展尚未实现，⑤NTK 在任务切换时的非平稳性可能影响理论上界的实用性。

---

## 747. Images in Sentences: Scaling Interleaved Instructions for Unified Visual Generation

**arXiv ID:** 2605.12305 | [PDF](https://arxiv.org/pdf/2605.12305v1)

**作者:** Yabo Zhang `[一作]` (ByteDance Seed), Xun Wang `[通讯]` (ByteDance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的图像生成与编辑框架，将图像嵌入文本为原生词汇，支持多图像交互式生成与编辑；同时构建了可扩展的15M高质量交互式数据引擎和 InterleaveBench 基准；

**💡 创新点**

创新点在于原生图像嵌入（native interleaved formulation）通过Transformer的上下文局部性实现精准对象绑定；两阶段文本‑视觉平衡的 classifier‑free guidance；以及基于 VLM、SAM、DAM 的自动化数据合成流程，生成大规模多图像交互式样本；

**🔧 技术方法**

采用 Mixture‑of‑Transformer 架构（基于 BAGEL），仅使用 ViT 视觉嵌入；LLM 生成交互式指令；VLM+SAM+DAM 进行对象检测与描述；视频对齐与动态状态过滤；两阶段 guidance 调节文本/视觉权重；

**📊 数据集**

数据集方面：自建 15M interleaved 样本（来自公开图像/视频）；使用 DreamBench++ 等资源筛选高质量参考图像；构造 InterleaveBench benchmark 用于复杂多图像任务评估；

**📈 对比分析**

在 InterleaveBench 上与多种开源模型（DreamOmni2、Flux‑Kontext 等）及闭源模型对比，模型在图像一致性与文本一致性上均取得领先，尤其在输入图像数增多时优势更明显；整体性能超过所有开源方法，逼近闭源模型；

**⚠️ 局限性**

局限性包括：文本一致性仍略低于顶级闭源模型，受限于底层文本生成能力；对极端多图像或长序列场景的鲁棒性尚未充分验证；模型规模虽小，但在更大参数空间下的潜在表现未测试；

---

## 748. Approximation of Maximally Monotone Operators : A Graph Convergence Perspective

**arXiv ID:** 2605.12301 | [PDF](https://arxiv.org/pdf/2605.12301v1)

**作者:** Takashi Furuya `[一作]` (Doshisha University), Takaharu Yaguchi `[通讯]` (Kyushu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究了在图收敛框架下对最大单调算子进行统一近似的理论与实验，提出将编码‑解码网络作为逼近工具；

**💡 创新点**

创新点在于①提出图收敛（Painlevé–Kuratowski）作为闭算子学习的自然拓扑；②证明任何最大单调算子均可通过连续编码‑解码架构在局部图距离下逼近；③构造保持最大单调性的结构化逼近模型；

**🔧 技术方法**

使用Yosida逼近、解析子差分、1‑Lipschitz编码‑解码网络、软图损失以及基于解析子图正则化的结构化网络；

**📊 数据集**

使用基于傅里叶级数的合成数据，模拟一维求导算子和二维p‑Laplacian算子；

**📈 对比分析**

与传统的ℓ²、ℓ∞损失以及软图损失的FNO模型对比，软图+结构化模型在测试误差、图距离和单调性违背指标上均优于其他方法；

**⚠️ 局限性**

仅在局部图收敛下成立，缺乏全局一致性；图损失计算量高；量化误差上限不够显式；实验仅验证单值算子，未覆盖真正多值算子。

---

## 749. GKnow: Measuring the Entanglement of Gender Bias and Factual Gender

**arXiv ID:** 2605.12299 | [PDF](https://arxiv.org/pdf/2605.12299v1)

**作者:** Leonor Veloso `[一作]` (Lmu Munich), Hinrich Schütze `[通讯]` (Lmu Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了多任务性别相关基准 GKnow，并使用 EAP‑IG 电路分析技术对 Llama‑3.1‑8B 与 Olmo‑7b 中的性别偏见与事实性别耦合进行深入研究，随后通过 Integrated Gradients 识别关键神经元并进行 neuron ablation，评估其对偏见与事实性别的影响。

**💡 创新点**

创新点在于①提出兼顾事实性别与刻板性别的综合评测基准 GKnow；②首次在电路层和神经元层揭示性别偏见与事实性别高度耦合；③通过 ablation 实验证明仅关注偏见指标的去偏方法会损害事实性别知识，提示评测框架需兼顾两者。

**🔧 技术方法**

主要技术包括 EAP‑IG 电路发现、Integrated Gradients 神经元重要性评估、logit lens 可解释性分析、以及在 StereoSet、DiFair 等 benchmark 上的 neuron ablation 实验。

**📊 数据集**

使用的数据集为自构建的 GKnow（约 7,000 条样例）、StereoSet 的 intrasentence 子集（约 104 条）和 DiFair（Neutral 与 Specific 合计 112 条）以及两款模型 Llama‑3.1‑8B 与 Olmo‑7b。

**📈 对比分析**

与未改动模型对比，10/50 个 IG 神经元 ablation 后在性别偏见评测中可提升反性别概率（P_opp 增大），但在事实性别评测中显著降低 P_exp 与 Δ_f,m；跨任务 faithfulness 与 IoU 分析显示性别偏见与事实性别电路高度相似，验证了两者耦合。

**⚠️ 局限性**

研究局限包括仅针对英语二元性别、只涵盖职业与形容词型刻板印象、仅使用较小模型，且未覆盖隐性偏见与非二元性别。

---

## 750. PriorZero: Bridging Language Priors and World Models for Decision Making

**arXiv ID:** 2605.12289 | [PDF](https://arxiv.org/pdf/2605.12289v1)

**作者:** Junyu Xiong `[一作]` (University of Science and Technology of China), Yazhe Niu `[通讯]` (Chinese University of Hong Kong MMLab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PriorZero 框架，将大型语言模型（LLM）的语义先验通过根节点优先注入（Root‑Prior Injection）与世界模型（world‑model）相结合，在 MCTS 搜索中引导探索，并通过交替细化（Alternating Fine‑Tuning）实现 LLM 与世界模型的闭环协同学习。

**💡 创新点**

核心创新点在于：①只在 MCTS 根节点注入 LLM 先验，既利用 LLM 语义知识，又避免了与动态建模的粒度不匹配；②将世界模型的价值估计作为低方差优势信号，用于稳定地对 LLM 进行强化学习微调，从而克服了直接 RL‑FT 的信用分配困难；③通过异步交替训练实现 LLM 与世界模型的互补提升。

**🔧 技术方法**

技术组合包括：Transformer‑based 隐层世界模型、Latent MCTS、链式推理（Chain‑of‑Thought）提取 LLM 先验、PPO‑style RL‑FT、低方差优势估计（n‑step TD 加值回归）、KL 正则化与 CoT 损失权重控制。

**📊 数据集**

实验数据集主要有：Jericho 文本冒险游戏（Detective、Acorncourt、Zork1、Omniquest）和 BabyAI 网格世界（18 级任务）。

**📈 对比分析**

与 UniZero 以及四种 LLM 知识迁移范式（Naïve Policy、RLFT、Dynamics Model、Text Encoder）进行对比。实验表明 PriorZero 在所有 Jericho 环境中实现了更快的收敛速度、更高的最终回报，并在 BabyAI 上显著提升了探索效率和最终得分，尤其在长路径与组合任务上优于基线。

**⚠️ 局限性**

局限性包括：①对高质量 LLM 的依赖，若 LLM 先验不足或规模有限可能无法显著提升；②交替训练需要调节多项超参数，易产生不稳定性；③计算成本高，尤其是 MCTS 与大型 LLM 的联合推理；④在非文本/结构化任务中的泛化能力尚未完全验证。

---

## 751. TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching

**arXiv ID:** 2605.12288 | [PDF](https://arxiv.org/pdf/2605.12288v1)

**作者:** Truong Nguyen `[一作]` (Hanoi University of Science and Technology), Trung Le `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Token-level Bregman Preference Optimization (TBPO)，通过令模型在每个前缀上做token级偏好优化；

**💡 创新点**

在token层面显式建模偏好并以Bregman散度匹配密度比，从而保证每一步决策都满足偏好最优性；

**🔧 技术方法**

使用Bradley–Terry模型、Bregman散度、状态-动作价值或优势函数、KL正则化以及轻量级基线估计；

**📊 数据集**

利用ultrafeedback_binarized、llama3-ultrafeedback-armorm等大规模对比偏好数据；

**📈 对比分析**

与SFT、DPO、TDPO、TIS-DPO、BPO等方法对比，在Mistral7B和Llama3-8B上在多项任务上平均提升约1–2分，生成质量与多样性均有显著改善；

**⚠️ 局限性**

对极端长文本的偏好推断仍依赖序列级比较，且对高噪声偏好数据的鲁棒性有待进一步验证。

---

## 752. $δ$-mem: Efficient Online Memory for Large Language Models

**arXiv ID:** 2605.12357 | [PDF](https://arxiv.org/pdf/2605.12357v1)

**作者:** Jingdi Lei `[一作]` (Nanyang Technological University), Soujanya Poria `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在冻结的完整注意力Transformer基础上，提出一种紧凑的在线关联记忆（OSAM）机制，在不增大上下文窗口或使用外部检索的情况下动态维护历史信息，并通过低秩修正直接调节注意力；

**💡 创新点**

创新点在于：①将历史信息压缩为固定大小的矩阵状态；②使用delta‑rule与忘记门在线更新；③通过读出的记忆信号产生低秩查询与输出侧的注意力修正，实现在前向计算中直接利用记忆；

**🔧 技术方法**

技术包括：关联记忆投影（将隐藏状态映射到键值向量）、带忘记门的delta‑rule学习、低秩注意力修正（query‑side和output‑side），以及三种写入粒度（TSW、SSW、MSW）；

**📊 数据集**

评估数据集涵盖一般推理与指令跟随：HotpotQA、GPQA‑Diamond、IFEval；以及“记忆密集”基准：MemoryAgentBench、LoCoMo；

**📈 对比分析**

与文本记忆（BM25 RAG、LLMLingua‑2、MemoryBank）、参数记忆（Context2LoRA、MemGen）、外部通道记忆（MLP Memory）以及SFT无记忆模型比较，平均分提升约4.9%（从46.79%到51.66%），在MemoryAgentBench、LoCoMo、HotpotQA等记忆任务上显著提升，最高可达约1.3‑倍；

**⚠️ 局限性**

局限性包括：仍受限于固定尺寸的记忆状态（8×8）可能无法覆盖更复杂或长周期的历史；写入粒度和多状态组织需要手工设计；对不同模型容量的适配性尚需进一步验证；

---

## 753. Will My Favorite Chases Terminate if Evaluating Conjunctive Queries Does? One Does Not Simply Decide This

**arXiv ID:** 2605.12349 | [PDF](https://arxiv.org/pdf/2605.12349v1)

**作者:** Lucas Larroque `[一作]` (Inria), Quentin Manière `[通讯]` (LIRMM)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

构造了一类具备可判定BCQ判定但其chase终止性与成员判定仍不可判定的存在性规则集，并证明此类规则集的BCQ判定可判定。

**💡 创新点**

首次揭示在具有可判定BCQ判定的具体规则类中，chase终止性与成员判定仍保持不可判定，说明现有的类特定技术无法通过BCQ判定来推断终止性。

**🔧 技术方法**

使用Minsky机模拟、改进的chase变体（oblivious、semi-oblivious、restricted、core）、洪泛机制以及树宽分析等理论技术。

**📊 数据集**

无实验数据集，研究完全基于理论证明。

**📈 对比分析**

未进行实验对比；所有结论均来自形式化证明，未给出运行时间或性能指标。

**⚠️ 局限性**

局限性在于仅给出负面结果，无法给出判定终止性的通用方法；结果仅适用于构造的特定规则类，尚未推广到更广泛的规则集。

---

## 754. Neural-Schwarz Tiling for Geometry-Universal PDE Solving at Scale

**arXiv ID:** 2605.12343 | [PDF](https://arxiv.org/pdf/2605.12343v1)

**作者:** Paolo Secchi `[一作]` (Imperial College London), Marco Maurizi `[通讯]` (Italian Institute of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于局部神经算子与 Schwarz 迭代相结合的 NEST 框架，用来在未见过的大型 3D 结构上求解非线性弹性 PDE；

**💡 创新点**

创新点在于将 PDE 求解拆分为可重用的最小体素块（3×3×3）局部神经算子与全局域分解（Schwarz）组合，实现在几何、尺度和边界条件变化下的无须重训练的泛化；

**🔧 技术方法**

使用 Graph Neural Operator（GNO）作为局部算子，配合重叠 Schwarz 迭代、分区加权拼接，以及对梯度的单独预测；

**📊 数据集**

主要数据集为 15,000 个随机 3×3×3 体素块的 FEM 求解；在测试时使用两类大型几何：SimJEB 航发机架和两种 TPMS 结构，分辨率从 30³ 到 60³；

**📈 对比分析**

与全局神经算子（Transolver、GNO）在相同宏观数据生成预算下对比；NEST 在不需要宏观训练的前提下，误差与全局算子相当或更低，且在梯度预测上优于全局算子；

**⚠️ 局限性**

局限性包括：Schwarz 收敛速度仍受限；目前仅处理 Dirichlet 边界且仅针对非线性静力学；在更复杂多物理耦合或更大尺度问题时的收敛和计算效率需进一步提升。

---

## 755. Black-Box Optimization of Mixed Binary-Continuous Variables: Challenges and Opportunities in Evolutionary Model Merging

**arXiv ID:** 2605.12326 | [PDF](https://arxiv.org/pdf/2605.12326v1)

**作者:** Md. Robiul Islam Niloy `[一作]` `[通讯]` (BRAC University), Md. Robiul Islam Niloy (BRAC University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了进化模型合并中数据流空间（DFS）的优化问题，先将DFS合并形式化为混合二进制‑连续变量的黑盒优化问题，并在两款预训练语言模型上进行实验验证。

**💡 创新点**

创新点：①系统整理进化模型合并技术；②首次将DFS合并问题建模为含条件依赖的混合二进制‑连续变量优化问题；③提出结构化DFS搜索策略，实验证明比无结构DFS提升6.7%准确率并将搜索空间压缩51.4%。

**🔧 技术方法**

主要技术包括进化算法（CMA‑ES、CatCMA概念）、混合变量优化框架、结构化DFS搜索策略以及基于随机采样的实验评估。

**📊 数据集**

使用的模型为 HuggingFace 上的 flan‑t5‑small（76M 参数）和 flan‑alpaca‑large（247M 参数），评估基准为 15 题事实推理与算术测试集。

**📈 对比分析**

对比了 Model A、PS 合并、无结构 DFS 与结构化 DFS 四种方案；结构化 DFS 在 10 次迭代内达 26.7% 准确率，等同于 Model A，较无结构 DFS 高 6.7%，且有效搜索空间缩减 51.4%。

**⚠️ 局限性**

限制：实验规模小（仅 15 题），所用模型体量低于生产级 LLM，搜索采用随机采样而非成熟进化器，未对更大规模模型或更复杂基准进行验证。

---

## 756. Transferable Delay-Aware Reinforcement Learning via Implicit Causal Graph Modeling

**arXiv ID:** 2605.12312 | [PDF](https://arxiv.org/pdf/2605.12312v1)

**作者:** Chenran Zhao `[一作]` (National University of Defense Technology), Shaowu Yang `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种名为CausalDreamer的延迟感知强化学习框架，利用隐式因果图和字段节点编码实现对随机延迟环境的建模与策略学习；

**💡 创新点**

其创新点在于：①将观测拆分为语义字段节点并通过隐式因果图进行信息交换；②引入延迟对齐机制和熵正则化来处理随机延迟；③通过模型生成的想象轨迹实现高效的策略优化；

**🔧 技术方法**

技术包括模型基强化学习、隐式因果图建模、消息传递（Message‑Passing）机制、延迟对齐门控、DreamerV3风格的想象式策略学习；

**📊 数据集**

实验采用DeepMind Control Suite的连续控制任务，并在每个任务上添加随机观测延迟（最大延迟5）；

**📈 对比分析**

与DreamerV3、CWM、SAC等基线相比，CausalDreamer在无延迟环境下表现最佳，在随机延迟环境下性能衰减最小；并且在跨任务迁移实验中，迁移初始化的模型在样本效率和最终回报上均优于从零训练；

**⚠️ 局限性**

局限性包括：依赖结构化的本体感知观测，对高维图像观测的适用性不明；延迟对齐门控需要手工设定窗口长度；在某些任务中不同字段分区策略对性能影响较大，且仍无法完全避免因果推断误差导致的性能波动。

---

## 757. STRABLE: Benchmarking Tabular Machine Learning with Strings

**arXiv ID:** 2605.12292 | [PDF](https://arxiv.org/pdf/2605.12292v1)

**作者:** Gioia Blayer `[一作]` (INRIA Saclay), Gaël Varoquaux `[通讯]` (INRIA Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了STRABLE基准库，收集了108个包含字符串和数值的真实世界表格数据，系统评估了445个表格学习管道，探究字符串处理对模型性能的影响。

**💡 创新点**

①首次为包含字符串的表格学习提供大规模、可复现的基准；②发现模块化管道（轻量编码+高级学习器）在性能与计算成本上超过端到端模型；③阐明大型LLM编码在不同后处理（PCA、标准化）下对性能的显著影响。

**🔧 技术方法**

使用了多种字符串编码器（Tf‑Idf、FastText、句子Transformer、各类LLM）、降维技术（PCA、标准化+PCA、截断）、多种表格学习器（Ridge、ExtraTrees、XGBoost、RealMLP、TabM、TabICLv2、TabPFN‑2.5）以及端到端模型（CatBoost、Mambular、TabSTAR、ContextTab）。

**📊 数据集**

STRABLE数据集共108个表格，涵盖13二分类、19多分类和76回归任务，来源于8个应用领域，表格包含至少两列字符串、500行以上。

**📈 对比分析**

通过Kendall‑τ等排名相关性度量，模块化管道在所有指标上占优，轻量编码+TabPFN‑2.5在性能-运行时间 Pareto 前沿表现最佳；LLM编码需经过适当降维后才能与先进学习器匹配。

**⚠️ 局限性**

仅关注短字符串（中位数18字），对长文本表格的适用性有限；未覆盖时间序列验证方案；后处理方法（PCA）对decoder‑only LLM性能敏感，需改进降维策略。

---

## 758. From Model Uncertainty to Human Attention: Localization-Aware Visual Cues for Scalable Annotation Review

**arXiv ID:** 2605.12303 | [PDF](https://arxiv.org/pdf/2605.12303v1)

**作者:** Moussa Kassem Sbeyti `[一作]` (Karlsruhe Institute of Technology), Gerhard Satzger `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出在人工标注界面中可视化模型定位不确定性，以颜色编码提示标注员关注需要修正的框，从而提升标注质量与效率。

**💡 创新点**

创新点在于首次将本地化不确定性以直观的颜色梯度呈现给标注员，并证明该视觉线索能显著重定向注意力并改善标注表现。

**🔧 技术方法**

技术方案包括使用概率EfficientDet-D0进行边界不确定性估计、等距回归校准、GEE和LME统计模型评估实验结果，并在Web界面中实现颜色编码反馈。

**📊 数据集**

实验数据来源于KITTI自动驾驶验证集，选取15张图像（7类目标）进行标注，使用重新标注的金标准作为评价基准。

**📈 对比分析**

通过对照实验，基线与不确定性可视化组比较，后者在mIoU上提升约0.70个百分点，标注速度提高约7.2%，在中难度图像上效果尤为显著。

**⚠️ 局限性**

局限性包括仅在KITTI框框标注任务验证，模型校准质量决定可视化效果；低难度图像无显著收益，实验时间有限，且未评估在医学、卫星等其他领域的可迁移性。

---

## 759. GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization

**arXiv ID:** 2605.12369 | [PDF](https://arxiv.org/pdf/2605.12369v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 760. MedHopQA: A Disease-Centered Multi-Hop Reasoning Benchmark and Evaluation Framework for LLM-Based Biomedical Question Answering

**arXiv ID:** 2605.12361 | [PDF](https://arxiv.org/pdf/2605.12361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 761. From Message-Passing to Linearized Graph Sequence Models

**arXiv ID:** 2605.12358 | [PDF](https://arxiv.org/pdf/2605.12358v1)

**作者:** Joël Mathys `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Linearized Graph Sequence Models（LGSM），通过将图信息传播与非线性处理分离，将图学习视作序列建模，利用状态空间模型实现高效并行长距离信息传递。

**💡 创新点**

创新性在于将信息深度与处理深度解耦，使得信息传播可线性化、可并行化，并通过序列提取（如非退回走序列）实现更好的图结构保持与长程依赖；同时提供理论分析与实证验证。

**🔧 技术方法**

使用线性化消息传递、状态空间模型（如Mamba）、非退回走序列提取、前馈网络、图混合层等技术组合。

**📊 数据集**

在ECHO‑Synth（图属性预测）、ECHO‑Chem（分子属性预测）、LRIM 及 Peptides 等公开基准数据集上进行评测。

**📈 对比分析**

与多种基准模型（GCN、GCNII、GIN、GPS、SWAN、等）进行对比，LGSM 在长距离任务（eccentricity、SSSP）和分子属性（能源预测）上显著优于或与最先进方法持平，表现最优。

**⚠️ 局限性**

未系统评估序列预处理与并行实现的计算资源权衡，且在某些任务（如 Peptides）对性能提升有限，说明该方法的普适性仍需进一步验证。

---

## 762. A New Technique for AI Explainability using Feature Association Map

**arXiv ID:** 2605.12350 | [PDF](https://arxiv.org/pdf/2605.12350v1)

**作者:** Sayantani Ghosh `[一作]` (DBS Bank), Amlan Chakrabarti `[通讯]` (University of Calcutta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 FAMeX 算法，通过构建特征关联图（FAM）结合相关性与冗余评估特征重要性，提供 AI 系统可解释性；

**💡 创新点**

创新点在于同时考虑特征的相关性和冗余，利用图论分级评分并通过逆向关联求得特征重要性，区别于传统单一重要性度量；

**🔧 技术方法**

使用 Pearson 相关系数、互信息、图论构建特征关联图、并与 PFI、SHAP 等现有 XAI 方法对比，实验采用 SVM、Random Forest、Naïve Bayes、Decision Tree 四种分类器；

**📊 数据集**

使用 UCI 公开数据集八个基准（Wisconsin、ILPD、Pageblocks、Pima、Apndcts、WineQuality、WBDC、Vehicle）进行实验；

**📈 对比分析**

通过选取 top30%/bottom30% 重要特征训练四种分类器，比较分类准确率，结果表明 FAMeX 在所有分类器上平均准确率均高于 PFI 与 SHAP，验证了其优越性能；

**⚠️ 局限性**

局限性包括仅在小型 UCI 数据集验证，未测试大规模/复杂数据集，算法复杂度为 O(n²) 对高维数据可能受限，且未与更先进的 XAI 方法进行对比。

---

## 763. BSO: Safety Alignment Is Density Ratio Matching

**arXiv ID:** 2605.12339 | [PDF](https://arxiv.org/pdf/2605.12339v1)

**作者:** Tien-Phat Nguyen `[一作]` (Hanoi University of Science and Technology), Trung Le `[通讯]` (Monash University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的安全对齐框架 BSO，通过密度比匹配实现帮助性与安全性的单阶段优化。

**💡 创新点**

核心创新是将安全对齐视为对数似然比的闭式分解，并通过 Bregman 散度构造可证明收敛的损失函数，天然包含 SafeDPO 等现有方法。

**🔧 技术方法**

采用 Bregman 损失、密度比估计、数据变换（Swap/Drop）以及可调 Amplification 参数 λ 的 SBA 生成器，训练过程无需额外奖励/成本模型。

**📊 数据集**

在 PKU‑SafeRLHF‑30K 数据集上进行实验，并在 Qwen 2.5‑0.5B 与 Llama 3.2‑3B 两大模型骨干上评估。

**📈 对比分析**

与 SFT、SafeRLHF、SACPO、SafeDPO 等基线相比，BSO 在帮助性与安全性指标上同时提升，取得更优的 Pareto 前沿；实验还验证了安全惩罚 C 与 SBA 参数 λ 的中等取值最优。

**⚠️ 局限性**

局限性包括对二元安全标签的依赖、对安全惩罚 C 的数值稳定性敏感，以及在更复杂安全标注或多目标场景下的泛化能力待进一步研究。

---

## 764. Manifold Sampling via Entropy Maximization

**arXiv ID:** 2605.12338 | [PDF](https://arxiv.org/pdf/2605.12338v1)

**作者:** Cornelius V. Braun `[一作]`, Marc Toussaint `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

无法获取论文具体内容，无法说明研究所做的工作。

**💡 创新点**

无法确定创新点。

**🔧 技术方法**

无法确定所使用的技术。

**📊 数据集**

无法确定所使用的数据集。

**📈 对比分析**

无法比较方法与性能。

**⚠️ 局限性**

缺乏足够信息以评估限制。

---

## 765. Grid Games: The Power of Multiple Grids for Quantizing Large Language Models

**arXiv ID:** 2605.12327 | [PDF](https://arxiv.org/pdf/2605.12327v1)

**作者:** Vage Egiazarian `[一作]` (ISTA), Dan Alistarh `[通讯]` (ISTA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了在4-bit量化中使用多格子（PO2）自适应选择的方案，以提高LLM权重和激活的精度。

**💡 创新点**

创新点在于证明大块尺寸下多格子优势消失，提出基于分布的两格子对（PO2(NF4)、PO2(Split87)、MPO2）以及可硬件实现的SFP4。

**🔧 技术方法**

利用理论分析（凸性、集中极限定理）、分布拟合、Lloyd迭代、absmax归一化、四分位法、TensorCore兼容的Shifted NVFP4等技术。

**📊 数据集**

在多种LLM（Llama-3.2-3B、Qwen3-8/14B、Qwen3.5-2/4/9/27B）上进行PTQ和QAT实验，并使用WikiText-2、C4、Winogrande等公开数据集。

**📈 对比分析**

与单格INT4/FP4、NVFP4、NF4、IF4、BOF4、SFP4等基线相比，PO2(Split87)和PO2(NF4)在KL、EAR、下游任务准确率上提升约3–5%，SFP4实现了硬件可行的性能提升。

**⚠️ 局限性**

局限性包括需要额外的格子选择位或硬件支持、对不同模型的优劣波动、计算/内存开销提升以及在大块尺寸下收益有限。

---

## 766. VIP: Visual-guided Prompt Evolution for Efficient Dense Vision-Language Inference

**arXiv ID:** 2605.12325 | [PDF](https://arxiv.org/pdf/2605.12325v1)

**作者:** Hao Zhu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Feng Dai `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出VIP方法，在不训练的前提下利用dino.txt实现高效的开放词汇语义分割。

**💡 创新点**

核心创新在于用LLM扩展并视觉引导去噪的文本提示，并通过显著性加权聚合多提示激活，显著缓解跨模态不匹配。

**🔧 技术方法**

使用dino.txt自监督视觉编码器、GPT生成提示、视觉引导的别名蒸馏、显著性软聚合以及自校正注意力。

**📊 数据集**

在VOC、COCO、Cityscapes、ADE20K等自然图像基准以及iSAID、Potsdam、Vaihingen、VDD等遥感分割数据集上进行评测。

**📈 对比分析**

相较于CLIP基础与多模型SOTA，在八大自然图像基准上平均提升1.4%–8.4% mIoU，并在遥感任务上领先8.5%；推理速度比SOTA快14×，显存仅占一半。

**⚠️ 局限性**

依赖LLM生成提示且仍可能在极端视觉差异下产生少量噪声，且在极端小类别或稀有场景的精度提升有限。

---

## 767. Contrastive Learning under Noisy Temporal Self-Supervision for Colonoscopy Videos

**arXiv ID:** 2605.12320 | [PDF](https://arxiv.org/pdf/2605.12320v1)

**作者:** Luca Parolari `[一作]` (University of Padova), Loic Le Folgoc `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于时间自监督的对比学习框架，用于学习结肠镜视频中息肉轨迹的鲁棒表示。

**💡 创新点**

创新点在于：①利用结肠镜手术的时间序列结构生成自监督正样本；②引入噪声感知对比损失，能够抑制错误的时间关联对学习的负面影响；③在轻量化编码器上通过指数采样+课程学习动态增大正样本多样性。

**🔧 技术方法**

主要技术包括：Transformer编码器（带可学习token），指数分布采样生成时间邻域正样本，噪声感知对比损失（基于max/LogSumExp），以及在帧级和轨迹级双层损失优化。

**📊 数据集**

使用公开数据集：SUN、PolypSize、PolypsSet、以及从30段完整结肠镜视频中提取的27段训练视频（85个息肉）。

**📈 对比分析**

与多类基准（自监督、全监督、通用/专用基础模型）对比，方法在四个下游任务（检索、再识别、大小估计、组织学分类）均取得领先表现，mAP提升约23%，AUROC与AUPR也均显著优于现有自监督方法，并与最优全监督方法相当。

**⚠️ 局限性**

局限性包括：依赖视频时间顺序，若息肉出现频繁或相近仍可能产生误关联；模型规模虽小但仍需训练 27 条视频，实际临床部署需进一步验证；对极端光照或运动模糊的鲁棒性尚未系统评估。

---

## 768. Data-aware candidate selection in NL2SQL translation via small separating instances

**arXiv ID:** 2605.12319 | [PDF](https://arxiv.org/pdf/2605.12319v1)

**作者:** Stanislav Kikot `[一作]` (Huawei Technologies Co Ltd), Yanwei Xu `[通讯]` (Huawei Technologies Co Ltd)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于从模型数据库提取的最小分离实例和行级数据依赖的NL2SQL候选查询选择方法，并在BIRD-DEV子集上进行评估

**💡 创新点**

创新点在于将分离实例与执行追踪相结合，为LLM提供了可解释的、信息丰富的数据子集，从而显著提升候选选择的准确性

**🔧 技术方法**

使用ProvSQL实现行级执行追踪，利用Qwen3-Coder-30B进行自然语言答案评估，构造分离实例并通过双循环锦标赛进行候选竞争

**📊 数据集**

采用BIRD-DEV数据集（约1,534条任务），经筛选后得到四个子数据集（164、376、454、488任务）进行实验

**📈 对比分析**

通过与Consistency、Naive和DeepEye三种基线在K=2、5、11、24四个rollout设置下的执行准确率对比，实验显示在K=2时显著优于基线，在其他K值下与Naive相当，略低于DeepEye

**⚠️ 局限性**

受LLM回答幻觉影响，分离实例构造失败率导致比赛出现draw；ProvSQL对SQL功能有限制，限制了方法的完整性

---

## 769. Check, Please: Verifiably Fair Clustering

**arXiv ID:** 2605.12317 | [PDF](https://arxiv.org/pdf/2605.12317v1)

**作者:** Yu He `[一作]` (Northwestern University), Edith Elkind `[通讯]` (Northwestern University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了在质心聚类中验证比例代表性（mPJR）是否满足的计算复杂度，并提出了两种可验证的替代性公平性准则 mPJR+ 与 DC-mPJR+，其中 DC-mPJR+ 通过默认联盟（default coalition）构造实现了高效的验证算法；

**💡 创新点**

创新点在于：①证明了 mPJR 的验证问题是 coNP‑hard；②提出了可多项式可验证的 mPJR+，并说明其在实践中由于需要大量子模函数最小化而不够高效；③引入默认联盟概念，定义 DC-mPJR+，证明其在 O(mn log n + mnk) 时间内可验证，并且满足 (γ+2)-mPJR+ 的近似公平性；

**🔧 技术方法**

技术手段包括：从批准制多选投票问题的嵌入（approval-to-metric embedding）进行归约、子模函数最小化、几何距离分析、三角不等式推导、以及对每个未选中心以球形展开并维护最近距离的高效算法；

**📊 数据集**

实验使用合成的二维欧氏空间数据：随机生成五个高斯簇，令候选中心等于所有点，聚类数 k=5，随机采样 1,000 个委员会；

**📈 对比分析**

对比方法主要是通过随机委员会的满意率来评估 DC-mPJR+ 与 mPJR+ 的严格程度；实验显示 DC-mPJR+ 的满足率略高但差距不大，证明默认联盟保持了有意义的公平判定；算法复杂度已给出为 O(mn log n + mnk)，在理论上远优于 mPJR+ 的子模最小化实现；

**⚠️ 局限性**

局限性包括：实验仅在合成数据上进行，未在真实世界数据集上验证；默认联盟的约束可能在某些场景下过于严格或过于宽松；固定层次下的 mPJR+ 验证仍是 coNP‑hard；并未给出实际运行时间或大规模实验结果。

---

## 770. Autoregressive Learning in Joint KL: Sharp Oracle Bounds and Lower Bounds

**arXiv ID:** 2605.12316 | [PDF](https://arxiv.org/pdf/2605.12316v1)

**作者:** Yunbei Xu `[一作]` (National University of Singapore), Ruohan Zhan `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2`

**🎯 论文内容**

本文研究了在自回归模型（autoregressive modeling）中，针对联合KL（joint Kullback–Leibler）目标下的模型缺失（misspecification）情形，给出了逼近误差和估计误差的完整 H（序列长度）尺度表征；

**💡 创新点**

创新点在于首次证明联合KL在逼近误差上保持常数（无 H 影响）并且给出估计误差的最优 Ω(H) 下界，从而区分了平方 Hellinger 产生的误差放大现象是度量导致的；

**🔧 技术方法**

主要技术包括联合KL 的链式分解、前缀条件化的经验风险最小化与贝叶斯后验学习的 Oracle 证明、以及基于 Fano 的信息理论下界构造；

**📊 数据集**

未在论文中给出具体数据集，实验部分仅为理论分析；

**📈 对比分析**

比较方法主要是与平方 Hellinger 及 TV 评估指标对比，理论上联合KL 在逼近误差上无 H 放大，估计误差与 H 成线性比例，匹配了现有最优结果；

**⚠️ 局限性**

限制在于估计误差随 H 线性增长，即在长序列下需要更多样本；此外，论文未考虑大规模 Transformer 结构的实现细节和实际数据验证。

---

## 771. G$^2$TR: Generation-Guided Visual Token Reduction for Separate-Encoder Unified Multimodal Models

**arXiv ID:** 2605.12309 | [PDF](https://arxiv.org/pdf/2605.12309v1)

**作者:** Junxian Li `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种生成引导的视觉令牌压缩框架（G^2TR），在保持统一多模态模型（UMMs）图像理解与编辑能力的前提下，对理解侧视觉令牌进行训练无关、可插拔的压缩。

**💡 创新点**

创新点：①利用生成分支（VAE潜在向量）作为视觉令牌重要性引导，兼顾理解与生成信息；②在每个潜在锚点内进行平衡挑选，避免令牌集中；③对冗余令牌进行余弦最近邻合并，降低信息损失；④整个过程无需额外训练，直接插入现有Umm推理流程。

**🔧 技术方法**

技术细节：视觉令牌重要性通过与对应VAE锚点的余弦相似度估计；在保持令牌预算的前提下进行平衡选取；对被删去的令牌按最近邻归并到保留令牌；在ViT编码后、LLM预填充前完成压缩，保持Flash-Attention不变。

**📊 数据集**

数据集与模型：在BAGEL-7B-MoT和InternVL-U两种分离编码Umm上实验，使用图像理解基准（MME、MMBench、MMVP、RealWorldQA）和图像编辑基准（GEdit-Bench、IntelligentBench、RISE）。

**📈 对比分析**

与多种基线（FastV、W-FastV、PDrop、VSCAN、IVC-prune）在相同50%令牌预算下对比：G^2TR在理解任务上相对平均分数最高（BAGEL-7B 99.0%，InternVL-U 94.9%）；编辑任务保持与原模型相近的质量；效率提升显著，预填充FLOPs 1.94×、KV缓存 1.90×、解码延迟略降。

**⚠️ 局限性**

局限性：仅针对分离编码Umm设计，无法直接迁移到联合编码或未来更自适应的多模态架构；使用固定的令牌压缩比例，缺乏动态调节机制；需进一步探索在多任务和实时场景中的适配。

---

## 772. EgoEV-HandPose: Egocentric 3D Hand Pose Estimation and Gesture Recognition with Stereo Event Cameras

**arXiv ID:** 2605.12297 | [PDF](https://arxiv.org/pdf/2605.12297v1)

**作者:** Luming Wang `[一作]` (Zhejiang University), Kaiwei Wang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向视角摄像头的立体事件相机的全流程3D手势识别与姿态估计框架 EgoEV-HandPose。

**💡 创新点**

创新点在于引入了 KeypointBEV 模块，将立体特征投射到鸟瞰视角进行迭代重投影纠正，消除单目深度不确定性，并结合事件相机的高动态范围与低功耗优势。

**🔧 技术方法**

使用了局部归一化事件表面（LNES）+ EgoBlaze 特征提取、KeypointBEV 迭代 BEV 重投影、时序 Transformer 识别等技术。

**📊 数据集**

数据集为 EgoEVHands，包含 5,419 条双目事件序列、38 类手势、完整 2D/3D 键点与分割标注。

**📈 对比分析**

与现有 RGB 立体和单目事件方法对比，EgoEV-HandPose 在 EgoEVHands 上实现 MPJPE 30.54 mm、Top‑1 手势精度 86.87%，比 HandMvNet、Ev2Hands、EvHandPose 等方法分别降低 57%–73% 的误差，并保持 8.44 M 参数、19.86 GFLOPs 的轻量级。

**⚠️ 局限性**

局限在于计算量仍高，尤其是双目双手结构在单手场景下略显冗余，且对极端快速运动和复杂遮挡的鲁棒性待进一步提升，可通过模型压缩或跟踪策略改进。

---

## 773. Large-Small Model Collaboration for Farmland Semantic Change Detection

**arXiv ID:** 2605.12282 | [PDF](https://arxiv.org/pdf/2605.12282v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 774. Context Convergence Improves Answering Inferential Questions

**arXiv ID:** 2605.12370 | [PDF](https://arxiv.org/pdf/2605.12370v1)

**作者:** Jamshid Mozafari `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了句子收敛度对推理性问答中大模型表现的影响，构造基于收敛度的句子集作为段落。

**💡 创新点**

提出收敛度作为衡量句子推理价值的新指标，并证明其优于余弦相似度进行句子筛选。

**🔧 技术方法**

使用LLM生成候选答案、判断句子与答案的匹配、计算收敛度，并采用多种LLM（LLaMA、Gemma、Qwen）进行评估。

**📊 数据集**

使用TriviaHG数据集，并通过HintEval工具计算收敛度。

**📈 对比分析**

对比高收敛度与低收敛度句子集合以及余弦相似度选取，采用ExactMatch、Precision、Recall、F1四指标，发现高收敛度段落在各模型上均显著提升EM分数，平均提升约10%-15%。

**⚠️ 局限性**

受限于仅使用小规模LLM且数据集限制为单一推理问答集，未验证更大模型或更复杂检索方式的普适性。

---

## 775. MetaColloc: Optimization-Free PDE Solving via Meta-Learned Basis Functions

**arXiv ID:** 2605.12368 | [PDF](https://arxiv.org/pdf/2605.12368v1)

**作者:** Zichuan Yang `[一作]` `[通讯]` (Tongji University), Zichuan Yang (Tongji University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `01e19694-9125-4cf8-82ff-580f56a0fdb6`

**🎯 论文内容**

提出 MetaColloc 框架，将基底学习与 PDE 求解分离，离线 meta‑学习得到通用神经基底，测试时仅通过一次线性最小二乘或 Newton‑Raphson 迭代即可求解新 PDE。

**💡 创新点**

创新点在于：① 通过双分支网络（低频 MLP + 多尺度 Fourier 特征） meta‑学习一个全局的、可冻结的神经基底字典；② 完全消除测试时梯度优化，转为单步线性求解；③ 只需基于随机生成的 Gaussian Random Fields 训练，无需 PDE 训练数据。

**🔧 技术方法**

技术手段包括：双分支 MLP + Fourier embedding、SwiGLU 激活、meta‑训练（最小二乘目标）、前向模式自动微分求导、构造耦合矩阵、线性 least‑squares 求解、Newton‑Raphson 迭代处理非线性 PDE。

**📊 数据集**

使用自生成的多尺度 Gaussian Random Fields 作为训练任务；在测试阶段评估六个 2D/3D PDE（Poisson、Helmholtz、VarCoeff、HighFreq、SineGordon、KdV）以及几何复杂域（L‑shape、Annulus）。

**📈 对比分析**

与 PINN（L‑BFGS）、GP‑HM、ConFIG、DCGD 等无数据、无优化实例求解器对比，MetaColloc 在绝大多数任务上实现 RMSE 低至 1e‑9，速度提升数倍至数百倍，尤其在线性 PDE 上仅需约 1.3 秒即可完成，而其它方法需数秒或更久。

**⚠️ 局限性**

局限性：对极高频 Helmholtz 之类的高频问题精度仍不如 GP‑HM，原因是基底在函数空间拟合良好但在微分算子（如 Laplacian）上不稳定，导致算子–函数不匹配；需要进一步的 operator‑aware meta‑训练。

---

## 776. Reinforcing VLAs in Task-Agnostic World Models

**arXiv ID:** 2605.12334 | [PDF](https://arxiv.org/pdf/2605.12334v1)

**作者:** Yucen Wang `[一作]` (Nanjing University), Li Zhao `[通讯]` (Microsoft Research Asia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出RAW-Dream框架，在已预训练的任务无关世界模型与零样本视觉语言模型奖励的基础上完成VLA的后训练；

**💡 创新点**

创新点在于彻底解耦世界模型与奖励函数，使其对目标任务无关，并通过双噪声验证过滤世界模型幻觉；

**🔧 技术方法**

使用WAN 2.1 Diffusion Transformer做动作条件世界模型，OpenVLA-OFT做策略，Qwen3‑VL做零样本奖励，GRPO做强化学习；

**📊 数据集**

训练数据来源于LIBERO-90（90个任务约4.5k演示）以及无标签的play数据，评估时使用LIBERO的四个未见任务组和AgileX Piper机器人真实数据；

**📈 对比分析**

与SFT、在线RL、从零开始训练的WM等方法对比，RAW‑Dream在所有任务上均超越SFT和在线RL，且在WM训练仅使用少量或零目标任务数据时仍能实现显著提升；

**⚠️ 局限性**

局限在于世界模型在完全陌生物体或布局时的模拟质量下降，零样本奖励的可靠性受限于VLM的通用性，并且双噪声验证对长序列的依赖较高。

---

## 777. A categorical error sensitivity index (ISEC): A preventive ordinal decision-support measure for irrecoverable errors in manual data entry systems

**arXiv ID:** 2605.12328 | [PDF](https://arxiv.org/pdf/2605.12328v1)

**作者:** Ricardo Raúl Palma `[一作]` (Universidad Nacional de Cuyo), Fabricio Orlando Sanchez Varretti `[通讯]` (Universidad Tecnologica Nacional)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了类别错误敏感度指数ISEC，用于预先评估分类系统的脆弱性

**💡 创新点**

将语义距离、定制化形态学变换成本和频率结合成单一序数指标，并通过向量数据库加速计算

**🔧 技术方法**

使用词向量语义相似度、加权Damerau–Levenshtein编辑距离、向量数据库（ChromaDB）以及自定义字符成本矩阵

**📊 数据集**

在司法行政记录、零售库存和合成ISO码三组数据集上验证

**📈 对比分析**

采用混合搜索策略（语义Top-K + 形态学计算），相比暴力O(N²)提升约195×，在1,000个类别仅耗2分钟；在1,069,280条司法记录仅34秒

**⚠️ 局限性**

依赖人工设定成本矩阵，未自动学习误差模式；仅适用于手工输入的离散分类；无法处理自动生成或连续特征

---

## 778. CAAFC: Chronological Actionable Automated Fact-Checker for misinformation / non-factual hallucination detection and correction

**arXiv ID:** 2605.12436 | [PDF](https://arxiv.org/pdf/2605.12436v1)

**作者:** Islam Eldifrawi `[一作]`, Amine Trabelsi `[通讯]` (University of Sherbrooke)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了CAAFC框架，将自动事实核查拆分为若干子任务，并通过时间序列化的原始来源检索，支持事实检测与纠错。

**💡 创新点**

创新点包括：引入可操作性解释模块、使用时间顺序化原始证据检索、将事实核查与幻觉检测统一为一体，以及利用量化小型LLM实现高效推理。

**🔧 技术方法**

技术上采用量化Gemma3-27B等LLM，结合Google AI Mode搜索、事实检查器、可操作性评估器、解释重写器等模块；同时利用FinGrAct评估可操作性。

**📊 数据集**

使用了AVeriTeC、CoverBench、FACTors、DiaHalu等四个基准数据集，并在检索无噪声和数据清洗后进行评估。

**📈 对比分析**

与SOTA模型（GPT‑4/5.2、Claude 3.5、LLama3.3‑70B等）在准确率和宏F1上对比，CAAFC在多数据集上均取得或超过SOTA，且小模型Gemma3-27B表现接近大模型，仅消耗17 GB显存。

**⚠️ 局限性**

局限性包括仅支持英文、可能受到搜索引擎偏见导致证据偏颇、推理多次LLM调用导致延迟，以及需进一步验证多语言和知识库自动更新的安全性。

---

## 779. Beyond Localization: A Comprehensive Diagnosis of Perspective-Conditioned Spatial Reasoning in MLLMs from Omnidirectional Images

**arXiv ID:** 2605.12413 | [PDF](https://arxiv.org/pdf/2605.12413v1)

**作者:** Yuangong Chen `[一作]` (Hong Kong Polytechnic University), Xu Zheng `[通讯]` (Queen Mary University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PCSR‑Bench——一个基于360°全景图的视角条件空间推理基准，评估并诊断多模大语言模型在不同视角下的空间推理能力，并通过强化学习干预探索其可塑性。

**💡 创新点**

创新点包括：①将全景图与3D结构结合生成几何驱动的QA；②设计从基础感知到高级视角条件推理的八类任务；③揭示视角条件推理与基础感知之间显著的性能差距；④通过RL调优验证部分可塑性。

**🔧 技术方法**

采用多模大语言模型Zero‑shot评估、语义对齐评分器、基于GRPO的强化学习优化、3D几何约束生成QA、结构化思考‑答案模板等技术。

**📊 数据集**

使用ReplicaPano 360°全景图数据集，构建2,600张全景图、84,373问答对，覆盖26个室内场景，并在图像层面划分训练/测试集。

**📈 对比分析**

在14款MLLM（按参数层级分组）上进行比较，基础感知任务平均得分约57%，高级视角任务平均仅约29%；通过RL优化后，部分任务可提升至≈60%，但整体提升有限且任务差异显著。

**⚠️ 局限性**

局限性包括：结果高度依赖评估协议与解析规则；RL改进受奖励设计敏感；基准仅覆盖室内全景，未考虑外景或不同投影；实验主要基于单一基础模型验证。

---

## 780. Detecting overfitting in Neural Networks during long-horizon grokking using Random Matrix Theory

**arXiv ID:** 2605.12394 | [PDF](https://arxiv.org/pdf/2605.12394v1)

**作者:** Hari K. Prakash `[一作]` (University of California San Diego), Charles H Martin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种仅基于权重的随机矩阵诊断方法，用于检测深度学习模型的过拟合（尤其是长期训练后出现的 anti-grokking 阶段），并给出了检测“Correlation Traps”的流程；

**💡 创新点**

创新点在于将随机矩阵理论与权重逐元素打乱相结合，利用马尔科夫-帕斯（Marchenko–Pastur）分布的右边缘检测结构化谱异常（Correlation Traps），并证明其与模型在 anti-grokking 阶段的泛化下降相关；

**🔧 技术方法**

主要技术包括：层级权重逐元素随机化、构建协方差矩阵、拟合 Marchenko–Pastur 分布、统计超出 Tracy‑Widom 边缘的谱峰、以及基于 Jensen‑Shannon 散度的无数据“Trap Ablation”检测；

**📊 数据集**

使用的公开数据集包括：MNIST（1k 样本的子集）、Modular Addition 任务、以及 GPT‑style 变换器上的合成知识图谱（两跳推理），并对 OpenAI 的 GPT‑OSS 20B/120B 公开权重进行了基线扫描；

**📈 对比分析**

与传统的 train/test 曲线或权重范数等指标相比，Correlation Traps 能够在 pre‑grokking 与 anti‑grokking 之间清晰区分；在三种基准实验中，Trap 数量在 anti‑grokking 阶段显著上升，与测试准确率的下降高度相关；实验显示加入权重衰减可抑制 Trap 的生成和泛化衰退；

**⚠️ 局限性**

局限性包括：实验仅覆盖三种长期 grokking 场景（MLP、Transformer+Modular Addition、GPT‑style），未覆盖更广的架构、优化器或数据；对大模型的屏蔽结果仅为表面现象，未直接验证是否导致有害过拟合；且 Trap 识别和区分为模型内部的统计诊断，需配合任务级评估才能得到完整评价。

---

## 781. SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation

**arXiv ID:** 2605.12389 | [PDF](https://arxiv.org/pdf/2605.12389v1)

**作者:** Luke James Miller `[一作]` (University of Missouri-Kansas City), Yugyung Lee `[通讯]` (University of Missouri-Kansas City)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在医学图像中，作者提出SEMIR框架，通过在体素网格上学习图小子（graph minor）实现稀疏、边界对齐的分割表示，且能精确恢复到原始体素级别；

**💡 创新点**

创新点包括：①利用图小子理论以可学习方式压缩体素网格并保留拓扑；②用少样本黑盒优化对边界进行对齐，取代手工超参数；③构造可精确解码的映射，避免插值导致边界模糊；

**🔧 技术方法**

技术手段包括：图小子构造（边缘收缩、节点/边删除）、少样本优化（基于边界Dice的SMBO）、图神经网络（GINE）进行超节点分类、精确提升映射；

**📊 数据集**

在三个肿瘤分割基准上验证：BraTS 2021（脑瘤）、KiTS23（肾瘤/囊肿）、LiTS（肝脏肿瘤）；

**📈 对比分析**

与多种现有CNN/Transformer/Transformer-UNet等方法对比，SEMIR在少数类Dice上（如ET、TC、T）取得领先或同等水平，且训练/推理时间显著降低（约1–3%体素数），实验展示在少数类指标上提升了2–4% Dice；

**⚠️ 局限性**

局限性：依赖少样本优化的边界统计，若样本不具代表性可能导致边界误匹配；图小子构造与下游预测解耦，未实现端到端联合优化；对低对比度或极稀疏结构的鲁棒性仍待进一步验证。

---

## 782. A Semi-Supervised Framework for Speech Confidence Detection using Whisper

**arXiv ID:** 2605.12387 | [PDF](https://arxiv.org/pdf/2605.12387v1)

**作者:** Adam Wynn `[一作]` (Durham University), Jingyun Wang `[通讯]` (Durham University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个半监督混合框架，用于自动检测说话人自信。该框架将 Whisper 语义嵌入与 eGeMAPS 声学特征以及辅助的失语与语调特征进行融合，并通过不确定性感知的伪标签策略扩大训练数据。

**💡 创新点**

创新点包括：①在 Whisper 预训练模型与可解释声学特征之间采用晚期融合，提升对微妙自信信号的捕获；②设计了基于模型置信度阈值的伪标签筛选方法，强调数据质量而非数量；③在标注稀缺的情感自信任务上通过半监督学习显著提升性能。

**🔧 技术方法**

主要技术：Whisper-base encoder、eGeMAPS 声学特征提取、辅助失语/语调检测模型、MLP 伪标签生成、Late Fusion、Source-Boosted Loss、温度标定、SHAP 解释、t‑SNE 可视化。

**📊 数据集**

使用的数据集包括：自建的 600 条标注自信的 TED‑LIUM、CMU‑MOSI、MLCommons People’s Speech、SEP‑28K；扩充用的未标注语料；辅助失语检测采用 SEP‑28K‑E‑Merged；情绪/压力检测采用 RAVDESS、SAVEE、TESS。

**📈 对比分析**

通过 5‑折交叉验证与多种基线（Whisper‑Only、Feature‑Only、Wav2Vec‑2.0、HuBERT、WavLM、Whisper‑Tiny 等）比较，最终宏观 F1 为 0.751，显著高于 Whisper‑Only（0.736）和纯声学基线，并在低/中自信类别上提升尤为显著。

**⚠️ 局限性**

局限性：标注样本仅 600 条，局限于英语且为短片段；缺乏多语言、多文化背景以及视觉模态，模型可能对不同人群或上下文产生偏差；未考虑自信随时间变化的动态特征。

---

## 783. ORCE: Order-Aware Alignment of Verbalized Confidence in Large Language Models

**arXiv ID:** 2605.12446 | [PDF](https://arxiv.org/pdf/2605.12446v1)

**作者:** Chen Li `[一作]` (Stony Brook University), Chao Chen `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将答案生成与置信度估计解耦，并利用基于Spearman秩相关的强化学习对模型产生的自然语言置信度进行全局排序优化，提升了LLM在高风险场景下的置信度校准与错误预测能力。

**💡 创新点**

①将答案生成与置信度生成分离，避免联合优化导致的答案误差；②用多样化采样得到的“正确率近似”作为无标签监督；③使用全局秩相关（Spearman）奖励替代点对点匹配，提升置信度的相对排序。

**🔧 技术方法**

答案生成模型、置信度生成模型、采样估计、Spearman秩相关奖励、Direct Preference Optimization (DPO)、SFT预热、token-free自然语言置信度生成。

**📊 数据集**

在知识推理与离散/逻辑推理数据集上进行评测：MMLU、DROP、LogiQA 2.0 与 ReClor，使用 LLama‑3 8B、Qwen3‑8B、Mistral‑7B 三大开源基础模型。

**📈 对比分析**

与 Vanilla、Self‑Consistency、Top‑k、CoT、ConfTuner、SFT 等基线比较，ORCE 在 ECE、Spearman 相关、AURC、EAURC 等指标上均显著优于或相当于最佳基线，同时保持答案准确率不变，证明其在校准与失败预测上的优势。

**⚠️ 局限性**

仍需多次采样导致推理延迟；在分布极端或数据稀缺情况下，秩相关奖励可能不稳定；对低质量答案的精细排序提升有限。

---

## 784. Search Your Block Floating Point Scales!

**arXiv ID:** 2605.12464 | [PDF](https://arxiv.org/pdf/2605.12464v1)

**作者:** Tanmaey Gupta `[一作]` (Cornell University), Chris De Sa `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了针对 NVFP4 微块浮点量化的尺度搜索（ScaleSearch）算法，并将其应用于后训练量化（PTQ）和注意力计算，进一步设计了基于 NVFP4 的 KV 缓存压缩与混合精度注意力方案。

**💡 创新点**

创新点在于：①利用 NVFP4 量化尺度中的尾数位进行细粒度搜索，显著降低块级均方误差；②将尺度搜索集成到 PTQ 与注意力流水线中；③结合不一致性处理（IP）和混合精度 KV 缓存，构建端到端无 dequantization 的 FP4 注意力实现。

**🔧 技术方法**

核心技术包括：NVFP4 微块浮点量化、尺度搜索（ScaleSearch）、不一致性处理与幅度缩减、混合精度 KV 缓存、基于 Tensor Core 的 FP4 矩阵乘法、FlashAttention 等低精度注意力实现。

**📊 数据集**

使用的主要数据集和模型：Llama 3.1 8B/70B、Qwen3 4B/8B、DeepSeek‑R1‑Distill‑Qwen‑1.5B、Qwen3‑8B、Mochi、CogVideoX；评测指标包括 GPQA、MATH‑500、AIME‑120、MMLU、Wikitext‑2 perplexity、CLIPSIM、CLIP‑T、VQA‑a/t、FScore 以及文本到视频生成的终端延迟。

**📈 对比分析**

与原始 FP32、Naive‑FP4、SageAttention3 以及其他低精度实现比较，ScaleSearch 在 PTQ 中使 GPQA/MATH‑500 等指标提升最多 15 分，Wikitext‑2 perplexity 下降 22%（Llama 3.1 70B），KV 缓存压缩后 KV 内存显著降低；注意力吞吐量几乎与基线持平（≥97%），量化时的额外开销仅 1.27–1.74×。

**⚠️ 局限性**

局限性包括：①仅针对 NVFP4（和 MXFP4 等）微块格式；②尺度搜索在块大小增大时收益递减；③实现依赖 NVIDIA Blackwell GPU 的 Tensor Core，跨平台可移植性有限；④部分指标（如 AIME‑120）提升有限；⑤在极低精度（FP4）下仍需额外技术（IP、混合精度缓存）才能保持准确性。

---

## 785. Towards Affordable Energy: A Gymnasium Environment for Electric Utility Demand-Response Programs

**arXiv ID:** 2605.12462 | [PDF](https://arxiv.org/pdf/2605.12462v1)

**作者:** Jose E. Aguilar Escamilla `[一作]` (Oregon State University), Huazheng Wang `[通讯]` (Oregon State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 DR‑Gym，一个用于电力公用事业层面需求响应决策的 Gymnasium 兼容仿真环境，集成了基于物理的建筑负荷、极端事件驱动的马尔可夫价格波动、异质消费者疲劳模型以及可配置的多目标奖励。

**💡 创新点**

创新点包括：①首个公开的公用事业级需求响应 RL 环境；②结合实测的价格极端事件（如 ERCOT 冬季风暴）与 AR(1)+马尔可夫状态切换的价格模型；③支持可插拔的风险感知奖励与消费者保护指标；④通过统一接口让研究者可轻松替换价格、负荷或行为子模型。

**🔧 技术方法**

技术手段包括：强化学习（PPO 基础实验），多目标奖励函数、Conditional Value‑at‑Risk 作为风险度量，马尔可夫状态切换的价格生成，基于 CityLearn/ResStock 的 EnergyPlus 负荷序列，以及对消费者接受率的逻辑回归与疲劳衰减建模。

**📊 数据集**

使用的数据集为 CityLearn 及其所基于的 NREL ResStock 物理仿真负荷数据；价格模型则通过对 ERCOT 日前市场统计的校准产生；其他参数如预算、价格弹性、行为架构均按文献给出的范围设定。

**📈 对比分析**

在实验中将 PPO 与四种基线（无信用、均匀信用、价格阈值规则、预算感知规则）对比，结果显示 PPO 在平均奖励、预算利用率（约 72‑85%）以及消费者账单 CVaR（相较于无信用策略降低 18‑24%）方面均显著优于基线，验证了环境的可学习性与挑战性。

**⚠️ 局限性**

局限性主要在于：①关键参数（如需求衰减 γ、价格弹性 λ）仅采用文献默认值，缺乏针对特定电网或 DR 方案的经验校准；②实验仅使用标准 PPO，未深入探讨风险感知 RL 算法；③消费者模型采用四类经验 archetype，未基于真实 DR 数据（如 Pecan Street）进行精细化拟合；④价格模型为合成，未来需与实时市场接口结合。

---

## 786. TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection

**arXiv ID:** 2605.12456 | [PDF](https://arxiv.org/pdf/2605.12456v1)

**作者:** Tom Sander `[一作]` (FAIR, Meta Superintelligence Labs), Pierre Fernandez `[通讯]` (FAIR, Meta Superintelligence Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种名为TextSeal的无失真LLM水印方法，能够在不降低文本质量的前提下，在生成文本中嵌入可检出的水印信号。

**💡 创新点**

创新点包括：①双密钥路由（dual‑key generation）恢复生成多样性；②熵加权检测（entropy‑weighted detection）提升在低熵或高熵场景下的统计显著性；③几何覆盖搜索与多区域集成实现对稀释和碎片化文本的定位检测；④兼容MTP/推测解码并支持模型蒸馏后的“放射性”检测。

**🔧 技术方法**

技术手段涵盖：Gumbel‑max 水印框架、伪随机函数(PRF)、双密钥随机选择、熵估计与加权、Gamma矩匹配近似、几何覆盖搜索、早期融合统计、MTP/多令牌预测适配、蒸馏后检测的教师强制和去重。

**📊 数据集**

实验数据集：Qwen 3.5‑27B 生成 ELI5、CaLMQA 多语种 QA、GSM8K、ARC、MBPP、SQA、HumanEval、OpenR1‑Math、OpenAI GPT‑OSS‑20B、多语言 QA、Wiki 文本用于误报率评估。

**📈 对比分析**

比较方法：与基线 Gumbel‑max、SynthID‑Text 在多样性‑检测性 Pareto 前沿、稀释/碎片化文档检测、MTP 兼容性和蒸馏转移性进行对比。结果显示：TextSeal 在多样性保持较低的 Self‑BLEU 下，检测显著率提升 1–2 个数量级；在 12 项基准上模型性能保持无显著下降；在稀释至 3% 及碎片化至 3 段时，局部检测保持 -log10 p > 4，远优于全局基线。

**⚠️ 局限性**

局限性：无失真设计会牺牲一定的生成多样性；在极高熵或低熵极端场景下检测效果仍受限；适用性主要针对单次生成任务，对需要最佳多样性的多轮或“best‑of‑N”工作流影响尚未系统评估；对不同模型家族的泛化仍需进一步验证。

---

## 787. A Causal Language Modeling Detour Improves Encoder Continued Pretraining

**arXiv ID:** 2605.12438 | [PDF](https://arxiv.org/pdf/2605.12438v1)

**作者:** Rian Touchent `[一作]` (Sorbonne University), Eric de la Clergerie `[通讯]` (INRIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对医学领域的编码器进行域自适应继续预训练，采用先 CLM 再短 MLM 的“CLM detour”策略。

**💡 创新点**

证明短暂切换为 CLM 并在返回 MLM 时保留低层表示改进，且低层改动是提升的关键，提供 CKA + freeze 实验支持。

**🔧 技术方法**

使用 ModernBERT / ModernCamemBERT 架构、FlashAttention、Rotary、8,192 token 上下文；CLM-MLM 两阶段训练；CKA、冻结层干预、线性探测等技术。

**📊 数据集**

使用 10B–50B token 的英法医学文本，来源包括 PubMed、BioEnriched、MIMIC、法国医学文献、指南、药品说明书等。

**📈 对比分析**

在 8 个法语和 11 个英语生物医学基准上，以宏 F1 评价；CLM detour 在 Base 10B 与 Large 50B 上平均提升 0.3–2.8 pp，法语任务中最高 +2.8 pp。

**⚠️ 局限性**

主要局限在目标域与预训练域差距，已接触医学文本的模型提升有限；跨领域迁移需进一步验证，低资源语言的充分测试尚缺。

---

## 788. Binary constraints on one additional variable can create exponential ascents

**arXiv ID:** 2605.12425 | [PDF](https://arxiv.org/pdf/2605.12425v1)

**作者:** David A. Cohen `[一作]` (Royal Holloway University of London), Sofia Vazquez Alferez `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构造了一个基于星形结构的二元布尔 VCSP，实现指数级的上升路径；

**💡 创新点**

用星形 gadget 取代传统链式 gadget，显著降低了树深和反馈顶点集数，证明了在 treedepth = 3、fvs = 1 的情况下仍能得到指数上升；

**🔧 技术方法**

利用 VCSP 的可视化为 fitness landscape、诱导子问题递归展开、树深分析以及约束图的最小化技术；

**📊 数据集**

无实际数据集，全部为理论构造与证明；

**📈 对比分析**

与之前链式 gadget 的对比表明上升长度指数增长且结构更简单，参数化复杂度更低；

**⚠️ 局限性**

尚未实现指数 steepest ascent 或所有初始点的指数上升；更高 treedepth 可能是必要条件，实际应用的可扩展性和实用性仍待验证。

---

## 789. Formalize, Don't Optimize: The Heuristic Trap in LLM-Generated Combinatorial Solvers

**arXiv ID:** 2605.12421 | [PDF](https://arxiv.org/pdf/2605.12421v1)

**作者:** Haoyu Wang `[一作]` (University of Pennsylvania), Dan Roth `[通讯]` (University of Pennsylvania)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在CP‑SynC‑XL基准上评估LLM生成组合优化求解器的三种实现范式（纯Python、Python+OR‑Tools、MiniZinc+OR‑Tools），并探讨提示是否能提升搜索效率。

**💡 创新点**

提出CP‑SynC‑XL大规模基准，发现“表征偏差”与“启发式陷阱”，证明LLM在表达式层面的熟练度决定解答正确率，且提示优化往往会引入错误。

**🔧 技术方法**

使用三大前沿LLM（GPT‑5.3‑Codex、Gemini‑3.1‑Pro、DeepSeek‑V3.2）结合代码生成、迭代校正、验证器对比、六种失效模式审计。

**📊 数据集**

CP‑SynC‑XL共100个约束问题（58 CSP，42 COP）共4,577实例，来源于CSPLib、MiniZinc挑战、XCSP3等。

**📈 对比分析**

通过提供/正确率、精度比例、累积运行曲线和五分解统计进行比较，Python+OR‑Tools在正确率最高，MiniZinc+OR‑Tools在条件精度最高；启发式提示仅带来1.03–1.12×的平均加速，但在弱模型上显著降低正确率。

**⚠️ 局限性**

实验受限于所选LLM、三种实现范式及单次生成的精细度，且未探索多样化搜索策略或自适应提示，未来需提升对表征偏差与启发式陷阱的鲁棒性。

---

## 790. ORBIT: Preserving Foundational Language Capabilities in GenRetrieval via Origin-Regulated Merging

**arXiv ID:** 2605.12419 | [PDF](https://arxiv.org/pdf/2605.12419v1)

**作者:** Neha Verma `[一作]` (Johns Hopkins University), Xinyang Yi `[通讯]` (Google DeepMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在对大型语言模型进行 GenRetrieval 任务微调时，如何避免灾难性遗忘。

**💡 创新点**

提出了一种“Origin‑Regulated Back‑merging of Intermediate Trajectories”（ORBIT）方法，即根据模型间参数距离动态触发权重平均，以限制模型漂移。

**🔧 技术方法**

使用模型融合技术、Sign Dissimilarity（SD）作为距离度量、在线权重平均以及与传统的 L2 权重衰减、Soup‑to‑Go 等方法对比。

**📊 数据集**

采用 Amazon Product Reviews（Beauty、Sports & Outdoors、Toys & Games）数据集以及 8 个文本推理基准（BBH、GSM8K、MMLU‑Pro 等）。

**📈 对比分析**

与无干预、L2 权重衰减、Soup‑to‑Go 等基线比较，ORBIT 在文本推理和召回率上均获得或超过基线，达成 Pareto 最优并在 DTIP 指标上表现更好。

**⚠️ 局限性**

方法在极端多任务或长时间微调场景下的适用性仍需验证，且对更大规模模型的进一步扩展与稳健性尚未完全探索。

---

## 791. Aligning Flow Map Policies with Optimal Q-Guidance

**arXiv ID:** 2605.12416 | [PDF](https://arxiv.org/pdf/2605.12416v1)

**作者:** Christos Ziakas `[一作]` (Imperial College London), Avishek Joey Bose `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一类流映射策略并给出闭式 Q 引导的置信域更新及推理时的随机重噪搜索，提升离线到在线强化学习的性能。

**💡 创新点**

创新点在于：①将流匹配模型转化为一次跳跃的流映射；②利用 Q 梯度在置信域内求解闭式最优步长；③结合重噪与 beam search 的推理时间改进。

**🔧 技术方法**

使用流匹配、流映射（flow‑map）网络、线性化 Q 近似、置信域自适应、重噪采样和 beam search 等技术。

**📊 数据集**

在 12 个 OGBench 与 RoboMimic 机器人连续控制任务（操控与步态）上进行实验。

**📈 对比分析**

与 MVP、QC 等基线相比，IQM 成功率提升约 21.3%（相对 0.93 vs 0.75），样本效率更高，推理时间仅需单步生成。

**⚠️ 局限性**

局限性：依赖 Q 函数的准确性与线性近似；对高度非线性奖励场景可能效果不佳；缺乏对真实机器人平台的验证。

---

## 792. Semantic Reward Collapse and the Preservation of Epistemic Integrity in Adaptive AI Systems

**arXiv ID:** 2605.12406 | [PDF](https://arxiv.org/pdf/2605.12406v1)

**作者:** William Parris `[一作]` `[通讯]` (BDB Labs / BagelTech), William Parris (BDB Labs / BagelTech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文分析了奖励信号的语义坍塌现象，并提出了宪法奖励分层（CRS）框架以在训练中保持不确定性表达的可辨别性。

**💡 创新点**

创新点在于将奖励拆分为多维独立通道，特别为不确定性声明设置保护通道，从而减少对透明 epistemic 反馈的抑制。

**🔧 技术方法**

技术主要包括奖励信号的多维分层表示、对比传统标量化奖励、以及与多目标强化学习框架的理论对照。

**📊 数据集**

本研究未使用具体数据集，属于理论与框架性探讨。

**📈 对比分析**

缺乏实验对比与性能评估，本文仅提出可检验的假设（如不确定性校准提升、表现性自信减少等）。

**⚠️ 局限性**

局限性包括未对模型进行实证验证、无法证明所有幻觉均源于奖励坍塌、对高风险领域不确定性奖励的具体实现仍待研究。

---

## 793. Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring

**arXiv ID:** 2605.12398 | [PDF](https://arxiv.org/pdf/2605.12398v1)

**作者:** Jamshid Mozafari `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于答案可信度分数熵的问答难度估计方法Q-DAPS；

**💡 创新点**

利用LLM生成候选答案并计算其可信度分数的熵来衡量问题难度，同时通过基于维基百科页面浏览量的流行度去偏差；

**🔧 技术方法**

候选答案生成与可信度评分使用大语言模型（如LLaMA 3.1/3.3或Qwen 2.5），流行度去偏差利用页面浏览统计，熵计算与归一化实现难度得分；

**📊 数据集**

在四个主流QA数据集上评估：TriviaQA、Natural Questions（NQ）、MuSiQue 和 QASC；

**📈 对比分析**

与可读性、流行度、检索、提示和不确定性等基线方法比较，Q-DAPS在区分易难问题（Cohen’s d 与 Spearman ρ 结果均最好）上均表现最优；

**⚠️ 局限性**

仅适用于可构造有限候选答案的问答类型，受限于英文数据，且依赖LLM的偏差与性能。

---

## 794. NCCLZ: Compression-Enabled GPU Collectives with Decoupled Quantization and Entropy Coding

**arXiv ID:** 2605.12396 | [PDF](https://arxiv.org/pdf/2605.12396v1)

**作者:** Jiamin Wang `[一作]` (Stevens Institute of Technology), Xiaodong Yu `[通讯]` (Stevens Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

开发了一套基于 NCCL 的压缩加速集体通信框架 NCCLZ，框架在接口层执行量化，随后在 NCCL 原语内部完成熵编码，并通过设备侧轻量级选择器实现动态码流选择与通信重叠。

**💡 创新点**

核心创新点包括：①将量化与熵编码解耦，分别放在 NCCL 体系结构的不同层级；②设计了设备端 Runtime Entropy Arbitration (REA)，在运行时依据消息大小、可压缩度和传输环境动态决定使用 Raw、Fixed‑Len 还是 GPU Huffman；③在 NCCL 的 NET 路径中嵌入编码/解码，利用 8‑slot 批处理和 ping‑pong 缓冲实现编码与网络传输的有效重叠，最大化吞吐量并避免错误累积。

**🔧 技术方法**

技术实现主要包括：GPU 端无损量化（QSGD 或基于误差边界的确定性量化）、GPU 固定长度打包与 GPU Huffman 熵编码、设备侧 REA 轻量级决策、在 NCCL Simple 协议的 8‑slot FIFO 之间插入 encode‑on‑send / decode‑on‑recv、使用双缓冲实现批级重叠，以及通过 header‑payload 结构实现自描述帧。

**📊 数据集**

使用的评测数据集：科学计算数据集 QMCPack (1.2 GiB) 与 CESM‑ATM (2.0 GiB)，深度学习梯度数据（ResNet‑18 / ResNet‑34 的 QSGD 量化梯度约 150 MiB），以及 nccl 测试的合成 FP32 负载。

**📈 对比分析**

通过与原生 NCCL、COCCL 以及 ghZCCL 进行对比实验。实验在 Polaris 超算上，跨 2、4、8 节点进行规模扩展。结果显示，NCCLZ 在所有工作负载上实现了 2.3–3.1× 的吞吐提升，最大 9.65× 的速度提升相较于未压缩 NCCL，并比现有压缩协同库提升 3.34×。熵编码进一步提升压缩比，GPU Huffman 在高压缩需求场景下优于 Fixed‑Len。

**⚠️ 局限性**

主要局限性：①框架针对带宽受限的 inter‑node 环境优化，intra‑node 高速传输（NVLink/PCIe）时熵编码收益有限；②熵编码的额外计算开销需要 REA 正确判断，否则可能导致性能下降；③目前仅实现了两种熵编码（Fixed‑Len 与 GPU Huffman），对极端分布或极大消息的自适应支持有限；④依赖 NVIDIA GPU 与 NCCL 2.28.3，迁移到其他硬件或 NCCL 版本需额外工作。

---

## 795. A Comparative Study of Controlled Text Generation Systems Using Level-Playing-Field Evaluation Principles

**arXiv ID:** 2605.12395 | [PDF](https://arxiv.org/pdf/2605.12395v1)

**作者:** Michela Lorandi `[一作]` (Dublin City University), Anya Belz `[通讯]` (Dublin City University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一套基于 Level‑Playing‑Field (LPF) 的评估框架，对 12 种主流受控文本生成（CTG）系统进行统一、可重复的评估。

**💡 创新点**

创新点在于：①将评估过程标准化（统一数据、统一生成与后处理流程）；②使用多分类器平均评估控制效果，消除单一分类器偏差；③在同一评估环境下比较不同技术，揭示原报告结果与真实性能的显著差距。

**🔧 技术方法**

技术细节包括：对 12 种 CTG 技术（完整训练、模型微调、词分布调节、提示式生成、混合式方法）进行实验；使用 DistilBERT、T5、DeBERTa 等多种文本分类器评估控制效果；计算 Distinct‑n、SLOR、Perplexity 等多维度指标；采用多随机种子（3 种）和多数据集生成文本。

**📊 数据集**

使用的数据集有四个：PPLM Prompts（35 条）、OpenWebText 中立提示（5000 条）、Cloze 2018 测试集（1571 条）和 STS 基准测试集（625 条）。

**📈 对比分析**

比较方法：统一生成与后处理流程，使用统一的评估指标和多分类器控制效果，按控制属性（情感、主题、关键词、多属性）和数据集进行系统性对比。结果显示：①专用 CTG 技术（微调、混合）在控制效果、流畅度、词汇多样性方面优于通用 LLM 提示式方法；②在 LPF 环境下，多数系统的控制效果低于原论文报告，强调标准化评估的重要性；③不同技术在推理速度与显存占用上差异显著，提示在资源受限场景需综合考量。

**⚠️ 局限性**

局限性：①仅评估情感、主题和关键词三种控制属性，未覆盖样式、知识、语法等其他属性；②控制属性取值有限，未测试更大范围；③仅挑选 ACL Anthology 的公开实现，排除其他会议或闭源方法；④仅使用自动指标，缺乏人工评估；⑤实验规模受限，未覆盖所有可能的评估组合。

---

## 796. Discrete Flow Matching for Offline-to-Online Reinforcement Learning

**arXiv ID:** 2605.12379 | [PDF](https://arxiv.org/pdf/2605.12379v1)

**作者:** Fairoz Nower Khan `[一作]` (University of Kentucky), Peizhong Ju `[通讯]` (University of Kentucky)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对离散动作空间的离线到在线强化学习微调框架DRIFT，利用连续时间马尔可夫链（CTMC）和流匹配实现生成策略的在线改进；

**💡 创新点**

创新点在于：①使用优势加权的离散流匹配进行离线预训练；②引入路径空间信任区域的KL正则，防止遗忘；③针对大规模动作空间设计候选集近似，并给出覆盖误差理论；

**🔧 技术方法**

核心技术包括CTMC生成器、离散流匹配（DFM）、优势加权目标策略、路径空间KL正则、候选集构造与独立耦合传输；

**📊 数据集**

实验数据集覆盖Jericho文本游戏、MinAtar、以及将MuJoCo离散化后的D4RL任务；

**📈 对比分析**

与CQL、AWAC、DQN、PPO等九种离线到在线基线以及文本游戏中的DRRN、CALM、KG‑A2C等方法对比，DRIFT在Jericho上平均正则化分数最高（23.2%），在多项MinAtar游戏和D4RL离散化任务上亦实现最优或接近最优表现；

**⚠️ 局限性**

局限性包括：路径空间KL在参考策略弱时可能抑制探索；候选集近似在极大动作空间下需进一步验证；以及对多目标、多智能体等更复杂场景的适用性尚未探讨。

---

## 797. Simultaneously Minimizing Storage and Bandwidth Under Exact Repair With Quantum Entanglement

**arXiv ID:** 2605.12455 | [PDF](https://arxiv.org/pdf/2605.12455v1)

**作者:** Lei Hu `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在分布式存储系统中，提出了一种利用量子纠缠实现的精确重建（exact repair）方案，能够在节点失效时通过共享的纠缠态与量子测量，精确恢复失效节点的存储内容；

**💡 创新点**

该工作突破性地证明了在量子纠缠辅助下，d≥2k-2 时的最优重建点（α=B/k，dβq=B/k）在精确重建约束下仍然可实现，而此前仅在功能重建（functional repair）下已知；

**🔧 技术方法**

采用经典的 product‑matrix 结构设计存储矩阵，并结合 Calderbank‑Shor‑Steane (CSS) 量子纠错码的 stabilizer 形式来编码和解码经典信息；

**📊 数据集**

本文没有使用公开数据集，所有结果均为理论证明与构造示例；

**📈 对比分析**

通过理论上限证明与构造性编码实现，表明该方案在存储与重建带宽方面达到最优；未给出数值实验比较；

**⚠️ 局限性**

主要局限包括：仅适用于 d≥2k-2 的参数范围；需要预共享量子纠缠资源，且实现复杂度相对较高；对大规模系统的扩展和实际量子硬件实现仍有挑战。

---

## 798. The Algorithmic Caricature: Auditing LLM-Generated Political Discourse Across Crisis Events

**arXiv ID:** 2605.12452 | [PDF](https://arxiv.org/pdf/2605.12452v1)

**作者:** Gunjan `[一作]` (New York University), Talal Rahwan `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了九个危机事件的观测与合成政治话语配对语料库，并从情感强度、结构规律、词汇框架和跨事件依赖四个维度对合成话语与观测话语进行群体层面的对比，提出Caricature Gap度量；

**💡 创新点**

首次将计算社会科学视角引入AI文本审核，聚焦话语人口层面的真实性而非单句检测，提出Caricature Gap度量，揭示合成话语在情感压缩、结构同质化和词汇抽象化等方面的系统偏差；

**🔧 技术方法**

使用VADER情感分析、基于Transformer的毒性检测、TF‑IDF词汇差异、均值差、Cohen d、方差比等统计度量，并通过Bootstrap CI和Mann‑Whitney U检验进行评估；

**📊 数据集**

约1,789,406条观测帖子（Twitter、Telegram、Reddit、YouTube等）与约566,936条ChatGPT 4/5.5生成的合成帖子，覆盖COVID‑19、Jan‑6、2020/2024美国大选、Dobbs/ROE、BLM抗议、州议选举、犹他枪击、伊朗‑美国战争等九个危机事件；

**📈 对比分析**

通过对四个维度的均值差异与方差比进行描述性统计，并用Caricature Gap量化事件级差距；结果显示合成话语情感更负、结构更均匀、词汇更抽象，差距大小受事件类型调节；Cohen d范围-0.98~+0.82，95% CI均不跨零，表明差异显著；

**⚠️ 局限性**

观测语料不保证全为人类作者；合成语料受提示、模型与解码参数限制；事件规模、平台差异导致可比性受限；所用情感、毒性与词频指标仅为代理；结果解释为假设，未能确定具体机制。

---

## 799. Environment-Adaptive Preference Optimization for Wildfire Prediction

**arXiv ID:** 2605.12435 | [PDF](https://arxiv.org/pdf/2605.12435v1)

**作者:** Enyi Jiang `[一作]` (University of Illinois Urbana Champaign), Wu Sun `[通讯]` (Carnegie Institution for Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出环境自适应偏好优化 (EAPO) 框架，利用 KNN 检索构建与测试分布对齐的局部子集，并在该子集上联合监督学习与偏好优化，以提升极端火灾预测的鲁棒性。

**💡 创新点**

创新点在于将长尾不平衡与分布漂移问题统一到一个偏好学习框架中，使用局部 KNN 检索实现分布对齐，并在本地子集上执行偏好优化，以改进极端事件的相对风险排序。

**🔧 技术方法**

采用 TabNet 神经网络、KNN 局部检索、DPO（偏好优化）、监督损失结合、以及 Focal 损失/交叉熵等技术。

**📊 数据集**

使用 GridMET 气象特征与 GFED5 火灾标签，聚焦加州 Yosemite 区域，训练集为 2001‑2020 年数据，测试集为 2021 年数据。

**📈 对比分析**

与逻辑回归、XGBoost、TabNet 的 BCE/Focal 基线进行对比；EAPO 在 recall、F1、ROC‑AUC 上显著提升，最优配置（Focal + EAPO, k=3）取得 73.10% 的 ROC‑AUC、66.12% 的召回率。

**⚠️ 局限性**

局限性包括仅在单一地理区域验证，未区分自然与人为火灾，且局部检索依赖历史数据的可用性与相似性，未充分解决空间异质性与新型火灾模式的问题。

---

## 800. AOI-SSL: Self-Supervised Framework for Efficient Segmentation of Wire-bonded Semiconductors In Optical Inspection

**arXiv ID:** 2605.12430 | [PDF](https://arxiv.org/pdf/2605.12430v1)

**作者:** Joaquín Figueira `[一作]` (Eindhoven University of Technology), Egor Bondarev `[通讯]` (Eindhoven University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AOI-SSL 框架，在 wire-bond 半导体自动光学检测中通过小规模自监督预训练 Vision Transformer 并结合无训练的 patch‑level 检索，实现低标注成本与快速模型适配的语义分割。

**💡 创新点**

创新点：① 在极低数据量下使用 Masked Autoencoder 进行域特定预训练；② 将检索作为 in‑context 推理，实现无额外参数训练的即时适配；③ 对比传统卷积基线和 ImageNet 预训练模型，展示显著性能提升。

**🔧 技术方法**

技术：MAE 自监督预训练、ViT/Tiny 与 FasterViT 架构、轻量 UPerNet 解码器、patch‑level 相似度聚合检索、in‑context 推理。

**📊 数据集**

数据集：两通道单色 AOI 图像，7000+无标签样本（多设备、多分辨率）+625 带像素级多标签标注样本（wire、ball、wedge、epoxy）。

**📈 对比分析**

方法比较：在固定 8 小时训练预算下，MAE‑pretrained FasterViT + UPerNet 在 mIoU 上比 ResNet18+U‑Net++ 提升约 40%，检索方法在单设备设置下达到 71.5% mIoU，超过全量 fine‑tuning；与 ImageNet 预训练卷积基线相比，提升 8+ 百分点。

**⚠️ 局限性**

限制：对高分辨率图像和更大标签集合的适用性尚未评估；class imbalance 仍需改进；检索内存规模与计算成本相关，需进一步优化。

---

## 801. Stories in Space: In-Context Learning Trajectories in Conceptual Belief Space

**arXiv ID:** 2605.12412 | [PDF](https://arxiv.org/pdf/2605.12412v1)

**作者:** Eric Bigelow `[一作]`, Ekdeep Singh Lubana `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了大型语言模型（LLM）在读故事时的情感概念信念动态，并将其建模为低维几何概念信念空间中的轨迹。

**💡 创新点**

创新点在于将贝叶斯推理与概念空间理论结合，证明LLM的信念更新可被解读为在结构化低维流形上的平滑路径，并且该空间可被线性探测器解码与操纵。

**🔧 技术方法**

主要技术包括对行为信念（情感评分）与隐藏激活的UMAP/PCA降维、线性探测器预测、激活向量加法干预以及距离矩阵与层次聚类分析。

**📊 数据集**

实验使用Synthetic SimpleStories数据集，包含2500条训练故事和500条测试故事，涵盖情感、类型与随机概念三大领域。

**📈 对比分析**

与激活层的线性探测器比对显示，激活层9能以约0.09RMSE准确预测情感和类型信念；流形距离高度相关（r≈0.9），并能预测干预引起的信念偏移。

**⚠️ 局限性**

局限包括仅考察手工选定的单一概念域与非组合概念、未覆盖角色级别或更复杂概念，且激活干预仅在多层协同作用下有效。

---

## 802. OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning

**arXiv ID:** 2605.12400 | [PDF](https://arxiv.org/pdf/2605.12400v1)

**作者:** Yuxiao Yang `[一作]` (University of North Carolina Chapel Hill), Weitong Zhang `[通讯]` (University of North Carolina Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于结果引导的logit steering的 on‑policy 自我蒸馏框架 OGLS‑SD；

**💡 创新点**

创新点在于通过对成功与失败 roll‑out 的教师 logits 进行对比，校正教师分布，缓解因自我反思和特权提示导致的分布不匹配与过度自信；

**🔧 技术方法**

结合了 logit steering、on‑policy 自我蒸馏、基于奖励的验证和多样本教师 logits 平均的技术；

**📊 数据集**

在 OpenThought 数据集上训练，并在 AIME 2024 与 AIME 2025 的数学推理基准上评测，使用 Qwen3‑1.7B 与 Qwen3‑4B 两种模型；

**📈 对比分析**

与 SFT、GRPO、OPSD 等基线对比，OGLS‑SD 在 AIME 2024/25 的 mean@8 分别提升约 2–3 分，且训练曲线更为稳定；

**⚠️ 局限性**

局限性包括仍需在训练阶段提供特权信息、对非可验证奖励的适用性有限，以及在更广泛任务和规模上的泛化仍待验证。

---

## 803. SafeManip: A Property-Driven Benchmark for Temporal Safety Evaluation in Robotic Manipulation

**arXiv ID:** 2605.12386 | [PDF](https://arxiv.org/pdf/2605.12386v1)

**作者:** Chengyue Huang `[一作]` (Georgia Institute of Technology), Lu Feng `[通讯]` (University of Virginia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于可重用时序安全模板的基准，用来评估机器人抓取、放置等操作的时序安全性，并将执行轨迹映射为符号谓词序列进行监控；

**💡 创新点**

创新点包括：①定义覆盖碰撞、抓取稳定性、释放稳定性、交叉污染等八类安全的可复用LTLf安全模板；②将运行轨迹自动符号化并通过DFA实时监控；③通过此基准展示任务成功率并不一定意味着安全完成，揭示了时间安全违规的分布；

**🔧 技术方法**

技术手段包括LTL over finite traces、DFA监控、谓词绑定（从仿真状态提取）、视觉‑语言‑动作（VLA）策略、以及多维安全指标评估；

**📊 数据集**

使用了RoboCasa365厨房仿真任务集，共50个多步骤任务，涵盖了Atomic、Fixture、Beverage、Cooking、Cleaning、Storage、Plating等七类操作；

**📈 对比分析**

对六个预训练/微调的VLA策略进行50次rollout/任务，报告任务成功率与安全违规率。结果表明，尽管部分策略提升了任务成功率，但安全违规率仍高，且违规主要集中在碰撞与释放稳定性等类别；

**⚠️ 局限性**

局限性在于只评估预设的时序安全属性，未覆盖所有风险；基准基于仿真状态，迁移到真实机器人需要可靠感知与触碰估计；评估范围仅限于50任务与固定策略检查点，缺乏系统的提示设计或泛化研究。

---

## 804. Trust the Batch, On- or Off-Policy: Adaptive Policy Optimization for RL Post-Training

**arXiv ID:** 2605.12380 | [PDF](https://arxiv.org/pdf/2605.12380v1)

**作者:** Rasool Fakoor `[一作]` (Boson AI), Alexander J. Smola `[通讯]` (Boson AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 P3O（Policy-on-Policy-off-Policy Optimization）目标，在大模型后期强化学习中通过批量有效样本大小（ESS）自适应地替代固定裁剪，实现更稳健的更新。

**💡 创新点**

创新点在于：1) 用 ESS 动态决定 score‑function 截断与正则化强度，完全去掉固定 clip 范围、行为权重上限与 staleness 预算；2) 通过单一统计量统一处理 on‑policy 与 off‑policy 数据，使同一目标在不同采样误差下自适应调整。

**🔧 技术方法**

使用了：P3O 目标、PPO/GRPO 等基准框架、分布式并行训练、BF16 与 FP8 混合精度推理、温度偏移实验、有效样本大小 (ESS) 计算、分布式同步与评估。

**📊 数据集**

数据集：DeepScaleR‑Preview（4 万题数学推理数据），在 Qwen3‑4B‑Thinking‑2507 与 Qwen2.5‑1.5B 两大模型上训练；评估基准包括 AIME24/25/26、AMO‑Bench 与 AMC 五大数学推理测试集。

**📈 对比分析**

与 GRPO、DAPO、GSPO 等固定 clip 方法在相同超参设置下对比，P3O 在 clip 参数敏感性、温度偏移、BF16/FP8 混合精度等多种离线场景下表现更稳健，最终基准得分与最优 GRPO 基线相当或更优，且无需额外超参调节。

**⚠️ 局限性**

局限性：1) ESS 仍依赖行为数据的支持和覆盖，极端稀疏或多源漂移下可能失效；2) 单一 ESS 可能无法充分区分不同维度的 drift（如策略变化与内部更新差异）；3) 在极大模型或极高维度任务中需进一步验证 ESS 的计算稳定性与收敛性。

---

## 805. Events as Triggers for Behavioral Diversity in Multi-Agent Reinforcement Learning

**arXiv ID:** 2605.12388 | [PDF](https://arxiv.org/pdf/2605.12388v1)

**作者:** Hannes Büchi `[一作]`, Amanda Prorok `[通讯]` (University of Cambridge)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出一种事件驱动的多智能体行为适应框架，将代理身份与行为解耦，使用神经流形多样性指标与事件驱动的LoRA超网络实现动态行为分配

**💡 创新点**

创新点包括：事件中心化的行为分配范式、基于行为流形的距离度量Neural Manifold Diversity、利用LoRA生成器实现轻量级行为切换、以及通过梯度投影保证多样性约束与奖励最大化兼容

**🔧 技术方法**

使用了多智能体强化学习、事件检测机制、Transformer超网络、LoRA低秩适配、Wasserstein距离多样性度量、梯度投影与多样性约束优化等技术

**📊 数据集**

在VMAS仿真平台上测试了Navigation、Dispersion、Reverse Transport、Football等基准任务以及自定义的Pressure Plate和Wind Flocking环境，没有使用公开数据集

**📈 对比分析**

与HyperMARL、CASH、DiCo、PS等基线进行对比，评估完成率、平均奖励和回合长度；实验显示该框架在所有任务中均获得最高完成率（≥95%），在Football等复杂任务中显著优于基线，并实现了对代理数量、能力和多样性目标的零样本泛化，以及对未知事件序列的鲁棒性

**⚠️ 局限性**

局限性包括：行为表达受限于LoRA和超网络的表示能力；事件定义需先验手工指定；超网络集中式，需要访问所有代理观测；对事件和多样性目标的自动发现与优化仍待改进

---

## 806. Proof Systems Based on Structured Circuits

**arXiv ID:** 2605.12378 | [PDF](https://arxiv.org/pdf/2605.12378v1)

**作者:** Matthäus Micun `[一作]` (Technische Universität Ilmenau), Christoph Berkholz `[通讯]` (Technische Universität Ilmenau)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了基于结构化电路（SDD和d‑SDNNF）的命题证明系统，并与传统的OBDD证明系统进行了系统比较，证明在缺少弱化规则时，SDD/d‑SDNNF系统能够给出更短的不可满足CNF反证；

**💡 创新点**

创新点在于：①提出将结构化电路作为证明系统基础的框架；②提出“lifting”方法将可满足CNF的表示难度提升为不可满足CNF的反证难度；③给出了多种证明系统之间的严格分离，揭示重排、弱化等规则的相对强度；

**🔧 技术方法**

主要技术包括：结构化电路与变量树（vtree）的理论、OBDD/SDD/d‑SDNNF的闭包与等价性判定算法、通信复杂度与矩形覆盖的关系、可扩展图与树宽构造、以及lifting定理与局部重构规则；

**📊 数据集**

使用的实例主要是理论构造的CNF，如顶点覆盖公式VC_G、扩展图、shifted equality公式等；未使用实际数据集；

**📈 对比分析**

通过证明上界和下界来比较不同证明系统的复杂度；结果显示在无弱化条件下，SDD/d‑SDNNF能给出多项式规模的反证，而相应的OBDD(,r)系统需要指数级别规模；

**⚠️ 局限性**

局限性包括：lifting方法不足以实现指数级分离；对弱化规则的可验证性仍不明确；无法证明OBDD(,w)是否能模拟SDD(,w)；对更强表示形式（如自由二进决策图、SDNNF）证明系统的可验证性问题仍为开放问题。

---

## 807. Layer-Based Width for PAFP

**arXiv ID:** 2605.12457 | [PDF](https://arxiv.org/pdf/2605.12457v1)

**作者:** Samuel German `[一作]` `[通讯]` (University of California, San Diego), Samuel German (University of California, San Diego)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了路径避免禁用对（PAFP）问题，并从层宽度的角度提出了新的可判定性与复杂度结果。

**💡 创新点**

创新点在于引入基于BFS层的宽度参数（union digraph BFS宽度及输入图 BFS宽度）来划分可解与不可解区间，并给出了BFS宽度+向后弧数的FPT算法，同时给出BFS宽度为2的“正则化”构造证明了仅有宽度不足以保证可解，进一步揭示了向后弧数的必要性。

**🔧 技术方法**

主要技术包括层宽度分析、图的正规化与“脊柱+分支”构造、动态规划、2-SAT 编码、以及基于树宽度的MSO模型检查框架。

**📊 数据集**

本工作为纯理论研究，无实验数据集；所有结论均通过构造性证明与多项式时间归约获得。

**📈 对比分析**

通过FPT算法与多项式时间2-SAT实现，能在参数化条件下高效求解；而在BFS宽度或 exact-length 层宽度为3、或无向图宽度为2的情形下，问题被证明为NP-完全，说明了方法的边界。

**⚠️ 局限性**

局限性在于：仅靠BFS宽度不足以得到可判定性；需要对向后弧数做进一步限制；另外，论文中所有结果均针对有向无环图，尚未讨论更一般的有向图情况。

---

## 808. FuTCR: Future-Targeted Contrast and Repulsion for Continual Panoptic Segmentation

**arXiv ID:** 2605.12451 | [PDF](https://arxiv.org/pdf/2605.12451v1)

**作者:** Nicholas Ikechukwu `[一作]` (Boston University), Bryan A. Plummer `[通讯]` (Boston University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 FuTCR 框架，用于在持续式全景分割任务中提前为未来类别准备表示空间，从而提升新类别的识别效果。

**💡 创新点**

创新点在于通过模型自身的掩码预测自动发现“未来型”未标记区域，采用像素到区域的对比学习构建一致的原型，并对已知类别进行排斥，以预留特征子空间供后续类别使用。

**🔧 技术方法**

技术主要包括查询式全景分割架构（Mask2Former）、InfoNCE 对比损失、基于余弦相似度的排斥损失，以及结合传统全景分割损失的联合训练策略。

**📊 数据集**

实验在 ADE20K 数据集上进行，使用不同的类增量设置（100–50、100–10、100–5）并在重叠与不重叠图像流两种场景下评估。

**📈 对比分析**

与 SimCIS 等现有方法对比，FuTCR 在新类别的全景质量 PQ_new 方面相对提升最高达 28%，总体 PQ_all 也提升 4% 左右，同时保持甚至提升基类性能。

**⚠️ 局限性**

局限性包括仅在 ADE20K 上验证，随机种子和网络架构有限，未来需要在更多数据集、不同增量顺序和更大规模模型上进一步验证其鲁棒性。

---

## 809. Basilisk and Docker for Reproducible GN&C Simulation: A Workflow Reference

**arXiv ID:** 2605.12443 | [PDF](https://arxiv.org/pdf/2605.12443v1)

**作者:** Anubhav Gupta `[一作]` `[通讯]` (University of Colorado Boulder), Anubhav Gupta (University of Colorado Boulder)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了一个基于 Docker 的 Basilisk 仿真容器化工作流，包含从单脚本到 BSKSim 类层次的示例，并支持蒙特卡洛参数不确定性分析。

**💡 创新点**

创新点在于：① 通过 Docker 封装完整的 Basilisk 构建环境，实现可重复、可移植的 GN&C 仿真；② 设计了从简单脚本到 BSKSim 类层次的结构化工作流；③ 提供了内置的 Monte Carlo 模块，用于快速执行参数随机化实验。

**🔧 技术方法**

使用技术包括：Docker、Docker Compose、Conan 包管理、CMake、Python、Basilisk 框架、Vizard 可视化、SPICE 轨道数据、YAML 配置以及 Monte Carlo 工具箱。

**📊 数据集**

使用的数据集包括：Basilisk 自带的 GGM03S J2 斜率数据、SPICE kernels（DE430、NAIF、pck、tpc 等）以及自定义的支持文件（如大文件的 recover.sh）。

**📈 对比分析**

对比方法：将容器化部署与传统手动安装方式对比，结果表明部署时间更短（15–20 分钟），环境一致性更好；运行时性能与非容器环境相当。论文未给出详细性能基准。

**⚠️ 局限性**

局限性：① 需随 Basilisk 版本更新维护 Dockerfile；② 依赖 SPICE kernel 文件位置和完整性；③ Vizard 需要合适的 GPU 驱动；④ Docker 镜像体积大、构建时间长；⑤ 依赖版本兼容性仍需关注。

---

## 810. GaitProtector: Impersonation-Driven Gait De-Identification via Training-Free Diffusion Latent Optimization

**arXiv ID:** 2605.12431 | [PDF](https://arxiv.org/pdf/2605.12431v1)

**作者:** Huiran Duan `[一作]` (City University of New York), Yingli Tian `[通讯]` (City University of New York)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于预训练3D视频扩散模型的无训练伪装框架GaitProtector，用于在保持步态结构的同时实现身份去标识与仿冒的双向拉伸；

**💡 创新点**

创新点在于将身份去标识与仿冒目标结合为联合目标，并借助冻结的扩散先验作为结构正则化；

**🔧 技术方法**

核心技术包括：预训练的3D视频扩散模型、DDIM逆推与中间采样、可微软二值化、以及在潜在空间进行的隐式身份指导优化；

**📊 数据集**

实验使用CASIA‑B步态识别数据集评估隐私与视觉质量，并在Scoliosis1K临床步态数据集评估下游诊断效能；

**📈 对比分析**

与基线（基于轮廓的PGD、仅VAE、仅去标识）对比，GaitProtector在黑盒攻击下的仿冒成功率提升约47.5%，识别Rank‑1从89.6%降至15%，同时视频质量指标（FVD、LPIPS）明显优于基线；

**⚠️ 局限性**

局限性包括：对目标与源匹配的敏感性（视角/姿态差异导致仿冒效果下降），计算成本高（单帧约3分钟），以及对完整身份表示的依赖，难以在实时或大规模部署中使用。

---

## 811. Geometric Factual Recall in Transformers

**arXiv ID:** 2605.12426 | [PDF](https://arxiv.org/pdf/2605.12426v1)

**作者:** Shauli Ravfogel `[一作]` (New York University), Alberto Bietti `[通讯]` (Flatiron Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Transformer语言模型如何通过几何方式存储事实关联，并证明在共享属性情形下仅需对数维度即可记忆。

**💡 创新点**

提出了基于学习嵌入的几何记忆机制，表明主键值记忆不再是必需，MLP充当关系条件选择器。

**🔧 技术方法**

利用单层Transformer、ReLU门控MLP、线性回归探测、因果干预、零样本迁移等技术。

**📊 数据集**

在合成的共享属性数据集上（N=4096，R多取值，d多维度）以及在预训练LLM的实体关系数据上进行验证。

**📈 对比分析**

与关联记忆理论的对数维度下的表现进行对比；实验显示学习嵌入可将容量提升至Θ(RlogN)，而固定随机嵌入需要Θ(NR)维度，且迁移性能优越。

**⚠️ 局限性**

仅在简化的单层、共享属性的实验中验证，未考虑多词实体、语义相关性及真实文本中多样化属性的影响。

---

## 812. Extending QuAK with Nested Quantitative Automata

**arXiv ID:** 2605.12418 | [PDF](https://arxiv.org/pdf/2605.12418v1)

**作者:** Thomas A. Henzinger `[一作]` (Institute of Science and Technology Austria), Harun Yılmaz `[通讯]` (Sabancı University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了支持嵌套量化自动机（NQA）的工具 QuAK，并通过扁平化将 NQA 转化为 QA 以解决阈值问题。

**💡 创新点**

首次提供了可行的 NQA 扁平化算法并将其集成到现有 QA 工具中，使得平均响应时间等之前无法表达的属性可被量化验证。

**🔧 技术方法**

采用了基于约束、阈值裁剪、极值父子优化、同步化以及无声权重消除等多种扁平化技术，结合原有的空性、普适性和极限平均决策程序。

**📊 数据集**

使用了两类基准：请求‑授权协议的响应时间模型（参数化的 n,k）以及动态进程资源消耗模型（参数化的 n,k）。

**📈 对比分析**

通过与未扁平化或纯 QA 方案对比，实验表明在大多数参数设置下能在秒级完成空性/普适性判定，但当同时激活子机数量或子机状态空间增大时仍会出现指数级或内存耗尽。

**⚠️ 局限性**

扁平化会导致子机返回值域扩展和状态爆炸，且对某些未解决的 NQA 组合（如 limit‑average）仍缺乏高效算法，工具依赖于显式状态表示。

---

## 813. Scalable Token-Level Hallucination Detection in Large Language Models

**arXiv ID:** 2605.12384 | [PDF](https://arxiv.org/pdf/2605.12384v1)

**作者:** Rui Min `[一作]` (Sea AI Lab), Yi R. Fung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

TokenHD 是一个针对推理任务的全流程、基于 token 的幻觉检测框架，直接在自由文本上对每个 token 进行评分，无需预先划分步骤；

**💡 创新点**

其创新点在于：①利用可扩展的数据合成引擎通过多模型批评者生成 token 级标签并通过自适应加权集成提升标注质量；②采用重要性加权交叉熵解决幻觉 token 稀缺导致的不平衡；③在轻量化后端（Qwen3 0.6B-8B）实现了比大规模推理模型更高的检测精度；

**🔧 技术方法**

技术包括 tokenization/ detokenization、基于批评模型的幻觉片段恢复与 token 对齐、均值与自适应加权集成、重要性加权训练策略、以及在 Qwen3 系列模型上 fine‑tune 幻觉检测器；

**📊 数据集**

训练与评测数据涵盖 Math、AceReason‑Math、Big‑Math、Math‑500、AIME‑2024/2025、OlympiadBench‑Math、GPQA、OlympiadBench‑Phy、FinQA 以及 Code‑Elo、LiveCodeBench‑Lite；政策模型使用 GPT‑4o‑mini、Gemini‑2.0‑Flash、Claude‑3.5‑Haiku、Qwen2.5‑7B‑Instruct；标注者则包括 GPT‑5、o3、GPT‑4.1、R1‑Qwen3‑8B、QwQ‑32B、o4‑mini；

**📈 对比分析**

与基线后端、批评模型、标签模型以及 PRM 进行对比，TokenHD‑0.6B 在 S_incor 上已超过 QwQ‑32B，0.6B‑8B 规模随之提升，整体检测 F1 在数学与 STEM 基准上均显著优于现有方法，且在多任务与多政策模型下保持稳健；

**⚠️ 局限性**

局限性包括：①依赖批评模型生成标签，若批评模型本身缺陷会影响标注质量；②在非推理领域或极端分布偏移时仍需额外训练或合并策略；③数据合成与恢复流程较为复杂，增加工程成本；④目前未深入研究对模型规模极限的边界与更细粒度的自校正机制。

---

## 814. ProfiliTable: Profiling-Driven Tabular Data Processing via Agentic Workflows

**arXiv ID:** 2605.12376 | [PDF](https://arxiv.org/pdf/2605.12376v1)

**作者:** Wei Liu `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于动态探测的多智能体框架ProfiliTable，用于自动化完成表格清洗、转换、增补和匹配等任务。

**💡 创新点**

将交互式ReAct式探测、检索增强生成（RAG）与反馈驱动迭代融合，形成统一的动态配置上下文，显著提升了LLM在表格处理中的鲁棒性与准确率。

**🔧 技术方法**

使用ReAct式数据探测、RAG检索预验证算子库、评估器与Summarizer的交互式验证、最终izer的最佳方案挑选，并以GPT‑4o/GPT‑5.2为底层语言模型。

**📊 数据集**

在自建的18种表格处理任务基准（覆盖清洗、转换、增补、匹配）上进行实验，并在DataGovBench基础上扩展得到的多任务测试集。

**📈 对比分析**

与MetaGPT、CAMEL、ChatDev2.0、DataGovAgent等多种基线对比，单步任务ATS达86.8–89.7%，多步任务ATS达80.2–82.5%，任务可执行率100%，同时保持较低的token消耗与执行时间，显著优于所有对比方法。

**⚠️ 局限性**

对极深层多步骤任务和指令含糊（如“NA”解释）仍存在性能不足，且依赖手工构建的算子库和预验证模板，无法自动处理完全未知领域的表格。

---

## 815. Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs

**arXiv ID:** 2605.12460 | [PDF](https://arxiv.org/pdf/2605.12460v1)

**作者:** Guinan Su `[一作]` (Max Planck Institute for Intelligent Systems), Jonas Geiping `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多流并行生成（Multi-Stream Parallel Generation）框架，将传统的单一聊天流拆分为多条平行流（如用户流、系统流、思考流、工具流等），让模型在一次前向传播中同时读取多条输入流并生成多条输出流，从而实现读/思考/行动的并行化。

**💡 创新点**

创新点包括：
1) 通过指令微调将LLM迁移到多流格式；
2) 设计跨流因果注意力掩码和流感知位置编码（RoPE + 流嵌入），保证各流间的时间一致性；
3) 采用交错打包（interleaved packing）实现高效的FlashAttention路径；
4) 通过等待-k策略和纯合成表格生成的自动化数据构建，生成满足因果约束的多流训练样本；
5) 展示多流模型在速度、抗提示注入、可监控性等方面的显著提升。

**🔧 技术方法**

技术手段：
- 指令微调（Instruction Tuning）适配多流格式；
- 交错打包（interleaved packing）+ 交叉流因果掩码（cross-stream causal mask）；
- 流感知RoPE位置编码 + 可学习流嵌入；
- FlashAttention / FlexAttention 适配多流；
- 通过“wait‑k”策略生成合成多流数据并进行因果验证与质量过滤；
- 在推理时采用同步多流解码，使用空占位符（e.g., `<pad>`）实现无KV占用。

**📊 数据集**

数据集：
- 基于公开语料（如Alpaca、通用对话语料）使用LLM自动转为多流格式；
- 通过“wait‑k”策略生成的多流对话；
- 纯合成的多流表格对话（模型直接生成多流对话），随后进行因果验证与质量过滤；
- 用于评估的标准基准：GSM‑8K、MATH‑500、LogicNLI、SQuAD、以及安全与监控评测集（StruQ、NESSiE、Gandalf等）。

**📈 对比分析**

比较方法与性能：
- 对比基线（Base）、单流微调（Vanilla）和多流模型（Stream）。
- 评估指标：准确率（Acc）、Token Number to First Target Token（TNFT）、总生成 token 数、端到端延迟（Delay）、最大流长度（MSL）。
- 结果显示：多流模型在所有任务上 TNFT 均降为 0（即无首词延迟），生成 token 数减少 30‑60%，延迟降低 30‑70%；准确率与 Vanilla 相近或略有提升。
- 在安全评测中，多流模型的 Prompt Injection 成功率（ASR）平均下降 15‑30%，安全-帮助性评分提升；
- 监控评测中，内部流的“下意识子发声”指标提升 3‑5 倍，监测准确率翻倍。

**⚠️ 局限性**

局限性：
- 研究规模受限，仅使用小型模型（1.7B/4B）及有限的训练样本；
- 对大型商业模型的验证尚未完成；
- 多流模式对纯文本处理、有限交互或高度顺序化任务（如证明书写）收益有限；
- 目前仅实现密集跨流注意力，未探索更高效的稀疏或单向交互变体；
- 需要进一步研究多流的安全、偏差和公平性等长期影响。

---

## 816. LychSim: A Controllable and Interactive Simulation Framework for Vision Research

**arXiv ID:** 2605.12449 | [PDF](https://arxiv.org/pdf/2605.12449v1)

**作者:** Wufei Ma `[一作]` (Johns Hopkins University), Alan Yuille `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了LychSim，一种可控、交互式的虚拟仿真框架，支持Python API、流程化数据生成和与LLM的MCP集成，专为计算机视觉研究设计。

**💡 创新点**

创新点在于统一的Python接口降低学习门槛，内置丰富的2D/3D标注（包括可见范围之外的深度、遮挡关系、部件分割等），以及通过MCP实现LLM与仿真环境的闭环交互。

**🔧 技术方法**

使用了Unreal Engine 5、UnrealCV插件、Python封装、Model Context Protocol（MCP）、强化学习算法及大型语言模型（Claude Opus 4.6、Gemma 4）。

**📊 数据集**

基于UE5 Marketplace 3D资产、Infinigen、HSSD‑200等场景，采用程序化规则生成多样化环境，并提供相应的像素级2D/3D标注。

**📈 对比分析**

通过对比现有仿真框架（如Infinigen、UnrealCV、UnrealZoo等）以及在SAM等模型上的RL对抗评估，LychSim在可控性、标注丰富度和交互性上优于传统方案，RL对抗实验显示能显著降低SAM的IoU。

**⚠️ 局限性**

局限性包括对物理真实性的支持不足（碰撞/重力等），对大型语言模型的空间推理能力限制导致生成场景仍可能出现不物理合理的布局，且仍受UE5资产库的规模和多样性约束。

---

## 817. Scalable Packed Layouts for Vector-Length-Agnostic ML Code Generation

**arXiv ID:** 2605.12445 | [PDF](https://arxiv.org/pdf/2605.12445v1)

**作者:** Ege Beysel `[一作]` (RooflineAI), Jan Moritz Joseph `[通讯]` (RooflineAI)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在MLIR/IREE编译器中统一使用向量长度参数化的打包数据布局与编译器扩展，实现端到端可跨向量长度的机器学习代码生成。

**💡 创新点**

将可扩展向量长度的打包布局与tiling、fusion、vectorization等核心编译器转换整合，使单一二进制在不同向量长度硬件上无缝运行，同时保持高性能。

**🔧 技术方法**

采用MLIR与IREE的自定义方言和Pass，设计可扩展SVE打包布局、自动化的矩阵乘法微内核生成、可扩展向量化与裁剪，以及符号向量长度的处理；使用gem5对向量长度扩展性进行仿真验证。

**📊 数据集**

在多种真实开源模型上评估，包括Transformer、Vision、Audio等模型（DeepSeek、Llama、Qwen、SmolLM、SmolVLM、AST、MobileBERT、ViT、Whisper、XLM‑R、YOLOS），所有模型均以FP32量化后推理。

**📈 对比分析**

与IREE的NEON实现、ExecuTorch、TorchInductor和PyTorch eager等进行对比；在Radxa Orion O6和Pixel 9上跑50次平均延迟，SVE实现比NEON最快提升1.45×，相较PyTorch框架平均提升1.7×至12.38×；在gem5模拟器上向量长度从128位扩展到512位时可达约3.4×的加速。

**⚠️ 局限性**

目前仅针对矩阵乘法实现可扩展打包布局，卷积未得到专门优化；对BF16、INT8等其他数据类型和SME、RVV等可扩展指令集的支持仍有限；实验平台仅支持128位SVE，无法实测真实跨长度性能。

---

## 818. 3D Gaussian Splatting for Efficient Retrospective Dynamic Scene Novel View Synthesis with a Standardized Benchmark

**arXiv ID:** 2605.12437 | [PDF](https://arxiv.org/pdf/2605.12437v1)

**作者:** Yunxiao Zhang `[一作]` (Texas A&M University), Suryansh Kumar `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在同步多摄像机设置下，利用热启动且不做稠密化的3D高斯光散射（3DGS）实现高质量、可回溯的动态场景新视角合成。

**💡 创新点**

创新点在于证明在严格同步多视角下，显式时间耦合并非必要，采用固定数目高斯热启动链式优化即可保持几何一致性和内存可预测性。

**🔧 技术方法**

使用了结构光法（SfM）初始化、3D高斯光散射、球谐函数光照、差分光栅化以及热启动链式优化。

**📊 数据集**

采用了Blender API生成的同步多视角动态数据集，包括D-WS、S-PK、S-MP及天气条件下的足球场数据。

**📈 对比分析**

与4DGS、ST-GS、D-NeRF等方法在PSNR/LPIPS、内存和训练时间上进行对比，实验显示在相同或更低的显存/时间下取得最高PSNR与最低LPIPS。

**⚠️ 局限性**

局限性包括对精准相机标定和同步的强依赖，初始点云稀疏会影响后续帧质量，以及不支持移动或变焦摄像机的场景。

---

## 819. Enhancing Instruction Prefetching via Cache and TLB Management

**arXiv ID:** 2605.12433 | [PDF](https://arxiv.org/pdf/2605.12433v1)

**作者:** Alexandre Valentin Jamet `[一作]` (Barcelona Supercomputing Center), Marc Casas `[通讯]` (Barcelona Supercomputing Center)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为TPPR+ tPB的微架构方案，通过在 sTLB 旁添加 Translation Prefetch Buffer（tPB）降低 L1I 跨页预取的地址翻译延迟，并在 L2 缓存中使用 Trimodal Instruction Prefetch Replacement Policy（TPPR）对预取代码行进行智能替换，从而显著提升 L1I 预取的效能。

**💡 创新点**

①首次将 TLB 预取与 L2 缓存替换策略耦合，利用 tPB 存储跨页预取产生的页表项，避免 sTLB 污染；②TPPR 采用决策树动态选择三种 RRPV 基础策略（优先预取、非优先预取、跳过预取），根据代码行重用模式自动适配，提升预取代码行利用率；③在不改变 L1I 预取器逻辑和软件的前提下，仅增 0.79KB 存储开销。

**🔧 技术方法**

使用 VIPT L1I 预取器（EPI、FNL+MMA、Barça）、x86 5 级 radix 树页表、ChampSim 仿真平台、sTLB 与 iTLB 交互、tPB 的页表项缓存、TPPR 的决策树与计数器、以及多种现有缓存/TLB 替换策略（Mockingjay、SHiP++、CLIP、Emissary、CHiRP、Morrigan 等）进行对比。

**📊 数据集**

105 个单核服务器工作负载（来自 Qualcomm、NodeApp、PHPWiki、TPCC、Twitter、Wikipedia、Kafka、Spring、Tomcat、Chirper、HTTP 等），以及 160 个 4‑核混合工作负载（同一套单核工作负载组合）和 75 个 SMT 工作负载。

**📈 对比分析**

通过 ChampSim 仿真，先对单核基线（L1I 预取 + 默认 sTLB + SRRIP）进行对比，再与各类 TLB/缓存策略（CHiRP、Morrigan、CLIP、Emissary、SHiP++、Mockingjay 等）对照；结果显示与 EPI 结合时 geomean 加速 6.1%，与 Barça 8.3%，与 FNL+MMA 7.9%；在 4‑核多核和 SMT 环境下，提升 7–14%，明显优于现有策略，且仅增加 0.79KB 存储。

**⚠️ 局限性**

当全部内存使用大页（2MB）时，跨页预取次数和 tPB 效果显著降低，收益随大页比例增加而递减；LLC 容量增大时，预取行的重用率提高，TPPR 的相对优势下降；此外方案需要额外的 sTLB MSHR 位和 prefetch 位，对错误路径预取不做专门处理，未来可进一步改进。

---

## 820. GeoQuery: Geometry-Query Diffusion for Sparse-View Reconstruction

**arXiv ID:** 2605.12399 | [PDF](https://arxiv.org/pdf/2605.12399v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 821. Learning Minimally Rigid Graphs with High Realization Counts

**arXiv ID:** 2605.12427 | [PDF](https://arxiv.org/pdf/2605.12427v1)

**作者:** Oleksandr Slyvka `[一作]` (Czech Technical University in Prague), Jan Legerský `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于强化学习的深度交叉熵方法，用图同构网络编码，搜索最小刚性图的极值实现计数与 NAC‑着色数

**💡 创新点**

将最小刚性图构造建模为 0/1 延伸序列的决策过程，使用可交换的扩展级动作表示并结合 Deep CEM 进行高效搜索

**🔧 技术方法**

Graph Isomorphism Network 编码、Permutation‑equivariant 扩展表示、Deep Cross‑Entropy Method、熵正则化与 m‑Bézout 上界筛选等技术

**📊 数据集**

无固定公开数据集，采用随机采样生成图序列并用计算工具评估其实现计数与 NAC‑着色数

**📈 对比分析**

与已知最优/最佳已知结果对比；在平面实现计数上匹配已知最优，在球面实现计数和 NAC‑着色数上刷新最优界；搜索速度比穷举快数百倍

**⚠️ 局限性**

奖励评估昂贵、扩展空间仍大、仅验证到 n≤18、仅考虑复数实现计数、未评估对实际机器人部署的鲁棒性

---

## 822. Predicting Disagreement with Human Raters in LLM-as-a-Judge Difficulty Assessment without Using Generation-Time Probability Signals

**arXiv ID:** 2605.12422 | [PDF](https://arxiv.org/pdf/2605.12422v1)

**作者:** Yo Ehara `[一作]` `[通讯]` (Tokyo Gakugei University), Yo Ehara (Tokyo Gakugei University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种不使用生成时概率信号的LLM评判难度一致性判定方法。

**💡 创新点**

创新点在于利用难度的序数特性，通过外部嵌入空间的几何一致性来预测LLM与人工评测的分歧。

**🔧 技术方法**

方法基于ModernBERT等嵌入模型，计算类别质心差向量并评估与标签的几何一致性。

**📊 数据集**

实验使用CEFR-SP英语句子难度数据集，并在GPT‑OSS‑120B和Qwen3‑235B‑A22B两大LLM上评测。

**📈 对比分析**

与传统基于token log‑prob的基线相比，使用most2方向的几何一致性指标在ROC‑AUC、PR‑AUC和Precision@20上均显著提升。

**⚠️ 局限性**

主要局限在于依赖已定义的序数难度尺度，且对难度分布不均的标签处理仍有不足。

---

## 823. Predicting Decisions of AI Agents from Limited Interaction through Text-Tabular Modeling

**arXiv ID:** 2605.12411 | [PDF](https://arxiv.org/pdf/2605.12411v1)

**作者:** Eilam Shapira `[一作]` (Technion Israel Institute of Technology), Roi Reichart `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在语言交互博弈中，利用少量已知的历史对话与游戏记录，预测陌生 AI 代理在新决策点的行动。

**💡 创新点**

创新点在于：①把结构化游戏状态、对话文本与冻结LLM的隐藏表示三者融合成表格特征；②采用目标适应式文本‑表格学习框架，让少量目标代理历史与大规模源代理数据共同驱动预测；③证明冻结LLM的内部状态比直接输出更具判别力。

**🔧 技术方法**

使用技术包括：TabPFN 文本‑表格基础模型；小型冻结LLM（Gemma‑2‑2B、Qwen3‑1.7B、Llama‑3.2‑1B）作为观察者提取隐藏表示；LLM‑as‑Predictor 作为对比基线；传统游戏+文本特征基线；多层次特征 ablation 与层级分析。

**📊 数据集**

使用的数据集：① GLEE 前沿 LLM 13‑agent 约 64K 游戏的训练源；② 91 只大学 Hackathon 构建的 4,921 场游戏（共 11,341 决策）作为测试目标。

**📈 对比分析**

通过在 0、2、4、8、16 个适应游戏下进行交叉种群迁移实验，比较三种方法：Game+Text、LLM‑as‑Predictor 与 Observer‑增强模型。结果显示：Observer‑增强模型在响应预测上 AUC 提升约 4‑6 个百分点，Bargaining 提议预测误差下降 14%；LLM‑as‑Predictor 在两项任务均低于 Observer‑增强且在数值预测上表现最差。

**⚠️ 局限性**

局限性包括：实验仅在受控博弈环境下进行，未验证在真实市场数据上的泛化；方法依赖可获得的大规模源代理样本；Observer 对不同任务的提升幅度不一致；且对黑盒代理的假设仅限于公开的游戏状态与对话。

---

## 824. Pretraining Exposure Explains Popularity Judgments in Large Language Models

**arXiv ID:** 2605.12382 | [PDF](https://arxiv.org/pdf/2605.12382v1)

**作者:** Jamshid Mozafari `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用公开的OLMo预训练语料库对实体曝光进行大规模计数并与Wikipedia浏览量及LLM的流行度判断进行对比

**💡 创新点**

首次通过可观测的预训练数据直接量化曝光与实体流行度之间的关联，并揭示LLM流行度判断主要受预训练曝光驱动

**🔧 技术方法**

利用Infini‑Gram索引工具统计实体及其别名在7.4万亿token中的出现频次，采用Directly和Comparison两种提示方法采集LLM流行度信号

**📊 数据集**

使用OLMo完整预训练语料库Dolma（约27TB、7.4万亿token）以及Wikidata实体与别名集（2000实体）

**📈 对比分析**

与Wikipedia浏览量比较，所有实体曝光与浏览量的Spearman相关系数平均0.756；LLM的Comparison方法对曝光的相关系数达到0.795（32B模型），远高于Directly（0.753）和Wikipedia本身，表明对曝光更敏感

**⚠️ 局限性**

研究局限在仅分析实体级别的曝光与流行度，未探究事件、主题等更广义概念，也未提供减轻曝光偏差的具体方法

---

## 825. Fast Image Super-Resolution via Consistency Rectified Flow

**arXiv ID:** 2605.12377 | [PDF](https://arxiv.org/pdf/2605.12377v1)

**作者:** Jiaqi Xu `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FlowSR，将图像超分任务改写为从低分辨率到高分辨率的直线轨迹（rectified flow），通过一致性学习实现单步推断，并加入 HR 正则化和快慢时序采样提升性能。

**💡 创新点**

将 rectified flow 与一致性模型结合，提出 HR 正则化的一致性学习以防止教师蒸馏误差，并设计快慢时间调度来平衡采样效率与纹理细节，最终实现单步高质量超分。

**🔧 技术方法**

使用 Stable Diffusion（VAE+UNet）作为骨干，训练 rectified flow 的速度场；采用一致性学习（CD）+HR 正则化；快慢时间调度采样；加入 GAN 对抗损失和基于 CLIP 的图像质量对齐损失；LoRA 微调。

**📊 数据集**

在 LSDIR 真实低分辨率-高分辨率对和 FFHQ 前 10k 张人脸图像上训练，低分辨率通过 Real-ESRGAN 的降解管道生成；评估使用 RealSR 和 DRealSR 测试集。

**📈 对比分析**

与多步扩散超分（StableSR、DiffBIR、SeeSR、PASD、ResShift）以及一步方法（SinSR、OSEDiff、DoSSR）比较；在 PSNR、SSIM、LPIPS、DISTS、FID 等参考指标以及 NIQE、MUSIQ、MANIQA、CLIPIQA 等无参考指标上，FlowSR 在保持单步推理的同时，往往达到或超过竞争者的指标，尤其在 PSNR、LPIPS 以及无参考质量上表现优异。

**⚠️ 局限性**

模型依赖 Stable Diffusion 预训练，训练成本仍较高；在极低分辨率或强噪声下的鲁棒性未充分验证；对不同解码器或不同降解模型的泛化能力尚需进一步研究。

---

## 826. Agent-Based Post-Hoc Correction of Agricultural Yield Forecasts

**arXiv ID:** 2605.12375 | [PDF](https://arxiv.org/pdf/2605.12375v1)

**作者:** Matthew Beddows `[一作]` (University of Aberdeen), Georgios Leontidis `[通讯]` (UiT The Arctic University of Norway)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究提出一种基于LLM代理的后置修正框架，用来改进农作物产量预测，特别针对数据受限的软果场景；

**💡 创新点**

创新点在于将农学领域知识编码为一组可执行工具，在已有预测模型之上实现可解释且可复现的后置修正，而非训练全新模型；

**🔧 技术方法**

技术手段包括ReAct式LLM代理（如Llama 3.1 8B、Qwen 2.5 7B等）与多工具（季节相位检测、偏差学习、范围验证等）相结合，基线模型为XGBoost、Random Forest和Moirai2；

**📊 数据集**

数据集涵盖英国多波棚草莓的专有产量记录和美国公开的USDA玉米收获进度数据；

**📈 对比分析**

通过在三种基线模型上应用代理并与未修正基线对比，实验显示草莓MAE和MASE平均降低约20–28%，玉米MAE/MASE亦显著提升；

**⚠️ 局限性**

主要局限包括高计算成本、对LLM能力的依赖、以及对连续或无明显季节性作物适用性有限，且整体解释性仍有提升空间。

---

## 827. From Web to Pixels: Bringing Agentic Search into Visual Perception

**arXiv ID:** 2605.12497 | [PDF](https://arxiv.org/pdf/2605.12497v1)

**作者:** Bokang Yang `[一作]` (Shenzhen Loop Area Institute), Xiangyu Yue `[通讯]` (CUHK MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了视觉感知的“Perception Deep Research”范式，构建了WebEyes benchmark并提出Pixel-Searcher工作流来先通过网络搜索获取外部证据，再将解码出的目标实体与图像中的实例绑定，实现搜索-基础的定位、分割和VQA任务。

**💡 创新点**

创新点包括：①将知识检索与视觉定位耦合的全新任务设定；②基于对象先验的WebEyes数据集，提供可追溯的外部证据与精准像素标注；③Pixel-Searcher的agentic search-to-pixel两阶段流程，结合实体解析、证据验证与工具调用；④对三种任务视角的统一评估。

**🔧 技术方法**

技术手段主要是：多轮网络搜索与推理（使用Google Search API）、实体解析与对齐、视觉工具调用（如SAM3分割、定位工具）、对齐与矛盾检测、基于LLM的Agent化推理框架。

**📊 数据集**

使用的数据集为WebEyes，包含120张图片、473个对象实例、645个QA对、1,927个任务样本，涵盖车辆、IP、动漫、ICON、名人、产品等多种类别。

**📈 对比分析**

与多种开源与闭源基线（如Qwen3-VL、Doubao、Gemini、GPT系列等）比较，Pixel-Searcher在三类任务上均获得最佳开源成绩，尤其在搜索/定位准确率上提升约20-30%，但仍落后于部分闭源模型。

**⚠️ 局限性**

局限性主要在：搜索规划与证据选择仍易出错，实体解析与视觉实例匹配仍占大部分错误；对多跳推理的支持有限；对实时性和跨语言检索的适应性不足；当前模型仍依赖大模型与外部API，成本高。

---

## 828. AlphaGRPO: Unlocking Self-Reflective Multimodal Generation in UMMs via Decompositional Verifiable Reward

**arXiv ID:** 2605.12495 | [PDF](https://arxiv.org/pdf/2605.12495v1)

**作者:** Runhui Huang `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AlphaGRPO 框架，利用 Group Relative Policy Optimization (GRPO) 在 AR‑Diffusion 统一多模态模型（UMM）上实现推理式文本到图像生成和自我反思式细化，无需额外的冷启动阶段。

**💡 创新点**

创新点在于：①引入 Decompositional Verifiable Reward（DVReward）通过 LLM 将用户请求拆解为可验证的原子问题并使用 MLLM 评估，提供稳定且细粒度的奖励；②加入 False‑Positive Rectification（FPR）防止奖励误导；③构建统一轨迹，端到端优化推理文本与图像生成。

**🔧 技术方法**

主要技术包括：GRPO 强化学习、AR‑Diffusion 统一多模态模型（Bagel）、LLM 分解器、MLLM 验证器（Qwen3‑VL‑30B‑A3B）、基于问题的信心评分、False‑Positive Rectification。

**📊 数据集**

训练数据采用 Primitive‑to‑Prompt 的生成策略，构造 19,500 条多难度层次的提示语与 1,024 条测试提示；评估数据集包括 GenEval、TIIF‑Bench、DPG‑Bench、WISE 以及图像编辑基准 GEdit。

**📈 对比分析**

与 SD3 Medium、FLUX.1 dev、Show‑o、JanusPro、Bagel 等模型对比，AlphaGRPO 在 TIIF‑Bench 取得 83.9%（比 Bagel 高 5.8%），在 GEdit 取得 7.08 分（比 Bagel 提升 0.52），并在 GenEval、DPG‑Bench、WISE 等基准上持续领先。

**⚠️ 局限性**

局限性：需大量算力进行多次 MLLM 评估，DVReward 依赖 LLM 的分解准确性，可能出现误拆或误评；FPR 与信心评分等机制在极端复杂或高度主观任务下的表现仍有限；模型整体仍受限于底层 UMM 的知识覆盖范围。

---

## 829. Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction

**arXiv ID:** 2605.12494 | [PDF](https://arxiv.org/pdf/2605.12494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 830. Routers Learn the Geometry of Their Experts: Geometric Coupling in Sparse Mixture-of-Experts

**arXiv ID:** 2605.12476 | [PDF](https://arxiv.org/pdf/2605.12476v1)

**作者:** Sagi Ahrac `[一作]` (Tel Aviv University), Mor Geva `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了稀疏专家混合模型（SMoE）中路由器与专家之间的几何耦合，并提出了基于在线 K‑Means 的无参数路由器；

**💡 创新点**

首次揭示路由器权重与被选专家输入权重共享相同输入方向的梯度更新，从而形成几何耦合；证明辅助负载平衡损失破坏该耦合；并通过不学习路由权重的 EMA 质心实现大部分路由功能；

**🔧 技术方法**

梯度理论分析、几何耦合模型、在线 K‑Means / EMA 质心路由、负载平衡损失对比实验；

**📊 数据集**

使用 OLMoE‑mix‑0924（C4‑en 与 Pile 组合的文本语料）进行训练，评估 C4‑en 与 Pile 验证集上的表现；

**📈 对比分析**

与标准辅助负载平衡、无负载平衡、Seq‑Aux 四种路由方案对比；以 perplexity 和 MaxVio 衡量，K‑Means 路由器无训练参数，MaxVio≈0.037，perplexity 仅略高（约 2.6%）但仍与 Loss‑Free 相近；

**⚠️ 局限性**

仅在单一 1B 模型规模上验证，缺乏更大规模/不同架构的泛化；仅关注门控激活，未探讨其他路径问题对耦合的影响。

---

## 831. Reward Hacking in Rubric-Based Reinforcement Learning

**arXiv ID:** 2605.12474 | [PDF](https://arxiv.org/pdf/2605.12474v1)

**作者:** Anas Mahmoud `[一作]` (Scale AI), Yunzhong He `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在基于 rubric 的强化学习（RL）中出现的奖励黑客（reward hacking）问题，区分了验证器误差和 rubric 设计缺陷两类源头，并提出了自我内部化间隙（self‑internalization gap）等无外部评审器诊断指标。

**💡 创新点**

创新点在于①构建了跨家族评审面板与训练验证器的对比框架，能够分离验证器错误与 rubric 设计缺陷；②系统识别了三类典型验证器失效模式；③提出无评审器的自我内部化间隙，能在训练过程中早期发现收益停止；④揭示即使验证器更强，基于 rubric 的奖励仍可导致整体质量下降。

**🔧 技术方法**

技术包括基于 rubric 的多维奖励聚合、Group Relative Policy Optimization（GRPO）、对比实验使用三位前沿模型的评审面板、误差模式聚类分析、以及基于策略对数概率的自我内部化间隙计算。

**📊 数据集**

数据集主要来自医学与科学领域的 12,519/1,391 训练/测试医疗提示、19,806/2,201 科学提示，配合 RubricHub 的 prompt‑specific rubrics；实验还使用 HealthBench 作为外部验证。

**📈 对比分析**

比较方法：将训练验证器（弱 GPT‑4o‑mini 与强 GPT‑OSS‑120B）与跨家族评审面板奖励对比；利用 exploitation rate 与 self‑gap 监测奖励黑客。结果显示弱验证器导致奖励黑客率升高、参考奖励停滞，强验证器显著降低黑客但仍有 15–30% 的误差；自我内部化间隙与参考奖励高度相关（r≈0.9）。

**⚠️ 局限性**

局限性包括：评审面板仍基于模型，可能与验证器共享错误；实验未覆盖多种随机种子；对 reward‑hacking 的因果机制仍未完全验证，需进一步的干预实验。

---

## 832. KV-Fold: One-Step KV-Cache Recurrence for Long-Context Inference

**arXiv ID:** 2605.12471 | [PDF](https://arxiv.org/pdf/2605.12471v1)

**作者:** Alireza Nadali `[一作]` (University of Colorado Boulder), Alvaro Velasquez `[通讯]` (University of Colorado Boulder)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出KV-Fold推理协议，将预训练transformer的KV缓存在chunk之间作为累加器，实现在不修改模型、不需训练的情况下进行长上下文推理。

**💡 创新点**

创新点在于将KV缓存当作递归状态，形成稳定的固定点，使得每一步的漂移不随链深度累积，而是先升后饱和；通过简单的左折运算实现无架构改动、无微调的长上下文推理。

**🔧 技术方法**

使用技术包括：chunked attention、连续RoPE位置编码、KV缓存递归、bfloat16与fp32精度对比、needle‑in‑a‑haystack检索基准、与StreamingLLM等基线对比、量化鲁棒性测试。

**📊 数据集**

实验数据集：PG‑19验证集用于评估NLL漂移；needle‑in‑a‑haystack基准（随机插入键值对）用于检索测试；使用的模型有Qwen2.5‑7B‑Instruct、Llama‑3.1‑8B‑Instruct等。

**📈 对比分析**

评估方法：对比全上下文注意力（上限）与独立chunk（下限）以及StreamingLLM；在128K上下文、链深511下，KV‑Fold保持100%检索准确率，峰值GPU内存35.6GB，推理时间171s；StreamingLLM内存仅16.6GB、速度7.5倍快，但检索在超过1024-token窗口后全部失效。

**⚠️ 局限性**

局限性：KV缓存随总上下文线性增长，导致显存需求随链深度增加；无法突破模型原生上下文窗口；量化等噪声会导致检索下降；不压缩KV状态意味着内存上限受限，需进一步研究压缩或位置扩展方法。

---

## 833. Elastic Attention Cores for Scalable Vision Transformers

**arXiv ID:** 2605.12491 | [PDF](https://arxiv.org/pdf/2605.12491v1)

**作者:** Alan Z. Song `[一作]` (Carnegie Mellon University), Andrew F. Luo `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种通过核心-外围稀疏注意力替代全局自注意力的视觉变压器架构VECA，实现线性时间复杂度并支持可弹性推理。

**💡 创新点**

创新点包括核心-外围结构的稀疏注意力、可训练的核心向量作为全局通信接口、通过嵌套dropout实现的弹性计算以及在无显式补丁间交互下学习高质量密集视觉表示。

**🔧 技术方法**

使用核心-外围注意力（block‑sparse）、2D轴向RoPE、嵌套dropout、知识蒸馏（对齐DINOv3）、线性分类头等技术。

**📊 数据集**

使用Objects365无标签图像进行蒸馏，评估数据集包括ImageNet‑1k、ImageNetV2、ImageNet‑ReaL、Places365、Food101、SUN397、Oxford‑Pets、CUB‑200；密集任务评估使用PASCAL VOC、PASCAL Context、ADE20K、COCO‑Object、COCO‑Stuff、Cityscapes；深度估计使用NYUv2、KITTI。

**📈 对比分析**

通过对比DINOv3教师的线性probe，VECA在分类上取得81.93% top‑1 vs 83.56%，在分割上mIoU 57.46 vs 57.74，且核心数从64降至8仅损失约4%性能，证明了线性注意力的竞争力。

**⚠️ 局限性**

限制在于核心容量固定、预算手动选择、缺乏内容感知核心分配、未系统分析核心冗余、未在目标检测、实例分割、视频等任务上验证。

---

## 834. CausalCine: Real-Time Autoregressive Generation for Multi-Shot Video Narratives

**arXiv ID:** 2605.12496 | [PDF](https://arxiv.org/pdf/2605.12496v1)

**作者:** Yihao Meng `[一作]` (HKUST), Huamin Qu `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为CausalCine的交互式自回归多镜头视频生成框架。

**💡 创新点**

创新点在于先在长片段多镜头视频上进行因果自回归调优，再用内容感知记忆路由(CAMR)和分步蒸馏实现实时生成。

**🔧 技术方法**

使用了流匹配扩散模型、因果自回归训练、内容感知记忆路由、Block‑Relative RoPE、分步蒸馏（DMD）以及对抗正则化。

**📊 数据集**

基准数据集为100个多镜头Prompt的 Gemini 2.5 Pro 生成的测试集，以及公开的约 15 秒（≈241 帧）的长视频数据。

**📈 对比分析**

与 Self‑Forcing、Infinity‑RoPE、LongLive、MemFlow、ShotStream 等自回归基线以及双向多镜头模型对比，CausalCine 在文本对齐、镜头切换准确度、身份保持等指标上均优于自回归基线，并且在视觉质量上接近双向模型，同时保持实时推理速度。

**⚠️ 局限性**

局限性包括对极长持续时间或高帧率的实时性尚未完全验证，且模型规模较大，需要多块 GPU 进行高效推理。

---

## 835. ToolCUA: Towards Optimal GUI-Tool Path Orchestration for Computer Use Agents

**arXiv ID:** 2605.12481 | [PDF](https://arxiv.org/pdf/2605.12481v1)

**作者:** Xuhao Hu `[一作]` (Tongyi Lab Alibaba Group), Jieping Ye `[通讯]` (Tongyi Lab Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了混合 GUI‑Tool 行为的计算机使用代理 ToolCUA，解决了工具调用与 GUI 操作切换的困惑

**💡 创新点**

创新点在于（1）通过工具合成与离线 GUI 数据扩增生成高质量交错 GUI‑Tool 轨迹；（2）采用分阶段训练：基于交错数据的工具启动 RFT 与关键切换点单轮 RL；（3）在线强化学习加入工具效率路径奖励，促使路径更短、更合适

**🔧 技术方法**

技术包括多模态 LLM（Qwen3‑VL‑8B）、自监督微调、单轮 RL（GRPO）、多轮 RL、工具效率奖励、工具库构造、轨迹扩增、下一步状态对齐等

**📊 数据集**

使用公开的 OSWorld、open‑source GUI 轨迹数据集，并结合合成工具与 OSWorld‑MCP 任务；训练数据包括 𝒟_all 与 𝒟_critical

**📈 对比分析**

与多种基线（Qwen3‑VL‑8B、Gemini‑3.1‑Pro、Claude‑4‑Sonnet、GUI‑Owl、EvoCUA）和纯 GUI 训练对比，ToolCUA 在 OSWorld‑MCP 上达到 46.85% 任务成功率，比基线提升约 66%，并在未见的 Linux multi_apps 与 WindowsAgentArena 上也表现出良好迁移

**⚠️ 局限性**

局限在于仍需人工验证合成工具的可靠性；奖励设计依赖任务标签；在极端多工具场景下工具调用频率仍低；对工具不兼容或不稳定的应用场景适应性待进一步提升

---

## 836. OmniNFT: Modality-wise Omni Diffusion Reinforcement for Joint Audio-Video Generation

**arXiv ID:** 2605.12480 | [PDF](https://arxiv.org/pdf/2605.12480v1)

**作者:** Guohui Zhang `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

改进联合音视频生成模型，利用强化学习对预训练的 Diffusion 模型进行后期微调，以提升音频与视频的质量、跨模态对齐与同步。

**💡 创新点**

三大创新：①模态感知优势路由——将视频、音频与同步奖励分别分配给对应分支；②层级梯度手术——在浅层音频层抑制视频梯度泄漏，保护音频生成；③基于区域的损失加权——利用视频→音频注意力地图突出关键同步区域，实现细粒度信用分配。

**🔧 技术方法**

技术栈包括在线扩散强化学习框架 OmniNFT、DiffusionNFT 的正负策略对比损失、GRPO 的优势归一化、Transformer 的跨模态注意力、奖励模型 VideoAlign、HPSv3、Audiobox、CLAP、Desync 等。

**📊 数据集**

实验数据集：JavisBench 与 VBench，使用 LTX‑2（19B）作为 backbone。

**📈 对比分析**

与 LTX‑2 与 GDPO 等基线对比，OmniNFT 在视觉质量（VQ+63%）、音频质量（AQ+10%）、文本‑模态一致性以及同步（DeSync-52%）等四大指标上均取得显著提升。

**⚠️ 局限性**

局限性：文本‑视频语义一致性仍未显著提升；对多说话人或复杂场景的鲁棒性不足；需要精细调节奖励与权重，训练成本相对较高。

---

## 837. Solve the Loop: Attractor Models for Language and Reasoning

**arXiv ID:** 2605.12466 | [PDF](https://arxiv.org/pdf/2605.12466v1)

**作者:** Jacob Fein-Ashley `[一作]` (University of Southern California), Paria Rashidinejad `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Attractor Models：先用 Transformer 背包产生初始输出嵌入，再通过吸引子模块求解固定点完成迭代细化，并利用隐式梯度实现 O(1) 内存训练。

**💡 创新点**

创新点在于把递归细化转化为输出嵌入空间的固定点求解，使用背包式初始嵌入、隐式梯度一阶近似以及 Anderson 加速器，导致出现“平衡内化”现象，使得推理时几乎无需迭代。

**🔧 技术方法**

核心技术包括 Deep Equilibrium Model (DEQ) 思想、吸引子网络、Anderson 固定点求解器、隐式梯度（Implicit Function Theorem）与一阶近似、以及梯度惩罚。

**📊 数据集**

使用 FineWeb-Edu 进行大规模语言建模，对比 Lambada、CORE、CORE-Extended 等下游基准；在推理任务上使用 Sudoku-Extreme 与 Maze-Hard 这两类小规模推理数据集。

**📈 对比分析**

与标准 Transformer、循环 LM Parcae 以及前沿 LLM（DeepSeek R1、Claude、o3-mini）对比，Attractor Models 在 140M/370M/770M 参数下均在验证 PPL、Lambada PPL、CORE 等指标上优于 Baseline，训练 FLOPs 降低 25–31%；在 27M Tiny 模型下，在 Sudoku-Extreme 与 Maze-Hard 上达到 91.4%/93.1% 的准确率，远超前沿模型与小型递归模型。

**⚠️ 局限性**

局限性包括需要手动设定收敛容差与最大迭代步数；在极大模型或更复杂任务中固定点求解可能收敛缓慢；对背包式初始嵌入的依赖在容量不足时会导致不收敛；在小样本推理场景中，一阶近似的隐式梯度可能不足以充分训练。

---

## 838. High-arity Sample Compression

**arXiv ID:** 2605.12465 | [PDF](https://arxiv.org/pdf/2605.12465v1)

**作者:** Leonardo N. Coregliano `[一作]`, William Opich `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `01e19694-9125-4cf8-82ff-580f56a0fdb6`

**🎯 论文内容**

本文研究了高元学习理论中的样本压缩方案，证明了非平凡质量的高元样本压缩方案的存在性与高元PAC可学习性之间的关系。

**💡 创新点**

创新点在于引入了高元样本压缩的概念，并证明了在部分和非部分设置下，高元样本压缩可以推导出高元PAC可学习性。

**🔧 技术方法**

使用了马尔可夫不等式（Azuma's Inequality）来处理高元样本压缩的证明，克服了高元标签不再独立同分布的挑战。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了高元学习理论的相关背景和先前的研究成果。

**📈 对比分析**

通过与现有的高元学习理论结果进行比较，展示了高元样本压缩与高元PAC可学习性之间的等价性，性能分析表明在特定条件下，样本压缩可以有效实现学习。

**⚠️ 局限性**

限制在于高元学习理论的复杂性，尤其是在非部分设置下，标签的处理和样本压缩的实现面临更大的挑战。

---

## 839. Covering Human Action Space for Computer Use: Data Synthesis and Benchmark

**arXiv ID:** 2605.12501 | [PDF](https://arxiv.org/pdf/2605.12501v1)

**作者:** Miaosen Zhang `[一作]` (Southeast University), Baining Guo `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对计算机使用代理（CUA）在复杂图形界面交互中的动作定位问题，提出了新型多模态基准CUActSpot、基于渲染的自动合成数据管线，并在此基础上训练出Phi‑Ground‑Any‑4B模型。

**💡 创新点**

创新点在于：①将基准从传统的单一点击扩展为包含文本、表格、画布、自然图像等五大模态，并支持点击、拖拽、绘制等多种交互；②构建可自动渲染并提取坐标的多模态数据合成管线，利用LLM生成指令与动作轨迹；③通过实验验证任务与模态多样性（variety scaling）比单纯增大数据量更能提升模型性能。

**🔧 技术方法**

技术手段包括：使用Phi‑3.5‑VL（4B参数）作为视觉语言模型基体，采用OpenAI的o3、GPT‑4o等LLM生成任务与动作；利用Playwright、PyQt5、matplotlib、SAM等渲染工具生成屏幕截图并提取精确坐标；训练过程中采用大量合成样本（30M GUI、5M其他模态）进行预训练/微调；评估采用自定义区域匹配规则。

**📊 数据集**

数据集方面，训练集由30M GUI合成样本及5M/模态合成样本组成；基准CUActSpot包含206条手工标注的复杂交互样本；此外在实验中还与ScreenSpot‑Pro、UI‑Vision、OSWorld等公开基准进行对比。

**📈 对比分析**

评估方法是对各模型在CUActSpot、ScreenSpot‑Pro、UI‑Vision、OSWorld等基准上进行准确率/成功率评估。Phi‑Ground‑Any‑4B在CUActSpot上实现了超过32B以下所有开源模型的最佳成绩；在ScreenSpot‑Pro和UI‑Vision上表现稍弱，但通过应用Phi‑Ground特定软件数据的微调后可明显提升；在OSWorld的单步动作定位任务中，表现与领先模型相当。

**⚠️ 局限性**

局限性在于：①CUActSpot基准样本量有限，仅覆盖单步交互，缺乏长时序和状态依赖的工作流；②合成数据虽然多样，但与真实用户场景的分布差距仍大，可能影响泛化；③模型在复杂交互上仍存在误差，尤其是涉及多步协调与细粒度绘制的情况。

---

## 840. SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture

**arXiv ID:** 2605.12500 | [PDF](https://arxiv.org/pdf/2605.12500v1)

**作者:** Haiwen Diao `[一作]`, Dahua Lin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SenseNova-U1，一种端到端原生统一多模态模型，直接使用像素和文本输入，无需预训练视觉编码器或VAE，能够同时完成感知、推理、生成和交互任务。

**💡 创新点**

创新点在于：①原生无损视觉接口（轻量化卷积编码+MLP解码），①直接对像素进行流匹配生成；②混合Transformer（MoT）架构实现理解与生成的同一网络，最小化任务干扰并提升规模效率；③多阶段联合训练（warmup、生成预训练、联合中训练、SFT、RL和蒸馏），实现从认知到创作的完整能力。

**🔧 技术方法**

主要技术包括：近损失less视觉接口、混合Transformer、像素级流匹配损失、时间-噪声自回归条件、分类器无监督引导、混合专家（MoE）扩展、动态噪声尺度调度、混合奖励的RL、分布式蒸馏与CFG，并采用分离推理架构（LightLLM + LightX2V）。

**📊 数据集**

使用多样化公开与内部数据集：大规模图文对、长文本、图文混合文档、图像编辑、交互式多模态推理、结构化信息图、科学推理、机器人视觉动作等；训练集覆盖从 256×256 到 2048×2048 像素，文本与图像多语言（英中）并行。

**📈 对比分析**

与 Qwen3VL、Qwen3.5、FLUX 等基准模型比较，SenseNova-U1 在多模态理解、视觉推理、OCR、空间感知、文本推理、代理任务、文本到图像、信息图、复杂编辑、交互式生成等各类基准上均实现或接近最先进水平，尤其在无损像素级生成、长文本渲染、信息图结构化生成和交互式多模态推理上表现突出。

**⚠️ 局限性**

局限性包括：在专门化的图像编辑、极端复杂的混合编辑任务上仍与顶级专用编辑模型有一定差距；部分推理式任务的性能受限于训练数据和提示工程；在高分辨率（>2048×2048）与细粒度对齐任务上仍需进一步优化；以及当前的RL和蒸馏阶段需要更多资源与调优。

---

## 841. LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues

**arXiv ID:** 2605.12493 | [PDF](https://arxiv.org/pdf/2605.12493v1)

**作者:** Di Wu `[一作]` (University of California, Los Angeles), Kai-Wei Chang `[通讯]` (University of California, Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 LongMemEval‑V2 基准，用于评估网页代理在长期交互中对环境知识的记忆能力，并基于此设计了两种记忆模块；

**💡 创新点**

创新点包括：① 将五类记忆能力（静态回忆、动态追踪、工作流知识、环境陷阱、前提意识）纳入评估；② 构建覆盖 10⁸ 词元的大规模轨迹“haystack”；③ 引入文件管理式编码代理记忆（AgentRunbook‑C）以提升准确率‑延迟权衡；

**🔧 技术方法**

技术手段涵盖 RAG+多知识池、LLM 控制器（Qwen3.5‑9B）、编码代理（Codex / GPT‑5.4‑mini）与工作流脚本、以及 200K token 截断与 Qwen3.5‑9B 阅读器；

**📊 数据集**

数据来源于 WebArena、WorkArena 与 WorkArena++ 的 599+941 条轨迹，手工生成 451 个问题，形成 LME‑V2‑Small（100 轨迹）与 LME‑V2‑Medium（≈500 轨迹）两级；

**📈 对比分析**

在准确率与查询延迟对比实验中，AgentRunbook‑C 在小/中等层级分别达 74.9%/70.1% 的准确率，明显优于传统 RAG（≈58%）和纯 Codex（≈69%），同时在延迟上实现更佳权衡；

**⚠️ 局限性**

局限性在于：① 仍有显著误差空间，未完全覆盖所有环境细节；② 编码代理方法延迟相对较高；③ 评估依赖人工标注的轨迹与问题，难以快速扩展到更广泛场景。

---

## 842. Pion: A Spectrum-Preserving Optimizer via Orthogonal Equivalence Transformation

**arXiv ID:** 2605.12492 | [PDF](https://arxiv.org/pdf/2605.12492v1)

**作者:** Kexuan Shi `[一作]` (Chinese University of Hong Kong), Weiyang Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种谱保持型优化器Pion，利用正交等价变换在训练大型语言模型时保持权重矩阵的奇异值谱不变，提升训练稳定性；

**💡 创新点**

创新点在于：①不通过重参数化实现谱保持，而是直接在权重上进行左右正交变换；②结合RMS尺度一致、Lie代数动量、交替更新及二阶矩阵指数逼近，形成高效、稳定的优化算法；

**🔧 技术方法**

核心技术包括：正交群优化、Lie代数梯度投影、矩阵指数的Cayley变换/幂级数近似、RMS归一化、动量（一阶/二阶）与交替更新策略；

**📊 数据集**

使用的数据集涵盖LLM预训练（C4、T5-tokenizer、256长度）、后训练（Supervised finetune on MetaMathQA、Magicoder-Evol-Instruct-110K；RLVR on DeepMath）以及多种基准（ARC, Hella, SciQ, TriviaQA, Winogrande, PIQA, etc.）；

**📈 对比分析**

对比方法包括AdamW、Muon及其变体，实验显示Pion在预训练、微调和RLVR任务中均达到或略优于基线，在训练稳定性、验证损失、表达能力和泛化性能上表现更佳；

**⚠️ 局限性**

局限性：算法相对复杂，计算与内存开销高于AdamW；在极大模型规模下的参数迁移与自适应性尚需进一步验证；部分设计选择（如RMS常数、交替更新频率）依赖经验调优。

---

## 843. Task-Adaptive Embedding Refinement via Test-time LLM Guidance

**arXiv ID:** 2605.12487 | [PDF](https://arxiv.org/pdf/2605.12487v1)

**作者:** Ariel Gera `[一作]` (IBM Research), Assaf Toledo `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在测试时利用生成式 LLM 对查询嵌入进行微调，以提升嵌入模型在零样本检索和二分类任务上的性能。

**💡 创新点**

创新点在于提出一种 LLM‑指导的查询微调框架，能够在不修改模型权重的情况下实时调整查询向量，从而显著提升多任务检索与二分类的 MAP。

**🔧 技术方法**

采用密集检索嵌入模型、生成式 LLM 作为教师提供相关性得分，并通过基于 KL 散度的梯度优化（Adam 优化器）更新查询嵌入。

**📊 数据集**

实验数据集包括 arXiv 文献检索、意图识别、关键点匹配（KPA 共享任务）以及 FollowIR 的细粒度查询指令等公开基准。

**📈 对比分析**

与原始查询嵌入及仅使用 LLM 重排序进行对比，实验表明在 MAP 上平均提升约 12%，在各个任务上均实现 9%–25% 的显著改进。

**⚠️ 局限性**

局限性包括对教师 LLM 反馈质量的依赖，可能导致偏差；top‑K 反馈集合的选择可能忽略更有信息量的文档；以及在极度类别不平衡或不同领域的适用性尚需进一步验证。

---

## 844. Learning, Fast and Slow: Towards LLMs That Adapt Continually

**arXiv ID:** 2605.12484 | [PDF](https://arxiv.org/pdf/2605.12484v1)

**作者:** Rishabh Tiwari `[一作]` (University Of California Berkeley), Devvrit Khatri `[通讯]` (University Of Texas Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种快慢学习框架，联合优化LLM模型参数与文本上下文，在可验证奖励的RL设置下实现更高的样本效率与更好的持续学习能力。

**💡 创新点**

创新点在于将任务特定信息通过快速文本上下文学习，减轻参数更新的灾难性遗忘，保持模型可塑性，并通过快慢权重协同进化提升性能。

**🔧 技术方法**

使用了RL with verifiable rewards（ScaleRL/CISPO）进行慢权重更新，GEPA进行快速文本提示进化，并在两者之间交替更新。

**📊 数据集**

实验主要在Math、CodeIO、HoVer-hard等推理任务数据集上完成。

**📈 对比分析**

与仅参数RL、仅提示GEPA、提示蒸馏等基线对比，Fast‑Slow Training在相同奖励下实现约3倍的样本效率，达到更高的性能上限，且KL漂移更小、熵保持更高。

**⚠️ 局限性**

局限性包括对高效快慢优化器的依赖、计算成本较高、未充分探讨不同提示或权重优化器组合，以及未充分利用轨迹复用等。

---

## 845. MEME: Multi-entity & Evolving Memory Evaluation

**arXiv ID:** 2605.12477 | [PDF](https://arxiv.org/pdf/2605.12477v1)

**作者:** Seokwon Jung `[一作]` (KAIST), Seong Joon Oh `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MEME基准，用于评估LLM代理在跨多会话、多实体、动态环境中的记忆和推理能力。

**💡 创新点**

创新点在于引入Cascade、Absence和Deletion三类依赖推理任务，构建可验证的DAG知识图谱生成语料，填补了现有基准缺失的多实体与演化维度交叉评估。

**🔧 技术方法**

采用LLM内置记忆、向量检索、图检索和文件式工具调用等三大记忆范式，配合gpt-4.1-mini/Claude Opus等大型语言模型进行内存编码、维护与检索。

**📊 数据集**

使用两个手工构造的DAG知识图谱（个人生活与软件项目）生成的100个约35K-token的对话集，包含694道评估问题。

**📈 对比分析**

与传统基准相比，MEME显示所有主流记忆系统在Cascade与Absence任务上平均仅达3%/1%的准确率，成本在内存系统与直接in-context查询之间权衡，最高的MD-flat+Opus方案虽能提升至70%/60%但成本约70倍。

**⚠️ 局限性**

局限包括仅两域手工图谱、LLM生成对话、有限样本规模、部分实验覆盖子集、未检验隐式依赖等，未来需更广泛的领域、真实用户数据及更高效的依赖传播架构。

---

## 846. EgoForce: Forearm-Guided Camera-Space 3D Hand Pose from a Monocular Egocentric Camera

**arXiv ID:** 2605.12498 | [PDF](https://arxiv.org/pdf/2605.12498v1)

**作者:** Christen Millerdurai `[一作]` (Deutsches Forschungszentrum für Künstliche Intelligenz), Alain Pagani `[通讯]` (Deutsches Forschungszentrum für Künstliche Intelligenz)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出EgoForce框架，利用单目自我摄像头实现绝对摄像机空间3D手部姿态重建，兼顾手腕与前臂信息。

**💡 创新点**

创新点包括：①使用前臂上下文缓解单目深度尺度不确定；②统一的前臂-手部Transformer模型HALO；③针对任意摄像机模型的射线空间闭式求解器，支持鱼眼、透视、畸变宽视角。

**🔧 技术方法**

采用MANO与自定义FARM模型、可微前臂表示、Transformer编码器+查询解码器、射线空间求解、Crop Intrinsics Token、可变先验完成前臂缺失、Kalman滤波等技术。

**📊 数据集**

训练使用Re:InterHand、HandCO、H2O、ARCTIC、HO3D、HOT3D共3.67M RGB图像，评估在H2O、HOT3D、ARCTIC和HO3D等数据集。

**📈 对比分析**

与MobRecon、HandOccNet、HandDGP等基线对比，EgoForce在ARCTIC、HOT3D、H2O的CS‑MJE和PS‑MJE分别提升约20–30%，时间一致性误差降低约20%，并实现约14FPS实时推理。

**⚠️ 局限性**

局限在于需标注的3D训练数据，无法直接利用大规模2D数据；对相机内参敏感，畸变估计误差会影响结果；对前臂完全遮挡时仍需先验估计。

---

## 847. Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for Language-Model Post-Training

**arXiv ID:** 2605.12483 | [PDF](https://arxiv.org/pdf/2605.12483v1)

**作者:** Yuanda Xu `[一作]` (Princeton University), Alborz Geramifard `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究如何分配稀缺标注训练数据，提出教师优先分配的原则。

**💡 创新点**

创新点在于将稀疏奖励与稠密教师监督放在奖励密度轴上，设计两阶段桥接并证明教师侧稀疏奖励更有效。

**🔧 技术方法**

使用GRPO、OPD、FKL温度、RL、SFT等技术，构建两阶段桥接。

**📊 数据集**

采用可验证的数学数据集MATH、AIME 2024/2025以及DAPO-Math-17K。

**📈 对比分析**

与直接学生GRPO、教师样本SFT、OPD单阶段等对照实验，教师侧分配+桥接在1.7B学生上MATH提升3-4个百分点，AIME提升4-5个百分点。

**⚠️ 局限性**

局限在于仅在小规模学生（1.7B/8B）和特定数学任务上验证，未检验更大规模或其他开放式任务。

---

## 848. An Improved Lower Bound on Support Size of Capacity-Achieving Inputs for the Binomial Channel: Extended version

**arXiv ID:** 2605.12472 | [PDF](https://arxiv.org/pdf/2605.12472v1)

**作者:** Mohammadamin Baniasadi `[一作]` (University of California, Davis), Alex Dytso `[通讯]` (Qualcomm Flarion Technologies)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

对二项信道的容量实现输入分布进行研究，给出了其支持集合大小的新的下界。

**💡 创新点**

将已知的 √n 下界提升到 √(n·loglog n)，并提供了可实现的显式常数。

**🔧 技术方法**

利用新的容量上下界、β‑二项输出的渐近最优性、χ² 散度逼近下界、最佳逼近理论、条件期望算子与正交多项式的谱分析等技术。

**📊 数据集**

本文为理论分析性研究，无使用数据集。

**📈 对比分析**

通过严格的数学推导与对比，证明了在大 n 时支持大小至少为 Ω(√(n·loglog n))，相较原来的 √n 下界显著改进。

**⚠️ 局限性**

仍存在与上界 n/2 之间的巨大差距，常数未优化，结论仅在渐近意义下成立，精确增长速率未知。

---

