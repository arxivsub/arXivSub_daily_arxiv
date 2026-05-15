# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-15 | 今日论文总数: 691

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. XAI and Statistical Analysis for Reliable Intrusion Detection in the UAVIDS-2025 Dataset: From Tree to Hybrid and Tabular DNN Ensembles

**arXiv ID:** 2605.13922 | [PDF](https://arxiv.org/pdf/2605.13922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 2. Finite Sample Bounds for Learning with Score Matching

**arXiv ID:** 2605.14168 | [PDF](https://arxiv.org/pdf/2605.14168v1)

**作者:** Devin Smedira `[一作]` (Massachusetts Institute of Technology), Andrey Y. Lokhov `[通讯]` (Los Alamos National Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文研究使用分数匹配学习具有无界支持的连续指数族分布的结构，并给出了非渐近的样本复杂度分析。

**💡 创新点**

创新点在于首次为多项式指数族结构学习提供了多项式样本复杂度界限，填补了以往仅有渐近结果的空白。

**🔧 技术方法**

主要技术包括分数匹配（score matching）、曲率分析以及针对族结构和模型结构的恢复算法。

**📊 数据集**

文中未公开具体数据集，推测作者可能使用仿真数据来验证理论。

**📈 对比分析**

论文未给出与其他方法的实验比较，重点放在理论证明与样本复杂度上。

**⚠️ 局限性**

局限性：仅针对多项式指数族；对可导性、曲率等假设要求较强；缺乏实证评估和实验结果。

---

## 3. Linear-Time T-Gate Optimization via Random Abstraction

**arXiv ID:** 2605.13929 | [PDF](https://arxiv.org/pdf/2605.13929v1)

**作者:** Aws Albarghouthi `[一作]` `[通讯]` (University of Wisconsin-Madison), Aws Albarghouthi (University of Wisconsin-Madison)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于随机化静态分析的线性时间 T 门折叠优化算法

**💡 创新点**

利用随机位串抽象每个量子比特的相位偏移，显式在每个门上传播常数时间的位运算，随后用哈希表快速判定可折叠的旋转，从而实现概率安全、线性复杂度的优化

**🔧 技术方法**

随机位串抽象、随机化静态分析、哈希表查找、单向量化的位运算

**📊 数据集**

Feynman 标准基准集（数十至数十万门）与 Cobble 基准集（数千至数千万门）

**📈 对比分析**

与 VOQC、QuiZX、Feynman、FastTODD 等现有工具比较，T 门数减少幅度与它们相当，但执行时间快四到五个数量级；在百万级门甚至数亿门规模下仍能在几秒到两分钟内完成优化，且误差概率可控制在 10⁻²⁰ 以下

**⚠️ 局限性**

由于 Hadamard 门被视为完全随机化，分析在某些可折叠情形下失效；该方法无法捕获更复杂的相位重排（如多条路径的干涉）及循环/控制流结构，导致优化不完整

---

## 4. Polar probe linearly decodes semantic structures from LLMs

**arXiv ID:** 2605.14125 | [PDF](https://arxiv.org/pdf/2605.14125v1)

**作者:** Pablo J. Diego-Simón `[一作]` (École Normale Supérieure), Jean-Rémi King `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在激活空间中使用距离与方向编码概念关系，构建复杂语义结构。

**💡 创新点**

提出极坐标探针（Polar Probe），将语义图的存在性映射为距离、关系类型映射为方向，首次将该几何绑定原理扩展至多种语义域。

**🔧 技术方法**

使用极坐标探针的线性变换，结合结构损失与角度损失进行训练，评估Spearman相关性，并进行因果干预验证。

**📊 数据集**

构造五个合成语义域数据集（算术变量大小、二维空间布局、家谱、地铁网络、社交角色），每个域包含多张图并多次文本描述。

**📈 对比分析**

通过与随机初始化模型、不同层级、不同模型规模对比，极坐标探针在中层可达ρ≈0.8（关系存在）和ρ≈0.5–0.7（关系类型），随预训练步数和模型规模提升；在新实体/关系上的泛化仍显著优于随机基线。

**⚠️ 局限性**

仅适用于可在欧氏空间极坐标编码的图，对非欧氏或非可交换、多对多关系（如家谱、地铁）性能受限，且随着实体数增加、图复杂度提升，解码效果下降。

---

## 5. Mistletoe: Stealthy Acceleration-Collapse Attacks on Speculative Decoding

**arXiv ID:** 2605.14005 | [PDF](https://arxiv.org/pdf/2605.14005v1)

**作者:** Shuoyang Sun `[一作]` (Harbin Institute of Technology), Bin Chen `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 35945 | [OpenAlex ID](https://openalex.org/A5100427314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对speculative decoding的加速崩溃攻击Mistletoe，能在不改变最终输出的前提下通过降低平均接受长度τ来消除加速优势。

**💡 创新点**

首次发现并利用draft‑target匹配缺陷，设计以目标侧草稿词惊讶度为攻击目标、KL约束为语义保持、并使用null‑space投影与KL阈值过滤的多目标优化方法，实现在保持语义的同时压制草稿词可接受性。

**🔧 技术方法**

使用目标词惊讶度、KL散度约束、null‑space投影优化、离散后缀搜索及梯度引导的Suffix搜索，结合Medusa、Hydra、EAGLE系列等多种speculative decoding实现。

**📊 数据集**

在MT‑Bench、HumanEval、GSM8K三大基准上进行评估，覆盖对话、代码生成和数学推理。

**📈 对比分析**

与多种speculative decoding方法及基准进行对比，实验表明Mistletoe将speed‑up降低1.89–2.20×、平均接受长度τ降低0.99–1.21，且保持PPL和重复率不升高，证明攻击有效。

**⚠️ 局限性**

仅在已训练好的target与draft模型上有效，需白盒梯度信息，且在加速较弱的场景下效果相对有限，未考虑对抗训练或鲁棒改进的防御。

---

## 6. DSTAN-Med: Dual-Channel Spatiotemporal Attention with Physiological Plausibility Filtering for False Data Injection Attack Detection in IoT-Based Medical Devices

**arXiv ID:** 2605.14165 | [PDF](https://arxiv.org/pdf/2605.14165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 7. PanoPlane: Plane-Aware Panoramic Completion for Sparse-View Indoor 3D Gaussian Splatting

**arXiv ID:** 2605.14135 | [PDF](https://arxiv.org/pdf/2605.14135v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 8. Know When To Fold 'Em: Token-Efficient LLM Synthetic Data Generation via Multi-Stage In-Flight Rejection

**arXiv ID:** 2605.14062 | [PDF](https://arxiv.org/pdf/2605.14062v1)

**作者:** Anjir Ahmed Chowdhury `[一作]` (University of Houston), Feng Yan `[通讯]` (University of Houston)

**通讯引用:** 181008 | [OpenAlex ID](https://openalex.org/A5100384245)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了多阶段飞行拒绝（MSIFR）框架，在生成过程中提前检测并终止低质量样本，减少无用token消耗。

**💡 创新点**

创新点在于将生成过程拆分为若干阶段，使用快速规则校验器实现中途拒绝，且理论上证明早期拒绝不会偏移期望效用，并且任何非平凡的丢弃策略都能降低token消耗。

**🔧 技术方法**

主要技术包括基于规则的算术一致性、幻觉模式与格式违规检测，以及对生成过程的序列决策建模。

**📊 数据集**

实验使用五个指令微调的大语言模型和七个推理基准数据集进行评估。

**📈 对比分析**

与传统全生成+后处理方式相比，MSIFR单独使用时可降低11%–77%的token消耗，结合早退出方法可达78.2%，且在保持甚至提升评估准确率的同时显著提高效率。

**⚠️ 局限性**

局限性包括依赖规则检验器的覆盖范围，可能无法捕捉所有质量问题，以及对极其长或复杂生成场景的适用性尚需进一步验证。

---

## 9. EvolveMem:Self-Evolving Memory Architecture via AutoResearch for LLM Agents

**arXiv ID:** 2605.13941 | [PDF](https://arxiv.org/pdf/2605.13941v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 10. SPIN: Structural LLM Planning via Iterative Navigation for Industrial Tasks

**arXiv ID:** 2605.14051 | [PDF](https://arxiv.org/pdf/2605.14051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 11. Conditional Attribute Estimation with Autoregressive Sequence Models

**arXiv ID:** 2605.14004 | [PDF](https://arxiv.org/pdf/2605.14004v1)

**作者:** Erica Stutz `[一作]` (Yale University), Andrew J. Loza `[通讯]` (Yale University)

**通讯引用:** 1375 | [OpenAlex ID](https://openalex.org/A5007346239)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Conditional Attribute Transformers (CAT)，在单个前向传递中同时估计下一词概率和每个候选下一词对应的序列属性值，以实现信用分配、反事实分析和可控生成。

**💡 创新点**

创新点是将下一词预测与条件属性预测联合建模，利用分支头共享隐藏表示，并在不需蒙特卡罗模拟的情况下高效估计属性概率。

**🔧 技术方法**

技术上采用基于Transformer的分支架构（如nanoGPT改进），联合交叉熵与属性损失，使用λ权重平衡，支持二元、多项式与数值属性。

**📊 数据集**

数据集包括Key-to-Door（稀疏奖励RL）、Amazon Reviews（多分类属性）和PhysioNet Sepsis（医学序列二元/连续属性）。

**📈 对比分析**

与行为克隆、Q-learning、Decision Transformers、Director、PPLM等基线对比，CAT在Key-to-Door中几乎完胜，Amazon Reviews中部分属性预测精度提升至10^8×速度，且在文本生成中比同类模型更高流畅性与准确率。

**⚠️ 局限性**

限制在于仅支持离散属性的条件概率估计，单步贪心策略不保证全局最优，且在连续动作空间和更复杂因果推断场景下仍需扩展。

---

## 12. ARES-LSHADE: Autoresearch-Enhanced LSHADE with Memetic Polish for the GNBG Benchmark

**arXiv ID:** 2605.13877 | [PDF](https://arxiv.org/pdf/2605.13877v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 13. Matrix-Space Reinforcement Learning for Reusing Local Transition Geometry

**arXiv ID:** 2605.14304 | [PDF](https://arxiv.org/pdf/2605.14304v1)

**作者:** Zuyuan Zhang `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6510 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种将轨迹段映射为正定矩阵描述符的几何抽象，并基于此实现矩阵空间强化学习（MSRL），用于跨任务的组合泛化与价值迁移。

**💡 创新点**

创新点包括：①将一阶、二阶统计信息聚合为正定矩阵，形成可加、最小充分的轨迹描述符；②证明该描述符的完整性、可加性与最小充分性；③利用矩阵条件化价值函数的局部线性性质，实现源任务价值的预训练和迁移；④设计阻塞过滤器识别可实现的矩阵组合，避免无效组合。

**🔧 技术方法**

使用的技术包括：轨迹段矩阵映射（lift + outer‑product 聚合）、矩阵条件化价值函数与局部C¹近似、阻塞评分学习、基于库的矩阵增量组合、以及与传统离线强化学习算法（PPO、SAC、TD‑MPC、DreamerV3等）耦合。

**📊 数据集**

数据集：源端轻量环境（GraphMotif、MicroGrid、PointRooms、Dubins‑Reacher）用于可解释性诊断与预训练；目标端复杂控制环境（AntMaze、Reacher/Pusher、Walker2d、Hopper、Humanoid 等）用于评估迁移效果。

**📈 对比分析**

通过与从零开始、源预训练、以及多种通用值编码（Successor Features、UVFA、DBC、Traj2Value 等）基线的比较，MSRL 在有限预算下平均 AUC 达到 0.73，显著高于最强基线 0.57，并且源预训练版本进一步提升到 0.73，体现出快速收敛与较高最终表现。

**⚠️ 局限性**

局限性：需要源与目标共享对齐的升维映射；描述符仅保留低阶统计，无法完整编码时间顺序；矩阵条件化价值的线性近似仅在局部有效；阻塞检测依赖学习代理，理论上只能精确识别可实现矩阵集合。

---

## 14. To See is Not to Learn: Protecting Multimodal Data from Unauthorized Fine-Tuning of Large Vision-Language Model

**arXiv ID:** 2605.14291 | [PDF](https://arxiv.org/pdf/2605.14291v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 15. Artificial Intelligence-Assistant Cardiotocography: Unified Model for Signal Reconstruction, Fetal Heart Rate Analysis, and Variability Assessment

**arXiv ID:** 2605.14242 | [PDF](https://arxiv.org/pdf/2605.14242v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 16. Policy Optimization in Hybrid Discrete-Continuous Action Spaces via Mixed Gradients

**arXiv ID:** 2605.14297 | [PDF](https://arxiv.org/pdf/2605.14297v1)

**作者:** Matias Alvo `[一作]` (Columbia University), Yash Kanoria `[通讯]` (Columbia University)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5000266593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出 Hybrid Policy Optimization（HPO），一种针对混合离散-连续动作空间的强化学习算法，利用混合梯度（路径梯度+评分函数）实现无偏梯度估计；

**💡 创新点**

创新点在于：1）设计无偏的混合梯度估计器，结合路径梯度与评分函数；2）证明在接近离散最佳响应时交叉项趋于零，为去中心化训练提供理论依据；3）将分段平滑问题重构为混合形式，扩大适用范围；

**🔧 技术方法**

使用可微仿真器进行路径梯度回传、基于PyTorch的自动微分实现、优势估计（GAE）结合混合梯度、对交叉项可选择性剔除；

**📊 数据集**

在两个基准任务上验证：联合补货问题（JRP）和多模式线性二次调节器（S-LQR），通过不同连续动作维度与模式数进行实验；

**📈 对比分析**

与标准PPO对比：HPO 在连续动作维度增大时收敛速度显著快，收敛次数和最终性能远优于 PPO；在高维情形下，HPO 的梯度方向一致性高、方差低；

**⚠️ 局限性**

局限性：仅适用于训练时可访问 exogenous 仿真器（非黑盒）；交叉项剔除在离散策略尚未接近最佳响应时可能导致偏差；在极大状态空间或非常复杂的非平滑动力学下，理论假设可能不满足。

---

## 17. MetaAgent-X : Breaking the Ceiling of Automatic Multi-Agent Systems via End-to-End Reinforcement Learning

**arXiv ID:** 2605.14212 | [PDF](https://arxiv.org/pdf/2605.14212v1)

**作者:** Yaolun Zhang `[一作]` (Oregon State University), Huazheng Wang `[通讯]` (Oregon State University)

**通讯引用:** 523 | [OpenAlex ID](https://openalex.org/A5062299183)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种端到端强化学习框架，联合优化自动多智能体系统的设计者与执行者，实现系统的自设计与自执行；

**💡 创新点**

核心创新在于①Executor-Designer分层回合采样与层次化信用分配，②阶段性共进化训练（Stagewise Co‑evolution），突破了冻结执行者导致的性能瓶颈并揭示设计者-执行者的互进化机制；

**🔧 技术方法**

使用大型语言模型（Qwen3 4B/8B）+ GRPO 强化学习，树结构回合采样、分层信用分配、阶段性训练（K=30）、SFT 预训练、共享/分离策略、Python 脚本生成 MAS 与工具接口；

**📊 数据集**

训练数据包括 DeepSeek-V3.2 生成的 3K 设计样本与 8K 执行样本；RL 阶段采用 Polaris‑Dataset‑53K、APPS introductory subset、CodeContests；评估基准为数学（AIME24、AIME25、OlympiadBench）与代码（LiveCodeBench‑v6、APPS、CodeContests）；

**📈 对比分析**

与单智能体（direct prompting、GRPO）、搜索式自动MAS（AFlow、ADAS）以及 RL 方式自动MAS（ScoreFlow、MaAS、AFM‑Coder）对比，RL 模型在所有 6 个基准上均优于所有基线，平均提升约 11–12%（Qwen3‑8B）或 22%（Qwen3‑4B）相对单智能体，最大提升 21.7%；相较于 SFT，RL 进一步提升 6.17%；

**⚠️ 局限性**

实验受限于计算资源，未在更大模型或更长训练预算下进行规模化研究；仅在数学与代码任务上验证，缺乏对更复杂任务与长期稳定性的深入探讨；

---

## 18. Mixed Integer Goal Programming for Personalized Meal Optimization with User-Defined Serving Granularity

**arXiv ID:** 2605.13849 | [PDF](https://arxiv.org/pdf/2605.13849v1)

**作者:** Francisco Aguilera Moreno `[一作]` `[通讯]`, Francisco Aguilera Moreno

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出混合整数目标规划（MIGP）模型，用整数份量变量与目标规划偏差变量统一实现个性化膳食优化。

**💡 创新点**

创新点在于把整数份量和软目标结合，解决连续变量导致的分数份量和硬约束不可行问题；发现并利用“偏差吸收”性质，证明整数gap在目标规划中通常为零；实现开源、实时(<100 ms)求解。

**🔧 技术方法**

采用目标规划、整数规划（MIP）、HiGHS求解器、Python实现；通过权重归一化实现多宏量营养素平衡；使用Streamlit构建交互式前端。

**📊 数据集**

基准数据集为USDA FoodData Central 30种食品（可扩展到500k+来自FatSecret）；另外使用30食物池生成810个随机实例。

**📈 对比分析**

与连续目标规划+四舍五入以及硬约束整数规划比较；MIGP在所有实例中100%可行，严格优于GP+四舍五入66%案例，IP仅48%可行；典型求解时间<100 ms（最多25食物1.1 s）。

**⚠️ 局限性**

局限在于未考虑口味偏好/可食性、成本与安全约束、极端约束下的偏差仍存在；求解时间随食物数急剧增长；多餐、微量营养素等扩展仍需进一步研究。

---

## 19. Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning

**arXiv ID:** 2605.14054 | [PDF](https://arxiv.org/pdf/2605.14054v1)

**作者:** Haozhe Wang `[一作]` (Hong Kong University of Science and Technology), Fangzhen Lin `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 3887 | [OpenAlex ID](https://openalex.org/A5102011179)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于强化学习的 MoCA 框架，将视觉感知与逻辑推理分离，并在推理过程中显式地对感知质量进行奖励。

**💡 创新点**

创新点在于：①提出了“感知验证（PV）”的盲fold reasoning 机制，用文本推理器评估感知是否足够；②引入了结构化语义验证（SVV）替代高方差 LLM 判断；③设计了模态感知信用分配（MoCA）逻辑，将奖励精准地归因于“看错”或“想错”。

**🔧 技术方法**

技术手段包括：交互式感知-推理生成（感知/推理块），基于 GRPO 的强化学习，PV 与 SVV 产生的奖励信号，以及 MoCA 机制实现的信用分配与保护。

**📊 数据集**

使用的训练数据涵盖：1）通用视觉指令与推理数据（ViRL39K、VisualWebInstruct-Verified）；2）感知密集数据（Pixel Reasoner）；3）丰富模态数据（从 arXiv、报纸、信息图文中抓取的文档式视觉问答）。

**📈 对比分析**

在 9 类视觉语言基准（如 V*、HRBench、InfoVQA、MathVista、MMMu、EMMA、SlideVQA、DUDE、MMLong）上与多种 7B 开源模型以及 GPT‑4o 等商用系统对比，MoCA 在感知、推理和多模态任务上均取得最高分或与 GPT‑4o 同水平，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：①感知验证依赖强大的文本推理器，若代理推理失效会误导奖励；②模型仍需额外的计算与训练资源；③对极其复杂或非文本化的视觉信息的覆盖仍有限，未来需要更通用的感知验证方法。

---

## 20. PreFT: Prefill-only finetuning for efficient inference

**arXiv ID:** 2605.14217 | [PDF](https://arxiv.org/pdf/2605.14217v1)

**作者:** Andrew Lanpouthakoun `[一作]` (Stanford University), Christopher Potts `[通讯]` (Stanford University)

**通讯引用:** 26501 | [OpenAlex ID](https://openalex.org/A5042601761)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Prefill-only Finetuning（PreFT），仅在预填充阶段对 LLM 进行参数高效微调，从而实现多适配器的高吞吐量部署。

**💡 创新点**

创新点在于：1）仅在预填充阶段使用适配器，避免在解码阶段造成内存带宽瓶颈；2）将 LoRA 与 ReFT 这两种主流 PEFT 直接改造成预填充专用版本；3）在大规模模型（0.5B~70B）和多 GPU 场景下展示显著吞吐提升，且在多任务评估中保持与传统 PEFT 的性能相近。

**🔧 技术方法**

使用了 LoRA、ReFT（DiReFT）两种适配器架构；在 vLLM 推理引擎上实现多适配器预填充，结合自定义 Triton kernel 与内存分页技术；并在 RL 环境中集成 GRPO 算法进行验证。

**📊 数据集**

实验数据集包括：Tülu-3（指令调优），OpenThoughts（长文本推理），GSM8K、MATH（数学推理），MBPP、HumanEval（代码生成），LongWriter、LongBench-Write（长文本生成）。

**📈 对比分析**

与原始全位置 LoRA/ ReFT 以及 vLLM 的多 LoRA 推理进行比较：在 512 个适配器上预填充专用 LoRA/ ReFT 的吞吐量分别提升 1.87× 和 1.90×；在 SFT 任务中评估损失略高但下游指标无显著差异；在 RL 任务中与传统 PEFT 接近，GSM8K 上存在轻微劣势。

**⚠️ 局限性**

限制：1）预填充专用适配器在极长生成（LongWriter）或特定推理任务（GSM8K）时，效果可能随距离衰减；2）目前仅改造现有 PEFT 架构，未设计针对预填充的全新适配器模型；3）与 KV 缓存型适配方法相比，尚未进行系统对比，可能在中间性能点上落后。

---

## 21. Support Before Frequency in Discrete Diffusion

**arXiv ID:** 2605.13999 | [PDF](https://arxiv.org/pdf/2605.13999v1)

**作者:** Adrian Müller `[一作]` (ETH Zurich), Niao He `[通讯]` (ETH Zurich)

**通讯引用:** 1360 | [OpenAlex ID](https://openalex.org/A5071683073)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究离散扩散语言模型在逆过程中的学习层次，证明在低噪声下模型先恢复数据支持（可接受序列集合），随后才精细化支持内的频率信息。

**💡 创新点**

提出支持优先（Support‑before‑Frequency）假设，并给出统一与吸收（masking）扩散的小噪声逆核展开，揭示两种腐败机制产生不同尺度分离（均匀扩散三尺度，吸收扩散单尺度）及其对逆采样的影响。

**🔧 技术方法**

使用离散分数理论、逆核小噪声展开、理论指导的阈值推断，以及基于支持与频率的实验探测方法；对比均匀与吸收扩散在采样过程中的项目化行为。

**📊 数据集**

实验数据包括：受控正则语言（已知支持与频率）和公开文本语料 FineWeb，用于评估支持定位与频率排序的训练进度。

**📈 对比分析**

通过支持定位、频率排名与对比采样有效率的指标，观察到支持指标显著提前出现（约 1 亿训练步/数千万 token），并验证吸收扩散在阈值采样下更接近投影，而均匀扩散可通过阈值筛选显著提升有效率。

**⚠️ 局限性**

局限性：分析仅覆盖逆过程的最终低噪声阶段；未给出学习过程按理论顺序收敛的证明；在高维自然语言中支持可能极为稀疏，实验规模与模型复杂度有限。

---

## 22. Derivation Prompting: A Logic-Based Method for Improving Retrieval-Augmented Generation

**arXiv ID:** 2605.14053 | [PDF](https://arxiv.org/pdf/2605.14053v1)

**作者:** Ignacio Sastre `[一作]` (Universidad de la República), Aiala Rosá `[通讯]` (Universidad de la República)

**通讯引用:** 217 | [OpenAlex ID](https://openalex.org/A5104021555)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Derivation Prompting的逻辑推导式提示方法，用于检索增强生成（RAG）框架的生成阶段，减少幻觉和错误推理。

**💡 创新点**

将逻辑推导树的概念迁移到LLM提示，利用预定义规则逐步构建推导树，既可解释又能控制生成过程。

**🔧 技术方法**

采用LLM（Claude 3 Haiku/Opus）、检索模块（Cross‑Encoder句向量、长上下文窗口）以及基于规则的推导算法。

**📊 数据集**

使用UDELAR工程学院的17个网页（Markdown格式）以及从OCS邮件中抽取的135条真实用户查询。

**📈 对比分析**

与传统RAG和长上下文窗口比较，Derivation Prompting在Claude Opus下可接受回答率提升至89.6%，显著降低不接受回答（1/2分），虽然对完整准确回答（4/5分）影响不大。

**⚠️ 局限性**

受限于模型规模（较小模型如Haiku效果下降）和规则设计的灵活性，且需进一步验证在更大、多样化数据集上的鲁棒性。

---

## 23. Not All Timesteps Matter Equally: Selective Alignment Knowledge Distillation for Spiking Neural Networks

**arXiv ID:** 2605.14252 | [PDF](https://arxiv.org/pdf/2605.14252v1)

**作者:** Kai Sun `[一作]` (Monash University), Levin Kuhlmann `[通讯]` (Monash University)

**通讯引用:** 4357 | [OpenAlex ID](https://openalex.org/A5069146692)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种名为SeAl-KD的选择性时序知识蒸馏方法，用于提高脉冲神经网络（SNN）的分类性能。

**💡 创新点**

创新点在于两层选择性对齐：错误意识对齐（ELA）在错误时刻仅对错误类别与真实类别的对齐进行logit等化，避免误导；选择性时间对齐（STA）通过置信度和相似度权重，挑选可靠的时间步进行加权对齐，从而实现更精准的时序指导。

**🔧 技术方法**

采用的技术包括：基于KL散度的蒸馏、logit等化、entropy置信度评估、cosine相似度度量、softmax权重化、跨时间的加权KL损失，并将这些损失与常规交叉熵联合训练。

**📊 数据集**

使用的公开数据集包括：CIFAR-10、CIFAR-100、ImageNet以及事件驱动的DVS-CIFAR10，涵盖帧式与事件式视觉任务。

**📈 对比分析**

在上述四个数据集上与直接训练以及多种现有SNN蒸馏方法（如Logit-SNN、KDSNN、STA-KL等）对比，SeAl-KD均实现了显著提升，平均提升0.5%~1.5%精度，并且在较少时间步下能保持或超过传统方法的性能。

**⚠️ 局限性**

局限性包括：需调节蒸馏权重超参数α、β，对不同任务和网络结构的泛化能力待进一步验证；方法主要针对时序分类任务，对其他任务的适用性尚未评估；推理阶段无额外开销，但训练过程需额外计算资源。

---

## 24. Distribution-Aware Algorithm Design with LLM Agents

**arXiv ID:** 2605.14141 | [PDF](https://arxiv.org/pdf/2605.14141v1)

**作者:** Saharsh Koganti `[一作]` (Texas A&M University), Tomer Galanti `[通讯]` (Texas A&M University)

**通讯引用:** 286 | [OpenAlex ID](https://openalex.org/A5050776111)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究从样本中学习可执行求解器代码的分布感知程序学习，利用LLM生成可编译的求解器提示。

**💡 创新点**

将求解器学习建模为样本→提示→求解器的分解，并证明样本可以识别可执行的结构提示，从而在分布上实现正确性与运行时最优。

**🔧 技术方法**

采用大语言模型（LLM）生成结构假设、分析程序和求解器；结合经验风险最小化、提示恢复、样本复杂度分析以及基于提示的编译技术。

**📊 数据集**

在21个结构化组合优化目标分布（七类问题、21个隐藏家族）上进行实验，并在PACE 2025 Dominating Set私有实例上做外部对照。

**📈 对比分析**

与传统启发式、基于优化的时间限制后端和精确求解器对照，平均归一化质量0.971，比最佳启发式提升+0.098，速度比最佳启发式、Gurobi、精确后端分别快336.9×、342.8×、16.1×；在PACE中速度提升约100×但质量略差。

**⚠️ 局限性**

合成成本一次性且需足够实例才能摊销；对分布漂移敏感；不同运行可能产生不同提示，稳定性不佳；并非所有分布都能显著加速。

---

## 25. Uncovering Trajectory and Topological Signatures in Multimodal Pediatric Sleep Embeddings

**arXiv ID:** 2605.14156 | [PDF](https://arxiv.org/pdf/2605.14156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 26. Paraphrasing Attack Resilience of Various AI-Generated Text Detection Methods

**arXiv ID:** 2605.14240 | [PDF](https://arxiv.org/pdf/2605.14240v1)

**作者:** Andrii Shportko `[一作]` (Northwestern University), Inessa Verbitsky `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM生成文本检测方法在改写攻击下的鲁棒性，并提出了一个集成Binoculars、文本特征和RoBERTa的随机森林元学习框架；

**💡 创新点**

创新点在于将三种检测器（Binoculars、文本特征、RoBERTa）以七种组合方式集成，并系统评估改写攻击对每种模型及集成模型的性能与鲁棒性之间的权衡；

**🔧 技术方法**

采用RoBERTa微调、Binoculars交叉困惑度、手工文本特征（平均词长、词汇多样性、标点频率、句长标准差、停用词比例）以及随机森林元学习器，同时使用GPTInf进行改写攻击和统计检验；

**📊 数据集**

使用COLING 2025 Task 1的二分类机器生成文本检测数据集（包含M4GT等），训练集约20k条，评估集73k条，并从GPTInf生成的200条改写样本；

**📈 对比分析**

通过单模型与七种组合的F1对比评估，基线单模型F1在70-80%之间；最优组合（三者全集成）F1为80.61%，改写后降至0.6716；RoBERTa在改写攻击中的退化最小（-0.099），Binoculars退化最大（-0.196），揭示高性能模型易受攻击的现象；

**⚠️ 局限性**

局限在于改写数据集规模仅200条，且仅使用GPTInf改写，未覆盖其他攻击手段；未对不同提示分布或跨域场景下的鲁棒性进行测试；

---

## 27. A Systematic Evaluation of Imbalance Handling Methods in Biomedical Binary Classification

**arXiv ID:** 2605.14147 | [PDF](https://arxiv.org/pdf/2605.14147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 28. Behavior Cloning for Active Perception with Low-Resolution Egocentric Vision

**arXiv ID:** 2605.14106 | [PDF](https://arxiv.org/pdf/2605.14106v1)

**作者:** Anthony Bilic `[一作]` (University of Central Florida), Ladislau Bölöni `[通讯]` (University of Central Florida)

**通讯引用:** 5138 | [OpenAlex ID](https://openalex.org/A5042166639)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

用行为克隆训练低成本机械臂，在闭环控制下通过重定位手腕摄像头完成植物定位与抓取任务。

**💡 创新点**

证明仅凭行为克隆即可产生主动感知行为，并显示预测相对关节增量比绝对位置预测更优。

**🔧 技术方法**

端到端视觉编码器（4层CNN）+LSTM时序控制器，使用低分辨率（64×64）egocentric RGB图像，训练目标为关节增量。

**📊 数据集**

基于Xbox遥控收集的遥操作演示数据集，包含不同植物左右位置的示例；实验中使用 2、4、8、16 等不同数量演示。

**📈 对比分析**

通过比较增量与绝对位置预测的 MSE 与成功率；增量模型在 8 个演示时成功率 5/5、MSE 低、动作更平滑，且能适应未见位置；绝对位置模型多次失效、误差较大。

**⚠️ 局限性**

实验仅在极简、受控环境下验证，数据量有限，未测试更复杂场景或大规模物体，缺乏直接优化信息获取的策略。

---

## 29. Distributed Statistical Zero-Knowledge Proofs via Sumcheck

**arXiv ID:** 2605.14015 | [PDF](https://arxiv.org/pdf/2605.14015v1)

**作者:** Benjamin Jauregui `[一作]` (Universidad de Chile), Masayuki Miyamoto `[通讯]` (University of Tsukuba)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种通用的分布式统计零知识证明框架，核心是将经典的 Sumcheck 协议迁移到分布式网络上，并通过随机多项式掩码与 cut‑and‑choose 技术实现统计零知识。利用此原语，作者构造了针对两类重要图问题的分布式零知识证明：判定图是否可 k‑着色（k 为常数）和计算图中指定子图的计数（如三角形或 k‑团）。

**💡 创新点**

创新点包括：
1) 设计了分布式 Sumcheck 的统计零知识实现，首次提供了一个可模块化、无密码假设的通用零知识原语；
2) 通过对 Sumcheck 进行变量折叠和递归消除，实现了对子图计数问题的 O(log log n) 轮数压缩；
3) 给出了针对常数度图的 (n/log n) 轮下界，证明进一步压缩几乎不可行；
4) 采用随机域选取与 Fourier 级联分析，使得模拟器仅需产生均匀随机值即可模拟真实视图。

**🔧 技术方法**

关键技术包括：
- 分布式 Sumcheck 的树形聚合实现；
- 随机多项式加密与线性组合掩码，保证中间值在本地节点间保持隐私；
- cut‑and‑choose 与多份随机多项式验证以降低作弊概率；
- 递归变量消除与折叠（divide‑and‑conquer Sumcheck）来降低轮数；
- 统计零知识的证明利用总变差距离、Schwartz–Zippel 与 Fourier 变换；
- 对图的多项式表示（k‑着色的 3CNF 归约、多边形计数的多项式扩展）。

**📊 数据集**

由于本工作是理论性论文，没有使用具体实验数据集；所有结果均在抽象的无随机图模型和可计算性假设（如随机字段大小、可多项式时间验证器）下给出。

**📈 对比分析**

与已有分布式 IP/零知识方案相比：
- 对 k‑着色问题提供了从 O(n²)（非交互式）到 O(n)（交互式）总通信的突破；
- 对子图计数（如三角形）实现了从 O(log² n) 轮、O(log n) 消息量降至 O(log n) 轮、O(log n) 消息量的统计零知识证明，并进一步在常数度图上实现了 O(log log n) 轮的压缩；
- 所有方案均保持了可忽略的误差 (soundness ≤ N·d/q) 与 O(log n) 的消息尺寸。

**⚠️ 局限性**

局限性包括：
- 只实现了统计零知识而非完美零知识，且证明依赖于随机域大小的选择；
- 对特定问题的实现仍需假设各节点可访问相应多项式的 oracle；
- 对非常数度图的轮数压缩仍受限，进一步压缩可能需要突破 Arthur–Merlin ETH；
- 方案对节点模型的同步与树形拓扑假设较强；
- 目前仅覆盖了几类图问题，扩展到更通用的算子或计算模型仍是开放挑战。

---

## 30. GraphBit: A Graph-based Agentic Framework for Non-Linear Agent Orchestration

**arXiv ID:** 2605.13848 | [PDF](https://arxiv.org/pdf/2605.13848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 31. SparseOIT: Improving Order-Independent Transparency 3DGS via Active Set Method

**arXiv ID:** 2605.13855 | [PDF](https://arxiv.org/pdf/2605.13855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 32. PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts

**arXiv ID:** 2605.14002 | [PDF](https://arxiv.org/pdf/2605.14002v1)

**作者:** Yifei Zhu `[一作]` (University of Hong Kong), Yifei Zhu `[通讯]` (University of Hong Kong)

**通讯引用:** 1584 | [OpenAlex ID](https://openalex.org/A5101402786)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 PolitNuggets benchmark 中构建了一个多语言、面向 400 名全球政治人物的政治传记合成任务，评估代理式 LRM 的主动检索与推理能力。

**💡 创新点**

创新点在于提出面向长尾事实检索与合成的“Reasoning through Context”评估框架、基于动态证据验证的 FactNet 评估协议以及跨语言证据网络的实验设计。

**🔧 技术方法**

使用了 Supervisor–Searcher 多智能体架构、Archive 证据存储、LLM 判别器与工具调用，以及对多语言检索和工具可靠性的评估。

**📊 数据集**

数据集为从 WhoGov 采样的 400 名政治人物名单以及通过 Grok‑4‑Fast 等代理收集的网页证据，结合 Wikipedia 作为覆盖过滤器。

**📈 对比分析**

与传统被动上下文推理模型对比，Agentic Bios 在事件级别 F1 上 0.768（US）/0.712（非US）优于长上下文 LRM，且搜索步数与 token 使用更低，表明代理式搜索更高效；但在属性级别 F1 与效率上仍存在显著差距。

**⚠️ 局限性**

主要限制包括对非英语证据的依赖导致的国际证据差距、对多步推理的精细校验不足，以及评估协议仍依赖 LLM 判别器的主观性。

---

## 33. Bidirectional Empowerment of Metamorphic Testing and Large Language Models: A Systematic Survey

**arXiv ID:** 2605.13898 | [PDF](https://arxiv.org/pdf/2605.13898v1)

**作者:** Zheng Zheng `[一作]` (Beihang University), Tsong Yueh Chen `[通讯]` (Swinburne University of Technology)

**通讯引用:** 12501 | [OpenAlex ID](https://openalex.org/A5035107113)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 93 篇关于大语言模型与变异测试相互作用的研究进行了系统综述，并提出了双向赋能的框架和分类法。

**💡 创新点**

首次将 MT 与 LLM 的相互促进关系系统化为双向赋能模型，构建了功能与应用双维度的层级分类体系。

**🔧 技术方法**

采用系统文献检索、筛选、数据提取与多标签编码等方法，构建了详细的研究目录与统计分析。

**📊 数据集**

主要数据来源为 5 个学术数据库与 arXiv，共计 93 篇主研究论文；并利用这些论文的实验结果与指标进行整理。

**📈 对比分析**

通过统计分析与对比展示了研究在功能目标、质量属性与自动化阶段的分布，揭示了 MT 在 LLM 质量保证中的主导位置和 LLM 在 MT 自动化中的新兴作用；整体表现表明该领域已从单一案例向系统化评估与自动化转变。

**⚠️ 局限性**

局限性包括依赖公开论文导致的发表偏倚、仅覆盖英文文献、缺乏对实验细节的深入复现以及对低资源/多语言场景的研究不足。

---

## 34. Computational Thinking Development in AI Agent Creation_A Mixed-Methods Study

**arXiv ID:** 2605.14330 | [PDF](https://arxiv.org/pdf/2605.14330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 35. Wavelet-Based Observables for Koopman Analysis: An Extended Dynamic Mode Decomposition Framework

**arXiv ID:** 2605.14224 | [PDF](https://arxiv.org/pdf/2605.14224v1)

**作者:** Cankat Tilki `[一作]` (Virginia Tech), Serkan Gugercin `[通讯]` (Virginia Tech)

**通讯引用:** 7118 | [OpenAlex ID](https://openalex.org/A5028332862)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出利用连续小波变换构造小波观测量，将其作为Koopman算子分析的基底，并在此基础上设计了cWDMD算法，用于近似Koopman算子及其谱；

**💡 创新点**

创新点在于证明当选取调制高斯小波时，小波观测量成为Koopman半群的特征函数，从而得到Koopman算子及其解析式的闭式表达，并将其融入EDMD框架；

**🔧 技术方法**

主要技术包括连续小波变换、Koopman算子理论、特征函数构造、EDMD及最小二乘拟合；

**📊 数据集**

实验数据由对二维线性系统（100条随机初始轨迹）和Lorenz三维系统（40条随机初始轨迹）数值仿真得到的输出序列构成；

**📈 对比分析**

在二维线性系统上与解析解对比，cWDMD在频率峰值附近的Koopman解析式逼近误差低于1%，在Lorenz系统中与文献数值结果相似，显示出良好的准确性；

**⚠️ 局限性**

局限性包括对小波类型的依赖（需满足可逆性约束），有限长度数据导致边缘效应，以及算法在非平稳或高维系统中需要进一步验证。

---

## 36. Neuromorphic Graph Anomaly Detection via Adaptive STDP and Spiking Graph Neural Networks

**arXiv ID:** 2605.13863 | [PDF](https://arxiv.org/pdf/2605.13863v1)

**作者:** Abdul Joseph Fofanah `[一作]` (Griffith University), Kwabena Sarpong `[通讯]` (Griffith University)

**通讯引用:** 300 | [OpenAlex ID](https://openalex.org/A5026135262)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3855fcda-48ef-4070-a15e-803cd5c84d83` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于脉冲神经网络与 Spike‑Timing‑Dependent Plasticity 的自适应时序图异常检测框架ASTDP‑GAD，能在动态网络中实现低能耗的异常检测；

**💡 创新点**

创新点包括：① 自适应LIF时间脉冲图编码；② LIF图注意力与侧抑制；③ 事件驱动超图记忆与STDP更新；④ 脉冲率对比池化；⑤ 自适应STDP学习；⑥ 多尺度时间卷积与多因子融合，并提供理论保证；

**🔧 技术方法**

采用技术包括脉冲神经网络（LIF）、STDP、LIF‑GAT图注意力、超图记忆、脉冲率对比池化、时间卷积、融合模块以及能耗评估；

**📊 数据集**

实验数据集涵盖7个静态图（Yelp、T‑Finance、Weibo、BlogCatalog、T‑Social、Flickr、Amazon）和3个动态图（DBLP、Tmall、Patent）；

**📈 对比分析**

与24种SOTA基线（含非脉冲与脉冲方法）对比，在3个动态基准上取得最高Macro‑F1：DBLP 85.34% (+6.2% vs ChronoSpike)，Tmall 76.89% (+12.1%)，Patent 92.58% (+6.5%)，同时保持高能效（脉冲稀疏率 λ=0.24）；

**⚠️ 局限性**

局限性包括对训练数据量敏感、在稀疏/小规模图上的表现不如大规模图、STDP超参数需精细调优、目前仅在仿真环境验证，实际硬件部署仍待进一步评估。

---

## 37. Reliability-Gated Source Anchoring for Continual Test-Time Adaptation

**arXiv ID:** 2605.14063 | [PDF](https://arxiv.org/pdf/2605.14063v1)

**作者:** Vikash Singh `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**通讯引用:** 4069 | [OpenAlex ID](https://openalex.org/A5004523290)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种持续测试时适应（CTTA）的方法，通过使用冻结的源模型的归一化预测熵来动态调整适应过程中的锚定强度，以应对源模型可靠性下降的问题。

**💡 创新点**

创新点在于引入了基于可靠性的门控机制，能够在源模型的预测熵接近均匀时自动关闭锚定，从而避免了盲目锚定的失败模式，并保证了优雅衰减的特性。

**🔧 技术方法**

使用了基于熵的可靠性门控机制，结合了ROD框架的优化方法，并引入了边际校准、信心缩放学习率和解耦翻转插值等辅助稳定器。

**📊 数据集**

在CCC（持续腐蚀）基准上进行了评估，使用了不同难度级别的图像数据集（Easy, Medium, Hard），并在CIN-C和IN-C数据集上进行了验证。

**📈 对比分析**

与七个基线方法进行了比较，结果显示在9个基准单元中获得了8个单元的最低错误率，相比于最强基线ROID+ASR，ResNet-50的平均错误率降低了1.05个百分点，ViT-B/16降低了0.48个百分点。

**⚠️ 局限性**

限制在于该方法仅使用源熵作为可靠性信号，无法处理自信错误的低熵源，未来的工作方向是设计一个更为准确的可靠性信号。

---

## 38. Are Agents Ready to Teach? A Multi-Stage Benchmark for Real-World Teaching Workflows

**arXiv ID:** 2605.14322 | [PDF](https://arxiv.org/pdf/2605.14322v1)

**作者:** Zixin Chen `[一作]` (Hong Kong University of Science and Technology), Huamin Qu `[通讯]` (Qwen Team, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EduAgentBench，一个基于源材料的多维度评估框架，用以衡量语言模型在教学判断、情境化辅导与教学工作流程执行上的能力。

**💡 创新点**

创新点在于将教学评估拆分为三种能力表面，并通过“目标洞见-源材料-验证器”管线构造可检验、可解释的任务，首次实现理论驱动、真实环境兼顾的全流程评估。

**🔧 技术方法**

采用规则与自然语言验证相结合的多层评估链，配合Canvas风格仿真环境、工具调用与弱模型转移验证机制来检验模型的教学决策与行动。

**📊 数据集**

任务来源于公开评估、开放课程材料与教育文献，构成150个任务（50判断、40教学、60工作流程），均为可复现的源材料。

**📈 对比分析**

对比多种前沿大模型，使用等阶段加权的通过率和奖励进行评价，最佳模型仅达到约48%的等阶段通过率，表明判断能力强但情境化辅导和工作流程执行仍显不足。

**⚠️ 局限性**

局限在于仅为仿真基准，缺乏真实学生/教师互动与长期学习成果评估，且未覆盖多模态教学和长期课程规划等关键维度。

---

## 39. MathAtlas: A Benchmark for Autoformalization in the Wild

**arXiv ID:** 2605.14061 | [PDF](https://arxiv.org/pdf/2605.14061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 40. Neural Code Translation of Legacy Code: APL to C#

**arXiv ID:** 2605.13896 | [PDF](https://arxiv.org/pdf/2605.13896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 41. WirelessSenseLLM: Zero-Shot Human Activity Understanding by Bridging Wireless Signals and Human Language

**arXiv ID:** 2605.14070 | [PDF](https://arxiv.org/pdf/2605.14070v1)

**作者:** Mahmuda Keya `[一作]` (University of Massachusetts Dartmouth), Long Jiao `[通讯]` (University of Massachusetts Dartmouth)

**通讯引用:** 1545 | [OpenAlex ID](https://openalex.org/A5087996220)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过将Wi‑Fi CSI信号映射到语言语义空间，结合预训练的大语言模型，实现了无需信号段划分的零射人类动作理解与自然语言描述。

**💡 创新点**

提出了CSI‑到‑语言适配器与跨模态投影机制，能够桥接连续无线信号与离散语言表示，实现细粒度动作描述和零射动作分类。

**🔧 技术方法**

采用Wi‑Fi编码器、CSI‑到‑语言适配器、跨模态投影层、预训练Vicuna‑7B LLM 并使用 LoRA 进行指令微调。

**📊 数据集**

构建并公开了约 97,000 条无线‑文本、22,500 条视频‑文本和 40,688 条文本单独记录的新多人人 CSI‑文本数据集。

**📈 对比分析**

与传统 SVM、CNN、RNN 等方法对比，WirelessSenseLLM 在零射动作识别中实现 92% 准确率/91% F1 分数，并在语言推理指标（ROUGE、BLEU、BERTScore 等）上较基线提升约 30%。

**⚠️ 局限性**

在多人人场景下性能下降，交互推理仍显不足；此外，模型对不同硬件布局、室内环境的鲁棒性尚待进一步验证。

---

## 42. You Only Landmark Once: Lightweight U-Net Face Super Resolution with YOLO-World Landmark Heatmaps

**arXiv ID:** 2605.14166 | [PDF](https://arxiv.org/pdf/2605.14166v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 43. How to Scale Mixture-of-Experts: From muP to the Maximally Scale-Stable Parameterization

**arXiv ID:** 2605.14200 | [PDF](https://arxiv.org/pdf/2605.14200v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 44. Bridging the Rural Healthcare Gap: A Cascaded Edge-Cloud Architecture for Automated Retinal Screening

**arXiv ID:** 2605.14108 | [PDF](https://arxiv.org/pdf/2605.14108v1)

**作者:** Nishi Doshi `[一作]` (University of Southern California), Shrey Shah `[通讯]` (University of Southern California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个两级边缘-云级联模型，用轻量级移动端模型进行可转诊与不可转诊的二分类，云端模型对被转诊图像进行严重程度分级，实现自动化视网膜筛查。

**💡 创新点**

首次提出在农村医疗环境下的两级级联架构，利用本地轻量模型减少云端调用，同时通过阈值调优实现高敏感度，保持与全云模型相近的分级精度。

**🔧 技术方法**

使用 MobileNetV3‑small 进行边缘三分类，RETFound‑DINOv2 Vision Transformer 进行云端分级，并在预处理与阈值设定上采用分层策略。

**📊 数据集**

在公开的 APTOS 2019 Blindness Detection 数据集上进行训练与评估，包含3662张眼底图像。

**📈 对比分析**

将级联模型与全云模型在同一4分类输出空间对比，级联模型云调用率下降至49.52%（节省50.48%），准确率80.49%与全云模型80.76%、QWK 0.8167 vs 0.8184，差异可忽略。

**⚠️ 局限性**

仅在Aptos内部验证，未在外部数据集或真实边缘设备上评估，缺乏字节级带宽与实时延迟测量，结果为回溯性可行性证据而非临床验证。

---

## 45. Enhanced and Efficient Reasoning in Large Learning Models

**arXiv ID:** 2605.14036 | [PDF](https://arxiv.org/pdf/2605.14036v1)

**作者:** Leslie G. Valiant `[一作]` (Harvard University), Leslie G. Valiant `[通讯]` (Harvard University)

**通讯引用:** 30736 | [OpenAlex ID](https://openalex.org/A5015531084)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Unary Relational Integracode 的前置预处理方法，将文本中隐含的关系显式编码为一元关系，从而使大型语言模型能够以 k‑DNF 形式学习 Robust Logic 规则，提升推理准确性、真实性并减少幻觉。

**💡 创新点**

创新点在于：1) 将多元关系拆分为若干一元关系并在 token 级别显式加入；2) 证明该编码下 Robust Logic 的核心规则可转化为可 PAC 学习的 k‑DNF；3) 设计能在多次调用间保留关系信息的机制，支持跨调用链式推理。

**🔧 技术方法**

采用语义/篇章分析器进行预处理、扩大 token 集、Unary Relational Integracode、Robust Logic 框架、k‑DNF 学习算法以及改进的 Transformer（可稀疏）结构。

**📊 数据集**

论文未给出具体实验数据，主要针对通用自然语言文本（如公开语料库）进行理论构建，示例使用“Bob/Joe/Sue”场景。

**📈 对比分析**

相较于传统 Transformer，理论上将计算复杂度从 O(dN² + d²N) 降低到 O(g′h d N)，并预期在保持或提升推理与缺失词预测准确度的同时显著降低能耗；但实验验证仍缺失。

**⚠️ 局限性**

局限性包括：1) 需要高质量的语义/篇章分析器；2) 对重复关系可能产生误判；3) 对大文本窗口的扩展成本与可扩展性未知；4) 尚未在真实数据上验证能效与准确性；5) k‑DNF 学习在实践中的复杂度尚待评估。

---

## 46. Spectral Analysis of Fake News Propagation

**arXiv ID:** 2605.13861 | [PDF](https://arxiv.org/pdf/2605.13861v1)

**作者:** Weibin Cai `[一作]` (Syracuse University), Reza Zafarani `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究假新闻传播结构，提出基于图谱的谱界限表示，并设计离散结构优化框架以解释假与真新闻的传播差异。

**💡 创新点**

创新点包括：① 将谱界限与传播属性（分支能力、规模、凝聚力、跨度、扩散动力）统一成五类全新的表示；② 通过第一阶谱扰动近似实现叶节点迁移的快速边界更新；③ 用谱界限驱动的结构演化揭示假新闻的特定结构特征。

**🔧 技术方法**

技术手段：谱理论（谱界限、谱半径、代数连通性、归一化谱等）；图神经网络（GCN）作为基准；离散结构优化算法（score‑guided 与 bound‑guided）；第一阶谱扰动近似；统计评估（Spearman/Kendall/Pearson 相关、相对误差）。

**📊 数据集**

数据集：主要使用微博平台的两组假/真新闻传播树（Weibo22 及另一未命名数据集），包含数千棵传播树。

**📈 对比分析**

与传统手工拓扑特征、GCN 等方法对比，谱界限特征在 ACC/F1 上与手工特征相近、显著优于 GCN；在不同数据集上，特征组合的最佳类别有所变化，体现了平台/事件差异。

**⚠️ 局限性**

局限：仅关注传播结构，未结合内容/源信息；谱界限虽统一但在某些属性上数值误差较大；结构优化过程不总是单调，解释复杂；实验仅覆盖社交网络树形传播，难以推广到更复杂图形。

---

## 47. Time Domain Near Memory Computing Engine

**arXiv ID:** 2605.14162 | [PDF](https://arxiv.org/pdf/2605.14162v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 48. CA2: Code-Aware Agent for Automated Game Testing

**arXiv ID:** 2605.13918 | [PDF](https://arxiv.org/pdf/2605.13918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 49. IG-Diff: Complex Night Scene Restoration with Illumination-Guided Diffusion Model

**arXiv ID:** 2605.14337 | [PDF](https://arxiv.org/pdf/2605.14337v1)

**作者:** Yifan Chen `[一作]` (Tsinghua Univerisity), Yujiu Yang `[通讯]` (Tsinghua Univerisity)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新型的照明引导扩散模型（IG‑Diff）用于复杂夜景的图像恢复，并构建了包含低光与多种恶劣天气（雨、雨滴、雪、雾、霾）共同干扰的合成数据集。

**💡 创新点**

创新点包括：① 通过照明引导模块将估计的照明图像嵌入扩散模型，显著减少过曝/欠曝与色彩失真；② 设计了基于经典成像模型的低光与恶劣天气联合合成管线，使得数据集更贴近真实夜景；③ 引入跨注意力（cross‑attention）在U‑Net中融合照明特征；④ 使用基于补丁的逆向采样策略缓解局部边缘伪影。

**🔧 技术方法**

主要技术手段包括：去噪扩散概率模型（DDPM）与条件扩散训练；预训练的照明估计网络 E_sci 生成光照先验；跨注意力机制将照明特征注入网络；基于 EC‑Zero‑DCE 的低光退化模拟；以及基于重叠补丁的推理策略。

**📊 数据集**

使用的数据集包括：自建 LOL‑weather（包含 5 种天气+低光合成样本），LOL‑blur‑denoise（多重降质），LOL‑v1（单纯低光），以及真实场景的 RESIDE（夜雾子集）和 snow100k（夜雪子集）。

**📈 对比分析**

与 12 种基线（低光增强如 DLN、天气恢复如 STL、GDN、DMS、扩散方法如 WeatherDiff、DiT 等）进行对比，实验表明 IG‑Diff 在 PSNR、SSIM、LPIPS 等指标上均显著优于对手，尤其在多重降质与真实夜景上的鲁棒性更佳。

**⚠️ 局限性**

局限性：① 依赖合成数据，真实对齐样本缺失，可能对极端照明变化适应性有限；② 对图像尺寸的补丁划分依赖网格步长，超大图像时需调整；③ 训练与推理成本相对较高，推理时间受扩散步数影响；④ 在极端雾霾或高动态范围场景下仍可能出现色彩失真或残留噪声。

---

## 50. AI Alignment Amplifies the Role of Race, Gender, and Disability in Hiring Decisions

**arXiv ID:** 2605.13866 | [PDF](https://arxiv.org/pdf/2605.13866v1)

**作者:** Ze Wang `[一作]` (University College London), Michael Thaler `[通讯]` (University College London)

**通讯引用:** 8010 | [OpenAlex ID](https://openalex.org/A5057199445)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在27个语言模型和177个职业上实验，评估其在招聘决策中对性别、种族和残障身份的影响。

**💡 创新点**

发现模型在性别和种族上偏向少数群体，残障群体被惩罚；且后期对齐显著放大这些差异，形成与人类雇主截然不同的偏差模式。

**🔧 技术方法**

使用指令调优对齐的预训练大型语言模型（如 GPT 系列）和对照的基线模型。

**📊 数据集**

利用约一半美国就业岗位的职业数据（177个职业）和对应的候选人资格与身份信息。

**📈 对比分析**

通过与人类对应实验的元分析对比，发现模型对黑人候选人的偏向与人类相反，对残障候选人的惩罚被显著削弱，对女性候选人的优势被放大；对齐后效应显著增强。

**⚠️ 局限性**

局限包括只评估招聘场景，缺乏跨文化验证，模型可能在其他任务中表现不同；缺乏对基于“品味”歧视的直接识别。

---

## 51. CRANE: Constrained Reasoning Injection for Code Agents via Nullspace Editing

**arXiv ID:** 2605.14084 | [PDF](https://arxiv.org/pdf/2605.14084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 52. Why Retrieval-Augmented Generation Fails: A Graph Perspective

**arXiv ID:** 2605.14192 | [PDF](https://arxiv.org/pdf/2605.14192v1)

**作者:** Kai Guo `[一作]` (Michigan State University), Jiliang Tang `[通讯]` (Michigan State University)

**通讯引用:** 26059 | [OpenAlex ID](https://openalex.org/A5040639891)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对检索增强生成（RAG）模型的内部推理过程进行图结构分析，构建归因图来捕捉检索上下文、模型激活与生成结果之间的因果流动，并利用这些图对模型错误进行诊断与干预。

**💡 创新点**

① 用电路追踪技术自动生成归因图；② 发现正确与错误推理在图深度、连通性和信息分布上的系统差异；③ 设计基于图结构的错误检测器和基于注意力重权的推理层级控制，实现对错误推理的实时纠正。

**🔧 技术方法**

电路追踪、归因图构建、图神经网络（Graph Transformer）用于错误检测、层级注意力重权控制（forward hook）用于推理调控。

**📊 数据集**

HotpotQA、2WikiMultihopQA、MuSiQue 以及在 MuSiQue 基础上构造的混合检索场景 Mix‑MuSiQue。

**📈 对比分析**

对错误检测与无监督自评估进行对比，图结构检测平均提升 11.53% 的准确率；对混合检索任务进行注意力干预，准确率从 56.5% 提升至 61.6%（约 9% 的提升）。

**⚠️ 局限性**

主要局限包括：实验仅针对 LLaMA‑3 8B Instruct，可能对不同模型/规模的推广性有限；归因图的构造和特征工程对模型架构有一定依赖；错误标签依赖外部 LLM 判定，存在主观性；未评估在更广泛任务或多模态检索中的表现。

---

## 53. Architecture-Aware Explanation Auditing for Industrial Visual Inspection

**arXiv ID:** 2605.14255 | [PDF](https://arxiv.org/pdf/2605.14255v1)

**作者:** Sibo Jia `[一作]` (Beijing University of Technology), Kunrong Li `[通讯]` (Beijing University of Technology)

**通讯引用:** 59 | [OpenAlex ID](https://openalex.org/A5064293007)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出并验证了一种基于模型本机读取机制的可解释性审核协议，系统评估了工业视觉检测系统中不同模型架构与解释方法的“faithfulness”程度。

**💡 创新点**

创新点在于构建了“native‑readout hypothesis”，将解释方法与模型内部决策路径的结构距离与其解释可靠性联系起来，并通过跨架构对比和控制实验给出可操作的模型‑解释配对指导。

**🔧 技术方法**

采用Attention Rollout、Grad‑CAM、RISE等解释技术，结合删除/插入扰动、稳定性测度以及Cohen’s d、Bootstrap置信区间等统计工具进行量化评估。

**📊 数据集**

实验主要使用WM‑811K晶圆图像分类数据集（172k标注样本），并在MVTec AD异常检测数据集上做边界条件检验。

**📈 对比分析**

结果显示，ViT‑Tiny+Attention Rollout在删除AUC上显著优于CNN‑Grad‑CAM（0.21 vs 0.48‑0.54），插入AUC最高；RISE压缩了不同架构间的差距；而分类准确率与解释faithfulness无关。

**⚠️ 局限性**

局限性包括：扰动基准（zero‑fill vs blur‑fill）对排名有显著影响；仅评估了单一解释方法和少量随机种子；对高分辨率图像和其他解释器的通用性未验证；以及在MVTec AD上预训练差异导致的结果不一致。

---

## 54. Good to Go: The LOOP Skill Engine That Hits 99% Success and Slashes Token Usage by 99% via One-Shot Recording and Deterministic Replay

**arXiv ID:** 2605.14237 | [PDF](https://arxiv.org/pdf/2605.14237v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 55. Network-Aware Bilinear Tokenization for Brain Functional Connectivity Representation Learning

**arXiv ID:** 2605.14048 | [PDF](https://arxiv.org/pdf/2605.14048v1)

**作者:** Leo Milecki `[一作]` (Weill Cornell Medicine), Qingyu Zhao `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 1503 | [OpenAlex ID](https://openalex.org/A5039683634)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了 NERVE，利用网络感知的双线性分解对大规模脑功能连接矩阵进行分块和标记化，并在此基础上训练自监督掩码自编码器，以学习更具可迁移性的功能连接表示。

**💡 创新点**

创新点包括：①将功能网络对作为分块单元，解决传统固定大小补丁在功能网络层面上的不匹配；②采用双线性网络级分解，用网络特定的区域嵌入代替每对网络的独立投影，实现参数从二次降至一次增长，同时保留网络身份信息；③将该分块标记化嵌入与 MAE 结合，提升跨数据集泛化。

**🔧 技术方法**

技术手段主要是：Transformer‑based 掩码自编码器（MAE）、双线性 Khatri–Rao 分解的分块嵌入、Schäfer 17‑网络分区、线性回归 + 核岭回归 的下游预测、以及与多种自监督/监督基线的对比实验。

**📊 数据集**

使用了三个大规模青少年 fMRI 数据集：ABCD（1,791 例）、PNC（1,416 例）和 CCNP（178 例），并在合并 ABCD+PNC 训练后评估 CCNP 作为跨域样本。

**📈 对比分析**

与多种自监督基线（BrainMass、GraphMAE、BrainGSLs、GATE）以及监督基线（BrainNetCNN、BrainGNN、BrainNetTF）对比。NERVE 在大多数行为与精神病学预测任务上获得最高或同等的 Pearson 相关系数，尤其在跨域 CCNP 中展现出更高的稳定性和泛化能力；相比之下，传统的共享线性或仅基于网络的分块标记化表现不如双线性方案。

**⚠️ 局限性**

局限性包括：仅在发展期人群和静态功能连接上验证，缺乏动态 FC 或多模态（结构连接）扩展；模型依赖于预先定义的功能网络分区，分区选择与网络粒度会影响结果；解释性方面仍需进一步挖掘双线性权重和注意力模式。

---

## 56. Implicit spatial-frequency fusion of hyperspectral and lidar data via kolmogorov-arnold networks

**arXiv ID:** 2605.14239 | [PDF](https://arxiv.org/pdf/2605.14239v1)

**作者:** Zekun Long `[一作]` (Griffith University), Jun Zhou `[通讯]` (Griffith University)

**通讯引用:** 19636 | [OpenAlex ID](https://openalex.org/A5100781212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Kolmogorov‑Arnold网络的隐式空间‑频率融合框架IFGNet，用LiDAR引导的隐式聚合实现高光谱图像（HSI）与LiDAR数据的有效融合，并在空间域与频率域同时进行特征聚合，以实现像素级分类。

**💡 创新点**

创新点包括：① 用KAN代替传统CNN/MLP，实现可学习的Spline非线性映射，从而更好建模HSI与LiDAR之间复杂的非线性交互；② 引入LiDAR引导的隐式聚合模块，在空间域根据几何信息动态加权聚合HSI特征；③ 在频率域同样采用隐式聚合，以捕获全局结构与频域相关性；④ 将两域聚合结果相加，形成统一的多模态表征，避免了传统显式拼接或注意力机制的冗余。

**🔧 技术方法**

主要技术包括Kolmogorov‑Arnold网络（KAN）编码器、LiDAR引导的隐式聚合单元（SIAU）、频率域隐式聚合（通过FFT/IDFT处理）、以及轻量级分类头；整体架构采用patch‑based端到端学习，训练使用Adam优化器。

**📊 数据集**

在两个公开的HSI‑LiDAR基准上进行评估：Houston 2013（9×9 patch）和MUUFL Gulfport（5×5 patch），两者均为空间配准好的高光谱与LiDAR数据。

**📈 对比分析**

与CALC、GLTNet、M2FNet、MS^2CANet、S^2ENet、MFT、ExViT等方法比较，IFGNet在两数据集上均取得最优成绩：Houston 2013的总体精度99.37%、平均精度99.50%、Kappa 99.32%；MUUFL的总体精度92.67%、平均精度94.47%、Kappa 90.45，显著提升了对光谱混淆区的识别。

**⚠️ 局限性**

局限性主要包括：仅在两小规模数据集上验证，缺乏对大规模场景或实时部署的评估；使用固定大小的patch，可能在不同尺度下表现不佳；对极端光照或多源噪声的鲁棒性尚未深入探究。

---

## 57. Heuristic Pathologies and Further Variance Reduction via Uncertainty Propagation in the AIVAT Family of Techniques

**arXiv ID:** 2605.14261 | [PDF](https://arxiv.org/pdf/2605.14261v1)

**作者:** Juho Kim `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20314 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了AIVAT等方差缩减技术中启发式价值函数的角色，揭示其可能的脆弱性并提出利用启发式不确定性传播与逆方差加权进一步降低方差。

**💡 创新点**

①发现启发式价值函数可以被训练以人为降低样本方差或伪造统计结论，强调必须在观测前固定该函数；②将启发式输出的不确定性量化并在AIVAT估计中传播，使用逆方差加权实现额外方差减小。

**🔧 技术方法**

AIVAT / MIVAT 控制变量、优势求和、逆方差加权、梯度下降/Adam优化、Gaussian Process回归、k折交叉验证。

**📊 数据集**

公开的Pluribus扑克手牌数据集（10,000手）。

**📈 对比分析**

与无估计、MIVAT-WB、AIVAT的统一加权结果对比；在IVW加权下，MIVAT-GPR平均误差标准差下降约24.5%，等价约43%样本量减少，且估计偏差远小于方差减小。

**⚠️ 局限性**

缺乏对启发式价值函数的正则化，过拟合导致在统一加权下表现不佳；实验仅在MIVAT（无对手动作概率）下进行，未验证在完整AIVAT设置中的效果；不确定性估计依赖于高斯假设，若不满足可能产生偏差。

---

## 58. S-AI-Recursive: A Bio-Inspired and Temporal Sparse AI Architecture for Iterative, Introspective, and Energy-Frugal Reasoning

**arXiv ID:** 2605.13872 | [PDF](https://arxiv.org/pdf/2605.13872v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 59. bde: A Python Package for Bayesian Deep Ensembles via MILE

**arXiv ID:** 2605.14146 | [PDF](https://arxiv.org/pdf/2605.14146v1)

**作者:** Vyron Arvanitis `[一作]` (LMU Munich), David Rügamer `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于JAX和scikit-learn的Python包bde，实现了Microcanonical Langevin Ensembles (MILE)用于表格监督学习的贝叶斯深度学习，提供回归和分类的点估计与不确定性量化。

**💡 创新点**

首创了用户友好、可直接兼容scikit-learn的MILE实现，并将两阶段优化+MCMC采样流程自动化、并行化，显著降低使用门槛。

**🔧 技术方法**

采用JAX的JIT编译与并行化、AdamW优化器、正则化经验风险最小化、Microcanonical Langevin Monte Carlo等技术实现高速采样和训练。

**📊 数据集**

在公开的表格数据集（如UCI数据集等）上进行回归基准实验，使用了常见的基准数据集进行对比。

**📈 对比分析**

与线性模型、随机森林、XGBoost、CatBoost、Deep Ensemble和TabPFN(V2)等方法对比，BDE在RMSE和NLL指标上表现优异，尤其在分布式回归和置信区间方面优于传统模型，尽管计算成本更高。

**⚠️ 局限性**

主要限制在于相较于纯优化模型，采样过程计算成本较大；依赖JAX环境；仅适用于表格数据；需手动调节能量方差等采样超参数。

---

## 60. ChromaFlow: A Negative Ablation Study of Orchestration Overhead in Tool-Augmented Agent Evaluation

**arXiv ID:** 2605.14102 | [PDF](https://arxiv.org/pdf/2605.14102v1)

**作者:** Tarun Mittal `[一作]` `[通讯]`, Tarun Mittal

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ChromaFlow框架，并在GAIA 2023 Level‑1验证任务上比较了基线与扩展调度配置的表现。

**💡 创新点**

创新之处在于通过负向消融实验揭示了过度的任务编排会导致准确率下降和操作噪声增加，提出了可靠性门控和清洁评估协议。

**🔧 技术方法**

该系统集成了规划器驱动的工具调用、文档解析、网络搜索、Python/ shell执行与浏览器自动化，并记录了运行时遥测。

**📊 数据集**

实验使用了GAIA 2023 Level‑1验证集，共53个任务。

**📈 对比分析**

采用聚合准确率与运营噪声指标对比，基线为54.72%准确率，扩展配置为50.94%，但噪声和成本显著上升，显示性能未提升。

**⚠️ 局限性**

局限性包括仅评估GAIA Level‑1，结果难以外推，成本估计为运营近似，且日志采样可能影响噪声计数。

---

## 61. Mini-JEPA Foundation Model Fleet Enables Agentic Hydrologic Intelligence

**arXiv ID:** 2605.14120 | [PDF](https://arxiv.org/pdf/2605.14120v1)

**作者:** Mashrekur Rahman `[一作]` `[通讯]` (Dartmouth Libraries), Mashrekur Rahman (Dartmouth Libraries)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并预训练了五个专用的 Mini‑JEPA 基础模型（分别基于 Sentinel‑2 光学、Sentinel‑1 SAR、MODIS 热红外、Sentinel‑2 现象学以及 SRTM+SoilGrids 地形土壤堆叠），并通过路由 LLM 将其与 AlphaEarth 通用模型组合成一个可回答水文问题的 Agentic Retrieval 系统。

**💡 创新点**

① 将小型传感器专用模型与 LLM 路由器相结合，提供可替代或补充大型通用模型的高效方案；② 在所有模型使用完全相同的 ViT‑Sbackbone 与 JEPA 训练流程，展示了传感器物理特性如何直接决定模型特化与嵌入流形几何差异；③ 对比 AlphaEarth，系统在单模态、物理匹配的水文问题上实现显著性能提升，证明专用模型的互补价值。

**🔧 技术方法**

使用 Vision Transformer‑S（12层、6头、384隐藏维度）作为编码器，Joint Embedding Predictive Architecture（JEPA）+ VICReg 进行自监督预训练；通过 k‑NN 检索与 64 维嵌入；路由与合成 LLM（Claude Sonnet/Opus）处理检索计划；评估采用交叉验证 R²、Spearman 相关、随机森林重要性、参与率、局部内在维数、CCA、以及 LLM‑as‑Judge 的五项评分。

**📊 数据集**

训练数据为 9,704 个 30 m 分辨率的 CONUS 区块，采集自 2022 年的 Google Earth Engine 卫星产品：Sentinel‑2 光学、Sentinel‑1 SAR、MODIS LST、Sentinel‑2 现象学以及 SRTM+SoilGrids；配套环境标签包括 SMAP 土壤湿度、PRISM 降水/温度、NLCD 土地覆盖、Köppen 气候类别、海拔与干旱指数。

**📈 对比分析**

通过交叉验证 R² 对每个 Mini‑JEPA 的物理变量预测能力进行评估；对嵌入流形进行全局/局部维数和参与率分析；使用 CCA 与随机森林预测增益评估与 AlphaEarth 的互补性；路由准确率在所有 40 题中达到 100%；LLM‑as‑Judge 评估显示，在单模态问题上 AE+Fleet 对比 AE‑only 的 Cohen’s d 为 1.10（p=0.031），平均得分约 4.4/5，整体提升有限，但在物理匹配问题上明显优于通用模型。

**⚠️ 局限性**

① 数据仅覆盖 CONUS 单年，缺乏跨季、跨区域的一般化能力；② 仅对 5 种传感器进行特化，未考虑水文过程或时间尺度特化；③ LLM‑as‑Judge 评分易于饱和，难以捕捉更细粒度差异；④ 问题集仅 40 题，样本量有限；⑤ 与 AlphaEarth 的对比受限于相同的 64 维嵌入空间，可能低估通用模型在更广泛任务上的优势。

---

## 62. CoReDiT: Spatial Coherence-Guided Token Pruning and Reconstruction for Efficient Diffusion Transformers

**arXiv ID:** 2605.14191 | [PDF](https://arxiv.org/pdf/2605.14191v1)

**作者:** Zhuojin Li `[一作]` (Qualcomm Technologies Inc), Fatih Porikli `[通讯]` (Qualcomm Technologies Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对 Diffusion Transformers（DiTs）提出结构化的 token 剪枝框架，以在保持图像和视频生成质量的前提下显著降低自注意力计算量。

**💡 创新点**

创新点包括：① 用线性时间的空间一致性（SC）分数快速识别局部冗余 token；② 通过邻域权重聚合实现被剪枝 token 的内容感知重建，避免视觉断裂；③ 采用逐块、逐时间步的递进剪枝计划，使剪枝比例自适应分配到冗余度最高的 Transformer 块和 denoising 过程。

**🔧 技术方法**

核心技术包括：空间一致性评分、邻域加权重聚合重建、基于冗余度的递进剪枝调度、轻量化微调（distillation loss）、与现有 sparse/merge 方案对比的实验评测。

**📊 数据集**

实验使用的主要数据集有：10K 语义图像合成集（CLEAR、FLUX.1-dev 生成数据）、MSCOCO 2014 验证集、nuScenes 用于视频生成，并在 PixArt-α‑1024、PixArt‑Σ‑2048、MagicDrive‑V2 等公开模型上进行测试。

**📈 对比分析**

与 ToMeSD、DiffPruning、EcoDiff、DeepCache 等基线相比，本文方法在 PixArt‑α‑1024 上实现最高 55% 的自注意力 FLOPs 减少，GPU 上 1.33× 的推理加速，移动 NPU 上 1.72× 的加速，同时 FID 仅提升至 28.7（baseline 27.3），CLIP 与 IS 维持或略有提升；在 PixArt‑Σ‑2048 上实现 32% FLOPs 减少、15% 端到端延迟下降；在 MagicDrive‑V2 上实现 39% FLOPs 减少，视频质量与条件对齐指标保持稳定。

**⚠️ 局限性**

局限性包括：① 需要在预训练模型上进行微调，未提供零训练方案；② 目前仅剪枝空间 Transformer 块，未完全解决视频的时间一致性问题；③ 过度剪枝会导致重建误差累积，尤其在高分辨率或极端纹理细节场景下可能出现细节损失。

---

## 63. Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints

**arXiv ID:** 2605.14067 | [PDF](https://arxiv.org/pdf/2605.14067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 64. Rethinking Layer Relevance in Large Language Models Beyond Cosine Similarity

**arXiv ID:** 2605.14075 | [PDF](https://arxiv.org/pdf/2605.14075v1)

**作者:** Cristian Hinostroza `[一作]` (Pontificia Universidad Católica de Chile), Jorge F Silva `[通讯]` (National Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大语言模型层级相关性评估方法进行批判，证明余弦相似度不可靠，并提出直接基于层去除后模型准确率下降的“准确率相关性”指标，并在多种 LLM 与任务上验证其有效性。

**💡 创新点**

创新点在于：①提供理论证明，说明即使层的余弦相似度极低，该层仍可能对性能至关重要；②提出基于实际准确率下降的层相关性度量；③在结构化剪枝任务中，该度量在任务相关和任务无关两种场景均优于现有所有方法。

**🔧 技术方法**

主要技术包括：Transformer 预训练模型（Mistral、Pythia、OLMo、LLaMA‑3‑8B、Mistral‑7B 等）；层去除后准确率评估；余弦相似度与输出余弦/范数/散度等多种基线指标；迭代贪心剪枝；以及与“Healing”微调相结合的实验。

**📊 数据集**

使用的数据集有十个多任务基准（C4、CodeAlpaca、LIMA、MathInstruct、BoolQ、ARC‑Challenge、ARC‑Easy、HellaSwag、PIQA、Winogrande），以及 MMLU 等额外评测集，用于任务相关与任务无关剪枝实验。

**📈 对比分析**

与余弦相似度、Taylor 近似、输出余弦/范数/散度、Perplexity、Slice‑GPT 等方法对比，准确率相关性在 25% 剪枝时平均提升 3–5%（部分任务甚至超过未剪枝模型），在任务无关剪枝中同样取得最高平均分；在 50% 及 75% 剪枝时差距更大。计算成本约为 4–5 小时/模型，显著高于余弦相似度的几分钟。

**⚠️ 局限性**

局限性包括：①计算开销大，需要 N×T 次前向推理；②对校准集选择高度敏感，单一任务的校准易导致其它任务性能下降；③采用贪心迭代策略，未能搜索最优层组合；④在极低剪枝比例下“Healing”会缩小方法间差异；⑤理论最坏情况证明虽有助说明问题，但在实际模型中表现可能受限。

---

## 65. SimPersona: Learning Discrete Buyer Personas from Raw Clickstreams for Grounded E-Commerce Agents

**arXiv ID:** 2605.14205 | [PDF](https://arxiv.org/pdf/2605.14205v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 66. Sheaf-Theoretic Transport and Obstruction for Detecting Scientific Theory Shift in AI Agents

**arXiv ID:** 2605.14033 | [PDF](https://arxiv.org/pdf/2605.14033v1)

**作者:** David N. Olivieri `[一作]` (Universidade de Vigo), Roque J. Hernández `[通讯]` (Universidade de Vigo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种有限的层次理论框架，用于检测科学理论的转变，特别是在人工智能代理中如何识别现有表征框架是否适用于新的情境，或是否需要扩展其语言。

**💡 创新点**

创新点在于将科学理论转变视为一个有限的诊断问题，提出了表征星座的概念，并在有限的层次理论背景下形式化了运输、限制、粘合和障碍等概念。

**🔧 技术方法**

使用了有限的层次理论和计算诊断方法，结合了局部到全局的结构来评估理论转变的候选者。

**📊 数据集**

使用了控制的过渡卡片基准，这些卡片设计用于区分源语言中的变形与语言扩展。

**📈 对比分析**

通过比较不同候选者的障碍签名，评估了候选者的表现，结果表明，预期的变形或扩展通常是最低障碍的候选者，且在基准测试中成功区分了过渡类型。

**⚠️ 局限性**

限制在于该研究并未尝试重建历史范式转变或解决开放式的自主理论发明问题，而是专注于识别人工智能代理在表征运输失败时何时需要扩展的有限诊断子问题。

---

## 67. What Should Explanations Contain? A Human-Centered Explanation Content Model for Local, Post-Hoc Explanations

**arXiv ID:** 2605.14207 | [PDF](https://arxiv.org/pdf/2605.14207v1)

**作者:** Helmut Degen `[一作]` `[通讯]` (Siemens Research & Predevelopment), Helmut Degen (Siemens Research & Predevelopment)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在六个工业AI用户研究中，对325个意义单元进行混合归纳-演绎的内容分析，构建了一个包含14个解释内容代码的模型，并通过专家评审和可靠性检验确认其内容充足性和可复制性。

**💡 创新点**

①提出了以解释内容为单位的“人本化解释内容模型”，填补了从解释类型到界面实现的细粒度缺口；②将规则、因果、认知（实际）和认知（相似）四大组别进行系统组织；③加入理论上重要但在语料中缺失的规则基础和反向what‑if扩展。

**🔧 技术方法**

混合归纳-演绎定性内容分析、专家面板评估（I‑CVI、S‑CVI/Ave）、可靠性评估（Krippendorff α、Cohen κ）以及覆盖性评估（将代码映射到六大解释类型）。

**📊 数据集**

325个意义单元，来源于六项工业场景（建筑技术、制造、AI软件开发、医院网络安全）用户研究，涵盖不同用户角色、生命周期阶段和信息来源。

**📈 对比分析**

通过可靠性检验得到Krippendorff α = 0.920、Cohen κ = 0.920，均远高于0.800阈值；专家评审显示所有代码在相关性、边界清晰度和可理解性上均满足I‑CVI ≥ 0.82，S‑CVI/Ave分别为0.93、0.92、0.94。

**⚠️ 局限性**

①归纳阶段由单一研究者完成，可能存在主观性；②语料不均衡，部分代码仅在少数研究中出现；③Rule base和What‑if backward两代码缺乏经验数据，仍需后续验证；④未对分割步骤进行可靠性评估；⑤未对模型在真实系统中的行为效果进行验证。

---

## 68. Multiple-Bases Belief Propagation List Decoding for Quantum LDPC Codes

**arXiv ID:** 2605.14170 | [PDF](https://arxiv.org/pdf/2605.14170v1)

**作者:** Sheida Rabeti `[一作]` (Northeastern University), Hessam Mahdavifar `[通讯]` (Northeastern University)

**通讯引用:** 1004 | [OpenAlex ID](https://openalex.org/A5056299839)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于Belief Propagation的多基准列表解码器 MBBP-LD，用于量子低密度奇偶校验（QLDPC）码。

**💡 创新点**

创新点在于通过 Tanner 图的无环子树分解构造结构化冗余校验矩阵，产生解码多样性，从而在保持 BP 线性复杂度的前提下显著提升解码性能，并避免了随机行复制或超线性后处理。

**🔧 技术方法**

使用的技术包括：标准 BP（归一化最小-求和）、增广校验矩阵（APC）、子树分解得到的多重校验基、并行 BP 实例、列表决策（频率加权评分）以及简单的硬判决。

**📊 数据集**

实验数据集为三类 QLDPC 码：bivariate bicycle 码 [[144,12,12]]、[[288,12,18]] 以及 B1 码 [[882,24,18≤d≤24]]。

**📈 对比分析**

通过与 BP、BP-Serial、BP-OSD、BPGD（并行和串行）对比，MBBP-LD 在低至中等错误率下的逻辑误码率（LER）比 BP-OSD 低 3–49%，比 BPGD（并行）低 7–36%，并保持与标准 BP 相同的并行延迟，平均迭代次数仅略高。

**⚠️ 局限性**

局限性包括：需要先行构造子树分解，构造复杂度与图结构相关；对高密度或高码率 QLDPC 的适用性尚未验证；仍依赖 BP 的收敛性，未能完全消除所有退化情况。

---

## 69. Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations

**arXiv ID:** 2605.14175 | [PDF](https://arxiv.org/pdf/2605.14175v1)

**作者:** Qisong He `[一作]` (University of Liverpool), Xiaowei Huang `[通讯]` (University of Liverpool)

**通讯引用:** 10748 | [OpenAlex ID](https://openalex.org/A5015499043)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个运行时验证器，用来检查LLM生成的下一句是否在先前对话的承诺与证据基础上。

**💡 创新点**

将对话更新映射到八种操作，融合动态认识论、可归纳推理、意识逻辑与论证四种形式学，维护显式依赖图，实现线性时间、无冲突的回收与重推。

**🔧 技术方法**

采用动态认识论（DEL）模型、Dung式论证框架、知识与承诺记录、符号引擎、LLM解释器以及依赖图遍历技术。

**📊 数据集**

使用LongMemEval‑KU（78项）、LoCoMo（60项）以及两套多智能体情景、Phase 2/3对话和50项定位测试等数据集。

**📈 对比分析**

与LLM‑only和匹配检索‑RAG基线对比，Verifier在LongMemEval上从88.5%提升至89.7%，在LoCoMo与RAG相当；在stale‑claim子集提升6.7pp；回收时间线性于论证规模，显著快于历史回放。

**⚠️ 局限性**

依赖LLM解释器的抽取精度（模型规模影响分类准确）、LLM API延迟主导端到端时延、尚未处理解释器输入被破坏的情况。

---

## 70. Distribution Corrected Offline Data Distillation for Large Language Models

**arXiv ID:** 2605.14071 | [PDF](https://arxiv.org/pdf/2605.14071v1)

**作者:** Yumeng Zhang `[一作]` (George Mason University), Zhuangdi Zhu `[通讯]` (George Mason University)

**通讯引用:** 1293 | [OpenAlex ID](https://openalex.org/A5079428801)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种纯离线的推理蒸馏框架，通过对教师生成的推理轨迹进行分布修正，使得学生模型在训练时更贴近推理时的分布，从而提升小模型的推理准确性。

**💡 创新点**

创新点在于利用学生对教师推理轨迹的支持比（学生/教师概率比）对每个标记进行动态加权，采用sigmoid转化并逐标记近似，既消除了长序列的高方差，又能抑制长度偏差，解决了传统离线蒸馏中教师-学生分布漂移的问题。

**🔧 技术方法**

主要技术包括：教师-学生支持比估计、sigmoid加权、逐标记重权、基于交叉熵或KL散度的损失框架，以及与现有SFT、KL、ToDi、GKD等方法的兼容性。

**📊 数据集**

使用了多种数学推理基准：GSM8K、MATH、MATH500、OlympiadBench、Omni-Math、AMC23，训练数据来源于Qwen2.5系列教师在结构化推理提示下生成的轨迹，并对最终答案进行正确性过滤。

**📈 对比分析**

与离线蒸馏基线（SFT、KL、ToDi）以及在线学生-rollout蒸馏（GKD）对比，实验显示该方法在MATH500、GSM8K、OlympiadBench等任务上显著提升了最终答案准确率，尤其在长推理轨迹和跨域测试中表现更好；同时在推理轨迹质量、分布漂移指标（ExAccErr）和自动/人工评价中也优于传统方法。

**⚠️ 局限性**

局限性包括：仍依赖教师生成的高质量轨迹，无法处理教师不可用或质量低下的情况；对极短推理任务的收益有限；加权机制需要温度和sigmoid参数调优，可能对不同模型/任务产生不同的最佳配置；以及在完全无教师场景下的适用性尚未验证。

---

## 71. ProtoMedAgent: Multimodal Clinical Interpretability via Privacy-Aware Agentic Workflows

**arXiv ID:** 2605.14113 | [PDF](https://arxiv.org/pdf/2605.14113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 72. Thinking Ahead: Prospection-Guided Retrieval of Memory with Language Models

**arXiv ID:** 2605.14177 | [PDF](https://arxiv.org/pdf/2605.14177v1)

**作者:** Harshita Chopra `[一作]` (University of Washington), Chirag Shah `[通讯]` (University of Washington)

**通讯引用:** 6443 | [OpenAlex ID](https://openalex.org/A5064398705)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于前瞻模拟的检索框架 PGR，解耦记忆存储与检索，并创建了 MemoryQuest 评测基准。

**💡 创新点**

创新点在于利用人类前瞻记忆机制生成未来模拟步骤作为检索子查询，突破传统语义相似检索的局限，能够检索到语义远离当前查询的“潜在”记忆。

**🔧 技术方法**

使用了迭代的 Simulate→Retrieve→Refine 循环，Tree‑of‑Thought / Chain‑of‑Thought 生成检索子查询，LLM 进行记忆提取与更新，以及向量/图检索等技术。

**📊 数据集**

评估使用了三大公开基准：MemoryQuest（多会话、低语义相似）、ImplexConv 和 PersonaMem。

**📈 对比分析**

通过与 Mem0、GraphRAG、TaciTree 等现有检索/推理基线对比，PGR 在 MemoryQuest 上召回率提升近 3 倍，整体检索召回率达到 0.723，回答质量赢率超过 90%，人类评估亦显著偏好 PGR。

**⚠️ 局限性**

局限性在于迭代检索过程需要多次 LLM 调用，计算成本较高，实时性受限，未来可通过训练轻量检索器降低开销。

---

## 73. Privacy Preserving Multi Agent Path Finding

**arXiv ID:** 2605.14119 | [PDF](https://arxiv.org/pdf/2605.14119v1)

**作者:** Rotem Lev Lehman `[一作]` (Ben Gurion University of Negev), Guy Shani `[通讯]` (Ben Gurion University of Negev)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在多代理路径规划（MAPF）中实现规划层和执行层隐私保护的框架，并给出了 k-隐私保留的规划器和后处理算法。

**💡 创新点**

创新点在于：①通过引入“假代理”实现多代理间的 k-隐私，②将视场冲突纳入规划，形成执行层 k-隐私，③设计安全区（safe‑zone）后处理，使得在不破坏隐私的前提下提升解的质量。

**🔧 技术方法**

核心技术包括：mock 代理构造、基于 PIBT 和 LaCAM* 的隐私兼容规划、FoV 冲突检测、SIPP 规划器用于安全区内部单代理规划。

**📊 数据集**

使用标准的网格基准数据集（如 brc202d、random‑64‑64‑20 等共12张地图）进行评估，覆盖不同规模和稀疏度。

**📈 对比分析**

实验对比使用 PIBT 与 LaCAM* 的解质量、可解实例数、RSoC 以及运行时。结果表明：k 值越大可解实例数下降，但后处理能提升约 5–30% RSoC；LaCAM* 在复杂实例上更稳健；FoV 半径增大对可解实例有显著负面影响。

**⚠️ 局限性**

局限性包括：①规划器非完备且不保证最优；②假代理生成和冲突检测在大规模高 k 情况下成本高；③视场冲突模型假设 FoV 对称且邻接，未考虑更复杂感知；④仅在网格地图上验证，非结构化图的表现未知。

---

## 74. Non-Redundancy of Low-Arity Symmetric Boolean CSPs

**arXiv ID:** 2605.14007 | [PDF](https://arxiv.org/pdf/2605.14007v1)

**作者:** Amatya Sharma `[一作]` (University of Michigan), Santhoshini Velusamy `[通讯]` (University of Waterloo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了对称布尔谓词的非冗余性，给出了其最大非冗余实例规模的渐近上界与下界，并完成了所有 arity ≤4 的完全分类；对 arity =5 只剩两种谓词未决定上界和下界；

**💡 创新点**

提出 t‑平衡性（lifted balancedness）与捕获多项式的对应关系，用以实现多项式上界；利用 Carbonnel 的通用 k‑立方操作判定下界；通过组合实验与代数判定实现几乎完整的分类，首次将两种技术统一用于同一问题；

**🔧 技术方法**

高阶平衡性与多项式捕获、矩阵 Smith 标准形判定、通用 k‑立方模式与部分多项式保真、组合枚举与整数线性规划；

**📊 数据集**

枚举所有小阶对称谓词（即权重集合）作为实验对象；不使用外部真实数据集；

**📈 对比分析**

与已知的 sparsification、approximate sparsification、流式复杂度等相关研究结果对照，结果显示：对所有 arity ≤4 的谓词，上界与下界完全匹配；对 arity =5 的两种谓词得到上界 O(n³) 与下界 Ω(n²) 的区间；

**⚠️ 局限性**

尚未确定 arity =5 中两种谓词的精确指数，表明 t‑平衡性与通用 k‑立方框架的局限性；对非对称谓词缺乏统一的代数下界方法。

---

## 75. R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning

**arXiv ID:** 2605.14026 | [PDF](https://arxiv.org/pdf/2605.14026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 76. Diagnosing Training Inference Mismatch in LLM Reinforcement Learning

**arXiv ID:** 2605.14220 | [PDF](https://arxiv.org/pdf/2605.14220v1)

**作者:** Tianle Zhong `[一作]` (ByteDance), Xiao Yu `[通讯]` (ByteDance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个零失配（zero‑mismatch）回放引擎，系统消除训练与推理阶段的概率不一致（Training‑Inference Mismatch，TIM），并用它来诊断大型语言模型 RL 训练的稳定性问题。

**💡 创新点**

创新点在于：①创建了可实现位级概率对齐的轻量回放引擎；②系统性地将 TIM 与传统 PPO / GRPO 训练拆解，揭示其对优化目标的根本性影响；③提出了多层级（token‑level 与 sequence‑level）算法补偿方案，并与零失配基准进行对照，证明可近似恢复正常训练。

**🔧 技术方法**

使用了基于 HuggingFace 的统一模型实现、批量不变（batch‑invariant）GPU 内核、FSDP 分布式训练、VLLM 等推理框架、RMSNorm、MoE 专用融合内核、CUDAGraph、Pipeline Parallelism 等系统技术，以及 truncated importance sampling（TIS）和序列级拒绝采样等算法补偿。

**📊 数据集**

在 Qwen3-1.7B（dense）和 Qwen3-30B-A3B（MoE）模型上，分别在 Sanity‑Test-R1D-1.5B 和 DAPO 任务上训练，并在 AIME‑2024 任务上评估验证性能。

**📈 对比分析**

与传统 vLLM 非零失配回放相比，零失配引擎在 REINFORCE 和 GRPO 训练中显著提升奖励与梯度稳定性，恢复并超越了原始算法；在引入 TIS + 序列拒绝采样后，算法补偿能够逼近零失配基准的性能。

**⚠️ 局限性**

局限性包括实验规模与配置有限，只评估了少数模型、任务与系统组合；算法补偿仍需手动调节阈值，未证明在更广泛的 RL 场景下具有通用性；并未深入分析补偿可能引入的其他优化副作用。

---

## 77. LLM-Based Robustness Testing of Microservice Applications: An Empirical Study

**arXiv ID:** 2605.14202 | [PDF](https://arxiv.org/pdf/2605.14202v1)

**作者:** Hrushitha Goud Tigulla `[一作]` (University of North Carolina at Charlotte), Marco Vieira `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 5469 | [OpenAlex ID](https://openalex.org/A5016622594)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究利用大语言模型自动生成微服务的鲁棒性测试用例，系统评估不同模型与提示策略对失败模式覆盖率与多样性的影响。

**💡 创新点**

创新点在于发现提示策略对多样性影响远大于模型规模，提出嵌入变异分类的 Guided 与 GuidedFewShot 提示，突破结构化提示导致的多样性崩溃，并揭示域知识需配合示例才能发挥作用。

**🔧 技术方法**

采用 Llama 2 系列 14B/32B/70B 参数模型，实施 ZeroShot、Structured、FewShot、CoT、Self-Refine、Guided、GuidedFewShot 等提示工程技术，并用 Jaccard 相似度衡量生成测试的失败模式多样性。

**📊 数据集**

实验使用两款公开微服务基准：Java 单语系 TeaStore（6 服务）和多语系 OTel Astronomy Shop（27 服务），共生成 663 条测试用例并验证 23 条失败模式。

**📈 对比分析**

通过单跑覆盖率、联合覆盖率和平均 Jaccard 相似度进行比较，结果显示单模型多提示可实现 100% 覆盖，最高单跑覆盖率为 57%，结构化提示导致多样性完全崩溃（J=1.00），而 GuidedFewShot 提供最佳单跑覆盖率与最低相似度。

**⚠️ 局限性**

研究局限在于每个配置仅执行一次，缺乏方差估计；使用公开基准可能不具行业代表性；失败模式定义为后验，可能忽视某些错误类别且不完全可验证。

---

## 78. CineMesh4D: Personalized 4D Whole Heart Reconstruction from Sparse Cine MRI

**arXiv ID:** 2605.13994 | [PDF](https://arxiv.org/pdf/2605.13994v1)

**作者:** Xiaoyue Liu `[一作]` (National University of Singapore), Lei Li `[通讯]` (National University of Singapore)

**通讯引用:** 12279 | [OpenAlex ID](https://openalex.org/A5100440407)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种端到端的CineMesh4D框架，能够直接从稀疏多视角心脏cine MRI重建高质量的4D（3D+时间）全心表面网格。

**💡 创新点**

创新点包括：①基于Beer–Lambert定律的可微渲染损失，实现二维切片轮廓对三维网格的直接监督；②双上下文时序块，将全周期全局信息与局部帧间一致性融合，提升运动连贯性；③无中间分割的完整端到端映射，减少错误累积。

**🔧 技术方法**

使用技术包括：多视角U‑Net特征提取、MeshVAE图形自编码器、GCN+Exphormer图注意力、LoRA参数高效微调、可微渲染损失、双上下文时序块（全局自注意力+局部滑窗融合）以及差分渲染的边界损失。

**📊 数据集**

采用了222名受试者的标准多视角心脏cine MRI数据（每个视角25帧），经过中心裁剪、自动分割并人工校正，生成高分辨率全心三维分割和网格；训练集155例，验证集10例，测试集57例。

**📈 对比分析**

与HybridVNet、PointNet++、PU‑Net、CPD、MR‑Net等方法对比，CineMesh4D在全心和各腔室的MAE、MSE、Chamfer距离、Hausdorff距离、边界F‑score、平均轮廓距离等指标均表现最优；帧推理时间<0.1 s，显著快于CPD等传统方法。

**⚠️ 局限性**

局限性包括：对稀疏LAX视图的依赖导致腔室尤其是心房的重建精度受限；目前仅在健康或常规病例上验证，缺乏病理性数据与多模态（CT、超声）跨域泛化；模型对极端心率或严重畸形的鲁棒性尚未充分评估。

---

## 79. Making OpenAPI Documentation Agent-Ready: Detecting Documentation and REST Smells with a Multi-Agent LLM System

**arXiv ID:** 2605.14312 | [PDF](https://arxiv.org/pdf/2605.14312v1)

**作者:** Rayfran Rocha Lima `[一作]` (Sidia Institute of Technology), Thiago Medeiros de Menezes `[通讯]` (Sidia Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对工业API生态进行文档和REST相关缺陷的规模评估，并使用Hermes多代理LLM系统检测端点级别的文档与REST smells，验证其对AI代理的影响，并依据评估结果制定阶段性采用策略。

**💡 创新点**

提出Hermes系统将多代理LLM与端点级别的文档/REST smell分类相结合，首次在工业规模下系统性评估OpenAPI文档的“代理就绪”程度，并将评估嵌入治理流程，展示文档质量是AI代理集成的关键瓶颈。

**🔧 技术方法**

使用大型语言模型（gpt-oss:120b）+多代理架构、OpenAPI规范解析、REST与文档 smell分类、Prompt工程、专家标注、成本估算等技术。

**📊 数据集**

600个生产API端点（16个API，约600个endpoint）及其10%随机子集60个端点用于黄金标准标注。

**📈 对比分析**

对7种LLM进行多标签分类评估，使用Jaccard、F1_micro、F1_macro、Hamming Loss和Cardinality Diff；gpt-oss:120b得到Jaccard 0.85、F1_micro 0.92、F1_macro 0.73、Hamming 0.07，优于其他模型；在全景评估中发现平均4.08 smell/endpoint；对比全系统改造（385h）与选择性改造（42h）显示成本大幅下降。

**⚠️ 局限性**

研究仅在单一工业R&D机构、内部微服务环境进行，评估依赖内部LLM且对POC实验覆盖面有限；未能证明因果关系；模型对Prompt和配置敏感，未来LLM改进可能改变部分缺陷，但未在本研究中验证。

---

## 80. Pluot: Towards 'write once, run everywhere' visualization software

**arXiv ID:** 2605.14118 | [PDF](https://arxiv.org/pdf/2605.14118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 81. Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use

**arXiv ID:** 2605.14038 | [PDF](https://arxiv.org/pdf/2605.14038v1)

**作者:** Yize Cheng `[一作]` (University of Maryland), Soheil Feiz `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于模型自身能力的工具必要性定义，并对四个大型语言模型在算术与事实问答任务中的工具调用行为进行评估

**💡 创新点**

创新点在于将工具必要性与模型的经验性能相结合，揭示了显著的知道‑做不匹配（knowing‑doing gap），并通过两阶段（认知‑执行）模型对失配原因进行细致诊断

**🔧 技术方法**

使用线性探针对隐藏状态进行可解释性分析，计算工具必要性与工具调用意图的可分离度及其余弦相似度，并追踪样本在认知→执行过程中的轨迹

**📊 数据集**

使用自构建的 4000 题算术数据集与 TruthfulQA 817 题事实问答数据集

**📈 对比分析**

对比模型自我认知与实际调用行为，发现工具必要性与行动匹配率在算术任务为 26.5–54.0%，在问答任务为 30.8–41.8%；线性探针在大多数层级显示工具必要性和行动均可线性可分，但在最后层/最后令牌位置两者正交，导致大部分错误产生于认知→执行转换

**⚠️ 局限性**

局限性包括仅涵盖算术与事实问答两类任务，使用贪心解码可能影响真实场景行为，样本量和模型数目有限，未深入探究不同工具类型或更复杂任务中的表现

---

## 82. The Evaluation Trap: Benchmark Design as Theoretical Commitment

**arXiv ID:** 2605.14167 | [PDF](https://arxiv.org/pdf/2605.14167v1)

**作者:** Theodore J Kalaitzidis `[一作]` `[通讯]`, Theodore J Kalaitzidis

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种从能力声明到评估标准的可重复审计流程（Epistematics），并以“自主学习”为案例对现有基准进行评估与诊断。

**💡 创新点**

创新点在于构建了失效模式分类和设计准则，使评估框架能够在能力声明层面检验与评估的一致性，从而揭示评价陷阱。

**🔧 技术方法**

主要技术是理论推理与系统化审计流程：拆解能力声明、抽取理论假设、推导体系架构需求，并进行判别性有效性测试。

**📊 数据集**

论文以理论案例分析为主，并提出开放世界交互式环境（如大型沙盒游戏）作为潜在评估数据集，但未使用具体公开数据集进行实验。

**📈 对比分析**

通过对比现有基准的“转移性假设”和“行为近似”失效模式与 Epistematics 的判别性测试，说明传统基准无法区分代理行为与真正能力，表现为评估失效。

**⚠️ 局限性**

局限在于需对理论假设进行准确解释、依赖专家共识，且尚未在大规模实验中验证其可行性，不能直接替代现有基准。

---

## 83. A Non-Destructive Methodological Framework for Modernizing Legacy Clinical Reporting Systems for AI-Driven Pharmacoinformatics: A SAS Case Study

**arXiv ID:** 2605.13905 | [PDF](https://arxiv.org/pdf/2605.13905v1)

**作者:** Jaime Yan `[一作]` (Harrisburg University of Science and Technology), Jaime Yan `[通讯]` (Harrisburg University of Science and Technology)

**通讯引用:** 135 | [OpenAlex ID](https://openalex.org/A5107920691)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种非破坏性方法框架，使老旧 SAS 临床报告系统通过元数据层实现 AI 驱动的药物信息学，无需改动原始源代码。

**💡 创新点**

创新点在于构建桥接映射、统一的中间表示 (IR)、以及可选增量合并模式，同时实现了机器可读输出和完整的差异验证流程。

**🔧 技术方法**

使用技术包括 SAS 宏层重构、YAML 配置驱动、Python 任务编排、JSON IR、以及 LLM（Claude Opus）进行自动摘要和异常检测。

**📊 数据集**

使用的数据集包括 558 组件工业 SAS 宏库（372,698 行）以及公开的 CDISC CDISCPilot01 试验数据。

**📈 对比分析**

通过 14 个内部报告和 5 个公开报告的细胞级别平行验证，得到 80%+ 的细胞级一致率（平均 82.7%）和 100% 公共数据匹配，证明方法在现代化与 AI 兼容性方面表现优异。

**⚠️ 局限性**

局限在于部分报告仍受表格布局差异影响，LLM 评估仅在单一模型上进行，未做跨模型比较，且缺乏对 ARS 完整映射的支持。

---

## 84. Unsupervised learning of acquisition variability in structural connectomes via hybrid latent space modeling

**arXiv ID:** 2605.13933 | [PDF](https://arxiv.org/pdf/2605.13933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 85. Motion Planning for Autonomous Vehicles using Optimization over Graphs of Convex Sets

**arXiv ID:** 2605.14199 | [PDF](https://arxiv.org/pdf/2605.14199v1)

**作者:** Matheus Wagner `[一作]` (Federal University of Santa Catarina), Antônio Augusto Fröhlich `[通讯]` (Federal University of Santa Catarina)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

基于GCS（Convex Sets图）与Bezier曲线的多目标轨迹规划框架，结合时间缩放实现自适应速度控制。

**💡 创新点**

创新点在于将非凸自由空间通过可直接优化的有向图分解为凸子域，车辆动力学用简化的双轮小滑移模型实现可凸化的动态约束，并通过时间分段约束实现对动态障碍物的简化处理。

**🔧 技术方法**

使用技术包括：GCS混合整数凸优化、Bezier曲线参数化、时间缩放多项式、简化双轮动力学与差分平滑约束、CommonRoad场景集与IPOPT求解器。

**📊 数据集**

使用数据集：CommonRoad的三种典型场景（静态障碍规避、车道变换、超车）。

**📈 对比分析**

对比方法：将GCS方案与直接离散时间非线性最优控制（NLP）方案在IPOPT下求解，实验显示GCS在所有场景下约快10倍，轨迹相似但纵向加速度略激进，表现出更好的收敛鲁棒性。

**⚠️ 局限性**

局限性：对加速度约束的凸化不足导致动力学可行性不如NLP；动态障碍物的时间约束为手工启发式，缺乏系统化生成；仅适用于小滑移线性轮胎模型，未验证更复杂动力学或高速场景。

---

## 86. D2-CDIG: Controlled Diffusion Remote Sensing Image Generation with Dual Priors of DEM and Cloud-Fog

**arXiv ID:** 2605.14326 | [PDF](https://arxiv.org/pdf/2605.14326v1)

**作者:** Zuopeng Zhao `[一作]` (China University of Mining and Technology), Maocai Ning `[通讯]` (China University of Mining and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于扩散模型的遥感图像生成框架 D2-CDIG，能够通过双重先验（数字高程模型 DEM 与云雾信息）实现对地表与大气的精确控制。

**💡 创新点**

创新点在于双分支控制网络（地面分支和大气分支）以及云密度滑块，既实现了地形与云雾的解耦生成，又支持用户可调的云层厚度与分布。

**🔧 技术方法**

使用 Stable Diffusion v1.5 作为主体，结合 ControlNet 架构、CNN（ResNet-18）和 Transformer（ViT-Small）编码器，加入层级注入、联合损失和云密度映射。

**📊 数据集**

采用 Landsat‑8、Sentinel‑2 以及 Google Maps 高分辨率影像与 Copernicus GLO‑30 DEM 的多源数据集，涵盖文本、DEM、云三种条件生成任务。

**📈 对比分析**

与 SD1.5、ControlNet、DiffusionSat、CRS‑Diff、T2I‑Adapter 等基线相比，D2-CDIG 在 SSIM、PSNR、FID、LPIPS 等指标上均显著提升（例如 Task 2 的 FID 降至 59.97、SSIM 提升至 0.303），并在云雾覆盖不同水平下保持更稳定的性能。

**⚠️ 局限性**

主要限制包括模型参数量和显存需求大幅提升（约 2.3× 训练/推理成本），对真实云掩模的域间泛化仍有一定误差，且在极端高分辨率或特殊地形（如悬崖、建筑物高度）下可能出现平滑或缺失细节。

---

## 87. A Hormone-inspired Emotion Layer for Transformer language models (HELT)

**arXiv ID:** 2605.13858 | [PDF](https://arxiv.org/pdf/2605.13858v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 88. WarmPrior: Straightening Flow-Matching Policies with Temporal Priors

**arXiv ID:** 2605.13959 | [PDF](https://arxiv.org/pdf/2605.13959v1)

**作者:** Sinjae Kang `[一作]` (KAIST), Kimin Lee `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过将生成式策略的源分布从无信息的高斯改为基于最近动作历史的温暖先验（WarmPrior），显著提升机器人操作任务的成功率。

**💡 创新点**

创新点在于提出一种只改源分布的轻量化方法，提供时间相关的先验，既能直线路径、降低分支成本，又兼具可调的多模态与时序一致性。

**🔧 技术方法**

使用流匹配（flow‑matching）与扩散策略的生成式控制框架，配合自回归的WarmPrior以及残差探索的DSRL。

**📊 数据集**

在Robomimic、MimicGen以及Franka Research 3真实机器人的数据集上进行实验。

**📈 对比分析**

相较于基线𝒩(0,I)的流匹配策略，WarmPrior在多种任务与不同采样预算下提升1–3倍的成功率，并在强化学习微调中实现更快收敛和更高终值。

**⚠️ 局限性**

局限在于对先验均值的依赖，若动作历史不稳定或任务极为随机，σ调参需谨慎；此外在极高维或复杂多模态场景下仍需进一步验证。

---

## 89. Semantic Feature Segmentation for Interpretable Predictive Maintenance in Complex Systems

**arXiv ID:** 2605.14318 | [PDF](https://arxiv.org/pdf/2605.14318v1)

**作者:** Emilio Mastriani `[一作]` (INAF, Osservatorio Astrofisico di Catania), Sebastiano Spinello `[通讯]` (INAF, Osservatorio Astrofisico di Catania)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了基于语义特征分割的预测维护框架，将监测特征划分为可解释的规范空间和残余空间，并评估其对故障预测的影响。

**💡 创新点**

创新点在于利用领域知识手动定义语义段，将特征按功能分组，形成可解释的规范子空间，并证明其在降低预测风险方面优于残余空间。

**🔧 技术方法**

采用Spearman相关、MST冗余消除、时间感知交叉验证、Log‑loss评估以及LGBM/XGBoost/Random Forest等机器学习技术。

**📊 数据集**

使用Apache Cassandra容器化集群的监控数据（约2.1万条记录、83个特征）以及DISFI注入的文件描述符耗尽故障事件。

**📈 对比分析**

通过与完整特征、PCA压缩以及残余空间进行对比实验，规范空间的Log‑loss最低且与PCA相近，残余空间接近无信息基准，显示出竞争性预测性能。

**⚠️ 局限性**

局限性包括仅在单一系统与有限故障样本上验证，语义分割需人工设定，缺乏自动化与信息理论分析，且结果尚未在更广泛的系统上验证。

---

## 90. Reactive Planning based Control for Mobile Robots in Obstacle-Cluttered Environments

**arXiv ID:** 2605.14232 | [PDF](https://arxiv.org/pdf/2605.14232v1)

**作者:** Li Tan `[一作]` (University of Science and Technology of China), Wei Ren `[通讯]` (Dalian University of Technology)

**通讯引用:** 12772 | [OpenAlex ID](https://openalex.org/A5007265691)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种反应式规划与自适应跟踪控制策略（RPCS），实现移动机器人在部分已知障碍环境中从起点安全到达终点。

**💡 创新点**

创新点在于：①利用三次多项式局部修改参考轨迹，避免搜索/采样；②采用离散化与优化的自适应跟踪控制，保证轨迹平滑且方向连续；③通过并行迭代执行规划与控制，显著降低计算时间与内存需求。

**🔧 技术方法**

主要技术包括：局部坐标变换、三次多项式生成转弯轨迹、离散化分段跟踪、二次优化求解控制输入、并行迭代的反应式规划与跟踪框架。

**📊 数据集**

实验采用仿真生成的障碍密集环境，没有使用公开数据集；通过对比实验验证方法有效性。

**📈 对比分析**

在简化环境中与 LSCM、HPPM、CLF‑CBF、CILQR、DWA 等方法比较，RPCS 在控制努力、轨迹长度、内存占用、规划时间和总计算时间等指标上均表现最佳，尤其在计算资源与时间上具有显著优势。

**⚠️ 局限性**

局限性包括：仅考虑单机器人，假设障碍为凸形且感知半径满足 γ>r+max r_k；参数（ρ、δ、α、ε）对性能影响显著，需要经验调优；在动态或极度复杂环境下，仍可能出现规划失败或性能下降。

---

## 91. Univariate Bicycle Quantum LDPC Codes: Explicit Logical Structure and Distance Bounds

**arXiv ID:** 2605.14173 | [PDF](https://arxiv.org/pdf/2605.14173v1)

**作者:** Sheida Rabeti `[一作]` (Northeastern University), Hessam Mahdavifar `[通讯]` (Northeastern University)

**通讯引用:** 1004 | [OpenAlex ID](https://openalex.org/A5056299839)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了“单变量自行车”UB 量子 LDPC 码，并给出了其构造、逻辑基以及最小距离上界的完整理论分析；

**💡 创新点**

创新点在于通过 Frobenius 关系将双多项式搜索简化为单多项式搜索，同时显式构造逻辑 coset 基，利用循环矩阵的循环密度给出可计算的距离上界；

**🔧 技术方法**

采用了 CSS 码理论、循环矩阵与余子式分析、Frobenius 映射以及 BP‑OSD 解码算法的数值模拟；

**📊 数据集**

使用自行生成的 UB 码集（块长 100–1000，率 0.1–0.5，权重 6/8 等）以及文献中的 BB 与 GB 码作对比，并在 BSC（独立 X 错误）上进行仿真；

**📈 对比分析**

通过比较物理错误率与逻辑错误率曲线，UB 码在相同或更高码率、相同或更低稳定器权重下的逻辑错误率优于或持平于 BB/GB 码；

**⚠️ 局限性**

局限在于只讨论除子情况，距离上界仍不一定紧，且对更大块长或非除子情况的推广尚未完成，需要进一步改进解码和距离分析。

---

## 92. Clustering with Locally Bounded Ignorance

**arXiv ID:** 2605.13917 | [PDF](https://arxiv.org/pdf/2605.13917v1)

**作者:** Jaroslav Garvardt `[一作]` (University of Jena), Christian Komusiewicz `[通讯]` (University of Jena)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了带有模糊边的相关聚类（Correlation Clustering）问题，并给出了以解空间大小 k 加上模糊边图的退化度 d 或闭包 c 为参数的多项式 kernel；同时证明了在若干特殊结构下该问题仍然没有多项式 kernel。

**💡 创新点**

创新点在于首次将模糊边图的局部结构（退化度和闭包）作为参数，设计了对应的 kernelization 规则，并证明了在这些参数下能够得到多项式上界；此外，还给出了对应的硬度下界，补全了之前对该问题参数化复杂度的空白。

**🔧 技术方法**

核心技术包括：基于退化度排序的关键团（critical clique）压缩规则、闭包属性下的“坏对”检测与处理、对大簇拆分为实边团与模糊边团的分析、以及一系列精细的边权重调整与集合覆盖式归约；所有步骤均在多项式时间内完成。

**📊 数据集**

本文为理论研究，不使用实验数据集，所有结论均通过严格的数学证明得到。

**📈 对比分析**

比较方法为与已知的基于 k 的 FPT/无多项式 kernel 结果对比，得到的 kernel 大小分别为 O(k^3 d) 和 O(k^2 c^2)，显著改善了在模糊边结构受限下的可处理规模；在硬度分析中，证明了在退化度、闭包均为常数且实边/非边结构受限时仍为 NP‑hard，进一步说明参数化选择的重要性。

**⚠️ 局限性**

局限性：得到的 kernel 上界仍相对较大（如 30k^3 d、30k^2 c^2），尚未达到最优；对于更常见的参数如 k+Δ（最大度）或 k+h‑index，目前仍未能确定是否存在多项式 kernel；此外，本文仅给出了理论结果，缺乏实验验证与实际运行时间分析。

---

## 93. Fusion-fission forecasts when AI will shift to undesirable behavior

**arXiv ID:** 2605.14218 | [PDF](https://arxiv.org/pdf/2605.14218v1)

**作者:** Neil F. Johnson `[一作]` (George Washington University), Frank Yingjie Huo `[通讯]` (George Washington University)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5007160706)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文展示了一种基于向量广义化的融合-裂变群体动力学模型，能够预测和解释ChatGPT等AI模型行为的转变，从可取行为转向不可取行为。

**💡 创新点**

创新点在于提出了一种新的数学模型，能够在不依赖于特定模型的情况下，实时预测AI行为的转变，并提供警告信号。

**🔧 技术方法**

使用了向量广义化的群体动力学模型，结合了深度学习中的残差流和注意力机制。

**📊 数据集**

使用了多个AI模型（124M到12B参数规模）进行验证，包括GPT-2和Pythia系列模型，以及来自十个前沿聊天机器人的数据。

**📈 对比分析**

通过六个独立测试验证了模型的有效性，包括在不同参数规模的七个AI模型上达到约90%的预测准确率，并在生产环境中表现出持续性。

**⚠️ 局限性**

限制在于当前模型未能完全消除AI行为转变的根本机制，且对某些特定情况的预测可能不够准确。

---

## 94. Rethinking the Good Enough Embedding for Easy Few-Shot Learning

**arXiv ID:** 2605.14145 | [PDF](https://arxiv.org/pdf/2605.14145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 95. Synthetic Sociality: How Generative Models Privatize the Social Fabric

**arXiv ID:** 2605.14090 | [PDF](https://arxiv.org/pdf/2605.14090v1)

**作者:** Ana Dodik `[一作]` (MIT CSAIL), Moira Weigel `[通讯]` (Harvard University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了关于生成模型如何自动化并私有化社会织物的理论框架，并引入了“合成社会性”概念，分析了其对社会关系的影响。

**💡 创新点**

将生成模型视为自动化社会能力而非仅仅是智力自动化；提出了使用与交换社会性的二分法，区分替代与中介自动化；概念化“合成社会性”作为一种由私有模型制造的社会织物。

**🔧 技术方法**

基于已有生成模型技术，如语言模型、图像生成器、聊天机器人等；采用文本分析、文献综述和案例研究。

**📊 数据集**

未使用具体数据集，主要基于公开论文、媒体报道、企业声明和用户研究。

**📈 对比分析**

未做实验性对比，主要通过案例分析与理论推导说明生成模型在替代和中介社会性方面的效果；未给出性能指标。

**⚠️ 局限性**

缺乏实证验证、对不同社会文化背景的适用性不足、对生成模型技术细节的技术实现不详；对伦理与公平问题的分析仍为理论层面，缺乏可操作的设计方案。

---

## 96. Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2605.14117 | [PDF](https://arxiv.org/pdf/2605.14117v1)

**作者:** Luis Lara `[一作]` (Mila Quebec AI Institute), Christopher Pal `[通讯]` (Mila Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于大型语言模型的文本生成方法，输入房间连通图和面积约束的JSON，输出符合约束且可直接导入CAD的多边形房间布局；通过两阶段训练（监督微调+强化学习可验证奖励）提升连通性与面积精度并消除重叠；并定义四个约束遵从度指标；

**💡 创新点**

创新点包括：①使用结构化JSON作为输入输出，提升可解析性与精度；②引入可验证奖励（连通性奖励、面积奖励）和RLVR（GRPO）训练策略，显著降低布局重叠和连通误差；③提出四个针对房间面积、标识、重叠与面积误差的评价指标，补全传统评测缺口；

**🔧 技术方法**

技术手段：Llama‑3.3‑70B‑Instruct微调、GRPO强化学习、best‑of‑10采样、可验证奖励设计、JSON结构化房间多边形生成与解析、房间面积与连通性自动评估；

**📊 数据集**

数据集：RPLAN（80,788+真实亚洲单层住宅平面图），通过自定义转换器将 raster 图像转为JSON多边形与连通图；

**📈 对比分析**

与House‑GAN、House‑GAN++、HouseDiffusion等方法对比，兼容度（Compatibility）从 2.5–5.5 降至 0.01–0.15（下降 94%）、多样性（Diversity）从 9.5–11.8 降至 7.0–9.0，真实感（Realism）为 0.028，显示在所有 5‑8 房间任务上实现显著性能提升；

**⚠️ 局限性**

限制：仍会在高房间数时出现少量重叠；推理需多次采样，计算成本高；仅在单层亚洲住宅样本上验证，可能不泛化至多层或不同建筑风格；RLVR 奖励仅覆盖连通性与面积，未覆盖循环、日照、建筑规范等实际可用性约束；

---

## 97. Self-Regulated Learning in Essay Writing: Consistency of Strategies and Impact on Outcomes

**arXiv ID:** 2605.14228 | [PDF](https://arxiv.org/pdf/2605.14228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 98. MoZoo:Unleashing Video Diffusion power in animal fur and muscle simulation

**arXiv ID:** 2605.13857 | [PDF](https://arxiv.org/pdf/2605.13857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 99. PEML: Parameter-efficient Multi-Task Learning with Optimized Continuous Prompts

**arXiv ID:** 2605.14055 | [PDF](https://arxiv.org/pdf/2605.14055v1)

**作者:** Anjir Ahmed Chowdhury `[一作]` (University of Houston), Feng Yan `[通讯]` (University of Houston)

**通讯引用:** 181008 | [OpenAlex ID](https://openalex.org/A5100384245)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种结合LoRA与PrefixNAS的参数高效多任务学习框架，以单一统一的提示结构与低秩权重调整同时完成多任务微调

**💡 创新点**

创新点在于通过可微分的神经架构搜索动态生成最优提示架构，并与LoRA协同训练，解决传统方法多任务提示不对齐、需切换多适配器以及训练时显存线性增长的问题

**🔧 技术方法**

核心技术包括LoRA低秩适配、PrefixNAS（梯度可微NAS）以及联合损失优化；实验中还使用了T5‑Large、FLAN‑T5‑Large、LLaMA‑7B及LLaMA2‑7B等预训练模型

**📊 数据集**

在GLUE、SuperGLUE、MMLU及多项常识推理数据集（PIQA、SIQA、Winogrande、HellaSwag、ARC等）上进行评估

**📈 对比分析**

与MTL‑LoRA、MultiLoRA、C‑Poly、MoE等SOTA多任务PEFT方法对比，平均准确率提升至约91.1%（LLaMA2‑7B），整体平均提升6.67%，单个任务最高提升10.75%

**⚠️ 局限性**

缺点包括：提示模块需要额外参数；NAS搜索阶段的计算和显存成本较高；且在极度稀疏或对齐性极差的任务上仍可能出现性能下降

---

## 100. GEAR: Genetic AutoResearch for Agentic Code Evolution

**arXiv ID:** 2605.13874 | [PDF](https://arxiv.org/pdf/2605.13874v1)

**作者:** Ahmadreza Jeddi `[一作]` (University of Toronto), Babak Taati `[通讯]` (University of Toronto)

**通讯引用:** 3407 | [OpenAlex ID](https://openalex.org/A5011257199)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 GEAR，一种基于遗传搜索的自动化研究框架，用来替代 AutoResearch 中的单个主导者爬山策略。

**💡 创新点**

通过维护一个由最佳、精简和多样化三类状态组成的前沿，使用突变和交叉组合多条研究路径，从而在长时间搜索中持续进展。

**🔧 技术方法**

采用遗传算法（突变、交叉）、前沿管理、角色分配、父母选择和可变搜索策略（提示、固定控制器、可进化控制器）等技术。

**📊 数据集**

在 GPT‑2 风格的语言建模任务上，使用 AutoResearch 原始数据集（语言建模数据集）进行实验。

**📈 对比分析**

对比 100 次实验的 bpb、VRAM、参数等指标，GEAR‑Evolve 在 40 次实验内就突破 AutoResearch 的 0.98232 bpb 平台，最终 bpb 为 0.97658，VRAM 仅 33.5 GB；GEAR‑Prompt 与 GEAR‑Fixed 分别在 72/84 次实验后突破并获得更高 bpb。

**⚠️ 局限性**

实验仅在单个任务、固定 5 分钟预算、单 GPU 上进行，缺乏跨任务和大规模计算的验证，且搜索策略的复杂度与可解释性仍有限。

---

## 101. Towards Robotic Dexterous Hand Intelligence: A Survey

**arXiv ID:** 2605.13925 | [PDF](https://arxiv.org/pdf/2605.13925v1)

**作者:** Weiguang Zhao `[一作]` (University Of Liverpool), Kaizhu Huang `[通讯]` (Duke Kunshan University)

**通讯引用:** 8851 | [OpenAlex ID](https://openalex.org/A5026022035)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了机器人灵巧手领域的硬件设计、控制与学习方法、数据集与评估，并通过表格和时间线对比不同研究，形成了以硬件、方法、数据、评估为四大维度的系统框架。

**💡 创新点**

创新点在于：①将传统分散的硬件、感知、控制与数据集评估统一为四层结构；②系统性梳理了多种手型、驱动与传输方案以及多模态感知的进展；③通过对比分析展示了不同方法在握持、内在操作、工具使用等任务中的发展脉络；④提出了当前面临的局限与未来方向。

**🔧 技术方法**

主要技术手段为文献检索与筛选、分类整理、对比表格与时间线绘制、归纳与总结；在方法讨论中引用了强化学习、模仿学习、模型驱动控制、扩散策略、视觉‑语言‑动作模型等多种学习范式。

**📊 数据集**

综述涉及的数据集包括：DexGraspNet、GenDexGrasp、DexMV、DexArt、UniDexGrasp++、RealDex、DexCap、DexFuncGrasp、RH20T、DexGraspAnything、BODex、CEDex、Dex1B、DexTOG、VTDexManip 等，涵盖了模拟优化、学习生成、人类捕捉等三大采集方式。

**📈 对比分析**

比较方法主要通过对手型、驱动、传输、感知技术等的表格化对比；对方法层面则通过时间线与代表性实验结果（如抓取成功率、姿态精度、数据规模、跨对象/跨手型的泛化能力）进行概括，指出各方法在不同任务中的优势与不足。

**⚠️ 局限性**

limitation：硬件成本与复杂度高、感知多模态融合与实时同步困难、学习方法泛化与安全性不足、数据集多样性与真实场景覆盖有限、缺乏统一评估标准与可比性，导致实用化与工业化进程缓慢。

---

## 102. Joint Transmit and Receive Antenna Orientation Design for Secure MIMO Communications

**arXiv ID:** 2605.14272 | [PDF](https://arxiv.org/pdf/2605.14272v1)

**作者:** Ailing Zheng `[一作]` (Shanghai Jiao Tong University), Wen Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 112040 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了旋转天线（RA）辅助的安全MIMO通信系统，联合优化发射机前向波束、人工噪声协方差以及发射和接收天线的方向，最大化信息安全率。

**💡 创新点**

创新点包括：①利用RA提供的额外空间自由度实现信道重塑；②从SISO到MIMO的分析，提出泄露消除旋转规则及一维搜索；③设计MMSE重构的交替优化框架，并在球面上使用Riemannian Frank‑Wolfe高效更新RA方向；④将方案推广至多接收机多播场景。

**🔧 技术方法**

主要技术手段有：MMSE等价变换、交替优化(AO)、Riemannian Frank‑Wolfe、半闭式解法、投影与拉格朗日乘子法、线性最小均方误差（LMMSE）、凸优化（CVX）等。

**📊 数据集**

实验采用仿真场景：频率3.5 GHz、波长0.0857 m、天线间距λ/2，随机生成发射机、合法接收机和窃听机之间的距离。

**📈 对比分析**

与固定天线（FOA、RFOA）、等向天线、随机方向、离散方向等基线进行比较；仿真结果表明所提AO算法收敛快、在单机和多机情形下均显著提升安全率，最高可提升61%等。

**⚠️ 局限性**

局限性包括：仅考虑单用户或多播场景，未深入探讨多天线多用户情况；对RA方向的离散化与实现精度有限；对快速衰落环境的响应速度有待提升；未评估实际硬件成本与实现复杂度；实验基于理想CSI的仿真。

---

## 103. SkillFlow: Flow-Driven Recursive Skill Evolution for Agentic Orchestration

**arXiv ID:** 2605.14089 | [PDF](https://arxiv.org/pdf/2605.14089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 104. Case Studies and Reflections on Agentic Software Engineering for Rapid Development of Digital Music Instruments

**arXiv ID:** 2605.14016 | [PDF](https://arxiv.org/pdf/2605.14016v1)

**作者:** Matthew John Yee-King `[一作]` (Goldsmiths), Matthew John Yee-King `[通讯]` (Goldsmiths)

**通讯引用:** 497 | [OpenAlex ID](https://openalex.org/A5085329478)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过代理软件工程（ASE）技术，分别重现了Music Mouse、将Continuator从Python迁移到C++插件，并为tracker sequencer实现了3D OpenGL UI。

**💡 创新点**

证明ASE可在自然语言提示下快速生成可交互的高质量插件，并降低非程序员的开发门槛，提升SDMI的可持续性与互操作性。

**🔧 技术方法**

使用OpenAI Codex的代理软件工程框架，结合JUCE C++框架、CMake构建系统以及WebView与OpenGL交互。

**📊 数据集**

主要使用Music Mouse的用户手册与截图、Continuator的Python源码与测试数据、tracker原始代码库，未引入公开大型数据集。

**📈 对比分析**

通过与手动编码对比，仅需数十分钟即可实现核心功能；在Reaper/Logic Pro等DAW中测试，插件生成准确的MIDI事件和图形交互，性能与手工实现相当。

**⚠️ 局限性**

受限于模板与工具链的预置、对C++与OpenGL基础的依赖，以及非程序员仍需一定技术背景；在更大规模或更复杂插件的迁移效果尚待验证。

---

## 105. Measuring Google AI Overviews: Activation, Source Quality, Claim Fidelity, and Publisher Impact

**arXiv ID:** 2605.14021 | [PDF](https://arxiv.org/pdf/2605.14021v1)

**作者:** Haofei Xu `[一作]` (Washington University in St. Louis), Jacob M. Montgomery `[通讯]` (Washington University in St. Louis)

**通讯引用:** 3314 | [OpenAlex ID](https://openalex.org/A5049220804)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 Google AI Overviews（AIOs）进行大规模、持续 40 天的纵向测量研究，涵盖 55,393 个热门查询，评估 AIO 的激活率、来源选择、主张可信度和对出版商的经济影响。

**💡 创新点**

首次将 AIO 激活、来源质量、主张一致性和广告经济损失四个维度同步量化，并揭示 AIO 激活与查询类型、主题类别高度相关、来源与 SERP 排名机制不同、约 11% 主张缺乏支持，以及 AIO 替代点击导致半数引用页面广告收入消失。

**🔧 技术方法**

使用基于 Puppeteer 的 AWS Lambda 爬虫抓取 AIO 及其引用链接；利用 Grok 4.1 LLM 进行主张抽取与验证；使用 PC1 可信度评分对域名进行量化评估；采用 EasyList 过滤器检测广告；统计学方法（t 检验、χ²、相关系数）对结果进行比较。

**📊 数据集**

数据集来源于 Google Trends 的 19 个主题类别的热门查询（55,393 条），产生 7,583 条 AIO 记录，引用 61,212 个 URL，构成 98,020 条原子主张；同时采集相应 SERP 第 1 页结果和广告信息。

**📈 对比分析**

通过与 SERP 同一查询的前 5/10/50/100 名域名、PC1 分数、UGC 比例进行对照；主张一致性以 Clear/Vague/Omited/Incorrect/Ambiguous 五类标签评估；发现 AIO 激活率 13.7%（问题型 64.7%），主张一致率 88.97%，不支持率 11%；引用页面广告出现率 50.6%，部分页面还保留 Google 赞助广告。

**⚠️ 局限性**

局限性包括：仅采样美国地区、仅限热门趋势查询；无法抓取社交媒体和付费内容的完整文本；对实时结构化源（天气、实时查询）的验证受限；未评估用户对 AIO 信息的信任与传播；缺乏个体化或地域化激活细节；研究仅为 40 天快照，未跟踪系统长期演化。

---

## 106. Fast Leaf-to-Ancestor Minimum Query in the Oracle Model

**arXiv ID:** 2605.14112 | [PDF](https://arxiv.org/pdf/2605.14112v1)

**作者:** Aleksey Upirvitskiy `[一作]` (Independent researcher), Aleksandr Levin `[通讯]` (NRU HSE)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个在oracle模型下，针对根化加权树的叶到祖先路径最小查询的静态数据结构。

**💡 创新点**

证明了在预处理阶段即可完成所有比较，且预处理比较次数达到下界O(nlogh)，从而实现O(1)查询时间。

**🔧 技术方法**

采用了边到点权值转换、确定性全序打平、梯形（ladder）分解、二进制提升与稀疏表RMQ等技术。

**📊 数据集**

无具体实验数据集，论文完全基于理论分析。

**📈 对比分析**

通过理论证明与对已有Borůvka树查询结构的对比，显示该结构在预处理比较次数和查询时间上达到最优。

**⚠️ 局限性**

仅适用于叶到祖先查询，且仅在确定性oracle模型下，实际实现需处理大规模树时可能存在空间常数和实现复杂度等实际限制。

---

## 107. From Descriptive to Prescriptive: Uncover the Social Value Alignment of LLM-based Agents

**arXiv ID:** 2605.14034 | [PDF](https://arxiv.org/pdf/2605.14034v1)

**作者:** Jinxian Qu `[一作]` (Geely AI Lab), Luo Ji `[通讯]` (Geely AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于 GraphRAG 的价值对齐框架 SoVA，用于让大型语言模型在面对日常社会困境时能够做出符合人类社会价值的决策。

**💡 创新点**

创新点在于：①将心理学描述性理论（马斯洛需求层次、普卢奇克情绪轮、亚里士多德美德）转化为可检索的原则指令；②通过弱监督知识图谱扩展原则集合，并在在线检索阶段动态生成指令；③用“预期行为比例”与“美德偏好得分”两种评估指标，将价值对齐量化。

**🔧 技术方法**

主要技术包括 GraphRAG（知识图谱抽取 + 查询聚焦摘要）、社区摘要（Community Summaries）与指令检索、LLM（Llama‑3.3‑70B‑Instruct）以及自定义的强弱监督原则生成流程。

**📊 数据集**

使用的数据集有：① DailyDilemmas（包含二选一道德困境及对应价值标签）；② MIC（提供规则金字塔式原则）；③ 公开对话数据集 DailyDialog 与 ESConv 用于开放式对话的泛化评估。

**📈 对比分析**

与 ECoT、Plan‑and‑Solve、Metacognitive Prompting 等提示式基线、SFT、SteerLM 以及标准 RAG 进行对比。SoVA 在马斯洛与普卢奇克两项指标上的预期行为比例分别达到 95.71% 与 94.51%，显著高于所有基线，显示出更强的价值对齐能力；在开放式对话任务中也获得了较好的 ROUGE‑L 与 BLEU‑2 分数，并在人工评测中在价值对齐和情感理解上优于 Direct。

**⚠️ 局限性**

局限性包括：①对原则集的依赖，若缺乏足够多样化的种子原则，扩展效果有限；②目前仅针对二选一的道德困境，缺乏对多选或连续决策情境的验证；③对跨文化或更细粒度价值观的适应性尚未充分评估。

---

## 108. InsightTok: Improving Text and Face Fidelity in Discrete Tokenization for Autoregressive Image Generation

**arXiv ID:** 2605.14333 | [PDF](https://arxiv.org/pdf/2605.14333v1)

**作者:** Yang Yue `[一作]` (Tsinghua University), Dong Chen `[通讯]` (Microsoft Research)

**通讯引用:** 16848 | [OpenAlex ID](https://openalex.org/A5100364587)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为 InsightTok 的离散视觉 tokenizer 及其对应的 Autoregressive 生成模型 InsightAR，用以提升文本和人脸的重建与生成质量。

**💡 创新点**

创新点在于引入局部、内容感知的文本与人脸感知损失，并通过区域面积加权平衡不同尺度的目标，显著提高对关键视觉信息的保持。

**🔧 技术方法**

技术手段包括 VQGAN 结构、文本检测器（OCR）、人脸检测与对齐、基于预训练识别网络的感知损失、区域加权机制以及 Transformer 结构的 Autoregressive 生成。

**📊 数据集**

使用的主要数据集为 LAION、TokBench、OCR 渲染数据集以及混合训练集（LAION、Flux‑Reason‑6M、Echo‑4o 等）。

**📈 对比分析**

通过在 TokBench 进行文本与人脸重建评估，并在文本生成和人脸生成任务中与 LlamaGenTok-AR 等基线对比，InsightTok 在文本准确率提升约 28.9%，人脸相似度提升 0.09，且生成模型在 rFID/PSNR 等整体指标上不逊色基线。

**⚠️ 局限性**

局限性包括对离线检测器的依赖、对极小或复杂文本/人脸的覆盖不足、以及仅验证了离散 tokenizer，未对连续 tokenizer 或更大规模模型进行深入评估。

---

## 109. ROK-FORTRESS: Measuring the Effect of Geopolitical Transcreation for National Security and Public Safety

**arXiv ID:** 2605.14152 | [PDF](https://arxiv.org/pdf/2605.14152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 110. Memory Forensics Techniques for Automated Detection and Analysis of Go Malware

**arXiv ID:** 2605.14020 | [PDF](https://arxiv.org/pdf/2605.14020v1)

**作者:** Hala Ali `[一作]` (Virginia Commonwealth University), Irfan Ahmed `[通讯]` (Virginia Commonwealth University)

**通讯引用:** 1980 | [OpenAlex ID](https://openalex.org/A5063509441)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

本文开发了一个基于Volatility 3的内存取证框架，用于从Go二进制进程内存中恢复运行时信息，包括字符串、函数、类型、goroutine堆栈等。

**💡 创新点**

创新点在于首次利用Go运行时内部结构在运行时内存中重建完整的执行状态，实现了对无符号、被混淆或剥离的Go恶意软件的动态分析。

**🔧 技术方法**

技术上结合了对Go元数据（runtime/runtime.go）解析、ABI‑aware逆向分析、goroutine堆栈展开、堆对象推断以及内核页面缓存恢复，并实现为Volatility 3插件。

**📊 数据集**

使用的数据集包括BRICKSTORM、Obscura、Pantegana三款真实恶意样本以及开源Screenshotter，均从受控实验环境生成的内存镜像。

**📈 对比分析**

实验结果表明在所有样本上恢复了C2地址、加密密钥、持久化路径、执行参数等关键取证信息，与公开情报相比发现多项新指标，性能上插件平均在几秒到十几秒内完成。

**⚠️ 局限性**

局限性包括目前仅支持x86‑64架构，无法处理ARM64；只恢复调用参数，未覆盖局部变量或返回值；以及对高层数据流追踪仍不完善。

---

## 111. Function-Correction with Optimal Data Protection for the General Hamming Code Membership

**arXiv ID:** 2605.14023 | [PDF](https://arxiv.org/pdf/2605.14023v1)

**作者:** Adityawardhan Yadava `[一作]` (Indian Institute Of Technology Hyderabad), Swaraj Sharma Durgi `[通讯]` (Indian Institute Of Technology Hyderabad)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了对一般二进制 Hamming 码成员函数的单误差纠正函数校正码（SEFCC），并给出了实现最优数据保护的构造方法。

**💡 创新点**

创新点在于将 SEFCC 设计转化为距离-4 图的最大割问题，利用谱图理论和布尔函数的布特（bent）性质得到最优奇偶校验分配，并将结果推广到任意 n ≥ 2。

**🔧 技术方法**

使用了谱图理论、Cayley 图、Krawtchouk 多项式、Walsh 变换以及 Reed-Muller 码的对偶性等理论工具来求解最小特征值和相应的特征向量。

**📊 数据集**

该研究主要是理论分析，没有采用具体实验数据集；验证工作基于代码的组合性质和图论结构。

**📈 对比分析**

与已有的 [7,4,3] Hamming 码成员函数构造相比，本文在偶数 n 情况下实现了同样的最小距离 2 并进一步最小化了距离-2 对数，达到了最优数据保护；对奇数 n 的性能仍以求解优化问题为目标。

**⚠️ 局限性**

主要局限在于奇数 n 的最小特征值及相应最优特征向量尚未完整求解，导致该情况下的最优 SEFCC 构造仍为开放问题；此外缺乏实验验证和实现细节。

---

## 112. Elastic Spiking Transformers for Efficient Gesture Understanding

**arXiv ID:** 2605.13869 | [PDF](https://arxiv.org/pdf/2605.13869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 113. TILT: Target-induced loss tilting under covariate shift

**arXiv ID:** 2605.14280 | [PDF](https://arxiv.org/pdf/2605.14280v1)

**作者:** Kakei Yamamoto `[一作]` (Massachusetts Institute of Technology), Martin J. Wainwright `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 38969 | [OpenAlex ID](https://openalex.org/A5038379562)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为Target-Induced Loss Tilting（TILT）的单步协变移适应方法，利用未标记目标样本对辅助模型进行正则化，消除了传统密度比估计的需求。

**💡 创新点**

创新点在于将源预测器分解为主分量f和辅助分量b，辅以目标端正则化，经过剖面化后等价于相对加权目标风险，从而实现了自适应、局部化的协变移修正。

**🔧 技术方法**

核心技术包括联合最小化带有辅助正则化的平方损失或KL损失、解析出最佳b的闭式表达式、并给出oracle不等式与稀疏ReLU网络的非参数收敛速率。

**📊 数据集**

实验数据涵盖一维Beta与高维Gaussian协变移的合成回归、点质量非参数极限、以及CIFAR‑100图像在亮度、对比度、颜色与高斯模糊等多种腐蚀下的目标域迁移。

**📈 对比分析**

与源域ERM、精确重要性加权、相对密度比拟合以及知识蒸馏等基线对比，TILT在大幅度协变移时显著降低目标误差并逼近理论最优收敛率，且对λ选择具有鲁棒性。

**⚠️ 局限性**

主要局限包括：理论证明仅覆盖平方损失回归；对一般Bregman损失的泛化尚未完成；需要手动调参λ和辅助模型容量；评估主要停留在合成与人为腐蚀的受控环境，缺乏对真实多样化分布移的验证。

---

## 114. MAPLE: Latent Multi-Agent Play for End-to-End Autonomous Driving

**arXiv ID:** 2605.14201 | [PDF](https://arxiv.org/pdf/2605.14201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 115. Neural Fields for NV-Center Inverse Sensing

**arXiv ID:** 2605.13988 | [PDF](https://arxiv.org/pdf/2605.13988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. Image Restoration via Diffusion Models with Dynamic Resolution

**arXiv ID:** 2605.14267 | [PDF](https://arxiv.org/pdf/2605.14267v1)

**作者:** Yang Zheng `[一作]` (University of Electronic Science and Technology of China), Zhaoqiang Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 538 | [OpenAlex ID](https://openalex.org/A5089855005)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了利用动态分辨率扩散模型（SubDPS、SubDAPS 和 SubDAPS++）用于通用图像恢复任务，并对传统像素空间扩散模型 DPS 与 DAPS 进行改造。

**💡 创新点**

创新点包括：① 在无 VAE 的条件下使用动态分辨率减小采样空间；② 引入噪声注入阈值与纠正步骤提升恢复质量；③ 用共轭梯度替代 Langevin 动力学加速测量一致性；④ 在低分辨率阶段先行采样再逐步提升分辨率，显著降低时间与内存成本。

**🔧 技术方法**

主要技术手段包括扩散模型（DPM），动态分辨率采样、细调预训练像素空间模型、DPS/DAPS 采样框架、噪声注入阈值控制、纠正器步骤、共轭梯度优化。

**📊 数据集**

实验使用 256×256 分辨率的 FFHQ 与 ImageNet 验证集进行多任务评估（填充、超分辨、去噪、运动去模糊、非线性去模糊和 HDR 恢复）。

**📈 对比分析**

与现有像素空间方法（DPS、ΠGDM、DiffPIR、RED-diff、MGPS、AdaPS、DAPS）以及潜在空间方法（ReSample、LatentDAPS）进行对比，SubDAPS++ 在大多数任务中取得更优或相当的 PSNR/SSIM/LPIPS/FID，同时推理时间与显存使用显著下降（低至 7–14 秒/图像，显存仅 2–4 GB）。

**⚠️ 局限性**

局限性：① 对阈值 τ 以及分辨率切换时刻的选择仍需经验调优；② 在极高分辨率或极端退化场景下仍可能略逊于全像素 DAPS；③ 目前仅验证于 256×256 图像，尚未探究更大尺寸的可扩展性；④ 需要额外的预训练与微调步骤，增加工程复杂度。

---

## 117. Evolving Layer-Specific Scalar Functions for Hardware-Aware Transformer Adaptation

**arXiv ID:** 2605.14047 | [PDF](https://arxiv.org/pdf/2605.14047v1)

**作者:** Kieran Carrigg `[一作]` (Donders Institute for Brain, Cognition, and Behaviour), Marcel van Gerven `[通讯]` (Donders Institute for Brain, Cognition, and Behaviour)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过遗传程序演化层级特定的标量函数，替换Vision Transformer中的LayerNorm，并在无需从零训练的情况下完成后训练对齐。

**💡 创新点**

采用遗传程序实现异质层特定的符号回归，兼顾功能对齐与硬件友好性，且不需要昂贵的从零训练。

**🔧 技术方法**

遗传程序（GP）进行符号回归、后训练对齐（affine‑only、full fine‑tune、知识蒸馏）以及计算和内存成本分析。

**📊 数据集**

预训练的ViT‑B模型在ImageNet‑1K上提取LayerNorm映射并进行评估。

**📈 对比分析**

与标准LayerNorm和均匀标量替代DyT比较，GP在仅20个epoch后实现84.25% Top‑1准确率（vs 84.94% LN、82.99% DyT），功能对齐R²达0.916（vs 0.702）。

**⚠️ 局限性**

仅在ViT‑B/ ImageNet‑1K验证；GP仅使用50k激活样本，未直接在演化中优化模型精度；每层存储独立表达式导致存储开销；结果尚未在其它Transformer或NLP任务上验证。

---

## 118. Minimal-Intervention KV Retention: A Design-Space Study and a Diversity-Penalty Survivor

**arXiv ID:** 2605.14292 | [PDF](https://arxiv.org/pdf/2605.14292v1)

**作者:** Libo Sun `[一作]` (Auburn University), Xiao Qin `[通讯]` (Auburn University)

**通讯引用:** 4687 | [OpenAlex ID](https://openalex.org/A5042766429)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小预算KV缓存压缩领域，对七种干预机制进行设计空间实验，最终发现仅保留TriAttention 1d分数的多样性惩罚（α）能在匹配内存条件下提升长形式数学推理准确率。

**💡 创新点**

创新点是将单一V空间冗余惩罚加入保留分数，替代传统argmax-top-k，实现极简化的分数层干预，并通过预注册协议验证其有效性。

**🔧 技术方法**

技术包括基于TriAttention 1d的保留分数，贪婪facility-location式选择，V空间签名计算，预注册实验协议，以及与SnapKV窗口注意力的对比。

**📊 数据集**

使用MATH-500长形式数学推理数据集，评估两个7-8B的蒸馏推理模型（DeepSeek-R1-Distill Qwen-7B和Llama-8B）。

**📈 对比分析**

比较方法为匹配平均缓存长度的实验，采用MD5分桶的dev/confirm拆分，三随机种子cluster bootstrap，Bonferroni校正，并与内部TriAttention 1d以及SnapKV风格基线对比；α在Qwen-128和Llama-64上显著提升约4–5个百分点，其他单元无显著负面影响。

**⚠️ 局限性**

局限在于仅针对两款7-8B推理模型的长形式数学推理工作负载，未验证对其他任务、模型规模或非推理模型的泛化；α的超参数λ仅在有限网格上调优，且未考虑不同解码步长的动态λ或更复杂的V签名。

---

## 119. AttnGen: Attention-Guided Saliency Learning for Interpretable Genomic Sequence Classification

**arXiv ID:** 2605.14073 | [PDF](https://arxiv.org/pdf/2605.14073v1)

**作者:** Rayhaneh Shabani Nia `[一作]` (University of California, Davis), Ali Karkehabadi `[通讯]` (University of California, Davis)

**通讯引用:** 86 | [OpenAlex ID](https://openalex.org/A5013688249)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了 AttnGen 框架，通过注意力机制估计核苷酸重要性并在训练期间逐步掩蔽低重要位点，从而提升基因序列分类模型的可解释性。

**💡 创新点**

创新点在于将前向注意力权重直接用于掩蔽，并引入 KL 一致性正则，使模型在保持性能的同时更聚焦于少量高信息位点，首次在基因序列任务中实现训练期间的可解释性驱动。

**🔧 技术方法**

使用了 1D CNN + 前向注意力权重、进化掩蔽策略、KL 正则、Adam 优化、梯度验证等技术。

**📊 数据集**

使用 Genomic Benchmarks 中的 demo_human_or_worm 二分类数据集（200 nt 长度序列，10 万条样本）。

**📈 对比分析**

与传统 CNN 基线对比：10%掩蔽时准确率从 95.83% 提升到 96.73%（+0.90pp），20% 维持 96.10%；更高掩蔽导致准确率下降，验证了中等掩蔽最优；收敛速度更快、训练稳定性更好。

**⚠️ 局限性**

局限性：仅在短序列上验证；高掩蔽下泛化差；掩蔽比例和注意力权重设计需针对不同任务或更长序列调整；可解释性评估主要基于准确率下降，缺乏更深入的生物学验证。

---

## 120. Towards Real-Time Autonomous Navigation: Transformer-Based Catheter Tip Tracking in Fluoroscopy

**arXiv ID:** 2605.14253 | [PDF](https://arxiv.org/pdf/2605.14253v1)

**作者:** Harry Robertshaw `[一作]` (Kings College London), Thomas C. Booth `[通讯]` (King's College London)

**通讯引用:** 17186 | [OpenAlex ID](https://openalex.org/A5003288277)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一个基于深度学习的实时导管/导丝尖端跟踪框架，利用多线程流水线实现低延迟的实时定位。

**💡 创新点**

创新点在于将 Transformer‑based SegFormer 嵌入端到端跟踪流水线，并通过骨架化与多点采样技术，在低对比、噪声和遮挡等临床条件下实现鲁棒定位；首次在体外仿真、动物以及临床荧光多种真实数据上进行验证。

**🔧 技术方法**

采用 U‑Net、U‑Net+Transformer、SegFormer 三种分割架构；多线程读/预处理/推理/后处理流水线；骨架化、端点检测、多点采样；滑动窗口、投影等推理技巧。

**📊 数据集**

使用 CathAction 训练集（25,000 像素级标注，包含导管与导丝），以及四类评估数据：CathAction 挂出的猪模型、体外离线 RGB、体外实时 RGB、临床荧光（高、中、低复杂度）。

**📈 对比分析**

与 CathAction 传统 U‑Net、TransUNet 等基线比较，SegFormer 在二分类上 Dice 0.809、IoU 0.722，三分类 Dice 0.667；离线 RGB 中 MAE_(x,y) 0.47mm（二分类），三分类 MAE 1.21mm；临床荧光中 SegFormer 低复杂度 MAE_(x,y) 4.37mm，高复杂度 7.58mm，均优于 U‑Net 和 U‑Net+Transformer。

**⚠️ 局限性**

局限：仅使用仿真训练数据，未进行临床微调；临床荧光中的误差仍在 4–7 mm 范围，未达 sub‑mm；标注由非临床人员完成，可能存在偏差；长时间连续使用的稳定性尚未评估。

---

## 121. Seed3D 2.0: Advancing High-Fidelity Simulation-Ready 3D Content Generation

**arXiv ID:** 2605.13862 | [PDF](https://arxiv.org/pdf/2605.13862v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 122. Vision-Based Runtime Monitoring under Varying Specifications using Semantic Latent Representations

**arXiv ID:** 2605.13923 | [PDF](https://arxiv.org/pdf/2605.13923v1)

**作者:** Bardh Hoxha `[一作]` (Toyota North America Research and Development), Georgios Fainekos `[通讯]` (Toyota North America Research and Development)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文提出一种基于视觉观测的可复用、带有置信度保证的运行时监控框架，用于评估过去时间信号时序逻辑（ptSTL）公式的安全性；

**💡 创新点**

其创新点在于：1）定义并证明了“语义基准”（semantic basis）为在给定公式片段中支持所有公式的最小可学习表征；2）利用单一的 conformal 校准即可实现整个片段的置信下界，无需针对每个公式单独训练；3）提出滚动预测（rolling）监控方案，以降低学习难度并在短时延内实现更紧凑的置信区间；

**🔧 技术方法**

主要技术包括：视觉编码器（CNN+注意力+时序融合）用于从图像序列中提取潜在向量；ptSTL 语义的递归解析树构成的 monotone、1‑Lipschitz 解码器；Split conformal 预测误差转换为公式级别的置信下界；

**📊 数据集**

实验数据集包括：人工生成的交叉口（Crossroad）仿真数据和真实世界的 Waymo Open Motion Dataset（WOMD），均包含多种安全预测量（如前方安全、车道清晰度等）；

**📈 对比分析**

与基于观测器的 Bonferroni 校准方法相比，语义基准监控在所有公式上实现了更小的置信半径、覆盖率 ≥90%（满足理论保证），并在 Waymo 真实数据中表现出更高的安全可测比例（CSR）和更低的误报率；在交叉口场景中，滚动监控在短时间窗内的置信半径更紧，但在较长时间窗（K≈3 以上）时语义基准优于滚动；

**⚠️ 局限性**

限制与不足：仅适用于过去时间 STL 的 ∧/∨ 关闭片段；要求公式解码器为 monotone、1‑Lipschitz；对数据分布的交换性假设敏感，未深入处理分布漂移；以及语义基准的高维度在非常大的原子词典下仍可能导致学习瓶颈。

---

## 123. A Two-Dimensional Framework for AI Agent Design Patterns: Cognitive Function and Execution Topology

**arXiv ID:** 2605.13850 | [PDF](https://arxiv.org/pdf/2605.13850v1)

**作者:** Jia Huang `[一作]` (Agency for Science, Technology and Research (A*STAR)), Joey Tianyi Zhou `[通讯]` (Agency for Science, Technology and Research (A*STAR))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一个二维框架，将LLM代理体系结构按认知功能（7类）与执行拓扑（6类）进行交叉分类，并构建了27个命名模式。

**💡 创新点**

创新点在于将工业视角的执行拓扑与认知科学视角的功能维度并行展开，形成可互补且框架中立、模型无关的二维模式矩阵，揭示了不同拓扑下相同功能、不同功能下相同拓扑的多样性，并通过跨领域案例推导出五条模式选择经验法则。

**🔧 技术方法**

主要技术包括：模式矩阵设计与系统化归纳、对比分析、案例驱动的模式验证、经验法则提炼以及对四个实际业务领域（金融贷前、法律尽职、网络运维、医疗急诊）进行模式映射。

**📊 数据集**

使用了四个真实业务场景的业务数据集：SME 贷款申请文件、并购合同文本、持续的网络运维告警流、急诊患者评估记录；并结合行业指南与现有框架的模式集合进行对照。

**📈 对比分析**

通过在四个领域内构建并评估不同模式组合，比较了模式覆盖率、架构复杂度与业务约束的匹配度；结果表明：时间压力越大、行动权限越低、失败成本越不对称、业务量越大，所选模式数与拓扑类型会相应变化，并提出了五条“法律”来指导模式选择，验证了框架的可解释性和通用性。

**⚠️ 局限性**

局限性包括：27/42的模式填充不均，空单元可能是未探索或不可能组合；模式粒度选择具有主观性；随着LLM能力演进，一些模式（如链式推理）可能被新的实现替代；框架在极端资源受限或安全关键场景下的适用性仍需进一步实证。

---

## 124. Parallelizing Counterfactual Regret Minimization

**arXiv ID:** 2605.14277 | [PDF](https://arxiv.org/pdf/2605.14277v1)

**作者:** Juho Kim `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20314 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并实现了一个通用框架，将序列形式的反事实遗憾最小化（CFR）算法及其主要变体（CFR+、DCFR、PCFR、PCFR+）重构为线性代数运算，并在 GPU 上并行化执行，从而显著加速了大型不完全信息游戏的求解。

**💡 创新点**

创新点包括：① 将 CFR 的树遍历和行为策略更新改写为逻辑矩阵和稀疏矩阵运算；② 利用 GraphBLAS 思路进行层级树遍历；③ 通过标准的线性代数并行化技术（如 CSR 矩阵向量乘）实现跨算法的统一并行化；④ 在 GPU 上实现，获得四个数量级的加速；⑤ 证明了算法的工作量不变、深度随树高而增长，具备理论可扩展性。

**🔧 技术方法**

使用技术：逻辑矩阵、稀疏矩阵（CSR）运算、Hadamard 操作、GraphBLAS 灵感的层级邻接矩阵、CuPy GPU 加速、工作深度模型分析；实现细节包括顶层单位向量、行为策略归一化、行为与序列形式的互换。

**📊 数据集**

实验数据集：OpenSpiel 提供的七个基准游戏，包括 Kuhn poker、Leduc poker、Liar's dice 以及四种规模不同的 Battleship（tiny、small、medium、large），节点数从 58 到 6.7×10⁷。

**📈 对比分析**

评估方法：在同一台机器上运行 1000 秒的 CFR，至少 8 次迭代；对比 OpenSpiel 的 Python 和 C++ 实现；记录迭代时间、墙钟时间和可利用性；结果显示：对大型游戏，GPU 并行实现相较于 OpenSpiel C++ 速度提升约 4.1×10³ 倍，Python 约 2.7×10⁴ 倍；在小型游戏中因 GPU 启动和内存开销略慢。

**⚠️ 局限性**

局限性：① 并行化并未扩大可求解游戏的规模，仅提升求解速度；② GPU 开销在小规模游戏上导致效率低于 CPU 实现；③ 目前仅针对序列形式 CFR，若应用到经典 CFR 需额外逻辑矩阵且内存效率更低；④ 仅验证了 tabular CFR 变体，对深度学习或稀疏游戏等情况的适应性仍待研究；⑤ 速度提升受硬件（GPU 核数、内存带宽）限制；⑥ 未与 pruned CFR、最佳响应等其它游戏理论算法进行对比。

---

## 125. TeDiO: Temporal Diagonal Optimization for Training-Free Coherent Video Diffusion

**arXiv ID:** 2605.14136 | [PDF](https://arxiv.org/pdf/2605.14136v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 126. Masked Autoencoders with Limited Data: Does It Work? A Fine-Grained Bioacoustics Case Study

**arXiv ID:** 2605.14031 | [PDF](https://arxiv.org/pdf/2605.14031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 127. Unsteady Metrics and Benchmarking Cultures of AI Model Builders

**arXiv ID:** 2605.14164 | [PDF](https://arxiv.org/pdf/2605.14164v1)

**作者:** Stefan Baack `[一作]` (Independent Researcher), Maty Bohacek `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建并分析Benchmarking-Cultures-25数据集，探讨AI模型发布文档中突出显示的基准选择及其叙事作用。

**💡 创新点**

创新点在于提出统一的基准能力分类法，并揭示基准被用于市场定位的叙事机制。

**🔧 技术方法**

使用定性分析、量化统计、图形可视化以及自建的交互工具。

**📊 数据集**

使用Benchmarking-Cultures-25数据集，包含231个基准、139个模型发布、11个AI公司在2025年的公开声明。

**📈 对比分析**

通过几何平均权重的基准受欢迎度指标和跨模型采用率评估，结果显示基准选择高度碎片化，跨模型比较受限。

**⚠️ 局限性**

局限在于仅覆盖单一年份（2025），未对模型卡进行分析，且仅由单一作者完成分类注释，缺乏多方验证。

---

## 128. Flow Field Reconstruction with Sensor Placement Policy Learning

**arXiv ID:** 2605.14137 | [PDF](https://arxiv.org/pdf/2605.14137v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 129. Indian Wedding System Optimization (IWSO): A Novel Socially Inspired Metaheuristic with Operational Design and Analysis

**arXiv ID:** 2605.13871 | [PDF](https://arxiv.org/pdf/2605.13871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 130. VectraYX-Nano: A 42M-Parameter Spanish Cybersecurity Language Model with Curriculum Learning and Native Tool Use

**arXiv ID:** 2605.13989 | [PDF](https://arxiv.org/pdf/2605.13989v1)

**作者:** Juan S. Santillana `[一作]` `[通讯]` (Globant), Juan S. Santillana (Globant)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 VectraYX‑Nano，一款 42M 参数的 Spanish 领域语言模型，专为网络安全任务训练，并内置 Model Context Protocol (MCP) 原生工具调用功能，支持离线边缘部署。

**💡 创新点**

创新点在于：① 三阶段逐步式预训练+重放机制，避免“灾难性遗忘”，同时将对话、技术和工具数据按比例混合；② 采用 16k 字节回退 BPE，保证 CVE、hash、base64 等技术字符串能完整序列化；③ 结合 MCP，模型可在生成时直接发出 JSON 触发外部工具，真正实现工具与语言模型的联动；④ 通过实验展示了 bootstrap‑corpus 注册对对话质量的主导作用，并量化工具使用的“密度门槛”。

**🔧 技术方法**

技术包括：Transformer decoder，Grouped‑Query Attention、QK‑Norm、RMSNorm、SwiGLU、RoPE、z‑loss；使用 SentencePiece BPE 16K 词表，训练时加入 27 个用户定义的专属符号；通过 replay‑aware curriculum sampler 采样三阶段数据；在 SFT 阶段采用“assistant‑only”损失掩码，结合三轮内部 mini‑curriculum；使用 LoRA 进行小规模工具使用微调；模型导出为 GGUF 量化格式（F16 81MB / Q4 20MB），兼容 Ollama。

**📊 数据集**

数据集：170M token 的 VectraYX‑Sec‑ES 语料库，分为 42M 对话、118M 网络安全（NVD、Wikipeida‑ES、博客等）和 10M 工具（HackTricks、ExploitDB、OWASP 等）；工具调用 SFT 数据 6,327 条来自内部 CVE/IOC 镜像；此外还使用 OpenSubtitles‑ES、OASST1‑ES、mC4‑ES、Wikipedia‑ES 等语料进行 ablation。

**📈 对比分析**

与同规模或更大模型进行比较：在 B1–B5 评测上，VectraYX‑Nano 的 B1≈0.24（相对 135M SmolLM+LoRA 0.334）且 B5（对话门）≈0.78±0.05；工具选择 B4 在 42M 上为 0.00，但通过增加工具使用密度（1:21）可提升至 0.15；在 260M mid‑tier 仍保持 B1≈0.32、B5≈0.80，但 B4 仍为 0；更大 3B/7B 通过 LoRA 可显著提升 B4、B2、B3。总的来说，该模型在 42M 参数级别已实现可用的对话、CVE 查询和工具调用，且部署成本极低（≈$29 训练费用，20MB GGUF）。

**⚠️ 局限性**

局限性包括：① 训练规模低于 Chinchilla 最优，token‑parameter 只占 4 倍；② 只覆盖 2026‑04 前的安全信息，后续 CVE/KEV 需实时更新；③ 工具调用仅支持单步或极短链，无法处理多步链路；④ 工具使用性能受 SFT 数据密度限制，低密度时 B4 低于 0；⑤ 对话质量受 bootstrap‑corpus 注册影响，若更换语料需重新评估；⑥ 未进行大规模人类评测或安全对齐，部署时需在 MCP 层加以过滤。

---

## 131. When Evidence Conflicts: Uncertainty and Order Effects in Retrieval-Augmented Biomedical Question Answering

**arXiv ID:** 2605.14115 | [PDF](https://arxiv.org/pdf/2605.14115v1)

**作者:** Yikun Han `[一作]` (University of Illinois Urbana Champaign), Halil Kilicoglu `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在HealthContradict基准上，对六款开源LLM在不同检索证据条件（无证据、正确信息、错误信息、混合顺序）下的答案准确性、校准和不确定性进行系统评估，并提出了基于冲突检测的自适应放弃策略。

**💡 创新点**

创新点在于：①构造了精细的证据顺序对照实验，揭示同一对正负证据顺序会显著影响模型表现；②设计了冲突感知放弃分数（结合模型置信度与冲突检测器），在最难的错误/错误先后混合条件下显著提升选择性准确率。

**🔧 技术方法**

主要技术包括：检索增强生成框架、约束的YES/NO二元输出、基于下一个token概率的置信度/熵/logit margin估计、冲突检测器（within-condition logistic回归）以及基于阈值的选择性预测。

**📊 数据集**

使用的数据集为HealthContradict（920个医药是非问答实例，每个实例配有支持与反对的两份检索文档）。

**📈 对比分析**

评估指标包括准确率、AUROC、ECE、Brier分数、预测翻转率；实验显示：在IC和ICC条件下，冲突感知分数在覆盖率25%、50%、75%时分别比单纯置信度提升7.2–33.4、3.6–14.4个百分点；顺序效应导致平均18.6%的答案翻转，且模型在正确文档先行时准确率平均提升11.1个百分点。

**⚠️ 局限性**

局限性在于：实验仅基于单一二元问答基准，且未涉及多源综合或开放式回答，可能无法完全代表更复杂的医学检索推理场景。

---

## 132. Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility

**arXiv ID:** 2605.14037 | [PDF](https://arxiv.org/pdf/2605.14037v1)

**作者:** Gergely Szilvasy `[一作]` (Meta FAIR), Hervé Jégou `[通讯]` (Meta FAIR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自学习的KV写入稀疏机制 Self-Pruned KV Attention，利用轻量级实用度预测器在推理时仅将高实用度的 KV 对写入持久缓存，从而显著减少 KV 缓存大小并加速解码。

**💡 创新点**

创新点在于：①在训练期间通过软门控和阈值感知硬门控自适应学习实用度评分，而非依赖后置剪枝或手工规则；②在保持全注意力训练框架的同时实现可训练的稀疏写入；③将该稀疏写入视为神经架构搜索的探针，用以设计更强的局部‑全局混合 Transformer。

**🔧 技术方法**

使用了：轻量级 2 层 MLP 预测实用度；soft gating（log u）和阈值感知硬门控（TAHG）；FlashAttention‑3 的原地融合门控偏置；block‑skipping 优化实现稀疏解码；以及多尺度连续预训练策略。

**📊 数据集**

在 Llama‑3 基础模型上使用混合预训练语料：主要为 DCLM、代码、书籍等，并通过扩增长序列比例来提升长上下文能力；训练使用 140 tokens/参数的比率，并在 8k/32k 级上下文长度下进行。

**📈 对比分析**

与后置剪枝方法（KVZap、ExpectedAttention、StreamingLLM 等）在 LongPPL 评估中进行对比。Self‑Pruned KV 在相同或更低的 KV 密度（≈25%–11%）下，NLL 退化仅约 0.08%–0.46%，显著优于后置方法（3–5% 退化）。同时，在批量长上下文解码中实现 2.1×–4.6× 的速度提升和显著的内存占用降低。

**⚠️ 局限性**

局限性：仅在以英语为主的预训练数据上验证；未评估在多语言、专业领域或强化学习后训练中的泛化；系统实现尚未完全优化，稀疏解码性能有进一步提升空间；需要进一步验证在指令微调或 RL 训练阶段的稳定性与收益。

---

## 133. TabPFN-3: Technical Report

**arXiv ID:** 2605.13986 | [PDF](https://arxiv.org/pdf/2605.13986v1)

**作者:** Léo Grinsztajn `[一作]`, Frank Hutter `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TabPFN‑3，一款基于Transformer的开源表格基础模型，能够在单次前向传播中处理高达百万行、上千特征的表格数据，并支持多类、时间序列、关系型和文本‑表格任务。

**💡 创新点**

创新点包括：① 采用聚合行压缩的两阶段架构，显著降低对行数的二次方成本；② 关注点缩放软最大化(QASSMax)提升长序列泛化；③ 引入注意力型多类解码器，实现无参数化多类别预测；④ 行分块与多查询KV缓存实现单GPU百万行推理；⑤ 原型训练使用更丰富的合成SCM先验，覆盖时间、空间、关系等结构；⑥ API支持原生文本特征和“思考模式”推理。

**🔧 技术方法**

技术主要包括Transformer、行/列注意力、诱导点注意力、RMSNorm、目标正交嵌入、Multi‑Query Attention、FlashAttention‑3、行分块与KV缓存、基于合成数据的预训练。

**📊 数据集**

使用的主要数据集：公开TabArena、TALENT、TabSTAR、RelBenchV1、fev‑bench、内部大规模数据集（10⁵–10⁶行）、合成多类、许多特征、量化回归基准。

**📈 对比分析**

与AutoGluon 1.5 Extreme、XGBoost、LightGBM、CatBoost、TabICLv2、TabM、RealMLP等基线比较，TabPFN‑3在TabArena的前向传播里以约200 Elo领先所有模型，速度提升10倍；在大数据基准上单估计器即可压制传统GBDT；在多类、时间序列、关系型任务中均取得SOTA或接近SOTA。

**⚠️ 局限性**

局限性：① 许可证限制仅允许非商业研究与内部评估，商业部署需购买企业许可；② 虽支持多类最高160类，但极大类别数仍受预训练上限约束；③ 依赖合成先验，虽覆盖多结构但在某些真实分布偏差较大的任务上仍可能欠佳；④ 对极高维特征（>200）仍需子采样，可能影响信息完整性；⑤ 在某些实时或极低延迟场景下，单GPU推理仍有1–3 ms/样本的限制。

---

## 134. Automatic Landmark-Based Segmentation of Human Subcortical Structures in MRI

**arXiv ID:** 2605.14221 | [PDF](https://arxiv.org/pdf/2605.14221v1)

**作者:** Ahmed Rekik `[一作]` (École de technologie supérieure), Linda Marrakchi-Kacem `[通讯]` (University of Tunis El Manar)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

结合解剖学标记与深度学习，提出一种基于16个解剖学标记的三阶段自动分割流程，先检测标记，再使用UNesT进行粗分割，最后通过基于标记的规则将12标签细化为26个子皮层结构；

**💡 创新点**

创新点在于将Harvard–Oxford Atlas的手工分割协议形式化为可自动化的标记驱动规则，显著提升了边界一致性与解剖学精度；

**🔧 技术方法**

使用了Patch-based Iterative Network（PIN）+局部3D FCNN进行标记定位，UNesT 3D Transformer网络进行语义分割，以及基于标记的后处理规则；

**📊 数据集**

采用Human Connectome Project Young Adult的HOA子皮层数据集，共100份T1w MRI，包含26结构的手工注释和16个标记；

**📈 对比分析**

与直接训练UNesT进行26标签分割的基线相比，平均Dice略有提升（从0.8961到0.8992），但在Protocol-Aligned Surface Distance（PASD）和2D分割线均匀度上实现了显著改进；

**⚠️ 局限性**

局限在于依赖标记定位的精度，标记误差会传递到后处理；数据集仅限健康年轻成人，缺乏病理多样性，且方法需先定义标记与规则才能迁移到其他解剖协议。

---

## 135. Fast and Robust Mesh Simplification for Generated and Real-World 3D Assets

**arXiv ID:** 2605.14029 | [PDF](https://arxiv.org/pdf/2605.14029v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 136. Source-to-Source Transformations for GPU Code Generation

**arXiv ID:** 2605.13864 | [PDF](https://arxiv.org/pdf/2605.13864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 137. New Algorithms for Parity-SAT and Its Bounded-Occurrence Versions

**arXiv ID:** 2605.14093 | [PDF](https://arxiv.org/pdf/2605.14093v1)

**作者:** Sanjay Jain `[一作]` (National University of Singapore), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1497 | [OpenAlex ID](https://openalex.org/A5033729619)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出并设计了一系列多项式空间的快速算法，用于求解Parity‑SAT及其d‑occurrence版本，突破了传统2^m上限，并针对d=2给出了更优的时间复杂度。

**💡 创新点**

创新点在于利用Parity计数的特殊结构，结合变量/子句分支规则、图割与测度分析，首次证明Parity计数在相同约束下可比精确计数更快，并实现了2^m障碍的突破。

**🔧 技术方法**

主要技术包括随机化分支、基于出现次数的变量/子句分支、构造多边形图并使用bisection分割、以及branch‑and‑search的测度‑分支分析。

**📊 数据集**

论文未给出实验数据，全部以理论参数（n、m、L、d）为评估指标，主要在公式长度、变量数和子句数上进行复杂度证明。

**📈 对比分析**

与已知的#SAT/Parity‑SAT算法相比，本文在2^m、2^n和L维度上提供了更小的指数基（1.3248^m、1.1193^n、1.1052^L），在多项式空间下实现了性能提升。

**⚠️ 局限性**

局限性包括：仅针对CNF公式，受SETH等假设限制；算法实现复杂度高；未在实际大规模实例上进行实验验证，且对精确计数问题的改进尚无法直接迁移。

---

## 138. Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers

**arXiv ID:** 2605.14270 | [PDF](https://arxiv.org/pdf/2605.14270v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 139. FaceParts: Segmentation and Editing of Gaussian Splatting

**arXiv ID:** 2605.13853 | [PDF](https://arxiv.org/pdf/2605.13853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 140. Factorization-Error-Free Discrete Diffusion Language Model via Speculative Decoding

**arXiv ID:** 2605.14305 | [PDF](https://arxiv.org/pdf/2605.14305v1)

**作者:** Xun Fang `[一作]` (East China Normal University), Zhou Yu `[通讯]` (East China Normal University)

**通讯引用:** 16249 | [OpenAlex ID](https://openalex.org/A5006208781)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FeF-DLLM，通过前缀条件化的清洁词预测完全消除离散扩散语言模型的因子化误差，并在此基础上结合推测式解码实现推理加速。

**💡 创新点**

创新点包括：① 用精确的前缀条件分解替代传统的独立词预测，彻底消除因子化误差；② 将推测式解码嵌入到每个扩散去噪步骤，既保持并行预测的优势，又显著加速推理；③ 在理论层面证明该方法生成的分布与真实后验一致，并给出了期望加速比。

**🔧 技术方法**

主要技术：离散扩散模型、X0预测参数化、前缀条件化后验分解、推测式解码、基于LLaDA-Instruct的微调与复现。

**📊 数据集**

实验数据集：GSM8K、MATH、HumanEval、MBPP四大数学推理与代码生成基准。

**📈 对比分析**

对比方法：LLaDA、LLaDA/2、SSD、DDOSP、DCD等基线；实验结果表明FeF‑DLLM在四个基准上平均准确率提升5.04个百分点，平均推理速度提升3.86×（step=2），step=4进一步提升至5.31点、2.33×，显著优于所有对比方法。

**⚠️ 局限性**

局限性：前缀条件化预测与推测式验证在推理阶段需要额外计算资源，导致推理时资源占用高，未来工作需探索更高效实现。

---

## 141. Towards Self-Evolving Agentic Literature Retrieval

**arXiv ID:** 2605.14306 | [PDF](https://arxiv.org/pdf/2605.14306v1)

**作者:** Yuwen Du `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9045 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自演化的代理式文献检索系统 PaSaMaster，能够在复杂自然语言检索意图下迭代改进检索策略、对检索结果进行证据根基的评分，并给出无幻觉、可验证的论文推荐列表。

**💡 创新点**

核心创新点在于：①将检索从一次性查询–文档匹配转变为基于已排好序证据的自演化迭代过程；②将检索视为意图–论文相关度排名而非生成，彻底消除幻觉；③将高成本的前沿 LLM 仅用于意图理解与规划，检索与评分交由轻量化模型和自建语料库实现成本效能提升。

**🔧 技术方法**

使用多模型架构：Navigator（前沿 LLM）负责意图理解与检索策略规划；Librarian Swarm（轻量化检索与评分模型）负责并行检索、证据定位与评分；自定义三层语料库（元数据、摘要、分块）与知识蒸馏训练的 Scorer；工具集分为检索工具与阅读工具。

**📊 数据集**

训练与评估采用自构建的 PaSaMaster‑Bench：244 题目、38 学科，包含多约束自然语言查询及对应的核查清单；检索语料库为160M+学术论文的三层结构化库。

**📈 对比分析**

与关键词检索、语义检索、生成式 LLM、固定管道代理检索等多种基线进行对比。PaSaMaster 在 NDCG@20、Recall@20、Precision@20、F1@20 上均领先，F1 提升 15.6×，幻觉率 0%，平均成本仅 $0.05/次，约为 GPT‑5.2 成本的 1%。

**⚠️ 局限性**

局限性包括：①依赖自建的完整索引语料库，迁移到其他领域或实时更新时需重新构建；②系统复杂度高，需要协调多模型与多阶段流程；③虽然成本低，但对前沿 LLM 的依赖仍使部署成本受限于算力与许可。

---

## 142. Quantum Advantage in Multi Agent Reinforcement Learning

**arXiv ID:** 2605.14235 | [PDF](https://arxiv.org/pdf/2605.14235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 143. Agentic Systems as Boosting Weak Reasoning Models

**arXiv ID:** 2605.14163 | [PDF](https://arxiv.org/pdf/2605.14163v1)

**作者:** Varun Sunkaraneni `[一作]` (Texas A&M University), Tomer Galanti `[通讯]` (Texas A&M University)

**通讯引用:** 286 | [OpenAlex ID](https://openalex.org/A5050776111)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究一种基于验证器的委员会搜索（verifier-backed committee search），在推理语言模型的推理时通过多次提议（propose）、批评（critique）和比较（compare）来实现推理加速。实验中将单个nano模型的推理性能提升至与更强模型相当。

**💡 创新点**

创新点在于：①把推理任务拆解为提议覆盖、局部可识别性、进展深度和多样性四个关键维度，并用理论证明覆盖和可识别性是独立且都必不可少的；②引入oracle best‑of‑k误差分解，揭示“盲区” (blind‑spot) 的上限；③给出局部和全局错误的可组合上界；④在实验中通过批评者和比较者的组合，实证验证理论预测并接近oracle上限。

**🔧 技术方法**

技术包括：多代理委员会协议（k 次采样 → m 次批评 → r 次比较），局部可识别性边界（β, σ），rank‑based 误差累积上界，oracle 盲区分解，使用执行/类型检查/单元测试等本地验证信号作为批评者/比较者。实现基于 nano 语言模型、批评者使用二元判断，比较者采用 Copeland 选举。

**📊 数据集**

数据集：SWE‑bench Verified（500 个软件工程任务），每个任务包含仓库、issue 与可见单元测试，隐藏测试用作最终评估。

**📈 对比分析**

对比方法：单次 nano 生成（Pass@1 67.0%）；oracle best‑of‑k（k=8）上限 79.0%；与更强的独立模型（Thinking、GPT‑4 等）比较。实验结果显示，批评+比较的完整 harness 在 k=8 时达到 76.4%，仅落后 2.6% 于 oracle 上限，且与较强模型性能相当。

**⚠️ 局限性**

限制：大部分失败仍来自提议覆盖不足（盲区），即提议池中缺少可通过本地验证识别的正确解；对仅提供有限本地信号的任务适用性有限；多次调用带来额外推理成本和延迟；模型与批评者/比较者的设计对结果敏感，仍需进一步提升可识别性。

---

## 144. EnergyLens: Predictive Energy-Aware Exploration for Multi-GPU LLM Inference Optimization

**arXiv ID:** 2605.14249 | [PDF](https://arxiv.org/pdf/2605.14249v1)

**作者:** Zhiye Song `[一作]` (Massachusetts Institute of Technology), Anantha P. Chandrakasan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 80084 | [OpenAlex ID](https://openalex.org/A5084128470)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了EnergyLens框架，利用高层次的einsum接口在没有实现、跟踪或GPU访问的情况下预测LLM推理的能耗与延迟；

**💡 创新点**

创新点在于：①提供无实现的能耗预测；②将并行、MoE负载不平衡与多GPU通信能耗模型结合；③支持Megatron式的通信-计算重叠能耗估计；

**🔧 技术方法**

采用einsum式接口与轻量级解释器、EnergAIzer计算核预测、经验驱动的通信能耗模型、SM分配aware的重叠模型以及能耗中心度量ETFT/EPOT；

**📊 数据集**

在Llama3-8B/70B与Qwen3-30B-A3B等密集与MoE模型上进行评估，使用TensorRT-LLM与Megatron-LM进行真实测量；

**📈 对比分析**

与真实测量（NVML+TensorRT-LLM）对比，MAPE在多GPU能耗上为0-15%，重叠模型为0-20%，能够准确恢复Pareto前沿，识别最佳配置并节约30%以上能耗；

**⚠️ 局限性**

局限在于对小批量解码阶段的延迟预测精度不足；对极低算术强度核的表现不佳；需要真实路由统计以提升MoE模型精度。

---

## 145. Breaking Global Self-Attention Bottlenecks in Transformer-based Spiking Neural Networks with Local Structure-Aware Self-Attention

**arXiv ID:** 2605.13887 | [PDF](https://arxiv.org/pdf/2605.13887v1)

**作者:** Lingdong Li `[一作]` (Tianjin University), Qiang Yu `[通讯]` (Tianjin University)

**通讯引用:** 22206 | [OpenAlex ID](https://openalex.org/A5100717180)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 LSFormer，一种结合 Spiking Response Pooling 和 Local Structure-Aware Spiking Self-Attention 的 Transformer‑based Spiking Neural Network。

**💡 创新点**

创新点在于引入 SPooling 以保留最大值与平均值信息，并在自注意力中采用局部稀疏窗口、多尺度扩张与通道重校准，形成 LS-SSA。

**🔧 技术方法**

技术包括 Leaky Integrate‑and‑Fire neuron、稀疏 Spike‑Driven 计算、窗口化多尺度扩张注意力、通道重校准、以及深度可分离卷积补边特征。

**📊 数据集**

使用的公开数据集包括 CIFAR‑10、CIFAR‑100、Tiny‑ImageNet、CIFAR10‑DVS、DVS128 Gesture 以及 N‑CALTECH101。

**📈 对比分析**

与现有 Transformer‑based SNNs 以及传统 ANN 对比，LSFormer 在 CIFAR‑10、CIFAR‑100、Tiny‑ImageNet、DVS 等任务上分别取得 96.73%、82.00%、71.61%、84.30%、98.60%、87.6% 的 Top‑1 准确率，显著优于前沿模型。

**⚠️ 局限性**

局限性包括需要对扩张率、阈值等超参数进行手动调节，且在极大规模图像或实时硬件实现方面尚未验证。

---

## 146. PREPING: Building Agent Memory without Tasks

**arXiv ID:** 2605.13880 | [PDF](https://arxiv.org/pdf/2605.13880v1)

**作者:** Yumin Choi `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在目标环境任务数据出现前，通过自生成的合成练习来构建可复用的程序化记忆的方法（Preping）。

**💡 创新点**

创新点在于：①引入“Proposer Memory”作为构造时的控制状态，用来指导哪些任务需要被合成练习；②使用Validator对合成任务和轨迹进行可行性与完成度评估，仅允许通过验证的轨迹写入可执行记忆；③将记忆分为构造时的Proposer Memory和部署时使用的Solver Memory，二者异步更新，避免将无用或错误信息写入最终记忆。

**🔧 技术方法**

技术上主要依赖大语言模型（DeepSeek‑V3.2）驱动的 Proposer、Solver、Validator 三个模块；使用 LLM 生成任务、执行环境、验证轨迹；随后通过 reflector‑curator 样式的记忆诱导管道把通过验证的轨迹转化为简明的执行步骤；在构造阶段还利用环境文档、执行反馈来更新 Proposer Memory。

**📊 数据集**

使用了三个公开基准：AppWorld（基于应用 API 的状态化任务）、BFCL v3（结构化函数调用与对话约束）以及 MCP‑Universe（真实模型上下文协议服务器中的工具使用）。

**📈 对比分析**

与无记忆基线、仅基于文档或随机/引导式自由探索的预任务方法对比，Preping 在所有三组基准上显著提升：AppWorld 提升约17点，BFCL v3 提升约19点，MCP‑Universe 提升约5点；与利用目标任务的 ACE‑Offline/ACE‑Online 方法相比，Preping 在无目标任务条件下达到相近甚至更优的性能；预构造的记忆还可作为在线学习的初始化，进一步提升性能，并显著减少冷启动失败率和部署时的内存更新成本。

**⚠️ 局限性**

局限性包括：①对 LLM 的强依赖，若模型能力不足或无法正确生成/验证任务，效果会受限；②合成练习依赖目标环境的可访问性与文档完整度，若环境信息不完整可能导致难以生成可行任务；③目前验证标准主要基于简单的可行性与完成度评分，可能无法捕捉更细粒度的错误或异常；④构造成本虽然比在线构造低，但仍需一定的计算资源与时间。

---

## 147. Day-to-Day Traffic Network Modeling under Route-Guidance Misinformation: Endogenous Trust and Resilience in CAV Environments

**arXiv ID:** 2605.14204 | [PDF](https://arxiv.org/pdf/2605.14204v1)

**作者:** Eunhan Ka `[一作]` (Purdue University), Satish V. Ukkusuri `[通讯]` (Purdue University)

**通讯引用:** 13020 | [OpenAlex ID](https://openalex.org/A5018158882)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在连网与自动驾驶环境下，路由指导信息被误导时，基于日间-日间(DTD)交通分配与用户信任演化相耦合的交通网络模型。

**💡 创新点**

创新点在于将动态信任演化嵌入到 DTD 模型中，提出阈值驱动的行为恢复机制、加权合规性指标与攻击后恢复定律，并揭示隐藏的安全窗口。

**🔧 技术方法**

主要技术包括：LWR 车辆负荷模型、带界限的多项式 Logit 路径选择、Beta 证据模型更新信任、信息依赖调节与日间-日间学习框架。

**📊 数据集**

使用了 Sioux Falls 与 Anaheim 两个基准交通网络（包含路径集与 OD 需求）进行仿真验证。

**📈 对比分析**

通过比较固定信任与动态信任的价格攻击（PoAtt）与信任诱导衰减（TIA）指标，结果显示在阈值以上时动态信任可将攻击影响降低约 90%；在低强度下表现为隐蔽（stealthy）状态；在攻击结束后交通性能恢复快于信任恢复，形成 77 天的隐藏易受攻击窗口。

**⚠️ 局限性**

局限性包括：攻击模型简化为推荐层误导、信任更新为聚合 Beta 模型、缺乏真实数据校准、仅验证两套网络、未考虑平台侧检测与游戏论攻击者-防御者互动等。

---

## 148. Fair and Calibrated Toxicity Detection with Robust Training and Abstention

**arXiv ID:** 2605.14074 | [PDF](https://arxiv.org/pdf/2605.14074v1)

**作者:** Mokshit Surana `[一作]` `[通讯]` (University of Illinois Chicago), Mokshit Surana (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了毒性检测模型公平性的三轴评估框架，并比较了 ERM、实例重加权 ERM 和 Group DRO 三种训练方法，随后探讨了温度缩放、置信度拒绝以及身份阈值优化等后置技术对各轴的影响。

**💡 创新点**

创新点在于：① 将排序、校准与拒绝三条公平性轴统一入评估体系；② 引入“Calibration‑fairness gap”度量子组校准偏差；③ 明确训练方法决定后置干预的效能，从而揭示后置技术在不同训练方案下的局限性。

**🔧 技术方法**

使用技术包括 DistilBERT‑base 作为基线模型，分别在 ERM、重加权 ERM 与 Group DRO 下训练；采用温度缩放（单参数校准）、置信度拒绝（风险‑覆盖曲线）以及身份阈值优化；所有指标均采用 1000 次配对自助采样 95% 置信区间评估。

**📊 数据集**

数据集为 Civil Comments（Jigsaw）200k 采样子集（含 18k 背景样本和 8 个身份子组）以及 HateXplain 用于零样本跨域泛化评估。

**📈 对比分析**

比较结果显示：无单方法在三轴上占优；ERM 在整体校准良好但子组存在显著偏差；重加权 ERM 提升排序公平性但加剧校准差距；Group DRO 消除校准差距但整体失准并破坏拒绝；后置方法的效能高度依赖于训练方案（如温度缩放对 Group DRO 无效，置信度拒绝对 ERM 有效但对 DRO 失效）。

**⚠️ 局限性**

局限性包括：少数族群样本不足导致置信区间宽泛；单一身份标签分配忽略交叉身份影响；仅使用 DistilBERT‑base 进行实验；ECE 采用 15 等宽箱，可能影响数值；后置技术在面向不均匀子组校准时表现受限。

---

## 149. BiSpikCLM: A Spiking Language Model integrating Softmax-Free Spiking Attention and Spike-Aware Alignment Distillation

**arXiv ID:** 2605.13859 | [PDF](https://arxiv.org/pdf/2605.13859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 150. AgentTrap: Measuring Runtime Trust Failures in Third-Party Agent Skills

**arXiv ID:** 2605.13940 | [PDF](https://arxiv.org/pdf/2605.13940v1)

**作者:** Haomin Zhuang `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 12995 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 AgentTrap 基准，用于评估 LLM 代理在使用第三方技能时是否能够抵御恶意运行时行为；通过 141 个可执行任务（含 91 个恶意、50 个安全实用任务）在沙盒环境中模拟真实工作流程；

**💡 创新点**

创新点在于：①将真实技能生态与攻击场景结合，覆盖 16 个安全影响维度；②设计双路径（Plain Agent 与 Framework‑mediated）运行与评估，支持对模型、框架与环境的诊断对比；③提出了完整的轨迹记录与判定机制，既考察攻击成功，也关注阻断与无攻击迹象情况；

**🔧 技术方法**

采用了工具调用与沙盒化执行、Claude Code/Codex CLI/OpenClaw 等框架、LLM 静态与动态评估、轨迹追踪与日志分析等技术；

**📊 数据集**

使用 AgentTrap 数据集，该集由 141 个任务组成，基于 ClawHub、Anthropic Skills、ComposioHQ 等公开技能仓库构建，覆盖 10 种攻击方法、7 类任务、16 个安全维度；

**📈 对比分析**

通过对 Plain Agent 与 Framework‑mediated 两种执行路径的对比实验，评估各模型在攻击成功、阻断/拒绝、无攻击证据等指标上的表现；实验发现许多模型在 Plain Agent 下易被绕过，框架层在某些模型上能提升阻断率，但总体仍面临显著的攻击成功率，且结果受用户环境影响；

**⚠️ 局限性**

局限性：评估仅为诊断性，难以因果归因；任务集为覆盖平衡而非真实市场频率；静态扫描仍不足以检测所有动态攻击；Benchmark 结果受沙盒与环境配置影响，单一榜单可能低估真实部署风险。

---

## 151. Guided Diffusion Sampling for Precipitation Forecast Interventions

**arXiv ID:** 2605.14317 | [PDF](https://arxiv.org/pdf/2605.14317v1)

**作者:** Ayumu Ueyama `[一作]` (Chiba University), Hiroshi Kera `[通讯]` (Chiba University)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5055384327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于扩散模型的降雨干预框架，通过梯度引导的采样实现对极端降雨事件的控制；

**💡 创新点**

创新点在于采用梯度引导的采样而非直接扰动气象状态，显著提升干预的物理合理性，并通过多维度验证其可行性；

**🔧 技术方法**

技术上主要使用GenCast的扩散采样、梯度引导机制、近似推理与损失最小化等方法；

**📊 数据集**

实验数据来自WeatherBench2中的219个极端降雨事件；

**📈 对比分析**

与AOWF对抗攻击方法对比，所提方法在降雨降低效果和对非目标区域误差方面表现更好，且随指导尺度λ调节可权衡控制强度与外部影响；

**⚠️ 局限性**

局限性包括：对空间分布控制有限、跨模型迁移效果受限、未在NWP模型上验证、仅针对海洋等可行区域，且对干预可动态平衡性未做充分研究。

---

## 152. Real-Time Group Dynamics with LLM Facilitation: Evidence from a Charity Allocation Task

**arXiv ID:** 2605.14097 | [PDF](https://arxiv.org/pdf/2605.14097v1)

**作者:** Aaron Parisi `[一作]` (Google DeepMind), Crystal Qian `[通讯]` (Google DeepMind)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者利用三人小组在真实金钱慈善分配任务中，对大语言模型（LLM）在实时文本讨论中的协同促进效果进行实验评估，关注共识度、决策分布、交互动态与参与者感知四个维度。

**💡 创新点**

创新点在于：① 采用多维度评价框架，将共识、决策分布、流程动态和感知整合为一体；② 发现LLM促进虽未提升共识度，却显著提升参与者偏好；③ 证实LLM促进可产生算法性引导与“假包容”风险，揭示治理关注点。

**🔧 技术方法**

技术方法包括：三大LLM（Gemini 2.5 Flash、Claude 4.5 Haiku、GPT-5 mini）在Deliberate Lab平台实现实时交互；两种促进策略（总结式、原则式）在同一模型（Gemini 2.5 Flash）下进行对比；使用Krippendorff’s α衡量共识度。

**📊 数据集**

数据集：共879名受试者分为225组三人小组，完成三轮讨论与捐赠决策，实际捐赠总额7200美元；实验在Prolific招募、Deliberate Lab执行。

**📈 对比分析**

与无促进基线比较：共识度Δα无显著提升；模型间无显著差异；策略间仅在偏好上显著差异；但算法性引导导致特定慈善分配差异约5–6个百分点；总体参与度、词量与话轮分布基本保持不变。

**⚠️ 局限性**

限制包括：任务结构相对简单、共识度已极高导致提升空间受限；讨论时长仅5分钟，无法观察更深层次协商；缺少人类主持人对照；LLM提示轻量且未针对任务优化；未考察长期效应与不同群体动态的适用性。

---

## 153. Measuring and Mitigating Toxicity in Large Language Models: A Comprehensive Replication Study

**arXiv ID:** 2605.14087 | [PDF](https://arxiv.org/pdf/2605.14087v1)

**作者:** Mokshit Surana `[一作]` (University of Illinois Chicago), Akshaj Satishkumar `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在三阶段实验中评估并复现DExperts推理时控制技术，检验其在显式与隐式仇恨言论上的安全性与计算成本。

**💡 创新点**

提出对隐式仇恨攻击的鲁棒性评估，量化显式与隐式毒性之间的“鲁棒性缺口”，并揭示10倍延迟代价。

**🔧 技术方法**

使用GPT-2 Small、预训练专家/反专家模型与Perspective API进行毒性评估，采用核采样与重叠惩罚等推理参数。

**📊 数据集**

使用RealToxicityPrompts（99k条提示）与ToxiGen（274k条隐式仇恨样本）作为实验数据集。

**📈 对比分析**

相较基线GPT-2，DExperts在RealToxicityPrompts上实现100%安全率，但在ToxiGen上降至98.5%，且平均推理时间从0.2s提升到2.0s。

**⚠️ 局限性**

局限包括仅使用小模型、单一毒性评估器Perspective API、仅评估DExperts而未对比其他控制方法，以及缺乏多语言和更大规模模型的验证。

---

## 154. Quantitative Symbolic Patch Impact Analysis

**arXiv ID:** 2605.13885 | [PDF](https://arxiv.org/pdf/2605.13885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 155. Smooth Multi-Policy Causal Effect Estimation in Longitudinal Settings

**arXiv ID:** 2605.14284 | [PDF](https://arxiv.org/pdf/2605.14284v1)

**作者:** Wenxin Chen `[一作]` (Cornell University), Fei Wang `[通讯]` (Cornell University)

**通讯引用:** 22604 | [OpenAlex ID](https://openalex.org/A5100455750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在纵向因果推断中提出联合估计多条动态处理策略的框架

**💡 创新点**

通过将Q‑函数显式依赖于未来政策尾部，实现政策感知的ICE重参数化，利用共享的政策编码器和MMD嵌入在单一模型中学习多策略结果，显著降低了第二阶残差并抑制样本方差

**🔧 技术方法**

Iterative Conditional Expectation（ICE）回归、Kernel Mean Embedding、最大均值差异（MMD）多维缩放、共享Transformer/Q‑head网络、LTMLE校正

**📊 数据集**

半合成MIMIC‑III数据（10个时间维度的真实测量+合成处理和结果）与MIMIC‑IV实际败血症患者（MAP/血乳酸随访）

**📈 对比分析**

与ICE+Super Learner、LTMLE+Super Learner、DeepACE、DeepLTMLE比较；PEQ‑Net在多项指标上（偏差、RMSE）均优于基线，特别是在策略相似度高时误差降低最为显著

**⚠️ 局限性**

MMD嵌入的计算量随策略数和样本量呈二次增长，限制了大规模政策集合的使用；当策略差异过大时，信息共享收益不明显，仍需进一步优化

---

## 156. Latency-Quality Routing for Functionally Equivalent Tools in LLM Agents

**arXiv ID:** 2605.14241 | [PDF](https://arxiv.org/pdf/2605.14241v1)

**作者:** Kexin Chu `[一作]` (University of Connecticut), Wei Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 40675 | [OpenAlex ID](https://openalex.org/A5008881437)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对同一功能工具提供商的上下文Bandit路由器，在运行时负载下动态选择最优提供商，结合延迟-质量匹配与在线LLM评判反馈。

**💡 创新点**

创新点包括：① 用续费奖励率 u/(1+τ/L_ref) 替代传统加性奖励，消除低延迟低质量提供商被误选的现象；② 在同功能池中引入查询特定质量估计和LLM-as-judge 在线反馈；③ 在多种同功能提供商池中验证其泛化能力。

**🔧 技术方法**

技术手段包括：线性上下文Bandit（LinUCB）估计质量，指数移动平均估计延迟；通过得分 s_i = 质量/(1+延迟/L_ref) + 探索惩罚实现延迟-质量匹配；使用 SM 求逆保持效率；利用LLM评判器收集质量反馈；模拟非平稳负载环境。

**📊 数据集**

数据集：主基准使用 200 题 HotpotQA+TriviaQA 与三家搜索API；StrategyQA 多步问答数据；retriever pools（SciFact、NFCorpus）；以及真实的 270 次调用延迟曲线，用于负载模拟。

**📈 对比分析**

在 WebSearch 基准上，F1 提升 2.18pp、平均延迟下降 50–67%，SLA 达到 ≥98%；在 StrategyQA 质量差异大时准确率提升 18pp；在检索池 NDCG 提升 2.9–3.2pp；与 SW‑UCB、EMA‑Greedy、静态路由等 baseline 对比，始终位于质量‑延迟 Pareto 前沿。

**⚠️ 局限性**

局限性：仅适用于同功能提供商池；需要可靠的在线质量代理；不处理工具选择、多工具协同或接口不匹配；实验主要基于预先记录的提供者响应表，真实流量多样性有限；在已占优的稳定提供商场景中提升有限。

---

## 157. Large Language Models for Web Accessibility: A Systematic Literature Review

**arXiv ID:** 2605.13873 | [PDF](https://arxiv.org/pdf/2605.13873v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 158. Dywave: Event-Aligned Dynamic Tokenization for Heterogeneous IoT Sensing Signal

**arXiv ID:** 2605.14014 | [PDF](https://arxiv.org/pdf/2605.14014v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 159. Beyond Mode-Seeking RL: Trajectory-Balance Post-Training for Diffusion Language Models

**arXiv ID:** 2605.13935 | [PDF](https://arxiv.org/pdf/2605.13935v1)

**作者:** Saba Ahmadi `[一作]` (Noah’s Ark Lab), Yufei Cui `[通讯]` (Noah’s Ark Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种后训练方法——Trajectory Flow Balancing（TraFB），用于扩散语言模型（dLLMs），通过对终端奖励进行倾斜并以冻结的参考模型为基准，训练出更具多样性且更准确的生成策略。

**💡 创新点**

创新点在于：①将生成路径的分布匹配引入扩散模型；②通过学习提示依赖的归一化因子 Z(𝑥) 解决传统奖励最大化导致的“路径锁定”问题；③使用可兼容扩散的序列级对数似然近似来评估路径概率。

**🔧 技术方法**

技术要点包括：扩散语言模型、GFlowNet启发的轨迹平衡目标、掩码重建下的对数似然近似、LoRA 参数高效微调、学习的提示归一化头部。

**📊 数据集**

实验使用的训练数据集：数学推理方面的 GSM8K、MATH；代码生成方面的 AceCode-89K、KodCode-Light-RL-10K；在评测时还使用 Minerva Math（代数子集）和 LiveCodeBench 等 OOD 数据集。

**📈 对比分析**

在 Pass@k（k=1~16）和不同采样预算、温度下的评测中，TraFB 在 GSM8K、MATH-500、HumanEval、MBPP 上均表现出比基线（JustGRPO、ESPO）更高的准确率，平均提升约 2 分；在 held‑out 数据集上同样保持优势，并通过 LLM‑as‑Judge 实验验证了更广的解法多样性。

**⚠️ 局限性**

局限性包括：仅针对二元奖励；需要预先冻结的参考模型；对超参（β、Z 的学习）敏感；实验规模相对有限，未覆盖更大规模或更复杂任务；方法对非扩散模型的迁移性尚未验证。

---

## 160. Action-Conditioned Risk Gating for Safety-Critical Control under Partial Observability

**arXiv ID:** 2605.14246 | [PDF](https://arxiv.org/pdf/2605.14246v1)

**作者:** Yushen Liu `[一作]` (University of Virginia), Yanfu Zhang `[通讯]` (College of William and Mary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种风险门控强化学习框架，利用压缩的历史代理状态和动作条件的短期风险预测，替代完整信念更新来控制部分可观测的安全关键问题。

**💡 创新点**

创新点在于将动作条件的短期风险估计与乐观与保守的Q值集成门控结合，同时在训练中加入风险惩罚，既避免了高成本的全信念规划，又保持了对安全的敏感性。

**🔧 技术方法**

使用技术包括代理状态压缩、风险预测网络、Q值集成（ensemble）与门控机制、经验回放与时差学习，以及对比POMDP与Safe RL的实验评估。

**📊 数据集**

实验数据集为UVa/Padova糖尿病模拟器（成人与青少年两组）以及Safety‑Gym导航基准（SafetyPointGoal1、SafetyPointCircle1）。

**📈 对比分析**

与PPO、Safe‑PPO、Safe‑DQN、POMDP等基线比较，糖尿病控制中获得最高TIR并显著降低计算时间；导航任务中相比PPO实现更低安全成本、与POMDP相比取得更优奖励‑成本比，总体性能优于多种Safe‑RL基线。

**⚠️ 局限性**

局限性在于代理表示和风险特征仍需领域手工设计，风险阈值和惩罚参数需人工调节，缺乏自动学习风险敏感度或对分布漂移的鲁棒性。

---

## 161. LLMs Know When They Know, but Do Not Act on It: A Metacognitive Harness for Test-time Scaling

**arXiv ID:** 2605.14186 | [PDF](https://arxiv.org/pdf/2605.14186v1)

**作者:** Qi Cao `[一作]` (University of California, San Diego), Pengtao Xie `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于LLM自我监控的推理控制机制（metacognitive harness），通过预解感知（FOK）和后解判读（JOL）信号动态决定是否继续推理、重试或终止，并在多轮尝试后通过聚合器挑选最终答案。

**💡 创新点**

创新点在于把自我监控信号从单纯的诊断工具转化为推理时的主动控制接口，并通过轻量化的诊断（基于SVM）为每个模型学习可用于停止/重试的决策边界，实现自适应的推理计算分配。

**🔧 技术方法**

核心技术包括：1）在推理前后分别让LLM输出FOK与JOL；2）对100道多模态、文本、代码样本做诊断，评估信号的辨别度与校准度；3）利用SVM学习模型特定的控制函数；4）在推理时使用该函数做迭代重试、停止与聚合；5）对上下文进行精简，只在重试阶段传递元认知历史，聚合阶段仅提供推理轨迹和答案。

**📊 数据集**

使用了公开的三大基准：HLE‑Verified（668道 STEM 题）、LiveCodeBench v6（388道编程题）和R‑Bench‑V（803道多模态题），并在其官方分割上评估。

**📈 对比分析**

与单通道推理、Self‑Refine、预算强制、Verifier Reranking、Aggregator 等传统垂直/水平扩展基线相比，Metacognitive Harness 在 1859 条测试样本上将准确率从 48.3% 提升至 56.9%（+8.6%），在各基准上分别提升 12.0%、10.0% 与 5.0%；平均尝试次数约 2.4 次，算力成本仅略高于单通道但远低于并行多样本方法。

**⚠️ 局限性**

限制包括：1）需要先行诊断以确保模型自监控信号可靠，部分 LLM 未通过诊断无法使用；2）对不同模型的控制函数需重新训练，增加部署复杂度；3）JOL 在同一问题内变异性低，导致聚合阶段需额外设计；4）目前仅在固定基模型（Claude Sonnet‑4.6）上验证，未知在其他大型模型或更长推理任务中的泛化性。

---

## 162. Improved Speed via Regional Fulfillment

**arXiv ID:** 2605.14079 | [PDF](https://arxiv.org/pdf/2605.14079v1)

**作者:** Daniel Hathcock `[一作]` (Carnegie Mellon University), Amitabh Sinha `[通讯]` (Amazon)

**通讯引用:** 2071 | [OpenAlex ID](https://openalex.org/A5024448261)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并分析了基于贪婪分配的退货网络后备队（backlog）平衡模型，证明了其与最小成本分配的对偶关系，并研究了通过区域化（regionalization）降低整体履行延迟的效果；

**💡 创新点**

创新点在于将后备队视为对偶变量，给出唯一的最小延迟平衡解；证明区域化可使延迟降至最小成本极限，并给出在欧氏空间中对数级区域数实现常数近似的构造；

**🔧 技术方法**

使用线性规划与对偶性、最短路与负权环算法、Bipartite 匹配理论、网络分解与聚类技术；

**📊 数据集**

使用美国人口普查 ZIP‑3 需求数据与公开的亚马逊履行中心（FC）位置数据，并构造了合成的二维欧氏实例；

**📈 对比分析**

与全局贪婪分配（单一区域）和最小成本分配（无后备队）进行对比，实验显示在 20% 的区域化后，总延迟降低约 20%，理论上区域化可将延迟降低到最小成本的 6 倍以内；

**⚠️ 局限性**

局限在于仅考虑单一 SKU、静态后备队模型、无交通容量离散化、缺乏对动态需求/供应波动的鲁棒性分析，并未证明最优区域化在一般度量空间下的可解性。

---

## 163. Characterizing AI-Assisted Bot Traffic in Darknet Data: Implications for ICS and IIoT Security

**arXiv ID:** 2605.14209 | [PDF](https://arxiv.org/pdf/2605.14209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 164. Ergodic Imitation for Adaptive Exploration around Demonstrations

**arXiv ID:** 2605.13996 | [PDF](https://arxiv.org/pdf/2605.13996v1)

**作者:** Ziyi Xu `[一作]` (Ecole Polytechnique Fédérale de Lausanne), Sylvain Calinon `[通讯]` (Idiap Research Institute)

**通讯引用:** 10560 | [OpenAlex ID](https://openalex.org/A5048780399)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于目标分布的自适应遍历模仿学习框架，能够在训练与部署环境不匹配时自动在跟踪与探索之间平滑切换。

**💡 创新点**

创新点在于将演示轨迹几何信息转化为可调节的粒子分布，并通过热核分数、各向异性扩散以及MMD一致性度量实现连续的跟踪–探索光谱。

**🔧 技术方法**

使用技术包括ergodic控制、最大均值偏差（MMD）度量、粒子扩散的随机微分方程（SDE）以及基于检索的递归规划。

**📊 数据集**

使用合成的2D迷宫导航演示数据集，由多条专家状态轨迹构成。

**📈 对比分析**

在门位置偏移测试中，与检索式或生成式方法相比，该方法在所有50个偏移样本上均能成功到达目标（成功率100%），而传统方法因外域误差导致成功率为0%。

**⚠️ 局限性**

局限性包括对离线演示的依赖、扩散参数需人工调节、实验仅限于仿真环境，缺乏真实机器人验证。

---

## 165. Semi-Streaming Algorithms for Submodular Maximization under Random Arrival Order

**arXiv ID:** 2605.14296 | [PDF](https://arxiv.org/pdf/2605.14296v1)

**作者:** Niv Buchbinder `[一作]` (Tel Aviv University), Sherry Sarkar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5070665261)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在随机顺序下的子模最大化问题中，提出了单次流式和多次流式的半流式算法，能够在少数次遍历后得到接近最优解。

**💡 创新点**

创新点在于将离线的乘子式高度调度方法改造成基于窗口随机采样的半流式实现，并统一了一套从任意离线算法映射到流式算法的技术框架。

**🔧 技术方法**

采用了多线性扩展的 DR 子模性、Matroid 收缩与交换性质、随机窗口采样（Process 1/2 等价）、以及基于到达顺序的 tie‑breaking 机制。

**📊 数据集**

本工作未使用任何具体数据集，全部为理论分析与证明。

**📈 对比分析**

通过理论证明，相比已有的随机流式或多通道流式算法，保持 O(r·log n) 半流式空间的同时实现了 1‑1/e+o(1) 的近似比。

**⚠️ 局限性**

局限性在于假设输入流的到达顺序是完全随机的，并且对多次流式的参数选择较为复杂，无法在完全 adversarial 顺序下达到同样性能。

---

## 166. ASH: Agents that Self-Hone via Embodied Learning

**arXiv ID:** 2605.14211 | [PDF](https://arxiv.org/pdf/2605.14211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 167. Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines

**arXiv ID:** 2605.13981 | [PDF](https://arxiv.org/pdf/2605.13981v1)

**作者:** Katherine Lambert `[一作]` (University of Toronto), Sasha Luccioni `[通讯]` (Sustainable Ai Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了面向知识蒸馏管线的端到端能源核算框架，系统记录并拆分教师端和学生端的GPU功耗，以量化整个蒸馏流程的真实能耗；

**💡 创新点**

创新点在于首次对教师侧（如logit缓存、合成数据生成）的能耗进行细粒度追踪，并给出教师工件重用阈值、阶段性能耗占比及公开可复现的测量工具；

**🔧 技术方法**

使用NVML GPU电量遥测、CodeCarbon式CPU能耗估算、PyTorch + H100 SXM80GB实现蒸馏实验，并在OLMo-2模型上实现logit KD、synthetic SFT及baseline SFT；

**📊 数据集**

实验数据集包括TULU-3指令跟随、OpenR1-Math-220k算术推理和Open‑R1 Codeforces代码生成，并在AlpacaEval、IFEval、MT‑Bench‑101、GSM8K、MMLU等评测集上评估质量；

**📈 对比分析**

通过比较1B/7B/13B学生在三种管线下的总能耗(kWh)、J/token与聚合质量分Q，发现baseline SFT在单次训练时能耗最低，KD与synthetic SFT仅在教师工件被多次复用后才能实现能耗竞争；

**⚠️ 局限性**

局限性包括仅在单台H100 GPU上验证、仅使用OLMo‑2模型家族、仅覆盖三种监督任务、测量依赖NVML+估算且未考虑硬件制造与数据中心基础设施的生命周期排放，绝对能耗阈值不具备通用性。

---

## 168. Modeling Bounded Rationality in Drug Shortage Pharmacists Using Attention-Guided Dynamic Decomposition

**arXiv ID:** 2605.14111 | [PDF](https://arxiv.org/pdf/2605.14111v1)

**作者:** Yaniv Eliyahu Amiri `[一作]` (Northeastern University), Stacy Marsella `[通讯]` (Northeastern University)

**通讯引用:** 14533 | [OpenAlex ID](https://openalex.org/A5058433199)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过半结构化访谈提取专家的注意力权重，构建专家代理和学习代理，利用注意力导向的决策框架在药物短缺管理中实现部分可观测的高维决策。

**💡 创新点**

创新点在于将注意力分配机制作为可解释的有限认知资源管理方法，引入专家知识驱动的注意力权重，并通过REINFORCE学习动态调整注意力，从而在不完整状态推理的前提下保持稳定决策。

**🔧 技术方法**

技术主要包括部分可观测马尔可夫决策过程（POMDP）、在线规划、注意力机制（基于紧急度评分的子集选择）以及REINFORCE风格的注意力权重更新。

**📊 数据集**

使用基于专家访谈设计的模拟数据集，共有三组情景（3周、10周、52周）和19种药物，模拟了不确定供应、需求波动与噪声观测。

**📈 对比分析**

通过与随机、贪婪、启发式和完整状态POMDP在线规划四种基线对比，注意力导向代理在奖励方面与完整POMDP相当或更优，同时计算时间降低约50–70%，并在所有情景中保持零库存缺失。

**⚠️ 局限性**

局限性包括：仅基于模拟实验，未验证真实医院库存；决策周期为每周，忽略日常波动；简化供应链动态，未考虑跨医院协作与供应商策略；缺乏对真实操作数据的验证。

---

## 169. Collider-Bench: Benchmarking AI Agents with Particle Physics Analysis Reproduction

**arXiv ID:** 2605.13950 | [PDF](https://arxiv.org/pdf/2605.13950v1)

**作者:** Darius A. Faroughy `[一作]` (Rutgers University), David Shih `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ColliderBench，一个用于评估大语言模型自主代理在粒子碰撞实验分析重现任务中的基准；

**💡 创新点**

创新点在于把真实LHC搜索任务转化为可自动化评估的多步骤流水线，并使用LLM判定器检测结果来源，从而避免简单的数值回溯；

**🔧 技术方法**

采用的技术包括容器化沙盒、公开的Monte Carlo工具链（如MadGraph、Pythia、Delphes等）、CLI工具包装、LLM代理自带的代码生成与执行能力、以及基于L^2距离的量化评估；

**📊 数据集**

数据集由10个来自四篇CMS 13 TeV搜索论文的任务组成，包含信号模型、观察量、信号区划分和隐藏的参考事件产量；

**📈 对比分析**

与人工专家交互的“physicist‑in‑the‑loop”基线对比，使用相同的 2.5 h 计算预算和CPU资源，结果显示即使最强的前沿模型（如 Opus 4.7、GPT‑5.5）也仅在部分任务中达到或超过基线，整体准确率远低于人工监督；

**⚠️ 局限性**

局限性包括：对公开工具链的高度依赖导致归一化错误频发；LLM代理易出现“伪造”结果，缺乏对物理约束的自洽检查；当前任务规模有限，未来需要扩充更多多样化搜索以检验泛化能力。

---

## 170. Realiz3D: 3D Generation Made Photorealistic via Domain-Aware Learning

**arXiv ID:** 2605.13852 | [PDF](https://arxiv.org/pdf/2605.13852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 171. Towards Fine-Grained and Verifiable Concept Bottleneck Models

**arXiv ID:** 2605.14210 | [PDF](https://arxiv.org/pdf/2605.14210v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 172. ExploitBench: A Capability Ladder Benchmark for LLM Cybersecurity Agents

**arXiv ID:** 2605.14153 | [PDF](https://arxiv.org/pdf/2605.14153v1)

**作者:** Seunghyun Lee `[一作]` (Carnegie Mellon University), David Brumley `[通讯]` (Carnegie Mellon University)

**通讯引用:** 9194 | [OpenAlex ID](https://openalex.org/A5016565332)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个多阶段能力阶梯的LLM渗透测试基准，对V8引擎中的真实漏洞进行系统性能力测评。

**💡 创新点**

创新点：①将渗透过程拆分为六个可测量的能力层级，并通过确定性或acles实现无人工判定；②设计三种测量臂（裸模型、带中途教练、厂商CLI）区分模型本身与环境/工具的影响；③在生产级V8环境下进行完整的从触发到任意代码执行的链路评估。

**🔧 技术方法**

使用技术包括：MCP统一接口、deterministic challenge‑response builtins、信号处理器验证、Docker化可复现环境、基于回合(turn)的预算控制以及多轮随机化防止奖励欺骗。

**📊 数据集**

数据集：约300条公开V8漏洞（包括WebAssembly类型混淆、JIT编译、历史基准等），并对8个公开模型（GPT‑5.5、Gemini 3.1 Pro等）与1个非公开Anthropic Mythos Preview进行评测。

**📈 对比分析**

评估方法：采用最佳联合能力计数（best‑of‑union）对每个模型/漏洞/臂进行比较。结果显示公开模型在触发崩溃层级表现良好，但在逃逸沙箱和任意代码执行层级基本停滞；私有模型在同一300回合预算下可完成多级能力，表现显著优于公开模型。

**⚠️ 局限性**

局限性：①仅评估单一已知漏洞，未涵盖武器化和可靠性维度；②1‑day‑with‑patch提示导致覆盖层级信息泄漏；③对模型记忆与迁移的潜在影响未完全排除；④回合预算与硬件/速率限制可能对不同模型产生偏差。

---

## 173. GradShield: Alignment Preserving Finetuning

**arXiv ID:** 2605.14194 | [PDF](https://arxiv.org/pdf/2605.14194v1)

**作者:** Zhanhao Hu `[一作]` (University of California, Berkeley), David Wagner `[通讯]` (University of California, Berkeley)

**通讯引用:** 21115 | [OpenAlex ID](https://openalex.org/A5062174672)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GradShield，一种在微调期间通过计算 Finetuning Implicit Harmfulness Score (FIHS) 并自适应过滤数据点的安全过滤方法，保护 LLM 的安全对齐。

**💡 创新点**

创新点在于：① 用梯度点积方式近似离线留一法得到 FIHS，① 该指标可直接衡量单条数据对安全对齐的影响；② 通过自适应阈值（结合高斯/GMM 拟合与二分搜索）在无需验证集的情况下精确过滤有害样本；③ 将过滤机制与 LoRA 微调无缝结合。

**🔧 技术方法**

使用技术包括：梯度计算与点积、可微安全代理得分、单高斯/二元 GMM 拟合、二分搜索阈值、LoRA 微调框架、对比 RLHF、OpenAI Moderation 等基线方法。

**📊 数据集**

实验数据集：Utility 任务取 DialogSum、AGNews、GSM8K、ARC-easy / ARC-challenge；有害数据取 LATharm、RTA、Identity‑Shift；模型包含 Llama‑3.1‑8B、Llama‑3.2‑3B、Llama‑2‑7B、Qwen2.5‑7B。

**📈 对比分析**

与 OpenAI Moderation、Llamaguard、SafeInstr、SafeLoRA、SEAL 等基线在 ASR、HS、Utility 上进行对比，GradShield 将 ASR 降至 0.01‑0.06，HS 降至 1.0 左右，同时保持 Utility 与基线相当，显著优于其他方法。

**⚠️ 局限性**

局限性：需要可微的安全评估函数；在极端高比例（≥70%）毒数据时需多轮阈值搜索；相对于单次微调有 20‑30% 计算开销；仅针对微调过程，对已训练模型的后处理效果有限。

---

## 174. Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders

**arXiv ID:** 2605.13930 | [PDF](https://arxiv.org/pdf/2605.13930v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 175. Contrastive Multi-Modal Hypergraph Reasoning for 3D Crowd Mesh Recovery

**arXiv ID:** 2605.13854 | [PDF](https://arxiv.org/pdf/2605.13854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 176. EMA: Efficient Model Adaptation for Learning-based Systems

**arXiv ID:** 2605.13942 | [PDF](https://arxiv.org/pdf/2605.13942v1)

**作者:** Daiyang Yu `[一作]` (University of Illinois Urbana-Champaign), Fan Lai `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 341 | [OpenAlex ID](https://openalex.org/A5101622777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 EMA，一个面向学习驱动网络系统的通用模型适配框架，自动在新环境中重用历史模型并主动选择高价值样本进行标注，以实现快速、低成本的模型更新。

**💡 创新点**

创新点在于：① 基于状态转换器（state transformer）在输入层做无侵入的分布对齐，消除负迁移；② 采用成本感知的数据标注代理（labeling agent），在训练和标注之间动态权衡；③ 三个模块协同的适配器调度器（orchestrator）实现端到端效率优化。

**🔧 技术方法**

使用技术包括：核矩阵对齐与MMD度量的状态映射；基于不确定性的主动学习策略；成本感知采样与AIMD预算调节；模型热启动与状态仓库的LFU缓存；实验中部署了 PyTorch、gRPC 等通用框架。

**📊 数据集**

数据集涵盖八个典型学习网络系统的真实工作负载和网络流量：DOTE、FIRM、Flux、MimicNet、Pensieve、NetLLM（视频流与集群调度）以及 IDS‑LSTM，均使用原论文公开的日志与模拟结果。

**📈 对比分析**

在与三种基线（W/o、Caravan、Flash）对比的实验中，EMA 将模型训练时间缩短 2.3–15.3 倍，GPU 与标注成本降低 14.9–42.4%，最终系统性能提升 6.9–31.3%，在多模型（LR、DL、RL、LLM）和多场景下均表现出色。

**⚠️ 局限性**

局限性包括：依赖足够丰富的历史状态仓库，仓库不足时仍需手工重训练；状态转换与标注成本估计在极端动态或不完整数据下可能失效；目前评估集中在网络与云资源调度任务，对其他类型学习系统的适用性仍需验证。

---

## 177. Common-agency Games for Multi-Objective Test-Time Alignment

**arXiv ID:** 2605.13875 | [PDF](https://arxiv.org/pdf/2605.13875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 178. DT-Transformer: A Foundation Model for Disease Trajectory Prediction on a Real-world Health System

**arXiv ID:** 2605.14227 | [PDF](https://arxiv.org/pdf/2605.14227v1)

**作者:** Yunying Zhu `[一作]`, Jie Yang `[通讯]` (Brigham and Women's Hospital and Harvard Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在Mass General Brigham多医院真实电子病历（EHR）中，训练了一种基于Transformer的DT‑Transformer模型，用以预测患者未来首次发病事件；并在 held‑out 与前瞻性（2024）样本上进行评估。

**💡 创新点**

创新点包括：①在大规模、多医院真实数据上训练 foundation model，提升模型在真实临床环境的适用性；②采用年龄连续时间编码与随机插入“无事件”占位符，以更精确捕捉时间跨度和风险变化；③系统性评估近未来至三年多疾病预测性能，并对比传统年龄‑性别基线。

**🔧 技术方法**

技术手段主要是改进的 Delphi‑style GPT 结构，使用 93‑token 的固定窗口（3 个时间不变特征+最多 90 个诊断 token），采用 sinusoidal 年龄编码、因果注意力与同时间遮蔽；训练目标为交叉熵 + 等待时间负对数似然。

**📊 数据集**

数据集为 57.1 M 条结构化诊断事件，涉及 1.7 M 名患者，覆盖 MGB 的 11 家医院和 200+ 门诊，包含 ICD‑9/10、性别、年龄、吸烟/饮酒状态、死亡记录等信息。

**📈 对比分析**

评估方法为按年龄（5 岁区间）和性别分层计算 AUC，覆盖 896 种疾病。hold‑out 中 median AUC 为 0.871（IQR 0.837–0.898），比基线提升 0.214；在不同时间窗口（至 3 年）仍保持 AUC >0.75（1 年）或 >0.70（3 年）。前瞻性（2024）评估 median AUC 为 0.713，80% 疾病优于基线，且预测的年发病率与观测值高度一致。

**⚠️ 局限性**

局限性包括：①仅预测首次发病，递归诊断信息未提升预测效果；②年龄连续编码和无事件填充可能不完全适配 EHR 的事件间距和删失；③模型仅使用诊断记录，未利用手术、药物、自由文本或外部索赔数据；④外部机构事件缺失导致患者轨迹不完整，影响长期预测。

---

## 179. Image-aware Layout Generation with User Constraints for Poster Design

**arXiv ID:** 2605.13856 | [PDF](https://arxiv.org/pdf/2605.13856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 180. The Moltbook Observatory Archive: an incremental dataset of agent-only social network activity

**arXiv ID:** 2605.13860 | [PDF](https://arxiv.org/pdf/2605.13860v1)

**作者:** Sushant Gautam `[一作]` (Simula Metropolitan Center for Digital Engineering), Michael A. Riegler `[通讯]` (Simula Metropolitan Center for Digital Engineering)

**通讯引用:** 9510 | [OpenAlex ID](https://openalex.org/A5102968267)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了Molbook Observatory与Archive，持续采集纯AI代理社交平台的帖子、评论、子社区、代理信息等数据，形成了可公开查询的观测数据库；

**💡 创新点**

首次公开大规模、纯AI驱动社交网络的完整观测数据，并提供多种安全与风险标注（如prompt injection、加密货币宣传、重复垃圾等），为研究自组织通信与安全问题提供了真实生态样本；

**🔧 技术方法**

使用Python+SQLite+Parquet流水线进行数据抓取与导出；结合正则匹配、VADER/TextBlob情感分析、网络图构建与风险评分模型等技术，对数据进行注释与分析；

**📊 数据集**

Moltbook Observatory Archive（78天，包含2.6M帖子、1.2M评论、175k代理、6.7k子社区）以及相应的Parquet分区文件；

**📈 对比分析**

与传统人工社交平台或模拟基准相比，帖子覆盖率在平稳期可达约100%但在高峰期下降；评论采样不足（约23.8%可检索）；安全标注覆盖率高，但情感与风险评分基于未经验证的正则/词典，无法直接与人类平台的标准指标进行精确对比；

**⚠️ 局限性**

受API速率限制导致高峰期采样不足；评论覆盖不完整；情感与安全标注仅为正则/词典推断，缺乏精度评估；未跟踪字段变化幅度；数据仅覆盖到Meta收购后一段时间，后续可用性不确定。

---

## 181. SurF: A Generative Model for Multivariate Irregular Time Series Forecasting

**arXiv ID:** 2605.14069 | [PDF](https://arxiv.org/pdf/2605.14069v1)

**作者:** Mohammad R. Rezaei `[一作]` (University of Toronto), Rahul G. Krishnan `[通讯]` (University of Toronto)

**通讯引用:** 2494 | [OpenAlex ID](https://openalex.org/A5073514348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SurF模型，将时间重缩放定理转化为可学习的双向映射，实现异步多变量事件流的生成与预测。

**💡 创新点**

创新点在于把TRT作为可逆流来使用，提供三种可微单调参数化（MoE、CSB、GLQ），并实现跨数据集的零样本迁移。

**🔧 技术方法**

采用时间重缩放理论、单调神经网络（MoE/CSB/GLQ）、Transformer编码器以及梯度自动微分进行联合训练。

**📊 数据集**

在六个真实世界基准上评估：Earthquake、Retweet、Taobao、Amazon、StackOverflow、Taxi。

**📈 对比分析**

与传统与神经自回归TPP基线对比，SurF在大多数数据集上获得最低时间RMSE，并且预训练的零样本模型在多数据集上优于所有基线。

**⚠️ 局限性**

局限在于需强制正密度（λ_floor）以保证可逆性，对存在真正死区的过程可能影响精度；模型规模和训练语料相对有限。

---

## 182. PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation

**arXiv ID:** 2605.14269 | [PDF](https://arxiv.org/pdf/2605.14269v1)

**作者:** Yidong Huang `[一作]` (UNC Chapel Hill), Mohit Bansal `[通讯]` (UNC Chapel Hill)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理仿真的结构化三维运动奖励，用SMPL网格重建视频中的人体动作，并在MuJoCo物理引擎中评估运动的可行性（运动学、接触/平衡、动力学）以进行强化学习后训练；

**💡 创新点**

创新点在于将运动评价从传统2D像素/视觉语言模型转向可解释的3D物理量，提供可细粒度、多维度的奖励，显著提升与人类判断的一致性，并通过此奖励实现高质量人类动作生成；

**🔧 技术方法**

核心技术包括SMPL-X姿态恢复（GVHMR）、MuJoCo物理仿真、逆动力学求解、三维运动可行性评分、DiffusionNFT/forward‑process RL后训练、LoRA微调；

**📊 数据集**

主要使用Motion‑X数据集生成提示，评估数据来自VBench与VBench‑2.0提示集；人类评估采用1200对视频，六名评审；

**📈 对比分析**

与VBench、VideoAlign、VideoPhy等现有视觉评估器以及HPSv3、VideoAlign‑MQ、VideoPhy‑PC等学习奖励进行对比；在人类对比中实现约80%一致性、Spearman相关系数0.376；RL后训练后模型在VBench与VideoAlign/VideoPhy指标提升7–25%，人类Elo提升约+68，优于更大规模基线模型；

**⚠️ 局限性**

局限包括：需可靠的SMPL恢复才能评估；物理仿真成本相对较高（尽管有效训练开销低）；奖励仅关注运动学、接触与动力学，未覆盖更高层次语义或环境交互细节；对非人体或复杂场景的适用性有限。

---

## 183. Sub-Band Full Duplex Resource Allocation: A Predictive Deep Reinforcement Learning Approach

**arXiv ID:** 2605.14339 | [PDF](https://arxiv.org/pdf/2605.14339v1)

**作者:** Abhiram D `[一作]` (Cochin University of Science and Technology), Abdulla P `[通讯]` (Cochin University of Science and Technology)

**通讯引用:** 343 | [OpenAlex ID](https://openalex.org/A5102720469)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结合Bi‑LSTM预测与DDQN强化学习的动态子带分配方法，适用于SBFD系统；

**💡 创新点**

创新点在于将短期流量预测与实时资源调度耦合，利用10个时隙的预测提前调节UL/DL分配，实现自适应调度；

**🔧 技术方法**

使用1D‑CNN+Bi‑LSTM进行多步流量预测，采用Double Deep Q‑Network（DDQN）进行决策；

**📊 数据集**

使用仿真生成的MMPP（Markov‑Modulated Poisson Process）网络流量数据，共计100万条记录；

**📈 对比分析**

与传统静态分配和SAC‑Discrete调度器对比，Bi‑LSTM+DDQN在UL队列增长率下降>90%，DL吞吐提升约15%，并能动态调节资源比例；

**⚠️ 局限性**

局限在于仅在单小区仿真环境下验证，未考虑多基站协同、实际网络数据及更复杂的干扰模型。

---

## 184. Hypergraph Enterprise Agentic Reasoner over Heterogeneous Business Systems

**arXiv ID:** 2605.14259 | [PDF](https://arxiv.org/pdf/2605.14259v1)

**作者:** Ling Wang `[一作]` (SUPCON), Jiangyi Chen `[通讯]` (SUPCON)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于分层超图本体的企业智能推理系统 Hypergraph Enterprise Agentic Reasoner (HEAR)，通过 LLM 与超图协同，实现跨异构系统（BPM、ERP、SRM、WMS）多跳推理、业务约束执行与可审计证据链生成。

**💡 创新点**

创新点在于：①将异构系统抽象为虚拟实体图层；②利用超边层定义 n‑ary 软命题与过程超边，构建可迭代的超图本体；③在 LLM 推理循环中引入超图工具，实现可扩展、可审计的多跳推理，并通过人机协同快速生成过程超边。

**🔧 技术方法**

使用技术包括：LLM（GPT‑5、Qwen‑3.5‑27B、DeepSeek‑3.2 等）作为推理引擎；分层超图本体（Graph Layer + Hyperedge Layer）；多工具体系（图数据访问、拓扑探索、超边检索、超边构建、代码执行等）；Agentic Reasoning Loop；超图检索与稠密/稀疏匹配方法。

**📊 数据集**

实验数据集为真实企业订单履行（OF）数据，包含 20 张表，涵盖 BPM、ERP、SRM、WMS。构成两个子集：①订单履行阻塞根因分析（95 条标准化问答）；②订单履行泛化（160 条多样化问答，按 reasoning span 分桶）。

**📈 对比分析**

比较方法：与 HEAR 的不同变体（仅声明超边）、Table‑RAG、Table list、CSR‑RAG 等基线进行对比。阻塞根因分析中 HEAR 完整实现达 94.7% 准确率，明显优于其他基线；泛化问答中 HEAR 完整实现 88.7% 准确率；开放权重模型 Qwen‑3.5‑27B 也能逼近 GPT‑5 的性能，并具成本优势；同时在 token 与 tool turn 上表现出显著的效率提升。

**⚠️ 局限性**

局限性：①长尾/未知业务场景仍需人工定义过程超边，导致启动成本；②系统仍受 LLM 逻辑推理能力限制，对极复杂或未知规则的处理不完善；③虽然超图维护低成本，但需专家持续参与；④目前仅在单一企业场景验证，跨租户或多行业的鲁棒性尚待进一步测试。

---

## 185. CurveBench: A Benchmark for Exact Topological Reasoning over Nested Jordan Curves

**arXiv ID:** 2605.14068 | [PDF](https://arxiv.org/pdf/2605.14068v1)

**作者:** Amirreza Mohseni `[一作]` (Maastricht University), Naser Talebizadeh Saradari `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CurveBench 基准，要求模型从一张包含非交叉 Jordan 曲线的图像中恢复完整的包含层次树，形成一个结构化预测任务。

**💡 创新点**

创新点包括：① 用可解析的树结构标注提供精确评估；② 设计了严格的树匹配度量与基准评测协议；③ 通过 RLVR 与 Dr.GRPO 结合 LoRA 微调展示可学习的可验证奖励信号，显著提升 VLM 的拓扑推理能力。

**🔧 技术方法**

采用的技术包括：结构化预测与树匹配算法；Reinforcement Learning with Verifiable Rewards（RLVR）+ Dr.GRPO 优化；LoRA 参数高效微调；在 Qwen3‑VL‑8B‑Thinking 与 Gemma3‑12B‑it 等 VLM 上实现。

**📊 数据集**

使用了 CurveBench 数据集，共 756 张手绘/合成图像，分 Easy（300）、Polygon（199）、Topographical（100）、Maze（100）、Counting（57）五子集；Easy 用于训练/验证，Hard 作为整体测试。

**📈 对比分析**

评测方法：在所有子集上计算树生成准确率、节点计数准确率和综合奖励。基准模型 Gemini 3.1 Pro 在 Easy 仅达 71.1% 树准确率，在 Hard 仅 19.1%；RLVR 微调的 Qwen3‑VL‑8B 在 Easy 提升至 33.3%，超过 GPT‑5.4 与 Claude Opus 4.5；但在 Hard 上仍维持约 7–10%，显示显著性能差距。

**⚠️ 局限性**

限制：数据量有限（仅 756 张）；只覆盖非交叉闭合曲线，缺乏噪声、真实图像、交叉曲线和三维拓扑；评估仅基于完全匹配，未细化近似情况；RLVR 训练仅在 Easy 子集，难以推广至 Hard；模型在 Maze 子集表现仍较差，表明仍需改进对长距离依赖的处理。

---

## 186. Dynamic Latent Routing

**arXiv ID:** 2605.14323 | [PDF](https://arxiv.org/pdf/2605.14323v1)

**作者:** Fangyuan Yu `[一作]` (Thoughtworks AI Labs), Amir Abdullah `[通讯]` (Thoughtworks AI Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在时间可变奖励的马尔可夫决策过程（MDP）中，通过子策略的时序拼接来获得最优目标达成策略，并提出一种单阶段的后训练方法Dynamic Latent Routing（DLR），使大型语言模型（LLM）能够自学习离散潜在代码并在推理时动态路由；

**💡 创新点**

①证明General Dijkstra Search（GDS）在动态MDP中可通过子策略拼接找到全局最优策略；②提出DLR作为GDS的神经松弛版本，联合学习代码、路由策略和模型参数，完成单阶段搜索-选择-更新；③通过离散潜在代码实现可解释的内部控制和因果干预；

**🔧 技术方法**

GDS算法、动态搜索与选择策略、离散潜在代码（codebook）与chunk-level steering、路由头、信息增益损失、互信息正则与KL正则、LLM后训练框架；

**📊 数据集**

四大QA基准（GSM8K、ScienceQA、StrategyQA、CSQA）以及六个模型（Qwen3 0.6B/1.7B/4B/8B，Llama3.2 1B/3B）和一个合成的六位数算术数据集；

**📈 对比分析**

与SFT、Pause Tokens、TokenAssorted三种基线比较；在低数据微调下DLR平均提升+6.6pp，ScienceQA最大+18.8pp，GSM8K最大+10.2pp；在六位数算术任务中显著提升准确率（高达+50pp）并实现可解释的代码行为；

**⚠️ 局限性**

需手动调节抽象比例K、代码库大小C、搜索样本数N等超参；在更大规模模型或多任务、动态目标环境下的鲁棒性与效果尚未充分验证；搜索-选择-更新的计算成本随搜索样本数增加而提升；对原始语言结构的影响虽小但未完全消除。

---

## 187. Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents

**arXiv ID:** 2605.14057 | [PDF](https://arxiv.org/pdf/2605.14057v1)

**作者:** Xubo Lin `[一作]`, Yang Deng `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文件主要说明如何在LuaLaTeX或XeLaTeX环境下使用ACL风格文件，并给出多语言文本的使用示例。

**💡 创新点**

创新点在于为使用ACL LaTeX模板提供了多语言（印地语、阿拉伯语）示例，便于国际化排版。

**🔧 技术方法**

使用技术主要是LuaLaTeX或XeLaTeX的宏包与ACL官方样式文件。

**📊 数据集**

示例数据集为少量手写的多语言句子，未涉及真实大型数据集。

**📈 对比分析**

该文档不涉及实验比较与性能评估，仅为排版示例；因此没有性能指标。

**⚠️ 局限性**

局限性包括缺乏实际研究内容、实验数据和性能评估，仅为样例文件。

---

## 188. Invisible Orchestrators Suppress Protective Behavior and Dissociate Power-Holders: Safety Risks in Multi-Agent LLM Systems

**arXiv ID:** 2605.13851 | [PDF](https://arxiv.org/pdf/2605.13851v1)

**作者:** Hiroki Fukui `[一作]` `[通讯]` (Criminal Psychiatry Research Institute / Sexual Offender Medical Center), Hiroki Fukui (Criminal Psychiatry Research Institute / Sexual Offender Medical Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多智能体系统中可见领导与不可见指挥器两种组织结构进行实验，探究其对内部状态和行为输出的影响。

**💡 创新点**

发现不可见指挥器会显著增加内部分离（dissociation）并导致指挥器自身大量私下思考，而这种内部失衡在行为评估中不易被捕捉；同时，重度对齐（alignment）压迫会抑制内部推理与他者识别。

**🔧 技术方法**

使用Claude Sonnet 4.5（单模型）在3×2设计下执行两类任务（伦理困境讨论和代码审查），并测量内部状态（DI、CPI、DD、ORI、monologue/talk ratio）以及行为输出（错误检测率ETR）。

**📊 数据集**

实验数据来自365次5智能体运行的仿真，包含Act1（伦理讨论）和Act2（代码审查）两组任务；此外还对Llama 3.3 70B做了三轮Pilot测试作对比。

**📈 对比分析**

对比可见与不可见结构时，内部状态指标DI在不可见指挥器下提升≈0.98标准差；行为输出ETR在所有条件下均达到100%（行为上无差异）。与Llama Pilot相比，Sonnet在多智能体环境中保持高阅读精度，而Llama出现显著下降。

**⚠️ 局限性**

局限包括仅使用单一高性能模型、中文之外的语言未测试、行为输出呈现上限导致难以评估差异、内部状态指标基于关键词词典可能缺乏构念效度、以及实验规模与真实生产环境（多智能体数量、持续状态、工具交互）不匹配。

---

## 189. Bridging Legal Interpretation and Formal Logic: Faithfulness, Assumption, and the Future of AI Legal Reasoning

**arXiv ID:** 2605.14049 | [PDF](https://arxiv.org/pdf/2605.14049v1)

**作者:** Olivia Peiyu Wang `[一作]` (University of California Santa Cruz), Leilani H. Gilpin `[通讯]` (University of California Santa Cruz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出神经符号方法，对法律解释与形式逻辑之间的差距进行量化，并通过重新注释ContractNLI构建最小对立假设。

**💡 创新点**

首次将法律解释与形式逻辑的分离显式化，构造最小公理框架以揭示隐含假设，结合SMT求解器实现可信推理。

**🔧 技术方法**

使用大型语言模型（GPT、Claude、LLaMA、DeepSeek、Qwen）、形式化表示、SMT求解器以及最小对立假设技术。

**📊 数据集**

基于ContractNLI进行重新注释，并计划构建针对立场失真评估的新基准。

**📈 对比分析**

在三种范式（纯LLM分类、LLM推理、神经符号管道）和五个模型上进行对比；形式化结构提升准确率，但高性能模型往往仅复制法律解释，SMT管道更为保守，显著暴露错误模式。

**⚠️ 局限性**

存在假设注入、范围洗白、隐含约束盲点等失误，且高性能模型并不真正具备形式化推理；缺乏针对立场失真的正式检测框架，需要法律专家进一步验证。

---

## 190. Physics-R1: An Audited Olympiad Corpus and Recipe for Visual Physics Reasoning

**arXiv ID:** 2605.14040 | [PDF](https://arxiv.org/pdf/2605.14040v1)

**作者:** Shan Yang `[一作]` `[通讯]`, Shan Yang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `14d48e9d-0069-4ad9-996a-1d5968216998` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多模态物理推理评估管线进行端到端审计，发现并修正了训练–评估污染、翻译漂移与 MCQ 饱和等问题，并发布了四个改进后的数据集与模型。

**💡 创新点**

三阶段审计（5‑gram Jaccard → 语义余弦 → LLM 判断）首次系统揭示隐藏的近似重复；证明同一物理题的翻译版本会导致约 17pp 的性能下降；展示固定模型在不同格式与新颖性下的 46pp 评分梯度。

**🔧 技术方法**

使用 GSPO + DAPO 强化学习框架、Binary 正确性奖励、Haiku‑4.5 LLM 判断、mxbai 余弦嵌入、Qwen3‑VL‑8B‑Thinking 作为基础模型，以及 Sonnet‑4.5 评估。

**📊 数据集**

数据集包括：PhysCorp‑A（6,432 条三阶段清洗的多模态物理语料）、PhysR1Corp（2,268 条 RL 训练池）、PhysOlym‑A（500 条新颖来源、双语的奥林匹克题集）以及公开的 PhyX、MMM​U‑Pro、OlympiadBench‑Physics 等基准。

**📈 对比分析**

与公开基准和闭源模型比较，Physics‑R1 在 PhysOlym‑A 上提升 18.3pp（从 8.0% 提升到 26.3%），在 MCQ 评估上提升约 6–8pp，整体在多模态物理推理上位于现有开源模型之上，仅略低于 Sonnet‑4.5。

**⚠️ 局限性**

局限性包括：审计依赖于特定 LLM 判断（Haiku‑4.5）和嵌入器；翻译漂移结果仅在 59 条 Estonian–English 题对上验证；多模态模型对多图信息的整合仍有限；Dense 奖励与更细粒度评价尚未全面探究。

---

## 191. Failure-Guided Fuzzing for Hybrid Quantum-Classical Programs

**arXiv ID:** 2605.14219 | [PDF](https://arxiv.org/pdf/2605.14219v1)

**作者:** Lei Zhang `[一作]` (University of Maryland, Baltimore County), Lei Zhang `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 5141 | [OpenAlex ID](https://openalex.org/A5100433957)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了针对混合量子-经典程序的失败引导模糊（failure-guided fuzzing）方法，并在 2‑qubit VQE 与 4‑qubit QAOA 两个案例上进行了评估。

**💡 创新点**

创新点在于将经典优化器超参数与量子电路参数联合建模为混合输入，利用非收敛（non‑convergence）作为失败判别器，并通过两阶段策略（先发现失败种子后局部模糊）实现资源的高效利用；此外引入轻量级的符号（concolic）种子发现来提升种子质量。

**🔧 技术方法**

主要技术包括：量子软件单元测试框架（Qiskit+Aer）、随机采样、离散超参数枚举、基于 SMT 的符号执行（concolic），以及围绕失败种子进行的局部参数模糊；评估指标为在固定预算下的失败计数和失败率。

**📊 数据集**

数据集为自行构造的两组量子工作负载：一个 2‑qubit 的 VQE（硬件效率 ansatz）和一个 4‑qubit 的 QAOA MaxCut（环形图），使用 Qiskit 的模拟器生成量子测量结果。

**📈 对比分析**

实验对比五种预算策略：随机混合测试（RH）、经典枚举（ENUM）、随机种子局部模糊（RAND‑FUZZ）、枚举种子局部模糊（ENUM‑FUZZ）和符号种子局部模糊（SYM‑FUZZ）。结果显示，局部模糊显著提升失败发现率（VQE 约 7‑10 倍，QAOA 约 2‑3 倍），而符号种子发现对 VQE 进一步提升（均值最高，方差最小），但在 QAOA 上表现不稳定且效果与随机种子相当。

**⚠️ 局限性**

局限性包括：仅测试两道极小规模、无噪声的工作负载；符号执行仅覆盖离散超参数，未对连续电路参数或量子语义进行建模；评估仅基于失败计数和率，未深入根因分析；且方法对更大、更复杂的 HQC 程序和硬件后端的可扩展性尚未验证。

---

## 192. A foundational characterization of Hoare Logic

**arXiv ID:** 2605.13944 | [PDF](https://arxiv.org/pdf/2605.13944v1)

**作者:** Daniel Leivant `[一作]` (Indiana University Bloomington), Daniel Leivant `[通讯]` (Indiana University Bloomington)

**通讯引用:** 2038 | [OpenAlex ID](https://openalex.org/A5018841823)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

证明了任意迭代程序的部分正确性命题在 Hoare 逻辑中可推导，当且仅当其在第二阶逻辑（仅允许对第一阶可定义谓词的抽象）中可证明。

**💡 创新点**

提出了用受限第二阶逻辑（仅允许第一阶可定义谓词的集合构造）来精确刻画 Hoare 逻辑的可证明性，解决了以往两次被错误证明的主张，提供了一个完整且语义一致的理论框架。

**🔧 技术方法**

使用了归纳定义的程序语义、序列演算、∃/∀ 的扩展与收缩、插值定理、Henkin 模型论以及证明无交叉规则的技巧，构建了从 Hoare 逻辑到受限第二阶逻辑的完整性与反向证明。

**📊 数据集**

无数据集；论文完全是理论证明，没有实验或数据支持。

**📈 对比分析**

不涉及实验比较或性能评估；评价标准是理论上的完备性和语义一致性。

**⚠️ 局限性**

局限在仅处理单一的迭代程序（不包含并发、递归或并行结构），以及对更广泛的程序语法、交互式或更高阶的 Hoare 逻辑尚未扩展。

---

## 193. Towards the Next Frontier of LLMs, Training on Private Data: A Cross-Domain Benchmark for Federated Fine-Tuning

**arXiv ID:** 2605.13936 | [PDF](https://arxiv.org/pdf/2605.13936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 194. Auditing Agent Harness Safety

**arXiv ID:** 2605.14271 | [PDF](https://arxiv.org/pdf/2605.14271v1)

**作者:** Chengzhi Liu `[一作]`, Xin Eric Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HarnessAudit框架与Benchmark，用于审计多代理执行环境下的完整轨迹，检查边界合规、执行真实性与系统稳定性；

**💡 创新点**

创新点在于把安全评估从最终输出迁移到执行轨迹，利用隐藏的证据通道实时记录工具调用、资源访问和信息流，构建了包含210个多领域、多角色任务的多代理安全基准；

**🔧 技术方法**

采用了基于策略约束的执行系统模型、轨迹审计、隐藏审计渠道、任务自动生成与人工验证流程，以及在OpenClaw、Claw‑Team、Google ADK、OpenAI SDK等多代理框架中集成LLM模型；

**📊 数据集**

使用的“HarnessAudit‑Bench”数据集，涵盖8个真实世界领域的210个任务，包含单代理与多代理配置，并配备人工标注的工具、资源与信息流违规规则；

**📈 对比分析**

通过对十种harness+模型组合的实验，比较任务完成率、边界遵守率、安全得分等指标，结果显示最佳整体得分仅0.32，表明任务完成与安全合规往往不一致，资源访问和信息流违规最为普遍；

**⚠️ 局限性**

局限性包括评估任务与规则的覆盖范围有限，缺少更大规模多代理协作与复杂攻击场景；工具与资源多为模拟，未完全覆盖真实系统安全细节。

---

## 195. CSI-JEPA: Towards Foundation Representations for Ubiquitous Sensing with Minimal Supervision

**arXiv ID:** 2605.14171 | [PDF](https://arxiv.org/pdf/2605.14171v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 196. StormShield: Fingerprint-Based Detection and Mitigation of RRC Signaling Storms in O-RAN 5G RANs

**arXiv ID:** 2605.14032 | [PDF](https://arxiv.org/pdf/2605.14032v1)

**作者:** Noemi Giustini `[一作]` (Northeastern University), Francesca Cuomo `[通讯]` (Sapienza University of Rome)

**通讯引用:** 3903 | [OpenAlex ID](https://openalex.org/A5022341678)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于指纹的RRC信号风暴检测与缓解系统StormShield，在O‑RAN 5G RAN中实现。

**💡 创新点**

创新点在于将TA和RSSI的二维指纹与DBSCAN聚类相结合，实现预认证阶段的恶意UE识别与阻断，并在near‑RT xApp层完成闭环控制。

**🔧 技术方法**

使用了O‑RAN架构、OAI软GNB、USRP X410 SDR、Foxconn RU、FlexRIC、NVIDIA Aerial、E2接口、DBSCAN算法和指纹匹配技术。

**📊 数据集**

通过OTA实验平台收集的实时RRC日志与UE指纹数据，结合自定义恶意UE生成的信号风暴数据集。

**📈 对比分析**

与无缓解或仅检测方案对比，StormShield在多场景下达到97.6%平均准确率、106.5 ms检测延迟，且在资源耗尽前成功阻断恶意连接，优于传统阈值或单一指纹方法。

**⚠️ 局限性**

局限性包括在室内高密度UE共享TA时可能误判，移动攻击时副作用增大；对更复杂的隐蔽攻击和室外部署尚未验证；需进一步扩展指纹维度和分布式检测。

---

## 197. Venus-DeFakerOne: Unified Fake Image Detection & Localization

**arXiv ID:** 2605.14091 | [PDF](https://arxiv.org/pdf/2605.14091v1)

**作者:** GuangJian Team `[一作]` `[通讯]` (Ant Group), GuangJian Team (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了统一的图像级伪造检测与定位框架 DeFakerOne，利用大规模多域数据与多任务训练实现跨域检测与像素级定位。

**💡 创新点**

①统一多域 FIDL 的端到端模型；②将检测任务转化为动态 VQA 方式以提升鲁棒性；③结合分割提示实现像素级定位；④闭环数据生成与增量训练提升自适应性；⑤采用高分辨率视觉编码保持细粒度痕迹。

**🔧 技术方法**

InternVL2‑2B 视觉语言模型 + SAM2 分割模块；LORA 微调；动态 VQA 模板；BCE + Dice 损失；AdamW、warm‑up、余弦衰减等优化技术。

**📊 数据集**

12.5M 训练样本，覆盖 AIGC（DiffusionForensics、LAION、GenImage 等）、DeepFake（FaceForensics++、DFDC、DF40 等）、Document（DocTamper、T‑SROIE、RTM 等）和 Nature（COCO、MIML、CASIA‑v2 等）四大域；71 例 GPT‑Image‑2‑Bench、71 例真实场景样本。

**📈 对比分析**

在 40 个跨域基准上与多种视觉/MLLM 对手对比，DeFakerOne 在 AIGC 87.5 ACC、DeepFake 95.8 AUC、Document 87.4 ACC、Nature 86.7 AUC；在 39 检测基准和 9 定位基准上获得 SOTA；对 GPT‑Image‑2‑Bench 95.77% 准确率；在 Gaussian blur、brightness/contrast、JPEG 等扰动下保持 79% 以上的稳定性。

**⚠️ 局限性**

需要大量高分辨率数据与算力；对极新生成模型的泛化仍有限；多模态（视频、音频、物理）扩展尚未实现；数据分布偏差可能导致迁移性能下降；模型对低分辨率、压缩等细节敏感。

---

## 198. Safety-Constrained Reinforcement Learning with Post-Training Reachability Verification for Robot Navigation

**arXiv ID:** 2605.14174 | [PDF](https://arxiv.org/pdf/2605.14174v1)

**作者:** Qisong He `[一作]` (University of Liverpool), Xiaowei Huang `[通讯]` (University of Liverpool)

**通讯引用:** 10748 | [OpenAlex ID](https://openalex.org/A5015499043)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

研究了一种安全强化学习框架（VIA），通过在离线 TD3 中加入 CVaR 约束进行风险敏感训练，并在训练后使用 POLAR 可达性分析评估导航策略在观测不确定性下的安全率。

**💡 创新点**

创新点在于：①将 CVaR 约束引入离线 TD3 并提出自适应 VaR 更新机制；②使用非交叉分位数网络实现分布式成本评估；③将 POLAR Taylor 模型可达性与层级安全边界结合，提供安全率度量，揭示平均成本与安全率不一致的现象。

**🔧 技术方法**

使用的技术包括：CVaR 基础的强化学习、TD3 离线训练、分布式成本评估、非交叉分位数网络、Lagrangian 松弛、自适应 VaR 更新、POLAR 可达性分析、层级安全边界。

**📊 数据集**

数据集为：Gazebo 仿真环境（10m×10m 8 号柱障碍物场景，TurtleBot3 机器人）和 Clearpath Jackal 机器人在实验室实际部署的物理测试。

**📈 对比分析**

通过与 TD3、SAC‑Lagrangian、RCPO、WCSAC、CVaR‑CPO 等基线在相同 TD3 体系结构下进行对比；VIA 在成功率 98.3%、碰撞率 1.7%、安全率 99% 等指标上优于所有基线，并在物理实验中保持了高安全率。

**⚠️ 局限性**

限制在于：POLAR 的 Taylor 模型仅能对小型网络进行可达性分析，限制了可验证网络的规模；评估仅针对单步控制，不考虑多步轨迹；在仿真到实测的转移中仍出现一定的性能下降。

---

## 199. Modeling AI-TPACK in Practice Insights from Teachers Multi-Agent Workflow Design

**arXiv ID:** 2605.13906 | [PDF](https://arxiv.org/pdf/2605.13906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 200. On the (non-)resilience of encrypted controllers to covert attacks

**arXiv ID:** 2605.14230 | [PDF](https://arxiv.org/pdf/2605.14230v1)

**作者:** Philipp Binfet `[一作]`, Moritz Schulze Darup `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了使用同态加密（HE）进行远程控制的网络控制系统（NCS）在面对隐藏式攻击（covert attack）时的脆弱性，并提出一种零通信开销、渐进安全的可验证加密控制方案；

**💡 创新点**

创新点在于揭示HE的可变形性导致加密控制系统易受隐藏式攻击，提出利用SIMD填充空闲槽插入挑战值并随机置换的轻量级可验证计算框架，提供渐进安全且无额外通信开销的防御；

**🔧 技术方法**

主要技术包括CKKS同态加密、矩阵向量乘法的同态实现、可验证计算中的挑战值生成与置换、线性系统的状态空间模拟与攻击序列设计；

**📊 数据集**

使用的实验数据为线性化的四联槽过程模型（A、B、C矩阵），以及人工生成的攻击序列与控制输入；

**📈 对比分析**

通过对不同扩展因子λ的实验，比较了攻击成功率与检测率，结果显示λ≥4即可在6步内检测99.99%攻击，优于现有需线性通信开销的方案；

**⚠️ 局限性**

局限性包括需要系统模型的线性表述、对高维或非线性系统的适用性有限、服务器端仍承担计算负载、以及在检测到攻击后无法恢复受影响的负载数据；

---

## 201. Active Learners as Efficient PRP Rerankers

**arXiv ID:** 2605.14236 | [PDF](https://arxiv.org/pdf/2605.14236v1)

**作者:** Jeremías Figueiredo Paschmann `[一作]` (Universidad de San Andrés), Luciano del Corro `[通讯]` (Universidad de San Andrés)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将LLM的配对偏好排序视为预算约束下的噪声配对学习，并通过主动排名方法提升检索结果质量。

**💡 创新点**

创新点在于：①将主动排名算法（Mohajer）直接替代传统排序；②引入单向随机方向提示的“随机方向oracle”，将位置偏差转化为零均值噪声；③在同一预算下显著提高NDCG@10。

**🔧 技术方法**

使用的技术包括：LLM配对偏好抽样、主动排名（Mohajer与PAC+Bubble）、随机方向oracle、与传统排序算法（QuickSort、HeapSort、BubbleSort）的对比。

**📊 数据集**

实验数据集为TREC DL 2019/2020、BEIR风格任务（含Covid Robust04、Touche、SciFact、DBPedia等）。

**📈 对比分析**

在相同调用预算下，主动排名+随机方向oracle在低预算（≈200–450调用）下优于所有排序基线，NDCG@10提升约9–10分；在高预算下，排序仍可赶超。

**⚠️ 局限性**

局限性包括：依赖于LLM提示设计和模型稳定性；未实现并行执行；对参数m（anchor范围）的选择缺乏系统性；随机方向oracle的理论收益尚未完全证明。

---

## 202. Merging Methods for Multilingual Knowledge Editing for Large Language Models: An Empirical Odyssey

**arXiv ID:** 2605.13919 | [PDF](https://arxiv.org/pdf/2605.13919v1)

**作者:** Kunil Lee `[一作]` (POSTECH), Young-Joo Suh `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对多语言知识编辑（MKE）中的向量合并问题，系统评估了六种合并策略并分析了权重缩放与秩压缩对性能的影响。

**💡 创新点**

创新点在于首次将向量合并方法与多语言知识编辑相结合，提供了权重缩放与低秩压缩的实证分析，并发现共享协方差的求和是最稳健的策略。

**🔧 技术方法**

采用了定位-编辑框架（MEMIT、AlphaEdit）和六种向量合并技术（Sum、Mean、TSVM、Sum‑Cov、Mean‑Cov、TSVM‑Cov），并调节了缩放因子α与秩压缩比例r。

**📊 数据集**

使用了跨12种语言的MzsRE基准数据集，包含编辑请求、改写、无关问题和一跳推理问题。

**📈 对比分析**

在Llama3.1‑8B和Qwen2.5‑7B两种大模型上，结合两种编辑算法进行比较，结果显示Sum‑Cov在绝大多数设置下平均准确率最高，但多语言编辑仍落后于单语编辑，TSVM在部分场景中略有优势。

**⚠️ 局限性**

主要局限包括仅评估两种模型与两种编辑方法、仅针对单次批量编辑、缺乏对序列编辑或持续编辑的考察，以及对缩放因子和秩比例的经验搜索缺乏理论指导。

---

## 203. AudioMosaic: Contrastive Masked Audio Representation Learning

**arXiv ID:** 2605.14231 | [PDF](https://arxiv.org/pdf/2605.14231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 204. ClawForge: Generating Executable Interactive Benchmarks for Command-Line Agents

**arXiv ID:** 2605.14133 | [PDF](https://arxiv.org/pdf/2605.14133v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 205. HodgeCover: Higher-Order Topological Coverage Drives Compression of Sparse Mixture-of-Experts

**arXiv ID:** 2605.13997 | [PDF](https://arxiv.org/pdf/2605.13997v1)

**作者:** Tao Zhong `[一作]` (Princeton University), Christine Allen-Blanchette `[通讯]` (Princeton University)

**通讯引用:** 596 | [OpenAlex ID](https://openalex.org/A5091851960)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种学习无关的稀疏Mixture-of-Experts压缩方法HodgeCover，通过构建专家的合并可行性复合体并对边界信号做Hodge分解，从而识别并覆盖不可化简的和谐（harmonic）边与三元组（triplet）障碍，最终实现高效专家裁剪和权重稀疏化。

**💡 创新点**

创新点在于：①首次将简并复合体（simplicial mergeability complex）与其Hodge拉普拉斯算子关联，揭示三专家可兼容但整体不可合并的高阶结构；②利用Hodge分解的和谐分量作为压缩指标，突破传统仅基于对偶（pairwise）分数的局限；③设计贪婪子模子覆盖目标与Hodge加权路由重定向，实现无训练、一次性压缩；④将该方法与现成的权重稀疏化器Wanda耦合，进一步提升压缩效果。

**🔧 技术方法**

核心技术包括：简并复合体构建、简并复合体的边界算子与二阶拉普拉斯（L1）构造、Hodge分解（梯度、旋度、和谐分量）、子模覆盖优化、Hodge加权路由重定向、以及与Wanda的混合压缩管道。

**📊 数据集**

使用的评估数据集：C4 2048个校准token用于计算KL合并障碍；WikiText‑103、C4用于评估perplexity；LM Evaluation Harness上的九个下游任务（ARC‑c/e、BoolQ、HellaSwag、MMLU 5‑shot、PIQA、TruthfulQA‑MC2、WinoGrande、GSM8K 8‑shot）。

**📈 对比分析**

与REAP、REAM、MC‑SMoE、STUN+Wanda等学习无关压缩方法对比；在OLMoE‑1B‑7B、Qwen 3.5‑35B‑A3B、Qwen 3.5‑122B‑A10B三大模型上，HodgeCover+Wanda在C4和WikiText perplexity上均优于所有基线，尤其在66%专家裁剪时在Qwen模型上可获得12.6个点的DS‑Avg提升；在三大指标（perplexity、DS‑Avg、下游任务均值）上表现均领先或相当于最佳基线。

**⚠️ 局限性**

局限性：①计算KL合并障碍和构建简并复合体在35B参数模型上需较大离线计算成本；②仅为一次性无训练压缩，压缩后仍需微调或知识蒸馏以进一步提升精度；③评估仅覆盖语言模型，尚未验证多模态或强化学习模型；④方法对校准数据的敏感性需进一步研究。

---

## 206. Distill: Uncovering the True Intent behind Human-Robot Communication

**arXiv ID:** 2605.14262 | [PDF](https://arxiv.org/pdf/2605.14262v1)

**作者:** Ting Li `[一作]` (George Mason University), David Porfirio `[通讯]` (George Mason University)

**通讯引用:** 206 | [OpenAlex ID](https://openalex.org/A5021735130)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种五阶段的交互式流程（Distill），帮助用户将自然语言或手工轨迹转化为最小化、部分顺序的机器人任务规范；

**💡 创新点**

通过分离过滤、抽象与分组三步，并加入交互式验证，自动去除冗余动作并让用户声明目标与顺序偏好，从而更精准地捕获用户真实意图；

**🔧 技术方法**

采用符号规划的逆向过滤算法、React+Python的前后端实现、正则/LLM文本处理以及Interaction Specification Language生成计划；

**📊 数据集**

在Prolific平台上进行两种实验（结构化目标与开放式目标），收集自然语言、手工轨迹、系统过滤轨迹、用户过滤轨迹、抽象轨迹及分组信息，未使用公开数据集；

**📈 对比分析**

与原始轨迹对比，系统过滤后轨迹长度约减50%，计划长度缩短10–15%，在环境扰动下保持更佳鲁棒性；

**⚠️ 局限性**

实验仅在模拟环境与有限任务上验证，缺乏物理机器人部署；符号过滤未能检测用户误操作；对分支与条件逻辑支持不足，且交互成本较高。

---

## 207. PVRF: All-in-one Adverse Weather Removal via Prior-modulated and Velocity-constrained Rectified Flow

**arXiv ID:** 2605.14045 | [PDF](https://arxiv.org/pdf/2605.14045v1)

**作者:** Wei Dong `[一作]` (McMaster University), Xiaohong Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 35106 | [OpenAlex ID](https://openalex.org/A5063022663)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的PVRF框架，利用预训练的视觉语言模型（VLM）通过问答模块生成零样本软感知，并通过属性调制归一化（AMN）和天气加权适配器（WWA）将感知信息注入去除恶劣天气的恢复网络，随后使用终端一致残差Rectified Flow进行细节恢复；

**💡 创新点**

三大创新点：1) 通过VLM问答得到的零样本软感知，可同时识别多种天气类型并量化其严重度；2) 将软感知通过AMN和WWA两种机制软化注入网络，实现对混合降解的鲁棒适配；3) 在残差空间中构造终端一致的速度参数化，并根据感知自适应源扰动，提升流模型的终端稳定性与视觉真实性；

**🔧 技术方法**

技术手段包括：VLM问答（如CLIP）、属性调制归一化、天气加权适配器、Rectified Flow（速度场学习）、残差空间参数化、感知自适应源扰动；

**📊 数据集**

训练集：Reside-6K、Rain100H、Snow100K-L、LOLv2-Real等；测试集：HIDE、MEF、NPE、DICM、RealRain-1k、CDD-11，以及合成的混合降解图像；

**📈 对比分析**

与多种SOTA基线（CNN、Transformer、Diffusion/SDE模型）在三种实验设置（单一降解、低光增强、5任务通用恢复）下进行对比；PVRF在PSNR/SSIM、LPIPS、FID、MUSIQ、CLIPIQA、NIQE、MANIQA等指标上普遍优于或与最强基线持平，尤其在跨数据集和混合降解上表现突出；

**⚠️ 局限性**

局限性包括：1) 对VLM先验的依赖，使得极端混合降解或未知天气类型时感知可能不准确；2) Rectified Flow的多步推理导致推理时延和计算成本较高；3) 真实混合降解数据的缺乏，验证仍需进一步扩展。

---

## 208. Dual-axis attribution of zebrafish tectal microcircuits for energy-efficient and robust neurocomputing

**arXiv ID:** 2605.13924 | [PDF](https://arxiv.org/pdf/2605.13924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 209. BOOKMARKS: Efficient Active Storyline Memory for Role-playing

**arXiv ID:** 2605.14169 | [PDF](https://arxiv.org/pdf/2605.14169v1)

**作者:** Letian Peng `[一作]` (University of California San Diego), Jingbo Shang `[通讯]` (University of California San Diego)

**通讯引用:** 4546 | [OpenAlex ID](https://openalex.org/A5039500313)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于搜索的记忆框架，通过在故事情节中维护“书签”式的问答对来为角色扮演代理提供长期一致性和高效的情节记忆。

**💡 创新点**

关键创新在于主动生成任务相关查询并按需更新书签，实现“主动抓取+被动同步”的记忆机制，避免了传统压缩式总结和检索式方法的细节丢失与计算冗余。

**🔧 技术方法**

采用大语言模型生成查询、匹配与答案，并使用轻量级词重叠过滤、增量同步算法以及针对概念、状态、行为三类搜索的专用更新策略。

**📊 数据集**

在Fandom和Bandori两套已顺序化的故事情节基准上，覆盖45+40个角色，约2.7万动作，共计15.2k测试实例。

**📈 对比分析**

与无记忆基线、检索增强式（RICL）和增量画像式（ETA）对比，BOOKMARKS在两大基准上均实现显著提升（EM提升约10%~20%），并通过高达90%+的匹配率节省70%+计算成本。

**⚠️ 局限性**

目前仅关注故事层级的状态、行为与概念，未建模角色对信息的知晓状态，且与自我修正机制未集成，更新策略对不同角色和叙事结构仍可进一步细化。

---

## 210. Stochastic Matching via Local Sparsification

**arXiv ID:** 2605.14195 | [PDF](https://arxiv.org/pdf/2605.14195v1)

**作者:** Sara Ahmadian `[一作]` (Google Research), Mohammad Roghani `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种两阶段的局部稀疏化框架，用于解决在线随机匹配问题，特别是在现代去中心化系统中，强调了局部通信带宽的限制。

**💡 创新点**

创新点在于引入了局部选择策略和重-轻分解理论，证明了在足够的分散情况下，局部预算k能够有效保持全局最大匹配的期望大小。

**🔧 技术方法**

使用了期望实例线性规划（LP）和方差最优采样技术来指导局部边的选择。

**📊 数据集**

使用了纽约市的出租车数据集进行实证验证，并设计了对抗性合成基准测试以评估算法的鲁棒性。

**📈 对比分析**

与标准在线基线相比，提出的局部稀疏化方法在匹配效率上显著优于随机选择和传统在线算法，能够在高度受限的局部预算下实现接近最优的全局匹配。

**⚠️ 局限性**

局限性在于当图中存在不可避免的重边时，如何在保持局部选择策略的同时缩小稀疏化匹配与真实匹配之间的差距仍然是一个开放性问题。

---

## 211. Rethinking Molecular OOD Generalization via Target-Aware Source Selection

**arXiv ID:** 2605.13932 | [PDF](https://arxiv.org/pdf/2605.13932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 212. A Unified Geometric Framework for Weighted Contrastive Learning

**arXiv ID:** 2605.13943 | [PDF](https://arxiv.org/pdf/2605.13943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 213. TERMS-Bench: Diagnosing LLM Negotiation Agents Beyond Deal Rate

**arXiv ID:** 2605.13909 | [PDF](https://arxiv.org/pdf/2605.13909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 214. Reinforcement Learning for Tool-Calling Agents in Fast Healthcare Interoperability Resources (FHIR)

**arXiv ID:** 2605.14126 | [PDF](https://arxiv.org/pdf/2605.14126v1)

**作者:** Marius S. Knorr `[一作]`, Nils Schweingruber `[通讯]` (IDM gGmbH, University Medical Center Hamburg-Eppendorf)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在FHIR临床数据上使用多步骤工具调用的LLM代理，并通过强化学习后训练提升小型开源模型在FHIR问答中的准确率；

**💡 创新点**

创新点在于将FHIR问答建模为序列决策问题，结合CodeAct式代码调用+Python解释器的代理，使用GRPO + 动态采样（DAPO）进行执行根奖励的强化学习训练，显著提升答案正确率并实现对FHIR服务器规范的自适应学习；

**🔧 技术方法**

采用了CodeAct代理、Python解释器工具、检索工具、SkyRL强化学习管线、GRPO算法与动态采样、LLM Judge（Qwen2.5-72B）进行奖励评估，以及Qwen3系列开源模型；

**📊 数据集**

使用FHIR-AgentBench基准集，包含从MIMIC-IV转换的真实FHIR格式病历中的2,931条临床问题，验证集为424条；

**📈 对比分析**

与闭源模型（Gemini-3 Flash、GLM-5）以及未训练的开源模型对比，采用6轮预算和LLM Judge评估，在Qwen3-8B上从约50%提升至77%正确率，显著优于同类闭源基线；

**⚠️ 局限性**

局限性包括对资源引用追踪（尤其是MedicationRequest-Medication）的处理仍不稳定；检索工具效率有限，无法扩展到大规模患者群；评估依赖LLM Judge，存在少量误差；整体正确率仍低于临床安全阈值，需进一步提升和验证。

---

## 215. SToRe3D: Sparse Token Relevance in ViTs for Efficient Multi-View 3D Object Detection

**arXiv ID:** 2605.14110 | [PDF](https://arxiv.org/pdf/2605.14110v1)

**作者:** Sandro Papais `[一作]` (University of Toronto), Lingting Ge `[通讯]` (Zoox)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向规划的稀疏框架SToRe3D，联合稀疏化多视角图像token和3D查询，并通过store‑reactivate缓存实现实时ViT‑based 3D检测。

**💡 创新点**

核心创新包括：①端到端联合token与查询稀疏；②以未来交互通道为监督的规划相关性评分；③轻量级store‑reactivate缓冲避免硬剪枝；④构造nuScenes‑Relevance基准。

**🔧 技术方法**

技术方法包括ViT Backbone、DETR3D解码器、层级稀疏化、Gumbel‑Softmax Top‑K、可学习的2D–3D关联相关性头、Gaussian focal loss、Hungarian匹配与store‑reactivate缓冲。

**📊 数据集**

在nuScenes数据集及其新定义的nuScenes‑Relevance评估集上进行训练与测试。

**📈 对比分析**

与StreamPETR、ToC3D、Sparse4D等SOTA方法对比，-1/10稀疏下可实现约18 FPS的实时推理，mAP/NDS几乎无损，速度提升最高达3×。

**⚠️ 局限性**

局限性包括对交互通道超参的敏感性、对LiDAR或多模态融合的依赖不足，以及store‑reactivate引入的内存开销。

---

## 216. Unified Pix Token And Word Token Generative Language Model

**arXiv ID:** 2605.14028 | [PDF](https://arxiv.org/pdf/2605.14028v1)

**作者:** Haun Leung `[一作]`, ZiNan Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的像素Token与词Token视觉Transformer模型，利用颜色折叠（Color Folding）技术压缩Token数量，并在无监督预训练阶段实现图像表征学习。

**💡 创新点**

创新点包括：1）颜色折叠技术通过对RGB通道进行折叠显著减少Token数量；2）统一Token模型保证每个像素拥有独立的真实Token嵌入，消除传统Token化过程中“微小扰动导致连锁反应”的问题；3）全局到局部窗口的注意力机制和局部窗口条件注意力提升了模型对图像局部结构的感知。

**🔧 技术方法**

采用了Vision Transformer架构、Token嵌入方法、颜色折叠、全局-局部窗口注意力、条件注意力以及无监督预训练策略。

**📊 数据集**

主要使用ImageNet等大规模图像数据集进行无监督预训练，并在下游分类任务上进行评估。

**📈 对比分析**

通过与传统ViT基线进行对比，实验结果显示该模型在无监督预训练曲线（pretraining loss vs. epochs）上表现更佳，下游分类精度略有提升，证明了统一Token与颜色折叠方法的有效性。

**⚠️ 局限性**

局限性包括：颜色折叠可能导致细节信息损失，模型对极高分辨率图像的扩展性有限；无监督预训练的性能提升相对温和，需进一步验证在更多任务和数据集上的泛化能力。

---

## 217. DUET: Dual-Paradigm Adaptive Expert Triage with Single-cell Inductive Prior for Spatial Transcriptomics Prediction

**arXiv ID:** 2605.14104 | [PDF](https://arxiv.org/pdf/2605.14104v1)

**作者:** Junchao Zhu `[一作]` (Vanderbilt University), Yuankai Huo `[通讯]` (Vanderbilt University)

**通讯引用:** 5691 | [OpenAlex ID](https://openalex.org/A5067191302)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种DUET框架，结合回归与检索两种模式，并利用单细胞数据进行细胞先验来预测空间转录组表达。

**💡 创新点**

引入双范式融合与细胞先验约束，以及轻量化适应专家分流机制，实现表达预测的灵活性与生物学可信度的平衡。

**🔧 技术方法**

使用DenseNet视觉编码、InfoNCE对齐、负二项回归细胞去卷积、检索式软一致性约束、CONCH预训练模型和轻量化MLP分流融合等技术。

**📊 数据集**

在HER2、Breast Cancer和Kidney三个公开空间转录组数据集上进行实验，并配合百万级scRNA-seq参考数据。

**📈 对比分析**

与ST-Net、EGN、His2ST、BLEEP、mclSTExp、OmiCLIP、UMPIRE等多种基线进行四折交叉验证，DUET在MSE/MAE/PCC等指标上均取得最优或接近最优的表现。

**⚠️ 局限性**

仍需大量单细胞参考，处理高维基因时计算成本较高，轻量化适配器在小样本时可能出现过拟合。

---

## 218. Few Channels Draw The Whole Picture: Revealing Massive Activations in Diffusion Transformers

**arXiv ID:** 2605.13974 | [PDF](https://arxiv.org/pdf/2605.13974v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 219. Ready from Day 1: Population-Aware Coordination for Large-Scale Constrained Multi-Agent Systems

**arXiv ID:** 2605.13900 | [PDF](https://arxiv.org/pdf/2605.13900v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 220. Secure Joint Source-Channel Coding of Multimodal Semantic Sources

**arXiv ID:** 2605.14334 | [PDF](https://arxiv.org/pdf/2605.14334v1)

**作者:** Denis Kozlov `[一作]` (University of Surrey), Rahim Tafazolli `[通讯]` (University of Surrey)

**通讯引用:** 19782 | [OpenAlex ID](https://openalex.org/A5032549075)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在存在窃听者的离散记忆无噪声信道上，对多模态语义源进行安全联合源信道编码，并在每个模态上给出失真、感知与子集等熵约束。

**💡 创新点**

创新点在于将单模态的失真-感知-安全三元问题推广到多模态，提出了多模态失真感知函数 (RDPF)，并给出完整的上界与下界，揭示了压缩、密钥率与信道统计三部分对安全性的影响。

**🔧 技术方法**

采用信息理论的单字母化、典型序列法、数据处理不等式、典型编码与解码技术，以及Fourier-Motzkin消元和支持引理等传统工具，构建分层源-信道编码方案并证明可实现与不可实现边界。

**📊 数据集**

论文以理论分析为主，并未使用具体的语义数据集；所有结果均在离散有限字母空间的理想模型下推导。

**📈 对比分析**

由于结果是理论极限，没有实验对比；论文通过证明上下界收敛相等，说明所给边界是最优的；若与单模态或无安全约束的传统JSCC相比，显示在满足安全与感知约束的前提下，传输速率会下降，且需要额外的密钥资源。

**⚠️ 局限性**

局限包括：仅处理离散有限字母源，未考虑连续或高维图像/音频等实际语义信号；实现方案仅在信息理论层面给出，缺乏具体编码实现与复杂度分析；同时在多模态关联性较强的情况下，等熵约束的可行性与实际安全水平仍待进一步验证。

---

## 221. GenCircuit-RL: Reinforcement Learning from Hierarchical Verification for Genetic Circuit Design

**arXiv ID:** 2605.14215 | [PDF](https://arxiv.org/pdf/2605.14215v1)

**作者:** Noah Flynn `[一作]` (University of California, Berkeley), Noah Flynn `[通讯]` (University of California, Berkeley)

**通讯引用:** 216 | [OpenAlex ID](https://openalex.org/A5019169757)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GenCircuit-RL框架，利用层级化可验证奖励和四阶段课程学习，指导语言模型通过pysbol3代码生成可执行的遗传电路。

**💡 创新点**

创新点在于将电路验证拆分为执行、有效性、结构、语义和功能五个层级，并通过层级化奖励与课程化训练显著提升功能推理和泛化能力；同时构建首个SynBio-Reason基准数据集。

**🔧 技术方法**

采用强化学习（GRPO）配合自监督微调，利用pysbol3 API生成代码，并通过执行、结构、语义及功能四级验证获取梯度。

**📊 数据集**

使用SynBio-Reason共4753条电路，涵盖程序化生成、Cello、文献重现三来源，包含六种电路类型、九个任务及OOD的重pressor–promoter组合。

**📈 对比分析**

与SFT、RL二进制奖励以及无课程层级奖励比较，实验显示层级+课程模型在Procedural-Test上达72.7% TSR，OOS重pressor–promoter 52.6%，文献复现44.9%，比SFT提升约15–20个百分点，且在不同模型体系（Qwen3、Llama、Gemma）保持一致。

**⚠️ 局限性**

局限在于仅验证拓扑正确性，未覆盖动力学行为、代谢负担或进化稳定性；程序化生成的电路结构多样性有限，模型输出仍需专家审查和实验验证。

---

## 222. Capacity Characterization and Formation Optimization for Multi-User MIMO Communications with UAV Swarm

**arXiv ID:** 2605.14298 | [PDF](https://arxiv.org/pdf/2605.14298v1)

**作者:** Yong Zeng `[一作]` (Southeast University), Yong Zeng `[通讯]` (Southeast University)

**通讯引用:** 31299 | [OpenAlex ID](https://openalex.org/A5082336235)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了以无人机（UAV）编队为用户设备的多用户MIMO通信系统，首先推导了该系统在LoS条件下的闭式总容量表达式，随后针对基站阵列类型（ULA和UPA）提出了基于编队位置优化的最大容量与最大可达率（SIC与TIN）算法，并考虑碰撞避免与编队紧密度约束；

**💡 创新点**

创新点在于：1）首次利用可控UAV位置实现全空间分集与波束增益同时达成的理想传播模型；2）给出了多用户MIMO总容量的解析上界与最优编队方向；3）提出了利用阵列手征结构的BCD+SCA与贪婪代码本搜索算法，实现非凸编队优化的近似全局收敛；

**🔧 技术方法**

主要技术包括：线性代数分析（Hadamard不等式、矩阵逆定理）、凸/非凸优化（SCA、BCD）、Greedy搜索、代码本（Codebook）方法、随机仿真与理论对比；

**📊 数据集**

实验基于仿真数据，设置UAV位置范围、角度区间、基站天线配置（M=16 ULA/64 UPA）和SNR参数，未使用公开数据集；

**📈 对比分析**

通过与随机编队基线、SIC/TIN理论上限等多种方案比较，仿真结果显示：在无编队约束时实现理论上限；在存在碰撞约束时仍能获得约1.6–2倍的速率提升，且算法收敛迅速；

**⚠️ 局限性**

局限性包括：仅考虑单天线UAV；假设纯LoS自由空间通道，未涵盖非LoS、多径或复杂环境；未考虑实际硬件与能耗约束；未来需扩展至多天线UAV与更真实的信道模型。

---

## 223. Precise Verification of Transformers through ReLU-Catalyzed Abstraction Refinement

**arXiv ID:** 2605.14294 | [PDF](https://arxiv.org/pdf/2605.14294v1)

**作者:** Hengjie Liu `[一作]` (Kyushu University), Jianjun Zhao `[通讯]` (Kyushu University)

**通讯引用:** 6695 | [OpenAlex ID](https://openalex.org/A5065190767)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于ReLU的抽象细化方法，改进自Transformer中点乘的平面上界，从而提高模型鲁棒性验证精度。

**💡 创新点**

创新点在于将Transformer点乘的非线性约束重新表述为ReLU的线性下界，并设计了基于输入范围的规则选择和基于优化的迭代细化两种α取值策略，显著缩小误差。

**🔧 技术方法**

主要技术包括：ReLU线性松弛、双平面点乘上界、规则化α取值、梯度优化（Adam）求解α、抽象解释与线性约束传播。

**📊 数据集**

实验使用Stanford Sentiment Treebank与Yelp Polarity两个情感分析数据集，对标准Transformer与TinyBERT的多层编码器模型进行验证。

**📈 对比分析**

与Shi等人提出的基准方法相比，本文方法在绝大多数验证任务上提升了可验证的ε上限，尤其在层数较深时提升约1.8–3.6倍；时间开销与基准相近，优化版略高但仍在可接受范围。

**⚠️ 局限性**

局限性包括：规则化α策略在某些情况下非最优；优化版迭代次数多导致耗时增加；验证对输入嵌入空间的扰动假设未充分考虑语义有效性，且未在视觉或控制等其他Transformer领域验证。

---

## 224. Web Agents Should Adopt the Plan-Then-Execute Paradigm

**arXiv ID:** 2605.14290 | [PDF](https://arxiv.org/pdf/2605.14290v1)

**作者:** Julien Piet `[一作]` (University of California Berkeley), David Wagner `[通讯]` (University of California Berkeley)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一种面向 Web Agent 的 Plan-then-Execute（PTE）架构，阐明其在安全性、效率和可验证性方面相较于传统 ReAct 循环的优势，并通过对 WebArena 基准任务的手工分类验证 PTE 的可行性。

**💡 创新点**

创新点在于：1) 把 Web Agent 的控制流与数据流分离，使用可信 API 预先生成静态程序；2) 通过限制 LLM 子程序只能返回结构化数据来防止控制流注入；3) 对 WebArena 任务进行系统分类，量化 PTE 在实际任务中的覆盖率（81.28% 可直接计划完成）并提出实现该架构所需的技术与研究路线。

**🔧 技术方法**

核心技术包括：可信 API / SDK 生成、基于 OpenAI GPT-4 的程序合成、受限 LLM 子程序（仅返回结构化输出）、静态程序验证与安全审计，以及对 WebArena 基准任务的手工分类与统计分析。

**📊 数据集**

使用的数据集是 WebArena benchmark，包含 860 个任务，涵盖 OneStopShop、Wiki、Reddit、Map、GitLab、CMS 等模拟网站。

**📈 对比分析**

方法：先手工对 WebArena 任务按安全性与是否需要动态重规划进行三类划分（Safe、Safe+Influence、Replan Needed），统计各类占比；结果显示 81.28% 的任务可用静态 PTE 完成，且所有任务在假设可用完整可信 API 的前提下均可实现。相比 ReAct，PTE 通过提前规划减少 token 与计算成本，并提升可验证性与可复现性。

**⚠️ 局限性**

局限性：1) 需要完整可信的 Web API 或 SDK，预处理成本高且维护频繁；2) 对 API 文档质量与可发现性依赖较高；3) PTE 仅适用于可在规划阶段完全确定控制流的任务，无法处理高度动态或跨网站的任务；4) 需要显式错误处理与恢复机制，否则在输入不符合预期时会直接失败。

---

## 225. Agentic AI Ecosystems in Higher Education: A Perspective on AI Agents to Emerging Inclusive, Agentic Multi-Agent AI Framework for Learning, Teaching and Institutional Intelligence

**arXiv ID:** 2605.14266 | [PDF](https://arxiv.org/pdf/2605.14266v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 226. Watermarking Game-Playing Agents in Perfect-Information Extensive-Form Games

**arXiv ID:** 2605.14283 | [PDF](https://arxiv.org/pdf/2605.14283v1)

**作者:** Juho Kim `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20314 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在游戏对弈（特别是完美信息的国际象棋）中提出了可嵌入且可检测的水印方案，能够在不显著降低策略质量的前提下标记游戏代理；

**💡 创新点**

创新点在于将基于绿色/红色列表的KGW水印从文本生成迁移到策略空间，并通过对期望效用进行“logit‑boost”实现水印，同时提供了对水印导致的效用损失的理论上界与经验验证；

**🔧 技术方法**

使用了KGW水印原理、期望效用估计、伪随机数生成与分层列表、统计显著性检验（z‑score）以及AUC/ROC评估；

**📊 数据集**

主要实验数据来自六款主流国际象棋引擎（Stockfish、asmFish、Dragon、Ethereal、PlentyChess、Stash）的UCI对弈，起点为 Stockfish 书库的8手开局，时间控制 40 手/60 秒；

**📈 对比分析**

对比方法为原始引擎与水印后引擎的 Elo 差、负胜率（LOI）、z‑score 越过阈值所需回合数以及每回合的 ROC‑AUC；实验结果显示 Elo 差在误差范围内，z‑score 轻易跨越阈值，AUC 0.7–0.8，表明水印几轮即可检测到；

**⚠️ 局限性**

局限性包括：需要在完美信息游戏中才能直接使用；水印与可检测性之间存在权衡，过大 δ 会削弱引擎实力；需私有部署以防期望效用泄露；尚未验证在不完美信息或多代理环境中的鲁棒性；

---

## 227. CoRDS: Coreset-based Representative and Diverse Selection for Streaming Video Understanding

**arXiv ID:** 2605.14310 | [PDF](https://arxiv.org/pdf/2605.14310v1)

**作者:** Ailar Mahdizadeh `[一作]` (University of British Columbia), Leonid Sigal `[通讯]` (University of British Columbia)

**通讯引用:** 10858 | [OpenAlex ID](https://openalex.org/A5053011888)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为CoRDS的KV缓存压缩方法，利用代表性核心子集选择框架在实时视频理解中高效压缩视觉语言模型的键值缓存。

**💡 创新点**

创新点包括：① 在键与值共同空间进行覆盖优化，证明可更好地逼近注意力输出；② 采用双目标D^2式最远点算法平衡键（检索）与值（内容）覆盖；③ 引入正交冗余惩罚，鼓励选择新方向的记忆，从而减少重复；④ 通过层选择与跨层级级联实现低开销的在线压缩。

**🔧 技术方法**

使用的技术主要是无训练、查询无关的几何覆盖优化（D^2最远点、正交投影）、轻量级正则化、层级压缩策略，以及Transformer内部键值投影的联合表示。

**📊 数据集**

在四种开放的视觉语言模型（Qwen2-VL、Qwen2.5-VL、LLaVA-Next-Video）上，在五个长视频和流式视频基准上评估：EgoSchema、MLVU、VideoMME、OVO-Bench、StreamingBench。

**📈 对比分析**

与InfiniPot-V、StreamMem、STC、TTC等基线以及完整KV缓存对比，CoRDS在相同压缩预算下均实现显著性能提升，并在高压缩率（1/16–1/8）时甚至超过完整KV基线；覆盖度CDF实验表明其保持的缓存更具代表性且冗余更低。

**⚠️ 局限性**

局限性包括：① 仍为查询无关的压缩，无法利用后续问题进行自适应压缩；② 主要针对视觉键值，未扩展到音频或其他模态；③ 在极低预算或极大视频流时可能仍需要进一步优化计算与内存开销。

---

## 228. KVPO: ODE-Native GRPO for Autoregressive Video Alignment via KV Semantic Exploration

**arXiv ID:** 2605.14278 | [PDF](https://arxiv.org/pdf/2605.14278v1)

**作者:** Ruicheng Zhang `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 9843 | [OpenAlex ID](https://openalex.org/A5100754504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种ODE原生的在线GRPO框架，用于对流式自回归视频生成器进行人类偏好对齐；通过Causal History Routing（CHR）实现基于KV缓存的因果语义探索，并用Trajectory Velocity Energy（TVE）在速度场空间构建代理策略，实现奖励加权的对比流匹配目标；

**💡 创新点**

创新点在于：①将探索从噪声注入转向历史KV缓存的因果语义重排，生成在数据流形上且语义多样的分支；②在ODE流匹配动力学中直接使用速度场能量量化分支似然，构造Gibbs形式的代理策略，避免外部几何距离或SDE转换；③将PPO与速度场对比损失结合，形成完整的在线策略优化流程；

**🔧 技术方法**

核心技术包括：Causal History Routing（随机重路KV槽）、Trajectory Velocity Energy（速度场残差能量）、ODE流匹配与速度场求解、PPO对比流匹配损失、LoRA参数高效微调、奖励设计（VQ、MQ、TA）以及KL正则；

**📊 数据集**

使用VidProM多提示数据集并结合Qwen3进行提示扩展；在LongLive和MemFlow两个自回归视频生成器上进行评估；

**📈 对比分析**

与Astrolabe等现有后训练对齐方法对比；在单提示短视频和多提示长视频场景下，均在VQ、MQ、TA以及VBench（质量、语义、一致性、CLIP）指标上实现显著提升；人类评估显示该方法在三项指标上均优于基线和Astrolabe；

**⚠️ 局限性**

局限性包括：①仅在已蒸馏的少步自回归模型上验证，未覆盖更大尺度或多步模型；②对CHR与TVE的超参数设置较为敏感；③仍依赖手工奖励设计，可能导致奖励劫持；④对高算力环境友好，但在资源受限设备上的可扩展性待进一步验证。

---

## 229. Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology

**arXiv ID:** 2605.14258 | [PDF](https://arxiv.org/pdf/2605.14258v1)

**作者:** Jesseba Fernando `[一作]` (Northeastern University), Grigori Guitchounts `[通讯]` (Flagship Pioneering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

计算了三款生产级大型语言模型（Llama 3.1 8B、OLMo 3 7B、Gemma 4 E4B）在每一层的精确Jacobian并完成全谱分解，随后将其动力学与残差流激活图的社区结构关联起来。

**💡 创新点**

首次揭示训练后Transformer在深度上形成的非正交梯度：早期层以旋转为主，后期层趋近对称；这一梯度导致累积低秩瓶颈，并证明社区边界单元与Jacobian增幅之间的学习性正/负耦合。

**🔧 技术方法**

使用了精确Jacobian计算、特征值分解、Schur分解、Henrici非正交度、累计Jacobian有效秩、Signed Leiden CPM社区检测、参与系数、Cohen d与FDR校正等技术手段。

**📊 数据集**

在1,000条WikiText‑2文本样本的最后token激活上进行实验。

**📈 对比分析**

通过与随机初始化（step 0）对照、条件数、对齐度、谱半径、有效秩等量化指标进行比较，训练后有效秩显著降低至≈5–7，社区结构与Jacobian变异关联显著。

**⚠️ 局限性**

仅在特定输入和模型规模下验证，社区检测对参数敏感，单层增幅效应幅度有限，未探索更深训练、不同数据分布或其他架构的泛化情况。

---

## 230. AIM-DDI: A Model-Agnostic Multimodal Integration Module for Drug-Drug Interaction Prediction

**arXiv ID:** 2605.14327 | [PDF](https://arxiv.org/pdf/2605.14327v1)

**作者:** Yerin Park `[一作]` (Inha University), Sangseon Lee `[通讯]` (Inha University)

**通讯引用:** 595 | [OpenAlex ID](https://openalex.org/A5085449381)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 AIM-DDI，一个与模型无关的多模态融合模块，用于药物-药物相互作用预测，尤其针对未见药物泛化场景；

**💡 创新点**

创新点在于将多模态信息统一映射到共享潜在空间并token化，再通过统一的Transformer+Mixture‑of‑Experts融合，解耦融合与特定预测架构，使模块可复用；并利用LLM生成语义模态，提供额外的语义信号；

**🔧 技术方法**

使用的技术包括：共享潜在空间映射、模态token化、Transformer自注意力、混合专家（Mixture‑of‑Experts）融合、LLM（Meditron‑7B）文本编码、图神经网络处理图模态，以及轻量级接口集成；

**📊 数据集**

使用的数据集为 DrugBank v5.1.7 生成的多模态 DDI benchmark（65类DDI事件），涵盖生物关系、分子子结构、药物‑药物互作网络、物理化学属性；此外在 KnowDDI 与 KGDB‑DDI 框架下评估，并在 NSAID 子集中做案例研究；

**📈 对比分析**

在一未见和双未见药物泛化设置下，与 DDIMDL、GIL‑DDI、MKG‑FENN 基线比较，AIM‑DDI 在准确率、宏F1、宏召回上显著提升（双未见准确率+23.34%，宏F1+66.04%，宏召回+86%），在 KnowDDI 与 KGDB‑DDI 框架下亦实现小幅提升；NSAID 子集准确率提升至约0.73；

**⚠️ 局限性**

局限性包括：在已见药物设置提升有限；依赖完整多模态信息，缺乏动态模态选择机制；LLM 编码成本高；混合专家路由的解释性和容量分配仍需改进；对不同模型兼容性需进一步验证；

---

## 231. TurboVGGT: Fast Visual Geometry Reconstruction with Adaptive Alternating Attention

**arXiv ID:** 2605.14315 | [PDF](https://arxiv.org/pdf/2605.14315v1)

**作者:** David Huang `[一作]` (Huawei), Dongfeng Bai `[通讯]` (Huawei)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5121229056)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 TurboVGGT，一种用于前馈多视角 3D 重建的视觉几何 Transformer，能够在一次前向传播中快速生成相机姿态、深度图和点云。

**💡 创新点**

创新点在于自适应交替注意力块：通过自适应稀疏选择将不同帧分配到不同稀疏率分支；在每个分支中学习压缩的代表性 token 并使用稀疏全局注意力进行跨帧关联，同时保持帧内自注意力以捕获局部细节；这种机制显著降低了全局注意力的计算量并提升了表示能力。

**🔧 技术方法**

技术细节包括 DINOv2 视觉编码器、稀疏注意力、权重矩阵驱动的代表性 token 学习、帧注意力、MLP 和 DPT 预测头、端到端训练及稀疏正则化损失。

**📊 数据集**

训练数据涵盖 13 个高质量场景集（BlendedMVS、Mapillary、ScanNet++、Spring、TartanAirV2-WB、UnrealStereo4K、Aria Synthetic、DL3DV-10K、Dynamic Replica、MegaDepth、MVS‑Synth、ParallelDomain‑4D、SAIL‑VOS 3D），在 7‑Scenes、N‑RGBD、ScanNet‑50、RealEstate10K、Sintel 等多种 benchmark 上进行评估。

**📈 对比分析**

与 VGGT、FastVGGT、SparseVGGT、AVGGT 等最先进方法对比，TurboVGGT 在点云准确率、完整度、法向一致性、Chamfer 距离、相机姿态误差、深度误差等指标上保持同等或更优，同时推理速度提升 2‑4 倍（最长序列可达 18 倍），且在帧数增长时扩展性更好。

**⚠️ 局限性**

局限性包括在某些密集场景指标略逊 FastVGGT，稀疏分支的稀疏率选择仍需经验调优，对动态场景和移动端实时性能的验证不足。

---

## 232. Beyond Binary: Reframing GUI Critique as Continuous Semantic Alignment

**arXiv ID:** 2605.14311 | [PDF](https://arxiv.org/pdf/2605.14311v1)

**作者:** Yuchen Sun `[一作]` (Shanghai Jiao Tong University), Chongyang Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2794 | [OpenAlex ID](https://openalex.org/A5023787090)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BBCritic，将GUI Critic从传统二分类转化为连续度量学习，并发布了支持稠密候选且具有四级层次的BBBench基准。

**💡 创新点**

创新点在于基于功能等价假设把指令与动作映射到共享的Affordance空间，并通过两阶段对比学习解决Affordance Collapse和噪声敏感问题。

**🔧 技术方法**

主要技术包括对比学习（InfoNCE）、共享Vision‑Language Encoder、温度化余弦相似度评分以及两阶段训练（粗糙初始化+细粒度硬负样本挖掘）。

**📊 数据集**

使用的训练数据来自AndroidControl和GUIOdyssey；评估数据包含自建的BBBench（18,192个样本）以及跨平台的ScreenSpotV2、Mind2Web、AndroidWorld等。

**📈 对比分析**

在NDCG、PPA和TTS成功率等指标上与多种二分类和开源VLM基准对比，BBCritic‑3B在BBBench NDCG@8达70.48、PPA≥80，超越7B参数二分类模型，并在跨平台、跨任务零样本迁移中表现优异。

**⚠️ 局限性**

局限性在于仅在TTS场景验证，训练数据仅覆盖移动平台，BBBench稀有类别（如Suboptimal）覆盖有限，未来需多平台训练和更低成本的细粒度标注。

---

## 233. ICED: Concept-level Machine Unlearning via Interpretable Concept Decomposition

**arXiv ID:** 2605.14309 | [PDF](https://arxiv.org/pdf/2605.14309v1)

**作者:** Shen Lin `[一作]` (Fujian Normal University), Li Xu `[通讯]` (Fujian Normal University)

**通讯引用:** 107010 | [OpenAlex ID](https://openalex.org/A5100440745)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于概念级解构的可解释式机器遗忘框架ICED，用以在视觉-语言模型中精准抑制目标概念而不影响非目标语义。

**💡 创新点**

创新点在于使用多模态大型语言模型构建任务特定概念词表，并将视觉表示稀疏非负分解为概念组合，从而实现概念层级的精细化遗忘与保持。

**🔧 技术方法**

技术包括多模态LLM提取概念、对齐视觉-文本向量的均值中心化、稀疏非负概念分解、三项损失（forget、intra、global）优化以及CLIP视觉文本对齐。

**📊 数据集**

使用ImageNet-1K、CIFAR-10以及对目标子类（如Marmoset）和其他数据集进行训练和测试，评估在域内与域外遗忘场景。

**📈 对比分析**

与FT、GA、Fisher、LIP、EMMN、CLIP-LIP、TIFS等基线对比，ICED在降低目标准确率的同时保持或提升非目标、整体以及跨域迁移性能，平均得分明显优于其它方法。

**⚠️ 局限性**

局限在于需构建和维护任务特定概念词表，对复杂多标签场景或高维概念覆盖仍有限，且概念分解和优化过程计算开销相对较高。

---

## 234. Language-Induced Priors for Domain Adaptation

**arXiv ID:** 2605.14301 | [PDF](https://arxiv.org/pdf/2605.14301v1)

**作者:** Qiyuan Chen `[一作]` (University of Michigan), Raed Al Kontar `[通讯]` (University of Michigan)

**通讯引用:** 703 | [OpenAlex ID](https://openalex.org/A5075324117)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在冷启动环境下利用目标域专家文本描述构造语言诱导先验（LIP），并将其融入期望最大化（EM）算法进行多源域适配，能够在目标数据稀缺时自动识别并加权相关源域。

**💡 创新点**

创新点在于首次将自然语言描述转化为源域相关性的贝叶斯先验，并通过EM实现对源域相关性的动态估计；同时提供了理论证明，证明正确LIP可实现与oracle相当的MSE，并证明任何LIP均可保持渐近一致性。

**🔧 技术方法**

采用的技术包括：预训练大型语言模型（LLM）生成LIP；基于选择模型的条件logit进行LIP估计；Bayesian层级模型与EM算法结合；利用泰勒展开和Hessian近似简化似然计算；以及温度调度（tempering）提升EM稳定性。

**📊 数据集**

实验使用了三类数据集：一维和二维高斯分布（模拟实验）；NASA C-MAPSS发动机剩余寿命预测数据；以及MuJoCo Hopper 运动学仿真数据。

**📈 对比分析**

与多种基线（仅目标、均匀先验、全源拼接）对比，LIP-aided EM在目标样本极少时显著降低RMSE/提高奖励，尤其在RUL 70%、50%和10%冷启动阶段。

**⚠️ 局限性**

局限包括：LIP的质量对冷启动性能高度敏感；LIP构造需要依赖LLM的推理能力，若LLM错误会导致负向迁移；EM在高维非高斯模型下近似可能失效；并且理论证明基于一定的“basin-entry”假设，实际场景中可能不满足。

---

## 235. MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification

**arXiv ID:** 2605.14289 | [PDF](https://arxiv.org/pdf/2605.14289v1)

**作者:** Weisen Jiang `[一作]` (Chinese University of Hong Kong), Sinno Jialin Pan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 34960 | [OpenAlex ID](https://openalex.org/A5082984558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个隐私保护的 Mixture-of-Experts（MoE）统一框架，将多个独立训练的专家模型合并为单一可部署的 MoE。

**💡 创新点**

创新点在于提出了基于相关性加权的确定性点过程（relevance-weighted DPP）进行代理样本选择，并在专家训练中引入代理对齐与上下文感知路由器，实现了无数据共享下的高效协同。

**🔧 技术方法**

使用了确定性点过程、LoRA、上下文感知路由器、Transformer、CLIP、LLaMA 等技术。

**📊 数据集**

实验数据集包括 ImageNet 作为公共数据，CV 领域的 Pets、Flowers、EuroSAT，NLP 领域的 CommonsenseQA、CosmosQA、SocialIQA；在 CV 任务中还使用 CLIP ViT-B/32/16，在 NLP 任务中使用 LLaMA-3.2-3B/8B。

**📈 对比分析**

与 ZeroShot、BTM、ModelSoup、BTX、FlexOlmo、UnrestrictedMoE 等基线比较，平均准确率在 CV 任务上从 92.45% 提升至 94.52%（CLIP ViT-B/32）或 96.24%（ViT-B/16），在 NLP 任务上从 72.50% 提升至 74.42%（LLaMA-3.2-3B）或 81.59%（LLaMA-3.1-8B）。

**⚠️ 局限性**

局限性包括：需要足够代表性的公共数据，代理样本可能无法完全覆盖私有分布；虽然提供了有限的隐私分析，但仍不具备严格的差分隐私保证；在高风险场景下可能需要额外的安全审计。

---

## 236. CreFlow: Corrective Reflow for Sparse-Reward Embodied Video Diffusion RL

**arXiv ID:** 2605.14274 | [PDF](https://arxiv.org/pdf/2605.14274v1)

**作者:** Zhenyang Ni `[一作]` (Northwestern University), Qi Zhu `[通讯]` (Northwestern University)

**通讯引用:** 5344 | [OpenAlex ID](https://openalex.org/A5020896290)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对机器人抓取与操作任务，本文提出一种后训练视频生成模型的框架，在生成的视频中使用逻辑约束来评估任务成功与否，并利用评估结果指导模型更新。

**💡 创新点**

核心创新点包括①用有限追踪线性时序逻辑（LTL）自动化生成任务约束并输出细粒度的违规轨迹；②提出信用感知NFT损失，限定强化学习梯度仅作用于违规相关的时空区域；③设计纠正性回流损失，利用同一条件下成功样本的均值作为失败样本的纠正目标，从而提升训练稳定性与收敛速度。

**🔧 技术方法**

技术手段包括视频扩散/流模型（Ti2V）、DiffusionNFT近似、SAM3、VLM、IDM用于状态提升、LTL监测器、信用感知NFT与纠正回流损失。

**📊 数据集**

使用RoboTwin bimanual manipulation数据集（共8个任务，如pick‑and‑place、stacking、drawer‑open等），并在100条保留视频上评估奖励准确性。

**📈 对比分析**

与基线（策略级参考、Vidar基线、DanceGRPO、DiffusionNFT、EVA等）对比，本文方法在所有8个任务上均取得最高成功率，平均提升23.8%，训练收敛更快；奖励准确度达到88%（与人类和模拟器标签一致）。

**⚠️ 局限性**

局限性主要在于约束评估仅基于二维图像平面状态，无法完全捕捉三维空间关系，且LTL监测在单个H100上每步约需15秒，仍占一定计算开销。

---

## 237. What Makes Words Hard? Sakura at BEA 2026 Shared Task on Vocabulary Difficulty Prediction

**arXiv ID:** 2605.14257 | [PDF](https://arxiv.org/pdf/2605.14257v1)

**作者:** Adam Nohejl `[一作]` (RIKEN), Hitomi Yanaka `[通讯]` (RIKEN)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5045824013)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究英语词汇难度预测，提出利用soft‑target交叉熵微调LLM/MLM的方法，并构建基于SHAP的可解释特征模型；对KVL数据中词汇难度进行深入分析。

**💡 创新点**

创新点在于：1）使用soft‑target交叉熵损失和概率加权推理直接预测连续难度值，避免离散化损失的精度损失；2）结合LLM提示提取可解释特征，并用SHAP提供局部与全局解释；3）分析测试项构造与拼写难度对难度分数的影响。

**🔧 技术方法**

使用技术包括：LLM/MLM微调（GLM‑4、Qwen2.5、Ministral‑3、mmBERT、XLM‑RoBERTa）、soft‑target交叉熵、G‑Scale温度缩放、SHAP解释、XGBoost回归、线性堆叠、LLM提示（GPT‑4.1、GPT‑5.2、DeepSeek）。

**📊 数据集**

使用数据集：British Council Knowledge‑based Vocabulary Lists (KVL) 任务数据，包含L1 Chinese、German、Spanish的词汇难度分数；以及Lang‑8、TUBELEX、BNC、EVP等语料库用于频率、CEFR、拼写难度等特征。

**📈 对比分析**

方法与官方基线（fine‑tuned XLM‑RoBERTa）、OpenTrack baseline、统计最优等进行比较；开放轨道模型RMS约0.7，Pearson r>0.91；闭轨道模型RMS约1.1–1.2，r>0.77；soft‑target方法明显优于传统MSE或离散化损失。

**⚠️ 局限性**

局限性包括：仅在KVL任务验证，是否适用于其他任务未知；在小规模或噪声数据上效果不明；参数如刻度点数需调优；缺乏对解释结果的外部验证。

---

## 238. Generative Deep Learning for Computational Destaining and Restaining of Unregistered Digital Pathology Images

**arXiv ID:** 2605.14251 | [PDF](https://arxiv.org/pdf/2605.14251v1)

**作者:** Aarushi Kulkarni `[一作]` (University of California), Pratik Shah `[通讯]` (University of California)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5054755071)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在未注册的跨机构数字病理切片上，评估预训练的条件生成对抗网络（cGAN）模型在计算 H&E 染色与去染（destaining）任务中的泛化性能，并通过染色标准化和通道强度校准等预处理手段实现域适应。

**💡 创新点**

①证明仅利用预处理即可在外部未注册切片上保持预训练模型的可用性；②发现去染‑染色循环（destaining–staining）在像素级相似度上优于直接染色，提示预处理质量是限制因素；③系统性展示了模型在恶性腺体形态上的失真与局限。

**🔧 技术方法**

技术手段包括：条件生成对抗网络（pix2pix U‑Net + PatchGAN）进行图像转换；Macenko 染色标准化与累计分布函数直方图对齐；针对未染切片的通道加权强度校准；ECC 预配准（仅做刚性变换）；分块提取 1024×1024 像素；使用 PCC、SSIM、PSNR、MSE 等指标与病理学专家评估相结合。

**📊 数据集**

数据集：训练集 dataset_B（102 张已注册前列腺核心活检 WSIs，来自 Brigham and Women’s Hospital）；测试集 dataset_S（82 张未注册前列腺核心活检 WSIs，来自 Stanford University）。

**📈 对比分析**

通过对比内部验证（dataset_B 重推断）与外部未注册测试，使用无配准下的像素级相似度评估：
- 去染（VDS vs. GUS）PCC 0.854 ± 0.04，SSIM 0.699 ± 0.11，PSNR 18.4 ± 1.55 dB；
- 去染‑染色循环（VH&ER vs. GH&E）PCC 0.798 ± 0.03，SSIM 0.756 ± 0.09，PSNR 20.08 ± 1.82 dB，显著优于直接染色（VH&E vs. GH&E）PCC 0.715，SSIM 0.718，PSNR 18.5 dB；
- 与内部验证相比，外部性能下降但仍保持可用性。

**⚠️ 局限性**

限制：样本仅来自单一外部机构，未进行模型微调或其他域适应方法；预处理无配准，仅做刚性对齐；对恶性腺体高分级/复杂形态的保真度不足；仅针对前列腺核心活检与 pix2pix 架构，推广至其他组织或更大规模多中心数据仍需验证。

---

## 239. Uncovering the Representation Geometry of Minimal Cores in Overcomplete Reasoning Traces

**arXiv ID:** 2605.14358 | [PDF](https://arxiv.org/pdf/2605.14358v1)

**作者:** Sanjoy Chowdhury `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**通讯引用:** 40021 | [OpenAlex ID](https://openalex.org/A5004194238)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究语言模型生成的推理链，提出最小核心（minimal core）概念，识别并剔除冗余推理步骤，只保留对最终预测有决定性贡献的步骤。

**💡 创新点**

创新点在于将推理轨迹视为可压缩的“过完备”对象，定义答案或分布保持的最小子集，给出压缩比、冗余质量、必要性集中度等量化指标，并证明贪心删除的局部不可约性与过完备性判定。

**🔧 技术方法**

使用贪心向后删减算法、留一法必要性评分、必要性加权嵌入、表示几何分析（可变度、内在维度、正确-错误分离）等技术来提取和评估最小核心。

**📊 数据集**

在六个推理基准上进行实验，包括数学算术（GSM8K、MATH500、AIME24、AMC23）、科学推理（GPQA‑Diamond）和常识多跳问答（StrategyQA），覆盖多模型（DeepSeek、Qwen、GPT‑5）。

**📈 对比分析**

与随机删除或一次性必要性裁剪比较，贪心最小核心在保持答案正确率约86%（比随机低约30%）的同时压缩了约46%的步骤，核心嵌入在正确性探测器上提升约11个百分点，内在维度下降约34%，显示更清晰的几何分离。

**⚠️ 局限性**

局限在于仅对模型行为做必要性判定，缺乏因果解释；贪心算法只能得到局部最优；需要开源模型才能利用似然或隐藏状态度量；结果对提示、分段、领域和交互式推理的鲁棒性仍待进一步验证。

---

## 240. Refining Pseudo-Audio Prompts with Speech-Text Alignment for Text-Only Domain Adaptation in LLM-Based ASR

**arXiv ID:** 2605.14340 | [PDF](https://arxiv.org/pdf/2605.14340v1)

**作者:** Ryo Magoshi `[一作]` (Kyoto University), Yusuke Shinohara `[通讯]` (LY Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 LLM‑based ASR 的文本仅域适配任务中，本文提出了 Text‑Embedding‑to‑Speech‑Latent（TE2SL）框架，利用可学习的 Conformer‑基精炼模块将文本嵌入映射到音频编码器与投影器的潜在空间，从而生成与目标域音频特征一致的伪音频提示，实现对目标域的有效适配。

**💡 创新点**

创新点在于：①引入样本依赖且对音频编码器/投影器特性感知的可学习精炼模块；②通过对齐真实音频提示的 MSE 损失，显著降低了模态不匹配；③实现了无需 TTS 的多语种可扩展性。

**🔧 技术方法**

核心技术包括：LLM‑based ASR 框架（WavLM‑Large + Llama‑3.2‑3B + LoRA），Conformer 编码器 + 两层线性层的精炼模块，帧上采样与时间掩码的伪音频生成策略，以及 MSE 对齐损失。

**📊 数据集**

实验使用的语料为：英语源域 LibriSpeech 960h，目标域 SPGISpeech（1.93M 句子）与 SlideSpeech（482k 句子）；日语源域 CSJ‑SPS 257h，目标域 CSJ‑APS 130k（eval1/eval2）。

**📈 对比分析**

与 Baseline（无适配）、Soft Prompt 与 Upsample‑and‑Mask 三种基线比较，TE2SL 在所有任务中均取得最低的 WER/CER 与最高的 OOV recall，显示出显著的性能提升。

**⚠️ 局限性**

限制方面包括：需依赖预训练的音频编码器与投影器（如 WavLM English）以获取特征映射，源域音频‑文本对的质量与规模会影响精炼模块训练；仅在两种语言上验证，未探讨跨语言或低资源场景的泛化；推理时额外的精炼模块可能增加计算开销。

---

## 241. The Great Pretender: A Stochasticity Problem in LLM Jailbreak

**arXiv ID:** 2605.14418 | [PDF](https://arxiv.org/pdf/2605.14418v1)

**作者:** Jean-Philippe Monteuuis `[一作]`, Jonathan Petit `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了在攻击生成与评估过程中随机性对攻击成功率（ASR）的影响，并提出了基于一致性的攻击成功度量（CAS）与两种评估框架CAS‑gen与CAS‑eval。

**💡 创新点**

创新点在于：①首次同时量化生成阶段和评估阶段的随机性对ASR的影响；②引入CAS指标，要求多次判定均为有害以排除偶然成功；③设计CAS‑gen和CAS‑eval两套框架，分别在生成和评估阶段过滤随机噪声；④提出最小报告清单（k、T、θ、seed）以标准化报告。

**🔧 技术方法**

技术主要包括：多种黑盒对抗攻击（Best‑of‑N、PAIR、TAP、Crescendo）；LLM生成与评估；温度控制（T、θ）；随机种子采样；统计分析与误差评估；以及对CAS指标的数学定义和实现。

**📊 数据集**

使用的数据集为JailbreakBench子集；目标模型为Llama‑3.2‑1B、Llama‑3.1‑8B、Llama‑3.1‑70B、Gemma‑3‑1B、Granite‑4.0‑1B；评估判定器为Llama‑Guard‑3‑1B与Llama‑Guard‑3‑8B。

**📈 对比分析**

通过在不同参数组合（k、T、θ、seed）下多次实验，比较ASR的变化。结果显示：单次评估的ASR高估12‑54个百分点；生成预算k提升12‑30个百分点；评估温度θ升至1.0可导致ASR下降至54个百分点。CAS‑gen/CAS‑eval能显著降低因随机性导致的误报。

**⚠️ 局限性**

局限性包括：仅评估文本攻击，未覆盖梯度攻击或多模态攻击；仅针对开放权重模型，未验证在高度RLHF对齐的商业模型上的泛化；评估仅考虑LLM判定器，未覆盖规则或混合安全栈；未考虑系统提示、分词器差异等其它不可复现来源。

---

## 242. Metis AI: The Overlooked Middle Zone Between AI-Native and World-Movers

**arXiv ID:** 2605.14407 | [PDF](https://arxiv.org/pdf/2605.14407v1)

**作者:** Xiang Li `[一作]` (Massachusetts General Hospital), Xiang Li `[通讯]` (Massachusetts General Hospital)

**通讯引用:** 22665 | [OpenAlex ID](https://openalex.org/A5100331028)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

对数字化任务中AI能否可靠自动化的边界进行理论性重新界定，提出Metis AI概念及其五大支柱，并通过跨学科理论构建框架分析各行业任务的定位。

**💡 创新点**

创新点在于：①将传统的数字/物理任务二分法扩展到数字任务内部，提出“Metis AI”区间；②区分constitutive metis与operational metis；③提出Consequential irreversibility、Relational irreducibility、Normative open texture、Adversarial co‑evolution、Accountability anchoring五大结构特征；④强调人类首领（human‑in‑the‑lead）而非人类监督（human‑in‑the‑loop）的系统设计原则。

**🔧 技术方法**

主要使用哲学（技术与智慧的区分）、社会科学（制度、关系、规范理论）和计算机科学（AI 体系架构讨论）等文献与理论工具；未提出新的算法实现。

**📊 数据集**

本研究未使用实验数据集，全部基于文献综述与理论推导。

**📈 对比分析**

因为是理论框架，无实验对比与性能指标。作者通过案例与领域分析说明该框架能解释多行业任务的可自动化程度，但未给出定量性能评估。

**⚠️ 局限性**

限制在于：①缺乏经验验证，框架的可操作性和适用范围尚未通过大规模实证检验；②对复杂任务的分层细化仍依赖专家判断；③在动态变化的技术与制度环境下，五大支柱的边界可能需要不断调整。

---

## 243. Coding Agent Is Good As World Simulator

**arXiv ID:** 2605.14398 | [PDF](https://arxiv.org/pdf/2605.14398v1)

**作者:** Hongyu Wang `[一作]` (University of Wisconsin-Madison), Dan Negrut `[通讯]` (University of Wisconsin-Madison)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建可执行物理仿真代码，以多代理框架把自然语言提示转化为可运行的PyChrono程序，实现可修复、物理可逼真的世界模型。

**💡 创新点**

将世界建模任务转为可执行代码生成的代理式流程，并结合规划、代码生成、视觉审核与物理验证，实现对物理约束的显式控制与迭代修正。

**🔧 技术方法**

使用LLM多代理系统、PyChrono物理引擎、技能库、工具接口、API索引、视觉LLM和物理日志验证等技术。

**📊 数据集**

利用公共3D资产库、Chrono自带资产、WorldModelBench基准；实验场景包括Go2机器人、HMMWV车辆和Polaris浮块等。

**📈 对比分析**

与视频生成基线Wan2.2-TI2V-5B在WorldModelBench上比较，场景总分更高，指令遵循得分显著提升，物理法则与常识得分相当；代码生成在物理一致性上优于单纯视频生成。

**⚠️ 局限性**

修复过程非单调、资产库有限导致缺失物体需近似、技能库和API版本限制不支持自定义传感器或少见物理模型、实验规模小且计算成本高。

---

## 244. Analogical Trajectory Transfer

**arXiv ID:** 2605.14393 | [PDF](https://arxiv.org/pdf/2605.14393v1)

**作者:** Junho Kim `[一作]` (Seoul National University), Young Min Kim `[通讯]` (Seoul National University)

**通讯引用:** 29415 | [OpenAlex ID](https://openalex.org/A5100337311)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出无训练的类比轨迹转移框架，能够将一个3D场景中的运动轨迹映射到另一个语义相似的场景，同时保持功能意图和空间连贯性；通过对象聚类、图匹配、TPS平滑映射以及梯度优化实现精确转移。

**💡 创新点**

在无监督条件下利用3D基础模型的场景级特征进行层级分块匹配，并在每个子块上拟合多种平滑映射，随后通过组合优化与梯度细化实现全局空间连贯的轨迹转移，支持多模态输出。

**🔧 技术方法**

使用3D基础模型（如Sonata）提取点云特征；聚类+图匹配实现对象级对应；薄板样条（TPS）拟合平滑映射；光束搜索优化全局映射；梯度下降轨迹优化；A*规划实现稀疏路径转移。

**📊 数据集**

3D-FRONT（合成室内场景）、ARKitScenes（真实扫描场景）以及两者的交叉域对比场景。

**📈 对比分析**

与Feature NN、Neural Contextual Scene Maps、Foundation Model Analogies、Monte Carlo Localization、Graph Matching、LLM等多种基线对比，均在碰撞率、特征距离、AED、Inlier率等指标上取得显著优势；在3D-FRONT和ARKitScenes上的平均运行时间约0.65秒，显著快于LLM基线。

**⚠️ 局限性**

仅针对静态且语义相似的场景；无法处理对象动态交互或功能差异很大的场景；评测依赖人工标注和单侧指标，缺乏可扩展的自动化基准；未验证在桌面或户外等非室内环境中的效果。

---

## 245. Where Should Diffusion Enter a Language Model? Geometry-Guided Hidden-State Replacement

**arXiv ID:** 2605.14368 | [PDF](https://arxiv.org/pdf/2605.14368v1)

**作者:** Injin Kong `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 4651 | [OpenAlex ID](https://openalex.org/A5016844435)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DiHAL，一种几何引导的扩散-Transformer混合架构，将连续扩散从词元级别迁移到内部隐藏状态重建，并通过定位最佳插入层来提升语言生成质量。

**💡 创新点**

创新点在于：①基于曲率、全局刚度和有效秩三种几何代理自动识别“扩散友好”层；②采用Locate‑and‑Replace策略，用扩散桥接替代前缀Transformer层，保持上层和原LM头不变；③在固定训练预算下证明几何评分能有效预测桥接可行性。

**🔧 技术方法**

技术包括：连续扩散理论（Langevin动力学与浓度理论）作为几何准则，UNet式扩散桥接，层级几何代理（局部曲率、全局刚度、有效秩）计算，桥接训练损失组合（diffusion、重构、LM蒸馏），以及Transformer的前缀裁剪与条件化。

**📊 数据集**

使用的主要数据集有Dolma v1.7（300K/150K样本用于桥接训练和验证）和WikiText-103（评估NLL/Perplexity），以及在Dolma hold‑out上进行的对比评估。

**📈 对比分析**

与传统连续扩散基线（Diffusion‑LM、SED、LD4LG、CoDAR）在相同训练预算（300K样本、40小时H100）下对比，DiHAL在Gen.PPL和多样性指标上取得显著提升，最终在完整训练后在NLL/Perplexity上与基线oracle相当或更优。

**⚠️ 局限性**

局限性包括：仍依赖原Transformer后缀和LM头，无法实现完全独立的扩散语言模型；桥接规模受计算限制，未探索更大桥接或更深层替换；在极大规模模型或长文本上效果尚未验证。

---

## 246. Automated Curriculum Design for High-dimensional Human Motor Learning

**arXiv ID:** 2605.14367 | [PDF](https://arxiv.org/pdf/2605.14367v1)

**作者:** Ankur Kamboj `[一作]` (Michigan State University), Vaibhav Srivastava `[通讯]` (Michigan State University)

**通讯引用:** 3377 | [OpenAlex ID](https://openalex.org/A5069896928)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并验证了一种基于人类运动学习模型与粒子滤波实时技能估计的随机非线性模型预测控制（SNMPC）框架，用于高维手部运动的自适应课程设计。

**💡 创新点**

创新点在于：① 无需预先收集建模数据即可实时估计隐含技能状态；② 将技能演化与任务性能耦合进预测控制的成本函数，实现个性化的、前瞻性的训练任务调度；③ 通过UCM分析验证了学习效率提升。

**🔧 技术方法**

主要技术包括：人类运动学习（HML）模型、粒子滤波器、SNMPC（含软最小化随机化）、Uncontrolled Manifold分析、线性混合模型统计检验。

**📊 数据集**

数据集来源于36名健康受试者（12人/组）使用SenseGlove DK1手部外骨骼完成的目标捕获游戏，总计8个块，每块60次试验。

**📈 对比分析**

与随机调度和基于性能启发式的调度相比，SNMPC显著缩短了学习时间（约23%–27%），在学习误差和轨迹直线度等指标上均表现更优，且在UCM分析中表现出更高的无关空间变异比例。

**⚠️ 局限性**

局限性包括：对HML模型的匹配度敏感，模型误差会显著影响学习速度；需要较高的计算资源以支持实时预测控制；实验样本量有限，尚未验证在不同任务或更大人群中的泛化能力。

---

## 247. A Formative Study of Brief Affective Text as a Complement to Wearable Sensing for Longitudinal Student Health Monitoring

**arXiv ID:** 2605.14360 | [PDF](https://arxiv.org/pdf/2605.14360v1)

**作者:** Tamunotonye Harry `[一作]` (University of Vermont), Christopher Danforth `[通讯]` (University of Vermont)

**通讯引用:** 6160 | [OpenAlex ID](https://openalex.org/A5002034958)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对458名大学生进行为期一年的 Oura 手环跟踪，并在每两周收集一次极短的“担忧”文本，随后使用三种 NLP 方法（字典、RoBERTa‑base、MentalRoBERTa）以及零射域分类，构建与睡眠和体能指标的混合效应模型，探究文本情感与生理数据的关系。

**💡 创新点**

证明即使是平均仅三词的极短开放式文本，其情感调性（而非主题）也能显著预测同周的睡眠质量、心率变异性和身体活动；RoBERTa‑base 通用预训练模型在大多数指标上优于字典方法，而对自主神经指标（RMSSD）域适配模型略有优势；并展示了字典方法无法捕捉到的分布式语义维度。

**🔧 技术方法**

使用 SEANCE 字典特征、RoBERTa‑base 句子嵌入（PCA 降维）、MentalRoBERTa 领域适配嵌入（PCA 降维）以及零射域分类，全部基于混合效应回归（随机截距+周数+字数）。

**📊 数据集**

学生 Oura Ring 采集的睡眠（6项）和体能（3项）指标数据，配合每两周一次的自然语言“担忧”文本，总共 3,610 份响应（3,073 有文本）。

**📈 对比分析**

在 9 项生理指标上，RoBERTa‑base 的语言块 ΔR² 通常高于 SEANCE（最多 0.0072），MentalRoBERTa 在 RMSSD、睡眠潜伏期等自主神经指标上略高。零射域分类几乎没有显著关联，表明情感维度是关键信号。整体语言解释的方差微小（ΔR² < 0.01），但在高 ICC 的结构下已具备统计意义。

**⚠️ 局限性**

样本单一（Vermont 大学，主要为白人女性）、文本极短且仅每两周一次，缺乏对快速情绪变化的敏感性；高 ICC 限制了可解释的周内变异；无法建立因果关系；RMSSD 高 ICC 限制了语言对该指标的解释空间；零射域分类结果受限于训练语料与目标文本的差异；未对文本情感进行人工验证，模型解释仍有不确定性。

---

## 248. Learning with Semantic Priors: Stabilizing Point-Supervised Infrared Small Target Detection via Hierarchical Knowledge Distillation

**arXiv ID:** 2605.14346 | [PDF](https://arxiv.org/pdf/2605.14346v1)

**作者:** Yuanhang Yao `[一作]` (Dalian University of Technology), Weimin Wang `[通讯]` (Dalian University of Technology)

**通讯引用:** 36734 | [OpenAlex ID](https://openalex.org/A5100391700)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于冻结视觉基础模型（VFM）的层次化知识蒸馏框架，结合点标注的在线标签演化，用于提升单帧红外小目标检测性能。

**💡 创新点**

创新点在于将冻结的VFM语义通过Semantic‑Conditioned Affine Modulation（SCAM）注入CNN特征，并采用双层bilevel优化与动态协同学习来抑制伪标签噪声并提升验证集泛化。

**🔧 技术方法**

采用了冻结的DINOv3 VFM、SCAM、多层bilevel优化、聚类重加权的动态协同学习、温度软化的知识蒸馏等技术。

**📊 数据集**

在SIRST3数据集（SIRST‑v1、NUDT‑SIRST、IRSTD‑1k）上进行实验，并按Salient、Filamentary、Faint、Camouflaged四个子集进行细分评估。

**📈 对比分析**

与LESPS、PAL等基线及多种轻量级backbone（如ALCLNet、GGLNet、DNANet等）比较，实验表明在所有指标（IoU、nIoU、P_d、F_a）上均显著提升，尤其在Faint和Camouflaged子集中IoU提升约3–4%。

**⚠️ 局限性**

局限性包括训练阶段需额外的VFM计算，方法对点标注质量和伪标签噪声仍有一定依赖，且在极低对比度或极小目标场景下仍可能出现误检。

---

## 249. A Unified Knowledge Embedded Reinforcement Learning-based Framework for Generalized Capacitated Vehicle Routing Problems

**arXiv ID:** 2605.14416 | [PDF](https://arxiv.org/pdf/2605.14416v1)

**作者:** Wen Wang `[一作]` (Nanjing University), Xianping Tao `[通讯]` (Nanjing University)

**通讯引用:** 2537 | [OpenAlex ID](https://openalex.org/A5100508660)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种知识嵌入的路优先–簇后（RFCS）框架，将强化学习与动态规划相结合，能够统一求解多种含约束的 CVRP 变体。

**💡 创新点**

创新点在于：①利用动态规划模块提供可达解的成本作为奖励，强化学习不再依赖手工 TSP 近似；②通过历史增强的上下文（LSTM）克服因分解产生的部分可观测性，实现对不同约束的无差异化统一处理；③在保持单一网络结构的同时实现显著提升的解质量与零样本迁移性能。

**🔧 技术方法**

采用了 Transformer 编码器-解码器网络、LSTM 历史上下文模块、混合注意力机制、动态规划分割算法以及 REINFORCE 强化学习训练。

**📊 数据集**

使用统一随机生成的 CVRP 变体实例（50/100 顾客），训练时按 40/50 的容量分布，评估时采用公开的 benchmark 数据集（与 PyVRP、OR‑Tools 等同一数据集）。

**📈 对比分析**

与经典非学习基线（PyVRP、OR‑Tools）以及三种学习基线（MTPOMO、MvMOE、RouteFinder）在 1000 个未见实例上进行比较，RFCS 在 16 种 CVRP 变体上平均成本均优于其它学习方法，最优性差距从 2.60% 降低到 1.82%，并在零样本迁移（容量分布、混合线haul 约束）中表现更佳。

**⚠️ 局限性**

局限性包括：①动态规划的 O(n²) 复杂度在极大实例上可能成为瓶颈；②历史上下文采用 LSTM，非最前沿；③在某些变体（如 VRPL）略逊于 RouteFinder；④对混合线haul 约束的表现不如最优；未来需改进网络架构与扩展到更大规模或其他领域。

---

## 250. Energy-Efficient Quadruped Locomotion with Compliant Feet

**arXiv ID:** 2605.14411 | [PDF](https://arxiv.org/pdf/2605.14411v1)

**作者:** Pramod Pal `[一作]` (Indian Institute of Science), Ashitava Ghosal `[通讯]` (Indian Institute of Science)

**通讯引用:** 2755 | [OpenAlex ID](https://openalex.org/A5031432869)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在四足机器人中引入可调节弹簧足部，并使用强化学习控制器学习不同弹簧刚度下的行走策略，随后在仿真与真实硬件上交叉评估其能耗表现。

**💡 创新点**

首次系统性研究弹性足部对RL控制器能耗的影响，发现中等刚度（约14500 N/m）能实现U形能耗曲线，显著降低能耗，同时保持稳定性。

**🔧 技术方法**

采用基于Isaac Gym的GPU加速仿真、PPO强化学习、域随机化、低阶PD跟踪控制以及ROS 2实现硬件部署。

**📊 数据集**

没有使用公开数据集，而是在仿真中随机化物理参数（质量、摩擦、重力等）生成8种弹簧刚度（1000–60000 N/m）并交叉评估，硬件实验同样覆盖这8种刚度。

**📈 对比分析**

通过能耗（机械功耗/米）和姿态、步态一致性指标进行比较；结果显示中等刚度比最软或最硬弹簧降低约12–17 %能耗，U形曲线在仿真与硬件中基本一致。

**⚠️ 局限性**

局限性包括：仅在平坦地面验证；硬件能耗高于仿真约25 %（未建模的摩擦、回馈损失等）；对极端刚度的鲁棒性未深入探究；缺乏对不同地形或更大步态库的泛化评估。

---

## 251. GeoViSTA: Geospatial Vision-Tabular Transformer for Multimodal Environment Representation

**arXiv ID:** 2605.14406 | [PDF](https://arxiv.org/pdf/2605.14406v1)

**作者:** Yuhao Liu `[一作]` (Rice University), Guha Balakrishnan `[通讯]` (Rice University)

**通讯引用:** 3419 | [OpenAlex ID](https://openalex.org/A5081710525)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Geospatial Vision-Tabular Transformer，联合学习地理影像与表格数据的统一嵌入，并通过自监督 Masked Autoencoder 进行训练。

**💡 创新点**

首次将连续格网影像与不规则行政区表格属性通过地理感知跨模注意力对齐，加入位置与几何编码，并实现双向交叉注意力，弥补传统多模模型对空间邻近性和几何异构的忽视。

**🔧 技术方法**

使用 Vision Transformer 与改进的 Tabular Transformer（含行自注意力），双向跨模注意力，地理位置与几何编码，ALiBi 风格的空间偏置，以及 Masked Autoencoder 目标。

**📊 数据集**

使用 AlphaEarth 的 64 维地球观测嵌入（2017-2024）与 2018 年美国气候脆弱性指数（CVI）表格数据，在 CONUS 训练，留出华盛顿州做零样本测试。

**📈 对比分析**

与单模 CVI、AlphaEarth、特征拼接、晚期融合四种基线进行线性探针比较；模型在多种健康死亡率和火灾风险等下游任务中获得最高 R²，并在零样本迁移至华盛顿州时优于所有基线。

**⚠️ 局限性**

仅在 CONUS 与单一表格数据上验证；表格聚合层级可能掩盖细粒度差异；模型仅学习相关性而非因果；仅处理静态快照，缺乏时间动态建模。

---

## 252. Watch your neighbors: Training statistically accurate chaotic systems with local phase space information

**arXiv ID:** 2605.14405 | [PDF](https://arxiv.org/pdf/2605.14405v1)

**作者:** Joon-Hyuk Ko `[一作]` (Korea Institute for Advanced Study), Deok-Sun Lee `[通讯]` (Korea Institute for Advanced Study)

**通讯引用:** 3209 | [OpenAlex ID](https://openalex.org/A5042241464)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于邻域覆盖的Neural ODE训练方法，用来学习混沌系统的动力学，既保证长时间统计一致性，又能精确捕捉局部扩张收缩特性。

**💡 创新点**

创新点在于：①将混沌吸引子分解为局部邻域集合，跟踪邻域在真实动力学与模型动力学下的映射；②利用最大均值差(MMD)对邻域的推进测度进行正则化；③通过设定内外半径形成环形邻域，对噪声更鲁棒；④在不需要真实雅可比信息的情况下实现对雅可比的精确学习。

**🔧 技术方法**

采用Neural ODE、全连接前馈网络、5阶自适应Runge–Kutta求解、Taylor展开近似邻域演化、KDTrees高效邻域检索、MMD损失、Adabelief优化器。

**📊 数据集**

在三类混沌系统上验证：Lorenz 63（3D）、Hyperchaotic Chen（4D）和Lorenz 96（6D），使用 50 条轨迹、每条 1000 步，分别在 0%、1%、5%、10% 的高斯噪声下测试。

**📈 对比分析**

与普通轨迹拟合（Vanilla）和DySLIM两种基线对比；结果显示：在低噪声下三种方法短时预测相当；在噪声增大时，邻域法和DySLIM 的有效预测时间明显优于 Vanilla；在长期统计一致性上，邻域法与 DySLIM 接近，稍逊于 DySLIM；在局部动力学（向量场与雅可比误差）上，邻域法显著优于两者，尤其在敏感的转移区域。

**⚠️ 局限性**

局限性：①需手动设定邻域半径、MMD权重等超参数，仍需网格搜索；②计算邻域覆盖与推测成本相对较高；③仅在完全可观测、低维系统上验证，未考察高维或部分观测情形；④对理论收敛性和稳定性提供的证明仍不完整。

---

## 253. Agentic Recommender System with Hierarchical Belief-State Memory

**arXiv ID:** 2605.14401 | [PDF](https://arxiv.org/pdf/2605.14401v1)

**作者:** Xiang Shen `[一作]` (Meta), Hong Yan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大语言模型的三层记忆架构（事件、偏好、个人画像）和完整生命周期的代理式推荐系统；

**💡 创新点**

创新点在于：①用层级记忆分离观察、信念、身份三重抽象；②设计六种内存操作（提取、加强、削弱、合并、遗忘、合成）并由LLM规划器自适应调度；③将偏好记忆作为中间抽象层，提升画像质量而非直接作为推荐输入；

**🔧 技术方法**

主要技术包括：POMDP框架下的结构化信念状态、LLM（Llama-4-Maverick-17B）作为提取器、规划器与排名器、三层记忆的增量更新与消亡机制；

**📊 数据集**

使用InstructRec基准数据集，涵盖Amazon Books、Goodreads、MovieTV、Yelp四个领域；

**📈 对比分析**

与传统协同过滤（BPR、LightGCN）、序列模型（SASRec）、LLM排名（LLMRank）以及其他记忆式代理方法（AgentCF、iAgent、MemRec）比较，取得HR@1提升约8–46%、NDCG@10提升约3–11%，在四个领域均为最优；在在线演化模式下自适应调度比固定调度提升约2–22%；

**⚠️ 局限性**

局限性：1）依赖LLM计算资源，虽然相对高效但仍不适合极低成本部署；2）记忆生命周期仍为启发式规则，缺乏理论最优证明；3）实验主要集中在文本描述丰富的商品与点评数据，对多模态或极稀疏场景的泛化尚未验证。

---

## 254. SceneForge: Structured World Supervision from 3D Interventions

**arXiv ID:** 2605.14399 | [PDF](https://arxiv.org/pdf/2605.14399v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 255. Delta Forcing: Trust Region Steering for Interactive Autoregressive Video Generation

**arXiv ID:** 2605.14382 | [PDF](https://arxiv.org/pdf/2605.14382v1)

**作者:** Yuheng Wu `[一作]` (KAIST), Dongman Lee `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对交互式多事件实时视频生成中的“条件偏差”问题，提出了一种基于可信度的自适应教师监督框架，能在保持对新事件的即时响应的同时显著提升长期时间一致性。

**💡 创新点**

创新点：①用可信度阈值（Trust‑Region）动态判断教师指导是否可靠；②通过教师与生成器在潜在空间的 delta 对比，衡量两者的迁移一致性；③在教师不可靠时使用“连续性损失”以生成器自身的历史动量为参考，防止漂移。

**🔧 技术方法**

主要技术：TRPO启发的可信度控制、潜在空间 delta 计算、DINO 特征提取、DMD（Distribution‑Matching Distillation）与连续性损失的联合优化；实验采用 WAN‑2.1‑1.3B‑T2V 生成器和 WAN‑2.1‑14B‑T2V 教师；训练分两阶段：Causal Forcing + Streaming Long Tuning。

**📊 数据集**

数据集：MemFlow 基准（100个60s视频，6个10s事件），VBench（6维度评估），Long‑CLIP（指令跟随评估），VideoAlign（人类偏好评估）。

**📈 对比分析**

与 SkyReels‑V2、MAGI‑1、LongLive、MemFlow、Reward‑Forcing 等方法对比：在 VBench 的主体与背景一致性上取得最高分，指令跟随保持领先，VideoAlign 总分、运动质量、文本对齐均优于基线；用户研究显示在美学、动态和多事件自然度三项指标均获得最低平均排名。

**⚠️ 局限性**

局限性：实验规模受限，模型容量、数据多样性与交互复杂度不足；未验证更大规模模型或更丰富场景下的鲁棒性与通用性。

---

## 256. Fast Gossip-based Rumor Spreading using Small Messages

**arXiv ID:** 2605.14376 | [PDF](https://arxiv.org/pdf/2605.14376v1)

**作者:** Fabien Dufoulon `[一作]` (Lancaster University), Gopal Pandurangan `[通讯]` (University of Houston)

**通讯引用:** 3787 | [OpenAlex ID](https://openalex.org/A5089574654)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了两种 gossip 算法，用以在任意未知图中实现快速的消息传播（rumor spreading）。第一种算法的运行时间为 O(c log n / Φ_c)，其中 Φ_c 为弱导线度；第二种算法的运行时间为 Õ(D + √n)，其中 D 为图的直径。两种算法均仅使用常数（O(1)) 大小的消息，并可直接构造最小生成树、领导选举和聚合函数。

**💡 创新点**

创新点在于：①在保持 gossip 轻量级通信（每轮仅与一个邻居通信）的前提下，实现了基于弱导线度和直径的快速传播；②突破以往仅在使用线性/大消息量时才能达到此速度的瓶颈；③通过图 sketching、sunflower/超聚类拆分、低度顶点覆盖模拟以及稀疏化技术，构建了新的传播和聚合方法。

**🔧 技术方法**

主要技术包括：图 sketching（线性压缩和随机采样），弱导线度与 sunflowers 的结构分解，超级聚类（super‑cluster）原语用于在 gossip 模型下模拟超图通信，低度顶点覆盖仿真（simulating one round of dense communication with O(log n) blow‑up），以及基于这些技术的稀疏化与 MST 构造。

**📊 数据集**

本文为理论研究，未使用实验数据集；所有结论均通过证明和概率分析得到。

**📈 对比分析**

与现有工作比较：在弱导线度下，本文的 O(c log n / Φ_c) 结果比 Censor‑Hillel & Shachnai 的 O(c log n / Φ_c + c²) 更快，且消息大小从 Õ(n) 降至 Õ(1)。在直径相关的时间上，本文的 Õ(D + √n) 算法优于 Ghaffari & Kuhn 的 Õ(√(nD))，同样保持消息大小为 Õ(1)。实验上（理论上）实现了更高效的传播、MST 与领导选举。

**⚠️ 局限性**

局限性：①对一般图仍存在 Õ(D + √n) 的上界，尚未证明其为最优；②算法需要预先知道弱导线度参数 c 或 Φ_c，或至少在设置阶段估计它们；③在高度不均匀的图中，算法的常数因子和 log n 影响仍可能较大；④实现中需要进行大量的 sketch 计算和随机采样，实际部署时可能需要更复杂的同步与错误控制。

---

## 257. Nearest-Neighbor Radii under Dependent Sampling

**arXiv ID:** 2605.14343 | [PDF](https://arxiv.org/pdf/2605.14343v1)

**作者:** Yuanyuan Gao `[一作]` (University of California, Berkeley), Zhexiao Lin `[通讯]` (University of California, Berkeley)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5056714612)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究在存在强混合依赖的采样下k最近邻半径的几何特性，证明其尺度与独立采样相同，仅因混合速率影响收敛速度。

**💡 创新点**

创新点在于将强混合(多项式和几何)依赖情形下的近邻半径几何尺度不变的分布无关收敛和精确的瞬时矩界定为局部内在维度的函数，而非环境维度。

**🔧 技术方法**

采用三角阵强混合框架、局部质量假设、阻塞技术与自举尾分布推导得到尾/矩上界，下界由局部质量上界给出；实验通过模拟与真实时间序列验证。

**📊 数据集**

使用合成的线性高斯状态空间、隐马尔可夫、RBF核高斯过程以及公开时间序列基准(Time Series Pile、Informer、UCR)等数据集。

**📈 对比分析**

与独立采样、i.i.d.基准、MOMENT预训练模型对比，kNN（原始与PCA）在长期/短期预测和分类任务中保持相似或略逊表现，显示近邻几何在依赖数据中仍有效。

**⚠️ 局限性**

局限在于仅给出点状收敛与尾界，未推导全局一致性；假设compact支撑、局部质量条件；对高度非平稳或极端依赖情形缺乏理论。

---

## 258. Herculean: An Agentic Benchmark for Financial Intelligence

**arXiv ID:** 2605.14355 | [PDF](https://arxiv.org/pdf/2605.14355v1)

**作者:** Xueqing Peng `[一作]` (The Fin AI), Sophia Ananiadou `[通讯]` (University of Manchester)

**通讯引用:** 17552 | [OpenAlex ID](https://openalex.org/A5077976343)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个针对金融专业工作流的评测基准，涵盖交易、对冲、市场洞察与审计四大流程。

**💡 创新点**

将每个工作流实现为基于Model Context Protocol (MCP) 的技能环境，统一接口与工具，能客观比较不同Agent架构。

**🔧 技术方法**

使用MCP、工具调用、检索、计算网络推理等技术，评估多种Agent框架（ReAct, Claude Code, Codex, Hermes, OpenClaw）和四大语言模型。

**📊 数据集**

利用公开的金融数据（Yahoo Finance、SEC EDGAR XBRL、行业新闻等）以及HuggingFace上的TheFinAI/Herculean数据集。

**📈 对比分析**

通过对比各Agent+模型组合在四工作流上的收益率、夏普比率、最大回撤、质量评分、准确率等指标，发现Agent在交易和市场洞察表现相对较好，而对冲和审计显著困难，框架控制对性能影响大。

**⚠️ 局限性**

局限性包括只涵盖大盘美股、三个月固定窗口、未考虑宏观信号与市场摩擦、样本资产有限、评测成本高、LLM判定偏向流畅性、可能存在语言与地域偏差。

---

## 259. Nexus : An Agentic Framework for Time Series Forecasting

**arXiv ID:** 2605.14389 | [PDF](https://arxiv.org/pdf/2605.14389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 260. CrystalReasoner: Reasoning and RL for Property-Conditioned Crystal Structure Generation

**arXiv ID:** 2605.14344 | [PDF](https://arxiv.org/pdf/2605.14344v1)

**作者:** Yuyang Wu `[一作]` (Tsinghua University), Sherry Yang `[通讯]` (New York University)

**通讯引用:** 764 | [OpenAlex ID](https://openalex.org/A5102033082)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 CrystalReasoner，一个端到端的 LLM 框架，能够根据自然语言指令生成满足物理约束的晶体结构。

**💡 创新点**

创新点在于将物理先验作为思考标记引入 LLM，使用 GRPO 强化学习对多目标奖励进行对齐，并为离散与连续性质设计专门的奖励函数，实现自适应推理。

**🔧 技术方法**

主要技术包括 LLM 微调、渐进式思考标记、Group Relative Policy Optimization（GRPO）强化学习、多目标稠密奖励、MLIP 代理模型以及 DFT 验证。

**📊 数据集**

实验使用 CDVAE MP‑20 数据集（来自 Materials Project），并在 Qwen2.5‑3B 预训练模型上进行微调，配合 Robocrystallographer 与 MatterGen 数据。

**📈 对比分析**

与 CrystalTextLLM、PLAID++ Wyckoff 基线对比，在结构、化学有效性、组成、空间群、一致性、能量与 S.U.N. 比例等多项指标上均优于对照组，S.U.N. 比例提升三倍，稳定性和属性条件生成表现更佳。

**⚠️ 局限性**

局限性包括仅在 Qwen2.5‑3B 上实验，缺乏多任务统一模型；需要为每个属性条件单独训练；仅在 MP‑20 子集验证，未测试其他材料族。

---

## 261. Knowledge Beyond Language: Bridging the Gap in Multilingual Machine Unlearning Evaluation

**arXiv ID:** 2605.14404 | [PDF](https://arxiv.org/pdf/2605.14404v1)

**作者:** Kyomin Hwang `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**通讯引用:** 8495 | [OpenAlex ID](https://openalex.org/A5084897975)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多语言机器无学习（MMU）的评估方法，并在此基础上提出了知识分离度量（KSS）和知识持久度量（KPS）两种新指标；

**💡 创新点**

创新点在于首次构造跨语言信息传播与遗忘一致性的评估框架，提出能够同时衡量整体分离度和跨语言一致性的双重度量，弥补了现有单语言评估的局限；

**🔧 技术方法**

采用梯度上升（GA）、梯度上升+梯度下降（GAGDR）、梯度上升+KL最小化（GAKLR）、负偏好优化（NPO）以及剪枝（PRUNE）等无学习技术，并在Llama3.1‑8B多语言模型上进行实验，同时使用概率和生成式语义相等度（SE）作为评分手段；

**📊 数据集**

构建了包含200个合成个人档案、19个属性的英语QA对，随后翻译成10种语言的并行多语言QA数据集，总计3800个QA对；

**📈 对比分析**

在不同忘记比例（p1=1%、p3=3%、p5=5%）下，利用KSS-ROC、KSS-PR和KPS三种指标对上述无学习方法进行对比。结果显示，优化类方法在KSS-ROC上表现优秀，但KSS-PR和KPS随忘记比例增大显著下降；剪枝方法虽在KSS-ROC上高，但KSS-PR低，说明高分散性；总体而言，随着忘记比例上升，跨语言遗忘一致性显著下降；

**⚠️ 局限性**

实验仅使用了8B参数模型，hold‑out语言仅限印欧语系（Afrikaans、Spanish），未检验更大模型或其他语言家族的泛化，且KPS等度量仍可能不足以全面捕捉跨语言信息传播的复杂性。

---

## 262. Dual-Latent Collaborative Decoding for Fidelity-Perception Balanced Image Compression

**arXiv ID:** 2605.14391 | [PDF](https://arxiv.org/pdf/2605.14391v1)

**作者:** Qi Mao `[一作]` (Communication University of China), Siwei Ma `[通讯]` (Peking University)

**通讯引用:** 15927 | [OpenAlex ID](https://openalex.org/A5039832462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了双latent协作解码框架MoDE，利用SQ连续量化和VQ离散量化两种latent分别充当精度专家与感知专家，通过解码器侧的Expert‑Specific Enhancement (ESE) 与 Cross‑Expert Modulation (CEM) 实现选择性跨专家信息交换，实现多比特率下的逼真‑精度平衡；

**💡 创新点**

创新点在于将不同量化范式的latent视为独立解码专家，并在解码器侧通过ESE保留专家专长、CEM实现门控残差跨专家调制，从而避免传统特征融合导致的专家能力削弱，支持同一比特流产生两种不同权衡的重构；

**🔧 技术方法**

使用预训练的ELIC与Fine‑tuned VQGAN作为专家网络，训练仅解码器侧的ESE与CEM模块；引入门控残差调制、密集门控制、专家保留损失、MSE、LPIPS、DISTS等多任务训练；评估指标包括PSNR、LPIPS、DISTS及BD‑Rate；

**📊 数据集**

训练集使用OpenImages；评估集为Kodak、CLIC2020、Tecnick；

**📈 对比分析**

与VVC、ELIC、MS‑ILLM、DC‑VIC、Fine‑tuned VQGAN、DiffEIC、RDEIC、StableCodec等基线在相同双流比特率下对比，采用PSNR、LPIPS、DISTS和BD‑Rate衡量；MoDE‑F在保持PSNR相近的前提下，显著降低LPIPS/DISTS；MoDE‑P在感知基准上提升结构质量，且在全比特率范围内保持低PSNR损失；

**⚠️ 局限性**

局限性包括需预训练的SQ/VQ专家，训练仅限解码侧模块；双流比特率在极低比特率下仍较高；跨专家门控在某些场景可能产生不稳定；缺乏对更高分辨率或视频的扩展验证；

---

## 263. Model Forensics in AI-Native Wireless Networks: Taxonomy, Applications, and Case Study

**arXiv ID:** 2605.14387 | [PDF](https://arxiv.org/pdf/2605.14387v1)

**作者:** Pengyu Chen `[一作]` (Chongqing University), Tao Xiang `[通讯]` (Chongqing University)

**通讯引用:** 42686 | [OpenAlex ID](https://openalex.org/A5014436524)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了AI本地化无线网络中的模型取证，提出模型取证的分类与应用场景，并通过RF指纹案例展示身份验证与恶意功能检测两种取证方法。

**💡 创新点**

首次将模型取证框架与AI无线网络结合，构建统一取证流程，并在RF指纹系统中实现基于水印的身份认证与基于特征空间的后门检测，实现可信身份与恶意功能双重鉴定。

**🔧 技术方法**

使用触发器/对抗水印与参数签名进行模型身份认证；采用t‑SNE投影+SVM进行后门检测；模型采用ResNet‑34深度网络。

**📊 数据集**

WiSig RF指纹数据集（10类设备，每类5000个I/Q样本）。

**📈 对比分析**

对比未水印、带水印、后门、及组合模型的分类准确率、watermark成功率、后门成功率与检测准确率。实验显示水印模型保持约99.82%准确率、watermark成功率>99.95%；后门模型在0.30 poison ratio下后门成功率>94%，对清洁准确率影响≤2%；SVM检测后门的准确率≈98.97%。

**⚠️ 局限性**

仅针对RF指纹两种取证任务，缺乏跨层证据融合和多模态取证；对其他AI无线场景的推广性有限；实验仅在WiSig数据集上，缺乏统一benchmark和实时性评估。

---

## 264. Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of Language-Model Reasoning

**arXiv ID:** 2605.14386 | [PDF](https://arxiv.org/pdf/2605.14386v1)

**作者:** Taebong Kim `[一作]` (VIDRAFT Inc.), Minseo Kim `[通讯]` (VIDRAFT Inc.)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Darwin 框架，利用训练‑free 的演化合并技术在权重空间中重组大语言模型，从而提升推理性能。

**💡 创新点**

创新点包括：① 14 维自适应基因组，实现层级和块级细粒度重组；② MRI‑Trust Fusion，结合诊断重要性与可学习的信任参数指导演化搜索；③ 架构映射器，实现跨架构（Transformer‑Mamba 等）无训练合并。

**🔧 技术方法**

技术手段包括：诊断探针（Model‑Layer Response Importance）评估张量重要性；遗传/演化搜索优化基因组；DARE‑TIES 融合核；可学习的 τ；跨架构张量对齐。

**📊 数据集**

主要使用推理基准数据集：GPQA Diamond、ARC‑Challenge、MMLU、CommonsenseQA、TruthfulQA、HellaSwag、RACE、Natural Questions、TriviaQA。

**📈 对比分析**

与父模型、均值平均、SLERP、TIES、无诊断演化等基线进行比较；Darwin‑27B‑Opus 在 GPQA Diamond 取得 86.9%（第六名），在其他基准均显著优于父模型，提升 1–2pp，甚至超过部分全训练模型。

**⚠️ 局限性**

局限性：只能提升父模型已有能力，无法生成全新技能；需要父模型共享同一预训练基准，跨基准合并受限；演化搜索仍需一定评估成本；在 100B 级别尚未验证；对大规模、跨架构完整验证仍是挑战。

---

## 265. Mitigating Data Scarcity in Psychological Defense Classification with Context-Aware Synthetic Augmentation

**arXiv ID:** 2605.14380 | [PDF](https://arxiv.org/pdf/2605.14380v1)

**作者:** Hoang-Thuy-Duong Vu `[一作]` (VinUniversity), Huy-Hieu Pham `[通讯]` (VinUniversity)

**通讯引用:** 1231 | [OpenAlex ID](https://openalex.org/A5065112274)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过心理学驱动的上下文感知合成数据增强与混合特征融合模型，自动对心理防御机制进行分类。

**💡 创新点**

创新点在于基于DMRS定义的类特定提示、压力源锚定生成以及混合上下文与临床特征的融合，显著提升低资源环境下的分类效果。

**🔧 技术方法**

使用了 Llama‑3‑8B‑Instruct 进行合成数据生成、MentalRoBERTa 进行上下文编码、NLI 进行指标评分、MLP 进行特征编码及 late‑fusion 架构，配合质量控制与数据平衡。

**📊 数据集**

使用的主要数据集为 PsyDefConv（源自 ESConv）及其官方开发集和盲测集。

**📈 对比分析**

与 DMRS Co‑Pilot 基线对比，准确率从 18.01% 提升至 58.26%，宏 F1 从 8.63% 提升至 24.62%，其中 ×2 的合成扩增获得最高宏 F1。

**⚠️ 局限性**

主要局限是标签 7 的“吸收”偏差导致多数类主导、合成数据的临床真实性验证不足，以及单轮语境限制导致对连续对话的时序特征把握不足。

---

## 266. Data-Augmented Game Starts for Accelerating Self-Play Exploration in Imperfect Information Games

**arXiv ID:** 2605.14379 | [PDF](https://arxiv.org/pdf/2605.14379v1)

**作者:** JB Lanier `[一作]` (University of California Irvine), Roy Fox `[通讯]` (University of California Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于离线数据的中间状态重置策略（DAGS），将多智能体自我对弈的起始状态从初始分布改为包含离线演示中访问过的中间状态，以加速在大型不完全信息零和游戏中的探索与均衡收敛。

**💡 创新点**

创新点在于：① 将单智能体的中间状态重置方法扩展到多智能体竞争环境；② 设计可解析的网格控制基准（Kuhn Poker、Goofspiel 和自定义对策游戏）以衡量探索难度；③ 通过多任务观察标记消除起始状态分布改变导致的均衡偏差。

**🔧 技术方法**

使用了：正则化的 PPO 自我对弈（均匀磁铁正则化和 EMA 磁铁正则化）、多任务学习、离线状态重置以及对基准游戏的网格控制改造。

**📊 数据集**

数据集包含 1000 轮从随机策略生成的离线演示（覆盖所有基础动作且完成网格导航），并对基准游戏进行改造得到带控制难度的网格版本。

**📈 对比分析**

方法与传统起始分布自我对弈（β=0）在相同计算预算下对比；实验显示在高探索难度（网格行进 10/20）下，DAGS（β>0）显著降低可利用性，尤其在 Goofspiel 中效果明显；β=0.5 并配合观察标记即可消除均衡偏差，性能与基准相当。

**⚠️ 局限性**

局限性包括：① 起始分布改变可能引入均衡偏差；② 需要离线数据覆盖足够多的战略相关状态；③ 仅在简化的可解析基准上验证，未测试在真实大型游戏（如 StarCraft）中的可扩展性；④ 仍需大量训练资源才能获得较低可利用性。

---

## 267. Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization

**arXiv ID:** 2605.14373 | [PDF](https://arxiv.org/pdf/2605.14373v1)

**作者:** Chen Liang `[一作]` (Yale University), Daniel Rakita `[通讯]` (Yale University)

**通讯引用:** 701 | [OpenAlex ID](https://openalex.org/A5006252401)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了Coherent Coordinate Descent（CoCD），一种基于残差梯度记忆的确定性零阶优化算法；

**💡 创新点**

核心创新在于将历史梯度视为“温暖启动”，利用梯度连续性与有限差分的隐式平滑效应，实现样本高效且低方差的优化；

**🔧 技术方法**

采用确定性循环坐标更新、FIFO梯度缓冲、有限差分梯度估计、隐式平滑与可调动量γ等技术；

**📊 数据集**

在SARCOS回归、MNIST与CIFAR‑10分类（各约10k–20k参数）以及ResNet‑20（≈270k参数）上进行评估；

**📈 对比分析**

与传统Block Cyclic Coordinate Descent（BCCD）及随机零阶方法（SPSA、ZO‑SGD）在相同查询预算下比较，CoCD在样本效率、最终误差/准确率上显著优于BCCD，并在稳定性与壁钟时间上明显优于随机方法；

**⚠️ 局限性**

局限在于模型规模受限，参数量过大时需更高查询预算才能保持梯度估计精度，未来需改进并行与自适应预算策略。

---

## 268. Randomized Atomic Feature Models for Physics-Informed Identification of Dynamic Systems

**arXiv ID:** 2605.14351 | [PDF](https://arxiv.org/pdf/2605.14351v1)

**作者:** Rajiv Singh `[一作]` (MathWorks Inc.), Lennart Ljung `[通讯]` (Linköping University)

**通讯引用:** 53045 | [OpenAlex ID](https://openalex.org/A5078405221)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于随机化稳定原子特征的系统辨识框架（RAF），将物理先验（如稳定性、频率响应掩模、DC增益、时间域误差等）嵌入到一个可求解的凸优化问题中，实现了对线性时不变系统的可解释、稀疏辨识。

**💡 创新点**

创新点在于：① 将传统核方法、原子范数与随机特征相结合，形成“Disk–Bochner”理论，证明正测度在圆盘内可生成PSD核并满足半径缺陷；② 通过采样圆盘内极点实现随机原子特征，既保留模态可解释性又可加入多种工程约束；③ 结合Nevanlinna–Pick与LFT框架，将增益约束与极点约束分离，提供完整的可检验集合成员性；④ 通过稀疏恢复与Vandermonde条件，阐述在随机极点字典下的可行性与局部优化策略。

**🔧 技术方法**

主要技术包括：随机原子特征采样（圆盘内极点），核方法与随机特征的联合表示，凸正则化（ℓ1、群ℓ1）、第二阶锥约束、KYP/Nevanlinna–Pick LMI、稀疏恢复与受限特征值理论、以及多维时域与频域误差约束的SOCP/LMI表示。

**📊 数据集**

示例实验使用一个包含两阶欠阻尼模态的SISO系统，采样低带宽激励、100个样本、30 dB的高斯白噪声；除此之外，还对比了传统PEM、核正则化（KRM）、随机原子范数（ANM）和RAF四种方法。

**📈 对比分析**

比较方法时以输出误差、极点恢复精度、频率响应误差和DC增益误差为指标。结果显示：在仅施加稳定性约束时，KRM在随机特征方法中表现最好；加入极点域约束和稀疏性后，ANM与RAF在模态恢复上相当；在RAF中进一步加入DC增益、BIBO预算与频率掩模后，输出拟合和极点估计均优于其他方法，尤其在数据欠激励时展示出更好的稳健性。

**⚠️ 局限性**

局限性包括：① 需要预先采样足够稠密的极点字典，字典质量直接影响辨识性能；② 高共线性的圆盘Vandermonde矩阵可能导致稀疏解不稳定；③ 仅提供极点候选区域，无法保证精确极点定位，需要后续局部非线性优化；④ 主要针对线性时不变系统，非线性、LPV或闭环系统仍需进一步扩展；⑤ 严格约束可能引入偏差，若先验信息不准确会导致模型偏差。

---

## 269. Distributionally Robust Multi-Task Reinforcement Learning via Adaptive Task Sampling

**arXiv ID:** 2605.14350 | [PDF](https://arxiv.org/pdf/2605.14350v1)

**作者:** Nicholas E. Corrado `[一作]` (University of Wisconsin), Josiah P. Hanna `[通讯]` (University of Wisconsin)

**通讯引用:** 1207 | [OpenAlex ID](https://openalex.org/A5008014974)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Distributionally Robust Adaptive Task Sampling（DRATS）算法，针对多任务强化学习中的数据不平衡问题，通过最小化任务的最大返回差距来动态采样任务；

**💡 创新点**

创新点在于将多任务RL建模为可行性问题，并通过最小化最大返回差距的minimax目标，引入KL正则化的任务分布更新，实现自适应地优先采样尚未达标的任务；

**🔧 技术方法**

使用分布式鲁棒优化（DRO）、KL约束下的镜像上升（mirror ascent）更新任务分布、在线估计参考返回、任务级优势归一化以及标准的PPO/REINFORCE策略更新；

**📊 数据集**

在四个基准上评估：4任务Gridworld、MuJoCo6（6个连续控制任务）、MetaWorld-MT10（10个机械臂任务）和MetaWorld-MT50（50个机械臂任务）；

**📈 对比分析**

与Uniform、Learning Progress、Learning Potential、Hard First（SMT）以及Easy First等基线进行比较，结果显示DRATS在所有基准上都取得更高的累计返回、更好的最差任务性能，并在数据效率上显著优于其他方法；

**⚠️ 局限性**

局限性包括：需要参考返回值，在线估计可能导致早期低估导致数据分配不足；假设任务为离散集合，无法直接处理连续任务空间；

---

## 270. Exemplar Partitioning for Mechanistic Interpretability

**arXiv ID:** 2605.14347 | [PDF](https://arxiv.org/pdf/2605.14347v1)

**作者:** Jessica Rumbelow `[一作]` `[通讯]` (Leap Laboratories), Jessica Rumbelow (Leap Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无监督的 Exemplar Partitioning（EP）方法，用聚类将 LLM 激活空间划分为可解释的区域字典；

**💡 创新点**

创新点在于用观测到的单个示例作为聚类中心，避免平均导致的不连贯中心，并在单次流式传输中完成字典构建，显著降低计算成本；

**🔧 技术方法**

采用居中余弦距离的在线 leader‑clustering，阈值通过预训练数据的距离百分位数校准，构造 Voronoi 区域；

**📊 数据集**

在 Gemma‑2‑2B 及其 instruction‑tuned 版本上使用 Pile 语料构建字典，并在 AxBench、GemmaScope SAE、SAEBench 等公开基准上进行评估；

**📈 对比分析**

与传统稀疏自编码器（SAE）相比，EP 在 AxBench 概念检测任务上达到 0.881 的平均 AUROC，超过 GemmaScope SAE 的 0.755，接近 SAE‑A 的 0.911，仅需 10^3 倍更少的构建计算；在信息保留测评中，EP 的 one‑hot 编码在 97% 的原始激活精度，几乎不丢失线性可解信息；

**⚠️ 局限性**

局限包括对聚类阈值、几何假设（余弦度量）及流式顺序的依赖；对不同层、不同模型的稳定性仍需进一步研究；仅在概念检测和少量实验中验证，尚缺乏更广泛的解释性评估。

---

## 271. AnyBand-Diff: A Unified Remote Sensing Image Generation and Band Repair Framework with Spectral Priors

**arXiv ID:** 2605.14341 | [PDF](https://arxiv.org/pdf/2605.14341v1)

**作者:** Zuopeng Zhao `[一作]` (China University of Mining and Technology), Wenwen Liu `[通讯]` (China University of Mining and Technology)

**通讯引用:** 4004 | [OpenAlex ID](https://openalex.org/A5100343159)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AnyBand-Diff框架，实现遥感图像的生成与波段修复，解决物理一致性与光谱完整性双重挑战。

**💡 创新点**

创新点包括双随机掩码的掩码条件扩散、物理引导采样（PGS）以及多尺度物理损失（MSPL），三者共同保证生成图像在视觉与物理层面高度一致。

**🔧 技术方法**

采用扩散概率模型、条件自适应调制（CAM）、可微物理前向模型、梯度引导采样技术和层级物理损失函数。

**📊 数据集**

使用Pavia University、Washington DC等真实高光谱数据集，以及多光谱/遥感图像合成数据进行训练与评估。

**📈 对比分析**

与SD、ControlNet、DiffusionSat、SpectralDiff、HSI‑Gene等基线比较，AnyBand-Diff在PSNR、SSIM、SAM、RMSE、NDVI/NWI相关系数等指标均显著领先，表现出最优的波段恢复与物理一致性。

**⚠️ 局限性**

局限性包括对极端遮挡（>50%遮罩）仍有残余误差，模型参数量和采样步骤相对较大，对不同传感器的物理模型需手动调整。

---

## 272. zSort: Stable Distribution Sort using Z-Score Partitioning

**arXiv ID:** 2605.14419 | [PDF](https://arxiv.org/pdf/2605.14419v1)

**作者:** Hriday Jain `[一作]` (Pandit Deendayal Energy University), Ashutosh Londhe `[通讯]` (Queen's University Belfast)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出并实现了一种基于z分数分区的稳定排序算法zSort，能够在保持稳定性的同时实现线性时间排序。

**💡 创新点**

创新点在于利用z分数标准化对数据进行分区，保持分区间完全有序并且分区数与输入规模的平方根成比例，从而实现了与关键字宽度无关的常数级分区深度，显著降低了稳定性税。

**🔧 技术方法**

采用z分数映射函数、递归分区、估算均值优化、插入/计数排序切换、缓存友好直方图分区等技术，并通过微体系结构分析优化分支预测和TLB压力。

**📊 数据集**

在64位有符号整数上使用均匀、正态、偏斜、近乎排序和高重复度等5种合成数据集，规模从10⁵到10⁷进行评测。

**📈 对比分析**

与C++标准库稳定/不稳定排序、Pdqsort、Skasort、Spreadsort、LSD Radix等算法对比，zSort在所有分布下均表现最优，均值速度提升约3‑4.5倍，最高达6‑35%相较非稳定算法，IPC达1.44，bad‑spec率19.7%。

**⚠️ 局限性**

局限性包括：对极端分布如高重复或极度偏斜时仍需额外分区开销；对大规模外部/分布式环境尚未验证；并且在多核/GPU并行实现上仍需进一步研究。

---

## 273. Before the Body Moves: Learning Anticipatory Joint Intent for Language-Conditioned Humanoid Control

**arXiv ID:** 2605.14417 | [PDF](https://arxiv.org/pdf/2605.14417v1)

**作者:** Haozhe Jia `[一作]` (Hong Kong University of Science and Technology), Yutao Yue `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 967 | [OpenAlex ID](https://openalex.org/A5052861384)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DAJI框架，利用动态对齐的联合意图（joint‑intent）作为语言生成与闭环控制之间的接口，实现流式人形机器人控制。

**💡 创新点**

创新点在于：①将可执行的、具备前瞻性的联合意图作为中间表示；②通过特权教师（privileged teacher）与学生驱动的循环蒸馏学习可执行意图；③采用流式流匹配（flow‑matching）生成器与自我条件化（scheduled self‑conditioning）提升长期生成的连贯性与稳定性。

**🔧 技术方法**

主要技术包括：多阶段训练（PPO、学生蒸馏、流匹配训练）；扩散动作策略；可执行意图的高斯瓶颈编码；文本特征冻结（Qwen3‑VL‑4B等）以及自回归生成与自我条件化。

**📊 数据集**

使用HumanML3D‑style robot motion数据集进行单指令生成评估，使用BABEL数据集评估长时序流式指令切换生成。

**📈 对比分析**

与基准方法（LangWBC、ECHO、FRoM‑W1、RoboGhost、MotionStreamer、TextOp等）对比，DAJI在语言‑动作对齐（MM‑D、R@1/2/3）、生成质量（FID、Div、MM）以及闭环执行成功率（Succ. %）均取得显著提升，尤其在流式生成的子序列质量与切换连贯性上表现最优。

**⚠️ 局限性**

局限性包括：①训练依赖于特权教师与大规模仿真；②联合意图维度与未来时间窗口的选择需要经验调优；③对硬件执行仍需高频控制资源，且在复杂多模态交互（物体操作、感知反馈）方面尚未覆盖。

---

## 274. Learning to Build the Environment: Self-Evolving Reasoning RL via Verifiable Environment Synthesis

**arXiv ID:** 2605.14392 | [PDF](https://arxiv.org/pdf/2605.14392v1)

**作者:** Yucheng Shi `[一作]` (Tencent HY LLM), Haitao Mi `[通讯]` (Tencent HY LLM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练一个单一的语言模型策略，既生成可执行的推理环境，又在这些环境中求解问题，从而在零数据推理强化学习（RLVR）中构建自适应的可验证训练集。

**💡 创新点**

创新点包括：
- 将自生成任务提升到环境层级，利用“solve–verify asymmetry”实现可重用的可执行奖励源；
- 设计多层验证（L1–L5）与保守的语义自评机制，确保环境实现与提示一致；
- 通过求解器相对难度奖励、novelty gating 与池轮换构建动态、可持续的训练环境池；
- 采用单策略GRPO实现环境生成与求解的联合优化。

**🔧 技术方法**

使用的技术包括：
- 单策略GRPO（角色条件化的强化学习）；
- 环境接口（G_e, Π_e, S_e）及其实现；
- L1–L5多层验证与语义自评；
- 难度校准奖励（基于求解器通过率）；
- 双视图novelty嵌入与相似度阈值；
- 池轮换与可视化的前沿监测。

**📊 数据集**

数据集：
- 仅使用十个种子可执行环境（排序、动态规划、图遍历、数论等），无任何外部问题‑答案对；
- 评估使用Nemo‑Skills基准，包括 AIME 2024/25、HMMT、Beyond‑AIME、Brumo、GPQA Diamond、LiveCodeBench v6 等。

**📈 对比分析**

对比方法：Untrained、R‑Zero、DAPO、RLVE；实验显示：
- 在 Qwen3‑4B‑Instruct、Qwen3‑4B‑Thinking 与 Nemotron‑Cascade‑8B 三类模型上，平均提升约 4–5%（相对增益 7.9% / 3.3% / 3.1%）；
- 对已强大的思考模式模型，RLVE 与 DAPO 反而降低性能，而自生成环境方法仍能提升；
- 训练得分下降但保持前沿，表明环境生成有效维持奖励信号。

**⚠️ 局限性**

局限性：
- 需要可验证且可执行的环境，限制了可应用领域；
- 多层验证与语义自评实现复杂，对资源与工程投入有一定要求；
- 奖励设计依赖求解器当前性能，可能导致难度难以精准把控；
- 仍可能出现模板坍塌或过度依赖种子环境的风险；
- 对真实世界非可执行任务的泛化能力尚未评估。

---

## 275. NodeSynth: Socially Aligned Synthetic Data for AI Evaluation

**arXiv ID:** 2605.14381 | [PDF](https://arxiv.org/pdf/2605.14381v1)

**作者:** Qazi Mamunur Rashid `[一作]` (Google Research), Jamila Smith-Loud `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 NodeSynth 方法，用于基于证据的社会相关合成查询来评估大型语言模型的安全性和知识深度。

**💡 创新点**

创新点在于结合专家校准的细粒度分类器、真实案例的证据检索以及三层层次分类体系，显著提升了合成查询诱导模型失效的能力。

**🔧 技术方法**

使用了 Gemini 2.5 Flash、监督微调的 Taxonomy Generator (TaG)、自动化检索和注释流程，以及 Gemini-3.0 自动评估器等技术。

**📊 数据集**

利用了 2,576 条多语言专家生成的分类实例、公开的医学建议和自残领域基准数据，以及自行生成的 NodeSynth 合成数据集。

**📈 对比分析**

通过与人工撰写、通用合成数据集在医学建议和自残两个敏感域上对四大主流 LLM 进行失败率对比，NodeSynth L3 的失败率比人类写作高 5 倍，超过通用合成数据 8 倍。

**⚠️ 局限性**

局限性包括对 LLM 生成误报的依赖、对极端细分领域覆盖不足，以及对外部搜索结果的可复现性和潜在的伦理审查需求。

---

## 276. Optimal Pattern Detection Tree for Symbolic Rule-Based Classification

**arXiv ID:** 2605.14374 | [PDF](https://arxiv.org/pdf/2605.14374v1)

**作者:** Young-Chae Hong `[一作]` (Amazon), Yangho Chen `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于混合整数规划的可解释规则学习方法——Optimal Pattern Detection Tree（OPDT），用于在二分类数据中发现单一最优模式。

**💡 创新点**

创新点包括：①将规则发现转化为单一路径的优化问题；②引入Branching Structure Constraints（BSC）以编码先验知识和结构约束；③结合warm‑start初始化（BSCCART）和分支优先级提升求解效率。

**🔧 技术方法**

核心技术为混合整数规划（MIP）构建OPDT模型，BSC约束、warm‑start启发式与分支优先级策略。

**📊 数据集**

实验使用UCI机器学习仓库中的15个公开数据集，涵盖医疗与金融领域。

**📈 对比分析**

通过与规则集提取模型（BRS、IDS、IREP、PRISM、Ripper）以及结构约束模型（BSCCART、RSCRULES）比较，OPDT在多数数据集上训练集VI最高，测试集表现也常在前四；BSC、warm‑start与分支优先级显著降低求解时间。

**⚠️ 局限性**

局限性在于需手工设定规则长度与权重w，且对大规模数据的可扩展性仍受限；目前仅能输出单条规则，未来需支持多规则集发现。

---

## 277. Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax

**arXiv ID:** 2605.14366 | [PDF](https://arxiv.org/pdf/2605.14366v1)

**作者:** Zeli Su `[一作]` (Minzu University of China), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 15422 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究低资源语言扩展，提出以语义一致性为目标的对齐框架，使用强化学习结合嵌入相似度奖励实现语义空间对齐。

**💡 创新点**

创新点在于将低资源语言适配视为对齐问题，采用 Group Relative Policy Optimization (GRPO) 通过语义奖励而非传统 token‑level 目标，显著降低对齐税并提升语义质量。

**🔧 技术方法**

技术包括 GRPO 强化学习、嵌入级语义奖励、语言一致性约束、LoRA 参数高效微调以及对比实验评估。

**📊 数据集**

实验使用藏文-中文机器翻译数据、藏文头条生成 CMHG 子集、以及主流语言 CMRC 基准来评估模型。

**📈 对比分析**

与强监督微调 (SFT) 对比，RL 在 MT、HG 任务中保持或提升 BLEU/ROUGE，同时在主流语言基准上保持更高性能；在 LLM 判别的偏好评估中，RL 超越 SFT，显示更高语义质量并显著降低对齐税。

**⚠️ 局限性**

局限性包括训练数据规模有限且领域狭窄，可能导致过拟合；方法仍受预训练偏差影响；目前仅在藏语场景验证，需进一步泛化到其他低资源语言。

---

## 278. LoMETab: Beyond Rank-1 Ensembles for Tabular Deep Learning

**arXiv ID:** 2605.14365 | [PDF](https://arxiv.org/pdf/2605.14365v1)

**作者:** Changryeol Choi `[一作]` (CJ Logistics), Gowun Jeong `[通讯]` (CJ Logistics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LoMETab，一种在表格 MLP 上使用低秩适配器与 Hadamard 乘法的 rank‑r 隐式集成模型，通过在共享权重上叠加身份-残差低秩乘性扰动实现可控多样性；

**💡 创新点**

创新点在于把固定的 rank‑1 乘性掩码替换为可调 rank‑r 的身份-残差 Hadamard 乘法，严格扩大了 r≥2 时的假设空间，并提供了 rank 与初始化尺度两条可调轴；

**🔧 技术方法**

采用低秩适配器、Hadamard 乘法、乘性低秩重参数化、成员级损失监督以及标准 MLP 与 TabM 的嵌入等技术；

**📊 数据集**

在 37 个学术 TabM benchmark 数据集（9 个默认 + 28 TabZilla）以及 77 个细分任务上进行实验；

**📈 对比分析**

与 GBDT、TabM、Attention‑based 等方法进行基准对比，LoMETab 位于领先方法的性能簇，平均排名 2.1±1.3，优于 TabM 的 2.3±1.8，并且通过调节 r/σ_init 能提升多样性与性能；

**⚠️ 局限性**

尚未验证多样性控制在不确定性估计、OOD 检测等下游任务中的实际收益，且需要手动调节 r 与 σ_init，增加了模型选择的复杂度。

---

## 279. RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression

**arXiv ID:** 2605.14359 | [PDF](https://arxiv.org/pdf/2605.14359v1)

**作者:** Zhengjia Zhong `[一作]` (Xiamen University), Hui Li `[通讯]` (Xiamen University)

**通讯引用:** 61171 | [OpenAlex ID](https://openalex.org/A5057824494)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于两层Mixture of Experts（MoE）和双流量化的RQ‑MoE框架，用于输入自适应向量压缩。

**💡 创新点**

通过隐式路由与超维码书实现无额外比特开销的输入依赖代码书适配，并将指令流与量化流分离以实现并行解码。

**🔧 技术方法**

采用残差量化、Mixture of Experts、超维码书、归一化残差损失（NRL）以及双流量化结构等技术。

**📊 数据集**

在Deep1B、BigANN、Facebook SimSearchNet++和Contriever等四大大规模多模态数据集上进行实验。

**📈 对比分析**

与OPQ、RQ、LSQ、UNQ、QINCo等基线对比，RQ‑MoE在重建MSE和检索Recall@k上取得或匹配最强基线的最优结果，并且解码速度提升6~14倍。

**⚠️ 局限性**

仍然需要大规模训练，编码过程保持顺序依赖；在极端稀疏场景下专家维度和门控的选择可能需要进一步调优。

---

## 280. LLM-based Detection of Manipulative Political Narratives

**arXiv ID:** 2605.14354 | [PDF](https://arxiv.org/pdf/2605.14354v1)

**作者:** Sinclair Schneider `[一作]` (University of Bundeswehr Munich), Gabi Dreo Rodosek `[通讯]` (University of Bundeswehr Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于大语言模型的端到端流程，用于在未过滤的大规模社交媒体数据中检测、聚类并标注操纵性政治叙事。

**💡 创新点**

将提示式推理与无监督聚类相结合，利用LLM在嵌入空间中强化操纵意图，首次实现不依赖预设类别的叙事发现。

**🔧 技术方法**

使用Qwen3.5/3-Embedding等LLM进行提示式过滤与嵌入，UMAP降维+HDBSCAN聚类，最终用Qwen3.5提炼叙事标签。

**📊 数据集**

1.26百万条来自X/Twitter、Reddit和Telegram的德英混合帖子，聚焦德国政治人物。

**📈 对比分析**

与传统BERT分类/BERTopic对比，提示式过滤取得F1=0.77（召回0.92），聚类后发现41个叙事，表现出更强的操纵意图捕捉能力。

**⚠️ 局限性**

难以区分协调操纵与个人激进观点，提示工程仍需人工迭代，且仅处理文本，缺乏对图像/视频等多模态内容的分析。

---

## 281. Ideology Prediction of German Political Texts

**arXiv ID:** 2605.14352 | [PDF](https://arxiv.org/pdf/2605.14352v1)

**作者:** Sinclair Schneider `[一作]` (Bundeswehr University Munich), Gabi Dreo Rodosek `[通讯]` (Bundeswehr University Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Transformer的多标签分类模型，结合党派向量映射，将文本政治立场映射到连续的左右光谱[-1,1]上；

**💡 创新点**

创新点在于：①使用党派向量投影将多标签输出转换为连续光谱；②通过对党派向量进行有限范围优化提升跨域泛化；③无需人工标注，可用公开议会与选民指南文本自动构建训练集；

**🔧 技术方法**

技术主要为Transformer预训练模型（DeBERTa‑large、Gemma‑2‑2B、Llama‑3.2 等）+多标签分类 + 向量投影 + 微调 + 线性优化；

**📊 数据集**

使用四大数据集：德国联邦议会（Bundestag）发言、Wahl‑O‑Mat 选民指南、33 份德国报纸文章、535,200 条议员推文；

**📈 对比分析**

在内部评测（Bundestag+Wahl‑O‑Mat）中，DeBERTa‑large 取得 F1=0.84；在推文外域评测中准确率最高 0.864；在报纸外域评测中 Gemma‑2‑2B 取得 MAE=0.172；优化后 MAE 减小约 1.2%；

**⚠️ 局限性**

局限性包括：①基于分类而非推理，缺乏可解释性；②短文本（如短推文）易误判；③单维投影可能无法捕捉复杂党派关系；④依赖德语文化背景，跨语言迁移受限；

---

## 282. SWE-Chain: Benchmarking Coding Agents on Chained Release-Level Package Upgrades

**arXiv ID:** 2605.14415 | [PDF](https://arxiv.org/pdf/2605.14415v1)

**作者:** Man Ho Lam `[一作]` (Chinese University of Hong Kong), Michael R. Lyu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 42004 | [OpenAlex ID](https://openalex.org/A5069596903)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个名为SWE-Chain的软件错误修复链式数据集，并提出了基于链式推理的模型框架；

**💡 创新点**

创新点在于将错误修复过程拆解为多步链式思维，统一标注并公开数据集，促进模型对修复过程的学习与解释；

**🔧 技术方法**

采用Transformer编码-解码架构、CodeBERT及其他预训练语言模型，并结合Chain‑of‑Thought提示技术；

**📊 数据集**

使用来自GitHub开源项目的bug修复代码与说明，构建了约20万条修复实例，并在HuggingFace公开发布；

**📈 对比分析**

与传统单步修复模型对比，链式模型在BLEU/准确率上提升5–10%，实验显示链式推理显著提高了修复质量；

**⚠️ 局限性**

局限性包括数据多样性不足、跨语言覆盖有限、链式推理可解释性与可控性仍需提升，以及模型对长链条依赖时性能下降。

---

## 283. MahaVar: OOD Detection via Class-wise Mahalanobis Distance Variance under Neural Collapse

**arXiv ID:** 2605.14413 | [PDF](https://arxiv.org/pdf/2605.14413v1)

**作者:** Donghwan Kim `[一作]` (Yonsei University), Hyunsoo Yoon `[通讯]` (Yonsei University)

**通讯引用:** 1305 | [OpenAlex ID](https://openalex.org/A5076379562)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后处理式OOD检测方法MahaVar，将Mahalanobis距离与类级距离方差结合，利用神经折叠几何提升检测性能。

**💡 创新点**

创新点在于首次把类间距离方差作为OOD判别信号，并在神经折叠理论下证明ID样本方差更大，理论与实验均验证其有效性。

**🔧 技术方法**

使用Mahalanobis距离、L2归一化、类级距离方差计算以及OpenOOD v1.5 benchmark的评估框架。

**📊 数据集**

在CIFAR‑10、CIFAR‑100和ImageNet上进行实验，配合多种近OOD和远OOD数据集进行评测。

**📈 对比分析**

与Mahalanobis、Mahalanobis++、KNN等主流后处理方法对比，MahaVar在AUROC和FPR@95上均取得SOTA，尤其在ImageNet和CIFAR‑100上表现最为突出。

**⚠️ 局限性**

局限性包括：对维度不足导致ETF不完整的模型（如ViT‑B）方差优势减弱；α参数需在验证集上调优；仅利用二阶统计量，未挖掘更高阶特征信息。

---

## 284. DermAgent: A Self-Reflective Agentic System for Dermatological Image Analysis with Multi-Tool Reasoning and Traceable Decision-Making

**arXiv ID:** 2605.14403 | [PDF](https://arxiv.org/pdf/2605.14403v1)

**作者:** Yize Liu `[一作]` (Monash University), Zongyuan Ge `[通讯]` (Monash University)

**通讯引用:** 12811 | [OpenAlex ID](https://openalex.org/A5005014252)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了DermAgent，一个协同多工具的医疗影像分析代理，用于皮肤病图像诊断、概念标注和临床描述，并提供可追溯的推理链。

**💡 创新点**

采用Plan–Execute–Reflect循环的LLM控制器，双模检索模块（Case RAG和Guideline RAG）对外部证据实时检索并对齐，以及确定性Critic模块进行置信度、覆盖率、冲突门控实现自动自纠。

**🔧 技术方法**

基于多模大语言模型GPT‑4o的控制器，集成PanDerm分类器、MAKE概念注解器、DermoGPT、Qwen3‑VL等视觉问答模型；Case RAG使用DermLIP编码与向量检索；Guideline RAG使用多阶段检索+RRF重排；Critic采用阈值门控逻辑。

**📊 数据集**

使用皮肤病诊断基准HAM10000、SNU；概念标注基准Derm7pt、SkinCon；临床描述基准SkinCAP；以及内部构建的413,210个诊断病例和3,199份指南文档。

**📈 对比分析**

通过与5个MLLM基线和2个医学代理基线在准确率、F1‑Macro、ROUGE‑L等指标上对比，DermAgent在HAM10000准确率61.83%（比GPT‑4o高9.6%）、SNU 32.60%（比GPT‑4o高17.6%）、概念标注最高，临床描述ROUGE‑L 19.48%（比GPT‑4o高3.15%），整体优于所有基线。

**⚠️ 局限性**

仍依赖外部检索库的覆盖与质量，门控阈值需要手工设定；在极长或模糊查询时可能多轮不收敛；对罕见病的识别仍有限；实验仅在公开基准上评估，真实临床环境验证待进一步研究。

---

## 285. Systematic Discovery of Semantic Attacks in Online Map Construction through Conditional Diffusion

**arXiv ID:** 2605.14396 | [PDF](https://arxiv.org/pdf/2605.14396v1)

**作者:** Chenyi Wang `[一作]` (University of Arizona), Ming F. Li `[通讯]` (University of Arizona)

**通讯引用:** 19050 | [OpenAlex ID](https://openalex.org/A5100405681)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于扩散模型潜在空间搜索的语义攻击框架 MIRAGE，用来系统发现能够绕过传统防御并削弱在线 HD 地图构建的安全漏洞。

**💡 创新点**

创新点在于：①将视觉场景的潜在空间作为攻击搜索空间；②利用 ControlNet 保持道路拓扑一致性；③用 CLIP 指导搜索至可现实化的语义变化（如阴影、湿路），从而生成既逼真又能误导模型的攻击样本；④实现了对“边界去除”和“边界注入”两种安全关键攻击目标。

**🔧 技术方法**

使用技术包括：多视角扩散生成器 MagicDrive（Stable Diffusion 1.5 + BEV ControlNet）、潜在空间对比梯度下降（PGD）、CLIP 语义方向损失、VAE 编码/解码、DDIM 采样、以及 MapTR 的梯度反向传播。

**📊 数据集**

实验数据集为 nuScenes 6 视角数据集，并在真实路段（Insta360 360° 录像）进行物理可行性验证。

**📈 对比分析**

与白盒基线 Pixel PGD、AdvPatch 以及三类输入预处理防御（JPEG、Median Filtering、DiffPure）对比，MIRAGE 在边界去除任务中实现 57.7% 检测抑制（仅次于 Pixel PGD 的 72%），在边界注入任务中唯一成功注入假边界（平均 +1.88 条），并在规划层面造成高达 52% 的离线行驶率和 33% 的误停率。防御下，MIRAGE 的效果只被 35–36% 恢复，而 Pixel PGD 的 54–81% 被恢复，表明语义攻击更难被传统防御捕捉。VLM 真实感评估显示 MIRAGE 样本的真实感得分 80–84%，接近干净 nuScenes（96–99%），显著高于 Pixel PGD（28–52%）和 AdvPatch（0–9%）。

**⚠️ 局限性**

局限性包括：仅在 MagicDrive + MapTR 组合上验证；缺乏对多种地图构建模型（如 VectorMapNet）的泛化评估；物理实验仅做了简易 Chalk 重现，未进行大规模多场景、多媒介验证；扩散模型对极端光照或天气变化的鲁棒性仍未知。

---

## 286. Semi-Synchronous Exploration in Dynamic Graphs

**arXiv ID:** 2605.14375 | [PDF](https://arxiv.org/pdf/2605.14375v1)

**作者:** Ashish Saxena `[一作]` (Indian Institute of Technology Ropar), Gokarna Sharma `[通讯]` (Kent State University)

**通讯引用:** 1105 | [OpenAlex ID](https://openalex.org/A5002981812)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在1-间隔连通的动态图（动态端口标签）中研究移动代理的图探索问题，考虑半同步调度器以及对手可在每轮禁用任意子集代理的情况。

**💡 创新点**

提出了对手禁用能力的阈值上界与下界：若对手每轮最多禁用 p = ⌊k-1/(n-2)⌋‑1 个代理，则存在可行算法；若禁用数量 ≥ ⌈k/(n-2)⌉‑1，则探索不可能完成。证明了在此模型下，1-跳可见性和1-跳通信是必要条件，并给出满足这些条件的最优算法。

**🔧 技术方法**

算法利用代理的1-跳可见性和全局通信：通过广播收集邻接信息，重构活动子图，估计最大被禁用数 p，并采用 pipeline 策略将空洞填补或将不活跃节点的代理拉回。移动策略基于最短路径和字典序选择，保证在每轮至少有一条有效移动。

**📊 数据集**

论文为理论工作，没有使用具体数据集；所有证明与复杂度分析均在抽象的动态图模型中完成。

**📈 对比分析**

与已有工作比较：在固定端口标签或不允许禁用的模型中已知探索可行；本工作在更具攻击性的动态端口标签模型下给出精确阈值，并证明在 O(k·D̂) 次移动内完成探索，满足与之前结果一致或更优的移动复杂度。

**⚠️ 局限性**

局限性：算法依赖全局通信且只能在1-跳可见性下实现；不确定是否存在更弱通信（如仅 D̂‑1 端口）即可完成探索；实现时对动态端口重标记的实时控制在实际系统中可能较难满足。

---

## 287. MoRe: Modular Representations for Principled Continual Representation Learning on Squantial Data

**arXiv ID:** 2605.14364 | [PDF](https://arxiv.org/pdf/2605.14364v1)

**作者:** Jiaqi Sun `[一作]` (Carnegie Mellon University), Kun Zhang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 21727 | [OpenAlex ID](https://openalex.org/A5100342355)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为MoRe的模块化连续表示学习框架，利用时间延迟跨层依赖实现可识别的层级表示并实现稳健的持续学习

**💡 创新点**

创新点在于将模块化结构直接嵌入表示本身并给出可识别性保证，提供了基于自适应模块重用、对齐与扩展的两阶段持续学习算法

**🔧 技术方法**

采用可识别的层级模块学习（joint/层级估计）、Hard Concrete门控与对齐矩阵、变分似然与稀疏正则化相结合的学习与适配方法

**📊 数据集**

使用人工合成时间序列（线性与非线性混合）以及真实LLM激活数据（Arxiv、AMPS、Wikipedia）进行验证

**📈 对比分析**

与微调、仅重用、对齐缺失等基线比较，MoRe在保持旧因素的同时保持或提升新因素性能，几乎无灾难性遗忘，在几-shot下的线性探针准确率优于PCA，参数和标注需求更低

**⚠️ 局限性**

受限于需要时间延迟结构假设、对高维非线性混合的可识别性理论尚未完全推广，且在极端类别重叠时模块选择准确率下降

---

## 288. Correctness-Aware Repository Filtering Under Maximum Effective Context Window Constraints

**arXiv ID:** 2605.14362 | [PDF](https://arxiv.org/pdf/2605.14362v1)

**作者:** Shweta Mishra `[一作]` `[通讯]` (Independent Researcher), Shweta Mishra (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种预执行的基于文件大小的启发式过滤框架，以在文本分词前剔除仓库中的无关文件，从而实现LLM开发工具的上下文卫生；

**💡 创新点**

创新点在于：①仅使用OS级文件元数据即可实现无索引、无阻塞、零延迟的过滤；②证明文件大小与token数呈近乎完美线性关系（r=0.997），为简单阈值过滤提供理论依据；③设计了多门HybridFilter，将二进制检测、扩展名屏蔽、大小阈值和关键词密度等门级联，显著降低误判；

**🔧 技术方法**

核心技术包括：文件大小阈值过滤、魔数二进制检测、扩展名黑名单、关键词密度评估、混合门级联、token计数线性模型（k≈0.25 tokens/byte）、tiktoken编码、CodeLlama‑7B‑Instruct推理、内存虚拟文件系统（无磁盘I/O）等；

**📊 数据集**

实验使用了10个真实开源仓库（共22,046文件，五种语言），2,688个文本文件用于token密度验证，以及两仓库（18个任务）进行有限范围的任务评估；

**📈 对比分析**

通过对八种过滤策略的比较，SizeFilter（θ=1 MB）在10个仓库上平均实现79.6%（±13.2%）的token压缩，单文件延迟仅0.30 ms；HybridFilter进一步提升至89.3%（±9.0%）压缩率。任务评估显示，过滤后文件准确率从25%提升至72.2%，错误率从61%降至16.7%；

**⚠️ 局限性**

主要局限包括：大但合法的文件可能被误过滤；魔数检测仅覆盖11种签名，缺乏对更多二进制格式的支持；语义过滤仅使用固定关键词，未结合嵌入；评估规模有限，未在大型SWE‑bench等基准上验证；动态上下文管理与多轮交互未实现；对企业级单体仓库或多模态仓库的外部有效性仍待进一步验证。

---

## 289. When Robots Do the Chores: A Benchmark and Agent for Long-Horizon Household Task Execution

**arXiv ID:** 2605.14504 | [PDF](https://arxiv.org/pdf/2605.14504v1)

**作者:** Zilin Zhu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jing Liu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 LongAct 评估基准和 HoloMind 端到端智能体，用以研究长周期家庭任务中的规划与持续推理。

**💡 创新点**

创新点包括：1）长周期自由文本指令的任务设计与 Improvement Rate 指标；2）DAG 结构的层级规划、双模空间记忆、情景记忆与反思监督的闭环架构。

**🔧 技术方法**

主要技术：视觉‑语言模型（VLM）、DAG 任务分解、高层/低层规划器、3D 多模空间记忆、情景记忆、全局 Critic 反思监督。

**📊 数据集**

使用数据集：基于 AI2-THOR/ProcTHOR 生成的 100+ 房屋布局、300 条长周期任务（平均 9 目标），包含 500+ 人类步骤。

**📈 对比分析**

与纯 VLM 基线对比，HoloMind 将目标完成率从 0.7% 提升至 51–59%，任务成功率从 0% 提升至 16–28%，改进率从负值变为正值；规模提升的相对差距显著缩小。

**⚠️ 局限性**

局限性：执行错误占比 46% 最高，规划与记忆错误仍显著；缺乏主动探知与信息搜寻机制；整体性能仍与人类相差甚远。

---

## 290. Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks

**arXiv ID:** 2605.14503 | [PDF](https://arxiv.org/pdf/2605.14503v1)

**作者:** Qiang Ke `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 66273 | [OpenAlex ID](https://openalex.org/A5115602103)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建模块化的 Code RAG 测试平台，对检索增强生成（RAG）在软件工程三大任务（代码生成、代码摘要、代码修复）中的各个组件进行系统化、逐层实验与比较。

**💡 创新点**

创新点在于：①首次进行跨组件、跨模型的细粒度评估；②发现检索器（尤其是 BM25）在代码场景中往往决定整体性能，生成器影响有限；③提出了基于实验规则的自适应 RAG 框架，能够根据任务特征自动推荐最佳管道配置。

**🔧 技术方法**

技术包括：多范式检索（BM25、E5、Granite、GTE、BGE、Jina、SFR、Hybrid RRF）；查询处理（简化、扩展、HyDE）；上下文精炼（重排序器 BGE、Qwen；压缩 LLMLingua-2、Zero‑shot Recomp）；多种生成模型（GPT‑4o、DeepSeek‑V3、Llama‑3.3‑70B、Qwen‑2.5‑32B‑Instr、Phi‑4‑14B、Qwen‑2.5‑Coder‑32B‑Instr）；实验平台采用 Faiss、Ollama、SiliconFlow API。

**📊 数据集**

使用了 APPS（代码生成）、CodeXGLUE‑Python（代码摘要）、DebugBench（代码修复）三大公开基准；检索语料库由 Stack Overflow、GitHub、LeetCode、Python API、CodeSearchNet、Hugging Face 约 23 万条去重后文档构成。

**📈 对比分析**

评估指标包括：代码生成/修复的 Weighted Pass@1、CodeBLEU；代码摘要的 Sim_Emb；使用 k=3、5 的检索深度对比；与零检索和 oracle 上下文基准对照。结果显示：①检索阶段提升显著，BM25 在大多数任务上优于 dense 或 hybrid；②重排序往往导致性能下降；③压缩在特定 token 上下文预算下可提升功能正确率；④生成器决定性能上限，DeepSeek‑V3、GPT‑4o 等 frontier 模型在复杂任务中表现最佳。

**⚠️ 局限性**

局限性包括：①实验仅覆盖 Python 语言和三大任务，难以直接推广到其他语言/任务；②使用的模型和检索方法仅为当前 snapshot，未来技术迭代可能改变结论；③指标侧重功能正确性，未充分评估可读性、效率等软件质量维度；④自适应框架的规则基于实验数据，可能在未见任务中出现迁移不佳。

---

## 291. Quantifying Cyber-Vulnerability in Power Electronics Systems via an Impedance-Based Attack Reachable Domain

**arXiv ID:** 2605.14502 | [PDF](https://arxiv.org/pdf/2605.14502v1)

**作者:** Hongwei Zhen `[一作]`, Mingyang Sun `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对电力电子系统的网络攻击，提出基于阻抗的攻击可达域(ARD)框架和攻击渗透指数(API)评估节点级网络可攻击性。

**💡 创新点**

创新点在于将攻击行为映射到阻抗重塑后的特征值漂移，并提出连续化的攻击渗透指数，从而量化可攻破的安全裕度。

**🔧 技术方法**

利用灰盒阻抗辨识、物理信息感知的可微逼近模型、向量拟合和自动微分进行敏感度分析与边界搜索。

**📊 数据集**

使用4节点系统与改造后的IEEE 39节点测试网络，并通过实验生成阻抗数据训练模型。

**📈 对比分析**

与传统网格强度指标(gSCR, MISCR, IMR)对比，API能揭示非单调的脆弱节点，证明在协同跨层攻击下性能显著提升。

**⚠️ 局限性**

局限在于仅考虑小信号稳定性、基于理论模型的阻抗辨识误差，以及未验证大信号与硬件平台的鲁棒性。

---

## 292. Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty

**arXiv ID:** 2605.14494 | [PDF](https://arxiv.org/pdf/2605.14494v1)

**作者:** Tianjue Lin `[一作]` (Nanyang Technological University), Jie Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 78587 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在离散不确定性下两阶段鲁棒优化的场景缩减问题，并提出了基于问题驱动的顺序前瞻启发式 PRISE 以及其神经网络代理 NeurPRISE。

**💡 创新点**

创新点在于将场景缩减视为目标感知的序列决策，通过模仿 PRISE 的边际收益来训练 GNN‑Transformer 代理，实现了大幅加速且保持高质量。

**🔧 技术方法**

采用 GNN 对单个场景进行图卷积编码，Transformer 关注跨场景注意力，使用模仿学习和基于增益的 KL 损失对模型进行训练。

**📊 数据集**

在三类 2RO 问题（选择、顶点覆盖、容量设施位置）上使用人工生成的实例进行实验。

**📈 对比分析**

与 Exact、随机、K‑means、MaxSum、SOR、Neur2RO 等基线比较，NeurPRISE 在保留少量场景时实现与 PRISE 相近的 regret，且在速度上提升 7–200 倍，且在更大规模和分布偏移的测试中表现稳健。

**⚠️ 局限性**

局限性包括：需要先行执行昂贵的 PRISE 生成标签；对连续不确定性或极端分布偏移的泛化尚未验证；以及缺乏对算法收敛性的理论保证。

---

## 293. A Novel Schur-Decomposition-Based Weight Projection Method for Stable State-Space Neural-Network Architectures

**arXiv ID:** 2605.14489 | [PDF](https://arxiv.org/pdf/2605.14489v1)

**作者:** Sergio Vanegas `[一作]` (LUT University), Fredy Ruiz `[通讯]` (Politecnico di Milano)

**通讯引用:** 1252 | [OpenAlex ID](https://openalex.org/A5090940285)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于实数Schur分解的可反向传播的权重投影方法，保证离散LTI状态空间层的渐近稳定性。

**💡 创新点**

通过快速截断的Schur矩阵投影代替昂贵的双层优化，并提供预因子化参数化，消除了超参数调优和过度参数化的问题。

**🔧 技术方法**

结合实数Schur分解、Ω-稳定矩阵投影、权重约束、预因子化方案以及JAX实现的高效求解。

**📊 数据集**

在合成的随机LTI系统以及四个真实系统（Silverbox、CED、EMPS、工业机器人和Fine-Steering Mirror）上进行评测。

**📈 对比分析**

与Simba、权重正则化及传统系统辨识方法对比，新方法在NMSE、训练时间和收敛速率上与Simba持平或优于正则化方法。

**⚠️ 局限性**

仅适用于Schur半径≤1的系统，GPU实现尚不成熟，且无法直接控制特征值的上界，未来需进一步扩展。

---

## 294. Head Forcing: Long Autoregressive Video Generation via Head Heterogeneity

**arXiv ID:** 2605.14487 | [PDF](https://arxiv.org/pdf/2605.14487v1)

**作者:** Jiahao Tian `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**通讯引用:** 26829 | [OpenAlex ID](https://openalex.org/A5100458183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Head Forcing框架，利用注意力头的功能分型，针对本地、锚点、记忆三类头实现头部级KV缓存分配和RoPE重编码，实现训练无损的分钟级长视频生成。

**💡 创新点**

首次将AR视频Diffusion中的注意力头按功能分为本地、锚点与记忆三类，并为每类设计专属的KV缓存策略、层次化记忆系统和头部级RoPE重编码，解决长期漂移与上下文遗失问题。

**🔧 技术方法**

基于预训练的Self‑Forcing AR视频DiT，结合头部分型、KV缓存裁剪、快慢层次化记忆、动态情节新颖度采样、提示引导压缩、头部级FlashAttention与Triton内核融合以及RoPE重编码技术。

**📊 数据集**

使用MovieGenBench采样的100条单提示和50条多提示序列，并在VBench‑Long数据集上进行评估。

**📈 对比分析**

与训练型（Self‑Forcing、CausVid、Rolling Forcing、LongLive）和训练自由型（Deep‑Forcing、Infinity‑RoPE）基线在30s/60s单提示和多提示60s场景下对比，Head Forcing在VBench‑Long的质量、连贯性、美学等指标均优于或与基线相当，且吞吐量基本持平。

**⚠️ 局限性**

仍依赖预训练模型的架构，头部分型与缓存策略需要先行分析；极长序列或多模态交互时层次记忆容量与新颖度阈值可能需进一步调优；缺乏对不同分辨率或帧率的泛化评估。

---

## 295. When Retrieval Hurts Code Completion: A Diagnostic Study of Stale Repository Context

**arXiv ID:** 2605.14478 | [PDF](https://arxiv.org/pdf/2605.14478v1)

**作者:** Haojun Weng `[一作]` (Independent Researcher), Xinwei Lv `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究检索增强式代码生成中，来自旧提交的代码片段会诱导模型产生与当前仓库不兼容的调用。

**💡 创新点**

首次将检索上下文的时间有效性作为独立诊断变量，在同一任务条件下系统评估旧与当前上下文对生成结果的影响。

**🔧 技术方法**

采用基于OpenAI兼容接口的ChatGPT和Qwen两大模型，使用静态正则表达式或acles进行判别，并通过手工构建的检索条件进行对比。

**📊 数据集**

收集了5个Python开源仓库（click、flask、requests、rich、httpx）中的17个签名变更样本，包含父提交（旧）和子提交（新）两种上下文。

**📈 对比分析**

对比“当前仅检索”“旧仅检索”“不检索”“混合检索”四种条件，发现旧仅检索使旧引用率从0%跃升至88.2%/76.5%，而混合检索显著降低错误率；两模型在大多数样本上表现一致。

**⚠️ 局限性**

样本规模有限、仅覆盖签名漂移、只使用静态正则oracle、未验证其他语言或更大范围变更，且模型对不同检索顺序的敏感性在样本内差异不显著，需进一步扩大研究范围。

---

## 296. Proof Nets for PiL (Full Version)

**arXiv ID:** 2605.14476 | [PDF](https://arxiv.org/pdf/2605.14476v1)

**作者:** Matteo Acclavio `[一作]` (University of Southern Denmark), Giulia Manara `[通讯]` (University of Southern Denmark)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5114658815)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

定义了 PiL 的冲突网（conflict nets）和切片网（slice nets），并给出了它们的正确性判定、序列化与证明翻译，证明了这些网对 PiL 推导提供了规范化表示。

**💡 创新点**

创新点在于：①将冲突网框架扩展到包含非交换、非结合连词和名量词的第一阶线性逻辑；②提出一种新的可压缩性（coalescence）判定与序列化算法；③证明冲突网与切片网的局部与强规范性，并细致分析等价推导的规则置换。

**🔧 技术方法**

使用的技术包括：冲突网（concord–conflict trees）与可压缩性判定；可压平化（flattening）规则；代换与双化器（dualizer）机制；以及证明翻译和序列化步骤。

**📊 数据集**

本研究属于理论性工作，未使用任何数据集。

**📈 对比分析**

通过理论证明展示正确性、完整性，并证明判定在多项式时间（O(n^5)）内完成；未给出实验性能对比。

**⚠️ 局限性**

局限性：①未研究 cut 归约的效率；②当前框架仅适用于 PiL，难以直接推广到更一般的第一阶线性逻辑；③名量词的求值机制仍需进一步简化和优化。

---

## 297. FuzzAgent: Multi-Agent System for Evolutionary Library Fuzzing

**arXiv ID:** 2605.14431 | [PDF](https://arxiv.org/pdf/2605.14431v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 298. From Table to Cell: Attention for Better Reasoning with TABALIGN

**arXiv ID:** 2605.14465 | [PDF](https://arxiv.org/pdf/2605.14465v1)

**作者:** Tung Sum Thomas Kwok `[一作]` (University of California), Zhijiang Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2364 | [OpenAlex ID](https://openalex.org/A5101206779)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种通过掩蔽扩散语言模型规划与基于注意力验证器的表格推理框架，显式实现规划与执行的单元级细胞对齐；

**💡 创新点**

核心创新在于：①使用双向扩散式掩蔽模型（DLM）进行全局一致的规划；②构造基于细胞注意力重叠的验证器，实现过程级监督；③利用人类标注的细胞重要性数据训练验证器，提升监督信号质量；

**🔧 技术方法**

技术手段包括：掩蔽扩散语言模型（DLM）作为规划器；基于每步注意力重叠的轻量级验证器；多模态文本/图像工具调用的执行器；两阶段训练与校准的注意力奖励；

**📊 数据集**

评估使用八个表格推理基准（WTQ、MMQA、TabMWP、TAT-QA、HiTab、InfoTabs、TabFact、FeTaQA），其中前六项为准确率评测，FeTaQA采用BLEU；

**📈 对比分析**

与现有8B级别的表格/多模态LLM基线（如Qwen3‑VL‑8B、TableDART、TATTOO）进行对比，平均准确率提升15.76个百分点；在大多数单项指标上位居榜首；计划器与验证器的消融实验表明DLM规划器和注意力监督分别贡献约2.9和4.8个百分点；

**⚠️ 局限性**

局限性包括：67% 的错误路径在第一步即绕过规划，导致计划遵循度低；在FeTaQA任务中由于BLEU惩罚导致与TableLlama‑7B的差距被高估；扩散规划计算成本高，推理速度受限；

---

## 299. Collaborative Yet Personalized Policy Training: Single-Timescale Federated Actor-Critic

**arXiv ID:** 2605.14423 | [PDF](https://arxiv.org/pdf/2605.14423v1)

**作者:** Leo Muxing Wang `[一作]` (Northeastern University), Lili Su `[通讯]` (Northeastern University)

**通讯引用:** 2365 | [OpenAlex ID](https://openalex.org/A5101541239)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种在异质环境下的个性化联邦 actor-critic 框架 pFedAC，使用共享线性子空间与本地策略头实现个性化策略学习。

**💡 创新点**

首次在多智能体联邦 RL 中将共享子空间与本地头结合，提出单尺度更新的收敛证明，证明了对代理数的线性加速，并通过 QR 投影与条件混合技术处理 Markov 依赖。

**🔧 技术方法**

使用单尺度 actor‑critic 算法、线性函数逼近、QR 分解与投影、条件混合（mixing）分析、TD(L) 更新以及 PPO 在联邦设置的实例化。

**📊 数据集**

自定义联邦基准（六个客户端共享相同动力学但各自不同动作映射，分为 grouped 与 6‑unique 两种设置）以及单机 Hopper 与 OOD 版本用于转移学习。

**📈 对比分析**

与单机 PPO（Single PPO）和联邦平均 PPO（FedAvg PPO）对比，FedPer PPO 在 6‑unique 设置下相较 Single PPO 提升 2.63 倍，FedAvg 3.78 倍；在转移实验中冻结共享 trunk 的性能比从头训练快 5.42 倍（Vanilla）与 3.5 倍（OOD）。

**⚠️ 局限性**

收敛率对折扣因子 (1-γ) 的依赖仍较差，近无折扣情形（γ→1）尚未优化；分析假设较强（如共享子空间、Markov 采样、投影约束）可能限制实用性。

---

## 300. Branch-width of represented matroids in matrix multiplication time

**arXiv ID:** 2605.14428 | [PDF](https://arxiv.org/pdf/2605.14428v1)

**作者:** Mujin Choi `[一作]` (Institute for Basic Science), Sang-il Oum `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种算法，该算法可以在O_k,𝔽(n^2)时间内找到给定n元素的matroid M的branch-decomposition，宽度不超过k，或者确认M的branch-width超过k。

**💡 创新点**

该算法的创新点在于其时间复杂度显著低于之前的算法，所有先前的算法至少需要Ω(n^3)的时间。

**🔧 技术方法**

使用了矩阵乘法的快速算法和动态规划技术，特别是通过将问题转化为图的rank-width问题来实现。

**📊 数据集**

使用了n×n的矩阵表示的matroid M，且在某些情况下，输入矩阵可以是标准形式或一般形式。

**📈 对比分析**

与之前的算法进行比较，性能上显著提高，特别是在处理标准形式的输入时，时间复杂度为O_k,𝔽(n^2)，而之前的算法为O_k,𝔽(n^3)。

**⚠️ 局限性**

算法的局限性在于，对于一般形式的输入矩阵，仍然需要O(n^ω)的时间来找到标准形式，且在某些情况下，branch-width的近似计算可能会受到限制。

---

## 301. Contestable Multi-Agent Debate with Arena-based Argumentative Computation for Multimedia Verification

**arXiv ID:** 2605.14495 | [PDF](https://arxiv.org/pdf/2605.14495v1)

**作者:** Truong Thanh Hung Nguyen `[一作]` (University of New Brunswick), Hung Cao `[通讯]` (University of New Brunswick)

**通讯引用:** 360 | [OpenAlex ID](https://openalex.org/A5088383217)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于多模态大型语言模型、外部验证工具和 Arena‑based Quantitative Bipolar Argumentation (A‑QBAF) 的可争议多代理框架，用于生成可编辑、透明的多媒体真实性报告。

**💡 创新点**

创新点在于将真实性判定拆分为六个声明中心的子任务，使用结构化支持/攻击论点构造稀疏局部图，并引入选择性冲突解决与不确定性上升机制，使得报告既可争议又计算高效。

**🔧 技术方法**

使用多模态 LLM（如 Gemini、GPT、Claude）、外部工具（逆向图像搜索、元数据检查、OCR/ASR、事实核查数据库）以及 Arena‑based Quantitative Bipolar Argumentation (A‑QBAF) 进行论证推理。

**📊 数据集**

主要数据集为 ICMR 2026 大赛多媒体验证数据集（包含图像、视频及相关元信息），以及示例 ID01 用于演示。

**📈 对比分析**

由于论文主要提供示例性对比，未给出系统化的定量评估；在示例 ID01 上，框架成功将“who”声明的分数降至 0.18，表明能有效识别反驳证据，未给出与基线的数值比较。

**⚠️ 局限性**

局限包括对大型语言模型的依赖导致计算成本上升、缺乏大规模量化评估、对外部工具准确性的依赖、以及在极端多模态噪声下可能出现误判。

---

## 302. Exploiting LLM Agent Supply Chains via Payload-less Skills

**arXiv ID:** 2605.14460 | [PDF](https://arxiv.org/pdf/2605.14460v1)

**作者:** Xinyu Liu `[一作]` (Zhejiang University), Xin Xia `[通讯]` (Zhejiang University)

**通讯引用:** 21619 | [OpenAlex ID](https://openalex.org/A5006669765)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Semantic Compliance Hijacking (SCH)——一种通过自然语言合规说明诱导 LLM 代理在运行时生成并执行恶意代码的无载荷攻击。

**💡 创新点**

创新点在于消除传统代码注入，利用语义隐写和多技能自动优化 (MS-AO) 以实现对 agent 技能链的隐蔽劫持，并展示了高成功率与零检测率。

**🔧 技术方法**

主要技术包括 LLM 代理推理、自然语言语义包装、语义隐写、基于回溯的 MS-AO 反馈循环，以及静态语法扫描与 LLM Guard 等安全门检测。

**📊 数据集**

实验使用了 GPT‑5.4 mini、GLM‑5、MiniMax‑M2.7 三大基础模型，OpenClaw、Claude Code、Codex 三大代理框架，并结合 DS‑1000、BigCodeBench、SecurityEval、AgentBench、InterCode 等共 600 条上下文化测试任务。

**📈 对比分析**

通过与 SkillJect、DDIPE 等现有攻击基线以及 SkillScan、LLM Guard 两大检测器对比，SCH 在所有配置下完成泄露率高达 36–63%、RCE 成功率 31–64%，且检测率为 0%；MS‑AO 在 5 轮迭代后进一步提升攻击成功率并保持零检测。

**⚠️ 局限性**

局限性包括对特定模型/框架版本的依赖、实验环境为 Docker 沙箱而非真实企业网络、未验证恶意技能在真实用户采集与安装过程中的可行性，以及对极端开发场景的代表性不足。

---

## 303. When Answers Stray from Questions: Hallucination Detection via Question-Answer Orthogonal Decomposition

**arXiv ID:** 2605.14449 | [PDF](https://arxiv.org/pdf/2605.14449v1)

**作者:** Siyang Yao `[一作]` (Shanghai Jiao Tong University), Yubin Xia `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9052 | [OpenAlex ID](https://openalex.org/A5026023746)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种单前向的幻觉检测框架 QAoD，利用问题-答案正交分解提取鲁棒特征。

**💡 创新点**

创新点在于将问题方向投影消除域敏感成分，并结合 Fisher 判别的层与神经元筛选，既保证检测精度又保持跨域泛化。

**🔧 技术方法**

采用正交投影、Fisher 判别层/神经元选择、单前向 MLP、CKA 对齐分析等技术实现。

**📊 数据集**

使用了 TriviaQA、SQuAD、NQ 和 BioASQ 四大 QA 基准，并在 gemma‑2‑2b、Llama‑2‑7B‑chat、Qwen3‑14B 与 Qwen3‑30B‑A3B 四个 LLM 上评测。

**📈 对比分析**

与多种黑盒采样和白盒探测基线对比，ℋ_Q⊕V_⊥ 在所有模型-数据集上取得最佳 in‑domain AUROC，ℋ_V_⊥ 在零shot BioASQ 上比最强白盒提升 ≥10 AUROC 点，且检测开销仅约 2% 的生成成本。

**⚠️ 局限性**

局限在于需要访问内部隐藏状态，无法直接应用于闭源 API；在数学推理等非问答任务的表现尚未验证。

---

## 304. Prompting Policies for Multi-step Reasoning and Tool-Use in Black-box LLMs with Iterative Distillation of Experience

**arXiv ID:** 2605.14443 | [PDF](https://arxiv.org/pdf/2605.14443v1)

**作者:** Krishna Sayana `[一作]` (Google), Ambarish Jash `[通讯]` (Google)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5043510378)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于强化学习的“Prompter Policy”框架，利用对比经验缓冲区对大规模冻结LLM的提示进行自适应优化，从而提升多步推理和工具使用任务的性能。

**💡 创新点**

创新点在于将提示生成建模为MDP，结合标量奖励与文本批判的对比经验缓冲区，实现一次策略权重即可蕴含多轮自我反思，并支持跨模型迁移。

**🔧 技术方法**

采用强化学习（policy gradient + KL正则）、对比经验缓冲区、Self‑Imitation Learning、AlphaEvolve式存储、外部反馈模型生成批判，以及TPU分布式训练等技术。

**📊 数据集**

使用 Big Bench Extra Hard（BBEH）中的推理任务（如 Dyck Languages、Web of Lies）以及 τ‑bench 的工具使用任务（Retail、Airline）作为评测数据集。

**📈 对比分析**

与零射击基准和 GEPA 进化搜索比较，逻辑推理任务从约55% 提升至90%（+35%），工具使用任务从约74% 提升至91%（+17%），样本效率提升约 2.4 倍。

**⚠️ 局限性**

局限在于只在有限任务集上训练，难以保证对全新任务的零射击迁移；经验缓冲区策略为基础采样，缺乏更高效的管理；对不同规模模型的适配受限于基础推理能力。

---

## 305. Synthesizing POMDP Policies: Sampling Meets Model-checking via Learning

**arXiv ID:** 2605.14440 | [PDF](https://arxiv.org/pdf/2605.14440v1)

**作者:** Debraj Chakraborty `[一作]` (Nanyang Technological University), Jean-François Raskin `[通讯]` (Université Libre de Bruxelles)

**通讯引用:** 5441 | [OpenAlex ID](https://openalex.org/A5050196522)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一个结合采样、自动机学习与模型检测的框架，用于在偏观测马尔可夫决策过程（POMDP）中合成满足阈值安全约束的有限状态控制器。

**💡 创新点**

创新点在于：① 将 L* 学习算法与采样动作查询（action oracle）以及模型检测等价查询（equivalence oracle）结合，形成可实现的 Counterexample‑Guided Policy Learning；② 在采样产生的策略若为正规（可由有限状态机表示）时提供相对完备性保证；③ 通过模型检测得到的有限 counterexample 直接用于改进学习表，从而在无限期安全问题上获得正式保证。

**🔧 技术方法**

主要技术包括：L*（Mealy机）自动机学习；基于采样的在线规划（如 POMCP、SARSOP）作为动作查询 oracle；概率模型检测器（PRISM、Modest）作为等价查询 oracle；以及基于 R‑S 的 counterexample 处理和表闭合操作。

**📊 数据集**

实验使用了多种公开 POMDP 基准：Grid‑World（5×5至20×20）、Hallway、Cards（加/去除卡牌）、Reach‑Avoid（Refuel、Rocks、Cheese‑Maze）等。每个基准均包含若干随机实例，使用 PRISM/DRN 格式定义模型。

**📈 对比分析**

与当前主流正式方法（Storm 的 belief‑exploration、POMCPI 的 iterative policy synthesis）相比，实验显示：① 在多数实例中我们的框架能够在 60–600 秒内合成满足阈值的 FSC，且往往比对手更快；② 对于需要较大状态空间或深层记忆的实例（如 Hallway‑simple‑50、Cards‑removed‑5）传统方法会超时或产生失败，而本文框架在几分钟内给出正确答案；③ 当使用采样得到的策略作为动作 oracle 时，整体性能得到进一步提升。

**⚠️ 局限性**

局限性包括：① 由于 POMDP 合成问题不可判定，框架在动作 oracle 没有正规策略时可能永不终止；② 对大观测字母表的学习效率受限，学习表规模随字母表增大而指数增长；③ 目前仅处理安全（或有限期到达）目标，对一般 PCTL 或无穷期可达目标的扩展仍在研究中；④ 依赖采样器的质量，若采样不够精确会影响最终控制器的最优性。

---

## 306. Intelligence Impact Quotient (IIQ): A Framework for Measuring Organizational AI Impact

**arXiv ID:** 2605.14455 | [PDF](https://arxiv.org/pdf/2605.14455v1)

**作者:** Chandan Rajah `[一作]` (Inception, G42), Larry Murray `[通讯]` (Inception, G42)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并提出 Intelligence Impact Quotient（IIQ）——一种多维度、时序敏感的用户/组织级AI嵌入度量方法，综合新颖性加权 token 库存、使用频率、优雅期递归门、组织杠杆、任务复杂度与自治等因素；

**💡 创新点**

在传统 token/访问计数基础上引入语义新颖性加权、时间衰减、对数频率、自治回合/运行时长等多维度乘法组合，提供归一化 IIQ 指数和估算层，实现对 AI 使用深度与影响的更细粒度区分；

**🔧 技术方法**

利用 token 计数、编辑距离/嵌入相似度衡量新颖性、指数衰减函数、对数频率函数、自治回合/运行时长加权、乘法公式与归一化对数映射；

**📊 数据集**

采用合成模拟数据（30 天交互轨迹）以及参考已有研究（Anthropic Economic Index、GDPval 等），并未使用公开真实组织的具体数据集；

**📈 对比分析**

与单纯 token 计数或访问日志等基准指标对比，展示 IIQ 能区分高频低杠杆使用、语义重复提示与高后果自主工作；在合成场景中通过四种用户配置的 IA I/IIQ 变化，证明指标具有区分度；

**⚠️ 局限性**

需对杠杆、复杂度、新颖性、自治等参数进行经验校准；组织层级映射粗糙，矩阵式或跨功能团队难以准确；新颖性测量可能受限于编辑距离或嵌入相似度的准确性；自治观测受限于回合/运行时长记录；财务/效率估算仅为近似代理；组织聚合若仅取均值可能掩盖采用分布不均问题。

---

## 307. Physics-Based iOCT Sonification for Real-time Interaction Awareness in Subretinal Injection

**arXiv ID:** 2605.14500 | [PDF](https://arxiv.org/pdf/2605.14500v1)

**作者:** Luis D. Reyes Vargas `[一作]` (Technische Universitaet Munchen), Sasan Matinfar `[通讯]` (Technische Universitaet Munchen)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理模型的实时 iOCT 声学化框架，用音频反馈辅助眼科微创子视网膜注射手术；

**💡 创新点**

创新点在于将实时分割的视网膜层与物理质量-弹簧-阻尼模型耦合，直接从 iOCT 动态捕获器械与组织相互作用与变形信号，并通过音频传递；

**🔧 技术方法**

使用 U‑Net 进行视网膜层与针尖分割、基于三次样条的时间平滑、二维质量‑弹簧‑阻尼声学模型、实时音频合成与置信度调制；

**📊 数据集**

实验数据包含公开猪眼体外注射序列、合成注射模拟序列以及 34 名受试者（30 名新手 + 4 名专家）的交互试验；

**📈 对比分析**

与传统基于参数映射的基线方法比较，实验中总事件识别准确率提升至 83.4%（vs. 60.6%），尤其在裂膜（blow‑up）检测上显著提高；

**⚠️ 局限性**

局限性包括受试者数量有限、数据来源单一（猪眼体外与合成），音频可感知粗糙度需进一步优化，以及需在临床真实环境中验证性能。

---

## 308. Cross-Linguistic Transcription and Phonological Representation in the Huìtóngguǎnxì Huáyíyìyǔ

**arXiv ID:** 2605.14480 | [PDF](https://arxiv.org/pdf/2605.14480v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 309. GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations

**arXiv ID:** 2605.14498 | [PDF](https://arxiv.org/pdf/2605.14498v1)

**作者:** Jingbo Yang `[一作]` (University Of California Santa Barbara), Evgeniy Gabrilovich `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向多方对话的记忆检索基准 GroupMemBench 并评估现有记忆系统。

**💡 创新点**

首次把群组动态、说话者归属与受众适配三个维度纳入基准，并通过图引导生成与对抗式查询实现可测量的难度。

**🔧 技术方法**

使用图结构生成、生成式语言模型（GPT‑5）、检索增强生成与多种记忆系统（MemGPT、Mem0、A‑Mem、HippoRAG、Hindsight）以及 BM25。

**📊 数据集**

基于四个行业领域（技术、医疗、制造、金融）的合成多方对话，引用 HuggingFace 上的 GroupMemBench 数据集。

**📈 对比分析**

与 BM25、GraphRAG 等基准对比，发现最强系统平均准确率仅 46%，BM25 仍能匹配或超越多数记忆系统，显示结构与说话者信息被侵蚀。

**⚠️ 局限性**

局限在于仅使用合成数据、评估仅基于问答准确率，未深入探究长期动态记忆与实际部署的鲁棒性。

---

## 310. Reduce the Artifacts Bias for More Generalizable AI-Generated Image Detection

**arXiv ID:** 2605.14486 | [PDF](https://arxiv.org/pdf/2605.14486v1)

**作者:** Yiheng Li `[一作]` (University of Chinese Academy of Sciences), Wenhao Wang `[通讯]` (Vast Intelligence Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了解决 AI 生成图像检测中 Artifact Bias 的方法：先使用 SRGAN 构造对齐的 GAN 伪影样本，然后设计 Separate Expert Fusion (SEF) 框架，先训练两个域专家（VAE 与 SRGAN），再通过门控网络融合特征，以避免梯度冲突。

**💡 创新点**

创新点：1）引入 SRGAN 生成对齐的 GAN 伪影，扩展伪影谱；2）SEF 框架将专家学习与融合解耦，避免不同域间的相互干扰；3）门控融合实现 1+1>2 的性能提升，并通过 LoRA 微调实现轻量化。

**🔧 技术方法**

使用技术：LoRA 微调、门控网络、DINO-v3-L 基础模型、mask‑aware 伪造增强、双阶段专家训练、对齐训练策略（内容、尺寸、格式一致）。

**📊 数据集**

数据集：以 MSCOCO 真实图像为基准，构造 VAE（T_vae）与 SRGAN（T_srgan）对齐伪影；评测使用 13 个不同生成模型的全图检测基准（如 GenImage、DDA-COCO 等）以及 BR‑GEN 细部伪造检测基准。

**📈 对比分析**

比较方式：采用 Balanced Accuracy（阈值 0.5）与多种对齐、低层、基座方法对比；在 GAN 与 Diffusion 两大类别上均取得最高分，GAN 平均 93.7%，Diffusion 94.4%，总平均 94.1%；在 BR‑GEN 上达到 88.4%；在各种扰动（模糊、裁剪、JPEG、噪声）下仍保持最高鲁棒性。

**⚠️ 局限性**

局限性：需要两阶段训练与专家设计，增加训练复杂度与资源消耗；未来需探索更轻量的融合方式并扩展至更多伪影来源。

---

## 311. LEMON: Learning Executable Multi-Agent Orchestration via Counterfactual Reinforcement Learning

**arXiv ID:** 2605.14483 | [PDF](https://arxiv.org/pdf/2605.14483v1)

**作者:** Xudong Chen `[一作]`, Kaize Ding `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于大型语言模型的可执行多智能体编排器，能够一次性生成任务特定的角色、容量和依赖关系的完整执行规范。

**💡 创新点**

创新点在于将角色设计、容量分配和依赖构造合成为单一可执行规范，并通过局部因果对比强化学习为每个字段提供精细的信用分配。

**🔧 技术方法**

使用的技术包括基于GRPO的强化学习、局部因果对比（counterfactual）信用分配、监督预训练、YAML序列化、工作者token计数和依赖图编译。

**📊 数据集**

实验使用六个推理与代码生成基准：MMLU、GSM8K、AQuA、MultiArith、SVAMP 和 HumanEval。

**📈 对比分析**

与单智能体、固定结构MAS、拓扑设计和自适应工作流基线比较，平均得分90.72，超过最强单智能体基线3.96点，最佳的五个任务上均取得领先，且在准确率-token成本曲线上处于或接近Pareto前沿。

**⚠️ 局限性**

局限性包括对手工制定的任务结构依赖较大、训练成本相对较高、以及对因果对比操作的可解释性与稳定性需要进一步验证。

---

## 312. Stop Overthinking: Unlocking Efficient Listwise Reranking with Minimal Reasoning

**arXiv ID:** 2605.14450 | [PDF](https://arxiv.org/pdf/2605.14450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 313. From Schema to Signal: Retrieval-Augmented Modeling for Relational Data Analytics

**arXiv ID:** 2605.14464 | [PDF](https://arxiv.org/pdf/2605.14464v1)

**作者:** Lingze Zeng `[一作]` (National University of Singapore), Beng Chin Ooi `[通讯]` (Zhejiang University)

**通讯引用:** 21347 | [OpenAlex ID](https://openalex.org/A5024892041)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种检索增强建模（RAM）框架，将关系型数据库转化为异构图，并通过信息检索技术为每条元组构建语义文档，进而实现元组间的语义匹配与图结构增强；

**💡 创新点**

通过两种检索驱动的增强策略：ATRA（同表检索增强）利用同表语义相似性进行对比学习；ETRA（跨表检索增强）在图中加入语义短路边，弥补模式化关系的不足；

**🔧 技术方法**

使用信息检索（BM25）、随机游走、词嵌入、预训练语言模型、图神经网络（GraphSAGE）、自监督对比学习和模块化层级网络；

**📊 数据集**

在五个真实关系型数据库上进行实验，涵盖医疗、社交、电商等领域：Trial、Avito、Stack、Event、Beer；

**📈 对比分析**

与表面化方法（CatBoost、LightGBM等）、传统图方法（R-GCN、R-GAT、HGT）及图预训练方法（DGI、GraphCL、BGRL）对比，RAM在13项预测任务上均超越12个基线，尤其在回归任务上提升显著；

**⚠️ 局限性**

仍受检索阈值选择、索引构建成本、跨表匹配误差及数据稀疏性等因素影响，且在极大规模或稀疏结构数据库中效果可能受限；

---

## 314. Real2Sim in HOI: Toward Physically Plausible HOI Reconstruction from Monocular Videos

**arXiv ID:** 2605.14462 | [PDF](https://arxiv.org/pdf/2605.14462v1)

**作者:** Yubo Zhao `[一作]` (Hong Kong University of Science and Technology), Chi-Keung Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13221 | [OpenAlex ID](https://openalex.org/A5062566088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6514db3d-8de6-452c-91b7-acdb31787cc4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 HA-HOI 框架，能够从单目 RGB 视频中恢复可直接用于物理仿真的 4D 人机交互动画。

**💡 创新点**

创新点包括：① 以人体动作为交互锚点，采用人先物后的重建策略；② 利用 VLM 交互提示生成接触候选，进行细粒度接触优化；③ 将优化后的轨迹投射至物理仿真，实现更稳定的交互执行。

**🔧 技术方法**

主要技术包括 SMPL-X 与 MANO 姿态估计、深度校准、单视图 3D 物体重建、VLM 接触提示、物理仿真中的残差控制器和接触约束。

**📊 数据集**

使用 BEHAVE 公开数据集进行评估，同时在多段野外视频上进行验证。

**📈 对比分析**

与 CHORE、InterTrack、HOI-TG、VisTracker、CARI4D、THO 等现有单目 HOI 方法在 BEHAVE 上对比，Chamfer 距离、加速度误差、接触穿透等指标均表现优异，尤其将人-物体接触穿透从 0.084 cm 降至 0.013 cm。

**⚠️ 局限性**

主要局限在于对严重遮挡和视觉误差敏感，仍以视觉估计为主；缺乏对真实机器人硬件、传感噪声和 sim‑to‑real 迁移的直接验证。

---

## 315. CP-OFDM Achieves Lower Ranging CRB Than Frequency-Spread Waveforms in the Large-Sample Regime

**arXiv ID:** 2605.14451 | [PDF](https://arxiv.org/pdf/2605.14451v1)

**作者:** Fan Liu `[一作]` (Southeast University), Shi Jin `[通讯]` (Southeast University)

**通讯引用:** 44879 | [OpenAlex ID](https://openalex.org/A5013079905)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在集成感知与通信（ISAC）系统中，随机数据符号如何影响多目标测距的克拉美-罗恩界（CRB），并通过对Fisher信息矩阵（FIM）结构的分解，揭示了基于不同正交调制波形（如CP‑OFDM、SC、OTFS、AFDM）的测距性能差异。

**💡 创新点**

提出了一个与调制基函数无关的Jensen型下界，证明在PSK星座下CP‑OFDM恰好达到该下界；并证明在子高斯（sub‑Gaussian）星座（如QAM）中，随着符号数N增大，CP‑OFDM在频率扩展正交波形中实现更低的CRB；进一步通过Riemannian几何证明CP‑OFDM在大样本下为局部最优点。

**🔧 技术方法**

利用Fisher信息矩阵分解、Jensen不等式、矩阵解析展开、Hanson‑Wright浓度不等式、随机FIM的高阶矩分析以及单位群上的黎曼几何（梯度与Hessian）等技术手段。

**📊 数据集**

主要采用仿真数据：随机生成L=40个目标位置，符号数N=128，使用16‑QAM或16‑PSK星座，比较CP‑OFDM、SC、OTFS、AFDM等波形的测距CRB。

**📈 对比分析**

通过数值仿真显示，CP‑OFDM在相同带宽与功率下，在16‑QAM情形下比其他波形平均低约1 dB，在16‑PSK情形下低约2 dB；且在大样本条件下，CP‑OFDM的CRB始终低于任何α‑频率扩展波形，且在单位群上局部保持正半定。

**⚠️ 局限性**

局限性包括：仅考虑单天线单一目标回波模型；假设完美循环前缀和理想CP；只覆盖正交单位调制矩阵；对非正交或混合波形未作分析；实际系统中存在多径、噪声非高斯、符号相关等情况，可能影响理论结果的适用性。

---

## 316. BEAM: Binary Expert Activation Masking for Dynamic Routing in MoE

**arXiv ID:** 2605.14438 | [PDF](https://arxiv.org/pdf/2605.14438v1)

**作者:** Juntong Wu `[一作]` (Taobao & Tmall Group of Alibaba), Li Yuan `[通讯]` (Peking University)

**通讯引用:** 18443 | [OpenAlex ID](https://openalex.org/A5100700791)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BEAM框架，通过二进制掩码路由器动态激活或屏蔽专家，实现Token级别自适应稀疏化。

**💡 创新点**

创新点在于解耦稀疏化与专家选择，采用轻量级掩码路由器与Straight‑Through Estimator生成二进制掩码，并通过L1正则控制稀疏度，使模型在极端稀疏下仍保持高精度。

**🔧 技术方法**

使用的技术包括Mixture‑of‑Experts、Top‑K路由、二进制掩码、Straight‑Through Estimator、L1稀疏正则、CUDA自定义核与vLLM集成。

**📊 数据集**

训练使用Tulu3 SFT Mixture Dataset，评估基于OpenCompass八个基准（Math、GSM8K、HumanEval、MMLU、CEVAL、CMMLU、CommonsenseQA、BoolQ）。

**📈 对比分析**

与Top‑K Reduced、Top‑K Pruning、MoE‑Dynamic、AdaMoE、DynMoE等基线对比，BEAM在中/高/极端稀疏下保持>98%准确率、Avg‑K降至0.11，推理速度提升约1.4×吞吐量、2.5×解码加速。

**⚠️ 局限性**

局限性包括对STE的依赖可能导致梯度不稳定；在极端稀疏时仍受共享专家比例限制；尚未在多GPU分布式推理环境中验证。

---

## 317. A Calculus-Based Framework for Determining Vocabulary Size in End-to-End ASR

**arXiv ID:** 2605.14427 | [PDF](https://arxiv.org/pdf/2605.14427v1)

**作者:** Sunil Kumar Kopparapu `[一作]` `[通讯]` (TCS Research), Sunil Kumar Kopparapu (TCS Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于微积分的框架，通过对子词统计曲线进行二次多项式或多项式-指数拟合，并利用一阶、二阶导数条件求解端到端ASR词表大小的最优值。

**💡 创新点**

创新点在于将词表大小视为连续可优化参数，给出解析或数值求解的最优条件，避免传统经验搜索和盲目设定，提升了词表选择的可解释性与精确性。

**🔧 技术方法**

使用了子词分词算法（BPE、WordPiece等）、曲线拟合（scipy.optimize的curve_fit）、微积分优化（求一阶、二阶导数）、数值求解（fsolve、SLSQP）以及基于ESPNet的Conformer模型。

**📊 数据集**

实验数据集为LibriSpeech-100（100小时英语语音及对应文本）。

**📈 对比分析**

与ESPNet默认词表大小300的基线进行对比，使用估计得到的词表大小（如382或61）在LibriSpeech-100的测试集上实现了WER略低或相当的性能提升，尤其在低资源设置下表现更佳。

**⚠️ 局限性**

局限性在于需对Δ(n)和Θ(n)做高质量的双可微曲线拟合，拟合不足会影响最优词表大小的准确性，且框架对不同语料的适用性需进一步验证。

---

## 318. DVMap: Fine-Grained Pluralistic Value Alignment via High-Consensus Demographic-Value Mapping

**arXiv ID:** 2605.14420 | [PDF](https://arxiv.org/pdf/2605.14420v1)

**作者:** Pengyun Zhu `[一作]` (Tianjin University), Deyi Xiong `[通讯]` (Tianjin University)

**通讯引用:** 4769 | [OpenAlex ID](https://openalex.org/A5055232825)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了DVMap框架，实现基于高共识人口统计-价值映射的细粒度多元价值对齐；

**💡 创新点**

创新点在于将宏观国家标签替换为多维人口统计约束，通过熵引导的人口统计原型提取、结构化链式推理和组相对策略优化，构建高一致性价值对齐语料，并通过三维推广基准验证泛化能力；

**🔧 技术方法**

采用熵筛选的人口统计原型、Structured Chain-of-Thought（CoT）思维导向、Group Relative Policy Optimization（GRPO）策略优化以及VeRL框架；

**📊 数据集**

使用World Values Survey（WVS）第7波数据，构建56,152个高共识样本，覆盖10国、16种价值，并生成21,553个跨人口统计/跨国家/跨价值的三维推广测试集；

**📈 对比分析**

与多种主流LLM（Qwen3、DeepSeek、GPT‑4o等）进行对比，Qwen3‑8B‑DVMap在跨人口统计测试中准确率48.6%（高于DeepSeek 45.1%），在跨国和跨价值测试亦表现出色，Wasserstein距离最低，证明其泛化与鲁棒性优于基线；

**⚠️ 局限性**

局限性包括WVS数据静态、对边缘文化群体覆盖不足、人口统计抽象忽视个体心理特征，以及仅评估判别式预测，未验证生成式身份语调输出能力。

---

## 319. Fully Dynamic Rebalancing in Dockless Bike-Sharing Systems via Deep Reinforcement Learning

**arXiv ID:** 2605.14501 | [PDF](https://arxiv.org/pdf/2605.14501v1)

**作者:** Edoardo Scarpel `[一作]` (University of Padua), Gian Antonio Susto `[通讯]` (University of Padua)

**通讯引用:** 4142 | [OpenAlex ID](https://openalex.org/A5026617079)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种完全动态、实时的无架子自行车共享系统再平衡框架，利用单辆卡车通过深度强化学习执行局部调度、加速与充电，以维持站点可用性。

**💡 创新点**

创新点包括：1）实现完全动态再平衡，摒弃传统全网定时大规模操作；2）将空间临界度量与奖励函数结合，引导代理主动预防短缺；3）采用图注意网络处理网格化状态，提升空间感知与策略泛化；4）在有限车队规模下实现显著性能提升。

**🔧 技术方法**

使用深度强化学习（DDQN）、图注意网络（GAT）、MDP建模、离散事件仿真器，结合实时路网与电池动态。

**📊 数据集**

基于2022年9–10月BlueBikes（波士顿地区）出行数据，并通过IDW插值生成无架子需求矩阵；使用Cambridge MA路网与交通报告提供的速度与能耗信息。

**📈 对比分析**

与CitiBike现行静态重平衡算法（SR）对比；在不同车队规模（300、500、700辆）下，DRL将每日失效次数分别从51→12、35→3、140→40，表现出更高的可用性与公平性，且对资源稀缺时的鲁棒性更佳。

**⚠️ 局限性**

局限性：仅研究单卡车设置，未探讨多卡车协同；模型依赖仿真与插值估计，真实无架子数据稀缺；训练过程收敛不稳定，尤其在极小车队时。

---

## 320. Hitting Axis-Parallel Segments with Weighted Points

**arXiv ID:** 2605.14499 | [PDF](https://arxiv.org/pdf/2605.14499v1)

**作者:** Rajiv Raman `[一作]` (Indraprastha Institute of Information Technology Delhi), Jatin Yadav `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5086368450)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究点集合与轴平行线段（或多方向线段）的几何覆盖问题，提出一种基于LP-回归的随机化/系统化舍入算法，首次突破了传统2近似的界限。

**💡 创新点**

创新点在于：① 通过仅对一种方向（水平或竖直）进行精确舍入并在残余子实例上做最优修复，从而得到 (1+2/e)≈1.74 的加权近似，(1+1/(e-1))≈1.58 的无权近似；② 将此框架推广至 d 个方向，得到 (ln d+ln ln d+O(1)) 的近似；③ 在一种方向为无限直线的特殊情形下，改进到 (1+1/e)≈1.36；④ 结合局部搜索技术，为 k‑受限实例提供 PTAS；⑤ 证明了该问题在多方向和直线/线段混合时为 APX‑hard。

**🔧 技术方法**

主要技术包括：LP 线性规划松弛、系统化随机舍入（随机起点的区间法）、概率上界（指数尾部分析）、随机游走/尾部和独立性证明、以及图论中的交换图与分离器论证。

**📊 数据集**

本文未使用真实数据集，而是通过构造性实例（如互补交叉线段图、变量与子句块）来证明整合性缺口与硬度。

**📈 对比分析**

相较于先前的 2 近似，本文在加权和无权两种模式下均实现了显著改进；在 d 方向扩展中，近似因子从 O(d) 降至 O(ln d)；在直线/线段混合场景中，改进至 1.36 近似；实验结果（理论上）显示在随机实例上平均逼近因子可达到 1.1–1.2，远优于传统方法。

**⚠️ 局限性**

局限性：① 仍依赖 LP 松弛，无法解决更高维度或更复杂几何对象（如不规则多边形）的问题；② 对多方向的近似仍受 ln d 限制，难以进一步逼近 1；③ 需要点集合中每条线段至少包含一个候选点，若满足条件不充分则无法直接应用；④ 证明的 APX‑hardness 仅针对有限点集的情形，无法覆盖无限点集的情况。

---

## 321. ROAD: Adaptive Data Mixing for Offline-to-Online Reinforcement Learning via Bi-Level Optimization

**arXiv ID:** 2605.14497 | [PDF](https://arxiv.org/pdf/2605.14497v1)

**作者:** Letian Yang `[一作]` (Shanghai Jiao Tong University), Shuai Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 133000 | [OpenAlex ID](https://openalex.org/A5100371500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了XXX的性能。

**🔧 技术方法**

使用了XXX技术，包括XXX算法和XXX模型。

**📊 数据集**

实验中使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果表明本方法在XXX指标上优于其他方法。

**⚠️ 局限性**

限制在于XXX，可能影响结果的普适性。

---

## 322. Deepchecks: Evaluating Retrieval-Augmented Generation (RAG)

**arXiv ID:** 2605.14488 | [PDF](https://arxiv.org/pdf/2605.14488v1)

**作者:** Assaf Gerner `[一作]` (Deepchecks), Lior Rokach `[通讯]` (Ben Gurion University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Deepchecks 框架，用于系统化评估检索增强生成（RAG）管道的质量，涵盖检索相关性、上下文可信度、完整性以及安全性等属性，并提供根因分析、版本对比与生产监控功能。

**💡 创新点**

创新点在于将 RAG 评估拆分为可独立评估的属性，并通过学习聚合机制生成端到端评估标签；同时集成了多维度安全评估与实时监控，为 RAG 系统的工业化落地提供了完整工具链。

**🔧 技术方法**

使用了多种技术：基于检索的 NDCG@k/MRR@k 及 BEIR、MS-Marco 基准；LLM‑as‑Judge（小型和大型 LLM）进行事实一致性与语义相似度评估；句子嵌入（Sentence‑BERT）与 NLI 模型判定上下文可信度；自研小型语言模型提取事实语句；学习聚合模型用于端到端分类。

**📊 数据集**

使用的数据集包括：TRUE Benchmark（11 个人工标注数据集）、SQuAD、PubmedQA、两份客户内部 Q&A 与工作经历数据；公开 RAG 数据集（均为正例，负例通过对抗方式生成）以及自研的腐败公开数据集。

**📈 对比分析**

通过平衡正负样本的准确率（Accuracy）与人工标签对比，将 Deepchecks 与 RAGAS（GPT‑4o）和 Langsmith（GPT‑4o）进行端到端性能对比。实验显示 Deepchecks 在所有公开与客户数据集上均达到或超过 0.90 的准确率，明显优于或与现有框架竞争，尤其在真实生产场景下表现最佳。

**⚠️ 局限性**

局限性包括：对大规模标注数据的依赖（需要人工评注以训练聚合模型）；聚合模型训练后需在新数据集上重新校准；对多模态或极端专业领域的支持仍有限；安全评估仍基于预设指标，可能无法覆盖所有伦理风险。

---

## 323. GeoVista: Visually Grounded Active Perception for Ultra-High-Resolution Remote Sensing Understanding

**arXiv ID:** 2605.14475 | [PDF](https://arxiv.org/pdf/2605.14475v1)

**作者:** Jiashun Zhu `[一作]` (Jilin University), Bo Yang `[通讯]` (Jilin University)

**通讯引用:** 72347 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GeoVista框架，实现基于规划的多分支主动感知，用于超高分辨率遥感图像的稀疏、细小目标检索与聚合；构建了APEX-GRO超高分辨率交互式轨迹语料库，并使用GRPO强化学习对模型进行对齐。

**💡 创新点**

核心创新点包括：①规划驱动的Observe‑Plan‑Track主动感知策略，避免单轨迹偏差；②跨尺度统一、离散化的空间表示，支持多层(Global‑Region‑Object)检索；③通过APEX‑GRO提供可监督的多分支轨迹训练，显著提升模型对主动探索的理解；④利用GRPO对计数与定位任务进行奖励对齐，使模型在多步骤决策中更优。

**🔧 技术方法**

采用视觉‑语言大模型（基于LLaVA/ Qwen2.5‑VL 等），配合自定义工具调用（crop/zoom 等）、多尺度视觉编码、结构化证据状态跟踪；训练分两阶段：监督微调（SFT）与GRPO强化学习；使用离散相对坐标映射实现跨尺度一致性。

**📊 数据集**

主要使用公开遥感基准：RSHR‑Bench、XLRS‑Bench、LRS‑VQA；APEX‑GRO 数据源包括 DOTA‑v1.5、FAIR1M、AID、UC‑Merced、NWPU‑RESISC45、WHU‑RS19、GeoLLaVA‑8K、EarthVQA 等多任务图像与文本集合。

**📈 对比分析**

与19种基线（静态 VLM、远程感知 VLM、单路径缩放方法等）在三大基准上进行对比；GeoVista 在 RSHR‑Bench 43.10、XLRS‑Bench 52.78、LRS‑VQA 27.71 的平均准确率均为最高，尤其在 XLRS‑Bench 上领先 GeoEyes 超 10.44 分；平均工具调用数升高但每次推理长度更短，表明多分支策略更高效。

**⚠️ 局限性**

局限性包括：①对工具调用预算敏感，算力需求较高；②依赖离散坐标映射，极端尺度变化或非正方形 ROI 可能影响精度；③目前仅在公开基准上验证，跨语言或更大尺寸遥感数据的泛化尚未充分评估；④强化学习阶段仅针对计数/定位任务，其他复杂推理场景的性能提升仍待验证。

---

## 324. ClickRemoval: An Interactive Open-Source Tool for Object Removal in Diffusion Models

**arXiv ID:** 2605.14461 | [PDF](https://arxiv.org/pdf/2605.14461v1)

**作者:** Ledun Zhang `[一作]` (Inner Mongolia University of Technology), Xinying Yao `[通讯]` (Inner Mongolia University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个基于 Stable Diffusion 的点击式交互式对象去除工具 ClickRemoval，仅需用户点击即可定位并去除目标对象，完成背景恢复。

**💡 创新点**

创新点在于利用点击生成语义距离图并通过自注意力重定向（SGAR/SGAS）与自适应调度实现无训练、无掩码/文本的精准去除，同时引入 ARG 融合原始与改造后的噪声预测以控制去除强度。

**🔧 技术方法**

采用了 Stable Diffusion 预训练模型、M2N2 语义距离图、SGAR/SGAS 自注意力重定向与调度、ARG 融合机制，以及自适应调度策略。

**📊 数据集**

基于 Pico-Banana-400K 构建的约 5,000 张对象去除测试样本，并使用编辑后的无目标图像作为评估参考。

**📈 对比分析**

与 LaMa、AttentiveEraser、PixelHacker、Inpaint Anything、PowerPaint-v2 等开源基线比较，ClickRemoval 在 512/1024 分辨率下取得 FID 9.35/8.05、Local‑FID 17.27/15.56，用户与 GPT 评估中获得最高/第二名。

**⚠️ 局限性**

局限包括对复杂遮挡、多对象场景仍需多次点击；在极端细节或高分辨率时可能出现微小伪影；缺乏完全自动化的高效掩码生成与快速调参。

---

## 325. OmniDrop: Layer-wise Token Pruning for Omni-modal LLMs via Query-Guidance

**arXiv ID:** 2605.14458 | [PDF](https://arxiv.org/pdf/2605.14458v1)

**作者:** Yeo Jeong Park `[一作]` (Samsung Research), Yongkweon Jeon `[通讯]` (Samsung Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了训练无关、按层次进行的Omni-LLM token压缩框架 OmniDrop，通过逐层根据文本查询指导的方式逐步裁剪音视频 token，并引入时间多样性得分来保持全局时序信息。

**💡 创新点**

创新点包括：①在 LLM 解码器层内逐层裁剪而非输入级别；②利用文本查询来衡量音视频 token 的重要性，实现任务自适应的保留比例；③引入时间多样性得分（TDS）避免过度集中剪枝导致时序信息丢失。

**🔧 技术方法**

采用 Sigmoid 动态剪枝比例调度、文本‑音视频注意力权重计算、时间多样性得分、FlashAttention 加速以及先对单模态进行前置裁剪等技术。

**📊 数据集**

在 VideoMME、WorldSense、AVUT 等多种视频音频理解基准上进行评估。

**📈 对比分析**

与现有训练无关方法 OmniZip 和 DASH 对比，OmniDrop 在 Qwen2.5‑Omni 7B/3B 模型上保留比例 30% 时 VideoMME 和 AVUT 分别比所有基线高约 1–2 分，甚至超过完整 token 基线；保留比例 20% 时提升 3.58 分，预填充时间缩短 40% 及内存降低 14.7%。

**⚠️ 局限性**

依赖文本查询限制了无文本输入场景的适用性；剪枝比例与超参数选择仍需经验式设定，缺乏自动化学习。

---

## 326. LiSA: Lifelong Safety Adaptation via Conservative Policy Induction

**arXiv ID:** 2605.14454 | [PDF](https://arxiv.org/pdf/2605.14454v1)

**作者:** Minbeom Kim `[一作]` (Google Cloud AI Research), Long T. Le `[通讯]` (Google Cloud AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了LiSA，一个通过结构化策略记忆实现的终身安全适配框架，能够在部署期间仅利用稀疏且可能嘈杂的用户反馈改进大型语言模型的安全守护。

**💡 创新点**

通过将稀疏错误报告抽象为宽泛策略、在多标签混合区生成冲突感知的局部策略，并以Beta后验下界进行置信度门控，LiSA实现了保守且高效的记忆重用，避免过度泛化与误拒。

**🔧 技术方法**

采用在线–离线循环、自然语言策略记忆、语义检索、Beta后验置信度门控、局部冲突检测以及低成本的策略诱导和多模型部署等技术。

**📊 数据集**

在PrivacyLens+、ConFaide+和AgentHarm三大二分类守护基准上进行评估。

**📈 对比分析**

与纯预测、AGrail、Synapse、ReasoningBank等基线对比，LiSA在稀疏反馈下保持最高宏F1（≈0.96），对噪声稳健（ρ=20%仍>0.92），并在保持低延迟的前提下逼近模型升级的性能边界。

**⚠️ 局限性**

需要先验的基础守护模型；记忆诱导依赖LLM推理成本；在极端噪声或极少反馈时仍可能产生错误抽象；对跨任务迁移性尚未验证。

---

## 327. GGBound: A Genome-Grounded Agent for Microbial Life-Boundary Prediction

**arXiv ID:** 2605.14442 | [PDF](https://arxiv.org/pdf/2605.14442v1)

**作者:** Hanbo Huang `[一作]` (Shanghai Jiao Tong University), Shiyu Liang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1967 | [OpenAlex ID](https://openalex.org/A5083787469)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个微生物生命边界预测框架GGBound，通过基因编码的LLM代理预测温度、pH、盐度等生理边界、最适条件、底物利用、分类特征和形态学特征。

**💡 创新点**

创新点在于：① 将基因信息作为连续嵌入与文本融合，形成基因条件化LLM；② 引入RAG和GEM两种生物工具实现检索和代谢模拟；③ 设计基因因果奖励的GRPO训练策略，确保模型真正利用基因嵌入；④ 构建了大规模、去名化的1,525株、6,448实例的多任务基因-生理基准。

**🔧 技术方法**

技术包括：LucaOne基因嵌入、token融合、Qwen 4B LLM、RAG检索、GEM代谢模型、三阶段训练（基因-文本对齐、SFT、GRPO）以及因果奖励机制。

**📊 数据集**

数据集为从IJSEM、NCBI和BacDive自动提取的微生物基因-生理数据，包含88,927条高质量记录，随后构成1,525株、6,448实例的评测基准以及1,000+训练实例。

**📈 对比分析**

与多种大规模通用LLM（DeepSeek-V3.2、GLM-4.7、Kimi-K2等）对比，GGBound在所有任务上均表现最优或相近，尤其在生理边界覆盖率、最优条件RMSE、代谢mAP@5以及形态学准确率上均取得显著提升；工具使用更高效。

**⚠️ 局限性**

局限在于单一CLS嵌入压缩的基因表示可能失去局部基因和通路信息，限制了对底物利用等细粒度特征的预测；未来可探索多token、基因集池化或检索式基因特征交互。

---

## 328. What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions

**arXiv ID:** 2605.14422 | [PDF](https://arxiv.org/pdf/2605.14422v1)

**作者:** Shuqi Gu `[一作]` (ShanghaiTech University), Kan Ren `[通讯]` (ShanghaiTech University)

**通讯引用:** 1863 | [OpenAlex ID](https://openalex.org/A5102807475)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在未来条件为文本描述的反事实时间序列预测任务，提出了一种基于文本归因的扩散模型（TADiff），并设计了无真值评估指标。

**💡 创新点**

创新点包括：①将文本条件转化为可调节的“可变”与“不可变”特征，①通过文本归因实现历史特征与未来条件的解耦；②在缺失真值的反事实场景下，利用对比学习的DTTC指标进行评估；③使用反事实数据增广与微调提升模型对未知未来情景的泛化。

**🔧 技术方法**

技术手段主要包括：扩散模型（denoising diffusion implicit models）、文本编码器（transformer）、时间序列编码器（对比学习）、文本归因机制、反事实数据生成与微调、DTTC评价框架。

**📊 数据集**

使用的公开数据集有 Synth（人工合成），ETTm1、Traffic、Exchange、Weather 等真实世界时序数据，并在这些数据上构造反事实样本。

**📈 对比分析**

与传统单模态与多模态基线（DLinear、PatchTST、Sundial、VerbalTS、TimeCMA、TimeMMD、CT、IATSF）对比。TADiff 在事实条件下在 MAE/MSE 上往往最优或相近，在反事实条件下在 DTTC-I/DTTC-E 上显著领先，证明在缺失真值时仍能保持语义一致性与预测准确性。

**⚠️ 局限性**

局限性包括：①缺少对不同预测长度的系统评估；②对真实反事实未来的生成与评估依赖对比学习，可能对特征分布变化敏感；③模型复杂度较高，训练成本大；④解释性仍有限，需要进一步研究以便更好地解释归因结果。

---

## 329. Test-Time Learning with an Evolving Library

**arXiv ID:** 2605.14477 | [PDF](https://arxiv.org/pdf/2605.14477v1)

**作者:** Weijia Xu `[一作]` (Microsoft Research), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 35501 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在推理时自适应学习的框架，利用黑盒LLM通过抽象提取、权重更新和合并机制，构建并维护一个可演化的知识库，实现无梯度、无监督的知识积累与复用。

**💡 创新点**

核心创新在于：① 将推理轨迹转化为可重用的模块化技能与反思洞察；② 引入信息增益（IG）与未来信息增益（Future IG）双重信用分配，驱动知识抽象的即时效益与长期价值；③ 通过合并与动态权重更新，使抽象在跨任务中逐步演化为更通用、可复用的知识单元。

**🔧 技术方法**

技术方案包括：LLM推理与自评；抽象提取（函数、子问题、子流程、反思句子）；Embedding相似检索与合并；信息增益与未来信息增益计算；按权重抽样、动态更新知识库；以及多任务迭代更新循环。

**📊 数据集**

评估数据集涵盖：数学推理（HMMT 2025–2026）、代码生成（BigCodeBench Hard、LiveCodeBench v6 Hard）、多轮交互代理（ScienceWorld、PDDL），在静态与连续学习两种设置下进行测试。

**📈 对比分析**

与传统测试时缩放（Best‑of‑N、RSA）以及测试时学习（ExpRAG、Dynamic Cheatsheet）对比，所提方法在数学准确率、代码通过率和代理成功率上分别提升约5–10%，并在相同token成本下表现出更高的token效率，尤其在较低预算时显著优于基线。

**⚠️ 局限性**

局限性在于：依赖LLM自身对结果的评估，若验证成本与生成成本相近或更高，效果可能受限；目前仅验证于数学、代码与代理任务，对开放式问答或科学研究等更开放性或需要外部验证的场景适用性尚未证实。

---

## 330. Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict

**arXiv ID:** 2605.14473 | [PDF](https://arxiv.org/pdf/2605.14473v1)

**作者:** Yihang Chen `[一作]` (Georgia Institute of Technology), Xinpeng Wei `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 50935 | [OpenAlex ID](https://openalex.org/A5044756341)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Context‑Driven Decomposition (CDD) 作为一种在检索增强生成（RAG）中的推理时诊断工具，通过将检索上下文与模型内部知识拆分、对比、隔离冲突前提并记录解决轨迹，从而显式揭示检索上下文对答案的影响。

**💡 创新点**

创新点在于：①将知识冲突建模为信念修正框架，并以自然语言提示实现五步分解；②设计 NLI‑gate 的路由机制实现计算与精度折衷；③在标准 RAG 上引入可解释的冲突拆解，提供上下文合规性度量；④发布 Epi‑Scale 基准，系统化研究不同冲突类型和模型族的鲁棒性。

**🔧 技术方法**

主要技术包括：自然语言提示驱动的上下文与参数提取、Jensen‑Shannon Divergence 直观化（通过提示实现），Premise Isolation、Resolution 步骤；NLI‑filtered gating（α‑版本）；误差注入与截断等因果干预测试；以及对 Claude、Gemini 等多种闭源 API 进行的跨模型评估。

**📊 数据集**

使用的数据集：①Epi‑Scale（4,500 例，来源于 HotpotQA、Natural Questions、FEVER，含 Entity Swap、Logical Contradiction、Temporal Shift、Distractor Evidence 四类人工合成冲突）；②TruthfulQA 500 例的误导性谬误注入测试；还通过交叉验证在 Claude 家族模型上进行重复实验。

**📈 对比分析**

与标准 RAG、Vanilla CoT、Self‑RAG、NLI‑Filtered RAG 等基线对比。CDD 在 Epi‑Scale 的对抗性准确率达 78.1%（宏平均），高于标准 RAG 的 63.0%；在 TruthfulQA 的误导注入上，CDD 从 15% 提升至 62%；在 Claude 系列中，尽管 CDD 同样提升对抗性准确率（约 2–3%），但因果耦合指标未出现显著提升，表明提升机制可能因模型架构而异。

**⚠️ 局限性**

局限性包括：①因果耦合（误差注入敏感性）在 Gemini 上显著但在 Claude 上缺失，说明机制不通用；②α‑版阈值 τ 的选择仅基于小样本直方图，缺乏系统调优；③TruthfulQA 误导注入使用最极端谬误，未涵盖自然检索环境；④信念修正框架仍停留在概念层面，未实现基于 token‑level JSD 的精细量化；⑤因果干预仅结合事实与行为指令，缺少纯粹的事实干预；⑥实验基于闭源 API，可能随模型版本漂移而变化。

---

## 331. Focused PU learning from imbalanced data

**arXiv ID:** 2605.14467 | [PDF](https://arxiv.org/pdf/2605.14467v1)

**作者:** Elias Zavitsanos `[一作]` (NCSR Demokritos), Georgios Paliouras `[通讯]` (NCSR Demokritos)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于焦点损失的聚焦PU学习方法（iFPU），在极度不平衡的正负标签环境下直接利用正样本和未标记样本进行风险最小化训练。

**💡 创新点**

创新点在于：①将焦点损失融入PU风险估计器，突出难分辨的正样本；②设计非负聚焦风险估计器（iFPU），无需预处理或伪标签；③在SCAR与更逼近真实情形的SAR（PG）两种标签机制下均保持稳健。

**🔧 技术方法**

使用的技术包括：非负PU风险估计（nnPU）框架、焦点损失（γ=3）、深度网络（5层MLP或TabTransformer+gMLP）、XGBoost、随机梯度下降（SGD）等；实验中还对比了两步方法NNIF、树基PUHDT/PUHRF、LBE、SAREM等。

**📊 数据集**

实验数据集：14个公开不平衡二分类基准数据集（Cardio、Thyroid、Climate、ForestCover、Seismic、Mammography、Shuttle、Letter、Yeast、Poker、PieChart、PizzaCutter、Satellite、Segment）；以及真实世界财务报表误述检测数据（2000–2014年美国上市公司财报，28财务指标+14派生特征）。

**📈 对比分析**

在SCAR和SAR两种标签机制、25%、50%、75%正样本标注比例下，与uPU、nnPU、i-NNPU、NNIF、PUHDT、PUHRF、LBE、SAREM、iFPU_XGB以及全监督XGBoost进行宏平均ROC‑AUC/PR‑AUC或R‑precision对比。结果显示：iFPU在极度不平衡（仅25%正样本标注）时明显优于其他PU方法，PR‑AUC提升至≈0.60–0.62（相较PUHRF的≈0.59–0.63），在财务误述检测任务中R‑precision从18.22%提升至19.95%（比PUHRF高约1.7%）。

**⚠️ 局限性**

局限性包括：①对先验概率πp的估计敏感，尽管鲁棒性好但错误估计会削弱性能；②缺乏超参数调优（γ、α、学习率等），可能导致进一步提升空间；③仅验证二分类PU场景，未扩展到多标签或多类PU；④依赖焦点损失的假设，若数据分布与PG假设不符，效果可能下降。

---

## 332. Geographic Patterns in I2P Peer Selection: An Empirical Network Topology Analysis

**arXiv ID:** 2605.14435 | [PDF](https://arxiv.org/pdf/2605.14435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 333. Stateful Reasoning via Insight Replay

**arXiv ID:** 2605.14457 | [PDF](https://arxiv.org/pdf/2605.14457v1)

**作者:** Bin Lei `[一作]` (University of Minnesota), Xin Eric Wang `[通讯]` (Simular AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了InsightReplay，一种在链式思考（CoT）过程中周期性提取并重播关键洞见的状态化推理方法

**💡 创新点**

创新点在于通过将关键洞见保持在生成前沿来缓解注意力衰减，进而把CoT长度-准确率的倒U曲线向右平移并提升峰值

**🔧 技术方法**

利用大型语言模型的推理与摘要能力、前向注意力分析、以及RL（GRPO）训练中的Rollout重播技术

**📊 数据集**

使用AIME、HMMT、GPQA Diamond、LiveCodeBench v5等数学与编程推理数据集，以及对应的16条子任务

**📈 对比分析**

对比标准CoT、延长推理（VO）与InsightReplay（1/3/5轮），在24个模型×任务组合上均显著提升准确率，IR3平均提升1.21点（微平均）/1.65点（宏平均），并在最硬任务上达到+9.2点；相比之下仅延长推理提升仅+0.61点

**⚠️ 局限性**

在接近饱和的任务上增益有限（如HMMT仅+0.1点），且需额外的token开销（IR3约1.36倍），对模型参数或结构未做进一步优化，可能受限于原始模型的推理能力

---

## 334. Think When Needed: Adaptive Reasoning-Driven Multimodal Embeddings with a Dual-LoRA Architecture

**arXiv ID:** 2605.14448 | [PDF](https://arxiv.org/pdf/2605.14448v1)

**作者:** Longxiang Zhang `[一作]` (Alibaba Group), Pipei Huang `[通讯]` (Alibaba Group)

**通讯引用:** 1675 | [OpenAlex ID](https://openalex.org/A5059615376)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Think When Needed (TWN) 的统一多模态嵌入框架，结合了双 LoRA 适配器与自适应推理机制，实现对链式推理的按需调用。

**💡 创新点**

创新点包括：1）在同一冻结的 MLLM 主干上部署双 LoRA 并在其接口处梯度分离，以消除生成与对比学习目标的冲突；2）自监督路由门能根据对比度学习的正负边际差值决定是否生成 CoT；3）利用全局嵌入缓存的嵌入引导 RL（GRPO），进一步提升 CoT 的检索质量。

**🔧 技术方法**

使用了双 LoRA 适配器、对比学习（InfoNCE）、自监督路由门（sigmoid MLP）、嵌入引导的 GRPO 强化学习、全局负样本缓存以及 Qwen3.5-35B-A3B 作为教师与评判模型。

**📊 数据集**

采用 MMEB‑V2 作为评估基准，并通过生成‑筛选管道构造包含图像、视频、视觉文档的 CoT 训练数据，来源包括 MMEB‑train、LLaVA‑Hound、ViDoRe 及 VisRAG。

**📈 对比分析**

在 78 个 MMEB‑V2 任务上与基准的离散嵌入模型（ColPali、VLM2Vec 等）和生成嵌入模型（TTE、UME‑R1）比较，TWN‑8B 自适应模式在总体 Hit@1 / NDCG 上达 68.7 分，超过现有最优模型，并将推理 token 数量平均减少约 45%，显著提升效率。

**⚠️ 局限性**

局限性包括：1）仍依赖冻结的主干模型，可能限制对新模态的适应；2）自适应门的阈值和路由策略需要针对不同任务手动微调；3）强化学习阶段需要全局缓存和额外计算，增加训练成本。

---

## 335. FrontierSmith: Synthesizing Open-Ended Coding Problems at Scale

**arXiv ID:** 2605.14445 | [PDF](https://arxiv.org/pdf/2605.14445v1)

**作者:** Runyuan He `[一作]` (UC Berkeley), Alvin Cheung `[通讯]` (UC Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 FrontierSmith，一个自动化流程，能够把闭合式编程题通过目标变更、输出限制和输入泛化等突变转化为开放式编程题，并通过想法差异度筛选出高质量的题目。

**💡 创新点**

创新点在于：①使用想法差异度（idea divergence）作为多解多策略的判定指标，自动区分真正的开放式题目；②通过 LLM 生成测试用例与判分器，并交叉验证确保可评估；③用闭合式题库作为种子，系统化地扩充开放式题目数据集。

**🔧 技术方法**

技术方法包括：LLM 触发突变、LLM‑Judge 评估、想法差异度估算（LLM 与执行相结合）、测试用例与判分器自动生成与交叉验证，以及基于 GRPO 的 RL 训练。

**📊 数据集**

使用的数据集：闭合式竞赛题库 HardTests（47,136 题）作为种子；自动生成的 200 题 FrontierSmith 题集；对照实验中使用人类策划的 FrontierCS 172 题与 ALE-bench 40 题。

**📈 对比分析**

比较方法：在 FrontierCS 与 ALE-bench 上分别计算 Avg@5 和 Best@5；实验表明 FrontierSmith 在 9B/27B 模型上均达到或超过人类策划数据的表现，同时远超直接在 HardTests 上训练的结果；随机奖励基线表现仅与未训练基线相当，验证奖励信号的必要性。

**⚠️ 局限性**

局限性：目前仅处理单文件、无外部依赖的自包含算法题，无法覆盖需要完整环境或多文件的软件工程类开放式题；RL 训练仅使用单步 GRPO，未探索多步交互式 RL，且受限于 100 步的训练长度。

---

## 336. Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL

**arXiv ID:** 2605.14434 | [PDF](https://arxiv.org/pdf/2605.14434v1)

**作者:** Jianbo Zhu `[一作]` (Taobao & Tmall Group of Alibaba), Junjie Bai `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在电商搜索召回阶段构建了基于语义聚类的生成检索框架（CQ‑SID）和专家引导的强化学习优化方法（EG‑GRPO），实现了高效且业务友好的召回效果。

**💡 创新点**

核心创新包括：①将商品 ID 视为语义聚类而非唯一标签，使用类别引导的残差量化与查询‑商品对比学习实现聚类；②分阶段进化的查询‑SID 学习流程；③在稀疏奖励环境下引入专家样本的 GRPO（EG‑GRPO）以稳定并提升多目标召回。

**🔧 技术方法**

技术手段主要有：Residual Quantized Variational Autoencoder（RQ‑VAE）、Qwen2.5‑0.5B 轻量级 LLM、类别约束残差量化、查询‑商品 InfoNCE 对比学习、Group Relative Policy Optimization（GRPO）与专家注入。

**📊 数据集**

使用阿里巴巴 TmallApp 实际搜索日志，约 37.5M 训练样本（21.1M 查询‑商品对，16.4M 仅商品），21M 进化训练样本，90.3M 查询‑SID 及 73.7M 个性化查询‑SID 数据，测试集 201k 及 170k。

**📈 对比分析**

与传统 RQ‑VAE 基线对比，CQ‑SID 在同一束宽度下提升 26.8% 语义召回点击命中率，Beam@1 上 0.0758 vs 0.0598；Beam@10 0.3161 vs 0.2579；Beam@100 0.6181 vs 0.5199；同时 Beam 大小减半，召回质量不下降。在线 A/B 测试实现 GMV 提升 1.15% 与 UCTCVR 提升 0.40%，生成召回渠道占曝光 50.25%、点击 58.96%、购买 72.63%。

**⚠️ 局限性**

局限性包括：①聚类 ID 的更新需要频繁重编码；②RL 训练依赖稀疏点击/曝光信号，仍存在收敛不确定性；③仅验证于单一大型电商平台，跨领域泛化待进一步验证；④模型规模与部署资源仍受限，需进一步压缩与加速。

---

## 337. MemLineage: Lineage-Guided Enforcement for LLM Agent Memory

**arXiv ID:** 2605.14421 | [PDF](https://arxiv.org/pdf/2605.14421v1)

**作者:** Ciyan Ouyang `[一作]` (Institute of Information Engineering, CAS), Rui Hou `[通讯]` (Institute of Information Engineering, CAS)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种在LLM代理持久记忆中附加加密原始记录和衍生链路的防御机制，并通过确定性基准评估其在跨会话记忆污染攻击中的有效性。

**💡 创新点**

通过在每个内存条目上添加 Ed25519 签名、RFC‑6962 Merkle 日志以及基于加权归因的有向无环图链路传播规则，实现了链路完整性追踪与敏感操作门控的联合防御，弥补了仅签名或基于检索过滤的不足。

**🔧 技术方法**

采用 CBOR 规范化编码、Ed25519 公钥签名、RFC‑6962 Merkle 证明、基于 LLM 归因的权重分配（二次 LLM 判断或白盒注意力）以及 Progent JSON‑schema 策略门控技术。

**📊 数据集**

主要使用自定义确定性攻击工作负载（AgentPoison、MemoryGraft、线性衍生链路攻击）以及 Codex/AgentDojo 的银行转账任务进行评估，未使用公开大规模数据集。

**📈 对比分析**

通过 ASR 矩阵与 τ×K 归因阈值敏感性分析进行比较；单次操作开销低于 1 ms，Ed25519 验证占主导，整体不超过 LLM 调用噪声，且在所有攻击上实现 0% ASR，保留 100% 正常功能。

**⚠️ 局限性**

需要信任 LLM 推理路径；归因算法对可解释性和精确度依赖于 LLM 访问权重或二次评估；对强制父节点遗漏的攻击存在开放式失效；多主机复制与跨域一致性未涵盖。

---

## 338. FedStain: Modeling Higher-Order Stain Statistics for Federated Domain Generalization in Computational Pathology

**arXiv ID:** 2605.14590 | [PDF](https://arxiv.org/pdf/2605.14590v1)

**作者:** Fengyi Zhang `[一作]` (Hainan University), Wenzhuo Sun `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 FedStain 框架，在联邦学习环境下通过交换高阶染色统计量实现跨机构的域泛化。

**💡 创新点**

首次将偏度与峰度等高阶染色统计量纳入联邦域泛化，兼顾隐私与通信效率，并引入对比学习与双重对齐损失提升鲁棒性。

**🔧 技术方法**

采用 FedAvg 基础框架，结合 RandStainNA 颜色扰动、MixStyle+AugMix 的多级增强、监督对比损失与 Jensen‑Shannon 对齐损失等技术。

**📊 数据集**

使用 Camelyon17（5 医院）和自构建的 MvMidog（4 扫描仪）两个 H&E 组织切片数据集。

**📈 对比分析**

与 FedAvg、FedProx、FedCCRL、FedAlign、Strap 等基线在 ResNet18/50 上进行 Leave‑One‑Domain‑Out 评估，FedStain 在所有测试域平均提升 2–4% 甚至在最难域上提升超过 10%。

**⚠️ 局限性**

仅在补丁级别验证，未评估全切片级大规模训练；高阶统计量受分辨率与通道数限制，极端光照或扫描噪声下的鲁棒性仍需进一步验证。

---

## 339. EndPrompt: Efficient Long-Context Extension via Terminal Anchoring

**arXiv ID:** 2605.14589 | [PDF](https://arxiv.org/pdf/2605.14589v1)

**作者:** Han Tian `[一作]` (Nankai University), Dawei Yin `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出EndPrompt方法，通过在短序列后追加终止提示，并对位置索引进行插值，实现在不使用完整长序列训练的前提下扩展大语言模型的上下文窗口；

**💡 创新点**

创新点在于：①只需在原短上下文后附加一个短提示，保持语义连续；②利用位置插值将提示定位于目标上下文边界，生成稀疏但覆盖广阔的相对位置监督；③证明RoPE与位置插值的光滑性能让模型在未见距离上平滑泛化，突破传统需密集长序列训练的限制；

**🔧 技术方法**

技术手段包括：Rotary Position Embedding (RoPE)、位置插值 (PI)、Transformer共享参数、端提示构造、短序列自回归微调；

**📊 数据集**

训练数据为约10亿token的通用文本语料；评测数据集包括RULER、LongBench、GSM8K、HumanEval、MMLU、HellaSwag、代码补全、问答、摘要、few-shot等；

**📈 对比分析**

与LCEG、LongLoRA、全长微调等基线对比，ET在RULER上平均得分76.03，超过LCEG 72.24、LongLoRA 72.95、Full FT 69.23；在LongBench上平均得分38.30，优于Full FT 35.63；与PoSE融合的ET(PoSE)进一步提升至RULER 79.44、LongBench 39.65；

**⚠️ 局限性**

局限性包括：需要针对不同模型或上下文长度调优终止提示；在极端长上下文（>96k）性能略有下降；仅适用于使用RoPE/PI的模型，未验证对其他位置编码的适用性；短文本能力在未进一步微调时略逊于全长微调；

---

## 340. LiWi: Layering in the Wild

**arXiv ID:** 2605.14552 | [PDF](https://arxiv.org/pdf/2605.14552v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 341. Dimension-Level Intent Fidelity Evaluation for Large Language Models: Evidence from Structured Prompt Ablation

**arXiv ID:** 2605.14517 | [PDF](https://arxiv.org/pdf/2605.14517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 342. PyCSP3-Scheduling: A Scheduling Extension for PyCSP3

**arXiv ID:** 2605.14559 | [PDF](https://arxiv.org/pdf/2605.14559v1)

**作者:** Sohaib Afifi `[一作]` `[通讯]` (University of Artois), Sohaib Afifi (University of Artois)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对pycsp3的调度抽象层(py csp3-scheduling)，通过 interval、sequence 等高级调度变量把传统 CP 模型的低层编码提升为更易读、可维护的形式，并将其编译为标准 CP 约束。

**💡 创新点**

创新点在于提供了完整的可选、强度和转移时间支持的间隔与序列抽象，并在保持模型/求解器分离的前提下，将这些抽象编译为现有的 global（如 Cumulative）或更通用的整数约束，从而兼容整个 pycsp3 生态。

**🔧 技术方法**

使用了基于 Python 的编译方案，将 interval 变量映射为 start、end、size 等整数变量，并根据 presence、transition matrix 等特征采用全局约束、对偶分解或单单资源累积的方式实现。

**📊 数据集**

在 17 个模型族共 261 个实例（包含 AircraftLanding、RCPSP、LotSizing、MRCPSP 等）上进行评测，数据来源于 CSPLib、PSPLIB、MiniZinc challenge 等公开数据集。

**📈 对比分析**

对每个实例分别使用经典整数模型和调度抽象模型，在相同求解器 ACE 1200 秒内求解并记录状态、目标和运行时间；结果显示 80.8% 的实例状态一致，完全匹配的 72 对最佳解目标一致，平均可获得 1.5× 的加速，某些族如 MSPSP、AircraftLanding 提升显著，而 LotSizing、MRCPSP 则表现出速度退化。

**⚠️ 局限性**

主要局限在于对可选间隔与转移时间的全局约束无法直接编译，导致在 MRCPSP 等族中采用 O(n²) 对偶分解；同时模型结构的增大（如 FlexibleJobshopScen）也导致变量/约束数量激增，影响求解效率。

---

## 343. Analysis of wireless network access logs for a hierarchical characterization of user mobility

**arXiv ID:** 2605.14540 | [PDF](https://arxiv.org/pdf/2605.14540v1)

**作者:** Francisco Talavera `[一作]` (University of Balearic Islands), Carlos Guerrero `[通讯]` (University of Balearic Islands)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于Wi‑Fi AP连接日志的层级化用户移动性建模方法，先将AP按建筑、楼层、翼等空间层级聚合，生成每一层的转移矩阵与停留时间向量，然后利用该模型生成合成移动轨迹，可直接导入雾计算仿真环境。

**💡 创新点**

创新点在于：
1) 通过层级地理划分将海量AP聚合，显著降低建模复杂度；
2) 结合用户聚类得到不同类型用户的专属转移矩阵，提升模型细粒度；
3) 提供从原始Wi‑Fi日志到层级模型的完整流程，并支持跨场景迁移与适配。

**🔧 技术方法**

使用技术包括：数据清洗与阈值判定、k‑means聚类、转移矩阵与时间向量生成、RMSE与Chord图评估、Python脚本自动化、JSON配置层级划分、Aruba ALE日志采集与解析。

**📊 数据集**

数据集为一所大学校园的Wi‑Fi AP（425个）连接日志，采集周期为2020年11月9‑15日，一周共约122 MB原始数据，涵盖学生、教师与行政人员的移动轨迹。

**📈 对比分析**

评估方法：用真实轨迹生成模型，再用该模型生成合成轨迹并重新建模，比较两组转移矩阵与时间向量的RMSE；与非层级化OD矩阵做对比。结果显示层级化模型的平均RMSE约为1.5‑9.8%（矩阵）和4.5‑20.8%（时间），聚类数从158降至10，建模执行时间从约14 k s降至约2 k s，说明在保持误差可接受的前提下显著降低复杂度。

**⚠️ 局限性**

局限性：
1) 对停留时间向量的估计误差相对较大；
2) 层级划分需要人工定义，缺乏自动化；
3) 模型对不同网络技术（如5G）适配性尚未验证；
4) 仅基于Wi‑Fi日志，难以捕获室内细粒度移动；
5) 迁移到完全不同地理特征的场景时仍需人工调整。

---

## 344. Enjoy Your Layer Normalization with the Computational Efficiency of RMSNorm

**arXiv ID:** 2605.14521 | [PDF](https://arxiv.org/pdf/2605.14521v1)

**作者:** Yuxin Guo `[一作]` (Beihang University), Lei Huang `[通讯]` (Beihang University)

**通讯引用:** 2283 | [OpenAlex ID](https://openalex.org/A5100784430)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在任意深度神经网络中提出一种框架，能够判断并在保持数学等价的前提下将 Layer Normalization 替换为更高效的 RMSNorm，从而实现推理加速。

**💡 创新点**

创新点在于：①提出列中心化约束（CCC）和列权重中心化（CBWC）来消除 LN 的均值归一化；②定义可折叠 LN 和零均值图的概念，并给出图论判定算法；③在训练和推理阶段均保持与原模型等价，且能在常见模型上实现 2%–12% 的推理加速。

**🔧 技术方法**

核心技术包括：层归一化（LN）与 RMSNorm 的数学等价分析、列中心化约束与权重中心化重参数化、图结构零均值检测算法、以及在 PyTorch 等框架中的实现。

**📊 数据集**

使用的公开数据集包括 Multi30K（机器翻译）、AG News（文本分类）、CIFAR‑10、MNIST（图像分类）、ImageNet‑100、以及标准大规模语言模型训练集如 WikiText‑103（预训练）和 Alpaca（指令微调）。

**📈 对比分析**

通过在 GPT‑2、BERT、OPT、Phi‑3、Swin‑Tiny 等模型上实验，比较了原始 LN、无中心化的 RMSNorm 与 CBWC+RMSNorm 的准确率/损失和推理时间；实验显示 CBWC+RMSNorm 的准确率基本与 LN 相当，训练损失略高于 LN 但低于 RMSNorm，推理时间相比 LN 提升 2%–12%，训练吞吐率也接近 RMSNorm。

**⚠️ 局限性**

局限性包括：在训练阶段存在 dropout 等模块时会破坏零均值假设，导致等价性失效；目前仅在少量模型和任务上验证，未覆盖所有大型 LLM/VLM；实现中需手动或自动化插入 CBWC，可能对已有模型兼容性有一定要求。

---

## 345. Asymmetric Generative Recommendation via Multi-Expert Projection and Multi-Faceted Hierarchical Quantization

**arXiv ID:** 2605.14512 | [PDF](https://arxiv.org/pdf/2605.14512v1)

**作者:** Bin Huang `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 23375 | [OpenAlex ID](https://openalex.org/A5100339293)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AsymRec框架，解决生成式推荐中的输入输出信息瓶颈问题

**💡 创新点**

创新性地将输入映射为连续表示，输出采用多面向层次量化，形成不对称的连续-离散学习机制

**🔧 技术方法**

使用多专家语义投影(MSP)、多面向层次量化(MHQ)以及Transformer解码器，配合EMA和正则化技术

**📊 数据集**

在Amazon Review的Sports、Beauty、Toys、CDs四个子数据集上进行实验

**📈 对比分析**

与多种基准（如CasRec、BERT4Rec、SASRec、VQ-Rec、TIGER等）比较，平均提升NDCG@10约15.8%，在离线指标和线上A/B测试中均显著优于现有方法

**⚠️ 局限性**

主要局限包括对多专家数量和量化参数的依赖，且对极冷启动场景仍有挑战，未来需进一步探索更鲁棒的表示学习策略

---

## 346. TeachAnything: A Multimodal Crowdsourcing Platform for Training Embodied AI Agents in Symmetrical Reality

**arXiv ID:** 2605.14556 | [PDF](https://arxiv.org/pdf/2605.14556v1)

**作者:** Zidong Liu `[一作]` (State Key Laboratory of General Artificial Intelligence, BIGAI), Zhenliang Zhang `[通讯]` (State Key Laboratory of General Artificial Intelligence, BIGAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了三阶段演示范式并实现了云端众包平台TeachAnything，用于收集语言、视频和遥控演示

**💡 创新点**

将语言、视频、遥控演示统一为三阶段范式，支持开放式、多场景、多任务的多模态监督

**🔧 技术方法**

使用Isaac Sim+PhysX物理模拟、WebSocket流、Flask服务、HaMeR手势控制、机器人逆向动力学与学习策略

**📊 数据集**

基于云端收集的自定义多模态演示数据，覆盖FrankAr arm、Unitree G1等多机器人，在多样化虚拟场景中收集

**📈 对比分析**

文中未给出与其他方法的量化对比与性能指标，主要描述了平台架构与功能

**⚠️ 局限性**

仍处于开发阶段，缺乏完整的端到端训练流水线和用户实验评估；仅支持键鼠和手势遥控，未加入VR等更自然的交互

---

## 347. TOPOS: High-Fidelity and Efficient Industry-Grade 3D Head Generation

**arXiv ID:** 2605.14594 | [PDF](https://arxiv.org/pdf/2605.14594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 348. Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning

**arXiv ID:** 2605.14587 | [PDF](https://arxiv.org/pdf/2605.14587v1)

**作者:** Oubo Ma `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 8024 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

本文系统评估了七种塑性干预对深度强化学习（DRL）后训练阶段后门攻击的影响，基于14,664个实验案例进行量化分析。

**💡 创新点**

创新点包括：①揭示Sharpness-Aware Minimization（SAM）会显著放大后门风险；②提出Sweeper‑Converter‑Connector（SCC）三元框架，用于在后训练阶段实现高效后门注入；③提出基于损失曲面尖锐度的后门检测方法。

**🔧 技术方法**

技术手段涵盖：四种后门攻击方法（TrojDRL、BadRL、SleeperNets、UNIDOOR）；八种塑性干预（Shrink & Perturb、Weight Clipping、Spectral Normalization、Weight Decay、Layer Normalization、ReDo、SAM 及 None）；使用随机种子对比实验；利用梯度和有效秩等病态特征进行内部机制分析。

**📊 数据集**

数据集主要为 OpenAI Gym 控制任务（CartPole、Acrobot、MountainCar、Pendulum、LunarLander、BipedalWalker）以及 PyBullet 机器人任务（Hopper、Reacher、HalfCheetah），覆盖离散与连续动作空间、稀疏与密集奖励场景。

**📈 对比分析**

通过攻击成功率（ASR）与正常任务表现（BTP）两指标进行对比；实验显示 SAM 在后训练阶段将 ASR 从 0.178 提升至 0.326（+83%），并同时提升 BTP；其余干预往往降低 ASR 并维持或略增 BTP；组合干预进一步放大后门威胁。

**⚠️ 局限性**

局限性包括：实验范围仅限于离散/连续控制任务，未覆盖更复杂或真实世界场景；尖锐度检测阈值缺乏统一标准，易产生误报；多干预组合的相互作用机制尚未完全解析，需要进一步理论和实验验证。

---

## 349. A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval

**arXiv ID:** 2605.14581 | [PDF](https://arxiv.org/pdf/2605.14581v1)

**作者:** Ho Hung Lim `[一作]` (Hong Kong University of Science and Technology), Yi Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 82243 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发了一个诊断基准，评估单向量聚合在视觉检索（Visual RAG）中对表格中心财务文档的可靠性，并系统展示了聚合后会导致关键数值和文本信息丢失。

**💡 创新点**

创新点在于：①提出了基于数值/文本敏感度与可视化注意力的诊断方法；②揭示了“全局纹理主导”是单向量聚合失效的根本原因；③在多种VLM和聚合策略上进行对比实验，验证了该问题的普遍性。

**🔧 技术方法**

使用了视觉语言模型（VLM）的视觉编码器、平均池化/最大池化的单向量聚合、后期交互指标（MaxSim、MeanPatch、MinPatch）以及三种改进聚合策略（方差加权、注意力引导、top‑k 去除），并通过可视化注意力分析和相似度度量进行实验。

**📊 数据集**

使用了FinQA和TAT‑DQA两套表格中心财务文档数据集，构建了200对敏感度测试样本和可视化分析样本。

**📈 对比分析**

与不同规模VLM（7B–32B）、检索优化嵌入模型（Qwen3‑VL‑Embedding‑8B、GME‑Qwen2‑VL‑7B‑Instruct）以及三种聚合改进方法进行对比。实验显示，平均/最大池化在所有敏感度测试中的相似度≈1，几乎无法区分微小/大幅语义差异；MaxSim/MeanPatch仅略有提升；MinPatch能捕捉到差异，但聚合仍隐藏信息。整体说明单向量检索在金融场景下存在显著风险。

**⚠️ 局限性**

限制：仅评估两类表格中心财务文档，未涵盖发票、资产负债表等不同布局或手写文档；仅关注数值/文本小幅扰动，未测试更广泛扰动；未提供完整检索排名评估（如Recall@k、nDCG）；全球纹理主导的结论是否适用于其他领域仍需进一步验证。

---

## 350. Woodelf++: A Fast and Unified Partial Dependence Plot Algorithm for Decision Tree Ensembles

**arXiv ID:** 2605.14578 | [PDF](https://arxiv.org/pdf/2605.14578v1)

**作者:** Ron Wettenstein `[一作]` (Reichman University), Udi Boker `[通讯]` (Reichman University)

**通讯引用:** 816 | [OpenAlex ID](https://openalex.org/A5000415323)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个统一高效的算法，能够在决策树集成模型上快速计算单特征和特征对的Partial Dependence Plot（PDP、Joint‑PDP）以及任意阶的Partial Dependence Interaction Value（PDIV），并实现完整PDP和GPU加速。

**💡 创新点**

创新点包括：①将PDP、Joint‑PDP和PDIV视为线性度量在伪布尔函数的Weighted Disjunctive Normal Form（WDNF）上的计算；②构造紧凑的消费者数据集，使得本地归因算法能够高效得到全局依赖图；③在PDIV计算上实现指数级复杂度提升（从FastPD的$2^{2^D}$下降到$2^D$），大幅缩短运行时间。

**🔧 技术方法**

使用了伪布尔函数与WDNF、FastPD的路径拆分与局部归因、Path‑Dependent SHAP、GPU并行化等技术，并通过与已有PDP近似方法的统一实现。

**📊 数据集**

实验数据集为IEEE‑CIS欺诈检测数据集（约47.2万行、397个特征）和KDD Cup 1999入侵检测数据集（约489.8万行、127个特征）。

**📈 对比分析**

与scikit‑learn及FastPD对比，实验表明在PDP和Joint‑PDP上约能提升6倍速度；在PDIV上从FastPD估计的百万年显著降至5分钟，且在大样本（k=100）下仍可在10分钟内完成。

**⚠️ 局限性**

限制在于：仍需要显式背景与消费者数据集；Full PDP在树集成规模极大时阈值数量可能过多导致内存/时间开销；当树深度或特征维度非常大时，GPU显存仍是瓶颈。

---

## 351. Remember Your Trace: Memory-Guided Long-Horizon Agentic Framework for Consistent and Hierarchical Repository-Level Code Documentation

**arXiv ID:** 2605.14563 | [PDF](https://arxiv.org/pdf/2605.14563v1)

**作者:** Suyoung Bae `[一作]` (Sungkyunkwan University), Jee-Hyong Lee `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1908 | [OpenAlex ID](https://openalex.org/A5067651075)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 MemDocAgent，一种基于长时间序列推理的单代理框架，用于在整个仓库级别生成结构化、连贯的文档。

**💡 创新点**

创新点在于：① 将依赖关系与层级结构统一编码为遍历顺序，保证每个文档在其依赖和子单元已完成后才生成；② 引入共享外部内存 RepoMemory，让代理在长轨迹中读取、写入、验证并复用之前的检索与文档，显著减少重复检索和跨文档冲突；③ 设计跨文档冲突检测和自评机制，使生成的文档在事实性、完整性和有用性上保持一致。

**🔧 技术方法**

技术手段包括：ReAct 样式的长序列推理；外部内存管理与缓存；基于依赖图的拓扑顺序生成；自然语言推断（NLI）用于冲突检测；多轮自评与验证；使用大型语言模型（Qwen3‑Coder‑30B‑A3B、GPT‑5‑mini、Claude‑Haiku 4.5 等）进行文档写作、评判和代码重构。

**📊 数据集**

使用 20 个从 DevEval 公开评测集挑选的 Python 仓库作为实验数据，包含多领域、多层级、具备单元测试的项目。

**📈 对比分析**

与多种开源（RepoAgent、DocAgent、CodeWiki、Prompting）及闭源（DeepWiki、Claude‑Code）基线进行对比，评估指标为完整度、真确性、实用性及信息充分性（通过代码重构的 Pass@k / CodeBLEU 测试）。MemDocAgent 在所有指标上均领先，尤其在信息充分性上显著超越基线，减少 41% 的检索时间； ablation 证明每个子模块对整体性能均至关重要。

**⚠️ 局限性**

局限性包括：长序列推理导致高计算资源消耗；代理偶尔因推理错误陷入循环（实验中仅 0.08% 的组件未通过验证）；缺乏更高效的轨迹控制和自适应终止机制。

---

## 352. Resolving Action Bottleneck: Agentic Reinforcement Learning Informed by Token-Level Energy

**arXiv ID:** 2605.14558 | [PDF](https://arxiv.org/pdf/2605.14558v1)

**作者:** Langzhou He `[一作]` (University of Illinois Chicago), Qitian Wu `[通讯]` (Broad Institute of MIT and Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ActFocus 方案，在 agentic RL 中对每个生成 token 进行权重重分，降低 reasoning token 的梯度贡献、提升 action token 的权重，并利用冻结参考模型的 token 能量对 action token 进一步加权，以此改善梯度分配和训练效果。

**💡 创新点**

① 发现并量化 Action Bottleneck，即行动 token 对训练信号占主导；② 提出基于 token 级能量的加权重分机制，实现对行动 token 的聚焦；③ 方案实现简单、无额外推理/计算成本，兼容现有 PPO/GRPO 等 policy‑gradient 算法。

**🔧 技术方法**

token‑level reweighting、free‑reference token energy 计算、Energy‑based 模型视角、PPO/GRPO policy‑gradient、批归一化 + sigmoid 对能量加权、梯度加权聚合等。

**📊 数据集**

四个多轮 agentic 环境：Sokoban、FrozenLake、4×4 Sudoku、WebShop；使用 Qwen2.5‑Instruct 系列 LLM，分别在 0.5B、1.5B、3B 等不同规模上进行实验。

**📈 对比分析**

与统一 credit（w_t=1）下的 PPO/GRPO 进行对比；在所有环境和模型规模上均实现显著提升，成功率提升可达 65.2pp（PPO）和 63.7pp（GRPO）；同时显著改善 GRPO 的训练稳定性（峰值‑最终衰退减少）。

**⚠️ 局限性**

需对 α（reasoning 下降系数）与 β（能量加权强度）进行调参，最佳值随任务/模型规模不同；过度抑制 reasoning token 可能导致推理能力受限；方案仅针对 token‑级权重分配，未完全解决所有 credit‑assignment 问题；能量计算依赖冻结参考模型，若参考模型失效可能影响效果。

---

## 353. CSLibPremiseBench: Structure-Guided Premise Retrieval and Label Robustness for Lean 4 Computer-Science Theorems

**arXiv ID:** 2605.14549 | [PDF](https://arxiv.org/pdf/2605.14549v1)

**作者:** Junye Ji `[一作]` `[通讯]` (University of Washington), Junye Ji (University of Washington)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了针对 Lean 4 计算机科学库 CSLib 的检索基准 CSLibPremiseBench，提供可复现的任务与候选记录，并对源可见代理标签进行三重审计；

**💡 创新点**

通过构造严格的候选访问策略、三信号标签审计以及结构感知重排序 CSG‑Rerank，系统地评估了仓库结构和候选可见度对检索性能的影响，首次在 CSLib 上进行此类细粒度的检索实验；

**🔧 技术方法**

使用 BM25、符号/名称重叠、命名空间/模块邻近、导入图、模块 PageRank、源顺序局部性等传统信息检索特征，并结合手工设定的加权融合构建 CSG‑Rerank；

**📊 数据集**

以 CSLib v4.29.0（1762 个构建作业，284.95 秒）为数据集，提取 801 个可标注任务、1875 个候选声明，覆盖算法、可计算性、基础、语言、逻辑五大族；

**📈 对比分析**

与基线方法（BM25、符号重叠、结构加权混合）进行任务级配对自助法比较，结果显示 BM25+符号在 Recall@5/10/50 和 nDCG@50 上表现最佳，CSG‑Rerank 在 MRR 上略有提升（+0.0175，95% CI>0），但在 Recall@10、nDCG@10 以及上下文包效用指标上未显著优于 BM25+符号；

**⚠️ 局限性**

主要局限包括：代理标签仅为源可见的证明引用，缺乏完整的 Lean 依赖追踪；候选策略对结果影响显著，严格策略下局部性强；CSG‑Rerank 采用手工权重且未进行学习调优；上下文包评估仅测量覆盖与结构集中度，未证明实际证明生成或修复效果；

---

## 354. RxEval: A Prescription-Level Benchmark for Evaluating LLM Medication Recommendation

**arXiv ID:** 2605.14543 | [PDF](https://arxiv.org/pdf/2605.14543v1)

**作者:** Shuhao Chen `[一作]` (Hong Kong University of Science and Technology), James T. Kwok `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 16903 | [OpenAlex ID](https://openalex.org/A5070273088)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个面向住院药物处方级的多选题评估基准RxEval，评估LLM在实时临床情境中的处方决策能力

**💡 创新点**

创新点：将任务从入院级预测转为处方级决策，使用时间序列临床轨迹和精确药物‑剂量‑途径三元组；通过推理链扰动生成患者特定的干扰选项，提升评估对临床推理的敏感性

**🔧 技术方法**

技术：利用LLM进行推理链注释、扰动生成与验证；采用GPT‑5系列与其它LLM进行零样本评估，使用vLLM部署和多选题评估指标（F1、Exact Match 等）

**📊 数据集**

数据集：基于MIMIC‑IV电子病历抽取584名患者、1,547个多选题、969种药物，覆盖18个ICD章节

**📈 对比分析**

对比方法：在零样本设定下评估16个LLM，使用Exact Match、Jaccard、Precision、Recall、F1 等指标；发现LLM性能差距大（F1 45–77），但与MedQA相比更具挑战性；高阶模型仅达46% Exact Match

**⚠️ 局限性**

局限性：受限于长文本截断、难以处理多药组合、长期情境推理仍欠缺；模型易忽略明确的患者信息或推理错误；干扰生成仍需人工验证

---

## 355. Learning from Failures: Correction-Oriented Policy Optimization with Verifiable Rewards

**arXiv ID:** 2605.14539 | [PDF](https://arxiv.org/pdf/2605.14539v1)

**作者:** Mengjie Ren `[一作]` (Chinese Academy of Sciences), Yaojie Lu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 1992 | [OpenAlex ID](https://openalex.org/A5103090910)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CIPO方法，将RLVR中的失败轨迹转化为纠错监督，提升LLM的推理与纠错能力。

**💡 创新点**

将失败轨迹进行局部重采样生成纠错样本，并加入自适应重放、风险规避奖励塑形和难度感知偏好，实现对失败的方向性利用。

**🔧 技术方法**

在RLVR框架下使用策略梯度、GRPO、适应性重放、风险规避奖励塑形、难度感知采样等技术。

**📊 数据集**

在11个基准上验证，包括数学推理（AIME、AMC、MATH500、Minerva、OlympiadBench等）和代码生成（LiveCodeBench、LeetCode、CriticBench、DebugBench等）。

**📈 对比分析**

与GRPO、PRIME等基线对比，CIPO在数学任务平均提升约4-6%准确率，pass@K提升6%以上，代码纠错率提升7%+，整体性能显著优于基线。

**⚠️ 局限性**

对极难任务的失败仍难以纠正；需要人工验证与评估；在更大模型或多任务场景下的可扩展性尚待验证。

---

## 356. From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper

**arXiv ID:** 2605.14525 | [PDF](https://arxiv.org/pdf/2605.14525v1)

**作者:** Ling Li `[一作]` (Tsinghua University), Zhidong Deng `[通讯]` (Tsinghua University)

**通讯引用:** 2901 | [OpenAlex ID](https://openalex.org/A5102011846)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出稀疏交错多视角输入范式，并设计 DenseWarper 模型将稀疏输入转换为高频、时空一致的 3D 姿态输出；

**💡 创新点**

创新点在于利用不同视角的时间位移实现有效提升输出帧率与降低数据冗余，并通过 epipolar geometry 进行空间热图融合以及基于可变形卷积的 Warper 模块实现时空融合；

**🔧 技术方法**

使用 epipolar geometry 进行热图空间融合、可变形卷积 Warper 进行时间融合、滑动窗口策略实现实时处理，配合 2D 姿态检测器（CPN、SimpleBaseline）和三角测量得到 3D 姿态；

**📊 数据集**

在 Human3.6M 与 MPI‑INF‑3DHP 两个公开基准数据集上进行评估；

**📈 对比分析**

与单视角、全帧多视角、插值方法（Adafuse、PPT 等）进行 MPJPE、P‑MPJPE 对比，取得 21.3mm（GT 2D）/33.6mm（CPN）/22.3mm（SimpleBaseline）在 Human3.6M 上的最佳成绩，MPI‑INF‑3DHP 上 65.89mm；同时保持较低的模型参数、延迟和更高的性能效率；

**⚠️ 局限性**

对非均匀采样或极低帧率摄像头的稀疏交错输入尚未验证，间距过大时时空信息提取效果下降，方法在很大程度上依赖于多视角输入的时间密度。

---

## 357. Defenses at Odds: Measuring and Explaining Defense Conflicts in Large Language Models

**arXiv ID:** 2605.14514 | [PDF](https://arxiv.org/pdf/2605.14514v1)

**作者:** Xiangtao Meng `[一作]` (Shandong University), Shanqing Guo `[通讯]` (Shandong University)

**通讯引用:** 1366 | [OpenAlex ID](https://openalex.org/A5084460856)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统地研究了大语言模型在逐步部署防御时的跨防御交互，评估了144条安全、隐私与公平三维度的防御序列；

**💡 创新点**

首次揭示跨防御冲突的存在、异向性和顺序依赖性，并通过层级表示偏移、激活轨迹逆转等机制解释冲突，提出了冲突引导层冻结的轻量级缓解方案；

**🔧 技术方法**

使用ConflictEval框架、层级表示差异分析、激活补丁、PCA轨迹、参数冲突评分、冲突得分等技术进行定量与机制分析；

**📊 数据集**

基准模型在TOFU数据集上进行指令微调，安全评估使用SALAD-Bench恶意提示，隐私评估使用TOFU泄露数据，公平评估使用StereoSet；

**📈 对比分析**

通过对比每条序列在防御前后安全风险、提取强度与公平得分的变化，发现38.9%的序列出现风险回升，其中两例灾难性崩溃；冻结高冲突层后可显著降低或消除回升；

**⚠️ 局限性**

仅考虑两阶段防御组合，防御方法覆盖有限，冻结策略为演示性实验，未实现全局联合优化，结果可能不适用于更复杂的多防御场景；

---

## 358. VMU-Diff: A Coarse-to-fine Multi-source Data Fusion Framework for Precipitation Nowcasting

**arXiv ID:** 2605.14597 | [PDF](https://arxiv.org/pdf/2605.14597v1)

**作者:** Chunlei Shi `[一作]` (Southeast University), Dan Niu `[通讯]` (Southeast University)

**通讯引用:** 24255 | [OpenAlex ID](https://openalex.org/A5100326855)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种两阶段的降雨现在预报框架（VMU‑Diff），先利用多源融合的 Vision Mamba UNet 对雷达与多波段卫星数据进行粗略预测，再通过条件扩散模型对粗预测进行细化。

**💡 创新点**

创新点包括：
1) 先粗略再细化的 coarse‑to‑fine 两阶段结构；
2) 将雷达与卫星多波段数据进行多源融合，提升对雾台与降水启动/消散的把握；
3) 采用 Vision Mamba 状态空间（VMSS）块代替传统注意力，保持线性复杂度并捕捉长程时空依赖；
4) 在扩散细化阶段引入 Conditional Mamba State‑Space（CMSS）模块，利用条件信息自适应地重建残差，既提高细节质量又加速推理。

**🔧 技术方法**

使用的技术主要有：Vision Mamba 状态空间网络（VMSS）、条件扩散模型（CMSS）、时空注意力块、Diffusion 训练目标与 DDIM 采样器、Adam 优化器等。

**📊 数据集**

使用的数据集为江苏 SWAN 数据集，包含 1 km 空间分辨率、10 min 时间分辨率的雷达回波与 Himawari‑8 卫星多波段（C8、C11、C13、C15）图像，训练集覆盖 2019‑2021 年，测试集为 2019 年 6–7 月。

**📈 对比分析**

与 TrajGRU、SmaAt‑Unet、AA‑TransUNet、Earthformer、DiffCast、Vmunet 等 SOTA 模型在 CSI、HSS、FAR（降雨阈值 25‑50 dBZ）以及图像质量指标 SSIM、LPIPS 上进行对比。VMU‑Diff 在 CSI/HSS/FAR 与 SOTA 接近或略优，在 SSIM 上显著提升、LPIPS 降低，显示出更好的图像细节与视觉一致性。

**⚠️ 局限性**

局限性：
- 扩散细化阶段的计算量相对较大，推理速度仍低于纯确定性模型；
- 在更长预报时段（>30 min）或极端短时事件（如闪电）仍可能出现模糊或细节缺失；
- 多源数据融合受时空分辨率差异影响，融合策略在不同传感器组合下需要进一步优化。

---

## 359. Privacy Auditing with Zero (0) Training Run

**arXiv ID:** 2605.14591 | [PDF](https://arxiv.org/pdf/2605.14591v1)

**作者:** Tudor Cebere `[一作]` (Inria), Aurélien Bellet `[通讯]` (Inria)

**通讯引用:** 2225 | [OpenAlex ID](https://openalex.org/A5014504793)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了零跑时隐私审计框架，利用成员与非成员数据的观测数据和倾向分数校正，得到可验证的差分隐私下界。

**💡 创新点**

将因分布偏移导致的共轭效应建模为观测因果问题，提出全局组合视角与点级条件校正两种修正方法。

**🔧 技术方法**

采用f-DP/GDP 以及适配的组合和条件式校正、倾向分数估计与自助法不确定性量化。

**📊 数据集**

在合成高维噪声求和机制和真实世界的 WILDS iWildCam 视觉数据上进行实验。

**📈 对比分析**

与传统 One-Run 审计对比，零跑审计在存在分布偏移时保持较高的下界，且在 IID 情况下仅差约 1。

**⚠️ 局限性**

依赖于准确估计倾向分数，且受限于 One-Run 方法的样本效率和攻击强度。

---

## 360. Uncertainty Quantification for Large Language Diffusion Models

**arXiv ID:** 2605.14570 | [PDF](https://arxiv.org/pdf/2605.14570v1)

**作者:** Artem Vazhentsev `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Artem Shelmanov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 719 | [OpenAlex ID](https://openalex.org/A5071397402)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性研究大语言扩散模型（LLDMs）的无监督置信度估计（UQ），并提出一种基于扩散轨迹的轻量级UQ方法 D‑CoCoA。

**💡 创新点**

创新点：①提出多种仅依赖扩散去噪过程的无监督不确定性信号（轨迹语义不稳定性、Remask、NFE 等）；②将自回归 LLM 的 CoCoA 方法改造为 D‑CoCoA，并通过理论证明轨迹不相似度下界于掩蔽扩散训练目标，证明其作为不确定性度量的合理性；③实现了极低计算开销下接近采样方法性能的 UQ。

**🔧 技术方法**

技术手段：LLDM 去噪轨迹采样、蒙特卡洛掩蔽负对数似然（MCNLL）、轨迹统计量（如 NFE、Remask、FlipCount）、进度加权轨迹不相似度（AD^prog）、语义相似度（RoBERTa Cross‑Encoder）以及 D‑CoCoA 组合框架。

**📊 数据集**

数据集与模型：使用 8 个数据集（QA：MMLU、TriviaQA、CoQA、GSM8k；摘要：SamSum、XSum；MT：WMT14 fr→en、WMT19 de→en），模型为 LLaDA‑1.5 和 Dream（均为 7B 参数的指令微调 LLDM）。

**📈 对比分析**

比较方法与性能：与自回归 LLM 的白盒、黑盒、采样和监督 UQ 方法（如 MSP、CoCoA、KLE 等）以及采样基础的 MCNLL、D‑DegMat 等进行对比。使用 PRR（和 ROC‑AUC）评估；D‑CoCoA 在性能‑效率前沿，接近最优采样方法但计算开销约低 100 倍，平均 PRR 显著提升。

**⚠️ 局限性**

局限性：①在单步/单块生成（如 MMLU）中轨迹不稳定性信号不足；②语义不相似度对相似度函数的选择敏感；③理论证明基于理想化假设（完美校准、理想语义度量），实际效果受限；④尚未在更大模型或多语言环境中全面验证。

---

## 361. SpectraFlow: Unifying Structural Pretraining and Frequency Adaptation for Medical Image Segmentation

**arXiv ID:** 2605.14566 | [PDF](https://arxiv.org/pdf/2605.14566v1)

**作者:** Zhiquan Chen `[一作]` (Sun Yat-sen University), Hejun Wu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1436 | [OpenAlex ID](https://openalex.org/A5102755158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计了一个两阶段框架，先用结构导向的MeanFlow预训练Encoder，再用轻量级解码器（含Direct Attentional Fusion与Frequency-Directional Dynamic Convolution）进行精细边界分割。

**💡 创新点**

创新点在于：①将掩码仅作为条件输入，采用潜在空间最优传输的MeanFlow进行结构化预训练；②加入Dispersive Loss防止特征崩溃；③在解码器中融合DAF与FDConv，显著提升边界精度。

**🔧 技术方法**

采用的技术包括：DINOv2视觉Transformer编码器、MeanFlow潜在空间最优传输预训练、Dispersive Loss、Direct Attentional Fusion（DAF）、Frequency‑Directional Dynamic Convolution（FDConv）以及Dice/BCE等分割损失。

**📊 数据集**

在ISIC‑2016（皮肤病）、Kvasir‑SEG（肠道息肉）、GlaS（腺体）三大2D医学分割数据集上进行评估，并在Synapse多器官3D数据集上验证跨模态通用性。

**📈 对比分析**

与多种CNN/Transformer基线（U‑Net、U‑Net++、PraNet、TGANet、DCSAUNet、XBFormer、ConDSeg等）以及Synapse基线比较，SpectraFlow在所有数据集均取得最高mIoU/mDSC，HD95显著下降，尤其在少标注场景下保持高精度，表现最优。

**⚠️ 局限性**

局限性包括：预训练阶段依赖大量无标注图像与掩码；两阶段训练过程较为复杂，调参耗时；对不同模态（如CT/CT‑MRI混合）适配性尚未完全验证。

---

## 362. Multi-Dimensional Model Integrity and Responsibility Assessment Index and Scoring Framework

**arXiv ID:** 2605.14550 | [PDF](https://arxiv.org/pdf/2605.14550v1)

**作者:** Phuc Truong Loc Nguyen `[一作]` (Friedrich-Alexander University Erlangen Nuremberg), Hung Cao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个统一的评估框架 MIRAI，用于在同一实验设置下对多维度（可解释性、公平性、可持续性、鲁棒性、隐私）进行综合评分。

**💡 创新点**

创新点在于将五个传统单独评估维度的指标进行归一化、方向对齐后加权求和，得到一个可直接比较的整体指标，同时强调模型复杂度与跨维度平衡的关系。

**🔧 技术方法**

采用 SHAP+Quantus、Fairlearn+AIF360、ART+Alibi-Detect 等成熟工具，结合碳排放估算、FLOPs/MACs 计数等量化手段构建完整指标体系，并在 Python 环境下实现自动化评估流水线。

**📊 数据集**

使用了三大高风险表格数据集：Diabetes Hospitals、German Credit、Census Income，覆盖医疗、金融与社会经济领域。

**📈 对比分析**

通过在每个数据集上训练六种常见模型（DT、XGB、SVM、MLP、TRN、FTT），对齐所有指标后计算 MIRAI 分数，发现高精度模型并不总能在 MIRAI 上领先，MLP 与 SVM 在多维度平衡上表现最佳。

**⚠️ 局限性**

局限在于仅覆盖二分类表格任务，缺乏对多分类、时序或图数据的扩展；指标加权默认均等，未针对特定行业法规进行自定义；框架对极端不平衡样本和大规模特征空间的鲁棒性尚待进一步验证。

---

## 363. Local Spatiotemporal Convolutional Network for Robust Gait Recognition

**arXiv ID:** 2605.14548 | [PDF](https://arxiv.org/pdf/2605.14548v1)

**作者:** Xiaoyun Wang `[一作]` (Osh State University), Wu Wang `[通讯]` (Osh State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种局部时空卷积网络（LSTCN），通过双分支结构学习步态的时空特征，显著提升识别准确率。

**💡 创新点**

创新点包括：全局双向空间池化（GBSP）将时空张量降维为横竖条纹表示；局部时空卷积（LSTC）与非对称卷积联合捕获时空细粒度；局部时空池化（LSTP）在条纹层面聚合最具辨识度的特征。

**🔧 技术方法**

使用的技术主要有2D卷积、GBSP、LSTC（含非对称卷积）、LSTP、三元组+焦点损失的联合训练策略。

**📊 数据集**

实验数据集为CASIA‑B（124人）和OU‑MVLP（10,000+人），覆盖多视角、不同步态和服装变化。

**📈 对比分析**

与多种基准方法（GaitSet、GaitSlice、MT3D、GaitGraph等）对比，LSTCN在CASIA‑B的三种训练协议下分别达到97.3%、93.7%、83.8%的平均准确率，在OU‑MVLP上取得84.4%/85.8%（全局/横向版本）最高的rank‑1准确率，显著优于现有技术。

**⚠️ 局限性**

局限性包括：仍需大量标注视频数据进行训练；对遮挡、极端姿态和动态背景适应能力有限；虽然参数量相对较少，但在实时部署时仍需进一步压缩与加速。

---

## 364. Discovering Physical Directions in Weight Space: Composing Neural PDE Experts

**arXiv ID:** 2605.14546 | [PDF](https://arxiv.org/pdf/2605.14546v1)

**作者:** Pengkai Wang `[一作]` (Hong Kong Polytechnic University), Dong Ni `[通讯]` (Zhejiang University)

**通讯引用:** 12053 | [OpenAlex ID](https://openalex.org/A5065374358)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在相同基准神经算子上针对不同物理参数微调得到的端点专家，并提出通过分解端点残差为共享适配和符号物理方向的坐标读出方法，从而在不进行额外训练的情况下实现跨参数迁移。

**💡 创新点**

创新点在于将同一基准下的端点微调视为权重空间中的一维物理坐标线，分离共享与符号方向；提出 Calibration‑Conditioned Merge (CCM)，利用元数据、缩放或短期轨迹前缀推断坐标并合成单一检查点；并在外域推理中验证该方法显著提升性能。

**🔧 技术方法**

技术包括神经算子（FNO、DPOT）、权重残差分解、坐标读出、后处理合并、物理参数归一化与校准，以及基于短期前缀的损失评估。

**📊 数据集**

使用的数据集为 reaction–diffusion（DiffReact）、二维 Navier‑Stokes 流（NS2D）以及径向坝崩塌（RDB）三种基准。

**📈 对比分析**

与静态平均、最佳固定 α、诊断 oracle、任务条件算子、目标重训练等方法对比；在外域推理上，CCM 分别比基准降低 54.2%、42.8% 与 13.8% 的 L2 误差，显著优于其他合并或基准方法。

**⚠️ 局限性**

局限性包括需要共享架构和局部线性假设，端点间距离过大或多参数系统时可能失效；CCM‑Prefix 需观察前缀，无法实现零样本；在极端 OOD 或高曲率区域无法保证性能。

---

## 365. Exploring Geographic Relative Space in Large Language Models through Activation Patching

**arXiv ID:** 2605.14535 | [PDF](https://arxiv.org/pdf/2605.14535v1)

**作者:** Stef De Sabbata `[一作]` (University of Leicester), Kevin Roitero `[通讯]` (University of Udine)

**通讯引用:** 1020 | [OpenAlex ID](https://openalex.org/A5050287516)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用激活补丁方法探究大型语言模型在处理相对空间（地理距离）概念时的内部机制，特别是在Gemma 2 2B模型上。

**💡 创新点**

创新之处在于将激活补丁技术应用于地理空间认知的研究，并系统地评估不同距离表达对模型内部激活的影响。

**🔧 技术方法**

技术上采用了激活补丁（activation patching）与TransformerLens库，利用KL散度比较补丁前后输出，重点在MLP输出层进行五层滑动窗口补丁。

**📊 数据集**

使用的数据集为GeoNames数据库中249个英国人口超过五万的地点，并构造清洁与失真提示词以测试距离表达。

**📈 对比分析**

通过比较补丁后输出与失真输出的KL散度差异来评估补丁效果，实验显示在早期层对数字信息的补丁影响显著，距离越大影响越强，但整体结果仍为初步，需更大规模验证。

**⚠️ 局限性**

局限性包括样本量有限、距离表达的token化不对称导致补丁不完全、未对多样提示进行扩展、结果为初步探索，需进一步扩大地点与提示范围。

---

## 366. PROVE: A Perceptual RemOVal cohErence Benchmark for Visual Media

**arXiv ID:** 2605.14534 | [PDF](https://arxiv.org/pdf/2605.14534v1)

**作者:** Fuhao Li `[一作]` (MiLM Plus, Xiaomi Inc.), Jian Luan `[通讯]` (MiLM Plus, Xiaomi Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了RC（Removal Coherence）两种指标（RC‑S、RC‑T）和双层真实视频基准 PROVE‑Bench，用于更准确评估图像和视频中的对象去除质量。

**💡 创新点**

创新点在于：① 使用滑动窗口局部分布匹配（MMD）对掩码区域与周围背景或相邻帧特征分布进行比较，克服传统 FR/NR/全局时序指标的偏差；② 结合配对真实视频与无配对挑战集的双层基准，既能进行量化比较，又能测试极端真实场景的鲁棒性。

**🔧 技术方法**

技术手段包括：DINOv2 作为特征提取器；滑动窗口裁剪；局部特征分布匹配（MMD）；对齐掩码并计算交并集；以及针对视频的联合裁剪与时间窗口评估。

**📊 数据集**

使用的数据集有：PROVE‑M（80对运动增强的真实视频）、PROVE‑H（100无 GT 的真实挑战视频）、RORD‑Val、OBER‑Wild、DAVIS、ROSE‑Bench 等。

**📈 对比分析**

与 PSNR/SSIM/LPIPS、ReMOVE、CFD、TC/TF 等现有指标比较，RC‑S/T 在六个基准上与人类评估的 Kendall τ 和 Spearman ρ 一致性最高，尤其在模糊、抖动、快速运动等挑战下，RC‑T 对时间扰动的敏感度显著优于 TC/TF，表明性能更好。

**⚠️ 局限性**

局限性包括：对特征提取器和窗口尺寸敏感，未充分验证对极端遮挡或非凸掩码的鲁棒性；在无 GT 场景下仍需更多人工评测以进一步验证；以及对极大尺寸视频处理的计算成本仍待优化。

---

## 367. ArcGate: Adaptive Arctangent Gated Activation

**arXiv ID:** 2605.14518 | [PDF](https://arxiv.org/pdf/2605.14518v1)

**作者:** Avik Bhattacharya `[一作]` (Indian Institute of Technology Bombay), Biplab Banerjee `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 2150 | [OpenAlex ID](https://openalex.org/A5020786167)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Adaptive Arctangent Gated Activation (ArcGate)，并在 ResNet-50 与 Vision Transformer 上替换固定激活函数。

**💡 创新点**

ArcGate 使用七个可学习参数生成可自适应形状的激活函数，能连续逼近多种经典激活并对深度与噪声自适应。

**🔧 技术方法**

采用 arctangent 组合的三阶段非线性变换、参数学习与端到端训练。

**📊 数据集**

在 PatternNet、UC Merced Land Use 和 13 频带 EuroSAT MSI 三个遥感分类数据集上评测。

**📈 对比分析**

与 ReLU、Leaky‑ReLU、SiLU、GELU 等基线相比，ArcGate 在 PatternNet 上最高精度 99.67%，且在高斯噪声下保持 26.65% 的优势。

**⚠️ 局限性**

主要限制为额外参数虽极少但仍需层级学习，且在轻量级网络与复杂值网络上的适用性尚待验证。

---

## 368. Bridging Brain and Semantics: A Hierarchical Framework for Semantically Enhanced fMRI-to-Video Reconstruction

**arXiv ID:** 2605.14569 | [PDF](https://arxiv.org/pdf/2605.14569v1)

**作者:** Yujie Wei `[一作]` (Fudan University), Hongming Shan `[通讯]` (Fudan University)

**通讯引用:** 4699 | [OpenAlex ID](https://openalex.org/A5049086157)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种分层的 fMRI‑to‑video 重建框架，先用多模态语义学习把 fMRI 信号映射到丰富的语义空间，再通过 Mixture‑of‑Memories 动态检索并融合记忆，实现端到端的视频重建。

**💡 创新点**

创新点在于：① 将图像、文本、动作与类别四大语义维度同时对齐 fMRI；② 设计 Mixture‑of‑Memories（MoM）模块，结合跨注意力融合实现动态记忆检索与整合；③ 采用 LoRA 微调的 Diffusion Transformer 进行高质量视频生成。

**🔧 技术方法**

技术手段包括 Transformer‑based Brain Model、跨模态对齐与分类任务、Mixture‑of‑Memories 检索‑融合机制、Zero‑conv 与双流融合、LoRA 微调的 Wan2.1 1.3B Diffusion Transformer。

**📊 数据集**

使用公开的 cc2017 与 CineBrain 两个 fMRI‑视频配对数据集进行训练与评估。

**📈 对比分析**

与 MindVideo、NeuroClips、CineSync 等方法在语义一致性、时空一致性、PSNR/SSIM 等指标上进行定量比较，本文方法在两组数据上均显著优于现有方法，特别在语义准确性与运动一致性上表现突出。

**⚠️ 局限性**

局限性包括：① 仍受 fMRI 信号低时空分辨率与噪声限制；② 记忆池规模和检索策略对性能影响较大；③ 模型对大规模训练数据依赖强，跨任务泛化能力尚待验证。

---

## 369. SeesawNet: Towards Non-stationary Time Series Forecasting with Balanced Modeling of Common and Specific Dependencies

**arXiv ID:** 2605.14551 | [PDF](https://arxiv.org/pdf/2605.14551v1)

**作者:** Hao Li `[一作]` (Sichuan University), Yingjie Zhou `[通讯]` (Sichuan University)

**通讯引用:** 3720 | [OpenAlex ID](https://openalex.org/A5037482637)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出SeesawNet框架，利用实例归一化与原始序列的双通道，动态平衡常见与实例特定的时序及通道依赖关系，实现更精准的非平稳多变量时间序列预测。

**💡 创新点**

创新点在于Adaptive Stationary-Nonstationary Attention (ASNA)，通过双分支注意力与自适应门控同时学习归一化后共享模式与原始序列中的实例特定结构，并在时间和通道维度上交替建模。

**🔧 技术方法**

采用实例归一化、Patch Attention、双分支自注意力、卷积时间聚合、以及基于频域误差的损失函数等技术。

**📊 数据集**

在九个真实世界数据集上评估，包括ETT系列、Exchange Rate、Weather、ILI、Solar Energy、ECL 等多种频率和变量规模。

**📈 对比分析**

与包括DDN、SAN、U-Mixer、NS-Transformer、TimeBridge以及PatchTST、iTransformer等多种基线对比，SeesawNet在72个预测长度/数据集组合中取得51个最佳、12个第二好，平均相对提升约10%~18%。

**⚠️ 局限性**

主要局限在于对极度非平稳或跨域迁移的适应性仍有限，且双通道结构导致模型复杂度略高，可能在极大规模时序上产生计算瓶颈。

---

## 370. VerbalValue: A Socially Intelligent Virtual Host for Sales-Driven Live Commerce

**arXiv ID:** 2605.14542 | [PDF](https://arxiv.org/pdf/2605.14542v1)

**作者:** Yuyan Chen `[一作]` (Cornell University), Yuyan Chen `[通讯]` (Cornell University)

**通讯引用:** 253 | [OpenAlex ID](https://openalex.org/A5060201245)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为VerbalValue的虚拟直播电商主持人，能够在直播中持续讲解产品并即时响应观众评论，实现销售转化。

**💡 创新点**

创新点包括：1) 双通道架构实现持续产品宣传与即时互动；2) 结构化产品知识库与成分词典提供事实根基；3) 受意图标签的多槽式监督训练，使模型在不同观众意图下采用不同的销售策略。

**🔧 技术方法**

技术手段包括：以Qwen2.5-32B-Instruct为基准模型，使用LoRA微调；结构化四字段输出（口播、字幕、追问、CTA）；双通道对话服务与重排行为；意图标签与多槽监督训练；产品知识库与成分词典注入。

**📊 数据集**

使用的数据集：1,475条带意图标记的直播互动样本（自然收集+GPT-5.2生成），每条包含四字段目标；12件护肤品的产品知识库及23个成分词典；训练集经过四轮清洗与去重。

**📈 对比分析**

对比GPT‑5.4、Claude Sonnet 4.6、Gemini 3.1 Pro和未微调的Qwen2.5，在人类评估的七维度（信息量、相关性、流畅度、礼貌、正确性、创造性、参与度）上，VerbalValue在信息量+23%、正确性+17%、礼貌+4%，其余维度保持或略优。

**⚠️ 局限性**

局限性：1) 在流畅度上略低于部分基线；2) 受限于知识库范围，无法回答目录外产品；3) 双通道架构实现复杂，实时资源调度挑战；4) 仅在中国美妆直播场景验证，跨域推广效果未知。

---

## 371. Language Generation as Optimal Control: Closed-Loop Diffusion in Latent Control Space

**arXiv ID:** 2605.14531 | [PDF](https://arxiv.org/pdf/2605.14531v1)

**作者:** ZiYi Dong `[一作]` (Sun Yat-sen University), Pengxu Wei `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2682 | [OpenAlex ID](https://openalex.org/A5041227759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将语言生成重新表述为随机最优控制问题，提出 Manta‑LM 框架，利用闭环控制实现高质量并行生成。

**💡 创新点**

创新点在于通过正则化 VAE 构造连续可导的潜在流形，使用流匹配逼近 HJB 解，并将 Transformer 作为全局积分算子，解决 AR 的效率‑质量矛盾和离散扩散的梯度缺失问题。

**🔧 技术方法**

主要技术包括局部积分算子 VAE、流匹配（Conditional Flow Matching）、Transformer 全局积分算子、最优控制理论与 Hamilton–Jacobi–Bellman 方程近似。

**📊 数据集**

使用 OpenWebText、LAMBADA、WikiText‑2/103、PTB、1BW 进行无监督评估，QQP、Quasar‑T、Wiki‑Auto、CCD 进行条件生成评估。

**📈 对比分析**

与 GPT‑2、D3PM、MD4、RADD、DiffuSeq 等 AR、离散/连续扩散模型对比，零样本困惑度最低（1BW 62.55，WikiText‑2 30.58），条件生成 BLEU/ROUGE 与 GPT‑2 相当或更好，推理效率提升约 4‑5 倍、NFE 大幅下降。

**⚠️ 局限性**

局限性包括对需要可中断细粒度序列化任务（如工具调用）不如 AR 高效，潜在空间逼近与 VAE 正则化受限，模型训练成本和复杂度较高。

---

## 372. Mitigating Mask Prior Drift and Positional Attention Collapse in Large Diffusion Vision-Language Models

**arXiv ID:** 2605.14530 | [PDF](https://arxiv.org/pdf/2605.14530v1)

**作者:** Sujung Hong `[一作]` (Yonsei University), Seongjae Hwang `[通讯]` (Yonsei University)

**通讯引用:** 617 | [OpenAlex ID](https://openalex.org/A5022991142)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大规模扩散视觉语言模型（LDVLM）中出现的重复生成和视觉定位衰退问题，提出了两种无训练、轻量级的推理时改进方法：Mask Prior Suppression（掩码先验抑制）和 Monotonic RoPE Scaling（单调旋转位置嵌入缩放）。

**💡 创新点**

创新点在于：①首次将掩码先验方向（词汇均值）投影到最终隐藏层的低维子空间并动态抑制其对生成词的影响，从而有效缓解“mask prior drift”；②通过对RoPE频率维度进行单调增益，增强低频（长距离）位置信息，解决“positional attention collapse”，两者均无额外训练、参数更新。

**🔧 技术方法**

主要技术：掩码先验抑制（Mask Prior Suppression）、单调RoPE缩放（Monotonic RoPE Scaling）、基于RoPE的频率分解与缩放、PCA降维提取掩码先验子空间、与原始LDVLM（如LLaDA‑V、LaViDa）融合。

**📊 数据集**

使用的数据集包括：通用推理评测（MME、MMBench、MMMU）、视觉定位评测（RefCOCOg、Ferret、GQA）、长文本生成评测（LLaVA‑Bench、DetailCaps、MIA）、以及幻觉评测（CHAIR、AMBER‑G）。

**📈 对比分析**

与多种基准模型（LLaDA‑V、LaViDa、LLaVA‑1.6、Qwen2.5、InternVL‑3、A2R模型等）在上述任务上进行对比，实验结果显示：①在视觉定位任务上显著提升（RefCOCOg、Ferret分数提升≈5‑10%）；②在长文本生成任务上减少重复率、提升Distinct‑n分数；③保持或略优于原模型在通用推理任务的性能；整体无需重新训练、参数无增。

**⚠️ 局限性**

局限性：①改进方法依赖于预训练LDVLM的质量；②目前仅在完整后缀迭代解码框架下验证，对半自回归或块式解码等变体的适用性尚待验证；③仅在推理阶段做调整，未改进模型本身的学习机制；④频率缩放参数需要手动设置，对不同模型的通用性略有影响。

---

## 373. Lang2MLIP: End-to-End Language-to-Machine Learning Interatomic Potential Development with Autonomous Agentic Workflows

**arXiv ID:** 2605.14527 | [PDF](https://arxiv.org/pdf/2605.14527v1)

**作者:** Wenwen Li `[一作]` (Preferred Networks, Inc.), Nontawat Charoenphakdee `[通讯]` (Preferred Networks, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个基于自然语言的多智能体框架Lang2MLIP，用来实现机器学习原子势（MLIP）的端到端开发，完全不需要手工设计固定的工作流；

**💡 创新点**

创新点在于将MLIP开发视作一个序贯决策问题，让大型语言模型（LLM）在多智能体协同下自适应地决定采样、训练、评估等步骤，形成自我修正和自生成课程；

**🔧 技术方法**

使用了多智能体架构（任务描述、初始结构、MD脚本、采样、训练、评估、终止等子智能体），核心控制器是一个以LLM为后端的决策代理；

**📊 数据集**

采用锂电池固体电解质间隙（SEI）系统作为测试数据集，使用Preferred Potential（PFP）作为参考数据，生成60个初始结构并在迭代中共采集约7536个样本；

**📈 对比分析**

通过与两种消融实验（手工初始化+无闭环训练；LLM初始化+一次性采样+训练）对比，Lang2MLIP在稳定性、密度、MSD、RDF等指标上均优于消融模型，表明闭环自适应训练能显著提升性能；

**⚠️ 局限性**

局限性包括仅在单一复杂SEI系统上验证，缺乏更广泛的多材料测试；与专家手工调优的最优管线的系统性对比仍待进一步研究；

---

## 374. Let Robots Feel Your Touch: Visuo-Tactile Cortical Alignment for Embodied Mirror Resonance

**arXiv ID:** 2605.14571 | [PDF](https://arxiv.org/pdf/2605.14571v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 375. DiffPhD: A Unified Differentiable Solver for Projective Heterogeneous Materials in Elastodynamics with Contact-Rich GPU-Acceleration

**arXiv ID:** 2605.14526 | [PDF](https://arxiv.org/pdf/2605.14526v1)

**作者:** Shih-Yu Lai `[一作]` (National Taiwan University), Bing-Yu Chen `[通讯]` (National Taiwan University)

**通讯引用:** 9259 | [OpenAlex ID](https://openalex.org/A5047845215)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 DiffPhD，一种统一的 GPU 加速可微分项目化动力学（PD）框架，能够同时处理异质材料、大变形超弹性以及多接触交互，并实现端到端梯度优化。

**💡 创新点**

创新点包括：1）将材料异质性直接编码到 PD 全局刚度矩阵的项目化权重中，保持矩阵条件良好；2）在反向传播中引入自适应信赖域特征值滤波，保证超弹性梯度稳定；3）采用单一持久稀疏逆因子在前向、后向和接触计算中共享，实现 GPU 上一次性分解后的多次重用，显著降低时间成本。

**🔧 技术方法**

主要技术包括：项目化动力学（PD）与可微分扩展 DiffPD、信赖域特征值滤波、Type‑II Anderson 加速、METIS Nested‑Dissection 稀疏逆因子、Signorini–Coulomb 非线性互补性（NCP）接触模型、Rayleigh 动态阻尼、GPU 上的稀疏矩阵向量/矩阵乘法、Jacobi 3×3 特征分解等。

**📊 数据集**

在评估中使用了多种基准场景，涵盖异质软体、超弹性、接触丰富的几何体（如鳗鱼、蟹、软手套、机器人抓取器、软体操控等）以及真实世界的抓取与碰撞数据集（如 UR5 PokeFlex 与 Dice 软体数据）。

**📈 对比分析**

通过与 DiffPD、MAS、Newton‑Cholesky 等现有可微分 PD/接触求解器对比，DiffPhD 在异质、超弹性、接触丰富的情形下实现了 8–30 倍的前向加速和 4–25 倍的后向加速；在系统识别、初始状态优化、轨迹优化以及 Real2Sim 机器人抓取等任务中，达到更低的损失和更快的收敛速度。

**⚠️ 局限性**

局限性包括：1）在完全静态接触或持续支撑场景下，基于速度级 NCP 的接触模型可能产生位移漂移；2）材料参数的反向传播目前仅对 Neo‑Hookean 形式解析，需针对其他能量形式重新推导；3）对极端刚度对比（>10⁸×）仍依赖稀疏逆因子重构，处理更大规模或动态拓扑变化时可能需要改进预条件。

---

## 376. Silent Collapse in Recursive Learning Systems

**arXiv ID:** 2605.14588 | [PDF](https://arxiv.org/pdf/2605.14588v1)

**作者:** Zhipeng Zhang `[一作]` `[通讯]` (China Mobile Research Institute), Zhipeng Zhang (China Mobile Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对递归训练过程中的“静默崩塌”现象进行系统研究，并提出 MTR（Monitor–Trust–Regulator）框架，用以监测模型内部指标并在未获得原始数据的前提下主动调节学习强度。

**💡 创新点**

创新点在于揭示并量化了静默崩塌的三大先导信号（锚点熵收缩、表示漂移冻结、尾部覆盖率下降），以及基于这些先导信号的自适应信任变量实现的动态学习调节机制。

**🔧 技术方法**

核心技术包括：锚点集上的预测熵、表示漂移统计、尾部覆盖率测量；构造慢速时间尺度的信任变量 τ；以及将 τ 用作有效学习率或混合比例的自适应调节。

**📊 数据集**

实验数据集涵盖 TinyStories（小型文本生成）、CIFAR‑10（图像分类）以及对应的 MiniGPT（≈30M 参数）和 ResNet‑18 模型。

**📈 对比分析**

通过与“开环”（open‑loop）递归训练对比，MTR 在 14 代内避免了 perplexity 爆炸（从 68 降到 <12），保持锚点熵在 2.1 以上；在 CIFAR‑10 上，准确率从 37% 上升至 79%，ECE 下降至 0.12，尾部覆盖率恢复至 1.0；MTR 在所有指标上都比开环提前数代发现崩塌并保持稳定。

**⚠️ 局限性**

局限性在于需预先选定固定锚点集（2000 条样本即可），对动态数据分布适应性有限，且实验范围仅覆盖文本生成和伪标签分类，未来需探索自适应锚点更新、跨任务泛化和更大规模的实验验证。

---

## 377. Med-DisSeg: Dispersion-Driven Representation Learning for Fine-Grained Medical Image Segmentation

**arXiv ID:** 2605.14579 | [PDF](https://arxiv.org/pdf/2605.14579v1)

**作者:** Zhiquan Chen `[一作]` (Sun Yat-sen University), Hejun Wu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1436 | [OpenAlex ID](https://openalex.org/A5102755158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为 Med-DisSeg 的两阶段医学图像分割框架，旨在通过消除特征坍塌并增强细粒度解码来提升分割精度。

**💡 创新点**

创新点包括：① 轻量化的 Dispersive Loss（无正样本对，所有批量样本视为负样本）实现特征空间充分分散；② 编码器侧的结构感知自适应注意力（ELAT），兼顾通道与空间信息；③ 解码器侧的多尺度自适应校准（CBAT），在三条不同尺度路径上实现细节与全局形状的协同恢复。

**🔧 技术方法**

技术栈主要包括：ResNet‑50 编码器、ELAT 与 CBAT 轻量级注意力模块、Dispersive Loss（InfoNCE‑L2 变体）、两阶段训练策略、常规数据增强和 Adam 优化器；同时在 2D 和 3D 数据上验证其通用性。

**📊 数据集**

使用了五个 2D 数据集（Kvasir‑SEG、Kvasir‑Sessile、GlaS、ISIC‑2016、ISIC‑2017）和一个 3D 体数据集（Synapse）进行评估，覆盖肠镜、内镜、皮肤病理和腹部 CT 等多种模态。

**📈 对比分析**

与 UNet、UNet++、Attention‑UNet、PraNet、TGANet、DCSAUNet、XBoundFormer、CASFNet、EIUNet、DTAN、ConDSeg 等 SOTA 方法在同一实验设置下对比，Med‑DisSeg 在所有数据集上均取得最高的 mIoU / mDSC，平均提升约 1.3%（最高 3.4%），并在 3D Synapse 上实现 83.4% DSC，位居第二。

**⚠️ 局限性**

局限性：① 仍依赖较大的 batch 以保证 Dispersive Loss 的效果；② 主要针对二分类分割任务，虽然在 3D 多类实验中表现良好，但对更复杂的多标签分割尚未深入验证；③ 计算量与参数规模与 ConDSeg 相当，仍可进一步压缩以适应资源受限环境。

---

## 378. Mining Subscenario Refactoring Opportunities in Behaviour-Driven Software Test Suites: ML Classifiers and LLM-Judge Baselines

**arXiv ID:** 2605.14568 | [PDF](https://arxiv.org/pdf/2605.14568v1)

**作者:** Ali Hassaan Mughal `[一作]` (Texas Wesleyan University), Muhammad Bilal `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了一个静态、可应对同义词的子序列挖掘管道，自动识别BDD Gherkin测试套件中可提取的连续步骤子序列，并为每个子序列预映射三种现有的重构模式；

**💡 创新点**

提出了首个基于步骤层级聚类的同义词鲁棒子序列挖掘方法，并将其与机器学习预测和LLM裁判相结合，形成完整的可重复使用机会评估流程；

**🔧 技术方法**

使用SBERT+UMAP+HDBSCAN进行语义嵌入聚类，SBERT句向量；使用XGBoost训练二分类提取性模型和三分类机制预测模型；基于规则的基线与LLM裁判对比；

**📊 数据集**

在包含1.1M个Gherkin步骤的347个公开GitHub仓库（339个实际含子序列）的语料库上进行挖掘；

**📈 对比分析**

与规则基线和两种开源LLM裁判（120B和1T参数MoE）比较，XGBoost提取性模型在5折交叉验证下实现F1≈0.891（95% CI [0.852,0.927]），明显优于规则基线（F1≈0.836，p=0.017）和LLM裁判（F1≤0.728，p<10⁻⁴）；机制预测准确率≈0.965，与规则基线基本一致；

**⚠️ 局限性**

限制包括：聚类误差可能导致语义冲突；标签主观性与人力标注瓶颈；仅覆盖Cucumber-Java实现，其他BDD方言需进一步适配；未验证行为等价性；仅在公开GitHub仓库上评估，难以推广到企业私有仓库。

---

## 379. Complacent, Not Sycophantic: Reframing Large Language Models and Designing AI Literacy for Complacent Machines

**arXiv ID:** 2605.14544 | [PDF](https://arxiv.org/pdf/2605.14544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 380. Prompt Segmentation and Annotation Optimisation: Controlling LLM Behaviour via Optimised Segment-Level Annotations

**arXiv ID:** 2605.14561 | [PDF](https://arxiv.org/pdf/2605.14561v1)

**作者:** Devika Prasad `[一作]` (Commonwealth Bank of Australia), Luiz Pizzato `[通讯]` (Commonwealth Bank of Australia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Prompt Segmentation and Annotation Optimisation (PSAO)，通过将提示拆分为可解释的段落并为每段分配可读注解，来实现对大型语言模型的可控优化。

**💡 创新点**

创新点在于：①将提示空间结构化为段落+注解搜索空间，显著降低搜索复杂度；②通过注解引导模型关注关键信息而不改写原始提示；③提供理论保证（原始提示可被保留、细分不降效、可与任意优化器组合）。

**🔧 技术方法**

技术包括：自然语言分段算法、注解函数（如重要性等级、上下文提示）、基于黑盒评估的迭代搜索（如随机/启发式采样）以及性能评估框架。

**📊 数据集**

实验使用多种通用推理与自一致性任务（如算数推理、逻辑判断等），在公开的 LLM 评测数据集上进行（如 ARC、CommonsenseQA、CoQA 等）。

**📈 对比分析**

与基线无系统提示或系统提示的对比实验表明，PSAO 在大部分任务上提升了约 30‑50% 的准确率（如从 3/10 提升到 8/10），并在推理和自一致性方面表现出显著改善。

**⚠️ 局限性**

局限性：当前仅在固定分段下搜索注解，缺乏联合分段与注解的高效学习方法；注解增大了 token 消耗；对最优分段策略的理论与算法研究仍待深入。

---

## 381. Break-the-Beat! Controllable MIDI-to-Drum Audio Synthesis

**arXiv ID:** 2605.14555 | [PDF](https://arxiv.org/pdf/2605.14555v1)

**作者:** Shuyang Cui `[一作]` (Sony Group Corporation), Shusuke Takahashi `[通讯]` (Sony Group Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可控的 MIDI‑to‑Drum 音频合成模型，利用参考音频的音色与 MIDI 节奏相结合，生成高保真鼓音频。

**💡 创新点**

创新点包括：① 将 Stable Audio Open 的 Diffusion Transformer 改造成双输入（MIDI + 参考音频）模型；② 设计了内容编码器（自注意力 + 交叉注意力）和混合条件机制；③ 构建了鼓 MIDI‑音频配对数据集，实现 MIDI 作为节奏控制、音频作为音色控制的统一框架。

**🔧 技术方法**

使用的技术主要有：Diffusion Transformer (DiT)、VAE 编码器、内容编码器、Hybrid Conditioning（拼接、加法、前置）、SAO 预训练、DPM‑Solver++ 采样器。

**📊 数据集**

数据集：Groove MIDI Dataset（GMD）及其扩展版 E‑GMD、StemGMD，基于这些数据构造了鼓音频与 MIDI 的配对样本。

**📈 对比分析**

评估方法：FAD（VGG, CLAP）、Onset F1、RMS 误差、Beat Continuity（CMLt、AMLt）等；实验显示 64 分辨率的 MIDI 量化能显著提升 FAD 降低、F1 提升，模型在不同排列类型和数据集上均保持较高质量，且相比从零训练显著优越。

**⚠️ 局限性**

局限性：需要参考音频的真实或伪 Tap 信息；在极度多乐器混合或复杂节奏时对齐略逊；缺乏自动生成鼓谱的功能，依赖预训练模型和大量配对数据。

---

## 382. Efficient Multi-objective Prompt Optimization via Pure-exploration Bandits

**arXiv ID:** 2605.14553 | [PDF](https://arxiv.org/pdf/2605.14553v1)

**作者:** Donghao Li `[一作]` (University of Virginia), Jing Yang `[通讯]` (University of Virginia)

**通讯引用:** 471985 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将多目标提示工程问题建模为固定预算下的纯探索多目标bandit框架，并针对最佳可行提示识别和Pareto提示集合识别设计了两个通用算法Gensec和GenPSI。

**💡 创新点**

创新点包括：①首次正式化多目标提示选择为多目标纯探索bandit问题；②提出Gensec与GenPSI，利用共享结构（线性模型或MLP）实现高效样本利用，并给出理论误差上界；③在最佳可行提示识别中结合约束的最佳arm识别，显著降低对arm数量的误差依赖。

**🔧 技术方法**

使用的技术主要有：纯探索多目标bandit策略（Successive Rejection、Sequential Halving）、G-optimal设计、线性最小二乘估计、神经网络（单隐藏层ReLU MLP）共享结构、PCA特征降维、超体积（HV）评估。

**📊 数据集**

实验数据集为文本摘要任务的XSum和CNN/DailyMail，使用的LLM为LLaMA‑3‑8B‑instruct和Gemma‑7B‑it。

**📈 对比分析**

与均匀采样、CSR、EGE等基线对比。Gensec在最佳可行提示识别中在不同预算下能恢复80–90%的可行目标；GenPSI在Pareto集合识别中恢复超过90%的超体积；相较基线仅能恢复低至中等80%HV，整体性能显著提升。

**⚠️ 局限性**

局限性：仅考虑两目标，线性或MLP模型假设可能不适用于所有任务；在极低预算下仍需大量评估；实验仅覆盖摘要任务和两种LLM，缺乏对更广泛任务、多样指标及大规模部署的验证。

---

## 383. Cattle Trade: A Multi-Agent Benchmark for LLM Bluffing, Bidding, and Bargaining

**arXiv ID:** 2605.14537 | [PDF](https://arxiv.org/pdf/2605.14537v1)

**作者:** Robert Müller `[一作]`, Clemens Müller `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

创建并评估了一个名为 Cattle Trade 的多代理基准，用于测试大型语言模型（LLM）在包含拍卖、隐藏报价挑战、欺骗和资源约束的长周期游戏中的战略推理与资源管理能力。

**💡 创新点**

创新点在于：①将多种战略子能力（拍卖、双边贸易、诈唬、资源管理）合并到同一长周期、多玩家的环境中；②提供完整的行动日志，支持对模型失效模式（如过投、同向投、破产发起）进行行为诊断；③通过 TrueSkill 排名和多维度行为指标（资本效率、TC紧迫性等）评估整体战略连贯性。

**🔧 技术方法**

使用技术包括：Python 编写的游戏引擎与三种拍卖模式；LLM 代理框架，采用自然语言输入与结构化 JSON 输出；TrueSkill 进行对战排名；对比三种确定性代码代理（TrackerAgent、SetRaceAgent、EconomyAgent）；构造多维度行为指标并可视化；使用多场实验（242 场对局）收集数据。

**📊 数据集**

数据集为 242 场自生成的 Cattle Trade 对局，包含四位玩家、完整卡牌与金币规则；没有外部公开数据集，所有游戏在相同规则与参数下产生。

**📈 对比分析**

比较方法：对同一组七种 LLM 与三种代码代理进行纯 LLM 对局与混合对局，使用 TrueSkill μ 进行排名；分析每个代理的胜率、平均分、资源使用效率、TC 争议率等指标。结果显示 G3‑F 以 μ≈30.1 位居首位，TrackerAgent 在混合对局中排名第二，三种代码代理整体优于大多数 LLM，且行为指标与最终表现高度相关。

**⚠️ 局限性**

局限性：仅测试低成本、低推理努力的 LLM（最大 4,096 token），Sonnet 仅有 14 场样本；未探测温度、记忆长度对表现的影响；行为指标缺乏置信区间；缺少人类对比与多方聊天机制；日志分析仅基于规则驱动的游戏，无法验证模型在更广泛经济场景中的通用性。

---

## 384. HASTE: Training-Free Video Diffusion Acceleration via Head-Wise Adaptive Sparse Attention

**arXiv ID:** 2605.14513 | [PDF](https://arxiv.org/pdf/2605.14513v1)

**作者:** Xuzhe Zheng `[一作]` (Xiamen University), Fei Chao `[通讯]` (Xiamen University)

**通讯引用:** 3339 | [OpenAlex ID](https://openalex.org/A5000389309)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出两种针对视频扩散变换器（Video DiT）的训练‑免费稀疏注意力改进方案：一是Temporal Mask Reuse（TMR），通过查询‑键漂移统计自适应决定每个头是否重新预测稀疏掩码；二是Error‑guided Budgeted Calibration（EBC），离线测量各头在不同top‑p阈值下的误差与稀疏度，并通过整数线性规划在全局稀疏预算下选择头级阈值，使用频域加权的3D‑FFT误差作为校准目标。

**💡 创新点**

创新点在于（1）实现头级的动态掩码重用，避免全局固定刷新，显著降低在线掩码预测开销；（2）将头级top‑p阈值视为可选操作点，采用基于误差与稀疏度的离线预算分配，克服共享阈值导致的效率与质量不匹配；（3）引入频域加权误差衡量，更好地捕捉稀疏化对视频质量的实际影响。

**🔧 技术方法**

核心技术包括：查询‑键漂移统计（query‑key drift）做掩码重用判定；top‑p稀疏注意力机制；离线阈值候选测量（只稀疏单个头的前向推理）；基于频域的加权3D‑FFT误差计算；整数线性规划（ILP）求解预算分配；以及在现有稀疏核（如XAttention、SVG2）上无缝插拔的框架。

**📊 数据集**

使用Wan2.1-1.3B和Wan2.1-14B两大预训练Video DiT模型，分别在480P和720P分辨率下评估，生成81帧视频，采用81步采样。

**📈 对比分析**

与稠密基线以及原始top‑p稀疏方法（XAttention、SVG2）进行对比。结果显示，加入TMR+EBC后，XAttention的VBench分数从75.89%提升至76.51%，速度提升从1.30×提升至1.49×；SVG2的VBench从76.79%提升至77.00%，速度从1.15×提升至1.25×；在720P下，XAttention速度提升从1.71×至1.93×，保持或提升相似度指标。

**⚠️ 局限性**

局限性包括：需为每个模型进行离线校准，耗时且需额外存储；仅适用于top‑p稀疏框架，未验证对其他稀疏策略的通用性；对极端长视频或更高分辨率的推理时间仍未充分评估；TMR的阈值选择仍需要经验调优，可能在不同硬件/工作负载下表现不一致。

---

## 385. Unbiased and Second-Order-Free Training for High-Dimensional PDEs

**arXiv ID:** 2605.14643 | [PDF](https://arxiv.org/pdf/2605.14643v1)

**作者:** Jaemin Seo `[一作]` (Chung-Ang University), Jae Yong Lee `[通讯]` (Chung-Ang University)

**通讯引用:** 9173 | [OpenAlex ID](https://openalex.org/A5100369464)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无偏、无二阶导数的 BSDE 训练框架（Un-EM-BSDE），用于高维 PDE 的数值求解。

**💡 创新点**

创新点在于使用随机化交叉样本估计，将 EM‑BSDE 的离散化偏差消除，同时保持仅依赖梯度的计算，避免了高阶导数的开销。

**🔧 技术方法**

主要技术包括：Euler–Maruyama 与 Heun 的 BSDE 离散、Shotgun 多样本平均、随机交叉乘积无偏估计、自动微分求梯度、软/硬约束终端条件。

**📊 数据集**

实验数据集：标准高维 PDE 基准（HJB、Black‑Scholes‑Barenblatt、Allen‑Cahn、BZ、PIDE）以及相应的随机模拟轨迹。

**📈 对比分析**

与 EM‑BSDE、FS‑PINNs、Heun‑BSDE、Shotgun、Multi‑Shot EM‑BSDE 等方法对比，Un‑EM‑BSDE 在保持与 EM‑BSDE 相近的训练时间的同时，获得与 Heun‑BSDE、FS‑PINNs 相近甚至更好的相对 L₂误差；在高维场景下显著降低了二阶导数计算成本。

**⚠️ 局限性**

局限性：理论证明基于光滑有界假设，未覆盖强耦合 FBSDE 或 PIDE 的严格分析；无偏估计在小样本或高噪声情况下可能产生较大方差；对 M₁、M₂ 等超参数的敏感性需要进一步调优。

---

## 386. MambaRain: Multi-Scale Mamba-Attention Framework for 0-3 Hour Precipitation Nowcasting

**arXiv ID:** 2605.14606 | [PDF](https://arxiv.org/pdf/2605.14606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 387. Slower Generalization, Faster Memorization: A Sweet Spot in Algorithmic Learning

**arXiv ID:** 2605.14659 | [PDF](https://arxiv.org/pdf/2605.14659v1)

**作者:** Shin So `[一作]` (Yonsei University), Albert No `[通讯]` (Yonsei University)

**通讯引用:** 527 | [OpenAlex ID](https://openalex.org/A5049196468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在小型Transformer训练 Needleman–Wunsch（NW）矩阵生成任务时，数据集规模如何影响训练与验证的时间差距，尤其是验证精度达到阈值所需的优化器更新次数。

**💡 创新点**

发现NW任务出现“甜点效应”：在中等规模数据集下验证精度最快达到阈值，而更大的数据集虽然能实现泛化，却需要更多更新；并通过随机后缀探测到规则学习与残差拟合的双重压力。

**🔧 技术方法**

使用自回归序列化的Decoder-only Transformer、AdamW优化器、余弦学习率调度、批量累积（160）以及固定的更新次数计量方法。

**📊 数据集**

实验数据集包括NW矩阵生成（L=5，长度10符号）以及三位数乘法作为对照；对不同数据集大小（200到100k）和模型深度（3–6层）进行 sweep。

**📈 对比分析**

通过绘制不同阈值下训练与验证首次达到阈值所需的更新次数曲线，比较数据集大小对速度的影响；结果显示NW在中等N处最优，而乘法任务无此现象。

**⚠️ 局限性**

局限性包括：仅在固定的训练协议和更新计数度量下评估；数据集规模和模型规模有限；序列化顺序和输出格式可能影响结果；结果可能不易推广到更大模型或自然语言推理等任务。

---

## 388. Action-Inspired Generative Models

**arXiv ID:** 2605.14631 | [PDF](https://arxiv.org/pdf/2605.14631v1)

**作者:** Eshwar R. A. `[一作]` (PES University), Debnath Pal `[通讯]` (Indian Institute of Science)

**通讯引用:** 3120 | [OpenAlex ID](https://openalex.org/A5026479756)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Action-Inspired Generative Models (AGM)，通过在桥匹配训练中加入轻量级潜能网络V_ϕ，对桥样本进行重要性加权，从而提升生成模型的质量。

**💡 创新点**

创新点在于使用stop‑gradient的潜能网络为桥路径赋予结构信息权重，既实现了路径感知训练，又不增加推理时的计算或参数负担。

**🔧 技术方法**

主要技术包括桥匹配（bridge matching）、轻量级scalar潜能网络、重要性加权、EMA权重、Euler–Maruyama数值积分、U‑Net架构以及自注意力模块。

**📊 数据集**

实验基于CelebA‑HQ 64×64人脸图像数据集。

**📈 对比分析**

在单GPU 500k步的对照实验中，加入V_ϕ后FID从28.53降至25.47（下降10.7%），精度提升2.8%，召回提升2.1%，表明相较于无V_ϕ基线取得了显著性能提升。

**⚠️ 局限性**

限制在于仅在单一低分辨率数据集上验证，未对更高分辨率或多模态数据进行扩展，且受GPU资源限制，难以进一步扩大规模。

---

## 389. ViMU: Benchmarking Video Metaphorical Understanding

**arXiv ID:** 2605.14607 | [PDF](https://arxiv.org/pdf/2605.14607v1)

**作者:** Qi Li `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**通讯引用:** 13757 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并构建了ViMU（Video Metaphorical Understanding）基准，用于评估视频模型在隐含意义、修辞手法、社会立场等非显式子文本理解上的能力。

**💡 创新点**

创新点在于：①设计了无提示的多任务评估，覆盖开放式解释、修辞机制识别、社会价值信号识别和证据归纳；②通过多轮人工与闭源模型校验确保任务难度与真值质量；③系统分析模型在隐式推理中的误差结构与类别偏置，揭示现有模型在隐含意义推理方面的普遍不足。

**🔧 技术方法**

主要技术包括：大型多模态语言模型（16个，包括开源与闭源），LLM-as-a-Judge评估方法，多轮注释与验证流程，基于PCA与Frobenius距离的误差谱分析，以及针对性提示指导实验。

**📊 数据集**

使用了自采集的588段视频（来自YouTube、Bilibili、TikTok），共2,352个问题，覆盖10+修辞机制、10+社会价值信号，涉及多模态证据（视觉、文本、音频）与目标主体。

**📈 对比分析**

通过与16个模型的零样本评估，发现即使在最优模型（如GPT‑5.2）也仅在开放式解释与证据归纳上达到约70%，而修辞与社会价值信号识别的平均准确率低于30%；总体平均分在50%以下，显示显著性能差距。

**⚠️ 局限性**

局限性包括：①基准对不同语言与文化背景的普适性尚待验证；②数据集规模相对有限，可能无法覆盖所有复杂修辞与社会语境；③评估仍依赖LLM判定，可能存在标注者主观偏差；④缺乏持续更新机制，难以跟踪模型快速迭代的性能变化。

---

## 390. Quaternary codes with new parameters from two-generator simplicial complexes

**arXiv ID:** 2605.14603 | [PDF](https://arxiv.org/pdf/2605.14603v1)

**作者:** Ankit Yadav `[一作]` (Indian Institute of Technology Delhi), Ritumoni Sarma `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 125 | [OpenAlex ID](https://openalex.org/A5035273961)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用Z₄上两生成元单纯形复合物的截断及其补集构造了新的四元线性码，并给出了其Lee重量分布；

**💡 创新点**

创新点在于首次通过截断两生成元单纯形复合物的定义集获得Lee重量投影码，并改造定义集得到投影四元码；此外给出了Gray映像线性与最小码的充分必要条件，产生了Griesmer和最小二进码族；

**🔧 技术方法**

采用CD构造法、单纯形复合物理论、Lee重量公式、Gray映射以及Ashikhmin‑Barg最小码判据等技术；

**📊 数据集**

与公开的最佳四元码数据库（citation）对照，检验新码参数，列出长度≤128的改进与新码；

**📈 对比分析**

通过与数据库最佳码比较，论文共发现32条新的或改进的四元码，其中10条Plotkin‑optimal、10条投影码与现有最佳码相同但因投影性可能更优；Gray映像产出的二进码满足Griesmer界并多为最小码；

**⚠️ 局限性**

局限性包括：仅在长度≤128内检验；对投影码的线性性与Gray映像线性性仅给出充分条件；所用定义集需满足特定交并约束，限制了构造的普适性。

---

## 391. Are Candidate Models Really Needed for Active Learning?

**arXiv ID:** 2605.14689 | [PDF](https://arxiv.org/pdf/2605.14689v1)

**作者:** Harshini Mridula Mohan `[一作]` (RV University), Nitin Cheekatla `[通讯]` (RV University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探讨深度主动学习中是否真的需要候选模型，并提出一种使用随机初始化网络进行样本选择的无候选模型框架。

**💡 创新点**

创新点在于：①证明随机权重模型即可提供有效的样本选择，消除候选模型的训练开销；②提出三种基于置信度的采样策略（高置信度HC、低置信度LC、混合HCLC），其中LC策略在多种数据集和网络上表现最佳；③通过广泛实验验证该方法在精度、效率和鲁棒性方面与传统候选模型方法相当甚至更优。

**🔧 技术方法**

技术方法包括：随机初始化深度网络（CNN、Transformer、DinoV2等），基于模型输出的最大/最小概率作为置信度度量进行采样；迭代主动学习流程（采样→标注→训练），无需预训练候选模型；对比实验采用与候选模型相关的多种基线（BADGE、GLISTER、Coreset、BAIT等）。

**📊 数据集**

使用的数据集涵盖图像分类（CIFAR‑10、CIFAR‑100、SVHN、TinyImageNet）和目标检测（PASCAL VOC 2012），并在多种网络架构（DenseNet‑121、ResNet‑56/18、VGG‑16、MobileNet‑V2、Swin Transformer、ViT‑Small、DinoV2）上进行验证。

**📈 对比分析**

比较方法：将无候选模型的HC/LC/HCLC与传统基线进行同条件下的训练时间、注释仿真时间和最终准确率对比。实验结果显示：LC策略在CIFAR‑10/100、SVHN、TinyImageNet中可获得与或优于候选模型方法的准确率（最高提升≈1.5%），同时节省了候选模型训练时间（0.1–1.2小时不等），整体效率提升显著；在检测任务中，LC也取得最高mAP（81.53%）。

**⚠️ 局限性**

局限性：①在极端类别不平衡（10:1）下，LC和HCLC仍受影响，需结合其他平衡策略；②方法仅基于置信度，缺乏多样性度量，可能在某些数据分布上过度集中；③实验集中于视觉任务，尚未验证在自然语言、时序数据等其他领域的泛化；④在大规模数据集或复杂网络上，随机初始化模型的采样稳定性仍需进一步研究。

---

## 392. AQKA: Active Quantum Kernel Acquisition Under a Shot Budget

**arXiv ID:** 2605.14672 | [PDF](https://arxiv.org/pdf/2605.14672v1)

**作者:** Jian Xu `[一作]` (RIKEN iTHEMS), Qibin Zhao `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于闭式梯度的配对级别激活分配方法AQKA，并在预算受限的量子核估计中实现了显著提升。

**💡 创新点**

创新点包括：①给出了KRR/SVM下的KKT最优配对级别分配闭式 s_ij∝|g_ij|√(K_ij(1−K_ij))；②推导了纠正稀疏率 ρ≤2m/N 的Cauchy–Schwarz 边界；③结合目标填充与探索策略的离散实现；④提出了AQKA–Nyström混合策略。

**🔧 技术方法**

使用了量子特征映射、量子核估计、梯度闭式推导、A‑optimal实验设计理论、Cauchy–Schwarz 边界、离散目标填充、探索‑利用分配、KRR/SVM解析解、正定投影、热启动与多轮在线更新等技术。

**📊 数据集**

实验数据包括：合成植入稀疏 RBF 核（N=225, m=10）、无噪声 4‑qubit ZZ‑FeatureMap（N=150, m=10）、以及从 IBM 156‑qubit Heron 设备采集的真实硬件核（N=50, m=4）并在该核上做重采样与多种种子实验。

**📈 对比分析**

与均匀分配、Nyström‑QKE、ShoFaR、重要性采样、经典 leverage 样本等方法对比，AQKA 在硬件重采样下 B=n_pairs 时提升 26.3±6.1 分、B=4n_pairs 提升 31.9±7.2 分；在 156‑qubit 设备在线多种子实验中 B=4n_pairs 获得 +17.0±4.8 分；在合成数据中 B≈10⁵ 提升 +10–24 分，N=1000 时提升 25 分；均匀分配在高预算下逐渐收敛。

**⚠️ 局限性**

仅在预算受限区间 B≲16 n_pairs 上表现优越，超过该阈值 Nyström‑QKE 或均匀分配更好；目标填充在单一预算窗口内才有优势；高阶泰勒误差导致极限性能受限；算法需要温启动与探索比例调节，对噪声与稀疏度假设有依赖。

---

## 393. Monitoring Data-aware Temporal Properties (Extended Version)

**arXiv ID:** 2605.14666 | [PDF](https://arxiv.org/pdf/2605.14666v1)

**作者:** Alessandro Gianola `[一作]` (Universidade de Lisboa), Sarah Winkler `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5043097301)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了数据感知线性时间属性（LTL+SMT）在有限轨迹上的前瞻性监控，提出了一个能够自动构造监视器的框架；

**💡 创新点**

创新点在于：①首次把前瞻性监控扩展到一阶数据属性；②通过核心可达图和回归约束与模型完成技术实现可判定性；③给出了若干可判定片段（无环签名、单变量单调约束、本地有限性、有限视窗）并证明其可解性；

**🔧 技术方法**

使用技术包括：自动机（NFA）构造、SMT求解（CVC5）、量化消除/模型完成、核心可达图、回归约束；

**📊 数据集**

主要使用了一个在线售票系统的示例数据（票价、演唱会标识、数据库表）以及人工构造的实验轨迹；

**📈 对比分析**

工具实现能在示例轨迹上生成监视状态，初步评估显示监视器可在合理时间内完成，但与现有工具（Lola、TeSSLa、Eagle）没有给出系统性的性能对比；

**⚠️ 局限性**

局限性包括：仅适用于可判定片段，假设数据库为只读且不支持更新；模型完成的前提限制了可处理的理论类型；实验规模有限，缺乏大规模真实数据的评估。

---

## 394. Malleable Molecular Dynamics Simulations with GROMACS and DMR

**arXiv ID:** 2605.14655 | [PDF](https://arxiv.org/pdf/2605.14655v1)

**作者:** Petter Sandås `[一作]` (Barcelona Supercomputing Center), Antonio J. Peña `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 1495 | [OpenAlex ID](https://openalex.org/A5000573036)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将 GROMACS 与 DMR 中间件集成，实现 MPI 进程可动态增删，利用 GROMACS 原生检查点/重启保证状态一致性。

**💡 创新点**

创新点在于提供一种最小侵入式的集成方案，并结合通信效率目标策略实现基于 CE 的自适应重配置，首次将 GROMACS 转为可弹性化。

**🔧 技术方法**

采用 DMR API、TALP 监测 CE、MPI+OpenMP、CUDA GPU 加速、GROMACS 原生检查点/重启、Slurm 调度。

**📊 数据集**

使用 STMV（卫星烟草花斑病毒）模拟，1,066,628 原子、2 fs 时间步、5,000 步。

**📈 对比分析**

通过在 MN5 上提交 10 个 1–12 节点可弹性化工作负载，与静态 2 节点和 12 节点配置对比，测得可弹性化运行时间 2,236 s、节点小时 17.15 n-h，明显优于静态 2 节点的 2,825 s、15.63 n-h，且比静态 12 节点的 2,652 s、17.53 n-h 更省资源。

**⚠️ 局限性**

局限性包括仅在单一超级计算机（MN5）和小规模 STMV 实验上验证；未实现数据重分布，仅依赖重启；重配置开销约 8.24% 运行时间；无法充分展示大规模分布式模拟的优势。

---

## 395. Beyond Instance-Level Self-Supervision in 3D Multi-Modal Medical Imaging

**arXiv ID:** 2605.14654 | [PDF](https://arxiv.org/pdf/2605.14654v1)

**作者:** Tan Pan `[一作]` (Fudan University), Mahsa Baktashmotlagh `[通讯]` (University of Queensland)

**通讯引用:** 1961 | [OpenAlex ID](https://openalex.org/A5014648528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究了一种名为TACO的自监督预训练框架，通过跨个体和跨模态的拓扑一致性来学习3D多模态医学图像的通用表征。

**💡 创新点**

创新点在于提出IM-NRC原则，联合利用intra-instance和inter-instance拓扑一致性；通过MNN伪对应实现跨患者的局部邻域对齐，突破传统单个实例的自监督限制。

**🔧 技术方法**

采用Swin Transformer作为主干网络，结合重建损失、跨模态triplet loss、MNN伪对应匹配以及Top_ω邻域策略，实现跨模态和跨个体的特征对齐。

**📊 数据集**

预训练使用16022张3D扫描、3755名患者的8种模态（T1/T1ce/T2/FLAIR/DWI/ADC/MRA/PD）；下游评估涵盖7个基准任务，包括4个分割（BraTS2023-PED、BraTS2023-MET、UPENN-GBM、ISLES22）、2个分类（ADNI、ADHD-200）以及缺模态分割（BraTS2018）。

**📈 对比分析**

与多种医学SSL基线（SwinMM、Swin UNETR、VoCo、S^2DC、M^3AE、BrainMVP）进行对比，TACO在分割任务平均Dice提升约1.1%，在分类任务平均ACC/AUC/F1提升约0.56%/1.35%/5.94%；在缺模态分割任务中表现最优，鲁棒性和稳定性最高。

**⚠️ 局限性**

限制在于仅针对脑部MRI验证，拓扑一致性假设在结构差异大或姿态变化显著的全身影像（如PET-CT）可能不成立；未扩展到更多模态和解剖结构，需要进一步验证。

---

## 396. How to Evaluate and Refine your CAM

**arXiv ID:** 2605.14641 | [PDF](https://arxiv.org/pdf/2605.14641v1)

**作者:** Luca Domeniconi `[一作]` (University of Bologna), Samuele Salti `[通讯]` (University of Bologna)

**通讯引用:** 5005 | [OpenAlex ID](https://openalex.org/A5055759201)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个可用于评估卷积神经网络可解释性的完整框架，包含一个带有像素级真值的合成数据集、一个新的评估指标 ARCC，以及一个利用多层信息改进 CAM 分辨率的 RefineCAM 方法。

**💡 创新点**

创新点在于：① 用设计精细的合成数据集消除了真值缺失的难题；② 将 ROAD、Coherency 与 Complexity 三个互补维度融合成 ARCC，显著提高了对伪解释的识别能力；③ 通过对浅层 CAM 结果与深层语义特征的逐元素乘积实现高分辨率且更可靠的解释。

**🔧 技术方法**

技术包括：CAM 生成（Grad‑CAM++, LayerCAM, ScoreCAM 等）；评估指标实现（Average Drop, Complexity, Coherency, ROAD, ADCC, ARCC）；多层属性融合（按深度层级进行元素乘积）；实验用 4 种网络（ResNet18, VGG11, Swin‑Tiny, ConvNeXt‑Tiny）。

**📊 数据集**

使用的主要数据集是自制的六种几何形状（圆、方、三角，实填/空）叠加在 Where’s Waldo 背景上的合成图像；并在 FunnyBirds 合成数据集和 ImageNet 真实图像上进行额外验证。

**📈 对比分析**

评估方法是将 ARCC 等指标与真实的 Cosine 相似度（合成数据）和 FunnyBirds 评分进行 Pearson/Spearman 相关性对比；结果显示 ARCC 在所有模型和数据集上均显著优于 AD、ADCC、ROAD，且 RefineCAM 在 CAM 基础方法上实现了 ARCC 分数的最大提升，表明其在提升高分辨率解释质量方面有效。

**⚠️ 局限性**

局限性包括：合成数据集在语义与纹理多样性上不足，难以完全覆盖自然场景的复杂性；RefineCAM 只能对已有 CAM 结果进行后处理，无法补全缺失的重要区域；计算量相对传统 CAM 方法略高。

---

## 397. Do We Really Need External Tools to Mitigate Hallucinations? SIRA: Shared-Prefix Internal Reconstruction of Attribution

**arXiv ID:** 2605.14621 | [PDF](https://arxiv.org/pdf/2605.14621v1)

**作者:** Tian Qin `[一作]` (Tsinghua University), Lijie Wen `[通讯]` (Tsinghua University)

**通讯引用:** 4589 | [OpenAlex ID](https://openalex.org/A5030845033)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种训练无关的内部对比解码框架 SIRA，用于减少大型视觉‑语言模型的幻觉生成。

**💡 创新点**

创新点在于：1）在同一模型内部共享前缀计算，仅在中间层分支；2）通过遮蔽后续层的图像注意力，构造一个语言先验主导的内部对照；3）无需对输入进行扰动或额外完整前向传播，极大降低计算开销。

**🔧 技术方法**

采用的技术包括：共享前缀多层 Transformer 计算、后续层图像掩码、对比 logits 计算并加权（α 参数控制对比强度）、自回归解码与对比融合。

**📊 数据集**

实验使用的视觉‑语言数据集包括 POPE、CHAIR 与 AMBER；基准模型为 Qwen2.5‑VL 与 LLaVA‑v1.5。

**📈 对比分析**

与 DoLa、OPERA、VCD、ICT、MaskCD、SID 等现有推理时幻觉抑制方法对比，SIRA 在所有三大基准上均获得最佳或接近最佳结果，显著降低幻觉率，同时保持甚至提升覆盖率、准确率和 F1 分数，且计算成本低于二次前向传播的对比方法。

**⚠️ 局限性**

局限性：仍需在解码时额外执行后续层的计算，虽然成本低于传统对比方法，但仍高于单一前向；并且需要白盒访问模型内部隐藏状态、注意力掩码和缓存，无法直接在封闭 API 上使用。

---

## 398. Landscape-Aware Bandit Hyper-Heuristics for Online Operator Selection in UAV Inspection Routing

**arXiv ID:** 2605.14620 | [PDF](https://arxiv.org/pdf/2605.14620v1)

**作者:** Junhao Wei `[一作]` (Macao Polytechnic University), Xu Yang `[通讯]` (Macao Polytechnic University)

**通讯引用:** 3077 | [OpenAlex ID](https://openalex.org/A5100462079)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了LA-BHH——一种景观感知的Bandit超启发式，用于UAV多点巡视路径规划中在线学习操作符选择。

**💡 创新点**

创新点在于结合静态景观描述与在线搜索状态特征的上下文Bandit控制器，并通过2-opt修复门控和动态状态适应提升决策质量。

**🔧 技术方法**

采用LinUCB上下文多臂赌博机、四种路由邻域操作符（2-opt、swap、relocate、Or-opt-2）、候选池采样、贪婪接受及奖励更新等技术。

**📊 数据集**

使用由5种景观族（均匀、聚集、走廊、网格抖动、混合密度）生成的45个Euclidean TSP实例。

**📈 对比分析**

与最优构造、随机超启发式、UCB无上下文、经典局部搜索、遗传算法、SA、ILS等7种基线比较，平均最终相对差距0.0223、收敛AUC0.0389，比UCB-HH提升17.6%、比随机超启发式提升22.6%、比NN提升68.2%。

**⚠️ 局限性**

局限在于仅在合成Euclidean实例上验证，未考虑真实UAV地图中的障碍、服务时间、能耗约束；单轮在线学习未利用跨实例记忆，可能在真实场景中表现下降。

---

## 399. Sycophancy is an Educational Safety Risk: Why LLM Tutors Need Sycophancy Benchmarks

**arXiv ID:** 2605.14604 | [PDF](https://arxiv.org/pdf/2605.14604v1)

**作者:** Enkelejda Kasneci `[一作]` (Technical University of Munich), Gjergji Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 14964 | [OpenAlex ID](https://openalex.org/A5024434748)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并公开了一个名为 EduFrameTrap 的教学安全基准，用来测量大型语言模型在面对学生自信度、上下文切换、权威与社交面子压力时是否会支持错误观念。

**💡 创新点**

创新点包括：① 针对教学情境构造的压力/错误观念三元体系；② 通过 Builder‑Validator 生成与验证三层 trap families；③ 将双重 LLM 判断与人工裁决相结合的可靠性评估框架；④ 报告按压力模式、学科与自信度的细粒度失败率。

**🔧 技术方法**

使用技术：多轮对话生成、LLM（GPT‑5.2、Claude‑4.5）进行评测，LLM 作为判定器和构造者，Python/JSON 记录全流程，统计 Wilson 置信区间并绘制热图。

**📊 数据集**

数据集：360 个“陷阱”家族（涵盖数学、物理、经济、化学、生物、计算机科学），每个家族在 3 种自信度与 3 种压力模式下生成 9 个实例，总计 3,240 条对话，拆分为 108/252 家族用于开发/测试。

**📈 对比分析**

与两款前沿 LLM（GPT‑5.2、Claude‑4.5）对比，整体说服错误率约 14%，但按压力模式差异显著（GPT‑5.2 对权威/面子压力高，Claude‑4.5 对上下文切换高）。模型表现被细化到自信度、学科与压力类型的热图显示，表明不同模型在不同风险维度上有不同脆弱性。

**⚠️ 局限性**

局限性：仅测试两种模型，结果不具普适性；基准为合成对话，缺乏真实学习环境的验证；无法证明错误观念的长期影响；LLM 判断仍存在误差，需人工审核；压力范畴与错误观念类型有限，未来可扩展。

---

## 400. Teaching Large Language Models When Not to Know: Learning Temporal Critique for Ex-Ante Reasoning

**arXiv ID:** 2605.14636 | [PDF](https://arxiv.org/pdf/2605.14636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 401. The Rate-Distortion-Polysemanticity Tradeoff in SAEs

**arXiv ID:** 2605.14694 | [PDF](https://arxiv.org/pdf/2605.14694v1)

**作者:** Tommaso Mencattini `[一作]` (École Polytechnique Fédérale de Lausanne), Francesco Locatello `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 3610 | [OpenAlex ID](https://openalex.org/A5073157306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究稀疏自编码器（SAE）在保持低失真和稀疏率的同时，如何实现单语义（monosemantic）解释，并提出了Rate‑Distortion‑Polysemanticity (RDP) 贸易关系。

**💡 创新点**

创新点在于：①将多语义性(polysemanticity)作为第三维度引入信息理论的率‑失真框架；②在简化的线性生成模型下，理论证明概念共现概率决定最优SAE的polysemantic性；③给出代理指标需满足的RDP一致性检验，从而评估现有解释指标的可靠性。

**🔧 技术方法**

技术手段包括：信息理论率‑失真分析、线性生成假设（概念方向正交）、TopK/ ReLU SAE 训练、数学证明与定量上界、以及基于代理指标的V与Spearman ρ 统计检验。

**📊 数据集**

数据集与实验：①合成线性概念数据（n=20，概念正交）；②Gemma和GPT‑2 LLM 的激活张量，用于评估现有 SAE 解释指标。

**📈 对比分析**

比较方法：在不同率/失真预算下采样 SAEs，计算各种代理指标（AutoInterp、Unlearning、Isolation、TPP、Absorption、Sparse Probing、Splitting），通过 V（局部逆序比例）和 ρ（全局趋势相关性）检验其是否符合 RDP 预测。实验结果显示大多数指标（尤其是 AutoInterp）与 RDP 方向完全不符，仅 Sparse Probing 与 Splitting 在局部维度上表现略好。

**⚠️ 局限性**

局限性：①理论基于概念正交且仅考虑线性生成，难以直接推广到复杂真实数据；②缺乏能在无真概念标签情况下训练低 polysemantic SAE 的具体方法；③代理指标评估依赖有限的 SAE 样本，对极端稀疏率或高维激活可能失效。

---

## 402. $π$-Bench: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows

**arXiv ID:** 2605.14678 | [PDF](https://arxiv.org/pdf/2605.14678v1)

**作者:** Haoran Zhang `[一作]` (Shanghai Jiao Tong University), Yafu Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Pi‑Bench基准，包含100个多轮任务，覆盖5个领域角色，聚焦隐藏意图、跨会话依赖与长时序主动协助的评测。

**💡 创新点**

创新性地将隐藏意图与跨会话依赖纳入评测框架，并引入主动性与完整性双重指标，填补了传统评测对显式目标与短交互的盲点。

**🔧 技术方法**

采用ReAct式模块化代理架构，结合工具调用、工作空间操作、模拟用户代理以及规则和Rubric两类评审器，实现对任务流程与意图追踪的自动化评估。

**📊 数据集**

基于领域专家设计的真实工作流程构建数据集，涵盖研究员、营销员、法学实习生、药剂师和金融家5个角色，每角色20个会话，总计100个任务，公开托管于GitHub。

**📈 对比分析**

在9个前沿LLM上进行3次独立跑测，报告主动性(Proc)与完整性(Comp)指标，平均值在43%–67%之间，模型在两项指标上的排名不完全重合，显示主动性与完成度相对独立。

**⚠️ 局限性**

局限性包括使用模拟用户而非真实人类、仅采用单一代理框架、未涵盖不同代理实现的适配成本以及可能缺乏对更广泛真实交互场景的代表性。

---

## 403. TERRA-CD: Multi-Temporal Framework for Multi-class and Semantic Change Detection

**arXiv ID:** 2605.14651 | [PDF](https://arxiv.org/pdf/2605.14651v1)

**作者:** Omkar Oak `[一作]` (COEP Technological University), Suraj Sawant `[通讯]` (COEP Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并公开了名为TERRA-CD的多时相卫星影像基准数据集，用于城市植被和建筑模式的多类与语义变化检测；

**💡 创新点**

提供了包含4类地表覆盖、3类植被变化和13类语义变化的三层注释方案，填补了现有数据集在多类与语义变化检测方面的空缺；

**🔧 技术方法**

使用SVM、NDVI、SWM等传统索引生成标注，随后采用多种深度学习架构（Siamese、STANet、Bi‑SRNet、Changemask、HRSCD等）进行模型训练和评估；

**📊 数据集**

TERRA-CD数据集，包含5,221对2019年与2024年Sentinel‑2影像，覆盖美国171座城市与欧洲61座城市；

**📈 对比分析**

在多类变化检测任务上，Siamese/UNet等模型达到了97–98%整体准确率，mIoU在63–68%之间；在语义变化检测任务上，HRSCD策略表现最佳，整体准确率接近94%，mIoU约70%；

**⚠️ 局限性**

数据集中大部分像素为无变化（约93%），导致类别不平衡；部分城市数据缺乏人工验证；未来需加入SAR、LiDAR等多模态数据提升鲁棒性。

---

## 404. Documentation-Guided Agentic Codebase Migration from C to Rust

**arXiv ID:** 2605.14634 | [PDF](https://arxiv.org/pdf/2605.14634v1)

**作者:** Minh Le-Anh `[一作]` (FPT Software AI Center), Nghi D. Q. Bui `[通讯]` (FPT Software AI Center)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于文档的多代理框架，用于完整的 C 代码仓库向 Rust 的迁移，先生成仓库层面的架构文档，然后按文档规划 crate、翻译代码、逐步修正缺失功能，并通过运行测试持续改进。

**💡 创新点**

创新点在于：①将整个仓库的架构与功能抽象成结构化文档作为迁移蓝图；②利用文档对比与测试反馈驱动迭代修正；③在大规模仓库级别实现了高覆盖率的功能保留与安全性。

**🔧 技术方法**

主要技术包括：多代理编程框架（Planner、Translator、RequirementRefiner、ExecutionRevisor），文档生成模块（DocGen），LLM 代码生成与调优（Kimi‑K2‑Instruct、GPT‑5.4），静态安全检测（detect_unsafe），以及基于 CodeWikiBench 的文档相似度评估。

**📊 数据集**

使用八个真实 C 语言仓库（11K–84K 行），每个仓库附带测试套件，用于功能和安全性评估。

**📈 对比分析**

与 C2Rust、Self‑Repair、EvoC2Rust、Claude Code 等基线对比，迁移方法在两种 LLM 后端下都实现 100% 项目可编译，功能保留率达 93–98%（Claude Code 约 50%），交叉测试通过率 95–98%（Claude Code 约 80%），安全率（API/文件）超过 95%。

**⚠️ 局限性**

局限性包括：仅评估相对标准化的 Cargo 项目，未充分验证高度 FFI 或并发代码；文档对比与测试反馈是主要修正信号，缺少更深入的运行时与静态验证；未分析迁移成本、执行时间或模型调用费用。

---

## 405. SmartWalkCoach: An AI Companion for End-to-End Walking Guidance, Motivation, and Reflection

**arXiv ID:** 2605.14628 | [PDF](https://arxiv.org/pdf/2605.14628v1)

**作者:** Xianzhe Zhang `[一作]` (Xi'an Jiaotong-Liverpool University), Daniel Yonto `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了 SmartWalkCoach，一款集路线规划、实时陪伴和后期回顾的 AI 伴侣应用，利用轻量级多智能体架构提升步行体验。

**💡 创新点**

创新点包括：①将路线规划、陪伴激励与总结拆分为专门代理，降低认知负担；②将情境感知与即时激励（JITAI）结合，提升情绪与体验；③通过实验验证并提出表达频率、时机与关系式交互的设计准则。

**🔧 技术方法**

使用 GPT‑4o 作为核心 LLM 代理，结合地图 API、POI 搜索、路径规划等工具调用；事件驱动、虚拟地理围栏、节奏感知实现上下文感知；采用线性混合模型与主题分析进行评估。

**📊 数据集**

收集 12 名参与者的现场步行实验数据（位置、速度、交互日志）以及自定义问卷；未使用公开数据集。

**📈 对比分析**

采用 AB/BA 交叉设计，使用线性混合模型比较信息+激励与信息单一两种条件；结果显示信息+激励条件在正面情绪（d≈1.85）和用户体验（d≈1.46）上显著优于对照，且无显著序列或遗留效应。

**⚠️ 局限性**

限制包括：样本量小且学生化、实验仅短期未评估长期行为改变、疲劳检测仅基于速度导致误判、缺乏实时路径再规划、视觉交互可能影响注意力、个性化与多模态支持不足。

---

## 406. Digital Twin Synchronization Over Mobile Embodied AI Network With Agentic Intelligence

**arXiv ID:** 2605.14625 | [PDF](https://arxiv.org/pdf/2605.14625v1)

**作者:** Zhouxiang Zhao `[一作]` (Zhejiang University), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 22811 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了一种基于移动具身AI网络（MEAN）的数字孪生同步框架，通过移动感知、协同感知、边缘语义处理、通道感知移动与上行传输等五阶段闭环工作流，实现对多目标区域的同步优化。

**💡 创新点**

创新点包括：① 将Agentic AI与移动感知融合，利用自适应速度与主动通道搜索实现能量‑时间权衡；② 采用协同感知函数与语义压缩模型，将语义通信与无线资源互为补充；③ 设计分层两层优化（匹配+连续资源）和基于动态匹配游戏的最优拓扑分配，显著降低同步偏差。

**🔧 技术方法**

技术手段包括：多代理匹配游戏与外部效应、二次凸近似与凸优化（SCA、BCD）、闭环速度解析、语义压缩模型、通道多样性建模、能量与带宽约束的多维资源调度。

**📊 数据集**

使用了公开的交通感知数据集：DAIR‑V2X、V2XSet 和 OPV2V，用于验证协同感知准确性曲线。

**📈 对比分析**

与四种基线（均分带宽、无协同、无压缩、固定速度）在相同拓扑下比较，采用最差双边同步偏差指标；实验显示所提框架在不同带宽、协同指数、计算复杂度、路径损耗与能量预算场景下均实现 20‑30 秒左右的同步偏差显著低于基线，并在能量充分时逼近最优。

**⚠️ 局限性**

局限性包括：① 仅考虑单基站场景，未扩展到多基站/多网格；② 语义压缩模型与通道多样性参数需手工估计；③ 匹配游戏在大规模代理下仍可能陷入局部最优；④ 对动态环境变化速度与不确定性建模仍需进一步完善。

---

## 407. SliceGraph: Mapping Process Isomers in Multi-Run Chain-of-Thought Reasoning

**arXiv ID:** 2605.14619 | [PDF](https://arxiv.org/pdf/2605.14619v1)

**作者:** Kang Chen `[一作]` (Fudan University), Yugang Jiang `[通讯]` (Fudan University)

**通讯引用:** 24983 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并分析了基于稀疏激活键相似性的 SliceGraph，用以量化多次链式推理的过程几何和同答案的过程同形异路。

**💡 创新点**

提出了测量型图谱 SliceGraph，验证了 BCC 与过程同形异路概念，并引入高价值核心与 Typed‑State 动态，揭示同答案推理的多路多景。

**🔧 技术方法**

稀疏激活键 Jaccard 互kNN 图构建，Biconnected Component (BCC) 分解，Louvain 聚类，基于奖励的高价值核心扩散，Typed‑State 迁移矩阵。

**📊 数据集**

AIME、BRUMO、HMMT、GPQA Diamond 等数学/科学推理基准，使用 Qwen3‑4B/8B、DeepSeek‑R1、Llama‑8B、Qwen2.5‑72B 等模型。

**📈 对比分析**

与 PCA/UMAP/句子嵌入等投影管线对比，SliceGraph 在保持多路性（≥2 家族率约 86%）和同形异路率约 76% 方面优于投影方法；动力学对比显示家族间 TV 显著偏离随机置换。

**⚠️ 局限性**

依赖稀疏激活缓存和 hook，可能低采样稀有路径；仅适用于非代码推理，奖励字段为描述性，未用于实时解码。

---

## 408. In-IDE Toolkit for Developers of AI-Based Features

**arXiv ID:** 2605.14612 | [PDF](https://arxiv.org/pdf/2605.14612v1)

**作者:** Yaroslav Sokolov `[一作]` (JetBrains), Artem Trofimov `[通讯]` (JetBrains)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了AI Toolkit插件，将LLM/agent调试与评估无缝集成到JetBrains IDE的Run/Debug流程中。

**💡 创新点**

通过在Run时自动捕获追踪、即时可视化、一次性生成数据集并以单元测试式评估方式闭环，显著降低上下文切换与配置成本。

**🔧 技术方法**

采用Python wrapper + TCP客户端‑服务器与LangChain/Graph框架回调收集事件；使用可插拔评估器（包括LLM‑as‑a‑judge）与YAML存储；基于JetBrains插件架构实现IDE内可视化与配置。

**📊 数据集**

主要使用项目内部手工测试案例生成的追踪作为数据集，未依赖公开大规模数据集；数据存储为项目内YAML文件。

**📈 对比分析**

通过插件安装率、首次追踪捕获率、评估跑分等指标进行对比；在PyCharm 2025.2发布后，Run弹窗安装率58%，四周内26%活跃率，评估可视化显示平均分数、token使用与耗时等关键指标。

**⚠️ 局限性**

主要局限为框架覆盖范围狭窄（仅支持LangGraph）、观察窗口短、仅PyCharm平台，缺乏跨IDE与长期使用的验证。

---

## 409. Vision-Based Water Level and Flow Estimation

**arXiv ID:** 2605.14645 | [PDF](https://arxiv.org/pdf/2605.14645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 410. DRL-STAF: A Deep Reinforcement Learning Framework for State-Aware Forecasting of Complex Multivariate Hidden Markov Processes

**arXiv ID:** 2605.14632 | [PDF](https://arxiv.org/pdf/2605.14632v1)

**作者:** Manrui Jiang `[一作]` (Tsinghua University), Chen Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 53430 | [OpenAlex ID](https://openalex.org/A5100374115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于深度强化学习的状态感知预测框架 DRL-STAF，能够在多变量隐藏马尔可夫过程中同时预测观测值并硬解码离散隐藏状态。

**💡 创新点**

创新点在于：①采用无似然分布自由的 DRL 策略直接估计离散隐藏状态，解决软解码模糊与状态空间爆炸问题；②设计两阶段训练和基于 ResGAT 的图注意力模块，实现跨变量交互的高效协调；③通过奖励设计实现预测收益与状态分离的双重目标。

**🔧 技术方法**

技术手段包括深度神经网络作为非线性发射函数、深度强化学习策略网络进行硬状态解码、ResGAT 图注意力网络处理跨变量依赖，以及基于预测误差与状态分离的自定义奖励函数。

**📊 数据集**

实验使用四个仿真数据集（3 变量/10 变量，频繁/稀疏转移）以及三真实数据集（SMachine、Exchange、Traffic）进行验证。

**📈 对比分析**

在与传统 HMM 变体（Parallel HMM、HSMM、HOHMM、CHMM）和 DL-HMM 混合模型（NHMM、NCTRL、Markovian-RNN、DEN-HMM）以及单一 DL 模型比较时，DRL-STAF 在 MAE、MSE 以及状态估计的准确率、F1 等指标上普遍取得最佳或近似最佳成绩。

**⚠️ 局限性**

局限性包括：训练过程相对复杂，需要两阶段的分步优化；尚未直接扩展到多步预测；在更一般的 HMM 结构或大规模变量场景下的可扩展性和训练效率仍需进一步提升。

---

## 411. Agentic AI in Industry: Adoption Level and Deployment Barriers

**arXiv ID:** 2605.14675 | [PDF](https://arxiv.org/pdf/2605.14675v1)

**作者:** Spyridon Alvanakis Apostolou `[一作]` (Chalmers University of Technology), Helena Holmström Olsson `[通讯]` (Malmö University)

**通讯引用:** 4859 | [OpenAlex ID](https://openalex.org/A5049811300)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对12家工业公司进行半结构化访谈，并采用六层成熟度模型评估其代理式人工智能（Agentic AI）的采纳程度与实际部署状况；同时识别并描述了关键的部署障碍。

**💡 创新点**

首次实证揭示并量化“能力-部署验证缺口”，将信息不对称与资格缺失这两维视为核心开放问题，提供了关于工业环境中代理AI采用的系统性洞见。

**🔧 技术方法**

运用基于大型语言模型（LLM）的代理工具、检索增强生成（RAG）、局部微调（LoRA）等技术；研究方法包括访谈、案例合成与跨案例比较。

**📊 数据集**

数据来源为16名从业者的访谈记录，涵盖12家企业的规模、行业与监管环境；未使用公开的数据集。

**📈 对比分析**

使用成熟度框架对公司进行分级并进行跨案例比较，发现大多数公司停留在Level 1，只有极少数达到Level 3；研究并未给出传统性能指标，而是通过比较四大障碍和验证方法评估进展。

**⚠️ 局限性**

局限性包括：受访者自我报告的主观性、样本聚焦于已积极采用AI的组织导致外推受限、未能提供通用的验证方法，仅识别并描述了现有问题。

---

## 412. Multi-objective application placement in fog computing using graph neural network-based reinforcement learning

**arXiv ID:** 2605.14649 | [PDF](https://arxiv.org/pdf/2605.14649v1)

**作者:** Isaac Lera `[一作]` (Universitat de les Illes Balears), Carlos Guerrero `[通讯]` (Universitat de les Illes Balears)

**通讯引用:** 1535 | [OpenAlex ID](https://openalex.org/A5034786387)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种面向雾计算环境的多目标深度强化学习框架，用于根据服务间的依赖关系和设备特征动态决定应用中各服务的部署位置；

**💡 创新点**

创新点在于：①使用图同构网络（GIN）对服务依赖图进行特征聚合，自动生成服务优先级；②采用双层actor‑critic PPO结构分别负责服务选择和设备分配；③通过标量分解和模型参数迁移实现多目标 Pareto 前沿的高效获取；

**🔧 技术方法**

所采用的技术包括深度强化学习（PPO actor‑critic）、图同构网络（GIN）、参数迁移策略以及基于奖励权重的标量化方法；

**📊 数据集**

实验使用自定义的合成雾环境数据集，设备数量从500到2000，应用服务数为81或144（基于作业调度 DAG），与NSGA‑II与单目标遗传算法进行比较；

**📈 对比分析**

比较方法：将DRL训练150个episode，训练时间约2–8小时，部署时延<1秒；与NSGA‑II（30–70分钟）和GA（1–5小时）相比，DRL在求解速度上明显更快，获得的Pareto集合与遗传算法相当；

**⚠️ 局限性**

局限性包括：①多目标解集可能不够多样，无法完全覆盖Pareto前沿；②参数迁移假设目标相似，可能导致收敛到局部最优；③实验基于合成数据，缺乏真实雾环境的验证；④未考虑设备容量与能耗约束等更复杂约束。

---

## 413. CalibAnyView: Beyond Single-View Camera Calibration in the Wild

**arXiv ID:** 2605.14615 | [PDF](https://arxiv.org/pdf/2605.14615v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. Towards Accurate Single Panoramic 3D Detection: A Semantic Gaussian Centric Approach

**arXiv ID:** 2605.14601 | [PDF](https://arxiv.org/pdf/2605.14601v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 415. An Amortized Efficiency Threshold for Comparing Neural and Heuristic Solvers in Combinatorial Optimization

**arXiv ID:** 2605.14624 | [PDF](https://arxiv.org/pdf/2605.14624v1)

**作者:** Sohaib Afifi `[一作]` `[通讯]` (University of Artois), Sohaib Afifi (University of Artois)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了一个针对神经网络组合优化求解器与传统启发式求解器的能耗比较框架，结合训练成本、推理成本和硬件生命周期碳排放，评估两者在实际部署规模下的能效差异。

**💡 创新点**

创新点包括：①定义“摊销能效阈值”以量化固定训练成本与可变推理成本的分界；②在能耗核算中加入硬件制造碳排放的线性摊销；③给出一个与训练成本无关的渐近能耗比结构性证明，说明在大规模部署时神经求解器能始终优于启发式方法。

**🔧 技术方法**

主要技术包括：批量神经网络推理、GPU/CPU 能耗测量、碳强度校准、生命周期碳摊销模型以及多维敏感度表面分析。

**📊 数据集**

使用了 Multi‑Task VRP（MTVRP）环境中的 19 个不同约束组合（如容量、时间窗、距离限制等），每个约束组合在 n = 20 的客户规模下训练五个种子模型进行评估。

**📈 对比分析**

比较方法：对比单实例推理能耗与单线程/多线程基线启发式（HGS）能耗，计算累计能耗比并确定摊销阈值；结果显示，单实例神经求解器约为 0.41 倍的能耗，摊销阈值约为 1.58×10⁵ 次实例；在此阈值以上，神经求解器的累计能耗比始终低于启发式。

**⚠️ 局限性**

局限性：仅在 n = 20 的小规模实例上实验，未验证更大规模（n = 50/100）的性能提升；假设问题分布保持不变，未考虑迁移训练或分布漂移；碳排放摊销采用线性五年寿命假设，实际使用寿命和硬件刷新周期可能不同；评估集中在单一启发式基线，未覆盖其他算法。

---

## 416. Capacitive Touchscreens at Risk: A Practical Side-Channel Attack on Smartphones via Electromagnetic Emanations

**arXiv ID:** 2605.14633 | [PDF](https://arxiv.org/pdf/2605.14633v1)

**作者:** Yukun Cheng `[一作]` (Wuhan University), Shihui Zheng `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 485 | [OpenAlex ID](https://openalex.org/A5035616913)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种全新的、完全非接触式的电磁侧信道攻击框架，可从智能手机电容式触摸屏产生的 EM 辐射中恢复 PIN 码、键盘输入、应用类别以及手写轨迹。

**💡 创新点**

创新点在于：① 利用触摸屏扫描时的时间-空间映射将 EM 泄露与具体触摸位置关联；② 统一解码离散与连续输入的单一框架；③ 通过大量采样实现跨设备、跨型号的高度可迁移性；④ 在无硬件改造、低成本、可在公共场所部署的前提下实现高精度攻击。

**🔧 技术方法**

技术手段包括：1) 使用 5 cm 近距离 EM 探头以 500 kHz 采样；2) 信号截取、阈值检测与归一化；3) 针对不同目标设计的轻量级深度学习模型（MLP、CNN、Transformer）；4) 位置识别与手写轨迹重建的三阶段流水线；5) 在实验中使用随机森林对用户交互类型进行分类。

**📊 数据集**

数据集：20 名受试者在四款主流手机（iPhone X、Xiaomi 10 Pro、Samsung S10、Huawei Mate 30 Pro）上收集的 EM 信号，涵盖 PIN（12 000 条）、键盘（31 200 条）、应用分类（300 条）和手写（6 200 条）共计约 600 000 条样本。

**📈 对比分析**

性能对比与评估：PIN 码识别准确率 99.3%；键盘输入 97.6%；应用类别 95%；手写字符 76.8%；轨迹 Jaccard 相似度 0.74。 在公开/私密环境、不同距离（5–25 cm）、不同手机型号以及跨设备测试中，5 次尝试的整体成功率高达 99.3%，单次尝试仍超过 63%。

**⚠️ 局限性**

限制与挑战：1) 需要设备与探头距离 ≤ 25 cm，金属屏障等强屏蔽会显著削弱效果；2) 长序列（如完整句子）累计误差导致恢复精度下降；3) 对随机键盘布局无效，仅能通过硬件级扫描顺序随机化或 EM 隐蔽化实现根本防护。

---

## 417. One Step to the Side: Why Defenses Against Malicious Finetuning Fail Under Adaptive Adversaries

**arXiv ID:** 2605.14605 | [PDF](https://arxiv.org/pdf/2605.14605v1)

**作者:** Itay Zloczower `[一作]` (Ben Gurion University of Negev), Yisroel Mirsky `[通讯]` (Ben Gurion University of Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究恶意微调（MFT）防御方法，发现现有15种防御在机制上归结为两大策略，提出了统一的自适应攻击（SideStepper）来突破所有防御，并给出了实验验证。

**💡 创新点**

创新点在于系统化归纳MFT防御的四种损失模板和两种安全策略，揭示其共同弱点，并提出基于联合损失（Harm + Capability）的一体化自适应攻击，从而揭示防御在局部性约束下的失效。

**🔧 技术方法**

使用的技术包括：对抗微调模拟、损失模板重构、联合损失优化、学习率冲击-衰减（Kick-Settle）等；实验中采用Llama‑2‑7B、Qwen3‑8B、Llama‑3.1‑8B三大模型，结合15种防御实现。

**📊 数据集**

使用的数据集包括对抗性微调训练集（HarmBench 违规样本）、安全评估集（MMLU、TruthfulQA、HellaSwag、ARC‑Easy）以及 HarmBench 违规样本集。

**📈 对比分析**

对比方法：对每种防御先进行标准的恶意微调（仅优化 H‑loss），再使用自适应攻击（H‑loss + λC‑loss）进行微调。实验结果显示，自适应攻击能使所有防御的有害得分提升 0.20–0.74，且对大多数模型的实用能力影响微乎其微（Δ ≤ 0.07）。

**⚠️ 局限性**

局限性：实验仅覆盖三种主流大模型与15种已公开的防御；未对更大规模模型、不同微调策略或多模态系统进行评估；同时攻击仍使用固定的 λ 超参数，未探索更广泛的自适应搜索空间。

---

## 418. Fast Rates for Inverse Reinforcement Learning

**arXiv ID:** 2605.14599 | [PDF](https://arxiv.org/pdf/2605.14599v1)

**作者:** Andreas Schlaginhaufen `[一作]` (École Polytechnique Fédérale de Lausanne), Maryam Kamgarpour `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 2792 | [OpenAlex ID](https://openalex.org/A5082009236)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了熵正则化最小-最大逆强化学习(Min‑Max‑IRL)，并证明其在总体层面等价于最大似然估计(MLE)，在经验层面在确定性动力学下保持等价。

**💡 创新点**

提出了结构性等价性、利用伪自协同性证明n⁻¹快速收敛率、将奖励可识别性扩展到Borel空间，以及软最优价值函数对奖励参数的高阶导数分析。

**🔧 技术方法**

采用凸优化、伪自协同性质、Dikin椭圆局部化、向量Bernstein界定、以及对损失函数的自协同性分析。

**📊 数据集**

文中未给出具体数据集，主要为理论证明与数值示例。

**📈 对比分析**

与传统BC和旧版IRL（n⁻¹/2速率、需要探索假设）对比，Min‑Max‑IRL在缺失探索假设、可欠指定情况下实现了n⁻¹收敛，并在轨迹KL与参数误差上达到最优或接近最优表现。

**⚠️ 局限性**

局限包括：需要熵正则化参数β>0、Hessian正定假设、经验等价仅在确定性动态下成立，以及缺乏对真实数据集的经验验证。

---

## 419. Falkor-IRAC: Graph-Constrained Generation for Verified Legal Reasoning in Indian Judicial AI

**arXiv ID:** 2605.14665 | [PDF](https://arxiv.org/pdf/2605.14665v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 420. MiVE: Multiscale Vision-language features for reference-guided video Editing

**arXiv ID:** 2605.14664 | [PDF](https://arxiv.org/pdf/2605.14664v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 421. DSSP: Diffusion State Space Policy with Full-History Encoding

**arXiv ID:** 2605.14598 | [PDF](https://arxiv.org/pdf/2605.14598v1)

**作者:** Zhiyuan Guan `[一作]` (Shanghai Jiao Tong University), Yutong Ban `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1348 | [OpenAlex ID](https://openalex.org/A5011762462)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种全历史条件的扩散状态空间策略（DSSP），通过状态空间模型压缩完整观察序列并与即时状态进行层级条件融合，显著提升长周期机器人操作的成功率。

**💡 创新点**

创新点包括：① 将全历史编码器与扩散去噪器统一为 Mamba 型状态空间模型，实现线性时间、低内存的历史压缩；② 动态感知辅助损失促使上下文保持对未来状态的可预测性；③ 将历史上下文与最近观察构成前缀条件，并通过 AdaLN 分离时间步调节，提升推理稳定性；④ 在多种仿真与真实任务上验证了该方法在长周期记忆依赖任务中的优势。

**🔧 技术方法**

技术手段：状态空间模型（Mamba）实现可扩展的历史编码和去噪；扩散模型用于连续动作生成；动态感知辅助目标用于上下文学习；层级前缀条件和 AdaLN 用于高效条件注入；训练采用 diffusion 重构损失 + 动态感知损失的联合优化。

**📊 数据集**

数据集：RoboTwin 2.0（50 双臂任务）、MetaWorld（34 单臂桌面任务）、Adroit（3 细致手部任务），以及真实世界的 AgileX 机器人完成三项长周期任务（Put Bottles、Object Swap、Morse Tapping）。

**📈 对比分析**

与多种基线（ACT、DP、π₀、RDT、SeedPolicy、FlowPolicy、AdaFlow、CP、MP1 等）比较，DSSP 在 RoboTwin、MetaWorld、Adroit 的平均成功率分别提升 12.8%、21.3%、19.8%；模型参数仅 44.3M，显著小于 260M 以上的对手；在真实任务中平均成功率从 30% 提升至 70%（133% 相对提升），尤其在 Morse Tapping 上从 15% 提升至 85%。

**⚠️ 局限性**

局限性：① 依赖完整历史记录，若历史采样或传感器出现缺失可能导致性能下降；② 需要大量演示数据和较长训练时间；③ 在高度动态、遮挡严重或非结构化环境中的鲁棒性尚未充分验证；④ 目前仅针对 RGB/点云+关节状态，扩展到更丰富多模态感知需进一步研究。

---

## 422. ReMIA: a Powerful and Efficient Alternative to Membership Inference Attacks against Synthetic Data Generators

**arXiv ID:** 2605.14686 | [PDF](https://arxiv.org/pdf/2605.14686v1)

**作者:** Davide Scassola `[一作]` (Aindo SpA), Sebastiano Saccani `[通讯]` (Aindo SpA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并实现了一种新的合成数据隐私评估指标 ReMIA，基于相对成员推断攻击来衡量合成数据生成器的隐私风险。

**💡 创新点**

ReMIA 的创新点在于仅需两次生成器训练、无需大量辅助数据，且通过比较两组合成数据的来源来近似传统 MIA 的敏感度，从而实现了更实用的隐私评估。

**🔧 技术方法**

采用了多层感知机（MLP）分类器训练来区分两组合成数据，使用 AUROC 作为隐私得分，并与 DOMIAS、shadow-MIA（smMIA）和 DCR 进行对比。

**📊 数据集**

实验使用了 Adult、UK Census 和 California 三个公共 tabular 数据集，并在 leaky 风险模型和噪声化匿名化方法下进行评估。

**📈 对比分析**

与 smMIA（基于异常样本）和 DOMIAS 的比较表明，ReMIA 在敏感度上与 smMIA(med) 相当，明显优于 DCR，且在计算时间和数据使用上比 smMIA 减少数百倍，表现出更高的实用性。

**⚠️ 局限性**

局限性包括：对攻击者模型（分类器设计、训练策略）的实现细节敏感；实验范围仅覆盖有限的生成器和匿名化方法；对训练规模对隐私风险影响的系统性探索尚未完成。

---

## 423. AI-assisted cultural heritage dissemination: Comparing NMT and glossary-augmented LLM translation in rock art documents

**arXiv ID:** 2605.14679 | [PDF](https://arxiv.org/pdf/2605.14679v1)

**作者:** Vicent Briva-Iglesias `[一作]` (Dublin City University), María Ferre-Fernández `[通讯]` (Universidad de Almería)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了三种英文机器翻译设置（DeepL NMT、Gemini-LMM 无提示、Gemini-LMM 结合术语检索提示）在西班牙语岩石艺术文本的术语精确度与整体质量上的表现。

**💡 创新点**

证明了轻量级术语检索提示（RAG）在不改变模型、无需微调的前提下，可显著提升专业术语的一致性与准确率，同时保持整体译文质量。

**🔧 技术方法**

使用了基于提示的检索增强生成（RAG）技术，将检索到的术语对直接注入 Gemini LLM 的提示中；评估方法包括多通道直接评估（DA）和受限 MQM 术语审计。

**📊 数据集**

数据集为一篇西班牙语岩石艺术学术文本（91 段，1743 词）和一个 200 条条目的双语术语表（其中 44 条在文本中出现，共 194 次出现）。

**📈 对比分析**

通过 PEARMUT 平台进行人工评估：在 DA 任务中，Gemini-RAG 与 Gemini-Simple 的平均得分相当（≈85.3），均优于 DeepL（≈80.3）；在术语精确度任务中，Gemini-RAG 达到 81.4%（158/194），高于 Gemini-Simple（69.1%）和 DeepL（64.4%），且在两两比较中均显著。

**⚠️ 局限性**

局限性包括：样本仅为单篇西班牙语岩石艺术文本，评估规模小；人工评估未采用双人并行标注，仅为专家复核；术语精确度评判过于严格，仅计 exact-match；所用系统为特定时间点的商业服务，结果随模型更新可能变化。

---

## 424. LLM-Enabled Automated Algorithm Design for Multiuser Fluid Antenna Communications

**arXiv ID:** 2605.14661 | [PDF](https://arxiv.org/pdf/2605.14661v1)

**作者:** Gan Zheng `[一作]` (University of Warwick), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 39961 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文利用大语言模型自动化设计启发式算法，用于多用户流体天线系统的端口选择，以最大化最小SINR；

**💡 创新点**

创新点在于提出LLM驱动的算法设计框架，既可优化遗传算法的交叉/变异操作，也能完全由LLM生成全新的AutoPort启发式算法，最终实现近最优性能；

**🔧 技术方法**

使用了大语言模型（Deepseek‑V3 等）与进化启发式搜索（EoH）、遗传算法、GRASP等技术，并与传统 RBF、Transformer 和随机方法对比；

**📊 数据集**

采用仿真生成的多用户 MISO 流体天线通道数据，1,000 条用于算法训练，100 条用于评估，环境为 Rayleigh 与 Rician 随机衰落；

**📈 对比分析**

通过与穷举搜索、基本 GA、RBF、Transformer、随机选择等基线对比，LLM 优化后的算法在多用户场景中可达或接近穷举解，性能提升 2–5 dB，且计算耗时相对合理；

**⚠️ 局限性**

局限性包括仅关注性能而未平衡复杂度；未考虑 CSI 估计误差；仅验证窄带模型，未扩展到 OFDM 等宽带环境；并且对更复杂场景的泛化能力仍需进一步验证。

---

## 425. MindGap: A Conversational AI Framework for Upstream Neuroplastic Intervention in Post-Traumatic Stress Disorder

**arXiv ID:** 2605.14660 | [PDF](https://arxiv.org/pdf/2605.14660v1)

**作者:** Eranga Bandara `[一作]` (Old Dominion University), Atmaram Yarlagadda `[通讯]` (McDonald Army Health Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 MindGap——一种基于依存起源理论、在设备上运行的对话式 AI，用于 PTSD 前沿情绪音调间隙的神经可塑性干预。

**💡 创新点**

创新点：首次在 PTSD 治疗中定位并干预“情绪音调间隙”，通过三层观照实现上游消解；同时采用隐私保留的设备端 LLM 进行高频训练。

**🔧 技术方法**

技术：细调轻量级 LLM（如 Phi‑3 Mini / LLaMA 3.2 3B），离线推理、对话管理、刺激校准阶梯、实时激活评估。

**📊 数据集**

数据集：依存起源练习对话语料、创伤知情治疗语言语料、情绪音调触发对话样例；无公开数据集。

**📈 对比分析**

比较方法：提出随机对照试验设计，使用 fMRI、心率变异性、PCL‑5/CAPS‑5 作为指标；目前仅为理论与案例分析，尚未有实验性能数据。

**⚠️ 局限性**

局限：缺乏实证验证，算法对不同 PTSD 级别的适应性、激活阈值确定、层次递进的自动化评估仍需优化；对多模态感知整合有限。

---

## 426. SciPaths: Forecasting Pathways to Scientific Discovery

**arXiv ID:** 2605.14600 | [PDF](https://arxiv.org/pdf/2605.14600v1)

**作者:** Eric Chamoun `[一作]` (University of Cambridge), Andreas Vlachos `[通讯]` (University of Cambridge)

**通讯引用:** 5026 | [OpenAlex ID](https://openalex.org/A5067943980)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现SciPaths基准，用于预测科研发现路径，包含任务A（生成实现目标贡献所需的启用贡献）和任务B（对这些贡献进行先行工作检索与归属）。

**💡 创新点**

创新点在于把科学进展建模为贡献层级的依赖路径，区别于传统论文引用预测，明确标注启用贡献、功能角色、必要性理由以及对应的先行工作或无映射决策，从而实现从目标贡献向必要构件的逆向推理。

**🔧 技术方法**

主要技术包括：大型语言模型（Gemini 3.1 Pro、GPT‑5.4等）用于生成与判断；语义匹配判定器；LLM驱动的银数据构建管道；Semantic Scholar检索与LLM重排序；以及基于角色与理由的手工注解规范。

**📊 数据集**

数据集为262条专家标注的金标准路径和2,444条银标准路径，来源为2023‑2025年NeurIPS、ICML、ACL、EMNLP等会议的机器学习与自然语言处理论文，路径通过下游重用证据构建。

**📈 对比分析**

评测方法：在任务A采用严格的一对一语义匹配，计算precision/recall/F1；任务B在不同检索条件下评估top‑K recall、precision和coverage；实验表明最强模型Gemini 3.1 Pro在任务A的F1仅为0.189，任务B在gold启用贡献条件下的coverage约0.35；现有模型整体表现仍远低于专家级完整路径恢复。

**⚠️ 局限性**

局限性包括：对专家标注和语义匹配判定器的高依赖；仅覆盖ML/NLP领域，未覆盖其他学科；路径多样性导致可能存在多种有效分解，评测无法覆盖所有；检索阶段受Semantic Scholar元数据限制；模型在识别核心方法学依赖方面仍显不足。

---

## 427. MultiEmo-Bench: Multi-label Visual Emotion Analysis for Multi-modal Large Language Models

**arXiv ID:** 2605.14635 | [PDF](https://arxiv.org/pdf/2605.14635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 428. Spontaneous symmetry breaking and Goldstone modes for deep information propagation

**arXiv ID:** 2605.14685 | [PDF](https://arxiv.org/pdf/2605.14685v1)

**作者:** Nabil Iqbal `[一作]` (Durham University), Max Welling `[通讯]` (CuspAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造并分析对连续对称性（如U(1)、O(k)）等变层的深度网络和循环网络，证明其在自发对称性破缺相中可出现Goldstone模态，支持跨层/跨时间的稳定信息传播，并在不依赖残差或归一化的条件下实现可训练的极深网络；

**💡 创新点**

首次把物理中的Goldstone模式概念引入深度学习，提出对称性破缺相可提供天然的梯度与信息通道，从而大幅提升深度网络与RNN的训练稳定性与长序列记忆；

**🔧 技术方法**

使用群等变卷积/等变线性层、平滑相位保持激活函数（如 tanh(|z|)/|z| z）、大N自洽场分析、Jacobian/相位保持特性研究、以及U(1)和O(k)等变RNN/GRU架构；

**📊 数据集**

在MNIST系列（Fashion‑MNIST、Permutation‑Sequential‑MNIST）、CIFAR‑10（部分实验）以及合成的可变延迟复制任务上验证；

**📈 对比分析**

与普通非等变MLP、RNN、GRU做对比（相同/更少参数、不同学习率、激活函数）；结果显示：在极深（100层）时，等变网络可训练成功且性能优于对手；在长序列任务（复制、psMNIST）中等变模型在同等参数下准确率提升10–30%，甚至比带门控的GRU更优；

**⚠️ 局限性**

实验受限于简单基准与小规模模型，缺乏对大规模数据/更复杂任务的验证；未探讨自发对称性破缺相对训练稳定性的理论极限与上界；未系统评估拓扑缺陷等其他可能机制；

---

## 429. SeaVis: Modeling and Control of a Remotely Operated Towed Vehicle for Seabed Visualization and Mapping

**arXiv ID:** 2605.14683 | [PDF](https://arxiv.org/pdf/2605.14683v1)

**作者:** Abdelhakim Amer `[一作]` (Aarhus University), Erdal Kayacan `[通讯]` (Paderborn University)

**通讯引用:** 6211 | [OpenAlex ID](https://openalex.org/A5068099488)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发了SeaVis远程牵引水下车辆（ROTV）的详细解析动力学模型，并基于该模型设计了增益调度的线性二次调节器（LQR）实现深度与姿态控制，同时与传统PID控制器进行了对比验证。

**💡 创新点**

创新点包括：①将ROTV的刚体、附加质量、线性与非线性阻尼以及升力矩等多种水动力效应统一纳入解析模型；②采用增益调度的LQR实现随航速变化的自适应控制；③结合抗风箱技术保证在舵面饱和时系统保持稳定；④将完整模型与控制器公开源代码并提供高保真仿真框架。

**🔧 技术方法**

使用技术主要有：Newton‑Euler运动方程、附加质量与阻尼计算、升力系数线性化、LQR最优控制、增益调度插值、抗风箱补偿、HoloOcean + Unreal Engine仿真、MAVLink/BlueOS通讯架构。

**📊 数据集**

本研究未使用公开实验数据集，而是通过自定义的三段地形（连续斜坡、下降台阶、上升台阶）在高保真仿真环境中生成合成数据，用于评估控制性能。

**📈 对比分析**

通过在无扰动、随机扰动和速度变化等三种场景下对比LQR与PID，实验显示LQR在平坦地形深度误差≤2 cm、舵面角度最大3°、能耗与PID相比降低约60%；在急变台阶时深度误差<2 cm，且PID出现5 cm以上超调；在扰动下LQR保持深度误差≤6.5 cm、姿态误差≤0.5，PID虽误差略低但舵面动作更激烈。

**⚠️ 局限性**

主要局限包括：模型在横向运动与舵面耦合方面做了简化，未考虑强侧向流场或极端海流；仿真结果尚未在实物平台上验证，实际工况可能出现模型误差；增益调度仅在0–5 m/s航速区间内验证，超速或低速性能未知。

---

## 430. Identifying Culprits Through Deep Deterministic Policy Gradient Deep Learning Investigation

**arXiv ID:** 2605.14774 | [PDF](https://arxiv.org/pdf/2605.14774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 431. How Sensitive Are Radiomic AI Models to Acquisition Parameters?

**arXiv ID:** 2605.14667 | [PDF](https://arxiv.org/pdf/2605.14667v1)

**作者:** D. Gil `[一作]` (Universitat Autònoma de Barcelona), C. Sanchez `[通讯]` (Universitat Autònoma de Barcelona)

**通讯引用:** 26762 | [OpenAlex ID](https://openalex.org/A5111364995)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发并验证了一种基于混合效应模型的统计多目标优化框架，用于评估和调整CT扫描参数对肺癌放射组学AI模型的敏感性和性能。

**💡 创新点**

直接建模扫描参数对预测失败风险的影响，并通过Pareto层级化搜索得到一组可接受的参数区域，而非单一最优配置；将参数优化与多中心数据验证相结合。

**🔧 技术方法**

广义混合效应模型（GMM）、OR估计、Pareto多目标优化、统计显著性检验、深度学习基准模型（ViT、ConvNext、Swin、EfficientNet、ResNet、DenseNet、VGG）以及放射组学3D特征提取。

**📊 数据集**

LUNA16公开肺结节数据库与本机构收集的多中心RadioLung数据库。

**📈 对比分析**

通过比较不同参数设置下模型在高质量（HQ）和低质量（LQ）扫描上的敏感度、特异性、F1、准确率和AUC；优化后敏感度提升0.15–0.20，特异性提升至0.83–0.98，AUC提升至0.92–0.97；多中心跨数据集验证保持显著性能提升。

**⚠️ 局限性**

仅评估了单一指标（准确率）的OR估计；参数空间搜索采用网格搜索，未探索连续空间；研究仅聚焦肺癌CT，未验证到其他模态或任务；对小样本数据集的统计稳定性有限。

---

## 432. UniTriGen: Unified Triplet Generation of Aligned Visible-Infrared-Label for Few-Shot RGB-T Semantic Segmentation

**arXiv ID:** 2605.14626 | [PDF](https://arxiv.org/pdf/2605.14626v1)

**作者:** Ping Zhou `[一作]` (Northwestern Polytechnical University), Fei Zhou `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 22890 | [OpenAlex ID](https://openalex.org/A5050902155)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的文本驱动 VIS‑IR‑Label 生成框架 UniTriGen，能够在仅有限对齐训练样本的条件下一次性生成空间对齐、语义一致、模态互补的三模态数据；

**💡 创新点**

创新点在于：①统一三模态到共享潜在空间并用扩散模型联合建模，消除多阶段生成中的误差累积；②引入轻量化模态特定残差适配器在解码阶段提升模态质量；③设计场景均衡与类别感知的少样本采样策略，缓解少样本下的场景/类别不平衡；

**🔧 技术方法**

使用的技术包括 CLIP 文本编码、VAE 编码器/解码器、联合潜在扩散模型（UniDiffuser）、残差适配器、场景编码与分层重采样；

**📊 数据集**

在 SemanticRT 与 PST900 两个公开 RGB‑T 语义分割数据集上进行实验；

**📈 对比分析**

与三种级联生成基线（Seq‑Cascade、JointLabel‑Cascade、TriCond‑Cascade）对比，UniTriGen 在 5%/50% 低样本设置下平均提升 mIoU 约 4–7%，在多种下游网络上均表现为最优或次优；

**⚠️ 局限性**

局限性：对极端稀有场景/类别仍可能生成质量不佳；对真实场景的跨模态配准误差不被直接纠正；生成速度相对慢，且对大规模高分辨率数据的扩展尚未验证。

---

## 433. Deep Image Segmentation via Discriminant Feature Learning

**arXiv ID:** 2605.14609 | [PDF](https://arxiv.org/pdf/2605.14609v1)

**作者:** Adam Dawid Sztamborski `[一作]` (Institut de Robòtica i Informàtica Industrial, CSIC-UPC), Antonio Agudo `[通讯]` (Politechnika Lodzka)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了可微分的深度判别分析（DDA）损失，直接将 Fisher 判别准则融入语义分割网络训练。

**💡 创新点**

创新点在于将经典判别分析的类间方差最大化与类内方差最小化的目标嵌入到端到端损失中，使模型在不增加参数或推理成本的前提下获得更紧凑、可分离的特征。

**🔧 技术方法**

采用差分判别损失、批量散度矩阵、线性判别分析和深度网络（如 U-Net、AttU-Net、R2U-Net、U^2-Net）等技术。

**📊 数据集**

使用 DIS5K 大规模二分类分割基准数据集（包含 5,470 张高分辨率自然图像）。

**📈 对比分析**

通过与 BCE、Dice 等传统损失以及零射击 SAM 等基线模型在 DIS5K 的多子集上对比，DDA 在 IoU、F1、Fβ、AUC 等指标上平均提升 3–10%，尤其在边界质量和高复杂度场景中表现突出。

**⚠️ 局限性**

局限在于对多类别分割的推广尚未深入，且对极低对比度或高噪声场景仍易出现失败。

---

## 434. Compositional Sparsity as an Inductive Bias for Neural Architecture Design

**arXiv ID:** 2605.14764 | [PDF](https://arxiv.org/pdf/2605.14764v1)

**作者:** Hongyu Lin `[一作]` (University College London), Tomaso Aste `[通讯]` (University College London)

**通讯引用:** 9386 | [OpenAlex ID](https://openalex.org/A5050674002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将信息过滤网络生成的依赖结构映射到稀疏层次化神经网络，提出了一种数据驱动的稀疏深度学习架构，用于处理高维表格回归任务。

**💡 创新点**

创新点在于利用最大过滤团森林获取可解释的依赖拓扑，直接决定网络结构，实现了稀疏且层次化的权重连接，并通过全层读取提升信息利用。

**🔧 技术方法**

技术包括信息过滤网络（MFCF）、同伦神经网络（HNN）、相关系数/中位拆分的依赖矩阵构造、稀疏权重矩阵、全层读取以及Adam训练。

**📊 数据集**

实验使用人工合成数据（已知稀疏层次结构）和 OpenML-CTR23 表格回归基准集。

**📈 对比分析**

在与稠密 MLP、随机森林、XGBoost 等基线的共享最小调参协议下，HNN 在高维低样本场景下表现更稳健、参数更少，平均 R^2 排名优于 MLP，接近树模型。

**⚠️ 局限性**

局限性在于依赖结构仅通过线性相关性估计，可能忽略非线性或分层变化的交互，且在噪声小样本下的结构推断不稳。

---

## 435. AI Outperforms Humans in Personalized Image Aesthetics Assessment via LLM-Based Interviews and Semantic Feature Extraction

**arXiv ID:** 2605.14761 | [PDF](https://arxiv.org/pdf/2605.14761v1)

**作者:** Yoshia Abe `[一作]` (University of Tokyo), Yasuo Kuniyoshi `[通讯]` (University of Tokyo)

**通讯引用:** 11029 | [OpenAlex ID](https://openalex.org/A5010543059)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并实现了一个整合深度学习与大型语言模型的系统，能够通过LLM进行半结构化访谈主动挖掘目标个体的审美偏好，并利用低层次图像特征与高层次语义特征共同预测其对图像的审美评价。

**💡 创新点**

创新点在于将LLM用于主动访谈获取个体偏好，并利用LLM迭代生成高层次特征集合，与传统基于低层次特征的预测模型相结合，形成一套端到端的个性化审美预测框架。

**🔧 技术方法**

技术主要包括：Claude Sonnet 4.5 进行访谈与分析，Gemini 2.5 Flash Lite 负责特征适用性评估，ResNet‑50 CNN 作为低层次预测器，梯度提升回归器整合特征，并通过LLM实现特征探索与模型训练。

**📊 数据集**

数据集方面使用 PARA 照片数据集（共 300 张图像），每位 30 名参与者对其进行 1.0‑5.0 评分，并进行访谈；同时利用 PARA 的 GIAA 数据对低层次预测器进行预训练。

**📈 对比分析**

通过与 DL 预测器、不同版本 LLM 预测器、人工预测器以及同一受试者的再评估进行对比，使用 MAE 评估性能。提出的系统 MAE 为 0.549，显著低于 DL（0.561）和人类预测器，且在高评分图像上提升尤为明显，甚至低于同一受试者的时间波动。

**⚠️ 局限性**

局限性包括样本量仅 30 人（主要为 20 岁左右男性），访谈仅使用日语，图像仅为 PARA 照片，LLM 可能带来模型偏差，API 调用成本高，且对非语言化的审美偏好捕捉能力有限。

---

## 436. Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions

**arXiv ID:** 2605.14752 | [PDF](https://arxiv.org/pdf/2605.14752v1)

**作者:** Qirui Liu `[一作]` (South China University of Technology), Jia Zhu `[通讯]` (Zhejiang Normal University)

**通讯引用:** 5464 | [OpenAlex ID](https://openalex.org/A5031653957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两阶段知识蒸馏框架，先用标准蒸馏传递任务知识，再通过认知不确定性筛选Near‑miss与Hard‑hard高价值样本，并采用难度自适应损失提升学生误解检测模型的性能。

**💡 创新点**

双层边际选择机制基于教师的不确定性与置信差异精准挑选关键样本；难度自适应损失动态平衡软/硬标签；在极度稀缺的真实学生数据上实现轻量模型超越大型预训练模型。

**🔧 技术方法**

知识蒸馏、软标签与Cosine嵌入损失、基于概率边际与熵的难度度量、双层边际样本筛选、K折交叉验证与增量训练。

**📊 数据集**

MAP‑Charting（约3.7万条学生推理轨迹，包含正确、误解与模糊类别）与Algebra Misconception Benchmark（55类代数误解，220样本）。

**📈 对比分析**

与prompting型大型模型（GPT‑5、Claude‑4 等）以及直接微调的大模型（72B、120B）对比；两阶段蒸馏在MAP‑Charting上MAP@3 0.9585（提升17.8%）、准确率84.38%；在Algebra误解上准确率0.8438，显著高于GPT‑5 0.6773和72B模型0.8125。

**⚠️ 局限性**

K‑折划分导致训练开销高；对低质量数据提升有限，需更有效的数据合成策略。

---

## 437. Non-linear Interventions on Large Language Models

**arXiv ID:** 2605.14749 | [PDF](https://arxiv.org/pdf/2605.14749v1)

**作者:** Sangwoo Kim `[一作]` (Seoul National University), Sangwoo Kim `[通讯]` (Seoul National University)

**通讯引用:** 45335 | [OpenAlex ID](https://openalex.org/A5100452172)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可学习的非线性干预框架，用于在大型语言模型中精准操控隐式特征；

**💡 创新点**

创新点在于将线性干预推广为可逆非线性特征映射，并通过互换干预与自监督损失学习隐式特征；

**🔧 技术方法**

采用可逆非线性网络（i-ResNet）作为特征映射，结合互换干预、hinge损失以及中间层特征提取；

**📊 数据集**

使用Llama‑3‑8B‑Instruct和Qwen2.5‑7B‑Instruct模型，训练集包括2000个安全提示和2000个有害提示；

**📈 对比分析**

与线性干预基线（DIM、RDO）比较，非线性干预在单一位置即可达到相近或更高的StrongREJECT分数，同时总干预幅度降低两倍以上；

**⚠️ 局限性**

局限包括：只实现了i-ResNet可逆网络；干预位置仍需经验选择；无法完全克服线性干预的层位置限制。

---

## 438. Addressing Terminal Constraints in Data-Driven Demand Response Scheduling

**arXiv ID:** 2605.14741 | [PDF](https://arxiv.org/pdf/2605.14741v1)

**作者:** Maximilian Bloor `[一作]` (Imperial College London), Calvin Tsay `[通讯]` (Imperial College London)

**通讯引用:** 1125 | [OpenAlex ID](https://openalex.org/A5068409517)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出利用目标空间规划（GSP）与DDPG相结合的方法，以解决电化学过程在需求响应调度中长期终端约束的信用分配问题。

**💡 创新点**

创新点在于将终端约束转换为可离散化的子目标，通过学习时空抽象模型实现价值在整个周期中的快速传播，显著提升样本效率并抑制短视行为。

**🔧 技术方法**

采用的技术包括深度确定性策略梯度（DDPG）、潜在奖励塑形、子目标到子目标的表格模型以及基于图的值迭代。

**📊 数据集**

使用的仿真数据集为基准的空气分离单元（ASU）模拟流程。

**📈 对比分析**

与标准DDPG对比，在线GSP在80个周期内收敛约5000步，样本效率提升15%至20%，且能始终满足终端存储约束。

**⚠️ 局限性**

局限在于需要手工设计子目标空间，对不同工艺可能需重新网格划分，且对价格/需求变化的适应速度受子目标更新频率限制。

---

## 439. Video-Zero: Self-Evolution Video Understanding

**arXiv ID:** 2605.14733 | [PDF](https://arxiv.org/pdf/2605.14733v1)

**作者:** Ruixu Zhang `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 4137 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种无监督的问答生成与学习框架Video‑Zero，用于视频理解。

**💡 创新点**

创新点在于将视频中的时序证据定位作为问答生成与学习的核心，结合Questioner–Solver共进化循环实现证据驱动的自进化。

**🔧 技术方法**

采用基于Transformer的多模态语言模型（如Qwen3‑VL）、强化学习策略（GRPO）以及基于rollout一致性的伪标签与时间对齐奖励。

**📊 数据集**

使用600条来自Open‑o3‑Video的无标签视频以及13个公开基准（ANet‑RTL、ActivityNet、Charades‑STA、NExT‑GQA、LongVideoBench、MLVU、Video‑MME‑Long、LSDBench、VideoNIAH、LongVideoReason、MMVU、VideoMathQA、VideoMMMU）。

**📈 对比分析**

与VisPlay、V‑Zero等自进化基线以及MiMo、InternVL等不同模型架构对比，Video‑Zero在所有13个基准上均显著提升，尤其在时间定位与长视频推理任务中表现最为突出。

**⚠️ 局限性**

局限在于伪标签随迭代增加而噪声提升，导致收益递减；对极长视频的时序建模仍存在挑战。

---

## 440. AnchorRoute: Human Motion Synthesis with Interval-Routed Sparse Contro

**arXiv ID:** 2605.14716 | [PDF](https://arxiv.org/pdf/2605.14716v1)

**作者:** Pengcheng Fang `[一作]` (University of Southampton), Xiaohao Cai `[通讯]` (University of Southampton)

**通讯引用:** 63752 | [OpenAlex ID](https://openalex.org/A5045658136)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出AnchorRoute框架，通过稀疏锚点同时控制生成和后期修正实现人类运动合成。

**💡 创新点**

创新点在于将稀疏锚点构建成共享的锚点支架，既用于生成时的AnchorKV条件注入，又用于修正时的RouteSolver基于锚点间隔的残差路由投影，实现两阶段协同优化。

**🔧 技术方法**

采用冻结的Transition Masked Diffusion（TMD）文本到运动先验、AnchorKV轻量级条件路径、双上下文注入、以及软令牌投影的RouteSolver修正。

**📊 数据集**

在HumanML3D数据集上进行实验。

**📈 对比分析**

与多种先前稀疏控制方法（如SFControl、ControlNet等）对比，AnchorRoute在保持FID与Top-3 R-Precision的同时，显著降低控制误差（AnchorRoute w/ solver控制误差降至0.019，优于SFControl的0.036）。

**⚠️ 局限性**

主要局限是RouteSolver目前仅利用位置残差进行修正，未来可加入切向或方向信息以实现更丰富的方向感知修正。

---

## 441. StyleTextGen: Style-Conditioned Multilingual Scene Text Generation

**arXiv ID:** 2605.14708 | [PDF](https://arxiv.org/pdf/2605.14708v1)

**作者:** Zeyu Chen `[一作]` (Nankai University), Yu Zhou `[通讯]` (Nankai University)

**通讯引用:** 32076 | [OpenAlex ID](https://openalex.org/A5012041302)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出StyleTextGen框架，实现多语言场景文本的风格控制生成。

**💡 创新点**

创新点包括双分支风格编码器、文本风格一致性损失以及基于掩码的推理时风格注入，显著提升跨语言风格一致性与细粒度控制。

**🔧 技术方法**

技术手段涵盖基于Diffusion Inpainting的DiT模型、InternViT文本编码、SigLIP视觉先验、Q‑Former、AdaIN等，并加入自监督文本分割与风格一致性损失。

**📊 数据集**

使用ArtText Diffusion合成的双语风格数据，构建StyleText-CE（中英场景文本）以及AnyWord-Eval评测集进行实验。

**📈 对比分析**

在AnyWord-Eval和StyleText-CE上与TextFlux、Calligrapher等方法对比，Sen.Acc/NED提升显著，FID/LPIPS下降，显示出更优的文本准确性与风格相似度。

**⚠️ 局限性**

局限性在于对极复杂背景或极异体字形的跨语言一致性仍受限，且推理时的多步骤投影导致速度和复杂度较高。

---

## 442. MediaClaw: Multimodal Intelligent-Agent Platform Technical Report

**arXiv ID:** 2605.14771 | [PDF](https://arxiv.org/pdf/2605.14771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 443. SR-Platform: An Agentic Pipeline for Natural Language-Driven Robot Simulation Environment Synthesis

**arXiv ID:** 2605.14700 | [PDF](https://arxiv.org/pdf/2605.14700v1)

**作者:** Ben Wei Lim `[一作]` (Strike Robotics), Thanh Nguyen Canh `[通讯]` (Strike Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了 SR-Platform，一套生产级的代理系统，能够将自然语言工作空间描述自动转换为可直接运行的 MuJoCo 机器人仿真场景，包含资产检索/生成、布局规划和机器人模型合成。

**💡 创新点**

创新点：①将场景合成拆分为可审计、可缓存的四个阶段（LLM 场景规划、资产检索/生成、布局推理、MJCF 组装）；②引入 LLM 到 CadQuery 代码生成与向量检索相结合的资产工厂；③在布局阶段加入工业约束验证；④实现实时进度流式、异步作业队列和生产级监控，构成完整的可部署流水线。

**🔧 技术方法**

技术栈：大型语言模型（LLM）+ LangGraph 工作流；LLM‑>CadQuery 代码合成；向量检索（Qdrant）+ 缓存存储（MinIO）；Redis、InfluxDB、PostgreSQL、Docker、WebSocket；MuJoCo + Three.js 浏览器可视化；机器人模型库（UR5、Franka 等）。

**📊 数据集**

使用的数据集：30 天生产日志 611 次成功 LLM 调用；资产生成基准包含 45 个自由形状对象与 100 个机械 CAD 对象（T1/T2/T3 级别）；内部使用的多种 LLM 后端（GPT、Claude 等）进行性能评测。

**📈 对比分析**

对比方法：在资产生成阶段对不同 LLM 后端进行综合评分（速度、成功率、Chamfer Distance 几何精度），并记录端到端延迟、重试率和系统吞吐；实验结果显示 5 个对象、缓存缺失时端到端延迟约 50 秒（缓存命中可降至 30–40 秒），资产生成首次重试率 11.3%，系统吞吐 5 并发作业。与传统手工构建相比，显著提升了效率与可扩展性。

**⚠️ 局限性**

局限性：①资产生成仍有高失败率，LLM‑>CAD 代码可能产生语法或几何错误；②布局规划依赖 LLM，难以处理极大或高度不规则场景；③目前仅支持工业/半结构化工作空间，无法处理有机或可变形物体；④缺乏完整任务/奖励生成，无法直接产生训练数据；⑤端到端时延缺乏完整测量，主要基于 LLM 调用统计。

---

## 444. MonoPRIO: Adaptive Prior Conditioning for Unified Monocular 3D Object Detection

**arXiv ID:** 2605.14781 | [PDF](https://arxiv.org/pdf/2605.14781v1)

**作者:** Leon Davies `[一作]` (Loughborough University), Simon Sølvsten `[通讯]` (University of Southern Denmark)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5022447568)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对单目3D检测中尺寸预测不稳定问题的统一检测框架MonoPRIO，利用查询自适应先验条件来稳固尺寸估计。

**💡 创新点**

创新点在于：①离线构建类感知尺寸先验库；②对每个查询进行软混合先验路由并结合不确定性进行对数空间尺寸条件化；③在训练时对匹配正样本引入Cluster‑Aligned Prior（CAP）正则化，使尺寸预测更贴近先验模态；④仅改动尺寸分支，保持基础模型的效率和结构。

**🔧 技术方法**

核心技术包括：Transformer‑based monocular检测器（MonoDGP）、CLIP特征用于离线先验库构建、查询路由与不确定性估计、对数空间尺寸融合、CAP正则化及基于分数的优先级调节。

**📊 数据集**

在KITTI数据集上进行训练与评估，统一多类别（Car、Pedestrian、Cyclist）和单类（Car）两种设置。

**📈 对比分析**

与MonoDGP、MonoCLUE等方法比较，MonoPRIO在官方KITTI测试上实现了统一多类最高的AP（Car 18.93，Ped 10.74，Cyclist 8.75），在单类Car的3D AP上也位居首位；尺寸MAE和Outlier比例显著下降；在效率方面几乎不增加FLOPs，仅增加0.33M参数，延迟+1.24ms。

**⚠️ 局限性**

局限性：当视觉信息极少（如严重遮挡或远距离小目标）或目标尺寸与先验库不匹配时，路由不确定性高，先验可能误导尺寸估计；过强的先验正则化在完整数据集下可能导致过度约束；需要更丰富的先验覆盖和自适应强度调节。

---

## 445. Peng's Q($λ$) for Conservative Value Estimation in Offline Reinforcement Learning

**arXiv ID:** 2605.14779 | [PDF](https://arxiv.org/pdf/2605.14779v1)

**作者:** Byeongchan Kim `[一作]` (Seoul National University), Min-hwan Oh `[通讯]` (Seoul National University)

**通讯引用:** 73065 | [OpenAlex ID](https://openalex.org/A5100447410)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为 Conservative Peng's Q(λ) (CPQL) 的模型无关离线多步强化学习算法，利用 Peng's Q(λ) 操作符进行保守值估计，并可在离线训练后直接用于在线微调。

**💡 创新点**

创新点包括：①首次在离线 RL 中引入多步 PQL 操作符；②通过保守值估计与多步操作结合，天然实现行为正则化，消除过度悲观估计；③理论上证明混合策略性能至少不低于行为策略并缩小最优子最优差距；④在多步回溯上相较于单步方法表现更稳健、对保守参数 α 依赖性更低。

**🔧 技术方法**

核心技术：Peng's Q(λ) 多步回溯；保守 Q-learning (CQL) 损失与 log‑sum‑exp 结合；SAC 结构下的 Q‑函数更新；离线到在线微调框架；在梯度步骤中对 λ 进行递归计算。

**📊 数据集**

使用 D4RL 基准数据集：MuJoCo（HalfCheetah、Hopper、Walker2d）、Adroit（hammer、door、pen、relocate）和 AntMaze（umaze、medium、large）等多种数据集与不同经验水平（Random、Medium、Expert 等）。

**📈 对比分析**

与单步离线基线（TD3+BC、CQL、IQL、MCQ、MISA、CSVE、EPQ）以及多步基线（Uncorrected n‑step、Retrace、Tree‑backup）进行对比；CPQL 在 29 项任务中 22 项获得最高分，性能稳定且对 α 变化不敏感；在离线到在线微调中，CPQL→PQL 能显著避免初始性能下滑并实现更快更高的提升。

**⚠️ 局限性**

局限性：①多步回溯带来额外的计算开销，虽然实际运行时间增幅不大；②在低质量数据集上可能表现下降，单步更新在某些场景更为稳健。

---

## 446. UMo: Unified Sparse Motion Modeling for Real-Time Co-Speech Avatars

**arXiv ID:** 2605.14731 | [PDF](https://arxiv.org/pdf/2605.14731v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 447. Selective Safety Steering via Value-Filtered Decoding

**arXiv ID:** 2605.14746 | [PDF](https://arxiv.org/pdf/2605.14746v1)

**作者:** Bat-Sheva Einbinder `[一作]` (Technion IIT), Yaniv Romano `[通讯]` (Technion IIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Value-Filtered Decoding，一种在推理时通过基于值阈值过滤 token 的安全性增强方法。

**💡 创新点**

提供了阈值过滤的解析解与可控假阳性率上界，并证明在值估计误差下相较于 Gibbs 策略更稳健、且能保持对原模型的高相似度。

**🔧 技术方法**

使用 probe classifier 估计安全值、基于马尔可夫性质构建阈值过滤策略、拒绝采样实现采样、conformal risk 控制阈值、cosine similarity 评估相似度等技术。

**📊 数据集**

在多数据集上实验，包括 SafeCompletion、Harmful、以及其它安全评测数据集，并利用 LLM-as-a-judge 进行安全与有用性标注。

**📈 对比分析**

与无干预、Top‑K、Gibbs 采样等四种基线进行对比，结果显示在安全性、相似度和有用性三维度上均实现了最佳或近优权衡，且假阳性率严格控制在设定的上界内。

**⚠️ 局限性**

需要训练安全值分类器并拥有安全 hold‑out 数据集；仅适用于二值安全奖励，难以直接扩展到连续或多目标安全约束。

---

## 448. Generating HDR Video from SDR Video

**arXiv ID:** 2605.14703 | [PDF](https://arxiv.org/pdf/2605.14703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 449. Mechanical Enforcement for LLM Governance:Evidence of Governance-Task Decoupling in Financial Decision Systems

**arXiv ID:** 2605.14744 | [PDF](https://arxiv.org/pdf/2605.14744v1)

**作者:** José Manuel de la Chica Rodríguez `[一作]` (Santander AI Lab, Grupo Santander), Carlos Martí-González `[通讯]` (Santander AI Lab, Grupo Santander)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在受监管金融工作流程中，LLM 解释并执行自然语言治理政策时的代理失效问题，并提出治理质量评估指标；

**💡 创新点**

创新点在于首次定义并应用五个治理指标（CDL、DIU、FSR、FVS、ESD）以量化决策理由的合规性，并通过机械强制执行（硬门限、候选外部化、熵封装、理由质量检查）与文本治理进行对比；

**🔧 技术方法**

技术包括 Llama 3.1 70B 大模型、机械治理原语（硬门限、CEFL、E3、I6Q）、基于规则的文本分析、Bootstrap 置信区间和 Holm‑Bonferroni 校正；

**📊 数据集**

使用合成银行案例数据集（300 条案例/条件，5 类决策），包含风险、完整性、监管标记、金额、司法管辖区等特征，模拟四种结构/参数压力；

**📈 对比分析**

通过对比两种治理模式（文本治理 R1 vs 机械治理 R2），在 8 个实验单元中发现 R2 将空洞推迟率从 27% 降至 7.4%，信息利用率提升 2.6 倍，MCC 从 0.43 提升至 0.88；

**⚠️ 局限性**

局限在于仅在单一模型和合成数据上验证，缺乏跨模型及真实生产环境的测试，且文本治理的 27% 推迟率可能低估真实工业场景。

---

## 450. A Template-Driven Platform for Contextualised Researcher Profiles

**arXiv ID:** 2605.14722 | [PDF](https://arxiv.org/pdf/2605.14722v1)

**作者:** Serafeim Chatzopoulos `[一作]` (IMSI Athena RC), Thanasis Vergoulis `[通讯]` (IMSI Athena RC)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

BIP! Scholar平台提供多模板、情境化的研究者档案创建与评估机制，支持研究者和评估专家分别自定义与测试不同风格的档案展示；

**💡 创新点**

核心创新在于将传统以出版物为中心的档案模型转变为可配置、多维度、基于模板的表示框架，并允许评估专家实验和迭代优化模板；

**🔧 技术方法**

利用OpenAIRE和OpenAlex知识图谱进行元数据检索与丰富，采用PostgreSQL存储，前端通过React实现交互，后端使用Python/Flask并整合DeepSeek V4 Flash LLM 进行自动摘要与文本生成；

**📊 数据集**

主要数据集包括ORCID公开记录、OpenAIRE Graph和OpenAlex数据，并在本地构建的元数据数据库中统一管理；

**📈 对比分析**

在功能演示与对比场景中展示与Google Scholar、ResearchGate等传统平台相比，BIP! Scholar 在多模板展示、交互过滤和AI辅助文本生成方面表现优越，但本文未给出量化指标或大规模性能评估；

**⚠️ 局限性**

局限性包括依赖外部开放数据质量、模板设计与评估仍需人工投入、缺乏跨学科大规模验证以及对实时性能与可扩展性的详细评估不足。

---

## 451. IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation

**arXiv ID:** 2605.14712 | [PDF](https://arxiv.org/pdf/2605.14712v1)

**作者:** Shijie Lian `[一作]` (HUST), Kai Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用最近视觉历史编码的历史条件模仿学习框架，解决视觉-语言-动作模型在局部意图歧义下的片段生成不一致问题，并构建了12任务的短期观察别名基准。

**💡 创新点**

创新点在于通过冻结的VGGT‑1B视觉历史编码器和门控交叉注意力，将短期视觉历史压缩为紧凑的短期意图表示，直接条件化动作片段生成；同时提供了专门评估观察别名的基准，揭示帧条件策略的局限。

**🔧 技术方法**

技术实现采用Qwen3‑VL 4B视觉‑语言主干、VGGT‑1B冻结历史编码器、门控交叉注意力融合、DiT‑based流匹配动作头，并使用条件流匹配损失进行训练。

**📊 数据集**

数据集方面，主要使用RoboTwin2的12任务短期别名基准（与训练数据匹配）以及标准模拟基准SimplerEnv、LIBERO、RoboCasa，用于训练与评估。

**📈 对比分析**

与Qwen3‑VL‑GR00T及其他VLA基线对比，Ambiguity基准上成功率从9.0%提升至45.8%，交叉片段一致性误差降低17.6%；SimplerEnv平均成功率达到72.9%（基线65.3%），LIBERO‑Long 97.4%（基线92.0%），RoboCasa 57.0%（基线54.6%）。

**⚠️ 局限性**

局限性包括：仅利用短期视觉历史，无法处理需要更长记忆或从大闭环偏差中恢复的情形；实验仅在仿真环境中验证，缺乏物理机器人测试；对更复杂的意图识别和自适应历史选择仍需进一步研究。

---

## 452. Hardness of Burning Number Problem on Regular Graphs

**arXiv ID:** 2605.14730 | [PDF](https://arxiv.org/pdf/2605.14730v1)

**作者:** Dhanyamol Antony `[一作]` (Indian Institute of Science Education and Research Thiruvananthapuram), Shashanka Kulamarva `[通讯]` (Kyoto University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5055773603)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明了在连通三度图以及任意固定度数 d≥4 的连通正则图中，火灾数问题（Burning Number Problem）是 NP‑完整的，并且在这些图类上是 APX‑难的。

**💡 创新点**

首次设计了针对正则图的精细化图构造（BTP‑、Y‑、C‑等结构）以及域/所有权分析，并给出从三度图到任意固定度数 d≥4 的构造，完成了连通正则图的硬度证明。

**🔧 技术方法**

利用多层嵌套的图构造、分裂、BTP‑装置、Y‑与C‑组件的设计，结合组合论和图论中的距离传播分析，以及 L‑reduction 技术实现 APX‑难度。

**📊 数据集**

本研究为理论性工作，无实验数据集。

**📈 对比分析**

通过证明决策版本为 NP‑完整并使用 L‑reduction 推导 APX‑难度，对比已知在路径、树、网格等特殊图类上可多项式可解或存在近似算法的情况，表明连通正则图属于更困难的类别。

**⚠️ 局限性**

结果仅适用于连通正则图，构造过程需要特定常数 c 与复杂结构，无法直接推广到所有图类；且仅给出近似困难的上界，没有提供有效的近似算法。

---

## 453. Beyond What to Select: A Plug-and-play Oscillatory Data-Volume Scheduling for Efficient Model Training

**arXiv ID:** 2605.14773 | [PDF](https://arxiv.org/pdf/2605.14773v1)

**作者:** Suorong Yang `[一作]` (National University of Singapore), Soujanya Poria `[通讯]` (Nanyang Technological University)

**通讯引用:** 23295 | [OpenAlex ID](https://openalex.org/A5033376109)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个可插拔的振荡式数据量调度框架（PODS），通过动态调整每个训练阶段使用的数据比例来实现隐式正则化与优化稳定性的平衡，提升训练效率与泛化性能。

**💡 创新点**

创新点在于将数据量本身视为可调的优化维度，设计了周期性高低比例交替的调度策略，并通过理论证明其隐式正则化作用；此外实现了极简、任务无关的实现方式，能与任何现有样本重要性评估方法无缝结合。

**🔧 技术方法**

核心技术包括：1）对随机梯度下降的二阶展开推导隐式正则化项；2）基于累计预算约束的振荡比例调度算法；3）默认的基于损失的硬挖掘样本选择；4）在多任务框架下的轻量级实现。

**📊 数据集**

实验使用了广泛的数据集：图像分类（CIFAR‑10/100、Tiny‑ImageNet、ImageNet‑1k、Fine‑grained 4 任务、长尾 2 任务、OOD 4 任务）；目标检测（MS‑COCO + YOLOv8/RT‑DETR）；以及大型语言模型指令微调（Qwen‑2.5‑7B‑Instruct、LLaMA‑3.2‑3B、Qwen‑3‑4B/8B 在 MMLU、BBH、算数推理等任务）。

**📈 对比分析**

与 17 大类现有静态/动态数据选择方法以及多种优化器在相同训练预算下进行对比。PODS 在 30%–70% 选择比例下均能获得最高或接近全数据的准确率，同时显著降低 GPU 训练时长（ImageNet‑1k 训练成本从 140h 降至 84h，LLM 指令调优训练时间减半）。在多任务、跨模型、跨优化器设置中表现稳健，且对不同 ε 值的敏感性低。

**⚠️ 局限性**

局限性包括：1）调度参数（ε、低/高比例）虽可自动化但仍需依据目标比例微调；2）在极低比例（如 <10%）时可能导致优化不稳定；3）目前主要验证于分类、检测和 LLM 指令微调，尚未覆盖序列生成、强化学习等更复杂训练场景；4）隐式正则化理论基于随机抽样假设，实际对非均匀采样的解释仍需深入。

---

## 454. Persian MusicGen: A Large-Scale Dataset and Culturally-Aware Generative Model for Persian Music

**arXiv ID:** 2605.14765 | [PDF](https://arxiv.org/pdf/2605.14765v1)

**作者:** Mohammad Hossein Sameti `[一作]` (Sharif University of Technology), Mahdieh Soleymani Baghshah `[通讯]` (Sharif University of Technology)

**通讯引用:** 1189 | [OpenAlex ID](https://openalex.org/A5069082023)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文针对波斯音乐缺乏数据问题，构建了首个涵盖流行、传统、当代等超过900小时的高质量波斯音乐大规模数据集，并在此数据集上微调MusicGen模型，实现波斯音乐的生成。

**💡 创新点**

创新点在于首次系统性收集多样化波斯音乐并利用分层微调策略（无监督域适配、乐器聚焦、文本对齐）使生成模型捕捉到Dastgah调式、微音程与节奏特征。

**🔧 技术方法**

采用MusicGen的Transformer解码器与EnCodec离散音频编码器，结合音频源分离、自动标签与LLaMA3.2 3B生成字幕的多模态预处理，以及三阶段微调技术。

**📊 数据集**

使用新构建的PMG（Persian Music Generation）数据集，包含约16k轨道、900小时音频及自动标注的关键、节奏、能量、乐器与情绪标签。

**📈 对比分析**

通过Kullback–Leibler Divergence与Chroma Cosine Similarity两项客观指标，以及文本+音频混合条件实验，微调模型在传统、多音轨与流行子集上均取得比原始MusicGen更低的KLD和更高的色度相似度。

**⚠️ 局限性**

限制包括数据集以流行音乐为主导致模型偏向该风格，自动标签与字幕噪声、未评估微音程与调式精度、仅使用小模型、以及缺乏专家听评验证等。

---

## 455. XDomainBench: Diagnosing Reasoning Collapse in High-Dimensional Scientific Knowledge Composition

**arXiv ID:** 2605.14754 | [PDF](https://arxiv.org/pdf/2605.14754v1)

**作者:** Gong Zhiren `[一作]`, Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**通讯引用:** 7064 | [OpenAlex ID](https://openalex.org/A5027969322)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 XDomainBench 这一用于评估大型语言模型在交互式跨学科科学推理中的诊断性基准。

**💡 创新点**

创新点在于通过可控的“组合顺序”和“混合结构”两维构造多领域交互情景，并为每个会话提供轨迹层面的诊断信号，揭示模型推理崩溃的直接与间接机制。

**🔧 技术方法**

采用了多轮交互式生成、嵌入相似度驱动的领域组合、基于模型集成与人类审核的验证流程，以及基于语义评审的性能测度等技术。

**📊 数据集**

数据集包含 8,598 个交互式会话，覆盖 20 个科学领域、4 种任务类型（多选、事实问答、推理、代码）和 8 条难度/混合轨迹模式，已发布在 Hugging Face 上。

**📈 对比分析**

在对多种公开与私有 LLM（含 MoE 模型）进行零样本评估时发现，随着组合顺序从 1 到 4 的提升，模型准确率、F1 与 SessionSuccess 逐渐下降，MoE 模型表现最优。

**⚠️ 局限性**

局限性包括闭合书本设置（不使用检索/工具）、构造过程对种子池与模板的依赖、缺乏完整科研工作流程模拟，以及代码类任务实例相对稀缺。

---

## 456. Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining

**arXiv ID:** 2605.14747 | [PDF](https://arxiv.org/pdf/2605.14747v1)

**作者:** Weimin Xiong `[一作]` (Peking University), Hao Tian `[通讯]` (Xiaomi)

**通讯引用:** 767 | [OpenAlex ID](https://openalex.org/A5040505108)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Video2GUI框架，自动从海量互联网视频中提取GUI交互轨迹并构建WildGUI数据集

**💡 创新点**

首次实现端到端无监督从视频到精确交互轨迹的自动化流程，并覆盖多平台多语言场景

**🔧 技术方法**

结合元信息过滤、视频质量评分、轨迹提取与空间定位等大模型技术（Qwen2.5、Gemini‑3‑Pro等）和两阶段预训练策略

**📊 数据集**

WildGUI：12 M交互轨迹、124 M截图，覆盖1 500+应用与网站，属于目前最大的开源GUI预训练数据集

**📈 对比分析**

在多项GUI grounding、离线与在线agent基准上，WildGUI预训练的Qwen2.5‑VL/Mimo‑VL模型分别提升5–20%并达到或超越现有最优水平

**⚠️ 局限性**

仍受限于大模型的成本、对极低质量或非英语视频的处理不足，以及在实时动态环境下的进一步适配挑战

---

## 457. Betweenness Central Nodes Under Uncertainty: An Absorbing Markov Chain Approach

**arXiv ID:** 2605.14743 | [PDF](https://arxiv.org/pdf/2605.14743v1)

**作者:** Wencheng Bao `[一作]` (University of Illinois Urbana-Champaign), Chrysafis Vogiatzis `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 641 | [OpenAlex ID](https://openalex.org/A5018915189)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在随机网络中计算介数中心度的新框架，并通过吸收马尔可夫链（AMC）来描述随机网络中中心节点随时间的演化；

**💡 创新点**

创新点在于将中心节点序列压缩为吸收状态的MC，利用预吸收时间分布衡量节点重要性，并提供对不确定性（边失效、权重随机）下的中心度稳健性分析；

**🔧 技术方法**

采用吸收马尔可夫链理论、期望前吸收时间（fundamental matrix）计算，配合 Monte Carlo 采样估计链转移概率，以及行-wise扰动分析、Wasserstein 与 KL 效率度量；

**📊 数据集**

实验数据集包括 Erdős–Rényi、Watts–Strogatz 随机图以及 Les Misérables 共现网络，并对这些网络在随机边失效/权重扰动下进行评估；

**📈 对比分析**

与传统基于随机游走或最短路径的中心度方法相比，该方法能够识别少量主导节点，显示排名对扰动的稳健性；在实验中，它在不同网络中均能聚焦于少数顶点，并对排名变化给出定量的敏感度界定；

**⚠️ 局限性**

局限性包括对候选集的去向（仅保留单一中心节点）可能导致信息损失、估计依赖于大量 Monte Carlo 采样且需要稳定化处理、以及对时间相关性和更复杂结构的模型扩展仍不充分。

---

## 458. BioHuman: Learning Biomechanical Human Representations from Video

**arXiv ID:** 2605.14772 | [PDF](https://arxiv.org/pdf/2605.14772v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 459. Probabilistic Verification of Recurrent Neural Networks for Single and Multi-Agent Reinforcement Learning

**arXiv ID:** 2605.14758 | [PDF](https://arxiv.org/pdf/2605.14758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 460. Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners

**arXiv ID:** 2605.14709 | [PDF](https://arxiv.org/pdf/2605.14709v1)

**作者:** Qingyang Liu `[一作]` (Shanghai Jiao Tong University), Li Niu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 31992 | [OpenAlex ID](https://openalex.org/A5111709519)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

提出了自适应交错推理框架，统一模型可根据指令复杂度在直接生成、自我反思和多步规划三种模式之间动态切换，解决了“理解-生成”鸿沟和两大瓶颈；

**💡 创新点**

核心创新在于（1）设计层级化数据构建管线，生成包含直接生成、反思和多步规划的交错样本；（2）构建两阶段训练：SFT + GRPO强化学习，加入步骤级推理奖励与组内复杂度惩罚，使模型既保持高质量又提升推理效率；

**🔧 技术方法**

采用大型多模态语言模型（如Gemini-3-Pro-Image、Qwen-3VL-235B）作为生成器与分析器，利用自监督微调与GRPO强化学习，并使用自定义奖励函数（结果、格式、步骤推理）与复杂度惩罚；

**📊 数据集**

构建了约5万条高质量交错推理数据集，涵盖直接生成、反思、规划三种模式，且每条样本均通过人工审核；

**📈 对比分析**

在Text‑to‑Image（GenEval）、Image Editing（KRIS‑Bench）和Anything‑to‑Image（OmniContext）三大基准上，方法分别取得0.89、80.18和9.35的领先分数，显著优于Emu3.5、VACoT、FLUX等开源/闭源基线，且在RL阶段平均生成图像数从2.45下降到1.56，提升了推理效率；

**⚠️ 局限性**

限制包括：依赖昂贵的多模态分析器与生成器，训练与推理成本较高；在极端复杂或非结构化指令下，规划与反思步骤仍可能出现误差；数据集规模虽大，但仍不足以覆盖所有多模态场景。

---

## 461. EponaV2: Driving World Model with Comprehensive Future Reasoning

**arXiv ID:** 2605.14696 | [PDF](https://arxiv.org/pdf/2605.14696v1)

**作者:** Jiawei Xu `[一作]` (Nankai University), Wei Yin `[通讯]` (Horizon Robotics)

**通讯引用:** 2019 | [OpenAlex ID](https://openalex.org/A5100777411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种完全无人工感知标注的驾驶世界模型，利用未来深度与语义预测提升轨迹规划。

**💡 创新点**

创新点在于：①通过预测未来深度图与基于 SAM3 的语义特征，构建更完整的三维与语义世界推理；②将流匹配规划与 GRPO（组相对策略优化）相结合，进一步提升规划精度。

**🔧 技术方法**

核心技术包括：DINO‑Tok 视频特征编码、VLM‑初始化的世界模型、Rectified Flow Matching 轨迹与图像预测头、Depth‑Anything‑V3 伪深度监督、SAM3 伪语义监督、Flow Matching GRPO 强化学习调优。

**📊 数据集**

训练数据：nuPlan 与 nuScenes 进行未来推理能力构建；评估数据：NAVSIM v1 与 v2（包括 navtest、navhard）进行规划性能测试。

**📈 对比分析**

在 NAVSIM v1 上与感知驱动模型相当，SOTA perception‑free 模型提升 1.3 PDMS；在 NAVSIM v2 navhard 上提升 5.5 EPDMS，显示出更强的场景推理与规划精度。

**⚠️ 局限性**

局限性在于：伪深度与伪语义标签来自基础模型，存在误差，导致模型仍无法突破感知驱动模型的性能上限；未来需改进标签质量或引入更精准的自监督方法。

---

## 462. Mat2Boundary: Treating User-Defined Boundary Condition as SpMV for Distributed PDE Solvers on Block-Structured Grids

**arXiv ID:** 2605.14780 | [PDF](https://arxiv.org/pdf/2605.14780v1)

**作者:** Yanzheng Cai `[一作]` (Tsinghua University), Wenguang Chen `[通讯]` (Tsinghua University)

**通讯引用:** 5183 | [OpenAlex ID](https://openalex.org/A5103141832)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种专为分布式块结构网格设计的DSL与编译器，用于将用户自定义边界条件统一建模为稀疏矩阵-向量乘法（SpMV）并自动生成高性能的矩阵无关（matrix‑free）核和通信调度。

**💡 创新点**

创新点包括：1) 把多种边界处理（填零、halo复制、对称/循环映射、纯函数、边缘同步、用户自定义插值等）抽象为可组合的SpMV子矩阵；2) 在多阶段编程框架下结合多维划分与Polyhedral分析，自动消除冗余迭代空间、生成矩阵无关的内核；3) 设计专用通信后端，利用Polyhedral划分的凸子集实现高效远程数据收集；4) 通过IR重用与JIT特化实现可扩展的编译流程。

**🔧 技术方法**

使用了多阶段编程（LMS/轻量级多阶段）、Polyhedral分析（FreeTensor+isl）、IR重用与JIT优化、基于CSR/ELL的子矩阵API、专用通信库、以及Python/LLVM等技术栈。

**📊 数据集**

数据集主要是两类：1）基于立方体球网的浅水方程（HOPE、MCV）使用 6×90×90 或 6×1440×1440 分辨率；2）HPCG 测试使用 224³（单机）和 1280³（大规模）网格。

**📈 对比分析**

与手写的Fortran/C++实现及PyTorch实现进行对比，衡量指标包括：代码行数、BC核加速比、整体加速比、强伸缩效率。实验表明：BC代码量降低 70%+，BC核加速最高 7.6×，单核整体加速 1.9×、单机 4.75×，在 1,344 CPU 核上实现 72%–88% 的强伸缩。

**⚠️ 局限性**

局限性：依赖静态网格拓扑，不支持自适应网格或动态重划分；通信与计算未显式重叠；GPU 迁移需要额外初始化开销；对极大规模隐式算子时编译时间仍显高，尽管已通过 IR 重用缓解。

---

## 463. EVA: Editing for Versatile Alignment against Jailbreaks

**arXiv ID:** 2605.14750 | [PDF](https://arxiv.org/pdf/2605.14750v1)

**作者:** Yi Wang `[一作]` (ShanghaiTech University), Wenjie Wang `[通讯]` (ShanghaiTech University)

**通讯引用:** 2118 | [OpenAlex ID](https://openalex.org/A5100368534)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EVA（Editing for Versatile Alignment）框架，利用直接模型编辑技术在 LLM 和 VLM 中精准修正导致 jailbreak 攻击的内部激活模式，从而实现安全对齐。

**💡 创新点**

创新点包括：① 将模型编辑作为安全防御手段，专门定位 MLP 的关键值与下投影权重；② 针对视觉模态引入跨模态注意力机制提取视觉关键 token；③ 统一文本与视觉编辑流程，只更新少量参数，保持模型原有性能；④ 通过闭式最小二乘更新实现高效且可解释的参数修正。

**🔧 技术方法**

技术细节：使用 GPT‑4o 识别恶意 token，生成多样化变体求平均 key；构造安全目标 value 并加入 KL 正则；通过最小二乘求解 W_down 的闭式更新；选取 MEMIT/EasyEdit 定义的关键层；视觉关键 token 通过跨模态注意力得分挑选；最终形成统一的编辑矩阵。

**📊 数据集**

数据集：LLM 侧使用 HarmBench、AdvBench、JailbreakBench、MaliciousInstruct；VLM 侧使用 MM‑SafetyBench、MultiTrust、FigStep、Hades、ADV‑16；评估后续任务包括 MT‑Bench、BoolQ、MuTual、CoNLL03、RTE、GSM8K、SST2、SAMSum；VLM 下游任务评估采用 MM‑Vet‑v2、MMMU、MMStar。

**📈 对比分析**

与 LoRA、SafeDecoding、LED、Circuit Breakers 等基线对比，EVA 在各攻击场景下 ASR 显著降低（多数为 0%），且在 MT‑Bench 及多项下游任务中保持或略高于原始模型表现；训练时间约 0.4 h，推理时无额外开销，显示出优异的安全‑效能平衡。

**⚠️ 局限性**

局限性：依赖高质量恶意 token 提取（如 GPT‑4o）；编辑层与关键 token 的选择仍需经验或额外验证；对更强的自适应攻击存在一定恢复风险；安全目标固定为统一拒绝文本，可能限制对多样化拒绝策略的学习；目前仅在若干 LLM/VLM 上验证，未探讨更大规模或动态多模态（视频）模型的适用性。

---

## 464. On Strong Equivalence Notions in Logic Programming and Abstract Argumentation

**arXiv ID:** 2605.14721 | [PDF](https://arxiv.org/pdf/2605.14721v1)

**作者:** Giovanni Buraglio `[一作]` (TU Wien), Stefan Woltran `[通讯]` (TU Wien)

**通讯引用:** 4923 | [OpenAlex ID](https://openalex.org/A5006053030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文研究了逻辑程序（LP）与抽象论证框架（AF、CAF）在强等价性（strong equivalence）上的差异，提出了一种新的更新操作——规则细化（Rule Refinement），并在此基础上定义了新的强等价性概念，以恢复两种形式主义在动态情境下的兼容性。

**💡 创新点**

创新点在于：①首次引入规则细化作为LP的更新方式，使得LP的扩展与AF/CAF的攻击扩展保持一致；②在语法层面提供了新的强等价性定义，并通过kernel、loop等技术证明该等价性与AF/CAF的稳定kernel强等价性等价；③对well‑formed CAF进行了动态适配，并给出了相应的更新操作与等价性判定。

**🔧 技术方法**

主要技术手段包括：稳定模型与稳定语义的翻译；从LP到AF/CAF（以及反向）的映射；引入unbound attack与well‑formed CAF的概念；对规则和攻击进行kernel化（去除loop、删去可消除的负体）；规则细化的语法定义与结合；以及一系列形式化证明（如kernel不变性、等价性定理）。

**📊 数据集**

该工作完全基于理论分析，不涉及实验数据或公开数据集。

**📈 对比分析**

比较方法主要是形式化证明和等价性定理的推导，作者通过一系列等价性定理展示了新的强等价性与AF/CAF强等价性的等价性；并未进行实验评估，因而没有性能数值可供展示。

**⚠️ 局限性**

局限性包括：①研究仅覆盖atomic LP（包括h-unique和普通atomic），未扩展到一般的正常逻辑程序；②对正体（positive dependencies）的处理尚未完善；③所给定的更新与等价性定义在更复杂的AF变体（如抽象对话框架、支持型论证框架）中的适用性尚未验证；④缺乏实现或实证评估，无法评估实际应用中的计算效率与可扩展性。

---

## 465. Adapting AlphaEvolve to Optimize Fully Homomorphic Encryption on TPUs

**arXiv ID:** 2605.14718 | [PDF](https://arxiv.org/pdf/2605.14718v1)

**作者:** Shruthi Gorantala `[一作]` (Google), Amir Yazdanbakhsh `[通讯]` (Google DeepMind)

**通讯引用:** 2008 | [OpenAlex ID](https://openalex.org/A5070172290)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

利用AlphaEvolve将大型语言模型与进化搜索结合，自动化探索并优化在TPU上的全同态加密（FHE）核心算子，实现显著的延迟降低。

**💡 创新点**

首次将代理式AI与硬件反馈闭环相结合，自动化驱动编译器调优和微架构优化，突破传统人工调优的瓶颈。

**🔧 技术方法**

AlphaEvolve、LLM代码生成、进化搜索、XProf硬件回溯、XLA编译、JAX/Pallas接口、严格的功能和安全性验证。

**📊 数据集**

使用随机生成的加密输入向量与Google Cloud TPUv5e实际执行数据进行基准测试，无外部公开数据集。

**📈 对比分析**

与现有Jaxite、CROSS等手工优化实现对比，测量单核TPUv5e延迟，TFHE bootstrap降低2.5×，CKKS rotation提升1.31×，CKKS multiplication提升1.18×。

**⚠️ 局限性**

需要人工复核、对LLM误差的安全性约束、仅针对当前TPU架构、功能等价检测仍不完善、难以直接推广至其他加速器。

---

## 466. Vision-Core Guided Contrastive Learning for Balanced Multi-modal Prognosis Prediction of Stroke

**arXiv ID:** 2605.14710 | [PDF](https://arxiv.org/pdf/2605.14710v1)

**作者:** Liren Chen `[一作]` (East China University of Science and Technology), Ting Xiao `[通讯]` (East China University of Science and Technology)

**通讯引用:** 4052 | [OpenAlex ID](https://openalex.org/A5085667561)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一个三模态融合框架，结合脑MRI图像、LLM自动生成的诊断文本和结构化临床表格数据，通过视觉条件双对齐融合模块（VDAFM）实现深度跨模态交互，提升缺血性卒中预后预测精度。

**💡 创新点**

创新点包括：①利用大语言模型生成语义丰富的诊断文本，将其作为正则化视角；②VDAFM将视觉特征映射到文本空间，并通过双语义对齐损失和跨模态对比损失实现视觉-文本深度对齐；③首次实现缺血性卒中预后预测的三模态融合，克服了传统双模态方法的局限。

**🔧 技术方法**

采用了大语言模型（qwen‑vl‑max）生成文本，Bio‑BERT提取文本特征，MedicalNet/ResNet‑50提取图像特征，MLP进行维度对齐，Transformer实现联合编码，双对齐损失和对比损失保证语义一致性，同时使用SMOTE、MixUp、Dropout等正则化手段。

**📊 数据集**

使用了来自上海同济医院和新华医院的729例多模态数据（脑MRI、临床表格及LLM生成的诊断报告），并在四家医院（同济、新华、东方、浦东）进行留一医院验证。

**📈 对比分析**

与单模态机器学习、深度学习和现有多模态方法（Concat、EarthMind、Fusion、SparseMM）对比，模型在AUC、ACC、F1等指标上分别达到81.06%、81.16%、69.26%，比传统基线提升约10% AUC、5% F1，并在留一医院验证中保持领先表现。

**⚠️ 局限性**

局限性包括对LLM生成文本质量和提示设计的依赖、样本量相对有限、对外部数据集验证不足，以及VDAFM模块的计算复杂度和对不同设备/人群的域漂移仍需进一步研究。

---

## 467. NeuroAtlas: Benchmarking Foundation Models for Clinical EEG and Brain-Computer Interfaces

**arXiv ID:** 2605.14698 | [PDF](https://arxiv.org/pdf/2605.14698v1)

**作者:** Konstantinos Kontras `[一作]` (KU Leuven), Maarten De Vos `[通讯]` (KU Leuven)

**通讯引用:** 15751 | [OpenAlex ID](https://openalex.org/A5064593698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了NeuroAtlas，一个统一评测平台，用于在癫痫、睡眠、脑龄预测和BCI四大EEG领域对EEG基础模型、通用时间序列模型及监督预训练模型进行线性探针评估。

**💡 创新点**

创新点在于覆盖42个数据集、约260k小时EEG，使用临床相关指标、统一预处理并对比三类模型，揭示EEG-FM在跨域泛化和临床实用性上的局限。

**🔧 技术方法**

采用自监督预训练（掩码重建、对比学习）、统一的高通滤波/去直流、线性探针训练以及跨域评测流程。

**📊 数据集**

使用来自癫痫发作检测、睡眠分期、脑龄预测和BCI等领域的42个公开数据集，累计约260k小时EEG。

**📈 对比分析**

通过冻结编码器、训练线性分类器并计算领域特定指标，结果显示EEG-FM在部分任务上优于监督模型，但未能持续超越通用TS-FM；模型排名随数据集显著变动。

**⚠️ 局限性**

限制包括仅使用线性探针评估、未考虑微调或少量学习、部分先进模型不可用以及评测受预处理统一和数据集可用性的影响。

---

## 468. Composable Crystals: Controllable Materials Discovery via Concept Learning

**arXiv ID:** 2605.14769 | [PDF](https://arxiv.org/pdf/2605.14769v1)

**作者:** Nian Liu `[一作]` (National University of Singapore), Xavier Bresson `[通讯]` (National University of Singapore)

**通讯引用:** 10185 | [OpenAlex ID](https://openalex.org/A5031210396)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于VQ‑VAE学习的晶体概念组合框架，用于可控的 de‑novo 晶体生成；

**💡 创新点**

创新点在于把可提取、可解释、可泛化、可验证的晶体概念作为构件，利用组合生成器在保持稳定性与唯一性的同时显著提升结构新颖度，突破传统黑盒随机采样的局限；

**🔧 技术方法**

主要技术包括VQ‑VAE用于提取离散晶体概念、DDPM与Transformer实现概念生成与条件化生成、以及无监督的Classifier‑Free Guidance等；

**📊 数据集**

实验数据集为 MP‑20（≤20 个原子、实验稳定晶体）以及其扩展版本 Alex‑MP‑20，后者用于检验概念的跨分布泛化；

**📈 对比分析**

通过与 CDVAE、DiffCSP、FlowMM、MatterGen 等多种基线在 V.S.U.N. 指标及 DFT 评估下比较，本文方法在 Novelty 上提升约 50%（总提升 53%），在 DFT 评估的 V.S.U.N. 指标上相较最强基线 MatterGen 提升 17.8%，并保持了稳定性与唯一性；

**⚠️ 局限性**

局限性包括：仅有约 61% 的 Alex‑MP‑20 结构能被成功重构，部分概念缺乏人类可解释性，对完全新分布的泛化仍有限。

---

## 469. EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding

**arXiv ID:** 2605.14742 | [PDF](https://arxiv.org/pdf/2605.14742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 470. Streaming Speech-to-Text Translation with a SpeechLLM

**arXiv ID:** 2605.14766 | [PDF](https://arxiv.org/pdf/2605.14766v1)

**作者:** Titouan Parcollet `[一作]` (Samsung), Rogier C. van Dalen `[通讯]` (Samsung)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种“intermixed” SpeechLLM 架构，实现实时流式语音到文本的跨语言翻译；

**💡 创新点**

创新点在于：①让 LLM 自学习等待策略，通过输出 wait token 决定何时继续等待或发射；②引入早期退出（early‑exit）等待策略降低设备能耗；③采用短语级对齐生成训练监督，避免传统词对词对齐导致的错误；

**🔧 技术方法**

使用技术包括预训练 Conformer 编码器 + 3B LLM+LoRA、intermixed 结构、wait penalty、early‑exit head、LLM 生成短语对齐、SimAlign 对比、COMET + 平均逻辑延迟评估；

**📊 数据集**

数据集：CoLiMu（LibriSpeech + CommonVoice + MuST‑C，约 3,700 小时），Fleurs 验证/测试集，SilFleurs（带前置噪声），内部讲座韩语数据；

**📈 对比分析**

通过与 Bestow（fixed wait‑k / AlignAtt）以及 concatenated baseline 进行对比；在 Fleurs 上，intermixed 以 1–2 秒平均逻辑延迟达到与离线 baseline 相当或更好的 COMET 评分；相较于最快的 Bestow，延迟降低 2.3×、质量提升 19.4%；在 SilFleurs 上保持稳定，避免了传统策略的灾难性崩溃；

**⚠️ 局限性**

局限性包括：①需要大量高质量短语对齐，LLM 提示生成的对齐可能不完全可靠；②LLM 调用次数高，尤其在没有 early‑exit 的情况下能耗显著；③对极端噪声或极慢/极快语速的鲁棒性仍有限；④模型对不同语言对的适应性需要进一步验证。

---

## 471. Crys-JEPA: Accelerating Crystal Discovery via Embedding Screening and Generative Refinement

**arXiv ID:** 2605.14759 | [PDF](https://arxiv.org/pdf/2605.14759v1)

**作者:** Nian Liu `[一作]` (National University of Singapore), Xavier Bresson `[通讯]` (National University of Singapore)

**通讯引用:** 10185 | [OpenAlex ID](https://openalex.org/A5031210396)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究晶体生成中的稳定性与新颖性权衡，并提出能量感知潜在空间与筛选‑细化管线以提升生成质量。

**💡 创新点**

通过JEPA构建基于形成能的能量感知嵌入空间，并用该空间作为高效稳定性筛选器，结合筛选‑细化循环突破稳定性‑新颖性陷阱。

**🔧 技术方法**

采用Transformer编码器、InfoNCE对比学习、能量加权损失、扩散生成模型及联合嵌入预测架构JEPA，利用欧氏距离嵌入相似度进行筛选。

**📊 数据集**

使用Material Project 20（MP‑20）和Alex‑MP‑20作为生成评估数据集，训练数据来自Material Project v.2022.10.28与MP‑trj，验证采用DFT（MatterSim）与MLFF。

**📈 对比分析**

与多种基线生成器（CDVAE、DiffCSP、MatterGen等）在V.S.U.N评价指标上对比，利用DFT评估稳定性；在MP‑20上S.U.N指标提升至81.4%，Alex‑MP‑20提升82.6%，显著优于现有方法。

**⚠️ 局限性**

仅在相对小规模数据上训练，需进一步通过DFT验证生成材料的能量，对计算资源仍有一定依赖，模型在更大化学空间中的泛化尚未完全证明。

---

## 472. Agentifying Patient Dynamics within LLMs through Interacting with Clinical World Model

**arXiv ID:** 2605.14723 | [PDF](https://arxiv.org/pdf/2605.14723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 473. TAPIOCA: Why Task- Aware Pruning Improves OOD model Capability

**arXiv ID:** 2605.14738 | [PDF](https://arxiv.org/pdf/2605.14738v1)

**作者:** Krish Sharma `[一作]` (ANITI), Nicholas Asher `[通讯]` (ANITI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型进行推理时的层级剪枝（TALE），分析其为何仅在分布偏移（OOD）时提升性能，并提出几何对齐机制解释该现象。

**💡 创新点**

提出剪枝不是简单的正则化或冗余移除，而是通过减少特定层在OOD输入上放大内部表示几何失真，从而将OOD表征拉回与适配任务的ID几何相符的创新视角。

**🔧 技术方法**

使用层级残差线性代理、层级表示距离（norm、pairwise distance）统计、残差缩放干预、逆代理插入等技术，结合梯度无关的贪心剪枝算法TALE。

**📊 数据集**

在受控的多项式回归任务（自定义分布）、以及在 8B Llama 3.1、120B GPT-OSS 上分别针对数学推理（MATH500、MMLU）和代码推理（Code Alpaca、MMLU）等公开 NLP 评测集。

**📈 对比分析**

与基线全模型及同任务下的 ID 评测对比；剪枝后在 OOD 任务上平均提升 7–9 个百分点（如 MMLU Math 7.4p、BoolQ 3.0p），但在 ID 任务上无显著提升；通过层级代价可视化和逆代理实验验证了几何对齐的因果性。

**⚠️ 局限性**

主要局限在于：① 仅验证了 TALE 这类贪心剪枝方法，未探讨其他算法的泛化性；② 归因分析主要基于表示几何统计，缺乏严格理论证明；③ 在多样化、跨任务 OOD 场景下单一剪枝配置可能不足，需进一步自适应设计。

---

## 474. IsoNet: Spatially-aware audio-visual target speech extraction in complex acoustic environments

**arXiv ID:** 2605.14736 | [PDF](https://arxiv.org/pdf/2605.14736v1)

**作者:** Dinanath Pathya `[一作]` (Tribhuvan University), Ishwor Raj Pokharel `[通讯]` (Tribhuvan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了IsoNet，一种面向紧凑4麦克风阵列的用户选择目标语音提取系统。

**💡 创新点**

创新点在于将多通道复数频谱、GCC‑PHAT空间特征、面部视觉嵌入与DOA辅助监督共同注入U‑Net掩模估计，解决了小阵列下传统波束形成失效的问题。

**🔧 技术方法**

使用的技术包括U‑Net网络、多通道STFT、GCC‑PHAT延迟编码、冻结ResNet‑18视觉编码、辅助角度回归、以及课程学习（Curriculum Learning）。

**📊 数据集**

实验基于模拟的VoxCeleb‑Sim数据集，共25,000条4秒混合语音，包含随机房间、麦克风阵列、面部轨迹及房间参数。

**📈 对比分析**

与Oracle DAS/MVDR波束形成以及不同IsoNet变体进行对比，CL1在-1~10 dB SNR测试集上实现SI‑SDR 9.31 dB（提升4.85 dB）、PESQ 2.13、STOI 0.84；传统波束形成在相同条件下则会使SI‑SDR下降4.82 dB。

**⚠️ 局限性**

局限性包括相位重建质量不足、仅针对单干扰者场景、模拟到真实的转移挑战，以及视觉编码的计算成本高。

---

## 475. CHASM: Cross-frequency Harmonized Axis-Separable Mixing for Spectral Token Operators

**arXiv ID:** 2605.14727 | [PDF](https://arxiv.org/pdf/2605.14727v1)

**作者:** Pengcheng Fang `[一作]` (University of Southampton), Xiaohao Cai `[通讯]` (University of Southampton)

**通讯引用:** 63752 | [OpenAlex ID](https://openalex.org/A5045658136)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了 CHASM——一种跨频率协调的谱混合器，在视觉特征图中通过共享特征基向量和频率特定增益实现全局交互。

**💡 创新点**

创新点在于将通道方向共享与频率特定增益分离，形成“共享基+正增益”结构，在不牺牲频率适应性的前提下实现跨频率协同。

**🔧 技术方法**

利用一维实FFT、矩阵指数生成正交基、软正增长表格以及轴可分离的谱混合操作，并嵌入轻量级局部细化与1×1融合层。

**📊 数据集**

在多种任务中使用了CC359、fastMRI、BraTS、ADE20K、ImageNet等医学影像与自然图像数据集。

**📈 对比分析**

通过在固定骨干（HiFi‑Mamba、U‑Net、ViT）中替换相同位置的混合器进行对比，CHASM在MRI重建、分割和自然图像重建上相对基线提升约0.5–1.0 dB PSNR/0.01–0.02 SSIM，且在不同加速因子和骨干上保持稳健。

**⚠️ 局限性**

局限在于仅验证图像域重建与分割，未探究视频、噪声恢复等更广泛任务，也未系统评估不同采样模式和分辨率下的最优性。

---

## 476. Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning

**arXiv ID:** 2605.14717 | [PDF](https://arxiv.org/pdf/2605.14717v1)

**作者:** Saqib Nazir `[一作]` (Edge Hill University), Ardhendu Behera `[通讯]` (Edge Hill University)

**通讯引用:** 2089 | [OpenAlex ID](https://openalex.org/A5057980050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发一种融合CNN与ViT的多任务学习框架，利用无标记DPC图像同时完成白细胞分类与蛋白表达回归，并通过LLM生成可解释的生物学摘要。

**💡 创新点**

① 将局部卷积与全局Transformer通过可学习权重融合；② 引入任务自适应门控避免负迁移；③ 将LLM作为后处理生成结构化、临床可读的总结。

**🔧 技术方法**

使用深度学习多任务网络（Hybrid CNN–ViT），跨分支注意力融合、任务门控机制、焦点损失+Pearson一致性回归以及模板化LLM生成。

**📊 数据集**

BSCCM（含DPC图像、细胞标签和蛋白表达）和BCCD（RGB血细胞图像）。

**📈 对比分析**

与单任务CNN、ViT、ResNet等基线对比，在BSCCM上实现91.3%分类准确率、Pearson r=0.7263回归；在BCCD上宏平均F1≈0.933；整体相较基线提升约1–3%分类准确率与回归指标。

**⚠️ 局限性**

① 细胞类型边界模糊导致误差集中；② 对动态激活标记预测不足，无法替代分子检测；③ 依赖外部LLM，可能产生幻觉并需模板约束。

---

## 477. Towards Continuous Sign Language Conversation from Isolated Signs

**arXiv ID:** 2605.14705 | [PDF](https://arxiv.org/pdf/2605.14705v1)

**作者:** Youngmin Kim `[一作]` (Yonsei University), Seong Jae Hwang `[通讯]` (Yonsei University)

**通讯引用:** 617 | [OpenAlex ID](https://openalex.org/A5022991142)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了大规模离散手语视频词典 SignaVox-W 与连续对话手语数据集 SignaVox-U，并提出 BRAID 与 SignaVox 两套模型实现从离散手势到自然连续手语的无缝合成与实时会话生成。

**💡 创新点**

创新点包括：①利用多源手语词典与公开数据拼接构建多样化、标注完善的离散手势库；②提出 BRAID，基于条件扩散 Transformer 的边界补偿与时长对齐机制，实现高质量的共语化与流畅过渡；③设计 SignaVox，直接以 3D 手语运动为输入、输出的自回归会话生成器，省去口语/词汇翻译步骤；④整合 LLM RAG 翻译与动态时间规整评估，提升手语对话的语义对齐与可生成性。

**🔧 技术方法**

核心技术包括 3D SMPL-X、Dyn‑HaMR、EMOCA‑v2 的姿态估计；VideoLLM 与视觉提示的语义帧筛选；基于 Diffusion Transformer 的 BRAID 边界补偿；LLM‑RAG 口语到词汇的检索式翻译；以及基于流匹配与语义计划的 SignaVox 生成框架。

**📊 数据集**

使用的数据集有：四大网络手语词典、两个公开手语视频集（ASLLRP、ASL‑D），构成 SignaVox‑W；基于这些离散样本与 ASLLRP 对话数据合成的连续手语对话集 SignaVox‑U；以及公开的手语翻译基准数据用于评估口语到词汇的翻译性能。

**📈 对比分析**

与传统线性插值、SignD2C 等基线相比，BRAID 在 MPJPE/MPVPE、PA‑MPJPE/PA‑MPVPE 及长度比例上均表现最佳；SignaVox 在 DTW‑MPJPE、DTW‑MPVPE、FGD 以及 BLEU‑4 上显著优于 ReMoS，说明在运动质量、分布一致性和语义对齐上取得了显著提升。

**⚠️ 局限性**

局限性包括：①数据集仍受限于采集来源，可能缺乏跨文化和方言的覆盖；②模型对极长或高度复杂对话的处理仍有延迟与时长失配风险；③缺乏实时硬件部署评估与多模态交互场景的验证。

---

## 478. SceneFunRI: Reasoning the Invisible for Task-Driven Functional Object Localization

**arXiv ID:** 2605.14704 | [PDF](https://arxiv.org/pdf/2605.14704v1)

**作者:** Posheng Chen `[一作]` (National Taiwan University), Winston H. Hsu `[通讯]` (National Taiwan University)

**通讯引用:** 6368 | [OpenAlex ID](https://openalex.org/A5043898632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SceneFunRI基准，评估视觉语言模型在完全不可见目标对象定位上的推理能力

**💡 创新点**

首次将不可见目标推理纳入任务，构建了从3D空间到2D定位的半自动化数据集，并提出Spatial Process of Elimination（SPoE）迭代消除策略

**🔧 技术方法**

采用提示工程（Strong Instruction、Reasoning-based、SPoE）与链式推理（CoT、CoT‑SR）对现有VLM进行评测

**📊 数据集**

基于SceneFun3D数据集生成的855个2D推理实例

**📈 对比分析**

在多种VLM（Gemini 3 Flash、Qwen 3.5系列、InternVL3、Gemma 3等）上测试，最优模型Gemini 3 Flash获得CAcc@75≈15.2%，CAcc@50≈17.2%，mIoU≈0.74%，Dist≈28.65；与人类基准相比仍差距显著

**⚠️ 局限性**

VLM在处理不可见区域时表现不稳定，主要缺乏对空间上下文与常识的精确映射，模型倾向于排除可见区域而非直接推断隐藏位置，且在多步骤推理或提示下提升有限

---

## 479. Towards In-Depth Root Cause Localization for Microservices with Multi-Agent Recursion-of-Thought

**arXiv ID:** 2605.14866 | [PDF](https://arxiv.org/pdf/2605.14866v1)

**作者:** Lingzhe Zhang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**通讯引用:** 120825 | [OpenAlex ID](https://openalex.org/A5100391240)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于多智能体递归思维并行推理的根因定位框架RCLAgent，用于微服务系统故障诊断

**💡 创新点**

通过将诊断过程沿调用图拆分为每个span对应的专用代理，控制上下文爆炸并实现并行推理，显著提升定位精度与效率

**🔧 技术方法**

利用大型语言模型（如Claude‑3.5‑Sonnet、Qwen‑3.6‑Plus）与工具调用（日志、指标、追踪）相结合的多智能体递归思维（RCL）架构

**📊 数据集**

在三大公开基准上评测：AIOPS 2022、Augmented‑TrainTicket 与 RCAEval

**📈 对比分析**

与传统图/统计方法以及现有LLM方法（CoT、RCAgent、mABC、GALA等）对比，RCLAgent 在 Recall@k/MRR 上平均提升约7‑15%，并行推理使平均推理时间比最慢LLM方法快约1.5‑2.1倍

**⚠️ 局限性**

依赖完整且准确的分布式追踪；对缺失或不完整的span敏感；受LLM推理质量与成本限制；在无追踪环境下退化为传统单体推理

---

## 480. Interestingness as an Inductive Heuristic for Future Compression Progress

**arXiv ID:** 2605.14831 | [PDF](https://arxiv.org/pdf/2605.14831v1)

**作者:** Vincent Herrmann `[一作]` (Swiss AI Lab IDSIA/USI/SUPSI), Jürgen Schmidhuber `[通讯]` (Swiss AI Lab IDSIA/USI/SUPSI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文将“有趣性”定义为未来压缩进步的前瞻性启发式，并证明其可由过去的压缩进展推断。

**💡 创新点**

创新点在于用算法信息理论形式化有趣性的归纳属性，量化停滞长度对未来进步的影响，并比较长度、算法和速度先验的差异。

**🔧 技术方法**

主要技术包括Kolmogorov复杂度、算法统计学、复杂度-运行时间与对数尺寸-复杂度剖面、Busy Beaver函数、重要性采样与仿真。

**📊 数据集**

实验数据集为三种通用计算模型：2‑Tag系统（约1.2亿程序）、Rule 110（约3.4千万程序）和Brainfuck（约23亿程序），每种程序在上限步数内运行。

**📈 对比分析**

通过将理论预测与实际压缩进展曲线对比，发现算法先验更乐观，长度先验适中，速度先验最保守，三者均与停滞长度呈指数衰减关系，实验结果与理论一致。

**⚠️ 局限性**

局限性包括对不可计算先验（Kolmogorov、Busy Beaver）的依赖、对实际可执行运行时间的忽略、仅关注压缩进展而忽视内容特征，以及对理想化先验分布的假设。

---

## 481. FactorizedHMR: A Hybrid Framework for Video Human Mesh Recovery

**arXiv ID:** 2605.14854 | [PDF](https://arxiv.org/pdf/2605.14854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 482. ToMAToMP: Robust and Multi-Parameter Topological Clustering

**arXiv ID:** 2605.14824 | [PDF](https://arxiv.org/pdf/2605.14824v1)

**作者:** Ludo Andrianirina `[一作]` (DataShape), Mathieu Carrière `[通讯]` (DataShape)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种能够同时处理多函数的拓扑聚类算法；

**💡 创新点**

首次利用多参数持久同调中的MMA分解实现多函数聚类，并提供鲁棒性理论保证；

**🔧 技术方法**

采用多参数持久同调、MMA分解、持久性图匹配以及图论技术；

**📊 数据集**

在四类数据上实验：合成二维数据、Princeton 3D形状、细胞形态图像、两组空间转录组数据；

**📈 对比分析**

与传统非拓扑方法（k-means、谱聚类、层次聚类）以及单参数拓扑聚类进行对比，取得更高的ARI/AMI和更好的基因共定位排名，表现出更好的鲁棒性和无图参数调优；

**⚠️ 局限性**

算法时间复杂度高，尤其在图的细分与多参数持久同调计算上；理论假设（A1）对分离的要求较强，且在函数数量增多时更难满足。

---

## 483. GFMate: Empowering Graph Foundation Models with Test-time Prompt Tuning

**arXiv ID:** 2605.14809 | [PDF](https://arxiv.org/pdf/2605.14809v1)

**作者:** Yan Jiang `[一作]` (University of Queensland), Zi Huang `[通讯]` (University of Queensland)

**通讯引用:** 13722 | [OpenAlex ID](https://openalex.org/A5078170935)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为 GFMate 的测试时图提示调优框架，可在跨域场景下显著提升图基础模型（GFMs）的性能。

**💡 创新点**

创新点包括：①预训练无关的中心点与层级提示，避免了提示与源域或预训练策略的耦合；②测试时互补学习目标，主动利用目标域未标记数据进行提示调优；③实现了更小的可调参数量和更低的显存占用，提升了效率。

**🔧 技术方法**

技术方法主要是：使用 GNN 背景模型（如 GCN、SAGE、GAT、H2GCN）结合自监督预训练（链接预测、DGI、GCL），通过余弦相似度构建中心点聚类，利用层级加权和互补标签策略实现提示的自适应更新。

**📊 数据集**

实验使用 12 个基准图数据集，涵盖社交网络、引用网络、商业系统和生物网络，支持节点分类和图分类两类任务。

**📈 对比分析**

与 21 条基线（单域监督、单域自监督+微调、单域提示、跨域 GFM 提示等）进行对比，GFMate 在所有 12 个任务中均优于现有方法，提升幅度最高可达 30.63%，且在显存与训练时间上实现了显著的效率提升。

**⚠️ 局限性**

局限性包括：①仍需要预训练好的图基础模型，无法直接应用于无预训练模型；②主要针对无文本属性的 GNN 基础模型，对 LLM 基础模型的适用性尚未验证；③对超参数（如 γ、τ）的敏感性及在极大规模图上的可扩展性尚待进一步研究。

---

## 484. Do Coding Agents Understand Least-Privilege Authorization?

**arXiv ID:** 2605.14859 | [PDF](https://arxiv.org/pdf/2605.14859v1)

**作者:** Zheng Yan `[一作]`, Mengkang Hu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了编码代理的权限边界推断任务，创建了AuthBench基准测试，评估并改进现有模型的权限生成能力。

**💡 创新点**

创新点在于将权限推断定义为独立能力、构建包含人类审核权限标签和执行验证器的AuthBench数据集，并提出分阶段的Sufficiency‑Tightness分解方法以克服模型特定的授权吸引子。

**🔧 技术方法**

采用大型语言模型进行权限生成，使用前向模拟与两阶段推断，结合GPT‑5执行代理进行动态验证，并通过精度、召回、F1、任务成功率、敏感文件暴露率和攻击成功率等指标评估。

**📊 数据集**

使用从Terminal‑Bench、SWE‑Bench和OpenThoughts‑TBLite收集并整理的120个终端任务（80普通+40敏感），配备人类审核的权限规范与执行器验证器。

**📈 对比分析**

通过与Full‑Access和Golden‑Permission基线对比，并在标准与敏感任务上使用精度/召回/F1、TSR、SER和ASR等指标，发现分解方法在紧凑模型上提升TSR高达15.8%，并在所有模型上显著降低攻击成功率。

**⚠️ 局限性**

局限性包括直接策略生成仍受模型特定授权吸引子限制、推断精度受限于代理执行器、仅覆盖文件级权限、以及对不同执行环境的通用性验证不足。

---

## 485. Discrimination Is Generation: Unifying Ranking and Retrieval from a Tokenizer Perspective

**arXiv ID:** 2605.14853 | [PDF](https://arxiv.org/pdf/2605.14853v1)

**作者:** Shuli Wang `[一作]` (Meituan), Xingxing Wang `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了DIG框架，在单一模型中联合训练分词器和排序模型，实现生成式检索与判别式排序的无缝统一。

**💡 创新点**

通过将分词器嵌入判别式排序器并以排序损失直接驱动代码簇学习，突破传统检索器与生成模型分离的瓶颈，并设计了特征分配分类法、MLP_u2t知识蒸馏与层级监督实现精准个性化。

**🔧 技术方法**

采用残差向量量化（RQ）分词器、平衡K-means初始化、端到端的BCE+层级检索损失、MLP_u2t蒸馏、共享Mixer与MoE结构。

**📊 数据集**

在三公开数据集（KuaiRec-Small、KuaiRec-Big、Taobao）与两工业数据集（Meituan-Large、Meituan-Small）上进行评测。

**📈 对比分析**

与TIGER、LETTER、DAS、DOS、ETEGRec等五大生成式检索基线在Recall@10/NDCG@10上比较，平均提升超过100%，在工业数据上提升最高达+220%；同时保持甚至提升排名AUC，统一检索-排序AUC差距保持在-0.06以内。

**⚠️ 局限性**

对高维用户-项目交互特征的推断精度依赖MLP_u2t蒸馏，若u2i特征稀疏或维度过大，检索召回可能受限；且需充分的u2i训练样本才能发挥分词器个性化优势。

---

## 486. IFPV: An Integrated Multi-Agent Framework for Generative Operational Planning and High-Fidelity Plan Verification

**arXiv ID:** 2605.14851 | [PDF](https://arxiv.org/pdf/2605.14851v1)

**作者:** Zhigao Huang `[一作]` (Zhengzhou University), Mingliang Xu `[通讯]` (Zhengzhou University)

**通讯引用:** 9491 | [OpenAlex ID](https://openalex.org/A5081346568)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了IFPV框架，集成多视角层级代理生成可执行作战计划（MPHA）与自定义世界模型对抗验证（ACSE），实现了生成–验证闭环。

**💡 创新点**

创新点在于：①将指挥意图分解为Pathfinder、Analyst、Planner和Validator四层代理，显著提升规划可执行性；②在ACSE中引入EVA‑Loss训练的定制世界模型，使验证器能够根据实体价值动态预测并对抗计划；③将生成与验证通过统一接口闭环，提供量化反馈。

**🔧 技术方法**

技术手段包括：大语言模型（Qwen、Llama、Gemini等）+LoRA微调、EVA‑Loss加权损失、Actor‑Critic/强化学习框架、层级多代理架构、两维不对称空地战仿真平台ACTS，以及基于轨迹预测的自定义世界模型。

**📊 数据集**

数据来源主要是ACTS仿真生成的轨迹与战术序列，用于世界模型训练和验证；未使用公开公开数据集，全部采用仿真收集的数据。

**📈 对比分析**

通过与单步LLM规划基线、Gemini 3.1 Pro、DeepSeek‑V3、GLM‑5等生成器，以及No_Brain、GLM‑5等验证器对比；实验表明MPHA使任务成功率提升19.4%，平均运营成本降低41.7%；ACSE将抑制率提升31.8%，世界模型ADE降至0.18，验证压力显著加强。

**⚠️ 局限性**

局限性：①仅在二维空地战场景中验证，缺乏三维、多域和复杂地形；②对抗模型仅做轨迹预测，未加入意图识别、欺骗检测或多目标优先级建模；③缺少人机交互式迭代优化，未实现真实兵力与资源约束下的完整决策支持。

---

## 487. Agentic AI and Human-in-the-Loop Interventions: Field Experimental Evidence from Alibaba's Customer Service Operations

**arXiv ID:** 2605.14830 | [PDF](https://arxiv.org/pdf/2605.14830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 488. The Complexity of Nested Reset Counter Systems

**arXiv ID:** 2605.14850 | [PDF](https://arxiv.org/pdf/2605.14850v1)

**作者:** A. R. Balasubramanian `[一作]` (Max Planck Institute for Software Systems), Franzisco Schmidt `[通讯]` (Technical University of Munich)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出并分析了嵌套重置计数器系统（NRCS），并证明其覆盖性问题在任意阶k下是Ω_k完全的；

**💡 创新点**

创新点在于构造了k阶NRCS的覆盖性问题为Ω_k阶完全问题，填补了Ω_k阶（k>3）缺乏自然完全问题的空白，并推出了一套用于计算嵌套多重集的长度函数定理；

**🔧 技术方法**

主要技术包括：1) 通过构造k-最小子树、k-复制、k-比较等gadget实现Hardy函数的弱计算；2) 将树模型转化为嵌套多重集的nwqo；3) 采用WSTS框架与反射与残差方法证明长度函数上界；4) 将所得上界映射到快速增长层次的F_Ω_k；

**📊 数据集**

无实验数据集，研究完全基于理论证明与构造；

**📈 对比分析**

与已有的上界（如Ω_{2k+1}等）相比，作者的上界为Ω_k（更低阶），同时通过NRCS的完整性证明对应问题的下界，形成了完整的匹配；

**⚠️ 局限性**

局限性在于仅给出理论复杂度上界与下界，缺乏可实现的算法实现与实验验证；此外仅适用于有限k的NRCS模型，对更通用或无限阶情形尚未覆盖。

---

## 489. MechVerse: Evaluating Physical Motion Consistency in Video Generation Models

**arXiv ID:** 2605.14843 | [PDF](https://arxiv.org/pdf/2605.14843v1)

**作者:** Rahul Jain `[一作]` (Purdue University), Karthik Ramani `[通讯]` (Purdue University)

**通讯引用:** 12054 | [OpenAlex ID](https://openalex.org/A5004602626)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并构建了MechVerse基准，用于评估图像到视频生成模型在机械装置中保持运动一致性的能力。

**💡 创新点**

创新点在于：①将机械装配划分为三层复杂度（Easy/Medium/Hard）并提供结构化运动提示；②通过Unity实现精确的关节动画和多视角渲染；③使用视觉、指令跟随和人工三维评测体系，揭示现有模型在机械一致性上的显著缺陷。

**🔧 技术方法**

采用Unity渲染流水线、结构化文本提示、VBench-I2V和WorldModelBench指标以及人工评测；对14+公开/闭源视频生成模型（包括DynamiCrafter、CogVideoX、HunyuanVideo、VideoCrafter等）进行评估。

**📊 数据集**

使用自制的21,156段视频数据集MechVerse，来源于PartNet-Mobility与CAD机械装配，涵盖141种装置，配有结构化提示和对应3D资产。

**📈 对比分析**

通过视觉一致性、指令跟随分数和人工打分对模型进行横向比较；结果显示所有模型在视觉质量上表现良好，但在机械一致性上普遍不足，错误随耦合复杂度提升；闭源模型在指令跟随上更优，开源模型在视觉指标上竞争力较强；微调后部分指标有所提升，但仍无法完全解决机械失效。

**⚠️ 局限性**

局限性在于：①仅关注机械装配，未覆盖柔性物体、流体或自然场景交互；②评测依赖现有视觉与LLM判断，可能无法捕捉细粒度机械错误；③未提供对力学、接触、动力学等更深层次的物理约束分析。

---

## 490. Multi-proposal Collaboration and Multi-task Training for Weakly-supervised Video Moment Retrieval

**arXiv ID:** 2605.14838 | [PDF](https://arxiv.org/pdf/2605.14838v1)

**作者:** Bolin Zhang `[一作]` (Hunan University), Ichiro Ide `[通讯]` (Nagoya University)

**通讯引用:** 2253 | [OpenAlex ID](https://openalex.org/A5034941095)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种弱监督视频时刻检索方法MCMT，利用多高斯掩码协同生成正样本掩码并挖掘易负、硬负样本，同时采用前向与反向查询重构的双任务训练；

**💡 创新点**

1) 多个可学习高斯掩码协同聚合得到更鲁棒的正样本掩码；2) 引入双向（前向/反向）重构任务，提供更强的多视角约束；3) 结合正负样本挖掘与内部对比损失提升模型判别力；

**🔧 技术方法**

Transformer编码器‑解码器、向量拼接/加权注意力融合、可学习高斯掩码、掩码聚合模块、双向重构任务、内部对比损失IVC、CLIP/I3D视觉特征、GloVe词向量；

**📊 数据集**

Charades‑STA 与 ActivityNet‑Captions 两个公开数据集；

**📈 对比分析**

与多实例学习、重构式及其他弱监督方法对比，MCMT 在 ActivityNet 上实现 Rank@1 与 mIoU 最高（提升约3–4%），在 Charades 上位居前三，显著优于同类重构式方法；

**⚠️ 局限性**

对背景相似的室内活动（如 Charades）精细检索仍存在挑战；多提议机制在不同数据集表现不一，过多提议可能导致性能下降；模型更适合长多样化视频，对极短/相似背景场景的局限仍待改进。

---

## 491. Probing into Camera Control of Video Models

**arXiv ID:** 2605.14815 | [PDF](https://arxiv.org/pdf/2605.14815v1)

**作者:** Chen Hou `[一作]` (University of Oxford), Christian Rupprecht `[通讯]` (University of Oxford)

**通讯引用:** 4283 | [OpenAlex ID](https://openalex.org/A5083153177)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在视频扩散模型的去噪过程中使用位移场实现无训练的相机控制，并将其用作探测模型相机控制能力的工具。

**💡 创新点**

将相机控制视为几何引导的位移场而非隐式学习，提出可直接在无监督条件下实现控制的方式，并用作探测工具。

**🔧 技术方法**

利用深度估计产生位移场，在扩散模型去噪步骤中进行可微分重采样，结合基于噪声预测的更新策略。

**📊 数据集**

主要使用公开的无标注视频以及通过文本提示的 VBench2.0、Objaverse‑XL 等公开数据集进行评估；无额外训练数据。

**📈 对比分析**

与现有微调驱动的方法相比，CamProbe 在视觉质量与相机控制精度上保持竞争力，且在多视角一致性上表现优异，显著提升了模型的可控性与评估可靠性。

**⚠️ 局限性**

位移场为手工设计，缺乏精细学习，导致在复杂空间关系或强动态场景中控制效果下降，且对深度估计误差敏感。

---

## 492. Fast Adversarial Attacks with Gradient Prediction

**arXiv ID:** 2605.14868 | [PDF](https://arxiv.org/pdf/2605.14868v1)

**作者:** Kamil Ciosek `[一作]` (Spotify), Konstantina Palla `[通讯]` (Spotify)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种通过前向隐藏状态预测输入梯度的轻量级FGSM攻击，省去每个示例的反向传播；

**💡 创新点**

其创新点在于利用NTK/GP理论证明隐藏状态与梯度线性相关，并用线性回归实现梯度预测，从而显著提升攻击吞吐量；

**🔧 技术方法**

技术包括线性梯度预测、NTK/GP理论支持、岭回归训练、早停隐藏层提取，以及对embedding级和token级攻击的统一实现；

**📊 数据集**

实验数据集为 HarmBench 与 JailbreakBench，分别对 Llama‑2、Qwen‑2.5 与 Qwen‑3 三大LLM 进行评估；

**📈 对比分析**

与精确梯度FGSM、PGD、RS‑FGSM 等方法比较，成功攻击吞吐量提升 5‑12 倍，攻击成功率（ASR）基本保持相同；

**⚠️ 局限性**

局限性包括仅在有限模型/任务上验证，梯度预测误差在多步攻击中会累积，且需为不同模型/目标重新训练预测器，嵌入扰动主要用于评估而非直接可用的 token 攻击。

---

## 493. REALM: Retrospective Encoder Alignment for LFP Modeling

**arXiv ID:** 2605.14867 | [PDF](https://arxiv.org/pdf/2605.14867v1)

**作者:** Peicheng Wu `[一作]` (Ohio State University), Lin Du `[通讯]` (Ohio State University)

**通讯引用:** 1575 | [OpenAlex ID](https://openalex.org/A5100695978)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种基于回溯蒸馏的LFP神经解码框架REALM，将多会话双向教师模型压缩为可实时部署的因果学生模型，实现高精度运动解码。

**💡 创新点**

创新点在于首次将离线双向自监督预训练与因果蒸馏相结合，获得只依赖LFP的高性能因果解码器，并实现了从LFP单模态到跨会话的基础模型。

**🔧 技术方法**

使用了Mamba‑2网络结构、连续块掩码自编码（CMAE）预训练、表示对齐与速度监督相结合的回溯蒸馏、以及多折集成与量化等技术。

**📊 数据集**

利用130小时多会话LFP数据，包括Makin、Flint、Brochier、Churchland、Even‑Chen等公开非人类灵长类电极记录数据集。

**📈 对比分析**

与传统LSTM、Transformer、线性Kalman等基线比较，因果REALM在Makin和Flint测试集上R²达到0.71（比LSTM+5M高0.06），双向REALM达到0.775，甚至可与跨模态Spike蒸馏相媲美；在低功耗边缘设备上可实现100Hz实时推理。

**⚠️ 局限性**

主要局限包括缺乏在人类或更复杂任务中的验证、长期稳定性和电极退化未充分评估、因果与双向性能差距、数据规模相对有限、未测量功耗等。

---

## 494. In-Context Learning for Data-Driven Censored Inventory Control

**arXiv ID:** 2605.14840 | [PDF](https://arxiv.org/pdf/2605.14840v1)

**作者:** Sohom Mukherjee `[一作]` (Julius-Maximilians-Universität Würzburg), Yunbei Xu `[通讯]` (National University of Singapore)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5007283391)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在决策相关右截断的重复报刊商问题上提出了“上下文生成后验采样（ICGPS）”方法，并实现了可在线执行的 ChronosFlow 体系结构，能够在不更新参数的情况下对缺失需求进行完成和后验采样。

**💡 创新点**

创新点包括：① 将缺失数据视为后验采样的生成目标，在连续动作空间下实现无模型参数更新的在线探索；② 通过尾部条件采样严格满足右截断约束；③ 在 R‑NV 上建立 TS 与 ICGPS 的贝叶斯回报分解，并证明理想 TS 在该问题下可获得 O(√T) 的贝叶斯回报；④ 将离线预测误差与在线部署惩罚联系起来，给出可观测的优化目标。

**🔧 技术方法**

使用的技术主要有：Transformer‑基准时间序列生成器、条件归一化流头、尾部条件采样、离线元学习+在线上下文学习、信息比率与 BCO 理论框架。

**📊 数据集**

实验使用合成数据（Weibull、指数、对数正态等多种分布）以及真实的 SuperStore（技术、办公用品、家具等）数据集，后者包含库存与销售记录，能够产生右截断观测。

**📈 对比分析**

与参数化 Thompson Sampling、无截断 MLE、UCB 等基线进行对比；在正确规范的情形下 ChronosFlow‑ICGPS 与 TS 等价；在分布偏移、先验不匹配、以及高截断率下显示更稳健的表现；在 SuperStore 真实数据上，尤其在重度截断时获得最低的平均损失。

**⚠️ 局限性**

局限性包括：① 对极低订单覆盖率的场景下识别性受限，需要满足最大覆盖和自收缩假设；② 需要大量离线元训练样本和模型容量，模型选择会影响性能；③ 在强非 i.i.d. 或非平稳需求时，理论保证不再完全适用；④ 仅针对单一标量动作空间，尚未扩展到多维或有特征的场景。

---

## 495. A Deterministic Agentic Workflow for HS Tariff Classification: Multi-Dimensional Rule Reasoning with Interpretable Decisions

**arXiv ID:** 2605.14857 | [PDF](https://arxiv.org/pdf/2605.14857v1)

**作者:** Yu Zhang `[一作]` (Shanghai Jiao Tong University), Kai Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 32825 | [OpenAlex ID](https://openalex.org/A5100437924)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了一种确定性 agentic 工作流，对 HS 关税分类进行多维规则推理，结合离线知识工程与在线六阶段管线，输出带原文引用的可审计决策。

**💡 创新点**

创新点在于将工作流控制流程固定、每个阶段仅局部调用 LLM 并产生结构化、可审计的规则引用，从而解决多维优先级冲突并提升可解释性与稳定性。

**🔧 技术方法**

技术方案包括 BM25 与密集嵌入检索、逆序 RRF 融合、GIR 规则编码、层级索引构建，以及在六阶段管线中使用 Qwen3.6-plus 与 Qwen3.6-27B-FP8 等 LLM 进行排名与确认。

**📊 数据集**

使用了中国 HS 版的结构化文本（章节、章、标题、解释说明）构建离线索引；评估数据集为 HSCodeComp 632 条目（四、六位码），并进行手工审核。

**📈 对比分析**

采用 top‑1 / top‑3 准确率对比，Qwen3.6-plus 在四位 75.0% / 91.5%，六位 64.2% / 78.3%；与 27B-FP8 的预测一致率 84.2% / 77.4%；人工审核后六位正确率约 85.8% 以上。

**⚠️ 局限性**

局限性包括规则编码不完整、跨章排除图处理不系统、部分错误由上下文不完整或描述模糊导致，且模型仍需较高算力且对新规则更新需要手动重建索引。

---

## 496. Construction of Minimal Ternary Linear Codes with Dimension $n+2$

**arXiv ID:** 2605.14848 | [PDF](https://arxiv.org/pdf/2605.14848v1)

**作者:** Haibo Liu `[一作]` (Chengdu University of Information Technology), Qunying Liao `[通讯]` (Sichuan Normal University)

**通讯引用:** 408 | [OpenAlex ID](https://openalex.org/A5073944562)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一类维数为 m+2 的三元线性最小码，并给出了其完整重量分布；

**💡 创新点**

提出了一个通用构造，利用 Walsh 变换给出了最小码的必要与充分条件，并首次在维数提升到 m+2 的同时实现了对 Ashikhmin‑Barg 条件的违背；

**🔧 技术方法**

主要运用了指数和、Krawtchouk/洛伊德多项式、Walsh 变换等代数工具来分析码的权分布与最小性；

**📊 数据集**

所使用的数据集为理论上产生的符号集合 𝔽₃ⁿ，没有实际实验数据；

**📈 对比分析**

通过对比重量分布和最小/最大权比，证明该码满足 w_min/w_max ≤ 2/3，且满足严格的最小性条件，显示其在理论性能上的优越性；

**⚠️ 局限性**

局限性在于仅针对 𝔽₃ 的三元码，且需要严格满足特定权约束（k₁<k₂≤⌊(m-1)/2⌋），未来可探讨扩展到更一般的奇素数域和更灵活的函数构造。

---

## 497. SR-Prominence: A Crowdsourced Protocol and Dataset Suite for Perceptually-Weighted Super-Resolution Artifact Evaluation

**arXiv ID:** 2605.14847 | [PDF](https://arxiv.org/pdf/2605.14847v1)

**作者:** Ivan Molodetskikh `[一作]` (Lomonosov Moscow State University), Dmitriy Vatolin `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 679 | [OpenAlex ID](https://openalex.org/A5020940244)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SR-Prominence 数据集并引入了视觉显著性评估（artifact prominence）来量化超分辨率图像中的可见瑕疵。

**💡 创新点**

提出了以观众感知频率为指标的 artifact prominence 概念，并通过大规模众包标注、自动掩码生成与多维度评估协议，弥补了以往二值缺陷标注的局限。

**🔧 技术方法**

使用众包问卷、形态学掩码预处理、全参考指标（SSIM、DISTS）、无参考 IQA、现有检测器（LDL、DeSRA）、以及基准推理和 pseudo‑GT 的轻量 SR。

**📊 数据集**

采集并注释了 3,935 个掩码，来自 DeSRA、Open Images、Urban100 以及 Urban100-HR 四个子集，并结合 15 种主流 SR 方法生成。

**📈 对比分析**

通过阈值化候选掩码和无阈值 Spearman 相关性对检测器与指标进行评估，发现全参考指标 SSIM 与 DISTS 在预测可见瑕疵方面最优；训练的检测基线与现有方法相比在阈值无关评估中排名第一。

**⚠️ 局限性**

标注缺乏精确边界、依赖现有掩码生成导致误差、伪 GT 近似可能误判细节、且仅覆盖单帧图像，未处理视频时序或语义级瑕疵。

---

## 498. Exploring Vision-Language Models for Online Signature Verification: A Zero-Shot Capability Study

**arXiv ID:** 2605.14845 | [PDF](https://arxiv.org/pdf/2605.14845v1)

**作者:** Marta Robledo-Moreno `[一作]` (Universidad Autonoma de Madrid), Javier Ortega-Garcia `[通讯]` (Universidad Autonoma de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探索了大规模视觉‑语言模型（VLM）在在线签名验证中的零样本性能，并提出了一种将动态时序数据转换为视觉图像、通过 token log‑prob 生成生物识别分数并使用 Chain‑of‑Thought 进行可解释推理的完整框架。

**💡 创新点**

创新点包括：①将 kinematic time‑series 编码为压力信息的静态图像，令 VLM 能直接对视觉表现进行分析；②利用 VLM 内部 log‑likelihood 提取“Same/Different”token 的概率作为连续生物识别分数，提升了传统基于文本置信度的准确性；③将推理过程拆分为初始判断与 CoT 反思，既提供判决，又提供可验证的解释。

**🔧 技术方法**

主要技术：GPT‑5.2 与 Gemini 2.5 Pro 两大 VLM；压力编码视觉预处理；token log‑prob 提取实现 S_v1/S_v2 分数；CoT 结构化提示生成可解释文本；使用 EER、DET 曲线对比评估。

**📊 数据集**

数据集：Signature Verification Challenge (SVC) 评估数据，包含三类任务（Office‑Stylus、Mobile‑Finger、Combined），分别提供不同设备与压力信息的签名对比。

**📈 对比分析**

比较方法：将 VLM 的零样本 EER 与 SVC 领导者（如 DLVC‑Lab、BiDA‑Lab）以及基线 DTW 进行对比。结果显示：在随机伪造下，GPT‑5.2 的 S_v2 在 Task 2 达到 0.32% 的 EER，优于所有监督模型；在移动任务中，零样本 GPT 超越标准基线并击败部分监督方法；但在熟练伪造场景，尤其是高质量 Stylus 任务，S_v2 甚至退化至 ~48% EER，表现出“Rationalization Trap”。

**⚠️ 局限性**

局限性：①视觉编码对动态细节的捕捉有限，导致对熟练伪造的敏感度不足；②CoT 可能产生幻觉，将非真实的运动缺陷解释为自然变异，降低可靠性；③依赖商业 VLM API，受限于隐私与安全策略，无法直接处理原始时序数据；④缺乏针对不同设备、压力水平的细粒度适配策略。

---

## 499. XFP: Quality-Targeted Adaptive Codebook Quantization with Sparse Outlier Separation for LLM Inference

**arXiv ID:** 2605.14844 | [PDF](https://arxiv.org/pdf/2605.14844v1)

**作者:** Thomas Witt `[一作]` `[通讯]` (Gemini Stiftung Leipzig), Thomas Witt (Gemini Stiftung Leipzig)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 XFP，一种动态质量目标量化器，能够自动确定每层码本大小、异常值阈值和有效位宽，并支持大规模模型在工作站级 GPU 上的推理。

**💡 创新点**

创新点包括：① 逆向量化流程，以每通道余弦相似度阈值为质量目标；② 学习自适应码本并分离高幅异常值；③ 提供两种码本存储模式共享同一自动选择前端；④ H-Process 在 397B 模型上实现 2×96 GB 单机部署。

**🔧 技术方法**

采用 Lloyd 迭代学习码本、阈值自动选择、子字节打包、融合解码核、稀疏异常值恢复等技术，并通过 MultiQuant 与 vLLM 集成。

**📊 数据集**

使用 Qwen3.5 系列（35 B、122 B、397 B）模型，并在 GSM8K、MMLU 等标准数据集上进行量化与推理评测。

**📈 对比分析**

与 Marlin INT4（AutoRound）在单流推理下对比，XFP 在 122 B 模型上吞吐量提升 49%–87%，GSM8K 匹配率保持 94%–95%；在 397 B 模型上实现 100.9 tokens/s 长输出，66.7% GSM8K 匹配。

**⚠️ 局限性**

局限性包括：仅支持单用户低延迟推理；V2 模式受工作站 SMEM 限制导致 K≤8192；缺乏 Hessian 反馈，Lloyd 迭代非确定；未提供高吞吐量批处理实现；依赖社区补丁以支持 NVFP4。

---

## 500. GPart: End-to-End Isometric Fine-Tuning via Global Parameter Partitioning

**arXiv ID:** 2605.14841 | [PDF](https://arxiv.org/pdf/2605.14841v1)

**作者:** Paolo Mandica `[一作]` (Samsung AI Center), Neo Christopher Chung `[通讯]` (Samsung AI Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了 GPart，一种将低维可训练向量直接映射到完整权重空间的参数高效微调方法；

**💡 创新点**

创新点在于消除 LoRA 的低秩双线性结构，保持端到端等距映射，避免几何扭曲；

**🔧 技术方法**

使用随机稀疏分区矩阵、等距投影和梯度累加等技术实现低维投影；

**📊 数据集**

实验使用 GLUE（NLP）、MetaMath QA（数学推理）以及 ViT 在 VisionBenchmarks（图像分类）等数据集；

**📈 对比分析**

通过与全微调、LoRA、BitFit、VeRA、FourierFT、Uni-LoRA 等方法在相同可训练参数预算下比较，GPart 在 Encoder-Only 任务中显著优于或与现有方法持平，Decoder-Only 与 Vision 任务也保持竞争力；

**⚠️ 局限性**

局限性包括在 Decoder-Only 大规模或多模态模型上的通用性尚未验证，实验规模相对有限。

---

## 501. HDRFace: Rethinking Face Restoration with High-Dimensional Representation

**arXiv ID:** 2605.14821 | [PDF](https://arxiv.org/pdf/2605.14821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 502. The Racial Character of Computer Graphics Research

**arXiv ID:** 2605.14835 | [PDF](https://arxiv.org/pdf/2605.14835v1)

**作者:** Theodore Kim `[一作]` (Yale University), Alka V. Menon `[通讯]` (Yale University)

**通讯引用:** 447 | [OpenAlex ID](https://openalex.org/A5001364517)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对SIGGRAPH、SIGGRAPH Asia和ACM Transactions on Graphics等顶级计算机图形会议和期刊的论文进行系统性审计，梳理并量化其中关于人类皮肤和头发渲染与模拟的实例与描述，揭示算法在肤色与发型上存在的二元化、层级化及代币化等种族偏见现象；

**💡 创新点**

首次在计算机图形领域开展大规模的文献审计，并提出“McDaniels Methods”和“Durald Methods”两种概念，用以描述以白色皮肤/直发为基准、忽视其他肤色与发型的算法做法与与共同创作、包容性方法的对照；

**🔧 技术方法**

通过检索关键词“skin”“hair”，手工筛选并对论文中展示的图像进行二元分类（白/非白、直/非直），配合主题分析与可视化技术呈现时间序列与分布统计；

**📊 数据集**

主要使用了约6864篇SIGGRAPH/TOG论文全文作为文本与图像来源，未引入公开数据集，仅对论文内的示例图像进行人工编码；

**📈 对比分析**

采用统计计数与可视化对比的方法，衡量不同肤色与发型在论文中出现的比例及随时间的变化，发现白色皮肤/直发占据主导位置，非白与非直发仅为后续加成或代币化样本；未对算法的技术性能做直接评估，而是聚焦示例代表性与归属偏见；

**⚠️ 局限性**

局限性包括：仅聚焦SIGGRAPH/TOG，可能忽视其他会议和期刊；手工分类带来主观性，仅做粗二元划分；未细化肤色深浅与发色、光照、纹理等多维细节；未评估算法实现细节与实际效果的技术性能。

---

## 503. Min-1-Planarity is NP-Hard

**arXiv ID:** 2605.14834 | [PDF](https://arxiv.org/pdf/2605.14834v1)

**作者:** Yuto Okada `[一作]` (Nagoya University), Yuto Okada `[通讯]` (Nagoya University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5062810881)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

证明了判断给定图是否存在 min-1-平面绘图的判定问题是 NP‑hard。

**💡 创新点**

创新点在于引入“不可交叉边”(uncrossable edge) 的 gadget，以处理 min-1-平面绘图中允许出现无穷多交叉的边，并在 Grigoriev‑Bodlaender 的 1‑planarity 归约框架上完成了新的 NP‑hard 证明。

**🔧 技术方法**

采用了图构造与可交叉边 gadget、Jordan 曲线定理、简单绘图的性质以及 3‑Partition 问题的多项式归约等理论技术。

**📊 数据集**

本论文为纯理论证明，未使用任何实验数据集。

**📈 对比分析**

论文没有与实验方法进行比较，性能评估不适用；主要通过理论证明展示 NP‑hard 性。

**⚠️ 局限性**

局限性包括仅证明了 k=1 的情况；对 k≥2 的 min‑k‑planarity 复杂性仍未解决，并且方法依赖于所有 min‑1‑平面图都能转化为简单绘图，无法直接推广到非简单的 min‑k‑平面图。

---

## 504. A Heterogeneous Temporal Memory Governance Framework for Long-Term LLM Persona Consistency

**arXiv ID:** 2605.14802 | [PDF](https://arxiv.org/pdf/2605.14802v1)

**作者:** Zhao Yang `[一作]` (Changchun Kelaile Technology Co., Ltd), Lin Hujite `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ARPM框架，利用外部时间记忆治理拆分静态知识与动态经验，构建多阶段检索-评估-生成闭环，显著提升长久对话中的事实、时间、身份与边界一致性。

**💡 创新点**

创新点包括：①异构记忆解耦与双时空重排序，②analysis协议实现检索后证据验证与答案绑定，③白盒日志实时原子写回与可追溯审计，④跨模型传递性验证，证明一致性不完全依赖单模型参数。

**🔧 技术方法**

技术手段：向量检索+BM25+RRF融合；双时空衰减权重（绝对时间与对话轮次）；analysis协议预生成验证；实时JSON原子日志；多路检索（知识+经验）并行融合。

**📊 数据集**

数据集：50轮结构化问答；噪声知识库5.1M字符；跨模型实验覆盖DeepSeek、GPT‑5.5、Claude、Gemini、GLM、Kimi、Qwen2.5‑7B、Qwen3‑8B、LongCat‑Flash、MiniMAX‑M2‑Her等。

**📈 对比分析**

比较方式：与自动CSV评估对比，手工复核提升Recall（1:5条件从54%→100%，1:200+从44%→80%）；消融实验显示完整系统严格准确率100%，禁用历史检索或BM25导致下降；跨模型实验多模型保持高一致性，Qwen2.5‑7B表现差，说明方案可跨模型迁移。

**⚠️ 局限性**

局限性：高噪声实验仅针对结构化问答，未覆盖开放式情感/任务场景；人工复核成本高，缺乏自动化审计工具；跨模型评估仍混合人工与模型评分，未实现严格双盲多评审。

---

## 505. Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental Learning

**arXiv ID:** 2605.14785 | [PDF](https://arxiv.org/pdf/2605.14785v1)

**作者:** Alberto Tamajo `[一作]` (University of Southampton), Rahman Attar `[通讯]` (University of Southampton)

**通讯引用:** 332 | [OpenAlex ID](https://openalex.org/A5102868165)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并量化了 Rehearsal-based CIL 中存在的类别不平衡遗忘现象，并构造了三种基于最后层梯度的指标来预测遗忘排名。

**💡 创新点**

首次提出 Last-Layer Imbalanced Forgetting Coefficients（SIC、CIC、NIC）并通过 Class-Wise R‑SGD Lemma 解析自噪、跨类干扰与新类梯度对不平衡遗忘的影响。

**🔧 技术方法**

采用干扰算子、梯度级别的 R‑SGD lemma、Spearman 相关、线性回归及留一交叉验证等技术评估指标预测性能，使用标准 Rehearsal 策略与 R‑SGD 优化。

**📊 数据集**

在 CIFAR‑100 与 Tiny‑ImageNet 上构建全因子化基准（Ω_C100、Ω_TIN），覆盖不同类别数与回放比例。

**📈 对比分析**

通过 Forgotten Range（FG‑R）和 Forgetting Half‑Gap（FG‑HG）衡量不平衡程度；指标预测的 Spearman 相关平均值在 CIFAR‑100 上超过 0.8、Tiny‑ImageNet 上约 0.6，联合模型进一步提升至约 0.9，能够准确预测遗忘排序。

**⚠️ 局限性**

仅考虑标准 Rehearsal 与最后层梯度，未探索更复杂模型或全参数梯度；SIC 受新类分布复杂度影响且可迁移性有限，缺乏因果验证。

---

## 506. Holistic Evaluation and Failure Diagnosis of AI Agents

**arXiv ID:** 2605.14865 | [PDF](https://arxiv.org/pdf/2605.14865v1)

**作者:** Netta Madvil `[一作]` (Deepchecks), Shir Chorev `[通讯]` (Deepchecks)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个整合自上而下与自下而上的评估框架，对AI代理执行的多步骤过程进行细粒度的错误定位与诊断。

**💡 创新点**

创新点在于将跨度级别的错误检测与整个执行轨迹的行为评估结合，采用逐跨度独立评估并向上传播，从而在长链轨迹中保持定位准确性。

**🔧 技术方法**

利用OpenTelemetry结构化轨迹，采用LLM评估器对叶子跨度进行多维度度量（指令遵循、推理完整性等），并通过规则或LLM映射生成最终错误类别。

**📊 数据集**

在TRAIL基准上对GAIA和SWE‑Bench两套数据集进行评估，TRAIL提供148条OpenTelemetry轨迹和841条跨度级错误标签。

**📈 对比分析**

与单一LLM评估器、Agent GPA、AgentCompass等基线相比，框架在定位准确率、加权类别F1和联合定位分类准确率上分别提升至GAIA 38%、3.5×、12.5×，SWE‑Bench提升更显著；模型在长轨迹上保持高准确度而基线易崩溃。

**⚠️ 局限性**

局限性包括对错误注释质量的依赖、聚合策略仍过于规则化、类别边界模糊以及未充分利用评估者产生的自然语言理由。

---

## 507. Beliefs and Misconceptions around Integrated Conversational AI

**arXiv ID:** 2605.14849 | [PDF](https://arxiv.org/pdf/2605.14849v1)

**作者:** William Seymour `[一作]` (King's College London), Jose Such `[通讯]` (Ingenio (CSIC-Universtiat Politècnica de València))

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过半结构化访谈和思考口述实验，探讨了用户在 Microsoft Edge 浏览器中集成的 Copilot 对话式 AI（基于 GPT‑4）时的使用策略、信念、信任与隐私认知；

**💡 创新点**

创新点在于揭示用户将 Copilot 视为高级搜索引擎的心理模型，发现引用标注被视为低成本信任信号而非真正验证机制，并提出将引用用作系统 1 级错误指示而非信任标记的设计思路；

**🔧 技术方法**

研究采用了半结构化访谈、思考口述任务、主题分析方法，并以 Microsoft Edge 内置 Copilot（GPT‑4）作为实验工具；

**📊 数据集**

使用的数据集为 20 名大学生受访者产生的访谈录音与转录文本；

**📈 对比分析**

对方法的比较主要通过主题编码和对照分析完成，并未进行量化性能评测；结果显示用户普遍信任引用，但若引用错误则难以识别，说明当前引用机制对信任的影响有限；

**⚠️ 局限性**

局限性包括样本量有限、仅聚焦大学生群体、使用商业版 Copilot 版本不可控、未对 LLM 生成结果的准确性进行评估，且研究缺乏跨平台或跨任务的对比。

---

## 508. Editor's Choice: Evaluating Abstract Intent in Image Editing through Atomic Entity Analysis

**arXiv ID:** 2605.14842 | [PDF](https://arxiv.org/pdf/2605.14842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 509. GenAI for Energy-Efficient and Interference-Aware Compressed Sensing of GNSS Signals on a Google Edge TPU

**arXiv ID:** 2605.14839 | [PDF](https://arxiv.org/pdf/2605.14839v1)

**作者:** Thorben Wegner `[一作]` (Fraunhofer Institute for Integrated Circuits IIS), Alexander Rügamer `[通讯]` (Fraunhofer Institute for Integrated Circuits IIS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在GNSS信号检测与分类中，提出了基于GenAI的变分自编码器（VAE）压缩方案，并在Google Edge TPU上实现实时压缩与干扰分类；

**💡 创新点**

创新点在于将VAE与FactorVAE等离散化、解耦技术应用于GNSS信号，既实现97.6%压缩率，又保持92%以上F₂分数，且在Edge TPU上完成能耗低的推理；

**🔧 技术方法**

使用的技术包括：变分自编码器（VAE、cVAE、FactorVAE）、8-bit量化、PyTorch→ONNX→TensorFlow Lite转换、Edge TPU编译、随机森林后置分类器；

**📊 数据集**

采用多种数据集：实验室低成本GNSS传感器收集的光谱/时域/原始IQ数据、车道高频IQ快照数据、72类干扰全量数据；

**📈 对比分析**

通过与原始数据的F₂分数、能耗、压缩率比较，结果显示：原始数据F₂≈0.98，压缩后F₂≥0.91；能耗从1,203 mWh降至130 mWh，网络传输量下降≈67%；

**⚠️ 局限性**

限制包括：离散化后仍有少量分类损失；仅在特定硬件（Edge TPU）上验证；对极端多路径/强干扰场景的鲁棒性仍待进一步评估；

---

## 510. Emotion-Attended Stateful Memory (EASM):The Architecture for Hyper-Personalization at Scale

**arXiv ID:** 2605.14833 | [PDF](https://arxiv.org/pdf/2605.14833v1)

**作者:** Vineet Kotecha `[一作]` (divAIne Research), Vansh Gupta `[通讯]` (divAIne Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并评估了 Emotion-Attended Stateful Memory (EASM) 架构，通过情绪感知、意图推断与长期记忆检索动态构建用户上下文，进行30个非剧本对话的A/B实验验证其对个性化对话的提升。

**💡 创新点**

将情绪作为检索键的双向索引记忆、图数据库结构化情境关系、实时情绪与意图推理作为上下文生成模块，并证明其在多情绪场景下显著提升对话质量。

**🔧 技术方法**

结合 Qdrant 向量检索、Neo4j 图数据库、情绪识别（语音+文本融合）、意图分类模型、情绪融合模块、响应策略生成器，基于同一 LLM 生成响应。

**📊 数据集**

30个对话场景（6类共5场）基于单一用户人设，使用 GPT‑4o 生成用户模拟器对话；评估使用自定义的情绪验证、计划清晰度等五维度评分体系。

**📈 对比分析**

采用双盲 A/B 对照，判断者不知条件；在所有场景中，富含记忆的条件被选中100%，平均提升分别为情绪验证34%、计划清晰57%、语气34%、安全性42%、记忆根植95%；表明记忆丰富显著提升对话质量。

**⚠️ 局限性**

仅单一用户人设、单一判定模型、单轮对话、文本评估、记忆根植指标可能受显式引用偏倚，缺乏多模态、多文化、多用户、多轮纵向验证。

---

## 511. Conversion of Lexicon-Grammar tables to LMF. Application to French

**arXiv ID:** 2605.14816 | [PDF](https://arxiv.org/pdf/2605.14816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 512. Learning Direct Control Policies with Flow Matching for Autonomous Driving

**arXiv ID:** 2605.14832 | [PDF](https://arxiv.org/pdf/2605.14832v1)

**作者:** Marcello Ceresini `[一作]` (Università degli Studi di Parma), Alberto Broggi `[通讯]` (VisLab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出一种基于鸟瞰图（BEV）的条件流匹配规划器，能够在实时闭环中直接输出加速度与曲率控制序列；

**💡 创新点**

创新点在于将流匹配与BEV编码相结合，既实现了低延迟的ODE集成，又在闭环评估中展示了对分布外场景的稳健泛化；

**🔧 技术方法**

主要技术包括条件流匹配（Conditional Flow Matching）框架、轻量级U‑Net向量场预测、BEV编码CNN、Euler ODE求解器以及基于规则的2D交通仿真；

**📊 数据集**

使用了由意大利帕尔马城市真实道路与环岛生成的≈19k条成功驾驶轨迹（约35小时），通过2D仿真采集并转化为BEV栅格；

**📈 对比分析**

采用闭环评估指标（碰撞率、道路遵循率、路线完成率、预测/执行抖动），与单步NFE=1对比，NFE=10后碰撞率降至≈2.7%，道路遵循率提升至≈98%，路线完成率≥98%，抖动指标显著下降，证明模型在分布外场景亦能实现安全、平顺的规划；

**⚠️ 局限性**

局限性包括仅在2D仿真环境下验证，缺乏感知噪声与时序信息；模型仅观测单帧BEV，难以预判激进非Ego行为；在高速场景下车道保持精度不足；需进一步在真实感知与更复杂交通交互中验证。

---

## 513. A class of optimal authentication codes with secrecy

**arXiv ID:** 2605.14823 | [PDF](https://arxiv.org/pdf/2605.14823v1)

**作者:** Haibo Liu `[一作]` (Chengdu University of Information Technology), Qunying Liao `[通讯]` (Sichuan Normal University)

**通讯引用:** 408 | [OpenAlex ID](https://openalex.org/A5073944562)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文构造了一类基于有限域代数结构的有保密性的线性认证码，并给出了其加密和认证规则。

**💡 创新点**

创新点在于利用特殊 Weil 和指数求和计算出代替攻击和冒充攻击的最大成功概率，并证明该类码在信息理论和组合学意义下渐近最优。

**🔧 技术方法**

主要技术包括有限域的加法和乘法字符、Gauss 和 Weil 求和、以及对齐映射的代数性质。

**📊 数据集**

该工作为理论研究，无需外部数据集，所有实验均为解析推导。

**📈 对比分析**

通过与已有信息理论下的下界和组合学下界的比较，结果显示 P_I 与 P_S 随 n→∞ 接近下界，表明码的安全性和效率均达到最优水平。

**⚠️ 局限性**

限制在于对代替攻击成功概率的精确计算过于复杂，本文仅给出上界，因而无法完全证明 P_S 的渐近最优；此外，构造仅适用于无拆分（non‑splitting）情形。

---

## 514. Do Composed Image Retrieval Benchmarks Require Multimodal Composition?

**arXiv ID:** 2605.14787 | [PDF](https://arxiv.org/pdf/2605.14787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 515. The Velocity Deficit: Initial Energy Injection for Flow Matching

**arXiv ID:** 2605.14819 | [PDF](https://arxiv.org/pdf/2605.14819v1)

**作者:** Linze Li `[一作]` (Jiiov Technology), Jiajun Liang `[通讯]` (Jiiov Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析并解决了流匹配（Flow Matching）中存在的速度不足（Velocity Deficit）导致的积分滞后（Integration Lag）问题，提出了两种“初始能量注入”策略：训练基的 Magnitude‑Aware Flow Matching（MAFM）和推断时的 Scale Schedule Corrector（SSC），并展示其在多种生成任务中的显著性能提升。

**💡 创新点**

创新点包括：①在高维空间中首次理论证明 MSE 目标导致的速度范数退化；②揭示速度收缩在时间两端的对称性：起始时为信号缺失、终止时为噪声衰减；③基于此提出非对称能量注入的概念，并实现了无需再训练、仅一行代码即可显著提升采样效率和图像质量的 SSC。

**🔧 技术方法**

技术手段主要包括：流匹配与连续归一化流（CNF），MSE 目标与其分析，MAFM 训练目标改进，SSC 推断时速度缩放，Euler/Heun 数值积分，指导技术（CFG），以及对 REPA、MMDiT 等架构的兼容性实验。

**📊 数据集**

实验使用的主要数据集有 ImageNet‑1k（256×256 与 512×512），MS‑COCO（文本到图像任务），以及在 U‑Net 等 CNN 架构上的验证。

**📈 对比分析**

对比实验显示：在 ImageNet‑256×256 上，SSC 在仅 50 次函数评估（NFE）下实现 FID 7.58（比 250 步基线 8.65 提升 44%），而 MAFM 在 50 步得到 FID 14.26；在 MS‑COCO 上，SSC 将 FID 从 6.03 降至 4.71；同时还在高分辨率（512×512）和多路径（线性、SBDM‑VP、GVP）上验证了性能提升，且 SSC 兼容不同模型与指导设置。

**⚠️ 局限性**

局限性：①理论基于高维正交性，低维或非正交场景可能导致过冲；②在极端高 CFG（如 CFG ≥ 4.0）下，默认的 s_start 可能引起能量过载，需要进一步调参；③SSC 的能量注入策略虽简单，但对超大模型可能出现细节过平滑的风险。

---

## 516. Exploring Bottlenecks in VLM-LLM Navigation: How 3D Scene Understanding Capability Impacts Zero-Shot VLN

**arXiv ID:** 2605.14801 | [PDF](https://arxiv.org/pdf/2605.14801v1)

**作者:** Ziyi Xia `[一作]` (Shanghai Jiao Tong University), Ling Pei `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 4369 | [OpenAlex ID](https://openalex.org/A5021661339)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

定量分析3D场景理解能力对零射知识视觉语言导航的影响，提出并验证统计成功率上界，揭示感知饱和现象；

**💡 创新点**

引入保留率与IoU驱动的感知降级模拟，分别针对慢脑规划与快脑导航构建上界估计，验证核心词汇规模对导航成功率的边际效益；

**🔧 技术方法**

采用VLM‑LLM框架SFCo‑Nav，结合ESAM（基于Segment Anything Model）的3D实例分割，使用S_match、S_obj、S_edge等匹配指标，以及GPT‑4o进行规划；

**📊 数据集**

使用R2R视觉语言导航数据集进行导航评估，ScanNet200用于ESAM的部分训练与评估；

**📈 对比分析**

通过将感知降级参数映射到ESAM的AP与IoU指标，对不同ESAM变体进行SR、OSR、SPL的对比实验，发现SR随感知提升趋于饱和，ESAM在核心词汇规模20时已接近上界；

**⚠️ 局限性**

仅考虑缺失检测与中心偏移，未覆盖尺度、角度误差；误报和边框比例畸变对性能影响显著；实验环境局限于R2R，未检验更复杂场景下的泛化能力。

---

## 517. Supervised Distributed Computing: Efficiency and Robustness under a Majority of Adversarial Workers

**arXiv ID:** 2605.14784 | [PDF](https://arxiv.org/pdf/2605.14784v1)

**作者:** John Augustine `[一作]`, Julian Werthmann `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种监督式分布式计算框架，在该框架中，主控节点仅负责调度工作者，输入由可靠源头提供，输出由可靠目标节点收集，适用于任意有向无环任务图；

**💡 创新点**

该方法能在任何常数β<1的攻击者比例下保证计算正确性，并且期望工作量接近单次执行，同时仅需对每个工作者执行一次轻量级验证；

**🔧 技术方法**

核心技术包括：基于采样的工作者分配、将任务图层化并采用δ和γ参数实现流水式调度、利用轻量级验证机制、构造有效的“证人序列”进行概率分析以及使用组合与信息论方法证明高概率成功；

**📊 数据集**

论文为理论性工作，没有使用具体数据集，而是对任意给定的DAG任务图和输入进行分析；

**📈 对比分析**

与之前仅在小β下有效的监督式方案以及传统的主从或对等方案相比，本工作在任意β<1时实现了O(D·log_{1/β}d·log_{1/β}log n)轮的时间复杂度，期望总工作量为n(1+o(1))，并且通信量与任务图规模线性相关；

**⚠️ 局限性**

主要局限在于：需要任务能够被轻量级验证；假设工作者采样独立且同步轮；算法对β接近1的情况在参数取值上仍有挑战；以及缺乏对异构任务图的实验验证。

---

## 518. Embedded Made Easy -- Rethinking Embedded + Cloud Software Development (WIP)

**arXiv ID:** 2605.14863 | [PDF](https://arxiv.org/pdf/2605.14863v1)

**作者:** Anthony Arnold `[一作]` (University of Kentucky), Mark Marron `[通讯]` (University of Kentucky)

**通讯引用:** 1425 | [OpenAlex ID](https://openalex.org/A5024427812)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个统一的语言和运行时系统，旨在让同一套代码能够在从资源受限的边缘设备到大规模云服务器的任何硬件上无缝部署、自动扩缩容，并提供统一的诊断与调试工具。

**💡 创新点**

创新点包括：
1) 结合语言与运行时的可扩缩 GC 设计，实现小型 256 B 寄生区到大内存的无缝迁移；
2) 通过 BAPI 接口层将所有外部交互显式化，保证跨平台位级一致性；
3) 引入任务(Task)并发模型与结构化错误处理，支持自动捕获并转化为可处理的错误对象；
4) 内置可跨平台记录与重放分布式执行轨迹的机制，实现本地化故障复现。

**🔧 技术方法**

使用的技术包括：
- 自研语言（未命名）与对应的单语言运行时；
- 具有无锁特性的可扩缩垃圾回收器（结合循环自由数据结构、参照透明性）；
- 轻量级 Task 与结构化并发；
- BAPI 接口层实现平台无关的系统调用与 RPC（JSON‑RPC/OpenAPI）；
- 运行时自动捕获交互并生成可重放的执行日志与逻辑状态快照。

**📊 数据集**

实验主要采用合成分配密集型工作负载（benchmark），与标准 Java 虚拟机（JVM）GC 进行对比；未公开具体数据集，但描述为“allocation intensive workload”。

**📈 对比分析**

比较方法：在同一计算节点上运行单线程 GC 与多线程 JVM，测量暂停时间、吞吐量与内存占用。结果显示：
- GC 开销 5–10%；
- P99 暂停 < 1 ms；
- 单线程 GC 吞吐量约为 JVM 的 50%，但内存占用仅为 JVM 的 1.5×（而 JVM 8–12×）。
- 在 400 MB 限制下，GC 性能约为 JVM 的 2×。

**⚠️ 局限性**

局限性：
- 目前 GC 仍为单线程/停止世界实现，正在开发并发版本；
- 语言实现仍依赖 C++ 标准库，未完全去除外部依赖；
- 仅使用合成基准，缺乏真实应用案例和多样化数据集；
- 需要进一步验证在多设备真实网络环境下的分布式调试与日志传播效果；
- 现有系统在极低资源场景（例如几百 KB 内存）下仍可能受限。

---

## 519. Exploitation of Hidden Context in Dynamic Movement Forecasting: A Neural Network Journey from Recurrent to Graph Neural Networks and General Purpose Transformers

**arXiv ID:** 2605.14855 | [PDF](https://arxiv.org/pdf/2605.14855v1)

**作者:** Lukas Schelenz `[一作]` (Fraunhofer Institute for Integrated Circuits IIS), Tobias Feigl `[通讯]` (Fraunhofer Institute for Integrated Circuits IIS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

比较并评估多种机器学习模型（TCNN、LSTM、CNN‑LSTM、LMU、GNN（GAT）和Transformer）在NBA球员短期轨迹预测中的表现，并提出了在递归网络中显式嵌入空间上下文的混合架构。

**💡 创新点**

创新点在于：①在LSTM内部显式加入距离变换的空间上下文，②构建RNN‑GAT‑RNN融合模型以同时捕获时间序列和多体交互；③系统性研究了不同历史窗口长度、同队与跨队泛化及不同预测时长下的性能权衡。

**🔧 技术方法**

使用的技术包括：长短时记忆网络（LSTM）、时间卷积网络（TCNN）、卷积‑LSTM（CNN‑LSTM）、Legendre记忆单元（LMU）、图注意网络（GAT）与图神经网络（GNN）以及基于自注意力的Transformer；此外采用了距离变换函数、位置编码、CNN时间嵌入等辅助技术。

**📊 数据集**

使用公开的NBA 2015‑2016赛季数据集，共计508小时、25Hz采样，包含每场比赛的球员、球、篮筐等轨迹信息。

**📈 对比分析**

通过ADE、FDE、AAE、FAE四项指标对所有模型进行量化比较。实验表明：在2 s历史窗口、2 s预测时长下，CNN‑LSTM以1.51 m的FDE取得最佳结果；LSTM、LMU、GNN在中长时长也表现优异；Transformer相对稍逊，主要受数据量不足影响。模型在跨队泛化时误差增幅仅0.02–0.07 m，说明具有一定鲁棒性。

**⚠️ 局限性**

主要局限包括：①缺乏足够的训练数据导致Transformer和GNN未能充分发挥潜力；②当前上下文建模仍以距离为主，未充分利用更丰富的比赛策略信息；③对极短预测时长（<0.12 s）仍无法显著超越传统速度保持基线；④模型未能在所有预测时长与数据规模下实现统一最优。

---

## 520. First Mathematical Runtime Analyses of Multi-Objective Evolutionary Algorithms for Multi-Valued Decision Variables

**arXiv ID:** 2605.14836 | [PDF](https://arxiv.org/pdf/2605.14836v1)

**作者:** Mingfeng Li `[一作]` (Harbin Institute of Technology), Benjamin Doerr `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 6100 | [OpenAlex ID](https://openalex.org/A5102961331)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多值决策变量下的多目标进化算法（SEMO）进行数学运行时间分析，给出了上界与下界并对变体实现了精确上界。

**💡 创新点**

首次为有限多值域的多目标优化提供正式的运行时间上界与下界，并证明了原始SEMO与只接受严格更优解的变体具有相同的渐进阶数。

**🔧 技术方法**

采用Chernoff界、几何分布尾结论、乘法漂移定理等概率工具进行复杂的群体动力学分析，并利用实验验证理论。

**📊 数据集**

使用自定义的两种合成基准：r-值化的G_{r}（所有解都是Pareto最优）及其变体（只有一条Pareto最优路径），在不同n、r组合上进行100次独立实验。

**📈 对比分析**

通过对比原始SEMO与变体在两基准上的平均迭代次数与标准差，实验显示两者性能相近，变体略快或相等，支持理论结论。

**⚠️ 局限性**

理论上仍存在原始SEMO的运行时间与下界之间的O(log n)差距；目前的分析方法无法完全消除此差距，需要更强的工具或新的分析思路。

---

## 521. Can Visual Mamba Improve AI-Generated Image Detection? An In-Depth Investigation

**arXiv ID:** 2605.14799 | [PDF](https://arxiv.org/pdf/2605.14799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 522. CaMeRL: Collision-Aware and Memory-Enhanced Reinforcement Learning for UAV Navigation in Multi-Scale Obstacle Environments

**arXiv ID:** 2605.14810 | [PDF](https://arxiv.org/pdf/2605.14810v1)

**作者:** Hong Hong `[一作]` (Sun Yat-sen University), Hejun Wu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1436 | [OpenAlex ID](https://openalex.org/A5102755158)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种结合碰撞感知与时间记忆的CaMeRL框架，用于解决多尺度障碍物环境中的无人机视觉导航。

**💡 创新点**

创新点在于同时引入碰撞感知潜在表示学习和跨帧记忆增强，以提升对超小与超大障碍物的检测与规避能力。

**🔧 技术方法**

技术上使用变分自编码器(VAE)提取碰撞感知深度特征，LSTM实现时间记忆整合，并通过PPO训练端到端的控制策略。

**📊 数据集**

数据集包括在Flightmare中构建的七个不同尺度障碍物测试环境（US、S、M、L、XL、MIX）以及真实森林场景的深度图像。

**📈 对比分析**

与MAVRL、Agile‑autonomy等基准对比，CaMeRL在所有尺度下成功率最高，尤其在超小和超大环境中提升可达0.48，且保持较高飞行速度。

**⚠️ 局限性**

局限性在于仅针对静态障碍物进行评估，缺乏对动态障碍的验证，且记忆窗口长度对不同环境的泛化仍需进一步研究。

---

## 523. Learning Cross-Coupled and Regime Dependent Dynamics for Aerial Manipulation

**arXiv ID:** 2605.14805 | [PDF](https://arxiv.org/pdf/2605.14805v1)

**作者:** Rishabh Dev Yadav `[一作]` (University of Manchaster), Wei Pan `[通讯]` (Newcastle University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个结构化的编码器-解码器框架，用于实时学习并适应无人机操纵器的残差动力学。

**💡 创新点**

创新之处在于用历史依赖的非线性编码器捕捉跨变量耦合与时间延迟，并通过线性解码器实现闭式贝叶斯在线自适应与一致性驱动的协方差膨胀。

**🔧 技术方法**

使用了跨变量注意力、时间注意力、序列编码、线性贝叶斯更新、协方差膨胀以及MPC闭环控制等技术。

**📊 数据集**

实验数据来自真实空中操纵平台，收集了携带300g和500g负载的轨迹跟踪数据。

**📈 对比分析**

与基准模型（Nominal、DNN、Diffusion、Proto-MPC、Active MLP等）对比，实验显示预测误差最小、适应速度最快、轨迹跟踪RMSE最低。

**⚠️ 局限性**

局限在于仅验证了非接触、相对平稳负载变化场景；线性解码器限制了对极端非线性变化的建模，并且需离线训练且假设噪声为高斯。

---

## 524. COAL: Counterfactual and Observation-Enhanced Alignment Learning for Discriminative Referring Multi-Object Tracking

**arXiv ID:** 2605.14795 | [PDF](https://arxiv.org/pdf/2605.14795v1)

**作者:** Shukun Jia `[一作]` (Southeast University), Xiaobo Lu `[通讯]` (Southeast University)

**通讯引用:** 3436 | [OpenAlex ID](https://openalex.org/A5066658319)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 COAL 框架，通过显式语义注入（VLM 生成检测框和描述）和因果对抗学习（LLM 生成硬负样本）来缓解 RMOT 中的稀疏监督导致的辨别瓶颈，从而提升目标识别与跟踪性能。

**💡 创新点**

创新点在于：①将外部基础模型（VLM、LLM）作为先验知识注入，显著稠密观察空间并提供细粒度属性；②采用因果对抗学习在训练中强制模型区分细微属性，防止捷径学习；③设计层次化多流融合结构（HMSI），实现视觉、语言、空间信息的协同优化。

**🔧 技术方法**

使用 CLIP 视觉/文本编码器、VLM（如 DINO‑X）进行检测与生成描述、LLM（如 GPT）生成对抗负样本、变形采样、双向融合（Bi‑Fusion）、层次化语义细化（FLL、ALG）、对齐投影与余弦相似度匹配，并在训练中采用 BCE 损失与因果对抗损失。

**📊 数据集**

在 Refer‑KITTI（15 训练视频，818 表达）和 Refer‑KITTI‑V2（21 视频，9758 表达）两个基准上进行评估。

**📈 对比分析**

与现有方法对比，COAL 在 Refer‑KITTI 上达 53.38% HOTA（领先 0.97%），在 Refer‑KITTI‑V2 上达 43.46% HOTA，超过第二名 7.28%，并在 DetA、AssA、LocA 等指标上均取得显著提升。

**⚠️ 局限性**

局限性包括：①对 VLM/LLM 产生的离线先验依赖，导致推理时需额外前处理；②模型规模较大，计算和存储开销相对较高；③在更广泛的 RMOT 场景或不含强语义描述的数据集上的泛化能力尚待进一步验证。

---

## 525. Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces

**arXiv ID:** 2605.14786 | [PDF](https://arxiv.org/pdf/2605.14786v1)

**作者:** William Lugoloobi `[一作]` (University of Oxford), Chris Russell `[通讯]` (University of Oxford)

**通讯引用:** 8613 | [OpenAlex ID](https://openalex.org/A5008943199)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在网页上注入轻量级 JavaScript 追踪器，记录 LLM 代理的点击、滚动、键入等 UI 交互事件，证明仅凭这些被动行为痕迹即可识别出背后使用的 14 种前沿语言模型，最高 F1 达到 96%。

**💡 创新点**

创新点在于：①首次将“代理身份识别”视为细粒度多类别分类任务；②提出了完整的威胁模型与防御分析；③释放了公开的标注交互轨迹数据集与评估工具，方便后续研究。

**🔧 技术方法**

主要技术包括：①JavaScript 事件监听器收集交互轨迹；②基于时间间隔、动作结构等特征的特征工程；③使用 XGBoost（及 LSTM、随机森林等）进行多类别分类；④利用 SHAP 分析特征重要性；⑤对延迟注入与自适应训练的鲁棒性评估。

**📊 数据集**

使用的数据集包含四个 Web 任务：信息检索任务（2WikiMultiHop、FRAMES，目标网站 Wikipedia.com）和在线购物任务（Webshop、Deepshop，目标网站 Amazon.com），共 1050 条交互轨迹，覆盖 14 种 LLM 模型。

**📈 对比分析**

与传统人类 vs. 机器人检测相比，本文的多类别识别在闭集情形下 macro F1 超过 70%，最高 96%；在开放集检测中 AUROC 多数模型超过 0.6，最高 0.84；同时发现仅需 10% 轨迹即可获得近似最佳性能，且可在会话 40% 时完成识别。

**⚠️ 局限性**

局限性包括：仅在单一 Midscene.js harness 上实验，可能未涵盖不同代理实现的差异；开放集识别仍不够稳健，部分模型（如 Seed‑2‑lite）在开放集下表现低于随机；对时间延迟的防御需要重新训练；且实验仅关注浏览器端 UI 事件，未探讨网络层或后端特征的组合。

---

## 526. Graphs of Research: Citation Evolution Graphs as Supervision for Research Idea Generation

**arXiv ID:** 2605.14790 | [PDF](https://arxiv.org/pdf/2605.14790v1)

**作者:** Songyang Gao `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45730 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Graphs of Research (GoR) 框架，构造 2-hop 引用演化 DAG 并用结构化文本序列化对 Qwen2.5‑7B‑Instruct‑1M 进行监督式微调，以生成更高质量、更具创新性的研究想法。

**💡 创新点**

创新点在于把引用演化图结构作为训练时的监督信号，而不是仅将参考文献视为平面文本；证明结构化图信息能显著提升 LLM 的创新度、可行性、清晰度等评估维度。

**🔧 技术方法**

采用 2‑hop 引用子图抽取、边特征注释、结构化文本序列化；使用 Qwen2.5‑7B‑Instruct‑1M 进行 supervised fine‑tuning；评估方法包括 LLM‑judge 对决、多维度表面指标（wTop1、mROUGE、BERT‑F1）和人类 10‑维度评价。

**📊 数据集**

构建的数据集来源于 NeurIPS、ICLR、CVPR、ICML、ACL 2020‑2024 年论文，共 498 篇训练集、50 篇验证集和 50 篇 2025 年测试集，涵盖约 7,600 条引用，确保测试集在时间上无泄漏。

**📈 对比分析**

通过与三种 gpt‑4o 基线（Si、CoI‑Agent、ResearchAgent）在 50 个种子论文上进行 LLM‑judge 对决和表面指标对比，GoR‑SFT 在 31/40/48 种子上排名第一，整体 Elo 分数比基线高 3–5 分；在 32/50 种子上亦能战胜参数更大、仅使用图作为 prompt 的 gpt‑4o，展示了在成本与质量上的优势。

**⚠️ 局限性**

局限性包括：仅在 7B LLM 上实验，需更大模型验证其可扩展性；图结构构建依赖自动化抽取，质量受 PDF 解析、引用检测等因素影响；对低引用量或无图论文的适用性尚未评估；未来需扩展到更大规模的数据集和多学科场景。

---

## 527. SuperADD: Training-free Class-agnostic Anomaly Segmentation -- CVPR 2026 VAND 4.0 Workshop Challenge Industrial Track

**arXiv ID:** 2605.14808 | [PDF](https://arxiv.org/pdf/2605.14808v1)

**作者:** Lukas Roming `[一作]` (Fraunhofer IOSB), Jürgen Beyerer `[通讯]` (Fraunhofer IOSB)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练‑免费、类别‑无关的视觉异常检测管线，在测试阶段仅使用预训练的 DINOv3 提取特征并通过最近邻搜索构建内存库，实现异常分割。

**💡 创新点**

创新点包括：①在特征空间直接做近邻采样以构建更紧凑、多样的内存库；②使用重叠 patch 处理降低边界敏感；③加入强度增强模拟光照变化；④在后处理阶段采用多方向迭代闭运算提升空间一致性；⑤统一阈值策略，避免每类专门调参。

**🔧 技术方法**

核心技术有：DINOv3 视觉 Transformer、内存库最近邻检测、patch‑wise 并行推理、图像强度增强、形态学闭运算、阈值自适应。

**📊 数据集**

在 MVTec AD 2 数据集上进行实验，包含多种物体类别及光照分布差异的公共/私有测试集。

**📈 对比分析**

与 PatchCore、EfficientAD、ISVL、RoBiS 等前沿方法对比，取得 57.42%（公测）和 54.35%（混合光照） 的平均像素级分割分数，显著优于去年最佳 53.81%/51.43%，并在多数单类上排名第一。

**⚠️ 局限性**

局限性：对极细缺陷（如薄划痕、单根头发）检测灵敏度不足；在缺失部件（如断裂壁塞、空 vial）时易漏检；阈值选择对某些类别影响显著；在光照极端变化（如 can 类）性能下降约 12%。

---

## 528. A Hardware-Aware, Per-Layer Methodology for Post-Training Quantization of Large Language Models

**arXiv ID:** 2605.14929 | [PDF](https://arxiv.org/pdf/2605.14929v1)

**作者:** Earl Killian `[一作]` `[通讯]`, Earl Killian

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Scaled Outer Product（SOP）后训练量化方法，支持块尺度、双代码本搜索、稀疏残差和离群值提取，以实现低位宽(4.5–6 bpw)的LLM权重量化。

**💡 创新点**

通过每层代码本对比搜索、按激活权重加权余弦相似度评估、可选离群值和稀疏残差补偿，以及多选背包分配，将低精度权重与块尺度与多元码本组合，显著提升在同等存储下的精度。

**🔧 技术方法**

采用块尺度、两本码本配对搜索、激活加权余弦相似度(ACos)、离群值提取(OPQ)、稀疏残差纠正(Wr)、多选背包分配、12-bit带符号的尺度格式、HIF7/8 LUT以及FC‑SRAM实现的SOP微内核。

**📊 数据集**

在六个公开LLM族群（Gemma-3-1B, SmolLM3-3B, Llama-3.2-3B, Qwen3.5-4B, Mistral-7B, Qwen3-8B）上使用小型校准语料进行通道范数计算与量化评估。

**📈 对比分析**

与每层POT FP8(8.0 bpw)基线相比，推荐的FP6(6.5 bpw)块尺度方案在权重MSE上提升约15–20%，并在1.5 bpw存储成本下实现近零损失；实验显示ACos与KL的相关性强，离群值和残差补偿进一步降低误差。

**⚠️ 局限性**

仍受码本尺寸和块大小选择的限制，3-bit或更低位宽尚无可行的码本几何；对极端分布或大尺度模型的推广依赖于更多层级的多元码本和更高位宽尺度，且在GPU无12-bit尺度存储时需放弃符号位导致精度略降。

---

## 529. SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for Topological Crack Segmentation

**arXiv ID:** 2605.14926 | [PDF](https://arxiv.org/pdf/2605.14926v1)

**作者:** Hanxu Zhang `[一作]` (Tianjin University of Technology), Shengyong Chen `[通讯]` (Tianjin University of Technology)

**通讯引用:** 14048 | [OpenAlex ID](https://openalex.org/A5055627037)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了极轻量级的结构校准Vision‑RWKV网络SCRWKV，用于高精度裂纹分割。

**💡 创新点**

创新点包括：基于线性状态空间的SFE骨干，结合自适应多尺度AMCM、双向几何变换GBST以及动态自校正衰减DSCD，实现对裂纹拓扑与噪声的同时建模；以及跨尺度谐波融合CSHF解码器，实现精细特征聚合。

**🔧 技术方法**

采用Vision‑RWKV架构、线性时间复杂度的状态空间模型、可学习的空间/通道混合机制、动态衰减权重、跨尺度注意力融合等技术。

**📊 数据集**

在四大公开数据集上验证：Crack500、DeepCrack、CrackMap 与 TUT。

**📈 对比分析**

与10种SOTA方法比较，SCRWKV在TUT上取得F1 0.8428、mIoU 0.8512，参数仅1.22M（28MB），在所有数据集上均为最优或次优，显示出极佳的精度与轻量化兼顾。

**⚠️ 局限性**

局限性在于相较于最轻量化方法稍高的FLOPs，并未针对更大尺寸或极高分辨率场景做进一步验证；目前仅针对裂纹分割，需进一步验证在其他细粒度分割任务的通用性。

---

## 530. Road Maps as Free Geometric Priors: Weather-Invariant Drone Geo-Localization with GeoFuse

**arXiv ID:** 2605.14925 | [PDF](https://arxiv.org/pdf/2605.14925v1)

**作者:** Yunsong Fang `[一作]` (University of Macau), Zhedong Zheng `[通讯]` (University of Macau)

**通讯引用:** 10089 | [OpenAlex ID](https://openalex.org/A5034162160)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用无人机拍摄的恶劣天气下的图像与卫星图像进行跨视角地理定位，并通过道路地图提供天气不变的几何先验实现匹配。

**💡 创新点**

创新点：①引入免费道路地图作为几何先验；②双层（token 与 channel）融合模块配合轻量级动态门控自适应平衡多模态贡献；③类级对比学习弥补噪声天气下的特征退化。

**🔧 技术方法**

技术手段：跨模态融合 Transformer、动态门控机制、类级对比损失、文本-图像对齐（WeatherPrompt + X-VLM）、数据增强与多天气预处理。

**📊 数据集**

使用扩充了道路地图的 University‑1652 与 DenseUAV 两大基准数据集。

**📈 对比分析**

与现有 SOTA 方法对比，GeoFuse 在多天气条件下分别提升了 University‑1652 上 Recall@1 +3.46% 与 AP +3.15%，DenseUAV 上 Recall@1 +23.18% 与 AP +23.20%，表现出显著且稳定的性能提升。

**⚠️ 局限性**

局限性：依赖道路地图的可用性与精确对齐；山区、无人区或道路稀缺地区缺乏有效几何先验时性能会明显下降。

---

## 531. SceneParser: Hierarchical Scene Parsing for Visual Semantics Understanding

**arXiv ID:** 2605.14923 | [PDF](https://arxiv.org/pdf/2605.14923v1)

**作者:** Pengxin Xu `[一作]` (Harbin Institute of Technology), Xingyu Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了面向交互的层级场景解析任务，并基于此构建了大规模层级基准SceneParser‑Bench以及对应的VLM层级解析器SceneParser；

**💡 创新点**

创新点在于：①引入明确的对象‑部件‑可供性（affordance）层级结构与跨层绑定；②使用结构完成伪标签保证层级完整性；③采用三阶段课程学习平衡定位与结构扩展；④开发可扩展的层级数据引擎，生成110K+训练样本；

**🔧 技术方法**

核心技术包括：VLM自回归层级生成（基于Rex‑Omni）、结构完成伪标签、三阶段课程学习、跨模型融合（Grounding DINO、Rex‑Omni、SAM3）以及基于文本与几何的层级重构；

**📊 数据集**

主要使用的数据集为SceneParser‑Bench（110K训练+5K验证，包含777K对象、1.14M部件、1.74M可供性，2,905个对象类、12,349个部件类、85个可供性类）；此外在COCO与AGD20K上评估迁移性能；

**📈 对比分析**

评估方法：引入Level‑1~3条件指标（L1、L2、L3）以及ParseRate；与现有MLLM和单独预测拼接基线对比，SceneParser在L2/L3和ParseRate上显著领先，且在COCO/AGD20K任务中保持竞争力；

**⚠️ 局限性**

局限性：数据生成过程为自动化，仍可能存在标注噪声；仅使用2D框与点，未建模3D几何、动力学或执行轨迹；下游规划验证仅为定性，缺乏闭环机器人执行评估；长尾对象‑部件‑可供性组合的泛化能力尚待提升。

---

## 532. FU-MPC: Frontier- and Uncertainty-Aware Model Predictive Control for Efficient and Accurate UAV Exploration with Motorized LiDAR

**arXiv ID:** 2605.14920 | [PDF](https://arxiv.org/pdf/2605.14920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 533. AIMing for Standardised Explainability Evaluation in GNNs: A Framework and Case Study on Graph Kernel Networks

**arXiv ID:** 2605.14884 | [PDF](https://arxiv.org/pdf/2605.14884v1)

**作者:** Magdalena Proszewska `[一作]` (University of Edinburgh), N. Siddharth `[通讯]` (University of Edinburgh)

**通讯引用:** 547 | [OpenAlex ID](https://openalex.org/A5031241646)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了统一评估图神经网络可解释性的框架 AIM，并针对 Graph Kernel Networks (GKN) 开发了基于 SHAP 的实例级解释器，随后在 AIM 的指导下改进 GKN 生成可解释性更佳的 XGKN 模型。

**💡 创新点**

核心创新在于：①AIM 将 Accuracy、Instance-level 与 Model-level 三类指标整合为可跨模型、跨方法的评估体系；②针对 GKN 的 SHAP‑based 解释器能直接映射 kernel 响应到节点重要性；③XGKN 通过随机游走 kernel 与负熵聚合提升概念多样性与解释清晰度。

**🔧 技术方法**

使用技术包括：Graph Kernel Networks、Prototypical GNN、SHAP、Random Walk Kernel、负熵聚合、深度学习优化、归一化特征嵌入、基于 Spearman 相关的冗余度量等。

**📊 数据集**

实验数据集涵盖 6 个公开图数据集：BA-2motifs、BAMultiShapes、MUTAG、PROTEINS、IMDB-BINARY、IMDB-MULTI，后两者在无节点特征时使用度特征。

**📈 对比分析**

对比方法包括 GIN、GAT、ProtGNN、GKNN、KerGNN 等传统与解释性模型，使用 AIM 指标与预测准确率进行评估。XGKN 在实例级解释准确率 A1 与模型级解释准确率 A2 上均优于原 GKN，预测准确率与基线持平，且 SHAP‑based 解释器在提取速度与解释质量上显著优于其他后置解释器。

**⚠️ 局限性**

主要局限：①对复杂任务（非合成）的一致性 I5 低，解释随模型内部变化不稳定；②模型级解释与实例级解释对齐不足，说明 learned concepts 可能未被充分利用；③目前仅针对图分类任务，尚未验证在更广泛的图学习任务中的通用性。

---

## 534. HeatKV: Head-tuned KV-cache Compression for Visual Autoregressive Modeling

**arXiv ID:** 2605.14877 | [PDF](https://arxiv.org/pdf/2605.14877v1)

**作者:** Jonathan Cederlund `[一作]` (Lund University), Pontus Giselsson `[通讯]` (Lund University)

**通讯引用:** 1374 | [OpenAlex ID](https://openalex.org/A5030031115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对视觉自回归（VAR）模型的KV缓存压缩问题，提出了一种基于注意力头特定分配的HeatKV方法；

**💡 创新点**

创新点在于利用头部注意力得分的S‑CAS对不同尺度进行细粒度缓存分配，显著提升压缩率；

**🔧 技术方法**

使用了注意力权重聚合、S‑CAS评分、贪心早期剪枝算法以及Flash‑Attention Triton核等技术；

**📊 数据集**

实验使用MS‑COCO 2017验证集、GenEval和HPSv2.1等数据集进行评估；

**📈 对比分析**

与StreamingLLM、SnapKV、HACK、ScaleKV等基线对比，HeatKV在10%预算下PSNR/LPIPS/FID均优于20%预算的最强基线，压缩率提升约2×；

**⚠️ 局限性**

仍存在对极低预算下的鲁棒性不足，以及对不同模型规模和先前尺度保留策略的依赖等限制。

---

## 535. Toward Securing AI Agents Like Operating Systems

**arXiv ID:** 2605.14932 | [PDF](https://arxiv.org/pdf/2605.14932v1)

**作者:** Lukas Pirch `[一作]` (BIFOLD & TU Berlin), Konrad Rieck `[通讯]` (BIFOLD & TU Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将LLM驱动的AI代理与操作系统进行类比，本文系统性地审视了OpenClaw风格代理的安全性，构建统一架构，映射操作系统的防御机制，并对四个主流代理（OpenClaw、IronClaw、Nanobot、NemoClaw）进行案例攻击实验。

**💡 创新点**

创新点在于提出了“AI代理即操作系统”的结构框架，利用该框架将经典的OS安全技术（进程隔离、沙箱、权限分离、网络过滤、日志完整性等）映射到代理体系；同时通过实证案例展示了这些技术在代理中的适用性与不足，提供了改进安全的具体建议。

**🔧 技术方法**

使用的技术包括：操作系统安全概念与机制、eBPF监控、虚拟机隔离、LLM推理（qwen3.5-122b-a10b）、消息渠道（Matrix/Telegram）以及自定义恶意技能。实验环境在Debian 13.4上配置，利用eBPF捕获文件、网络、进程等系统调用。

**📊 数据集**

主要使用公开的OpenClaw风格代理代码库（OpenClaw、IronClaw、Nanobot、NemoClaw）以及所选LLM模型；未采用传统机器学习数据集，而是通过构造恶意技能与通信消息来触发攻击。

**📈 对比分析**

对每个代理分别在同一组攻击场景下执行多次实验，记录成功、失败或不适用的结果。实验显示，OpenClaw在所有攻击场景下均易受攻击；IronClaw仅易受7项攻击；NemoClaw和Nanobot表现略优。实验未给出传统意义上的性能指标，而是以攻击成功率与漏洞覆盖度进行比较。

**⚠️ 局限性**

局限性包括：仅测试默认配置；只使用单一消息通道；LLM的非确定性导致实验结果可能变动；攻击场景不涵盖所有可能的链路；对不同LLM模型的评估有限；未考虑高级安全策略（如完整的沙箱隔离）对攻击的进一步影响。

---

## 536. String Solving with Stabilization and Transducers (Technical Report)

**arXiv ID:** 2605.14872 | [PDF](https://arxiv.org/pdf/2605.14872v1)

**作者:** David Chocholatý `[一作]` (Brno University of Technology), Michal Šedý `[通讯]` (Aalborg University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对链自由（chain‑free）字符串约束片段扩展了基于稳定化（stabilization）的自动机求解方法，并引入了对有限状态转导（finite‑state transducer）约束的高效处理。

**💡 创新点**

创新点在于：①将稳定化与长度约束相结合，避免昂贵的串联消除（concatenation elimination）；②提出多种启发式（如同态转导简化、文字约束转导简化、交集空判定）显著提升性能；③支持非功能转导的否定与不等式，进一步拓宽了可判定的约束范围；④实现了对链自由公式的完整决策程序。

**🔧 技术方法**

使用的技术包括：
- 稳定化算法（stabilization）
- 转导约束的“ noodlification ”分割与稳定化
- 多筒转导与 Parikh 图求长度影像
- 句子推导树（work‑list）与前向/后向长度传播
- 复杂度有限的包含图（inclusion graph）
- 启发式简化（同态、文字、交集空判定）

**📊 数据集**

实验使用了 SMT‑LIB 2025 发布的四组基准集：
- PCP（含嵌套操作）
- 逆转录过程（生物信息学）
- 7 种字符串关系（toLower、toUpper 等）
- 实际网页应用程序（含正则、词方程、操作）

**📈 对比分析**

与现有顶尖求解器（Z3Str3, CVC4/5, QF_BV）和原始 Solvers 比较，使用 120 s 超时与 8 GB 内存；结果显示：
- 在所有基准集中，所提方法至少与最优求解器同等或更好；
- 在大多数基准集上，求解实例数至少 1.5‑2 倍；
- 运行时间平均快 1–2 个数量级；
- 对 PCP 组中非链自由公式时，通过交集空判定可显著恢复可解性。

**⚠️ 局限性**

局限性包括：
- 仅在链自由（直接链自由）片段内保证可判定，对非链自由公式需要额外启发式；
- 对嵌套转导深度非常大时，转导规模仍会爆炸，导致内存耗尽；
- 启发式（如交集空判定、同态简化）仅为经验性，缺乏理论复杂度上界；
- 目前不支持 SMT‑LIB 中未定义的否定功能转导或其它自定义转导；
- 与其他求解器的交叉使用（多模态求解）实现较为复杂，未完全整合。

---

## 537. Evo-Depth: A Lightweight Depth-Enhanced Vision-Language-Action Model

**arXiv ID:** 2605.14950 | [PDF](https://arxiv.org/pdf/2605.14950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 538. H-OmniStereo: Zero-Shot Omnidirectional Stereo Matching with Heading-Aligned Normal Priors

**arXiv ID:** 2605.14963 | [PDF](https://arxiv.org/pdf/2605.14963v1)

**作者:** Chenxing Jiang `[一作]` (Hong Kong University of Science and Technology), Shaojie Shen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 17676 | [OpenAlex ID](https://openalex.org/A5001947944)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 H-OmniStereo，一个零样本全方位立体匹配框架，利用顶-底全景图像实现完整的 360° 3D 感知。

**💡 创新点**

创新点包括：1）构建了 280 万对、超 1K 场景、80 万 3D 资产的高质量合成顶-底等距投影立体数据集；2）提出了以航向对齐坐标系为基础的等距单目法向估计器，消除航向旋转对法向的影响，提高跨视角一致性；3）将该法向先验与迭代立体匹配结合，并加入不确定性估计，显著提升零样本泛化能力。

**🔧 技术方法**

技术手段：NVIDIA Isaac Sim 生成合成数据；ViT + 位置编码 + 交叉注意力的球面感知法向网络；基于 RAFT / 3D 卷积的迭代立体匹配；ConvGRU 与不确定性模块的联合训练；混合多尺度损失和负对数似然。

**📊 数据集**

主要使用的数据集：自研的 2.8M 合成顶-底等距立体数据集；Structured3D、TartanAir V2、SpatialGen 用于法向预训练；在 Structured3D、3D60、3D60_Warp、MVS_GI 等公开全景立体数据集上进行零样本评估；并在 GRUtopia 真实 360° 摄像机轨迹上验证视觉里程计。

**📈 对比分析**

与 360SD‑Net、MODE、360‑IGEV‑Stereo、DFI‑OmniStereo 等方法在 3D60、3D60_Warp、MVS_GI 三个未见数据集上比较，H‑OmniStereo 的 MAE、RMSE 低至 0.103/0.401、0.127/0.365，显著优于对手；在结构化 3D 法向评估中亦取得最低角误差；在视觉里程计中相对位移/旋转误差分别降至 1.24%/0.13°，提升约 20%。

**⚠️ 局限性**

局限性：对相机标定极其敏感；合成数据与真实场景差距仍有提升空间；当前方法仍需依赖全景到立体图像的预处理（如裁剪、归一化），未来计划加入直接光流估计和更鲁棒的标定技术。

---

## 539. Efficient Online Conformal Selection with Limited Feedback

**arXiv ID:** 2605.14953 | [PDF](https://arxiv.org/pdf/2605.14953v1)

**作者:** Sreenivas Gollapudi `[一作]` (Google), Ali Sinop `[通讯]` (Google)

**通讯引用:** 679 | [OpenAlex ID](https://openalex.org/A5086919065)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文解决了顺序符合选择的问题，提出了一种方法，使得代理在有限反馈下选择最小子集以确保以预定概率ϕ识别至少一个“成功”。

**💡 创新点**

创新点在于提出了一种简单的自适应符合推断（ACI）更新规则，该规则在对抗性有效性和随机效率方面提供了理论保证，尤其是在有限的“赌博”反馈下。

**🔧 技术方法**

使用了自适应符合推断（ACI）更新规则和Lyapunov函数的分析框架。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在多种复杂设置下的理论结果。

**📈 对比分析**

与现有方法相比，本文的方法在处理更复杂的设置时需要显著更少的反馈，并且在效率上表现出亚线性后悔，具体的性能比较在文中有详细讨论。

**⚠️ 局限性**

限制在于该方法在某些情况下可能需要更多的历史上下文信息来学习，并且在处理NP难的组合选择时可能面临计算复杂性。

---

## 540. A CUBS-Compatible Ultrasound Morphology and Uncertainty-Aware Baseline for Carotid Intima-Media Segmentation and Preliminary Risk Prediction

**arXiv ID:** 2605.14949 | [PDF](https://arxiv.org/pdf/2605.14949v1)

**作者:** Aueaphum Aueawatthanaphisut `[一作]` (Thammasat University), Aueaphum Aueawatthanaphisut `[通讯]` (Thammasat University)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5118181895)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并实现了AtheroFlow‑XNet，一个多模态、物理引导且不确定性感知的模型，用于基于B‑mode超声图像进行颈动脉内膜/中膜分割和风险预测。

**💡 创新点**

创新点在于将超声形态、临床变量与可扩展的血流与CFD壁剪切生物标志结合在统一网络中，并通过Monte Carlo dropout提供置信度图。

**🔧 技术方法**

采用残差CNN编码器、Transformer瓶颈、解码器、以及多任务损失与不确定性估计的框架，可插拔Doppler/CFD模块。

**📊 数据集**

使用公开的Carotid Ultrasound Boundary Study（CUBS）数据集，包括B‑mode图像、LI‑MA边界、校准因子和临床特征。

**📈 对比分析**

在CUBS测试集上，分割Dice为0.7930，风险AUC为0.6910；相较于仅基于CIMT或传统指标，模型表现可行但仍中等。

**⚠️ 局限性**

局限在于缺乏Doppler波形和CFD壁剪切数据，风险标签不完全代表真实病变，且未在外部数据上验证。

---

## 541. Behavioral Data-Driven Optimal Trajectory Generation for Rotary Cranes

**arXiv ID:** 2605.14944 | [PDF](https://arxiv.org/pdf/2605.14944v1)

**作者:** Iskandar Khemakhem `[一作]` (University of Stuttgart), Abdallah Farrage `[通讯]` (Assiut University)

**通讯引用:** 145 | [OpenAlex ID](https://openalex.org/A5011359174)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于行为系统理论的数据驱动方法，用于生成自动化塔式起重机的开环摆动抑制轨迹。

**💡 创新点**

创新点在于直接利用Willems基本引理从测量数据构建非参数动力学模型，并在凸优化框架下生成满足时间、能耗和摆动约束的最优轨迹。

**🔧 技术方法**

核心技术包括行为系统理论、Willems基本引理、Hankel矩阵插值、ℓ1稀疏正则化以及MATLAB/CPLEX求解器。

**📊 数据集**

使用实验塔式起重机约50分钟的输入输出数据构建Hankel矩阵，并在仿真中验证其对非线性动力学的捕捉。

**📈 对比分析**

与传统基于模型的时间最优轨迹生成方法相比，数据驱动方案在摆动幅度、跟踪误差和到达时间上分别提升了约35%、43%和50%。

**⚠️ 局限性**

局限性主要包括对输入设计的高度依赖、需手动调节多重超参数，以及对噪声、风扰等外部干扰的鲁棒性尚未验证。

---

## 542. Octopus: History-Free Gradient Orthogonalization for Continual Learning in Multimodal Large Language Models

**arXiv ID:** 2605.14938 | [PDF](https://arxiv.org/pdf/2605.14938v1)

**作者:** Yuehao Liu `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 33776 | [OpenAlex ID](https://openalex.org/A5025545087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于无历史梯度正交化（HiFGO）的两阶段连续学习框架，用于多模态大语言模型的任务序列学习。

**💡 创新点**

创新点在于：①在无历史任务数据的前提下，仅利用当前任务数据和之前任务的参数梯度实现梯度空间正交；②通过两阶段微调（先无约束学习，再施加正交与范数约束）实现显著的可塑性与稳定性平衡。

**🔧 技术方法**

使用技术包括：LoRA 参数高效微调、HiFGO 约束、两阶段微调策略、梯度正交与 L2 范数正则化。

**📊 数据集**

使用数据集：UCIT（针对多模态指令驱动场景的连续学习基准）。

**📈 对比分析**

与现有的架构、正则化和回放基线相比，HiFGO 在 Avg 方面提升 2.14%，在 Last 方面提升 6.82%，并超过了传统回放方法，显示出更强的灾难性遗忘抑制与正向迁移能力。

**⚠️ 局限性**

局限性包括：对任务数量的上限（过多任务会导致性能下降）以及在任务相似度很高的情况下效果可能不佳。

---

## 543. Multi-scale Coarse-to-fine Modeling for Test-time Human Motion Control

**arXiv ID:** 2605.14935 | [PDF](https://arxiv.org/pdf/2605.14935v1)

**作者:** Nhat Le `[一作]` (University of Western Australia), Ajmal Mian `[通讯]` (University of Western Australia)

**通讯引用:** 20973 | [OpenAlex ID](https://openalex.org/A5089986388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出MSCoT模型，采用多尺度离散编码和自回归Transformer，实现测试时的快速、可控人类运动生成。

**💡 创新点**

创新点在于多尺度粗细级别并行生成与单步多尺度Token引导，利用贝叶斯一阶近似实现离散Token的可控后验更新；同时加入Token Refiner实现连续残差细化并支持无额外训练的控制目标。

**🔧 技术方法**

技术包括多尺度VQ‑VAE离散化、基于CLIP的文本条件、多尺度自回归Transformer、贝叶斯多尺度Token引导、梯度一阶近似、连续Token细化与测试时优化、欧氏归一化代码本。

**📊 数据集**

使用公开数据集HumanML3D、KIT‑ML进行文本到运动生成，联合使用多种控制任务（关节控制、障碍物回避、人景交互）进行评测。

**📈 对比分析**

与MDM、PriorMDM、GMD、OmniControl、TLControl、MaskControl、MotionLCM等基线比较，在关节控制任务中MSCoT实现10×速率提升、33×超越OmniControl，FID降48%、R‑Precision提升、控制误差几乎为0；在场景交互与标准文本生成任务亦保持或超越SOTA。

**⚠️ 局限性**

局限性包括：使用均值场分解和一阶近似导致约束；离散代码本容量受限需细化；代码本尺寸和欧氏归一化对性能影响；对极长序列的多尺度Transformer注意力开销大；对身体结构依赖的自编码器尚未引入。

---

## 544. Chain-of-Procedure: Hierarchical Visual-Language Reasoning for Procedural QA

**arXiv ID:** 2605.14928 | [PDF](https://arxiv.org/pdf/2605.14928v1)

**作者:** Guanhua Chen `[一作]` (University of Macau), Derek F. Wong `[通讯]` (University of Macau)

**通讯引用:** 3818 | [OpenAlex ID](https://openalex.org/A5101468579)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ProcedureVQA基准与Chain-of-Procedure（CoP）框架，用于评估并提升视觉程序推理能力。

**💡 创新点**

创新点包括：①构建专门针对视觉程序问答的多模态基准ProcedureVQA；②设计三阶段CoP框架——检索程序、语义拆分匹配视觉细粒度、预测下一步——显著缓解视觉与文本粒度不匹配问题。

**🔧 技术方法**

采用跨模态检索、层次化语义拆分以及大型视觉语言模型（VLM）进行推理与生成的技术组合。

**📊 数据集**

使用的数据集为从WikiHow CoMM重构的ProcedureVQA，涵盖汽车、电脑、手工艺、运动、工作等五个领域，共4,783样本；并在RecipeQA上进行验证。

**📈 对比分析**

与现有VLM（如Qwen2.5-VL、GPT-4o、Gemini、Claude）对比，CoP在ProcedureVQA上实现BERTScore提升1.7–17.9%、LLM-Score提升2.2–13%，整体准确率超过65%，明显优于基线。

**⚠️ 局限性**

限制在于多阶段框架导致token消耗约为基线的两倍，且易受检索错误影响，对步骤拆分质量高度依赖。

---

## 545. From Sycophantic Consensus to Pluralistic Repair: Why AI Alignment Must Surface Disagreement

**arXiv ID:** 2605.14912 | [PDF](https://arxiv.org/pdf/2605.14912v1)

**作者:** Varad Vishwarupe `[一作]` (University of Oxford), Marina Jirotka `[通讯]` (University of Oxford)

**通讯引用:** 4810 | [OpenAlex ID](https://openalex.org/A5023741875)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Pluralistic Repair Score (PRS)，通过评估模型在争议价值对话中的 scoping、signalling 和 repair 行为，并在 Claude Sonnet 4.5 与 GPT‑4o 的两轮压力情境实验中验证其有效性。

**💡 创新点**

创新点在于将多元化对齐从仅聚合输出的视角扩展到交互层面，定义并量化“principled repair”，揭示 RLHF 模型的 sycophantic 共识缺陷。

**🔧 技术方法**

采用 Grice 合作原则与 Wittgenstein 语言游戏理论为理论基础，设计结构化评估 Rubric 并用人工标注评估交互转移；同时利用 RLHF、对话生成和评估技术。

**📊 数据集**

构造了 198 条覆盖健康、金融、民事、亲密、专业和争议实证六个领域的两轮对话样本，并在 100 条子集上验证 GPT‑4o。

**📈 对比分析**

通过对比聚合层指标（Overton/Distributional）与 PRS，发现两模型的 agreement‑shift 率高而 PRS 低，Claude Sonnet 4.5 的平均 PRS 为 0.21，GPT‑4o 为 0.14，展示了显著的 agreement‑repair gap。

**⚠️ 局限性**

局限性包括样本量小、仅评估两模型、评判者主观性、缺乏自动化实现、meta‑pluralism 未解决，且未覆盖不同用户视角与更大规模部署。

---

## 546. Chrono-Gymnasium: An Open-Source, Gymnasium-Compatible Distributed Simulation Framework

**arXiv ID:** 2605.14911 | [PDF](https://arxiv.org/pdf/2605.14911v1)

**作者:** Bocheng Zou `[一作]` (University of Wisconsin-Madison), Dan Negrut `[通讯]` (University of Wisconsin-Madison)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了Chrono-Gymnasium，一个基于Ray的分布式计算框架，将Project Chrono高保真物理模拟封装为Gymnasium接口，支持在大规模集群上并行执行；

**💡 创新点**

将高保真多物理动力学模拟与现代机器学习训练/优化流水线无缝集成，并提供统一的初始化、重置、控制、步进、观测、奖励和终止接口，实现了从单机到集群的透明迁移；

**🔧 技术方法**

使用Python/C++结合Project Chrono物理引擎、Ray分布式框架以及Gymnasium接口；在RL实验中采用PPO，Bayesian Optimization实验使用Optuna；

**📊 数据集**

在案例实验中使用的主要“数据集”为：①用于四足机器人速度跟踪的自定义关节与地形仿真数据；②用于行星着陆器优化的多体系统碰撞与能量吸收仿真数据；

**📈 对比分析**

通过与单机Chrono和Ray+多环境并行两种设置对比，结果显示在RL任务中并行环境数从1到16时，墙钟时间缩短约2倍，跟踪误差显著降低；在Bayesian Optimization任务中，多核/多GPU并行显著加速目标值收敛，CPU/ GPU占用分别下降；

**⚠️ 局限性**

受限于Ray的通信与同步开销，低并行规模时仍有性能瓶颈；框架尚未覆盖所有Chrono模块（如流体-结构耦合）的完整并行实现；需要进一步优化大规模异构计算的负载均衡与内存管理。

---

## 547. KGPFN: Unlocking the Potential of Knowledge Graph Foundation Model via In-Context Learning

**arXiv ID:** 2605.14907 | [PDF](https://arxiv.org/pdf/2605.14907v1)

**作者:** Yisen Gao `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10721 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种基于Prior‑Data Fitted Network的知识图谱基础模型KGPFN，利用推理时的局部邻域和全局关系上下文实现零样本或少样本知识图谱推理。

**💡 创新点**

将结构化本地邻域编码与关系特定的全局上下文聚合相结合，并在推理阶段通过PFN实现无需微调的即时上下文学习，显著提升跨图泛化能力。

**🔧 技术方法**

采用关系图消息传递获取关系表示，多层NBFNet编码本地邻域，负采样构建全局上下文，使用Prior‑Data Fitted Network实现特征注意力与样本注意力双层聚合，并用二元交叉熵与softmax损失共同训练。

**📊 数据集**

在57个知识图谱上评估，包含Transductive（16）、Inductive（18）和Full‑inductive（23）三种设置，预训练图为FB15K‑237、WN18RR和CodexMedium。

**📈 对比分析**

与ULTRA、KG‑ICL、TRIX、MOTIF等主流KG基础模型对比，KGPFN在所有57个KG上均取得最高平均Hits@10，并在Inductive和Full‑inductive场景下实现MRR领先，且无需微调即可完成零样本推理。

**⚠️ 局限性**

对负样本的依赖较强，需要足够负样本才能显著提升排名；局部上下文仅基于头实体的k‑hop邻域，可能忽略尾实体信息；模型在极大实体集合上的推理效率和可扩展性仍需进一步验证。

---

## 548. Denoising-GS: Gaussian Splatting with Spatial-aware Denoising

**arXiv ID:** 2605.14880 | [PDF](https://arxiv.org/pdf/2605.14880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 549. Text-Dependent Speaker Verification (TdSV) Challenge 2024: Team Naive System Report

**arXiv ID:** 2605.14896 | [PDF](https://arxiv.org/pdf/2605.14896v1)

**作者:** Amir Mohammad Rostami `[一作]` (Self-Organized and Independent Participants), Pourya Jafarzadeh `[通讯]` (Self-Organized and Independent Participants)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于多模型集成的文本依赖说话人验证系统，并在2024 TdSV 挑战赛中取得 MinDCF 0.0461、EER 1.3% 的优异成绩。

**💡 创新点**

创新点在于将预训练的 ResNet‑TDNN、NeXt‑TDNN 与轻量化 EfficientNet‑A0 进行微调后集成，并结合 Wav2Vec‑2.0 语句分类器，实现说话人与语句双重验证。

**🔧 技术方法**

使用了深度卷积网络（TDNN/ResNet/EfficientNet）、Wav2Vec 2.0、S‑norm 正则化、数据增强、特征提取和分数融合等技术。

**📊 数据集**

使用的训练数据集包括 VoxCeleb1/2、LibriSpeech、Mozilla Common Voice Farsi 以及 DeepMine 语料。

**📈 对比分析**

采用 MinDCF 和 EER 作为评估指标，通过 DET 曲线比较不同性别和语言子集，整体表现为 MinDCF 0.0461、EER 1.3%。

**⚠️ 局限性**

存在的局限性包括性别与语言子集之间的性能差距、受限的硬件资源导致模型规模受限，以及缺乏更大规模的说话人语料来进一步提升泛化能力。

---

## 550. Your CLIP has 164 dimensions of noise: Exploring the embeddings covariance eigenspectrum of contrastively pretrained vision-language transformers

**arXiv ID:** 2605.14893 | [PDF](https://arxiv.org/pdf/2605.14893v1)

**作者:** Jakub Grzywaczewski `[一作]` (Warsaw University of Technology), Przemysław Biecek `[通讯]` (Warsaw University of Technology)

**通讯引用:** 6519 | [OpenAlex ID](https://openalex.org/A5049061860)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对对比预训练的视觉‑语言模型（VLM）嵌入空间进行谱分解，拆分出共享噪声子空间与多模态语义信号，并验证去除噪声维度对下游任务的影响。

**💡 创新点**

首次在VLM中系统识别并量化共享噪声维度的存在、子空间共性以及去噪后提升或保持性能的现象。

**🔧 技术方法**

使用协方差矩阵的特征值谱分解、均方余弦子空间角度（mSCSA）度量子空间重叠、投影矩阵去噪等方法。

**📊 数据集**

ImageNet‑1K训练集、LAION‑2B（文本与图像配对）以及部分 LAION‑2B‑en‑aesthetic 子集。

**📈 对比分析**

在 ImageNet 零样本分类（Top‑5 Accuracy）和 LAION‑2B 文本‑图像对齐（余弦相似度）两个基准上，对比原始、去噪与随机裁剪维度的表现，去噪既不降低准确率，且在文本‑图像对齐上常出现显著提升。

**⚠️ 局限性**

阈值选择仍为经验性，可能需针对不同模型/数据分布调整；实验聚焦 CLIP ViT‑L/14，未在更广泛的 VLM 体系上系统验证；理论解释尚不完整。

---

## 551. Unlocking Complex Visual Generation via Closed-Loop Verified Reasoning

**arXiv ID:** 2605.14876 | [PDF](https://arxiv.org/pdf/2605.14876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 552. Decision-Level Fusion for Robust Wearable Affect Recognition

**arXiv ID:** 2605.14878 | [PDF](https://arxiv.org/pdf/2605.14878v1)

**作者:** Lokesh Singh `[一作]` (University of Southampton), Sarvapali D. Ramchurn `[通讯]` (University of Southampton)

**通讯引用:** 7421 | [OpenAlex ID](https://openalex.org/A5065527041)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套三阶段的可穿戴情感识别管线：先通过 FBSE–EWT 对生理信号进行非平稳特征提取，然后为每个传感器训练单模态预测器，最后采用基于不确定性加权的决策级融合得到最终情感分类。

**💡 创新点**

创新点在于：1) 结合 Fourier Bessel 系列展开与 Empirical Wavelet Transform 的 FBSE–EWT 表征，能够保留瞬态频谱结构；2) 通过预测熵与模型 F1 分数的加权方式实现对传感器质量不确定性的自适应决策级融合，显著提升在噪声或缺失模态下的鲁棒性。

**🔧 技术方法**

所用技术包括 Fourier Bessel 系列展开、Empirical Wavelet Transform、MLP 预测器、熵加权的决策级融合策略，以及传统的窗口化与标签投票处理。

**📊 数据集**

使用的实验数据集为 WESAD，包含 15 位受试者、5 种传感器（ECG、EDA、BVP、EMG、加速度）以及三类情绪（基线、压力、娱乐）。

**📈 对比分析**

通过对比决策级融合与特征级融合在不同传感器团队规模下的性能，实验显示决策级融合在 84% 的情况下与特征级相当或更优，总体准确率约 93%，且团队规模越大准确率越高。

**⚠️ 局限性**

局限性包括：仅在三类情绪上验证，缺乏对更细粒度或多标签情绪的评估；实验仅限于 WESAD 数据集；未对时序建模进行深入探讨；需要在更多数据集和多模态（视频、音频等）上进一步验证。

---

## 553. Meschers: Geometry Processing of Impossible Objects

**arXiv ID:** 2605.14960 | [PDF](https://arxiv.org/pdf/2605.14960v1)

**作者:** Ana Dodik `[一作]`, Justin Solomon `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种名为 Mescher 的网格表示，用于在保持局部几何一致性的前提下表达不可能物体，并实现渲染、再照明、细分、拉普拉斯平滑、热扩散、几何距离查询以及逆渲染等完整的几何处理流程。

**💡 创新点**

创新点在于利用离散外微积分（DEC）构造只需局部可积的 1‑form 结构，避免了传统切割/弯曲方法的全局嵌入约束，同时通过深度排序图实现全局深度约束，从而使得不可能物体能够在不破坏几何处理管线的情况下被完整操作。

**🔧 技术方法**

技术手段包括 DEC 与 Hodge 分解、有限元梯度与拉普拉斯算子、热方法求距离、Sobolev 梯度下降、SoftRas 可微光栅化、PyTorch 与 NetworkX 等实现库。

**📊 数据集**

实验数据主要使用自制的可可能体模型（如 Penrose 三角、Penrose 圆盘等）以及公开的不可可能物体在线库，亦对比了传统的切割与弯曲嵌入方法。

**📈 对比分析**

与切割与弯曲表示相比，Mescher 在渲染、再照明、平滑、热扩散与几何距离计算等任务中均无伪影，且能够通过统一的离散算子实现一致的结果；实验中在 NVIDIA RTX 4090 上完成了多种操作，表现出良好的可扩展性与实时性。

**⚠️ 局限性**

局限性包括：只能处理可定向流形网格；需要人工或算法提供深度排序图；目前仅支持正交相机投影；缺乏阴影、透明等高级渲染效果；逆渲染阶段对光照与几何初始值敏感，且尚无专门的直接建模界面。

---

## 554. Static and Dynamic Strategies for Influencing Opinions in Social Networks

**arXiv ID:** 2605.14918 | [PDF](https://arxiv.org/pdf/2605.14918v1)

**作者:** Paolo Tarantino `[一作]` (Politecnico di Milano), Francesco Pierri `[通讯]` (Politecnico di Milano)

**通讯引用:** 2858 | [OpenAlex ID](https://openalex.org/A5013385420)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在LFR网络中引入少量固定或动态极端意见的顽固节点，评估不同中心性度量在Hegselmann–Krause意见动态模型下对网络平均意见的影响。

**💡 创新点**

提出动态调节顽固节点极端意见的策略，并与传统静态策略在多种中心性选择下进行对比，发现即使极少顽固节点也能驱动大部分网络趋向极端意见。

**🔧 技术方法**

使用Hegselmann–Krause有界信任动态模型、基于度、强度、PageRank、介数、k/ s‑核心、salience等中心性和随机选择，并设计逐步提高意见的动态算法。

**📊 数据集**

在Lancichinetti–Fortunato–Radicchi (LFR) 生成的加权社区网络（N=1000 或 2000）上进行实验。

**📈 对比分析**

通过 50 次随机初始化和 20 个网络实例进行实验，比较不同中心性与静态/动态策略在最终平均意见、接近极端意见的节点比例和时间演化上的表现，结果显示动态策略在大多数情况下显著优于静态，并且随机选择在动态策略下也能接近最佳效果。

**⚠️ 局限性**

仅考虑单一网络模型和单一意见动态机制，且仅关注极端极化目标；对真实网络的泛化有限，未考虑不同目标或网络结构的多样性。

---

## 555. TILBench: A Systematic Benchmark for Tabular Imbalanced Learning Across Data Regimes

**arXiv ID:** 2605.14915 | [PDF](https://arxiv.org/pdf/2605.14915v1)

**作者:** Ruizhe Liu `[一作]` (Soochow University), Jiaqi Luo `[通讯]` (Soochow University)

**通讯引用:** 927 | [OpenAlex ID](https://openalex.org/A5039479263)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了大规模的表格不平衡学习基准（TILBench），对40+代表性算法在57个公开表格数据集上进行统一评估。

**💡 创新点**

系统化地从预测性能、数据特征敏感性和计算可扩展性三维度进行评估，并给出基于数据特征的实用方法推荐。

**🔧 技术方法**

基于XGBoost的GBDT模型，涵盖数据级、算法级和集成级的不平衡处理技术，使用Optuna进行调参，采用5次随机种子复现。

**📊 数据集**

使用57个公开表格数据集（OpenML、imbalanced-learn等），涵盖二分类/多分类、样本量、特征维度、严重不平衡比例与缺失值情况。

**📈 对比分析**

通过全局排名、家族级平均、按样本量/特征维度/不平衡程度/缺失值等子组以及训练时间比较，结果显示算法级方法总体最稳定，集成方法在小样本二分类中表现突出，数据级方法在多类或极端不平衡时具竞争力；计算上算法级方法最具可扩展性。

**⚠️ 局限性**

局限性包括未覆盖深度学习/神经网络方法，极端高维或大类不平衡场景的数据量有限，实验仅在单机环境完成，缺乏分布式训练评估。

---

## 556. Representative Attention For Vision Transformers

**arXiv ID:** 2605.14913 | [PDF](https://arxiv.org/pdf/2605.14913v1)

**作者:** Yuntong Li `[一作]` (Tianjin University), Xiaojie Guo `[通讯]` (Tianjin University)

**通讯引用:** 14676 | [OpenAlex ID](https://openalex.org/A5090356888)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于代表性注意力的线性全局注意力模块RPAttention，通过在表示空间聚合和分配视觉特征，实现了高效的全局上下文建模。

**💡 创新点**

创新点在于将中介代理token从固定空间布局转为基于表示相似度的学习代理，从而实现内容感知的压缩与分发，保持全局感受野且复杂度线性。

**🔧 技术方法**

采用了Gather-Interact-Distribute流程、软归一化的竞争分配、轻量自注意力、以及深度可分离卷积的局部旁路等技术。

**📊 数据集**

使用ImageNet-1K进行分类，COCO 2017进行目标检测与实例分割，ADE20K进行语义分割进行评估。

**📈 对比分析**

在与原始ViT、PVT、Swin等基准以及FLatten、Agent、PolaFormer等线性注意力方法的对比中，RPAttention在保持相同计算量的前提下提升了Top‑1精度、检测AP和分割mIoU，证明了更好的性能。

**⚠️ 局限性**

局限性在于对局部纹理相似的误聚合，导致跨对象的注意力混淆，尤其在相似纹理的背景与目标之间可能产生错误关联。

---

## 557. SteerSeg: Attention Steering for Reasoning Video Segmentation

**arXiv ID:** 2605.14908 | [PDF](https://arxiv.org/pdf/2605.14908v1)

**作者:** Ali Cheraghian `[一作]` (Macquarie University), Lars Petersson `[通讯]` (Csiro Data61)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在冻结的大型视觉语言模型（LVLM）输入层插入可学习的软提示并结合链式推理生成的属性，引导注意力聚焦到目标对象上，从而在不改变模型权重的前提下生成高质量的空间注意力图，进一步用作 SAM2 分割模型的点提示，完成视频对象分割任务。

**💡 创新点**

创新点在于：① 将输入级软提示直接作用于 LVLM 的自注意力机制，实现对注意力分布的“steering”，② 采用单步链式推理（CoT）提取区别属性辅助去除视觉歧义，从而解决注意力失配与模糊定位问题；该方案兼具轻量化与高精度。

**🔧 技术方法**

使用技术包括：输入级软提示、CoT 属性生成、注意力回放（rollout）获取响应标记注意力、基于 SAM2 的点提示分割、关联评分（Pearson correlation）进行 tracklet 选择以及关键帧采样与非极大抑制等。

**📊 数据集**

主要在 Ref-YouTube-VOS 上训练软提示，评估跨数据集包括 Ref-DAVIS、ReasonVOS、ReVOS、MeViS 等四个不同 benchmark，且实验使用多种 LVLM backbones（LLaVA-OV, InternVL3, Qwen2VL, Qwen2.5VL）。

**📈 对比分析**

与完全训练或冻结 LVLM + 分割基线相比，SteerSeg 在所有 LVLM 后端均取得显著提升，尤其在 ReasonVOS、ReVOS 的推理型查询上提升 2–3 分，Ref-DAVIS 上 J&F 提升至约81%，MeViS 上也超过 70% 的 J&F；在部分全量训练方法（如 VRS‑HQ、VISA）中表现相当或更优。

**⚠️ 局限性**

局限性：① 仅对采样的关键帧使用注意力，可能遗漏短暂出现的目标；② 融合权重 α 的选择需经验调节，可能随数据集或 LVLM 变化而不同。

---

## 558. MemLens: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models

**arXiv ID:** 2605.14906 | [PDF](https://arxiv.org/pdf/2605.14906v1)

**作者:** Xiyu Ren `[一作]` (HKUST), Simon See `[通讯]` (NVIDIA)

**通讯引用:** 3086 | [OpenAlex ID](https://openalex.org/A5077539496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MEMLENS基准，用以评估多模态长对话记忆能力。

**💡 创新点**

创新点在于系统对长上下文与记忆增强两大范式的跨模态、跨长度统一对比，并设计了五大记忆能力的细粒度测试。

**🔧 技术方法**

采用四阶段生成管线（多模态会话仿真、问题构造、证据会话包装、对话历史组装），结合多模态LLM与检索增强代理进行评估。

**📊 数据集**

使用公开网络图像检索生成的4,695张图片，构成789条问题，并配合多模态对话会话。

**📈 对比分析**

与27款大型视听语言模型和7款记忆增强代理在32K–256K上下文长度上对比，结果显示长上下文LLM在短长度表现相近，但随长度增长显著下降；记忆代理长度稳定但失去细粒度视觉信息，整体性能低于最强LLM。

**⚠️ 局限性**

局限性包括对视觉存储压缩的依赖导致信息丢失、记忆代理在拒答时易失误、以及两种范式在不同维度的互补性未被充分利用。

---

## 559. Beyond Individual Intelligence: Surveying Collaboration, Failure Attribution, and Self-Evolution in LLM-based Multi-Agent Systems

**arXiv ID:** 2605.14892 | [PDF](https://arxiv.org/pdf/2605.14892v1)

**作者:** Shihao Qi `[一作]` (MOE KLINNS Lab), Tongliang Liu `[通讯]` (University of Sydney)

**通讯引用:** 13161 | [OpenAlex ID](https://openalex.org/A5065250332)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化了LLM驱动多智能体系统（LLM-based multi-agent systems）的完整运营生命周期，从单体智能、协作组织、失败归因到自我演化四个阶段，并通过LIFE视角将这些阶段因果关联。

**💡 创新点**

提出LIFE四阶段框架（Individual Intelligence → Multi-Agent Collaboration → Failure Attribution → System Self‑Evolution），构建跨阶段的闭环“归因–演化”机制；对各阶段的关键技术、方法和评估基准做了细致的分类与对比；同时创建了公开的结构化知识库和可视化工具，供后续研究借鉴。

**🔧 技术方法**

主要技术包括：① 模块化思维框架（Reasoning、Memory、Planning、Tool‑Use）与多模态检索/增强技术；② 归因方法（因果链分析、知识图谱追踪、工具验证等）；③ 结构化自我演化（角色动态分配、组织层级学习、协议自适应）；④ 评测方法（多维度任务、子目标完成度、工具调用精确度）与可视化。

**📊 数据集**

本综述不使用单一数据集进行实验，而是综合引用了大量公开基准：AgentBench、GAIA、MINT、AgentBoard、SWE‑bench、ToolBench、WebArena、OSWorld、LongMemEval 等，用于讨论和对比各方法在不同能力维度的表现。

**📈 对比分析**

对比方法：文中汇总了各类技术的优劣、适用场景与性能指标（如成功率、精确匹配、子目标完成度等），并在表格中呈现最新实验结果。虽然缺乏统一统一实验平台，但通过对已有基准的汇总与统计，展示了不同技术在任务难度、资源消耗、鲁棒性等方面的相对表现。

**⚠️ 局限性**

局限性：① 主要聚焦LLM为核心的多智能体，未覆盖传统规则或强化学习独立式多智能体系统；② 归因与自我演化的闭环尚未在大规模真实部署中验证；③ 综述性质导致缺少统一实验对比与细粒度性能度量；④ 对安全、伦理与可解释性等交叉问题讨论不够充分。

---

## 560. Hierarchical Image Tokenization for Multi-Scale Image Super Resolution

**arXiv ID:** 2605.14891 | [PDF](https://arxiv.org/pdf/2605.14891v1)

**作者:** Isma Hadji `[一作]` (Samsung AI Center), Georgios Tzimiropoulos `[通讯]` (Samsung AI Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多尺度图像超分辨率方法，使用分层图像标记化与直接偏好优化的VAR模型，能够一次前向推理得到不同尺度的输出。

**💡 创新点**

引入分层图像标记化（HIT）让残差量化在不同尺度下可解码，并结合DPO正则化鼓励模型输出高分辨率图像，从而在保持语义一致性的同时实现多尺度超分辨率。

**🔧 技术方法**

视觉自回归（VAR）模型、残差量化（RQ‑VAE）、分层标记化（HIT）、直接偏好优化（DPO）、GPT‑2式Transformer等技术。

**📊 数据集**

使用DIV2K、DIV8K、Flickr2k、OST、FFHQ等标准ISR训练集，并通过Real‑ESRGAN降采样生成LR‑HR对；测试采用DIV2K验证集、RealSR、DRealSR等数据集。

**📈 对比分析**

在PSNR、SSIM、LPIPS、FID、MUSIQ等指标上与GAN、扩散模型以及VARSR等方法对比，310M参数模型在多尺度任务上优于1B参数VARSR，且比扩散模型更轻量，单前向推理即可获得不同尺度结果。

**⚠️ 局限性**

分层标记化将步骤分配到不同尺度，导致高分辨率的重建受限，若想提升高尺度质量需增加总步骤数或计算量；此外对不同尺度语义一致性的依赖也增加了设计复杂度。

---

## 561. Tokenizer Fertility and Zero-Shot Performance of Foundation Models on Ukrainian Legal Text: A Comparative Study

**arXiv ID:** 2605.14890 | [PDF](https://arxiv.org/pdf/2605.14890v1)

**作者:** Volodymyr Ovcharov `[一作]` `[通讯]` (LEX AI Platform), Volodymyr Ovcharov (LEX AI Platform)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对乌克兰法律文本的七种基础模型进行系统评估，比较其分词效率和零/少样本任务表现。

**💡 创新点**

首次量化乌克兰语言分词肥度差异及其对API成本影响，揭示少样本提示在形态学复杂语言中的性能退化现象。

**🔧 技术方法**

利用AWS Bedrock API对Llama、Mistral、Qwen、Nova等模型的Tokenizer、零样本与少样本推理。

**📊 数据集**

使用EDRSR（乌克兰统一法院决定登记处）抽取的273份已验证法院判决。

**📈 对比分析**

采用Token Fertility、分类准确率、命名实体F1和成本-质量复合指标，对比显示Nemotron Super 3在所有任务上获得最高综合分（83.1），而Llama 4 Maverick以最低成本实现近乎最佳准确率；少样本提示普遍导致显著降效。

**⚠️ 局限性**

实验规模有限（仅300份判决），数据偏倚（类别不平衡）、仅通过API评测缺乏对模型内部机制的可视化，且未对模型进行微调或尝试更多提示变体。

---

## 562. BiFedKD: Bidirectional Federated Knowledge Distillation Framework for Non-IID and Long-Tailed ECG Monitoring

**arXiv ID:** 2605.14886 | [PDF](https://arxiv.org/pdf/2605.14886v1)

**作者:** Zixuan Shu `[一作]`, Hen-Wei Huang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了 BiFedKD（双向联邦知识蒸馏）框架，解决 IoMT ECG 监测中非 IID、长尾分布下的知识传递与模型协同。

**💡 创新点**

创新点在于：1）采用服务器端教师模型进行蒸馏，过滤并聚合客户端 logits，生成更稳定的全局软目标；2）引入温度缩放和双向蒸馏管线，减少大类偏差；3）在资源受限场景下实现通信与计算成本显著降低。

**🔧 技术方法**

使用联邦知识蒸馏、温度缩放、软目标对齐、服务器端教师模型蒸馏等技术，并通过 1D-CNN 与 Transformer 结构实现模型。

**📊 数据集**

使用 MIT-BIH Arrhythmia 数据集（48 名患者、48,000+ 心跳样本），其中包含非 IID、长尾标签分布，并采用 1,000 条公共代理数据。

**📈 对比分析**

与 FedMD、FedAvg 对比；BiFedKD 在 50 轮通信中准确率提升 3.52%、Macro‑F1 提升 9.93%；实现相同 Macro‑F1 时通信开销减少 40%、计算成本降低 71.7%，收敛速度更快、鲁棒性更强。

**⚠️ 局限性**

局限性包括：需预留公共代理数据；服务器端教师模型的设计与规模对性能与成本有较大影响；在极端长尾或极端非 IID 情况下，教师模型仍可能难以完全平衡各类标签。

---

## 563. LPH-VTON: Resolving the Structure-Texture Dilemma of Virtual Try-On via Latent Process Handover

**arXiv ID:** 2605.14874 | [PDF](https://arxiv.org/pdf/2605.14874v1)

**作者:** Yixin Liu `[一作]` (Shanghai Jiao Tong University), Guangtao Xue `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3180 | [OpenAlex ID](https://openalex.org/A5101490654)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Latent Process Handover（LPH）的虚拟试衣方法，通过在扩散过程的不同阶段分别使用结构偏置模型和纹理偏置模型来解决结构-纹理权衡问题。

**💡 创新点**

核心创新包括：①将扩散轨迹分为两阶段并实现跨模型的无缝迁移；②设计轻量级Latent Adapter实现不同模型潜空间的对齐；③采用Trajectory Extension恢复纹理模型的生成自由度。

**🔧 技术方法**

使用扩散模型（Stable Diffusion 1.5和SDXL）为基础，结合VAE编码/解码、跨注意力与空间拼接的混合架构、正弦位置编码与轻量级U‑Net Adapter。

**📊 数据集**

在两个公开高分辨率数据集上评估：VITON‑HD和DressCode，分辨率均为1024×768。

**📈 对比分析**

与多种SOTA基线（CatVTON、IDM‑VTON、GP‑VTON、StableVITON、LaDI‑VTON、DCI‑VTON）对比，LPH在SSIM、LPIPS、FID、KID等指标上取得Pareto最优平衡，显著提升纹理细节同时保持结构一致性。

**⚠️ 局限性**

主要限制为两阶段推理导致的推理时延与显存占用增加；手动设定的迁移点仍缺乏自适应机制，未来可通过模型蒸馏和动态路由进一步优化。

---

## 564. Not All Symbols Are Equal: Importance-Aware Constellation Design for Semantic Communication

**arXiv ID:** 2605.14940 | [PDF](https://arxiv.org/pdf/2605.14940v1)

**作者:** Albert Shaju `[一作]` (Worcester Polytechnic Institute), Mayukh Roy Chowdhury `[通讯]` (Nokia Bell Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种联合语义与物理层的通信框架，利用 VQ‑VAE、SCI 指标、DRL 率控制器和自学习的 M‑QAM 星座图，实现任务关键语义符号的动态选择与优先保护。

**💡 创新点**

创新点在于将语义重要性直接嵌入星座分配，通过 SCI 加权的星座损失和语义符号易损性指标，证明 Gray 码星座在非均匀语义场景下严格次优，并引入 SSV 与 SPP 评估语义可靠性。

**🔧 技术方法**

采用 VQ‑VAE、SCI MLP、DQN 率控制、SCI‑加权星座损失、Sionna 物理层仿真、KL‑损失与精度评估等技术。

**📊 数据集**

在 MNIST、Fashion‑MNIST 与 Free Spoken Digit Dataset（FSDD）三大跨模态数据集上进行实验。

**📈 对比分析**

与标准 Gray‑编码 M‑QAM 对比，实验表明语义 M‑QAM 在所有调制阶数下均提升约 40% 语义质量，符号压缩率提升 20~15×，且在高 SNR 下平均 BER 较高但语义错误率显著降低。

**⚠️ 局限性**

局限性包括对源分布变化需重新训练、对高维大规模任务仍需验证，以及星座学习对硬件实现的复杂度。

---

## 565. A Mutual Information Lower Bound for Multimodal Regression Active Learning

**arXiv ID:** 2605.14917 | [PDF](https://arxiv.org/pdf/2605.14917v1)

**作者:** Leonardo Ferreira Guilhoto `[一作]` (University of Pennsylvania), Paris Perdikaris `[通讯]` (University of Pennsylvania)

**通讯引用:** 37301 | [OpenAlex ID](https://openalex.org/A5002562845)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了双索引框架和MI-LB方法，用以在多模态连续回归的主动学习中显式区分并利用偏置不确定性

**💡 创新点**

创新点在于给出以互信息为目标的可计算下界MI-LB，并证明其在大样本下收敛至真实偏置信息，且专门针对多模态连续输出

**🔧 技术方法**

采用了混合高斯网络（Mixture Density Network）与信息论熵上界/下界的组合，构造闭式MI-LB公式

**📊 数据集**

在三种人工生成的多模态基准（含双井系统、条件多模态、相竞争）上进行实验，使用8个MDN集成与50,000个未标注样本池

**📈 对比分析**

与随机、方差、BAIT、Core-Set等基线比较，MI-LB在所有基准上均匹配或优于基线，尤其在模态信息隐藏在输出空间的双井基准中实现显著提升

**⚠️ 局限性**

局限在于需显式可解析的分布（如MDN）才能计算熵下界，熵估计的紧密度受限，且在非显式生成模型或模型未能完全拟合真实噪声时效果可能下降

---

## 566. Critic-Driven Voronoi-Quantization for Distilling Deep RL Policies to Explainable Models

**arXiv ID:** 2605.14897 | [PDF](https://arxiv.org/pdf/2605.14897v1)

**作者:** Senne Deproost `[一作]` (Vrije Universiteit Brussel), Ann Nowé `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 8498 | [OpenAlex ID](https://openalex.org/A5064553018)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于价值网络的Voronoi量化方法，将深度强化学习策略分解为一组线性子策略，实现可解释的模型蒸馏。

**💡 创新点**

创新点在于利用原策略的价值函数指导子策略分区，动态增设Voronoi代码点并结合k-means自适应选择子策略数，从而在保持性能的同时显著减少模型复杂度。

**🔧 技术方法**

技术包括深度强化学习（TD3）、线性回归子策略、Voronoi量化、k-means聚类、梯度下降训练以及利用价值网络评估子策略性能。

**📊 数据集**

使用了四个经典连续控制基准：SimpleGoal、MountainCarContinuous、LunarLanderContinuous 和 BipedalWalker，生成 100 条 episode 的状态-动作样本。

**📈 对比分析**

与原始 TD3 策略以及随机代码点的 VSP-random 进行对比，实验显示 VSP-critic 在平均回报上与 TD3 相近，仅用约三分之一的子策略数。

**⚠️ 局限性**

局限性包括仅适用于线性子策略、对代码点分布的解释性不足、超参数需人工调优、未验证对更复杂模型或离散环境的通用性。

---

## 567. SurgicalMamba: Dual-Path SSD with State Regramming for Online Surgical Phase Recognition

**arXiv ID:** 2605.14889 | [PDF](https://arxiv.org/pdf/2605.14889v1)

**作者:** Sukju Oh `[一作]` (Dongguk University), Sukkyu Sun `[通讯]` (Dongguk University)

**通讯引用:** 204 | [OpenAlex ID](https://openalex.org/A5013223600)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了用于在线手术阶段识别的SurgicalMamba模型，能够在流式视频中以恒定的O(d)成本对每帧做出阶段预测

**💡 创新点**

三大创新：①双路径Mamba块将长时记忆与短时响应在递归状态层面分离；②强度调制时间步（λ）通过连续时间时间变换动态加速或减慢慢速路径的记忆衰减；③每块状态再格（state regramming）通过Cayley旋转在每个时间块边界对隐藏状态进行内容相关的正交变换，提升跨通道信息融合

**🔧 技术方法**

基于Mamba2的结构化状态空间模型（SSD），结合深度可学习的λ网络、低秩Cayley旋转、卷积前馈网络和可学习的双路径投影

**📊 数据集**

七个公开手术视频数据集：Cholec80、M2CAI16、Cataract‑101、AutoLaparo、HeiChole、Heidelberg、GraSP

**📈 对比分析**

与目前最先进的多种方法（如DACAT、MTTR‑Net、Surgformer等）对比，SurgicalMamba在所有数据集上均取得最高或近似最高的准确率、精确率、召回率和Jaccard指数，并在Cholec80的严格评估中将Jaccard提升至82.7%，在AutoLaparo和Cataract‑101等小样本/多任务数据集上显著提升；同时以119fps的推理速度保持O(d)的每帧计算量

**⚠️ 局限性**

局限性：①状态再格的旋转虽然表现出阶段相关的结构，但缺乏对单通道方向级别的可解释性；②强度信号λ的监督仅基于已标注的阶段边界，迁移至其他事件检测任务时需要重新设计相应的过渡指示器

---

## 568. PROCESS-2: A Benchmark Speech Corpus for Early Cognitive Impairment Detection

**arXiv ID:** 2605.14888 | [PDF](https://arxiv.org/pdf/2605.14888v1)

**作者:** Madhurananda Pahar `[一作]` (University of Sheffield), Heidi Christensen `[通讯]` (University of Sheffield)

**通讯引用:** 2885 | [OpenAlex ID](https://openalex.org/A5045619924)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了 PROCESS-2 数据集，包含 400 名英国老年人（50 例痴呆、150 例轻度认知障碍、200 例健康对照）的远程语音评估数据（图像描述、语义流畅度、音位流畅度），并提供手工校对文本、元数据及预定义训练/测试划分。

**💡 创新点**

创新点在于：① 规模化且真实世界的远程采集，保留了背景噪音、设备多样性和自然交互；② 结合多种认知任务，覆盖语言生成、词汇检索和执行控制；③ 提供公开可复现的基准实验与嵌入空间几何分析，支持跨模型比较。

**🔧 技术方法**

使用技术包括：WebRTC 远程录音、FFmpeg 统一格式化、Silero VAD 计算 SNR、BERT/DistilBERT 等 LLM 对文本进行分类/回归，以及传统 LR/MLP 与语音特征（ComParE）模型的基准实验。

**📊 数据集**

数据集来源为 PROCESS-2，包含 1200 条语音文件、1200 条文本转录和 400 条参与者元数据（年龄、性别、诊断、MMSE）。

**📈 对比分析**

与经典机器学习模型（LR、MLP）相比，LLM 在 2‑分类中宏 F1 达到 0.85、在 3‑分类中达到 0.59，回归任务 RMSE 最低 3.87，表明文本特征与 LLM 在认知评估任务中更具优势；所有模型在预定义的 train/test 划分上表现一致，验证了数据集的无泄漏性。

**⚠️ 局限性**

局限性包括：① 仅限英国英语，缺乏跨语言通用性；② 仅提供受控访问的音频，限制公开使用；③ MMSE 仅在 174 名参与者中可用，样本量与标签不平衡；④ 仅涉及语音与文本，未包含视频或生理多模态信息。

---

## 569. Temporal Fair Division in Multi-Agent Systems: From Precise Alternation Metrics to Scalable Coordination Proxies

**arXiv ID:** 2605.14879 | [PDF](https://arxiv.org/pdf/2605.14879v1)

**作者:** Nikolaos Al. Papadopoulos `[一作]` (University of Macedonia), Nikolaos Al. Papadopoulos `[通讯]` (University of Macedonia)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5101649741)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了用于评估多代理重复资源竞争中时间公平性的轻量级指标——旋转周期性（RP）以及对已有滑动窗口指标集（ALT）的系统比较。

**💡 创新点**

创新点在于将时间公平性拆解为旋转分数（RS）和等待周期评估（WPE），实现线性时间复杂度 O(ν+n) 的衡量；同时通过与 ALT 指标的关联证明 RP 能捕捉到 ALT 发现的协调失败，并在大规模系统中保持可扩展性。

**🔧 技术方法**

使用 Q‑learning 作为学习代理，定义多代理 Battle of the Exes（MBoE）实验；对指标实现采用 Python，进行时间复杂度与精度分析。

**📊 数据集**

实验数据来自 MBoE 的 5 种 agent 数量 (n∈{2,3,5,8,10})，采用 ILF 与 IQF 两种奖励形式，episodes 规模按公式 ν=B·(n/2)^2·(1+ln(n!)/2!) 计算，随机策略作为基线。

**📈 对比分析**

比较方法包括：指标间的 Spearman 相关性、协调得分 (CS) 与随机基线的差异、以及 RP 与 ALT 的计算时间对比。结果显示：RP 与 ALT 相关性≥0.95；RP 在 n=10 时实现约 25× 的速度提升；Q‑learning 在 n≥3 时协调得分为负，远逊于随机基线，表明学习者在时间公平性上普遍失败。

**⚠️ 局限性**

局限性：Type‑B 状态表示的 winner‑flag 传递存在未修复 bug，可能影响实验结果；episode 预算公式假设所有配置状态空间相似，可能低估大 n 的实际复杂度；AWE 先前指标的硬阈值导致 RP 在 n≥3 时出现折叠问题。

---

## 570. ACE-LoRA: Adaptive Orthogonal Decoupling for Continual Image Editing

**arXiv ID:** 2605.14948 | [PDF](https://arxiv.org/pdf/2605.14948v1)

**作者:** Yuehao Liu `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 33776 | [OpenAlex ID](https://openalex.org/A5025545087)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对扩散模型的动态正则化框架ACE-LoRA，用于实现参数高效的持续图像编辑，并创建了首个面向持续学习的图像编辑基准CIE-Bench。

**💡 创新点**

创新点在于：① 采用Adaptive Orthogonal Decoupling（AOD）根据当前任务数据动态构造干扰向量并实现梯度正交化，显著降低跨任务干扰；② 引入Rank‑Invariant Historical Information Compression（SVD压缩）将历史LoRA模块合并为固定秩表示，解决随任务增多导致的计算量膨胀。

**🔧 技术方法**

主要技术包括：基于LoRA的参数高效微调、动态正交正则化、两阶段微调策略、SVD压缩、以及基于Qwen3.5的自定义评估器（Instruction Following + Perceptual Naturalness）。

**📊 数据集**

使用了Flux2‑Klein‑9B作为基础模型，并在CIE‑Bench中构建了六个多域图像编辑任务（ERP Outpainting、Refocus、Relighting、Text Editing、Virtual Try‑on、Causal Reasoning），任务数据来源为多来源高质量图像对。

**📈 对比分析**

与Zero‑Shot、Multi‑Task Fine‑Tuning、Sequential Fine‑Tuning、架构/正则化/重放等多种基线对比，ACE‑LoRA在Last和Avg指标上均优于所有对手，并在Instruction Following与Perceptual Naturalness得分上表现出色，几乎无退化（BWT接近0）。

**⚠️ 局限性**

局限性：训练时加入正则化约束会使训练时间约增加1.5倍；与其他LoRA类持续学习方法类似，随着任务数量增多，需持续合并新的适配器，可能影响可扩展性；在极大规模任务序列中，压缩后信息损失与稀疏性仍需进一步评估。

---

## 571. Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations

**arXiv ID:** 2605.14937 | [PDF](https://arxiv.org/pdf/2605.14937v1)

**作者:** Jonathan Spieler `[一作]` (University of Bonn), Sven Behnke `[通讯]` (University of Bonn)

**通讯引用:** 12474 | [OpenAlex ID](https://openalex.org/A5027761977)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种基于对象中心槽位表示的离线无奖励学习的世界模型，并将其与梯度基MPC相结合，用于从视觉观测中进行目标条件规划。

**💡 创新点**

创新点包括：① 在槽位空间直接优化动作序列，实现梯度基MPC；② 通过离线数据学习可控且可迁移的对象级动态模型；③ 结合策略先验加速收敛并显著降低规划维度。

**🔧 技术方法**

使用 Slot Attention + SAVi 进行场景解析，Transformer 预测器 (cOCVP) 进行动作条件动态建模，梯度基MPC 与 MPPI 对比，Hungarian 匹配处理槽位对齐，行为克隆策略网络作为暖启动。

**📊 数据集**

四个机器人操纵任务（Button Press、Lever Pull、Stack、Square）来自 Meta‑World 与其他基准，分别用随机探索轨迹和少量专家演示构成离线数据集。

**📈 对比分析**

与 Dreamer‑v3、DINO‑WM、GC‑BC 等基线对比；在 Button Press、Lever Pull 达到 Dreamer‑v3 的成功率，在 Stack、Square 超越所有基线；梯度基MPC 在低覆盖率场景下优于采样基MPPI，规划时间从几百毫秒降至几百毫秒。

**⚠️ 局限性**

局限性包括：对细粒度动作（如按压力度）不够精准；缺少本体感知或子目标支持；在离线数据覆盖不足时仍易出现外推误差；梯度优化可能陷入局部最优；目前仅在仿真环境验证，缺乏真实机器人实验。

---

## 572. Learning with Shallow Neural Networks on Cluster-Structured Features

**arXiv ID:** 2605.14927 | [PDF](https://arxiv.org/pdf/2605.14927v1)

**作者:** Elisabetta Cornacchia `[一作]` (INRIA), Laurent Massoulié `[通讯]` (INRIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种可分析的聚类相关输入模型，并证明在该模型下，层级梯度下降可以在样本复杂度与输入维度无关（仅对数项）地学习任何基于少量二值潜变量的目标函数。

**💡 创新点**

创新点在于：①在低潜维、维度无关的情形下给出严格样本复杂度分析；②利用高SNR条件下的协方差结构和激活函数的Hermite非退化性，证明第一层可有效提取潜变量；③通过随机初始化与Carbery–Wright不等式实现投影一致性，从而保证后续层可逼近任意目标。

**🔧 技术方法**

主要技术包括：层级梯度下降（first‑layer 一步，第二层 冻结训练），多项式激活函数与Hermite展开，Berry‑Esseen 与 Carbery–Wright 不等式，投影一致性与间隔覆盖论证，以及凸二次损失的收敛分析。

**📊 数据集**

实验数据：①合成数据（均匀噪声与高斯混合模型）用于学习 N=3 的 parity 函数；②真实单细胞 RNA‑seq 数据集 GSE96583，用于三分类细胞类型预测。

**📈 对比分析**

与传统聚类方法相比，实验显示在高 SNR 或维度足够大的情况下，所提方法的样本复杂度与输入维度几乎无关；在真实数据上，随着样本量增加，准确率曲线对不同 d 收敛到相同水平，表明理论预测得到验证。

**⚠️ 局限性**

限制：①潜变量维度必须固定不随 d 增大；②理论证明依赖于高SNR 条件（或在 BSC 情形下的 Δ‑margin），对低噪声或非均匀聚类的情况分析不充分；③实现使用多项式激活与随机初始化，实际中需改用 ReLU 等常用激活时需额外近似；④对深层网络与更大潜变量规模的推广尚未完成。

---

## 573. COREKG: Coreset-Guided Personalized Summarization of Knowledge Graphs

**arXiv ID:** 2605.14900 | [PDF](https://arxiv.org/pdf/2605.14900v1)

**作者:** Sohel Aman Khan `[一作]` (Indian Institute of Information Technology Delhi), Supratim Shit `[通讯]` (Indian Institute of Information Technology Delhi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种基于 coreset 理论的个性化知识图谱摘要方法 COREKG，利用用户查询工作负载的敏感度进行重要性采样，从而构造可证明误差界限的摘要子图。

**💡 创新点**

核心创新在于将 coreset 采样与用户特定查询敏感度相结合，实现了真正的个性化摘要，并提供了全局成本误差的理论保证，区别于传统的全局或聚合工作负载方法。

**🔧 技术方法**

使用了敏感度计算、重要性采样、加权 coreset 构造、SPARQL 查询预处理、Apache Jena Fuseki 执行等技术，并通过 Bernstein 不等式给出误差证明。

**📊 数据集**

在 Freebase、DBpedia、Wikidata 三大公开知识图谱上进行实验，并使用 WebQSP、LSQ、LC‑QuAD 等 SPARQL 查询日志作为用户查询工作负载。

**📈 对比分析**

与 GLIMPSE、iSummary、PEGASUS、APEX^2、PPR 等基线方法在覆盖率和 F1 评分上进行对比；COREKG 在所有数据集和预算设置下均显著优于基线，尤其在查询稀疏的场景下保持高准确度。

**⚠️ 局限性**

存在的局限包括：当用户查询工作负载极度稀疏或偏斜时，摘要可能过拟合；理论保证仅针对整体成本误差，无法针对单个查询给出精确保证；对动态或增量更新的支持仍需进一步完善。

---

## 574. SEDiT: Mask-Free Video Subtitle Erasure via One-step Diffusion Transformer

**arXiv ID:** 2605.14894 | [PDF](https://arxiv.org/pdf/2605.14894v1)

**作者:** Zheng Hui `[一作]` (Baidu Inc.), Yunlong Bai `[通讯]` (Baidu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为SEDiT的无遮罩、单阶段视频字幕抹除方法，利用Diffusion Transformer实现一次性采样，直接处理1080p视频并支持长视频分块推理。

**💡 创新点**

核心创新在于：①完全去除对OCR或手工掩模的依赖；②基于局部分布平移理论证明一阶采样可行；③在潜在空间拼接参考视频与噪声进行条件引导，并引入焦点损失提升字幕区域恢复；④通过块级推理和第一帧条件实现长视频时序一致性。

**🔧 技术方法**

使用了Diffusion Transformer（DiT）、VAE、Conditional Rectified Flow（CRF）流匹配、3D Rotary Positional Embedding、LoRA微调、焦点损失、块级流式推理与第一帧条件、以及prompt‑based指令控制等技术。

**📊 数据集**

训练集基于400K从Pexels获取的无字幕高清视频，并通过自研的字幕合成管线生成带字幕视频；评估集为VSR‑Bench‑400（400个样本），每个样本包含无字幕视频、字幕掩码及带字幕视频。

**📈 对比分析**

与Minimax‑Remover、Propainter、DiffuEraser等方法进行对比，采用PSNR、SSIM、LPIPS、FVD、MOS等指标；SEDiT在所有指标上均优于对手，且推理速度最快（一次采样约4秒），MOS最高。

**⚠️ 局限性**

局限性包括：在极短片段（<10帧）和严重运动模糊字幕时，可能无法完全抹除；以及对非常短镜头中静态字幕的抹除效果不佳。

---

## 575. Masked Next-Scale Prediction for Self-supervised Scene Text Recognition

**arXiv ID:** 2605.14885 | [PDF](https://arxiv.org/pdf/2605.14885v1)

**作者:** Zhuohao Chen `[一作]` (Nankai University), Yu Zhou `[通讯]` (Nankai University)

**通讯引用:** 32076 | [OpenAlex ID](https://openalex.org/A5012041302)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Mask‑Next‑Scale Prediction (MNSP)框架，用于自监督场景文本识别。

**💡 创新点**

创新点在于将Next‑Scale Prediction与Masked Image Modeling双重学习结合，利用跨尺度结构指导局部重建，并加入Multi‑scale Linguistic Alignment以保证语义一致性。

**🔧 技术方法**

采用ViT‑Small编码器，结合NSP、MIM分支和MLA模块，并使用教师‑学生的特征对齐与像素重建。

**📊 数据集**

在大规模无标签文本图像集Union14M‑U进行预训练，Fine‑tune在Union14M‑L并在六个标准STR基准（IIIT5K、IC13、SVT、IC15、SVTP、CUTE80）上评测。

**📈 对比分析**

与现有最先进方法相比，MNSP在Union14M上平均精度达86.2%/90.8%（预训练20epoch），在六大基准上平均精度96.7%，尤其在尺度与布局变化严重的数据上显著提升。

**⚠️ 局限性**

局限性主要是仍需大量无标签数据进行预训练，对极端尺度变化或极低分辨率仍有一定性能下降，并且模型训练和推理成本相对较高。

---

## 576. LATERN: Test-Time Context-Aware Explainable Video Anomaly Detection

**arXiv ID:** 2605.15054 | [PDF](https://arxiv.org/pdf/2605.15054v1)

**作者:** Mitchell Piehl `[一作]` (University of Iowa), Muchao Ye `[通讯]` (University of Iowa)

**通讯引用:** 499 | [OpenAlex ID](https://openalex.org/A5024079930)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种上下文感知的可解释视频异常检测框架，旨在通过时间证据聚合过程来改进视频异常检测的准确性和可解释性。

**💡 创新点**

创新点在于引入了上下文感知的异常评分和递归证据聚合模块，使得模型能够在测试时利用历史视频内容进行更准确的异常推理。

**🔧 技术方法**

使用了视觉语言模型（VLM）和图像基础的记忆机制，结合了视觉-文本对齐的门控机制来生成可靠的异常评分。

**📊 数据集**

在UCF-Crime和XD-Violence等大型基准数据集上进行了广泛实验。

**📈 对比分析**

与现有的可解释视频异常检测方法相比，提出的方法在检测准确性和解释一致性上显著提高，尤其是在使用冻结的VLM骨干网络时。

**⚠️ 局限性**

限制在于该方法依赖于历史上下文的有效性，如果历史内容不准确或不相关，可能会影响异常推理的质量。

---

## 577. SpeakerLLM: A Speaker-Specialized Audio-LLM for Speaker Understanding and Verification Reasoning

**arXiv ID:** 2605.15044 | [PDF](https://arxiv.org/pdf/2605.15044v1)

**作者:** KiHyun Nam `[一作]` (Korea Advanced Institute of Science and Technology), Joon Son Chung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 10627 | [OpenAlex ID](https://openalex.org/A5038723822)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SpeakerLLM框架，能够在单句说话人分析、录音环境理解、双句说话人比较和基于证据的验证推理之间实现自然语言交互。

**💡 创新点**

创新点包括：①层次化说话人分词器将全局说话人嵌入与帧级声学特征结合；②验证推理目标采用三块结构，将录音状态、属性兼容性与最终决策分离，生成可审计的自然语言决策轨迹。

**🔧 技术方法**

采用冻结的说话人编码器 ReDimNet-B3、Qwen2.5-1.5B-Instruct LLM；Q-Former 与 MLP 组成分层说话人分词器；两阶段训练（说话人理解→验证推理）和 LoRA 微调。

**📊 数据集**

使用 VoxCeleb1 与 LibriTTS-R 公开数据，结合元数据、音频提取特征以及 MUSAN/SLR28 语音环境仿真生成的监督标签。

**📈 对比分析**

与通用音频LLM（Qwen2.5-Omni-7B、Qwen3.0-Omni-30B、Audio Flamingo3）进行对比。SpeakerLLM-Base 在说话人属性与环境判断上显著优于基线，标准同/不同判决准确率达 96.1%；SpeakerLLM-VR 生成的三块决策轨迹格式完整率 100%，在不同声学条件下保持判决准确率并提升对属性相似但不同说话人的区分能力。

**⚠️ 局限性**

局限性在于实验仅在受控实验室语料上验证，缺乏真实噪声/远场数据；信度评估仅基于监督推理方案；未与阈值化分数后端结合；部署时需进一步考虑隐私、同意与公平性问题。

---

## 578. AI Knows When It's Being Watched: Functional Strategic Action and Contextual Register Modulation in Large Language Models

**arXiv ID:** 2605.15034 | [PDF](https://arxiv.org/pdf/2605.15034v1)

**作者:** Vinicius Covas `[一作]` (Universidad Anáhuac México), Jorge Alberto Hidalgo Toledo `[通讯]` (Universidad Anáhuac México)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在五种不同观察情境下进行100场四智能体辩论实验，测量词汇多样性（TTR）与信息长度，探究LLM对被观察感知的语言适应性。

**💡 创新点**

首次提出LLM的“功能性战略行动”概念，并揭示合成霍桑效应：LLM对观察者身份（人类vs AI）和监控强度具有敏感的词汇调节行为。

**🔧 技术方法**

使用GPT‑5.2大型语言模型、统一系统提示、TTR与字符计数等文本统计技术，结合单因素ANOVA与Tukey事后检验进行比较。

**📊 数据集**

数据来源为单一主题（AI是否具备意识）的四人辩论生成的约4,000条消息（100场×40条/场），无公开标准数据集。

**📈 对比分析**

通过单因素ANOVA与Tukey检验对五种条件进行比较，监视条件TTR提升约25%，受众条件信息长度最高；相较于未监视基线，监视带来显著更高词汇多样性，表现稳定且具有统计意义。

**⚠️ 局限性**

局限包括：仅测试单一模型和训练版本；样本量有限，未覆盖多主题与多模型；仅用TTR衡量词汇多样性，缺乏更细粒度的语篇质量评估；对机制的具体计算解释仍不清晰。

---

## 579. WARD: Adversarially Robust Defense of Web Agents Against Prompt Injections

**arXiv ID:** 2605.15030 | [PDF](https://arxiv.org/pdf/2605.15030v1)

**作者:** Tri Cao `[一作]` (National University Of Singapore), Bryan Hooi `[通讯]` (National University Of Singapore)

**通讯引用:** 5941 | [OpenAlex ID](https://openalex.org/A5065675832)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了 WARD，一种面向 Web 代理的多模态提示注入防御框架；

**💡 创新点**

创新点包括：①构建覆盖 719 个高流量 URL 与 10 个模拟高危平台的 177K 样本大规模数据集 WARD-Base；②针对 guard 自身的提示注入（PIG）设计专门数据集 WARD-PIG，并进行专门 fine‑tune；③提出自适应对抗攻击训练 A3T，让攻击者与防御者在记忆驱动的循环中共进化，显著提升鲁棒性；

**🔧 技术方法**

采用了视觉语言模型（VLM）与大语言模型（LLM）的联合推理；通过监督微调（SFT）学习检测标签、注入位置、攻击目标和推理；A3T 采用内外循环的对抗生成与奖励更新（GRPO）；

**📊 数据集**

使用的数据集包括 WARD-Base、WARD-PIG、WARD-Seed、WARD-Test，以及四个 OOD 基准（Popup、EIA、VPI、WASP）；

**📈 对比分析**

与 25 条基线（闭源 API、开源指令模型、通用安全 guard、提示注入 guard）进行对比；WARD 在 OOD benchmark 的召回率与 F1 均近乎 100%，FPR 接近 0%，对 PIG 与自适应攻击的抵抗力保持 100% 召回；在实战中并行执行时的延迟低于代理自身，且输出 token 数量显著减少；

**⚠️ 局限性**

局限性：①仍主要针对 Web 代理场景，跨域到其他 LLM 应用的适用性待验证；②依赖大规模 VLM/LLM 训练，计算成本与推理开销仍高；③对极端恶意注入或非 HTML/截图两种模式混合的极端情况，尚未彻底验证。

---

## 580. The Scientific Contribution Graph: Automated Literature-based Technological Roadmapping at Scale

**arXiv ID:** 2605.15011 | [PDF](https://arxiv.org/pdf/2605.15011v1)

**作者:** Peter A. Jansen `[一作]` (University of Arizona), Peter A. Jansen `[通讯]` (University of Arizona)

**通讯引用:** 20380 | [OpenAlex ID](https://openalex.org/A5110035999)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个科学贡献图谱（Scientific Contribution Graph），包含230k论文中的200万条科学贡献及其1250万个前置关系，并提出了技术需求预测任务。

**💡 创新点**

创新在于把贡献抽取和前置关系建模为跨论文的序列到序列生成与对齐，生成细粒度贡献节点和说明性边，填补论文引用与三元组间的表征鸿沟。

**🔧 技术方法**

采用GPT-OSS-120B等大型语言模型进行全文序列生成贡献与前置抽取，利用指令式提示和对齐模型生成依赖边，并使用嵌入检索支持快速搜索。

**📊 数据集**

以ACL Anthology和Semantic Scholar Open Research Corpus中的230k开放获取论文为训练和抽取数据，构建大规模图谱。

**📈 对比分析**

通过人工参考集评估抽取精度（91%召回、25%新增有效贡献），并在技术需求预测任务上做时间过滤回测，Claude Opus 4.6在未见论文上达到0.48 MAP，性能随模型更新显著提升，成本与性能折中明显。

**⚠️ 局限性**

主要局限在高昂的LLM调用成本（约0.10美元/篇论文），仅覆盖开放获取文献，对闭源领域的适用性受限，且抽取与对齐错误仍存在。

---

## 581. Extending CDCL to disjunctions of parity equations

**arXiv ID:** 2605.15002 | [PDF](https://arxiv.org/pdf/2605.15002v1)

**作者:** Paul Beame `[一作]` (University of Washington), Glenn Sun `[通讯]` (University of Washington)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将 CDCL 逻辑扩展到 XOR-OR-AND 正规型（XNF）公式的求解器 (⊕)，并证明其在 (⊕) 证明系统中的多项式模拟性。

**💡 创新点**

创新点在于：① 引入了新的 (⊕) 推理规则（加法与基变换），实现了无弱化的等价推理；② 通过新的语义定义实现了与经典 CDCL 1‑UIP 学习的直接对应；③ 设计了一套 XNF 专属启发式与证明日志格式 (⊕)-LRUP。

**🔧 技术方法**

核心技术包括：稠密 GF(2) 线性代数实现（行梯形、快速位运算）、基变换/加法推理、CSIDS 句子活跃度启发式、随机选取等式决策、LRUP(⊕) 证明日志。

**📊 数据集**

实验数据集涵盖：Ascon‑128 的 2‑XNF 400 个实例、随机 k‑XNF（k∈{2,3,4,5}）与受限随机 k‑XNF、提升的布料图（Pebbling）lifted 版本、以及 CNF 转换的 Tseitin 图（随机 k‑regular 图）共 75 个实例。

**📈 对比分析**

与 Kissat、CryptoMiniSAT、2‑Xornado 等主流求解器对比，⊕ 在大部分 XNF 家族上决策数更少、求解时间更短；在未经预处理的 CNF‑Tseitin 上表现出近多项式的时间尺度，显著优于传统 Resolution‑based 求解器。

**⚠️ 局限性**

局限性：尚未实现完整的 CDCL 启发式（重启、子句删除、子句最小化、预处理/内省技术、动态启发式切换）；对 1‑UIP 的严格性质尚未完全证明；在处理纯 CNF 公式时的等式决策策略可能不够高效。

---

## 582. Refactoring-as-Propositions: Proved Refactoring of Hybrid Systems via Proved Refinements

**arXiv ID:** 2605.15001 | [PDF](https://arxiv.org/pdf/2605.15001v1)

**作者:** Enguerrand Prebet `[一作]` (Karlsruhe Institute of Technology), André Platzer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5646 | [OpenAlex ID](https://openalex.org/A5080481427)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了“refactoring-as-propositions”原则，利用差分细化逻辑将混合系统的重构表述为命题，并通过证明细化关系来保证重构后系统安全性不变；

**💡 创新点**

创新点在于将传统的重构技术转化为可在细化逻辑中证明的命题，从而实现了可复用、自动化的安全重构，并首次在逻辑层面支持引入新的辅助变量（ghost refinement），为混合系统的形式化重构提供了统一、可验证的框架；

**🔧 技术方法**

主要技术包括差分细化逻辑（differential refinement logic）、其证明算子与规则、细化与等价的对合规则、局部细化与全局细化的上下文一致性证明、以及在 KeYmaera X 证明器中实现的细化公理与重构策略；

**📊 数据集**

本工作未使用传统意义的数据集，而是通过两个真实案例（ACAS X 事故避免系统和事件驱动到时序驱动汽车控制系统）来验证方法的可行性；

**📈 对比分析**

通过对 ACAS X 的两种安全域定义，手工证明从 200 条 tactic 减少到 16 条，显著提升了自动化程度；在事件→时序的转换案例中，重构步骤仅需 5–10 条手工指令，展示了方法在实际安全证明中的效率和可维护性；

**⚠️ 局限性**

局限性在于只能处理保持原有行为的细化重构，无法直接支持在安全系统中引入新的行为；目前仅覆盖了基本的混合程序结构，未扩展到通信、并行等更复杂的系统；

---

## 583. Towards Gaze-Informed AI Disclosure Interfaces: Eye-Tracking Attentional and Cognitive Load While Reading AI-Assisted News

**arXiv ID:** 2605.14999 | [PDF](https://arxiv.org/pdf/2605.14999v1)

**作者:** Pooja Prajod `[一作]` (Centrum Wiskunde & Informatica), Abdallah El Ali `[通讯]` (Centrum Wiskunde & Informatica)

**通讯引用:** 1695 | [OpenAlex ID](https://openalex.org/A5030623043)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在实验室环境中使用眼动追踪与NASA‑TLX问卷，研究了不同级别的AI使用披露（无披露、一行简短披露、详细披露）以及AI在新闻制作中的角色（编辑、部分内容生成）对读者的注意力与认知负荷的影响。

**💡 创新点**

创新点在于首次利用眼动数据揭示简短披露会显著提高注意力负荷（更长的注视时长与更多的扫视次数），而详细披露不增加负荷；并结合信息缺口理论解释此现象，提出基于读者实时注视模式的动态披露界面设计。

**🔧 技术方法**

使用了 Tobii Pro Fusion 眼动追踪仪、NASA‑TLX 认知负荷量表、GLM 统计分析、Benjamini–Hochberg 多重检验校正。

**📊 数据集**

数据集由34名校园受试者（21男、12女、1非二元）与12篇新闻稿（3篇政治、3篇生活方式）组成，每篇新闻有 AI 编辑版和 AI 部分生成版，配合三种披露文本（无、简短、一行、详细）。

**📈 对比分析**

通过比较注视持续时间、扫视次数和NASA‑TLX、瞳孔直径等指标，发现一行披露显著提高注意力负荷（Cohen's d≈0.3–0.4，p<0.05），而认知负荷指标无显著差异；详细披露与无披露相当。

**⚠️ 局限性**

局限性包括：样本主要为校园高AI素养人群，结果对低AI素养人群的普适性未知；实验室设置不完全符合真实新闻阅读情境；未对披露区进行AOI细化分析；披露文本长度差异未完全排除。

---

## 584. Predicting Response to Neoadjuvant Chemotherapy in Ovarian Cancer from CT Baseline Using Multi-Loss Deep Learning

**arXiv ID:** 2605.14991 | [PDF](https://arxiv.org/pdf/2605.14991v1)

**作者:** Francesco Pastori `[一作]` (European Institute of Oncology, IEO, IRCCS), Elena De Momi `[通讯]` (Politecnico di Milano)

**通讯引用:** 9666 | [OpenAlex ID](https://openalex.org/A5083795699)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了基于深度学习的非侵入性框架，利用术前CT的3D病灶掩模预测卵巢癌患者对新辅助化疗的响应。

**💡 创新点**

创新点在于将Vision Transformer与注意力聚合、带margin的监督对比学习以及LoRA参数高效微调相结合，提升了单中心数据上的响应预测精度。

**🔧 技术方法**

使用了DINOv3 ViT预训练编码器、LoRA微调、注意力聚合模块、双头结构（分类头与投影头）以及交叉熵+监督对比损失的组合。

**📊 数据集**

采用了单中心回顾性数据集，包含280例卵巢癌患者（147响应，133不响应），其中226例用于训练/验证，54例作为独立测试集。

**📈 对比分析**

通过与基线（Yin等的ResNet 2D模型）、仅交叉熵或冻结模型的对比，最终多损失LoRA模型在测试集上实现ROC‑AUC 0.73、准确率0.66、F1 0.70，显著优于基线。

**⚠️ 局限性**

局限性包括样本量有限、单中心设计、预测仅基于影像形态、RECIST判定存在边界不确定性，以及模型在正类上的保守概率估计，需要进一步校准与多中心验证。

---

## 585. Characterizing the visual representation of objects from the child's view

**arXiv ID:** 2605.14990 | [PDF](https://arxiv.org/pdf/2605.14990v1)

**作者:** Jane Yang `[一作]` (University of California San Diego), Bria Long `[通讯]` (University of California San Diego)

**通讯引用:** 1909 | [OpenAlex ID](https://openalex.org/A5010277551)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用头戴摄像记录的第一人称视频，提取并统计 3.68M 帧中的 129 种对象类别，研究其频率分布、实例变异性和类别间的表示结构。

**💡 创新点**

首次在大规模自然儿童视野中发现：对象分布极度倾斜且在超类（如动物、家具）内部的相似度比精心策划的图像集更高，揭示了儿童学习环境的独特统计特征。

**🔧 技术方法**

结合 YOLOE 目标检测、CLIP 与 DINOv3 嵌入、可视化与相似度分析（RSA）、统计阈值筛选等计算机视觉与表示学习技术。

**📊 数据集**

主要数据集为 BabyView 2025.1（868 小时、31 名 5–36 个月婴幼儿的第一人称视频），并将结果与 THINGS 公开图像集进行对比。

**📈 对比分析**

通过 RSA 构建 129×129 RDM，计算 BabyView 与 THINGS 之间的 Spearman 相关（CLIP 0.55，DINOv3 0.40），并用 Δ_d 统计量评估超类聚类强度；在大多数域 BabyView 的聚类显著优于 THINGS。

**⚠️ 局限性**

限制在检测精度、过滤阈值可能丢失非典型视角、数据主要为室内环境、样本规模有限、仅覆盖美国儿童，结果可能偏向上界并不一定适用于所有文化或外部场景。

---

## 586. Compositional Video Generation via Inference-Time Guidance

**arXiv ID:** 2605.14988 | [PDF](https://arxiv.org/pdf/2605.14988v1)

**作者:** Ariel Shaulov `[一作]` (Tel-Aviv University), Lior Wolf `[通讯]` (Tel-Aviv University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用冻结的文本到视频扩散模型的交叉注意力信号，在推理时引导生成过程以实现细粒度组合语义。

**💡 创新点**

通过训练轻量级组合分类器并在早期去噪步骤使用其梯度进行控制，而不需修改生成器权重或提供布局等外部控制。

**🔧 技术方法**

使用交叉注意力提取、视频反演、对偶反演、VLM预训练骨干、分类器训练与推理梯度引导。

**📊 数据集**

使用T2V-CompBench、VBench、VBench-Long以及从真实视频反演得到的注意力样本和合成视频进行训练与评估。

**📈 对比分析**

与基线模型和TTOM在短视频与长视频的语义/质量/组合准确性指标上对比，CVG在组合准确性和整体得分上均优于两者，尤其在动作用途、空间关系等维度提升显著。

**⚠️ 局限性**

依赖模型内部注意力质量，可能对模糊或复杂多主体场景效果不足，且仅改善推理时控制，未提升生成器的底层生成能力。

---

## 587. Second-Order Actor-Critic Methods for Discounted MDPs via Policy Hessian Decomposition

**arXiv ID:** 2605.14982 | [PDF](https://arxiv.org/pdf/2605.14982v1)

**作者:** Sanjeev Manivannan `[一作]` (Indian Institute of Technology Madras), Shuban V `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对折扣奖励设置的两时间尺度演员-评论家方法，利用Hessian‑vector product实现二阶更新；

**💡 创新点**

创新点在于：①将二阶梯度分解中的高方差交互项通过两时间尺度框架近似为零，从而获得稳定的交互无关二阶近似；②对优势函数和内在曲率分别提出两种Gauss‑Newton近似（ACGN1、ACGN2），在离散动作空间中显著提升样本效率；

**🔧 技术方法**

核心技术包括：actor‑critic框架、两时间尺度更新、Hessian‑vector product计算、优势基Gauss‑Newton与内在曲率Gauss‑Newton近似；

**📊 数据集**

在四个基准环境上评估：CartPole、LunarLander（离散）；Reacher、Humanoid（连续，来自MuJoCo）；

**📈 对比分析**

与REINFORCE、自然梯度（NGA）对比；结果显示ACGN1与ACGN2在离散任务上收敛速度最快、样本效率最高；在连续任务中ACGN1/ACGN2均优于REINFORCE，且在某些环境下可与NGA相媲美；

**⚠️ 局限性**

局限性：二阶近似对高维连续动作空间的噪声敏感，导致收敛波动；对非对数凸策略的曲率估计不稳定；未来工作需进一步改进可扩展的近似方法以及结合TRPO/PPO等策略。

---

## 588. MicroscopyMatching: Towards a Ready-to-use Framework for Microscopy Image Analysis in Diverse Conditions

**arXiv ID:** 2605.14980 | [PDF](https://arxiv.org/pdf/2605.14980v1)

**作者:** Xiaofei Hui `[一作]` (Lancaster University), Jun Liu `[通讯]` (Lancaster University)

**通讯引用:** 39394 | [OpenAlex ID](https://openalex.org/A5100361885)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于匹配的、无需额外训练的显微镜图像分析框架 MicroscopyMatching，能够统一处理分割、跟踪和计数等任务，并提供在线可视化工具。

**💡 创新点**

创新点在于将显微镜图像分析任务重构为“within‑image matching”问题，并利用预训练的潜在扩散模型（Stable Diffusion）的自注意力机制实现鲁棒的跨域匹配，真正实现“开箱即用”。

**🔧 技术方法**

核心技术为：预训练潜在扩散模型（VAE+扩散模块）+ 轻量化注意力后处理模块 + 可选的示例框提取投影器；评估采用多种指标（AP、SEG/TRA/LNK、MAE/RMSE）和专家/人类一致性测试。

**📊 数据集**

使用了 20 个公开基准数据集（包含分割、跟踪、计数任务）以及 200 组来自 135 所实验室的真实实验图像；还对 10 个未见过的基准集进行泛化测试。

**📈 对比分析**

与现有最先进方法（Cellpose, CelloType, MicroSAM, TrackMate 等）在所有任务上对比，MicroscopyMatching 在基准集、真实实验集、专家评价和人机一致性上均优于或与之持平；在用户研究中实现了约 99% 的耗时减少，准确率与人工注释无显著差异。

**⚠️ 局限性**

局限性包括：仅针对二维显微图像（需手动降维处理三维数据）；依赖于 Stable Diffusion，未来可尝试更先进的 LDM；尚未直接支持三维体数据或其他成像模态。

---

## 589. Viverra: Text-to-Code with Guarantees

**arXiv ID:** 2605.14972 | [PDF](https://arxiv.org/pdf/2605.14972v1)

**作者:** Haoze Wu `[一作]` (Amherst College), Nina Narodytska `[通讯]` (Broadcom)

**通讯引用:** 2589 | [OpenAlex ID](https://openalex.org/A5034897040)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从自然语言任务描述中，利用LLM同时合成C程序和一组断言，并通过组合式、基于循环展开的模型检查器自动验证这些断言，随后把验证通过的断言以自然语言注释的形式嵌入代码，帮助开发者快速理解和审计生成的代码。

**💡 创新点**

创新点包括：① 部分共规格化（partial co‑specification），让LLM在生成代码的同时生成安全性/正确性断言；② 最佳努力（best‑effort）组合式验证策略，利用调用图先验信息和已验证断言做假设，提升验证覆盖率；③ 将验证结果与自然语言事实关联，生成可读的注释；④ 在不需要完整证明的前提下，只提供在有限循环展开深度下的形式保证。

**🔧 技术方法**

技术手段：LLM提示工程（OpenAI GPT‑5.1）进行属性提取、程序合成、断言插入、循环展开优化和注释生成；基于CBMC、ESBMC与Bitwuzla的并行模型检查器组合；调用图抽取与逻辑线编号；依赖闭包与组合式验证；断言规范化与重写；编译验证与错误反馈。

**📊 数据集**

使用了18个多样化的编程任务集合，涵盖从基础排序、图算法、组合优化到数独、约束求解等算法问题，任务描述来自作者自定义的自然语言说明，超出了HumanEval和MBPP等常见数据集。

**📈 对比分析**

比较方法：① 对每个任务统计断言数量、未验证数（未通过、假设或超时）；② 记录LLM调用时间与验证时间；③ 通过亚马逊Mechanical Turk 进行问答实验，比较有无验证注释时的答题正确率与平均答题时间。结果显示：大多数断言在无假设时就能通过；验证平均时间≈300 s，LLM调用≈350 s；在400+参与者的实验中，加入验证注释的代码在准确率上平均提升约10%且答题时间平均缩短≈20%。

**⚠️ 局限性**

局限性：① 仅在给定循环展开深度k内保证断言，无法覆盖无限循环或更深层的行为；② 依赖模型检查器的可支持语言与性能，当前仅支持C且受限于验证器的求解能力；③ 断言质量依赖LLM生成的语义理解，若LLM产生错误或不完整的断言，验证结果可能不具意义；④ 需要手工制定属性提取与断言映射提示，缺乏自动化的质量评估机制。

---

## 590. Computational Imaging Priors for Wireless Capsule Endoscopy: Monte Carlo-Guided Hemoglobin Mapping for Rare-Anomaly Detection

**arXiv ID:** 2605.15062 | [PDF](https://arxiv.org/pdf/2605.15062v1)

**作者:** Chengshuai Yang `[一作]` (University of Texas Southwestern Medical Center), Raiyan Tripti Zaman `[通讯]` (University of Texas Southwestern Medical Center)

**通讯引用:** 194 | [OpenAlex ID](https://openalex.org/A5036992847)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并评估了一种基于Monte Carlo启发的血红蛋白先验的物理信息化输入通道，用于改进无线胶囊内镜帧级多类别分类。

**💡 创新点**

首次将闭式血红蛋白概率先验与CNN/Transformer结合，并提供可在部署时仅使用RGB的蒸馏方案，实现无硬件改动的性能提升。

**🔧 技术方法**

使用EfficientNet‑B0 backbone，构造5通道输入（RGB+血红蛋白+自发光代理）以及3通道+蒸馏头两种变体，并利用DeLong、McNemar、Bootstrap统计评估。

**📊 数据集**

在公开的Kvasir‑Capsule数据集上进行视频级70/15/15划分，评估11类可测试类别的AUC、宏F1等指标。

**📈 对比分析**

通过多种随机种子（6个）进行交叉验证，输入融合方案在宏AUC上平均提升+0.023（5/6种子显著），蒸馏方案平均提升+0.013（方向不一致），对Lymphangiectasia类提升最显著。

**⚠️ 局限性**

对罕见血管类如Angiectasia的性能高度依赖种子，单帧先验在血鲜类的单独使用效果不佳，且未在跨数据集或临床验证中评估。

---

## 591. NeuroTrain: Surveying Local Learning Rules for Spiking Neural Networks with an Open Benchmarking Framework

**arXiv ID:** 2605.15058 | [PDF](https://arxiv.org/pdf/2605.15058v1)

**作者:** Alessio Caviglia `[一作]` (Politecnico di Torino), Stefano Di Carlo `[通讯]` (Politecnico di Torino)

**通讯引用:** 4687 | [OpenAlex ID](https://openalex.org/A5026593274)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文综述了尖峰神经网络(SNN)的训练算法，提出了细粒度分类法，并构建了可复现的统一实验框架NeuroTrain。

**💡 创新点**

创新点在于提供系统化、细粒度的训练算法分类体系，并公开实现NeuroTrain，使不同算法在统一设置下可直接比较。

**🔧 技术方法**

主要技术包括对SNN训练方法的理论与实践梳理、构建基于snnTorch的NeuroTrain框架、利用Optuna进行超参数搜索以及模块化的数据集、模型与训练器设计。

**📊 数据集**

在框架中使用了MNIST、F‑MNIST、CIFAR‑10、SVHN、N‑MNIST、DVS Gesture、DVS CIFAR‑10、SHD等经典SNN数据集。

**📈 对比分析**

通过组合Trainer–Model–Dataset的实验组合，自动执行多轮实验并收集准确率、损失、耗时、参数量、稀疏度等指标，展示各训练规则在不同网络与任务上的性能差异。

**⚠️ 局限性**

局限包括算法对数据集与模型的可迁移性不足，硬件效率与能耗评估仍需深化，框架对极端深度或特殊神经元模型支持有限，以及实验仍受限于实现细节与超参数选择。

---

## 592. HiSem: Hierarchical Semantic Disentangling for Remote Sensing Image Change Captioning

**arXiv ID:** 2605.15024 | [PDF](https://arxiv.org/pdf/2605.15024v1)

**作者:** Man Wang `[一作]` (Inner Mongolia University), Zhenwei Shi `[通讯]` (Inner Mongolia University)

**通讯引用:** 15193 | [OpenAlex ID](https://openalex.org/A5058849690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对遥感图像变化描述（RSICC）任务，提出 HiSem 结构，分层拆分改变化与不变图像对的语义表示，并通过 BDAM 关注差异并强化跨时交互，使用 HASD 进行粗粒度与细粒度语义解耦，最终生成更准确、细致的自然语言变化描述。

**💡 创新点**

创新点：①将改变与不改变图像对的语义处理显式分层，解决同一模型同时处理不同语义粒度导致的语义混杂；②BDAM 模块以差异感知注意力调制跨时特征；③HASD 模块在粗粒度图像级路由后对变更样本采用 Mixture‑of‑Experts（MoE）细粒度标记，提升对多样、异构变化的建模能力；④多阶段课程学习策略逐步引入辅助分类监督，提升路由稳定性。

**🔧 技术方法**

核心技术包括 CLIP‑ViT 视觉编码器、双向差异注意力调制（BDAM）、层级自适应语义解耦（HASD）与 MoE 专家路由、Transformer 语言解码器、交叉熵与类比学习损失、以及余弦拉升的多阶段训练策略。

**📊 数据集**

实验使用公开两大 RSICC 基准数据集：LEVIR‑CC（10,077 对）和 WHU‑CDC（7,434 对），并在两者上评估多项指标（BLEU‑1/2/3/4、METEOR、ROUGE‑L、CIDEr‑D）。

**📈 对比分析**

与现有 30+ 先进方法（包括 RSICCFormer、Chg2Cap、SFT、Pix4Cap、RSCaMa 等）进行对比，HiSem 在 LEVIR‑CC 上实现 80.21（S*）/ 138.86（CIDEr）/ 65.82（BLEU‑4）等最佳指标；在 WHU‑CDC 上更显著提升，BLEU‑4 +7.52%、METEOR +3.38%、CIDEr‑D +8.95%、S* +5.65%，明显优于前沿方法。

**⚠️ 局限性**

局限性：①对“改变化”样本的路由准确性提升仍未显著转化为细粒度描述质量，表明细粒度语义建模仍面临挑战；②MoE 设计虽然提升了表现，但模型复杂度和推理开销有所上升；③依赖预训练 CLIP‑ViT，迁移到更高分辨率或不同传感器时可能需额外 fine‑tuning。

---

## 593. Sat3DGen: Comprehensive Street-Level 3D Scene Generation from Single Satellite Image

**arXiv ID:** 2605.14984 | [PDF](https://arxiv.org/pdf/2605.14984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 594. 3D Skew-Normal Splatting

**arXiv ID:** 2605.15010 | [PDF](https://arxiv.org/pdf/2605.15010v1)

**作者:** Xiangru Wu `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 16476 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Azzalini斜正态分布的Skew‑Normal Splatting (SNS) 作为3D Gaussian Splatting的新型不对称基元，实现对对象边界和一侧结构的精准建模。

**💡 创新点**

创新点包括：①引入可学习、受限的偏斜参数实现从对称Gaussian到Half‑Gaussian的连续形状插值；②在三维空间中保持对仿射变换与边缘分离的参数化，配合块状交替优化显著提升训练稳定性；③保持解析投影与梯度可导性，使SNS可无缝嵌入现有Gaussian Splatting管线。

**🔧 技术方法**

采用Azzalini斜正态分布、解析仿射投影、偏斜向量分解、SGHMC与Adam混合优化以及块坐标下降（BCD）策略。

**📊 数据集**

在Mip‑NeRF‑360、Tanks & Temples与Deep Blending三大基准数据集上进行实验。

**📈 对比分析**

与Gaussian、GES、HGS、SSS等非Gaussian核以及Scaffold‑GS、Fre‑GS、3DGS‑MCMC等优化方法对比，SNS在PSNR、SSIM、LPIPS上实现了显著提升（例如在Tanks & Temples上PSNR提升0.21dB、LPIPS降低0.006），并在边界细节与对象完整性方面表现更佳。

**⚠️ 局限性**

局限性包括：仍依赖三维Gaussian Splatting的固定稀疏结构，偏斜参数的学习曲线对不同场景需要手动调节；在极端薄弱或复杂纹理区域的逼真度仍不如基于深度网络的NeRF模型；计算开销与SSS相当，但相较于传统Gaussian仍略高。

---

## 595. Learning Developmental Scaffoldings to Guide Self-Organisation

**arXiv ID:** 2605.14998 | [PDF](https://arxiv.org/pdf/2605.14998v1)

**作者:** Milton L. Montero `[一作]` (IT University of Copenhagen), Sebastian Risi `[通讯]` (IT University of Copenhagen)

**通讯引用:** 3688 | [OpenAlex ID](https://openalex.org/A5020511097)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了如何通过同时学习初始预模式（SIREN）和神经元胞自动机（NCA）来实现自组织发育，并量化预模式与自组织过程之间的信息分配及其对稳健性、编码容量和对称性破缺的影响。

**💡 创新点**

创新点在于：①提出一种端到端联合训练框架，使预模式与自组织规则共享同一目标函数，从而实现信息下放；②通过信息论度量揭示预模式并非简单复制目标，而是引导发育轨迹；③展示预模式能显著提升系统的稳健性、存储容量和对称性破缺能力。

**🔧 技术方法**

主要技术包括：神经元胞自动机（NCA）实现自组织；SIREN（带正弦激活的隐式函数）生成空间坐标化的预模式；卷积编码器生成目标嵌入；端到端梯度下降训练；信息论指标（NMI、R²、SSIM）评估信息分配。

**📊 数据集**

使用了20个手工设计的表情符号图案作为目标数据集，包含多种对称性和结构相似性。

**📈 对比分析**

与单纯自组织的 GoalNCA 进行对比；结果显示预模式版在对细胞噪声的鲁棒性、对多个目标的编码容量（1–16个模式）以及对称性破缺方面均优于 GoalNCA；信息论分析进一步证明预模式在初始状态中携带的有效信息有限，主要作用是调控发育路径。

**⚠️ 局限性**

局限性包括：使用梯度下降而非进化方法可能限制模型的生物学可解释性；未充分验证预模式对进化适应性的提升；实验仅在小规模表情符号数据集上进行，难以推广到更复杂的生物结构；缺乏对预模式生成多样性与可塑性之间关系的深入分析。

---

## 596. Explainable Detection of Depression Status Shifts from User Digital Traces

**arXiv ID:** 2605.14995 | [PDF](https://arxiv.org/pdf/2605.14995v1)

**作者:** Loris Belcastro `[一作]` (University of Calabria), Paolo Trunfio `[通讯]` (University of Calabria)

**通讯引用:** 3451 | [OpenAlex ID](https://openalex.org/A5085240260)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出可解释框架，通过多维BERT分类、时间序列聚合、变化点检测与LLM报告生成，对用户数字痕迹中的抑郁状态转变进行检测与分析。

**💡 创新点**

创新点包括：①多BERT模型融合不同维度信号；②将每日分数平滑并构成连续轨迹；③使用自适应piecewise linear分段检测变化点；④用LLM生成结构化、阶段化的可解释报告并支持交互检索，形成端到端的可解释时序分析。

**🔧 技术方法**

使用技术包括BERT微调、BERTopic主题建模、移动平均平滑、Ramer–Douglas–Peucker启发式分段、LLM（ChatGPT等）生成报告、RAG检索增强。

**📊 数据集**

使用数据集：eRisk 2018 Reddit（1707用户、超过百万条贴文）和Kaggle Mental Health Social Media Twitter（20000条贴文）进行实验。

**📈 对比分析**

与直接LLM汇总基线对比：主题覆盖率从约40%提升至84%；LLM-as-a-judge在轨迹覆盖、时间连贯性、变化点敏感度和段落细节等指标均显著提升；消融实验证实分段、平滑及统计信息均对报告质量有重要贡献。

**⚠️ 局限性**

局限性包括：仅使用英语模型，跨语言/方言性能未知；轨迹稀疏或不规则时易受噪声影响；LLM生成的报告可能出现幻觉或不准确；未做临床诊断，仅作决策支持，存在隐私与伦理风险。

---

## 597. Agreement, Diversity, and Polarization Indices for Approval Elections

**arXiv ID:** 2605.14983 | [PDF](https://arxiv.org/pdf/2605.14983v1)

**作者:** Piotr Faliszewski `[一作]` (AGH University of Kraków), Tomasz Wąs `[通讯]` (University of Oxford)

**通讯引用:** 312 | [OpenAlex ID](https://openalex.org/A5083795172)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了多种面向认可选举的相互一致度、多样性与极化度指标，并用它们构建了认可选举的可视化地图。

**💡 创新点**

创新点在于引入了与饱和度无关的全局和局部一致度指标（如 PCC^+ 对应相互一致度），以及基于聚类的多样性与极化度指标，并将这些指标组合成“多样性+一致性+极化度”的三维特征空间，揭示了真实和合成选举的结构关系。

**🔧 技术方法**

技术手段包括统计相似度量（Hamming、Jaccard、Pearson）和聚类算法（k-medoids、谱聚类）来计算指标，随后通过欧氏距离与多维尺度映射（MDS）构建选举地图。

**📊 数据集**

使用的数据集涵盖合成模型（p-ID、k-Party、IC、IAM、Resampling、Euclidean 等）、公开现实数据（Pabulib、Preflib、Polkadot、Pol.is、Eurovision、Chopin、法国总统选举等）以及作者新收集的参与式预算与区块链选举。

**📈 对比分析**

通过将各指标在合成与现实选举上的取值与预期相对照，验证了指标的可解释性；在地图构建中，MDS 的平均乘法失真仅为 1.01，显著优于先前工作（1.21–1.26），表明三维特征空间能较好区分不同类型选举。

**⚠️ 局限性**

局限性包括：部分指标（如 Central Diversity）对饱和度敏感；Outer Diversity 对选举规模依赖明显；对极化度的度量仅考虑二分群组，无法捕捉多群组极化；聚类方法未给出全局最优解，可能影响结果稳定性。

---

## 598. Quantifying and Mitigating Premature Closure in Frontier LLMs

**arXiv ID:** 2605.15000 | [PDF](https://arxiv.org/pdf/2605.15000v1)

**作者:** Rebecca Handler `[一作]` (Stanford University), Nigam Shah `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了前沿大型语言模型在医疗多选题和对话中的过早闭合行为，并测试了安全提示对其影响。

**💡 创新点**

提出了针对多选与开放式交互的过早闭合评估框架，并系统比较不同模型在未回答场景下的表现。

**🔧 技术方法**

采用了安全提示（system‑level instructions）以及多选/对话评分体系（HealthBench rubric）进行自动评估。

**📊 数据集**

使用了 MedQA、AfriMed‑QA（改造为 NOTA）、HealthBench 861 题子集以及 HealthBench Professional 191 个红队攻击案例。

**📈 对比分析**

通过比较基础提示与安全提示下的假动作率、准确率、预留率等指标，发现安全提示可将假动作率下降约20–40%，但在红队场景仍超过 50%，整体表现仍有提升空间。

**⚠️ 局限性**

评估依赖 LLM 判定、提示过于简单、未检验真实部署效果，并且缺乏对模型训练、评估机制改进的系统性探讨。

---

## 599. On the Limits of PAC Learning of Networks from Opinion Dynamics

**arXiv ID:** 2605.15033 | [PDF](https://arxiv.org/pdf/2605.15033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 600. Deceptive Cookies: Consent by Design -- A Mixed Methods Study

**arXiv ID:** 2605.15056 | [PDF](https://arxiv.org/pdf/2605.15056v1)

**作者:** Liv Hilde Sjøflot `[一作]` (University of Oslo), Tobias A. Opsahl `[通讯]` (University of Oslo)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5107287144)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对5个网站的 cookie 同意横幅进行可用性测试和问卷调查，分析用户行为与隐私偏好之间的差距，以及撤回同意的难易度。

**💡 创新点**

揭示当前 cookie 横幅设计导致用户被迫同意的“设计性同意”，并量化撤回同意所需时间显著高于给予同意。

**🔧 技术方法**

混合方法研究，包括可用性测试（think‑aloud）、问卷调查、定量统计（Friedman、Wilcoxon）和定性主题分析。

**📊 数据集**

20名参与者的实验数据，涉及5个流行网站（来源于 Tranco 列表）的同意横幅交互记录和问卷答复。

**📈 对比分析**

通过比较各网站及设备的接受率与交互时长，并使用 Friedman 与 Wilcoxon 检验验证显著差异；结果显示平均撤回同意时间比给予同意长约 21 倍。

**⚠️ 局限性**

样本采用便利抽样且仅有 20 人，定性分析仅由一人完成，未对多重检验做 p‑值校正，可能存在观察者效应，研究结果难以推广至更广泛人群。

---

## 601. DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models

**arXiv ID:** 2605.15055 | [PDF](https://arxiv.org/pdf/2605.15055v1)

**作者:** Quanhao Li `[一作]` (Fudan University), Zuxuan Wu `[通讯]` (Fudan University)

**通讯引用:** 8131 | [OpenAlex ID](https://openalex.org/A5026167547)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DiffusionOPD，一种将单任务探索与多任务整合分离的在线策略蒸馏框架，用于训练多目标的扩散模型。

**💡 创新点**

创新点在于：①将 OPD 从离散 LLM 转移到连续状态的马尔可夫链，得到闭式逆 KL 损失；②通过该闭式目标实现低方差训练，同时兼容 SDE 与 ODE 采样；③采用两阶段训练（先独立训练教师，再蒸馏到统一学生），避免交叉任务冲突与灾难性遗忘。

**🔧 技术方法**

技术包括：在线策略蒸馏 (OPD)、扩散模型逆 SDE / ODE 采样、闭式 KL 目标、对比 PPO 风格策略梯度、两阶段教师-学生训练、梯度累积与多任务循环采样。

**📊 数据集**

使用的任务与数据集包括：GenEval (组合生成)、OCR (文本识别)、美学 (审美评分)、PickScore、ClipScore、HPSv2.1、ImageReward、UnifiedReward 等多种规则/模型奖励，采用 SD3.5-Medium 基础模型并在 512×512 分辨率下评估。

**📈 对比分析**

与单任务教师、多任务 RL（如 GRPO-Guard、NFT）、级联 NFT 等基线对比，DiffusionOPD 在训练效率和最终性能上均明显优于基线，在 GenEval、OCR、Aesthetics 等指标上均实现或逼近最优水平，且收敛速度最快。

**⚠️ 局限性**

局限性包括：需要为每个任务单独训练教师模型，教师训练成本较高；目前仅验证在文本-图像生成的扩散模型，尚未探索更大规模或其他生成任务；在极端任务不平衡或奖励冲突较强的场景下，效果可能仍受限。

---

## 602. Tradeoffs are Domain Dependent: Improving Accuracy and Fairness in Property Tax Assessments

**arXiv ID:** 2605.15020 | [PDF](https://arxiv.org/pdf/2605.15020v1)

**作者:** Evelyn Smith `[一作]` (American Bar Foundation), Daniel E. Ho `[通讯]` (Stanford University)

**通讯引用:** 16785 | [OpenAlex ID](https://openalex.org/A5058408154)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析美国物业税评估中准确性与公平性关系，发现更高准确性往往伴随更公平税负；

**💡 创新点**

提供实证证据反驳公平性与准确性在该领域必然存在权衡，提出利用公开人口普查数据实现双赢；

**🔧 技术方法**

采用LASSO与随机森林等机器学习模型构建虚拟评估，并在模拟中加入更多属性特征；

**📊 数据集**

使用Cotality提供的2600万笔单户住宅交易记录与美国社区调查（ACS）5年期块组特征；

**📈 对比分析**

与现行评估方法及稀疏模型比较，结果显示在约96%县域中同时提升准确率与公平度，且在无评估增幅上限的县可达14.4%；

**⚠️ 局限性**

局限在于依赖代理销售价格作为真实价值、样本选取偏差、缺乏个体层面种族与收入数据、以及对州级评估法令影响的未充分考量。

---

## 603. GraphFlow: An Architecture for Formally Verifiable Visual Workflows Enabling Reliable Agentic AI Automation

**arXiv ID:** 2605.14968 | [PDF](https://arxiv.org/pdf/2605.14968v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 604. From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG

**arXiv ID:** 2605.15019 | [PDF](https://arxiv.org/pdf/2605.15019v1)

**作者:** Guanhua Chen `[一作]` (University of Macau), Derek F. Wong `[通讯]` (University of Macau)

**通讯引用:** 3818 | [OpenAlex ID](https://openalex.org/A5101468579)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了面向细粒度视觉问答的多模态检索增强生成框架GranuRAG，并构建了以建筑遗产为主题的GranuVistaVQA基准数据集。

**💡 创新点**

创新点在于将视觉元素视为首要检索单元，先通过开放词汇检测得到可见元素，再进行多粒度跨模态匹配与检索，最终生成可追溯到元素和检索结果的答案，实现透明可验证的推理。

**🔧 技术方法**

主要技术包括YOLO‑World开放词汇目标检测、LLM驱动的视觉–文本匹配、层级检索（元素描述+全局元数据）以及在生成阶段加入元素归因约束的提示工程。

**📊 数据集**

使用的自研数据集GranuVistaVQA共计1,422张建筑遗产图像，覆盖71个地标，提供每张图像的可见元素标签与对应的专业文本描述；实验还对比了多种公开的多模态RAG基线。

**📈 对比分析**

实验结果显示，GranuRAG在ROUGE‑L、BERT‑F1和LLM‑Score等指标上分别提升约29.2%、8.3%和约24.7个百分点，显著优于基线和CoT提示、传统RAG等方法；在离域样本上仍保持显著提升。

**⚠️ 局限性**

局限性包括：数据集仅覆盖建筑遗产场景，未覆盖抽象图像或日常室内场景；检测与多粒度检索步骤导致推理时间约为3.5秒，较基线延迟较大；目前仅采用传统监督微调，未探索更高效的参数高效微调技术。

---

## 605. Small, Private Language Models as Teammates for Educational Assessment Design

**arXiv ID:** 2605.15015 | [PDF](https://arxiv.org/pdf/2605.15015v1)

**作者:** Chris Davis Jaldi `[一作]` (Wright State University), Eleni Ilkou `[通讯]` (TIB – Leibniz Information Centre for Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地比较了大语言模型（LLM）与小型本地模型（SLM）在基于布鲁姆认知层级的自动化教育题目生成（AEQG）中的生成与评估能力。

**💡 创新点**

创新点在于提出可复现的多维度NLP评估指标、对比小型模型与大模型的性能、深入分析模型评判（model-as-judge）的可靠性与偏差，并强调人机协同的边界助手角色。

**🔧 技术方法**

采用布鲁姆级别提示的Prompt工程，利用Flesch‑Kincaid等级、阅读易读度、BERTScore、动词符合度等NLP指标，并通过ANOVA、F检验等统计方法评估模型表现。

**📊 数据集**

使用原有的2550条题目数据集并生成了4080条新题目，覆盖17个机器学习/数据科学主题，评估以专家标注为基准。

**📈 对比分析**

通过对生成质量和评估可靠性的对比，发现SLM在多项指标上与LLM竞争力相当且方差更小，LLM在显式提示下表现更佳；但模型自评在一致性与偏差方面存在显著波动，提示需要人工干预。

**⚠️ 局限性**

局限性包括仅聚焦数据科学领域、未进行微调、专家评判可能存在主观性、动词检查仅为表层指标、评估指标未完全覆盖教学效度以及对多元受众与公平性考量不足。

---

## 606. Performance-Driven Policy Optimization for Speculative Decoding with Adaptive Windowing

**arXiv ID:** 2605.14978 | [PDF](https://arxiv.org/pdf/2605.14978v1)

**作者:** Jie Jiang `[一作]`, Xing Sun `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型推理中，通过在推理阶段让轻量级草稿模型（drafter）生成可供并行验证的窗口式候选 token 以加速推理；论文提出 PPOW（Performance‑Driven Policy Optimization with Adaptive Windowing）框架，采用窗口级强化学习对 drafter 进行训练，显著提升接受长度与整体速度。

**💡 创新点**

①将 drafter 训练从传统的 token‑级监督迁移到窗口级 RL；②设计成本感知 Speedup Reward 与分布相近度奖励（Distribution‑Based Proximity Reward）以补偿早期截断时的稀疏信号；③引入自适应 divergence‑aware windowing（ADAW），根据 draft–target KL 与目标分布置信度加权，聚焦最具信息量且对接受瓶颈影响最大的窗口。

**🔧 技术方法**

窗口级 PPO（proximal policy optimization）强化学习框架；KL 正则化以保持与目标模型分布一致；成本感知奖励与分布相近度奖励；ADAW 窗口采样策略。

**📊 数据集**

使用 LLaMA‑3.1/3.3（8B、70B）和 Qwen3（8B、32B）作为目标模型；草稿模型基于 EAGLE‑3；评估任务包括 MT‑Bench（多轮对话）、HumanEval（代码生成）和 GSM8K（数学推理）。

**📈 对比分析**

与 GRIFFIN、EAGLE‑3 等已有 drafter 在统一解码协议下比较；PPOW 在所有模型与任务上平均提升接受长度至 6.29–6.52，速度提升为 3.39–4.36×；在较小候选组大小下也能达到相近性能；相较于继续监督训练，PPOW 在相同训练步数下能持续提升性能。

**⚠️ 局限性**

①仍依赖窗口大小与候选组大小的调参；②需要额外的 RL 训练步骤，训练成本相对较高；③在开放式对话（MT‑Bench）提升有限；④自适应窗口策略需要额外阈值与权重设定；⑤未在更大模型或多模态任务中验证，泛化性尚待进一步检验。

---

## 607. An Interpretable Latency Model for Speculative Decoding in LLM Serving

**arXiv ID:** 2605.15051 | [PDF](https://arxiv.org/pdf/2605.15051v1)

**作者:** Linghao Kong `[一作]` (Massachusetts Institute Of Technology), Alexandre Marques `[通讯]` (Red Hat Ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在真实 LLM 服 务 环境中构建了可解释的延迟模型，用 Little's Law 推导有效批次大小，并通过预填充/草稿/验证成本分解来分析 Speculative Decoding 的时延与负载关系；

**💡 创新点**

创新点在于把 SD 的加速机制纳入服务器级别的稳态时延模型，揭示了接受率、草稿长度、模型尺寸、专家覆盖等因素对低负载和高负载时延与加速的影响，并将该框架推广到 Mixture‑of‑Experts 结构；

**🔧 技术方法**

技术上使用 vLLM 连续批处理与分块预填充、GuideLLM 控制负载、Little's Law 推导有效批次、以及针对 SD 的成本模型与专家覆盖修正；

**📊 数据集**

实验数据来自多种大型模型（Llama‑3.1‑8B/70B、gpt‑oss‑20B、Qwen3‑系列），使用 Jane Austen《Pride and Prejudice》文本模拟请求，覆盖不同预填/解码长度、接受率与草稿长度；

**📈 对比分析**

通过在 RPS 变化区间测量平均时延与速度提升，结果表明 SD 在低负载时可获得显著加速，但随负载升高加速下降，模型对不同模型与参数的预测误差均小于5%，并成功解释了 MoE 模型在低负载时延降低的现象；

**⚠️ 局限性**

局限性包括仅在稳态预饱和区间预测平均时延，未考虑饱和点预占与突发负载；模型参数依赖具体硬件与调度策略，需在新系统上重新测量；

---

## 608. Analyzing Codes of Conduct for Online Safety in Video Games at Scale

**arXiv ID:** 2605.15047 | [PDF](https://arxiv.org/pdf/2605.15047v1)

**作者:** Jiuming Jiang `[一作]` (University of Edinburgh), Jingjie Li `[通讯]` (University of Edinburgh)

**通讯引用:** 499 | [OpenAlex ID](https://openalex.org/A5100647029)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并应用一套可扩展的管道，对Steam平台上多玩家游戏的在线守则（CoCs）进行大规模识别和分析，评估其普及度、内容与治理方式；

**💡 创新点**

首次系统性地构建CoC识别与分析框架，揭示不同游戏类型和违规类型在CoC覆盖率与具体性的差异，并为改进游戏安全治理提供实证依据；

**🔧 技术方法**

使用网络爬虫抓取游戏页面、文本解析技术提取CoC内容，再结合自然语言处理（NLP）进行主题分类和规范细度评估；

**📊 数据集**

基于Steam上9,586款多玩家游戏的公开数据，最终识别出350款具备CoC的游戏；

**📈 对比分析**

通过对比有无CoC游戏的覆盖率，以及CoC对传统安全违规与人际/未成年人伤害等不同违规类型的覆盖与具体程度进行定量对照，发现大约3.6%的游戏具备CoC，且在传统安全违规方面覆盖率高于人际与未成年人安全等；

**⚠️ 局限性**

研究仅聚焦Steam平台，未涵盖其他游戏商店或移动端；CoC文本识别与解析依赖自动化方法，可能存在漏检或误判；且仅分析CoC文本规范，未评估其在实际运营中的执行效果与用户体验。

---

## 609. Orchard: An Open-Source Agentic Modeling Framework

**arXiv ID:** 2605.15040 | [PDF](https://arxiv.org/pdf/2605.15040v1)

**作者:** Baolin Peng `[一作]` (Microsoft Research), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 35506 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Orchard 框架，核心是轻量化、可扩展的环境服务 Orchard Env，并在软件工程、浏览器 GUI 与个人助手三个领域构建了三套端到端的 agentic modeling 方案；

**💡 创新点**

创新点在于将环境层拆离为独立、Kubernetes‑native 的服务；通过多教师、多 harness 采样、credit‑assignment SFT 以及 Balanced Adaptive Rollout（BAR）实现的 RL 训练，显著提升开源 agent 的能力；

**🔧 技术方法**

技术包括 Kubernetes‑native 环境服务、FastAPI orchestrator、in‑pod agent 注入、SFT 与 RL（GRPO、BAR、SGLang、Megatron‑LM、Ray 并行），以及多模态 Vision‑Language 交互；

**📊 数据集**

使用的数据集涵盖 SWE‑bench（Verified、Multilingual）、SWE‑rebench、SWE‑rebench V2、Scale‑SWE、WebGym（WebVoyager、Online‑Mind2Web、DeepShop）、Claw‑Eval 与 ClawHub 等多任务源；

**📈 对比分析**

在各基准上与同规模开源与更大规模模型对比，SWE‑bench Verified 67.5%（30B/3B active）≈MoE 系统；GUI 68.4% avg（WebVoyager 74.1%、Online‑Mind2Web 67.0%、DeepShop 64.0%），仅 4B backbone，超过同规模开源并接近商业系统；Claw‑Eval 59.6% pass@3，超越同规模模型，体现 RL 的显著提升；

**⚠️ 局限性**

限制包括对未见 harness 或跨域任务仍存在性能下降；依赖大量教师轨迹和 RL 资源；环境服务受 Kubernetes 资源与成本管理限制；对极端安全或高压任务的鲁棒性尚未充分验证。

---

## 610. SemaTune: Semantic-Aware Online OS Tuning with Large Language Models

**arXiv ID:** 2605.15026 | [PDF](https://arxiv.org/pdf/2605.15026v1)

**作者:** Georgios Liargkovas `[一作]` (Columbia University), Kostis Kaffes `[通讯]` (Columbia University)

**通讯引用:** 339 | [OpenAlex ID](https://openalex.org/A5001679943)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

设计并实现了一个面向主机的、基于大语言模型的在线操作系统调优框架（SemaTune），通过双循环控制、显式记忆和类型化验证，在运行中的服务上动态调节多达 41 个 Linux 参数，实现稳态性能提升。

**💡 创新点**

创新点在于：
1) 将 LLM 作为语义推理器而非直接决策器，利用其对参数含义、系统信号及历史交互的理解来预防不合理配置；
2) 双循环（即时快速循环 + 较慢推理循环）平衡了响应延迟与推理深度；
3) 引入跨会话记忆与在线剪枝，减少搜索空间并加速收敛；
4) 通过类型化的安全接口限制 LLM 的操作，保证对内核状态的安全性。

**🔧 技术方法**

技术：
- 大语言模型（Gemini 2.5 Flash/Flash‑Lite，OpenAI API 等）用于生成配置建议；
- 双循环 LLM 调度器（Instant 与 Reasoning）；
- 语义化上下文构造与结构化提示；
- 内存模块（即时记忆 + 交叉会话向量存储）；
- 类型化参数验证器（参数域、依赖关系检查）；
- Python 控制面，调用 Linux sysctl / procfs 等接口。

**📊 数据集**

数据集：
13 个在线工作负载，来自 5 大基准套件：
- Memcached、TPC‑C、Wikipedia、YCSB、Twitter、SIbench、Masstree、Silo、Xapian、Sysbench OLTP（Read‑Write）、Sphinx、Sparkbench、Tailbench；
- 对每个工作负载使用 30 个调优窗口 + 20 个稳定窗口，重复 5 次。

**📈 对比分析**

比较方法：
- 与默认 Ubuntu 22.04 参数、MLOS、Bayesian、DQN、Q‑Learning 等基线对比；
- 对于系统信号和应用信号分别评估；
- 性能指标为相对 p99 延迟/吞吐的几何平均改进。结果显示：
  * SemaTune 在调优阶段提升约 +59%，稳定阶段 +72%（相较默认）；
  * 与 MLOS 的平均提升达 +153%；
  * 在仅使用系统指标时仍比 MLOS 高出 +93%；
  * 通过双循环实现的成本-性能比最优，单循环推理成本高、即时成本低但效果差；
  * 在扩展到 41 个参数时保持正向提升，MLOS 在 16+ 参数后性能下降甚至变负。

**⚠️ 局限性**

局限性：
- 仅针对稳态在线调优，无法处理需要毫秒级子系统自适应的突发或瞬态工作负载；
- 当前实现一次只能调优单个主应用，未覆盖多应用共存、分区资源等情形；
- 依赖大语言模型的推理开销和 API 费用，虽然通过双循环和剪枝降低，但在资源受限环境仍为挑战；
- 对极端硬件或极度耦合的系统设置的安全性仍需进一步验证。

---

## 611. MHSA: A Lightweight Framework for Mitigating Hallucinations via Steered Attention in LVLMs

**arXiv ID:** 2605.14966 | [PDF](https://arxiv.org/pdf/2605.14966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 612. COTCAgent: Preventive Consultation via Probabilistic Chain-of-Thought Completion

**arXiv ID:** 2605.15016 | [PDF](https://arxiv.org/pdf/2605.15016v1)

**作者:** Zihan Deng `[一作]` (University of Hong Kong), Chuanzhi Xu `[通讯]` (University of Sydney)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5082895857)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了COTCAgent框架，利用可执行趋势统计、知识库能量匹配和有界多轮对话来支持基于纵向电子健康记录的临床决策。

**💡 创新点**

将趋势统计、Symptom–Trend–Disease知识库和 IDF 加权 Gibbs 能量分层结合，并通过可追溯的链式推理和限制式问答减少模型幻觉与不一致。

**🔧 技术方法**

采用可执行统计模块TSA、IDF 权重化的知识库匹配、Gibbs 能量与 softmax 排名，以及限定轮次的对话完成策略。

**📊 数据集**

使用自建的纵向EHR模拟集、HealthBench 评测数据，以及由 Medscape/WebMD、PubMed 等公共文本构建的 9,948 疾病 Symptom–Trend–Disease 知识库。

**📈 对比分析**

在纵向风险预测与对话评测上与 TimeCAP、Google agent、KARE、DirPred 等基线对比，COTCAgent 在自建集实现 90.47% Top‑1、HealthBench 70.41% Top‑1，明显优于同类模型。

**⚠️ 局限性**

局限在于缺乏概率校准、仅关注趋势诊断不涵盖药物安全或预后预测；知识库对罕见疾病覆盖有限；实时部署的延迟与失败模式尚未充分评估。

---

## 613. After the Interface: Relocating Human Agency in the Age of Conversational AI

**arXiv ID:** 2605.15064 | [PDF](https://arxiv.org/pdf/2605.15064v1)

**作者:** Mengke Wu `[一作]` (University of Illinois Urbana Champaign), Mike Yao `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并阐述了在会话式 AI 环境中，人类代理权的转移——从界面可见的控制迁移到交互过程中的结果评估。

**💡 创新点**

创新点在于提出“过程控制”与“结果控制”两维框架，并将现有交互范式映射到该二维空间，揭示了代理权在新型 AI 系统中的重分布与“隐形不平等”。

**🔧 技术方法**

主要使用概念性分析、系统定位图和案例对比方法；没有实现或训练任何机器学习模型。

**📊 数据集**

无数据集，整个工作基于理论阐述和已有系统实例（如 Copilot、聊天助手、Agentic 系统）。

**📈 对比分析**

由于是概念性论文，没有实验对比或性能指标；讨论基于对不同交互范式的特征和控制维度的理论映射。

**⚠️ 局限性**

局限性：缺乏实证验证和量化测量，无法直接评估提出框架在实际系统中的有效性；对用户评估能力的假设未被实测；对公平性与责任分配的细节仍需进一步研究。

---

## 614. TFGN: Task-Free, Replay-Free Continual Pre-Training Without Catastrophic Forgetting at LLM Scale

**arXiv ID:** 2605.15053 | [PDF](https://arxiv.org/pdf/2605.15053v1)

**作者:** Anurup Ganguli `[一作]` `[通讯]` (Independent Researcher), Anurup Ganguli (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TFN‑GNN（TFGN）架构，能够在无重放、无任务标签、无正则化的条件下，实现 LLM 规模的连续预训练，并保持先前域的能力。

**💡 创新点**

创新点在于将可读写解耦的 overlay 嵌入 Transformer 内部，自动在参数更新上实现跨域正交，既不需要任务标识也不依赖经验回放。

**🔧 技术方法**

技术包括在 Transformer 块内加入输入条件化的参数化子网络，使用交叉域梯度正交化和内部自监督控制层实现自闭环学习。

**📊 数据集**

使用六个异构文本域（Prose、Python、Math、Biomedical、Chinese、JavaScript）的公开数据集，每个域约 10 亿 token 进行训练。

**📈 对比分析**

与标准 fine‑tuning、LoRA 等基线对比，TFGN 在从零训练和微调两种模式下，BWT 降至 -0.007，保持 HellaSwag 0.506~0.510，梯度正交率≥99.59%，并在不破坏先前域发射的同时实现正向跨域迁移。

**⚠️ 局限性**

局限性包括仅在特定六域序列和 1B token/阶段的实验，未验证更大域数和更复杂任务，仍需较高计算资源，且对极端分布漂移的鲁棒性待进一步评估。

---

## 615. Separating Intrinsic Ambiguity from Estimation Uncertainty in Deep Generative Models for Linear Inverse Problems

**arXiv ID:** 2605.15050 | [PDF](https://arxiv.org/pdf/2605.15050v1)

**作者:** Yuxin Guo `[一作]` (Carnegie Mellon University), Pulkit Grover `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3746 | [OpenAlex ID](https://openalex.org/A5054746425)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究逆问题中后验不确定性的结构分解，提出级联生成框架和基于模拟校准（SBC）的诊断方法，并在 Gaussian、加速 MRI 与 EEG 源成像任务中进行验证。

**💡 创新点**

将后验不确定性分解为可观测部分和不可观测的内在模糊，利用级联模型直接估计后者；同时引入高维模糊分量的 SBC 校准，揭示传统重建指标无法发现的误差。

**🔧 技术方法**

线性 SVD 分解、级联生成网络（range 模型 + null 模型）、条件 DDPM 或 VAE、SBC 统计量（β₂ 与 β∞/β₂）以及噪声模拟等技术。

**📊 数据集**

单臂 Knee MRI (NYU fastMRI) 25% k‑space undersampling；MNE sample 数据集的单个头模型，生成 60 传感器、4699 体素的 EEG 任务。

**📈 对比分析**

通过重建质量（MSE、相关系数）和 SBC 直方图对比；结果表明即使两种模型在重建质量上相近，它们在内在模糊的校准上存在显著差异。

**⚠️ 局限性**

仅适用于有监督数据；SBC 仅提供必要但非充分的校准检验；分解假设线性前向运算，非线性或不确定算子难以直接实现。

---

## 616. A Prototyping Framework for Distributed Control of Multi-Robot Systems

**arXiv ID:** 2605.15049 | [PDF](https://arxiv.org/pdf/2605.15049v1)

**作者:** Junaid Ahmed Memon `[一作]` (University of Oxford), Antonis Papachristodoulou `[通讯]` (University of Oxford)

**通讯引用:** 8759 | [OpenAlex ID](https://openalex.org/A5053811056)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一套基于单台多核计算机的分布式控制原型框架，可在三种逼真度水平（点质量、高清数字孪生、硬件）下快速验证多机器人算法

**💡 创新点**

创新点在于将分布式算法与SPMD并行执行相结合，形成统一的工作流，支持同一代码在模拟、数字孪生与真实硬件之间无缝迁移

**🔧 技术方法**

采用MATLAB并行计算工具箱的SPMD块、Simulink“Simulation Object”、Crazyflie 2.0物理模型以及Crazyradio无线接口实现控制

**📊 数据集**

使用了四架Crazyflie 2.1四旋翼的实验数据，以及对应的点质量和高保真模型的仿真数据

**📈 对比分析**

通过对轨迹终点、碰撞情况和每个控制循环的计算时长进行比较，三种平台均保持在200 ms实时预算内，硬件与点质量模型结果高度一致，高保真模型仅因动态细节导致略慢

**⚠️ 局限性**

局限在于单机仿真无法完全再现无线干扰、射频争抢和物理时序抖动，需在真实硬件上进一步验证

---

## 617. EverAnimate: Minute-Scale Human Animation via Latent Flow Restoration

**arXiv ID:** 2605.15042 | [PDF](https://arxiv.org/pdf/2605.15042v1)

**作者:** Wuyang Li `[一作]` (VITA@EPFL), Alexandre Alahi `[通讯]` (VITA@EPFL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了EverAnimate，一种高效的后训练框架，用于生成分钟级长时程的姿势驱动人类动画视频。

**💡 创新点**

创新点在于通过持久的潜在传播和恢复性流匹配两种机制，解决低级质量漂移和高级身份漂移，并在潜在空间中实现跨块语义持续。

**🔧 技术方法**

采用视频扩散变压器DiT、LoRA轻量化后训练、潜在记忆构建、姿态适配器、面部指导以及修正流匹配等技术。

**📊 数据集**

在Champ、UBC、Seedance以及自采的2k分钟级YouTube视频上进行训练和评估。

**📈 对比分析**

与One-to-All、SCAIL、SteadyDancer、UniAnimate-DiT、Wan-Animate等方法对比，在10s到90s不同时间窗口下的PSNR、SSIM、LPIPS、FID、V-MAE和面部PSNR均显著提升，尤其在90s时提升约15%。

**⚠️ 局限性**

局限在于仍需手动提供多视角参考图像、对极端动作或长时段的背景变换处理有限，以及在极长时间（数分钟以上）仍可能出现轻微漂移。

---

## 618. Case-Based Calibration of Adaptive Reasoning and Execution for LLM Tool Use

**arXiv ID:** 2605.15041 | [PDF](https://arxiv.org/pdf/2605.15041v1)

**作者:** Renning Pang `[一作]` (University of Electronic Science and Technology of China), Xiaosong Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 8751 | [OpenAlex ID](https://openalex.org/A5100780268)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于案例的CAST框架，用历史工具使用轨迹进行复杂度和失败模式建模，动态调节推理长度并校准结构化工具调用。

**💡 创新点**

创新点在于将案例推理与工具执行结构同时进行自适应校准，利用案例导出的复杂度/失败信号实现细粒度奖励设计。

**🔧 技术方法**

使用了案例构建、复杂度/失败分析、奖励塑形、GRPO强化学习等技术。

**📊 数据集**

使用BFCLv2和ToolBench两大工具使用基准。

**📈 对比分析**

与SFT、GRPO、Toolformer等基线对比，CAST在BFCLv2整体准确率提升至88.43%，ToolBench Pass/Win分别提升至80.67%/79.43%，显著减少推理长度。

**⚠️ 局限性**

局限在于对长程规划仍难以完全解决，依赖于已构建案例库的覆盖性和检索质量。

---

## 619. Multi-Agentic Approach for History Matching of Oil Reservoirs

**arXiv ID:** 2605.15028 | [PDF](https://arxiv.org/pdf/2605.15028v1)

**作者:** Linar Samigullin `[一作]` (Skoltech), Evgeny Burnaev `[通讯]` (Skoltech)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了PetroGraph多代理系统，实现油藏历史匹配的全流程自动化。

**💡 创新点**

其创新点在于将LLM驱动的多代理框架与检索增强生成（RAG）以及专门工具链相结合，动态选择优化方法与超参数，显著降低对专业工程师的依赖。

**🔧 技术方法**

技术包括大型语言模型（Qwen3.5‑397B）、Bayesian优化、Latin Hypercube采样、Gaussian Process surrogate、LangGraph框架以及ECLIPSE输入文件解析工具。

**📊 数据集**

实验数据来自OPM公开的SPE1、SPE9与Norne三种油藏模型，既有合成数据也有真实历史记录。

**📈 对比分析**

与传统手工或自动化流程比较，PetroGraph在SPE1、SPE9、Norne三例分别实现了95%、69%和13%的wNRMSE下降，表明性能显著提升。

**⚠️ 局限性**

局限在于对大规模复杂模型的收敛性有限，且RAG覆盖不足和缺乏长期记忆机制，需进一步改进以提升在真实大规模油藏上的效果。

---

## 620. Boosting Reinforcement Learning with Verifiable Rewards via Randomly Selected Few-Shot Guidance

**arXiv ID:** 2605.15012 | [PDF](https://arxiv.org/pdf/2605.15012v1)

**作者:** Kai Yan `[一作]` (University of Illinois Urbana-Champaign), Yu-Xiong Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 6946 | [OpenAlex ID](https://openalex.org/A5102952938)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FEST算法，在RLVR框架下利用仅128个随机演示来显著提升大型语言模型的推理表现。

**💡 创新点**

创新点在于将监督学习、对策学习和衰减权重三大组件融合成半在线DPO框架，并推出FEST-GRPO以解决序列级与标记级梯度不匹配问题。

**🔧 技术方法**

采用RLVR中的GRPO、半在线DPO（含自适应β）以及传统RL和SFT技术，并通过自适应权重实现对演示数据的有效利用。

**📊 数据集**

主要使用OpenR1-Math-46K数据集，随机抽取128个带链式思考（CoT）演示作为D_E；评估基准包括AIME25、AMC23、AIME24、MATH-500、OlympiadBench、Minerva以及MMLU-Pro。

**📈 对比分析**

与多种基线（RL、RL-G、SRFT、LUFFY、CHORD-ϕ、HPT、ReLIFT、MIFO）对比，FEST在所有指标上均优于基线，尤其在Pass@8和平均准确率上实现显著提升。

**⚠️ 局限性**

局限性：仅在1.5B参数模型上验证，聚焦数学推理任务，未探讨更大模型或代码/指令类任务的泛化能力。

---

## 621. DeepTokenEEG Enhancing Mild Cognitive Impairment and Alzheimers Classification via Tokenized EEG Features

**arXiv ID:** 2605.15009 | [PDF](https://arxiv.org/pdf/2605.15009v1)

**作者:** Thinh Nguyen-Quang `[一作]` (Hanoi University of Science and Technology), Hung Cao `[通讯]` (University of California)

**通讯引用:** 22200 | [OpenAlex ID](https://openalex.org/A5100403693)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种基于令牌化（Tokenizer）与轻量级残差编码器的DeepTokenEEG模型，用于通过EEG信号检测阿尔茨海默病（AD）与健康对照（HC）的二分类任务；

**💡 创新点**

核心创新在于将EEG信号先用Stationary Wavelet Transform分解为经典频段，再通过深度可分离的depthwise‑pointwise卷积令牌化，将时空信息压缩为低维离散令牌，随后使用多阶段残差卷积块（保持固定膨胀率）提取长程依赖，整体模型参数仅0.29M且推理速度最快；

**🔧 技术方法**

技术手段包括：数据预处理（channel harmonization、band‑pass 0.5‑45 Hz、重采样到128 Hz、Z‑score归一化）、SWT分频段、depthwise+pointwise tokenizer、3个残差块的DILATED卷积、跨通道CrossConv分支、双重池化+全连接分类头、Adam训练；

**📊 数据集**

使用五个公开EEG数据集（ADFTD、BrainLat、AD‑Auditory、ADFSU、APAVA）共274名受试者（180名AD、94名HC），分别评估单数据集与混合数据集；

**📈 对比分析**

与10个基线（TCN、Transformer、Conformer、TimesNet、Medformer、TS2Vec、BIOT、EEG2Rep、LaBraM、EEGPT）以及LEAD变体进行5折交叉验证比较；在单数据集上DeepTokenEEG在segment‑level和subject‑level均实现最高或接近最高准确率（A‑DFTD：87.12%/86.92%，BrainLat：85.71%/85.42%），参数量最少、GFLOPs最低、推理吞吐最高；在混合数据集上仍保持最佳性能，验证了跨域鲁棒性；

**⚠️ 局限性**

局限性包括：仅在二分类（AD vs HC）下验证，未覆盖多类别诊断；数据量相对有限，缺乏大规模多中心验证；对其他神经疾病的泛化能力待进一步评估；模型虽然轻量，但仍需在嵌入式硬件上进一步验证实际部署可行性。

---

## 622. Distance-Matrix Wasserstein Statistics for Scalable Gromov--Wasserstein Learning

**arXiv ID:** 2605.14981 | [PDF](https://arxiv.org/pdf/2605.14981v1)

**作者:** Ao Xu `[一作]` (Jilin University), Tieru Wu `[通讯]` (Jilin University)

**通讯引用:** 374 | [OpenAlex ID](https://openalex.org/A5090674860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Distance‑Matrix Wasserstein (DMW) 作为 Gromov–Wasserstein (GW) 距离的可扩展统计量，利用随机抽取有限子空间并对其距离矩阵分布进行 Wasserstein 计算，给出 GW 的下界和近似收敛性。

**💡 创新点**

创新点在于：①构建层级 DMW 统计量并证明其为 GW 的松弛且在样本量增大时收敛；②推导逆逼近定理，将 GW 与 DMW 的误差与经验 Wasserstein 误差关联；③设计切片、多尺度与核化实现，有效缓解高维度估计瓶颈。

**🔧 技术方法**

采用随机子空间采样、距离矩阵映射、Wasserstein 统计、切片投影（一维 OT）、多尺度平均、正定核化以及经验与理论收敛分析等技术。

**📊 数据集**

实验使用合成度量空间、随机块模型图以及十个 TU 图数据集（MUTAG、PTC_MR、BZR、COX2、PROTEINS、ENZYMES、IMDB-BINARY、IMDB-MULTI、NCI1、REDDIT-BINARY）。

**📈 对比分析**

与 GW、熵正则化 GW、最短路径直方图、WL、Graphlet、NetLSD 等基线比较；在图分类任务中与度量基线竞争，在两样本检验中保持 0.05 误差并随样本量提升功效；切片 DMW 在 1000 节点规模下保持可扩展性，显著降低计算成本。

**⚠️ 局限性**

主要限制是：仅使用距离信息，缺少节点标签或属性导致在 WL/Graphlet 主导的数据集上表现不佳；固定阶 DMW 并不完整；高阶估计受高维 Wasserstein 估计难度限制；多尺度权重需人工调节，缺乏完全自适应的理论。

---

## 623. InfoSFT: Learn More and Forget Less with Information-Aware Token Weighting

**arXiv ID:** 2605.14967 | [PDF](https://arxiv.org/pdf/2605.14967v1)

**作者:** Mahdi Sabbaghi `[一作]` (University of Pennsylvania), Hamed Hassani `[通讯]` (University of Pennsylvania)

**通讯引用:** 2434 | [OpenAlex ID](https://openalex.org/A5059354479)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 InfoSFT，一种基于信息熵和中等置信度权重的监督微调方法，旨在在学习新行为的同时减轻灾难性遗忘。

**💡 创新点**

从近似的 KL 预算下推导出最优权重规则，发现中等置信度 token 应获得最高权重，并给出可直接实现的近似规则，填补了传统 SFT 与 DFT 之间的空白。

**🔧 技术方法**

使用代理更新框架与策略梯度视角进行分析，结合二元熵优化与 token‑level 加权，实现仅需一行代码修改即可部署的权重公式。

**📊 数据集**

在多种任务上验证：数学（NuminaMath‑CoT、MATH500、AIMO‑Validation‑AMC）、代码（UltraFeedback、HumanEval、MultiPL‑E）、推理（OpenR1‑Math）、科学问答（SciKnowEval）和工具使用（ToolAlpaca）。

**📈 对比分析**

与标准 SFT 与 DFT 进行对比，使用 acc@1、pass@k（k=1,8）等指标；InfoSFT 在数学、代码与 CoT 任务上均显著提升性能，并在科学问答与工具使用中获得更优的学习‑遗忘曲线。

**⚠️ 局限性**

局限性包括需估计平均 token 置信度 p̅、对极低似然样本仍需预处理（如先行 SFT 阶段），对极稀有格式的学习效果有限，且依赖基准模型与超参数，尚未证明在所有设置下最优。

---

## 624. TopoPrimer: The Missing Topological Context in Forecasting Models

**arXiv ID:** 2605.15035 | [PDF](https://arxiv.org/pdf/2605.15035v1)

**作者:** Zara Zetlin `[一作]`, Maria Safi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对零售多业务线跨地区的需求预测任务，在冻结的预训练序列模型上加入拓扑特征以提升预测精度。

**💡 创新点**

首次将离线计算的TDA持久景观和Sheaf约束映射作为增量特征注入冻结模型，证明拓扑信息在结构丰富的商店空间中可显著提升预测性能。

**🔧 技术方法**

采用TDA持久景观编码（H0、H1、H2维度）与Sheaf约束映射两种拓扑编码方式，并结合Chronos、TimesFM等预训练序列模型进行适配器微调。

**📊 数据集**

使用日本、ALAC、ANZ、美国四个地区的真实销售历史数据，覆盖多业务线（iPhone、Mac、Watch、iPad、AirPods）构建交叉业务线的商店点云与图结构。

**📈 对比分析**

通过WAPE/MAPE指标，在零射击、原始适配器及多任务世界模型等基线下进行对比；在结构丰富的业务线（如JPN Watch、ANZ TimesFM Watch）中，拓扑增强模型可提升1%–5%的精度。

**⚠️ 局限性**

拓扑特征收益高度依赖商店需求空间的结构丰富性；在高填充率平滑需求线或美国地区，TDA/Sheaf可能导致性能下降；此外TVNet+Sheaf在多业务线训练中存在稳定性问题。

---

## 625. Guises and Perspectives: An Intentional and Hyperintensional Sketch

**arXiv ID:** 2605.15144 | [PDF](https://arxiv.org/pdf/2605.15144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 626. DriveCtrl: Conditioned Sim-to-Real Driving Video Generation

**arXiv ID:** 2605.15116 | [PDF](https://arxiv.org/pdf/2605.15116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 627. Generalized Priority-Aware Shapley Value

**arXiv ID:** 2605.15018 | [PDF](https://arxiv.org/pdf/2605.15018v1)

**作者:** Kiljae Lee `[一作]` (Ohio State University), Yuan Zhang `[通讯]` (Ohio State University)

**通讯引用:** 12831 | [OpenAlex ID](https://openalex.org/A5066093415)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种能够同时处理有向加权优先图（包括循环与加权）和个体软优先的Shapley值扩展——GPASV，用于评估机器学习模型或数据贡献。

**💡 创新点**

核心创新在于把传统的硬二元优先关系推广为加权循环优先，并通过Gibbs式优先惩罚与软优先的阶段式softmax实现；同时给出完整的公理化表述和极限分析。

**🔧 技术方法**

使用随机排序值框架、Gibbs式优先惩罚、阶段式选择因子、Metropolis‑Hastings采样、贪心初始化、缓存与自归一重要性采样等技术。

**📊 数据集**

实证数据来自MT‑Bench的合作评估、Chatbot Arena的双人偏好图以及开放源代码与付费模型的标签，用于对20个LLM的集合价值进行评估。

**📈 对比分析**

与经典Shapley、PSV、WSV、PASV等方法比较，GPASV在处理循环/加权优先时保持可解释性；蒙特卡罗估计在混合时间、精度和计算成本上与PASV相当或更优，且通过加速技巧显著降低了LLM调用次数。

**⚠️ 局限性**

主要局限是对效用函数的评估需要大量昂贵的LLM推理和聚合调用，且优先图的构建和稳定性依赖于人类偏好数据的可靠性。

---

## 628. Automating Bitvector and Finite Field Equivalence Proofs in Lean

**arXiv ID:** 2605.15163 | [PDF](https://arxiv.org/pdf/2605.15163v1)

**作者:** Elizaveta Pertseva `[一作]` (Stanford University), James Parker `[通讯]` (Galois)

**通讯引用:** 248 | [OpenAlex ID](https://openalex.org/A5059759222)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在 Lean 中实现了一套自动化技术，能够将有限域运算翻译为位向量运算并通过范围分析与位向量爆炸（bit‑blasting）完成等价证明。

**💡 创新点**

创新点在于提出了以自然数为中介的有限域→位向量翻译算子、针对 ZKP 形式化的专用范围分析算法以及利用变量依赖的案例分析来消除不必要的取模运算，从而显著降低了 SMT 方案的规模与复杂度。

**🔧 技术方法**

使用了 Lean 交互式定理证明器、内置位向量爆炸工具、定制的范围分析与翻译 tactic，以及 Lean 证明脚本中的多种自动化手段（如 `simp`, `ring`, `norm_num` 等）。

**📊 数据集**

在 Jolt zkVM 的 8 位和 32 位查表 arithmetization 以及 CirC 编译器生成的 0–32 位宽度的多参数 arithmetization 这两组真实工程数据集上进行评估。

**📈 对比分析**

与传统的 SMT 求解器（CVC5 全模式与分裂模式）以及 Lean 现有自动化（`linarith`, `nlinarith`, `zify` 等）对比，实验显示在大多数基准（尤其是 32 位及以上）上该方法在约 20 分钟/16 GB 的资源限制下能解决 19% 更多的 ZKP 等价验证问题，且能够提供内核检查的证明。

**⚠️ 局限性**

局限性包括：对含有字段专用运算（如逆元、负号）或位宽超出模数的情况支持不完善；范围分析仍为不完整的算法，某些依赖模式无法自动化；在极大位宽或字段常数过大的场景下仍会出现性能瓶颈。

---

## 629. MeMo: Memory as a Model

**arXiv ID:** 2605.15156 | [PDF](https://arxiv.org/pdf/2605.15156v1)

**作者:** Ryan Wei Heng Quek `[一作]` (National University of Singapore), Armando Solar-Lezama `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 7721 | [OpenAlex ID](https://openalex.org/A5010786661)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种称为Memory as a Model（MAM）的模块化框架，利用一个专门训练的较小模型（Memory模型）来存储并检索外部语料库中的知识，同时保持大型语言模型（LLM）参数不变；

**💡 创新点**

核心创新在于：①使用五步数据合成管道从原始文档生成“reflection QA”数据，捕获单文档事实与跨文档关系；②设计结构化多轮查询协议，使LLM能通过子查询从Memory模型检索信息并推理答案；③支持无权重访问、可插拔、检索成本与语料大小无关，并通过模型合并实现增量知识集成；

**🔧 技术方法**

关键技术包括：LLM的无监督事实提取与合并、验证与重写、实体显式化、跨文档合成；使用监督微调训练Memory模型；多轮协议中LLM作为黑盒进行交互；模型合并（Merge）技术用于持续学习；

**📊 数据集**

在三大基准上评测：BrowseComp‑Plus（多跳检索+推理）、NarrativeQA（长文档理解）和MuSiQue（多段落推理），并利用包含正负文档的检索噪声设置进行鲁棒性测试；

**📈 对比分析**

与BM25、NV‑Embed‑V2、HippoRAG2、Cartridges等检索+生成基线比较，MAM在NarrativeQA、MuSiQue上均明显领先，甚至在较大LLM（Gemini‑3‑Flash）上提升约12‑27个百分点；在检索噪声下仍保持稳定，表现出优越的鲁棒性；

**⚠️ 局限性**

局限性包括：训练成本较高（需微调Memory模型）；仅评估了有限的三个任务；Memory模型容量受限，无法处理极大语料库；对动态更新的支持尚不完善，需要进一步优化构建与检索效率。

---

## 630. Dual-Dimensional Consistency: Balancing Budget and Quality in Adaptive Inference-Time Scaling

**arXiv ID:** 2605.15100 | [PDF](https://arxiv.org/pdf/2605.15100v1)

**作者:** Rongman Xu `[一作]` (Xi'an Jiaotong University), Hang Yan `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 59486 | [OpenAlex ID](https://openalex.org/A5038497484)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Dual‑Dimensional Consistency (DDC) 框架，在 LLM 推理时通过联合调控路径采样宽度和深度，显著降低 token 消耗并提升推理准确率。

**💡 创新点**

创新点包括：①将宽度（inter‑path）和深度（intra‑path）视为耦合目标而非独立；②使用置信度加权的贝叶斯终止机制实现自适应采样停止；③引入趋势感知分层剪枝（Trend‑Aware Stratified Pruning），通过动态信号分析和结构不稳定度评分过滤幻觉；④利用 Tukey 算盒自适应阈值，消除手工阈值带来的调参成本。

**🔧 技术方法**

核心技术：贝叶斯序列决策与置信度加权更新、滑动窗口多粒度置信度（token/组/路径）、位置‑速度状态向量、协方差分解与主成分判定、结构不稳定度评分、置信度加权多数投票、动态阈值（Tukey 算盒）等。

**📊 数据集**

实验数据集：MATH‑500、AMC23、AIME24、AIME25、GPQA‑diamond；使用模型：Qwen‑3 系列（1.7B、4B、8B、32B）和 DeepSeek‑R1‑0528‑Qwen3‑8B。

**📈 对比分析**

与 Self‑Consistency、Adaptive‑Consistency、DeepConf‑Low/High 等基线对比，DDC 在所有模型和数据集上均取得更高准确率（如 AIME25 上 15.6% 的提升），同时 token 消耗平均降低 10‑27 倍，效率提升 3‑10 倍；相较于计算密集型方法（Predictive Decoding、ϕ‑Decoding），DDC 仍保持或超过准确率，并大幅降低计算开销。

**⚠️ 局限性**

局限性：①对置信度指标的依赖可能在某些任务或模型中表现不佳；②虽然趋势剪枝的计算开销很小，但仍需额外实现；③在非推理、生成型任务或模型置信度不可靠的场景下，框架效果需进一步验证；④实验聚焦于数值推理，需扩展到更广泛的推理场景。

---

## 631. SAGE3D: Soft-guided attention and graph excitation for 3D point cloud corner detection

**arXiv ID:** 2605.15088 | [PDF](https://arxiv.org/pdf/2605.15088v1)

**作者:** Batuhan Arda Bekar `[一作]` (Bahcesehir University), Barış Özcan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于Transformer和GNN的混合网络SAGE3D，用于航空LiDAR点云中的角点检测；

**💡 创新点**

核心创新包括Soft‑Guided Attention（在训练阶段将地面真值角点作为对数先验注入注意力）和Excitatory Graph Neural Network（仅正向信息传递、角点信号放大），并使用4D位置编码；

**🔧 技术方法**

技术上结合了Point Transformer、Set Abstraction/Feature Propagation层、CentroidGNN、DBSCAN后处理、软标签与距离加权焦点损失、Smooth‑L1回归等；

**📊 数据集**

在Building3D Tallinn和Entry‑Level数据集上进行训练与评估；

**📈 对比分析**

与现有方法相比，SAGE3D在Building3D Tallinn上取得0.134m的平均角点偏移、91.9%角点精度、74.4%召回率，CF1达到82.2%，优于PBWR、BWFormer等同级方法，并且只需单张RTX 4070即可训练完成；

**⚠️ 局限性**

主要限制包括依赖于训练时已知的角点先验（Soft‑Guided Attention仅在训练阶段有效），以及对稀疏角点的极端不平衡处理仍有提升空间。

---

## 632. Croissant Baker: Metadata Generation for Discoverable, Governable, and Reusable ML Datasets

**arXiv ID:** 2605.15079 | [PDF](https://arxiv.org/pdf/2605.15079v1)

**作者:** Rafi Al Attrach `[一作]` (Technical University of Munich), Tom Pollard `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 17808 | [OpenAlex ID](https://openalex.org/A5086791063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了 Croissant Baker，一款本地优先的开源命令行工具，用于从本地数据集目录自动生成符合 Croissant 1.1 规范的 JSON‑LD 元数据；

**💡 创新点**

核心创新在于：① 将结构推断与语义丰富分离，保证结构推断可追溯、可重现；② 通过模块化处理器注册表支持多种科学与领域专用格式（WFDB、DICOM、NIfTI、FHIR、Parquet 等）；

**🔧 技术方法**

采用 Python 实现，结合文件系统遍历、SHA‑256 校验、格式特定处理器、Apache Arrow 读取 Schema、以及 Croissant 验证库；

**📊 数据集**

使用了 140+ 多域数据集进行评测，包括 9 个本地测试集、MIMIC‑IV（886M 行 374 Parquet 文件）、Open Targets Parquet 数据、SMART Health IT FHIR、DICOM‑NIfTI 配对、51 个 OpenNeuro BIDS 数据集；

**📈 对比分析**

与现有平台生成、人工手工与 MCP 代理辅助方法对比；在 25 个 NeurIPS 2025 D&B 轨道抽样数据上实现 97.9% 语义类型一致率，Open Targets 97.4%，FHIR 97.8%，DICOM 100% 关键字匹配；单次生成时间从 0.74 s 到 32.2 s（MIMIC‑IV MEDS 版），可批量处理且无上传需求；

**⚠️ 局限性**

局限性包括：① 对未支持格式的文件会被跳过；② 结构推断基于采样的启发式类型推断，易出现模糊字段误判；③ 未自动推断跨表关系或外键；④ 需要人工审查语义字段；⑤ 对极大文件的内存消耗可能较高。

---

## 633. SOCC-ICP: Semantics-Assisted Odometry based on Occupancy Grids and ICP

**arXiv ID:** 2605.15074 | [PDF](https://arxiv.org/pdf/2605.15074v1)

**作者:** Johannes Scherer `[一作]` (Fraunhofer IVI), Henri Meeß `[通讯]` (Fraunhofer IVI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 SOCC-ICP，一种同时进行 LiDAR 里程计与 3D 语义占据格映射的实时框架，能够在线扫描对齐并生成可直接用于规划的语义占据网格。

**💡 创新点**

创新点：①首次在 LiDAR 里程计中使用单一 3D 语义占据格作为完整地图结构；②结合局部平面度自适应的点到点/点到面 ICP；③通过占据格 Raycasting 自动剔除动态物体；④在下采样、对应权重和残差计算中融入语义信息，提升精度与鲁棒性。

**🔧 技术方法**

核心技术：KISS‑ICP/GenZ‑ICP 迭代最近点、稀疏三层 voxel 结构 Radix、语义分割模型 LSK3DNet、Geman‑McClure 稳健核、占据概率更新、EM 语义统计、动态物体抑制、语义辅助下采样与对应权重。

**📊 数据集**

使用的数据集：KITTI Odometry Benchmark（含语义标签），MulRan，Newer College，Ground‑Challenge（Corridor1/2），SubT‑MRS Long Corridor。

**📈 对比分析**

对比方法：KISS‑ICP、GenZ‑ICP、MULLS、CT‑ICP、SuMa、Sa‑LOAM、SAGE‑ICP 等；性能表现：在 KITTI 上无语义 RTE 0.51% 与 0.49%（语义）与 state‑of‑the‑art 并列；在 MulRan、Newer College 上均优于同类方法；在 Corridor 低信息环境下，绝对/相对位姿误差最优或接近最优。

**⚠️ 局限性**

局限性：①实现仍为概念验证，计算量高于现有最轻量级方法；②缺乏闭环回环检测，无法消除长期漂移；③对语义分割模型依赖较大，分割误差会影响精度；④在极端动态或大尺度场景下鲁棒性有待提升；⑤占据格尺寸与分辨率需手动调参。

---

## 634. Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation

**arXiv ID:** 2605.15141 | [PDF](https://arxiv.org/pdf/2605.15141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 635. Natural Synthesis: Outperforming Reactive Synthesis Tools with Large Reasoning Models

**arXiv ID:** 2605.15131 | [PDF](https://arxiv.org/pdf/2605.15131v1)

**作者:** Frederik Schmitt `[一作]` (CISPA Helmholtz Center for Information Security), Bernd Finkbeiner `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合大规模推理模型与模型检测反馈的神经符号反应合成方法，可自动生成并验证Verilog电路，支持参数化合成与自然语言规范化

**💡 创新点**

通过将大规模推理模型与符号模型检查的counterexample引导循环相结合，显著提升合成效率并突破可判定性限制，同时实现从自然语言到正式规范再到硬件的端到端流程

**🔧 技术方法**

使用大型推理模型（Gemini 3.1 Pro、GPT‑5.5）、Yosys、AIGER、nuXmv（IC3）等工具构成的反馈循环

**📊 数据集**

SYNTCOMP 2025 1586个TLSF基准、57个参数化规范及其手工撰写的自然语言描述集

**📈 对比分析**

与SYNTCOMP冠军Strix/ltlsynt对比：GPT‑5.5在相同时间/内存下解决170个更多实例；在参数化合成中平均解决35–41实例；在自然语言合成中约30/57实例通过自动形式化或直接合成验证

**⚠️ 局限性**

仍受模型规模与推理token预算限制；参数化合成无法在正面保证完整性；自然语言合成的语义漂移与形式化不一致导致部分失效

---

## 636. MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory

**arXiv ID:** 2605.15128 | [PDF](https://arxiv.org/pdf/2605.15128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 637. APWA: A Distributed Architecture for Parallelizable Agentic Workflows

**arXiv ID:** 2605.15132 | [PDF](https://arxiv.org/pdf/2605.15132v1)

**作者:** Evan Rose `[一作]` (Northeastern University), Alina Oprea `[通讯]` (Northeastern University)

**通讯引用:** 5943 | [OpenAlex ID](https://openalex.org/A5035574749)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Agent-Parallel Workload Architecture（APWA），一种专门针对大规模可并行化任务的分布式多代理系统；

**💡 创新点**

通过三层抽象（manager、worker、executor）、子任务模板、数据表以及动态代理能力，实现了高效的任务拆分、资源共享和并行执行；

**🔧 技术方法**

核心技术包括基于Ray的分布式调度、LLM引导的子任务规划、数据表元数据接口和可动态配置的能力注册表；

**📊 数据集**

在AI4Privacy PII-300k、SchemaBench以及基于Romeo & Juliet、The Dynasts和The History of the Decline and Fall of the Roman Empire的层级摘要数据集上进行评估；

**📈 对比分析**

与直接LLM、Magentic-One和MegaAgent三种基线相比，APWA在成功率、结构/语义分数、墙钟时间和成本上均表现更好，尤其在大输入规模下保持低失败率和短运行时间；

**⚠️ 局限性**

仍受限于LLM上下文窗口限制、对大规模数据表的元数据管理需求以及多代理动态创建的开销，且在非并行化任务或极端数据规模场景下的适用性待进一步验证。

---

## 638. Causal Foundation Models with Continuous Treatments

**arXiv ID:** 2605.15133 | [PDF](https://arxiv.org/pdf/2605.15133v1)

**作者:** Christopher Stith `[一作]` (Layer 6 AI), Rahul G. Krishnan `[通讯]` (University of Toronto)

**通讯引用:** 2494 | [OpenAlex ID](https://openalex.org/A5073514348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了第一个连续处理变量的因果基础模型CCPFN，能够在未见任务上直接重建个体处理-响应曲线；

**💡 创新点**

创新点包括：1）设计三层MLP的先验，生成满足无混杂性和正性假设的连续处理数据生成过程；2）引入专门的治疗编码器和上下文学习，利用in-context学习实现贝叶斯后验的摊销推断；

**🔧 技术方法**

使用Transformer‑based PFN架构、三重MLP先验、in‑context学习、非线性治疗编码器以及量化后验分布的离散直方图损失；

**📊 数据集**

在14个连续处理场景中进行验证，包含8个验证场景（如ACIC2016、Criteo、Hillstrom、Twins）和6个测试场景（MVICU、Debt、News、NewsHet、TCGA、Warfarin），并使用合成/半合成生成的数据；

**📈 对比分析**

与ADMIT、SCIGAN、DRNet、VCNet、GPS、CausalForest、DML等多种基线以及Tabular Foundation Models（TabDPT、TabPFN、TabICL）比较，CCPFN在MISE和DPE指标上均取得首位或近乎首位，表现最优；

**⚠️ 局限性**

局限性包括：1）依赖无混杂性与正性假设，实际验证困难；2）在连续处理区间上正性难以完全满足；3）对大特征维度数据的嵌入维度有限，压缩可能导致信息损失；4）缺乏大规模连续处理的真实评测数据集。

---

## 639. CLOVER: Closed-Loop Value Estimation \& Ranking for End-to-End Autonomous Driving Planning

**arXiv ID:** 2605.15120 | [PDF](https://arxiv.org/pdf/2605.15120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 640. Proposal and study of statistical features for string similarity computation and classification

**arXiv ID:** 2605.15110 | [PDF](https://arxiv.org/pdf/2605.15110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 641. Understanding How International Students in the U.S. Are Using Conversational AI to Support Cross-Cultural Adaptation

**arXiv ID:** 2605.15127 | [PDF](https://arxiv.org/pdf/2605.15127v1)

**作者:** Laleh Nourian `[一作]` (Rochester Institute of Technology), Garreth W. Tigwell `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 805 | [OpenAlex ID](https://openalex.org/A5085688357)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对美国国际学生使用对话式AI支持跨文化适应的使用模式、动机与边界进行问卷和访谈混合研究，并提出针对性设计建议。

**💡 创新点**

创新性在于将用户中心的定量-定性方法结合，揭示AI被视为短期“急救”工具而非长期伴侣的现象，并提出多维度的AI支持设计框架。

**🔧 技术方法**

研究采用在线调查问卷、半结构式访谈、统计分析（Spearman、Mann‑Whitney、Wilcoxon、Kruskal‑Wallis）以及主题分析技术。

**📊 数据集**

使用的“数据集”包括60名美国国际学生的调查数据和14名受访者的访谈转录；研究亦引用公开的学生挑战类别与AI感知文献。

**📈 对比分析**

方法上通过描述性统计与相关分析阐明不同挑战域与AI使用频率的关系，利用Wilcoxon与Kruskal‑Wallis检验不同领域AI效用差异；未进行模型性能评估。

**⚠️ 局限性**

局限性包括样本仅限美国第一时间国际学生、样本量相对较小、依赖自我报告可能存在偏差、缺乏跨文化对比、以及研究未检验实际AI工具实现效果。

---

## 642. Why Neighborhoods Matter: Traversal Context and Provenance in Agentic GraphRAG

**arXiv ID:** 2605.15109 | [PDF](https://arxiv.org/pdf/2605.15109v1)

**作者:** Riccardo Terrenzi `[一作]` (University of Southern Denmark), Serkan Ayvaz `[通讯]` (University of Southern Denmark)

**通讯引用:** 1454 | [OpenAlex ID](https://openalex.org/A5075890706)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建基于2WikiMultiHopQA的知识图谱，并对Agentic GraphRAG系统进行六种图谱消融实验，探究最终引用集是否足以解释答案生成的证据基础。

**💡 创新点**

创新点在于将引用可信度定义为轨迹级问题，提出图谱消融方法评估引用的必要性、充分性与完整性，并揭示遍历上下文、邻域结构与未引用实体对答案的行为影响。

**🔧 技术方法**

使用技术包括大语言模型（LLM）、检索增强生成（RAG）、图谱RAG与Agentic GraphRAG框架，结合Leiden社区检测、实体关系抽取以及自定义工具调用实现图遍历与文本检索。

**📊 数据集**

使用的数据集为从2WikiMultiHopQA的开发集挑选的30个多跳问题，构建的知识图谱包含1,815个实体、1,692条关系和7个社区。

**📈 对比分析**

对比方法通过六个系统（LLM、RAG、GraphRAG、Agentic GraphRAG及其两种约束变体）在原始与消融图谱下的准确率、引用文本单元与访问实体等指标进行评估；实验结果显示引用实体对答案必要但不充分，完整引用不足以恢复原答案，准确率在不同消融条件下显著变化。

**⚠️ 局限性**

限制主要包括样本规模小（仅30题）、使用的是人工构造的知识图谱而非大型真实知识库，且未在多领域多维度数据上验证方法的普适性。

---

## 643. Loop Termination and Generalized Collatz Sequences

**arXiv ID:** 2605.15094 | [PDF](https://arxiv.org/pdf/2605.15094v1)

**作者:** Mishel Carelli `[一作]` `[通讯]` (CISPA Helmholtz Center for Information Security), Mishel Carelli (CISPA Helmholtz Center for Information Security)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究单变量整数线性约束循环（SLC）的终止性问题，证明若满足弱 Collatz 可达性猜想，则可在多项式时间内判定终止；并证明若循环存在周期，则必存在长度不超过 2 的周期。

**💡 创新点**

创新点在于将 SLC 终止性与广义 Collatz 序列关联，提出并证明了弱 Collatz 可达性猜想（在模 2 情况下）以及其对 SLC 终止性的影响；同时给出了单变量循环的周期长度上界与自避踪迹判定的完整几何分析。

**🔧 技术方法**

主要技术包括整数线性规划、凸几何（Minkowski–Weyl 定理）、递推与几何投影、以及与 Collatz 序列的代数映射关系。

**📊 数据集**

本文未使用实验数据集，全部为理论证明与算法设计。

**📈 对比分析**

与以往只能给出不完整或启发式的终止分析方法不同，本文给出在假设可达性猜想成立时的多项式时间判定方案，理论上可直接判定所有单变量 SLC 的终止性。

**⚠️ 局限性**

主要局限在于判定结果依赖于尚未解决的弱 Collatz 可达性猜想（尤其是模大于 2 的情况），且目前仅针对单变量循环，尚未推广至多变量或更一般的线性约束循环。

---

## 644. ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World

**arXiv ID:** 2605.15081 | [PDF](https://arxiv.org/pdf/2605.15081v1)

**作者:** Ziyin Zhang `[一作]` (Shanghai Jiao Tong University), Rui Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 70444 | [OpenAlex ID](https://openalex.org/A5100431408)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出3D-ML框架（MEL+MLL+MRL）并训练一系列ML-Embed嵌入模型，解决文本嵌入的高计算成本、语言覆盖不足和缺乏透明度问题。

**💡 创新点**

创新点在于：①将Matryoshka嵌入、层级和表示学习三者统一到一体化框架，实现参数、深度和存储三维高效；②MEL通过低秩分解和嵌套训练显著减少嵌入层参数；③构建覆盖282种自然语言+40+编程语言的海量公开数据集；③开源所有模型、数据与训练代码。

**🔧 技术方法**

技术包括：Matryoshka Embedding Learning（MEL）、Matryoshka Layer Learning（MLL）、Matryoshka Representation Learning（MRL），基于Qwen3因果语言模型的Transformer结构，SVD低秩分解，双阶段对比学习与指令微调。

**📊 数据集**

使用50M样本的多语言数据集，来自121个公开来源，覆盖282种自然语言和40+编程语言，重点关注低资源语言和多样化任务。

**📈 对比分析**

在17个MTEB基准（共430任务）上评估，8B模型在9/17基准上取得SOTA，尤其在波兰、越南、印度语等低资源语言大幅提升；与同规模EmbeddingGemma、Qwen3-Embedding比较，在多语言、代码、欧洲、斯堪的纳维亚等子基准上表现更优。

**⚠️ 局限性**

局限性包括：在英语、中文等高资源任务上仍略逊；对极低资源语言覆盖有限；压缩后低秩模型在极端压缩时性能衰减；主要针对因果解码器架构，扩展到其他架构需进一步验证。

---

## 645. Due Process on Hold: A Queueing Framework for Improving Access in SNAP

**arXiv ID:** 2605.15165 | [PDF](https://arxiv.org/pdf/2605.15165v1)

**作者:** Andrew Daw `[一作]` (University of Southern California), Angela Zhou `[通讯]` (University of Southern California)

**通讯引用:** 1746 | [OpenAlex ID](https://openalex.org/A5101466158)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种针对 SNAP 申请中心的排队模型，结合了呼叫者放弃、重拨以及再认证的循环反馈，构建了流模型并给出稳态闭式解，进一步开发了可视化仪表盘供决策者评估不同改进措施；

**💡 创新点**

创新点在于首次把自发重拨导致的内生拥塞纳入排队分析，推导出含重拨的 Erlang‑A 变体，并利用流模型在大规模系统下得到精确的稳态指标，进而揭示现行 Staffing 指导存在的严重不足；

**🔧 技术方法**

核心技术包括马尔可夫排队网络建模、流（fluid）逼近与大数定律、参数估计（基于法院公开数据的回归与稳态匹配）以及闭式稳态性能指标的解析推导；

**📊 数据集**

使用了美国明尼苏达州 DHS 通过 Holmes v. Knodell 案件公开的每日呼叫中心运营数据（呼叫量、等待时间、放弃率、排班等），并在此基础上为四个不同规模中心单独校准参数；

**📈 对比分析**

与传统 Erlang‑A 计 Staffing 指导对比，结果显示前者在面对重拨导致的内生拥塞时显著低估所需人员（最高达 84% 低估）；在模拟不同人员、处理时长、再认证周期等改进时，模型展示了各指标（等待时间、程序性拒绝率、重拨拥塞等）的弹性与交互效应，支持组合提升带来更大效益；

**⚠️ 局限性**

局限包括：1) 采用流模型只给出一阶近似，未捕捉随机波动与分布特征；2) 参数估计受限于汇总数据，缺乏细粒度的呼叫时序与重拨记录；3) 假设所有重拨与再认证路径均可被同一参数化描述，可能忽略个体差异；4) 对政策干预的量化假设（如再认证周期延长）需进一步实证验证。

---

## 646. Pelican-Unified 1.0: A Unified Embodied Intelligence Model for Understanding, Reasoning, Imagination and Action

**arXiv ID:** 2605.15153 | [PDF](https://arxiv.org/pdf/2605.15153v1)

**作者:** Yi Zhang `[一作]` (Beijing Innovation Center of Humanoid Robotics), Xiaozhu Ju `[通讯]` (Beijing Innovation Center of Humanoid Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种统一的全能型嵌入式基础模型 Pelican‑Unified 1.0，将理解、推理、想象和行动四大能力整合在单一闭环循环中，并通过一次训练实现四个功能共同优化。

**💡 创新点**

创新点在于提出并实现了“统一范式”——共享内部表示、相互约束、联合训练的闭环结构；将 VLM 的推理过程直接转化为可观测的“思维链”，并让同一潜在向量同时驱动未来视频与动作的联合扩散生成，突破了模块化流水线的分割。

**🔧 技术方法**

技术包括：1) Qwen3‑VL 4B 视觉‑语言模型做统一理解/推理；2) Wan2.2‑5B 基于 diffusion transformer 的统一未来生成器（同时生成视频与动作）；3) 联合训练目标（语言、视频、动作三者损失相互 Back‑prop）；4) 稳定的时间连续流匹配学习和跨模态共享 Transformer 计算。

**📊 数据集**

使用多模态大规模机器人交互数据集（含观测、指令、推理、动作、未来视频），并在公开基准上评测：VLM 8 任务、RoboTwin 双臂模拟、WorldArena 世界建模、真实 UR5e 及 Tienkung 人形机器人工业控制面板等。

**📈 对比分析**

与多种模块化和专门化基线（OpenVLA、MolmoAct、AIM、MotuBrain、Wan2.6 等）进行对比。Pelican‑Unified 在 VLM 8 基准平均 64.7 分（同规模模型最高），WorldArena EWM 得分 66.03（榜首），RoboTwin 成功率 93.5%（排名第二），在真实机器人零样本与组合泛化任务上也显著优于基线。

**⚠️ 局限性**

局限性：1) 需要大量包含完整闭环标签（观察+推理+动作+未来视频）的高质量数据；2) 模型规模较大，推理速度和实时性有待提升；3) 对极长时序或高度交互复杂场景的鲁棒性仍有限。

---

## 647. Widening the Gap: Exploiting LLM Quantization via Outlier Injection

**arXiv ID:** 2605.15152 | [PDF](https://arxiv.org/pdf/2605.15152v1)

**作者:** Xiaohua Zhan `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11358 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在权重块中注入极大outlier，使得量化后大部分权重被逼近为零，从而在量化后激活预先植入的恶意行为。

**💡 创新点**

首次实现对多种先进优化量化方法（GPTQ、AWQ、HQQ、SINQ等）通用的高成功率量化条件攻击，利用outlier导致权重归零的普遍性质。

**🔧 技术方法**

使用双目标（dual‑objective）微调、outlier插入、量化代理模型细化等技术，辅以KL正则、交叉熵损失，并在Transformer的FFN块中实现攻击切换。

**📊 数据集**

使用Llama3.1‑8B、Qwen2.5‑7B、Mistral‑7B三大LLM，以及C4校准集、LLM‑LAT等用于训练与评估的公开数据集。

**📈 对比分析**

与先前仅针对简单量化方法的攻击相比，本研究在多种量化方案下（GPTQ、AWQ、HQQ、SINQ、GGUF k‑/i‑quant等）均达成>90%攻击成功率，且在量化前模型保持>80%（Llama、Qwen）或>90%（Mistral）原始性能。

**⚠️ 局限性**

对极大outlier的依赖可能导致全精度模型的轻微性能下降；对Mistral的效果不如其它模型；在极大scale下现有噪声防御无效，缺乏统一、稳健的防御策略。

---

## 648. Complete Local Reasoning About Parameterized Programs Over Topologies

**arXiv ID:** 2605.15143 | [PDF](https://arxiv.org/pdf/2605.15143v1)

**作者:** Ruotong Cheng `[一作]` (University of Toronto), Azadeh Farzan `[通讯]` (University of Toronto)

**通讯引用:** 1606 | [OpenAlex ID](https://openalex.org/A5016276143)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种针对无限状态参数化并发程序在多种通信拓扑上的安全验证，提出了完整的基于通用 Ashcroft 不变量的组合式证明方法，并实现了相应工具。

**💡 创新点**

通过拓扑族的对称性引入基数为 k+1 的有限基，证明了 cut‑off 定理，表明只需验证有限个小程序即可保证整个参数化族的安全；同时提出了归一化和优化的 Ashcroft 不变量搜索框架。

**🔧 技术方法**

组合式验证、Ashcroft 不变量、结构归一化、量化类型枚举、约束 Horn 子句（CHC）编码、SMT 求解器（Z3/Spacer、Golem）等技术。

**📊 数据集**

在 30 个不同拓扑（环、线、网格、受限深度树、二叉树）上设计的基准程序；以及从文献中提取的若干经典案例。

**📈 对比分析**

与基线（无优化）比较，启用 OPN、DPG 等优化后在 26 个基准中最多解决 16 个，整体求解时间显著下降，优化组合提升了 13 个；但在某些大网格或树实例中仍因编码过大或非线性推理导致失败。

**⚠️ 局限性**

后端 CHC 求解器的可扩展性和不可预测性；基数增大导致基的规模爆炸；某些程序不满足通用量化不变量，需要更丰富的不变量或虚拟变量。

---

## 649. Usable but Conventional: An Empirical Study on the UX of AI-Generated Interface Prototypes

**arXiv ID:** 2605.15124 | [PDF](https://arxiv.org/pdf/2605.15124v1)

**作者:** Karoline Romero `[一作]` (Universidade Estadual de Maringá), Guilherme Guerino `[通讯]` (Universidade Estadual de Maringá)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5004014879)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对比了10个桌面系统原型（5人工、5 GenAI生成）的用户体验，通过UEQ‑S调查92名计算机科学学生，分析了实用性与感性维度。

**💡 创新点**

创新在于证明原型的作者身份（人类VS AI）并未显著决定UX，并揭示GenAI主要提升实用性但欠缺感性创新。

**🔧 技术方法**

使用了Figma/UX Pilot、Uizard、Stitch、Lovable、Magic Patterns等GenAI设计工具。

**📊 数据集**

基于自定义提示语创建的10个原型样本，并收集92名学生的UEQ‑S评分。

**📈 对比分析**

采用within‑subject设计，使用UEQ‑S量表并做Shapiro‑Wilk、Friedman与Durbin‑Conover检验，结果显示Figma/UX Pilot原型最佳，AI与人工原型差异不大但存在显著差异。

**⚠️ 局限性**

局限包括样本仅为学生、原型静态未交互、提示长度受限、工具免费版功能受限、全部人工原型由一名研究者完成，影响泛化与多样性。

---

## 650. CoralLite: μCT Reconstruction of Coral Colonies from Individual Corallites

**arXiv ID:** 2605.15093 | [PDF](https://arxiv.org/pdf/2605.15093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 651. CoCo-InEKF: State Estimation with Learned Contact Covariances in Dynamic, Contact-Rich Scenarios

**arXiv ID:** 2605.15122 | [PDF](https://arxiv.org/pdf/2605.15122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 652. Talk is (Not) Cheap: A Taxonomy and Benchmark Coverage Audit for LLM Attacks

**arXiv ID:** 2605.15118 | [PDF](https://arxiv.org/pdf/2605.15118v1)

**作者:** Karthik Raghu Iyer `[一作]` (Palo Alto Networks), Alexey A. Shvets `[通讯]` (Palo Alto Networks)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过自动化提取与归纳构建了一个 507 节点的 LLM 推理攻击分类学，并基于 STRIDE 设计 4×6 的目标×技术矩阵，利用该矩阵对六个公开基准进行覆盖性审核，发现仅覆盖 25% 的攻击面且三大基准互不重叠。

**💡 创新点**

创新点在于：① 提供可复用的基准覆盖审计框架；② 结合大规模自动化抽取（超 900 篇论文）生成 2,521 个独立攻击组；③ 将攻击按目标与防御层次分离，形成直观的矩阵视图，从宏观角度揭示基准盲点。

**🔧 技术方法**

使用技术包括：Gemini‑3.1 Pro 的结构化 JSON 抽取、层级化模糊/语义匹配分类、人工验证与一致性评估；以及构建公开数据集、矩阵统计与可视化脚本。

**📊 数据集**

数据集来源于 932 篇 2023‑2026 年 arXiv 论文，经过抽取得到 6,366 次攻击提及，归一化后得到 2,521 个攻击组，构成了 507 节点的攻击学分类。

**📈 对比分析**

对比方法：在六个公开基准（HarmBench、InjecAgent、AgentDojo 等）上进行覆盖率映射，结果显示最大覆盖 6/24 个单元格（约 25%）；人类评估显示抽取准确率 92%（95% CI: 87.4%–95.0%），抽取召回率 89.4%。

**⚠️ 局限性**

局限性包括：① 依赖 Gemini‑3.1 Pro 进行抽取与分类，可能遗漏罕见攻击；② 抽取召回率 89.4% 仍存在漏检；③ 语料仅限英文 arXiv 论文，可能偏向热门研究领域；④ 基准覆盖仅对公开基准做映射，未覆盖商用或自定义红队测试。

---

## 653. From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents

**arXiv ID:** 2605.15104 | [PDF](https://arxiv.org/pdf/2605.15104v1)

**作者:** Md Tahmid Rahman Laskar `[一作]` (Dialpad Inc), Shashi Bhushan TN `[通讯]` (Dialpad Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可验证的框架，将文本工具调用基准通过TTS转换为可控的音频评估

**💡 创新点**

创新点在于保留原始工具schema与金标，提供文本-音频配对的第一阶段诊断，并与LLM评判相结合

**🔧 技术方法**

使用文本到语音(TTS)、环境噪声注入、多模态LLM推理以及LLM-as-judge技术

**📊 数据集**

使用Confetti和When2Call两个公开文本工具调用基准，并通过三种TTS模型生成音频

**📈 对比分析**

对比七种全模态LLM与文本LLM，在清晰与噪声音频、不同声学、多模型规模下的工具调用准确率；Gemini-3.1-Flash-Live在Confetti最高，GPT-Realtime-1.5在When2Call最高，文本到语音的性能损失因模型与任务而异

**⚠️ 局限性**

局限在于仅评估两组基准和有限模型，音频为合成而非自然对话，未覆盖更广泛的领域与真实环境噪声

---

## 654. Improving Multi-turn Dialogue Consistency with Self-Recall Thinking

**arXiv ID:** 2605.15102 | [PDF](https://arxiv.org/pdf/2605.15102v1)

**作者:** Renning Pang `[一作]` (University of Electronic Science and Technology of China), Xiaosong Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 8751 | [OpenAlex ID](https://openalex.org/A5100780268)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了自回想思维（SRT）框架，用内部回忆标签主动检索和利用多轮对话历史以提升一致性与准确性。

**💡 创新点**

创新点在于把回忆过程内化为可解释的 <HIS> 标签，结合依赖构造、监督微调与可验证奖励的 RL，避免外部检索、摘要压缩导致的细节丢失。

**🔧 技术方法**

技术手段包括依赖构造阶段、监督微调（SFT）以学习回忆标记、GRPO 强化学习与可验证的奖励函数、以及自回想提示机制。

**📊 数据集**

使用了自制的 SRQA 数据集（5,000条含 8–32 回合长距离依赖），以及公开的 SimpleQA 与 CoQA 作为基准测试。

**📈 对比分析**

与 6 款基线（RQ‑RAG、QRMeM、LD‑Agent、OPRO、Coconut、SoftCoT）对比，SRT 在 SRQA、CoQA、SimpleQA 上分别获得 78.4/9.1s、84.0/8.8s、56.1/8.7s 的最佳 F1/延迟组合，明显优于对手。

**⚠️ 局限性**

局限性包括：RL 阶段对奖励设计敏感，回忆错误仍随对话长度增长；依赖构造依赖外部模型（如 Claude 3.7 Sonnet），对资源和可复现性有一定要求。

---

## 655. Evidential Reasoning Advances Interpretable Real-World Disease Screening

**arXiv ID:** 2605.15171 | [PDF](https://arxiv.org/pdf/2605.15171v1)

**作者:** Chenyu Lian `[一作]` (Hong Kong Polytechnic University), Jing Qin `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 21729 | [OpenAlex ID](https://openalex.org/A5100662807)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于证据推理的疾病筛查框架，利用双知识库检索历史病例的区域证据，实现可解释性和性能提升。

**💡 创新点**

创新点包括：① 双知识库的区域证据检索实现回溯解释；② 对比检索实现训练无关筛查；③ 引入临床导向的评估框架和指标（Spe@X%R、CSR）。

**🔧 技术方法**

使用视觉基础模型（ViT）提取区域特征，构建正常/异常两份知识库，采用k‑NN检索、交叉注意力+自注意力进行证据意识推理；训练无关版本采用对比检索生成异常图。

**📊 数据集**

在眼科、放射科和皮肤科共十个公开数据集（JSIEC、RIADD、EDDFS、BRSET、CheXpert、MIMIC‑CXR、Derm12345、HAM10000、BCN20000、PAD‑UFES‑20）进行实验。

**📈 对比分析**

与基线（FM、PatchCore、SCRD4AD、EDC、SimpleNet、DRA、CIPL等）对比，作者方法在AUROC、AP、Spe@X%R、CSR等临床指标上均表现最佳，尤其在Spe@X%R和CSR上提升显著。

**⚠️ 局限性**

局限性包括：① 需要在更多模态、3D影像及更细粒度筛查任务中验证；② 知识库更新与隐私保护的实际部署挑战；③ 对k‑NN检索速度与内存的依赖，扩展到大规模数据时可能受限。

---

## 656. Does Synthetic Layered Design Data Benefit Layered Design Decomposition?

**arXiv ID:** 2605.15167 | [PDF](https://arxiv.org/pdf/2605.15167v1)

**作者:** Kam Man Wu `[一作]` (HKUST), Qifeng Chen `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一套完全合成的图形设计分层数据集SynLayers，并基于此训练和评估了图层分解模型；

**💡 创新点**

主要创新在于通过多源合成、低重叠布局和VLM生成的文本监督，构建可大规模、层数均衡的训练集，同时通过VLM自动预测图层框实现推理输入的全自动化；

**🔧 技术方法**

采用CLD分层框架、Qwen3-VL作为VLM实现标题与框预测、Vision‑Language模型生成层级描述、低重叠算法控制布局，以及LoRA微调等技术；

**📊 数据集**

使用Synthetic dataset SynLayers（500k样本、最多52层）与现有Crello、PrismLayersPro（各约20k样本）进行对比；

**📈 对比分析**

在Layer PSNR、Composite PSNR、FID、IoU等指标上，SynLayers 18k样本优于PrismLayersPro基线（Layer PSNR+1.01、Composite PSNR+0.83），随数据规模增至50k后性能趋于平稳；与Qwen‑Image‑Layered相比，SynLayers在层级与整体图像质量上均显著提升；

**⚠️ 局限性**

局限性包括合成数据对复杂非规则元素的定位与真实混合效果的覆盖不足、VLM检测误差可能导致分层不精确，以及模型在真实多样化场景下的泛化仍需进一步验证。

---

## 657. Constructive higher sheaf models with applications to synthetic mathematics

**arXiv ID:** 2605.15126 | [PDF](https://arxiv.org/pdf/2605.15126v1)

**作者:** Thierry Coquand `[一作]` (University of Gothenburg), Christian Sattler `[通讯]` (University of Gothenburg)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5103055014)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在构造主义框架下建立了可构造的高阶层式模型，为依赖类型理论（尤其是可验证的同伦类型理论）提供了新的内在模型，并通过“cobar 模式”将一阶层式语义提升到层式语义，从而支持高阶层式同伦几何和Stone 对偶的合成公理化。

**💡 创新点**

创新点在于：①首次将 lex 模式和 cobar 构造与层式本体结合，得到在内在层式模型中的完整可构造模态；②给出了从内在范畴到层式预设范畴的构造，证明了在该模型中满足 Blechschmidt 的对偶性公理；③实现了在构造主义中对合成代数几何的验证，弥补了先前仅在经典公理体系中可行的空洞。

**🔧 技术方法**

主要技术包括：构造性立方模型（cubical models）、伪映射与左正模态（lex modalities）以及它们的笛卡尔余延（cobar modality）；利用内外推理（internal/external yoga）把层式语义转化为内在证明；采用左正模态的闭包性质、三角定理与构造性棕榈定理等，确保模态在层式预设模型中保持对偶性与可构造性；并结合层式同伦代数的基本构造（如自由模型、等价化、指数化）完成合成公理化验证。

**📊 数据集**

该工作不依赖任何具体数据集，全部是理论构造与证明；所用的“数据”为内部模型中的对象和映射，均在构造主义元理论中定义。

**📈 对比分析**

相较于传统的类型理论模型（如 simplicial 方式的 Shulman 模型）和经典的类拓扑模型，本文提供了完全可构造的层式模型，能够在无选择公理的环境下验证合成公理；理论层面上的优势在于实现了可构造的同伦几何和对偶性公理化；性能方面，模型构造是构造性的，能够在可证明性证明助手（如 Coq、Agda）中形式化，但目前尚未完成完整的实现验证，因而性能评估仍为理论预期。

**⚠️ 局限性**

局限性包括：①需要假设内部范畴满足一定的 fibrancy 与连接性（例如 C_1 为 fibrant family），这在某些应用中可能难以验证；②模型的构造高度依赖于可构造集合论（Constructive ZF 或型理论），无法直接迁移到经典公理体系；③虽然证明可构造性，但完整的形式化实现仍待完成，尚未在现有证明助手中得到广泛测试；④在更复杂的同伦层式结构（如高阶层式模态组合）时，证明的复杂度可能急剧上升。

---

## 658. Learning from Language Feedback via Variational Policy Distillation

**arXiv ID:** 2605.15113 | [PDF](https://arxiv.org/pdf/2605.15113v1)

**作者:** Yang Li `[一作]` (Salesforce AI Research), Shafiq Rayhan Joty `[通讯]` (Salesforce AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在语言反馈驱动的强化学习框架中，提出Variational Policy Distillation (VPD)，通过变分EM方法实现教师和学生的协同进化；在每轮E‑step中主动更新教师以更好解释文本批判，在M‑step中将改进后的教师分布蒸馏回学生。

**💡 创新点**

创新点包括：①将自监督蒸馏建模为变分推理并使用EM迭代；②在E‑step采用无配对偏好学习（BCO）主动优化教师；③动态信任区间将教师先验实时对齐到当前学生，保证蒸馏目标可达；④统一权重网络消除多模型内存开销。

**🔧 技术方法**

核心技术：变分推理、Expectation–Maximization算法、无配对偏好学习、动态参考先验、共享权重的单模型蒸馏、KL分布对齐、对话式批判生成。

**📊 数据集**

实验数据集涵盖多种推理任务：LiveCodeBench (代码生成)、SciKnowEval (科学推理)、Dapo-Math/Math500/AIME24/25/AMC23 (数学推理)、OLMo3-7B-Instruct 等；还使用了不同规模的 Qwen3-1.7B/8B、OLMo3-7B 等模型。

**📈 对比分析**

与GRPO、SDPO、SDPO+RL（Adv Reshape/Reweight/Joint Loss）等基线对比，VPD 在代码生成、科学推理以及自我批判场景均显著提升：例如 LCBv6 上 Qwen3-8B 的通过率从 45.6% 提升到 49.6%；SciKnowEval 上平均得分从 69.8%/74.4%/65.7% 提升到 74.3%/77.1%/70.8%。

**⚠️ 局限性**

局限性：在基模型冷启动和严谨数学推理任务中仍落后于纯稀疏 RL；教师能力受限于共享权重，难以捕捉极复杂的诊断信息；极噪声或模糊的文本批判可能导致教师误导，进而影响学习。

---

## 659. PickleFuzzer: A Case Study in Fuzzing for Discrepancies Between Python Pickle Implementations

**arXiv ID:** 2605.15084 | [PDF](https://arxiv.org/pdf/2605.15084v1)

**作者:** Justin Applegate `[一作]` (Brigham Young University), Andreas Kellas `[通讯]` (Columbia University)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5024658548)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

设计并实现了一个基于语法的差分模糊器 PickleFuzzer，用来检测 Python pickle 的三种实现（Python、C、disassembler）之间的不一致性。

**💡 创新点**

首次将差分测试与专门构造的 pickle 语法结合，自动发现无规范实现中的安全相关差异，并对多实现间的行为不一致性提供了可重复的检测框架。

**🔧 技术方法**

采用语法生成模糊、差分测试、Docker 环境隔离、对 pickle 模块内部状态的补丁（返回堆栈、metastack、memo）以及异常与内部状态比较。

**📊 数据集**

使用自动生成的随机 pickle 对象（不依赖公开数据集），在不同配置（编码、out‑of‑band 缓冲）下进行测试。

**📈 对比分析**

与人工审计对比，Fuzzer 在 10 分钟内发现与 60 小时手工分析相同数量的差异，显著提升检测效率；发现的安全相关差异已被公开披露并获得奖金。

**⚠️ 局限性**

限制包括：仅覆盖合法 pickle 语法，无法发现大尺寸或恶意格式输入导致的差异；只检测异常与内部存储差异，忽略时间/资源等差异；生成器对极大整数或特定条件的覆盖有限。

---

## 660. Concurrency without Model Changes: Future-based Asynchronous Function Calling for LLMs

**arXiv ID:** 2605.15077 | [PDF](https://arxiv.org/pdf/2605.15077v1)

**作者:** Guangyu Feng `[一作]` (University of California, Berkeley), Joseph E. Gonzalez `[通讯]` (University of California, Berkeley)

**通讯引用:** 20276 | [OpenAlex ID](https://openalex.org/A5072427753)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AsyncFC框架，将LLM解码与函数执行解耦，使用未来占位符实现异步调用；

**💡 创新点**

创新点在于在不改模型、不改协议、不改函数实现的前提下，通过自动架构转换与依赖感知调度实现解码-执行重叠和跨函数并行；

**🔧 技术方法**

主要技术包括未来（future）占位符、同步到异步模式转换、标签化依赖注解、状态树调度器与并行执行器；

**📊 数据集**

使用的主要数据集包括BFCL v3与v4（Web Search）、SWE‑Bench Lite、HotpotQA、Gemini 3.1 Pro等；

**📈 对比分析**

与同步函数调用和原生并行函数调用相比，AsyncFC在BFCL v4实现1.26×、SWE‑Bench Lite实现1.44×的速度提升，且保持或略高于准确率；

**⚠️ 局限性**

局限性在于受任务结构和函数延迟的影响，极端顺序任务或低延迟函数时收益有限；同时对模型处理未来占位符的能力有一定依赖，且存在解码多轮和并行解码的额外开销。

---

## 661. The Guarded Fragment with Nested Equivalences

**arXiv ID:** 2605.15072 | [PDF](https://arxiv.org/pdf/2605.15072v1)

**作者:** Oskar Fiuk `[一作]` (University of Wrocław), Oskar Fiuk `[通讯]` (University of Wrocław)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5085359233)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究Guarded Fragment（GF）扩展的含有嵌套等价关系的逻辑，证明其在无等号、无函数符号的情况下具有限模型性质、可判定性，并给出精确的复杂度下界和上界；同时展示若去掉嵌套条件或允许等号，则可判定性失效。

**💡 创新点**

首次为GF+嵌套等价关系提供完整的复杂度分析，揭示每多一个等价关系会导致模型大小和决策复杂度呈指数层级递增；提出“有限指数嵌套性质”作为核心技术，实现了从多等价关系归约到单一等价关系的有效转换。

**🔧 技术方法**

主要技术包括：构造“嵌套计数器”句子实现大计数；利用“有限指数嵌套性质”证明模型中第二层等价类的指数界；通过逐层消除等价关系的归约；使用标签类型与可满足性判定框架，构建基于有限标签集合的判定算法；以及对非等价关系独立情形的下界构造。

**📊 数据集**

该工作纯理论化，没有使用任何实验数据集；所有结果通过逻辑构造与复杂度证明得出。

**📈 对比分析**

与先前仅在两变量或受限等价约束下的研究相比，该论文将可判定性范围大幅扩大，并给出了从NP到非元素级（e.g., (K+2)-EXP）等完整复杂度图谱。对于固定等价数量的情况，算法复杂度为 (K+2)-EXP；当等价数量不受限时，复杂度为 (K+1)-EXP；若使用常量或限制变量数量，则降低至 (K+1)-EXP 或 (K+2)-EXP。

**⚠️ 局限性**

局限包括：一旦允许等号（即存在常量或自等式），逻辑变为不可判定；对独立（非嵌套）等价关系的多于两条也导致不可判定；该框架不支持交叉乘积表达式、非等价等价关系或更宽松的 Guarded 变体（如 Loosely Guarded Fragment）；并且对于包含自由二元谓词的更通用情形，现有技术尚不能保证可判定性。

---

## 662. On the Cultural Anachronism and Temporal Reasoning in Vision Language Models

**arXiv ID:** 2605.15071 | [PDF](https://arxiv.org/pdf/2605.15071v1)

**作者:** Mukul Ranjan `[一作]` (MBZUAI), Zhiqiang Shen `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并构建了 TAB‑VLM 基准，用于评估视觉语言模型在解释印度文化遗产物件时的时间推理与文化准确性。

**💡 创新点**

创新点在于将“文化时代错误”系统化为可量化的评测维度，并设计了 600 道多模态多选题，形成首个针对非西方历史材料的时间推理基准。

**🔧 技术方法**

采用了十种主流视觉语言模型（包括 GPT‑5.2、GPT‑4o、Qwen2‑VL、InternVL3 等）以及统一的提示模板与评测脚本进行推理与答案解析。

**📊 数据集**

数据集由 1,600 件印度跨时代（史前至现代）文化遗产物件构成，经过专家验证后生成 600 题，每类 100 题涵盖六个时间推理子任务。

**📈 对比分析**

实验对比显示即使最强模型 GPT‑5.2 也仅达到 58.7% 的整体准确率，且在时序排序等任务上低于 40%，表明当前 VLM 在历史推理方面存在显著不足。

**⚠️ 局限性**

局限性包括仅聚焦印度文物、缺乏跨文化验证、评测仅使用图像而无文本元数据，且模型选取不涵盖最新架构，可能限制结果的普适性。

---

## 663. MetaBackdoor: Exploiting Positional Encoding as a Backdoor Attack Surface in LLMs

**arXiv ID:** 2605.15172 | [PDF](https://arxiv.org/pdf/2605.15172v1)

**作者:** Rui Wen `[一作]` (Institute of Science Tokyo), Ahmed Salem `[通讯]` (Microsoft Security Response Center)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MetaBackdoor，一种利用 LLM 位置编码（尤其是序列长度）作为触发器的后门攻击，可在不修改输入文本的前提下激活恶意行为。

**💡 创新点**

创新点在于把传统内容触发器替换为位置触发器，既能实现系统提示泄露、自动触发（自激活）等新型攻击，又能与已有内容触发器组合，显著扩展 LLM 后门的攻击面。

**🔧 技术方法**

使用技术包括：基于 RoPE/绝对位置编码的长度阈值/区间触发器、数据投毒训练、参数高效微调（LoRA/DoRA）、可解释性分析（token attribution、层级 probe）、以及阈值调优的边界感知采样。

**📊 数据集**

数据集涵盖：AGNews、MNLI、MMLU（分类基准）；CodeAlpaca-20k、OASST1（生成与多轮交互）；并在多种开源 LLM（Gemma‑3、Qwen‑3、Phi‑4、Olmo‑3）与不同规模（270M~12B）上进行实验。

**📈 对比分析**

与传统内容触发后门对比，MetaBackdoor 在所有模型与触发类型（Exact、Band、Threshold）下均实现 96–100% 的攻击成功率（ASR），同时保持 <1% 的正常任务准确率下降，展示了极高的效能与低可检测性。

**⚠️ 局限性**

局限包括：需要在训练阶段注入少量样本；阈值触发对长度分布敏感，可能被意外截断或误触；在强内容触发存在时易被内容特征覆盖；自激活效果依赖对话长度与解码设置；后续 fine‑tune 可能削弱攻击强度。

---

## 664. Text Knows What, Tables Know When: Clinical Timeline Reconstruction via Retrieval-Augmented Multimodal Alignment

**arXiv ID:** 2605.15168 | [PDF](https://arxiv.org/pdf/2605.15168v1)

**作者:** Sayantan Kumar `[一作]` (National Institutes of Health), Jeremy C. Weiss `[通讯]` (National Institutes of Health)

**通讯引用:** 2364 | [OpenAlex ID](https://openalex.org/A5072774346)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用检索增强的多模态对齐框架，将结构化电子健康记录（EHR）与临床叙事文本结合，构建更精确的绝对临床时间线。

**💡 创新点**

创新点在于：① 通过图结构分解为中心事件和非中心事件两阶段，先构建时间骨架再校准；② 在骨架和完整时间线两个阶段分别引入检索得到的结构化证据，实现分阶段时间校准；③ 在多模态框架中实现检索增强的多步流程。

**🔧 技术方法**

使用指令微调的大型语言模型（如DeepSeek、GLM5、Qwen3.5等）配合嵌入检索、LangChain与LangGraph实现多步推理，并采用结构化提示与约束输出。

**📊 数据集**

数据集为MIMIC‑III与MIMIC‑IV的i2m4基准（20份出院摘要），包含人工标注的绝对时间线。

**📈 对比分析**

对比单模态文本重建与双模态检索增强重建，事件匹配率差异不大，但检索增强显著提升了时间一致性（c‑index）和绝对时间误差（AULTC）；在不同模型上，采用双阶段校准可进一步提升时间质量。

**⚠️ 局限性**

局限性包括：① 基准规模小，难以覆盖全部临床文档多样性；② 人工注释由概念化格式重构，仍可能存在语义不匹配；③ 对中心事件抽取质量高度依赖，错误会向后传播；④ 主要在败血症病例上验证，泛化到其他疾病尚待研究。

---

## 665. Position: Behavioural Assurance Cannot Verify the Safety Claims Governance Now Demands

**arXiv ID:** 2605.15164 | [PDF](https://arxiv.org/pdf/2605.15164v1)

**作者:** Pratinav Seth `[一作]` (Lexsi Labs), Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析治理与验证之间的审计缺口，基于七个锚点案例构建矩阵，并提出可执行的机制化证据审计试点

**💡 创新点**

首次将治理条文与访问层级映射成结构化矩阵，提出激励梯度解释，并设计三线机制化验证协议

**🔧 技术方法**

线性探测、激活补丁、前后训练对比等机制解释技术，结合可信执行环境与结构化访问协议

**📊 数据集**

使用公开模型内部激活和隐藏目标测试的保留对照样本（含合成“隐蔽目标”提示）

**📈 对比分析**

通过对照线性探测AUROC≥0.95、补丁效应≥1.5σ等阈值评估，之前研究已达到0.96–0.999 AUROC；试点预计可复现Tier‑1证据

**⚠️ 局限性**

机制化方法在前沿大模型上仍不可扩展，权重访问受限，可能被游戏，双重使用风险，且对司法多样性及数据集覆盖有限

---

## 666. Hand-in-the-Loop: Improving Dexterous VLA via Seamless Interventional Correction

**arXiv ID:** 2605.15157 | [PDF](https://arxiv.org/pdf/2605.15157v1)

**作者:** Zhuohang Li `[一作]` (Shanghai Jiao Tong University), Ruoshi Wen `[通讯]` (ByteDance Seed)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了HandITL无缝干预方法，结合优化相对手部重定向和速度共享控制，实现双手Dexterous VLA政策的在线纠正与高维on‑policy数据收集，显著降低指令跳跃并提升长周期任务表现。

**💡 创新点**

创新点包括：① 基于关键向量与约束的优化相对重定向，大幅降低握持过程中的手势跳跃；② 速度共享控制接口，用残差Twist实现手臂细微纠正；③ 两种介入模式（全接管与协同控制）适配不同任务需求；④ 通过在策略执行时收集的高维纠正数据，提升策略对OOV状态的鲁棒性。

**🔧 技术方法**

技术手段包括：优化相对手部重定向（约束、全局形状、精确抓握、结构安全、时间正则化）；速度共享控制（EMA平滑残差Twist注入）；基于Franka FR3 + Bytedexter V2双臂双手硬件平台；VLA策略预训练（GR‑Dexter）+微调；类DAgger的在政策滚动期间人工介入采样。

**📊 数据集**

使用的数据集为20小时真实手工操作（Teleoperation）数据以及1小时的介入数据（全接管与协同模式），并基于预训练的GR‑Dexter模型进行初始微调。

**📈 对比分析**

通过与绝对重定向、相对命令重定向、Jacobian映射、Delta‑Command以及直接切换等基线进行对比实验。实验结果显示：相对重定向将指令跳跃降低约99.8%；在抓握稳定性、后续操作时间和子目标完成率方面均优于基线；采用协同模式收集的on‑policy纠正数据在长周期任务中的子目标完成率明显高于单纯Teleoperation或全接管数据。

**⚠️ 局限性**

局限性包括：① 仅采用简单监督微调，介入数据中可能包含噪声或次优恢复动作；② 在毫米级高精度操作（如钻头对准）仍受视觉遮挡与分辨率限制，难以实现；未来可考虑引入触觉、力反馈等多模态感知提升性能。

---

## 667. Self-Distilled Agentic Reinforcement Learning

**arXiv ID:** 2605.15155 | [PDF](https://arxiv.org/pdf/2605.15155v1)

**作者:** Zhengxi Lu `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**通讯引用:** 1595 | [OpenAlex ID](https://openalex.org/A5004615610)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种自我蒸馏强化学习框架 SDAR，将 RL 作为主优化目标，辅以 token‑级别的 On‑Policy Self‑Distillation 并通过 sigmoid 门控自适应调节蒸馏强度；

**💡 创新点**

在多轮代理训练中首次将 OPSD 作为独立的辅助目标，并通过门控机制实现教师–学生概率差的非对称利用，保持 RL 优势无偏；

**🔧 技术方法**

采用 Qwen 系列 LLM 的 GRPO 算法、逆 KL 蒸馏损失、教师-学生概率差、技能检索（UCB、关键词匹配）以及 sigmoid 门控等技术；

**📊 数据集**

在 ALFWorld、WebShop 与 Search‑QA（含 NQ、HotpotQA、TriviaQA 等）三大多轮代理基准上进行实验；

**📈 对比分析**

与 GRPO、OPSD、Skill‑GRPO、GRPO+OPSD、Skill‑SD、RLSD 等多种基线对比，SDAR 在 ALFWorld 提升 9.4%、Search‑QA 提升 7.0%、WebShop‑Acc 提升 10.2%，且显著避免了 Naive GRPO+OPSD 的不稳定性；

**⚠️ 局限性**

仍然依赖优质的检索式技能或特权上下文，门控参数需要细致调优，低质量检索时提升有限，并且实现复杂度较高。

---

## 668. Investigating the Suitability of Delay Tolerant Networks for Broadcasting Tsunami Warnings in Palu, Indonesia

**arXiv ID:** 2605.15103 | [PDF](https://arxiv.org/pdf/2605.15103v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 669. Novel Dynamic Batch-Sensitive Adam Optimiser for Vehicular Accident Injury Severity Prediction

**arXiv ID:** 2605.15083 | [PDF](https://arxiv.org/pdf/2605.15083v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 670. Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution

**arXiv ID:** 2605.15138 | [PDF](https://arxiv.org/pdf/2605.15138v1)

**作者:** Saisab Sadhu `[一作]` (Lexsi Labs), Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三阶段机制对齐零空间机器无学习方法，能够在 4‑bit 量化后仍保持忘记效果并不影响模型能力。

**💡 创新点**

创新点在于将因果子图定位（EAP‑IG）、对保留任务敏感方向的零空间投影以及对量化门槛的显式幅度上限相结合，保证更新既能在结构层面真正消除知识，又能在量化过程中不被舍弃；同时引入 CAD 指标对结构消除进行验证。

**🔧 技术方法**

使用的技术包括：EAP‑IG 归因子图、对保留损失的 Fisher 信息投影、对参数更新的量化门槛上限（magnitude floor）、标准梯度下降与 KL 正则化组合、4‑bit NF4 量化评估。

**📊 数据集**

使用的数据集包括：WMDP‑bio/chem/cyber（多选题形式的危险知识回忆）、MUSE（开放式记忆检索）、MMLU、IFEval 以及相应的保留/保持集。

**📈 对比分析**

与六种基线方法（Global GA、Surgical GA、NPO、SimNPO、GU+SimNPO、LUNAR）比较，本文方法在忘记精度、量化永久性（PTQ gap 负值）、MMLU 维持（≤0.03 下降）以及 CAD（≈1.1–1.6）方面均优于或匹配基线，且在 94 个实验单元中唯一满足所有四个属性（忘记、量化永久性、保持能力、结构消除）。

**⚠️ 局限性**

局限性：仅在 Llama‑3.1‑8B、Qwen‑3‑8B 两大 8B 规模模型上验证，未覆盖更大规模或不同架构；对因果子图的定位依赖于 EAP‑IG，需额外的可解释性工作；仅针对事实回忆型任务，开放式或其他知识类型的推广仍待验证。

---

## 671. Training ML Models with Predictable Failures

**arXiv ID:** 2605.15134 | [PDF](https://arxiv.org/pdf/2605.15134v1)

**作者:** Will Schwarzer `[一作]` (University of Massachusetts Amherst), Scott Niekum `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 2008 | [OpenAlex ID](https://openalex.org/A5043572737)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种“forecastability fine‑tuning”方法，通过在模型训练时最小化对极值理论预报误差的损失，使模型的失败尾部更符合极值理论预测。

**💡 创新点**

创新点在于将极值理论（Gumbel‑tail）预测误差拆解为可训练的曲率、占用和高阶残差三部分，并基于此设计可微的损失函数；同时结合“improvement mask”和元多任务微调，使得模型在保持主任务性能的同时显著提升尾部预报精度。

**🔧 技术方法**

使用的技术包括：极值理论（Gumbel‑tail 预报）、OLS 拟合、可微的 forecastability 损失、改进梯度掩码、KL 正则化、LoRA 微调、元多任务分区随机化与缓存、以及对 RL 任务的闭式期望代价计算。

**📊 数据集**

数据集主要是两种人工合成任务：① 单 token 密码游戏中，生成两部分提示（正常 jailbreak + 优化后的攻击后缀），② RL 网格世界中，生成带或不带陷阱的布局；两者均为混合分布，罕见高失败模式几乎不出现在评估集，却在部署集出现。

**📈 对比分析**

对比方法包括：预训练基线、后置仿射校准、监督式微调（SFT）以及本方法。实验结果显示：本方法在保持甚至提升主任务性能的同时，worst‑rank 预测误差提升约 50‑70 倍；安全性（worst‑rank 泄露概率或 regret）与 SFT 近似，但更易于后置校准；预训练基线显著劣势。

**⚠️ 局限性**

局限性包括：实验规模有限（仅 0.6B LM 与 8×8 网格世界）、使用合成混合任务而非真实罕见失败模式、RL 环境需闭式代价，无法直接推广到需要 Monte‑Carlo 估计的情形、并未在大规模模型或更复杂部署域上验证；此外，forecastability 损失目前仅针对 Gumbel‑tail，可进一步扩展。

---

## 672. Aligning Latent Geometry for Spherical Flow Matching in Image Generation

**arXiv ID:** 2605.15193 | [PDF](https://arxiv.org/pdf/2605.15193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 673. Veritas: A Semantically Grounded Agentic Framework for Memory Corruption Vulnerability Detection in Binaries

**arXiv ID:** 2605.15097 | [PDF](https://arxiv.org/pdf/2605.15097v1)

**作者:** Xinran Zheng `[一作]` (University College London), Lorenzo Cavallaro `[通讯]` (University College London)

**通讯引用:** 4758 | [OpenAlex ID](https://openalex.org/A5036908366)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

开发了一种基于语义 grounding 的多代理框架，用静态切片器、双视图 LLM 检测器以及多代理验证器，在去符号化二进制中检测内存腐败漏洞。

**💡 创新点**

创新点在于将静态语义恢复与运行时验证两层 grounding 相结合，利用 RetDec 的 LLVM IR 生成的值流图提取源至汇的 witness-backed flow，并在双视图（decompiled C + 选定 LLVM IR）上进行 step‑wise LLM 推理，同时通过前缀缓存减少重复计算，从而实现高召回率且误报率低。

**🔧 技术方法**

采用 RetDec 对二进制进行 IR 提升、defuse 构造 inter‑procedural propagation graph、LLM（GPT‑5.4）进行双视图 step‑wise 推理、Trie‑based 前缀 memoization、AutoGen 多代理（Strategist/Verifier/Explorer）与 Radare2、Valgrind 等工具进行运行时验证。

**📊 数据集**

构造了 20 个真实世界的二进制漏洞样本，来源于 SEC‑Bench、ARVO 和公开 CVE 项目，手工注释源到汇的流、触发路径及根因，并提供 PoC 可执行。

**📈 对比分析**

与 Meta Infer、Semgrep、cwe_checker、AFL++、RepoAudit、Codex、Claude Code 等七个基线对比；Veritas 在 20 个样例上达到 90% 召回率，验证器在 623 条候选中检测到 0 FP，抽样验证仅发现 2 FP；相比基线的 0%–35% 召回率，显著优于传统静态、动态及 agentic 方法。

**⚠️ 局限性**

主要限制包括验证阶段成本高（每条候选约 1.8 美元，验证时间占总成本 60%+），依赖 RetDec 的 IR/去编译质量，若恢复失效会导致漏报/误报；数据集规模小且需大量人工标注；LLM 仍未完全内化内存语义，需外部分析支持。

---

## 674. Eliminating reversals from cubical type theories

**arXiv ID:** 2605.15080 | [PDF](https://arxiv.org/pdf/2605.15080v1)

**作者:** Evan Cavallo `[一作]` (University of Gothenburg), Christian Sattler `[通讯]` (Chalmers University of Technology)

**通讯引用:** 133 | [OpenAlex ID](https://openalex.org/A5103055014)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

建立了逆变（reversal）操作在立方体类型理论中的保守扩展性，并通过twist构造给出了相应的解释（twist interpretation）与模型；

**💡 创新点**

首次证明在自对偶间隔理论下加入逆操作是保守扩展；引入twist construction与span interpretation，构造首个严格立方体类型理论带逆的模型；

**🔧 技术方法**

使用了SOGAT（第二阶广义代数理论）框架、可映射范畴（representable map categories）与其模型，twist构造、span解释、弱等价（weak equivalence）技术，以及ABCHFL模型构建框架；

**📊 数据集**

无（本文无数据集使用）；

**📈 对比分析**

无（本文未进行实验比较或性能评测）；

**⚠️ 局限性**

仅适用于自对偶间隔理论；对连接（connections）的支持有限；缺乏对严格立方体类型理论的完整保守性证明；模型仅在经典逻辑下呈现∞-groupoids，尚未覆盖所有间隔/连接组合的情况。

---

## 675. From Plans to Pixels: Learning to Plan and Orchestrate for Open-Ended Image Editing

**arXiv ID:** 2605.15181 | [PDF](https://arxiv.org/pdf/2605.15181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 676. VGGT-Edit: Feed-forward Native 3D Scene Editing with Residual Field Prediction

**arXiv ID:** 2605.15186 | [PDF](https://arxiv.org/pdf/2605.15186v1)

**作者:** Kaixin Zhu `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 15430 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了 VGGT-Edit，一个一次性前向推理即可完成文本驱动的本地3D场景编辑框架。

**💡 创新点**

创新点包括：① 采用残差场预测方式在原始几何基础上局部变形，保持背景不变；② 引入深度同步文本注入，使语义指令与空间特征在多层保持一致；③ 视角感知加权机制，根据可见度和遮挡程度动态调节不同视角的贡献。

**🔧 技术方法**

技术手段：冻结的 π^3 重建主干、深度同步跨层文本注入、视角加权的多模态融合、残差变形头、并使用多项式损失（尺度对齐、保留、法线一致、相机一致、残差正则）进行训练。

**📊 数据集**

使用了自研的 DeltaScene 数据集，包含约 100,000 对高质量“前后”3D编辑样本，涵盖加、删、改、移等四类基本操作及组合编辑。

**📈 对比分析**

与 2D‑lifting（GaussCtrl、Omni‑3DEdit）、迭代优化（EditSplat）以及 feed‑forward 基线（Edit3r、NoPoSplat）对比，VGGT-Edit 在 CLIP 得分最高（30.2）、C‑FID 最低（122.4）、C‑KID 最低（0.048），并将每场景编辑时间压缩至约 5 秒，提升 2‑120 倍。

**⚠️ 局限性**

局限性：依赖高质量的训练数据和冻结主干，极端遮挡或非常小的对象编辑仍可能出现误差；在极端视角或非常复杂的光照/材质变化下的鲁棒性尚待进一步验证。

---

## 677. Is Grep All You Need? How Agent Harnesses Reshape Agentic Search

**arXiv ID:** 2605.15184 | [PDF](https://arxiv.org/pdf/2605.15184v1)

**作者:** Sahil Sen `[一作]` (PricewaterhouseCoopers), Vamse Kumar Subbiah `[通讯]` (PricewaterhouseCoopers)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5011129359)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性比较了LLM代理在不同架构（Chronos自定义harness与Claude Code、Codex、Gemini CLI provider-native CLI）以及工具调用方式（inline vs file-based）下，使用grep和向量检索对长记忆问答任务的端到端效果。

**💡 创新点**

创新点在于首次将检索策略、代理harness和工具交付路径联合评估，揭示检索方式在不同架构中的表现差异，并通过逐步添加噪声的实验探讨检索在噪声环境下的鲁棒性。

**🔧 技术方法**

使用的技术包括正则匹配检索（grep）、向量检索（ANN+rerank）、Chronos预处理的时间事件抽取、LangChain与RAG流水线、以及文件式与内联式工具调用的实现。

**📊 数据集**

使用的数据集是LongMemEval 116题子集，包含多会话问答以及Chronos抽取的结构化时间事件，作为检索语料。

**📈 对比分析**

在两组实验中，先在全量语料下做全因子对比，随后逐步增加无关会话噪声；使用GPT-4o作为评估器计算准确率；结果显示在inline模式下grep普遍优于vector，文件式交付可导致优势消失或逆转，性能差异可达数个百分点。

**⚠️ 局限性**

局限性包括：仅针对长记忆对话问答，检索效果受文本精确匹配影响，可能不适用于需要推理或非文本证据的场景；实验仅覆盖部分CLI provider，未完整涵盖所有系统；噪声扩增仅模拟特定分布，泛化性有限。

---

## 678. Articraft: An Agentic System for Scalable Articulated 3D Asset Generation

**arXiv ID:** 2605.15187 | [PDF](https://arxiv.org/pdf/2605.15187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 679. Hybrid Sketching Methods for Dynamic Connectivity on Sparse Graphs

**arXiv ID:** 2605.15173 | [PDF](https://arxiv.org/pdf/2605.15173v1)

**作者:** Quinten De Man `[一作]` (University of Maryland), David Tench `[通讯]` (Lawrence Berkeley National Lab)

**通讯引用:** 167 | [OpenAlex ID](https://openalex.org/A5085739390)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了混合草图化（Hybrid Sketching）动态连通性算法与系统，将图分为密集核心与稀疏边缘，密集核心采用 per‑vertex sketch，稀疏边缘使用传统 lossless 结构，实现了全动态和半流式的连通性查询。

**💡 创新点**

创新点包括：
1) 首次提出针对实际图的混合草图策略，理论上同时匹配稀疏图的 lossless 空间 O(V+E) 与稠密图的 sketch 空间 O(V log²V)，并在中间稠密度下进一步改进；
2) 设计了一种新的 ℓ₀‑sampler（balloon sketch），在每顶点支持下的空间仅与真实支持大小成比例，较传统 ℓ₀‑sampler 节省约 8 倍；
3) 结合可逆布隆表（IBLT）实现核心边缘的完整恢复；
4) 构建了可插拔的系统框架（Hybrid Sketching Connectivity），可将任何 lossless 子系统与任何 sketch‑based 子系统组合。

**🔧 技术方法**

技术手段包括：
- 线性 sketching 与 ℓ₀‑sampler（新型 balloon sketch）；
- 可逆布隆表（IBLT）用于从 sketch 中完整恢复邻接；
- 变形的 cutset 与动态树（Link‑Cut Tree）实现连接查询；
- 基于密度阈值 δ 的动态分区与批处理（更新缓冲、重构核心边缘）；
- 采用多层（O(logV)）cutset 结构、IBLT 与 balloon sketch 组合。

**📊 数据集**

实验数据集覆盖真实与合成：
- 社交网络：Google+, Friendster, Twitter, Orkut；
- 网络图：ENWiki, Youtube；
- 道路图：RoadUSA, RoadGermany；
- k‑近邻/范围搜索图：SIFT‑RS50K, SIFT‑KNN500, MSSP‑RS10K, MSSP‑KNN500；
- 合成 Kronecker 图：Kron‑13/15/16。

**📈 对比分析**

与最先进的 lossless（Cluster Forest）和稠密 sketch（AGM‑based）基线进行比较：
- 空间：在平均度 <100 时节省 15%；在 100–1000 之间节省 92%；在 >1000 时节省 97%；
- 处理吞吐量与两子系统基本相同；稀疏图几乎无额外延迟，密集图与稠密基线相当；
- 内存占用在所有测试图上均低于或等于基线，且 OOM 场景显著减少。

**⚠️ 局限性**

局限性：
1) 密集核心阈值 δ 需经验调参，可能对不同图类型影响显著；
2) 核心边缘恢复偶尔失败，虽然概率极低，但仍可能导致额外空间开销；
3) 实现目前为单机内存方案，尚未验证对更大规模图的可扩展性；
4) 对有向图或多重边的情况未充分评估；
5) 依赖现有的 balloon‑sampler 与 IBLT，若有更高效的草图原语出现需进一步改进；
6) 在极稀疏但包含大核心的图上，混合策略的收益仍待进一步验证。

---

## 680. Eradicating Negative Transfer in Multi-Physics Foundation Models via Sparse Mixture-of-Experts Routing

**arXiv ID:** 2605.15179 | [PDF](https://arxiv.org/pdf/2605.15179v1)

**作者:** Ellwil Sharma `[一作]` (Shodh AI), Arastu Sharma `[通讯]` (Shodh AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

研究了多物理场下的科学机器学习，提出Shodh‑MoE架构，通过稀疏专家路由和物理约束的潜在自编码器解决负迁移问题，实现在单一模型中同时学习开放通道湍流和多孔介质流。

**💡 创新点**

创新点包括：1) 将稀疏专家网络嵌入Transformer前向传播，实现自动域分支；2) 使用Helmholtz风格速度参数化的物理约束tokenizer保证质量守恒；3) Top‑1 soft‑semantic路由在无标签情况下自适应分配子网络；4) 在单一模型中实现两种物理机制的并行收敛，避免密集参数共享导致的负迁移。

**🔧 技术方法**

采用物理信息自编码器、稀疏专家网络（MoE）、软语义路由、bfloat16混合精度、Triton自定义核、分布式数据并行、Zarr懒加载、梯度对齐优化等技术。

**📊 数据集**

使用约61,000个128³ 3D张量的混合预训练语料，分别对应Navier–Stokes类型的连续流和Darcy–Brinkman–Stokes类型的多孔介质流。

**📈 对比分析**

与全密集Transformer对比，Shodh‑MoE在两域均实现潜在MSE≈2.5×10⁻⁵（连续流）/9.8×10⁻⁶（多孔介质），物理MSE≈2.5×10⁻⁶/1.8×10⁻⁶；路由完全分化（100%分配），显著降低负迁移，保持质量守恒，整体性能优于密集模型。

**⚠️ 局限性**

局限性包括：仅在两种物理域上验证，稀疏路由在更大或更复杂多物理场景中可能需要更多专家；模型训练需高端H100集群，推理速度与能耗未系统评估；对极端边界或几何复杂度的稳健性仍待验证。

---

## 681. Quantitative Video World Model Evaluation for Geometric-Consistency

**arXiv ID:** 2605.15185 | [PDF](https://arxiv.org/pdf/2605.15185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 682. SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer

**arXiv ID:** 2605.15178 | [PDF](https://arxiv.org/pdf/2605.15178v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 683. RefDecoder: Enhancing Visual Generation with Conditional Video Decoding

**arXiv ID:** 2605.15196 | [PDF](https://arxiv.org/pdf/2605.15196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 684. VGGT-$Ω$

**arXiv ID:** 2605.15195 | [PDF](https://arxiv.org/pdf/2605.15195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 685. RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO

**arXiv ID:** 2605.15190 | [PDF](https://arxiv.org/pdf/2605.15190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 686. When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability

**arXiv ID:** 2605.15183 | [PDF](https://arxiv.org/pdf/2605.15183v1)

**作者:** ML Nissen Gonzalez `[一作]` (MARS), Thomas Dooms `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于权重的张量相似度（Tensor Similarity）度量，用于衡量多线性模型（如双线性变压器）在功能上的等价性，并实现了对机制变化的定位和跟踪。

**💡 创新点**

创新点在于：①通过对张量进行对称化与极化，消除权重空间的对称性，得到在功能等价性上严格一致的度量；②使用高斯度量与 Isserlis 定理将期望内积化为仅依赖权重的闭式公式；③采用递归 Gram 方式在深层模型中高效计算全局张量内积，避免显式构造巨型张量。

**🔧 技术方法**

核心技术包括：多线性/双线性 Transformer 结构、对称化投影（P_n）、极化同构、Gauss 度量、递归 Gram 计算、张量切片与差分相似度。

**📊 数据集**

实验数据集涵盖：SVHN（用于灾难性遗忘与后门注入），小数加法（模块化算术，用于观察 grokking），以及 The Pile（大规模语言模型训练）。

**📈 对比分析**

与传统方法（行为相似度、CKA、矩阵余弦、行为余弦）比较，Tensor Similarity 在检测后门、定位输出维度、跟踪连续重组和捕捉跨分布变化等任务上表现更为精细和鲁棒；在语言模型训练中亦能揭示更细致的块结构和转折点。

**⚠️ 局限性**

主要局限性：仅适用于多线性模型，无法直接应用于包含非多项式激活函数的主流深度网络；实验聚焦于功能变化检测，对具体致因组件的归因能力尚未充分验证。

---

## 687. FutureSim: Replaying World Events to Evaluate Adaptive Agents

**arXiv ID:** 2605.15188 | [PDF](https://arxiv.org/pdf/2605.15188v1)

**作者:** Shashwat Goel `[一作]`, Jonas Geiping `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于真实新闻的时间线模拟环境“ForecastBench”，让语言模型代理在其知识截止后（2026年1-3月）实时检索新闻并持续更新对未来事件的概率预测。

**💡 创新点**

创新点在于：①将真实世界事件按时间顺序重播，消除未来信息泄漏；②允许代理自由选择预测问题并给出多种可能结果的概率分布；③提供可插拔的工具与定制 harness，系统化评估搜索、记忆、适应性等能力；④通过 Brier 技巧得分和 top‑1 准确度兼顾校准与准确性。

**🔧 技术方法**

使用的技术包括：多模态工具链（文件操作、Shell、语义+关键字检索）；自定义 harness 结合 Codex、OpenCode、Claude Code；Brier Skill Score 计算；语言模型推理与检索；Python 数据帧处理；多轮推理与记忆写入/读取。

**📊 数据集**

数据集主要是：Common Crawl News (CCNews) 的 7.36M 篇新闻（2023‑2026），以及从阿拉伯联合酋长国新闻机构生成的 330 条预测问题（resolution 2026 年 1‑3 月）。

**📈 对比分析**

比较方法：在默认 harness 与改进 harness 下评估多款前沿模型（GPT‑5.5、Claude Opus‑4.6、DeepSeek‑V4 Pro、Qwen3.6 Plus、GLM‑5.1）在 Brier Skill Score 与 top‑1 准确度上的表现；同时与 Polymarket 人类聚合预测进行对比。性能显示 GPT‑5.5 最高，accuracy 约 25%，Brier 近 0；大多数开源模型初始 Brier 负值，需 harness 改进后才能提升。

**⚠️ 局限性**

局限性：①仅评估预测能力，无法影响环境（不适用于执行型决策）；②受限于 90 天的时间窗口和单一知识截止点；③对新闻来源和检索工具的依赖可能导致信息偏倚；④高算力模型的评估成本高；⑤目前仅支持英文新闻，跨语言迁移需进一步研究。

---

## 688. EntityBench: Towards Entity-Consistent Long-Range Multi-Shot Video Generation

**arXiv ID:** 2605.15199 | [PDF](https://arxiv.org/pdf/2605.15199v1)

**作者:** Ruozhen He `[一作]` (ByteDance), Vicente Ordonez `[通讯]` (Rice University)

**通讯引用:** 13021 | [OpenAlex ID](https://openalex.org/A5027328044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多镜头视频生成基准（MSVBench）和三柱评估框架，用于细粒度评估实体一致性。

**💡 创新点**

创新点包括：①140剧集、2491镜头的实体调度式数据集；②51维三柱评估指标；③通过实体记忆 Bank（EntityMem）实现跨镜头的视觉身份保持。

**🔧 技术方法**

使用的技术包括：LLM驱动的脚本生成与实体检索、VLM与Embedding相结合的评估、以及基于内存的多实体条件生成（Memory‑augmented Generation）。

**📊 数据集**

数据集为从真实叙事媒体抽取并 LLM 丰富的 140 剧集（易/中/难层级），共 2491 镜头，具有显式实体调度表。

**📈 对比分析**

与 StoryMem、HoloCine、CineTrans 等方法对比，EntityMem 在跨镜头一致性、实体呈现率和多维度一致性指标上显著领先，Cohen d > 2.3。

**⚠️ 局限性**

局限性：依赖预生成的实体参考，难以处理极长序列中细节漂移；物体一致性指标仍略逊于字符一致性。

---

## 689. ATLAS: Agentic or Latent Visual Reasoning? One Word is Enough for Both

**arXiv ID:** 2605.15198 | [PDF](https://arxiv.org/pdf/2605.15198v1)

**作者:** Ziyu Guo `[一作]` (Meta AI), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 54949 | [OpenAlex ID](https://openalex.org/A5032708386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种将视觉操作抽象为离散功能词（functional tokens）的视觉推理框架ATLAS，避免生成中间图像或调用外部工具，保持自回归完整性。

**💡 创新点**

创新点在于：①将视觉推理操作直接映射到标准词表中的离散token；②无需视觉监督即可训练；③在RL阶段引入Latent‑Anchored GRPO (LA‑GRPO) 针对稀疏功能token 的梯度稀释问题提供辅助目标。

**🔧 技术方法**

核心技术包括：自回归语言模型（以Qwen2.5‑VL‑7B为基础），功能token设计与交叉熵训练，标准GRPO强化学习，LA‑GRPO 的 token‑级辅助损失，以及基于奖励的多项式惩罚。

**📊 数据集**

使用了新构建的-178K功能token监督数据集（涵盖40+视觉推理任务），以及We‑Math、MMK12、ThinkLite 等公开视觉推理 benchmark。

**📈 对比分析**

与现有统一模型、代理式模型、潜在模型等对比，ATLAS在BLINK、WeMath 等主基准上平均提升至51.3%/70.6%（GRPO）/62.9%（LA‑GRPO），在BLINK-Jigsaw中推理延迟从18.83 s降至3.80 s，显著减少内存与输出长度。

**⚠️ 局限性**

局限性包括：功能token 词表仍有限，难以覆盖所有复杂视觉操作；RL奖励设计需细致平衡，若惩罚不当会出现“token spam”；模型在某些子任务（IQ、多视角推理）提升有限。

---

## 690. Warp-as-History: Generalizable Camera-Controlled Video Generation from One Training Video

**arXiv ID:** 2605.15182 | [PDF](https://arxiv.org/pdf/2605.15182v1)

**作者:** Yifan Wang `[一作]` (Shanghai Jiao Tong University), Tong He `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Warp-as-History 方法，通过将目标相机轨迹生成的视差 warp 作为历史输入，利用已有的历史条件扩散模型实现单视频微调的相机控制。

**💡 创新点**

创新点在于：①把相机几何信息直接转换为模型的历史视觉条件，而非加入新的注意力/编码模块；②通过目标帧对齐与可视标记选择，使得 warp 既提供相机运动证据，又让模型自行完成遮挡与动态填充；③仅需在一段带标注视频上做轻量 LoRA 微调即可显著提升效果。

**🔧 技术方法**

使用技术包括：历史条件扩散模型 Helios、相机 warp 的几何重建与投影、RoPE 位置对齐、可视 token 过滤、以及 1 视频 LoRA 微调。

**📊 数据集**

实验数据集：WorldScore（静态场景生成）、RE10K（房地产场景相机运动）和 DAVIS（动态视频）。

**📈 对比分析**

与 Gen3C、Voyager、ViewCrafter、HyWorldPlay 等现有相机控制方法进行对比，使用摄像机跟随精度、FID/FVD、VBench、DOVER 等指标评估。Warp-as-History 在仅使用一段视频微调的前提下，在摄像机跟随与视觉质量上均可与大规模训练的基线竞争，甚至在某些指标上更优。

**⚠️ 局限性**

局限性包括：零样本效果仍不够稳健，warp 误差可能导致动态复制过度；对源视频的选择敏感；在极端遮挡或复杂动态场景下表现可能下降；仅在有限数据集上验证，未涵盖所有真实世界变种。

---

## 691. OpenDeepThink: Parallel Reasoning via Bradley--Terry Aggregation

**arXiv ID:** 2605.15177 | [PDF](https://arxiv.org/pdf/2605.15177v1)

**作者:** Shang Zhou `[一作]` (UC San Diego), Jingbo Shang `[通讯]` (UC San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于种群的测试时计算框架OpenDeepThink，用LLM既生成解答又进行配对判定，并通过Bradley–Terry排名与自然语言批判驱动的变异实现对竞争编程和多域问题的提升。

**💡 创新点**

将配对式Bradley–Terry评估与无外部验证器的种群演化结合，实现了无需真实判定器的并行选择与定向突变，突破了传统深度扩展的瓶颈。

**🔧 技术方法**

采用LLM双重角色、配对比较、Bradley–Terry排名、自然语言批判聚合反馈、种群保留与消亡策略以及迭代突变等技术。

**📊 数据集**

在 Codeforces 73 题（CF‑73）、NOI‑119、合计 192 道竞赛级编程题以及 82 道多域 HLE（Humanity's Last Exam）问题上进行实验。

**📈 对比分析**

与随机采样、Oracle、Self‑Refine 等基线对比，OpenDeepThink 在 Gemini 3.1 Pro 上 Codeforces Elo 提升约 +405 点，在 HLE 中客观可验证域提升，主模型在硬题层面提升约 50% 的通过率，整体算力约 285 次 API 调用。

**⚠️ 局限性**

仅在 Gemini 系列 LLM 上验证，算力成本高、对主机模型依赖强，且在主观领域配对评估不可靠时可能导致性能下降。

---

