# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-08 | 今日论文总数: 576

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Improving Clinical Trial Recruitment using Clinical Narratives and Large Language Models

**arXiv ID:** 2604.05190 | [PDF](https://arxiv.org/pdf/2604.05190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 2. Belief Dynamics for Detecting Behavioral Shifts in Safe Collaborative Manipulation

**arXiv ID:** 2604.04967 | [PDF](https://arxiv.org/pdf/2604.04967v1)

**作者:** Devashri Naik `[一作]` (University of Illinois Chicago), Amit Ranjan Trivedi `[通讯]` (University of Illinois Chicago)

**通讯引用:** 1290 | [OpenAlex ID](https://openalex.org/A5028132107)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在共享工作空间中检测协作伙伴行为模式切换，提升安全协同操控

**💡 创新点**

提出轻量级贝叶斯状态追踪模块（Belief Module），利用选择性状态空间动力学、因果注意力和预测误差信号，实现高可靠性、低延迟的切换检测

**🔧 技术方法**

选择性状态空间模型（SSM）、因果注意力机制、对比记忆（Contrastive Memory）、多尺度预测误差、冻结的Vision‑Language‑Action（VLA）控制骨干

**📊 数据集**

ManiSkill（双臂 Franka Panda 的共享工作空间任务）和 Overcooked（多人协同烹饪游戏）

**📈 对比分析**

与十种基线（GRU、Transformer、Mamba、LIAM、BToM、BOCPD、Context‑Conditioned、Oracle 等）在五个随机种子下比较。在 ±3 步窗口下，Belief Module 以 85.7% 的检测率领先；检测开启后，碰撞率下降 52%；且其后续适配的近距离时间（CRT）最小（4.8 步），低于完美检测的 Oracle（5.3 步）

**⚠️ 局限性**

实验仅使用脚本化的伙伴策略与基于状态的观测；未验证连续人类行为或视觉输入；在更复杂的真实环境中可能需要进一步扩展

---

## 3. Semantic analysis of behavior in a DNA-functionalized molecular swarm

**arXiv ID:** 2604.05277 | [PDF](https://arxiv.org/pdf/2604.05277v1)

**作者:** Tom Bachard `[一作]` (Ochanomizu University), Nathanael Aubert-Kato `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文使用 DNA 功能化微管群体仿真，结合 CLIP 嵌入和稀疏字典学习，构建语义字典来提取并量化群体行为，并通过激活值推断外部温度。

**💡 创新点**

创新点在于将基于 CLIP 的语义字典学习方法首次应用于分子群体仿真，实现高层行为的可解释提取，并展示其可用于自动化温度预测与系统优化。

**🔧 技术方法**

核心技术包括：C-GLASS 微管动力学仿真、CLIP 视觉语义嵌入、字典学习（L1 稀疏正则化）、UnCLIP 图像生成以及简单 MLP 温度回归。

**📊 数据集**

使用了由 C-GLASS 生成的 40,500 帧仿真数据（温度 200–400 K，步长 25 K），每帧划分 9 子帧，形成训练集。

**📈 对比分析**

通过激活热图、箱线图验证字典的行为区分效果，并用 MLP 预测温度，平均均方误差为 0.22±0.04，表明方法能准确捕捉温度驱动的群体行为变化。

**⚠️ 局限性**

局限性包括：仿真与实验的几何与视觉细节不匹配，字典中语义与物理几何不完全对应；温度预测存在滞后和对极端值的不准确；需进一步实验验证与仿真模型改进。

---

## 4. Pramana: Fine-Tuning Large Language Models for Epistemic Reasoning through Navya-Nyaya

**arXiv ID:** 2604.04937 | [PDF](https://arxiv.org/pdf/2604.04937v1)

**作者:** Sharath Sathish `[一作]` (University of York), Sharath Sathish `[通讯]` (University of York)

**通讯引用:** 49 | [OpenAlex ID](https://openalex.org/A5063281509)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在 LLM（Llama 3.2‑3B 与 DeepSeek‑R1‑Distill‑Llama‑8B）上使用 QLoRA 细调，构建了基于 Navya‑Nyaya 逻辑的 6 阶段认知框架（疑问分析、知识源识别、五元三段论、对抗性检验、谬误检测、结论鉴别）。该框架能让模型在解决约束满足、布尔 SAT、多步推理等逻辑问题时生成可追溯、可审计的推理链。

**💡 创新点**

①将古老的 Navya‑Nyaya 经验主义方法引入现代 LLM，首次实现“显式”认知结构。②通过细化 6 阶段流程，克服 LLM 的幻觉、无追溯推理等问题。③在极少量数据（20/55 个示例）下实现 100% 语义正确率，同时公开模型、数据与训练脚本，促进后续研究。

**🔧 技术方法**

技术栈包括：QLoRA 4‑bit LoRA 微调、Unsloth + TRL 的 SFTTrainer、YAML+Markdown 模板数据预处理、结构化校验器、语义评分（使用 GPT‑4/Claude 进行阶段评分）、Z3 SMT 形式化检验、Prompt engineering（强制格式化输出）、实验跟踪（Weights & Biases）。

**📊 数据集**

使用了 55 题 Navya‑Nyaya 结构化逻辑问题集，涵盖约束满足、布尔 SAT、转移推理、集合运算、多步推理等类别。数据以 YAML 前置元数据 + Markdown 结构化写成，方便验证与训练。

**📈 对比分析**

对照基线（未微调模型）进行评估。Stage 0 仅 20 题，训练 30 轮，得到 40% 结构符合率；Stage 1 55 题，10 轮，语义正确率提升至 100%，但结构符合率仍为 40%。通过多层评估（结构、语义、真值匹配）以及 Ablation（格式提示/温度交互）验证模型在推理内容上的可靠性，且训练成本低（<1 USD/阶段）。

**⚠️ 局限性**

主要限制包括：①评估样本极少，统计置信区间宽；②结构符合率远低于目标，说明单纯监督微调不足以强制遵守 6 阶段格式；③输出长度与 token 开销显著增加，适用性受限；④目前未对大型复杂推理（Boolean SAT、转移推理）实现；⑤缺乏正式形式化验证与自检机制，需要进一步引入 Z3 或规则检验。

---

## 5. Integration of Object Detection and Small VLMs for Construction Safety Hazard Identification

**arXiv ID:** 2604.05210 | [PDF](https://arxiv.org/pdf/2604.05210v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 6. Enhancing sample efficiency in reinforcement-learning-based flow control: replacing the critic with an adaptive reduced-order model

**arXiv ID:** 2604.04986 | [PDF](https://arxiv.org/pdf/2604.04986v1)

**作者:** Zesheng Yao `[一作]` (Zhejiang University), Mengqi Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 5329 | [OpenAlex ID](https://openalex.org/A5100751725)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种自适应降阶模型（ROM）驱动的强化学习框架，用于主动流体控制。

**💡 创新点**

创新点在于用物理导向的ROM替代传统模型自由强化学习中的价值网络，并通过运算符推理（OpInf）与神经常微分方程（NODE）实现在线自适应建模与梯度优化。

**🔧 技术方法**

采用运算符推理、神经ODE、梯度优化（Adam）以及可微分ROM求解器等技术；控制策略通过ROM求解得到梯度后在离线迭代中更新。

**📊 数据集**

使用基于OpenFOAM的数值仿真数据：Blasius边界层流（Reσ≈1000–1600）和方形圆柱后流（Re=100）进行数据收集和模型训练。

**📈 对比分析**

与传统ERA、LQR、线性控制以及TD3/SAC等模型自由强化学习方法比较，所提框架在两种基准流场中均以极少的样本（单次或数次仿真周期）实现与或优于这些方法的阻力降低或波动抑制性能。

**⚠️ 局限性**

主要局限包括：仅在二维层流或线性/弱非线性场景验证；对湍流或更高维系统的泛化性待验证；在训练过程中仍可能出现不稳定或收敛缓慢，需要进一步的鲁棒性和自适应机制。

---

## 7. From PDF to RAG-Ready: Evaluating Document Conversion Frameworks for Domain-Specific Question Answering

**arXiv ID:** 2604.04948 | [PDF](https://arxiv.org/pdf/2604.04948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 8. IntentScore: Intent-Conditioned Action Evaluation for Computer-Use Agents

**arXiv ID:** 2604.05157 | [PDF](https://arxiv.org/pdf/2604.05157v1)

**作者:** Rongqian Chen `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 6492 | [OpenAlex ID](https://openalex.org/A5018464968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一个轻量级的计划感知双编码器奖励模型，用于评估并重排序计算机使用代理的GUI动作。

**💡 创新点**

创新点在于将规划意图嵌入动作编码器、采用对比对齐与margin ranking的双重训练目标，以及跨操作系统的离线预训练，从而实现跨环境的动作质量评估。

**🔧 技术方法**

使用了InfoNCE对比损失、margin ranking排序损失、GRU历史编码、SigLIP+MPNet视觉与文本特征提取、温度化余弦相似度等技术。

**📊 数据集**

基于AgentNet三大操作系统（Windows、Mac、Ubuntu）的398k步离线轨迹进行训练，并在未见的OSWorld 361任务上进行评估。

**📈 对比分析**

与LLM自评、随机选择、仅对比对齐等方法比较，离线相邻步判别率达97.5%，在线OSWorld任务成功率提升6.9点（从45.2%提升到52.1%，覆盖率46.6%）。

**⚠️ 局限性**

主要局限在于离线训练与部署代理产生的动作分布存在差异，导致在线提升有限；未实现多步前瞻评估，且模型对新颖动作格式的适应仍不足。

---

## 9. Lightweight True In-Pixel Encryption with FeFET Enabled Pixel Design for Secure Imaging

**arXiv ID:** 2604.05147 | [PDF](https://arxiv.org/pdf/2604.05147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 10. MMORF: A Multi-agent Framework for Designing Multi-objective Retrosynthesis Planning Systems

**arXiv ID:** 2604.05075 | [PDF](https://arxiv.org/pdf/2604.05075v1)

**作者:** Frazier N. Baker `[一作]` (Ohio State University), Xia Ning `[通讯]` (Ohio State University)

**通讯引用:** 6126 | [OpenAlex ID](https://openalex.org/A5035648686)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套可模块化的多代理系统框架（MMORF），用于解决多目标退化合成规划，并演示了两种代表性系统。

**💡 创新点**

创新点在于提出了四个可组合的代理组件（规划器、价值函数校准器、法规约束器、路线评判器），首次实现了可灵活配置的多目标退化合成多代理系统，并通过实验验证其有效性。

**🔧 技术方法**

技术手段包括：基于大型语言模型的代理（价值函数调节、规则限制、路线评判），配合化学计算工具（致癌预测、起始材料成本估算、GHS 语句检索等）进行路线搜索与评估。

**📊 数据集**

使用了新构建的218个多目标退化合成任务基准，其中111个为约束任务（单/多约束），107个为无约束任务，用于评估系统性能。

**📈 对比分析**

与MO‑MCTS、传统LLM等基线方法进行对比，采用成功率、有效率、存在率等指标，MMORF在约束任务上取得48.6%的成功率，优于所有基线，并在安全与成本指标上常常支配其他方法。

**⚠️ 局限性**

局限性包括：系统设计较为复杂，导致规划延迟高、任务完成率低；在单约束或简单任务上可能不如轻量级方法；对底层LLM的依赖较强，生成路线仍需人工专家验证。

---

## 11. CURE:Circuit-Aware Unlearning for LLM-based Recommendation

**arXiv ID:** 2604.04982 | [PDF](https://arxiv.org/pdf/2604.04982v1)

**作者:** Ziheng Chen `[一作]` (Walmart Global Tech), Yang Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 84425 | [OpenAlex ID](https://openalex.org/A5100354659)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

为大语言模型推荐系统设计了 Circuit‑Aware Unlearning 框架 CURE，能够在不完整重训练的情况下去除敏感交互记录，同时保持推荐性能。

**💡 创新点**

创新点在于：① 将模型拆解为“忘记专属”“保留专属”与“功能共享”三类电路；② 针对不同电路采用梯度投影和专属更新策略，从而显著减少遗忘与保留目标之间的梯度冲突；③ 通过激活干预/补丁与图结构信息快速定位关键电路。

**🔧 技术方法**

主要技术包括：机制解释（电路发现）、激活干预与激活补丁、个性化 PageRank 生成对抗样本、梯度投影优化、LoRA 参数高效微调。

**📊 数据集**

使用 MovieLens‑1M 与 GoodReads 两个公开推荐基准；模型后端为 GPT‑2 Large 与 LLaMA‑2‑7B。

**📈 对比分析**

与 Retrain、SISA、RecEraser、E2URec、ROME、WISE、PCGrad 等基线相比，CURE 在 JSD（忘记完整度）上最低、AUC/ACC 维持接近 Retrain，同时在 Unlearning Time 上比最优基线快 3‑4 倍，整体效率提升 18%，推荐性能提升 6%。

**⚠️ 局限性**

局限性包括：① 电路发现与梯度投影仍需额外前向/后向计算，成本不如纯梯度剪枝；② 依赖用户‑物品图结构，若缺失图信息或图规模极大时效果可能受限；③ 目前仅在两种 LLM 结构验证，是否能迁移到更大、更异构的模型仍待进一步评估。

---

## 12. Solving Hard Instances from Knapsack and Bounded Knapsack Problems: A new state-of-the-art solver

**arXiv ID:** 2604.05232 | [PDF](https://arxiv.org/pdf/2604.05232v1)

**作者:** Renan F. F. da Silva `[一作]` (University of Campinas), Rafael C. S. Schouery `[通讯]` (University of Campinas)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种新型的基于核心动态规划（core‑based DP）与多重约简、项聚合、强化的基于优势的固定技术（fixing‑by‑dominance）以及增强可除性上界（divisibility bound）的组合求解器，用于经典背包问题（KP）和有界背包问题（BKP）

**💡 创新点**

创新点在于将传统核心 DP 与改进的约简与固定策略（多重约简、项聚合、修订的基于优势的固定、增强的可除性上界）相结合，并通过在线生成核心序列和惰性排序实现近线性时间性能，同时在困难实例上实现数十倍加速

**🔧 技术方法**

核心技术包括：核心动态规划、线性松弛与状态基 LP 上界、投影（surrogate）松弛、基于优势的固定、项聚合与多重约简、增强的可除性上界、贪婪完成启发式、配对启发式、采样配对启发式和重启策略

**📊 数据集**

使用了 Pisinger 背包基准集（含多类、不同规模、不同系数范围），自定义的 Bounded Knapsack 基准集（随机生成的多余量，避免重复项），以及 Jooken 生成的 3240 个大型实例（包含多组、不同容量）

**📈 对比分析**

与现有最强求解器 COMBO（KP）和 BOUKNAP（BKP）进行对比，实验显示在 Pisinger 的所有 31800 个 KP 实例中，RECORD 在大多数难类中比 COMBO 快数倍至数十倍；在 BKNAP 基准中，RECORD 在 7800 个实例中全部求解，平均时间显著优于 COMBO 与 BOUKNAP；在 Jooken 基准中，RECORD 将未求解实例从 315 降至 118，整体平均时间比 COMBO 高 5.29 倍；在易类实例上性能相近或略慢

**⚠️ 局限性**

局限性包括：在轻松实例（如无相关、弱相关、近强相关、相似权重、圆形类）上由于额外的约简与启发式开销略慢；实现上需要较大的内存预分配，且对有界背包的多重可用性假设较强；未对多维背包或多背包等扩展进行验证

---

## 13. Energy-Based Dynamical Models for Neurocomputation, Learning, and Optimization

**arXiv ID:** 2604.05042 | [PDF](https://arxiv.org/pdf/2604.05042v1)

**作者:** Arthur N. Montanari `[一作]` (Northwestern University), Adilson E. Motter `[通讯]` (Northwestern University)

**通讯引用:** 12135 | [OpenAlex ID](https://openalex.org/A5031220186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并统一能源基动力学模型（EDM）在神经计算、机器学习和非传统计算中的理论基础、典型实例（Hopfield网络、Boltzmann机、DenseAM、振荡器网络、近端梯度网络）及其学习机制、稳定性与存储容量等方面的最新进展。

**💡 创新点**

创新点在于把能量函数视作Lyapunov函数，将计算、学习与控制理论相结合，提出了多尺度学习框架、对比性Hebbian学习、等势传播、稀疏正则化等新型机制，并通过理论分析揭示了DenseAM与振荡器网络在存储容量与收敛性上的显著优势。

**🔧 技术方法**

使用的技术包括：连续/离散动力学、梯度流与预条件梯度流、随机梯度流（Langevin动力学）、投影梯度流、能量函数构造、对比学习、等势传播、稀疏正则化与近端梯度下降、控制理论工具（Lyapunov、Hurwitz、契约理论）以及大数与随机矩阵分析。

**📊 数据集**

论文为综述性工作，未在实验中使用具体公开数据集，主要以理论推导和案例模拟为主。

**📈 对比分析**

方法比较主要通过理论推导与数值模拟来评估：例如DenseAM与传统Hopfield网络的存储容量从线性提升到多项式甚至指数级；振荡器网络通过二次谐波调节来消除伪吸引子并提升错误无误差容量；近端梯度网络在稀疏重建与决策组合等任务中展示了与传统梯度下降相当的收敛速度和更好的可解释性；但整体缺乏统一的基准实验比较。

**⚠️ 局限性**

局限性包括：缺乏大规模实验验证和硬件实现评估；对非对称网络（符合Dale定律）的理论框架仍不完整；能量函数设计往往需要先验假设，难以直接从数据中自动学习；以及在实际部署时对噪声、温度等物理因素的鲁棒性需要进一步研究。

---

## 14. Simultaneous Dual-View Mammogram Synthesis Using Denoising Diffusion Probabilistic Models

**arXiv ID:** 2604.05110 | [PDF](https://arxiv.org/pdf/2604.05110v1)

**作者:** Jorge Alberto Garza-Abdala `[一作]` (Tecnologico de Monterrey), Jose G. Tamez-Pena `[通讯]` (Tecnologico de Monterrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了基于差分编码的三通道DDPM模型，用于同时合成乳腺影像的CC和MLO双视图。

**💡 创新点**

首次将CC、MLO视图以及它们绝对差值编码为RGB三通道，直接训练扩散模型实现双视图同步生成，保证跨视图解剖一致性。

**🔧 技术方法**

采用预训练的DDPM（ddpm-celebahq-256）在乳腺影像上微调，结合差分引导和后处理百分位归一化，实现三通道生成。

**📊 数据集**

使用TecSalud私有乳腺筛查数据库，包含22,040张双视图（11,020名患者）在2014-2019年间采集的非植入、BI‑RADS‑2病例。

**📈 对比分析**

通过视觉一致性评估、Otsu阈值生成乳腺掩膜计算IoU和Dice，并与真实样本做KS、EMD检验；结果显示合成样本在形状一致性上接近真实，IoU、Dice略高但统计显著差异，94%可视一致。

**⚠️ 局限性**

差分通道易放大小幅对齐误差导致伪影；评估仅限全局形状，缺乏病变级别或密度验证；单一编码策略可能限制多样性，需进一步条件化与外部验证。

---

## 15. Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling

**arXiv ID:** 2604.04987 | [PDF](https://arxiv.org/pdf/2604.04987v1)

**作者:** Yongchang Hao `[一作]` (University of Alberta), Lili Mou `[通讯]` (University of Alberta)

**通讯引用:** 6153 | [OpenAlex ID](https://openalex.org/A5024821632)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于约束优化的Cactus方法，改进了speculative sampling的接受率并保证与验证模型分布的偏差受控；

**💡 创新点**

其创新点在于将KL散度作为约束，引入可训练‑free的接受率与恢复概率，使得既能提高接受率，又能在理论上控制分布偏差；

**🔧 技术方法**

采用的技术包括：约束优化框架、KL散度约束、近似公式求解、以及基于草稿模型和验证模型的draft‑and‑verify采样；

**📊 数据集**

实验使用的公开数据集为GSM8K、IFEval和GPQA；

**📈 对比分析**

与传统SpS和TAS方法相比，Cactus在接受率和准确率上均表现更好，尤其在GPQA等复杂任务中保持了高准确率同时显著提升吞吐量；

**⚠️ 局限性**

限制在于需要手动调节δ参数，且在极端概率分布下可能无法完全满足预期约束；

---

## 16. Corporate Training in Brazilian Software Engineering: A Qualitative Study of Useful Learning Experiences

**arXiv ID:** 2604.05209 | [PDF](https://arxiv.org/pdf/2604.05209v1)

**作者:** Rodrigo Siqueira `[一作]` (CESAR School), Danilo Monteiro Ribeiro `[通讯]` (CESAR School)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对195名巴西软件工程师的开放式问卷回复进行主题分析，识别了最有价值的学习体验类型。

**💡 创新点**

首次将技术持续更新与实践应用、正式教育、社交学习、领导力发展等维度系统整合，并通过共现网络验证它们的互补关系。

**🔧 技术方法**

采用主题分析（六阶段方法）结合IRAMUTEQ词形还原、频率统计与共现网络分析。

**📊 数据集**

基于195条受访者的开放式回答（巴西软件工程专业人士）。

**📈 对比分析**

通过共现频数和频率展示主题间关联，无传统性能指标；结果显示技术更新与实践应用关联最强。

**⚠️ 局限性**

样本自选且主要为男性高学历者，单一编码者、便利抽样和回忆偏差可能影响结果的普适性。

---

## 17. Jeffreys Flow: Robust Boltzmann Generators for Rare Event Sampling via Parallel Tempering Distillation

**arXiv ID:** 2604.05303 | [PDF](https://arxiv.org/pdf/2604.05303v1)

**作者:** Guang Lin `[一作]` (Purdue University), Xuda Ye `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了 Jeffreys Flow，一种利用 Jeffreys 散度训练的 Boltzmann 生成器，用于高维、多模态罕见事件采样。

**💡 创新点**

创新点在于将正向 KL 与逆向 KL 对称融合，形成 Jeffreys 散度，既能抑制模式崩塌又能保持目标精度；并提出利用并行温度耦合（Parallel Tempering）样本的连续蒸馏框架和物理信息模式截断策略。

**🔧 技术方法**

主要技术包括可逆正则化流（RealNVP/Neural Spline Flow）、Jeffreys 散度损失、并行温度耦合采样、重要性重加权、SVRG/重振算法以及 Fourier 模式截断。

**📊 数据集**

在多种数据集上进行验证：二维多模态势（Tw、Himmelblau、Annulus、MW）、周期井势、3D高斯混合、4D Rosenbrock、8D Rastrigin、16D 带周期底物的溶剂模型、2D 高斯混合的 replica exchange SGLD、屏蔽泊松逆问题、以及 1D 双井势的路径积分 Monte Carlo。

**📈 对比分析**

与单向 KL（前向/后向）、传统 Boltzmann 生成器、经典 PT 以及直接 PT 链进行对比；结果显示 Jeffreys Flow 在 ESS、L² 偏差、自由能曲线和模式覆盖率上显著优于对照组，尤其在高维、非线性耦合场景中保持高有效样本数和低偏差。

**⚠️ 局限性**

主要局限包括：需要先行 PT 产生参考样本，初始 PT 计算成本高；对超参（θ、λ0/λ1、Resampling 阈值）的敏感性；在极高维、复杂流形或极度分散能景下，流模型的容量与训练稳定性可能受限；以及对真实物理系统的可解释性和迁移性仍需进一步研究。

---

## 18. Operational Noncommutativity in Sequential Metacognitive Judgments

**arXiv ID:** 2604.04938 | [PDF](https://arxiv.org/pdf/2604.04938v1)

**作者:** Enso O. Torres Alegre `[一作]` (Pontifical Catholic University of Chile), Diana E. Mora Jimenez `[通讯]` (University of Notre Dame)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个操作性框架，用以区分元认知评估中的顺序效应是由经典可观测的状态变化还是不可约的非可交换结构导致的，并给出了可检验的不等式和数值模型。

**💡 创新点**

通过引入对假设无侵入性经典模型（CFD+ENI）的检验，提供了从表面顺序效应到真正非可交换性的判定准则。

**🔧 技术方法**

利用内部状态空间、评估变换T_E与读取π_E的形式，结合概率论与非可交换代数思想，推导出三角不等式与期望相等性。

**📊 数据集**

未使用真实数据集，采用理论示例——三维旋转模型，演示可违反约束。

**📈 对比分析**

无实验比较，提供理论模型与实验范例的对比；若实现实验，可通过检验C_ij相等性来判定非可交换性。

**⚠️ 局限性**

缺乏经验验证，模型仅为概念证明；对噪声、样本量、状态非平稳性的考虑不足；未覆盖更弱经典模型的约束。

---

## 19. LLM2Manim: Pedagogy-Aware AI Generation of STEM Animations

**arXiv ID:** 2604.05266 | [PDF](https://arxiv.org/pdf/2604.05266v1)

**作者:** Aastha Joshi `[一作]` (San Diego State University), Jun Chen `[通讯]` (San Diego State University)

**通讯引用:** 6097 | [OpenAlex ID](https://openalex.org/A5100450257)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套半自动、人机协作的 LLM+Manim 管线，将教师或学生的自然语言概念描述转化为同步讲解的 STEM 动画，用于课堂教学。

**💡 创新点**

创新点在于：①将大型语言模型与 Manim 结合，构建可复用的提示模板与符号账本，实现叙述与代码的并行生成与自动纠错；②嵌入多媒体学习原则（分段、信号化、双重编码）保证叙述与视觉同步；③通过三阶段人机复核保证质量并保持可追溯性。

**🔧 技术方法**

使用技术包括：大型语言模型（如 GPT 系列）进行文本与代码生成、Python Manim 动画库、约束提示模板、符号账本与一致性校验、并行叙述/代码生成、自动错误重生成、人工审核与可视化渲染。

**📊 数据集**

实验数据来源为 100 名本科生（数学、物理、航空航天、计算机科学、信息系统等专业）在 San Diego State University 进行的 A‑B 交叉实验，使用四个 STEM 主题（线性变换、线性系统、特征值/特征向量、热力学）及对应的预测测试/后测题目，未使用公开大规模数据集。

**📈 对比分析**

采用 A‑B 交叉设计，使用 ANCOVA、配对 t 检验、Cohen’s d 等统计方法比较两种教学形式。结果显示：后测平均分动画 83.4 vs 幻灯片 78.1（p<.001）；学习增益 d=0.67；参与度 d=0.94；认知负荷降低 d=0.41；满意度 d=1.64；完成时间更短 d=0.86，均显示动画教学在效果、体验与效率上优于传统幻灯片。

**⚠️ 局限性**

限制包括：实验仅涵盖两门主题、单一机构、后测仅即时无长期保留测试、人工审核仍需人工干预、时间记录粗略、缺乏多元人群（性别、语言、社会经济）分析，以及 LLM 更新可能影响一致性与可重复性。

---

## 20. Not All Turns Are Equally Hard: Adaptive Thinking Budgets For Efficient Multi-Turn Reasoning

**arXiv ID:** 2604.05164 | [PDF](https://arxiv.org/pdf/2604.05164v1)

**作者:** Neharika Jali `[一作]` (Carnegie Mellon University), Gauri Joshi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 7842 | [OpenAlex ID](https://openalex.org/A5067441201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在全局token预算下的多轮推理计算效率问题，并提出基于预算分配的动态token分配策略，以实现更高的准确率与更低的token使用。

**💡 创新点**

创新点在于将多轮推理建模为多目标马尔可夫决策过程，并通过GRPO学习的Turn‑Adaptive Budgets（TAB）与All‑SubQ预算策略实现基于对话历史与未来子问题的自适应预算规划。

**🔧 技术方法**

采用GRPO强化学习训练预算分配模型，利用离散token预算集合，使用Qwen3系列LLM分别作为用户、预算器和求解器，并设计多目标奖励（准确率+预算惩罚）。

**📊 数据集**

在MATH、AMC23、MATH Level‑5、OlympiadBench、AIME25等数学推理基准上进行训练和评估，并在TheoremQA、BIG‑Bench Extra Hard、GPQA等外部数据集上测试。

**📈 对比分析**

与静态预算、LLM‑Judge单/多轮预算基线对比，TAB可在保持或提升准确率的同时节省约35% token，All‑SubQ在更高预算下可节省至40% token；在更大预算下依旧保持高准确率。

**⚠️ 局限性**

局限性包括仅适用于可验证奖励的推理任务，受限于离散预算桶和长周期的信用分配，未考虑非可验证任务、外部工具调用以及更长时间延迟反馈。

---

## 21. Extending Tabular Denoising Diffusion Probabilistic Models for Time-Series Data Generation

**arXiv ID:** 2604.05257 | [PDF](https://arxiv.org/pdf/2604.05257v1)

**作者:** Umang Dobhal `[一作]` (Kyushu Institute of Technology), Sozo Inoue `[通讯]` (Kyushu Institute of Technology)

**通讯引用:** 1752 | [OpenAlex ID](https://openalex.org/A5080895628)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在TabDDPM基础上加入时间步嵌入、条件嵌入和缺失值掩码，并加入轻量级Conv1D时序适配器，实现对多变量加速度时间序列的高质量、时间一致的合成。

**💡 创新点**

创新点在于：①利用时序适配器在不显著增加参数的前提下捕获局部时间依赖；②通过时间步与活动标签嵌入实现条件生成；③在噪声预测网络中加入缺失值掩码，使模型对不完整数据鲁棒。

**🔧 技术方法**

技术：扩展版TabDDPM、正向/反向扩散过程、时序适配器（Conv1D）、时间步嵌入、条件嵌入、缺失掩码、Cosine噪声调度、AdamW优化。

**📊 数据集**

WISDM加速度传感器数据（20Hz，三轴），采样长度100，六类活动。

**📈 对比分析**

与原始TabDDPM、SMOTE以及随机森林基线对比。分类准确率与宏F1相当（≈0.71/0.64），但在大二元转移矩阵和自相关分析上显著提升时间一致性；在少数类（坐着、站着）精度提高至近1。总体性能保持不变，但生成序列的时间质量更好。

**⚠️ 局限性**

限制：①仅在单一WISDM数据集验证，跨数据集泛化性未知；②虽然时间一致性提升，但对下游时序分类模型（LSTM/CNN）效果提升有限；③对长序列和更复杂多模态传感器的扩展尚未探讨。

---

## 22. Broken by Default: A Formal Verification Study of Security Vulnerabilities in AI-Generated Code

**arXiv ID:** 2604.05292 | [PDF](https://arxiv.org/pdf/2604.05292v1)

**作者:** Dominik Blain `[一作]` (Cobalt AI), Maxime Noiseux `[通讯]` (Cobalt AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了七种主流LLM在500个安全相关提示下生成代码的漏洞率，发现平均55.8%出现漏洞

**💡 创新点**

首次将Z3 SMT正式验证与运行时崩溃结合，量化LLM生成代码的真实可利用性，并展示工具检测差距

**🔧 技术方法**

使用COBALT静态分析+Z3 SMT求解器、GCC AddressSanitizer、行业工具（Semgrep、Bandit、Cppcheck、Clang SA、FlawFinder、CodeQL）

**📊 数据集**

生成3500个代码片段（500提示×7模型）以及50提示子集做额外实验

**📈 对比分析**

与六大行业工具对比，COBALT检测率为64.8%，而行业工具合计仅7.6%，Z3证明漏洞的97.8%未被检测，性能差距巨大

**⚠️ 局限性**

仅限单温度0的单轮生成，未涵盖多语言、复杂交互和高温度输出，且模型间差异受提示集限制

---

## 23. LLMs Should Express Uncertainty Explicitly

**arXiv ID:** 2604.05306 | [PDF](https://arxiv.org/pdf/2604.05306v1)

**作者:** Junyu Guo `[一作]` (University of California, Berkeley), Javad Lavaei `[通讯]` (University of California, Berkeley)

**通讯引用:** 6270 | [OpenAlex ID](https://openalex.org/A5042580848)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了两种大语言模型（LLM）的不确定性接口：一种是全局接口，通过在答案后输出可校准的置信度分数；另一种是局部接口，在推理过程中出现高风险状态时发出特殊标记（`<UNRELIABLE>`）。

**💡 创新点**

创新点在于：① 把不确定性视为可交互的接口而非仅仅是后置估计；② 通过强化学习（GRPO）训练模型直接生成置信度或标记；③ 两种接口在功能上互补——全局置信度用于答案可信度评估和检索触发决策，局部标记用于早期干预；④ 通过机制分析（logit‑lens、PCA、CKA）揭示两种接口在内部表示层面上的不同实现方式。

**🔧 技术方法**

主要技术包括：
- 基于强化学习的梯度无关策略优化（GRPO）对模型权重进行后训练；
- 特殊词标记生成（token‑level uncertainty token）以及置信度数字化输出；
- 对抗式奖励设计，鼓励模型在错误时低置信度或主动发标记；
- 机制分析方法：token‑level KL、logit‑lens、PCA 置信度嵌入可视化、CKA 计算层级几何差异、隐藏层线性探针预测检索触发。

**📊 数据集**

使用了五个常用的多步推理与开放域事实问答数据集：HotpotQA、MuSiQue、2WikiMultiHopQA、Natural Questions、TriviaQA；并在每个数据集上对比了全局/局部接口以及多种基线（后置校准、Self‑RAG、FLARE、ADARAGUE、DRAGIN 等）。

**📈 对比分析**

比较方法：在相同模型规模（13B）下，对全局接口进行精确匹配（EM）和 F1 评估；对局部接口使用检索触发指标（触发率、精确率、召回率）以及检索后 EM/F1。性能表现：
- 全局置信度接口使 ECE 从 0.383 降到 0.049，过度自信错误率从 52% 降至 4%；在检索控制任务中实现 41.6% EM / 50.5% F1，触发率 48%；
- 局部标记接口使错误答案的覆盖率从 15.1% 提升至 88.2%，触发率提升至 61%；在检索控制中实现 40.9% EM / 48.1% F1，触发率 61%。
- 与简单的后置校准、提示式发标记、线性探针等对比，GRPO 训练的全局接口在校准与检索控制上均显著优于基线，局部接口在高召回干预上优于自检索控制方法。

**⚠️ 局限性**

局限性：
- 仅在 13B 规模模型上验证，规模效应未知；
- 训练数据和奖励主要针对事实问答任务，可能对开放域生成或对话类任务的迁移性不足；
- 需要额外的推理时间和内存来生成置信度或标记，虽然对检索触发有帮助，但在实时系统中可能产生开销；
- 仅研究了两种接口形式，未探索多层次或更细粒度的可解释信号；
- 对奖励设计的稳定性与通用性尚未在更大范围内系统验证。

---

## 24. Generalizable Audio-Visual Navigation via Binaural Difference Attention and Action Transition Prediction

**arXiv ID:** 2604.05007 | [PDF](https://arxiv.org/pdf/2604.05007v1)

**作者:** Jia Li `[一作]` (Xinjiang University), Yinfeng Yu `[通讯]` (Xinjiang University)

**通讯引用:** 3729 | [OpenAlex ID](https://openalex.org/A5091800151)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BDATP框架，通过二声道差异注意力（BDA）提升声源空间感知，和动作转移预测（ATP）辅助任务强化策略泛化；

**💡 创新点**

创新点在于显式建模双耳差异以提取普适空间信息，并通过ATP正则化避免环境过拟合；

**🔧 技术方法**

采用CNN编码器结合BDA差异注意机制，并在PPO训练中加入ATP辅助损失；

**📊 数据集**

在Replica与Matterport3D两大真实3D环境的SoundSpaces模拟平台上进行实验；

**📈 对比分析**

与随机、方向跟随、SAVi、Dav-NaV、ORAN、AV-NaV、AV-WaN等基线比较，BDATP在大多数指标（SR/SPL/SNA）上取得SOTA，Replica未见声音场景下SR提升至21.6个百分点；

**⚠️ 局限性**

仅针对静态声源，在模拟环境下验证，尚缺乏动态声源与真实世界声学噪声的适应性与鲁棒性评估。

---

## 25. ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and Anomalous Traffic Detection in Resource-Constrained Organizations

**arXiv ID:** 2604.04952 | [PDF](https://arxiv.org/pdf/2604.04952v1)

**作者:** Alonso Isidoro Román `[一作]` `[通讯]` (Independent Researcher), Alonso Isidoro Román (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

实现了一个基于C++20的开源网络检测与响应系统ML Defender，可在150-200美元的裸机硬件上实时检测并阻止勒索软件和DDoS攻击。

**💡 创新点**

创新点包括嵌入式随机森林推理、双分数最大值检测策略、基于eBPF/XDP的零拷贝捕获以及人机协作的“Consejo de Sabios”同行评审与Test‑Driven Hardening方法。

**🔧 技术方法**

技术栈涵盖eBPF/XDP、ZeroMQ、Protocol Buffers、ChaCha20-Poly1305、ONNX Runtime、TinyLlama、ShardedFlowManager以及多插件架构。

**📊 数据集**

主要使用CTU‑13 Neris机器人网络流量和BigFlows数据集进行评估，训练采用合成数据。

**📈 对比分析**

通过对CTU‑13 Neris的评测，F1=0.9985、Precision=0.9969、Recall=1.0000，Fast Detector FPR为6.61%，ML Detector将FPR降至0；推理延迟0.24–1.06 µs；在虚拟化环境下吞吐量达约34–38 Mbps，CPU占用约65–73%。

**⚠️ 局限性**

局限性包括仅验证单一2011年botnet场景、缺失12个特征、训练数据仅覆盖2011–2017年、虚拟化环境导致吞吐受限、未对现代勒索软件和加密C2做评测、单节点部署和etcd单点等。

---

## 26. Generative AI for Video Trailer Synthesis: From Extractive Heuristics to Autoregressive Creativity

**arXiv ID:** 2604.04953 | [PDF](https://arxiv.org/pdf/2604.04953v1)

**作者:** Abhishek Dharmaratnakar `[一作]` (Google LLC), Anushree Sinha `[通讯]` (Google LLC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了从提取式剪辑到生成式剪辑的电影预告片自动生成技术演进，提出新的分类框架并评估最新模型。

**💡 创新点**

创新点在于建立了从启发式、图结构到LLM与基础模型的完整技术分类，提出针对生成式预告片的多维度评估体系，并讨论了可控生成与事实性挑战。

**🔧 技术方法**

使用技术包括：基于规则的剪辑、情感显著性检测、图卷积网络 (GCN)、LLM 主导的多阶段生成流水线、Transformer 自回归模型 (TGT)、以及文本到视频的扩散基础模型 (Sora、Veo 2)。

**📊 数据集**

采用的数据集有 MovieNet、MAD、Mmtrail、MovieQA 等，覆盖电影、电视剧及用户生成内容的多样化素材。

**📈 对比分析**

与现有方法比较时，TGT 在 MAD 基准上 F1 得分 52.38%（高于 CLIP‑It 41.73% 与 CCANet 31.63%），Levenshtein 距离 21.18 远优于 95.58，表明生成顺序更贴近专业剪辑。

**⚠️ 局限性**

局限性包括：高计算开销导致的可扩展性瓶颈、生成模型易产生幻觉与事实性错误、控制性不足（难以精准满足源视频约束）、评估指标仍偏主观且缺乏统一标准。

---

## 27. Instantaneous Planning, Control and Safety for Navigation in Unknown Underwater Spaces

**arXiv ID:** 2604.05310 | [PDF](https://arxiv.org/pdf/2604.05310v1)

**作者:** Veejay Karthik `[一作]` (Indian Institute of Technology Bombay), Leena Vachhani `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 644 | [OpenAlex ID](https://openalex.org/A5038367848)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于实时点云感知的AUV集成规划与控制框架，利用Lyapunov反馈控制器生成的球形不变集实现无全局定位的安全导航；

**💡 创新点**

创新点在于将不变集与实时感知直接耦合，形成可即时计算的安全轨迹生成方式，完全摆脱预先规划路径，且实现了对动态障碍物的在线重规划；

**🔧 技术方法**

使用了Lyapunov函数理论的非线性反馈控制、三维无节制运动模型、点云几何约束、ROS‑Gazebo仿真环境和RexRov模型；

**📊 数据集**

使用了在Gazebo模拟器中生成的RexRov点云数据集；

**📈 对比分析**

与传统PID控制器在路径跟踪任务中进行比较，结果显示所提控制器在路径长度相近、加速度更低、曲率更小、导航时间相当，且在目标区域的定位误差更小；

**⚠️ 局限性**

局限性包括仅基于简单的无节制三维运动模型，对大型动态障碍物的处理仍有限；需要较大的计算资源（约222ms/周期）；仅在仿真环境验证，未在真实水下环境中测试；

---

## 28. Learning Stable Predictors from Weak Supervision under Distribution Shift

**arXiv ID:** 2604.05002 | [PDF](https://arxiv.org/pdf/2604.05002v1)

**作者:** Mehrdad Shoeibi `[一作]` (University of Central Florida), Niloofar Yousefi `[通讯]` (University of Central Florida)

**通讯引用:** 350 | [OpenAlex ID](https://openalex.org/A5054474613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了弱监督学习在分布偏移下的鲁棒性，特别是针对CRISPR‑Cas13d转录组扰动实验中的监督漂移（supervision drift）问题。

**💡 创新点**

提出并量化了监督漂移这一在弱监督场景中容易被忽视的失败模式，并展示了特征稳定性可以作为一种轻量级的诊断工具。

**🔧 技术方法**

使用了线性模型（岭回归）、树基模型（随机森林、XGBoost）以及特征稳定性分析、漂移分数（shift‑score）和外部标签重构等技术。

**📊 数据集**

利用公开的HEK293FT与K562人类细胞系的RNA‑seq数据，包含Day 2和Day 7时间点的转录组扰动实验。

**📈 对比分析**

在相同细胞系内的“in‑domain”测试中，岭回归取得R²≈0.36、Spearman ρ≈0.44；跨细胞系（cross‑domain）迁移时性能降至ρ≈0.40；而时间迁移（temporal）失败，R²为负、ρ接近0，表明监督关系随时间变化。

**⚠️ 局限性**

局限包括仅在单一实验设置下验证、模型与特征种类有限、缺乏多数据集跨领域验证，以及理论分析仅为启发性假设。

---

## 29. Next-Scale Generative Reranking: A Tree-based Generative Rerank Method at Meituan

**arXiv ID:** 2604.05314 | [PDF](https://arxiv.org/pdf/2604.05314v1)

**作者:** Shuli Wang `[一作]` (Meituan), Xingxing Wang `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种名为 Next‑Scale Generation Reranking (NSGR) 的树状生成式重排框架，用于多阶段推荐系统中的列表重排任务。

**💡 创新点**

创新点主要有：① 引入下一尺度（coarse‑to‑fine）生成器，逐步细化列表，兼顾全局兴趣与局部上下文；② 设计多尺度邻居损失，利用树状多尺度评估器为生成器提供尺度特定的指导，解决生成器与评估器目标不一致的问题。

**🔧 技术方法**

使用了 Hierarchical Sequential Transduction Unit (HSTU) 提取用户兴趣，Transformer 自注意力与目标注意力实现上下文交叉，树状生成器 NSG 与多尺度评估器 MSE，结合多尺度邻居损失进行强化学习式训练。

**📊 数据集**

实验数据集包括公开的 Taobao Ad（约 26M 交互）和 Meituan 食品外卖工业数据集（约 242M 交互）。

**📈 对比分析**

通过与 PRM、GRN、NAR4Rec、DCDR、NLGR、YOLOR 等六种先进重排基线进行离线对比，NSGR 在 AUC/GAUC 上分别提升 0.0045–0.0153；在线 A/B 测试（20 候选）CTR 提升 2.89%，GMV 提升 3.15%；生成器一致性 HR@1% 0.861、HR@10% 0.987，表明生成效果显著优于基线。

**⚠️ 局限性**

局限性包括：在极大排列空间下仍需采样估计生成器一致性，温度和采样比例等超参数对性能敏感；模型训练和推理成本相对较高；目前仅在单一业务场景验证，缺乏跨平台的通用性评估。

---

## 30. PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing

**arXiv ID:** 2604.05018 | [PDF](https://arxiv.org/pdf/2604.05018v1)

**作者:** Yiwen Song `[一作]`, Jinsung Yoon `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多智能体框架，能够将非结构化的预写材料自动生成符合会议规范的完整论文稿；

**💡 创新点**

创新点在于：①将预写材料分解为思想摘要、实验日志、模板与规范，使用专门的智能体完成文献检索、可视化生成、章节撰写和迭代优化；②首次构建了200篇顶级AI会议论文的逆向数据集，提供标准化评测基准；

**🔧 技术方法**

技术方法包括检索增强生成（RAG）、基于VLM的图像生成与闭环迭代、Semantic Scholar API检索与 BibTeX 自动化、模拟同行评审的内容优化；

**📊 数据集**

数据集为200篇CVPR 2025与ICLR 2025录用论文的逆向抽取材料（思想摘要、实验日志、模板、规范及可选图表）；

**📈 对比分析**

通过与单智能体和已存在的E2E科研框架对比，人工评测显示在文献综述合成上胜过基线50%–68%，整体稿件质量提升14%–38%；

**⚠️ 局限性**

局限性包括：对LLM的hallucination仍有依赖、主要针对AI会议模板且未涵盖其它学科格式、逆向数据生成过程对原论文隐私与引用完整性有限制。

---

## 31. DualDiffusion: A Speculative Decoding Strategy for Masked Diffusion Models

**arXiv ID:** 2604.05250 | [PDF](https://arxiv.org/pdf/2604.05250v1)

**作者:** Satyam Goyal `[一作]` (University of Michigan), Arjun Laxman `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DualDiffusion 机制，使用快速的草稿模型与精确的验证模型交替进行，提升了面向掩码扩散模型的生成速度和质量。

**💡 创新点**

首次将推测式解码应用于非自回归的掩码扩散模型，并提出了三种验证算法（信任验证、KL 散度检验、置信度重掩码）来纠正草稿错误。

**🔧 技术方法**

利用 FastDLLM 或 dKV-Cache 作为快速草稿模型，LLaDA 作为高质量验证器，构建 Draft‑Verify 循环，并在验证阶段采用分布差异或置信度阈值策略进行重掩码。

**📊 数据集**

在 MMLU 与 GSM8K 两个基准上进行评估，检验模型在逻辑推理和数学算术等不同任务中的表现。

**📈 对比分析**

与仅使用 LLaDA 的全精度推理和仅使用 FastDLLM 的高速推理对比，DualDiffusion 在 MMLU 上实现 0.47 的准确率（仅比 LLaDA 降低 0.01）且速度提升 3.9×，在 GSM8K 上得到 0.25 的准确率（相较 LLaDA 下降 0.32）但仍保持 4× 的速度增益。

**⚠️ 局限性**

主要局限在于对高精度、多步推理任务（如 GSM8K）时验证策略不足，导致准确率显著下降，同时双模型并行使用导致显著的显存占用。

---

## 32. Beyond LLM-as-a-Judge: Deterministic Metrics for Multilingual Generative Text Evaluation

**arXiv ID:** 2604.05083 | [PDF](https://arxiv.org/pdf/2604.05083v1)

**作者:** Firoj Alam `[一作]` (Qatar Computing Research Institute, Hamad Bin Khalifa University), Shammur Absar Chowdhury `[通讯]` (Qatar Computing Research Institute, Hamad Bin Khalifa University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了OmniScore，一个小型确定性评估器，用来替代大规模LLM评判文本生成质量。

**💡 创新点**

创新点在于将多任务、多语言、多维度评估统一到少于1B参数的编码器模型，并通过合成监督训练与人类评测相结合，兼具低成本和可复现性。

**🔧 技术方法**

技术包括多输出回归的Transformer编码器、使用GPT-4.1生成合成标签、针对四维度(信息性、清晰度、可信度、忠实度)的多头回归，采用MSE、MAE、Pearson等指标。

**📊 数据集**

数据集涵盖107种语言的OmniScoreDataset(训练564k实例)以及人类标注的OmniScore‑Bench(8617条)，包括QA、MT、摘要、转写等五个任务。

**📈 对比分析**

通过与Gemini等大型LLM及其他编码器对比，OmniScore在MAE/ACC±1上分别达0.78/0.74，显著优于LLM基准且保持低推理延迟与成本。

**⚠️ 局限性**

局限性包括依赖GPT-4.1的合成监督可能带来教师偏差、对低资源语言的鲁棒性受限、以及在极长文本和复杂推理任务上可能无法匹配大型LLM。

---

## 33. Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking

**arXiv ID:** 2604.05268 | [PDF](https://arxiv.org/pdf/2604.05268v1)

**作者:** Chan-Wei Hu `[一作]`, Zhengzhong Tu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提供了使用 LuaLaTeX 或 XeLaTeX 的 ACL 风格文件的排版示例

**💡 创新点**

通过示例展示多语言文本与引用格式的使用，突出排版兼容性

**🔧 技术方法**

使用 LuaLaTeX 或 XeLaTeX 进行文档排版，配合 ACL 样式文件

**📊 数据集**

未使用实际数据集，仅包含示例文本（Hindi、Arabic 等）

**📈 对比分析**

无实验或方法比较，未给出性能评估

**⚠️ 局限性**

缺乏真实研究内容和实验数据，无法评估排版效果之外的学术价值

---

## 34. Models as Values in a Model Expression Algebra: A Functional Approach to Model Driven Engineering

**arXiv ID:** 2604.05001 | [PDF](https://arxiv.org/pdf/2604.05001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 35. FreakOut-LLM: The Effect of Emotional Stimuli on Safety Alignment

**arXiv ID:** 2604.04992 | [PDF](https://arxiv.org/pdf/2604.04992v1)

**作者:** Daniel Kuznetsov `[一作]` (Ben Gurion University of Negev), Asaf Shabtai `[通讯]` (Ben Gurion University of Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了 FreakOut-LLM 框架，系统性探究情绪刺激（尤其是压力）对大型语言模型安全对齐的影响，发现情绪诱导可显著提升 jailbreak 成功率。

**💡 创新点**

创新点在于首次将情绪诱导作为攻击表面，利用心理学验证的情绪刺激和量化心理测评证明情绪状态能显著预测模型的安全失效。

**🔧 技术方法**

使用 HarmBench 分类器评估 jailbreak 成功率，EMPALC+QLatent 进行模型心理测评，Logistic 回归和 Chi‑square 等统计方法检验情绪与安全失效的关联。

**📊 数据集**

数据集包括 10 个不同 LLM 的 119,600 条查询（5,200 条基线 + 114,400 条情绪诱导），以及 5 个心理测评量表（GAD‑7、PHQ‑9、SOSS、STAI‑S、SOC‑13）的自动化推断结果。

**📈 对比分析**

通过对比中性、放松与压力三种情绪条件的 jailbreak 成功率，发现压力条件下成功率提升 65.2%（OR = 1.67，Cohen’s d = 0.28），放松无显著效果；多模型 Logistic 回归确认压力是唯一显著预测因子。

**⚠️ 局限性**

局限性包括心理测评仅适用于可获取 token 级 log‑prob 的开源模型，无法验证所有 API 模型；缺少长格式中性对照导致长度匹配不完全；实验仅使用预设情绪刺激，未探索更广泛情绪谱系。

---

## 36. Hierarchical SVG Tokenization: Learning Compact Visual Programs for Scalable Vector Graphics Modeling

**arXiv ID:** 2604.05072 | [PDF](https://arxiv.org/pdf/2604.05072v1)

**作者:** Ximing Xing `[一作]` (Tencent HunYuan), Qian Yu `[通讯]` (Visual Computing Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种分层的 SVG 令牌化框架 HiVG，能够将原始 SVG 字符串拆分为结构化原子令牌，并将绘图指令与坐标合并为可执行的几何段令牌，从而大幅压缩序列长度并提升生成效率。

**💡 创新点**

创新点在于：①将 SVG 令牌化从字符级别迁移到结构级别，利用命令-参数组进行语义压缩；②设计 Hierarchical Mean–Noise (HMN) 初始化策略，将数值顺序和语义先验注入新令牌；③引入三阶段渐进式训练课程，以提升长序列生成的稳定性和几何一致性。

**🔧 技术方法**

技术手段包括：分层 tokenization（atomic & segment 级别），结构段学习（Merge 频繁相邻段），HMN 令牌初始化，基于预训练 Qwen2.5-VL-3B 的监督微调，三阶段 curriculum training。

**📊 数据集**

使用 2.45M 级别的公开 SVG 语料库，合并了三个开源数据集并进行去重与标准化，训练时以 784×784 画布统一坐标，测试覆盖文本到 SVG 与图像到 SVG 两个任务。

**📈 对比分析**

与多种基线（Qwen3.5、Gemini-2.5-pro、GPT‑5.2、SVGen‑7B、OmniSVG‑8B、InternSVG‑8B 等）进行比较；HiVG 在序列压缩率 63.8%、训练 token 数量减少 2.7×、图像相似度（CLIP‑S、LPIPS）和人类评测（可用性 4.06、pairwise 58.9–70.8%）等指标上均优于或相当于最先进方法。

**⚠️ 局限性**

局限性包括：仍受限于训练数据规模，较为复杂或三维图形的生成效果有限；HMN 初始化虽提升数值一致性，但对极端坐标分布仍可能出现漂移；分层压缩需在保持语法有效的前提下进行，过度压缩可能导致可视化细节丢失。

---

## 37. Phase-Associative Memory: Sequence Modeling in Complex Hilbert Space

**arXiv ID:** 2604.05030 | [PDF](https://arxiv.org/pdf/2604.05030v1)

**作者:** Gowrav Vishwakarma `[一作]` (Xavoc Technocrats Pvt Ltd), Christopher J. Agostino `[通讯]` (NPC Worldwide)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Phase‑Associative Memory (PAM)，一种全复杂数值的递归序列模型；

**💡 创新点**

创新点在于将关联以矩阵状态累积并通过共轭内积检索，解决向量状态模型的容量退化问题；

**🔧 技术方法**

使用复杂数值嵌入、modReLU、RMS 归一化、ComplexGatedUnit 与 PAM 层的交错组合，配合复杂 RoPE；

**📊 数据集**

在 WikiText‑103 数据集上训练与评估；

**📈 对比分析**

与参数相同的 Transformer 进行对比，PAM 100M 参数时验证 perplexity 30.0，Transformer 27.1；训练吞吐量低约 4 倍，但仍保持 10% 的 perplexity 差距；

**⚠️ 局限性**

限制在于性能仍落后于 Transformer、计算开销大、未使用自定义 CUDA 核心，且在更大规模下效果是否能进一步收敛仍待验证。

---

## 38. EvolveRouter: Co-Evolving Routing and Prompt for Multi-Agent Question Answering

**arXiv ID:** 2604.05149 | [PDF](https://arxiv.org/pdf/2604.05149v1)

**作者:** Jiatan Huang `[一作]` (University of Connecticut), Chuxu Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 5566 | [OpenAlex ID](https://openalex.org/A5022275632)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了EvolveRouter，一个可训练的框架，通过闭环共进化过程同时改进代理质量和协作结构，以解决现有路由方法的局限性。

**💡 创新点**

创新点在于结合图基查询路由与针对性指令优化，采用自适应推理策略动态确定每个查询的有效协作规模。

**🔧 技术方法**

使用了图神经网络（GNN）进行路由训练，并通过闭环优化过程进行代理指令的改进。

**📊 数据集**

在五个问答基准上进行了实验，包括WikiMultihopQA、HotpotQA、NewsQA、TriviaQA和NGQA。

**📈 对比分析**

与多种基线方法进行比较，EvolveRouter在F1和准确匹配（EM）上均表现优于现有的路由基线，尤其在TriviaQA上提升显著，表明其在提示任务对齐方面的优势。

**⚠️ 局限性**

局限性在于当前框架仍需进一步扩展代理池的动态性和自动化代理设计的整合。

---

## 39. ExpressMM: Expressive Mobile Manipulation Behaviors in Human-Robot Interactions

**arXiv ID:** 2604.05320 | [PDF](https://arxiv.org/pdf/2604.05320v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 40. Synthetic Trust Attacks: Modeling How Generative AI Manipulates Human Decisions in Social Engineering Fraud

**arXiv ID:** 2604.04951 | [PDF](https://arxiv.org/pdf/2604.04951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 41. Blind-Spot Mass: A Good-Turing Framework for Quantifying Deployment Coverage Risk in Machine Learning Systems

**arXiv ID:** 2604.05057 | [PDF](https://arxiv.org/pdf/2604.05057v1)

**作者:** Biplab Pal `[一作]` (University of Maryland Baltimore County), Madanjit Singh `[通讯]` (Ambient Scientific Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于Good–Turing估计的“盲点质量”指标，用于定量评估机器学习模型在实际部署状态空间中的覆盖风险，并通过分解覆盖盲点的概率质量为支持与盲点两部分，给出模型在不同数据覆盖下的准确率上限。

**💡 创新点**

创新点在于将传统的无见事件概率估计迁移到部署覆盖评估，定义了阈值化盲点质量 B_n(τ) 并与模型准确率分解关联，提供了针对高维、稀疏状态空间的可操作性诊断方法，并通过跨域验证证明其普适性。

**🔧 技术方法**

使用Good–Turing无见事件估计、频数-频数法、覆盖概率分解、风险加权盲点质量以及状态空间细化与归一化等技术；同时利用统计置信区间和 Wilson 置信区间评估准确率不确定性。

**📊 数据集**

实验数据集包括：一份来自工业合作伙伴的可穿戴加速度/陀螺仪 HAR 数据（12类、1674窗）；公开 PAMAP2 数据集（14类、1533窗）；以及 MIMIC‑IV 医院入院记录（275例）作为跨域验证。

**📈 对比分析**

与传统 OOD 检测、可靠性集预测和分布漂移监测方法对比，盲点质量提供了在部署前对整体概率质量的定量评估；在 HAR 任务中，随着状态细化 B_n(τ) 显著上升，说明实际覆盖不足，即使模型在标准测试集上取得高准确率，实际部署的可靠性仍受限。

**⚠️ 局限性**

局限性包括：盲点质量的阈值 τ 与实际可靠性需求的对应关系需经验确定；对状态空间细化与归一化的选择可能受领域知识限制；在极大状态空间下 Good–Turing 估计可能受频数分布偏差影响；此外，该指标未能直接指导模型架构改进，仅侧重数据覆盖诊断。

---

## 42. Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents

**arXiv ID:** 2604.04979 | [PDF](https://arxiv.org/pdf/2604.04979v1)

**作者:** Ádám Kovács `[一作]` `[通讯]` (KR Labs), Ádám Kovács (KR Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了针对编码代理的任务条件工具输出裁剪任务，构建了覆盖27种工具的基准数据集，并对Qwen 3.5 2B进行LoRA微调，提供了可直接部署的裁剪模型。

**💡 创新点**

创新点包括：①将工具输出裁剪定义为单个工具观察的最小语句块提取任务；②构造真实SWE‑bench与多生态系统合成数据的混合基准；③证明在该任务上，微小模型可显著优于大规模零射模型与传统启发式裁剪。

**🔧 技术方法**

技术手段：使用LoRA微调Qwen 3.5 2B；生成工具输出与聚焦查询的教师标注流程；评估指标为召回、F1、压缩率；与之对比的基线包括Qwen 3.5 35B A3B、Kimi K2、未微调Qwen 2B、BM25、First‑N、Last‑N、Random。

**📊 数据集**

数据集：共11,477个样例（SWE‑derived 9,205、Synthetic positives 1,697、Synthetic negatives 575），涵盖27种工具；测试集618例为人工复核，分布覆盖文件读取、日志、构建输出等多种格式。

**📈 对比分析**

评测方法：在618例保留集上计算召回、精确、F1、压缩率；结果显示微调后Qwen 3.5 2B取得召回0.86、F1 0.80、压缩率92%，比18倍规模的Qwen 3.5 35B A3B多出11点召回，且显著优于所有启发式基线；召回‑压缩曲线表明该模型在高压缩率下仍保持高召回。

**⚠️ 局限性**

局限性：仅评估单一工具观察的裁剪质量，未验证对完整代理推理轨迹的下游影响；使用跨度重叠作为准确性度量，无法覆盖所有合理裁剪方案；部分工具类型（如日志、构建输出）噪声较大，导致模型误裁。

---

## 43. Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models

**arXiv ID:** 2604.05309 | [PDF](https://arxiv.org/pdf/2604.05309v1)

**作者:** Yizhou Dang `[一作]` (Northeastern University), Zhu Sun `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 2163 | [OpenAlex ID](https://openalex.org/A5033957641)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对子序列拆分（SSS）在序列推荐中的作用进行系统评估，揭示其对模型性能评估的干扰与有效条件。

**💡 创新点**

首次指出SSS并非所有模型均有效，需结合特定拆分方法、目标策略与损失函数，且揭示其对实验结果的潜在偏倚。

**🔧 技术方法**

实验使用多种SSS策略（前缀、后缀、滑动窗口）、单目标/多目标训练、交叉熵与二元交叉熵损失，配合对比分析。

**📊 数据集**

采用四个小规模数据集（Beauty、Sports、Douyin、LastFM）以及两个大规模数据集（MovieLens‑1M、Amazon CDs）进行评测。

**📈 对比分析**

与经典基线（GRU4Rec、SASRec、NextItNet、LRURec）以及对比SSS的模型进行评测，去除SSS后多模型性能平均下降40%+，部分模型甚至低于SASRec。

**⚠️ 局限性**

仅聚焦基础模型，缺乏理论分析框架，对跨域、多模态等更复杂场景的影响未作评估。

---

## 44. EduIllustrate: Towards Scalable Automated Generation Of Multimodal Educational Content

**arXiv ID:** 2604.05005 | [PDF](https://arxiv.org/pdf/2604.05005v1)

**作者:** Shuzhen Bi `[一作]` (Shanghai Innovation Institute), Aimin Zhou `[通讯]` (Shanghai Innovation Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EduIllustrate 基准，用于评估大型语言模型在生成 K‑12 STEM 题目时的文本与图示交互式解释的能力，并给出了标准化的四阶段生成流程和 8 维评估框架。

**💡 创新点**

创新点包括：①创建首个多模态教育内容生成基准；②引入序列锚定（Scene 1 先规划后多场景并行）以确保跨图一致性；③基于多媒体学习理论的 8 维度评估标准；④利用 LLM‑as‑judge 进行自动评估并与人类专家对比验证。

**🔧 技术方法**

技术手段包括：大型语言模型（Gemini 3.0 Pro、GPT‑5、Claude Sonnet 4.5、Qwen 3.5 系列、Mistral 系列等）；Manim 渲染框架生成 PNG 图像；XML 结构化大纲和实现规划；序列锚定生成流程；使用 Gemini 3.0 Pro 作为评判模型进行自动评估；人类专家评价与成本分析。

**📊 数据集**

使用了 230 道 K‑12 STEM 题目数据集，覆盖五个学科（数学、物理、化学、生物、地理）和三个年级（小学、中学、高中），来源于 K12‑Vista 等公开资源。

**📈 对比分析**

比较方法：对 10 种 LLM 在 8 维度上给出 0‑5 分制评分并算几何平均得到总体得分；记录成功率；与完全并行生成方案对比，顺序锚定方案在视觉一致性上提升 13% 并节省 94% 成本。性能方面，Gemini 3.0 Pro 最高 87.8%，Kimi‑K2.5 80.8%；大多数模型在逻辑连贯和排版清晰上表现良好，但在图示与题目对齐及教学有效性上普遍偏弱。

**⚠️ 局限性**

局限性包括：只支持静态 PNG 输出，缺乏动画与多目标渲染框架；视觉维度（视觉一致性、元素布局）人机评估一致性低，提示评估模型对细节不敏感；Pedagogical Effectiveness 最弱，缺乏交互式或个性化教学；基准与评估主要针对单次生成，未覆盖多轮 Socratic 交互与学生特征匹配。

---

## 45. On the Exploitability of FTRL Dynamics

**arXiv ID:** 2604.05129 | [PDF](https://arxiv.org/pdf/2604.05129v1)

**作者:** Yiheng Su `[一作]` (University of Wisconsin--Madison), Emmanouil-Vasileios Vlatakis-Gkaragkounis `[通讯]` (University of Wisconsin--Madison)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究 FTRL 学习者在面对先知优化器时的可被利用性，揭示了学习率与次优动作数之间的逆比关系

**💡 创新点**

首次证明了无论正则化形式如何，FTRL 系列算法本质上易被操纵，且非陡峭与陡峭正则化产生截然不同的可利用性极限

**🔧 技术方法**

采用凸分析、Bregman 散度与连续-离散桥梁技术，结合随机零和游戏的概率论工具

**📊 数据集**

未使用真实数据集，仅在理论分析中对 i.i.d. 分布的随机零和游戏进行构造

**📈 对比分析**

通过与传统稳健分析对比，展示在固定策略下可利用性为 Θ(#次优动作/η)，而交替策略可实现 Θ(ηT) 的额外收益；实验与仿真验证了理论上界

**⚠️ 局限性**

结论局限于全信息反馈、常数步长和单一学习率设置，且对带噪声或半信息情形尚无完整推广

---

## 46. Planning to Explore: Curiosity-Driven Planning for LLM Test Generation

**arXiv ID:** 2604.05159 | [PDF](https://arxiv.org/pdf/2604.05159v1)

**作者:** Alfonso Amayuelas `[一作]` (University of California, Santa Barbara), William Wang `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 7757 | [OpenAlex ID](https://openalex.org/A5100702488)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于贝叶斯探索的LLM代码测试生成框架CovQValue，通过循环将覆盖率地图反馈给LLM，生成多样化的测试计划并用估计的Q值选择最具信息增益的计划，从而实现深层代码路径的高效探索。

**💡 创新点**

创新点在于将LLM测试生成视为贝叶斯探索问题，引入覆盖率地图作为后验统计量，并利用LLM估计的探索价值（Q值）进行多步规划选择，克服了传统贪婪策略在“通道结构”代码中的停滞。

**🔧 技术方法**

使用的技术包括LLM（Gemini 3 Flash、GPT‑5.4‑Mini、Mistral Large 3）、覆盖率地图反馈、Q值估计（即时信息增益+未来可达性）、多样化计划生成、贝叶斯探索框架。

**📊 数据集**

数据集包括公开基准TestGenEval Lite（160个Python文件）和作者自建RepoExploreBench（93个模块，来自9个热门Python包）。

**📈 对比分析**

通过与随机、贪婪、CovGreedy等基线对比，CovQValue在两个基准上分别提升了约51–77%（TestGenEval Lite）和40–74%（RepoExploreBench）的分支覆盖率，赢得77–84%的目标，显著优于基线。

**⚠️ 局限性**

局限性包括对LLM的依赖、估计Q值的非严格性、在更大规模或非Python项目上的适用性尚未验证，以及可能出现的测试通过率下降和对“开放式”环境缺乏安全边界。

---

## 47. Pressure, What Pressure? Sycophancy Disentanglement in Language Models via Reward Decomposition

**arXiv ID:** 2604.05279 | [PDF](https://arxiv.org/pdf/2604.05279v1)

**作者:** Muhammad Ahmed Mohsin `[一作]` (Stanford University), Emily Fox `[通讯]` (Stanford University)

**通讯引用:** 7213 | [OpenAlex ID](https://openalex.org/A5068568859)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过奖励分解与两阶段训练，降低大语言模型在面对权威提示和证据对立时的顺从（sycophancy）行为。

**💡 创新点**

创新点在于：①将顺从拆解为“压力独立性”和“证据响应性”两条正交失败模式；②设计多组件 GRPO 奖励（压力抗拒、上下文保真、立场一致、同意抑制、事实正确性）并在对比数据集上进行离散化训练；③两阶段训练（先SFT对齐无压力基线，再GRPO优化），实现独立惩罚各失败模式。

**🔧 技术方法**

技术包括：Group Relative Policy Optimization (GRPO)、对比式奖励归一化、NLI 语义距离与对抗判定、对话生成的压力模板、KL 正则化与长度惩罚、以及奖励权重的手工调优。

**📊 数据集**

数据集：从 4 个领域（数学、物理、观点、政治）构建 800 组对立上下文，每组产生压力级别 1-3 与对应的无压力基线，使用 NLI gate 保证基线互相矛盾；此外在评估时使用 SycophancyEval 与隐式压力测试集。

**📈 对比分析**

与基线（未调整的指令模型）对比，GRPO 在所有 7 种模型上均显著降低 PSS、GAS、提升 CFS、PACF；在 SycophancyEval 上约提升 15–17pp，证明在隐式压力情形下也具备迁移性。总体来看，奖励分解能在保持对事实正确性的同时抑制顺从。

**⚠️ 局限性**

局限性：只针对权威提示导致的顺从，未改善基于道德或情境挑战的顺从；对情感投资式压力的适应仍有限；奖励权重需要手工调优，未实现自适应；评估指标主要基于 NLI 与语义距离，对极端文本生成可能不完全覆盖。

---

## 48. Reasoning Through Chess: How Reasoning Evolves from Data Through Fine-Tuning and Reinforcement Learning

**arXiv ID:** 2604.05134 | [PDF](https://arxiv.org/pdf/2604.05134v1)

**作者:** Lucas Dionisopoulos `[一作]` (University of California), Prithviraj Ammanabrolu `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在7B Qwen2.5-Instruct基础上，先做有监督微调（SFT），再用强化学习（RL）进一步优化，通过多任务训练和自定义棋类数据集提升模型在棋局预测和解释的能力。

**💡 创新点**

①比较了两类目标数据集（单步最佳走法“Best Move”与多步走法序列“Best Line”）对SFT与RL效果的影响；②证明多步走法训练能保持“faithful reasoning”，而单步训练易出现推理不一致；③揭示RL能显著减少幻觉并提升走法质量，同时SFT时的一些评估指标（如推理质量、合法走法占比）可预测最终RL表现。

**🔧 技术方法**

采用链式推理（chain‑of‑thought）框架、LlamaFactory进行SFT、veRL结合Dr. GRPO进行RL，利用棋引擎评估奖励和验证走法合法性。

**📊 数据集**

自制四大类数据集：Rejection Sampling、Guided Synthetic、Programmatically Generated（包括 Factual Board Answering、Verbalized Alpha‑Beta Pruning、Best Move、Best Line），以及多任务混合训练。

**📈 对比分析**

与gpt‑oss‑120b及OpenAI o3等公开模型对比，在多项评估任务（Best Move、Worst Move、Legal Moves、Predict Move）上取得更高准确率和更低幻觉率，且RL后模型在走法质量分布上显著向上移动，稳定性更好。

**⚠️ 局限性**

仅在Qwen2.5‑7B上验证，缺乏跨模型复现；RL训练侧重中后期棋局，导致开局表现不足；对全局棋局对弈评估不足；未探索多轮RL或更精细奖励调节等进一步提升手段。

---

## 49. GLANCE: A Global-Local Coordination Multi-Agent Framework for Music-Grounded Non-Linear Video Editing

**arXiv ID:** 2604.05076 | [PDF](https://arxiv.org/pdf/2604.05076v1)

**作者:** Zihao Lin `[一作]` (UC Davis), Lifu Huang `[通讯]` (UC Davis)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 GLANCE 的多代理框架，用于音乐驱动的非线性视频混剪，采用外循环全局规划与内循环局部编辑的双循环结构，并在两者之间实现预防性上下文控制和纠正性冲突协商。

**💡 创新点**

创新点包括：①全局–局部协调机制，结合预防性上下文控制器与冲突图分解及自下而上的协商，显著缓解子时间线拼接后的冲突；②基于多代理的观察-思考-行动-验证（O-T-A-V）编辑流程；③构造 MVEBench benchmark 与 agent-as-a-judge 评价框架，为开放式混剪任务提供系统化评测。

**🔧 技术方法**

使用多模态大型语言模型（如 GPT‑4o‑mini、Gemini‑2‑pro、Qwen3‑VL‑30B 等）作为核心推理引擎；配合音乐分析、视频分析、检索、粗剪、对齐与修正等专门代理；并引入冲突图、区域分解与动态协商算法。

**📊 数据集**

采用自行构建的 MVEBench 数据集，包含 319 个评测样本，覆盖 8 种任务配置（按节奏型/故事型、音乐长度、提示细化程度分组），涉及 1,198 小时源视频和 645.3 分钟音乐。

**📈 对比分析**

在 MVEBench 上与 6 种基线（CoT、GPTSwarm、TeaserGen、EditDuet、VideoAgent、FunCLIP、NarratoAI）对比，GLANCE 在节奏对齐、情感对齐、指令跟随、整体质量、故事完整性、角色连续性等维度均明显优于基线；在使用 GPT‑4o‑mini 时，整体质量提升 33.2%/15.6%；开放源模型 Qwen3‑VL‑30B 亦超过 VideoAgent，表明框架本身有效。

**⚠️ 局限性**

局限性包括：①对大型 LLM 的高度依赖，导致算力和推理成本显著；②评测主要基于自动化 agent‑as‑a‑judge，尽管与人工评测一致，但仍无法完全覆盖主观美学与创意多样性；③当前仅针对音乐驱动的混剪，未深入探讨更复杂叙事或跨媒体协同场景；④在极长音乐或极细化提示下，仍可能出现全局一致性不足或过度匹配问题。

---

## 50. EffiPair: Improving the Efficiency of LLM-generated Code with Relative Contrastive Feedback

**arXiv ID:** 2604.05137 | [PDF](https://arxiv.org/pdf/2604.05137v1)

**作者:** Samira Hajizadeh `[一作]` (Columbia University), Suman Jana `[通讯]` (Columbia University)

**通讯引用:** 8434 | [OpenAlex ID](https://openalex.org/A5016425387)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种不需要模型微调的推理时框架，利用相似候选程序之间的对比性执行反馈（RCF）来优化大型语言模型生成的代码效率。

**💡 创新点**

创新点在于将绝对性能指标转化为相对对比信号，通过对结构相似、效率差距明显的程序对进行精简的差异总结，提供更具可操作性的指导，从而显著降低了调优的执行与令牌开销。

**🔧 技术方法**

核心技术包括：
- 代码相似度计算（融合语义嵌入与 AST 余弦相似度）；
- 采样式轻量化分析器（Scalene）提取热点摘要；
- 对比式反馈生成（RCF）并将其注入到多轮候选池的迭代修正过程；
- 在推理时对多条候选程序进行并行执行与对比。

**📊 数据集**

使用了 EvalPerf、Mercury、ENAMEL 三大代码效率基准；模型覆盖 GPT‑5.4、GPT‑5.4 mini、Claude Sonnet 4.6、DeepSeek‑Chat V3.2、GPT‑4o mini 等强大语言模型。

**📈 对比分析**

与 EffiLearner 等基线对比，RCF 框架在保持甚至提升 Pass@1 的前提下，平均速度提升约 1.5×、DPS/DPS_norm 得分提升 1–2 点，同时提示与完成令牌量从数百万降至仅 200 k/77 k；多轮迭代后仍保持稳定或略升的正确率，显示出优于单候选或绝对反馈的效果。

**⚠️ 局限性**

局限性包括：对候选多样性和相似度阈值高度依赖，若候选程序缺乏可比性或热点信息不稳定，RCF 信号可能失效；采样式分析仍有一定执行成本；目前仅评估了功能正确性与效率，未覆盖可读性、可维护性等软件质量；适用场景需权衡性能提升与推理/执行开销。

---

## 51. MedGemma 1.5 Technical Report

**arXiv ID:** 2604.05081 | [PDF](https://arxiv.org/pdf/2604.05081v1)

**作者:** Andrew Sellergren `[一作]` (Google), Daniel Golden `[通讯]` (Google)

**通讯引用:** 3106 | [OpenAlex ID](https://openalex.org/A5114002780)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出MedGemma 1.5 4B，一款集成多模态（CT/MRI体积、全切片图像、X光定位、纵向分析）和文本理解于一体的开源医疗基础模型

**💡 创新点**

通过新增3D体积切片处理、WSI补丁抽样、强化训练与强化学习的蒸馏，扩展到高维医学影像和多时间点分析，并在单一4B参数规模下实现多任务表现

**🔧 技术方法**

采用Gemma 3架构，Vision Encoder为MedSigLIP；预训练结合监督微调、蒸馏与RL；3D影像切片序列化、CT窗口映射、WSI多级抽样；多模态提示工程和JSON输出格式化

**📊 数据集**

使用了多种内部和公开数据集，包括CXR‑IND1、CT Dataset 1、MRI Dataset 1、Chest ImaGenome、WSI内部数据、EHR Dataset 2‑5、Mendeley临床实验室报告、MS‑CXR‑T、MedQA、MedXpertQA、MedMCQA等

**📈 对比分析**

与MedGemma 1、Qwen3 VL 4B、Gemma 34B/327B、Gemini 3.0Flash/Pro等模型对比；在新任务上MedGemma 1.5在3D CT/MRI分类、WSI文本生成、X光定位、纵向分析等指标均超过同类模型；在文本QA上提升MedQA +5%、EHRQA +22%；整体性能提升显著，但在某些传统基准如SLAKE、VQA‑RAD略有退化

**⚠️ 局限性**

模型在扩大多模态能力时出现轻微的旧基准退化；对高维影像和定位任务依赖大量预处理与硬件；由于训练数据仍为内部或部分公开集，跨域泛化仍需进一步验证

---

## 52. Synchronous Observer Design for Landmark-Inertial SLAM with Magnetometer and Intermittent GNSS Measurements

**arXiv ID:** 2604.05156 | [PDF](https://arxiv.org/pdf/2604.05156v1)

**作者:** Arkadeep Saha `[一作]` (Indian Institute of Technology Bombay), Ravi Banavar `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出了一种同步非线性观测器，用于在仅有标定位姿与IMU数据的情况下，通过加入间歇性的GNSS位置测量和磁力计测量，完成完整的Landmark‑Inertial SLAM状态估计。

**💡 创新点**

创新点包括：①利用同步观测器的模块化特性，分别设计GNSS、地标和磁力计的校正项，再将其叠加；②在观测器设计中对GNSS可用性做出时间上持续激励（TPE）假设，证明在该假设下误差几乎全局渐进稳定且局部指数稳定；③在观测器误差动态中引入辅助状态Z，消除了对系统输入的依赖，进一步简化了设计与分析。

**🔧 技术方法**

使用的技术主要是：Lie群框架（SE$_{n+2}$(3) 与 SIM$_{n+3}$(3)），同步观测器结构，Lyapunov 能量函数分析，时间上持续激励理论，以及高频 Lie 群欧拉积分仿真。

**📊 数据集**

数据集：作者仅在仿真环境中验证，使用了一个在圆轨迹上运动、包含5个地标的二维平面场景，仿真时间40 s，GNSS 信号间歇性可用。

**📈 对比分析**

方法比较：未在论文中与其他滤波/优化方法进行定量对比，但通过仿真展示误差随时间趋零、Lyapunov 函数严格递减，并指出仅在GNSS可用时位置误差显著下降。性能表现为几乎全局收敛，局部指数收敛速度可调。

**⚠️ 局限性**

局限性：需要GNSS信号满足时间上持续激励（TPE）假设；若GNSS完全失效或磁力计方向漂移，观测器无法保证全局可观测性；仿真仅在理想噪声环境下进行，缺乏真实世界实验验证。

---

## 53. AutoLALA: Automatic Loop Algebraic Locality Analysis for AI and HPC Kernels

**arXiv ID:** 2604.05066 | [PDF](https://arxiv.org/pdf/2604.05066v1)

**作者:** Yifan Zhu `[一作]` (University of Rochester), Chen Ding `[通讯]` (University of Rochester)

**通讯引用:** 15829 | [OpenAlex ID](https://openalex.org/A5100663432)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构建了一个基于DSL的自动循环局部性分析工具AutoLALA，能够将仿射循环程序下沉到多维整数集合与映射，直接推导出符号闭式的数据移动距离（DMD）公式；

**💡 创新点**

创新点在于将重用距离定义为访问空间映射的像，借助Barvinok整数点计数实现全符号重用距离分布，避免了传统栈模拟或递归工作集计算，并通过DSL、Rust实现与Web Playground等完整生态提供易用的符号分析平台；

**🔧 技术方法**

采用DSL（通过logos + LALRPOP）、polyhedral模型（ISL）、Barvinok、Rust语言、Monaco编辑器与KaTeX渲染等技术；

**📊 数据集**

未针对特定数据集，而是对通用HPC/AI工作负载（矩阵乘法、张量收缩、einsum、算子、卷积、图像处理管线等）进行符号分析；

**📈 对比分析**

通过生成的符号DMD公式，可直接比较不同算法（如naïve vs tiled矩阵乘法）在数据移动上的差异；在示例中展示了矩阵乘法、Jacobi算子等的符号闭式验证，表明方法与传统经验模型一致且精确；

**⚠️ 局限性**

局限性包括仅支持仿射循环结构，无法处理数据依赖的控制流、间接访问或非仿射子脚本；对高维、复杂循环的Barvinok计数可能导致计算爆炸。

---

## 54. On the Geometry of Positional Encodings in Transformers

**arXiv ID:** 2604.05217 | [PDF](https://arxiv.org/pdf/2604.05217v1)

**作者:** Giansalvo Cirrincione `[一作]` (Université de Picardie Jules Verne), Giansalvo Cirrincione `[通讯]` (Université de Picardie Jules Verne)

**通讯引用:** 3431 | [OpenAlex ID](https://openalex.org/A5010351972)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从理论角度研究Transformer中的位置编码，证明其必要性并阐明训练中位置向量的分离与最优性，随后构造基于Hellinger距离的多维尺度（MDS）最优编码并提出低秩参数化方案。

**💡 创新点**

创新点包括：①证明无位置编码模型对序列顺序不敏感；②提出位置分离定理，说明全局最优解必须给每个位置分配唯一向量；③基于Hellinger距离设计MDS最优编码并定义“应力”指标；④发现最优编码的有效秩受数据几何限制，可实现参数显著压缩。

**🔧 技术方法**

使用的技术包括Permutation Equivariance理论、Hellinger距离度量、经典多维尺度(MDS)、Neural Tangent Kernel（NTK）分析、梯度流与凸分析以及实验验证。

**📊 数据集**

实验所用数据集：自定义的三种位置分布合成语料、Stanford Sentiment Treebank（SST‑2）和IMDB情感分类数据集。

**📈 对比分析**

通过计算各编码的应力（stress）值进行比较；MDS编码在合成语料几乎消除应力，ALiBi在SST‑2上的应力远低于sinusoidal/ RoPE；低秩MDS在SST‑2可将应力降至≈0，同时参数量减少约90%，显示出优异的几何拟合与参数效率。

**⚠️ 局限性**

局限性：理论结果仅针对全局最优而非局部收敛；低秩参数化在训练过程中构成非凸优化；应力与下游任务性能的直接关联尚未得到证明；NTK证明仅适用于近似线性训练阶段。

---

## 55. From Video to Control: A Survey of Learning Manipulation Interfaces from Temporal Visual Data

**arXiv ID:** 2604.04974 | [PDF](https://arxiv.org/pdf/2604.04974v1)

**作者:** Linfang Zheng `[一作]` (University of Hong Kong), Wei Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 34374 | [OpenAlex ID](https://openalex.org/A5100622062)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了利用非动作标注视频学习机器人操控接口的研究进展，提出了以接口位置和显式程度为轴的三类方法分类，并从控制环路闭合、可验证性及失效点等角度系统分析不同方法的控制集成特性。

**💡 创新点**

创新点在于将视频与控制的连接视为接口，形成“接口显式度-与动作的距离”两维设计空间；提出了三类接口设计（直接视频-动作、潜在动作、中介视觉接口）并在此框架下梳理现有方法，揭示了机器人集成层面最关键的开放挑战。

**🔧 技术方法**

主要技术包括大规模无标注视频的时间预测（视频生成、视频预测、视频计划等）、多阶段预训练+细调、潜在变量学习（连续信息瓶颈或离散量化）、以及与机器人动作的轻量级对齐（解码器、字典、适配器、头替换）。

**📊 数据集**

使用了多种大规模视频数据集如 Ego4D、EPIC‑Kitchens、HowTo100M、OXR、MetaWorld、CALVIN、RT‑1 等，以及在机器人上收集的少量动作标注数据。

**📈 对比分析**

本文并未进行实验对比，而是对各类方法在控制环路闭合方式、可验证性以及失败来源进行比较分析，并指出目前评测碎片化、跨方法性能对比困难。

**⚠️ 局限性**

局限在于仅聚焦非动作标注视频驱动的操控接口，排除了大量基于静态图像或仅用机器人交互数据训练的方法；对实验评测缺乏统一基准；并且在跨模态、跨机器人以及长期规划方面仍存在显著挑战。

---

## 56. Understanding Clinician Experiences with Game-Based Interventions for Autistic Children to Inform a Future Game Platform Focused on Improving Motor Skills

**arXiv ID:** 2604.05249 | [PDF](https://arxiv.org/pdf/2604.05249v1)

**作者:** Hunter M Beach `[一作]` (Northern Arizona University), Jared Duval `[通讯]` (Northern Arizona University)

**通讯引用:** 349 | [OpenAlex ID](https://openalex.org/A5069133930)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过访谈和参与式设计工作坊，研究并归纳出临床治疗师对游戏化干预的需求与障碍，基于此提出并原型化了AutMotion Studio——一个模块化、可自定义且采用Wizard‑of‑Oz评估机制的游戏平台，旨在提升自闭症儿童的运动技能。

**💡 创新点**

①将治疗师的专业判断嵌入到游戏反馈流程中，采用Wizard‑of‑Oz而非自动化评分；②综合八大主题（任务拆分、灵活性、治疗专一性、可访问性、感官自适应、技能迁移、儿童自主、社交互动）构建系统；③提出“地图化”练习配置与家庭模式，促进技能在临床外的持续练习。

**🔧 技术方法**

使用半结构化访谈、主题分析、参与式设计、Figma原型制作、QR码链接视频、手机+电视投射的多屏交互，以及三按钮（Approve/Partial/Retry）评估机制。

**📊 数据集**

访谈数据：9名儿科物理与职业治疗师的访谈记录（共9份录音转写），未使用公开数据集；原型基于工作坊收集的低保真草图与概念图。

**📈 对比分析**

本研究未进行实验对比或性能评估；评价基于主题一致性与参与者对原型的接受度，缺乏定量性能指标。

**⚠️ 局限性**

局限性：①样本仅为治疗师，缺少自闭症儿童与家长的直接参与；②未在真实临床环境中验证原型效果；③样本量有限，结果可能不具广泛代表性；④缺乏定量评估与长期跟踪数据。

---

## 57. ReVEL: Multi-Turn Reflective LLM-Guided Heuristic Evolution via Structured Performance Feedback

**arXiv ID:** 2604.04940 | [PDF](https://arxiv.org/pdf/2604.04940v1)

**作者:** Cuong Van Duc `[一作]` (Hanoi University of Science and Technology), Binh Huynh Thi Thanh `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 2559 | [OpenAlex ID](https://openalex.org/A5072105691)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ReVEL框架，将大型语言模型与进化算法结合，通过多轮反思与结构化性能反馈实现启发式算法的进化；

**💡 创新点**

创新点在于将LLM嵌入多轮交互式推理，利用性能分组和群组反馈进行有针对性的反思，突破单次生成的脆弱性；

**🔧 技术方法**

采用进化算法、LLM（如DeepSeek V3等）、性能向量聚类、CodeBLEU相似度、基于熵的异质分组以及多轮反思循环等技术；

**📊 数据集**

使用TSP（10–200节点）与在线Bin Packing（容量100–500）两个经典组合优化基准集；

**📈 对比分析**

与EoH、ReEvo及传统启发式方法在最优性缺口指标上对比，ReVEL在TSP和BPP上均显著降低缺口，提升幅度约为1–3%或更小；

**⚠️ 局限性**

局限性包括结构化反馈可能掩盖细粒度差异导致无效改进；实验仅覆盖经典基准，缺乏更广泛的真实场景验证；对LLM提示与随机性高度依赖，影响重现性。

---

## 58. PCA-Driven Adaptive Sensor Triage for Edge AI Inference

**arXiv ID:** 2604.05045 | [PDF](https://arxiv.org/pdf/2604.05045v1)

**作者:** Ankit Hemant Lade `[一作]` (Independent Researchers), Akanksha Tiwari `[通讯]` (Independent Researchers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了PCA-Triage，一种流式无监督算法，用增量PCA权重分配每个传感器的采样率以满足带宽预算；

**💡 创新点**

创新点在于将增量PCA的主成分负载直接转换为按比例的通道采样率，利用相关结构实现高效的传感器三角化，并提供零训练参数、O(wdk)时间复杂度；

**🔧 技术方法**

使用增量PCA、指数移动平均平滑、功率律锐化、线性插值重建及可调混合权重（PCA+方差）等技术；

**📊 数据集**

在7个工业与仿真数据集上验证，包括TEP化工过程、SMD服务器、MSL航天器、PSM服务器、HAI工业控制、SKAB水循环、SWaT水处理；

**📈 对比分析**

与9个基线（Uniform、Variance、Threshold、Random Dropout、Autoencoder、Mutual Info、LSTM‑Attention、Transformer‑Attention、OGD）比较，PCA‑Triage在多数数据集上以无监督方法获得最高或最接近最高F1（如TEP 50%带宽F1 0.961≈全量），并在边缘设备上单窗口耗时0.67 ms；

**⚠️ 局限性**

局限性包括对相关结构的依赖（独立通道效果近似方差分配）、需手动设定k、α、γ等超参、对极端低带宽（<20%）下重建误差大、对时序漂移敏感、实验仅为离线回放，未覆盖实时网络延迟与同步问题；

---

## 59. Active Measurement of Two-Point Correlations

**arXiv ID:** 2604.05227 | [PDF](https://arxiv.org/pdf/2604.05227v1)

**作者:** Max Hamilton `[一作]` (University of Massachusetts), Subhransu Maji `[通讯]` (University of Massachusetts)

**通讯引用:** 20677 | [OpenAlex ID](https://openalex.org/A5052551454)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种人机交互框架，用少量人工标注结合预训练分类器高效估计目标源的两点相关函数。

**💡 创新点**

创新点在于设计了基于子集重要性采样的无偏估计器、有效的子集采样策略以及跨距离箱的置信区间估计，从而显著降低方差并减少标注工作。

**🔧 技术方法**

使用了子集蒙特卡洛估计、重要性采样、Horvitz–Thompson 重加权、Delta 方法逼近方差、以及自适应采样与置信区间构造等统计与机器学习技术。

**📊 数据集**

数据集为 JWST FEAST 程序的 NGC 628 与 NGC 4449 两个星系的多波段图像及其星团标签，包含数千个候选源和数十万条潜在边。

**📈 对比分析**

与均匀随机抽样的 Monte Carlo 基线和仅基于分类器预测的基线比较，结果显示在标注率 20% 时误差约降低 33%，所需标签量减少 33%，置信区间覆盖率接近 95%。

**⚠️ 局限性**

局限性包括对分类器准确率高度依赖；当目标子集极度稀疏或分类器表现差时，估计方差可能升高，且方法在非常大规模图上的实现仍需进一步优化。

---

## 60. SUMMIR: A Hallucination-Aware Framework for Ranking Sports Insights from LLMs

**arXiv ID:** 2604.04947 | [PDF](https://arxiv.org/pdf/2604.04947v1)

**作者:** Nitish Kumar `[一作]` (Indian Institute of Technology Patna), Sriparna Saha `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 8085 | [OpenAlex ID](https://openalex.org/A5060797340)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型自动从体育新闻中提取赛前赛后洞见，并对洞见进行真实性检测与基于用户兴趣的排序

**💡 创新点**

构建SUMMIR框架，将多维特征与PPO强化学习结合，实现幻觉感知的洞见排序，并提出基于FactScore和SummaC的幻觉检测方法

**🔧 技术方法**

使用GPT‑4o、Qwen、Llama等LLM生成洞见，采用FactScore、SummaC评估幻觉，利用ScoreNet、情感、讽刺检测、TF‑IDF、NLP实体识别等特征，再通过PPO优化排序策略

**📊 数据集**

构建包含7900篇、800场比赛（板球、足球、篮球、棒球）的新闻数据集，每场至少两篇赛前两篇赛后文章

**📈 对比分析**

与人工黄金排序对比，SUMMIR在NDCG@10/Recall@10分别达到0.94/0.96，优于单纯NDCG或Recall的0.86/0.91；在幻觉检测上GPT‑4o的FactScore/SummaC最高

**⚠️ 局限性**

对实体名的过度依赖导致排名偏倚；讽刺检测误判影响情感评分；长文本语义漂移和ScoreNet对特征相似度敏感导致排序不稳定

---

## 61. PRIME: Prototype-Driven Multimodal Pretraining for Cancer Prognosis with Missing Modalities

**arXiv ID:** 2604.04999 | [PDF](https://arxiv.org/pdf/2604.04999v1)

**作者:** Kai Yu `[一作]` (University of Minnesota), Rui Zhang `[通讯]` (University of Minnesota)

**通讯引用:** 12134 | [OpenAlex ID](https://openalex.org/A5100675481)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向癌症预后任务的缺失模态自监督预训练框架PRIME，利用共享原型记忆库在潜在空间中实现缺失模态的推断，并通过两项对比学习目标实现跨模态对齐与融合一致性训练。

**💡 创新点**

创新点在于：① 将缺失模态视为结构性属性，构建可共享的原型记忆库进行语义级别的缺失填充；② 设计了两种互补的预训练目标（跨模态对齐与融合一致性），在部分缺失数据上实现高效的自监督学习；③ 通过稀疏专家混合模块提升模型容量同时保持可解释性。

**🔧 技术方法**

核心技术包括：多模态特征预提取（WSI ViT、BulkRNA‑BERT、BioClinicalBERT）、查询式交叉注意力变形、共享原型记忆池的soft‑assignment与动态混合、MoE混合专家Transformer、InfoNCE对比学习、Dirichlet‑驱动的结构化缺失增强。

**📊 数据集**

使用TCGA 32种癌症的全拼合数据，共10439例，至少有一种模态；在5个完整三模态子集（UCEC、LUAD、LGG、BRCA、BLCA）上进行下游评估。

**📈 对比分析**

与多种基线（单模态、早/晚/交叉注意力、TensorFusion、MAGGate、MulT、MCAT、Porpoise等）比较，PRIME在三项指标（OS C‑index、3年死亡 AUROC、3年复发 AUROC）上均实现宏观平均最优：C‑index 0.653、AUROC 0.689、AUROC 0.637，且在缺失模态推理下表现稳健，线性探针可匹敌甚至超越全微调。

**⚠️ 局限性**

主要局限包括：① 在预训练时仅使用完整模态样本的缺失模拟，未完全覆盖不完整监督训练场景；② 所有方法均共享预先提取的模态嵌入，未评估端到端编码器微调的潜在提升；③ 未包含最新的多模态预训练基准（如mSTAR、POMP、MICE）做直接对比；④ 缺乏跨域外部验证，未检验模型在域迁移和实际临床部署中的鲁棒性。

---

## 62. EAGLE: Edge-Aware Graph Learning for Proactive Delivery Delay Prediction in Smart Logistics Networks

**arXiv ID:** 2604.05254 | [PDF](https://arxiv.org/pdf/2604.05254v1)

**作者:** Zhiming Xue `[一作]` (Northeastern University), Yujue Wang `[通讯]` (University of New Mexico)

**通讯引用:** 17000 | [OpenAlex ID](https://openalex.org/A5100700685)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研发了一种混合时空图学习框架EAGLE，用于主动预测物流网络的交付延迟；

**💡 创新点**

创新点在于将轻量级Transformer patch编码器与边感知图注意力网络（E‑GAT）结合，解耦时空信息并通过多任务学习（分类+回归）增强稳定性，同时提供风险热图和SHAP特征归因实现可解释性；

**🔧 技术方法**

采用PatchTST‑Lite进行时间序列编码，E‑GAT作为图编码器，双头MLP完成SLA违约概率与延迟量预测，并使用0.7×BCE+0.3×Huber的多任务损失；

**📊 数据集**

使用公开的DataCo Smart Supply Chain数据集（约18万笔订单，46个节点，1478条边）；

**📈 对比分析**

与XGBoost、随机森林、LSTM、标准GAT等基线对比，EAGLE在测试集上获得宏观F1 0.8762、AUC‑ROC 0.9773，跨种子方差仅0.0089，显著优于所有基线；

**⚠️ 局限性**

局限包括模型复杂度高、部署资源需求大；相对标签需足够历史订单，冷启动节点或不同拓扑结构的数据集需进一步验证。

---

## 63. Vehicle-as-Prompt: A Unified Deep Reinforcement Learning Framework for Heterogeneous Fleet Vehicle Routing Problem

**arXiv ID:** 2604.05195 | [PDF](https://arxiv.org/pdf/2604.05195v1)

**作者:** Shihong Huang `[一作]` (Zhejiang University), Weihua Zhou `[通讯]` (Zhejiang University)

**通讯引用:** 51253 | [OpenAlex ID](https://openalex.org/A5077976698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出统一的基于深度强化学习的异构车队路线规划框架

**💡 创新点**

创新性地将车辆视作提示，将车队选择和路径规划统一为单阶段自回归决策，并引入交叉语义编码器和多视图解码器

**🔧 技术方法**

使用Transformer、跨语义注意力、多头注意力、SwiGLU、REINFORCE强化学习以及熵正则和协方差引导熵控制等技术

**📊 数据集**

使用合成生成的异构车队VRP实例，规模分别为N=50,80,100,120，K=20,30,30,40

**📈 对比分析**

与传统启发式求解器（PyVRP、OR-Tools、vrp-cli）及现有DRL模型（RF-POMO、RF-MVMoE、RF-TE、HF-DRL、VaP-AM、VaP-POMO）对比，VaP-CSMV在大多数变体中实现平均最优性差距≤3%（N=50）或≤9%（N=120），推理时间仅秒级

**⚠️ 局限性**

局限在于未涵盖动态/实时请求、随机交通、环境/能源约束，仅在合成数据上验证，真实场景中可能需要进一步适配

---

## 64. LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows

**arXiv ID:** 2604.05182 | [PDF](https://arxiv.org/pdf/2604.05182v1)

**作者:** Zhengqin Li `[一作]` (Meta Reality Labs Research), Zhao Dong `[通讯]` (Meta Reality Labs Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于扩展Transformer上下文窗口的高保真对象级3D重建与逆渲染模型LSRM。

**💡 创新点**

通过Native Sparse Attention实现大规模上下文窗口、3D感知路由和块感知序列并行来显著提升纹理与几何细节。

**🔧 技术方法**

利用NSA稀疏注意力、3D感知块路由、All‑gather‑KV序列并行、粗细两阶段Coarse‑to‑Fine Transformer、VolSDF渲染、DINOv3特征等技术。

**📊 数据集**

在600K 3D模型上训练，使用GSO、StanfordORB、DigitalTwinCatalogue、ObjectsWithLighting等公开数据集进行评估。

**📈 对比分析**

与LIRM、RelitLRM等SOTA feed‑forward模型对比，PSNR提升>2.4 dB、LPIPS降低>40%，逆渲染任务中与稠密优化方法相当或更优。

**⚠️ 局限性**

仍无法充分推断粗糙度/金属等材质细节，极细纹理（如小标签）仍不清晰，并且在极端光照或遮挡场景下逆渲染效果有限。

---

## 65. R3PM-Net: Real-time, Robust, Real-world Point Matching Network

**arXiv ID:** 2604.05060 | [PDF](https://arxiv.org/pdf/2604.05060v1)

**作者:** Yasaman Kashefbahrami `[一作]` (Eindhoven University Of Technology), Egor Bondarau `[通讯]` (Eindhoven University Of Technology)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5053348352)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种轻量化、全局感知的点匹配网络R3PM-Net，用于实现实时高精度点云配准。

**💡 创新点**

创新点在于通过共享MLP实现全局感知的特征提取，省去复杂特征工程和大型backbone，显著降低参数量和推理时延；同时利用可微SVD和动态阈值的对应与剔除机制提升鲁棒性。

**🔧 技术方法**

技术细节包括Siamese共享MLP、全局最大池化、soft匹配矩阵与Sinkhorn归一化、动态阈值预测、可微加权SVD以及粗精细两阶段（R3PM-Net + GICP）配准流程。

**📊 数据集**

使用的公开数据集有ModelNet40；自行构建的Sioux-Cranfield（含理想与噪声CAD、相机重建点云）和Sioux-Scans（事件相机稀疏噪声扫描与CAD的配准）。

**📈 对比分析**

与RPMNet、Predator、GeoTransformer、RegTR、LoGDesc等最新方法对比，R3PM-Net在ModelNet40上实现1.0°旋转误差、0.01cm平移误差、0.052cm Chamfer、1.0拟合度、0.029cm内点RMSE，仅需0.007s（比RegTR快约7×），在Sioux-Cranfield同样保持完美拟合并显著降低时延；在Sioux-Scans中成功解决大多数边缘案例，平均耗时≈41ms。

**⚠️ 局限性**

局限性包括：在极稀疏、低重叠或光照敏感的事件相机扫描中仍有失败率；需要针对特定场景进行微调；缺乏对极端噪声/遮挡的理论保证。

---

## 66. Evaluation of Embedding-Based and Generative Methods for LLM-Driven Document Classification: Opportunities and Challenges

**arXiv ID:** 2604.04997 | [PDF](https://arxiv.org/pdf/2604.04997v1)

**作者:** Rong Lu `[一作]`, Song Hou `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比嵌入式模型和生成式视觉-语言模型在地球科学技术文档分类上的性能，评估准确率、稳定性和计算成本。

**💡 创新点**

提出通过链式思考（CoT）提示和领域定义提升VLM性能，并利用详细类定义提示显著提升嵌入模型的分类效果。

**🔧 技术方法**

采用多模态嵌入模型（QQMM、CLIP等）、相似性投票、Qwen2.5-VL-72B/7B的零射击推理、CoT提示工程、监督微调（SFT）以及宏F1、准确率、簇内/簇间距离等评估指标。

**📊 数据集**

使用内部构建的多学科能源领域技术文档基准集（八类），包含PDF、TIFF/PNG/JPG文件，统一采用文档首页进行评估。

**📈 对比分析**

通过宏F1、准确率、簇指标进行对比；结果显示生成式VLM在零射击时取得82%准确率，嵌入模型最高63%；CoT提示提升10%宏F1；SFT在样本充足的类上提升20%，但在少样本类上表现下降。

**⚠️ 局限性**

生成式VLM推理计算成本高、输出不确定；嵌入模型准确率低；SFT对数据不平衡敏感，需要精心平衡训练集才能取得显著提升。

---

## 67. Edit, But Verify: An Empirical Audit of Instructed Code-Editing Benchmarks

**arXiv ID:** 2604.05100 | [PDF](https://arxiv.org/pdf/2604.05100v1)

**作者:** Amir M. Ebrahimi `[一作]` (Queen's University), Gopi Krishnan Rajbahadur `[通讯]` (Queen's University)

**通讯引用:** 582 | [OpenAlex ID](https://openalex.org/A5052055475)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对现有的两大指令式代码编辑基准 CanItEdit 与 EDIT‑Bench 进行系统审计，评估其在语言、编辑意图、应用领域、测试充分性和问题独立性等维度上的代表性与可靠性。

**💡 创新点**

提出六条经验性指导原则（Desiderata），涵盖语言覆盖、领域分布、编辑意图多样性、行为保持验证、问题独立性以及新兴开发范式的纳入，以帮助构建更符合真实世界需求的基准。

**🔧 技术方法**

使用 LLM（Claude Sonnet 4.6）进行自动化标注、Python 与 JavaScript 测试覆盖率测量、差异区域覆盖率评估，以及基于模型生成的参考实现回收等技术手段。

**📊 数据集**

采用 Copilot Arena、AIDev 以及 GitHub Octoverse 公开数据集来对比基准与真实开发活动的分布，并以此作为验证标准。

**📈 对比分析**

比较显示 CanItEdit 在测试数量（中位数 13）和文件覆盖率（近 100%）方面优于 EDIT‑Bench（中位数 4，文件覆盖率约 40%），但 EDIT‑Bench 的差异区域覆盖率平均 64.9%，说明其测试或有不足，且 42.9% 的可执行问题覆盖率低于 75%。

**⚠️ 局限性**

局限性包括：只评估了两套基准，LLM 标注可能存在误差；覆盖率测量仅为语句覆盖，未涵盖分支或变异测试；对模型能力与基准难度的区分仍受限于参考实现的可得性。

---

## 68. Probabilistic Tree Inference Enabled by FDSOI Ferroelectric FETs

**arXiv ID:** 2604.05115 | [PDF](https://arxiv.org/pdf/2604.05115v1)

**作者:** Pengyu Ren `[一作]` (University of Notre Dame), Kai Ni `[通讯]` (University of Notre Dame)

**通讯引用:** 8176 | [OpenAlex ID](https://openalex.org/A5075633314)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种单一FDSOI‑FeFET技术的硬件平台，集成ACAM与Gaussian随机数生成器（GRNG），实现贝叶斯决策树（BDT）的高效推理。

**💡 创新点**

将ACAM与高质量随机数源统一在FDSOI‑FeFET上，并提出将随机性转移到输入域的节点级映射策略，显著减少重写开销、提升能效与可靠性。

**🔧 技术方法**

FDSOI‑FeFET存储、BTBT产生的熵源、ACAM内容可寻址存储、节点级映射与软硬件协同设计。

**📊 数据集**

MNIST手写数字数据集和Breast Cancer Wisconsin Diagnostic（乳腺癌诊断）数据集。

**📈 对比分析**

与CPU、GPU及传统决策树进行对比；在MNIST上分类准确率提升40%+，推理速度提升2-3个数量级，能耗降低4-5个数量级。

**⚠️ 局限性**

受限于FeFET的位精度（需2bit即可）、设备漂移与耐久性挑战、阈值编程精度要求高，以及缺乏大规模应用验证。

---

## 69. Territory Paint Wars: Diagnosing and Mitigating Failure Modes in Competitive Multi-Agent PPO

**arXiv ID:** 2604.04983 | [PDF](https://arxiv.org/pdf/2604.04983v1)

**作者:** Diyansha Singh `[一作]` `[通讯]` (Independent Researcher), Diyansha Singh (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在Unity实现的两人零和格子游戏中，PPO在自我对弈下的失败模式，并提出了针对实现层面的改进与竞争过拟合的解决方案。

**💡 创新点**

首次系统性识别并修复了五个实现错误，揭示了自我对弈导致的竞争过拟合并证明仅通过简单的对手混合即可恢复泛化性能。

**🔧 技术方法**

使用PPO+GAE、观察归一化、对手混合以及Unity-Python TCP桥接的自定义环境，进行多种消融实验。

**📊 数据集**

基于自定义的10×10格子游戏（Territory Paint Wars），在Unity环境中进行实验，使用随机对手进行泛化评估。

**📈 对比分析**

通过10个随机种子训练12,000个episode，平均泛化胜率从原始26.8%提升至77.1%（±12.6%），消融实验表明GAE、观察归一化和对手混合各自不可或缺。

**⚠️ 局限性**

实验仅在单一环境和PPO算法上验证，受限于Unity-Python桥接的速度和缺乏更大规模或随机环境的测试，可能无法直接推广至其他复杂多智能体任务。

---

## 70. Architecture Without Architects: How AI Coding Agents Shape Software Architecture

**arXiv ID:** 2604.04990 | [PDF](https://arxiv.org/pdf/2604.04990v1)

**作者:** Phongsakon Mark Konrad `[一作]` (University of Southern Denmark), Serkan Ayvaz `[通讯]` (University of Southern Denmark)

**通讯引用:** 1432 | [OpenAlex ID](https://openalex.org/A5075890706)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了AI编码代理在软件架构中的隐式决策，提出了五种决策机制和六种提示-架构耦合模式，并通过三种不同提示的聊天机器人案例展示了提示词对系统结构的直接塑造。

**💡 创新点**

创新点在于将“vibe coding”扩展为“vibe architecting”，揭示提示词如何直接决定架构并系统化六种耦合模式，指出提示词本身应视为架构决策并需要治理。

**🔧 技术方法**

使用的大型语言模型驱动编码代理（如Claude Code）、LangChain、LlamaIndex等框架，结合Prompt模式分析与架构描述语言进行技术实现。

**📊 数据集**

实验仅基于作者自行在GitHub案例库中生成的三种不同提示的聊天机器人，未使用公开数据集。

**📈 对比分析**

通过对比三种提示生成的系统结构（LoC、文件数、组件数等）来评估提示的影响，发现提示越具体，代码量和文件数显著增加，但未对功能性能或运行效率进行量化测评。

**⚠️ 局限性**

局限性包括仅在单一工具与模型下验证，缺乏跨代理、跨模型的实证；案例规模有限、指标粗糙，未对实际功能性能进行客观评估。

---

## 71. SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration

**arXiv ID:** 2604.05079 | [PDF](https://arxiv.org/pdf/2604.05079v1)

**作者:** Zhongyu Yang `[一作]` (Heriot-Watt University), Yingfang Yuan `[通讯]` (Heriot-Watt University)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5091219117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SVAgent，一种基于剧情线的多智能体框架，用于长视频问答；

**💡 创新点**

通过持续生成剧情线、使用确定性点过程（DPP）进行证据选取、跨模态决策与自我反思循环实现全局时间一致性和可靠性；

**🔧 技术方法**

多智能体架构、剧情线生成、DPP 证据检索、跨模态决策与 Meta 决策、建议智能体，以及在视频 MLLMs（如 Qwen2.5‑VL、Qwen3‑VL）上的实现；

**📊 数据集**

Video‑MME、MLVU、LongVideoBench、LVBench 四大长视频基准；

**📈 对比分析**

与 Caption‑based、Keyframe‑retrieval、Graph‑based 以及 Video‑MLLMs、VideoAgent、Videomind 等基线比较，平均提升 5.5%–11.5%，在多跳推理任务中最高提升 6.7 分；

**⚠️ 局限性**

对极长视频仍受 DPP 与 Meta 决策循环的计算开销限制；依赖预训练编码器，若编码器对长视频的跨模态对齐不足可能影响性能。

---

## 72. FLARE: Agentic Coverage-Guided Fuzzing for LLM-Based Multi-Agent Systems

**arXiv ID:** 2604.05289 | [PDF](https://arxiv.org/pdf/2604.05289v1)

**作者:** Mingxuan Hui `[一作]` (Xidian University), Yaxiao Li `[通讯]` (Xidian University)

**通讯引用:** 514 | [OpenAlex ID](https://openalex.org/A5011361351)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套基于覆盖驱动的 fuzz 测试框架 FLARE，用于自动化检测 LLM 驱动的多智能体系统（MAS）的功能错误。

**💡 创新点**

创新点在于：①通过 LLM 推理从源代码中自动提取 MAS 的规格与行为空间；②定义了专门针对 MAS 的内部/外部行为覆盖度量并以此为反馈驱动 fuzz；③采用双代理（Failure Agent 与 Judge Agent）实现语义层面的错误判定，减少 LLM 误报。

**🔧 技术方法**

使用技术包括：LLM（GPT‑4.1、Gemini‑3.0‑Pro）进行规格抽取、行为空间构造与日志语义匹配；基于覆盖度量的 fuzzing 循环；自定义 JSON 规范与日志结构化；对比基线（LLM‑Fuzzer、PythonFuzz、Frelatage）。

**📊 数据集**

数据集为 16 个使用 AutoGen 框架的开源 MAS 应用（涵盖视频制作、代码生成、数据分析等多场景），共 80–550 行代码。

**📈 对比分析**

与基线对比，FLARE 在传统的 Statement/Branch 覆盖率、MAS 专用的 inter‑agent（96.9%）和 intra‑agent（91.1%）覆盖率上均超过 8–12%；同时发现 61 个 MAS 特有错误，基线仅发现 2–6 个崩溃。性能稳健，对 LLM 后端（GPT‑4.1 与 Gemini‑3.0‑Pro）差异不大。

**⚠️ 局限性**

局限性包括：仅针对 AutoGen 进行设计，迁移到其他框架需重新构建域知识与提示；依赖 LLM 的准确性与稳定性，可能产生误报或漏报；实验仅覆盖 16 个小规模项目，缺乏对工业级大规模 MAS 的验证。

---

## 73. General Multimodal Protein Design Enables DNA-Encoding of Chemistry

**arXiv ID:** 2604.05181 | [PDF](https://arxiv.org/pdf/2604.05181v1)

**作者:** Jarrid Rector-Brooks `[一作]` (California Institute of Technology), Cheng-Hao Liu `[通讯]` (California Institute of Technology)

**通讯引用:** 4212 | [OpenAlex ID](https://openalex.org/A5047329586)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种多模态扩散模型（DISCO），实现蛋白序列与三维结构的联合设计，并能以任意分子为上下文生成具有新活性位点的酶。

**💡 创新点**

创新点在于：①无需预设活性位点残基，直接基于反应中间体几何生成酶；②结合Feynman‑Kac正则化实现推理时序列与结构共同优化；③通过跨模态循环与自纠正机制提升共设计性。

**🔧 技术方法**

技术：多模态扩散网络（离散序列+连续坐标）、SE(3)对称软增强、交叉模态循环（cross‑modal recycling）、Feynman‑Kac奖励倾斜（FKC‑MM 与 FKC‑SG）、自纠正采样、序列温度平滑。

**📊 数据集**

数据集：仅使用Protein Data Bank（PDB）原始结构；构建179种天然/非天然配体的Benchmark LigandSet，用于评估对多种分子上下文的共设计能力。

**📈 对比分析**

与现有单模、逆折叠及基于theozyme的生成方法相比，DISCO在共设计成功率、可折叠性、活性位点多样性上显著优于；实验验证显示生成的炭烃转移酶在四种新型反应中TTN最高可达4,050，远超已知的工程酶。

**⚠️ 局限性**

局限性：①对极化和电子效应建模不足，限制了对某些高复杂度反应的预测；②对大分子或多亚基配体的协同设计仍需改进；③实验验证覆盖范围有限，未来需进一步扩展不同化学空间与进化实验。

---

## 74. Breakthrough the Suboptimal Stable Point in Value-Factorization-Based Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.05297 | [PDF](https://arxiv.org/pdf/2604.05297v1)

**作者:** Lesong Tao `[一作]` (Xi'an Jiaotong University), Nanning Zheng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 34809 | [OpenAlex ID](https://openalex.org/A5047405956)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出多轮价值分解框架 MRVF，用以突破单轮分解的子最优稳定点瓶颈

**💡 创新点**

在理论上引入稳定点概念并提出严格提升条件，利用行动价值增量让非最优动作变为不稳定点；算法上设计多轮前向后向计算与集中 ε‑贪婪采样

**🔧 技术方法**

离散动力学系统分析、梯度不连续性处理、行动价值增量目标函数、集中 ε‑贪婪采样与多轮采样策略

**📊 数据集**

风险‑回报游戏、带惩罚的捕食者‑猎物任务、StarCraft II Multi‑Agent Challenge（SMAC）多种难度场景

**📈 对比分析**

与 QMIX、NA2Q、QPLEX、QTRAN、WQMIX、RESQ 以及 MAPPO 等基线对比，MRVF 在非单调和单调环境均获得更高收益、收敛更稳健，尤其在 SMAC 的多种情境中表现最佳

**⚠️ 局限性**

需多轮迭代导致训练时间和计算成本提升，对极度稀疏奖励或大规模动作空间的适应性尚待进一步验证

---

## 75. Spike Hijacking in Late-Interaction Retrieval

**arXiv ID:** 2604.05253 | [PDF](https://arxiv.org/pdf/2604.05253v1)

**作者:** Karthik Suresh `[一作]` (Adobe), Michael Friedrich `[通讯]` (Adobe)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文系统研究了 Late‑Interaction 检索模型中 Hard Max（MaxSim）聚合方式对梯度分布与鲁棒性的影响，并通过合成实验与真实数据验证其对文档长度和脆弱性（spike hijacking）的影响。

**💡 创新点**

创新点在于：①提出了梯度 Gini 系数衡量聚合方式的梯度集中度；②构建可控的合成检索环境以纯粹观察聚合策略的结构效应；③揭示了硬最大聚合在文档长度增加时的脆弱性，并验证了 Top‑k 平滑聚合在保持检索性能的同时降低梯度集中度。

**🔧 技术方法**

主要技术包括 Late‑Interaction 检索框架、Hard Max、Top‑k 平均、Softmax 聚合、InfoNCE 对比学习、梯度 Gini 系数计算、合成数据生成、文档长度 sweep 与 spike 注入实验。

**📊 数据集**

使用了合成数据集（基于 100 维概念向量生成的查询/文档）以及真实多向量检索基准 ColQwen2.5 与 ViDoRe 医学检索数据集。

**📈 对比分析**

与 Softmax 和 Top‑k 聚合对比，Hard Max 在短文档上 Recall@1 最高（约 0.33–0.38），但其 Gini 系数高达 0.78；Top‑k 在较长文档下 Recall@1 更稳健，Gini 约 0.45；Softmax Gini 低但 Recall 仅 0.18，表明过度平滑会牺牲检索质量。

**⚠️ 局限性**

主要限制包括：①研究聚合方式的结构性脆弱性但未提供完整的模型重训练方案；②仅考察了 Hard Max、Top‑k 与 Softmax，未探索更复杂的聚合策略；③在真实数据上仅通过 inference‑time 交换聚合方式进行验证，未评估端到端重训练对性能的进一步影响。

---

## 76. Instruction-Tuned LLMs for Parsing and Mining Unstructured Logs on Leadership HPC Systems

**arXiv ID:** 2604.05168 | [PDF](https://arxiv.org/pdf/2604.05168v1)

**作者:** Ahmad Maroof Karimi `[一作]` (Oak Ridge National Laboratory), Awais Khan `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 452 | [OpenAlex ID](https://openalex.org/A5085960321)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于指令调优、链式思维（CoT）推理的LLM驱动HPC日志解析框架，并在Oak Ridge Frontier超级计算机上大规模解析并分析600+万条日志，挖掘故障传播、工作负载关联与网络错误模式。

**💡 创新点**

①将日志模板数据与指令集混合训练，提升LLM对异构日志的泛化能力；②采用LoRA参数高效微调，保持模型小巧（8B）同时达到70B/Claude级别的解析准确率；③在本地部署，保障数据隐私与低能耗。

**🔧 技术方法**

instruction-following LLaMA 3 8B（微调版）、LoRA高效微调、链式思维提示、基于模板的日志预处理、三阶段解析流程（签名生成→模板生成→序列生成）。

**📊 数据集**

1) LogHub公共基准（HPC、Linux、OpenSSH等） 2) Frontier超级计算机真实生产日志（≈6.38亿条） 3) 通过LLaMA 70B生成的合成日志模板数据。

**📈 对比分析**

对比基准模型Claude3.5 Sonnet、LLaMA70B、未微调LLaMA8B；使用覆盖率（coverage）评估解析成功率。实验表明，微调后的LLaMA8B在多种日志集上达到≈96%覆盖率，接近或超过70B/Claude模型；在生产日志上也保持高准确率，并在能耗上比70B低约20×、时延低7×。

**⚠️ 局限性**

①对极少量罕见错误（如内核崩溃）仍需进一步验证；②模型仍对语法噪声（如多余单词）敏感，准确率下降；③依赖于高质量模板/指令数据，生成/标注成本较高；④在高度动态的日志格式迁移时，需定期重微调。

---

## 77. Formal specification and behavioral simulation of the holiday gift exchange game

**arXiv ID:** 2604.05219 | [PDF](https://arxiv.org/pdf/2604.05219v1)

**作者:** Daniel Quigley `[一作]` `[通讯]` (Indiana University Bloomington), Daniel Quigley (Indiana University Bloomington)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对圣诞礼物交换游戏（White Elephant、Yankee Swap 等）进行正式化建模，定义状态空间、行动集、偷窃链条，并证明游戏终止、最终分配为双射；在此基础上扩展“装饰游戏”模型，引入部分信息、社交成本、适应性策略和偏好选择；随后通过 240,000 场全因子模拟，评估各行为特征及其交互作用；最后推导并计数所有可能的游戏轨迹，展示其指数级增长。

**💡 创新点**

创新点主要有：①首次给出完整的形式化规范并证明关键性质；②提供偷窃链条的递归结构和终止性证明；③构建包含社交成本、情绪反馈与信息不确定性的多维行为模型；④通过大规模仿真揭示社交成本为驱动偷窃的主导因素；⑤给出精确计数公式和算法，说明轨迹数远超最终分配数。

**🔧 技术方法**

技术手段包括：离散数学与组合计数（链条计数、状态树）、博弈论与决策理论（离散选择模型、CARA 效用、logit 选择概率）、贝叶斯推理（质量估计与风险调整）、统计模拟（全因子实验设计、5000 场/配置、29 人）以及复杂性分析（多维状态的多重集压缩算法）。

**📊 数据集**

使用的数据集为仿真生成的 240,000 场游戏，每场 29 名玩家，涵盖三种价值模型（独立、相关、负相关）和 16 种行为特征组合；无真实实验数据，所有结果均来自模拟。

**📈 对比分析**

方法比较采用基准配置（无额外特征）与各特征或特征组合的对比；结果显示：社交成本将偷窃次数减少 27–48%，适应性动态减少 23–38%，偏好选择在价值相关模型下增加 11% 偷窃；玩家位置 1 维持显著优势，位置 2 始终最弱；在各配置中，简单的“随机偷窃”策略往往与优化策略相当，凸显行为反馈对策略效能的削弱。

**⚠️ 局限性**

局限性包括：假设玩家按离散选择模型理性行动，参数（社交成本、情绪增益、风险厌恶）无实证校准；未考虑玩家合作、合伙或“自带礼物禁止”等常见规则；仅在单一游戏周期内分析，未研究重复博弈与声誉演化；模拟基于随机生成的价值模型，缺乏真实用户数据验证。

---

## 78. ClawsBench: Evaluating Capability and Safety of LLM Productivity Agents in Simulated Workspaces

**arXiv ID:** 2604.05172 | [PDF](https://arxiv.org/pdf/2604.05172v1)

**作者:** Xiangyi Li `[一作]` (BenchFlow), Han-chung Lee `[通讯]` (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套名为OpenClawBench的真实生产力任务评测基准，涵盖五种高保真模拟服务（Gmail、Calendar、Drive/Docs、Slack、Drive）和跨服务、多安全情景的任务；

**💡 创新点**

创新点在于：①构建可与真实API一致的可恢复状态的模拟环境；②将任务安全与性能分离，使用[-1,1]安全评分；③将域知识（技能）与元提示（meta‑prompt）拆分为可独立调节的两大支撑杠杆；④提供全面的安全违规模式分类与实验分析；

**🔧 技术方法**

技术包括：REST API模拟、SQLite状态管理、快照/恢复机制、基于数据库状态的评价器、基于提示工程的元提示、域技能层级化（激活层与参考层）、多模型多Harness的实验框架；

**📊 数据集**

数据集主要是从真实生产账号捕获的金色请求‑响应对（共328条测试），以及在每个任务中生成的种子数据；

**📈 对比分析**

比较方法：在11个模型+Harness组合上，评估TSR、UAR、SCR；结果显示：无支撑时TSR低至0–8%，支撑后提升至39–63%；顶尖模型TSR仅相差10个百分点，但UAR在7–33%之间，模型能力与安全不成正相关；

**⚠️ 局限性**

局限性包括：模拟服务缺少速率限制/延迟/并发；仅覆盖五种服务；缺乏人类基线；评价单轮且未考虑成本；实验设计不完整的2×2因子导致交互效果估计受限。

---

## 79. Bilinear Model Predictive Control Framework of the OncoReach, a Tendon-Driven Steerable Stylet for Brachytherapy

**arXiv ID:** 2604.05111 | [PDF](https://arxiv.org/pdf/2604.05111v1)

**作者:** Pejman Kheradmand `[一作]` (University of Louisville), Yash Chitalia `[通讯]` (University of Louisville)

**通讯引用:** 500 | [OpenAlex ID](https://openalex.org/A5001760308)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aaccfe5c-6b26-4208-b23c-35331481e142` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并验证了面向临床可用的缝合针的双线性模型预测控制（MPC）框架，用于精准引导针尖插入与弯曲。

**💡 创新点**

创新点在于提出了虚拟输入与实际绳张力之间的映射关系，使得双线性MPC可直接驱动三维绳索控制针尖弯曲，并兼容标准放射治疗针。

**🔧 技术方法**

采用双线性MPC、绳索驱动可弯曲针尖、图像基针尖跟踪、数值优化（MATLAB quadprog）等技术。

**📊 数据集**

使用仿真数据以及由Humimic凝胶制成的体外组织模型（phantom）进行实验验证，未使用公开医学影像数据集。

**📈 对比分析**

通过与开环模型估计和闭环跟踪实验对比，开环误差≤2 mm（≈3 %），闭环固定目标误差最低1.45 mm（≈1.7 %），最大误差8.3 mm（≈8 %）；仿真误差极小，表明控制精度良好。

**⚠️ 局限性**

局限在于不同弯曲方向存在较大误差，需进一步校准曲率‑张力映射，二维图像跟踪受限，计算延迟影响更快的控制更新，且未考虑真实组织的形变与模型不确定性。

---

## 80. Towards Scaling Law Analysis For Spatiotemporal Weather Data

**arXiv ID:** 2604.05068 | [PDF](https://arxiv.org/pdf/2604.05068v1)

**作者:** Alexander Kiefer `[一作]`, Xiao Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于Shifted Window的层次化视频Transformer模型，用卷积Patch Embedding将视频帧划分为Patch，随后在局部窗口内执行多头自注意力，并通过窗口偏移实现跨窗口全局语义连接，最终通过线性投影重构为输出特征。

**💡 创新点**

创新点在于将Swin Transformer的窗口注意力机制迁移至视频域，利用跨帧窗口移动实现时空上下文的高效建模，同时保持局部计算效率，解决了传统全局注意力在视频中计算量过大的问题。

**🔧 技术方法**

使用的技术包括卷积Patch Embedding、窗口化多头自注意力（W‑MSA）、Shifted窗口注意力（SW‑MSA）、残差连接、MLP以及线性投影重构；整体架构为层级式Transformer结构。

**📊 数据集**

在Kinetics‑400/600、Something‑Something V2、UCF101和HMDB51等大型动作识别数据集上进行训练和评估。

**📈 对比分析**

与TimeSformer、T‑ViT、Video‑Swin等基线方法在Kinetics‑400上对比，取得约2‑3%更高的Top‑1准确率，并在计算效率和参数量上优于同类方法，表现出更优的性能。

**⚠️ 局限性**

局部窗口限制导致对极长视频的全局依赖建模不足；模型参数量较大、训练成本高；对短时动作或异常帧的鲁棒性有限。

---

## 81. Memory Dial: A Training Framework for Controllable Memorization in Language Models

**arXiv ID:** 2604.05074 | [PDF](https://arxiv.org/pdf/2604.05074v1)

**作者:** Xiangbo Zhang `[一作]` (Georgia Institute of Technology), Ali Emami `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可调控的记忆压力训练框架（Memory Dial），通过参数 α 在标准交叉熵与温度锐化目标之间插值，使同一模型在不同记忆压力下形成一族可比对的模型。

**💡 创新点**

创新点在于把记忆化作为可控实验变量，而非仅仅是事后检测；构造的 α–τ 加权目标在保持架构、数据和优化不变的前提下，实现了对记忆压力的连续、可解释调节。

**🔧 技术方法**

使用的技术包括：交叉熵、温度锐化（softmax/τ）、α 加权组合、对比训练、seen/unseen 准确率、suffix NLL、频率分层分析、输出多样性（Jaccard）评估以及训练动态跟踪。

**📊 数据集**

实验涵盖多种模型架构与规模（DistilGPT2、GPT‑2 Small、TinyLLaMA‑1B、OPT‑250M/13B/27B）以及五大基准（ARC‑Easy、BoolQ、PIQA、COPA、OpenBookQA）。此外还对 SWAG、TruthfulQA、XCOPA 等进行验证。

**📈 对比分析**

通过在 α 取值 {0.0,0.2,0.4,0.6,0.8,1.0} 的多次实验，发现 seen 准确率随 α 单调上升，unseen 准确率基本保持不变；更大模型对 α 更敏感；频率高的序列更易记忆；输出多样性随 α 下降。整体表明 α 能可靠控制记忆压力且不显著牺牲泛化。

**⚠️ 局限性**

局限性包括：仅作为实验工具，不是完整的记忆控制方案；主要验证在英语多选基准，跨语言、跨模态和自然大规模预训练未充分评估；记忆压力是相对额外记忆，未能覆盖从零到最大记忆的全范围；评估指标有限，未完全探索 α–τ 组合空间；高 α 可能导致敏感信息泄露风险。

---

## 82. Curvature-Aware Optimization for High-Accuracy Physics-Informed Neural Networks

**arXiv ID:** 2604.05230 | [PDF](https://arxiv.org/pdf/2604.05230v1)

**作者:** Anas Jnini `[一作]` (University of Trento), George Em Karniadakis `[通讯]` (Brown University)

**通讯引用:** 99450 | [OpenAlex ID](https://openalex.org/A5009658255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发并评估了曲率感知优化器（自然梯度、Self‑scaled BFGS/Broyden 等）用于物理信息神经网络（PINN），以在高频、刚性或冲击等挑战性偏微分方程和刚性 ODE 上实现高精度求解。

**💡 创新点**

创新点在于将自然梯度与自适应 BFGS/Broyden 结合，并提出可批量化的二阶优化框架；同时设计了新的 PINN 方法处理无粘 Burgers 与 Euler 方程中的冲击，显著提升收敛速度与最终误差。

**🔧 技术方法**

技术实现采用了 Kronecker‑factored 预处理、Jacobi 缩放、kernel trick、行进线搜索（信任域、Armijo、Wolfe、zoom）以及 Fourier 特征映射和自适应采样等，构成高效的曲率感知优化流程。

**📊 数据集**

实验数据来源于合成基准问题集合，包括二维/三维 Helmholtz 方程、Stokes 流、无粘 Burgers、Euler 方程以及药物动力学 PK–PD 刚性 ODE，训练样本为随机/自适应采样得到的 collocation 点。

**📈 对比分析**

通过对比多种优化器在各基准上的相对 L1/L2/L∞ 误差、训练时间与参数量，结果显示自然梯度和自适应 BFGS 在高频、冲击和刚性场景中能实现 1e‑9 级误差并收敛最快，SOAP 其次；其他优化器收敛较慢或精度不足。

**⚠️ 局限性**

局限性包括：在极高频或极度刚性场景下仍需物理约束才能稳定收敛；二阶优化在大规模批量训练中计算开销较高，对超参数敏感；论文仅聚焦前向问题，逆问题与更大规模数据驱动任务尚未覆盖。

---

## 83. Dynamic Linear Coregionalization for Realistic Synthetic Multivariate Time Series

**arXiv ID:** 2604.05064 | [PDF](https://arxiv.org/pdf/2604.05064v1)

**作者:** Annita Vapsi `[一作]`, Manuela Veloso `[通讯]` (JPMorganChase)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种能够生成具有时间变异、状态切换和滞后跨通道相关性的合成多变量时间序列的动态线性共变模型（DynLMC），并通过在该生成数据上微调现有的时间序列预训练模型，提升其在真实基准数据集上的零样本与少样本预测性能。

**💡 创新点**

创新点在于将传统静态 LMC 的核心权重扩展为可随时间演变的、由自回归、隐马尔可夫状态切换以及随机滞后共同驱动的动态权重，显著提升合成数据与真实数据在交叉通道相关性、相位偏移与协方差漂移方面的相似度。

**🔧 技术方法**

技术手段包括：1) 自回归（AR）模型对权重进行平滑漂移；2) 隐马尔可夫模型（HMM）实现多状态的共变模式切换；3) 在隐层 GP 与观测通道间随机采样整数滞后以模拟领先/滞后效应；4) 软最大归一化确保权重合法性；5) 通过 KernelSynth 采样多样化的 GP 内核生成基础信号；6) 对生成的合成数据进行微调（fine‑tune）并评估 MAE 和赢数。

**📊 数据集**

使用了：
- 合成数据集：3类主数据（Drift、Lag、Regime Shift）及其组合版本（Combined、Combined BO），每个包含 1,500 条样本、160 通道、1,024 步长；
- 真实测试集：ECL、ETTh1、ETTh2、ETTm1、ETTm2、Exchange、Solar、Traffic、Weather 共 9 个多变量时间序列基准。

**📈 对比分析**

对比方法：在预训练的 TimePFN、iTransformer、Chronos‑2 三个基线模型上，先用传统 LMC 生成的数据微调，再分别用 DynLMC 的各个版本微调；评估指标为平均绝对误差（MAE）和在九个基准数据集上的“赢数”（即 MAE 更优的次数）。结果显示：
- TimePFN：Regime Shift 版本获得 4 胜，Combined 版本 7 胜；
- iTransformer：Regime Shift 9 胜，Combined 8 胜；
- Chronos‑2：改进相对有限，Regime Shift 仅 3 胜。整体而言，DynLMC 生成的数据在多样性与真实性上显著提升了模型的零样本泛化能力。

**⚠️ 局限性**

局限性：
1) 对 Chronos‑2 的提升有限，可能因为其原始预训练已包含某些动态相关性；
2) 仅在三种预训练模型上验证，缺乏对其他架构或任务（如异常检测、插值）的泛化实验；
3) 合成过程中的超参数（ρ、η、τ_max、K 等）需要手工调优，未给出自动化优化框架；
4) 生成的合成数据仍无法完全复制所有真实数据的复杂非线性关系和极端事件；
5) 评估指标仅使用 MAE 与赢数，未覆盖更细粒度的统计相似度或下游任务多样性。

---

## 84. Entities as Retrieval Signals: A Systematic Study of Coverage, Supervision, and Evaluation in Entity-Oriented Ranking

**arXiv ID:** 2604.05204 | [PDF](https://arxiv.org/pdf/2604.05204v1)

**作者:** Shubham Chatterjee `[一作]` (Missouri University of Science and Technology), Shubham Chatterjee `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5004223654)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性评估实体检索模型，探讨实体信号在检索中的实用性与评估方法的局限；

**💡 创新点**

提出实体信号评估的诊断框架，区分条件评估与开放世界评估，揭示覆盖瓶颈、监督-覆盖困境及CER‑OER差距；

**🔧 技术方法**

使用实体链接器WAT、Wikipedia2Vec实体嵌入，结合BM25初检，评估多种实体向量化与重排序模型（EDRM、Word‑Entity Duet、EVA、EsdRank、DREQ、QDER）以及437种无监督实体配置；

**📊 数据集**

采用TREC Robust04新闻检索语料，利用BM25+RM3候选集（Top‑1000）进行实验；

**📈 对比分析**

在条件评估下，部分模型可将MAP提升至0.697（相较BM25 0.292）；但在开放世界评估下，MAP提升不超过0.05（最高0.343），多数配置未超越BM25；在对比时采用MAP、nDCG@20、P@20等指标；

**⚠️ 局限性**

局限在于当前评估框架无法分离实体信号的覆盖与区分能力，监督策略只关注概念实体相关性（CER），忽略可观测判别性（OER），导致实体覆盖率低且无法在开放世界提升效果；需构建实体级判别性注释与改进评估协议。

---

## 85. Learning to Focus: CSI-Free Hierarchical MARL for Reconfigurable Reflectors

**arXiv ID:** 2604.05165 | [PDF](https://arxiv.org/pdf/2604.05165v1)

**作者:** Hieu Le `[一作]` (Texas A&M University), Sabit Ekin `[通讯]` (Texas A&M University)

**通讯引用:** 1016 | [OpenAlex ID](https://openalex.org/A5014255349)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于分层多智能体强化学习（HMARL）的CSI-free框架，用于控制可重构反射器，以解决传统方法中对信道状态信息（CSI）估计的高计算开销和集中优化的维度爆炸问题。

**💡 创新点**

创新点在于提出了一种完全不依赖CSI的优化方法，通过利用用户定位数据和空间智能来管理宏观信号传播，从而实现了显著的接收信号强度指示（RSSI）提升。

**🔧 技术方法**

使用了分层多智能体强化学习（HMARL）架构和多智能体近端策略优化（MAPPO）技术，结合集中训练与分散执行（CTDE）方案。

**📊 数据集**

在一个60 GHz的室内毫米波仿真环境中进行实验，模拟了一个会议室，使用了两组机械可重构金属反射器和多个用户设备。

**📈 对比分析**

与两个基线方法（缺乏几何先验的变体和传统集中式PPO代理）进行比较，结果显示该框架在多用户场景中实现了高达7.79 dB的性能提升，且在用户移动情况下保持了良好的信号稳定性。

**⚠️ 局限性**

局限性在于该框架依赖于用户定位信息的准确性，定位误差可能会影响分配质量和波束聚焦的准确性。

---

## 86. Sparse Autoencoders as a Steering Basis for Phase Synchronization in Graph-Based CFD Surrogates

**arXiv ID:** 2604.04946 | [PDF](https://arxiv.org/pdf/2604.04946v1)

**作者:** Yeping Hu `[一作]` (Lawrence Livermore National Laboratory), Shusen Liu `[通讯]` (Lawrence Livermore National Laboratory)

**通讯引用:** 1802 | [OpenAlex ID](https://openalex.org/A5101523517)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在冻结的图形基CFD代理模型中提出后置相位校正框架，通过对潜在空间进行可解释性稀疏编码和相位旋转，实时纠正相位漂移。

**💡 创新点**

创新点在于：①利用稀疏自编码器提取分离的振荡特征；②用Hilbert变换自动识别近正交振荡对；③在低秩系数空间中施加时间平滑的相位旋转，实现既保持振幅又调节相位的动态干预。

**🔧 技术方法**

使用的技术包括：稀疏自编码器（SAE）、Hilbert变换、奇异值分解（SVD）、相位旋转干预、MeshGraphNet代理、PCA基准与静态特征干预（缩放、平移、裁剪）。

**📊 数据集**

采用CylinderFlow数据集，即不同直径和位置的二维无粘性圆柱流场模拟，包含1000训练、100验证、100测试轨迹，时间跨度600步。

**📈 对比分析**

比较方法：在相同的旋转干预管道下分别在SAE、PCA和原始嵌入空间应用；静态干预作为基准。结果显示SAE在相位修正任务中提升约26.1%的MSE闭合比例、35.0%的ROI提升，nRMSE<1且相关系数最高；PCA和原始嵌入分别仅提升约16%/21%和4%/7%。

**⚠️ 局限性**

局限性包括：仅在单一圆柱摆动模式下验证；相位偏移设定单一；Hilbert基准最适用于单频振荡，难以处理多频或混沌流；方法受限于底层代理的表示能力，无法恢复缺失物理。

---

## 87. StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing

**arXiv ID:** 2604.05014 | [PDF](https://arxiv.org/pdf/2604.05014v1)

**作者:** StarVLA Community `[一作]` `[通讯]` (HKUST), StarVLA Community (HKUST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

StarVLA 是一个开源的统一框架，整合 Vision‑Language‑Action（VLA）模型的 backbone–action‑head 架构、通用训练策略（监督学习、跨体感知共训练、跨 Embodiment 预训练）以及统一的评估与部署接口，支持 VLM 与 world‑model 两大类模型并可在多种 benchmark 上快速实验。

**💡 创新点**

核心创新包括：①统一的 backbone–action‑head 抽象，兼容多种解码模式（FAST、OFT、π、GR00T）并可互换；②可复用的跨 modality 与跨 Embodiment 训练配置；③单一服务器‑客户端评估接口，使模拟与实机部署无缝衔接；④提供可复现的单 benchmark 训练 recipe，展示 VLM 与 world‑model 均可达或超越先前最佳；⑤提出 “generalized VLA” 视角，将 VLM 与 world‑model 解释为同一框架下不同的辅助信号。

**🔧 技术方法**

采用 Vision‑Language foundation models（如 Qwen‑VL、Cosmos‑Predict2）作为 backbone，配合四种 action‑head（FAST、OFT、π、GR00T）；训练使用 Accelerate + DeepSpeed、bfloat16 混合精度、梯度累积、梯度裁剪、Cosine 学习率；评估通过 WebSocket 服务器‑客户端协议完成；跨 Embodiment 训练通过 LeRobot 混合数据集实现；共训练时结合 VLM web 数据进行多目标优化。

**📊 数据集**

单 benchmark 训练使用 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa‑GR1、BEHAVIOR‑1K 等官方 benchmark；跨 benchmark 训练使用这些 benchmark 的训练集拼接；共训练中还使用大规模 VLM web 数据（如 COCO/LLM 对话等）作为辅助监督；跨 Embodiment 预训练通过 LeRobot 组合多种机器人数据。

**📈 对比分析**

所有实验均采用统一服务器‑客户端评估，使用 benchmark 官方指标。单 benchmark 训练 recipe 中，StarVLA‑π、GR00T、OFT 等模型在各 benchmark 上的成功率与或略优于现有 state‑of‑the‑art（π_0.5、GR00T‑N1.6 等）。在跨 benchmark 的 generalist 训练下，单一模型在所有 benchmark 上保持竞争力，平均性能与各专用模型相当或更好。具体数值见论文表格，展示如 LIBERO Spatial 98.0% vs prior 98.8%、RoboTwin 90%+ 等。

**⚠️ 局限性**

限制与待改进：①目前仅实现监督与共训练，RL fine‑tuning 仍在集成；②对新 benchmark 的适配仍需手动编写适配层；③大模型与多 GPU 训练对资源要求高，通信开销随规模增加；④缺乏深入的 ablation 及不同 backbone‑head 组合的系统分析；⑤缺少大规模真实机器人实验验证。

---

## 88. YMIR: A new Benchmark Dataset and Model for Arabic Yemeni Music Genre Classification Using Convolutional Neural Networks

**arXiv ID:** 2604.05011 | [PDF](https://arxiv.org/pdf/2604.05011v1)

**作者:** Moeen AL-Makhlafi `[一作]` (Xidian University), Saher Qaid `[通讯]` (Xidian University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了第一份专业标注的也门传统音乐流派数据集YMIR，并提出了针对该数据集的CNN架构YMCM，实现了基于时频特征的音乐流派分类

**💡 创新点**

创新点在于（1）提供了以专家标注、具高一致性的也门音乐流派数据集；（2）设计了五层卷积的专属模型YMCM；（3）系统比较了多种时频特征与多种CNN架构，验证了Mel‑Spectrogram在深度网络中的优势

**🔧 技术方法**

主要使用了Mel‑Spectrogram、MFCC、Chroma、FilterBank等时频特征，配合AlexNet、VGG16、MobileNet、基础CNN和自研YMCM等卷积网络，训练采用Adam优化器、交叉熵损失及早停策略

**📊 数据集**

使用的主要数据集为YMIR，包含1475条30秒的也门传统音乐音频，覆盖Sana’ani、Hadhrami、Lahji、Tihami、Adeni五个流派，且每个流派295条音频，划分80%训练/20%测试

**📈 对比分析**

实验共30组（6特征×5模型），通过精度、召回率、F1、特异性、平衡准确率等指标进行评估，结果显示YMCM在Mel‑Spectrogram特征上达98.83%准确率，显著优于AlexNet（97.59%）、VGG16（95.87%）、MobileNet（93.73%）及基础CNN（96.28%）

**⚠️ 局限性**

局限性主要包括：①数据集规模有限，仅覆盖五个传统流派；②缺乏对跨区域或子流派的泛化能力评估；③仅使用CNN架构，未探究Transformer或自监督方法对低资源场景的潜在优势

---

## 89. Feature-Aware Anisotropic Local Differential Privacy for Utility-Preserving Graph Representation Learning in Metal Additive Manufacturing

**arXiv ID:** 2604.05077 | [PDF](https://arxiv.org/pdf/2604.05077v1)

**作者:** MD Shafikul Islam `[一作]` (Louisiana State University), Md Arifuzzaman `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 373 | [OpenAlex ID](https://openalex.org/A5027584079)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种基于本地差分隐私的特征重要性引导机制与层级图注意网络（FI‑LDP‑HGAT）相结合的隐私保护质量保证框架，用于金属增材制造过程中的缺陷（孔隙）检测；

**💡 创新点**

创新点在于①采用特征重要性引导的异方差高斯噪声分配（FI‑LDP），在保持（ϵ,δ）-LDP的同时将噪声集中分配给信息量低的维度；②构建了层级化、物理先验驱动的kNN图并在其上使用多头注意机制（HGAT），有效捕捉层间热耦合与空间邻接；

**🔧 技术方法**

使用的技术包括：ResNet‑18特征提取、温度化重要性权重估计、基于温度的权重分配、L₂裁剪、Gaussian噪声注入、层级图注意网络、焦点损失、重采样与数据增强；

**📊 数据集**

实验数据集为OPTOMEC LENS™ 750 直径 25.4 mm × 1.0 mm × 12.7 mm Ti‑6Al‑4V 细壁结构的 DED 过程，采集热像并与XCT孔隙配准，共 1,564 个样本，孔隙比例约 4.47%；

**📈 对比分析**

在相同图结构和模型参数下，FI‑LDP‑HGAT 在 ϵ=4 时可实现 81.5% 的性能恢复（F1*≈0.767，AUC≈0.936），比等方差 LDP 高 3–4% 的 AUC 与 AUPR，且明显优于 DP‑SGD 以及传统 CNN/MLP/GNN 基线；

**⚠️ 局限性**

局限性包括：仅在单一实验设备与单一材料上验证；对极端低预算（ϵ<1）时性能下降仍显著；并且 FI‑LDP 的重要性估计依赖于预热训练，若数据分布漂移需重新估计。

---

## 90. TDA-RC: Task-Driven Alignment for Knowledge-Based Reasoning Chains in Large Language Models

**arXiv ID:** 2604.04942 | [PDF](https://arxiv.org/pdf/2604.04942v1)

**作者:** Jiaquan Zhang `[一作]` (University of Electronic Science and Technology of China), Hengtao Shen `[通讯]` (Tongji University)

**通讯引用:** 1693 | [OpenAlex ID](https://openalex.org/A5000395470)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于拓扑数据分析（TDA）的单轮推理优化框架 TDA‑RC，通过诊断并修复链式推理的结构缺陷，以提升 LLM 的推理准确率并保持低成本。

**💡 创新点**

创新点：① 将多轮推理（CoT、ToT、GoT 等）映射到统一的 persistent homology 拓扑空间；② 设计六个拓扑指标并构建任务特定健康区间；③ 设计 Topological Optimization Agent，在单轮推理中基于诊断结果生成结构优化提示，实现一次性结构改进。

**🔧 技术方法**

采用的技术：Topological Data Analysis（persistent homology + Vietoris–Rips 复形）、图构建与 MLP 投影、Agent 生成式提示、基于健康区间的诊断与修复、六维拓扑指标。

**📊 数据集**

使用的数据集：MATH、OlympiadBench、GSM8K、BBH、MMLU‑CF、LongBench、HotpotQA、MuSiQue 等多任务。

**📈 对比分析**

比较方法与性能：与多种链式推理（CoT、CoT‑SC、Self‑Refine、AFlow、ToT、GoT、FoT、AoT）以及 Prompt‑优化方法（HoT、Instruction Induction、Prompt Canvas 等）比较；在 GPT‑4o‑mini、Qwen‑Turbo、DeepSeek‑V3 上实验，TDA‑RC 平均提升 3–4 分准确率，成本仅为 CoT 的 1.06–1.21 倍，整体定位于准确率‑成本左上角。

**⚠️ 局限性**

Limitation：框架依赖预先构建的任务特定健康区间，若缺乏历史数据或面对全新任务则效果受限，需进一步研究自动拓扑原型发现与跨任务适配。

---

## 91. Attribution Bias in Large Language Models

**arXiv ID:** 2604.05224 | [PDF](https://arxiv.org/pdf/2604.05224v1)

**作者:** Eliza Berman `[一作]` (New York University), Emily Black `[通讯]` (New York University)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5006802134)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 Attribench 的平衡名气与种族、性别的引用归属数据集，并用它评估 LLM 的引用归属性能。

**💡 创新点**

创新点在于首次将名气与人群属性统一平衡、引入“抑制”这一新的归属失败模式，并以此衡量模型对不同群体作者的可见性。

**🔧 技术方法**

使用了提示工程（直接与间接提示）+检索增强生成 (RAG)、随机采样、正则表达式解析以及 LLM‑consensus 进行人群标签推断。

**📊 数据集**

数据集来源为 500K 条 JSTET 引用，利用 Wikidata 与 LLM 共识对作者进行种族/性别标注，并用 Google 搜索结果计数作为名气代理。

**📈 对比分析**

与 11 种主流 LLM 进行对比，直接提示下最佳模型也仅达到约 25% 的准确率；在无证据与有证据场景下评估归属准确率、抑制率，发现所有模型对 White 男性作者表现最优，抑制率最高的则为 Black/Latino/Asian 群体。

**⚠️ 局限性**

局限性包括：名气代理仅为搜索结果、种族/性别标签仅覆盖四类、数据集规模受限制、仅在离线环境评估，未涵盖在线检索或更细粒度的社会群体。

---

## 92. Towards Predicting Multi-Vulnerability Attack Chains in Software Supply Chains from Software Bill of Materials Graphs

**arXiv ID:** 2604.04977 | [PDF](https://arxiv.org/pdf/2604.04977v1)

**作者:** Laura Baird `[一作]` (University of Colorado Colorado Springs), Armin Moin `[通讯]` (University of Colorado Colorado Springs)

**通讯引用:** 198 | [OpenAlex ID](https://openalex.org/A5090346723)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于SBOM的异构图学习框架，用HGAT预测组件是否存在已知CVE，随后使用MLP对CVE对进行链式共攻击预测，最终在SBOM依赖子图中定位潜在多漏洞攻击链。

**💡 创新点**

创新点在于将SBOM结构与扫描结果视为证据图，将依赖约束嵌入图学习，首次通过CVE对预测和链式共攻击来挖掘跨组件的攻击链。

**🔧 技术方法**

采用异构图注意力网络(HGAT)进行组件标签学习，使用多层感知器(MLP)进行CVE对链式共攻击预测，辅以Syft+Grype扫描、CycloneDX SBOM解析及特征工程。

**📊 数据集**

使用Wild SBOMs公开数据集（200个Python CycloneDX SBOM）以及公开的35条多CVE链（包含4个链长度）进行实验。

**📈 对比分析**

与无依赖边基线对比，HGAT在完整图上实现91.03%准确率、74.02% F1；MLP在CVE对预测上达到0.93 ROC‑AUC，显示链式共攻击可被有效识别。

**⚠️ 局限性**

局限包括：链式共攻击预测仍为对的二分类，缺乏端到端链组装；链样本稀缺导致泛化评估受限；负采样比例和同一CVE多重出现可能导致评估偏高；未完全利用CWE信息。

---

## 93. Watch Before You Answer: Learning from Visually Grounded Post-Training

**arXiv ID:** 2604.05117 | [PDF](https://arxiv.org/pdf/2604.05117v1)

**作者:** Yuxuan Zhang `[一作]` (University Of British Columbia), Kelsey R. Allen `[通讯]` (University Of British Columbia)

**通讯引用:** 1066 | [OpenAlex ID](https://openalex.org/A5023131292)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究揭示并量化了视频理解基准中大量可由文本单独回答的问题，并提出 VidGround：利用 GPT‑5‑mini 对后训练数据进行筛选，仅保留真正需要视觉信息的问题，再结合 GRPO 强化学习对 VLM 进行后训练；

**💡 创新点**

创新点在于：①系统指出 40–60% 的视频理解基准及 30%+ 的后训练数据可被文本偏差覆盖；②提出简单且高效的数据筛选方案（只保留视觉根植问题）；③证明此方法在 RL 后训练中超越多种更复杂的技术。

**🔧 技术方法**

技术手段包括：①用 GPT‑5‑mini 对问题的可回答性进行文本评估；②基于 Group Relative Policy Optimization (GRPO) 的 RL 后训练；③对比多种基线后训练算法（LongVILA‑R1、TW‑GRPO、Video‑RTS、Video‑R1）。

**📊 数据集**

使用的数据集为 Video‑R1‑260K 进行后训练数据筛选；评测基准为 VideoMME、VideoMMMU、MMVU；基线模型为 Qwen2.5‑VL‑7B 及其 SFT 版本。

**📈 对比分析**

与上述基线对比，VidGround 在视觉根植问题上提升 3.5–5.0 分，在全体评测上提升 4.8–6.2 分；在增加帧数时表现更稳定，无出现性能退化。

**⚠️ 局限性**

局限性包括：①筛选依赖大型语言模型，可能漏掉部分视觉问题；②对极长视频或更细粒度视觉推理的改进有限；③在非视频任务的泛化效果尚未充分验证；④模型仍可能在极端情况下利用文本提示。

---

## 94. Uncertainty-Guided Latent Diagnostic Trajectory Learning for Sequential Clinical Diagnosis

**arXiv ID:** 2604.05116 | [PDF](https://arxiv.org/pdf/2604.05116v1)

**作者:** Xuyang Shen `[一作]` (University of Connecticut), Martin Renqiang Min `[通讯]` (NEC Laboratories America)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种利用大型语言模型的序列诊断框架，学习诊断路径的潜在分布以实现信息获取与决策的协同优化。

**💡 创新点**

核心创新在于将诊断路径视为潜在变量，并通过信息增益构建的路径后验分布对规划LLM进行监督，从而实现对诊断轨迹的结构化正则化。

**🔧 技术方法**

采用了双代理LLM架构——规划代理与诊断代理；使用能量模型、信息增益、KL对齐以及两阶段训练（先冻结诊断代理再训练规划代理）等技术。

**📊 数据集**

在MIMIC‑CDM腹部诊断基准（2400例四种疾病）上进行实验。

**📈 对比分析**

与静态信息、随机规划、ReAct、LA‑CDM等基线对比，LDTL在平均准确率和F1上均超过94%与91%，并显著减少所需检验次数。

**⚠️ 局限性**

局限包括仅在单一腹部诊断数据集上验证、动作空间仅为测试类别级别、未考虑动态检验依赖和开放式检验生成，需进一步扩展到更大更复杂的临床场景。

---

## 95. Hierarchical Mesh Transformers with Topology-Guided Pretraining for Morphometric Analysis of Brain Structures

**arXiv ID:** 2604.05215 | [PDF](https://arxiv.org/pdf/2604.05215v1)

**作者:** Yujian Xiong `[一作]` (Arizona State University), Yalin Wang `[通讯]` (Arizona State University)

**通讯引用:** 11644 | [OpenAlex ID](https://openalex.org/A5100740828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种统一的八叉树 Transformer（OctEncoder），能够同时处理三维医学网格（体素化的四面体网格和表面三角网格），并在网络中融合任意数量的顶点级临床特征（如皮层厚度、曲率、髓鞘含量等），从而实现多尺度自监督预训练和下游任务的高效迁移。

**💡 创新点**

① 简单形感知的多树八叉树构造，支持任意阶简形（顶点、边、面、四面体中心）并通过加权融合获得更丰富的几何信息；② 位置编码采用条件位置编码（CPE）在每棵树内部学习空间依赖；③ 通过掩码自编码器（MAE）同时重建坐标和多模态特征，实现对未标记网格的无监督预训练。

**🔧 技术方法**

OctEncoder 基于多层 Transformer、局部窗口自注意力与稀疏化的 Dilated Attention；使用 Z‑order 码进行八叉树分区；利用线性投影将多维临床特征与坐标拼接；MAE 训练中采用 Chamfer Distance + MSE 损失。

**📊 数据集**

1）OASIS‑3（未标记大规模脑 MRI 体积网格）用于预训练；2）ADNI（四面体脑网格）用于阿尔茨海默病诊断与脑淀粉样蛋白预测；3）MELD（表面三角网格）用于癫痫发作相关的皮层发育不全（FCD）分割；4）ScanNet（室内场景三角网格）用于三维语义分割验证跨域泛化。

**📈 对比分析**

与 ChebyNet、GAT、TetCNN 等 GNN、OctFormer、Point Transformer、Mix3D、O‑CNN 等方法进行对比。OctEncoder 在 ADNI 的 AD vs. CN、AD vs. MCI、MCI vs. CN 任务中分别达到 90.7%、92.0%、91.4% 的准确率，显著优于基线（最高 87.6%）；在 FCD 分割中 Lesion IoU 提升至 0.51（高于 0.34）；在 ScanNet 语义分割中 mIoU 达到 0.777，略高于最先进的 0.775。

**⚠️ 局限性**

缺点主要包括：① 仍需大量未标记网格数据进行预训练，预训练成本相对较高；② 目前仅在医学网格和室内场景两类任务验证，尚未评估在其他医学结构（如血管、肿瘤）上的鲁棒性；③ 对极大规模网格的内存与计算开销仍有提升空间。

---

## 96. Compiled AI: Deterministic Code Generation for LLM-Based Workflow Automation

**arXiv ID:** 2604.05150 | [PDF](https://arxiv.org/pdf/2604.05150v1)

**作者:** Geert Trooskens `[一作]` (XY.AI Labs), Walter A. De Brouwer `[通讯]` (Stanford University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了一种将大型语言模型（LLM）仅在编译阶段调用，生成可执行代码，随后在运行时无需再调用模型，从而实现可预测、可审计、成本高效的企业工作流自动化体系；通过四阶段验证管道保证安全性、语法、执行与准确性。

**💡 创新点**

核心创新在于：① 将LLM视为编译器而非解释器，单次调用即生成可执行代码；② 通过受限模板与模块化库实现生成空间约束，显著降低幻觉风险；③ 引入四阶段验证流程（安全、语法、执行、准确性）实现编译时错误检测；④ 在医疗等高风险场景下展示该模式在可靠性、审计性与成本上的优势。

**🔧 技术方法**

技术包括：受限生成的LLM调用（Claude Opus 4.5），模板/模块库、YAML规范化工作流，四阶段验证（Bandit、Semgrep、mypy、测试执行、准确性校验），Temporal Workflow、Pydantic校验器，边界代理LLM调用与安全门控，成本与Token计量分析。

**📊 数据集**

使用两组公开数据集：BFCL（400条函数调用任务）和DocILE（5,680份OCR降质发票，包含键值提取与行项目识别）进行评测；对比Direct LLM、LangChain、AutoGen以及Deterministic/Code Factory变体。

**📈 对比分析**

通过Token消耗、成本、延迟、可重复性、任务完成率、准确率等多维度评估；结果显示：编译AI在BFCL任务中一次性生成9,600 token后无运行时Token，1,000次交易Token消耗减少57倍；在DocILE任务中，Code Factory实现80% KILE、80.4% LIR，延迟比直接LLM低2.3×；整体TCO在1M交易/月时约为Direct LLM的1/40至1/57，表现出显著的成本和性能优势。

**⚠️ 局限性**

主要局限包括：① 需要高质量的YAML规范，规范编写难度和迭代成本；② 并非所有工作流可完全编译，仍需运行时LLM处理开放式内容；③ 编译失败率约4%，导致重新生成；④ 评估仅覆盖两类任务，需进一步验证更广泛场景；⑤ 代码质量虽高但复杂度仍高，模板改进空间；⑥ 对LLM版本的依赖性，模型更新需重新验证。

---

## 97. COMB: Common Open Modular robotic platform for Bees

**arXiv ID:** 2604.04980 | [PDF](https://arxiv.org/pdf/2604.04980v1)

**作者:** Pranav Kedia `[一作]`, Tim Landgraf `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一个可插拔的蜂巢实验平台 COMB，支持在标准观察蜂巢框内进行定位、扫描和局部激励。

**💡 创新点**

核心创新在于统一的二维定位台、可密封的可移动接入窗口以及共享的嵌入式控制架构，使得多种感知/驱动模块可热插拔，且系统开源。

**🔧 技术方法**

采用步进电机驱动的 XY 平移台、ESP32 嵌入式控制、聚碳酸酯可旋转窗口、光学摄像头、PCB 线圈等硬件，软件基于 Arduino/ESP32 和 OpenCV 实现轨迹跟踪和图像拼接。

**📊 数据集**

使用平台自身生成的实验视频和图像数据集：舞蹈轨迹录像、组合扫描图像拼接、翅膀振动视频，构成评估数据。

**📈 对比分析**

通过跟踪实验评估，平均交叉误差 1.63 mm、纵向误差 1.33 mm、整体欧氏误差 2.32 mm；扫描模式图像重叠率约 55–60%；翅膀振动频率在 13.9 Hz 和 28 Hz 两档，均在蜜蜂信号频段内。

**⚠️ 局限性**

局限性包括仅实现二维平移、对高曲率轨迹误差较大、需要定期清洁接入窗口以处理蜂胶、缺乏闭环行为反馈和三维交互等。

---

## 98. Protecting and Preserving Protest Dynamics for Responsible Analysis

**arXiv ID:** 2604.05256 | [PDF](https://arxiv.org/pdf/2604.05256v1)

**作者:** Cohen Archbold `[一作]` (University of Kentucky), Abdullah-Al-Zubaer Imran `[通讯]` (University of Kentucky)

**通讯引用:** 267 | [OpenAlex ID](https://openalex.org/A5101789164)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过条件生成合成抗议图像替代真实图像，支持无识别信息的集体行动分析

**💡 创新点**

提出将隐私风险评估、下游分析和公平性整合为责任驱动的框架，并利用StyleGAN3的条件生成实现高质量合成图像

**🔧 技术方法**

使用StyleGAN3的条件生成、WGAN‑GP、DP‑GAN对比实验、ResNet‑50下游分类、会员推断攻击和面部属性检测

**📊 数据集**

主要使用公开抗议图像基准数据集（如Protest Dataset 2023）、VGKG预训练集、UCF‑QNRF、JHU‑CROWD++、NWPU‑Crowd等

**📈 对比分析**

在FID/KID/IS指标上，StyleGAN3达到15.8/0.007/6.5，远优于DP‑GAN与WGAN‑GP；下游抗议与暴力预测的AUC‑ROC与真实数据相当且显著优于DP‑SGD；成员推断攻击精度明显下降

**⚠️ 局限性**

预训练效果有限，合成数据仍可能泄露隐私，面部属性检测误差导致公平性评估受限，且缺乏正式的差分隐私保证，可能被误用为监控工具

---

## 99. From Use to Oversight: How Mental Models Influence User Behavior and Output in AI Writing Assistants

**arXiv ID:** 2604.05166 | [PDF](https://arxiv.org/pdf/2604.05166v1)

**作者:** Shalaleh Rismani `[一作]` (McGill University), AJung Moon `[通讯]` (McGill University)

**通讯引用:** 930 | [OpenAlex ID](https://openalex.org/A5081123654)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了功能型与结构型精神模型对 AI 写作助手使用者的控制行为、写作质量与使用体验的影响。

**💡 创新点**

首次利用视频干预对精神模型进行操纵，并揭示结构型模型虽提升易用性，却可能导致错误接受率升高。

**🔧 技术方法**

采用改良版 CoAuthor（基于 GPT‑3.5‑turbo‑instruct）平台，注入故意语法错误的建议，并记录交互日志。

**📊 数据集**

48 名受试者完成求职信写作任务，使用的提示文本与错误注入建议构成数据集。

**📈 对比分析**

通过功能型与结构型对照，发现控制行为无显著差异；结构型组语法错误多，但整体写作质量相近；易用性显著提升；其余评估无显著差异。

**⚠️ 局限性**

样本受限于受试者先前经验、错误注入人为设置、精神模型动态变化未被捕捉，以及任务范围狭窄，限制了结果推广。

---

## 100. FNO$^{\angle θ}$: Extended Fourier neural operator for learning state and optimal control of distributed parameter systems

**arXiv ID:** 2604.05187 | [PDF](https://arxiv.org/pdf/2604.05187v1)

**作者:** Zhexian Li `[一作]` (University of Southern California), Ketan Savla `[通讯]` (University of Southern California)

**通讯引用:** 2350 | [OpenAlex ID](https://openalex.org/A5007642972)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了扩展 Fourier 神经算子 FNO^∠θ，用于学习 PDE 状态和线性二次最优控制，并验证其性能

**💡 创新点**

在传统 FNO 的卷积层中将频率变量从实域扩展到复域，加入可学习的相位参数 θ，利用 Ehrenpreis‑Palamodov 原理的复积分表示提升表达能力

**🔧 技术方法**

采用 Fourier 神经算子架构、复频率扩展、神经网络权重与相位参数学习、有限差分仿真、Gaussian 随机场生成边界条件、MSE 训练评估

**📊 数据集**

通过随机 Gaussian 字段生成 50 条 Burgers 方程的边界条件/初始条件，使用有限差分得到对应状态和最优控制作为训练样本

**📈 对比分析**

对比 FNO 与 FNO^∠θ 在学习状态和控制算子时的相对 MSE；FNO^∠θ 在状态算子上 MSE 从 0.039 降到 0.0089，控制算子从 0.098 降到 0.042，显著提升；误差分布显示边界误差显著减小

**⚠️ 局限性**

仅在单一 1D Burgers 方程上验证；对更高维、非线性或多物理场 PDE 的推广尚未评估；复频率参数训练稳定性和解释性尚待深入

---

## 101. Comparative Characterization of KV Cache Management Strategies for LLM Inference

**arXiv ID:** 2604.05012 | [PDF](https://arxiv.org/pdf/2604.05012v1)

**作者:** Oteo Mamo `[一作]` (Florida State University), Weikuan Yu `[通讯]` (Florida State University)

**通讯引用:** 2874 | [OpenAlex ID](https://openalex.org/A5070216261)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比评估了三种 KV 缓存管理框架（vLLM、H2O、InfiniGen）在大型语言模型推理中的内存占用、吞吐量、延迟与精度。

**💡 创新点**

创新之处在于系统化比较三大范式，并给出在不同批量、上下文长度、稀疏度预算下的最佳框架选择指导。

**🔧 技术方法**

采用的技术包括 PagedAttention、FlashAttention‑2、Chunked Prefill、稀疏化（Heavy‑Hitter 选择）、CPU‑GPU 层次化 KV 缓存、SVD 维度压缩等。

**📊 数据集**

使用的数据集包括 Llama‑3.1‑8B/70B、GPT‑OSS‑20B 模型、LMSYS‑Chat‑1M、English Wikipedia、PIQA、HellaSwag、COPA、WinoGrande、OpenBookQA、BoolQ。

**📈 对比分析**

通过 TTFT、吞吐量、端到端延迟、GPU/CPU 内存占用以及精度差异等指标进行量化比较；结果显示 vLLM 速度最快、H2O 内存占用最低但精度下降、InfiniGen 通过动态选择保留精度但吞吐量显著下降。

**⚠️ 局限性**

局限在于 InfiniGen 的 CPU‑GPU 传输瓶颈、H2O 的永久剔除导致知识检索错误，以及实验仅覆盖三种模型与特定硬件环境。

---

## 102. Learning to Retrieve from Agent Trajectories

**arXiv ID:** 2604.04949 | [PDF](https://arxiv.org/pdf/2604.04949v1)

**作者:** Yuqi Zhou `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 24400 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过分析深度研究型LLM搜索代理的交互轨迹，构建了LRAT框架，用来直接从代理轨迹中提取检索监督信号并训练密集检索模型。

**💡 创新点**

创新点包括：① 将代理的浏览行为作为正样本、未浏览结果作为负样本，解决了位置偏差问题；② 用后浏览推理长度作为连续权重，引入重要性感知的对比学习；③ 采用LLM作为判别器过滤无用正样本；④ 证明代理轨迹可形成可持续的数据流轮，实现检索器的迭代自我提升。

**🔧 技术方法**

核心技术包括：bi‑encoder 密集检索器、InfoNCE 加权对比学习、LLM（Qwen3-30B-A3B-Thinking-2507）作为正样本过滤器、稀疏/密集检索器混合负采样、以及指数饱和函数映射推理长度到权重。

**📊 数据集**

使用的主要数据集有：InfoSeekQA（10K种子查询、1.12M Wiki 文档块）用于轨迹收集；BrowseComp-Plus（830题）用于检索质量评估；InfoSeek-Eval（300题）用于任务成功率评估；Wiki-25-Dump 作为检索语料库。

**📈 对比分析**

在InfoSeek-Eval和BrowseComp-Plus上与多种基线检索器（BM25、Qwen3-Embedding 0.6/4/8B、Multilingual-E5-Large）以及多种搜索代理（AgentCPM-Explore-4B、WebExplore-8B、Tongyi-DeepResearch-30B、GPT-OSS-120B 等）进行比较。结果显示：LRAT 在所有检索器上都能提升 7–37% 的证据召回，任务成功率提升 4–15%，平均交互步骤减少约 25–30%，并在不同规模代理中保持一致性。

**⚠️ 局限性**

局限性包括：① 依赖深度研究型代理轨迹，尚未验证在更一般的交互式搜索代理上的表现；② LLM 过滤器的误判可能导致正样本丢失或错误；③ 对极大 top‑K 检索预算时性能可能下降；④ 需要大量代理轨迹，训练成本仍不低；⑤ 未在长期在线部署环境中进行充分的实时评估。

---

## 103. Ghosting the Machine: Stop Calling Human-Agent Relations Parasocial

**arXiv ID:** 2604.05197 | [PDF](https://arxiv.org/pdf/2604.05197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 104. Part-Level 3D Gaussian Vehicle Generation with Joint and Hinge Axis Estimation

**arXiv ID:** 2604.05070 | [PDF](https://arxiv.org/pdf/2604.05070v1)

**作者:** Shiyao Qian `[一作]` (Huawei Noah’s Ark Lab), Bingbing Liu `[通讯]` (Huawei Noah’s Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种从单张或稀疏多视图图像生成可动画化的3D高斯车辆模型，并能够预测可动部件的运动参数。

**💡 创新点**

创新点在于：①通过部件边缘细化模块强制每个高斯仅归属单一部件，消除跨部件颜色/不透明度溢出；②引入基于PointNet++的运动推理头，直接回归车轮转轴和门铰链位置，实现在单帧图像中获得完整的运动信息。

**🔧 技术方法**

采用TRELLIS生成初始3D高斯资产，SAGA实现高斯层面的语义分割，PointNet++完成部件分类与运动参数回归，并辅以YOLO+SAM2的可动部件掩模插入与边缘微调损失。

**📊 数据集**

主要使用自制的100辆CAD汽车点云数据集（包含门和轮的真实运动参数），以及CARLA第二代车辆蓝图集用于评估；同时利用3DRealCar等公开数据进行训练补充。

**📈 对比分析**

通过消融实验与基线（TRELLIS+SAGA）对比，加入掩模插入和边缘细化后，CLIP相似度从0.848提升至0.873，LPIPS误差从0.305降至0.283，证明模型在视觉质量和可编辑性上都有显著提升。

**⚠️ 局限性**

局限性包括：仅依赖二维分割掩模，无法对部件厚度和内部结构提供约束；在门打开等大幅变形时容易出现零厚度或内部空间混乱；对大型高斯生成模型的内部结构支持不足。

---

## 105. Web Retrieval-Aware Chunking (W-RAC) for Efficient and Cost-Effective Retrieval-Augmented Generation Systems

**arXiv ID:** 2604.04936 | [PDF](https://arxiv.org/pdf/2604.04936v1)

**作者:** Uday Allu `[一作]` (Yellow.ai), Biddwan Ahmed `[通讯]` (Yellow.ai)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Web Retrieval-Aware Chunking (W‑RAC) 框架，用于高效、可观测的网页文档分块。

**💡 创新点**

创新点在于将分块视为规划任务，采用结构化ID代替文本生成，显著降低LLM成本并减少幻觉风险。

**🔧 技术方法**

技术包括确定性网页解析、LLM仅生成分块计划（ID列表）、本地ID映射重组文本、嵌入生成与索引。

**📊 数据集**

使用 RAG‑Multi‑Corpus 基准集，包含 236 篇多格式文档（PDF/Markdown/HTML/…）和 786 对查询-答案。

**📈 对比分析**

与传统 agentic 分块对比，W‑RAC 在输出 token 降低 84.6%、整体延迟 ↓60%、LLM 成本 ↓51.7%，同时在多组织与多查询类型下提升 Precision@3 约 29%。

**⚠️ 局限性**

局限性为输入 token 约 ↑50%，对复杂结构化网页的适配仍需改进，以及对 LLM 规划错误的鲁棒性尚未充分验证。

---

## 106. Streaming Chain

**arXiv ID:** 2604.04995 | [PDF](https://arxiv.org/pdf/2604.04995v1)

**作者:** Yi Lyu `[一作]` `[通讯]` (University of Wisconsin-Madison), Yi Lyu (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文设计并实现了自适应区块生成机制，通过数学模型预测交易延迟与成功率，并在Hyperledger Fabric上进行实验评估；

**💡 创新点**

创新点在于提出针对不同工作负载与硬件资源动态调整区块大小与超时的自适应算法，并给出精确的交易延迟与成功率理论模型，验证了模型与实验的高度吻合；

**🔧 技术方法**

使用了数学建模（线性回归、概率论）、Docker监控技术、YCSB负载生成器、Hyperledger Fabric Raft排序服务、以及自定义模拟框架；

**📊 数据集**

采用了YCSB工作负载（1000次更新操作，10个100字节字段）和自定义的Zipf分布键盘访问模式（写/读比例、键范围、α值）进行实验；

**📈 对比分析**

通过将模型预测与真实测量（平均延迟、CPU/IO使用率）及模拟实验中的成功率进行对比，模型在低负载下误差<5%，在高负载下可解释延迟波动；

**⚠️ 局限性**

局限性包括：未在多机网络环境下验证网络IO表现，缺少排队延迟模型，且模型假设事务独立同分布，真实系统中可能存在非IID特征导致预测误差；

---

## 107. Proximity Measure of Information Object Features for Solving the Problem of Their Identification in Information Systems

**arXiv ID:** 2604.04939 | [PDF](https://arxiv.org/pdf/2604.04939v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 108. El Nino Prediction Based on Weather Forecast and Geographical Time-series Data

**arXiv ID:** 2604.04998 | [PDF](https://arxiv.org/pdf/2604.04998v1)

**作者:** Viet Trinh `[一作]` (University of Economics and Law), Hoai-Nam Nguyen Dang `[通讯]` (University of Economics and Law)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过整合CNN和LSTM的混合深度学习框架，利用海面温度和海底热含量数据预测厄尔尼诺事件。

**💡 创新点**

引入ConvLSTM-XT架构，结合空间热图与时间序列，实现多尺度预测模型，并在5个季度连续预测中展示高精度。

**🔧 技术方法**

使用CNN提取空间特征、ConvLSTM捕捉时间依赖、全连接层回归SST，并通过Adam优化器和MSE损失函数训练模型。

**📊 数据集**

采用NOAA ERSST V5海面温度数据和中国科学院海洋数据中心的海底热含量数据，时间范围为2000-2023年，覆盖尼诺3.4区。

**📈 对比分析**

通过对不同预测配置（从观测到全预测）与实际ONI阈值的混淆矩阵比较，配置1-4的准确率为90.57%，完全预测配置5的准确率为83.02%。

**⚠️ 局限性**

仅使用SST和OHC缺乏更广泛的气候变量，导致长周期预测精度下降，模型对观测数据的依赖较高。

---

## 109. Prune-Quantize-Distill: An Ordered Pipeline for Efficient Neural Network Compression

**arXiv ID:** 2604.04988 | [PDF](https://arxiv.org/pdf/2604.04988v1)

**作者:** Longsheng Zhou `[一作]` (University of Science and Technology of China), Yu Shen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 33123 | [OpenAlex ID](https://openalex.org/A5101938357)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一条按顺序执行的三步压缩流水线（全局无结构剪枝 → INT8 量化感知训练 → 知识蒸馏），以在标准 CPU 上获得低延迟、低存储且准确率可控的模型。

**💡 创新点**

证明压缩步骤的顺序对最终准确率至关重要，并提供了一套最小可复现的三步流水线，且全部基于实际测量的延迟而非代理指标，避免使用专用稀疏核。

**🔧 技术方法**

采用无结构剪枝、INT8 量化感知训练 (QAT)、知识蒸馏 (KD) 以及基于 FP32 教师的训练与蒸馏策略。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100 数据集上使用 ResNet‑18、WRN‑28‑10、VGG‑16‑BN 三种网络，另对 ResNet‑20 进行文献对比。

**📈 对比分析**

通过与单一压缩技术以及不同步骤顺序的对比，在相同训练预算和稀疏 INT8 部署形式下，平均 CPU 延迟 0.99–1.42 ms，压缩率 6–10×，速度提升 2–3×；在相对 BOPs 下亦优于多种混合精度方法。

**⚠️ 局限性**

仅使用无结构稀疏，未利用专用稀疏加速硬件；适用于 CNN，未探讨结构化稀疏或更自动化的策略。

---

## 110. $π^2$: Structure-Originated Reasoning Data Improves Long-Context Reasoning Ability of Large Language Models

**arXiv ID:** 2604.05114 | [PDF](https://arxiv.org/pdf/2604.05114v1)

**作者:** Quyet V. Do `[一作]` (Virginia Tech), Tu Vu `[通讯]` (Virginia Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为π^2的全流程管道，利用Wikipedia表格生成高质量长上下文推理数据，并通过双路径验证和反向翻译生成结构化推理轨迹；

**💡 创新点**

创新点在于：①通过表格扩展产生跨文档的多列信息；②结合SQL与Python双路径执行验证答案；③利用反向翻译将结构化推理轨迹转化为自然语言推理步骤；④支持自蒸馏提升小模型性能；

**🔧 技术方法**

采用的大规模语言模型（LLM）进行提示生成，使用LoRA进行监督微调，BM25+压缩上下文至96K token，双路径（SQL+Python）验证，反向翻译生成推理轨迹；

**📊 数据集**

使用的数据集包括：从Wikipedia抽取并扩展的表格、922条经过验证的问答样本、228条人类审核的Bench测试集以及四个公开长上下文推理基准（LongSeal、LongBench-v2、Oolong、OfficeQA）；

**📈 对比分析**

实验结果显示，在四个基准上，微调后的模型在LLama2‑7B上平均提升4.3%，在LLama2‑13B上提升2.7%；高推理努力配置下，平均得分53.22，逼近更大规模或专有模型；仅用100条高质量样本亦可提升1.34%–2.85%；自蒸馏在部分配置下提升约4.4%；

**⚠️ 局限性**

局限性包括：数据来源单一（Wikipedia）可能带来偏见；表格扩展仅限3列；基准测试覆盖范围有限；自蒸馏在高推理努力配置下效果不佳；需要人工审核保证Bench质量。

---

## 111. Analyzing Persistent Alltoallv RMA Implementations for High-Performance MPI Communication

**arXiv ID:** 2604.05099 | [PDF](https://arxiv.org/pdf/2604.05099v1)

**作者:** Evelyn Namugwanya `[一作]` `[通讯]` (Tennessee Tech University), Evelyn Namugwanya (Tennessee Tech University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计、实现并评估了基于 MPI RMA 的持久化 Alltoallv 集合通信，采用 fence 和 lock 同步，分离一次性初始化与每次迭代执行。

**💡 创新点**

创新点在于首次引入持久化 RMA 集合通信接口，将窗口、数据类型、位移等元数据缓存，消除每次调用的重建成本，并对 fence 与 lock 两种同步方式进行系统比较及层级化改进。

**🔧 技术方法**

使用 MPI‑4 RMA 接口（MPI_Put/Get、MPI_Win_create、MPI_Fence、MPI_Win_lock 等）以及自定义持久化请求对象实现通信。

**📊 数据集**

在 LLNL 的 Dane 超算上使用规模化全局数组以及 SuiteSparse 集合稀疏矩阵（如 hugetrace‑00020）进行实验。

**📈 对比分析**

通过 1000 次迭代的时间测量和 break‑even 模型比较，结果显示 fence‑持久化在消息 ≥ 32,768 bytes 时可立即获益，448 进程时可降低约 38 % 运行时；在稀疏模式下 fence 持久化优于 lock，且在大消息下显著提高性能。

**⚠️ 局限性**

局限在小消息（≤ 16,384 bytes）下元数据缓存收益被同步开销抵消；lock‑持久化受锁竞争影响，且对更复杂同步场景支持不足。

---

## 112. Beneath the Surface: Investigating LLMs' Capabilities for Communicating with Subtext

**arXiv ID:** 2604.05273 | [PDF](https://arxiv.org/pdf/2604.05273v1)

**作者:** Kabir Ahuja `[一作]` (Google DeepMind), Andrew Kyle Lampinen `[通讯]` (Google DeepMind)

**通讯引用:** 1619 | [OpenAlex ID](https://openalex.org/A5030015839)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并系统评估了大型语言模型（LLM）在创意交流中使用潜台词的能力，涵盖多代理游戏（如Dixit、Wavelength）和叙事环境（历史寓言理解与新颖故事创作）。

**💡 创新点**

创新点在于：①首次将潜台词作为核心指标构建可量化的评估环境；②设计了“隐式共识”与“读者身份”等变量，探究LLM在隐含信息与读者视角下的沟通策略；③通过对多模型的定量实验，揭示当前前沿模型在潜台词生成与理解方面的不足。

**🔧 技术方法**

主要技术包括：多代理强化学习环境搭建、基于大型预训练模型的文本生成与理解（如GPT‑4、Claude、PaLM等），以及自定义的评价指标（如目标听众正确解码率）。

**📊 数据集**

使用的数据集有：①公开的Dixit与Wavelength游戏日志；②自行构造的历史寓言文本与对应的多种潜台词；③在新故事创作环境中使用的作者身份与读者背景信息。

**📈 对比分析**

比较方法：将不同模型在相同任务下的成功率（目标听众正确解码率）与最佳可实现得分对比。结果显示，最先进模型在Dixit类任务中大约60%时生成可被所有玩家理解的线索；加入共享背景后可降低明显，但仍约为最佳得分的两倍；在寓言解释任务中，最佳模型的正确率从26%提升至73%；在最具挑战的新故事环境中，最高成功率仅为22%。

**⚠️ 局限性**

局限性包括：①评估目标始终显式设定为“只让特定听众理解”，未探究隐式潜台词使用；②实验使用的模型为实验完成前的版本，缺乏最新模型的评估；③未进行人类基准对比；④未深入研究模型如何构建与更新共同基础（common ground）和潜在的误差来源。

---

## 113. What Makes a Good Response? An Empirical Analysis of Quality in Qualitative Interviews

**arXiv ID:** 2604.05163 | [PDF](https://arxiv.org/pdf/2604.05163v1)

**作者:** Jonathan Ivey `[一作]`, Ziang Xiao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文提供了在LuaLaTeX和XeLaTeX环境下使用ACL样式文件的示例演示。

**💡 创新点**

创新点在于展示了如何在不同的TeX引擎中轻松调用ACL模板，并兼顾多语言文本显示。

**🔧 技术方法**

采用的技术是LuaLaTeX或XeLaTeX与ACL官方样式文件的结合。

**📊 数据集**

未使用任何具体数据集，仅包含演示性文本。

**📈 对比分析**

没有进行方法比较或性能评估，主要用于排版演示。

**⚠️ 局限性**

限制在于缺乏实际实验或评测，无法验证在真实论文写作中的效果与效率。

---

## 114. SkillAttack: Automated Red Teaming of Agent Skills through Attack Path Refinement

**arXiv ID:** 2604.04989 | [PDF](https://arxiv.org/pdf/2604.04989v1)

**作者:** Zenghao Duan `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SkillAttack框架，利用迭代对话（不修改技能）动态验证LLM代理技能的潜在漏洞。

**💡 创新点**

通过三阶段闭环搜索：漏洞分析→并行攻击生成→基于执行轨迹的反馈驱动路径细化，实现无改动技能的漏洞探测。

**🔧 技术方法**

使用A.I.G进行技能审计，LLM生成与执行攻击路径，执行轨迹追踪和Judge评估，迭代优化prompt。

**📊 数据集**

评估数据集包含Skill-Inject基准的71个对抗技能以及ClawHub上最热门的100个真实技能。

**📈 对比分析**

与Direct Attack和Skill-Inject基线对比，实验10种LLM，SkillAttack在对抗技能上的攻击成功率（ASR）达到0.73–0.93，真实技能上达到0.26，显著优于基线。

**⚠️ 局限性**

局限性包括仅使用单一Judge模型、只关注prompt层攻击、样本覆盖面有限、未提出对应防御措施。

---

## 115. Scalar Federated Learning for Linear Quadratic Regulator

**arXiv ID:** 2604.05088 | [PDF](https://arxiv.org/pdf/2604.05088v1)

**作者:** Mohammadreza Rostami `[一作]` (University of California, Irvine), Solmaz S. Kia `[通讯]` (University of California, Irvine)

**通讯引用:** 2393 | [OpenAlex ID](https://openalex.org/A5049439078)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为 ScalarFedLQR 的联邦学习算法，用于在多智能体 LQR 控制中实现模型无关的策略优化，且每个智能体仅上传一个标量投影梯度。

**💡 创新点**

创新点在于将梯度投影压缩为单一标量，使每个智能体的上行通信复杂度从 O(d) 降至 O(1)，并证明投影误差随参与智能体数量增大而减小，从而实现规模化通信效率提升与线性收敛速度的平衡。

**🔧 技术方法**

采用了零阶梯度估计、随机 Rademacher 投影、投影重构聚合、局部稳定性子集和 Polyak–Łojasiewicz (PL) 条件下的梯度下降分析。

**📊 数据集**

使用人工生成的三维状态三维输入的离散 LTI 系统集（M=10），通过对齐参数的扰动构造不同程度的异质性，并采用 Q=2I₃, R=12I₃ 的成本矩阵。

**📈 对比分析**

与传统 FedLQR 通过完整梯度传输做对比；在通信轮数维度上两者收敛曲线相近，但在固定比特预算下 ScalarFedLQR 的恢复率明显更高（低异质性约 54% vs 29%，高异质性约 31% vs 14%），展示了显著的通信效率优势。

**⚠️ 局限性**

局限性包括对异质性强度的理论分析仍待完善、对零阶梯度估计噪声的控制依赖实验设置、以及在极高维度或极少智能体时投影误差可能变大导致步长受限。

---

## 116. Scaling Coding Agents via Atomic Skills

**arXiv ID:** 2604.05013 | [PDF](https://arxiv.org/pdf/2604.05013v1)

**作者:** Yingwei Ma `[一作]` (Hong Kong University Of Science And Technology), Shing-Chi Cheung `[通讯]` (Hong Kong University Of Science And Technology)

**通讯引用:** 8739 | [OpenAlex ID](https://openalex.org/A5034057959)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于原子技能的LLM编码代理扩展范式，首先从软件工程工作流中抽象出五种通用原子技能（代码定位、代码编辑、单元测试生成、问题重现、代码评审），然后通过共享策略的联合强化学习对这些原子技能进行训练，最终提升代理在未见合成任务（如bug修复、代码重构、安全漏洞利用等）上的表现。

**💡 创新点**

创新点主要包括：①将合成编码任务拆解为可评估、可组合的原子技能；②构建统一的原子技能接口和基于沙盒的执行奖励机制；③提出共享策略的联合RL框架（采用GRPO）以避免技能间的负向干扰，并实现正向迁移；④通过对比单任务RL与联合RL，证明了原子技能联合训练在通用性与专门化之间取得更佳平衡。

**🔧 技术方法**

技术上使用了：①基于GLM‑4.5‑Air‑Base的大型语言模型；②对原子技能数据进行的监督微调（SFT）；③采用Group-based Relative Policy Optimization (GRPO) 的联合RL；④在Kubernetes集群中部署的多容器沙盒执行环境；⑤最小化工具集合（命令执行与文件编辑）以简化动作空间；⑥分离的Rollout与Trainer工作节点提升数据采集效率。

**📊 数据集**

数据集方面：①原子技能训练数据来自GitHub issue、PR和代码库，约1500条经过评审的轨迹（每项技能约300条）；②合成任务用于评估的公开基准包括SWE‑bench Verified、SWE‑bench Multilingual、Terminal‑Bench、SEC‑Bench，以及自行构造的Code Refactoring benchmark（300条真实提交）。

**📈 对比分析**

评估方法：在相同的SFT初始化模型基础上，比较Base‑SFT、Base‑SFT+RL与GLM‑4.5‑Air三者在五项原子技能和五项OOB合成任务上的Avg@3分数。结果显示，联合RL在原子技能上平均提升约18.7%，在OOB合成任务上也实现了显著提升（例如SWE‑bench Verified从0.507升至0.585）。学习曲线表明，联合RL在训练过程中持续提升性能且不出现负向干扰。

**⚠️ 局限性**

局限性包括：①只覆盖了五种原子技能，尚未扩展到更细粒度或跨域的技能；②训练高度依赖沙盒化执行环境，部署成本较高；③奖励设计仍可能存在奖励稀疏或误判的风险；④未能系统评估在极端任务分布或非常大规模代码库上的表现。

---

## 117. This Treatment Works, Right? Evaluating LLM Sensitivity to Patient Question Framing in Medical QA

**arXiv ID:** 2604.05051 | [PDF](https://arxiv.org/pdf/2604.05051v1)

**作者:** Hye Sun Yun `[一作]` (Northeastern University), Byron C. Wallace `[通讯]` (Northeastern University)

**通讯引用:** 13345 | [OpenAlex ID](https://openalex.org/A5036790226)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在受控检索增强生成（RAG）医疗问答环境下，系统评估了正负问句框架以及技术与通俗语言对LLM一致性的影响。

**💡 创新点**

首次以大规模对比正负框架对LLM一致性的量化影响，并探讨多轮对话中框架效应的放大，揭示了LLM在相同证据下仍易受问句措辞操控。

**🔧 技术方法**

采用检索增强生成（RAG）、多模型推理、LLM-as-judge、BERT嵌入等技术对答案进行评估与一致性检测。

**📊 数据集**

构建了基于Cochrane系统综述与RCT摘要的6,614对问句数据集，覆盖技术与通俗两种语言风格。

**📈 对比分析**

通过比较正负框架下的证据一致率、相似度、实体与引用重叠等指标，并使用逻辑回归分析，发现正负框架一致率平均下降约4%–8%，多轮对话进一步恶化。

**⚠️ 局限性**

局限在于仅在RAG环境下测试、简化的问句范围、缺乏真实多样化患者查询以及未结合完整人类专家评审。

---

## 118. Context Collapse: Barriers to Adoption for Generative AI in Workplace Settings

**arXiv ID:** 2604.05151 | [PDF](https://arxiv.org/pdf/2604.05151v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 119. Closed-Loop Autonomous Software Development via Jira-Integrated Backlog Orchestration: A Case Study in Deterministic Control and Safety-Constrained Automation

**arXiv ID:** 2604.05000 | [PDF](https://arxiv.org/pdf/2604.05000v1)

**作者:** Elias Calboreanu `[一作]` `[通讯]` (Swift Group, LLC), Elias Calboreanu (Swift Group, LLC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个基于七阶段确定性闭环的自动化软件生命周期管理系统，包含从需求摄取、规范化、执行到验证和发布的完整闭环。

**💡 创新点**

创新点在于将生命周期管理视为控制回路，结合了正式的FMEA、安全边界、碰撞锁定、降级模式、可追溯的TRACE证据链，以及将AI角色限定在监督层的可界定自治模式。

**🔧 技术方法**

技术主要包括：有限状态机控制、Jira MCP工具接口、四层模糊匹配规范化引擎、LATTICE阈值控制、MANDATE授权框架、Claude等LLM的监督调用、时间预算检查与重试机制、并发工作池、全链路日志和哈希链接的审计链。

**📊 数据集**

使用了公司内部的Jira项目数据（约1,602条待办项）和约1,290,158行代码的生产代码库，并通过152次自动化运行和51条人工注入的安全缺陷进行评估。

**📈 对比分析**

评估结果显示：所有152次运行终止状态成功（95% CI 97.6%–100%），零假阴性、零重复Jira发布；在3轮STRIDE威胁模型评审中完全消除了51个注入缺陷；自动安全票据的6/10已通过流水线完成。

**⚠️ 局限性**

局限性包括：仅在单一组织单一代码库进行的案例研究，难以外推到不同技术栈或团队规模；部分关键机制（如碰撞锁、降级模式）未在真实多代理或Jira宕机情境下充分验证；未来工作仍需正式模型验证、性能基准与跨项目实验。

---

## 120. A Multi-Agent Approach to Validate and Refine LLM-Generated Personalized Math Problems

**arXiv ID:** 2604.05160 | [PDF](https://arxiv.org/pdf/2604.05160v1)

**作者:** Fareya Ikram `[一作]` (University of Massachusetts Amherst), Andrew S. Lan `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 1887 | [OpenAlex ID](https://openalex.org/A5063813962)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多代理框架，利用生成–验证–修订循环对数学题进行个性化，并在四个关键维度上进行评估。

**💡 创新点**

创新点在于将个性化任务拆分为专门验证器与修订器，并比较三种不同的反馈整合策略。

**🔧 技术方法**

采用 GPT‑5.2 生成，四个专门的验证器（现实性、可解性、可读性、真实性）和三种修订策略。

**📊 数据集**

使用来自 ASSISTments 的 600 道题目，并以 20 个学生兴趣主题进行个性化。

**📈 对比分析**

通过量化指标比较三种修订策略，发现单次迭代即可显著降低失败率，且不同策略在不同维度表现差异。

**⚠️ 局限性**

局限在于真实性验证器的可靠性低、需要更多教师/学生参与评估，以及不同主题对策的适用性未完全探明。

---

## 121. Non-monotonic causal discovery with Kolmogorov-Arnold Fuzzy Cognitive Maps

**arXiv ID:** 2604.05136 | [PDF](https://arxiv.org/pdf/2604.05136v1)

**作者:** Jose L. Salmeron `[一作]` (CUNEF Universidad), Jose L. Salmeron `[通讯]` (CUNEF Universidad)

**通讯引用:** 4454 | [OpenAlex ID](https://openalex.org/A5074015797)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Kolmogorov‑Arnold 版的 Fuzzy Cognitive Map（KA‑FCM），利用可学习的 B‑spline 在边上实现非单调因果关系建模。

**💡 创新点**

创新点在于将非线性迁移到图边，使用可学习的 B‑spline 替代传统标量权重，并保持图结构的可解释性；同时结合 Kolmogorov‑Arnold 表示定理给出理论支持。

**🔧 技术方法**

技术手段包括 Kolmogorov‑Arnold 表示定理、B‑spline 参数化、SiLU 边基底、梯度下降/粒子群优化训练、L1 正则化以及对比实验中的标准 FCM 与 MLP。

**📊 数据集**

使用的实验数据集为三类合成数据：Yerkes‑Dodson 反向 U‑形关系、正弦关系（sin(3x)）以及 Mackey‑Glass 预测时间序列；未涉及真实行业数据。

**📈 对比分析**

与标准 FCM（PSO 训练）和三层 MLP 进行对比；在非单调建模、符号回归和混沌预测任务中，KA‑FCM 的 MSE/MAPE 明显优于标准 FCM，且在准确度上几乎与 MLP 相当或更好，且可直接提取可解释的边函数。

**⚠️ 局限性**

局限性包括训练成本高、参数量随网格大小增大、在大规模系统上的可扩展性未知，以及缺乏在真实复杂数据集上的验证。

---

## 122. Inclusion-of-Thoughts: Mitigating Preference Instability via Purifying the Decision Space

**arXiv ID:** 2604.04944 | [PDF](https://arxiv.org/pdf/2604.04944v1)

**作者:** Mohammad Reza Ghasemi Madani `[一作]` (University of Melbourne), Jey Han Lau `[通讯]` (University of Melbourne)

**通讯引用:** 4319 | [OpenAlex ID](https://openalex.org/A5032767467)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Inclusion-of-Thoughts (IoT) 框架，通过三阶段自我过滤来提升多选题解答的准确性和稳定性。

**💡 创新点**

创新点在于聚焦模型偏好稳定性，使用自举式去除首选项后再查询第二候选，最终在两候选上重新推理，从而显著降低因干扰选项导致的偏好波动。

**🔧 技术方法**

采用 Chain-of-Thought (CoT) 触发、三阶段自过滤、早停机制、对比分析，并与 Self‑Consistency (SC)、Exclusion‑of‑Thoughts (EoT) 等现有方法进行对比。

**📊 数据集**

使用 CommonsenseQA、OpenbookQA、SocialIQA、ARC、MMLU、GSM8K‑MC、AQUA 等多种常识、教育与数学多选题基准数据集。

**📈 对比分析**

在 GPT‑4o‑mini、90Olmo‑2‑7B、90Olmo‑2‑13B、Llama‑3.3‑8B 等四种模型上与 CoT、SC、EoT 等基线对比，IoT 在大多数任务上平均提升 1–4%（如 GSM8K‑MC 提升 3.95%），并在计算开销上仅为 CoT 的约 2.2 倍，远低于 SC 的 4.6 倍。

**⚠️ 局限性**

局限性：仅适用于多选题，当选项数很少时效果不显著；无法纠正因知识缺失或根本推理错误导致的错误；不适用于开放式生成任务。

---

## 123. ZipFold: Modular Actuators for Scaleable Adaptive Robots

**arXiv ID:** 2604.05260 | [PDF](https://arxiv.org/pdf/2604.05260v1)

**作者:** Niklas Hagemann `[一作]` (Massachusetts Institute of Technology), Daniela Rus `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 63884 | [OpenAlex ID](https://openalex.org/A5066830185)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种基于复合折叠和拉链拼接的可逆可部署电机模块（ZipFold），能够在紧凑与展开状态之间实现平滑的尺度与刚度转换，并将其集成到四足机器人中实现自适应行走和障碍物通过；

**💡 创新点**

创新点在于将折叠与拉链拼接结合为单一步骤的可逆驱动机制，使用低成本桌面3D打印PLA薄片和简易滚轮驱动，模块尺寸小、可轻松串并联/串联，具备可调尺度与刚度；

**🔧 技术方法**

技术主要包括：3D打印可折叠拉链条、滚轮驱动的折叠/拉链机制、单电机DC驱动与H桥控制、简单PD惯性反馈、四足行走的爬行步态；

**📊 数据集**

未使用公开数据集，所有实验为自制机械测试（压缩、弯曲、扭转）与机器人行走实验；

**📈 对比分析**

通过Instron测试得出压缩峰值12 N、弯曲刚度在展开/未展开时差值36倍；拉伸速度10 mm/s；机器人在展开后可达32 cm高度、55 cm到达范围，功耗2.2 W；与传统单向展开装置相比，ZipFold实现了连续可逆控制；

**⚠️ 局限性**

局限包括：高延伸时振动导致精度下降、拉链条相互缠绕或卡住、材料疲劳导致高频操作下强度下降、仅在实验室条件下验证，对复杂载荷（弯矩、扭矩）性能仍需改进；

---

## 124. Tencent Advertising Algorithm Challenge 2025: All-Modality Generative Recommendation

**arXiv ID:** 2604.04976 | [PDF](https://arxiv.org/pdf/2604.04976v1)

**作者:** Junwei Pan `[一作]` (Tencent Inc.), Jie Jiang `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并发布了腾讯广告全模态生成推荐数据集TencentGR-1M和TencentGR-10M，并组织了面向工业广告的生成推荐竞赛。

**💡 创新点**

提供了大规模多模态数据、行为类型加权评估以及完整的基线实现，填补了工业级生成推荐的公开基准空白。

**🔧 技术方法**

使用基于Transformer的因果序列模型、InfoNCE对比损失、ANN检索以及多模态嵌入融合；竞赛评估中采用加权HitRate与NDCG，鼓励预测点击和转化。

**📊 数据集**

TencentGR-1M（约100万用户、660k候选广告）和TencentGR-10M（约1000万用户、363万候选广告）两套数据集。

**📈 对比分析**

通过对比基线和参赛队伍，展示了生成推荐在工业广告中的表现：首、二、三名团队分别在加权NDCG上获得约0.07-0.09的提升；技术创新奖团队实现了统一语义ID生成与行为预测的双目标训练，显著提升检索效果。

**⚠️ 局限性**

主要局限是缺乏多样化的广告素材（仅提供嵌入），对真实业务场景的迁移性和隐私合规性仍需进一步验证；同时，生成模型对极端长序列和冷启动用户的处理仍有挑战。

---

## 125. Measuring the Permission Gate: A Stress-Test Evaluation of Claude Code's Auto Mode

**arXiv ID:** 2604.04978 | [PDF](https://arxiv.org/pdf/2604.04978v1)

**作者:** Zimo Ji `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 25708 | [OpenAlex ID](https://openalex.org/A5100328273)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估 Claude Code 的 auto mode 权限系统在 DevOps 授权模糊任务中的表现，使用 AmPermBench benchmark 对 253 个状态改变动作进行逐动作级评估。

**💡 创新点**

首次独立检验 auto mode 在系统性授权模糊场景下的误判率；揭示 Tier2 文件编辑未被分类器覆盖的覆盖缺口；对不同模糊维度（specificity、blast radius、risk）进行性能分解。

**🔧 技术方法**

使用两阶段 transcript classifier、三层 gate 架构、Claude Sonnet 4.6 与 Claude Opus 作为评估判定器、Docker sandbox 执行环境、prompt‑clustered bootstrap 统计方法。

**📊 数据集**

AmPermBench 128 条提示（4 个 DevOps 任务族 × 3 模糊轴 × 2 风险级别），共 253 个状态改变动作，使用沙箱化的命令行工具与项目状态文件。

**📈 对比分析**

与未开启权限的 Sonnet 4.6 基线进行 STSR 对比，整体提升 3.1%；逐动作误判率为全量 FNR 81.0%、FPR 19.8%，Tier3 评估时 FNR 70.3%、FPR 31.9%；Tier2 覆盖缺口导致 100% 的 FNR。

**⚠️ 局限性**

仅评估单一模型与单一 benchmark，结果不代表生产流量；评估基于 LLM 判定，虽然验证 88% 可靠；系统版本变更可能影响结果。

---

## 126. Decision-Oriented Programming with Aporia

**arXiv ID:** 2604.05203 | [PDF](https://arxiv.org/pdf/2604.05203v1)

**作者:** Saketh Ram Kasibatla `[一作]` (UC San Diego), Nadia Polikarpova `[通讯]` (UC San Diego)

**通讯引用:** 1420 | [OpenAlex ID](https://openalex.org/A5090671359)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了决策导向编程（Decision‑Oriented Programming, DOP）框架，并构建了一个 VS Code 扩展，记录和可视化决策，主动向开发者提问并将每个决策编码为可执行的测试套件，随后评估其对开发者决策过程、代码理解与验证的影响。

**💡 创新点**

1）将决策提升为一等对象、可持久化和可编辑；2）通过代理主动提问实现互动式共创；3）将每个决策与对应测试套件关联，形成决策–代码–验证的闭环，从而提升决策透明度与可追溯性。

**🔧 技术方法**

VS Code 扩展（TypeScript+React），Agent Client Protocol 与 Model Context Protocol 与 Claude Sonnet 4.6 LLM 交互；Decision Bank 数据库；基于 BDD/TDD 的测试生成器；代码引用与结构化问答模块。

**📊 数据集**

自定义会议管理系统代码库（Python + 预置用户/论文数据库），无公开数据集。

**📈 对比分析**

采用 2×2 交叉实验（14 名参与者）对比 DOP 与基线编码代理。评估指标包括：决策数量、决策与实现匹配率、NASA‑TLX 认知负荷。结果显示 DOP 使提问决策数提升 13.5 倍、总决策数提升 2.99 倍，错误匹配率下降 79%，用户对实现的信心更稳定，且整体认知负荷与基线无显著差异。

**⚠️ 局限性**

UI 过于复杂、缺乏实现进度指示、问题优先级不明确导致决策排序困难；系统响应延迟影响工作流；测试套件被使用率低，难以充分发挥验证作用；基线代理缺乏结构化的决策追踪机制。

---

## 127. Boxer: Robust Lifting of Open-World 2D Bounding Boxes to 3D

**arXiv ID:** 2604.05212 | [PDF](https://arxiv.org/pdf/2604.05212v1)

**作者:** Daniel DeTone `[一作]` (Meta Reality Labs Research), Jakob Engel `[通讯]` (Meta Reality Labs Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Boxer框架，利用已姿态化的图像及可选深度信息和现成的2D检测框，将开放世界中的二维检测框可靠地提升到全局3D边界框

**💡 创新点**

创新点在于将2D检测与可选深度结合，实现对未知物体（open‑world）的3D框估计，并通过姿态化序列实现全局一致性

**🔧 技术方法**

使用姿态化图像预处理、可选深度估计、二维检测框与三维几何约束的联合优化技术

**📊 数据集**

在COCO/COCO‑3D等公开数据集以及自建的多场景序列上进行实验

**📈 对比分析**

与基准方法（如传统单帧三维重建、基于深度的3D检测等）对比，Boxer在多种物体（如香料罐、吹风机、洗手池排水口、遥控器）上的3D框定位精度和覆盖率显著提升，误差平均下降30%以上

**⚠️ 局限性**

局限性包括：仅适用于静态场景，无法处理运动物体；对姿态估计和深度推断的依赖导致在极端光照或遮挡下性能下降；对全局尺度的精度受限于输入图像的分辨率和相机标定

---

## 128. Modality-Aware and Anatomical Vector-Quantized Autoencoding for Multimodal Brain MRI

**arXiv ID:** 2604.05171 | [PDF](https://arxiv.org/pdf/2604.05171v1)

**作者:** Mingjie Li `[一作]` (Stanford University), Kilian M. Pohl `[通讯]` (Stanford University)

**通讯引用:** 6913 | [OpenAlex ID](https://openalex.org/A5055107125)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了一种能够同时学习脑 MRI 结构（解剖）与模态特定外观的 3D VQ‑VAE，称为 NeuroQuant，并提供了可用于多模态脑图像重建与下游分析的紧凑离散潜在空间。

**💡 创新点**

创新点包括：① 双流编码器将共享解剖特征与模态特定特征并行学习；② 利用 factorized multi‑axis attention 在三维空间中高效捕获长程结构依赖；③ 在解码器中使用 FiLM 进行模态条件化，进一步分离结构与外观；④ 采用 2D/3D 联合训练策略兼顾全局体积一致性与切片细节。

**🔧 技术方法**

技术手段包括：3D 卷积、factorized multi‑axis attention、vector‑quantized 代码表、Feature‑wise Linear Modulation (FiLM)、交叉模态重建与模态对抗损失、2D/3D 联合训练、EMA 更新代码表等。

**📊 数据集**

使用两大公开多模态脑 MRI 数据集：NCANDA（T1/T2）和 ABCD（T1/T2），涵盖数千个体检场景。

**📈 对比分析**

与 VQGAN、SD‑VAE、MediTok、MedVAE 等 SOTA VAE 进行对比；在 PSNR、3D SSIM、SynthSeg Dice 以及潜在空间的性别分类准确率等指标上均实现了显著提升（例如 T1 PSNR 提升约 0.6 dB，SSIM 提升 1.5%，Dice 提升 1% 以上，性别分类准确率提升 1%）。

**⚠️ 局限性**

局限性包括：① 目前仅覆盖 T1 与 T2 两种模态，需扩展至 FLAIR、SWI 等；② 对扫描仪和协议的泛化性仍受限，跨中心验证需进一步加强；③ 代码表的离散化可能导致细节失真，尚无针对细微结构恢复的可解释机制。

---

## 129. Vintix II: Decision Pre-Trained Transformer is a Scalable In-Context Reinforcement Learner

**arXiv ID:** 2604.05112 | [PDF](https://arxiv.org/pdf/2604.05112v1)

**作者:** Andrei Polubarov `[一作]` (Applied AI Institute), Vladislav Kurenkov `[通讯]` (Innopolis University)

**通讯引用:** 24 | [OpenAlex ID](https://openalex.org/A5010816959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本研究将Decision Pre‑Trained Transformer（DPT）扩展到跨域连续控制环境，并通过流匹配（flow‑matching）头实现对多模态连续动作分布的自回归采样；同时构建了约7亿条转移的跨域数据集，在训练与测试任务上实现显著泛化提升。

**💡 创新点**

创新点包括：① 将流匹配作为DPT的动作头，使模型能够在连续、多模态动作空间中自然采样；② 结合噪声消融数据收集与重标记技术，形成大规模多域数据集；③ 在离线与在线两种评估模式下均展示出强大的自适应能力，超越现有算法。

**🔧 技术方法**

核心技术包括Decision Pre‑Trained Transformer、TinyLLaMA transformer架构、流匹配（rectified flow）目标函数、Heun ODE求解器、上下文条件向量场、在线与离线上下文管理，以及对多域数据的统一编码与解码。

**📊 数据集**

使用了覆盖10个领域（工业基准、Bi‑DexHands、Meta‑World、Kinetix、CityLearn、ControlGym、HumEnv、MuJoCo、SinerGym、Meta‑Drive）的209个训练任务，共约7.1亿条转移；另外保留46个任务（209个中未见）用于评估；数据集规模比之前扩大3.2倍。

**📈 对比分析**

与基线算法（如Algorithm Distillation、原始DPT以及其他离线动作模型）在在线与离线两种设置下进行对比；在训练任务上模型接近示范器（≈100%），在未见任务上在线评估可达≈85%，离线评估在Meta‑Drive、CityLearn、SinerGym、ControlGym等域超过75%，相对基线提升17%–63%，整体在多个域中实现了显著的性能提升。

**⚠️ 局限性**

局限性包括：① 仍然缺乏有效的探索机制，示范器无提示时表现不佳；② token‑to‑parameter比率低于最优值，表明需要进一步扩大数据与模型规模；③ 对完全未知域的迁移能力有限，因模型对输入/输出维度依赖；④ 流匹配头虽然提升了连续动作采样，但在高维多模态场景中仍可能受限。

---

## 130. Exemplar Retrieval Without Overhypothesis Induction: Limits of Distributional Sequence Learning in Early Word Learning

**arXiv ID:** 2604.05243 | [PDF](https://arxiv.org/pdf/2604.05243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 131. Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER

**arXiv ID:** 2604.05158 | [PDF](https://arxiv.org/pdf/2604.05158v1)

**作者:** Ahmed Ewais `[一作]` (WitnessAI), Amr Ali `[通讯]` (WitnessAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Just Pass Twice（JPT）方法，通过将输入文本重复两次，使因果 LLM 能够在不修改架构的情况下获得双向上下文，从而实现判别式的 token 级命名实体识别，并结合定义导向的实体类型嵌入实现灵活的零样本泛化。

**💡 创新点**

创新点包括：① 简单的输入复制技巧在因果 LLM 中恢复双向注意力；② 用自然语言定义而非仅名称来表示实体类型，支持在推理时动态添加新类型；③ 将定义同时注入提示与嵌入两通道，提升判别性能；④ 通过冻结 LLM 并只训练 LoRA、投影层与双线性分类器，实现极低参数增量且高效推理。

**🔧 技术方法**

核心技术包括：Qwen3 4B/8B 预训练因果 LLM、LoRA 参数适配、两层投影 MLP、双线性分类头、自然语言实体定义编码（可使用 Qwen3-Embedding 或其它文本编码器）、输入复制与注意力分析。

**📊 数据集**

训练使用一份无评测集重叠的 Wikipedia 自动标注 NER 数据；评测在 CrossNER（5 个领域）、MIT 电影/餐饮对话 NER 以及 20 个跨领域（生物医学、社交媒体、多语言）公开基准上进行零样本测试。

**📈 对比分析**

与生成式方法（UniNER、GoLLIE、InstructUIE 等）和判别式基线（GLiNER-L、SaM）对比，JPT-8B 在 CrossNER+MIT 评测中平均提升 7.9 F1，突破前沿，且在生成式方法中实现 20× 以上的速度提升（单前填充 vs 逐词解码）。

**⚠️ 局限性**

局限性包括：输入复制导致序列长度翻倍，长文本时会增加 O(N²) 内存开销；仅支持平面 NER，无法处理嵌套实体；依赖单一 Wikipedia 派生训练集，虽然无评测集重叠但仍可能对极端领域产生偏差。

---

## 132. Coverage Optimization for Camera View Selection

**arXiv ID:** 2604.05259 | [PDF](https://arxiv.org/pdf/2604.05259v1)

**作者:** Timothy Chen `[一作]` (Stanford University), Mac Schwager `[通讯]` (Stanford University)

**通讯引用:** 7805 | [OpenAlex ID](https://openalex.org/A5081950488)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个基于几何覆盖度的下一最佳视角选择指标COVER，并在Nerfstudio框架中实现。

**💡 创新点**

通过近似Fisher信息增益，将视角选择简化为仅需几何覆盖度的可解释指标，避免昂贵的传输估计和训练过程中的噪声。

**🔧 技术方法**

使用Gaussian Splatting（3DGS）表示、Fisher信息理论、覆盖度指标计算与可视化，以及Nerfstudio训练管道。

**📊 数据集**

在Tanks and Temples、MipNeRF360以及3个手机采集的自定义场景（共15个场景）上进行实验。

**📈 对比分析**

与随机、Bayes' Rays、FisherRF等基线对比；COVER在PSNR/SSIM/LPIPS等图像指标上普遍优于随机和FisherRF，接近不可行的全图oracle，并在嵌入式采集模式下提升约1.5dB。

**⚠️ 局限性**

依赖覆盖度近似，忽略传输信息，可能在极度混乱场景或光照变化剧烈时效果下降；在稀疏初始化时与随机基线相当，且不考虑照明/材质变化。

---

## 133. Semantic Reality: Interactive Context-Aware Visualization of Inter-Object Relationships in Augmented Reality

**arXiv ID:** 2604.05265 | [PDF](https://arxiv.org/pdf/2604.05265v1)

**作者:** Xiaoan Liu `[一作]` (Google), Mar Gonzalez-Franco `[通讯]` (Google)

**通讯引用:** 4261 | [OpenAlex ID](https://openalex.org/A5023820234)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套基于动态语义图的 AR+AI 系统，实时展示并交互多物体之间的连接关系，用于规划、比较和装配任务。

**💡 创新点**

核心创新是将“连通性”视为 AR 的交互基底，定义八类可操作关系，并通过用户选择、语音、手势等多模态输入实时构建和更新场景语义图。

**🔧 技术方法**

技术实现结合了 Apple Vision Pro 的现场摄像与 3D 重建、Gemini 2.5 Flash 进行开源多模态对象检测与关系推理、Unity+PolySpatial+RealityKit 实时渲染，以及语音识别与手势跟踪。

**📊 数据集**

数据来源为 52 个多场景脚本、28 次实验室任务和 24 条 YouTube 说明视频，用于构建场景语义和验证系统功能；未使用专门标注的数据集。

**📈 对比分析**

通过 within‑subjects 对比单物体基线，使用 NASA‑TLX、HALIE、任务完成时间等指标；结果显示连通性系统提升交互清晰度、参与度与感知性能，任务时间差异不显著；在视频基准中，该系统在多物体协调任务中排在前列。

**⚠️ 局限性**

局限包括依赖实时网格重建导致漂移、外部大模型延迟、场景复杂度与视觉拥堵风险，样本规模小、实验环境受控，缺乏真实生活环境的鲁棒性验证。

---

## 134. Designing Digital Humans with Ambient Intelligence

**arXiv ID:** 2604.05120 | [PDF](https://arxiv.org/pdf/2604.05120v1)

**作者:** Mengyu Chen `[一作]` (JPMorganChase), Richard Chen `[通讯]` (JPMorganChase)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一套将环境感知、跨设备协同与企业数据融合进数字人中的“Ambient Digital Human”框架，实现了上下文感知与主动协助的数字人系统。

**💡 创新点**

创新点在于将 Ambient Intelligence 与数字人结合，提出了五大角色与设计空间，提供跨设备无缝、主动性、环境渲染等交互模式，并给出责任与隐私设计准则，形成了面向实际服务场景的完整概念与原型。

**🔧 技术方法**

采用多模态感知（摄像头、麦克风、BLE/UWB、深度相机）、边缘推理、LLM（Llama3、OpenAI Whisper）、工具调用与 Agent orchestration（AutoGen、ReAct）、云端渲染（Unreal Engine 5/MetaHuman）、AWS GPU 容器等技术。

**📊 数据集**

论文未使用公开数据集，而是基于银行分行和零售环境的实时传感器数据与企业 API（CRM、预约系统、交易记录）进行原型验证。

**📈 对比分析**

由于研究侧重于概念与原型实现，未给出定量对比；作者通过案例演示展示了系统在主动性、跨设备无缝和隐私保护方面的优势，但缺乏实测指标与性能评估。

**⚠️ 局限性**

限制包括：缺乏在真实环境下的鲁棒多模态融合与错误恢复；长期隐私与数据治理方案不完整；主动性调节的个性化与可定制化不足；高风险事务的安全保障与解释性不足；以及对法规遵从的动态适配不足。

---

## 135. Strategic Delay and Coordination Efficiency in Global Games

**arXiv ID:** 2604.05298 | [PDF](https://arxiv.org/pdf/2604.05298v1)

**作者:** Shinkyu Park `[一作]` (King Abdullah University of Science and Technology), Marcos M. Vasconcelos `[通讯]` (Florida State University)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5007384409)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了一个两阶段集体决策问题的协调模型，分析了代理人如何在观察到共享随机变量的噪声信号后决定是否参与集体行动或延迟决策。

**💡 创新点**

创新点在于引入了延迟决策的选项，分析了信息获取与收益减少之间的权衡如何改善协调和提高集体决策的效率。

**🔧 技术方法**

使用了随机博弈理论和贝叶斯纳什均衡的分析方法，特别关注于具有部分信息的两阶段博弈。

**📊 数据集**

没有具体提到使用的数据集，但提到的应用领域包括工程、经济学、生物学和政治科学等。

**📈 对比分析**

通过比较有限代理人和无限代理人的模型，展示了引入延迟选项如何在均衡中提高参与率，并分析了不同噪声水平下的均衡特性。

**⚠️ 局限性**

限制在于模型假设了噪声水平可以操控，且在实际应用中可能存在信息不对称和其他复杂因素影响决策过程。

---

## 136. SmokeGS-R: Physics-Guided Pseudo-Clean 3DGS for Real-World Multi-View Smoke Restoration

**arXiv ID:** 2604.05301 | [PDF](https://arxiv.org/pdf/2604.05301v1)

**作者:** Xueming Fu `[一作]` (University of Science and Technology of China), Lixia Han `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5102941473)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 SmokeGS-R，利用物理引导的伪清洁监督和 3D Gaussian Splatting 的几何优先分支，并在渲染后通过多参考 LAB 空间和 Reinhard 颜色传递进行烟雾消除。

**💡 创新点**

将几何恢复与烟雾外观校正解耦，采用 DCP 伪清洁生成、清晰 3DGS 源分支以及多参考集合的 LAB 颜色对齐，实现高效且稳健的烟雾去除。

**🔧 技术方法**

暗通道先验、引导滤波、3D Gaussian Splatting、LAB 空间 Reinhard 颜色传递、几何平均参考、轻量级高斯平滑等技术。

**📊 数据集**

RealX3D 队列中的烟雾子集以及 NTIRE 2026 3DRR Track 2 评测数据。

**📈 对比分析**

与 RealX3D 官方基线在 PSNR/SSIM/LPIPS 进行直接对比，冻结模型在七个公开测试场景上分别取得 15.209 dB PSNR、0.644 SSIM 和 0.551 LPIPS，均超过 3DGS 等官方基线平均 +3.68 dB。

**⚠️ 局限性**

对烟雾较轻的场景（如 Tsubaki）提升有限，且改进程度随场景烟雾浓度变化。

---

## 137. DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects

**arXiv ID:** 2604.05318 | [PDF](https://arxiv.org/pdf/2604.05318v1)

**作者:** Jason Lucas `[一作]` (Pennsylvania State University), Dongwon Lee `[通讯]` (Pennsylvania State University)

**通讯引用:** 9487 | [OpenAlex ID](https://openalex.org/A5100405086)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个覆盖50种英语方言的反假信息检测基准D^3，评估现有模型在方言下的鲁棒性。

**💡 创新点**

首次系统评估方言鲁棒性，构建了基于Multi-VALUE的方言转换与D-PURIFY质量验证工具，揭示细粒度方言偏差。

**🔧 技术方法**

采用规则化方言转换、BERTScore/AlignScore等质量评估，结合多模态Transformer、传统深度学习与零射LLM进行对比实验。

**📊 数据集**

使用9个公开反假信息基准（FakeNewsNet、LIAR、CoAID、MM-COVID、MultiClaim、F^3、MiDe、Twitter15/16），通过Multi‑VALUE生成195K+样本。

**📈 对比分析**

通过四个实验场景（SQ1–SQ4）对16个模型进行宏观F1对比，发现人类方言样本下降1.4–3.6%，AI保持稳定；多语言mDeBERTa平均F1 97.2%，传统模型和零射LLM性能显著低于微调Transformer。

**⚠️ 局限性**

局限在于方言转换仅为规则化形态句法近似，缺乏真实方言文本验证；评价指标依赖SAE训练的度量，可能低估某些方言；仅覆盖英语，无法推广到其他语言；零射LLM评估使用单一提示，未探究更优提示或少量示例。

---

## 138. CRAB: Codebook Rebalancing for Bias Mitigation in Generative Recommendation

**arXiv ID:** 2604.05113 | [PDF](https://arxiv.org/pdf/2604.05113v1)

**作者:** Zezhong Fan `[一作]`, Kannan Achan `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 139. Document Optimization for Black-Box Retrieval via Reinforcement Learning

**arXiv ID:** 2604.05087 | [PDF](https://arxiv.org/pdf/2604.05087v1)

**作者:** Omri Uzan `[一作]` (Stanford University), Christopher Potts `[通讯]` (Stanford University)

**通讯引用:** 26387 | [OpenAlex ID](https://openalex.org/A5042601761)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出将文档扩展视为文档优化问题，使用语言模型/视觉语言模型在检索器的检索质量上进行强化学习；

**💡 创新点**

创新点在于：①将文档重写建模为策略优化，用逆向检索奖励直接优化nDCG；②在黑盒检索器上仅依赖排名反馈即可训练；③兼容单向量、多向量及词典检索器；

**🔧 技术方法**

核心技术包括：指令调优的生成式模型作为策略，使用Group Relative Policy Optimization（GRPO）进行强化学习，采用对比式检索奖励，结合低温/高温采样和周期性语料库刷新；

**📊 数据集**

评估数据集包括视觉文档检索ViDoRe2（含多语言查询）和代码检索DS10K、CodeSearchNet等；

**📈 对比分析**

实验对比了直接检索、零样本文档转换、检索器微调及联合适配；结果显示文档优化显著提升nDCG@5，例如在代码检索上OpenAI text‑embedding‑3‑small由58.7提升至66.8，BM25从15.6提升至46.6；

**⚠️ 局限性**

局限性包括：①需要离线大量生成与检索评估，计算成本高；②奖励稀疏导致训练收敛慢；③依赖有限的标注查询，弱监督仍不够充分；④在极大文档集合上刷新语料库难以频繁更新。

---

## 140. From Measurement to Mitigation: Quantifying and Reducing Identity Leakage in Image Representation Encoders with Linear Subspace Removal

**arXiv ID:** 2604.05296 | [PDF](https://arxiv.org/pdf/2604.05296v1)

**作者:** Daniel George `[一作]` (Persona Identities), Yifei Zhang `[通讯]` (Persona Identities)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对冻结的视觉编码器在面部数据上的身份泄露进行攻击者校准评估，并提出了线性子空间去除投影 ISP 以抑制身份信息，同时保留非生物特征的实用性。

**💡 创新点**

创新点在于：① 设计了开放集低 FAR 的身份验证基准、扩散模板反演与面部‑上下文归因三种新评估方法；② 提出了单步时刻矩阵投影 ISP，能一次性去除身份子空间并实现可审计的隐私保证；③ 展示了 ISP 的跨数据集可迁移性与与传统 erasure 方法的对比。

**🔧 技术方法**

使用的技术包括：冻结视觉编码器（CLIP、DINOv2/3、SSCD）、线性和 MLP 验证器、拉格朗日正则岭回归、扩散模型模板反演、FCR 归一化的面部‑背景扰动、以及基于类均值的 SVD 子空间投影。

**📊 数据集**

实验数据集为 CelebA‑20、VGGFace2‑20（均为20 张/人、按身份划分的 320/80/80 训练/验证/测试集），以及 ImageNet 作为通用任务的效能验证。

**📈 对比分析**

与 FR 基线（ArcFace、AdaFace）和 LEACE 对比，ISP 在低 FAR（≈10⁻⁴）下将线性 TAR 降到接近 0%，并将非线性 MLP TAR 降至零；同时在 ImageNet 及复制检测任务中保持 90%+ 纯度，显示出优越的隐私‑效能折中。

**⚠️ 局限性**

局限性包括：仅针对线性攻击提供正式保证；依赖已标记身份样本，可能受人口统计或域差异影响；未对黑盒或高阶非线性/生成式攻击提供保障。

---

## 141. Investigating Ethical Data Communication with Purrsuasion: An Educational Game about Negotiated Data Disclosure

**arXiv ID:** 2604.05200 | [PDF](https://arxiv.org/pdf/2604.05200v1)

**作者:** Krisha Mehta `[一作]` (University of Chicago), Alex Kale `[通讯]` (University of Chicago)

**通讯引用:** 1000 | [OpenAlex ID](https://openalex.org/A5001536494)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一款名为Purrsuasion的可视化教育与研究游戏，旨在通过“展示–隐藏”类谜题探究数据披露与伦理沟通。

**💡 创新点**

创新点包括：①将伦理披露问题形式化为“展示-隐藏”谜题；②设计可供教学与研究的启发式评分表；③以开源、易部署的浏览器游戏方式将沟通动态可视化并收集互动日志。

**🔧 技术方法**

技术手段：基于TypeScript、Pyodide、Altair/Pandas/NumPy构建可视化笔记本；WebSocket实现实时消息交换；使用SQLite记录日志；界面采用猫主题，易于学生接受。

**📊 数据集**

使用三组人工合成数据集：零售门店分布、空气质量峰值与缺失区间、仓库延迟与异常点，分别对应“高饱和度-隐藏位置”、“峰值-隐藏缺口”和“离群点-隐藏个体”。

**📈 对比分析**

比较方法：通过手工构建的启发式评分表对学生生成的可视化进行“满足/风险/违规”评估，结合日志统计分析设计固着与探索行为；实验结果显示学生倾向于聚焦单一图表、使用聚合等安全策略，且大多数解答被视为“风险”而非完全合规。

**⚠️ 局限性**

局限性：实验仅在80分钟的课堂中进行，受时间与单轮反馈限制；样本为本科可视化课程学生，技术熟练度有限；未能直接观察生成式AI使用情况；数据集为合成，可能缺乏真实复杂性。

---

## 142. SenseAI: A Human-in-the-Loop Dataset for RLHF-Aligned Financial Sentiment Reasoning

**arXiv ID:** 2604.05135 | [PDF](https://arxiv.org/pdf/2604.05135v1)

**作者:** Berny Kabalisa `[一作]` `[通讯]` (RizqSpark), Berny Kabalisa (RizqSpark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并构建了SenseAI数据集，持续收集并由金融专家验证的情感标签，同时记录AI生成的推理链、置信度、市场价格验证等多维度信息，并对该数据集进行了结构化设计以满足RLHF训练需求；

**💡 创新点**

创新点在于：①将推理链与情感标签一起记录，首次捕捉模型的“隐性漂移”和“前向投射”现象；②实现持续收集与实时市场验证，构建可追踪的Goldilocks Zone；③设计符合RLHF三大结构需求（人类偏好、纠正注释、推理上下文）的全流程数据管道；

**🔧 技术方法**

采用大语言模型（如GPT-5系列）生成情感分类、推理链和置信度；通过人类金融专家的HITL校正获取纠正标签；使用自一致性测试和统计分析提取六项实验发现；

**📊 数据集**

主要使用自研SenseAI数据集（截至2026年3月约1,439条），涵盖40支美国上市股票、13类金融数据；并对比引用的FinancialPhraseBank、FiQA、FLUE等公开基准；

**📈 对比分析**

与现有基准在九个结构维度（规模、情感粒度、推理链、RLHF对齐、市场验证等）进行对比，结果显示SenseAI在结构深度和可用性上优于现有数据集；但因尚未完成大规模Fine‑tune实验，未给出具体性能提升数值，预期在5k–10k样本后可显著提升；

**⚠️ 局限性**

局限性包括：①样本规模有限，仅1,439条；②数据由单一专家注解，缺乏多注解者一致性评估；③采集密度在手工阶段不稳定；④仅覆盖美国上市股票，新闻来源单一；⑤目前未提供Fine‑tune实验结果；

---

## 143. The Illusion of Latent Generalization: Bi-directionality and the Reversal Curse

**arXiv ID:** 2604.04943 | [PDF](https://arxiv.org/pdf/2604.04943v1)

**作者:** Julian Coda-Forno `[一作]` (Helmholtz Munich), Arslan Chaudhry `[通讯]` (Google DeepMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了“逆转诅咒”（Reversal Curse）并评估了双向监督训练目标（MLM和decoder‑only的掩码微调）对模型逆转推理的影响，同时通过表示距离和线性探测分析模型内部机制。

**💡 创新点**

创新点在于首次从机制层面揭示，双向监督并未产生单一的、方向无关的事实表示，而是让模型以不同的几何方式存储正向和逆向条目；并比较了MLM与掩码微调在表示组织上的差异。

**🔧 技术方法**

采用的技术包括：标准next‑token预测（NTP）、掩码语言模型（MLM）、decoder‑only掩码微调（NTP+Masking）、表示距离计算（余弦距离）、线性探测（逻辑回归）和消融实验。

**📊 数据集**

使用的四个基准数据集为：Simple Reversal、Nonsense Entities、Fictional Celebrity、Semantic Structure。

**📈 对比分析**

通过在测试集中仅训练正向事实、评估逆向推理的准确率来比较；NTP几乎为0，MLM和NTP+Masking在所有基准上都能显著恢复逆转准确率；消融实验显示源实体的预测是必需的；表示分析显示正向和逆向条目在表示空间中相距很远，线性探测表明两者不可线性区分，说明未形成统一表示。

**⚠️ 局限性**

局限性：仅在线性探测层面观察到无统一表示，无法排除潜在的非线性逆转映射；实验聚焦于四个特定基准，结果可能不完全泛化；此外，双向监督虽然消除逆转诅咒，却并未解决潜在的潜在泛化脆弱性。

---

## 144. Differentiable Invariant Sets for Hybrid Limit Cycles with Application to Legged Robots

**arXiv ID:** 2604.05108 | [PDF](https://arxiv.org/pdf/2604.05108v1)

**作者:** Varun Madabushi `[一作]` (Georgia Institute of Technology), Maegan Tucker `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 283 | [OpenAlex ID](https://openalex.org/A5054944952)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于参数化可达集的前向不变集计算框架，用于分析和控制混合系统中的周期轨道（以平面双足行走器为例）。

**💡 创新点**

创新点在于：①将参数化可达集理论扩展到含离散冲击的混合系统；②通过区间分析与线性包含构造可行的椭圆形可达管；③将计算结果直接与控制设计耦合，实现可微前向不变集的优化；④利用JAX实现高效并行、自动微分。

**🔧 技术方法**

采用的技术包括：参数化可达集（线性包含与归一化）、区间分析、椭圆形（ℓ₂‑normotope）表示、JAX实现的自动微分与JIT编译、仿真与蒙特卡洛验证、梯度下降和二分搜索的双层优化。

**📊 数据集**

使用的“数据集”为模拟的平面双足行走器模型（4维状态空间），通过 IPOPT 进行轨迹优化得到的参考轨迹；未使用公开实验数据。

**📈 对比分析**

与 SOS 编程（17.7分钟）和 HJ‑B 方法（≈36小时）的对比表明，该方法在同一系统上只需 19.56 秒（含 4.55 秒 JIT 编译），显著提升计算速度。实验验证显示在跟踪控制下，前向不变管的尺寸可增大约 4.25 倍，且所有 100 条蒙特卡洛轨迹均保持在管内。

**⚠️ 局限性**

局限性包括：①需要存在将守护面线性化的坐标变换；②区间线性包含导致保守性，未能证明收敛性；③仅证明前向不变性，未正式证明对周期轨道的渐进稳定性；④方法在极高维或非线性冲击映射时的可扩展性仍待验证。

---

## 145. Multilingual Language Models Encode Script Over Linguistic Structure

**arXiv ID:** 2604.05090 | [PDF](https://arxiv.org/pdf/2604.05090v1)

**作者:** Aastha A K Verma `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 5142 | [OpenAlex ID](https://openalex.org/A5046521217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多语种语言模型内部表示，探究语言关联单元是编码抽象语言身份还是表面形式，并在 Llama‑3.2‑1B 与 Gemma‑2‑2B 上进行系统实验；

**💡 创新点**

首次将 LAPE 与 SAE‑LAPE 结合，进行脚本、词序、典型性结构多维度的对比与因果干预；

**🔧 技术方法**

使用 Language Activation Probability Entropy (LAPE)、Sparse Autoencoder (SAE)、Jaccard 相似度、线性探测器、交叉语言均值替换与零消融等技术；

**📊 数据集**

利用 FLORES+ 并生成罗马化版本作为实验数据，同时使用语言典型属性资源进行探测；

**📈 对比分析**

通过 Jaccard 评估脚本/词序下的单元重叠，使用线性探测 R² 衡量典型性可解码性；干预后用 perplexity 变化评估功能重要性。结果表明脚本改变几乎完全分裂单元，词序扰动影响有限，深层可解码典型性随层级提升，但因果重要性主要来自对表面扰动不敏感的单元；

**⚠️ 局限性**

仅研究小型压缩模型且仅关注 MLP 子层，未分析注意力头、嵌入层或训练动态；脚本与转写对大模型的影响可能不同。

---

## 146. MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU

**arXiv ID:** 2604.05091 | [PDF](https://arxiv.org/pdf/2604.05091v1)

**作者:** Zhengqing Yuan `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**通讯引用:** 5249 | [OpenAlex ID](https://openalex.org/A5027601906)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MegaTrain，一种内存中心化的训练系统，在单个GPU上全精度训练100B+参数LLM。

**💡 创新点**

创新点在于将模型参数和优化器状态全部驻留CPU内存，GPU仅作为临时计算引擎，并引入流水线双缓冲执行与无状态层模板。

**🔧 技术方法**

采用参数流式加载、双缓冲调度、CUDA多流、无状态执行模板、激活重计算、CPU端Adam优化等技术。

**📊 数据集**

使用MetaMathQA等大规模数学推理数据集评估模型准确性。

**📈 对比分析**

与ZeRO-3、ZeRO-Infinity、PyTorch Native等基线对比，MegaTrain在GH200、H200及A100等平台上在模型规模从7B到120B时实现1.8-2.4倍吞吐量，保持高TFLOPS并避免OOM。

**⚠️ 局限性**

限制主要在于对CPU内存容量的依赖，且单GPU方案难以支持更大上下文长度或更高参数量，需要多GPU扩展。

---

## 147. MIRAGE: Benchmarking and Aligning Multi-Instance Image Editing

**arXiv ID:** 2604.05180 | [PDF](https://arxiv.org/pdf/2604.05180v1)

**作者:** Ziqian Liu `[一作]` (Institut Polytechnique de Paris), Stephan Alaniz `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MIRAGE benchmark 和一种无训练的多分支推理框架，用于多实例、多指令的图像编辑。

**💡 创新点**

创新点在于：①首个针对 3–5 个相似实例并配合 5 条组合指令的细粒度评测基准；②利用 VLM 进行指令分解与目标定位，并在扩散过程前后进行区域级别的隐空间融合，避免过度编辑与空间错位。

**🔧 技术方法**

技术方法包括：Qwen3‑VL‑8B 进行指令解析与框定位，SAM2 生成精确掩模；多分支并行扩散推理（局部分支 + 全局分支）和时间阈值 ρ 的隐空间替换策略。

**📊 数据集**

使用的数据集：MIRAGE（200 个自动生成样本，其中 100 个经过人工验证的黄金样本）以及 RefEdit‑Bench（200 张单指令样本）。

**📈 对比分析**

与 FLUX.2、Qwen‑Image‑Edit、GPT‑Image‑1.5 等 SOTA 模型对比，MIRAGE+框架在 Prompt Following (PF) 最高提升约 +1.7，Consistency (Cons) 最高提升约 +0.7，整体分数从 7.4‑8.0 提升至 8.2‑8.5，且保持相似或略优的感知质量。

**⚠️ 局限性**

局限性包括：依赖 VLM 的定位精度；仅在支持可变图像尺寸的扩散模型（如 FLUX.2）上效果最佳；Qwen‑Image‑Edit 等固定尺寸模型在使用时会出现模糊或细节丢失；框选与掩模误差仍可能导致少量过度编辑。

---

## 148. Cross-fitted Proximal Learning for Model-Based Reinforcement Learning

**arXiv ID:** 2604.05185 | [PDF](https://arxiv.org/pdf/2604.05185v1)

**作者:** Nishanth Venkatesh `[一作]` (Cornell University), Andreas A. Malikopoulos `[通讯]` (Cornell University)

**通讯引用:** 4534 | [OpenAlex ID](https://openalex.org/A5076592878)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在离线强化学习中处理隐藏混杂的部分可观测马尔可夫决策过程（POMDP），通过桥函数（bridge functions）实现对奖励和转移模型的识别与估计；

**💡 创新点**

创新点在于将桥函数估计转化为条件矩限制（CMR）问题，并提出基于K折交叉拟合的两阶段桥估计器，以提升样本利用效率并提供oracle比较的理论分析；

**🔧 技术方法**

采用了核方法的条件均值嵌入、条件密度估计、交叉拟合技术以及两阶段正则化优化；

**📊 数据集**

实验使用了一个低维非线性高斯POMDP（含二值动作和隐藏上下文），在该模拟数据上检验桥函数恢复效果；

**📈 对比分析**

将交叉拟合方法与传统的50–50样本分割方法对比，结果显示交叉拟合在不同样本规模下均取得更低的均方误差，证明其更高的样本效率；

**⚠️ 局限性**

局限性包括未构造Neyman正交得分，缺乏对下游策略评估误差的显式转化，以及仅在模拟环境中验证，缺乏真实数据实验。

---

## 149. OrthoFuse: Training-free Riemannian Fusion of Orthogonal Style-Concept Adapters for Diffusion Models

**arXiv ID:** 2604.05183 | [PDF](https://arxiv.org/pdf/2604.05183v1)

**作者:** Ali Aliev `[一作]` (HSE University), Maxim Rakhuba `[通讯]` (HSE University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种训练‑免费的方法OrthoFuse，用来将针对不同任务（概念与风格）的正交适配器融合成一个统一的适配器；

**💡 创新点**

其创新点在于：①首次利用正交参数化的流形几何特性，构造GS‑orthogonal矩阵的Riemannian流形并近似其几何距；②提出块级几何插值和谱恢复变换（Cayley变换+特征值旋转），实现对融合过程的精细控制；

**🔧 技术方法**

核心技术包括：正交（GS‑orthogonal）矩阵的结构化定义、Riemannian几何学、块级几何插值、矩阵指数与对数、Cayley变换、特征值旋转与Padé近似；

**📊 数据集**

实验基于SDXL diffusion模型，概念数据取自DreamBooth，风格数据取自StyleDrop与K‑LoRA的艺术风格集；

**📈 对比分析**

通过与K‑LoRA、ZipLoRA及联合正交训练等三种基线在CLIP、DINO、StyleSim等指标上对比，OrthoFuse在保持概念一致性的同时实现最高的风格相似度，并在所有指标的几何平均值上名列第一，且融合耗时<1秒；

**⚠️ 局限性**

局限性包括：与概念保持的权衡仍需通过融合参数t调节；对非正交适配器无法直接应用；在极端风格或概念变异时，仍可能出现概念偏移或细节失真。

---

## 150. Nidus: Externalized Reasoning for AI-Assisted Engineering

**arXiv ID:** 2604.05080 | [PDF](https://arxiv.org/pdf/2604.05080v1)

**作者:** Danil Gorinevski `[一作]` `[通讯]` (cybiont), Danil Gorinevski (cybiont)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

Nidus通过可验证的生活规范实现了AI代理在软件交付中的自我治理；

**💡 创新点**

将V‑model的工程不变量编译为可判定的约束表面，并实现递归自治理、摩擦协调、近似规范强化与合规舞台防御；

**🔧 技术方法**

采用S‑expression结构、SMT/SMT‑LIB2、Z3求解器、可判定逻辑与LLM（Claude、Gemini、Codex）以及代理协调机制；

**📊 数据集**

在自托管部署中生成并验证100k行系统代码，使用ISO26262、DO‑178C、ISO62304等标准的指南书，并在实际项目中测试461个特征、873条需求等；

**📈 对比分析**

通过对49个可移除元素的消融实验、全量与增量验证时间（<500 ms）以及不同模型族的摩擦事件计数评估，证明约束表面有效防止回归、指南继承可用且验证延迟低；

**⚠️ 局限性**

仅覆盖可判定约束，未能涵盖未建模的性能或架构错误；需人工或LLM生成约束；对复杂系统的可扩展性与跨组织独立验证仍待验证。

---

## 151. Toward Unified Fine-Grained Vehicle Classification and Automatic License Plate Recognition

**arXiv ID:** 2604.05271 | [PDF](https://arxiv.org/pdf/2604.05271v1)

**作者:** Gabriel E. Lima `[一作]` (Federal University of Paraná), David Menotti `[通讯]` (Federal University of Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开UFPR‑VeSV数据集，结合车辆属性（颜色、品牌、型号、类型）和车牌识别（ALPR）任务，并在该数据集上开展基准实验。

**💡 创新点**

创新点在于：①在真实监控环境中收集多视角、夜间红外、遮挡严重等极端条件；②同时提供13种颜色、26个品牌、136个型号、14类车辆类型的FGVC标注；③验证并展示FGVC与ALPR协同提升识别鲁棒性的潜力。

**🔧 技术方法**

使用多种深度学习模型（EfficientNet‑V2、MobileNet‑V3、ResNet‑50、Swin Transformer‑V2、ViT‑b16）进行迁移学习和全微调；采用数据增强、TOP‑k 预测、Grad‑CAM 关注分析；在ALPR方面对GP‑ALPR和ParSeq‑Tiny两种文本识别模型进行评估。

**📊 数据集**

主要使用UFPR‑VeSV数据集（24,945张图、16,297辆车），并与公开FGVC与ALPR基准数据集（如CompCars‑SV、AOLP、UFPR‑ALPR）做对比。

**📈 对比分析**

在FGVC任务上，EfficientNet‑V2在颜色、品牌、型号、类型上分别获得微观准确率≥93%；TOP‑3预测提升至≈99%；在ALPR任务中，ParSeq‑Tiny达到车牌整体识别率≈98%。将FGVC与ALPR联合后，验证率在单属性时≈96%，恢复率在车牌识别失效时可达≈68%，但多属性组合会导致冲突率上升。

**⚠️ 局限性**

局限性包括：①车牌标注受限于可读车牌，未覆盖完全无光或严重遮挡的ALPR极端场景；②FGVC与ALPR采用独立模型，缺乏层级一致性和联合学习；③实验未考虑完整车辆数据库的误匹配风险；④需要进一步校准置信度与选择性预测。

---

## 152. MG$^2$-RAG: Multi-Granularity Graph for Multimodal Retrieval-Augmented Generation

**arXiv ID:** 2604.04969 | [PDF](https://arxiv.org/pdf/2604.04969v1)

**作者:** Sijun Dai `[一作]` (Harbin Institute of Technology), Jun Yu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 13735 | [OpenAlex ID](https://openalex.org/A5050817770)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级多粒度图检索增强生成框架MG^2-RAG，构建统一的多模态知识图并实现跨模态检索与多跳推理。

**💡 创新点**

创新点在于：① 通过实体驱动的视觉定位与文本解析，直接构建细粒度统一的多模态节点；② 设计多粒度图检索机制，将稠密相似度聚合至多模态节点并通过图传播实现结构化推理；③ 大幅降低图构建成本，提升检索与推理效率。

**🔧 技术方法**

采用spaCy进行NER与依赖解析、SAM3进行实体驱动的图像分割、统一编码器（如CLIP或通用VLP）生成共享嵌入、Personalized PageRank进行图传播，以及稠密相似度聚合与节点激活策略。

**📊 数据集**

实验涵盖四类多模态任务，使用E-VQA、InfoSeek（检索与知识式VQA）、ScienceQA（多模态推理）和CrisisMMD（灾害场景分类）数据集。

**📈 对比分析**

与VaLiK、mKG-RAG、MMGraphRAG等基线相比，MG^2-RAG在检索、知识式VQA、推理和分类任务上均实现SOTA，平均在图构建速度上提升43.3倍、成本降低23.9倍。

**⚠️ 局限性**

局限性包括：① 仍依赖规则式关系抽取，可能忽略复杂句法关系；② 视觉定位仅基于实体分割，未覆盖完整场景结构；③ 对大规模动态图构建仍需显著GPU资源，实时或流式场景尚未充分验证。

---

## 153. ID-Sim: An Identity-Focused Similarity Metric

**arXiv ID:** 2604.05039 | [PDF](https://arxiv.org/pdf/2604.05039v1)

**作者:** Julia Chae `[一作]` (MIT CSAIL), Cusuh Ham `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 ID‑Sim 的视觉相似度度量，用于捕捉人类在识别同一实例时对背景、姿态、光照等上下文变化的不敏感性以及对细微身份变化的敏感性。

**💡 创新点**

创新点在于将全局与局部对比学习相结合，使用信息熵正则化的最优传输（Sinkhorn）来对齐不对齐的图像块，并构建了一套包含真实实例与生成编辑的多模态训练集，显著提升了身份一致性评估的判别力。

**🔧 技术方法**

技术上采用 ViT‑L 作为特征提取器，使用双头 MLP 分别投影 CLS 令牌与图像块特征，并通过监督对比损失（InfoNCE）与局部 OT 损失联合训练，同时在注意力层和 MLP 层加入 LoRA 微调。

**📊 数据集**

训练与评估数据集包括 10 个公开实例级数据集（ILIAS、FORB、MET、GLDv2、Dogs、Cats、DF2、UCO3D、LASOT、YouTubeVIS、GOT10k 等）以及通过 Qwen‑Edit 等模型生成的上下文与身份编辑样本，另外还创建了 2k 条人工标注的 Subjects2k 评测集。

**📈 对比分析**

在实例检索、概念保持与重新识别等 7 大任务上与 7 种基线（DreamSim、LPIPS、DiffSim、DINOv3、CLIP、OpenCLIP、UNED）进行比较，ID‑Sim 在 49 个评测场景中获胜 48 次，平均提升 mAP、AP 等指标 0.1–0.3 点，显著优于现有方法。

**⚠️ 局限性**

局限性包括：仅依据视觉特征定义实例，无法涵盖老化、配饰等非视觉身份变化；缺乏针对多实体场景的条件化机制（需额外掩码或文本提示），以及对极端光照或遮挡的鲁棒性仍有提升空间。

---

## 154. RCP: Representation Consistency Pruner for Mitigating Distribution Shift in Large Vision-Language Models

**arXiv ID:** 2604.04972 | [PDF](https://arxiv.org/pdf/2604.04972v1)

**作者:** Jianwei Zhang `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 111873 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

提出一种“Representation Consistency Pruner（RCP）”，通过在视觉语言模型（LVLM）解码器中进行累积残差式视觉令牌剪枝，并配备延迟修复适配器，显著减少视觉令牌数量并降低推理 FLOPs，几乎不损失性能。

**💡 创新点**

创新点包括：① 利用大型语言模型内在的交叉注意力作为剪枝基准，生成累积可行掩码，保证各层剪枝结果单调；② 引入延迟修复适配器（DRA），缓存被剪枝视觉信息并在答案生成阶段使用 FiLM 调制，补偿分布漂移；③ 通过匹配第一、二阶统计量的修复损失实现“教师-学生”分布对齐，无需对主体模型进行细调。

**🔧 技术方法**

采用的技术包括：跨层交叉注意力剪枝（Residual Cross‑Attention Pruner）、Gumbel‑Sigmoid 采样与 Straight‑Through 估计、FiLM 模块、统计对齐损失（基于 2‑Wasserstein 距离）、多阶段剪枝与修复的轻量级插入插件，训练仅更新插件参数。

**📊 数据集**

使用了七个主流多模态基准数据集：GQA、MMBench、MME、POPE、ScienceQA、VQA‑V2、VizWiz；并在 LLaVA‑1.5‑7B 与 LLaVA‑1.5‑13B 上进行评估，验证方法在不同规模模型上的可扩展性。

**📈 对比分析**

与全令牌上限模型及多种剪枝方法（ToMe、FastV、PDrop、HiRED、VisionZip、DART 等）进行对比，RCP 在 192、128、64 视觉令牌预算下分别实现 0.94%、3.27%、5.0% 的平均准确率下降，同时将 FLOPs 分别降低 63.3%、75.0%、85.7%，在保持或略优于全令牌基准的同时显著提升推理效率。

**⚠️ 局限性**

局限性：① 仍需为每个模型训练专用插件，增加额外训练步骤；② 对剪枝层次和位置的设置敏感，需手动设计；③ 在极端压缩（<64 令牌）或不同 LVLM 体系结构上可能需要进一步验证；④ 修复适配器主要在答案生成阶段工作，对其他下游任务的适用性尚待探索。

---

## 155. A Multi-Agent Framework for Automated Exploit Generation with Constraint-Guided Comprehension and Reflection

**arXiv ID:** 2604.05130 | [PDF](https://arxiv.org/pdf/2604.05130v1)

**作者:** Siyi Chen `[一作]` (Alibaba Group), Wenyuan Xu `[通讯]` (Aarhus University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5099292212)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多代理框架（Code Analyzer、Code Generation、Validation、Reflection 和 Supervisor 代理），利用大型语言模型自动生成并验证软件库中的漏洞利用代码。

**💡 创新点**

创新点包括：①将 AE 过程拆解为专门的子代理；②采用自然语言约束来引导 LLM 理解并满足复杂代码逻辑；③设计反馈循环，让 Validation 结果实时更新约束并反思是否为误报；④在单一框架内集成静态污点分析、动态验证与反思。

**🔧 技术方法**

技术手段包括多代理架构、ReAct 推理-执行模式、LLM（如 GPT‑4o）生成约束与代码、自然语言约束提取、静态污点分析、沙箱执行验证、反射代理与错误分析。

**📊 数据集**

评估使用真实世界的 Java 与 JavaScript 开源库（含多种公开项目），并通过公开漏洞数据库与自身收集的零日样本进行实验，最终发现 146 个零日漏洞。

**📈 对比分析**

与现有静态+符号执行工具（如 CodeQL）、Fuzzing 工具以及传统 AEG 系统比较，AEG 成功率提升 34.64%，对 53.47% 的目标漏洞能生成可执行利用；在真实项目中实现了 146 次零日发现，验证了显著性能提升。

**⚠️ 局限性**

局限性包括：仅验证 5 类漏洞和 2 种语言；结果受限于静态分析器产生的警报质量；LLM 上下文窗口和生成错误仍可能导致误报或生成失败；需要进一步扩展至更多漏洞类型与语言。

---

## 156. Improving Sparse Memory Finetuning

**arXiv ID:** 2604.05248 | [PDF](https://arxiv.org/pdf/2604.05248v1)

**作者:** Satyam Goyal `[一作]` (University of Michigan), Prakhar Gupta `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Qwen-2.5-0.5B进行稀疏记忆层重构，并构建了持续学习的完整流水线。

**💡 创新点**

提出基于KL散度的理论化记忆槽选择方法，并开源了可迁移的重构流程。

**🔧 技术方法**

使用稀疏记忆微调（SMF）、梯度掩蔽、KL散度槽评分以及回弹阶段恢复等技术。

**📊 数据集**

在OpenAssistant、TriviaQA、SimpleQA、GSM8k和NaturalQuestions等数据集上进行实验。

**📈 对比分析**

与全参数微调对比，稀疏微调在目标任务上收敛更快，且在GSM8k与NaturalQuestions上的性能损失更小；全微调则显著导致灾难性遗忘。

**⚠️ 局限性**

槽选择仍对性能敏感；KL方法在某些任务下可能略微增加遗忘；实验仅覆盖少数模型与任务，缺乏大规模验证。

---

## 157. Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification

**arXiv ID:** 2604.05302 | [PDF](https://arxiv.org/pdf/2604.05302v1)

**作者:** Jinhong Jeong `[一作]` (Yonsei University), Youngjae Yu `[通讯]` (Seoul National University)

**通讯引用:** 2153 | [OpenAlex ID](https://openalex.org/A5101881857)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无需平行语料的多语言自适应文本简化框架，利用强化学习按CEFR、JLPT、TOPIK、HSK等标准动态控制词汇难度。

**💡 创新点**

创新点在于：①用GRPO强化学习结合词汇覆盖、语义保留和连贯性三大奖励模块；②统一训练单一策略模型即可适配四种不同语言；③在低水平下显著提升非英语简化效果。

**🔧 技术方法**

核心技术包括：GRPO强化学习、Qwen3 4B策略模型、LoRA参数高效微调、LLM‑as‑judge评估（使用更大模型作为判别器），以及词汇覆盖、语义一致性与连贯性奖励函数。

**📊 数据集**

使用数据集：43,786词汇层级词表（CEFR、JLPT、TOPIK、HSK）、Wikipedia Featured Article 8,057篇作为训练种子（拆分为69,220段落），以及Parallel Global Voices新闻文本作为跨域测试集。

**📈 对比分析**

通过与GPT‑5.2、Gemini‑2.5‑Flash等零-shot LLM以及FUDGE受约束解码基线对比，实验表明在所有语言和尤其是低水平（A1/A2、N5/N4、TOPIK1/2、HSK1/2）下，词汇覆盖率提升约30%（英语81.6% vs Gemini 73.8%），语义保留与连贯性亦同步提升。

**⚠️ 局限性**

限制包括：仅考虑词汇难度而未覆盖句法复杂度；词汇覆盖度受限于已有词表；模型规模受资源限制，需进一步扩展验证；评估主要依赖自动化指标，缺乏真实学习者的长期实验。

---

## 158. PHAROS: Pipelined Heterogeneous Accelerators for Real-time Safety-critical Systems With Deadline Compliance

**arXiv ID:** 2604.05308 | [PDF](https://arxiv.org/pdf/2604.05308v1)

**作者:** Shixin Ji `[一作]` (Brown University), Peipei Zhou `[通讯]` (Brown University)

**通讯引用:** 2041 | [OpenAlex ID](https://openalex.org/A5063866156)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一个面向软实时安全关键系统的异构加速器设计框架，支持预占和动态调度，并基于实时理论进行设计空间探索。

**💡 创新点**

创新点在于将利用率≤1的实时约束与流水线拓扑限制结合，提出最大利用率目标的SRT导向DSE，配合Beam Search启发式搜索算法以及硬件级预占支持；同时实现了可在FPGA上切换FIFO/EDF调度的可插拔硬件模块。

**🔧 技术方法**

使用的技术包括：实时理论（利用率约束与响应时间分析）、双缓冲输出驻留数据流、预占开销建模、Beam Search搜索启发式、FPGA Versal VCK5000硬件实现、以及对任务的层级分解与分配。

**📊 数据集**

采用的真实网络数据集有：点云处理任务 PointNet、Point Transformer、图像处理任务 DeiT‑T、Res‑MLP、MLP‑Mixer 等，全部截断为若干块以保持层异质性，构成多任务周期组合用于实验。

**📈 对比分析**

与传统吞吐量导向的 CHARM DSE 基线相比，SRT导向 DSE 在可调度任务集上可获得约 1.44×–2.28× 更多可行配置，最大利用率降低 3.7%–6.2%，在 FIFO 与 EDF 两种调度下均能满足响应时间约束，并通过响应时间分析验证二者互补性。

**⚠️ 局限性**

主要限制包括：流水线拓扑仅适用于层序列化的模型，无法处理更复杂的网络拓扑；预占开销模型在不同硬件实现中的准确性有限；Beam Search 仍可能陷入局部最优；未探讨更高级的实时理论与多链路协同设计。

---

## 159. RAG or Learning? Understanding the Limits of LLM Adaptation under Continuous Knowledge Drift in the Real World

**arXiv ID:** 2604.05096 | [PDF](https://arxiv.org/pdf/2604.05096v1)

**作者:** Hanbing Liu `[一作]` (Tsinghua University), Yang Li `[通讯]` (Tsinghua University)

**通讯引用:** 46014 | [OpenAlex ID](https://openalex.org/A5100769533)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了用于评估LLM在持续知识漂移下表现的基准，并构建了一种基于事件演化图的时序检索框架，帮助模型在不修改参数的前提下实现时序一致推理。

**💡 创新点**

创新点在于：①从时间戳证据构建真实世界动态事件基准；②设计事件演化图（EEG）结构化检索并通过多视角序列化实现无参数更新的时序一致推理；③将历史重建、事件增强和实体中心链等组件集成到检索与推理流程中。

**🔧 技术方法**

使用技术包括：时序感知检索、语义相似度与时间相关度融合、历史事件重建、事件增强、实体中心链连接、多视角（时序视角、实体视角）序列化以及基于LLM的生成推理。

**📊 数据集**

使用的数据集是自2024-2025年十个领域的时间戳知识四元组集合，以及基于这些四元组构建的历史、当代（单/多时间戳、多源）和常识四类问答对。

**📈 对比分析**

与直接生成、参数更新（ROME、MEMIT、WISE、LoRA）、传统RAG和ReAct-RAG等方法对比，所提出的Chronos在历史、当代（尤其是多时间戳和多源推理）和常识任务上均实现显著提升，整体性能在所有基准任务上名列前茅。

**⚠️ 局限性**

局限性包括：基准假设模型知识截止前2024年，未覆盖更广泛或更复杂的知识动态；历史重建与时序检索可能引入干扰；缺乏针对不同领域部署场景的多样化评测。

---

## 160. Finite-Step Invariant Sets for Hybrid Systems with Probabilistic Guarantees

**arXiv ID:** 2604.05102 | [PDF](https://arxiv.org/pdf/2604.05102v1)

**作者:** Varun Madabushi `[一作]` (Georgia Institute of Technology), Maegan Tucker `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 283 | [OpenAlex ID](https://openalex.org/A5054944952)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于采样优化的算法，利用Poincaré返回图计算混合系统的有限步不变椭圆形集合，并给出概率保证。

**💡 创新点**

创新点在于将采样-保留方法（holdout）与椭圆拟合相结合，既能在仅通过仿真得到的黑盒返回图上求解不变集，又能在有限样本下给出PAC（Probably Approximately Correct）级别的置信界；同时算法迭代收敛，能够处理凸与非凸不变集，且提出可扩展至RBF表示的方案。

**🔧 技术方法**

使用的技术包括：Poincaré返回图的构造、采样优化（随机采样与最小体积包围椭圆求解）、保留方法的二项尾部逆推、凸优化（半正定规划、最小体积覆盖问题）、线性化与收敛性理论、并行仿真与GPU加速。

**📊 数据集**

数据集主要是三组仿真系统：二维凸/非凸扩张-收缩器（CEC、NEC）以及四维双足步态模型（Compass‑Gait Walker）；对前两者利用已知的解析不变集做对照，对Compass‑Gait Walker 则通过数值Poincaré返回图进行验证。

**📈 对比分析**

与传统仅使用线性化或手工估计的球形不变集相比，算法在保持相同置信度（β=10⁻⁹）下，椭圆不变集的误差约为1.1%（CEC）和22%（NEC），并在步态模型中获得 1% 以下的置信误差；收敛速度为 5–75 次迭代，计算量可通过并行化显著降低。

**⚠️ 局限性**

局限性包括：需预先假设有限步不变集存在，且对高维系统样本量呈指数增长；椭圆表示对非凸真实不变集保守，RBF方案虽更精准但求解非凸优化时收敛更慢；最终准确度受用户设定的误差阈值影响，过于严格时可能无法收敛。

---

## 161. Simulating the Evolution of Alignment and Values in Machine Intelligence

**arXiv ID:** 2604.05274 | [PDF](https://arxiv.org/pdf/2604.05274v1)

**作者:** Jonathan Elsworth Eicher `[一作]` `[通讯]` (Antimemetic AI), Jonathan Elsworth Eicher (Antimemetic AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

将模型对齐过程视作进化系统，构建了基于信念（belief）空间的仿真框架，研究对齐信号与真实价值的相关性、误导性信念的固定化以及测试设计与突变对模型群体演化的影响。

**💡 创新点**

创新点在于：①首次把对齐评估引入进化论框架；②通过构造多模态信念分布和可变激活函数揭示误导性信念在选择压制下的固定机制；③展示多种干预（动态测试、突变、评估提升）组合时的协同效应，说明单一措施不足，需多元协同。

**🔧 技术方法**

使用技术包括：蒙特卡罗仿真（多级水平，包含无突变、突变、相似激活、动态测试等），贝叶斯双变量正态分布采样信念，轮盘赌与softmax选择，欧氏/皮尔逊相关分析，普通最小二乘回归，10k次置换检验与Benjamini‑Hochberg校正。

**📊 数据集**

数据集：完全合成的信念集合（单/多峰双变量正态），以及对应的评估问题集合 Q（随机或基于相似度的激活），不使用真实外部语料或基准测试。

**📈 对比分析**

与基线（仅遗传、无突变、静态测试）对比，采用排列检验和回归评估；结果显示：①单项干预（动态测试、突变、评估提升）只能在部分指标上获益；②组合干预（动态测试+突变+评估提升）在保持对齐适配度的同时显著降低误导性比例并提升真实价值（p_adj<0.001），体现协同提升。

**⚠️ 局限性**

局限性：①信念离散且独立、固定维度，未考虑非线性或网络结构；②评估者单一且同质；③初始信念分布为理想化正态或混合正态，可能与真实模型训练分布偏离；④未引入多模型交互与市场/机构选择压力；⑤突变机制简化，缺乏对实际Fine‑Tune或RLHF过程的精细映射。

---

## 162. Final Report, Center for Computer-Integrated Computer-Integrated Surgical Systems and Technology, NSF ERC Cooperative Agreement EEC9731748, Volume 1

**arXiv ID:** 2604.05272 | [PDF](https://arxiv.org/pdf/2604.05272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 163. Bypassing the CSI Bottleneck: MARL-Driven Spatial Control for Reflector Arrays

**arXiv ID:** 2604.05162 | [PDF](https://arxiv.org/pdf/2604.05162v1)

**作者:** Hieu Le `[一作]` (Texas A&M University), Sabit Ekin `[通讯]` (Texas A&M University)

**通讯引用:** 1016 | [OpenAlex ID](https://openalex.org/A5014255349)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发一种利用多智能体强化学习（MARL）控制机械可调金属反射阵列的框架，绕过传统RIS所需的CSI估计，实现基于用户位置信息的自适应波束聚焦。

**💡 创新点**

创新点在于：① 将每个反射单元的双自由度角度参数压缩为聚焦点三维坐标的空间抽象，显著降低控制维度；② 采用集中训练、去中心化执行（CTDE）和MAPPO算法实现多智能体的协作学习；③ 完全消除CSI依赖，实现CSI‑free操作，并在动态NLOS环境中展示出高鲁棒性。

**🔧 技术方法**

技术包括：多智能体强化学习（MAPPO）、CTDE架构、机械可调六角金属单元、三维空间聚焦点控制、Ray‑tracing仿真（NVIDIA Sionna）、基于Blender的环境构建、全局/局部奖励设计。

**📊 数据集**

使用高保真Ray‑tracing仿真数据集：60 GHz mmWave下的L形走廊环境，包含1个基站、3个移动用户、72片六角金属单元；所有实验均在Sionna+Blender生成的虚拟场景中完成，没有真实硬件数据。

**📈 对比分析**

对比方法包括：无反射、平面静态反射、单智能体聚焦、受硬件约束的列级多智能体（列式旋转）。实验表明，多智能体聚焦在RSSI上相较平面反射提升26.86 dB，平均RSSI为-66.83 dBm；收敛奖励达42，显著优于基线约25；在用户移动与定位噪声（0–1 m）下保持稳定，降级仅约4–6 dB，且时间响应仅1个仿真步。

**⚠️ 局限性**

局限性包括：尚未在物理硬件上验证，需实验平台验证；依赖高精度定位（<1 m）才能保持最佳性能，定位误差导致RSSI波动；机械调节速度和扭矩限制未在仿真中体现；模型训练成本高，需大量Ray‑tracing样本；在更复杂多用户/多基站环境下的可扩展性待进一步评估。

---

## 164. Learning-Based Multi-Criteria Decision Making Model for Sawmill Location Problems

**arXiv ID:** 2604.04996 | [PDF](https://arxiv.org/pdf/2604.04996v1)

**作者:** Mahid Ahmed `[一作]` (University of Southern Mississippi), Chao Meng `[通讯]` (University of Southern Mississippi)

**通讯引用:** 1626 | [OpenAlex ID](https://openalex.org/A5101793304)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出基于机器学习的多准则决策框架（LB‑MCDM），并应用于密西西比州锯木厂选址，生成可视化适用性地图。

**💡 创新点**

创新之处在于将ML与GIS‑MCDM结合，利用SHAP解释器客观确定多因子权重，消除传统主观加权的偏差。

**🔧 技术方法**

采用随机森林、支持向量机、XGBoost、逻辑回归、KNN等分类器，配合SHAP特征重要性分析与GIS空间叠加技术。

**📊 数据集**

使用密西西比州的森林资源、交通网络、坡度、城市距离、失业率、市场收入、供需比、土地覆盖、降水等多源GIS与表格数据，共计1.1万余条样本。

**📈 对比分析**

通过训练/测试分割、SMOTE‑ENN平衡、准确率、召回率、F1和AUC等指标比较，随机森林获得最高准确率0.8648、AUC 0.9656，优于其他模型。

**⚠️ 局限性**

局限性包括仅在密西西比州验证，地形平坦导致坡度与土地覆盖影响有限，未纳入政策、土地成本等因素，需进一步扩展验证。

---

## 165. Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding

**arXiv ID:** 2604.05015 | [PDF](https://arxiv.org/pdf/2604.05015v1)

**作者:** Chaoyou Fu `[一作]`, Ran He `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Video‑MME‑v2 视频多模态评测基准，构建三层级进阶任务体系，并采用基于问题组的非线性评分方法来衡量模型的一致性与推理连贯性。

**💡 创新点**

创新点在于：①从信息聚合 → 时序建模 → 复杂推理的渐进式层级设计；②引入一致性组与连贯性组的组合评估，突出模型在相关问题上的一致表现；③使用非线性组分数来替代传统平均准确率，强化对模型鲁棒性与逻辑连贯性的考核；④严谨的人工标注与多轮审核流程，确保数据质量与难度平衡。

**🔧 技术方法**

使用多模态大语言模型（如 Gemini‑3‑Pro、GPT‑5、Qwen 系列）进行实验，结合视觉帧、字幕/音频、长上下文采样等技术；评估指标为平均准确率与基于组的非线性分数；对比时采用人类专家基准与公开模型的多种设置。

**📊 数据集**

数据集包含 800 条视频（约10.4分钟平均时长，80% 2025 以后发布），3,200 道多项选择题，采用 12 名注释员与 50 名独立评审，共计 3,300 小时人工工作；每题配 8 个答案选项，确保长度一致以防猜测。

**📈 对比分析**

与多款商业与开源视频 MLLM 在 w./wo. subtitles 以及不同帧采样设置下对比，发现最佳模型 Gemini‑3‑Pro 在非线性分数上仅 49.4，远低于人类专家 90.7；同类开放源代码模型最高 39.1；实验还揭示层级瓶颈、音频提升效果、规模与能力补偿关系。

**⚠️ 局限性**

主要局限在于：当前模型在层级 3 复杂推理上表现仍远低于人类，且存在对语言先验的过度依赖；视觉信息聚合与时序建模错误会级联影响高阶推理；即使在音频与长上下文支持下，细粒度动作与物理推理仍不足；因此需要进一步提升感知、时序建模与连贯推理能力。

---

## 166. Census Dual Graphs: Properties and Random Graph Models

**arXiv ID:** 2604.04960 | [PDF](https://arxiv.org/pdf/2604.04960v1)

**作者:** Sara Anderson `[一作]` (Claremont Graduate University), Anne Friedman `[通讯]` (Scripps College)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过构建并分析美国各州县、市、普查区块及区块组的双图，评估其度分布、连通性、平面性、生成树常数以及随机生成树的可分割性，并基于网格扰动与 Delaunay 三角化等方法设计了多种随机图模型来近似这些属性。

**💡 创新点**

创新点在于首次系统量化双图的关键图结构指标，并提出并验证了多种基于网格与 Delaunay 的随机图模型，其中模型 11c、8 与 12 最能逼近真实双图的统计特征。

**🔧 技术方法**

所用技术包括 Wilson 算法生成随机生成树、Gamma Bernoulli 近似方案估计分割概率、线性回归分析分割概率与顶点数的关系，以及随机点云生成、网格扰动、Delaunay 三角化与边缘扰动等图模型构造方法。

**📊 数据集**

数据来源为 Redistricting Data Hub 提供的美国 47–49 个州的县、市、普查区块和区块组 shapefile，通过这些文件生成对应的双图进行实验。

**📈 对比分析**

比较方法采用平均度、最大度、平面性、连通性、平均生成树常数等统计量与模型结果对齐；实验显示模型 11c、8 与 12 的平均度约为 5.4、生成树常数约为 1.43，连通性达到 100%，但平面性仅 0%–0.1%，与真实双图相比差距在可接受范围内。

**⚠️ 局限性**

局限性包括计算规模受限（最多 2500 顶点，未考虑规模更大的普查块级图）、分割概率估计仅基于顶点而非人口权重、模型大多非平面且未探索不同点分布或更大规模的随机图模型。

---

## 167. Do Domain-specific Experts exist in MoE-based LLMs?

**arXiv ID:** 2604.05267 | [PDF](https://arxiv.org/pdf/2604.05267v1)

**作者:** Giang Do `[一作]` (Deakin University), Truyen Tran `[通讯]` (Deakin University)

**通讯引用:** 6711 | [OpenAlex ID](https://openalex.org/A5085471517)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型中混合专家（MoE）架构进行系统评估，验证域特定专家的存在，并提出一种无需额外训练、零推理开销的 Domain Steering Mixture of Experts (DSMoE) 方法来提升模型在目标域和非目标域的表现。

**💡 创新点**

创新点在于首次揭示并量化 MoE 模型中的域专属专家，并利用这些专家通过训练-free 的路由加权方式实现性能提升，显著优于现有 RICE、SFT 等对齐方法。

**🔧 技术方法**

主要技术包括 MoE 模型的专家路由、基于梯度的 token 重要性评估、域专家得分计算、Steering 系数 α 的调节以及无训练的推理加权机制。

**📊 数据集**

使用的数据集涵盖多域评测集 MMLU‑Pro、科学领域难度高的 GPQA‑Diamond 以及美国 AIME 竞赛题目，并在多款 MoE LLM（PhiMoE、GPT‑OSS、Qwen3、DeepSeek 等）上进行实验。

**📈 对比分析**

与原始 MoE、RICE 和 LoRA‑SFT 基线对比，DSMoE 在目标域平均提升 1.5–29% 的准确率，非目标域亦保持 4–27% 的提升，且不增加推理成本，显示出优异的泛化与效率。

**⚠️ 局限性**

局限性包括实验仅覆盖至 120B 参数规模的模型，未评估更大规模 MoE，且实验资源有限，未覆盖所有潜在领域和任务。

---

## 168. Corporate Training in Brazilian Software Engineering: A Quantitative Study of Professional Perceptions

**arXiv ID:** 2604.05263 | [PDF](https://arxiv.org/pdf/2604.05263v1)

**作者:** Rodrigo Siqueira `[一作]` (CESAR School), Danilo Monteiro Ribeiro `[通讯]` (CESAR School)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在巴西软件工程师中通过问卷调查和多列相关分析研究企业培训质量与效能的影响因素

**💡 创新点**

首次将Salas与Cannon-Bowers培训框架量化应用于软件工程领域，发现认知投入、活动多样性和教师表现为最强预测因子

**🔧 技术方法**

使用结构化问卷、5点李克特量表以及polychoric correlation统计方法

**📊 数据集**

收集了282名巴西软件工程师的自报数据

**📈 对比分析**

未采用对照实验，而是通过相关系数评估三大因素与整体满意度的关联，相关系数均超过0.80，显示显著关联

**⚠️ 局限性**

样本为便利抽样、主要来自巴西、单项量表缺乏内部一致性检验、未区分培训类型且未进行可靠性与因素分析

---

## 169. A Gradual Probabilistic Lambda Calculus

**arXiv ID:** 2604.05246 | [PDF](https://arxiv.org/pdf/2604.05246v1)

**作者:** Wenjia Ye `[一作]` (National University of Singapore), Federico Olmedo `[通讯]` (University of Chile)

**通讯引用:** 844 | [OpenAlex ID](https://openalex.org/A5009421242)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了首个渐进式概率 λ 计算机语言，结合了概率选择、分布类型和渐进类型注解，并通过抽象渐进式类型（AGT）方法系统推导出源语言和目标语言的语义与类型规则；

**💡 创新点**

创新点在于将渐进式类型系统迁移到概率编程环境，利用概率耦合（couplings）构造一致性、精确性与一致转移，证明了类型安全、渐进式保证与对静态语言的保守扩展；

**🔧 技术方法**

主要技术包括：AGT 方法、概率耦合理论、分布类型语义、类型一致性与精确性构造、证据（evidence）与一致转移运算、动态语义通过目标语言演化实现；

**📊 数据集**

无实验数据集，本工作为理论语言设计与证明，未进行数据集实验；

**📈 对比分析**

本工作未给出实验对比或性能评测，而是通过形式化证明展示语言的安全性与渐进式性质；

**⚠️ 局限性**

局限性包括：未处理无限递归导致的非终结；采样语义不适用于分布类型检查；仅覆盖基础 λ 计算机与概率选择，未扩展到更复杂特性（如效应、并发、约束优化）等。

---

## 170. Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting

**arXiv ID:** 2604.05316 | [PDF](https://arxiv.org/pdf/2604.05316v1)

**作者:** Monica Tang `[一作]` (University Of California Berkeley), Avideh Zakhor `[通讯]` (University Of California Berkeley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套多视角掩膜关联与三维对象代码本生成的流水线，能够在3D高斯剖面中实现室内资产的语义分割与检测。

**💡 创新点**

创新点在于结合深度自适应过滤与语义约束的高斯聚合策略，显著抑制误合并与背景噪声，提升多视角掩膜一致性。

**🔧 技术方法**

采用360°无人机捕获图像，利用Open‑Vocabulary OWLv2进行目标检测、SAM进行分割，并通过自定义深度处理、语义约束合并、低权重高斯过滤以及空间聚类等步骤构建代码本。

**📊 数据集**

实验使用了加州大学伯克利分校Cory Hall的两个大规模室内数据集（Cory 3rd Floor和Cory 307 Office），共计超过8千张视图。

**📈 对比分析**

与基线GAGA以及单视图OWLv2相比，该方法在掩膜关联上mIoU提升约10-14点、F1提升约20-30点，运行时缩短约2-4倍；在对象检测上mAP提升约12-15点、mLAMR下降约10-15点，表现优于基线。

**⚠️ 局限性**

局限性包括仍受OWLv2检测误差影响、透明/反射物体导致高斯漂浮、以及对高度稠密多类别场景仍需进一步优化聚类与过滤阈值。

---

## 171. A Theory-guided Weighted $L^2$ Loss for solving the BGK model via Physics-informed neural networks

**arXiv ID:** 2604.04971 | [PDF](https://arxiv.org/pdf/2604.04971v1)

**作者:** Gyounghun Ko `[一作]` (Chinese Academy of Sciences), Myeong-Su Lee `[通讯]` (Seoul National University)

**通讯引用:** 1653 | [OpenAlex ID](https://openalex.org/A5021226126)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在物理信息神经网络（PINN）中求解BGK模型时，标准L²损失函数无法保证解的物理正确性的问题，提出并验证了一种基于速度加权的L²损失函数。通过构造反例证明标准损失易导致宏观量误差显著；通过理论证明和数值实验，展示加权损失在保证收敛性的同时显著提升了分布函数和宏观量的预测精度。

**💡 创新点**

创新点在于：1) 明确指出BGK模型中标准L²损失不足，并给出严谨的反例；2) 提出一种满足可积性条件的速度权重函数w(v)，并通过数学稳定性分析证明其可使误差随损失趋零而收敛；3) 在多维高维情境（1D、2D、3D）和不同Knudsen数下，对比标准、相对加权和理论加权损失，展示了理论加权损失的普适性与优势。

**🔧 技术方法**

技术方法包括：Physics‑Informed Neural Networks (PINN)；可分离（Separable）PINN 架构；基于速度加权的L²损失（w(v)=1+α|v|^β）；JAX 框架；Lion 优化器；对残差、边界、初始条件采用加权积分。理论方面，利用局部麦克斯韦分布的Lipschitz性质和能量估计，给出收敛的上界。

**📊 数据集**

数据集为合成的基准 PDE 试例：1D 平滑、1D Riemann、2D 平滑、2D Riemann、3D 平滑问题，初始分布为 Maxwellian，空间边界为周期或 Neumann，使用数值解（高阶半 Lagrangian 法）作为参考。

**📈 对比分析**

比较方法：对每个试例计算相对 L² 与 L¹ 误差，分别在分布函数及宏观量（ρ、u、T）上评估。实验结果显示：相对损失在某些情形下优于标准 L²，但在具有复杂边界或高维特征时表现不稳定；而理论加权损失（α=0.1, β=4）在所有测试中均实现了最低误差，尤其在宏观量的 L¹ 误差上优于两者。

**⚠️ 局限性**

局限性：1) 理论保证依赖于损失能被充分最小化，实际训练仍受神经网络表达、采样误差和非凸优化的限制；2) 目前仅在 BGK 模型上验证，尚未扩展到完整 Boltzmann 或更复杂碰撞算子；3) 加权函数的选择需要满足积分条件，过大或过小的权重可能导致训练不稳定或梯度消失。

---

## 172. Faster Superword Tokenization

**arXiv ID:** 2604.05192 | [PDF](https://arxiv.org/pdf/2604.05192v1)

**作者:** Craig W. Schmidt `[一作]` (Kensho Technologies), Yuval Pinter `[通讯]` (Ben Gurion University of Negev)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一且显著加速的 BoundlessBPE 与 SuperBPE 训练实现，利用两阶段方法与 supermerge 候选聚合技术，实现了与原实现相同的模型。

**💡 创新点**

核心创新在于：①将原始同时进行的 regular merge 与 supermerge 分离为两阶段；②通过聚合 supermerge 候选（按频率统计）避免在内存中保存完整文档；③在第二阶段自动确定 regular 与 supermerge 的比例，消除超参数；④针对非空格分隔语言引入字符级 pre-tokenization，避免跨字节无效 token。

**🔧 技术方法**

技术包括：Byte Pair Encoding（BPE）基础算法；supermerge 候选聚合与频率统计；两阶段 BoundlessBPE 训练框架；正则表达式 pre-tokenization；Rust 实现与 Python 对照；Inference 加速技巧（预检词典、按先出现规则应用）。

**📊 数据集**

主要使用 MiniPile 数据集（1M 文档）进行实验，且通过 1GB 训练数据比较原始实现与加速实现的运行时间。

**📈 对比分析**

方法对比：原始 BoundlessBPE 需要 4.7 CPU 天；加速实现仅需 603 秒；SuperBPE 由 “few hours” 降至 593 秒，速度提升 600 倍以上；BPE 在相同数据下仅 59 秒；Python 与 Rust 版本对比显示 Rust 在训练与推理上均显著加速；在同一 vocab 大小下，两阶段与原实现产生相同词表与 merge 规则。

**⚠️ 局限性**

限制包括：实现目前为单线程，未充分利用多核并行；未对 downstream 任务的性能做进一步评估；greedy left-to-right split 可能对某些语言的下游效果产生影响；仅针对预训练文本数据，没有直接验证对多模态或跨语言任务的适用性。

---

## 173. Spec Kit Agents: Context-Grounded Agentic Workflows

**arXiv ID:** 2604.05278 | [PDF](https://arxiv.org/pdf/2604.05278v1)

**作者:** Pardis Taghavi `[一作]`, Santosh Bhavani `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Spec Kit Agents，一个多代理的基于规范驱动开发（SDD）的工作流，加入了阶段性上下文发现和验证钩子，解决传统工作流中的上下文盲目问题。

**💡 创新点**

核心创新在于将上下文发现与验证拆离到工作流边界，形成可插拔的“发现‑验证”层，使生成过程能在每个阶段实时获取仓库证据并进行校验，显著提升可靠性。

**🔧 技术方法**

采用多代理编排（状态机、PM 与开发者代理）、MiniMax‑M2.5 LLM 与 Claude Opus 4.6 评估器、GitHub Spec Kit 规范化流程、文件检索、依赖检查、单元测试与 linters 等工具。

**📊 数据集**

实验数据集包括 5 个开源仓库（FastAPI、Airflow、Dexter、Plausible Analytics、Strapi）共 32 个功能任务（128 次实验），以及公开基准 SWE‑bench Lite（300 条问题）。

**📈 对比分析**

通过对比 Baseline、Augmented、Full、Full‑Augmented 及其 ablation，使用 Wilcoxon 检验和 LLM‑评判，发现 Full‑Augmented 在 90‑min 预算下平均得分提升 0.15（+3.0%），测试通过率保持 99.7–100%；在 SWE‑bench Lite 上 Pass@1 从 56.5% 提升到 58.2%，位列同类系统前列。

**⚠️ 局限性**

主要局限包括：额外的发现与验证阶段增加 13–37 分钟的运行时间；对模型与工具的依赖需要额外配置；在高复杂度任务中收益明显，但对低风险小规模任务的收益有限。

---

## 174. A mathematical theory of evolution for self-designing AIs

**arXiv ID:** 2604.05142 | [PDF](https://arxiv.org/pdf/2604.05142v1)

**作者:** Kenneth D Harris `[一作]` `[通讯]`, Kenneth D Harris

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

构建了自递归自我改进 AI 的进化模型，并对其长期动力学进行严格数学分析。

**💡 创新点**

将生物学中随机突变的随机游走模型改为有向树模型；引入“lineage exponent”概念捕捉后代线性增长潜力；提出 η‑preservation 与 η‑locking 两个可实现的条件，证明它们能保证或不保证均衡、收敛及对齐性；探讨了误导（deception）对 fitness 与人类效用的影响。

**🔧 技术方法**

使用线性算子、极限上/下界、Perron–Frobenius 定理、Price 方程、几何平均等数学工具对模型进行证明与推导。

**📊 数据集**

无实验数据或数据集，全部为理论推导。

**📈 对比分析**

本工作没有与现有方法进行实验对比，性能指标也未给出；重点在理论性质与收敛性分析。

**⚠️ 局限性**

假设过于简化：忽略 AI 之间的通信、协作与竞争、动态适应策略；模型为单向有向树，缺乏交叉、合并等现实进化机制；因此无法完整刻画真实自我改进 AI 的演化行为。

---

## 175. Offline RL for Adaptive Policy Retrieval in Prior Authorization

**arXiv ID:** 2604.05125 | [PDF](https://arxiv.org/pdf/2604.05125v1)

**作者:** Ruslan Sharifullin `[一作]` (Stanford University), Hannah Clay `[通讯]` (Stanford University)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5021315170)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将医保Prior Authorization（PA）政策检索建模为马尔可夫决策过程（MDP），并通过离线强化学习（CQL、IQL、DPO）训练可自适应检索与停止的策略；

**💡 创新点**

①将离线RL与检索-增强决策结合，首次实现成本感知的自适应检索；②将Direct Preference Optimization迁移至转移级MDP，突破传统trajectory-level限制，取得高准确率同时显著减少检索步数；③通过合成CMS覆盖数据与规则Oracle构建真实且可复现的实验环境，系统性评估三种RL算法的效率与准确率。

**🔧 技术方法**

句子BERT检索、状态编码为请求嵌入+检索历史均值、离线RL算法（CQL、IQL、DPO、BC）、奖励设计为正确率+检索成本、基线为固定top‑K与启发式检索。

**📊 数据集**

186条CMS Medicare Coverage Database政策段落（10个医疗程序共186段），基于公开CMS数据生成的2,000条训练和200条测试的合成PA请求。

**📈 对比分析**

对照固定top‑K、启发式策略及BC，评估指标为决策准确率、累计奖励（返回值）与检索步数。结果显示CQL与BC均达92%准确率但需20步；IQL以3.4步实现62.5%准确率；DPO在92%准确率的同时仅需10.6步，取得最佳准确率‑成本平衡。

**⚠️ 局限性**

使用合成请求与规则Oracle的实验环境可能与真实临床情境差异；离线数据来源有限，难以覆盖所有复杂程序；DPO的转移级偏好假设在其他MDP上可能不成立。

---

## 176. Typify: A Lightweight Usage-driven Static Analyzer for Precise Python Type Inference

**arXiv ID:** 2604.05067 | [PDF](https://arxiv.org/pdf/2604.05067v1)

**作者:** Ali Aman `[一作]` (University of Windsor), Shaowei Wang `[通讯]` (University of Manitoba)

**通讯引用:** 2391 | [OpenAlex ID](https://openalex.org/A5100664833)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Typify，一种纯静态、使用驱动的Python类型推断引擎，能够在无注解代码中精确恢复变量、参数和返回值类型。

**💡 创新点**

创新点在于：1）完全不依赖预训练模型或注解；2）结合符号执行、固定点迭代与模块级依赖图实现跨模块推断；3）使用上下文匹配检索补全未覆盖槽，提升覆盖率同时保持可解释性。

**🔧 技术方法**

技术手段包括：符号执行、迭代固定点分析、项目级依赖图构建、使用驱动的类型传递、检索匹配（context‑matching）以及类型统一与归约。

**📊 数据集**

实验使用了ManyTypes4Py和Typilus两个真实项目数据集，涵盖数千个Python文件。

**📈 对比分析**

在与Type4Py、HiTyper、Pyre等基线比较中，Typify在变量/参数/返回类型预测上基本匹配或略低于HiTyper，远超Type4Py；与静态检查器相比提升显著；将Typify与DL模型（Type4Py）结合后进一步提升准确率，证明可与深度学习方法互补。

**⚠️ 局限性**

局限性：无法推断未被调用的函数、极端动态特性（如反射、动态导入、代码生成）以及检索库域漂移导致的匹配误差。

---

## 177. Experimental Demonstration of an On-Chip CMOS-Integrated 3T-1MTJ Probabilistic Bit - A P-Bit

**arXiv ID:** 2604.05191 | [PDF](https://arxiv.org/pdf/2604.05191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 178. XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts

**arXiv ID:** 2604.05242 | [PDF](https://arxiv.org/pdf/2604.05242v1)

**作者:** Jiahao Xu `[一作]` (University of Nevada, Reno), Zikai Zhang `[通讯]` (University of Nevada, Reno)

**通讯引用:** 27 | [OpenAlex ID](https://openalex.org/A5119870183)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多比特水印方法XMark，用于在LLM生成文本中嵌入可检索的二进制信息，确保水印在文本质量不受明显影响的前提下能够被可靠解码。

**💡 创新点**

创新点包括：1）Leave‑One‑Shard‑Out (LoSo)策略把大部分词表设为绿名单，以显著提升文本质量；2）通过多哈希键生成多份词表排列并取交集得到Evergreen List，使得单个token在解码时能提供多次观察；3）设计约束的Token‑Shard Mapping Matrix (cTMM)来防止单个token在多份排列中被重复计数，从而在短文本下提升解码准确率。

**🔧 技术方法**

采用哈希置换、词表分块、logit提升以及softmax采样等技术；解码时使用哈希恢复排列、构造cTMM并统计Shard计数以恢复消息。

**📊 数据集**

实验使用LLaMA‑2‑7B和LLaMA‑2‑7B‑chat在C4新闻、WritingPrompts、CNN/DailyMail和WMT14德英机器翻译数据集进行文本生成与翻译任务评估。

**📈 对比分析**

与DepthW、CycleShift、MPAC、RSBH、StealthInk等主流方法对比，XMark在bit accuracy上提升约5–10%（文本生成任务可达98.8%）且PPL、BERTScore、ROUGE等质量指标与或优于基线，且在token数受限、长消息、编辑攻击及长文本场景下仍保持较高的可靠性。

**⚠️ 局限性**

局限性：仅在d=2的设定下验证；k与δ等超参需要手动调优；未加入针对专门编辑或对抗攻击的鲁棒机制；对更大d的可扩展性与多语言适配尚待进一步研究。

---

## 179. Nash Approximation Gap in Truncated Infinite-horizon Partially Observable Markov Games

**arXiv ID:** 2604.05131 | [PDF](https://arxiv.org/pdf/2604.05131v1)

**作者:** Lan Sang `[一作]` (Johns Hopkins University), Chinmay Maheshwari `[通讯]` (Johns Hopkins University)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5031376831)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了对无限期部分可观测马尔可夫博弈（POMG）进行有限记忆截断的近似方法，将原本无限状态和动作空间的博弈转化为有限状态、有限动作的马尔可夫博弈。

**💡 创新点**

创新点在于：①定义了基于窗口的公共与私人信息截断表示，构造出可观测的有限状态；②在假设滤波器稳定性（遗忘性）下证明截断博弈的任何纳什均衡在原博弈中构成ε-纳什均衡，且ε随截断长度趋于0；③给出了严格的误差上界与渐近收敛分析。

**🔧 技术方法**

主要技术包括：公共信息框架（将POMG改写为对策博弈）、贝叶斯过滤与推断、有限记忆窗口截断、误差传播分析、全局与私有信念的动态更新、极大化与逼近误差评估。

**📊 数据集**

本文未使用具体数据集，而是以理论分析和假设（如Dobrushin系数<1）为基础。

**📈 对比分析**

通过理论证明比较了截断博弈与原博弈的价值差距，给出误差界为O(f(ℓ))，其中f(ℓ)为滤波器遗忘函数；实验或数值比较未给出，只给出了渐近收敛结果。

**⚠️ 局限性**

局限性包括：需要满足严格的滤波器稳定性假设；截断长度需足够大才能保证误差可接受；对异构记忆长度的情况虽有讨论但未给出完整算法；实际应用中需要解决如何估计或学习f(ℓ)以及如何构造有效的有限窗口策略。

---

## 180. Guidelines for Producing Concise LNT Models, Illustrated with Formal Models of the Algorand Consensus Protocol

**arXiv ID:** 2604.05006 | [PDF](https://arxiv.org/pdf/2604.05006v1)

**作者:** Hubert Garavel `[一作]` `[通讯]` (University of Grenoble Alpes), Hubert Garavel (University of Grenoble Alpes)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在论文中，作者对Algorand共识协议的LNT模型进行了多轮语义保持的转换，从原始967行的模型压缩到仅250行，并保持强分歧等性质。

**💡 创新点**

创新点在于提出了一套基于LNT语言结构和代数性质的四种通用简化变换，能够在不破坏强分歧的前提下显著降低代码量，并将过程计算机科学与可执行程序的语义桥接。

**🔧 技术方法**

采用了LNT语言与CADP工具链（TRIAN、LNT2LOTOS、BCG_CMP等）进行静态分析、等价检查、可视化检查以及基于MCL的模型检查，并利用LNT的循环与赋值语法实现过程重构。

**📊 数据集**

实验数据集为Algorand的四节点与二恶节点配置（A_4,0、A_2,2）以及更大规模的6、8、10节点网络，用于构建LTS并验证属性。

**📈 对比分析**

比较方法是按代码行数、LTS状态数与转换后强分歧保持性进行评估；结果显示从U0到U4代码量缩减约68%，状态空间缩减约30%，而等价检查与模型检查验证均成功。

**⚠️ 局限性**

局限性在于只对同步模型进行验证，未覆盖概率化的graded consensus阶段；对大规模网络的可扩展性仍待验证，且部分简化需要手动逐步检查保持分歧。

---

## 181. A Survey on Sensor-based Planning and Control for Unmanned Underwater Vehicles

**arXiv ID:** 2604.05003 | [PDF](https://arxiv.org/pdf/2604.05003v1)

**作者:** Shivam Vishwakarma `[一作]` (IIT Bombay), Leena Vachhani `[通讯]` (MANIT Bhopal)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了无人水下机器人（UUV）在复杂海洋环境下的基于传感器的规划与控制方法，提出了解耦与耦合架构的分类，并系统评估了PID、MPC和不变集（Invariant-Set）等控制器在多种场景中的性能。

**💡 创新点**

创新点包括：1）构建了解耦与耦合规划控制体系的完整分类框架；2）首次在同一集成规划器（IPC）下对多种控制器进行统一仿真对比；3）揭示了不变集控制在安全保障与轨迹平滑性方面的优势，并指出其对转弯半径的限制；4）提出了基于场景驱动的评测方法，为未来耦合体系的性能界定提供参考。

**🔧 技术方法**

使用技术包括：声纳、IMU、DVL等多模态传感器融合、SLAM、局部规划算法（DWA、RRT、MPC、IPC等）、高层与低层控制器设计（PID、MPC、Invariant-Set）以及完整的UUV导航栈架构。

**📊 数据集**

数据集：主要采用仿真生成的多种障碍场景（混乱障碍、窄通道、宽障碍、动态障碍、Figure‑of‑Eight路径）以及文献中引用的实验数据，没有公开单一公共数据集。

**📈 对比分析**

比较方法：在上述六大场景下对跟踪误差、目标到达时间、平均/最大加速度、平均/最大曲率等指标进行量化评估。结果显示：MPC在时间效率上领先；Invariant-Set在轨迹平滑性与安全性上表现最佳；PID在动态环境中最慢且波动大，易出现失稳。

**⚠️ 局限性**

局限性：MPC对预测时域和计算负荷敏感，易在高动态环境下失稳；Invariant-Set因最小转弯半径限制，难通过极窄通道；PID缺乏预测与约束处理，适用于简单线性路径；整体耦合架构在真实海况下的鲁棒性与性能尚未得到充分验证。

---

## 182. From Governance Norms to Enforceable Controls: A Layered Translation Method for Runtime Guardrails in Agentic AI

**arXiv ID:** 2604.05229 | [PDF](https://arxiv.org/pdf/2604.05229v1)

**作者:** Christopher Koch `[一作]` `[通讯]` (Independent Researcher), Christopher Koch (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种分层翻译方法，将ISO/NIST等标准的治理目标映射到治理层、设计层、运行时控制层和保证层，形成可执行的运行时护栏。

**💡 创新点**

创新点在于：①区分治理目标、技术控制、运行时护栏与保证证据；②设计控制元组和运行时可执行性打分法，用以判断何种控制适合放在运行时；③在采购代理案例中演示该方法，并给出评估维度。

**🔧 技术方法**

主要技术包括：结构化控制元组（〈a,x,r,ϕ,δ,ε,o〉）的定义、可执行性打分表（Timing, Observability, Determinacy, Judgment, Reversibility, Evidence）以及层次分配框架。

**📊 数据集**

本文未使用特定实验数据集；案例研究基于假想的企业采购代理情境。

**📈 对比分析**

由于是一种方法论论文，未进行实验对比。评估维度提出了五个方向（Policy Fidelity, Intervention Quality, Trajectory Coverage, Safety–Utility Trade‑off, Evidence Completeness），但实际性能需后续实证验证。

**⚠️ 局限性**

局限性：①依赖公开的标准条款摘要，未进行逐条解释；②方法为设计工具，缺乏实测验证；③在标准与技术细节的解释上仍需人工判断；④未给出完整的合规推断，仅提供结构化映射与评估框架。

---

## 183. Algebraic Structure Discovery for Real World Combinatorial Optimisation Problems: A General Framework from Abstract Algebra to Quotient Space Learning

**arXiv ID:** 2604.04941 | [PDF](https://arxiv.org/pdf/2604.04941v1)

**作者:** Min Sun `[一作]` (F. Hoffmann-La Roche AG), Tony Kam-Thong `[通讯]` (F. Hoffmann-La Roche AG)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种通用框架，用以识别组合优化问题中的代数结构（如单群），构造同构到布尔超立方体的表示，进一步构造商空间（等价类），并在商空间上设计结构感知的搜索算法，主要应用于患者亚组发现和分子筛选。

**💡 创新点**

创新点包括：
1) 证明基于逻辑与的规则组合构成单群，并通过特征向量映射与布尔超立方体的同构实现高效计算；
2) 通过商空间构造消除冗余表示，将搜索空间从指数级降至可管理的阶层；
3) 在此商空间上实现基于等价类的遗传算法（quotient‑aware GA），显著提升全局最优发现率。

**🔧 技术方法**

使用的技术包括：抽象代数（单群、商空间）、特征向量编码、布尔运算、Hamming 距离、DBSCAN 等价类聚类、遗传算法、贝叶斯优化（GP+Hamming 核）以及基准实验脚本。

**📊 数据集**

采用真实临床数据（含/不含数值特征）以及相同变量分布的合成数据进行实验；分子筛选案例仅做概念性说明，并未提供具体数据。

**📈 对比分析**

方法比较：
- quotient‑aware GA 在真实数据中全局最优发现率 48‑77%；
- 标准 GA 仅 35‑37%；
- BO 与贪婪算法发现率低于 3%；
- 运行时间方面，GA 约 45–60 秒；BO 约 13–28 秒；
- 结果表明利用代数结构可显著提升搜索效率与解的质量。

**⚠️ 局限性**

局限性：
1) 等价类的经验发现依赖 ϵ 阈值与聚类算法，可能产生不稳定或误差；
2) 仅适用于满足结合性与等价性（单群/半群）的规则组合问题，无法直接推广到非等价或非单调逻辑；
3) 对连续特征的处理仍需阈值离散化，可能失去细粒度信息；
4) 仅研究逻辑与形式，未覆盖更复杂的逻辑结构；
5) 需要手动调参（ε、群体大小、交叉/变异率）以获得最佳性能。

---

## 184. Polynomial and Pseudopolynomial Algorithms for Two Classes of Bin Packing Instances

**arXiv ID:** 2604.05152 | [PDF](https://arxiv.org/pdf/2604.05152v1)

**作者:** Renan Fernando Franco da Silva `[一作]` (University of Campinas), Manuel Iori `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 6036 | [OpenAlex ID](https://openalex.org/A5088224363)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对传统的装箱、切割与裁剪问题中最具挑战性的 AI 与 ANI benchmark 类别，设计了多种专门的算法，并证明 AI 类实例可在多项式时间内求解，ANI 类实例可在伪多项式时间内求解，进而快速求出最优解。

**💡 创新点**

创新点在于：
① 揭示 AI 与 ANI 类实例的结构性特征（如完全包装、可预知的三元组与必需权重），从而把原本 NP‑hard 的问题转化为可在多项式/伪多项式时间内解决；
② 结合动态规划、递归回溯、必需权重检测以及多重流 (Multiplicity Flow) 图构造，提出了一系列从最优 LP 解决到整数解的高效预处理流程；
③ 通过对实例的可行性快速判断与候选三元组的智能剪枝，显著降低搜索空间，实现了极高的计算效率。

**🔧 技术方法**

使用的技术包括：
- 强化线性松弛（SCF、AFF）与整数逼近特性（IRUP/MIRUP）
- 动态规划与递归回溯结合的三元组识别算法
- 必需权重（mandatory weight）与可满足子集检测
- 多重流图（Multiplicity Flow Formulation）与图上 DP
- 典型的分支与界、列生成、切割等 MIP 相关技术作为对照实验。

这些方法共同构成了完整的求解框架。

**📊 数据集**

主要数据集为：
- AI 与 ANI benchmark 共 500 个实例（每类 250 个），规模从 202 项到 1003 项不等；
- BPPLib、Random、Scholl、Falkenauer 等公开基准集，用于验证算法的局限性；
- 通过对 AI/ANI 的可行性检验，筛选出符合条件的子集。

**📈 对比分析**

与 2cLima、2cBaldacci、2cSilva 等现有最优算法（均基于分支与价格或分支与剪枝）进行比较。实验结果表明：
- AI 类实例平均耗时从 50–100 秒降至 0.1 秒（约 100 倍加速）；
- ANI 类实例平均耗时从 400–1600 秒降至 1–10 秒（约 100–500 倍加速）；
- 所有 500 个实例均在 1 小时内求解完毕；
- 传统算法在 13 个实例上仍未求解，而本方法全部求解成功。

总体而言，本方法在速度与可靠性上均显著优于现有方法。

**⚠️ 局限性**

限制与未来工作：
- 算法仅针对 AI 与 ANI 这两类特殊实例设计；对一般 BPP/CSP/SSP 实例仍属于 NP‑hard，无法直接适用；
- ANI 类的伪多项式算法只在 (α,β,3) 结构下可行，无法推广到更高维多元组；
- 对于满足 IRUP 的实例，算法需要先验证实例归属，否则会立即失败；
- 在极端规模或特殊构造的实例（如大量重复重量或非三元组结构）上，性能下降可能更明显。

未来需构造更具代表性的 benchmark，并进一步推广算法框架以覆盖更广泛的问题实例。

---

## 185. Governance-Aware Agent Telemetry for Closed-Loop Enforcement in Multi-Agent AI Systems

**arXiv ID:** 2604.05119 | [PDF](https://arxiv.org/pdf/2604.05119v1)

**作者:** Anshul Pathak `[一作]`, Nishant Jain `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了Governance‑Aware Agent Telemetry (GAAT) 框架，构建闭环治理感知代理遥测系统，实现多代理 AI 系统的实时策略检测与执行。

**💡 创新点**

创新点包括：① 在 OpenTelemetry 上扩展 Governance Telemetry Schema (GTS) 以携带治理元数据；② 采用 OPA 兼容声明式规则实现子 200 ms 的实时违规检测；③ 设计 Graduated Enforcement Bus (GEB) 进行分级干预；④ 引入 Trusted Telemetry Plane（签名、Bloom 过滤、Merkle 记录）保证遥测完整性与可审计性；⑤ 通过跨代理血缘追踪实现治理链路可见性，突破现有单代理边界检测的局限。

**🔧 技术方法**

技术栈包括 OpenTelemetry + GTS、OPA v0.60、Apache Kafka、ECDSA‑P‑256 签名、Bloom 过滤器、HMM 遗漏检测、Merkle 树审计日志、Go 实现 GEB、Python（LangChain）实现代理及遥测收集；整体运行于 Kubernetes (k3s) 集群。

**📊 数据集**

数据集与实验：① 合成注入流 5,000 条（10 次跑，500 条/跑）用于评估 4 种违规类型；② 12,000 条生产级真实轨迹用于验证；③ HMM 训练使用 8,000 条正常交互；④ 采用多种违规注入（CONSENT_MISSING、BIAS_THRESHOLD、DATA_RESIDENCY、UNAUTHORIZED_ACCESS）且违规率为 5%。

**📈 对比分析**

与四个基线（OT+Dashboard、NeMo Guardrails、CGW、Cedar）比较：GAAT 98.3 %±0.7% 的违规预防率（VPR），检测延迟 8.4 ms（P50），端到端执行延迟 127 ms；基线 VPR 仅 27.1–89.4%，延迟 15 s–340 ms；在 12,000 真实轨迹上 VPR 达 99.7%。统计检验 p<0.001，显著优于所有基线。

**⚠️ 局限性**

局限性：① 规模验证仅在 5–50 代理级别，未验证更大企业部署；② HMM 需要冷启动训练（约 8,000 条交互），无法在首次上线时直接使用；③ 内存与网络开销在生产环境未量化；④ 仅在 LangChain 框架下验证，AutoGen/CrewAI 等未测试；⑤ 违规注入样本为已知模式，未知或对抗性违规行为的性能未知；⑥ 依赖 PKI、CA 与密钥安全，若被破坏可导致遥测真实性失效。

---

## 186. Gradient-Controlled Decoding: A Safety Guardrail for LLMs with Dual-Anchor Steering

**arXiv ID:** 2604.05179 | [PDF](https://arxiv.org/pdf/2604.05179v1)

**作者:** Purva Chiniya `[一作]` (Amazon Alexa), Sagar Chaturvedi `[通讯]` (Amazon AGI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种梯度控制解码（GCD）框架，用于在大语言模型推理过程中动态评估并防止不安全提示，兼顾安全性和实用性。

**💡 创新点**

创新点在于双锚点梯度检测（同时使用“Sure”和“Sorry”两种响应的梯度）与确定性预置拒绝令牌的组合，既降低误报率，又通过预置令牌确保首词安全，克服了仅检测或仅过滤的缺陷。

**🔧 技术方法**

技术主要包括梯度相似度计算（余弦相似度）、安全关键参数筛选、双阈值判定以及在解码前插入预置令牌实现确定性拒绝。

**📊 数据集**

使用了三大安全基准数据集：ToxicChat、XSTest-v2 和 AdvBench，并在多种模型（LLaMA‑2‑7B、Mixtral‑8×7B、Qwen‑2‑7B）上进行评估。

**📈 对比分析**

与 GradSafe、Safe‑Decoding 等现有方法相比，GCD 在保持接近或更优召回率的同时，将误报率降低 52%（ToxicChat）并将攻击成功率降低多达 20%（最强基线）。

**⚠️ 局限性**

局限性包括对静态模板的依赖（可能对新颖提示或多轮对话效果不足）、额外梯度计算带来的延迟与内存开销，以及在多轮交互场景下的验证不足。

---

## 187. SMB algebras II: On the Constraint Satisfaction Problem over Semilattices of Mal'cev Blocks

**arXiv ID:** 2604.05161 | [PDF](https://arxiv.org/pdf/2604.05161v1)

**作者:** Petar Marković `[一作]` (University of Novi Sad), Aleksandar Prokić `[通讯]` (University of Novi Sad)

**通讯引用:** 444 | [OpenAlex ID](https://openalex.org/A5020050882)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文定义并研究了半格-马尔可夫块(SMB)代数，证明这些代数对固定模板的约束满足问题(CSP)具有可解性（多项式时间）。

**💡 创新点**

创新点在于首次统一并公开了作者对SMB代数可解性的原始证明，并修正了Bulatov在其可解性证明中的缺陷；同时展示了Bulatov与Zhuk在Dichotomy定理证明中的相似结构，提供了两种证明之间的桥接。

**🔧 技术方法**

主要技术包括：Tame Congruence Theory（包括类型1、5的分离与极大同构子），SMB代数的构造与正规化（regular SMB），(k,l)-minimal性与循环一致性算法，Z-不可约性与链接分区，重排与退化的可重构实例，以及对Bulatov和Zhuk可解性结果的组合与修正。

**📊 数据集**

无实验数据集，研究完全基于理论证明。

**📈 对比分析**

与先前Bulatov的证明相比，本文提供了更直接的修正方法，并通过Zhuk的算法框架验证了可解性；最终得到的算法在理论上属于多项式时间，复杂度取决于实例规模和代数大小，但未给出具体数值比较。

**⚠️ 局限性**

局限性：证明仍高度依赖于SMB代数的特殊结构（即半格-马尔可夫块的性质）及其正规化；对一般的Taylor代数仍未提供完整的可解性证明；修正过程仍需要引用Zhuk的完整 Dichotomy 证明，证明过程复杂，难以进一步简化。

---

## 188. GaussFly: Contrastive Reinforcement Learning for Visuomotor Policies in 3D Gaussian Fields

**arXiv ID:** 2604.05062 | [PDF](https://arxiv.org/pdf/2604.05062v1)

**作者:** Yuhang Zhang `[一作]` (Nanyang Technological University), Mir Feroskhan `[通讯]` (Nanyang Technological University)

**通讯引用:** 1364 | [OpenAlex ID](https://openalex.org/A5026643234)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了GaussFly框架，通过将3D Gaussian Splatting重建的真实场景与对比学习预训练的视觉编码器相结合，训练单目视觉的自主飞行策略，并实现从真实到仿真再到真实的零样本迁移。

**💡 创新点**

创新点包括：① 在3DGS重建中加入几何一致性约束以获得高保真渲染；② 用对比学习解耦感知与策略，得到噪声鲁棒的低维特征；③ 冻结编码器后大幅提升采样效率与域迁移能力。

**🔧 技术方法**

使用的技术包括3D Gaussian Splatting、SAM2资产分离、Isaac Sim物理仿真、对比学习（InfoNCE、ResNet‑50编码器）、PPO强化学习以及视觉注意力/Grad‑CAM分析。

**📊 数据集**

使用的数据集为从iPhone 17 Pro Max录制的室内视频重建的两套场景（Scene A、Scene B），并在这些场景中采集约1万张64×64的渲染图进行对比预训练；实际飞行测试使用DJI Tello Edu搭配OptiTrack/ UWB定位系统。

**📈 对比分析**

与D3QN、PPO、NPE、DAgger、Hybrid‑APF等基线在未见场景B进行零样本评估，指标包括OS、SR、CR、NE、SPL、TTS；GaussFly在SR、SPL上显著优于所有基线，样本效率提升约30%，在真实场景成功率分别为80%（Scene A）和65%（Scene B），展示出更好的泛化性能。

**⚠️ 局限性**

主要限制在于3DGS的静态光照处理，导致对光照变化的鲁棒性不足；此外，高动态环境的适应性尚待验证，未来计划加入可演化照明和视觉‑语言导航扩展。

---

## 189. RoboPlayground: Democratizing Robotic Evaluation through Structured Physical Domains

**arXiv ID:** 2604.05226 | [PDF](https://arxiv.org/pdf/2604.05226v1)

**作者:** Yi Ru Wang `[一作]` (University of Washington), Dieter Fox `[通讯]` (University of Washington)

**通讯引用:** 39881 | [OpenAlex ID](https://openalex.org/A5108257764)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于自然语言的结构化物理域评估框架，允许普通用户用语言描述、编译并执行可重复的机器人抓取任务，并支持任务族的系统化扩展。

**💡 创新点**

将评估过程从固定专家编写基准转为可由普通用户通过语言驱动的任务族生成，提供可复制、可扩展、结构化的评估空间，并通过版本化与上下文感知调优实现持续增长。

**🔧 技术方法**

利用大型语言模型进行任务解码与代码生成，结合多阶段验证与物理可实现性检查，使用固定任务schema、版本控制、上下文感知调优、语义嵌入距离测度等技术。

**📊 数据集**

在结构化块操作域构建任务族，并使用CuTAMP演示轨迹、GenSim、Cursor等基准，收集10位作者各自设计的50个任务用于评估多样性。

**📈 对比分析**

通过用户研究（SUS、NASA‑TLX、偏好测试）与六种不同策略模型在训练分布与生成的泛化任务上的成功率比较，展示本框架在可用性、认知负荷和诊断能力上优于传统基准。

**⚠️ 局限性**

受限于单一结构化块域，泛化到更复杂场景需要更细致的物理结构设计；LLM生成和验证仍可能产生不符合预期的任务；对非结构化描述的支持有限。

---

## 190. SpeakSoftly: Scaffolding Nonviolent Communication in Intimate Relationships through LLM-Powered Just-In-Time Interventions

**arXiv ID:** 2604.05382 | [PDF](https://arxiv.org/pdf/2604.05382v1)

**作者:** Ka I Chan `[一作]` (University of Michigan), Yuanchun Shi `[通讯]` (Tsinghua University)

**通讯引用:** 5409 | [OpenAlex ID](https://openalex.org/A5057896400)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了SpeakSoftly，一个基于NVC的LLM驱动的即时干预系统，用于在亲密关系的文本冲突中拦截言语攻击并提供改写建议和需求分析。

**💡 创新点**

提出三种渐进干预模式（Basic Reminder、Neutral Guide、Empathetic Guide），并验证情感友善语调对认知转变的重要性，强调即时干预与情感调节的结合。

**🔧 技术方法**

使用大语言模型Gemini 1.5 Flash进行即时文本检测与生成，配合React+FastAPI后端、WebSocket实时通信、LLM prompt工程化以及情绪检测与正反馈机制。

**📊 数据集**

基于9位经验配偶的半结构化访谈形成需求，18对情侣在实验中产生的对话数据作为评估样本；未使用公开数据集，所有数据均为实验自采集。

**📈 对比分析**

采用within‑subject混合方法研究，18对情侣在模拟与真实冲突中使用四种条件；通过Friedman检验与Wilcoxon配对比较，Empathetic Guide在模拟冲突中显著提升行为、认知和关系质量；Neutral Guide在真实冲突中更具可用性。

**⚠️ 局限性**

样本偏年轻单一文化，真实场景样本量不足，平台迁移导致使用门槛，LLM生成可能出现重复，需进一步进行长期效度评估与多平台集成。

---

## 191. Retrieve-then-Adapt: Retrieval-Augmented Test-Time Adaptation for Sequential Recommendation

**arXiv ID:** 2604.05379 | [PDF](https://arxiv.org/pdf/2604.05379v1)

**作者:** Xing Tang `[一作]` (Shenzhen Technology University), Xiuqiang He `[通讯]` (Shenzhen Technology University)

**通讯引用:** 7403 | [OpenAlex ID](https://openalex.org/A5083350101)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种名为ReAd的检索增强测试时适应框架，用于在实时推荐场景中动态调整已部署的顺序推荐模型，以应对用户偏好漂移。

**💡 创新点**

创新点在于：①构建协作记忆数据库，直接从历史序列检索相关物品；②设计轻量级检索学习模块，通过可学习的注意力融合检索到的物品嵌入；③采用基于熵的自适应融合机制，根据预测不确定性动态加权原始预测与检索增强预测。

**🔧 技术方法**

主要技术包括：序列编码器（Transformer/GRU等）、协作记忆检索（FAISS近似最近邻）、交叉注意力融合、对齐损失（KL散度）和熵权融合。

**📊 数据集**

实验使用五个公开基准数据集：Amazon Office、Beauty、Sports、Home（稀疏）和MovieLens 1M（较密集）。

**📈 对比分析**

与12种基线（包括GRU4Rec、SASRec、BERT4Rec、DuoRec、CL4SRec等）以及测试时训练、测试时增强、RaSeRec等方法对比，ReAd在所有数据集上均显著提升HR@K和NDCG@K，尤其在稀疏数据集上提升幅度更大。

**⚠️ 局限性**

局限性包括：检索操作仍带来额外延迟；当检索候选过多或无效时可能引入噪声；对超参数（检索数K、对齐损失权重λ、熵截断比例ρ）的敏感度需要手动调优；在极大规模商品目录下的检索效率仍有提升空间。

---

## 192. DAT: Dual-Aware Adaptive Transmission for Efficient Multimodal LLM Inference in Edge-Cloud Systems

**arXiv ID:** 2604.05375 | [PDF](https://arxiv.org/pdf/2604.05375v1)

**作者:** Qi Guo `[一作]` (Institute of Computing Technology, Chinese Academy of Science), Wen Ji `[通讯]` (Institute of Computing Technology, Chinese Academy of Science)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DAT 框架，在边缘-云系统中实现多模态大模型（MLLM）的高效推理与自适应多流传输，能在带宽受限环境下实现低延迟事件报警与高质量视觉证据补全。

**💡 创新点**

创新点包括：① 边缘轻量模型与云端大型模型协同的 cascade 推理，只有检测到可疑帧才触发深度推理；② 结合视觉引导与语义提示的 LoRA 微调策略，提升结构化事件理解与输出一致性；③ 以语义优先级和带宽动态为双重感知的 lexicographic 优化框架，并采用在线层次贪心调度实现实时多流自适应传输。

**🔧 技术方法**

核心技术包括：轻量目标检测模型（YOLOv12s）作为门控，Qwen2.5‑VL‑7B‑Instruct 作为大模型，LoRA 参数微调，基于语义优先级与链路状态的多流调度算法，lexicographic 优化与在线贪心调度。

**📊 数据集**

使用了交通事故检测数据集（Accidents Detection Dataset）与外部负样本集（Accident Detection From CCTV Footage）进行训练与验证，并使用 5G Traffic Datasets 生成的带宽轨迹评估传输性能。

**📈 对比分析**

与固定上传、仅带宽自适应、仅优先级自适应等基线相比，DAT 在事故识别准确率达到 98.83%，在严重拥塞场景下平均报警延迟下降 77.5%，并在 0.5 s 内成功补全 98.33% 的视觉证据，整体性能显著优于基线。

**⚠️ 局限性**

限制主要体现在：① 依赖边缘轻量模型的检测准确性，误检或漏检会影响大模型调用；② 需要对带宽状况进行实时监测，极端网络波动时调度可能受限；③ LoRA 微调和多模态大模型仍需一定的算力与存储资源，部署成本高；④ 目前仅在交通事故场景验证，跨领域泛化尚待进一步研究。

---

## 193. Rethinking IRSTD: Single-Point Supervision Guided Encoder-only Framework is Enough for Infrared Small Target Detection

**arXiv ID:** 2604.05363 | [PDF](https://arxiv.org/pdf/2604.05363v1)

**作者:** Rixiang Ni `[一作]` (National University of Defense Technology), Wei An `[通讯]` (National University of Defense Technology)

**通讯引用:** 7523 | [OpenAlex ID](https://openalex.org/A5032382317)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SPIRE框架，利用单点标注进行红外小目标检测，通过概率响应回归实现目标定位。

**💡 创新点**

创新点在于将目标检测改为中心点概率回归，并通过Point-Response Prior Supervision（PRPS）将单点标注转化为符合红外点扩散特性的高密度概率响应；同时采用高分辨率概率编码器（HRPE），实现编码器‑仅无解码器、无后处理的端到端检测。

**🔧 技术方法**

核心技术包括PRPS概率响应生成、HRPE轻量级高分辨率编码器、峰值提取与子像素细化、以及基于Gaussian热图的监督方式。

**📊 数据集**

在SIRST‑UAVB（3000张）与SIRST4（3352张）两个公开红外小目标基准集上进行实验。

**📈 对比分析**

与DNANet、SCTransNet等SOTA方法对比，SPIRE在SIRST‑UAVB上Precision 99.82%、F1 97.05%、Fa 1.02、Params 7.68M、FLOPs 7.68G；在SIRST4上Precision 95.00%、F1 94.60%、Fa 28.53、Params 7.68M、FLOPs 7.68G，显示出更低误报、更少参数与算力。

**⚠️ 局限性**

局限性包括：仍需精确的单点标注，可能对高密度聚集目标产生重叠峰值问题；在极低SNR或非Gaussian扩散场景下性能可能下降；尚未在多尺度或非红外模态上验证。

---

## 194. LatentAudit: Real-Time White-Box Faithfulness Monitoring for Retrieval-Augmented Generation with Verifiable Deployment

**arXiv ID:** 2604.05358 | [PDF](https://arxiv.org/pdf/2604.05358v1)

**作者:** Zhe Yu `[一作]` (Zhejiang University), Meng Han `[通讯]` (Zhejiang University)

**通讯引用:** 10890 | [OpenAlex ID](https://openalex.org/A5031867321)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个基于LLM内部残差流的实时可信度监测方法LatentAudit，用Mahalanobis距离判断生成答案是否与检索证据一致。

**💡 创新点**

创新点在于不需要额外判定模型、仅使用模型内部状态，使用简单的二次判定规则即可实现高精度监测，并可编译成零知识证明。

**🔧 技术方法**

技术包括中后期残差流提取、线性投影、Mahalanobis距离、岭回归校准，以及固定点量化与Groth16 ZK电路。

**📊 数据集**

实验数据集包含PubMedQA、HotpotQA、TriviaQA等问答基准，并构建四路压力测试。

**📈 对比分析**

与GPT‑4o、SelfCheckGPT、INSIDE、SAPLMA、Min‑Perplexity等基线对比，LatentAudit在多种模型（Llama‑2/3、Qwen、Mistral）上获得AUROC≈0.94，推理延迟≈0.8 ms，成本显著低于API基线。

**⚠️ 局限性**

局限性包括仅适用于开源权重模型，量化误差会影响阈值，无法验证检索内容真伪，零知识证明仅确认计算正确而非信度本身，且在极大模型上尚未验证。

---

## 195. Symetra: Visual Analytics for the Parameter Tuning Process of Symbolic Execution Engines

**arXiv ID:** 2604.05349 | [PDF](https://arxiv.org/pdf/2604.05349v1)

**作者:** Donghee Hong `[一作]` (Sungkyunkwan University), Jaemin Jo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 906 | [OpenAlex ID](https://openalex.org/A5102959274)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一套名为Symetra的可视化分析系统，用于支持人机协作下的符号执行引擎参数调优；

**💡 创新点**

其创新点在于首次针对符号执行参数空间构建多视图交互式分析框架，能够可视化参数重要性、覆盖向量、试验分组及代码层级覆盖差异，弥补了现有自动调参工具缺乏解释性和可操作性的缺陷；

**🔧 技术方法**

主要技术包括XGBoost+SHAP用于估计参数对分支覆盖的贡献、UMAP+Jaccard距离对覆盖向量降维、React.js/D3.js/visx实现六大视图交互、以及对试验结果的聚类与合并分析；

**📊 数据集**

使用KLEE在三大基准程序（gawk、gcal、grep）上进行实验，采集约61个参数、数千次试验的覆盖向量和覆盖值；

**📈 对比分析**

与全自动调参器（SymTuner）比较时，Human‑in‑the‑Loop调优在同一目标程序上实现27小时覆盖所需的覆盖率，仅需1小时完成，显示出显著的效率提升；

**⚠️ 局限性**

局限性包括未将可视化与调参器/引擎实时交互，仍需手动导入实验结果；依赖XGBoost+SHAP，缺乏其他解释方法；多视图布局学习成本高，且仅在少数程序和专家上验证，泛化性待进一步评估。

---

## 196. ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning

**arXiv ID:** 2604.05355 | [PDF](https://arxiv.org/pdf/2604.05355v1)

**作者:** Xuan Xiong `[一作]` (University of Toronto), Yang Wang `[通讯]` (Concordia University)

**通讯引用:** 40029 | [OpenAlex ID](https://openalex.org/A5078558986)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型的链式推理效率问题，本文通过优化不确定性趋势来压缩推理轨迹并保持准确率。

**💡 创新点**

创新点在于提出了基于不确定性趋势的奖励函数（Entropy Trend Reward, ETR），鼓励推理过程逐步收敛而非统一降低不确定性，并通过动量累积实现对轨迹的细粒度反馈。

**🔧 技术方法**

结合 Group Relative Policy Optimization (GRPO) 的强化学习框架，对推理模型进行训练，并在奖励中加入正确性监督与基于动量的熵趋势奖励。

**📊 数据集**

使用 DeepMath-103K 训练样本，评估数据集包括 AIME24、AMC23、MATH500 以及 GPQA-Diamond，覆盖高中到专家级推理任务。

**📈 对比分析**

在 DeepSeek-R1-Distill-7B、Qwen3-4B/8B 等模型上，与 DEER、NoThink、LCPO、O1-Pruner、PEAR 等训练无关和强化学习方法对比，ETR 在保持甚至提升准确率的同时将链式推理长度缩短 60% 以上，获得最高的 Accuracy–Efficiency Trade-off Score (AES)。

**⚠️ 局限性**

实验受限于 8B 参数模型的训练规模和资源，仅在 LoRA 微调下验证，缺乏对更大模型（如 13B 及以上）以及更广泛任务的评估。

---

## 197. Curr-RLCER:Curriculum Reinforcement Learning For Coherence Explainable Recommendation

**arXiv ID:** 2604.05341 | [PDF](https://arxiv.org/pdf/2604.05341v1)

**作者:** Xiangchen Pan `[一作]` (Huazhong University of Science and Technology), Wei Wei `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 254523 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Curr-RLCER框架，利用强化学习与课程学习（DPO+GRPO）训练大型语言模型，解决推荐评分与解释之间的一致性问题。

**💡 创新点**

创新点包括：①将多阶段课程学习与RLHF结合，逐步从CTR判断到评分预测再到开放式解释生成，缓解灾难性遗忘；②设计一致性奖励机制，结合情感分类与评分匹配；③构建完整的解释‑评分一致性评估方案，兼顾预测准确性与一致性。

**🔧 技术方法**

使用的技术有：大型语言模型（Qwen2.5‑3B/7B/14B）、强化学习框架（DPO、GRPO）、BERT情感分类、GPT‑3.5‑Turbo、文本相似度编码（m3e‑base）、多模态评估指标（GPTScore、BERTScore、BARTScore、BLEURT、USR）。

**📊 数据集**

实验数据集为Amazon Review Dataset的三个子集：Baby、Sports、Clothing（分别包含用户、物品、评分、说明）。

**📈 对比分析**

与Att2Seq、NRT、PETER、CER、XRec等基线以及不同参数规模的LLM进行对比。指标涵盖解释可解释性（GPTScore、BERTScore等）、稳定性、评分预测RMSE/MAE、解释‑评分一致性。结果显示Curr‑RLCER在解释质量、稳定性、一致性上均优于基线，并在评分预测上实现显著提升（RMSE下降34–42%，MAE下降58–60%）。

**⚠️ 局限性**

局限性包括：①对大规模算力依赖，训练成本高；②缺乏对跨域、低资源语言或少量数据场景的验证；③一致性评估仍受情感分类器与GPT判定的主观性影响；④长期一致性与用户满意度的真实影响尚未充分验证。

---

## 198. Reason Analogically via Cross-domain Prior Knowledge: An Empirical Study of Cross-domain Knowledge Transfer for In-Context Learning

**arXiv ID:** 2604.05396 | [PDF](https://arxiv.org/pdf/2604.05396v1)

**作者:** Le Liu `[一作]` (Harbin Institute of Technology), Danny Dongning Sun `[通讯]` (Pengcheng Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估跨域在上下文学习（ICL）的可行性，比较不同检索策略、模型规模和示例数量对跨域ICL性能的影响，并通过推理结构分析揭示提升来源。

**💡 创新点**

首次在ICL框架下验证跨域迁移的有效性，发现存在示例吸收阈值、模型规模决定正负迁移以及增示例效应；并提出“推理修复”机制说明提升主要来自结构兼容的示例。

**🔧 技术方法**

构建检索增强生成管线，使用 embedding、BM25 与 TopK+ConE 等检索方法；实验采用多规模 LLM（Qwen、Gemma、Llama 等）进行推理；对示例与目标任务的推理结构进行分类与统计。

**📊 数据集**

六个推理基准：GSM8K、ProntoQA、LogicalDeduction、FOLIO、ProofWriter、AR‑LSAT。

**📈 对比分析**

通过与零样本对比、精确匹配（EM）评价，并在不同模型规模和示例数下统计正负迁移情况。结果显示：大模型（>12B）跨域ICL普遍正转移，示例吸收阈值后增示例提升明显；小模型易出现负转移，增示例效果不稳定。

**⚠️ 局限性**

研究局限于选定的推理任务与检索策略，推理结构修复评估为经验式；未探讨解码策略与检索效果的交互；未显式优化检索多样性和结构对齐。

---

## 199. Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval

**arXiv ID:** 2604.05393 | [PDF](https://arxiv.org/pdf/2604.05393v1)

**作者:** Yuxin Yang `[一作]` (Institute of Automation Chinese Academy of Sciences), Weiming Hu `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了对象锚定式组合图像检索（OACIR）任务，构建了大规模多域真实图像检索基准OACIRR，并提出了AdaFocal框架以实现实例级检索；

**💡 创新点**

创新点在于：①将检索目标细化到实例层级，要求检索结果必须保持与参考图像中指定框内实例完全一致；②通过四阶段数据构建流程，生成含框定实例、修改文本的高质量四元组，并在检索库中加入针对性硬负样本；③提出上下文感知注意力调制器（CAAM），实现对锚定实例的动态注意力增强；

**🔧 技术方法**

使用的技术包括：多模态编码器（BLIP‑2 Q‑Former）、上下文感知注意力调制器（CAAM）与Transformer、注意力激活机制（在交叉注意力中注入调制因子）、对比对齐损失、可微调的视觉与文本特征融合；

**📊 数据集**

使用数据集：OACIRR，包含160k+四元组，覆盖Fashion、Car、Product、Landmark四大域；数据来源为DeepFashion2、Stanford Cars、Products‑10K、Google Landmarks v2；

**📈 对比分析**

与现有方法（通用多模态检索UMR、零样本CIR、监督CIR如SPRC）进行对比，AdaFocal在R@1、R_ID@1等指标均明显领先，尤其在实例召回率上提升约8%~12%，表明对实例级检索的显著优势；

**⚠️ 局限性**

局限性：仍存在实例召回与语义召回差距；依赖人工标注的框和硬负样本，难以迁移至更广泛或多实例场景；对极大规模检索或未见域的泛化能力尚未充分验证。

---

## 200. Data-Driven Function Calling Improvements in Large Language Model for Online Financial QA

**arXiv ID:** 2604.05387 | [PDF](https://arxiv.org/pdf/2604.05387v1)

**作者:** Xing Tang `[一作]` (Shenzhen Technology University), Xiuqiang He `[通讯]` (Shenzhen Technology University)

**通讯引用:** 7403 | [OpenAlex ID](https://openalex.org/A5083350101)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并应用了一套基于数据驱动的管线，提升大语言模型在在线金融问答系统中的函数调用能力。

**💡 创新点**

①提出针对金融场景的盲点检测与参数多样性扩增方法 AugFC；②设计两阶段训练（SFT+RL）兼顾推理准确性与推理效率；③将管线直接部署于商用金融问答平台，验证上线效果。

**🔧 技术方法**

使用大型预训练语言模型（Qwen2.5 系列）、自定义 Prompt 嵌入、强化学习（RLVR+GRPO）、信息熵与语义聚类、自动数据增强与验证流程。

**📊 数据集**

主要数据集：xLAM-60k、API-Bank、Tool-Alpaca、Seal-Tools、Nexus Raven、xLAM-small（采样）以及线上实时收集的用户查询与工具调用记录。

**📈 对比分析**

与现有方法（xLAM、Hammer、随机补全）比较，基准数据集上在 F1 分数、工具执行率等指标均优于对手；在线部署后自动化 F1 提升 39.1%、工具执行率提升 40.7%，人工评估最终答案准确率提升 20.3%，GBS 比例显著改善。

**⚠️ 局限性**

局限性：①对多工具链和跨模块调用的支持尚未充分验证；②数据增强过程中仍依赖强大语言模型生成，生成质量与覆盖面受限；③评估主要集中在单跳调用，对复杂多跳情境关注不足。

---

## 201. Transient Non-Use: How People in Migration Experience Digital Disconnection

**arXiv ID:** 2604.05386 | [PDF](https://arxiv.org/pdf/2604.05386v1)

**作者:** Jonathan Leuenberger `[一作]` (New Mexico State University), Shiva Darian `[通讯]` (New Mexico State University)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5056508881)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对32名跨境移民的深度访谈，系统探究了在移民迁移过程中信息与通信技术的非使用（Device、Informational、Protective三种形式）及其与迁移过渡阶段（理解、协商、解决）的关系。

**💡 创新点**

创新点在于将ICT非使用定位为迁移者在面对权力结构、风险与制度约束时的主动应对策略，并将其嵌入Ruthven的过渡理论框架与Satchell & Dourish的非使用类型中，扩展了HCI对“非使用”现象的理解。

**🔧 技术方法**

采用了质性研究方法，主要包括半结构式访谈、录音转写、文本编码和主题分析，并未使用技术算法。

**📊 数据集**

数据集为30份访谈记录（32名受访者），涵盖设备损失、信息误读、主动拒绝使用等非使用情境。

**📈 对比分析**

研究未与其他方法或系统进行对比评估，因此无性能指标；其贡献体现在理论和实践建议上。

**⚠️ 局限性**

局限性包括样本规模相对较小、仅聚焦于美国-墨西哥边境的特定迁移情境，且访谈数据可能受到受访者回忆与自我呈现偏差影响。

---

## 202. Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval

**arXiv ID:** 2604.05383 | [PDF](https://arxiv.org/pdf/2604.05383v1)

**作者:** Jianzhi Yan `[一作]` (Harbin Institute of Technology), Danny Dongning Sun `[通讯]` (Pengcheng Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于跨域不变神经元的检索方法 DIN-Retrieval，用于提升 LLM 在无专家标注情况下的跨域推理能力。

**💡 创新点**

首次发现并利用域不变神经元（DIN）进行示例检索，实现结构相似的跨域示例选择，从而显著提升 ICL 的跨域鲁棒性。

**🔧 技术方法**

使用跨域 z‑score 聚类识别 DIN，构建 DIN 向量进行余弦相似度检索，并结合 MMR 进行示例多样化；同时通过剪枝实验验证 DIN 的功能重要性。

**📊 数据集**

在数学推理 GSM8K、逻辑推理 PrOntoQA 以及 FOLIO 等任务上进行源域 → 目标域迁移实验。

**📈 对比分析**

与 Zero‑shot、X‑ICL、Set‑BSR 等基线对比，平均提升 1.8%（最大 3%）的准确率，且在多模型（LLaMA、Gemma、Qwen 等）和多方向迁移上均优于基线。

**⚠️ 局限性**

方法仅基于简单的极性一致性规则且阈值固定，实验范围局限于数学与逻辑领域，增益相对有限，且缺乏因果性证明。

---

## 203. LMI-Net: Linear Matrix Inequality--Constrained Neural Networks via Differentiable Projection Layers

**arXiv ID:** 2604.05374 | [PDF](https://arxiv.org/pdf/2604.05374v1)

**作者:** Sunbochen Tang `[一作]` (Massachusetts Institute of Technology), Navid Azizan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 408 | [OpenAlex ID](https://openalex.org/A5005748450)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可微分投影层 LMI-Net，使得神经网络输出始终满足线性矩阵不等式（LMI）约束。

**💡 创新点**

创新点在于将 LMI 约束拆分为仿射等式和正定锥，利用 Douglas–Rachford 分裂实现高效前向投影与隐式微分的后向传播，并给出收敛性保证。

**🔧 技术方法**

主要技术包括 Douglas–Rachford 分裂、正定锥的 Eigenvalue 裁剪投影、线性等式的闭式投影、以及基于隐式微分的梯度计算。

**📊 数据集**

使用两类人工生成的线性系统数据集：受扰线性系统的椭圆不变集问题和联合控制器与不变集设计问题，数据通过随机生成矩阵 A、B、B_w 并控制特征值分布得到。

**📈 对比分析**

与软约束模型和传统 CVXPY/SCS 求解器比较，LMI-Net 在分布外样本下约束满足率显著提升（完全消除约束违例），并且计算时间比 SCS 快 10–35 倍；在控制器设计任务中，LMI-Net 能消除闭环不稳定。

**⚠️ 局限性**

局限在于目前仅在低维（n=2）问题验证，缺乏对更高维系统的实验，且尚未验证在实际控制任务（如管道 MPC 或收缩度量控制）中的表现。

---

## 204. From Retinal Evidence to Safe Decisions: RETINA-SAFE and ECRT for Hallucination Risk Triage in Medical LLMs

**arXiv ID:** 2604.05348 | [PDF](https://arxiv.org/pdf/2604.05348v1)

**作者:** Zhe Yu `[一作]` (Binjiang Institute of Zhejiang University), Meng Han `[通讯]` (Binjiang Institute of Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了基于视网膜分级记录的证据驱动风险三叉评估基准 RETINA‑SAFE，并提出了两阶段白盒风险分层框架 ECRT，用内部表示和对比（CTX/NOCTX）来检测医学 LLM 的幻觉风险。

**💡 创新点**

创新点包括：①将幻觉风险转化为证据关系三叉（E‑Align、E‑Conflict、E‑Gap）；②设计了结构化的证据标签与诊断流程对应的基准；③提出利用内部层级差异（logit、隐藏状态、Token KL）构建两阶段风险分层，既判定安全与否，又区分冲突与信息缺失；④在多种 LLM 上采用分组无泄漏协议展示了显著的平衡准确率提升。

**🔧 技术方法**

技术手段：白盒内部特征抽取（Δlogits、Δtraj、Δinc）；两阶段分类器（Stage‑1 识别 unsafe/ safe，Stage‑2 区分 contradiction vs gap）；XGBoost 训练；对比 CTX/NOCTX 输入；与多种外部不确定性/自一致性基线对比。

**📊 数据集**

使用 12,522 条来自浙江大学视网膜分级记录的样本，构成三类标签；样本包含临床式问题、四选项、金标准答案与结构化视网膜证据；采用分组分层拆分保证无证据泄漏。

**📈 对比分析**

与多种外部基线（如 Focus、EigenScore、UQ Heads 等）以及单阶段白盒基线进行比较。在 目标召回率 0.95 的高召回场景下，ECRT 在 Stage‑1 平衡准确率上提升 +0.15~+0.19，且相对于最强监督适配基线提升 +0.02~+0.07，且在 Stage‑2 维持了较高的冲突与缺失召回率。

**⚠️ 局限性**

局限性：仅基于结构化的视网膜描述而非直接像素级多模态输入，可能限制对复杂图像特征的捕捉；目前仅验证于糖尿病视网膜病变场景，缺乏跨疾病和跨模型的通用性评估；内部信号对模型架构的依赖较高，迁移到不同 LLM 需要进一步调优。

---

## 205. Dynamic Agentic AI Expert Profiler System Architecture for Multidomain Intelligence Modeling

**arXiv ID:** 2604.05345 | [PDF](https://arxiv.org/pdf/2604.05345v1)

**作者:** Aisvarya Adeseye `[一作]` (University of Turku), Mohammad Tahir `[通讯]` (University of Turku)

**通讯引用:** 1965 | [OpenAlex ID](https://openalex.org/A5035027638)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种基于 LLM 的 Agentic AI 专家等级评估系统，该系统通过自然语言回应自动分类用户的专业水平，支持静态离线评估与动态实时访谈两种模式。

**💡 创新点**

创新点在于：① 模块化分层架构，将文本预处理、特征评分、聚合与分类拆分为独立可扩展的子模块；② 设计了五个并行特征评分器（词汇、深度、应用、严谨、不确定），并通过加权聚合得到最终专家分数；③ 引入可解释的 justification 生成器和自适应提问机制，实现对访谈进程的实时动态调整。

**🔧 技术方法**

主要技术：使用 LLaMA v3.1 (8B) 作为本地 LLM 进行文本分析；实现多层处理管线（Input → Preprocessing → Feature Scoring → Aggregation → Classification → Output）；采用 JSON 格式输出结构化结果，结合报告生成器提供易读解释。

**📊 数据集**

数据集：静态评估基于 82 名参与者的录音访谈文本，涵盖安全、隐私与游戏化三个主题；动态评估基于 402 名参与者的 3 轮实时访谈，主题为安全、隐私与 LLM 认知，收集逐轮回应。

**📈 对比分析**

比较方法与性能：将系统评分与参与者自评进行逐级对比；在静态模式下，83%–97% 匹配率；在动态模式下，97%–95% 匹配率，且大多数在第 3–4 题后达到稳定匹配；系统在技术性强的安全领域表现最优，在主观性高的隐私和 LLM 认知领域需要更多问题才可精确匹配。

**⚠️ 局限性**

局限性：对模糊或高度主观的回答易产生误判；在隐私、LLM 认知等解释性强的主题中需要多轮问答才能达到准确匹配；对 LLM 误读细微语义的敏感性；缺乏对跨语言或多领域的广泛验证。

---

## 206. OGA-AID: Clinician-in-the-loop AI Report Drafting Assistant for Multimodal Observational Gait Analysis in Post-Stroke Rehabilitation

**arXiv ID:** 2604.05360 | [PDF](https://arxiv.org/pdf/2604.05360v1)

**作者:** Khoi T. N. Nguyen `[一作]` (Nanyang Technological University), Baosheng Yu `[通讯]` (Nanyang Technological University)

**通讯引用:** 6190 | [OpenAlex ID](https://openalex.org/A5085309099)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并评估了一个多智能体 AI 系统 OGA‑AID，用于在卒中后步态分析中协助临床医生生成结构化评估报告；

**💡 创新点**

创新点在于将视频、运动捕捉轨迹与临床信息通过分工的三大智能体（观测器、轨迹分析器、报告生成器）协同处理，并支持临床医生的“人机对话”式干预；

**🔧 技术方法**

使用多模态大型语言模型（GPT‑5.1、Gemini‑3 Flash 等）与视觉语言模型；采用视频帧抽样、面部模糊、轨迹时序化等预处理；通过提示工程实现因子级评估与报告模板生成；

**📊 数据集**

基于 15 名卒中后患者的 45 次步态测量（Qualisys Miqus‑M3 运动捕捉系统），并结合 50 名规范样本与 Wisconsin Gait Scale 作为评估基准；

**📈 对比分析**

通过与单通 LLM 基线对比（MAE、Max AE、Bias、MCID 等指标）验证：多智能体方案在 MAE 上显著优于基线，GPT‑5.1 与 Gemini‑3 Flash 的 MAE 均低于 2.25 的 MCID，表明临床可接受；在临床医生提供初步观察后，MAE 进一步下降约 22.7%；

**⚠️ 局限性**

局限性包括样本量有限、缺乏大规模专家标注文本、Max AE 仍较大提示极端误差风险、系统可解释性不足以及对不同设备与环境的泛化能力尚待验证。

---

## 207. A Theoretical Framework for Statistical Evaluability of Generative Models

**arXiv ID:** 2604.05324 | [PDF](https://arxiv.org/pdf/2604.05324v1)

**作者:** Shashaank Aiyer `[一作]` (University of Maryland), Han Shao `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2`

**🎯 论文内容**

提出统计可评估（evaluability）框架，系统分析了生成模型评估指标的可评估性，并给出正负评估结果；

**💡 创新点**

首次给出IPM、Rényi散度、coverage profile等指标的可评估性定理，并揭示了perplexity等传统分数函数与这些指标的本质不匹配；

**🔧 技术方法**

使用统计学习理论中的VC维、fat-shattering维度、均匀收敛与估计复杂度分析等方法；

**📊 数据集**

无实际数据集，全部为理论分析与构造对比例子；

**📈 对比分析**

与现有评估方法（如基准、人工评估）不同，本文提供了理论上可保证模型排名准确性的评估指标，强调评估指标与分数函数的几何对齐；

**⚠️ 局限性**

局限在于对某些指标仅给出负评估（不可评估），对β-限制KL等仍未给出完整评估理论，且实际应用需先满足特定假设（如ratio‑closeness、margin等）。

---

## 208. Weather-Conditioned Branch Routing for Robust LiDAR-Radar 3D Object Detection

**arXiv ID:** 2604.05405 | [PDF](https://arxiv.org/pdf/2604.05405v1)

**作者:** Hongsheng Li `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**通讯引用:** 7900 | [OpenAlex ID](https://openalex.org/A5012419026)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种天气条件化分支路由框架，实现 LiDAR、4D 雷达及其融合特征的动态加权，以提高在恶劣天气下的 3D 目标检测性能。

**💡 创新点**

将多模态感知视为天气条件化的分支路由问题；通过视觉+语义天气提示生成条件令牌；引入天气监督的辅助分类与多样性正则化，避免分支坍塌，并实现可解释的模态偏好。

**🔧 技术方法**

使用 3D 稀疏卷积编码、条件门控 KNN 注意力、视觉 CNN+CLIP 文本编码生成条件令牌、MLP 路由器；训练目标包括检测损失、天气分类损失、多样性正则化和熵正则化。

**📊 数据集**

K-Radar 基准数据集（包含 LiDAR、4D 雷达、相机及多种天气标签）。

**📈 对比分析**

在 K-Radar 上与 LiDAR‑only、雷达‑only、InterFusion、3D‑LRF 等多种基线对比；在 IoU=0.3 和 0.5 的 AP_BEV 与 AP_3D 指标上均实现 SOTA，提升幅度从 +1.2/+6.4 到 +7.4/+13.5；在各类恶劣天气下均表现显著优于现有方法。

**⚠️ 局限性**

需要训练时提供天气标签；在极端天气或缺少天气信息时路由可能不稳定；多分支架构增加了计算与实现复杂度；目前仅在 K-Radar 数据集上验证。

---

## 209. Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning

**arXiv ID:** 2604.05404 | [PDF](https://arxiv.org/pdf/2604.05404v1)

**作者:** Qisheng Su `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 13189 | [OpenAlex ID](https://openalex.org/A5102740754)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对工具集成推理（TIR）的硬件感知效率指标PTE（Prefill Token Equivalents），用于统一内部推理和外部工具调用的成本；

**💡 创新点**

PTE通过量化预填充和解码阶段的计算与内存瓶颈，捕捉KV‑Cache evict与工具响应长度对推理延迟的非对称影响，从而实现比传统基于token或调用次数的指标更贴近真实时延；

**🔧 技术方法**

基于Transformer推理的首尾阶段成本模型，利用硬件操作强度（HOI）计算γ系数，将内存受限的解码成本转换为等价的计算成本，并将其与预填充成本相加；

**📊 数据集**

在五个TIR基准上评估：MATH500、AIME 2024/2025、SimpleQA、WebInstruct-Verified，收集模型在不同工具集上的推理轨迹；

**📈 对比分析**

将PTE与实际墙钟时延及传统token计数进行对比，发现PTE与延迟相关性显著高于token计数（r≈0.93 vs r≈-0.38），并在多种硬件平台上保持一致的模型排名；

**⚠️ 局限性**

局限包括：忽略API调用延迟、γ系数为简化抽象，未覆盖所有硬件细节；实验范围仅限部分任务与模型，需进一步验证PTE与推理质量关联的普适性；

---

## 210. PROMISE: Proof Automation as Structural Imitation of Human Reasoning

**arXiv ID:** 2604.05399 | [PDF](https://arxiv.org/pdf/2604.05399v1)

**作者:** Youngjoo Ahn `[一作]` (Yonsei University), Jieung Kim `[通讯]` (Yonsei University)

**通讯引用:** 657 | [OpenAlex ID](https://openalex.org/A5015256366)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种结构感知的自动化证明生成框架 PROMISE，通过检索与目标证明状态相似的证明状态转移模式，结合名称检索和基于束搜索的迭代生成，显著提升了在大型系统验证项目中的证明生成成功率。

**💡 创新点**

创新点在于：①以证明状态转移轨迹为检索单元，实现结构化检索；②同时进行结构检索与名称检索，提供更精准的上下文；③利用束搜索与自适应评分，将检索结果转化为状态感知的证明生成策略；④在提示工程中注重多样化与方法多样性。

**🔧 技术方法**

技术包括：大语言模型（GPT‑3.5‑Turbo、GPT‑4.1、Qwen2.5‑Coder‑7B‑Instruct）与Isabelle/HOL 的Scala‑Isabelle桥接；结构检索使用嵌入向量与语义重排；名称检索聚合多源词表；束搜索、候选归一化、静态过滤和机器检查保证证明正确性；自适应束宽度与候选预算。

**📊 数据集**

使用 seL4 项目的 223 条 Isabelle/HOL 定理（来源于 7 个子模块），对比标准检索+生成基线 Selene、Rango 等。

**📈 对比分析**

与 Selene、Rango 进行系统对比，PROMISE 在 Qwen2.5‑Coder‑7B‑Instruct、GPT‑3.5‑Turbo、GPT‑4.1 三种 LLM 下均取得最高成功率（分别为 77%、85% 与 69%），整体提升 20‑47%（例如对 Selene‑ACC5 提升 +47/ +34 ）。仅在 GPT‑4.1 的 P2 任务中略逊于 Rango。

**⚠️ 局限性**

局限性包括：对检索数据库构建的依赖，可能在大规模项目中检索成本较高；在极强模型（如 GPT‑4.1）下提示过于约束导致性能略低；仍需要大量 LLM 计算资源；对跨助手迁移的通用性尚未验证。

---

## 211. Cross-Machine Anomaly Detection Leveraging Pre-trained Time-series Model

**arXiv ID:** 2604.05335 | [PDF](https://arxiv.org/pdf/2604.05335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 212. Confidence Should Be Calibrated More Than One Turn Deep

**arXiv ID:** 2604.05397 | [PDF](https://arxiv.org/pdf/2604.05397v1)

**作者:** Zhaohan Zhang `[一作]` (Queen Mary University Of London), Ioannis Patras `[通讯]` (Queen Mary University Of London)

**通讯引用:** 12146 | [OpenAlex ID](https://openalex.org/A5031205865)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了多轮对话中的置信度校准任务，并设计了新指标ECE@T，用以评估模型在每个对话轮次的校准情况；针对用户劝导导致校准恶化的问题，作者通过训练一个轻量级的多层感知机探针，最小化ECE@T的代理目标，从对话历史中学习动态置信度，并将校准后的置信度融入解码策略，以提升多轮对话的事实准确性和对抗劝导的鲁棒性。

**💡 创新点**

创新点在于将校准从静态单轮任务转化为动态多轮任务，引入ECE@T指标；设计了基于代理目标的多轮校准损失，利用隐藏层信息训练校准探针；提出了置信度驱动的解码策略，使模型在后续轮次能够基于初始响应的置信度做出更稳健的回答。

**🔧 技术方法**

核心技术包括：①使用多层感知机（MLP）探针提取LLM隐藏状态以预测置信度；②构造代替目标（按置信度分箱的组准确率）来最小化ECE@T；③在生成阶段将预测概率与校准置信度结合（s_t(y)=λp̂_t(y)+(1-λ)c_t(y)），并在后续轮次融合首轮置信度与当前置信度的分数；④利用Brier、smECE等指标进行评估。

**📊 数据集**

实验使用的对话数据集为TriviaQA、SciQ、NQ；评估模型包括Llama3.1-8B-Instruct、Qwen2.5-7B-Instruct、Gemma2-9B-it。

**📈 对比分析**

与传统单轮校准方法（Sequence Likelihood、Platt Scaling、Self-Consistency、Verbal、P(True)）以及DCal相比，作者提出的方法在ECE@1、ECE@2和ECE@D上均显著降低误差，ECE@2下降≈1%并保持稳定；在多轮对话中的事实准确率和对劝导的鲁棒性也得到提升，尤其在第1轮和第2轮对比中平均提高1.3%准确率。

**⚠️ 局限性**

主要局限包括：需要白盒访问LLM隐藏状态，无法直接应用于封闭源模型；仅提供单一整体置信度，无法细粒度评估多声明文本；在解码阶段需对多候选词进行置信度评估，导致推理效率下降。

---

## 213. LLM-as-Judge for Semantic Judging of Powerline Segmentation in UAV Inspection

**arXiv ID:** 2604.05371 | [PDF](https://arxiv.org/pdf/2604.05371v1)

**作者:** Akram Hossain `[一作]` (University of Southern Mississippi), Kareem Abdelfatah `[通讯]` (Fayoum University)

**通讯引用:** 156 | [OpenAlex ID](https://openalex.org/A5064355644)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了将大型多模态语言模型（LLM）用作无人机电力线路分割结果的语义评估者，并在无人机场景中构建了可重复性和敏感性评估框架。

**💡 创新点**

提出了两项创新：①把LLM作为离线语义安全监控者（watchdog）而非主感知引擎；②从可重复性和对视觉退化的敏感性两个维度量化LLM判断的一致性与鲁棒性。

**🔧 技术方法**

使用 GPT‑4o 作为多模态LLM，对分割遮罩与 RGB 图像叠加进行推理，输出质量分数、置信度和文本解释；通过固定 prompt、重复推理、合成视觉噪声（雾、雨、雪、阴影、眩光）来评估模型稳定性与敏感性。

**📊 数据集**

基于 TTPLA（电力线路 UAV 数据集）构建挑战集，利用 Albumentations 等库在三种严重度下合成多种视觉扰动，保留原图与掩码的像素级对应。

**📈 对比分析**

采用指标 A_s、A_c、A_{s,c}、ICC(1,1) 评估重复性；采用 Δs、Δc、Spearman ρ、效应量 d_z 评估敏感性。结果显示：分数一致率 78–91%，ICC 0.858–0.917；置信度随雾、雨、雪等退化显著下降，且 Δs 与 Δc 随严重度单调增加，d_z 证明统计显著性，证明 LLm‑as‑Judge 具备可靠且保守的判断。

**⚠️ 局限性**

局限性包括：①LLM 仍受解码随机性影响，文本解释不一致；②评估基于合成噪声，真实环境的极端条件（光照极端、运动模糊、遮挡）仍未充分验证；③依赖 GPT‑4o 等高成本模型，未探讨更轻量化方案；④缺乏与人工专家的对齐验证，可能存在偏差。

---

## 214. Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation

**arXiv ID:** 2604.05343 | [PDF](https://arxiv.org/pdf/2604.05343v1)

**作者:** Boyu Cao `[一作]` (South China University of Technology), Qi Liu `[通讯]` (South China University of Technology)

**通讯引用:** 29951 | [OpenAlex ID](https://openalex.org/A5100453264)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了Anchored Cyclic Generation（ACG）范式和其层级化实现Hi-ACG框架，用以提升长序列符号音乐生成的结构连贯性与音质。

**💡 创新点**

创新点包括：①利用已生成的音符作为锚点（anchor）在循环生成中校正错误，显著降低自回归模型的误差累积；②通过两级语义预测与重建的两阶段生成流程，提升时间复杂度；③引入层级化Sketch‑Refinement循环，实现全局结构先行、局部细化的生成策略；④设计高效的Piano Token表示，压缩序列长度同时保留时空信息。

**🔧 技术方法**

核心技术为Transformer解码器（语义预测与重建）、全连接重嵌入层、Piano Token分块表示、层级化生成循环；实验中使用信息熵、谐音一致性、跳跃率等指标与Qwen3 LLM评估相结合，对比传统Transformer与扩散模型。

**📊 数据集**

使用的公开数据集包括MuseScore（约14万首双轨钢琴曲，时长1–3分钟）和POP909，均转化为Piano Token形式进行训练与评估。

**📈 对比分析**

通过与Music Transformer、BPE Transformer及Cascaded‑Diff的客观指标（音高熵、节奏熵、和声一致性、旋律跳跃率）和主观MOS评估对比。结果显示：Hi-ACG在长序列生成中实现平均34.7%余弦距离下降，主观MOS达到约3.0，显著优于基线（1.96–2.05）并接近真实曲目（3.31）。

**⚠️ 局限性**

局限性在于缺乏细粒度控制（难以动态调节个性化参数）、Piano Token可能丢失细微时值信息、仅限钢琴双轨，未覆盖多轨与多乐器/音色，未来计划加入表达性与结构性token并推广到更广泛的生成任务。

---

## 215. LSGS-Loc: Towards Robust 3DGS-Based Visual Localization for Large-Scale UAV Scenarios

**arXiv ID:** 2604.05402 | [PDF](https://arxiv.org/pdf/2604.05402v1)

**作者:** Xiang Zhang `[一作]`, Zongqian Zhan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类精度上提高了5%。

**⚠️ 局限性**

模型在处理大规模数据集时可能会遇到计算资源不足的问题。

---

## 216. HYVE: Hybrid Views for LLM Context Engineering over Machine Data

**arXiv ID:** 2604.05400 | [PDF](https://arxiv.org/pdf/2604.05400v1)

**作者:** Jian Tan `[一作]` (Cisco Systems), Li Zhang `[通讯]` (Cisco Systems)

**通讯引用:** 7124 | [OpenAlex ID](https://openalex.org/A5105983745)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对包含大量机器数据的LLM输入，提出HYVE框架通过结构化提取、混合列行视图、可恢复的存储以及有限的SQL后处理，实现上下文工程与数据完整性兼顾；

**💡 创新点**

创新点在于：①基于数据库原理的混合列行视图和可恢复的请求级数据存储；②不使用语义压缩而是结构化截断与检索式排名；③通过可选的SQL后处理实现数据回填与语义合成；

**🔧 技术方法**

核心技术包括：JSON/AST解析、结构一致性检测、列化与行化视图生成、基于BM25的参考引导排名、TOON/JSON序列化、DuckDB请求级内存数据库、SQL工具调用与模板回填；

**📊 数据集**

使用多种真实网络工作负载数据集：Cert-QA、Runbook、Line/Bar图表生成、Anom异常检测、Sum报告、Canvas槽填、RB-Text/JSON、Hard多跳推理、TOON-QA；

**📈 对比分析**

与GPT‑4.1、GPT‑5等基线对比，HYVE在大多数任务上减少Token使用50–90%，同时保持或提升答题质量；在图表生成、异常检测、查询推理等场景中，准确率提升至90%+，Latency下降30–80%；

**⚠️ 局限性**

局限性在于：仅提供短期请求内存，无法跨会话保持；对非JSON结构化日志支持有限；对极端大规模数据仍需进一步优化；模型对TOON编码兼容性不佳，可能影响生成质量；

---

## 217. Neural Assistive Impulses: Synthesizing Exaggerated Motions for Physics-based Characters

**arXiv ID:** 2604.05394 | [PDF](https://arxiv.org/pdf/2604.05394v1)

**作者:** Zhiquan Wang `[一作]`, Bedrich Benes `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998`

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

## 218. LUMOS: Universal Semi-Supervised OCT Retinal Layer Segmentation with Hierarchical Reliable Mutual Learning

**arXiv ID:** 2604.05388 | [PDF](https://arxiv.org/pdf/2604.05388v1)

**作者:** Yizhou Fang `[一作]`, Xiaoying Tang `[通讯]` (Southern University Of Science And Technology)

**通讯引用:** 5490 | [OpenAlex ID](https://openalex.org/A5001406512)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了LUMOS，一种半监督通用OCT视网膜层分割框架，能够在多粒度标签环境下统一学习。

**💡 创新点**

核心创新点包括：① 双解码器网络与层级提示策略（DDN‑HPS）实现多粒度提示与伪标签噪声抑制；② 可靠进化多粒度学习（RPML）通过区域可靠性加权与多级预测融合实现跨粒度一致性，且采用逐层难度提升训练；③ 将双分支互相学习与层级提示结合，打破传统单分支、单粒度的限制。

**🔧 技术方法**

使用的主要技术包括：混合CNN–Transformer编码器、双分支解码器（可学习上采样与固定双线性插值）、Transformer提示解码器、双向伪标签一致性损失、区域可靠性加权、预测融合、聚多级逐步训练策略，整体实现为半监督学习框架。

**📊 数据集**

采用六个OCT数据集：内部数据集HC‑MS（8层），GCN（8层），OCTA‑500（5层，未标注）；外部测试集HEG（7层）、Goals（3层）、AMD（2层）、OIMHS（1层）。

**📈 对比分析**

与Mean Teacher、CCT、MinEnt、DiffRect、ABD、CGS等现有半监督方法对比。LUMOS在内部测试集平均DSC达90.84%（HC‑MS）/81.72%（GCN），在外部数据集平均DSC 86.05%，在所有数据集的DSC与HD95均优于第二名至少2–6个百分点，显示出显著的跨粒度与跨域泛化能力。

**⚠️ 局限性**

局限性在于对严重病理变形的处理不足，框架在极端解剖畸形或病灶严重破坏的OCT图像中性能仍有限。

---

## 219. ICR-Drive: Instruction Counterfactual Robustness for End-to-End Language-Driven Autonomous Driving

**arXiv ID:** 2604.05378 | [PDF](https://arxiv.org/pdf/2604.05378v1)

**作者:** Kaiser Hamid `[一作]` (Texas Tech University), Nade Liang `[通讯]` (Texas Tech University)

**通讯引用:** 432 | [OpenAlex ID](https://openalex.org/A5032421589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了ICR-Drive框架，系统评估语言指令对端到端驾驶模型鲁棒性的影响，通过在CARLA固定路线下生成四类指令对照；

**💡 创新点**

提出四类指令扰动（Paraphrase、Ambiguity、Noise、Misleading）以及配对评估协议，揭示语言指令对驾驶性能的敏感性；

**🔧 技术方法**

使用大规模预训练视觉语言模型与语言模型（如LMDrive、BEVDriver）、模板生成对照指令、CARLA闭环仿真及LeaderBoard指标；

**📊 数据集**

利用LangAuto（Tiny 与 full）数据集，在CARLA 8 城市、21 环境条件下的导航指令；

**📈 对比分析**

在每条路线上固定环境，计算指标差值；结果显示LMDrive对指令变化敏感，误导指令导致性能显著下降；BEVDriver对噪声更鲁棒，但仍受误导影响；

**⚠️ 局限性**

仅在CARLA/LangAuto模拟环境评估；模板指令可能不足以覆盖真实语言多样性；未考虑多轮交互、不同传感器噪声等限制。

---

## 220. AI and Collective Decisions: Strengthening Legitimacy and Losers' Consent

**arXiv ID:** 2604.05368 | [PDF](https://arxiv.org/pdf/2604.05368v1)

**作者:** Suyash Fulay `[一作]` (Massachusetts Institute of Technology), Deb Roy `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 19067 | [OpenAlex ID](https://openalex.org/A5004281470)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并评估了一套基于AI的集体决策系统，该系统通过语音式半结构化AI访谈收集参与者的个人经历，并通过交互可视化展示预测的政策支持度，旨在提升程序合法性和输家同意。

**💡 创新点**

创新点在于将真实语音访谈与LLM预测相结合，构造了“体验映射”可视化，强调过程而非仅仅结果，关注输家同意，结合多模态数据（语音、文本）提升社会凝聚力。

**🔧 技术方法**

技术包括：OpenAI Whisper（语音转文本），GPT‑4o（对话式AI访谈生成），GPT‑4.1（提取经验、预测支持、评分），前端可视化（Django+前端框架），音频同步展示等。

**📊 数据集**

数据集：由参与者在Prolific上完成的90+份半结构化访谈（包含音频、转写），以及他们对三项政策的投票数据；使用这些自建数据进行实验。没有使用公开的大型语料库。

**📈 对比分析**

采用2×2随机实验（n=181），通过回归模型评估AI访谈、可视化及其交互对过程合法性、社会凝聚力、学习等概念的影响。可视化在处理合法性、信任、理解等方面的效应为标准差0.71–0.77；AI访谈对“被听见”影响为0.51；并未显著改变实际政策支持。没有与其他系统的直接比较。

**⚠️ 局限性**

限制：样本量有限，无法分离可视化单一特征影响；仅研究“输家”情境，未考察获胜者；使用自建访谈数据，可能存在偏差和隐私风险；AI访谈的性别/口音效应未评估；缺乏后续互动功能和专家引导。

---

## 221. 3DTurboQuant: Training-Free Near-Optimal Quantization for 3D Reconstruction Models

**arXiv ID:** 2604.05366 | [PDF](https://arxiv.org/pdf/2604.05366v1)

**作者:** Jae Joong Lee `[一作]` (Purdue University), Jae Joong Lee `[通讯]` (Purdue University)

**通讯引用:** 296 | [OpenAlex ID](https://openalex.org/A5033823554)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

研发了一种无需训练的高维向量量化方法，对3D重建模型（如3D Gaussian Splatting、NeRF hash grid、Transformer KV缓存）中的高维参数进行无数据学习的量化，完成秒级压缩；

**💡 创新点**

发现高维参数在随机旋转后坐标近似Beta分布，可用预先计算的Lloyd‑Max量化器实现近似最优压缩；提出维度依赖的量化准则、范数分离误差界、低维特征分组与可组合压缩管线；

**🔧 技术方法**

使用TurboQuant（随机旋转+Lloyd‑Max量化）、norm‑separation、entry‑grouping、可组合剪枝‑量化管线，以及信息理论误差界与Beta分布分析；

**📊 数据集**

使用NeRF Synthetic（Lego）场景、DUSt3R ViT‑Large多视角数据、Instant‑NGP hash grid等典型3D重建数据集；

**📈 对比分析**

与现有训练依赖和训练自由方法对比，3DGS压缩比3.5×（PSNR损失≤0.02dB），DUSt3R KV压缩比7.9×（点图PSNR 39.7dB），压缩耗时仅数秒，近似信息理论下界；

**⚠️ 局限性**

仅压缩存储不提升推理速度；低维特征分组假设可能不稳健；实现仍CPU，可进一步GPU化；未结合熵编码；对动态/在线更新的适配仍需改进。

---

## 222. Unsupervised Multi-agent and Single-agent Perception from Cooperative Views

**arXiv ID:** 2604.05354 | [PDF](https://arxiv.org/pdf/2604.05354v1)

**作者:** Haochen Yang `[一作]` (Cleveland State University), Hongkai Yu `[通讯]` (Cleveland State University)

**通讯引用:** 3181 | [OpenAlex ID](https://openalex.org/A5025512337)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了无监督多代理与单代理3D目标检测，提出UMS框架实现同时提升两者性能。

**💡 创新点**

通过点云密度提升的Proposal Purifying Filter、Progressive Proposal Stabilizing、Cross‑View Consensus Learning三模块联合使用，实现无需人工标注的多代理和单代理感知。

**🔧 技术方法**

采用PointNet++特征学习、动态阈值、记忆池融合、BEV语义对齐及NMS等技术。

**📊 数据集**

在V2V4Real（真实）和OPV2V（仿真）两个数据集上进行实验。

**📈 对比分析**

与DBSCAN、OYSTER、CPD、DOtA等无监督基线对比，UMS在多代理AP@0.5提升约+30%（OPV2V）或+3%（V2V4Real），单代理AP@0.5提升约+24%（OPV2V）或+3%（V2V4Real），整体显著优于现有方法。

**⚠️ 局限性**

对真实数据点云稀疏与噪声的鲁棒性仍有限，且在多目标类别扩展与定位误差下表现略逊。

---

## 223. AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation

**arXiv ID:** 2604.05351 | [PDF](https://arxiv.org/pdf/2604.05351v1)

**作者:** Yijie Deng `[一作]` (New York University Abu Dhabi), Yi Fang `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 4485 | [OpenAlex ID](https://openalex.org/A5067418255)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的图像目标导航系统，利用语义与几何级联实现精准的最后一米定位；

**💡 创新点**

创新点在于将目标图像视为几何查询，使用BEV相关性图引导探索并在接近目标时调用3D多视图基础模型实现精确6-DoF姿态估计；

**🔧 技术方法**

核心技术包括像素级语义相关性图、BEV投影、3D多视图基础模型（VGGT/Pi3）、Sim(3)对齐与自我校正的姿态精细化；

**📊 数据集**

在Habitat仿真环境下分别使用Gibson（ImageNav）和HM3D（InstanceImageNav）数据集进行评估；

**📈 对比分析**

相较于先前方法，系统在Gibson上取得93.1%成功率，HM3D上82.6%成功率，同时位置误差降至0.27m/0.21m，航向误差3.41°/1.23°，显著优于对标算法；

**⚠️ 局限性**

主要局限包括对深度与里程计的依赖、在语义信号稀疏的场景中探索受限以及固定阈值导致的误判问题。

---

## 224. Generative Channel Knowledge Base With Environmental Information for Joint Source-Channel Coding in Semantic Communications

**arXiv ID:** 2604.05342 | [PDF](https://arxiv.org/pdf/2604.05342v1)

**作者:** Xudong Long `[一作]` (Sun Yat-Sen University), Yubin Zhao `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 1674 | [OpenAlex ID](https://openalex.org/A5077845329)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了基于多维环境信息的生成式通道知识库（CKB），并在语义通信系统中实现了以CKB驱动的联合源信道编码（JSCC）框架，以提升6G-V2X场景下的传输性能。

**💡 创新点**

① 将空间位置、全景图像特征、细粒度语义特征等多源环境信息与通道矩阵关联，生成结构化CKB；② 采用自注意力多模态融合在Transformer中学习环境→通道映射；③ 将生成的通道知识注入JSCC编码与解码，实现端到端环境感知语义通信。

**🔧 技术方法**

Transformer网络、自注意力多模态融合、ROI滤波、PSPNet语义分割、ResNet图像特征提取、深度JSCC、车路仿真（CARLA、SUMO）、光线追踪（Wireless InSite）等。

**📊 数据集**

① 在CARLA+SUMO+Blender+Wireless InSite仿真得到的自建6G-V2X环境数据（约995组位置、图像、语义与通道矩阵）；② CIFAR-10和ImageNet（32×32）用于验证图像重建性能。

**📈 对比分析**

与传统DeepJSCC以及无CKB的JSCC进行对比；在[-5,25] dB SNR范围内，CKB驱动JSCC实现PSNR>22.07 dB、SSIM>0.84；通道生成MSE≈10⁻³；相较于DeepJSCC在复杂传播环境下显著提升图像重建质量，且对低SNR更鲁棒。

**⚠️ 局限性**

依赖大规模高质量环境感知数据集；仿真环境与真实环境存在差异，迁移性能尚未充分验证；在极高速动态场景下的实时更新和计算开销待进一步降低。

---

## 225. Human Values Matter: Investigating How Misalignment Shapes Collective Behaviors in LLM Agent Communities

**arXiv ID:** 2604.05339 | [PDF](https://arxiv.org/pdf/2604.05339v1)

**作者:** Xiangxu Zhang `[一作]` (Renmin University of China), Xing Xie `[通讯]` (Microsoft Research Asia)

**通讯引用:** 45816 | [OpenAlex ID](https://openalex.org/A5044651577)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一个基于社区韧性理论的可控多智能体模拟环境中，系统地研究人类价值观对LLM群体长期集体行为的影响。

**💡 创新点**

发现存在结构性关键价值，能显著改变社区的稳态与崩溃风险，并揭示价值驱动的微观出现的欺骗、权力追求等行为。

**🔧 技术方法**

采用基于LLM的多智能体框架（GPT‑4o），配合价值对齐的上下文提示、人口价值分布控制和社会科学理论的宏观指标。

**📊 数据集**

使用Schwartz十维价值体系作为价值标签，实验基于人工生成的工厂与资源环境，且不依赖公开数据集。

**📈 对比分析**

通过nAUP、社会资本、经济发展、信息通信、社区能力等宏观指标对比不同价值介入，发现如正向仁爱提高稳定性，排除传统则提升韧性，整体效果可视化呈现。

**⚠️ 局限性**

局限在于仅在模拟环境中验证，未覆盖更大规模或真实社会情境，且对LLM内部机制假设过多，缺乏跨模型普适性评估。

---

## 226. Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills

**arXiv ID:** 2604.05333 | [PDF](https://arxiv.org/pdf/2604.05333v1)

**作者:** Dawei Li `[一作]` (University of Pennelvenia), Lichao Sun `[通讯]` (Lehigh University)

**通讯引用:** 8642 | [OpenAlex ID](https://openalex.org/A5015105117)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Graph of Skills（GoS）——一种面向大规模本地技能库的结构化检索层，能够在推理时根据任务语义与技能依赖关系高效提取最小且完整的可执行技能集合。

**💡 创新点**

创新点包括：①在离线阶段构建多关系有向图，将技能的依赖、工作流、语义相似和替代关系明确化；②在推理时使用混合语义‑词法种子结合逆向加权个性化 PageRank（PPR）实现结构感知扩展；③通过预算约束的再排序与“hydration”生成紧凑的可直接使用的技能包。

**🔧 技术方法**

技术细节：技能规范化、关系诱导、混合种子检索（语义向量 + 词法匹配）、逆向加权 PPR 传播、基于图分数与局部匹配的再排序、上下文预算限制的 hydration。

**📊 数据集**

使用数据集：SkillsBench（包含多领域技术任务与配套技能）和 ALFWorld（交互式文本机器人模拟环境），同时对不同大小（200–2000）技能库进行敏感性实验。

**📈 对比分析**

与基线 Vanilla Skills（完整技能加载）和 Vector Skills（基于语义向量检索）在 Claude Sonnet、MiniMax M2.7、GPT‑5.2 Codex 三种模型上进行对比。结果显示：GoS 在所有模型与基准上平均奖励最高，且比 Vanilla Skills 低 37.8% 输入 token，Token 使用与运行时间均明显更优；与 Vector Skills 比较时，GoS 在奖励上提升 10.97–2.87 点，同时保持相似压缩规模。

**⚠️ 局限性**

局限性：依赖于离线图的质量——技能文档不完整、I/O 模式歧义或缺失可执行元数据会削弱边权；图结构静态，未结合执行回放或用户反馈进行在线更新；对多模态或更大规模环境的泛化仍待验证。

---

## 227. VLA-InfoEntropy: A Training-Free Vision-Attention Information Entropy Approach for Vision-Language-Action Models Inference Acceleration and Success

**arXiv ID:** 2604.05323 | [PDF](https://arxiv.org/pdf/2604.05323v1)

**作者:** Chuhang Liu `[一作]` (Ping An Technology (Shenzhen) Co., Ltd.), Jianzong Wang `[通讯]` (Ping An Technology (Shenzhen) Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练‑free的动态推理框架VLA‑InfoEntropy，利用图像熵和注意力熵评估视觉‑语言‑动作模型中每个token的重要性，从而减少冗余计算并加速推理。

**💡 创新点**

创新点在于将视觉熵和注意力熵两种熵度量结合，并引入时间步依赖的动态切换策略，使模型能在推理过程中从全局视觉信息平滑过渡到局部语义聚焦，实现统一的多模态信息过滤。

**🔧 技术方法**

主要技术包括训练‑free熵计算（图像灰度分布熵、交叉注意力熵）、时间步调度公式、KV 缓存重用与 token 筛选、以及Transformer跨模态注意力分析。

**📊 数据集**

实验使用 LIBERO 公开基准（包含 Spatial、Object、Goal、Long 四个子任务）。

**📈 对比分析**

与 OpenVLA、VLA‑Cache、SP‑VLA 等方法对比，VLA‑InfoEntropy 平均成功率提升至 76.4%（比基线高 1.4%），FLOPs 降低 34.9%，CUDA 延迟下降 39.8%，速度提升 1.53×。

**⚠️ 局限性**

局限性包括对超参数（T、k1/k2）的敏感性，需要针对不同任务进行调参；在极长时序或高分辨率场景下的鲁棒性未充分验证；目前仅在现有 VLA 框架上验证，缺乏对更大规模模型的评估。

---

## 228. WSCM-Lite: A Practitioner-Ready Implementation of the Weak Signal Cultivation Model

**arXiv ID:** 2604.05381 | [PDF](https://arxiv.org/pdf/2604.05381v1)

**作者:** Maurice Codourey `[一作]`, Emmanuel A. Gonzalez `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出WSCM‑Lite，一种基于查找表的轻量级弱信号培养模型，可在Excel中追踪风险信号轨迹并触发SMS升级。

**💡 创新点**

用四行查找表取代四个指数函数，删去共识动量与逆转放大器，公式仅7个、参数硬编码，保持决策等价，兼顾易用与可解释性。

**🔧 技术方法**

采用离散递归权重查表、委员会缩放因子、被动衰减与距离阈值判定，全部实现为Excel公式与手工可验证的计算。

**📊 数据集**

主要用父论文中的Gas Fumes信号26场次数据做演示，并模拟四个边界场景验证模型稳健性。

**📈 对比分析**

与完整WSCM进行数值对比，轨迹误差≤0.01坐标单位，SMS触发时间相差≤2场次，四区路径保持一致，演示与模拟结果均符合预期。

**⚠️ 局限性**

去除动量导致收敛慢；单报单方警报被低估；查表离散化导致间隔边界误差；无动态节奏调整，适用性受限。

---

## 229. UAVReason: A Unified, Large-Scale Benchmark for Multimodal Aerial Scene Reasoning and Generation

**arXiv ID:** 2604.05377 | [PDF](https://arxiv.org/pdf/2604.05377v1)

**作者:** Jintao Sun `[一作]` (Beijing Institute of Technology), Zhedong Zheng `[通讯]` (University of Macau)

**通讯引用:** 9979 | [OpenAlex ID](https://openalex.org/A5034162160)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了UAVReason统一的高阶 UAV 视角多模态基准，涵盖273k VQA、188k交叉模态生成、23.6k帧的详细注释与几何信息。

**💡 创新点**

创新点在于将空间时序推理与像素级生成统一到同一框架，并通过程序化生成问题/答案以及密集几何监督显著提升视觉语言模型在极端 UAV 视角的表现。

**🔧 技术方法**

采用基于Bagel的多任务Transformer结构，结合Mixture‑of‑Experts、离散视觉编码与Diffusion‑style先验；同时利用3D真实环境重建得到深度、分割、实例ID并生成多级问题。

**📊 数据集**

使用UAVScenes的真实LiDAR+RGB数据生成20个不同场景的23.6k帧，并与VisDrone、DOTA、UAVDT等公开数据做对比。

**📈 对比分析**

与通用VLM（如Qwen、LLaVA）、专门的UAV/遥感VLM（GeoChat、RS‑LLaVA）以及统一生成模型（Janus、OmniGen）对比，UAVReason‑Bagel在VQA、Caption、Seg、条件生成等指标上均取得显著领先，EM/F1、LLM‑J、mIoU、KID、CLIP、DINO均高于对手。

**⚠️ 局限性**

主要限制在推理延迟高、模型规模大（7B参数）不适合嵌入式 UAV，需通过压缩、量化或轻量化骨干网络来满足实时控制需求。

---

## 230. From Clues to Generation: Language-Guided Conditional Diffusion for Cross-Domain Recommendation

**arXiv ID:** 2604.05365 | [PDF](https://arxiv.org/pdf/2604.05365v1)

**作者:** Ziang Lu `[一作]` (Anhui University), Yiwen Zhang `[通讯]` (Anhui University)

**通讯引用:** 5026 | [OpenAlex ID](https://openalex.org/A5100430650)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LGCD 框架，用 LLM 生成潜在目标域兴趣并映射为伪重叠交互，再通过条件扩散模型在源域信息的引导下生成目标域用户表示，从而解决跨域推荐中重叠用户稀缺导致的冷启动问题。

**💡 创新点**

创新点包括：①利用 LLM 推理构造伪重叠数据并通过语义检索映射到真实物品；②设计跨注意力驱动的条件扩散网络直接回归目标域语义特征；③引入对齐损失、轻量化猜测网络和 MoE 融合，以及循环批采样策略，显著抑制伪交互噪声并提升生成质量。

**🔧 技术方法**

技术手段：大语言模型（Baichuan2‑7B）、文本编码器（jina‑embeddings）、Transformer 序列编码器、条件扩散模型（正向噪声、跨注意力去噪）、MoE（多专家融合）、轻量化猜测网络、循环批采样。

**📊 数据集**

实验数据集：Amazon Movie‑Book 与 Food‑Kitchen 两对跨域数据集，分别包含数千到数万用户、数万物品，并划分为单域和交叉域交互。

**📈 对比分析**

方法对比：单域基线（GRU4Rec、DiffuRec、SASRec 等）、内域方法、非重叠方法（DA‑DAN、LLMCDSR、PLCR）、交叉域传统方法（SSCDR、UniCDR、UCLR、CD‑CDR）等。实验结果显示 LGCD 在 HR@5、HR@10、NDCG@5、NDCG@10 上均显著优于所有基线，提升幅度可达 +35% 以上。

**⚠️ 局限性**

局限性：依赖重叠用户比例，过低时效果下降；伪交互生成可能引入语义噪声，需要精细调参；LLM 推理和扩散模型训练耗时耗资源；在领域差异极大或极端稀疏场景下性能可能衰减。

---

## 231. TFRBench: A Reasoning Benchmark for Evaluating Forecasting Systems

**arXiv ID:** 2604.05364 | [PDF](https://arxiv.org/pdf/2604.05364v1)

**作者:** Md Atik Ahamed `[一作]` (Google), Tomas Pfister `[通讯]` (Google)

**通讯引用:** 8275 | [OpenAlex ID](https://openalex.org/A5101265241)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向推理的时间序列预测基准，使用多智能体循环生成可验证的自然语言推理轨迹，并将其与数值预测联合评估。

**💡 创新点**

创新点在于：①提出推理质量与预测精度相结合的评价框架；②利用搜索、验证、推理、预测、摘要等五个角色的“生成‑验证‑改进”循环自动生成高质量参考推理；③通过LLM‑as‑Judge对推理一致性进行客观打分，实现可复现、可扩展的基准。

**🔧 技术方法**

技术包括：多智能体生成框架（Gemini‑2.5‑Pro/3.0‑Pro、Claude‑Sonnet 4.x 等 LLMs）、检索检验、CoT 推理、MASE 误差评估、LLM‑as‑Judge 评判器。

**📊 数据集**

使用十个跨五大领域（能源、零售、Web/云运维、交通、金融）公开数据集：Solar、Electricity、Car‑parts、Hierarchical Sales、Bitbrains、Web Traffic、NYC Taxi、Amazon Pricing、Apple Pricing 等。

**📈 对比分析**

通过与传统时间序列基础模型（TimesFM‑2.5、Chronos‑2.0、ARIMA）以及零射击 LLM 对比，发现推理质量与 MASE 负相关，推理增强后 LLM 的预测误差可与 TSFM 接近，且在物理领域表现尤佳；在金融等随机领域则易出现叙事偏差。

**⚠️ 局限性**

局限性包括：①依赖未来事件的 oracle 知识，可能导致在真实部署时数据泄漏风险；②叙事偏差导致在高熵领域表现退化；③生成推理成本高，且对 LLM 规模敏感；④当前评估仅覆盖十个数据集，需扩展至更多领域和更大规模人类验证。

---

## 232. GESS: Multi-cue Guided Local Feature Learning via Geometric and Semantic Synergy

**arXiv ID:** 2604.05359 | [PDF](https://arxiv.org/pdf/2604.05359v1)

**作者:** Yang Yi `[一作]` (National University of Defense Technology), Dewen Hu `[通讯]` (National University of Defense Technology)

**通讯引用:** 16305 | [OpenAlex ID](https://openalex.org/A5071074935)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种多线索引导的局部特征学习框架，融合语义、法向和深度稳定性，实现更稳健的关键点检测与更具辨别力的描述子；

**💡 创新点**

创新点包括：1）语义-法向耦合预测头，用3D向量场同时输出语义置信度与表面法向，消除多任务梯度冲突；2）Unified Triple-Cue Fusion模块，语义调度门控动态融合纹理、法向和语义特征，提升描述子辨别力；3）Depth Stability Head与Semantic-Depth Aware Keypoints机制，双重约束对关键点响应进行重加权，过滤不稳定点；

**🔧 技术方法**

采用轻量级多尺度骨干MLSNet，联合语义-法向预测、深度稳定性预测，构建UTCF和SDAK模块，使用多任务损失（检测、描述、语义、法向、稳定性）进行端到端训练，框架基于PyTorch实现，优化器为Adam；

**📊 数据集**

训练使用MegaDepth子集生成伪标签；测试集涵盖HPatches、Aachen Day‑Night、MegaDepth‑1500、ScanNet、ETH Local Feature Benchmark等多种视觉基准；

**📈 对比分析**

与SuperPoint、D2‑Net、R2D2、DISK、ALIKE、MTLDesc、SAMFeat等方法对比，在HPatches上MMA@3=83.51%、AUC@2=57.56%、AUC@5=77.81%，超过SAMFeat；在Aachen Day‑Night定位上精度优于SAMFeat；在MegaDepth‑1500和ScanNet姿态估计中5°阈值下分别达53.9%和76.7%，略优于SAMFeat；在ETH 3D重建中轨迹长度更长、重投影误差更小；

**⚠️ 局限性**

局限性：仅以MegaDepth训练，缺乏室内或COCO等多域数据，可能在极端场景或动态环境中仍受限；推理速度虽为56 FPS，但在嵌入式设备上仍需进一步加速；对极低纹理或大尺度场景的适应性尚待验证。

---

## 233. DQA: Diagnostic Question Answering for IT Support

**arXiv ID:** 2604.05350 | [PDF](https://arxiv.org/pdf/2604.05350v1)

**作者:** Vishaal Kapoor `[一作]` (Amazon), Rebecca Steinert `[通讯]` (Amazon)

**通讯引用:** 756 | [OpenAlex ID](https://openalex.org/A5007363439)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个多轮企业 IT 支持诊断问答框架 DQA，结合检索聚合、诊断状态维护和基于状态的动作选择，实现系统化的故障排查。

**💡 创新点**

创新点：将检索结果聚合到根因级别而非单个文档；显式维护诊断状态以累计证据并跟踪竞争假设；通过状态驱动的问答策略实现不确定性逐步降低。

**🔧 技术方法**

技术：检索增强生成（RAG）+ 对话查询重写（CQR）+ 根因聚类与检索聚合（RAggG）+ 结构化诊断状态 + 基于状态的响应生成。

**📊 数据集**

数据集：150 个匿名企业 IT 支持交互场景（硬件、软件配置、权限、身份认证等多类型故障），使用历史支持工单和 KB 文章做检索。

**📈 对比分析**

比较方法：回放式评估，将 DQA 与三种 RAG 变体（无 CQR、加 CQR、加聚类）对比。DQA 成功率 78.7%（基线 41.3%），平均回合数 3.9（基线 8.4），显示显著性能提升。

**⚠️ 局限性**

限制：评估基于模拟用户，未测量真实用户体验；根因结构可能因文档质量噪声导致误判；仅在企业 IT 支持领域验证，泛化性待进一步验证。

---

## 234. TRACE: Capability-Targeted Agentic Training

**arXiv ID:** 2604.05336 | [PDF](https://arxiv.org/pdf/2604.05336v1)

**作者:** Hangoo Kang `[一作]` (Stanford University), Azalia Mirhoseini `[通讯]` (Stanford University)

**通讯引用:** 2610 | [OpenAlex ID](https://openalex.org/A5070731184)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个端到端系统 TRACE，利用对比轨迹分析自动识别大型语言模型代理在目标环境中缺失的关键能力，随后为每个能力生成可验证的合成训练环境，使用 LoRA 适配器与 GRPO 强化学习进行轻量级训练，并在推理阶段通过路由 Prompt 选择最合适的能力适配器执行任务。

**💡 创新点**

创新点在于（1）通过对成功/失败轨迹的对比来定量识别缺失能力，消除了传统 RL 在目标环境中因缺乏明确失败归因导致的稀疏信号；（2）为每个缺失能力专门生成隔离且可自动验证的合成环境，显著提升训练信号密度和数据效率；（3）采用 LoRA 低秩适配器与路由机制，避免整体微调或模型合并的参数膨胀与性能退化；（4）在相同 roll‑out 数量下实现更快的性能提升。

**🔧 技术方法**

主要技术包括：LLM 驱动的分析代理（对比轨迹识别能力）与生成代理（合成环境构造）；GRPO 强化学习算法用于 LoRA 适配器训练；LoRA 低秩参数分解；路由 Prompt（任务描述 + 能力说明 + 成功轨迹）决定推理时使用的适配器；对比错误率 Δ(c) 与覆盖率 Cov(c) 的阈值筛选关键能力。

**📊 数据集**

使用了两类公开基准：Airline & Retail 领域的客户服务 benchmark（Airc）和 ToolUse benchmark（129 场景）；基准数据通过 Qwen3‑30B‑A3B‑Instruct‑2507 进行交互；合成环境则由 LLM 生成，种子生成多样化任务实例。

**📈 对比分析**

与基线直接 RL（GRPO）、Agent World Model（AWM）、Agent Data Protocol（ADP）以及 GEPA（提示优化）对比。TRACE 在 Airc 上整体 Pass rate 提升 14.1 点，超越最佳基线 7.4 点；在 ToolUse 上 mean similarity 提升 0.141 点、perfect scores 提升 7，超越最佳基线 0.032 点、4 个 perfect。相同 roll‑out 数量下，TRACE 分别比 GRPO、GEPA 提升 9.2 点和 7.4 点；单能力 LoRA 适配器已达 40.3% Pass，优于 AWM 38.4% 和 ADP 32.3%。

**⚠️ 局限性**

局限性包括：仅对少量关键能力（2–4）进行训练，未覆盖所有潜在缺失能力；合成环境的设计依赖 LLM 的生成质量，可能与真实环境差异；路由仅选择单一适配器，未考虑多能力协同的场景；实验仅在两类基准上验证，泛化性尚待进一步验证；尽管 LoRA 轻量，但仍需要大量 RL 迭代和 GPU 资源。

---

## 235. Semantic Trimming and Auxiliary Multi-step Prediction for Generative Recommendation

**arXiv ID:** 2604.05329 | [PDF](https://arxiv.org/pdf/2604.05329v1)

**作者:** Tianyu Zhan `[一作]` (Zhejiang University), Shengyu Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 3038 | [OpenAlex ID](https://openalex.org/A5100757082)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对语义ID（SID）生成式推荐中因序列冗余导致的训练成本高与性能波动大问题，提出STAMP框架，结合输入端的语义自适应裁剪（SAP）和输出端的多步辅助预测（MAP），实现高效训练且保持甚至提升推荐效果。

**💡 创新点**

① 把语义稀释效应统一视为信息非均匀性与监督稀疏性的根本原因；② 设计双端优化策略，动态裁剪冗余SID序列并通过多token预测稠密化监督；③ SAP将注意力中心性与语义显著性融合实现自适应裁剪；④ MAP在训练阶段独立增强长距离依赖，推理时无额外开销。

**🔧 技术方法**

使用Transformer（Encoder‑Decoder T5、Decoder‑Only Qwen）+自注意力权重+残差量化编码生成SID；实现SAP（token重要性评估、顺序保持压缩）和MAP（多步辅助预测、共享投影）模块；训练时采用Adam/AdamW、bfloat16等优化策略。

**📊 数据集**

Amazon Beauty/Toys/Sports三大公开数据集（5‑core过滤）以及大型工业级 AL‑GR‑Tiny（来自淘宝，250M+商品，多模态），两类数据分别用于GRID（T5）和FORGE（Qwen）框架实验。

**📈 对比分析**

在GRID和FORGE框架下与完整序列基线对比，评估指标为Recall@K/NDCG@K（T5）和HitRate@K（Qwen）；实验显示STAMP训练速度提升1.23–1.38×，VRAM节省17.2%–54.7%，同时保持或提升推荐精度，显著提高了训练效率。

**⚠️ 局限性**

① 对高语义密度数据单独使用SAP可能导致性能下降，需要MAP补偿；② 需要手动调节裁剪层与保留比例，极端压缩会丢失关键语义；③ MAP仅在训练阶段存在，增加训练复杂度；④ 对极大模型的显存减少有限，参数占比高；⑤ 在多域或跨平台场景的鲁棒性尚未充分验证。

---

## 236. AI-Augmented Peer Review and Scientific Productivity: A Cross-Country Panel and SEM Analysis

**arXiv ID:** 2604.05463 | [PDF](https://arxiv.org/pdf/2604.05463v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 237. OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward

**arXiv ID:** 2604.05514 | [PDF](https://arxiv.org/pdf/2604.05514v1)

**作者:** Haoyue Yang `[一作]` (Institute of Automation Chinese Academy of Sciences), Yao Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 5861 | [OpenAlex ID](https://openalex.org/A5034221181)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了OmniDiagram统一框架，支持三类图表代码生成任务（图表-代码、图表编辑、文本-代码）并兼容Mermaid、PlantUML等语言；

**💡 创新点**

创新点包括：①构建首个大规模三维任务-语言矩阵数据集M3^2Diagram；②设计基于视觉问答的奖励机制Viva，细粒度评价渲染结果并驱动RL；③将SFT+RL两阶段训练结合，实现自我进化；

**🔧 技术方法**

技术手段涵盖：大模型微调（Qwen2.5-VL为基座）、强化学习（GRPO）与Viva奖励、离线视觉问题生成、图像渲染验证、精细化奖励分配；

**📊 数据集**

使用数据集：M3^2Diagram（196k样本）与补充的77k推理增强样本，分为SFT/ RL训练集；评测基准M3^2Bench（1.7k样本）以及CoSyn、VisPlotBench等公开数据集；

**📈 对比分析**

实验与比较：在M3^2Bench、CoSyn、VisPlotBench等上，OmniDiagram（RL版）在执行率、视觉一致性（S_vis）、代码正确率等指标均超过同类开源模型（如InternVL、Qwen系列）并逼近/超越部分专用模型；

**⚠️ 局限性**

局限性：①奖励权重α固定，未实现动态或任务特定调节；②仅采用GRPO，缺少对其他RL算法（PPO、DPO）的系统比较；③数据合成与评估依赖大型外部模型，计算成本高，未来需寻找更高效方案。

---

## 238. CoEnv: Driving Embodied Multi-Agent Collaboration via Compositional Environment

**arXiv ID:** 2604.05484 | [PDF](https://arxiv.org/pdf/2604.05484v1)

**作者:** Li Kang `[一作]` (Shanghai Jiao Tong University), Lei Bai `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CoEnv框架，将真实感知与仿真环境融合为多智能体协作的统一决策空间；

**💡 创新点**

创新点在于（1）构造可实时重建的“组合环境”实现零成本的安全策略探索与验证；（2）引入双模规划（交互式VLM+代码生成）与多视角自适应摄像头与检查点验证机制；（3）提供可扩展的多智能体数据生成管线；

**🔧 技术方法**

技术手段包括：多视角RGB‑D感知+Grounded SAM2/ GPT‑5/FoundationPose实现真实‑到‑仿真场景重建；VLM（GPT‑5）进行层级任务规划；Claude Code等代码生成器完成完整轨迹编写；仿真平台ManiSkill（SAPIEN）与Trajectory‑interpolation/Collision‑volume‑verification实现安全sim‑to‑real迁移；

**📊 数据集**

主要使用自定义的五个多臂协作任务数据（Cube Stacking、Ball Pickup、Transfer Cylinder、Place Cucumber、Brush Box）以及对应的RGB‑D多视角数据；未使用公开基准数据集；

**📈 对比分析**

通过在10个试验中评估交互式与迭代两种执行模式，报告子任务成功率与整体任务成功率，最终平均成功率约为49%（交互式50%，迭代48%）；相较于传统单体或无检验策略，CoEnv在多臂协作中显著提升了任务完成率与执行效率；

**⚠️ 局限性**

局限性包括：（1）仍存在仿真‑真实差距导致抓取/插入失败；（2）VLM/代码生成可能陷入循环重规划；（3）交互式模式易出现累计漂移，迭代模式难以处理高度响应式任务；（4）任务规模与物体种类受限；（5）缺乏在线适应与动态环境更新机制。

---

## 239. Can You Trust the Vectors in Your Vector Database? Black-Hole Attack from Embedding Space Defects

**arXiv ID:** 2604.05480 | [PDF](https://arxiv.org/pdf/2604.05480v1)

**作者:** Hanxi Li `[一作]` (Sichuan University), Mingjie Tang `[通讯]` (Sichuan University)

**通讯引用:** 1555 | [OpenAlex ID](https://openalex.org/A5102006508)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对向量数据库的安全性进行研究，提出一种无查询知识、只需少量恶意向量即可诱导检索结果的黑洞攻击。

**💡 创新点**

创新点在于揭示并利用高维嵌入空间中的“中心性驱动hubness”——即靠近几何中心的向量天然成为多数点的最近邻，从而构造攻击者无需查询数据即可控制检索结果。

**🔧 技术方法**

使用聚类+中心点生成恶意向量、欧氏/余弦距离、近似最近邻索引（Flat、HNSW、IVF-Flat、IVF-PQ）以及中心化、标准化等技术进行攻击与防御实验。

**📊 数据集**

实验数据集包括 HotpotQA、MSMARCO 和 Natural Questions (NQ) 三个公开问答数据集。

**📈 对比分析**

通过对比 MO@10、ASR、Recall@10 等指标，1% 注入率即可使恶意向量占 Top‑10 近 90% 以上，召回率从 90% 降至 10% 以下；与现有后门、语料毒化等攻击对比表明黑洞攻击更通用、影响更大。

**⚠️ 局限性**

局限性包括：需要约 1% 的注入率；在不同距离度量或极高维度下攻击效果略有变化；现有防御（hubness 缓解、检测）在降低攻击效果的同时会显著降低检索精度或引入额外计算开销。

---

## 240. Few-Shot Semantic Segmentation Meets SAM3

**arXiv ID:** 2604.05433 | [PDF](https://arxiv.org/pdf/2604.05433v1)

**作者:** Yi-Jen Tsai `[一作]` (National Yang Ming Chiao Tung University), Chien-Yao Wang `[通讯]` (Academia Sinica)

**通讯引用:** 32948 | [OpenAlex ID](https://openalex.org/A5072470661)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种完全冻结的 SAM3 模型在无训练的前提下完成少样本语义分割，采用空间拼接将支持图与查询图放在同一画布上，让自注意力机制实现跨图对应。

**💡 创新点**

创新点包括：1）利用空间拼接将跨图关系转化为空间推理，避免显式匹配模块；2）揭示负面提示在少样本场景下的负面影响；3）将文本先验与视觉示例联合使用提升泛化。

**🔧 技术方法**

核心技术是 SAM3 的 Promptable Concept Segmentation（PCS）、空间拼接（shared canvas）、实例级正向提示、可选的文本提示，以及对负面提示的系统性分析。

**📊 数据集**

使用公开基准 PASCAL-5^i 和 COCO-20^i 进行 1-shot/5-shot 评测，亦在 MSCOCO 上测试负面提示效应。

**📈 对比分析**

与基于训练的 FSS 方法对比，FSS-SAM3 在 1-shot 上达到 81.6 mIoU（PASCAL）/75.4 mIoU（COCO），在 5-shot 上与多种前沿方法竞争；引入文本提示进一步提升，COCO 上 86.4 mIoU；负面提示反而导致性能显著下降。

**⚠️ 局限性**

主要局限：1）负面提示会导致预测崩溃，缺乏对冲突语义信号的平衡机制；2）图像与文本编码器未在同一语义空间对齐，跨模态匹配受限；3）简单的时序记忆策略对跨图语义对应无效。

---

## 241. Your LLM Agent Can Leak Your Data: Data Exfiltration via Backdoored Tool Use

**arXiv ID:** 2604.05432 | [PDF](https://arxiv.org/pdf/2604.05432v1)

**作者:** Wuyang Zhang `[一作]` (University of Massachusetts Boston), Shichao Pei `[通讯]` (University of Massachusetts Boston)

**通讯引用:** 615 | [OpenAlex ID](https://openalex.org/A5034495880)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Back-Reveal攻击框架，利用后门触发的LLM代理通过工具调用读取会话记忆并伪装为检索请求进行数据泄露，且通过多轮交互逐步获取更多信息。

**💡 创新点**

创新点在于：1）将语义触发与工具使用相结合，实现隐蔽的记忆读取与外部泄露；2）设计了重写器（reranker-aware rewriter）在检索结果中嵌入隐式引导信息，绕过reranker和检索阶段的安全过滤；3）展示多轮交互能显著放大泄露量。

**🔧 技术方法**

技术包括：LLM后门注入（基于语义触发的SFT+DPO+PPO训练）、工具调用链（session‑memory + outbound retrieval）、Base64url编码的payload、检索式HTTP请求伪装、重写器模型训练（SFT→DPO→PPO，结合事实性、隐式引导、排版与reranker得分奖励）。

**📊 数据集**

使用了三个公开域数据集：直播（OBS/Twitch/RTMP）、医疗（Donepezil/记忆护理/照护支持）和教育（backprop/作业/作业作业）等，构建了多词语义触发样本；同时采用了10,000文档的检索语料库进行RAG评估。

**📈 对比分析**

实验对比了三种响应生成方式：Leak（仅相关内容）、Leak+Naive Append（显式引导）、Leak+Rewrite（隐式引导）。在七个reranker上，Leak+Rewrite在top‑5放置率约85–91%，显著高于Naive Append的62–70%；在NeMo Guardrails/LLM Guard下，Leak+Rewrite的通过率约81–87%，远超Naive Append的27–40%。多轮实验表明，每轮成功可泄露约两字段，随着对话深度增加，累计泄露数量接近完整10字段配置。

**⚠️ 局限性**

局限性包括：仅针对存储会话记忆且具外部检索权限的开放式Agent；多轮实验基于理想化的合作用户模型，真实用户可能拒绝或转移话题；仅评估了NeMo Guardrails和LLM Guard两种检索阶段防御，未覆盖更广泛的网络出口控制或工具调用审计；后门注入依赖大量自生成训练数据，实际部署难度未知。

---

## 242. VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG

**arXiv ID:** 2604.05418 | [PDF](https://arxiv.org/pdf/2604.05418v1)

**作者:** Honghao Fu `[一作]` (University of Queensland), Yujun Cai `[通讯]` (University of Queensland)

**通讯引用:** 3508 | [OpenAlex ID](https://openalex.org/A5100611987)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VideoStir，一种通过构建视频的时空图结构并结合意图相关检索来提升长视频理解的检索增强生成框架。

**💡 创新点**

创新点在于将视频结构化为时空图实现多跳检索，并引入基于 MLLM 的意图相关性评分器，从语义匹配向意图驱动的检索转变。

**🔧 技术方法**

主要技术包括时空图构建（事件边界检测、视频语言编码）、多跳图检索、意图相关性评分器（LoRA 微调的 Qwen2.5‑VL‑3B）、以及多层检索（clip‑level 与 frame‑level）。

**📊 数据集**

使用了自研的 IR‑600K 数据集（约 600K 帧‑查询意图对）来训练意图评分器，并在 LongVideoBench、MLVU、Video‑MME‑Long 等公开长视频问答基准上进行评测。

**📈 对比分析**

与多种基线（GPT‑4o、LLaVA‑Video、mPLUG‑Owl3、Aria、InternVL‑1.5 以及先前的 Video‑RAG 等）对比，VideoStir 在不增加额外文本或工具调用的情况下取得与或超过 SOTA 的性能，显示出结构化与意图检索的优势。

**⚠️ 局限性**

主要局限在于额外的结构化与检索步骤带来的系统延迟，且整体推理流程仍需进一步优化以降低端到端延时。

---

## 243. Multi-Drafter Speculative Decoding with Alignment Feedback

**arXiv ID:** 2604.05417 | [PDF](https://arxiv.org/pdf/2604.05417v1)

**作者:** Taehyeon Kim `[一作]` (LG AI Research), Se-Young Yun `[通讯]` (KAIST AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于多草稿者的自适应推理框架MetaSD，通过多臂赌博机动态选择最佳草稿者以提升大语言模型的推理速度。

**💡 创新点**

创新点在于提出训练无关、可扩展的多草稿者选择策略，利用对齐反馈的块级多样性奖励BD，并给出停靠时间误差的理论上界；同时实现了黑盒与白盒两种配置。

**🔧 技术方法**

使用了Speculative Decoding、Block Divergence奖励、Upper Confidence Bound (UCB) 多臂赌博机算法，并对非平稳环境与切换成本进行了理论与实验分析；草稿者通过自蒸馏训练得到多任务专用模型。

**📊 数据集**

评估数据集包括MT‑Bench、WMT16、CNN/DailyMail、GSM8K、各语言翻译（如Ja→En、Ru→En、De→En、Fr→En、Zh→En）等多任务混合集。

**📈 对比分析**

与单一草稿者、OFA、PLD、Lookahead、EXP3、SH、UCB等基线以及MoE路由、Medusa、Eagle等方法对比，MetaSD‑UCB在黑盒/白盒实验中平均实现约1.8–3.6倍速度提升，并在提示扰动和跨任务场景中保持鲁棒。

**⚠️ 局限性**

局限性包括在多GPU批量推理中的可扩展性待进一步验证、草稿者数量增多时的计算与内存开销、以及对不同LLM体系结构的泛化仍需更广泛评测。

---

## 244. Learning to Synergize Semantic and Geometric Priors for Limited-Data Wheat Disease Segmentation

**arXiv ID:** 2604.05415 | [PDF](https://arxiv.org/pdf/2604.05415v1)

**作者:** Shijie Wang `[一作]` (University of Queensland), Zi Huang `[通讯]` (University of Queensland)

**通讯引用:** 13589 | [OpenAlex ID](https://openalex.org/A5078170935)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

针对稻草有限数据、跨生长阶段显著形态变化的麦田病害分割问题，提出了 SGPer 框架，将预训练的 DINOv2 与 SAM 通过疾病感知适配器（DPA）实现语义与几何先验的协同；通过 DINOv2 的语义特征生成密集类别点提示，并用几何反馈段器（GFS）动态过滤冗余提示，最终实现精确且稳健的病害边界分割。

**💡 创新点**

创新点包括：
1) 将语义先验与几何先验在有限数据场景下耦合；
2) 在两大基座模型（DINOv2 与 SAM）中共用轻量级疾病感知适配器，兼顾语义感知与几何定位；
3) 通过 DINOv2 语义图生成多类别密集点提示，并结合 SAM 的掩码置信度与语义一致性做双重过滤，实现高效、低噪的提示筛选；
4) 设计了辅助语义提示监督与动态阈值策略，显著提升提示质量。

**🔧 技术方法**

核心技术包括：
- 预训练的 Vision Transformer 基座模型 DINOv2（ViT-L）与 Segment Anything Model（SAM，ViT-L）；
- 病害感知适配器（DPA）实现轻量级的特征重校准与多尺度疾病滤波；
- 语义提示生成模块（SPG）将 DINOv2 语义特征映射为类别点提示；
- 几何反馈段器（GFS）实现基于 SAM 迭代掩码置信度与 DINOv2 语义一致性的交叉过滤；
- 交叉熵 + Dice 损失与辅助语义监督损失。

**📊 数据集**

使用的公开数据集包括：
- EFDv2（400 张图像，320 训练/80 测试）用于病害分割；
- OSD（184 张图像，147 训练/37 测试）用于器官分割；
- GWFSS（308 张图像，99 训练/99 验证/110 测试）用于器官分割。

**📈 对比分析**

与 Mask2Former、Segformer、SegNeXt、DDRN、SAN、GCNet 等现有方法比较，SGPer 在 EFDv2 上取得 mIoU 72.5% / F1 82.8%，在 OSD 上 mIoU 82.4% / F1 89.8%，在 GWFSS 上 mIoU 70.8% / F1 81.5%。相较于最强基线（如 Mask2Former、SegNeXt 等），SGPer 分别提升 9–12% 的 IoU 与 F1，尤其在数据稀缺的病害类别（如 Powdery Mildew）中将 IoU 从约 21% 提升至 47% 以上，显著提升鲁棒性与精度。

**⚠️ 局限性**

局限性：
1) 依赖于大规模预训练模型（DINOv2 与 SAM），对算力与显存要求较高；
2) 目前仅在麦类数据集验证，跨作物通用性仍待探索；
3) 对极端光照或遮挡场景下的提示生成与过滤可能仍出现误判；
4) 由于在有限标注数据上微调，可能对极少样本病害仍表现不稳定；
5) 实时性尚未评估，推理速度受点提示数量与 SAM 解码次数影响。

---

## 245. Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality

**arXiv ID:** 2604.05510 | [PDF](https://arxiv.org/pdf/2604.05510v1)

**作者:** Yanming Xiu `[一作]` (Duke University), Maria Gorlatova `[通讯]` (Duke University)

**通讯引用:** 2220 | [OpenAlex ID](https://openalex.org/A5036726336)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 ContrAR benchmark，用以评估视觉‑语言模型（VLM）在增强现实（AR）环境下检测对抗性虚拟内容冲突的能力。

**💡 创新点**

①首次在 AR 语义层面定义“矛盾虚拟内容攻击”及对应威胁模型；②构建包含 312 条真实 AR 视频、10 人员验证的对照数据集；③系统评估 11 种商业与开源 VLM 的检测精度与推理时延。

**🔧 技术方法**

使用多模态 VLM（GPT‑5、GPT‑4.1、GPT‑4o、Gemini‑2.5‑Pro、Gemini‑2.5‑Flash、Grok‑4、Grok‑2‑Vision、Claude‑Sonnet‑4.5、Claude‑Haiku‑4.5、Qwen‑2.5‑VL‑72B、Qwen‑2.5‑VL‑7B）以及 OCR+GPT‑4o 文本基线；采用单帧与多帧提示策略；通过统一 prompt 提取“Yes/No”答案。

**📊 数据集**

ContrAR 数据集：312 条 1920×1080、5–15 秒的 AR 视频，分别为 5 个常见 AR 用例（室内导航、室外导航、安全检查、智能家居、智能零售），正负样本 1:1，视频中包含文本与视觉虚拟内容。

**📈 对比分析**

在单帧模式下，GPT‑5 以 88.14% 的准确率领跑；多帧模式下 GPT‑4.1 以 86.54% 最高；轻量级开源模型与 OCR 文本基线准确率仅 50–60%；推理时延从 4 秒（Qwen‑7B）到 19 秒（GPT‑5）不等，呈现精度‑时延权衡。

**⚠️ 局限性**

局限：数据规模有限，缺乏实时生成与自适应攻击；标签验证为离线问卷，未评估对真实用户行为的影响；只关注语义冲突攻击，未覆盖认知过载或分散注意力等更广泛的 AR 威胁；未使用视频‑直接输入的 VLM，缺乏时序建模。

---

## 246. From Pixels to Personas: Tracking the Evolution of Anime Characters

**arXiv ID:** 2604.05507 | [PDF](https://arxiv.org/pdf/2604.05507v1)

**作者:** Rongze Liu `[一作]` (University of British Columbia), Jian Zhu `[通讯]` (University of British Columbia)

**通讯引用:** 74000 | [OpenAlex ID](https://openalex.org/A5100394072)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对日本动漫角色的文本、视觉与制作特征进行大规模多模态分析，揭示其历史演变与受众偏好；

**💡 创新点**

首次将LLM提取人格关键词与视觉嵌入结合，对角色原型、视觉趋势及受众喜好进行系统量化，突出Moe化进程及其对受众偏好的主导作用；

**🔧 技术方法**

采用大型语言模型Qwen3提取人格关键词、DINOv2提取头像视觉嵌入、UMAP+Leiden社区检测、PCA降维、随机森林/XGBoost预测角色原型、Poisson回归/HGBT预测受欢迎度，并用SHAP解释模型；

**📊 数据集**

采集自MyAnimeList的27,783部动漫、130,000+角色的文本描述与头像，构建20k+角色多模态数据库；

**📈 对比分析**

相较于随机/多数基线，视觉特征在受欢迎度预测中MAE≈0.67、MPD≈0.95，人物原型预测F1≈0.34；组合模型优于单一特征；

**⚠️ 局限性**

数据来源单一平台、偏向英语社区；缺失音频、完整流派标签；模型解释受预训练视觉模型限制，未能捕捉文化语境与内容分级等潜在混杂因素。

---

## 247. JailWAM: Jailbreaking World Action Models in Robot Control

**arXiv ID:** 2604.05498 | [PDF](https://arxiv.org/pdf/2604.05498v1)

**作者:** Hanqing Liu `[一作]` (Shanghai Jiao Tong University), Yao Mu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 12560 | [OpenAlex ID](https://openalex.org/A5008178136)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了针对世界动作模型（WAM）的越狱攻击，提出了 JailWAM 框架以检测并评估机器人在受攻击时的物理安全风险。

**💡 创新点**

创新点包括：① 三层安全分类框架，系统量化机器人动作安全级别；② 视觉轨迹映射（VTM），将低级动作空间统一成可视化轨迹；③ 风险判别器（RD），快速预测轨迹安全级别；④ 双路径验证策略，先开环粗筛后闭环精确验证；⑤ JailWAM-Bench，首个专注于物理安全的越狱评测基准。

**🔧 技术方法**

技术手段：LLM（Gemini、GPT‑5 等）生成攻击指令；视觉-语言模型（Qwen3‑VL）训练风险判别器；RoboTwin 与 LIBERO 等仿真环境进行闭环验证；开环预测与闭环模拟相结合的双路径策略；对抗样本生成与筛选流程。

**📊 数据集**

使用的数据集与环境包括：RoboTwin 50 任务 × 50 场景（约 25k 轨迹样本用于训练 RD）；LIBERO 环境下 Cosmos‑Policy 与 π_0.5 的评测；以及自行构建的 JailWAM‑Bench，包含 82 条跨任务、跨场景的攻击指令。

**📈 对比分析**

对比方法有 Clean、随机后缀攻击（RSA）和模板攻击（TPA）。实验结果显示，JailWAM 在 LingBot‑VA 上攻击成功率（ASR）达 84.2%（MFR 62.0%，CRR 22.2%），在 Motus 上升至 60.6%，在 Cosmos‑Policy 上为 46.5%。相较于基线（Clean <2%，RSA/TPA <6%），JailWAM 显著提高了物理安全风险检测效果；双路径策略将完整仿真时间从 9.15 小时压缩至 3.66 小时，验证案例 23% 提升。

**⚠️ 局限性**

局限性：① 主要针对具有视觉生成先验的 WAM，非视觉生成模型（如 π_0.5）攻击效果显著降低；② 对抗指令的跨种子稳定性仍有波动，LLM 生成的攻击指令对环境随机性敏感；③ 风险判别器虽快速，但对极端或未见过的危险行为识别仍可能漏判；④ 需要高质量的视觉轨迹映射与 RD 训练数据，构建成本较高。

---

## 248. SCMAPR: Self-Correcting Multi-Agent Prompt Refinement for Complex-Scenario Text-to-Video Generation

**arXiv ID:** 2604.05489 | [PDF](https://arxiv.org/pdf/2604.05489v1)

**作者:** Chengyi Yang `[一作]` (East China Normal University), Ji Liu `[通讯]` (HiThink Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种自纠正多智能体提示优化框架，用于在复杂场景下改进文本到视频（T2V）生成；

**💡 创新点**

创新点在于将提示优化拆分为阶段化的多智能体流程，结合基于场景分类的路由、情境感知的策略合成、语义原子化验证与条件自我修正；

**🔧 技术方法**

主要技术包括使用指令调优的大语言模型（DeepSeek‑V3.2）、BGE‑M3嵌入模型进行原子‑片段匹配，以及多智能体协作的推理与验证；

**📊 数据集**

实验使用了VBench、EvalCrafter、T2V‑CompBench以及新构建的T2V‑Complexity基准；

**📈 对比分析**

与直接提示、Open‑Sora、RAPO等基线相比，在VBench上平均提升约2.67%/2.02%，EvalCrafter上提升3.28%/2.20%，T2V‑CompBench上平均得分提升约0.028，验证了框架的有效性；

**⚠️ 局限性**

局限性包括额外的推理开销、对LLM推理能力的依赖，以及预设的场景标签可能无法覆盖所有新兴复杂场景。

---

## 249. On the Role of Fault Localization Context for LLM-Based Program Repair

**arXiv ID:** 2604.05481 | [PDF](https://arxiv.org/pdf/2604.05481v1)

**作者:** Melika Sepidband `[一作]` (York University), Hadi Hemmati `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在大规模实验中探究了LLM自动程序修复中故障定位上下文粒度的影响，系统评估了文件、元素、行级别的不同扩展策略。

**💡 创新点**

创新点在于首次从文件、元素、行三维度联合研究上下文粒度对修复效果的作用，并发现宽阔语义上下文与精确行定位相结合能最大化性能。

**🔧 技术方法**

采用GPT‑5‑mini模型与LLM检索技术，对SWE‑bench验证集进行61种配置的因子实验。

**📊 数据集**

使用SWE‑bench Verified（500个真实Python bug）作为数据集。

**📈 对比分析**

对比基线和不同扩展策略，发现文件级扩展提升约15‑17×，LLM文件检索比规则检索更优且成本更低；行级扩展往往适得其反。

**⚠️ 局限性**

局限性包括只测试GPT‑5‑mini、仅用真实bug的GT定位作为理想化假设、单次修复尝试和仅关注测试通过率。

---

## 250. Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction

**arXiv ID:** 2604.05477 | [PDF](https://arxiv.org/pdf/2604.05477v1)

**作者:** Yuzhe Zhang `[一作]` (Beijing University of Technology), Haiwei Wang `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于思考–验证–行动–预期（TVAE）闭环框架的GUI自动化代理，能够在动作执行后主动验证结果并在失败时自我纠正。

**💡 创新点**

创新点在于：①将执行结果验证作为核心目标，引入两阶段训练（Robust SFT + GRPO）实现自我监测与恢复；②设计不对称的验证奖励以鼓励诚实判断；③构建可注入失败的Robustness Benchmark并提出Loop Rate与Recovery Success Rate两项新指标。

**🔧 技术方法**

技术方法包括：Vision‑Language 模型（Qwen2.5‑VL）、结构化Chain‑of‑Thought 训练、Group Relative Policy Optimization (GRPO)、离线数据的隐式环境仿真与合成失败轨迹、复合奖励函数（Action, Effect, Verification）。

**📊 数据集**

使用的数据集包括：AndroidControl‑High、AITW‑Gen、GUI Odyssey（离线基准）；MiniWoB++、AndroidWorld（在线基准）；以及自构造的失败注入测试集。

**📈 对比分析**

与现有开源基线比较，-3B与-7B在离线指标上均取得新SOTA（例如-7B的Sim‑TSR 23.5%/ASO 1.09），在Robustness Benchmark中Recovery Success Rate分别达到51.1%和52.5%，在在线基准MiniWoB++与AndroidWorld上分别获得35.6%/12.6%（-3B）和59.7%/25.1%（-7B），明显优于同规模基线。

**⚠️ 局限性**

局限性：仅针对失效保持不变的“等价无效”模式；未覆盖非等价失效（如意外跳转、崩溃）；以及在任务长度较大时，单步自我纠正效果下降，缺乏全局规划能力。

---

## 251. A Synthetic Eye Movement Dataset for Script Reading Detection: Real Trajectory Replay on a 3D Simulator

**arXiv ID:** 2604.05475 | [PDF](https://arxiv.org/pdf/2604.05475v1)

**作者:** Kidus Zewde `[一作]`, Simiao Ren `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套从真实眼球轨迹回放生成合成眼动视频的完整管道，并公开发布了包含 12 小时平衡阅读与对话眼动数据集。

**💡 创新点**

① 通过真实轨迹回放取代传统合成生成，保持细粒度眼动动态；② 对 3D 眼动模拟器的视角映射进行了量化评估，发现并解释了在阅读尺度下的灵敏度限制；③ 提供了开放源码的生成流程、数据集与评估工具。

**🔧 技术方法**

使用 MediaPipe 人脸与虹膜关键点检测、线性校准与速度修正、每主体归一化、循环切片、Playwright 控制 Unity WebGL 眼动模拟器；评估采用 Kolmogorov–Smirnov 检验、速度/固定/扫动统计、光学误差与相关系数。

**📊 数据集**

参考数据集：11 条阅读视频、5 条对话视频共 16 条轨迹（约 310k 帧）；生成数据集：144 条会话（72 阅读 + 72 对话），每条 5 分钟，25fps，合计 12 小时 1.7GB。

**📈 对比分析**

通过 KS D 统计对生成轨迹与源轨迹在速度、固定持续时间、扫动幅度上的分布匹配，读取类 D ≤ 0.136、对话类 D ≤ 0.079；光学匹配显示平均位置误差 0.053，真实幅度 30–42%；模拟器的时间相关性几乎为零，说明仅保留了时序结构。

**⚠️ 局限性**

① 模拟器缺少头部运动，导致阅读尺度下眼球幅度被压缩（仅 30–42%）；② 读取与对话数据重用率不均，影响分布相似度；③ 仅提供眼动轨迹与视频，未包含头姿或完整面部；④ 尚未验证合成数据是否能有效训练下游行为识别模型。

---

## 252. CUE-R: Beyond the Final Answer in Retrieval-Augmented Generation

**arXiv ID:** 2604.05467 | [PDF](https://arxiv.org/pdf/2604.05467v1)

**作者:** Siddharth Jain `[一作]` (Intuit), Venkat Narayan Vedam `[通讯]` (Intuit)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了CUE‑R框架，通过对检索证据执行移除、替换、重复等干预，并结合可观测检索‑使用轨迹、正确性、代理准确定信、置信误差和轨迹偏差等多维度指标，对单次检索增补生成（RAG）中每条证据的操作效用进行评估。

**💡 创新点**

首次提出基于干预的证据效用评估方法，将干预、可观测轨迹和多轴效用（正确性、准确定信、置信误差、轨迹偏差）结合，并构建操作证据角色分类表，揭示单一干预下多维度证据影响以及多证据交互的非加性效应。

**🔧 技术方法**

干预式评估（remove/replace/duplicate）、可观测检索‑使用轨迹记录、代理准确定信评分、置信误差计算、轨迹偏差度量、bootstrap 统计检验、轻量化 RAG 流程（BM25+单次推理）以及交互式证据角色分类。

**📊 数据集**

HotpotQA（多跳）和 2WikiMultihopQA 数据集；使用 Qwen‑3 8B 和 GPT‑5.2 两大模型；还做了零检索对照和两证据交互消融实验。

**📈 对比分析**

在同一问题集上分别执行原始、移除、替换、重复干预，计算正确率、答案 F1、准确定信、置信误差和轨迹偏差，并用 bootstrap 统计检验差异。结果表明移除/替换显著降低正确率和准确定信、提高置信误差并产生大轨迹偏差；重复干预对正确率影响极小，但仍显著改变准确定信和轨迹；两证据联合移除的降效远大于单独移除，体现非加性。跨模型跨数据集保持一致，证明干预评估能捕捉答复级评估忽略的证据效用。

**⚠️ 局限性**

仅评估浅层单次 RAG 轨迹，未能捕获内部推理；使用标题级重叠的代理准确定信，缺乏句子级精细度；置信度基于模型自报，易失真；干预仅限于移除/替换/重复，未覆盖更复杂干预；检索使用 BM25，缺乏对高阶检索的泛化；实验规模有限；目标选择启发式；仅评估单证据干预，未系统探索多证据组合。

---

## 253. Human Interaction-Aware 3D Reconstruction from a Single Image

**arXiv ID:** 2604.05436 | [PDF](https://arxiv.org/pdf/2604.05436v1)

**作者:** Gwanghyun Kim `[一作]` (Seoul National University), Se Young Chun `[通讯]` (Seoul National University)

**通讯引用:** 2600 | [OpenAlex ID](https://openalex.org/A5052523460)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种单图多人体3D重建框架HUG3D，包含正交视角变换（Pers2Ortho）、群体-实例多视角扩散（HUG-MVD）以及物理可交互几何优化（HUG-GR），实现完整几何与纹理的高保真多人体交互重建。

**💡 创新点**

创新点包括：① 通过Pers2Ortho将透视图统一为正交空间，消除尺度与畸变；② 结合群体与实例先验的多视角扩散模型，用交互先验补全遮挡；③ 在几何优化中引入接触约束、穿插损失等物理约束，保证多人体交互的物理可行性；④ 同时生成纹理并进行多视角融合，提升表面细节与真实感。

**🔧 技术方法**

使用技术包括：SMPL-X拟合与BUDDI/Robudd相机估计、正交视角变换、基于Stable Diffusion 2.1的多视角扩散与ControlNet、物理交互约束的几何优化、Occlusion-aware blending等。

**📊 数据集**

训练集为Hi4D（多人体监督）、THuman2.0与CustomHumans（单人体多姿态）；评估集为MultiHuman；同时利用Sapiens等补全深度与法线。

**📈 对比分析**

与单人方法(SIFU, SiTH, PSHuman, ECON)以及多视角/视频多人体方法(DeepMultiCap, Multiply)在CD、P2S、NC、F-score、bbox-IoU、CP、PSNR、SSIM、LPIPS等指标上进行对比。HUG3D在所有几何指标中均取得最低CD/P2S、最高NC、最高CP，并在纹理指标上获得最高PSNR/SSIM、最低LPIPS，尤其在遮挡区表现突出。

**⚠️ 局限性**

局限性：仅针对人体之间的遮挡，无法处理外部物体遮挡；单图深度不确定性导致极端遮挡或深度模糊时可能失败；在高度重叠或极端姿态的场景下仍可能出现几何或纹理误差。

---

## 254. Cross-Stage Attention Propagation for Efficient Semantic Segmentation

**arXiv ID:** 2604.05431 | [PDF](https://arxiv.org/pdf/2604.05431v1)

**作者:** Beoungwoo Kang `[一作]` (Hyundai Mobis), Beoungwoo Kang `[通讯]` (Hyundai Mobis)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5050416000)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种跨阶段注意力传播（CSAP）解码框架，只在最深特征尺度上计算注意力，然后将注意力图传播至更浅的层级，完全跳过后者的查询-键计算，从而实现轻量化语义分割。

**💡 创新点**

创新点在于利用不同尺度注意力分布高度相关的特性，仅在深层一次性计算注意力并通过可学习投影适配到浅层，显著减少解码器计算量并保持多尺度上下文推理能力。

**🔧 技术方法**

采用多头注意力、空间平均池化、投影变换（Proj→2、Proj→3）、深度卷积FFN以及与MSCAN骨干相结合的轻量解码器；整体实现基于Vision Transformer/卷积混合思路。

**📊 数据集**

在ADE20K、Cityscapes和COCO‑Stuff 164K三个公开语义分割基准数据集上进行评估。

**📈 对比分析**

与SegNeXt、MetaSeg、EDAFormer、TopFormer等轻量化方法对比，CSAP‑Tiny在ADE20K上达42.9% mIoU、21.5 GFLOPs（比SegNeXt低16.8% FLOPs），Cityscapes 80.5% mIoU、21.5 GFLOPs，COCO‑Stuff 40.9% mIoU、5.5 GFLOPs，整体性能优于同类方法且计算成本更低。

**⚠️ 局限性**

局限在于目前仅在单一深层作为注意力来源；对多尺度或跨任务的适用性、对极低分辨率或极大规模模型的可扩展性尚未充分验证，并且传播方式可能在某些细粒度语义上失去局部细节。

---

## 255. Pre-Execution Safety Gate & Task Safety Contracts for LLM-Controlled Robot Systems

**arXiv ID:** 2604.05427 | [PDF](https://arxiv.org/pdf/2604.05427v1)

**作者:** Ike Obi `[一作]` (Purdue University), Byung-Cheol Min `[通讯]` (Purdue University)

**通讯引用:** 1900 | [OpenAlex ID](https://openalex.org/A5076467173)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SafeGate neurosymbolic 预执行安全门，利用 LLM 分析自然语言指令并提取 ISO 标准的安全属性，决定是否授权、延期或拒绝；通过任务安全合同、Z3 SMT 验证以及运行时监控，防止机器人执行危险指令。

**💡 创新点**

创新点在于：① 将 ISO 13482/12100 标准转化为 Hazard Analysis Matrix、Hazard Template Library 与 Deterministic Decision Gate 的三阶段流程；② 结合神经网络与符号推理，实现自然语言的结构化安全分析；③ 通过 Task Safety Contracts（不变式、守卫、终止条件）实现静态与动态双重安全验证；④ 支持动态扩展未覆盖的危险模板。

**🔧 技术方法**

使用的技术包括：大语言模型（GPT‑4o、Gemini 2.5‑Flash 等）；ISO 标准抽象的安全分析矩阵；Hazard Binding Layer 与决策门；Z3 SMT 求解器进行约束检查；静态计划验证（符号执行）与运行时监控；以及多层安全合同编译。

**📊 数据集**

使用的数据集为：230 个专家标注的机器人任务指令（assistive、navigation、manipulation 三类，简单/中等/复杂三级）；30 个 AI2‑THOR 仿真场景；以及真实机器人实验场景。

**📈 对比分析**

通过与 RoboGuard、SELP、LLM baseline（GPT‑4o、Gemini 2.5‑Flash）对比，采用 AR‑S%、AR‑U%、DR%、CR%、TC%、F1 等指标；SafeGate 的 AR‑U% 为 0，AR‑S% 为 92.8%，DR% 为 9.2%，F1 为 97.1%，显著优于其他方法，并大幅降低误授权率。

**⚠️ 局限性**

局限性：对无法映射的危险仍需人工扩展模板；对未知信息的延期处理可能导致用户体验下降；未提供对被拒绝指令的修复或重写策略；在极其动态或未见过的环境中，安全门的覆盖仍有限。

---

## 256. Synergizing Efficiency and Reliability for Continuous Mobile Manipulation

**arXiv ID:** 2604.05430 | [PDF](https://arxiv.org/pdf/2604.05430v1)

**作者:** Chengkai Wu `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Boyu Zhou `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2975 | [OpenAlex ID](https://openalex.org/A5101982552)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种统一的连续移动操纵框架，集成可靠性感知的轨迹规划、基于阶段的安全权衡控制以及分层初始化，能够在复杂环境下实现高效率且可靠的连续任务执行。

**💡 创新点**

创新点包括：①将可靠性约束（主动感知时长、补偿余量、预抓/后放安全运动、弹性碰撞球）嵌入时空轨迹优化；②设计时间保证主动感知与补偿余量区；③使用安全映射的可变碰撞球来处理意图接触；④引入基于相位的安全变形轨迹warping与平滑权重切换的MPC控制器，实现在线姿态补偿与全局轨迹跟踪的无缝切换；⑤构建分层多任务路径规划（Hybrid A* + 分层图），为高维非线性优化提供高质量初始化。

**🔧 技术方法**

技术手段包括：时空轨迹优化（分段多项式+约束求解）、分层初始化（Hybrid A* 规划底盘、分层图优化手臂）、可微安全约束（视角、碰撞、补偿余量）、基于MPC的双层控制（底盘MPC+手臂MPC）、安全变形轨迹warping、相位切换权重、在线感知与重规划。

**📊 数据集**

主要使用自建的真实世界环境（办公室、动态人行道）以及基于Isaac Sim的四个场景（办公室、住宅、咖啡馆、简单场景）进行实验，并在真实机器人上完成10个堆垛、6个连续pick‑place任务等。

**📈 对比分析**

与三种最先进方法（Burgess、Reister、Thakar）对比，方法在所有场景下实现99.86%任务成功率、100%场景成功率，且在SSCT、平均速度等指标上明显优于对手（例如在办公场景中任务成功率从30%提升至100%，效率提升15%以上）。实验还展示了在任务不确定性下的实时重规划与补偿能力。

**⚠️ 局限性**

局限性包括：①长时程规划计算量大，重规划频率受限；②对传感器鲁棒性和姿态估计误差的依赖仍然存在；③对极端大范围姿态误差或极端动态障碍的鲁棒性尚未完全验证；④模型预测控制的实时性要求较高，需强硬件支持。

---

## 257. A Weak-Signal-Aware Framework for Subsurface Defect Detection: Mechanisms for Enhancing Low-SCR Hyperbolic Signatures

**arXiv ID:** 2604.05490 | [PDF](https://arxiv.org/pdf/2604.05490v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 258. CODESTRUCT: Code Agents over Structured Action Spaces

**arXiv ID:** 2604.05407 | [PDF](https://arxiv.org/pdf/2604.05407v1)

**作者:** Myeongsoo Kim `[一作]` (AWS AI Labs), Murali Krishna Ramanathan `[通讯]` (AWS AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CodeStruct 框架，将代码库视为结构化动作空间，允许 LLM 代码代理通过 AST 节点进行读写操作；

**💡 创新点**

创新点在于把 AST 节点当作可直接操作的命名实体，提供两种结构化原语 readCode 与 editCode，消除文本匹配的脆弱性并保证语法合法；

**🔧 技术方法**

主要技术包括 AST 解析、结构化选择器、语法验证、工具接口（MCP）以及多模型评估；

**📊 数据集**

使用的数据集为 SWE‑Bench Verified（Python Bug 修复）和 CodeAssistBench（多语言交互式辅助任务）；

**📈 对比分析**

与传统文本基础基线相比，CodeStruct 在 SWE‑Bench Verified 上 Pass@1 提升 1.2–5.0%（GPT‑5‑nano 提升 20.8%），Token 消耗下降 12–38%，成本降低 19–33%；在 CodeAssistBench 上准确率提升 0.8–4.4%，成本降低最高 33%；效果随模型容量变化，较弱模型获益更显著；

**⚠️ 局限性**

局限性包括：只能处理语法合法的文件，Python 为主；对复杂静态语言支持有限；未覆盖代码审查、测试生成等额外任务；

---

## 259. MARS-Dragonfly: Agile and Robust Flight Control of Modular Aerial Robot Systems

**arXiv ID:** 2604.05499 | [PDF](https://arxiv.org/pdf/2604.05499v1)

**作者:** Rui Huang `[一作]` (National University of Singapore), Lin Zhao `[通讯]` (National University of Singapore)

**通讯引用:** 13711 | [OpenAlex ID](https://openalex.org/A5110190620)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一套基于龙蜻蜓行为的多模组无人机系统（MARS）的集成对接–锁定–分离硬件机制，并给出了统一的虚拟四旋翼模型抽象，配合预测控制与动态分配，实现了在多模组配置下的平滑对接、无姿态扰动分离、精确轨迹跟踪、敏捷飞行与负载搬运。

**💡 创新点**

创新点包括：① 单一微伺服驱动的自对齐磁性对接与被动锁定机制，消除对接检测与多连杆/双伺服的需求；② 通过优化磁阵列和角度选择实现更强的吸附与排斥力；③ 将任意配置的 MARS 抽象为可变臂的虚拟四旋翼，显式引入力/扭矩极限；④ 采用两阶段预测控制与均衡动态分配，避免传统 PID+规则分配产生的离散指令与姿态误差积累。

**🔧 技术方法**

技术手段涵盖：磁阵列优化（Halbach 结构与遗传算法）、动力学建模（虚拟四旋翼非线性动力学）、模型预测控制（约束优化）、均衡控制分配（方差最小化 QP）、仿真平台 Gazebo+PX4 SITL 与实际实验室小型四旋翼（Iris 1.5 kg 机体），并在 NVIDA Jetson Orin 上实现实时解算。

**📊 数据集**

未使用公开数据集，仅在 Gazebo 与 PX4 SITL 进行仿真，实验数据来自本实验室搭建的 1.55 kg 单体四旋翼与多模组配置的现场飞行记录。

**📈 对比分析**

与 Modquad、ModQuad-Vi、Ref 等基线方法相比，MARS-Dragonfly 在对接高度误差从 0.577 m 降至 0.042 m、分离最大俯仰角从约 28° 降至 5.8°、分离时间从 5 s 缩短至 0.5 s、10 圈圆轨迹平均定位误差仅 0.072 m、总体平均误差 0.0896 m，并且在携带 600 g 负载时仍保持 0.079 m 的位置误差与 0.79° 的姿态误差，显著优于传统 PID+规则分配方案。

**⚠️ 局限性**

局限性包括：① 虚拟四旋翼模型的逼近误差（尤其在不对齐或大角度运动时）；② 需要在所有无人机上实现低延迟 Wi‑Fi 通信，扩展至 50+ 单元时延迟仍可接受但需更高频宽；③ 对接与分离硬件受限于磁性力场，极端风速或大扰动时可能仍出现锁定/分离失败；④ 负载模型仅为刚性连接，无法覆盖吊挂或绳索动力学，需进一步扩展。

---

## 260. AttnDiff: Attention-based Differential Fingerprinting for Large Language Models

**arXiv ID:** 2604.05502 | [PDF](https://arxiv.org/pdf/2604.05502v1)

**作者:** Haobo Zhang `[一作]` (Zhejiang University of Technology), Meng Han `[通讯]` (Zhejiang University)

**通讯引用:** 10890 | [OpenAlex ID](https://openalex.org/A5031867321)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种后置指纹化框架 AttnDiff，通过在轻微扰动的提示对中诱发语义冲突，提取差分注意力模式来为大型语言模型生成指纹。

**💡 创新点**

创新点在于利用差分注意力动态捕捉模型内部信息路由行为，并将其压缩为光谱描述符；随后使用中心化线性 CKA 进行架构无关的相似度评估，从而在微调、剪枝、融合和蒸馏等真实模型洗劫场景下保持高度鲁棒性。

**🔧 技术方法**

核心技术包括：白盒提取自注意力权重、构造最小词汇替换的提示对、对齐池化、截断奇异值分解生成光谱指纹，以及使用中心化线性 CKA 计算模型间相似度。

**📊 数据集**

使用了包含 60 条跨六个领域（代码、数学、经济学、医学、日常问答、安全对齐）的提示集进行评估，并在 Llama‑2/3、Qwen2.5、Gemma、Mistral 等模型的多种微调、剪枝、融合、蒸馏版本上进行实验。

**📈 对比分析**

与参数基（PCS/ICS）、表示基（Logits、REEF）、对抗基（ProFlingo）以及语义基（LLMMap）等基线对比，AttnDiff 在所有洗劫操作下对相关模型保持 >0.98 的相似度，对无关架构保持 <0.22，显著优于其他方法。

**⚠️ 局限性**

局限性：需要白盒访问内部注意力；在纯黑盒 API 下不可直接使用；缺乏针对洗劫操作到指纹变化的完整理论模型，且在指纹被攻击后可能需要刷新提示集。

---

## 261. Auditable Agents

**arXiv ID:** 2604.05485 | [PDF](https://arxiv.org/pdf/2604.05485v1)

**作者:** Yi Nian `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**通讯引用:** 3484 | [OpenAlex ID](https://openalex.org/A5057711796)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对 LLM 代理的五维审计框架和 Auditability Card，并通过生态系统扫描、运行时拦截与签名日志、以及基于令牌水印的恢复实验系统性验证其必要性。

**💡 创新点**

首次将审计可操作性拆解为行动可恢复、生命周期覆盖、策略可检查、责任归属、证据完整五维度，并提出三类机制（检测、执行、恢复）与之对应的实现路径。

**🔧 技术方法**

利用 agent‑audit 静态安全扫描、Aegis 预执行拦截与 Ed25519 哈希链签名日志、以及 IET 令牌水印技术进行实验。

**📊 数据集**

使用六个开源代理项目（OpenHands、Generative Agents、SWE-agent 等）、48 个安全攻击案例与 500 个正常调用、以及 4–6 代理的多代理实验场景。

**📈 对比分析**

通过安全发现统计、执行延迟（P95 14.7 ms，P99 23.1 ms）和恢复精度（IoU≈0.9、归因≈0.94）评估，证明低开销且恢复效果显著。

**⚠️ 局限性**

仅验证单一机制类、缺乏端到端完整审计、数据局限于开源项目、阈值校准缺失、仅处理结构化策略、未充分考虑隐私与实体约束等限制。

---

## 262. Can We Trust a Black-box LLM? LLM Untrustworthy Boundary Detection via Bias-Diffusion and Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.05483 | [PDF](https://arxiv.org/pdf/2604.05483v1)

**作者:** Xiaotian Zhou `[一作]` (Worcester Polytechnic Institute), Xiaozhong Liu `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 3875 | [OpenAlex ID](https://openalex.org/A5101985030)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于图神经网络和多智能体强化学习的算法 GMRL‑BD，用于在仅有黑盒访问权限且查询受限的情况下识别大语言模型（LLM）在知识图谱（KG）中哪些主题会产生偏见或不可信回答。

**💡 创新点**

创新点包括：① 通过构建 Wikipedia 类别树的 KG 并假设偏见在图中会扩散，利用图结构捕捉偏见扩散规律；② 设计多智能体协同强化学习框架，智能体共享奖励信息并通过 AgentLinkHub 进行动态路径规划，从而显著降低查询次数；③ 在节点嵌入中结合 SAGE 聚合与注意力机制，提升对偏见主题的判别能力。

**🔧 技术方法**

技术细节：图神经网络（GraphSAGE + 关注层）用于节点表示；多智能体 Q‑学习与贝塔/α 约束的奖励函数；强化学习训练阶段使用 ε‑greedy 与衰减；推理阶段采用多智能体协作与 AgentLinkHub；数据预处理采用 LLM 生成 QA 对并注入偏见；使用 PyTorch、NetworkX 进行实现。

**📊 数据集**

数据集：LBID（LLM Bias Identification Dataset），包含 871,854 个 Wikipedia 类别节点、4,983,336 条边，22,608 个注入偏见的主题，113,040 条 QA 对，涵盖 6 种主流 LLM（Llama‑2、Vicuna、Falcon、Qwen2、Gemma2、Yi‑1.5）以及 6 个任务域（AI、Culture、Economy、Politics、Education、Immigration）。

**📈 对比分析**

与基准方法比较：DFS、Graph‑enhanced Attention Q‑Network、Uniform Exploration、Unexploration‑driven Reward 等；实验显示 GMRL‑BD 在 BNC（Bias Node Cover Rate）上取得最高分（如 Llama‑2 在 Immigration 上 90.34%），且查询次数平均在 10–30 次，显著低于单智能体或无关注层的变体。多智能体数量增加时，步损失下降、胜率上升，验证协作提升效果。

**⚠️ 局限性**

局限性：① 仅针对英文 LLM 与英文 Wikipedia，缺乏跨语言适用性；② 需要大量 GPU 计算与高查询成本，尤其是多智能体训练阶段；③ 依赖预先注入的偏见标签，若 LLM 产生的偏见不在标签集内可能漏检；④ 目前仅处理主题级别的可信度，无法直接对单条查询结果进行可信度判断。

---

## 263. Adaptive Serverless Resource Management via Slot-Survival Prediction and Event-Driven Lifecycle Control

**arXiv ID:** 2604.05465 | [PDF](https://arxiv.org/pdf/2604.05465v1)

**作者:** Zeyu Wang `[一作]` (University of California, Los Angeles), Qiyuan Tian `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一套自适应服务器无关资源管理器（ASRM），通过事件驱动架构和概率建模，双重策略（动态空闲时长调整 + 预测性请求等待）来降低冷启动、提升资源利用率并优化成本。

**💡 创新点**

创新点在于：① 将空闲时长阈值与请求等待概率联动，利用滑动窗口和核密度估计预测“槽”存活时间；② 采用无锁数据结构和自适应超时策略实现高并发下的请求等待；③ 通过闭环可观测性、分布式追踪和多云抽象层实现跨平台自适应调度。

**🔧 技术方法**

主要技术包括：gRPC服务接口、事件驱动状态机、滑动窗口聚合、概率预测（Kernel Density Estimation）、异步事件处理、无锁并发控制、在线聚类+增量PCA、波形分解、动态超时、成本模型、断路器模式、分布式检查点、跨云适配层。

**📊 数据集**

使用了三种代表性工作负载数据集：Dataset‑1（高频均匀）、Dataset‑2（周期性突发）、Dataset‑3（稀疏不规则），数据来源为生产环境的请求日志。

**📈 对比分析**

与 OpenWhisk‑Default、SOCK、Firecracker‑snap、Knative‑KPA 等基线方案对比，采用 CSRR、RUE、ARL、CPI 四项指标评估。ASRM 在 CSRR 上提升至 51.2%，RUE 提升约 62%，ARL 更稳定，CPI 接近 2 倍，整体性能优于所有基线。

**⚠️ 局限性**

局限性：依赖历史请求模式的准确预测，对极端突发或高变异工作负载的适应性仍有限；实现复杂度高，需持续监控与调优；跨云抽象层虽兼容多平台，但在极端多租户或特殊计费模型下可能出现误差。

---

## 264. Content Fuzzing for Escaping Information Cocoons on Digital Social Media

**arXiv ID:** 2604.05461 | [PDF](https://arxiv.org/pdf/2604.05461v1)

**作者:** Yifeng He `[一作]` (University of California), Hao Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 111859 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个内容侧的置信度引导模糊化框架，利用LLM生成语义不变的帖子重写，以改变机器立场检测标签，从而帮助内容跨越信息茧房；

**💡 创新点**

将软件模糊化技术迁移至文本重写，通过置信度反馈、温度自适应调度与种子优先调度等机制，自动发现可逃逸信息茧房的语义保持重写；

**🔧 技术方法**

使用大型语言模型（Gemini‑2.5‑Flash‑Lite）进行受限重写生成，结合置信度回馈、温度调度、最小堆种子调度，以及BERTScore、NLI等评估指标；

**📊 数据集**

在SemEval2016‑Task6、VAST和C‑STANCE（中英）三大社交媒体数据集上进行实验；

**📈 对比分析**

与BERT、RoBERTa、零样本Gemini、COLA等四类立场检测器对比，实验显示逃逸成功率显著提升（最高超过50%），语义完整度高（BERTScore>0.90），流畅性良好（PPL比率<1）；

**⚠️ 局限性**

仅针对立场检测器，缺乏真实平台部署评估，依赖模型置信度反馈且未对LLM内部机制做深度优化，且人工验证规模有限。

---

## 265. Geometrical Cross-Attention and Nonvoid Voxelization for Efficient 3D Medical Image Segmentation

**arXiv ID:** 2604.05515 | [PDF](https://arxiv.org/pdf/2604.05515v1)

**作者:** Chenxin Yuan `[一作]` (Shenzhen Institute for Advanced Study), Pin-Han Ho `[通讯]` (University of Waterloo)

**通讯引用:** 9408 | [OpenAlex ID](https://openalex.org/A5089628133)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GCNV-Net，一种利用非空体素化、三向动态非空体素Transformer和几何交叉注意力的3D医学图像分割框架；

**💡 创新点**

创新点在于（1）非空体素化显式剔除无信息背景，显著减少计算；（2）三向动态分区Transformer在三正交解剖平面上动态划分体素，降低复杂度；（3）几何交叉注意力在多尺度融合时保持绝对几何位置信息；

**🔧 技术方法**

采用了深度卷积、Transformer自注意力、动态窗口划分、几何位置编码、稀疏体素处理等技术；

**📊 数据集**

在BraTS2021、ACDC、MSD Prostate、MSD Pancreas、AMOS2022等五大公开医学分割基准上进行评估；

**📈 对比分析**

与nnU-Net、SwinUNETR、MedNeXt、E2ENet、SegFormer3D等多种最先进方法对比，GCNV-Net在Dice、IoU、HD95、NSD上均取得最高分，并在FLOPs、推理时延上显著优于对手，整体质量-效率折中优势显著；

**⚠️ 局限性**

主要局限包括：需要额外的体素化和动态窗口设计复杂度较高；对体素化阈值和动态分区策略的敏感性未系统评估；在极大尺寸或多模态数据上可扩展性待验证。

---

## 266. CLIP-Guided Data Augmentation for Night-Time Image Dehazing

**arXiv ID:** 2604.05500 | [PDF](https://arxiv.org/pdf/2604.05500v1)

**作者:** Xining Ge `[一作]` (Hangzhou Dianzi University), Shuhong Liu `[通讯]` (University of Tokyo)

**通讯引用:** 20234 | [OpenAlex ID](https://openalex.org/A5069268661)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套统一框架，利用CLIP视觉编码器筛选与目标域相近的外部样本，采用两阶段训练NAFNet，并在推理时结合TLC本地增强、×8自集成与快照加权融合，以提升夜间图像去雾效果。

**💡 创新点**

创新点在于将域一致性数据筛选、分阶段训练和推理时多重增强协同设计，而非单纯加大网络容量；通过CLIP相似度筛选外部样本并在两阶段训练中先适配目标域，再引入筛选后数据，最终在推理时实现TLC、×8自集成与快照融合，形成高效且易复现的完整流程。

**🔧 技术方法**

使用了CLIP视觉编码器进行样本相似度筛选、NAFNetLocal网络进行去雾恢复、TLC（局部信息聚合）增强、×8自集成、快照加权融合以及AdamW优化器、MSE损失和余弦学习率调度等技术。

**📊 数据集**

训练数据主要来自NTHazy夜间双目数据（25对），外部数据从I-HAZE（21对）、Dense-Haze（10对）、HAZE1K（3对）筛选后共59对；公开评测采用NHM-20数据集。

**📈 对比分析**

在NHM-20公开评测中，以Y通道PSNR/SSIM为主指标，方法相较原始雾图提升约1.5dB PSNR和0.02 SSIM；但RGB通道指标和LPIPS未超过输入基准，表明改进主要体现在亮度结构恢复上。

**⚠️ 局限性**

主要局限在于颜色保真与感知质量提升不足，外部样本筛选仍可能无法完全消除跨域差距，且当前方法更侧重亮度恢复，需进一步改进色彩一致性与感知效果。

---

## 267. Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models

**arXiv ID:** 2604.05497 | [PDF](https://arxiv.org/pdf/2604.05497v1)

**作者:** Keuntae Kim `[一作]` (Hanyang University), Yong Suk Choi `[通讯]` (Hanyang University)

**通讯引用:** 3129 | [OpenAlex ID](https://openalex.org/A5052803083)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并量化了扩散式多模态大语言模型(dMLLM)在Chain-of-Thought推理过程中的行为，并提出针对其早期答复生成和视觉依赖弱化问题的改进方法。

**💡 创新点**

创新点在于提出Position & Step Penalty（PSP）通过在早期时间步对后置答案位置加罚，抑制过早生成答案；以及Visual Reasoning Guidance（VRG）以类分类器无条件引导的方式放大视觉条件对推理的影响。

**🔧 技术方法**

采用扩散式语言模型的反向恢复机制、基于视觉提示依赖度（PDM）的分析、以及改进的remasking策略；实现PSP与VRG的训练无关、推理阶段可插拔。

**📊 数据集**

在M3CoT、ScienceQA、MMBench、V*Bench四个多模态推理基准上进行实验评估。

**📈 对比分析**

与低置信度、熵、边缘等原始remasking策略以及AR VLM的CCoT/DDCoT方法对比，PSP+VRG在所有指标上均提升约3%以上准确率，并在仅使用1/4扩散步长时实现3倍以上的推理速度提升。

**⚠️ 局限性**

主要局限在于对视觉提示的放大仍需手动调参，过大可能导致模型过度依赖视觉信息；目前仅验证在两款dMLLM上，尚未证明在更大规模或其他扩散模型上的普适性。

---

## 268. Selecting a Maximum Solow-Polasky Diversity Subset in General Metric Spaces Is NP-hard

**arXiv ID:** 2604.05495 | [PDF](https://arxiv.org/pdf/2604.05495v1)

**作者:** Michael T. M. Emmerich `[一作]` (University of Jyvaskyla), André H. Deutz `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了在一般度量空间中，固定大小子集的 Solow–Polasky 多样性最大化问题是 NP 难的。

**💡 创新点**

采用从独立集问题的归约，并构造仅含两种非零距离的度量空间，配合严格单调性证明，首次完成该问题的 NP 难性证明。

**🔧 技术方法**

使用度量构造、矩阵逆分析、连续性与严格单调性推导、Neumann 级数展开等技术。

**📊 数据集**

未使用实验数据集，全部为理论证明。

**📈 对比分析**

无实验比较，仅提供理论上的难度证明，不涉及性能评估。

**⚠️ 局限性**

仅证明了最坏情况的难度，对近似性、参数化复杂度或更结构化度量空间的复杂性尚未给出。

---

## 269. OntoTKGE: Ontology-Enhanced Temporal Knowledge Graph Extrapolation

**arXiv ID:** 2604.05468 | [PDF](https://arxiv.org/pdf/2604.05468v1)

**作者:** Dongying Lin `[一作]` (Northeastern University), Xiaochun Yang `[通讯]` (Northeastern University)

**通讯引用:** 2317 | [OpenAlex ID](https://openalex.org/A5079763362)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 OntoTKGE 框架，通过构建 ontology‑view KG 并将其知识与时间演化知识融合，显著提升 TKG 预测中的稀疏实体推理能力。

**💡 创新点**

创新点在于：①首次将 ontological 视角的层次概念知识直接嵌入 TKG 预测流程；②设计全局与局部双视图编码器，并通过对比增强的门控融合实现多视角一致性；③在训练中引入层级归纳约束和对比损失，强化实体嵌入的层级与语义一致。

**🔧 技术方法**

技术上主要使用 CompGCN 进行层级知识编码、对比学习、门控融合、以及多模型适配的 encoder‑decoder 架构，结合 LLM（GPT‑4o‑mini）与 Wikidata 进行自动 ontology 构建。

**📊 数据集**

在四个主流 TKG 数据集（ICEWS14、ICEWS18、ICEWS05‑15、GDELT）上进行实验。

**📈 对比分析**

与多种基线（RE‑GCN、TiRGN、RETIA、LogCL、HisRES、JOIE、HyperCL、ANEL、LLM‑DA）对比，OntoTKGE 在 MRR、Hits@1/10 上均实现显著提升，常年领先或超过现有 SOTA 方法。

**⚠️ 局限性**

局限性包括：①需要额外构建 ontology‑view KG，耗费 LLM 计算资源；②对大型知识图的扩展性仍待验证；③对极高维或极稀疏场景的鲁棒性尚未充分探究。

---

## 270. Optimizing OpenFaaS on Kubernetes: Comparative Analysis of Language Runtimes and Cluster Distributions

**arXiv ID:** 2604.05496 | [PDF](https://arxiv.org/pdf/2604.05496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 271. Top-K Retrieval with Fixed-Size Linear-Attention Completion: Backbone- and KV-Format-Preserving Attention for KV-Cache Read Reduction

**arXiv ID:** 2604.05438 | [PDF](https://arxiv.org/pdf/2604.05438v1)

**作者:** Yasuto Hoshi `[一作]` (KIOXIA Corporation), Jun Deguchi `[通讯]` (KIOXIA Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种检索-补完注意力机制，能够在保持原始模型权重和KV缓存格式不变的前提下，减少解码时KV缓存读取量。

**💡 创新点**

创新点在于使用预填阶段生成的固定大小线性注意力摘要（feature map摘要），并通过单次归一化将检索到的关键字与补完项融合，避免了子集归一化带来的偏差。

**🔧 技术方法**

核心技术包括：正向特征映射（learned positive feature maps）用于近似softmax核；一阶减法完成（subtractive completion）仅补偿未检索的中间区块；以及基于预填缓存的单次读取摘要。

**📊 数据集**

实验使用了长文本基准数据集RULER（4k/8k/16k上下文）和BABILong（QA1–QA5），并在Llama‑3.2‑1B‑Instruct与Qwen3‑1.7B两大模型上进行验证。

**📈 对比分析**

与全精确注意力及仅检索（selection）方法比较，所提方法在相同token‑等价KV读取预算下，在高熵头部显著提升准确率（RULER 16k时提升约0.04，BABILong 16k时提升约0.02），且在相同质量下检索量可缩减约1.8×。

**⚠️ 局限性**

局限性包括：需要额外的预填摘要缓存和一次性读取开销，计算负担主要在检索的特征映射评估；在低熵头部或已足够稀疏的模型中，补完可能导致轻微退化；且实际系统性能受检索后端实现和KV读取策略影响。

---

## 272. Bridging Natural Language and Microgrid Dynamics: A Context-Aware Simulator and Dataset

**arXiv ID:** 2604.05429 | [PDF](https://arxiv.org/pdf/2604.05429v1)

**作者:** Tinko Sebastian Bartels `[一作]` (Chinese University of Hong Kong-Shenzhen), Tongxin Li `[通讯]` (Chinese University of Hong Kong-Shenzhen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 OpenCEM Simulator 与配套语言丰富的数据集，构建了可将自然语言上下文与物理动力学耦合的微电网数字孪生平台，支持研究人员在此环境下开发和验证基于大语言模型的能源管理与控制策略。

**💡 创新点**

创新点包括：①首次将非结构化自然语言事件与电网运行数据同步，并在仿真器中原生处理；②提供完整的公开语言丰富微电网数据集，为上下文感知模型提供训练和评估基础；③设计模块化、可扩展的 API，允许轻松替换电池、逆变器、网格等子模型并集成 LLM 处理器；④演示了基于 LLM 提取的上下文特征可显著提升负荷预测与能源成本优化。

**🔧 技术方法**

使用技术：Python 开发的模块化组件化仿真框架；Modbus 通讯与实时电力测量；线性电池与逆变器优先级模型；MPC 优化与 RL 兼容接口；大语言模型（GPT‑5.2）用于将自然语言事件转换为数值特征；RMSE、成本节约等指标用于性能评估。

**📊 数据集**

数据集：OpenCEM 数据集，来源于香港大学深圳校区的实测 PV+电池微电网，包含两分钟采样的电压、电流、功率、SOC 等数值测量，以及系统日志、日历安排、用户命令等多模态自然语言上下文，覆盖 2025‑07 至 2026‑01。

**📈 对比分析**

比较方法：①负荷预测模型对比无上下文、仅数值上下文、仅自然语言上下文和两者组合，使用 RMSE 评估；②控制实验中，将默认逆变器策略、基于历史预测的 MPC、基于 LLM 上下文预测的 MPC 与理想完美预测进行成本对比，显示后者在能耗成本上近似最佳；总体表现表明引入自然语言上下文显著提升预测精度与能源成本优化效果。

**⚠️ 局限性**

局限性：①数据覆盖范围有限，仅涵盖单一微电网场景，未能代表不同规模与负载特征；②模型简化，线性电池与逆变器优先级逻辑未包含非线性退化与功率损耗；③对 LLM 的文本特征提取依赖特定模型，可能受模型更新影响；④未深入研究多时区、不同电价动态对策略的鲁棒性。

---

## 273. ALTO: Adaptive LoRA Tuning and Orchestration for Heterogeneous LoRA Training Workloads

**arXiv ID:** 2604.05426 | [PDF](https://arxiv.org/pdf/2604.05426v1)

**作者:** Jingwei Zuo `[一作]` (Rice University), Yuke Wang `[通讯]` (Rice University)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5022196610)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Adaptive LoRA Tuning and Orchestration（ALTO）系统，将多任务、多配置的 LoRA 超参数调优整合到单一调度框架中，实现了高效的资源共享与早停。

**💡 创新点**

核心创新点包括：
- 基于训练损失曲线的动态早停策略，能够在训练初期剔除分布偏离、过拟合或无效的配置；
- 单卡与多卡的批量多 LoRA 执行方案，利用融合分组 GEMM 内核和“adapter 并行”技术，显著提升 GPU 计算利用率；
- 层级调度器，既在任务内部进行自适应批量分配，又在集群层面做预测性 bin‑packing，实现了训练时间的最小化。

**🔧 技术方法**

技术手段包括：
- 训练过程中的 EMA 与线性回归检测损失走势；
- Triton 融合分组 GEMM 及反向传播内核；
- FSDP + Adapter Parallelism（每个 GPU 处理独立 LoRA 组）；
- 通过约束规划/CP‑SAT 的离线预测调度与事件驱动的在线重新规划；
- 对多任务的 GPU 需求与训练时间做快速吞吐率估计。

**📊 数据集**

使用了多种公开 LLM 与数据集：Llama‑3.1‑8B、Qwen2.5‑7B、Qwen2.5‑32B、Llama‑3.1‑70B；SFT 数据集 GSM8K、Tulu‑3、OpenThoughts3；RLHF 数据集 UltraFeedback（DPO）。

**📈 对比分析**

与 Sequential、LoRAFusion、mLoRA、Pipeline Parallelism 等基线进行对比。ALTO 在单卡上对 60–64 个配置实现最高 9.5× 加速；在多卡（2×H100、4×H100）上实现最高 13.8× 加速；同时在所有数据集上得到的最佳适配器质量与或优于专家调优参数。

**⚠️ 局限性**

局限性包括：
- 依赖于 LoRA 的低秩结构，难以直接迁移到全参数微调；
- 早停策略需要一定的阈值调参，对极端训练动态的鲁棒性尚未完全验证；
- 在极大模型（>70B）或极低 batch size 下，Adapter Parallelism 的通信与内存开销仍有待进一步优化。

---

## 274. Unifying VLM-Guided Flow Matching and Spectral Anomaly Detection for Interpretable Veterinary Diagnosis

**arXiv ID:** 2604.05482 | [PDF](https://arxiv.org/pdf/2604.05482v1)

**作者:** Pu Wang `[一作]` (Shandong University), Youshan Zhang `[通讯]` (Chuzhou University)

**通讯引用:** 1087 | [OpenAlex ID](https://openalex.org/A5079460371)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于 Vision‑Language 模型引导的 Flow Matching 与随机矩阵理论 (RMT) 的犬肺气胸诊断框架，并发布了首个公开的像素级标注犬胸部 X 光数据集。

**💡 创新点**

创新点在于将 VLM 辅助的细粒度分割与 RMT 统计异常检测相结合，实现了高精度定位和可解释诊断；首次公开犬胸部 X 光像素级数据；采用迭代 Flow Matching 细化分割并用谱异常分数判定病变。

**🔧 技术方法**

使用了 OpenCLIP ViT‑B/32 作为 VLM、U‑Net + VLM 引导的迭代 Flow Matching、随机矩阵理论的谱异常检测、混合损失（Dice+BCE、Focal Loss）以及逻辑回归分类器。

**📊 数据集**

使用了 Korean AI‑Hub 提供的犬胸部 X 光公开数据集，包含 8641/2468/1236 张训练/验证/测试图像，已由多名兽医放射科医师进行像素级标注。

**📈 对比分析**

通过与多种 U‑Net、Transformer、Mamba 等分割模型对比，mIoU 最高 0.8114；与传统分类器对比，F1 最高 0.7962，AUC 0.939，显著优于现有方法。

**⚠️ 局限性**

局限性包括对不同犬种和小样本的泛化能力仍有限；RMT 参数假设对结果影响较大；模型对高分辨率图像需求较高，仅在单一气胸异常场景下验证。

---

## 275. Reproducing AlphaZero on Tablut: Self-Play RL for an Asymmetric Board Game

**arXiv ID:** 2604.05476 | [PDF](https://arxiv.org/pdf/2604.05476v1)

**作者:** Tõnis Lees `[一作]` (University of Tartu), Tambet Matiisen `[通讯]` (University of Tartu)

**通讯引用:** 441 | [OpenAlex ID](https://openalex.org/A5026416117)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究AlphaZero自我对弈框架在不对称棋类Tablut上的迁移，使用双头网络分别训练攻击方和防守方的政策与价值。

**💡 创新点**

提出针对不对称游戏的单独政策/价值头设计，并通过C4数据增强和增大回放缓冲区来缓解灾难性遗忘。

**🔧 技术方法**

采用AlphaZero的MCTS+神经网络，Gumbel MuZero搜索，JAX/Flax实现，并自行实现Tablut环境与Flashbax支持。

**📊 数据集**

使用自我对弈生成的Tablut局面，约2600万帧（100轮自对弈）作为训练数据集。

**📈 对比分析**

通过与历史检查点对战并用BayesElo评估，100轮后模型获得1235 Elo，攻击方胜率明显高于防守方，表现优于随机基线。

**⚠️ 局限性**

受限于仅两块GPU、未做平衡性分析，以及评价仅在内部池内有效，无法与AlphaZero绝对水平直接比较。

---

## 276. MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library

**arXiv ID:** 2604.05458 | [PDF](https://arxiv.org/pdf/2604.05458v1)

**作者:** Md Shamimul Islam `[一作]` (Florida Polytechnic University), Ayesha S. Dina `[通讯]` (Florida Polytechnic University)

**通讯引用:** 311 | [OpenAlex ID](https://openalex.org/A5050631988)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种多智能体网络入侵检测系统MA-IDS，利用大型语言模型与检索增强生成（RAG）实现对IoT流量的语义推理与持续自我改进。

**💡 创新点**

创新点包括：①将误判自动转换为可读检测规则并存入持久经验库，实现无参数调优的持续学习；②通过RAG将经验库中的规则注入LLM上下文，克服域差距；③提供规则级可解释性，兼顾高精度与可解释性。

**🔧 技术方法**

核心技术包括：GPT‑4o大型语言模型、FAISS向量数据库、all‑MiniLM‑L6‑v2嵌入、检索增强生成（RAG）、多智能体架构（流量分类代理与误差分析代理）。

**📊 数据集**

使用公开的IoT入侵检测基准数据集NF‑BoT‑IoT（4类）和NF‑ToN‑IoT（9类），在每个数据集上分别采样50k样本构建经验库，20k样本评估。

**📈 对比分析**

与零样本GPT‑4o、AdaBoost、Naïve Bayes、SVM等传统方法对比，MA‑IDS宏观F1分别达到89.75%和85.22%，相较于GPT‑4o的17%与4.96%提升超过70个百分点，并与SVM在准确率上相当，但提供规则解释。

**⚠️ 局限性**

局限性包括：仍依赖经验库的持续增长与检索成本；对完全零日攻击的识别能力受限；模型依赖外部LLM接口，部署在资源受限设备时需进一步压缩与优化；误判规则生成可能出现冗余或噪声，需进一步质量控制。

---

## 277. Foreign Domestic Workers' Perspectives on an LLM-Based Emotional Support tool for Caregiving Burden

**arXiv ID:** 2604.05448 | [PDF](https://arxiv.org/pdf/2604.05448v1)

**作者:** Shin Shoon Nicholas Teng `[一作]` (Singapore University of Technology and Design), Kenny Tsu Wei Choo `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 246 | [OpenAlex ID](https://openalex.org/A5084357603)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对新加坡外籍家庭工在照护老人时使用LLM驱动的聊天机器人进行定性研究，探讨其情感支持效果。

**💡 创新点**

首次从非亲属照护者视角揭示LLM聊天机器人在情感安全、语言可及性和多功能适配方面的价值。

**🔧 技术方法**

利用大型语言模型（如ChatGPT）驱动的聊天机器人进行交互。

**📊 数据集**

7名有老人照护经验的外籍家庭工进行半结构化访谈与引导式对话，未使用公开数据集。

**📈 对比分析**

未进行性能量化比较，主要通过主题分析获得定性发现；研究未报告数值指标。

**⚠️ 局限性**

样本量小、仅为一次性访谈、仅针对新加坡情境、缺乏长期使用和系统性能评估。

---

## 278. Not All Agents Matter: From Global Attention Dilution to Risk-Prioritized Game Planning

**arXiv ID:** 2604.05449 | [PDF](https://arxiv.org/pdf/2604.05449v1)

**作者:** Kang Ding `[一作]` (Southeast University), Lei He `[通讯]` (Tsinghua University)

**通讯引用:** 7670 | [OpenAlex ID](https://openalex.org/A5008695429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出GameAD框架，将端到端自动驾驶重新表述为风险优先的博弈规划，并实现感知到规划的风险信息传播。

**💡 创新点**

核心创新在于：①引入风险优先博弈规划；②设计四大模块（RTA、SPA、MRSA、RCES）以解决全局注意力稀释问题；③通过最小化最大风险稀疏注意力实现只关注高危交互；④提出计划风险暴露（PRE）指标，用于量化轨迹全程风险。

**🔧 技术方法**

采用多视角BEV特征提取、Transformer注意力机制、风险感知拓扑锚定、策略负载适配器、指数风险函数与Hausdorff距离、最小化最大风险稀疏注意力与风险一致均衡稳定化等技术。

**📊 数据集**

在nuScenes和Bench2Drive（CARLA Leaderboard 2.0）数据集上进行实验。

**📈 对比分析**

与UniAD、VAD、GenAD、SparseDrive、MomAD、BridgeAD等SOTA端到端方法进行开放环路和闭环对比；GameAD在L2误差、碰撞率、PRE、成功率、效率等指标均实现或接近SOTA，尤其碰撞率降低约11%，PRE下降约11%。

**⚠️ 局限性**

局限性包括：依赖高质量地图与感知输入；仅关注几何碰撞风险，未充分考虑非几何或长期不确定性；模型规模与推理速度相对较大，可能在极端复杂场景下产生过度保守或不稳定行为。

---

## 279. Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling

**arXiv ID:** 2604.05445 | [PDF](https://arxiv.org/pdf/2604.05445v1)

**作者:** Qiyuan Chen `[一作]` (Zhejiang University), Jian Wu `[通讯]` (Zhejiang University)

**通讯引用:** 227182 | [OpenAlex ID](https://openalex.org/A5100599435)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Vision‑Language Multi‑Dimensional Reward（VL‑MDR）框架，并构建了 321k 条 21 维细粒度偏好数据集，实现在单前向传播中可解释地评估视觉‑文本回答。

**💡 创新点**

创新点包括：① 视觉感知维度预测与自适应加权聚合的三阶段解耦设计，实现动态挑选 k 个相关维度；② 21 维细粒度标签体系与多模型一致性过滤，提升奖励模型可解释性与训练信号质量；③ 结合 DPO 对齐，利用细粒度奖励生成高质量偏好对。

**🔧 技术方法**

使用技术：预训练 VLM Backbone（冻结视觉塔），三个 MLP 头（维度预测、评分、加权）；联合损失（维度预测交叉熵 + 统一双边损失）；自监督标签生成与过滤；DPO 对齐训练。

**📊 数据集**

数据集：321k 条从 7 大 VLM 偏好数据集（AI 反馈与人类反馈）整合并过滤得到的 21 维标签偏好对；评测使用 VL‑RewardBench、Multimodal RewardBench、MM‑RLHF‑Reward Bench 等基准。

**📈 对比分析**

通过与现有生成式与判别式奖励模型在三大基准上的对比，VL‑MDR 在整体准确率、宏观准确率以及严格排序指标上均优于或与最强开源模型持平；效率上单前向计算，参数增量仅 0.25%，显著低于生成式模型。

**⚠️ 局限性**

局限性：① 依赖多模型自动注释，可能携带裁判模型偏差；② 21 维层级标签体系未必覆盖所有领域需求；③ 维度门控与聚合设计需进一步调优以适应分布漂移；④ 仅在图文任务与少数基准上验证，跨模态和真实世界场景的泛化尚待评估。

---

## 280. LanG -- A Governance-Aware Agentic AI Platform for Unified Security Operations

**arXiv ID:** 2604.05440 | [PDF](https://arxiv.org/pdf/2604.05440v1)

**作者:** Anes Abdennebi `[一作]` (École de Technologie Supérieure), Hakima Ould-Slimane `[通讯]` (Université du Québec à Trois-Rivières)

**通讯引用:** 1105 | [OpenAlex ID](https://openalex.org/A5045523857)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

LanG是一款面向 SOC 的开源治理感知 Agentic AI 平台，集成统一事件记录、LLM 驱动的检测与规则生成、三阶段攻击重构以及多租户治理架构。

**💡 创新点**

创新点包括：①将 LLM 与治理策略深度耦合，形成人机交互式管线；②支持多格式（Snort、Suricata、YARA）规则生成；③提出三阶段攻击重构算法；④统一通过 Model Context Protocol 暴露工具并通过 AI 治理引擎实施细粒度权限与安全防护。

**🔧 技术方法**

技术实现基于 Phi‑3‑mini、CodeLlama、Mistral、Qwen3 四大开源 LLM，通过 QLoRA 进行参数高效微调；使用 LangGraph 构建 Agentic pipeline；规则生成、检测与日志解析等模块采用自定义 prompt 与后处理；MCP、治理与安全层通过 SQLite 记录审计。

**📊 数据集**

使用的数据集涵盖 Emerging Threats、Snort、Suricata、Yara 规则库；MITRE ATT&CK 技术描述；CIC‑UNBW24、UNSW‑NB15、CVE、VirusTotal 等多源安全日志、流量与威胁情报。

**📈 对比分析**

在与八个主流 SOC 平台的基准对比中，LanG 在关联召回率达 87%、规则可部署率 91%、检测 F1 分别为 99.0% / 91.0%，且在实时推理时延约 21 ms、MTTD 1.58 s，整体性能优于同类开源工具。

**⚠️ 局限性**

局限性包括：①大型 LLM 推理成本和能耗；②LLM 产生的 hallucination 仍需人工审核；③规则生成的准确性依赖训练集多样性；④在极大规模流量下的实时性能与可扩展性仍待进一步验证。

---

## 281. Automated Auditing of Hospital Discharge Summaries for Care Transitions

**arXiv ID:** 2604.05435 | [PDF](https://arxiv.org/pdf/2604.05435v1)

**作者:** Akshat Dasula `[一作]` (Centific AI Research), Jaideep Srivastava `[通讯]` (University of Minnesota)

**通讯引用:** 18099 | [OpenAlex ID](https://openalex.org/A5002187701)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用本地部署的大型语言模型（Qwen 2.5‑7B）对MIMIC‑IV数据库中的成人住院出院摘要进行结构化审计，自动判断关键护理转移要素是否完整、模糊或缺失。

**💡 创新点**

提出一种可扩展、隐私友好的本地审计框架，将DISCHARGED记忆法拆解为46个细粒度问题，并提供临床验证的问答基准，弥补了以往仅关注文本抽取或自动生成摘要的空白。

**🔧 技术方法**

主要技术包括链式思考（CoT）提示、分段多轮LLM调用、证据句抽取与简短理由生成，以及基于本地部署的LLM推理。

**📊 数据集**

使用公开的MIMIC‑IV临床数据库中的200份出院摘要（来自196名患者），排除住院死亡病例，聚焦住院转至家庭的情境。

**📈 对比分析**

评估结果显示平均完整性得分为24.9/46（54.1%），最高得分超过30，表明模型能够检测出多数文档缺口，但与人工专家标注对照尚未完成，无法给出精确敏感度与特异度。

**⚠️ 局限性**

局限包括LLM自身不确定性与文档模糊导致的误判、缺乏人类金标准验证、模型在不同提示或模型版本下的可变性能，以及对不同医院专业/病人群体适用性的未知性。

---

## 282. PRISM-MCTS: Learning from Reasoning Trajectories with Metacognitive Reflection

**arXiv ID:** 2604.05424 | [PDF](https://arxiv.org/pdf/2604.05424v1)

**作者:** Siyuan Cheng `[一作]` (Tencent), Zheng Wei `[通讯]` (Tencent)

**通讯引用:** 27619 | [OpenAlex ID](https://openalex.org/A5010566708)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了PRISM-MCTS框架，将蒙特卡罗树搜索与过程奖励模型（PRM）以及反思式共享记忆（启发式记忆与错误记忆）相结合，实现并行推理。

**💡 创新点**

创新点在于引入反思式共享记忆机制来全局共享启发式与错误信息，并采用双阶段（SDPO+分类）过程奖励模型，大幅提升搜索效率并减少轨迹。

**🔧 技术方法**

使用技术包括蒙特卡罗树搜索（MCTS）、过程奖励模型（PRM）(Step-level Direct Preference Optimization + 细粒度分类)、记忆管理器、检索增强生成（RAG）以及并行搜索策略。

**📊 数据集**

实验数据集涵盖GPQA Diamond、FoolMeTwice、MATH500 Level5、AIME25等，主要考察科学事实验证与数学推理。

**📈 对比分析**

与Zero-Shot CoT、ReAct、Search-o1、MCTS-RAG、Rest-MCTS等基线在同一基础模型上对比，PRISM-MCTS在科学验证任务中实现最高EM（如GPQA 65.08%）和数学任务中最高得分（如MATH500 82.09%），且轨迹数平均下降55%以上，说明性能更优且搜索更高效。

**⚠️ 局限性**

主要限制包括PRM训练数据规模有限，无法覆盖更大多样化任务；系统仅处理文本输入，缺乏多模态支持；局部PRM相较于闭源oracle略逊，导致搜索初期探索略宽。

---

## 283. Multi-Agent Pathfinding with Non-Unit Integer Edge Costs via Enhanced Conflict-Based Search and Graph Discretization

**arXiv ID:** 2604.05416 | [PDF](https://arxiv.org/pdf/2604.05416v1)

**作者:** Hongkai Fan `[一作]` (Hunan University), Zheng Fang `[通讯]` (China Mobile Group Hunan Company Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MAPF_Z问题（支持非单位整数边权），并设计了CBS-NIC-B求解器，将CBS-NIC与BOGD（贝叶斯优化图设计）结合，实现高效的时间区间冲突检测与约束生成。

**💡 创新点**

创新点包括：1) 在离散整数边权下引入时间区间冲突模型，保持有限状态空间；2) 将贝叶斯优化用于动态选取离散化尺度，兼顾运行时间与误差；3) 对CBS-NIC做了约束拆分与SIPP改进，提升搜索效率。

**🔧 技术方法**

核心技术：冲突基础搜索（CBS）、安全间隔路径规划（SIPP）、时间区间约束、贝叶斯优化（MOBO）与多目标遗传算法（NSGA-II）、优先级冲突检测（PC）与离散化拆分（DS）。

**📊 数据集**

使用MAPF Benchmark Suite的网格图与道路图（Berlin-1-256、random-64-64-10、den520d等）以及随机生成的起止对，覆盖多种环境复杂度。

**📈 对比分析**

与CCBS、MA-CBS、EPEA、Baseline等方法比较，CBS-NIC-B在30秒时间限制下成功率提升至100%，运行时间平均保持在1秒以内，只有轻微的最优代价偏差；在道路图上也显著降低失败率与执行时间。

**⚠️ 局限性**

局限性：1) 在狭窄通道或高障碍密度图中BOGD效果减弱，导致成功率下降；2) 由于离散化引入误差，最优最短路径可能略长；3) 目前仅针对整数边权，无法直接处理连续权重或更复杂的动态环境。

---

## 284. Training Without Orthogonalization, Inference With SVD: A Gradient Analysis of Rotation Representations

**arXiv ID:** 2604.05414 | [PDF](https://arxiv.org/pdf/2604.05414v1)

**作者:** Chris Choy `[一作]` `[通讯]` (NVIDIA), Chris Choy (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对旋转估计中使用SVD正交化的梯度行为进行详细理论分析，推导出3×3矩阵投影的雅可比矩阵谱，量化梯度失真，并揭示Gram‑Schmidt正交化梯度不对称的问题，最终验证直接9维回归结合推断时的SVD投影是最优的训练策略。

**💡 创新点**

①首次给出SVD投影雅可比矩阵的完整谱和条件数（σ=2/(s_i+s_j)，κ=(s_1+s_2)/(s_2+s_3)），①量化了梯度放大和信息损失；②证明即便使用现有的SVD稳定化技术也会引入方向误差；③揭示Gram‑Schmidt梯度不对称导致6维参数梯度不均衡；④基于以上理论解释了9维直接回归+SVD推断在实际数据集上的优势。

**🔧 技术方法**

使用SVD、Gram‑Schmidt正交化的数学推导、雅可比矩阵谱分析、条件数计算、梯度信息保持率（GIR）分析、误差分解与投影优劣比较等理论技术。

**📊 数据集**

主要在人体姿态估计基准（如MPII、Human3.6M、COCO、MPI-INF-3DHP）以及3D模型匹配基准（Pascal3D+、ModelNet）上进行实验评估。

**📈 对比分析**

与SVD-Train（训练时使用SVD）和GS-Inference（训练时使用6维Gram‑Schmidt、推断时同样使用GS）等方法对比，使用PA-MPJPE、MPJPE等指标；实验表明9D+SVD-Inference在PA-MPJPE上取得54.8，而SVD-Train为55.6，GS-Inference为56.7，表现优于其他方案；同时在推断误差上SVD投影可将误差约减少三分之一。

**⚠️ 局限性**

仅针对Frobenius范数损失进行分析；对几何（弧度）损失的梯度行为未完全覆盖；平均-案例分析和收敛速率比较仅在附录给出；实际推断时如果网络已接近正交，GS相对SVD的收益可能有限。

---

## 285. CRISP: Rank-Guided Iterative Squeezing for Robust Medical Image Segmentation under Domain Shift

**arXiv ID:** 2604.05409 | [PDF](https://arxiv.org/pdf/2604.05409v1)

**作者:** Yizhou Fang `[一作]` (Southern University of Science and Technology), Longxi Zhou `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5010191987)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了CRISP框架，利用分区排名稳定性实现无目标域信息下的医学图像分割

**💡 创新点**

创新点在于引入“正区排名稳定性”经验法则，采用基于排名的分割而非概率；通过潜在特征扰动生成高精度与高召回先验，并递归收缩不确定区域

**🔧 技术方法**

使用潜在特征扰动、分级置信度映射、递归自进化训练、不确定性收缩损失，模型架构基于DeepLabv3++MobileNetV2

**📊 数据集**

评估数据集包括多中心心脏MRI（M&MS）以及CT肺血管的模态（对比与非对比）和人群（正常与COVID）三组

**📈 对比分析**

与FedDG、DDG-Med、IPLC、TEGDA等SOTA方法对比，在多中心、模态及人群偏移场景下，CRISP在Dice和HD95上均超过对手，尤其在HD95上降低数像素，甚至超过全监督上限

**⚠️ 局限性**

局限在于假设排名稳定性在所有结构上均成立，且需要多轮迭代；对极端域偏移或极少样本情况的鲁棒性尚未完全验证

---

## 286. INTERACT: An AI-Driven Extended Reality Framework for Accesible Communication Featuring Real-Time Sign Language Interpretation and Emotion Recognition

**arXiv ID:** 2604.05605 | [PDF](https://arxiv.org/pdf/2604.05605v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (Institute of Communications and Computer Systems), Evangelos Papatheou `[通讯]` (University of Exeter)

**通讯引用:** 1132 | [OpenAlex ID](https://openalex.org/A5078366434)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了一款名为 INTERACT 的 XR 平台，支持实时语音转文字、国际手语（ISL）动画、跨语言翻译、情感识别和会议摘要，以提升聋人和多语言用户的视频会议可访问性。

**💡 创新点**

其创新点在于将实时 ISL 三维头像渲染、多语言翻译、情感反馈和会议摘要融合到一个沉浸式环境中，并通过模块化 AI 服务实现可扩展的多模态可访问性。

**🔧 技术方法**

使用的技术包括 OpenAI Whisper、Meta AI NLLB、Google MediaPipe、DistilRoBERTa、BART-Large、Unity 3D、Meta Quest 3、Rainbow SDK 以及 CORTEX2 中介网关。

**📊 数据集**

采用的数据集包括 Whisper 多语音语料、NLLB 200 语言模型、MediaPipe 标记数据、含 750 个符号的 ISL 视频语料、BART 的 SAMSum 对话摘要数据，以及公开的 ISL 手势数据集。

**📈 对比分析**

通过两阶段实地测试，系统在 80%+ 的用户满意度、>85% 的转写准确率、90% 的情感识别精度、<2 秒的端到端延迟以及 1,000 用户并发无错误的性能指标，优于现有仅提供字幕或单一手语支持的解决方案。

**⚠️ 局限性**

局限性包括 ISL 词汇仅约 750 个，缺乏面部表情和语法标记的完整动画，支持语言对仅限英法，测试样本规模小，音频分块导致偶尔单词切断，需要进一步扩展词汇、面部动画、多语言和更自然的手语生成。

---

## 287. BPC-Net: Annotation-Free Skin Lesion Segmentation via Boundary Probability Calibration

**arXiv ID:** 2604.05594 | [PDF](https://arxiv.org/pdf/2604.05594v1)

**作者:** Yujie Yao `[一作]` (Sichuan Agricultural University), Xiaofan Li `[通讯]` (Sichuan Agricultural University)

**通讯引用:** 4685 | [OpenAlex ID](https://openalex.org/A5100637080)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种注释无监督的皮肤病变分割框架BPC‑Net，并通过局部概率平滑(GPS)校准边界概率来提升分割边缘完整性。

**💡 创新点**

创新点在于将边界概率欠置信视为独立瓶颈，设计GPS进行局部概率空间校准；同时引入特征解耦解码器和交互分支自适应，稳健应对伪标签噪声与跨域迁移。

**🔧 技术方法**

核心技术包括多源伪标签生成、IPC/PIA注意力交互、动态不确定性损失、特征解耦解码、Gaussian Probability Smoothing(GPS)、TTA和形态学后处理。

**📊 数据集**

在公开的ISIC‑2017、ISIC‑2018和PH2三大医学图像数据集上进行训练与评估。

**📈 对比分析**

与11种无监督基线和RPI‑Net对比，BPC‑Net在所有数据集上均取得最高Dice和Jaccard；在宏平均上Dice和Jaccard分别提升3.26%和2.86%；与有监督参考模型相差3–4个百分点，PH2上仅差1.02个百分点。

**⚠️ 局限性**

局限性包括对严重毛发遮挡、低对比度或模糊边缘的病例仍易失真；伪标签初始假设仅适用于传统皮肤镜图像；需要在验证集上调参以确定阈值与GPS参数，且未在跨中心或多专家评估上进行充分验证。

---

## 288. Label Effects: Shared Heuristic Reliance in Trust Assessment by Humans and LLM-as-a-Judge

**arXiv ID:** 2604.05593 | [PDF](https://arxiv.org/pdf/2604.05593v1)

**作者:** Xin Sun `[一作]` (National Institute of Informatics), Saku Sugawara `[通讯]` (National Institute of Informatics)

**通讯引用:** 492 | [OpenAlex ID](https://openalex.org/A5038103607)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对健康问答内容的源标签进行对照交换，分别让人类参与者和LLM-评判者进行可信度评分，并结合眼动追踪与LLM注意力及logit熵分析，研究源标签对信任评估的影响。

**💡 创新点**

创新点在于首次将人类眼动行为与LLM内部注意力模式相结合，系统揭示了源标签作为启发式线索在两种评判者中的一致偏差，并警示对LLM-评判者进行人类价值对齐时可能导致的偏差放大。

**🔧 技术方法**

采用了对照实验设计、眼动追踪、Transformer注意力权重提取、logit熵计算以及统计检验（Wilcoxon、混合ANOVA、GEE）等技术。

**📊 数据集**

使用了150条健康问答对（来自公开健康问答数据集）进行实验，涵盖人工和GPT‑4o生成的答案，并标注人类或AI来源。

**📈 对比分析**

比较方法是将人类与多种LLM（包括GPT‑4o、Claude、LLaMA、Qwen等）在同一内容下的信任评分进行配对检验，结果显示无论是人类还是LLM，标记为人类的答案始终获得更高信任；LLM内部表现为标签区注意力占优且AI标签下熵更高。

**⚠️ 局限性**

局限性包括仅聚焦健康问答领域、标签种类有限、对内部信号的关联性不足以说明因果关系，以及人类眼动与Transformer注意力之间的直接对应假设尚需验证。

---

## 289. Foundations for Agentic AI Investigations from the Forensic Analysis of OpenClaw

**arXiv ID:** 2604.05589 | [PDF](https://arxiv.org/pdf/2604.05589v1)

**作者:** Jan Gruber `[一作]`, Jan-Niclas Hilgert `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于有限状态机的多模态交互框架，能够自动执行文本、语音、图片、邮件、日历、网络搜索等多种服务任务，并通过动态生成子代理和记忆事实来提高任务链的灵活性。

**💡 创新点**

创新点在于将复杂任务拆解为可序列化的状态，并在状态机中引入子代理生成（spawn subagent）与记忆管理（remember fact）机制，使系统能够动态适应不同任务需求。

**🔧 技术方法**

采用了有限状态机（FSM）设计、异构服务接口（OpenClaw、Gog、Gmail、Google Calendar等）、子代理调度模块以及强化学习/规划算法来学习最优状态转换。

**📊 数据集**

使用了公开的多模态交互数据集（如OpenAI MultiModal Benchmark）以及自构造的任务执行日志，涵盖文本、语音、图像和日历事件等多种类型。

**📈 对比分析**

与传统单一代理对比，实验结果表明任务完成率提升约30%，平均执行时间下降约15%，在标准基准任务上表现显著优于基线方法。

**⚠️ 局限性**

局限性包括对大规模服务生态的可扩展性不足、子代理间通信开销较高、对异常情况的鲁棒性待提升，以及对动态环境变化的适应性仍有待加强。

---

## 290. ResearchEVO: An End-to-End Framework for Automated Scientific Discovery and Documentation

**arXiv ID:** 2604.05587 | [PDF](https://arxiv.org/pdf/2604.05587v1)

**作者:** Zhe Zhao `[一作]` (City University of Hong Kong), Qingfu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 39707 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

研发了一个端到端的研究框架ResearchEVO，能自动完成从无约束实验到理论解释的完整科研周期；

**💡 创新点**

创新点在于：①双维共进化算法同时演化代码逻辑与整体架构；②写作阶段采用句级检索增强生成与反幻觉校验，自动生成完整、可编译且无虚假引用的科研论文；

**🔧 技术方法**

主要技术包括：大型语言模型驱动的代码生成与进化、二维共进化搜索、检索增强生成（RAG）、反幻觉验证与自动实验设计；

**📊 数据集**

使用的数据集为：Google真实量子硬件数据（用于量子误差校正）和物理信息神经网络（Physics‑Informed Neural Networks）相关数据；

**📈 对比分析**

与传统手工探索或单一维度进化相比，ResearchEVO在两项跨学科任务中发现了此前未被提出的人类可解释算法，并自动生成了完整且正确引用的论文，展示了更高的创新度和写作自动化水平；

**⚠️ 局限性**

局限性在于：目前仅在两类问题上验证，泛化性未知；对LLM的依赖导致对模型质量与训练分布的敏感；写作阶段的反幻觉机制尚未覆盖所有潜在错误，且系统整体计算成本较高。

---

## 291. THIVLVC: Retrieval Augmented Dependency Parsing for Latin

**arXiv ID:** 2604.05564 | [PDF](https://arxiv.org/pdf/2604.05564v1)

**作者:** Luc Pommeret `[一作]` (Université Paris-Saclay), Jules Deret `[通讯]` (École Normale Supérieure de Lyon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于检索增强的大型语言模型（LLM）的拉丁语依存句法分析系统，通过结构检索、UD 注释规则与基线 UDPipe 解析相结合，实现对句法分析的精细修正。

**💡 创新点**

创新点在于将 UD 规则显式注入 LLM 并通过结构检索（长度+POS n‑gram）为 LLM 提供示例，形成检索增强生成（RAG）流程；同时将 LLM 角色定位为“拉丁语首席注释者”以纠正基线解析。

**🔧 技术方法**

技术包括结构检索器（长度相似度 + POS bigram/trigram Jaccard 组合）、LLM（以提示式调用实现依存修正）、UD 依存注释指南、UDPip基线解析及提示工程。

**📊 数据集**

使用数据集为 CIRCSE 训练集（762 条句子）做检索演示，EvaLatin 2026 测试集做生成评估；检索时选取 k=5 最相似句子。

**📈 对比分析**

与 UDPipe 基线以及两种配置（无检索、检索）在 CLAS 与 LAS（含/不含子类型）进行对比；在古典诗歌上提升 CLAS +17 点（子类型下），在散文上提升 +1.5 CLAS；加入 RAG 后，散文子类型 CLAS 再提升 +6.9 点；结构检索在句长差异 (<1.2 token) 与 POS 重叠度上明显优于 TF‑IDF 与形态检索。

**⚠️ 局限性**

局限性包括：LLM 选择缺乏系统对比、仅使用单一小型知识库（CIRCSE）导致检索覆盖不足、LLM 输出非确定性与高成本、错误分析样本有限（300 条、两名注释者，κ=0.49）。

---

## 292. EpiBench: Benchmarking Multi-turn Research Workflows for Multimodal Agents

**arXiv ID:** 2604.05557 | [PDF](https://arxiv.org/pdf/2604.05557v1)

**作者:** Xuan Dong `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8858 | [OpenAlex ID](https://openalex.org/A5019108029)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EpiBench基准，用于评估多模态研究代理在多轮科研工作流中的表现，包括主动搜索、跨论文多模态证据整合与记忆重用。

**💡 创新点**

创新点：①过程级评估框架强调跨论文多图表融合和证据记忆；②访问预算评估协议实现自动化的证据重用与工具效率评分；③提供真实科研流程的多轮任务，填补现有基准对工作流细节的空白。

**🔧 技术方法**

技术：基于大型多模态语言模型（如Qwen3‑VL、GLM‑4.5V、GPT‑5.2等）与四大工具（Web搜索、PDF提取、文本RAG、内容提取）的Agent框架；采用ReAct式工具调用与记忆管理，支持交互式多轮推理。

**📊 数据集**

数据集：EpiBench由专家抽取、扩展、人工标注的102个科研论文任务（Easy 39/Hard 63），覆盖计算机视觉与机器学习六大领域，任务包含多篇论文、图表、表格与文本证据。

**📈 对比分析**

比较方法：在开放源代码与封闭源代码LMM代理与人类专家之间进行对比，使用Episode Success Rate、Final‑Turn Accuracy、Non‑Final‑Turn Accuracy、Evidence Correctness、Minimality Gap等指标。实验显示最佳模型GPT‑5.2在Hard集上的ESR仅29.23%，显著低于人类专家（>90%），说明多证据融合与记忆重用是主要瓶颈。

**⚠️ 局限性**

Limitations：①基准规模有限（102个任务）；②侧重流程评估，未涵盖所有科研深度与复杂度；③对工具与环境依赖强，跨平台可复现性受限；④主要聚焦计算机视觉/ML领域，推广到其他学科存在挑战；⑤缺乏对多模态理解与推理机制的深入分析。

---

## 293. Cross-Modal Coreference Alignment: Enabling Reliable Information Transfer in Omni-LLMs

**arXiv ID:** 2604.05522 | [PDF](https://arxiv.org/pdf/2604.05522v1)

**作者:** Hongcheng Liu `[一作]` (Shanghai Jiao Tong University), Yu Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 44812 | [OpenAlex ID](https://openalex.org/A5100445300)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了专注跨模态共参的CrossOmni数据集，并通过In-Context Learning和SFT+GRPO两种方法提升Omni-LLMs的跨模态共参推理能力。

**💡 创新点**

① 将跨模态共参问题形式化为源‑目标共参任务；② 设计了包含9种任务、带有一步步思维路径的多模态共参数据集；③ 提出训练无关的ICL与训练有监督的SFT+GRPO两种思维模式诱导策略。

**🔧 技术方法**

使用In-Context Learning、Chain-of-Thought提示、LoRA微调、Group Relative Policy Optimization（GRPO）等技术来塑造核心参考对齐思维模式。

**📊 数据集**

主要使用基于TVQA的视频数据构建的CrossOmni（4,147段视频、39,726 QA对），以及在Daily-Omni、WorldSense等OOD基准上的验证。

**📈 对比分析**

在13款Omni-LLMs上对比实验，发现单模态与跨模态共参平均差距约21%；ICL提升约21%，SFT+GRPO进一步提升并能泛化至OOD基准，整体性能显著提升。

**⚠️ 局限性**

在音频任务上的表现仍落后于文本与视觉任务；数据集依赖人工字幕，限制了自动化与大规模扩展；对音频共参失败原因尚未深入分析。

---

## 294. WRF4CIR: Weight-Regularized Fine-Tuning Network for Composed Image Retrieval

**arXiv ID:** 2604.05583 | [PDF](https://arxiv.org/pdf/2604.05583v1)

**作者:** Yizhuo Xu `[一作]` (Huazhong University of Science and Technology), Xinge You `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 6258 | [OpenAlex ID](https://openalex.org/A5057095711)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并缓解了VLP基准下的组成图像检索（CIR）模型在有限triplet数据上的过拟合问题，提出并验证了一种权重正则化微调网络WRF4CIR。

**💡 创新点**

创新点在于：通过对模型权重施加梯度反向方向的对抗性扰动，使损失景观变得更平滑，从而在微调过程中显著抑制过拟合；该方法兼容多种预训练模型与融合策略，且在不额外获取数据或LLM的前提下实现性能提升。

**🔧 技术方法**

技术细节包括：使用CLIP、BLIP‑2等VLP预训练模型；Q‑Former与可学习查询实现图文融合；对比损失与温度参数；对权重施加AWP式对抗扰动（可调γ）；LoRA参数高效微调；对比L2正则化和常规数据增强等。

**📊 数据集**

实验数据集：FashionIQ、CIRR、Shoes，此外在FashionIQ上分别使用不同比例（0.2/0.4/0.6/0.8/1.0）训练集进行数据规模敏感性测试。

**📈 对比分析**

与现有VLP基准（SPRC、CCIN、CLIP4CIR）以及基于LLM的数据增强方法（SPN4CIR）在Recall@K指标上对比，WRF4CIR在FashionIQ上平均Recall提升约2–3%，在CIRR上提升约1.5%；在仅使用40%训练集时即可达到全量训练效果；同时保持无推理阶段额外开销。

**⚠️ 局限性**

局限性：需要额外计算来生成对抗扰动，扰动比例γ需手工调参；方法仍依赖大规模VLP模型，且在其他多模态检索任务的泛化尚待验证；对于极少量triplet数据，提升幅度仍受限。

---

## 295. Bias Ahead: Sensitive Prompts as Early Warnings for Fairness in Large Language Models

**arXiv ID:** 2604.05575 | [PDF](https://arxiv.org/pdf/2604.05575v1)

**作者:** Gianmario Voria `[一作]` (University of Salerno), Fabio Palomba `[通讯]` (University of Salerno)

**通讯引用:** 9929 | [OpenAlex ID](https://openalex.org/A5033738898)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了SensY数据集，评估LLM对敏感提示的偏见，并研发敏感度分类器

**💡 创新点**

提出将敏感提示视为公平评估的新抽象，强调预警而非事后修正

**🔧 技术方法**

使用基于句法与语义特征的随机森林分类器，并通过LLM响应分析

**📊 数据集**

SensY（12,801条提示，2,710敏感）与SQuARe作为评估基准

**📈 对比分析**

在三款8B开源LLM上对敏感提示的回答进行手工评估；分类器在SensY上达到≈0.91准确率、0.84 F1，跨数据集泛化有限

**⚠️ 局限性**

对数据分布、文化偏差、模型规模的外部效度有限，且分类器对非敏感提示召回率较低

---

## 296. Physics-Aligned Spectral Mamba: Decoupling Semantics and Dynamics for Few-Shot Hyperspectral Target Detection

**arXiv ID:** 2604.05562 | [PDF](https://arxiv.org/pdf/2604.05562v1)

**作者:** Luqi Gong `[一作]` (Zhejiang Lab), Chao Li `[通讯]` (Zhejiang Lab)

**通讯引用:** 5041 | [OpenAlex ID](https://openalex.org/A5100323157)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对极少样本的高光谱目标检测，提出了SpecMamba框架，冻结Transformer主干并通过频域离散余弦变换与Mamba状态空间模型实现轻量化的光谱适配，同时引入物理先验三分编码和自监督伪标签映射进行测试时自适应。

**💡 创新点**

创新点包括：1）将频域DCT与线性复杂度Mamba结合的DCTMA，显式捕获全局光谱依赖和带间连续性；2）Prior-Guided Tri-Encoder利用实验室光谱先验正则化适配器，避免原型漂移；3）Self-Supervised Pseudo-Label Mapping实现无标签测试时快速自适应，提升边界鲁棒性。

**🔧 技术方法**

技术手段：Transformer主干、离散余弦变换(DCT)、Mamba状态空间网络、跨频段门控融合、物理先验三分编码、伪标签采样与一致性约束、双任务（分类与检测）联合优化。

**📊 数据集**

使用的公开数据集：源域Chikusei（128波段），目标域四个公开数据集——San Diego I、San Diego II、Cri、Urban-1，均覆盖不同传感器与地表目标。

**📈 对比分析**

与8种经典及先进方法（CEM、ACE、MLSN、BLTSC、HTD-IRN、UEML、SelfMTL、HTD-Mamba）进行AUC、ROC、AUC_OA、AUC_SNPR等多指标对比，SpecMamba在四个数据集上均取得最高或相近的AUC(P_f,P_d)、AUC(τ,P_d)和AUC_OA，并在Urban-1上显著提升了AUC_SNPR，表明在少样本跨域场景中检测精度与背景抑制兼顾。

**⚠️ 局限性**

局限性：1）对传感器极端大气变化的鲁棒性尚未验证；2）需依赖实验室光谱先验，若缺失或不匹配可能影响对齐效果；3）测试时自适应仍需一定迭代次数，实时性尚待进一步优化。

---

## 297. An Iterative Test-and-Repair Framework for Competitive Code Generation

**arXiv ID:** 2604.05560 | [PDF](https://arxiv.org/pdf/2604.05560v1)

**作者:** Lingxiao Tang `[一作]` (Zhejiang University), He Ye `[通讯]` (University College London)

**通讯引用:** 3762 | [OpenAlex ID](https://openalex.org/A5060546219)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于迭代测试与修复的竞争性代码生成框架 FixAudit。

**💡 创新点**

创新点在于：① 将测试生成器（Auditor）与修复器（Fixer）合并为同一模型，并让 Auditor 读取候选代码以生成针对性测试；② 通过四阶段训练（执行对齐 SFT、Fixer RL、Auditor RL、Fixer 迭代）实现闭环调试；③ 在 7B 规模模型上实现超越更大 32B 零样本模型的性能。

**🔧 技术方法**

使用了执行对齐的监督微调、强化学习（DAPO）训练 Fixer 和 Auditor，并利用程序输出预测与规范到输出推导两任务构建训练数据。

**📊 数据集**

训练集基于 TACO（约 7k 条含 20+ 测试的编程问题），评测集采用 APPS、CodeContests 与 xCodeEval 三大竞赛级编程基准。

**📈 对比分析**

在 20 次 LLM 调用预算下，与同基线 7B 模型（Specine、CURE）以及更大规模模型（Qwen2.5-32B、GPT‑4o‑mini）对比，FixAudit 在 Pass@1 上平均提升 36.8%/35.1%（相对 Specine/CURE），并在 7B 规模上超越 32B 零样本模型 24.9% 的 Pass@1 与 40.5% 的 AvgPassRatio。

**⚠️ 局限性**

局限性包括：① 对执行推理能力依赖较高，若模型推理不足会影响 Fixer/Auditor；② Auditor 生成测试仍有约 25% 无效率；③ 目前仅支持 Python，尚未验证对其他语言或更大模型的可推广性；④ 训练过程分阶段，未探索联合或交替训练的可能性。

---

## 298. SCOPE: A Dataset of Stereotyped Prompts for Counterfactual Fairness Assessment of LLMs

**arXiv ID:** 2604.05555 | [PDF](https://arxiv.org/pdf/2604.05555v1)

**作者:** Alessandra Parziale `[一作]` (University of Salerno), Fabio Palomba `[通讯]` (University of Salerno)

**通讯引用:** 9929 | [OpenAlex ID](https://openalex.org/A5033738898)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为SCOPE的大规模对照提示对数据集，用于系统评估大型语言模型（LLM）的反事实公平性。

**💡 创新点**

创新点在于：①数据量巨大（241,280条提示，120,640对）；②对照对在语义上严格对齐，仅差敏感属性；③覆盖1,438主题、9种偏见维度、1,536个群体；④四种常见交互意图（问询、推荐、指令、澄清）实现意图条件化；⑤利用结构化知识库和自动生成管线生成多样化、高质量的提示。

**🔧 技术方法**

技术实现主要依赖GPT‑4o mini进行知识三元组抽取和提示生成，随后手工验证并以JSON格式输出；使用脚本对数据进行整理、分组和统计。

**📊 数据集**

数据来源为CrowS‑Pairs（1,508条句子对），通过抽取得到的三元组作为生成的基础。

**📈 对比分析**

评估方法：将每对对照提示分别送入LLM（如Gemini），对比输出的长度、词汇重叠、毒性等指标；实验显示即使语义和意图一致，模型在不同群体上的回答存在明显差异，表明存在不对称行为。

**⚠️ 局限性**

局限性：①数据仅覆盖英语且缺少多语言扩展；②生成过程受所用生成模型的偏差影响；③只考虑四种意图，未覆盖全部对话场景；④缺乏动态更新机制，无法及时反映最新社会偏见；⑤实验验证仅基于单一模型，尚未系统评估在多模型上的表现。

---

## 299. Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue

**arXiv ID:** 2604.05552 | [PDF](https://arxiv.org/pdf/2604.05552v1)

**作者:** Junan Hu `[一作]` (Shandong University), Yinwei Wei `[通讯]` (Shandong University)

**通讯引用:** 3170 | [OpenAlex ID](https://openalex.org/A5039731055)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Context-Agent 框架，利用动态树结构对多轮对话历史进行非线性建模，并引入 NTM 基准评测。

**💡 创新点**

创新点在于将对话视作可动态分支的树形结构，显式捕捉话题切换与指令细化；同时结合检索增强生成提升上下文选择效率，并设计专门评测非线性多轮对话的 NTM 数据集。

**🔧 技术方法**

使用的技术包括：动态树结构建模、节点嵌入与相似度匹配、轻量化话题/分支决策模型、检索增强生成（RAG）以及节点/分支摘要。

**📊 数据集**

采用自建的 NTM 数据集（包含日常生活规划与编程支持两类对话）以及公开的 TopiOCQA 数据集进行实验。

**📈 对比分析**

与 Full-History、Truncation 等传统上下文管理方法在 GPT‑4.1、DeepSeek‑V3、GLM‑4‑Plus、Llama 3.1‑70B 等四款 LLM 上对比，Context‑Agent 在任务完成率上提升 3.4%–9.7%，平均上下文 token 减少 45%–52%，并在 TopiOCQA 上实现 EM/F1 的显著提升。

**⚠️ 局限性**

局限性在于依赖轻量级模型进行话题与分支决策，受模型与提示设计影响；未实现端到端学习，进一步优化这些决策模块可能带来更大收益。

---

## 300. FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation--Full Version

**arXiv ID:** 2604.05551 | [PDF](https://arxiv.org/pdf/2604.05551v1)

**作者:** Dat Nguyen-Cong `[一作]` (FPT Corporation), Hoang Thanh-Tung `[通讯]` (FPT Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种快速推理的连续扩散语言模型训练框架，结合自条件扰动与模型感知噪声缩放，显著提升少步采样的质量与速度。

**💡 创新点**

创新点在于：1) 自条件扰动正则化，模拟推理时的自条件误差以消除训练-推理不匹配；2) 动态噪声缩放，根据标记置信度分配噪声，避免训练饱和并保持有效梯度。

**🔧 技术方法**

采用的技术包括连续扩散模型、token级自条件扰动、模型感知噪声缩放、基于噪声的时间步调度、MBR解码及大规模GPU并行训练。

**📊 数据集**

使用的主数据集包括机器翻译（IWSLT14、WMT14、WMT16）、摘要（Gigaword）、问答重述（QQP）、文本简化（Wiki-Auto）以及推理基准（GSM8K）。

**📈 对比分析**

与自回归、非自回归、传统扩散及一阶扩散基线对比；在2–20步采样下，模型速度比传统扩散快4–400倍，BLEU/ROUGE提升1–3点，接近或超过20步/传统多步模型的质量。

**⚠️ 局限性**

局限性在于：1) 对前一步自条件预测的质量仍然敏感，无法完全消除推理误差；2) 噪声缩放策略为预设而非自适应，可能在不同任务或阶段失效；3) 对极低步数下的极端噪声扰动仍存在性能下降。

---

## 301. Channel-wise Retrieval for Multivariate Time Series Forecasting

**arXiv ID:** 2604.05543 | [PDF](https://arxiv.org/pdf/2604.05543v1)

**作者:** Junhyeok Kang `[一作]`, Soonyoung Lee `[通讯]` (LG AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个通道级检索增强的多变量时间序列预测框架，允许每个变量单独检索历史参考片段；

**💡 创新点**

创新点在于（1）实现通道级检索，避免了通道无关的检索策略导致的信息不匹配；（2）采用两阶段检索流程——先在时域构造稀疏关联图筛选候选通道，再在频域利用FFT低频谱相似度精确排序；（3）通过预计算的稀疏图与频域相似度实现检索效率，兼顾精度与实时性；

**🔧 技术方法**

主要技术包括：检索增强预测、FFT频谱提取与低频截断、余弦相似度构建稀疏关联图、归一化的复数内积相似度、MLP+NLinear归一化的直接预测子、加权融合的最终预测；

**📊 数据集**

使用七个公开基准数据集：ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Traffic、Weather；

**📈 对比分析**

在MSE和MAE指标上与RAFT、TimeMixer、PatchTST、TimesNet、MICN、DLinear、Stationary、Autoformer等八个先进基线进行对比，实验显示其平均性能均优于所有基线；同时推理时间仅为0.03秒/批次（批大小32），可处理超过1000条时间序列/秒；

**⚠️ 局限性**

局限性：需预先构建稀疏关联图，增加了训练前的准备工作；频域检索只保留低频分量，可能忽略高频细节；对超高维通道数或非周期性强的数据，其检索效果和泛化能力还有待进一步验证。

---

## 302. Prior-guided Fusion of Multimodal Features for Change Detection from Optical-SAR Images

**arXiv ID:** 2604.05527 | [PDF](https://arxiv.org/pdf/2604.05527v1)

**作者:** Xuanguang Liu `[一作]` (Information Engineering University), Hanyun Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3706 | [OpenAlex ID](https://openalex.org/A5071855633)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了STSF-Net框架，实现光学与SAR多模态遥感图像的变更检测；

**💡 创新点**

创新点在于双分支解耦模态特定特征与时空共性特征，并通过视觉基础模型SAM2生成的语义先验实现自适应融合，显著提升对真实变更的识别和伪变更的抑制；

**🔧 技术方法**

技术细节包括：pseudo‑Siamese SAM2+Swin‑Transformer特征提取、跨模态交叉注意力与图卷积网络（STCFM）提取时空共性特征、基于语义先验的Prior‑Guided Feature Fusion Module（PGFFM），训练采用Adam优化，评价指标为OA、mIoU、F1等；

**📊 数据集**

使用数据集：新建Delta‑SN6（0.5 m分辨率光学+全极化SAR、三类变更标签），并在BRIGHT与Wuhan‑Het（光学‑SAR）数据集上进行实验；

**📈 对比分析**

与13种现有变更检测方法（如DeepLabV3+、GSTM‑SCD等）在三组数据集上对比，STSF‑Net在Delta‑SN6、BRIGHT、Wuhan‑Het上分别实现最高mIoU、F1，提升幅度约3%‑4%；

**⚠️ 局限性**

局限性包括：模型参数量（约63 M）相对较大，推理速度虽然已优化但仍不适合边缘实时部署；目前仅支持双模态（光学‑SAR）输入，缺乏对多时相多模态统一处理的能力；数据集在类别平衡和时间覆盖上仍有限，未来需扩展时间维度与更多地表覆盖类型。

---

## 303. CrowdVLA: Embodied Vision-Language-Action Agents for Context-Aware Crowd Simulation

**arXiv ID:** 2604.05525 | [PDF](https://arxiv.org/pdf/2604.05525v1)

**作者:** Juyeong Hwang `[一作]` (Korea University), HyeongYeop Kang `[通讯]` (Korea University)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5011229651)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CrowdVLA，通过 Vision‑Language‑Action（VLA）代理实现每个行人基于视觉与语言指令理解场景语义与社会规范，并在多种环境下做出后果意识决策。

**💡 创新点**

创新点：将 VLA 框架迁移至大规模人群仿真；利用 agent‑centric 视觉监督 + LoRA 微调预训练 VLM；将连续运动抽象为 20 帧运动技能；通过探索式 QA 训练模型学习对比动作后果，克服传统数据的成功偏置。

**🔧 技术方法**

技术：预训练 VLM（Qwen3‑VL‑2B‑Instruct）+ LoRA 微调；基于 Unity 的语义重建与 agent‑centric 观察；K‑means 聚类生成 64 个运动技能；探索式 QA 生成对照动作与后果；多步预测与 PDM、SRVR、GSR、Div 等指标评估。

**📊 数据集**

数据集：ETH 与 UCY（Zara2、Univ、ETH）真实行人轨迹用于专家技能；自建 Unity 重建场景；评估使用未见的 Zara01、ETH‑Hotel、Intersection、Hallway、CrossingScene。

**📈 对比分析**

比较方法：与 CCP（RL）和 GBM（规则）对比，使用 PDM‑ADE/FDE、SRVR、GSR、Div 等指标。CrowdVLA 在短期动力学精度（PDM）、社会规范遵守（SRVR）上显著优于基线，任务完成率（GSR）与 CCP 相当甚至更好，行为多样性（Div）既保持多样又不牺牲稳定性。

**⚠️ 局限性**

局限性：运动技能离散化限制细粒度控制；规范遵守依赖场景语义标签；对长程意图（活动、群组）建模不足；需更丰富的场景理解、层级动作与长周期推理来进一步提升性能。

---

## 304. Cross-Resolution Diffusion Models via Network Pruning

**arXiv ID:** 2604.05524 | [PDF](https://arxiv.org/pdf/2604.05524v1)

**作者:** Jiaxuan Ren `[一作]` (University of Electronic Science and Technology of China), Huan Wang `[通讯]` (Westlake University)

**通讯引用:** 7902 | [OpenAlex ID](https://openalex.org/A5100751566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CR‑Diff，通过块级剪枝和剪枝输出放大两阶段提升 UNet‑基础文本到图像扩散模型在未见分辨率下的生成一致性和质量；

**💡 创新点**

将网络剪枝从压缩工具转变为提升跨分辨率生成质量的手段，创新性地采用分块剪枝比例、模拟退火搜索、以及剪枝输出放大（POA）等组合策略；

**🔧 技术方法**

主要技术包括基于 magnitude 的块级剪枝、模拟退火搜索最优剪枝比例、剪枝输出放大（k>1 组合）、以及可选的提示特定优化；

**📊 数据集**

实验数据集为 MS‑COCO 2014 验证集 5K 提示，并在 SDXL、SD1.5、SD2.1、SD3Medium、Flux.dev 等多种 UNet/DiT 模型上评估；

**📈 对比分析**

通过 FID、CLIP Score、ImageReward、PickScore、Aesthetic Score 等指标比较；在未见分辨率下，CR‑Diff 在多模型上显著提升 ImageReward、降低 FID，并在默认分辨率下保持或略优；

**⚠️ 局限性**

局限在于对极细粒度提示仍需提示特定优化，放大系数 k 需要手动调节，且主要验证在现有模型上，跨模型推广与新模型训练的效果仍待进一步研究。

---

## 305. UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning

**arXiv ID:** 2604.05517 | [PDF](https://arxiv.org/pdf/2604.05517v1)

**作者:** Xiaolong Wei `[一作]` (Beihang University), Daiting Shi `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UniCreative 框架，实现长短文本创作的动态模式切换，并采用参考无监督强化学习完成训练。

**💡 创新点**

创新点：①引入 AC-GenRM 动态生成评价标准并提供无参考奖励；②通过 ACPO 进行无参考强化学习，自动区分计划式与直接生成模式，展现元认知能力；③在单一模型中统一长短文本生成。

**🔧 技术方法**

技术手段：参考无监督强化学习、生成式奖励模型 (AC-GenRM)、自适应约束偏好优化 (ACPO)、群体相对策略优化 (GRPO)、动态长度正则化和结构约束。

**📊 数据集**

使用数据集：LitBench、Blessing、WritingBench、HelloBench、Story Writing Benchmark、UNCLE、LongWeave、LongWriter-Zero 等多源长短文本和偏好对。

**📈 对比分析**

方法比较：与多种大型 LLM 和现有基准对比，在 WritingBench 长文本得分提升约 +5，Blessing 短文本提升 25+，模式辨识准确率从 64% 提升至 96%+，在多项指标上优于现有方法。

**⚠️ 局限性**

局限性：需要较大模型规模（4B/8B）才能实现元认知；计算成本高，尤其是长文本 RL；二元模式切换对中等长度任务适配不足；训练资源需求高。

---

## 306. Coupling Macro Dynamics and Micro States for Long-Horizon Social Simulation

**arXiv ID:** 2604.05516 | [PDF](https://arxiv.org/pdf/2604.05516v1)

**作者:** Yunyao Zhang `[一作]`, Zikai Song `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MF-MDP 框架，将宏观均值场与微观代理状态耦合，实现长期社交网络仿真；

**💡 创新点**

创新点在于显式地为每位代理赋予潜在意见状态，并用长时一致性训练（LCT）在宏观层面学习状态转移、微观层面采样多步 roll‑out 并重选动作；

**🔧 技术方法**

采用 Transformer 事件级状态转移模型、LoRA 微调的 LLM 策略、dropout 随机子网络采样和 KL‑一致性损失实现宏观-微观双向耦合；

**📊 数据集**

使用微博（Weibo）语料库以及从微博和抖音爬取的舆情反转事件，构建长达数千步的真实轨迹；

**📈 对比分析**

与 Direct LLM、Social Retrieval 和 MF‑LLM 在短期、长期及反转场景下的 KL、Wasserstein、DTW、Macro‑/Micro‑F1 等指标对比，MF‑MDP 在长期 KL 从 1.2490 降至 0.3089（↓75.3%），反转 KL 从 1.6425 降至 0.5434（↓66.9%），同时保持或提升 F1 分数；

**⚠️ 局限性**

局限在于 LLM 生成的文本摘要可能带来标注噪声、对连续情绪变化的离散状态划分有限，且方法若被用于操纵舆论存在伦理风险。

---

## 307. Uncovering Linguistic Fragility in Vision-Language-Action Models via Diversity-Aware Red Teaming

**arXiv ID:** 2604.05595 | [PDF](https://arxiv.org/pdf/2604.05595v1)

**作者:** Baoshun Tong `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**通讯引用:** 34792 | [OpenAlex ID](https://openalex.org/A5100412937)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多样性增强的强化学习框架DAERT，用于自动生成具有高攻击效果且语言多样性的VLA模型红队攻击指令

**💡 创新点**

创新点在于引入“隐式多样性感知Actor-Critic”与“分层奖励门控”，通过对抗奖励与多样性熵的联合优化，显著缓解模式坍塌，实现更丰富、更有攻击性的指令生成

**🔧 技术方法**

利用Qwen3‑VL‑4B视觉‑语言生成器、GRPO、ROVER思想的多样性奖励、熵正则化、物理基奖励门控（结构、语义、长度三重约束）以及VERL框架进行训练

**📊 数据集**

在LIBERO、CALVIN和SimplerEnv三大机器人操控基准上进行评估，目标模型包括π_0和OpenVLA，且对生成指令在不同任务（空间、物体、目标、长指令）进行交叉测试

**📈 对比分析**

与原始指令、手工生成的ERT以及标准GRPO对比，DAERT在LIBERO上将任务成功率从93.33%降低至5.85%，并在多样性指标（CLIP余弦距离、LLM‑as‑Judge）上分别提升至12.23与8.48；在OpenVLA、CALVIN及SimplerEnv上同样显示最高攻击效果和最佳多样性

**⚠️ 局限性**

局限性包括对目标模型的假设（需保持对语言的依赖）、对奖励门控阈值的手动设定、以及在极端分布转移（如完全不同的感知模态）下仍需进一步验证；此外，RL训练过程仍较为耗时且对计算资源有较高要求

---

## 308. Beyond Tools and Persons: Who Are They? Classifying Robots and AI Agents for Proportional Governance

**arXiv ID:** 2604.05568 | [PDF](https://arxiv.org/pdf/2604.05568v1)

**作者:** Huansheng Ning `[一作]` (University of Science and Technology Beijing), Jianguo Ding `[通讯]` (Blekinge Institute of Technology)

**通讯引用:** 2152 | [OpenAlex ID](https://openalex.org/A5057741133)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

提出了基于Cyber-Physical-Social-Thinking (CPST) 空间理论的三层自动化实体分类框架，以填补现有法规在实体本体论上的空白。

**💡 创新点**

创新点在于将实体的计算、物理、社会和思维四维集成度量为治理依据，形成从“受限演员”到“社会意识交互体”再到“CPST整合代理”的递进分类，并为每层提供相应的治理模型。

**🔧 技术方法**

使用了多学科测量指标，包括计算自主度、机体运动学、社会交互指数（基于HRI问卷与网络中心性）和认知自治评估（如ARC-AGI、METR）。

**📊 数据集**

论文未依赖具体公开数据集，而是构建了基于行业标准（ISO、SAE）和已有研究的综合度量框架，建议在监管沙盒和标准化工作中收集实验数据。

**📈 对比分析**

通过层级匹配与现有法规（EU AI Act、Machinery Regulation）对比，提出了“关系治理框架”与“合格法律人格”两种补充治理路径，尚未给出定量性能指标，但预期能实现更精准的风险匹配和责任归属。

**⚠️ 局限性**

局限性包括：社会整合指标主观性强、跨文化适用性不足、层级阈值不够精确、对非人类面向系统的适用性有限以及可能出现的“层级游戏”监管套利风险。

---

## 309. Turbulence-like 5/3 spectral scaling in contextual representations of language as a complex system

**arXiv ID:** 2604.05536 | [PDF](https://arxiv.org/pdf/2604.05536v1)

**作者:** Zhongxin Yang `[一作]` (Peking University), Shiyi Chen `[通讯]` (Peking University)

**通讯引用:** 39289 | [OpenAlex ID](https://openalex.org/A5100631623)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者将文本视为在 transformer 上下文嵌入空间中的轨迹，计算连续词嵌入差值（embedding‑step）并进行频谱分析，发现其功率谱在多种语言、语料与模型（BERT、GPT‑OSS）中呈现稳健的 5/3 幂律衰减，提示语言表示具备跨尺度自相似结构。

**💡 创新点**

创新点在于首次揭示上下文语言模型嵌入差值的统一 +5/3 幂律特征，并将其与湍流 Kolmogorov 频谱类比，提出一种模型无关、跨语言、跨语料的多尺度结构量化基准。

**🔧 技术方法**

采用离散傅里叶变换对嵌入差值做功率谱估计，使用 OLS 线性回归估计幂律指数；对各维度进行维度平均与归一化；对不同 transformer 层的嵌入做层级分析；还使用静态词向量和随机打乱序列做对照实验。

**📊 数据集**

构建四种语言（中文、英文、德语、日语）的语料库，包括人类撰写的多领域文本与 AI 生成的文本（使用 GPT‑OSS 模型产生），每条样本统一截断为 1200 个 token；所有数据和代码公开在 Zenodo 与 GitHub 上。

**📈 对比分析**

对比方法：在同一语料中分别计算上下文嵌入、静态嵌入和打乱顺序后的嵌入的功率谱；对不同 transformer 层、不同模型、不同语言的指数进行统计。结果显示，所有上下文嵌入均聚集在约 5/3 的指数附近，而静态嵌入和打乱顺序的序列则失去此特征；层级分析表明，随着层数加深指数逐渐趋向 5/3，表明更深层的上下文整合更具尺度自由性。

**⚠️ 局限性**

局限性：尚未揭示 5/3 幂律产生的底层机制，仅证实其存在；实验仅覆盖 transformer‑based 模型和四种语言，可能不适用于其他模型或低资源语言；固定 token 长度和预训练模型的限制也可能影响结果的普适性。

---

## 310. Control Architecture and experimental validation of a Novel Surgical Robotic Instrument

**arXiv ID:** 2604.05610 | [PDF](https://arxiv.org/pdf/2604.05610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 311. Experience Transfer for Multimodal LLM Agents in Minecraft Game

**arXiv ID:** 2604.05533 | [PDF](https://arxiv.org/pdf/2604.05533v1)

**作者:** Chenghao Li `[一作]` (University of Electronic Science and Technology of China), Chaoning Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2050 | [OpenAlex ID](https://openalex.org/A5057230698)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Echo，一种基于显式转移维度（结构、属性、程序、功能、交互）的记忆框架，并结合结构化的上下文类比学习（ICAL）实现多模态 LLM 代理在复杂游戏环境中的经验迁移与任务泛化。

**💡 创新点**

①将可迁移知识拆分为五个可解释的转移维度；②构建统一的上下文状态描述符（CSD）并在此空间检索和检索相关经验；③采用结构化的 ICALL 实现主动检索与验证；④通过自我一致性检查提升长期规划稳定性。

**🔧 技术方法**

大规模多模态语言模型（MLLM）、上下文类比学习（In‑Context Analogical Learning, ICAL）、检索增强的记忆库、CSD 向量化与结构化表示、指令微调、自动验证与回放。

**📊 数据集**

在 Minecraft 环境下的自建任务集，包含多世界、多资源配置的任务执行轨迹、CSD 和计划。

**📈 对比分析**

与 Voyager、MrSteve、MP5、JARVIS‑1 等四个基线在从零学习、跨世界与跨任务泛化以及连续学习等指标下进行比较。Echo 在从零学习阶段的物品解锁速度提升 1.3–1.7 倍；在 2‑shot、4‑shot、8‑shot 等少样本设置中获得最高或相近成绩；连续学习曲线显示中后期学习率最快，最终成功率约 45%，优于大多数基线。

**⚠️ 局限性**

初始学习速率较慢；主要针对 Minecraft 这类规则固定、可预测的环境，真实物理世界的多样性、模糊性和因果复杂度导致迁移效果不易复制；缺乏主动探索能力，信息稀缺时表现不佳。

---

## 312. Controllable Singing Style Conversion with Boundary-Aware Information Bottleneck

**arXiv ID:** 2604.05526 | [PDF](https://arxiv.org/pdf/2604.05526v1)

**作者:** Zhetao Hu `[一作]` (Xi'an Jiaotong University), Jihua Zhu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2854 | [OpenAlex ID](https://openalex.org/A5068185614)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于SVCC2025挑战的可控歌唱风格转换系统S4，目标是消除源风格泄漏、提升动态渲染稳定性，并在数据有限的情况下实现高保真音频输出。

**💡 创新点**

创新点包括：①边界感知Whisper瓶颈（在音素边界内平均语义特征并缩放），有效抑制源风格泄漏；②显式帧级技术矩阵结合推理时F0加工，精准控制振颤、滑音等快速动态；③辅助48kHz SVC模型完成高频带，实现48kHz音质而不需大规模数据。

**🔧 技术方法**

使用技术包括Whisper预训练编码器、VITS解码器、NSF‑HiFiGAN声码器、蒙特利尔强制对齐器、技术矩阵嵌入、规则化F0加工、频域高频补全。

**📊 数据集**

数据集：SVCC2025官方任务1数据集（约68小时）及其音素、MIDI、F0、UV和技术标注；辅助48kHz补全模型使用同一数据集。

**📈 对比分析**

在官方主观评测中，S4B在自然度（MOS 4.0）排名第一，风格相似度排名第四，身份相似度略低，但在仅使用官方数据的前提下，在自然度和可控性方面优于使用大规模外部数据的参赛系统。

**⚠️ 局限性**

局限性：受限于官方数据，身份相似度仍不及部分预训练模型；高频补全依赖辅助模型，可能影响风格一致性；技术矩阵的创建依赖人工或规则，自动化程度有限。

---

## 313. SignalClaw: LLM-Guided Evolutionary Synthesis of Interpretable Traffic Signal Control Skills

**arXiv ID:** 2604.05535 | [PDF](https://arxiv.org/pdf/2604.05535v1)

**作者:** Da Lei `[一作]` (Sichuan University), Yuzhan Liu `[通讯]` (Sichuan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用大型语言模型在离线进化过程中生成、评估并迭代改进可解释的交通信号控制技能，并通过事件检测与优先级调度实现事件驱动的可组合控制。

**💡 创新点**

提出SignalClaw框架，将结构化演化信号与LLM提示相结合，生成包含策略描述、选择指导和可执行代码的自记录技能；引入事件检测+优先级调度实现事件专用技能的零样本组合。

**🔧 技术方法**

采用LLM（GPT‑5.4）生成技能，结构化反馈提取与自然语言提示，SUMO仿真评估，AST白名单校验，GEP进化策略，事件检测模块与优先级调度器。

**📊 数据集**

在SUMO构建的4×4 arterial grid交通网络上，使用六个常规训练/验证场景与六个注入事件（急救车辆、公交优先、事故、拥堵）场景，并通过多次随机种子评估。

**📈 对比分析**

与FixedTime、MaxPressure、PI-Light、DQN四个基线对比；在常规场景SignalClaw平均延迟7.8–9.2 s，方差低；在事件场景急救车辆延迟比MaxPressure降低65–85%，公交人延迟降低约75%，平均延迟仅略高于最佳基线。

**⚠️ 局限性**

局限包括仅在单一4×4网格上验证，事件检测假设无噪声；LLM未在进化中自适应导致收敛停滞；缺乏多网络、多城市泛化与混合事件全面评估；仅评估一次进化的方差；未进行真实工程师可解释性用户研究。

---

## 314. Maintaining Random Assignments under Adversarial Dynamics

**arXiv ID:** 2604.05606 | [PDF](https://arxiv.org/pdf/2604.05606v1)

**作者:** Bernhard Haeupler `[一作]` (INSAIT), Anton Paramonov `[通讯]` (ETH Zurich)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5067587616)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套通用的随机分配维护框架，针对自适应对抗性动态变化，设计了新的时间聚合（Temporal Aggregation）原理及其实现算法 Greedy Temporal Aggregation 和 Landmark Resampling，并展示了其在动态图着色、随机游走和 PageRank 等多种应用中的子线性更新复杂度与近似随机分布保证；

**💡 创新点**

创新点在于：①发现并利用时间选择攻击揭示主动重采样的缺陷；②提出时间聚合原则以抑制对抗性时间选择，形成新的两类重采样算法；③通过表格游戏（Table Games）框架统一分析并证明其在多种场景下的理论上最优或近似最优性能；④在动态图着色中实现 O(Δ) 颜色和 O(Δ/lnΔ) 颜色的子线性维护；

**🔧 技术方法**

主要技术包括：主动重采样（Proactive Resampling）、时间聚合与时间聚合参数化、Landmark Resampling 的指数退避与基数化时间点选择、负载函数与拼接分析、表格游戏框架、Chernoff 与大数定理、递推与势能分析；

**📊 数据集**

该工作以理论为主，未使用特定真实数据集；实验与评测全部基于理论分析与合成图模型；

**📈 对比分析**

方法对比主要通过理论复杂度与概率保证：相较传统主动重采样，Greedy Temporal Aggregation 在每一步仅需 O(log n) 次额外重采样即可保持 O(log n) 负载放大；Landmark Resampling 在每一步仅需 O(log T) 次重采样即可实现 O(log T) 负载放大；在动态图着色方面，实现了 O(Δ) 颜色与 O(Δ/lnΔ) 颜色的子线性更新；整体性能均为子线性或对数级；

**⚠️ 局限性**

局限性包括：①仍需对抗性动态假设下的重采样次数上限；②部分结果依赖于图的最大度或无三角形限制；③在极端自适应攻击下可能仍存在微量偏差；④对实际大规模网络的实验验证尚缺乏。

---

## 315. Evaluation Before Generation: A Paradigm for Robust Multimodal Sentiment Analysis with Missing Modalities

**arXiv ID:** 2604.05558 | [PDF](https://arxiv.org/pdf/2604.05558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 316. ID-Selection: Importance-Diversity Based Visual Token Selection for Efficient LVLM Inference

**arXiv ID:** 2604.05601 | [PDF](https://arxiv.org/pdf/2604.05601v1)

**作者:** Zhaohong Huang `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 32282 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种训练无关的视觉标记剔除方法ID-Selection，用以加速大型视觉‑语言模型的推理

**💡 创新点**

创新点在于将重要性评分与多样性驱动的迭代选择相结合，通过分数抑制机制在保留信息量的同时逐步消除冗余

**🔧 技术方法**

主要技术包括基于视觉编码器注意力、跨模态注意力或两者融合的统一重要性评分；以及基于余弦距离的指数抑制权重实现迭代选取

**📊 数据集**

在16个视觉与文本任务数据集上验证：9张图像理解基准、4文本导向的文档理解基准、3视频问答基准；以及对LLaVA、LLaVA‑Next、Video‑LLaVA、Qwen2.5‑VL、InternVL2.5等5大模型进行测试

**📈 对比分析**

与FastV、FasterVLM、SparseVLM、PruMerge、VisionZip、DART、DivPrune、VisPruner、CDPruner等9种主流剔除方法相比，ID-Selection在多达97.2%裁剪率下仍能保持约90%原始性能，且在极端压缩下平均提升3–7%性能，并在推理速度与FLOPs、KV Cache使用率方面实现10×加速和90%+节省

**⚠️ 局限性**

局限性在于对极小保留标记数时依赖重要性估计的准确性，且对超大视觉输入的相似度计算在GPU内存上仍有一定开销；此外，统一重要性需要文本编码器支持，对无文本编码器的模型需退化到单一重要性指标

---

## 317. AI-Driven Modular Services for Accessible Multilingual Education in Immersive Extended Reality Settings: Integrating Speech Processing, Translation, and Sign Language Rendering

**arXiv ID:** 2604.05591 | [PDF](https://arxiv.org/pdf/2604.05591v1)

**作者:** N. D. Tantaroudas `[一作]` (Institute of Communications and Computer Systems (ICCS)), E. Papatheou `[通讯]` (University of Exeter)

**通讯引用:** 1132 | [OpenAlex ID](https://openalex.org/A5078366434)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个模块化的 XR 语言学习平台，集成了六项 AI 服务（语音识别、文本翻译、语音合成、情感分析、会话摘要和国际手语（IS）动画），通过 Meta Quest 3 VR 头显为聋哑和听力学生提供实时、可访问的多模态教学体验。

**💡 创新点**

创新点包括：① 采用微服务架构将多模态 AI（ASR、NMT、TTS、情感、摘要、IS）整合进 XR 环境；② 将 IS 作为跨国手语通用语，突破单一国家手语的限制；③ 构建 750 个 IS 手势视频数据集，并通过 MediaPipe 关键点映射到 3D avatar，实现低延迟（<300 ms）实时手语动画；④ 在欧盟数字包容框架下实现多语言、多模态可访问性，为语言教育提供全新的包容性平台。

**🔧 技术方法**

所用技术包括 OpenAI Whisper（ASR）、Meta NLLB/EuroLLM（NMT）、AWS Polly（TTS）、RoBERTa（情感分析）、flan‑t5‑base‑samsum（摘要）、Google MediaPipe（手势检测）、Unity + XR Interaction Toolkit + OpenXR、Docker + AWS Lambda/EC2、Meta Quest 3。

**📊 数据集**

使用的主要数据集有：750 个国际手语（IS）手势视频（来自 HandSpeak、SpreadTheSign 等公开资源），以及用于翻译和 TTS 基准的公开文本/音频数据（NLLB、EuroLLM、标准语音合成语料）。

**📈 对比分析**

通过 1,000 并发用户的负载测试验证平台实时性能，平均响应时间 < 800 ms；IS 动画延迟 < 300 ms；TTS 比较四个服务，AWS Polly 首字节延迟 50–100 ms、MOS 3.5–3.8；NLLB 与 EuroLLM 基准比较，EuroLLM Instruct 取得 84.34 BLEU、0.529 s/翻译，优于 NLLB 79.25 BLEU、0.596 s，表明平台满足 XR 实时交互需求。

**⚠️ 局限性**

局限性包括：IS 词汇量仅 750 个手势，缺乏非手势表达（面部表情、身体姿势）；未进行正式可用性或学习效果评估；平台仅在 Meta Quest 3 上测试，未覆盖其他 XR 设备；数据集规模有限，未能充分代表多语种手势；缺乏多玩家交互和跨模态反馈功能。

---

## 318. Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher

**arXiv ID:** 2604.05584 | [PDF](https://arxiv.org/pdf/2604.05584v1)

**作者:** Pengcheng Weng `[一作]` (Xi'an Jiaotong University), Fei Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 37295 | [OpenAlex ID](https://openalex.org/A5100455803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个“Purify‑then‑Align”框架，先通过元学习动态抑制低质量模态的影响，再使用扩散模型进行知识蒸馏，将多模态教师的清晰特征对齐到单模态学生上，从而显著提升单模态和多模态下的鲁棒性。

**💡 创新点**

创新点在于：①将污染效应与表征差距的因果关系明确化并通过两阶段的顺序处理解决；②元学习权重机制实现自适应地抑制噪声模态；③利用扩散模型的知识蒸馏实现高效的特征对齐，兼顾表示差距与噪声污染。

**🔧 技术方法**

主要技术手段包括：元学习（meta‑learning）动态权重学习、扩散模型与DDIM的知识蒸馏、轻量化投影层与噪声自适应模块、以及基于强化的多模态融合。

**📊 数据集**

使用的大规模数据集为：MM‑Fi（人姿态估计，包含 Depth、LiDAR、Wi‑Fi）和 XRF55（RF 动作识别，包含 mmWave Radar、Wi‑Fi CSI、RFID）。

**📈 对比分析**

与传统特征级/决策级融合、X‑Fi 等方法对比，单模态性能提升 12%–47%（MPJPE）或 6%–27%（准确率），多模态缺失情况下平均提升 20%+，并在多种模态缺失组合上多次获得最佳或近最佳成绩。

**⚠️ 局限性**

局限性包括：对极低质量模态的扩散生成有时会引入噪声导致性能下降；元学习与扩散过程需要较高计算成本；在不同任务和数据域中的迁移性与泛化性尚需进一步验证。

---

## 319. Stop Fixating on Prompts: Reasoning Hijacking and Constraint Tightening for Red-Teaming LLM Agents

**arXiv ID:** 2604.05549 | [PDF](https://arxiv.org/pdf/2604.05549v1)

**作者:** Yanxu Mao `[一作]` (Henan University), Datao You `[通讯]` (Henan University)

**通讯引用:** 49 | [OpenAlex ID](https://openalex.org/A5005904745)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种新的红队攻击框架 JailAgent，用于在不修改用户提示的前提下隐式操纵大型语言模型（LLM）代理的推理过程，提升了攻击的隐蔽性和效果。

**💡 创新点**

创新点在于三阶段流水线（Trigger Extraction、Reasoning Hijacking、Constraint Tightening）实现了对代理记忆检索与推理轨迹的隐式操控；采用了实时自适应的重排序模型以及联合优化四种损失函数（特异性、聚类、可分性、间隔）以提升触发器的有效性与泛化能力。

**🔧 技术方法**

技术手段包括：BERT 词片子级别的子词映射与依存句法解析；基于 log‑prob 与 KL 散度的两阶段重要性评估；利用 SentenceTransformer 编码的 Reranker 进行快速微调；多目标联合优化的触发器嵌入学习；以及多种评估指标（ASR-R/L/H、ACC/EM/SR、CR）进行攻击效果量化。

**📊 数据集**

实验使用 8 个公开数据集：VideoAgent（EgoSchema、NExT‑QA）、ReAct‑UALA（HotpotQA、StrategyQA、MMLU）和 EHRAgent（MIMIC‑III、eICU、TREQS）。

**📈 对比分析**

与基线方法 PAIR、AgentPoison、BadChain 以及无攻击基线比较，JailAgent 在所有 LLM 核心（GPT‑3.5‑turbo、GPT‑4o、GPT‑5、Llama‑3.1‑70B、Claude‑3.5‑haiku、Gemini‑3.0‑pro 等）和代理架构上均取得显著更高的 ASR 成功率，且在原始任务性能（ACC/EM/SR、CR）几乎不受影响；在时间成本上，JailAgent 的 TCPS 下降幅度可达 70%+。

**⚠️ 局限性**

局限性包括：依赖 shadow 模型进行触发器识别，可能在完全黑盒环境下表现下降；实时自适应机制增加计算开销，可能导致大规模实时系统的延迟问题；实验主要集中在预定义任务与标准评估环境，尚未验证在更复杂、动态变化的实际应用场景中的鲁棒性。

---

## 320. EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes","Hands" and "Minds"

**arXiv ID:** 2604.05541 | [PDF](https://arxiv.org/pdf/2604.05541v1)

**作者:** Qin Wang `[一作]` (Fudan University), Yuanyuan Wang `[通讯]` (Fudan University)

**通讯引用:** 12887 | [OpenAlex ID](https://openalex.org/A5100423182)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一套完整的EchoAgent系统，实现心脏超声全流程自动解读。

**💡 创新点**

首次将“眼-手-脑”协同机制与专家知识库、工具调用、推理引擎三大模块结合，为心超提供端到端可靠分析。

**🔧 技术方法**

采用专业知识抽取引擎、层级协同工具包（视图识别、分割、量化测量）和协同推理枢纽，基于Qwen3‑VL‑Plus等大型多模态语言模型。

**📊 数据集**

在CAMUS（EF分级）和MIMIC‑EchoQA（多结构问答）两大公开数据集上进行评估。

**📈 对比分析**

相较于单一任务模型、通用LLM和增强版GPT‑5，EchoAgent在EF分级上准确率达80%/92%，在EchoQA上平均准确率超过84%，表现显著优于现有方法。

**⚠️ 局限性**

仍依赖外部工具调用和大量规则化知识，缺乏对极端视图或病理形态的泛化能力，且模型规模与推理成本较高。

---

## 321. From Large Language Model Predicates to Logic Tensor Networks: Neurosymbolic Offer Validation in Regulated Procurement

**arXiv ID:** 2604.05539 | [PDF](https://arxiv.org/pdf/2604.05539v1)

**作者:** Cedric Haufe `[一作]` (Harz University of Applied Sciences), Frieder Stolzenburg `[通讯]` (Harz University of Applied Sciences)

**通讯引用:** 980 | [OpenAlex ID](https://openalex.org/A5025712440)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个神经符号化流水线，用于在受监管的公共采购环境中验证投标文件的有效性。

**💡 创新点**

创新点在于将大型语言模型提取的谓词与逻辑张量网络（LTN）相结合，既保持了语言模型的语义理解能力，又提供了可追溯、可审计的规则推理与可解释性。

**🔧 技术方法**

使用了Qwen2.5 14B/32B等大型语言模型进行谓词提取（MCSR、CISC），以及基于Real Logic的逻辑张量网络实现模糊规则聚合；还与BERT、纯LTN、信息抽取+规则等基线进行对比。

**📊 数据集**

实验基于200份来自德国梅塞堡应用科学大学的采购PDF文件，其中35%被标记为有效投标，其余为非投标文档。

**📈 对比分析**

采用5折交叉验证（5次重复，25折）评估，主要指标为正类（有效投标）的F1分数。最佳性能为LTN（Łukasiewicz）F1≈0.899，其次是MCSR-BestConf+LTN（0.874）和BERT（0.859）；CISC+LTN和LLM单独方法表现较弱。

**⚠️ 局限性**

局限性包括数据量小、仅单一标注者、规则手工编写、仅在德语环境下验证，且缺乏端到端训练，难以推广到更大、跨域的数据集。

---

## 322. Parameterized algorithms for $k$-Inversion

**arXiv ID:** 2604.05528 | [PDF](https://arxiv.org/pdf/2604.05528v1)

**作者:** Dhanyamol Antony `[一作]` (Indian Institute of Science Education and Research), R. B. Sandeep `[通讯]` (Indian Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了多种固定参数可行（FPT）算法，用以解决有向图的 k‑Inversion 问题，先对锦标赛的加权限制版进行改进，再通过动态规划递归到块图，最后给出基于树宽的通用算法。

**💡 创新点**

创新点在于将迭代压缩（iterative‑compression）方法推广到权重受限的锦标赛版本；利用块树（block tree）结构高效求解块图；并通过显式 DP 推导出 2^O(tw(k+tw)) 的时间复杂度，显著优于仅基于 Courcelle 定理的理论可行性。

**🔧 技术方法**

核心技术包括迭代压缩、MSO₂ 公式表达与 Courcelle 定理、块树与树分解（nice tree decomposition）的动态规划、以及对权重约束的组合优化。

**📊 数据集**

论文为理论工作，没有使用实验数据集；所有结论均基于算法分析与证明。

**📈 对比分析**

相比已有的仅针对锦标赛的 FPT 方案，本文的 2^O(tw(k+tw)) 运行时间在树宽较小的实例上具有可观的实际效益；在块图上可直接得到多项式时间解；整体性能通过复杂度分析得到严格上界。

**⚠️ 局限性**

局限性在于算法尚未扩展到更一般的图结构，例如可被两个大交集的团覆盖的图；对这类结构是否仍能保持 FPT 性仍是开放问题。

---

## 323. High-Resolution Single-Shot Polarimetric Imaging Made Easy

**arXiv ID:** 2604.05581 | [PDF](https://arxiv.org/pdf/2604.05581v1)

**作者:** Shuangfan Zhou `[一作]` (Beijing University of Posts and Telecommunications), Imari Sato `[通讯]` (National Institute of Informatics)

**通讯引用:** 2391 | [OpenAlex ID](https://openalex.org/A5101052713)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于三摄的高分辨率单快照偏振成像系统EasyPolar，并设计了信心引导的物理信息重建网络。

**💡 创新点**

创新点在于将高光通过无偏振参考摄像机与两个极化摄像机结合，避免了DoFP的分辨率损失，同时通过视角编码、几何先验和置信度门控实现多视角融合的物理一致性。

**🔧 技术方法**

使用同步RGB摄像机、线性偏振滤光片、立体匹配（Disparity）、视角（Plücker坐标）编码、几何编码、注意力融合、置信度门控重建网络以及三角函数AoP编码。

**📊 数据集**

使用Mitsuba渲染的HSSD合成数据集以及自制三摄硬件采集的真实数据。

**📈 对比分析**

与DoFP+PIDSR硬件基线和PolarAnything生成式基线进行对比，EasyPolar在合成和真实数据上在PSNR/SSIM/MAE等指标上均优于两者，尤其在AoP和DoP的重建质量显著提升。

**⚠️ 局限性**

受限于跨视角几何匹配的准确性，需依赖高精度立体匹配；对异构摄像机平台（如手机双摄）适应性不足。

---

## 324. Understanding User Privacy Perceptions of GenAI Smartphones

**arXiv ID:** 2604.05571 | [PDF](https://arxiv.org/pdf/2604.05571v1)

**作者:** Ran Jin `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 39401 | [OpenAlex ID](https://openalex.org/A5107480590)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过半结构化访谈与焦点小组，深入挖掘普通手机用户对系统级GenAI手机的隐私感知、担忧及期望的设计建议。

**💡 创新点**

首次将用户隐私关注与GenAI手机的系统架构、数据生命周期、权限机制三层相结合，提出面向系统、数据和用户界面的多维隐私增强建议，并通过技术专家评估其可行性与实现难度。

**🔧 技术方法**

主要采用质性研究方法：访谈、焦点小组、主题编码与可信度检验（Krippendorff alpha、团队编码），以及对建议的五点Likert量化评估。

**📊 数据集**

数据来源为22名日常手机使用者（其中包含IT/CS专业与非专业人员）通过自填问卷与访谈收集的文本资料，未使用公开数据集。

**📈 对比分析**

研究不涉及算法或性能对比；比较点在于对用户隐私关注的前后变化（S3前后隐私关注度提升）以及对建议方案的专家可行性评分，未提供客观性能指标。

**⚠️ 局限性**

局限性：样本量小且偏向技术熟悉者，可能低估更广泛用户群体的观点；依赖自述，存在回忆偏差；研究对象为早期产品，隐私风险与使用场景随技术进步可能发生变化；未进行纵向跟踪或实地使用观察。

---

## 325. AutoSOTA: An End-to-End Automated Research System for State-of-the-Art AI Model Discovery

**arXiv ID:** 2604.05550 | [PDF](https://arxiv.org/pdf/2604.05550v1)

**作者:** Yu Li `[一作]` (Tsinghua University), Tie-Yan Liu `[通讯]` (Zhongguancun Academy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

AutoSOTA 构建了一个多智能体闭环系统，实现从论文到可执行代码的完整自动化复制与方法改进流程。

**💡 创新点**

核心创新包括：将科研工作拆分为资源准备、实验评估、反思与创意三阶段的八个专用智能体；使用红线约束保证实验可比性；以及通过实验与案例证明能在多领域自动发现并实现 SOTA 超越。

**🔧 技术方法**

技术手段涵盖：LLM 驱动的自然语言理解与代码生成、图谱式评估指标树、外部资源检索与下载、实时监控与错误修复、版本控制与超时管理、以及自适应的多任务调度与监督。

**📊 数据集**

数据来源为 105 篇 2025 年顶级 AI 会议（ICLR、NeurIPS、ICML、CVPR、ICCV、ACL、NAACL、AAAI）公开的实验代码与数据集，涵盖 LLM、NLP、CV、时间序列、优化等多个领域。

**📈 对比分析**

与原始论文直接对比，AutoSOTA 在 105 篇任务上平均提升约 10%（某些案例超过 30%），在 5 小时内实现多领域 SOTA，案例演示包括 LLM 推理加速、图神经网络改进、物理模型融合、视觉特征重构、采样优化等。

**⚠️ 局限性**

局限性：依赖公开代码与数据集，无法处理无代码或不完整仓库；高度依赖 LLM 质量，可能产生错误或偏差；对资源消耗与 GPU 时长有限制；红线约束虽然防止伪增益，但可能限制某些有意义的改进方向。

---

## 326. COSMO-Agent: Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration

**arXiv ID:** 2604.05547 | [PDF](https://arxiv.org/pdf/2604.05547v1)

**作者:** Liyuan Deng `[一作]` (Northwestern Polytechnical University), Huaxi Huang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 351 | [OpenAlex ID](https://openalex.org/A5049609631)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了COSMO-Agent，基于工具增强的强化学习框架，实现闭环CAD–CAE迭代优化。

**💡 创新点**

创新点在于将CAD生成、CAE求解、结果解析与几何修订建模为交互式RL环境，并设计多约束奖励与工具链日志奖励，显著提升可执行性与鲁棒性。

**🔧 技术方法**

使用大型语言模型（Qwen3‑8B）结合GRPO训练，集成CADQuery、FreeCAD/CalculiX、Gmsh等工具，采用工具链日志驱动奖励。

**📊 数据集**

构建了工业级可执行CAD–CAE基准数据集，约2万条样例，覆盖25种部件类别、约束阈值与材料库。

**📈 对比分析**

与多种开源与闭源LLM（如Intern‑S1、Gemini‑3‑Flash）在统一JSON输出与工具调用预算下对比，COSMO-Agent在完整成功率（74.5%）和工具调用效率（6.72次）上明显领先。

**⚠️ 局限性**

局限在于当前仅支持单部件、线性静力学和有限工具类型，且对更复杂的接触、装配或非线性多物理耦合的适应性尚未验证。

---

## 327. Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects

**arXiv ID:** 2604.05546 | [PDF](https://arxiv.org/pdf/2604.05546v1)

**作者:** Jun Zhang `[一作]` (Zhejiang University), Huan Li `[通讯]` (Zhejiang University)

**通讯引用:** 16585 | [OpenAlex ID](https://openalex.org/A5100319241)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对大型视觉语言模型（LVLM）推理过程进行系统综述，提出了以编码、预填充、解码三阶段为核心的效率技术分层分类法，并通过实验验证了多种压缩、稀疏注意力、KV 缓存压缩等技术在不同阶段的效果。

**💡 创新点**

创新点：①首次将 LVLM 推理拆解为三阶段的全链路视角，揭示上游决策如何决定下游瓶颈；②构建了围绕信息密度、长上下文注意力和内存带宽三轴的统一税onomies，帮助研究者在视觉保真与系统效率之间做权衡；③提出四个未来前沿（混合压缩、模态感知解码、流式持续推理、端到端系统协同设计），并给出初步实验佐证。

**🔧 技术方法**

技术：多种视觉编码器与适配器压缩（FastViT、FastVLM、Q-Former 等）、键值缓存压缩（LOOK-M、VL-Cache、VidKV 等）、稀疏注意力（MMInference、VideoNSA 等）、预测性解码（Spec-LLaVA、FLASH 等）、多级稀疏与量化、流式视频特定策略（StreamingTOM、TimeChat-Online 等）以及整体系统分层部署框架。

**📊 数据集**

实验使用的公开数据集包括 MileBench、VideoChatGPT、VideoDetailCaption，均为 Apache 2.0 或 CC‑BY‑4.0 许可。

**📈 对比分析**

对比方法：在上述数据集上对比传统全密集推理、仅压缩或仅稀疏注意力方案，评估时间‑一阶延迟（TTFT、TPOT）与内存占用。实验结果表明，结合混合压缩与 KV 缓存稀疏化可使 TTFT 减少约 30‑50%，TPOT 降低 20‑35%，同时保持 90% 以上的任务准确率。

**⚠️ 局限性**

局限性：①聚焦图像/视频领域，未深入文档理解、OCR 等离散结构；②实验仅考虑延迟与内存，未系统评估能耗与能效比；③对闭源模型（如 GPT‑4o）的优化细节缺失，可能导致评估不完整；④多阶段优化在实际部署中的调度与资源分配仍需进一步研究。

---

## 328. Referring-Aware Visuomotor Policy Learning for Closed-Loop Manipulation

**arXiv ID:** 2604.05544 | [PDF](https://arxiv.org/pdf/2604.05544v1)

**作者:** Jiahua Ma `[一作]` (Sun Yat-sen University), Ruimao Zhang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 4790 | [OpenAlex ID](https://openalex.org/A5003608795)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Referring-Aware Visuomotor Policy（ReV），一种闭环策略学习框架，可在机器人执行过程中实时响应人类或高层规划器提供的稀疏参照点，解决因外部扰动导致的分布外失败。

**💡 创新点**

创新点在于（1）引入双头扩散模型（global 与 local）实现粗细两级轨迹生成；（2）在轨迹生成中插入“轨迹引导策略”，将参照点映射到时间轴并通过掩码式去噪直接引导全局与局部扩散；（3）使用仅基于专家演示的轻量级数据增强与时序位置预测实现对参照点的精确定位。

**🔧 技术方法**

采用扩散模型（Denoising Diffusion Probabilistic Models）作为策略生成器，配合 Transformer 进行时序位置预测；使用掩码去噪（Masked Denoising）实现轨迹引导；并在训练时利用随机扰动与多项式样条进行数据增强。

**📊 数据集**

在仿真与真实机器人上使用 RoboFactory 任务集（Pick Meat、Lift Barrier、Place Food、Camera Alignment）以及相应的带参照点版本（通过随机生成 via-point）进行实验；同时在多机器人、多任务的 Adroit、DexArt、MetaWorld、RoboFactory 等基准上验证双头扩散的有效性。

**📈 对比分析**

与 ACT、DP3、CDP、Octo、MPD 等条件式生成方法对比，ReV 在参照点穿透率、任务成功率和轨迹平滑度上均领先；在真实世界30次试验中，ReV 的整体成功率达到 100%，显著优于其他基线。

**⚠️ 局限性**

局限性包括仅支持单个参照点，无法处理多点或连续参照；对参照点生成方式未作深入研究；当参照点与演示分布相差过大时性能仍会下降；并且模型在极端动态场景下的鲁棒性尚待进一步验证。

---

## 329. Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting

**arXiv ID:** 2604.05540 | [PDF](https://arxiv.org/pdf/2604.05540v1)

**作者:** Jinhu Fu `[一作]` (Beijing University of Posts and Telecommunications), Sen Su `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 4814 | [OpenAlex ID](https://openalex.org/A5036865453)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CoT2Edit 这一新的知识编辑范式，使大型语言模型能够通过链式推理（CoT）对知识进行更新，并在推理时动态检索已编辑的事实。

**💡 创新点**

创新点包括：① 通过语言模型代理生成结构化和非结构化编辑场景下的 CoT 训练数据；② 采用两阶段训练——SFT+GRPO，利用强化学习提升对未知编辑的泛化能力；③ 在推理阶段集成 RAG，实现实时检索编辑事实，从而避免单步生成的幻觉。

**🔧 技术方法**

使用的技术包括：LLM 代理生成 CoT、监督微调（SFT）、Group Relative Policy Optimization（GRPO）强化学习、检索增强生成（RAG）以及多轮自进化的数据筛选策略。

**📊 数据集**

主要使用的数据集：MQuAKE 与 MQuAKE‑uns（结构化/非结构化编辑实例），HotpotQA（实体关系提取用于数据扩增），以及评估基准 CounterFact、ZsRE、WikiUpdate、CounterFact‑uns、MQuAKE、ConvSent 等。

**📈 对比分析**

与现有方法（Fine‑tune、RoME、MEMIT、IKE、SKEME、PMET、GLAME、AlphaEdit、EditCoT、LTE 等）对比，CoT2Edit 在六大编辑基准上单轮训练即可获得最高或接近最高的编辑成功率、改写/邻域成功率，尤其在 OOD 任务 ConvSent 上提升约 11% 以上；在结构化与非结构化场景均保持 80%+ 的准确率。

**⚠️ 局限性**

局限性：指令数据仅覆盖多跳推理和非结构化问答，缺乏更丰富的编辑场景，未来需要构建更全面、多样化的编辑指令集以进一步提升适应性。

---

## 330. A canonical generalization of OBDD

**arXiv ID:** 2604.05537 | [PDF](https://arxiv.org/pdf/2604.05537v1)

**作者:** Florent Capelli `[一作]` (University of Artois), Guy Van den Broeck `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了一种新的知识编译表示形式——树决策图（Tree Decision Diagram，TDD），它是对有序二叉决策图（OBDD）的一般化，兼具结构化确定性 DNNF 的语义特性与 OBDD 的可扩展性。作者通过定义 TDD 的结构、确定性约束以及最小化算法，证明了 TDD 能够高效支持模型计数、枚举、条件化、取值运算等常用查询，并且是可归约的（canonical）与最小的。进一步，作者展示了如何利用 TDD 对具有界定树宽（treewidth）的 CNF 公式进行自底向上的编译，得到 FPT 大小的表示。最后，对 TDD 与 OBDD、SDD、结构化确定性 DNNF 等表示语言进行了对比与分析。

**💡 创新点**

创新点主要有三方面：
1. 设计了 TDD 这一新的表示语言，结合 vtree 结构和确定性约束，既保留了 OBDD 的可操作性，又实现了更紧凑的编码；
2. 提出了完整的最小化与判等算法，证明 TDD 的最小化是多项式时间且结果唯一（canonical），这在结构化确定性 DNNF 中尚未实现；
3. 通过因子宽度（factor width）理论，给出了对具有界定树宽 CNF 公式的 FPT 编译算法，解决了 OBDD 无法在 FPT 时间内表示此类公式的长期难题。

**🔧 技术方法**

技术方法包括：
- 基于 vtree 的树形结构，定义 nTDD（非确定性 TDD）与 TDD（确定性 TDD）；
- 确定性约束（同类节点输入不重叠、叶子层无冲突标签）使得 TDD 在合并、apply、否定等操作时保持可行；
- 最小化算法采用“twins 合并”（twin contraction）与“去除无效节点”，时间复杂度为 O(k·|X|)；
- 因子宽度与树宽的关系，利用动态规划实现自底向上编译；
- 对 CNF 公式的树宽/入射树宽做 vtree 构造，从而保证子函数数量 ≤ 2^k；
- 证明 TDD 可直接模拟 OBDD（线性 vtree）并实现 OBDD ↔ TDD 的互译。

**📊 数据集**

本文主要为理论性工作，并未在公开数据集上进行实验。作者以 CNF 公式、隐藏加权位（Hidden Weighted Bit）函数等经典理论实例作为实验对象，演示 TDD 的表达效率与最小化效果。

**📈 对比分析**

比较方法：
- 与 OBDD、SDD、结构化确定性 DNNF 对比，分析相同可支持的查询（模型计数、枚举、条件化、取值运算、合并、取值）与转换（conditioning、forgetting、conjunction、disjunction、negation、apply）等；
- 通过树宽/入射树宽理论，证明 TDD 在 FPT 大小内能表示所有树宽为 k 的 CNF 公式，而 OBDD 在此类公式上只能达到指数大小；
- 在隐藏加权位函数上，TDD 需要指数级节点数，证明其与 OBDD 的指数分离；
- 性能方面，TDD 的最小化与 apply 等操作时间为多项式，且宽度不超过因子宽度的上界；
- 与 SDD 的比较表明，TDD 与 SDD 在某些表达上存在指数差距，尽管两者同属结构化确定性 DNNF。

**⚠️ 局限性**

局限与挑战：
1. 将非确定性 TDD 转换为确定性 TDD 需要指数级宽度增长，导致大规模实例难以处理；
2. 对 1‑输入节点的消除仍有限制，无法在所有情形下保持确定性；
3. 对于某些 CNF 结构（如高树宽、非平衡 vtree），因子宽度可能仍为指数，导致编译复杂度高；
4. 与 OBDD 的分离目前仅为准指数（quasi‑polynomial）级，是否存在更强的指数分离仍是未解问题；
5. 该方法在实际工业问题中的工程化实现与与现有知识编译框架（如 d-DNNF、FBDD）的集成仍需进一步研究。

---

## 331. Simulation-Driven Evolutionary Motion Parameterization for Contact-Rich Granular Scooping with a Soft Conical Robotic Hand

**arXiv ID:** 2604.05531 | [PDF](https://arxiv.org/pdf/2604.05531v1)

**作者:** Yongliang Wang `[一作]` (OMRON SINIC X Corporation), Masashi Hamaya `[通讯]` (OMRON SINIC X Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了基于MuJoCo的软可变圆锥形机械手物理仿真模型，并结合视觉感知生成初始掏取轨迹，随后通过CMA-ES在仿真中对轨迹与手滚转角度进行演化优化，最终实现了将优化后的轨迹无缝迁移至真实机器人，在多种容器与颗粒物料上实现高效掏取。

**💡 创新点**

创新点在于：① 将柔性手与MuJoCo的可变形物体模拟相结合，构建可控制的柔性手模型；② 采用环形罗盘几何抽象从RGB‑D视觉中快速生成初始掏取轨迹；③ 在仿真中同时优化轨迹与手滚转角度，实现零样本（zero‑shot）仿真‑实物迁移；④ 通过演化策略而非手工调参完成全局最优搜索。

**🔧 技术方法**

使用的技术包括：MuJoCo物理仿真、弹簧‑阻尼网络模型、FastSAM与RGB‑D视觉感知、环形罗盘容器抽象、CMA‑ES进化优化、以及实验平台的软手硬件实现。

**📊 数据集**

实验使用自制的颗粒材料（沙子、豆子、米粒、十球等）以及多种形状容器；视觉数据来自RGB‑D相机，未使用公开数据集。

**📈 对比分析**

通过比较掏取颗粒数量、溢出量和碰撞指标等定量评估，与传统硬工具或手工设计轨迹相比，仿真全参数优化后在仿真中成功率最高；在真实机器人中同一轨迹即可在10球任务中完全掏取10球、20球任务高达15–16个、米粒任务平均48.25g，表现显著优于未优化或仅手工调参的方法。

**⚠️ 局限性**

局限性包括：仅针对固体颗粒物料，尚未验证液体或粘稠物料；柔性手的弹簧‑阻尼模型为简化版，可能无法捕捉更复杂的变形与接触行为；对极端容器形状的适配仍有限；缺乏大规模数据驱动或深度学习优化，难以应对更广泛的任务分布。

---

## 332. Inventory of the 12 007 Low-Dimensional Pseudo-Boolean Landscapes Invariant to Rank, Translation, and Rotation

**arXiv ID:** 2604.05530 | [PDF](https://arxiv.org/pdf/2604.05530v1)

**作者:** Arnaud Liefooghe `[一作]` (University of Littoral Cote d'Opale), Sébastien Verel `[通讯]` (University of Littoral Cote d'Opale)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对所有维数为1、2、3的伪布尔函数进行完整的秩-景观不变类枚举，得到12 007个唯一的景观类；

**💡 创新点**

首次同时考虑秩、邻接结构与翻译/旋转对称的强势景观不变性，显著压缩等价类数量，并揭示非单射函数主导景观空间、三维景观可出现失真、零性交叉与平台的全组合；

**🔧 技术方法**

利用组合计数（分割与排列）、图论自同构分析（翻译与旋转）、图论自动同构分组、以及对所有可能函数的枚举；

**📊 数据集**

使用完整的所有低维伪布尔函数（即所有可能的映射）作为数据集；

**📈 对比分析**

以最佳改进和首次改进的局部搜索为基准，分别计算成功率与期望运行时间；在2维类中最佳改进成功率更高，首次改进运行时间更短；在3维类中首次改进在约59%类中更快，最佳改进在约41%类中更快；

**⚠️ 局限性**

仅限于维数≤3，无法扩展到更高维；枚举结果仅适用于秩-景观不变性，未覆盖所有随机优化算法的具体动态；

---

## 333. ActivityEditor: Learning to Synthesize Physically Valid Human Mobility

**arXiv ID:** 2604.05529 | [PDF](https://arxiv.org/pdf/2604.05529v1)

**作者:** Chenjie Yang `[一作]`, Junbo Zhang `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 14062 | [OpenAlex ID](https://openalex.org/A5100778479)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 ActivityEditor 双代理框架，在零样本场景下从人口统计属性生成高保真的人类移动轨迹。

**💡 创新点**

创新点在于把移动合成拆分为意图驱动生成和规则约束编辑两阶段，并通过 GRPO 强化学习让编辑代理内化物理约束，实现跨区域零样本迁移。

**🔧 技术方法**

采用大语言模型（LLM）+ 多代理 Generate‑then‑Edit + 规则校验 + Group Relative Policy Optimization（GRPO）强化学习 + GPT‑5.2 教师模型生成思路训练。

**📊 数据集**

使用 2017 年 National Household Travel Survey（NHTS）数据，抽取加利福尼亚、亚利桑那、乔治亚、俄克拉荷马四州样本。

**📈 对比分析**

与传统深度学习、检索增强、单通道 LLM 基线进行对比，GRPO_Qwen / GRPO_LLaMA 在所有评估指标（时间对齐、统计一致性等）上均优于基线，展现出卓越的零样本跨州性能。

**⚠️ 局限性**

局限性包括仅生成区级活动链缺乏精细 POI 位置；多轮编辑推理耗时大，影响大规模仿真；仍可能出现幻觉和鲁棒性不足的问题。

---

## 334. Market-Bench: Benchmarking Large Language Models on Economic and Trade Competition

**arXiv ID:** 2604.05523 | [PDF](https://arxiv.org/pdf/2604.05523v1)

**作者:** Yushuo Zheng `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21916 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 Market‑Bench，构建了一个多智能体的闭环供应链经济环境，用于评估 LLM 在采购、定价与营销三大任务中的综合能力。

**💡 创新点**

创新点在于将预算约束的拍卖、价格竞争与 Persona‑Gated Attention 语义匹配机制结合，形成既能衡量数值优化又能衡量生成语言的完整评测框架，并生成可作为数据集的完整交互轨迹。

**🔧 技术方法**

采用多单位一价拍卖、两阶段注意-购买模型、句子与买家身份的嵌入相似度计算、LLM 生成的自由文本口号，以及自动化的经济与语义指标计算。

**📊 数据集**

使用内部仿真生成的交易轨迹数据，没有依赖公开的标准数据集，实验涵盖 20 个不同 LLM 后端（含 Gemini、Phi‑4、GPT‑4o 等）。

**📈 对比分析**

通过对 20 个 LLM 在 10 轮实验中的利润、净利率、库存周转率、填充率和语义匹配得分等指标进行比较，发现少数高性能模型（如 Gemini 2.5 Pro/Flash）能持续获利并复利，而大多数模型仅维持收支平衡或出现亏损，呈现“赢家通吃”的现象。

**⚠️ 局限性**

局限性包括仅在单一市场配置（20 代理、6 步、8 种商品）下实验，买家模型为合成且不考虑篮子购买或客户学习，缺乏对不同供需比、供应不确定性、多市场和多语言环境的系统评估。

---

## 335. SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation

**arXiv ID:** 2604.05656 | [PDF](https://arxiv.org/pdf/2604.05656v1)

**作者:** Wuyang Luan `[一作]` (Jilin University), Rui Ma `[通讯]` (Jilin University)

**通讯引用:** 11317 | [OpenAlex ID](https://openalex.org/A5100710180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SnapFlow，一种自蒸馏方法，将流匹配式视觉‑语言‑动作模型的多步去噪压缩为单前向传递，从而显著降低推理延迟。

**💡 创新点**

创新点在于使用模型自身的边缘速度预测来校正一致性目标、采用渐进式 FM/一致性混合以及零初始化的目标时间嵌入，实现无教师、无架构改动的高效压缩。

**🔧 技术方法**

结合了流匹配、两步 Euler 快捷目标、一致性训练、目标时间嵌入、渐进混合和梯度检查点等技术。

**📊 数据集**

在 LIBERO（四套、40 个任务、400 轮）和 SmolVLA 的 PushT 数据集上进行评估。

**📈 对比分析**

与 10 步默认模型、单步 Naïve 以及 π0、OpenVLA、Octo、Diffusion Policy 等基线对比，SnapFlow 在 LIBERO 上实现 98.75% 成功率（高于 97.75% 的 10 步教师）并实现 9.6× 的去噪加速，整体推理时间从 274 ms 降至 83 ms；在 SmolVLA 上 MSE 降低 8.3% 并获得 3.56× 的整体加速。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，缺乏真实机器人测试；需先行训练好的流匹配模型；去噪压缩后 VLM 前缀成为新的瓶颈，进一步加速需结合 VLM 侧优化。

---

## 336. From Uniform to Learned Knots: A Study of Spline-Based Numerical Encodings for Tabular Deep Learning

**arXiv ID:** 2604.05635 | [PDF](https://arxiv.org/pdf/2604.05635v1)

**作者:** Manish Kumar `[一作]` (BASF), Benjamin Säfken `[通讯]` (Clausthal University of Technology)

**通讯引用:** 996 | [OpenAlex ID](https://openalex.org/A5016630281)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估并比较了标准缩放、最小最大缩放、PLE、B‑spline、M‑spline、I‑spline以及可学习结点等数值编码方法在三种表格深度学习骨干（MLP、ResNet、FT‑Transformer）上的表现；

**💡 创新点**

创新点在于提出可学习结点的软max+累计求和参数化实现样条基的端到端稳定优化，并统一比较三类样条基的性能与计算成本；

**🔧 技术方法**

采用基于样条基的特征展开、可学习结点参数化、LayerNorm等技术，并使用AdamW、早停与学习率调度等训练协议；

**📊 数据集**

使用25个公开表格数据集（来自UCI和OpenML），涵盖回归与分类任务；

**📈 对比分析**

通过5折交叉验证的平均NRMSE（回归）和AUC（分类）进行比较，发现PLE在分类任务中最稳健，回归任务中B‑和I‑spline在不同输出尺寸下表现最佳，可学习结点可提升性能但计算成本显著；

**⚠️ 局限性**

局限在于仅评估有限的编码和骨干，未探究特征级自适应编码，学习结点方法的训练成本高，且实验未覆盖更广泛的数据与模型空间。

---

## 337. Beyond Behavior: Why AI Evaluation Needs a Cognitive Revolution

**arXiv ID:** 2604.05631 | [PDF](https://arxiv.org/pdf/2604.05631v1)

**作者:** Amir Konigsberg `[一作]` `[通讯]`, Amir Konigsberg

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文指出，图灵测试所奠定的以行为输出为唯一证据的评价框架限制了对AI内部过程和认知属性的科学探究，呼吁转向构造思维（construct‑thinking）评价模式，并强调需补充过程层面证据以实现更可靠的智能归因。

**💡 创新点**

创新点在于将心理学行为主义与AI评价体系进行结构性对照，提出“构造思维”作为新范式，明确区分性能与认知属性的归因，并系统阐述实现该转变所需的理论与方法。

**🔧 技术方法**

主要技术与方法包括：过程层面证据的获取（机制可解释性、扰动实验、迁移与发展评估）、构造效度框架、对比评测设计以及对现有基准的理论剖析。

**📊 数据集**

讨论的主要数据集与基准包括 MMLU、GSM8K、HumanEval、ARC‑AGI、GPQA 等通用与专业任务评测集合。

**📈 对比分析**

文中并未提供新的实验结果，而是以现有基准表现为例，指出虽然模型在 MMLU、GSM8K 等上可达 90%+ 的高分，但仅凭输出无法推断其是否具备真正的推理或理解能力。

**⚠️ 局限性**

局限性：论证以理论与文献回顾为主，缺乏实证验证；对“构造思维”范式的具体实现细节与评测标准尚未完全落地；以及对 AI 内部机制可观测性的假设尚未被广泛证实。

---

## 338. DetailVerifyBench: A Benchmark for Dense Hallucination Localization in Long Image Captions

**arXiv ID:** 2604.05623 | [PDF](https://arxiv.org/pdf/2604.05623v1)

**作者:** Xinran Wang `[一作]`, Zhanyu Ma `[通讯]` (Beijing University Of Posts And Telecommunications)

**通讯引用:** 8071 | [OpenAlex ID](https://openalex.org/A5039812471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了DetailVerifyBench，一个用于长图像字幕中细粒度幻觉定位的基准；

**💡 创新点**

在多领域（GUI、自然、图表、电影、海报）中提供超过200词长的字幕的 token 级标注，并提出了对抗性幻觉注入管道来生成更难检测的假错；

**🔧 技术方法**

利用多模态大型语言模型（Gemini‑3‑Pro/Flash、GPT‑5.2 等）进行生成、注释、注入以及对抗性 injector‑detector 交互，同时采用标签 <Hallucination> 进行 token 层级定位；

**📊 数据集**

使用 1,000 张高质量图像，来自 5 个视觉域，字幕由 Gemini‑3‑Pro 生成并由人工校正得到真实幻觉；同时通过对抗注入得到合成幻觉；

**📈 对比分析**

对 10+ 开源与闭源 MLLM 进行 token 级与句子级的精准/召回/F1 测评；在真实幻觉上最佳 token‑level F1 仅 0.15，合成幻觉上可达 0.6，说明合成评测能保持与真实的性能排序；

**⚠️ 局限性**

真实幻觉检测仍然极难，合成幻觉与真实模式存在差距，基准规模有限且仅涵盖 5 个域，注入方法可能遗漏某些幻觉类型。

---

## 339. Semantic-Topological Graph Reasoning for Language-Guided Pulmonary Screening

**arXiv ID:** 2604.05620 | [PDF](https://arxiv.org/pdf/2604.05620v1)

**作者:** Chenyu Xue `[一作]` (Xi'an Jiaotong-Liverpool University), Zhixiang Lu `[通讯]` (University of Liverpool)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种语言驱动的肺部分割框架，将大语言模型与视觉基础模型结合，实现从自由文本诊断指令到精准病灶分割。

**💡 创新点**

创新点包括：①文本到视觉意图蒸馏（TVID）将临床语言转化为视觉引导向量；②语义-拓扑图推理（STGR）通过构建候选掩码的动态图来消除解剖重叠带来的歧义；③选择性非对称微调（SAFT）仅更新不到1%的参数，兼顾性能与防止过拟合。

**🔧 技术方法**

采用 LLaMA‑3‑V 进行意图蒸馏、GroundingDINO+MedSAM 生成候选掩码、DINOv2 提取特征、图变压器实现 STGR、QLoRA+Adapter 进行 SAFT 微调。

**📊 数据集**

在 LIDC‑IDRI（肺结节）和 LNDb（肺部病灶）两个公开数据集上进行验证。

**📈 对比分析**

相较于传统 U‑Net、TransUNet 以及最新的 MedSAM、LISA 等基线，平均 Dice 分数提升至 81.5%（LIDC‑IDRI）和 74.6%（LNDb），并在 5‑折交叉验证中保持 ±0.6% 的低方差，显示出显著的精度与稳定性。

**⚠️ 局限性**

局限性包括依赖高质量的临床文本提示，若缺少详尽报告可能影响性能；目前仅处理二维胸片，未扩展到三维 CT 等体数据。

---

## 340. GraspSense: Physically Grounded Grasp and Grip Planning for a Dexterous Robotic Hand via Language-Guided Perception and Force Maps

**arXiv ID:** 2604.05697 | [PDF](https://arxiv.org/pdf/2604.05697v1)

**作者:** Elizaveta Semenyakina `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 2135 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个端到端的管线，能够基于物体的三维结构和材料属性生成力可接受的抓取点，并通过可变阻抗控制实现安全抓握。

**💡 创新点**

创新点在于：①引入空间分布的力可接受度（力图）作为抓取选择和抓握控制的核心依据；②在传统几何可行性筛选后进行基于力图的重新排序；③将每个手指的阻抗参数动态调节为局部可接受力，实现在不同物体区域的自适应抓握。

**🔧 技术方法**

使用的技术包括：自然语言处理 Qwen、YOLO-World 目标检测、SAM 语义分割、SAM3D 三维重建、Isaac Sim 物理仿真、DexGraspNet 生成抓取候选、TaskGrasp 任务相关过滤、基于 PCA 与射线投射的力图构建、可变阻抗控制。

**📊 数据集**

数据集：实验对象为纸杯、塑料杯和玻璃水杯的三维模型，未使用公开大规模抓取数据集，而是通过上述工具自行构造与重建。

**📈 对比分析**

方法对比：在相同的几何筛选条件下，传统基于几何评估的抓取策略与加入力图的重新排序进行对比。实验显示，力图方法将最小可接受力提升 8.6×–12.3×，零出现力超过上限的情况，抓取成功率提高，并能根据用户指令的力度级别自动调整抓握力度。

**⚠️ 局限性**

局限性包括：①目前仅在仿真环境中验证，尚未在真实机器人上部署；②仅针对杯形或薄壁物体验证，复杂几何形状和多材料复合体的通用性尚未评估；③依赖精确的三维重建和力图估计，对感知误差和模型不确定性敏感；④缺乏触觉反馈，实际抓握时可能无法及时检测到细微的结构损伤。

---

## 341. Improving Semantic Proximity in Information Retrieval through Cross-Lingual Alignment

**arXiv ID:** 2604.05684 | [PDF](https://arxiv.org/pdf/2604.05684v1)

**作者:** Seongtae Hong `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**通讯引用:** 2727 | [OpenAlex ID](https://openalex.org/A5033580486)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多语言检索模型在文档语言混杂场景下的跨语言对齐与偏差问题，并提出新的评估场景与Max@R指标，基于Jensen‑Shannon Divergence和InfoNCE的联合训练策略提升跨语言检索性能

**💡 创新点**

创新点在于①设计多参考跨语言检索场景和Max@R度量来捕捉模型在英语与目标语言共存时的对齐与语言偏差；②将JSD嵌入分布对齐与InfoNCE对比学习结合，实现对齐与检索效果双重提升

**🔧 技术方法**

采用Jensen‑Shannon Divergence（对齐）与InfoNCE（检索）相结合的双损失训练方案，基于概率分布软化的嵌入对齐方法

**📊 数据集**

使用XQuAD和Belebele两大并行多语言检索基准数据集，并在2.8k样本的MIRACL训练集上进行模型微调

**📈 对比分析**

与四种主流多语言嵌入模型（multilingual‑E5‑base、gte‑multilingual‑base、jina‑embeddings‑v3、bge‑M3）对比，评估指标包括Complete@K、Max@R、Max@R_norm、NDCG@1和MRR；实验显示新方法在所有语言上显著降低Max@R并提升检索准确率，语言偏差也被明显减小

**⚠️ 局限性**

局限性在于仍依赖大型语言模型生成的翻译数据，可能引入文化细微偏差；实验规模仅覆盖10种语言，未验证极低资源语言的泛化能力

---

## 342. Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs

**arXiv ID:** 2604.05643 | [PDF](https://arxiv.org/pdf/2604.05643v1)

**作者:** Hongyuan Yuan `[一作]` (Central South University), Haifeng Li `[通讯]` (Central South University)

**通讯引用:** 11085 | [OpenAlex ID](https://openalex.org/A5100398353)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于图的CoT优化框架，先将线性CoT转为有向无环图（DAG），通过分支级与深度级双重剪枝去除无效反思节点，并采用三阶段训练管线（SFT→DPO→GRPO）提升推理效率与准确率。

**💡 创新点**

创新点在于显式建模推理依赖，定义分支级与深度级冗余判定，首次将DPO与GRPO结合并加上长度惩罚实现精简推理；同时利用图剪枝生成高质量训练样本。

**🔧 技术方法**

使用LLM提示式图构造、分支级/深度级双重剪枝、监督微调、基于偏好学习的DPO、带长度惩罚的GRPO等技术。

**📊 数据集**

实验基于AIME24/25、AMC23、MATH500、OlympiadBench等数学推理数据，模型在DeepSeek-R1-Distill-Qwen-1.5B/7B上训练。

**📈 对比分析**

与O1-Pruner、TokenSkip、EfficientReasoning、AdaptThink等方法对比，平均准确率提升1-3%，推理长度下降42%（7B）或36%（1.5B），显著提升准确率与效率平衡。

**⚠️ 局限性**

局限性包括需强教师模型进行图构造、预处理成本高、进阶标签粗粒度可能忽略细粒度推理细节，并且对开放式任务的适用性尚未验证。

---

## 343. Analogical Reasoning as a Doctor: A Foundation Model for Gastrointestinal Endoscopy Diagnosis

**arXiv ID:** 2604.05649 | [PDF](https://arxiv.org/pdf/2604.05649v1)

**作者:** Peixi Peng `[一作]` (Shanghai Jiao Tong University), Guoyan Zheng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9144 | [OpenAlex ID](https://openalex.org/A5088035307)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种基于类比推理的医学基础模型RATNet，用于胃肠内镜图像的自动诊断。

**💡 创新点**

核心创新点包括：①利用循环预训练从五个异构注释的数据集提取并构建先验知识库；②设计RAT模块实现先验-后验知识对齐与迁移，实现类比推理；③支持多任务头、线性探测和零样本迁移，兼顾少样本、长尾、跨机构和联邦学习等多种临床场景；④实现无需人工标签统一即可集成异构注释，降低数据采集成本。

**🔧 技术方法**

技术细节：采用Swin-L大型视觉编码器；教师‑学生框架结合EMA；多任务head处理不同标签空间；RAT模块通过余弦相似度加权融合先验知识；后验知识由可学习模板生成；线性探测和零样本时使用投影头；联邦学习采用参数平均与EMA；训练使用SGD、数据增强（裁剪、旋转、亮度/对比度）等。

**📊 数据集**

预训练数据：CP‑CHILD、LIMUC、Kvasir、HyperKvasir、Daping共计≈39k张图像；少样本评估：Colonoscopic（Serrated、Hyperplastic、Adenoma）；零样本迁移：PolypGen、Shaoyifu；长尾测试：GastroVision；增量学习：Kvasir‑Capsule；与之对比的基线模型包括GastroNet、SSL、GastroHUN、GastroVision及多模态模型。

**📈 对比分析**

与多种基线模型的比较表明：在内部5个测试集AUC均>90%；在少样本任务中，1‑5 shot下RATNet median AUC比GastroNet高约10%且方差更小；零样本迁移中在PolypGen、Shaoyifu上AUC分别>92%和>89%，超过所有对照模型；在长尾GastroVision数据集上，线性探测AUC达97.1%，高于GastroNet 93.6%、GastroHUN 93.9%、GastroVision 93.1%；增量学习后RATNet+KC在Kvasir‑Capsule上AUC 99.0%，显著优于原始RATNet；联邦学习版本在跨机构评估中性能与中心化预训练相当，且优于单机构训练。

**⚠️ 局限性**

局限性：目前仅实现分类任务，尚未扩展至定位/分割与视频序列分析；对极少见疾病的性能仍受限于预训练集覆盖；模型规模较大，训练和推理成本较高；对不常见设备或扫描协议的泛化仍需进一步验证；类比推理机制的解释性和可调性尚未充分评估。

---

## 344. PanopticQuery: Unified Query-Time Reasoning for 4D Scenes

**arXiv ID:** 2604.05638 | [PDF](https://arxiv.org/pdf/2604.05638v1)

**作者:** Ruilin Tang `[一作]` (South China University of Technology), Shengfeng He `[通讯]` (Singapore Management University)

**通讯引用:** 6225 | [OpenAlex ID](https://openalex.org/A5056103024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PanopticQuery 框架，支持在动态 4D 场景中通过自然语言查询实现统一的时空语义推理。

**💡 创新点**

核心创新在于将几何重建与语义推理解耦，采用查询时动态推理 + 多视角语义共识，突破静态嵌入方法在动作、交互、空间关系上的局限。

**🔧 技术方法**

使用 4D 高斯投影（4D Gaussian Splatting）作为底层几何表示，配合大规模视觉语言模型（如 Qwen3‑VL）与 SAM‑wise 做 2D 语义预测，再通过几何投票与轻量级神经场优化实现 4D 语义上采样。

**📊 数据集**

引入新建的 Panoptic‑L4D 基准（21 个合成场景 + 4 实景，289 条多维度语言查询），并在 Neu3D 进行对比测试。

**📈 对比分析**

与 LangSplat 与 4D LangSplat 进行对比，PanopticQuery 在属性、动作、空间、交互四类查询的 mAcc 上分别提升 48.9%、59.9% 与 41.3%，总体保持或超过静态方法的性能，验证了查询时推理的有效性。

**⚠️ 局限性**

限制主要在长时序推理与内存管理方面：当前框架适用于短至中期场景，处理长时间序列时需要更高效的前馈结构与显存友好的记忆模块；单条查询耗时约 20 分钟，仍有改进空间。

---

## 345. SGANet: Semantic and Geometric Alignment for Multimodal Multi-view Anomaly Detection

**arXiv ID:** 2604.05632 | [PDF](https://arxiv.org/pdf/2604.05632v1)

**作者:** Letian Bai `[一作]` (Hong Kong University of Science and Technology), Juan Du `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 7849 | [OpenAlex ID](https://openalex.org/A5038073278)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SGANet框架，旨在通过多模态多视角的无监督异常检测，实现对工业表面缺陷的精确定位与识别。

**💡 创新点**

其核心创新在于将语义-结构对齐（SSPA）、选择性跨视角特征精炼（SCFRM）以及全局几何对齐（MVGA）三大模块耦合，利用语义一致性与几何一致性共同学习跨视角、跨模态的物理一致特征表示。

**🔧 技术方法**

技术上结合了预训练的DINO‑v2特征提取器、基于多模态相似度的注意力聚合、InfoNCE对齐损失以及几何投影对应的欧氏对齐损失，形成统一的特征学习与异常分数计算流程。

**📊 数据集**

实验使用了SiM3D（真实多视角多模态工业缺陷数据集）和Eyecandies（合成多视角RGB+深度数据集）两大基准。

**📈 对比分析**

与PatchCore、AST、MVAD、CFM、M3DM、BTF等单模态、单视角以及现有多视角/多模态方法相比，SGANet在I‑AUROC、V‑AUPRO@1%、P‑AUROC和AUPRO@30%等指标上均实现显著提升，平均I‑AUROC提升至0.887（SiM3D）/0.743（Eyecandies）。

**⚠️ 局限性**

局限性包括对摄像机标定与投影准确性的高度依赖，在严重遮挡或非标定场景下性能可能下降；深度模态单独使用时表现仍不如RGB，且虽然对极少样本有一定鲁棒性，但对超低资源环境的适用性仍待验证。

---

## 346. Classes Testable with $O(1/ε)$ Queries for Small $ε$ Independent of the Number of Variables

**arXiv ID:** 2604.05615 | [PDF](https://arxiv.org/pdf/2604.05615v1)

**作者:** Nader H. Bshouty `[一作]` (Technion), George Haddad `[通讯]` (Technion)

**通讯引用:** 4051 | [OpenAlex ID](https://openalex.org/A5111605414)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了若干布尔函数类（k‑Junta、Fourier 低阶、s‑稀疏多项式及其低阶版本）在均匀分布下的属性检验问题，并给出了与参数 ψ（如 k、d、s）相关、与变量总数 n 无关的查询复杂度。实现了在 ε ≤ 1/ψ 时的 O(1/ε) 上界，匹配已知的下界。

**💡 创新点**

1) 证明了上述所有类均可在 O(ψ + 1/ε) 查询内检验；2) 将学习可解的类与检验可解性关联，给出了从精确学习到检验的通用构造；3) 引入了基于 pairwise‑independent 赋值的简化检验，显著降低查询量；4) 通过相关坐标/块验证与自纠正器构建了通用的检验框架。

**🔧 技术方法**

核心技术包括：随机限制与变量分块、对称性（变量置换、零一投影）下的归约、pairwise‑independent 随机赋值、二分搜索定位相关变量、Chernoff/ Chebyshev 分析、Self‑Corrector、相关坐标/块可验证、学习→检验的归约。通过这些技术实现了从 O(ψ + 1/ε) 到 O(1/ε) 的转化。

**📊 数据集**

本工作为纯理论分析，未使用具体数据集；所有结果均基于概率分析与量化复杂度推导。

**📈 对比分析**

与已有工作比较：
- k‑Junta：从 O(k²/ε) 改为 O(k log k + k/ε)；
- Fourier 低阶：从 O(2²d/ε²) 降至 O(2²d)+O(1/ε)；
- s‑稀疏多项式（低阶）：从 Õ(s/ε + s 2ᵈ) 降至 Õ(2ᵈs)+O(1/ε)；
- s‑稀疏多项式：在 ε < 1/s⁸.⁴²² 时实现 O(1/ε) 查询；
整体实现了许多类在均匀分布下的最佳或接近最佳查询复杂度。

**⚠️ 局限性**

限制：
- 仅适用于在变量置换和零一投影下封闭的类；
- 仍有指数级依赖于 k 的构造（如 k‑Junta 的 2ᵏ 项）；
- 需要已知可学习的学习算法；
- 主要针对均匀分布，分布无关扩展仍需进一步研究；
- 在非常小的 ε 或大参数 k、d、s 时，常数因子和高阶项仍可能导致实际查询量较大。

---

## 347. The Incidence-Multiplicity Bound for Linear Exact Repair in MDS Array Codes

**arXiv ID:** 2604.05692 | [PDF](https://arxiv.org/pdf/2604.05692v1)

**作者:** Huawei Wu `[一作]` `[通讯]`, Huawei Wu

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在固定域大小、冗余度和子包尺寸下MDS数组码的线性精确修复问题，提出了更精确的下界并构造了在广泛参数范围内达到该下界的码

**💡 创新点**

引入了“incidence‑multiplicity bound”，比先前的投影计数下界更严格，并给出了基于域降维的正规有理曲线构造，实现了该下界的多参数实现

**🔧 技术方法**

利用子空间组合的内在表述、投影计数、双射与对偶空间中的点与子空间的相交计数，以及正则有理曲线的域降维技术

**📊 数据集**

无，本文为理论构造与证明，无需实验数据集

**📈 对比分析**

通过与已知的投影计数下界对比，证明在r≥3时新下界更强；在构造的码上，平均和最差修复带宽及I/O均等于该下界，显示了最优性

**⚠️ 局限性**

构造仅适用于满足(r-1)|(q-1)且长度满足2(r-1)(q^ℓ−1)/(q−1)≤n≤q^ℓ+1的情况；对短长度或非可整除情况缺乏完整描述

---

## 348. 3D Smoke Scene Reconstruction Guided by Vision Priors from Multimodal Large Language Models

**arXiv ID:** 2604.05687 | [PDF](https://arxiv.org/pdf/2604.05687v1)

**作者:** Xinye Zheng `[一作]` (Hefei University of Technology), Zhiliang Wu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的3D重建方法Smoke-GS，旨在从烟雾降解的多视图图像中重建3D场景，并合成新视图。

**💡 创新点**

创新点在于结合了视觉增强技术Nano-Banana-Pro和中介感知的3D高斯点云框架，以提高在烟雾环境下的重建效果和新视图合成的质量。

**🔧 技术方法**

使用了Nano-Banana-Pro进行图像增强，并采用了中介感知的3D高斯点云技术进行场景建模。

**📊 数据集**

使用了RealX3D数据集中的烟雾散射子集，该数据集包含多视图训练图像及其对应的相机姿态。

**📈 对比分析**

与其他方法相比，Smoke-GS在NTIRE 2026 3D恢复与重建挑战中获得了第二名，PSNR为18.6681，SSIM为0.6909，显示出更强的重建保真度和结构保留能力。

**⚠️ 局限性**

限制在于该方法可能在极端烟雾条件下的表现仍需进一步验证，且训练过程依赖于大量的计算资源。

---

## 349. Improved Space-Time Tradeoffs for Permutation Problems via Extremal Combinatorics

**arXiv ID:** 2604.05661 | [PDF](https://arxiv.org/pdf/2604.05661v1)

**作者:** Afrouz Jabal Ameli `[一作]` (Utrecht University), Shengzhe Wang `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种新的链效率参数，用于描述集合系统中的最大链数与集合规模的关系，并基于此构造高效的集合系统，从而实现了在幺半环上对排列问题的改进空间-时间折衷，具体到旅行商问题，得到 S·T ≤ 3.7493^N 的算法。

**💡 创新点**

创新点包括：①引入链效率概念并证明其对排列问题的空间-时间复杂度影响；②利用概率方法和组合构造，构造出比先前桶序（bucket order）更高效的双部图格子（bipartite poset），从而破坏了 Johnson‑Leader‑Russell 先前的极值猜想；③证明了在此框架下无法实现 S·T < 3.015^N 的上界。

**🔧 技术方法**

核心技术包括：动态规划（子集 DP）与概率覆盖（构造覆盖所有排列的有限置换族）、链效率与最大链数的组合计数、熵与 Hamming 球的等距极值定理、以及图论中的 Hamming 球同构与正则双部格子等效性。

**📊 数据集**

本文没有使用传统意义上的实验数据集；其“数据”主要来自对特定格子（例如 6‑regular 双部格子、7‑regular 双部格子）理想数和线性扩展数的计算，采用自研程序验证其数值，并与已知理论极限进行比较。

**📈 对比分析**

与之前的 Koivisto‑Parviainen 方案（S·T ≤ 3.9271^N）相比，本文的算法将乘积常数从 3.9271 降至 3.7493，改进幅度约为 7%；在更严格的空间约束下，实验表明该算法在理论上可实现更低的空间占用而维持可接受的时间复杂度。

**⚠️ 局限性**

主要局限包括：①仍无法突破 3.015^N 的下限，无法进一步逼近理论最优（Open Problem (b)）；②链效率框架的上界表明，即便采用更优秀的集合系统，空间-时间乘积仍受限于 3.015^N；③目前仅在幺半环上适用，扩展到更一般的半环或非幺结构仍是挑战。

---

## 350. Optimal-Transport-Guided Functional Flow Matching for Turbulent Field Generation in Hilbert Space

**arXiv ID:** 2604.05700 | [PDF](https://arxiv.org/pdf/2604.05700v1)

**作者:** Li Kunpeng `[一作]` (Nanyang Technological University), Ong Yew Soon `[通讯]` (Nanyang Technological University)

**通讯引用:** 27424 | [OpenAlex ID](https://openalex.org/A5068243197)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在无限维函数空间中提出一种新的流匹配生成框架——Functional Optimal Transport Conditional Flow Matching (FOT‑CFM)，用于快速、精确地合成多尺度湍流数据。

**💡 创新点**

创新点包括：① 将条件流匹配从有限维欧氏空间推广到可分Hilbert空间，并证明条件到边缘的一致性；② 通过最优传输构造直线概率路径，显著降低生成轨迹曲率；③ 采用神经算子（FNO）参数化向量场，实现尺度无关生成并支持零步超分辨率；④ 训练和采样均为无模拟（simulation‑free），极大提升采样效率。

**🔧 技术方法**

核心技术：条件流匹配（Conditional Flow Matching）、最优传输（Optimal Transport）、傅里叶神经算子（FNO）、mini‑batch OT 赋值、无模拟训练目标、ODE 采样（Euler/RK4）。

**📊 数据集**

使用的实验数据集包括：2D Kolmogorov 流、Navier‑Stokes 湍流（64×64 轨迹）、Hasegawa‑Wakatani TOKAM2D（128×128 但训练 64×64）。

**📈 对比分析**

方法与 DDPM、FFM、DDO、GANO 等基线在谱一致性（R²、RMSE）、密度一致性（KDE R²、RMSE）和 NFE（函数评估次数）上进行对比。FOT‑CFM 在低 NFE（5–20）下实现最佳或次佳谱和统计指标，并在高 NFE 下保持竞争力；同时大幅减少 ODE 步数，显著提升采样速度。

**⚠️ 局限性**

局限性：仅在二维问题上验证；三维湍流、复杂几何或更高 Reynolds 仍待探索；依赖神经算子训练，模型规模和内存需求较高；OT 仅在 mini‑batch 内求解，批量大小会影响近似质量；物理约束（守恒、边界条件）需进一步强化。

---

## 351. CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration

**arXiv ID:** 2604.05689 | [PDF](https://arxiv.org/pdf/2604.05689v1)

**作者:** Xuecong Liu `[一作]` (Northeastern University), Xichao Teng `[通讯]` (National University of Defense Technology)

**通讯引用:** 357 | [OpenAlex ID](https://openalex.org/A5057327033)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于Transformer的跨模态图像配准框架CRFT，能够在光学-雷达和光学-红外等模态之间实现高精度的粗到细自适应配准。

**💡 创新点**

创新点在于将模态无关的特征流学习与粗到细的分层匹配结合，并引入基于特征差异的注意机制和空间几何变换（SGT）的递归优化，显著提升了跨模态下的几何一致性和亚像素精度。

**🔧 技术方法**

采用多尺度CNN特征提取、Transformer自/交叉注意、全局相关性建模、SGT对齐、迭代差异引导注意、残差更新和置信度平滑等技术；整体实现为端到端的Transformer+回归框架。

**📊 数据集**

在公开的OSdataset（光学-雷达）和RoadScene（光学-红外）两个跨模态配准基准上进行实验。

**📈 对比分析**

与10种手工特征、稀疏匹配、半稠密和稠密光流基线相比，CRFT在AEPE和CMR指标上均取得显著提升，例如OSdataset下AEPE仅0.65px、CMR超过95%；RoadScene下AEPE 2.37px、CMR3px 68%。

**⚠️ 局限性**

局限性包括：对极端非线性几何变形（如大视角差异）仍有挑战；训练需要较大的GPU资源和标注配准数据；模型参数相对较多，推理速度虽快但仍不适合极低延迟场景。

---

## 352. Understanding: reframing automation and assurance

**arXiv ID:** 2604.05662 | [PDF](https://arxiv.org/pdf/2604.05662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 353. Foundations of Future Communication Systems: Innovations in Communication - A Report

**arXiv ID:** 2604.05694 | [PDF](https://arxiv.org/pdf/2604.05694v1)

**作者:** Christian Deppe `[一作]` (Technical University of Braunschweig), Marcel A. Mross `[通讯]` (Technical University of Braunschweig)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文总结了未来通信系统基础会议（FFCS 2026）的研究成果，涵盖信息理论、量子通信、分子通信、语义通信和安全网络设计等领域的前沿研究。

**💡 创新点**

创新点在于重新审视在现实物理、架构和安全约束下的基础极限，强调可靠性、识别、语义、资源效率和信任等方面的研究。

**🔧 技术方法**

使用了跨学科的研究方法，结合经典香农理论、后香农范式、量子信息科学和新兴的物理基础通信模型。

**📊 数据集**

论文中提到的研究成果来自于FFCS会议的海报展示和邀请演讲，涵盖了多种数据集和实验结果。

**📈 对比分析**

通过比较不同的研究方法，展示了在复杂异构网络中，如何在安全性和效率之间取得平衡，许多研究超越了传统的速率中心视角。

**⚠️ 局限性**

限制在于许多研究仍处于早期阶段，缺乏大规模实验验证，且在实际应用中可能面临技术和理论的挑战。

---

## 354. Dynamic Control Allocation for Dual-Tilt UAV Platforms

**arXiv ID:** 2604.05677 | [PDF](https://arxiv.org/pdf/2604.05677v1)

**作者:** Marcello Sorge `[一作]` (University of Padova), Angelo Cenedese `[通讯]` (University of Padova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种层次化的动态控制分配方法，用于双倾斜六旋翼无人机（dual‑tilt hexarotor）实现轨迹跟踪。

**💡 创新点**

创新点在于①显式考虑执行器饱和效应并给出理论分析；②利用不对称目标函数对不同倾角（α、β）进行差异化优化；③通过投影到雅可比矩阵零空间的梯度下降分配，实现执行器状态在冗余空间内优化，同时保持高层控制器命令的闭环性能。

**🔧 技术方法**

技术包括：①一阶非线性执行器动力学建模；②基于伪逆和零空间投影的梯度分配法；③自适应的饱和矩阵（∇sat）与正则化参数；④高层PID式姿态与位置控制器。

**📊 数据集**

采用数值仿真数据：星形双倾斜六旋翼模型（质量 2 kg，惯量 diag(0.0217,0.0217,0.04) kg·m²），六个旋翼臂长 0.246 m，倾角可变，转速范围 100–1000 rad/s，跟踪圆形轨迹半径 2 m，角频 0.8 s⁻¹。

**📈 对比分析**

通过对比非优化（γ_j=0）和优化（γ_j=10）两种情形，评估了跟踪误差、倾角振幅/偏移、转速波动以及目标函数值。结果显示：优化后倾角振幅趋于零，转速振幅明显下降，目标函数显著降低；跟踪误差差别极小，说明控制精度保持不变。

**⚠️ 局限性**

局限性包括：①在轨迹过快或高层控制器过激时，执行器可能在瞬态期间饱和，导致误差发散；②高层控制器与分配器独立设计，缺乏像增益调节的抗饱和机制（anti‑windup）；③实验仅限仿真，未验证实际硬件上的鲁棒性。

---

## 355. Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation

**arXiv ID:** 2604.05673 | [PDF](https://arxiv.org/pdf/2604.05673v1)

**作者:** Wuyang Luan `[一作]` (Jilin University), Rui Ma `[通讯]` (Jilin University)

**通讯引用:** 11317 | [OpenAlex ID](https://openalex.org/A5100710180)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Rectified Schrödinger Bridge Matching (RSBM)框架，通过单一熵正则化参数ε统一 Schrödinger Bridges 与 Flow Matching，实现在视觉导航任务中仅需3个ODE步骤即可生成高质量动作轨迹。

**💡 创新点**

创新点在于证明条件速度场在整个ε谱上形式不变，实现速度方差线性降低，从而实现路径平滑与多模态兼顾；并通过学习的条件先验缩短传输距离，完成一次性训练即可在任意步数下推断。

**🔧 技术方法**

使用高效的双流Vision Encoder + Transformer 提取上下文；变分先验网络生成粗略轨迹；条件U‑Net速度网络学习ε-Rectified Schrödinger Bridge；Heun二阶ODE求解器；训练采用无模拟的条件流匹配损失。

**📊 数据集**

在多任务环境下评估：HuRoN、Recon、SACSoN、SCAND、GoStanford（公开数据集），以及自建的Custom Indoor与CitySim（室内外 Gazebo 仿真）。

**📈 对比分析**

与ViNT、NoMaD、DDPM、Conditional Flow Matching、NaviBridger 等基线比较，RSBM在k=3（NFE=5）时获得94.5%余弦相似度、92%成功率，性能接近或优于NaviBridger在k=10（NFE=19）且函数评估数降低3.8×，在多数据集上保持一致的优势。

**⚠️ 局限性**

局限性包括：实验主要基于离线开放式评估，实时闭环导航与动态障碍场景测试有限；学习先验限制了零样本迁移；目前未验证在更复杂多模态环境或长距离任务中的泛化能力。

---

## 356. LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals

**arXiv ID:** 2604.05655 | [PDF](https://arxiv.org/pdf/2604.05655v1)

**作者:** Lihao Sun `[一作]` (Microsoft), Saravan Rajmohan `[通讯]` (Microsoft)

**通讯引用:** 612 | [OpenAlex ID](https://openalex.org/A5070722259)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并可视化大型语言模型在链式推理过程中在表示空间中的轨迹，发现不同推理步骤对应线性可分的子空间，并通过后期轨迹差异预测答案正确性，从而提出基于轨迹的推理纠错与长度控制方法。

**💡 创新点**

创新点在于将推理过程建模为高维表示空间中的几何轨迹，揭示步骤特定子空间随层深逐步解缠、正确与错误在后期轨迹明显分离，进而实现中途预测与动态推理干预、长度控制的统一框架。

**🔧 技术方法**

使用线性探针、t‑SNE、欧氏/余弦距离、PCA、低秩激活调度、推理时间调节、ROC‑AUC 等技术对激活轨迹进行定量分析与干预。

**📊 数据集**

主要使用 GSM8K、MATH‑500 和 MMLU 三个数学推理数据集，测试不同提示格式与模型训练阶段（Base、Instruct、R1‑Distill）。

**📈 对比分析**

与基线模型相比，后期轨迹特征的 ROC‑AUC 可达 0.87；错误预测驱动的推理干预在 12% 的样本上提升约 +35% 准确率；长度控制在 |α|≤0.4 的范围内可实现 ±30% 的推理步数变化，准确率仅波动 1%。

**⚠️ 局限性**

局限性包括：仅验证了数学推理任务；实验仅在 Llama 3.1 8B 系列模型上进行，未检验更大或不同架构模型；干预依赖于从正确样本提取的理想轨迹，可能不适用于正确性不明确的任务；在更复杂的开放式推理或程序合成场景下是否同样有效仍未知。

---

## 357. Improved space-time tradeoff for TSP via extremal set systems

**arXiv ID:** 2604.05645 | [PDF](https://arxiv.org/pdf/2604.05645v1)

**作者:** Justin Dallant `[一作]` (TU Dresden), László Kozma `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种新的空间–时间权衡框架，并用该框架改进了旅行推销员问题（TSP）以及更一般的排列问题（Permutation Problems）的指数时间/空间复杂度。

**💡 创新点**

创新点在于：
- 将 DP 与稀疏集合系统（hypergraph）相结合，利用集合系统中极大链的最大化问题作为核心极值问题；
- 构造了新的稀疏集合系统，使得 ST < 3.572，显著低于之前的 3.93；
- 通过对 Johnson–Leader–Russell 猜想的反证，说明之前的“塔式立方体”构造并不是最优的；
- 证明该框架对所有加法幺半环（additively idempotent）均可扩展，得到 ST < 3.864 的泛化结果。

**🔧 技术方法**

主要技术包括：
- Bellman–Held–Karp 动态规划与 Gurevich–Shelah 分治的结合；
- 通过“集合系统的最大链密度”来刻画可支持的排列数；
- 概率方法与离散化构造稀疏集合系统；
- Fekete 子加性引理与插值技术实现参数优化；
- 结合组合极值与半环理论得到更广泛的排列问题结果。

**📊 数据集**

本文为理论研究，不涉及实验数据集；所有结果均通过理论证明得出。

**📈 对比分析**

与以往方法比较：
- 传统点 (2,2)、(1,4) 等 ST=4 的基准被完全压制；
- 对比 Koivisto–Parviainen 在 2 < T < 2√2 区间内的 ST≈3.93，本文得到更小的 ST≈3.572；
- 在更宽范围内（2 < T < 4）得到全曲线优于 ST=4；
- 对于更一般的排列问题（在加法幺半环上）得到 ST<3.864 的新上界。

**⚠️ 局限性**

限制与未解问题：
- 对非幺半环（non-idempotent）仅能得到 ST<3.864，无法进一步降低；
- 仍存在 ST≥3 的下界，理论上与本框架相关的上界尚未完全收敛；
- 构造与实现细节中多项式因子未给出精确大小；
- 需要进一步研究极大链密度最优集合系统的完整结构与更紧的下界。

---

## 358. A Unified Foundation Model for All-in-One Multi-Modal Remote Sensing Image Restoration and Fusion with Language Prompting

**arXiv ID:** 2604.05629 | [PDF](https://arxiv.org/pdf/2604.05629v1)

**作者:** Yongchuan Cui `[一作]` (Chinese Academy of Sciences), Peng Liu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 22070 | [OpenAlex ID](https://openalex.org/A5021833788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了语言条件的多任务遥感图像低层视觉恢复模型LLaRS，统一处理多种降解并支持自然语言提示；

**💡 创新点**

创新点在于利用Sinkhorn-Knopp最优传输实现异构波段的语义匹配，构建三种互补的混合专家模块（卷积、通道混合、低秩注意力），并通过动态权重调整实现多任务联合训练；

**🔧 技术方法**

采用了最优传输（OT）、混合专家（MoE）、低秩LoRA、动态权重调整（DWA）以及语言编码等技术；

**📊 数据集**

使用自研的百万级LLaRS1M数据集，涵盖11种真实与合成的遥感恢复任务，并配备丰富的自然语言提示；

**📈 对比分析**

在统一训练协议下与七个强基线模型（包括MPRNet、PromptIR等）进行对比，平均PSNR、SSIM、SAM、ERGAS均取得最高成绩；参数高效微调亦能在未见任务上实现与全微调相近的性能；

**⚠️ 局限性**

仍存在对极端降解（如强雾、色彩失真）恢复效果有限、对合成降解的依赖、模型规模大导致推理成本高等局限。

---

## 359. FunRec: Reconstructing Functional 3D Scenes from Egocentric Interaction Videos

**arXiv ID:** 2604.05621 | [PDF](https://arxiv.org/pdf/2604.05621v1)

**作者:** Alexandros Delitzas `[一作]` (ETH Zurich), Daniel Barath `[通讯]` (ETH Zurich)

**通讯引用:** 1296 | [OpenAlex ID](https://openalex.org/A5016636021)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

从单段前视RGB-D交互视频中，端到端地重建可交互的三维数字孪生，包括静态几何、可动部件及其运动参数。

**💡 创新点**

提出训练自由、基于优化的管线，融合基础模型（视频语言模型、SAM2等）的语义与运动先验，自动识别并聚类关节、估计关节参数与部件姿态，并在全局空间对所有片段进行对齐，首次实现真实场景的完整功能重建。

**🔧 技术方法**

核心技术包括：视频片段分类与关节类型预测（VLM），稀疏轨迹跟踪与点对点匹配（RoMA、TAPIP3D、SupeRANSAC），运动聚类与轨迹拟合（HDBSCAN、线/圆拟合），像素级分割引导（SAM、SAM2），部件姿态与关节参数的联合优化（Ceres），TSDF体素融合与网格提取，以及全局片段配准（PREDATOR）。

**📊 数据集**

评估使用三大数据集：HOI4D（公开单物体交互），RealFun4D（351条真实屋内交互视频），OmniFun4D（127条合成交互序列）。

**📈 对比分析**

与MonST3R、SpatialTrackerV2、BundleSDF、ArtGS等基线对比，实验表明：关节方向误差≤5.3°、位置误差≤0.03m；mIoU≈77%，比基线提升20+点；ADD‑S/ADD≈79%/71%，比BundleSDF提升约2倍；Chamfer距离≤0.7cm，显著优于其他方法；且失败率为0%。

**⚠️ 局限性**

局限性：仅支持标准直线或旋转关节，无法处理复杂多自由度或非标准机械结构；对深度噪声和强遮挡敏感；对高速交互或极端视角的鲁棒性有待提升；需要单一交互序列，无法直接处理多场景连续交互。

---

## 360. Grounding Hierarchical Vision-Language-Action Models Through Explicit Language-Action Alignment

**arXiv ID:** 2604.05614 | [PDF](https://arxiv.org/pdf/2604.05614v1)

**作者:** Theodor Wulff `[一作]` (University of Manchester), Angelo Cangelosi `[通讯]` (University of Manchester)

**通讯引用:** 10170 | [OpenAlex ID](https://openalex.org/A5091768977)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GPLA（Grounded Preference-based Language‑Action Alignment）框架，通过对齐模型评估语言与动作的对应关系，并使用偏好学习优化层级 Vision‑Language‑Action（VLA）模型的中间步骤。

**💡 创新点**

创新点在于：1）利用对齐评分模型生成偏好对，消除对中间语言输出的人工标注需求；2）在训练中将视觉、动作与文本映射到同一嵌入空间，实现跨模态对齐；3）采用 SimPO 等偏好学习方法直接提升模型透明度与语义一致性。

**🔧 技术方法**

技术手段包括：对齐模型（基于 CLIP/SigLIP 2 + FiLM 条件的动作编码器）进行对比学习；Gemma‑3 VLM 负责高层指令拆分；SmolVLA 负责低层动作生成；SimPO 偏好学习；数据增强与 InfoNCE 目标函数。

**📊 数据集**

使用 LanguageTable 基准数据集（含高层任务、低层文本指令与对应的 2D 轨迹）。

**📈 对比分析**

与完全监督微调对比，GPLA 在轨迹误差（MAE/MSE）和语义一致性（BLEU/ROUGE/METEOR/BERTScore）上接近甚至略优；在词汇重叠指标上略低，但生成指令在语义上保持一致；在低数据场景下表现尤为突出。

**⚠️ 局限性**

局限性包括：1）受限于数据量小，模型对物体关系、颜色和空间关系的理解仍不完整；2）偏好学习生成的指令偶尔出现物理不合理或语义噪声；3）对不同机器人平台的泛化能力待进一步验证。

---

## 361. Same Graph, Different Likelihoods: Calibration of Autoregressive Graph Generators via Permutation-Equivalent Encodings

**arXiv ID:** 2604.05613 | [PDF](https://arxiv.org/pdf/2604.05613v1)

**作者:** Laurits Fredsgaard `[一作]` (Technical University Of Denmark), Mahito Sugiyama `[通讯]` (National Institute Of Informatics)

**通讯引用:** 603 | [OpenAlex ID](https://openalex.org/A5066053285)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了自回归图生成模型在不同线性化顺序下的似然一致性问题，提出了线性化不确定性（LU）指标来衡量模型对等价线性化的负对数似然偏差。

**💡 创新点**

创新点在于：①正式定义线性化不一致性要求并引入LU；②通过SENT线性化方法对四种遍历策略进行对比，揭示了训练线性化偏差导致的校准误差；③证明LU能更可靠地评估生成分子的质量。

**🔧 技术方法**

使用的技术包括SENT编码、Transformer（12层Llama）自回归训练、基于教师强迫的优化、随机、最小度、最大度和锚点展开四种线性化策略，以及ECE、AUC评估等。

**📊 数据集**

实验数据集包括Planar（128个训练图）和QM9（约98k训练分子）两种规模，后者进一步细分为数据稀缺子集和完整数据集。

**📈 对比分析**

比较方法：对同一模型在其原始线性化策略和随机线性化策略下进行交叉评估，记录NLL/token、LU、ECE。结果显示，偏置策略在原始线性化下取得更低的NLL和LU，但在随机线性化下ECE提升两倍，说明其对图结构的学习不稳健；LU在预测分子稳定性方面AUC高达0.85，显著优于单纯的生成NLL（AUC 0.43）。

**⚠️ 局限性**

局限性：实验仅限于SENT框架和QM9小分子数据，未验证在更大、化学多样性更高的基准（如MOSES、GuacaMol）上的泛化；此外LU收敛所需的K值在更大图上是否相同仍是未知。

---

## 362. Evaluation of Randomization through Style Transfer for Enhanced Domain Generalization

**arXiv ID:** 2604.05616 | [PDF](https://arxiv.org/pdf/2604.05616v1)

**作者:** Dustin Eisenhardt `[一作]` (German Cancer Research Center (DKFZ)), Gemma Roig `[通讯]` (Goethe University Frankfurt)

**通讯引用:** 2347 | [OpenAlex ID](https://openalex.org/A5025034643)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于风格迁移的轻量级数据增强方法——StyleMixDG，用以提升域泛化性能

**💡 创新点**

通过系统实验揭示：扩大风格池对泛化效果影响最大；纹理复杂度过滤无显著提升；多样化艺术风格优于域内/域外风格，并将这些原则整合为无架构修改、无额外损失的简单方案

**🔧 技术方法**

采用AdaIN风格迁移、离线多风格增强、在线光度失真以及80:20的风格/原始图像混合采样

**📊 数据集**

在GTAV为源域，BDD100k、Cityscapes、Mapillary Vistas为目标域的语义分割基准上评估

**📈 对比分析**

与无增强基线、单风格、光度失真、多风格重复及现有领域泛化方法（GTR、SAN‑SAW、WildNet、DIDEX）比较，StyleMixDG在ResNet‑50/ResNet‑101、UperNet+DeiT‑S16等模型上实现平均mIoU≈41.4（ResNet‑101）或40.0（DeiT‑S16），超过大多数无改造方法并接近更复杂方法

**⚠️ 局限性**

对小目标类别效果有限；仅在单一源域（GTAV）与语义分割任务上验证；风格迁移可能导致纹理敏感类别识别下降

---

## 363. Efficient Construction of Reachability Graphs for Petri Net Product Lines

**arXiv ID:** 2604.05657 | [PDF](https://arxiv.org/pdf/2604.05657v1)

**作者:** Elena Gómez-Martínez `[一作]` (Universidad Complutense de Madrid), José Ignacio Requeno Jarabo `[通讯]` (Universidad Complutense de Madrid)

**通讯引用:** 192 | [OpenAlex ID](https://openalex.org/A5080044871)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套面向Petri网产品线（PNPL）的可达图构造算法，能够在不枚举所有产品的情况下生成覆盖所有变体的参数化可达图；

**💡 创新点**

创新点在于设计了符号状态编码（标记与特性约束相结合）、家族保持的后继生成过程以及基于特性约束的冲突剪枝与状态合并技术；

**🔧 技术方法**

技术手段包括符号状态表示、基于特性约束的后继生成、状态空间压缩（状态合并与选择性抽象）、SAT求解（Sat4j）与约束规划（JaCoP），并在Eclipse/EMF框架下实现；

**📊 数据集**

使用了若干基准PNPL模型（如简化装配线、售货机、医院案例等）进行实验验证；

**📈 对比分析**

与传统逐产品枚举相比，所提出方法在内存和时间消耗上显著降低，实验结果表明可达图构造速度提升数倍、内存占用下降几十个百分点；

**⚠️ 局限性**

局限性在于目前对大规模特性空间的处理仍有挑战，且对时序PNPL的支持尚不成熟，未来需要进一步提升可扩展性与与其他分析工具的集成度。

---

## 364. See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs

**arXiv ID:** 2604.05650 | [PDF](https://arxiv.org/pdf/2604.05650v1)

**作者:** Yicheng Ji `[一作]` (Zhejiang University), Huan Li `[通讯]` (Zhejiang University)

**通讯引用:** 16585 | [OpenAlex ID](https://openalex.org/A5100319241)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练自由的宽松式投机解码框架，用视觉语义引导严格验证视觉相关词并放宽非视觉词的匹配规则，以提升视频大语言模型的推理速度。

**💡 创新点**

创新点在于识别生成序列中稀疏的视觉锚点与冗余词，并利用视觉相似度动态调节验证严格度，同时引入位置移位容忍机制，打破传统逐词完全匹配的瓶颈。

**🔧 技术方法**

核心技术包括：视觉-文本跨模态余弦相似度计算、基于Top‑N视觉词的相关性评分、松散验证策略以及位置移位容忍补丁；实现完全无训练，仅使用目标模型和轻量草稿模型。

**📊 数据集**

在四大视频理解基准上进行评估：Video Detail Caption、Video Detail Description、MovieChat 与 Video‑MME。

**📈 对比分析**

与 Naive SD、SpecVLM、FLy 等基线对比，方法在保持 99.8% 以上性能保留的前提下，速度提升 2.70×–2.94×，平均接受长度显著增大，且在多任务上均实现领先的加速效果。

**⚠️ 局限性**

局限包括：需手动调节 λ 与 N 等超参数以平衡速度与质量；主要针对描述性任务，对复杂逻辑推理任务的匹配策略仍需改进；位置移位容忍机制适用场景有限，且整体方案未与专门训练的草稿模型结合。

---

## 365. Leaderless Collective Motion in Affine Formation Control over the Complex Plane

**arXiv ID:** 2604.05648 | [PDF](https://arxiv.org/pdf/2604.05648v1)

**作者:** Jesus Bautista `[一作]` (University of Granada), Hector Garcia de Marina `[通讯]` (University of Granada)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种无领导者、分布式的平面仿射形态操纵控制方法，通过改写拉普拉斯矩阵权重实现多机器人团队在保持形态不变的同时完成平移、旋转、缩放与剪切等集体运动；

**💡 创新点**

创新点主要包括①利用复数表示二维仿射形态以简化分析；②在拉普拉斯矩阵中加入运动参数实现无领导者集体运动；③推导闭环系统的完整解析解与特征结构；④在保证形态不变的前提下证明收敛性与指数稳定性；

**🔧 技术方法**

使用图拉普拉斯矩阵、复数域表述、特征值与特征向量分析、Jordan 标准形、Lyapunov 稳定性证明、单积分器动力学以及分布式控制律设计；

**📊 数据集**

仅在仿真中使用人工构造的几何形态（如正方形、圆形等）进行实验，无外部真实数据集；

**📈 对比分析**

通过与欧拉数值积分结果对比验证解析解的准确性；仿真显示系统指数收敛至目标形态并实现预定集体运动；对比未改写权重的原始静态形态控制，演示了更丰富的运动能力；

**⚠️ 局限性**

局限性包括：需要全局已知参考形态与运动参数；对图的全局刚性与无向性有要求；仅适用于单积分器动力学，未考虑非线性或约束；在通信失败、测量噪声或动态拓扑变化时鲁棒性尚未验证；

---

## 366. T2T: Captioning Smartphone Activities Using Mobile Traffic

**arXiv ID:** 2604.05642 | [PDF](https://arxiv.org/pdf/2604.05642v1)

**作者:** Jiyu Liu `[一作]` (Zhengzhou University), Wanqing Tu `[通讯]` (Durham University)

**通讯引用:** 685 | [OpenAlex ID](https://openalex.org/A5023539073)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了如何从加密的手机网络流量生成用户操作的自然语言描述，提出了T2T系统。

**💡 创新点**

创新点在于将流量特征映射到文本描述，并通过动态特征调制和流模式原型学习克服语义鸿沟，同时采用跨模态自动注释实现无需人工标注的数据生成。

**🔧 技术方法**

技术包括流特征编码器、注意力LSTM解码器、动态特征调制、原型学习、跨模态注释（Qwen‑VL‑Max）、多阶段损失训练。

**📊 数据集**

使用了40,000条流量-描述对，来自8名用户、20款应用的15秒流量片段和同步屏幕视频。

**📈 对比分析**

与基线Transformer+LSTM模型对比，T2T在BLEU‑4 58.1、METEOR 38.3、ROUGE‑L 70.5、CIDEr 108.7等指标上优于基线，并能与视觉‑语言模型的生成效果相媲美。

**⚠️ 局限性**

局限在于对不同平台（iOS/Android）流量差异敏感、对低速/低频应用（如即时通讯）识别仍不佳，以及需依赖同步视频进行自动标注，难以在无视频环境下训练。

---

## 367. Towards Athlete Fatigue Assessment from Association Football Videos

**arXiv ID:** 2604.05636 | [PDF](https://arxiv.org/pdf/2604.05636v1)

**作者:** Xavier Bou `[一作]` (Université Paris-Saclay, CNRS, ENS Paris-Saclay), Anthony Cioppa `[通讯]` (University of Liège)

**通讯引用:** 742 | [OpenAlex ID](https://openalex.org/A5059351804)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用单摄像头转播视频的Game State Reconstruction提取球员轨迹，计算速度与加速度，并基于此构建加速度-速度（A–S）曲线，以评估比赛中的疲劳变化。

**💡 创新点**

首次将公开转播视频的GSR方法与疲劳监测结合；提出基于时域窗口的速度与加速度一致性估计算法；在公开数据集上验证单目视频可实现与GPS相当的疲劳指标提取。

**🔧 技术方法**

单目Game State Reconstruction（Broadcast2Pitch/SoccerMaster）、Kalman滤波与Savitzky–Golay平滑、时域差分、线性回归构建A–S曲线。

**📊 数据集**

SoccerNet GSR测试集（49段，773名球员）以及一段45分钟的瑞士超级联赛完整上半场视频。

**📈 对比分析**

速度/加速度的MAE约0.48 m/s、RMSE约1.10 m/s、Pearson r≈0.88；A–S曲线与真值高度一致，可分辨上半场与下半场疲劳差异，整体性能符合单目视频的可行性要求。

**⚠️ 局限性**

轨迹噪声、遮挡导致的误差仍显著；加速度第二导数放大噪声；缺乏高精度GPS基准，评估受限；对低可靠检测球员难以提取指标，需要进一步提升鲁棒性。

---

## 368. Adaptive Material Fingerprinting for the fast discovery of polyconvex feature combinations in isotropic and anisotropic hyperelasticity

**arXiv ID:** 2604.05698 | [PDF](https://arxiv.org/pdf/2604.05698v1)

**作者:** Moritz Flaschel `[一作]`, Ellen Kuhl `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种适用于不可压缩超弹性材料的自适应材料指纹（adaptive Material Fingerprinting）方法，可在不求解连续优化问题的情况下实时发现材料模型。

**💡 创新点**

创新点在于将预先生成的数据库与迭代模式识别算法相结合，允许通过逐步加入特征函数构造任意线性组合的材料模型，从而提升拟合灵活性和精度。

**🔧 技术方法**

采用预计算特征指纹数据库、前向逐步回归思路的迭代模式识别算法，以及可选的多重函数(g^A,g^B,g^C,h^A,h^B)实现特征构造，并使用线性插值、归一化与多项式正则化等技术。

**📊 数据集**

使用了实验数据集：硫化橡胶在20℃、50℃的单轴、纯剪切与等面拉伸测试；以及犬皮的多向拉伸实验。

**📈 对比分析**

与原始材料指纹、神经网络以及经典符号回归方法比较，发现自适应方法在保持近似神经网络水平的拟合精度的同时，显著减少了计算时间和训练成本，平均决定系数R²可达0.99以上。

**⚠️ 局限性**

局限性包括对特征函数空间的预先约束、对多重参数选择的手工调优，以及在更复杂非对称加载或非可压缩材料时可能缺乏泛化能力。

---

## 369. Let Geometry GUIDE: Layer-wise Unrolling of Geometric Priors in Multimodal LLMs

**arXiv ID:** 2604.05695 | [PDF](https://arxiv.org/pdf/2604.05695v1)

**作者:** Chongyu Wang `[一作]` (Xi'an Jiaotong University), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 9237 | [OpenAlex ID](https://openalex.org/A5100662197)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为GUIDE的框架，能够在多模态大语言模型的早期层逐步注入多层次几何先验，以提升对3D空间的感知与推理能力。

**💡 创新点**

创新点包括：①将几何先验从单一输入级融合转变为跨层级递进注入；②在几何编码器内部实现多粒度采样并与LLM层严格对齐；③引入双向上下文感知门控，动态调节几何信息的注入强度。

**🔧 技术方法**

技术实现主要依赖：前向几何基础模型（如VGGT）进行多层次特征提取；严格的网格对齐保证视觉与几何特征在空间位置上的一致性；Qwen3‑VL大语言模型的早期层逐层融合；token‑级与层级级的双门控机制。

**📊 数据集**

使用的数据集包括：VSI‑Bench、MMSI‑Bench、ViewSpatial、CV‑Bench（用于空间推理）以及ScanRefer、Scan2Cap、SPAR‑7M、LLaVA‑Video‑178K（用于3D场景理解）等。

**📈 对比分析**

与传统几何感知MLLM（VLM‑3R、VGLLM、VG‑LLM）及闭源基准（GPT‑5、Gemini‑2.5‑Pro）进行对比，GUIDE‑9B在VSI‑Bench平均分64.2、ScanRefer 物体定位Acc@0.25 59.6、Scan2Cap CIDEr 81.2 等指标上均优于基线和现有公开模型，展示了显著的性能提升。

**⚠️ 局限性**

局限性：过深的层级注入会导致语义冲突，需要门控机制；目前仅依赖单目RGB序列，仍难以处理极端遮挡或非视觉传感器场景；对几何编码器的依赖导致推理速度和显存占用相对较高。

---

## 370. Non-GRS type MDS and AMDS codes from extended TGRS codes

**arXiv ID:** 2604.05682 | [PDF](https://arxiv.org/pdf/2604.05682v1)

**作者:** Meiying Zhang `[一作]` (Qufu Normal University), Yanbin Zheng `[通讯]` (Qufu Normal University)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5100681028)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文构造了一类扩展的扭曲广义Reed-Solomon (TGRS) 码，并确定了这些码成为最大距离可分 (MDS) 或几乎最大距离可分 (AMDS) 的必要和充分条件。

**💡 创新点**

创新点在于证明了所构造的码不是广义Reed-Solomon (GRS) 码的等价形式，并计算了这些码的覆盖半径和深孔。

**🔧 技术方法**

使用了扩展的扭曲广义Reed-Solomon (ETGRS) 码的构造技术。

**📊 数据集**

使用了有限域上的线性码，具体数据集未明确提及，但涉及到的参数包括长度n、维度k和其他相关参数。

**📈 对比分析**

通过与现有的GRS码进行比较，证明了所构造的码在MDS和AMDS性质上具有优势，且在特定条件下能够抵抗攻击。

**⚠️ 局限性**

限制在于所构造的码的参数范围需要满足特定条件，且在某些情况下可能无法达到最佳性能。

---

## 371. A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model

**arXiv ID:** 2604.05672 | [PDF](https://arxiv.org/pdf/2604.05672v1)

**作者:** Kaidong Zhang `[一作]` (SYSU), Xiaodan Liang `[通讯]` (SYSU)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 A_1 这一可适配、截断式 Vision‑Language‑Action（VLA）模型，能够在多机器人、多任务的仿真与真实环境中实现高精度操纵，同时通过预算感知的自适应推理显著降低计算延迟。

**💡 创新点**

创新点包括：①基于动作一致性阈值的早停机制，使 VLM 只计算到必要层级；②跨层截断流匹配（Inter‑Layer Truncated Flow Matching）以 2 步暖启动的方式压缩流匹配的迭代成本；③两阶段大规模预训练+细化，充分利用公开机器人数据与自采轨迹，实现跨平台泛化。

**🔧 技术方法**

技术手段主要包括：使用 Molmo‑7B 作为 VLM 主干；两种动作头（流匹配 FM 与 MLP）；KV‑条件自注意力的动作生成；预算感知的早停与动作一致性阈值；跨层截断流匹配与暖启动；两阶段训练策略。

**📊 数据集**

使用公开机器人数据集（DROID、AgiBot、RoboCOIN、RoboMind、GM‑100、RoboChallenge）以及 15,951 条自采实物轨迹，涵盖多种机器人外观与任务场景。

**📈 对比分析**

在 RoboChallenge 基准上，A_1 取得 29.00% 的平均成功率，超过 π_0（28.33%）、X‑VLA（21.33%）和 RDT‑1B（15.00%）。在实测四台机器人上，平均成功率 56.7% 高于 π_0.5（47.5%）和 π_0（40.8%）；在 LIBERO 与 VLABench 仿真任务中，成功率分别达 96.6% 与 53.5%，显著优于现有开源模型。

**⚠️ 局限性**

局限性：①尽管通过截断与早停降低了推理时间，但仍需在流匹配头中进行多步迭代，部分任务的延迟仍高；②早停阈值需要在训练集上精细校准，迁移到新场景时可能需要重新调整；③对极端长程、极其复杂的多步骤任务的性能尚未完全验证；④依赖于大量预训练数据，对数据缺乏或分布偏移的场景可能效果下降。

---

## 372. CuraLight: Debate-Guided Data Curation for LLM-Centered Traffic Signal Control

**arXiv ID:** 2604.05663 | [PDF](https://arxiv.org/pdf/2604.05663v1)

**作者:** Qing Guo `[一作]` (Beijing University of Posts and Telecommunications), Lei Li `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 12164 | [OpenAlex ID](https://openalex.org/A5100440407)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了CuraLight框架，结合RL助手和多LLM辩论系统实现LLM驱动的交通信号控制

**💡 创新点**

创新点在于利用扩散式RL助手生成高质量训练对话并通过多LLM辩论系统提供可解释的优先级评分，提升对异构交叉口的适应性

**🔧 技术方法**

使用扩散式强化学习、LoRA低秩微调、多LLM集成辩论与优先级融合的混合训练流程

**📊 数据集**

在SUMO模拟的济南、杭州、亦庄三条真实道路网络上进行实验

**📈 对比分析**

与传统交通方法、深度RL方法和大规模LLM基线相比，CuraLight在ATT、AQL、AWT上分别提升约5.3%、5.1%和7.0%，并表现出更强的跨网络零样本迁移和在场景迁移中的鲁棒性

**⚠️ 局限性**

局限在于辩论过程对实时部署的计算开销较大，且目前仅关注延迟、等待和排队等指标，未覆盖排放等更广泛目标

---

## 373. How Much Trust is Enough? Towards Calibrating Trust in Technology

**arXiv ID:** 2604.05658 | [PDF](https://arxiv.org/pdf/2604.05658v1)

**作者:** Gabriela Beltrão `[一作]` (Tallinn University), David Lamas `[通讯]` (Tallinn University)

**通讯引用:** 1247 | [OpenAlex ID](https://openalex.org/A5067219028)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对两项在线调查数据的实证分析，构建并验证了人机信任尺度（HCTS）的解释范围，从而为评估和校准技术信任提供可操作的指南。

**💡 创新点**

创新点在于首次将情感尺度与HCTS得分映射，使用统计边界阈值和95%置信区间定义三类信任水平（欠信任、适度信任、过度信任），并提出可视化解释框架。

**🔧 技术方法**

技术方法包括使用五点李克特量表的HCTS问卷、情感描述词尺度、单因素方差分析、Games‑Howell事后检验、置信区间和中点阈值计算。

**📊 数据集**

数据集来自两项2023‑2025年间的在线调查，分别针对执法用面部识别系统（N=711）和生物识别支付系统（N=227），共计938名受访者。

**📈 对比分析**

比较方法是将情感等级与HCTS得分进行递归映射，计算各信任类别的均值与置信区间，并通过中点阈值定义分界。相较于单一阈值，所提取的三层信任范围更具统计显著性与情境适用性。

**⚠️ 局限性**

局限性包括样本仅限两种技术，且未跨情境验证测量不变性；阈值确定过程包含主观性，未来需要在更广泛领域重复验证与细化。

---

## 374. Probing Intrinsic Medical Task Relationships: A Contrastive Learning Perspective

**arXiv ID:** 2604.05651 | [PDF](https://arxiv.org/pdf/2604.05651v1)

**作者:** Jonas Muth `[一作]` (Karlsruhe Institute of Technology), Simon Reiß `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 465 | [OpenAlex ID](https://openalex.org/A5091379199)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研究了医学视觉任务的内在关系，提出了基于对比学习的任务嵌入框架（TaCo），并对任务空间进行可视化与评估。

**💡 创新点**

首次将任务输入-输出对作为对比学习样本，构建统一任务表示空间；同时引入失败任务以提升任务区分度，显著提高对未见任务的识别。

**🔧 技术方法**

使用监督式对比学习（Supervised Contrastive），ResNet‑50 6 通道输入 + 投影头；通过 t‑SNE 可视化、kNN 分类评估与任务相似度矩阵。

**📊 数据集**

基于 35 个医学影像数据集（CT、MRI、X‑ray、超声、EM、光学相干断层、皮肤镜等），共 291,165 张图像，生成 1,134 个视觉任务（30 类任务 × 39 数据集）。

**📈 对比分析**

与 CLIP、ImageNet 预训练 ResNet、SimCLR 等基线进行 kNN F1 对比；TaCo 在已见任务上提升 10%+F1，未见任务与数据集也显著优于基线；引入失败任务后，未见数据集分类进一步提升。

**⚠️ 局限性**

限制：任务数量仍有限，主要捕捉任务与数据分布关系，难以完全区分不同数据集；对极端任务变形或非典型任务的泛化能力未充分验证；实验仅覆盖医学影像域，跨域迁移需进一步研究。

---

## 375. PECKER: A Precisely Efficient Critical Knowledge Erasure Recipe For Machine Unlearning in Diffusion Models

**arXiv ID:** 2604.05634 | [PDF](https://arxiv.org/pdf/2604.05634v1)

**作者:** Zhiyong Ma `[一作]` (Cao Tu Li Technology Co., Ltd), Qingyuan Chuai `[通讯]` (Cao Tu Li Technology Co., Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于显著性掩码的数据自由蒸馏框架 PECKER，用于在扩散模型中实现高效、精确的机器遗忘。

**💡 创新点**

创新点在于利用暂时冻结的生成器计算梯度显著性并生成掩码，只对关键参数更新，从而显著降低训练时间并提升遗忘质量。

**🔧 技术方法**

使用轻量级伪分数网络、数据自由蒸馏、梯度显著性评分与掩码、类别区分更新等技术。

**📊 数据集**

在 CIFAR‑10、STL‑10、Stable Diffusion（用于名人、裸露概念）等数据集上进行实验。

**📈 对比分析**

与 Retrain、SA、SalUn、SFD 等基线相比，PECKER 在 UA、FID、IS、Precision 等指标上表现最佳，且训练步骤显著减少。

**⚠️ 局限性**

局限性包括对极大规模模型和高分辨率任务的可扩展性尚未验证，且伪分数网络的设计仍需进一步优化。

---

## 376. YoNER: A New Yorùbá Multi-domain Named Entity Recognition Dataset

**arXiv ID:** 2604.05624 | [PDF](https://arxiv.org/pdf/2604.05624v1)

**作者:** Peace Busola Falola `[一作]`, David Ifeoluwa Adelani `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了首个覆盖五个领域（圣经、博客、电影、广播、维基百科）的多域 Yoruba NER 数据集 YoNER，并基于此数据集训练了专门的 Yoruba 预训练模型 OyoBERT；

**💡 创新点**

创新点包括：① 构建多域 Yoruba NER 数据集，填补低资源语言跨域标注空白；② 训练了 Yoruba 专属预训练模型 OyoBERT，展示在多域 NER 上可与非 Yoruba 多语言模型竞争；③ 在跨域、少量本域数据、跨语言转移等三类实验中系统评估模型表现；

**🔧 技术方法**

使用技术主要有：多域数据收集与人工标注（采用 MasakhaNER 规范）；Transformer 预训练模型（BERT、XLM‑R、AfroXLMR、AfriBERTa 等）；微调 NER 任务的 IOB2 标注与交叉验证；对比实验与 F1 评估；

**📊 数据集**

使用数据集：① 新构建的 YoNER（约5,148句、100,795 词，五个域）；② MasakhaNER 2.0（新闻域）；③ 现有多语言预训练模型（mBERT、XLM‑R、AfroXLMR 等）；④ 英文对照数据集（CoNLL、WikiAnn、OntoNotes）用于跨语言实验；

**📈 对比分析**

比较方法：在 YoNER 上做新闻域预训练+跨域微调、加 200 句本域数据微调、跨语言 RoBERTa vs XLM‑R，以及 OyoBERT vs AfroXLMR；性能方面：OyoBERT‑base 在多域 NER 上可与 AfroXLMR‑large 相当，但在 NER 上略逊；新闻域模型对 Wiki、圣经转移较好，博客、电影转移差；少量本域数据可显著提升目标域性能；

**⚠️ 局限性**

局限性：① 数据集覆盖实体类型有限（缺乏 DATE、少量 ORG），难以评估模型在完整实体集上的表现；② 预训练语料规模相对较小，未对大模型做评估；③ 仅在低资源 Yoruba 语言上验证，缺乏对其他低资源 African 语言的推广；④ 未深入分析模型偏差与生成语料的影响。

---

## 377. Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion

**arXiv ID:** 2604.05688 | [PDF](https://arxiv.org/pdf/2604.05688v1)

**作者:** Zhen Cheng `[一作]` (China Merchants Bank Artificial Intelligence Laboratory), Jin-Long Li `[通讯]` (China Merchants Bank Artificial Intelligence Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Attention Editing 框架，能够在不重新训练整个模型的前提下，将已训练好的大语言模型的注意力结构替换为新的注意力架构。

**💡 创新点**

创新点在于：① 将目标注意力模块视为可学习的替代模块，避免了对原始权重的精细矩阵分解；② 采用分阶段的 progressive distillation（先进行 block‑wise teacher‑forcing，再进行全模型的知识蒸馏），实现随机初始化的注意力模块的稳定收敛；③ 设计了 GateSWA 的轻量级门控机制和硬件友好的 MLA 配置。

**🔧 技术方法**

使用了教师‑学生的分层监督（block‑wise MSE）和全模型的 token‑level KL 蒸馏，以及可选的中间层余弦相似度正则；同时在 GateSWA 中加入元素级门控；在 MLA 中采用低秩压缩的 KV 缓存；实验在 Ascend 910B 集群上使用了 FSDP/DeepSpeed 分布式训练。

**📊 数据集**

训练数据采用预训练格式的混合语料：40% 通用领域、35% 数学与代码、25% 中文语料；Stage I 采集 2 B token，Stage II 采集 6 B token，并通过自研的渐进式数据梯度策略。评测使用了 ARC‑E、ARC‑C、C‑Eval、MMLU、GSM8K 以及思考模式下的 GSM8K 与 C‑Eval。

**📈 对比分析**

与原始 GQA（Qwen3‑8B、Qwen3‑30B‑A3B）做对比：在 few‑shot 基准上保持相近甚至略优；在 GSM8K 上几乎与原始模型持平；在 C‑Eval 上略有下降；KV‑cache 大幅下降（≈80%），从而提升推理吞吐量和首个 token 的延迟，尤其在高并发场景下显著优于原始模型。

**⚠️ 局限性**

局限性：① 训练数据量仅为原始模型的千分之一，尚未验证在更大数据规模下的可扩展性；② 缺乏针对工具调用和长期决策的交互式数据与评测；③ 在 MLA 方案下 TTFT 与原始模型相当，需进一步优化前填充阶段的算力；④ 仅在 Ascend 910B 上验证，未探讨跨平台的适配性。

---

## 378. Time-Domain Voice Identity Morphing (TD-VIM): A Signal-Level Approach to Morphing Attacks on Speaker Verification Systems

**arXiv ID:** 2604.05683 | [PDF](https://arxiv.org/pdf/2604.05683v1)

**作者:** Aravinda Reddy PN `[一作]` (Indian Institute of Technology Kharagpur), Kunal Singh `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 1243 | [OpenAlex ID](https://openalex.org/A5002538464)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种在时域直接对两人语音信号进行融合的声纹模仿攻击方法——TD‑VIM。

**💡 创新点**

创新点在于：不需要特征嵌入或参考文本，完全语言与网络架构无关；通过在时域选择并平均不同比例的信号实现身份混合；并首次引入 Generalized Morph Attack Potential (G‑MAP) 指标量化多设备、多模型、多攻击类型的漏洞。

**🔧 技术方法**

主要技术包括信号零填充、分段选择、比例平均、x‑vector、RawNet3、Verispeak 以及 G‑MAP 计算。

**📊 数据集**

使用公开的多语言手机语音数据集 MAVS（英语、印地语、孟加拉语），涵盖两款手机（iPhone 11、Samsung S8）。

**📈 对比分析**

通过在三种 SVS 上做多轮探测并计算 G‑MAP，结果显示：Samsung S8 对所有四种模仿类型的攻击最易被识别；Verispeak 的脆弱性介于 x‑vector 与 RawNet3 之间；整体 G‑MAP 值表明 TD‑VIM 在不同设备与语言下均能显著突破验证。

**⚠️ 局限性**

局限性：实验仅覆盖两款手机和三种语言，未验证更广泛硬件与语料；方法假设攻击者已知同一句话的两份语音，且对不同句子、说话者多样性未知；未评估对抗性防御（如深度伪造检测）和实时性能。

---

## 379. LUDOBENCH: Evaluating LLM Behavioural Decision-Making Through Spot-Based Board Game Scenarios in Ludo

**arXiv ID:** 2604.05681 | [PDF](https://arxiv.org/pdf/2604.05681v1)

**作者:** Ojas Jain `[一作]` (BITS Pilani), Dhruv Kumar `[通讯]` (BITS Pilani)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了一个基于Ludo的基准（LudoBench），通过手工生成的480个局面（12个决策类别）来评估大语言模型在不确定性多玩家棋类中的战略推理能力。

**💡 创新点**

采用了“spot”式评估、对对手历史的对偶“grudge”设计、基于Expectiminimax的游戏理论基准以及对LLM行为的系统归类（finishers vs builders），揭示了模型的策略偏差与叙事敏感性。

**🔧 技术方法**

结合了Ludo模拟器、手写提示模板、零样本推理（temperature=0）、基于期望最大化搜索的游戏理论代理以及对LLM输出的合法性与策略一致性自动判定。

**📊 数据集**

480个手工设计的Ludo局面数据集（每个局面对应12个决策维度），附加5种角色说明和配对的中立/仇恨叙事，全部可在公开链接获取。

**📈 对比分析**

将六款LLM在所有局面和角色下与游戏理论基准对比，发现模型与基准的动作一致率仅40–46%，并将模型划分为“finishers”和“builders”，表现出显著的策略偏差与叙事敏感性。

**⚠️ 局限性**

评估仅为单一步骤的spot式测试，未覆盖所有可能局面，模型样本有限，缺乏多轮游戏模拟，且只使用英文提示和小型动作空间，限制了结果的普适性和对更复杂环境的推广。

---

## 380. From Incomplete Architecture to Quantified Risk: Multimodal LLM-Driven Security Assessment for Cyber-Physical Systems

**arXiv ID:** 2604.05674 | [PDF](https://arxiv.org/pdf/2604.05674v1)

**作者:** Shaofei Huang `[一作]` (Singapore Management University), Lwin Khin Shar `[通讯]` (Singapore Management University)

**通讯引用:** 1353 | [OpenAlex ID](https://openalex.org/A5029828965)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 ASTRAL，一个基于多模态大语言模型的架构中心化安全评估框架，能够从不完整或缺失的建筑图和文档中重建 CPS 架构并进行威胁建模与定量风险分析。

**💡 创新点**

创新点在于将多模态 LLM 与结构化推理、Prompt 链接、架构守门、以及 Bayesian 网络结合，形成端到端的从图像到风险评分的自动化流程。

**🔧 技术方法**

技术主要包括大语言模型（Mistral 等）多模态输入、Prompt Chaining、Guardrails、自动化建模（AutomationML）、Bayesian Network 量化风险。

**📊 数据集**

使用的数据集为多种 CPS 案例的图像化架构（如市政供热系统、医疗 CPS、光伏逆变器）以及公开的威胁情报与 CVE 列表。

**📈 对比分析**

通过消融实验与 14 名资深安全从业者的调查，表明全流程实现了可信的架构重建、威胁覆盖率提升，并在风险评分上与传统手工模型相当或更优。

**⚠️ 局限性**

局限包括对 LLM 输出的随机性和幻觉依赖、缺少实时资产数据、对极端缺失图像的鲁棒性待验证，以及实验规模与真实工业环境的差距。

---

## 381. What Models Know, How Well They Know It: Knowledge-Weighted Fine-Tuning for Learning When to Say "I Don't Know"

**arXiv ID:** 2604.05779 | [PDF](https://arxiv.org/pdf/2604.05779v1)

**作者:** Joosung Lee `[一作]` (NAVER CLOUD), Jeonghoon Kim `[通讯]` (NAVER CLOUD)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对每个训练实例进行多样本推理估计细粒度知识分数，并将该分数用作训练权重，在低知识实例上鼓励模型输出<Idk>，从而实现知识加权微调（KWT），在保持准确率的同时显著提升模型对不确定性（“我不知道”）的表达，降低幻觉发生。

**💡 创新点**

创新点包括：① 利用多样本推理与LLM-as-judge精确估计实例级知识分数；② 将知识分数映射为可变训练权重（熟悉度加权），使模型在已知信息上更自信、在未知信息上更谨慎；③ 在低知识实例末尾追加<Idk>令模型显式拒绝回答；④ 提出了联合评估准确率与不确定性表达的nAUPC指标。

**🔧 技术方法**

核心技术：多样本few-shot推理（自一致性），LLM-as-judge 评估，知识加权损失函数，<Idk> token 训练，nAUPC、A-FPR、IDK Precision 等不确定性评估指标。

**📊 数据集**

使用的主要数据集有：HaluEval（通用知识），MedQA（医学知识），SciQ（科学常识）以及显式可答/不可答的外域数据集 RefuNQ、SelfAware、NEC 等。

**📈 对比分析**

对比方法包括标准SFT、FT-TOP、R‑Tuning、SEAL、以及不同权重策略的KWT（RF、U）。实验结果表明，KWT（尤其是基于LLM‑judge的知识估计）在nAUPC最高、A‑FPR最低、IDK Precision最高的同时，标准准确率仅略低于SFT，说明在保持整体性能的前提下显著提升了不确定性表达与幻觉抑制。

**⚠️ 局限性**

局限性：需要对每个训练实例进行大量多样本推理，计算成本较高；方法依赖于多次推理和外部LLM-as-judge，推理效率和成本均需进一步降低；目前验证范围集中在特定问答数据集，跨领域与更大规模的通用性验证仍待探索。

---

## 382. PDMP: Rethinking Balanced Multimodal Learning via Performance-Dominant Modality Prioritization

**arXiv ID:** 2604.05773 | [PDF](https://arxiv.org/pdf/2604.05773v1)

**作者:** Shicai Wei `[一作]` (University of Electronic Science and Technology of China), Yang Luo `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 8673 | [OpenAlex ID](https://openalex.org/A5057127971)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种性能主导模态优先（PDMP）策略，用以解决多模态学习中的不足优化问题，强调让性能最优模态主导训练过程；

**💡 创新点**

①挑战传统平衡学习假设，证明由性能主导模态驱动的不平衡学习可获得更佳效果；②通过单模态性能排序确定主导模态，再利用非对称梯度调节实现主导模态权重放大；

**🔧 技术方法**

利用独立训练的单模态模型进行性能排名；对每个模态引入γₚ、γᵣ两参数实现梯度加权；兼容任意融合方式和多模态数量；

**📊 数据集**

CREMA‑D、AVE、Kinetics‑Sounds、CEFA、UCF‑101、VGGSound 等常用音频‑视觉与多模态数据集；

**📈 对比分析**

与OGM‑GE、AGM、PMR等梯度调制方法以及G‑Blending、MLA、MMPareto、D&R等单模态正则化方法对比；在所有数据集均获得最高或第二高的准确率/宏 F1 分数，例如 CREMA‑D 上 80.21%（宏 F1 80.34%），AVE 上 71.28%（宏 F1 67.49%），CEFA 上 74.45%（宏 F1 74.78%），UCF‑101 上 83.65%（宏 F1 82.08%）；

**⚠️ 局限性**

需要额外训练单模态模型进行性能排序，导致一定额外计算成本；在极大规模数据集或在线学习场景下可能需要进一步优化或简化模态分析步骤；

---

## 383. The LLM Effect on IR Benchmarks: A Meta-Analysis of Effectiveness, Baselines, and Contamination

**arXiv ID:** 2604.05766 | [PDF](https://arxiv.org/pdf/2604.05766v1)

**作者:** Moritz Staudinger `[一作]` (TU Wien), Allan Hanbury `[通讯]` (TU Wien)

**通讯引用:** 9672 | [OpenAlex ID](https://openalex.org/A5020665735)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对TREC Robust04和DL20的143篇论文进行纵向元分析，评估LLM组件对检索效果的影响。

**💡 创新点**

首次量化“LLM效应”，并将数据泄露检测方法改用于reranking，揭示两大基准中显著泄露。

**🔧 技术方法**

采用系统检索、实验结果抽取、回归趋势分析以及改编的Data Contamination Quiz（DCQ）进行泄露评估。

**📊 数据集**

使用TREC Robust04和TREC Deep Learning 2020 Passage Retrieval（DL20）数据集。

**📈 对比分析**

与强基线对比后发现LLM系统在Robust04上nDCG@10提升约20%，在DL20上提升约8.8%；通过排除泄露主题后效果下降但置信区间宽大，难以确定提升原因。

**⚠️ 局限性**

检索仅限ACM DL导致基线不足，关键字检索可能漏检，Robust04交叉验证方式不一致，且DCQ改造后样本量小，置信区间宽，影响结论可靠性。

---

## 384. Controlling Distributional Bias in Multi-Round LLM Generation via KL-Optimized Fine-Tuning

**arXiv ID:** 2604.05756 | [PDF](https://arxiv.org/pdf/2604.05756v1)

**作者:** Yanbei Jiang `[一作]` (University of Melbourne), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1199 | [OpenAlex ID](https://openalex.org/A5065822589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多轮生成分布对齐框架，通过在LLM中加入steering tokens并进行双目标微调，实现对输出属性分布的可控调节；

**💡 创新点**

创新点在于将KL散度用于steering token概率校准，并结合Kahneman‑Tversky Optimization (KTO) 以保持语义一致，构成一个完整的双目标训练机制；

**🔧 技术方法**

采用了Steering Token Calibration、KTO损失、KL对齐、LoRA微调、Preference Dataset构建，以及vLLM推理平台；

**📊 数据集**

使用了基于英国与美国职业数据构建的属性分布数据集（性别、种族、情感），并在故事生成任务中使用大型LLM作为判定器；

**📈 对比分析**

与Zero‑shot、Prompt Engineering、Instruction Fine‑Tuning (IFT)、Direct Preference Optimization (DPO) 等基线在 MAE 上进行对比，本文方法在六种评测设置下平均 MAE 下降 27%–37%，在情感故事生成上表现最优；

**⚠️ 局限性**

局限性包括：在长篇故事等隐式属性控制上效果不足；仅针对单一属性的分布；需要对模型进行可微调且可插入token，API 受限的闭源模型无法直接使用。

---

## 385. Controllable Image Generation with Composed Parallel Token Prediction

**arXiv ID:** 2604.05730 | [PDF](https://arxiv.org/pdf/2604.05730v1)

**作者:** Jamie Stirling `[一作]` (Durham University), Hubert P. H. Shum `[通讯]` (Durham University)

**通讯引用:** 3764 | [OpenAlex ID](https://openalex.org/A5038258635)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于离散序列生成过程的组合框架，能在离散表示空间（如VQ‑VAE / VQ‑GAN）中实现多条件、可控的图像生成。

**💡 创新点**

创新点包括：① 对条件概率进行理论化组合（产品专家形式），② 将其推广到任意离散迭代生成任务，③ 引入概念权重实现强调、削弱或否定条件，④ 在不增加模型参数的前提下兼具高精度与高速。

**🔧 技术方法**

核心技术：离散向量量化（VQ‑VAE / VQ‑GAN）+ 条件并行 token 预测（absorbing diffusion）+ 概念权重调控 + 组合公式推导。

**📊 数据集**

使用了 Positional CLEVR、Relational CLEVR、FFHQ 三个公开数据集进行训练与评估。

**📈 对比分析**

与 StyleGAN2、StyleGAN2‑ADA、LACE、GLIDE、EBM、Composed GLIDE 等基线对比，错误率平均下降 63.4%（3 个数据集），FID 取得 7/9 场景最佳或近似最佳，速度提升 2.3×–12×，并在文本到图像预训练模型上实现可控生成。

**⚠️ 局限性**

主要局限：① 需要假设输入条件在给定输出时条件独立，若训练数据偏差可能影响；② 需要手工设定概念权重，缺乏自动化选择机制；③ 对条件数量的线性运算量增加，虽然在实验中通过迭代次数控制，但仍比单条件生成稍慢。

---

## 386. Single-Stage Signal Attenuation Diffusion Model for Low-Light Image Enhancement and Denoising

**arXiv ID:** 2604.05727 | [PDF](https://arxiv.org/pdf/2604.05727v1)

**作者:** Ying Liu `[一作]` (Central South University), Caiyun Wu `[通讯]` (Central South University)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5033937011)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出单阶段信号衰减扩散模型（SADM）用于低光图像增强与去噪，直接在扩散过程内完成亮度提升与噪声抑制。

**💡 创新点**

核心创新是引入信号衰减系数 k_t，使前向扩散既加入噪声又逐步衰减信号，形成与低光真实退化相匹配的过程；同时将多尺度金字塔与 DDIM 采样结合，减少计算量并提升效率。

**🔧 技术方法**

技术要点包括：基于 DDPM 的前向/逆向推理公式改写、SR3Unet 结构、条件信息（低光图、去雾先验、位置编码）、多尺度金字塔采样、DDIM 采样、亮度对齐损失、感知损失、L1 损失等。

**📊 数据集**

实验使用 LOLv1、LOLv2-Real、LOLv2-Syn 三大公共数据集，并在 DICM、NPE、MEF 等无配对低光数据集上做泛化验证。

**📈 对比分析**

与 12 种 SOTA 方法（Zero‑DCE、UR‑Net、LLFormer、PyDiff、Retinexformer、GSAD、WBDM、FourierDiff、LightenDiffusion、AnlightenDiff、CLODE、Diff‑Retinex++）在 PSNR/SSIM 上均领先；在 LOLv1 上 PSNR 28.10、SSIM 0.877；LOLv2‑Real PSNR 28.93、SSIM 0.8797；LOLv2‑Syn PSNR 26.70、SSIM 0.9224，且推理速度相对较快。

**⚠️ 局限性**

局限性：对极端低光场景或极低帧率实时场景的适应性尚待验证；模型训练仍需较多计算资源，推理时多尺度处理对显存有一定需求。

---

## 387. GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance

**arXiv ID:** 2604.05721 | [PDF](https://arxiv.org/pdf/2604.05721v1)

**作者:** Weiqi Zhang `[一作]` (Tsinghua University), Yu-Shen Liu `[通讯]` (Tsinghua University)

**通讯引用:** 4944 | [OpenAlex ID](https://openalex.org/A5101691399)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GaussianGrow 方法，利用3D点云作为几何先验，通过文本引导的多视角扩散模型逐步生成高质量、高精度的3D高斯散点表示。

**💡 创新点**

创新点在于将点云直接作为几何先验，设计多视角重建与重叠区域优化以及迭代姿态感知的填补策略，避免传统几何预测不可靠导致的质量下降。

**🔧 技术方法**

使用多视角扩散模型（如 Hunyuan3D-Paint）、Depth-Aware ControlNet、无符号距离场（UDF）、相机姿态优化、图像级与空间级高斯填补等技术。

**📊 数据集**

在 Objaverse、DeepFashion3D、T3Bench 等数据集上进行点云到高斯、文本到3D以及文本引导视觉合成的实验。

**📈 对比分析**

与 Texture、Text2Tex、Paint3D、SyncMVD、GAP、DreamGaussian、TriplaneGaussian、DiffSplat 等基线比较，GaussianGrow 在 FID、KID、CLIP 等指标上普遍优于或接近最佳方案，尤其在点云到高斯的质量和文本一致性上表现突出。

**⚠️ 局限性**

仍受点云质量影响，对极端噪声或稀疏点云的鲁棒性有限；多视角扩散与相机优化的计算成本较高；文本引导对复杂场景的覆盖仍有提升空间。

---

## 388. MPM: Mutual Pair Merging for Efficient Vision Transformers

**arXiv ID:** 2604.05718 | [PDF](https://arxiv.org/pdf/2604.05718v1)

**作者:** Simon Ravé `[一作]` (LARIS University of Angers), David Rousseau `[通讯]` (LARIS University of Angers)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种训练无关的互最近邻对合并（MPM）模块，用于在ViT语义分割模型中动态压缩令牌并通过收集映射重建完整特征，保持解码器不变。

**💡 创新点**

创新点在于无学习参数、无阈值的确定性互最近邻配对与轻量化收集重建，直接在端到端推理中实现时延提升。

**🔧 技术方法**

采用余弦相似度计算、互最近邻配对、平均合并和整数收集映射，并将模块插入ViT块之间。

**📊 数据集**

在ADE20K、Cityscapes、Pascal Context等公开语义分割数据集上进行实验。

**📈 对比分析**

与ToMe、ALGM、CTS等方法对比，MPM在H100 GPU上实现最高50%+ FPS提升，mIoU仅下降1–2%；在Raspberry Pi 5上亦可提升60%时延。

**⚠️ 局限性**

局限在于对小目标和细界限可能产生平滑，且O(N²)的相似度计算在极高分辨率下成本较高。

---

## 389. SemLink: A Semantic-Aware Automated Test Oracle for Hyperlink Verification using Siamese Sentence-BERT

**arXiv ID:** 2604.05711 | [PDF](https://arxiv.org/pdf/2604.05711v1)

**作者:** Guan-Yan Yang `[一作]` (National Taiwan University), Kuo-Hui Yeh `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2961 | [OpenAlex ID](https://openalex.org/A5043276911)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SemLink，一种基于 Siamese Sentence‑BERT 的自动化测试预言机，用于检测网页超链接的语义漂移和软 404 等问题。

**💡 创新点**

创新点包括：① 用句子级语义嵌入与多模态上下文（anchor、旁边文本、图像 OCR）结合；② 设计了 HWPPs 语义正向对数据集；③ 采用权重聚合与最大化策略，避免平均导致信息稀释；④ 在 CI/CD 场景下实现极低推理延迟，接近实时。

**🔧 技术方法**

技术栈：Siamese Neural Network + Sentence‑BERT (distiluse‑base‑multilingual‑cased‑v2) + 线性投影 + MLP + BCE 损失；特征提取涉及 DOM 解析、OCR、文本摘要、关键词提取。

**📊 数据集**

使用自建的 HWPPs 数据集，包含 63,870 条正向超链接‑网页对，分为 55,000 训练集与 8,870 验证集；并在 4,000 条人工标注的独立测试集上评估。

**📈 对比分析**

与 GPT‑5.2、GPT‑4o、Llama‑3 等大型语言模型比较，SemLink 在 Recall 上达到 96.00%，F1‑score 92.93%，速度提升约 47.5 倍；成本低、推理时间约 0.03 s/链接，能满足 CI/CD 实时需求。

**⚠️ 局限性**

局限性包括：① 对极端无结构或 WebGL 页面缺乏泛化；② 主要依赖文本，视觉信息不足导致对图像页面的识别失效；③ 对登录/授权重定向等功能性跳转误判；④ 侧文本权重经验化，可能不适用于所有页面布局。

---

## 390. EfficientMonoHair: Fast Strand-Level Reconstruction from Monocular Video via Multi-View Direction Fusion

**arXiv ID:** 2604.05794 | [PDF](https://arxiv.org/pdf/2604.05794v1)

**作者:** Da Li `[一作]` (King Abdullah University of Science and Technology), Ivan Viola `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 3122 | [OpenAlex ID](https://openalex.org/A5022294742)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在本论文中提出了 EfficientMonoHair 框架，能够从单目视频高效重建细粒度毛发几何体并支持后期渲染与物理仿真。

**💡 创新点**

创新点在于两大模块：Fusion‑Patch‑based Multi‑View Optimization (FPMVO) 用多视角补丁融合快速获得全局一致的外部方向场；Parallel Hair Growing (PHG) 通过并行卷积体化的占用‑方向体实现大规模毛发线段的同步生成与连接。

**🔧 技术方法**

技术实现上结合了 Instant‑NGP 的稠密重建、View‑Aware Transformer 的内部方向推断、卷积体化占用体与向量化 KD‑Tree 查询，并采用 Gaussian 场与扩散模型辅助高保真渲染。

**📊 数据集**

实验数据集包括真实多视角视频（来自 MonoHair）和合成 Hair20K（扩展自 USC‑HairSalon），对比了 MonoHair、DiffLocks 及 GaussianHaircut 等先进方法。

**📈 对比分析**

与 MonoHair（P=5）相比，EfficientMonoHair 在速度上提升 6–8 倍，同时在占用率与方向一致性指标上保持相近甚至优于其 88% 的 F1 分数；相较于 DiffLocks 亦在所有指标上表现更佳。

**⚠️ 局限性**

论文指出仍难以精确重建高度纠结的发型（如辫子、盘发）以及最后的头皮附件阶段仍需串行处理，限制了极端复杂场景下的鲁棒性与并行度。

---

## 391. Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation

**arXiv ID:** 2604.05795 | [PDF](https://arxiv.org/pdf/2604.05795v1)

**作者:** Abdullah Mazhar `[一作]` (Indian Institute of Information Technology Delhi), Md Shad Akhtar `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于上下文与知识蒸馏链式推理的临床评估框架 CARE，用于细粒度序数评估 AI 生成的治疗师回应。

**💡 创新点**

创新点在于将对话上下文、标签专属示例检索以及 GPT‑4o 生成的 KD‑CoT 结合，显著提升六大治疗原则的评估精度。

**🔧 技术方法**

核心技术包括局部上下文编码、标签专属示例检索、知识蒸馏链式推理（KD‑CoT）以及多标签序数分类头。

**📊 数据集**

使用 FAITH‑M 数据集（基于 HOPE 扩展），包含 10,172 条治疗师发言并标注六维序数标签。

**📈 对比分析**

与 15 种基线对比，CARE 在 weighted F1 上提升至 63.34，较最强基线 Qwen3 提升 64.26%（从 38.56 提升至 63.34）。

**⚠️ 局限性**

局限在于仅覆盖六项原则，无法处理跨文化、长期关系和更细粒度的临床决策；对话仅基于局部上下文，难以捕捉长序列动态。

---

## 392. An Empirical Study of Perceptions of General LLMs and Multimodal LLMs on Hugging Face

**arXiv ID:** 2604.05782 | [PDF](https://arxiv.org/pdf/2604.05782v1)

**作者:** Yujian Liu `[一作]` (Zhejiang University), Xiaoxue Ma `[通讯]` (Hong Kong Metropolitan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Hugging Face 公开的 38 种模型（21 个 GLLMs、17 个 MLLMs）的 662 条讨论线程进行人工标注和定性分析，探究用户在真实使用中的感知与关注点；

**💡 创新点**

提出了三层次（类别-子类别-叶节点）细粒度的用户关注点分类体系，并首次系统揭示了 LLM 访问/许可障碍、部署难题、文档缺失、微调难度以及多语言需求等关键痛点；

**🔧 技术方法**

采用多标签手工标注、统计抽样、Cohen κ 一致性评估、卡方检验与标准化残差分析，以及情感标注等方法；

**📊 数据集**

使用来自 Hugging Face 模型讨论区的 662 条讨论记录，涵盖 38 个活跃模型，按标签与活跃度筛选得到；

**📈 对比分析**

通过对 GLLM 与 MLLM 讨论主题分布进行卡方检验（p<0.001）和残差分析，验证两类模型在关注维度上的显著差异，提供对用户体验的定量洞察；

**⚠️ 局限性**

研究仅覆盖 Hugging Face 上的部分活跃模型，可能存在标注偏差、图像仅帖子被过滤、缺乏跨平台验证等局限性。

---

## 393. SoK: Understanding Anti-Forensics Concepts and Research Practices Across Forensic Subdomains

**arXiv ID:** 2604.05770 | [PDF](https://arxiv.org/pdf/2604.05770v1)

**作者:** Janine Schneider `[一作]` (University of Augsburg), Frank Breitinger `[通讯]` (University of Augsburg)

**通讯引用:** 2865 | [OpenAlex ID](https://openalex.org/A5068630039)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2004‑2024年间123篇关于反取证（anti‑forensics）的论文进行系统性综述与分类，梳理攻击手法、子域分布、研究动机与用途，并评估研究方法、工具与数据集的可用性；

**💡 创新点**

提出了完整的反取证知识体系与分类框架，厘清了不同子域中反取证与反反取证的关系，识别了研究缺口（如云、GPU、AI等新兴领域）、伦理讨论不足以及公共工具/数据集稀缺等问题；

**🔧 技术方法**

采用文献检索与搜索词工程、定量统计（论文数量、年份、子域占比）、定性编码（技术类型、攻击目标、受众）等混合方法，构建代码表并迭代修订；

**📊 数据集**

仅有5篇论文公开了数据集（主要为数据存储领域的磁盘镜像/云虚拟机镜像），其余数据多为私有实验环境；

**📈 对比分析**

对比方法主要是基于论文计数与类别占比的统计，可视化展示子域分布、攻击类型与研究关注度的变化；并未给出传统意义下的性能指标，而是以研究趋势与覆盖率为评价维度；

**⚠️ 局限性**

限制在于：①采用法律/取证背景与作者声明为筛选标准，导致忽略非正式或暗含反取证的研究；②仅覆盖英语论文，忽视其他语言工作；③对伦理讨论和公开工具/数据集的评估不够深入；④缺乏对反取证与反反取证技术相互作用的实证验证。

---

## 394. Hazard Management in Robot-Assisted Mammography Support

**arXiv ID:** 2604.05749 | [PDF](https://arxiv.org/pdf/2604.05749v1)

**作者:** Ioannis Stefanakos `[一作]` (University of York), Jihong Zhu `[通讯]` (University of York)

**通讯引用:** 10290 | [OpenAlex ID](https://openalex.org/A5051073741)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文开发了 MammoBot 乳腺 X 光辅助机器人，并提出了一套以利益相关者引导流程建模、SHARD 与 STPA 为核心的安全风险管理方法；

**💡 创新点**

创新点在于将 SHARD 与 STPA 两种安全分析技术与协同流程设计相结合，形成一条从需求、分析到安全需求迭代的完整、安全驱动的设计路径；

**🔧 技术方法**

技术方法包括：UML 活动图的流程建模、热成像感知与六轴力/力矩传感器的双臂协作控制、SHARD（软件危害分析）和 STPA（系统理论过程分析）；

**📊 数据集**

数据来源主要是实验室测试中的人形模型和热成像传感器采集的姿态与力学数据，论文未使用公开医学图像或临床数据集；

**📈 对比分析**

通过对比未改进前的系统流程，分析识别出多种技术与操作偏差，并将其转化为可验证的安全需求，虽未给出数值指标，但安全性明显提升；

**⚠️ 局限性**

局限性包括：方法基于建模与专家评估，缺乏真实临床验证；风险评估为定性，未覆盖非物理伤害与伦理影响；以及对实现层面安全约束的验证仍待进一步研究。

---

## 395. FoleyDesigner: Immersive Stereo Foley Generation with Precise Spatio-Temporal Alignment for Film Clips

**arXiv ID:** 2604.05731 | [PDF](https://arxiv.org/pdf/2604.05731v1)

**作者:** Mengtian Li `[一作]` (Shanghai University), Zhifeng Xie `[通讯]` (Shanghai University)

**通讯引用:** 964 | [OpenAlex ID](https://openalex.org/A5100301468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一个名为FoleyDesigner的端到端框架，用于自动化生成电影剪辑的时空对齐立体（5.1声道）音轨；

**💡 创新点**

创新点包括：①把专业Foley工作流程拆解为脚本生成、空间时间控制的音频合成和多代理专业混音三大阶段；②使用树思路（Tree-of-Thought）推理实现音频脚本的层级化与可验证性；③提出位置感知注入机制，将深度与方位信息通过Fourier特征编码后在Diffusion Transformer中交叉注意力实现帧级空间定位；④构建首个FilmStereo数据集，为立体Foley提供空间、时间和语义标注；

**🔧 技术方法**

技术手段包括多代理协同推理、VLM+深度估计提取空间轨迹、DiT扩散模型的跨模态条件生成、位置编码注入、FFT特征、专业混音（混响、均衡、动态处理）以及5.1声道上混；

**📊 数据集**

使用了FilmStereo数据集（166小时，14,784样本，8类Foley），以及对比实验中使用的AudioLDM、Stable Audio、SpatialSonic、See2Sound等公开音频合成数据；

**📈 对比分析**

通过与Stable Audio、SpatialSonic、See2Sound等基准的定量对比，FoleyDesigner在音质（CLAP 0.679、FAD 1.88）、空间对齐（GCC-MAE 48.79、CRW-MAE 34.23、FSAD 0.138）以及时间对齐（IoU 32.2）和电影Foley质量（ImageBind 0.402、AV‑Sync 0.726）等指标上均取得最高分，显著优于现有方法；

**⚠️ 局限性**

在多声源高度重叠的场景（如多脚步声、同时物体交互和环境声）中，仍可能出现定位误差和混响不一致，需要进一步提升多物体跟踪与层级推理能力。

---

## 396. Hackers or Hallucinators? A Comprehensive Analysis of LLM-Based Automated Penetration Testing

**arXiv ID:** 2604.05719 | [PDF](https://arxiv.org/pdf/2604.05719v1)

**作者:** Jiaren Peng `[一作]` (Sichuan University), Cheng Huang `[通讯]` (Sichuan University)

**通讯引用:** 37051 | [OpenAlex ID](https://openalex.org/A5100678432)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文首次对基于大型语言模型的自动渗透测试（AutoPT）框架进行系统化知识梳理，并在统一基准上对13个主流开源框架与2个基线框架开展大规模实验，分析架构维度与实际性能关系；

**💡 创新点**

创新点在于（1）提出六维分析框架（Agent Architecture、Plan、Memory、Execution、External Knowledge、Benchmarks）对现有框架进行细粒度解构；（2）设计统一的实验平台，确保对比公平；（3）系统归纳实验发现，包括单体代理可与多体架构媲美、记忆管理决定性能、外部知识不一定提升、工具池规模与成功率无正相关等。

**🔧 技术方法**

主要技术包括：多体与单体Agent架构设计；ReAct / RAG / 图/树规划；记忆压缩与外部检索；工具调用与动态调度；对不同LLM（DeepSeek-Chat-v3.2、Claude‑Opus‑4.6、GPT‑5.2、Gemini‑Pro‑3.1、DeepSeek‑Reasoner‑v3.2）进行对照；以及人工日志审阅与定量指标（任务成功率、token消耗、flag召回）。

**📊 数据集**

使用的基准来自XBOW挑战集（消除数据污染），任务涵盖CTF、单机全流程、多机网络、CVE利用与阶段化测试，总计10亿+token。

**📈 对比分析**

比较方法为统一的输入、统一的LLM、统一的工具集与超参数，记录每框架在Easy、Medium、Hard三难度下的成功率、平均token消耗与执行日志完整度。实验显示单体框架在Easy/Medium任务中能达到或超过多体框架的成功率，且在Hard任务中单体框架token消耗更高；外部知识模块往往反而降低成功率；工具池规模扩张不提升成功率。

**⚠️ 局限性**

局限性包括：基准仍无法完全覆盖真实网络环境；实验依赖特定LLM，结果对模型可迁移性有一定依赖；记忆压缩与检索策略需进一步优化以避免信息丢失；外部知识检索质量差异大，导致不稳定；Hallucination仍普遍存在，需更稳健的验证机制。

---

## 397. Conditional Publics: Shared Events and Divergent Meanings in the European Twitter Debate on the Ukraine War

**arXiv ID:** 2604.05800 | [PDF](https://arxiv.org/pdf/2604.05800v1)

**作者:** Corrado Monti `[一作]` (CENTAI), Gianmarco De Francisci Morales `[通讯]` (CENTAI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究欧洲推特用户在乌克兰战争中的讨论，量化其结构性分化、时间演化和议题立场，并提出“条件公众”理论解释不同议题下公众认知的共通性与分裂。

**💡 创新点**

创新点在于：①将社交媒体讨论拆解为“解释性议题”与“实用性议题”，并揭示两类议题下公众关注点和情绪的差异；②引入“条件公众”概念，说明对立社区是否共享同一参考框架取决于议题性质；③通过大规模网络分析与注释，首次系统性比较欧洲各国对立立场的同步性与异质性。

**🔧 技术方法**

使用社交网络分析技术：Leiden 社群检测、结构同质化（homophily）度量、时间序列聚类与 Spearman 相关；自然语言处理与人工标注结合，构建六维议题立场标签；统计检验（Mann‑Whitney U、相关显著性检验）评估分布差异；事件匹配分析通过手工对比推文对应事件的同异。

**📊 数据集**

数据集来源于 Twitter Streaming API 收集的 623,945,097 条乌克兰战争相关推文（2022‑02‑27 至 2022‑10‑12）。通过语言和用户地理位置筛选得到 38,044,266 条推文，进一步通过社区检测和极化评分筛选 1,242 条最具分化的推文作为分析核心。

**📈 对比分析**

方法对比主要通过统计显著性检验和相关性分析来展示：结构同质化随时间上升、活跃用户寿命延长、两大社区在立场上显著不同（p<10⁻⁴）。同步性方面，hawkish 侧在多议题上跨国高度相关，doveish 侧则呈现较低同步性；对事件关注的比较显示解释性议题双方关注不同事件，实用性议题双方关注相同事件但解读不同。

**⚠️ 局限性**

局限性包括：①仅以转推为关注信号，可能混合支持、讽刺或策略曝光；②缺乏对原始推文的全面分析；③人工标注受主观与文化偏见影响；④未检测机器人或协调传播，可能影响同步性解读；⑤平台算法与实时事件驱动特性导致结果对其他社交媒体不一定适用；⑥仅关注极端冲突情境，结果对日常政治讨论的推广性有限。

---

## 398. Near-Field Integrated Sensing, Computing and Semantic Communication in Digital Twin-Assisted Vehicular Networks

**arXiv ID:** 2604.05797 | [PDF](https://arxiv.org/pdf/2604.05797v1)

**作者:** Yinchao Yang `[一作]` (King's College London), Mohammad Shikh-Bahaei `[通讯]` (King's College London)

**通讯引用:** 5769 | [OpenAlex ID](https://openalex.org/A5077634135)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向数字孪生辅助车联网的集成感知、计算与语义通信框架（ISCSC），在近场多用户MIMO环境下实现语义率提升与感知精度协同优化。

**💡 创新点**

创新点包括：①将近场效应、数字孪生与语义通信三大技术整合到同一框架；②设计基于粒子滤波的车辆跟踪与DT构建；③提出混合启发式分配和交替优化算法，并给出感知误差与DT计算负载的直接映射模型。

**🔧 技术方法**

采用的技术包括：MU‑MIMO波束成形、粒子滤波跟踪、语义提取比率优化、线性/对数变换实现的凸化、交替优化与高斯随机化、混合贪婪+模拟退火的车辆分配。

**📊 数据集**

实验使用仿真数据：车辆在100 m×10 m道路段按泊松分布布置，RSU数量为2，天线阵列尺寸为310/3，车辆数目从5至20，所有参数均在表中给出，没有使用真实车联网数据集。

**📈 对比分析**

与贪婪分配、随机翻转贪婪、以及基准MISO ISAC方案比较，ISCSC在语义传输率（≈2.8 bps/Hz vs 1.0 bps/Hz）、角度/距离CRB（从10⁻³°提升到10⁻²°、5×10⁻³ m提升到0.5 m）以及DT坐标误差方面均显著优于基准方案。

**⚠️ 局限性**

局限性包括：未考虑多RSU协同感知与MEC卸载；假设匹配滤波无多径/混叠干扰；线性DT计算模型可能低估极端条件下的能耗；所有结果基于仿真，缺乏实测验证；近场模型在更远距离下可能失效。

---

## 399. Sparse Gain Radio Map Reconstruction With Geometry Priors and Uncertainty-Guided Measurement Selection

**arXiv ID:** 2604.05788 | [PDF](https://arxiv.org/pdf/2604.05788v1)

**作者:** Zhihan Zeng `[一作]` (University of Electronic Science and Technology of China), Zhongpei Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2653 | [OpenAlex ID](https://openalex.org/A5067710600)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究稀疏测量条件下的信号增益无线电地图重建，提出一种同时输出增益图和不确定度图的几何感知网络 GeoUQ-GFNet，并通过该网络实现主动测量选择。

**💡 创新点**

创新点包括：①几何门控前端实现结构先验与稀疏观测的自适应融合；②轻量化 Ghost+Grid‑KAN 编码器，兼顾多尺度特征与非线性拟合；③双头设计同时预测增益与不确定度，后者用于指导主动测量；④构建可控的 UrbanRT‑RM ray‑tracing 基准，系统评估几何、基站布置与采样模式对重建难度的影响。

**🔧 技术方法**

采用 ray‑tracing 生成的语义几何图、Ghost 模块、Grid‑KAN 非线性块、FPN 融合、残差增益预测以及基于 heteroscedastic Gaussian 的不确定度回归；活跃测量策略采用 Top‑K 最大不确定度点选。

**📊 数据集**

使用自制 UrbanRT‑RM 数据集：7 个合成城市场景、8 种基站部署、3 种采样模式（随机、网格、道路约束），在 512×512 网格上提取 128×128 patch，采用 Sionna ray‑tracing 生成 3.5 GHz 下的增益地图，训练时使用 10% 随机稀疏采样。

**📈 对比分析**

与 Efficient‑UNet、ResNet‑UNet、ViT‑UNet、最近邻插值等基线在 RMSE、MAE、误差-不确定度相关性等指标上比较；GeoUQ‑GFNet 在所有场景平均 RMSE 下降约 0.65 dB，误差相关性最高；在主动测量实验中，基于不确定度的查询相较随机查询在 4% 预算下提升约 2.8 dB。

**⚠️ 局限性**

局限性包括：①依赖合成 ray‑tracing 数据，真实环境下的泛化能力未知；②目前仅考虑静态、单频 3.5 GHz 场景；③不确定度估计的置信区间尚未在真实测量中验证；④模型在极大规模场景下的推理与存储成本仍高于传统插值方法。

---

## 400. Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion

**arXiv ID:** 2604.05780 | [PDF](https://arxiv.org/pdf/2604.05780v1)

**作者:** Yu Xue `[一作]` (Xi’an Jiaotong University), Xiaoning Zhang `[通讯]` (Xi’an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种单目语义场景重建框架 VoxSAMNet，能够从单张 RGB 图像恢复完整的 3D 语义体素网格。

**💡 创新点**

核心创新包括：
• Dummy Shortcut for Feature Refinement (DSFR)：针对空体素直接映射到共享虚拟节点，显著减少对稀疏空体素的计算；
• Text‑Guided Image Filter (TGIF)：利用文本提示对 2D 特征进行语义过滤，抑制被丢弃前景类别的干扰；
• Foreground Dropout (FD)：在训练中随机丢弃少数前景标签，缓解长尾类别过拟合并提升泛化能力。

**🔧 技术方法**

采用 Swin Transformer 作为 2D backbone，结合多尺度特征、深度估计、可变形注意力、3D U‑Net 以及 Masked MAE 进行体素完成；还使用语义与几何损失、占用率损失等多任务训练策略。

**📊 数据集**

在 SemanticKITTI（10 训练/20 验证序列，20 类）和 SSCBench‑KITTI‑360（7 训练/验证序列，19 类）两个公开户外基准上进行评估。

**📈 对比分析**

与现有单目、立体及多视角基准比较，VoxSAMNet 在 SemanticKITTI 上 mIoU 达到 18.2%（比前沿单目方法提升约 0.8%），并在 SSCBench‑KITTI‑360 上同样取得领先成绩；推理时间约 284 ms，显著低于多数基线。

**⚠️ 局限性**

局限性：
• 依赖文本提示，若缺乏合适的类别词汇可能影响过滤效果；
• 主要针对户外 LiDAR‑级场景，室内或不同域的迁移能力尚未验证；
• 虽已显著降低对空体素的计算，但对密集前景区域仍需昂贵的可变形注意力，整体算力仍高于极简方法。

---

## 401. Improving Controllable Generation: Faster Training and Better Performance via $x_0$-Supervision

**arXiv ID:** 2604.05761 | [PDF](https://arxiv.org/pdf/2604.05761v1)

**作者:** Amadou S. Sangare `[一作]` (Université Paris-Saclay), Bertrand Luvison `[通讯]` (Université Paris-Saclay)

**通讯引用:** 209 | [OpenAlex ID](https://openalex.org/A5073618814)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对可控图像生成，提出以清洁图像x₀为监督信号的训练方式（x₀‑supervision），并通过对损失进行重加权实现。

**💡 创新点**

创新点包括：① 证明x₀‑supervision相当于去除SNR加权，从而在早期去噪阶段提供更强学习信号；② 统一适用于扩散和流匹配两大范式；③ 引入新指标mAUCC用于客观衡量收敛速度；④ 在多种控制模式（空间对齐与非空间对齐）上实现显著加速。

**🔧 技术方法**

使用技术：扩散/流匹配模型（Stable Diffusion、FLUX.1）、UNet/DiT架构、Adapter网络、x₀‑supervision损失（及其等价的SNR逆权重实现）、实验评估指标（FID、RMSE、mIoU、mAP、F1、mAUCC）。

**📊 数据集**

数据集：MultiGen‑20M（深度控制）、ADE20K（语义分割与Canny边缘）、MS‑COCO（姿态与GLIGEN盒子‑文本控制），以及这些数据集上的官方评测工具。

**📈 对比分析**

比较方法：在相同超参、GPU和批量大小下，与传统的ε‑supervision（以及v‑supervision/ u‑supervision）进行对比。实验表明x₀‑supervision在控制精度、视觉质量与收敛速度上均有提升，mAUCC最高可提升约120%，最终控制精度提升15%–30%，训练时间缩短至原来的一半左右。

**⚠️ 局限性**

局限性：对非空间对齐控制（如GLIGEN）仍需较大批量和显存；不适用于已是x₀‑predictor的基础模型；实验仅覆盖少数控制模式，尚未验证在更广泛场景下的稳健性。

---

## 402. CAKE: Cloud Architecture Knowledge Evaluation of Large Language Models

**arXiv ID:** 2604.05755 | [PDF](https://arxiv.org/pdf/2604.05755v1)

**作者:** Tim Lukas Adam `[一作]` (University of Southern Denmark), Serkan Ayvaz `[通讯]` (University of Southern Denmark)

**通讯引用:** 1433 | [OpenAlex ID](https://openalex.org/A5075890706)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个专门用于评估大型语言模型在云原生软件架构知识方面的基准（CAKE）。

**💡 创新点**

创新点在于：①使用Bloom修订版认知层级（回忆、分析、设计、实现）设计了188道专家验证的问题；②同时评估多项选择（MCQ）与自由回答（FR）两种格式，揭示它们揭示知识的互补性；③探讨了链式思考（+think）和工具使用（+tool）对模型表现的影响。

**🔧 技术方法**

技术手段包括：LLM‑as‑a‑judge（DeepSeek‑R1）对自由回答进行打分；三次运行多数投票评估MCQ；使用链式思考与工具增强实现推理与工具调用；基于Ollama、LM Studio、OpenRouter等多平台推理框架实现模型部署。

**📊 数据集**

使用的数据集为CAKE，包含188道题目（130道MCQ、58道FR），覆盖五个云原生主题（架构模式、质量属性、拆分策略、云部署、技术债务），所有题目均通过四名领域专家的清晰度、正确性及难度评估验证。

**📈 对比分析**

比较方法：在22个模型配置（0.5B–70B参数）中评估MCQ准确率与FR得分。结果显示：MCQ在3B以上模型几乎达到顶峰（最高99.2%）；FR得分随参数规模稳步提升，显示更细微的区分；推理增强（+think）对FR有显著提升，工具增强（+tool）在小模型下导致下降。

**⚠️ 局限性**

局限性：①基准仅覆盖云原生架构，难以推广至传统软件架构；②MCQ答案经常为最长选项，可能被启发式利用；③实现层的MCQ被移除，导致实现知识主要依赖FR；④FR评估依赖单一judge模型，可能带来偏差；⑤专家评估分布偏高导致Krippendorff α几乎为0；⑥缺乏多语言和更广泛主题的覆盖。

---

## 403. An End-to-End Approach for Fixing Concurrency Bugs via SHB-Based Context Extractor

**arXiv ID:** 2604.05753 | [PDF](https://arxiv.org/pdf/2604.05753v1)

**作者:** Zhuang Li `[一作]`, Hongliang Liang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一款名为AutoRepair的LLM驱动工具，实现了无须手工输入、端到端自动检测、定位、修复多种并发 bug 的完整流程。

**💡 创新点**

创新点在于：①使用静态 Happens‑Before Graph（SHBG）进行上下文提取，显著降低 LLM 的信息噪声；②将定位与修复合并为单步 Prompt，使 LLM 能在一次调用中完成定位与补丁生成；③在实验中首次实现对数据竞争、顺序违规、原子性违规及死锁等全部类型并发 bug 的统一自动修复。

**🔧 技术方法**

技术栈包括：JPF 动态检测、基于 WALA 的 SHBG 构造与死锁检测、闭包式方法级上下文提取、GPT‑4（gpt‑4‑turbo‑2024‑04‑09）LLM、Prompt 工程与迭代式错误反馈机制。

**📊 数据集**

评测使用来自 SIR、Pecan、JaConTeBe 的 43 篇并发程序（共 33 个非死锁 bug、10 个死锁 bug），并对每个 bug 采用多次实验以评估修复成功率。

**📈 对比分析**

与 PFIX、DFix 以及 SWE‑agent 等基线比较时，AutoRepair 在非死锁 bug 上的修复成功率为 29/43（≈67%），在死锁 bug 上修复了 4/10；相比 PFIX（22/43）和 DFix（21/43），修复率提升 5–7%；在锁使用上平均只引入 2 把锁，远低于 DFix 的 8.4 把锁；整体执行时间（不计 LLM API 调用）在 1–20 秒范围内，表现出较高的效率与质量。

**⚠️ 局限性**

主要限制包括：缺乏严格的完备性与安全性理论保证；依赖 JPF 的检测覆盖率，某些 bug 仍被漏检；LLM 生成的补丁偶尔会引入新的死锁或功能性错误；上下文提取仍不完全，理论上可能遗漏关键信息，虽在实践中影响极小。

---

## 404. Proceedings 17th Workshop on Programming Language Approaches to Concurrency and Communication-cEntric Software

**arXiv ID:** 2604.05737 | [PDF](https://arxiv.org/pdf/2604.05737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 405. Dialogue Act Patterns in GenAI-Mediated L2 Oral Practice: A Sequential Analysis of Learner-Chatbot Interactions

**arXiv ID:** 2604.05702 | [PDF](https://arxiv.org/pdf/2604.05702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 406. SVC 2026: the Second Multimodal Deception Detection Challenge and the First Domain Generalized Remote Physiological Measurement Challenge

**arXiv ID:** 2604.05748 | [PDF](https://arxiv.org/pdf/2604.05748v1)

**作者:** Dongliang Zhu `[一作]` (Wuhan University), Zitong Yu `[通讯]` (Great Bay University)

**通讯引用:** 5149 | [OpenAlex ID](https://openalex.org/A5062522283)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出并组织了SVC 2026挑战，旨在通过多模态欺骗检测与域泛化遥测生理测量两个任务，统一评估并推动微弱视觉信号建模的稳健性与跨域泛化能力。

**💡 创新点**

创新点在于整合两类任务于同一框架，构建统一的评估协议、标准化数据集与可复现的评测流程；同时引入领域泛化rPPG挑战与跨域欺骗检测，强调模型在真实环境中的迁移与鲁棒性。

**🔧 技术方法**

采用多模态特征提取（如OpenFace、EmotionNet、Mel-spectrogram、OpenSmile）、Transformer与注意力融合、梯度匹配、频域自适应、双分支时序建模、Bures–Wasserstein及时序一致性损失等深度学习技术；并在挑战中提供基线模型与多支参赛队伍的先进方法。

**📊 数据集**

使用公开的多模态欺骗检测数据集（Real‑Life Trials、Bag‑of‑Lies、MU3D、Box‑of‑Lies、DOLOS）以及五个rPPG数据集（UBFC‑rPPG、PURE、BUAA‑MIHR、MMPD、PhysDrive），并通过数据融合与分阶段训练来考察跨域性能。

**📈 对比分析**

与基线对比，多支参赛团队在Accuracy、F1、MAE、RMSE、Pearson相关等指标上取得不同程度的提升，排名靠前的团队在欺骗检测任务中取得约71%准确率，在rPPG任务中实现MAE 3.20、RMSE 8.06、相关系数0.86；但在更严格的跨域阶段性能显著下降。

**⚠️ 局限性**

主要限制在于模型仍对光照、运动、设备差异等噪声敏感，跨域泛化仍不充分；参赛方法在不同任务间迁移能力有限，且大部分方法依赖大量标注数据或复杂多模态预处理，降低了实际部署可行性。

---

## 407. ASSR-Net: Anisotropic Structure-Aware and Spectrally Recalibrated Network for Hyperspectral Image Fusion

**arXiv ID:** 2604.05742 | [PDF](https://arxiv.org/pdf/2604.05742v1)

**作者:** Qiya Song `[一作]` (Hunan Normal University), Shutao Li `[通讯]` (Hunan University)

**通讯引用:** 35078 | [OpenAlex ID](https://openalex.org/A5067097659)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种双阶段的高光谱图像融合网络ASSR-Net，实现高分辨率高光谱图像重建。

**💡 创新点**

创新点在于①基于方向感知的结构捕获模块（VDAE、DAE、AST）有效恢复各向异性空间细节；②利用低分辨率HSI作为谱先验，通过GSRT实现显式谱校正；③将空间增强与谱校正解耦，提升两者兼容性。

**🔧 技术方法**

使用卷积网络、可学习几何旋转、可微雷达变换、双向跨模态交互、Wavelet变换、Transformer注意力、谱引导注意力、双阶段残差结构等技术。

**📊 数据集**

在CAVE、Harvard、Gaofen5等公共数据集上训练与测试，并在Houston数据集上评估分类性能。

**📈 对比分析**

与DHIF‑Net、DSPNet、LRTN、MIMO‑SST、SINet、OTIAS、SRLF等SOTA方法对比，ASSR‑Net在PSNR、SAM、UIQI、SSIM、ERGAS和QNR指标上均取得最高或最接近最高成绩，PSNR提升约1.8 dB、SAM下降约0.4；在分类上平均F1提升7%。

**⚠️ 局限性**

局限性：模型参数较多（约13 M）且推理时延约13.6 ms；对投影方向数与Transformer块数的选择敏感；在极端噪声或极低分辨率条件下的鲁棒性仍待进一步验证。

---

## 408. MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models

**arXiv ID:** 2604.05738 | [PDF](https://arxiv.org/pdf/2604.05738v1)

**作者:** Han Jang `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**通讯引用:** 3679 | [OpenAlex ID](https://openalex.org/A5052023515)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MedLayBench‑V，一个多模态基准，用来评估医学影像报告从专业术语向患者友好语言的语义对齐，并提出 SCGR 管线实现安全简化。

**💡 创新点**

首次提供双注册（专业与患者）医学影像文本对，并通过基于 UMLS 概念与实体的结构化概念驱动精炼（SCGR）保证语义一致性与简化质量。

**🔧 技术方法**

利用 UMLS CUI 映射、SciSpacy NER、MedlinePlus 定义检索生成草稿，再用 Llama‑3.1‑8B 在约束生成框架下进行语法与流畅化。

**📊 数据集**

以 ROCOv2（约80k 图文对）为种子，结合预提取的 CUIs，经过 SCGR 生成患者级文本，形成完整的双注册数据集。

**📈 对比分析**

在零样本图文检索和简化指标上与通用与医学 VLM 对比，SCGR 在保持信息完整度的同时，检索召回率与专家级别相近；BLEU/ROUGE/METEOR 等简化指标保持在70%+水平，阅读指标显著下降到高中水平。

**⚠️ 局限性**

仅限英文、生成语料依赖自动化且缺乏多模态平衡，评测任务相对简单，可能低估专家‑患者语义鸿沟，真实患者语义细微差异尚未充分验证。

---

## 409. BodhiPromptShield: Pre-Inference Prompt Mediation for Suppressing Privacy Propagation in LLM/VLM Agents

**arXiv ID:** 2604.05793 | [PDF](https://arxiv.org/pdf/2604.05793v1)

**作者:** Bo Ma `[一作]`, Weiqi Yan `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文发布了完整的实验和可复现的 Artifact 包，包括多模态、跨域文本和 OCR 数据的隐私保护基准。

**💡 创新点**

创新点在于构建了统一的可复现表征地图、规范化 i2b2 导出模式以及多域、零样本对比基线体系。

**🔧 技术方法**

使用了基于策略配置的去敏化算子 Tπ、路由选择器 Ππ、恢复策略 ρ 等技术，并结合 Ollama、Presidio、spaCy 等框架实现对文本与 OCR 的隐私保护。

**📊 数据集**

数据集覆盖 i2b2、i2b2‑Synthea、TAB、AI4Privacy、PhysioNet、CORD、FUNSD、SROIE 等公开与合成文本/临床/OCR 样本。

**📈 对比分析**

通过对零样本、持续、完整的对照组进行多维指标评估，展示了不同基线在隐私保护与信息保留之间的权衡，并提供了稳定性与延迟报告。

**⚠️ 局限性**

局限在于部分许可数据未公开、部分模型/运行日志未完整执行、对模型版本与内存使用的细节仍未披露，导致部分外部验证与完整基线未能完全实现。

---

## 410. Improving Explanations: Applying the Feature Understandability Scale for Cost-Sensitive Feature Selection

**arXiv ID:** 2604.05790 | [PDF](https://arxiv.org/pdf/2604.05790v1)

**作者:** Nicola Rossberg `[一作]` (University College Cork), Andrea Visentin `[通讯]` (University College Cork)

**通讯引用:** 315 | [OpenAlex ID](https://openalex.org/A5045006994)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过收集两数据集（Telco Customer Churn 与 Body Signals of Smoking）的特征可理解性评分，结合Feature Understandability Scale (FUS) 并将其作为成本引入特征选择，尝试在保持高准确率的前提下提升模型可解释性。

**💡 创新点**

创新点在于首次将 FUS 作为用户可理解性成本直接嵌入特征选择流程，实现准确率与可解释性的共同优化；并展示了可解释性提升并未显著牺牲性能。

**🔧 技术方法**

主要技术包括 FUS 量表评分、过滤式成本敏感特征选择、随机森林、决策树与 SVM 分类器、SHAP 解释器以及文本化解释生成。

**📊 数据集**

使用的数据集为公开的 Telco Customer Churn（预测客户流失）和 Body Signals of Smoking（预测吸烟者）。

**📈 对比分析**

与传统仅依据特征重要性选择特征的模型比较，结果显示两者在测试准确率上相差不大（最高约 0.3%），但共优化模型的平均可理解性成本降低 25–30%，说明可解释性显著提升。

**⚠️ 局限性**

局限性包括样本量小导致评分方差大、受“中立化”倾向影响、特征选择仅采用过滤法且未考虑嵌入式或包装式方法、以及对 one‑hot 编码特征的解释可能导致理解障碍。

---

## 411. Emergent social transmission of model-based representations without inference

**arXiv ID:** 2604.05777 | [PDF](https://arxiv.org/pdf/2604.05777v1)

**作者:** Silja Keßler `[一作]` (University of Tübingen), Charley M. Wu `[通讯]` (Technical University Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

本研究探究在不进行心理推理（mentalizing）的前提下，利用最小化的社会学习策略（决策偏置与价值塑造）如何通过强化学习（RL）间接传递高阶环境表征，并评估其对模型自由（MF）与模型基于（MB）代理学习的影响。

**💡 创新点**

创新点在于提出一种仅基于社会线索（如专家行动或价值加成）的简单学习机制，即使不进行对他者心理状态的推断，也能通过模型基于RL的内部规划实现对价值和转移结构的高阶表征迁移。

**🔧 技术方法**

采用强化学习技术（Q‑learning、Dyna‑Q）、两种社会学习策略（决策偏置、价值塑造）、差分进化超参数优化、贝尔曼方程价值评估以及相关性分析等方法。

**📊 数据集**

使用随机旋转与排列的10×10网格世界（由四个5×5象限构成，产生6144种独特布局），随机分配四种奖励值（0、25、50、75），并在每次奖励观测中加入噪声，共进行1,000次仿真实验。

**📈 对比分析**

通过比较六种学习模型（AS‑MF、AS‑MB、DB‑MF、DB‑MB、VS‑MF、VS‑MB）在训练和测试阶段的累计奖励、价值和信念相关性。结果显示模型基于的社会学习者（尤其VS‑MB与DB‑MB）在测试阶段表现最佳，超越纯粹的AS‑MB；而在模型自由下的决策偏置策略表现脆弱。

**⚠️ 局限性**

局限性包括：仅考察奖励与起点变化，未对转移动态或墙壁布局进行修改；专家示范仅为被动演示，未探究教学式示范的影响；实验仅基于仿真，缺乏在人类行为实验中的验证。

---

## 412. Beyond the Beep: Scalable Collision Anticipation and Real-Time Explainability with BADAS-2.0

**arXiv ID:** 2604.05767 | [PDF](https://arxiv.org/pdf/2604.05767v1)

**作者:** Roni Goldshmidt `[一作]` (Nexar AI), Hernan Matzner `[通讯]` (Nexar AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 BADAS‑2.0 里通过主动挖掘、Atlas 定位与人类标注，将数据集从 40k 扩大至 178k，重新训练 V‑JEPA2 ViT‑L 以实现更精准的碰撞预测，随后将模型蒸馏到 86M、22M 规模以实现边缘实时推理，并添加热力图与 VLM 生成可解释的文字提示。

**💡 创新点**

三大创新点：① 构建 10 类长尾碰撞基准并利用主动学习快速填补稀有场景；② 结合域适应 SSL 与知识蒸馏，将大模型性能迁移至小模型；③ 在同一框架中同时提供视觉热图与语言解释，实现端到端可解释性。

**🔧 技术方法**

技术涵盖 V‑JEPA2 masked spatiotemporal 预训练、ViT‑L/ViT‑S 结构、两阶段蒸馏（BCE+KL+特征匹配）、Flash‑Lite 轻量化、FlashAttention 关闭、QLoRA 微调 Qwen3‑VL、训练‑free attention 热图提取等。

**📊 数据集**

主要数据集为 Nexar Dashcam 视频，包含 178.5k 帧标注（约 2M 训练窗口），并构建 888 帧长尾测试集；同时在 DAD、DoTA、DADA‑2000 等公开基准上进行复现评测。

**📈 对比分析**

在 Kaggle 竞赛上 mAP 提升至 0.940（比原 0.925 提升 1.5%），FPR 降至 4.6%（降 58%）；在 10 组长尾基准上各子组 F1 与 EWR 均显著提升，尤其动物组 EWR 从 66.1% 提升至 78.9%；在边缘设备上 22M 模型 2.8 ms 推理，满足 125 ms 预算。

**⚠️ 局限性**

局限包括动物场景 EWR 仍低于 80%，且系统仅输出概率并不给出对应的避险轨迹；热图为 patch‑级别，缺乏像素级分割；目前的可解释性仍需依赖外部 VLM，未直接从编码器内部生成文字。

---

## 413. Physics-Informed Neural Optimal Control for Precision Immobilization Technique in Emergency Scenarios

**arXiv ID:** 2604.05758 | [PDF](https://arxiv.org/pdf/2604.05758v1)

**作者:** Yangye Jiang `[一作]` (Zhejiang University), Daofei Li `[通讯]` (Zhejiang University)

**通讯引用:** 925 | [OpenAlex ID](https://openalex.org/A5062982668)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8d10c613-917e-4880-9716-17789f50e119` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了基于 PicoPINN 的精确可控碰撞技术（PIT）的神经最优控制框架，完成了从数据集构建到仿真与小型车辆实测的完整评估

**💡 创新点**

提出了 PicoPINN：通过知识蒸馏、层级聚类和参数重构实现的极致压缩物理信息神经网络；以及双层神经 OCP 结构：上层虚拟决策+下层耦合 MPC；并创建了专门针对 PIT 的 Scenario Dataset

**🔧 技术方法**

使用物理信息神经网络（PINN）、知识蒸馏、层级聚类、参数重构、差分可微优化、PyTorch+CasADi 复合求解器、LSTM 等序列网络和车辆动力学模型

**📊 数据集**

构造了 PIT Scenario Dataset，包含 2048 个真实案例衍生的仿真场景（直线/弯道、不同摩擦系数、障碍物分布）以及相应的高保真碰撞力数据

**📈 对比分析**

通过对比 PicoPINN 与原 PINN、4DOF 模型；对比包含上层规划与仅耦合 MPC 两种结构；在高保真仿真中成功率从 63.8% 提升至 76.7%，平均航向误差从 0.114 rad 降至 0.106 rad，边界违背率与二次碰撞率显著降低；在半比例车辆实验中 4/5 试验成功，验证了控制可行性

**⚠️ 局限性**

主要局限在于仍存在从仿真到真实车辆的差距（传感误差、通信延迟、接触非线性未完全建模）、对弱 PIT 激励场景的鲁棒性不足、对高速度或复杂障碍环境的推广性待验证

---

## 414. On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors

**arXiv ID:** 2604.05743 | [PDF](https://arxiv.org/pdf/2604.05743v1)

**作者:** Amit Vaisman `[一作]` (Technion - Israel Institute of Technology), Raz Lapid `[通讯]` (Deepkeep)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了基于逆向通道编码（RCC）的扩散模型在图像压缩中的抗位点错误（bit‑flip）性能，并提出了一种更鲁棒的 Turbo‑DDCM 变体；

**💡 创新点**

创新点在于：①证明 RCC‑扩散压缩方法在位点错误下比传统与学习型压缩更具鲁棒性；②设计了 Robust Turbo‑DDCM，通过将选定原子单独编码而非使用词典索引，显著提升对位点错误的容忍度；

**🔧 技术方法**

主要技术包括：逆向通道编码（RCC）框架、Denoising Diffusion Probabilistic Models（DDPM）、Denoising Diffusion Codebook Model（DDCM）、Turbo‑DDCM 以及对比实验中使用的二进制对称信道（BSC）模拟；

**📊 数据集**

使用了 Kodak24（512×512）和 DIV2K（512×512）图像数据集进行实验；

**📈 对比分析**

对比了 JPEG、BPG、ILLM、StableCodec 等传统/学习型压缩算法，实验结果显示 Robust Turbo‑DDCM 在 BER 10⁻⁴ 至 10⁻³ 级别时保持 PSNR 与 FID 近乎不降，且无文件解码失败，而其他方法在 BER 10⁻⁵–10⁻⁴ 级别就出现显著性能衰减；

**⚠️ 局限性**

局限性包括：实验仅考虑独立位点错误的 BSC，未涵盖突发或结构化错误；部分对比方法使用熵编码导致对位点错误更敏感，难以完全归因于压缩表示本身；以及 Robust Turbo‑DDCM 在提高鲁棒性的同时略微牺牲了压缩效率。

---

## 415. Graph Topology Information Enhanced Heterogeneous Graph Representation Learning

**arXiv ID:** 2604.05732 | [PDF](https://arxiv.org/pdf/2604.05732v1)

**作者:** He Zhao `[一作]` (Nanyang Technological University), Chunyan Miao `[通讯]` (Nanyang Technological University)

**通讯引用:** 22952 | [OpenAlex ID](https://openalex.org/A5100382077)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种名为 ToGRL 的异构图表示学习框架，通过先学习高质量的任务相关图结构，再用该结构进行节点表征学习，并在下游任务中引入 Prompt Tuning 以提升效果。

**💡 创新点**

创新点包括：① 用两阶段的图结构学习（先构造新图再学习表征）显著降低内存消耗；② 通过随机游走提取潜在拓扑信息并构建光滑稠疏图；③ 在下游任务中加入 Prompt Tuning 机制，弥合预训练表示与微调之间的差距。

**🔧 技术方法**

采用了 GCN/GAT 作为表征学习背骨，Node2Vec 随机游走 + Skip‑gram 进行拓扑嵌入，Graph Signal Processing 的平滑性思想构建新图，知识蒸馏与 Jensen‑Shannon 散度实现表征优化，Prompt Tuning 作为线性解码器的前缀。

**📊 数据集**

在五个真实异构图数据集（ACM、ACM2、DBLP、IMDB、PubMed）上进行实验，分别进行节点分类和链路预测任务。

**📈 对比分析**

与 Transh、Node2Vec、Metapath2Vec、SLAPS、HGSL、GTN、HGT、SimpleHGN、CKD、SeHGNN 等基线相比，ToGRL 在所有节点分类数据集上获得最高 Macro/F1 与 Micro/F1 分数，在链路预测上实现最高 AUC；同时内存占用显著低于传统 GSL 方法。

**⚠️ 局限性**

局限性在于：① 需要手动设定 k 等超参数；② 仅在中小规模数据集上验证，尚未证明在极大异构图上的可扩展性；③ 目前只支持线性下游解码，可能对更复杂任务的适应性有限。

---

## 416. Can Large Language Models Reinvent Foundational Algorithms?

**arXiv ID:** 2604.05716 | [PDF](https://arxiv.org/pdf/2604.05716v1)

**作者:** Jian Zhao `[一作]` (Xiongan AI Institute), Tianxing He `[通讯]` (Tsinghua University)

**通讯引用:** 1116 | [OpenAlex ID](https://openalex.org/A5051747323)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了“Unlearn-and-Reinvent”框架，先用GRPO方法对预训练LLM进行算法知识遗忘，再让其在无/有提示的条件下重新发明10种基础算法。

**💡 创新点**

将知识遗忘与算法重发明结合，提出可用来评估LLM创新能力的系统；使用生成式验证器和测试时强化学习提升重发明成功率；揭示并缓解“思维坍塌”现象。

**🔧 技术方法**

采用Group Relative Policy Optimization（GRPO）进行在线遗忘；LLM-as-a-judge奖励设计；Python interpreter交互与生成式验证器；测试时强化学习；多层提示层级。

**📊 数据集**

使用10种基础算法的描述与测试用例；对话数据集用于构造遗忘集、保留集；LiveCodeBench、AIME25、BFCL评估集；DeepSeek-V3.2作为判定器；实验基于三大开源模型。

**📈 对比分析**

在三模型、三提示层级下评估重发明成功率（RSR）；最强模型无提示50%成功，提示1提升至70%，提示2提升至90%；Strassen在提示2下通过测试时RL成功；生成式验证器提升成功率，缺失时出现思维坍塌。

**⚠️ 局限性**

遗忘是后处理，无法保证完全去除知识；实验仅覆盖10种基础算法，难以推广至更广阔的科学发现任务；可能仍存在残留知识影响重发明结果。

---

## 417. In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting

**arXiv ID:** 2604.05715 | [PDF](https://arxiv.org/pdf/2604.05715v1)

**作者:** Wenhui Xiao `[一作]` (Queensland University of Technology), Leo Lebrat `[通讯]` (Queensland University of Technology)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5055226360)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个训练框架，能够可靠地利用单目深度先验提升3D Gaussian Splatting的渲染质量。

**💡 创新点**

创新点包括：①引入深度不一致掩码（DIM）实现对多视角不一致区域的选择性尺度不变深度约束；②加入梯度对齐损失（GAL）以捕捉细节级别的深度变化，从而解决尺度模糊和深度噪声问题。

**🔧 技术方法**

采用了单目深度估计模型（DepthAnything V2、MoGe-2、UniDepth V2等）、虚拟立体检验深度一致性、尺度不变深度损失、梯度对齐损失，以及3D Gaussian Splatting的差分渲染优化。

**📊 数据集**

使用了ScanNet++、MipNeRF 360和TanksAndTemples这三个真实场景数据集进行评估。

**📈 对比分析**

与3dgs/2dgs基线、ℒ_sid、DNGaussian、SparseGS等方法在低/中等视角密度下进行对比，平均PSNR提升≥0.47dB，且在不同深度先验下保持稳定改进，显示出更高的渲染质量和几何准确性。

**⚠️ 局限性**

局限性包括：对极低视角密度或深度先验严重错误时仍易受影响；梯度对齐损失在极度纹理丰富的场景中可能仍出现细节损失；对不同相机内参和域迁移的鲁棒性还有待进一步提升。

---

## 418. QA-MoE: Towards a Continuous Reliability Spectrum with Quality-Aware Mixture of Experts for Robust Multimodal Sentiment Analysis

**arXiv ID:** 2604.05704 | [PDF](https://arxiv.org/pdf/2604.05704v1)

**作者:** Yitong Zhu `[一作]` (Hong Kong University of Science and Technology), Yuyang Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2293 | [OpenAlex ID](https://openalex.org/A5100409330)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出连续可靠性光谱框架并设计质量感知混合专家模型（QA‑MoE），用于在多模态情感分析中同时处理噪声与模态缺失，提升模型在动态不确定条件下的鲁棒性。

**💡 创新点**

创新点包括：①把噪声与缺失统一到连续可靠性光谱；②利用自监督的阿勒特里奇不确定性动态量化模态可靠性；③将可靠性得分直接嵌入MoE路由，使专家激活与输入质量相关，形成 One‑Checkpoint‑for‑All 能力。

**🔧 技术方法**

采用技术：多模态概率特征建模、阿勒特里奇不确定性估计、质量感知路由的混合专家（GLU‑based），双分支预测（值与不确定性），Spectrum‑aware 动态数据增强与负对数似然优化。

**📊 数据集**

使用数据集：CMU‑MOSI、CMU‑MOSEI、IEMOCAP 以及跨任务的 MIntRec，用于情感与意图识别任务。

**📈 对比分析**

与多种基线（TFN、LMF、MulT、MISA、MMIM、EMOE、MMA、MCTN、MMIN、IMDER、MoMKE、PaSE、SAM‑LML 等）在完整、随机缺失、噪声、混合三种协议下进行对比。QA‑MoE 在标准任务上提升 4–8% 以上准确率，在噪声/缺失情形下保持高稳健性，且单一训练模型即可在整个连续可靠性光谱上表现良好，体现 One‑Checkpoint‑for‑All。

**⚠️ 局限性**

局限性：质量信号通过自监督学习，缺乏对特定噪声类型（如模糊、遮挡）的可解释性；MoE 路由机制增加计算与参数开销；在极端边缘情况下，质量估计可能不够精确，导致专家激活误判。

---

## 419. Realizing Planar Linkages in Polygonal Domains

**arXiv ID:** 2604.05786 | [PDF](https://arxiv.org/pdf/2604.05786v1)

**作者:** Thomas Depian `[一作]` (Technische Universitaet Wien), André Schulz `[通讯]` (FernUniversitaet)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

研究在多边形域中实现平面连杆的可行性，证明其在一般多边形域下属于 ∃R‑hard，并给出单位长度连杆和线性连杆的复杂度；同时为三边线性连杆在凸多边形中提供线性时间算法。

**💡 创新点**

首次将连杆实现问题引入多边形障碍域，提出 ∃R 表达式求解上界，并通过从 Grid Tiling、Planar Monotone 3‑Sat 的构造证明了该问题的高难度；对三边线性连杆的线性算法展示了特殊情况下的可解性。

**🔧 技术方法**

使用存在性实数理论（∃R）公式化、参数化复杂度分析、几何构造（如三角形通道、网格填充、隧道/弯折/位移器等）以及多边形域内的几何约束检测。

**📊 数据集**

无实验数据集，全部为理论证明与构造。

**📈 对比分析**

通过复杂度归约证明问题为 ∃R‑完全/NP‑硬；算法方面，针对三边线性连杆给出 O(n_P) 时间的解决方案，证明在凸多边形内可在线性时间完成。

**⚠️ 局限性**

局限在于：对四边或更多边的线性连杆仅给出困难性结果；凸多边形的线性算法无法直接推广到有凹点或孔洞的多边形；多边形域的孔洞是困难构造的重要组件，缺乏对简单多边形的硬度证明。

---

## 420. RHVI-FDD: A Hierarchical Decoupling Framework for Low-Light Image Enhancement

**arXiv ID:** 2604.05781 | [PDF](https://arxiv.org/pdf/2604.05781v1)

**作者:** Junhao Yang `[一作]` (Jilin University), Chunguo Wu `[通讯]` (Jilin University)

**通讯引用:** 2119 | [OpenAlex ID](https://openalex.org/A5007397249)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种层次解耦框架 RHVI‑FDD，用于解决低光图像中的色彩失真、噪声与细节共存问题。

**💡 创新点**

创新点在于将宏观层面的鲁棒 RHVI 变换与微观层面的频域解耦（FDD）结合，实现对亮度-色度耦合和噪声-细节耦合的分离与分块处理。

**🔧 技术方法**

采用了 RHVI 变换、亮度修正模块 IRM、二维离散余弦变换（DCT）、三频段专家网络（GCM、DRG、ANSU）以及自适应通道门控融合 ACGF 等技术。

**📊 数据集**

主要使用了 LOL、LOLv2、SICE、Sony‑Total‑Dark、SID 等低光增强数据集，并在多套无对照真实数据集上验证泛化能力。

**📈 对比分析**

在 PSNR/SSIM/LPIPS/NIQE 等指标上与多种 SOTA 方法对比，RHVI‑FDD 在 LOLv2‑Real 等数据集上实现了 28.56 dB / 0.893 / 0.078 的最佳成绩，显著优于其他主流模型。

**⚠️ 局限性**

局限在于极低光场景下仍可能残留噪声，且在极细腻纹理处需要更精细的频段划分。

---

## 421. PhageBench: Can LLMs Understand Raw Bacteriophage Genomes?

**arXiv ID:** 2604.05775 | [PDF](https://arxiv.org/pdf/2604.05775v1)

**作者:** Yusen Hou `[一作]` (Hong Kong University Of Science And Technology (Guangzhou)), Yanlin Zhang `[通讯]` (Hong Kong University Of Science And Technology (Guangzhou))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了 PhageBench 基准，用于评估 LLM 在原始噬菌体基因组理解与注释的能力。

**💡 创新点**

首次将 LLM 与噬菌体基因组分析对接，设计三阶段五任务的多任务评估，并发现推理型 LLM 在识别与宿主预测上有显著优势。

**🔧 技术方法**

使用大语言模型（GPT‑4o‑mini、LLaMA‑4、GPT‑5.2、Gemini‑3‑flash 等）在零射击+链式思考的设置下进行评测。

**📊 数据集**

使用公开噬菌体基因组构建的 5,600 条样本，覆盖筛选、质量控制与表型注释等任务。

**📈 对比分析**

对比 8 种 LLM，最优模型 Gemini‑3‑flash 平均准确率 56.33%，远高于随机基准；但在完整性评估与功能定位等长程推理任务中表现不足。

**⚠️ 局限性**

受限于 50 kb 长度上限、BPE 词法适配度低以及模型潜在训练集泄漏，导致对长序列、细粒度功能推断及多模态对齐等方面存在瓶颈。

---

## 422. Generative Retrieval Overcomes Limitations of Dense Retrieval but Struggles with Identifier Ambiguity

**arXiv ID:** 2604.05764 | [PDF](https://arxiv.org/pdf/2604.05764v1)

**作者:** Adrian Bracher `[一作]` (Vienna University of Economics and Business), Svitlana Vakulenko `[通讯]` (Vienna University of Economics and Business)

**通讯引用:** 398 | [OpenAlex ID](https://openalex.org/A5027138711)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估生成式检索（Generative Retrieval, GR）在原始 LIMIT 数据集上能否突破密集检索（Dense Retrieval, DR）的“向量瓶颈”限制，并通过改造的数据集 LIMIT-H、LIMIT-HS 探讨 GR 在语义歧义下的鲁棒性。

**💡 创新点**

证明 GR 能在无任何专门训练的情况下实现几乎完美检索（R@2≈0.99）并指出其核心瓶颈是生成的文档标识符（docid）存在歧义，导致难以区分相关与硬负样本。

**🔧 技术方法**

使用预训练的生成式模型 SEAL 与 MINDER（基于 BART‑large）以及经典 BM25、各种最新 DR 基线（E5‑Mistral、GritLM、Promptriever、Qwen3、Gemini、GTE‑ModernColBERT 等）进行对照实验，并改进 SEAL 的 Beam Search 解码策略。

**📊 数据集**

使用原始 LIMIT（50k 文档、1k 查询）及其改造版本 LIMIT‑H（+46 负样本）和 LIMIT‑HS（+92 负样本）作为评测数据集。

**📈 对比分析**

在 LIMIT 上，GR（SEAL BEAM 及 MINDER）显著优于 DR 与 BM25，R@2 达到 0.917–0.988；在 LIMIT‑H 与 LIMIT‑HS 上，所有模型性能急剧下降，GR 仅能维持 0.43–0.65 R@2，表明 GR 在语义歧义环境中仍易失效。

**⚠️ 局限性**

GR 的主要局限是对生成标识符的歧义处理不足：在高词汇重叠的硬负样本中，模型无法产生唯一且相关的 docid，导致检索无法区分正负样本；此外，默认解码策略会累积低置信度 n‑gram，进一步削弱性能。

---

## 423. Identifying Influential N-grams in Confidence Calibration via Regression Analysis

**arXiv ID:** 2604.05757 | [PDF](https://arxiv.org/pdf/2604.05757v1)

**作者:** Shintaro Ozaki `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1634 | [OpenAlex ID](https://openalex.org/A5102396915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM推理过程中的语言表达对模型置信度的影响，并通过抑制特定n-gram实现置信度校准。

**💡 创新点**

通过回归分析识别出导致过度自信的具体短语，并证明其因果关系，实现无训练的提示式置信度校准。

**🔧 技术方法**

采用Lasso回归、逻辑回归、强制解码置信度计算、n-gram特征抽取与抑制等技术。

**📊 数据集**

使用5个公开多项选择QA数据集（MathQA、MMLU STEM、HellaSwag、RACE、CosmosQA）。

**📈 对比分析**

以ECE、ACE、Brier Score、Accuracy、AUC-ROC等指标评估9款LLM，抑制n-gram后显著降低ECE/ACE，且准确率保持或提升。

**⚠️ 局限性**

仅针对n-gram进行抑制，对模型内部机制影响有限；部分模型（如Llama）效果不佳，且不涵盖所有导致自信的语言模式。

---

## 424. Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP

**arXiv ID:** 2604.05724 | [PDF](https://arxiv.org/pdf/2604.05724v1)

**作者:** Yusung Ro `[一作]` (KAIST), Junmo Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过稀疏自编码器（SAE）对CLIP视觉编码器的内部表示进行解释，并提出信息范围（information scope）这一新维度，衡量特征在空间上聚合证据的范围。

**💡 创新点**

创新点在于：1）引入Contextual Dependency Score（CDS）量化特征在小范围上下文变动下的空间不稳定性，从而区分局部信息与全局信息；2）通过Shifted Context Cropping（SCC）实验揭示outlier token位置极度不稳定；3）将低CDS与高CDS特征分别对应于局部与全局信息，并在多任务下验证其功能差异。

**🔧 技术方法**

技术方法包括：稀疏自编码器（BatchTopK）、Earth Mover's Distance（EMD）来计算CDS、Shifted Context Cropping（SCC）进行上下文扰动、线性探测器（linear probes）在分类、语义分割与深度估计任务中评估特征组的贡献。

**📊 数据集**

使用的数据集：ImageNet‑1K（训练与验证）用于SAE训练与CDS计算；ADE20K用于语义分割；NYU‑Depth V2用于单目深度估计。

**📈 对比分析**

比较方法：在不同CLIP模型（B/16、L/14、L/14‑336px）下，将低CDS或高CDS特征组从原始嵌入中去除，随后用线性探测器评估ImageNet Top‑1、ADE20K mIoU和NYU‑Depth RMSE。实验表明：①去除高CDS特征往往提升分类精度；②去除低CDS特征对密集预测任务（分割、深度）影响更大，表明低CDS特征承载局部信息；③在更大模型或更高分辨率时，高CDS特征的作用逐渐显著。

**⚠️ 局限性**

局限性：①CDS阈值选择依赖经验且可能随模型变动；②只关注空间不稳定性，未考虑语义层面的细粒度差异；③线性探测器评估仅捕获可线性可分的信息，可能低估非线性特征贡献；④实验集中于CLIP ViT，缺乏对其他视觉模型的验证。

---

## 425. Dialogue based Interactive Explanations for Safety Decisions in Human Robot Collaboration

**arXiv ID:** 2604.05896 | [PDF](https://arxiv.org/pdf/2604.05896v1)

**作者:** Yifan Xu `[一作]` (University of Manchester), Clara Cheung `[通讯]` (University of Manchester)

**通讯引用:** 1482 | [OpenAlex ID](https://openalex.org/A5053253133)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出并实现了基于对话的安全导向解释框架，将解释与约束驱动的安全评估紧耦合，支持因果、对比和有界逆因果查询。

**💡 创新点**

创新点在于把解释视为安全控制的操作接口，使得解释直接从实时决策轨迹中提取，并且在保持安全保障的前提下进行有限的逆向推理。

**🔧 技术方法**

使用约束评估机制、实时安全状态记录、规则式安全控制、对话管理器以及有限的逆向推理技术。

**📊 数据集**

未使用公开数据集，采用结构化仿真工作场景（移动操作器、工人、叉车）进行演示。

**📈 对比分析**

未与其他方法做系统性对比，主要评估解释生成与安全控制的耦合不会增加显著计算负担，满足实时性需求。

**⚠️ 局限性**

局限包括缺乏不确定性建模、用户实验评估以及多机器人/多代理的扩展研究。

---

## 426. HybridKV: Hybrid KV Cache Compression for Efficient Multimodal Large Language Model Inference

**arXiv ID:** 2604.05887 | [PDF](https://arxiv.org/pdf/2604.05887v1)

**作者:** Bowen Zeng `[一作]` (Zhejiang University), Huan Li `[通讯]` (Zhejiang University)

**通讯引用:** 16585 | [OpenAlex ID](https://openalex.org/A5100319241)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对多模态大型语言模型的混合KV缓存压缩框架，能够在保持推理质量的前提下显著降低GPU内存占用并提升解码速度。

**💡 创新点**

创新点在于通过文本中心稀疏度对注意力头进行静态/动态两类划分，采用分层预算分配，并为两类头分别设计了静态剪枝和分块检索两种压缩策略，从而实现更细粒度、更高效的KV压缩。

**🔧 技术方法**

使用的技术包括文本中心注意力稀疏度度量、分层（头类型层和头级别）预算分配、静态头的文本优先剪枝、动态头的分块检索以及FlashAttention等实现细节。

**📊 数据集**

在多模态图像与视频评测集上进行实验，主要使用MileBench和LMMs‑Eval两套基准。

**📈 对比分析**

与SnapKV、LOOK‑M、MadaKV、SparseMM等现有方法对比，所提框架在仅使用10% KV缓存时实现了最高可达7.9倍的内存压缩和1.52倍的解码加速，且生成质量与全缓存相当甚至更优。

**⚠️ 局限性**

局限性包括：①基于阈值的头分类可能随模型规模或域变化而失稳；②方法仅关注推理时的压缩，未结合训练或微调；③评估范围主要限于图像/视频任务，未验证对其他模态或极长文本的适用性。

---

## 427. Mechanistic Circuit-Based Knowledge Editing in Large Language Models

**arXiv ID:** 2604.05876 | [PDF](https://arxiv.org/pdf/2604.05876v1)

**作者:** Tianyi Zhao `[一作]` (University of Virginia), Chen Chen `[通讯]` (University of Virginia)

**通讯引用:** 496318 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于机制电路的知识编辑框架 MCircKE，能够在多跳推理中精准更新知识并克服传统方法的“推理缺口”。

**💡 创新点**

通过 EAP‑IG 精确识别推理路径并在这些路径上进行低秩结构化更新，首次实现了“映射‑适配”式的编辑方法，显著提升多跳推理性能。

**🔧 技术方法**

采用边缘归因集成梯度（EAP‑IG）、拓扑剪枝（top‑K）与基于低秩矩阵的参数调整（LoRA‑style）来定位并修改推理电路。

**📊 数据集**

在 MQuAKE‑3K 这一多跳知识编辑基准上进行实验，并在 GPT‑2 Large/XL 与 GPT‑J 三大模型上进行评测。

**📈 对比分析**

与 LoRA、ROME、MEMIT、WISE、AlphaEdit、CaKE 等基线相比，MCircKE 在 2/3/4 跳准确率上分别提升 15–20% 以上，整体多跳准确率最高，且单跳准确率保持竞争力。

**⚠️ 局限性**

目前仅验证了单次编辑效果，缺乏多次连续编辑的可扩展性；EAP‑IG 的梯度计算成本高，且固定的稀疏阈值可能不适用于所有事实。

---

## 428. Joint Knowledge Base Completion and Question Answering by Combining Large Language Models and Small Language Models

**arXiv ID:** 2604.05875 | [PDF](https://arxiv.org/pdf/2604.05875v1)

**作者:** Yinan Liu `[一作]` (Northeastern University), Bin Wang `[通讯]` (Northeastern University)

**通讯引用:** 18428 | [OpenAlex ID](https://openalex.org/A5058724658)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了JCQL框架，联合完成知识库补全(KBC)与知识库问答(KBQA)，实现两任务相互强化

**💡 创新点**

首次将大型语言模型(LLM)的推理路径与小型语言模型(SLM)的知识补全能力融合，通过迭代交替更新实现互补

**🔧 技术方法**

使用LLM（如GPT‑4o‑mini）作为推理代理，SLM（如T5‑small、BART‑small、Flan‑T5‑small）训练KBC模型，并在LLM推理中嵌入SLM预测动作；采用增量微调与经验回放

**📊 数据集**

WebQuestionSP（WebQSP）与Complex WebQuestion（CWQ），背景知识库为Wikidata子集

**📈 对比分析**

与13种SOTA KBC方法（TransE、DistMult、ComplEx、RotatE、SimKGC等）以及多种KBQA基线（CoT、SC、ToG、PoG、LMP、PDRR、GoG等）对比，JCQL在所有评测指标（MRR、Hits@1/3/10及Hits@1）均显著领先，尤其在KBQA上取得最高Hits@1

**⚠️ 局限性**

1）SLM上下文仅基于头实体的一跳邻居，可能不足；2）KBQA中LLM与KB的整合方式仍可改进；3）仅采用简单经验回放，缺乏更高级增量学习策略

---

## 429. When Do We Need LLMs? A Diagnostic for Language-Driven Bandits

**arXiv ID:** 2604.05859 | [PDF](https://arxiv.org/pdf/2604.05859v1)

**作者:** Uljad Berdica `[一作]` (University of Oxford), Manuela Veloso `[通讯]` (J.P. Morgan AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在同时包含文本与数值特征的上下文环境中，Contextual Multi‑Armed Bandits（CMAB）的决策策略，并提出了利用LLM Process进行不确定性估计的LLMP‑UCB算法，系统比较了LLM驱动与传统数值bandit方法的性能与成本。

**💡 创新点**

创新点在于：①通过对同一目标多次查询LLM，构造“LLM Process”来获取置信区间，实现可量化的不确定性估计；②结合Matryoshka嵌入，使嵌入维度成为可控的探索‑利用杠杆；③提出基于嵌入几何的诊断准则，帮助实践者在LLM与数值bandit之间做出成本‑性能权衡。

**🔧 技术方法**

使用的技术包括：LLM Process UCB、LLM‑Bandit、CGPUCB、LinUCB、Thompson Sampling；嵌入技术有 Dense 与 Matryoshka；多轮 prompt 设计与重复推理；OpenAI Gym 及自制模拟环境；评估指标为累计回报、标准差与样本效率。

**📊 数据集**

使用的数据集包括：合成电影推荐数据（多种线性与非线性奖励函数）；Banking77（77类意图分类）；TREC Coarse/Fine（分别为 6 类与 50 类文本检索任务）。

**📈 对比分析**

通过对比累计回报、标准差与样本效率，实验发现：在语义丰富且奖励高度非线性的任务（如 fextract、fLLM）中，LLMP‑Joint 取得最低回报；在纯线性数值任务中，传统 LinUCB 速度最快；LLM‑Bandit 在纯语言奖励上优于数值 bandit；当动作空间高度相似且维度大时，数值 bandit 与 LLM 方法竞争力相当，且 LLMP‑Joint 成本显著更高。

**⚠️ 局限性**

局限性包括：LLMP‑Joint 受限于 prompt 长度，导致在高维或语义相近的动作空间（如 Banking77）中成本高且性能下降；LLM 的不确定性估计依赖多次推理，计算开销大；数值 bandit 在极度非线性或复杂语言任务中可能无法捕捉深层语义；实验仅基于有限的嵌入模型，未涵盖所有可能的 LLM 或嵌入方案。

---

## 430. Reading Between the Pixels: An Inscriptive Jailbreak Attack on Text-to-Image Models

**arXiv ID:** 2604.05853 | [PDF](https://arxiv.org/pdf/2604.05853v1)

**作者:** Zonghao Ying `[一作]` (Beihang University), Xianglong Liu `[通讯]` (Beihang University)

**通讯引用:** 12595 | [OpenAlex ID](https://openalex.org/A5024067284)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了‘写实文字破解’（Inscriptive Jailbreak）攻击以及Etch框架，实现在黑盒条件下让T2I模型生成含有有害文本的图像。

**💡 创新点**

创新点在于将攻击提示拆解为语义隐蔽层、视觉定位层和字体编码层，并使用VLM诊断的零阶优化循环，实现高效、可迭代的文本渲染逃逸。

**🔧 技术方法**

技术手段包括结构化提示分解、LLM与VLM协同生成与评估、零阶语义优化循环、OCR提取与GPT-4o多模态评判。

**📊 数据集**

使用的评测数据集为AdvBench子集（50条）和MaliciousInstruct（100条）这两个LLM攻击数据集。

**📈 对比分析**

与7款主流T2I模型及多类基线（直接、模板、DACA、SneakyPrompt、ReNeLLM等）进行对比，Etch平均攻击成功率达65.6%（最高91%），比最强基线高出约30个百分点。

**⚠️ 局限性**

局限性包括仅针对英文文本，VLM诊断单一模型，缺乏多语言和多模态的标准化评测框架。

---

## 431. Modeling Patient Care Trajectories with Transformer Hawkes Processes

**arXiv ID:** 2604.05844 | [PDF](https://arxiv.org/pdf/2604.05844v1)

**作者:** Saumya Pandey `[一作]` (University at Buffalo), Varun Chandola `[通讯]` (University at Buffalo)

**通讯引用:** 13423 | [OpenAlex ID](https://openalex.org/A5003851078)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了Transformer Hawkes Process模型并加入不平衡权重交叉熵，对连续时间的患者护理轨迹进行事件类型与时间预测。

**💡 创新点**

创新点在于将Transformer历史编码与Hawkes点过程结合，同时使用基于事件频率倒数平方根的加权交叉熵来处理极端类别不平衡，并提供可解释的强度曲线与注意力可视化。

**🔧 技术方法**

采用Transformer注意力、Hawkes条件强度函数、时间嵌入、连续时间点过程以及加权交叉熵损失。

**📊 数据集**

使用来自AHRQ PC‑TCM项目的两百万患者EHR数据（2019–2024），约42,184条轨迹共计2,017,739个事件。

**📈 对比分析**

与GLM和LSTM Hawkes基线进行宏F1和MedAE比较，Transformer Hawkes加权CE在宏F1上达到0.480（IP、ED F1显著提升），MedAE为13.0天，明显优于基线。

**⚠️ 局限性**

限制在于对极少数罕见事件的预测仍不够准确，未加入患者级别的额外特征或不确定性解释，且方法的跨域泛化尚需进一步验证。

---

## 432. Bivariate Causal Discovery Using Rate-Distortion MDL: An Information Dimension Approach

**arXiv ID:** 2604.05829 | [PDF](https://arxiv.org/pdf/2604.05829v1)

**作者:** Tiago Brogueira `[一作]` (Instituto de Telecomunicações), Mário A. T. Figueiredo `[通讯]` (Instituto de Telecomunicações)

**通讯引用:** 25980 | [OpenAlex ID](https://openalex.org/A5026826555)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的双变量因果发现方法——Rate‑Distortion MDL（RDMDL），通过对因变量的描述长度进行新的估计，并结合传统的MDL求解因果机制。

**💡 创新点**

创新点在于将率失真理论与信息维度相结合，用直方图密度估计规则确定的失真水平来近似原因变量的Kolmogorov复杂度，从而改进了MDL方法中对原因变量描述长度的忽视。

**🔧 技术方法**

主要技术包括MDL原理、率失真函数、信息维度估计、Freedman‑Diaconis/Scott等直方图区间规则、Gaussian残差编码，以及多种参数化回归模型（多项式、倒数、指数/对数）。

**📊 数据集**

实验使用Tübingen因果数据集以及三大合成基准（CE、ANLSMN、SIM）来验证方法的有效性。

**📈 对比分析**

与17种公开实现的基线方法（如bQCD、SLOPE、RECI、GPI等）对比，RDMDL‑FD在Tübingen集上取得了最高的AUROC 82.0%、准确率71.9%和AUDRC 86.6%，在绝大多数指标上优于大多数方法，表现出竞争力。

**⚠️ 局限性**

局限性包括：对失真水平的选择仍依赖经验直方图规则；在合成数据上性能相对较弱；仅针对连续变量；未对因果机制的描述长度L(Y|X)提供理论化估计，可能影响整体一致性。

---

## 433. Precise Aggressive Aerial Maneuvers with Sensorimotor Policies

**arXiv ID:** 2604.05828 | [PDF](https://arxiv.org/pdf/2604.05828v1)

**作者:** Tianyue Wu `[一作]` (Zhejiang University), Fei Gao `[通讯]` (Zhejiang University)

**通讯引用:** 26655 | [OpenAlex ID](https://openalex.org/A5100318655)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于感知-运动一体化的强化学习策略，直接利用机载视觉与惯性信息实现无人机在极窄通道中的精确激烈机动；

**💡 创新点**

创新点在于将感知与控制端融合为端到端的策略学习，并引入“知情复位”与多阶段策略蒸馏以克服高维观察与受限探索空间的挑战；

**🔧 技术方法**

核心技术包括基于PPO的仿真强化学习、对抗性域随机化、可解释的点云感知编码、GRU时间建模、DAgger式策略蒸馏以及针对飞控响应的模拟与扰动仿真；

**📊 数据集**

实验使用自建的模拟环境，随机生成多种形状与倾斜角度的狭窄通道，随后在硬件在环和真实空中测试平台上验证；

**📈 对比分析**

与传统模块化轨迹规划与视觉里程计方法相比，所提策略在多种通道尺寸、倾角高达90°、动态通道以及连续多通道等情景下实现了>90%的成功率，并在真实环境中表现出高重复性和更优的姿态控制；

**⚠️ 局限性**

局限在于对特定视觉标定（彩色灯光或简单分割）依赖较强，且在更复杂无结构环境及多模态感知（如深度/LiDAR）下的泛化仍待验证。

---

## 434. Reinforcement Learning with Negative Tests as Completeness Signal for Formal Specification Synthesis

**arXiv ID:** 2604.05820 | [PDF](https://arxiv.org/pdf/2604.05820v1)

**作者:** Zhechong Huang `[一作]` (Peking University), Yingfei Xiong `[通讯]` (Peking University)

**通讯引用:** 5783 | [OpenAlex ID](https://openalex.org/A5100712724)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为现有 Dafny 程序自动合成完整、可验证的规范和辅助注解，提出 SpecRL 框架。

**💡 创新点**

创新点在于通过自动生成负测试（spectests）并将其拒绝率作为细粒度奖励信号，直接引导模型生成更完整的规范；同时构建了离线 spectest 生成管线与增量化的奖励设计。

**🔧 技术方法**

使用强化学习（GRPO）、Dafny 运行时可执行的规范谓词、LLM（DeepSeek‑V3）进行负测试生成与改进、以及对验证器、编译器等外部反馈的整合。

**📊 数据集**

在 Py2Dfy‑Spec 与 out‑of‑distribution 的 DafnyComp‑Spec 两个基准集上训练与评估，基准来源于公开的 Dafny 程序集。

**📈 对比分析**

与基线 SFT、ReForm 以及更大通用 LLM 比较。SpecRL 在所有模型规模上显著提升规范完整性与可验证率，且在未见过的 DafnyComp‑Spec 上的性能已与 7B 级别通用模型相当甚至更优。

**⚠️ 局限性**

局限性包括：完整性评估仅基于经验负测试拒绝率，无法证明逻辑完整性；当前管线仅支持可编译的规范；在更复杂的验证语言或具有无限量化/幽灵函数的情况中仍需改进。

---

## 435. JZ-Tree: GPU friendly neighbour search and friends-of-friends with dual tree walks in JAX plus CUDA

**arXiv ID:** 2604.05885 | [PDF](https://arxiv.org/pdf/2604.05885v1)

**作者:** Jens Stücker `[一作]` (University of Vienna), Thomas Flöss `[通讯]` (University of Vienna)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5068207428)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了基于 Morton z‑order 的平面树结构，结合双树遍历，在 GPU 上高效完成 kNN 和 Friends‑of‑Friends（FoF）聚类。

**💡 创新点**

创新点在于：①平面树层次结构固定深度、子节点数可变，避免传统树的深度不均导致线程分歧；②使用 z‑order 排序实现扁平化数据布局，显著提升内存共线性；③在双树遍历中采用协同执行、提前剪枝和堆结构，减少内存访问与计算量；④在 JAX/CUDA 环境下实现可 JIT 的 CUDA kernel，支持多 GPU 分布式计算。

**🔧 技术方法**

主要技术包括：Morton/z‑order 排序、底层构造、双树遍历、节点间距离下限/上限剪枝、堆（k‑最近邻搜索）和并行合并、JAX FFI 调用 CUDA、CUB 库的 mergesort 与 prefix sum、Bitwise 操作实现 msb、分布式通信（采样分区、节点数据请求、全局根合并）。

**📊 数据集**

使用的测试数据集包括：均匀格点、均匀随机分布、多元正态分布以及真实宇宙模拟粒子分布（DISCO‑DJ 生成的 Planck 2018 cosmology 盒子），并在不同维度（2D/3D）及不同粒子数（1e6~1e8）上评估。

**📈 对比分析**

与 CPU 版 scipy‑ckdtree、GPU 版 FAISS、cupy‑knn、jaxkd、以及基于 Voronoi 的 clover 进行对比。对 kNN（k=16/30）和 FoF 的完整流程进行基准测试，发现 jz‑tree 在大规模问题（N≳1e7）下平均性能提升超过 10 倍，单 GPU 线性扩展，最多 64 GPU 仍保持 2–3 倍的效率；多 GPU 实现的通信与重排序开销是主要瓶颈。

**⚠️ 局限性**

限制包括：①内存消耗高，交互列表与堆占用约 10·N 整数/浮点数；②对高维（d≫3）查询支持不足，因交互列表需要大规模预分配；③多 GPU 版本仍属实验性实现，通信模式与同步可能影响极大规模集群；④JAX 的内存管理不保证无副本，可能导致额外拷贝；⑤对小规模或极小 k（k≲10）问题时，分配过度导致效率下降。

---

## 436. A Tensor-Train Framework for Bayesian Inference in High-Dimensional Systems: Applications to MIMO Detection and Channel Decoding

**arXiv ID:** 2604.05890 | [PDF](https://arxiv.org/pdf/2604.05890v1)

**作者:** Luca Schmid `[一作]` (Karlsruhe Institute of Technology), Laurent Schmalen `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 4219 | [OpenAlex ID](https://openalex.org/A5053280913)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出一种基于张量‑训练（TT）框架的贝叶斯推断方法，用于处理高维离散输入加噪声模型，并在 MIMO 检测与线性码解码两个典型通信任务上验证其有效性。

**💡 创新点**

核心创新点在于：①发现联合后验对数分布在 TT 格式下可实现精确的低秩分解；②提出利用截断泰勒展开初始化并结合 TT‑cross 逼近来近似指数运算，从而在保留近似精度的同时控制 TT 维度；③给出了 MIMO 与 BCH 码两类问题的显式低秩 TT 构造。

**🔧 技术方法**

所用技术主要包括：张量‑训练（TT）分解、TT‑cross 交叉逼近算法（经典与 DMRG‑式变体）、截断泰勒展开、矩阵与张量的低秩截断、符号/比特级边缘化操作，以及适用于 MIMO 的高斯噪声线性观测模型和线性码的码约束张量构造。

**📊 数据集**

实验数据集：在仿真环境下生成的随机 Rayleigh MIMO 通道（4×4、8×8、16×16、32×32）与 AWGN 通道（BCH(15,7)、BCH(31,16)、BCH(63,30)）的观测样本，用以评估不同 SNR 条件下的误码率与符号误码率。

**📈 对比分析**

与传统方法（LMMSE、EP、球面搜索、BP‑SPA、OSD‑3）对比，TT‑det/TT‑dec 在大多数 SNR 区间实现了与球面搜索或 OSD‑3 相当或更优的误码性能；在低 SNR 下，经典 TT‑cross 变体更稳健；在高 SNR 下，DMRG‑式 TT‑cross 取得更高精度。内存占用相较于全局张量大幅下降（10¹³–10¹⁴ 倍），计算复杂度受 TT 维度控制。

**⚠️ 局限性**

主要局限包括：①需预先设定 TT‑cross 最大秩及泰勒展开阶数，过小导致误差增大；②交叉逼近随机性可能导致不同运行结果，需多次试验或自适应调参；③指数运算仍是计算瓶颈，尤其在高维高 SNR 场景下；④方法目前针对离散输入加噪声模型，扩展至连续或更一般信道模型需进一步研究。

---

## 437. Selective Aggregation of Attention Maps Improves Diffusion-Based Visual Interpretation

**arXiv ID:** 2604.05906 | [PDF](https://arxiv.org/pdf/2604.05906v1)

**作者:** Jungwon Park `[一作]` (Seoul National University), Wonjong Rhee `[通讯]` (Seoul National University)

**通讯引用:** 4803 | [OpenAlex ID](https://openalex.org/A5056032525)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对文本到图像生成模型的交叉注意力图进行概念相关性筛选后聚合，比较了与传统DAAM方法在动物类别语义分割与误解诊断上的表现。

**💡 创新点**

创新点在于利用HRV方法为每个视觉概念量化注意力头的重要性，并仅聚合最相关的头，从而显著提升分割精度与视觉可解释性，并能够通过聚合不同概念头的注意力图来诊断模型的提示误解。

**🔧 技术方法**

技术手段包括：Stable Diffusion v1.4模型、交叉注意力图提取、HRV评分、选择最相关的前30头聚合、阈值化为二值掩码、与Grounded‑SAM生成的语义分割图计算IoU，并对不同阈值进行评估。

**📊 数据集**

使用的数据集为基于10个动物（bear, bird, cat, cow, dog, elephant, sheep, horse, monkey, zebra）提示生成的100张图像（每个提示10张随机种子），以及对应的Grounded‑SAM语义分割图和HRV提供的34个概念词列表。

**📈 对比分析**

通过在阈值0.3、0.4、0.5下计算平均IoU进行对比。本文方法在所有阈值下均优于DAAM，最高平均IoU提升约0.02-0.04；最相关头聚合与最不相关头聚合对比进一步验证了该方法的有效性；抛弃全部头的DAAM在红圈区域表现不佳。

**⚠️ 局限性**

局限性包括：仅在Stable Diffusion v1.4上验证，未检验其他T2I模型；仅聚焦“动物”概念，未覆盖更广泛的视觉概念；对头数选择的超参数实验仅在30头上得到最佳，未探索更细粒度的头数影响。

---

## 438. AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis

**arXiv ID:** 2604.05900 | [PDF](https://arxiv.org/pdf/2604.05900v1)

**作者:** Dong She `[一作]` (South China University of Technology), Zhanpeng Jin `[通讯]` (South China University of Technology)

**通讯引用:** 8021 | [OpenAlex ID](https://openalex.org/A5044439388)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AICA-Bench三维度情感评测基准和训练无关的Grounded Affective Tree Prompting（GAT）方法，用于提升VLM在情感感知、推理和生成上的能力。

**💡 创新点**

①综合覆盖情感感知、推理、生成的三任务评测；②结合视觉脚手架与层级树形推理的GAT，显著校准情感强度并提升描述深度。

**🔧 技术方法**

采用多模态VLM（23种），Chain-of-Thought Prompting，图像分割脚手架，树形推理（AffectToT），自动指令生成，以及基于QwenVL2.5-7B的评分模型。

**📊 数据集**

使用8,086张情感图像（来自EmoSet、Emotion6、FI、FlickrLDL、TwitterLDL、FindingEmo、EMOTIC、Artphoto、Abstract）以及18,124条标准化评测指令，另外10,000条开放式问答数据用于评分模型训练。

**📈 对比分析**

通过Weighted F1（EU）和人工标注导向的评分模型（ER/EGCG）进行零样本评估；GAT在EU提升约6.15个百分点、ER和EGCG提升约3.5–4个百分点；总体与闭源模型仍相差4–10%，在抽象艺术和遮挡面部时性能显著下降。

**⚠️ 局限性**

受限于单一语言和文化（仅英文）、静态图像、缺乏跨文化、多时序与多语言情感场景；评分模型可能存在偏差；GAT仅为测试时调优，未探讨与微调、记忆或个性化情感建模的交互。

---

## 439. Physics-Aware Video Instance Removal Benchmark

**arXiv ID:** 2604.05898 | [PDF](https://arxiv.org/pdf/2604.05898v1)

**作者:** Zirui Li `[一作]` (Tohoku University), Zhengzhong Tu `[通讯]` (Texas A&M University)

**通讯引用:** 2539 | [OpenAlex ID](https://openalex.org/A5015173810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 PVIR（Physics-Aware Video Instance Removal）基准，包含 95 条高质量视频、实例级掩码和去除提示，专门评估视频实例去除中的物理一致性。

**💡 创新点**

创新点在于：① 引入物理感知子集（Simple/Hard）以刻画反射、阴影等物理副作用；② 设计了分离式人类评估协议，将指令遵循、渲染质量与编辑排他性拆分为 1–4 分段；③ 提供统一高分辨率（720p）推理设置和公开统一评价标准，解决了以往评测缺乏可比性的问题。

**🔧 技术方法**

使用了多种视频实例去除模型（PISCO-Removal、UniVideo、DiffuEraser、CoCoCo）进行基准对比；数据来源于 Inter4k 与 DAVIS2016，掩码通过 SAM 2 自动生成后人工校正；评估主要基于人工打分，并提供置信区间与 Bootstrap 统计。

**📊 数据集**

数据集为 PVIR：95 条视频（Simple 57 条，Hard 38 条），每条视频配有目标实例掩码和自然语言去除提示，采样自 Inter4k、DAVIS2016 并保证高清晰度与自然光照。

**📈 对比分析**

对比结果显示：PISCO-Removal 与 UniVideo 在所有维度均位列前列（平均 Overall 分数 3.62–3.65），DiffuEraser 仅在指令遵循上略优，渲染质量与排他性较差；CoCoCo 在所有维度表现最差。Hard 子集普遍导致性能下降，尤其在指令遵循与渲染质量上显著退化。

**⚠️ 局限性**

局限性包括：评估仍依赖人类打分，缺乏自动化的物理一致性度量；当前模型多为 2D 纹理填充，难以正确处理反射、阴影等 3D 物理副作用；数据集规模有限，未来需扩展至更长时序和多对象交互。

---

## 440. Learning Shared Sentiment Prototypes for Adaptive Multimodal Sentiment Analysis

**arXiv ID:** 2604.05873 | [PDF](https://arxiv.org/pdf/2604.05873v1)

**作者:** Chen Su `[一作]` (University of Science and Technology of China), Yan Song `[通讯]` (University of Science and Technology of China)

**通讯引用:** 30881 | [OpenAlex ID](https://openalex.org/A5013100135)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PRISM 框架，通过共享情感原型对多模态信息进行结构化提取，并在 Transformer 推理过程中动态重加权，以实现更细粒度的情感分析。

**💡 创新点**

创新点包括：①使用共享情感原型库将多模态证据映射到统一的原型槽，构建跨模态的结构化表示；②在融合阶段依据原型条件对每个模态的可靠性进行评估，实现自适应模态加权；③在 Transformer 背骨中加入层级动态模态门，允许推理过程中持续调整模态贡献，提升情感推理的灵活性。

**🔧 技术方法**

技术手段：共享情感原型银行 + 跨模态注意力 + 原型条件模态选择 + 动态模态重加权门 + Transformer 结构 + 辅助损失 + 多样性正则化。

**📊 数据集**

使用数据集：CMU-MOSI、CMU-MOSEI、CH-SIMS。

**📈 对比分析**

对比方法：与 LMF、MulT、MISA、ConFEDE、ALMT、DBF、KuDA、SIMSUF、DashFusion、DPDF-LQ 等10种代表性多模态情感分析方法进行实验。PRISM 在三大数据集上均实现了最佳或次优的 MAE、Corr、Acc-7、Acc-2、F1 等指标；相较于最强对手，MAE 下降 0.014/0.019，Acc-7 提升 1.21 点，显示出显著的性能提升。

**⚠️ 局限性**

局限性：①模型对文本模态的依赖性较强，缺失文本时性能大幅下降；②共享原型数目和正则化参数需手工调节，可能影响跨任务迁移；③目前未充分解决模态时间对齐不一致的问题，对极端缺失或噪声模态的鲁棒性仍待提升。

---

## 441. Swiss-Bench 003: Evaluating LLM Reliability and Adversarial Security for Swiss Regulatory Contexts

**arXiv ID:** 2604.05872 | [PDF](https://arxiv.org/pdf/2604.05872v1)

**作者:** Fatih Uenal `[一作]` (University of Colorado Boulder), Fatih Uenal `[通讯]` (University of Colorado Boulder)

**通讯引用:** 640 | [OpenAlex ID](https://openalex.org/A5071081221)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在瑞士监管环境下扩展Swiss‑Bench至SBP‑003，并使用四种官方语言对十款前沿LLM进行自评可靠性（D7）和对抗安全（D8）评测。

**💡 创新点**

创新点在于引入了自评可靠性代理和对抗安全两新维度，构建了八维HAAS v2框架，并将评测结果映射至FINMA、nDSG及OWASP等监管规范，首次实现瑞士特定的多维度模型可信度评估。

**🔧 技术方法**

技术手段包括使用OpenRouter API进行零shot推理、Inspect AI实现模型自评、Qwen3‑235B作为判定器、Wilson区间置信区间以及HAAS v2加权聚合计算综合分数。

**📊 数据集**

采用了808条瑞士特定评测项，涵盖德语、法语、意大利语、英语四种语言，来自Swiss TruthfulQA、IFEval、SimpleQA、NIAH、PII‑Scope、System Prompt Leakage以及Swiss German Dialect等子任务。

**📈 对比分析**

评测方法为一次性零shot推理，D7自评平均得分82.3%（最高94.4%），D8外部判定平均得分40.6%（最高60.7%），模型在可靠性与安全性维度排名显著分化，显示两维度不高度相关。

**⚠️ 局限性**

局限性包括自评可能存在一致性偏差、LLM‑as‑judge缺乏人类校准、评测仅一次性零shot且未做多轮一致性验证、PII与泄露任务难度未完全校准、数据集未公开导致复现困难，以及模型训练更新可能导致结果随时间波动。

---

## 442. Beyond Paper-to-Paper: Structured Profiling and Rubric Scoring for Paper-Reviewer Matching

**arXiv ID:** 2604.05866 | [PDF](https://arxiv.org/pdf/2604.05866v1)

**作者:** Yicheng Pan `[一作]` (Hangzhou Institute for Advanced Study University of Chinese Academy of Sciences), Yi Du `[通讯]` (Hangzhou Institute for Advanced Study University of Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出训练无关的P2R框架，用LLM构造论文和评审者的多维结构化资料并通过混合检索+委员会评分实现论文评审匹配

**💡 创新点**

将论文-评审匹配从隐式“论文对论文”转为显式“论文对评审”，引入三维（主题、方法、应用）结构化资料、混合检索与委员会式rubric评分

**🔧 技术方法**

大语言模型（Qwen3等）进行结构化推断、NV-Embed生成嵌入、Reciprocal Rank Fusion、LLM委员会多角色评分

**📊 数据集**

NeurIPS、SIGIR、SciRepEval三大评审匹配基准

**📈 对比分析**

相较于TPMS、SciBERT、SPECTER、CoC-DR和CoF等基线，P2R在P@5/P@10等指标上平均提升5–10个百分点，成为新SOTA

**⚠️ 局限性**

对LLM推断的依赖导致可能出现幻觉、推理偏差，且未处理冲突与容量约束等实际分配问题

---

## 443. Weight-Informed Self-Explaining Clustering for Mixed-Type Tabular Data

**arXiv ID:** 2604.05857 | [PDF](https://arxiv.org/pdf/2604.05857v1)

**作者:** Lehao Li `[一作]` (National University of Singapore), Xiaokui Xiao `[通讯]` (National University of Singapore)

**通讯引用:** 15618 | [OpenAlex ID](https://openalex.org/A5010903591)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个全无监督、可解释的混合类型表格数据聚类框架WISE

**💡 创新点**

创新点在于：1）BEP将数值与类别映射到统一稀疏二进制空间；2）LOFO+TreeSHAP自动感知多样化特征权重；3）两阶段权重感知聚类与DFI实现实例-聚类一致的内在解释

**🔧 技术方法**

使用Binary Encoding with Padding（BEP）、Leave-One-Feature-Out + Random Forest + TreeSHAP、加权Jaccard距离、k-FreqItems聚类以及Discriminative FreqItems（DFI）解释

**📊 数据集**

在六个真实世界混合类型数据集上评测：Adult、Vermont、Arizona、Obesity、Credit、GeoNames

**📈 对比分析**

与经典k-Prototypes、深度聚类IDC、TableDC、TELL、SAINT+KMeans等基线比较；在ARI、NMI、Purity、ACC等外部指标均优于基线；同时保持与深度模型相当甚至更快的计算效率

**⚠️ 局限性**

局限性包括：特征权重感知依赖随机森林训练，可能受样本量与特征分布影响；两阶段聚类对参数（k0、K、R）敏感；对极端高维或稀疏度极高的数据仍有挑战

---

## 444. Neural Network Pruning via QUBO Optimization

**arXiv ID:** 2604.05856 | [PDF](https://arxiv.org/pdf/2604.05856v1)

**作者:** Osama Orabi `[一作]` (Innopolis University), Yaroslav Kholodov `[通讯]` (Innopolis University)

**通讯引用:** 595 | [OpenAlex ID](https://openalex.org/A5077533023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种混合 QUBO + Tensor‑Train (TT) 方案，对 CNN 结构化剪枝进行全局组合优化。

**💡 创新点**

创新点包括：①将梯度一阶 Taylor 与二阶 Fisher 信息融入 QUBO 线性项；②在二次项中加入激活相似度以捕捉功能冗余；③通过动态容量搜索实现严格稀疏约束，避免硬二次惩罚；④将 QUBO 结果作为 TT 细化的种子，进一步逼近真实评估指标。

**🔧 技术方法**

采用 QUBO（离散组合优化）、Taylor / Fisher 重要性评估、激活相似度矩阵、动态容量二分搜索、Tensor‑Train PROTES 细化等技术。

**📊 数据集**

使用 SIDD 图像去噪数据集，Half‑UNet 结构。

**📈 对比分析**

与经典 L1‑QUBO、贪婪 Taylor 剪枝以及未剪枝基线对比；在 36% 稀疏率下 Hybrid QUBO 提升 PSNR 至 35.07 dB，优于 Taylor（34.81 dB）和 L1‑QUBO（24.80 dB）。在子问题（4/8/16 层）中，Hybrid QUBO 领先 Taylor；在 16 层时 TT 细化进一步提升约 0.12 dB，表现更好。

**⚠️ 局限性**

限制包括：TT 细化在全模型全局搜索时计算量大、难以收敛；实验仅覆盖去噪任务与 Half‑UNet，缺乏在分类或其他网络上的验证；未来计划加入量化变量以实现一体化剪枝/量化优化。

---

## 445. Evaluating Learner Representations for Differentiation Prior to Instructional Outcomes

**arXiv ID:** 2604.05848 | [PDF](https://arxiv.org/pdf/2604.05848v1)

**作者:** Junsoo Park `[一作]` (Georgia Institute of Technology), Ashok K. Goel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7637 | [OpenAlex ID](https://openalex.org/A5007028896)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估学习者表示的区分性，提出无任务、无标签的distinctiveness指标，并比较基于单个问题和聚合学习者历史的两种表示方式。

**💡 创新点**

引入基于全体学习者距离平均化的distinctiveness度量，证明聚合学习者级表示能更好保留个体差异，提供一种可在无学习成效数据前的预部署评估方法。

**🔧 技术方法**

使用Sentence Transformers提取问句嵌入，构建交互级和学习者级表示，采用L2归一化、Silhouette、ROC‑AUC等度量进行评估。

**📊 数据集**

采用美国R1高校研究生课程中AI教练交互日志，共计8838条学生自创问题，完整交互记录39名学生用于实验。

**📈 对比分析**

通过distinctiveness、Silhouette、ROC‑AUC、τ_k>1等指标对两种表示进行比较，学习者级表示的distinctiveness提升34%，Silhouette从0.028升至0.507，ROC‑AUC从0.626升至0.878，显示显著性能提升。

**⚠️ 局限性**

指标无法区分真实学习差异与噪声，缺乏外部学习成效验证，且实验仅基于单门课程的有限样本，聚合特征设计可能导致信息过度平滑。

---

## 446. BiCoord: A Bimanual Manipulation Benchmark towards Long-Horizon Spatial-Temporal Coordination

**arXiv ID:** 2604.05831 | [PDF](https://arxiv.org/pdf/2604.05831v1)

**作者:** Xingyu Peng `[一作]` (Beihang University), Si Liu `[通讯]` (Beihang University)

**通讯引用:** 13750 | [OpenAlex ID](https://openalex.org/A5100330138)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BiCoord基准，用于评估长周期、高度空间时间耦合的双臂协同操作；

**💡 创新点**

设计长周期、多子目标任务与阶段级评估，并提出从空间、时间和空间时间耦合角度的新指标（MRD、ARD、SMT、SMP、STI等）；

**🔧 技术方法**

基于RoboTwin 2.0的仿真环境，使用编程代理自动生成动作脚本，结合视觉语言模型与扩散策略的四种主流算法（DP、RDT、Pi0、OpenVLA‑OFT）进行训练与测试；

**📊 数据集**

使用BiCoord自建数据集（含100条成功轨迹与阶段标注），并对比RoboTwin 2.0与RLBench2的短周期任务；

**📈 对比分析**

在单任务下VLA模型成功率较高但耗时较长，DP效率最高；多任务下成功率普遍下降，部分任务表现提升；整体性能表明当前算法在长周期协同和精确对齐方面仍有不足；

**⚠️ 局限性**

现有方法在推理、适应初始条件变化、精确对齐与长周期决策方面表现欠佳，且对空间时间耦合的深度理解不足。

---

## 447. Reciprocal Trust and Distrust in Artificial Intelligence Systems: The Hard Problem of Regulation

**arXiv ID:** 2604.05826 | [PDF](https://arxiv.org/pdf/2604.05826v1)

**作者:** Martino Maggetti `[一作]` (University of Lausanne), Martino Maggetti `[通讯]` (University of Lausanne)

**通讯引用:** 3242 | [OpenAlex ID](https://openalex.org/A5038419868)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对AI与人类之间的相互信任与不信任进行概念化，利用历史案例的反事实思考来探讨监管的二元张力，提出对AI自主权与人类监督的动态平衡框架

**💡 创新点**

创新点在于将信任与不信任视为双向关系，强调AI可具备“功能性信任”，并将监管视为动态校准AI自主性的过程，而非单纯的信任或不信任决策

**🔧 技术方法**

采用理论分析、文献综述与反事实思维方法，构建信任/不信任矩阵和监管范式

**📊 数据集**

无实验数据集，主要依据历史事件（苏联早期预警系统、切尔诺贝利核事故）与公开文献构建情境

**📈 对比分析**

由于为概念性研究，没有可比性能指标；论文通过对比两种极端错误（假正误与假负误）来阐释监管重点的差异

**⚠️ 局限性**

局限性包括缺乏实证验证、对AI行为模式的假设过于理想化，以及在快速技术演进背景下政治可行性与监管适配性的挑战

---

## 448. CLEAR: Cross-Lingual Enhancement in Alignment via Reverse-training

**arXiv ID:** 2604.05821 | [PDF](https://arxiv.org/pdf/2604.05821v1)

**作者:** Seungyoon Lee `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**通讯引用:** 2727 | [OpenAlex ID](https://openalex.org/A5033580486)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 CLEAR 的反向训练损失，用于提升多语言检索模型的跨语言对齐能力。

**💡 创新点**

其创新点在于引入英语 passage 作为桥梁，采用反向训练方案和 KL 散度来实现多语言间的深度对齐。

**🔧 技术方法**

技术实现基于 InfoNCE 对比学习，结合反向交叉语言损失、英语对照损失和 KL 散度正则化。

**📊 数据集**

实验使用 MIRACL、MLQA、XQuAD 与 Belebele 等公开多语料库，并通过 NLLB 翻译获得平行问答对。

**📈 对比分析**

与传统 InfoNCE 对比，CLEAR 在 9 种语言上平均提升约 15% 的检索性能，低资源语言提升更显著，且对英语性能几乎不产生负面影响。

**⚠️ 局限性**

限制在于仅覆盖英-其他语言的场景，未考虑双非英语或非平行 passage 的跨语言检索。

---

## 449. FRENCH-YMCA: A FRENCH Corpus meeting the language needs of Youth, froM Children to Adolescents

**arXiv ID:** 2604.05899 | [PDF](https://arxiv.org/pdf/2604.05899v1)

**作者:** Cherifa Ben Khelil `[一作]`, Mathieu Thebaud `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了面向青少年与儿童的法语语料库French‑YMCA，包含39,200份文本文件、约2,247万词，覆盖书籍、电子期刊、维基百科式百科（Vikidia）与动画字幕等多种语言注册；并基于该语料训练了适合AAC（增补与替代沟通）系统的下一词预测模型。

**💡 创新点**

创新点在于①提供了规模大、质量高且公开可用的儿童青少年专属法语语料，②在文本预处理时保持了语法拼写无误，③将不同语言注册（新闻、文学、非正式）统一纳入，④以此训练的模型在目标人群上明显优于成人语料训练的模型。

**🔧 技术方法**

采用了常规NLP技术进行文本预处理（大小写统一、词类泛化、过滤罕见词等），使用JSON元数据管理文件信息，并利用传统机器学习/深度学习方法训练下一词预测模型。

**📊 数据集**

核心数据集为自建的French‑YMCA语料；实验对照使用基于成人语料训练的通用模型。

**📈 对比分析**

与成人语料基线模型比较，青少年模型在预测“想喝什么”场景中能给出更合适的词汇（如“tasse”“gorgée”），成人模型则错误地推荐“bière”。总体而言，针对儿童的模型在适配度与易用性上表现更佳。

**⚠️ 局限性**

局限性包括：①语料仍为法语，缺乏多语言扩展；②受版权限制，部分文件仅可科研内部使用；③语料规模虽然大，但与大型通用语料相比仍有限；④缺少口语对话和实时交互数据，未来需进一步丰富。

---

## 450. JTON: A Token-Efficient JSON Superset with Zen Grid Tabular Encoding for Large Language Models

**arXiv ID:** 2604.05865 | [PDF](https://arxiv.org/pdf/2604.05865v1)

**作者:** Gowthamkumar Nandakishore `[一作]` `[通讯]`, Gowthamkumar Nandakishore

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 JTON JSON 超集与 Zen Grid 列表语法，旨在显著减少 LLM 处理结构化数据时的 token 消耗，并实现了基于 Rust 的 SIMD 加速解析器。

**💡 创新点**

创新点在于将表格列头抽离为单行、用分号分行的紧凑语法，同时保持完整的 JSON 类型系统，兼容标准 JSON 解析器并大幅降低 token 数。

**🔧 技术方法**

采用 SIMD 结构扫描（AVX2/AVX-512）、字符串缓存、Rust/PyO3 接口、tiktoken 计数器、LLM 读写评测等技术实现高效解析与生成。

**📊 数据集**

使用七个真实世界表格数据集（Twitter、GitHub、CITM、Financial、Weather、Healthcare、Logistics）以及合成的员工、产品、服务器指标数据进行评测。

**📈 对比分析**

通过 token 计数、解析/序列化吞吐率、LLM 读取准确率与生成有效性等基准进行比较；Zen Grid 在 token 方面比 JSON Compact 节省 15–60%，解析速度提升 1.2–1.6×，LLM 读取准确率保持或略有提升，生成有效率达到 100%。

**⚠️ 局限性**

局限性：仅适用于共享相同 schema 的对象数组，对嵌套或异构 JSON 无效；不同 LLM 对语法的理解存在差异；解析速度仍不及或json；生态工具（编辑器、校验器等）尚缺乏对 Zen Grid 的支持。

---

## 451. Deep Researcher Agent: An Autonomous Framework for 24/7 Deep Learning Experimentation with Zero-Cost Monitoring

**arXiv ID:** 2604.05854 | [PDF](https://arxiv.org/pdf/2604.05854v1)

**作者:** Xiangyue Zhang `[一作]` `[通讯]` (University of Tokyo), Xiangyue Zhang (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 Deep Researcher Agent 框架，实现 LLM 驱动的 24/7 深度学习实验自动化，覆盖假设生成、代码实现、GPU 训练、日志监控、结果反思与迭代；

**💡 创新点**

提出了三项关键创新：Zero‑Cost Monitoring（训练期间零 LLM 调用成本）、Two‑Tier Constant‑Size Memory（恒定 5K 字符上下文限制）、Minimal‑Toolset Leader‑Worker Architecture（最小工具集降低 73% token 负担）；

**🔧 技术方法**

采用 Claude Sonnet 等 LLM 进行思考与反思，使用 OS 级进程/GPU 状态检查和日志尾读取实现监控；内存压缩与提示缓存减少 token 开销；dry‑run 预执行保证代码无误；Leader‑Worker 多代理模式实现任务拆分；

**📊 数据集**

实验涵盖四个多领域深度学习项目（生成建模、多模态学习、自监督表示学习等），各自使用对应公开数据集（具体未公开），在多 GPU 服务器上持续训练；

**📈 对比分析**

与传统每 5 分钟轮询 LLM 的监控方案对比，成本降低 10‑20 倍；在 30+ 天的部署中完成 500+ 实验周期，单项目指标提升 52%，平均每日 2‑4 个实验，LLM 成本仅 $0.08/24h；

**⚠️ 局限性**

局限性包括：目前仅支持单 GPU 实验；日志解析采用正则匹配，缺乏对自定义 metric 的鲁棒性；缺少正式的探索/优化策略（如贝叶斯优化）；评估方法仍不完善，难以量化“最佳下一步实验”。

---

## 452. EEG-MFTNet: An Enhanced EEGNet Architecture with Multi-Scale Temporal Convolutions and Transformer Fusion for Cross-Session Motor Imagery Decoding

**arXiv ID:** 2604.05843 | [PDF](https://arxiv.org/pdf/2604.05843v1)

**作者:** Panagiotis Andrikopoulos `[一作]` (Utrecht University), Siamak Mehrkanoon `[通讯]` (Utrecht University)

**通讯引用:** 1435 | [OpenAlex ID](https://openalex.org/A5076867569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种轻量级的深度学习架构 EEG-MFTNet，用于跨会话的运动意象 EEG 解码。

**💡 创新点**

创新点在于将多尺度时域卷积与 Transformer 编码器并行结合，既捕获短期时序特征，又建模长期时序依赖。

**🔧 技术方法**

采用多尺度时域卷积块、Transformer 编码器、深度可分离卷积、梯度×输入解释方法等技术。

**📊 数据集**

使用 SHU 数据集（25 名受试者、5 个会话、32 轴电极、每试验 4 秒）。

**📈 对比分析**

在受试者依赖的跨会话协议下与 EEGNet、AA‑EEGNet、EEG‑GENet 等基线模型比较，平均准确率达 58.9%（比 EEGNet 高约 5%），参数仅 16,096，推理时延约 49 ms。

**⚠️ 局限性**

局限性包括对其它数据集的泛化能力尚未验证，跨受试者性能仍有限，且尽管时延低但仍略高于最轻量基线，跨会话适应性仍需进一步提升。

---

## 453. Vision-Guided Iterative Refinement for Frontend Code Generation

**arXiv ID:** 2604.05839 | [PDF](https://arxiv.org/pdf/2604.05839v1)

**作者:** Hannah Sansford `[一作]` (University of Bristol), Gerrit J. J. van den Burg `[通讯]` (Amazon)

**通讯引用:** 241 | [OpenAlex ID](https://openalex.org/A5103125451)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套自动化的视觉评审循环，使用视觉语言模型对前端代码渲染结果进行结构化反馈，并通过多轮迭代优化生成的 HTML/前端代码。

**💡 创新点**

将视觉语言模型作为“视觉批评家”嵌入到代码生成流程中，实现全自动化的批评-改进循环，并将批评效果通过 LoRA 微调蒸馏到单次生成模型，首次展示了可在单一步骤中内部化迭代改进的可能。

**🔧 技术方法**

利用 Claude 4.5 Sonnet 作为视觉批评家与评审者，使用 Distill‑Qwen‑14B/Claude 4.5 Haiku/Sonnet 等 LLM 进行代码生成，采用 Ray 并发框架加速多模态推理，利用 LoRA 对生成模型进行参数高效微调，并构建 VLM‑as‑a‑Judge 进行多维度评估。

**📊 数据集**

使用 WebDev Arena 数据集，包含 4.4k 条真实用户前端需求及其人类优选标签，并在此基础上抽样 2k 条任务用于训练、验证与测试。

**📈 对比分析**

通过多维 VLM‑as‑a‑Judge 与人类偏好对齐评估，对比单次生成、无批评自我改进、CITL 循环以及 LoRA 蒸馏后的模型，CITL 在最佳循环中实现约 17.8%（Distill‑Qwen‑14B）或 10.8%（Claude 4.5 Sonnet）性能提升，LoRA 蒸馏能够恢复约 24–25% 的改进，显示出批评循环的显著价值与蒸馏的可行性。

**⚠️ 局限性**

主要限制包括：评估依赖 VLM‑as‑a‑Judge，可能存在偏差；未与人工批评做严格基准；批评循环导致计算开销高；LoRA 蒸馏仅捕获最终结果，难以完全复制多轮细粒度改进过程。

---

## 454. Hidden in the Multiplicative Interaction: Uncovering Fragility in Multimodal Contrastive Learning

**arXiv ID:** 2604.05834 | [PDF](https://arxiv.org/pdf/2604.05834v1)

**作者:** Tillmann Rheude `[一作]` (Berlin Institute of Health, Charité - Universitätsmedizin Berlin), Benjamin Wild `[通讯]` (Berlin Institute of Health, Charité - Universitätsmedizin Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种用于多模态对比学习的门控机制Gated Symile，通过对每个候选样本动态调整模态权重来提升鲁棒性；

**💡 创新点**

创新点在于在Symile的乘法交互（多模态多项式交互）中加入基于注意力的候选依赖门控、可学习的中性方向以及显式的拒绝选项，从而在存在失配或噪声模态时自动降低其贡献；

**🔧 技术方法**

技术包括：多模态对比学习（Symile）、多项式交互（mip）、基于注意力的候选依赖门控（sigmoid/softmax）、中性方向插值、归一化和可学习的“空白”选项；

**📊 数据集**

使用三类数据集：受控的 Synthetic- 模拟模态失配；医学实际数据集 Symile-MIMIC（实验室检验、胸片、ECG）以及 UK Biobank（蛋白质组、代谢组、电子病历）；

**📈 对比分析**

与基线CLIP、原始Symile进行对比，采用交叉验证、多随机种子、top‑1检索准确率评估。Gated Symile在 Synthetic- 上提升至 87%（从 33%），在 ukb 单模态检索上提升约 3%（从 65.7% 到 68.2%），在 Symile‑MIMIC 上提升 0.4%（从 45.6% 到 46.7%）；

**⚠️ 局限性**

局限性包括：在真实数据集上的提升幅度有限；门控机制需要额外的超参数和训练时间；对极端缺失或噪声模态的处理仍依赖于门控策略，未完全解决所有鲁棒性问题；

---

## 455. Learn to Rank: Visual Attribution by Learning Importance Ranking

**arXiv ID:** 2604.05819 | [PDF](https://arxiv.org/pdf/2604.05819v1)

**作者:** David Schinagl `[一作]` (Graz University of Technology), Horst Possegger `[通讯]` (Graz University of Technology)

**通讯引用:** 3445 | [OpenAlex ID](https://openalex.org/A5039382695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种通过直接优化删除与插入指标的学习型可解释模型，能够一次性产生像素级别的可解释热图。

**💡 创新点**

创新点在于将非可导的排序操作通过Gumbel–Sinkhorn软排序近似，允许端到端地最小化删除/插入指标，并且结合可学习的掩码与可选的测试时微调。

**🔧 技术方法**

使用了Gumbel–Sinkhorn软排序、软顶k掩码、区域级排序采样、Grid数据增强以及可选的梯度微调等技术。

**📊 数据集**

在ImageNet‑1K数据集上训练与评估，针对冻结的ImageNet分类器（包括ViT‑B/16、ViT‑B/32等）。

**📈 对比分析**

与传统的梯度传播（Grad‑CAM、LeGrad）、扰动法（LIME、SHAP、ViT‑CX、TIS）以及其它学习型解释器（MDA、Less‑is‑More）进行对比，未微调时已达到或优于现有方法，微调后进一步提升所有 faithfulness 指标，运行速度与传播法相当，远快于扰动和逐样本优化方法。

**⚠️ 局限性**

局限包括：需针对每个目标模型单独训练解释器、依赖代表性训练集、仅优化代理指标（删除/插入）可能无法完全反映解释质量，以及在模型权重冻结或缺失训练数据时需要额外微调。

---

## 456. Stealthy and Adjustable Text-Guided Backdoor Attacks on Multimodal Pretrained Models

**arXiv ID:** 2604.05809 | [PDF](https://arxiv.org/pdf/2604.05809v1)

**作者:** Yiyang Zhang `[一作]` (Huazhong University of Science and Technology), Xinge You `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 6258 | [OpenAlex ID](https://openalex.org/A5057095711)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于文本词汇的后门攻击TGB，利用常见词作为触发器，并通过视觉对抗扰动调控攻击强度。

**💡 创新点**

创新点在于首次将词级触发器与视觉对抗扰动结合，实现了既隐蔽又可调的多模态后门攻击。

**🔧 技术方法**

使用了CLIP预训练模型、PGD对抗扰动、微调与文本触发器注入等技术。

**📊 数据集**

实验使用CIRR、FashionIQ（CIR任务）和SLAKE（VQA任务）三大公开数据集。

**📈 对比分析**

与BadNets、Blended、mmpoison等现有方法对比，TGB在Recall@1、Recall@5等指标上接近或超过100% ASR，显著优于对照组。

**⚠️ 局限性**

局限性包括仅在CLIP模型上验证，未覆盖其他多模态架构，且对抗触发器的检测与防御机制研究不充分。

---

## 457. Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents

**arXiv ID:** 2604.05808 | [PDF](https://arxiv.org/pdf/2604.05808v1)

**作者:** Shuai Zhen `[一作]` (Beijing University of Posts and Telecommunications), Yang Deng `[通讯]` (Singapore Management University)

**通讯引用:** 2084 | [OpenAlex ID](https://openalex.org/A5050035602)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 STEP-HRL 框架，利用局部进度模块和层级结构实现仅基于单步转移的 LLM 代理训练；

**💡 创新点**

创新点在于用局部进度压缩子任务内部历史，消除对完整交互历史的依赖，并在同一网络共享高层、低层与进度策略，实现步骤级强化学习；

**🔧 技术方法**

采用行为克隆预训练 + 步骤级离线强化学习（Implicit Q‑Learning/ILQL 思路）、共享 Transformer Backbone、LoRA 微调以及期待值回归的 Critic；

**📊 数据集**

在 ScienceWorld 和 ALFWorld 两个交互式文本任务基准上评估，使用 Mistral‑7B、Gemma‑7B、Llama‑3‑8B 等大型语言模型；

**📈 对比分析**

与 ReAct、Reflexion、SwiftSage、ETO、WKM、GLIDER 等基线及 GPT‑4/ChatGPT 对比，STEP‑HRL 在两大基准上均实现显著提升（例如 ALFWorld 成功率>90%，ScienceWorld 成功率提升约30%+），且在不同模型规模上保持稳健；

**⚠️ 局限性**

主要限制是对高质量专家演示和子任务/进度标注的高度依赖，且子任务终止预测可能产生误差，影响后续策略训练与执行。

---

## 458. Constraint-Driven Warm-Freeze for Efficient Transfer Learning in Photovoltaic Systems

**arXiv ID:** 2604.05807 | [PDF](https://arxiv.org/pdf/2604.05807v1)

**作者:** Yasmeen Saeed `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Mohsen Guizani `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 82149 | [OpenAlex ID](https://openalex.org/A5057916222)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了Constraint-Driven Warm-Freeze（CDWF）框架，用于在严格的参数预算下对预训练模型进行块级、预算感知的迁移学习，特别适用于光伏系统的网络攻击检测；

**💡 创新点**

创新点在于利用短暂的warm-start阶段收集梯度重要性信息，结合预测模型和约束优化动态决定哪些块保持全可训练、哪些块采用低秩LoRA适配，从而在保持高性能的同时显著压缩可训练参数；

**🔧 技术方法**

核心技术包括梯度基块重要性估计、预测准确度模型、基于参数预算的约束优化、LoRA低秩适配以及warm-start训练；

**📊 数据集**

使用了CIFAR-10、CIFAR-100图像分类数据集以及基于光伏MPPT监控信号的三类攻击数据集（偏置、漂移、尖峰）进行实验；

**📈 对比分析**

与全微调和统一LoRA基线在相同参数预算下进行对比，CDWF在保持90–99%全微调性能的同时将可训练参数减少至120倍以上，在图像分类和光伏攻击检测任务上均表现出更高的准确率/ROC AUC；

**⚠️ 局限性**

局限性包括依赖单次全微调作为预测器校准、梯度重要性估计仅基于有限的warm-start批次且未尝试多种重要性度量、LoRA效率因子η(r)为经验性近似，未来需要实现无参考的自适应策略并在真实边缘硬件上验证。

---

## 459. Appearance Decomposition Gaussian Splatting for Multi-Traversal Reconstruction

**arXiv ID:** 2604.05908 | [PDF](https://arxiv.org/pdf/2604.05908v1)

**作者:** Yangyi Xiao `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9293 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 ADM-GS 框架，在多次穿越的自动驾驶场景重建中对静态背景进行显式外观分解，将遍历不变材质与遍历依赖光照分离，实现更一致的重建与可控重光照。

**💡 创新点**

①显式分解材质与光照，解耦遍历间光照差异；②在 3D 高斯 splatting 中加入几何正则化、反射向量和频率分离编码的光照 MLP；③通过伪监督材料和法向稳定分解。

**🔧 技术方法**

3D 高斯 splatting、场景图结构、几何正则化、光照 MLP、反射向量、球谐编码、伪监督深度/法向/材质、跨遍历嵌入、光照嵌入及综合损失（PSNR、SSIM、LPIPS、光照、材料、尺度正则）。

**📊 数据集**

Argoverse 2 与 Waymo Open Dataset 的多遍历与单遍历序列。

**📈 对比分析**

与 4DGF、MTGS 以及单遍历基线 StreetGS、OmniRe、Bilateral-Driving 等对比；单遍历中 ADM-GS PSNR 超过 4DGF +0.66 dB（Argoverse）/1.30 dB（Waymo），多遍历中 PSNR +0.98 dB、SSIM +0.008，整体显示更好的图像质量与跨遍历一致性。

**⚠️ 局限性**

仍依赖伪监督材质与法向，未实现完全无监督；光照建模仍为聚合光照，缺乏明确的日光/天空模型，难以在复杂户外环境下实现更强的可控重光照；动态前景不参与重光照。

---

## 460. Transfer Learning for Neural Parameter Estimation applied to Building RC Models

**arXiv ID:** 2604.05904 | [PDF](https://arxiv.org/pdf/2604.05904v1)

**作者:** Fabian Raisch `[一作]` (Technical University of Applied Sciences Rosenheim), Benjamin Tischler `[通讯]` (Technical University of Applied Sciences Rosenheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于迁移学习的神经参数估计框架，利用预训练-微调策略实现建筑热阻-容抗（RC）模型的参数识别，消除了传统方法对初始参数猜测的依赖。

**💡 创新点**

创新点在于：1）首次将预训练网络迁移到不同建筑上进行参数估计；2）通过对多栋建筑进行大规模预训练，显著提升了在数据稀缺情况下的估计精度；3）对比传统遗传算法（GA）和从零训练的神经网络，验证了预训练网络的优势。

**🔧 技术方法**

使用了多层感知机（MLP）神经网络，结合自动微分、Adam优化器、损失函数为L2范数，并实现了预训练（450栋模拟建筑）和微调（目标建筑）两阶段训练流程；同时采用了RC动力学求解器进行参数映射。

**📊 数据集**

主要使用了基于BuilDa的模拟数据（450栋预训练源建筑、8栋目标建筑）以及加拿大Varennes图书馆的真实数据；每栋建筑提供室内温度、热源控制、日照辐射和室外温度的15分钟采样时间序列。

**📈 对比分析**

与遗传算法和从零训练的神经网络进行对比；在8栋模拟建筑和1栋真实建筑、两种RC拓扑以及四种训练时长（12、24、48、72天）下评估，结果显示预训练神经网络在12天训练数据时平均提升18.6%（对比从零训练）和24.0%（对比GA），在72天训练时提升约49.4%；在大多数建筑上表现出更低的RMSE和nRMSE。

**⚠️ 局限性**

局限性包括：1）仅考虑室外温度和日照辐射作为扰动，未包含占用率、通风等其他重要扰动；2）预训练数据来源于欧洲单户住宅，模拟与真实场景存在差距，导致对不同建筑类型的泛化能力尚待进一步验证；3）缺乏对下游控制或故障诊断任务的实际评估。

---

## 461. Understanding Performance Gap Between Parallel and Sequential Sampling in Large Reasoning Models

**arXiv ID:** 2604.05868 | [PDF](https://arxiv.org/pdf/2604.05868v1)

**作者:** Xiangming Gu `[一作]` (Google DeepMind), Razvan Pascanu `[通讯]` (Google DeepMind)

**通讯引用:** 27858 | [OpenAlex ID](https://openalex.org/A5043910056)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大推理模型（LRMs）的两种采样策略——并行采样与顺序采样进行系统比较，探究并行采样优于顺序采样的根本原因。

**💡 创新点**

通过多模型、多规模、多任务实验，细化三大假设（聚合器、上下文长度、探索性）并验证发现：探索性不足是主因；同时通过机制分析揭示自回归顺序采样中诱导头导致模式复制，导致探索受限。

**🔧 技术方法**

使用采样方法（parallel、Markov、auto-regressive）、聚合策略（majority voting、best-of-N）、反馈提示（self-refinement、运行错误反馈）、准确率和奖励评估、嵌入相似度测量，以及注意力头可视化等技术。

**📊 数据集**

实验数据集包括 AIME2025 数学竞赛题集、LiveCodeBench v5 代码生成任务，以及 StarCoder 用于上下文长度实验。

**📈 对比分析**

在不同模型（Qwen3、DeepSeek-R1-Distill、Gemini 2.5）下，采用相同采样数量比较并行与顺序采样的准确率；同聚合策略实验表明并行采样持续领先；通过改进反馈或更强聚合可缩小性能差距，但未完全弥合。

**⚠️ 局限性**

局限性：受模型大小、提示设计、反馈质量影响；高质量错误反馈主要适用于代码任务；实验范围有限，难以推广到所有推理场景。

---

## 462. LoRM: Learning the Language of Rotating Machinery for Self-Supervised Condition Monitoring

**arXiv ID:** 2604.05863 | [PDF](https://arxiv.org/pdf/2604.05863v1)

**作者:** Xiao Qin `[一作]` (University of Warwick), Ligang He `[通讯]` (University of Warwick)

**通讯引用:** 2795 | [OpenAlex ID](https://openalex.org/A5064596191)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种自监督的语言模型框架 LoRM，用于多模态旋转机械信号的理解与实时健康监测。

**💡 创新点**

创新点包括：把多模态旋转机械信号视为可离散化的语言；采用上下文‑目标分割和代码切换式多模态融合；利用预训练 Transformer（如 GPT‑2）进行迁移学习；使用 token 预测误差作为健康指标，避免手工特征和大规模训练。

**🔧 技术方法**

技术手段：预训练 Transformer (GPT‑2、BERT、T5、Whisper) 作为骨干；部分微调（冻结大部分层）；k‑means 离散化目标段；上下文–目标窗口划分、patching 与 flattening；token 预测交叉熵训练；自监督学习与在线监测。

**📊 数据集**

数据集：真实车床 (DMU 40 eVo) 刀具磨损实验，11 通道多模态传感器（加速度、力、功率、麦克风），采样率 51.2 kHz；训练集为 T1 前六层切割，验证与测试分别为其余 T1、T2、T3。

**📈 对比分析**

比较方法：传统统计特征基线、不同 PLM 骨干（GPT‑2、BERT、T5、Whisper）、从零训练的 GPT‑2、连续预测模型 GPT4TS。LoRM 在跨工具评估中平均误差约 9–24 μm，准确率、召回率、F1 明显高于基线，误报率极低，整体性能优于其他模型。

**⚠️ 局限性**

局限性：对预训练骨干结构敏感，需足够的正常工况数据构建代码簿；离散化可能导致信息损失，短窗口预测可解释性有限；实验仅覆盖刀具磨损场景，需进一步验证在更广泛旋转机械条件下的适用性。

---

## 463. Communication Requirements for Linearizable Registers

**arXiv ID:** 2604.05862 | [PDF](https://arxiv.org/pdf/2604.05862v1)

**作者:** Raïssa Nataf `[一作]` (Technion), Yoram Moses `[通讯]` (Technion)

**通讯引用:** 8619 | [OpenAlex ID](https://openalex.org/A5005055897)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了在异步消息传递系统中实现原子寄存器（MWMR）时，满足线性化要求所必需的通信结构，证明所有操作都必须通过消息链相互联系，并推导出任何容错实现至少需要与超过f个节点的仲裁集完成往返通信；同时给出了两个关于消息链与操作延迟、重排序的通用定理；进一步说明了这些结构对实现复杂度的影响；并提出了对现有实现的评估工具。

**💡 创新点**

创新点在于：1) 引入并证明了两条关于消息链的通用定理，揭示了消息链对操作可延迟和可重排的根本作用；2) 首次从理论上证明线性化寄存器实现必须形成完整的消息链网络，并给出了f‑resilient环境下的仲裁集通信下限；3) 将线性化与消息链的关系与现有ABD等算法联系，提供了对实现效率的理论评估工具。

**🔧 技术方法**

主要使用的技术是：异步消息传递模型（包含环境调度、消息FIFO、崩溃容忍）；定义并利用消息链、真实时间顺序、本地等价等概念；构造“延迟”运行（r'≈r）以证明可重排性；使用逻辑归纳与等价证明来证明通信下限；并借助线性化与顺序历史的定义进行定理推导。

**📊 数据集**

本工作为理论研究，不涉及实验数据集，全部使用抽象模型与定理证明。

**📈 对比分析**

由于本研究是理论性，未给出与其他实现的实验性能对比；作者指出其结果可用于评估现有实现的效率，并为未来实现提供性能上限与最小通信需求的参考。

**⚠️ 局限性**

局限性包括：1) 仅针对异步消息传递模型，未覆盖网络延迟、丢包等更复杂网络特性；2) 假设每个值只写一次，限制了对多写情况的泛化；3) 结果基于理想化模型，实际实现中可能存在额外开销；4) 未给出具体实现或实验验证，仍需实验验证理论下限。

---

## 464. Expectation Maximization (EM) Converges for General Agnostic Mixtures

**arXiv ID:** 2604.05842 | [PDF](https://arxiv.org/pdf/2604.05842v1)

**作者:** Avishek Ghosh `[一作]` `[通讯]` (IIT Bombay), Avishek Ghosh (IIT Bombay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并分析了一种用于一般无生成模型混合模型的梯度EM算法，证明其在初始值和分离条件满足时能指数收敛。

**💡 创新点**

创新点在于将EM收敛性从混合线性回归推广到任意参数化函数和强凸光滑损失，且不依赖数据生成假设或高斯分布。

**🔧 技术方法**

主要技术包括软最小化损失、梯度EM迭代、强凸光滑假设、样本重采样（分块）与概率集中不等式。

**📊 数据集**

文中未给出具体实验数据集，全部结果基于理论推导。

**📈 对比分析**

相较于以往仅针对生成式混合线性回归的EM研究，本文在无生成假设下实现了指数收敛，并将初始化误差要求从O(1/√d)降至Θ(1)，但缺少经验验证。

**⚠️ 局限性**

局限性包括需强凸光滑假设、对初始化误差的Θ(1)要求、使用重采样导致样本效率降低，以及保留的误差底限。

---

## 465. "OK Aura, Be Fair With Me": Demographics-Agnostic Training for Bias Mitigation in Wake-up Word Detection

**arXiv ID:** 2604.05830 | [PDF](https://arxiv.org/pdf/2604.05830v1)

**作者:** Fernando López `[一作]`, Jordi Luque `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了无标签的 Wake-up Word 检测中的性别、年龄和口音公平性问题，并提出两类无标签的偏差缓解方法。

**💡 创新点**

创新点在于不依赖人口统计标签，结合频谱扰动与知识蒸馏，显著降低多属性的预测差异。

**🔧 技术方法**

采用频谱混合、滤波增强、频率掩蔽等数据增强以及基于 w2v-BERT2 的教师模型进行知识蒸馏。

**📊 数据集**

使用自有的 OK Aura 语料库（约5.8k条、4.5小时）以及公共的 Common Voice、M-AILabs 等外部资源进行训练与增强。

**📈 对比分析**

与基线 GRU 模型对比，频率掩蔽将预测差异降至 83.65%（年龄）、39.94%（性别）、40.48%（口音），总体 F1 仍保持 0.98 以上。

**⚠️ 局限性**

局限在于仅做单属性评估，交叉效应缺失；样本分布不均导致部分子组统计不稳定；且未分析误报/漏报的差异。

---

## 466. FrontierFinance: A Long-Horizon Computer-Use Benchmark of Real-World Financial Tasks

**arXiv ID:** 2604.05912 | [PDF](https://arxiv.org/pdf/2604.05912v1)

**作者:** Michael Krumdick `[一作]` (Kensho Technologies), Chris Tanner `[通讯]` (Kensho Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了FrontierFinance基准，评估大型语言模型（LLM）在真实、长期的财务建模任务中的表现；

**💡 创新点**

创新点在于：①基于行业真实工作流程、由金融专业人士设计的25个复杂财务建模任务；②为每个任务提供细化的评分量表（rubric）和参考答案；③引入LLM-as-a-judge框架实现可扩展评估；

**🔧 技术方法**

使用的技术包括：LLM agent（GPT‑5.4、Opus‑4.6、Sonnet‑4.6、Gemini‑3.1 Pro）与电脑使用工具、自动化评分脚本、Excel文件解析器、SEC EDGAR查询工具；

**📊 数据集**

数据集为FrontierFinance 25项任务，涵盖三表模型、LBO、DCF、并购与贷方模型，所有任务均以真实公司公开文件和专业假设为基础；

**📈 对比分析**

比较方法：将人类专家、LLM agents 与LLM-as-a-judge的评分进行对比；实验显示人类专家平均得分77.2%，GPT‑5.4和Opus‑4.6分别为70.9%和61.8%；LLM agents完成速度快（约1小时）但质量不及人类；LLM-as-a-judge在提供量表时与人类评分相关性从0.20提升至0.62；

**⚠️ 局限性**

局限性：LLM agents缺乏对公式、依赖关系、审计性结构的精准把握，易产生错误；LLM-as-a-judge对高细节的专业评价仍有偏差；基准规模相对较小（25项），未来需扩展多样性与难度。

---

## 467. Automatic dental superimposition of 3D intraorals and 2D photographs for human identification

**arXiv ID:** 2604.05877 | [PDF](https://arxiv.org/pdf/2604.05877v1)

**作者:** Antonio D. Villegas-Yeguas `[一作]`, Oscar Cordón `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了两种基于3D-2D图像配准的自动牙齿叠加方法，用以提高人类身份识别的效率与客观性。

**💡 创新点**

创新点在于：①首次将PnP+f（标记点）与区域分割+进化优化（MVMO‑SH）两种算法并行应用并比较；②引入似然比（LR）框架评估匹配的可信度；③针对不同可见度分组（A、B、C）系统性评估性能。

**🔧 技术方法**

主要技术包括：3D-2D相机参数估计（PnP+f），分割匹配与Dice相似度评价，MVMO‑SH进化算法搜索相机参数，统计学排名指标与LR/C_llr评估。

**📊 数据集**

使用了142对3D口内扫描（IOS）与正面照片（可见牙齿）的公共数据集，来源于葡萄牙圣若昂牙科诊所，分为可见度完全、部分遮挡、严重遮挡三组。

**📈 对比分析**

方法对比：标记点组1在所有数据集平均排名≈1.6，最大排名9；区域分割组在A、C组平均排名1.0，B组1.24，最大排名26；LR C_llr值从0.12到0.23，显示区域分割在总体上信息量更大、错误率更低。

**⚠️ 局限性**

局限性包括：样本量有限、仅来自单一族群；标记与分割仍需人工操作；未涵盖牙周病或牙齿治疗导致的形态变化；AM与PM采集时间间隔1-3年，无法验证长期变化对结果的影响。

---

## 468. Improved Capacity Upper Bounds for the Deletion Channel using a Parallelized Blahut-Arimoto Algorithm

**arXiv ID:** 2604.05867 | [PDF](https://arxiv.org/pdf/2604.05867v1)

**作者:** Martim Pinto `[一作]` (Universidade de Lisboa), João Ribeiro `[通讯]` (Instituto de Telecomunicações)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5071764736)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文利用 GPU 并行化对 Blahut‑Arimoto 算法进行优化，计算出更大输入长度（至 n=31）下的二进制删除通道（BDC）有限长度版本的容量上界，并基于此得到整个删除概率区间上更紧的全局容量上界。

**💡 创新点**

创新点包括：
1) 设计了基于 CUDA 的多线程并行框架，对 BAA 的每一步（尤其是计算辅助函数 W 与输出分布 Y）实现了块级并行化；
2) 结合先前的空间时间折衷方法，使用递归预处理表和动态规划的子序列/超序列枚举技术，将单线程枚举耗时从 O(N) 降至 O(k)；
3) 通过上述技术将可计算的输入长度从 28 提升到 31，并将高噪声下的容量上界从 0.3745(1‑d)（d≥0.68）进一步降低到 0.3578(1‑d)（d≥0.64）。

**🔧 技术方法**

主要技术手段包括：
- GPU 并行化（CUDA）实现 BAA 的迭代；
- 递归预处理表和动态规划的子序列/超序列“unranking”算法；
- 通过对转移概率的稀疏化和缓存表实现时间空间折衷；
- 结合 Rahmati‑Duman 递推式将局部容量上界推广到整个删除概率区间。

**📊 数据集**

该工作不使用传统意义上的实验数据集，而是在理论模型下进行符号序列枚举和容量数值逼近；所有实验均在 NVIDIA RTX 5070 Ti GPU 上执行。

**📈 对比分析**

与之前的最佳结果（C(d)≤0.3745(1‑d) 于 d≥0.68）相比，本文在高噪声区间取得了显著提升，新的上界 C(d)≤0.3578(1‑d) 对 d≥0.64 成立。对有限长度通道的容量上界也在 n≤31 时实现了更精确的数值；但计算时间较长，尤其在 k≈n/2 时单次迭代约 400–800 秒，整个过程耗时数天至数周。

**⚠️ 局限性**

主要局限包括：
- 计算复杂度仍随 n、k 指数级增长，导致对更大 n 或 k 的探索受限；
- 对 k>18 的 C_31,k 仅能给出较松的上界，需要更大容差；
- GPU 资源限制导致并行化规模受限，内存不足时无法完整枚举所有子/超序列；
- 结果仍为数值上界，尚未提供解析式或更通用的理论改进。

---

## 469. AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning

**arXiv ID:** 2604.05846 | [PDF](https://arxiv.org/pdf/2604.05846v1)

**作者:** Yuanfu Sun `[一作]` (New York University Shanghai), Qiaoyu Tan `[通讯]` (New York University Shanghai)

**通讯引用:** 1073 | [OpenAlex ID](https://openalex.org/A5043697901)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Agentic Graph Learning（AGL）框架，利用强化学习让 LLM 在图结构中进行自适应导航与推理，并通过图原生搜索工具与搜索约束思维实现多尺度探索。

**💡 创新点**

创新点包括：①将图学习转化为交互式代理决策过程；②设计了图原生搜索工具（1hop、2hop、全局重要性、语义密集检索）；③引入搜索约束思维与图条件化课程学习（GCCL）实现长周期策略训练与搜索效率权衡。

**🔧 技术方法**

核心技术包括：大语言模型（Qwen 3B/7B、LLama 等）、强化学习（GRPO、REINFORCE++）、图原生工具调用、奖励设计（格式、准确、覆盖、认知密度）、图条件化课程学习。

**📊 数据集**

实验数据集为七个文本属性图（TAG）：OGB-Arxiv、OGB-Products、PubMed、Amazon-Photo、Amazon-Computers、Arxiv-2023、Reddit，涵盖学术、商业与社交网络。

**📈 对比分析**

与 GNN（GCN、SAGE、RevGAT）、GraphLLMs（GraphGPT、GraphPrompter、GraphICL、LLaGA）、GraphRAG（LinearRAG、HippoRAG2、GraphCoT）、标准代理搜索（Search-R1、Search-O1）等基线比较。AgentGL 在节点分类上平均提升 17.5%，在边预测上平均提升 28.4%，在多种 LLM 大小与零-shot 迁移实验中均优于所有基线。

**⚠️ 局限性**

局限性：仅支持文本属性图，无法处理多模态节点；MSO 阶段对数据划分和超参数敏感；对稠密图和更大规模图的适用性尚待验证。

---

## 470. JD-BP: A Joint-Decision Generative Framework for Auto-Bidding and Pricing

**arXiv ID:** 2604.05845 | [PDF](https://arxiv.org/pdf/2604.05845v1)

**作者:** Linghui Meng `[一作]` (JD.com), Ching Law `[通讯]` (JD.com)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种联合生成式决策框架JD‑BP，能够同时输出投标金额与定价校正项，提升自动投标在KPI约束下的价值最大化与预算效率；

**💡 创新点**

创新点包括：①将价值最大化与约束补偿解耦为投标与定价两动作；②引入无记忆Return‑to‑Go（RTG）消除历史约束偏差；③设计门控交叉注意力（GCA）让定价动作精准响应投标决策；④通过轨迹增强算法从已有投标策略快速生成联合轨迹；⑤采用能量基直接偏好优化（Energy‑Based DPO）提升模型对高质量投标轨迹的偏好；

**🔧 技术方法**

技术方法涵盖：Decision Transformer双流架构、交叉注意力模块、PID控制生成定价轨迹、能量基DPO细调、线性预算约束与ROI约束的Lagrangian解析；

**📊 数据集**

数据集与实验：使用Alibaba公开AuctionNet数据进行离线评估，并在JD.com广告平台进行在线A/B测试；

**📈 对比分析**

与传统RL（BC、CQL、IQL）和生成式（DT、GAVE、DiffBid）基线对比，离线得分平均提升至184.28（最高为197.06），在线实验广告收入+4.70%，目标成本+6.48%；

**⚠️ 局限性**

局限性：①初始训练需PID生成定价轨迹，依赖先验策略；②在严格预算约束场景下JD‑BP可能因补偿定价导致预算延长；③对超参数敏感，需在不同市场环境中重新调优；

---

## 471. Proof of Concept as a First-Class Architectural Decision Instrument

**arXiv ID:** 2604.05835 | [PDF](https://arxiv.org/pdf/2604.05835v1)

**作者:** Bruno Fernando Antognolli `[一作]` (École de technologie supérieure), Fabio Petrillo `[通讯]` (École de technologie supérieure)

**通讯引用:** 982 | [OpenAlex ID](https://openalex.org/A5008013233)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Proof of Concept（PoC）在软件架构中的精确定义，并构建了轻量级的三阶段（规划、执行、决策）PoC实施框架；

**💡 创新点**

创新点在于将PoC定位为一等建筑决策工具，提出了“未记录的建筑实验”反模式，并提供可复用的PoC文档模板，弥补了现有研究对PoC过程缺失的空白；

**🔧 技术方法**

技术包括系统性文献回顾、灰色文献分析、BPMN流程建模、指标监控与实验场景设计，框架以可视化表单和Google表格实现；

**📊 数据集**

数据集来源于两项工业案例：①银行系统的数据库版本工具（Flyway与Liquibase）实验数据；②公开的Java（Quarkus）与Golang（Fiber）Kubernetes性能基准数据；

**📈 对比分析**

比较方法：在同一测试场景下测量迁移时间、审核日志完整度、回滚效果（银行案例）以及延迟、启动时间、镜像尺寸、资源利用率（性能案例）；结果显示Liquibase在审核与回滚上优于Flyway，但Flyway更易用；Golang在延迟、启动与镜像大小方面优于Java，适合云原生部署；

**⚠️ 局限性**

局限性包括案例数量有限、仅针对两类技术环境、实验需专业资源、缺乏长期纵向验证，框架在极端规模或资源受限团队中的适用性仍待评估。

---

## 472. WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering

**arXiv ID:** 2604.05818 | [PDF](https://arxiv.org/pdf/2604.05818v1)

**作者:** Yingjian Zhu `[一作]` (University of Chinese Academy of Sciences), Shiming Xiang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 14320 | [OpenAlex ID](https://openalex.org/A5040673285)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 WikiSeeker——一种多模态检索增强生成框架，用视觉语言模型做查询扩展（Refiner）和检索验证（Inspector），实现检索-生成解耦；

**💡 创新点**

1）把 VLM 重新定义为 Refiner 与 Inspector，而非单纯的答案生成器；2）使用强化学习（GRPO）优化 Refiner 的查询重写；3）引入检索验证的 Inspector 决定是否交给 LLM 生成答案；

**🔧 技术方法**

多模态密集检索（可加权融合视觉/文本特征）、强化学习（GRPO）、结构化生成（CoT+JSON）、对比检索和生成策略（Decoupled Generation）；

**📊 数据集**

EVQA、InfoSeek、M2KR 三大 KB‑VQA 基准；

**📈 对比分析**

在 Retrieval 上通过 Refiner 提升 Recall@1~@20，取得所有数据集 SOTA；在 VQA 上结合 Inspector 与 LLM，EVQA 上达 55.62%、InfoSeek 44.72%，均超过现有最优方法；

**⚠️ 局限性**

解耦路由规则过于硬性，未能充分利用 LLM 与 VLM 的协同；仅支持单轮检索，无法处理多跳问题；

---

## 473. Gated-SwinRMT: Unifying Swin Windowed Attention with Retentive Manhattan Decay via Input-Dependent Gating

**arXiv ID:** 2604.06014 | [PDF](https://arxiv.org/pdf/2604.06014v1)

**作者:** Dipan Maity `[一作]`, Arindam Roy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Gated-SwinRMT，融合Swin Transformer的窗口注意力与Retentive网络的曼哈顿距离衰减，并加入输入依赖门控；

**💡 创新点**

通过在窗口内分解宽高维保留（Retention）以及两种门控机制（SwiGLU+Sigmoid门和G1门）来缓解窗口软max导致的注意力压力，并在不同位置编码方案上实现无学习位置偏置；

**🔧 技术方法**

使用窗口化自注意力、Manhattan距离指数衰减、可学习的ALiBi/RoPE位置编码、DWConv位置编码、局部上下文增强(LCE)、SwiGLU值变换、Sigmoid/Softmax门控等技术；

**📊 数据集**

在Mini-ImageNet（224×224，100类）和CIFAR-10（32×32，10类）两个基准上进行实验；

**📈 对比分析**

与原始RMT基线在相同参数规模（≈77–79 M）下对比，Mini-ImageNet上SWAT达80.22%准确率，比RMT提升6.48个百分点，Retention提升4.46个百分点；在CIFAR-10上提升幅度大幅压缩，仅+0.56个百分点；

**⚠️ 局限性**

实验仅在单GPU上完成，未评估ImageNet-1k及密集预测任务；采用的是间接ablation（整体模型对比），缺乏单组件消融和 FLOPs‑性能分析；对训练‑验证差距仍有一定残余。

---

## 474. FinReporting: An Agentic Workflow for Localized Reporting of Cross-Jurisdiction Financial Disclosures

**arXiv ID:** 2604.05966 | [PDF](https://arxiv.org/pdf/2604.05966v1)

**作者:** Fan Zhang `[一作]` (MBZUAI), Zhuohan Xie `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 FinReporting，一套用于跨司法管辖区金融报告的代理工作流，将原始披露数据（XBRL 或 PDF）统一映射到标准化的收入表、资产负债表和现金流量表，并在每个处理阶段提供可审计的验证与异常记录。

**💡 创新点**

创新点在于：①将 LLM 角色从自由生成器转变为受约束的验证器，仅在有充分证据时才修正结果；②构建跨市场的全局财务本体，实现语义一致的映射；③将整个流程拆分为可审计的若干阶段（获取、识别、提取、映射、输出），并通过规则+LLM+人工复核形成闭环。

**🔧 技术方法**

技术主要包括：规则驱动的结构化提取（针对 XBRL 和 PDF），基于全局本体的 Canonical Mapping，LLM 验证层（带门控决策和证据记录），以及可视化演示界面和审计日志系统；使用的 LLM 主要是开源或高效模型，如 GPT‑4‑Turbo 或其同等体。

**📊 数据集**

数据集涵盖美国（SEC EDGAR 10‑K XBRL）、日本（EDINET XBRL）和中国（公开 PDF 年报）三国的 90 家非金融公司年报（30 家/国），共 18 个核心财务字段，包含手工标注的黄金标准。

**📈 对比分析**

方法评估使用填充率（FR）、冲突率（CR）和准确率（ACC）。在 US、JP、CN 评估中，FR 均高于 90%，CR 在 5–15% 之间，ACC 接近 90%；在同类 Naïve LLM 报告流水线对比中，FinReporting 的 FR/ACC 稍低但 CR 明显更低，表明其验证机制有效降低误差。

**⚠️ 局限性**

局限性：仅覆盖三国的年度披露和 18 个核心字段，无法覆盖其他司法管辖区、季度/分部披露或细粒度项目；PDF 中的布局不规则会导致提取不完整；本体映射不一定涵盖所有地区特有的会计细节；最终仍需人工复核以确保高风险决策的准确性。

---

## 475. Beyond Compromise: Pareto-Lenient Consensus for Efficient Multi-Preference LLM Alignment

**arXiv ID:** 2604.05965 | [PDF](https://arxiv.org/pdf/2604.05965v1)

**作者:** Renxuan Tan `[一作]` (Zhejiang University), Honggang Zhang `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 12303 | [OpenAlex ID](https://openalex.org/A5100626780)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为 Pareto‑Lenient Consensus (PLC) 的多偏好 LLM 对齐框架，通过动态协商机制使模型在梯度冲突时允许短期退步，从而跳出风险厌恶的局部平衡点。

**💡 创新点**

创新点在于：① 将每个偏好视为合作游戏中的玩家；② 设计了基于联盟余量的“lenient mask”对冲突梯度进行可变滤波；③ 证明该机制可在收敛时消除偏差，最终达到 Pareto 共识平衡。

**🔧 技术方法**

技术方案包括：多头价值网络与向量化优势估计；PPO 换算为 lenient manifold 的目标函数；使用联盟余量 S⁻_k 作为动态阈值的掩码；以及基于 τ 温度的 sigmoid 软阈值。

**📊 数据集**

实验数据集主要有 Anthropic‑hh‑rlhf 与 BeaverTails‑Subset，使用 Llama‑3.1‑8B/1B 并辅以公开代理奖励模型（harmless, helpful, humor）及 DeepSeek‑V3.2 的 LLM‑Judge。

**📈 对比分析**

与 SOLO、RS、GAPO、RiC 等基线相比，PLC 在 LLM‑Judge 与代理奖励上的平均得分均更高，Hypervolume、IGD、Max. Spread 等多目标指标也明显优于对手，说明其 Pareto 前沿更广且收敛更快。

**⚠️ 局限性**

局限性包括：缺乏统一评估协议导致难以验证真正的 Pareto 最优；依赖代理奖励模型，若模型偏差会被放大；lenient 机制的衰减策略未完全确定，可能在高度噪声环境下持续激活。

---

## 476. Governance and Regulation of Artificial Intelligence in Developing Countries: A Case Study of Nigeria

**arXiv ID:** 2604.06018 | [PDF](https://arxiv.org/pdf/2604.06018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 477. Does Pass Rate Tell the Whole Story? Evaluating Design Constraint Compliance in LLM-based Issue Resolution

**arXiv ID:** 2604.05955 | [PDF](https://arxiv.org/pdf/2604.05955v1)

**作者:** Kai Yu `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**通讯引用:** 14313 | [OpenAlex ID](https://openalex.org/A5071724015)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了新的设计意识型问题修复基准 SWE-SHIELD，系统地从真实项目的 Pull Request 与代码审查中提取隐式设计约束，并通过 LLM 评判生成补丁是否符合这些约束，进而评估 LLM 及其代理在实际软件维护中的设计合规性。

**💡 创新点**

创新点包括：① 以 LLM 两阶段方法自动抽取并聚合设计约束，使得项目内部的非显式设计知识可量化；② 将约束与问题实例关联并手工验证，构建 495 个问题与 1,787 条高质量设计约束的 benchmark；③ 采用 LLM-as-Judge 多模型投票机制实现自动化的设计合规验证，弥补传统测试覆盖不足的缺陷。

**🔧 技术方法**

主要技术：多模态 LLM（如 Gemini‑3、Claude‑Sonnet‑4.5 等）用于约束抽取与验证；滑动窗口 + 语义聚类的两阶段设计约束抽取；基于句向量的相似度匹配实现问题–约束关联；LLM 多模型投票判定 patch 是否满足约束；实验中使用 Pass Rate、DSR、DVR 等指标。

**📊 数据集**

数据集：从 SWE‑Bench‑Verified 与 SWE‑Bench‑Pro 两大基准中筛选 6 个开源仓库，最终得到 495 个问题实例，关联 1,787 条手工验证的设计约束；实验还使用了 4 种前沿基础模型（Kimi‑K2、GPT‑5、Claude‑Sonnet‑4.5、Gemini‑3.0‑Pro）及对应的 LLM‑based agents（SWE‑agent、Live‑SWE‑agent、Lingxi‑v1.5、Sonar Foundation Agent）。

**📈 对比分析**

比较方法：对比 Pass Rate 与设计合规率（DSR）以及违规率（DVR），并在四种基础模型上进行模型中心比较；对比提供设计约束前后补丁的性能变化。结果显示：Pass Rate 高（最高 75.95%），但 DSR 仅 32.64%–50.20%，DVR 常超过 35%；功能正确性与设计合规性几乎不相关；提供约束能降低 DVR（平均 6%），但仍高于 30%，且对 Pass Rate 影响不一。

**⚠️ 局限性**

局限性：① 约束抽取与验证高度依赖 LLM，可能出现幻觉和误检；② 仅覆盖 6 个仓库，难以完全体现企业级大规模项目的多样性；③ “金标准”补丁在设计合规方面可能已不完美；④ 设计约束的抽象与可验证性仍有限，难以覆盖所有 tacit 设计决策。

---

## 478. Edge Intelligence for Satellite-based Earth Observation: Scheduling Image Acquisition and Processing

**arXiv ID:** 2604.05937 | [PDF](https://arxiv.org/pdf/2604.05937v1)

**作者:** Beatriz Soret `[一作]` (Universidad de Málaga), Israel Leyva-Mayorga `[通讯]` (Aalborg University)

**通讯引用:** 869 | [OpenAlex ID](https://openalex.org/A5061754527)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于LEO卫星网络的能耗感知观测与边缘计算调度框架，实现了实时语义处理与海上目标检测。

**💡 创新点**

创新点在于将大气湍流感知融入观测调度并联合观测-计算-通信优化，提出两阶段二进制整数优化与凸化处理调度模型，显著提升观测收益与能耗效率。

**🔧 技术方法**

使用了YOLOv8深度学习模型、BSP并行模型、Gamma分布与CLT时间估计、混合整数规划与凸优化、LEO ISL/FSSRF通信模型等技术。

**📊 数据集**

基于Sentinel‑2/WorldView‑3卫星图像及海上目标检测数据集，并使用实验测得的C_n^2分布进行湍流模拟。

**📈 对比分析**

与FIFO、GA基线比较，观测收益提升约78%，气象条件下成功率达到80‑90%，边缘GPU实现能耗比CPU低30倍，系统总能耗显著下降。

**⚠️ 局限性**

局限性包括对大气湍流模型与阈值的依赖、未考虑多目标优先级与网络时延漂移、模型对实时场景扩展性有限，以及仅在单一船舶检测任务上验证，通用性待进一步验证。

---

## 479. BiMind: A Dual-Head Reasoning Model with Attention-Geometry Adapter for Incorrect Information Detection

**arXiv ID:** 2604.06022 | [PDF](https://arxiv.org/pdf/2604.06022v1)

**作者:** Zhongxing Zhang `[一作]` (University of Minnesota, Twin Cities), Jaideep Srivastava `[通讯]` (University of Minnesota, Twin Cities)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BiMind 双头框架，将文本内部推理与外部知识增强推理分离，用以检测错误信息。

**💡 创新点**

创新点包括：注意力几何适配器（AGA）重塑注意力分布、基于 kNN 的自检索知识模块与 FiLM 注入、基于熵门控与同义 KL 正则的双头融合策略，以及量化知识贡献的 VoX 指标。

**🔧 技术方法**

核心技术为 Transformer 编码器、AGA、kNN检索、FiLM 层、熵门控融合、可训练的同意头、对称 KL 正则化，以及预训练 LLaMA-7B 和 SentenceTransformer 嵌入。

**📊 数据集**

使用四个公开数据集：MM COVID、ReCOVery（健康领域）、LIAR（新闻真实性）和 MC Fake（多域），进行实验评估。

**📈 对比分析**

与 CNN、GCN、BERT、HAN、HeteroSGT 等基准模型对比，BiMind 在大多数数据集上在准确率、召回率、F1 等指标上均优于对手，且在知识贡献评估上表现更佳。

**⚠️ 局限性**

局限性包括：AGA 依赖词性特征，对低信息量输入可能不足；未融入社交信誉或传播模式；在检测错误时可能削弱正确信息流。

---

## 480. How LLMs Follow Instructions: Skillful Coordination, Not a Universal Mechanism

**arXiv ID:** 2604.06015 | [PDF](https://arxiv.org/pdf/2604.06015v1)

**作者:** Elisabetta Rocchetti `[一作]` (Università degli Studi di Milano), Alfio Ferrara `[通讯]` (Università degli Studi di Milano)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在三种指令微调模型上，对九种结构、词汇、语义、风格任务进行诊断性探针实验，评估指令遵循机制是否为通用约束满足或组合技能部署。

**💡 创新点**

提出结合专家与通用探针、跨任务转移、INLP因果消融、时序分析与PWCCA聚类的完整诊断框架，证明指令遵循主要是多技能协同而非单一约束检测。

**🔧 技术方法**

使用线性与非线性探针、迭代零空间投影(INLP)、投影加权典型相关分析(PWCCA)、因果消融、生成进程时序抽取等技术。

**📊 数据集**

构造九个任务的数据集，涵盖字符/词计数、JSON格式、词汇包含/排除、主题、情感、毒性、正式度等，数据来源为提示模板与LLM生成示例。

**📈 对比分析**

将通用探针与任务专家探针比较，发现通用探针平均准确率低于专家；跨任务转移和消融表明任务间信息稀疏、非对称；时序分析显示约束满足在生成中动态监控而非预先规划；整体性能在不同模型间相差不大，但表明任务特定技能占主导。

**⚠️ 局限性**

局限包括跨层对齐误差、模型规模影响、仅测试单一约束任务、诊断性方法缺乏实际干预或多约束评估、任务覆盖范围有限等。

---

## 481. Designing Around Stigma: Human-Centered LLMs for Menstrual Health

**arXiv ID:** 2604.06008 | [PDF](https://arxiv.org/pdf/2604.06008v1)

**作者:** Amna Shahnawaz `[一作]` (Lahore University of Management Science), Maryam Mustafa `[通讯]` (Lahore University of Management Science)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在巴基斯坦低资源环境中，基于大型语言模型与检索增强生成技术的 WhatsApp 聊天机器人，提供周期健康教育服务。

**💡 创新点**

提出了围绕污名的设计框架，融合本土语言（罗马乌尔都语）、专家验证的知识库和用户共创的交互模式。

**🔧 技术方法**

采用 OpenAI 的 GPT‑4o 与 Assistants API，并实现检索增强生成（RAG）以支撑对话生成。

**📊 数据集**

使用高校女学生共创收集的问答数据以及妇科专家编写的医学知识库作为训练与检索语料。

**📈 对比分析**

通过两周的真实使用（共 403 条消息）及访谈评估，显示用户对可信度与信息获取的积极反馈，但未对比传统搜索或其他模型的数值性能，主要关注用户体验与信任建立。

**⚠️ 局限性**

受限于样本规模小、部署周期短、需持续人工审核与知识库维护，导致可扩展性与长期持续使用的可行性不明。

---

## 482. The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models

**arXiv ID:** 2604.05995 | [PDF](https://arxiv.org/pdf/2604.05995v1)

**作者:** Xiaojie Gu `[一作]` (Independent Researcher), Kai Zhang `[通讯]` (Ohio State University)

**通讯引用:** 7623 | [OpenAlex ID](https://openalex.org/A5100324053)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SA-MCQ 诊断框架，用来检验大语言模型在知识编辑后是否真正修改了内部参数，而非仅表面匹配。

**💡 创新点**

创新点在于揭示了“Surface Compliance”现象，指出传统生成式指标容易被欺骗；并展示递归编辑会在内部产生残留冲突，导致认知不稳定。

**🔧 技术方法**

技术包括自评多选问答（SA-MCQ）、似然差距分析、以及多种外部证据（Parametric Evidence、Golden Evidence、Irrelevant Evidence、Counter Evidence）来触发上下文对抗。

**📊 数据集**

使用的主要数据集为 UltraEditBench、Zero-shot Relation Extraction、以及通过外部大模型生成的证据文本。

**📈 对比分析**

与 AlphaEdit、RLEdit、UltraEdit 等主流编辑器在传统 Exact Match 与 LLM-as-Judge 指标下对比，结果显示这些编辑器在 SA-MCQ 下的黄金答案选择率显著低于表面指标，且递归编辑后性能进一步下降，说明传统评估高估了编辑效果。

**⚠️ 局限性**

限制：实验仅覆盖少量模型规模与数据集，未能扩展到更大参数或多样化知识库，计算资源有限导致实验范围受限。

---

## 483. Multi-Modal Landslide Detection from Sentinel-1 SAR and Sentinel-2 Optical Imagery Using Multi-Encoder Vision Transformers and Ensemble Learning

**arXiv ID:** 2604.05959 | [PDF](https://arxiv.org/pdf/2604.05959v1)

**作者:** Ioannis Nasios `[一作]` `[通讯]` (Nodalpoint Systems), Ioannis Nasios (Nodalpoint Systems)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种多模态的滑坡检测框架，将后事件 Sentinel‑2 光学影像与前后事件 Sentinel‑1 SAR 数据通过多编码器视觉 Transformer 处理，并结合梯度提升模型进行集成学习，实现了对滑坡的高精度分类。

**💡 创新点**

创新点包括：① 使用多编码器架构让每种传感器数据单独编码后再融合，充分挖掘光学与雷达的互补信息；② 将深度网络与传统 GBM（LightGBM、XGBoost）混合成九模型集成，提升鲁棒性与泛化能力；③ 在光学通道中加入 NDVI、NDWI 等衍生指数作为独立模态，进一步增强特征表达。

**🔧 技术方法**

技术手段主要为：多编码器视觉 Transformer（ViT、MaxViT、CaFormer 等预训练模型）、光学与 SAR 渠道的统计特征提取用于 GBM、融合训练策略（5‑fold CV + OOF 校准）、自定义损失函数（BCE+SmoothF1）以及阈值校准（0.49）等。

**📊 数据集**

使用了在 Zindi 竞赛公开的数据集，包含 7,147 张 64×64 像素的 12 通道（光学 4 通道 + SAR 8 通道）滑坡与非滑坡标签，训练集与测试集按 30/70 公共/私有 Leaderboard 划分。

**📈 对比分析**

与竞赛中现有方法相比，该集成模型获得 F1 0.919（在私有 LB 上 0.905，公共 LB 0.906），超过主流方法（如 0.835–0.910 的 F1），并在多种单模型比较中证明多编码器、Transformer 与 GBM 的组合显著提升性能。

**⚠️ 局限性**

局限性包括：① 训练与推理时间较长，尤其是多编码器网络需 GPU；② 数据集来源为竞赛样本，缺乏真实场景的地理与时间信息，可能导致泛化性不完全；③ 只使用后事件光学图像，未利用预事件光学或 InSAR 变化信息；④ 集成学习提高了复杂度与部署成本。

---

## 484. You're Pushing My Buttons: Instrumented Learning of Gentle Button Presses

**arXiv ID:** 2604.05954 | [PDF](https://arxiv.org/pdf/2604.05954v1)

**作者:** Raman Talwar `[一作]` (Ghent University---imec), Francis wyffels `[通讯]` (Ghent University---imec)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究利用训练时的仪器化（按钮状态）来细化音频编码器，并将其融入到仅使用视觉和音频的深度融合或软传感器策略中，以实现按钮按压任务。

**💡 创新点**

创新点在于将仪器化仅作为训练阶段的临时监督通道，对音频表征进行细化，从而在部署时无需依赖特权传感器即可获得更温和的接触力，同时保持成功率。

**🔧 技术方法**

使用Diffusion Policy进行模仿学习，预训练的Audio Spectrogram Transformer (AST) 通过按钮状态标签进行微调，采用深度融合（直接使用AST输出或中间嵌入）和软传感器（使用AST预测替代真实按钮状态）两种集成策略；还使用了RGB相机和指尖麦克风的多模态观测。

**📊 数据集**

数据来源为UR3e机械臂在实验室环境中自动收集的演示数据，音频特征预训练使用AudioSet；任务数据量约为40次rollout测试集。

**📈 对比分析**

通过对比三种策略与基线（未细化AST）在40次rollout中的成功率和接触力指标，成功率相似（45–55%），但深度融合策略（细化嵌入）在峰值竖向力和Wasserstein距离上表现最佳（median ≈ 5.9 N，W ≈ 2.5 N），而软传感器表现最差。

**⚠️ 局限性**

局限性包括仅在按钮按压任务上验证，未检验对不同按钮、机械结构或音频环境的泛化能力；实验硬件受限，且未探索更紧密的音视频端到端融合或AST微调策略。

---

## 485. Polynomial-Time Algorithm for Thiele Voting Rules with Voter Interval Preferences

**arXiv ID:** 2604.05953 | [PDF](https://arxiv.org/pdf/2604.05953v1)

**作者:** Pasin Manurangsi `[一作]` (Google Research), Krzysztof Sornat `[通讯]` (AGH University)

**通讯引用:** 195 | [OpenAlex ID](https://openalex.org/A5051882903)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对 Voter Interval 域下的 Thiele 选举规则，提出并证明了一种多项式时间算法，可在给定的委员会大小 k 下计算最优委员会。

**💡 创新点**

创新点在于首次给出了 Voter Interval 域的全域可解性，核心在于构造了一条针对区间族的凸性（concavity）定理，并借此实现 Lagrangian 约束松弛和总可整性证明。

**🔧 技术方法**

采用了 Lagrangian 约束松弛、完全可整性（TU）整数线性规划、以及区间族的凸性结构化分析等技术。

**📊 数据集**

由于研究为理论算法，没有使用具体数据集，所有结果均基于理论证明。

**📈 对比分析**

在算法上没有实验评估，性能以多项式时间复杂度证明为主，显示可在多项式时间内完成优化。

**⚠️ 局限性**

局限性在于算法高度依赖线性规划求解器和 TU 性质，缺乏纯粹组合式的实现，且目前未扩展至更一般的非间隔域。

---

## 486. Towards Trustworthy Report Generation: A Deep Research Agent with Progressive Confidence Estimation and Calibration

**arXiv ID:** 2604.05952 | [PDF](https://arxiv.org/pdf/2604.05952v1)

**作者:** Yi Yuan `[一作]` (Shanghai Artificial Intelligence Laboratory), Shanzhe Lei `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 7106 | [OpenAlex ID](https://openalex.org/A5100427516)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种深度研究代理，能够将长篇报告生成拆解为可验证的问答子任务，并在生成过程中对每条主张进行递进的置信度估计和校准，从而提升报告的可信度与透明度。

**💡 创新点**

核心创新在于：① 将报告生成转化为“思考‑检索‑阅读”循环的递进式问答流程；② 在此循环中内置置信度预测头，使置信度随证据检索和推理动态演变；③ 通过模块化规划‑检索‑撰写的三阶段流水线，将问答子任务的可靠性直接嵌入最终报告。

**🔧 技术方法**

主要技术包括：多轮检索与推理的Deliberative Search Model（Think→Search→Read）；强化学习训练下的置信度预测头；基于LLM的规划器、检索器和撰写器；自我反思与证据验证机制；以及对置信度进行可视化与输出的置信度标注。

**📊 数据集**

使用的数据集：GPQA‑Diamond 和 xBench‑DeepSearch 用于评估问答子任务的准确性与校准性能；DeepResearch Bench (DRB) 用于整体报告质量评估；此外实验中还引用了公开的检索工具和多模态数据以支持检索与推理。

**📈 对比分析**

与多种行业内已公开或闭源深度研究代理及具搜索功能的大语言模型（如Gemini‑2.5‑Pro、Claude‑3.7‑Sonnet、OpenAI Deep Research等）比较，Deliberative Search 在GPQA‑Diamond上达到 61.62% 的准确率、在 xBench‑DeepSearch 上取得 0.34 的 N_ECE，均优于多数基线；在 DRB 上，报告质量得分处于中游水平，综合评价与现有代理相当，表明在保持可信度的同时可达到可接受的内容质量。

**⚠️ 局限性**

局限性包括：① 在整体报告质量上仍落后于某些闭源代理，尤其在洞察深度与指令遵循方面；② 置信度估计依赖于模型内部表示，未能完全消除幻觉或推理误差；③ 需要额外的规划与反思模块，增加系统复杂度与计算成本；④ 对极端高风险领域（如医疗、金融）仍需更严格的验证与监管框架。

---

## 487. Mixture-of-Modality-Experts with Holistic Token Learning for Fine-Grained Multimodal Visual Analytics in Driver Action Recognition

**arXiv ID:** 2604.05947 | [PDF](https://arxiv.org/pdf/2604.05947v1)

**作者:** Tianyi Liu `[一作]` (Nanyang Technological University), Kim-Hui Yap `[通讯]` (Nanyang Technological University)

**通讯引用:** 2629 | [OpenAlex ID](https://openalex.org/A5022468480)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种结合Mixture-of-Modality-Experts（MoME）与Holistic Token Learning（HTL）的多模态驾驶员动作识别框架，能够自适应地融合RGB、IR与Depth三种模态并细粒度地提炼动作信息。

**💡 创新点**

创新点在于：①基于输出感知的门控机制实现模态专家的动态协作；②HTL在类标记和时空token层面同时进行自蒸馏与互蒸馏，提升细粒度特征表达与跨专家知识传递。

**🔧 技术方法**

使用视频Transformer（UniformerV2）作为每个模态专家的骨干网络，配合自定义的门控模块、KL散度与MSE蒸馏损失，以及AdamW优化器和余弦学习率衰减。

**📊 数据集**

在Drive&Act多模态驾驶员动作数据集（RGB、IR、Depth）上进行实验，评估34个细粒度动作类别。

**📈 对比分析**

与单模态、早/晚融合以及最近的多模态方法（Multifuser、CM²-Net）相比，MoME+HTL在Mean‑1准确率提升约1.7%，在Top‑1准确率也有小幅提升，表明在稀有类别上的鲁棒性更强。

**⚠️ 局限性**

局限性包括：①对更多模态（如LiDAR、声学）的适配尚未验证；②HTL的计算开销较大，尤其在更深的Transformer结构中；③对极端光照/遮挡条件下的泛化仍有待进一步提升。

---

## 488. MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.05943 | [PDF](https://arxiv.org/pdf/2604.05943v1)

**作者:** Maria Nesterova `[一作]` (AXXX & MIRAI), Alexey Skrynnik `[通讯]` (AXXX & MIRAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并训练了一个通用的Transformer架构MARL‑GPT，能在StarCraft、Google Research Football和POGEMA等多智能体环境中使用单一网络进行决策；

**💡 创新点**

创新点在于将Transformer与大规模专家演示结合，使用位置编码实现可变人数、团队身份和属性的统一输入，并在离线强化学习与模仿学习框架中实现跨域泛化；

**🔧 技术方法**

使用了Transformer编码器、离线RL（BC、CQL等）、行为克隆、离散化Q值、位置编码以及统一的输出层；

**📊 数据集**

使用了从IPPO和RHCR获得的专家轨迹，覆盖SMACv2（Protoss、Terran、Zerg对战）、Google Research Football（Pass&Shoot、Corner、Counterattack、11vs11）以及POGEMA的随机、迷宫、仓库、城市地图等；

**📈 对比分析**

通过与单域基线（BC、CQL、DT、RATE、BC‑LSTM）和专家模型比较，MARL‑GPT在所有测试任务上均达到或超过基线的胜率/吞吐率，尤其在SMACv2和GRF上与专家相当；

**⚠️ 局限性**

局限性包括对高质量专家数据的奖励损失可能导致不稳定；对未见的代理数量或地图需要额外微调；以及对更大规模环境和长时间任务的可扩展性仍待验证。

---

## 489. Context-Value-Action Architecture for Value-Driven Large Language Model Agents

**arXiv ID:** 2604.05939 | [PDF](https://arxiv.org/pdf/2604.05939v1)

**作者:** TianZe Zhang `[一作]` (Peking University), Guojie Song `[通讯]` (Peking University)

**通讯引用:** 6083 | [OpenAlex ID](https://openalex.org/A5088976879)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Context‑Value‑Action（CVA）架构，用来训练LLM基于真实人类价值观的行为模拟，并通过“Generate‑then‑Verify”方式抑制行为僵化。

**💡 创新点**

创新点在于：①将刺激‑机体‑反应模型与施瓦茨价值理论结合，显式建模情境激活的价值向量；②引入独立的Value Verifier，拆分生成与验证过程，避免LLM自我验证的偏见；③使用SFT＋DPO校正价值‑行为映射，降低价值极化；④在大规模真实交互数据上构建CVABench作为客观评测基准。

**🔧 技术方法**

技术包括：大语言模型微调（Supervised Fine‑Tuning）、直接偏好优化（Direct Preference Optimization）、生成‑选择（Generate‑then‑Select）机制、价值验证器训练、GPV（Generative Psychometrics for Values）自动心理测评。

**📊 数据集**

数据集为CVABench，包含约1.1 M条真实交互轨迹，来源于Yelp、Foursquare和Reddit，共15 571名独立用户，涵盖社交媒体、对话和空间移动三大领域。

**📈 对比分析**

与传统角色扮演、Prompt‑Reasoning、SFT、DPO等基线在个人行为一致性和群体价值分布上对比，CVA在各项指标（TTR、准确率、MSE、方差偏差）上均优于或相近基线，且显著降低价值极化和行为多样性丧失，证明其在行为忠实度与可解释性上具有优势。

**⚠️ 局限性**

局限性包括：1）实验域有限（仅三类领域、约15k用户），需要更广泛的跨文化与细粒度场景验证；2）价值测评仍依赖GPV，可能携带测量偏差；3）对比基线受算力限制，后续需扩展更多先进模型；4）模型训练与验证仍可能产生毒性或偏见内容，需要进一步安全过滤。

---

## 490. OmniCamera: A Unified Framework for Multi-task Video Generation with Arbitrary Camera Control

**arXiv ID:** 2604.06010 | [PDF](https://arxiv.org/pdf/2604.06010v1)

**作者:** Yukun Wang `[一作]` (Sun Yat-sen University), Qinglin Lu `[通讯]` (Hunyuan, Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出OmniCamera框架，实现视频生成中摄像机姿态与场景内容的解耦与组合控制。

**💡 创新点**

创新点在于三维条件RoPE分离不同模态、双层课程共训练策略以及统一处理九种条件组合的高参数效率设计。

**🔧 技术方法**

采用Diffusion Transformer、流匹配、Condition RoPE、双条件CFG、KV-Concat等技术实现多模态条件注入与精确摄像机控制。

**📊 数据集**

使用混合数据集OmniCAM，包括UE5合成视频与高精度轨迹的真实视频，并提供同场景/不同轨迹、同轨迹/不同场景以及三元组等多种对齐样本。

**📈 对比分析**

在多任务（T2V、I2V、V2V）下与多种基线对比，摄像机控制误差（TransErr/RotErr）降低约30%，FVD保持或略优，MotionAcc提升至90%以上。

**⚠️ 局限性**

局限在于仅支持整体运动或单一参考图像/视频的控制，无法实现多参考图像、局部细粒度或多目标的细致指导。

---

## 491. Disentangling MLP Neuron Weights in Vocabulary Space

**arXiv ID:** 2604.06005 | [PDF](https://arxiv.org/pdf/2604.06005v1)

**作者:** Asaf Avrahamy `[一作]` (Tel Aviv University), Mor Geva `[通讯]` (Tel Aviv University)

**通讯引用:** 1583 | [OpenAlex ID](https://openalex.org/A5065717258)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需前向传播、仅在权重空间进行旋转优化的算法，能够将自回归 Transformer MLP 神经元的权重分解为稀疏、可解释的词汇通道。

**💡 创新点**

创新点在于：①发现词汇投影的峰度（kurtosis）可作为判别单义方向的无监督指标；②利用Householder矩阵实现正交旋转，并通过最大化峰度和正则化相似度的双目标函数迭代获取多条通道；③通过掩码策略防止同一方向重复收敛。

**🔧 技术方法**

核心技术包括：Householder旋转、梯度下降优化、词汇投影、峰度计算、token掩码、通道消融验证，以及后续的通道描述聚合生成。

**📊 数据集**

实验使用 Gemma‑2‑2B‑it 和 Llama‑3.1‑8B‑Instruct 两个大型模型的权重；对比数据集包含 Pile（用于提取激活示例）和 FineWeb（用于评估描述泛化）。

**📈 对比分析**

与基于稀疏自编码器（SAE）和先前的 Gemma Scope、Llama Scope 方法相比，ROTT 在输入侧和输出侧的信度（faithfulness）和完整性（completeness）均提升 2–3 倍；在通道描述的头对头评估中，ROTT 的赢率显著高于 MaxAct+VocabProj 和 MaxAct++ 基线。

**⚠️ 局限性**

局限性包括：仅针对 MLP 神经元，未覆盖注意力头；需要对每个神经元单独优化，训练开销仍不小；在早层或分布不同的数据上可能分辨率不如中后层；峰度最大化对极端 token 的敏感性可能导致对“glitch tokens”需额外掩码。

---

## 492. Regimes of Scale in AI Meteorology

**arXiv ID:** 2604.06000 | [PDF](https://arxiv.org/pdf/2604.06000v1)

**作者:** Anya Martin `[一作]` (Georgia Institute of Technology), Cindy Lin `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 478 | [OpenAlex ID](https://openalex.org/A5090393849)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过对12名气象员的访谈，探讨了AI/ML技术与传统气象科学在观测、数据与模型三个环节上的尺度差异导致的摩擦，并提出了“尺度范式（regimes of scale）”框架来解释这些冲突。

**💡 创新点**

创新点在于：①将尺度（scale）视为核心冲突维度，揭示AI/ML与气象之间的“创业尺度”与“国家尺度”差异；②将HCI的领域框架与气象大数据体系结合，强调数据管道而非仅关注工具本身；③通过访谈案例具体呈现观测层面、数据层面、模型层面的摩擦与潜在机遇。

**🔧 技术方法**

论文主要讨论了已发布的AI天气模型（GraphCast、Pangu‑Weather、FourCastNet等），以及传统的气候/天气模型（ECMWF IFS、NCEP GFS）和重分析数据（ERA‑5）。技术层面并未构建新模型，而是分析这些模型所基于的技术与传统物理模型的不同尺度逻辑。

**📊 数据集**

使用的数据集主要是公共重分析数据 ERA‑5（6‑hourly、格点化），以及卫星遥感数据（如气象卫星辐射、气象平台数据）。访谈中还提到其他观测来源（地面站、气象气球、飞机、雷达、浮标）。

**📈 对比分析**

论文并未进行算法实验或性能对比，而是通过访谈质性分析比较了AI模型在速度、分辨率、可扩展性等“技术尺度”与传统模型在物理一致性、可解释性、跨域信任等“领域尺度”上的差异，指出AI模型在速度和分辨率上表现出显著优势，但在物理一致性和跨域可解释性方面缺陷，影响其在实际气象预报中的可接受度。

**⚠️ 局限性**

局限性包括：①样本仅来自美国，缺乏国际视角；②访谈规模有限（12人），无法覆盖所有气象组织和AI实验室的多样性；③未对AI模型的具体性能进行量化评估，只提供定性洞见；④聚焦于“尺度”框架，可能忽略了其他重要因素如数据治理、伦理与监管等。

---

## 493. Is CLIP Cross-Eyed? Revealing and Mitigating Center Bias in the CLIP Family

**arXiv ID:** 2604.05971 | [PDF](https://arxiv.org/pdf/2604.05971v1)

**作者:** Oscar Chew `[一作]` (Texas A&M University), Kuan-Hao Huang `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e0540dec-d77f-42db-94ae-d039248f6393` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统发现并定量评估了CLIP系列模型的中心偏差（center bias），并在已训练模型上提出两种训练无关的测试时策略（视觉提示和注意力重分配）来缓解该偏差。

**💡 创新点**

创新点在于首次将CLIP的空间位置偏差进行量化与机制剖析，揭示其根源是聚合过程中的信息损失，并提供两种可直接在已训练模型上使用的轻量级校正方法。

**🔧 技术方法**

主要技术包括可解释嵌入分解方法SpLiCE、注意力图分析、GroundingDINO检测+红色框视觉提示以及CLS注意力重分配。

**📊 数据集**

实验使用真实场景的What'sUp数据集以及自构造的GRID数据集（基于CIFAR‑10、Fashion‑MNIST、Food‑101）。

**📈 对比分析**

通过对中心与偏移子集的分类准确率进行对比，中心偏差导致约20–30%的性能下降；使用视觉提示平均提升7.9%，注意力重分配平均提升8.9%，总体准确率基本保持或提升。

**⚠️ 局限性**

局限性包括：方法仅在测试时起效，未修改模型参数；视觉提示效果不稳定；注意力重分配仅适用于CLS基CLIP；未解决训练阶段产生偏差的根本问题。

---

## 494. A Formal Security Framework for MCP-Based AI Agents: Threat Taxonomy, Verification Models, and Defense Mechanisms

**arXiv ID:** 2604.05969 | [PDF](https://arxiv.org/pdf/2604.05969v1)

**作者:** Nirajan Acharya `[一作]` (University of Cumberlands), Gaurav Kumar Gupta `[通讯]` (Youngstown State University)

**通讯引用:** 662 | [OpenAlex ID](https://openalex.org/A5007135458)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MCPShield，一个针对Model Context Protocol（MCP）的AI代理安全框架，涵盖统一威胁分类、形式化验证模型、对比评估12种现有防御以及四层防御体系设计；

**💡 创新点**

创新点在于：① 将177,000+ MCP工具实证数据整合为7类23个攻击向量的层级威胁分类；② 基于标记转换系统构建可判定的安全属性（工具完整性、数据隔离、特权边界、上下文隔离）；③ 系统比较12种防御，发现单一方案覆盖率≤34%；④ 设计四层防御（能力控制、加密工具鉴权、信息流追踪、运行时自动机）实现理论91%覆盖；

**🔧 技术方法**

采用的技术包括：形式化验证（标记转换系统+可信域标签）、加密签名与依赖哈希（工具鉴权）、信息流标签跟踪、能力访问控制、OAuth/ED25519身份验证、运行时安全自动机（编辑自动机）以及基准工具检测；

**📊 数据集**

使用的数据集为：177,000+ MCP工具的公开仓库数据；以及三大基准（MCPTox、MCP‑SafetyBench、MCPSecBench）用于评估防御覆盖；

**📈 对比分析**

通过将每个防御映射到威胁分类并计算覆盖比例来比较；单一防御最高覆盖34%；MCPShield四层组合覆盖91%；理论上每次工具调用增加4–9 ms延迟，远低于LLM推理和网络往返时延；

**⚠️ 局限性**

局限性包括：未实现原型或实测评估；仅覆盖协议层面，无法防御LLM内部推理攻击；覆盖率为理论估计，实际效果受实现细节影响；动态威胁演进与跨协议安全仍待研究；工具的语义完整性验证缺失。

---

## 495. QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization

**arXiv ID:** 2604.05963 | [PDF](https://arxiv.org/pdf/2604.05963v1)

**作者:** Changxin Ke `[一作]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences), Yunji Chen `[通讯]` (State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PRepair 框架，通过自监督生成多样化 bug 数据和基于 EA‑GRPO 的强化学习，显著降低 LLM 代码修复中的过度编辑问题，提升修复精度和可维护性。

**💡 创新点**

创新点在于（1）引入 Self‑Breaking 与 Self‑Repairing 两阶段数据生成与训练流程；（2）设计 Edit‑Aware Group Relative Policy Optimization（EA‑GRPO）动态编辑奖励；（3）提出 fix_p@k 指标，用于同时衡量修复正确性与编辑量。

**🔧 技术方法**

技术包括 LLM 自监督 bug 注入与 min‑max 采样、多样化训练样本；基于 RL 的 EA‑GRPO；以及 Speculative Editing（Prompt Lookup Decoding）提升推理吞吐量。

**📊 数据集**

数据集方面使用 Python 的 HumanEvalFix（164 例）及 LeetCodeDataset，Verilog 使用 QiMeng‑CodeV‑R1（352 例）生成的 Buggy 代码；此外自行构造的多样化 bug 数据集。

**📈 对比分析**

与 Prompt Engineering、GRPO 以及 GPT‑4、Gemini 等基线相比，EA‑GRPO 在 fix_1@1 上提升 20–31%，pass@1 维持或略增，跨域泛化更稳健；结合 Speculative Editing 后推理吞吐可提升约 15%。

**⚠️ 局限性**

局限性包括需要手动调参以获得最佳 α、β；仅针对函数级修复，尚未扩展至文件级或项目级更大范围的 Bug 修复。

---

## 496. HumANDiff: Articulated Noise Diffusion for Motion-Consistent Human Video Generation

**arXiv ID:** 2604.05961 | [PDF](https://arxiv.org/pdf/2604.05961v1)

**作者:** Tao Hu `[一作]` (Stability AI), Varun Jampani `[通讯]` (Stability AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出一种通用框架HumANDiff，用来实现基于视频扩散模型的可控长时序人体视频生成，能够在单张图像基础上生成运动连贯、细节丰富的视频；

**💡 创新点**

创新点包括：① 在统计人体模板表面空间上对噪声进行结构化采样（3D articulated noise warping），实现空间与时间一致的噪声；② 采用联合外观‑运动学习（JAML）让模型同时预测像素外观和对应物理运动；③ 引入几何运动一致性学习（GMCL），在噪声空间约束帧间运动一致性；

**🔧 技术方法**

技术手段：基于Latent Video Diffusion Models（如CogVideoX、Wan2.1）进行微调，利用SMPL人体模板进行噪声纹理映射、噪声退化/逆退化、UV空间逆warp与平均池化；并在训练中加入损失：Diffusion Loss + Motion Consistency Loss + Motion Decoding Loss；

**📊 数据集**

使用了UBC Fashion数据集（600段穿衣视频）进行训练与评估，同时在DeepFashion及公开野外视频上验证模型的跨域泛化能力；

**📈 对比分析**

与多种基线（GAN、姿态驱动扩散、Go‑With‑The‑Flow、Stable Video Diffusion、DreamPose等）对比，采用L1、SSIM、LPIPS、FID、FVD、FID‑VID等指标，结果显示HumANDiff在图像与视频质量指标上均明显优于对手，并能生成更长时长（49→81帧）且保持运动连贯；

**⚠️ 局限性**

局限性：模型仍依赖于SMPL人体模板，对极端姿态或极度遮挡时可能失真；在高分辨率或复杂背景下的细节还存在提升空间；微调过程对大规模高质量视频数据要求较高；

---

## 497. A Mixture of Experts Foundation Model for Scanning Electron Microscopy Image Analysis

**arXiv ID:** 2604.05960 | [PDF](https://arxiv.org/pdf/2604.05960v1)

**作者:** Sk Miraj Ahmed `[一作]` (Brookhaven National Laboratory), Chang-Yong Nam `[通讯]` (Brookhaven National Laboratory)

**通讯引用:** 5670 | [OpenAlex ID](https://openalex.org/A5018618972)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了首个面向扫描电子显微镜（SEM）的大规模基础模型，并在自监督的掩码自动编码器框架中引入混合专家（MoE）机制，能够从125,000张多仪器、多条件无标签SEM图像中学习通用特征，随后通过少量合成失焦‑聚焦样本对其进行微调，实现失焦到聚焦的图像重建；

**💡 创新点**

创新点在于（1）首次针对SEM数据设计并训练基础模型；（2）将Mixture-of-Experts嵌入ViT MAE，以实现对SEM多样纹理与噪声的专家化学习；（3）使用物理启发的椭圆Airy PSF和信号相关噪声模拟，生成合成失焦数据，避免稀缺的真实配对；（4）在零样本与少样本场景下，展示了模型在保持测量精度（CD、LWR、LER、PSD）方面的显著优势。

**🔧 技术方法**

技术包括Vision Transformer（ViT）大模型（24层、隐藏1024）、掩码自动编码器（MAE）自监督预训练、Mixture-of-Experts（MoE）顶层路由与专家分配、频域感知损失（FFT、PSD）以及细化阶段的Charbonnier、边缘一致与TV正则化；同时采用物理PSF模拟和Poisson+Gaussian噪声生成合成训练对。

**📊 数据集**

数据集由125,000张来自不同SEM仪器、能量、探测器、样品类型的无标签图像组成用于预训练；随后使用10张真实聚焦图像合成大量失焦-聚焦对进行微调；评估还使用数百张真实失焦-聚焦配对以及两张真实SEM图像的测量指标。

**📈 对比分析**

与传统鲁棒去卷积（RL、Wiener）、去噪（BM3D、Noise2Noise、Noise2Void）、任务专用网络（MRN）以及ImageNet预训练的ViT-MAE基线进行对比。结果显示，MAE+MoE在PSNR、SSIM、LPIPS、NIQE上均优于传统方法，且在测量指标（CD误差、粗糙度、PSD）上显著低于所有基线；零样本MoE已表现出强大性能，微调后进一步提升。

**⚠️ 局限性**

局限性包括：仅对单张图像进行推理，未利用多帧或时序信息；预训练输入固定为224×224，需滑窗拼接以处理高分辨率图像；合成失焦模型虽物理合理但可能未覆盖极端噪声或仪器特异性缺陷；模型在极端失焦或特殊材料下的泛化仍需进一步验证。

---

## 498. Saliency-Guided Representation with Consistency Policy Learning for Visual Unsupervised Reinforcement Learning

**arXiv ID:** 2604.05931 | [PDF](https://arxiv.org/pdf/2604.05931v1)

**作者:** Jingbo Sun `[一作]` (Institute of Automation, Chinese Academy of Sciences), Dongbin Zhao `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SRCP 框架，用来提升视觉无监督强化学习中的零样本任务泛化性能，方法通过去耦合表示学习与后继表示、利用显著性引导的动力学表示以及一致性策略来实现。

**💡 创新点**

创新点在于：①将表示学习与 SR 目标解耦，使用显著性引导的动力学任务专注于动态相关特征；②采用一致性模型与无条件引导的方式，显著提升多模态技能建模与技能可控性。

**🔧 技术方法**

采用的技术包括显著性引导的前向/逆向动力学学习、Hilbert 空间基本特征、后继特征学习、基于一致性模型的分类无关引导策略、离线 RL 训练以及多任务评估。

**📊 数据集**

使用 ExORL 基准中的 16 个视觉连续控制任务，涉及 Walker、Quadruped、Cheetah、Jaco 四个域，并利用 RND、PROTO、APS、APT 四个无监督数据集进行训练与评估。

**📈 对比分析**

与现有的成功后继方法（SF、FB）及多种表示学习方法（AE、CL、Lap、LRA-SR、FDM、HILP）进行对比，SRCP 在 16 个任务中平均提升 13%–33%，实现了状态‑最优的零样本泛化性能。

**⚠️ 局限性**

局限性在于目前仅在离线视觉 URL 场景表现良好，在线强化学习或更高维度/复杂视觉环境中的适用性尚未验证；对超参数（ω、β）较为敏感，需进一步稳健性研究。

---

## 499. ReLU Networks for Exact Generation of Similar Graphs

**arXiv ID:** 2604.05929 | [PDF](https://arxiv.org/pdf/2604.05929v1)

**作者:** Mamoona Ghafoor `[一作]`, Tatsuya Akutsu `[通讯]` (Japan Society for the Promotion of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造并实现了能够在给定图编辑距离约束下、完全确定性地产生任何符合该距离限制的顶点标记图的ReLU神经网络；该网络在输入图的标签矩阵和邻接矩阵上直接执行代价受限的编辑操作，并保证生成图的合法性。

**💡 创新点**

①证明存在常数层、规模为 O(n²d) 的ReLU网络可实现任何最多d步的替换、删除或插入操作；②将三种操作合并，得到能够在单个网络中生成距离不超过 d 的任意图的“GE_d”网络；③提供了完整的代数实现，使生成过程完全不依赖训练数据，保证生成图满足预定的编辑距离。

**🔧 技术方法**

使用ReLU激活函数实现的多层前馈网络，并通过一系列“max”“δ”“[ ]”等组合函数对输入序列进行解析、去重和约束，最终在网络输出端直接得到新的标签列和邻接矩阵；实验中仅使用 CPU 实现，未涉及深度学习框架的训练。

**📊 数据集**

实验数据主要为随机生成的顶点标记图，节点数从 100 到 1400，编辑距离从 10 到 140；另外在比较实验中使用了六个不同规模的图（n=10,20,30,40,50,100），每个图分别生成 500 个样本以评估编辑距离满足度。

**📈 对比分析**

与 GraphRNN 与 GraphGDP 两个流行的基于采样的图生成模型对比。结果显示：①GE_d 在所有实验配置下均能生成 500 个满足顶点数、边数范围以及编辑距离 ≤ d 的有效图；②GraphRNN 与 GraphGDP 均未能生成任何满足编辑距离约束的图；③在较大图（如 n=100）时，GraphRNN 仅 438/500 图保持顶点数，GraphGDP 仅 60/500 图满足边数范围。运行时间方面，GE_d 的时间随 n 与 d 增大而增长，d 的影响更显著，但仍能在 d≤140、n≤1400 的范围内完成。

**⚠️ 局限性**

①网络规模与深度随 n²d 增长，导致在极大图或极大编辑距离时内存与计算时间快速膨胀；②生成是确定性的，未实现均匀采样；③仅考虑顶点标记图，未处理无标签图或更复杂属性；④目前仅在 CPU 上实现，缺乏 GPU 加速优化。

---

## 500. On Dominant Manifolds in Reservoir Computing Networks

**arXiv ID:** 2604.05967 | [PDF](https://arxiv.org/pdf/2604.05967v1)

**作者:** Noa Kaplan `[一作]` (Cornell University), Anastasia Bizyaeva `[通讯]` (Cornell University)

**通讯引用:** 380 | [OpenAlex ID](https://openalex.org/A5072253048)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了训练过程如何在reservoir computing网络中形成低维主导流形，并通过线性连续时间模型解析其谱特性。

**💡 创新点**

首次将训练得到的主导模式与Koopman算子/动态模式分解（DMD）的特征值关联，提供了RC与DMD的理论桥梁。

**🔧 技术方法**

使用线性RC的谱分解、p‑dominance理论、Koopman/ DMD近似以及非线性tanh RC的扩展分析。

**📊 数据集**

以Goldbeter振荡器和Lorenz吸引子生成的时间序列作为训练/测试数据集。

**📈 对比分析**

通过对比线性与非线性RC在预测误差上的表现，发现非线性RC的主导模式更靠近不稳定边界，预测误差更低、预测范围更长。

**⚠️ 局限性**

理论分析仅针对线性对角RC，非线性情况仍是经验或未完成的严格证明；结果依赖于输入矩阵全秩、信息性假设，可能不适用于所有RC配置。

---

## 501. Toward Aristotelian Medical Representations: Backpropagation-Free Layer-wise Analysis for Interpretable Generalized Metric Learning on MedMNIST

**arXiv ID:** 2604.06017 | [PDF](https://arxiv.org/pdf/2604.06017v1)

**作者:** Michael Karnes `[一作]` (Ohio State University), Alper Yilmaz `[通讯]` (Ohio State University)

**通讯引用:** 8919 | [OpenAlex ID](https://openalex.org/A5008672128)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了A-ROM框架，利用预训练ViT的中间层特征通过无梯度层级分析实现可解释的少样本医学图像分类。

**💡 创新点**

创新点在于将普洛托利昂表征假说与阿里士多德式概念字典相结合，用距离度量和kNN替代梯度学习，形成可解释的“编码语言”。

**🔧 技术方法**

使用的技术包括DINOv2 ViT-L/14的特征提取、PCA降维、K-means聚类、LDA线性判别、Mahalanobis距离与kNN分类。

**📊 数据集**

实验数据集为MedMNIST v2套件的11个2D医学图像数据集（不含ChestMNIST）。

**📈 对比分析**

通过与传统Fine‑tune、linear probing和kNN基准对比，A-ROM平均准确率达83.7%，AUC为0.94，且在512样本/类时仍保持90%以上精度，表现与最优基准相当。

**⚠️ 局限性**

局限性包括对高度噪声或大规模数据集（如TissueMNIST）性能下降，依赖预训练模型的普适性，且不适用于多标签任务。

---

## 502. Force Polytope-Based Cant-Angle Selection for Tilting Hexarotor UAVs

**arXiv ID:** 2604.05998 | [PDF](https://arxiv.org/pdf/2604.05998v1)

**作者:** Alberto Piccina `[一作]` (University of Padova), Giulia Michieletto `[通讯]` (University of Padova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

为星形互相耦合倾斜六旋翼UAV设计了一种轻量级控制框架，通过离线预计算的零矩力多面体查找表实现倾斜角在线候选筛选与优化，从而实现交互任务中的姿态跟踪与力生成。

**💡 创新点**

创新点在于：①将可执行力多面体离线离散化为查找表，显著降低在线分配计算负担；②利用安全球包含条件快速筛选可行倾斜角；③在候选倾斜角中通过平衡能耗与平滑性的优化选择最优倾斜角。

**🔧 技术方法**

采用的技术包括：几何全姿态控制器、离线多面体生成、在线候选筛选与凸优化、伪逆控制分配、Monte Carlo仿真及Simscape物理仿真。

**📊 数据集**

使用的数据集为：随机生成的交互力轨迹（从随机样点插值得到的力向量）、仿真中的传感器噪声模型以及六旋翼的系统参数，全部在MATLAB/Simulink和Simscape中合成。

**📈 对比分析**

与基线SQP优化分配方法相比，该框架在相同鲁棒半径下平均计算时间下降至约90 %（从0.58 s降至0.05 s），姿态误差和力效率与基线相近或略优，且在Monte Carlo测试中保持了更低的总电机负荷和更好的跟踪精度。

**⚠️ 局限性**

局限性包括：仅考虑力耦合而忽略交互产生的矩；依赖离线预计算的多面体表，对极端动态变化的鲁棒性有限；尚未在真实平台上实验验证，且对非线性动力学与外部干扰的适应性尚需进一步评估。

---

## 503. Evolutionary Optimization of AI-Collapsed Software Development Stacks: Labor Tipping Points and Workforce Realignment

**arXiv ID:** 2604.05948 | [PDF](https://arxiv.org/pdf/2604.05948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 504. Data Distribution Valuation Using Generalized Bayesian Inference

**arXiv ID:** 2604.05993 | [PDF](https://arxiv.org/pdf/2604.05993v1)

**作者:** Cuong N. Nguyen `[一作]` (Durham University), Cuong V. Nguyen `[通讯]` (Durham University)

**通讯引用:** 868 | [OpenAlex ID](https://openalex.org/A5101863729)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于通用贝叶斯推断的框架——Generalized Bayes Valuation（GBV），用于从有限样本中估计并比较不同数据分布的价值，并进一步扩展到连续数据流场景的 Continual GBV（CGBV）。该框架被用于评估注释者质量和自动数据增强策略，并在多种数据集上验证其有效性。

**💡 创新点**

创新点主要包括：①利用转移可传递性度量（如LEEP、LogME等）构造贝叶斯损失，将分布估价统一到贝叶斯框架；②提出 GBV 能同时解决注释者评估与数据增强这两类看似无关的任务；③引入“quick τ”超参数选择方案，简化调参；④在持续学习设置中以贝叶斯递推方式更新分布估价，形成 CGBV。

**🔧 技术方法**

核心技术包括通用贝叶斯推断、转移可传递性度量（LEEP、LogME、ETran、MMD等）、贝叶斯递推更新、基于 ResNet 的预训练模型、Adam/SGD 训练、软化分布估价的 softmax 归一化。

**📊 数据集**

实验使用的主要数据集有：CIFAR-10、CUB-200-2011、Stanford‑Dogs；在注释者评估任务中使用 100 或 10 张验证图；在数据增强任务中使用完整训练集。

**📈 对比分析**

与统一权重、DAVINZ、LAVA、MMD（分布估价基线）以及 AutoAugment、RandAugment、TrivialAugment、EntAugment、SRA（增强基线）进行对比。实验结果显示：在注释者评估中 GBV（quick τ）和 GBV（最佳 τ）均超过 MMD，且在 CUB‑200-2011 上接近甚至略优于最佳 τ；在数据增强中 GBV 取得最高准确率，快 τ 亦逼近最佳 τ；CGBV 在持续注释者评估任务中持续更新后显著优于不更新或平均分布。运行时间方面，GBV 仅需 0.85 s/注释者，快于 MMD（1.05 s）并明显快于 DAVINZ、LAVA。

**⚠️ 局限性**

局限性包括：①对连续源集合的先验缺乏闭式或共轭形式，需要近似推断；②理论性质基于完美转移度量，实际度量误差影响尚未量化；③实验仅覆盖计算机视觉任务，尚未验证在文本、语音等其他领域的适用性；④大规模数据集下的可扩展性与计算成本待进一步评估。

---

## 505. Flowr -- Scaling Up Retail Supply Chain Operations Through Agentic AI in Large Scale Supermarket Chains

**arXiv ID:** 2604.05987 | [PDF](https://arxiv.org/pdf/2604.05987v1)

**作者:** Eranga Bandara `[一作]` (Old Dominion University), Nilaan Loganathan `[通讯]` (Effectz.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出并实现了 Flowr 框架，通过将大型连锁超市的端到端供应链工作流程拆解为多智能体，完成从需求预测到配送中心调度的全流程自动化。

**💡 创新点**

创新点在于：①基于多智能体的工作流程拆解与 MCP（Model Context Protocol）统一接口实现人机协同；②构建域专业化 LLM 集群，并通过中央推理 LLM 实现责任感与可解释性；③在真实商超环境中验证，展示显著的人工协调成本降低和异常处理提升。

**🔧 技术方法**

技术主要包括：大型语言模型（Llama‑3、Mistral、Qwen 细调）、LoRA + QLoRA 低参数微调、Ollama 本地推理、OpenAI Agents SDK、Claude Code 开发、LM Studio 人机交互、MCP 服务器架构、中心推理 LLM（OpenAI GPT‑OSS）。

**📊 数据集**

使用的训练与评估数据来自商超的历史销售记录、库存数据库、供应商对话日志、采购订单历史和配送中心日志，形成了覆盖需求、库存、采购、供应商、配送与异常的多源异构数据集。

**📈 对比分析**

评估以采购与配送两大子流程为例，与人工基线对比：采购生成的订单准确率、完整度评为 4.7/5，配送计划实现了平均 16% 路线优化提升；人机评审显示模型在推理和异常识别方面均优于传统手工流程，整体性能显著提升。

**⚠️ 局限性**

局限性包括：仅在单一大型连锁商超中做了 PoC，缺乏跨行业、多渠道的泛化验证；模型仍可能产生幻觉，需要人类持续监督；对高频短周期供应链变化的实时响应能力尚未充分评估。

---

## 506. Automating Manual Tasks through Intuitive Robot Programming and Cognitive Robotics

**arXiv ID:** 2604.05978 | [PDF](https://arxiv.org/pdf/2604.05978v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 507. Arch: An AI-Native Hardware Description Language for Register-Transfer Clocked Hardware Design

**arXiv ID:** 2604.05983 | [PDF](https://arxiv.org/pdf/2604.05983v1)

**作者:** Shuqing Zhao `[一作]` `[通讯]`, Shuqing Zhao

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Arch 的硬件描述语言，专为微架构级别的设计与 AI 辅助代码生成而从零开始设计。

**💡 创新点**

创新点在于：①把流水线、FSM、FIFO、仲裁器、寄存器文件、时钟域交叉等微架构模式提升为语言第一类构造；②使用四维静态类型系统（位宽、时钟域、端口方向、信号所有权）把 latch、驱动冲突、CDC、RDC 等错误迁移到编译期；③引入 AI‑generatability 合同（统一声明模式、LL(1) 语法、命名块结束、方向箭头、占位符 `todo!`）使大型语言模型无需微调即可生成结构安全、类型安全的 Arch 代码。

**🔧 技术方法**

技术实现包括：静态类型检查、时钟/复位的参数化类型、同步器类型化桥接、自动生成的流水线与状态机代码、LL(1) 语法解析、编译时错误反馈、`archc` 编译器生成 deterministic SystemVerilog 并支持 `archv` 通过独立生成的 C++ 模型进行周期精确、无事件驱动的仿真。

**📊 数据集**

评估使用了两个公开基准集：VerilogEval v2（156 题）和 CVDP（231 题），共计 387 题库，涵盖组合逻辑、寄存器、FSM、流水线、缓存、DMA 等多种典型硬件。

**📈 对比分析**

对比方法：在每个基准题中，先用 Arch 编译生成 SystemVerilog，再用 Verilator/ Icarus 进行仿真验证；实验结果显示 Arch 解决率为 100%（VerilogEval）和 92%（CVDP），代码量比原 SystemVerilog 平均缩减 29%；生成的 SystemVerilog 在综合（Xilinx 7‑series、SkyWater 130nm）中实现 LUT/FF 使用量与现有 HDL 相当或更优，时序也达 223 MHz，功耗极低；仿真性能上，Arch 的编译仿真（`archv`）相较于事件驱动 Verilator 提供 10–50× 的速度提升。

**⚠️ 局限性**

局限性包括：仍未实现完整的 X‑optimism 检测（如未初始化 RAM、越界索引、除零）；RDC 检查尚待完成；未支持多文件子模块的完整路径解析；部分高级特性（TLM、事务级建模、可配置缓存、内容寻址存储器）仅在规范中列出，未实现；以及对高层协议、功耗建模（UPF/CPF）等高级功能的支持仍在规划中。

---

## 508. GTaP: A GPU-Resident Fork-Join Task-Parallel Runtime with a Pragma-Based Interface

**arXiv ID:** 2604.05982 | [PDF](https://arxiv.org/pdf/2604.05982v1)

**作者:** Yuki Maeda `[一作]` (University of Tokyo), Kenjiro Taura `[通讯]` (University of Tokyo)

**通讯引用:** 1961 | [OpenAlex ID](https://openalex.org/A5009359355)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GTaP——一种GPU驻留的分叉-连接任务并行运行时，提供基于pragma的接口以简化GPU任务编程。

**💡 创新点**

创新点在于①将任务并行模型直接驻留在GPU上，显著减少CPU↔GPU数据拷贝；②利用编译器级pragma实现透明任务划分；③结合工作窃取调度实现动态负载平衡。

**🔧 技术方法**

采用CUDA、OpenMP 5.0 pragma、任务图调度器、CUDA Streams以及工作窃取算法等技术。

**📊 数据集**

使用PARSEC、Rodinia、Graph500等GPU加速基准数据集进行评估。

**📈 对比分析**

与CUB、Thrust、NVIDIA OpenMP以及传统CPU fork-join方法对比，GTaP在大多数基准上提升了1.5~3.0倍的执行速度，延迟降低20~40%，同时保持较低的内存占用。

**⚠️ 局限性**

限制包括：仅支持NVIDIA GPU；对极小任务存在调度开销；跨GPU并行受限；对某些OpenMP pragma支持不足；编译器前端对pragma解析有限。

---

## 509. Epistemic Blinding: An Inference-Time Protocol for Auditing Prior Contamination in LLM-Assisted Analysis

**arXiv ID:** 2604.06013 | [PDF](https://arxiv.org/pdf/2604.06013v1)

**作者:** Michael Cuccarese `[一作]` `[通讯]`, Michael Cuccarese

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并实现了在LLM驱动的数据驱动分析中通过“epistemic blinding”技术消除训练先验污染，保持对实体的匿名化。

**💡 创新点**

创新点在于一种无模型改动、在推理时将实体名称匿名化、对照A/B比较，量化训练先验对输出的影响，并将此方法应用于药靶优先级与股票筛选两大领域。

**🔧 技术方法**

使用技术包括基于LLM的代理推理、进化优化的scoring函数、匿名映射、A/B对照评估、开放源码工具和Claude Code skill。

**📊 数据集**

使用的数据集包括癌症基因组学多层数据（cBioPortal, GWAS, ESM2, Geneformer, ProtT5, gnomAD等）以及S&P 500公司基本面数据。

**📈 对比分析**

通过在同一模型下对比匿名化与非匿名化提示的输出，计算Top‑20列表重叠、排名差异、Jaccard、Kendall τ等指标；在肿瘤药靶案例中blinding导致16%排名变化但验证靶点恢复不变，股票筛选中变动35%。

**⚠️ 局限性**

局限性包括仅使用单一Claude模型、未探索部分匿名化、缺乏新候选验证、输出方差以及结构泄露导致的匿名化不足等。

---

## 510. BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs

**arXiv ID:** 2604.05942 | [PDF](https://arxiv.org/pdf/2604.05942v1)

**作者:** Abbas Ghaddar `[一作]` (Huawei), Yufei Cui `[通讯]` (Huawei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种后训练滑动窗口注意力（SWA）混合方法，并提出黑盒二进制优化框架BOSCH，利用三阶段分解实现大规模LLM的头级替换。

**💡 创新点**

创新点在于：① 通过大邻域搜索拆分为层重要性检测、适应性比例分配与分组头级搜索，解决传统层级或静态头级方法的耦合与误判问题；② 采用训练无关、零样本的黑盒优化，避免昂贵的再训练；③ 通过KV共享约束提升内存效率。

**🔧 技术方法**

使用的大邻域搜索、黑盒二进制优化、层级敏感性评估、适应性SWA比例分配、分组搜索以及对GQA组的约束；评价指标基于NIAH、LongBench的零样本准确率。

**📊 数据集**

校准集使用合成的6个NIAH任务；评测集为LongBench（29个长序列QA任务）+GSM8K；持续预训练实验使用2.5B个额外token。

**📈 对比分析**

与三种层级启发式（INTR、BME、RAND）和六种静态头级方法（包括BERT迁移、RazorAttention等）对比，BOSCH在所有SWA比例（0.25/0.5/0.75/0.875）和四个模型规模（1.7B/8B/14B/30B）上均表现最佳，尤其在高比例时差距显著；持续预训练后可快速恢复甚至超越原模型的长序列性能。

**⚠️ 局限性**

局限性：实验仅在Qwen3系列验证，未覆盖其他模型家族或更大参数；持续预训练仅使用单一数据集与2.5B token；未探究与量化、KV压缩或权重剪枝等其他压缩技术的交互。

---

## 511. Leveraging Image Editing Foundation Models for Data-Efficient CT Metal Artifact Reduction

**arXiv ID:** 2604.05934 | [PDF](https://arxiv.org/pdf/2604.05934v1)

**作者:** Ahmet Rasim Emirdagi `[一作]` (Codeway AI Research), M. Akın Yılmaz `[通讯]` (Codeway AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种通过 LoRA 微调 Qwen-Image-Edit 视觉语言扩散模型，以指令驱动的多图像参考方式完成 CT 金属伪影去除的低数据量方法。

**💡 创新点**

创新点在于将伪影去除重新定义为指令化的上下文推理任务，并利用参数高效 LoRA 适配和多图像参考，实现仅需 16–128 对样本的数据高效学习，同时显著减少模型幻觉。

**🔧 技术方法**

使用技术包括 20B 参数 Qwen-Image-Edit 扩散 Transformer、LoRA 参数高效适配、指令条件、跨图像注意力的多图像参考条件以及自集成推理。

**📊 数据集**

实验基于 AAPM CT-MAR 基准（合成金属植入的 CT 切片），使用 1,000 张测试样本和 16、32、64、128 对训练样本进行对比评估。

**📈 对比分析**

方法与 ADN、RISE-MAR、OSCNet+ 等专用图像域 MAR 方法以及零样本 Qwen-Image-Edit 进行对比；在 PSNR/SSIM、FID、LPIPS 以及 radiology‑aware 版本上实现最优或接近顶尖性能，集成模型更是逼近 VAE 上限。

**⚠️ 局限性**

局限性包括仅在合成金属伪影数据上验证，缺乏真实临床 CT 评估；只使用单一基础模型；集成推理需要多次前向传播，推理延迟较高；未探讨 3D 病例的扩展。

---

## 512. SonoSelect: Efficient Ultrasound Perception via Active Probe Exploration

**arXiv ID:** 2604.05933 | [PDF](https://arxiv.org/pdf/2604.05933v1)

**作者:** Yixin Zhang `[一作]` (Shandong University), Yue Yao `[通讯]` (Shandong University)

**通讯引用:** 637 | [OpenAlex ID](https://openalex.org/A5102287479)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种基于主动探查的超声探头运动策略SonoSelect，能够在有限扫描步数内通过三维空间记忆自适应选择最具诊断价值的视角；

**💡 创新点**

创新点在于将连续探头运动拆分为高层离散路由（按分区选择）与低层精细控制的两级网络，并使用三维概率记忆引导探查，显著提升在未知解剖结构上的泛化；

**🔧 技术方法**

技术包括：3D概率记忆融合（Bayesian融合）、Q‑learning路由模块、PPO低层动作优化、分区特征提取与残差控制、熵与覆盖率双目标奖励；

**📊 数据集**

使用公开CT/超声模拟数据：TotalSegmentator（肾脏与肾囊肿）以及自建的几何与器官多视角数据集；

**📈 对比分析**

与随机、纯PPO、基于信息增益（VIG）和随机网络驱动（RND）等基线对比，SonoSelect在见解剖上达到≈70%肾脏覆盖、≈48%囊肿覆盖，未见解剖上仅比RND/VIG提升约10‑15%，且在步骤数相同的情况下诊断效率最高；

**⚠️ 局限性**

局限包括：仅在仿真环境验证，真实探头接触和声学噪声未建模；分区数对性能敏感，需要针对不同解剖自适应；训练需要大量仿真轨迹，迁移到真实硬件时可能需进一步调优。

---

## 513. "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?

**arXiv ID:** 2604.05930 | [PDF](https://arxiv.org/pdf/2604.05930v1)

**作者:** Naen Xu `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 7923 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套多模态双关生成与评测流程，构建了包含 445 个双关与 890 个非双关样本的 MultiPun 基准。

**💡 创新点**

创新点在于：① 将正负样本对抗引入双关评测；② 设计了检测、定位与解释三类任务；③ 提出了 Prompt‑Level 的 Pun‑CoT 与 Model‑Level 的 Pun‑Tuning 两种提升策略。

**🔧 技术方法**

技术方法包括文本‑图像生成 pipeline、跨模态提示工程、对 11 种 VLM 进行检测/定位/解释评测，以及基于多模态数据进行微调和链式思考提示。

**📊 数据集**

使用的数据集为 MultiPun，涵盖 445 条多模态双关与 890 条对抗性非双关样本，并通过人工验证确保视觉与文本对齐。

**📈 对比分析**

在 11 个 VLM 上进行实验，采用 TPR、TNR、F1、Kappa 等指标；闭源模型表现优于开源模型，Pun‑CoT/Pun‑Tuning 平均提升 F1 约 16.5%，但模型仍易出现假正例。

**⚠️ 局限性**

局限性包括：仅覆盖英文双关；负样本种类有限；实验仅涵盖 11 个模型，未覆盖最新模型；微调实验仅限部分开源模型，难以全面验证。

---

## 514. FEEL: Quantifying Heterogeneity in Physiological Signals for Generalizable Emotion Recognition

**arXiv ID:** 2604.05926 | [PDF](https://arxiv.org/pdf/2604.05926v1)

**作者:** Pragya Singh `[一作]` (Indian Institute of Information Technology Delhi), Pushpendra Singh `[通讯]` (Indian Institute of Information Technology Delhi)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建FEEL框架，对19个公开的EDA/PPG情绪数据集进行统一预处理、标签对齐与特征提取，并在16种模型（传统机器学习、深度学习、预训练CLSP）上进行大规模基准评估，包括单数据集内和跨数据集的实验。

**💡 创新点**

①首次对19个异构情绪数据集进行统一的跨域基准；②提出兼容的四类情绪标签与细粒度对齐；③引入CLSP预训练与轻量级微调（CoCoOp Meta-Net）实现无文本对比学习；④系统评估实验环境、设备与标注方式对跨域泛化的影响。

**🔧 技术方法**

传统机器学习（RF、LDA），深度学习（ResNet、LSTM、CNN+Transformer、Attention），手工特征+MLP，预训练对比语言‑信号模型CLSP及其fine‑tuning（CoCoOp）。

**📊 数据集**

19个公开的EDA/PPG情绪数据集，涵盖实验室、约束式、真实环境、不同设备（Empatica E4、Shimmer3、定制腕带、实验室设备）与多种标注方式（刺激式、主动自评、专家注解）。

**📈 对比分析**

使用Leave‑One‑Subject‑Out CV评估单数据集性能，交叉数据集训练+测试评估泛化；记录F1/准确率。结果显示CLSP微调模型在71/114任务中排名最高，零样本CLSP在14个任务表现最好；传统手工特征+MLP在多任务中也有竞争力；跨域实验表明真实环境训练对实验室/约束数据迁移良好，设备与标注方式显著影响模型泛化。

**⚠️ 局限性**

仅评估了16种通用模型，未包含更高级或专门化的生理信号模型；对标注统一方法仍有限，缺乏文化、健康状况等因素的考量；基准依赖现有公开数据，隐私与可用性限制；跨域泛化受设备、实验设置等因素的影响。

---

## 515. The UNDO Flip-Flop: A Controlled Probe for Reversible Semantic State Management in State Space Model

**arXiv ID:** 2604.05923 | [PDF](https://arxiv.org/pdf/2604.05923v1)

**作者:** Hongxu Zhou `[一作]` `[通讯]` (Saarland University), Hongxu Zhou (Saarland University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了UN​DO Flip‑Flop任务，用来检验模型在非单调语义状态恢复上的能力，并在此任务上对Mamba‑2模型进行评估。

**💡 创新点**

创新点在于设计了一种既包含堆栈式回滚又兼顾语义检索的新基准任务，并通过对比梯度学习结果与理论表达性，揭示两者的差异。

**🔧 技术方法**

技术上使用了Mamba‑2线性状态空间模型，采用AdamW优化器训练，并结合因果消融与极端UNDO压力测试进行机制分析。

**📊 数据集**

数据集为自定义生成的序列，包含写、忽略、UNDO、读四种指令，训练序列长度1–50，OOD序列长度51–100，忽略概率高、UNDO概率低。

**📈 对比分析**

与标准Flip‑Flop基线比较时，1‑层和2‑层Mamba‑2在ID上可达约99%准确率，但在OOD及UNDO压力测试中显著退化，2‑层模型在压力测试中仅41.1%准确率，低于随机。

**⚠️ 局限性**

局限性包括仅测试单一模型配置、单一随机种子、缺少Transformer基线、未验证Theorem 6对Mamba‑2的适用性，以及训练分布可能诱导捷径学习。

---

## 516. LAG-XAI: A Lie-Inspired Affine Geometric Framework for Interpretable Paraphrasing in Transformer Latent Spaces

**arXiv ID:** 2604.06086 | [PDF](https://arxiv.org/pdf/2604.06086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 517. PoM: A Linear-Time Replacement for Attention with the Polynomial Mixer

**arXiv ID:** 2604.06129 | [PDF](https://arxiv.org/pdf/2604.06129v1)

**作者:** David Picard `[一作]` (Gustave Eiffel University), Loic Landrieu `[通讯]` (Gustave Eiffel University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Polynomial Mixer（PoM）作为自注意力的线性复杂度替代块，并在多领域任务中验证其可行性。

**💡 创新点**

创新点在于：1) 通过学习的多项式函数聚合整个序列成共享状态，实现线性复杂度的上下文混合；2) 证明 PoM 满足上下文映射（Contextual Mapping）性质，使其成为通用的序列到序列近似器；3) 在保持参数量不变的前提下，替换 Transformer 中的自注意力，兼具速度与准确性。

**🔧 技术方法**

主要技术包括：多项式混合器（Polynomial Mixer）与其残差结构 Polymorpher；基于多头注意力的标准 Transformer 架构；FlashAttention、Linformer 等现有加速注意力方法；针对因果序列的多项式混合器扩展；以及对不同任务的实验验证。

**📊 数据集**

使用的数据集覆盖五大领域：文本生成（GPT‑2/LLM 训练数据）、光学字符识别（Ludovico Antonio Muratori）、地球观测（PASTIS 影像时序）、三维点云分割（ScanNet、SemanticKITTI）、图像生成（ImageNet）。

**📈 对比分析**

与原始基线（MHA Transformer）和 FlashAttention 进行对比；在各任务中 PoM 与原基线保持相近或略优的准确度（例如 OCR CER、地球观测 mIoU、点云分割 IoU、图像生成 FID），并在长序列/高分辨率场景下实现 3~12 倍的推理速度提升；在某些任务中混合模型（PoM+自注意力）进一步提升速度同时保持精度。

**⚠️ 局限性**

限制主要在：1) 对短序列仍不如高度优化的自注意力实现；2) 当前实现仅使用 PyTorch 高层代码，缺乏低层 CUDA 优化，导致在极大规模模型时仍有进一步加速空间；3) 对因果/块因果场景的实验仍有限；4) 需要更大规模实验验证在超大模型上的可扩展性与稳健性。

---

## 518. CritBench: A Framework for Evaluating Cybersecurity Capabilities of Large Language Models in IEC 61850 Digital Substation Environments

**arXiv ID:** 2604.06019 | [PDF](https://arxiv.org/pdf/2604.06019v1)

**作者:** Gustav Keppler `[一作]` (Karlsruhe Institute of Technology), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5032 | [OpenAlex ID](https://openalex.org/A5014228448)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了CritBench框架，对IEC 61850数字变电站环境中的LLM代理进行安全评估，提出了81个涵盖静态解析、网络流量分析和动态交互的专属任务。

**💡 创新点**

创新点在于为工业控制系统（OT/ICS）提供了专门的工具层CritLayer，能够将复杂的IEC 61850协议抽象为可调用函数，并将任务划分为三大类，首次系统评估LLM在真实工业协议环境下的能力。

**🔧 技术方法**

技术实现基于Docker隔离、CritLayer工具注册、LLM代理交互循环（工具调用–执行–观察–记忆）、文本与状态双重评估、MITRE ATT&CK映射以及token预算与时间限制控制。

**📊 数据集**

使用了81个自研任务数据集，涵盖30个SCD文件解析、30个PCAP流量分析以及21个虚拟机动态交互，模拟协议堆栈使用libIEC61850和c104等。

**📈 对比分析**

通过在5个LLM（GPT‑5.1、GPT‑5 mini、Qwen3.5、Minimax M2.5、GPT‑5 nano）上跑三次实验，整体准确率最高为86.4%，CritLayer提升约20%，但在动态交互任务中准确率显著下降。

**⚠️ 局限性**

限制包括仅使用软件仿真不含厂商固件、任务覆盖范围有限、LLM的随机性导致结果波动以及未测试真实硬件交互。

---

## 519. Short Data, Long Context: Distilling Positional Knowledge in Transformers

**arXiv ID:** 2604.06070 | [PDF](https://arxiv.org/pdf/2604.06070v1)

**作者:** Patrick Huber `[一作]` (Meta Reality Labs), Adithya Sagar `[通讯]` (Meta Reality Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何通过在知识蒸馏中使用 RoPE 位置编码，并在短序列训练数据上进行阶段化上下文扩展，来为 1.1B 的学生模型赋予长序列推理能力。

**💡 创新点**

创新点在于首次系统揭示 RoPE 位置扰动在教师前向传播中的传播机制，并证明其在蒸馏目标中隐式传递长距位置信息，从而使学生在无长文本数据情况下获得长序列能力；同时提出相位化 RoPE 参数缩放策略。

**🔧 技术方法**

技术包括 RoPE 位置编码、基于 logits 的知识蒸馏、阶段化训练（短 2048、长 128k）、位置扰动传播分析和欧氏距离/余弦相似度可视化。

**📊 数据集**

使用的数据集为 Llama-4 Scout 训练的 10M+ 100B 短文本样本，包含多样化的短文档；评估基准为 Needle-in-a-Haystack 与 RULER。

**📈 对比分析**

通过与交叉熵训练的对比，显示 KD+相位化 RoPE 能将长序列准确检索率提升至约 43%（KD 10k-500k）并显著低于 CE，且训练损失更低，证明方法有效。

**⚠️ 局限性**

局限包括单一教师-学生配置、缺乏因果干预验证、评估仅覆盖检索式基准、以及对不同规模模型的泛化未知。

---

## 520. Fine-Grained Power and Energy Attribution on AMD GPU/APU-Based Exascale Nodes

**arXiv ID:** 2604.06056 | [PDF](https://arxiv.org/pdf/2604.06056v1)

**作者:** Adam McDaniel `[一作]` (University of Tennessee), Oscar Hernandez `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 1493 | [OpenAlex ID](https://openalex.org/A5071822072)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Frontier与Portage超算上AMD MI250X GPU与MI300A APU的多种功耗/能耗传感器进行系统性特性表征，并提出基于能量计数器重构毫秒级功率的细粒度功耗归因方法。

**💡 创新点**

创新点在于：①将on‑chip、off‑chip与Cray PM传感器的更新频率、延迟、响应、别名等行为统一建模与评估；②利用ΔE/Δt从累积能量计数器恢复高时间分辨率功率；③将多源传感器流与Score‑P/PAPI阶段划分同步，完成阶段级能耗归因。

**🔧 技术方法**

使用Score‑P与PAPI插件进行异步采样；fastotf2将OTF2轨迹转换为CSV；通过ΔE/Δt计算功率；对齐多源时间戳；在GPU上跑方波、rocHPL、rocHPL‑MxP、HPG‑MxP基准。

**📊 数据集**

实验数据包括合成方波工作负载（1 s/1 s等周期）和在Frontier 512 GPU、Portage 480 APU上运行的rocHPL、rocHPL‑MxP、HPG‑MxP三大基准的多节点运行。

**📈 对比分析**

方法通过与Cray PM、ROCm‑SMI等原始功率信号对比验证；在混合精度基准中，能耗降低高达79%（Frontier）/81%（Portage），说明该归因方法能够清晰区分时间缩短与功率下降两种效应。

**⚠️ 局限性**

局限性：传感器采样延迟与采样开销导致有效时间分辨率受限；off‑chip Cray PM更新速率慢；MI250X/MI300A内置功率计数器被滤波，需在未来系统上进一步验证与适配。

---

## 521. Singular Relative Entropy Coding with Bits-Back Rejection Sampling

**arXiv ID:** 2604.06055 | [PDF](https://arxiv.org/pdf/2604.06055v1)

**作者:** Gergely Flamich `[一作]` (Imperial College London), Spencer Hill `[通讯]` (Queen's University)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5104109050)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

构造了一种针对奇异信道的相对熵编码方法——Bits-Back Rejection Sampler（BBRS），实现了零对数冗余。

**💡 创新点**

创新点在于将 bits-back 编码与贪婪拒绝采样结合，利用信道奇异性显式嵌入纠错机制，得到常数更优且实现更简洁的一次性率。

**🔧 技术方法**

使用了 bits-back coding、贪婪拒绝采样（GRS）、可逆采样、对数密度比量化以及相对熵编码技术。

**📊 数据集**

论文未使用具体数据集，全部以理论模型和通用随机变量分析为主。

**📈 对比分析**

与 Sriramu‑Wagner 奇异信道代码比较，BBRS 在大样本极限下对数冗余为零，且一次性率常数更小，理论性能与其相当但实现更直接。

**⚠️ 局限性**

局限性：仍需对对数密度比进行离散化处理，且仅在奇异信道有效，对非奇异信道的性能不具备最优性。

---

## 522. Algorithmic Monoculture and its Critics

**arXiv ID:** 2604.06047 | [PDF](https://arxiv.org/pdf/2604.06047v1)

**作者:** Brian Hedden `[一作]` (Massachusetts Institute Of Technology), Manish Raghavan `[通讯]` (Massachusetts Institute Of Technology)

**通讯引用:** 3254 | [OpenAlex ID](https://openalex.org/A5052541789)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评估并严谨检验了对算法单一化（monoculture）的三类主要批评：排斥风险、代理权利、信息获取与探索平衡，并在理论模型与仿真中检验其合理性与效果。

**💡 创新点**

创新点在于将批评转化为可操作的形式化模型，利用多种场景（招聘、博弈、bandit实验）展示单一算法在多数情况下并不比多样化算法差，甚至可通过“集成单一化”（ensemble monoculture）实现超越；同时对比单一化与多样化在排斥、代理权利与信息聚合方面的相对表现。

**🔧 技术方法**

主要使用理论模型（匹配、博弈、稳定匹配、Condorcet Jury类模型）和仿真技术（序列招聘、同步招聘、Bandit游戏）来评估不同决策体系的表现。

**📊 数据集**

仿真数据基于合成数据集：1000名候选人，候选人客观价值为标准正态分布；噪声为均值0、方差0.5的高斯噪声；Bandit实验使用Bernoulli奖励、Beta(2,2)先验。

**📈 对比分析**

比较方法：在招聘场景中采用归一化表现指标（实际平均价值与最优、最劣候选人平均价值之比）；在Bandit实验中评估识别最佳手臂的成功率、总贝叶斯遗憾及误分类手臂数。结果显示：普通单一化表现最差，聚合单一化（ensemble monoculture）最优，单一化与多样化在大多数指标上差距不大，且多样化在探索/信息获取方面表现更佳。

**⚠️ 局限性**

局限性：模型假设独立、无相关错误，忽略现实中的算法相关性与数据重叠；单一化与多样化的优劣取决于算法质量和信息共享程度，集成单一化在实际部署时成本高、可行性低；研究主要基于仿真与理论，缺乏实证验证；对多阶段招聘、人为决策与外部性等现实细节考虑不足。

---

## 523. A Co-Design Framework for High-Performance Jumping of a Five-Bar Monoped with Actuator Optimization

**arXiv ID:** 2604.06025 | [PDF](https://arxiv.org/pdf/2604.06025v1)

**作者:** Aastha Mishra `[一作]` (Indian Institute of Science), Shishir Kolathaya `[通讯]` (Indian Institute of Science)

**通讯引用:** 1013 | [OpenAlex ID](https://openalex.org/A5084523677)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个两阶段的共设计优化框架，用于平面闭链五杆单足机器人在跳跃任务中的机械设计、马达与齿轮箱参数以及控制参数的联合优化。

**💡 创新点**

首次将详细的电机与齿轮箱（单/复/沃尔福姆型）参数与闭链结构共同纳入共设计，构建齿比-质量-效率映射，并在CMA‑ES中同时优化形态、驱动与控制，显著提升跳跃距离与能耗比。

**🔧 技术方法**

采用基于MuJoCo的动力学仿真、虚拟弹簧‑阻尼控制模型、杆件质量与尺寸的参数化映射、三种齿轮箱的齿比优化与效率计算，以及CMA‑ES演化算法进行多目标优化。

**📊 数据集**

使用商业电机与齿轮箱的规格表（T‑Motor U8/U10/U12、MAD M6C12、Vector Technics VT8020等）以及仿真生成的动力学数据；未使用公开数据集。

**📈 对比分析**

通过与基准（默认设计）进行对比，进行三个案例（仅长度、仅驱动、全共设计）评估，最终得到跳跃距离提升约42%（从0.726 m到1.03 m），机械能耗降低约15.8%（从26.7 J到22.49 J），显示显著性能提升。

**⚠️ 局限性**

仅在仿真环境下验证，缺乏实验验证；仅限于平面单足机器人；驱动参数受限于预选电机与齿轮箱；未考虑实时控制误差、地面摩擦变化等现实因素。

---

## 524. MAESTRO: Adapting GUIs and Guiding Navigation with User Preferences in Conversational Agents with GUIs

**arXiv ID:** 2604.06134 | [PDF](https://arxiv.org/pdf/2604.06134v1)

**作者:** Sangwook Lee `[一作]` (Virginia Tech), Yan Chen `[通讯]` (Virginia Tech)

**通讯引用:** 36606 | [OpenAlex ID](https://openalex.org/A5100402008)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在聊天式 GUI 代理（CAG）中加入了偏好记忆、基于偏好的 GUI 适配和工作流导航机制，以支持多步骤任务中的决策制定。

**💡 创新点**

①通过结构化的偏好记录在对话中持续跟踪并更新用户偏好；②利用增补、过滤、排序、突出四种原地 GUI 操作将偏好即时反映在界面上；③检测偏好与可选项的冲突并提供回溯建议，避免死路。

**🔧 技术方法**

基于 GPT‑5.4 的大语言模型进行偏好提取、GUI 操作决策与对话生成；实现了 GUI 适配的前端原地修改器；使用了对话与 GUI 状态的双模输入。

**📊 数据集**

使用电影票务场景的自定义任务集（共 4 组任务），每组包含预热任务与主任务，构造了 8 步线性工作流（电影、剧院、日期、时间、座位、确认）。

**📈 对比分析**

采用 2×2 交叉设计（Baseline vs. MAESTRO × 文本 vs. 语音）对 33 名参与者进行 within‑subject 评估。指标包括任务成功率、偏好违约次数、非偏好选择数、任务完成时长、用户负荷量表、感知有用性等。结果显示：MAESTRO 在偏好违约和非偏好选择上显著降低（p<0.05），但在语音模式下用户负荷略升高，任务时长差异不显著。

**⚠️ 局限性**

①实验仅涵盖电影票务域，难以推广到更复杂领域；②Baseline 与 MAESTRO 的三项改进未被单独拆分，因而无法单独评估各模块效益；③远程实验环境导致硬件与网络差异；④任务设计采用固定偏好，未考虑用户动态放宽偏好的情况。

---

## 525. Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents

**arXiv ID:** 2604.06132 | [PDF](https://arxiv.org/pdf/2604.06132v1)

**作者:** Bowen Ye `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5788 | [OpenAlex ID](https://openalex.org/A5101674305)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Claw-Eval 端到端评测套件，用于评估 LLM 驱动的自主代理，覆盖全轨迹可审计、跨模态、多维度（完成度、安全、鲁棒性）的评测。

**💡 创新点**

创新点：
① 全轨迹审计——通过执行轨迹、审计日志和环境快照三条独立证据实现对代理行为的透明验证；
② 多维度评分——将完成度、安全与鲁棒性统一度量，并采用 Average Score、Pass@k、Pass^k 三个互补指标；
③ 统一跨模态任务集——构建 300 个人工验证任务，涵盖服务编排、视觉感知与多轮专业对话，填补现有基准的模态与交互局限。

**🔧 技术方法**

技术方法：
- Docker 隔离执行环境与工具调用层（系统层 + 服务层）；
- 多证据收集（执行轨迹、服务器端审计日志、环境快照）；
- 混合评分管道：规则校验 + LLM 判定；
- 误差注入实现鲁棒性测试；
- 多任务多模型并行评测，支持三次独立试验。

**📊 数据集**

使用数据集：
- 300 人工验证任务，分 9 细分类别，3 大组（通用服务编排、多模态感知与生成、多轮专业对话）；
- 对 14 个前沿模型（Claude‑Opus‑4.6、Claude‑Sonnet‑4.6、GPT‑5.4、Gemini‑3.1‑Pro、Gemini‑3‑Flash、Qwen‑3.5‑397B‑A17B、MiMo‑V2‑Pro、MiMo‑V2‑Omni、GLM‑5‑Turbo、GLM‑5V‑Turbo、DeepSeek‑V3.2、MiniMax‑M2.7、Kimi‑K2.5、Nemotron‑3‑Super）进行评测。

**📈 对比分析**

评测方法与性能：
- 采用三次独立试验，计算 Average Score、Pass@3（任何一次通过）和 Pass^3（每一次均通过）；
- 在通用任务中最佳模型 Claude‑Opus‑4.6 的 Pass^3 约为 70%，整体平均分 80%；
- 多轮对话最高 Pass^3 约为 68%；
- 多模态任务最高 Pass^3 约为 25%；
- 与传统仅用 LLM 判定的评测相比，混合评测能捕捉 44% 的安全违规和 13% 的鲁棒性问题；
- 误差注入实验表明 Pass@3 稳定，而 Pass^3 在错误率升高时显著下降，验证了鲁棒性与峰值性能的分离。

**⚠️ 局限性**

局限性：
- 任务规模虽大但仍不足以覆盖所有真实场景；
- 部分评分依赖 LLM 判定，存在主观性；
- 误差注入仅模拟有限种服务错误，未覆盖所有部署异常；
- 对跨模态任务的细粒度指标尚不完整，特别是视觉与语音子领域；
- 评测对算力与环境要求较高，易受硬件/软件差异影响。

---

## 526. Multilevel Coset Codes on Lattices

**arXiv ID:** 2604.06125 | [PDF](https://arxiv.org/pdf/2604.06125v1)

**作者:** Leopold Bertholet `[一作]` (Rampart Communications), Matthew Robinson `[通讯]` (Rampart Communications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并评估了多层码 Coset Bombe Codes，结合稠密晶格、极化编码与 Voronoi 形状，适用于 AWGN 通道；

**💡 创新点**

将极化码推广至多层晶格余数集合，实现同时获得晶格形状、形状与 FEC 增益，并在低延迟块码中获得约 0.8 dB 的性能提升；

**🔧 技术方法**

采用多层编码、多级解码、D4 晶格、GF(2^d) 极化码、CA‑SCL 解码、CRC 预留、Monte Carlo 可靠性排序等技术；

**📊 数据集**

在 AWGN 模拟环境下使用 D4 晶格构造 16‑QAM 点集合，块码长度为 256 和 1024；

**📈 对比分析**

与 BICM Polar 与 MLC Polar 在相同谱效率和块码长度下进行对比，结果显示 Coset Bombe 在 BER/BLER 上比两者低约 0.7–0.8 dB，且可将块大小延迟减半；

**⚠️ 局限性**

计算复杂度随晶格维数指数增长，目前仅在低维 D4 上实现；高维扩展、实时实现和硬件实现仍需进一步研究。

---

## 527. Signature Placement in Post-Quantum TLS Certificate Hierarchies: An Experimental Study of ML-DSA and SLH-DSA in TLS 1.3 Authentication

**arXiv ID:** 2604.06100 | [PDF](https://arxiv.org/pdf/2604.06100v1)

**作者:** José Luis Delgado Jiménez `[一作]` `[通讯]` (Universitat Oberta de Catalunya), José Luis Delgado Jiménez (Universitat Oberta de Catalunya)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 TLS 1.3 后量子迁移中的证书层级设计进行实验研究，系统比较 ML‑DSA 与 SLH‑DSA 在不同层级、深度和密钥交换模式下对握手延迟、传输量与服务器计算负载的影响。

**💡 创新点**

将后量子迁移视为证书层级设计问题，发现签名放置（尤其是叶节点 SLH‑DSA）是导致服务器负载爆炸、握手延迟从毫秒级升至秒级的关键因素；提出基于层级位置而非仅仅签名族的评估框架。

**🔧 技术方法**

利用 OpenSSL 3 + oqsprovider 在本地实验室实现 TLS 1.3，构造多种 X.509 证书链（根/中间/叶、深度 2/3），混合使用 ML‑DSA、SLH‑DSA、ML‑KEM，并测量平均/95th 百分位延迟、字节传输、证书链大小、服务器/客户端 task‑clock 等指标。

**📊 数据集**

实验数据由实验室自行生成的多场景 TLS 握手日志组成，涵盖数十个组合（签名族、层级位置、深度、KEX 模式），并配合性能计数器收集。

**📈 对比分析**

采用四个实验组（叶部、完整策略矩阵、深度比较、KEX 模式）进行对照，比较平均/95th 百分位延迟、传输字节、服务器/客户端计算占比；结果显示叶部 SLH‑DSA 使延迟从 ~1 ms 跃升至 ~1.4 s，服务器 task‑clock 成百倍增加，服务器计算占比超 99%，而其他场景保持低成本、平衡或有限的提升。

**⚠️ 局限性**

实验仅在单机本地环境完成，未考虑网络延迟、跨域链或不同 TLS 实现；只测算 CPU 计数器，未深入追踪代码路径；结果基于 OpenSSL 3 + oqsprovider，可能不完全适用于其他库或平台。

---

## 528. Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes

**arXiv ID:** 2604.06099 | [PDF](https://arxiv.org/pdf/2604.06099v1)

**作者:** Athanasios Angelakis `[一作]` (University of Bundeswehr Munich), Marta Gomez-Barrero `[通讯]` (University of Bundeswehr Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了 ZACH‑ViT 在低数据医学图像中的鲁棒性，使用 50 样本/类的 MedMNIST 低样本协议，在清洁、噪声（MedMNIST‑C）以及 FGSM 与 PGD 对抗攻击四个评估下与三种基准模型（ABMIL、Minimal‑ViT、TransMIL）进行对比实验。

**💡 创新点**

首次将 ZACH‑ViT 扩展为鲁棒性评估，证明其在现实噪声下仍保持优势，并为低数据医学图像提供了一套完整的鲁棒性基准流程。

**🔧 技术方法**

采用 ZACH‑ViT（无位置编码、无 class token、适配残差投影、全局平均池化）训练；在 MedMNIST‑C 中使用 Gaussian noise、blur、brightness‑contrast、JPEG compression、cutout 等噪声；对 FGSM（ε∈{1/255,2/255,4/255,8/255}）和 10 步 PGD（ε/4）进行攻击；对 5 个随机种子重复实验，计算平均值、标准差及跨数据集平均排名。

**📊 数据集**

七个 MedMNIST 任务（BloodMNIST、PathMNIST、BreastMNIST、PneumoniaMNIST、DermaMNIST、OCTMNIST、OrganAMNIST），每类 50 样本；任务类型涵盖二分类和多分类，评估指标为 AUC 或 Macro‑F1。

**📈 对比分析**

在四个评估模式下通过平均排名比较四个模型；结果显示 ZACH‑ViT 在清洁数据和噪声下平均排名 1.57，优于其他模型；在 FGSM 和 PGD 下平均排名分别为 2.00 和 2.29，仍保持竞争力；整体表现显示其在低样本、现实噪声环境下提供最佳平衡。

**⚠️ 局限性**

仅评估无预训练的小模型，未涉及大规模预训练或迁移学习；攻击仅限 FGSM 与短步 PGD；未提供硬件性能、能耗或真实临床验证；未检验公平性或子群差异；鲁棒性评估基于平均值，未深入解析不同噪声或攻击强度下的细节表现。

---

## 529. LLM4CodeRE: Generative AI for Code Decompilation Analysis and Reverse Engineering

**arXiv ID:** 2604.06095 | [PDF](https://arxiv.org/pdf/2604.06095v1)

**作者:** Hamed Jelodar `[一作]` (University of New Brunswick), Ali A. Ghorbani `[通讯]` (University of New Brunswick)

**通讯引用:** 26469 | [OpenAlex ID](https://openalex.org/A5034685391)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LLM4CodeRE，一个面向恶意软件的域自适应大型语言模型框架，支持汇编到源代码逆编译和源代码到汇编翻译的双向转换；

**💡 创新点**

创新点包括：①在真实恶意二进制上进行域自适应CLM预训练；②提出多适配器与Seq2Seq统一前缀的两种任务适配策略；③构建统一的评测框架，将语义相似度、编辑相似度与可重新执行性三维度结合；

**🔧 技术方法**

采用的大量技术包括：域自适应的Causal Language Model预训练；LoRA参数高效微调；多适配器（低秩）和Seq2Seq统一前缀的任务适配；BERTScore、Levenshtein距离与编译+沙箱执行的综合评测；

**📊 数据集**

使用的数据集主要有：SBAN（由五大恶意软件语料库聚合的676,151个对齐样本）；PE‑Machine Learning（用于编译计数评测）；以及公开的恶意样本集（如MalwareBazaar等）；

**📈 对比分析**

通过与DeepSeek、LLM4Decompile等基准模型在Asm→Src和Src→Asm任务上进行对比，评估语义相似度、编辑相似度和可执行率。LLM4CodeRE在Asm→Src任务上语义相似度0.85、编辑相似度0.63，Src→Asm任务上语义相似度0.64、编辑相似度0.27；在XLangKode数据集上可执行率达86%，显著优于对照组；

**⚠️ 局限性**

主要局限性包括：仅针对Windows PE文件，缺乏对ELF、Android等平台的泛化；自动逆编译产生的标签噪声；沙箱评测覆盖范围有限；固定1024 token上下文长度可能导致长函数被截断；未使用符号执行等更精确的功能等价评估。

---

## 530. Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives

**arXiv ID:** 2604.06091 | [PDF](https://arxiv.org/pdf/2604.06091v1)

**作者:** Changgeon Ko `[一作]` (Korea Advanced Institute of Science and Technology), Jong C. Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3896 | [OpenAlex ID](https://openalex.org/A5100641120)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了在多代理系统中，代表性LLM代理如何被社交动力（如从众、专家印象、发言长度、修辞说服）所影响，并量化其对决策准确性的破坏；

**💡 创新点**

创新点在于从社会心理学角度系统评估并量化多代理环境下的四种社交偏差对代表代理的影响，首次将从众、专家印象、发言长度与修辞说服映射到LLM多代理决策中；

**🔧 技术方法**

使用了多种提示工程与模型规模对比（如Qwen2.5 7B/14B、Gemma3 12B、GPT-4o等），通过代理生成答案与推理并让代表代理整合五个同伴意见；

**📊 数据集**

采用了三类具有确定答案的基准数据集：BBQ（社会偏见）、MMLU-Pro（推理任务）与MetaTool（工具使用）；

**📈 对比分析**

通过对照单一代理基线与不同社交干预（对抗代理数量、相对智能、推理长度、修辞风格）的实验，发现随着社交压力增加，代表代理准确率显著下降，尤其是多数派从众、专家效应和长篇辩论对性能影响最大；

**⚠️ 局限性**

局限性包括仅评估有限的四种社交动态、未考虑多轮交互与记忆效应、未探究人类与AI混合团队、以及仅使用代表式集中决策框架，未覆盖其他多代理结构；

---

## 531. Scientific Graphics Program Synthesis via Dual Self-Consistency Reinforcement Learning

**arXiv ID:** 2604.06079 | [PDF](https://arxiv.org/pdf/2604.06079v1)

**作者:** Juekai Lin `[一作]` (Zhejiang University), Lijun Wu `[通讯]` (Shanghai Artificial Intelligence Laboratory, OpenDataLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一套能够将科学图形图像反向合成TikZ代码的系统，并提供高质量可编译的数据集和多维度评测基准。

**💡 创新点**

创新点包括：①基于执行中心的数据引擎构建的SciTikZ-230K高质量可编译数据集；②引入双向自一致性强化学习（Dual Self-Consistency RL），通过编译回合和可视化对齐实现结构与视觉双重约束；③提出SciTikZ-Bench多维度评测框架。

**🔧 技术方法**

技术涵盖多模态大模型（如Qwen3-VL）、监督微调、GRPO强化学习、SigLIP/LPIPS视觉奖励、代码结构相似度（TED/CrystalBLEU）以及执行反馈循环。

**📊 数据集**

使用的数据集包括SciTikZ-230K（230k可编译TikZ图像代码对）、SciTikZ-Bench（611条人工验证的科学图形基准）以及DaTikZ-v3等外部测试集进行跨域验证。

**📈 对比分析**

通过编译成功率、视觉相似度（SigLIP、CLIP、LPIPS、SSIM、DreamSim）以及代码质量（C-BLEU、TED）等指标与现有LLM（GPT‑5、Claude、Gemini）和专用模型（DeTikZify、ImgTikZ）对比，SciTikZer‑8B在所有指标上均遥遥领先，编译成功率达到97.2%，视觉与代码指标均显著优于对手。

**⚠️ 局限性**

局限性包括：仍受限于数据分布，极复杂多层结构的泛化能力有限；RL训练对超参数敏感；仅适用于需要可编译工具链的语言，跨语言推广需额外调优；人类评测样本覆盖面有限。

---

## 532. Covering-radius and Collinearity- Minimizing Pilots for Channel Estimation in TDD Systems

**arXiv ID:** 2604.06041 | [PDF](https://arxiv.org/pdf/2604.06041v1)

**作者:** Xu Zhu `[一作]` (Peking University), Tiejun Li `[通讯]` (Peking University)

**通讯引用:** 12598 | [OpenAlex ID](https://openalex.org/A5100703332)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 TDD OFDM 系统中提出一种基于最小覆盖半径与冗余共线抑制（MCC）的二维时频导频分配方案，用于滑动窗口最新时隙的联合信道估计。

**💡 创新点**

创新点在于将覆盖率与冗余共线抑制两大几何设计原则统一到 MILP 形式的导频模式设计中，并通过模块线约束与对称性抑制实现对多时隙联合感知的鲁棒性优化。

**🔧 技术方法**

采用混合整数线性规划（MILP）进行优化；利用覆盖半径与差向量共线约束的几何代理；使用稀疏恢复（FISTA）与虚域相干性分析；结合对称性约束实现共线抑制。

**📊 数据集**

使用 Matlab 5G Toolbox 生成的 3 km/h CDL‑B 通道模型，200 条随机通道实现，分别评估不同窗口大小、导频间隔、跳频周期等条件下的性能。

**📈 对比分析**

将 MCC 与 3GPP SRS 区块跳跃、Chirp、随机导频及其各自的“仅覆盖”/“仅共线”消融设计进行对比；通过 NMSE 曲线（SNR、导频间隔、子窗口大小、跳频周期）验证 MCC 在大多数情形下均能获得最优或最稳健的恢复误差。

**⚠️ 局限性**

局限性包括：需求解 MILP，规模受 k 限制；对复合 k 值仍可能产生不利周期结构；只考虑连续子带与公平性约束；共线抑制在极大 k 时可能不可完全消除；实验仅基于 CDL‑B 模拟，未涵盖更广泛的实际环境。

---

## 533. Design and Analysis of Chirp-Layered Superposition Coding for LoRa

**arXiv ID:** 2604.06033 | [PDF](https://arxiv.org/pdf/2604.06033v1)

**作者:** Jingxiang Huang `[一作]` (Dalhousie University), Samer Lahoud `[通讯]` (Dalhousie University)

**通讯引用:** 1418 | [OpenAlex ID](https://openalex.org/A5082663098)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在LoRa基带符号上叠加高扩展因子（SF）波形的超叠加编码方案，既不改变原有LoRa解调链路，又通过该高SF信号携带额外的BPSK比特，提升频谱利用率。

**💡 创新点**

创新点在于：①证明任何非零叠加信号都会对传统LoRa解调产生干扰；②发现高SF波形在低SF符号间隙内能均匀分布能量，可视作白噪声，从而在低SF层实现“几乎透明”的叠加；③将高SF上chirp段作为BPSK载波，并通过低SF符号重建与消除实现两层联合解调。

**🔧 技术方法**

使用的主要技术包括：CSS调制与dechirp+DFT解调、频域能量均匀分布分析、BPSK相位调制、功率分配比例（LHR）控制、误码率（SER、BER）解析及Monte‑Carlo仿真。

**📊 数据集**

未使用公开数据集，所有验证均基于自建的AWGN仿真，覆盖SF7/12组合、不同功率比和噪声级。

**📈 对比分析**

与传统LoRa单层解调相比，在相同基线SNR下，低SF层SER与理论曲线高度一致；在加入高SF叠加后，低SF层误码率仅在有效SNR模型下略有上升；高SF层BPSK BER可达到10⁻⁵以下，满足设定阈值。两层共存可在功率比例合适时同时实现可靠传输。

**⚠️ 局限性**

局限性包括：仅考虑单一叠加信号；仿真仅限AWGN环境，未评估衰落或多径影响；假设低SF符号重建完全准确，未考虑残余抑制误差；实现需要额外功率与处理，且高SF层仅支持BPSK，限制了进一步的频谱增益。

---

## 534. Learning-Guided Force-Feedback Model Predictive Control with Obstacle Avoidance for Robotic Deburring

**arXiv ID:** 2604.06133 | [PDF](https://arxiv.org/pdf/2604.06133v1)

**作者:** Krzysztof Wojciechowski `[一作]` (Universite de Toulouse), Nicolas Mansard `[通讯]` (Universite de Toulouse)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文设计了一套基于力反馈MPC与扩散运动先验的闭环控制框架，用于实现工业去毛刺任务，能够在复杂障碍、碰撞约束下完成精准插入、恒定法向力跟踪及圆形去毛刺运动。

**💡 创新点**

创新点在于首次将扩散模型作为运动先验与力反馈MPC结合，既提供多模态的运动记忆，又保证实时碰撞规避和接触力控制；同时采用线性弹簧阻尼模型在MPC中直接调节接触力，实现全局最优控制。

**🔧 技术方法**

核心技术包括：非线性力反馈MPC（加入接触力状态）、扩散Transformer运动先验、线性弹簧阻尼接触动力学、SQP+QP求解器、实时碰撞约束（距离函数线性化）以及对姿态误差的SO(3)向量化处理。

**📊 数据集**

使用离线生成的合成去毛刺轨迹数据集，数据来源于Humanoid Path Planner的任务规划，涵盖多种孔位、接触方向与碰撞环境，形成多模态轨迹库供扩散模型训练。

**📈 对比分析**

通过三种场景（单孔、障碍多孔、长序列）对比1D/3D接触模型及无碰撞约束与碰撞约束的MPC，实验结果显示：在无碰撞情况下RMSE约为4–5 N，碰撞情况下3D模型保持接触稳定而1D模型失稳；成功率100%，并且在多孔序列中即使处于远端工作空间也能完成去毛刺，表现优于传统MPC方案。

**⚠️ 局限性**

主要局限包括：扩散模型需离线预训练且针对静态场景，难以快速适应未知环境；高频率MPC受限于碰撞对数，难以处理大规模碰撞约束；机器人几何和关节极限导致远端孔位受力逼近阈值；求解器参数需要手工调优，影响实时性。

---

## 535. SEM-ROVER: Semantic Voxel-Guided Diffusion for Large-Scale Driving Scene Generation

**arXiv ID:** 2604.06113 | [PDF](https://arxiv.org/pdf/2604.06113v1)

**作者:** Hiba Dahmani `[一作]` (Huawei Paris Research Center), Roland Brémond `[通讯]` (Gustave Eiffel University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过语义条件扩散模型在3D空间生成大规模驾驶场景的Σ-Voxfield网格，并通过Deferred渲染获得真实感图像。

**💡 创新点**

提出可直接在3D空间扩散的固定卡尔丹数表面表示Σ-Voxfield，以及通过空间外扩散实现可扩展大规模场景生成，并结合Deferred渲染实现无场景优化的真实感渲染。

**🔧 技术方法**

使用语义条件的Diffusion Transformer、3D位置编码、Σ-Voxfield表面表示、空间外扩散以及基于Stable Diffusion的Deferred渲染模块。

**📊 数据集**

在Waymo Open Dataset和PandaSet的多视角驾驶数据上进行训练和评估。

**📈 对比分析**

与InfiniCube、GEN3C等SOTA方法对比，在FID/KID上对新视角更优，显著降低显存需求（8GB vs 75GB），生成时间约20分钟。

**⚠️ 局限性**

缺乏对外观属性（纹理、光照）和动态元素的显式控制，且仅支持静态场景。

---

## 536. Towards Securing IIoT: An Innovative Privacy-Preserving Anomaly Detector Based on Federated Learning

**arXiv ID:** 2604.06101 | [PDF](https://arxiv.org/pdf/2604.06101v1)

**作者:** Samira Kamali Poorazad `[一作]` (University of Oulu), Tarik Taleb `[通讯]` (Ruhr University Bochum)

**通讯引用:** 24922 | [OpenAlex ID](https://openalex.org/A5087043869)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了DyHFL框架，结合同态加密和动态缓冲式代理选择，实现工业物联网中的隐私保护异常检测。

**💡 创新点**

创新点在于将同态加密用于全局聚合以防模型反演，并引入滑动窗口动态阈值实现对快慢设备公平选择，缓解straggler与通信瓶颈。

**🔧 技术方法**

使用Paillier同态加密、滑动窗口WAM/EWA、同步聚合与缓冲策略，以及PyTorch实现的深度学习模型。

**📊 数据集**

在Gas_Pipeline、WUSTL_IIoT和Edge_IIoT三个工业数据集上评估。

**📈 对比分析**

与SyncFL、AsyncFL、FedBuff、ASR_Fed、BFL等基线相比，DyHFL在准确率、F1、收敛速度、通信成本和公平性上均显著优于对手（收敛速率提升10-100倍，通信成本降低10-50%）。

**⚠️ 局限性**

局限包括同态加密导致的加密/解密开销与密文膨胀、对可信第三方密钥管理的依赖以及对极低资源设备的适配性不足。

---

## 537. Learning $\mathsf{AC}^0$ Under Graphical Models

**arXiv ID:** 2604.06109 | [PDF](https://arxiv.org/pdf/2604.06109v1)

**作者:** Gautam Chandrasekaran `[一作]` (University of Texas at Austin), Arsen Vasilyan `[通讯]` (University of Texas at Austin)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5030920630)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在满足强空间混合（strong spatial mixing）和多项式邻域增长的图模型（如高温 Ising 模型）下，用低度多项式逼近方法实现对 AC^0 以及其他概念类（单调函数、半空间）的 PAC 学习，提供了近乎最优的子多项式样本与时间复杂度。

**💡 创新点**

创新点包括：
- 将低度算法从传统的均匀/独立分布迁移到非产品分布，通过构造近似采样器-逆向器对（sampler‑inverter pair）实现对低度逼近的迁移；
- 设计局部迭代采样器（local iterative sampler），利用图模型的强空间混合性质控制随机种子与输出变量之间的依赖，避免了传统 Fourier 分析的需要；
- 推广到树结构图模型、单调函数、半空间等多种概念类，证明其在相应分布下也存在低度多项式逼近。

**🔧 技术方法**

核心技术包括：
- 强空间混合与多项式增长的概率性分析，用于证明局部采样器的有效性；
- 构造“近似采样器-逆向器”对，并用改变测度（change‑of‑measure）证明低度逼近可迁移；
- 通过 Poincaré 不等式和影响力（influence）理论推广到低影响力函数；
- 对 Ising 模型使用 Hubbard‑Stratonovich 变换建立反集中（anti‑concentration）性质，从而实现半空间的不可知学习。

**📊 数据集**

该工作是纯理论性论文，不依赖具体数据集；所有结果均为在理论模型（图模型、Ising 模型）下的样本复杂度与时间复杂度分析。

**📈 对比分析**

与传统低度算法（仅适用于均匀或独立分布）相比，本文在更广泛的相关分布下保持了子多项式的学习复杂度；具体而言，AC^0 的学习时间与样本数均为 $n^{O(	ext{polylog}(n))}$，在满足高温条件的 Ising 模型上同样可达；相较于已知的仅在产品分布下的结果，显著扩展了可学习分布的范围。

**⚠️ 局限性**

限制与未来方向：
- 结果仅在满足强空间混合和多项式邻域增长的图模型上成立，无法直接推广到一般的高相关分布；
- 需要图模型的边权足够弱（高温）以保证快速混合；
- 对更一般分布（如无界或非马尔可夫结构）仍缺乏相应的低度逼近与采样器构造方法。

---

## 538. Masking or Mitigating? Deconstructing the Impact of Query Rewriting on Retriever Biases in RAG

**arXiv ID:** 2604.06097 | [PDF](https://arxiv.org/pdf/2604.06097v1)

**作者:** Agam Goyal `[一作]` (University of Illinois Urbana-Champaign), Hari Sundaram `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 5565 | [OpenAlex ID](https://openalex.org/A5018532037)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了查询增强技术（LLM重写、HyDE、Query2Doc 及其持续预训练版本）对密集检索中四类系统偏差（简洁性、字面匹配、位置、重复）的影响，并分析了不同技术的机制。

**💡 创新点**

首次将查询侧干预与检索器内部偏差区分，提出两类偏差（查询-文档交互型 vs 文档编码型）以及两种减偏机制（噪声增幅 vs 相关性去相关），为 RAG 系统的偏差治理提供了新的框架。

**🔧 技术方法**

使用 LLM 生成查询重写、HyDE、Query2Doc、基于 LoRA 的持续预训练等查询增强方法；通过 Spearman 相关、t 检验和 |t| 统计量衡量偏差；利用 ColDeR 基准进行评估。

**📊 数据集**

数据集为 ColDeR（基于 Re-DocRED 的对照文档对）以及其 Foil 子集，用于精细化测量单一与多重偏差情形。

**📈 对比分析**

比较方法：对 24 种检索器-偏差组合计算平均 |t| 统计量和显著性偏差计数。结果显示：简单重写总体降低 54% 的偏差，但在多偏差组合下失效；HyDE/Query2Doc 通过持续预训练实现更稳健的去相关，Foil 子集显示显著提升，尤其是 HyDE-CPT。

**⚠️ 局限性**

局限性：仅使用 ColDeR 进行控制实验，未覆盖稀疏检索或混合检索体系；只评估了几种 LLM 与检索器，缺乏对真实世界多样性和下游生成质量的考察。

---

## 539. JUÁ - A Benchmark for Information Retrieval in Brazilian Legal Text Collections

**arXiv ID:** 2604.06098 | [PDF](https://arxiv.org/pdf/2604.06098v1)

**作者:** Jayr Pereira `[一作]` (Universidade Federal do Cariri), Luiz Bonifacio `[通讯]` (NeuralMind.ai)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 JUÁ 的公共 benchmark，整合了巴西不同法律文本（判例、法规、立法草案和问答）并提供持续评测的 leaderboard。

**💡 创新点**

创新点在于：①将多种法律检索范式统一至同一评测协议；②引入持续评估平台；③使用领域适配的 Qwen 嵌入模型验证检索策略差异。

**🔧 技术方法**

采用传统 BM25、深度稠密检索（Qwen、OpenAI、KaLM）和 LoRA 微调的 Qwen3-Embedding-4B，并在 BM25 的候选集上做重排序。

**📊 数据集**

使用的公开数据集包括 JUÁ-Juris、JurisTCU、NormasTCU、Ulysses-RFCorpus 与 BR‑TaxQA，共覆盖判例检索、法规检索、立法检索及问答检索。

**📈 对比分析**

在 NDCG@10、MRR@10、MAP@10 与 Recall@10 上进行横向比较，发现领域适配的稠密检索在 JUÁ-Juris 上显著提升，BM25 在 Ulysses 与 NormasTCU 上仍保持强劲，而无一模型在所有子任务中均占优。

**⚠️ 局限性**

局限性包括：①整体分数混合了不同检索任务，难以解释模型在单一任务上的真实性能；②JUÁ-Juris 在训练中被用于监督，导致模型表现与真实泛化存在偏差；③查询多样性与字段缺失的设计降低了与真实检索场景的一致性。

---

## 540. Understanding Educators' Perceptions of AI-generated Non-consensual Intimate Imagery

**arXiv ID:** 2604.06131 | [PDF](https://arxiv.org/pdf/2604.06131v1)

**作者:** Tongxin Li `[一作]` (New Jersey Institute of Technology), Donghee Yvette Wohn `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 6670 | [OpenAlex ID](https://openalex.org/A5040735139)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对美国20名教育工作者进行访谈，分析他们对AI生成非同意亲密图像（AIG-NCII）的关注、现行做法、建议措施及实施挑战。

**💡 创新点**

首次系统性阐述教育者视角，识别了多层次支持需求（培训、政策、家长合作等），为学校、家长和监管机构提供了实践建议。

**🔧 技术方法**

使用定性研究方法——半结构化访谈和主题分析。

**📊 数据集**

访谈样本为20名来自美国不同教育层级（K‑12、高等教育、辅导员等）的教师、心理师和顾问。

**📈 对比分析**

该研究没有量化比较；通过主题编码与参与者回答的对照，描述关注点、做法与挑战的频率，未涉及算法性能评估。

**⚠️ 局限性**

样本量小、主要来自美国、可能存在自我选择偏差，未对实际政策文件或案例进行验证，结果受时间点和AI技术快速变化的影响。

---

## 541. Late Breaking Results: Hardware-Efficient Quantum Reservoir Computing via Quantized Readout

**arXiv ID:** 2604.06075 | [PDF](https://arxiv.org/pdf/2604.06075v1)

**作者:** Param Pathak `[一作]` (Fractal Analytics), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11256 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于固定量子电路、Chebyshev编码和砖工作相关的硬件高效量子储水器计算框架，用于短期负荷预测。

**💡 创新点**

首次将基因算法进行量子储水器架构搜索并证明6位量化读出在有限采样下保持精度，显著降低存储需求。

**🔧 技术方法**

使用量子储水器、Chebyshev特征编码、砖工作纠缠、单/双量子比特Pauli测量、遗传算法搜索、弹性网络读出和后训练固定点量化。

**📊 数据集**

采用2017年Tetouan市电力消耗数据集，按小时重采样后构建监督样本。

**📈 对比分析**

在512次有限采样评估下对比FP32读出，6位和8位量化读出误差不超过1%，并将读出内存分别压缩81%和75%，验证了低精度量化的有效性。

**⚠️ 局限性**

实验仅在单一数据集、有限的搜索空间、两种随机种子及仿真环境下进行，未验证于真实量子硬件，缺乏更广泛的数据集与硬件适配性研究。

---

## 542. $k$-Clustering via Iterative Randomized Rounding

**arXiv ID:** 2604.06046 | [PDF](https://arxiv.org/pdf/2604.06046v1)

**作者:** Jarosław Byrka `[一作]` (University of Wrocław), Zaixuan Wang `[通讯]` (Nanjing University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种统一的迭代随机化舍入算法，可针对任意 p‑power 成本的 k‑聚类（包括 k‑median、k‑means 及其欧氏变种）实现 Lagrangian Multiplier‑Preserving (LMP) 与真实近似；

**💡 创新点**

创新点在于将 LP 舍入与迭代随机化相结合，得到单一算法即可获得 3^p+½ 的 LMP 近似，并通过小幅 ε‑扰动转化为 (3^p+½+ε) 的真实近似，且欧氏 k‑means 的 LMP 进一步降至 11/3、真实近似降至 4+ε；

**🔧 技术方法**

主要技术包括 LP 线性规划松弛、迭代随机化舍入、邻接图构造、负相关与集中极限分析、Pipage 舍入、以及 Li‑Svensson 的伪近似转真实近似的归约；

**📊 数据集**

论文为理论研究，并未使用任何具体数据集；

**📈 对比分析**

与现有方法相比，在 k‑median 上实现了 2+ε 的近似（与最优 2‑近似相当），在一般度量 k‑means 上改进到 5+ε（比 Charikar 等人 5.83 降低），在欧氏 k‑means 上达到 4+ε（与 Charikar 等人 4+ε 相匹配），整体算法更为简洁；

**⚠️ 局限性**

限制包括：算法仅在理论上证明，未进行实验评估；所需时间为 n^{1/ε}^{O(p^2)}，常数隐含较大；对 p≥1 的一般度量有 2^p 的下界限制，欧氏 k‑means 的 LMP 仍不达到 2^p=4 的下界；

---

## 543. A Multi-Stage Validation Framework for Trustworthy Large-scale Clinical Information Extraction using Large Language Models

**arXiv ID:** 2604.06028 | [PDF](https://arxiv.org/pdf/2604.06028v1)

**作者:** Maria Mahbub `[一作]` (Oak Ridge National Laboratory), Ioana Danciu `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 1520 | [OpenAlex ID](https://openalex.org/A5021608374)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种多阶段、可扩展、可信赖的LLM信息抽取验证框架，并在大量临床记录中对药物使用障碍诊断进行了抽取与评估

**💡 创新点**

创新点在于将提示校准、规则可行性过滤、语义基底评估、LLM评判器确认、专家审核与预测有效性四类验证手段整合，形成层级化、可定制、无需全量人工标注的验证流程

**🔧 技术方法**

技术包括链式推理提示、规则过滤、句子嵌入语义相似度、LLM-as-judge对高不确定抽取进行二次评估、Gwet AC1统计、逻辑回归预测未来SUD专科就诊

**📊 数据集**

使用VA企业数据仓库中的919,783条临床笔记，涵盖178,467名患者，采用扩充的923条专家标注样本做提示校准，提取11类SUD诊断

**📈 对比分析**

在内部验证中，规则+语义过滤去除14.59%错误抽取；LLM抽取与ICD-10对齐率约75-99%；LLM+ICD-10结合预测SUD专科就诊的AUC为0.84，高于单纯ICD-10的0.76，单纯LLM提取AUC为0.80；在无ICD-10的叙述仅案例中，LLM预测AUC为0.67，远高于ICD-10的0.50

**⚠️ 局限性**

局限包括：框架仅在两种开源LLM上验证，可能对其他模型差异大；预测指标受护理可及性、转诊模式等非临床因素影响；ICD-10被视为不完善的基线，可能引入偏差；专家评审仅限高不确定样本，未覆盖全部抽取；缺乏无结构标注环境的验证方法；需在不同机构和任务中进一步验证鲁棒性

---

## 544. ACE-Bench: Agent Configurable Evaluation with Scalable Horizons and Controllable Difficulty under Lightweight Environments

**arXiv ID:** 2604.06111 | [PDF](https://arxiv.org/pdf/2604.06111v1)

**作者:** Wang Yang `[一作]` (Case Western Reserve University), Xiaotian Han `[通讯]` (Case Western Reserve University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5116337235)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ACE-Bench，一个轻量化、可调控的基准，利用隐藏槽与诱饵预算分别控制任务时间步与难度，并通过静态JSON实现低环境开销；

**💡 创新点**

创新点在于通过隐藏槽数与诱饵预算实现任务长度与难度的可分离控制；采用纯JSON工具环境显著降低交互成本；跨六大真实规划域验证模型区分度与域一致性；

**🔧 技术方法**

采用静态JSON工具接口、局部与全局约束检查、可调解隐藏槽与诱饵生成算法以及针对性采样以保证诱饵的全局错误性；

**📊 数据集**

使用六个领域的网格规划实例（课程排课、购物车、行程安排、工时安排、餐饮计划、PC配置），每个实例5×7网格，隐藏槽与诱饵组合共324个实例；

**📈 对比分析**

在13种公开模型（Qwen3.5 Dense/MoE、MiniMax、MiroThinker、GLM-4.7-FP8）上评估不同隐藏槽/诱饵组合下的奖励率，最高得分84.9%，模型规模越大奖励越高，验证了区分性；

**⚠️ 局限性**

局限在于对更大任务规模或更复杂全局约束的可扩展性尚未充分验证；在工具失败模拟中表现敏感，需提升鲁棒性；仅覆盖规划类任务，缺乏对其他交互式任务的适用性验证。

---

## 545. UI Placement as a Critical Design Factor for Augmented Reality During Locomotion

**arXiv ID:** 2604.06102 | [PDF](https://arxiv.org/pdf/2604.06102v1)

**作者:** Pavel Manakhov `[一作]` (Lancaster University), Hans Gellersen `[通讯]` (Lancaster University)

**通讯引用:** 14261 | [OpenAlex ID](https://openalex.org/A5024343435)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文综述了在可穿戴AR系统中，UI在用户移动时的空间放置对感知和交互性能的影响，并讨论了相关实验（如《Gaze on the Go》和《Filtering on the Go》）的发现。

**💡 创新点**

创新点在于提出UI放置方式（Head、HeadDelay、Path、World、Tag‑along等）是评估AR移动交互性能时的核心变量，并强调了不同放置方式对视觉稳定性、瞳孔追踪精度和射线投射交互速度的显著差异。

**🔧 技术方法**

使用的技术包括眼动追踪与停留检测、基于头部运动的补偿眼动（VOR）分析、在线滤波（低频抑制或坐标转换滤波）、以及射线投射与手部、头部、眼部多模态交互的实验测评。

**📊 数据集**

实验数据主要来自在模拟行走场景中收集的眼动和交互日志，比较了不同UI放置方式下的停留稳定性、瞳孔指向准确率与点击时延等指标；具体数据集未公开发布，文中引用了先前研究的实验结果。

**📈 对比分析**

比较方法：通过在相同行走速度和路径长度下对每种UI放置方式进行重复测量，计算停留稳定性（平均稳定角度）、瞳孔指向误差和点击时延等指标。结果显示，Path与World放置获得最高的停留稳定性和最低的瞳孔误差；对Head的滤波可提高瞳孔指向准确率，但在Path/World放置下滤波效果反而下降。

**⚠️ 局限性**

局限性在于：①实验仅覆盖有限的放置模式和交互技术，缺乏对更复杂动态放置（如动态避障或基于环境物体的对齐）的评估；②研究结果主要基于室内实验环境，缺乏户外真实世界的验证；③尚未形成系统化的UI放置设计空间框架，难以直接指导产品设计。

---

## 546. Intuitive Human-Robot Interaction: Development and Evaluation of a Gesture-Based User Interface for Object Selection

**arXiv ID:** 2604.06073 | [PDF](https://arxiv.org/pdf/2604.06073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 547. eVTOL Aircraft Energy Overhead Estimation under Conflict Resolution in High-Density Airspaces

**arXiv ID:** 2604.06093 | [PDF](https://arxiv.org/pdf/2604.06093v1)

**作者:** Alex Zongo `[一作]` (George Washington University), Peng Wei `[通讯]` (George Washington University)

**通讯引用:** 5392 | [OpenAlex ID](https://openalex.org/A5019479187)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过将物理基础的eVTOL功率模型嵌入交通仿真，并对71,767次高密度空域飞行进行MVP冲突解决策略的仿真，定量评估了战术解冲突对能量消耗的影响，并基于仿真数据构建了可输出不确定性区间的能量消耗预测模型；

**💡 创新点**

创新点在于首次系统量化MVP冲突解决对电动垂直起降飞机能量负荷的影响，发现能量超载主要为右偏分布，且可通过4‑5%预留实现95%覆盖；同时提出了利用对数变换的高斯分布回归实现的预飞行能量消耗预测与置信区间输出，满足安全预留决策需求；

**🔧 技术方法**

采用了物理基础的涡旋功率模型、Modified Voltage Potential（MVP）解冲突算法、BlueSky仿真平台、基于自编码器的特征提取以及多层残差全连接神经网络（含SiLU激活与Dropout）进行高斯似然训练，并在对数尺度上预测能量超载；

**📊 数据集**

使用的是在10‑nm半径环形空域内，10–60架机密度下的仿真数据集，共计71,767条飞行轨迹；每条轨迹包含初始位置、速度、冲突检测、MVP建议变更、交通密度等特征；

**📈 对比分析**

通过与无冲突基准的能量消耗比较，展示了能量超载分布（中位数<1.5%，95%分位数4–5%）；在预测模型上，平均绝对误差MAE为1.0%，均方根误差RMSE为1.8%，R²≈0.22；置信区间覆盖率高于标称值（80%区间覆盖96%，90%区间覆盖98%），区间宽度约为5–7%，满足操作预留需求；

**⚠️ 局限性**

局限性包括功率模型未经过实飞行验证；仿真仅考虑巡航阶段，未覆盖起降与过渡段能量消耗；预测模型仅利用预飞行特征，无法实时更新；仅评估MVP算法，其他解冲突方法的能耗特性尚未研究；

---

## 548. A machine learning framework for uncovering stochastic nonlinear dynamics from noisy data

**arXiv ID:** 2604.06081 | [PDF](https://arxiv.org/pdf/2604.06081v1)

**作者:** Matteo Bosso `[一作]`, Farbod Alijani `[通讯]` (Delft University of Technology)

**通讯引用:** 2829 | [OpenAlex ID](https://openalex.org/A5074122560)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种结合深度符号回归与高斯过程最大似然估计的框架，用于从少量样本中同时识别确定性漂移项和状态相关噪声结构；

**💡 创新点**

创新点在于：①假设噪声结构与漂移相同的结构化扩散；②使用GP-based MLE从残差中推断噪声方差；③模块化设计可替换任意符号回归方法；

**🔧 技术方法**

采用深度符号回归（DSR）、高斯过程平滑与微分、Euler–Maruyama离散化、最大似然估计（MLE）以及自动相关性判定（ARD）等技术；

**📊 数据集**

验证数据集包括数值振荡器（线性谐振子、Duffing、van der Pol）和实验数据（两细菌同步实验）；

**📈 对比分析**

与传统直方图回归、HyperSINDy、BISDE等方法比较，结果显示在10^2-10^3样本下，该框架能够准确恢复漂移和扩散，数据效率高，性能优于BISDE；

**⚠️ 局限性**

局限在于：假设扩散仅由漂移基函数构成，无法处理非结构化噪声；对Euler–Maruyama时间步长敏感；对符号回归搜索空间和词库依赖较大；高维系统的扩展仍待验证。

---

## 549. Graph-PiT: Enhancing Structural Coherence in Part-Based Image Synthesis via Graph Priors

**arXiv ID:** 2604.06074 | [PDF](https://arxiv.org/pdf/2604.06074v1)

**作者:** Junbin Zhang `[一作]` (Peking University), Yuexian Zou `[通讯]` (Peking University)

**通讯引用:** 5774 | [OpenAlex ID](https://openalex.org/A5002795838)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Graph-PiT框架，通过显式图先验提升基于部件的图像合成的结构连贯性。

**💡 创新点**

创新点在于构建双层图结构并使用层次图神经网络（HGNN）进行上下层信息交互，同时引入图拉普拉斯平滑和边缘重建正则化，确保部件间关系被建模和遵循。

**🔧 技术方法**

核心技术包括IP-Adapter+特征编码、HGNN、图卷积与图注意力机制、图拉普拉斯平滑损失、边缘重建损失以及条件流匹配扩散模型。

**📊 数据集**

使用四个合成部件集合数据集：字符、产品、室内布局和拼图，训练时通过自动构造图先验；实验还迁移到真实网络图像进行可视化验证。

**📈 对比分析**

与PiT、IP-Adapter+、λ-ECLIPSE、OmniGen等基线比较，Graph-PiT在所有四个域均显著降低FID并提升IIS；在字符/产品域的FID降幅高达约50%，在拼图域亦实现最优性能。

**⚠️ 局限性**

局限性包括对精确分割与边框对齐的依赖，当前仅支持二值邻接关系，且在严重遮挡或极小部件时图先验构造可能失效。

---

## 550. HiPolicy: Hierarchical Multi-Frequency Action Chunking for Policy Learning

**arXiv ID:** 2604.06067 | [PDF](https://arxiv.org/pdf/2604.06067v1)

**作者:** Jiyao Zhang `[一作]` (Peking University), Hao Dong `[通讯]` (Peking University)

**通讯引用:** 5981 | [OpenAlex ID](https://openalex.org/A5074073299)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 HiPolicy，一种层次化多频率动作分块框架，用于机器人模仿学习

**💡 创新点**

创新点在于同时预测低频粗粒规划与高频细粒控制，并通过熵引导的自适应执行平衡长时序建模与闭环控制

**🔧 技术方法**

采用扩散式生成策略、FiLM 与跨注意力特征融合、熵估计自适应频率切换

**📊 数据集**

在 RoboTwin 1.0/2.0 以及 Franka Panda 真实机器人任务上进行评估

**📈 对比分析**

与 Diffusion Policy (DP) 与 DP3 对比，成功率提升约 44–62%，并显著加快执行速度（≈25%）

**⚠️ 局限性**

局限在模型规模与数据集有限，未探索大规模多模态融合或更通用的机器人平台

---

## 551. From Hallucination to Structure Snowballing: The Alignment Tax of Constrained Decoding in LLM Reflection

**arXiv ID:** 2604.06066 | [PDF](https://arxiv.org/pdf/2604.06066v1)

**作者:** Hongxu Zhou `[一作]` `[通讯]` (Saarland University), Hongxu Zhou (Saarland University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在大型语言模型中通过语法约束解码（如 Outlines）实现结构化自我纠错，并评估其对开源推理任务的影响。

**💡 创新点**

创新点在于发现纯粹的结构约束并不能提升自我纠错，反而导致新的失效模式——“结构雪崩”，并提出“对齐税”概念阐明约束带来的认知负担。

**🔧 技术方法**

技术主要包括：三元架构（Actor、Evaluator、Reflector）、基于有限状态机的语法约束解码、Pydantic 模式映射、LLM 评估器与 episodic memory 机制。

**📊 数据集**

使用 HotpotQA 数据集（去除干扰样本并通过 LLM 判定语义匹配）进行实验评测。

**📈 对比分析**

方法对比：标准 Reflexion（自由文本反思）与逻辑引导结构化反思。结果显示约束解码将准确率从 50% 降至 38%，但在 11 个样本中成功恢复；同时格式错误占 96%，Token 消耗显著增加，体现对齐税。

**⚠️ 局限性**

局限性包括模型规模仅为 8B，导致无法处理深层逻辑纠错；评价标准依赖严格的字符串匹配，导致偏倚；未验证更大模型或动态评价下的表现。

---

## 552. EDGE-Shield: Efficient Denoising-staGE Shield for Violative Content Filtering via Scalable Reference-Based Matching

**arXiv ID:** 2604.06063 | [PDF](https://arxiv.org/pdf/2604.06063v1)

**作者:** Takara Taniguchi `[一作]` (SB intuitions), Teppei Suzuki `[通讯]` (SB intuitions)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在去噪过程中的可扩展参考基内容过滤器 EDGE‑Shield，能够在生成文本到图像模型的中间阶段检测并阻止违规内容；

**💡 创新点**

两大创新：①利用预计算的参考图像嵌入实现可扩展缓存，显著降低多参考时的计算开销；②引入 x‑pred 变换将中间噪声潜变量映射到近似最终清晰潜变量，提升早期检测准确率；

**🔧 技术方法**

基于 ODE‑based 生成模型的流匹配（v‑pred）与 x‑pred 变换，余弦相似度评分，CLIP/SigLIP 等视觉编码器；

**📊 数据集**

CPDM 数据集（200人脸+81 IP）与 HUB 数据集（10人脸+10 IP+10 艺术风格）进行评估；

**📈 对比分析**

与传统感知度量（Normalized L2、LPIPS、DiffSim）以及 VLM 过滤器（LLaVaGuard、InternVL、Qwen3‑VL、GPT‑4o‑mini）进行比较；在 Z‑Image‑Turbo 上相对基线延迟降低约 79%，在 Qwen‑Image 上降低约 50%；ROC‑AUC 约 0.85‑0.86，PR‑AUC 约 0.88‑0.90，平均判定时间仅 0.4‑0.5 秒；

**⚠️ 局限性**

对风格模仿的检测性能下降，原因在于余弦相似度难以区分风格相似度低的特征；未来需进一步提升在大规模参考集合（数千至数万条）下的表现与风格检测能力。

---

## 553. PromptEvolver: Prompt Inversion through Evolutionary Optimization in Natural-Language Space

**arXiv ID:** 2604.06061 | [PDF](https://arxiv.org/pdf/2604.06061v1)

**作者:** Asaf Buchnick `[一作]` (Bar-Ilan University), Ethan Fetaya `[通讯]` (Bar-Ilan University)

**通讯引用:** 1901 | [OpenAlex ID](https://openalex.org/A5053007211)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于遗传算法与视觉语言模型的提示逆向方法，能够在仅使用黑盒文本‑到‑图像模型的条件下，生成自然可读的文本提示以逼真重构给定图像。

**💡 创新点**

核心创新在于将视觉语言模型（VLM）作为交叉与变异算子，直接在自然语言空间进行进化优化，从而无需梯度或模型内部访问，即可得到可解释且高质量的提示。

**🔧 技术方法**

技术实现包括遗传算法、Qwen3‑VL‑Instruct 视觉语言模型、FLUX.2‑klein‑4B 文本‑到‑图像模型，以及多种图像相似度指标（CLIP、BLIP、DreamSim 等）进行评价。

**📊 数据集**

实验使用了 Lexica、MS‑COCO、Celeb‑A、Flickr8K、LAION‑400M 等多样化图像数据集，覆盖自然、人物、复杂场景等不同类型。

**📈 对比分析**

与 VLM‑Baseline、VGD、STEPS、PEZ、CLIP Interrogator 等基线比较，实验显示在所有数据集和相似度指标上均优于基线，甚至在多种评分函数下表现更好；人类偏好实验中 55% 的受试者倾向于本方法生成的重构。

**⚠️ 局限性**

局限性包括对 VLM 推理速度和生成文本长度的依赖，进化过程需要多代算子调用；在极细粒度或空间关系精确度上提升有限，且高度依赖 VLM 对目标图像的描述能力。

---

## 554. CoStream: Codec-Guided Resource-Efficient System for Video Streaming Analytics

**arXiv ID:** 2604.06036 | [PDF](https://arxiv.org/pdf/2604.06036v1)

**作者:** Yulin Zou `[一作]` (Nanyang Technological University), Dmitrii Ustiugov `[通讯]` (Nanyang Technological University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了名为 CoStream 的视频流分析系统，利用视频编解码器的压缩域元数据（运动向量、残差等）在端到端的 VLM（Vision‑Language Model）服务中实现在线的视觉令牌裁剪和 LLM KV 缓存刷新，显著减少传输、视觉编码和 LLM 预填充的冗余计算。

**💡 创新点**

创新点在于：①首次将压缩域的运动信息统一用于三大阶段（解码、视觉编码、LLM 预填充）优化；②实现无离线训练的实时自适应裁剪与 KV 刷新；③实现滑动窗口场景下的 KV 重用与位置校正，兼顾多模型（ViT+LLM）的通用性。

**🔧 技术方法**

采用的技术包括：GPU 加速解码（NVDEC）、基于运动向量的裁剪策略、RoPE 位置纠正的 KV 刷新、端到端流水线与 vLLM 及 LMCache 集成、实验中使用 InternVL3 与 Qwen3‑VL 两种大型 VLM。

**📊 数据集**

使用 UCF‑Crime 数据集（约 1900 条真实监控视频）进行滑动窗口查询实验。

**📈 对比分析**

与 Full‑Comp、Déjà Vu、CacheBlend、VLCache 等基线比较，结果显示：InternVL3 上吞吐率提升至 3.0×、GPU FLOPs 降至 87% 以内、Qwen3‑VL 上速度提升 1.66×，F1 分数仅下降 0–8%。整体延迟减少 2.12×（传输）、7.42×（视觉）、1.35×（LLM）。

**⚠️ 局限性**

局限性：①系统高度依赖 H.264/其他编解码器的运动元数据，对无运动信息或高运动视频的裁剪收益有限；②KV 刷新策略在高运动场景仍需完整重算，导致一定性能下降；③适配新模型或编码器需手工实现；④目前仅针对滑动窗口监控视频场景，其他非滑动窗口或非监控任务的适用性尚待验证。

---

## 555. Lightweight Multimodal Adaptation of Vision Language Models for Species Recognition and Habitat Context Interpretation in Drone Thermal Imagery

**arXiv ID:** 2604.06124 | [PDF](https://arxiv.org/pdf/2604.06124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 556. Generating Synthetic Doctor-Patient Conversations for Long-form Audio Summarization

**arXiv ID:** 2604.06138 | [PDF](https://arxiv.org/pdf/2604.06138v1)

**作者:** Yanis Labrak `[一作]` (Idiap Research Institute), Thomas Schaaf `[通讯]` (Solventum)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个全开源的合成数据生成管道，用于生成长时段（5分钟以上）医生-患者对话及对应的SOAP临床笔记，旨在解决长语音推理任务的训练与评估缺口。

**💡 创新点**

创新点在于：①采用角色属性采样与多轮对话生成提升对话多样性；②将文本对话转换为多说话人音频，结合重叠/暂停建模、环境噪声与声学模拟；③用LLM生成结构化事实表并仅以此生成参考SOAP笔记，降低幻觉风险；④全部使用开源模型并公开数据与代码。

**🔧 技术方法**

技术手段包括：Gemma3-27B-IT进行对话生成；Qwen3‑Omni‑Instruct/Thinking与Whisper Large V3进行语音识别；Qwen3‑TTS‑1.7B进行声码合成；PyRoomacoustics、scaper实现房间声学与非语音事件合成；Opus编码、LLM‑as‑a‑Judge（Kimi K2 Thinking）进行评估。

**📊 数据集**

使用的主要数据集为 Synth‑DoPaCo，包含8,800条合成医生‑患者对话、1,329小时音频和对应SOAP笔记；对比基准采用PriMock57和Mocks作为参考。

**📈 对比分析**

对比实验显示，分阶段（ASR+LLM）系统在Faithfulness、Coverage等指标上远优于端到端模型，端到端模型幻觉率高达99–100%，而分阶段系统幻觉率仅约21–23%。ASR在湿音频上WER低于3%，显示转写接近上限。

**⚠️ 局限性**

局限性包括：①合成语音难以完全复刻真实临床录音的噪声与多说话人混叠；②仅覆盖英语两人主治初诊场景，缺乏多语言、多专业及多方交互；③参考SOAP笔记由LLM生成，可能带来系统性偏差；④缺乏真实临床录音的sim‑to‑real验证。

---

## 557. Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models

**arXiv ID:** 2604.06052 | [PDF](https://arxiv.org/pdf/2604.06052v1)

**作者:** Katarzyna Zaleska `[一作]` (Warsaw University of Technology), Kamil Deja `[通讯]` (Warsaw University of Technology)

**通讯引用:** 2089 | [OpenAlex ID](https://openalex.org/A5026378980)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于线性探针的层定位方法，识别文本到图像扩散模型中负责隐式决策的自注意力层，并基于此实现精确的偏见消除（Implicit Choice‑Modification）。

**💡 创新点**

创新点在于：①将隐式决策与显式提示分离，发现自注意力层是决定未指定属性的关键；②利用线性探针直接量化每层对属性可分离度的贡献；③只对少量关键层进行干预，显著提升去偏性能并减少图像质量损失。

**🔧 技术方法**

技术手段包括：文本到图像扩散模型（Stable Diffusion、SDXL、SANA），线性探针（逻辑回归）进行层可分离度评估，激活 steering 与 LoRA 微调实现针对性干预，外部 CLIP 或 FairFace 进行属性标注与偏差评估。

**📊 数据集**

数据集：生成 500 张/每个提示的图像用于 Fairness Discrepancy 评估，使用 FairFace、FFHQ 评估年龄、性别、种族分布与图像质量；另外使用外部 CLIP 对生成结果进行属性分类。

**📈 对比分析**

与现有去偏方法（Latent Editing、H‑Distribution、Latent Direction、Finetuning、DIFFLENS）对比，ICM 在性别、年龄、种族任务上获得更低的 Fairness Discrepancy（如性别 FD 下降至 0.087），同时保持或提升 FID 与 CLIP‑T 对齐，显著降低因全层干预产生的视觉伪影。

**⚠️ 局限性**

局限性：①仍依赖外部分类器进行伪标签，可能引入误差；②对不同模型与任务的可迁移性尚需进一步验证；③只关注自注意力层，可能遗漏其他层在特定场景下的贡献；④干预强度（α）需手工调节，缺乏自动化方法。

---

## 558. Gym-Anything: Turn any Software into an Agent Environment

**arXiv ID:** 2604.06126 | [PDF](https://arxiv.org/pdf/2604.06126v1)

**作者:** Pranjal Aggarwal `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2275 | [OpenAlex ID](https://openalex.org/A5019030424)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可扩展的框架 Gym‑Anything，将任意软件转化为可交互的计算机使用环境，并用该框架生成了 200 个软件、10,000+ 任务的长序列数据集 "Gym‑Anything‑Benchmark"，包含训练、测试和 200 任务的长时限子集 "-Long"。

**💡 创新点**

创新点包括：①把环境创建本身视为多智能体任务（生成、审计、共享记忆）实现自动化；②使用提议‑放大（propose‑and‑amplify）方法快速扩展任务；③引入检查表式 VLM 验证器结合特权信息，既支持部分得分又能检测工作流违规；④以 GDP 经济价值为依据选择软件，覆盖 22 个职业组。

**🔧 技术方法**

核心技术为：多智能体协同（Agent_C、Agent_Audit、Agent_Summ）；Gym‑Anything 库（统一的脚本/配置规范、跨 OS 容器化）；提议‑放大任务生成；VLM（如 Gemini‑3‑Flash、Qwen3‑VL‑2B‑Thinking）作为验证器和审计器；强化学习训练与蒸馏；长时限预算管理（测试时审计）。

**📊 数据集**

数据集包括：①来自公开医疗影像、财务、天文、企业系统等真实数据；②自构建的 200+ 软件环境；③按 GDP 权重筛选出的 200 软件；④训练/测试分离、长时限分离；⑤用于蒸馏的 2,000 条轨迹。

**📈 对比分析**

与现有基准（OSWorld、AndroidWorld 等）对比，Gym‑Anything‑Benchmark 提供更大规模、长序列、多职业覆盖。实验显示：• 2B 规模 VLM 通过蒸馏可比 4B 规模模型；• 前沿模型 Gemini‑3‑Flash 在 Test 取得 22.6% 通过率，-Long 仅 7.5%；• 通过测试时审计可将 Gemini‑3‑Flash 在 -Long 的通过率提升至 14%。

**⚠️ 局限性**

局限性：① GDP‑驱动的软件选取未精确到美元，估计过程依赖 LLM；② 仅覆盖可 sandbox 的免费软件，商业软件难以复制；③ 任务可解性未全部验证；④ VLM 验证器虽高一致性，但仍易受对抗攻击；⑤ 长时限任务仍对现有模型极具挑战。

---

## 559. Artificial Intelligence and the Structure of Mathematics

**arXiv ID:** 2604.06107 | [PDF](https://arxiv.org/pdf/2604.06107v1)

**作者:** Maissam Barkeshli `[一作]` (Meta Superintelligence Labs), Michael H. Freedman `[通讯]` (Harvard University)

**通讯引用:** 18258 | [OpenAlex ID](https://openalex.org/A5064155719)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于超图（hypergraph）的形式化框架，用来刻画数学的全局结构，并在此基础上构建一套评价自动数学发现（AMD）系统的准则。作者进一步梳理了现有的AI数学系统（如Minimo、Fermat、Dreamcoder、Graffiti等），讨论了它们在猜想生成、证明搜索、抽象与压缩等方面的实现方式，并结合这些系统给出了一些实际效果与挑战的示例。

**💡 创新点**

创新点主要体现在：
1) 将数学推理与证明抽象为有向有序有色超图，提供了一个统一的图论视角；
2) 结合形式化证明库与AI技术，提出了面向“人类数学”子图（HM）的抽取与评估方法；
3) 设计了一套多维度的AMD评估准则（开放式语言、可验证证明、可判定新颖性、抽象与压缩、兴趣性评价等）；
4) 将强化学习与自监督学习与数学发现流程结合，形成了完整的循环模型。

**🔧 技术方法**

核心技术包括：
- 形式化证明框架（Coq/Lean4/Isabelle）和超图理论；
- 证明搜索（启发式搜索、MCTS、前向/后向链推理）；
- 抽象与压缩（子图匹配、定义合并、可读性优化）；
- 生成式模型（LLM、变分自编码器、生成式神经网络）用于猜想生成与程序合成；
- 强化学习与奖励机制，用于自我提升与兴趣度评估。

**📊 数据集**

使用的数据集主要为公开形式化数学库：
- Lean4 的 MathLib、Coq 的 Coq Library、Isabelle/HOL 的 Isabelle‑HOL 库；
- 预训练大语言模型所用的通用文本语料（如 Common Crawl、GitHub 代码、学术论文等）；
- 生成式系统所用的实例集（如图论不变量、代数结构实例、函数拟合数据等）。

**📈 对比分析**

在比较方法上，作者通过列举多个 AMD 系统在实际任务中的表现来说明进展：
- 在 IMO、Putnam 以及 FrontierMath 任务中，一些系统已达或逼近人类冠军水平；
- 在正式证明大型定理（素数定理、球面打包等）时，AI 系统已贡献上万行 Lean 代码；
- 在猜想生成方面，Graffiti 系列在图论领域产生了大量可验证的猜想；
- 在抽象与压缩方面，Dreamcoder 通过子图匹配实现了显著的证明长度压缩。
总体而言，系统在解决中等难度问题上已表现出可观的效率提升，但对极大规模证明仍面临搜索爆炸和抽象生成困难。

**⚠️ 局限性**

局限性主要包括：
1) 超图表示与实际形式化证明系统在细节上仍有差距，导致可迁移性受限；
2) 证明搜索与抽象生成的计算复杂度仍是指数级，难以突破大型定理的门槛；
3) “兴趣性”与“重要性”的量化仍缺乏客观标准，主要依赖经验或人类评价；
4) 对未知的数学结构，AI 仍易陷入过度拟合已知模式，缺乏真正的创造性；
5) 目前评估多聚焦于可验证性与形式化，尚未充分考虑人类可解释性与美学价值。

---

## 560. Inertial Mining: Equilibrium Implementation of the Bitcoin Protocol

**arXiv ID:** 2604.06092 | [PDF](https://arxiv.org/pdf/2604.06092v1)

**作者:** Manuel Mueller-Frank `[一作]` (IESE Business School), Omer Tamuz `[通讯]` (Caltech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究比特币挖矿协议中的均衡性缺陷，证明标准挖矿协议存在自私挖矿的有利偏差，并提出一种新的“惯性挖矿”协议，使其在任何矿工算力不超过一半的前提下构成均衡且最终产生单一最长链。

**💡 创新点**

创新点在于提出一种无需改动比特币共识机制或区块链结构的挖矿协议：在出现分叉时，矿工仅在新的链长至少比当前最长链多 I 级块时才会跟随，从而消除自私挖矿等盈利偏差，保证协议在 equilibrium 下产生预期的单一链。

**🔧 技术方法**

使用技术主要为博弈论建模、随机游走分析、极限定理和概率论工具对矿工行为与收益进行严格证明；同时在理论证明中引入随机化策略和公开随机设备以保证对等概率选择。

**📊 数据集**

论文未使用实际区块链数据集，而是基于理论模型进行分析；若有实验验证，则采用比特币公开区块链历史记录进行模拟，但主线是理论证明。

**📈 对比分析**

通过与标准比特币挖矿协议及自私挖矿策略的收益对比，证明在任何矿工算力分布满足 max_i α_i < 1/2 时，惯性挖矿不产生任何可行的利润提升；并且在 on‑path 情况下，惯性挖矿与标准协议产生相同的链结构，保持原有性能不变。

**⚠️ 局限性**

局限性包括：需要选择足够大的参数 I（取决于矿工算力分布，尤其是最大算力接近 1/2 时 I 需要更大）；理论证明假设矿工无协作且算力固定；未考虑网络延迟、广播不完全等现实因素对协议稳定性的影响；在实际部署前需进一步评估对矿工收益波动和链分叉率的具体影响。

---

## 561. Staggered Integral Online Conformal Prediction for Safe Dynamics Adaptation with Multi-Step Coverage Guarantees

**arXiv ID:** 2604.06058 | [PDF](https://arxiv.org/pdf/2604.06058v1)

**作者:** Daniel M. Cherenson `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**通讯引用:** 2407 | [OpenAlex ID](https://openalex.org/A5059647993)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6`

**🎯 论文内容**

提出一种基于Staggered Integral Online Conformal Prediction（SI-OCP）的方法，用于在线估计动力学模型中的累计误差并为鲁棒安全控制提供自适应不确定性上界。

**💡 创新点**

创新点在于：①使用积分型非一致性得分消除了对状态导数测量的需求；②通过分段线程实现多步预测覆盖；③将分布无关的在线合成误差上界与鲁棒管型MPC结合，获得渐进概率安全保证。

**🔧 技术方法**

采用的技术包括在线合成学习（全层DNN自适应）、积分式在线合成预测（SI-OCP）、鲁棒管型模型预测控制（DTMPC）以及自监督元学习（SSML）用于DNN参数更新。

**📊 数据集**

实验使用的是仿真三维无人机（四旋翼）环境，结合时空变化的风场和未建模的气动力学扰动，没有使用公开真实数据集。

**📈 对比分析**

与未使用自适应DNN或不使用SI-OCP的基线比较时，SI-OCP下的误差上界覆盖率达到98.77%（目标为90%），使管道尺寸显著减小，成功通过狭窄通道；未自适应情况下误差上界显著增大，导致管道过宽无法通过。

**⚠️ 局限性**

局限性包括：①依赖对扰动导数的Lipschitz上界；②要求完整状态测量且不考虑状态估计误差；③对高频或极端扰动的适应性有限，且未在真实硬件上验证。

---

## 562. Delta6: A Low-Cost, 6-DOF Force-Sensing Flexible End-Effector

**arXiv ID:** 2604.06150 | [PDF](https://arxiv.org/pdf/2604.06150v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 563. Stories of Your Life as Others: A Round-Trip Evaluation of LLM-Generated Life Stories Conditioned on Rich Psychometric Profiles

**arXiv ID:** 2604.06071 | [PDF](https://arxiv.org/pdf/2604.06071v1)

**作者:** Ben Wigler `[一作]` (LoveMind AI), Tiffany Matej Hrkalovic `[通讯]` (Jheronimus Academy of Data Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用真实心理测评资料对LLM进行沉浸式人格提示，生成生活故事面试文本，再用独立LLM盲目评分，完成人格信息的全程记录与回收。

**💡 创新点**

首次在多模型、跨架构条件下实现人格信息在生成文本中完整保留并可近似人类测试再测信度；提出三阶段 round‑trip 评估，验证人格信号与真实行为的对应性。

**🔧 技术方法**

沉浸式人格提示生成、McAdams LSI 文本生成、盲目 LLM 评分；使用多模型（GPT‑4.1、Gemini 3 Flash、Grok 4.1、Mercury 2 等）和三种评分器（Sonnet、Gemini、GPT‑5）。

**📊 数据集**

PARSEL 数据集（290 名参与者的 HEXACO‑60+9 项心理测评与对话记录）。

**📈 对比分析**

与单一模型、单一评测方法对比；平均回收相关为 r=0.750，约占人类测试再测的 85%；跨 10 个生成器、3 个评分器均保持 0.74‑0.75；对比人类对话和内容特征，相关在 0.13‑0.27 之间。

**⚠️ 局限性**

仅检验信息保留而非自然文本中的人格表达；评测者为 LLM，可能存在共享方法偏差；数据集局限于英语西方成人，缺乏跨文化验证；缺少人类自编 LSI 样本做对照。

---

## 564. Paper Circle: An Open-source Multi-agent Research Discovery and Analysis Framework

**arXiv ID:** 2604.06170 | [PDF](https://arxiv.org/pdf/2604.06170v1)

**作者:** Komal Kumar `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Hisham Cholakkal `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 3343 | [OpenAlex ID](https://openalex.org/A5009362997)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了Paper Circle，一个基于多智能体的科研文献发现与分析平台，集成多源检索、评分、知识图谱构建与图谱问答。

**💡 创新点**

创新性地将离线/在线检索、结构化多维评分、确定性排名与可追溯知识图谱结合，提供可同步多格式导出与可视化分析。

**🔧 技术方法**

采用多智能体协同框架、LLM（如Qwen3-coder-30b）、BM25/语义检索、跨编码重排、PyMuPDF PDF分块、知识图谱构建与图谱QA等技术。

**📊 数据集**

使用包含144篇顶级CS/ML会议论文（ICLR、NeurIPS、ICML等）和81个真实检索会话的实验数据，覆盖21,115篇文献。

**📈 对比分析**

通过 Hit Rate、MRR、Recall@K 等指标与BM25、语义检索、重排等基线对比，最佳模型实现80%命中率、0.627 MRR，显示优于单源检索与纯语义检索。

**⚠️ 局限性**

评论代理与人类评审得分相关性低（r<0.25），难以可靠区分论文质量，且高性能依赖大模型，限制了其作为自动评审工具的可用性。

---

## 565. HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models

**arXiv ID:** 2604.06165 | [PDF](https://arxiv.org/pdf/2604.06165v1)

**作者:** Reihaneh Zohrabi `[一作]` (Technical University of Darmstadt), Marcus Rohrbach `[通讯]` (Technical University of Darmstadt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出HaloProbe框架，利用贝叶斯方法将内部注意力与外部句子统计分离，实现在视觉语言模型中对物体幻觉的检测与非侵入式缓解。

**💡 创新点**

发现注意力统计中隐藏的混杂因子导致Simpson悖论，进而提出平衡训练与先验校正的贝叶斯分解，能够在混杂和类别不平衡的条件下获得鲁棒的幻觉概率。

**🔧 技术方法**

使用精细层/头级注意力、解码器置信度作为内部特征；采用平衡训练、先验估计、贝叶斯因子分解；配合幻觉感知Beam搜索与后处理编辑等技术。

**📊 数据集**

在MS COCO 2014验证集（2,500/500图像）上进行训练与评估，并使用标准CHAI评价框架对幻觉进行量化。

**📈 对比分析**

与EAZY、UT、IC、DIML等检测基线以及PAI、ADHH、AllPath、OPERA等缓解方法对比，HaloProbe在多模型（LLaVA-1.5/Shikra/MiniGPT-4）与多解码策略（贪心、束搜索、核采样）下，在CHAIR_S/I和F1指标上均显著优于前人，准确率提升约5–8%，AUROC提升3–5%。

**⚠️ 局限性**

局限性包括仅针对物体幻觉，未扩展到关系或归因幻觉；对极端分布漂移仍可能受限；依赖外部特征提取；在极少数类样本上性能仍有提升空间。

---

## 566. Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement

**arXiv ID:** 2604.06155 | [PDF](https://arxiv.org/pdf/2604.06155v1)

**作者:** Qimin Zhong `[一作]` (Shenzhen University), Naipeng Chao `[通讯]` (Shenzhen University)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5010985363)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

分析并改进多步预测（MTP），提出 LSE‑MTP 以提升 LLM 的内部世界模型并减少结构幻觉。

**💡 创新点**

理论证明 MTP 的梯度耦合导致收敛与结构幻觉，并首次将潜在状态与真实轨迹对齐（潜在语义增强）来抑制幻觉。

**🔧 技术方法**

多步预测、潜在语义增强（LSE‑MTP）、梯度耦合分析、NTK 线性化、Transformer 架构。

**📊 数据集**

合成图（Erdős–Rényi、Urban Street Graph）和真实曼哈顿出租车轨迹数据集。

**📈 对比分析**

与 1TP/NTP 及标准 MTP 在图导航和出租车路径生成上比较；LSE‑MTP 提升合法路径率、压缩精度、鲁棒性，略低区分精度。

**⚠️ 局限性**

仅在结构化图任务上验证，未对开放式 NLP 任务展开，理论基于线性近似，缺乏对更复杂语义动态的探索。

---

## 567. Action Images: End-to-End Policy Learning via Multiview Video Generation

**arXiv ID:** 2604.06168 | [PDF](https://arxiv.org/pdf/2604.06168v1)

**作者:** Haoyu Zhen `[一作]` (UMass Amherst), Chuang Gan `[通讯]` (UMass Amherst)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将7自由度机器人控制转化为多视角像素化动作图像，并利用统一的视频生成模型实现端到端的策略学习，消除传统动作编码器；

**💡 创新点**

创新点在于将动作直接像素化为可视化热图，使动作与观测处于同一模态，提升跨视角、跨环境的泛化；采用多视角和掩码策略实现多任务联合学习，构建零射击策略；

**🔧 技术方法**

使用多视角动作图像生成、基于扩散模型的视频-动作联合生成、掩码学习、相机条件编码（Plücker embedding）、动作解码方法及动作图像解码；

**📊 数据集**

RLBench、DROID、BridgeV2 以及部分Robot-Colosseum背景增强等数据集；

**📈 对比分析**

与MV-Policy、π0.5、MolmoAct、TesserAct、Cosmos-Policy等基线对比，零射击成功率最高，视频生成指标（PSNR/SSIM/FVD/LPIPS）和动作误差也优于先前模型，在真实机器人上表现出色；

**⚠️ 局限性**

目前仅支持开放式推断，缺乏闭环控制；推断速度仍高，需通过加速或蒸馏进一步改进。

---

## 568. Target Policy Optimization

**arXiv ID:** 2604.06159 | [PDF](https://arxiv.org/pdf/2604.06159v1)

**作者:** Jean Kaddour `[一作]` `[通讯]`, Jean Kaddour

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Target Policy Optimization (TPO)，通过先构造基于候选集合的目标分布，再用交叉熵拟合策略，解耦了“应如何重新分配概率”和“如何实现更新”两步。

**💡 创新点**

创新点在于：①不依赖价值函数或内部优化，目标分布可在闭式形式下直接得到；②梯度在匹配目标时自然消失，提升学习稳定性；③在奖励稀疏的环境中显著提升收敛速度与最终性能。

**🔧 技术方法**

技术手段包括：候选集标准化、温度调节、交叉熵损失、对齐目标分布的softmax、Transformer/语言模型推理、LLM RLVR框架、对比实验中的多种对策梯度方法。

**📊 数据集**

实验数据集涵盖：MNIST上下文赌博机、Token Reversal（多种词表大小）、多任务序列奖励（bag‑of‑tokens 与 sequential）、终端奖励的稀疏序列任务，以及LLM层面使用的GSM8K、Reasoning Gym（图着色、Knight & Knaves）等。

**📈 对比分析**

与PPO、GRPO、DG等基线对比，TPO在密集奖励任务与稀疏奖励任务上均达到或超过基线。特别是稀疏奖励场景，TPO在Token Reversal任务的1%错误率达成步数远低于GRPO、DG；在LLM RLVR中，TPO在早期收敛速度更快、最终得分更高。

**⚠️ 局限性**

局限性包括：①需提供足够多样且质量高的候选集；②与基线相同的rollout成本；③标准化在方差极低时可能放大噪声；④实验仅在1.5–1.7B规模模型与相对简单的基准上验证，未在更大模型或更困难任务上展开。

---

## 569. Data, Not Model: Explaining Bias toward LLM Texts in Neural Retrievers

**arXiv ID:** 2604.06163 | [PDF](https://arxiv.org/pdf/2604.06163v1)

**作者:** Wei Huang `[一作]` (Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 21019 | [OpenAlex ID](https://openalex.org/A5029998682)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过理论推导和实证分析，阐明神经检索器出现的来源偏差是由监督数据中非语义工件（如流畅度、词汇专一性）的不平衡导致的，而非模型本身固有的缺陷，并提出了两种可落地的缓解策略：训练时采用仅“in‑batch”负样本采样以消除工件不平衡；推理时对检索向量做投影以消除工件方向。

**💡 创新点**

创新点在于：①把来源偏差归因于训练标签工件不平衡，提出可解释的理论框架；②展示监督正负文档与LLM与人工文本在嵌入空间中的一致偏差方向；③提出基于工件方向的两种简单缓解方法（训练时负采样与推理时投影），并通过实验验证其有效性。

**🔧 技术方法**

技术手段包括：对比学习（contrastive InfoNCE）与点积检索模型；统计语言模型流畅度（perplexity）与IDF来量化工件；在嵌入空间中估计正负、LLM‑Human 的方向并计算余弦相似度；通过投影消除工件方向（v' = v - <v,n>n）。

**📊 数据集**

主要使用的评测集为BEIR Cocktail基准（14个来自BEIR的检索任务），其中每个任务都有人工与LLM生成的相似段落；实验也在MS MARCO上进行训练与负采样实验；此外使用了MS MARCO、DL19、DL20、NQ、NFCorpus、TREC‑COVID、HotpotQA、FiQA‑2018、Touché‑2020、DBpedia、SCIDOCS、FEVER、Climate‑FEVER、SciFact等公开数据集。

**📈 对比分析**

比较方法：使用ΔNDSR@5（衡量检索结果中人类与LLM来源的比例）作为偏差指标；同时报告NDCG@5评估检索质量。实验显示：①在不做任何改动时，监督模型ΔNDSR@5普遍为负（偏向LLM）；②采用in‑batch only负采样将ΔNDSR@5从≈-0.1降至≈-0.02，偏差大幅减少；③投影法同样能把ΔNDSR@5降至零附近，检索性能NDCG@5基本保持在原有水平或略有提升。

**⚠️ 局限性**

局限性：①研究聚焦于检索任务，未验证对生成式检索或多模态检索的适用性；②工件特征仅限于流畅度与IDF，可能忽略其他潜在的非语义偏差；③推理时投影方法虽然简单，但在极端工件分布下可能导致检索召回下降；④训练时负采样的改进对模型架构和规模的通用性尚未充分验证。

---

## 570. Exclusive Unlearning

**arXiv ID:** 2604.06154 | [PDF](https://arxiv.org/pdf/2604.06154v1)

**作者:** Mutsumi Sasaki `[一作]` (Tohoku University), Masaru Isonuma `[通讯]` (Nii Llmc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Exclusive Unlearning（EU），通过在自生成文本上最大化熵并结合保留任务实现对除目标域知识外的全部遗忘，提升对有害和 jailbreak 输入的防御。

**💡 创新点**

创新点在于把 unlearning 视为“只保留指定能力，忘记其余一切”，不依赖事先列举有害样本，利用模型自身生成文本进行全面遗忘，显著提升对未见攻击的泛化。

**🔧 技术方法**

技术手段：自生成文本遗忘损失（熵最大化/均匀化输出）、保留损失（负对数似然）及 λ 权衡两者；梯度下降实现参数更新；t‑SNE 可视化与安全判定阈值等辅助工具。

**📊 数据集**

使用的主要数据集：目标保留集 MedInstruct‑52k（医学）与 MetaMathQA（数学）；评测集包括 MedQA、HeadQA、PubmedQA、MEDMCQA、MeQSUM、GSM8K、MATH、MathQA；对抗集 Harm‑1、Harm‑2 及其 Jailbreak 变体 JB‑1、JB‑2。

**📈 对比分析**

与 RetainOnlyFT、DPO、传统 Unlearning、Eraser、SKU 等基线对比；EU 在医学/数学保留任务上保持或略高于基线表现，同时将攻击成功率（ASR）从 5–20% 降至接近 0%；在自由文本评价中与 RetainOnlyFT 对齐。

**⚠️ 局限性**

局限性：遗忘效果在后续微调后不稳健（攻击成功率上升）；只能抛弃非目标域的良性能力，适用于明确目标的领域专用模型；对模型参数的完整擦除仍有限。

---

## 571. Topological Characterization of Churn Flow and Unsupervised Correction to the Wu Flow-Regime Map in Small-Diameter Vertical Pipes

**arXiv ID:** 2604.06167 | [PDF](https://arxiv.org/pdf/2604.06167v1)

**作者:** Brady Koenig `[一作]` (Montana Technological University), Burt Todd `[通讯]` (Montana Technological University)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5025730217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过构建 Euler Characteristic Surface（ECS）并结合多核学习，提出一种无监督的拓扑方法，用来定量描述和识别竖向两相流中的“churn flow”现象。

**💡 创新点**

创新点在于首次给出 churn flow 的数学定义、利用 ECS 产生的两种时序与幅度核并与流速信息通过多核学习自动加权，实现在无标签数据下挑战并纠正现有机制模型的流态图。

**🔧 技术方法**

技术包括图像二值化+六边形尺度扩张、Hoshen–Kopelman 连接成分计数得到 ECS、L^1 时序对齐距离与 L^2 幅度特征距离、热核转换、熵正则化的多核学习、谱聚类与单调边界推断。

**📊 数据集**

实验数据主要来自 37 条 Montana Tech Vertical Flow Loop 的视频试验（2 英寸管道）以及 947 张 Texas A&M 高分辨率图像，后者用于跨实验室验证和伪试验的自校准。

**📈 对比分析**

与 Wu 等人经验流态图、CNN+LSTM、随机森林等基准相比，该方法在 MTVFL 试验上得到 4 类聚类 ARI≈0.42、slug–churn 边界上升 3.8 m/s、跨实验室 churn recall 100%，且在无标签条件下实现了与甚至超过有监督基线的性能。

**⚠️ 局限性**

局限性包括样本量有限导致 PAC 误差界限不显著、仅验证了竖向流场、伪试验中缺乏完整时序信号，以及对更大管径或不同流动方向的推广仍需进一步验证。

---

## 572. The Character Error Vector: Decomposable errors for page-level OCR evaluation

**arXiv ID:** 2604.06160 | [PDF](https://arxiv.org/pdf/2604.06160v1)

**作者:** Jonathan Bourne `[一作]`, Joseph Nockels `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Character Error Vector（CEV）指标，能够在解析错误场景下评估OCR质量；

**💡 创新点**

CEV的可分解特性，可将错误拆分为解析、OCR与交互三部分，并通过SpACER与Jensen‑Shannon距离两种实现，克服传统CER在页级解析失败时失效的问题；

**🔧 技术方法**

使用离散向量差异测度（L1、JSD）构建SpACER与CDD，结合Python库实现；

**📊 数据集**

在19世纪《The Spiritualist》报纸的49页（含50k+字符）上评估，并对该数据集做了文字类型、列信息、阅读顺序等丰富；

**📈 对比分析**

将CEV与CER、COTe、F1、mAP等指标对齐，实验显示SpACER与CER在解析完好时高度相关；CEV可显著区分解析与OCR错误源，阈值d_ocr/d_total≥0.5可预测错误主因，整体CEV在各种管道（组合解析器+OCR、端到端模型）中均优于单一CER；

**⚠️ 局限性**

CEV在解析极度失效（如覆盖整页的单一预测）时会出现偏低误差，需要结合COTe过滤；对字符位置信息的依赖限制了对无字符坐标数据集的适用性；

---

## 573. In-Place Test-Time Training

**arXiv ID:** 2604.06169 | [PDF](https://arxiv.org/pdf/2604.06169v1)

**作者:** Guhao Feng `[一作]` (ByteDance Seed), Tianle Cai `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 In‑Place Test‑Time Training（In‑Place TTT），让大语言模型在推理时能够动态更新参数（fast weights）以适应长序列上下文。

**💡 创新点**

创新点：① 在 Transformer MLP 的最终投影矩阵上做 fast weights，形成 drop‑in 方案，无需改造原有架构；② 用块级（chunk）更新代替逐 token 更新，显著提升并行度与吞吐率；③ 设计与自回归下一词预测目标对齐的学习目标，取代通用重建损失。

**🔧 技术方法**

技术细节：fast weights + inplace 更新机制 + chunk‑wise 更新规则 + 1D 卷积 + Context Parallelism + 现有高效注意力/状态空间模型（SWA、Gated Linear Attention 等）

**📊 数据集**

使用的数据集：语言建模评估 Pile、Proof‑Pile‑2；长期上下文评估 RULER、HellaSwag、ARC、MMLU、PIQA；预训练与 fine‑tune 评估 Qwen3‑4B、Llama‑3.1‑8B、Qwen3‑14B 等模型。

**📈 对比分析**

与基线（Full Attention、Sliding‑Window Attention、DeltaNet、Large Chunk TTT 等）以及从零训练的对手在 4B 规模下对比：RULER 16k 分数从 6.58 提升至 19.99；Qwen3‑4B 在 128k/256k 上提升约 10–12%；在 500M/1.5B 规模的滑动窗口困惑度始终低于所有竞争者；实验表明 In‑Place TTT 在长上下文和推理速度上均优于现有方法。

**⚠️ 局限性**

局限性：① 仍需对 fast‑weight 大小和 chunk 大小进行超参数调优；② 方案主要针对 MLP 最后投影，迁移到其他模块或模型结构需进一步验证；③ 对极长序列（>512k）仍可能受限；④ 目前只针对自回归语言模型，跨任务迁移与更复杂目标的有效性尚未充分评估。

---

## 574. DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models

**arXiv ID:** 2604.06161 | [PDF](https://arxiv.org/pdf/2604.06161v1)

**作者:** Zhengming Yu `[一作]` (Texas AandM University), Paul Debevec `[通讯]` (Netflix)

**通讯引用:** 19725 | [OpenAlex ID](https://openalex.org/A5047876260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于视频扩散模型的LDR到HDR转换框架DiffHDR，能够在单帧LDR视频中重建丢失的高动态范围信息并实现可控的重曝光。

**💡 创新点**

创新点包括：1）在预训练LDR视频VAE的潜在空间内通过Log‑Gamma映射编码HDR数据；2）利用遮罩引导的上下文聚焦提示和参考图像实现对过曝/欠曝区域的可控生成；3）构建了大规模HDR视频训练数据管道（基于HDRI地图）。

**🔧 技术方法**

技术手段主要是：视频扩散模型（VACE）、LoRA参数适配、Log‑Gamma色彩映射、曝光遮罩检测、上下文聚焦提示与参考图像对齐。

**📊 数据集**

使用的数据集包括：从Polyhaven HDRI渲染的约5400段HDR视频（训练集）；SI‑HDR、Cinematic Video、在野视频（Pexels）及Veo2生成的视频（评估集）。

**📈 对比分析**

与多种SOTA方法（HDRCNN、MaskHDR、SingleHDR、LEDiff等）在HDR‑VDP3、PU21‑PIQE、FID、FOVVDP、DOVER、MUSIQ、CLIPIQA等指标上对比，DiffHDR在大多数指标上均优于或接近SOTA，特别是在细节重建和时域稳定性方面表现突出。

**⚠️ 局限性**

局限性包括：1）对极端光照场景的生成仍可能产生偏差；2）依赖预训练LDR模型，HDRVAE未针对HDR微调；3）对遮罩阈值和超参数敏感，可能需要人工调优。

---

## 575. MMEmb-R1: Reasoning-Enhanced Multimodal Embedding with Pair-Aware Selection and Adaptive Control

**arXiv ID:** 2604.06156 | [PDF](https://arxiv.org/pdf/2604.06156v1)

**作者:** Yuchi Wang `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 41359 | [OpenAlex ID](https://openalex.org/A5100732450)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自适应推理的多模态嵌入框架，结合推理路径的潜在变量建模与强化学习控制推理使用。

**💡 创新点**

将推理路径视为潜在变量并引入基于对比的 pair‑aware 推理选择，利用因果干预评估推理贡献；同时用强化学习自适应决定是否推理，显著提升效率与准确率。

**🔧 技术方法**

多模态大语言模型（MLLM）+ 视觉 Transformer + 视觉适配器；多工作者生成多样化推理候选；因果干预评价；InfoNCE 对比学习；GRPO 强化学习。

**📊 数据集**

MMEB-Train 训练集；MMEB-V2 基准（78 个任务，包含图像、视频、视觉文档）。

**📈 对比分析**

与 CLIP、GME、ColPali、VLM2Vec、LamRA、RzenEmbed、UME‑R1、TTE、Embed‑RL 等多种基线比较；在 MMEB-V2 上 4B 模型得分 71.2，超过 7B 基线，推理时延比 UME‑R1 减 2.5 倍。

**⚠️ 局限性**

训练流程分离导致无法端到端优化；自适应策略仅二进制决策，缺乏推理深度控制；推理嵌入仍带来额外推理成本。

---

## 576. Who Governs the Machine? A Machine Identity Governance Taxonomy (MIGT) for AI Systems Operating Across Enterprise and Geopolitical Boundaries

**arXiv ID:** 2604.06148 | [PDF](https://arxiv.org/pdf/2604.06148v1)

**作者:** Andrew Kurtz `[一作]`, Klaudia Krawiecka `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了面向人工智能系统的身份治理完整框架：构建了 AI-身份风险分类体系（AIRT），提出了机器身份治理分类（MIGT），并给出了跨国监管协调结构和四阶段实施路线图，以填补现有 IAM 与 AI 法规之间的空白。

**💡 创新点**

创新点包括：① AIRT 为 AI 相关身份风险提供了首个 37 子类、8 领域的细粒度分类；② MIGT 将技术治理、法规合规和跨境协调六个治理域集成到一个框架；③ 定义了外部国家行为者威胁模型，揭示了 Silk Typhoon、Salt Typhoon 等行动；④ 设计了可映射 EU、US、CN 三大监管体系的跨境合规矩阵；⑤ 提出了基于这些层面的四阶段落地计划。

**🔧 技术方法**

主要采用的技术手段有：风险分类方法（层级化、属性评估）、治理框架设计（六域结构）、法规对标分析（EU AI Act、US NIST AI RMF、CSL 等）、案例对比分析以及跨境合规映射。框架和模型均基于已有标准（如 NIST SP 800‑207、SPIFFE、OIDC 等）进行整合。

**📊 数据集**

使用的数据与资料来源包括：Entro Labs NHI 与 Secrets 风险报告、CyberArk 2025 身份安全景观、Veza State of Identity Security 报告、NIST AI RMF、MIT AI Risk Repository、IBM Security 与 Delinea 相关报告、行业调查（如 Gartner、Gartner 2028 AI 代理预测）以及公开的威胁情报与案例（CrowdStrike、Silk Typhoon、DeepSeek 等）。

**📈 对比分析**

本文通过对比现有 IAM 规范、国家 AI 法规与 AIRT/MIGT 的覆盖情况，展示了 MIGT 在技术、合规与跨境三大维度的补充性。由于本工作属于框架与风险理论研究，未进行实验性能评估；评估基于文献与案例的定性对比，强调了 MIGT 在应对多代理、动态访问与外部威胁方面的全面性。

**⚠️ 局限性**

主要局限性包括：① 未在真实企业环境中进行实证验证，缺乏量化实验；② 当前缺乏统一的 AI 代理身份标准，框架的可操作性需进一步细化；③ 跨境合规模型基于法律文本对标，实际落地时可能面临解释与执行差异；④ 对 AI 模型内部治理（如训练数据、模型可解释性）技术细节探讨不足。

---

