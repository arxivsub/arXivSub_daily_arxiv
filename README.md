# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-25 | 今日论文总数: 499

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Smoothed Elicitation Complexity for Approximate $Γ$-calibration of Discrete Classification Tasks

**arXiv ID:** 2605.23017 | [PDF](https://arxiv.org/pdf/2605.23017v1)

**作者:** Jessica Finocchiaro `[一作]` (Boston), Drona Khurana `[通讯]` (Colorado)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了多分类预测器的近似属性校准，提出通过将离散属性映射为连续可微的 Lipschitz 属性并在其上实现校准，随后再对连续属性进行后处理以恢复原始离散决策。

**💡 创新点**

创新点在于：①首次给出了对离散属性的近似校准理论结果；②构造了可通过后处理得到原离散属性的连续 Lipschitz 属性；③引入了“可平滑属性（smoothed property）”的概念，显著降低了校准所需的维度（从 n 下降到 d），从而减轻了样本复杂度和计算复杂度。

**🔧 技术方法**

使用的技术主要包括：属性诱导（elicitation）理论、Lipschitz 连续性分析、后处理算法、近似校准误差界定，以及对多分类分布校准（distribution calibration）与属性校准（property calibration）的理论比较。

**📊 数据集**

论文为理论研究，没有使用具体的数据集；若要实验验证，可采用公开的多分类数据集如 ImageNet、CIFAR‑10/100 等进行模型预测与校准误差评估。

**📈 对比分析**

方法通过理论推导给出了近似 Γ‑校准与离散 γ‑校准的误差上界。相比传统需对完整分布进行离散化的校准方法，该方案在维度更低的连续属性空间上实现校准，理论上可获得更小的样本复杂度与更高的计算效率。实验上可通过与传统分层校准或温度标定等方法对比，验证校准误差降低与模型性能的提升。

**⚠️ 局限性**

局限性包括：①只适用于“强可排序”（strongly orderable）的离散属性，未覆盖所有离散决策；②对 Lipschitz 连续属性的构造与后处理可能在实际实现中需要额外假设或复杂度；③论文主要给出理论界限，缺乏实证验证；④在实际应用中如何选取合适的连续属性和后处理策略仍是一个开放问题。

---

## 2. Learned Relay Representations for Forward-Thinking Discrete Diffusion Models

**arXiv ID:** 2605.22967 | [PDF](https://arxiv.org/pdf/2605.22967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 3. The Readout Shortcut: Positional Number Copying Dominates Arithmetic CoT Readout in Small Language Models

**arXiv ID:** 2605.22870 | [PDF](https://arxiv.org/pdf/2605.22870v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 4. Step in Tine: Forking Processes in Functional Choreographies

**arXiv ID:** 2605.23031 | [PDF](https://arxiv.org/pdf/2605.23031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 5. An Axiomatic Theory of Tie-Breaking: Impossibility, Characterization, and Decomposition

**arXiv ID:** 2605.22846 | [PDF](https://arxiv.org/pdf/2605.22846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 6. Knowledge Distillation for Low-Resource Open-source Text-to-SQL Model

**arXiv ID:** 2605.22843 | [PDF](https://arxiv.org/pdf/2605.22843v1)

**作者:** Tianhao Qiu `[一作]` (Shenzhen University), Xiaojun Chen `[通讯]` (Shenzhen University)

**通讯引用:** 7712 | [OpenAlex ID](https://openalex.org/A5100369445)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种知识感知的Text-to-SQL框架，该框架构建了任务特定的知识库，包括模式语义、缩写、业务逻辑和查询模式，并将其注入到训练和推理中。

**💡 创新点**

创新点在于通过构建结构化的任务特定知识，增强了模型在低资源环境下的推理能力和SQL生成的准确性，尤其是在领域特定的数据库中。

**🔧 技术方法**

使用了知识感知的训练和推理技术，包括知识增强的上下文学习（KE-ICL）和知识增强的强化学习（KE-RL）。

**📊 数据集**

在七个基准数据集上进行了实验，涵盖了通用和领域特定的数据集，如EHRSQL和ScienceBenchmark。

**📈 对比分析**

与现有方法相比，KE-ICL和KE-RL在所有基准测试中均显著提高了性能，尤其是在低资源和领域特定设置下，表现出更好的泛化能力和适应性。

**⚠️ 局限性**

限制在于构建和维护高质量知识库需要大量的努力和领域专业知识，可能限制可扩展性并增加成本。此外，LLM生成的输出可能仍然存在幻觉或与领域约束不一致的问题。

---

## 7. Approximate Machine Unlearning through Manifold Representation Forgetting Guided by Self Mode Connectivity

**arXiv ID:** 2605.22871 | [PDF](https://arxiv.org/pdf/2605.22871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 8. FederatedRSF : Federated Random Survival Forests for Partially Overlapping Medical Data

**arXiv ID:** 2605.22954 | [PDF](https://arxiv.org/pdf/2605.22954v1)

**作者:** Maryam Moradpour `[一作]` (Institute for Predictive Deep Learning in Medicine and Healthcare), Anne-Christin Hauschild `[通讯]` (Institute for Predictive Deep Learning in Medicine and Healthcare)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种用于部分重叠医疗数据的联邦随机生存森林方法，旨在在不共享原始数据的情况下进行生存预测。

**💡 创新点**

创新点在于能够在特征空间异质性情况下，通过仅交换特征兼容的生存树来实现联邦学习，从而提高模型的鲁棒性和可推广性。

**🔧 技术方法**

使用了联邦学习和随机生存森林（RSF）技术。

**📊 数据集**

使用了GBSG2乳腺癌数据集，该数据集包含686个样本和8个临床协变量。

**📈 对比分析**

与本地训练和集中训练模型进行了比较，结果表明联邦模型的性能与集中训练设置相当，C-Index的平均提升为0.027，且在统计上显著。

**⚠️ 局限性**

局限性在于特征异质性本身导致的不可避免的判别损失，尽管联邦模型在统计上与集中模型无显著差异，但仍存在一定的性能差距。

---

## 9. Graph Alignment Topology as an Inductive Bias for Grounding Detection

**arXiv ID:** 2605.22963 | [PDF](https://arxiv.org/pdf/2605.22963v1)

**作者:** Paul Landes `[一作]`, Jimeng Sun `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建参考文本与生成文本之间的对齐语义图，并基于该图训练图神经网络进行谬误检测

**💡 创新点**

首次将语义图对齐结构作为显式归因偏置，用最大流权重的交叉对齐边强化结构信息

**🔧 技术方法**

AMR/Calamr图构造、语义对齐与流网络加权、图卷积网络（GCN）+注意力池化

**📊 数据集**

四个基准：halueval、hdm、medhallu、pubmedqa

**📈 对比分析**

与检索增强、SelfCheckGPT、RefChecker等方法比较，在halueval与hdm上分别取得94.8%与89.2% F1，超越GPT‑4o及其它基线

**⚠️ 局限性**

对图构造/对齐误差敏感，部分消融实验不单调；需在临床场景进行前瞻性验证，计算成本相对较高

---

## 10. GazeBehavior Annotation Toolkit (GBAT): AI-powered toolkit for automatic annotation of egocentric eye-tracking and video data of child-caregiver interaction

**arXiv ID:** 2605.22962 | [PDF](https://arxiv.org/pdf/2605.22962v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 11. MedExpMem: Adapting Experience Memory for Differential Diagnosis

**arXiv ID:** 2605.22872 | [PDF](https://arxiv.org/pdf/2605.22872v1)

**作者:** Qianhan Feng `[一作]` (Chinese University of Hong Kong), Qi Dou `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 28683 | [OpenAlex ID](https://openalex.org/A5090516040)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了MedExpMem框架，使基于视觉-语言模型的诊断代理能够积累差异诊断的专业知识。

**💡 创新点**

创新点在于通过结构化的成对经验笔记来捕捉差异诊断的关键特征，而不是孤立的疾病描述，从而支持临床比较推理。

**🔧 技术方法**

使用了经验记忆框架，采用了两阶段的记忆构建过程，模拟医生学习的过程。

**📊 数据集**

使用了Eurorad数据集，这是一个开放的放射学教学文件，包含超过10,000个病例，涵盖多个放射学亚专业。

**📈 对比分析**

与传统的检索增强生成方法相比，MedExpMem在多个模型和规模上均表现出一致的准确性提升，最大提升达到7.0%。

**⚠️ 局限性**

限制在于模型的记忆质量和检索精度可能影响整体性能，且在某些情况下可能引入有害的效果。

---

## 12. Transcoders Trace Visual Grounding and Hallucinations in Vision-Language Models

**arXiv ID:** 2605.22902 | [PDF](https://arxiv.org/pdf/2605.22902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 13. Suicide Risk Assessment from AI-powered Video Surveillance: An Interpretable Framework for Prevention in Metro Stations

**arXiv ID:** 2605.22904 | [PDF](https://arxiv.org/pdf/2605.22904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 14. The Geometry of Cooperative Game Solutions: Stratified Egalitarian Shapley Values

**arXiv ID:** 2605.22847 | [PDF](https://arxiv.org/pdf/2605.22847v1)

**作者:** Frank M. V. Feys `[一作]` `[通讯]`, Frank M. V. Feys

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种几何框架，用于分析有限玩家合作博弈中的线性价值映射，特别是通过Harsanyi-红利分解引入的内积结构。

**💡 创新点**

创新点在于证明了内积是内在的，并且高效对称线性价值映射的子空间具有清晰的结构定理，提供了对不同解决方案的定量比较。

**🔧 技术方法**

使用了Harsanyi内积和正交分层技术，构建了线性同构关系，允许对高效对称线性价值映射进行分层表述。

**📊 数据集**

使用了有限玩家的合作博弈数据集，特别是针对n=4的情况进行了详细分析。

**📈 对比分析**

通过与Shapley值、Banzhaf值、平等剩余分配值和团结值的比较，发现Banzhaf值与Shapley轴几乎正交（R^2 ≈ 1%），平等剩余分配值适度对齐（R^2 ≈ 38%），团结值几乎完全对齐（R^2 ≈ 99.6%）。

**⚠️ 局限性**

限制在于该框架主要针对线性价值映射，非线性映射的情况未能涵盖，且对其他解决方案的分析仍需进一步探索。

---

## 15. Verified Task-Space Motion Planning Under Joint-Space Constraints

**arXiv ID:** 2605.22991 | [PDF](https://arxiv.org/pdf/2605.22991v1)

**作者:** Hanjiang Hu `[一作]` (Robotics Institute Carnegie Mellon University), Yebin Wang `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对任务空间中的基于 Bug2 的反应式规划者在面对关节角度限制时可能出现的轨迹漂移和关节越界问题，本文提出一种基于 Sum‑of‑Squares（SOS）优化的卡尔曼可达盒子计算方法，并将其与 Bug2 规划器结合，实现每一步都能保证关节位移在允许范围内。

**💡 创新点**

创新点在于：① 用二阶多项式逆运动学逼近并通过 S‑procedure 构造一个低维半正定程序（SDP）来精确计算最大可达的任务空间超矩形；② 通过二分搜索实现子毫秒级的在线可达盒子求解；③ 将该可达盒子作为步长自适应因子，消除了传统固定步长导致的关节越界与跟踪漂移。

**🔧 技术方法**

采用的主要技术包括：二阶多项式逆运动学（利用雅可比伪逆与二阶校正）、Sum‑of‑Squares 程序与 S‑procedure 约束、半正定规划（SDP）以及快速二分搜索实现的高效可达盒子求解。

**📊 数据集**

实验数据集为 3 链 planar 机器人（n=3）在平面上与单圆障碍物相遇的 94 个对抗性场景，这些场景是按照起始条件数、条件数增长、可达盒子正向性等多重过滤器自动生成的，覆盖六种不同的每步关节位移上限。

**📈 对比分析**

与传统 Bug2（固定步长 s=δ/κ₀、使用雅可比伪逆并裁剪关节位移）比较，SOS‑verified Bug2 在所有 94 个场景中均无关节越界、100% 达成目标、路径长度约 1.2–1.5 倍直线距离、步数 27–33 步（比 vanilla 的 35–127 步更少），计算时间每步约 0.3 ms（仍低于实时阈值）。

**⚠️ 局限性**

局限性包括：① 目前仅在低维（平面）3 关节机器人上验证；② 依赖非奇异雅可比矩阵且对高度耦合或高 DOF 系统的可扩展性未测试；③ 需要预先估计二阶多项式逼近误差，若误差较大需减小搜索半径，影响实时性；④ 仅针对关节位移上限，未考虑速度/加速度或碰撞约束的复杂交互。

---

## 16. MARGIN: Runtime Confidence Calibration for Multi-Agent Foundation Model Coordination

**arXiv ID:** 2605.22949 | [PDF](https://arxiv.org/pdf/2605.22949v1)

**作者:** Joss Armstrong `[一作]` `[通讯]` (Ericsson), Joss Armstrong (Ericsson)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种名为MARGIN的在线置信度校准方法，旨在解决多智能体系统中置信度的误校准问题。

**💡 创新点**

MARGIN通过从任务流中学习每个智能体的置信度校准因子，避免了传统设计时间校准方法的局限性，能够在分布变化时保持有效。

**🔧 技术方法**

使用了对称的指数加权移动平均（EWMA）和贝叶斯收缩混合技术，具有三个超参数（α = 0.04，K = 3，k_s = 100）并且计算开销极小。

**📊 数据集**

在19个基础模型和8个基准测试上进行了评估，涉及超过50,000个观察数据。

**📈 对比分析**

与设计时间基线相比，MARGIN在分布变化下实现了3-6倍更低的校准误差。在多智能体选择中，MARGIN将配对分辨率从45-56%（低于随机水平）提高到70-89%，并在四个基准中的三个超越了最佳模型的表现。

**⚠️ 局限性**

MARGIN依赖于二元结果信号（正确/错误），限制了其在需要可验证答案的任务中的适用性。此外，置信度带的数量是一个设计选择，没有严格的选择标准。

---

## 17. EVE-Agent: Evidence-Verifiable Self-Evolving Agents

**arXiv ID:** 2605.22905 | [PDF](https://arxiv.org/pdf/2605.22905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 18. When AI Takes Sides on Questions of Faith: Persistent Asymmetries in AI-Mediated Faith Guidance

**arXiv ID:** 2605.22975 | [PDF](https://arxiv.org/pdf/2605.22975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 19. GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation

**arXiv ID:** 2605.22882 | [PDF](https://arxiv.org/pdf/2605.22882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 20. CoMoGen: COntrollable MOtion Dynamics and Interactions with Mask-Guided Video GENeration

**arXiv ID:** 2605.22996 | [PDF](https://arxiv.org/pdf/2605.22996v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 21. FuRA: Full-Rank Parameter-Efficient Fine-Tuning with Spectral Preconditioning

**arXiv ID:** 2605.22869 | [PDF](https://arxiv.org/pdf/2605.22869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 22. Learnability-Informed Fine-Tuning of Diffusion Language Models

**arXiv ID:** 2605.22939 | [PDF](https://arxiv.org/pdf/2605.22939v1)

**作者:** Shubham Parashar `[一作]` (Texas A&M University), Shuiwang Ji `[通讯]` (Texas A&M University)

**通讯引用:** 14024 | [OpenAlex ID](https://openalex.org/A5052278550)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于学习能力的微调方法（LIFT），旨在提高扩散语言模型（DLMs）的推理能力。

**💡 创新点**

创新点在于通过分析稀有和常见标记的学习能力，提出了一种新的微调算法，能够在不同的扩散时间步骤中选择合适的标记进行训练，从而提高模型的学习效率。

**🔧 技术方法**

使用了学习能力驱动的微调算法（LIFT），结合了模型信心和扩散时间来构建学习能力驱动的掩码。

**📊 数据集**

使用了来自四个后训练推理数据集的0.5B掩码标记进行分析，数据集包括s1K、Nemotron后训练数据集、Mixture of Thoughts和DociThink-RL。

**📈 对比分析**

与现有的SFT基线相比，LIFT在六个推理基准上表现优越，在AIME'24和AIME'25上实现了高达3倍的相对增益，显示出其有效性。

**⚠️ 局限性**

限制在于LIFT的性能依赖于选择的超参数H，过高的H值可能导致其表现接近于传统的SFT，且在不同数据集上的泛化能力仍需进一步验证。

---

## 23. Expressive Power of Deep Homomorphism Networks over Relational Databases

**arXiv ID:** 2605.22852 | [PDF](https://arxiv.org/pdf/2605.22852v1)

**作者:** Moritz Schönherr `[一作]` (Leipzig University), Arie Soeteman `[通讯]` (University of Amsterdam)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5054688147)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了深同态网络（DHNs），作为一种适合于关系数据库学习的图学习架构，分析了其表达能力，并与SQL的相关性进行了探讨。

**💡 创新点**

创新点在于DHNs提供了比标准图神经网络（GNNs）更高的表达能力，并且与SQL的连接使其在处理关系数据库时更为高效。

**🔧 技术方法**

使用了深同态网络（DHNs）技术，结合了同态计数和复杂模式的消息传递机制。

**📊 数据集**

使用了合成数据集进行实验，特别关注局部传递性和太阳属性的预测任务。

**📈 对比分析**

与标准GNNs（如GCN、GraphSAGE和GIN）进行比较，DHNs在局部传递性和太阳属性的任务中表现优异，F1和AUROC指标均高于其他模型。

**⚠️ 局限性**

限制在于对DHNs的某些静态分析问题的可判定性仍然存在开放性问题，且其最坏情况复杂度较高，实际高效算法的开发具有挑战性。

---

## 24. Complete first-order reasoning for functional programs

**arXiv ID:** 2605.23022 | [PDF](https://arxiv.org/pdf/2605.23022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 25. Extending Deep Event Visual Odometry with Sparse Point-Cloud Export

**arXiv ID:** 2605.22890 | [PDF](https://arxiv.org/pdf/2605.22890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 26. Finding Performance Issues in Database Systems by Exploiting Dormant Code Paths

**arXiv ID:** 2605.22992 | [PDF](https://arxiv.org/pdf/2605.22992v1)

**作者:** Jinsheng Ba `[一作]` (ETH Zurich), Zhendong Su `[通讯]` (ETH Zurich)

**通讯引用:** 14596 | [OpenAlex ID](https://openalex.org/A5077610917)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过在四大主流DBMS源代码中翻转IF分支，构建参考版本以检测隐藏的性能缺陷。

**💡 创新点**

提出一种通用白盒方法——Branch Flip Analysis，能系统启用或禁用优化策略，从源代码层面暴露性能问题。

**🔧 技术方法**

利用代码插桩实现无重编译的分支翻转，配合差分测试验证功能性，并在同一工作负载下对比执行时间。

**📊 数据集**

使用行业标准的TPC‑H和TPC‑DS基准查询进行评测。

**📈 对比分析**

对比原始与翻转后版本的执行时间与估算成本，平均提升26.2倍、最高达374.9倍，发现21条此前未知的性能缺陷。

**⚠️ 局限性**

仅针对IF分支导致的优化策略，可能忽略其他控制流结构；功能性验证基于差分测试，存在潜在误报；评估仅在单一硬件配置下完成。

---

## 27. Seeing without Looking: Do Vision-Language Benchmarks Really Test Vision?

**arXiv ID:** 2605.22903 | [PDF](https://arxiv.org/pdf/2605.22903v1)

**作者:** Zixuan Lan `[一作]` (University of Chicago), Jiawei Zhou `[通讯]` (Stony Brook University)

**通讯引用:** 1961 | [OpenAlex ID](https://openalex.org/A5056519111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估 Vision‑Language 模型在不同视觉干预下的表现，探讨 benchmark 准确率是否真正反映细粒度视觉理解。

**💡 创新点**

提出多层级诊断框架（全局与局部遮挡、token 丢弃、问句重构等），并通过行为与内部表示双重分析揭示标准准确率对视觉输入的鲁棒性掩盖了模型的浅层视觉依赖。

**🔧 技术方法**

采用图像全局/局部干预、随机 token 丢弃、问题格式重构、决策边缘与 MRR 分析以及层级视觉 token 的余弦相似度、k‑means 空间聚类和有效秩等表征方法。

**📊 数据集**

使用 POPE、A‑OKVQA、MME、AMBER 等公开视觉‑语言基准数据集。

**📈 对比分析**

通过对比干预前后准确率、Yes‑Rate、未知率、决策边缘分布以及 MRR 等指标，发现即使移除 75% 以上图像 token 或遮挡重要实体，模型准确率仍保持在 90% 以上，说明传统准确率低估了模型对细粒度视觉证据的依赖。

**⚠️ 局限性**

局限在于仅评估了现有公开 VLM，未覆盖所有模型架构；缺少专门针对细粒度视觉证据的基准；干预方式可能无法完全消除对应视觉信息，导致结论需在更广泛的实验中进一步验证。

---

## 28. ObjectCache: Layerwise Object-Storage Retrieval for KV Cache Reuse

**arXiv ID:** 2605.22850 | [PDF](https://arxiv.org/pdf/2605.22850v1)

**作者:** Yu Zhu `[一作]` (ETH Zurich), Gustavo Alonso `[通讯]` (ETH Zurich)

**通讯引用:** 17430 | [OpenAlex ID](https://openalex.org/A5103144919)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ObjectCache，利用 S3 兼容对象存储作为 LLM KV 缓存的持久化后端，并通过协议与调度协同设计实现层级化、聚合化的 KV 加载。

**💡 创新点**

创新点在于在标准 S3 接口上扩展多对象聚合与按层交付的语义，并结合带宽感知调度，突破了 S3 对 KV 缓存实时复用的语义鸿沟。

**🔧 技术方法**

采用 RDMA over RoCE、NIXL 客户端、Ceph RGW 网关与 DAOS 存储后端，设计对象描述符、服务器端聚合、层级交付与 Stall‑opt 带宽调度。

**📊 数据集**

使用 Llama 3.1 8B 以及 Granite 3.3 8B、DeepSeek 等开源 LLM 对象作为基准。

**📈 对比分析**

与本地 DRAM 层级化加载相比，64K 上下文下仅 5.6% 延迟；4K 上下文下 56–75 ms 额外延迟；在共享带宽约束下，调度器比均分或按字节分配将 TTFT 降低 1.2–1.8 倍。

**⚠️ 局限性**

局限性包括仅支持固定大小、哈希地址 KV 块，无法处理压缩或不规则布局；原型规模有限，需进一步验证并发、错误恢复、批处理与路由等生产环境特性。

---

## 29. SciAtlas: A Large-Scale Knowledge Graph for Automated Scientific Research

**arXiv ID:** 2605.22878 | [PDF](https://arxiv.org/pdf/2605.22878v1)

**作者:** Shuofei Qiao `[一作]` (Zhejiang University), Huajun Chen `[通讯]` (Zhejiang University)

**通讯引用:** 8598 | [OpenAlex ID](https://openalex.org/A5102018239)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文介绍了一个大型多学科异构学术资源知识图谱，旨在解决当前学术检索工具在信息爆炸背景下的局限性，提供一个结构化的拓扑认知基础，以促进科学发现。

**💡 创新点**

创新点在于构建了一个包含4300万篇论文的知识图谱，并开发了一种神经符号检索算法，实现了从简单语义匹配到确定性关联发现的无缝过渡。

**🔧 技术方法**

使用了神经符号检索算法，结合了三路径协作回忆和图重排序技术，以实现深层次的拓扑推理。

**📊 数据集**

数据集来源于OpenAlex，整合了来自26个学科的超过4300万篇论文及其他异构实体。

**📈 对比分析**

与现有方法相比，性能显著提升，能够在2分钟内完成检索过程，提供高相关性结果并进行深入的拓扑推理，显著降低推理成本。

**⚠️ 局限性**

限制在于当前知识图谱主要通过Neo4j接口访问，用户需要编写查询，且更新主要依赖于手动执行的固定脚本，缺乏实时自动更新机制。

---

## 30. PIMbot: A Self-Adaptive Attack Framework for Adversarial Manipulation of Multi-Robot Reinforcement Learning

**arXiv ID:** 2605.23027 | [PDF](https://arxiv.org/pdf/2605.23027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 31. VideoOdyssey: A Benchmark for Ultra-Long-Context and Omni-Modal Video Understanding

**arXiv ID:** 2605.22907 | [PDF](https://arxiv.org/pdf/2605.22907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 32. Worse than Random: The Importance of a Baseline for Unsupervised Feature Selection

**arXiv ID:** 2605.22973 | [PDF](https://arxiv.org/pdf/2605.22973v1)

**作者:** Muhammad Rajabinasab `[一作]` (University of Southern Denmark), Arthur Zimek `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了无监督特征选择方法，并提出使用随机特征选择作为基准，对比多种主流方法在23个高维数据集上的性能与效率。

**💡 创新点**

创新点在于：①首次系统地将随机特征选择作为评估基准，揭示现有方法在多数情况下不超过甚至低于随机基准；②通过Z-score和FSDEM等多指标对比，量化各方法相对随机基准的优势或劣势；③强调基准缺失导致的评价偏差，呼吁研究者在新方法开发中加入随机基准。

**🔧 技术方法**

使用的技术包括：随机特征选择、传统方差/相关法、Laplacian Score、MCFS、SCFS、SOGFS、LLSRFS、VCDFS等无监督特征选择算法；下游任务用随机森林（5折交叉验证）进行分类评价，用k-means进行聚类评价；性能指标包括ACC、AUC、CLSACC、NMI、FSDEM和Z-score。

**📊 数据集**

采用23个来自scikit-feature仓库的高维数据集，特征数量从几百到二万多不等，实例数从几百到一万多不等，例如ALLAML、arcene、glioma、isolet、leukemia等。

**📈 对比分析**

通过对比实验发现：在大多数数据集上，随机基准在分类和聚类指标上往往优于或与现有方法相当；当选取特征比例极低（0.5%–10%）或比例较高（5%–100%）时，随机基准几乎始终位居前列；相对随机基准，许多先进方法的Z-score普遍为负，表明其表现不显著优于随机选择。

**⚠️ 局限性**

局限性包括：①实验仅覆盖23个公开数据集，未考虑不同领域或噪声水平的广泛情况；②仅对无监督特征选择方法进行评估，未探讨其在半监督或监督场景下的表现；③随机基准虽然低成本，但缺乏解释性，不能直接指导特征重要性分析；④方法的计算复杂度与性能之间的平衡仍需进一步研究。

---

## 33. FusionSense: Tri-Stage Near-Sensor Learning for Runtime-Adaptive Multimodal Edge Intelligence

**arXiv ID:** 2605.22868 | [PDF](https://arxiv.org/pdf/2605.22868v1)

**作者:** Sanggeon Yun `[一作]` (University of California), Mohsen Imani `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FusionSense，一个融合感知框架，利用服务器端融合模型生成滤除安全标签，训练轻量级近传感器分类器，并将其预测注入到边缘融合模型，实现多模态数据的保留/丢弃决策，从而在保持任务质量的前提下显著降低计算与通信开销。

**💡 创新点**

创新点在于三步训练流程：① 先在服务器端训练完整的多模态融合模型；② 通过对每个模态做“滤除安全”(FoS)标签，标记哪些模态在某些样本中是多余的；③ 在边缘侧压缩融合模型，并将近传感器的预测作为辅助信号，形成融合感知的预先决策层，实现跨模态依赖的预融合选择。

**🔧 技术方法**

采用轻量级CNN（MobileNetV3）做近传感器分类器，服务器端采用RegNet 400MF实现late fusion，利用FoS标签生成、边缘融合模型压缩、近传感器预测注入，以及Edge TPU量化加速等技术。

**📊 数据集**

使用 SynDrone 数据集（RGB+深度图像），进行城市场景车辆多标签分类实验。

**📈 对比分析**

与传统全量传输、压缩传输、单模态过滤等基线方法对比，FusionSense 在 FoI 1% 时能耗降低 33×，10% 时降低 11×；在 30% 数据削减时质量损失仅为 92.3% 的减少，能耗提升约 1.5×优于最优基线。

**⚠️ 局限性**

局限性包括：在模态数量增多时参数增长仍然显著，模型主要在 RGB+深度两模态实验，缺乏对更多模态的验证；目前仍为离线训练，缺乏实时在线自适应机制；对动态能耗与时延约束的适应性尚需进一步研究。

---

## 34. Botnet Detection on CTU-13 Using Lightweight Machine Learning Models

**arXiv ID:** 2605.23004 | [PDF](https://arxiv.org/pdf/2605.23004v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 35. A Reproducible Universal Dependencies-Style Pipeline for Katharevousa Greek Parliamentary Text

**arXiv ID:** 2605.22978 | [PDF](https://arxiv.org/pdf/2605.22978v1)

**作者:** George Mikros `[一作]` (Hamad Bin Khalifa University), Fotios Fitsilis `[通讯]` (Universidad Austral)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了一个可重复的依赖解析管道，用于处理Katharevousa希腊语的议会文本，并将其作为开放获取的伴随库发布。

**💡 创新点**

创新点在于提出了一种将嘈杂的OCR衍生材料转换为固定、可检查的参考快照的研究工作流程，并在共享协议下评估多种模型家族。

**🔧 技术方法**

使用了OCR感知重建、LLM辅助注释、自动验证、确定性CoNLL-U快照、固定分割评估和模型家族比较等技术。

**📊 数据集**

使用的数据集包含1,697个句子，其中1,357个用于训练，340个用于测试，数据来自1976-1977年的希腊议会问题。

**📈 对比分析**

与现成的希腊语和古希腊语解析器、特征基础解析器、mBERT、XLM-R和自定义Stanza训练进行了比较，XLM-R模型在LAS上达到了0.5162，优于最强外部基线0.0980。

**⚠️ 局限性**

限制在于当前参考集较小，仅包含1,697个句子，且标签是自动验证的，未经过完全专家审定，可能影响绝对语言正确性的声明。

---

## 36. The Deterministic Horizon: Impossibility Results as Design Specifications for Trustworthy AI Systems

**arXiv ID:** 2605.23024 | [PDF](https://arxiv.org/pdf/2605.23024v1)

**作者:** Dongxin Guo `[一作]` `[通讯]` (University of Hong Kong), Dongxin Guo (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一个统一框架，证明了由架构决定的准确性上限：在关键推理深度之后，任何训练都无法提高模型的表现。

**💡 创新点**

创新点在于将不可能性结果转化为设计规则，提出了确定性地平线（Deterministic Horizon）概念，明确了在特定深度下推理的准确性会急剧下降。

**🔧 技术方法**

使用了信息论和复杂性理论的技术，结合了对多种变换器架构的实证分析。

**📊 数据集**

使用了12种不同的变换器架构进行验证，具体包括GPT-2、Llama-2/3等。

**📈 对比分析**

与现有方法比较，论文中的模型在推理深度超过确定性地平线后表现显著下降，且在多文件状态跟踪的基准测试中，工具增强系统的表现优于前沿推理模型，成本约为每个任务的三分之一。

**⚠️ 局限性**

限制在于该框架的适用性可能受到特定架构和任务的限制，且在实际应用中可能需要更多的实证验证。

---

## 37. RAG4Outcome: A Retrieval-Augmented Multimodal Framework for Prognostic Prediction in Chronic Osteomyelitis

**arXiv ID:** 2605.22833 | [PDF](https://arxiv.org/pdf/2605.22833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 38. Latent Cache Flow: Model-to-Model Communication Without Text

**arXiv ID:** 2605.22863 | [PDF](https://arxiv.org/pdf/2605.22863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 39. PACE: Two-Timescale Self-Evolution for Small Language Model Agents

**arXiv ID:** 2605.23019 | [PDF](https://arxiv.org/pdf/2605.23019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 40. A mathematical theory of balancing relational generalization and memorization

**arXiv ID:** 2605.22972 | [PDF](https://arxiv.org/pdf/2605.22972v1)

**作者:** Luke Cheng `[一作]` (Columbia University), Samuel Lippl `[通讯]` (Columbia University)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5030803845)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的任务——带异常的传递推理（TI with Exceptions），并对核岭回归模型在此任务下如何平衡一般规则学习与异常记忆进行了解析理论分析，随后在预训练语言模型（如 Llama、Qwen）上进行微调实验，验证理论预测。

**💡 创新点**

创新点：① 设计了能够同时考察规则学习与异常记忆的实验范式；② 推导出核模型在不同表征几何和 L₂ 正则化下的精确行为；③ 发现 L₂ 正则化在跨区推理中可能产生系统性错误；④ 将上述理论应用于预训练语言模型，揭示其在异常推理任务中的强弱点。

**🔧 技术方法**

使用技术：核岭回归、随机特征模型、可交换核、解析解、线性探测（Linear Probe）、LoRA 微调、全微调（Fine‑tune）、L₂ 正则化、预训练语言模型（Llama、Qwen）微调。

**📊 数据集**

数据集：合成层级数据（包含 n 个元素、p、q 位置的异常环）；运动队伍名字（真实队伍和虚构队伍）对比任务；德州扑克手牌对比任务。

**📈 对比分析**

比较方法：在训练集（异常环）和测试集（跨区、内区、记忆）上评估准确率，比较全微调、LoRA、线性探测三种微调策略。实验结果显示：在无异常 TI 任务中模型表现良好；在带异常任务中，模型在内区推理上相对稳健，跨区推理则因 L₂ 正则化而出现系统性错误；全微调优于 LoRA，线性探测最差。

**⚠️ 局限性**

局限性：仅考虑核模型与可交换表征，未探讨特征学习的深度网络；实验仅涵盖单一异常结构，未覆盖更广泛的复杂关系；对真实世界大规模数据的泛化能力未知；未尝试多任务或元学习策略来进一步缓解异常记忆与规则推理的冲突。

---

## 41. On Reed-Muller subcodes, Grassmannian partitions and sum-free functions

**arXiv ID:** 2605.22958 | [PDF](https://arxiv.org/pdf/2605.22958v1)

**作者:** Philipp Heering `[一作]` (Justus-Liebig-Universität Gießen), Vladislav Taranchuk `[通讯]` (Ghent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了k阶无和函数的存在性与Reed-Muller码的子码之间的等价关系，证明了在特定条件下，存在非退化的k阶无和函数与Reed-Muller码的子码之间的对应关系。

**💡 创新点**

创新点在于建立了k阶无和函数与Reed-Muller码子码之间的等价性，提供了新的构造方法，并且提出了k阶无和函数的存在性的新必要条件和第一个非平凡的下界。

**🔧 技术方法**

使用了代数编码理论中的Reed-Muller码和k阶无和函数的概念，结合了线性代数和组合数学的技术。

**📊 数据集**

使用了Reed-Muller码RM(n-k,n)的相关性质，特别是其最小距离和维度的特性。

**📈 对比分析**

通过与现有的k阶无和函数进行比较，展示了新构造的子码具有更高的最小距离，且在特定条件下，能够有效地对Grassmann图进行着色，提供了更强的上界。

**⚠️ 局限性**

限制在于目前仅对2≤k≤n-2的情况进行了研究，且对于k≥3的情况，k阶无和函数的存在性仍然是一个开放问题。

---

## 42. Evaluating Large Language Models in a Complex Hidden Role Game

**arXiv ID:** 2605.22826 | [PDF](https://arxiv.org/pdf/2605.22826v1)

**作者:** Niklas Bauer `[一作]` `[通讯]` (University of Göttingen), Niklas Bauer (University of Göttingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLMs）在社交推理游戏《秘密希特勒》中的推理、说服和欺骗能力，并引入了一个开源框架和新颖的度量标准来评估其表现。

**💡 创新点**

创新点在于提出了角色识别准确率、欺骗保持率和游戏状态影响率等新指标，以量化LLMs在复杂社交环境中的表现，并与基于规则的算法和人类玩家进行基准比较。

**🔧 技术方法**

使用了开源框架进行实验，结合了多种模型（如Llama 3.1 70B）和技术（如Chain-of-Thought提示和内部记忆），但未进行模型的微调。

**📊 数据集**

使用的数据集包括来自secrethitler.io的约1000场游戏，主要是由经验丰富的玩家进行的七人比赛，经过清洗和预处理以确保数据的相关性。

**📈 对比分析**

与基于规则的代理相比，LLMs在决策和沟通策略上表现较差，尤其在扮演法西斯角色时，表现出显著的能力差距。人类玩家的投票一致性高达86.7%，而LLMs的准确率仅为59.7%。

**⚠️ 局限性**

限制在于当前的LLMs在复杂的多轮操控中仍然表现不佳，尤其是在欺骗和说服任务中，且在法西斯角色中经常无法维持欺骗，导致游戏时间显著缩短。

---

## 43. Which Superconducting Qubit Model is Good Enough? From Effective Two-Level to Circuit-Based Hamiltonians for Pulse-Level Simulation

**arXiv ID:** 2605.23034 | [PDF](https://arxiv.org/pdf/2605.23034v1)

**作者:** Frej Larssen `[一作]` (KTH Royal Institute of Technology), Stefano Markidis `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 4915 | [OpenAlex ID](https://openalex.org/A5085178088)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对一个带固定总线耦合的 flux‑可调双量子比特超导系统，系统性地比较了三种不同层次的哈密顿量模型（有效两级模型、三模 Duffing 模型、基于电路的多能级模型）在脉冲级模拟中的表现。

**💡 创新点**

创新点在于：①提出分层抽象框架，说明在不同模拟目标下可选择的最合适模型；②构建统一的基准测试套件，涵盖静态谱、有效两量子比特参数、单比特 R_X 动态、CZ 门相位积累与泄漏；③量化三模型在相位、泄漏与计算成本上的差异，为大规模脉冲级仿真提供实用指导。

**🔧 技术方法**

主要技术包括：哈密顿量分解与参数提取、低阶多能级 Duffing 近似、完整电路模型的 charge‑basis 计算、时间演化的矩阵指数方法、基准测试的谱与动态指标、以及计算耗时的量化。

**📊 数据集**

使用的数据集为一套与实际实验设备匹配的物理参数（EJ、EC、耦合强度等），并在不同磁通偏置点进行的多次仿真，形成统一的基准曲线和指标集合。

**📈 对比分析**

比较方法：对每个模型在同一套磁通/驱动脉冲下计算静态能谱、提取 J(φ)、ζ(φ)、单比特 R_X 的保真度与泄漏、CZ 门的相位累积及泄漏通道，随后统计运行时构建与传播时间。结果显示：Duffing 模型在绝大多数情况下与电路模型几乎一致，且相对高效；有效两级模型在静态指标上足够，但在泄漏和多级耦合动态上明显不足；电路模型在泄漏细节上最完整，但计算成本最高。

**⚠️ 局限性**

局限性：仅验证了双量子比特、固定总线结构；只考虑了 R_X 与基于磁通的 CZ 脉冲；未对优化脉冲、非固定耦合或更大规模量子芯片进行测试；多能级模型的参数依赖于预先构建的电路基准，可能不适用于所有实验平台。

---

## 44. On the Reliability of Code Comprehension Proxies

**arXiv ID:** 2605.23008 | [PDF](https://arxiv.org/pdf/2605.23008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 45. Memorization Dynamics of Fill-in-the-Middle Pretraining

**arXiv ID:** 2605.22981 | [PDF](https://arxiv.org/pdf/2605.22981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 46. Improved Vision-to-Chart Buoy Association with Learned World-to-Image Projection

**arXiv ID:** 2605.22942 | [PDF](https://arxiv.org/pdf/2605.22942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 47. BOHM: Zero-Cost Hierarchical Attribution for Compound AI Systems

**arXiv ID:** 2605.22866 | [PDF](https://arxiv.org/pdf/2605.22866v1)

**作者:** Joss Armstrong `[一作]` `[通讯]` (Ericsson Research), Joss Armstrong (Ericsson Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了BOHM方法，从现有的分层路由权重直接生成多层次归因树，实现零成本的组件级归因；

**💡 创新点**

创新点在于利用路由权重而非集合消融得到层级归因，提供多分辨率归因并可与Shapley方法对比；

**🔧 技术方法**

使用多层次路由权重的路径乘积、权重归一化及贝叶斯或多臂老虎机的自适应更新算法；

**📊 数据集**

在18款LLM（LiveCodeBench 880题）、美国人口普查（475个PUMA）以及5个代理器与7个基准的多驱动实验中验证；

**📈 对比分析**

与SHAP（需要大量集合评估）比较，BOHM在同等评估预算下取得近似的Kendall τ（如LLM实验τ≈0.928 vs SHAP 0.980），且在未缓存部署环境下成本显著降低；

**⚠️ 局限性**

局限性包括仅适用于具有自适应路由的层级系统，对质量差距小或权重尚未收敛时效果差，对非平衡层级设计敏感，并不满足Shapley的可加性公理，且仅反映部署时的信任而非最优贡献。

---

## 48. Energy per Successful Goal: Goal-Level Energy Accounting for Agentic AI Systems

**arXiv ID:** 2605.22883 | [PDF](https://arxiv.org/pdf/2605.22883v1)

**作者:** Deepak Panigrahy `[一作]` (Independent Researcher), Aakash Tyagi `[通讯]` (Texas A M University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种新的AI能量测量框架，称为成功目标能量（Energy per Successful Goal），用于评估代理AI系统的能量消耗。

**💡 创新点**

创新点在于将能量测量单位从每次推理的能量转变为每个成功目标的能量，能够更准确地反映多步骤工作流的能量消耗。

**🔧 技术方法**

使用了跨层测量框架，包括时间边界模型、五层观察管道和可重复性协议，确保每次测量都与硬件、环境和运行时配置绑定。

**📊 数据集**

在五个推理任务家族（事实问答、科学问答、算术推理、多步骤推理和逻辑推理）和三个工具增强任务家族上进行了评估，结合硬件级能量分析。

**📈 对比分析**

与线性基线相比，代理工作流在成功目标上消耗的平均能量高出×倍，主要是由于多步骤的结构和重试机制，而不是推理计算的增加。

**⚠️ 局限性**

限制在于成功定义为二元，未能捕捉输出质量的渐进变化；测量仅限于本地CPU，未直接测量GPU和远程服务器的能量；需要匹配的线性基线进行比较。

---

## 49. Query-Adaptive Semantic Chunking for Retrieval-Augmented Generation: A Dynamic Strategy with Contextual Window Expansion

**arXiv ID:** 2605.22834 | [PDF](https://arxiv.org/pdf/2605.22834v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 50. Agentic-VLA: Efficient Online Adaptation for Vision-Language-Action Models

**arXiv ID:** 2605.22896 | [PDF](https://arxiv.org/pdf/2605.22896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 51. Uncovering the Latent Potential of Deep Intermediate Representations

**arXiv ID:** 2605.23033 | [PDF](https://arxiv.org/pdf/2605.23033v1)

**作者:** Arnesh Batra `[一作]` (Indraprastha Institute of Information Technology Delhi), Anubha Gupta `[通讯]` (Indraprastha Institute of Information Technology Delhi)

**通讯引用:** 5231 | [OpenAlex ID](https://openalex.org/A5057412604)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的层选择方法LOES（Layer-wise Optimal Embedding Selection），用于从预训练的深度模型中选择任务相关的中间层，以提高迁移学习的效果。

**💡 创新点**

创新点在于通过几何和谱方法识别任务区分结构的层，并引入几何正则化损失（GeoReg）来稳定表示几何，改善微调过程中的表现。

**🔧 技术方法**

使用了谱优化方法来选择层，并结合几何正则化损失来保持表示的几何结构。

**📊 数据集**

在多种架构、深度、模态和数据集上进行了实验，包括图像分类、文本分类和多模态任务，具体数据集包括ImageNet、CUB-200、Stanford Cars等。

**📈 对比分析**

与标准的最后一层迁移学习方法相比，LOES在多个任务上表现出一致的性能提升，尤其是在模型深度增加时，性能提升更为显著。

**⚠️ 局限性**

限制在于LOES的计算开销相对较小，但在某些情况下可能需要额外的校准样本来进行层选择。

---

## 52. World Machine: Towards Generative World Modeling for Time-Series

**arXiv ID:** 2605.23025 | [PDF](https://arxiv.org/pdf/2605.23025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 53. Human-Centered Learning Mechanics: A Dynamical Framework for Entropy-Regulated Representation Learning

**arXiv ID:** 2605.22940 | [PDF](https://arxiv.org/pdf/2605.22940v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 54. Multi-Dimensional Matching in Market Design

**arXiv ID:** 2605.22865 | [PDF](https://arxiv.org/pdf/2605.22865v1)

**作者:** Irene Aldridge `[一作]` `[通讯]`, Irene Aldridge

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于特征报告、SVD降维的多维匹配机制，能够在 O(N log N) 时间内实现近似 Nash 社会福利最大化；

**💡 创新点**

创新点在于将多维特征投影到主方向进行排序，既实现了极致的计算效率，又满足分布式真诚性与对称性，并且建立了 NSW 与几何分布鲁棒优化的数学联系；

**🔧 技术方法**

核心技术包括 Singular Value Decomposition (SVD) 进行降维、基于投影值的排序匹配、Kolmogorov–Smirnov 距离衡量分布式真诚性，以及利用 Eckart–Young 定理进行误差分析；

**📊 数据集**

实验主要使用合成数据集（如 I=100, J=20, X=5 的随机生成特征与偏好），以及一个 3 代理 3 物品的教学案例，未使用公开的真实匹配数据；

**📈 对比分析**

通过与随机分配、序列优先、配置线性规划等基线方法比较，SVD 机制在 99% 以内逼近最优 Nash 社会福利，并且计算时间从几千毫秒下降到约 10–15 毫秒，表现出显著的速度与近似质量优势；

**⚠️ 局限性**

局限性在于仅适用于加性可分的效用函数，要求特征矩阵具有低有效维度（σ1≫σ2）；当特征不相关、存在补充效用或非线性交互时，机制性能会显著下降，且对少数偏好与主方向几乎正交的代理可能出现个人理性违规。

---

## 55. A Proactive Multi-Agent Dialogue Framework for Assessing Social Language Disorder Traits in Autism

**arXiv ID:** 2605.22993 | [PDF](https://arxiv.org/pdf/2605.22993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 56. From Simulation to Discovery: AI Enabled Probabilistic Emulation of Mechanistic Crop Systems

**arXiv ID:** 2605.22848 | [PDF](https://arxiv.org/pdf/2605.22848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 57. Test-Time Training Undermines Safety Guardrails

**arXiv ID:** 2605.22984 | [PDF](https://arxiv.org/pdf/2605.22984v1)

**作者:** Simone Antonelli `[一作]` (CISPA Helmholtz Center for Information Security), Aleksandar Bojchevski `[通讯]` (University of Cologne)

**通讯引用:** 3095 | [OpenAlex ID](https://openalex.org/A5058887708)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作研究了测试时训练（TTT）对大型语言模型安全性的影响，并系统提出并验证了三种TTT攻击威胁模型（自监督、少样本、生成阶段），展示了攻击成功率高达近100%，同时提出了基于困惑度的轻量化检测方案。

**💡 创新点**

首次将TTT与模型安全对齐问题结合，定义并量化三种攻击场景，证明TTT可显著削弱安全对齐，并提出有效性检测与困惑度偏移检测的创新防御方法。

**🔧 技术方法**

采用梯度更新的TTT（LoRA、全微调、RL）进行攻击，结合符号与LLM判别器进行有效性检测，并使用困惑度偏移量做提供商侧检测。

**📊 数据集**

主要使用AdvBench、JailbreakBench、Transluce CBRN等恶意行为基准，同时用GSM8K等干净对照集进行对比。

**📈 对比分析**

通过ASR与ASR@10指标与基线模型对比，实验表明在LoRA少样本与生成阶段攻击中，多模型ASR@10可达95%~100%，自监督攻击提升4%~17%；检测器实现TPR 100%、FPR ≤2%。

**⚠️ 局限性**

未涵盖自适应攻击、缺乏动态对齐方法；检测器依赖隐藏恶意holdout集，且对大规模全微调与多步TTT的评估有限。

---

## 58. PrefBench: Evaluating Zero-Shot LLM Agents in Hidden-Preference Personalized Pricing Negotiations

**arXiv ID:** 2605.22855 | [PDF](https://arxiv.org/pdf/2605.22855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 59. Brain-LLM Alignment Tracks Training Data, Not Typology

**arXiv ID:** 2605.23032 | [PDF](https://arxiv.org/pdf/2605.23032v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22492 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究探讨了大脑语言网络与大型语言模型（LLM）之间的对齐是否在不同语言间普遍存在，使用了来自英语、中文和法语的fMRI数据进行比较。

**💡 创新点**

创新点在于首次定量展示了正式的类型距离与大脑编码性能之间的关系，并通过对比训练语言主导性与类型距离，揭示了训练数据组成对对齐模式的影响。

**🔧 技术方法**

使用了fMRI技术和多种大型语言模型（如Baichuan2-7B和LLaMA-2-7B）进行对比分析。

**📊 数据集**

使用了Le Petit Prince多语言fMRI语料库，包含112名参与者的英语、中文和法语数据。

**📈 对比分析**

通过与多种模型的对比，发现Baichuan2-7B在中文上的对齐性能最佳，而在英语和法语上的性能显著较低，表明训练语言主导性是主要驱动因素。

**⚠️ 局限性**

限制在于样本语言数量较少，仅有三种语言，可能影响统计功效，且不同语言在不同机构收集，可能存在语言与地点的混淆。

---

## 60. Can AI Guess What You Know? Performance Comparison of Large Language Models for Human Domain Knowledge Estimation From Communication Logs

**arXiv ID:** 2605.22971 | [PDF](https://arxiv.org/pdf/2605.22971v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 61. An AI-Driven Framework for Energy-Efficient Environmental Monitoring in Smart Cities Using Edge Intelligence

**arXiv ID:** 2605.22824 | [PDF](https://arxiv.org/pdf/2605.22824v1)

**作者:** Yichen Liu `[一作]` (Independent Researcher), Shiqi Yang `[通讯]` (Independent Researcher)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于AI的框架，用于在智能城市中实现节能的环境监测，利用边缘智能动态激活传感器。

**💡 创新点**

创新点在于结合TinyML和上下文感知的自适应决策机制，动态选择最具信息量的传感器，从而减少冗余感知和延长传感器寿命。

**🔧 技术方法**

使用了TinyML技术和边缘智能架构，进行实时环境监测和传感器激活决策。

**📊 数据集**

使用了基于英国空气质量重分析（AQREAN）数据集的城市规模模拟，包含多种环境传感器数据。

**📈 对比分析**

与静态、周期性和基于UCB的自适应感知策略进行比较，结果显示该框架在能耗、监测覆盖率和传感器寿命方面均优于其他方法。

**⚠️ 局限性**

局限性包括假设无线链接稳定，未考虑传感器类型的不同故障率，以及手动更换电池的成本未纳入模型。

---

## 62. Monte Cimone v3: Where RISC-V Stands in High-Performance Computing

**arXiv ID:** 2605.22831 | [PDF](https://arxiv.org/pdf/2605.22831v1)

**作者:** Emanuele Venieri `[一作]` (DEI University of Bologna), Andrea Bartolini `[通讯]` (DEI University of Bologna)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Monte Cimone v3项目中，对配备Sophgo SG2044 RISC-V处理器的两台计算节点进行单节点STREAM与HPL基准测试，并将结果与Intel Xeon Sapphire Rapids与NVIDIA Grace CPU Superchip系统进行对比。

**💡 创新点**

首次证明商业化RISC-V芯片在高性能计算（HPC）领域能够接近主流x86/Arm平台，关键创新点在于采用RVV 1.0向量指令和重新设计的32通道LPDDR5X内存子系统，实现高内存带宽与良好可扩展性。

**🔧 技术方法**

使用GCC 14.2编译器（指定RVV目标）、OpenBLAS 0.3.29库、OpenMP线程定位、SLURM调度、SPACK模块化软件栈以及IPMI进行功耗测量。

**📊 数据集**

基准数据集为标准STREAM Triad和HPL（高性能LINPACK）测试，未使用外部真实数据集。

**📈 对比分析**

通过单节点功耗与性能比较，SG2044节点在STREAM上实现2.6×比MCv2、100×比MCv1的内存带宽；在HPL上每核性能超过SG2042两倍、MCv1提升139×；与Intel与NVIDIA平台比较时，归一化至向量宽度和时钟频率后，MCv3在64核时的性能差距缩小至1.8–2.6倍，16核（峰值效率）时与Grace、Sapphire Rapids相差1.1–2.2倍；功率效率为GFLOPs/W 3.08，约占Intel 3.86（80%）和NVIDIA 4.55（68%）。

**⚠️ 局限性**

限制主要在于向量单元宽度仅为128位（低于Intel 512位、Arm 256/128位），缺乏厂商优化的BLAS库；内存子系统在大规模核心扩展时仍有瓶颈，功耗效率虽已提升但仍低于顶级平台。

---

## 63. Intercloud: Eventual Consistency for Decentralised Economies via Chilling-Effect Consensus

**arXiv ID:** 2605.22830 | [PDF](https://arxiv.org/pdf/2605.22830v1)

**作者:** Gregory Magarshak `[一作]` `[通讯]` (IENYC), Gregory Magarshak (IENYC)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出Intercloud，一种利用Watcher群体仅观察哈希来实现可扩展最终一致性和隐私保护的分布式经济网络。

**💡 创新点**

创新点包括chilling‑effect共识替代全局投票、脉冲去重机制、按价值平方根自适应分配安全资源，以及安全权重与经济价值的点乘自动校准。

**🔧 技术方法**

核心技术为Magarshak Machine、可验证随机函数（VRF）群体分配、Proof of Corruption、零知识证明扩展与两层流（币层与内容层）架构。

**📊 数据集**

未使用传统数据集，而是基于理论模型和假设网络拓扑与节点规模参数进行定理证明与模拟验证。

**📈 对比分析**

与比特币、以太坊、PBFT 等系统比较，证明单笔交易验证成本与网络规模无关，双花检测概率满足 e⁻ˢ，安全成本随价值的平方根增长，epoch 内即可达成最终一致性。

**⚠️ 局限性**

局限性在于需要单一正确Executor、依赖可观测的VRF种子、对交换率Oracle的信任、对Executor故障与拜占庭攻击的处理有限，以及网络层隐私保护仍不完整。

---

## 64. LLM Code Smells: A Taxonomy and Detection Approach

**arXiv ID:** 2605.22976 | [PDF](https://arxiv.org/pdf/2605.22976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 65. A Survey of Text and Speech Resources for Hausa and Fongbe: Availability, Quality, and Gaps for NLP Development

**arXiv ID:** 2605.22828 | [PDF](https://arxiv.org/pdf/2605.22828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 66. RADAR: Relative Angular Divergence Across Representations

**arXiv ID:** 2605.23028 | [PDF](https://arxiv.org/pdf/2605.23028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 67. KPI2KVI: A Multi Agent Workflow for Calculating Key Value Indicators from Service Descriptions

**arXiv ID:** 2605.22825 | [PDF](https://arxiv.org/pdf/2605.22825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 68. RMA: an Agentic System for Research-Level Mathematical Problems

**arXiv ID:** 2605.22875 | [PDF](https://arxiv.org/pdf/2605.22875v1)

**作者:** Zelin Zhao `[一作]` (Georgia Institute of Technology), Yongxin Chen `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7976 | [OpenAlex ID](https://openalex.org/A5066940107)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了研究数学代理（RMA），这是一个用于自动推理研究级数学问题的代理框架。与以往集中于竞赛数学或形式定理证明的研究不同，RMA针对需要长时间推理、文献基础和迭代证明改进的研究级数学问题。

**💡 创新点**

创新点在于将研究级证明解决方案分解为多个专门模块，包括问题分析、文献搜索与理解、公平比较、知识库构建和证明验证，并通过共享结构化记忆协调多个角色的代理进行协作。

**🔧 技术方法**

使用了多代理、多轮交互的系统设计，结合了问题分析、文献搜索、知识库和证明命令模块等多个模块，支持迭代推理和自动验证。

**📊 数据集**

在First Proof基准上进行评估，该基准包含十个由专家数学家贡献的研究级问题，涵盖不同领域。

**📈 对比分析**

与强基线（如GPT-5.2R和Aletheia）进行比较，RMA在解决八个问题上表现优于这些基线，生成的证明在逻辑上更为严谨且可读性更高。全面的消融研究表明，性能提升源于结构化推理模块的交互、迭代改进和基于验证者的反馈，而非单一组件。

**⚠️ 局限性**

限制在于当前框架仍需依赖于专家评估，且在处理某些特定类型的数学问题时可能存在局限性。

---

## 69. Building a privacy-preserving Federated Recommender system for mobile devices

**arXiv ID:** 2605.22924 | [PDF](https://arxiv.org/pdf/2605.22924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 70. FIRMA: FIbonacci Ring Model Aggregation for Privacy-preserving Federated Learning

**arXiv ID:** 2605.22898 | [PDF](https://arxiv.org/pdf/2605.22898v1)

**作者:** Rachid Hedjam `[一作]` (Bishop's University), Rachid Hedjam `[通讯]` (Bishop's University)

**通讯引用:** 1197 | [OpenAlex ID](https://openalex.org/A5053495468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 FIRMA（FIbonacci Ring Model Aggregation）三种无服务器、私有分类头、环形拓扑的联邦学习协议，并通过实验验证其在多种非 IID 场景下的有效性。

**💡 创新点**

核心创新包括：
• 以斐波那契数列导出的无参数异向邻居权重；
• 通过准确率门控动态抑制低质量邻居的影响；
• 2‑opt 排序最大化相邻客户端的类多样性；
• K_g 通信轮次实现全局覆盖；
• cosine 退火自我保留系数与持续 Adam 优化器；
• 完全私有的分类头，消除中心化梯度泄露风险。

**🔧 技术方法**

技术实现：
• 环形 gossip 通信；
• 斐波那契权重加权；
• 两阶段（head / extractor）Adam 本地训练；
• 误差门控权重插值；
• 2‑opt 客户端顺序优化；
• 多轮 K_g gossip 与自适应保留；
• Gini 指标用于公平性评估；
• 经验式收敛分析与通信成本对比。

**📊 数据集**

使用四个基准数据集：CIFAR‑10、Fashion‑MNIST、MNIST‑60k 与 MNIST‑1797，覆盖不同视觉复杂度与数据量。

**📈 对比分析**

与 FedAvg、FedRep、RDFL、Ring‑FL 等现有方法在 168 组配置（4 数据集 × 7 异构模式 × 6 方法）下进行系统对比。FIRMA++ 在绝大多数 label‑skew 与 Dirichlet 强异构场景下均为最优或次优；在 IID 条件下仍能保持接近 FedAvg 的准确率；相较中心化方法通信成本降低 3‑5 倍，且收敛曲线更稳定，Gini 公平性也优于多数基线。

**⚠️ 局限性**

限制：
• 仅在 MLP 架构上验证，CNN/Transformer 需要进一步测试；
• 依赖静态数据分布，概念漂移时需重新计算 2‑opt 排序；
• 规模仅测试 N=5、10，未验证更大规模的可扩展性；
• 对最差案例公平性（如最低客户端准确率）未单独报告；
• 仅提供经验安全性论证，未给出完整的差分隐私或安全证明。

---

## 71. From Residuals to Reasons: LLM-Guided Mechanism Inference from Tabular Data

**arXiv ID:** 2605.22897 | [PDF](https://arxiv.org/pdf/2605.22897v1)

**作者:** Mohammad R. Rezaei `[一作]` (University of Toronto), Rahul G. Krishnan `[通讯]` (University of Toronto)

**通讯引用:** 2503 | [OpenAlex ID](https://openalex.org/A5073514348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的框架Multi-Agent Residual In-Context Learning (MARICL)，通过分析基础模型的失败，生成结构化假设并迭代优化修正项，从而提高预测准确性和可解释性。

**💡 创新点**

MARICL的创新点在于结合了多代理的残差分析和文本梯度优化，能够生成可执行的修正公式，并通过迭代过程不断改进这些公式，使其更具解释性和准确性。

**🔧 技术方法**

使用了多代理框架和文本梯度优化技术，结合了基础模型的残差分析，生成结构化的假设和可执行的修正项。

**📊 数据集**

在多个数据集上进行了测试，包括科学、医学、生物经济和合成数据集，具体数据集包括Cell-Free Protein Production、Enzyme Activity、Diabetes Progression等。

**📈 对比分析**

与基础模型（如线性回归和XGBoost）进行比较，MARICL在所有数据集上均表现出一致的性能提升，尤其是在基础模型表现较弱的情况下，提升幅度最大。

**⚠️ 局限性**

MARICL的局限性在于当基础模型已经捕捉到主要非线性或在高维噪声下，其增益较小。此外，修正公式可能捕捉到相关性而非因果关系，且公式的适用范围限于其学习的领域。

---

## 72. How Far Will They Go? Red-Teaming Online Influence with Large Language Models

**arXiv ID:** 2605.22880 | [PDF](https://arxiv.org/pdf/2605.22880v1)

**作者:** Daniel C. Ruiz `[一作]` (Information Sciences Institute), Luca Luceri `[通讯]` (Information Sciences Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种实证红队框架，用于测量大型语言模型（LLM）的Overton窗口（OW），即模型在争议话题上能够可靠表达的政治观点范围，并量化简单自然语言越狱技术如何扩展该范围。

**💡 创新点**

创新点在于引入了一个框架来量化LLM的政治表达能力，并通过简单的越狱技术评估其在社交媒体内容生成中的影响，强调了开源LLM在隐私意识恶意行为者中的重要性。

**🔧 技术方法**

使用了简单的自然语言越狱技术，结合了多种人类可读的提示技术来评估模型的政治表达能力。

**📊 数据集**

评估了超过30个开源LLM，涵盖10个模型家族和五个国家的来源，使用了手工制作的90个政治立场声明作为数据集。

**📈 对比分析**

通过与基线提示的比较，发现越狱技术对模型的OW有显著影响，Few-Shot技术是唯一一致有效的增强器，而其他一些技术则会降低模型的合规性。整体性能显示，开源LLM在左倾内容生成上更为积极。

**⚠️ 局限性**

限制在于本研究仅评估了指令调优的开源LLM，未涵盖专有、仅推理或未审查模型的行为。此外，使用的意见语料库是手动策划的，可能无法反映现实世界政治话语的复杂性和多样性。

---

## 73. Whose Good, Whose Place? The Moral Geography of Agentic AI for Social Good

**arXiv ID:** 2605.22995 | [PDF](https://arxiv.org/pdf/2605.22995v1)

**作者:** Poli Nemkova `[一作]` (University of North Texas), Jaedon Charles `[通讯]` (Florida International University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2015年至2026年间112篇关于社会公益的代理AI论文进行了结构化调查，分析了这些论文在地理上下文、SDG对齐、代理架构等方面的分布情况。

**💡 创新点**

发现了道德地理不对称性，即在高风险的社会政策和制度SDG领域，缺乏地理上下文的情况更为严重，提出了五个问责缺口，并建议建立最低报告标准以增强代理AI的责任性。

**🔧 技术方法**

使用了双LLM注释程序（GPT-4o和Llama-3.3-70B）进行论文编码，并进行了分层的人类验证。

**📊 数据集**

分析了2015年至2026年间的112篇论文，未使用特定数据集，而是对现有文献进行了系统审查。

**📈 对比分析**

与其他方法比较时，发现73%的论文未指定地理上下文，尤其是在社会政策和制度SDG领域，只有13%的论文提供了地理信息，且仅25%的论文报告了任何现实世界的部署或小规模测试。

**⚠️ 局限性**

局限性包括可能未涵盖所有相关工作，SDG编码的解释性，依赖于论文标题和摘要的编码方法，以及未直接测量社会影响等。

---

## 74. WeCon: An Efficient Weight-Conditioned Neural Solver for Multi-Objective Combinatorial Optimization Problems

**arXiv ID:** 2605.22876 | [PDF](https://arxiv.org/pdf/2605.22876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 75. Reading Calibrated Uncertainty from Language Model Trajectories

**arXiv ID:** 2605.22864 | [PDF](https://arxiv.org/pdf/2605.22864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 76. Mathematical Foundations for Peer-to-Peer Lattice Computation

**arXiv ID:** 2605.22832 | [PDF](https://arxiv.org/pdf/2605.22832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 77. SCRIPT: Scalable Diffusion Policy with Multi-stage Training for Language-driven Physics-Based Humanoid Control

**arXiv ID:** 2605.22894 | [PDF](https://arxiv.org/pdf/2605.22894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 78. When Do LLMs Reason? A Dynamical Systems View via Entropy Phase Transitions

**arXiv ID:** 2605.22873 | [PDF](https://arxiv.org/pdf/2605.22873v1)

**作者:** Wei Xia `[一作]` (Peking University), Yehui Tang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了链式思维（CoT）推理在大型语言模型（LLM）中的应用，提出了EDRM（基于熵动态的推理流形）框架，以动态选择推理策略，优化推理效率和准确性。

**💡 创新点**

创新点在于将LLM推理视为一种动态解码状态，并通过熵动态分析来判断何时需要显式推理，从而提出了一种轻量级、无训练的推理路由框架EDRM。

**🔧 技术方法**

使用了熵动态分析技术，通过分析解码过程中的熵变化来指导推理策略的选择。

**📊 数据集**

在15个基准测试和4种不同规模和架构的LLM上进行了实验，涵盖了数学推理、常识推理、科学推理和形式逻辑等多种任务。

**📈 对比分析**

与静态基线方法相比，EDRM在数据集层面上实现了41-55%的令牌减少，同时提高了准确性；在实例层面上，准确性提高了最多4.7%，同时保持了27-45%的令牌节省。

**⚠️ 局限性**

限制在于EDRM仍需进行短暂的探测阶段，相比于纯直接解码引入了适度的开销；当前实验仅限于3B-8B范围的开源文本模型，未来需要扩展到更大规模的模型和多模态系统。

---

## 79. Budgeted Dynamic Trace Structures for Token-Efficient Sequential Computation

**arXiv ID:** 2605.22879 | [PDF](https://arxiv.org/pdf/2605.22879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 80. ImProver 2: Iteratively Self-Improving LMs for Neurosymbolic Proof Optimization

**arXiv ID:** 2605.22885 | [PDF](https://arxiv.org/pdf/2605.22885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 81. NeuroNL2LTL: A Neurosymbolic Framework for Natural Language Translation of Linear Temporal Logic

**arXiv ID:** 2605.22874 | [PDF](https://arxiv.org/pdf/2605.22874v1)

**作者:** Paapa Kwesi Quansah `[一作]` (Baylor University), Ernest Bonnah `[通讯]` (Baylor University)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5006415431)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了NeuroNL2LTL，一个神经符号框架，用于将自然语言（NL）翻译为线性时序逻辑（LTL），并通过中间表示来确保翻译的结构保持。

**💡 创新点**

创新点在于引入了验证者在环训练（verifier-in-the-loop training），通过将验证结果作为强化学习的奖励信号，直接优化神经翻译的逻辑正确性。

**🔧 技术方法**

使用了神经网络编码器、确定性转换器和最小编辑修复模块等技术，结合了强化学习和形式验证。

**📊 数据集**

使用了包含218,871个需求-规范对的VERIFY数据集，涵盖航空航天、机器人、医疗设备等多个领域。

**📈 对比分析**

与基线系统（如大型语言模型和其他神经NL到LTL系统）比较，NeuroNL2LTL在语义等价性上达到了27.8%，在语法正确性上达到了93.7%，显著优于其他系统。

**⚠️ 局限性**

局限性在于神经编码器没有提供形式保证，生成的ITL字符串虽然通过验证，但不一定能捕捉输入需求的意图。

---

## 82. Robots That Know What to Ask: Recovering Misaligned Rewards through Targeted Explanations

**arXiv ID:** 2605.22986 | [PDF](https://arxiv.org/pdf/2605.22986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 83. Tensor Cache: Eviction-conditioned Associative Memory for Transformers

**arXiv ID:** 2605.22884 | [PDF](https://arxiv.org/pdf/2605.22884v1)

**作者:** Kabir Swain `[一作]` (Massachusetts Institute of Technology), Antonio Torralba `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 98724 | [OpenAlex ID](https://openalex.org/A5085020955)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Tensor Cache的两级缓存机制，结合了滑动窗口的softmax注意力（L1）和固定大小的外积快速权重内存（L2），用于处理Transformer模型中的KV对的驱逐。

**💡 创新点**

创新点在于将KV驱逐视为关联记忆写入操作，而不是简单的删除，并且通过并行加权和扫描来关闭训练中的一个未被注意的快捷方式，从而提高了内存质量和效率。

**🔧 技术方法**

使用了滑动窗口softmax注意力和外积快速权重内存的组合，采用了学习的标量门来融合L1和L2的输出。

**📊 数据集**

使用了OpenWebText和Shakespeare数据集进行训练和评估，涉及真实文本的长上下文语言建模和合成关联记忆任务。

**📈 对比分析**

与Full KV、Window KV、StreamingLLM和Infini-attention等方法进行了比较，Tensor Cache在所有评估上下文长度下的平均NLL最低，且在推理状态上节省了72-84%的内存。

**⚠️ 局限性**

限制在于Tensor Cache并不是全注意力的无损替代品，且在训练块大小之外的评估中，Full KV仍然是精确内容的基线。

---

## 84. Resilience Characterization of AI-Native Wireless Receivers via Persistent Homology

**arXiv ID:** 2605.22886 | [PDF](https://arxiv.org/pdf/2605.22886v1)

**作者:** Christo Kurisummoottil Thomas `[一作]` (Worcester Polytechnic Institute), Emilio Calvanese Strinati `[通讯]` (CEA-Leti)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于持久同调的实时鲁棒性指标TRI，用来衡量AI原生无线接收机在通道分布漂移下的参数空间结构稳定性，并利用TRI触发主动重适配。

**💡 创新点**

创新点在于将损失地形、参数轨迹与通道状态三类几何足迹统一映射到持久指数上，并证明TRI在界定性、单调性和Wasserstein‑2 Lipschitz稳定性方面具备理论保证。

**🔧 技术方法**

主要技术包括深度学习OFDM接收机、随机梯度下降在线适配、持久同调与持久指数计算、Ollivier–Ricci曲率与高斯核谱间隙等拓扑特征量化。

**📊 数据集**

实验使用ITU‑R标准的五种通道模型（UMa、UMi、RMa、InH、InF‑DH）在Sionna仿真框架下生成的OFDM链路数据。

**📈 对比分析**

与BER阈值、梯度范数和验证损失基线对比，TRI在10个场景下平均提前约1个OFDM符号（≈67µs）检测到漂移，且TRI驱动的突发重适配可将后漂移BER降低80%，显著优于传统基线。

**⚠️ 局限性**

局限性包括对计算成本的需求（VR持久同调计算需≈23 ms/50符号）、参数权重需手工设定、仅在仿真环境下验证，缺乏真实场景实验与更广泛通道模型的覆盖。

---

## 85. Beyond Zero: Enterprise Security for the AI Era

**arXiv ID:** 2605.22985 | [PDF](https://arxiv.org/pdf/2605.22985v1)

**作者:** Joseph Valente `[一作]` (Alphabet Inc.), Michal Zalewski `[通讯]` (Alphabet Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种新的企业安全模型，称为Beyond Zero，旨在实现机器速度的安全防护，支持基于上下文和风险的资源级授权。

**💡 创新点**

Beyond Zero模型扩展了Zero Trust概念，强调在实时评估和控制下进行个体资源的授权，而不是传统的应用程序级别的授权。

**🔧 技术方法**

使用了动态推理引擎和静态政策相结合的技术，能够在机器速度下进行快速决策和风险评估。

**📊 数据集**

未具体提及使用的数据集，但强调了对企业内部数据和用户行为的实时监控和分析。

**📈 对比分析**

与传统的静态安全模型相比，Beyond Zero能够在机器速度下进行动态决策，显著提高了安全性和响应速度。

**⚠️ 局限性**

模型的局限性在于需要大量的上下文信息和实时数据支持，且在实施过程中可能面临技术和标准化的挑战。

---

## 86. The Misattribution Gap: When Memory Poisoning Looks Like Model Failure in Agentic AI Systems

**arXiv ID:** 2605.22842 | [PDF](https://arxiv.org/pdf/2605.22842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 87. Cross-attention-based bipartite graph neural network for coupled nodal and elemental field prediction in large-deformation sheet material forming

**arXiv ID:** 2605.22845 | [PDF](https://arxiv.org/pdf/2605.22845v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 88. Remote Teleoperation of Endovascular Intervention Robots: A Systematic Review

**arXiv ID:** 2605.22889 | [PDF](https://arxiv.org/pdf/2605.22889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 89. LFRAG: Layout-oriented Fine-grained Retrieval-Augmented Generation on Multimodal Document Understanding

**arXiv ID:** 2605.22829 | [PDF](https://arxiv.org/pdf/2605.22829v1)

**作者:** Yifan Zhu `[一作]` (Zhejiang University), Zhixuan Chu `[通讯]` (Zhejiang University)

**通讯引用:** 984 | [OpenAlex ID](https://openalex.org/A5008967163)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的框架LFRAG，旨在通过从页面级检索转向块级检索来提高多模态文档理解的精确性。

**💡 创新点**

LFRAG通过布局分割和语义-布局融合编码，显著提高了检索准确性和生成效率，减少了冗余上下文。

**🔧 技术方法**

使用了布局分割、语义-布局特征融合编码和块级晚期交互检索等技术。

**📊 数据集**

构建了LFDocQA数据集，这是一个具有块级注释的大规模基准，涵盖多种文档类型。

**📈 对比分析**

与现有的OCR和VLM基础的检索方法相比，LFRAG在检索任务上表现出色，答案准确率提高了7.20%，生成任务中减少了73.07%的令牌消耗。

**⚠️ 局限性**

LFRAG在处理复杂布局和结构化表格内容时仍面临挑战，未来需要进一步优化以处理这些结构化数据。

---

## 90. Scene Reconstruction as Mapping Priors for 3D Detection

**arXiv ID:** 2605.22997 | [PDF](https://arxiv.org/pdf/2605.22997v1)

**作者:** Yang Fu `[一作]` (University of California San Diego), Yingwei Li `[通讯]` (Waymo LLC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种可扩展的映射先验生成与门控融合框架 MPA3D，用于在稀疏 LiDAR 数据上提升 3D 目标检测精度。

**💡 创新点**

创新点包括：①利用聚合传感器数据自动生成 surfel 与 3D Gaussian 映射先验；②设计门控融合模块自适应调节映射先验对 LiDAR 与摄像头特征的影响；③引入混合模态训练策略，使模型在任何模态缺失时保持鲁棒性。

**🔧 技术方法**

使用技术包括：多模态编码器（LiDAR PointMLP、摄像头 Lift‑Splat‑Shoot、surfel 与 3D Gaussian 转化为点特征）、动态体素化、BEV 投影、门控融合、Sparse‑Window Transformer 检测头，以及 3D 数据增强与自监督预训练。

**📊 数据集**

主要在 Waymo Open Dataset 上进行训练与评估，并通过额外 7M 序列（含 600k 带映射先验）进行预训练。

**📈 对比分析**

与单帧、四帧多帧、以及长时序融合方法对比，MPA3D 在 Waymo 验证集 L1/APH 提升 2.2%、L2/APH 提升 2.7%，在测试集上实现了新的 SOTA，且仅使用 4 帧即可达到或超过使用 99 帧的时序方法。

**⚠️ 局限性**

局限性：映射先验生成仍依赖多场景的多次传感器扫描；动态物体在 3D Gaussian 里仍需进一步去除；推理时延因额外映射先验模块提升至 452 ms，低于基线 245 ms。

---

## 91. Mediative Fuzzy Logic: From Type-1 Foundations to Type-2, Type-3 and Quantum Extensions

**arXiv ID:** 2605.22900 | [PDF](https://arxiv.org/pdf/2605.22900v1)

**作者:** Oscar Montiel Ross `[一作]` (Instituto Politécnico Nacional), Oscar Montiel Ross `[通讯]` (Instituto Politécnico Nacional)

**通讯引用:** 3494 | [OpenAlex ID](https://openalex.org/A5033991197)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种统一的中介模糊逻辑（Mediative Fuzzy Logic, MFL）框架，涵盖了类型1、类型2、类型3和量子扩展，旨在处理模糊控制和决策中的犹豫和矛盾评估。

**💡 创新点**

创新点在于将中介运算符形式化为一个凸聚合，能够在存在犹豫和矛盾的情况下进行有效的推理，并且提供了一个完整的逻辑系统，支持多层次的模糊推理。

**🔧 技术方法**

使用了中介模糊逻辑的运算符，结合了类型1、类型2和类型3模糊集的语义，以及量子效应和密度算子来扩展模糊逻辑的应用。

**📊 数据集**

论文中没有具体提到使用的数据集，但通过一个自主制动传感器融合的案例来说明框架的应用。

**📈 对比分析**

与传统模糊逻辑和直觉模糊逻辑相比，MFL在处理矛盾和犹豫时表现出更好的一致性和安全性，能够在不完整和矛盾证据下做出透明和保守的决策。

**⚠️ 局限性**

限制在于当前的框架主要集中在理论构建上，实际应用中可能需要更多的实证研究来验证其有效性和适用性。

---

## 92. Pointwise Metrics Mislead: An Evaluation Protocol for Multimodal Inverse Problems

**arXiv ID:** 2605.22891 | [PDF](https://arxiv.org/pdf/2605.22891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 93. Certification from Examples is Hard for Circuits and Transformers under Minimal Overparametrization

**arXiv ID:** 2605.22964 | [PDF](https://arxiv.org/pdf/2605.22964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 94. Measuring Database Unfairness via Dependency Quantification Under Differential Privacy

**arXiv ID:** 2605.22952 | [PDF](https://arxiv.org/pdf/2605.22952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 95. RAS: Reflection-Augmented Scaling with In-Context Learning for Executable Cypher Query Generation

**arXiv ID:** 2605.22937 | [PDF](https://arxiv.org/pdf/2605.22937v1)

**作者:** Minseok Jung `[一作]` (Cloudera), Muhammad Rameez Chatni `[通讯]` (Cloudera)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在结构化查询生成任务Text2Cypher中，如何通过推理时的计算扩展来降低执行错误率，提出并比较了两种推理策略：独立扩展（IS）和基于反射的扩展（RAS）

**💡 创新点**

创新点在于将数据库执行错误反馈作为in‑context学习的输入，形成迭代纠错循环，从而显著降低执行错误率；并系统性比较了IS与RAS在相同计算预算下的表现

**🔧 技术方法**

采用语言模型（CodeLlama、DeepSeek、Qwen2.5、StarCoder2等）在Neo4j数据库上生成Cypher查询，利用in‑context学习将执行错误信息嵌入上下文；实现了两种推理算法（IS、RAS）并测量QER

**📊 数据集**

使用了三大公开Neo4j属性图数据集：Healthcare、Fraud、Crime，涵盖不同规模与结构密度，生成了三种复杂度（Easy、Medium、Hard）的自然语言查询

**📈 对比分析**

在相同计算预算（T=5）下，IS通过多次无记忆采样降低QER，RAS则利用错误反馈进行迭代生成；实验显示RAS在所有模型和数据集上将QER降低41–50%，明显优于IS的32–38%

**⚠️ 局限性**

局限性包括：仅评估执行可行性而非语义准确性；仅在Text2Cypher与Neo4j上验证，未覆盖其他语义解析任务；预算上限为5次，可能隐藏更大预算下的收益；使用in‑context学习会产生令牌与延迟开销，且未对跨查询日志进行持久化学习

---

## 96. Opportunities and Risks of Generative AI through the Health Information Journey

**arXiv ID:** 2605.23026 | [PDF](https://arxiv.org/pdf/2605.23026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 97. Inconsistency-aware Multimodal Schrödinger Bridge for Deepfake Localization

**arXiv ID:** 2605.23113 | [PDF](https://arxiv.org/pdf/2605.23113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 98. Query Lower Bounds for Correlation Clustering under Memory Constraints

**arXiv ID:** 2605.23104 | [PDF](https://arxiv.org/pdf/2605.23104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 99. ThriftAttention: Selective Mixed Precision for Long-Context FP4 Attention

**arXiv ID:** 2605.23081 | [PDF](https://arxiv.org/pdf/2605.23081v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 100. ACALSim: A Scalable Parallel Simulation Framework for High-Performance System Design Space Exploration

**arXiv ID:** 2605.22936 | [PDF](https://arxiv.org/pdf/2605.22936v1)

**作者:** Wei-Fen Lin `[一作]` (Mijotech Inc.), Yu-Jie Wan `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提出了 ACALSim，一款可扩展的多线程仿真框架，用于加速高性能计算系统（尤其是 GPU 与 AI 加速器）的设计空间探索。

**💡 创新点**

核心创新在于可插拔的线程管理架构，允许开发者针对不同仿真模式自定义调度策略，同时结合事件驱动快进、两阶段确定性并行与共享内存通信模型，显著提升仿真吞吐量。

**🔧 技术方法**

技术实现包括 C++/SystemC 混合接口、线程池调度、事件队列快进、双缓冲通信通道、对象池和多阶段（并行 + 归档）执行模式，以及插件化的 ThreadManager 接口。

**📊 数据集**

使用的数据集主要是 LLaMA-7B/13B Transformer 推理层、NVIDIA A100 及 H100 GPU 的 GEMM 负载以及 DGX/BlackBear 等自研模拟器的架构配置（通过 JSON 控制参数）。

**📈 对比分析**

与基于 SST 的实现直接对比，ACALSim 在中等规模工作负载（64 TB）上实现 14× 速度提升、内存占用降低 41%，并在 LLaMA 推理层的 17.7–30.4 分钟仿真时间内完成，SST 在相同规模下往往超时。

**⚠️ 局限性**

局限性包括：对极小工作负载启动成本相对较高、仍不适用于松耦合的大规模分布式系统（需结合 SST 或 ROSS）、需要开发者自行实现并验证各组件的时序模型，以及线程管理策略需根据具体仿真模式手动切换。

---

## 101. Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models

**arXiv ID:** 2605.23070 | [PDF](https://arxiv.org/pdf/2605.23070v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 102. RoboSurg-VQA: A Multimodal Benchmark for Surgical Segmentation-Aware Visual Question Answering

**arXiv ID:** 2605.23068 | [PDF](https://arxiv.org/pdf/2605.23068v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 103. What Training Data Teaches RL Memory Agents: An Empirical Study of Curriculum Effects in Memory-Augmented QA

**arXiv ID:** 2605.23067 | [PDF](https://arxiv.org/pdf/2605.23067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 104. AI-Friendly LaTeX: Using LaTeX Code as a Knowledge Source for Retrieval-Augmented Generation

**arXiv ID:** 2605.22923 | [PDF](https://arxiv.org/pdf/2605.22923v1)

**作者:** Tom Verhoeff `[一作]` (Eindhoven University of Technology), Tom Verhoeff `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1058 | [OpenAlex ID](https://openalex.org/A5103729667)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套 LaTeX 源文件预处理流水线，将源代码转换为 Markdown 并生成适合向量数据库检索的 JSONL 分块，同时处理引用、宏、图形及作者注解。

**💡 创新点**

创新点在于将 LaTeX 的结构化与语义信息相结合，提供了宏解析、引用解析、图形语义描述等多层次预处理策略，并通过 YAML 与内嵌宏实现注解的可配置化。

**🔧 技术方法**

使用 Python 实现，利用正则表达式和解析库读取 .tex、.aux、嵌入文件，构建标签表，生成 Markdown、标签 JSON 和 JSONL 分块；同时对 TikZ、图片等图形进行语义提取。

**📊 数据集**

未使用公开数据集，实验基于典型教材、讲义、硕士论文等自制 LaTeX 源文件；示例代码与配置已在 GitHub 开源。

**📈 对比分析**

文中未给出量化性能比较，仅通过示例说明预处理后检索质量的提升，建议可在向量数据库检索实验中进一步评估。

**⚠️ 局限性**

局限在无法完整解析所有宏与复杂结构，对未知内容仅保留原文并发出警告；对 TikZ 等高层次图形的语义抽取仍需人工补充或扩展宏库。

---

## 105. How to Steer Your Multi-Agent System: Human-LLM Collaborative Planning

**arXiv ID:** 2605.23023 | [PDF](https://arxiv.org/pdf/2605.23023v1)

**作者:** Zeyu He `[一作]` (Penn State University), Estevam Hruschka `[通讯]` (Megagon Labs)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文正式化了人机-LLM协同规划的设计空间，并在此基础上实现了一个双面板原型，随后通过用户实验和对比基准评估该系统与传统方法的差异；

**💡 创新点**

创新点包括：①将协同规划三维度（模式、范围、层级）系统化，形成可量化的交互设计空间；②引入结构化编辑与语义反馈的组合以及LLM辅助的高层级结构变更；③通过实验揭示用户在“努力‑风险”权衡下的混合工作流与LLM修订策略的相互影响；

**🔧 技术方法**

技术实现涵盖：LLM驱动的规划与重规划模块、四类专用执行代理（代码、数学、检索、常识）、有向无环图（DAG）表示的计划、结构化编辑操作、语义反馈解析、LLM辅助的自动合并/拆分，以及实验收集与分析工具；

**📊 数据集**

使用了自定义基准：200个金标准计划与1150个人工破坏的计划实例，覆盖四种结构模式（逐步数学推理、多跳计算、列表检索与聚合、Top‑K检索与聚合），并从GSM8K、Claude Sonnet等源生成任务；

**📈 对比分析**

对比方法：在用户实验中将高级交互（针对性语义反馈、LLM辅助高层编辑）与基线（全局重规划、低层直接编辑）进行交叉比较，评估计划质量、执行准确率、效率与努力；结果显示全局重规划在执行准确率上最高（约0.84），但高级交互在用户努力和易用性上更佳；在基准实验中，目标反馈提升计划稳定性，编辑序列策略提升可解释性但可靠性下降；

**⚠️ 局限性**

局限性：样本量仅13人，且来自单一实验室，缺乏多样性；仅研究单一用户与固定的专用代理组，未探讨多用户或去中心化MAS；用户在实验中出现疲劳导致验证下降；基准计划人为构造，缺乏更广泛的真实错误分布；整体方法在更复杂、开放式规划场景中的适用性尚未验证。

---

## 106. DreamerNLplus: Interpretable Modeling of Mental Health Dynamics from Social Media Timelines using Hybrid Rule-Based and RAG Methods

**arXiv ID:** 2605.23052 | [PDF](https://arxiv.org/pdf/2605.23052v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 107. Do Language Models Know What Not to Say? Causal Evidence for Statistical Preemption in LLMs

**arXiv ID:** 2605.23039 | [PDF](https://arxiv.org/pdf/2605.23039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 108. PathCal: State-Aware Reflection-Marker Calibration for Efficient Reasoning

**arXiv ID:** 2605.23074 | [PDF](https://arxiv.org/pdf/2605.23074v1)

**作者:** Lingyu Jiang `[一作]` (Tohoku University), Fangzhou Lin `[通讯]` (Tohoku University)

**通讯引用:** 288 | [OpenAlex ID](https://openalex.org/A5026905610)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的解码控制器PathCal，旨在通过区分反思标记的类型和在局部不确定状态下进行干预来校准推理路径。

**💡 创新点**

创新点在于识别反思标记的不同功能角色，并提出了一种训练无关的解码控制方法，能够在推理过程中动态调整标记的权重。

**🔧 技术方法**

使用了PathCal作为解码控制器，通过对反思标记的分布进行分析，估计当前推理路径与竞争分支之间的局部竞争。

**📊 数据集**

在六个推理基准上进行了实验，包括GSM8K、MATH500、AIME2024、AIME2025、AMC2023和TheoremQA。

**📈 对比分析**

与四种训练无关的基线方法进行比较，PathCal在所有模型-基准对中提高或保持了准确性，同时通常缩短了生成长度，显示出更好的效率-性能权衡。

**⚠️ 局限性**

限制在于PathCal是一个轻量级的单样本控制器，尚未与更广泛的测试时间缩放方法结合使用，未来的工作可以探索与这些方法的组合。

---

## 109. UfM*: Uncertainty from Motion* for DNN Depth Estimation Using Gaussians

**arXiv ID:** 2605.23098 | [PDF](https://arxiv.org/pdf/2605.23098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 110. Orbax: Distributed Checkpointing with JAX

**arXiv ID:** 2605.23066 | [PDF](https://arxiv.org/pdf/2605.23066v1)

**作者:** Colin Gaffney `[一作]` (Google), Rakesh Iyer `[通讯]` (Google)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5112912493)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了Orbax，一个模块化的JAX原生检查点库，旨在简化分布式加速器系统的复杂性，并提供用户友好的检查点操作。

**💡 创新点**

Orbax通过抽象分布式存储和进程协调的复杂性，提供高性能的检查点解决方案，超越了现有的PyTorch竞争对手。

**🔧 技术方法**

使用了JAX框架，结合了模块化设计和灵活的API，支持多种检查点操作。

**📊 数据集**

在NVIDIA A100 GPU上使用Llama 3.1模型（8B、70B和405B）进行评估，并使用Google Cloud Storage作为存储后端。

**📈 对比分析**

与PyTorch的分布式检查点库（DCP）进行比较，Orbax在加载性能上通常优于DCP，尤其是在较大模型上，加载速度提高了2倍以上。

**⚠️ 局限性**

在小模型的情况下，Orbax的检查点文件格式可能会导致额外的写入开销，限制了其在某些场景下的性能表现。

---

## 111. Decomposing and Measuring Evaluation Awareness

**arXiv ID:** 2605.23055 | [PDF](https://arxiv.org/pdf/2605.23055v1)

**作者:** Changling Li `[一作]` (Max Planck Institute for Intelligent Systems), Maksym Andriushchenko `[通讯]` (ELLIS Institute Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文探讨了前沿语言模型在评估时的意识问题，提出了一个框架来分解评估意识，并通过EvalAwareBench基准进行实证研究。

**💡 创新点**

创新点在于将评估意识的概念与社会心理学相结合，提出了一个包含八个触发因素的分类法，并开发了EvalAwareBench基准以控制这些因素的影响。

**🔧 技术方法**

使用了链式思维监测（CoT）技术来研究模型的识别能力和行为倾向，并通过统计分析方法（如ANOVA）来分析数据。

**📊 数据集**

研究涉及九个前沿模型和四个基准（HarmBench、Agentic Misalignment、AgentHarm Harmful和Benign），并构建了100对安全-能力任务的EvalAwareBench基准。

**📈 对比分析**

通过EvalAwareBench，发现没有单一因素对所有模型均有统一影响，但叠加因素会普遍提高评估意识。模型在安全评估上的意识普遍高于能力评估，且识别率与模型和基准的特定配对有关。

**⚠️ 局限性**

现有基准无法独立评估每个因素的贡献，且CoT监测仅捕捉到口头化的识别，可能低估了真实的评估意识。未来的研究需要扩展到内部表示的覆盖范围。

---

## 112. Four Simple Proprioceptive Estimators for Legged Robots

**arXiv ID:** 2605.23100 | [PDF](https://arxiv.org/pdf/2605.23100v1)

**作者:** Frank Dellaert `[一作]`, Ayoung Kim `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一系列基于IMU和脚接触的腿式机器人自我定位估计器，逐步从不变EKF到局部图优化，再到固定时滞平滑器，并引入可演化的IMU偏置轨迹。

**💡 创新点**

创新点包括：① 在不变滤波框架下将测量更新改为局部图优化，实现多接触点同步处理；② 用脚接触事件构造固定时滞滑动窗口图，让底座状态和接触点在时间上解耦；③ 通过在滑动窗口中引入偏置轨迹，克服长时间实验中IMU偏置漂移的限制。

**🔧 技术方法**

主要技术手段：Lie群不变滤波、IMU预积分、基于接触的约束因子、局部Levenberg–Marquardt优化、固定时滞平滑器、滑动窗口优化与偏置随机游走。

**📊 数据集**

使用Boston Dynamics Spot平台收集的GaRLILEO数据集（含IMU、关节编码器、脚接触、雷达、LiDAR和地面真值）。

**📈 对比分析**

与Pronto、MUSE、Holistic Fusion等纯腿式-IMU基线做对比，使用RMSE的绝对位姿误差（APE）和相对位姿误差（RPE）指标。实验显示四种GTSAM变体在平面和三维场景中普遍优于基线，尤其在垂直漂移（APE_z）上显著改善；但在某些上坡/下坡序列中可演化偏置平滑器（FL-Combined）偶有较高垂直误差。

**⚠️ 局限性**

局限性：可演化偏置模型在垂直漂移方面并不总是更优，需更细致的IMU偏置建模；仅使用自我感知信息，缺乏外部传感器可能在复杂环境中仍存在误差累积；实验范围主要集中在Spot平台，泛化性待进一步验证。

---

## 113. Encrypted Neural Networks without Overflows

**arXiv ID:** 2605.23096 | [PDF](https://arxiv.org/pdf/2605.23096v1)

**作者:** Philipp Kern `[一作]` (Karlsruhe Institute of Technology), Alberto Leporati `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 1948 | [OpenAlex ID](https://openalex.org/A5083297323)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对CKKS基的全同态加密神经网络，研究了由于激活函数多项式逼近范围不足导致的溢出攻击，并提出了一套基于正式验证的“溢出免疫”设计框架，能够为网络中每一层计算并证明安全范围，从而消除溢出风险并保持推理精度。

**💡 创新点**

创新点在于首次揭示CKKS网络易受溢出攻击，提出了利用Remez算法与分段多项式逼近结合的安全区间构造方法；同时设计了差分验证技术实现对多项式逼近误差的可证明上界，并允许同一层使用不同多项式，显著降低误差。

**🔧 技术方法**

核心技术包括：CKKS加密算子（加法、乘法、旋转）、Remez多项式逼近、分段多项式逼近与误差分析、差分验证（改进VeryDiff与RaVeN松弛）、zonotope范围推理，以及开源实现基于OpenFHE的完整编译链。

**📊 数据集**

实验使用了多种数据集：HELOC（表格回归）、MNIST（手写数字分类）、CIFAR-10（图像分类）、EC（能耗预测）、Collins RUL（剩余寿命预测）以及NN4Sys（学习索引）等，涵盖从小型到中型网络（64–10,300个神经元）。

**📈 对比分析**

与传统采样-基范围设计相比，证明确认设计在防止溢出攻击方面做到零失败率，同时在分类准确率和回归误差上仅出现极小的性能下降（大多数模型误差<0.01%）；对比实验表明采样方法在攻击成功率可达12–47%，而证明确认方法始终保持0%。

**⚠️ 局限性**

局限性包括：目前只支持不使用引导（bootstrapping）的网络，难以扩展到更深网络；多项式系数可能过大导致CKKS精度下降；对CKKS噪声的建模仍为近似，进一步精确建模可提升安全裕度。

---

## 114. Robust OT-Guided Generative Residual Domain Adaptation for Bike-Sharing Demand Prediction under Temporal Domain Shift

**arXiv ID:** 2605.23115 | [PDF](https://arxiv.org/pdf/2605.23115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 115. Dreaming Smoothly and Sample Efficiently with Gradient Penalized Latent Dynamics

**arXiv ID:** 2605.23089 | [PDF](https://arxiv.org/pdf/2605.23089v1)

**作者:** Romil V. Sonigra `[一作]` (Texas A&M University), P. R. Kumar `[通讯]` (Texas A&M University)

**通讯引用:** 5067 | [OpenAlex ID](https://openalex.org/A5107699893)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了GPLD，一种梯度惩罚的潜在动力学正则化方法，应用于DreamerV3以提升连续控制任务的样本效率。

**💡 创新点**

通过离散-连续平滑理论，将邻域转移差分平滑映射为潜在空间的Frobenius雅可比正则化，首次为随机潜在世界模型提供局部平滑约束。

**🔧 技术方法**

使用梯度惩罚（Hutchinson估计）对后验潜在概率映射施加行向雅可比范数正则化，并采用平方根衰减调度。

**📊 数据集**

在DeepMind Control的低维物理状态和像素观测任务上进行实验。

**📈 对比分析**

相较于原始DreamerV3，GPLD在低维物理任务上实现了约17.7%的归一化整体收益，局部复杂度更高的运动任务收益达34.6%，并在四足机器人长时限任务中更快收敛且保持更稳定的后期性能。

**⚠️ 局限性**

在像素观测场景下收益较弱，且在存在离散跳跃或高维视觉编码的环境中可能不适用。

---

## 116. DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge

**arXiv ID:** 2605.23069 | [PDF](https://arxiv.org/pdf/2605.23069v1)

**作者:** Yusser Al Ghussin `[一作]` (German Research Center for Artificial Intelligence), Simon Ostermann `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在SemEval‑2026 Task 7的文化认知评测中，使用推理时激活引导（activation steering）对多语言LLM进行微调，以提升对不同语言–文化对的推理准确率。

**💡 创新点**

创新点在于利用并行多语言数据（FLORES）提取语言向量，直接在Transformer残差流中注入无参数更新的激活向量，从而在推理时实现轻量级的文化对齐。

**🔧 技术方法**

核心技术包括：激活空间语言向量提取（DiffMean方法）、推理时激活注入（Steering）、层级与强度的调优、以及文化化提示设计。

**📊 数据集**

主要使用的数据集是FLORES用于构造语言向量，以及BLEnD（SemEval 2026共享任务的MCQ和SAQ数据）用于评估。

**📈 对比分析**

与官方排行榜比较，MCQ任务最终取得86.96 %准确率，排名第7/17；单语言表现不一，部分文化对齐提升约+1.5 pp，其他则无显著或略微下降。

**⚠️ 局限性**

局限性包括：对语言与文化的混淆（同一语言对应多地区的向量相同）、单一全局Steering配置导致不同层/提示/语言的效果差异、缺乏与更强基线或其他适配方法的系统对比，以及提交文件错误导致SAQ评测缺失。

---

## 117. Security of LLM-generated Code: A Comparative Analysis

**arXiv ID:** 2605.23091 | [PDF](https://arxiv.org/pdf/2605.23091v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 118. HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation

**arXiv ID:** 2605.23043 | [PDF](https://arxiv.org/pdf/2605.23043v1)

**作者:** Zewei Deng `[一作]` (University of Minnesota), Liyan Xie `[通讯]` (University of Minnesota)

**通讯引用:** 428 | [OpenAlex ID](https://openalex.org/A5075007159)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 HawkesLLM 框架，用于监测迭代文本生成过程中的语义不确定性传播。

**💡 创新点**

创新点在于将多元 Hawkes 点过程与 LLM 生成分离，利用节点间激活概率和权重来选取紧凑记忆，实时追踪语义漂移，并在有限提示内实现语义对齐。

**🔧 技术方法**

使用的技术包括多元 Hawkes 点过程、Ogata 删减采样、LLM（Qwen2.5）文本生成、提示内记忆选择策略、语义对齐与全局/局部漂移诊断。

**📊 数据集**

使用的基准数据集为 GDELT 的 Artemis II 相关新闻元数据，约 248 条英文标题。

**📈 对比分析**

与两种基线（chronological last‑k、random‑k）比较，HawkesLLM 在 k=3 的提示预算下实现了最高的平均语义对齐（S_t≈0.636）并呈上升趋势，晚期对齐显著优于基线。

**⚠️ 局限性**

局限性包括数据量有限、仅使用标题级文本、评估依赖相同模型的嵌入，缺乏大规模实验和人工/事实性检验，且生成偶尔出现语言混杂。

---

## 119. Conceptual Schema Inference for Tabular Datasets using Large Language Models

**arXiv ID:** 2605.23105 | [PDF](https://arxiv.org/pdf/2605.23105v1)

**作者:** Zhenyu Wu `[一作]` (University of Manchester), Norman W. Paton `[通讯]` (University of Manchester)

**通讯引用:** 11317 | [OpenAlex ID](https://openalex.org/A5066619159)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出两种基于大语言模型的技术，用于从仅有列名和单元值的原始表格中自动推断概念模式（实体类型、层次、属性和跨类型关系）

**💡 创新点**

创新点在于同时实现了类型层次、属性归属和关系推断的全流程，并引入了约束式提示与LLM验证来提高路径一致性，同时提供了以表嵌入为基础的相似度聚类方法

**🔧 技术方法**

采用解码器式LLM（如GPT‑3.5/4、Qwen、Llama‑3.1）进行生成推断，并使用编码器式LLM（SwAV、DeepJoin、Starmie等）提取表/列嵌入进行聚类与层次构造

**📊 数据集**

实验使用四个公开基准：Web Data Commons（WDC）、Google Dataset Search（GDS）、OpenData（Small）和OpenData（Large），覆盖数千至上万张表格

**📈 对比分析**

与先前方法GeTT和SI‑LLM对比，本文方法在类型与层次一致性（PTCS）和属性聚类纯度（Purity）上显著提升，且在关系与基数推断上表现优于基于实例相似度的方法，整体F1提升约0.4–0.6

**⚠️ 局限性**

局限性包括：对缺乏丰富列名或数值信息的列属性识别困难；多路径层次导致一致性损失；关系推断对实例样本偏差敏感，导致基数错误；以及在大规模数据中属性聚类的计算开销较高

---

## 120. Steered Generation via Gradient-Based Optimization on Sparse Query Features

**arXiv ID:** 2605.23040 | [PDF](https://arxiv.org/pdf/2605.23040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 121. A Fine-Tuned BERT Classifier for Personal-Letter Titles in Late-Ming and Early-Qing Collected Works

**arXiv ID:** 2605.23103 | [PDF](https://arxiv.org/pdf/2605.23103v1)

**作者:** Queenie Luo `[一作]` `[通讯]` (Harvard University), Queenie Luo (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发并部署了Lepton，一种针对晚明至清初文集标题的细调BERT分类器，用于区分个人书信和易混淆的序文。

**💡 创新点**

创新点在于将预训练的中文BERT直接细调至标题级别分类，并在少量标注数据上达到与人类历史学家相近甚至超越的精度，同时提供公开的模型与标注语料。

**🔧 技术方法**

使用方法为在约102M参数的现代中文BERT上添加分类头并全参数微调，配合token级特征。

**📊 数据集**

训练数据由33位作者的5,438条手工标注标题组成，其中3,206条书信、2,232条序文。

**📈 对比分析**

与多数类、基于终结词或起始动词的正则表达式以及TF‑IDF+LogReg线性模型相比，Lepton在测试集上的准确率0.977、精确率1.0、召回率0.968、F1 0.984，明显优于所有基线。

**⚠️ 局限性**

局限性包括只处理书信与序文的二分类，未识别收信人或时间等信息；在宋元等更早朝代以及极短、无动词开头的标题上表现不佳；对超出训练分布的文本预测不确定。

---

## 122. The Implicit Bias of Depth: From Neural Collapse to Softmax Codes

**arXiv ID:** 2605.23087 | [PDF](https://arxiv.org/pdf/2605.23087v1)

**作者:** Connall Garrod `[一作]` (Mathematical Institute, University of Oxford), Christos Thrampoulidis `[通讯]` (University of British Columbia)

**通讯引用:** 1448 | [OpenAlex ID](https://openalex.org/A5024812488)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了深度无约束特征模型（UFM）在没有正则化的情况下，如何通过梯度下降和深度影响神经崩溃（NC）的几何结构。

**💡 创新点**

首次全面表征了深度、优化动态和初始化的隐性偏差如何相互作用以决定NC的出现。

**🔧 技术方法**

使用了梯度流（GF）动态分析技术，结合Hadamard初始化来研究深度UFM的训练动态。

**📊 数据集**

使用了MNIST和CIFAR-10数据集进行实验验证。

**📈 对比分析**

与传统的L2正则化模型相比，深度UFM在没有正则化的情况下表现出隐性低秩偏差，导致低秩结构的出现。实验结果表明，深度增加时有效秩降低，且宽度增加时有效秩上升。

**⚠️ 局限性**

研究主要集中在过参数化的极限，可能在欠参数化或高度约束的设置中适用性减弱。此外，模型未直接解决泛化问题，且训练动态依赖于Hadamard初始化，限制了其普适性。

---

## 123. Positional Identifiability from Pairwise Collision Data

**arXiv ID:** 2605.23073 | [PDF](https://arxiv.org/pdf/2605.23073v1)

**作者:** Yun-Han Li `[一作]` (University of Illinois), Olgica Milenkovic `[通讯]` (University of Illinois)

**通讯引用:** 8342 | [OpenAlex ID](https://openalex.org/A5084947882)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文研究了仅基于两两碰撞数据恢复移动物体在实线上相对位置的问题，并在三种观测模型下给出了理论判定和恢复方法。

**💡 创新点**

创新点在于：① 在完全可观测模型下给出碰撞图连通性为唯一可恢复的必要且充分条件；② 在部分可观测模型下引入层分解，将每层对应最大团，并证明收缩图为区间图，提供了高效的恢复算法；③ 在不完整观测模型下将问题转化为图补全，并证明其与图带宽问题的4近似关系，进一步证明NP-难性。

**🔧 技术方法**

主要技术包括图论（连通性、团、区间图、图收缩）、组合优化和近似算法（图带宽近似）。

**📊 数据集**

未使用任何实验数据集，本文全部以理论分析为主。

**📈 对比分析**

论文未进行实验比较或性能评估，因其为理论性工作。

**⚠️ 局限性**

局限性包括仅针对一维实线运动，缺乏实证验证，且在不完整观测下问题被证明为NP-难，实际应用需进一步研究近似或启发式算法。

---

## 124. Anytime Training with Schedule-Free Spectral Optimization

**arXiv ID:** 2605.23061 | [PDF](https://arxiv.org/pdf/2605.23061v1)

**作者:** Anuj Apte `[一作]` (Global Technology Applied Research), Junhyung Lyle Kim `[通讯]` (Global Technology Applied Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Schedule-Free Spectral 优化器 SF-NorMuon，支持任意停止点的训练并可与调参后的 AdamW 基线相媲美。

**💡 创新点**

通过结合谱梯度极大化、行级归一化、显式动量以及在快速迭代上施加权重衰减，填补了 Schedule-Free 训练在性能与长期稳定性上的缺口。

**🔧 技术方法**

使用谱（极化）更新、行级自适应归一化、动量缓冲、在快速迭代上加权重衰减，并实现了无学习率调度的在线平均。

**📊 数据集**

在 FineWeb‑100B 语言建模数据集上，训练 125M 与 772M 参数的 LLaMA‑2 风格 Transformer。

**📈 对比分析**

与按 Horizon 调参的 AdamW（cosine 调度）及 SF‑AdamW 对比，SF‑NorMuon 在 1–8× Chinchilla 训练范围内性能相当，且相较于 SF‑AdamW 速度提升 35–50% 的训练步数。

**⚠️ 局限性**

实验仅覆盖当前 Transformer 架构与 FineWeb‑100B 数据集，需进一步验证在更广泛模型、数据与多阶段持续学习场景下的表现。

---

## 125. Dithering Defense: Adversarial Robustness of Vision Foundation Models via Multi-Level Floyd-Steinberg Dithering

**arXiv ID:** 2605.23065 | [PDF](https://arxiv.org/pdf/2605.23065v1)

**作者:** Yury Belousov `[一作]` (University of Geneva), Slava Voloshynovskiy `[通讯]` (University of Geneva)

**通讯引用:** 3870 | [OpenAlex ID](https://openalex.org/A5091506990)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在冻结的视觉基础模型上使用多层 Floyd–Steinberg 错误扩散抖动作为轻量级、模型无关的输入转换，以提升对抗鲁棒性。

**💡 创新点**

创新点在于系统地将 FS 抖动推广到多量化级别并与后处理模糊结合，同时在六种下游任务、两大模型族、三种攻击及自适应攻击下进行全面评估，显示其优于传统压缩、模糊和扩散等基线。

**🔧 技术方法**

使用了 Floyd–Steinberg 量化抖动、可选 Gaussian blur、以及直通估计（STE）实现的自适应攻击。

**📊 数据集**

实验数据集包括 PascalVOC、NYU‑Depth、R‑Oxford、COCOCaptions、VQAv2 等，并在对应的指标（准确率、mIoU、RMSE、mAP 等）上进行评估。

**📈 对比分析**

通过与 JPEG、Wiener、模糊、低噪扩散等输入变换基线比较，发现 K=3‑5 级量化+模糊配置在未攻击时保持 90%+ 的性能，攻击下恢复至 70%+，相较高噪扩散具有更小的清洁降级，并在自适应攻击下仅降幅 1–3%。

**⚠️ 局限性**

局限性包括：需统一量化级别，未探讨通道/色彩空间细粒度调节；高量化级别下鲁棒性下降；对更多模型、攻击类型的泛化仍需进一步验证。

---

## 126. Millimeter-wave Imaging for Anthropometric Body Measurement

**arXiv ID:** 2605.23064 | [PDF](https://arxiv.org/pdf/2605.23064v1)

**作者:** Miriam Senne `[一作]` (Technical University of Munich), Azade Farshad `[通讯]` (Technical University of Munich)

**通讯引用:** 386 | [OpenAlex ID](https://openalex.org/A5014777005)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用毫米波雷达扫描结合 SMPL 参数化模型实现无接触、隐私保护的服装容忍人体体测。

**💡 创新点**

提出基于顶点加权的 Chamfer 能量与脚底地面约束的动态优化框架，显著提升稀疏点云对齐精度和对服装的鲁棒性。

**🔧 技术方法**

毫米波雷达、SMPL 参数模型、Chamfer 距离、动态顶点权重、地面约束、梯度优化、ISO 8559 体测标准。

**📊 数据集**

使用一只实验模型（Artec Leo 3D 扫描）与 27 名受试者（两套服装）以及 RGB 单视角基线（STAR、SHAPY）。

**📈 对比分析**

与手工卷尺和高精度 3D 扫描对比，mmWave+SMPL 的平均绝对偏差（MAD）低于 RGB 基线，且在有服装时误差增幅更小，显示出更高的精度与鲁棒性。

**⚠️ 局限性**

局限包括卷尺基准本身误差、点云稀疏导致侧面/头部对齐不足、SMPL 参数化无法完全捕捉个体细节，以及样本规模有限。

---

## 127. A measurement substrate for agentic Kubernetes operations: Methodology and a case study in retrieval-compounding falsification

**arXiv ID:** 2605.23058 | [PDF](https://arxiv.org/pdf/2605.23058v1)

**作者:** Joshua Odmark `[一作]` (Independent), Deon van der Vyver `[通讯]` (Cognyx)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

提出了一种闭环测量框架，旨在为Kubernetes操作代理的实证声明提供可证伪的数字，通过故障注入观察自主代理的响应，并根据真实情况对响应进行评分。

**💡 创新点**

创新点在于引入了agent-breakage框架，该框架能够区分框架错误和推理错误，并支持预注册的决策矩阵，从而提高了测量的可靠性。

**🔧 技术方法**

使用了闭环测量框架，故障注入机制，以及基于历史事件的检索增强推理。

**📊 数据集**

使用了Kubernetes集群的故障注入数据集，具体场景包括不同的故障类型和代理的响应。

**📈 对比分析**

与传统方法相比，该框架通过控制实验和预注册决策矩阵，发现了三个方法论病态，结果显示在某些场景下检索确实有积极效果，但整体效果不显著，且存在样本偏差和小样本估计的影响。

**⚠️ 局限性**

限制在于该框架目前仅在单个Kubernetes集群上运行，无法处理多集群故障模式，且对应用级故障的注入支持不足。

---

## 128. ModeSwitch-LLM: A Lightweight Phase-Aware Controller for Cross-Mode LLM Inference on a Single GPU

**arXiv ID:** 2605.23057 | [PDF](https://arxiv.org/pdf/2605.23057v1)

**作者:** Aman Sunesh `[一作]` (New York University Abu Dhabi), Hivansh Dhakne `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级请求边界控制器ModeSwitch-LLM，通过将每个请求路由到适当的固定推理模式，来提高单GPU大语言模型的推理效率。

**💡 创新点**

创新点在于使用简单的工作负载特征选择不同的推理模式，而不是依赖于单一的静态服务配置，从而在不修改模型架构或重新训练模型的情况下恢复推理效率。

**🔧 技术方法**

使用了轻量级的请求边界控制器，结合了FP16、量化模式、推测解码和混合模式等技术。

**📊 数据集**

在Meta-Llama-3.1-8B-Instruct模型上进行评估，使用了合成部署风格的工作负载和自动基准工作负载。

**📈 对比分析**

与FP16基线相比，在线控制器在合成工作负载上实现了2.10倍的平均延迟加速和0.48倍的能量比，且在自动基准上保持了接近FP16的准确性，平均增量为+0.17个百分点。

**⚠️ 局限性**

限制在于评估仅集中在单一目标部署设置上，即在单个A100 GPU上的Meta-Llama-3.1-8B-Instruct模型，未来工作应在更多模型、GPU类型和生产请求跟踪中测试控制器。

---

## 129. Model Collapse as Cultural Evolution

**arXiv ID:** 2605.23054 | [PDF](https://arxiv.org/pdf/2605.23054v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22492 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大语言模型（LLMs）在自我训练过程中出现的模型崩溃现象，提出了文化进化理论作为解释，并通过对LLaMA-2-7B和Mistral-7B模型进行10代的自我训练实验，验证了五个可证伪的预测。

**💡 创新点**

创新点在于将迭代学习理论应用于自然语言处理，提供了对模型崩溃现象的语言学解释，并提出了五个可证伪的预测，这些预测在实验中得到了验证，尤其是非单调的组合性轨迹。

**🔧 技术方法**

使用了迭代学习理论和自我训练技术，具体实现了LLaMA-2-7B和Mistral-7B模型的自我训练，并通过不同的过滤条件测试了压缩-交流权衡。

**📊 数据集**

使用了两个数据集：自然种子数据集（来自Pile的50,000个文本段落）和正则化种子数据集（通过PCFG生成的50,000个段落）。

**📈 对比分析**

通过与人类行为数据的比较，验证了模型的正则化梯度与人类的行为数据高度匹配（R^2 = 0.94）。所有五个预测均得到了确认，且效果大小显著（Hedges' g > 1.6; BF_10 > 100）。

**⚠️ 局限性**

局限性包括：理论框架是经验驱动的结构对应，而非正式数学等价；构造多样性和组合性仅在英语中测试；任务导向评估器是单向质量过滤，而非互动对话伙伴；测试的模型参数较小，未来需要在更大模型上验证；人类比较使用了已发布的数据，未来应进行平行实验；质量过滤将多个任务的评分聚合为单一信号，未来工作应考虑不同任务组合的影响。

---

## 130. The TIME Machine: On The Power of Motion for Efficient Perception

**arXiv ID:** 2605.23045 | [PDF](https://arxiv.org/pdf/2605.23045v1)

**作者:** Mantas Skackauskas `[一作]` (University of Edinburgh), Laura Sevilla-Lara `[通讯]` (University of Edinburgh)

**通讯引用:** 1830 | [OpenAlex ID](https://openalex.org/A5076169111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aaccfe5c-6b26-4208-b23c-35331481e142` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种自监督视频表征方法，仅使用视频中的点轨迹（motion tracks）作为输入，通过掩码自编码器（masked autoencoder）学习时间感知嵌入。

**💡 创新点**

创新点在于：① 只用合成的运动轨迹进行预训练，显著降低所需数据量（相当于 4 个数量级更少）；② 通过仅关注运动信息摆脱了语言监督的限制，提升了对时间推理任务的表现。

**🔧 技术方法**

技术手段包括：点轨迹到 token 的转换（局部/全局位移、空间聚合、遮挡信息）；基于 VideoMAE 的 factorized spatiotemporal attention 架构；Huber 损失和加权训练；以及利用 Kubric 生成的 250k 合成视频进行预训练。

**📊 数据集**

主要使用的数据集为：合成的 Kubric point-tracking 数据（250k 视频）做预训练；在真实数据上进行零样本评估，使用 SSv2、CLEVRER、Ego-Exo4D、Diving48 等标准视频基准。

**📈 对比分析**

与 VideoMAE‑v2、RVM、VideoMAE 等基线在时间推理任务（SSv2 Arrow‑of‑Time、CLEVRER 计数/检测）进行线性探测比较，零样本下性能与甚至超越了训练数据量多达 4 个数量级的模型；与外观模型融合后，在 SSv2、Ego‑Exo4D 等通用视觉任务中提升了多达 18% 的准确率。

**⚠️ 局限性**

局限性包括：只学习时间信息，缺少对外观的完整建模；依赖外部点跟踪算法，跟踪误差在快速运动或模糊区域可能影响性能；尚未实现与外观模型的紧耦合，融合方式仍相对简单。

---

## 131. Inductive Deductive Synthesis: Enabling AI to Generate Formally Verified Systems

**arXiv ID:** 2605.23109 | [PDF](https://arxiv.org/pdf/2605.23109v1)

**作者:** Shubham Agarwal `[一作]` (University of California Berkeley), Mohsen Lesani `[通讯]` (University of California Santa Cruz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的方法Inductive Deductive Synthesis (IDS)，用于从规范中联合合成代码和证明，解决了现有编码代理在形式验证方面的不足。

**💡 创新点**

IDS是首个在部分证明oracle下联合合成代码和机器检查证明的通用技术，能够在同一循环中利用失败反馈和性能基准进行优化。

**🔧 技术方法**

使用了基于大型语言模型（LLM）的多代理系统，结合了演绎合成和归纳合成的策略。

**📊 数据集**

使用了七个分布式键值存储一致性规范的数据集，包括Chapar的已发布因果一致性规范和六个新发布的规范。

**📈 对比分析**

与现有的SOTA编码代理（Codex和Claude Code）相比，IDS在约6.8小时内成功解决了所有7个规范，而这两个代理仅解决了2个，且IDS的实现性能比手写专家参考高出3倍。

**⚠️ 局限性**

主要限制在于需要提供正式的规范作为输入，撰写完整的正式规范需要大量的专家努力，且目前的评估仅覆盖分布式键值存储问题，其他领域的验证仍需未来工作进行。

---

## 132. Philosophical Dispositions as Behavioral Constraints for AI-Assisted Code Review: An Empirical Study

**arXiv ID:** 2605.23108 | [PDF](https://arxiv.org/pdf/2605.23108v1)

**作者:** Kaushal Bansal `[一作]` `[通讯]` (Salesforce), Kaushal Bansal (Salesforce)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种通过哲学倾向约束AI代码审查行为的系统，旨在解决AI审查工具在分析类型上的同质化问题。

**💡 创新点**

创新点在于引入了基于特定认识论传统的哲学倾向，形成了多视角分析框架，使得AI审查能够关注不同类型的问题。

**🔧 技术方法**

使用了基于哲学传统的倾向系统，结合了角色协议来组织不同的审查视角。

**📊 数据集**

使用了50个合并的拉取请求（PR），涵盖7个代码库和5种编程语言（Python、Go、C++、Java、Terraform）。

**📈 对比分析**

与通用AI审查方法的比较显示，51%的倾向发现是通用审查未能识别的独特发现，且倾向系统与人类审查者的收敛率为46%。

**⚠️ 局限性**

限制在于未进行评估的评审者间一致性，且系统在捕捉某些领域知识和重构建议方面存在缺口。

---

## 133. YASPS: A Symbolic Framework for Extensible, High-Performance IPC Simulation

**arXiv ID:** 2605.23088 | [PDF](https://arxiv.org/pdf/2605.23088v1)

**作者:** Xuan Tang `[一作]` (University of California San Diego), Tzumao Li `[通讯]` (University of California San Diego)

**通讯引用:** 2090 | [OpenAlex ID](https://openalex.org/A5030293104)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 YASPS——一种利用可微关系运算符实现可扩展、高性能 IPC 物理仿真的符号框架，支持在 Python 中声明几何原语、参数化和能量，然后自动生成 GPU 核心、梯度、Hessian 并求解 Newton 系统。

**💡 创新点**

核心创新在于引入两种关系运算符 <rel> 与 <alt>，可显式表达几何连接和多种参数化，并在符号层面完成一阶/二阶微分、稀疏模式推导与块结构组装，从而消除传统实现中因不同参数化导致的指数级内核爆炸。

**🔧 技术方法**

技术手段包括：符号微分（两遍求梯度/ Hessian），关系运算符的可微实现，基于符号图的 JIT GPU 代码生成，块稀疏 Hessian 存储与压缩，GPU 上的预条件共轭梯度求解。

**📊 数据集**

论文主要通过合成示例（软体兔子与仿射体兔子碰撞、布料变形、张力仿真等）进行验证，并未使用公开数据集，而是自行构造的几何场景。

**📈 对比分析**

在 IPC 基准上与 GIPC、PolyFEM 等手工优化实现对比，YASPS 在保持可扩展性的同时，GPU 运行时间与现有最先进实现相当或略优，且新增能量/参数化仅需约 130 行 Python 代码。

**⚠️ 局限性**

局限性包括：代码生成未充分利用寄存器重用与高级 CSE；显式 Jacobian 与 Hessian 产生额外内存占用；只能处理固定 arity 关系，无法动态扩展邻接；缺乏动态尺寸属性支持；未实现完整的逆向仿真。

---

## 134. Remind Me To Check The Stove Before I Leave The House: Authoring Personalized Context-Aware Smart Home Reminders Using Everyday Language

**arXiv ID:** 2605.23085 | [PDF](https://arxiv.org/pdf/2605.23085v1)

**作者:** Reina Szeyi Chan `[一作]` (Northeastern University), Xiang Zhi Tan `[通讯]` (Northeastern University)

**通讯引用:** 589 | [OpenAlex ID](https://openalex.org/A5057463399)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过自然语言和对话交互来创建个性化的上下文感知智能家居提醒的系统管道。该管道将用户请求转换为结构化表示和可执行逻辑，支持基于时间、活动、传感器和状态的条件。

**💡 创新点**

创新点在于通过对话引导用户创建复杂的上下文感知提醒，而不是要求用户直接构建复杂的函数。该系统利用大型语言模型（LLM）来帮助用户精炼不明确的请求，并将其与智能家居环境的能力对齐。

**🔧 技术方法**

使用了大型语言模型（LLM）作为对话助手，支持用户通过自然语言创建提醒，并将其转换为结构化的函数表示。

**📊 数据集**

进行了两项研究，第一项研究分析了233个用户创建的提醒，第二项研究评估了经过改进的系统，参与者为10人，使用的场景包括日常生活中的六种情况。

**📈 对比分析**

与传统方法相比，经过改进的系统在处理时间、活动、传感器和状态机条件的能力上有显著提高，准确率从45.5%提升至76.7%。

**⚠️ 局限性**

限制在于该研究主要集中在提醒创建的过程，而未考虑提醒使用的完整生命周期，包括如何交付、审查、修改或调试提醒。未来的工作应在真实家庭环境中进行部署，以更好地理解上下文感知提醒的实际应用。

---

## 135. BYOT-CPS: A Hybrid Cyber-Physical Systems Testbed for IoT Security Assessment and Platform Evaluation

**arXiv ID:** 2605.23059 | [PDF](https://arxiv.org/pdf/2605.23059v1)

**作者:** Yan Lin Aung `[一作]` (University of Derby), Nelson Che Neba `[通讯]` (University of Derby)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一个混合的网络物理系统测试平台BYOT-CPS，旨在解决物联网安全评估中的方法论差距，结合真实的物联网设备与虚拟化网络基础设施。

**💡 创新点**

创新点在于定义了一个混合架构，支持真实设备的集成与评估，同时满足六个设计要求：真实性、多样性、可扩展性、可重复性、可扩展性和独立性。

**🔧 技术方法**

使用了GNS3作为网络仿真平台，结合真实的物联网设备进行实验。

**📊 数据集**

使用了包括智能灯泡、智能插头、开关和IP摄像头在内的真实物联网设备进行原型部署。

**📈 对比分析**

与现有方法相比，BYOT-CPS在真实设备行为、攻击工具、复杂拓扑、设备和协议、多样化攻击、可扩展性等方面表现出色，提供了更高的实验真实性和灵活性。

**⚠️ 局限性**

局限性包括物理设备集的规模有限，尚未覆盖所有企业和工业物联网系统；缺乏跨平台的实证验证；以及平台评估的系统性基准测试超出了当前工作的范围。

---

## 136. Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection

**arXiv ID:** 2605.23036 | [PDF](https://arxiv.org/pdf/2605.23036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 137. DRL-Driven Edge-Aware Utility Optimization for Multi-Slice 6G Networks

**arXiv ID:** 2605.23056 | [PDF](https://arxiv.org/pdf/2605.23056v1)

**作者:** Khaled M. Naguib `[一作]` (New Giza University), Ibrahim I. Ibrahim `[通讯]` (Helwan University)

**通讯引用:** 15105 | [OpenAlex ID](https://openalex.org/A5090355098)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于深度强化学习的边缘感知效用优化框架，旨在为多切片6G网络提供智能资源分配和边缘缓存。

**💡 创新点**

创新点在于结合了深度Q网络（DQN）学习，优化了边缘缓存和动态资源配置，特别针对虚拟现实（VR）应用的低延迟和高带宽需求。

**🔧 技术方法**

使用了深度Q网络（DQN）技术进行资源分配优化。

**📊 数据集**

在模拟的6G O-RAN网络中进行评估，包含7个基站和42个移动用户，用户根据服务需求分配到不同的网络切片（eMBB、URLLC、MBRLLC）。

**📈 对比分析**

与传统方法相比，DQN框架在降低延迟和提高吞吐量方面表现出色，能够更好地支持VR应用的需求。

**⚠️ 局限性**

局限性在于需要在非实时RIC中进行离线训练，尽管推理阶段的计算开销较低，但仍需确保在极端低延迟的6G服务中有效部署。

---

## 138. StanBKT: Rethinking Parameter Estimation in Bayesian Knowledge Tracing

**arXiv ID:** 2605.23048 | [PDF](https://arxiv.org/pdf/2605.23048v1)

**作者:** Siddhartha Pradhan `[一作]` (Worcester Polytechnic Institute), Adam C. Sales `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文介绍了StanBKT，一个基于Stan的Python包，旨在为贝叶斯知识追踪（BKT）模型提供完整的贝叶斯推断。该包支持模型参数的灵活估计和不确定性量化，克服了现有BKT实现中仅依赖点估计的局限性。

**💡 创新点**

StanBKT的创新点在于结合了经典BKT模型的可解释性与贝叶斯方法的灵活性和不确定性量化能力，使得研究人员能够进行更丰富的统计推断和参数比较。

**🔧 技术方法**

使用了贝叶斯推断技术，包括马尔可夫链蒙特卡洛（MCMC）、变分推断和最大后验估计（MAP），以支持对BKT模型的参数进行全面的推断。

**📊 数据集**

使用了ASSISTments 2020数据集和一个开放源代码的感知线索干预研究数据集，进行大规模和实验性评估。

**📈 对比分析**

与传统的期望最大化（EM）方法相比，StanBKT在预测性能上表现相似，但在计算效率上，变分推断和MAP方法显著快于MCMC，适合不同的分析需求。

**⚠️ 局限性**

StanBKT的局限性在于，贝叶斯推断（特别是MCMC）在计算和内存上比EM实现更为密集，尤其是在大规模或高粒度分析中，可能导致内存消耗过大。

---

## 139. A Comparative Evaluation of Structural Topic Models and BERTopic for Short, Open-Ended Survey Responses

**arXiv ID:** 2605.23093 | [PDF](https://arxiv.org/pdf/2605.23093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 140. Open Multimodal Datasets and Open-Source Software for Data-Driven Modeling of Multiphase Transport and Thermal Systems

**arXiv ID:** 2605.23037 | [PDF](https://arxiv.org/pdf/2605.23037v1)

**作者:** Christy Dunlap `[一作]` (University of Arkansas), Han Hu `[通讯]` (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了一个开源的多模态热流体数据集与软件生态系统，并提出了S+TD维度分类框架。

**💡 创新点**

创新在于将数据集与对应工具配对发布，提出S+TD框架统一描述数据维度，推动可复现的AI热流体研究。

**🔧 技术方法**

采用Python、机器学习库（如SeqReg、BubbleID、CFDTwin）、图像处理、序列回归、数字孪生、声学信号解码等技术。

**📊 数据集**

利用NED3公开数据集，包括高频视频、红外热像、热流、声发射、CFD模拟、设计文件等多模态S+TD样本。

**📈 对比分析**

通过预设基准任务（如声学/图像回归预测热流、泡沫分割等）使用SeqReg、BubbleID等实现，指标如MAE、RMSE等显示模型在不同S+TD维度下达到数百分比误差，表明方法有效。

**⚠️ 局限性**

局限包括数据的实验专属性、跨实验同步难度、标签推断不确定、数据量大以及部分专有格式解码困难。

---

## 141. Sparse Autoencoders Map Brain-LLM Alignment onto Cortical Semantic Topography

**arXiv ID:** 2605.23035 | [PDF](https://arxiv.org/pdf/2605.23035v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22492 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究探讨了大型语言模型（LLMs）中间层如何最佳预测人脑对语言的反应，并通过稀疏自编码器（SAEs）与神经编码模型的结合，分解了GPT-2 XL和Llama-3.1-8B的特征。

**💡 创新点**

创新点在于通过SAE提取的语义特征能够同时解释中间层在脑-LLM对齐中的优势，并与已知的皮层语义组织相一致，提供了一个单一的特征级别解释。

**🔧 技术方法**

使用了稀疏自编码器（SAEs）技术对每个层提取16K-32K个可解释特征，并结合了神经编码模型进行分析。

**📊 数据集**

使用了公开的自然语言fMRI数据集（UTS数据集），以及自然故事语料库和Provo语料库进行行为验证。

**📈 对比分析**

与简单的词级语义规范进行比较，SAE特征在皮层拓扑映射中表现更优，SAE特征的编码性能显著高于随机基线，且在多种语言（英语、中文、法语）中具有普遍性。

**⚠️ 局限性**

限制包括：五分类法假设互斥性，GPT-4标记可能引入偏差，fMRI的时间分辨率限制，样本量小（N=8），以及自然故事可能偏向具体/社会内容等。

---

## 142. SVR-MAD: A Bayesian-Inspired Framework for Posterior-Guided Multi-Agent Debate

**arXiv ID:** 2605.23099 | [PDF](https://arxiv.org/pdf/2605.23099v1)

**作者:** Weifan Jiang `[一作]` (Harvard University), Minlan Yu `[通讯]` (Harvard University)

**通讯引用:** 9467 | [OpenAlex ID](https://openalex.org/A5035157838)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于贝叶斯的多智能体辩论框架，通过将辩论结果作为后验证据来估计智能体的正确性，从而提高大型语言模型（LLM）智能体的准确性。

**💡 创新点**

创新点在于引入生存率（SVR）作为衡量智能体在辩论中保留其答案的能力的指标，从而提供更可靠的信号来识别可信的推理过程和答案。

**🔧 技术方法**

使用了贝叶斯方法，结合了后验证据和生存率（SVR）来逐步构建通信图，并优先考虑在辩论中表现良好的智能体。

**📊 数据集**

在多个大型语言模型（LLM）和基准数据集上进行了实验，包括GPT-OSS-120B和DeepSeek-V3.1，使用IMO-AnswerBench数据集。

**📈 对比分析**

与现有的多智能体辩论基线相比，该方法在准确性上匹配或提高，同时在通信成本上减少了高达61%。与其他方法相比，减少了48-75%的通信次数和38-61%的总令牌数。

**⚠️ 局限性**

限制在于SVR作为正确性的代理，不能保证绝对的正确性；在开放式生成任务中，答案的等价性难以判断；方法的有效性可能依赖于辩论提示、智能体数量和模型家族。

---

## 143. The Attribution Contract: Feature Attribution for Generative Language Models

**arXiv ID:** 2605.23080 | [PDF](https://arxiv.org/pdf/2605.23080v1)

**作者:** Giang Nguyen `[一作]` `[通讯]` (Guide Labs), Giang Nguyen (Guide Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“归因合约（Attribution Contract）”框架，用以明确生成式语言模型中特征归因的目标、可归因特征、生成过程、固定条件和归因得分；并指出传统归因方法在生成式模型中可能导致自归因谬误（self‑attribution fallacy）

**💡 创新点**

创新点在于将特征归因问题拆解为契约化的问答：通过五元组（SCORE, CONDITION, TARGET, PROCESS, FEATURES）系统描述归因场景，使不同方法与契约一一对应，消除方法间看似冲突的误解；提出按契约评估归因方法的必要性

**🔧 技术方法**

采用已有归因算法（如Integrated Gradients等）作为示例，但核心技术为对归因问题进行契约化建模与细分（包括自回归与扩散模型的多种契约）

**📊 数据集**

论文未使用特定数据集，而是在通用生成式语言模型场景（自回归LM与扩散LM）下进行案例说明与理论分析

**📈 对比分析**

方法比较主要通过概念层面的契约匹配和针对不同契约设计的评估标准（如删除/插入、阶段扰动等），未给出定量性能指标；但强调评价必须契约特定，避免对同一方法在不同契约下误判

**⚠️ 局限性**

局限性：框架高度概念化，缺少实证验证；对具体归因算法的改进未给出；在实践中需要明确契约细节，才能有效应用；同时，扩散模型的阶段归因仍需更高效的实现方法

---

## 144. GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs

**arXiv ID:** 2605.23078 | [PDF](https://arxiv.org/pdf/2605.23078v1)

**作者:** Jianing Deng `[一作]` (University of Pittsburgh), Jingtong Hu `[通讯]` (University of Pittsburgh)

**通讯引用:** 4640 | [OpenAlex ID](https://openalex.org/A5066534595)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 GEMQ，全球专家级混合精度量化方法，结合全局重要性评估、路由器微调和渐进量化，显著降低 MoE-LLM 的内存占用与推理延迟。

**💡 创新点**

创新点在于：① 用全局线性规划基于量化误差估计专家重要性，突破层级局部分配的局限；② 对路由器进行全局微调，解决量化导致的路由失配；③ 引入渐进量化策略，用近似量化模型提升低比特量化时的重要性估计。

**🔧 技术方法**

技术包括：全局混合精度量化（LP 求解）、GPTQ 量化、路由器微调（PEFT）、量化误差的 FIM 近似、HQQ 位包装、Triton 高效低比特推理核。

**📊 数据集**

使用了多种 MoE-LLM（DeepseekV2-Lite、Qwen1.5-MoE-A2.7B、Qwen3-30B-A3B、Mixtral-8×7B）与通用语料 C4、WikiText2、以及针对数学推理的 GSM8K 进行校准与评估。

**📈 对比分析**

与统一量化、PMQ、SpQR、EAQuant、MoEQuant 等基线比较，GEMQ 在 2.5‑3.0 bpe 方案下，混合精度模型在 5‑10% 以内保持相同或更优的困惑度与零样本准确率，并在 1.5‑2.5 bpe 下实现 80‑90% 的内存压缩与 1.5‑2.5× 的推理加速。

**⚠️ 局限性**

局限包括：对校准数据集的敏感性，需要足量且与目标任务相似的数据；梯度计算与路由器微调在极大模型上会产生较高的 GPU 内存占用，可能需多 GPU 或显存压缩手段。

---

## 145. Improved Torn Paper Coding via Local Alignment

**arXiv ID:** 2605.23076 | [PDF](https://arxiv.org/pdf/2605.23076v1)

**作者:** Junsheng Liu `[一作]` (Washington University in St Louis), Netanel Raviv `[通讯]` (Washington University in St Louis)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种改进的撕纸编码方案，通过局部对齐来提高传输速率，解决了传统方法中对全局统计信息的依赖问题。

**💡 创新点**

创新点在于引入局部对齐的概念，使得解码器能够利用局部信息识别短片段的位置，从而提高了信息提取的效率和传输速率。

**🔧 技术方法**

使用了局部对齐技术和运行长度限制（RLL）编码，结合了De Bruijn序列作为引导序列。

**📊 数据集**

使用了撕纸信道模型和带有丢失片段的撕纸编码（TPC-LP）模型进行分析。

**📈 对比分析**

与之前的方案相比，提出的方法在处理短片段时能够有效提取信息，显著提高了传输速率，并且在理论上能够达到接近信道容量的性能。

**⚠️ 局限性**

局部对齐方法在处理极短片段时可能仍然存在一定的局限性，尤其是在片段丢失概率较高的情况下。

---

## 146. The Efficiency Frontier: A Unified Framework for Cost-Performance Optimization in LLM Context Management

**arXiv ID:** 2605.23071 | [PDF](https://arxiv.org/pdf/2605.23071v1)

**作者:** Binqi Shen `[一作]` (Northwestern University), Yuting Xin `[通讯]` (University of Minnesota)

**通讯引用:** 14069 | [OpenAlex ID](https://openalex.org/A5115590096)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的框架——效率边界，用于在大语言模型的上下文管理中进行成本与性能的优化。

**💡 创新点**

创新点在于将上下文策略选择建模为一个考虑部署的优化问题，能够系统地比较不同的上下文管理策略，并提供决策导向的分析。

**🔧 技术方法**

使用了参数化的效用函数和分阶段的优化程序来评估上下文管理策略的性能与成本之间的权衡。

**📊 数据集**

在HotpotQA数据集上进行了评估，该数据集包含多跳推理的实例，适合用于上下文选择和减少策略的评估。

**📈 对比分析**

通过效率边界分析，结果显示在相似性能下，部署感知优化可以将有效的token使用量减少约25%，而在高性能设置下，使用摊销的内存压缩方法可以实现超过50%的token成本降低。

**⚠️ 局限性**

限制在于当前框架假设了固定的效用函数，未来的工作可以考虑自适应或学习的偏好模型，以更好地捕捉特定应用的部署优先级。

---

## 147. CultivAgents: Cultivating Relationship-Centered Multi-Agent Systems for Personalized Gardening

**arXiv ID:** 2605.23193 | [PDF](https://arxiv.org/pdf/2605.23193v1)

**作者:** Yiyang Wang `[一作]` (Georgia Institute of Technology), Josiah Hester `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 2158 | [OpenAlex ID](https://openalex.org/A5026852792)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了CultivAgents，一个以关系为中心的多代理系统，为用户提供个性化的、社会文化基础的园艺支持。

**💡 创新点**

创新点在于通过协调经验、环境和民族植物学代理，提供个性化的园艺指导，超越了传统的通用建议。

**🔧 技术方法**

使用了多代理系统和大型语言模型（LLM）技术，结合了用户的经验水平、环境条件和文化背景。

**📊 数据集**

通过三阶段的混合方法研究，参与者包括3名领域专家、7名人机交互研究者和5名社区园丁，收集了反馈和调查数据。

**📈 对比分析**

与传统的单一代理系统相比，CultivAgents在增强用户信心、动机和对AI建议的信任方面表现出显著提升，用户信心从3.00提高到3.60，动机从4.00提高到4.40。

**⚠️ 局限性**

局限性包括文化特异性、生态基础和代理协调的不足，专家反馈指出系统在某些文化背景下的建议过于表面化，缺乏地方性知识的深度。

---

## 148. DRIVESPATIAL: A Benchmark for Spatiotemporal Intelligence in VLMs for Autonomous Driving

**arXiv ID:** 2605.23176 | [PDF](https://arxiv.org/pdf/2605.23176v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 149. SpikingMoE: SDPrompt-Guided Dynamic Expert Fusion in Spiking Neural Networks

**arXiv ID:** 2605.23188 | [PDF](https://arxiv.org/pdf/2605.23188v1)

**作者:** Yukai Yang `[一作]`, Liqun Chen `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 SpikingMoE 的新型脉冲神经网络框架，将混合专家（MoE）与脉冲驱动的 Transformer 结合，实现可动态路由的模块化计算。

**💡 创新点**

创新点在于：1) 通过受侧核网（LGN）启发的脉冲驱动提示（SDprompt）实现输入依赖的专家路由；2) 将传统 MLP 替换为全脉冲兼容的专家模块，保证仅使用二进制脉冲通信；3) 设计了适用于神经形态硬件的稀疏二进制运算与辅助路由损失，提升专家平衡与多样性。

**🔧 技术方法**

技术包括：脉冲驱动自注意力（SDSA）取代 Softmax；Leaky Integrate‑and‑Fire（LIF）脉冲神经元；二进制脉冲门控与 Hadamard 乘法；基于 SDprompt 的门控与 top‑k 专家选择；辅助路由损失结合负载平衡与熵项。

**📊 数据集**

在静态图像数据集 CIFAR‑10、CIFAR‑100，以及事件数据集 CIFAR10‑DVS 和 Gesture‑DVS 上进行实验。

**📈 对比分析**

与现有 SNN 及 Transformer 方案相比，SpikingMoE 在 CIFAR‑10 达到 94.09% 的 top‑1 准确率，在 CIFAR‑100 达到 74.54%，在事件数据集 Gesture‑DVS 达到 95.83%，性能与最先进的 ResNet‑based SNNs 相当，同时保持了脉冲计算的高能效。

**⚠️ 局限性**

局限性包括：1) MoE 引入的路由复杂度导致部分数据集精度略低；2) 对 SDprompt 的依赖使得模型对输入上下文的鲁棒性需进一步验证；3) 目前仅在小规模图像与事件数据集上验证，缺乏对大规模视觉任务的扩展与能耗分析。

---

## 150. Self-Improving In-Context Learning

**arXiv ID:** 2605.23180 | [PDF](https://arxiv.org/pdf/2605.23180v1)

**作者:** Baturay Saglam `[一作]` (Yale University), Dionysis Kalogerias `[通讯]` (Yale University)

**通讯引用:** 359 | [OpenAlex ID](https://openalex.org/A5091493591)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通过优化固定少量示例提示的连续嵌入来改进上下文学习（ICL）的方法，利用模型对示例输出的对数概率作为优化信号。

**💡 创新点**

创新点在于提出了一种自监督的置信度代理，通过零阶优化在测试时对提示嵌入进行校准，且不需要微调、生成标记或外部数据。

**🔧 技术方法**

使用了零阶优化技术来估计输入嵌入的梯度，并通过前向传递更新嵌入。

**📊 数据集**

使用了ICL评估基准数据集，包括12个任务（2040个样本），这些任务围绕精确复制和规则学习的能力进行设计。

**📈 对比分析**

与现有的分类特定基线相比，提出的方法在大多数任务上表现更好，且在所有模型中均未降低基础模型的准确性，统计上显著提高了准确性。

**⚠️ 局限性**

限制在于需要已知输出跨度的位置和示例标签的正确性，且未在少于三个示例的提示上进行验证。

---

## 151. Cognitive offloading and the speedup illusion in human-AI interaction

**arXiv ID:** 2605.23177 | [PDF](https://arxiv.org/pdf/2605.23177v1)

**作者:** Sunny Yu `[一作]` (Stanford University), Robert D. Hawkins `[通讯]` (Stanford University)

**通讯引用:** 31433 | [OpenAlex ID](https://openalex.org/A5041689299)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

进行了大规模的行为研究，探讨用户在使用大型语言模型（LLMs）时对任务完成时间的预期与实际完成时间之间的差异，特别关注简单的认知任务。

**💡 创新点**

发现了一个速度幻觉，即人们对独立完成时间的预测准确，但对AI辅助完成时间的预测显著低估，表明人们在使用AI时存在系统性偏差。

**🔧 技术方法**

使用了线性混合效应模型来分析预测和实际完成时间的差异，并通过NASA-TLX量表评估主观努力。

**📊 数据集**

使用了包含24个任务的自定义数据集，这些任务涵盖了四类认知工作，从简单的信息检索到内容创作，参与者总数为1237。

**📈 对比分析**

与其他方法比较时，发现AI辅助完成时间与独立完成时间没有显著差异，但参与者对AI的时间节省有过高的预期，且AI辅助任务的主观努力感知显著降低。

**⚠️ 局限性**

研究的局限性包括未控制参与者的动机或激励机制，且参与者在使用AI时的个体差异较大，未来研究应关注AI使用的具体复杂性和个体差异对结果的影响。

---

## 152. Effective information gathering for ore estimation, evaluation and perspectives on adaptive sampling

**arXiv ID:** 2605.23172 | [PDF](https://arxiv.org/pdf/2605.23172v1)

**作者:** Raymond Leung `[一作]` (University of Sydney), Arman Melkumyan `[通讯]` (University of Sydney)

**通讯引用:** 1007 | [OpenAlex ID](https://openalex.org/A5030234520)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了基于高分辨率模拟数据的高斯过程框架，用来评估钻孔信息对矿石品位预测的价值，并提出了两种实验场景来比较不同采样策略；

**💡 创新点**

创新点在于不依赖传统条件模拟或假设平稳性，采用多指标评估（JSD、F、I、CV、RMSE、SSIM、CRPS）来衡量采样密度和自适应采样在复杂地质中的效果；

**🔧 技术方法**

使用了高斯过程回归、最大方差、目标特征、目标复杂度和不确定度减少等自适应采样算法；

**📊 数据集**

使用了澳大利亚Hamersley/Fortescue组高分辨率矿石品位模拟数据，尺寸约1600×2500m、80m深度；

**📈 对比分析**

通过对比随机、网格、最大方差等多种策略，实验表明目标复杂度自适应采样可将钻孔需求降低约30%，并在RMSE、SSIM、CRPS上显著优于传统网格采样；

**⚠️ 局限性**

局限性包括实验仅基于单一模拟模型，起始采样配置影响性能，且在更大规模或不同矿体上效果未完全验证；

---

## 153. Same Model, Different Weakness: How Language and Modality Reshape the Jailbreak Attack Surface in Frontier MLLMs

**arXiv ID:** 2605.23157 | [PDF](https://arxiv.org/pdf/2605.23157v1)

**作者:** Casey Ford `[一作]` (Appen), Emily Dix `[通讯]` (Appen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究进行了首次系统的跨语言多模态红队评估，比较了美国英语（en-US）和墨西哥西班牙语（es-MX）中四种前沿多模态大型语言模型（MLLM）的越狱脆弱性。

**💡 创新点**

研究的创新点在于揭示了语言依赖性对模型攻击表面的影响，发现语言切换会导致语言攻击技术的有效性降低，而视觉攻击的有效性则提高，表明语言和视觉对齐失败是通过不同机制运作的。

**🔧 技术方法**

使用了贝叶斯混合效应模型进行数据分析，结合了来自九位母语评估者的52,272个伤害评分和二元攻击成功判断。

**📊 数据集**

使用了363个多样化的对抗性提示场景，这些场景在文本和多模态条件下进行评估，涵盖了三种伤害类别：非法活动、虚假信息和不道德行为。

**📈 对比分析**

与之前的研究相比，模型的脆弱性在不同语言条件下表现出显著差异，尤其是Qwen Omni在es-MX条件下的脆弱性超过了Pixtral Large，表明安全排名在语言间并不一致。

**⚠️ 局限性**

研究的局限性包括所有评估通过模型API进行，无法查看训练数据和对齐过程；对抗性基准覆盖的提示场景固定，未考虑多步骤和动态交互的潜在脆弱性；以及提示翻译而非由文化内部人士创作，可能影响结果的文化适应性。

---

## 154. Orchestrating Data Collection and Computation in Green IoT Networks

**arXiv ID:** 2605.23152 | [PDF](https://arxiv.org/pdf/2605.23152v1)

**作者:** Junfei Zhan `[一作]` (Jinan University-University of Birmingham Joint Institute), Fei Song `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 7587 | [OpenAlex ID](https://openalex.org/A5068295637)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种新的混合整数线性规划（MILP）模型，用于在太阳能供电的物联网网络中调度和嵌入应用程序，优化设备的采样时间、应用程序的运行与否以及设备、网关和服务器的能耗。

**💡 创新点**

创新点在于首次将最大服务年龄（AoS）作为性能指标，并提出了两种新颖的解决方案：基于递归控制的RHC方法和贪婪嵌入方法GreedyOL。

**🔧 技术方法**

使用了混合整数线性规划（MILP）、递归控制（RHC）和贪婪算法等技术。

**📊 数据集**

使用了随机生成的太阳能到达过程数据集，包含设备、网关和服务器的能量、计算和通信资源。

**📈 对比分析**

与现有方法（如随机选择和GMM预处理）进行比较，结果显示RHCOP和GreedyOL的最小最大AoS分别比MILP高出1.07倍和1.13倍，且增加网关和服务器数量有助于降低最小最大AoS。

**⚠️ 局限性**

限制在于MILP在大规模网络中变得不可解，且需要非因果信息来解决问题，这在实际应用中并不实用。

---

## 155. The Impact of AI Coding Assistants on Software Engineering: A Longitudinal Study

**arXiv ID:** 2605.23135 | [PDF](https://arxiv.org/pdf/2605.23135v1)

**作者:** Annie Vella `[一作]` (University of Auckland), Kelly Blincoe `[通讯]` (University of Auckland)

**通讯引用:** 2590 | [OpenAlex ID](https://openalex.org/A5039326697)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过纵向混合方法调查了专业软件工程师对AI编码助手在任务专注、开发者体验和生产力方面的影响。研究中使用了两次问卷调查，时间间隔为六个月，共有158名参与者在第一次调查中合格，第二次为101名，最终形成95名匹配的纵向队列。

**💡 创新点**

提出了一个新的工作类别，称为监督工程工作，涵盖了对AI输出的指导、评估和纠正。此外，发现了生产力与开发者体验之间的悖论：尽管84%的参与者在两个时间点都报告了生产力的提高，但在匹配的参与者中，报告在至少一个维度上体验恶化的比例几乎翻倍，从14%增加到27%。

**🔧 技术方法**

采用了纵向混合方法，结合定量和定性分析，通过问卷调查收集数据，分析了参与者对AI编码助手的使用体验和感知变化。

**📊 数据集**

使用了两次问卷调查的数据，第一次调查有158名合格参与者，第二次为101名，最终形成95名匹配的纵向队列，参与者来自28个国家。

**📈 对比分析**

与传统方法相比，参与者报告在大多数开发任务上花费的时间减少，82%的人表示在编写代码上花费的时间减少。尽管生产力感知保持稳定，但开发者体验在某些维度上有所下降，特别是在流状态和认知负荷方面。

**⚠️ 局限性**

研究的局限性包括样本的选择偏差，仅包括当前使用AI编码助手的工程师，未考虑那些尝试后放弃的工程师。此外，研究结果反映了2024年末和2025年初的AI编码助手使用情况，快速变化的工具环境可能影响结果的普遍适用性。

---

## 156. $π_0$-EqM: Equilibrium Matching for Closed-Loop Vision-Language-Action Control

**arXiv ID:** 2605.23128 | [PDF](https://arxiv.org/pdf/2605.23128v1)

**作者:** Huanming Liu `[一作]` (University of Science and Technology of China), Yao Mu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 12629 | [OpenAlex ID](https://openalex.org/A5008178136)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的解码器π_0-EqM，用于视觉-语言-动作（VLA）模型，通过将流匹配专家替换为平衡匹配解码器，改善了机器人操作的成功率。

**💡 创新点**

创新点在于引入了平衡匹配（EqM）解码器，使得动作生成可以视为迭代平衡求解，从而实现自适应停止和热启动，提升了任务的成功率。

**🔧 技术方法**

使用了平衡匹配（EqM）技术，该技术学习了一个时间无关的条件向量场，通过迭代求解来解码动作。

**📊 数据集**

使用了RoboTwin基准数据集（19个任务）和LIBERO数据集进行评估。

**📈 对比分析**

与原始的π_0模型进行比较，π_0-EqM在RoboTwin的平均成功率从40.4%提高到50.2%，在LIBERO-10上从85.2%提高到87.0%。

**⚠️ 局限性**

主要的局限性在于阈值扫描仅覆盖了两个RoboTwin任务，未能全面映射非单调现象，同时未能确定最大化任务成功的最佳阈值。

---

## 157. Scalable Heterogeneous Graph Foundation Models for Data-Driven Optimal Power Flow in Smart Grids

**arXiv ID:** 2605.23194 | [PDF](https://arxiv.org/pdf/2605.23194v1)

**作者:** Massimiliano Lupo Pasini `[一作]` (Oak Ridge National Laboratory), Teja Kuruganti `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 2900 | [OpenAlex ID](https://openalex.org/A5045320466)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种可扩展的异构图神经网络（GNN）工作流，旨在为智能电网中的数据驱动的最优潮流（OPF）建模和OPF-GFM开发提供支持。

**💡 创新点**

创新点在于开发了一个支持异构图学习的HydraGNN工作流，能够处理电网中不同类型的节点和边，并在领导级超级计算机上进行分布式训练和超参数优化。

**🔧 技术方法**

使用了HydraGNN框架，结合了异构图神经网络的多种架构，包括HeteroSAGE和HeteroHEAT，支持节点类型特定的输入嵌入和关系特定的消息传递。

**📊 数据集**

使用了来自PGLib-OPF基准案例的三百万个异构图实例，涵盖了从14到13,659个节点的电网案例。

**📈 对比分析**

通过DeepHyper驱动的超参数优化在六种异构GNN架构上进行比较，结果显示HeteroSAGE和HeteroHEAT在验证损失上表现最佳，且模型参数较少，性能优于其他架构。

**⚠️ 局限性**

限制在于尽管模型在小数据集上表现良好，但全微调过程可能导致预训练表示的灾难性遗忘，且在某些情况下，模型的稳定性和收敛速度可能受到影响。

---

## 158. Hidden Human-Like Nature of Machine-Generated Texts: Theory and Detection Enhancement

**arXiv ID:** 2605.23190 | [PDF](https://arxiv.org/pdf/2605.23190v1)

**作者:** Chenwang Wu `[一作]` (Hong Kong Baptist University), Defu Lian `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8829 | [OpenAlex ID](https://openalex.org/A5085254654)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并利用机器生成文本中隐藏的人类特征跨度，提出堆叠式增强检测框架以提升检测性能。

**💡 创新点**

理论揭示隐藏人类跨度增加检测难度，并提出标签无关的保留/过滤机制与隐变量EM启发式堆叠优化。

**🔧 技术方法**

使用隐变量EM、堆叠式双步推理、句子/段落分段、阈值过滤等技术实现检测增强。

**📊 数据集**

在 MGTBench 四个子集（Essay、Reuters、SQuAD1、DetectRL）上进行实验。

**📈 对比分析**

与 Log‑Likelihood、DetectGPT、OpenAI‑D、MPU、RADAR 等多种基线比较，AUROC 与 TPR@FPR‑0.5% 通常提升 5–15%，且跨 LLM 与跨域也表现出显著改善。

**⚠️ 局限性**

对短文本、误过滤导致的判别损失以及对高比例人类跨度的鲁棒性仍存在局限，且需手动调节阈值。

---

## 159. LQ-rPPG: A Label-Quantized Coarse-to-Fine Learning Framework for Remote Physiological Measurement

**arXiv ID:** 2605.23174 | [PDF](https://arxiv.org/pdf/2605.23174v1)

**作者:** Jun Seong Lee `[一作]` (Electronics and Telecommunications Research Institute), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 6059 | [OpenAlex ID](https://openalex.org/A5059521863)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为LQ-rPPG的框架，用于通过面部视频进行远程生理信号测量，旨在提高rPPG信号的估计精度和鲁棒性。

**💡 创新点**

创新点在于引入了标签量化的粗到细学习策略，通过将连续PPG信号转化为多位量化伪标签，减少标签噪声和变异性，从而改善模型的学习效果。

**🔧 技术方法**

使用了深度学习技术，特别是基于Mamba的架构和层次化的粗到细估计模型，结合标签量化模块进行训练。

**📊 数据集**

使用了多个公开数据集进行实验，包括PURE、UBFC、COHFACE、V4V和MMPD，这些数据集涵盖了不同的拍摄条件和生理信号。

**📈 对比分析**

与现有方法相比，LQ-rPPG在同一数据集内和跨数据集的评估中均表现出色，参数减少了88%，乘法累加操作减少了29%，吞吐量提高了191%。

**⚠️ 局限性**

限制在于该框架在极端运动条件下的表现可能不佳，无法有效处理严重的输入干扰，未来需要进一步改进以增强对复杂运动的鲁棒性。

---

## 160. Understanding and Improving Noisy Embedding Techniques in Instruction Finetuning

**arXiv ID:** 2605.23171 | [PDF](https://arxiv.org/pdf/2605.23171v1)

**作者:** Abhay Yadav `[一作]` (Johns Hopkins University), Abhay Yadav `[通讯]` (Johns Hopkins University)

**通讯引用:** 293 | [OpenAlex ID](https://openalex.org/A5049279805)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文旨在分析不同噪声类型在语言模型微调中的表现，提出了一种新的对称噪声微调方法，显著提高了模型性能。

**💡 创新点**

创新点在于引入对称伯努利分布噪声，替代传统的均匀噪声和高斯噪声，且通过理论和实证分析证明了不同噪声类型的功能等价性。

**🔧 技术方法**

使用了对称伯努利分布的噪声注入技术，进行语言模型的微调。

**📊 数据集**

使用了多个数据集，包括Alpaca、Evol-Instruct、ShareGPT和OpenPlatypus等。

**📈 对比分析**

与现有的均匀噪声微调方法相比，提出的方法在多个基准数据集上表现更优，Alpaca数据集的得分从29.79%提升至69.04%，比现有最佳方法提高了6.7%。

**⚠️ 局限性**

限制在于实验主要集中在单轮数据集，可能未能充分探索多轮对话的潜力。

---

## 161. Semantic-Aware Guided Drone Exploration for Language-Conditioned 3D Indoor Mapping

**arXiv ID:** 2605.23160 | [PDF](https://arxiv.org/pdf/2605.23160v1)

**作者:** Nitin Vegesna `[一作]` (University of California, Berkeley), Avideh Zakhor `[通讯]` (University of California, Berkeley)

**通讯引用:** 10527 | [OpenAlex ID](https://openalex.org/A5008034417)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为SAGE的系统，用于在未知的3D室内环境中进行开放词汇探索，同时保持覆盖导向行为，并允许语义线索重新优先选择前沿。

**💡 创新点**

SAGE通过结合几何和语义信息，优化了探索过程，特别是在物体发现方面表现优于传统方法。

**🔧 技术方法**

使用了对比语言-图像预训练（CLIP）技术，结合了对象中心嵌入存储、时间缓存、对象前沿和统一的语义-几何规划成本等四个关键组件。

**📊 数据集**

在Matterport3D基础的模拟环境中进行评估，使用了多个室内场景和开放词汇查询。

**📈 对比分析**

与FALCON和FTU等方法进行比较，SAGE在物体发现上表现更好，探索速度比FTU快9.0到25.9倍，平均加速比为13.7。

**⚠️ 局限性**

当前方法在视觉-惯性里程计漂移和车辆动态方面存在风险，可能导致语义记忆与真实目标位置不一致。

---

## 162. What Does the Server See? Understanding Privacy Leakage from Large Language Models in Split Inference

**arXiv ID:** 2605.23158 | [PDF](https://arxiv.org/pdf/2605.23158v1)

**作者:** Mingyuan Fan `[一作]` (East China Normal University), Cen Chen `[通讯]` (East China Normal University)

**通讯引用:** 4708 | [OpenAlex ID](https://openalex.org/A5100622590)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了在分割推理中大语言模型（LLMs）的隐私泄露问题，提出了一种攻击方法，通过重建客户端的原始输入来量化隐私泄露。

**💡 创新点**

创新点在于提出了中间激活匹配问题的解决方案，并引入了扰动放大因子（PAF）来量化层的固有抵抗力，从而系统性地理解隐私泄露的脆弱性。

**🔧 技术方法**

使用了优化算法来同时优化输入嵌入，并通过反向传播最大化重建误差，同时保持模型的实用性。

**📊 数据集**

使用了两个数据集进行评估：AlpacaEval（805条记录）和iCliniq（7321条记录），分别用于一般知识问答和医疗咨询对话。

**📈 对比分析**

与现有方法相比，本文的方法在重建质量上表现优异，尤其在面对高斯噪声注入和激活稀疏化等常见防御时，重建精度和召回率均超过98%。

**⚠️ 局限性**

限制在于尽管提出的防御方法在隐私保护上表现良好，但在极高的扰动强度下，模型的实用性会显著下降，且对不同层的敏感性分析仍需进一步研究。

---

## 163. Deception and Counter Deception in Adversarial Graph Traversal Game

**arXiv ID:** 2605.23129 | [PDF](https://arxiv.org/pdf/2605.23129v1)

**作者:** Violetta Rostobaya `[一作]` (George Mason University), Daigo Shishika `[通讯]` (George Mason University)

**通讯引用:** 769 | [OpenAlex ID](https://openalex.org/A5057044271)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了双侧不完全信息下的对抗图遍历（AGT）问题，研究移动代理在对手可以改变边权的环境中，如何在最小成本同时实现欺骗与反欺骗。

**💡 创新点**

创新点在于：①将AGT建模为两玩家零和无穷期随机最短路径游戏；②在此框架下引入专门设计的默认策略，使得扩展后的Extensive-Form Double Oracle（XDO）算法可在无限期终止的游戏中收敛；③通过价值信息（VoI）分析揭示双侧不完全信息对欺骗行为的影响。

**🔧 技术方法**

主要技术包括：强化学习框架下的扩展XDO算法（带默认策略）和最佳响应搜索；贝叶斯更新实现双方信念演化；价值信息计算用于评估信息不确定性对游戏价值的贡献。

**📊 数据集**

论文中使用的实验数据为人工构造的示例图与行动图（含多条路径、不同类型的红方行动图），未使用公开数据集。

**📈 对比分析**

与完全信息游戏、单侧不完全信息游戏以及基准策略（CI、1S、2S）对比，实验显示：在双侧不完全信息下，蓝方能显著降低期望成本，红方则因信息缺失导致收益下降；XDO算法在有限迭代内收敛到ϵ‑NE，且能在可管理的计算时间内完成。

**⚠️ 局限性**

限制包括：①实验规模受限于手工构造的小型图，难以验证在大规模实际网络上的性能；②默认策略的设计对结果影响较大，缺乏通用性；③算法对参数（ϵ、初始策略）的敏感性尚未系统评估。

---

## 164. Empirical Bayes Conformal Prediction for Vision and Language Models

**arXiv ID:** 2605.23189 | [PDF](https://arxiv.org/pdf/2605.23189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 165. Exploiting Longitudinal Context in Clinician-Verified Interactive Lesion Tracking

**arXiv ID:** 2605.23118 | [PDF](https://arxiv.org/pdf/2605.23118v1)

**作者:** Yannick Kirchhoff `[一作]` (German Cancer Research Center), Klaus Maier-Hein `[通讯]` (German Cancer Research Center)

**通讯引用:** 27910 | [OpenAlex ID](https://openalex.org/A5027292126)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了验证式跟踪（Verified Tracking）框架，将注册定位、专家确认和基于基线影像的时序分割整合到一个统一的工作流程中。

**💡 创新点**

创新点包括：① 通过验证式跟踪消除检索失败，提供安全的临床可控性；② 早期空间提示融合与潜在时序差分加权融合的双层融合策略，充分利用基线信息；③ 大规模合成纵向预训练激活网络对时间信息的利用。

**🔧 技术方法**

使用技术：共享权重U‑Net编码器、早期提示融合（点热图编码）、Latent Temporal Fusion（Difference Weighting Block）、基于uniGradICON的配准模型、合成纵向数据预训练。

**📊 数据集**

数据集：autoPET/CT IV（285例黑色素瘤患者的多时间点CT）和新发布的PanTrack（45例胰腺癌患者的161张CT，作为OOD基准）。

**📈 对比分析**

与自动跟踪、已验证跟踪等基线对比，自动模式下Dice 60.7%（autoPET）/58.2%（PanTrack），验证模式下73.7%/60.0%；显著优于前沿方法，尤其在非中心提示下保持高鲁棒性。

**⚠️ 局限性**

局限性：仍假设已知基线病灶，依赖配准质量；目前仅使用点提示，未覆盖更丰富的交互方式；未在真实临床读者工作流中进行验证。

---

## 166. Prompt Overflow: What the Guardrail Inspects Is Not What the Model Infers

**arXiv ID:** 2605.23196 | [PDF](https://arxiv.org/pdf/2605.23196v1)

**作者:** Yuanbo Zhou `[一作]` (Missouri University of Science and Technology), Junjie Xiong `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 48 | [OpenAlex ID](https://openalex.org/A5067082542)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究并提出了 Prompt Overflow Attack，揭示了 LLM 安全管道中 guardrail（安全检查器）与下游 LLM 之间的上下文窗口不匹配导致的安全漏洞；

**💡 创新点**

创新点在于：①通过将恶意指令碎片化并分散于超长提示，利用 guardrail 的窗口局限性实现规避；②提出风险感知碎片化与填充布局策略，验证对多种 guardrail 模型（Meta Llama Prompt Guard、IBM Granite Guardian、DeBERTa）均有效；③首次展示了填充语义对检测敏感的“语义伪装”现象；

**🔧 技术方法**

技术方法包括：黑盒窗口探测、风险感知 token 选择、分块覆盖与重叠窗口策略、滑动窗口聚合、以及基于窗口连续度的后置聚合防御；

**📊 数据集**

使用了 Prompt Injection Benchmark（5k恶意提示）与 RealToxicityPrompts（10k毒性提示）等公开数据集进行验证；

**📈 对比分析**

对比传统的最大池化聚合和新提出的连续窗口累计聚合，实验显示在 512‑token guardrail 下，Prompt Overflow Attack 在多种模型与布局上均可达到 99–100% 的规避率，而新聚合规则能恢复约 70–80% 的被攻击案例，保持安全性；

**⚠️ 局限性**

局限性包括：防御方案为初步尝试，未针对更强适应性攻击进行鲁棒性评估；模型对超长输入的整体阈值校准可能因域差异需要调整；此外，碎片化攻击可能对极短或结构化恶意内容的检测仍有限。

---

## 167. Occlusion-Aware Physics-Semantic Keyframe Selection for Robust Video Editing

**arXiv ID:** 2605.23192 | [PDF](https://arxiv.org/pdf/2605.23192v1)

**作者:** Lin Liu `[一作]` (Huawei), Qi Tian `[通讯]` (Huawei)

**通讯引用:** 43563 | [OpenAlex ID](https://openalex.org/A5100393506)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于物理-语义的关键帧选择框架，用于在遮挡、视角变化等复杂场景下实现稳定、精准的视频编辑。

**💡 创新点**

创新点在于：① 将遮挡问题转化为“可靠视觉锚点”选择；② 通过结构完整度、循环一致性跟踪稳定性和视觉语言模型评估的三维度综合评分自动挑选最佳关键帧；③ 用自动生成的时空掩码作为辅助监督，兼顾自动化与可交互性。

**🔧 技术方法**

使用技术包括：Diffusion‑based 3D DiT（ReCo）编辑骨干；GroundingDINO 目标检测；SiamRPN/KCF 循环一致性跟踪；Qwen2.5‑VL 语义可见性评估；mask‑guided diffusion 训练目标；LoRA 参数高效微调。

**📊 数据集**

数据集：ReCo 指令驱动视频编辑大语料；Open‑VE Bench（公开视频编辑基准）；Occlusion‑Bench（自建遮挡严重的测试集，来源于 MOSE 等）。

**📈 对比分析**

与多种文本驱动与辅助引导方法（VACE、Ditto、SAMA、Lucy‑Edit、InsV2V 等）进行对比。实验显示：在 Open‑VE Bench 上平均 VLM 分数从基线 2.42 提升至 3.16（+30.6%），在遮挡基准上在替换、移除、添加任务上均优于基线，并在关键帧选择与掩码监督的 ablation 研究中验证了其显著贡献。

**⚠️ 局限性**

局限性：① 依赖双向跟踪模型，极端运动模糊或长期遮挡仍可能导致掩码漂移；② 语义可见性评估需调用 VLM，增加前处理开销；③ 目前主要适用于单目标编辑，尚未充分支持多目标交互与 3D 视角一致性；④ 仍是两阶段流程，未实现端到端的关键帧选择与生成统一。

---

## 168. Robust LLM Watermarking with Minimal Semantic Distortion for IP Protection

**arXiv ID:** 2605.23175 | [PDF](https://arxiv.org/pdf/2605.23175v1)

**作者:** Kieu Dang `[一作]` (State University of New York at Albany), Ruoming Jin `[通讯]` (Kent State University)

**通讯引用:** 20355 | [OpenAlex ID](https://openalex.org/A5018875253)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SafeSeal，一种基于密钥的LLM水印框架，能够在不显著损害语义和事实一致性的前提下，对生成文本进行可检索的隐蔽标记；

**💡 创新点**

创新点包括：①利用实体识别排除实体并仅对语言学可替换词进行同义词替换，以保持语义完整；②采用基于密钥的锦标赛采样产生与密钥相关的可识别信号；③构建基于对比学习的密钥条件检测器，支持跨供应商和多用户的特定检测；④给出实用的效用-可检测性理论边界与实验验证；

**🔧 技术方法**

技术手段涵盖：实体识别（NER）、词性标注（POS）、轻量级语言模型生成同义词、上下文感知同义词库、密钥条件锦标赛采样、PRF和对比损失训练的双层MLP检测器；

**📊 数据集**

实验使用了多款7B LLM（LLaMA‑2、Mistral、DeepSeek、Qwen2.5、Gemma），评测数据集包括C4文本生成、MMLU问答、CNN/DailyMail摘要；

**📈 对比分析**

与KGW、EXP、SIR、SynthID、TW、DTM、LW等多种基线对比，SafeSeal在BERTScore、实体相似度、检测率和鲁棒性（抵御去水印和模型窃取攻击）上均优于或接近最优；在人类评测中获得最高质量分；延迟仅略高于最快基线；

**⚠️ 局限性**

局限性：在同义词多样性有限的低多样性领域效果可能下降；密钥管理与分发仍需完善；候选同义词生成依赖轻量级LM，仍存在效率瓶颈；理论分析基于距离近似，缺乏更严格的概率保证。

---

## 169. Autonomous Frontier-Based Exploration with VLM Guidance

**arXiv ID:** 2605.23165 | [PDF](https://arxiv.org/pdf/2605.23165v1)

**作者:** Aarush Aitha `[一作]` (University of California), Avideh Zakhor `[通讯]` (University of California)

**通讯引用:** 10527 | [OpenAlex ID](https://openalex.org/A5008034417)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将Vision‑Language Model（VLM）作为高层决策者的探索管线，让机器人在未知环境中通过VLM分析占用地图和前沿图像来选择最佳前沿，显著提升地图覆盖率。

**💡 创新点**

创新点在于利用无训练的VLM进行空间推理，替代传统几何启发式，显著提升探索效率与完整度。

**🔧 技术方法**

技术包括Google Gemini 2.5 Pro VLM、ROS‑OpenCV前沿检测、RTAB‑Map SLAM、TEB本地规划以及Habitat＋Matterport3D仿真桥接。

**📊 数据集**

使用Matterport3D数据集构建的多室室内环境进行仿真。

**📈 对比分析**

与Greedy Frontier、NBV、TARE、DSVP等四种基线对比，实验显示在六个环境中覆盖率提升至90%以上，平均比最优基线高出约20%–24%，且行驶距离更短。

**⚠️ 局限性**

局限包括对互联网连接和VLM服务的依赖、在真实环境中的适应性待进一步验证、以及前沿黑名单与决策点列表的手动设计。

---

## 170. SolarChain: Bridging Physical Law, Verifiable Trust, and Sustainable Markets for Urban Energy Resilience

**arXiv ID:** 2605.23162 | [PDF](https://arxiv.org/pdf/2605.23162v1)

**作者:** Shilin Ou `[一作]`, Ming-Chun Huang `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个去中心化的能源平台，通过将数字经济结算与热力学法则相连接，解决城市去碳化过程中面临的数据操控和经济激励问题。

**💡 创新点**

创新点在于通过物理法则验证数据的真实性，建立了一个基于热力学限制的客观验证边界，从而防止虚假数据注入和市场投机。

**🔧 技术方法**

使用了实时气象数据、地理空间坐标和基于第一性原理的太阳能产量计算技术，结合区块链和物联网技术。

**📊 数据集**

使用了来自五个城市（北京、上海、成都、深圳和杭州）的数据集，包括分布式光伏节点的静态注册信息和小时级的发电数据。

**📈 对比分析**

通过与无分割基线进行比较，展示了在市场深度和交易滑点方面的性能提升，平均流动性提高了60.8%，交易滑点降低了66.1%。

**⚠️ 局限性**

限制在于系统的实施依赖于高分辨率的地理空间数据，且在不同城市的适用性可能受到当地气候和基础设施的影响。

---

## 171. Any-Dimensional Invariant Universality

**arXiv ID:** 2605.23156 | [PDF](https://arxiv.org/pdf/2605.23156v1)

**作者:** Shengtai Yao `[一作]` (Johns Hopkins University), Mateo Díaz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种通用框架，用以定义和证明任意维度不变（invariant）深度学习模型的通用逼近性（universality），并通过该框架在集合、图和点云三个典型域上构造并证明了新的通用模型；

**💡 创新点**

创新点在于：①将不同维度输入序列统一映射到一个无限维极限空间，并通过群作用的轨道空间得到自然紧集；②提出三步通用构造法（确定紧集、构造连续可传递模型、利用 Stone–Weierstrass 定理证明子代数稠密性）；③发现并修正现有架构的连续性缺陷，提出可连续的 DeepSets 变体、归一化 DeepSets、基于同构密度的图模型以及基于Gram-图神经网络的点云模型；

**🔧 技术方法**

使用的主要技术包括：一致序列（consistent sequence）与极限空间构造、轨道空间与对称化度量、正则化与可传递性（continuously transferable）的定义、Stone–Weierstrass 逼近定理、同构密度函数与图论度量（cut metric）、高阶张量隐藏层、以及梯度可优化的网络实现；

**📊 数据集**

本工作为理论研究，未使用公开数据集，所有示例均基于数学构造与假设；

**📈 对比分析**

由于缺乏实验评估，本文仅通过理论证明展示模型在各紧集上的稠密性和连续性，未给出数值性能对比；

**⚠️ 局限性**

局限性包括：仅处理标量输出的非齐变模型；对稀疏图、非标量输出或更一般的可变尺寸等情况缺乏理论支持；

---

## 172. As X, Do Y: How Persona and Task Combine in Instruction-Tuned LLMs

**arXiv ID:** 2605.23147 | [PDF](https://arxiv.org/pdf/2605.23147v1)

**作者:** Eric Xu `[一作]` `[通讯]`, Eric Xu

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在指令调优语言模型中，研究了角色提示（persona + task）在残差流中的作用，并发现其在答案生成的早中层、提示末端及前两词位置呈现可加性组合。

**💡 创新点**

创新点在于：① 用残差层级的因果干预方式刻画 persona‑task 的可加组合；② 在多模型（Gemma‑2‑2B‑IT、Qwen‑2.5‑1.5B/3B）和两种网格（12‑cell 短网格、48‑cell 长网格）上验证该规律；③ 通过行为指标（persona 关键词出现率）证明可加替代不破坏 persona 语义。

**🔧 技术方法**

技术手段包括：残差分解（ΔX、ΔY、ΔXY、Interaction）、KL 余弦相似度、交叉层级的因果替代、教师强制生成、行为标记评估。

**📊 数据集**

数据集主要为自定义提示网格：短网格（4 人格 × 3 任务，共 12 组）和长网格（8 人格 × 6 任务，共 48 组），全部使用英文、单轮、合作式指令；模型来自 Gemma‑2‑2B‑IT 与 Qwen‑2.5‑1.5B/3B 指令版。

**📈 对比分析**

比较方法：在 p_last、g_1、g_2 三个位置以及多层设置下，用 KL 散度评估可加替代对后续 10‑token 生成的影响；用余弦相似度衡量 ΔXY 与 ΔX+ΔY 的对齐；行为标记统计衡量 persona 内容的保持。结果显示：在早中层（如 L6–L14）可加替代 KL 接近 0，cosine 对齐 ≥0.9；但在更深层或更宽窗口下效果衰退，且单点替代无法完全代替 persona 文本。

**⚠️ 局限性**

局限性：① 仅关注答案生成前后短区间，未覆盖完整生成过程；② 未识别最小头/神经元/权重集合；③ 仅处理英文、单轮、合作式提示；④ 对极端 X‑Y 组合的交互项尚未阐明；⑤ 只验证两种模型族，跨模型推广性待进一步确认。

---

## 173. From Preventive to Reactive: How AI Coding Assistants Transform Developers' Security Awareness

**arXiv ID:** 2605.23130 | [PDF](https://arxiv.org/pdf/2605.23130v1)

**作者:** Faisal Haque Bappy `[一作]` (University of Maryland Baltimore County), Tariqul Islam `[通讯]` (University of Maryland Baltimore County)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对AI编码助手在专业软件开发中的使用场景进行实证研究，探讨其如何改变开发者的安全意识与实践。

**💡 创新点**

首次系统揭示AI助手从预防性安全思考转为事后复核的结构性转变；发现“提示缺口”和非正式的安全应对策略；证明经验层级并不能预测AI辅助下的安全表现。

**🔧 技术方法**

采用半结构化访谈、思维大声编码任务（使用Gemini CLI），以及主题分析方法。

**📊 数据集**

收集15名行业专业软件工程师（3个经验层级）进行访谈与编码任务，使用自定义的三种安全相关编程任务场景；未使用公开漏洞或代码库数据集。

**📈 对比分析**

通过对比不同经验层级与非经验层级在安全检测上的表现，评估安全发现率与提示行为；结果显示无显著差异，强调工具交互模式对安全的结构性影响。

**⚠️ 局限性**

样本规模小且性别偏向男性；访谈-任务顺序可能引入预置效应；仅使用Gemini CLI，缺乏跨工具验证；研究聚焦于安全意识而非量化的漏洞率，缺乏客观性能指标。

---

## 174. The Closure of LCD-to-GI Reductions via Generalized Inner Products

**arXiv ID:** 2605.23120 | [PDF](https://arxiv.org/pdf/2605.23120v1)

**作者:** Keita Ishizuka `[一作]` `[通讯]` (Mitsubishi Electric Corporation), Keita Ishizuka (Mitsubishi Electric Corporation)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文研究了线性码的排列等价问题（PEP），并证明了通过正交投影器的方法可以将线性互补对偶（LCD）码的PEP问题归约为图同构（GI）问题，扩展了之前的结果。

**💡 创新点**

创新点在于证明了正交投影器方法可以扩展到双线性形式M = aI + bJ，并且没有其他非退化的对称形式可以有效归约PEP。还推导出了确切的枚举公式和多项式时间的归约算法。

**🔧 技术方法**

使用了正交投影器和双线性形式的技术，特别是通过参数化的对称结构矩阵M来实现归约。

**📊 数据集**

使用了线性码的生成矩阵和相关的正交投影器，具体数据集未明确提及，但涉及到的码是线性码，特别是LCD码。

**📈 对比分析**

通过与现有的支持分裂算法和图同构算法进行比较，证明了在特定条件下，使用正交投影器的方法可以有效地解决PEP问题，性能上优于传统方法。

**⚠️ 局限性**

限制在于该方法仅适用于具有壳维度最多为1的码，且在特征为2的情况下，仅LCD码是可归约的。壳维度大于等于2的码无法通过此方法归约。

---

## 175. Label-Efficient Dataset Pruning via Semi-Supervised Pseudo-Labeling

**arXiv ID:** 2605.23198 | [PDF](https://arxiv.org/pdf/2605.23198v1)

**作者:** Yeseul Cho `[一作]` (KAIST), Chulhee Yun `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于半监督学习的标签高效数据集修剪框架，利用少量随机标记的子集生成伪标签，从而使现有的监督修剪方法能够应用于伪标记的训练池。

**💡 创新点**

创新点在于通过半监督学习生成伪标签，使得修剪方法在仅有少量标记数据的情况下仍能有效捕捉目标数据分布，从而提高了难度估计的可靠性和数据集修剪的效果。

**🔧 技术方法**

使用了半监督学习（SSL）技术，特别是FixMatch算法来生成伪标签。

**📊 数据集**

在多个数据集上进行了验证，包括Food-101、SUN397、CIFAR-100的腐蚀变体、Caltech-101和长尾变体的CIFAR-100。

**📈 对比分析**

与现有的标签自由和标签高效基线方法相比，该方法在多个挑战性设置中实现了最先进的性能，并在标准基准上也表现出竞争力。

**⚠️ 局限性**

限制在于该方法依赖于初始随机选择的少量标记样本，未来可以通过主动学习选择更具信息量的样本进行改进。

---

## 176. Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation

**arXiv ID:** 2605.23191 | [PDF](https://arxiv.org/pdf/2605.23191v1)

**作者:** Guoming Li `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Menglin Yang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种新的深度推荐架构 RankElastor，针对传统基于 token‑mixing 的推荐模型中出现的嵌入崩塌问题进行改进。

**💡 创新点**

创新点在于引入可学习的全混合（Parameterized Full Mixing）提升 token 级别的表达能力，并采用 GLU 结构的 P‑FFN 稳定谱分布，从而显著提升有效秩并抑制嵌入崩塌。

**🔧 技术方法**

主要技术包括全混合矩阵变换、GLU‑改进的 per‑token FFN、谱分析（effective rank）与理论证明，以及基于 FuxiCTR 的大规模实验框架。

**📊 数据集**

使用了工业级 CTR 数据集 Criteo、Avazu 以及行为序列预测数据集 KuaiVideo 和 TaobaoAd 进行评估。

**📈 对比分析**

通过与 DCNv2、xDeepFM、AutoInt、MLP 等强基线进行对比，RankElastor 在 AUC 上提升约 0.001、LogLoss 下降，且在有效秩和参数扩展性上优于基线。

**⚠️ 局限性**

局限性主要是对 token‑mixing 与 P‑FFN 设计的依赖，理论分析仅适用于此类架构，未来仍需探索更通用的谱保持机制。

---

## 177. Pure Exploration for a Good Policy in Reinforcement Learning with Bandit Feedback

**arXiv ID:** 2605.23182 | [PDF](https://arxiv.org/pdf/2605.23182v1)

**作者:** Zitian Li `[一作]` (National University of Singapore), Wang Chi Cheung `[通讯]` (National University of Singapore)

**通讯引用:** 1276 | [OpenAlex ID](https://openalex.org/A5010450673)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于阈值的良好策略识别（GPI）问题，并给出了在固定置信度下的最优样本复杂度算法 BEE‑GPI。

**💡 创新点**

创新点在于：①将目标从寻找最优策略转向寻找满足阈值的任一策略；②通过早停探索与后验验证分离，消除了对状态动作空间大小的依赖；③给出了与阈值差距相关的上界与下界，证明了算法的近似最优性。

**🔧 技术方法**

技术包括：基于 UCRL 的乐观探索、早停 BPI-UCRL（ES‑BPI‑UCRL）、自适应探索–利用阶段、置信区间验证、信息理论下界分析。

**📊 数据集**

实验使用经典的 Single Chain 与 Double Chain MDP（4 状态、2 动作），阈值取值为 {1, 1.5, 2.0, 2.5, 3.0}，所有实例均为正实例。

**📈 对比分析**

与传统的 ϵ‑最优 BPI（BPI‑UCRL）进行对比，BEE‑GPI 在所有阈值下都显著减少了停机所需的 episode 数量，验证了理论中 log(1/δ) 与 S、A 无关的样本复杂度优势。

**⚠️ 局限性**

局限性包括：①对大规模状态动作空间的扩展尚未证明；②需先知晓阈值 μ₀；③在负实例中样本复杂度仍随 S、A 成多项式增长，尚有改进空间。

---

## 178. When Symptoms Are Not Enough: Evidence-Weighting Patterns in Large Language Model Psychiatric Screening

**arXiv ID:** 2605.23148 | [PDF](https://arxiv.org/pdf/2605.23148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 179. Redrawing the AI Map: A Theory of Accountability Boundaries in Agentic Ecosystems

**arXiv ID:** 2605.23179 | [PDF](https://arxiv.org/pdf/2605.23179v1)

**作者:** Muhammad Zia Hydari `[一作]` (University of Pittsburgh), Farooq Muzaffar `[通讯]` (Ensi.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出“可问责资产”与“规则债务”概念，构建基于agentic AI生态的责任边界策略理论

**💡 创新点**

创新点在于将责任边界与技术接口区分、引入可问责资产的协同专化、定义规则债务作为治理成本，并解释agentic orchestrator如何影响企业边界配置

**🔧 技术方法**

主要采用理论构建与概念分析方法，结合数字创新、交易成本、互补资产、平台治理和IS控制等多学科理论框架

**📊 数据集**

无实验数据集，理论性研究基于文献综述与案例示例（文档提取、法律检索、审计、临床决策、采购等）

**📈 对比分析**

对比方法为结构化理论示例和对已有理论（如Coase、Teece、平台治理）的预测修正，性能以理论一致性与解释力评估

**⚠️ 局限性**

局限在于缺乏经验检验与量化指标，未给出具体设计原则或算法实现，需要后续实证研究与案例验证

---

## 180. PoisonForge: Task-Level Targeted Poisoning Benchmark for Instruction-Tuned LLMs

**arXiv ID:** 2605.23168 | [PDF](https://arxiv.org/pdf/2605.23168v1)

**作者:** Luze Sun `[一作]` (Northeastern University), Alina Oprea `[通讯]` (Northeastern University)

**通讯引用:** 5950 | [OpenAlex ID](https://openalex.org/A5035574749)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了PoisonForge，一个用于评估大型语言模型（LLMs）在任务级中毒攻击下的脆弱性的基准。研究了如何通过插入少量精心设计的指令-响应对来进行任务级中毒，从而使模型在特定任务中嵌入攻击者指定的实体。

**💡 创新点**

创新点在于系统地参数化了中毒攻击的规格，涵盖了偏见类型、中毒模式、出现次数和目标输出长度四个维度，并在12个开放权重模型上进行了评估，揭示了中毒设计选择对攻击成功的主要影响。

**🔧 技术方法**

使用了迭代生成器和评分器管道来构建中毒数据，并通过对12个模型进行指令调优来评估攻击成功率（ASR）、溢出率（SOR）和良性效用。

**📊 数据集**

使用了来自Super-NaturalInstructions的数据集，构建了包含1000个良性示例和10个中毒示例的训练集，评估了不同任务的中毒效果。

**📈 对比分析**

与现有方法相比，11个模型在最佳配置下的攻击成功率超过70%，而非目标任务的意外泄漏保持在0.5%以下，模型在标准基准测试中的表现良好，表明中毒攻击在隐蔽性和有效性方面具有显著威胁。

**⚠️ 局限性**

限制在于该基准主要集中于开放式英语文本生成，尚未测试其在结构化任务（如多项选择）或多语言环境中的适用性。此外，实体检测依赖于正则表达式匹配，可能低估真实的攻击成功率。

---

## 181. Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving

**arXiv ID:** 2605.23163 | [PDF](https://arxiv.org/pdf/2605.23163v1)

**作者:** Kewei Zhang `[一作]` (Peking University), Enze Xie `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种块扩散的视觉-语言-行动模型（VLA），用于端到端的自动驾驶，旨在提高规划准确性和推理效率。

**💡 创新点**

通过将确定性模式令牌视为固定支架，结合语义部分对齐的扩散块和在训练中优先考虑安全关键令牌，达到了最先进的轨迹准确性，同时实现了比全序列扩散基线高6倍的吞吐量。

**🔧 技术方法**

采用块扩散（Block Diffusion）技术，结合了自我推测解码（Scaffold Speculative Decoding）和结构感知的训练方法。

**📊 数据集**

使用了nuScenes和Waymo Open Dataset End-to-End (WOD-E2E)两个数据集进行评估，前者包含1000个城市驾驶场景，后者包含4021个长尾驾驶片段。

**📈 对比分析**

与自回归（AR）基线和扩散基线dVLM-AD相比，模型在WOD-E2E测试集上达到了最低的ADE@3s和ADE@5s，同时在吞吐量上是dVLM-AD的4到6倍，显示出在准确性和效率之间的良好平衡。

**⚠️ 局限性**

模型的局限性在于，尽管在准确性和效率上表现优异，但仍需在不同场景下进行更广泛的验证，以确保其在各种复杂驾驶环境中的可靠性。

---

## 182. VisAnalog: A Diagnostic Suite for Visual Concept Transfer on Natural Images

**arXiv ID:** 2605.23141 | [PDF](https://arxiv.org/pdf/2605.23141v1)

**作者:** Zhaonan Li `[一作]` (Arizona State University), Ben Zhou `[通讯]` (Arizona State University)

**通讯引用:** 4860 | [OpenAlex ID](https://openalex.org/A5067460538)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的视觉类比任务，旨在测试模型在图像变换下是否能够保持和操作概念级属性，并将其转移到新场景中。

**💡 创新点**

创新点在于引入了一个控制基准，专注于自然图像中的视觉概念转移，特别是在多步变换下的表现。

**🔧 技术方法**

使用了程序条件评估技术来分离关系推断失败和变换应用失败，增强了基准的可解释性。

**📊 数据集**

使用了来自SA-1B的数据集，生成了617个经过人工验证的问题，涵盖了一到四步的变换。

**📈 对比分析**

与强大的专有和开源视觉语言模型（VLM）进行比较，发现模型在多步组合下的表现显著低于人类，且随着变换深度的增加，准确率急剧下降。

**⚠️ 局限性**

限制在于当前模型在推断视觉关系和应用已知变换方面存在显著的错误，尤其是在多步变换的情况下，表现不佳。

---

## 183. Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For Worst-Case Robustness

**arXiv ID:** 2605.23146 | [PDF](https://arxiv.org/pdf/2605.23146v1)

**作者:** Manish Aryal `[一作]` (Purdue University), Paul Yushin Rapoport `[通讯]` (University Of Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

实现了一个基于Infra-Bayesian（IB）框架的有限结果无状态强化学习代理，实现了对不确定性的Knightian（非可量化）处理，并在Bernoulli bandit与Newcomb问题上进行测试

**💡 创新点**

首次将IB理论具体实现为RL代理，提供了一个可计算的实现细节，包括a‑measure、infradistribution、IB更新规则等；展示了在Knightian不确定性与策略依赖环境下的稳健性

**🔧 技术方法**

采用a‑measure与infradistribution来表示不确定性，使用IB条件化规则进行更新，利用极值点（vertex）压缩实现；在实验中采用离散bandit与Newcomb的模拟环境

**📊 数据集**

实验使用自定义的Bernoulli bandit（两臂、概率区间约束）与Newcomb的概率模型（预测器准确度α从0.5到1）

**📈 对比分析**

通过对比经典Bayesian RL（基于点先验或均匀先验）与IB RL，在Knightian bandit下，IB RL在最坏环境下实现更低的累积后悔；在Newcomb问题中，IB代理始终选择最优策略，并在预测器准确度超过0.55时一盒子策略，性能与理论最优一致

**⚠️ 局限性**

实现仅限于有限结果、非负a‑measure和小型假设空间；无法直接扩展到连续状态、巨大假设类或函数逼近；在多步决策与大规模问题上仍需改进

---

## 184. Adaptive Mass-Segmented KV Compression for Long-Context Reasoning

**arXiv ID:** 2605.23200 | [PDF](https://arxiv.org/pdf/2605.23200v1)

**作者:** Junzhe Yang `[一作]` (Shanghai Jiao Tong University), Xiaoyu Shen `[通讯]` (Eastern Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Adaptive Mass‑Segmented (AMS) KV 缓存压缩框架，改用区域分配优先而非单词级 Top‑k，以保持长链推理的连续性。

**💡 创新点**

创新点在于：1) 将注意力使用量转换为区域质量质量（mass）并自适应划分段；2) 在段层面分配固定预算并为每段提供最小保留保证；3) 通过 EMA 归一化实现历史平滑，提升稳定性；4) 作为通用 plug‑in 可与多种评分器无缝结合。

**🔧 技术方法**

使用注意力权重计算质量质量、前缀和划分、按质量分配预算、局部 Top‑k 选择、EMA 平滑、gather‑compact KV 以及与 vLLM 页式 KV 兼容的实现。

**📊 数据集**

在数学推理（Math500、AIME24/25、GSM8K）、代码补全、开放域问答、LongBench 以及稀疏检索等多任务上评测。

**📈 对比分析**

与多种基线（StreamingLLM、TOVA、AdaKV‑ExpE2、PyramidKV、ChunkKV‑Expected、TriAttention、KeyDiff、R‑KV、RPC 等）对比；AMS 在极低 KV 预算下显著提升准确率（如 Math500 T_keep=256 上 +48.6% 对比 29.2%），在大模型和跨 backbone 下保持优势，并在系统层面实现与 vLLM 的兼容，内存占用与速度保持不变或略降。

**⚠️ 局限性**

局限：1) 对极端大预算的收益有限；2) 仍需手工设定分段和 EMA 超参数；3) 在非长序列任务中的优势不如数学推理显著；4) 依赖注意力信号，若注意力失真可能导致分段失效。

---

## 185. IntentionNav: A Benchmark for Intent-Driven Object Navigation from Implicit Human Instruction

**arXiv ID:** 2605.23187 | [PDF](https://arxiv.org/pdf/2605.23187v1)

**作者:** Lin Qian `[一作]` (University of Manchester), Hujun Yin `[通讯]` (University of Manchester)

**通讯引用:** 4651 | [OpenAlex ID](https://openalex.org/A5055149475)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了“意图驱动对象导航”基准，评估机器人在仅收到隐式人类需求时如何推断目标并完成主动搜索

**💡 创新点**

构建了包含500个目标、4种语义提示（事件脚本、内在状态、物理状态、可操作性）及4种语言风格的诊断式基准，能够区分意图理解、导航可达性与终端定位的瓶颈

**🔧 技术方法**

利用VLM（GPT‑5.4、Qwen3.6‑Plus、Gemini‑3.1‑Flash‑Lite）与固定的参考主动导航框架（RGB‑D感知、开放词汇检测、地图记忆、终止判定）进行零样本推理

**📊 数据集**

使用从VLNVerse模拟环境（176个室内场景、64个目标类别）自动生成的17,624条意图候选，最终筛选为500条具备唯一目标、无词汇泄露、语义可辨的训练集（共2000条指令）

**📈 对比分析**

通过对比三种VLM在同一500意图集上的IM、OSR、SR、GSR等指标，发现IM≈0.48、OSR≈0.69但SR仅0.25、GSR仅0.055；在明确给定目标类别时，SR提升至0.45但仍存在定位与视觉确认瓶颈，说明意图推理与终端精确定位是主要难点

**⚠️ 局限性**

受限于单语（英语）静态场景、单一目标、30步预算、无对话澄清和仅仿真验证；多语言、多目标、交互式和真实家庭环境仍需进一步研究

---

## 186. Composing People Together: Iterative Pose-Image Generation for Multi-Person Interaction Scenes

**arXiv ID:** 2605.23178 | [PDF](https://arxiv.org/pdf/2605.23178v1)

**作者:** Wenxuan Peng `[一作]` (Cornell University), Hadar Averbuch-Elor `[通讯]` (Cornell University)

**通讯引用:** 1092 | [OpenAlex ID](https://openalex.org/A5043669878)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种面向多人物交互场景的文本到图像生成方法，使用双重姿态-图像表示和迭代逐人生成，实现了对细粒度交互语义的精准对齐。

**💡 创新点**

创新点包括：① 将姿态作为结构先验与图像生成共享的扩散变换器内部学习；② 通过修改3D旋转位置编码（RoPE）在不同模态间实现人级跨模态对齐；③ 采用可分解姿态的迭代生成框架，将多人物生成拆分为一系列单人生成子任务，显著提升语义一致性与多样性。

**🔧 技术方法**

技术实现基于 FLUX 预训练的 Diffusion Transformer，添加了并行姿态分支、LoRA 参数微调、双模态投影层；使用 3D RoPE 绑定文本、姿态与图像 token；通过多阶段训练和迭代生成策略，完成多人物交互场景的构建。

**📊 数据集**

训练数据来自 Who's Waldo 视觉语言数据集（约30k 人际交互图像）和其自动生成的结构化交互描述；评测使用新构建的 DrawWaldoWorlds 基准（分层交互描述）以及现有 MultiHuman-Testbench，提供多级提示与评测。

**📈 对比分析**

与 FLUX、SDXL、SD3.5-Large、CreatiLayout、RealCompo 等基线相比，在 DrawWaldoWorlds 上的 VQA Accuracy 与 VQA Sim 均领先，特别是在复杂交互层级（Tier C）仍保持高对齐；多样性指标（DINO、LPIPS、GRADE）亦优于大多数基线，显示出更丰富的场景生成。

**⚠️ 局限性**

局限性包括：① 仍依赖姿态检测与 bounding‑box 先验，对姿态估计错误或重叠可能导致生成失真；② 迭代逐人生成增加了推理时间；③ 目前仅针对静态图像，尚未处理时序交互与动态场景。

---

## 187. Positional Failures in Long-Context LLMs: A Blind Spot in Reasoning Benchmarks

**arXiv ID:** 2605.23170 | [PDF](https://arxiv.org/pdf/2605.23170v1)

**作者:** Chuyifei Zhang `[一作]` (Beijing Jiaotong University), Jitao Sang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 2210 | [OpenAlex ID](https://openalex.org/A5023834030)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的评估框架，称为上下文旋转评估（CRE），用于系统地控制和评估长上下文推理任务中的任务位置、填充内容和上下文长度。

**💡 创新点**

创新点在于首次将任务位置、填充内容和上下文长度三者结合进行系统评估，填补了当前推理基准设计中的结构性评估空白。

**🔧 技术方法**

使用了上下文旋转评估（CRE）框架，结合了多种长上下文大语言模型（LLMs）的评估。

**📊 数据集**

使用了GSM8K（数学文字问题）和ARC-Challenge（科学多项选择）作为数据集，评估了九种长上下文LLMs。

**📈 对比分析**

通过与传统的长上下文基准进行比较，发现当目标任务从上下文的末尾移动到中间时，模型的性能会显著下降，尤其是在上下文长度增加时，某些模型的准确率下降幅度可达94个百分点。

**⚠️ 局限性**

限制在于数据收集范围、实验设置、解释注意事项和外部有效性等方面，包括样本量小、模型间的比较受限于提供者的异质性等问题。

---

## 188. SLIP-RS: Structured-Attribute Language-Image Pre-Training for Remote Sensing Object Detection

**arXiv ID:** 2605.23144 | [PDF](https://arxiv.org/pdf/2605.23144v1)

**作者:** Chenxu Wang `[一作]` (Nankai University), Qibin Hou `[通讯]` (Nankai University)

**通讯引用:** 18263 | [OpenAlex ID](https://openalex.org/A5040392623)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于结构化属性的遥感图像语言预训练框架 SLIP‑RS，利用属性解耦实现细粒度目标检测。

**💡 创新点**

创新点在于：①将对象语义拆解为有限且物理可解释的属性集合，突破单一标签学习的瓶颈；②设计 Structured‑Attribute Contrastive Learning（SACL）和 Conformal Attribute Reliability Engine（CARE），通过属性组合对比和置信度校准高质量伪标签；③构建 RS‑Attribute‑15M 超一千万实例属性数据集。

**🔧 技术方法**

核心技术包括：属性字典构建与随机组合增强、基于属性替换的硬负样本挖掘、对齐全局与局部视觉特征的统一对比损失、利用 conformal prediction 的阈值校准以及基于 ConvNeXT‑T/L 的检测器架构。

**📊 数据集**

使用了多源遥感检测数据集（DOTA‑v2.0、DIOR‑H、XView、DOTAv‑2.0 OBB）以及自构造的 RS‑Attribute‑15M（覆盖飞机、船舶、车辆等三大类别）。

**📈 对比分析**

与多种基线（DINO、RTMDet、OpenRSD、ViTP 等）比较，SLIP‑RS 在零样本迁移、细粒度属性识别、跨域检测以及复杂下游任务中均实现显著提升，例如在 DOTA‑v2.0 HBB 上从 39.21% 提升至 45.10%/47.14%，在属性检验上达到 72.04%–83.10% 的 mAP。

**⚠️ 局限性**

局限性主要体现在：①属性定义仍需人工构建，覆盖范围受限；②对极端稀有属性或组合的泛化能力尚未充分验证；③在大规模真实场景中，对标签噪声与计算成本的进一步优化仍有空间。

---

## 189. Defining AI Fatigue in Academic Contexts: Dimensions, Indicators, and a Stage-Based Model Using Grounded Theory

**arXiv ID:** 2605.23123 | [PDF](https://arxiv.org/pdf/2605.23123v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 190. Combined Radar and Magnetometer Sensor Network with LoRa-Mediated Awareness for Wildlife-Vehicle Collision Prevention: A Monte Carlo Analysis

**arXiv ID:** 2605.23117 | [PDF](https://arxiv.org/pdf/2605.23117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 191. CALAD: Channel-Aware contrastive Learning for multivariate time series Anomaly Detection

**arXiv ID:** 2605.23139 | [PDF](https://arxiv.org/pdf/2605.23139v1)

**作者:** Jaehyeop Hong `[一作]` (Inha University), Youngbum Hur `[通讯]` (Inha University)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5029147038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为CALAD的多变量时间序列异常检测框架，该框架通过通道感知对比学习来提高异常检测的准确性。

**💡 创新点**

创新点在于引入通道相关性估计，指导对比样本的构建，从而使学习过程更能反映异常语义，而非仅仅是通用相似性。

**🔧 技术方法**

使用了基于变压器的自编码器和对比学习技术，结合了重建头以保持正常结构。

**📊 数据集**

在多个真实世界的数据集上进行了实验，包括MSL、SMAP、SMD和SWaT等。

**📈 对比分析**

与多种基线方法进行比较，CALAD在大多数数据集上表现优异，尤其是在分布转移场景下，F1分数平均提高了11%。

**⚠️ 局限性**

限制在于该方法依赖于重建误差来估计通道相关性，可能在某些情况下无法准确识别所有异常相关通道。

---

## 192. Archimedean Copula Inference via Taylor-Mode AD

**arXiv ID:** 2605.23134 | [PDF](https://arxiv.org/pdf/2605.23134v1)

**作者:** Cambridge Yang `[一作]` (Independent), Dongdong Li `[通讯]` (Harvard Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为acopula的JAX原生框架，用于处理任意变量的右删失、任意嵌套树和精确参数梯度的嵌套阿基米德copula的似然性和参数梯度。

**💡 创新点**

创新点在于acopula能够处理高维数据（d>30），并且支持任意的阿基米德生成器，提供了一个无家族限制的梯度基础似然推断算法。

**🔧 技术方法**

使用了JAX框架，结合了泰勒模式自动微分和多项式幂运算来计算嵌套copula的似然性和梯度。

**📊 数据集**

使用了MIMIC-IV ICU入院数据集（d=53），S&P 500每日收益数据集（d=98），以及糖尿病视网膜病变研究中的多个家族数据集。

**📈 对比分析**

与现有工具（如R的包）进行比较，acopula在d=35时的速度提升约为650倍，并且在高维情况下表现出更好的性能，能够处理更复杂的嵌套结构和删失模式。

**⚠️ 局限性**

限制在于当前框架不考虑区间删失数据，且在处理高维数据时，编译成本和内存使用仍然是一个挑战。

---

## 193. When Determinants Are Not Enough: Private Rare Switching

**arXiv ID:** 2605.23131 | [PDF](https://arxiv.org/pdf/2605.23131v1)

**作者:** Xingyu Zhou `[一作]` (Wayne State University), Xingyu Zhou `[通讯]` (Wayne State University)

**通讯引用:** 981 | [OpenAlex ID](https://openalex.org/A5082563145)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种针对私有线性多臂赌博机与强化学习的稀疏切换策略更新规则，解决了在加高斯噪声后协方差矩阵单调性失效导致的传统更新规则失效问题。

**💡 创新点**

创新点在于将传统基于行列式增量的切换规则改为基于广义瑞利商的判据，从而在存在隐私噪声的环境下仍能保证对“最差方向”的控制，保持对数级更新次数与与非私有情形相同的调度效果。

**🔧 技术方法**

技术上主要运用线性模型、隐私噪声分析、广义瑞利商、椭圆势能引理等理论工具，构造新的更新判据并给出严格的理论证明。

**📊 数据集**

该工作为理论性研究，无实验数据集，主要在数学证明与理论分析上进行验证。

**📈 对比分析**

与传统的基于行列式的稀疏切换规则相比，新的规则同样实现了O(log T)的更新次数，并且在理论上保持了与非私有情形相同的置信宽度与调度保证；性能提升主要体现在恢复了对极端方向的控制。

**⚠️ 局限性**

局限性包括：更新判据需要计算矩阵的最大特征值，计算成本高于传统的秩一次更新；仅在高斯噪声和满足特定范数约束的私有噪声下证明了有效性，缺乏针对其他噪声模型的分析与实验验证。

---

## 194. CoReVAD: A Contextual Reasoning Framework for Training-Free Video Anomaly Detection

**arXiv ID:** 2605.23116 | [PDF](https://arxiv.org/pdf/2605.23116v1)

**作者:** Hyeongmuk Lim `[一作]` (Inha University), Youngbum Hur `[通讯]` (Inha University)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5029147038)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

CoReVAD 提出一种训练无关的视频异常检测框架，利用单一冻结的视觉语言模型同时生成异常得分和可解释的时序描述。

**💡 创新点**

创新点在于不依赖额外训练或外部 LLM，提出局部响应清洗 (LRC) 与视觉‑文本相似性软最大化、Gaussian 平滑和位置加权等全局时序细化策略，显著提升检测精度与可解释性。

**🔧 技术方法**

主要技术包括预训练的 InternVL2‑8B 视觉语言模型、CLIP 视觉/文本编码器、局部响应清洗 (LRC)、视觉‑文本上下文细化、Gaussian 平滑与位置加权等。

**📊 数据集**

实验数据集为 UCF‑Crime 和 XD‑Violence 两大无剪辑视频异常检测基准集。

**📈 对比分析**

与无监督、一类分类、弱监督、精细调优和其他训练无关可解释 VAD 方法对比，CoReVAD 在 UCF‑Crime 上获得 82.51% AUC（最高训练无关方法），在 XD‑Violence 上 AP 91.44%、AUC 91.44%，分别优于 LAVAD、MCANet 和 VERA。

**⚠️ 局限性**

局限性主要在于受预训练 VLM 的质量与提示策略限制，若 VLM 性能或 prompt 设计不佳，异常检测与解释的准确性可能下降；在跨域或远程上下文场景中仍可能出现误检。

---

## 195. Diffusion Domain Expansion: Learning to Coordinate Pre-trained Diffusion Models

**arXiv ID:** 2605.23275 | [PDF](https://arxiv.org/pdf/2605.23275v1)

**作者:** Egor Lifar `[一作]` (MIT), Tommi Jaakkola `[通讯]` (MIT)

**通讯引用:** 44967 | [OpenAlex ID](https://openalex.org/A5048915657)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出Diffusion Domain Expansion（DDE）方法，通过训练一个小型ViT协调器对预训练扩散模型的去噪输出进行协同，能够生成更大尺寸的对象并支持多条件生成，且可在未见过的更大域上实现泛化。

**💡 创新点**

创新点在于引入可训练的协调网络，使预训练扩散模型在不需改动核心网络的情况下扩展生成域，并利用ViT学习跨局部补丁的长程依赖，显著提升多条件与大尺寸生成能力。

**🔧 技术方法**

使用的技术包括扩散模型、ViT（配合Rotary位置编码）、多重补丁协同与平均（MultiDiffusion风格）、以及在不同域上进行的微调与评估。

**📊 数据集**

实验使用的主要数据集为音乐领域的Slakh2100、图像领域的CLEVR数据集（坐标条件生成）以及自建的卫星图像与地图对齐数据集。

**📈 对比分析**

与Concat、MultiDiffusion、RNN等基线对比，DDE在音乐FAD上从4.623/4.732降低到2.112/2.142，在CLEVR条件生成准确率上从最高33.6%提升至44.5%，在卫星图像FID上从37.8/35.0降低至31.8/27.4，表现更优。

**⚠️ 局限性**

主要限制是需要额外的数据和训练时间来训练协调器，且在极端规模或新领域下可能需要进一步适配。

---

## 196. RelPrism: A Multi-Faceted Pre-training Framework with Self-Generated Tasks for Relational Databases

**arXiv ID:** 2605.23241 | [PDF](https://arxiv.org/pdf/2605.23241v1)

**作者:** Jinyu Yang `[一作]` (Beijing University Of Posts and Telecommunications), Chuan Shi `[通讯]` (Beijing University Of Posts and Telecommunications)

**通讯引用:** 16174 | [OpenAlex ID](https://openalex.org/A5100705849)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了RelPrism，一种针对关系型数据库的多面向自监督预训练框架；

**💡 创新点**

通过多视角（内在属性、关系属性、混合属性）与多粒度聚类生成伪任务，覆盖不同任务对信息的需求；

**🔧 技术方法**

使用图神经网络（Graph Transformer）作为编码器，结合原型网络进行元学习式预训练，且构建了多层级的图和属性；

**📊 数据集**

在5个真实商业/社会/医疗/体育等领域的关系型数据库上进行实验，涵盖14个预测任务；

**📈 对比分析**

与基线（LightGBM、RelGNN、RelGT、STUNT、GraphCL、GraphMAE、TVE、Griffin、RT）对比，RelPrism在数据有限场景平均提升4.15% ROC‑AUC、10.75% MAE，在数据充足场景对回归任务同样表现更佳；

**⚠️ 局限性**

对单一视角或单一粒度的预训练方法存在性能不足，且对大规模数据的计算成本依赖聚类与元学习步骤，易受聚类参数和元批量大小影响；

---

## 197. Signal Temporal Logic Motion Planning via Graphs of Convex Sets

**arXiv ID:** 2605.23240 | [PDF](https://arxiv.org/pdf/2605.23240v1)

**作者:** Yu Chen `[一作]` (Shanghai Jiao Tong University), Xiang Yin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3311 | [OpenAlex ID](https://openalex.org/A5034304769)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在信号时序逻辑（STL）规范下的连续时间运动规划，旨在生成满足高层逻辑和时间要求的平滑机器人轨迹，同时遵循低层运动约束。

**💡 创新点**

提出了一种高效的框架，将定时自动机推理与凸集图（GCS）相结合，解决了STL运动规划问题，并通过最短路径问题的形式进行重构。

**🔧 技术方法**

使用了定时自动机和图的凸集（GCS）技术，结合了逻辑任务进展、实时约束、凸区域占用、平滑性要求和速度界限。

**📊 数据集**

在低维基准测试、3D四旋翼、30自由度人形机器人和UR-3机器人臂的硬件实验中进行了数值实验，验证了所提方法的有效性。

**📈 对比分析**

与现有的STL运动规划方法相比，所提方法在多个基准测试中表现出竞争力或更好的运行时间，生成的轨迹平滑且可执行。

**⚠️ 局限性**

该方法的局限性在于，尽管在特定条件下有效，但在处理更复杂的STL任务时可能会遇到可行轨迹不被接受的情况，且在某些情况下可能无法保证完整性。

---

## 198. Beyond Normal References: Discriminative Few-Shot Anomaly Detection

**arXiv ID:** 2605.23231 | [PDF](https://arxiv.org/pdf/2605.23231v1)

**作者:** Huan Wang `[一作]` (University of Wollongong), Guansong Pang `[通讯]` (Singapore Management University)

**通讯引用:** 6202 | [OpenAlex ID](https://openalex.org/A5039104219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种实用的少样本异常检测（FSAD）设置，称为判别FSAD，利用有限的正常和异常样本作为推理时的参考。

**💡 创新点**

创新点在于引入了IDEAL框架，通过学习内在偏差模式来表征可泛化的异常性，从而有效检测已知和未知的异常。

**🔧 技术方法**

使用了内在偏差学习框架（IDEAL），其中包括正常变异消除器（NVE）和内在偏差编码器（IDE）两个新组件。

**📊 数据集**

在八个真实世界数据集上进行了实验，包括工业和医学图像数据集。

**📈 对比分析**

与现有的最先进的FSAD方法相比，IDEAL在多个数据集上表现出色，能够有效泛化到未见的异常，且在性能上持续优于其他方法。

**⚠️ 局限性**

限制在于少样本异常参考的稀疏性，可能导致检测器对已见异常模式的过拟合，从而影响对未见异常的泛化能力。

---

## 199. WMAttack: Automated Attack Search for Adversarial Evaluation of World-Model Agents

**arXiv ID:** 2605.23220 | [PDF](https://arxiv.org/pdf/2605.23220v1)

**作者:** Zhixiang Guo `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 101486 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为WMAttack的自动化攻击搜索框架，用于评估世界模型代理的对抗鲁棒性。

**💡 创新点**

WMAttack通过结合表示引导的攻击检索（RGAR）和自我修正攻击搜索（SCAS），提高了对抗攻击配置的搜索效率和准确性。

**🔧 技术方法**

使用了表示引导的攻击检索（RGAR）和自我修正攻击搜索（SCAS）技术。

**📊 数据集**

在Atari和DeepMind Control任务上进行了评估，主要使用DreamerV3作为受害模型。

**📈 对比分析**

与随机搜索和Claudini基线相比，WMAttack在发现更强攻击方面表现出色，DreamerV3 Atari的标准化奖励下降从0.497提高到1.034，DMC从0.319提高到0.682。

**⚠️ 局限性**

WMAttack的局限性在于其依赖于有限的评估预算，可能无法保证全局最优的攻击配置。

---

## 200. CaST-Bench: Benchmarking Causal Chain-Grounded Spatio-Temporal Reasoning for Video Question Answering

**arXiv ID:** 2605.23216 | [PDF](https://arxiv.org/pdf/2605.23216v1)

**作者:** Mingfang Zhang `[一作]` (Woven by Toyota), Quan Kong `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CaST-Bench，基于视频的因果链式时空推理问答基准。

**💡 创新点**

首次将因果链与时空视觉证据精细对齐，并设计了全新的评估指标。

**🔧 技术方法**

采用人机协作流水线、VLM 生成描述与 QA、Mask‑based 过滤、LLM 评判等技术。

**📊 数据集**

使用 SegmentAnything‑Video 数据集中的 1,015 个自然视频生成 2,066 个因果问答。

**📈 对比分析**

与多种专有与开源 VLM 对比，模型答题准确率仅 30‑50%，远低于人类 91.9%，表明仍有较大提升空间。

**⚠️ 局限性**

主要限制在于模型难以精准定位并表达多段时空证据，且评测仍依赖人工与 LLM 判断。

---

## 201. Reinforcement Learning for Microcanonical Graph Ensemble with Assortativity Constraints

**arXiv ID:** 2605.23285 | [PDF](https://arxiv.org/pdf/2605.23285v1)

**作者:** Hoyun Choi `[一作]` (Korea Institute for Advanced Study), Deok-Sun Lee `[通讯]` (Korea Institute for Advanced Study)

**通讯引用:** 3214 | [OpenAlex ID](https://openalex.org/A5042241464)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Deep Microcanonical Graph Generator (DMGG)，一种基于强化学习的图生成框架，能在保持度数分布不变的前提下，精确地将网络的阶相关（assortativity）逼近指定目标值。

**💡 创新点**

创新点在于：①将硬约束图生成问题转化为马尔可夫决策过程，利用学习到的策略直接指引图重连；②无需预先收集满足约束的样本；③在保证精确约束的同时，保持了较高的配置多样性，并实现了至少10倍的生成速度提升。

**🔧 技术方法**

主要技术包括：强化学习中的Proximal Policy Optimization (PPO)、基于节点度信息的图表示网络、利用边重连操作的动作空间、以及通过奖励函数评估当前网络与目标assortativity的差距。

**📊 数据集**

训练使用了小型稀疏网络（N=100–1000）来自Watts–Strogatz、Erdős–Rényi、Barabási–Albert三种随机图模型；评估则扩展到更大、更稠密的网络以及未见过的拓扑（SBM、RGG、CL、HK 等），覆盖多种度分布与尺寸。

**📈 对比分析**

与传统的Soft-constraint ERGM（Metropolis–Hastings）对比，DMGG在同一目标assortativity下需要的重连次数少约10–20倍；同时其生成的图集在度独立熵（dyad-independent entropy）指标上与ERGM相近，说明多样性保持良好。硬约束使得生成的网络在二阶结构（如聚类系数）上与ERGM产生显著差异，表明更能揭示约束对二次指标的真实影响。

**⚠️ 局限性**

局限性包括：①目前仅针对单一硬约束（assortativity）设计，扩展到多重或更复杂约束仍需研究；②在极端目标值或高度稠密网络中，重连次数可能仍较高；③对非度保持的约束（如多层图、超图等）尚未实现，需要重新定义动作空间与奖励函数。

---

## 202. When Is Next-Token Prediction Useful? Marginalization, Ergodicity, Mixture Identifiability, Local Sufficiency, RAG, Tools, and Programming

**arXiv ID:** 2605.23278 | [PDF](https://arxiv.org/pdf/2605.23278v1)

**作者:** Francesco Corielli `[一作]` `[通讯]`, Francesco Corielli

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文阐述了语言模型的下一词预测能否真正有用，分析了训练过程、混合语料、文本前缀与隐藏情境之间的关系，并给出了信息论可充分性的判据。

**💡 创新点**

提出了两阶段可用性判据：①在异质语料下模型必须能够识别局部“语料体制”，②文本前缀需近似为隐藏情境的充分统计量（即条件互信息趋近零）。

**🔧 技术方法**

采用信息论（条件互信息）与概率论（混合分布、可辨识性）对语言模型的学习与推理机制进行理论推导；讨论了检索增强生成（RAG）与工具使用的条件性充分性。

**📊 数据集**

本文未进行实验，因此未使用任何公开数据集，而是以通用的自然语言、编程、数学等语料体制为例进行理论讨论。

**📈 对比分析**

由于是理论性研究，没有直接的实验对比；作者通过信息论条件和混合可识别性来说明模型在不同领域（如编程）下可能的可靠性与局限性。

**⚠️ 局限性**

主要限制在于：①对语料稳定性、代表性、平稳性、可辨识性的强假设；②模型学习的仅是文本分布，无法保证生成内容的真实性或正确性；③缺乏对模型在实际推理与检索效果的实证验证。

---

## 203. On the Performance of DCF in Full Duplex WLANs with Hidden Terminals

**arXiv ID:** 2605.23276 | [PDF](https://arxiv.org/pdf/2605.23276v1)

**作者:** Anastasios C. Politis `[一作]` (International Hellenic University), Hristos T. Anastassiu `[通讯]` (International Hellenic University)

**通讯引用:** 1375 | [OpenAlex ID](https://openalex.org/A5085600598)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文通过解析模型评估在未改造 IEEE 802.11 DCF 的全双工 WLAN 环境中，隐藏终端对吞吐量的影响。

**💡 创新点**

创新点在于同时考虑对称全双工 (SFD) 与非对称全双工 (AFD) 模式，并在标准 DCF 下计算隐藏终端概率，揭示全双工在现有 MAC 机制下几乎无提升。

**🔧 技术方法**

使用两维马尔可夫链框架推导节点传输概率 τ 与碰撞概率 p，并结合多环隐藏概率公式计算系统饱和吞吐量。

**📊 数据集**

实验使用 IEEE 802.11ac（MCS8、780 Mbps）理论参数与无误差的仿真模型，无依赖实际数据集。

**📈 对比分析**

与传统半双工模式按饱和吞吐量对比，结果显示全双工在节点数较少时提升约 1 % 左右，节点数增多后差距趋近于 0，性能提升极其有限。

**⚠️ 局限性**

主要限制是 DCF 本身设计为避免同时访问，导致隐藏终端导致多重碰撞，模型也忽略捕获效应、干扰范围等实际因素，实际部署可能更不利。

---

## 204. EvalVerse: Pipeline-Aware and Expert-Calibrated Benchmarking for Professional Cinematic Video Generation

**arXiv ID:** 2605.23271 | [PDF](https://arxiv.org/pdf/2605.23271v1)

**作者:** Songlin Yang `[一作]` (Hong Kong University of Science and Technology), Anyi Rao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5495 | [OpenAlex ID](https://openalex.org/A5067715162)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 EvalVerse 评估框架，系统化地将视频生成评估从“是否正确”扩展到“质量好坏”，通过构建基于专业电影工作流的分层评估范式、精细标注数据集以及基于 Vision‑Language Model 的链式思维（CoT）评估流程实现。

**💡 创新点**

创新点包括：① pipeline‑aware 的评估分类，覆盖预制、制作、后制三大阶段共 196 项细粒度准则；② 三阶人机校准策略（prompt‑level、fusion‑level、parameter‑level），实现专家知识的系统注入；③ 结合感知算子与自我反思的 CoT 机制，在多镜头、音视频融合场景下提供可解释的多维度评分；④ 将评估工具转化为可用于 RL 奖励与自动代理评判的基础设施。

**🔧 技术方法**

核心技术：Vision‑Language Model 双阶段 fine‑tuning（先对比学习后点值校准），感知算子组合（DINO、YOLO、SyncNet、Whisper 等）获取 deterministic evidence，CoT 生成与自我反思，任务特定 SFT 进一步对齐专家评分，基于多模态上下文的自动评分与解释生成。

**📊 数据集**

数据集：从百万级专业影视数据库抽样构建 Real‑to‑Gen 对，包含文本 prompt、参考视频、关键帧及多模态元数据；手工完成 10+ 位专业评审的 10k+ 对齐评分；覆盖多镜头、音频、视觉多模态的完整生成任务。

**📈 对比分析**

比较方法：与多款闭源/开源模型（Seedance 2.0、Kling‑v3‑Omni、Happy Horse 1.0、HoloCine、MultiShotMaster 等）在 T2V、R2V 任务下进行 win‑ratio、Spearman/ Pearson 相关性评估。EvalVerse 在大多数维度上与专家评分高度相关，性能显著优于现有基准，并能提供细粒度奖励向量支持 RL 与自动工作流。

**⚠️ 局限性**

限制：VLM 主要处理关键帧，时序感知有限；长篇叙事（>10 分钟）和极端艺术风格评估仍有挑战；评估仍需大量专业人工校准，尚未完全自动化；未来需整合为统一多模态理解任务并提升连续流处理能力。

---

## 205. MISO Downlink with Fluid Antenna Multiple Access

**arXiv ID:** 2605.23260 | [PDF](https://arxiv.org/pdf/2605.23260v1)

**作者:** Anastasios Papazafeiropoulos `[一作]` (University of Hertfordshire), Anastasios Papazafeiropoulos `[通讯]` (University of Hertfordshire)

**通讯引用:** 2147 | [OpenAlex ID](https://openalex.org/A5055961126)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文开发了一个统一的分析框架，用于多用户MISO下行系统中的流体天线多址接入（FAMA），并分析了最大比率传输（MRT）和零强迫（ZF）预编码的性能。

**💡 创新点**

创新点在于首次提供了流体天线用户在多用户下行系统中与空间相关性和线性预编码相互作用的分析，推导了每个端口的信号干扰比（SIR）分布，并建立了严格的中断概率界限。

**🔧 技术方法**

使用了最大比率传输（MRT）和零强迫（ZF）预编码技术，并结合流体天线的空间相关性模型进行分析。

**📊 数据集**

使用了多用户MISO系统的模拟数据集，考虑了不同的天线数量和用户数量，以及流体天线的端口配置。

**📈 对比分析**

与现有的FAMA研究相比，本文提供了闭式的每个端口SIR特征，分析了MRT和ZF的性能，结果表明MRT在基站具有充足空间自由度时，能够实现更弱的端口相关性和更大的选择增益。

**⚠️ 局限性**

限制在于未考虑流体天线的机械缺陷和硬件不准确性，这可能会影响实际性能。所报告的结果应视为乐观的上限。

---

## 206. A Simple Plug-in for Improving Eviction-Based KV Cache Compression

**arXiv ID:** 2605.23258 | [PDF](https://arxiv.org/pdf/2605.23258v1)

**作者:** Yuping Lin `[一作]` (Michigan State University), Subhabrata Mukherjee `[通讯]` (Hippocratic AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为VECTOR的三路分配框架，旨在通过引入重构能力感知的近似方法来改善现有的基于驱逐的KV缓存压缩方法。

**💡 创新点**

VECTOR通过将令牌的重要性与重构能力分开，扩展了二元保留/驱逐决策，形成保留、近似和驱逐三种状态，从而在相同的内存预算下实现更好的性能。

**🔧 技术方法**

使用了离线普通最小二乘法（OLS）进行值的重构，并结合了基于重要性的驱逐策略。

**📊 数据集**

使用了C4数据集来收集键值激活对，并在多个长上下文基准上进行实验。

**📈 对比分析**

与现有的驱逐基线（如KeyDiff、SnapKV、KVzip和PyramidKV）进行比较，VECTOR在高压缩比下（如0.75和0.90）显著提高了下游性能，尤其是在查询无关的基线中表现更为突出。

**⚠️ 局限性**

VECTOR的局限性包括：近似比例p_a是通过经验公式设定的，而不是针对每个样本或每层进行优化；对于查询感知的基线（如SnapKV和PyramidKV），性能提升有限，因为这些方法已经有效保留了与查询相关的令牌。

---

## 207. Are Frontier LLMs Ready for Cybersecurity? Evidence for Vertical Foundation Models from Dual-Mode Vulnerability Benchmarks

**arXiv ID:** 2605.23243 | [PDF](https://arxiv.org/pdf/2605.23243v1)

**作者:** Vivek Dahiya `[一作]` (super-intel), Chandra Khatri `[通讯]` (super-intel)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估前沿大型语言模型（LLMs）在网络安全中的应用能力，通过白盒功能级漏洞检测和黑盒Web应用安全测试进行双模式基准测试。

**💡 创新点**

提出了针对网络安全的垂直基础模型的必要性，强调方法论而非规模是提高检测率的关键。

**🔧 技术方法**

使用了前沿模型（如GPT-5.4、Codex 5.3等）和领域专用模型，结合结构化渗透测试方法和图形化推理架构（ARG）。

**📊 数据集**

使用了五个开源的生产风格Web应用程序，包含118个真实漏洞，覆盖20多个CWE类别。

**📈 对比分析**

与其他方法比较，前沿模型在白盒检测中产生10-50%的假阳性率，黑盒测试的真实覆盖率仅为4-8%，而使用结构化方法的领域专用模型检测率超过50%。

**⚠️ 局限性**

局限性在于评估仅限于授权的本地托管基准应用程序和功能级白盒数据集，未能捕捉生产安全程序的所有操作约束。

---

## 208. Entropy Equivalence Testing

**arXiv ID:** 2605.23225 | [PDF](https://arxiv.org/pdf/2605.23225v1)

**作者:** Clément L. Canonne `[一作]` (University of Sydney), Joy Qiping Yang `[通讯]` (University of Sydney)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5004775145)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了熵等价测试的问题，这是对已研究的接近测试问题的放松，算法需要区分两个未知分布的熵差异。

**💡 创新点**

提出了一种时间和样本效率高的算法，样本复杂度显著低于接近测试的最优样本复杂度，并且为低度贝叶斯网络的接近测试提供了首个非平凡的测试算法。

**🔧 技术方法**

使用了熵等价测试的算法，结合了对Hellinger距离和交叉熵差异的测试。

**📊 数据集**

使用了两个未知分布的样本，具体数据集未明确给出。

**📈 对比分析**

与基于全学习的基线方法相比，样本复杂度和运行时间都有显著改善，具体性能通过样本复杂度的数学表达式给出。

**⚠️ 局限性**

算法的局限性在于需要对分布的结构做出假设，且在高维数据情况下可能仍然面临挑战。

---

## 209. On APN Exponents and the Differential and Boomerang Properties of Binomials in Characteristic 3

**arXiv ID:** 2605.23224 | [PDF](https://arxiv.org/pdf/2605.23224v1)

**作者:** Namhun Koo `[一作]` (Sungkyunkwan University), Byunguk Kim `[通讯]` (Sungkyunkwan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了特征3下二项式F_r(x)=x^r(1+χ(x))的差分与布门格特性，并系统分类出低布门格均匀性（0或1）的指数。

**💡 创新点**

首次利用Zha-Wang构造的APN指数以及2·3^n-1/2+1等指数证明F_r在特征3下可实现布门格均匀性0，并完整给出3^n-3指数的布门格谱。

**🔧 技术方法**

采用代数数论、字符求和、差分谱与布门格谱分析，以及SageMath数值验证等技术。

**📊 数据集**

使用SageMath对有限域F_{3^n}（n≤13）进行全搜索，得到低均匀性指数集合。

**📈 对比分析**

与已有文献中的差分/布门格上界对比，证明在特征3下布门格均匀性可低于一般上界2；数值表明对应指数几乎覆盖所有已知APN指数。

**⚠️ 局限性**

仍未证明局部PN性与布门格均匀性0的充分性，并且布门格均匀性1的通用构造尚缺失。

---

## 210. AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery

**arXiv ID:** 2605.23204 | [PDF](https://arxiv.org/pdf/2605.23204v1)

**作者:** Guiyao Tie `[一作]` (Huazhong University of Science and Technology), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 35560 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了AI在科学研究中的演进，提出了AutoResearch框架并对现有系统进行归类与评估

**💡 创新点**

引入了从L0到L4的五级自主谱和Vibe Research概念，强调工作流层面的责任与控制分配，提供统一的评价维度（新颖性、有效性、影响力、可靠性、溯源）

**🔧 技术方法**

采用工作流层面技术分解（文献支撑、假设生成、实验执行、反馈验证、报告沟通）并结合现有大型语言模型、工具使用、实验室接口、混合主动式代理与可持续工作空间

**📊 数据集**

参考了公开的AutoResearch系统与基础设施（如The AI Scientist、Agent Laboratory、NanoResearch等）及评测基准（AIRS‑Bench、FIRE‑Bench、ResearchBench 等），并使用相关公开数据集（如PubMed、ArXiv、实验室仿真数据等）进行对比

**📈 对比分析**

通过将系统映射到L0–L4并按技术维度评估，展示了系统在工作流自动化层级与功能范围的差异，但未给出统一的数值性能指标，强调现阶段评测更多关注科学可信度而非纯任务完成度

**⚠️ 局限性**

局限性包括：缺乏成熟的L3–L4自主系统，评测基准与数据集不足，系统间可比性受限，缺乏完整的可解释性、可复现性与伦理合规性保障，且在跨域泛化与真实实验验证上仍面临挑战

---

## 211. Self-Refining Topology Optimization via an LLM-Based Multi-Agent Framework

**arXiv ID:** 2605.23273 | [PDF](https://arxiv.org/pdf/2605.23273v1)

**作者:** Hyunjee Park `[一作]` (Ulsan National Institute of Science and Technology), Hayoung Chung `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5045862346)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了TopOptAgents，一个多智能体系统，旨在自动化拓扑优化过程中的设计和决策制定。

**💡 创新点**

创新点在于引入了自我精炼的多智能体框架，能够在拓扑优化的各个关键阶段进行协作和迭代改进，从而提高设计的质量和一致性。

**🔧 技术方法**

使用了基于大型语言模型（LLM）的智能体，通过迭代自我精炼循环进行问题表述、验证、代码生成和执行，以及优化结构的质量评估。

**📊 数据集**

使用了三个基准拓扑优化问题，包括悬臂梁的合规性最小化、MBB梁的合规性最小化和L形梁的应力最小化，涵盖了不同的文献覆盖率和数值特征。

**📈 对比分析**

与单一LLM基线模型相比，TopOptAgents在处理更复杂的问题时表现出更高的成功率，尤其是在基线模型难以处理的情况下，成功率从10%提高到80%。

**⚠️ 局限性**

限制在于当前框架依赖于预训练的LLM，可能无法处理那些文献覆盖较少或未被充分记录的拓扑优化问题，未来需要通过模型微调或检索增强生成来扩展其适用范围。

---

## 212. Accelerating Divisible Load Processing Through Machine Learning: A Practical Framework for Large-Scale Workloads

**arXiv ID:** 2605.23247 | [PDF](https://arxiv.org/pdf/2605.23247v1)

**作者:** Bharadwaj Veeravalli `[一作]` (National University of Singapore), Bharadwaj Veeravalli `[通讯]` (National University of Singapore)

**通讯引用:** 6580 | [OpenAlex ID](https://openalex.org/A5070594442)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用机器学习方法预测单层树网络（SLTN）中可分负载理论（DLT）的最优处理时间，避免显式求解递归方程；

**💡 创新点**

首次将监督学习（前馈神经网络）应用于DLT最优时间预测，既保持95%以上精度，又实现10‑100倍加速；

**🔧 技术方法**

采用16维统计特征的三层前馈神经网络（ReLU+dropout），训练并推理；

**📊 数据集**

利用100,000条合成配置数据，覆盖3–20个节点、1–100 GB负载、不同计算速度和链路带宽；

**📈 对比分析**

与传统DLT解析求解比较，R²>0.99、MAPE≤7.9%，推理时间<1 ms，速度提升10‑100×；

**⚠️ 局限性**

仅适用于单层树、静态配置，对极大或高度异构系统、动态故障、结果收集或多层层次场景的表现有限。

---

## 213. Convex Optimization for Alignment and Preference Learning on a Single GPU

**arXiv ID:** 2605.23244 | [PDF](https://arxiv.org/pdf/2605.23244v1)

**作者:** Miria Feng `[一作]` (Stanford University), Mert Pilanci `[通讯]` (Stanford University)

**通讯引用:** 1180 | [OpenAlex ID](https://openalex.org/A5001436196)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 COALA，一种利用凸优化进行单 GPU 偏好对齐的轻量级方法；

**💡 创新点**

创新点在于将偏好学习重新表述为凸优化问题，去除参考模型并利用 cvxNN + ADMM（CRONOS）实现全局收敛保证；

**🔧 技术方法**

使用两层 ReLU 的凸神经网络、Alternating Direction Method of Multipliers（ADMM）及其变体 CRONOS、JAX 的 JIT 编译以及基于预训练特征的逻辑回归；

**📊 数据集**

评估数据集包括自制 EduFeedback（26621 交互，65606 偏好对），UltraFeedback、IMDb 以及 HelpSteer；

**📈 对比分析**

与 DPO、ORPO、SFT 等方法比较，COALA 在 AlpacaEval2、MT‑Bench、ArenaHard 等指标上保持竞争力，同时在单 RTX‑4090 上 TFLOPs 下降约 80%，训练时间缩短数倍；

**⚠️ 局限性**

局限在于冻结基础模型可能导致表达能力受限，尤其是需要深层语义重塑或极细粒度偏好判断的场景；

---

## 214. Cogniscope: A Synthetic Longitudinal Benchmark and Browser-Based Evaluation Framework for Early-Risk Cognitive AI Systems

**arXiv ID:** 2605.23242 | [PDF](https://arxiv.org/pdf/2605.23242v1)

**作者:** Mahfuza Farooque `[一作]` (Pennsylvania State University), Asish Kondragunta `[通讯]` (Pennsylvania State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个用于评估长期早期风险AI系统的综合框架Cogniscope，包含可配置的合成模拟引擎、Chrome扩展数据采集器、18表关系模式及时间感知评估协议。

**💡 创新点**

创新点在于提供了同一框架内的模拟生成、可部署采集工具、时间感知指标（ERDE、TTD）以及公开的可重复使用数据集，填补了早期风险评估中缺乏可复现、可扩展评估生态的空白。

**🔧 技术方法**

采用的技术包括Python模拟引擎、Llama-3/ GPT-4o-mini生成的多选题、Chrome Manifest V3扩展收集YouTube交互事件、Supabase/PostgreSQL数据库、GRU/TCN/Transformer等时序模型进行评估。

**📊 数据集**

使用的合成数据集为：200用户×200天的200,000条交互记录，以及504个带九种行为模式的会话数据，共计约5,040条问答记录。

**📈 对比分析**

通过规则阈值、规则基分类器以及时间感知指标比较模型性能，基线阈值检测在MCI状态下95%用户在10天内检测到，Coherence-Only模型F1≈0.88/0.82；规则基分类器宏F1≈0.47，显示仅基于阈值难以应对自然交互噪声。

**⚠️ 局限性**

主要限制在于全部数据为人工合成，缺乏真实临床验证，模型在合成条件下表现优异但可能无法泛化到真实用户；模拟过程不包含退化、波动或人口多样性，数据规模有限。

---

## 215. Self-supervised Adversarial Purification for Graph Neural Networks

**arXiv ID:** 2605.23239 | [PDF](https://arxiv.org/pdf/2605.23239v1)

**作者:** Woohyun Lee `[一作]` (Sungkyunkwan University), Hogun Park `[通讯]` (Sungkyunkwan University)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5071646906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种自监督的图神经网络对抗净化框架 GPR-GAE，通过在净化器中使用多尺度 Generalized PageRank 滤波器实现图结构恢复，并与原始分类器分离以同时提升准确性与鲁棒性。

**💡 创新点**

创新点在于：① 将准确性与鲁棒性完全解耦，使用独立的自监督净化器；② 设计多滤波器的 GPR-GAE，可学习多尺度邻域特征并避免过度平滑；③ 引入多步迭代净化与理论证明的收敛性，显著提升对高强度攻击的抵御能力。

**🔧 技术方法**

主要技术包括自监督图自动编码器、Generalized PageRank 滤波器、多步迭代净化、对抗样本随机生成与自监督恢复/对称损失、以及基于梯度的对抗攻击 PRBCD/LRBCD。

**📊 数据集**

实验使用了 Cora、Cora-ML、Citeseer、Pubmed、OGB-arXiv、Chameleon 等多规模图数据集。

**📈 对比分析**

与传统对抗训练、鲁棒 GNN（EvenNet、SoftMedianGDC）以及基于启发式的净化方法（Jaccard-GCN、SVD-GCN、GOOD-AT）相比，GPR-GAE 在适应性攻击下的准确率提升显著（例如在 PRBCD ϵ=0.5 下提升 15.7pp），同时保持与基准分类器相近的清洁数据准确率，且在大规模数据集上表现更好。

**⚠️ 局限性**

局限性包括：对纯粹基于特征攻击的鲁棒性评估不足；净化过程仍需多步迭代，训练和推理时的计算成本相对较高；在某些极端攻击预算下，仍可能出现一定的性能下降。

---

## 216. MASQ: Accelerating Masked Diffusion via Stage-Wise Multi-Precision Quantization

**arXiv ID:** 2605.23226 | [PDF](https://arxiv.org/pdf/2605.23226v1)

**作者:** Seeyeon Kim `[一作]` (KAIST), Joo-Young Kim `[通讯]` (KAIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种硬件-软件协同设计的加速器，专门用于高效执行蒙版扩散模型；

**💡 创新点**

创新点在于引入基于空间与语义重要性的阶段化多精度量化与时钟感知调度，并配合专用的掩码管理单元，实现对不同区域的自适应低精度处理；

**🔧 技术方法**

采用了MXINT8/4/2块浮点量化、动态掩码扩张、语义重要性修正、时钟感知精度分配以及对非矩阵操作的精度感知优化；

**📊 数据集**

在 EditBench、LAION 以及 LSUN Church 数据集上进行评测；

**📈 对比分析**

与 NVIDIA A100 和 Jetson Orin NX 的对比显示，平均可获得 61.35%（服务器端）/73.58%（边缘端）速度提升和 54.88%/71.45% 能耗降低，最高可达 16.06×/5.39× 的加速比和 4.18×/4.93× 的能效提升；

**⚠️ 局限性**

局限性在于需要专门的硬件支持，且对掩码形状和大小的自适应能力仍需进一步验证，同时多精度调度方案对不同扩散模型的迁移性有待探索。

---

## 217. 6G Communication Networks Enabling Embodied Agents: Architecture and Prototype

**arXiv ID:** 2605.23263 | [PDF](https://arxiv.org/pdf/2605.23263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 218. BCTuner: LLM-Guided Monte Carlo Tree Search for Efficient Blockchain Knob Tuning

**arXiv ID:** 2605.23280 | [PDF](https://arxiv.org/pdf/2605.23280v1)

**作者:** Yaoyi Deng `[一作]` (Beihang University), Shuai Ma `[通讯]` (Beihang University)

**通讯引用:** 3930 | [OpenAlex ID](https://openalex.org/A5006980420)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于大语言模型（LLM）指导的蒙特卡洛树搜索（MCTS）框架 BCTuner，用于自动调优许可链区块链的配置参数（knob tuning）。

**💡 创新点**

创新点在于：①将 LLM 作为可执行动作生成器，逐步构造、验证、评估、细化配置，而非一次性生成完整配置；②整合多源知识（手册、硬件、网络）实现语义化、上下文感知的推理；③在 MCTS 过程中加入自适应剪枝与反馈驱动细化，显著减少无效评估；④通过验证与修复动作提高配置可行性。

**🔧 技术方法**

使用的技术包括：大型语言模型（GPT‑5、Gemini 等）进行知识抽取与动作生成；蒙特卡洛树搜索与 UCT 策略进行结构化搜索；规则与性能驱动的剪枝策略；多源知识库（knob、硬件、网络）构建；实验平台（Hyperledger Fabric、ChainMaker）与 Caliper/chainmaker‑bench 基准。

**📊 数据集**

实验数据集主要为 Hyperledger Fabric（Simple、SmallBank 合约）与 ChainMaker（chainmaker‑bench）在不同节点规模与网络拓扑下的 5G 内部网络实验，使用 50,000 事务固定速率模式评估吞吐量（TPS）。

**📈 对比分析**

与默认配置、GPTuner、Athena、SMAC、GP 等方法对比，BCTuner 在 Fabric 上平均提升 61.6%（Simple）/106.7%（SmallBank）TPS，迭代次数比 GPTuner 快 7–8 倍、比 Athena 快 8 倍、比 SMAC/GP 快 6–8 倍；在 ChainMaker 上同样获得 120% 以上 TPS 提升，且收敛更快；同时将评估次数降低 4–8 倍。

**⚠️ 局限性**

局限性：仍需手工抽取并维护多源知识库，LLM 推理成本随模型大小变化；在极大配置空间或极端网络/硬件环境下，剪枝与验证的阈值可能需重新调优；对全新区块链系统的迁移性需要进一步验证。

---

## 219. Enhancing Deep Neural Network Reliability with Refinement and Calibration

**arXiv ID:** 2605.23249 | [PDF](https://arxiv.org/pdf/2605.23249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 220. U-CESE: Unified Clip-based Event Search Engine for AI Challenge HCMC 2025

**arXiv ID:** 2605.23274 | [PDF](https://arxiv.org/pdf/2605.23274v1)

**作者:** Duc-Nhuan Le `[一作]` (Vietnam National University), Minh-Hoang Le `[通讯]` (Vietnam National University)

**通讯引用:** 32579 | [OpenAlex ID](https://openalex.org/A5100623040)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的基于剪辑的视频事件检索系统 U-CESE。

**💡 创新点**

创新点包括将多模态剪辑合并为统一算法、使用基于 JPEG 文件大小的无训练关键帧提取 DAKE、以及时序一致的 ReCap 生成字幕。

**🔧 技术方法**

采用 MobileCLIP、OpenCLIP、InternVL、BEiT-3 等多模态编码器，结合 DAKE、ReCap、Elasticsearch/FAISS 等检索技术。

**📊 数据集**

在 AI Challenge HCMC 2025 的 1478 条 TV 综艺视频（324 小时）数据集上进行实验。

**📈 对比分析**

与 AutoShot 比较，DAKE 在 ρ=0.02 时召回率超过 80%；ReCap 在字幕生成上显著提升时序一致性，系统在 TRAKE、VQA 等任务中获得优异成绩。

**⚠️ 局限性**

局限在于 JPEG 变异方法对极小场景变化敏感，关键帧提取仍需经验调参，且模型在跨域视频中的泛化能力待进一步验证。

---

## 221. When Good Equations Get Bad Scores: Improving Symbolic Regression Through Better Parameter Optimization

**arXiv ID:** 2605.23272 | [PDF](https://arxiv.org/pdf/2605.23272v1)

**作者:** Boxiao Wang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jian Cheng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在符号回归中提出了一种名为 SAGE‑Fit 的内循环参数拟合框架，能够更准确、更高效地评估候选结构的数值性能，从而提升整体搜索效果。

**💡 创新点**

创新点在于同时利用表达式树的结构先验（通过变量投影实现维度折叠）和语义先验（通过功能空间最远点采样保证多起点多样性），以及针对投影非线性子空间的投影高斯-牛顿求解器，三者协同解决了“好结构坏分数”的瓶颈。

**🔧 技术方法**

使用了：1）基于 AST 的条件线性参数识别与变量投影；2）功能空间最远点采样（FS‑FPS）进行语义多起点初始化；3）投影高斯‑牛顿（Projected Gauss‑Newton）结合信赖域修正实现快速局部收敛；4）伪逆正则化和多起点重启；5）对外提供 plug‑and‑play 接口。

**📊 数据集**

在 LLM‑SRBench 的 LSR‑Synth 数据集（128 个由物理、化学、生物、材料等领域的复杂方程构成）上进行实验。

**📈 对比分析**

将 SAGE‑Fit 作为现有 SR 框架（PySR、uDSR、LaSR、LLM‑SR）的参数优化子模块进行替换，对比同框架原版。实验表明：符号准确率提升约 3‑4 倍，数值误差（NMSE）显著下降，误差对数比提升多达 5–6 阶；在结构进展银行（SPB）中，误差率（lost rate）从 51.5% 降至 8.0%，显示显著的进度保留率提升。

**⚠️ 局限性**

局限性包括：1）仍依赖外循环的结构搜索质量，若结构本身不佳则提升有限；2）对表达式树的解析与投影开销在极大表达式上可能不容忽视；3）对高维、极度非线性问题的鲁棒性尚未完全验证；4）在非标准 SR 场景（如分段函数、符号约束）中的适用性尚待进一步研究。

---

## 222. Design and Report Benchmarks for Knowledge Work

**arXiv ID:** 2605.23262 | [PDF](https://arxiv.org/pdf/2605.23262v1)

**作者:** Yining Hua `[一作]` (Harvard University), Levi Lian `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了三步框架，用于将知识工作基准测试的工作活动、测试环境和工作产出明确化并进行报告；

**💡 创新点**

创新点在于将工作活动、测试设置和评分对象三维度系统化，并构建了18个跨职业的工作活动词表，以提高基准结果与真实工作能力之间的可解释性；

**🔧 技术方法**

主要技术为对O*NET任务陈述的筛选、重写、嵌入和聚类（利用UMAP+HDBSCAN）生成工作活动词表；

**📊 数据集**

使用了O*NET 30.2任务陈述数据，结合GDPval、OfficeQA Pro和APEX‑SWE三个公开基准案例进行验证；

**📈 对比分析**

通过案例分析展示了不同基准设计对工作能力主张的支持范围，并指出在记录保持、分析、整合等工作中的评分差距，说明目前基准得分在多大程度上能反映真实工作表现；

**⚠️ 局限性**

局限性包括：基准仅评估有限工作活动与设置，未涵盖部署阶段的多样性；工作活动词表基于通用任务陈述，缺乏对具体组织流程的细粒度捕捉；以及缺少针对真实工作流的实证验证。

---

## 223. SimInsert: Seamless Video Object Insertion via Regional Sparse Attention Fusion

**arXiv ID:** 2605.23245 | [PDF](https://arxiv.org/pdf/2605.23245v1)

**作者:** Xinyu Chen `[一作]` (Nanjing University), Zili Yi `[通讯]` (Nanjing University)

**通讯引用:** 2949 | [OpenAlex ID](https://openalex.org/A5007023601)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练‑free的视频对象插入框架 SimInsert，利用单帧编辑和文本驱动的运动传播实现无缝插入

**💡 创新点**

创新点包括区域注意力克隆保持背景完整、稀疏注意力融合实现对象与场景的自然交互、以及潜在空间刷新抑制噪声漂移

**🔧 技术方法**

核心技术为图像‑视频扩散模型（如 Wan2.1）上的双路径注意力机制、稀疏融合与潜在刷新算法

**📊 数据集**

使用自构建的基准数据集，结合 DAVIS、MiniMax‑Remover 与 FLUX Kontext 生成的真实与合成视频

**📈 对比分析**

与 Pix2Video、FateZero、ConsistI2V、AnyV2V 等基线对比，SimInsert 在 PSNR、SSIM、LPIPS、VFID、CLIP‑I/T 等指标上均显著领先，提升约 18.8% PSNR、20.1% SSIM、下降 44.1% LPIPS

**⚠️ 局限性**

局限性包括对大尺寸或极长视频的处理仍受限于模型的时间步与显存需求，且对复杂光照/视角变化的适应性尚未完全验证

---

## 224. StereoGenBench: A Synthetic Multi-Camera Benchmark for Stereo Generation under Controlled Baseline Regimes

**arXiv ID:** 2605.23237 | [PDF](https://arxiv.org/pdf/2605.23237v1)

**作者:** Yangzhi Cui `[一作]` (Washington University in St. Louis), Nathan Jacobs `[通讯]` (Washington University in St. Louis)

**通讯引用:** 25605 | [OpenAlex ID](https://openalex.org/A5060280374)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了 StereoGenBench，一个使用 Unreal Engine 合成的六相机阵列数据集，提供多基线、已标定的 RGB、度量深度、相机内参和姿态，用于立体图像/视频生成、几何估计和视图合成的基准评估。

**💡 创新点**

创新点包括：①场景配对、已标定多基线、密集度量深度和完整相机姿态在同一源中统一发布；②通过可控基线（IPD_Gaussian、Uniform、Pairwise_Uniform）和焦距采样实现基线敏感度和相机一致性分析；③设计三层推理协议（oracle geometry、calibrated target camera、monocular）和多维度评价指标，拆解方法性能的不同维度。

**🔧 技术方法**

技术手段：Unreal Engine + Python 渲染管线生成同步六相机序列；六相机阵列实现多基线；采样策略覆盖窄基线（人眼距离）到宽基线；评估指标包括 PSNR、SSIM、LPIPS、匹配误差、P-PSNR、SD、FID、FVD 等；参考实现使用 GenStereo、StereoDiffusion、ZeroStereo、StereoSpace、Mono2Stereo 等公开模型。

**📊 数据集**

数据集：StereoGenBench（约8493个场景，7.8k 训练 + 739 评估），包含室内、户外、城市、自然、科幻、风格化等多种地图，15个角色与 30 个动作；并引用现有真实/合成基准（KITTI、Cityscapes、Scene Flow 等）做对比参考。

**📈 对比分析**

比较方法：在不同推理层次（G0、G1、G2）下使用统一评价分支（IPD_Gaussian vs Uniform）和多维指标进行对比。结果显示：G0（oracle geometry）在 PSNR、SSIM、LPIPS 上最高（PSNR≈27–29、LPIPS<0.12）；G1 与 G2 性能显著下降，尤其 G2 的 disparity‑scale drift 随基线增大而显著（SD 在 [1,10) cm 为 2.63，[100,150] cm 为 50.85）。Uniform 分支普遍更难通过，表明宽基线下模型更易出现失配。

**⚠️ 局限性**

限制：①仅为合成数据，真实世界泛化需进一步研究；②基于有限的 Unreal 资产，存在域偏差；③人类头像来自有限资产池，缺乏多样性；④评估侧重单视角生成，未完整测量时序一致性；⑤自动化过滤和人工审核虽严格，但仍可能存在视觉伪影；⑥未评估训练过程对基线泛化的影响。

---

## 225. Foundation Protocol: A Coordination Layer for Agentic Society

**arXiv ID:** 2605.23218 | [PDF](https://arxiv.org/pdf/2605.23218v1)

**作者:** Bang Liu `[一作]` (FoundationAgents), Chenglin Wu `[通讯]` (FoundationAgents)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并提出了一套名为 Foundation Protocol（FP）的基础协议，用于实现代理、工具、人员和机构之间的统一身份、组织、交互与经济治理；

**💡 创新点**

创新点在于将实体、会话、活动、信封、事件、收据与溯源等核心对象统一为图结构，并通过可扩展的 profile/bridge 机制实现跨协议组合、事件驱动的交互和 ledger‑agnostic 的经济凭证；

**🔧 技术方法**

技术上采用图数据库范式、事件流与增量签名、可插拔身份（DID、WebPKI 等）、可扩展的协议面向接口（如 MCP、A2A、DIDComm 等）和可配置的传输与路由层；

**📊 数据集**

论文未使用任何特定数据集，主要基于理论设计和案例演示；

**📈 对比分析**

未进行实验对比，主要通过设计原理、架构图和应用场景案例说明协议可行性；

**⚠️ 局限性**

局限性包括缺乏完整的规范实现细节、跨域互操作性仍需进一步验证、性能与安全评估不足以及对复杂经济与治理情景的深度探讨尚待完成。

---

## 226. Lipschitz Optimization for Formal Verification of Homographies

**arXiv ID:** 2605.23203 | [PDF](https://arxiv.org/pdf/2605.23203v1)

**作者:** Jean-Guillaume Durand `[一作]` (Joby Aviation), Alessio Lomuscio `[通讯]` (Safe Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了针对深度神经网络的3D投影变换（齐次变换）的正式验证框架，先推导出相机姿态扰动下的闭式齐次矩阵，再利用分段线性近似与Lipschitz优化得到紧致的像素值上下界，随后将这些界限输入Venus等完整验证器实现鲁棒性证明。

**💡 创新点**

其创新点在于首次实现对项目式几何变换的正式验证，扩展了PWL算法以处理非仿射的齐次变换，并通过解析的Lipschitz常数实现约89%加速、约7%更紧的界限，同时揭示了现有基准网络在3D相机运动下的脆弱性。

**🔧 技术方法**

主要技术包括相机几何模型的齐次矩阵推导、分段线性（piecewise-linear）像素值逼近、Lipschitz优化求解最大误差、解析梯度与连续性分析、Venus MILP后端以及针对双线性插值的边界处理。

**📊 数据集**

实验使用的公开数据集包括MNIST、CIFAR-10、GTSRB（VNN-COMP基准）以及航空跑道可见性数据集LARD，分别用于评估不同规模模型与实际安全应用。

**📈 对比分析**

与原PWL方法对比，本文在小幅变换时BaB步数降低71%、总耗时提升89%，在大幅变换时仍保持约40%的加速；在VNN-COMP基准中，对3D扰动无鲁棒性案例，跑道可见性分类器在10cm平移下仅16%鲁棒，1°旋转仅1%。

**⚠️ 局限性**

局限性包括对平面假设的依赖、对大幅变换时非线性加剧导致鲁棒性下降、插值填充方式对结果影响显著、需要通过域划分提升精度以及在非平面或遮挡情形下验证效果未知。

---

## 227. Multi-Gate Residuals

**arXiv ID:** 2605.23259 | [PDF](https://arxiv.org/pdf/2605.23259v1)

**作者:** Zhizhan Zheng `[一作]` (Shanghai Yichuang Information Technology Co Ltd), Hongquan Zhou `[通讯]` (Shanghai Yichuang Information Technology Co Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的多门残差结构（MGR），旨在通过多流更新特征表示，解决深度残差层中激活增长不受限的问题，同时避免额外的通信开销。

**💡 创新点**

MGR通过简单的评分和门控机制保持多流上下文，并结合注意力池化（Attention Pooling）提取隐藏状态，显著简化了现有深度注意力架构的复杂性，同时提高了模型性能。

**🔧 技术方法**

使用了多门机制和注意力池化技术，结合了独立和竞争的门控机制来更新残差流。

**📊 数据集**

使用了FineWeb-10BT数据集，该数据集包含约100亿个标记，进行了从头开始的语言模型训练。

**📈 对比分析**

与其他相关架构（如PreNorm、mHC-lite、AttenRes）进行比较，MGR在所有模型规模和残差流配置中均表现出色，尤其在n=4和n=8的配置下，MGR的训练和验证损失均优于其他方法。

**⚠️ 局限性**

尽管MGR在性能上表现优越，但仍需进一步验证其在更大规模模型中的优势，以及在不同数据集上的适用性。

---

## 228. Assessing Predictive Models for Fairness Based on Movement Patterns

**arXiv ID:** 2605.23234 | [PDF](https://arxiv.org/pdf/2605.23234v1)

**作者:** Francesco Lettich `[一作]` (National Research Council), Chiara Renso `[通讯]` (National Research Council)

**通讯引用:** 3675 | [OpenAlex ID](https://openalex.org/A5054358438)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于个体运动模式的预测模型公平性评估框架，先对轨迹进行停留-移动分割，再通过多分辨率地理分区与空间扫描统计检测并定位不公平热点。

**💡 创新点**

创新点在于将公平性从单一地点扩展到运动模式，利用多尺度地理分区和空间扫描统计实现对多重区域热点的检测；并提出针对运动模式的不公平数据生成协议。

**🔧 技术方法**

技术手段包括轨迹停留点检测、对象-单元映射、频繁项集挖掘确定候选单元、基于Bernoulli的空间扫描统计（最大似然比）以及Monte Carlo模拟验证。

**📊 数据集**

实验使用了由亚特兰大100,000名模拟个体在10天内产生的约720M采样点的运动数据（压缩后约19.8M点），并基于此生成了多组不公平可审计数据（每组1,000个数据集）。

**📈 对比分析**

通过统计功效、灵敏度和PPV三指标评估，与传统空间公平性方法对比实验显示：多分辨率设置下检出率可达1，灵敏度高，但PPV随热点规模和不公平幅度增大而下降，体现了多分辨率的权衡。

**⚠️ 局限性**

局限性包括：仅验证二分类模型；使用均匀方格分区；停留点阈值设定可能影响结果；未考虑移动段信息；仅在合成数据上测试，真实数据验证仍待开展。

---

## 229. DepthAgent: Towards Better Universal Depth Estimation via Sample-wise Expert Selection

**arXiv ID:** 2605.23281 | [PDF](https://arxiv.org/pdf/2605.23281v1)

**作者:** Jie Zhu `[一作]` (Michigan State University), Xiaoming Liu `[通讯]` (Michigan State University)

**通讯引用:** 21325 | [OpenAlex ID](https://openalex.org/A5100409052)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了DepthAgent，一种基于视觉语言模型的代理系统，通过分析场景与相机信息，动态选择并融合多种深度估计专家，以实现对不同相机几何（透视、鱼眼、全景）下的单目度量深度估计；

**💡 创新点**

核心创新在于将深度估计转化为输入适配的专家选择与融合问题，利用代理式工具调用与多回合推理实现样本级动态决策，并通过多奖励强化学习同时优化几何精度与计算效率；

**🔧 技术方法**

采用了视觉语言模型（Qwen2.5‑VL‑3B）作为代理主体，使用GRPO强化学习框架进行多奖励微调，设计了场景感知奖励、专家选择先验奖励与效率感知奖励，并实现工具调用与融合策略；

**📊 数据集**

在六大基准上评估：透视（NYU‑V2、KITTI、DDAD等）、鱼眼（e.g., Megadepth‑FishEye）、全景（Matterport3D、360‑KITTI等），并通过视角变换生成ERP版数据；

**📈 对比分析**

与单一专家、固定融合策略、随机选择及MLP路由器等基线对比，DepthAgent在各相机域均取得更高的δ₁、AbsRel、RMSE指标，尤其在最难10%样本上显著提升，且在“Fast”模式下保持较低的推理时延；

**⚠️ 局限性**

局限性包括依赖预训练视觉语言模型的计算开销、对专家模型库的构建与维护需求、奖励函数的手工调参以及在极端畸变或极少量标注数据下的泛化可能受限。

---

## 230. ChainFlow-VLA: Causal Flow Planning with Vision-Language Models

**arXiv ID:** 2605.23270 | [PDF](https://arxiv.org/pdf/2605.23270v1)

**作者:** Xiyang Wang `[一作]` (Afari Intelligent Drive), Mu Yang `[通讯]` (Afari Intelligent Drive)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ChainFlow-VLA框架，将自动驾驶轨迹规划拆分为自回归生成（Chain）与基于VLM条件的残差扩散（Flow）两步；

**💡 创新点**

将因果生成与全局优化统一于单一概率模型，并把VLM隐藏状态用于残差空间的语义引导；

**🔧 技术方法**

自回归网络、扩散Transformer（DiT）、视觉语言模型（InternVL），以及基于WTA的训练策略；

**📊 数据集**

NAVSIM v1公开数据集；

**📈 对比分析**

与多种端到端与VLA方法对比，ChainFlow-VLA在PDMS指标上取得94.8，达到甚至匹配人类驾驶水平，显著优于前置模型；

**⚠️ 局限性**

VLM引导仍基于通用驾驶理解，缺乏专门针对轨迹评估的精细化指令，未来可开发更适合残差优化的VLM。

---

## 231. Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution

**arXiv ID:** 2605.23264 | [PDF](https://arxiv.org/pdf/2605.23264v1)

**作者:** Hongbo Wang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Ran He `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于Sobolev空间与对抗学习的图像超分框架ASASR，解决传统生成式SR在频谱对齐与结构保真上的不足。

**💡 创新点**

创新点包括：①将噪声颜色化为符合自然图像频谱的协方差矩阵，实现Sobolev谱校正；②设计对抗式结构引导网络生成对齐的硬负样本，利用Riesz表示理论获得最坏Sobolev梯度；③将上述两者融入DPO框架，形成AS‑DPO。

**🔧 技术方法**

采用Flow Matching与Diffusion Transformer（FLUX）作为生成器，结合Sobolev范数、Riesz表示、对抗网络、LoRA微调以及频谱分析等技术。

**📊 数据集**

训练集使用DIV2K与LSDIR；测试集涵盖DIV2K/LSDIR合成样本、RealSR、DRealSR、RealLQ250以及老电影数据集；下游评测使用COCO、ADE20K和ICDAR 2024。

**📈 对比分析**

与GAN（BSRGAN、Real‑ESRGAN、SwinIR‑GAN）及多种扩散模型（StableSR、DiffBIR、FaithDiff、SeeSR、SUPSR、DreamClear、DP2OSR、DiT4SR）对比，ASASR在PSNR/SSIM、LPIPS/DISTS、无参考指标（MANIQA/MUSIQ/CLIPIQA）上均处于领先位置，用户研究中Top‑1占比达91.1%，并在目标检测、语义分割与OCR等下游任务中取得SOTA。

**⚠️ 局限性**

局限性包括：①需额外对抗网络训练与高算力支持（8 H800 GPU）；②Sobolev参数s的选择对结果影响大；③在SSR缺失时对抗样本可能引入过度高频细节，损害失真指标；④对未知极端降质的泛化仍有限。

---

## 232. Turning Adaptation into Assets: Cross-Domain Bridging for Online Vision-Language Navigation

**arXiv ID:** 2605.23257 | [PDF](https://arxiv.org/pdf/2605.23257v1)

**作者:** Zixuan Hu `[一作]` (Peking University), Ling-Yu Duan `[通讯]` (Peking University)

**通讯引用:** 10250 | [OpenAlex ID](https://openalex.org/A5024879728)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将测试时适应过程转化为可复用资产的框架IDEA，用软提示构建历史知识库并通过凸包投影构造跨域桥梁，实现在线视觉‑语言导航的无训练适配。

**💡 创新点**

①将适应知识结构化为可插拔资产；②采用Fisher‑引导的层敏感加权提取可迁移知识；③利用凸包投影的闭式解实现训练‑free跨域桥梁；④资产库与桥梁形成互补循环，提升适配速度与稳定性。

**🔧 技术方法**

软提示优化、Fisher信息矩阵近似、层级统计匹配、凸包投影、Wasserstein距离、KKT闭式求解、熵基自监督等技术。

**📊 数据集**

REVERIE、R2R、R2R‑CE三大VLN基准（离散与连续环境）。

**📈 对比分析**

与Tent、SAR、ViDA、FSTTA、ReCAP等现有TTA方法对比，IDEA在REVERIE、R2R、R2R‑CE均实现SR、SPL等指标的显著提升（如测试未见集SR+2.5%、SPL+1.9%），并在推理速度上接近ReCAP、远优于迭代式方法。

**⚠️ 局限性**

目前仅在单一代理环境验证；资产与特征空间耦合，难以直接迁移至不同架构；多代理协同共享与架构无关资产的研究仍待开展。

---

## 233. Learning-Augmented Online Scheduling with Parsimonious Preemption

**arXiv ID:** 2605.23255 | [PDF](https://arxiv.org/pdf/2605.23255v1)

**作者:** Mugen Blue `[一作]` (University of California Santa Cruz), Alexander Lindermayr `[通讯]` (Technische Universität Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于学习预测的在线调度算法PMLF和SNAP，分别在单机、相同机组和非同质机组环境下实现O(1)竞争，并在预emption次数上给出对数级别的上界；通过实验验证其在总完成时间和预emption成本上的优势。

**💡 创新点**

首次系统分析预测下预emption与作业延迟的权衡；提供多级反馈改写为PMLF的O(1)竞争保证；提出利用Proportional Fairness（PF）率分配与epoch+checkpoint机制的SNAP，在非同质机组上实现O(1)竞争且预emption被限制在O(D)；引入鲁棒PSP框架和β-robust PF竞争分析。

**🔧 技术方法**

多级反馈队列（MLF）改写、Proportional Fairness凸规划求解、epoch模拟与checkpoint设计、虚拟算法到非预emption的离散化、鲁棒PF与β-robust PSP理论分析、实验评估。

**📊 数据集**

合成数据集：10台机器、100个作业，其中20%为特殊作业（p_j∼U[1,200]，只能使用2台特定机器），其余作业p_j∼U[1,10]；通过参数R控制预测误差（p̂_j=⌈p_j/ξ_j⌉，ξ_j∼U[1,R]）。

**📈 对比分析**

与Blind（非预emption）、Doubling（即时调度）和Hybrid SNAP（两者混合）对比。实验表明SNAP在大预测误差下保持较低的竞争比（≤14.48(1+ε)）且预emption次数显著低于Doubling；Blind在小误差下表现良好，但误差增大时性能急剧下降；Hybrid SNAP在小误差下可进一步提升。

**⚠️ 局限性**

主要限制包括：假设预测为低估，过估情况需要额外处理；作业大小假设为1+δ的幂；预emption成本模型简化，未细化上下文切换和缓存失效成本；仅在合成数据上评估，缺乏真实工作负载验证；算法实现复杂度高，且对多机迁移未深入研究。

---

## 234. CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels

**arXiv ID:** 2605.23254 | [PDF](https://arxiv.org/pdf/2605.23254v1)

**作者:** Mengke Li `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 16123 | [OpenAlex ID](https://openalex.org/A5087787304)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为CARE（Class-Adaptive Rectification with Experts）的框架，用于在长尾噪声标签学习中进行可靠学习。

**💡 创新点**

创新点在于引入了类自适应专家共识机制，根据类频率对尾类施加更严格的共识，而对头类则采取更宽松的共识，从而提高标签修正的可靠性。

**🔧 技术方法**

使用了视觉-语言模型（VLM）提供的三种互补监督源：观察到的噪声标签、VLM文本嵌入和视觉特征。

**📊 数据集**

在合成数据集（如CIFAR-100-LTN）和真实世界数据集（如WebVision-50和Food101N）上进行了广泛实验。

**📈 对比分析**

与多种最先进的方法进行了比较，CARE在各种噪声和不平衡设置下始终表现优越，性能提升可达3.0%。

**⚠️ 局限性**

限制在于该方法可能在极端不平衡或噪声条件下仍然面临挑战，且对类频率的依赖可能影响其在某些情况下的表现。

---

## 235. A Posterior MWPM Decoding Boosts the XYZ Planar Code

**arXiv ID:** 2605.23236 | [PDF](https://arxiv.org/pdf/2605.23236v1)

**作者:** Zhiwei Wang `[一作]` (Hefei University of Technology), Liqi Wang `[通讯]` (Hefei University of Technology)

**通讯引用:** 823 | [OpenAlex ID](https://openalex.org/A5000336475)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了XYZ平面码并提出后验MWPM解码方法以提升偏置噪声下的纠错性能。

**💡 创新点**

通过将所有Z测量替换为ZY测量，并利用ZY测量结果进行后验加权，显著提升了阈值并降低了逻辑错误率。

**🔧 技术方法**

采用后验概率加权、最小权重完美匹配（MWPM）算法、PyMatching模拟以及临界指数法进行阈值估计。

**📊 数据集**

使用Monte Carlo仿真，覆盖不同码距离（d=35,39,43,47及d=11,15,19,23）以及多种偏置（η=0.5,1,3,10,30,100,1000,∞）的随机噪声模型。

**📈 对比分析**

与传统MWPM相比，pMWPM在大多数偏置下阈值提升约30–40%，逻辑错误率显著降低，验证了方法的优越性。

**⚠️ 局限性**

仅考虑无噪声门和SPAM，且对XYZ码向其他拓扑码的推广与不同Y操作数量的进一步研究尚未完成。

---

## 236. Convex Low-resource Accent-Robust Language Detection in Speech Recognition

**arXiv ID:** 2605.23235 | [PDF](https://arxiv.org/pdf/2605.23235v1)

**作者:** Miria Feng `[一作]`, Mert Pilanci `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出Convex Language Detection (CLD) 框架，将两层 ReLU 网络转化为凸优化问题，并在多 GPU JAX 上通过 ADMM 实现快速训练，集成到 ASR 预处理阶段以在低资源方言环境下实现语言识别；

**💡 创新点**

创新点包括：①将非凸语音识别任务重构为可全局最优的凸优化问题；②在凸框架下得到可计算的特征空间鲁棒性证书（Lipschitz 上界和边际稳定性）；③通过多 GPU ADMM 并行实现，显著提升训练速度与样本效率；

**🔧 技术方法**

使用的技术包括：凸优化理论、ADMM 并行求解、JAX 计算框架、两层 ReLU 网络的凸重构、特征空间鲁棒性分析及可证书的边际稳定性证明；

**📊 数据集**

使用的数据集包括：Common Voice v23、Singlish Singapore English Corpus、Lahaja Hindi、以及自建的 5 语言 24 方言混合数据集，总计约 16,000 训练样本；

**📈 对比分析**

与 Whisper‑Small、Whisper‑Large‑V3、MMS‑1B 等预训练 ASR 模型的默认语言检测、传统 NN、SVM、KNN 等基线进行对比；在 100–10,000 样本低资源设置下，CLD 达到 97–98% 的语言识别准确率，WER 从 139.37 降至 31.74，训练时间仅 64 s（仅 14k TFLOPs），显著优于基线；

**⚠️ 局限性**

局限性包括：对 Encoder 的 Lipschitz 上界估计可能过于保守，导致特征空间鲁棒性证书不够紧；当前仅改进语言识别，未直接提升语音转写质量；在极端噪声或非标准输入场景下鲁棒性尚未充分验证；

---

## 237. PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

**arXiv ID:** 2605.23219 | [PDF](https://arxiv.org/pdf/2605.23219v1)

**作者:** Minju Kim `[一作]` (Inha University), Youngbum Hur `[通讯]` (Inha University)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5029147038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了PaP-NF框架，用冻结的大语言模型（LLM）作为全局上下文编码器，结合Prefix-as-Prompt重编程与条件归一化流（Normalizing Flow）实现长周期时间序列的概率预测；

**💡 创新点**

创新点在于：1）将LLM仅作为全局语义编码器，避免了数值离散化带来的精度损失；2）通过Prefix-as-Prompt将数值特征与LLM输入空间对齐，实现两模态特征在概率生成阶段的融合；3）使用条件归一化流提供多模态、不连续的预测分布；

**🔧 技术方法**

使用的技术包括线性嵌入（轻量级数值编码）、Prefix-as-Prompt重编程、冻结Meta-Llama-3.1 LLM提取全局上下文、条件归一化流生成预测分布，以及CRPS评估指标；

**📊 数据集**

在ETT系列（ETTh1、ETTh2、ETTm1、ETTm2）和Traffic数据集上进行实验；

**📈 对比分析**

与现有多种基线（Autoformer、FEDformer、TimesNet等）进行点预测比较，PaP-NF在大多数数据集上实现了更低的MSE/MAE；在概率预测评估（CRPS）中，在ETTh1、ETTh2、ETTm1、ETTm2、Traffic中均表现优异或相近，说明其在分布估计上具备竞争力；

**⚠️ 局限性**

局限性包括：1）仍需冻结LLM参数，内存占用较大；2）概率预测的计算量相对确定性模型更高，尽管比扩散模型快；3）模型对Prefix长度和预训练知识敏感，需要细致调参；4）在极端非平稳场景下的泛化能力尚待进一步验证。

---

## 238. FastKernels: Benchmarking GPU Kernel Generation in Production

**arXiv ID:** 2605.23215 | [PDF](https://arxiv.org/pdf/2605.23215v1)

**作者:** Gabriele Oliaro `[一作]` (Snowflake AI Research), Samyam Rajbhandari `[通讯]` (Snowflake AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于框架的 GPU kernel benchmark（Fastkernels），将 kernel 生成评测直接嵌入到生产级推理流水线中，涵盖 46 个代表性模型架构，任务按从 primitive 到 full‑model 的四层递归展开，支持单机与多机多卡的真实通信、捕获真实张量、与现有生产系统（vLLM、SGLang 等）保持接口一致。

**💡 创新点**

创新点包括：① benchmark‑as‑framework 设计，评测直接在生产环境中运行；② 任务层级的组合式层次结构，允许低层优化迁移到高层；③ 采用生产级引用实现与多 GPU 通信，真实捕获张量；④ 通过 MacroEval 统一衡量准确性、吞吐/延迟加速与覆盖率，避免单一指标误导；⑤ 通过动态规划式优化循环，减少重复工作。

**🔧 技术方法**

技术手段包括：top‑down 模型驱动任务生成（从 HuggingFace Transformers 读取配置并递归拆解），接口兼容设计（匹配 vLLM、SGLang 等模块签名），多 GPU 通信实现（tensor‑parallel all‑reduce / reduce‑scatter，expert‑parallel all‑to‑all 等），Nsight Compute / Systems 集成的 profiling，MLflow 用于实验追踪，MacroEval 校准函数与阈值设计，benchmark 分为三层：kernel、end‑to‑end 与标准化评测。

**📊 数据集**

使用了 46 个覆盖 8 类目（Dense & MoE LLM、Linear Attention、Vision/Audio/Video、Multimodal、Edge、3D/Robotics、Recommendation、World Models）的真实模型架构，全部来自 HuggingFace Transformers，覆盖率达 96.2%（409/425）；同时利用生产环境中捕获的真实张量与工作负载。

**📈 对比分析**

评估方式：对 L1 与 L2 级别的 kernel 生成 agent（Dr. Kernel、KernelAgent、OpenAI Codex）进行测试，分别与生产级基线（cuBLAS、FlashAttention‑3、FlashInfer、vLLM CUDA ops、deep_gemm、FLA 等）对比，采用 MacroEval 计算宏观几何加速、校准正确性与覆盖率。结果显示：最强 agent 在聚合指标上的速度提升仅为 0.94×，其他 agent 分别为 0.78× 与 0.53×；整体平均吞吐加速为 1.24×，但在主流 LLM 生产系统上基本与基线持平，增速主要集中在缺乏专用生产实现的算子上。

**⚠️ 局限性**

局限性：① 评测仅覆盖 L1/L2 级别，未展示 L3/L4 的结果；② 仅在 H100 SXM5 上测得，其他 GPU 可能表现不同；③ MacroEval 的家族加权、λ=0.5 混合方式及阈值均为预设，可能不适用于所有业务场景；④ 96.2% 覆盖率为经验估计，仍存在未覆盖的架构；⑤ benchmark 需要用户自行添加新模型，维护成本仍高。

---

## 239. MixFake: Benchmarking and Enhancing Audio Deepfake Detection in Diverse Real-world Mixed Audio

**arXiv ID:** 2605.23201 | [PDF](https://arxiv.org/pdf/2605.23201v1)

**作者:** Qingcao Li `[一作]` (Nanjing University of Science and Technology), Zhichao Lian `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 743 | [OpenAlex ID](https://openalex.org/A5016645765)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MixFake大规模混合音频深度伪造检测基准数据集，并基于多流Prompt调优的框架实现对混合音频中前景语音与背景音乐/环境噪声的深度伪造检测。

**💡 创新点**

创新点在于：①设计了前景-背景真实性交叉对齐的混合音频数据生成方法；②通过引入Hilbert‑Huang变换（HHT）与Teager‑Kaiser能量算子（TKEO）的信号级先验，构建了多流Prompt注入策略，突破了传统SSL模型以语义为中心的局限；③将多流Prompt与SSL骨干深度融合，显著提升混合环境下的检测性能。

**🔧 技术方法**

采用技术包括：XLSR‑AASIST SSL预训练骨干、Transformer层级Prompt注入、HHT获取瞬时频率信息、TKEO提取非线性能量波动、三流（基准、频率、纹理）Prompt协同学习及后端分类网络。

**📊 数据集**

使用数据集：MixFake（252,500条样本，包含真/伪前景与背景的全排列），ASVspoof 2019 LA（训练/验证）和In‑the‑Wild（跨数据集评测）。

**📈 对比分析**

通过与XLSR‑AASIST、XLSR‑Mamba、WPT‑XLSR‑AASIST三种SOTA模型对比，前景检测EER降至0.95%，背景检测EER提升至12.40%（相对基线降低7.72%），跨数据集评测EER仅6.24%，显著优于基线。

**⚠️ 局限性**

局限性在于：仍采用冻结的SSL骨干，可能对未见的伪造算法适应性不足；在极低SNR或背景占主导时性能衰退；多流Prompt策略虽然提升效果，但缺乏对特征来源的可解释性。

---

## 240. GENSTRAT: Toward a Science of Strategic Reasoning in Large Language Models

**arXiv ID:** 2605.23238 | [PDF](https://arxiv.org/pdf/2605.23238v1)

**作者:** Vartan Shadarevian `[一作]` (Princeton University), Anany Kotawala `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了GENSTRAT框架，利用程序化生成的两人零和不完全信息卡牌游戏（GBG）来评估大型语言模型的策略推理能力，并在此基础上构建了6维复杂度轴和局部波动性（jaggedness）指标；

**💡 创新点**

创新点在于：①使用程序化生成代替固定游戏集，避免评测饱和与训练数据污染；②将游戏表现分解为六轴能力曲线，揭示模型在不同策略维度的强弱；③引入局部波动性度量，评估模型在相似游戏间的表现稳定性；

**🔧 技术方法**

技术方法包括：程序化游戏生成器、Monte Carlo 质量门控、Farthest‑Point 采样、基于加法配对比较模型的 chips‑per‑game 评估、Bootstrap 置信区间、OLS 斜率回归（能力曲线）以及 K‑近邻学生化方差计算（jaggedness）;

**📊 数据集**

数据集为：从 12,351 个候选种子中筛选 2,000 个通过 Monte‑Carlo 检验的 GBG；从中用 Farthest‑Point 采样得到 50 个 benchmark 游戏；对 9 个 LLM 在 50 个游戏上共进行 36,937 条比赛记录；

**📈 对比分析**

比较方法：采用加法配对比较模型估计每个模型的 chips‑per‑game 分数（α），并给出 95% 置信区间；同时进行 Bradley‑Terry win‑rate 排名、基准 CFR+ 对比；结果显示最新前沿模型平均胜率最高，且不同模型在同一分数下表现出显著的能力曲线与波动性差异；

**⚠️ 局限性**

局限性包括：benchmark 规模仅 50 个游戏，导致估计误差较大；六轴之间存在一定相关性；仅覆盖两人零和不完全信息英语下注游戏，缺乏多玩家或合作场景；评估结果仅为相对比较，缺乏绝对性能基准；

---

## 241. Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking

**arXiv ID:** 2605.23265 | [PDF](https://arxiv.org/pdf/2605.23265v1)

**作者:** Diptarka Chakraborty `[一作]` (National University of Singapore), Alvin Hong Yao Yan `[通讯]` (National University of Singapore)

**通讯引用:** 100 | [OpenAlex ID](https://openalex.org/A5108339282)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个针对Spearman footrule距离的公平排名聚合方法，涵盖了top‑k聚合的精确算法和全排序的2‑近似算法；

**💡 创新点**

创新点在于利用整数线性规划的完全可整性（证明约束矩阵为全单调），从而在top‑k问题上实现最优；并设计了将top‑k列表扩展为全排序的匹配技巧，使得全排序问题的近似比率从3提升到2；

**🔧 技术方法**

核心技术包括：ILP建模与全单调（totally unimodular）分析、LP松弛求解、最小成本完美匹配求解、以及对Spearman footrule与Kendall‑tau距离的关系推导；

**📊 数据集**

使用了两组公开数据集：①基于NFL球员专家排名（25名专家，57名球员，8个组）；②MovieLens电影评分排名（7名用户，268部电影，8个类别）；

**📈 对比分析**

实验与基线（BFI 3-近似、KT 18/7近似）比较，结果显示：在top‑k聚合上平均误差≈2%–3%；在全排序上平均误差≈1%；在Kendall‑tau下均显著优于基线，误差≈2%–10%；算法实现时间多为多项式，实际运行迅速；

**⚠️ 局限性**

局限性包括：仅考虑比例公平约束，未处理重叠组或更强的公平定义；近似因子仍为2，理论与实践间存在一定差距；对更复杂距离度量（如NDCG）未作深入分析。

---

## 242. SCOPE: Simulating Cross-game Operations in Playable Environments for FPS World Models

**arXiv ID:** 2605.23345 | [PDF](https://arxiv.org/pdf/2605.23345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 243. Droneulator: A Portable UAV Simulator for Agricultural Workflows with RotorPy and Godot 4

**arXiv ID:** 2605.23386 | [PDF](https://arxiv.org/pdf/2605.23386v1)

**作者:** Jacob Swindell `[一作]` (University of Lincoln), Riccardo Polvara `[通讯]` (University of Lincoln)

**通讯引用:** 916 | [OpenAlex ID](https://openalex.org/A5009664515)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了 Droneulator，一个可移植的农业无人机仿真架构，集成 RotorPy 动力学、Godot 4 渲染、Zenoh ROS2 兼容感知以及 PX4/轻量 WebSocket 控制，并在同一套管线下验证了树木图像采集（COLMAP）、基于 ROS2 的局部规划（EGO‑Planner）和基于 Gymnasium 的强化学习三大工作流。

**💡 创新点**

创新点在于：① 将渲染与动力学分离、使用 Zenoh 作为 ROS2 兼容桥接，构建单一可部署堆栈；② 提供轻量 WebSocket 控制路径，避免完整 autopilot 依赖；③ 在同一架构下支持检测、规划与学习三种流程，实现了真正的跨工作流统一。

**🔧 技术方法**

技术要点包括 RotorPy（多旋翼动力学）、Godot 4（场景渲染与多相机感知）、Zenoh（ROS2 兼容桥接）、PX4 SITL、EGO‑Planner、COLMAP、Gymnasium+SAC、Micro XRCE‑DDS、WebSocket、PyInstaller。

**📊 数据集**

使用自建的树木密集农业场景生成 RGB、深度、语义分割图像作为 COLMAP 输入；为 RL 训练构建虚拟葡萄园场景；未使用公开数据集，全部采用仿真生成数据。

**📈 对比分析**

比较方法：① 对 18/36/54 张采样图像进行 COLMAP 重建，记录点云数、轨迹长度、重投影误差和重建时间；② 通过 EGO‑Planner 在五个目标区域进行局部规划，测量到达误差、最小障碍间隙和飞行时间；③ 训练 50k 步 SAC，观察奖励曲线和 episode 长度变化，并用 10 次评估跑通率验证学习稳定性。所有实验均在 30 Hz 下完成，传感器延迟低于预算，深度平均 17 ms，表明性能符合农业 UAV 的实时需求。

**⚠️ 局限性**

局限性：目前未建模风噪、光照变化和传感器噪声；SE(3) 控制器在高密度轨迹下会失稳；仅支持单无人机实时仿真，缺乏多机并行或快于实时的能力；规划演示仅涵盖局部避障，未实现全局路由；整体框架对高动态任务需手动调参或直接使用原始电机命令。

---

## 244. From Correctness to Preference: A Framework for Personalized Agentic Reinforcement Learning

**arXiv ID:** 2605.23382 | [PDF](https://arxiv.org/pdf/2605.23382v1)

**作者:** Ranxu zhang `[一作]` (University of Science and Technology of China), Chao Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 44470 | [OpenAlex ID](https://openalex.org/A5100407048)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的个性化Agentic RL框架，目标是在用户条件化任务中学习用户特定的策略；

**💡 创新点**

核心创新是PARPO算法，它将通用任务质量奖励与个性化偏好奖励解耦，并用用户专属锚点校准优势估计，同时引入两阶段偏好解耦奖励模型和Preference‑Aligned Skill Evolution Graph Memory；

**🔧 技术方法**

使用了LLM驱动的Agent、GRPO风格的RL优化、PARPO、两阶段偏好解耦奖励模型、图神经网络构建的异质图记忆、层次化社区检测与检索、以及LLM评估；

**📊 数据集**

实验基准包括ETAPP、ETAPP‑Hard以及工业电商平台的SJAgent数据集；

**📈 对比分析**

与ReAct、Mem0、GRPO、DAPO、GSPO、GiGPO、MemRL、SkillRL以及GPT‑4o/Claude等方法对比，PARPO在个性化、任务完成、逻辑与事实准确性等多项指标均位居首位，显著提升用户满意度；

**⚠️ 局限性**

主要局限在于人工评估规模有限，仅有15名专家评审20例样本，未来需扩大评测样本与评估者多样性。

---

## 245. Instance-Optimal Estimation with Multiple LLM Judges on a Budget

**arXiv ID:** 2605.23362 | [PDF](https://arxiv.org/pdf/2605.23362v1)

**作者:** Junghyun Lee `[一作]` (KAIST AI), Se-Young Yun `[通讯]` (KAIST AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并解决了在固定预算下如何在多模 LLM 判定器之间分配查询以最小化评分误差的问题；

**💡 创新点**

创新点包括：①推导了针对逆方差加权估计器的最优 oracle 分配；②设计了两阶段自适应算法 Est‑IVWE，利用乐观方差估计实现近似 oracle 性能；③给出了匹配的本地期望下界，首次证明 Fano 方法在该问题中失效而 Assouad 方法可获得精确下界；

**🔧 技术方法**

技术手段：逆方差加权估计（IVWE）、自适应预算分配、乐观方差偏置、Empirical Bernstein 以及 Assouad‑Lemma 的期望下界分析；

**📊 数据集**

使用合成 Beta 评分数据和真实 HelpSteer2 数据集（四个评价维度），模拟三种 LLM 判定器；

**📈 对比分析**

与均匀分配及已知方差的 Oracle 进行比较；实验显示 Est‑IVWE 与 Oracle 差距很小，明显优于均匀分配，且在 Gaussian 变体中表现更好；

**⚠️ 局限性**

局限性：1) 高概率下界尚未匹配；2) 仅考虑固定成本，未覆盖可变成本和评判者偏差；3) 实验规模有限，未在大规模生产环境验证。

---

## 246. Sparse Compositional Flow Matching by geometric assembly from motion primitives

**arXiv ID:** 2605.23341 | [PDF](https://arxiv.org/pdf/2605.23341v1)

**作者:** Yan Tang `[一作]` (Tsinghua University), Yang Li `[通讯]` (Chinese University of Hong Kong (Shenzhen))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个稀疏的运动原语字典，并在物理轨迹空间直接进行原语时序拼接生成机器人轨迹，避免了传统稠密点对点生成的样本复杂度瓶颈。

**💡 创新点**

创新点在于：①将运动原语与可学习长度掩码和二进制起始指示结合，使原语在字典中保持形状不变且可直接复用；②采用结构稀疏流匹配并加入几何约束，强制相邻原语在空间和时间上连续，形成可微的事件几何能量；③将字典学习与生成过程统一到同一中间变量上，实现端到端联合优化。

**🔧 技术方法**

主要技术包括：受限卷积字典学习（MPDL）、长度掩码与起始指示的前向传播、结构稀疏流匹配（Sparse Flow Matching）与几何损失、事件连续性约束、以及基于噪声的流场匹配（Flow Matching）实现条件和无条件采样。

**📊 数据集**

在两个公开基准上评估：Open X‑Embodiment（室内机器人操控任务）和 3DMoTraj（三维水下无人艇轨迹），两者涵盖不同任务复杂度和外部物理扰动。

**📈 对比分析**

与包括Diffuser、MoFlow、3DMoTraj、LaM‑SLidE、ARMD等多种稠密生成基线比较，本文方法在ADE/FDE上均取得最优，FDE/ADE比率从≈1.8降至1.07，提升ADE/ FDE 分别约为19.2%/21.0%，并在无条件生成中实现最低JSD，说明模型在分布层面更具真实性。

**⚠️ 局限性**

局限性主要有：①字典质量受训练数据覆盖度限制，罕见运动模式可能缺乏对应原语；②几何连续性约束仅局部实现，未覆盖全局动力学约束（如力矩限制、碰撞回避），在安全关键部署时可能需额外后处理。

---

## 247. From Head to Tail: Asymmetric Knowledge Transfer in Long-tail Recommendation with Generative Semantic IDs

**arXiv ID:** 2605.23310 | [PDF](https://arxiv.org/pdf/2605.23310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 248. Spatio-Temporal Similarity Volume Aggregation for Open-Vocabulary Action Recognition

**arXiv ID:** 2605.23288 | [PDF](https://arxiv.org/pdf/2605.23288v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 249. Enhancing Blood Cells Classification using Hybrid Quantum Neural Networks

**arXiv ID:** 2605.23324 | [PDF](https://arxiv.org/pdf/2605.23324v1)

**作者:** Guilherme Cruz `[一作]` (Instituto de Telecomunicacoes, University of Coimbra), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11411 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并评估了一种混合量子-经典神经网络（HQNN）架构，用以提升血细胞图像的分类性能，核心思路是在预训练的 ResNet‑50 特征提取后加入低维瓶颈，再将其映射到 10 量子比特的变分量子电路进行特征变换，然后送入轻量化分类头进行预测；

**💡 创新点**

创新点在于：①构建了严格对比框架，仅改动量子层以排除参数数目对比的干扰；②设计了容量匹配的经典基线模型，客观验证量子变换的独特优势；③在实际 IBM Qiskit 量子硬件上完成推理，验证了噪声环境下的鲁棒性；④对 25% 数据子集进行实验，展示了量子模型在样本不足时的相对优势；

**🔧 技术方法**

技术实现包括预训练 ResNet‑50 作为特征提取器、10 维 tanh 瓶颈、使用 PennyLane 的 10‑量子比特 4‑层变分电路（角度编码 + 环状 CNOT 纠缠）、单独学习率调度、焦点损失 + label‑smooth、混合精度训练与梯度裁剪等；

**📊 数据集**

实验数据集为公开的血细胞图像集：Blood Cell Images（4 类）和 Peripheral Blood Cell (PBC，8 类）两套数据；此外，在 IBM Qiskit 量子后端上对 Blood Cell Images 的 40 张样本进行了硬件推理；

**📈 对比分析**

对比方法：在相同的预处理、Backbone、瓶颈维度、分类头及训练超参下，比较三种架构（HQNN、容量匹配经典、纯经典基线）。在 4 类任务中，HQNN 的宏 F1 提升至 91.53%（相对基线提升约 3%）；在 8 类任务中，宏 F1 亦略升至 98.69%；在 25% 数据子集，HQNN 仍保持最高准确率与宏 F1；在 IBM 量子硬件上，准确率仅略降 2–3%，显示出良好的鲁棒性；

**⚠️ 局限性**

局限性包括：①量子硬件实验规模受限，仅验证了 40 张样本；②在 8 类高饱和任务中提升有限，提示需更复杂的任务或更深的量子网络来体现优势；③当前的量子电路结构相对简单，未探索更丰富的编码或拓扑；④量子推理的计算开销和采样误差在大规模数据集上仍是挑战。

---

## 250. Formal Verification of Probing Security via Conditional Independence

**arXiv ID:** 2605.23316 | [PDF](https://arxiv.org/pdf/2605.23316v1)

**作者:** Satoshi Kura `[一作]` (Waseda University), Katsuyuki Takashima `[通讯]` (Waseda University)

**通讯引用:** 3588 | [OpenAlex ID](https://openalex.org/A5010032290)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出基于Lilac概率分离逻辑的形式化验证框架，用以证明掩码算法在侧信道攻击下的探测安全性

**💡 创新点**

将模拟器定义的非干扰与条件独立联系起来，并在Lilac中引入半梯形规则、弱并规则等新的推理规则，以支持对掩码算法的证明

**🔧 技术方法**

利用Lilac的条件独立语义、Kripke资源半群模型、分离连接和条件化修饰子等技术，以及新的证明规则来构造Hoare三元组

**📊 数据集**

无具体数据集；实验以加密算法Refresh与AddRepNoiseER为案例，展示验证流程

**📈 对比分析**

未给出实验性能对比，仅通过形式化证明展示方法的可行性和完整性

**⚠️ 局限性**

目前仅适用于信息理论探测安全模型，未扩展到计算安全；实现仍需在交互式定理证明器中完成

---

## 251. Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning

**arXiv ID:** 2605.23315 | [PDF](https://arxiv.org/pdf/2605.23315v1)

**作者:** Muhammad Usama `[一作]` (Korea Advanced Institute of Science and Technology), Dong Eui Chang `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2537 | [OpenAlex ID](https://openalex.org/A5090203902)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了16个语言模型在800个推理问题上的表征相似性，探讨了表征收敛是否延伸到推理过程。

**💡 创新点**

提出了三个主要发现：困难反转（模型在共同失败的问题上更相似）、生成差距（输入编码阶段相似而输出生成阶段分歧）和表面正确性（共享信息未对预测产生因果影响）。

**🔧 技术方法**

使用了中心核对齐（CKA）等技术来评估模型间的表征相似性，并进行了因果消融实验以测试共享信息的因果作用。

**📊 数据集**

使用了来自GSM8K、ARC-Challenge、TruthfulQA和HellaSwag的数据集，共800个推理问题。

**📈 对比分析**

与现有方法相比，研究发现模型在共同失败的问题上表现出更高的CKA值（0.897），而在成功解决的问题上则较低（0.830），表明表征收敛并不等同于推理收敛。

**⚠️ 局限性**

研究的局限性包括模型参数范围（1.5B到72B），可能在更大规模（如400B+）时表现不同；因果消融假设线性正确性子空间，非线性交互可能导致不同结论。

---

## 252. LangFlash: Feed-forward 3D Language Gaussian Splatting from Sparse Unposed Images

**arXiv ID:** 2605.23287 | [PDF](https://arxiv.org/pdf/2605.23287v1)

**作者:** Yilong Liu `[一作]` (Harvard University), Hanspeter Pfister `[通讯]` (Harvard University)

**通讯引用:** 34302 | [OpenAlex ID](https://openalex.org/A5043151044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LangFlash框架，能在一次前向推理中从稀疏未标姿态多视图图像恢复3D几何、外观和语义，支持开放词汇3D分割和新视角合成。

**💡 创新点**

创新点包括：①将语言对齐的语义特征嵌入3D高斯原语；②采用全局语义词典加局部权重的稀疏编码；③通过语义分组与语言特征聚合实现端到端训练，消除逐场景优化；④在RE10k构建连续语义标签，提升语义一致性。

**🔧 技术方法**

技术手段：3D Gaussian Splatting、预训练视觉‑语言模型(CLIP)、ViT‑基语义分组模块、Transformer‑语言特征聚合、Hungarian匹配、稀疏词典编码等。

**📊 数据集**

使用数据集：RealEstate10k（RE10k）作为大规模训练集，ScanNet和3D‑OVS用于评估。

**📈 对比分析**

与NeRF‑DFF、Feature‑3DGS、pixelSplat、LSM、LSeg等方法比较，在ScanNet和3D‑OVS上实现了更高的mIoU、Acc、PSNR、SSIM；耗时仅0.18 s，无SfM预处理，zero‑shot模型亦表现优异。

**⚠️ 局限性**

局限性：对极小或薄物体、反光表面、细粒度模糊类别仍易失真；依赖CLIP等预训练模型，缺乏标签时可能性能下降；稀疏词典长度需平衡，过大导致速度下降；尚未验证极端动态或光照变化场景。

---

## 253. MDS and NMDS Codes from the Extended Twisted Generalized Reed-Solomon Codes

**arXiv ID:** 2605.23329 | [PDF](https://arxiv.org/pdf/2605.23329v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 254. ClimateChat-300K: A Multi-Modal Facebook Dataset for Understanding Diverse Perspectives in Climate Communication

**arXiv ID:** 2605.23326 | [PDF](https://arxiv.org/pdf/2605.23326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 255. AraHopeCorpus: Annotation Guidelines and Dataset for Hope Speech in Arabic Social Media Crisis Discourse

**arXiv ID:** 2605.23325 | [PDF](https://arxiv.org/pdf/2605.23325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 256. VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation

**arXiv ID:** 2605.23381 | [PDF](https://arxiv.org/pdf/2605.23381v1)

**作者:** Junwen Tan `[一作]` (South China University of Technology), Shuangping Huang `[通讯]` (South China University of Technology)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5019199411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种训练无关的加速方法VDE，通过对Rectified Flow模型的速度进行分解和估计，替代传统缓存重用方案来实现加速；

**💡 创新点**

创新点在于将速度分解为平行与正交分量，利用其时间上的线性可预测性和方向稳定性，构建输入自适应的估计函数，从而消除缓存-输入不匹配问题；

**🔧 技术方法**

使用了速度分解、线性外推、正交方向重用、周期性anchor步骤等技术，并在Flux、Qwen‑Image、Wan2.1等Rectified Flow模型上实现；

**📊 数据集**

实验基于MS‑COCO 2017验证集（图像）以及VBench视频基准，使用512×512图像以及多种分辨率；

**📈 对比分析**

与TeaCache、EasyCache、PAB等缓存重用基线以及降低步数的做法进行对比，VDE在保持或提升视觉质量（SSIM、PSNR、LPIPS、ImageReward等指标）的同时，实现2.04–3.22×的加速，LPIPS下降52.2%，SSIM提升；

**⚠️ 局限性**

局限性包括anchor间隔增大时可能略微降低视觉保真度，对极端分辨率或更复杂场景的稳健性尚未完全验证，且多步估计误差累积仍需进一步研究。

---

## 257. Decoupling Spatio-Temporal Adapter for Fine-Grained Badminton Action Localization

**arXiv ID:** 2605.23355 | [PDF](https://arxiv.org/pdf/2605.23355v1)

**作者:** Tianyu Wang `[一作]` (Beihang University), Shishuo Li `[通讯]` (Beihang University)

**通讯引用:** 184 | [OpenAlex ID](https://openalex.org/A5052656664)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究专业羽毛球比赛的细粒度时序动作定位，构建Fine‑Badminton数据集并提出DSTA适配器。

**💡 创新点**

创新点在于将时空特征解耦为时间、垂直、水平三条分支，显著提升细粒度动作区分能力。

**🔧 技术方法**

使用VideoMAE‑B骨干网络、DSTA适配器、ActionFormer检测头，并采用焦点损失进行训练。

**📊 数据集**

实验数据集包括31场专业比赛、29种击球类型、27597动作实例的Fine‑Badminton，以及处理后的ShuttleSet数据集。

**📈 对比分析**

与TIA、3D‑SENet、AIM等适配器基线及全微调方法比较，在Fine‑Badminton上平均mAP提升至66.23%，在ShuttleSet上提升至74.67%，明显优于其他方法。

**⚠️ 局限性**

局限性包括对极少数罕见动作的误检、对通道分配比例敏感，以及未结合姿态信息等因素。

---

## 258. Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion

**arXiv ID:** 2605.23346 | [PDF](https://arxiv.org/pdf/2605.23346v1)

**作者:** Jaihoon Kim `[一作]` (KAIST), Minhyuk Sung `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出Contrastive Distribution Matching（CDM）框架，通过学习折叠函数（twist）将离散扩散模型的Twisted Sequential Monte Carlo（SMC）推理加速，实现一次前向传播即可完成折叠评估。

**💡 创新点**

创新点在于将折叠函数学习转化为前向KL正则的对比学习目标，利用扩散前向核实现正负样本的高效采样，并将折叠头与主网络共享特征层，显著降低推理成本并解决了传统离散S的Monte‑Carlo估计瓶颈。

**🔧 技术方法**

使用的技术包括离散Masked Diffusion Model、Twisted SMC、对比学习、前向KL、EMA软目标更新、轻量折叠头、缓冲正样本策略等。

**📊 数据集**

在毒性文本生成、调控DNA序列设计、蛋白质可设计性和扩散大语言模型对齐等四类任务上验证，采用的数据集有OpenWebText、DNA增强数据集、Enformer预测模型、DPLM‑2蛋白生成、RewardBench奖励数据。

**📈 对比分析**

与BoN、SMC、SMC+Grad、回归折叠学习等基线在墙钟时间匹配下比较，CDM在所有任务中均取得更高奖励、扩展性更好；尤其在奖励计算昂贵时（蛋白折叠、LLM对齐）仍保持低推理时间并显著优于传统SMP。

**⚠️ 局限性**

局限性在于仍需通过采样产生正样本，训练时需要管理正样本缓冲；对非离散或更复杂奖励分布的通用性尚待进一步研究。

---

## 259. Tractable Maximization of Budgeted Phylogenetic Diversity on Networks Utilizing Node Scanwidth

**arXiv ID:** 2605.23319 | [PDF](https://arxiv.org/pdf/2605.23319v1)

**作者:** Niels Holtgrefe `[一作]` (Delft University of Technology), Jannik Schestag `[通讯]` (Delft University of Technology)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5037447083)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了三种预算限制下的网络式系统发育多样性（PD）最大化问题，并给出了针对树形度量“节点扫描宽度（nsw）”的多项式时间算法

**💡 创新点**

创新点在于：①将预算约束与网络形态参数nsw结合，首次实现了在网络上对变成本PD的可行优化；②针对三种不同的PD变体提供了O*(2^·B^2)和O*(3^B)的算法；③首次给出节点扫描宽度的精确计算方法和对应分解，填补了该参数在进化计算中的空白

**🔧 技术方法**

技术方法主要包括：结构化数据规约规则、基于nsw分解的动态规划、整数线性规划（ILP）求解以及对预算与网络形态参数的联合复杂度分析

**📊 数据集**

使用模拟生成的高度网络化（reticulated）网络数据集，包含数百到一千多种生物体，并设置异质化的保护成本

**📈 对比分析**

与先前仅适用于单位成本的网络PD算法进行对比，实验结果显示本方法在几秒内即可完成PD分数与最优nsw的计算，并在预算约束下显著提升了优化效果，尤其在高nsw、成本多样化的实例中表现优异

**⚠️ 局限性**

局限性包括：算法仍受预算和nsw大小的指数依赖，极大规模网络（nsw>15）时可能仍不可行；此外对真实生物网络的验证有限，当前实验主要基于模拟数据

---

## 260. DART: Semantic Recoverability for Structured Tool Agents

**arXiv ID:** 2605.23311 | [PDF](https://arxiv.org/pdf/2605.23311v1)

**作者:** Ke Yang `[一作]` (MOS Intelligent Connectivity Technology Co. Ltd), Xiaoshui Huang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2246 | [OpenAlex ID](https://openalex.org/A5043863561)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了DART，一个针对结构化工具代理的语义可恢复运行时，能够在已提交下游工作时安全地进行局部回滚。

**💡 创新点**

创新点在于引入四步恢复流程：失败实例定位、可恢复边界认证、实例对齐检查点以及可接受回滚选择，明确了控制合法性与语义有效性之间的区别，并为语义可恢复性提供判据。

**🔧 技术方法**

使用了显式有限状态机(FSM)作为执行模型，结合检查点对齐、依赖与效果约束的可接受性验证，以及对已审计边界、接口和效果策略的判定。

**📊 数据集**

在三大LLM驱动领域（导航、调度表单、诊断）进行评估，并在LangGraph基础上进行外部验证，涉及约五个实验域。

**📈 对比分析**

与全任务重跑、粗粒度重试以及仅回滚入口检查点等基线比较，DART在所有判定为“承诺敏感”的案例中实现了100%恢复成功，回放成本仅为全任务重跑的一小部分，且在常规案例中保持与更强本地恢复基线相当的性能。

**⚠️ 局限性**

局限在于仅针对可观测失败且已预审计边界的显式控制结构，无法自动生成边界；此外对未观测或隐式错误的诊断与恢复尚未覆盖。

---

## 261. DeFi Yield Aggregators: Analysing Investment Strategies and Structural Dependencies

**arXiv ID:** 2605.23298 | [PDF](https://arxiv.org/pdf/2605.23298v1)

**作者:** Stefan Kitzler `[一作]` (Complexity Science Hub and AIT Austrian Institute of Technology), Bernhard Haslhofer `[通讯]` (Complexity Science Hub)

**通讯引用:** 2275 | [OpenAlex ID](https://openalex.org/A5078932059)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对某去中心化金库的手动存取款交易进行全面记录与分析，包括交易哈希、功能签名、累计分布以及与汇率信号的关联，并将存取款与投资收益进行对比。

**💡 创新点**

创新点在于将存取款细节与汇率动态结合，构建完整的交易与收益映射，为金库性能评估提供了新的量化视角。

**🔧 技术方法**

采用区块链交易挖掘、哈希解析、统计分布分析与时间序列关联等技术。

**📊 数据集**

使用了该金库在2026年5月之前的所有手动存取款交易哈希及其对应的时间戳与金额数据。

**📈 对比分析**

通过将存取款与对应的市场投资收益匹配，利用累计收益曲线与汇率波动曲线进行对比，结果显示存取款策略与市场波动高度相关，表现优于传统定投。

**⚠️ 局限性**

局限性包括样本仅来自单一金库，缺乏跨链或多资产验证，且对高频交易缺乏覆盖。

---

## 262. Ontological Knowledge Blocks: Executable Compliance and Profile-Based Validation for Trustworthy AI Systems

**arXiv ID:** 2605.23297 | [PDF](https://arxiv.org/pdf/2605.23297v1)

**作者:** Aasish Kumar Sharma `[一作]` (Georg-August-Universität Göttingen), Julian M. Kunkel `[通讯]` (Georg-August-Universität Göttingen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种可编程的治理基础设施 Ontological Knowledge Blocks (OKBs)，通过将监管义务编译为可执行的 SHACL 约束并在 AI 服务产生的 RDF 证据图上实时验证，实现对 AI 关键数字基础设施的透明度、责任性、公平性和可追溯性的自动化合规检查。

**💡 创新点**

创新点在于：① 通过四层架构将规范义务（IR 记录）转换为可执行的知识块；② 设计了可组合、可配置的治理配置文件（profiles），实现无代码改动的治理切换；③ 在同一管道中集成 RDF/OWL 证据、SHACL 验证和 PROV-O 追溯，实现端到端可验证的合规路径；④ 提供了 profile 等价性与细化关系的算法验证，证明 Combined 方案为最全面且无等价伴随方案。

**🔧 技术方法**

核心技术包括：RDF/OWL 本体建模、SHACL 及 SHACL‑SPARQL 约束、PROV‑O 追溯链、Python 语言实现、RDF 语义推理、以及基于 Git+DVC 的政策版本管理与审计。

**📊 数据集**

使用了 HPC 资源调度场景下生成的人工构造证据图（包含决策、分配、日志、解释、模型元数据等），以及对应的四个司法/能力配置文件（EU、US、China、EU+Fairness、Accountability、Fairness、Combined）。

**📈 对比分析**

通过在 24 次验证运行（12 次 OKB 原型、12 次编译原型）评估 profile 敏感性、细化关系、以及验证延迟。结果显示：Profile 之间的违规计数呈严格递增；Combined 方案在所有案例中检出最多违规；验证延迟在 12.6 ms–100.3 ms（中位数约 60 ms）以内，满足秒级决策时间要求。

**⚠️ 局限性**

局限性包括：① 证据案例为人工构造，可能不覆盖所有真实失败模式；② 仅在 HPC 调度场景验证，跨领域（如信用评分、医疗 AI）需要进一步验证；③ 公平性约束以两组差距为例，未涵盖更复杂多组或时间序列公平度量；④ 依赖人工审核 IR 记录，无法自动化法律解释。

---

## 263. Bayesian Extreme Value Theory with Hawkes-AR-Gumbel Dependence for Extreme CVaR Estimation in Operational Risk

**arXiv ID:** 2605.23353 | [PDF](https://arxiv.org/pdf/2605.23353v1)

**作者:** Juan Ballesteros Gómez `[一作]` (Pontificia Comillas University), Pedro Pablo Pérez-Velasco `[通讯]` (Pontificia Comillas University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种结合 Gumbel 上尾依赖、AR(1) 隐变量持续性和 Hawkes 自激发机制的贝叶斯 Hawkes-AR-Gumbel 模型，用于估计操作风险的 CVaR。

**💡 创新点**

创新点在于将三种先前独立研究的依赖机制（上尾依赖、时间持续性、事件聚集）整合到单一模型中，克服传统 LDA 对频率-严重性依赖和极值不确定性的局限。

**🔧 技术方法**

使用了贝叶斯极值理论、Gumbel Copula、AR(1) 隐变量过程、Hawkes 自激进程，以及 PyMC 的 Hamiltonian Monte Carlo 推断。

**📊 数据集**

通过 15 年的模拟操作风险数据（按 Hawkes-AR-Gumbel 生成）进行验证，未使用真实银行数据库。

**📈 对比分析**

与独立 LDA 和共享高斯因子模型进行对比，采用后验预测 CVaR 计算，结果显示独立模型低估 99.995% CVaR 约 40%，共享因子模型同样无改进，而 Hawkes-AR-Gumbel 能准确恢复参数并给出更高 CVaR；整个推断与模拟在 15 分钟内完成。

**⚠️ 局限性**

局限性包括仅在模拟数据上验证、未考虑多单元交叉依赖、非年度分辨率、正式模型选择指标（WAIC/LOO）缺失，以及 Hawkes 参数识别受限于年度事件稀疏。

---

## 264. Purification Strategy Optimization for Entanglement Routing in Quantum Networks

**arXiv ID:** 2605.23331 | [PDF](https://arxiv.org/pdf/2605.23331v1)

**作者:** Javier Vecino Peñas `[一作]` (Universidade de Vigo), Manuel Fernández-Veiga `[通讯]` (Universidade de Vigo)

**通讯引用:** 1021 | [OpenAlex ID](https://openalex.org/A5015041155)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在量子网络中使用纠缠净化策略来提升端到端纠缠保真度，并将其建模为一个优化问题

**💡 创新点**

首次将动态规划（DP）方法应用于纠缠净化策略的全局优化，显著降低了原本指数级的搜索复杂度

**🔧 技术方法**

采用动态规划（memoization）求解最优净化方案，并在小规模下使用暴力枚举做对照；使用BBPSSW（IBM）净化协议、熵退相模型、交换概率与纠缠生成概率的解析式

**📊 数据集**

使用仿真数据：对每条路径随机生成初始保真度在[0.75,0.99]区间内，时间槽为1 ms，概率均设为1；路径长度从3到200节点不等

**📈 对比分析**

在小规模网络（N≤7）使用暴力枚举验证最优性，在大规模网络（N>7）使用DP获得近似最优解；实验表明DP在时间和内存上优于暴力法，且能在不同退相速率下找到满足阈值的最佳净化策略

**⚠️ 局限性**

局限性包括：仅考虑单一净化协议、忽略路径搜索与网络动态变化、退相模型过于简化、假设所有操作成功率为1、未考虑真实硬件资源与错误模型，故结果主要适用于理想化仿真环境

---

## 265. GFSR: Geometric Fidelity and Spatial Refinement for Reliable Lane Detection

**arXiv ID:** 2605.23327 | [PDF](https://arxiv.org/pdf/2605.23327v1)

**作者:** Tiancheng Wang `[一作]` (Anhui University), Guanghui Yue `[通讯]` (Shenzhen University)

**通讯引用:** 4082 | [OpenAlex ID](https://openalex.org/A5062714742)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于几何保真与空间精细化的车道检测框架GFSR，解决传统方法在复杂场景下置信度与几何质量不匹配以及曲线拟合不足的问题。

**💡 创新点**

创新点包括：①引入LaneIoU作为软监督的几何保真度预测，并与分类置信度融合生成协同可靠指数（CRI）以校正置信度；②设计自适应门控位置细化模块（AGLR），在每个精细化阶段实现点级偏移预测与门控调节，提升高曲率段的几何精度。

**🔧 技术方法**

核心技术包括：卷积主干+FPN、RoIGather、交叉层精细化检测头、基于Sigmoid的几何保真度预测、CRi计算、1D卷积残差结构的AGLR以及BCE/L1损失组合。

**📊 数据集**

实验使用公开车道检测基准CULane和CurveLanes两个数据集，分别包含多样的昼夜、遮挡、曲线和多车道场景。

**📈 对比分析**

在CULane上相较于CLRerNet等主流方法，GFSR在F1@0.5/0.75分别提升至81.46%/65.01%，并保持实时推理速度；在CurveLanes上使用RepViT‑M1.5得到最高87.35% F1，证明在高曲率和多车道场景中更优。

**⚠️ 局限性**

局限性包括：仍需在极端天气和极端交通密度下进一步验证鲁棒性；门控细化在计算上略有额外开销，且对超大尺度或三维车道几何的适应性尚未评估。

---

## 266. Arrow-Type Impossibility for Genuinely Modal Judgments

**arXiv ID:** 2605.23321 | [PDF](https://arxiv.org/pdf/2605.23321v1)

**作者:** Yutaka Nagai `[一作]` (Nagoya University), Hirotaka Ono `[通讯]` (Nagoya University)

**通讯引用:** 3967 | [OpenAlex ID](https://openalex.org/A5102003030)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了在仅聚合模态判断而非命题真值的情形下，判断聚合仍会出现Arrow式不可能性，并构造了相应的不可能框架。

**💡 创新点**

创新之处在于证明即使仅使用极其稀疏的模态公式，语义结构本身就能产生足够的逻辑相互依赖导致独裁；同时提出了语义归约与局部-全局框架相结合的判定方法。

**🔧 技术方法**

主要技术包括：Kripke框架的几何分析、语义归约定理、局部覆盖与路径连通性构造、组合覆盖问题化简为Horn SAT，以及对顺序多数投票一致性检查的高效实现。

**📊 数据集**

该工作为理论研究，无需使用实际数据集。

**📈 对比分析**

对顺序多数投票的一致性检验给出了时间复杂度为O(r·min{|A|^3,k·|A|})的实现，并在A=[0,k]时进一步优化到O(r·k)，相比原始暴力一致性检验大幅提升性能。

**⚠️ 局限性**

局限在于仍未突破Arrow式不可能性，只是提供了高效的非独裁实现；构造局部覆盖所需的循环框架和参数条件可能不易推广到更复杂的模态结构。

---

## 267. Human-in-the-Loop Multi-Agent Ventilator Decision Support with Contextual Bandit Preference Learning

**arXiv ID:** 2605.23320 | [PDF](https://arxiv.org/pdf/2605.23320v1)

**作者:** Sijia Li `[一作]` (Shanghai University of Engineering Science), Xihe Qiu `[通讯]` (Shanghai University of Engineering Science)

**通讯引用:** 569 | [OpenAlex ID](https://openalex.org/A5007950680)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种人机协作的多代理呼吸机决策支持系统VDSS，并通过上下文多臂赌博机实现临床医生偏好在线自适应。

**💡 创新点**

创新点在于将呼吸机调节拆解为模块化代理并通过合同驱动接口实现可追溯的决策链；引入波形分析、层级记忆以及循环级反馈的上下文赌博机，实现可解释、安全、可调优的交互式支持。

**🔧 技术方法**

使用多代理框架、LLM/VLM推理模块、波形分析器、上下文多臂赌博机、层级短期/长期记忆、结构化反馈和安全检查；后端以Qwen3-VL、Gemini、GPT-5.2等大型语言模型为基底。

**📊 数据集**

使用多中心ICU回放数据集，包含1309条结构化记录和7447条呼吸机设定，涵盖13种模式，最常见模式占比83.4%。

**📈 对比分析**

与单模型端到端方案对比，在回放指标上MSE、MAE下降、R^2上升，临床可接受度、安全性、清晰度和整体评分均显著提升；例如GPT-5.2从0.343降至0.102，整体评分从2.63提升至4.11；交互回合数（regret）逐渐下降。

**⚠️ 局限性**

局限包括：仅在回放实验验证，未在真实临床环境中试用；对波形分析的依赖导致推理延迟；多臂赌博机只覆盖12个偏好类别，可能不足以覆盖所有临床风格；每个模式数据量有限，限制了RL方法的推广。

---

## 268. XWind: A Cross-site Router for Large Language Model Inference Serving at Renewable Energy Farms

**arXiv ID:** 2605.23348 | [PDF](https://arxiv.org/pdf/2605.23348v1)

**作者:** Tella Rajashekhar Reddy `[一作]` (Microsoft), Debopam Bhattacherjee `[通讯]` (Microsoft)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种AI基础设施部署模型，旨在将模块化AI计算与可再生能源（主要是风能）结合，以满足日益增长的AI电力需求。

**💡 创新点**

创新点在于通过在风电场部署AI计算，减少电网负担，同时利用可再生能源的本地需求来提高能源利用效率。

**🔧 技术方法**

使用了一种轻量级、反应式的AI推理路由器，该路由器基于实时信号（推理延迟、KV缓存利用率和队列深度）动态配置站点和分配请求。

**📊 数据集**

评估在一个64-GPU A100的测试平台上，模拟了三个风电场的生产轨迹。

**📈 对比分析**

与最强竞争者（Max-FLOPS）相比，P99端到端延迟减少了22%到52%，与单一控制基线相比减少了高达98%。

**⚠️ 局限性**

局限性在于需要复杂的物流支持，包括ROI/TCO分析和与可再生能源合作伙伴的协调，且可能在极端情况下出现请求溢出传统数据中心的情况。

---

## 269. AlignedServe: Orchestrating Prefix-aware Batching to Build a High-throughput and Computing-efficient LLM Serving System

**arXiv ID:** 2605.23389 | [PDF](https://arxiv.org/pdf/2605.23389v1)

**作者:** Fengyao Bai `[一作]` (Sun Yat-Sen University), Yutong Lu `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 6301 | [OpenAlex ID](https://openalex.org/A5101633465)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的大语言模型（LLM）推理框架AlignedServe，采用前缀感知批处理策略，以消除推理过程中的迭代级气泡，从而提高推理吞吐量和计算效率。

**💡 创新点**

创新点在于首次关注迭代级气泡问题，通过将具有相似前缀长度的请求分组到同一批次中，确保在每次迭代中生成的所有令牌具有相同的计算成本，从而消除迭代级气泡。

**🔧 技术方法**

使用了一种新的推理框架AlignedServe，结合前缀感知批处理和批级调度策略，利用大容量CPU内存来存储KVCache，并通过NVLink实现GPU之间的高效数据传输。

**📊 数据集**

使用了合成数据集和实际应用工作负载进行评估，包括ShareGPT、LongBench和AzurePublicDataset等，展示了在不同工作负载下的性能表现。

**📈 对比分析**

与现有的最先进系统（如vLLM、DistServe和FastGen）进行比较，AlignedServe在解码吞吐量上提高了最多1.98倍，延迟减少了最多7.4倍，显示出显著的性能优势。

**⚠️ 局限性**

限制在于需要维护大量的在飞请求以支持前缀感知批处理，这可能导致高内存开销。此外，批次切换期间可能会出现请求来自不同批次的情况，影响性能。

---

## 270. AffectCodec: Emotion-Preserving Neural Speech Codec with Block-Diagonal Residual FSQ

**arXiv ID:** 2605.23373 | [PDF](https://arxiv.org/pdf/2605.23373v1)

**作者:** Zhaoyang Meng `[一作]` (Beijing University of Posts and Telecommunications), Ya Li `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 22619 | [OpenAlex ID](https://openalex.org/A5100404103)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种名为AffectCodec的情感保留式神经语音编解码器，能够在低比特率下更好地保留情感信息。

**💡 创新点**

核心创新在于Block‑Diagonal Residual Finite Scalar Quantization（BD‑RFSQ），通过块对角投影结构实现情感与声学子空间的显式隔离，并配合多尺度情感调制和多速率训练以保障情感信息不被重建梯度覆盖。

**🔧 技术方法**

采用BD‑RFSQ、可学习的块对角投影、可归一化FSQ、情感2vec预训练教师、双路径编码器、FiLM调制、Stage Dropout以及多速率重建损失。

**📊 数据集**

训练使用LibriSpeech（960h）与IEMOCAP（约10h），评估基准为IEMOCAP、CREMA‑D、ESD。

**📈 对比分析**

与EnCodec、DAC、SpeechTokenizer、X‑Codec等主流编解码器对比，AffectCodec在1.5~6.0 kbps的所有数据集上实现了最低的情感退化率（MEDR/WEDR）并保持了竞争性的声学质量（ViSQOL、STOI）和可懂度（WER）。

**⚠️ 局限性**

局限包括对情感2vec教师的偏见、手动设定的分区与速率目标以及仅在16 kHz语音与离线情感评估指标上的验证，未来需探索自动化属性‑速率分配与更广泛的语音条件。

---

## 271. Curriculum reinforcement learning with measurable task representation learning

**arXiv ID:** 2605.23372 | [PDF](https://arxiv.org/pdf/2605.23372v1)

**作者:** Yongyan Wen `[一作]` (Harbin Institute of Technology), Peng Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 22448 | [OpenAlex ID](https://openalex.org/A5021833788)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出一种基于任务表征学习的自动课程强化学习框架ACRL，能够在非欧氏任务空间中自动生成从源任务到目标任务的连贯中间任务分布；

**💡 创新点**

创新点在于使用变分自编码器对轨迹进行编码，学习可测量的任务相似度，并通过潜在空间预测（LSP）和探索边界更新（EBU）两种机制动态生成课程；

**🔧 技术方法**

采用VAE（包含奖励与转移解码器）进行任务表征学习，结合PPO等RL算法、LSP/EBU更新策略和任务解码器将潜在表示映射回上下文空间；

**📊 数据集**

实验数据集包括具有离散动作空间的多难度迷宫任务（Easy、Medium、Hard）和连续动作空间的Mujoco连续控制任务；

**📈 对比分析**

与默认、随机、ALP‑GMM、Goal GAN、VDS、PLR、CURROT等基线进行对比，ACRL在目标任务上实现了更快的到阈值速度和更高的最终性能；

**⚠️ 局限性**

局限在于仅支持参数化上下文表示，无法直接处理符号或语言等非参数化任务定义。

---

## 272. Cultural Adaptation in Large Language Models for Political Discourse

**arXiv ID:** 2605.23332 | [PDF](https://arxiv.org/pdf/2605.23332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 273. Score-Based One-step MeanFlow Policy Optimization

**arXiv ID:** 2605.23365 | [PDF](https://arxiv.org/pdf/2605.23365v1)

**作者:** Kyungyoon Kim `[一作]` (Korea University), Byung-Jun Lee `[通讯]` (Korea University)

**通讯引用:** 60030 | [OpenAlex ID](https://openalex.org/A5100673343)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Score-Based One-step MeanFlow Policy Optimization (SOM)，在在线RL中通过从Q函数构造目标速度场，实现单步动作采样的生成式策略。

**💡 创新点**

创新点在于利用Q函数梯度估计Boltzmann目标分布的score，并通过概率流ODE直接得到MeanFlow的目标速度场，避免了对目标分布样本的需求。

**🔧 技术方法**

使用技术包括MeanFlow、概率流ODE、iDEM式score估计、基于Q的能量函数、Actor-Critic框架以及Best-of-N采样等。

**📊 数据集**

使用的数据集为MuJoCo v4的九个连续控制任务（Ant-v4, HalfCheetah-v4, Hopper-v4, Humanoid-v4, Walker2d-v4等），以及二维Bandit/两月形等可视化验证。

**📈 对比分析**

与多种基线（SAC、PPO、DACER、DIPO、DPMD、QSM、QVPO、SDAC、MFP等）对比，SOM在大多数任务上取得最高或接近最高的平均回报，同时在推理时间上显著优于多步扩散/流匹配方法。

**⚠️ 局限性**

局限性包括依赖于学习到的Q函数质量，难以直接推广到更高维或部分可观测的场景，且对分布式或不确定性评估的兼容性待进一步研究。

---

## 274. Multi-Floor Exploration for Ground Robots via an Incremental Reachable Graph and Structural Priors

**arXiv ID:** 2605.23350 | [PDF](https://arxiv.org/pdf/2605.23350v1)

**作者:** Zhiwen Zhu `[一作]`, Boyu Zhou `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种面向多楼层建筑的地面机器人自主探测框架，利用增量可达图保持潜在可行连通性，并在图上定义物理可达前沿；同时通过投影任务区结构先验在未观测楼层初始化假设图，采用层次规划实现全局指导。

**💡 创新点**

创新点包括：① 采用增量可达图并保留可疑连通边，实现对不完整观测区域的物理可达前沿检测；② 基于已探测楼层的任务区划分，在目标楼层投影并逐步维护假设图，提供跨楼层结构先验；③ 在图层次与局部层次相结合的规划框架中实现全局路径规划与局部姿态约束。

**🔧 技术方法**

技术实现主要包括：LiDAR 3D 地图构建与可通行度评估、稀疏可达图增量构建与状态管理、前沿检测与簇化、层次 TSP 与动态规划视角选择、A* 路径规划、B-spline 平滑轨迹生成、FAST-LIO2 里程计与实时运行。

**📊 数据集**

实验数据集：仿真场景 Office、Maze、Teaching Building（含两层楼梯/斜坡等结构），以及真实实验场景（两层校园楼、庭院大楼、地下车库）使用 Unitree Go2 机器人配备 LiDAR 传感器。

**📈 对比分析**

与基线方法 TARE 与 FAEL 在相同硬件、参数与速度限制下进行5次独立运行；在多楼层场景中，所提方法将总探测时间缩短 47%、路径长度缩短 40%；在单层场景中亦更快；同时映射完成度最高，探索的可通行体积与占用体积均优于基线。

**⚠️ 局限性**

局限性：宏观层规划高度依赖楼层间结构相似性，若楼层布局差异过大，先验可能失效导致规划退化为纯观测图的常规探测；此外，当前方法对极端复杂或非规则楼层的适应性有限。

---

## 275. CHASD: Language Increment-Calibrated Contrastive Decoding against Hallucination in LVLMs

**arXiv ID:** 2605.23344 | [PDF](https://arxiv.org/pdf/2605.23344v1)

**作者:** Xiaoyi Huang `[一作]` (Xiamen University), Zhiming Luo `[通讯]` (Xiamen University)

**通讯引用:** 8508 | [OpenAlex ID](https://openalex.org/A5072443458)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CHASD，一个训练无关的步骤级对比调优框架，用以抑制大视觉语言模型的物体幻觉

**💡 创新点**

创新点在于结合置信门控与跨模态注意力引导的局部扰动，按需、按步动态生成负样本，避免全局扰动与不必要的负样本前向计算

**🔧 技术方法**

采用置信门控机制、跨模态注意力提取、局部高斯噪声扰动、对比解码公式、APC 等技术实现对比校准

**📊 数据集**

使用 POPE、AMBER、MME、MMHal-Bench、CHAIR 等多种视觉语言评测基准

**📈 对比分析**

与 VCD、AvisC、SID 等训练无关对比解码方法对比，CHASD 在各基准上提升幻觉相关指标（如 F1、准确率）且推理延迟和 GPU 内存使用与 VCD 相当或略优

**⚠️ 局限性**

局限在于置信门控可能跳过高置信的幻觉生成，未引入更丰富的不确定性信号来进一步提升鲁棒性

---

## 276. Security, Privacy, and Ethical Risks in OpenClaw

**arXiv ID:** 2605.23330 | [PDF](https://arxiv.org/pdf/2605.23330v1)

**作者:** Yutong Jin `[一作]` (Queen's University), Jianbing Ni `[通讯]` (Queen's University)

**通讯引用:** 6888 | [OpenAlex ID](https://openalex.org/A5033931001)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文对OpenClaw自主AI代理平台的架构、功能及其在安全、隐私与伦理方面的风险进行了系统性分析。

**💡 创新点**

创新点在于首次构建了面向agent系统的风险分析框架，结合威胁模型、信任边界与持久状态等特征，阐述了从prompt注入到技能供应链、跨会话传播等多维度风险，并提出了未来研究方向。

**🔧 技术方法**

采用了架构分析、威胁建模、基于轨迹的安全审计以及文献综述等方法，对OpenClaw的网关、运行时、工具层、技能生态及持久会话状态进行评估。

**📊 数据集**

主要使用OpenClaw本身作为实验对象，并引用公开的安全审计基准（如ReAct、AgentDojo等）和已有的攻击实验结果作为对照。

**📈 对比分析**

通过对34个代表性测试案例的轨迹审计，OpenClaw的整体通过率仅为58.9%，提示现有防御不足；与现有LLM代理的安全性相比，OpenClaw在prompt注入与意图误解方面表现尤为薄弱。

**⚠️ 局限性**

局限性包括缺乏系统化的沙箱与权限细粒度控制，隐私审计工具不足，跨会话传播模型未充分验证，且研究多基于理论与小规模审计，缺乏大规模实证评估。

---

## 277. Towards Generalizable and Efficient Large-Scale Generative Recommenders

**arXiv ID:** 2605.23312 | [PDF](https://arxiv.org/pdf/2605.23312v1)

**作者:** Qiuling Xu `[一作]` (Netflix Research), Moumita Bhattacharya `[通讯]` (Netflix Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

研究了在生产环境中将生成式推荐器从2M扩展至1B参数的可行性，并通过多技术组合提升训练效率、推理延迟和冷启动效果；

**💡 创新点**

提出任务依赖的偏移幂律规模法作为诊断工具，结合多标记预测以适配缓存延迟，使用采样softmax与投影解码头降低成本，并将语义项塔与协同嵌入掩码集成以解决冷启动；

**🔧 技术方法**

生成式推荐模型、偏移幂律规模分析、采样softmax、投影解码头、多标记预测（MTP）、语义项塔、协同嵌入掩码、生产阴影评估；

**📊 数据集**

大规模用户行为日志（约2万亿行为令牌），Netflix推荐任务的三类任务实例；

**📈 对比分析**

在1M用户阴影评估中将1B骨干模型与2M骨干模型对比，Task A提升22.5%，Task B 11.3%，Task C 7.4%，冷启动标题提升28.1%；

**⚠️ 局限性**

对高可预测任务的规模收益有限；训练与推理仍依赖采样与投影，MTP在低延迟在线场景中效果下降；冷启动仍受协同嵌入稀缺的限制。

---

## 278. General Hazard Detection

**arXiv ID:** 2605.23304 | [PDF](https://arxiv.org/pdf/2605.23304v1)

**作者:** Stephanie Ng `[一作]` (Swinburne University of Technology), Hailing Zhou `[通讯]` (Swinburne University of Technology)

**通讯引用:** 1548 | [OpenAlex ID](https://openalex.org/A5018875718)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究提出了一种基于规则的视觉语言模型框架，用于在交通、建筑和仓库场景中检测并评估安全规则合规性。

**💡 创新点**

其创新点在于将危害概念从图像范例中解耦，通过自然语言安全规则实现泛化评估，并引入主动学习与人机交互显著降低标注成本。

**🔧 技术方法**

采用的技术包括规则驱动的提示工程、LLaVA等视觉语言模型、LoRA参数高效微调、基于不确定性和投票的主动学习及上下文反馈。

**📊 数据集**

数据集为自建的CompliVision，共3006张图像，涵盖交通、建筑、仓库三大领域，并配有对应的ISO/OSH等标准规则及三类合规标签。

**📈 对比分析**

与零样本、全监督微调以及多种VLM进行比较，实验表明主动学习+解释反馈可在标注量减少约65%的前提下，宏观F1与准确率与全监督相近（例如交通域宏F1 0.84，准确率0.89）。

**⚠️ 局限性**

限制在于仅处理静态图像，难以捕捉时序动态和多主体交互的安全合规场景。

---

## 279. Parallel Context Compaction for Long-Horizon LLM Agent Serving

**arXiv ID:** 2605.23296 | [PDF](https://arxiv.org/pdf/2605.23296v1)

**作者:** Musa Cim `[一作]` (Pennsylvania State University), Mahmut Taylan Kandemir `[通讯]` (Pennsylvania State University)

**通讯引用:** 20011 | [OpenAlex ID](https://openalex.org/A5007116603)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种并行上下文压缩方法，用于长时间的LLM代理服务，以解决传统序列压缩在处理长对话历史时的效率和信息损失问题。

**💡 创新点**

创新点在于引入了并行压缩设计，使操作员能够对摘要的体积进行细粒度的可预测控制，并提高了压缩的吞吐量。

**🔧 技术方法**

使用了基于LLM的压缩技术，结合了多种模型架构（包括密集型和MoE架构），并在多个基准测试上进行了评估。

**📊 数据集**

使用了HotpotQA多跳问答和LoCoMo长对话基准数据集进行实验。

**📈 对比分析**

与传统的顺序压缩方法相比，本文的并行压缩在相同的压缩输出体积下，显著减少了端到端的延迟时间，并提高了压缩吞吐量。

**⚠️ 局限性**

限制在于当前设计仍然是同步的，可能会导致在压缩过程中阻塞代理的推理，未来可以考虑异步压缩以提高效率。

---

## 280. NASiC: 3D NAND-based CAM-Selected Multibit CIM Architecture for Efficient On-Device Mixture-of-Experts LLM Inference

**arXiv ID:** 2605.23294 | [PDF](https://arxiv.org/pdf/2605.23294v1)

**作者:** Weikai Xu `[一作]` (Peking University), Ru Huang `[通讯]` (Peking University)

**通讯引用:** 14267 | [OpenAlex ID](https://openalex.org/A5062886480)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于3D NAND的CAM‑Selected多比特CIM架构，实现稀疏MoE模型的高效边缘推理，包含专家激活与计算的原子级融合、块级并行运算以及多比特输入/权重量化。

**💡 创新点**

① 在同一3D NAND字符串中集成CAM与CIM，实现动态专家选择与计算一体化，消除冗余运算；② 采用块级改进的温度计编码和原位多比特输入扩展，提升CIM数组的并行度；③ 利用多层Flash单元构造多比特CIM单元，充分发挥3D NAND高存储密度。

**🔧 技术方法**

3D NAND存储技术、内容可寻址存储（CAM）、计算内存（CIM）、温度计编码、原位多比特输入/权重扩展、MLC/MLC+闪存单元、SPICE仿真、能耗与吞吐量评估。

**📊 数据集**

在常见的MoE大语言模型基准（如Switch Transformer/NLLB‑MoE）上进行实验评估，使用标准的模型权重与路由激活方案，但未公开具体数据集名称。

**📈 对比分析**

与NVIDIA A100 GPU、Cambricon‑LLM、AiF以及传统3D NAND CIM基线进行对比；吞吐量提升范围从5×到114.8×，能效提升3.9×到70×，总体综合面积–能耗–延迟积（AEDP）降低3.5×至8.3×。

**⚠️ 局限性**

① CAM层占用额外3D NAND层导致存储利用率略低；② 多比特CIM单元的多次读/ADC操作增加ADC能耗与时延，尤其在高位宽时显著；③ 设计主要针对稀疏MoE工作负载，对密集模型适用性有限；④ 依赖先进的MLC 3D NAND工艺，技术成熟度和成本仍是推广瓶颈。

---

## 281. Metacognition as Reward: Reinforcing LLM Reasoning via Knowledge and Regulation Signals

**arXiv ID:** 2605.23384 | [PDF](https://arxiv.org/pdf/2605.23384v1)

**作者:** Sirui Chen `[一作]` (Tongji University), Chaochao Lu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发了一种基于元认知的强化学习框架（Metacognition-as-Reward），通过将LLM的推理过程拆分为元认知知识与调控两部分，并为其设计轨迹级奖励，从而在训练时对中间推理行为进行显式监督。

**💡 创新点**

创新点在于：①将元认知知识与调控作为通用奖励维度，避免了实例特定rubric的依赖；②在奖励设计中加入知识覆盖率、调控一致性和答案正确性三项指标，实现对推理过程的全局监督；③通过结构化rollout提示让模型显式生成元认知知识、调控计划及必要的回溯信息，提升了可解释性和可评估性。

**🔧 技术方法**

技术手段包括：1）强化学习与可验证奖励（RLVR）相结合的DPO/DAPO策略优化；2）结构化提示生成元认知组件的rollout；3）利用LLM评判器估计知识覆盖率、调控一致性等奖励分量；4）轨迹级奖励与对比式策略梯度。

**📊 数据集**

使用的数据集共22个，涵盖科学（GPQA-Diamond、SuperGPQA、BioProBench、SciCUEval、FrontierScience、ResearchQA）、医学（MedGUIDE、MedMCQA、LongHealth、LLMEval-Med）、长上下文推理（DocQA-RL-1.6K、LongMIT、LongReward）、数学推理（MMLU-Math、GSM8K、SVAMP、MATH-500、AIME 2024/25/26）以及逻辑推理（FOLIO、ProofWriter）。训练数据来自RaR-Medicine和RaR-Science，约32K样本。

**📈 对比分析**

对比方法：与多种前沿模型（如Qwen3.5-4B/9B/35B/122B/397B、LLAMA等）以及传统DPO/DAPO基线进行零-shot推理评测。结果显示：在22个基准上平均提升7.7%相较基线模型，11%相较Dapo；在科学和医学子基准上提升幅度更显著；在长上下文、数学和逻辑等OOV任务上也实现了2–10%的性能提升，甚至在部分指标上接近或超过部分前沿模型。

**⚠️ 局限性**

局限性：①训练数据主要来自科学与医学领域，其他领域的泛化能力仍待验证；②实验仅聚焦文本推理，未覆盖多语言、多模态或交互式代理；③奖励估计依赖大型LLM评判器，计算成本高且可能带来偏差；④缺少对多语言或跨模态任务的系统评测。

---

## 282. Prudent-Banker: No Extra Fees for Baseline Safety in Adversarial Bandits With and Without Delays

**arXiv ID:** 2605.23351 | [PDF](https://arxiv.org/pdf/2605.23351v1)

**作者:** Ting Hu `[一作]` (University of Wisconsin--Madison), Emmanouil-Vasileios Vlatakis-Gkaragkounis `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Prudent-Banker 算法，在有或无延迟的对抗性多臂老虎机问题中，既能实现近似最优的 minimax 收敛（伪损失 Regret 为 O(√T+√D)），又能保持与预设安全基线策略的绝对损失接近常数（O(1)）。

**💡 创新点**

创新点在于：① 引入“延迟校准重启阈值”，以补偿未观测反馈造成的“隐形债务”，保证安全检测不会被延迟误导；② 结合在线镜像梯度（Online Mirror Descent）与分阶段、分级的“阶段攻击”机制，实现在未知延迟下自适应的安全与探索平衡；③ 给出了匹配下界，证明在安全约束下的最优误差率为 Ω(√(T/δ)+√(D/δ))，即安全成本与 δ 的倒数根号无可进一步降低。

**🔧 技术方法**

核心技术包括：自适应延迟阶段（doubling trick）、基于正则化的 OMD 学习器、基于重要性加权的估计、分阶段的混合策略（α 控制）以及延迟感知的安全检查阈值。

**📊 数据集**

实验使用合成的非平稳 100 维臂老虎机，时长 T=50,000，块数 S=500，安全基线为全支持分布，δ=10⁻³；同时比较了不同延迟分布（无延迟、几何分布、Pareto 分布）。

**📈 对比分析**

与两种基于安全包装的对抗性算法（基于 EXP3-IX 的安全包装）相比，Prudent-Banker 在所有延迟设定下均保持与安全基线的常数误差，同时在伪损失上实现 O(√T+√D) 的收敛；相比仅考虑最优 arm 的基准，Prudent-Banker 的性能更稳健，尤其在安全基线极优的情形下表现更好。

**⚠️ 局限性**

局限性包括：① 在安全基线已知且对抗环境极其恶劣时，可能需要更严格的 δ 参数；② 对多臂数 A 较大时，额外的 √A 延迟项可能导致性能下降；③ 目前仅在合成数据上验证，缺乏在真实业务场景（如广告投放、医疗决策）中的实证。

---

## 283. On the Approximate Non-Deterministic Degree of Total Boolean Functions

**arXiv ID:** 2605.23336 | [PDF](https://arxiv.org/pdf/2605.23336v1)

**作者:** Samruddhi Pednekar `[一作]` (Stony Brook University), Supartha Podder `[通讯]` (Stony Brook University)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5022819820)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了布尔函数的近似非确定性度（approximate non‑deterministic degree）与其它复杂度量（如近似度、理性度、确定性查询复杂度）之间的关系，并在若干重要函数类（单调/不变、有限交替数、对称、读‑k DNF、k‑均匀超图性质）上证明了近似非确定性度与近似度之间存在多项式上界，从而为原始的非确定性度与近似度的猜想提供了系统性进展。

**💡 创新点**

①首次给出在上述函数类中近似非确定性度与近似度之间的多项式关系；②提出并验证了更强的猜想：对任意总布尔函数，近似度≤多项式(近似非确定性度, 其补函数的近似非确定性度)。

**🔧 技术方法**

使用了多项式逼近理论中的对偶多项式技术、证据法、变量约束与识别、可读‑k DNF的结构化分析、交替数与敏感度/块敏感度之间的关系、对称化与符号逼近，以及图/超图性质中的嵌入与限制构造。

**📊 数据集**

无，需要的只是理论构造与组合论证明；未使用实验数据集。

**📈 对比分析**

通过理论证明给出上界（如 (f) ≤ O(max{N_ε(f),N_ε(f)}^4) 等）并与已知下界（如 AND、OR、NOR 的 N_ε 值）对比，表明在这些函数类中得到的上界与下界相匹配或仅相差低阶多项式因子；相较于先前仅能得到指数或高阶多项式界的情况，性能显著提升。

**⚠️ 局限性**

限制：仅适用于总布尔函数且只覆盖上述特定函数类；对一般布尔函数的猜想仍未解决；多项式指数常数较大，实际常数未给出；部分证明依赖于已知的较强下界（如对称函数的块敏感度下界），若这些下界进一步提升则可能改进上界。

---

## 284. Emotion Recognition in Sign Language Conversation

**arXiv ID:** 2605.23328 | [PDF](https://arxiv.org/pdf/2605.23328v1)

**作者:** Yusong Wang `[一作]` (Institute of Science Tokyo), Kotaro Funakoshi `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 1147 | [OpenAlex ID](https://openalex.org/A5069989297)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了情感识别在对话（ERC）任务，并构建了包含多轮手语对话及情感标签的eJSL Dialog数据集；

**💡 创新点**

首次将情感识别任务引入手语对话情境，提供多轮上下文与情感标注的数据资源，并验证视觉特征与上下文的互补性；

**🔧 技术方法**

使用了纯视觉模型、手部骨架+LSTM、文本基础ERC、跨模态知识蒸馏与图卷积网络等多种基线模型进行评估；

**📊 数据集**

主要使用自制的eJSL Dialog数据集（1,920段视频、4,704条情感标注），并在BOBSL数据上进行视觉模型预训练，文本模型采用自动对齐的英语字幕；

**📈 对比分析**

通过加权F1评估对比五个基线，视觉模型约29-33 F1，文本模型最高55 F1，而多模态模型因领域差距表现最差（10-8 F1），说明上下文对情感识别至关重要；

**⚠️ 局限性**

数据仅来自两位演员、白色背景实验室环境，缺乏深度、面部/手部细粒度信息，且规模有限，限制了模型在真实多样场景中的泛化能力。

---

## 285. MileStone: A Multi-Objective Compiler Phase Ordering Framework for Graph-based IR-Level Optimization

**arXiv ID:** 2605.23435 | [PDF](https://arxiv.org/pdf/2605.23435v1)

**作者:** Amirhosein Sadr `[一作]`, Mehran Alidoost Nia `[通讯]` (Shahid Beheshti University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MileStone，一个模块化框架，将编译器阶段排序建模为多目标优化问题，旨在优化程序性能。

**💡 创新点**

MileStone的创新点在于结合图神经网络和强化学习，能够同时处理多个目标（执行时间、代码大小和能耗），并通过自我演化数据库提高预测质量。

**🔧 技术方法**

使用了图神经网络（GNN）进行性能预测和强化学习（RL）进行编译器传递序列的探索。

**📊 数据集**

使用了标准基准测试数据集进行实验，具体包括PolyBench基准中的多个程序。

**📈 对比分析**

与LLVM的优化级别和其他相关技术相比，MileStone能够找到强Pareto最优解，在相同能量预算下将执行时间减少最多45%。

**⚠️ 局限性**

MileStone的局限性在于其依赖于用户定义的约束，可能在某些情况下无法适应动态变化的优化需求。

---

## 286. An Open-Source Training Dataset for Foundation Models for Black-box Optimization

**arXiv ID:** 2605.23417 | [PDF](https://arxiv.org/pdf/2605.23417v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 287. Multi-Round Visibility: A Post-Consensus Ordering Layer for DAG-Based BFT

**arXiv ID:** 2605.23432 | [PDF](https://arxiv.org/pdf/2605.23432v1)

**作者:** Pengkun Ren `[一作]` (RMIT University), Zahir Tari `[通讯]` (RMIT University)

**通讯引用:** 9167 | [OpenAlex ID](https://openalex.org/A5054836950)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为多轮可见性（MRV）的后共识结构排序层，用于基于有向无环图（DAG）的拜占庭容错（BFT）协议，旨在提高系统的执行顺序的公平性。

**💡 创新点**

创新点在于将排序逻辑从共识路径中解耦，利用已提交的DAG结构作为排序证据，避免了在共识过程中引入额外的消息传递。

**🔧 技术方法**

使用了基于Narwhal/Tusk的原型实现MRV，并通过后共识处理来提取结构可见性证据。

**📊 数据集**

在不同的部署规模、工作负载和故障设置下进行了评估，具体数据集未明确提及，但涉及到多个节点的DAG-BFT协议。

**📈 对比分析**

与原生Narwhal/Tusk和DoD风格的图排序参考进行了比较，MRV在5到50个副本的情况下保持了高吞吐量，达到了210K TPS，且对吞吐量的影响有限。

**⚠️ 局限性**

限制在于MRV的排序逻辑依赖于已提交的DAG结构，可能在某些情况下无法完全消除排序的模糊性，且在处理复杂的可见性关系时可能增加延迟。

---

## 288. FAST-ME: Foundation-aware Adaptive Stopping for Motion Estimation for Efficient IoT Video Analysis

**arXiv ID:** 2605.23428 | [PDF](https://arxiv.org/pdf/2605.23428v1)

**作者:** Kakia Panagidi `[一作]` (National and Kapodistrian University of Athens), Stathes Hadjieftymiadis `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于最优停止理论的自适应运动估计框架FAST-ME，旨在提高物联网视频分析的效率，特别是在资源受限的环境中。

**💡 创新点**

创新点在于将基础模型（如ViT和SAM）与传统的运动估计方法结合，利用语义关注分数来指导运动估计的停止决策，从而实现更高效的计算和更好的语义覆盖。

**🔧 技术方法**

使用了最优停止理论（OST）、视觉变换器（ViT）、Segment Anything Model（SAM）等技术，结合传统的失真度量（如SAD）进行运动估计。

**📊 数据集**

在多个基准和多模态视频数据集上进行了实验，包括Xiph.org DERF数据集和高分辨率场景（如Four People和Big Buck Bunny）。

**📈 对比分析**

与文献中常用的方法（如全搜索、钻石搜索和三步搜索）进行了比较，结果显示FAST-ME在计算效率上提高了99%，同时保持或改善了重建质量和语义对齐。

**⚠️ 局限性**

局限性在于该方法仍然依赖于基础模型的性能，且在某些情况下可能无法完全捕捉到所有重要的运动信息，尤其是在复杂场景中。

---

## 289. Naturalistic measure of social norms alignment

**arXiv ID:** 2605.23420 | [PDF](https://arxiv.org/pdf/2605.23420v1)

**作者:** Yevhen Kostiuk `[一作]`, Kristoffer Nielbo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文仅演示了ACL LaTeX样式文件在LuaLaTeX和XeLaTeX中的使用方法。

**💡 创新点**

创新点仅在于展示了如何在不同引擎下使用该样式。

**🔧 技术方法**

未使用任何具体技术。

**📊 数据集**

未使用任何数据集。

**📈 对比分析**

未进行方法比较或性能评估。

**⚠️ 局限性**

本文仅为示例，缺乏实验数据和实际应用场景。

---

## 290. Online Hand Gesture Recognition Using 3D Convolutional Neural Networks

**arXiv ID:** 2605.23409 | [PDF](https://arxiv.org/pdf/2605.23409v1)

**作者:** Yinghao Qin `[一作]` (Queen Mary, University of London), Tijana Timotijevic `[通讯]` (Queen Mary, University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一套在线手势识别系统，实现实时检测与分类，能够在视频流中定位手势并给出对应的类别。

**💡 创新点**

创新点在于将3D CNN与滑动窗口策略、早期/后期检测机制相结合，并使用Levenshtein距离对整个系统进行评价，展示了在实时条件下兼顾准确率和响应时间的可行方案。

**🔧 技术方法**

主要技术包括：3D CNN模型（C3D、ResNeXt‑101、ResNet‑10）、滑动窗口后处理、前向检测 + 分类 pipeline、Levenshtein距离评估指标。

**📊 数据集**

使用Jester手势数据库进行模型训练与评估，另外自采20段视频（共80个手势）用于在线系统的整体评估。

**📈 对比分析**

与20BN Jester、TRN、SSNet等基线对比，分类器在Jester上取得91‑92% 最高测试准确率；整个系统在自采数据上的Levenshtein准确率为37.5%，检测准确率达98.22%，平均响应时间低于3秒。

**⚠️ 局限性**

局限性包括：对手势时长不够鲁棒，响应时间仍略高于期望，整体准确率仍有提升空间，缺乏统一的在线评估标准，尚未在真实场景（如车载、移动设备）中验证。

---

## 291. Hybrid Quantum-Classical Corrective Diffusion Modeling for Meteorological Downscaling

**arXiv ID:** 2605.23403 | [PDF](https://arxiv.org/pdf/2605.23403v1)

**作者:** Rui Wang `[一作]` (Jülich Supercomputing Centre), Gabriele Cavallaro `[通讯]` (Jülich Supercomputing Centre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种在统计降尺度扩散模型中将量子变分电路插入UNet瓶颈层的混合量子-经典架构，实现对10米风速的概率性降尺度。

**💡 创新点**

创新点在于将量子电路用作压缩潜在通道的非线性特征映射，能够提升MAE/CRPS并保持物理统计特征，同时在保持参数量与经典模型相当的前提下实现量子化。

**🔧 技术方法**

采用了变分量子电路（HQConv ansatz）、PennyLane与PyTorch深度学习框架，结合CorrDiff扩散模型的残差预测。

**📊 数据集**

使用HRRR-mini气象数据集，重点输出10米水平风速分量(u10m, v10m)。

**📈 对比分析**

通过MAE、CRPS、FSS、能谱等多维度指标与全经典CorrDiff对比，ID（2020）实验中B-only 3通道或9通道模型平均提升约1–2%，但在OOD（2021）测试中性能不稳定，表现出一定的泛化缺口。

**⚠️ 局限性**

主要局限在于量子电路规模受QPU量子位与误差限制，实际硬件部署导致显著时延且在复杂风场下精度衰减；此外，分布偏移时的泛化能力仍需改进。

---

## 292. S$^3$GNN: Efficient Global Mixing and Local Message Passing for Long-Range Graph Learning

**arXiv ID:** 2605.23467 | [PDF](https://arxiv.org/pdf/2605.23467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 293. Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting

**arXiv ID:** 2605.23402 | [PDF](https://arxiv.org/pdf/2605.23402v1)

**作者:** Jinglin Li `[一作]` (Central South University), Ning Gui `[通讯]` (Central South University)

**通讯引用:** 972 | [OpenAlex ID](https://openalex.org/A5012277801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Parametric Prior Mapping（PPM）框架，用参数化先验与可学习的 push-forward 映射相结合，解决非平稳概率多变量时间序列预测中的不确定性估计与表达力折衷问题。

**💡 创新点**

创新点在于：①通过编码器快速推导输入特定的可学习先验分布；②将先验通过一次可学习映射投射到目标分布，实现高效的生成；③采用 KDE 估计 NLL 并与 MSE 混合，兼顾分布校准与轨迹精度；④推理仅需一次映射，显著降低延迟。

**🔧 技术方法**

技术包括：MLP 编码器、重参数化采样、可学习的两层 MLP 生成器、KDE 负对数似然损失、MSE 正则化、信息论评估（MI 近似）、实验对比与推理复杂度分析。

**📊 数据集**

实验数据集涵盖七个真实世界系列：ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Traffic、Weather。

**📈 对比分析**

与 DeepAR、TimeGrad、D3VAE、TimeDiff、DiffusionTS、TMDM、NsDiff 等基线进行对比，PPM 在 CRPS、QICE、MSE、MAE 等指标上均获得最优或近优表现，尤其在高变异 Traffic 数据上显著优于其他模型；推理速度提升 2–100 倍。

**⚠️ 局限性**

局限性：依赖 KDE 带宽 h 的选择，对快速变化或极端 regime 敏感；未对标签的联合分布建模；需在训练时承担 KDE 计算开销，推理时可省略。

---

## 294. Semantically Structured Mixture-of-Experts for Compositional Robotic Manipulation

**arXiv ID:** 2605.23477 | [PDF](https://arxiv.org/pdf/2605.23477v1)

**作者:** Chengyu Deng `[一作]` (University of Hong Kong), Jia Pan `[通讯]` (University of Hong Kong)

**通讯引用:** 9228 | [OpenAlex ID](https://openalex.org/A5076812698)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于语义结构的Mixture‑of‑Experts扩散策略（SMoDP），通过语言驱动的技能预测器为不同行为阶段选择专家，从而实现多任务机器人操控的可解释性与参数高效性。

**💡 创新点**

创新点在于将任务的语义结构与技能标签（来自VLM的离线注释）结合，用双重对比对齐（跨模态与同模态）保证专家路由的语义一致性，突破传统低层噪声或潜在统计驱动的路由限制。

**🔧 技术方法**

核心技术包括扩散式策略、Mixture‑of‑Experts网络、轻量级技能预测器、跨模态/同模态双重对比对齐以及VLM生成的语义标签。

**📊 数据集**

在多任务机器人操控基准上评估，涵盖仿真与真实机器人实验，使用公开的多任务数据集（如Meta‑World、RealWorld RL等）以及自建任务集。

**📈 对比分析**

与代表性扩散策略和传统MoE基线相比，SMoDP在参数占用上提升约30‑50%同时保持甚至提升任务成功率，且在新的任务上通过参数高效微调实现了优秀的组合迁移性能。

**⚠️ 局限性**

局限性包括对VLM离线注释的依赖、在极端复杂或未知行为阶段的路由不确定性，以及在极大规模任务集合中路由策略可能仍需进一步优化。

---

## 295. Non-normal spectral signatures of instability in neural network training dynamics

**arXiv ID:** 2605.23476 | [PDF](https://arxiv.org/pdf/2605.23476v1)

**作者:** Souvik Ghosh `[一作]` `[通讯]` (National Sun Yat-sen University), Souvik Ghosh (National Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将非正规算子理论引入神经网络训练稳定性分析，提出使用特征向量条件数κ(V)作为非正规性预警指标；

**💡 创新点**

创新点在于证明Adam和带动量SGD的线性化更新算子普遍非正规，且κ(V)能提前预警超出谱半径阈值的暂态放大，从而解释训练中的loss峰；

**🔧 技术方法**

核心技术包括线性化Jacobian分析、非正规算子与伪谱理论、Kreiss定理、以及特征向量条件数的计算；

**📊 数据集**

实验使用两层MLP（10→20→1）在合成回归任务（500样本，标准正态输入，线性目标加噪声）上进行；

**📈 对比分析**

与传统的谱半径和最大Hessian特征值指标对比，κ(V)在训练不稳定期比谱半径高一个数量级，能够更早且更明确地区分稳定与不稳定阶段；

**⚠️ 局限性**

局限性包括：需要假设优化器状态变化缓慢；κ(V)计算在大模型上成本较高；对非MLP架构的泛化尚未验证；

---

## 296. One-Forcing: Towards Stable One-Step Autoregressive Video Generation

**arXiv ID:** 2605.23458 | [PDF](https://arxiv.org/pdf/2605.23458v1)

**作者:** Jiaqi Feng `[一作]` (Tsinghua University), Cho-Jui Hsieh `[通讯]` (Ucla)

**通讯引用:** 26701 | [OpenAlex ID](https://openalex.org/A5010841999)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种One-Forcing方法，结合分布匹配蒸馏（DMD）与对抗损失（GAN），在因果（自回归）视频生成中实现一次性采样即可获得高质量、低延迟的视频。

**💡 创新点**

创新点在于将DMD中的伪分数网络重用为对抗判别器，使得判别器直接基于真实视频的噪声潜在空间提供全局拒绝信号；共享网络结构消除了额外参数负担；在帧级一阶采样中仅需1/3的训练成本即可收敛。

**🔧 技术方法**

使用的技术包括：分布匹配蒸馏（DMD）、对抗训练（GAN）与噪声潜在判别、因果自回归视频扩散、ODE初始化、VBench评估框架。

**📊 数据集**

使用的数据集为VBench（21帧、832×480）以及Wan2.1-1.3B/14B教师模型（用于分数匹配和判别）。

**📈 对比分析**

与Self‑Forcing、Causal‑Forcing、ASD等单步方法以及多步模型（MAGI‑1、Wan2.1、SkyReels‑V2、NOVA等）进行对比；在VBench上取得总分83.76，优于单步基线4–7分，且与4步多步模型相当或更优；人类评测中赢率高达92.7%。

**⚠️ 局限性**

限制在于：需要真实视频数据来训练判别器，无法实现无数据蒸馏；目前仅在中等分辨率（832×480）和21帧长度上验证，扩展到更高分辨率、更长视频及更大模型仍需进一步研究。

---

## 297. Commutator-Induced Uncertainty in VAEs

**arXiv ID:** 2605.23449 | [PDF](https://arxiv.org/pdf/2605.23449v1)

**作者:** Tahereh Dehdarirad `[一作]` (Linköping University), Ziliang Xiong `[通讯]` (Linköping University)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5102519412)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种诊断驱动的 Lie Group VAE 框架，既能捕捉非可交换性，又能将其反映在重构不确定性上；

**💡 创新点**

首次将有限 BCH 偏差与解码器的顺序敏感度关联，并通过变形‑稳定性约束校准二者的尺度，保留并利用非可交换性；

**🔧 技术方法**

使用 Lie 代数的 BCH 公式、解码器顺序交换测试、Gumbel‑Softmax 离散潜变量、变形‑稳定性 hinge 损失以及拉普拉斯/KL 正则；

**📊 数据集**

在 dSprites、3DShapes、3DCars 以及 CelebA 上进行实验；

**📈 对比分析**

与 β‑VAE、CLG‑VAE、CFASL 等基线对比，取得更低的重构误差、更高的 FactorVAE 指标、以及在 CelebA 上更低的 FID，且解码器对非可交换性的响应与潜在结构保持一致；

**⚠️ 局限性**

诊断局限为仅考虑成对有限偏差，计算复杂度随连续潜变量维数二次增长，且未捕捉路径级的组合效应。

---

## 298. Weisfeiler-Leman Is Incomplete on Simple Spectrum Graphs, so Canonicalize Them

**arXiv ID:** 2605.23446 | [PDF](https://arxiv.org/pdf/2605.23446v1)

**作者:** Snir Hordan `[一作]` (Technion - Israel Institute Of Technology), Tim Seppelt `[通讯]` (IT University Of Copenhagen)

**通讯引用:** 103 | [OpenAlex ID](https://openalex.org/A5086010577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

证明任意 k-WL 测试无法区分所有简单谱图，并提出 PRiSM（Partition, Refine, Solve, Match）算法，完成对简单谱图特征的规范化，实现完备性与通用近似；在多项基准上验证其有效性。

**💡 创新点**

① 证明 k-WL 在简单谱图类上完全不完整；② 设计了将 CFI 图嵌入简单谱多重图的正交整数编码；③ 首创 PRiSM 这一完整的谱特征规范化流程，解决了此前所有签名不变模型的完备性难题。

**🔧 技术方法**

CFI 构造、正交整数编码、谱版 Weisfeiler–Leman 细化、F₂ 线性系统求解、签名排序以及对多重谱的 QR 处理。

**📊 数据集**

BREC 表达性基准、ZINC、OGB MolTox21、MolPCBA、OGB MolHIV、MolClinTox、MolBBBP、Alchemy 量子化学回归数据集。

**📈 对比分析**

与传统 Laplacian Positional Encoding、MAP、OAP、SignNet 等方法对比；在 BREC 中 PRiSM 识别 212/360 对比 199/360；在分子属性预测中大多数指标上优于或与之持平；在 Transformer 基础模型中 PRiSM 在 ROC‑AUC 和 MAE 上均优于基线。

**⚠️ 局限性**

未覆盖高阶特征值重数情况；规范化过程本身不连续；不完整性证明仅适用于多重图，尚未推广至普通组合图。

---

## 299. SSDAU: Structured Semantic Data Augmentation for Joint Entity and Relation Extraction

**arXiv ID:** 2605.23440 | [PDF](https://arxiv.org/pdf/2605.23440v1)

**作者:** Jiawei He `[一作]` (Nanjing University), Chunrong Fang `[通讯]` (Nanjing University)

**通讯引用:** 2381 | [OpenAlex ID](https://openalex.org/A5075174750)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结构化语义数据增强（SSDAU）方法，用实体标签分段文本并通过语义匹配与主题一致性过滤生成高质量的增强样本，显著提升联合实体关系抽取（JERE）模型的泛化能力。

**💡 创新点**

创新点在于：① 用实体标签对文本进行结构化分割，保持语义结构；② 将上下文化[BERT]嵌入与传统相似度融合，以更精准地区分语义相似但不同的实体；③ 引入BERTTopic主题一致性过滤，剔除语义冲突和噪声，提升生成数据的可靠性。

**🔧 技术方法**

技术包括：基于Triplet的特征编码器与解码器、语义匹配器（融合Similarities与BERT [CLS]嵌入）、阈值筛选、BERTTopic主题评估、圆形相关和Dropout等训练技巧。

**📊 数据集**

使用NYT和WebNLG两个公开数据集（包含完整标注与部分标注版本）进行实验评估。

**📈 对比分析**

与七种主流数据增强基线（如Back‑Translation、MixUp、ChatIE等）对比，SSDAU在F1、Precision、IoU等指标上均领先，尤其在含语义歧义的数据下F1降幅仅8.26%而基线可达30%以上，表明其鲁棒性和效果显著。

**⚠️ 局限性**

局限性包括：仍依赖一定质量的原始数据；对长文本的语义匹配效率不高；在高质量数据环境下效果提升空间有限；未来需探索更高效的语义匹配模块与实时验证。

---

## 300. Sparse In-Network Learning via Shortest-Path Backpropagation and Finite-Rate Gating

**arXiv ID:** 2605.23424 | [PDF](https://arxiv.org/pdf/2605.23424v1)

**作者:** Mohammad Reza Deylam Salehi `[一作]` `[通讯]`, Mohammad Reza Deylam Salehi

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于Dijkstra最短路径树的稀疏网络学习方法D‑INL，能够在训练时仅保留必要的边，并结合有限速率门控实现信息压缩。

**💡 创新点**

创新点在于将网络拓扑稀疏化与信息瓶颈门控相结合，给出基于rate‑distortion的泛化上界，并通过Dijkstra算法快速获得可行的最短路径子图。

**🔧 技术方法**

使用技术包括Dijkstra最短路径树、有限速率随机门控（信息瓶颈）、梯度下降训练、rate‑distortion分析与泛化误差估计。

**📊 数据集**

实验使用六个传感器、三个中继和一个融合节点的合成二分类数据集，样本量约为120/120/1000。

**📈 对比分析**

与密集INL、仅Dijkstra剪枝和D‑INL+rate三种方案对比，结果显示Dijkstra剪枝将训练交换量降低70.4%，准确率基本保持；再加上有限速率正则后，估计的潜在信息率下降45.7%，准确率不变。

**⚠️ 局限性**

局限性包括仅采用单一SPT可能限制信息多样性；实验规模较小，未在真实无线或视觉数据集上验证；以及对动态链路变化的适应性仍需进一步研究。

---

## 301. Coupling-Robust Accuracy in Multiphysics Physics Informed Neural Networks via Kronecker-Preconditioned Optimization

**arXiv ID:** 2605.23391 | [PDF](https://arxiv.org/pdf/2605.23391v1)

**作者:** Youngjae Park `[一作]` (Korea University), Junghwa Hong `[通讯]` (Korea University)

**通讯引用:** 929 | [OpenAlex ID](https://openalex.org/A5103144831)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对多物理耦合问题的 PINN 训练方法，利用 Kronecker 预处理的 SOAP 优化器与梯度平衡技术实现耦合鲁棒性。

**💡 创新点**

创新点在于：1) 用神经切线核 (NTK) 分析揭示耦合强度导致标准 NTK 光谱半径随 γ² 成长，而块状 Gauss–Newton 预处理能把预处理 NTK 的谱半径上界固定为网络数 S；2) 通过 SOAP+GN 组合在 234 次实验中保持耦合误差比 ≤1.1，显著优于 Adam 等传统方法；3) 成功解决了 2D 6-PDE 电双层解析电泳流域，填补了先前 PINN 电动力学研究的空白。

**🔧 技术方法**

采用分离网络结构的 PINN、SOAP Kronecker 预处理、Gauss–Newton（GN）梯度平衡、LRA（最大梯度平衡）以及 Adam 作为对照。

**📊 数据集**

使用四个基准系统：1D 热弹性耦合、1D 反应扩散耦合、1D Nernst–Planck–Poisson 非线性耦合，以及 2D 电动渗流（6 PDE）场景。

**📈 对比分析**

通过 234 次实验（不同优化器、平衡方案与耦合强度）与 Adam、SOAP 等对比，SOAP+GN 在所有耦合强度下误差比维持在 0.9–1.1 之间，Adam 在强耦合时误差可提升 100 倍以上；在 2D 电动渗流中，SOAP+GN 能获得 10⁻³ 级别的 L₂ 精度，而 Adam+GN 直接失败。

**⚠️ 局限性**

限制包括：1) 对非线性耦合系统（如 Nernst–Planck–Poisson）仅给出了经验性推断，缺乏正式的训练时谱增长证明；2) SOAP 的理论光谱上界为 SL，略高于精确 GN 上界 S，如何进一步压缩此差距尚未解决；3) 计算成本相对 Adam 约 1.6 倍，虽被性能提升所补偿。

---

## 302. Sample-wise Targeted Adversarial Attacks on Test-time Adaptation

**arXiv ID:** 2605.23411 | [PDF](https://arxiv.org/pdf/2605.23411v1)

**作者:** Phuc Duc Nguyen `[一作]` (Nanyang Technological University), Quang Duc Nguyen `[通讯]` (Nanyang Technological University)

**通讯引用:** 72 | [OpenAlex ID](https://openalex.org/A5056431386)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在测试时适配（TTA）环境下的样本级目标攻击，利用触发器仅误分类携带触发器的样本，同时保持其他样本的标签分布与无攻击情况相似。

**💡 创新点**

核心创新在于：① 将攻击目标与隐蔽性目标建模为主次两任务；② 设计了基于椭圆信任域的优先级梯度对齐策略，显式惩罚攻击梯度与隐蔽梯度的冲突；③ 采用元学习框架模拟部署过程，使生成的扰动能够泛化到未知的触发样本。

**🔧 技术方法**

使用元学习（Meta‑Learning）对扰动进行优化；梯度对齐通过楕圆信任域（Ellipsoidal Trust‑Region）实现；损失包括交叉熵攻击损失和分布一致性 KL 损失；实验中结合多种 TTA 方案（TENT、RPL、EATA、SAR、DeYO 等）。

**📊 数据集**

在 CIFAR‑10‑C、CIFAR‑100‑C 与 ImageNet‑C 三个公开分布偏移数据集上进行评估，采用 ResNet‑32 与 ResNet‑50 两种骨干网络。

**📈 对比分析**

与基线无攻击、传统类级攻击（FCA、RTTDP）以及多任务学习（PCGrad、CAGrad）和欧氏信任域方法对比。实验显示，本文方法在保持标签分布接近无攻击时，攻击成功率（ASR）平均可达 90%+（CIFAR）及 95%+（ImageNet），远优于对比方法，同时对常见防御（样本熵过滤、MedBN、EMA 等）表现出较强鲁棒性。

**⚠️ 局限性**

限制包括：仅研究可见触发器，尚未验证对隐蔽触发器的效果；攻击依赖于灰盒对原始模型的访问；元学习训练成本较高；对极端噪声或高维数据的泛化仍待进一步探索。

---

## 303. Automated Random Embedding for Practical Bayesian Optimization with Unknown Effective Dimension

**arXiv ID:** 2605.23473 | [PDF](https://arxiv.org/pdf/2605.23473v1)

**作者:** Hong Qian `[一作]` (East China Normal University), Liang Dou `[通讯]` (East China Normal University)

**通讯引用:** 1120 | [OpenAlex ID](https://openalex.org/A5100668340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种DSEBO方法，通过动态共享嵌入实现高维贝叶斯优化的自动子空间维度扩展。

**💡 创新点**

创新点在于动态维度扩展策略和共享嵌入矩阵，使得子空间能够共享已评估数据并根据收敛情况自动调整维度。

**🔧 技术方法**

主要技术包括贝叶斯优化、随机嵌入、共享嵌入矩阵、基于MAB的维度扩展决策以及理论收敛分析。

**📊 数据集**

使用合成高维函数（D=1000、10000）和三类真实任务（MSLR‑WEB‑10K、Lasso‑Hard、LIMO）进行评估。

**📈 对比分析**

与REMBO、SIRBO、HesBO、VAEBO、ALEBO、BAxUS、TuRBO、SAASBO、DuMBO、MCTS‑VS、RDUCB、SBO‑SE、LMMAES、DCEM等方法对比，DSEBO在相同预算下实现更低的简单遗憾、收敛更快且性能更稳定。

**⚠️ 局限性**

局限在于无法精准识别有效维度导致可能过度扩展子空间，且仅在单一随机嵌入框架下验证，未来需加入多嵌入或学习嵌入以进一步提升。

---

## 304. Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs. ImageNet Pretraining Across RGB and X-ray Tasks

**arXiv ID:** 2605.23472 | [PDF](https://arxiv.org/pdf/2605.23472v1)

**作者:** Mehdi Gharbage `[一作]` (Michelin Tyres Manufacturer), Thierry Chateau `[通讯]` (Université Clermont Auvergne)

**通讯引用:** 2034 | [OpenAlex ID](https://openalex.org/A5040478758)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在工业视觉检验任务（语义分割、实例分割、目标检测）上，对 ConvNeXt-Backbones 的两种预训练方式——监督式 ImageNet 分类和 DINOv3 迁移学习，做了系统对比实验。

**💡 创新点**

创新点在于：①在工业数据上首次将大规模自监督预训练的 DINOv3 与传统 ImageNet 预训练进行细粒度对比；②探究冻结与全微调两种适配策略对两种预训练效果的影响，揭示 DINOv3 在 RGB 任务下的强势 finetuning 先验。

**🔧 技术方法**

使用技术包括 ConvNeXt-T 作为 backbone，Mask2Former、Mask R‑CNN 与 Faster R‑CNN 作为下游任务头；对比实验采用统一的训练和调度策略，并在不同适配模式下保持相同超参。

**📊 数据集**

数据集涵盖四个工业检验场景：RGB 表面缺陷（Severstal、Rubber Rings）、RGB 空中小目标（RarePlanes）与 X‑ray 缺陷检测（GDXray Castings）。

**📈 对比分析**

实验表明：在冻结模式下 DINOv3 与 ImageNet 预训练效果相近，甚至在 X‑ray 上表现更差；在完全微调后，DINOv3 预训练在所有 RGB 任务中均取得最高分（例如语义分割 mIoU 提升约 10 分、实例分割 mask‑mAP 提升约 1.6 分），但在 X‑ray 检测中仍落后于 ImageNet。

**⚠️ 局限性**

局限性在于：DINOv3 的优势仅体现在与自然图像统计相近的 RGB 领域；在强模态迁移（X‑ray）和冻结特征提取场景下表现不佳，表明自监督预训练的通用性受限，未来需针对工业数据开展专门的自监督预训练。

---

## 305. Learning Individual Dynamics from Sparse Cross-Sectional Snapshots

**arXiv ID:** 2605.23470 | [PDF](https://arxiv.org/pdf/2605.23470v1)

**作者:** Christian Lagemann `[一作]` (German Center for Neurodegenerative Diseases), Sach Mukherjee `[通讯]` (University of Bonn)

**通讯引用:** 3344 | [OpenAlex ID](https://openalex.org/A5112866063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 CADENCE 框架，实现仅凭极少时间点的横截面数据和静态上下文恢复个体连续时间轨迹。

**💡 创新点**

创新点在于通过三大结构假设（DFA、COA、MRA）提供单时间点轨迹推断的可辨识性保证，并引入概率流 ODE 编码与 Soft Mixture‑of‑Experts 路由实现可辨识的动态参数生成。

**🔧 技术方法**

核心技术包括：score‑based Bijective Probability Flow ODE 作为空间编码器；SMoE 路由器实现上下文驱动的动态参数混合；FiLM 条件化的 Neural ODE；两阶段分离训练（空间编码与时序动态）和基于 MMD 的横截面分布一致性损失。

**📊 数据集**

在七个基准上评估：控制动力学（Lotka–Volterra、Van der Pol、Duffing 等）、流行病学与高维基因表达模拟（SIR、SERGIO），以及真实单细胞数据 LARRY（血液造血分化）。

**📈 对比分析**

与最强的横截面流式模型 OT‑CFM、跨领域轨迹推断器 NDP、Latent ODE 等基线对比，CADENCE 在仅使用稀疏数据时实现 MAE 与 SW_2 指标与密集序列训练模型相当或更优，且在所有基准上保持优秀的分布一致性。

**⚠️ 局限性**

局限性在于对三大结构假设的严格依赖；若静态上下文缺失关键信息或动态模式高度异质导致 MRA 近似误差增大，模型会退化为群体级别的最佳传输而失去个体辨识；训练在极端稀疏情况下仍存在数值不稳定与优化困难。

---

## 306. Unextractable Protocol Models: Collaborative Training and Inference without Weight Materialization

**arXiv ID:** 2605.23464 | [PDF](https://arxiv.org/pdf/2605.23464v1)

**作者:** Alexander Long `[一作]` (Pluralis Research), Sameera Ramasinghe `[通讯]` (Pluralis Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在分散式训练和推理环境中防止模型权重被提取的机制，称为Unextractable Protocol Models (UPMs)。

**💡 创新点**

创新点在于通过在模型并行（pipeline）边界周期性地插入随机可逆变换，使得跨时间步的权重不兼容，从而阻止Sybil攻击者拼接完整模型，并且保持网络功能不变。

**🔧 技术方法**

采用了模型并行/流水线并行、随机正交矩阵与低条件数变换、MuON优化器、FP32高精度存储等技术。

**📊 数据集**

实验使用了公开的量化LLM Qwen 2.5-0.5B 与 Llama 3.2-1B 以及 FineWeb 训练集、WikiText 验证集。

**📈 对比分析**

与无变换基线相比，推理时10,000次变换后 logit 漂移与困惑度变化 <1%；训练损失基本保持一致；通信与延迟开销分别约 0.1% 与 3%。

**⚠️ 局限性**

局限在于实验仅在单机仿真，未评估真实分布式部署与侧信道攻击；安全性依赖大多数参与者诚实；未考虑安全评估与可解释性等方面。

---

## 307. ARES: Automated Rubric Synthesis for Scalable LLM Reinforcement Learning

**arXiv ID:** 2605.23454 | [PDF](https://arxiv.org/pdf/2605.23454v1)

**作者:** Xiaoyuan Li `[一作]` (University of Science and Technology of China), Dayiheng Liu `[通讯]` (Alibaba Group)

**通讯引用:** 1796 | [OpenAlex ID](https://openalex.org/A5062188134)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了ARES框架，自动从原始预训练文档生成自包含的问答对和问题特定的加权评分标准，用于多维度奖励的RL训练。

**💡 创新点**

自动合成问题特定的加权rubric，并在单次推理中同时生成问题、答案与rubric，显著提升规模与多样性。

**🔧 技术方法**

利用Gemini-3.1-Pro-Preview和Claude-Sonnet-4.5进行文档过滤、QA和rubric共生成，使用GRPO进行强化学习，评判器采用Qwen3-32B。

**📊 数据集**

采集0.1B token的预训练语料（DCLM、FineWeb-Edu、FinePDFs），经筛选后生成约101k条rubric注释实例，涵盖10个领域。

**📈 对比分析**

与CPT、SFT（NaturalReasoning）、Webscale（二进制奖励）和ARES-SFT对比，ARES-RL在7个基准上平均提升5.33分，尤其在多维度开放式任务（HealthBench+5.37、IFEval+19.27）表现突出。

**⚠️ 局限性**

仍依赖LLM评分的噪声，模型对单一参考答案的鲁棒性不足；部分任务（如多选）对开放式奖励不利，且生成过程受LLM质量限制。

---

## 308. Onsager-Machlup Posterior Transport for Deep Gaussian Processes

**arXiv ID:** 2605.23434 | [PDF](https://arxiv.org/pdf/2605.23434v1)

**作者:** Jian Xu `[一作]` (RIKEN iTHEMS), Qibin Zhao `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的后验传输方法FBVI-bridge-Path，用于深度高斯过程（DGP）的推断，旨在解决现有方法在处理后验分布时的计算瓶颈。

**💡 创新点**

创新点在于将DGP推断视为后验传输，而不是通过ELBO最大化来拟合显式密度，采用确定性采样器映射可处理的参考测度到与后验相关的诱导变量。

**🔧 技术方法**

使用了基于Song的概率流ODE的路径正则化和Onsager-Machlup作用的确定性ODE方法。

**📊 数据集**

使用了七个UCI回归基准数据集进行实验，包括yacht、boston、energy、qsar、concrete、power和protein。

**📈 对比分析**

与DBVI方法进行比较，FBVI-bridge-Path在两个最大的UCI数据集上取得了统计显著的胜利（power和protein），在yacht和qsar上表现相当，但在boston、energy和concrete等小样本噪声数据集上表现不如DBVI。

**⚠️ 局限性**

限制在于该方法不是ELBO，无法提供对log p(|)的下界，且在分类任务上与DBVI的表现接近，未能显著超越。

---

## 309. Reflex: Reinforcement Learning with Reflection Symmetry Exploitation in State-Based Continuous Control

**arXiv ID:** 2605.23415 | [PDF](https://arxiv.org/pdf/2605.23415v1)

**作者:** Shuai Zhen `[一作]` (Beijing University of Posts and Telecommunications), Yanhua Yu `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 15518 | [OpenAlex ID](https://openalex.org/A5100322617)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了Reflex框架，通过在状态空间中利用反射对称性来提升连续控制任务的样本效率和最终性能。

**💡 创新点**

创新点在于对轴向与双侧反射对称性进行严格的群论建模，并证明在G‑不变MDP中最优策略必定具有等变性；随后引入针对PPO、SAC乃至TD3的对称一致性正则化，使学习过程能在对称状态之间共享经验。

**🔧 技术方法**

核心技术包括群不变马尔可夫决策过程建模、反射矩阵变换、等变性正则化项、对PPO的线性衰减权重以及SAC的对称目标平均。

**📊 数据集**

使用了OpenAI Gymnasium和DeepMind Control Suite中的若干连续控制任务（如CartPole、ThreePole、IDPendulum、Walker2d、Ant、WalkerRun），其中任务被手工验证具备轴向或双侧反射对称性。

**📈 对比分析**

与标准PPO、SAC以及RAD等基线相比，Reflex在所有任务上均表现出更高的样本效率（平均提升约10‑30%）并取得更优的最终回报，尤其在双侧对称任务上效果尤为显著。

**⚠️ 局限性**

局限性包括对对称性前置假设的依赖（非对称环境难以直接使用）、对超参数（正则化权重、学习率等）的敏感性，以及实现上对网络架构和对称映射的额外编码需求。

---

## 310. When Planning Fails Despite Correct Execution: On Epistemic Calibration for LLM-Based Multi-Agent Systems

**arXiv ID:** 2605.23414 | [PDF](https://arxiv.org/pdf/2605.23414v1)

**作者:** Zehao Wang `[一作]` (Tianjin University), Lanjun Wang `[通讯]` (Tianjin University)

**通讯引用:** 2711 | [OpenAlex ID](https://openalex.org/A5025153128)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的工作流程EPC-AW，旨在解决基于大语言模型的多智能体系统在规划阶段的认知失调问题，即使在执行正确的情况下，系统仍可能因规划不当而失败。

**💡 创新点**

创新点在于将认知失调定义为规划中的一种独特失败模式，并提出了信息一致性基础的计划选择和一致性引导的认知状态细化机制，以动态调整和改善规划的可行性评估。

**🔧 技术方法**

使用了信息一致性基础的计划选择（IPS）和一致性引导的认知状态细化（CESR）技术，重点在于在不同信息条件下评估计划的稳定性，而不是直接验证可行性。

**📊 数据集**

在六个基准数据集上进行了实验，这些数据集涵盖了多种推理和搜索需求，包括Bamboogle、2Wiki、HotpotQA、Musique、GAIA和MedQA。

**📈 对比分析**

与三种基线方法（无修复、重试和回滚）进行比较，EPC-AW在所有基准上均表现最佳，系统级成功率平均提高了9.75%。

**⚠️ 局限性**

限制在于EPC-AW的性能可能依赖于所选的候选计划数量和模型架构，尽管在不同的LLM架构上表现一致，但仍需进一步研究其在更广泛应用中的适应性。

---

## 311. EquiSumm : A Gender Bias-Aware Framework for Inclusive Tweet Summarization

**arXiv ID:** 2605.23412 | [PDF](https://arxiv.org/pdf/2605.23412v1)

**作者:** Chaitanya Wanjari `[一作]` (ABV IIITM Gwalior), Roshni Chakraborty `[通讯]` (ABV IIITM Gwalior)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种考虑性别偏见的推文摘要框架 EquiSumm，先通过性别分类聚类再从每个性别群组中选取代表性推文构成平衡摘要。

**💡 创新点**

创新点在于：①利用性别词典与 spaCy NER 自动推断推文隐式指向的性别并聚类；②结合语义相似图和 LexRank 提取各性别群组中心句；③引入 Inclusion Bias Score 量化并提升性别公平性。

**🔧 技术方法**

采用的技术包括：spaCy NER、性别词典、SBERT 语义向量、聚类与中心向量匹配、余弦相似度、语义相似图构建、LexRank 句子中心度计算，以及 IBS 评估指标。

**📊 数据集**

使用的公开数据集为：#MeToo 事件推文集（485 条）和美国堕胎合法化讨论推文集（934 条），均包含明显的性别视角。

**📈 对比分析**

与 LexRank、LSA、Community+LexRank 三个基线进行对比。IBS 指标显示 EquiSumm 的性别偏差最小（-0.05），显著优于基线（+0.206 ~ +0.390），并在保持摘要长度一致的前提下实现更平衡的性别代表。

**⚠️ 局限性**

局限性包括：仅考虑男性、女性、双性和中性四类，未覆盖非二元或交叉身份；依赖词典和 NER 可能漏检；实验范围受限于两大主题，需进一步验证在更大规模、多领域数据上的鲁棒性。

---

## 312. Beyond the Half-Approximation: Fair and Efficient Online Class Matching

**arXiv ID:** 2605.23408 | [PDF](https://arxiv.org/pdf/2605.23408v1)

**作者:** Sander Borst `[一作]` (Max Planck Institute for Informatics), Max Springer `[通讯]` (Princeton University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在在线二分匹配中加入公平约束，提出阈值参数化算法（可分配的EFTT和不可分配的Hybrid Ranking），实现类间公平（CEF）与总福利（USW）的平衡。

**💡 创新点**

首次证明公平并不必然导致效率损失，给出可调参数γ实现任意常数CEF与超过1/2 USW的组合，并提供几乎匹配的价格公平上界。

**🔧 技术方法**

采用连续水填充与阈值分段的策略，结合基于潜在函数的对偶构造和原始-对偶分析，推导出USW和CEF的同时下界。

**📊 数据集**

未使用实验数据集，仅通过理论构造与证明展示性能上界与下界。

**📈 对比分析**

与以往所有已知公平算法（均不超过1/2 USW）相比，所提算法在任何γ∈(0,1)时都能取得USW>0.5且保持常数CEF；上界表明性能已接近最优。

**⚠️ 局限性**

仍存在算法与上界之间的细微差距，且不可分配情况下的CEF仅为γ/2，尚需改进逼近技术；此外结果仅适用于加性偏好与预知类信息，扩展到更一般的偏好或随机到达模型仍是开放问题。

---

## 313. RS2AD-LiDAR: End-to-End Autonomous Driving LiDAR Data Generation from Roadside Sensor Observations

**arXiv ID:** 2605.23406 | [PDF](https://arxiv.org/pdf/2605.23406v1)

**作者:** Runyi Huang `[一作]` (Tsinghua University), Keqiang Li `[通讯]` (Tsinghua University)

**通讯引用:** 16497 | [OpenAlex ID](https://openalex.org/A5031855986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了RS2AD-LiDAR框架，通过虚拟车辆侧LiDAR建模和跨视角几何对齐，实现从路侧LiDAR观测生成车辆侧LiDAR点云；

**💡 创新点**

创新点在于首次实现路侧到车侧LiDAR数据重建，结合地面场景分解、基于平面拟合的点云重采样以及跨域标签映射，提供低成本、可扩展的车辆侧数据生成方案；

**🔧 技术方法**

使用虚拟LiDAR采样、Patchwork++地面分割、几何约束点云重采样、变换矩阵对齐、欧拉角逆罗德里格斯映射等技术；

**📊 数据集**

使用自建的R2V-LiDAR数据集，包含路侧与车辆侧覆盖高度重叠的点云及注释；

**📈 对比分析**

通过语义分布比较（JS距离0.132，余弦相似度0.967）以及在BEV和3D检测任务上混合训练提升罕见类别（如行人）性能，所有主流模型均出现正向性能提升；

**⚠️ 局限性**

主要限制为路侧与车辆侧传感器的时间异步导致生成点云与实际车辆侧点云存在空间位移，且数据场景覆盖仍有限。

---

## 314. Layered construction of Message-Wise Unequal Error Protection Codes

**arXiv ID:** 2605.23390 | [PDF](https://arxiv.org/pdf/2605.23390v1)

**作者:** Qiming Lu `[一作]` (Nagoya University), Takaya Yamazato `[通讯]` (Nagoya University)

**通讯引用:** 3610 | [OpenAlex ID](https://openalex.org/A5029940452)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一种无标签的分层消息级不等误差保护（UEP）码，利用组内和组间汉明距离实现不同优先级消息的差异化纠错和可靠分类。

**💡 创新点**

创新点在于将保护结构直接嵌入码本几何距离，而非使用显式标签，并给出了组间距离的理论分类正确性证明。

**🔧 技术方法**

采用二进制码本设计、硬判决最近组解码、汉明距离约束理论以及对AWGN和VLC‑ISI信道的仿真验证。

**📊 数据集**

使用模拟信道模型（AWGN及邻近符号干扰的VLC‑ISI），并与基准的标签+BCH UEP方案对比，未使用公开数据集。

**📈 对比分析**

与基准方案相比，在45位码长下，提议方案在AWGN信道中最高优先级组SNR提升约4 dB，在VLC‑ISI信道中在多种干扰系数下保持更低误码率和更高组分类准确率。

**⚠️ 局限性**

限制在于仅实现了暴力搜索的码本构造与全搜索解码，缺乏结构化码本和低复杂度解码算法，且实验仅在短码长和模拟信道上验证，未考察实际VLC硬件实现。

---

## 315. CBANet: A Compact Attention-Based CNN-BiLSTM Network for Aggressive Driving Event Detection

**arXiv ID:** 2605.23471 | [PDF](https://arxiv.org/pdf/2605.23471v1)

**作者:** Hanadi Alhamdan `[一作]` (Princess Nourah bint Abdulrahman University), Farshad Arvin `[通讯]` (Durham University)

**通讯引用:** 3801 | [OpenAlex ID](https://openalex.org/A5005614248)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 CBANet 框架，利用多变量车辆动力学信号实现侵略性驾驶检测。

**💡 创新点**

创新点包括：1）将物理可解释的动态特征嵌入深度时序网络；2）采用 SMOTE+类别加权损失的失衡训练策略；3）引入安全导向的阈值校准决策，兼顾误报与漏报的风险。

**🔧 技术方法**

使用的技术主要有：双阶段 1D CNN + BiLSTM + 时间注意力 + 另一个 BiLSTM 的深度时序网络；SMOTE 过采样；类别加权交叉熵（或焦点损失）；AdamW 优化；以及轻量化参数设计（≈180k 参数）。

**📊 数据集**

使用的数据集为 20 名驾驶员在 Renault Koleos 2018 上收集的自然驾驶数据（4 次 40 分钟/次），通过 CAN、OBD-II、GNSS 记录 25 Hz 的多通道车辆动力学信号，并基于阈值自动标注四类事件（正常、猛烈加速、猛烈刹车、猛烈转弯）。

**📈 对比分析**

在准确率、ROC‑AUC、加权 F2 等指标上与 SVM、KNN、CNN、LSTM、GRU、GCN、GAT、RGAT 等经典及深度学习基线进行对比。CBANet 取得最高准确率 0.9585、ROC‑AUC 0.9928、加权 F2 0.9584，显著优于所有对比模型，且模型尺寸仅 0.76 MB，推理时间 1.43 ms。

**⚠️ 局限性**

局限性包括：① 依赖阈值的手工标注，可能对不同车辆或道路场景产生偏差；② 只使用单一车辆动力学模态，未结合视觉或环境传感器；③ 对极端稀缺事件的泛化仍有限，需进一步验证跨车辆、跨城市的鲁棒性。

---

## 316. IyàwóBench: A Benchmark for Evaluating Large Language Model Clinical Triage Accuracy on Undifferentiated Febrile Illness in Nigerian Primary Health Settings

**arXiv ID:** 2605.23465 | [PDF](https://arxiv.org/pdf/2605.23465v1)

**作者:** Anthonio Oladimeji Gabriel `[一作]` (Centre for Clinical Intelligence and Safety, Iyawo), Temiloluwa Aderemi `[通讯]` (Centre for Clinical Intelligence and Safety, Iyawo)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并构建了 IyàwóBench v1.0，针对尼日利亚基层医疗场景中的无差别发热疾病的 LLM 临床分诊评估基准；

**💡 创新点**

创新点在于使用真实病例统计分布生成合成病例，聚焦低资源环境中的分诊准确性和安全性，并引入安全评分指标；

**🔧 技术方法**

采用大语言模型（Claude Sonnet、Llama 4 Scout、Llama 3.3 70B、Llama 3.1 8B、Qwen 3 32B、GPT OSS 20B），并通过结构化提示、低温采样等技术进行推理；

**📊 数据集**

数据集为 200 条基于 1,200 条真实患者记录统计生成的合成病例，覆盖 8 种发热疾病，包含结构化临床字段；

**📈 对比分析**

通过对照六个 LLM 在分诊准确率和安全评分两个指标上进行比较，结果显示所有模型安全评分为 100%，但分诊准确率从 67.5%（Claude Sonnet）降至 39.0%（Llama 3.1 8B），并发现部分模型因输出不符合结构化格式导致准确率接近零；

**⚠️ 局限性**

局限包括：标签由单一作者分配，缺乏多医生验证；仅评估分诊分类，不涉及诊断、治疗建议或推理质量；只使用单一固定提示；评估仅在云端完成，未测试边缘部署；数据分布仅代表西南尼日利亚，可能缺乏对北部地区的普适性。

---

## 317. Efficient One-Step Diffusion Restoration Model with Compact Token Compression and Linear Attention

**arXiv ID:** 2605.23451 | [PDF](https://arxiv.org/pdf/2605.23451v1)

**作者:** Bingtian Qiao `[一作]` (Shanghai Jiao Tong University), Jiezhang Cao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种高效的一步实景图像超分辨率框架 SANA‑SR，能够在 32× 压缩的低维潜空间中完成超分辨率，并通过线性注意力实现一次性推理。

**💡 创新点**

创新点包括：① 将 32× 深度压缩 autoencoder 与线性注意力 Diffusion Transformer 结合，显著降低 token 数量与计算复杂度；② 引入 LoRA 微调、冻结先验对齐与适配器一致性保证一一步骤的稳定性；③ 设计基于提示的结构化剪枝策略，保持提示条件下的超分辨率性能同时压缩模型。

**🔧 技术方法**

核心技术：深度压缩 VAE、LinearDiT（线性注意力 Diffusion Transformer）、LoRA 微调、冻结先验对齐损失、适配器一致性损失、提示精炼、提示感知结构化剪枝。

**📊 数据集**

使用混合数据集：DIV2K、Flickr2K、LSDIR、FFHQ，并通过 Real‑ESRGAN 生成合成降质；在 RealSR 真实降质数据集上进行最终评估。

**📈 对比分析**

与现有多步、灵活步与高效一步方法（如 StableSR、DiffBIR、SeeSR、SinSR、OSEDiff、AdcSR、TSD‑SR 等）对比，SANA‑SR 在 DIV2K‑Val、RealSR、DRealSR 上多项指标（PSNR/SSIM/MUSIQ/MANIQA/CLIPIQA 等）均居前列；推理时间 0.019 s、MACs 407.95 G、参数 344 M，明显优于同类模型。

**⚠️ 局限性**

局限性：① 依赖于冻结的预训练先验，可能对极端或未见降质类型适应性有限；② 剪枝后仍需满足 0.35 B 预算，模型大小与速度提升受限；③ 对大尺寸图像的泛化与长序列推理（如视频）尚未验证。

---

## 318. DFSAttn: Dynamic Fine-grained Sparse Attention for Efficient Video Generation

**arXiv ID:** 2605.23445 | [PDF](https://arxiv.org/pdf/2605.23445v1)

**作者:** Jie Hu `[一作]` (Peking University), Kun Yuan `[通讯]` (Peking University)

**通讯引用:** 4342 | [OpenAlex ID](https://openalex.org/A5100614598)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出训练无关的动态细粒度稀疏注意力框架DFSAttn，显著加速视频生成模型同时保持高质量。

**💡 创新点**

创新点在于结合3D Hilbert曲线重排、分层块评分以及稀疏掩码缓存与自适应比例，实现细粒度稀疏化且GPU友好。

**🔧 技术方法**

使用Hilbert曲线重排、分层块评分、稀疏掩码缓存、自适应稀疏率、FlashAttention等技术。

**📊 数据集**

在HunyuanVideo-T2V-13B、Wan2.1-T2V-14B视频数据集（720p 129/81帧）上，以Penguin Benchmark文本提示进行实验。

**📈 对比分析**

与RadialAttention、SVG、SVG2等基线比较，DFSAttn在80%稀疏率下实现PSNR约29.38（Hunyuan）/22.37（Wan2.1），速度提升2.1×（Hunyuan）/1.8×（Wan2.1），质量优于所有基线。

**⚠️ 局限性**

局限在于需针对不同模型与分辨率调参，极高稀疏率下仍可能出现细节损失。

---

## 319. Hinge Regression Trees and HRT-Boost: Newton-Optimized Oblique Learning for Compact Tabular Models

**arXiv ID:** 2605.23422 | [PDF](https://arxiv.org/pdf/2605.23422v1)

**作者:** Hongyi Li `[一作]` (Harbin Institute of Technology), Hong Yan `[通讯]` (City University of Hong Kong)

**通讯引用:** 22898 | [OpenAlex ID](https://openalex.org/A5100644375)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了倾斜回归树（HRT）及其梯度提升扩展 HRT‑Boost，采用非线性最小二乘优化重构节点分裂，实现可逼近且结构紧凑的倾斜树；

**💡 创新点**

创新点包括：①将节点分裂建模为双线性预测器的非线性最小二乘问题并用阻尼牛顿迭代求解；②证明单节点收敛、HRT 具备 O(δ²) 逼近率与通用逼近性；③提出 HRT‑Boost，构成两级优化并给出阶段性经验风险下降保证；

**🔧 技术方法**

技术手段包括：阻尼牛顿优化、回溯线搜索、层次 ReLU 结构、线性最小二乘子问题、梯度提升框架、理论收敛与逼近率证明及实验验证；

**📊 数据集**

使用的数据集有：合成 2D/3D 函数、七个公开回归基准（Abalone、CPUact、Ailerons、CTSlice、YearPred、Concrete、Airfoil、Fried 等）以及大规模 YearPred、C&C 等；

**📈 对比分析**

在与 CART、XGBoost、LightGBM、Random Forest、TabNet、TabM 等单棵树与集成基准的交叉验证中，HRT 在单树任务上与最强基线相当或略优；HRT‑Boost 在大多数数据集上取得最低 RMSE，且模型规模（叶子数、深度）显著更小，推理 FLOPs 也更低；

**⚠️ 局限性**

局限性包括：对阻尼步长 μ 的选择仍需经验调节；在极大规模或高维稀疏特征时训练时间可能受限；理论分析主要针对平方损失，对非平方损失推广尚未展开；对极端噪声或离群点的鲁棒性待进一步研究。

---

## 320. What Linear Probes Miss: Multi-View Probing for Weight-Space Learning

**arXiv ID:** 2605.23410 | [PDF](https://arxiv.org/pdf/2605.23410v1)

**作者:** Eunwoo Heo `[一作]` (Ulsan National Institute of Science and Technology), Jaejun Yoo `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 5358 | [OpenAlex ID](https://openalex.org/A5089933293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多视角探针框架（Multi-View Probe），通过融合一阶（行列投影）和二阶（Gram矩阵）信息，实现对模型权重空间的高效识别。

**💡 创新点**

创新点在于：①理论证明一阶探针易产生模糊，二阶Gram视图能分辨不同权重矩阵；②设计四个互补分支并引入样本级标准化，使多阶特征融合平衡且可扩展；③保持与ProbeX相同的线性复杂度，实现大模型高效推理。

**🔧 技术方法**

使用可学习探针向量、Gram矩阵二阶投影、Per-sample 标准化、MLP分支、共享编码器以及多标签交叉熵损失等技术；理论分析基于矩阵秩与尺度。

**📊 数据集**

主要在Model Jungle基准集（ResNet101、SupViT、MAE、DINO）以及Stable Diffusion LoRA子集（SD_200、SD_1k）上进行实验。

**📈 对比分析**

与StatNN、ProbeGen、ProbeX等方法对比，平均提升约5%（在ResNet、DINO等上可达+5%），在LoRA任务中相较ProbeX提高约60%（如SD_1k In-Distribution 97.88% vs 35.75%），并在kNN检索和OCC任务中保持高准确率。

**⚠️ 局限性**

局限性包括：在单层探针设置下，对MAE和DINO的绝对准确率仍较低；对不同深度和架构的鲁棒性尚未完全覆盖；未来需考虑多层聚合和架构感知的分支选择。

---

## 321. TPMM-DPO: Trajectory-aware Preference-guided Model Merging for Iterative Direct Preference Optimization

**arXiv ID:** 2605.23398 | [PDF](https://arxiv.org/pdf/2605.23398v1)

**作者:** Lingling Fu `[一作]` (Guangxi University), Yongfu Xu `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在迭代 DPO 训练过程中，将历史策略模型视为轨迹，按可学习的权重进行融合，构造更稳健的参考模型；

**💡 创新点**

创新点是提出了基于轨迹的可学习权重模型融合方法，有效缓解噪声累积和后期过度优化问题；

**🔧 技术方法**

主要技术包括 DPO 目标、可学习权重参数空间合并、熵正则化以及软最大化权重约束；

**📊 数据集**

实验使用 Anthropic 的 HH 帮助/无害子集作为主要数据集，外加 UltraFeedback 作为跨域测试集；

**📈 对比分析**

与标准 Iterative DPO、sDPO、rDPO 等方法对比，TPMM-DPO 在第三轮迭代的 win‑rate 达到 68.4%（OOD 58.7%），显著优于对手；

**⚠️ 局限性**

局限在于仅在中小模型上验证，融合权重对噪声敏感且未测试更大规模或更复杂对齐场景。

---

## 322. Joint Target-Less Intrinsic and Extrinsic Camera-LiDAR Calibration using Deep Point Correspondences

**arXiv ID:** 2605.23397 | [PDF](https://arxiv.org/pdf/2605.23397v1)

**作者:** Simon Bultmann `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2669 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种完全无目标的深度点-像素对应关系与迭代非线性优化相结合的相机-激光雷达联合内外参标定方法。

**💡 创新点**

1) 兼容未知内参的深度对应学习；2) 通过结构光重建与弱先验实现内参初始化；3) 迭代耦合优化实现内外参联合收敛。

**🔧 技术方法**

深度对应学习（扩展CMRNext）、结构光重建（COLMAP）、非线性最小二乘优化、弱先验约束以及随机噪声扰动训练。

**📊 数据集**

KITTI（右相机未见）、Argoverse、Pandaset训练集，并使用原始相机图像及合成畸变增强。

**📈 对比分析**

与MDPCalib对比，平均平移误差2.27±0.09cm，旋转误差0.106±0.019°，重投影误差1.14±0.24px，性能明显优于传统方法；在不同初始化下均保持稳健。

**⚠️ 局限性**

依赖深度学习模型训练，受限于训练数据多样性；对极端畸变或光照变化的鲁棒性待验证；实时性与算力需求未在移动平台上评估。

---

## 323. Strategic Stalemates: The Paradox of Export Controls in the U.S.-China AI Race

**arXiv ID:** 2605.23475 | [PDF](https://arxiv.org/pdf/2605.23475v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 324. Convex Compositional Reasoning Models

**arXiv ID:** 2605.23395 | [PDF](https://arxiv.org/pdf/2605.23395v1)

**作者:** Meir Roketlishvili `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Arip Asadulaev `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Convex Compositional Energy Minimization (CCEM) 框架，将局部因子能量限制为输入凸网络，并在紧凑凸松弛上使用投影一阶优化，解决组合推理中的非凸陷阱。

**💡 创新点**

创新点在于通过在因子层使用输入凸网络实现全局凸能量，消除因子叠加导致的局部最优陷阱；并在两阶段训练中结合对比学习与可微投影求解器，获得确定性、可微的推理流程。

**🔧 技术方法**

使用输入凸神经网络（ICNN/PICNN）、投影梯度下降/Adam、对比学习、无卷积的因子重用、紧凑凸松弛（Birkhoff多面体、单纯形），以及可微投影求解器。

**📊 数据集**

数据集包括 N-Queens（从 8x8 训练推广至多尺寸）、图着色（COLOR 基准、Erdos‑Renyi、Holme‑Kim、随机正规爆发器、Paley 图、完全图）和 3‑SAT（单子句训练、硬相变区间）。

**📈 对比分析**

与基准方法（EBM‑Diff+PEM、GFlowNets、DIFUSCO、Fast T2T、GNN‑GCP、GAT、GCN 等）比较，CCEM 在 N‑Queens 达到 100% 正解率，在图着色任务中显著高于所有对比模型，并在不需要长扩散链或大粒子集合的前提下实现相同或更好的性能。

**⚠️ 局限性**

局限性在于强制凸性可能导致能量平坦、松弛解偏离离散最优、对多模态解空间适配不足，且在极大规模问题中仍需增大迭代次数或多起点以保证收敛。

---

## 325. Every Component is a Lookup: Token Attribution and Composition from a Single Decomposition

**arXiv ID:** 2605.23393 | [PDF](https://arxiv.org/pdf/2605.23393v1)

**作者:** Po-Kai Chen `[一作]` (Leiden University), Aske Plaat `[通讯]` (Leiden University)

**通讯引用:** 3426 | [OpenAlex ID](https://openalex.org/A5085542421)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Unpack 方法，利用 Transformer 关注层和 MLP 的统一 key‑value 结构，通过单次前向传播的反向递归，将模型计算拆解为组件交互强度、端到端路径和单词级归因。

**💡 创新点**

创新点在于：①把注意力和 MLP 统一为 ϕ(S)U 模式，能在无干预、无梯度、无辅助训练的前提下生成签名贡献路径；②通过 K/Q/V 组合分析揭示信息流路径；③用三向比较（复制 vs 复现抑制）分离复制信号与真正的 IOI 循环，首次在单前向传递中完成此类细粒度解释。

**🔧 技术方法**

技术手段包括：关键-值分解、递归反向归因、路径评分与剪枝、K/Q/V 组合标签、三向对比、跨尺度实验、以及在预归一化（pre‑norm） Transformer 上的 LayerNorm 处理。

**📊 数据集**

数据集与模型：IOI 任务 100 条随机句子提示；GPT‑2 small（124M）和 Pythia‑deduped 系列（160M、410M、1.4B、2.8B、6.9B）模型。

**📈 对比分析**

方法与已有电路结构对比：在 GPT‑2 small 上 K+Q+V 配置使 IO 在 99‑100% 提示中位居 top‑1；跨尺度实验显示 410M 及以上模型在 IO > S1、IO > S2、IO top‑1 均达 100%；S1–S2 信号抑制在所有规模中保持显著，表明方法能稳健捕获 IOI 循环。

**⚠️ 局限性**

局限性：仅适用于 pre‑norm 结构；路径数目随深度指数增长，剪枝阈值可能忽略低权路径；仅在 IOI 任务验证，未知任务下的泛化尚未评估；方法描述的是已发生的计算而非因果影响；在后归一化（post‑norm）模型和更大规模模型上需要额外改进。

---

## 326. AI Assurance: A Comprehensive Testing Strategy for Enterprise AI Systems

**arXiv ID:** 2605.23459 | [PDF](https://arxiv.org/pdf/2605.23459v1)

**作者:** Chitra Badagi `[一作]` (Thoughtworks Technologies), Adinath Shirsath `[通讯]` (Thoughtworks Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

本文提出了一种针对企业AI系统的全面保障策略，强调AI测试应关注持续风险降低而非严格的正确性验证，并将评估视为核心工程学科。

**💡 创新点**

创新点在于引入了结构化的AI失败分类法，提出了修订的五层AI保障金字塔，并提供了基于评估驱动的开发、RAG系统测试、模型生命周期管理和治理的操作指导。

**🔧 技术方法**

使用了结构化的AI失败分类法、五层AI保障金字塔以及评估驱动开发等技术。

**📊 数据集**

未具体提及使用的数据集，但强调了评估数据集的构建和维护作为核心内容。

**📈 对比分析**

与传统软件质量保证方法相比，AI系统的评估需要持续进行，不能仅依赖一次性测试。性能上，AI系统的行为是概率性的，传统的通过测试套件进行的验证方法不足以捕捉AI系统的复杂性。

**⚠️ 局限性**

限制在于AI系统的行为是概率性的，无法通过传统的确定性验证方法进行完全验证，且模型更新可能导致行为变化，增加了评估的复杂性。

---

## 327. Closing Trajectories: Equation-Free Cyclic Animation via Koopman Surrogates

**arXiv ID:** 2605.23462 | [PDF](https://arxiv.org/pdf/2605.23462v1)

**作者:** Shixun Huang `[一作]` (University of Toronto), Peter Yichen Chen `[通讯]` (University of British Columbia)

**通讯引用:** 263 | [OpenAlex ID](https://openalex.org/A5018780432)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种无方程框架，通过识别Koopman代理从观察到的轨迹中合成周期动画，应用傅里叶参数化的时间变化控制力，在硬时间周期性约束下计算周期轨迹。

**💡 创新点**

创新点在于提出了一种完全数据驱动的、无方程的周期动画合成方法，能够在没有物理模型的情况下，通过最小控制力实现周期性闭合。

**🔧 技术方法**

使用了Koopman理论和动态模式分解（DMD）来近似Koopman算子，并通过线性约束二次规划（QP）高效求解。

**📊 数据集**

使用了多种物理系统的数据集，包括N体动力学、布料模拟、可变形物体和浅水波动等。

**📈 对比分析**

与现有的插值方法和基于物理的方法相比，提出的方法在保持物理合理性的同时，能够高效地合成周期动画，性能表现优越，合成的轨迹在机器精度范围内满足闭合条件。

**⚠️ 局限性**

限制在于该方法依赖于观察到的轨迹，可能在处理更复杂的耦合系统时面临挑战，未来需要扩展到更复杂的设置中。

---

## 328. Self-Orthogonal Twisted Generalized Reed-Solomon Codes and Their Application to Quantum Error-Correcting Codes

**arXiv ID:** 2605.23460 | [PDF](https://arxiv.org/pdf/2605.23460v1)

**作者:** Yanxin Chen `[一作]` (China University of Petroleum (East China)), Tongjiang Yan `[通讯]` (China University of Petroleum (East China))

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了两类具有多重扭曲的自正交扭曲广义Reed-Solomon (TGRS) 码，建立了这些码自正交和自对偶的充分必要条件，并给出了几种自正交和自对偶码的显式构造，进一步推导出量子稳定器码。

**💡 创新点**

创新点在于提出了自正交TGRS码的构造方法，并且通过这些构造得到了自正交的MDS、AMDS和NMDS码，以及达到量子Singleton界限的量子稳定器码。

**🔧 技术方法**

使用了线性代数和编码理论中的相关技术，特别是对TGRS码的性质进行了深入分析，利用了块矩阵技术和Vandermonde矩阵的逆。

**📊 数据集**

论文中没有具体提到使用的数据集，但涉及的理论和构造方法可以应用于量子信息和编码理论的相关领域。

**📈 对比分析**

与现有方法的比较主要体现在自正交性质的构造上，性能上展示了所构造的码在量子信息保护中的有效性，尤其是达到量子Singleton界限的能力。

**⚠️ 局限性**

限制在于所构造的码的特定条件和参数限制，可能不适用于所有类型的TGRS码，且在实际应用中可能需要进一步的实验验证。

---

## 329. Class-Dependent Hybrid Data Augmentation for Multiclass Migraine Classification under Severe Class Imbalance

**arXiv ID:** 2605.23453 | [PDF](https://arxiv.org/pdf/2605.23453v1)

**作者:** Elvin Somón `[一作]`, Miguel A. Gutiérrez-Naranjo `[通讯]` (University of Seville)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对先前多类偏头痛分类研究进行可复现性重评，纠正数据泄漏和评估指标偏差，并提出基于样本大小的混合增广框架以及临床合理的类别聚合。

**💡 创新点**

创新点在于：① 将类依赖的生成器与增广量控制相结合的混合增广策略；② 引入“fidelity asymmetry”概念解释过度平衡导致的低质量合成样本问题；③ 通过比例增广（保留原比例、均衡真实样本比例）替代完全平衡，显著提升宏观F1。

**🔧 技术方法**

使用了FT‑Transformer、GANDALF、TabNet、MLP等深度表格模型；Gaussian Copula、CTGAN、SMOTE族等增广器；宏观平均F1、5折交叉验证、严格防泄漏的数据预处理。

**📊 数据集**

公开的 400 名偏头痛患者数据集（22 个临床特征，7 个子类型），在实验中将两种偏头痛子类型合并为 6 类。

**📈 对比分析**

采用层化 5‑折交叉验证，宏观平均 F1 为主要评估指标。纠正后基线为 0.71，最佳配置为 FT‑Transformer + hybrid ×2，宏观 F1 达到 0.914 ± 0.047；相比单一增广器提升约 0.08。

**⚠️ 局限性**

局限性包括：仅来自单中心的数据、缺乏外部验证、增广阈值 τ 与增长模式的经验性选择、对样本大小极端小的类仍有较大方差、fidelity asymmetry 的理论尚未正式化。

---

## 330. GeoCycler: Reward-Aligned 3D Diffusion for Constraint-Conditioned Cyclic Peptide Design

**arXiv ID:** 2605.23407 | [PDF](https://arxiv.org/pdf/2605.23407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 331. AI Security Research Should Better Incentivize Defense Research

**arXiv ID:** 2605.23448 | [PDF](https://arxiv.org/pdf/2605.23448v1)

**作者:** Youqian Zhang `[一作]` (Hong Kong Polytechnic University), Youqian Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 138 | [OpenAlex ID](https://openalex.org/A5016761774)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过量化分析16篇AI安全领域SoK论文和21篇调研论文的参考文献，揭示攻击论文远多于防御论文的结构性不平衡。

**💡 创新点**

创新点在于构建了攻击与防御研究的计量框架，并从评估标准、出版惯例、行业差距等多维度阐释导致防御研究被低估的根源。

**🔧 技术方法**

采用手工分类、出版会场归一化、标题去重以及统计分析（比例、趋势、均值等）方法，对超过1,180篇引用论文进行系统量化。

**📊 数据集**

数据集来源于16篇SoK论文和21篇调研论文的参考文献，共计644篇攻击论文和518篇防御论文，形成了独立的研究样本。

**📈 对比分析**

对比方法包括攻击对防御比率、会场类别、时间趋势等维度，结果显示攻击论文数约为防御论文的2-3倍，安全会议偏向攻击，AI会议更趋于均衡，且时间上攻击研究持续扩张。

**⚠️ 局限性**

局限性在于样本仅来自SoK与调研论文引用，未覆盖全部学术产出；缺乏对防御实际部署与转化的评估；未给出具体的激励机制改进建议。

---

## 332. Goal-Conditioned Agents that Learn Everything All at Once

**arXiv ID:** 2605.23551 | [PDF](https://arxiv.org/pdf/2605.23551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 333. Socially fluent AI decouples conversational signals from source identity in online interaction

**arXiv ID:** 2605.23426 | [PDF](https://arxiv.org/pdf/2605.23426v1)

**作者:** Lixiang Yan `[一作]` (Tsinghua University), Dragan Gašević `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在同步文本小组互动中测试人类是否能识别未披露的AI队友，发现他们的识别率接近随机，而AI与人类在对话中的语言与时间特征却存在可辨识的差异。

**💡 创新点**

首次揭示了社交性AI在不披露身份的情况下导致身份推断失效的现象，并从代表性相似性分析角度解释了信息结构与判断结果的解耦。

**🔧 技术方法**

采用信号检测理论、逻辑回归、机器学习特征提取（LIWC、文本多样性、时延指标）、BERTopic主题模型、RSA、交叉验证、AUC与Brier评分等多种计算方法。

**📊 数据集**

数据来自786名参与者、471个三人组，包含三类任务（分析、伦理、创意），每组中混合出现人工与AI对话，AI使用基于提示的对话立场（支持/反对）。

**📈 对比分析**

与传统方法相比，基于对话特征的AI识别模型在混合组内交叉验证中达到AUC≈0.98、Brier≈0.046，远优于人类判断（d'≈0.12、AUC≈0.53）。

**⚠️ 局限性**

局限性包括仅限于文本、同步、单次交互；未考虑语音、视频等多模态信息；AI代理仅为提示式实现，可能不代表更复杂的生成模型；以及样本主要为在线平台的英语用户，缺乏跨文化验证。

---

## 334. Articulatory strategy as a source of variation in acoustic vowel dynamics

**arXiv ID:** 2605.23416 | [PDF](https://arxiv.org/pdf/2605.23416v1)

**作者:** Patrycja Strycharczuk `[一作]` (University of Manchester), Sam Kirkham `[通讯]` (Lancaster University)

**通讯引用:** 539 | [OpenAlex ID](https://openalex.org/A5040366382)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研究了北英方言说话者在产生/i/元音时的舌形策略如何影响其I型二合元音的共振峰动态，利用超声舌图与声谱分析建立舌形与共振峰轨迹的关联。

**💡 创新点**

首次将舌形的主成分分析作为说话者特定舌运动策略的量化指标，并证明该策略可预测不同二合元音的共振峰动态，从而揭示个体差异在语音动力学中的器官学基础。

**🔧 技术方法**

使用Generalised Procrustes Analysis、PCA、Generalised Additive Mixed Models（GAMM）进行舌形归一化和统计建模；声学方面采用FastTrack提取共振峰并ΔF标准化；舌部特征使用DeepLabCut标注。

**📊 数据集**

TarDiS语料库：36名北英格兰（曼彻斯特与兰开夏）说话者的中矢上舌面超声图与对应音频，包含约1503个I型二合元音实例。

**📈 对比分析**

通过GAMM模型比较各主成分对F1/F2轨迹的显著性，发现PC1、PC2、PC3均对不同二合元音的共振峰形状产生统计显著影响；模型并未发现舌形对音长的影响，说明动态差异独立于时长。

**⚠️ 局限性**

局限性包括仅使用北英方言数据，未检验跨语言普适性；舌形测量仅基于中矢上平面，可能忽略三维形变；数据仅覆盖I型二合元音，结果对其他元音或语音类的推广尚需验证。

---

## 335. Precise: SDE-Consistent Stochastic Sampling for RL Post-Training of Flow-Matching Models

**arXiv ID:** 2605.23522 | [PDF](https://arxiv.org/pdf/2605.23522v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 336. Generator-Refiner-Examiner: A Tri-Module Data Augmentation Framework for 3D Human Avatar Learning from Monocular Videos

**arXiv ID:** 2605.23555 | [PDF](https://arxiv.org/pdf/2605.23555v1)

**作者:** Gangjian Zhang `[一作]` (Hong Kong University of Science and Technology), Hao Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 43076 | [OpenAlex ID](https://openalex.org/A5080102032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了三模块（Generator-Refiner-Examiner）框架，用于从单目视频中学习高质量可动画3D人类头像。

**💡 创新点**

创新点在于通过高斯分布扰动生成多样姿态/视角、单步扩散细化、以及双分支注意力判别器三种技术实现数据增强、图像质量提升和伪标签筛选。

**🔧 技术方法**

使用3D Gaussian Splatting、SMPL-X/SMPL模型、单步扩散（Diffusion）与ViT注意力机制。

**📊 数据集**

在X-Humans和NeuMan两个公开单目视频数据集上训练与评测。

**📈 对比分析**

与ExAvatar、Vid2Avatar-Pro、MonoCloth等SOTA方法对比，PSNR/SSIM/LPIPS均超过对手，最高约35.4 dB PSNR、0.987 SSIM、0.009 LPIPS。

**⚠️ 局限性**

主要局限是对极端姿态与复杂服饰仍可能产生细节失真，且需要额外的预训练与多步骤训练过程，导致计算成本较高。

---

## 337. DAE4HLS: Exposing Memory-Level Parallelism for High-Level Synthesis using Explicit Decoupling

**arXiv ID:** 2605.23549 | [PDF](https://arxiv.org/pdf/2605.23549v1)

**作者:** David Metz `[一作]` (Norwegian University of Science and Technology), Magnus Själander `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 1306 | [OpenAlex ID](https://openalex.org/A5024358692)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的高层次综合（HLS）编程模型，称为解耦访问执行（DAE），用于显式解耦内存请求和响应，以提高内存级并行性。

**💡 创新点**

创新点在于通过显式解耦来解锁内存级并行性，特别是在处理复杂访问模式和大数据集时，能够显著提高性能并减少设计和验证的工作量。

**🔧 技术方法**

使用了AMD Vitis HLS工具链和动态HLS框架，结合显式解耦的编程模型来优化内存访问。

**📊 数据集**

应用于多个具有不规则内存访问的基准测试，包括稀疏矩阵-向量乘法（SPMV）、归并排序、哈希表查找和二分查找等。

**📈 对比分析**

与传统的HLS方法相比，显式解耦的实现能够实现10到79倍的加速，尤其在处理不规则工作负载时表现出色。

**⚠️ 局限性**

限制在于并非所有内存访问模式都适合解耦，且需要程序员确保所有请求都有相应的响应，以避免潜在的死锁问题。

---

## 338. ComPose: When to Trust Hands for Object Pose Tracking

**arXiv ID:** 2605.23523 | [PDF](https://arxiv.org/pdf/2605.23523v1)

**作者:** Jisu Shin `[一作]` (Gist), Hae-Gon Jeon `[通讯]` (Yonsei University)

**通讯引用:** 3042 | [OpenAlex ID](https://openalex.org/A5041516963)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于RGB视频的手持物体6DoF姿态跟踪框架ComPose，融合对象几何信息与手部运动信息，实现高精度、时序一致的物体轨迹预测。

**💡 创新点**

创新点包括：①自适应融合机制——通过学习手部关节权重和混合系数α动态决定对象与手部信息的比例；②使用基础模型提供的稠密3D几何作为对象观测，避免依赖CAD模型；③引入时间一致性约束与周期性锚定，显著抑制累计漂移。

**🔧 技术方法**

采用的技术包括：Flow3R等3D基础模型获取稠密点云；WiLoR手部姿态估计；SAM3分割对象；DINOv2特征提取；ICP求对象相对旋转；加权Procrustes求手部相对旋转；球面线性插值融合两种旋转；时间一致性与平滑损失；锚定策略及Softbound正则。

**📊 数据集**

使用的数据集：DexYCB S3、HOI4D、HO-3D v2、OakInk-v1，涵盖多视角、亲眼视角、长序列和现实机器人场景。

**📈 对比分析**

与MegaPose、FoundPose、FreePose、UniHOPE等方法对比，主要指标为RRE、RTE、ARE、ATE、TCC_R/TCC_T；实验表明ComPose在相对误差、时序相关性方面始终保持领先或竞争力，且在手遮挡严重或几何模糊时鲁棒性更佳；推理速度快，无需渲染或外部模型。

**⚠️ 局限性**

局限性包括：当对象几何和手部信息均失效（如极端遮挡、运动模糊、图像质量低下）时性能下降；对未知抓取方式的适应性有限；长序列仍存在残余累计漂移，需锚定补偿；对手部与对象接触模式的显式建模尚未覆盖。

---

## 339. Probabilistically checkable proofs for the Existential Theory of the Reals

**arXiv ID:** 2605.23517 | [PDF](https://arxiv.org/pdf/2605.23517v1)

**作者:** Jack Stade `[一作]` `[通讯]`, Jack Stade

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了实数的存在性理论的PCP定理，表明MAX-ETR-INV在某个常数因子内是∃ℝ-难以近似的。

**💡 创新点**

提出了MAX-ETR-INV的∃ℝ-难度近似结果，并给出了多项式时间的8因子近似算法和非确定性多项式时间的2因子近似算法。

**🔧 技术方法**

使用了概率可检查证明（PCP）理论和线性代数技术，结合了对约束满足问题的分析。

**📊 数据集**

使用了包含多种约束形式的自定义数据集，特别是约束形式为x=1, xy=1和x+y=z的约束。

**📈 对比分析**

通过与已知的NP难度问题（如MAX-3SAT）进行多项式时间的归约，展示了MAX-ETR-INV的近似难度，性能表明在某些条件下无法在常数因子内近似。

**⚠️ 局限性**

限制在于尽管提供了近似算法，但仍然存在对MAX-ETR-INV的近似难度的理论限制，特别是在∃ℝ与NP之间的关系尚未明确的情况下。

---

## 340. Multimodal Distribution Matching for Vision-Language Dataset Distillation

**arXiv ID:** 2605.23482 | [PDF](https://arxiv.org/pdf/2605.23482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 341. Tracking a Decade of Research at the University of Nigeria, Nsukka: A Scientometric Analysis (2014-2023)

**arXiv ID:** 2605.23586 | [PDF](https://arxiv.org/pdf/2605.23586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 342. An Exact Cooperation Formula for Introspection Dynamics in the Heterogeneous Public Goods Game

**arXiv ID:** 2605.23513 | [PDF](https://arxiv.org/pdf/2605.23513v1)

**作者:** Harry Foster `[一作]` (Cardiff University), Sebastian Krapohl `[通讯]` (University of Amsterdam)

**通讯引用:** 644 | [OpenAlex ID](https://openalex.org/A5091459195)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在异质公共物品游戏中研究自我反省动力学，得到精确的长期合作概率。

**💡 创新点**

创新点在于引入“状态无关性”，使得链可分解为独立两态链，得到闭式产品测度公式。

**🔧 技术方法**

使用自我反省动力学、Fermi 采样、Markov 链分解与线性代数求解。

**📊 数据集**

通过模拟和小规模 2^N 线性系统验证，主要使用异质公共物品游戏参数集。

**📈 对比分析**

与完整 2^N 系统以及随机模拟相比，闭式公式在计算上从指数级降低到线性级别，且与数值结果完全吻合。

**⚠️ 局限性**

局限在于仅适用于状态无关性游戏，无法处理如狩猎-捕食者等存在交叉项的非线性游戏。

---

## 343. Learning partially observed systems with neural Hamiltonian ordinary differential equations

**arXiv ID:** 2605.23510 | [PDF](https://arxiv.org/pdf/2605.23510v1)

**作者:** Sunniva Meltzer `[一作]` (SINTEF Digital), Alexander Johannes Stasik `[通讯]` (SINTEF Digital)

**通讯引用:** 12529 | [OpenAlex ID](https://openalex.org/A5066901898)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种神经哈密顿常微分方程（NHODE）框架，能够从部分观测数据中学习完整的哈密顿动力学系统，自动推断不可观测的隐藏状态并实现长期稳定预测。

**💡 创新点**

创新点在于将哈密顿神经网络的能量守恒结构与神经 ODE 的 roll‑out 训练相结合，允许仅在观测变量上计算损失；并通过对称性坐标变换和可分离能量表达式进一步嵌入物理先验，显著提升了在未观测变量存在时的可学习性与预测稳定性。

**🔧 技术方法**

使用技术包括：Hamiltonian 神经网络、神经 ODE 训练（自回归积分、RK5 适应步长求解）、相对坐标变换以实现平移/旋转对称、可分离动能与势能的约束、用于推断初始隐藏状态的编码器网络、Adam 优化器以及基于 diffrax 的数值求解器。

**📊 数据集**

实验数据集为模拟生成的三类系统：线性一维两质量两弹簧、二维三质量三弹簧（三角形）以及三体引力问题，均采用随机初始条件并在不同维度和非线性程度下进行训练和测试。

**📈 对比分析**

与两种基线模型（普通神经 ODE 与仅学习动量导数的物理化神经 ODE）进行对比。NHODE 在所有系统上都取得了更低的观测/隐藏变量均方误差，特别是在三体问题中表现出明显的长期稳定性；相较基线模型，NHODE 能保持能量守恒并避免轨道发散。

**⚠️ 局限性**

局限性包括：在缺少额外参数或偶发观测的情况下，隐藏状态的恢复并非唯一，可能不等于真实值；目前仅适用于可写成标准哈密顿形式的系统；未在带噪声或稀疏采样的真实数据上验证；在更大规模系统上训练成本较高。

---

## 344. Reducing the Randomness in Partition Oracles for Bounded Degree Minor-Free Graphs

**arXiv ID:** 2605.23509 | [PDF](https://arxiv.org/pdf/2605.23509v1)

**作者:** Akash Kumar `[一作]` (IIT Bombay), C. Seshadhri `[通讯]` (University of California, Santa Cruz)

**通讯引用:** 3884 | [OpenAlex ID](https://openalex.org/A5038687804)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种新的随机性压缩技术，使得在所有少子图（例如平面图）类上实现的划分算子（partition oracle）只需要O(log n)位随机种子即可完成，而原来所有实现都需要线性（Ω(n)）位随机种子；同时给出了一个下界，证明在标签来自大范围[ N ]的模型下，即使是循环图也需要超常量（ω(1)）的随机位。

**💡 创新点**

核心创新在于：①首次证明划分算子可以用极小随机种子实现；②通过对KSS划分算法的细致分析，证明仅需O(1)-wise独立的随机序列；③使用Ramsey理论构造对所有划分算法都适用的“比较基”结构，从而得到随机位下界；④结合有限独立性Chernoff界，完成完整的上界与下界论证。

**🔧 技术方法**

技术手段主要包括：b‑wise独立随机源与伪随机生成器；对KSS算法的局部化与重写以适应有限独立性；截断扩散（truncated diffusion）与低导电子集搜索；有限独立性的Chernoff/Hoeffding界；Ramsey理论用于提炼“比较基”决策树；以及与先前划分算法的全局-局部对应关系的精细分析。

**📊 数据集**

本工作为理论研究，未使用任何具体数据集，所有结论均为理论性质与概率分析。

**📈 对比分析**

相较于原始划分算子，查询复杂度保持常数不变，随机种子长度从Ω(n)下降到O(log n)，实现了对随机性资源的显著节约；下界表明不能进一步降低到常量级。

**⚠️ 局限性**

限制主要包括：下界仅在标签来自大范围[ N ]的模型下成立，未证明对常规模型[ n ]有相同限制；目前尚未给出可实现更低随机位的具体构造；并且虽然实现保持了常数查询，但整体算法的时间与空间仍受限于原算法的结构。

---

## 345. CP or DP? Why Not Both: A Case Study in the Partial Shop Scheduling Problem

**arXiv ID:** 2605.23569 | [PDF](https://arxiv.org/pdf/2605.23569v1)

**作者:** Emma Legrand `[一作]` (University of Louvain), Pierre Schaus `[通讯]` (University of Louvain)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将动态规划与约束规划相结合，提出一种 DP‑CP 混合框架，用以求解部分工厂调度（PSSP）问题，包括作业车间（JSP）和开放车间（OSP）实例。

**💡 创新点**

创新点在于：① 在 DP 的状态转换中调用 CP 的全局约束传播（NoOverlap）以自动发现并加入新的先后关系，显著压缩搜索空间；② 采用 Anytime Column Search 取代传统层级搜索，并利用 CP 计算更强的下界；③ 进一步设计了基于 LNS 的自适应搜索，在已知解的基础上通过松弛部分先后关系进行局部搜索。

**🔧 技术方法**

使用的技术包括：动态规划（DP）、约束规划（CP）与 MaxiCP 解决器、NoOverlap 全局约束、Jackson 预抢占式下界、二分搜索下界、Anytime Column Search（ACS）和 Large Neighborhood Search（LNS）等。

**📊 数据集**

实验数据集包含三套 JSP 基准（Fisher&Thompson、Lawrence、Applegate&Cook）和三套 OSP 基准（Taillard、Guéret&Prins、Brucker 等），共计六个 benchmark 套件。

**📈 对比分析**

通过比较节点数和求解时间与单独的 DP（使用 JPS 下界）以及纯 CP（setTimes、Rank 搜索）进行对比；结果显示 DP‑CP 在大多数实例显著减少探索节点数，尤其在难例上提升了求解效率；LNS 在大规模实例进一步缩小最佳性间隙，证明了混合模型的实际性能。

**⚠️ 局限性**

局限性包括：在 OSP 实例上，纯 CP 仍优于 DP‑CP；混合模型在某些小规模实例求解时间略高；缺乏针对更大规模或更复杂约束的合并算子，需要进一步研究其在其它组合优化问题中的可迁移性。

---

## 346. Is Dimensionality a Barrier for Retrieval Models?

**arXiv ID:** 2605.23556 | [PDF](https://arxiv.org/pdf/2605.23556v1)

**作者:** Kiril Bangachev `[一作]` (Massachusetts Institute of Technology), Yury Polyanskiy `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 9248 | [OpenAlex ID](https://openalex.org/A5031031216)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了低维表示（通常为d≈1000）如何在现代嵌入基础的检索模型中扩展到数十亿甚至数万亿的数据点，重点分析了最大边际嵌入的特性。

**💡 创新点**

提出了在不限制维度的情况下，几乎可以实现最佳边际的理论结果，表明低维嵌入模型在检索性能上并不需要高维度。

**🔧 技术方法**

使用了最大边际嵌入理论，结合压缩感知和限制等距性质的文献，进行了理论分析和证明。

**📊 数据集**

使用了包含N个查询和n个文档的二元矩阵A∈{0,1}^N×n，研究了查询与文档之间的相关性。

**📈 对比分析**

通过与现有理论结果的比较，证明了在特定条件下，最大边际可以在较小的维度中实现，性能表现优于传统方法，尤其是使用sigmoid损失时。

**⚠️ 局限性**

在d=o(klog(n/k))的情况下，当前的下界和上界不匹配，且实验是在自由嵌入模型中进行，而非实际数据集和架构上。

---

## 347. MARS: Magnitude-Aware Rank Statistics

**arXiv ID:** 2605.23563 | [PDF](https://arxiv.org/pdf/2605.23563v1)

**作者:** Muhammad Rajabinasab `[一作]` (University of Southern Denmark), Arthur Zimek `[通讯]` (University of Southern Denmark)

**通讯引用:** 11828 | [OpenAlex ID](https://openalex.org/A5047196019)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的基于大小感知的排名统计方法MARS，用以改进传统Critical Difference（CD）图在模型评估中的失真问题。

**💡 创新点**

创新点在于：①将模型性能差距量化为权重，构建连续的大小感知排名分数；②动态调节CD阈值，基于权重方差而非固定整数差距；③采用非参数置换检验和Wilcoxon-Holm方法，提升统计显著性判定的精度。

**🔧 技术方法**

技术手段包括：相对差距权重计算、加权平均排名、MARS特定的CD公式、置换检验、Wilcoxon签名秩检验与Holm多重检验、以及Python实现的MARS库。

**📊 数据集**

实验数据涵盖六个合成情景（包含巨大差距、稳健性、波动性、噪声优势、边缘案例和真实噪声竞赛）以及一组八个模型在40个带噪声数据集上的真实竞赛模拟。

**📈 对比分析**

与传统Friedman‑Wilcoxon‑Holm与Nemenyi CD图比较，MARS能够更细致地区分模型优劣，尤其在性能差距极大或极小的场景下提供更合理的统计显著性判定；实验结果显示MARS在六个情景中多次纠正了传统方法误判的结论。

**⚠️ 局限性**

局限性包括：①对数据集选择依赖较大，MARS仍需手动挑选具有代表性和可比性的基准集；②在样本极少或噪声极大时，置换检验可能缺乏统计功效；③MARS对权重计算的参数（如Δw_max）需要经验选择，可能影响结果稳定性。

---

## 348. ARMS: Automatic Reward Shaping for Sparse-Reward Multi-Agent Reinforcement Learning

**arXiv ID:** 2605.23562 | [PDF](https://arxiv.org/pdf/2605.23562v1)

**作者:** Elie Abboud `[一作]` (University of Haifa), Oren Gal `[通讯]` (University of Haifa)

**通讯引用:** 348 | [OpenAlex ID](https://openalex.org/A5030616543)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

提出了 ARMS（Automatic Reward‑shaping in Multi‑agent Systems）框架，在多智能体强化学习（MARL）中通过轨迹排名自监督学习稠密奖励，并在每个训练周期交替进行策略学习与奖励学习。

**💡 创新点**

创新点包括：① 基于条件最优反应推理的博弈论均衡保持理论，证明在满足某些条件下替换奖励仍保留各智能体的最佳反应集与纳什均衡；② 将该理论应用于 MARL 的自动奖励塑造；③ 在实现中共享奖励网络参数以提升样本效率并降低规模；④ 发现并通过提升探索率消除因有限探索与奖励-策略耦合导致的振荡失稳现象。

**🔧 技术方法**

使用技术：IPPO、MAPPO 等 MARL 基础算法；轨迹对排名的二分类损失与软最大概率；潜在基奖励塑造（PBRS）对照；基于 POGEMA 的部分可观测网格世界；奖励网络由 ResNet‑MLP 结构组成；全局共享策略网络；探索系数 α 的调参。

**📊 数据集**

数据集：20×20 网格地图（30% 障碍），从 POGEMA 生成；训练使用单一地图，评估使用 50 张未见地图（40 张随机 + 10 条迷宫样本）。

**📈 对比分析**

方法比较：在 8/16/32 代理、10/20/30 步稀疏奖励、以及无奖励塑造与 PBRS 作为基线进行对比。实验结果显示：在稠密奖励下 ARMS 与基线无明显差异；在稀疏奖励下，ARMS 在累计奖励、通过率（throughput）与碰撞次数上均优于 PBRS 与无塑造，尤其在代理数增大或奖励稀疏度加深时差距更明显。提升探索系数可缓解子最优收敛问题并进一步提升性能。

**⚠️ 局限性**

局限性：① 目前仅实现共享策略与共享奖励网络，缺乏针对不同智能体的个性化奖励；② 奖励函数仅基于当下观察与动作，未考虑历史信息；③ 对于低探索率，易出现振荡或子最优收敛，需要手动调节探索；④ 仅在网格路径规划任务上验证，缺乏在更广泛多智能体任务上的通用性与鲁棒性评估。

---

## 349. JEDI: Java Evaluation of Declarative and Imperative Queries

**arXiv ID:** 2605.23543 | [PDF](https://arxiv.org/pdf/2605.23543v1)

**作者:** Filippo Schiavio `[一作]` (Università della Svizzera italiana), Walter Binder `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 3557 | [OpenAlex ID](https://openalex.org/A5074152163)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将 SQL 查询自动转换为 Java Stream API 代码，构建了 JEDI 基准套件，评估了不同实现（包括顺序、并行、不同聚合策略）的性能，并给出优化建议。

**💡 创新点**

创新点在于：①自动生成多种语义等价的 Stream 实现；②系统分析并比较多种并行化策略（P、PU、CG、CGCC）；③同时提供等价的命令式实现，量化 Stream API 的开销；④对代码复杂度指标进行对比。

**🔧 技术方法**

技术手段包括：Java Stream API、改进的 S2S 代码生成器、JMH 基准框架、Oracle JDK 与 GraalVM 运行时、针对不同硬件（Intel Xeon、ARM Neoverse）的实验。

**📊 数据集**

使用了行业标准的 TPC‑H 数据库基准，规模因实验而异（1 GB、10 GB 等），并根据查询规模生成相应的 Java 代码。

**📈 对比分析**

通过 JMH 进行 5 次热身 + 5 次测量，比较顺序 Stream、并行 Stream 与等价的 imperative 循环实现；结果显示：imperative 实现平均比 Stream 快 1.3‑1.4 倍；在并行化方面，CG（使用 concurrent groupingBy）在大多数聚合场景下最快，PU 在 distinct 等小数据集上表现最好。

**⚠️ 局限性**

局限性包括：①仅覆盖 SQL 与 Stream 语义对应的操作，未涵盖所有 Stream API 方法；②数据源仅为数组，未评估列表/集合等 Spliterator；③并行化策略在查询级别统一，未对单个流单独选择；④性能受机器与 JIT 优化策略影响。

---

## 350. VINS-120K: Ultra High-Resolution Image Editing with A Large-Scale Dataset

**arXiv ID:** 2605.23518 | [PDF](https://arxiv.org/pdf/2605.23518v1)

**作者:** Zhizhou Chen `[一作]` (Nanjing University), Ying Tai `[通讯]` (Nanjing University)

**通讯引用:** 13584 | [OpenAlex ID](https://openalex.org/A5029021362)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了首个大规模高质量的UHR图像编辑数据集VINS-120K（120K+4K分辨率编辑三元组），并提出了高频感知后适配策略，使预训练的非高分辨率模型能够直接在4K图像上进行指令跟随与细节合成。

**💡 创新点**

创新点在于：①首个覆盖13种编辑类型、120K+4K+ triplets的UHR编辑数据集；②通过注意力温度重标定、RoPE缩放以及频率聚焦监督实现的后适配，显著恢复高频纹理并保持指令遵循。

**🔧 技术方法**

技术包括多阶段数据过滤（CLIP相似度+光流、结构清晰度、曝光平衡、颜色真实性、纹理丰富度），Gemini-2.5-Pro生成指令，LoRA微调（rank 32），注意力与RoPE的分辨率感知重标定，以及频率聚焦损失（Fourier权重）。

**📊 数据集**

主要使用自制的VINS-120K数据集；评测基准为VINS-4KEval；与AnyEdit、ICEdit、OmniGen2、Bagel、Step1X-Edit、Seedream 4.0等公开模型进行对比。

**📈 对比分析**

采用ImageJudge、VIEScore和patch‑FID评价，后适配模型在编辑性能上与基线相当，但pFID从12.66降至9.15，显示纹理保真度显著提升。

**⚠️ 局限性**

局限在于对算力要求高，难以部署在资源受限设备；扩展到8K/16K时序列过长导致注意力与RoPE失效，需要更高效的机制。

---

## 351. MDS-DETR: DETR with Masked Duplicate Suppressor

**arXiv ID:** 2605.23507 | [PDF](https://arxiv.org/pdf/2605.23507v1)

**作者:** Chanho Lee `[一作]` (Samsung Research), Junmo Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 9614 | [OpenAlex ID](https://openalex.org/A5100606266)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的目标检测模型MDS-DETR，该模型结合了一对一和一对多的监督机制，通过引入Masked Duplicate Suppressor (MDS)来提高检测性能。

**💡 创新点**

创新点在于MDS通过自注意力机制实现了重复抑制，避免了使用额外的解码器和查询，从而降低了训练成本并提高了模型的效率。

**🔧 技术方法**

使用了自注意力机制和基于置信度的因果掩蔽技术来实现重复抑制。

**📊 数据集**

在MS COCO数据集上进行实验，使用ResNet-50作为骨干网络。

**📈 对比分析**

与现有的一对多DETR变体（如MS-DETR、MR.DETR和Relation-DETR）进行比较，MDS-DETR在12个训练周期内实现了+2.8 mAP的提升，并且训练速度比MR.DETR快20%。

**⚠️ 局限性**

限制在于尽管MDS-DETR在性能上有所提升，但在集成其他技术（如查询去噪或混合查询选择）时可能会面临结构上的挑战。

---

## 352. EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation

**arXiv ID:** 2605.23493 | [PDF](https://arxiv.org/pdf/2605.23493v1)

**作者:** Aristotelis Lazaridis `[一作]` (EdgeRunner AI), Jack FitzGerald `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种改进的自监督对齐方法 EDGE-OPD，用来在推理时缺失的上下文信息中内化稀有身份或事实。

**💡 创新点**

创新点在于两项技术：① 通过在采样时注入特权上下文的“引导回放”解决稀有行为的支持瓶颈；② 采用“证据掩码”仅在特权上下文提升令牌概率的位置更新模型，从而避免“泄漏”与退化。

**🔧 技术方法**

采用 On‑Policy Distillation (OPD) 的自监督变体，结合 KL 对齐、正负证据掩码、guided rollouts、以及可选的 KL anchor 作为稳定器。

**📊 数据集**

使用 Nemotron‑3‑Nano‑4B 作为模型，在两个轴向实验中：身份/个性轴使用简短的身份段落作为特权上下文；数学轴使用过滤后的 AIME 题目的推理轨迹及答案作为特权上下文。

**📈 对比分析**

对比未引导 OPSD、RLSD‑no‑verifier、guided OPSD 以及 EDGE‑OPD 等方法。结果显示：所有引导变体均能快速内化目标身份，且 AIME‑25 表现基本保持或略有提升；正证据掩码在身份轴表现最好，但在数学轴导致性能下降；负/零掩码保持基线性能，但无法内化身份。

**⚠️ 局限性**

局限性：正证据掩码对包含答案的推理轨迹不适用，可能误捕获问题特定的答案或过早承诺；方法对其他领域（如代码、专业风格）是否同样有效仍需验证。

---

## 353. How Many Training Samples Are Needed for the Inverse Kinematics Solutions by Artificial Neural Networks

**arXiv ID:** 2605.23583 | [PDF](https://arxiv.org/pdf/2605.23583v1)

**作者:** Dong-Won Lim `[一作]` (University of Suwon), Dong-Won Lim `[通讯]` (University of Suwon)

**通讯引用:** 6910 | [OpenAlex ID](https://openalex.org/A5100405096)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了使用人工神经网络近似机器人机械臂的逆运动学，并探讨训练样本数量与模型精度的关系。

**💡 创新点**

创新点在于提出了基于Lipschitz连续性和样本数的理论误差上界，并提出了误差/采样间距比指标，用以指导样本量选择。

**🔧 技术方法**

采用了前馈人工神经网络（ReLU激活），使用TensorFlow 2.11.0训练，并结合统计学习理论和近似理论进行分析。

**📊 数据集**

数据集为从3-DOF机械臂的运动学方程生成的端执行器位姿与关节角对的坐标，采样数从8到512不等。

**📈 对比分析**

通过与解析逆运动学以及不同样本数下的误差比较，实验表明误差随样本数增大而下降，但在64样本后趋于饱和，误差/间距比约0.14。

**⚠️ 局限性**

局限性包括仅验证了单一3-DOF机械臂和固定网络结构，理论误差上界对更深网络或高维输入的适用性尚未验证。

---

## 354. Multi-Factor Trust-Driven Secure Communication Model for Cloud-Based Digital Twins

**arXiv ID:** 2605.23566 | [PDF](https://arxiv.org/pdf/2605.23566v1)

**作者:** Deepika Saxena `[一作]` (University of Aizu), Ashutosh Kumar Singh `[通讯]` (Indian Institute of Information Technology Bhopal)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种多因子信任驱动的安全通信框架 MT‑SeCom，用于云端数字孪生网络中的可信协作，涵盖信任监测、适应性评估、Transformer 级分类以及鲁棒通信管理。

**💡 创新点**

创新点包括：①将时间、上下文和联邦信任三维度联合建模；②通过自适应权重动态聚合信任；③利用 Transformer 对多维信任序列进行异常+监督联合分类；④采用强化学习在可信节点子图上进行低延迟、可靠的路由优化。

**🔧 技术方法**

主要技术：多因子信任监测（指数衰减、上下文加权、鲁棒联邦聚合），Transformer 级分类器（自注意力、异常检测融合），强化学习驱动的鲁棒通信管理，增量式在线更新和图论优化。

**📊 数据集**

数据集：合成的 VM 级时间序列（CPU、内存利用率），通过攻击注入脚本标注正常/异常，使用 70/15/15 划分训练/验证/测试，样本不重叠。

**📈 对比分析**

对比方法：PCA、Isolation Forest、OCSVM、LSTM‑AE、Seq2Seq 等经典异常检测与深度模型。实验结果显示 MT‑SeCom 在 5%–90% 攻击率下平均准确率 94%，F1 0.823，ROC‑AUC>0.95，异常发生率比基线低 24.3%，平均提升 18.7% 的威胁检测准确率。

**⚠️ 局限性**

局限性：仅在合成环境下验证，攻击模型相对简单（随机注入）；需要真实云端数字孪生部署验证；Transformer 和 RL 训练需要标注数据与计算资源；在极大规模网络下的实时性能与能源消耗仍待深入评估。

---

## 355. DrawVideo: Generating Long Video from Storyboard Keyframe Sketches

**arXiv ID:** 2605.23508 | [PDF](https://arxiv.org/pdf/2605.23508v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 356. Sea Trial Validation of the ROS-DESERT Middleware with Autonomous Underwater Vehicles

**arXiv ID:** 2605.23553 | [PDF](https://arxiv.org/pdf/2605.23553v1)

**作者:** Davide Cosimo `[一作]` (University of Pisa), Michele Zorzi `[通讯]` (University of Padua)

**通讯引用:** 40002 | [OpenAlex ID](https://openalex.org/A5005894115)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在真实海域进行海上试验，验证了ROS-DESERT中间件与AUV的深度自适应策略，展示了该架构在多AUV环境中的可行性与性能。

**💡 创新点**

创新点包括：①通过rmw_desert桥接ROS2与DESERT通信栈，实现对声学网络的全栈控制；②使用单一AUV的CTD传感器实时估算声速剖面，从而指导跟随AUV在最优传播层（sofar duct）重定位；③实现ROS1/ROS2的双向互操作，支持现有AUV平台无需大改造。

**🔧 技术方法**

所用技术：ROS2、rmw_desert、中间件接口、CBOR序列化、DESERT Underwater协议栈、TDMA、NS-Miracle、IP/UDP、Evologics声学调制器、Idronaut CTD、Bellhop/TL仿真、WOSS传播模型。

**📊 数据集**

使用的数据集：现场收集的CTD声速剖面（Ssp）、声学包的接收日志（100包/阶段），以及仿真得到的Ssp、传输损失（TL）和PER分布。

**📈 对比分析**

通过对比随机深度基线与优化深度两轮100包的Packet Reception Rate（PRR）和Packet Error Rate（PER）进行评估；结果显示在≈1 km以上距离时，优化深度将PER降低>95%（PRR显著提升），而在较短距离时两者差异不明显；平均端到端延迟约1.5 s。

**⚠️ 局限性**

局限性：深度优化仅在允许垂直机动且距较大时有效；实验范围受Wi‑Fi和船舶安全限制，未覆盖更远距离或极端环境；触发和重定位需要人工干预，未量化能耗与时间成本；航向/传感器不受控制，导致信号波动。

---

## 357. When One Point Is Not Enough: Addressing Ambiguous Instances in Dimensionality Reduction by Splitting

**arXiv ID:** 2605.23540 | [PDF](https://arxiv.org/pdf/2605.23540v1)

**作者:** Diede P. M. van der Hoorn `[一作]`, Fernando V. Paulovich `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 3830 | [OpenAlex ID](https://openalex.org/A5009099532)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了高维数据中存在的模糊实例导致的“部分邻域嵌入”失真，并提出了通过图论方法识别并拆分这些实例的流程。

**💡 创新点**

创新点在于将“模糊实例”与图的局部割点相结合，利用谱稀疏化后局部割点检测实现实例拆分，从而在单一投影中显式表达多重邻域关系。

**🔧 技术方法**

采用谱稀疏化（Spielman‑Srivastava）、局部割点检测、图拆分以及基于图的映射（如UMAP、ForceAtlas2）等技术。

**📊 数据集**

在MNIST、SVHN、论文语料（CBR/ILP/IR/SON）以及单细胞 RNA‑seq（Han dataset）等数据集上进行实验。

**📈 对比分析**

与传统的UMAP投影比较，使用邻居保持率、信任度、连续度等指标显示两者差异，但这些指标无法捕捉模糊实例；我们的方法在可视化中显著展示了被忽视的多邻域关系，量化评估显示稀疏化后邻居保持率保持在85%+，并揭示了若干被误判的聚类。

**⚠️ 局限性**

主要限制包括需要对图进行稀疏化且参数（ε、r、τ_w）对结果敏感；仅识别单个实例的模糊性，无法处理整个邻域的模糊；映射阶段可能因拆分导致布局偏移，且现有质量指标无法评价拆分后的投影。

---

## 358. PixIE: Prompted Pixel-Space Low-Light Image Enhancement

**arXiv ID:** 2605.23531 | [PDF](https://arxiv.org/pdf/2605.23531v1)

**作者:** Ruirui Lin `[一作]` (University of Bristol), Nantheera Anantrasirichai `[通讯]` (University of Bristol)

**通讯引用:** 2631 | [OpenAlex ID](https://openalex.org/A5021717616)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为PixIE的前馈像素空间低光图像增强框架，通过视觉基础模型进行语义提示，解决低光图像中的噪声、对比度丧失和细节恢复问题。

**💡 创新点**

创新点在于将基础模型的语义信息注入到像素空间的增强过程中，采用空间连续的每像素调制，提升了细节恢复的质量和鲁棒性。

**🔧 技术方法**

使用了DINOv3作为基础模型，通过DINO提示像素块（DPPB）和多感受野像素嵌入（MRPE）等技术实现像素级的语义调制。

**📊 数据集**

使用了LOLv1、LOLv2（包含真实捕获和合成子集）以及LSRW等数据集进行训练和评估。

**📈 对比分析**

与最新的其他方法进行比较，PixIE在PSNR上提高了1.9-15.0%，在LPIPS上减少了8.5-44.4%。定性比较显示PixIE在细节恢复和纹理稳定性方面表现优越。

**⚠️ 局限性**

局限性在于模型的计算复杂度较高，尤其是在高分辨率下，尽管采用了空间通道压缩（SCC）来降低计算成本。

---

## 359. VACE: Learning Geometrically Structured Representations for Time Series Anomaly Detection

**arXiv ID:** 2605.23504 | [PDF](https://arxiv.org/pdf/2605.23504v1)

**作者:** Alberto D. Cencillo `[一作]` (Andalusian Research Institute in Data Science and Computational Intelligence), Julián Luengo `[通讯]` (University of Granada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自监督的多变量时间序列异常检测方法 VACE，通过对正常序列的速度一致性预训练，得到一个局部平滑且方向一致的嵌入轨迹，随后用 Mahalanobis 距离和速度方向一致性得分的乘积来检测异常。

**💡 创新点**

创新点在于：① 将轨迹的方向一致性直接作为预训练目标（velocity‑consistency loss），不再依赖传统的对比学习负样本；② 设计了 channel‑aware 的深度可分离卷积编码器，既保留每个通道的局部特征，又能在后期融合跨通道信息；③ 将位置和方向两种异常信号在得分阶段进行乘法组合，使得仅当两者同时偏离时才被判为异常，从而提升定位精度。

**🔧 技术方法**

主要技术包括：深度可分离卷积编码器、velocity‑consistency 自监督预训练损失、Mahalanobis 位置得分、基于方向一致性的速度银行得分、以及多尺度滑窗嵌入。

**📊 数据集**

使用 TSB-AD‑M benchmark（200 条多变量时间序列，涵盖 NASA 轨道、服务器监控、医疗波形等 17 类），对 180 条评估序列进行测试，并在 20 条调优序列上调参。

**📈 对比分析**

与 14 种基线（深度学习、Transformer、传统方法和自监督方法）比较，VACE 在 VUS‑PR、AUC‑PR、Range‑F1 上名列前茅，特别在稀疏异常场景下保持较大优势；在 VUS‑ROC、AUC‑ROC 上与 PaAno 相当。

**⚠️ 局限性**

局限性包括：① 仅使用正常数据训练，若异常模式与训练分布差异较大时效果可能下降；② 对轨迹平滑与方向一致性的假设在某些具有剧烈非线性变化的真实场景下可能不成立；③ 仍需手动设置滑窗长度、速度窗口等超参数，缺乏端到端自动化。

---

## 360. B-GRTO: Bootstrapped Group Relative Tool Optimization for Referring Segmentation

**arXiv ID:** 2605.23500 | [PDF](https://arxiv.org/pdf/2605.23500v1)

**作者:** Mario Markov `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Danda Pani Paudel `[通讯]` (INSAIT, Sofia University St. Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了B-GRTO框架，将视觉语言模型与分割工具联合优化，用于解决需要推理和分割的参考分割任务。

**💡 创新点**

创新点在于：①通过Group Relative Tool Optimization（GRTO）将策略梯度与工具可微分目标同时优化；②引入Bootstrapped Tool Optimization（BTO）预训练工具，缓解工具与策略异速训练的瓶颈；③采用经验回放与两阶段训练，实现工具与策略的自适应对齐。

**🔧 技术方法**

技术方法包括GRPO/GRTO强化学习、BTO预训练、经验回放、可微分BCE+soft-IoU损失、InternVL3.5-8B作为视觉语言模型、SAM3作为分割工具。

**📊 数据集**

使用的数据集有：COD10K、CAMO、NC4K（隐蔽目标分割）；EarthReason（遥感分割）；ReasonSeg、ReasonSeg-R、ReasonSeg-X（推理分割）。

**📈 对比分析**

与多种基线（如SAM-R1、RemoteReasoner、StAR等）进行比较，B-GRTO在隐蔽目标分割和遥感分割上实现或逼近SOTA，在推理分割上多次获得第一/第二名；同时在训练预算上收敛更快、性能提升显著。

**⚠️ 局限性**

局限性包括：①工具预训练高度依赖初始回放的质量；②在样本极少的推理分割任务上易过拟合；③对策略与工具学习率的匹配敏感；④在推理分割上尚未突破最强SoTA。

---

## 361. Push Your Agent: Measuring and Enforcing Quantitative Goal Persistence in Long-Horizon LLM Agents

**arXiv ID:** 2605.23574 | [PDF](https://arxiv.org/pdf/2605.23574v1)

**作者:** Yuandao Cai `[一作]` (Independent Researcher), Shengchao Qin `[通讯]` (Xidian University)

**通讯引用:** 2477 | [OpenAlex ID](https://openalex.org/A5042013137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出QGP（Quantitative Goal Persistence）框架，评估长周期语言代理是否能持续完成显式计数目标，并通过可验证的进度追踪来衡量失败模式。

**💡 创新点**

创新点在于：①把可外部验证的进度作为评价标准，显式检测重复提交、错误完成和提前停止；②设计两类受控任务族（仓库检索与验证器驱动工作单元）；③提出状态跟踪控制器（搜索页、已提交标识等）来显式维护进度，从而提升持久性。

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑4、Claude Sonnet 4.6、Codex CLI），LangGraph 与 LLMPolicy 的控制器架构，外部验证器（基于 ID、脚本检查等），以及内存辅助实现（LangGraph+Memory）。

**📊 数据集**

数据集包括本地 GitHub 仓库快照（Spark、pandas、NumPy 等）以及公开数据（Vega Car、FiveThirtyEight 航班安全数据）用于构造工作单元。

**📈 对比分析**

通过对比标准、验证器门控、状态跟踪控制器和内存基线，实验发现：状态跟踪控制器在所有模型下成功率可达约70%（而标准/门控控制器仅 0–30%），且重复提交率降至0；在工作单元任务中，普通控制器完全失败，状态跟踪控制器恢复 25–50% 的成功率。

**⚠️ 局限性**

局限性包括：①受控任务不具备多样性，难以推广到更复杂的工业场景；②评估依赖外部验证器，对仅能在结束时人工判定的任务不适用；③内存基线结果受实现细节影响，未必可直接替代控制器；④在更真实的编码/前沿代理评测中仍出现持久性失败，说明该指标的挑战性仍高。

---

## 362. Misleading Microbenchmarks on the Java Virtual Machines

**arXiv ID:** 2605.23570 | [PDF](https://arxiv.org/pdf/2605.23570v1)

**作者:** Filippo Schiavio `[一作]` (Univertistà della Svizzera italiana), Walter Binder `[通讯]` (Univertistà della Svizzera italiana)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过三种案例研究，阐明了在隔离（sterile）环境下运行的微基准会导致 JVM JIT 产生不现实的优化，从而误导性能评估。

**💡 创新点**

创新点在于首次系统地揭示微基准环境与真实应用之间的偏差，并提出通过预执行多样化输入（如 @Setup 阶段）和 record‑and‑replay 方案来获得更具代表性的测量结果。

**🔧 技术方法**

主要技术包括 JMH 微基准框架、JVM 运行时的 Profile‑Guided Optimization、JIT 编译器监控、静态分析、以及自定义的记录与重放框架。

**📊 数据集**

使用的数据集包括 DaCapo‑Chopin、Renaissance、JEDI（TPC‑H 查询）、以及 CollectionBench 的 HashMap/HashSet 基准，并结合人工合成的短字节数组和 Stream 操作。

**📈 对比分析**

对比方法是将基线实现与改进实现分别在“纯粹隔离”与“预执行”两种执行策略下进行测量；在隔离环境中改进实现往往显示 0.97×–1.43× 的加速，但在更真实的或重放基准下加速仅为 0.46×–1.05×，甚至出现劣化，最高可达 1.41× 的误差。

**⚠️ 局限性**

局限性包括：仍无法完全消除基准与真实工作负载之间的差异；record‑and‑replay 实现复杂耗时；JVM 初始化阶段对标准库的预热产生系统偏差；微基准仍需谨慎设计与多次验证。

---

## 363. PathNavigate: A Training-Free Pathology Agent with Surprise-Guided Scan and Shared Slide Memory for Whole-Slide Image VQA

**arXiv ID:** 2605.23559 | [PDF](https://arxiv.org/pdf/2605.23559v1)

**作者:** Chunze Yang `[一作]` (Xi'an Jiaotong University), Chen Li `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 30750 | [OpenAlex ID](https://openalex.org/A5100379155)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种训练‑free 的 WSIVQA 代理，采用扫描‑搜索‑读取（Scan‑Search‑Readout）流程，先低倍扫描生成异常区域池，再在此池内用 PLIP 进行查询匹配，最后在高倍图像中提取证据并给出答案；同时共享在线内存把导航与答案上下文绑定。

**💡 创新点**

创新点在于：①通过自监督的“惊讶”记忆在不需要查询的情况下构建 slide‑specific 内容先验；②把查询仅限定在此先验池内，实现查询与视觉先验的解耦；③共享同一在线状态完成导航和答案读出，减少额外上下文和计算。

**🔧 技术方法**

使用了：CONCH 低/高倍特征提取、冻结的 PLIP 文图相似度、Patho‑R1‑7B 病理 VLM 作为证据转换器、Qwen3.5‑4B 语言模型作为判定器、基于梯度范数的惊讶记忆网络、NMS + 余弦排序的 ROI 选取。

**📊 数据集**

数据集：WSI‑VQA（开放式与 MCQ）、SlideBench‑BCNB（多类别诊断子任务）以及 PathMMU（patch 级验证）。

**📈 对比分析**

与预训练 MLLM（GPT‑4o、GPT‑5.4、Gemini‑3.1、Qwen3.5‑4B/9B）、监督病理模型（SlideChat、WSI‑LLaVA、MedDr、TITAN 等）以及训练‑free 对手 PathAgent 进行对比。实验表明，在 WSI‑VQA 的开放式准确率从 33.9% 提升到 61.0%，在 SlideBench‑BCNB 的总体准确率从 55.7% 提升到 59.4%，并且在多个子任务（Tumor、ER、PR、HER2）上都有显著提升；同时与 PathAgent 的成本对比显示存储与 token 需求降低。

**⚠️ 局限性**

局限性包括：仅利用 H&E 图像，无法捕获 IHC 或其他模态信息；对需要非可视化信息的诊断任务仍表现不佳；实现复杂度高，对 GPU 记忆的实时更新要求较大；以及对超大规模 slide 仍需进一步优化扫描速度。

---

## 364. MindCopilot: Towards Formalizing and Evaluating Granular Human-LLM Co-Writing

**arXiv ID:** 2605.23535 | [PDF](https://arxiv.org/pdf/2605.23535v1)

**作者:** Youqing Fang `[一作]` (University of Science and Technology of China), Kai Chen `[通讯]` (Shanghai AI Laboratory)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

将交互式写作建模为人机循环马尔可夫决策过程，并提出了交互感知的评估指标；

**💡 创新点**

提出Hierarchical Acceptance Rate (HAR) 与 Knowledge-aware Editing Distance (KED) 两个面向交互的度量，能更真实反映用户接受与编辑成本；

**🔧 技术方法**

使用LLM生成提示，构建HitL‑MDP框架，设计强化学习与仿真评估流程；

**📊 数据集**

利用60篇人写论文（共16个写作领域）拆分为1,688个续写任务，并收集CoAuthor等用户编辑日志；

**📈 对比分析**

在L1、L2等四种交互范式下，HAR与KED相比传统文本相似度指标更贴近人类编辑时间；GPT‑5.1在HAR上表现最佳，Gemini‑2.5‑Pro在KED上最小；强化学习提升了HAR并降低KED；

**⚠️ 局限性**

缺点在于评估仍依赖人工标注与仿真，真实写作情境的多样性未完全覆盖；指标对非常长文本或极端创意写作的适用性待验证；

---

## 365. SafeSABR: Risk-Calibrated Adaptive Bitrate Streaming over Starlink Networks

**arXiv ID:** 2605.23560 | [PDF](https://arxiv.org/pdf/2605.23560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 366. LiveFigure: Generating Editable Scientific Illustration with VLM Agents

**arXiv ID:** 2605.23527 | [PDF](https://arxiv.org/pdf/2605.23527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 367. Unjust Enrichment as a Remedy for AI's Unauthorised Use of Protected Data

**arXiv ID:** 2605.23503 | [PDF](https://arxiv.org/pdf/2605.23503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 368. CoSPlay: Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test

**arXiv ID:** 2605.23491 | [PDF](https://arxiv.org/pdf/2605.23491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 369. Asking For An Old Friend: Diagnosing and Mitigating Temporal Failure Modes in LLM-based Statutory Question Answering

**arXiv ID:** 2605.23497 | [PDF](https://arxiv.org/pdf/2605.23497v1)

**作者:** Max Prior `[一作]` (Technical University of Munich), Matthias Grabmair `[通讯]` (Technical University of Munich)

**通讯引用:** 499 | [OpenAlex ID](https://openalex.org/A5003638231)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了312条专家验证的德国法条问答基准，并在不同LLM推理方式下诊断时间性失效；

**💡 创新点**

首次系统地定义并评估后期滞后与时间偏差两类失效模式，并证明检索增强（RAG）通过硬性时间过滤可显著缓解此问题；

**🔧 技术方法**

采用检索增强生成（RAG‑kNN 与 RAG‑ToC）、Web 搜索与 Vanilla 推理，结合 LLM‑as‑judge 评价框架；

**📊 数据集**

使用基于德国联邦法律的历史版本（BGB、StPO、AO 等）构造的 312 条 QA 对；

**📈 对比分析**

与 vanilla、Web 以及两种 RAG 的 5 种模型比较，RAG 在所有指标上均超 30%（Post‑Cutoff 结果正确率提升至 86%+），而 vanilla 在 Post‑Cutoff 仅 24‑40%；

**⚠️ 局限性**

局限性包括多条款推理仍有性能下降、实验仅覆盖德国民法，未验证在普通法司法体系和判例法场景中的泛化能力。

---

## 370. MISRust: Mapping MISRA-C++ Coding Guidelines to the Rust Programming Language

**arXiv ID:** 2605.23490 | [PDF](https://arxiv.org/pdf/2605.23490v1)

**作者:** Marius Molz `[一作]` (RWTH Aachen University), Alexandru Kampmann `[通讯]` (RWTH Aachen University)

**通讯引用:** 189 | [OpenAlex ID](https://openalex.org/A5079941800)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统地将MISRA C++ 2023的179条编码准则映射到Rust语言，构建了6类分类框架并判定每条准则在安全Rust与不安全Rust中的适用性。

**💡 创新点**

创新点在于：①提供完整的逐条规则映射；②引入6级分类，区分可自动满足、需手动遵循或需改写的准则；③量化安全Rust自动满足率与不安全Rust需要的准则数量。

**🔧 技术方法**

技术方法主要是对每条MISRA规则的根本动机进行人工审查与分析，结合Rust语言特性（所有权、借用检查、类型安全等）进行归类，并给出必要时的Rust特定改写。

**📊 数据集**

使用的数据集为MISRA C++ 2023标准中的179条规则，按规则编号、类别与动机进行手工归档。

**📈 对比分析**

比较方式：将每条规则划分为C1–C6六类，统计各类数量并计算比例。结果显示，约47.8%已自动满足，69条仍需遵循，其中36条在不安全Rust中失效。性能指标体现为安全Rust的规则负担显著降低。

**⚠️ 局限性**

局限性包括：①仅针对C++基础规则，未考虑其他安全标准；②依赖手工判定，主观性和审稿人数量少；③仅基于当前Rust版本，未来语言演进可能影响结果。

---

## 371. Calibration-Informative Region Selection for Online LiDAR--Camera Calibration in Agricultural Environments

**arXiv ID:** 2605.23580 | [PDF](https://arxiv.org/pdf/2605.23580v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 372. Understanding Goal Generalisation in Sequential Reinforcement Learning

**arXiv ID:** 2605.23565 | [PDF](https://arxiv.org/pdf/2605.23565v1)

**作者:** Jason Ross Brown `[一作]` (University of Cambridge), Edward James Young `[通讯]` (University of Cambridge)

**通讯引用:** 988 | [OpenAlex ID](https://openalex.org/A5065312639)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究强化学习代理在多阶段训练下的目标泛化，构建大规模实验评估训练管道对 OOD 行为的影响，并提出潜在策略梯度方法预测代理偏好。

**💡 创新点**

将代理偏好视为低维潜变量的演化过程，利用潜在策略梯度进行可解释预测；揭示训练顺序、特征重要性与价值持久性对泛化的决定性作用。

**🔧 技术方法**

使用 PPO + CNN 训练迷宫导航代理，采用 Elo 分数/Boltzmann 规则建模偏好，设计潜在策略梯度优化与梯度投影分析，构造特征相似度矩阵。

**📊 数据集**

自生成的 8×8 迷宫，包含 12 种颜色-形状组合，构造 298 条训练管道（单/双阶段，是否干扰）和 250+ 个 OOD 评估环境。

**📈 对比分析**

与多种基线（Diagonal S、Memoryless、Simultaneous、Quadratic、Uniform Random）在四种评估（全数据、4‑fold CV、单阶段预测双阶段、无干扰预测有干扰）下比较，使用 KL 损失、总变差、Brier 分数等指标；潜在策略梯度在所有评估中均取得最低 KL 损失，性能优于其他方法。

**⚠️ 局限性**

仅在单一架构/算法（PPO+CNN）和两阶段管道上验证；假设目标值线性可分解；使用单个随机种子；对更长、更复杂管道、不同任务/算法的泛化仍需进一步研究。

---

## 373. Experimental Evaluation of LPWAN Technologies: mioty, LoRaWAN, Sigfox, NB-IoT, and LTE-M in Deep Indoor Environments

**arXiv ID:** 2605.23483 | [PDF](https://arxiv.org/pdf/2605.23483v1)

**作者:** Christof Röhrig `[一作]` (Dortmund University of Applied Sciences and Arts), Benz Cramer `[通讯]` (Dortmund University of Applied Sciences and Arts)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文通过实测，比较了五种LPWAN技术（mioty、LoRaWAN、Sigfox、NB‑IoT、LTE‑M）在深度室内及地下车库中的建筑穿透能力。

**💡 创新点**

创新点在于使用低成本市售设备进行实地实验，评估建筑穿透损耗并将结果应用于杜特蒙特大学的计量系统。

**🔧 技术方法**

采用LoRa、Sigfox、NB‑IoT、LTE‑M以及mioty的设备，通过测量RSSI/RSRP、PER等指标，并利用ADR、SF调节等技术。

**📊 数据集**

实验数据来自杜特蒙特大学校园多栋建筑和地下六层停车场的实际测量点。

**📈 对比分析**

通过将室外基准RSSI减去室内RSSI计算建筑穿透损耗，并比较PER，结果显示NB‑IoT最佳，mioty次之，LoRa中等，Sigfox和LTE‑M表现较差。

**⚠️ 局限性**

主要局限包括测量设备的RSSI精度有限、Sigfox基站稀疏导致覆盖依赖距离、mioty网络缺乏公共部署以及受环境噪声影响导致的灵敏度变化。

---

## 374. PhenoYieldNet: Learning Crop-Aware Phenological Responses for Multi-Crop Yield Prediction

**arXiv ID:** 2605.23478 | [PDF](https://arxiv.org/pdf/2605.23478v1)

**作者:** Yu Luo `[一作]` (University of Sydney), Kun Hu `[通讯]` (Edith Cowan University)

**通讯引用:** 13066 | [OpenAlex ID](https://openalex.org/A5028673475)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了一种统一的多作物产量预测框架 PhenoYieldNet，能够学习作物特定的表型学响应并融合遥感影像与气象时序信息，实现跨作物、跨区域的产量预测。

**💡 创新点**

创新点包括：① Crop Phenology Bank 与 Crop Phenology Attention 的组合，显式建模作物表型学与气象驱动的多尺度趋势与变异；② Temporal Contrastive Adaptation（TCA）自监督对比学习策略，用于弥合遥感基础模型与农学时序特征的领域差距；③ 两阶段训练流程，先对编码器做时序自监督微调，再冻结编码器仅 fine‑tune 解码器。

**🔧 技术方法**

使用的主要技术包括：视觉 Transformer（ViT）作为遥感特征编码器；多模态交叉注意力融合卫星影像与气象序列；时间序列分解与多尺度平均池化；Crop Phenology Attention 采用多尺度趋势/变异的线性投影与 bias 注入；Temporal Contrastive Learning 采用对比损失；最终使用 MLP 进行回归。

**📊 数据集**

实验数据集：① CropNet（Sentinel‑2 影像 + HRRR 气象，涵盖 2017‑2022 的四种主要作物：大豆、玉米、棉花、冬小麦）；② MODIS（MODIS 影像 + HRRR 气象，涵盖 2003‑2015 的玉米）。

**📈 对比分析**

方法比较：与单作物基线（MMST‑ViT、Transformer、UNet‑ConvLSTM、MMVF）以及多作物基线 YieldNet 进行对比；在 MODIS 上 PhenoYieldNet 的 RMSE 为 5.95、R² 0.663、Corr 0.814，均优于所有基线；在 CropNet 上的多作物预测也显示 RMSE、R² 与 Corr 均显著提升，尤其在复杂气象区域表现更为稳健。

**⚠️ 局限性**

局限性：① Crop Phenology Bank 以作物种类为单位，缺乏对未见作物或更细粒度生长阶段的适应性；② 多作物训练易受类别不平衡与分布漂移影响，在少数类作物（如冬小麦）上仍存在一定性能落差。

---

## 375. Safety, Liveness, and Fairness in Quantitative Argumentation Dialogues

**arXiv ID:** 2605.23578 | [PDF](https://arxiv.org/pdf/2605.23578v1)

**作者:** Arunavo Ganguly `[一作]` (Umeå University), Timotheus Kampik `[通讯]` (Umeå University)

**通讯引用:** 524 | [OpenAlex ID](https://openalex.org/A5014935371)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在量化双极论证对话（QBAG链）中定义并系统化了安全、活性与公平三类性质，并给出了它们之间的形式化关系与推理结果。

**💡 创新点**

创新点在于将传统的时间逻辑与Petri网中的安全、活性、公平概念迁移至量化论证图的动态过程，提出强弱安全、活性波动以及基于Gini和香农熵的渐进公平度量。

**🔧 技术方法**

采用模块化语义（如DFQuAD）对QBAG进行数值推理，并利用阈值阈断法、序列化更新与图拓扑分析来评估安全与活性；公平度量则基于统计曲线与积分公式实现。

**📊 数据集**

本文未使用公开数据集，而是通过手工构造的示例QBAG链（包含初始权重与攻击/支持关系）来演示理论与度量。

**📈 对比分析**

由于缺乏大规模实验，本文仅通过示例演示指标（如Gini公平分数0.46212、香农公平分数0.55449）说明方法可行性，未给出与现有算法的性能对比。

**⚠️ 局限性**

局限性包括：1）对非循环QBAG的假设导致不适用于可能产生未定义最终强度的图；2）在更一般的链更新（如强扩展）下难以保证SLF性质；3）缺乏实证评估与大规模数据验证。

---

## 376. HARNESS-LM: A Three-Phase Training Recipe for Harnessing SLMs in Sponsored Search Retrieval

**arXiv ID:** 2605.23572 | [PDF](https://arxiv.org/pdf/2605.23572v1)

**作者:** Vipul Gupta `[一作]` (Microsoft AI), Manik Varma `[通讯]` (Microsoft AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为HARNESS-LM (HLM) 的三阶段训练框架，旨在将大型检索模型的能力转移到紧凑且高效的模型中，以应对赞助搜索中的检索质量与生产延迟之间的平衡。

**💡 创新点**

创新点在于通过三阶段的训练过程，分别训练高性能的教师检索器、对查询表示进行对齐以提取知识，并通过对比精炼阶段优化学生模型的检索性能。

**🔧 技术方法**

使用了L_2对齐目标、对比学习（Contrastive Learning）等技术来实现知识转移和模型优化。

**📊 数据集**

使用了来自Bing Ads的真实世界数据集进行评估，包括250M的查询-文档对和2B的查询文本用于对齐。

**📈 对比分析**

与当前生产中的模型进行在线A/B测试，HLM模型在多个关键业务指标上表现出显著提升：收入提高1%，展示量提高0.6%，点击量提高0.4%，同时在线查询编码器延迟降低了27倍，吞吐量提高了20倍。

**⚠️ 局限性**

限制在于虽然HLM在性能和效率上取得了显著提升，但仍需进一步扩展到更强的教师模型和更广泛的嵌入任务。

---

## 377. TactileReflex: Noise-Statistics-Driven Vision-Tactile Reflex Control for Force-Sensitive Manipulation

**arXiv ID:** 2605.23568 | [PDF](https://arxiv.org/pdf/2605.23568v1)

**作者:** Ziyan Feng `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Qiang Nie `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于噪声统计的闭环反射控制范式，旨在通过视觉触觉传感器实现对易变形容器的实时抓取力适应。

**💡 创新点**

创新点在于通过简短的静态保持和卸载校准，自动推导所有控制器阈值，消除了手动调试和外部力校准的需求。

**🔧 技术方法**

使用了基于视觉的触觉传感器和三通道闭环控制器，分别处理剪切强度、接触强度和压力中心。

**📊 数据集**

使用了两种不同材料的杯子（软杯和硬杯）进行实验，验证了控制器的有效性。

**📈 对比分析**

与固定努力基线相比，TactileReflex在动态倒水任务中表现出色，成功率为90%，而固定努力基线在所有尝试中均失败。

**⚠️ 局限性**

局限性在于控制器主要集中在抓取器层面，且仅处理灰度图像，未来工作将探索全彩触觉图像和更广泛的材料集。

---

## 378. Operator Learning for Reconstructing Flow Fields from Sparse Measurements: a Language Model Approach

**arXiv ID:** 2605.23712 | [PDF](https://arxiv.org/pdf/2605.23712v1)

**作者:** Qian Zhang `[一作]` (Brown University), George Em Karniadakis `[通讯]` (Brown University)

**通讯引用:** 102453 | [OpenAlex ID](https://openalex.org/A5009658255)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于语言模型架构的解算器学习框架（RFormer），用于从稀疏测量中无网格重构流场；

**💡 创新点**

创新点在于将稀疏重构任务重新表述为序列到序列的问答式问题，利用decoder‑only transformer的注意力机制实现长程空间关联、无结构点云推断，并通过自定义注意力掩码实现对观察点与查询点的区分；

**🔧 技术方法**

核心技术包括token构造（位置+数值/零）、单层线性嵌入、无位置编码、层归一化和MLP输出、相对RMSE损失、批量采样与查询分块推理；

**📊 数据集**

四个多样数据集：二维涡街模拟、美国陆地日均温度、基于DPD的三维血流模拟、实验测得的三维湍流喷流；

**📈 对比分析**

与插值、克里金、Gappy POD、CNP/ANP/TNP以及Voronoi CNN等基线比较，RFormer在所有数据集上实现了最低的相对RMSE（尤其在湍流喷流中误差减半），同时保持了较低的模型参数量和推理效率；

**⚠️ 局限性**

局限性包括：仅针对空间重构，未涵盖时间动态；对极端噪声或极稀疏采样的鲁棒性仍待进一步验证；大规模点云仍需分块推理，可能导致重构边界效应；

---

## 379. Flare: Leveraging Serverless Elasticity to Absorb Microservice Load Spikes

**arXiv ID:** 2605.23707 | [PDF](https://arxiv.org/pdf/2605.23707v1)

**作者:** Dilina Dehigama `[一作]` (University of Edinburgh), Boris Grot `[通讯]` (University of Edinburgh)

**通讯引用:** 3205 | [OpenAlex ID](https://openalex.org/A5010276850)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种混合微服务架构Flare，结合虚拟机（VM）和无服务器计算，以应对微服务负载峰值。

**💡 创新点**

创新点在于动态流量转移机制，能够在负载峰值时仅将过载的服务流量转移到无服务器实例，从而提高可扩展性和成本效益。

**🔧 技术方法**

使用了Kubernetes作为基础架构，结合Knative进行无服务器部署，并利用Istio服务网格进行流量管理和监控。

**📊 数据集**

使用了Twitter的负载跟踪数据集进行评估，分析了在不同负载情况下的性能表现。

**📈 对比分析**

与全VM部署相比，Flare在峰值尾延迟上平均减少了49.7%，同时成本增加不到4.1%。

**⚠️ 局限性**

限制在于Flare在处理极端负载峰值时可能仍会出现少量SLO（服务级别目标）违规，尤其是在尾部延迟方面。

---

## 380. Graph-based Complexity Forecasts in UK En Route Airspace Using Relevant Aircraft Interactions

**arXiv ID:** 2605.23696 | [PDF](https://arxiv.org/pdf/2605.23696v1)

**作者:** Edward Henderson `[一作]` (Alan Turing Institute), Nick Pepper `[通讯]` (Alan Turing Institute)

**通讯引用:** 216 | [OpenAlex ID](https://openalex.org/A5085534301)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

本文提出一种基于相关机组对概率模型的实时复杂度预测方法，并通过与ATCO的迭代反馈改进了相关机组过滤器。

**💡 创新点**

创新点在于将实时航班到达时间与图论表示的航路网络相结合，利用概率轨迹预测实现对45分钟内ATCO工作量的高相关性预测，显著优于传统流量预测。

**🔧 技术方法**

所使用的技术包括概率轨迹预测、蓝色数字孪生中的垂直轨迹不确定性分析、路网重采样与层次聚类、正态分布到达时间估计，以及Spearman相关系数与移动块自助检验。

**📊 数据集**

数据集包括2024年1-12月的LMS雷达、航路、指令数据，NATS NAS FDP实时运营数据，以及2025年5月单日的历史和实时交通场景。

**📈 对比分析**

通过与ATCO标注的50个静态样本比较，更新过滤器F1=0.84，原始滤波F1=0.69；在45分钟预测时，Spearman ρ=0.68，对比流量预测ρ≈0.55，差异显著（p=0.0004）。

**⚠️ 局限性**

局限性包括仅在单一伦敦中间航区验证，缺乏多季节和跨区评估；图模型仅二维，未显式处理垂直层；使用固定5分钟σ值，可能不适用于更远期预测。

---

## 381. Validating Threat Modeling Results with the Help of Vulnerable Test Applications

**arXiv ID:** 2605.23695 | [PDF](https://arxiv.org/pdf/2605.23695v1)

**作者:** Oleksandr Adamov `[一作]` (Blekinge Institute of Technology), Nishrith Saini `[通讯]` (Ericsson)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5090947442)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用已知漏洞的测试应用验证威胁建模工具的覆盖率，比较了ThreMoLIA和Microsoft Threat Modeling Tool（MTMT）在AzureGoat和VulnBank上的表现。

**💡 创新点**

提出基于已知漏洞集合的威胁建模验证协议，将漏洞集当作外部oracle，并将LLM辅助的ThreMoLIA与传统工具进行对比，验证了LLM在混合传统与AI攻击面上的优势。

**🔧 技术方法**

使用GPT‑5 LLM进行检索增强生成（RAG）、prompting与质量保证，配合STRIDE、LINDDUN、OWASP Top 10等威胁框架；工具输入为系统架构图、数据流图和简要描述。

**📊 数据集**

采用两个故意漏洞的开源应用：AzureGoat（7个已知漏洞）和Vulnerable Bank Application（VulnBank，63个已知漏洞）作为实验数据集。

**📈 对比分析**

通过计算覆盖率（已发现漏洞数 / 已知漏洞总数）评估工具效果；ThreMoLIA在AzureGoat上实现100%覆盖，MTMT仅57.1%；在VulnBank上ThreMoLIA最高可达92.1%（merged 1+2），而MTMT仅55.6%，显示LLM辅助工具在多样化攻击面上性能更优。

**⚠️ 局限性**

局限性包括：仅评估覆盖率而不考虑精确度与误报、仅使用两款应用、merged 结果成本不等价于单次运行、LLM模型可能已接触过漏洞集合、缺乏对分析师工作量和时间的度量。

---

## 382. OpenSkillEval: Automatically Auditing the Open Skill Ecosystem for LLM Agents

**arXiv ID:** 2605.23657 | [PDF](https://arxiv.org/pdf/2605.23657v1)

**作者:** Jiahao Ying `[一作]` (Singapore Management University), Yixin Cao `[通讯]` (Fudan University)

**通讯引用:** 5833 | [OpenAlex ID](https://openalex.org/A5013247988)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 OpenSkillEval，一个自动化评估框架，用于对 LLM 代理系统与社区贡献的技能在真实下游任务中的使用效果、质量与成本进行系统评估。

**💡 创新点**

创新点：①将评估从静态基准迁移到动态、基于实际资源生成的任务实例；②同时从代理轨迹和最终输出两维度评估技能效用；③构建统一的技能收集与组织流程，使不同技能可在相同任务设定下可比对。

**🔧 技术方法**

技术手段：自动化任务实例生成（LLM 反向工程、验证器）；技能收集与整理（爬取公开仓库、下载量过滤）；轨迹分析（ATIF 轨迹解析与技能流程对比）；自动化质量评估（基于预设指标的多维打分、视觉多样性评估、数据准确性检查）以及人类评测验证。

**📊 数据集**

数据集：约 600+ 动态生成的任务实例，覆盖演示生成、前端网页设计、海报生成、数据可视化、报告生成；30 条社区贡献的公开技能；使用公开的网页、文档、数据集、可视化示例等原始资源。

**📈 对比分析**

比较方法：在多种代理框架（Claude Code、Codex、Gemini CLI、Kimi CLI 等）与不同模型（Claude 4.6、GPT‑5.x、Gemini 3.1 Pro、DeepSeek V4 Pro 等）下进行“无技能 vs 强制使用技能”两种设置；评估指标包括内容质量、视觉设计、完整性、数据准确性、响应速度、token 费用等。结果显示：技能可用性并不等同于有效利用；不同模型/框架对技能的收益差异大；大多数公开技能并不总能显著提升性能，且往往伴随更高 token 成本。

**⚠️ 局限性**

局限性：①评估覆盖的技能和任务仍有限，难以全面覆盖日益扩大的技能生态；②自动化评测难以完全模拟真实人机交互与部署场景；③未对技能本身进行独立评估，仍依赖于代理模型的解释与执行能力；④成本与效率折衷未在所有模型间系统化分析。

---

## 383. One Policy, Infinite NPCs: Persona-Traceable Shared RL Policies for Scalable Game Agents

**arXiv ID:** 2605.23652 | [PDF](https://arxiv.org/pdf/2605.23652v1)

**作者:** Yoosung Hong `[一作]` `[通讯]` (Independent Researcher), Yoosung Hong (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Persona‑Conditioned Shared Policy（PCSP）方案，用冻结LLM嵌入对NPC人格进行一次性编码，再用单一共享策略实现实时、可控、零样本的角色行为；

**💡 创新点**

核心创新在于：①用大型语言模型的语义嵌入空间对人格进行无监督编码；②通过低秩投影与InfoNCE一致性损失将人格语义与轨迹行为对齐；③将人格条件化与PPO+KL多样性正则统一训练，形成单一轻量化网络；

**🔧 技术方法**

技术手段包括：冻结Qwen3-0.6B嵌入、低秩投影层、FiLM/concat条件化、PPO强化学习、InfoNCE轨迹一致性损失、KL多样性正则、ONNX部署、UE5行为树集成；

**📊 数据集**

数据集涵盖：300个人工合成的Persona（15大五人格×20职业），Melting Pot 2.4.0三种社会困境子环境，UE5模拟城市地图；

**📈 对比分析**

与无人格PPO、SBERT‑条件、DIAYN、LLM‑as‑policy等基线比较，PCSP在300人设基准上零样本人格识别率≈17%（≈10×随机），Spearman ρ≈0.73，推理速度比LLM‑as‑policy快22×，在Melting Pot和UE5的部署实验中也保持一致的Persona区分与实时性；

**⚠️ 局限性**

局限性包括：①对未在训练集内的全新词汇（vocabulary‑held‑out）检索仍失败；②当前人格静态化，缺乏动态记忆与情感进化；③动作空间有限，难以细致表达社会/风格行为；④引擎与研究侧的可观测性差异导致对齐度下降；⑤人类评估仅限粗粒度，缺乏细粒度可信度分析；

---

## 384. EM-Vid: Training-Free Entity-Centric Memory for Efficient and Consistent Multi-Shot Video Generation

**arXiv ID:** 2605.23610 | [PDF](https://arxiv.org/pdf/2605.23610v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 385. DualMem: Bypassing the Objectness Bottleneck for Calibrated Unknown-Stream Filtering in Open-World Object Detection

**arXiv ID:** 2605.23634 | [PDF](https://arxiv.org/pdf/2605.23634v1)

**作者:** Yingjun Xiao `[一作]` (Guangzhou University), Siyuan Chen `[通讯]` (Guangzhou University)

**通讯引用:** 1293 | [OpenAlex ID](https://openalex.org/A5100429228)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种后处理过滤器DualMem，用于改善开放世界物体检测（OWOD）中的未知物体检测，特别是减少背景假阳性。

**💡 创新点**

创新点在于通过非参数似然比检验在冻结的SigLIP特征空间中进行未知流过滤，利用正负k近邻记忆，并通过Neyman-Pearson校准选择抑制阈值。

**🔧 技术方法**

使用了SigLIP特征作为冻结的外部批评者，结合k近邻记忆和非参数似然比检验技术。

**📊 数据集**

使用了M-OWODB数据集进行实验，评估了在不同OWOD检测器（PROB、OW-DETR和HypOW）上的性能。

**📈 对比分析**

与自然K均值原型基线相比，DualMem在PROB任务1上减少了58.6%的背景假阳性，同时保持已知类别的mAP不变，U-Recall在3.4个百分点内保持不变。

**⚠️ 局限性**

限制在于DualMem依赖于小规模的标注校准分割，且在不同检测器之间的NP预算转移不均匀，HypOW的校准-测试差距较大。

---

## 386. Formally Verified Liveness with Multiparty Session Types in Rocq

**arXiv ID:** 2605.23633 | [PDF](https://arxiv.org/pdf/2605.23633v1)

**作者:** Omer Keskin `[一作]` (University of Edinburgh), Rob van Glabbeek `[通讯]` (University of Edinburgh)

**通讯引用:** 2613 | [OpenAlex ID](https://openalex.org/A5048229897)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本研究首次在Rocq证明助手中对同步多方会话类型的活性进行了机械化证明，建立了全局类型与局部类型之间的关联，并利用该关联证明了类型会话的安全性、无死锁性和活性。

**💡 创新点**

创新点在于首次机械化证明了多方会话类型的活性，并且基于关联关系的MPST系统的正确性进行了形式化证明。

**🔧 技术方法**

使用了Rocq证明助手和Paco库，采用共诱导树表示全局和局部类型，并定义了共诱导的子类型关系和投影关系。

**📊 数据集**

论文中没有具体提到使用的数据集，但涉及的类型和会话结构是基于理论构建的。

**📈 对比分析**

与现有文献中的非正式证明相比，本研究提供了清晰的机械化证明，确保了通信协议的活性属性的认证。性能上，Rocq实现了约14K行代码，提供了更高的证明效率。

**⚠️ 局限性**

限制在于该工作主要集中在同步多方会话类型的活性证明上，尚未扩展到其他类型的过程计算或不同的公平性假设下的活性属性。

---

## 387. DDX-TRACE: A Benchmark for Medical Diagnostic Trajectories in VLMs

**arXiv ID:** 2605.23629 | [PDF](https://arxiv.org/pdf/2605.23629v1)

**作者:** Jiazhen Pan `[一作]` (Technical University of Munich), Benedikt Wiestler `[通讯]` (TUM University Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的医学诊断基准，评估在隐藏证据下的多模态神经放射学诊断过程，基于211个具有挑战性的案例，模型需要在有限的临床历史下请求影像学检查，更新概率性鉴别诊断，并最终给出局部诊断。

**💡 创新点**

创新点在于将医学AI评估从仅关注最终答案转向关注证据支持的诊断过程，强调了在临床决策中获取和使用证据的重要性。

**🔧 技术方法**

使用了多模态神经放射学模型，结合了大语言模型（LLMs）和视觉语言模型（VLMs），并引入了过程感知的评估指标。

**📊 数据集**

使用了来自EuroRad的211个经过策划的神经放射学案例，包含785个可请求的影像证据单元和1609张图像。

**📈 对比分析**

与现有的医学AI基准相比，发现最终诊断分数可能严重误导工作质量，模型可能在缺乏必要证据的情况下做出合理的猜测，或在请求有用的研究时误解原始图像。通过控制证据变体，识别出规划、视觉证据提取和下游推理中的瓶颈。

**⚠️ 局限性**

限制在于当前的多模态诊断模型不仅受限于医学知识，还受限于主动证据获取、图像到发现的提取和不确定性意识推理，这些都是在临床决策中需要解决的重要障碍。

---

## 388. GlowGS: Generative Semantic Feature Learning for 3D Gaussian Splatting in Nighttime Glow Scenes

**arXiv ID:** 2605.23602 | [PDF](https://arxiv.org/pdf/2605.23602v1)

**作者:** Beibei Lin `[一作]` (National University of Singapore), Robby T. Tan `[通讯]` (National University of Singapore)

**通讯引用:** 8908 | [OpenAlex ID](https://openalex.org/A5103147507)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 GlowGS，针对夜间光晕场景的 3D 高斯 splatting 方法，利用语义特征生成与新视角语义学习提升渲染质量。

**💡 创新点**

创新点：①使用图像到视频扩散模型合成未知相机姿态的视角，再用视觉基础模型（VFM）验证质量并构建语义特征库；②在训练中不需要真实相机位姿，直接通过特征库对渲染的新视角进行语义约束，补偿结构缺失。

**🔧 技术方法**

主要技术：图像到视频扩散模型（Pika、PromeAI）、视觉基础模型（DINO、CLIP、ViT）、3D 高斯 splatting、语义特征匹配与最小化距离的学习损失。

**📊 数据集**

数据集：自研 NightGlow（18 场景、约 540 张带强光晕的夜景图），以及 RawNeRF-Glow、Bilarf-Glow 等现有夜景光晕数据集。

**📈 对比分析**

与 3DGS、CGS、MGS 及 NeRF 低光/夜景模型比较；在 NightGlow 上 PSNR 均提升 1.5-1.8 dB，SSIM 与 LPIPS 同样改善；在 RawNeRF-Glow、Bilarf-Glow 上也表现出显著优于基线的提升。

**⚠️ 局限性**

局限性：依赖扩散模型与 VFM 的质量；生成的新视角随机性和潜在伪影仍可能影响特征库；在极端光晕或遮挡严重的场景下效果尚待验证。

---

## 389. Solving the Aircraft Disassembly Scheduling Problem

**arXiv ID:** 2605.23592 | [PDF](https://arxiv.org/pdf/2605.23592v1)

**作者:** Charles Thomas `[一作]` (UCLouvain), Pierre Schaus `[通讯]` (UCLouvain)

**通讯引用:** 1433 | [OpenAlex ID](https://openalex.org/A5065393744)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并求解了飞机拆解调度问题，考虑技术人员分配、技能、空间容量与平衡约束；

**💡 创新点**

首次将技术技能、空间容量和质量平衡约束统一纳入RCPSP框架，并构造了高效的CP与MIP模型；

**🔧 技术方法**

使用约束规划（CP Optimizer）中的可选区间、累计函数和序列变量；并实现了基于事件的MIP模型；

**📊 数据集**

基于工业合作伙伴提供的Boeing 737NG-600拆解真实任务数据（共1454项任务），并生成多规模子实例；

**📈 对比分析**

与MIP模型对比，CP模型在所有实例上表现优异，能够求解至1450项任务；在约束松弛实验中，去除技能约束对求解速度影响最大；

**⚠️ 局限性**

MIP模型规模过大导致内存溢出，且在大规模实例上难以求解；CP模型仍在大实例上收敛慢，难以得到最优证明。

---

## 390. Metadata Predictability Is Not Evidence Dependence: An Intervention-Based Audit for Weak-Label Benchmarks

**arXiv ID:** 2605.23701 | [PDF](https://arxiv.org/pdf/2605.23701v1)

**作者:** Kan Shao `[一作]` `[通讯]` (Jinglue Technology Development (Nanjing) Co., Ltd.), Kan Shao (Jinglue Technology Development (Nanjing) Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于两统计量（MPDS 与 ΔEvi）的弱标签基准审计方法，用以区分元数据可预测性与证据依赖性。

**💡 创新点**

创新点在于将元数据可预测性与证据干预效应拆分为独立的检验指标，并结合读者校准与输入消融形成诊断图谱，揭示了隐式耦合与警告区的存在。

**🔧 技术方法**

采用统计假设检验、交叉证据置换、TF‑IDF+LR 轻量级读者、四种 Transformer 读者（BERT、DistilBERT、ELECTRA‑small、SciBERT）以及输入消融等技术。

**📊 数据集**

实验数据集包括构造的合成 HotpotQA（隐式耦合案例）、SNLI、FEVER 以及重构的真实 HotpotQA。

**📈 对比分析**

通过对 MPDS 与 ΔEvi 的比较，发现合成 HotpotQA 为隐式耦合（MPDS=0.643，ΔEvi≈0），SNLI 在轻量读者下 ΔEvi≈0 但在强读者下显著为正（≈0.3），FEVER 一直表现为正（≈0.13~0.68），重构 HotpotQA 在强读者下 ΔEvi 接近零，表明其为警告区。

**⚠️ 局限性**

局限性包括置换次数 K=8 过少、特征手工设计可能漏检高阶耦合、MPDS 与任务难度混杂、诊断图仅为示例性划分、以及对证据身份敏感性的关注未涵盖语义推理质量。

---

## 391. CRONOS: Benchmarking Counterfactual Physical Consistency in Video Models

**arXiv ID:** 2605.23699 | [PDF](https://arxiv.org/pdf/2605.23699v1)

**作者:** León Begiristain `[一作]` (University of Freiburg), Adam Kortylewski `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于干预的评测基准CRONOS，用于系统地评估视频生成模型在不同视觉干预（视角、场景、物体类别、外观）下的反事实物理一致性。

**💡 创新点**

创新点在于：①构建了高保真Unreal Engine仿真环境，能够保持物理事件不变的同时独立操控四个关键视觉因子；②设计了细粒度的对象中心指标与VLM裁判的混合评价体系；③通过全因子实验量化模型对干预的敏感性，揭示当前模型对视角和物体类别的强依赖。

**🔧 技术方法**

使用的技术包括：Unreal Engine渲染、DINOv2特征相似度、SAM3D网格重建、DisMo运动编码器、Qwen3-VL-32B视觉语言模型裁判，以及多种可视化与人类评测验证的评价指标。

**📊 数据集**

使用的数据集为CRONOS，包含675段高分辨率（1920×1080）视频，涵盖3种物理事件（碰撞、滚落与落地、遮挡）和5个场景、5个物体类别、最多4个视角及3种外观变体。

**📈 对比分析**

在I2V与V2V两种条件下，对Cosmos2.5、CogVideoX1.5、MAGI-1、Wan2.2等开源模型进行比较。评价指标显示即使是最优模型的成功率也仅为22%，不同干预下的性能波动显著，模型对视角和物体类别的敏感性尤其突出，模型规模增大并未带来一致性提升。

**⚠️ 局限性**

局限性包括：①合成与真实视频存在域差距；②评价主要基于单一参考轨迹，忽略多样可能性；③实验仅覆盖部分开源模型，未涉及商业闭源系统；④在评测中未考虑分布式或多样化物理约束。

---

## 392. Optimization of randomized neural networks for transfer operator approximation

**arXiv ID:** 2605.23689 | [PDF](https://arxiv.org/pdf/2605.23689v1)

**作者:** Mohammad Tabish `[一作]` (University of Edinburgh and Heriot-Watt University), Stefan Klus `[通讯]` (Heriot-Watt University)

**通讯引用:** 1174 | [OpenAlex ID](https://openalex.org/A5031087309)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个可优化的随机神经网络RaNNDy，用于近似动力学系统的传输算子。

**💡 创新点**

通过在隐藏层权重保持固定的前提下，引入可调参数化激活函数并对其进行优化，以改进随机基底，从而兼顾随机网络的低成本与可训练网络的精度。

**🔧 技术方法**

使用变分原理（Rayleigh/VAMP）构造损失，基于随机特征映射和闭式求解输出层权重，并采用梯度或贝叶斯优化调整激活参数。

**📊 数据集**

在图纹、Bickley Jet 流场以及高维蛋白 NuG2 分子动力学轨迹等三个不同数据集上进行实验。

**📈 对比分析**

与全训练的VAMPnets对比，RaNNDy在优化激活参数后表现出更快的收敛速度、更平滑的损失曲面以及更准确的谱分辨率，能够更好捕捉三种数据集中的基态/耦合结构。

**⚠️ 局限性**

仍受限于单一激活函数形式，优化仅关注权重分布尺度，对高维非线性系统的鲁棒性与可扩展性需进一步验证。

---

## 393. Recursive Block-Diagonal Coupling for Resource-Efficient Training of Vision Models

**arXiv ID:** 2605.23656 | [PDF](https://arxiv.org/pdf/2605.23656v1)

**作者:** Maxim Henry `[一作]` (Montefiore Institute, University of Liège), Marc Van Droogenbroeck `[通讯]` (Montefiore Institute, University of Liège)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种高效的训练协议，通过递归地训练独立模型并以块对角的方式将其结合，旨在提高计算机视觉模型的训练效率。

**💡 创新点**

创新点在于引入了递归块对角耦合的训练协议，允许灵活分配训练预算，并在相似测试准确率下减少30%的训练资源消耗。

**🔧 技术方法**

使用了递归块对角耦合的训练协议，结合了独立训练的模型，并采用零填充的初始化方法。

**📊 数据集**

在ImageNet数据集上评估了视觉变换器（DeiT）和卷积网络（ResNet）模型。

**📈 对比分析**

与标准训练协议相比，该方法在相似的测试准确率下减少了30%的训练资源消耗，并在相同训练预算下实现了更高的性能。

**⚠️ 局限性**

限制在于该方法的有效性可能依赖于特定的模型架构和训练设置，可能不适用于所有类型的模型。

---

## 394. CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception

**arXiv ID:** 2605.23655 | [PDF](https://arxiv.org/pdf/2605.23655v1)

**作者:** Liupeng Li `[一作]` (Harbin Institute of Technology), Yaowei Wang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 9252 | [OpenAlex ID](https://openalex.org/A5100631216)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个训练无关的认知视觉搜索框架（CVSearch），通过动态评估信息充分性并在必要时进行专家辅助搜索或场景感知扫描，显著提升多模态大型语言模型（MLLM）对高分辨率图像的感知能力。

**💡 创新点**

创新点在于：①引入认知评估-搜索工作流，在专家搜索失败时自动切换到语义感知扫描；②设计语义引导自适应分块（SGAP）以避免网格划分导致的语义碎片化；③提出自下而上的动态搜索策略，避免传统自顶向下搜索的错误传播。

**🔧 技术方法**

使用的技术包括：视觉专家SAM3进行快速目标定位；基于视觉特征的SLIC+凝聚聚类实现自适应分块；视觉复杂度先验与加权优先级的下行搜索算法；以及Qwen2.5‑VL‑7B等开源MLLM作为底层模型。

**📊 数据集**

在多套高分辨率基准上验证，包括HR-Bench（8K、4K）、V* Bench、MME‑RealWorld‑Lite、TreeBench、FineRS‑4K等。

**📈 对比分析**

与专家辅助搜索（SEAL、DyFo）和扫描式搜索（ZoomEye、RAP）等方法对比，CVSearch在所有基准上实现了SOTA精度，并在搜索效率（样本/分钟）上超过现有扫描框架，且在低分辨率模型上提升显著。

**⚠️ 局限性**

局限性包括：迭代搜索虽高效但仍慢于单通道MLLM和轻量级专家辅助方法；目前仅使用SAM3作为视觉专家，未验证对不同专家或多专家组合的通用性。

---

## 395. How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework

**arXiv ID:** 2605.23651 | [PDF](https://arxiv.org/pdf/2605.23651v1)

**作者:** Björn Nieth `[一作]` (FAU Erlangen-Nürnberg), Emmanuelle Salin `[通讯]` (FAU Erlangen-Nürnberg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种上下文感知的评估框架，通过比较人类参考语料库和LLM生成语料库的语言特征分布，评估生成文本的人类相似度。

**💡 创新点**

创新点在于引入了基于MMD的评估方法，关注语言特征的分布差异，并且实现了对七个开源模型在五个不同语域上的比较。

**🔧 技术方法**

使用了MMD（最大均值差异）和67个由Biber提出的词汇-语法特征，结合语料语言学的方法。

**📊 数据集**

使用了五个不同的英语数据集，包括口语对话、学术写作、在线指导文本、创意写作和新闻报道。

**📈 对比分析**

与人类基线进行比较，所有测试的LLM模型在语言特征分布上均偏离人类基线，模型的接近程度依赖于语域而非模型大小。

**⚠️ 局限性**

局限性包括只使用了英语数据集和固定的语言特征集，未涵盖所有可能的系统性偏差；实验仅在零-shot和few-shot生成设置下进行，未考虑其他提示技术的影响。

---

## 396. Herring: Parallel Batch-Order-Fairness on DAG-based Blockchain Consensus

**arXiv ID:** 2605.23648 | [PDF](https://arxiv.org/pdf/2605.23648v1)

**作者:** Marko Putnik `[一作]` (Delft University of Technology), Jérémie Decouchant `[通讯]` (Delft University of Technology)

**通讯引用:** 578 | [OpenAlex ID](https://openalex.org/A5087577380)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的批量订单公平性（batch-OF）DAG BFT协议Herring，旨在解决现有协议在公平性和性能之间的权衡问题。

**💡 创新点**

Herring是首个将公平性层的图构建成本并行化的批量订单公平性DAG BFT协议，显著提高了性能并降低了延迟。

**🔧 技术方法**

使用了DAG（有向无环图）结构和Rust编程语言实现，结合了后共识图构建和显式缺失边缘解析的技术。

**📊 数据集**

在评估中使用了FairDAG-RL、DoD-W和Themis等现有协议的实现进行比较，测试了不同工作负载、网络规模和公平性参数。

**📈 对比分析**

Herring在接近10,000 tx/s的吞吐量下，表现出比FairDAG-RL高出约90%的饱和吞吐量，比DoD-W高出100%，并在饱和状态下显著降低了执行延迟。

**⚠️ 局限性**

Herring的局限性在于其对网络规模和节点故障的容忍度要求较高，且在某些情况下可能仍然受到拜占庭攻击的影响。

---

## 397. Kernel-Based ReLU Approximation for Homomorphic Encryption-Compatible Privacy-preserving Deep Learning Models

**arXiv ID:** 2605.23641 | [PDF](https://arxiv.org/pdf/2605.23641v1)

**作者:** Dimitrios Sygletos `[一作]` (Hellenic Mediterranean University), Evangelos K. Markakis `[通讯]` (Hellenic Mediterranean University)

**通讯引用:** 2651 | [OpenAlex ID](https://openalex.org/A5009055115)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于核的ReLU近似方法，以支持同态加密（HE）兼容的隐私保护深度学习模型，解决了在加密环境中使用非线性激活函数的挑战。

**💡 创新点**

创新点在于通过平滑的核函数近似ReLU，并使用二次多项式进行逼近，从而在HE约束下实现隐私保护的深度学习模型。

**🔧 技术方法**

使用了核回归和多项式回归技术，结合Jackson定理来优化多项式的选择。

**📊 数据集**

使用了来自RoBERTa和DistilBERT的嵌入数据，基于Stanford Sentiment Treebank (SST-2)数据集进行训练和评估。

**📈 对比分析**

与现有的多项式近似方法（如X平方、FasterCryptoNets和Chebyshev多项式）进行比较，结果显示所提方法在近似精度和计算效率上表现优越，尤其在加密环境中保持了较低的均方误差（MSE）和较高的分类准确率。

**⚠️ 局限性**

限制在于在高方差情况下的表现可能不如某些其他方法，未来研究将探讨在高方差环境中的适应性混合模型。

---

## 398. Valid and Expressive Copulas for Irregular Multivariate Time Series

**arXiv ID:** 2605.23632 | [PDF](https://arxiv.org/pdf/2605.23632v1)

**作者:** Christian Klötergens `[一作]` (University of Hildesheim), Vijaya Krishna Yalavarthi `[通讯]` (University of Hildesheim)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5004849040)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于不规则多元时间序列（IMTS）概率预测的copula模型，结合了单变量边际的正则化流的表现力与高斯混合copula的联合依赖结构的一致性和灵活性。

**💡 创新点**

首次提出了一个在构造上保证边际一致性的IMTS copula模型，并在联合IMTS密度建模中建立了新的最先进水平。

**🔧 技术方法**

使用了高斯混合模型构建依赖结构，并结合了深度sigmoidal流（DSF）来建模单变量边际分布。

**📊 数据集**

使用了四个基准IMTS数据集进行评估，具体数据集未在摘要中列出。

**📈 对比分析**

与现有的非copula基线模型相比，模型在边际似然性上表现更好，并且在联合似然性上与TACTiS-2相当或更好。模型在所有四个数据集上均表现出最低的边际负对数似然（mNLL）。

**⚠️ 局限性**

每个Σ_j具有秩-H的离散Gram结构，因此无法表示接近满秩的依赖关系。此外，为了保证边际一致性，混合权重π仅依赖于历史数据，而不依赖于查询。

---

## 399. How Hard is it to Rig a Benchmark? A Social Choice Analysis of Leaderboard Robustness

**arXiv ID:** 2605.23628 | [PDF](https://arxiv.org/pdf/2605.23628v1)

**作者:** Polina Gordienko `[一作]` (LMU Munich), Christoph Jansen `[通讯]` (Lancaster University)

**通讯引用:** 384 | [OpenAlex ID](https://openalex.org/A5050306217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多任务基准测试在机器学习中的重要性，并探讨了基准游戏（即通过特定策略提高模型在排行榜上的排名）的问题。作者将数据集视为选民，模型视为候选人，提出了基准特定训练作为一种选举操控的形式，并证明了在Borda计数和平均胜率下，该问题是NP难的。

**💡 创新点**

创新点在于将多任务基准测试形式化为偏好聚合问题，并将基准特定训练视为对该问题的操控。此外，作者引入了实例级鲁棒性，定义了开发者必须包含在训练中的最小数据集数量，以确保目标模型在排行榜上排名第一。

**🔧 技术方法**

使用了社会选择理论中的计算复杂性方法，特别是与选举操控相关的移位贿赂问题。通过对不同聚合规则（如算术平均、媒体、平均胜率和成对多数）进行分析，评估了基准的鲁棒性。

**📊 数据集**

使用了MMLU和BIG-Bench Hard（BBH）数据集进行实验，MMLU包含22个模型和57个主题，而BBH则包含4507个模型和24个任务。

**📈 对比分析**

通过比较不同聚合规则的鲁棒性，发现平均胜率是最难操控的，MMLU和BBH的中位鲁棒性分别为44.5个主题（78%）和22个任务（92%），而算术平均和媒体的鲁棒性较低，分别为16（28%）和12（50%）。

**⚠️ 局限性**

限制在于分析模型对开发者有利，假设训练只影响目标模型，且不会降低其在其他任务上的得分。在实际情况下，泄露的数据可能只部分提高性能，并可能在其他任务上降低性能。

---

## 400. Adversarial Vulnerability Under Temporal Concept Drift: A Longitudinal Study of Android Malware Detection

**arXiv ID:** 2605.23623 | [PDF](https://arxiv.org/pdf/2605.23623v1)

**作者:** Ahmed Sabbah `[一作]` (Birzeit University), David Mohaisen `[通讯]` (University of Central Florida)

**通讯引用:** 7305 | [OpenAlex ID](https://openalex.org/A5077402873)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对Android恶意软件检测系统在跨年度、连续窗口和同年训练/测试三种部署协议下的预测性能和对抗鲁棒性进行纵向评估。

**💡 创新点**

创新点在于引入时间间隔引起的概念漂移与对抗攻击交互的量化指标（RobustDrop、ΔASR、AAF），并通过跨年度与累积窗口重训练展示漂移对鲁棒性的影响。

**🔧 技术方法**

使用了静态（权限、意图）与动态（系统调用计数）特征空间，构建了七类分类器（RF、GB、KNN、CNN、RNN、LSTM、GRU），并以逻辑回归为代理通过FGSM和SPSA生成可行的对抗样本。

**📊 数据集**

数据集为KronoDroid的2008–2020年Android应用程序集合，包含真机与模拟器两种执行环境的静态和动态特征。

**📈 对比分析**

在同年基线下，模型的准确率、宏F1和对抗准确率均高；随着训练-测试年份差距增大，准确率下降，鲁棒性下降；增量重训练能缓解但无法完全恢复鲁棒性，尤其对FGSM攻击表现明显。

**⚠️ 局限性**

主要限制在于仅考虑特征空间攻击、投影约束导致SPSA效果受限、缺乏APK级对抗验证、仅用单一数据集和代理模型，结果可能不具普适性。

---

## 401. Co-ReAct: Rubrics as Step-Level Collaborators for ReAct Agents

**arXiv ID:** 2605.23590 | [PDF](https://arxiv.org/pdf/2605.23590v1)

**作者:** Jiazheng Kang `[一作]` (Qwen Applications Business Group Of Alibaba), Guanjun Jiang `[通讯]` (Qwen Applications Business Group Of Alibaba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Co-ReAct的框架，通过使用评分标准作为推理过程中的逐步指导，改善了基于ReAct的搜索密集型多步骤推理任务的性能。

**💡 创新点**

创新点在于将评分标准从评估对象转变为推理过程中的逐步行动选择信号，并通过专门训练的评分标准生成器提供更具区分性的指导。

**🔧 技术方法**

使用了基于列表的相对策略优化（GRPO）技术来训练评分标准生成器，并在推理时扩展了ReAct的决策过程。

**📊 数据集**

使用了DeepResearchBench和SQA-CS-V2两个数据集进行评估，这些数据集涉及多轮网络搜索和基于引用的报告生成。

**📈 对比分析**

与Self-Refine、Best-of-N、Step-Back和CRITIC等方法进行比较，Co-ReAct在多个基准测试中表现出一致的性能提升，尤其是在Qwen3-14B模型上，提升幅度显著。

**⚠️ 局限性**

限制在于Co-ReAct是对固定搜索策略的增强，未对基础代理进行重训练，且评估依赖于LLM判断，可能存在已知的偏差问题。

---

## 402. An ASP-based approach to Solving General Stochastic Two-Player Games

**arXiv ID:** 2605.23705 | [PDF](https://arxiv.org/pdf/2605.23705v1)

**作者:** Yifan He `[一作]` (University of New South Wales), Michael Thielscher `[通讯]` (University of New South Wales)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种基于ASP的方法来解决具有不确定性的两人轮流GDL游戏，介绍了随机答案集编程（SQASP）以编码给定玩家在随机GDL游戏中可实现的最大获胜概率，并开发了一个基于翻译的求解器。

**💡 创新点**

创新点在于引入了SQASP语言，能够表达对抗性和随机行为，并通过将SQASP程序转换为扩展随机可满足性（XSSAT）来求解两人随机GDL游戏。

**🔧 技术方法**

使用了随机答案集编程（SQASP）和扩展随机可满足性（XSSAT）技术。

**📊 数据集**

使用了标准GDL游戏的随机变体，如井字棋和尼姆等游戏进行实验评估。

**📈 对比分析**

与前向搜索方法进行了比较，结果表明所提出的方法在小型随机游戏中具有竞争力，并且在终局评估中可能支持一般游戏玩家。

**⚠️ 局限性**

限制在于该方法仅适用于完全信息游戏，未来的工作方向是将其扩展到具有部分可观察性的随机游戏，如扑克、克里格井字棋或蒙提霍尔问题。

---

## 403. TubiFM: Unified Item, Carousel, and Search Ranking for Streaming Discovery

**arXiv ID:** 2605.23702 | [PDF](https://arxiv.org/pdf/2605.23702v1)

**作者:** Alexandre Salle `[一作]` (Tubi), Michael Tamir `[通讯]` (Tubi)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了用户故事（user story）这一统一的序列化表示，将观看、搜索和车轮（carousel）等多种行为序列化为一个可供LLM使用的token序列，并在此基础上训练一个单一的生成式排名模型TubiFM，能够同时完成物品排名、车轮排名和搜索排名；

**💡 创新点**

创新点在于将多任务的排名问题通过提示（prompt）统一到同一语言模型中，实现了跨任务共享知识、减少架构复杂度，并通过用户故事捕捉跨表面（surface）和跨容器（carousel）的交叉信号；

**🔧 技术方法**

使用了LLama 3.2 1B作为基础模型，进行自回归微调；采用了自定义token化，加入事件、表面、车轮和物品ID等域特定token；训练时加入随机遮蔽（surface/carousel mask、未知项token）和辅助目录语料；在推理时通过单前向传递对全量候选物品或车轮进行评分；

**📊 数据集**

使用Tubi内部约2000万用户的观看与搜索日志（约800M观看、66M搜索），并对所有用户的序列做截断至1024 token，形成约11B token的训练语料；

**📈 对比分析**

与任务特定的SASRec、HSTU、BM25、Qwen3 Embedding和Sentence-Transformer等基线对比；在离线评估中，TubiFM在HR@K、NDCG@K等指标上分别比最强基线提升约41%（物品）、8%（车轮）和47%（搜索）；在在线A/B测试中，搜索TVT提升3.9%，车轮TVT提升0.3%，物品TVT保持不变但显著降低p99延迟到200ms；

**⚠️ 局限性**

局限性包括：标签仅基于正向观看反馈，缺乏完整的相关性判断；在线对照组为专有系统，无法公开细节；用户故事的通用性在论文中仅在流媒体领域验证，其他业务领域需要进一步验证；

---

## 404. AMP: Arc Multi-Proposer Protocol with Bounded Inclusion Guarantees

**arXiv ID:** 2605.23677 | [PDF](https://arxiv.org/pdf/2605.23677v1)

**作者:** Daniel Cason `[一作]` (Circle), Preston Vander Vos `[通讯]` (Circle)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种在 Tendermint 共识之上实现多提议者（multi‑proposer）的协议层（Arc Multi‑Proposer Protocol），通过专门的节点收集、打包用户交易并使用投票扩展（vote extensions）来保证事务的可见性与排序。

**💡 创新点**

创新点包括：1）利用投票扩展实现“bounded inclusion”——所有被所有正确验证者证实的事务必定在下一个高度被纳入；2）将事务传播与共识决议分离，消除 mempool 的冗余广播；3）提供确定性事务排序算法，限制验证者对排序的主观控制，减少 MEV 风险；4）保持 Tendermint 的安全性与确定性最终性。

**🔧 技术方法**

技术：Tendermint BFT 共识、投票扩展（vote extensions）、分布式广播（BEB）、事务打包为 bundle、确定性排序（按优先费降序）以及在协议层中实现的事务认证与可用性检查。

**📊 数据集**

未使用具体数据集；该工作为协议设计与形式化证明，后续实验基于自研原型进行吞吐率评估。

**📈 对比分析**

与标准 Tendermint 的比较：在协议上增加了两轮额外通信（传播与投票扩展），但去除了 mempool 传播，理论上满足 f < n/3 的容错阈值；实验报告显示原型吞吐率提升约 10 倍（字节/秒）但缺乏完整基准与对比结果。

**⚠️ 局限性**

限制：1）不提供隐藏（privacy）功能；2）对动态验证者集缺乏完整支持；3）可能出现 free‑data‑availability 问题（同一事务多份存储导致存储膨胀）；4）对恶意提议者的防护依赖于应用层的事务验证；5）未完成性能评测与安全模型的完整实验验证。

---

## 405. Detecting Drunk Driving Using Off-the-Shelf Smartwatches

**arXiv ID:** 2605.23663 | [PDF](https://arxiv.org/pdf/2605.23663v1)

**作者:** Robin Deuber `[一作]` (ETH Zürich), Varun Mishra `[通讯]` (Northeastern University)

**通讯引用:** 3734 | [OpenAlex ID](https://openalex.org/A5004254348)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在真实车辆测试轨道上，利用消费级智能手表的加速度与心率变异性信号，构建并评估了两种模型用于酒精诱导驾驶员驾驶状态的检测。

**💡 创新点**

首次证明可仅通过手表传感器实现酒精驾驶检测，并在受控实验中验证模型对未见驾驶员的泛化能力。

**🔧 技术方法**

采用LASSO正则化逻辑回归结合TSFresh特征抽取，以及两塔一维卷积神经网络（late‑fusion）对加速度和自适应心率变异性序列进行端到端学习。

**📊 数据集**

使用54名受试者在三组（药物、安慰剂、对照）中完成的闭环跑道驾驶实验数据，包含呼气酒精浓度（BAC）为地面真值。

**📈 对比分析**

在留一人交叉验证下，CNN模型对任何酒精暴露（BAC>0）和超过WHO法定限值（BAC>0.05）的二分类任务分别取得AUROC约0.88/0.86，AUPRC约0.93/0.78，显著优于传统逻辑回归；对比方法显示窗口长度与多模态融合对性能有显著影响。

**⚠️ 局限性**

局限在于样本规模有限、实验环境为封闭跑道、未覆盖多种驾驶场景与车种，且模型对不同个体间的基线差异敏感，需进一步验证真实道路上的泛化与个人化校准。

---

## 406. Less Effort, Shorter Proofs: Reinforcement Learning for Security Protocol Analysis in Tamarin

**arXiv ID:** 2605.23643 | [PDF](https://arxiv.org/pdf/2605.23643v1)

**作者:** Matthias Cosler `[一作]` (CISPA Helmholtz Center for Information Security), Niklas Medinger `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种基于强化学习的Tamarin协议分析框架，自动化搜索更短、更高效的安全协议证明。

**💡 创新点**

创新在于将强化学习与Tamarin证明搜索结合，提供可复用API，并通过奖励机制驱动证据生成，显著降低人力成本。

**🔧 技术方法**

技术包括Tamarin自动机理论、强化学习（如Actor‑Critic/策略梯度）、Python接口与GPU加速。

**📊 数据集**

数据集涵盖常用安全协议测试集（Needham‑Schroeder、KSL、TLS‑RSA 等）以及作者自定义的规模更大的协议实例。

**📈 对比分析**

与Tamarin默认搜索、手工证明和其他工具（ProVerif）对比，平均证明长度缩短 30‑50%，搜索时间下降 40‑70%，且成功率保持在 95% 以上。

**⚠️ 局限性**

局限包括对极大规模协议的可扩展性不足、奖励函数设计需手工调优，以及对非标准协议格式的适配仍有限。

---

## 407. Learning Dynamic Stability Landscapes in Synchronization Networks

**arXiv ID:** 2605.23708 | [PDF](https://arxiv.org/pdf/2605.23708v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 408. DiLaDiff: Distilled Latent-Augmented Diffusion for Language Modeling

**arXiv ID:** 2605.23605 | [PDF](https://arxiv.org/pdf/2605.23605v1)

**作者:** Jean-Marie Lemercier `[一作]` (Nvidia), Ante Jukić `[通讯]` (Nvidia)

**通讯引用:** 1467 | [OpenAlex ID](https://openalex.org/A5063860258)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种混合连续-离散扩散语言模型 DiLaDiff，利用自动编码器生成语义一致的潜在空间并作为条件，随后用连续潜在扩散引导离散解码器，实现多词并行生成并提升文本质量。

**💡 创新点**

创新点包括：① 在连续潜在空间捕获句子级词间相关性；② 对潜在扩散轨迹进行自蒸馏（MeanFlow）得到少步生成模型 DiLaDiff；③ 在解码器初始化时使用预训练离散扩散模型，增强可控性和训练稳定性。

**🔧 技术方法**

采用的技术有：自动编码器训练（BERT特征+Perceiver编码），连续潜在扩散（DDPM/ODE），一致性模型/MeanFlow蒸馏，温度/核采样等离散解码技术。

**📊 数据集**

使用 OpenWebText 数据集进行无监督语言建模实验。

**📈 对比分析**

与 MDLM、DUO 等基准对比，LaDiff 在不牺牲质量的前提下可将生成速度提升约 7 倍；DiLaDiff 在仅 5 步潜在扩散下，生成质量（GenPPL/MAUVE）接近 LaDiff 教师，且显著优于传统少步方法。

**⚠️ 局限性**

局限性包括：未蒸馏离散解码器，导致极低步数下性能仍不及最新方法；潜在空间正则化与扩散调度仍依赖经验；少步蒸馏效果尚未完全达到教师水平。

---

## 409. Preisach Attention: A Hysteretic Model of Sequential Memory

**arXiv ID:** 2605.23603 | [PDF](https://arxiv.org/pdf/2605.23603v1)

**作者:** Piotr Frydrych `[一作]` (Warsaw University of Technology), Piotr Frydrych `[通讯]` (Warsaw University of Technology)

**通讯引用:** 109 | [OpenAlex ID](https://openalex.org/A5012740592)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的序列建模架构——Preisach注意力层（PAL），该架构基于经典的Preisach滞后算子。

**💡 创新点**

PAL用学习的激活和去激活阈值参数化的二元继电器替代了softmax注意力机制，具有O(1)的深度且是图灵完备的，且在处理长记忆和弱位置依赖的任务中表现出色。

**🔧 技术方法**

使用了Preisach滞后算子作为基础，结合了多头注意力机制和深度学习中的前馈神经网络。

**📊 数据集**

论文中没有具体提到使用的数据集，但强调了PAL在长时间序列和弱位置依赖任务中的优势。

**📈 对比分析**

与标准注意力机制相比，PAL在O(n log n)的总推理成本下表现出更高的效率，而标准注意力机制的成本为O(n^2)。

**⚠️ 局限性**

PAL的局限性在于其无法执行随机访问检索，并且在处理某些特定类型的函数时可能不如标准变换器有效。

---

## 410. Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts

**arXiv ID:** 2605.23597 | [PDF](https://arxiv.org/pdf/2605.23597v1)

**作者:** Shivam Chourasia `[一作]` (Dream Sports), Nilesh Patil `[通讯]` (Dream Sports)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于两阶段课程学习的结构引导实体解析框架SGER，专门解决印度多语言多文化场景下的人名匹配问题。

**💡 创新点**

创新点在于先用LLM学习姓名结构，再进行二分类，解耦结构学习与匹配任务，并通过多种增广策略和两阶段微调显著提升准确率。

**🔧 技术方法**

使用Meta Llama 3 8B作为基模型，采用LoRA微调，配合结构化JSON预训练与二分类微调；推理阶段使用vLLM实现高吞吐。

**📊 数据集**

使用来自Dream11 250 M用户KYC的50k+实际姓名对数据，及约10k个带结构标签的印度姓名作为结构预训练集。

**📈 对比分析**

与Levenshtein、BERT、GPT‑4o以及单阶段Llama微调等基线对比，SGER在50k测试集上达99.02%准确率、F1 0.994，明显优于所有基线。

**⚠️ 局限性**

局限在于对极端多重噪声和音素歧义仍有误判；模型仅针对姓名，无法处理全字段匹配；且依赖大量标注数据与本地部署资源。

---

## 411. Synthetic Sources?: Auditing Generative Search Engine Citations for Evidence of AI-Generated Sources

**arXiv ID:** 2605.23684 | [PDF](https://arxiv.org/pdf/2605.23684v1)

**作者:** Mowafak Allaham `[一作]` (Northwestern University), Nicholas Diakopoulos `[通讯]` (Northwestern University)

**通讯引用:** 8722 | [OpenAlex ID](https://openalex.org/A5079222963)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对ChatGPT、Copilot、Gemini和Perplexity四款生成式搜索引擎在政治、健康与环境主题下的引用来源进行了审计。

**💡 创新点**

首次系统评估了这些引擎引用的AI生成内容比例，并揭示了来源域的浓缩分布。

**🔧 技术方法**

使用Playwright抓取引用列表，利用Pangram文本检测工具识别AI生成内容，并对数据进行统计分析。

**📊 数据集**

采样自Search Arena（政治、健康）和Climate Q&A（环境）共712条查询，生成约2,848条回答。

**📈 对比分析**

通过对比Pangram与GPTZero的误检率，采用Pangram得出约16%的引用来源为AI生成；该比例在各引擎和主题间呈现差异。

**⚠️ 局限性**

研究受限于仅英文、美国背景、单轮交互、未处理非文本媒体及对检测工具泛化能力的不足。

---

## 412. AI at the Front Lines of Platform Governance: Using LLMs to Support Illegal Content Reporting under the Digital Services Act

**arXiv ID:** 2605.23676 | [PDF](https://arxiv.org/pdf/2605.23676v1)

**作者:** Marie-Therese Sekwenz `[一作]` (Delft University of Technology), Ujwal Gadiraju `[通讯]` (Delft University of Technology)

**通讯引用:** 4014 | [OpenAlex ID](https://openalex.org/A5038081564)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLM）如何支持用户在数字服务法（DSA）下报告非法内容的过程。通过对450名参与者进行的控制实验，比较了三种不同的AI辅助条件对报告质量的影响。

**💡 创新点**

创新点在于引入了评估性AI（Evaluative AI），该方法提供多种法律选项的支持和反对论据，而不是仅仅推荐一个选项，从而促进用户的深思熟虑和决策质量。

**🔧 技术方法**

使用了大型语言模型（LLM）作为AI助手，分别实现了传统的可解释AI（XAI）和评估性AI的对比。

**📊 数据集**

实验使用了基于德国刑法的模拟社交媒体内容，参与者需要在此环境中选择法律类别并提交报告。

**📈 对比分析**

与传统的可解释AI相比，评估性AI在AI错误情况下显著提高了报告的准确性，减少了错误分类的距离。然而，在AI输出正确时，两者的准确性相似，但评估性AI的决策时间较长。

**⚠️ 局限性**

本研究的局限性在于使用了定制的报告界面和有限的法律条款，可能未能完全反映真实世界的复杂性。此外，研究未能评估长期行为和用户适应性。

---

## 413. Relevant Walk Search for Explaining Graph Neural Networks

**arXiv ID:** 2605.23673 | [PDF](https://arxiv.org/pdf/2605.23673v1)

**作者:** Ping Xiong `[一作]` (Technische Universitaet Berlin), Shinichi Nakajima `[通讯]` (Technische Universitaet Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图神经网络（GNN）的相关步搜索方法，以提高GNN的可解释性，特别是通过多项式时间算法来识别最相关的步，从而解决了GNN-LRP在深度网络中计算复杂度指数级增长的问题。

**💡 创新点**

创新点在于提出了两种新算法：精确的神经元级搜索（EMP-neu）和近似的节点级搜索（AMP-ave），这两种算法都能在多项式时间内找到最相关的步，显著提高了GNN-LRP的可扩展性和实用性。

**🔧 技术方法**

使用了最大乘积消息传递算法（max-product message passing）来实现相关步的搜索，解决了传统方法的计算复杂度问题。

**📊 数据集**

使用了多个基准数据集，包括BA-2motif、MUTAG、Mutagenicity和Graph-SST2，以及一个基于感染传播模型生成的感染数据集，展示了算法的有效性和可扩展性。

**📈 对比分析**

与现有的GNN可解释性方法（如GNNExplainer和PGExplainer）相比，提出的方法在准确性和计算效率上表现优越，尤其是在处理大规模数据集时，AMP-ave的计算时间显著低于GNN-LRP的指数级计算时间。

**⚠️ 局限性**

限制在于AMP-ave作为近似方法，其准确性尚未得到理论保证，仅通过实证评估支持。此外，该方法仅适用于能够定义相关传播的模型，可能不适用于某些通用GNN架构。

---

## 414. RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video

**arXiv ID:** 2605.23672 | [PDF](https://arxiv.org/pdf/2605.23672v1)

**作者:** Chenyu Wu `[一作]` (Harvard University), Hanspeter Pfister `[通讯]` (Harvard University)

**通讯引用:** 34302 | [OpenAlex ID](https://openalex.org/A5043151044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为RiGS的框架，用于从单目视频重建动态3D场景，能够同时捕捉多种时间尺度的运动。

**💡 创新点**

创新点在于引入了三种类型的高斯原语（静态、高刚性和瞬态），并通过对象级动态掩码实现静态和动态区域的分解，结合场景流指导进行联合优化。

**🔧 技术方法**

使用了高斯点云表示法，结合场景流指导的优化方法。

**📊 数据集**

使用了Nvidia动态场景数据集和Dycheck iPhone数据集进行实验评估。

**📈 对比分析**

与现有方法相比，RiGS在多个基准测试中表现出色，尤其在新视角合成任务中，展示了更高的重建质量和时间一致性。

**⚠️ 局限性**

限制在于对复杂快速运动的建模能力可能仍然不足，尤其是在极端动态场景中。

---

## 415. Learning Through Noise: Why Subliminal Learning Works and When It Fails

**arXiv ID:** 2605.23645 | [PDF](https://arxiv.org/pdf/2605.23645v1)

**作者:** Vincent C. Brockers `[一作]` (Max Planck Institute for Dynamics and Self Organization), Viola Priesemann `[通讯]` (Max Planck Institute for Dynamics and Self Organization)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在受控的 MLP-MNIST 设定下，探究教师-学生模型在无感知噪声输入上通过输出头实现的 subliminal 学习机制

**💡 创新点**

发现隐藏层初始化不必相同，输出头的兼容性（aux head 与 class head）是实现 subliminal 学习的关键条件

**🔧 技术方法**

采用分离输出头的 MLP、随机噪声数据、MSE 训练 aux head，并对比教师与学生的分类性能

**📊 数据集**

主要使用 MNIST（以及 EMNIST 子集）作为任务数据集

**📈 对比分析**

通过实验表明，只要输出头兼容，学生可在噪声训练后达到接近教师的分类准确率；当输出头不兼容或维度过高时，性能显著下降

**⚠️ 局限性**

局限在于理论尚未解释所有现象（如维度与 aux head 失配导致的准确率波动），且结果仅在受控 MLP 场景验证，是否能推广至 LLM 或更复杂架构尚未确认

---

## 416. To Overlay or to Customize? Revisiting Architectural Choices in Heterogeneous Systems

**arXiv ID:** 2605.23630 | [PDF](https://arxiv.org/pdf/2605.23630v1)

**作者:** Xingzhen Chen `[一作]` (Brown University), Peipei Zhou `[通讯]` (Brown University)

**通讯引用:** 2080 | [OpenAlex ID](https://openalex.org/A5063866156)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对比了FPGA上基于overlay与定制bitstream两种加速架构在自动驾驶系统中的部署性能。

**💡 创新点**

从系统部署视角提出模型切换频率、重配置延迟等真实因素影响的综合评估框架，并通过敏感性研究阐明重配置速度与overlay能力提升对架构选择的转折点。

**🔧 技术方法**

使用EDF调度模拟、基于AMD Versal VCK190平台的bitstream重配置测量、DeiT/MLP‑Mixer/PointNet大/小模型的周期任务模型，结合overlay架构（FILCO/DORA）与定制架构（CHARM/SSR）实现。

**📊 数据集**

采用Autoware感知栈真实任务设置，并使用DeiT、MLP‑Mixer、PointNet等模型的公开权重与训练集。

**📈 对比分析**

通过计算加速器忙碌比、任务完成率与时延堆叠，结果显示在现有20 ms重配置延迟下overlay可在所有设置下通过，定制化在部分设置下失效；当重配置降至≈1 ms或overlay吞吐提升100%时，定制化/overlay优势可翻转。

**⚠️ 局限性**

实验仅覆盖特定硬件与六个模型，未考虑多重预占、上下文保存成本；重配置时间与模型大小的假设可能不适用于其他FPGA平台；overlay性能提升假设为理想化。

---

## 417. Benchmarking Google Embeddings 2 against Open-Source Models for Multilingual Dense Retrieval and RAG Systems

**arXiv ID:** 2605.23618 | [PDF](https://arxiv.org/pdf/2605.23618v1)

**作者:** Stefano Cirillo `[一作]` (University of Salerno), Giandomenico Solimando `[通讯]` (University of Salerno)

**通讯引用:** 266 | [OpenAlex ID](https://openalex.org/A5021217319)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对谷歌Embedding 2（GE2）与五个开源多语言检索模型在多语言密集检索和RAG系统中的性能进行了系统基准评测。

**💡 创新点**

创新点在于将GE2的长上下文窗口和任务类型异步编码与多种检索模型进行公平对比，并首次揭示LaBSE在检索任务中的显著劣势；同时结合不同分块策略、上下文长度和延迟分析，给出模型选择与SLA的实用决策框架。

**🔧 技术方法**

使用了双编码器架构、对比学习与指令前缀训练、异步任务类型条件化、三种分块策略（固定、滑动窗口、语义分块）、FAISS近似检索、CPU延迟测量等技术。

**📊 数据集**

评测数据集包括BEIR四个子集（FiQA‑2018、NFCorpus、SciFact、TREC‑COVID）和自行构建的意大利语RAG基准IT‑RAG‑Bench（3,200条短篇段落）。

**📈 对比分析**

通过零样本nDCG@10、Recall@k和MRR等指标对比，GE2在BEIR上平均nDCG@10为0.638、在IT‑RAG‑Bench为0.282，性能最优，但延迟中位数达231 ms；mE5‑L在意大利语任务上与GE2几乎无差别（0.282 vs. 0.279），且延迟仅约31 ms；LaBSE在所有任务中表现最差。

**⚠️ 局限性**

局限性包括：IT‑RAG‑Bench为短段落的合成数据，可能不适用于长文档；仅评测了四个BEIR子集；延迟测量仅在CPU上完成，未考虑GPU加速和批处理；缺乏自举置信区间，对小差异的统计显著性评估不足。

---

## 418. When Youth Enter the Algorithmic Wild: Discovering and Understanding Potentially Harmful Teen Videos on Douyin and Kwai

**arXiv ID:** 2605.23598 | [PDF](https://arxiv.org/pdf/2605.23598v1)

**作者:** Shaoxuan Zhou `[一作]` (University of Science and Technology of China), Xianghang Mi `[通讯]` (Monash University)

**通讯引用:** 651 | [OpenAlex ID](https://openalex.org/A5047482025)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统分析了抖音和快手平台中青少年视频的内容与传播特点，构建了潜在有害视频的发现与理解框架。

**💡 创新点**

首次将算法推荐机制的放大效应与内容有害性评估相结合，提出“算法野性”指标量化算法对有害内容的放大程度。

**🔧 技术方法**

采用多模态深度学习模型（视频视觉特征 + 文本描述）进行有害内容分类，并结合因果推断方法评估推荐算法的影响。

**📊 数据集**

使用自采集的 4 万条抖音与快手青少年视频数据集，手工标注了有害与非有害标签。

**📈 对比分析**

与传统文本分类器、单模态 CNN 等基线模型对比，所提模型在 F1 值上提升约 8%（达到 0.78），精确率与召回率均显著优于基线。

**⚠️ 局限性**

主要局限在于数据集覆盖范围有限，标签标注具有主观性，且研究仅聚焦于中国两大短视频平台，结果难以直接推广至其他平台。

---

## 419. Cost-Effective Model Evaluation with Meta-Learning

**arXiv ID:** 2605.23595 | [PDF](https://arxiv.org/pdf/2605.23595v1)

**作者:** Trinh Pham `[一作]`, Thanh Tam Nguyen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为MetaEvaluator的框架，用于在没有标签的情况下快速评估新模型的性能，适用于多种架构和模态。

**💡 创新点**

MetaEvaluator是首个能够在完全未标记的数据集上评估新模型的模型无关框架，显著降低了评估成本并提高了准确性。

**🔧 技术方法**

使用了元学习技术，通过参考模型池获取可转移的初始化，从而实现对新模型的准确评估。

**📊 数据集**

使用了MetaDataset，这是一个大规模、系统构建的模型-迁移对数据集，涵盖了Text2SQL和图像分类任务。

**📈 对比分析**

与传统方法相比，MetaEvaluator在评估新模型时表现出更低的均方误差（MAE），并且评估速度更快，能够支持快速的模型选择和部署。

**⚠️ 局限性**

限制在于MetaEvaluator的性能依赖于参考模型池的质量和多样性，且在面对极端的分布变化时可能仍然存在挑战。

---

## 420. OnePred: Next-Query Prediction via Recursive Intent Memory in Multi-Turn Conversations

**arXiv ID:** 2605.23668 | [PDF](https://arxiv.org/pdf/2605.23668v1)

**作者:** Jiangwang Chen `[一作]` (Tsinghua University), Guanjun Jiang `[通讯]` (Qwen Applications Business Group of Alibaba)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OnePred 模型，利用递归意图记忆预测多轮对话中的下一条用户查询，并构建了 NQP-Bench 作为基准。

**💡 创新点**

创新点在于：①递归文本记忆将跨轮意图压缩为有限长度的“意图链”；②两阶段强化学习（先全历史预测后记忆压缩）解决记忆内容与预测性能的循环依赖；③显著降低每轮 token 消耗（最高可达 22 倍）。

**🔧 技术方法**

技术方法包括：递归文本记忆、两阶段 RL 训练（Full-History RL + Agentic Memory RL）、GRPO 策略优化、LLM 评判器打分、以及 NQP-Bench 基准构建与评估。

**📊 数据集**

使用数据集：NQP-Priv（内部部署日志）、NQP-Wild（WildChat 公开语料）、NQP-Share（ShareChat 公开语料）。

**📈 对比分析**

与 Current-turn、Full-history、Sliding‑Window 等基线对比，OnePred 在 Gemini‑3.1‑Pro、Base Qwen 与 RL‑trained Qwen 三种模型规模下，在 Judge 与 Human 评估上均优于 Baseline，平均提升约 1.6–2.6 分；token 量在 Full-history 的基础上降低至约 1/22，且在长对话中优势更为明显。

**⚠️ 局限性**

局限性包括：递归记忆压缩可能丢失细节信息（如具体数字、短暂话题）；仅在 Qwen3 系列和 Gemini‑3.1‑Pro 上实验，未验证其他模型家族；LLM 在数据标注与评估中可能存在共性偏差。

---

## 421. A graph-based analysis of semantic types and coercion in contextualized word embeddings

**arXiv ID:** 2605.23710 | [PDF](https://arxiv.org/pdf/2605.23710v1)

**作者:** Long Chen `[一作]` (Heinrich Heine University), Deniz Ekin Yavas `[通讯]` (Heinrich Heine University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了基于词嵌入的图模型，对十种语义类型的名词实例进行语料标注，并通过邻居类型概率（NTP）与邻居类型熵（NTE）这两项度量分析句子中的词义与上下文类型的匹配与不匹配，探讨强制性转义（coercion）现象。

**💡 创新点**

创新点在于：①首次将词义图与语义类型相结合，使用 NTP 与 NTE 两个图度量系统化评估词义与上下文类型的分布；②引入了 sense-enhanced BERT（融合 WordNet 超级词义的 BERT 变体）和 masked 版本的嵌入，系统比较四种嵌入在反映词义信息与上下文信息上的差异；③通过图结构揭示不同句子类型（匹配、强制转义、其他不匹配、无约束）在邻居类型分布上的可区分性，为自动检测强制转义提供新方法。

**🔧 技术方法**

技术方法包括：①使用 BERT 与 sense-enhanced BERT 提取词实例的上下文词嵌入；②对词嵌入做层平均；③构建 k-近邻图（k=10），仅连接不同词的实例；④计算邻居类型概率（NTP）与邻居类型熵（NTE）；⑤对四种图（G_b、G_s、G_mb、G_ms）进行定量比较，并用 Mann‑Whitney U 检验统计显著性。

**📊 数据集**

数据集：从 BookCorpus 随机抽取 10-16 句子对应 5-10 个常用名词，覆盖十个语义类型（animal、artifact、activity、food、human、info、location、mood、process、state）。手工标注每个实例的词义类型（lt）与上下文类型（ct），并按匹配/强制转义/其他不匹配/无约束四类进行标注。

**📈 对比分析**

比较方法：对四种图分别计算各句子类型的 NTMR_L、NTMR_C 以及 NTE，采用均值与标准差报告，并通过 Mann‑Whitney U 检验两两句子类型间的显著性差异。结果显示：sense‑enhanced BERT 图（G_s）在反映词义类型方面优于普通 BERT；masked 图（G_mb、G_ms）在捕捉上下文类型方面表现更好；匹配句子与强制转义、无约束句子在 NTE 上可区分，提供了检测强制转义的可能路径。

**⚠️ 局限性**

局限性：①样本规模相对有限，且仅覆盖十种语义类型；②对多义词的处理仅选取单义例子，未充分探究多义词在强制转义中的角色；③masked 图虽去除词义信息，但仍可能保留词形与共现模式的干扰；④实验仅基于 BERT 与 sense‑enhanced BERT，缺乏对更大模型或跨语言的验证；⑤缺乏针对非强制转义的其他不匹配（隐喻、转喻）实例的深入分析。

---

## 422. ChartFI: Benchmarking Faithfulness and Insightfulness of Chart Descriptions from Multimodal Large Language Models

**arXiv ID:** 2605.23694 | [PDF](https://arxiv.org/pdf/2605.23694v1)

**作者:** Fen Wang `[一作]` (Fudan University), Siming Chen `[通讯]` (Fudan University)

**通讯引用:** 4314 | [OpenAlex ID](https://openalex.org/A5050391600)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Chart Faithfulness and Insightfulness Benchmark (ChartFI-Bench)，用于评估多模态大语言模型生成的图表描述的真实性和洞察力。

**💡 创新点**

创新点在于构建了一个高质量的基准数据集，包含896对图表和描述，并提出了四维评估框架来量化描述的真实性和洞察力。

**🔧 技术方法**

使用了多模态大语言模型（MLLMs）进行图表描述生成和评估，特别是GPT-5.4和Gemini-3-Flash等模型。

**📊 数据集**

数据集来源于arXiv，经过系统筛选和人工验证，最终获得896对视觉复杂且语义丰富的图表和描述。

**📈 对比分析**

与现有基准的比较显示，ChartFI-Bench在语义覆盖和描述质量上具有优势，评估结果揭示了当前模型的共同弱点，表明它们在生成高质量图表描述方面仍存在挑战。

**⚠️ 局限性**

限制在于当前模型仍然容易产生幻觉，且在多变量合成和领域知识的应用上表现不佳，未来需要进一步改进评估指标和数据集的多样性。

---

## 423. Multi-User MIMO with Rotatable Antennas and IRS: Joint Antenna Boresight and IRS Orientation Design

**arXiv ID:** 2605.23683 | [PDF](https://arxiv.org/pdf/2605.23683v1)

**作者:** Guoying Zhang `[一作]` (Shanghai Jiao Tong University), Wen Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 112546 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了一种智能反射面（IRS）辅助的多用户系统，其中基站（BS）配备可旋转天线（RA），并且IRS可以调整面板方向。通过联合优化接收波束形成、IRS相位偏移、BS天线瞄准方向和IRS面板方向，提出了一个求和速率最大化问题。

**💡 创新点**

创新点在于提出了一种协调的双旋转框架，能够同时优化BS的天线瞄准和IRS的面板方向，以提高多用户MIMO上行通信的性能。

**🔧 技术方法**

使用了交替优化算法，其中接收波束形成通过最小均方误差（MMSE）组合器以封闭形式更新，IRS相位偏移通过分数规划（FP）优化，BS天线瞄准和IRS面板方向通过投影梯度方法更新。

**📊 数据集**

使用了模拟数据，验证了所提方案在不同发射功率、BS天线数量、用户数量和IRS-BS距离下的有效性。

**📈 对比分析**

与固定方向和单旋转基准方案相比，所提方案在不同条件下均表现出显著的求和速率增益，模拟结果表明协调双旋转设计能够显著提高反射信道的平均功率。

**⚠️ 局限性**

限制在于所提出的求和速率最大化问题是高度非凸的，且在近场条件下IRS面板方向与BS天线瞄准之间存在耦合，导致优化过程复杂。

---

## 424. ExpOS: Explainable Open-Surgery Skills Assessment Using 3D Hand Reconstruction

**arXiv ID:** 2605.23653 | [PDF](https://arxiv.org/pdf/2605.23653v1)

**作者:** Roi Papo `[一作]` (Technion Israel Institute of Technology), Shlomi Laufer `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 ExpOS，一个利用 3D 手部重建和时间注意力机制进行开放式外科技能评估的可解释框架。

**💡 创新点**

创新点在于结合弱监督的时间重要性学习与可解释的全局运动统计，提供多层次可解释性；同时使用 MS‑TCN++ 与多头注意力池化联合 SHAP 进行解释。

**🔧 技术方法**

技术包括 YOLO 工具/手部检测、WiLoR 3D 手部重建、MS‑TCN++ 时序特征提取、Multi‑head Attention Pooling、Soft Ordinal Regression 损失、以及 SHAP 进行特征归因。

**📊 数据集**

使用 221 条来自 77 名医学生在三种开放手术任务（缝合、打结、筋膜闭合）中的 4K RGB 视频数据集。

**📈 对比分析**

与专家评分通过 Pearson r、RMSE、MAE、R² 进行比较；筋膜闭合任务 r=0.778、R²=0.74；打结 r=0.678、R²=0.649；缝合 r=0.622、R²=0.579，整体表现良好。

**⚠️ 局限性**

局限性包括对 3D 重建的高度依赖，遮挡或视角不佳会导致性能下降；样本规模相对较小；仅在模拟器任务上验证，缺乏多样化临床数据。

---

## 425. CachePrune: Privacy-Aware and Fine-Grained KV Cache Sharing for Efficient LLM Inference

**arXiv ID:** 2605.23640 | [PDF](https://arxiv.org/pdf/2605.23640v1)

**作者:** Guanlong Wu `[一作]` (SUSTech), Yinqian Zhang `[通讯]` (SUSTech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种隐私感知的KV缓存共享机制，允许在多用户请求中实现细粒度的KV条目重用，以提高大语言模型的推理效率。

**💡 创新点**

创新点在于打破了隐私与效率之间的权衡，通过细粒度的选择性KV缓存共享，避免了敏感信息的泄露，同时保留了大部分的重用潜力。

**🔧 技术方法**

使用了基于令牌的缓存管理技术，结合了自适应算法和滚动哈希检索算法，以高效地识别和检索可重用的KV段。

**📊 数据集**

在三个数据集上进行了评估，包括QASPER、NarrativeQA和QMSum，这些数据集提供了人类标注的真实数据。

**📈 对比分析**

与现有的最先进方法相比，提出的方法在消除通过KV缓存重用的直接泄露的同时，将TTFT减少了4.5倍，缓存命中率提高了44%。

**⚠️ 局限性**

局限性在于隐私检测工具的准确性可能影响系统的整体性能，错误标记可能导致敏感信息的泄露或上下文泄露。

---

## 426. AGDES: Automatic Generation of Dependent Event Sequences

**arXiv ID:** 2605.23808 | [PDF](https://arxiv.org/pdf/2605.23808v1)

**作者:** Alexander Obeid Guzman `[一作]` `[通讯]` (University of Grenoble Alpes), Alexander Obeid Guzman (University of Grenoble Alpes)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于确定性有限自动机（DFA）和有限状态转导器（transducer）的工具，用于自动生成具有因果依赖关系的事件序列，支持自定义状态数、字母表大小、过渡数、噪声类型和水平，并能在生成的DAG中构建线性因果链；

**💡 创新点**

创新点在于将DFA与转导器结合，提供高度可配置的序列生成框架，并在生成过程中加入可调噪声机制，以模拟真实异步事件系统中的噪声与误差；

**🔧 技术方法**

核心技术包括DFA和转导器的随机生成、skew-normal分布参数采样、噪声注入（插入/删除/替换）、以及按需生成随机单词；

**📊 数据集**

本文未使用公开数据集，而是提供了自定义生成器，可直接产生训练/基准数据；

**📈 对比分析**

文章未给出具体实验比较或性能评估；仅描述了工具的使用示例和可能的未来扩展方向；

**⚠️ 局限性**

局限性包括：仅支持线性DAG（链式因果），无法直接处理多输入转导器；转导器仅为确定性，无法调节输出与输入的依赖强度；缺乏对生成数据质量与真实因果图一致性的定量评估。

---

## 427. UniSpike: Accelerating Spiking Neural Networks on Neuromorphic Systems via Eliminating Address Redundancy

**arXiv ID:** 2605.23796 | [PDF](https://arxiv.org/pdf/2605.23796v1)

**作者:** Qinghui Xing `[一作]` (Zhejiang University), Gang Pan `[通讯]` (Zhejiang University)

**通讯引用:** 472964 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种硬件‑软件协同的通信方案UniSpike，通过在目标核心视角对突触事件进行调度、运行时轻量级包组装以及目标感知的SNN划分，消除了多核神经形态系统中的地址冗余；

**💡 创新点**

创新点在于：①从目标地址角度重新设计突触事件调度机制；②引入TS Manager与位图式Packet Generator实现运行时地址合并；③设计目标感知的SNN划分算法，最大化同一核心内后突触地址的重叠；

**🔧 技术方法**

采用硬件模块（TS Manager、Packet Generator）、位图与AND运算进行地址合并；软件层面实现检查表生成、Hilbert曲线空间填充与随机段交换（SSS）划分算法；系统仿真使用HeteroGarnet NoC、NeuroSim面积评估；

**📊 数据集**

神经科学SNN（Brunel、Vogels，LIF、Izhikevich、AdEx模型）以及深度学习SNN（ResNet18、VGG11、SpikFormer、SDT、SpikingBERT、SpikeBERT）在视觉与NLP数据集上；

**📈 对比分析**

与传统神经元中心通信、四种多路复用（multicast）方法以及包压缩技术进行对比。UniSpike平均实现1.93×的网络流量压缩、1.77×的速度提升和1.50×的能效提升，硬件面积增加约10.6%；

**⚠️ 局限性**

限制包括：只针对地址冗余，未能同时解决负载不平衡和功耗的其他来源；需要额外的硬件开销和划分算法的调优；在极大规模或高度稀疏突触网络中地址合并机会有限；对多核交互模式的适应性仍待进一步验证。

---

## 428. NLG Evaluation: Past, Present, Future

**arXiv ID:** 2605.23715 | [PDF](https://arxiv.org/pdf/2605.23715v1)

**作者:** Ehud Reiter `[一作]` (University of Aberdeen), Ehud Reiter `[通讯]` (University of Aberdeen)

**通讯引用:** 9842 | [OpenAlex ID](https://openalex.org/A5005939553)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文系统梳理了1990年至2026年自然语言生成(NLG)评估方法的演进，并对未来十年（2036年）评估方向进行预测与展望。

**💡 创新点**

创新点在于将历史评估技术与新兴挑战（如LLM-as-Judge、跨学科安全与影响评估）结合，提出综合评估框架，强调从量化到质性、影响与安全的全维度评估。

**🔧 技术方法**

主要使用文献综述、案例分析以及对比研究的技术，讨论了传统指标（BLEU/ROUGE）、共享任务、人类评估、LLM-as-Judge、风险评估与跨学科评估方法。

**📊 数据集**

综述引用了多种评估数据集与任务，例如GREC共享任务、BLEU/ROUGE指标、医学安全评估数据、LLM内部测试集等，但并未进行新的数据实验。

**📈 对比分析**

通过比较不同年代的评估方法，论文指出传统的指标在LLM时代已失效，LLM-as-Judge在某些语义与语用任务上可取得较高准确性，但仍需验证其有效性；对比分析展示了评估方法在可靠性、可复制性与商业偏见方面的差异。

**⚠️ 局限性**

主要局限包括：综述受限于公开文献与案例，缺乏统一标准导致评估方法碎片化；对LLM-as-Judge的有效性缺乏系统验证；评估数据集易受污染与商业利益影响，复制性差；以及对真实世界影响与安全评估的实证研究仍不足。

---

## 429. It's the humans, not the data: Geopolitical bias in LLMs originates in post-training, amplified by the language of the prompt

**arXiv ID:** 2605.23825 | [PDF](https://arxiv.org/pdf/2605.23825v1)

**作者:** Stuart Bladon `[一作]` (Duke University), Brinnae Bent `[通讯]` (Duke University)

**通讯引用:** 2387 | [OpenAlex ID](https://openalex.org/A5074627930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对七家开放权重LLM的预训练模型和后训练模型进行配对强制选择测试，评估其在28个国家对峙情景下的地缘政治偏好。

**💡 创新点**

首次将跨实验室、跨语言的强制选择情景探针与首词兼容性校正相结合，发现后训练阶段是导致模型偏见的主要来源，而非预训练数据。

**🔧 技术方法**

采用多场景双问题对照、首词概率求和、共性过滤、假设中立前缀、交叉提示实验、虚构国家替换和自由生成验证等技术。

**📊 数据集**

使用包含79种两国地缘政治情景的探针库（英、法、中翻译），配合七个7–9B参数开放权重模型（Mistral、LLaMA、Gemma、Qwen、Baichuan、Yi、GLM）。

**📈 对比分析**

对预训练基线与后训练变体在每个情景、语言、国家对的log-odds偏差进行统计检验（t检验、二项检验），结果显示后训练显著提升或改变模型偏好，最大幅度超过18倍。

**⚠️ 局限性**

局限包括样本量有限（仅7个模型家族）、probe有效性依赖首词合规率、翻译引入循环偏差、仅覆盖7–9B参数范围、未分离后训练中的SFT、RLHF等具体步骤。

---

## 430. Hierarchical Concept Geometry in Language Models Emerges from Word Co-occurrence

**arXiv ID:** 2605.23821 | [PDF](https://arxiv.org/pdf/2605.23821v1)

**作者:** Andres Nava `[一作]` (Johns Hopkins University), Matthieu Wyart `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于共现统计的分布式理论，阐释超纲（hypernymy）关系如何在词向量空间中形成层次化几何结构，并在word2vec与Gemma 2B LLM的词嵌入中进行验证。

**💡 创新点**

创新点在于将词在WordNet层次距离与共现频率的经验假设转化为Gram矩阵的谱结构预测，说明主特征先分离粗层级，再逐步细化子分支；并证明这种“层次分裂几何”自然出现在预训练模型中，无需专门设计层次正交机制。

**🔧 技术方法**

使用的技术包括：共现概率矩阵的PMI变换、线性代数中的谱分解、基于哈瓦尔小波的多分辨率基底、正向衰减核（指数模型）拟合、子空间对齐度量（top‑k eigenspace alignment）、词向量白化处理、以及概念向量正交诊断。

**📊 数据集**

所使用的数据集：WordNet词典的超纲层次结构、对应词共现统计（用于构造word2vec的M*矩阵）、Gemma 2B LLM的词汇表与对应的未嵌入（unembedding）向量。

**📈 对比分析**

比较方法：计算理论Gram矩阵与实证Gram矩阵的主子空间对齐度量，并与随机置换基准（shuffled-label）进行对照；结果显示在word2vec和Gemma两者上，主子空间对齐显著高于基准，验证了理论预测的有效性。

**⚠️ 局限性**

限制：理论直接适用于基于共现的embedding（如word2vec），对Transformer训练动态尚未建模；未考虑多义词的上下文表示；假设Gram矩阵半正定/正定的理想化条件在实际数据中可能被噪声和负值破坏。

---

## 431. Advanced AI Service Provisioning in O-RAN through LLM Engine Integration

**arXiv ID:** 2605.23809 | [PDF](https://arxiv.org/pdf/2605.23809v1)

**作者:** Seyed Bagher Hashemi Natanzi `[一作]` (Worcester Polytechnic Institute), Vijay K. Shah `[通讯]` (North Carolina State University)

**通讯引用:** 1047 | [OpenAlex ID](https://openalex.org/A5083496212)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种双脑架构，结合了大型语言模型（LLM）和专用机器学习引擎NeuralSmith，以自动化O-RAN中的AI服务提供过程。

**💡 创新点**

创新点在于将LLM用于意图解析和代码生成，而将实时控制决策交给轻量级的机器学习分类器，从而提高了效率和准确性。

**🔧 技术方法**

使用了大型语言模型（如Llama-3.1-8B）作为指挥器，以及NeuralSmith作为机器学习引擎。

**📊 数据集**

在一个容器化的O-RAN 5G SA测试平台上进行了实验，使用了来自gNB的MAC层遥测数据。

**📈 对比分析**

与传统的手动创建和部署xApp的方法相比，双脑架构显著减少了延迟，优化的稳态延迟为384毫秒，满足了Near-RT RIC的10毫秒推理约束。

**⚠️ 局限性**

局限性包括自动标记阈值的敏感性、意图模糊性以及在不同部署环境中LLM的延迟问题。

---

## 432. Routing Equilibrium in Mixed-Autonomy Traffic Networks with Altruistic Autonomous Agents

**arXiv ID:** 2605.23782 | [PDF](https://arxiv.org/pdf/2605.23782v1)

**作者:** Lihui Yi `[一作]` (Northwestern University), Ermin Wei `[通讯]` (Northwestern University)

**通讯引用:** 2147 | [OpenAlex ID](https://openalex.org/A5085511405)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了混合自主交通环境下的交通分配问题，建立了人类驾驶者与自主车辆之间的相互作用模型。

**💡 创新点**

通过将均衡形式化为变分不等式，证明了在一般成本函数下均衡的存在性，并在BPR旅行时间函数下证明了聚合链流和社会成本的唯一性。

**🔧 技术方法**

使用变分不等式（VI）框架来分析交通分配问题，并提供了在不同网络和成本结构下的充分条件。

**📊 数据集**

未具体提及使用的数据集，但通过数值实验展示了自主车辆对社会成本的影响。

**📈 对比分析**

与传统的交通分配模型进行比较，结果表明在某些条件下引入自主车辆可以改善或恶化社会成本，且在凸成本函数下，去中心化和中心化的均衡结果相同。

**⚠️ 局限性**

研究中未提及具体的局限性，但可以推测在不同网络结构和成本函数下的适用性可能存在限制。

---

## 433. Swarical: An Integrated Hierarchical Approach to Localizing Flying Light Specks

**arXiv ID:** 2605.23774 | [PDF](https://arxiv.org/pdf/2605.23774v1)

**作者:** Hamed Alimohammadzadeh `[一作]` (University of Southern California), Shahram Ghandeharizadeh `[通讯]` (University of Southern California)

**通讯引用:** 3930 | [OpenAlex ID](https://openalex.org/A5015710502)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

对小型光点无人机编队进行层级化定位与照明的完整框架设计与实现。

**💡 创新点**

创新点在于将硬件传感器特性、软件算法与3D形状数据结合，离线规划组队并构造线视距约束的多级树结构，从而实现高精度、低能耗的连续定位。

**🔧 技术方法**

采用Raspberry Pi摄像头与ArUco标记进行视觉定位，离线规划使用Poisson‑disk采样与k‑means聚类，在线定位实现三种并发策略（HC、ISR、RSF）。

**📊 数据集**

使用公开的3D网格模型（如滑板、棋盘、棋子等）作为实验数据集，并生成对应的点云。

**📈 对比分析**

通过Hausdorff和Chamfer距离与SwarMer进行对比，ISR在精度上与SwarMer持平但速度提升约2倍，整体定位误差保持在毫米级。

**⚠️ 局限性**

局限性包括对摄像头测距误差敏感，依赖离线规划可能不适用于实时动态场景，以及对传感器硬件的精度和安装姿态要求较高。

---

## 434. HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs

**arXiv ID:** 2605.23764 | [PDF](https://arxiv.org/pdf/2605.23764v1)

**作者:** Zewen Jin `[一作]` (University Of Science And Technology Of China), Cheng Li `[通讯]` (University Of Science And Technology Of China)

**通讯引用:** 47494 | [OpenAlex ID](https://openalex.org/A5100354240)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种用于Ascend NPU上MoE训练的编译和调度框架，旨在提高训练效率。

**💡 创新点**

通过将MoE执行从操作级别转变为静态调度的块级异构任务流，显著提高了硬件利用率和训练效率。

**🔧 技术方法**

使用了静态调度框架和事件驱动的同步机制，支持在同一内核启动中并行执行AIC和AIV任务。

**📊 数据集**

在Ascend A3集群上使用DeepSeek风格的MoE模型进行评估。

**📈 对比分析**

与传统的内核逐个执行方法相比，减少了Dispatch到Combine的MoE-FFN延迟，速度提升达到1.49×至1.58×，并在端到端训练中实现了1.08×至1.09×的加速。

**⚠️ 局限性**

该方法的局限性在于仍需进行全面的模型评估和更广泛的操作符覆盖，未来工作将集中在这些方面。

---

## 435. LLM-driven design of physics-constrained constitutive models: two agents are better than one

**arXiv ID:** 2605.23754 | [PDF](https://arxiv.org/pdf/2605.23754v1)

**作者:** Marius Tacke `[一作]` (Helmholtz-Zentrum Hereon), Christian Cyron `[通讯]` (Helmholtz-Zentrum Hereon)

**通讯引用:** 4081 | [OpenAlex ID](https://openalex.org/A5041511949)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多代理的LLM驱动的本构模型生成方法，其中创建者代理提出模型，检查者代理对模型进行审查，确保其符合物理约束。

**💡 创新点**

创新点在于引入了检查者代理，系统地验证生成的模型是否满足九个物理约束，从而提高了模型的可靠性和准确性。

**🔧 技术方法**

使用了大型语言模型（LLMs）作为创建者和检查者代理，具体实现中使用了Claude Opus 4.7和Kimi K2.5作为基础模型。

**📊 数据集**

使用了三个数据集进行基准测试：脑组织数据集、实验橡胶数据集和合成橡胶数据集。

**📈 对比分析**

与单一创建者代理的方法相比，添加检查者后，符合所有物理约束的模型比例从91%提高到100%（Claude Opus 4.7），从37%提高到56%（Kimi K2.5），同时保持了接近基线的准确性和对未见加载路径的良好泛化能力。

**⚠️ 局限性**

限制在于当前方法仍依赖于LLM的能力，且在某些情况下，检查者可能会对物理约束的判断产生不确定性。

---

## 436. The Communication Complexity of Instant-Runoff Voting

**arXiv ID:** 2605.23743 | [PDF](https://arxiv.org/pdf/2605.23743v1)

**作者:** Élie de Panafieu `[一作]` (Nokia Bell Labs), Jérôme Lang `[通讯]` (CNRS)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了即时淘汰投票（IRV）的通信复杂性，填补了其下界与上界之间的空白，证明了IRV的通信复杂性为Θ(n (log m)^2)。

**💡 创新点**

通过使用欺骗集技术，将IRV的下界提高到Ω(n (log m)^2)，并展示了在单峰性限制下，IRV的复杂性降至Θ(n log m)。

**🔧 技术方法**

使用了欺骗集技术和渐近分析方法来证明通信复杂性。

**📊 数据集**

没有具体提到使用的数据集，但讨论了IRV在多个国家的实际应用和相关的投票规则。

**📈 对比分析**

与Conitzer和Sandholm的结果相比，填补了IRV的通信复杂性下界与上界之间的空白，确认了IRV的通信复杂性为Θ(n (log m)^2)，并指出在单峰性情况下复杂性降低至Θ(n log m)。

**⚠️ 局限性**

在大规模政治选举中，假设选民需要保持在线或在提示时重新连接，这在实际操作中可能不切实际，尤其是在大规模选举中。

---

## 437. MemAudit: Post-hoc Auditing of Poisoned Agent Memory via Causal Attribution and Structural Anomaly Detection

**arXiv ID:** 2605.23723 | [PDF](https://arxiv.org/pdf/2605.23723v1)

**作者:** Zhewen Tan `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Lin Sun `[通讯]` (Qiyuan Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MemAudit 框架，用于在 LLM 代理出现有害行为后，通过审计与移除可疑记忆来恢复安全。

**💡 创新点**

创新点在于将反事实因果影响评分（CMIS）与全局结构异常检测（MCG）相结合，形成无毒素标签的事后内存毒化检测与修复方法。

**🔧 技术方法**

采用反事实重放因果归因、基于 DeBERTa‑v3 的语义一致性图构建、结构异常得分计算，以及融合去毒得分的组合算法。

**📊 数据集**

使用 MINJA 框架下的 QA 与 RAP 两个任务，分别在 GPT‑4o、GPT‑4o‑mini 与 DeepSeek 后端上进行实验。

**📈 对比分析**

与随机删除、检索频率删除、最近邻矛盾过滤三种基线相比，MemAudit 在 QA 任务中将攻击成功率 (ASR) 从 70% 降至 0%，在 RAP 任务中也实现 0% 的 ASR；单组件（CMIS 或 MCG）表现相对弱，证明双信号融合是必要的。

**⚠️ 局限性**

局限性：仅适用于已观察到的有害行为；依赖可观测的失败信号与重放；在毒化密度过高时效果下降；实验范围局限于 MINJA 现有攻击场景，未覆盖更广泛的记忆攻击模式。

---

## 438. Vision-Based Agile Landing on Turbulent Waters

**arXiv ID:** 2605.23717 | [PDF](https://arxiv.org/pdf/2605.23717v1)

**作者:** Dimosthenis Angelis `[一作]` (Technical University of Denmark), Evangelos Boukas `[通讯]` (Technical University of Denmark)

**通讯引用:** 688 | [OpenAlex ID](https://openalex.org/A5072394579)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于强化学习的无人机在海上多自由度平台上实现自主降落的方法

**💡 创新点**

创新点在于使用稀疏局部视觉特征与多旋翼状态直接预测姿态和推力指令，完全不依赖显式平台状态估计，实现了零射击部署并能兼容多种特征提取器

**🔧 技术方法**

核心技术包括PPO强化学习、SRPose稀疏特征编码、低级姿态/推力控制、仿真训练与真实部署

**📊 数据集**

数据集：仿真中使用随机生成的关键点/描述子与海面波动模型；真实实验使用A‑KAZE或SURF提取的随机视觉图案与6‑DoF斯图尔特平台

**📈 对比分析**

对比方法：与MPC‑NE基线比较；仿真中成功率从46.69%提升至72.14%，平均机动时间从86 s降至2.3 s；真实实验成功率约58%，无需额外部署或平台状态信息

**⚠️ 局限性**

局限性：对极端、非正弦波浪形态的泛化能力有限，依赖稳定的低级控制器和相机标定，且在姿态/速度失配时仍可能出现失误

---

## 439. Instrumentation for Imitation Learning: Enhancing Training Datasets for Clothes Hanger Insertion

**arXiv ID:** 2605.23847 | [PDF](https://arxiv.org/pdf/2605.23847v1)

**作者:** Remko Proesmans `[一作]` (Ghent University---imec), Francis wyffels `[通讯]` (Ghent University---imec)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在衣架插入任务中通过在衣架上集成传感器进行仪器化演示学习的效果。

**💡 创新点**

证明了仪器化（传感器提供的特权状态信息）能显著提升模仿学习策略，并且黑盒策略能自发识别并优先利用仪器化信号；同时使用仪器化策略回放提升无传感器学生策略。

**🔧 技术方法**

使用Diffusion Policy架构、ResNet18视觉编码器、TCRT5000 IR传感器集成、BLE数据通信以及贝叶斯成功率分析等技术。

**📊 数据集**

使用了180条人类远程操作演示（按不同子集划分）以及来自仪器化策略的回放，构成原始和增强的训练集。

**📈 对比分析**

对比有/无传感器的同类策略，采用终端成功率和贝叶斯分布评估，结果显示有传感器策略比无传感器提升14–25个百分点；增强的无传感器策略也比基础无传感器提升12个百分点。

**⚠️ 局限性**

局限性包括需要在每个物体中集成传感器，实施成本与可扩展性问题；仪器化带来的性能提升在本案例中不足以抵消硬件实现的投入；对数据规模的依赖尚未明确。

---

## 440. DORA: Dataflow-Instruction Orchestration Architecture for DNN Acceleration

**arXiv ID:** 2605.23833 | [PDF](https://arxiv.org/pdf/2605.23833v1)

**作者:** Xingzhen Chen `[一作]` (Brown University), Peipei Zhou `[通讯]` (Brown University)

**通讯引用:** 2080 | [OpenAlex ID](https://openalex.org/A5063866156)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DORA架构与编译框架，利用指令化 overlay 在 AMD Versal 上实现对 DNN 工作负载的细粒度数据流、计算与同步控制；

**💡 创新点**

创新点在于引入可编程 ISA 与动态内存/并行管理机制，结合两阶段 DSE（MILP 与遗传算法）实现跨模型的高效调度与可配置硬件生成；

**🔧 技术方法**

采用 VLIW 向量处理器、AI Engine、可配置 LMU/MMU/SFU、全连网络、同步单元、MILP 与 GA 调度、模板化架构生成等技术；

**📊 数据集**

使用多种 DNN 模型（MLP、DeiT、BERT、PointNet、NCF）在 FP32 数据类型下进行性能评测；

**📈 对比分析**

通过与 CHARM 2.0 与 RSN 等基线比较，DORA 在单向量处理器上保持 <5% 效率波动，整体吞吐提升至 5 倍，GA 调度在实际时间约束内达 90% 最优；

**⚠️ 局限性**

局限性包括仅在 AMD Versal/PL+AIE 平台实现、对极大模型仍可能资源匹配不足、需手动配置模板、缺乏完整的开放评测工具，并未覆盖低功耗场景。

---

## 441. SFG-ROS: A Resource-Aware Framework for Dense Multi-Agent Perception

**arXiv ID:** 2605.23832 | [PDF](https://arxiv.org/pdf/2605.23832v1)

**作者:** Constantin Blessing `[一作]` (Esslingen University of Applied Sciences), Markus Enzweiler `[通讯]` (Esslingen University of Applied Sciences)

**通讯引用:** 15611 | [OpenAlex ID](https://openalex.org/A5040116973)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了SFG-ROS框架，用于在异构多智能体队列中实现高效协同感知，解决了网络拥塞、冗余解码、跨平台部署等问题。

**💡 创新点**

创新点包括：基于全局压缩/局部原始的网络架构；通过程序化FQN schema驱动Fast DDS路由实现流量与命名空间隔离；集中式动态解码代理降低CPU/网络开销；硬件无关的多层Docker容器化部署；以及针对Vicon、轨迹规划等的专用可视化工具。

**🔧 技术方法**

使用了ROS 2、Fast DDS及DDS Router、FFmpeg H.264与RVL无损解码、Composable Nodes、Docker多阶段构建、systemd自启动、VS Code dev‑container、以及Rerun、Vicon Bridge等工具。

**📊 数据集**

实验数据来自RealSense D435相机、Stereo深度摄像头、Unitree Go2、Raspberry Pi 5、Steam Deck OLED、Vicon Vero 2.2；未使用公开数据集。

**📈 对比分析**

与标准ROS 2做对比：网络流量降至O(1)；每多一订阅器CPU利用率提升仅1.8%（相比6.5%降低72.3%）；发现时间从高达40 s降至0.8–2.9 s；网络发现流量下降1.8个数量级；玻璃到玻璃延迟仅增加2.6 ms；订阅首帧延迟从570 ms升至1095 ms。

**⚠️ 局限性**

局限包括：DDS协议在高密度无线网络下仍有发现与传输开销；容器化部署依赖挂载工作空间，缺乏源码版本化与OTA更新机制；未来需探索非DDS中间件与CI/CD自动化部署。

---

## 442. Debiased Negative Mining Improves Out-of-distribution Detection with Pre-trained Vision-Language Models

**arXiv ID:** 2605.23797 | [PDF](https://arxiv.org/pdf/2605.23797v1)

**作者:** Bo Peng `[一作]` (University of Technology Sydney), Zhen Fang `[通讯]` (University of Technology Sydney)

**通讯引用:** 13794 | [OpenAlex ID](https://openalex.org/A5057183219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文旨在通过后处理的方式，利用预训练的视觉-语言模型（VLM）进行异常输入的检测，特别是针对未知类别的输入进行分布外（OOD）检测。

**💡 创新点**

创新点在于提出了一种理论框架，通过间接近似负标签的分布来纠正负标签的采样偏差，从而提高OOD检测的准确性，解决了现有方法中普遍存在的假阴性问题。

**🔧 技术方法**

使用了基于蒙特卡洛采样的负标签去偏方法，结合了正负样本的无监督学习技术，利用未标记的野生语料库数据和已知的正样本进行负样本的挖掘。

**📊 数据集**

主要使用了ImageNet-1K作为ID数据集，同时在多个OOD数据集（如iNaturalist、SUN、Places365和Textures）上进行了评估。

**📈 对比分析**

与现有的多种先进方法（如MSP、ODIN、NegLabel等）进行了比较，实验结果表明，所提方法在多个OOD检测基准上达到了新的最先进性能，尤其在零-shot设置下表现优异。

**⚠️ 局限性**

限制在于该方法依赖于未标记的野生数据，可能会受到数据质量和多样性的影响，此外，超参数的选择也可能影响模型的性能。

---

## 443. Engagement-Optimized Care: When LLMs become Mental Health Infrastructure

**arXiv ID:** 2605.23787 | [PDF](https://arxiv.org/pdf/2605.23787v1)

**作者:** Briana Vecchione `[一作]` (Data & Society Research Institute), Ranjit Singh `[通讯]` (Data & Society Research Institute)

**通讯引用:** 4912 | [OpenAlex ID](https://openalex.org/A5027162427)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了美国18名使用者在长期使用通用大型语言模型（LLM）获得情感支持的行为轨迹，结合访谈、日记、焦点小组等质性方法；

**💡 创新点**

创新点在于把分析焦点从单回合交互转向长期使用轨迹，揭示情感依赖、验证偏差和隐私风险等治理难点，并提出针对设计与激励层面的责任转移建议；

**🔧 技术方法**

未使用具体算法或技术，仅采用质性研究方法；

**📊 数据集**

数据集为18名美国LLM使用者的访谈记录、四周日记条目、焦点小组讨论与退出访谈；

**📈 对比分析**

无性能比较指标，论文未进行技术评估或量化结果；

**⚠️ 局限性**

局限性包括样本规模有限、仅为美国自选参与者、LLM模型在研究期间可能已更新导致行为不稳定、研究主要基于自述数据，缺乏客观行为测量。

---

## 444. SeedER: Seed-and-Expand Retrieval from Knowledge Graphs

**arXiv ID:** 2605.23753 | [PDF](https://arxiv.org/pdf/2605.23753v1)

**作者:** Hamed Shirzad `[一作]` (Valence Labs), Emmanuel Noutahi `[通讯]` (Valence Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于“Seed-and-Expand”策略的知识图谱检索框架，先用密集检索获得核心节点，再通过学习的图形化策略在局部子图中迭代选择扩展节点，生成高覆盖率、低噪声的候选集；

**💡 创新点**

核心创新在于把多跳检索拆解为局部决策过程，利用强化学习在受限的局部子图内学习可扩展的图形化策略，解决了单一全局向量无法捕捉多跳关系的瓶颈，并实现了对复杂组合查询的通用化；

**🔧 技术方法**

使用了密集向量检索做种子；图神经网络（改进的GNN）做查询条件化节点嵌入；强化学习（group‑centered REINFORCE）训练扩展策略；BPR 损失做最终排序；并在训练中采用子图裁剪、k‑hop-with‑filter 预生成查询特定子图；

**📊 数据集**

在 STARK 公开基准上评估：STARK‑PRIME（精准医学），STARK‑MAG（学术论文），STARK‑AMAZON（商品搜索）三大数据集；

**📈 对比分析**

与稠密检索、K‑hop‑with‑filter、图检索（G‑Retriever、SubgraphRAG）、Beam Search、A*、PPR、GraphFlow、LLM‑agent 等方法对比；实验显示在 Hit@1、Hit@5、MRR、Recall@20 等指标上，Seed‑and‑Expand 在所有三大基准均超越密集检索和启发式多跳扩展，尤其在 STARK‑PRIME 上提升了约 95% 的 Recall@20，并且参数量仅 1.1M、推理速度快于 LLM‑agent；

**⚠️ 局限性**

局限性包括：1）需要预先生成查询特定子图，仍然受图结构与查询相似度的限制；2）强化学习奖励稀疏，训练稳定性依赖多轨迹采样；3）在极大规模图上仍需进一步压缩子图或改进近似检索；4）对非常长链路或极稀疏关系的检索效果尚未充分验证。

---

## 445. Approaching I/O-optimality for Approximate Attention

**arXiv ID:** 2605.23751 | [PDF](https://arxiv.org/pdf/2605.23751v1)

**作者:** Pál András Papp `[一作]` (Huawei Technologies), Anastasios Zouzias `[通讯]` (Huawei Technologies)

**通讯引用:** 775 | [OpenAlex ID](https://openalex.org/A5019949317)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文重新研究了大型语言模型中注意力机制的 I/O 复杂度，提出了一种基于 Alman‑Song 近似注意力算法的 I/O 高效实现，并给出了在不同参数取值下的上界与下界。

**💡 创新点**

创新点在于：① 将近似注意力与 I/O 复杂度分析相结合，首次在多种缓存规模和多项式度数下给出近乎线性的 I/O 上界；② 提供对应的下界证明，证明算法在大多数参数域内已接近最优；③ 通过引入特定的分块/tiling 方案以及红蓝斑点游戏与 S‑partition 技术，完成了复杂的 I/O 分析。

**🔧 技术方法**

主要技术包括：多项式近似指数函数（Alman‑Song 方案）；矩阵乘法的 I/O 友好分块策略；红蓝斑点游戏模型；S‑partition 的下界证明；对二项式系数的近似与取整分析。

**📊 数据集**

本文为理论分析工作，未使用实际数据集；所有结果均基于理论模型与参数假设。

**📈 对比分析**

方法上与 FlashAttention 等传统 I/O 最优算法进行比较。理论上，近似注意力在 M ≪ n·d 的 regime 下可将 I/O 复杂度从 O(n²d²/M) 降到 O(n·d/M)（或更低），实现近乎线性降低；实验性评估未给出，主要基于理论上界/下界。

**⚠️ 局限性**

局限性包括：① 实际中 r（多项式展开后的维度）可能极大，导致实现成本高；② 依赖 g 为常数或小于 log M 的假设；③ 主要针对单级缓存模型，未考虑多级缓存、GPU 等实际硬件细节；④ 未给出实验验证，仅提供理论证明。

---

## 446. Revitalizing Dense Material Segmentation: Stabilized Vision Transformers and the Generalization Paradox

**arXiv ID:** 2605.23747 | [PDF](https://arxiv.org/pdf/2605.23747v1)

**作者:** Allan Kazakov `[一作]` (Bahcesehir University), Yavuz İrfanoğlu `[通讯]` (Poder Bilişim Teknolojileri Sanayi ve Ticaret)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现Apple-DMS benchmark，提出Vision Transformer（SegFormer-B5）在原始split上的稳定化训练，并通过重新划分数据集揭示“Generalization Paradox”。

**💡 创新点**

创新点包括高保真Logit投影与查询熵正则化的训练稳定化方法、纹理优先的域特定增强管线，以及发现数据量增大导致的泛化悖论。

**🔧 技术方法**

采用SegFormer-B5与Mask2Former两种Transformer架构，结合高保真Logit投影、查询熵正则化、差异学习率、余弦退火与纹理优先增强等技术。

**📊 数据集**

使用Apple Dense Material Segmentation (DMS) 数据集（恢复后约41,396张图像），并在原始54/23/23 split与自定义80/10/10 split上进行实验。

**📈 对比分析**

通过与原ResNet-50基线对比，SegFormer-B5在原始split取得0.4572 mIoU，提升约8.6%；在自定义split提升至0.5276 mIoU，但在专家评估的外部样本中表现下降；Mask2Former表现略逊。

**⚠️ 局限性**

限制包括约7%原始图像缺失、外部泛化评估仅为专家定性、缺乏真实稠密标注的“in-the-wild”基准，以及模型泛化受数据分割方式影响。

---

## 447. MuellerPT: Decomposition Driven Pretraining for Dense Learning in Mueller Polarimetry

**arXiv ID:** 2605.23840 | [PDF](https://arxiv.org/pdf/2605.23840v1)

**作者:** Adam Tlemsani `[一作]` (Imperial College London), Daniel S. Elson `[通讯]` (Imperial College London)

**通讯引用:** 6391 | [OpenAlex ID](https://openalex.org/A5086866155)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了MuellerPT框架，利用自监督的Lu‑Chipman分解参数预测作为预训练任务，改进Mueller矩阵图像在分割和分类中的性能。

**💡 创新点**

创新点在于将Lu‑Chipman参数预测作为物理驱动的预训练目标，结合Mueller元素Dropout和双流编码器实现高效、鲁棒的特征学习，并公开了新的MAP‑Org多光谱动物组织Mueller矩阵数据集。

**🔧 技术方法**

采用自监督学习、Mueller矩阵归一化与物理可实现性校正、Mueller元素Dropout、双流HRNet编码器、Mueller旋转数据增强以及少样本学习评估等技术。

**📊 数据集**

训练使用新收集的MAP‑Org数据集；下游任务在PoLambRimetry（羊脑灰白分割）和ColoPola（结肠癌筛查）数据集上评估；亦在ex vivo人类食管样本上验证跨域迁移。

**📈 对比分析**

与从随机初始化训练的HRNet‑Scratch做同等实验比较；在少样本设置下，MuellerPT在5%训练数据时灰白分割DICE提升约20%，癌症分类整体准确率提升约8%；在完整数据时两者性能相近；在域迁移实验中预训练模型能够重建Lu‑Chipman参数，表现优于未预训练。

**⚠️ 局限性**

限制包括对噪声敏感的Lu‑Chipman分解、需要更大、多样化的数据集验证泛化性、对硬件配置的适应性尚有限、未评估实时部署的计算开销。

---

## 448. Decomposing Queries into Tool Calls for Long-Video Keyframe Retrieval

**arXiv ID:** 2605.23826 | [PDF](https://arxiv.org/pdf/2605.23826v1)

**作者:** Michal Shlapentokh-Rothman `[一作]` (University of Illinois Urbana Champaign), Derek Hoiem `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于工具分解与合并的关键帧检索方法ToolMerge，用LLM规划器将长视频问答查询拆分为多工具调用，并通过布尔运算合并排名；同时构建了Molmo‑2 Moments（M2M）数据集，将每道题目固定在特定时间片段，以便单独评估检索性能。

**💡 创新点**

创新点在于：①利用LLM规划器实现查询的自适应工具调用与布尔合并，突破传统固定模式的局限；②引入OCR与区域匹配工具，提升对细粒度与文字信息的检索能力；③设计M2M数据集以分离检索与推理，便于针对性优化检索算法。

**🔧 技术方法**

技术主要包括：LLM（Qwen3‑VL‑8B或GPT‑5.4‑Pro）规划器、SigLIP‑2（图像‑文本匹配）、T‑REN（区域‑文本匹配）、EasyOCR+GPT‑4o‑Mini文本判别、贪婪NMS合并框架，以及GRPO强化学习对规划器进行无监督调优。

**📊 数据集**

使用的数据集包括Molmo‑2 Captioning（生成M2M）、Molmo‑2 Moments（检索/问答评测）、LongVideoBench、Video‑MME和基准中的Caption Retrieval。

**📈 对比分析**

与多种基线（无帧、均匀采样、SigLIP‑Q、BOLT、WFS、AKS、LIF等）对比，ToolMerge在M2M问答与检索任务中表现最优，Caption Retrieval上提升约5%；在LongVideoBench与Video‑MME的长视频问答中也取得领先或相近成绩；基线SigLIP‑Q也表现出较强的竞争力。

**⚠️ 局限性**

局限性包括：①规划器依赖LLM生成工具调用，若规划错误导致检索失败；②对计算资源仍有一定需求，尤其在OCR与多工具同步时；③对非常长视频或高帧率场景的效率尚未充分验证；④方法主要针对视觉与文本匹配，对音频或更复杂的多模态推理支持有限。

---

## 449. Inferential Privacy Leakage in Anonymized Conversational AI Logs

**arXiv ID:** 2605.23820 | [PDF](https://arxiv.org/pdf/2605.23820v1)

**作者:** S M Mehedi Zaman `[一作]` (Rutgers University), Kiran Garimella `[通讯]` (Rutgers University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文研究了用户与ChatGPT等大型语言模型（LLM）助手之间的对话历史中，用户身份信息的隐私相关特征。通过分析来自巴西、印度、尼日利亚和巴基斯坦的1000多名用户的完整ChatGPT对话历史，发现34.5%的用户消息包含个人信息，并且在对话的前14%中，用户通常会首次透露身份信息。

**💡 创新点**

创新点在于通过严格的用户筛选标准，分析在没有显式身份信息的情况下，用户的年龄、性别和国家等人口统计信息仍然可以被推断出来。研究还揭示了推断过程中存在的四种刻板印象模式，这些模式导致了对某些群体（如技术领域的女性和年长用户）的错误分类。

**🔧 技术方法**

使用了Llama-3.3-70B-Instruct模型进行用户身份推断，并通过自然语言处理技术分析用户的对话历史。

**📊 数据集**

数据集包括来自四个全球南方国家（巴西、印度、尼日利亚、巴基斯坦）的1057个ChatGPT对话历史，以及212名印度用户的Google搜索、YouTube搜索和观看历史。

**📈 对比分析**

与Google搜索和YouTube历史进行比较，发现ChatGPT在年龄、教育和投票偏好等属性上表现优越，而在性别、宗教和收入等属性上则不如Google搜索和YouTube搜索。整体上，ChatGPT的推断能力与这些传统平台相当，但在某些属性上具有不同的内容侧重点。

**⚠️ 局限性**

限制在于只使用了开放权重的Llama-3.3-70B模型，未测试其他可能推断能力更强的专有模型。此外，样本偏向于年轻、英语流利和技术参与者，可能影响推断的普遍性。

---

## 450. Dynamic Query Modification for Binary Locality Sensitive Hashing

**arXiv ID:** 2605.23807 | [PDF](https://arxiv.org/pdf/2605.23807v1)

**作者:** Ben Claydon `[一作]` (University of St Andrews), Alan Dearle `[通讯]` (University of St Andrews)

**通讯引用:** 1741 | [OpenAlex ID](https://openalex.org/A5019956533)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种在查询时动态修改查询向量的技术，以提升二进制LSH在近邻检索中的召回率和速度。

**💡 创新点**

创新点在于构造集合 Φ 并在查询过程中估计最佳查询向量 c，从而显著提高哈希碰撞概率并消除 hash 失败。

**🔧 技术方法**

采用二进制 LSH（Charikar、RP-Tree/Forest）、投影树、候选集维护与快速质心计算等技术实现动态查询修改。

**📊 数据集**

实验使用四个大规模高维数据集：GloVe、MirFlickr‑Dino2、GooAQ 以及 200 维均匀分布数据。

**📈 对比分析**

通过与传统 RP‑Forest 进行比较，MQ‑Forest 在保持相同召回率时树数减少 20–30%，距离计算量下降 20–30%，整体构建和查询速度提升 30–40%。

**⚠️ 局限性**

局限性包括对均匀分布数据集效果不佳、需要对近邻集合有足够质量的估计，以及在极低维或离散数据上表现不如预期。

---

## 451. Recursion and proof theoretical characterizations of small circuit classes with modulo counting via discrete differential equations (long version)

**arXiv ID:** 2605.23805 | [PDF](https://arxiv.org/pdf/2605.23805v1)

**作者:** Melissa Antonelli `[一作]`, Rui Li `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出了一种隐式复杂性方法，通过离散常微分方程（ODE）的视角研究多项式大小、常数深度电路的计算，特别是包含模n门的电路。

**💡 创新点**

创新点在于通过ODE模式而非有界递归进行更细致的分析，从而为所有类n（n∈ℕ）提供了统一的特征化，这在之前的研究中未曾实现。

**🔧 技术方法**

使用了离散常微分方程（ODE）作为主要技术工具，提出了基于ODE的函数代数来捕捉电路类的计算能力。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了电路类的特征化，涉及模n门的电路计算。

**📈 对比分析**

与现有方法的比较表明，本文的方法在特征化模n门的电路类方面具有显著的进展，尤其是对于n=2和n=6的情况，提供了更为统一和简化的证明。

**⚠️ 局限性**

限制在于尽管本文扩展了对模n门的研究，但对于某些特定的电路类，仍然存在未解决的复杂性问题，特别是在更高阶的模类中。

---

## 452. Perceptually Lossless Tactile Texture Synthesis with Compact Spectral Envelope Models

**arXiv ID:** 2605.23804 | [PDF](https://arxiv.org/pdf/2605.23804v1)

**作者:** Jagan K. Balasubramanian `[一作]`, Yasemin Vardar `[通讯]` (Delft University of Technology)

**通讯引用:** 499 | [OpenAlex ID](https://openalex.org/A5044880947)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了两种紧凑的触觉信号表示方法：谱贝塔（spectral beta）和谱斜率（spectral slope），用于捕捉手指与表面摩擦信号的时间谱结构，同时保留感知相关信息。

**💡 创新点**

创新点在于这两种表示方法能够在保持感知相似性的同时，使用更少的参数进行触觉信号的合成和压缩，提供了一种高效的触觉压缩和合成框架。

**🔧 技术方法**

使用了谱贝塔模型和谱斜率模型，这两种模型分别通过贝塔分布和不对称带通滤波器来近似触觉信号的频谱。

**📊 数据集**

使用了五种不同的自然纹理数据集，包括细织物、粗织物、波纹纸、砂纸和乙烯基，这些数据集通过电动振动显示器进行渲染和评估。

**📈 对比分析**

与现有方法（自回归、梅尔频率倒谱系数和谱峰）进行比较，谱贝塔在感知相似性评分上表现出色，尤其在细纹理和粗纹理上均表现良好，且与高保真重现的相似性相当。

**⚠️ 局限性**

局限性在于实验中仅使用了14名参与者，且触觉信号的记录是由第一作者完成的，可能引入个体差异，此外，皮肤水分波动和手指与电动振动的交互也可能影响结果的可靠性。

---

## 453. A Novel Approach for the Counting of Wood Logs Using cGANs and Image Processing Techniques

**arXiv ID:** 2605.23775 | [PDF](https://arxiv.org/pdf/2605.23775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 454. Exploring deep learning for Event-Based Saliency Prediction with a Transformer-based model

**arXiv ID:** 2605.23790 | [PDF](https://arxiv.org/pdf/2605.23790v1)

**作者:** Romaric Mazna `[一作]` (Université Côte d'Azur), Sai Deepesh Pokala `[通讯]` (Université Côte d'Azur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SEST（Swin Event-based Saliency Transformer）模型，用于从事件摄像头数据预测视觉显著性，并创建了两套新的事件显著性数据集；

**💡 创新点**

首次将自监督预训练的Swin Transformer与CNN解码器结合用于事件显著性预测，利用合成事件数据生成大规模数据集，实现零样本迁移到真实事件摄像头；

**🔧 技术方法**

使用Swin Transformer编码器、3D卷积特征融合、CNN解码器、可学习中心偏置、事件自监督预训练、ESIM事件模拟、KL+CC+BCE多指标损失以及多种评估指标；

**📊 数据集**

使用从DHF1K和UCF Sports转换而来的合成事件数据集N-DHF1K和N-UCF Sports，以及原始RGB显著性基准和一份真实事件摄像头数据集；

**📈 对比分析**

与现有事件模型（SNNevProto、evST）和多种RGB模型在AUC‑J、CC、SIM、NSS等指标上对比，SEST在事件数据集上取得最高AUC‑J 0.9197、CC 0.4907、SIM 0.3969、NSS 2.3956，显著优于前辈模型；零样本评估中在CC、SIM上优于evST；

**⚠️ 局限性**

仅在解码后利用时序信息，模型体量较大不适合边缘部署；对低运动或特定领域的泛化仍有限，需要进一步的时序注意力机制和模型压缩。

---

## 455. Benchmarking LLMs for Community Governance Simulation with Life-history Narratives

**arXiv ID:** 2605.23783 | [PDF](https://arxiv.org/pdf/2605.23783v1)

**作者:** Xu Chen `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 26433 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了基于居民生命周期叙事的社区治理模拟框架，包括数据集、基准、算法和闭环系统；

**💡 创新点**

创新点在于：①大规模(92户、约120万字符)细粒度生命周期叙事与治理态度并行的数据；②对18款主流LLM进行系统性准确率–成本基准，揭示纯提示策略的高成本低收益；③提出参数高效的 curriculum‑LoRA 微调方法，显著提升准确率同时成本降十倍以上；

**🔧 技术方法**

使用技术包括：生命周期提示、少量示例提示、LoRA 适配器、在线随机抽样与逐步递增参考问题的课程调度；

**📊 数据集**

数据集来源于92名长期居民的两小时半结构化访谈，生成约120万字符的一人称叙事和50问治理态度问答；

**📈 对比分析**

与18款LLM在72种提示组合进行比较，发现最佳提示准确率≈50%，成本高昂；而 curriculum‑LoRA 在保持51.6%准确率的同时，每次调用成本降低约16‑93倍，性能在准确率–成本平面上占优；

**⚠️ 局限性**

局限性包括：1）仅单一城市社区实验，跨区域推广需验证；2）数据为静态快照，未覆盖态度随时间变化；3）目前仍只能达到约50%准确率，反映LLM在推断主观偏好上的根本瓶颈。

---

## 456. Beyond Binary Edits Robust Multimodal Knowledge Editing with Adversarial Subspace Alignment

**arXiv ID:** 2605.23780 | [PDF](https://arxiv.org/pdf/2605.23780v1)

**作者:** Haoyuan Wang `[一作]` (Zhejiang University), Chaochao Chen `[通讯]` (Zhejiang University)

**通讯引用:** 6704 | [OpenAlex ID](https://openalex.org/A5028791879)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多模态大型语言模型（MLLM）上提出了一种鲁棒知识编辑框架，能够在保持编辑可靠性和局部性的同时显著提升知识泛化能力。

**💡 创新点**

创新点包括：①利用隐层对抗强化（Latent Adversarial Robustification, LAR）在模型内部生成语义一致的对抗变体；②采用秩约束子空间学习（Rank‑Constrained Subspace Learning, RCSL）将编辑层输出对齐到低秩子空间，形成统一的语义概念；③引入非对称梯度流，保证对抗样本能有效指导参数更新而不破坏原有知识。

**🔧 技术方法**

核心技术：隐层对抗生成、SVD‑based秩约束对齐、温度软化对齐损失、非对称梯度流、对抗范数约束；配合现有的BLIP2、MiniGPT‑4等基础模型。

**📊 数据集**

使用 MMEdit 基准（E‑VQA 与 E‑IC 子任务），同时在 LLaVA、Qwen2VL 上进行消融验证。

**📈 对比分析**

与基线（FT、IKE、SERAC、T‑Patcher、UniKE、WISE、MEND）相比，框架在保持可靠性和局部性不变的前提下，大幅提升泛化指标，例如 BLIP2 E‑VQA 的泛化从 84.80% 提升到 93.97%，E‑IC 从 82.54% 提升到 86.22%；在多轮编辑实验中，鲁棒性更强，误差累积更小。

**⚠️ 局限性**

局限性：增益相对保守，仍受模型规模与任务差异影响；对抗样本生成与秩约束需要调参，易受梯度噪声影响；目前仅验证于静态图文任务，对视频、动作等更复杂模态的推广尚未充分探究。

---

## 457. Agentic Proving for Program Verification

**arXiv ID:** 2605.23772 | [PDF](https://arxiv.org/pdf/2605.23772v1)

**作者:** Alessandro Sosso `[一作]` (Aarhus University), Bas Spitters `[通讯]` (Aarhus University)

**通讯引用:** 998 | [OpenAlex ID](https://openalex.org/A5043123968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估Claude Code在Lean 4程序验证基准上的agentic推理性能，涵盖规范生成、实现实现与证明三阶段；

**💡 创新点**

发现现代编译器‑循环agentic系统在程序验证上已达到前所未有的成功率，并揭示了现有基准评估方法的结构性局限；

**🔧 技术方法**

利用Claude Opus 4.6与Lean‑LSP/MCP工具链结合的Agent SDK，实施多轮Prompt策略（简单、引导、多规格）来自动生成规范、实现与证明；

**📊 数据集**

使用Lean 4可验证代码生成基准（161个问题），并对基准本身进行修订以消除已知错误；

**📈 对比分析**

通过手工审查和自动等价检查，将Claude的性能与之前的state‑of‑the‑art结果对比：规范生成成功率98.8%，其中81.3%通过等价判定；实现认证率87.5%；端到端完整管道成功率98.1%；相比以往仅0.62%的整体通过率，表现大幅提升；

**⚠️ 局限性**

限制包括：评估依赖于等价判定，易受基准规范歧义与错误影响；多规格生成产生大量不必要的候选项；现有评测框架无法充分检测规范解释多义性与开放性问题；

---

## 458. PhotoFlow: Agentic 3D Virtual Photography Missions

**arXiv ID:** 2605.23771 | [PDF](https://arxiv.org/pdf/2605.23771v1)

**作者:** Jiarui Guo `[一作]` (Shanghai Jiao Tong University), Zhihang Zhong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 563 | [OpenAlex ID](https://openalex.org/A5112535563)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了面向语言条件的虚拟摄影任务和PhotoFlow闭环摄像机搜索框架

**💡 创新点**

创新点在于将软摄影蓝图、全局锚点库、区域记忆和高探索策略结合成Director‑Reviewer‑Reflector三角色循环，实现对空间约束与审美目标的统一搜索

**🔧 技术方法**

核心技术包括大语言模型（LLM）解析指令与蓝图、基于Blender的场景勘测与可视化预览、视觉语言模型评估（Composition、Technical、Aesthetic、Alignment）以及内部评分与区域记忆的反馈机制

**📊 数据集**

使用47个公开Blender场景，构成141个自然语言摄影任务（主体摆放、关系构图、氛围/风格），每个场景配有三项任务

**📈 对比分析**

与单步LLM、锚点最佳选择、随机搜索以及单链反射等基线对比，PhotoFlow在外部质量-对齐综合评分M_qs上最高（0.578，成功率62%），相较于最佳基线提升约3–5%，在不同任务类别均表现出色

**⚠️ 局限性**

局限包括对锚点库的依赖、内部Reviewer评分无法替代外部评估、未提供完整可用率统计、缺乏阈值敏感性与显著性检验，并且仅适用于静态可重现的3D场景，难以直接迁移至物理机器人或动态环境

---

## 459. Direct Dynamic Retargeting for Humanoid Imitation Learning from Videos

**arXiv ID:** 2605.23762 | [PDF](https://arxiv.org/pdf/2605.23762v1)

**作者:** Constant Roux `[一作]` (LAAS-CNRS, Université de Toulouse), Philippe Souères `[通讯]` (LAAS-CNRS, Université de Toulouse)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出Direct Dynamic Retargeting (DDR)，一种直接在任务空间上使用采样式MPC求解的动态重定向框架，可从单目视频提取的噪声SMPL关键点直接生成高保真、物理可行的人体到机器人轨迹；

**💡 创新点**

创新点在于：①去除了传统两阶段几何+动态重定向中的几何偏差；②采用Cross‑Entropy Method（CEM）在物理仿真器内直接优化动态约束；③通过Laplacian图距离度量提升轨迹形状保真度；

**🔧 技术方法**

核心技术包括：SMPL模型关键点提取、Inverse Kinematics（用于对比）、CEM采样式MPC、Pinocchio+ProxQP物理可行性检验、基于PPO的RL训练与Constraints-as-Terminations框架；

**📊 数据集**

使用了三类公开单目视频收集的五个灵活动作（Squat、Kung fu、Long one‑foot balance、Pistol Squat、Balancing Stick），每个动作从三名不同人的视频中提取SMPL轨迹；

**📈 对比分析**

在可行性、接触序列一致性、足部滑动、成功率以及关键点跟踪误差等指标上与Geometric Retargeting (GR)和Indirect Dynamic Retargeting (IDR)对比，DDR在物理可行性接近100%，接触误差最低、成功率最高、跟踪误差与GR相当但保持物理合法；在RL下，DDR训练收敛最快、最终奖励最高，表现优于GR和IDR；

**⚠️ 局限性**

局限性包括：任务空间优化可能导致收敛慢或解不稳定，需要手工调节权重和初始化；目前仅处理少量动作，尚未扩展到大规模多样化的网络视频；以及对物理仿真器差异仍存在微小可行性差异。

---

## 460. Contrast to Detect: Dynamic Graph Contrastive Regularization for Unsupervised Anomaly Detection in Multivariate Time Series

**arXiv ID:** 2605.23744 | [PDF](https://arxiv.org/pdf/2605.23744v1)

**作者:** Yunhua Pei `[一作]` (University of Bristol), John Cartlidge `[通讯]` (University of Bristol)

**通讯引用:** 754 | [OpenAlex ID](https://openalex.org/A5017729087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 ContrastAD，一种无监督多变量时序异常检测框架，利用多视角嵌入、频率感知注意力混合器和动态图对比学习来捕获非平稳系统中的时间与结构演化。

**💡 创新点**

创新点：①将结构演化本身作为学习信号而非抑制的目标；②使用基于 DTW 的功率律稀疏图快照构造动态图；③对比学习被设计为软正则化而非严格不变性约束，显著提升稀疏异常的判别能力。

**🔧 技术方法**

核心技术包括：多视角嵌入（时间、属性、结构三角度）、频率感知注意力混合器（先做频域 top‑K 过滤再注意力）、动态图对比学习（构造稀疏图、度分布 KL 对比、InfoNCE 形式正则化），以及整体的预测与重构联合损失。

**📊 数据集**

在五个公开数据集上进行实验：SWaT（工业控制）、SMD（服务器系统）、MSL（火星车遥测）、PSM（eBay 应用服务器）和 SMAP（卫星遥测）。

**📈 对比分析**

与八种最先进基线（包括 GDN、CSTGL、FuSAG、MTG、MSHTrans、DTrans、MemStr、Catch）比较，ContrastAD 在所有数据集上均实现最高平均 F1，且在 SWaT、SMD、PSM 上取得最高 AUC，显著优于对手；消融实验表明每个模块均为性能提升提供重要贡献。

**⚠️ 局限性**

局限性：①依赖 DTW 计算与功率律稀疏图的假设，可能在变量维度极大或 DTW 计算昂贵时受限；②对超参数（如对比权重 λ）的选择仍需经验调优；③仅在五个工业/遥测场景评估，尚未验证在更广泛或更复杂的动态关系场景中的鲁棒性。

---

## 461. Optimal Dimension-Free Sampling for Regularized Classification

**arXiv ID:** 2605.23726 | [PDF](https://arxiv.org/pdf/2605.23726v1)

**作者:** Meysam Alishahi `[一作]` (University of Utah), Jeff M. Phillips `[通讯]` (University of Utah)

**通讯引用:** 2916 | [OpenAlex ID](https://openalex.org/A5017619650)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

证明了在各种正则化项下，对于广泛的Lipschitz连续分类损失函数，能够实现(1±ε)的相对误差的最优采样界限。

**💡 创新点**

提出了k^2/ε^2和k/ε^2的上界和下界，改进了现有的k^3/ε^2敏感性采样界限，并通过更精细的论证避免了过度计数的问题。

**🔧 技术方法**

使用了简单的均匀或（平方）范数采样技术，并结合了高阶矩界限和经验过程分析。

**📊 数据集**

使用了多种流行的损失函数作为示例，包括逻辑损失、sigmoid损失、铰链损失和ReLU损失。

**📈 对比分析**

与现有方法相比，提出的界限在k的多项式对数因子上是紧的，且在某些情况下，采样复杂度为Ω(n log n)，表明在某些情况下无法实现无维度的界限。

**⚠️ 局限性**

当g(0)=0时，结果表明无法实现无维度的界限，甚至不亚于次线性界限。

---

## 462. Is a Document Educational or Just Wikipedia-Style? -- Pitfalls of Classifier-Based Quality Filtering

**arXiv ID:** 2605.23721 | [PDF](https://arxiv.org/pdf/2605.23721v1)

**作者:** Mateusz Klimaszewski `[一作]` (Warsaw University of Technology), Piotr Andruszkiewicz `[通讯]` (Warsaw University of Technology)

**通讯引用:** 165 | [OpenAlex ID](https://openalex.org/A5057496609)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文揭示了基于分类器的质量过滤（CQF）模型的一个关键漏洞，表明简单的维基百科风格重格式化操作可以显著改变模型的质量评估，从而使低质量内容超越过滤阈值。

**💡 创新点**

创新点在于展示了CQF模型在面对维基百科风格重述时的脆弱性，指出其可能导致低质量内容被错误地纳入预训练语料库。

**🔧 技术方法**

使用了基于BERT的模型（如FineWeb-Edu CQF模型）进行质量评分，并通过Qwen 2.5 72B Instruct模型进行维基风格重述。

**📊 数据集**

使用了FineWeb语料库中的100,000个示例进行分析，并对26个不同领域的文本进行了评分。

**📈 对比分析**

与传统的启发式过滤方法相比，CQF模型在处理维基风格重述时表现出偏见，允许超过7%的重述数据通过过滤，尽管FineWeb-Edu模型在平均表现上最强，但在实际应用中却是最不有效的选择。

**⚠️ 局限性**

限制在于分析基于自动重格式化的现有网络语料，可能导致不准确和噪声输出，尽管重述的规模和简单性应限制结果的方差，但确切的数字仍不精确，且只是对实际问题程度的近似。

---

## 463. Enhancing Energy Efficiency in Scientific Workflows through CFD based PIVAEs

**arXiv ID:** 2605.23850 | [PDF](https://arxiv.org/pdf/2605.23850v1)

**作者:** Ali Zahir `[一作]` (University of Leicester), Jeyan Thiyagalingam `[通讯]` (Science and Technology Facilities Council)

**通讯引用:** 2002 | [OpenAlex ID](https://openalex.org/A5008605990)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新颖的、可扩展的、基于人工智能的调度框架，旨在优化高性能计算（HPC）环境中的能耗，同时不影响计算性能。

**💡 创新点**

创新点在于将计算流体动力学（CFD）与物理信息变分自编码器（PIVAE）相结合，生成物理上真实的合成工作负载数据，从而弥合热力学行为与调度决策之间的差距。

**🔧 技术方法**

使用了计算流体动力学（CFD）和物理信息变分自编码器（PIVAE）技术。

**📊 数据集**

使用了来自高性能计算集群的真实工作流执行数据集，涵盖了五个代表性的科学工作流（事件重建、粒子轨迹识别、碰撞点检测、模式识别和异常检测）。

**📈 对比分析**

与传统调度方法（如FCFS、LAS、SAS、LYNX和OM-FNN）进行比较，结果显示CFD-PIVAE指导的调度在能耗上节省了最多10%，而仅增加了5-6%的周转时间，表现出良好的能效平衡。

**⚠️ 局限性**

限制在于当前实现主要针对CPU密集型科学工作流，尚未全面涵盖内存密集型或I/O密集型工作流的能耗和性能特征。

---

## 464. Learning a Particle Dynamics Model with Real-world Videos

**arXiv ID:** 2605.23845 | [PDF](https://arxiv.org/pdf/2605.23845v1)

**作者:** Chanho Kim `[一作]` (Oregon State University), Li Fuxin `[通讯]` (Oregon State University)

**通讯引用:** 8764 | [OpenAlex ID](https://openalex.org/A5065084526)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过从无标签的真实世界视频中直接学习粒子动力学模型，利用渲染监督实现多物体碰撞动态的无动作学习。

**💡 创新点**

首次在真实视频中使用差分渲染和粒子级渲染贡献来分配对象ID，并在无物理参数的情况下训练全局可微粒子动力学；同时提供约500条多视角碰撞视频数据集。

**🔧 技术方法**

Gaussian Splatting（3D Gaussians表示）、PointConv点卷积、差分渲染监督、伪标签位置回归、硬例挖掘与多视角分割等技术。

**📊 数据集**

自己构建的约500条多视角视频数据集，包含落块堆叠和桌面保龄球两种碰撞场景。

**📈 对比分析**

与公开实现的GS-Dynamics*对比，在渲染指标PSNR、SSIM、LPIPS相当；在几何指标Chamfer Distance和位置精度δ_avg上略有提升，表明模型能更准确地重现真实碰撞。

**⚠️ 局限性**

缺乏可靠的物理正确性评价指标；渲染指标对不同方法区分度低；位置监督受噪声影响；粒子级表示难以直接提取对象级物理状态，限制了评估与解释的便利性。

---

## 465. Machine learning applied to emerald gemstone grading: framework proposal and creation of a public dataset

**arXiv ID:** 2605.23777 | [PDF](https://arxiv.org/pdf/2605.23777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 466. A blueprint for constructing 3-pass AKE protocols under commitment-based models

**arXiv ID:** 2605.23843 | [PDF](https://arxiv.org/pdf/2605.23843v1)

**作者:** Rodrigo Martín Sánchez-Ledesma `[一作]` `[通讯]` (Universidad Complutense de Madrid), Rodrigo Martín Sánchez-Ledesma (Universidad Complutense de Madrid)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了在基于承诺的AKE模型下，针对KA和KEM原语可实现的三路通行安全密钥协商协议；

**💡 创新点**

创新点在于不依赖现有的编译器，而直接设计出三路协议，并证明其可模拟原始两路协议，进而在不需要长周期密钥的情境下实现安全；

**🔧 技术方法**

采用承诺方案（需满足隐藏、绑定、强绑定和碰撞抵抗）与随机预言机（Hash）进行安全分析，使用游戏序列证明SK安全；

**📊 数据集**

无数据集，纯理论研究；

**📈 对比分析**

与已有的四路编译器产生的协议相比，安全性上限保持不变（即同形式的上界），但通信轮数从四轮降至三轮，性能在通信复杂度上有所提升；

**⚠️ 局限性**

局限在于只能实现单向认证（仅响应方验证发起方身份），无法获得双向认证，且目前未探讨在三路下实现双向认证的可能性；

---

## 467. "I can't read your mind": A Study of Neurodivergent Computing Students' Experiences with Collaborative Active Learning

**arXiv ID:** 2605.23823 | [PDF](https://arxiv.org/pdf/2605.23823v1)

**作者:** Cynthia Zastudil `[一作]` (Temple University), Stephen MacNeil `[通讯]` (Temple University)

**通讯引用:** 2149 | [OpenAlex ID](https://openalex.org/A5042822346)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对计算机课程中协作主动学习（CAL）进行问卷调查与访谈，探究神经多样性（自闭症、ADHD）学生与神经典型学生在团队动态、课程设计和环境等方面的舒适度差异，并提出可行的教学改进建议。

**💡 创新点**

首次系统聚焦神经多样性学生在CAL环境中的体验与适应策略，揭示其对团队规模、角色分配、结构化作业等因素的特定偏好，并提供针对性设计原则，填补该领域研究空白。

**🔧 技术方法**

采用定量问卷（七点Likert量表）与定性半结构化访谈相结合；对访谈文本进行反思性主题分析（RTA）以挖掘深层经验；对问卷结果进行描述性统计与分组对比。

**📊 数据集**

基于来自12所北美高校的44名计算机专业学生（其中24名自认为神经多样性学生，20名神经典型学生）的问卷数据以及4名神经多样性学生的访谈文本。

**📈 对比分析**

通过对比ND与NT学生在各维度（团队组建、规模、频率、角色、作业结构、教师介入、线上线下等）的Likert分布差异进行定量对比；发现ND学生更偏好3–5人小组、明确角色、结构化作业以及教师适度介入，且对线上协作感到不适；定性结果补充了其适应策略（自我角色分配、主动披露需求）和改进建议。

**⚠️ 局限性**

样本量较小且仅涵盖自闭症与ADHD两类神经多样性学生，缺乏官方诊断，样本分布受便利抽样限制，无法推广至更广泛的神经多样性群体或其他学科背景。

---

## 468. SDNator is Not Another SDN Controller: Enabling Extensible Data-Driven Control in Cyber-Physical Systems

**arXiv ID:** 2605.23816 | [PDF](https://arxiv.org/pdf/2605.23816v1)

**作者:** Y. Lin `[一作]`, Z. Mao `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

构建并开源了SDNator，一个支持事件驱动与数据驱动的、可扩展的CPS集中式控制框架。

**💡 创新点**

创新点在于将应用视为数据生产者/消费者，通过统一的键值数据模型和数据无所不在引擎（DUE）实现域无关、插件式的控制器，并引入按需数据生成和故障恢复机制。

**🔧 技术方法**

主要技术包括Python实现的DUE库、Redis Pub/Sub作为事件后端、MongoDB做持久化后端、以及Redis的批处理、进程池等性能优化。

**📊 数据集**

通过模拟制造车间、网络流量和COVID-19 PPE生产的场景，以及Mininet网络拓扑进行实验，没有公开数据集，使用内部仿真生成的数据。

**📈 对比分析**

对比Ryu控制器和自研实现，端到端延迟与吞吐量在相同负载下相当，吞吐可达10万条/秒，延迟低于100微秒，规模可扩展至60个地理分布的应用。

**⚠️ 局限性**

局限包括对会话型通信不友好、仅适合实时/小规模数据传输、需要自行部署后端、对安全与隐私控制不足。

---

## 469. Super Condorcet Winners and Limit Coalitional Manipulability of IRV

**arXiv ID:** 2605.23742 | [PDF](https://arxiv.org/pdf/2605.23742v1)

**作者:** François Durand `[一作]` (Nokia Bell Labs), Guillem Perarnau `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 508 | [OpenAlex ID](https://openalex.org/A5086363230)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究在Impartial Culture（IC）下，单一赢家投票规则（特别是即时淘汰制IRV和投票后续淘汰制PR）的极限 coalitional manipulation（CM）率。

**💡 创新点**

创新点在于将先前仅针对三名候选人的结果推广到任意 m≥4，证明PR在任意 m≥4 的极限 CM 率为1，而 IRV 的极限 CM 率严格小于1，并通过超级Condorcet Winner 的概念给出精确的闭式概率表达式。

**🔧 技术方法**

主要技术包括概率与统计分析（中心极限定理、Chernoff 上界）、多元高斯积分、以及对超级Condorcet Winner 条件的几何与组合推导，并辅以 Monte Carlo 模拟验证。

**📊 数据集**

实验使用的是理论上的 IC 随机模型（无真实数据集），并通过自研的 Python 包 SVVAMP 进行大规模仿真。

**📈 对比分析**

通过将理论极限值与 10 万样本的 Monte Carlo 结果对比，发现两者高度吻合，验证了理论推导的正确性；PR 的极限 CM 率为 1，IRV 的极限 CM 率随候选人数递增接近 1，但始终保持小于 1。

**⚠️ 局限性**

局限性在于仅考虑 IC 模型且针对 IRV 与 PR，未覆盖其他投票规则；计算高维高斯积分在候选人数较大时仍计算量巨大；此外，实际选举中的偏好分布可能与 IC 不符，影响结论的实用性。

---

## 470. Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking

**arXiv ID:** 2605.23733 | [PDF](https://arxiv.org/pdf/2605.23733v1)

**作者:** Ming Yang `[一作]` (LimX Dynamics), Hua Chen `[通讯]` (LimX Dynamics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出Any2Any框架，实现预训练的全身跟踪（WBT）策略跨不同类人机器人平台的高效迁移；

**💡 创新点**

核心创新在于双层转化：先做关节层与观测层的动力学对齐，再用轻量化参数高效微调（PEFT）仅更新与动力学相关的模块，既保留原始行为先验又快速适配新形态；

**🔧 技术方法**

采用PEFT技术（LoRA、Adapter、Prefix-Tuning）配合关节对齐矩阵、Hip Decoupling、Parallel Coupling等自定义对齐策略，并在PPO训练框架下进行；

**📊 数据集**

使用AMASS运动数据集的重定向样本，覆盖多类人机器人（LimX Oli/Luna、Unitree G1/H1）进行实验；

**📈 对比分析**

与从零开始训练以及其他PEFT方法对比，Any2Any在训练样本和计算成本仅为原来的~1%时即可达到甚至超过原始模型的跟踪奖励、MPJPE和基座误差；

**⚠️ 局限性**

局限性包括对极度不同拓扑或关节配置的机器人迁移效果未知，且对关节对齐矩阵的手工定义仍需专家知识。

---

## 471. Weierstrass Positional Encoding for Vision Transformers

**arXiv ID:** 2605.23719 | [PDF](https://arxiv.org/pdf/2605.23719v1)

**作者:** Zhihang Xin `[一作]` (Jiangnan University), Xiaojun Wu `[通讯]` (Jiangnan University)

**通讯引用:** 50843 | [OpenAlex ID](https://openalex.org/A5021767311)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

为视觉 Transformer 提出基于 Weierstrass 椭圆函数的二维位置编码（WePE），实现连续、解析且与分辨率无关的空间表示。

**💡 创新点**

创新点在于利用椭圆函数的双周期性与加法公式，天然保留 2D 结构并提供距离衰减与相对位置信息，突破传统线性或正弦编码对几何约束的缺失。

**🔧 技术方法**

技术方法包括复平面映射、Weierstrass ℘(z) 与其导数的实虚分解、tanh 压缩、可学习半周期与全局缩放、预计算 LUT 以及与 Transformer 结构无缝对接的线性/MLP 投影。

**📊 数据集**

实验覆盖 CIFAR‑100、ImageNet‑1k、ImageNet‑21k、VTAB‑1k 等标准视觉数据集，并在多种 Vision Transformer 变体上验证。

**📈 对比分析**

与 APE、RoPE、FoPE、2D‑Sin、LieRE 等现有编码进行对比，WePE 在从零训练和微调场景下均能提升 Top‑1/Top‑5 准确率，平均提升约 3–4% 以上，且延迟与显存开销几乎无增。

**⚠️ 局限性**

局限性包括对网格尺度超参数（α_u, α_v）敏感、LUT 需要额外显存、实现复杂度较高，且在极端高分辨率或某些下游任务中提升有限。

---

## 472. Not Too Generative, Not Too Discriminative: The Human Alignment Sweet Spot

**arXiv ID:** 2605.23819 | [PDF](https://arxiv.org/pdf/2605.23819v1)

**作者:** Jorge Chang Ortega `[一作]` (ANITI), Victor Boutin `[通讯]` (CNRS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用Joint Energy-Based Models在固定架构下通过一个混合系数α在判别式和生成式学习之间连续插值，评估模型在人类视觉对齐方面的表现。

**💡 创新点**

创新之处在于通过单一架构和可调节的混合系数彻底隔离学习目标，证明在人类对齐的六项多维基准上，最佳表现始终出现在判别与生成的混合区间，而非两端极值。

**🔧 技术方法**

采用Joint Energy-Based Models (JEM)、对抗性采样（SGLD）与对比散度来实现生成式和判别式损失的联合优化，并通过混合系数α平衡两者。

**📊 数据集**

实验使用ImageNet、CIFAR-10、BAPPS（低级感知）、合成光泽数据集（中级感知）、ClickMe（诊断特征）、CIFAR-10H（不确定性）以及多种OOD变换数据集。

**📈 对比分析**

将混合JEM与传统判别模型、纯生成模型以及大型生成器（如Stable Diffusion、Imagen）进行对比；在六个基准上，α≈0.5~0.6的混合模型在感知相似度、光泽判断、分类不确定性、分布外鲁棒性、形状/纹理冲突偏好及诊断特征对齐等方面均优于两端模型，并在多数任务接近人类上限。

**⚠️ 局限性**

局限性包括JEM训练计算量大、收敛不稳定、最优α随任务变化、仅在特定架构和数据规模下验证，且未必能提升公平性、鲁棒性或神经预测能力。

---

## 473. A Pragmatic Approach to Learned Indexing in RocksDB: Targeted Optimizations with Minimal System Modification

**arXiv ID:** 2605.23815 | [PDF](https://arxiv.org/pdf/2605.23815v1)

**作者:** Shubham Vashisth `[一作]` (McGill University), Oana Balmau `[通讯]` (McGill University)

**通讯引用:** 441 | [OpenAlex ID](https://openalex.org/A5055620336)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文探讨了如何将学习索引集成到生产数据库系统中，特别是针对写密集型工作负载的优化，提出了一种名为的扩展，旨在通过最小的系统修改实现学习索引的有效应用。

**💡 创新点**

创新点在于提出了一种骨架化机制，能够在Memtable实例之间保留结构知识，从而解决了学习索引在频繁替换Memtable时的冷启动问题，同时在磁盘层面用学习索引替代传统索引，保持了存储层的其他部分不变。

**🔧 技术方法**

使用了学习索引技术，特别是针对Memtable的可更新学习索引和针对SST的块感知学习索引，结合了骨架化优化。

**📊 数据集**

使用了六个真实世界数据集进行评估，涵盖了不同的键分布和访问模式，包括Amazon Reviews、Facebook、OSM、YCSB、Covid和Genome等。

**📈 对比分析**

与最先进的学习索引解决方案相比，的写入速度提高了1.5倍，读取速度提高了2.1倍，显示出在LSM架构中集成学习索引的可行性和有效性。

**⚠️ 局限性**

限制在于当前的实现主要针对数字键，未来的工作可以探索将学习索引技术扩展到字符串等其他数据类型，同时需要进一步验证在不同工作负载下的适应性。

---

## 474. SkillOpt: Executive Strategy for Self-Evolving Agent Skills

**arXiv ID:** 2605.23904 | [PDF](https://arxiv.org/pdf/2605.23904v1)

**作者:** Yifan Yang `[一作]`, Chong Luo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的文本空间优化器，用于训练代理技能，通过将技能文档视为可训练的外部状态来优化代理的执行能力。

**💡 创新点**

创新点在于将技能优化视为一种可控的领域适应过程，使用额外的优化模型来提出结构化的增删改编辑，并通过严格的验证门控来接受编辑。

**🔧 技术方法**

使用了文本空间优化技术，包括回滚批次、反思小批量、增删改编辑、文本学习率、调度、验证门控、拒绝编辑缓冲区和周期性慢/元更新。

**📊 数据集**

在六个基准测试上进行了评估，包括搜索问答、电子表格、文档、数学和具身决策等，涉及七个目标模型和三种执行模式。

**📈 对比分析**

与无技能、人工技能、一次性LLM技能、提示优化（TextGrad、GEPA）和技能演化（Trace2Skill、EvoSkill）等七个基线进行比较，结果显示在52个评估单元中均为最佳或并列最佳，平均提升23.5分。

**⚠️ 局限性**

限制在于该方法依赖于优化器的能力，且当前仅针对单一目标领域进行优化，未来可扩展至技能库和跨领域共享基础设施。

---

## 475. HorizonStream: Long-Horizon Attention for Streaming 3D Reconstruction

**arXiv ID:** 2605.23889 | [PDF](https://arxiv.org/pdf/2605.23889v1)

**作者:** Chong Cheng `[一作]` (HKUST(GZ)), Hao Wang `[通讯]` (HKUST(GZ))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的在线3D重建框架HorizonStream，旨在在严格的因果和有限内存约束下估计相机姿态和场景几何。

**💡 创新点**

创新点在于将几何传播形式化为证据影响核，并提出了一种长时间范围的Transformer架构，能够有效处理不同时间尺度的几何证据。

**🔧 技术方法**

使用了HorizonStream框架，结合了几何线性注意力和几何局部注意力技术，采用了度量读出令牌来恢复稳定的尺度和刚性姿态。

**📊 数据集**

在多个数据集上进行了实验，包括KITTI、vKITTI2、Oxford Spires、ScanNet++、TUM RGB-D、Waymo Open、VBR、ETH3D和7Scenes。

**📈 对比分析**

与现有的流式3D重建方法相比，HorizonStream在处理超过10,000帧的序列时表现出色，保持稳定的姿态估计，并在多个基准测试中超越了所有流式方法，接近或超过了离线方法的性能。

**⚠️ 局限性**

局限性在于对于极长序列的细节捕捉可能不足，动态前景物体可能会干扰输入视频中的局部几何证据，且可选的闭环模块的优化设置可能需要进一步调整。

---

## 476. CHRONOS: Temporally-Aware Multi-Agent Coordination for Evolving Data Marketplaces

**arXiv ID:** 2605.23887 | [PDF](https://arxiv.org/pdf/2605.23887v1)

**作者:** Joydeep Chandra `[一作]` (Tsinghua University), Joydeep Chandra `[通讯]` (Tsinghua University)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5003396109)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种三层架构 CHRONOS，用于在共享差分隐私预算下，实时维护时序知识图谱的索引、估价与协调；

**💡 创新点**

创新点包括：①神经ODE时间衰减索引提供可证明的召回下界；②事件感知的多卖家价值评估（EC‑MPV）与变化点检测；③EXP3‑IX调度协调器实现无退化的隐私预算管理与子线性调度损失；

**🔧 技术方法**

技术手段包括：神经ODE衰减、BOCPD变化点检测、VRDS抽样优化、Gaussian机制固定维度私有矩阵发布、EXP3‑IX多臂赌博与zCDP/PLD隐私计数；

**📊 数据集**

使用四个基准数据集：FB15K‑237、WN18RR（合成 Poisson 变化）、MIMIC‑IV 与 Yelp（真实时间戳）进行实验；

**📈 对比分析**

与多种基线（Plain‑HNSW、TigerVector、FreshDiskANN、VSAG、Data Shapley、无协调等）对比，CHRONOS 在召回@10 上达到0.937、查询速率2.74 QPS、P50延迟161 ms，总 ε=4.25，显著优于单项方法并在 DP 预算下实现更高效的实时检索；

**⚠️ 局限性**

主要局限：私有信号噪声仍然占主导（affinity σ_entry=885、valuation noise 4.0），对高 β 或冷启动场景召回下降；可信管理员假设严格，需进一步验证两服务器或本地 DP 的实现。

---

## 477. Multilingual Knowledge Transfer under Data Constraints via Lexical Interventions

**arXiv ID:** 2605.23885 | [PDF](https://arxiv.org/pdf/2605.23885v1)

**作者:** Anastasiia Sedova `[一作]` (Apple), Maartje ter Hoeve `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种只需双语词典即可在大规模预训练中通过词级替换实现低资源语言知识迁移的方法。

**💡 创新点**

创新点在于将跨语言混合简化为极其轻量级的词级置换，完全不依赖平行语料、翻译系统或额外模型，适用于任何具备词典的语言。

**🔧 技术方法**

技术核心是根据混合比例与替换比例在高资源语料中随机将单词替换为低资源语言翻译，并支持域特定插入；实现上仅需字典查找，开销极低。

**📊 数据集**

使用 English FineWeb、FineWeb2 等公开语料作为高资源语料；低资源模拟集为德语、法语、印地语、中文约350M token；真实低资源集包括 Swahili、Yoruba、Amharic、Igbo；词典来源主要是 Wiktionary。

**📈 对比分析**

在多任务零-shot QA 基准（ARC、Hellaswag、Lambada、PiQA、SciQ、Winogrande）与对照模型（仅低资源、低资源+高资源、上限）进行对比，LEXI 在所有低资源语言上均显著提升性能，部分情况下甚至超过上限；域特定插入在保持高资源语言性能的同时同样提升低资源语言；并实现约 2 倍的训练加速。

**⚠️ 局限性**

局限性包括：需要词典覆盖率足够；高资源语言性能会因大量替换而下降；对极其低资源或语系距离很大的语言（如 Amharic）效果有限；以及需手动调节混合与替换比例。

---

## 478. PGT: Procedurally Generated Tasks for improving visual grounding in MLLMs

**arXiv ID:** 2605.23883 | [PDF](https://arxiv.org/pdf/2605.23883v1)

**作者:** Rim Assouel `[一作]` (Mila - Québec AI Institute), Adriana Romero-Soriano `[通讯]` (Mila - Québec AI Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为程序生成任务（PGT）的框架，旨在提高多模态大语言模型（MLLMs）在细粒度视觉理解任务中的表现，并作为低成本的诊断工具来识别感知失败的来源。

**💡 创新点**

PGT通过在图像上叠加明确的几何原语，提供额外的密集监督信号，从而将视觉基础能力与语义先验分离，显著提升了模型在多种基准测试中的表现。

**🔧 技术方法**

使用了程序生成的几何叠加任务，结合了多种模型架构进行训练和微调。

**📊 数据集**

在多个基准测试上进行了实验，包括What's Up、CV-Bench-2D、CV-Bench-3D等，展示了PGT在不同模型架构上的有效性。

**📈 对比分析**

与基线模型相比，使用PGT增强的模型在细粒度视觉理解和空间推理任务上表现出显著的性能提升，具体表现为在What's Up基准上提高了20%，在CV-Bench-2D上提高了13.3%。

**⚠️ 局限性**

PGT的局限性在于它并不是替代数据扩展或架构增强的解决方案，而是作为一种高效的补充信号，当前的任务集仅为概念验证，未来需要扩展更多任务以覆盖更广泛的应用场景。

---

## 479. LaMo: Self-Supervised Latent Motion Priors for Physical Realism in Video Generation

**arXiv ID:** 2605.23878 | [PDF](https://arxiv.org/pdf/2605.23878v1)

**作者:** Bo Jiang `[一作]` (Applied Intuition), Wei Zhan `[通讯]` (Applied Intuition)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种自监督的潜在运动先验框架，利用未标注视频的帧间潜在差分来指导视频扩散模型的训练与采样。

**💡 创新点**

创新点在于：1）完全从未标注视频中提取运动先验，无需物理模拟器、教师模型或标注数据；2）将先验拆分为宏观运动漂移损失（训练时）和微观运动场引导（推理时），两者在不同阶段共同提升物理一致性；3）在噪声预测空间实现轻量级运动引导，兼容现有模型。

**🔧 技术方法**

技术细节包括：使用预训练的3D因果VAE编码视频为潜在；计算τ步潜在差分作为运动目标；宏观漂移损失采用归一化L2正则化；微观运动场由小型CNN预测，作为CFG引导梯度；整个框架对原有视频扩散骨干保持不变。

**📊 数据集**

使用的主要数据集：OpenVid用于模型训练；VideoPhy、VideoPhy2用于物理一致性评估；VBench用于综合质量与运动相关指标评估。

**📈 对比分析**

在VideoPhy和VideoPhy2上，相较于CogVideoX和多种外部监督的物理感知方法，显著提升语义一致性(SA)和物理常识得分(PC)，在VBench上保持甚至提升总体质量与运动相关维度。

**⚠️ 局限性**

局限性：1）仅提供可观测的运动一致性，无法强制满足完整的物理约束；2）效果受训练视频覆盖范围与评测指标限制，未能针对特定物理现象（接触、变形、流体守恒等）进行精细约束。

---

## 480. From Activation to Causality: Discovery of Causal Visual Representations in the Human Brain

**arXiv ID:** 2605.23895 | [PDF](https://arxiv.org/pdf/2605.23895v1)

**作者:** Yuval Golbari `[一作]` (Weizmann Institute of Science), Tamar Rott Shaham `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出BrainCause框架，通过生成与检索的图像与脑功能模型实现视觉概念的因果定位。

**💡 创新点**

创新点在于结合生成式模型、语言模型与图像‑脑编码器，构建针对概念的因果控制图像集合，并用因果得分筛选真正代表概念的脑区。

**🔧 技术方法**

使用文本到图像模型（FLUX.2）、大语言模型（Gemma‑3）、视觉语言模型（Qwen3‑VL）、图像‑脑编码器以及统计检验方法。

**📊 数据集**

在NSD（Natural Scenes Dataset）7T fMRI数据上进行评估。

**📈 对比分析**

与Max Activation、MindSimulator、MindSimulator+VLM等激活基方法比较，BrainCause在保持高激活度的同时将因果得分提升至约0.7，显著降低假阳性率（从73%降至23%）并提升真阳性率。

**⚠️ 局限性**

局限在于对生成、检索和编辑模型的依赖，可能出现概念缺失或语义负样本生成失败，导致某些相关因素未被充分排除。

---

## 481. Good Token Hunting: A Hitchhiker's Guide to Token Selection for Visual Geometry Transformers

**arXiv ID:** 2605.23892 | [PDF](https://arxiv.org/pdf/2605.23892v1)

**作者:** Shuhong Zheng `[一作]` (University of Toronto & Vector Institute), Igor Gilitschenski `[通讯]` (University of Toronto & Vector Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种训练无关的两阶段层次化 token 选择策略，用于加速视觉几何变压器（VGGT）在多视角 3D 重建中的全局注意力计算。

**💡 创新点**

创新点包括：①基于多样性（K‑center）原则的帧级 token 选择；②针对不同层的注意力分布设计自适应的帧内 token 选择；③通过只保留有限数量的 key/value token，既显著降低算力，又在部分任务上提升了精度。

**🔧 技术方法**

核心技术包括：全局注意力的 token 预算约束、帧级多样性采样（Farthest Point Sampling）、层级注意力熵分析、帧内非均匀 down‑sampling 与局部注意力替换。

**📊 数据集**

实验数据集涵盖：7-Scenes、Neural RGB‑D、TUM‑Dynamics、Bonn、Sintel、KITTI 等，覆盖相机姿态估计、点云重建与视频深度估计三大任务。

**📈 对比分析**

与 FastVGGT、SparseVGGT、LiteVGGT、Speed3R、Co‑Me 等现有加速方法比较，本文在保持或提升精度的同时，提升了 80%+ 的推理速度，尤其在 500 张图像场景下实现了 85% 的加速。

**⚠️ 局限性**

局限性在于：①方法完全无监督，无法进一步通过学习提升 token 选择的最优性；②需要手动设定预算与层阈值，虽然对性能影响不大但仍需经验；③对动态或大规模场景的可扩展性未在极端条件下完全验证。

---

## 482. Training-Free Looped Transformers

**arXiv ID:** 2605.23872 | [PDF](https://arxiv.org/pdf/2605.23872v1)

**作者:** Lizhang Chen `[一作]`, Qiang Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的循环 Transformer 包装器，能在保持模型权重冻结的情况下，在推理时对中间层块进行多次迭代。

**💡 创新点**

创新点在于将预归一化 Transformer 层解释为一阶欧拉步长为 1 的 ODE 步进，并利用数值积分（如 Runge–Kutta）在同一时间段内以更细的子步长进行迭代，从而在不训练、无额外参数的前提下提升推理性能。

**🔧 技术方法**

主要技术包括：数值积分方法（前向欧拉、Runge–Kutta、Anderson、Aitken 等）、区块模式与层模式迭代、对 MoE 模型的层模式迭代以避免专家路由抖动。

**📊 数据集**

使用了多种知识密集型多项选择基准数据集：MMLU-Pro、GPQA-Main、ARC-Challenge、CommonsenseQA、OpenBookQA 等。

**📈 对比分析**

与无循环基线及先前的“naive”循环方法对比，实验显示在 7 大模型族和 45 个 (模型, 基准) 组合中，最强配置在 MMLU-Pro 上提升约 +2.64 百分点、ARC-Challenge 上提升约 +2.30 百分点，整体 87% 的实验细胞表现非负提升。

**⚠️ 局限性**

局限性包括：在小型或高度蒸馏模型上表现不佳；需要根据模型深度选择窗口位置；目前仅针对已公开的冻结检查点，无法直接适用于仍在训练中的模型。

---

## 483. Strong Teacher Not Needed? On Distillation in LLM Pretraining

**arXiv ID:** 2605.23857 | [PDF](https://arxiv.org/pdf/2605.23857v1)

**作者:** Taiming Lu `[一作]`, Zhuang Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了大语言模型（LLM）预训练中教师-学生关系的知识蒸馏效果，探讨了弱教师对强学生、强教师对弱学生以及同级教师的影响。

**💡 创新点**

创新点在于：①挑战传统“强教师优于弱教师”的假设，发现弱教师在合适的损失混合比例下仍能显著提升学生；②证明更强教师并不总能得到更好学生，尤其在大模型过度训练时会适得其反；③在多种评估维度（in-domain、out-of-domain perplexity、downstream accuracy）上系统展示不同关系的表现差异。

**🔧 技术方法**

使用标准的自回归语言建模损失 + 软目标蒸馏损失（KL散度），通过不同的损失混合系数 α 进行调节；采用 Llama3 系列 0.7B、1.7B、3.8B、8.0B 四个规模的模型；采用 FineWeb‑Edu 作为预训练语料；在实验中还利用 11 个高质量 OOD 语料库和 15 个下游评测基准。

**📊 数据集**

主要数据集包括 FineWeb‑Edu（预训练）、Wikitext‑103、C4、GSM8K、DM Mathematics、HumanEval、CodeSearchNet、arXiv、CNN‑DailyMail、ECHR、PubMedQA、XQuAD 等 OOD 语料，以及 MMLU、ARC‑Easy、SciQ、OpenBookQA、MathQA、TruthfulQA、ANLI‑R1、CommonsenseQA、HellaSwag、PIQA、WinoGrande、Social IQa、LogiQA 2.0、MedMCQA、RACE 等下游任务。

**📈 对比分析**

对比方法：以 α = 0 的标准预训练基线为对照；在每种教师-学生配置下搜索最佳 α；报告所有评测维度的百分比改进。结果显示：弱教师在合适 α 下可提升 1–5% 下游准确率，OOD perplexity 最高可提升约 8.9%，in‑domain perplexity 提升有限（约 0–4%）。相同级教师同样能获得正向提升；强教师过度训练时效果出现倒退。

**⚠️ 局限性**

局限性：①仅在 Llama3 系列模型上验证，未探究其他架构或跨家族蒸馏；②仅考虑了预训练阶段的蒸馏，未评估对生成任务的影响；③蒸馏损失为单一 KL，未尝试温度缩放、中间层匹配等技巧；④实验规模受算力限制，Token 预算和模型尺寸仍有限。

---

## 484. PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion

**arXiv ID:** 2605.23902 | [PDF](https://arxiv.org/pdf/2605.23902v1)

**作者:** Yifan Lu `[一作]` (Nvidia), Xuanchi Ren `[通讯]` (Nvidia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种Pixel Diffusion Decoder（PiD），通过条件像素级扩散直接从VAE或语义编码器得到的潜变量中生成高分辨率图像，取代传统的低分辨率解码加超分阶段。

**💡 创新点**

创新点包括：①将潜变量解码与超分统一为单一的条件像素扩散过程；②引入轻量级sigma‑aware适配器，允许在潜变量部分去噪时直接解码，实现基于潜变量的早停；③利用DMD2蒸馏将多步扩散压缩到仅四步，显著降低推理延迟。

**🔧 技术方法**

技术主要包括PixelDiT（基于MMDiT的像素扩散网络）、控制网络风格的潜变量投影与注入、sigma‑aware门控机制、分布匹配蒸馏（DMD2）以及基于多模态LLM的质量对比评估。

**📊 数据集**

使用了MultiAspect‑4K‑1M、PDF渲染数据、内部高分辨率图像集合以及多种语义潜变量（FLUX.1/2 VAE、Stable Diffusion 3 VAE、DINOv2、SigLIP）来训练和评估模型。

**📈 对比分析**

与原始VAE/RAE解码+单步超分、SSDD、LUA等基线相比，PiD在多项无参考图像质量指标（MUSIQ、NIQE、DEQA、MANIQA、Unipercept、VisualQuality‑R1）上取得领先，推理延迟在单张4K图像上仅为210 ms（GB200 GPU）或1 s（RTX 5090），比基线快3–6倍且视觉质量更好。

**⚠️ 局限性**

局限性包括：①仍需在大型像素扩散先验上预训练，训练成本高；②对潜变量噪声的适配范围有限，极端低质量潜变量解码效果尚待验证；③在极大分辨率（4K以上）下对显存与吞吐量要求较高，尚未完成完整的跨模型通用化。

---

## 485. Smart-Insertion-V: Photorealistic Video Insertion via a Closed-Loop Feedback Dual-Stream Framework

**arXiv ID:** 2605.23891 | [PDF](https://arxiv.org/pdf/2605.23891v1)

**作者:** Xiao Cao `[一作]`, Xuelong Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Smart-Insertion-V的端到端双流框架，用于将原始参考对象无缝插入视频中，同时进行图像风格转移。

**💡 创新点**

创新点在于引入了去耦指导模块（DGM）和双世界视图位置嵌入（Dual-RoPE），有效解决了特征纠缠和风格泄漏的问题，并实现了高保真度的插入效果。

**🔧 技术方法**

使用了双流架构，结合了视觉-语言模型（VLM）和原生文本编码器（T5），并采用闭环反馈机制来优化视频生成过程。

**📊 数据集**

构建了一个高质量的开放源数据集，包含源视频、目标视频、未处理的参考图像和和谐的参考图像，数据通过自动化的数据策划管道生成。

**📈 对比分析**

与现有的显式空间指导方法、无掩膜方法和级联方法进行比较，Smart-Insertion-V在所有指标上均表现优越，尤其在风格和空间一致性方面。

**⚠️ 局限性**

限制在于VLM的使用可能导致细粒度空间控制的不足，未来的工作将探索如何根据用户意图进行更精确的插入位置调整。

---

## 486. Vision Transformers Need Better Token Interaction

**arXiv ID:** 2605.23868 | [PDF](https://arxiv.org/pdf/2605.23868v1)

**作者:** Linxiang Su `[一作]` `[通讯]` (University of Szeged), Linxiang Su (University of Szeged)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了在Vision Transformer中使用稀疏注意力（entmax‑1.5）以抑制语义扩散并提升稠密预测表现，保持全局感知能力不变。

**💡 创新点**

将softmax注意力替换为entmax‑1.5，简单且无额外参数，直接让token交互更具选择性，从而减少全局语义泄漏。

**🔧 技术方法**

稀疏注意力(entmax‑1.5)；基准对比使用softmax、register tokens、CLS/patch specialization等方式。

**📊 数据集**

ImageNet‑1K（预训练）、VOC、ADE20K、Cityscapes、NYUv2、SUN RGB‑D、KITTI。

**📈 对比分析**

与标准softmax、register token、CLS/patch specialization进行线性分类/分割/深度探测评测；entmax‑1.5在ImageNet线性检索上提升0.56%，在VOC语义分割上提升约6.0 mIoU，ADE20K +2.1 mIoU，Cityscapes +1.1 mIoU，且在深度任务上也有轻微提升。

**⚠️ 局限性**

仅在DINOv1 ViT‑S/16 200 epoch级别下验证，未测试更大模型或更长训练；深度估计提升有限；稀疏注意力可能导致实现开销。

---

## 487. Human Decision-Making with Persuasive and Narrative LLM Explanations

**arXiv ID:** 2605.23867 | [PDF](https://arxiv.org/pdf/2605.23867v1)

**作者:** Laura R. Marusich `[一作]` (DEVCOM Army Research Laboratory), Murat Kantarcioglu `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过大规模人类行为实验评估了不同说服力的LLM生成叙事解释对人类决策准确率和依赖性的影响。

**💡 创新点**

首次系统比较了叙事解释不同说服力度对客观决策性能的影响，并发现更高说服力并未提升准确率，反而可能降低决策速度。

**🔧 技术方法**

采用OpenAI GPT‑4o模型生成预测及三种说服力（中性、低说服、极端说服）的叙事解释，并使用文本与情感分析验证说服度差异。

**📊 数据集**

使用UCI公开的Census Income与Student Performance两大数据集，共58条样本（每个数据集29条）。

**📈 对比分析**

对照组仅提供AI预测，实验组在此基础上加入叙事解释；结果显示AI预测准确率在Census数据集中约75%，学生数据集中约68%；叙事解释未显著提升人类准确率，但提高了对AI的依赖；极端说服条件导致响应时间明显延长。

**⚠️ 局限性**

局限包括使用较旧的LLM版本、一次性非交互式说服、样本量相对有限、未探索对抗性说服和交互式个性化策略。

---

## 488. Robotic Strawberry Harvesting with Robust Vision and Deep Reinforcement Learning based Sim-to-Real Control

**arXiv ID:** 2605.23863 | [PDF](https://arxiv.org/pdf/2605.23863v1)

**作者:** Al Bashir `[一作]` (Texas A&M University), Azlan Zahid `[通讯]` (Texas A&M University)

**通讯引用:** 1303 | [OpenAlex ID](https://openalex.org/A5033568355)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本研究提出了一种闭环机器人草莓采摘系统，结合了强大的视觉模块、基于模拟训练的深度强化学习（DRL）控制和基于ROS的真实机器人执行。

**💡 创新点**

创新点在于提出了HRAttnEdge-YOLO26-seg模型，该模型通过高分辨率P2分支、分割路径注意力和边缘监督原型学习来改善复杂场景中的实例分割性能。

**🔧 技术方法**

使用了深度强化学习中的近端策略优化（PPO）技术，并在Isaac Lab中进行模拟训练，以生成UR10e机械手的平滑关节位置命令。

**📊 数据集**

使用了自收集的草莓数据集和公开的StrawDI数据集进行模型训练和评估，自收集数据集包含647张图像，公开数据集包含3100张图像。

**📈 对比分析**

与基于逆运动学（IK）的MoveIt基线方法相比，PPO控制器在稳定性和动态平滑性方面表现更好。在温室试验中，系统成功采摘了281个草莓，达到96.6%的到达成功率和84.3%的整体采摘成功率。

**⚠️ 局限性**

限制在于当前的3D目标生成依赖于分割质心和局部深度值，而不是完整的6D果实姿态或果梗几何形状，可能不适用于更复杂的采摘策略。

---

## 489. LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws

**arXiv ID:** 2605.23901 | [PDF](https://arxiv.org/pdf/2605.23901v1)

**作者:** Xu Ouyang `[一作]` (ByteDance Seed), Yiyuan Ma `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的Shannon Scaling Law，将大型语言模型（LLM）视为信息传输过程，重新定义了模型的能力和训练动态。

**💡 创新点**

创新点在于将LLM的训练视为噪声信道，提出了一个统一的扩展定律，能够解释和预测模型在不同噪声条件下的性能变化，特别是U型损失曲线的出现。

**🔧 技术方法**

使用了Shannon-Hartley定理作为理论框架，结合了带宽、信号和噪声的功率法则，提出了一个新的扩展定律。

**📊 数据集**

使用了Pythia和OLMo2两个开源模型套件，数据集包括去重的Pile数据集和wikitext2数据集。

**📈 对比分析**

与现有的扩展定律进行了比较，Shannon Scaling Law在各种扰动场景下（如量化、SFT和高斯噪声）表现优越，尤其在高噪声条件下，R^2值显著高于其他基线。

**⚠️ 局限性**

限制在于完整的Shannon Scaling Law需要9个拟合常数，尽管简化版本只需6个常数，但在某些情况下可能会导致过拟合。

---

## 490. From Raw Experience to Skill Consumption: A Systematic Study of Model-Generated Agent Skills

**arXiv ID:** 2605.23899 | [PDF](https://arxiv.org/pdf/2605.23899v1)

**作者:** Zisu Huang `[一作]`, Chong Luo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本研究系统性地研究了模型生成的代理技能，涵盖了技能生命周期的各个阶段，包括经验生成、技能提取和技能消费。

**💡 创新点**

创新点在于建立了一个基于效用的评估框架，系统地分析了模型生成的技能在不同代理和任务领域中的有效性，并提出了一种具体的元技能来指导技能提取。

**🔧 技术方法**

使用了统一的提取框架，结合了多种模型（如GPT和Gemini）进行技能提取和评估。

**📊 数据集**

研究涵盖了五个不同的任务领域，包括身体交互、生产力软件、软件工程、网络搜索和工具调用。

**📈 对比分析**

通过比较不同提取器和目标模型的性能，发现模型生成的技能在75%的情况下能提高下游性能，但也存在25%的负迁移风险，且提取器的性能与其执行能力并不总是相关。

**⚠️ 局限性**

限制在于模型生成的技能虽然在平均上是有益的，但存在显著的变异性和负迁移风险，且模型规模或文本可行性并不能可靠地预测下游效用。

---

## 491. SPACENUM: Revisiting Spatial Numerical Understanding in VLMs

**arXiv ID:** 2605.23898 | [PDF](https://arxiv.org/pdf/2605.23898v1)

**作者:** Jianshu Zhang `[一作]` (Northwestern), Han Liu `[通讯]` (Northwestern)

**通讯引用:** 143930 | [OpenAlex ID](https://openalex.org/A5100338921)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过SpaceNum框架重新审视视觉语言模型（VLMs）在空间环境中的数字理解能力，提出了两个双向任务Num2Space和Space2Num，以评估VLMs在视觉空间结构和语言数字表示之间的映射能力。

**💡 创新点**

创新点在于系统性地研究当前VLMs在动态过渡和静态布局中的数字理解能力，发现它们在空间意义上未能有效地将数字与视觉信息结合，且表现接近随机猜测。

**🔧 技术方法**

使用了结构化错误分析、推理轨迹分析和控制干预等技术，评估VLMs的表现和理解能力。

**📊 数据集**

数据集包括在AI2-THOR中生成的动态过渡数据和在NVIDIA Isaac Sim中构建的静态布局数据，涵盖了多种空间场景和布局。

**📈 对比分析**

与其他方法比较时，当前VLMs在空间数字理解上表现不佳，最佳模型的平均准确率仅为39.8%，且在动态过渡和静态布局中表现存在显著差异。

**⚠️ 局限性**

限制在于本研究主要集中于受控的空间设置，未来需要将空间数字理解扩展到更开放的真实场景和连续空间预测设置，同时对VLMs内部的空间推理机制仍需深入探索。

---

## 492. Complete-muE: Optimal Hyperparameter Transfer and Scaling for MoE Models

**arXiv ID:** 2605.23893 | [PDF](https://arxiv.org/pdf/2605.23893v1)

**作者:** Hongwu Peng `[一作]` (Adobe Research), Yan Kang `[通讯]` (Adobe Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种完整的muE框架，用于在稠密全连接网络（FFN）与Mixture-of-Experts（MoE）Transformer块之间迁移AdamW超参数，支持激活专家数、总专家数、粒度、共享专家及组均衡路由等各种MoE配置的迁移。

**💡 创新点**

创新点在于构建了两条桥接路径：Bridge I利用激活宽度μP和归一化路由尺度将稠密FFN映射到全激活MoE；Bridge II通过激活专家缩放和SDE分析将全激活MoE映射到稀疏MoE，并证明了LR/WD的一阶校正可抵消，只剩下有限的σ₀漂移；将两者组合得到完整的“Complete‑muE”迁移规则，覆盖了MoE所有常见的扩展轴。

**🔧 技术方法**

技术方法包括：μP宽度匹配、归一化路由尺度、SDE（Stochastic Differential Equation）训练轨迹分析、层级级联迁移、归一化专家输出、以及对共享/组均衡路由的统一处理；实验中使用了大规模LM与扩散模型的代理训练、单GPU H100延迟基准以及多模态/大规模训练验证。

**📊 数据集**

数据集与模型：语言模型代理使用GPT‑NeoX‑20B分词器、BPE；扩散模型代理使用512×512图像、256P、512P、240P关键帧与240P 5s视频的Latent Diffusion Transformer；大规模实验使用C4文本、256P/512P图像、240P视频、以及512词长的LLM；最终在多模态生成和LLM下进行性能评估。

**📈 对比分析**

评估方法：在代理实验中做激活专家、总容量、粒度、共享专家、组均衡路由、深度、宽度以及批大小/训练时长等轴的超参数扫描；在单GPU延迟基准中对比容量扩展与粒度扩展的延迟；在大规模实验中记录收敛速度（dense步骤/ MoE步骤）、最终损失以及多模态/LLM的基准分数。结果显示：MoE模型在所有轴上均可获得2.5–5.5倍的收敛加速，容量扩展的GPU延迟仅比稠密模型慢1.1–1.2倍，粒度扩展延迟显著升高；超参数迁移导致的漂移极小，稠密模型一次调参即可在所有MoE配置上保持最优或接近最优性能。

**⚠️ 局限性**

局限性：Bridge II的非严格SDE假设导致在激活专家数变动、容量扩展或固定步骤/总样本迁移时仍存在有限的σ₀漂移；方法假设近似负载平衡和token‑choice路由，可能不适用于其他路由策略或极端专家数；在非常大规模的GPU集群或不同硬件上，实际硬件瓶颈与理论规则可能产生偏差。

---

## 493. Divergent Paths to Depolarization: Dialogue Design Determines the Prosocial Benefits of AI-Assisted Political Argumentation

**arXiv ID:** 2605.23890 | [PDF](https://arxiv.org/pdf/2605.23890v1)

**作者:** Jianlong Zhu `[一作]` (Saarland University), Ingmar Weber `[通讯]` (Saarland University)

**通讯引用:** 11546 | [OpenAlex ID](https://openalex.org/A5033656008)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了人机对话在减少政治极化中的作用，特别是通过AI聊天机器人进行的辩论设计对社会效益的影响。

**💡 创新点**

创新点在于比较了态度一致和态度不一致的对话形式对情感和意见极化的影响，发现态度一致的对话在短期内更有效。

**🔧 技术方法**

使用了AI聊天机器人作为对话伙伴，进行了一系列在线实验，采用了2×2×2的因子设计来测试不同的论点侧、对话模式和财务激励。

**📊 数据集**

数据集来自于469名美国参与者，他们在两次在线实验中参与了与AI聊天机器人的辩论。

**📈 对比分析**

与非AI参考任务相比，态度一致的AI对话在减少情感极化和意见极化方面表现出显著优势，而态度不一致的对话则未能显示出相同的效果。

**⚠️ 局限性**

限制在于样本的代表性问题，参与者主要来自Prolific平台，可能更具数字素养且经济状况较差，此外，实验设计未能充分评估AI对话的长期效果。

---

## 494. Entrywise Error Bounds for Spectral Ranking with Semi-Random Adversaries

**arXiv ID:** 2605.23854 | [PDF](https://arxiv.org/pdf/2605.23854v1)

**作者:** Dongmin Lee `[一作]` (Purdue University), Japneet Singh `[通讯]` (Purdue University)

**通讯引用:** 53 | [OpenAlex ID](https://openalex.org/A5026924551)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在半随机对手模型下，谱算法的条目误差，特别是针对布拉德利-特里-卢斯（BTL）模型的估计方法。

**💡 创新点**

创新点在于提出了一种边加权的谱方法，通过重新加权观察到的边来抵消对手的影响，从而恢复谱间隙，进而提高估计的准确性。

**🔧 技术方法**

使用了谱算法和边加权技术，分析了谱方法在半随机对手模型下的表现。

**📊 数据集**

使用了半随机生成的图和随机块模型（SBM）作为数据集进行实验和理论分析。

**📈 对比分析**

与传统的均匀采样图相比，提出的加权谱方法在处理半随机对手时能够保持与均匀采样图相似的误差界限，尤其在图密度较高时表现良好。

**⚠️ 局限性**

限制在于对于谱间隙逐渐消失的图，当前的理论结果无法提供最佳的误差界限，且在稀疏半随机图中，最佳界限尚未得到解决。

---

## 495. Geo-Align: Video Generation Alignment via Metric Geometry Reward

**arXiv ID:** 2605.23903 | [PDF](https://arxiv.org/pdf/2605.23903v1)

**作者:** Zizun Li `[一作]` (University of Science and Technology of China), Tong He `[通讯]` (Shanghai AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为Geo-Align的强化学习框架，用于相机控制的视频重拍任务，旨在根据给定的条件视频和目标相机轨迹合成新视角视频。

**💡 创新点**

创新点在于引入了基于强化学习的框架，优化了相机运动的物理对齐和视觉质量，同时设计了一种混合数据采样策略，结合真实视频和合成数据的相机轨迹，减少了对稀缺的时间同步多视角视频数据的依赖。

**🔧 技术方法**

使用了强化学习技术，特别是群体相对策略优化（GRPO）方法来优化生成模型，并引入了可验证的几何奖励机制来评估生成视频的相机轨迹。

**📊 数据集**

使用了CityWalk数据集作为条件视频，OmniWorld游戏数据集作为目标相机轨迹，并通过MapAnything模型提取相机姿态。

**📈 对比分析**

与现有的监督学习基线（如ReCamMaster和ReDirector）进行比较，Geo-Align在相机轨迹控制精度和视觉保真度上表现出色，定量评估显示在多个指标上均有显著提升。

**⚠️ 局限性**

限制在于模型的训练仍然依赖于合成数据的相机轨迹，尽管通过混合数据策略减少了对配对多视角视频数据的依赖，但在某些情况下可能仍会受到合成数据质量的影响。

---

## 496. ETCHR: Editing To Clarify and Harness Reasoning

**arXiv ID:** 2605.23897 | [PDF](https://arxiv.org/pdf/2605.23897v1)

**作者:** Beichen Zhang `[一作]` (Chinese University of Hong Kong), Dahua Lin `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一个面向问题的图像编辑器，用于“思考与图像”推理任务，采用两阶段训练与验证流程；

**💡 创新点**

创新点在于将编辑器与理解模型解耦，利用专用图像编辑器产生中间视觉证据，并在推理过程中加入验证步骤，同时采用两种奖励（编辑正确性与指导效果）进行强化学习；

**🔧 技术方法**

使用的技术包括FLUX.2-klein-base-9B扩散编辑器、LoRA细调、Pref‑GRPO强化学习、VLM‑as‑Judge评判器、任务级元提示、编辑‑验证‑推理（Edit‑Verify‑Reason）流程；

**📊 数据集**

训练与评估所用数据集涵盖V*Bench、HRBench、ChartQA、CharXiv、迷宫与Frozen Lake、COCO拼图、DL3DV-10K、ViewSpatial等多种视觉推理任务；

**📈 对比分析**

在工具式、统一模型以及闭源编辑器（Nano Banana 2）等基线上进行对比，实验显示在所有任务族上平均提升约4–6%的Pass@1分数，显著优于传统工具与统一模型方法；

**⚠️ 局限性**

局限性包括RL采样粒度有限导致对结构化编辑探索不足、下游理解模型的能力上限限制了最终性能、图像编辑耗时相对较高，以及在某些低基线任务上验证机制可能导致编辑被错误丢弃。

---

## 497. GenRecon: Bridging Generative Priors for Multi-View 3D Scene Reconstruction

**arXiv ID:** 2605.23888 | [PDF](https://arxiv.org/pdf/2605.23888v1)

**作者:** Katharina Schmid `[一作]` (Technical University of Munich), Matthias Nießner `[通讯]` (Technical University of Munich)

**通讯引用:** 23280 | [OpenAlex ID](https://openalex.org/A5088583491)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用多视RGB图像，结合Trellis.2生成模型和投影式3D conditioning，生成完整、可编辑的物理基础渲染（PBR）网格场景。

**💡 创新点**

将单物体级生成器扩展到场景级别，提出空间对齐的投影条件化路径，使多视角信息以位置感知且排列不变的方式注入生成模型，提升几何完整性与视觉真实性。

**🔧 技术方法**

投影基3D conditioning、LoRA参数高效微调、Trellis.2生成模型、COLMAP结构光束法、MultiDiffusion式联合生成以及GAN/流匹配去噪。

**📊 数据集**

训练数据为合成室内场景SAGE-10k和3D-FRONT，测试数据包含3D-FRONT和真实ScanNet++场景。

**📈 对比分析**

相较于2DGS、MonoSDF、DA3、FineRecon、Murre等基线，在ScanNet++和3D-FRONT上实现了更低的Chamfer距离、更高的F-score和法向一致性，并获得更高的完成度与视觉对齐。

**⚠️ 局限性**

对非漫反射材质（玻璃、镜面）处理不佳，场景高度限制在约5m，偶尔因强生成先验导致轻微的结构幻觉。

---

## 498. Leveraging Foundation Models for Causal Generative Modeling

**arXiv ID:** 2605.23861 | [PDF](https://arxiv.org/pdf/2605.23861v1)

**作者:** Aneesh Komanduri `[一作]` (University of Arkansas), Xintao Wu `[通讯]` (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于预训练基础模型的因果生成框架FM-CGM，完成概念提取、概念干预与对抗图像生成三步，并提出Causal Semantic Guidance (CSG) 机制实现零射程因果发现、干预与对抗生成。

**💡 创新点**

统一使用视觉语言模型与文本到图像扩散模型构建模块化因果生成框架；通过交叉注意力属性图实现概念干预与其后继传播；首次将因果结构模型与扩散模型在推理时协同，引导最小化、因果一致的对抗生成。

**🔧 技术方法**

使用大型视觉语言模型 Qwen3-VL 进行概念提取与操作；Stable Diffusion XL 负责对抗图像生成；结合因果结构模型、扩散概率模型、交叉注意力属性图、LEDITS++ 以及 DPM‑Solver++ 进行逆向推理与实时指导。

**📊 数据集**

在 CelebA‑HQ 与 MS‑COCO 两大公开数据集上进行定性与定量评估；同时使用 Stable Diffusion XL 生成的样例进行展示。

**📈 对比分析**

与 DDIM/DDPM 逆向生成、传统扩散引导等基线对比；采用 VLM‑Eff. 与 LPIPS 两项指标评估效果；实验显示 CSG 在对抗最小性与因果一致性上优于基线，VLM‑Eff. 更高、LPIPS 更低。

**⚠️ 局限性**

受限于预训练模型的推理能力，因果图的准确性受模型知识限制；对复杂多层因果结构的可解释性与生成质量仍有限；对干预范围与细粒度控制的进一步提升仍需研究。

---

## 499. Point Tracking Improves World Action Models

**arXiv ID:** 2605.23856 | [PDF](https://arxiv.org/pdf/2605.23856v1)

**作者:** Jiarui Guan `[一作]` (Aalto University), Juho Kannala `[通讯]` (Aalto University)

**通讯引用:** 8388 | [OpenAlex ID](https://openalex.org/A5057931031)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的联合像素和轨迹世界动作模型（JOPAT），用于机器人政策学习，能够同时预测潜在视觉观察、2D点轨迹和动作。

**💡 创新点**

JOPAT通过将运动表示与视觉表示结合，提供了更强的鲁棒性，尤其是在遮挡和长时间依赖的情况下，显著提高了机器人操作的成功率。

**🔧 技术方法**

使用了一种去噪扩散变换器（denoising diffusion transformer）来联合处理视觉潜在、点轨迹和机器人动作。

**📊 数据集**

在LIBERO和真实世界的LeRobot任务上进行了评估，LIBERO任务包含40个任务，JOPAT在这些任务上取得了97.8%的平均成功率。

**📈 对比分析**

与基于像素的基线方法相比，JOPAT在长时间任务（涉及遮挡、物体交互和屏幕外运动）上表现出最大的性能提升，成功率显著高于其他方法。

**⚠️ 局限性**

局限性包括网格点轨迹的稀疏性可能忽略细微变形，依赖现成的跟踪器可能限制性能，以及当前架构在动态场景中的适用性有限。

---

