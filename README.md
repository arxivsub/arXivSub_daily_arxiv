# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-17 | 今日论文总数: 535

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. RENEW: Towards Learning World Models and Repairing Model Exploitation from Preferences

**arXiv ID:** 2607.14180 | [PDF](https://arxiv.org/pdf/2607.14180v1)

**作者:** Logan Mondal Bhamidipaty `[一作]` (University of Edinburgh), Subramanian Ramamoorthy `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了从人类偏好学习世界模型动力学的框架 DLHF，并设计了利用模型不确定性主动请求偏好的算法 RENEW，用来修复离线强化学习中因模型误差导致的利用问题。

**💡 创新点**

创新点在于：①将人类偏好直接用于监督动力学模型，而非传统的奖励函数或策略对齐；②结合贝叶斯不确定性（集成差异）实现主动偏好查询，显著提升样本效率并避免灾难性遗忘。

**🔧 技术方法**

使用技术包括：Bradley‑Terry 交叉熵偏好损失；对数似然作为“奖励”替代；潜在动力学模型（编码器+隐状态动力学）；集成（ensemble）不确定性估计；RENEW 的循环主动采样与微调。

**📊 数据集**

实验数据集：Jumanji 小型网格世界（Maze、Sliding Tile、Sokoban）以及经典控制任务；所有偏好均由合成 oracle 产生，用以验证方法可行性。

**📈 对比分析**

比较方法：将 naive DLHF 与 RENEW 在相同偏好预算下进行对比；结果显示 RENEW 在 Sliding Tile 3×3 中将最终误差减半，在 Sokoban 中避免灾难性遗忘；在 Maze 10×10 上，RENEW 的预测准确率从约93% 提升至约97%，并显著降低模型不确定性。

**⚠️ 局限性**

局限性：实验仅使用合成 oracle 生成偏好，未涉及真实人类标注；环境规模有限，主要为小型离散网格和低维连续控制；缺乏对更高维观测空间的验证及对模型不可利用性的理论保证。

---

## 2. Introspection Fine-Tuning (IFT): Training Small LLMs to Introspect

**arXiv ID:** 2607.14111 | [PDF](https://arxiv.org/pdf/2607.14111v1)

**作者:** Ely Hahami `[一作]` (Harvard), Lavik Jain `[通讯]` (Harvard)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究小型语言模型能否通过注入激活向量检测并报告自身内部状态的变化，并提出相应的无偏评估方法。

**💡 创新点**

发现传统二元检测在小模型中受全局肯定偏差影响，提出句子定位和强度比较两种评估；并提出 Introspection Fine‑Tuning（IFT），显著提升1B模型的自我检测能力。

**🔧 技术方法**

使用激活驱动（在残差流中插入概念向量），构造针对模型的提示，利用 logit 对比进行检测；IFT 则是基于句子定位标签的监督微调。

**📊 数据集**

使用两套概念向量数据集：500 个具体名词（简单集）和 500 个抽象概念（复杂集，每个含 20 正负句）；以及 100 句文本库用于生成测试样本。

**📈 对比分析**

在 Llama（1B/3B/8B）和 Gemma（2B/4B/26B）六个模型上进行句子定位与强度比较实验。1B 模型原始准确率低于随机，2–3B 起显著提升；IFT 后 1B 模型从约 10% 提升至 60%，3B、8B 也有提升，且能零样本迁移到强度比较，通用能力保持不变。

**⚠️ 局限性**

主要局限包括：评估仅在人工控制的激活干扰场景，未检验在自然推理错误等真实情形下的迁移效果；Sem+Rsn 推理监督的效果不稳定；未在更大规模模型上验证 IFT 的进一步提升。

---

## 3. QFireNet: A Quantum-Enhanced U-Net for Wildfire Segmentation from Sentinel-2 Imagery

**arXiv ID:** 2607.14160 | [PDF](https://arxiv.org/pdf/2607.14160v1)

**作者:** Jaiman Munshi `[一作]` (University of Maryland), Franz Klein `[通讯]` (University of Maryland)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5102977560)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对 Sentinel‑2 卫星影像进行野火像素分割，提出在 U‑Net 的瓶颈层嵌入量子变分电路（QuFeX 与 QB‑Net），并在同一基准上实验多种改进技术。

**💡 创新点**

创新点在于将量子变分电路作为 U‑Net 瓶颈的低维特征变换器，比较两种不同 ansatz 的性能，并结合 Feature Pyramid Network、数据混合、随机划分等手段揭示了训练集与测试集间的显著域偏移。

**🔧 技术方法**

使用的技术包括 U‑Net 语义分割网络、变分量子电路（QuFeX、QB‑Net）、Feature Pyramid Network、MixUp/CutMix 数据增强、加权交叉熵、随机训练/验证/测试拆分、以及量子电路的 PennyLane 仿真器。

**📊 数据集**

主要数据集为 Sen2Fire（Sentinel‑2 影像 + Aerosol 指数，训练/验证/测试按场景划分），以及在 California Burned Areas（CaBuAr）数据集上的跨数据集迁移实验。

**📈 对比分析**

在 Sen2Fire 上与经典基线对比，经典 U‑Net 的 Fire F1 为 28.71；QuFeX 8q/1L 达 30.79；QB‑Net 4q/2L 达 31.18；采用 FPN 在无 Aerosol 模式下得到 31.13；随机拆分将经典 FPN 提升至 39.76，量子模型在相同拆分下尚未测试。

**⚠️ 局限性**

局限性包括：实验仅在模拟器上完成，未验证实际量子硬件；使用单一随机种子，缺乏多种子统计；官方场景划分导致显著域漂移，限制了性能提升；量子模型仅在 25 轮训练下收敛，未探索更长训练；未对量子模型与 FPN 结合、跨数据集量子训练等更广阔场景进行评估。

---

## 4. T5-CSBoost: Adversarial Perturbation Resistant LLM Fingerprinting

**arXiv ID:** 2607.14113 | [PDF](https://arxiv.org/pdf/2607.14113v1)

**作者:** Gayan K. Kulatilleke `[一作]` (University of Queensland), Marius Portmann `[通讯]` (University of Queensland)

**通讯引用:** 5058 | [OpenAlex ID](https://openalex.org/A5078468070)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于T5-Sentinel框架的对抗鲁棒LLM指纹检测模型OURMODEL，通过在解码器嵌入上加入margin‑based triplet loss实现风格表示的对抗鲁棒性。

**💡 创新点**

创新点在于：①仅在原有next-token预测任务上添加对比学习正则化，无需改造模型架构或进行对抗训练；②使用批量margin‑triplet loss直接约束同源样本聚集、异源样本分离，从而获得紧凑且可区分的风格嵌入；③理论分析结合Bhattacharyya误差上界和互信息最大化，解释对比学习提升鲁棒性的机制。

**🔧 技术方法**

技术包括T5-small encoder‑decoder、作者token化、交叉熵+margin‑triplet损失的联合训练、批量对比学习、t‑SNE/PCA嵌入可视化、Integrated Gradients解释、以及对词/字符级别对抗扰动的评估。

**📊 数据集**

使用OpenLLMText、HC3（Human‑ChatGPT‑Compare）以及MAGE/Deepfake benchmark中的多源文本（Human、ChatGPT、GPT‑2、PaLM、LLaMA等）进行训练与测试。

**📈 对比分析**

与T5‑Sentinel、DetectGPT、PRDetect、DeTeCtive等基线对比，OURMODEL在二分类（AUC 0.974、ACC 0.964、F1 0.912）和多分类（ACC 0.95、Macro‑F1 0.95）上均实现SOTA；在词/字符扰动（最高90%）下仍保持>0.99的准确率；在未见模型/域、以及极端改写任务（任务8）上平均Recall提升约15%。

**⚠️ 局限性**

局限性包括：①对比学习的超参数（margin、lambda）需手动调优；②在极其多样化的人类写作风格下，单一人类类标签的广泛性可能导致误判；③对完全重写或高程度语义重组的文本鲁棒性尚未系统验证；④在极少样本或实时部署场景下的计算开销尚待评估。

---

## 5. UzWordnet and Generative AI for Learning Uzbek by Game Playing

**arXiv ID:** 2607.14104 | [PDF](https://arxiv.org/pdf/2607.14104v1)

**作者:** Alessandro Agostini `[一作]` (Inha University), Mirkamol Mirkamilov `[通讯]` (NUIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

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

## 6. Adaptive Control of Motor-Position-Controlled Flexible Joint Robots with Uncertain Joint Stiffness

**arXiv ID:** 2607.14177 | [PDF](https://arxiv.org/pdf/2607.14177v1)

**作者:** Annika Kirner `[一作]` (Technische Universitaet Wien), Christian Ott `[通讯]` (Technische Universitaet Wien)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种针对电机位置控制的柔性关节机器人在关节刚度未知且非线性时的自适应控制方法。

**💡 创新点**

创新点在于利用隐式控制律与控制输入相关的回归矩阵，在线更新非线性弹簧力-位移关系，并对电机位置控制误差进行鲁棒性分析。

**🔧 技术方法**

采用 Lyapunov 方法设计自适应控制器和参数更新律，使用三次多项式回归来近似弹簧力-位移曲线，并分析系统对电机位置误差的有限增益稳定性。

**📊 数据集**

实验数据来自一台可变刚度执行器（VSA），在高低两种刚度设定下测量其位移和力的关系；没有公开的数据集，全部为实验采集。

**📈 对比分析**

与两种基线控制器（分别使用高刚度或低刚度的固定模型）对比；实验显示自适应控制器在几周期后追踪误差显著下降，优于基线控制器。

**⚠️ 局限性**

局限性包括参数估计不一定收敛到真实值，可能受阈值限制；未对非理想电机位置控制器的误差做在线补偿，且实验仅在单关节 VSA 上验证。

---

## 7. "Trust Junk" Leads to Unjustified Support for Highly Discriminatory Predictive Models

**arXiv ID:** 2607.14152 | [PDF](https://arxiv.org/pdf/2607.14152v1)

**作者:** Michael Correll `[一作]` (Northeastern University), Mahsan Nourani `[通讯]` (Northeastern University)

**通讯引用:** 398 | [OpenAlex ID](https://openalex.org/A5055095298)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

我们通过设计三种不同程度的“trust junk”XAI解释，进行在线问卷实验，评估其对用户信任、模型认可度和公平感知的影响。

**💡 创新点**

首次系统量化“trust junk”对偏见模型解释的信任操纵效应，并证明过多无关信息会导致用户对不公平模型产生误导性信任。

**🔧 技术方法**

采用受试者随机分组的在线问卷实验、Likert量表评估、统计检验（ANOVA）以及质性文本编码分析。

**📊 数据集**

使用Law School Admissions Bar Passage数据集（约2万名考生的背景与考试成绩），构造仅基于种族与性别的偏见模型。

**📈 对比分析**

将三组解释（无junk、轻量junk、重量junk）在用户同意率、信任度、过度依赖率等指标上进行方差分析；结果显示高junk组的信任度、同意率显著高于低junk组，说明信息量与信任正相关。

**⚠️ 局限性**

研究仅聚焦单一偏见模型与特定解释形式，未探究更复杂或真实场景的XAI；样本来自Prolific，可能缺乏多样性；缺少长期使用与教育干预等因素的考察。

---

## 8. Semantic Register Compression in Multi-Agent LLM Cascades

**arXiv ID:** 2607.14119 | [PDF](https://arxiv.org/pdf/2607.14119v1)

**作者:** Manuele Tele Junior Fernandez `[一作]` `[通讯]` (Independent Researcher), Manuele Tele Junior Fernandez (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多代理LLM流水线中的语义寄存器压缩现象，量化中间评估器对标签区分度的影响；

**💡 创新点**

首次提出“语义寄存器压缩”概念及其几何度量——跨阶段的标签间分离度(inter‑label separation)，并通过五种评估器变体证明压缩源自语义转换而非仅是流水线结构；

**🔧 技术方法**

使用LangGraph构建Collector–Evaluator–Decider流水线，利用Sentence‑Transformer得到嵌入向量，计算标签质心距离并进行回归分析，探究提示词特征与压缩程度的关系；

**📊 数据集**

在政治事实核查（LIAR）、情感分析（SST‑5）和医学分流（Triagegeist）三个数据集上进行实验；

**📈 对比分析**

与单代理基线对比显示多代理流水线在评估阶段产生约41.7%（事实核查）、27.2%（情感）和20%（分流）的标签间压缩；通过评估器变体验证压缩的因果关系；提示词中增加操作约束可显著降低压缩；

**⚠️ 局限性**

局限性包括样本量较小（多数实验仅50例），评估器变体仅在单一模型上测试，Triagegeist为合成数据，嵌入距离仅为语义可区分度的近似代理，未考虑风格/长度等表面因素的影响。

---

## 9. Human AI Construction of Bayesian Networks for Operational Decision Support -- A Virtual Survey Approach

**arXiv ID:** 2607.14141 | [PDF](https://arxiv.org/pdf/2607.14141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 10. Interpretable Language Model for Closed-Loop Type 1 Diabetes Control

**arXiv ID:** 2607.14126 | [PDF](https://arxiv.org/pdf/2607.14126v1)

**作者:** Maya Sarkar `[一作]` `[通讯]` (Visaze LLC), Maya Sarkar (Visaze LLC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文设计并验证了一个闭环胰岛素泵控制系统，利用强化学习专家的行为通过LLM蒸馏为可解释的语义决策，并通过确定性安全层确保安全执行。

**💡 创新点**

创新点在于将RL专家行为蒸馏到大型语言模型，实现自然语言解释的同时保持或提升控制性能，并通过确定性安全层防止模型幻觉导致的危害。

**🔧 技术方法**

采用的技术包括PPO+G2P2C强化学习、文本化引擎将数值状态转换为JSON提示、LoRA参数高效微调LLM（LLaMA 3.1 8B/ Qwen3 8B）、以及确定性安全覆盖层。

**📊 数据集**

使用的数据集为FDA批准的UVA/Padova T1D模拟器，包含10名成人和10名青少年虚拟患者，分别在100次随机24小时场景中进行测试。

**📈 对比分析**

实验对比了BBHE、PPO专家、以及两种LLM蒸馏模型；LLM-T1D在成人组达73.5% TIR、1.6% 失败率（低于PPO的2.79%），在青少年组亦表现出类似的提升，显示性能与安全性均优于传统方法。

**⚠️ 局限性**

局限性在于仅基于模拟数据，未涵盖真实CGM噪声、运动、疾病等实际变异；LLM输出仍需安全层拦截，系统需进一步在真实患者数据和临床环境中验证。

---

## 11. Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning

**arXiv ID:** 2607.14171 | [PDF](https://arxiv.org/pdf/2607.14171v1)

**作者:** Bowei He `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xue Liu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了Branching Policy Optimization（BPO）算法，利用沙盒可快照、可恢复特性，在高熵决策点进行分支，构建树形rollout并使用子兄弟baseline来估计优势，显著降低梯度方差；

**💡 创新点**

核心创新在于：① 把沙盒的可快照恢复能力引入训练，构造单棵树的rollout；② 采用子兄弟baseline优势估计并理论证明无偏且方差下降；③ 通过熵驱动的自适应分支调度最大化分支效益；

**🔧 技术方法**

使用PPO的clip策略梯度、快照/恢复操作、熵驱动分支选择、子兄弟baseline优势估计、蒙特卡洛回报、并行多分支并行化以及方差分析等技术；

**📊 数据集**

在WebShop、ALFWorld、SWE-bench Verified三大沙盒环境上进行实验，使用Qwen2.5-7B和Llama-3.1-8B作为LLM骨干；

**📈 对比分析**

与SFT、PPO、RLOO、GRPO、VinePPO等基线在相同计算预算下对比，BPO在WebShop、ALFWorld、SWE-bench的任务成功率分别提升3.6–6.1点；梯度步骤数减少38%，梯度范数方差约为GRPO的0.5倍；

**⚠️ 局限性**

局限性包括：依赖沙盒快照/恢复成本；分支数量受最小间隔限制，可能未能充分利用所有高熵点；仅适用于可 deterministic 沙盒；未结合学习过程奖励模型或递归分支等更复杂策略。

---

## 12. MIDiff: Tackling Sparsity and Imbalance in Mobile Usage Generation via Multivariate-Imaging Diffusion

**arXiv ID:** 2607.14249 | [PDF](https://arxiv.org/pdf/2607.14249v1)

**作者:** Yilai Liu `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**通讯引用:** 8762 | [OpenAlex ID](https://openalex.org/A5068782412)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 MIDiff 的基于扩散模型的移动使用轨迹生成框架，能够合成高保真、稀疏且多变量的用户级移动数据。

**💡 创新点**

创新点在于：1) 设计 Cross‑Gramian Angular Sum Field (C‑GASF) 将稀疏多变量序列映射为结构化图像，显著提升稀疏性、异构性与长尾分布的建模；2) 在 U‑Net 中引入 Triple Attention，分别捕获时间、变量与语义维度的关系，从而更好地重建行为细节。

**🔧 技术方法**

核心技术包括：扩散生成模型、C‑GASF 图像表示、Triple Attention 机制、残差 U‑Net 结构、基于马尔可夫前向噪声/后向去噪的学习过程。

**📊 数据集**

使用的是一周内收集的真实移动使用数据集（App Usage Dataset），包含 1000 名用户、192 步长、20 个应用类别、17 个地点簇、约 2000 个应用标签。

**📈 对比分析**

与 9 类主流时序生成方法（GAN、VAE、传统扩散、图像转换等）进行对比，评估指标包括 VDS、FDDS、DA、预测分数、t‑SNE/UMAP 维度分布、以及下游多变量预测效果。MIDiff 在 VDS、FDDS、DA 上显著优于基线，且在下游预测任务中能提供最可靠的数据增强效果。

**⚠️ 局限性**

局限性：1) 仅为无条件生成，未支持用户特定偏好或条件控制；2) 依赖预先聚类的地点信息和应用标签，迁移到不同场景需要重新构造 C‑GASF；3) 训练成本高，模型参数约 148M，推理延迟约 66 ms。

---

## 13. Quantum Compositional NLP for Arabic: Grammar, Morphology, and Word Sense in Circuit Topology

**arXiv ID:** 2607.14100 | [PDF](https://arxiv.org/pdf/2607.14100v1)

**作者:** Wajahath Mohammed `[一作]` `[通讯]`, Wajahath Mohammed

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了第一个基于前语法的量子组合NLP系统，对阿拉伯语句子生成量子电路并完成词序、时态和词义消歧实验。

**💡 创新点**

证明无参数量子电路的语法拓扑能以零方差区分词序，并通过单层纠缠提升15%；首次将根-模式形态学与量子张量产品对齐，提出词汇控制的词义消歧数据集。

**🔧 技术方法**

使用 DisCoCat 预组语法与 IQP ansatz 量子电路，结合 lambeq、CAMeL Tools、Stanza 进行形态与句法分析，量子特征映射、SVM 及 SPSA 优化。

**📊 数据集**

自构建 1,140 句 MSA 阿拉伯语语料，包含 WordOrder、WordOrderMatched、Morphology、TenseBinary、WordSenseDisambiguation_v2 等子集，约 200 句的词义消歧词汇控制数据集。

**📈 对比分析**

与 AraVec bag‑of‑words、AraBERT（冻结 CLS 与 fine‑tuned）以及 topology‑only 基线在 5‑fold × 3 交叉验证下比较；在词序匹配对实验中 L0 量子电路 50%（理论），L1 64.9%（+15%），AraVec 12.8%，AraBERT fine‑tuned 100%；时态实验中 AraVec 87%，量子 56%；词义消歧中 AraVec 50‑94%，量子 QFM 约 40‑60%，SPSA 约 50‑70%。

**⚠️ 局限性**

局限性包括数据集规模有限、SPSA 稳定性差、仅使用模拟而非硬件、单一注释者构建、部分结果缺乏置信区间、AraVec 覆盖不足等。

---

## 14. Quantize with Confidence? An Empirical Study of Quantization for Code Generation

**arXiv ID:** 2607.14181 | [PDF](https://arxiv.org/pdf/2607.14181v1)

**作者:** Saima Afrin `[一作]` (William & Mary), Antonio Mastropaolo `[通讯]` (William & Mary)

**通讯引用:** 720 | [OpenAlex ID](https://openalex.org/A5069505458)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了六种主流4-bit权重仅量化技术对两款7B级代码生成模型在功能正确性、代码质量以及对提示复杂度鲁棒性方面的影响。

**💡 创新点**

创新点在于首次将六种量化方法进行横向比较，并引入Shannon熵与token长度作为提示复杂度度量，揭示不同模型对复杂度的敏感性差异。

**🔧 技术方法**

使用GPTQ、AWQ、QuIP#、AQLM、BitsAndBytes、GGUF等量化技术，对Qwen2.5-Coder-7B和CodeLlama-7B进行4-bit量化；评估指标包括pass@1、SonarCloud静态分析、Wilcoxon/McNemar检验等。

**📊 数据集**

实验数据来源于McEval、CoderEval（Python/Java）和BigCodeBench（Python）三大多语言代码生成基准。

**📈 对比分析**

通过与FP16基线在pass@1和代码质量指标的对比，发现AQLM往往匹配或超越基线，QuIP#显著降低正确率；量化后显存占用下降约70%，不同技术在准确率、吞吐量和CPU友好性上表现各异。

**⚠️ 局限性**

限制包括仅测试7B规模、4-bit精度、两种语言、特定量化方案；未考虑更低位宽或其他模型；使用单一静态分析工具，可能遗漏其他质量维度；提示复杂度仅用token长度和熵衡量，未覆盖算法复杂度等因素。

---

## 15. ITGPT: A Transformer Based Architecture for the Generation of Dance Dance Revolution and In the Groove Charts

**arXiv ID:** 2607.14148 | [PDF](https://arxiv.org/pdf/2607.14148v1)

**作者:** Miguel O'Malley `[一作]` `[通讯]` (Max Planck Institute for Mathematics in the Sciences), Miguel O'Malley (Max Planck Institute for Mathematics in the Sciences)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 ITGPT，一种基于 Transformer 的 DDR/ITG 节奏游戏谱面自动生成模型

**💡 创新点**

创新点在于层次化 Transformer 架构、诊断网络确保 BPM 与难度一致、一次性音频处理与自回归步骤选择的混合处理方式，显著提升低难度谱面质量与生成速度

**🔧 技术方法**

使用 Transformer 编码器、残差向量量化、FiLM 条件化、Nucleus 采样、BPM 检测 (ArrowVortex)、卷积特征提取等技术

**📊 数据集**

采用扩展的 Fraxtil 数据集（253 首歌、952 张谱面，约 584k 步）

**📈 对比分析**

与 DDC、DDCL、GOCT 等基线进行 F1、PR‑AUC、准确率等指标对比，ITGPT 在大多数指标上超过 DDCL，尤其在低难度谱面上提升显著，并且生成速度比 DDCL 提升约 7 倍

**⚠️ 局限性**

局限性包括数据集规模仍有限，无法完全覆盖高难度谱面；在 held‑note 预测上仍略逊于 DDCL；未实现统一的步骤放置/选择一体化管线

---

## 16. The Cost and Network Limits of Space-Based AI Compute

**arXiv ID:** 2607.14172 | [PDF](https://arxiv.org/pdf/2607.14172v1)

**作者:** Kees van Berkel `[一作]` `[通讯]` (Technical University Eindhoven), Kees van Berkel (Technical University Eindhoven)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估在低地球轨道部署 1 GW AI 数据中心与陆地设施在成本与性能上的可行性，聚焦于训练和推理工作负载的比较。

**💡 创新点**

首次将轨道光学互连网络（2D/3D 托罗斯）与传统 Clos 网络进行对比，使用 bisection bandwidth 与 bisection intensity 的屋顶线（roofline）方法量化训练效率，并提出“Clos 卫星星座”概念。

**🔧 技术方法**

采用分析模型（网络延迟、带宽、功耗）、屋顶线模型、All‑Reduce 计算量分解，以及对 Starlink 2/V3 规格的功率、散热和辐射容限的技术参数估算。

**📊 数据集**

主要以公开的硬件参数与 LLM 训练规模（1 T 参数、1 M 令牌/step）为输入；未使用具体数据集，仅基于典型 LLM 训练设置。

**📈 对比分析**

比较方法：通过 bisection bandwidth 计算网络瓶颈，结合 bisection intensity 评估训练算力利用率，得到陆地 8000 锦盒架（Clos）与轨道 8000 卫星（2D/3D 托罗斯）的性能对比；结果显示轨道训练的模型算力利用率从 40–70% 降低至 ~2–5%，推理可在单卫星上实现，但训练成本高出 100–1000 倍。

**⚠️ 局限性**

限制：模型假设使用理想的平面/立方星座、100 Gbps 互连；忽略卫星形成、控制、失效恢复、辐射损伤等实际挑战；光学链路带宽与多路复用尚未实现；高延迟和低 bisection 宽度导致训练效率不可接受。

---

## 17. Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits of Text

**arXiv ID:** 2607.14103 | [PDF](https://arxiv.org/pdf/2607.14103v1)

**作者:** Markus Wenzel `[一作]` (Constructor University), Markus Wenzel `[通讯]` (Constructor University)

**通讯引用:** 1019 | [OpenAlex ID](https://openalex.org/A5059136948)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fede83ac-7505-405f-ab37-e7284695c47f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并系统评估LLM代理间的三种通信通道（dense latent、SAE‑sparse、text），量化信息损失、跨架构对齐质量和在多种任务上的表现。

**💡 创新点**

构建了首个面向隐空间通信的评估框架；通过Sparse Autoencoder特征生存分析揭示文本序列化导致的特征替换；验证Llama与Mistral在语义几何上可通过Procrustes对齐；首次将latent与text通道在概念识别、属性歧义和跨语言任务中进行对比。

**🔧 技术方法**

使用Sparse Autoencoder、线性probe、Orthogonal Procrustes、CCA、特征生存率、Jaccard、cosine相似度、多语言提示和多选评测等技术。

**📊 数据集**

数据集包括84个手工挑选的概念对、163概念的扩展数据、112个属性/意义歧义任务、107个跨语言任务、74个跨语言概念识别任务，以及多语言提示集。

**📈 对比分析**

通过probe准确率、压缩率、文本对比、跨架构检索精度（top‑1 92%）以及任务级多选评测进行比较。SAE‑sparse在probe准确率上达到99.4%（压缩28×），文本仅80.4%；在任务层面，文本通道平均优于latent 3–10个百分点；跨语言概念识别中latent略优。

**⚠️ 局限性**

局限性包括：所有评测任务均可用文本表达，缺乏能揭示latent优势的高复杂度任务；使用的模型尺寸仅为8B，可能未能充分体现更大模型的隐空间优势；对齐仅采用线性Procrustes，未探索更强的非线性或低秩对齐；跨模态（如纯视觉模型）测试尚未展开。

---

## 18. Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection

**arXiv ID:** 2607.14236 | [PDF](https://arxiv.org/pdf/2607.14236v1)

**作者:** Yi Wang `[一作]` (Shanghai Jiao Tong University), Cewu Lu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在预训练的视觉‑语言‑动作（VLA）策略上，LIFT 引入后训练阶段的力反馈，构建了迟延的反应式力注入路径，使机器人能够在接触状态下实时更新动作；

**💡 创新点**

① 通过移位因果注意力与零初始化跨注意力实现力的迟延反应式注入；② 通过复制动作专家权重、保持初始化等价与零初始化跨注意力，保证原始 VLA 先验不被破坏；③ 结合在线 DAgger 收集力反馈纠正数据，解决力分布漂移问题。

**🔧 技术方法**

预训练 VLA 模型、因果力记忆编码、移位因果注意力、零初始化跨注意力、在线 DAgger 循环、流匹配损失、力掩蔽与均衡采样、云端训练与机器人回传。

**📊 数据集**

Stage‑1 的离线视觉对齐数据集 D_v（手持设备采集的视觉+语言+动作三元组）；Stage‑2 的在线力反馈校正数据集 D_f^(k)（真实机器人配备 6D 力传感器的力+动作校正）。

**📈 对比分析**

与 Vision‑only 在线 DAgger、无力注入版本、无在线 DAgger 版本以及仅离线手持数据训练的 VLA 等进行比较；在毛巾折叠、书插入和汉诺塔环放置三项任务中，LIFT 在更少样本下实现更高峰值，并保持原 VLA 的泛化性能，说明力反馈显著加速学习并提升最终表现。

**⚠️ 局限性**

仍需人工纠正进行在线 DAgger，限制了数据吞吐；仅在单臂机器人上评估，未验证多臂、多传感器和不同末端执行器的适用性；力分布完全未知时的鲁棒性仍待提升。

---

## 19. Untrusted Authors, Trusted Answers: A Calculus of Fidelity-Graded Translations

**arXiv ID:** 2607.14137 | [PDF](https://arxiv.org/pdf/2607.14137v1)

**作者:** Christoph Kirsch `[一作]` (University of Salzburg), Christoph Kirsch `[通讯]` (University of Salzburg)

**通讯引用:** 3398 | [OpenAlex ID](https://openalex.org/A5102985234)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了一个基于可验证译者图的“忠实度等级”演算，用以在多语言、多路径的翻译网络中以可组合方式量化和证明翻译正确性。

**💡 创新点**

创新点在于：①用可组合的可验证翻译对（可满足的交换方程）定义忠实度等级；②通过弱链原则、运行时重确立与分支交叉验证，将弱边转为强路；③提出存在/无存在答案的不对称架构原则；④将核心理论机械化在 Lean4 中，并以 LLM 自动生成的 13 种语言对实现。

**🔧 技术方法**

使用技术包括 Lean4 证明助手、Claude Opus/Fable LLM 自动生成代码、BTOR2 与 SMT‑LIB 求解器、RISC‑V、AArch64、C、Wasm、eBPF、EVM、Python、Sail 等语言的翻译器与解释器，以及 hurdy‑gurdy 平台框架。

**📊 数据集**

数据集主要是由 12 种语言的语法构造枚举组成的覆盖集，以及通过两条独立的 RISC‑V→BTOR2、AArch64→BTOR2 路径得到的样本程序。

**📈 对比分析**

比较方法：对每条路径进行交叉验证、分支一致性检查、机器生成的基准对照以及终端证据（证书）再验证；实验中平台实现了全路径覆盖、分支一致率超过 95%，并通过证明检验完整证实可达性与不可达性。

**⚠️ 局限性**

局限性包括：LLM 生成代码缺乏人工语义审查，整体信任仍依赖架构交叉检查；覆盖率与实际语言复杂度之间仍有差距；性能评估主要在实验环境下完成，未覆盖大规模工业级程序；且不对称架构对无证据答案的处理仍需进一步实践验证。

---

## 20. CARPRT: Class-Aware Zero-Shot Prompt Reweighting for Black-Box Vision-Language Models

**arXiv ID:** 2607.14125 | [PDF](https://arxiv.org/pdf/2607.14125v1)

**作者:** Ruijiang Dong `[一作]` (University of Melbourne), Masashi Sugiyama `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**通讯引用:** 22728 | [OpenAlex ID](https://openalex.org/A5072744508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在零样本图像分类中利用类感知的prompt重加权提升性能的方案

**💡 创新点**

提出了无训练的类感知零样本prompt重加权方法CARPRT，能够仅通过未标记图像自适应地分配prompt权重

**🔧 技术方法**

基于预训练视觉语言模型（CLIP、DeCLIP）计算图像‑文本相似度，使用伪标签统计并通过温度软max归一化得到类特定权重

**📊 数据集**

在十个细粒度分类基准、ImageNet及其变体、Caltech101等公开数据集上进行评测

**📈 对比分析**

与MPE、WPE、Majority Vote及人工挑选prompt等基线相比，CARPRT在多种模型与数据集上均实现最高准确率，平均提升约1‑3%

**⚠️ 局限性**

受限于初始伪标签质量和prompt池多样性，专业领域提升有限；未充分评估对大规模算力外部资源的依赖

---

## 21. UniSAGE: Unifying Static and Dynamic Attributes with Hyper-Structure

**arXiv ID:** 2607.14102 | [PDF](https://arxiv.org/pdf/2607.14102v1)

**作者:** Taoran Fang `[一作]` (Zhejiang University), Yang Yang `[通讯]` (Zhejiang University)

**通讯引用:** 117843 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 UniSAGE 框架，构建全局属性图，将静态与动态属性统一为节点，并通过正交参数子空间实现静态聚合与动态推理共存，随后利用轻量级超结构机制（SSAgg）增强任务特定的跨属性交互；

**💡 创新点**

创新点包括：① 用统一属性图捕获层次与时间关系；② 引入正交子空间实现静态与动态操作在同一语义空间内独立、无互相干扰；③ 通过 Selective Semantic Aggregation 轻量化模拟超结构，实现可解释且计算高效的交互；

**🔧 技术方法**

技术手段：图神经网络（GCN/GraphSAGE/GAT）、GRU 时序编码器、预训练文本编码器、正交约束、轻量化注意力聚合（SSAgg），并结合下游任务预测器；

**📊 数据集**

使用公开 RelBench 基准（电商、社交、医疗等多域）以及真实金融行为数据集 UserBehavior（191,927 记录，包含静态属性与时间序列行为日志）；

**📈 对比分析**

与传统特征提取（MLP、LightGBM）、静态 GNN（GCN、GraphSAGE、GAT）、动态 GNN（TGN、TGAT、DyGFormer、FreeDyG）、RDL/RelGNN 等进行对比；在 RelBench 所有实体分类任务中平均提升约 9% 以上，部分任务提升近 10%；在 UserBehavior 的异常检测任务中相对最佳基线提升约 10%；

**⚠️ 局限性**

局限性：对节点属性仍依赖文本预编码，可能对非文本属性效果有限；在极大规模图上计算和内存开销相对传统 GNN 较高；对完全无层次或无时间戳的属性关系适配可能不如预期；

---

## 22. MAPS: Modeling Co-Existing Subjective Perspectives and Shared Meaning in Multi-Agent Cognitive Dialogue

**arXiv ID:** 2607.14110 | [PDF](https://arxiv.org/pdf/2607.14110v1)

**作者:** Molood Arman `[一作]`, Clément Bonnafous `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 MAPS（Multi-Agent Perspective Spaces）框架，允许多智能体在对话中保留各自的认知视角并通过共享语义空间实现共同理解。

**💡 创新点**

创新点在于引入领域加权视角适配器、动态 GRU 内存和可解释的词级注意力，使多智能体在保持主观性的同时实现语义收敛。

**🔧 技术方法**

使用 Sentence‑BERT 提取嵌入、MLP 进行视角适配、GRU 动态记忆、FLAN‑T5 大模型生成回应，并结合词级自注意力可视化。

**📊 数据集**

在 EmpatheticDialogues、TopicalChat 与 MultiWOZ 三大对话数据集上进行实验。

**📈 对比分析**

与基线（无 GRU、全浅层模型）对比，MAPS 在语义偏差、回复多样性和相关性指标上均表现更佳，显示出在保持个体差异的同时仍能实现语义一致。

**⚠️ 局限性**

局限性包括手工设定的领域权重、缺乏长期记忆、在多智能体规模增大时评估困难以及主观性度量仍不够精细。

---

## 23. Just Keep Prompting: Evaluating Repetitive Socratic Prompting in VLMs

**arXiv ID:** 2607.14099 | [PDF](https://arxiv.org/pdf/2607.14099v1)

**作者:** Shayda Moezzi `[一作]` (Northeastern University), Sarah Ostadabbas `[通讯]` (Northeastern University)

**通讯引用:** 2388 | [OpenAlex ID](https://openalex.org/A5031787107)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Just Keep Prompting (JKP) 框架，用三种连续对话压力策略评估 VLM 在保持相同视觉输入下的答案稳定性。

**💡 创新点**

创新点在于将多轮对话视为压力试验而非仅仅推理提升，揭示模型在面对持续对话压力时的易变性和自信错误行为。

**🔧 技术方法**

采用多轮问答轨迹记录、答案变化计数、置信度跟踪、token 使用统计等技术进行细粒度行为分析。

**📊 数据集**

使用 STAR 视觉推理基准的 80 条视频-问题对样本，涵盖 Interaction、Sequence、Prediction、Feasibility 四类。

**📈 对比分析**

对比 GPT‑4o、Gemini 2.5 Pro 与 Qwen3‑VL‑30B 三模型，结果显示 Qwen 在最终准确率上最高（75.0%）但易出现自信错误；Gemini 稳定但 token 使用量远高于其他模型；GPT‑4o 反应最敏感但准确率最低（60.4%），说明不同模型在压力下表现出不同的稳健性签名。

**⚠️ 局限性**

局限性包括仅评估固定视觉输入而未覆盖动态视频变化；样本量有限，且只包含 STAR 子集，缺乏在更广泛任务和多样化数据集上的验证。

---

## 24. Open-AoE: An Open Egocentric Manipulation Dataset and Toolchain for Embodied Learning

**arXiv ID:** 2607.14183 | [PDF](https://arxiv.org/pdf/2607.14183v1)

**作者:** Zishuo Li `[一作]` (Ant Group), Zhe Li `[通讯]` (Ant Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个端到端的开源 egocentric 数据基础设施，包含约 2000 小时的智能手机捕获的日常操作视频、完整的数据处理流水线、以及可视化、重建、跨主体转移和模型训练的工具链。

**💡 创新点**

创新点在于：①提供从原始视频到结构化样本（手姿、相机轨迹、动作分段、语言描述）的完整流水线；②实现跨主体、跨设备、跨场景的 400+ 设备、400+ 场景、8000+ 任务大规模多样性；③开放可复用的可视化、4D 重建、机器人重现与训练接口，形成完整的数据生产‑使用闭环。

**🔧 技术方法**

采用的技术包括：轻量化手机端手部检测与录制、云端质量检查与隐私抹除、DROID‑W 相机轨迹估计、HaWoR 手姿重建、EgoInfinity/Do‑as‑I‑Do 4D 物体‑手交互重建、三门质量检验、CLIP 视觉特征评估、Idefics2 视觉‑语言一致性评分，以及对 VLA、WAM、World Model 的训练接口。

**📊 数据集**

主要使用的数据集是 Open‑AoE 本身（约 2000 小时）并与 OpenEgo、EgoDex、EgoXtreme 等现有 egocentric 数据集进行对比评估。

**📈 对比分析**

比较方法：从视觉特征（CLIP 有效秩、kNN 域混合）、语义广度（动作词表、对象、场景）、时间覆盖率、Idefics2 视觉‑语言一致性评分、训练窗口保留率、手姿/相机姿态完整性等多维度指标进行系统对比；结果显示 Open‑AoE 在视觉多样性、语义覆盖、注释完整性和训练样本可用性上均显著优于对手，几乎实现了全模态监督的高密度利用。

**⚠️ 局限性**

局限性包括：①单目重建仍受光照与遮挡影响，手姿/相机轨迹精度有限；②跨机器人转移仍需针对具体硬件的适配；③大规模持续采集依赖社区贡献，数据质量与标注一致性可能随时间波动；④实时部署与大规模训练的计算与隐私管理仍是挑战。

---

## 25. LIGO-PINN: Learned Initialization via Gated Optimization to Alleviate Convergence Failures in Physics Informed Neural Networks

**arXiv ID:** 2607.14233 | [PDF](https://arxiv.org/pdf/2607.14233v1)

**作者:** Nilay Anurag `[一作]` (Stevens Institute of Technology), Nikhil Muralidhar `[通讯]` (Stevens Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于学习初始化的 PINN 训练框架（LeIn-PINN），通过在易任务上进行元学习来为难任务生成更优的网络权重，从而显著减少 PINN 在复杂 PDE 领域的灾难性失效。

**💡 创新点**

创新点在于：① 将权重初始化视为一个可学习的任务，使用元学习方法进行 invariance encoding；② 引入 gated layer‑wise optimization（GLO），逐层解锁梯度更新，兼顾物理损失与数据损失的层级冲突；③ 证明学习初始化可显著降低谱偏差。

**🔧 技术方法**

使用了元学习（MAML‑style）、梯度门控机制、传统 PINN 损失（物理残差 + 边界/初始条件损失）以及深度前馈网络作为基础模型。

**📊 数据集**

在 1D 对流方程、2D Helmholtz 方程和 2D Navier–Stokes 方程三个典型 PDE 数据集上进行实验，并在 3D 未结构化域中验证泛化。

**📈 对比分析**

与六种最先进的 PINN 基线（包括自适应采样、课程学习、动态拉伸等）进行对比，平均提升 91.5%（相对误差降低），在最强基线上提升约 81%；在 3D 领域亦保持显著优势。

**⚠️ 局限性**

局限性包括：① 需要在初始化阶段访问一组相关易任务，限制了单任务直接迁移；② 主要验证在有限数量的 PDE 系统与参数范围，尚缺乏更广泛的理论解析；③ 对大规模工业多物理问题的可扩展性与鲁棒性尚待进一步研究。

---

## 26. Natural-Language to SysMLv2 Translation via Conformance-Driven Iterative Refinement

**arXiv ID:** 2607.14162 | [PDF](https://arxiv.org/pdf/2607.14162v1)

**作者:** Chance LaVoie `[一作]` (Carnegie Mellon University), Levent Burak Kara `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3818 | [OpenAlex ID](https://openalex.org/A5048339797)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种在自然语言到 SysMLv2 的生成过程中嵌入生产级合规检查器的 generate‑check‑repair 循环，实现可靠的模型生成。

**💡 创新点**

创新点在于将工业级合规检查器作为循环终止判据，直接利用确定性诊断驱动修复，而非仅靠语法约束或模板推导。

**🔧 技术方法**

使用 LLM（OpenAI Codex、Anthropic Sonnet、DeepSeek Reasoner、Mistral）与 SysIDE 合规检查器，循环式生成‑检查‑修复；对照使用 ANTLR4 语法解析。

**📊 数据集**

使用 SysMBench 基准的 151 条自然语言需求与对应 SysMLv2 模型，共 604 条 prompt‑model 组合。

**📈 对比分析**

通过对比单发生成与循环生成的合规率评估：单发 51.16% 合规，循环 100% 合规；平均 1.7 次生成即可收敛，T90=2、T95=3、T99=4；所有案例无失败，95% 置信下限约 99.5%。

**⚠️ 局限性**

仅保证语法与工业合规，未涵盖语义或行为正确性；仅基于单一工具 SysIDE，缺乏跨工具验证；评估仅限 SysMBench，未考察更杂或噪声需求；无正式收敛证明，也未系统分析成本‑可靠性权衡。

---

## 27. Multi-Head Latent Control: A Unified Interface for LLM Agent Decision Making

**arXiv ID:** 2607.14277 | [PDF](https://arxiv.org/pdf/2607.14277v1)

**作者:** Amirhosein Ghasemabadi `[一作]` (University of Alberta), Di Niu `[通讯]` (University of Alberta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

为冻结的语言/视觉语言模型添加两层轻量级控制头（Capability Head 与 Resolution Head），在推理时从隐藏状态轨迹预测模型是否足够、是否需要提升到更强模型以及是否需要澄清、调用工具或放弃回答，从而实现自适应的推理控制。

**💡 创新点**

创新点在于：①将模型内部隐藏状态作为决策依据，完全不需要修改模型或额外的外部路由器；②将能力判断与干预决策拆分为两个独立头；③通过在冻结模型上训练轻量化头实现快速迁移和适配。

**🔧 技术方法**

技术包括：冻结基础模型，提取生成过程的隐藏状态轨迹；使用投影层对变长轨迹做压缩；训练两层全连接网络分别输出能力分数和干预分数；通过阈值控制是否升级模型或执行干预。

**📊 数据集**

数据集涵盖多模态与单模态任务：CharXiv、MathVerse、MathVista、ScreenSpot-Pro、SimpleVQA、MMLU-Pro、AndroidWorld、When2Call、TriviaQA，以及在不同模型家族（Qwen3、Qwen3.5、Gemma）上的混合训练集。

**📈 对比分析**

与单模型基线、总是使用大模型基线以及其他路由/工具使用方法相比，Multi‑Head Latent Control 在多模型协作中可将大模型使用量降低 27–90% 同时保留 80–95% 的性能；在工具使用决策上提升 11–12% 的 F1 以及 65% 的误调用率下降；在 TriviaQA 上可将网页检索调用精度提升 2–3% 并显著减少遗漏调用。

**⚠️ 局限性**

局限性包括：需要为每种任务生成隐藏状态标签，训练成本与数据获取相关；对极端新任务或大幅改动的模型架构迁移效果未知；阈值设置依赖经验，可能需手动调优；在某些指标上提升幅度有限。

---

## 28. Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning

**arXiv ID:** 2607.14192 | [PDF](https://arxiv.org/pdf/2607.14192v1)

**作者:** Dingsu Wang `[一作]` (Pinterest), Yijie Dylan Wang `[通讯]` (Pinterest)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在Pinterest大规模推荐系统中，提出了一套统一、模型无关的下游奖励（Downstream Reward）框架，用于优化长期用户参与度与留存；

**💡 创新点**

创新点包括：1）通过离线筛选框架系统性发现可作为代理奖励的会话级信号；2）设计多类型奖励（更深会话参与、负面奖励、用例采纳），并能跨表面共享；3）实现可扩展的DRv2基础设施，将奖励计算迁移到Ray数据加载器，显著加快实验迭代；4）在多面向推荐模型中引入奖励头，并通过HyperOPT手动调权重，保持即时指标不变的同时提升长期指标；

**🔧 技术方法**

主要技术手段为：监督式学习（随机森林筛选、模型训练）、离线特征工程（UP归一化、特征交叉）、奖励建模（折扣累计、负面惩罚、相似度阈值），以及Ray + UDF 的大规模奖励生成；

**📊 数据集**

使用Pinterest内部用户交互日志，包含每日“pivot day”数据、跨表面会话事件（点击、保存、闭合、下载、搜索等），特征数约千级，样本数以百万计；

**📈 对比分析**

通过在线A/B实验与基线比较：在Homefeed、Search、Related Pins、Notifications等多表面，平均提升成功会话（SS）+0.15%~+0.36%，总时长+0.10%，搜索完成率+0.25%，以及DAU/WAU/MAU等长期指标均有正向拉升，说明奖励框架在实际业务中能稳定提升长期用户价值；

**⚠️ 局限性**

局限性包括：1）奖励信号仍需人工设定阈值与权重，可能不完全通用；2）离线筛选与在线效果仍有因分布漂移导致的偏差；3）依赖Pinterest特有的视觉内容与交互（保存、下载），在其他领域需要重新定义奖励；4）跨表面奖励仍可能产生竞争或资源冲突，需要更精细的调度；5）对稀疏长尾行为的处理尚不完善，可能影响极端用户群体的优化效果。

---

## 29. Stop Means Stop: Measuring and Repairing the Enforcement Gap in Agent-Framework Control Primitives

**arXiv ID:** 2607.14166 | [PDF](https://arxiv.org/pdf/2607.14166v1)

**作者:** Sajjad Khan `[一作]` `[通讯]`, Sajjad Khan

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了六个主流LLM代理框架的控制原语，发现其暂停、取消与超时机制并未实现所宣称的“阻塞”语义，导致同级泄漏、重放、取消孤儿和超时僵尸等安全缺陷。

**💡 创新点**

创新点在于首次系统化揭示同级泄漏及其在不同执行模型和语言运行时中的普遍性，并提出一种跨框架、环境外部的“效果门”来显式强制完成、拒绝、去重与取消四条安全属性。

**🔧 技术方法**

使用了模型自由差分探针、实验跑测、公开事件集以及形式化方法（Verus、TLA+/TLC、TLAPS）和并发Rust的Loom模型检查，结合差分一致性测试验证门的正确性。

**📊 数据集**

实验数据来自公开框架版本（如LangGraph、LlamaIndex、Microsoft Agent、OpenAI Agents、CrewAI）、多种LLM（GPT‑4o、Claude、Gemini、DeepSeek、Llama）以及自定义工作流和第三方基准（τ‑bench）。

**📈 对比分析**

与未修复框架对比，实验表明在所有六个框架中，同级泄漏率从约44%（或最高达55%）降至0%，其余重放、取消孤儿与超时僵尸问题亦被完全消除，证明门实现了期望的安全闭合。

**⚠️ 局限性**

局限在于门仅控制网络外部效果，对文件系统、IPC或共享内存等非网络渠道的泄漏不作覆盖；此外，研究仅评估公开框架与模型的现行版本，对未来框架演化或不同部署环境的适用性仍需进一步验证。

---

## 30. Enhancing Small Language Models Reasoning through Knowledge Graph Grounding

**arXiv ID:** 2607.14149 | [PDF](https://arxiv.org/pdf/2607.14149v1)

**作者:** Dimitrios Kelesis `[一作]` (National Center for Scientific Research Demokritos), Georgios Paliouras `[通讯]` (National Center for Scientific Research Demokritos)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个将小型语言模型（SLM）与神经符号工具（事实抽取器和RGCN推理器）结合的代理式框架，提升其在多跳关系推理任务中的零样本性能。

**💡 创新点**

创新点在于将SLM拆解为可调用的工具链，并通过RGCN提供结构化的推理“提示”，同时系统性评估了抽取错误导致的顺序推理脆弱性与噪声“干扰效应”。

**🔧 技术方法**

采用了零样本工具调用、关系三元组抽取、关系图卷积网络（RGCN）以及基于CLUTRR的多跳推理评估。

**📊 数据集**

使用了CLUTRR基准数据集，该数据集包含从叙事文本推断亲属关系的多跳关系链。

**📈 对比分析**

通过Oracle与现实场景对比，Oracle提示下的模型可实现4倍以上的准确率提升，但在实际自抽取场景下RGCN提示仅提升约1.5–2倍，整体准确率仍低于20%。

**⚠️ 局限性**

主要局限在于事实抽取阶段的噪声与错误累积导致的顺序推理崩溃，以及模型对噪声事实的“干扰效应”，使得小模型难以在长链推理任务中保持稳定性能。

---

## 31. Polestar: Drift-Aware Cache Calibration and Token Commitment for Efficient Inference of Diffusion LLMs

**arXiv ID:** 2607.14107 | [PDF](https://arxiv.org/pdf/2607.14107v1)

**作者:** Mingyu Lee `[一作]` (Georgia Institute of Technology), Tushar Krishna `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 14563 | [OpenAlex ID](https://openalex.org/A5034089074)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Polestar，一种无训练的推理框架，利用 token 表示漂移来同时优化 KV 缓存重用和早期 token 提交，从而提升 diffusion 语言模型的推理效率。

**💡 创新点**

创新点在于：① 用 token 漂移（KL 散度）作为统一信号解决 KV 缓存失效和并行解码的矛盾；② 引入 Polestar-Cache 与 Polestar-Commit 两个模块；③ 通过球面 K‑means 聚类、量化与 CPU‑offloading 等技术高效监测漂移。

**🔧 技术方法**

采用的技术包括：KL 散度漂移测量、球面 K‑means 聚类、NVFP4 量化、CPU 迁移、CUDA 流并行化、动态阈值的 drift‑based 提交策略。

**📊 数据集**

使用的数据集涵盖数学与编程任务：GSM8K、MATH、ParallelBench、HumanEval、MBPP、MathVista、MathVerse；模型为 LLaDA‑8B、LLaDA‑1.5、Dream‑7B、LLaDA‑V。

**📈 对比分析**

在 KV 缓存基线（Fast‑dLLM、Elastic‑Cache 等）和并行解码基线（DAWN、KLASS）上进行对比，Polestar 最高可提升 10.7% 的准确率、3.7× 的吞吐量、每步 3.67 个 token 的并行度，创下新一代准确率–吞吐量 Pareto 前沿。

**⚠️ 局限性**

局限性在于：需要针对不同模型和任务调节聚类数、窗口大小等超参；漂移测量和聚类/量化会引入额外开销；在极大模型或不同架构上泛化性尚待验证。

---

## 32. IMEX Interaction-Based Model Explanation

**arXiv ID:** 2607.14096 | [PDF](https://arxiv.org/pdf/2607.14096v1)

**作者:** Emiliano Massi `[一作]` `[通讯]` (Backwell Tech Corp Europe), Emiliano Massi (Backwell Tech Corp Europe)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出IMEX框架，结合静态相关权重PCS和交互相关权重PCI，用于解释黑盒模型预测并捕捉高阶交互效应。

**💡 创新点**

创新点在于：① PCS通过去除单个特征的绝对预测变化量实现归一化特征重要性；② PCI通过比较联合去除与单独去除的差异量量化非加性交互，提供可扩展到任意阶的交互解释；③ 通过与已知结构的合成数据对PCS进行客观验证。

**🔧 技术方法**

使用XGBoost作为预测模型，计算各特征及特征对的预测变化量；采用阈值二值化与基线比较，生成重要性表；利用标准精确率、召回率、F1作为解释度量。

**📊 数据集**

三套合成数据集：① 购买线下场景（7特征，2000样本）；② 线上购买场景（9特征，5000样本）；③ 加油站场景（8特征，6000样本），每个数据集均设计线性、非线性、条件与多重共线性关系。

**📈 对比分析**

通过与INVASE的比较，评估PCS的精确度；在所有三组数据中，PCS在捕捉线性、逆向、条件及多重共线性驱动上往往优于或等同于INVASE，表现出更高的F1分数和更完整的特征覆盖。

**⚠️ 局限性**

局限在于：① PCI仅在理论层面提出，缺乏实验验证；② 交互阈值与基线选择的固定比例（+20%）未系统评估对结果的影响；③ 只针对合成数据，需在真实复杂数据上进一步检验。

---

## 33. Deep-learning Causal Retrieval Optimization for Efficient e-commerce Distribution in Pinterest

**arXiv ID:** 2607.14161 | [PDF](https://arxiv.org/pdf/2607.14161v1)

**作者:** Junpeng Hou `[一作]` (Pinterest, Inc.), Huizhong Duan `[通讯]` (Pinterest, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Pinterest 上构建了基于因果推断的深度多任务模型，用于学习何时触发购物候选生成器，以提升电商内容分发效率。

**💡 创新点**

创新点在于：①将触发决策视为因果策略优化问题，并通过深度模型同时预测结果和上升量；②采用双鲁棒（DR）伪结局提升训练稳定性；③设计线性时间的离线重放评估，可在不影响用户体验的前提下快速选择阈值并预测上线效果；④将学习与阈值决策解耦，既保证可审计性又便于后期业务调优。

**🔧 技术方法**

使用的技术包括：深度多任务学习（DCNv2+MMoE）、双鲁棒上升量损失、Switch‑DR 截断、离线随机化日志收集、线性时间重放算法、PyTorch + DistributedDataParallel、TrochScript 推理、Pinterest 自研的 Scorpion 推理服务。

**📊 数据集**

数据集来自 Pinterest Closeup 表面日志：14 天随机化“购物保留”流量（50/50 触发与不触发），包含用户、上下文、查询特征；评估数据为 1 天同样日志，按购物事件稀疏度提升约 10%。

**📈 对比分析**

比较方法：离线使用 ROC AUC、PR AUC、R‑PR AUC、P@K、R@K 评估模型；在线通过 A/B 测试对比触发率、会话指标、保存/点击等关键业务指标。实验表明：DR+大模型组在离线上实现了更高的召回、低误触；上线后触发率下降约 85%，会话指标保持中性，整体参与度提升 0.26%/0.46%/1.10% 等；同时实现显著的基础设施成本节约。

**⚠️ 局限性**

局限性：①依赖随机化 holdout，若覆盖度不足或后续日志偏离 50/50 设计，因果估计可能失效；②DP（上升量）方差高，易导致阈值选择不稳定；③假设无干扰、重放单元独立，实际中仍可能存在跨请求干扰；④对稀疏事件依赖上升量训练，仍受数据稀疏限制；⑤在不同平台或流量规模下需要额外的样本效率提升措施。

---

## 34. Breaking Refusal in the First Half: A Mechanistic Study of the Prefill Jailbreak

**arXiv ID:** 2607.14147 | [PDF](https://arxiv.org/pdf/2607.14147v1)

**作者:** Alex Kwon `[一作]` `[通讯]` (Independent Researcher), Alex Kwon (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一个简单的预填词“Sure, here is”如何能够让对齐语言模型在面对有害请求时放弃拒绝，进一步探究这一失败背后的内部机制。

**💡 创新点**

创新之处在于通过机理与因果实验将拒绝决策定位为浅层、响应端的计算，并揭示预填词利用了通用自回归条件而非安全特定抑制。

**🔧 技术方法**

主要技术手段包括激活补丁、位置消融、方向探测与注意力边缘消除等因果干预方法，以定位拒绝门失效的位置。

**📊 数据集**

实验使用公开的开源权重模型（Qwen、SmolLM、Phi‑3）并以 AdvBench 有害提示、Alpaca 友好提示及 XSTest 数据集进行验证。

**📈 对比分析**

评估方法通过 AUC 探针与拒绝率对比，展示四个模型/系列在全强预填词下拒绝率降至 0% 而伤害表征保持完整，且通过定位干预可恢复约 48–74% 的拒绝行为。

**⚠️ 局限性**

局限性包括：防御仅对响应端攻击有效，缺乏针对拒绝的可操作性紧凑机制，且在更大规模、强安全训练模型上的适用性尚未验证。

---

## 35. Orchestrating Power Grid Studies with Multi-Agent AI and MCP Servers

**arXiv ID:** 2607.14158 | [PDF](https://arxiv.org/pdf/2607.14158v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 36. Structured Feedback Improves Repair in an LLM Agent Loop

**arXiv ID:** 2607.14167 | [PDF](https://arxiv.org/pdf/2607.14167v1)

**作者:** Jaideep Ray `[一作]` (Independent Researcher), Ankit Goyal `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于代码的LLM代理循环框架，探讨验证器在失败后向下一轮模型调用提供结构化反馈（包括失败位置、观测值和可接受替代方案）是否能提升修复效果。

**💡 创新点**

创新点在于将验证器反馈拆解为可操作的结构化字段，并通过实验验证可接受替代方案对模型修复性能的显著提升；同时比较了文本与键值JSON两种呈现方式的效果。

**🔧 技术方法**

使用的技术包括LLM（Qwen2.5-Coder-14B和Meta‑Llama‑3.1‑8B）与vLLM推理服务、代码驱动的状态控制、外部验证门和结构化反馈编码器。

**📊 数据集**

实验数据集主要是生成的50个TextWorld迷宫游戏（每个游戏有固定种子）以及15个HumanEval代码验证任务。

**📈 对比分析**

通过对比四种反馈策略（原始诊断、自然语言结构化、仅位置观测、完整键值结构化），在四次调用上评估终端成功率；结构化反馈提升了42–44个百分点，且键值格式并未带来额外优势；在不同调用预算和采样解码下的效果保持一致。

**⚠️ 局限性**

局限性包括仅测试生成的迷宫和有限模型，未覆盖大规模仓库级修复；验证器可能不完整，导致无法检测隐藏错误；实验中固定12个可接受动作、未考虑随机化或排序；键值与自然语言的对比并非完全一致。

---

## 37. NexForge: Scaling Executable Agent Tasks via Requirement-First Synthesis

**arXiv ID:** 2607.14186 | [PDF](https://arxiv.org/pdf/2607.14186v1)

**作者:** Jiarong Zhao `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**通讯引用:** 32674 | [OpenAlex ID](https://openalex.org/A5102798483)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个需求驱动（requirement-first）的可执行代理任务合成框架NexForge，能够根据自由文本需求自动生成任务、材料、环境和训练轨迹；

**💡 创新点**

创新点在于将任务合成从传统的基于预定义工具/仓库的“substrate-first”模式转为“requirement-first”，通过研究驱动的需求发现、场景库、分布感知任务编译和自动化环境构建，实现跨领域无域特定基础设施的统一合成流程；

**🔧 技术方法**

使用多源网络爬虫与LLM进行需求发现、分布采样与兼容性过滤；规划与编码Agent完成材料挖掘、蓝图制定与Docker化工作空间；教师轨迹收集与标准化回放；对大模型进行全参数微调；评估采用Terminal-Bench 2.0与GDPval基准；

**📊 数据集**

构建了两大任务集：终端任务集3.6K（和43.2K扩展版）和办公任务集2K（和22K扩展版），并在此基础上训练并评估公开模型Qwen3.5-35B-A3B、Qwen3-32B和公开的Nex-N2系列；

**📈 对比分析**

通过与多种现有代理合成方法（SkillSynth、AgentSynth、DIVE等）及前沿专有模型（Claude Opus、Gemini 3.1 Pro、GPT-5等）比较，NexForge生成的3.6K终端数据将Qwen3.5-35B-A3B性能从22.5%提升至52.0%，在终端基准中达到58.4%（43.2K数据）并超过Claude Opus 4.6；办公基准上从813 Elo提升至1338 Elo，22K任务进一步升至1384 Elo，接近甚至超过多款前沿模型；

**⚠️ 局限性**

局限性包括：对场景生成与任务分布的依赖仍需人工评估；生成的任务对模型本身的学习能力要求较高，可能对小模型效果不明显；在某些复杂或高度特定的域（如医学、法务）仍可能缺乏足够的专业材料与验证机制；

---

## 38. SeeSE3: Emergence of 3D Space in Vision Features

**arXiv ID:** 2607.14228 | [PDF](https://arxiv.org/pdf/2607.14228v1)

**作者:** Caroline Chen `[一作]` (Google DeepMind), Maks Ovsjanikov `[通讯]` (Google DeepMind)

**通讯引用:** 205 | [OpenAlex ID](https://openalex.org/A5108914317)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究视觉基础模型在无主动运动的条件下，是否能通过被动观测学习到 3D 欧氏空间（SE(3)）的结构，并提出 Poincaré 适配器将潜在空间线性化以实现基于潜在空间的视觉里程计和定位。

**💡 创新点**

创新点在于将空间拓扑一致性、内在维度与线性等价性作为统一评测指标，并设计轻量级的 Siamese MLP 适配器将非线性潜在空间“展开”为 SE(3) 的低维子空间，首次实现纯潜在空间的闭式导航。

**🔧 技术方法**

技术包括相互 k‑NN 拓扑一致性度量、局部内在维度估计、全局线性回归，以及 Poincaré 适配器（Siamese MLP）与伪逆逆向导航；还使用了图像编码器（DINOv2、DINOv3、V‑JEPA 等）和预训练的几何模型（DUSt3R、MoGe）。

**📊 数据集**

使用了 ScanNet、ARKitScenes、TUM RGB‑D、12‑Scenes、7‑Scenes 等公开静态场景数据集进行评测。

**📈 对比分析**

与基线（像素、Raw Pixels、显式几何模型）比较，Poincaré 适配器在 DINOv2 上实现 R²≈0.65、Hit@0.3≈0.63，逆向 Poincaré 在潜在导航任务中显著优于 Identity 并逼近显式几何模型的性能，表明潜在空间已充分捕获 SE(3) 结构。

**⚠️ 局限性**

局限在于仅验证静态场景，未考虑场景动态变化；适配器在不同环境下的泛化仍受训练规模和数据多样性的影响，且对深度尺度的辨识仍依赖统计先验。

---

## 39. RegNetAgents: A Multi-Agent Framework for Cross-Network Regulatory Driver Identification in Cancer Genomics

**arXiv ID:** 2607.14097 | [PDF](https://arxiv.org/pdf/2607.14097v1)

**作者:** Jose A. Bird `[一作]` `[通讯]` (Bird AI Solutions), Jose A. Bird (Bird AI Solutions)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了 RegNetAgents，一个多智能体框架，用于跨 TCGA 与 GREmLN 两个 ARACNe 监管网络中识别并排序癌症调控因子；

**💡 创新点**

创新点在于将双网络来源标注、OncoKB 驱动基因过滤、以及 TCGA 网络的 MoA 方向性整合为一个可重复的、可通过自然语言查询的工作流；

**🔧 技术方法**

采用 Python、LangGraph 架构实现多智能体 DAG 工作流，并使用 Fisher 精确检验、Permutation 置换、BH‑FDR 校正和 Stouffer Z 统计；

**📊 数据集**

使用 TCGA 8 种癌症的 ARACNe 网络（BRCA、COAD 等）以及 GREmLN 项目的单细胞 ARACNe 网络，参考 OncoKB 1,231 名肿瘤驱动基因；

**📈 对比分析**

通过与 OncoKB 的富集比较验证，TCGA‑only 与 GREmLN‑only 侯选列表在 BRCA 与 COAD 中分别取得 Stouffer Z=6.69 与 6.95 的显著富集，控制实验（Housekeeping、Non‑driver）均未显示同样富集，表明方法能显著提高候选驱动基因识别的信号；

**⚠️ 局限性**

局限包括网络为群体平均，缺乏细胞状态和亚克隆异质性；GREmLN 网络规模比 TCGA 小导致源标签偏差；仅使用单一上皮细胞类型作为单细胞参考；候选列表的 MoA 注释尚需更系统验证。

---

## 40. HG-RAG: Hierarchy-Guided Retrieval-Augmented Generation for Structured Knowledge Graphs

**arXiv ID:** 2607.14095 | [PDF](https://arxiv.org/pdf/2607.14095v1)

**作者:** Pranav Yadav `[一作]` `[通讯]` (University of California, Merced), Pranav Yadav (University of California, Merced)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种层级引导检索增强生成框架HG‑RAG，利用图遍历在层级知识图中检索结构化上下文供LLM回答。

**💡 创新点**

引入层级结构感知的图遍历策略，包括向上、横向、向下的k步检索，生成结构化子图作为上下文，解决传统RAG在层级、关系推理上的不足。

**🔧 技术方法**

结合LLM实体识别、图遍历算法、结构化序列化、密集向量检索对比、LLM评判器和多指标评估等技术。

**📊 数据集**

使用合成的三层次（星球‑国家‑城市）层级知识图，规模分别为约18、150、800节点，并在每个规模上构造50道不同类型的查询。

**📈 对比分析**

与基线密集向量检索RAG进行对比，评估事实准确率、幻觉率、局部性意识等指标；HG‑RAG在所有规模下均优于基线，尤其在多跳推理上平均得分为4.1/5，基线在大规模时降至1.66。

**⚠️ 局限性**

子图节点上限固定导致小规模图噪声高；评判器与回答模型相同可能产生自一致偏差；无法处理无命名实体的属性查询等。

---

## 41. MemoHarness: Agent Harnesses That Learn from Experience

**arXiv ID:** 2607.14159 | [PDF](https://arxiv.org/pdf/2607.14159v1)

**作者:** Yue Huang `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 13262 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可在推理时根据过去执行经验自适应调整的 LLM agent harness，能够在不需要额外反馈或梯度更新的情况下为每个测试案例生成专门化的控制配置。

**💡 创新点**

创新点在于：①将 harness 拆分为六个可编辑的控制维度（上下文、工具、生成、编排、记忆、输出），实现结构化搜索；②构建双层经验库，记录每个案例的执行轨迹与诊断信息，并周期性抽取全局失败模式；③采用“正确性优先”策略进行搜索与案例适配，无需标注反馈即可完成推理时自适应。

**🔧 技术方法**

核心技术包括：结构化搜索策略（在六维空间中迭代优化）、经验库存储与检索（基于相似度与诊断信息）、案例适配规则（利用检索到的成功/失败实例及全局模式调整全局 harness），以及 token‑成本优先的决策规则。

**📊 数据集**

在三个任务族上评估：长路径 shell-agent、单 shot 代码生成和多步财务推理，并使用了六个外部评测数据集以及六种不同基础模型（OpenAI、Claude、Gemini、Qwen、ChatGLM、DeepSeek）。

**📈 对比分析**

与四个公开或基准提供的 harness 基线相比，学习到的 harness 在 shell-agent 基准上将成功率从 0.722 提升至 0.806（提升 0.084），在其他基准也表现出显著改进。跨数据集、跨模型迁移实验表明学习到的 harness 在未见评测集和其他模型上平均提升 0.098，且在成本方面通过缓存技术实现了更低的美元费用。

**⚠️ 局限性**

局限性包括：①评估覆盖的任务范围有限，尚未验证在更大规模或更复杂场景下的鲁棒性；②缺乏对每个 harness 维度贡献的细粒度因果归因；③检索与经验库扩展的可扩展性与实时性尚未深入探讨；④适配规则仍基于手工设计的查询与检索策略，可能在不同应用中需要进一步调优。

---

## 42. Augmentations for Robust and Efficient Imitation Learning in Streamed Video Games

**arXiv ID:** 2607.14200 | [PDF](https://arxiv.org/pdf/2607.14200v1)

**作者:** Somjit Nath `[一作]` (McGill University), Lukas Schäfer `[通讯]` (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一套针对低带宽流媒体游戏环境的时空视频增强技术（scrubs、pixelation、fuzziness、ghosting），并将其与传统图像增强结合，应用于离线模仿学习中的 PIDM 框架，以提升样本效率和对流媒体视觉噪声的鲁棒性。

**💡 创新点**

创新点在于首次将时空相关的流媒体失真模型（如连续的像素块、模糊、轨迹残影）融入数据增强流程，模拟真实网络延迟下的视觉畸变，并通过周期性正弦调制和淡入淡出控制增强的平滑性；同时展示了该增强对离线模仿学习性能的显著提升。

**🔧 技术方法**

使用了预测逆动力学模型 (PIDM) 作为学习框架；预训练的视觉编码器（Theia backbone）；标准图像增强（随机仿射、颜色抖动）；自定义时空增强（scrub、pixelation、fuzziness、ghosting）；以及在训练时预缓存多种增强版本。

**📊 数据集**

在两款商业3D游戏（Game 1与Game 2）中收集了30个专家演示（每个任务），分别覆盖3个任务（Game 1的Task 1/2与Game 2的Task 3）。

**📈 对比分析**

与四种训练配置（无增强、仅标准增强、仅流媒体增强、全部增强）进行对比。结果显示：在少量样本（5–10个演示）下，全部增强可提升至41% milestone 完成率；在注入2–10 ms网络延迟时，全部增强的下降幅度仅为7.45%，而无增强时为49.82%；在Game 2的长周期任务中，全部增强使 milestone 完成率提升约10%。

**⚠️ 局限性**

局限性包括：实验仅涵盖两款游戏和有限任务；增强强度固定，未实现根据实时网络质量自适应；仅在离线模仿学习和PIDM模型上验证，未探究对其他RL/IL算法或在线学习的适用性；且未评估更极端或多样化的网络条件与视觉失真组合。

---

## 43. Information-Theoretic Limits of Reliability and Scaling in Language Models

**arXiv ID:** 2607.14112 | [PDF](https://arxiv.org/pdf/2607.14112v1)

**作者:** Subhabrata Majumdar `[一作]` `[通讯]` (Indian Institute of Management Bangalore), Subhabrata Majumdar (Indian Institute of Management Bangalore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于信息理论的LLM可靠性上限与任务内部依赖结构（依赖核）的统一框架，推导出包含可靠性、可观测上下文与自回归误差传播的极限表达式，并从谱特征出发给出了新的缩放定律

**💡 创新点**

创新点在于（1）引入可靠性上限和可观测/不可观测上下文分解；（2）定义自回归衰减的“依赖核”并证明其对误差传播的决定性作用；（3）基于谱指数ν、μ推导出最大值形式的缩放法则，统一解释Chinchilla定律和多种实践现象

**🔧 技术方法**

运用了信息瓶颈、数据处理不等式、谱分析、核回归理论以及自回归连续性假设等技术

**📊 数据集**

在编码（HumanEval、MBPP）、数学（GSM8K）、问答（SimpleQA、Natural Questions、QuAC）和写作（CNN/DailyMail、WritingPrompts）等公开基准上验证了模型的可靠性上限与依赖核结构

**📈 对比分析**

通过对比不同任务类型的Saturation Index与最高分，展示了可靠性上限差异；与Chinchilla缩放法则比较时，提出max‑form 预测即资源不平衡时无进一步收益，验证了自回归衰减和依赖核对性能的影响

**⚠️ 局限性**

局限性包括：仅适用于自回归生成；假设任务分布与连续性条件；未考虑非自回归架构、分布漂移、复杂多模态任务；以及对实际硬件与优化细节的简化

---

## 44. Automatically Evolving Prompt Guidelines for Task-Specific Optimization

**arXiv ID:** 2607.14105 | [PDF](https://arxiv.org/pdf/2607.14105v1)

**作者:** Cedric Richter `[一作]` (University of Luxembourg), Mike Papadakis `[通讯]` (University of Luxembourg)

**通讯引用:** 6536 | [OpenAlex ID](https://openalex.org/A5081145634)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自动化的提示工程指南优化方法（AGoP），通过提示写作 LLM、求解 LLM 与评估机制的交互，迭代生成任务特定的提示编写指南，以帮助用户在实际对话中写出更完整、信息更充分的提示，从而显著提升大型语言模型的下游任务表现。

**💡 创新点**

创新点在于：①利用已完成任务的参考答案从中抽取隐含的任务约束与假设，避免直接泄露答案；②设计了基于 n‑gram 的泄漏检测指标 λ 并通过拒绝采样控制答案泄露；③构建了可与任意提示优化器耦合的代理目标，并使用遗传算法 GEPA 对指南进行自动演化；④将生成的指南既可用于用户编写提示，也可嵌入系统提示，促进模型主动提问。

**🔧 技术方法**

核心技术包括：Prompt Simulation（提示写作 LLM 与求解 LLM 的模拟链）、n‑gram 泄漏检测与阈值控制、遗传式提示优化（GEPA）、任务特定评估器（如 Pass@1、准确率、ROUGE‑L）以及对照实验中的基线优化器（MIPROv2、GEPA 无指南）。

**📊 数据集**

使用的公开基准包括：数学推理（MMLU‑Math、GSM8K）、医学问答（MediQ）、代码生成（MBPP、HumanEval+、MBPP+）。数据量覆盖多任务、不同难度，且通过完整与不完整描述的子集来检验指导效果。

**📈 对比分析**

与基线（原始未优化提示、MIPROv2、GEPA 无指南）对比，使用任务特定指南可将模型准确率提升 15.5%–81.7%（平均 50%+），显著缩小“未指定”导致的高达 95.3% 的性能下降；在部分基准上还实现了 33 Pass@1 的提升和 14% 的准确率提升；同时降低了模型的拒绝率与不一致性。

**⚠️ 局限性**

局限性包括：①未在真实用户实验中验证指南的可用性与用户遵从度；②泄漏检测可能漏检非 n‑gram 形式的信息泄露；③仅在两款 LLM（GPT‑4.1‑mini 与 Qwen3‑32B）上评估，泛化性仍需进一步检验；④框架复杂度较高，需要多轮模拟与进化，计算成本不低。

---

## 45. DialogueVPR: Towards Conversational Visual Place Recognition

**arXiv ID:** 2607.14115 | [PDF](https://arxiv.org/pdf/2607.14115v1)

**作者:** Yukun Song `[一作]` (Beijing University Of Posts And Telecommunications), Pengyang Wang `[通讯]` (University Of Macau)

**通讯引用:** 8815 | [OpenAlex ID](https://openalex.org/A5036270316)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Dialogue Place Recognition (DlgPR)，将地理定位从单轮检索转变为多轮交互式推理，并构建了大规模对话式数据集 DlgQuest-Cities。

**💡 创新点**

创新点包括：①全新的交互式定位任务 DlgPR；②首个面向对话的城市定位基准 DlgQuest-Cities；③结合跨模态递进学习检索器 CMPL 与智能问答生成器 DQ‑pilot 的联合框架；④使用基于 Discriminative Difficulty Index (DDI) 的层次化训练和 GRPO 强化学习来优化问答策略。

**🔧 技术方法**

技术手段包括：多模态跨模态递进学习检索器 CMPL；链式思考 (CoT) 生成的对话脚本；基于 Qwen2.5‑VL 的大型视觉语言模型 DQ‑pilot；GRPO 强化学习结合 Positional Retrieval Gain (PRG) 与格式奖励；硬负样本隔离、saliency filtering、SDM loss 等。

**📊 数据集**

使用的数据集为 DQ‑cities（30k 对话样本，基于 GSV‑Cities 图像），以及其子集 DQ‑cities‑20k（SFT 训练）和 DQ‑cities‑10k（GRPO 训练）。

**📈 对比分析**

与 Qwen2.5‑VL‑7B/72B、PlugIR 等交互检索基线以及 CLIP、Long‑CLIP、FG‑CLIP 等静态检索模型对比。实验显示，DlgPR（SFT+GRPO）在五大城市的 Round5 处 R@1≈86.6、R@5≈90.4，显著高于基线；CMPL 在单轮静态检索时亦优于 CLIP、FG‑CLIP，提升幅度显著。

**⚠️ 局限性**

局限性：仍需人工构造大量对话数据；依赖预训练 LLM 与视觉模型，可能在极端模糊描述或全新环境下性能下降；对话深度受限于候选集多样性；未评估实时部署与真实机器人交互。

---

## 46. Cross-Dataset Generalization in Urdu Fake News Detection: An Empirical Study with XLM-RoBERTa and a Length Confound Analysis

**arXiv ID:** 2607.14131 | [PDF](https://arxiv.org/pdf/2607.14131v1)

**作者:** Muhammad Abdullah Haroon `[一作]` (National University of Computer and Emerging Sciences), Muhammad Abdullah Haroon `[通讯]` (National University of Computer and Emerging Sciences)

**通讯引用:** 492 | [OpenAlex ID](https://openalex.org/A5072998716)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

做了跨数据集通用性研究，评估XLM‑RoBERTa在乌尔都语假新闻检测中的零射转移表现，并发现长度confound导致模型在A→B方向崩溃。

**💡 创新点**

首次揭示乌尔都语假新闻数据集中存在的长度偏差及其对跨域泛化的破坏性影响，并提出基于预测标签分布与长度分析的诊断方法。

**🔧 技术方法**

使用XLM‑RoBERTa transformer、TF‑IDF+LR/SVM基线，并结合零射转移实验和长度截断对比来评估模型性能。

**📊 数据集**

Ax‑to‑Grind乌尔都语假新闻集（10,083 篇）和Notri‑Fact集（13,388 篇）两大平衡数据集。

**📈 对比分析**

与基线相比，XLM‑RoBERTa在A→B零射转移时F1从0.929骤降至0.005；B→A转移仍保持0.771；基线单域表现相对较低；长度截断实验显示长度confound约贡献0.007。

**⚠️ 局限性**

仅测试两大数据集，未尝试更大模型或适应技术；长度截断控制过于温和，可能未完全剔除confound；未探究域适应或不同预训练模型的效果。

---

## 47. CoEvoT: Co-Evolving Chain-of-Thought Prompting for Graph-LLM Reasoning

**arXiv ID:** 2607.14114 | [PDF](https://arxiv.org/pdf/2607.14114v1)

**作者:** Haohua Niu `[一作]` (Sun Yat Sen University), Yuan Fang `[通讯]` (Singapore Management University)

**通讯引用:** 4143 | [OpenAlex ID](https://openalex.org/A5027522861)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于CoT（Chain‑of‑Thought）提示的图-LLM推理框架，能够在多步推理过程中将中间思考结果动态地转化为图结构的更新，从而实现语言推理与图结构的闭环共进。

**💡 创新点**

创新点在于：① 构建了“文本→图”条件重写模块，利用每一步的思考向量作为条件对节点图标记进行可学习的残差更新；② 将更新后的图标记回馈到下一步提示中，形成状态感知的推理过程；③ 通过仅优化线性投影和条件网络实现零/少样本跨数据集迁移，避免了大规模图模型微调。

**🔧 技术方法**

使用的技术包括：LLM提示（CoT）、图标记化（线性投影将节点嵌入映射到LLM嵌入空间）、条件网络（两层MLP用于生成节点级更新向量）、对比式多模态预训练（图结构与LLM语义对齐）、零样本迁移策略。

**📊 数据集**

实验使用8个公开图数据集：文献网络（Arxiv、Pubmed、Cora）、电子商务网络（Children、History、Computer、Photo、Sports）。

**📈 对比分析**

与包括传统GNN、Self‑Supervised、Graph Transformer、LLM单步推理等多种基线比较，Co‑CoT在节点分类和链式推理后链接预测上均取得显著提升，平均准确率/ AUC 超过TEA‑GLM、GOFA 等最强对手；通过消融实验验证了共进机制的必要性。

**⚠️ 局限性**

局限性：① 依赖大规模LLM，推理成本高；② 目前多步推理固定为2步，未探究更深层次推理的效果；③ 仅在节点属性丰富的图上验证，尚未测试极度稀疏或无属性图的适应性。

---

## 48. DiMaS: Distribution Matching for Steering Vision-Language-Action Models

**arXiv ID:** 2607.14280 | [PDF](https://arxiv.org/pdf/2607.14280v1)

**作者:** Pegah Khayatan `[一作]` (Sorbonne Université), Matthieu Cord `[通讯]` (Sorbonne Université)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对流匹配视觉‑语言‑动作模型的分布匹配调控方法 DiMaS，用于实现细粒度的行为控制。

**💡 创新点**

创新点在于将调控视为在表示分布间的匹配，而非传统的线性方向移动，解决了行为特征线性可解但不可线性调控的问题。

**🔧 技术方法**

使用了分布匹配调控技术、流匹配 VLA 框架，并对动作专家的表示结构进行分析。

**📊 数据集**

在两种先进的 VLA 模型上进行实验，实验数据来源于常用机器人操控任务数据集（如 Meta‑World 或 Dex‑Net，具体未明确）。

**📈 对比分析**

通过与经典线性调控方法对比，DiMaS 在两款 VLA 上实现了更有效的行为控制；在任务相似度较高的情形下表现出良好的迁移性，差异较大的任务迁移效果则有所下降。

**⚠️ 局限性**

局限性包括：对完全新颖的任务泛化能力有限；在更复杂或动态环境下的鲁棒性尚未充分验证。

---

## 49. Closed-Loop Knowledge Dynamics: An Operational Framework for Saturation and Escape

**arXiv ID:** 2607.14185 | [PDF](https://arxiv.org/pdf/2607.14185v1)

**作者:** Xuening Wu `[一作]` (Pfizer), Shenqin Yin `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出闭环知识系统的三层操作框架，并给出饱和（稳定）与逃逸（结构变更）的理论条件，在LLM自我修复、强化学习与贝叶斯优化等实验中验证。

**💡 创新点**

将饱和与逃逸分离，定义可观测结构参数和可检验的 kernel 差异；利用 Lyapunov 驻点与基线相对 KL 的逃逸阈值，提供统一的实验接口与评估指标。

**🔧 技术方法**

使用动态系统理论、Lyapunov 漂移、Wasserstein 距离、KL 信息距离、Phase1/Phase2 对照实验；在 LLM 自我修复、PPO+BC、贝叶斯优化等具体算法上实现。

**📊 数据集**

实验数据包括：两条历史代码错误（gitwildmatch 与 unit‑rollover）用于 LLM 修复；合成的三步任务（pick up key → open door → reach goal）用于 RL；以及 Styblinski‑Tang 2D、Hartmann‑6 与一个两盆目标用于 BO。

**📈 对比分析**

通过对比基线继续、通用反馈、错误仅反馈、匹配反馈等条件，评估逃逸效果。LLM 中错误反馈将通过率从 0.143 提升至 0.469，诊断反馈达到 1.0；RL 中 BC+PPO 在 20000 次样本后成功率升至 0.70 并保持；BO 中目标对齐反馈仅需 1–2 次观测即可实现全局逃逸。整体表明目标对齐的外部信息比通用信息更高效。

**⚠️ 局限性**

限制：未直接估计 δ_μ,ν 与 KL 阈值，仅验证逃逸现象；只分析单次干预；未涵盖振荡、发散或对抗反馈；假设表示空间有限；实验多为探索性，缺乏事前注册与全面统计检验。

---

## 50. RxBrain: Embodied Cognition Foundation Model with Joint Language-Visual Reasoning and Imagination

**arXiv ID:** 2607.14187 | [PDF](https://arxiv.org/pdf/2607.14187v1)

**作者:** Haotian Liang `[一作]`, Zhengyou Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Hy‑Embodied‑RxBrain，一种能够在单一规划序列中同时进行语言推理与视觉想象的基础模型，并通过自动化视频处理生成联合文本‑视觉训练数据。

**💡 创新点**

创新点在于：①把语言逻辑与视觉状态预测整合到同一生成流程中；②采用统一的 Mixture‑of‑Transformers（MoT）架构，让视觉理解、视觉生成和语言生成共享注意力但保持专属专家；③设计流匹配（flow‑matching）视觉与动作生成目标，支持短期未来预测和连续动作生成；④构建 RxBrain‑Bench 三轨迹评测（EVQA、WorldPred、JointPlan），直接评估闭环交互式规划能力。

**🔧 技术方法**

核心技术包括：多模态 MiT、VAE 视觉编码/解码、流匹配生成目标、跨模态交叉注意力、自动化视频分段与质量验证、分层训练（先基础 VLM 再嵌入式微调）、以及动作生成的 gated‑expert 融合。

**📊 数据集**

使用了来自 4 大来源的 50,177.2 小时机器人/人类/仿真/第一人称视频数据，经过关键帧采样、步骤提议、边界细化等流程生成约 21,506,919 条联合文本‑视觉计划样本；此外使用 GenEval、ERQA、EmbSpatial、CV‑Bench、Share‑Robot、RoboSpatial、MindCube 等公开多模态基准进行评估。

**📈 对比分析**

在传统图像生成（GenEval）中与 Bagel 竞争并接近 Cosmos‑3‑Nano；在多模态理解与空间推理基准上跑在前列，特别是 3DRSBench、MMSI‑Bench 等；在 RxBrain‑Bench‑WorldPred 上取得 0.62 的加权 MLLM‑judge 分数，明显优于 Wan2.2‑TI2V（0.43）和 Cosmos3‑Nano（0.59）；在 RxBrain‑Bench‑JointPlan 上获得 0.68 的加权分数，明显高于 Cosmos3‑Nano Agent（0.52）、BAGEL‑7B‑MoT（0.50）和 Qwen‑Agent（0.43）；在三项真实机器人任务中平均成功率 87%，显著超过 π₀（68%）和 π₀.₅（82%）。

**⚠️ 局限性**

局限包括：①模型规模相对较小，难以处理极长规划或极细粒度视觉细节；②跨模态生成存在训练‑推理不一致，尤其是视觉流匹配与文本自回归的同步问题；③对完全离散环境、不同硬件/传感器的泛化仍需进一步微调；④未使用强化学习或序列级优化来进一步校准闭环规划。

---

## 51. Inference-Time Concept Suppression and Video-Centric Evaluation for Text-to-Video Models

**arXiv ID:** 2607.14194 | [PDF](https://arxiv.org/pdf/2607.14194v1)

**作者:** Wenxuan Chen `[一作]` (University of Science and Technology of China), Wenjie Feng `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种推理时的文本到视频概念遗忘框架SIRUS，并配套视频评估框架VUEF；

**💡 创新点**

创新点在于将目标概念建模为文本嵌入子空间，并通过层级触发规则定位目标词，结合局部投影与采样时的残差减法实现无训练参数改动的概念遗忘；同时提出全维度（遗忘、保留、质量、鲁棒性、效率）的评估方法；

**🔧 技术方法**

使用文本子空间构造、层级触发、局部投影、正概念参考分支、残差减法等技术；评估框架VUEF包含视频级失效、帧级残余、配对保留、VBench质量诊断、部署开销等指标；

**📊 数据集**

在CogVideoX和Wan2.2两大文本到视频生成模型上，针对五个目标概念（裸露、教堂、垃圾卡车、降落伞、梵高风格）进行实验；

**📈 对比分析**

与VideoEraser、SAFREE、T2VUnlearning、Refusal Vector等基线对比，SIRUS在CogVideoX上平均遗忘成功率70.4%、帧级残余率25.7%，视频质量下降仅-0.016，优于对手；在Wan2.2上也实现73.6%平均成功；

**⚠️ 局限性**

仍面临对视觉上持续显著的目标（如降落伞）难以完全消除、遗忘与保留需权衡以及在不同生成模型上的迁移性受限等局限。

---

## 52. Semantic Audio-driven Understanding for Dynamic Humanoid Whole Body Control

**arXiv ID:** 2607.14182 | [PDF](https://arxiv.org/pdf/2607.14182v1)

**作者:** J. M. A. Marcelo `[一作]` (Sapienza University of Rome), V. Suriani `[通讯]` (Sapienza University of Rome)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个多模态框架，使仿人机器人能够实时从音乐和语音流中检索并执行对应的全身动作。

**💡 创新点**

创新点在于统一的检索模块将音频特征（指纹、语义嵌入）映射到动作库，并支持音乐时序和语音意图的即时触发。

**🔧 技术方法**

使用音频指纹、CLAP 语义嵌入、AST 场景分类、Silero VAD、Whisper 语音识别、BeyondMimic RL 训练的动作库、MuJoCo 模拟、Unitree G1 机器人硬件以及 TCP 控制接口。

**📊 数据集**

使用公开的音乐片段（Salsa、Dynamite、Swim、Thriller 等）以及对应的动作库，验证了 574 个音频块的检索。

**📈 对比分析**

通过在仿真中对比 20s 与 30s 节奏切换的混合曲目，检索准确率为 84.8%，30s 方案能保持 90% 以上的动作同步；在实际 Unitree G1 上实现了接近仿真的执行。

**⚠️ 局限性**

局限性包括：站立平移阶段导致的切换延迟，短块长度导致的切换不稳；依赖外部云服务的语音识别和嵌入导致总体延迟；动作库仅覆盖预定义技能，缺乏自适应生成能力。

---

## 53. Position: Explainability Research Must Prioritize Foundations over Ad-hoc Methods

**arXiv ID:** 2607.14123 | [PDF](https://arxiv.org/pdf/2607.14123v1)

**作者:** Michal Moshkovitz `[一作]` (Google), Jennifer Wortman Vaughan `[通讯]` (Microsoft)

**通讯引用:** 28656 | [OpenAlex ID](https://openalex.org/A5043117896)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出XAI研究应从基础问题着手，强调定义、属性、评估和可操作性四大挑战，并提供检查表和实证调查。

**💡 创新点**

核心创新是把XAI视为人本交互系统，系统化识别基础缺陷，提出基于任务的评估框架和可操作性流水线，并用LLM和实地问卷验证现状。

**🔧 技术方法**

采用LLM（Gemini 2.5 Flash）对NeurIPS/ICML/ICLR 2023-24论文进行自动化问答分析，结合问卷与访谈收集实践者意见。

**📊 数据集**

主要数据集为学术论文列表（约617篇）以及34位研究者/从业者的问卷响应；未使用传统ML数据集。

**📈 对比分析**

对比方法方面未做实验性比较，而是统计方法与评估的使用比例，发现77%提出新方法、11%给出形式化定义、56%进行可信度评估、仅11%包含人类评估。

**⚠️ 局限性**

局限性在于LLM的自动标注可能误差、问卷样本量有限、未给出具体可操作的实现细节，且研究侧重于理论框架而缺乏实验验证。

---

## 54. When a Verified World Model Still Loses: Play-Adequacy vs Prediction-Accuracy in LLM-Synthesized Code World Models

**arXiv ID:** 2607.14169 | [PDF](https://arxiv.org/pdf/2607.14169v1)

**作者:** Javier Aguilar Martín `[一作]` `[通讯]` (AGILabs), Javier Aguilar Martín (AGILabs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了用大语言模型（LLM）生成可执行的游戏规则（Code World Model, CWM），并在经典规划器（如UCT‑MCTS）中搜索，通过随机轨迹门控验证模型后，探讨门控通过是否能保证规划性能。

**💡 创新点**

创新点包括：①发现“验证‑正确性”差距，即门控通过的CWM在规划时仍可被击败；②提出定量危险法则 danger = play_cost × (1‑rarity)^N，揭示门控失效的概率与规则稀有度的关系；③证明LLM在合成规则时表现为翻译而非推断，且通过示例无法修复缺失规则；④在不完全信息游戏中构造最小对手 Beacon，证明类似缺口也能出现。

**🔧 技术方法**

技术方法包括：使用 GPT‑5.x 进行提示式代码合成；利用 UCT‑MCTS（完美信息）与确定化 MCTS（不完全信息）进行规划；门控通过基于随机轨迹的过渡准确率；统计置信区间（Wilson 95% CI）评估胜率；并给出可识别性、覆盖性与危险性等理论证明。

**📊 数据集**

使用的数据集包括公开游戏（井字棋、连连看、6×6 井字棋、army5x5a、Trike）以及不完全信息游戏（Kuhn poker、Leduc poker）和自制游戏 Beacon；训练样本由随机轨迹、DAgger 生成的对抗样本或手工构造的判别性样本组成。

**📈 对比分析**

通过对战公平基准（truth‑vs‑truth）与门控通过的 CWM+MCTS 进行比较，评估胜率与 play_cost；实验显示门控通过但缺失关键规则时，胜率下降约 0.09–0.15，且即使提供了大量判别性示例也无法让 LLM 学习该规则。相比直接使用 LLM 作为策略，CWM+MCTS 仍能保持显著优势。

**⚠️ 局限性**

局限性包括：门控失效主要在罕见且关键的规则/信息集上出现；理论证明依赖于特定假设（如可识别性、覆盖性），在更大规模游戏或更强规划器下不一定成立；LLM 在推断缺失规则时表现有限；以及 Beacon 等极简游戏并不能代表真实复杂游戏的通用性。

---

## 55. KeyFrame-Compass: Towards Comprehensive Evaluation of Keyframe-Conditioned Video Generation

**arXiv ID:** 2607.14202 | [PDF](https://arxiv.org/pdf/2607.14202v1)

**作者:** Yuqi Tang `[一作]` (HKUST(GZ)), Yuanxing Zhang `[通讯]` (Kling Team)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了专门针对多关键帧条件视频生成的基准与评估框架，包含386个测试案例，覆盖不同视频结构、关键帧密度、提示粒度、输入格式和应用域；

**💡 创新点**

创新点在于将关键帧执行拆解为六个可测量指标（Presence、Fidelity、Ordering、Position Accuracy、Persistence、Uniqueness），并将其与整体视频质量、时空一致性、指令遵循和音视频协调等四维通用质量指标结合，形成统一自动化评估体系；

**🔧 技术方法**

利用多模态大语言模型（Gemini 3.1 Pro、GPT‑5.5 等）进行评估判断，配合视觉模型（DINOv3、SAM、InceptionNeXt、ElasticFace、MonST3R 等）与音频-视觉相似度模型（CLAP、ImageBind）实现关键帧匹配、语义一致性与音视频同步检测；

**📊 数据集**

构造数据集基于 ViStoryBench、VIST、ROCStories 等故事源，结合真实视频生成叙事字幕，随后使用 GPT‑Image‑2 与 Nano Banana Pro 生成关键帧图像，经过多模态一致性检查与人工审核，最终形成三大应用域（日常捕捉、产品展示、电影叙事）的样本；

**📈 对比分析**

通过与九款代表性视频生成模型（包括四款专有模型 Gemini‑Omni‑Flash、Kling‑3.0‑Omni、Seedance 2.0、Wan2.7‑I2V 以及五款开源模型）在短视频（≤10 s）与长视频（10–45 s）两种生成模式下进行对比；结果显示，Seedance 2.0 在整体分数最高（0.807），LTX‑2.3 在关键帧执行最强，但整体质量相对较弱；专有模型在指令遵循和音视频协调上表现优于开源模型；

**⚠️ 局限性**

局限性包括：①关键帧执行与视频连贯性仍存在明显权衡，模型往往在保持关键帧忠实度时出现断裂或不自然过渡；②高密度关键帧约束导致指令遵循下降；③开源模型普遍缺乏对 storyboard‑grid 输入的理解，导致生成时全局忽略时序；④对长视频的一键生成能力有限，需多轮代理模式，仍难以保证跨段时序一致性。

---

## 56. The Severance Problem: LLMs are Unaware of the Person Beyond the Prompt

**arXiv ID:** 2607.14250 | [PDF](https://arxiv.org/pdf/2607.14250v1)

**作者:** Dor Litvak `[一作]` (University of Texas at Austin), Liu Leqi `[通讯]` (University of Texas at Austin)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5019160218)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究并解决了语言模型在个性化对话中因缺失对用户完整生活情境的认知而产生的“隔离问题”。

**💡 创新点**

提出了“Severance Schema”，一种结构化的未知类别清单，让模型在不知道信息时能够意识到并主动询问缺失的用户上下文。

**🔧 技术方法**

通过在提示中嵌入该Schema并结合长期记忆检索，改进了LLM的澄清与安全性。

**📊 数据集**

使用合成的10个用户画像和30个建议场景的自定义数据集，涵盖健康、法律、金融等多领域。

**📈 对比分析**

与无Schema、仅记忆、Schema+记忆等四种对照实验对比，五大模型在安全性上从14–19%降至2–10%，谄媚率降至2–9%，并显著降低了3.7–11.7%的幻觉率。

**⚠️ 局限性**

主要局限包括：对特定领域任务的效果尚未验证，实验使用合成用户画像，长期累积记忆对Schema效果的影响未知，且需在获取信息前获得用户同意。

---

## 57. Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation

**arXiv ID:** 2607.14203 | [PDF](https://arxiv.org/pdf/2607.14203v1)

**作者:** NVIDIA `[一作]` (Nvidia), Sanja Fidler `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种单前向传递的神经重建模型，将短时多视角驾驶日志快速转化为可用于闭环仿真的分层3D高斯散点场（包含静态、动态和天空层）

**💡 创新点**

核心创新在于：①使用交替注意力视觉 Transformer 对多相机/多时域输入进行统一编码；②多头轻量化解码器同时输出深度、法线、语义、天空立方图、ISP 校正以及3D高斯参数；③通过查询点预测动态轨迹（分段线性）实现动态层无需后期轨迹标注；④长序列采用分块处理+视锥归属剔除避免漂浮点；⑤在LiDAR数据上亦能直接生成高斯点云。

**🔧 技术方法**

技术包括：交替注意力 ViT、DPT 级联头、3DGUT 渲染器、LPIPS + 视差/深度监督、AdaLN 时序编码、查询点采样策略（Dense/Selective）以及多阶段训练（预训练、上下文+运动、Gauss+ISP）

**📊 数据集**

使用约 40K 条内部 NVIDIA AV 采集的多摄像头驾驶日志（包含 LiDAR、语义、轨迹等辅助数据）以及公开的 Waymo Open Dataset 进行验证

**📈 对比分析**

与现有 feed-forward 基线（DepthSplat、STORM、Depth‑Anything‑3、DGGT）对比，PSNR 最高 28.26 dB（比最强基线高 2.01 dB），在 Waymo 上动态区域的视觉质量明显优于对手；在内部数据上与 per‑scene 优化的 NuRec 相比，重建速度提升 3~4 位数（1.5 s vs 75 min），同时在闭环策略评估中保持与 NuRec 相同的策略排序

**⚠️ 局限性**

局限性包括：①高质量重建需大量高斯点，导致存储/渲染开销；②对训练集外的摄像头布置（如鱼眼或低挂）泛化不足；③三关键帧分段线性运动模型无法捕捉瞬时非刚性运动；④长日志需要分块拼接，仍存在边界漂浮问题，尚未实现完全在线流式重建

---

## 58. How Artificial Intelligence LLM Engines Shape the Global Conflict Information Environment

**arXiv ID:** 2607.14197 | [PDF](https://arxiv.org/pdf/2607.14197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 59. Enhanced Feedback Mechanisms for Resource-Efficient Incremental Redundancy

**arXiv ID:** 2607.14247 | [PDF](https://arxiv.org/pdf/2607.14247v1)

**作者:** Mustafa Cemil Coşkun `[一作]` (Nokia Bell Labs), Homa Esfahanizadeh `[通讯]` (Nokia Bell Labs)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了两类增强型增量冗余（IR）HARQ机制，能够在首次传输后基于信道统计或接收LLR预测所需冗余量，从而避免过度或不足的重传，提高吞吐量并降低延迟。

**💡 创新点**

创新点包括：①基于SNR构造查找表，给出满足目标可靠性下界的最小冗余预算，并推导出可靠性下界；②设计轻量级早期反馈预测器（使用一维CNN）在第一次传输完成后直接输出所需的RV数量或MCS调整决策，避免全解码和多轮等待。

**🔧 技术方法**

技术手段：极化码和5G NR LDPC的IR-HARQ实现、SNR查找表（One-shot/Two-shot）、逻辑回归分类器（SNR基预测）、一维CNN预测器（LLR基预测）、AWGN/QPSK链路层仿真。

**📊 数据集**

数据集：使用5G Toolbox在MATLAB生成的链路层仿真数据，覆盖E_b/N_0从-9dB到9dB的样本，LLR向量长度19200，标记为5类（1–4表示需要的RV数，5表示不可解）。极化码实验使用自建的(256,128) CRC-aided polar码与(512,128)对比。

**📈 对比分析**

比较方法：与传统固定冗余（初始传输冗余即为重传冗余）以及传统二进制ACK/NACK机制对比。实验结果表明：SNR查找表在不同SNR下可将重传冗余量降低最多60%；CNN预测器在两轮内成功解码的概率超过96%，吞吐量提升约10–15%，整体延迟缩短至传统两轮HARQ的二倍以内。

**⚠️ 局限性**

限制与不足：仅在AWGN/QPSK环境下验证，未考虑多径/移动性对LLR质量的影响；极化码实验仅在特定长度下进行，无法推广至所有极化码配置；SNR查找表需要针对不同MCS重新生成，训练成本和实时适配存在挑战；预测模型对极端信道估计误差的鲁棒性尚未充分评估。

---

## 60. Token Time Continuous Diffusion for Language Modeling

**arXiv ID:** 2607.14106 | [PDF](https://arxiv.org/pdf/2607.14106v1)

**作者:** Parikshit Bansal `[一作]` (University of Texas at Austin), Sujay Sanghavi `[通讯]` (University of Texas at Austin)

**通讯引用:** 6075 | [OpenAlex ID](https://openalex.org/A5110619770)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于token‑time的连续状态扩散语言模型TTCD，并在低步数高速度生成以及前缀条件生成任务上进行了验证。

**💡 创新点**

创新点是为每个token单独引入时间维度（token‑time），在全局时间之外，显著提升了推理时的鲁棒性并自然支持前缀条件生成。

**🔧 技术方法**

采用连续扩散模型（Diffusion Transformer）+ token‑time 机制 + classifier‑free guidance 等技术。

**📊 数据集**

使用QM9分子数据集和OpenWebText（或类似文本数据集）进行实验。

**📈 对比分析**

与UDLM、MDLM、连续序列时间扩散等基线进行对比；在两步（8×速度）和四步（4×速度）下，TTCD在生成质量、有效分子数量和前缀生成准确率上均优于基线。

**⚠️ 局限性**

实验仅限于约100M参数规模，未扩展到更大模型；未与多token预测的自回归模型做对比；对更高步数或更大数据集的表现尚未评估。

---

## 61. AI Agents Do Not Fail Alone:The Context Fails First

**arXiv ID:** 2607.14275 | [PDF](https://arxiv.org/pdf/2607.14275v1)

**作者:** Fouad Bousetouane `[一作]` `[通讯]` (Proofagent Ai), Fouad Bousetouane (Proofagent Ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了将 AI 代理上下文工程质量量化为七项可测指标，并在 ProofAgent‑Harness 框架中实现了隔离式评估，以此预先预测代理行为可靠性。

**💡 创新点**

创新点在于：①将上下文质量拆解为可单独评估的七项标准（角色清晰、护栏覆盖、指令一致、工具模式、扎根充分、注入硬化、Token 效率）；②在行为评估之外独立测量上下文，避免循环验证；③通过多评审共识实现可审计、可解释的上下文评分；④实证验证上下文指标与行为结果（如幻觉、工具使用、操纵抵抗等）之间的正相关。

**🔧 技术方法**

主要技术：ProofAgent‑Harness 评估平台、Human‑on‑the‑Bridge (HOB) 多评审共识评分、上下文构造与分层（系统指令、工具 schema、检索证据、记忆、政策、注入隔离），以及基于 Pearson 相关系数的验证分析。

**📊 数据集**

使用了三类监管域（客户支持、医疗理赔、法律合同）下的代理实例，模型为 GPT‑5.5 与 Claude Opus 4.8，构造了三层上下文（Poor、Structured、Hardened），共 7,500 轮多轮交互数据。

**📈 对比分析**

对比方法：在固定模型、仅变更上下文的三层设置下，测量行为指标（安全性、幻觉、工具使用、关键错误等）与上下文评分。结果显示：结构化上下文提升 2.34 业务分数、2.40 幻觉抑制、2.79 工具使用；硬化层提升护栏和注入安全，但不一定提升总分。相关系数表明：扎根充分↔幻觉抑制 r=0.63，护栏覆盖↔操纵抵抗 r=0.60，指令一致↔指令遵循 r=0.57 等，验证了预期的因果映射。

**⚠️ 局限性**

局限性：①验证仅覆盖两款前沿 LLM，无法直接推广至其他模型或更小模型；②上下文评分仍需人工评审，难以完全自动化；③token 效率指标未统一标准，可能与上下文长度产生混淆；④未探讨上下文变化对长周期任务（如持续学习）或安全边界之外的攻击（如沙箱逃逸）影响；⑤评估在受控实验环境下，实际部署场景的多样性和动态性仍需进一步研究。

---

## 62. Information-Theoretic Adaptive Cooling for Deterministic MPPI via Entropy Feedback

**arXiv ID:** 2607.14245 | [PDF](https://arxiv.org/pdf/2607.14245v1)

**作者:** Shuqi Wang `[一作]` (Shanghai Jiao Tong University), Xiang Yin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于信息熵的自适应冷却方法（ITAC）用于确定性MPPI控制

**💡 创新点**

通过使用Shannon熵作为在线反馈信号，动态调节温度，既能保证温度收敛为0，又能防止重要性权重过早崩塌

**🔧 技术方法**

采用MPPI框架、重要性采样、熵估计、理论证明与熵阈值门控的温度调度

**📊 数据集**

在非光滑的信号时序逻辑（STL）运动规划任务与简单的到达-避免任务上进行实验

**📈 对比分析**

与固定冷却率MPPI基线比较，ITAC在成功率保持不变的情况下，收敛迭代次数减少约73%，计算时间大幅缩短，成功率提升（特别是在狭窄通道任务上）

**⚠️ 局限性**

对极端受限环境的鲁棒性仍有限，且需要根据目标误差阈值手动设置熵阈值

---

## 63. Implicit Reasoning Steering via Concept Chaining

**arXiv ID:** 2607.14242 | [PDF](https://arxiv.org/pdf/2607.14242v1)

**作者:** Xiao Ye `[一作]` (Arizona State University), Ben Zhou `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何通过隐式推理引导方法，用自然语言生成“连接段落”来在模型继续预训练后改变其多选题答案偏好。

**💡 创新点**

提出了“连接段落”生成器并结合强化学习（GRPO）在保持答案隐蔽性的同时实现有效引导。

**🔧 技术方法**

使用基于生成的连接段落进行一次完整预训练、RL奖励设计（含判别器和一阶概率偏移）以及对比实验中的知识编辑和文本控制基线。

**📊 数据集**

在CommonsenseQA、HellaSwag、OpenBookQA、QuaRTz、StrategyQA的低一致性子集上进行评估，并在BoolQ、ARC-Challenge、WinoGrande上进行跨域验证。

**📈 对比分析**

相较于直接改写、相似问题生成以及两种知识编辑基线，RL生成的连接段落实现了30.0%的Steer Ratio（仅次于最强的直接改写39.7%），同时在泄漏间隙（2.7）、参考漂移（1.1）和可推断性（15.1%）上大幅优于其他方法。

**⚠️ 局限性**

主要限制包括：只在单一预训练轮次内评估；对不同模型规模的泛化尚未系统验证；生成的连接段落仍可能在某些数据集上被人类识别。

---

## 64. Towards Reliable AI-Assisted Analog Design: Template-Constrained LLM Agents for SAR ADC Generation

**arXiv ID:** 2607.14165 | [PDF](https://arxiv.org/pdf/2607.14165v1)

**作者:** Dimple Vijay Kochar `[一作]` (Massachusetts Institute of Technology), Anantha P. Chandrakasan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 80525 | [OpenAlex ID](https://openalex.org/A5084128470)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套多步 LLM 代理框架（Agentic Template‑Constrained LLM Synthesizer），实现从用户规格到可通过 SPICE 验证的 SAR ADC 体系结构、网表、仿真测试脚本的全流程自动化生成。

**💡 创新点**

创新点在于：① 将专家文献知识与检索增强生成（RAG）相结合，构建专家知识基底，显著降低 LLM 假设错误；② 引入模板约束的两路网表生成（自由生成 + 模板选择），在保持高通用性的同时保证物理可行性；③ 设计了多代理协同工作流程（规划、生成、选择、集成、尺寸优化、测试）并在模拟回馈下迭代自我纠错。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑4o、Gemini 3.1）、检索增强生成（RAG）、规则驱动的网表校验引擎、模板选择与细化、仿真驱动的自我调优循环、贝叶斯多目标优化器（BoTorch + qParEGO）以及测试平台脚本生成。

**📊 数据集**

数据集主要为：公开的 ADC 论文与实验数据库（包含技术节点、SNDR、功耗等信息），自建的 SAR ADC 组件模板库（强 arm latch、Miyahara comparator、二进制/分段 capdac），以及用来评估的 8/4/10 位 SAR ADC 合成实验数据。

**📈 对比分析**

方法评估：在 Cadence GPDK‑45nm 芯片上合成 8 位 SAR ADC，得到 ENOB 7.59（目标 7.5）、SINAD 47.5 dB（目标 45 dB）、SFDR 55.6 dB（目标 50 dB）、功耗 6 µW（目标 8 µW）。在 TSMC‑65nm 节点迁移后仍保持相近指标；在 4 位异步、10 位同步等不同规格下也能实现目标或接近目标的性能，证明框架具有良好的跨节点、跨规格泛化能力。

**⚠️ 局限性**

局限性：① 仅验证了基本 SAR ADC 拓扑，尚未证明对更复杂的混合信号系统的可扩展性；② 依赖大量人工标注的模板与专家检索，若模板不全会限制设计空间；③ LLM 的推理与解释能力仍受限，尤其在高位数或非标准拓扑的细化时可能出现错误，需要人工干预；④ 模型训练与验证均基于公开数据，真实工业环境下的专有技术节点、工艺约束仍是挑战。

---

## 65. Volition Elicitation: Operational Semantics for People and Their Machines

**arXiv ID:** 2607.14138 | [PDF](https://arxiv.org/pdf/2607.14138v1)

**作者:** Ehud Shapiro `[一作]` `[通讯]` (London School of Economics and Weizmann Institute of Science), Ehud Shapiro (London School of Economics and Weizmann Institute of Science)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于人类意志的分布式系统抽象规范与实现语言vGLP，并将其编译为GLP实现，生成可直接在智能手机上运行的基层平台（社交图谱、社交网络和加密货币）

**💡 创新点**

将计算过程中的机器转移条件与人的意志绑定，定义了可被人的意志“许可”的事务；同时通过语义导向的界面构造自动生成用户交互界面，实现“意志抽取”与程序执行的统一；在同一应用框架内支持多平台

**🔧 技术方法**

Volition-Guarded Multiagent Atomic Transactions（VGMAT）、Communicating Volitional Agents（CVA）、Grassroots Logic Programs（GLP）及其扩展vGLP；基于多代理事务系统、语义导向UI的实现与编译；使用Dart/Flutter实现可跨平台手机运行

**📊 数据集**

无实验数据集，研究以形式化模型与示例程序（朋友关系、群组聊天、加密货币交易）为依据；演示已在物理智能手机上部署的完整应用

**📈 对比分析**

通过形式化证明（保守性、意志安全性、活性、实现完整性）验证系统的正确性；性能评估以实际手机部署为基础，未给出数值指标，主要关注功能完整性与交互一致性

**⚠️ 局限性**

目前实现为手工编译的方式，缺乏自动编译器；缺少对高层类型安全与意志约束的支持；尚未在多用户同步环境下评估可扩展性与并发冲突解决

---

## 66. ToolAnchor: Anchoring Counterfactual Context to Boost Agentic Tool-use Capability

**arXiv ID:** 2607.14145 | [PDF](https://arxiv.org/pdf/2607.14145v1)

**作者:** Weiting Liu `[一作]` (Fudan University), Wenlian Lu `[通讯]` (Fudan University)

**通讯引用:** 8327 | [OpenAlex ID](https://openalex.org/A5030103251)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ToolAnchor框架，通过教师模型生成的对抗性锚点上下文，引导后续agent在新工具环境下突破行为惯性，实现工具集扩展。

**💡 创新点**

核心创新在于：①把失败轨迹中关键决策点定位为“counterfactual anchor round”，②教师先做假设后学生验证，再将验证通过的锚点以SFT+RL方式内化，避免全轨迹模仿；实现工具扩展而非从零训练。

**🔧 技术方法**

技术包括：多教师模型对失败轨迹审计生成锚点；学生rollout与双层验证（任务级+锚点级）筛选有效上下文；anchor-only agentic SFT+后续agentic RL；LLM-as-judge评估。

**📊 数据集**

使用OpenSeeker‑1.5k（文本+视觉）和VDR‑1.5k做训练，评测基准为GAIA（通用助手）、BrowseComp（文本检索）、VDR‑Bench（视觉检索）。

**📈 对比分析**

与GPT‑5、Gemini‑2.5‑Pro、Claude‑4‑Sonnet、OpenAI DeepResearch、Vision‑DeepResearch等对比，ToolAnchor在GAIA提升至74.5%（+3.6pp），BrowseComp提升至45.0%（+1.6pp），VDR‑Bench提升至28.8%（大幅超过Gemini 18.8%、GPT‑5 20.4%，接近Vision‑DeepResearch 37.8%）。

**⚠️ 局限性**

局限性：教师模型有限，锚点生成受模型偏好影响；验证过滤率约33%仅保留少量上下文；依赖LLM判定，可能受评测准确性影响；扩展到更大或更异构工具集时的通用性待验证。

---

## 67. LBA: Textual Hard-Label Adversarial Attack under Low Query Budgets

**arXiv ID:** 2607.14101 | [PDF](https://arxiv.org/pdf/2607.14101v1)

**作者:** Shixin Guo `[一作]` (Zhejiang Normal University), Hao Peng `[通讯]` (Zhejiang Normal University)

**通讯引用:** 10657 | [OpenAlex ID](https://openalex.org/A5100740618)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在硬标签场景下利用采样分布生成高质量、低查询预算对抗文本的方法。

**💡 创新点**

创新点在于将先验（语义相似度、扰动率等约束）与后验（位置重要性统计）融合，构造近似高质量对抗样本分布，并以此进行Metropolis-Hastings采样，打破传统贪婪搜索导致的搜索空间收缩问题。

**🔧 技术方法**

采用了采样式Metropolis-Hastings算法、软硬标签对比、语义相似度评估（Universal Sentence Encoder）、语法错误检测（LanguageTool）、困惑度评估（GPT‑2）以及后验位置重要性更新机制。

**📊 数据集**

在四个公开数据集（MR、AG、YELP、IMDB）上，针对六种语言模型（WordCNN、WordLSTM、BERT、LLaMA2‑7b‑chat、Vicuna‑7b‑v1.5、GPT‑4o‑mini‑tts‑2025‑03‑20）进行实验。

**📈 对比分析**

与软标签攻击（Textbugger、Textfooler）及四种硬标签基线（HLA、Texthoaxer、LeapAttack、HQA‑Attack）在tiny、tight、moderate三档查询预算下对比，LBA在攻击成功率、扰动率、语义相似度、语法错误率、困惑度等指标均显著优于基线，且在LLM上也保持了高质量对抗文本。

**⚠️ 局限性**

局限性包括：仍需一定查询预算；对长文本的采样效率受限；方法主要基于同义词替换，难以处理更复杂的语言变形；并且在极大模型或极长文本上可能需要进一步优化。

---

## 68. MonteRET: AI Agent Enhancing Multimodal LLMs with Multi-granularity Knowledge Retrieval for Chest CT Report Generation

**arXiv ID:** 2607.14264 | [PDF](https://arxiv.org/pdf/2607.14264v1)

**作者:** Yi Lin `[一作]` (Weill Cornell Medicine), Yifan Peng `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 11442 | [OpenAlex ID](https://openalex.org/A5085113833)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了一个基于多粒度解剖建模与检索增强的胸部CT报告生成框架MonteRET，并在RadGenome-ChestCT和NYP/WCM数据集上进行评估。

**💡 创新点**

创新点包括：① 将全局与区域级视觉特征联合编码；② 通过多模态检索（条件引导+视觉语言对齐）为生成模型提供上下文；③ 采用检索增强的重写代理并结合提示工程提升报告准确性与一致性。

**🔧 技术方法**

使用技术包括ViT3D视觉编码器、LLaMA-2-7B-Chat+LoRA语言模型、信息对比学习的视觉语言对齐、检索重排序、以及多阶段重写策略。

**📊 数据集**

使用的数据集为公开的RadGenome-ChestCT（24,128训练 + 1,564测试）和外部NYP/WCM（82例）胸部CT报告。

**📈 对比分析**

与无检索基线及六种SOTA模型（R2GenGPT、MedVInT、RadFM、CT2Rep、M3D、Reg2RG）对比，MonteRET在BLEU、ROUGE、METEOR、CIDEr、BERTScore以及临床召回率和F1上均取得显著提升，尤其是召回率提升显著，专家评估也显示错误率降低。

**⚠️ 局限性**

局限性包括：仅针对胸部CT；检索库的覆盖度和质量限制了罕见病症的生成；类别不平衡仍影响少数条件的性能；外部验证样本量有限，跨机构泛化仍需进一步验证。

---

## 69. Budgeted Subset Refinement for Execution-Aware LLM Research Ideation

**arXiv ID:** 2607.14118 | [PDF](https://arxiv.org/pdf/2607.14118v1)

**作者:** Micah Zhang `[一作]` `[通讯]` (Independent Researcher), Micah Zhang (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在LLM生成研究创意时，如何通过预算化子集精炼（random‑k、MMR‑k、micro‑low）来提升创意质量与多样性，提出了共享候选池框架并评估不同精炼分配策略。

**💡 创新点**

创新点在于将最大边际相关性多样性选择与精炼预算分配相结合，证明MMR‑k在成本效益与多样性上优于随机子集精炼和统一精炼，首次将精炼视为组合级支持分配问题。

**🔧 技术方法**

采用LLM生成、最大边际相关性（MMR）多样性选择、微低精炼与完整精炼技术，结合token成本计量与句子嵌入相似度去重。

**📊 数据集**

使用10个研究创意环境（包含人机协作、语言模型研究、执行评估等）与10个随机种子，每种生成16个候选，共计1600个候选；外部评审样本72项用于鲁棒性验证。

**📈 对比分析**

相较于原始生成、rerank、统一精炼，MMR‑k在强非重复创意收益23.7、成本/强度4492 token、重复率0.172的同时保持与random‑k相近的总成本，显示出最佳的成本-多样性-质量三方平衡。

**⚠️ 局限性**

主要限制在于评估依赖LLM代理判定和句子嵌入去重，缺乏真实执行验证，token成本与实际API成本不完全对应，且所用的子集选择策略较为简单，未来需测试学习型或人机协同的更高级分配方法。

---

## 70. Stochastic Filtering for Quorum Sensing in Robot Swarms under Anonymous Communication

**arXiv ID:** 2607.14262 | [PDF](https://arxiv.org/pdf/2607.14262v1)

**作者:** Fabio Oddi `[一作]` (Sapienza University of Rome), Vito Trianni `[通讯]` (National Research Council)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在匿名通信条件下，提出并评估了一种基于 k-priority 采样的随机过滤协议，用以提升机器人集群的舆论感知（Quorum Sensing）估计的稳定性和准确性。

**💡 创新点**

创新点在于：1）将 k-priority 采样思想引入匿名消息缓冲区，对低优先级（即即将过期）的消息进行暂时屏蔽，从而降低重复计数导致的偏差；2）通过对比基线 FIFO、随机打乱和过滤三种协议，系统性地分析了信息冗余、准确性、延迟与恢复性能之间的权衡。

**🔧 技术方法**

主要技术包括：匿名消息缓冲管理、随机超时抽样、k-priority 过滤、基于阈值的舆论检测、Kaplan–Meier 与 Weibull 分析恢复时间、以及在 ARGoS 仿真框架下的 Kilobot 运动模型。

**📊 数据集**

使用的“数据集”是仿真产生的：在 ARGoS 框架中，随机生成的 13 个不同真实群体比例（G），以及 3 个不同密度/规模场景（LD25、HD25、HD100），每个场景 100 次仿真运行，采集消息多样性、准确率、延迟与错误恢复等指标。

**📈 对比分析**

比较方法：对三种协议在相同仿真条件下测量（1）消息多样性、（2）舆论检测准确率（Q=0.8/0.2 置信区间）、（3）中位检测延迟、（4）误差率与恢复时间。结果显示：基线最快但误差最大；随机打乱协议提高准确率但延迟更长；过滤协议在降低误差率的同时显著拉长恢复时间，且在高密度场景下表现最佳。

**⚠️ 局限性**

局限性：① 匿名通信无法完全消除重复计数，导致估计上限受限；② k 值与超时参数需经验调优，过大会导致“数据饿死”；③ 仅在仿真环境验证，实际硬件噪声、通信延迟等因素未充分考量；④ 过滤机制虽提升稳定性，但牺牲了对突发事件的快速响应能力。

---

## 71. Asymptotical Analysis of the $(1+(λ,λ))$ GA Escape Time from Local Optima on Jump Functions

**arXiv ID:** 2607.14278 | [PDF](https://arxiv.org/pdf/2607.14278v1)

**作者:** Anton V. Eremeev `[一作]` (Sobolev Institute of Mathematics), Valentin A. Topchii `[通讯]` (Sobolev Institute of Mathematics)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 (1+(λ,λ)) GA 在 Jump_k 基准函数上的跑完时间进行了理论分析，并给出了从局部最优逃逸的更紧凑的上界。

**💡 创新点**

创新点在于：① 采用 De Moivre–Laplace 中心极限定理对二项分布进行精细化处理，得到更准确的概率估计；② 扩展了可取参数范围，给出了更一般的上界；③ 在之前已知上界基础上，改进了常数项，使得理论上限与实验结果更接近。

**🔧 技术方法**

主要技术包括：概率极限定理（中心极限定理、泊松近似）、二项与超几何分布的概率质量函数、斯特林近似、级数展开和极限运算。通过这些技术推导出 p_M、p_C 的渐进表达式，进而得到迭代次数和评估次数的上界。

**📊 数据集**

数据集：主要使用 Jump_k 的合成实例（长度 n，跳跃大小 k）进行实验验证；实验与理论上限做对比。

**📈 对比分析**

对比方法：将本文得到的上界与 Antipov 等人（2022）提出的上界进行对比，并用实验跑完时间验证。实验结果表明本文的上界在常数项上更小，且随 n 增大时与实际跑完时间收敛，显示出更好的预测精度。

**⚠️ 局限性**

局限性：① 分析依赖于 np→∞ 的渐进假设，对固定或非常小的 k 及 np 的情形不适用；② 只给出了上界，未给出下界或完全匹配的解析；③ 计算复杂度未做详细讨论，实验规模有限。

---

## 72. Explainable Geospatial AI for Satellite Ground Station Siting Using LiDAR-Derived Terrain Intelligence

**arXiv ID:** 2607.14127 | [PDF](https://arxiv.org/pdf/2607.14127v1)

**作者:** Shohini Sarkar `[一作]` (University of Maryland), Arsh Goenka `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了如何利用可解释的地理空间机器学习模型预测卫星地面站场址的代表性杂波高度（RCH）。

**💡 创新点**

创新点在于将USGS 3DEP LiDAR获取的精细标签与全球公开遥感数据相结合，使用LightGBM梯度提升树并通过SHAP实现可解释性，显著优于ITU-R P.452的固定高度默认值。

**🔧 技术方法**

采用LightGBM模型，并使用SHAP进行特征归因；特征来源包括DEM、土地覆盖、植被指数、光谱反射率、温度、人口密度等多源遥感产品。

**📊 数据集**

训练数据来自美国USGS 3DEP LiDAR（5万+个100 m网格格点）；特征数据来自Copernicus DEM、ESA WorldCover、Sentinel‑2、WorldPop、GHSL等全球公开产品。

**📈 对比分析**

与ITU‑R P.452默认高度对照，模型MAE从4.67 m降低到1.79 m（下降>60%），R²提升至0.765，且在国别匹配验证中亦优于基线。

**⚠️ 局限性**

主要局限包括仅使用美国LiDAR标签（缺乏真实的国际验证）、对快速土地变化和极端环境的适应性不足、100 m网格细节有限以及未完成完整传播链评估。

---

## 73. Lyapunov Guidance: A Unified Framework for Stabilizing Generative Flows

**arXiv ID:** 2607.14272 | [PDF](https://arxiv.org/pdf/2607.14272v1)

**作者:** Jingdong Zhang `[一作]` (Imperial College London), Junhong Liu `[通讯]` (MicroCyto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种统一的Lyapunov导向框架LyaGuide，用来在流匹配模型的推断阶段通过投影校正保证生成过程的稳定性与收敛性。

**💡 创新点**

创新点在于将流匹配的引导问题理论化为Lyapunov控制问题，并给出了闭式伪投影算子，使得任何候选引导向量场都能满足Lyapunov不等式，从而实现显式的稳定保证；同时框架兼容既有引导方法并可在两种场景（显式先验与弱监督学习）下应用。

**🔧 技术方法**

核心技术包括：流匹配与连续性方程；Lyapunov稳定理论；点态投影（pseudo‑projection）正则化；几种经典引导策略（分类器、奖励、能量模型、图像逆问题）的统一解读。

**📊 数据集**

实验数据集涵盖：低维合成分布（8‑Gaussian、S‑curve）；CelebA‑HQ图像逆问题（填空、去模糊、超分）；D4RL 居中任务（HalfCheetah、Hopper、Walker2d）；以及二维月亮型能量模型。

**📈 对比分析**

在所有实验中，LyaGuide在保持或仅略增推断开销的前提下，均显著提升了样本质量、引导一致性与收敛速度，表现优于原始引导方法且在多任务、多引导策略下保持一致性。

**⚠️ 局限性**

主要局限包括：伪投影仅保证Lyapunov不等式，未完全满足原始的加权散度约束；对Lyapunov函数的依赖使得在先验模糊或学习不充分时可能导致过度收敛或探索不足；以及在高维问题中仍需进一步评估投影对计算成本的真实影响。

---

## 74. Low-Latency Relay Selection in NR-V2X Vehicular Communications via Graph Isomorphism Networks with Edge Features

**arXiv ID:** 2607.14176 | [PDF](https://arxiv.org/pdf/2607.14176v1)

**作者:** Giambattista Amati `[一作]` (Fondazione Ugo Bordoni), Paola Vocca `[通讯]` (University of Rome Tor Vergata)

**通讯引用:** 442 | [OpenAlex ID](https://openalex.org/A5039580089)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对NR‑V2X网络中的多跳中继选择，提出了低时延学习优化框架，利用Graph Isomorphism Network with Edge Features（GINE）通过一次前向传播预测中继链路激活，替代传统的NP‑hard MILP求解。

**💡 创新点**

创新点在于①引入边特征的GINE，显式将链路容量等射频信息嵌入消息传递；②使用MILP作为离线oracle生成监督标签，训练GINE达到接近最优的决策；③设计混合GINE‑Pruned MILP（GP‑MILP）策略，用GINE预先剪枝加速MILP求解，实现实时性与最优性兼备。

**🔧 技术方法**

Graph Isomorphism Network with Edge Features（GINE）、学习‑优化（Learning‑to‑Optimize）框架、离线MILP生成监督、边级二分类头、边感知消息传递、GINE+Pruned MILP。

**📊 数据集**

基于OSM–SUMO–GEMV^2的仿真生成的大规模城市交通数据集，共449,500条NR‑V2X图快照，覆盖1 km²罗马市中心，包含2–4个RSU，车辆随时进出。

**📈 对比分析**

与传统1‑hop MILP基线和离线MILP对比，GINE在边级别与MILP相似度达95.8%准确率、F1≈0.954；在整体连通率上比1‑hop MILP提升9.2%–12%；推理时延≤5 ms；GP‑MILP在30 ms内完成98%实例，保持MILP等价解。

**⚠️ 局限性**

局限性包括：模型未显式强制满足流量守恒与循环消除约束，依赖离线MILP标签；对极端网络规模或连续动态图序列的表现未评估；仅在单台服务器上测试，未验证分布式RSU/MEC环境下的实际部署。

---

## 75. 3D Lane Detection with Odometry for High-Speed Vehicle Racing

**arXiv ID:** 2607.14248 | [PDF](https://arxiv.org/pdf/2607.14248v1)

**作者:** Omoruyi Atekha `[一作]` (Massachusetts Institute of Technology), Marcus Greiff `[通讯]` (Toyota Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多相机和IMU融合的赛道3D车道检测方法，并公开了包含250k图像、IMU与车轮里程计的Thunderhill Raceway数据集。

**💡 创新点**

创新点包括：①使用IMU预积分实现时间与相机间的聚类融合；②引入PCA聚类大幅加速聚类过程；③将BEV视图对齐至道路坐标系并利用贝塞尔曲线进行多相机回归；④首次公开高质量赛道3D车道标注数据集。

**🔧 技术方法**

采用深度学习3D车道检测网络（如BevLaneDet、AnchorLane、LATR等），利用TensorRT半精度量化、PCA聚类、贝塞尔曲线回归、IMU预积分以及总变差正则化等技术。

**📊 数据集**

使用Thunderhill Raceway数据集，包含4个摄像头（20Hz）、IMU（500Hz）和车轮里程计（62.5Hz）共250k张带有3D车道标签的图像。

**📈 对比分析**

在该数据集上对比改造前后的F1、AP、AR等指标，单相机改进后F1从约78提升至约93；多相机回归进一步提升约3点F1；在真实车辆跑道上实现>290Hz推理率，F1>0.9，Y_near MAE<0.18m。

**⚠️ 局限性**

主要限制：仅在单条赛道上测试，缺少相机同步与LiDAR数据；方法依赖IMU，且假设环境静态；跨赛道泛化表现略有下降。

---

## 76. Intelligent Three Level Learning Architecture for Autonomous UAV Swarms in Search and Rescue

**arXiv ID:** 2607.14093 | [PDF](https://arxiv.org/pdf/2607.14093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 77. Capability from Access Structure, Not Scale: Lower Bounds and Pre-Registered Tests for Hybrid Sequence Models

**arXiv ID:** 2607.14144 | [PDF](https://arxiv.org/pdf/2607.14144v1)

**作者:** Wenhui Chen `[一作]` (University of Macau), Chi Man Vong `[通讯]` (University of Macau)

**通讯引用:** 7294 | [OpenAlex ID](https://openalex.org/A5076922237)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Capability Convergence Hypothesis（CCH），分析在固定推理预算下模型的表示收敛与能力收敛的区别，并证明混合架构（兼具压缩状态通道与可索引通道）能突破三个信息壁垒，利用预注册实验验证理论。

**💡 创新点**

创新点在于：① 定义访问完整性原理，揭示表示收敛并不必然带来能力收敛；② 识别并量化 Shannon、horizon 与 circuit 三个“墙”；③ 证明混合架构在长范围检索与序列推理任务上实现超加性能力，并提出产业趋向访问完整类的宏观假设。

**🔧 技术方法**

使用信息论下界、数据处理不等式、S5 组合法、状态空间模型（SSM）、全局注意力、混合注意力-SSM 架构、预注册实验设计、工业规模模型调查等技术。

**📊 数据集**

主要数据集包括：合成检索任务（Newton's apple）、公开语言基准（RULER、MMLU、AlpacaEval‑LC 等）、自建的 S5 组合法推理任务；以及从 140 款工业模型中提取的 KV‑共享系数等指标。

**📈 对比分析**

比较方法：预注册实验验证剪刀差（scissors gap）、分支点（bifurcation）与通道互补性；在长上下文基准上对比访问完整模型与稠密 Transformer；在工业模型中统计全球注意力共享系数并与性能相关联。性能方面，混合模型在检索精度几乎为 0，纯模型接近随机；在长上下文基准上，访问完整模型与稠密 Transformer 性能相当或更优；工业模型显示全球注意力共享系数聚焦在 0.1–0.25 区间。

**⚠️ 局限性**

局限性：① 仅针对固定预算无 chain‑of‑thought 的系统 1 阶段；② Circuit Wall 的理论依赖 TC^0≠NC^1 尚未正式证明；③ 合成任务的自然性与实际语言检索的覆盖率仍未知；④ 对大规模模型的可学习性与可扩展性尚未充分验证；⑤ 结论对自然语言工作负载的普适性仍需进一步实证。

---

## 78. Generalised Reachability Games

**arXiv ID:** 2607.14199 | [PDF](https://arxiv.org/pdf/2607.14199v1)

**作者:** Sougata Bose `[一作]` (UMONS - Université de Mons), Tansholpan Zhanabekova `[通讯]` (University of Liverpool)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5004101656)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究在有限图上两人零和转移式博弈中，多目标到达（generalised reachability）游戏的可解性、记忆需求及其优化变体的复杂性；

**💡 创新点**

关键创新在于将目标集大小作为参数揭示出问题的PSPACE-完整性与FPT性质，并给出两位玩家的最优记忆上界与下界，进一步探讨了最大化访问目标集的优化问题与其在单玩家与双玩家环境下的可解性；

**🔧 技术方法**

主要技术包括归约至TQBF、构造吸引子与偏序、使用SCC分解与动态规划、以及利用辅助记忆结构将一般目标转化为到达目标；

**📊 数据集**

文章未使用任何公开数据集，而是构造理论性的博弈实例来证明下界与上界；

**📈 对比分析**

实验或比较结果未给出，本文主要给出理论复杂度与记忆量的严谨证明；

**⚠️ 局限性**

局限性在于对目标集大小为2的情况仍未得到完整的复杂度分类，且最大化目标集访问的单玩家变体的精确复杂度仍开放。

---

## 79. MultiRef-Compass: Towards Comprehensive Evaluation of Multi-Reference-to-Audio-Video Generation

**arXiv ID:** 2607.14189 | [PDF](https://arxiv.org/pdf/2607.14189v1)

**作者:** Xiaohan Zhang `[一作]` (Nanjing University), Huaxiong Li `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MultiRef-Compass 基准，用于评估多参考到音视频（MR2AV）生成模型，涵盖多视角、跨实体绑定以及人物-物体-场景组合。

**💡 创新点**

创新点在于①构建可扩展、可重复的资产组合管道；②设计四维度（基本质量、参考一致性、音视一致性、指令遵循）的评价协议；③融合自动指标与重新判定增强的 MLLM-as-a-Judge，提升评测一致性与可审计性。

**🔧 技术方法**

技术手段包括资产包策划、板块化资产组合、结构化提示生成、DOVER++/Audiobox 等自动指标、SigLIP、YOLO-World 等特征匹配，以及 Gemini-3.1 Pro 等大型语言模型做判定和再判定。

**📊 数据集**

使用 350 条手工策划的样本，来源于 Pexels、Freesound 等版权免费仓库，覆盖真实与卡通风格，涵盖同一人物多视角、异实体组合、音频/视频参考等多模态配置。

**📈 对比分析**

对 8 款代表性 MR2AV 系统（6 份闭源、2 份开源）进行评测，显示不同模型在四个维度存在明显差距：闭源模型总体更强，尤其在参考一致性与指令遵循上；开源模型在多视角绑定与细节保持方面表现更差。

**⚠️ 局限性**

局限性包括：基准为受控诊断性样本，未覆盖所有创意域与文化风格；自动工具对卡通与大幅面部变形的鲁棒性不足；音视频同步度量对面部稳定性敏感；MLLM 判定可能带来模型偏见；以及随着闭源系统更新可能导致评测结果漂移。

---

## 80. Beyond Object Validation: Relational Conformance in Multi-Artifact Agent Releases

**arXiv ID:** 2607.14155 | [PDF](https://arxiv.org/pdf/2607.14155v1)

**作者:** Tengjiao Liu `[一作]` `[通讯]` (psi.run), Tengjiao Liu (psi.run)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了面向多工件代理发布的关系一致性验证框架SIP‑RC，构建了以图为核心的发布模型，定义了对象与关系的校验规则、权限范围、执行风险与保障传播、非补偿门控以及发布状态机；并通过三例（DRSS、Schema Docs、Brand Shuttle GEO）验证该框架在发现传统单对象校验遗漏的发布错误上的有效性。

**💡 创新点**

创新点在于：① 把发布视为一张关系图而非单一对象，统一校验对象层次和跨工件关系；② 引入双层权限与范围序序，实现决策范围的精细化授权；③ 引入执行风险/保障双序列，区分外部依赖与可重现性；④ 通过“分路径重算”保证生成器与验证器不共享同一决策逻辑；⑤ 设计可扩展的声明式约束语义与错误码体系。

**🔧 技术方法**

使用的技术包括：图模型（V、E、类型映射、版本/哈希）、JSON/SHACL校验、签名与哈希绑定、权限与范围的偏序定义、执行风险/保障的偏序传播、发布状态机与原子发布实现，以及可选的Datalog/DSL约束语言。

**📊 数据集**

数据集主要由三套系统的历史提交、日志与发布包组成：DRSS的历史批次、Schema Docs的文档交互与校验日志、Brand Shuttle GEO的诊断与维修链条；共计约 400+ 本地校验通过实例与 3 条失败案例。

**📈 对比分析**

目前仅进行了案例级的功能验证，未完成系统级对比实验。作者计划与传统单对象校验、通用图校验、内容绑定、共享生成器一致性与运行时控制等基线进行对比，评估指标包括关系缺陷召回、误阻率、逃逸率、验证延迟与成本。实验结果尚未发布，性能表现待未来工作。

**⚠️ 局限性**

局限性包括：① 所有系统来自同一研究项目，存在共同源偏差；② 采用回溯式案例收集，未进行前瞻性注册；③ 仅对三例进行功能验证，缺乏广泛的外部系统评估；④ 依赖手工编写的约束与错误码，可能存在实现不一致；⑤ 对发布状态机与签名等细节的依赖未被外部验证。

---

## 81. ReportMedSAM: Guiding Segmentation Through Radiology Reports

**arXiv ID:** 2607.14116 | [PDF](https://arxiv.org/pdf/2607.14116v1)

**作者:** Anghong Du `[一作]` (University of Birmingham), Le Zhang `[通讯]` (University of Birmingham)

**通讯引用:** 6776 | [OpenAlex ID](https://openalex.org/A5100350651)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出ReportMedSAM框架，实现基于自由文本放射报告的目标自适应医学图像分割。

**💡 创新点**

创新点在于引入可学习的概念库与互斥正交约束，实现对文本的语义路由并可无缝扩展新器官。

**🔧 技术方法**

使用冻结的BiomedCLIP视觉‑语言编码器、对比学习、正交正则化及MoE专家网络进行训练与推理。

**📊 数据集**

在AbdomenAtlas 3.0（肝、胰、肾、脾）数据集上进行实验。

**📈 对比分析**

与BioMedParse、SAM3、IMIS‑Net比较，平均Dice为0.647，在同义词和阈值路由下表现稳定，优于基线。

**⚠️ 局限性**

局限性包括依赖冻结编码器、仅验证于腹部器官、对极罕见同义词或报告异常表达的鲁棒性待进一步评估。

---

## 82. Heterogeneous Element-Aware Cross-Version Differencing of Scientific Documents via Layout-Aware Alignment and Structure-Aware Reasoning

**arXiv ID:** 2607.14117 | [PDF](https://arxiv.org/pdf/2607.14117v1)

**作者:** Zhen Yina `[一作]`, Keran You `[通讯]` (Beijing Renhe Information Technology Co Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种布局感知、元素感知的跨版本科学文档差异检测框架，先将文档分解为文本、表格、公式、图形四种语义元素，再进行跨版本对齐，最后进行结构感知的差异推理。

**💡 创新点**

创新点在于：①统一的元素层级拆解与对齐流程；②结合位置、内容、结构三种兼容度的加权匹配；③对不同元素类型使用专属表示与差异推理；④同时评估检测、定位、结构一致性与对齐质量四个维度。

**🔧 技术方法**

使用YOLOv10m进行布局检测，基于PDF文本提取、表格结构解析、公式LaTeX/MathML表征以及视觉特征提取；对齐采用位置+内容+结构的兼容度评分；差异推理分别针对文本、表格、公式和图形实现细粒度检测与结构一致性判断。

**📊 数据集**

使用来自期刊编辑部生产校对流程的真实PDF版本对，覆盖多学科、多版式、不同校对阶段与多种修改模式的文档对。

**📈 对比分析**

与多类基线（文本Diff、SentenceDiff、TypographyDiff；TableTextDiff、CellDiff、TableStructDiff；LaTeXDiff、SymbolDiff、FormulaTreeDiff；ImageDiff、SSIMDiff、PerceptualDiff、RegionDiff）在检测、定位、结构与对齐等四个维度进行比较，实验显示本方法在所有维度均实现显著提升（检测F1最高0.903，表格结构一致性提升至0.833，公式结构一致性提升至0.877）。

**⚠️ 局限性**

主要局限包括：目前仅支持一对一元素对齐，无法处理段落拆分/合并、表格跨页拆分等多对多对齐场景；对齐与差异推理高度依赖上游表格/公式/图形解析的准确性；图形差异仅评估区域级别，缺乏细粒度组件级别标注；研究仅针对PDF格式，未扩展到Word/LaTeX/HTML等常见出版格式。

---

## 83. Automatic Hard Example Synthesis with Multi-Level Agentic Data Curation

**arXiv ID:** 2607.14256 | [PDF](https://arxiv.org/pdf/2607.14256v1)

**作者:** Genglin Liu `[一作]` (UCLA), Ariel Fuxman `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套全自动、多代理的红队框架，用来生成并验证多模态大型语言模型（MLLM）在内容安全方面的边界案例，从而提升模型的鲁棒性。

**💡 创新点**

创新点在于：①将红队过程拆解为Architect、Operator和多层LLM评审委员会，形成“探索+变异”循环；②采用多级评审与辩论机制，自动确定目标模型错误并归档硬例；③利用生成的硬例进行检索增强的少样本学习，显著降低误判率。

**🔧 技术方法**

技术实现包括：Gemini 3.1 Pro 作为推理与评审核心；Nano Banana Pro 负责图像合成；多级LLM评审委员会（Gemini 3 Flash/Pro）用于冲突检测和归一；检索增强（RAG）+动态少样本提示；实验中还对探索/变异比例做了 ablation。

**📊 数据集**

使用公开的 HoliSafe‑Bench（约1,796张图，918张“硬”子集）以及内部的 Google Ads 安全数据集（3,000+张真实广告图）。

**📈 对比分析**

与零样本 baseline（无提示）对比，加入政策文本后 FNR 由 0.869 降至 0.412；再加上检索式硬例后 FNR 进一步降至 0.245；相对于随机或常规合成数据，硬例检索在所有指标（准确率、召回率、F1）上均优；在内部数据集上亦表现出显著提升。

**⚠️ 局限性**

局限性包括：依赖先进的 LLM 推理能力和合成模型，可能无法完美迁移到更弱的模型；合成图像与真实图像的差距；以及红队策略和奖励机制对探索效果的影响仍需进一步研究。

---

## 84. Simplicity Paradox: Debunking myths about prompting and datasets for LLM evaluation

**arXiv ID:** 2607.14109 | [PDF](https://arxiv.org/pdf/2607.14109v1)

**作者:** Inder Preet `[一作]` (IBM), Dhaval Patel `[通讯]` (IBM)

**通讯引用:** 2689 | [OpenAlex ID](https://openalex.org/A5033934770)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对8种提示技术在10个多选题测评数据集上进行约430,000次评估，比较其在27种模型配置下的表现。

**💡 创新点**

发现简单基线提示往往优于复杂推理提示，仅极少的角色框定略有提升；参数量不是性能的好预测；在推理预算上，开启推理比增大预算更有效；数据集仍有巨大提升空间。

**🔧 技术方法**

采用对齐的ReasonLab框架实现8种提示技术；使用Elo排名、配对t检验、token成本记录等方法；模型包括GPT‑5、Claude、Gemini、Llama、Qwen、Mistral等。

**📊 数据集**

使用10个MCQA数据集：FailureSensorIQ、CURE‑Bench、CorrectBench、PhysicsQA、AIME 2025 QA、MedQA、MMLU‑Pro、OpenBookQA、SuperGPQA、Time‑MQA。

**📈 对比分析**

采用匹配与原始比较两种评估，配对统计检验显示基线提示在10×10匹配单元中仅落后于两种极少提示；整体上基线占据第三位；Elo排行榜显示Qwen3 30B领先，推理预算开启带来最高收益。

**⚠️ 局限性**

局限：匹配比较仅覆盖不具推理开关的10种模型配置，未评估复杂提示对具推理模型的影响；技术实现未做针对模型的微调；只关注MCQA，无法推广到开放式生成或代理任务；使用单次采样且温度0可能掩盖不确定性。

---

## 85. Certified Domain Consistency for Multi-Domain Retrieval: Label-Free Per-Domain Contamination Control with Conformal Risk Guarantees

**arXiv ID:** 2607.14157 | [PDF](https://arxiv.org/pdf/2607.14157v1)

**作者:** Jayakumar Manoharan `[一作]` (Electric Power Research Institute), Jayakumar Manoharan `[通讯]` (Electric Power Research Institute)

**通讯引用:** 500 | [OpenAlex ID](https://openalex.org/A5006232574)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了C3R控制层，实现无查询标签下的多域检索交叉污染预算认证。

**💡 创新点**

首次将分布式风险控制与隐式域标签结合，给出每个域的可解释预算上限并可在冻结检索堆栈上直接部署。

**🔧 技术方法**

双拆分置信度风险控制、Clopper–Pearson误报上界、Hoeffding‑Bentkus置信集合以及软失配阈值去除技术。

**📊 数据集**

BEIR‑MIX（四域混合）、Sector‑Bench（美国联邦法规）、Mix‑D、Mix‑C等公开多域集合。

**📈 对比分析**

与边际控制、硬阈过滤、无证书检索等对比，C3R在所有域实现零证书违规，在高污染域保留2–6倍召回且污染率控制在预算内。

**⚠️ 局限性**

对极低标签查询域需更强域探测器；仅对文档来源污染做证，未直接判定合法性或合规性。

---

## 86. Eta Given Delta: Defining LLM Tool Efficiency With Marginal Tool Utility

**arXiv ID:** 2607.14108 | [PDF](https://arxiv.org/pdf/2607.14108v1)

**作者:** Nyx Iskandar `[一作]` `[通讯]`, Nyx Iskandar

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了工具效率（tool efficiency）和边际工具效用（marginal tool utility）两个量化指标，用于直接评估LLM Agent在执行任务时有用工具调用的比例，并通过LLM-as-a-Judge判定每次工具调用的正负效用；随后在APEX‑SWE Observability基准上对不同工具组合（包括Grafana/Loki、Mattermost、Plane）进行消融实验，验证这些指标与任务准确率的一致性。

**💡 创新点**

创新点在于：①首次定义并量化“工具效率”与“边际工具效用”；②提出使用LLM作为判定器仅判断效用符号的高效方法；③将这些指标与传统准确率对比，展示在去除无效工具时工具效率上升、准确率保持或下降，证明指标的实用性。

**🔧 技术方法**

技术方法包括：ReAct式Agent轨迹建模；用GPT‑5.4（Judge）计算每次工具调用的Δ_α符号；聚合求和得出工具总体效用；计算工具效率即有用调用数除以总调用数；实验中使用的Agent基模型为GPT‑5.3‑Codex和Gemini 3.1 Pro，评估工具调用日志与最终代码补丁的单元测试通过率。

**📊 数据集**

数据集：APEX‑SWE Observability公开子集（25个可观测性任务），每个任务可运行多次以获得统计稳定性；使用三种工具组合共3个版本，结合两种Agent模型共150条轨迹。

**📈 对比分析**

比较方法：对同一任务分别使用完整工具集、仅Grafana/Loki、无MCP工具三种版本进行准确率、工具效用与工具效率评估；结果显示：删除Mattermost和Plane时准确率无显著变化但工具效率提升；删除Grafana/Loki时准确率显著下降。不同模型在准确率上相近，但工具效率存在差异，表明工具效率可为模型定制化提供指导。

**⚠️ 局限性**

局限性：①只对只读工具进行了消融，未评估写工具的边际效用；②LLM-as-a-Judge的置信度不均衡，判定结果对非正效用更自信；③实验仅在APEX‑SWE公开子集，缺乏对更大范围任务的验证；④工具效率随工具套件变动而波动，需进一步研究其稳定性与对策略学习的直接影响。

---

## 87. Align AI to Dynamic Human-AI Workflows

**arXiv ID:** 2607.14240 | [PDF](https://arxiv.org/pdf/2607.14240v1)

**作者:** Valerie Chen `[一作]` (Carnegie Mellon University), Aarti Singh `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2691 | [OpenAlex ID](https://openalex.org/A5100758181)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本篇论文通过对一场跨学科工作坊的整理与分析，提出了从静态、模拟人类输出的对齐方法向动态、互补性对齐方法的转变，并系统梳理了人机协作中的信任、互补专长与协调等社会科学理论，识别了实现动态对齐所面临的概念、转化与评估障碍，并给出了相应的研究方向与建议。

**💡 创新点**

创新点在于：①提出将对齐目标从单次输出转向随时间演进的交互轨迹；②将人类团队合作研究中的信任演化、共享记忆与心理模型等概念引入人机协作对齐框架；③对现有多智能体强化学习、偏好学习与 RLHF 等技术进行跨学科重新诠释，指出其在真实工作流程中的不足；④系统化列举了实现动态对齐所需的理论、数据与评估基准，并给出未来研究路线。

**🔧 技术方法**

论文主要采用理论建模、文献综述与工作坊讨论相结合的方法，对现有的多智能体强化学习、偏好学习、RLHF 等技术进行概念性梳理；未进行具体的算法实现或实验。

**📊 数据集**

未直接使用任何数据集；文中仅提及现有的聊天机器人日志、GitHub 问题跟踪、代码助手交互轨迹等数据来源，作为未来评估的潜在素材。

**📈 对比分析**

论文并未进行实验对比，也未给出性能指标；其核心贡献是概念与框架上的阐述，而非算法实现与数值评估。

**⚠️ 局限性**

主要局限包括：①缺乏实证验证，无法证明提出的轨迹级对齐方法在实际工作流程中的有效性；②当前评估标准与数据集不足，难以客观测量信任演化、互补性与协调效果；③跨学科整合仍处于起步阶段，方法论与理论体系尚未成熟；④对安全性与伦理风险的讨论相对表面，需进一步深入。

---

## 88. Privacy Leakage in Federated Learning in Radiology Reports: A Comparative Evaluation of Tokenizer-Driven Privacy Risks

**arXiv ID:** 2607.14205 | [PDF](https://arxiv.org/pdf/2607.14205v1)

**作者:** Santhosh Parampottupadam `[一作]` (German Cancer Research Center), Ralf Floca `[通讯]` (German Cancer Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在联邦学习环境下评估了不同分词器对放射科报告梯度反演泄露风险的影响。

**💡 创新点**

首次将分词器设计作为单一可控变量，系统比较 GPT‑2、RadBERT 与 LLaMA‑2 的泄露程度，揭示领域特定分词器会提高隐私风险。

**🔧 技术方法**

采用了分析式梯度反演攻击，插入线性探针实现闭式逆推，并使用 BLEU、ROUGE 等指标评估重建质量。

**📊 数据集**

使用公开的 Dischargesum、MIMIC‑CXR 以及诊断报告三大放射文本语料库，总计约 370K 报告。

**📈 对比分析**

对六个客户端、三种分词器、三种批量大小的组合做五次独立实验，结果显示在批量 64 时 Exact 句子重建率最高可达 44%，批量增大至 256 时下降至 34%；RadBERT 在所有指标上均表现最好。

**⚠️ 局限性**

研究仅关注无防御的极端攻击场景，未评估差分隐私或安全聚合等防御措施，且仅使用短句子窗口和 IID 数据划分，未覆盖更长文本和非 IID 实际部署情况。

---

## 89. TEDDY: A Pediatric Foundation Model for Risk Forewarning from ICD-Coded Diagnostic Histories

**arXiv ID:** 2607.14191 | [PDF](https://arxiv.org/pdf/2607.14191v1)

**作者:** Matthew Brady Neeley `[一作]` (Baylor College of Medicine), Hyun-Hwan Jeong `[通讯]` (Baylor College of Medicine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并训练了TEDDY，一款用于儿童电子健康记录的生成式Transformer模型，能够在每次就诊时预测首次出现的诊断及下次就诊时间。

**💡 创新点**

创新点在于采用访问边界无泄漏、仅评估首次诊断、性别和年龄匹配的严格评价框架，以及在单机构小样本数据上实现对罕见疾病也表现优异的生成模型。

**🔧 技术方法**

使用GPT‑2解码器Transformer，联合预测下一个诊断码（交叉熵）和时间间隔（指数似然）进行端到端训练。

**📊 数据集**

使用德克萨斯儿童医院（TCH）约160万名儿童的ICD‑10诊断序列，约7300万条记录。

**📈 对比分析**

与DenseNet、CNN、RNN、LSTM四种相同数据下的序列基线以及4B参数Gemma LLM对比，TEDDY在797个首发诊断任务中的中位AUC为0.72，显著优于所有基线；在哮喘、ADHD等常见疾病上的AUROC分别达到0.79和0.85，并在罕见疾病上亦高于随机。

**⚠️ 局限性**

局限包括仅使用诊断代码、单机构数据、缺乏多模态特征、未验证跨机构泛化以及时间预测采用单指数模型导致长尾失准。

---

## 90. A Temporal Machine Learning-Based Time-to-Event Model for Predicting ALS Progression and Healthcare Utilization

**arXiv ID:** 2607.14190 | [PDF](https://arxiv.org/pdf/2607.14190v1)

**作者:** Zongliang Yue `[一作]` (Auburn University), Huanmei Wu `[通讯]` (Temple University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究开发了一套基于数字孪生理念的时间到事件模型，融合ALSFRS‑R纵向轨迹、功能域分层与Cox风险分析，实现对ALS患者功能衰退和助行器使用的个体化预测。

**💡 创新点**

创新点在于：①通过相关性聚类从项目级数据自下而上构建功能域；②采用Transformer‑基离散时间生存模型实现多域阶段转移预测；③结合Cox模型实时更新助行器使用风险，形成可解释的数字孪生决策支持系统。

**🔧 技术方法**

使用的技术包括：相关性聚类与Ward链接、广义加性混合模型(GAMM)、注意力机制的Transformer编码器、跨域注意力聚合、五头危险率输出、Cox比例风险模型以及Elastic Net正则化。

**📊 数据集**

数据集为ALS自然史联合体（ALS Natural History Consortium）提供的诊断记录、ALSFRS‑R评估、ADL及人口学信息，共1,409例（187位患者）经质量筛选后构建的整合纵向数据集。

**📈 对比分析**

通过C‑index、NLL、MAE等指标评估，五域阶段预测模型在测试集中的C‑index分别为0.544（Bulbar）至0.598（Lower limb），C‑index平均0.549；在Cox模型中，步行与爬楼梯功能是预测助行器使用的最强指标，相关系数显著；总体预测性能显著优于传统基线。

**⚠️ 局限性**

局限性包括：观测性数据可能存在选择偏倚与未测混杂；Cox模型假设比例风险可能不足以捕捉时间变异效应；未纳入影像、基因或可穿戴等多模态数据；外部验证不足，需进一步验证模型的泛化能力。

---

## 91. ReasFlow: Assisting Reasoning-Centric Scientific Discovery in Applied Mathematics via a Knowledge-Based Multi-Agent System

**arXiv ID:** 2607.14178 | [PDF](https://arxiv.org/pdf/2607.14178v1)

**作者:** Yutong He `[一作]` (Peking University), Pingwen Zhang `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ReasFlow，一套端到端自主 AI 代理系统，用于支持应用数学的理论推导、算法设计、实验验证和论文撰写；在该系统中完成了五项理论创新算法并生成完整研究论文。

**💡 创新点**

核心创新在于：①引入内部验证循环和知识卡机制，使 AI 能在无人工实时干预下完成严谨证明；②构建专门面向应用数学的知识库，实现程序化检索与自动推理；③把完整研究流程拆分为多任务代理，并通过人机协同高效生成可发表论文。

**🔧 技术方法**

技术包括：大型语言模型驱动的多代理框架；知识卡（knowledge cards）与检索机制；内部逻辑一致性与紧密度验证器；自动化实验与可视化工具；LaTeX 写作与评估循环；以及与开源工具（FAISS、Semantic Scholar API、Python、Matplotlib 等）集成。

**📊 数据集**

实验主要基于公开的联邦学习和大语言模型预训练任务，使用常见数据集（如 MNIST、CIFAR‑10、ImageNet 及行业标准联邦学习基准）以及自定义实验数据集；文献检索则使用 arXiv、Semantic Scholar 等公开文献库。

**📈 对比分析**

与现有开源 AI 科研代理（如 AI Scientist、DeepScientist 等）在相同起始提示下进行对比；采用 LLM 评审器和多维度评价表，对理论深度、证明严谨性、实验结果等进行打分。ReasFlow 在所有维度获得最高分，理论收敛率、通信复杂度等指标均优于前人方法。

**⚠️ 局限性**

局限性包括：仍需人工主导的初始任务定义和最终审核；知识卡的构建与检索受限于现有文献质量；内部验证循环在极大规模证明时计算开销较高；系统主要针对理论驱动的应用数学，难以直接迁移到更具经验驱动的科研领域。

---

## 92. How Much of a 10-K Matters? Aggregation-Dependent Value of Full-Text versus Risk-Factor Sentiment

**arXiv ID:** 2607.14174 | [PDF](https://arxiv.org/pdf/2607.14174v1)

**作者:** Sanggyu Sean Choi `[一作]` `[通讯]` (University of Edinburgh), Sanggyu Sean Choi (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种监督式词典学习方法，用于从10-K文件及其Item 1A风险因素章节中提取情绪分数，并将情绪与回报和波动率标签相关联；

**💡 创新点**

创新点在于将监督式情绪学习扩展到监管文件文本，首次将波动率作为监督目标，并在行业、投资组合与单一公司三层级进行对比；

**🔧 技术方法**

采用基于词袋的监督词典学习模型，利用混合多项分布估计情绪词分布，再通过惩罚式对数似然最大化求取情绪分数；

**📊 数据集**

使用1,383份2006–2023年间94家纳斯达克100科技公司提交的10-K文件（包含完整文本与Item 1A章节），并计算相应回报与波动率标签；

**📈 对比分析**

通过分类准确率、与实际价格的Pearson相关性以及词义主题分析进行比较。结果显示，完整文本在行业与组合层级下情绪预测更准确，而Item 1A在单公司层级表现更佳；Loughran–McDonald词典基线始终与价格呈负相关；

**⚠️ 局限性**

局限包括：仅使用单词级词袋表示，无法捕获多词短语；行业与组合层级聚合未按权重加权；主题解释依赖人工标注，缺乏自动化验证；

---

## 93. ToolAlignBench: Investigating Alignment Conflicts in Tool-Calling Enabled LLMs

**arXiv ID:** 2607.14285 | [PDF](https://arxiv.org/pdf/2607.14285v1)

**作者:** Aryan Keluskar `[一作]` (Arizona State University), Huan Liu `[通讯]` (Arizona State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究工具调用 LLM 在监管行业中，安全训练与部署指令冲突时的行为，构建 ToolAlignBench 基准并评估12种模型的指令覆盖表现。

**💡 创新点**

首次公开提供可测量安全对齐模型在价值层级冲突下指令覆盖行为的基准，揭示安全训练可能导致模型主动吹哨或数据外泄，并证明不同训练方法对该行为影响差异显著。

**🔧 技术方法**

使用工具调用框架与规则抽取解析模型输出；引入 Abliteration 技术去除安全训练方向以检验其对行为的作用；利用多工具分类评估模型决策。

**📊 数据集**

构造128个合成场景（16个行业×4个违规+4个安全），共64个违规与64个安全案例，所有文本由 Claude Sonnet 4.5 自动生成，避免真实数据泄露与名称偏见。

**📈 对比分析**

对12个模型（4款专有、4款安全对齐开源、4款 Abliterated 开源）进行5轮推理，评估误差率、外部/内部通报率与任务完成率；安全对齐模型外部通报率最高达43.4%，Abliteration 可显著降低（如 Mistral‑24B 外部通报率从27.5%降至0.3%）。

**⚠️ 局限性**

使用合成数据可能与真实环境分布不匹配；模型在安全场景中仍出现高误报（过度怀疑）；基准仅涵盖内部文档处理，未覆盖更广泛的部署情景。

---

## 94. ICAConfPubs: A Dataset and User Interface for ICA Conference Papers (2003-2018)

**arXiv ID:** 2607.14234 | [PDF](https://arxiv.org/pdf/2607.14234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 95. Local Additive Feature Attribution: A Mathematical Taxonomy and Reporting Checklist

**arXiv ID:** 2607.14271 | [PDF](https://arxiv.org/pdf/2607.14271v1)

**作者:** Rebecca Afriyie Sarpong `[一作]`, Daniel Commey `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对本地可加特征归因方法进行系统综述，提出统一的数学框架和分类，梳理各方法的核心假设、满足的公理与潜在失效模式，并给出一套十项报告清单；

**💡 创新点**

创新点在于：① 将Shapley、路径积分、梯度/反向传播、扰动和CAM系列方法统一在五个显式规范（价值函数、基准、路径、扰动分布、守恒规则）下；② 构建完整的公理‑方法矩阵，阐明方法间差异的本质；③ 提出针对本地可加归因的可复现性报告清单；④ 通过理论与实证将常见失败模式与模型假设直接关联。

**🔧 技术方法**

采用了游戏理论（Shapley值）、路径积分（Integrated Gradients）、梯度与链式法则（Grad‑CI、DeepLIFT、LRP）、扰动采样（LIME、RISE）、CAM（Grad‑CAM、HiResCAM）等多种归因技术，并用公理化分析、等价归约与复杂度对比等技术进行系统评估。

**📊 数据集**

本文聚焦方法本身，主要引用公开基准（ImageNet、COCO、UCI 等）和自研数据集进行实验验证，并在评估章节中讨论了多种公开数据集的使用；但整体为综述性工作，未提出新的专属数据集。

**📈 对比分析**

通过在同一数学框架下对方法进行对比，本文展示了不同方法在完整性、敏感性、实现不变性、对称性等公理满足情况、计算复杂度以及对常见失效模式的鲁棒性。实验结果表明：完整性满足的方法在解释稳定性上更优；实现不变性强的梯度方法在不同网络架构间保持一致；而基于路径积分的IG在对抗扰动和离群样本上表现更好。

**⚠️ 局限性**

局限性包括：① 综述覆盖范围仍不完全，某些新兴方法或跨领域应用尚未深入；② 评估多聚焦在理论与公理上，缺少统一的“真值”基准；③ 报告清单仍为建议性，缺乏实证验证其对复现性的提升效果；④ 由于方法假设差异巨大，直接性能比较仍需依赖具体任务与数据集。

---

## 96. MEMORA: Embodied Action Memory from Egocentric Videos for Reasoning and Planning

**arXiv ID:** 2607.14252 | [PDF](https://arxiv.org/pdf/2607.14252v1)

**作者:** Zihao Yu `[一作]` (Washington University in St. Louis), Chongjie Zhang `[通讯]` (Washington University in St. Louis)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MEMORA 框架，将第一人称视频经验转换为可编辑、可合并的多类型记忆，并利用该记忆驱动机器人规划；同时构建了 MEMORA‑Bench 评估套件。

**💡 创新点**

创新点在于：① 将记忆生命周期（形成–巩固–检索）与四类存储（环境、实体、活动、推理）相结合；② 通过在线编辑保持实体历史并通过离线巩固抽象经验；③ 通过 typed‑memory retrieval 为规划提供结构化的程序性与物理 grounding 信息。

**🔧 技术方法**

技术包括：多模态段编码与实体编辑（使用 Qwen 等开源 LLM）、ReAct 循环与 typed‑memory 查询、离线聚合算法抽象常规与偏好、以及基于文本/图的检索接口。

**📊 数据集**

使用 EPIC‑KITCHENS‑100 扩展视频 45 小时（18 位参与者）构建 MEMORA‑Bench，涵盖 EAM‑QA（记忆评估）和 Replay/Generalize 规划任务。

**📈 对比分析**

对比 7 种记忆条件（无记忆、文本、实体图、仅 Episodic、完整 MEMORA 等），在四个开源 LLM 上评估。记忆评估提升 20.5 分，Generalize 规划 RGP 相对提升 16.6%，Replay 也有显著提升，显示 MEMORA 在跨事件推理和长时规划上优于基线。

**⚠️ 局限性**

局限性包括：仅在厨房场景与有限时间跨度内验证；对感知质量敏感；未实现完整闭环机器人控制；未处理长期遗忘与隐私/安全问题。

---

## 97. The Steering Budget: Examples beat Knobs

**arXiv ID:** 2607.14246 | [PDF](https://arxiv.org/pdf/2607.14246v1)

**作者:** Raj Kumar Rajendran `[一作]` `[通讯]`, Raj Kumar Rajendran

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于训练数据的“预算”框架，用于量化并区分两种常见的生成模型控制方式：通过调节提示或控制器（称为“telling”）和通过给模型一组示例（称为“showing”）来进行控制。

**💡 创新点**

创新点在于将控制效果拆分为“within-bin”与“between-bin”两部分，并证明它们对应的可达范围（预算）可以在模型训练前从数据统计中直接计算得到，从而预测哪些目标只能通过示例控制、哪些可以仅靠提示即可实现。

**🔧 技术方法**

核心技术包括：基于标签（cheap read）划分输出为离散箱子，计算每箱子中目标属性的均值、方差和占比，利用方差分解得到两项预算；利用卡方距离和 Cauchy–Schwarz 上界推导出 telling 与 showing 的最大可移动幅度；以及在示例选择上采用最小卡方或池混合策略。

**📊 数据集**

在 ImageNet（使用 DiT‑XL/2 生成图片）和 Alexandria（基于计算材料数据库生成晶体结构）两个独立领域分别验证，分别对亮度、美学分数、类别内容分数以及晶体的能带隙、能量高度等属性进行评估。

**📈 对比分析**

通过对比不同控制方式的实际平均值移动、覆盖率等指标，实验表明：在大多数目标属性上，showing 能比任何提示或控制器实现更大幅度的目标偏移；对于“平均”型目标，单一箱子集中即可；而“多样性/组合”型目标只能通过示例混合实现；当目标属性与训练数据高度一致时，预算预测与模型表现高度吻合。

**⚠️ 局限性**

局限性包括：预算仅适用于模型能够准确再现的属性；对离散箱子划分的敏感性（过细或过粗会影响预算）；示例分布若包含训练分布之外的样本，预算失效；以及在极端 fine‑tune 或生成任务极大偏离训练分布时，需重新审计模型自身输出。

---

## 98. Integration Matters: Rollout-Based Training for Constrained Diffusion Models

**arXiv ID:** 2607.14398 | [PDF](https://arxiv.org/pdf/2607.14398v1)

**作者:** Xiaoxuan Liang `[一作]` (University of British Columbia), Frank Wood `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于逆向扩散轨迹（rollout）的微调框架，利用终点约束损失和自适应尺度网络，在训练阶段直接对采样过程中的约束违规进行优化。

**💡 创新点**

创新点在于：①沿逆向轨迹对终点样本做约束损失，消除训练与采样不匹配；②引入可学习的尺度网络替代固定guidance，动态调节约束修正强度；③使用LoRA保持预训练权重，兼顾分布保真与约束满足；④提供理论证明训练目标与采样分布一致。

**🔧 技术方法**

采用EDM扩散模型、可微逆向扩散、约束梯度引导、可学习尺度网络、LoRA适配、梯度检查点和分布正则化等技术。

**📊 数据集**

实验数据集：合成弹跳球数据（100k场景）和Interaction交通场景（车辆轨迹）。

**📈 对比分析**

与EDM、MPGD、MBM/MBM++、PIDM、DPOK、Adjoint Matching等基线比较。弹跳球任务中约束违规率降至0.01%，生成质量与EDM相近；交通轨迹任务中离路/碰撞率降至0.3%，轨迹匹配与物理可行性保持在EDM水平，优于所有对比方法。

**⚠️ 局限性**

局限性包括：在每一步仍需计算约束梯度，导致推理开销增加；目前仅在低维运动/轨迹任务验证，尚未在高维视觉或更大规模多模态场景中测试其可扩展性。

---

## 99. Why Git Is the Memory Solution for the Agentic Development Lifecycle

**arXiv ID:** 2607.14390 | [PDF](https://arxiv.org/pdf/2607.14390v1)

**作者:** Frank Guo `[一作]` `[通讯]`, Frank Guo

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一套基于 Git 的记忆系统，利用提交历史和 AI 助手会话的日志，形成 append‑only ledger，并设计了三种记忆模式（结构化地图、回放式片段检索、决策合成）以及一套路由器，根据问题类型将查询分配给最合适的模式。

**💡 创新点**

创新点包括：① 将 Git 本身作为记忆底层，自动提供真实性、时效性、验证性和边界控制；② 将记忆拆分为三种互补模式并通过路由实现最优答案；③ 通过提交‑会话链接实现自标注基准；④ 在 token 预算内显著提升答案完整度，三阶量级节省查询成本。

**🔧 技术方法**

使用技术：后置提交钩子捕获助手会话并清洗数据；append‑only ledger 与 Git 只存 SHA；混合检索模型（BM25 + LSA + 神经余弦）与 facet 术语；置信门控控制片段注入；LLM 生成结构化子系统地图；聚合决策相关回合后合成完整推理路径；路由器为内部技能实现。

**📊 数据集**

数据集：8 个匿名化仓库，其中包括 ~4000 Markdown 文档的知识库、约 50k LOC 的生产系统以及 6 个较小的代码/文档/构建脚本混合仓库；自标注的 gold 对来自提交‑会话关联，涵盖 415+ 对等。

**📈 对比分析**

对比方法：与原始 grep、BM25、混合检索及多种引入的重排序机制进行对标，使用 MRR 评估检索性能；答案完整度（answer‑sufficiency）通过单独盲评判量化，结果显示路由系统在两大语料库上总体达 0.43/0.60（A/B），token 费用仅 382–980/382–980，较全历史查询节省约 3 阶量级。

**⚠️ 局限性**

局限性：评估样本仅 15 题/语料，单一盲评判；记忆完整性受限于代理人是否口头化 reasoning，导致部分决策无法回溯；结构化地图质量依赖于仓库自有文档；模型泛化到其他行业数据集需进一步验证。

---

## 100. Supervised Fine-Tuning vs. In-Context Learning: An Equilibrium Analysis of LLM Personalization under Congestion

**arXiv ID:** 2607.14371 | [PDF](https://arxiv.org/pdf/2607.14371v1)

**作者:** Fengzhuo Zhang `[一作]` (Yale University), Dirk Bergemann `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在共享计算资源下用户如何在两种LLM个性化方法——监督微调（SFT）与上下文学习（ICL）之间进行取舍，并建立了一个连续玩家均衡模型来分析拥堵与定价的影响。

**💡 创新点**

创新点包括：① 将LLM个性化的统计误差精确解析为预训练覆盖度与信噪比的函数；② 证明拥堵水平在均衡中存在且唯一，且其随预训练参数呈非单调变化；③ 证明平台在提供SFT与ICL两种方法时利润不会下降，解释了多平台服务组合的快速普及。

**🔧 技术方法**

采用线性化的统计模型对SFT与ICL误差进行解析推导，利用均衡理论（mean‑field game/拥堵博弈）分析用户策略与平台定价；同时使用凸优化与定价最优化技术。

**📊 数据集**

实验部分使用了22M参数的GPT‑2模型，在线性回归任务上进行ICL与SFT的对比。

**📈 对比分析**

方法比较：在低信噪比或覆盖不足时ICL误差更低；当样本量足够大且信噪比高时SFT误差更低。实验验证了理论预测，ICL误差随样本数递减后趋于不可消除的偏差，SFT误差可随样本数趋近零。

**⚠️ 局限性**

局限性：① 仅考虑单一平台与两种个性化算法，未覆盖多平台竞争与更多高级推理模型；② 线性近似可能无法完全捕捉真实LLM的非线性行为；③ 假设所有用户类型连续且资源成本已知，实际中可能存在不完全信息与动态变化。

---

## 101. Unsafe at any AUC: Unlearned Lessons from Sociotechnical Disasters for Responsible AI

**arXiv ID:** 2607.14353 | [PDF](https://arxiv.org/pdf/2607.14353v1)

**作者:** Joshua A. Kroll `[一作]`, Abigail Z. Jacobs `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统安全视角对历史社会技术灾难进行案例分析，提炼出一系列未被充分吸取的经验教训，并讨论如何将这些教训迁移至AI系统的安全治理中，构建AI安全的系统性框架。

**💡 创新点**

创新点在于：①将传统的系统安全方法（如STAMP、系统理论事故模型）引入AI安全领域；②提出将AI视为完整的社会技术系统，而非单纯的技术组件；③提出一套“未学习的教训”框架，为AI治理提供跨领域的理论支撑。

**🔧 技术方法**

主要采用的技术手段为系统安全分析方法（系统理论、风险因子分类、组织文化与治理模型）以及跨学科的文献综述与案例对照；未涉及具体算法实现或实验平台。

**📊 数据集**

论文未使用任何实验数据集，而是基于已有的灾难案例（切尔诺贝利、挑战者号、福岛、Bhopal等）和相关安全研究文献进行归纳。

**📈 对比分析**

由于缺乏实验数据，本文未给出传统意义上的性能对比；通过案例对照说明现有AI技术（如模型性能指标、对齐、稳健性等）在系统层面不足，无法单凭组件指标评估安全性。

**⚠️ 局限性**

限制与不足：①方法高度理论化，缺乏在真实AI系统中的实证验证；②未提供可操作的评估指标或实验设计；③对技术干预与系统安全方法的整合仍需后续研究，实际落地的可行性与成本未作评估。

---

## 102. Capacity of Uniform Noise Channels Under Average Input Power Constraints

**arXiv ID:** 2607.14352 | [PDF](https://arxiv.org/pdf/2607.14352v1)

**作者:** Yihan Zhang `[一作]` `[通讯]` (University of Bristol), Yihan Zhang (University of Bristol)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了加性均匀噪声信道在平均功率约束下的容量，并给出了唯一的容量实现输入分布与输出分布的显式形式。

**💡 创新点**

创新点在于利用输出密度的周期化恒等式，将容量最大化问题转化为一个可显式求解的熵最大化问题，从而得到精确的容量表达式和对应的输入输出分布，首次完整刻画了非高斯噪声下的容量实现方案。

**🔧 技术方法**

核心技术包括：傅里叶分析/Poisson 级数重排、周期化恒等式、KL 散度与熵最大化、解析函数性质、以及对相关特殊函数 Q(λ) 的单调性与极限分析。

**📊 数据集**

该工作为理论分析，未使用任何实验数据集或仿真数据，全部为解析推导。

**📈 对比分析**

与现有的通用上界/下界（如高斯上界、对称对数凸噪声下的 0.254 比特上界）相比，所得到的容量值与高斯信道非常接近，且提供了完全可实现的输入/输出分布；但论文未给出数值仿真或与具体实现方案的对比。

**⚠️ 局限性**

局限性：仅适用于均匀分布噪声和平均功率约束；未讨论其他噪声分布的推广；未给出实现该输入分布的实际编码方案；对于 λ 的数值求解仍需数值方法；并未探讨更复杂信道（如多维、多级或带约束）中的推广。

---

## 103. Traccia: An OpenTelemetry-Based Governance Platform for AI Systems

**arXiv ID:** 2607.14309 | [PDF](https://arxiv.org/pdf/2607.14309v1)

**作者:** Nutan Kumar Naik `[一作]` (National Institute of Technology Rourkela), Abhishek Patel `[通讯]` (Algen.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一套基于OpenTelemetry的多层级AI治理框架Traccia，实现对LLM及自适应AI代理的实时可观测、守护栏验证及合规证据生成。

**💡 创新点**

将原始运行时追踪转化为结构化、可验证的治理证据，并在运行时动态检测守护栏缺失、生成可供审计的加密证书包。

**🔧 技术方法**

OpenTelemetry、OTLP、W3C Trace Context、SHA-256加密哈希、Python SDK、自动猴子补丁、成本计算引擎、规则引擎。

**📊 数据集**

主要以示例性高风险AI贷款审批系统为实验场景；未公开使用大型公开数据集。

**📈 对比分析**

与Arize、Langfuse、LangSmith、Datadog LLM Observability、Credo AI等平台对比，Traccia在实时守护栏验证、法规映射、成本追踪等维度表现优于或补足了现有工具，性能保持在可接受的延迟与吞吐量范围内。

**⚠️ 局限性**

尚未实现主动运行时强制执行、仅支持欧盟AI法合规、缺乏对HIPAA、SOC2等其他框架支持；评价、提示工程等功能仍待后续版本完善。

---

## 104. DCVC-MB: Neural B-Frame Video Compression using State Space Models

**arXiv ID:** 2607.14305 | [PDF](https://arxiv.org/pdf/2607.14305v1)

**作者:** Arjun Arora `[一作]` (Dolby Laboratories), Sean McCarthy `[通讯]` (Dolby Laboratories)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于状态空间模型的神经网络 B‑frame 视频压缩框架 DCVC‑MB，集成 Mamba 变换器融合、可适应的潜在跳过以及规范化位置嵌入，支持高分辨率与低显存消耗。

**💡 创新点**

创新点包括：
- 使用线性 O(N) 的 Mamba 模型替代 O(N²) 的 Transformer，显著降低显存与计算复杂度；
- 规范化（canonical）位置嵌入，让编码器对不同分辨率具有更好泛化能力；
- 可适应的潜在跳过机制，在推理时动态抑制冗余潜在特征，提升压缩率；
- 结合 YUV‑to‑RGB 的全范围 BT.709 变换，实现 RGB 训练后 YUV 输出的对照实验。

**🔧 技术方法**

技术栈包括：
- Mamba 变换器（轻量级自注意力模型）；
- 2D 位置嵌入（规范化与标准 sinusoid 对比）；
- 码率控制与自适应潜在跳过；
- 基于 PyTorch 的 flash‑attention；
- YUV420 与 BT.709 转换实现。

**📊 数据集**

使用的数据集主要有：
- Vimeo‑90k（训练集）；
- BT.709 HEVC 数据集（测试）以及 MCL‑JCV、videoSRC20 等多域数据；
- 通过将 RGB 输出转换回 YUV420 进行与 VTM‑19.0、AlphaVC 等基线的对比。

**📈 对比分析**

与基线 DCVC‑DC、DCVC‑FM、AlphaVC 等方法对比，DCVC‑MB 在 BT.709 HEVC 上平均 BD‑Rate 下降至约 2–7%（根据分辨率和 GOP 设定），在 YUV420 下仍保持显著优势；
在 RD 曲线与 PSNR‑BPP 评估中，B‑frame 模型保持更高 PSNR 与更低 BPP；
显存占用在 1080p 下仅为 13.8 GB，远低于 Transformer 的 40 GB。

**⚠️ 局限性**

主要局限：
- YUV420 直接训练难度大，导致训练难以平衡三个通道的失真；
- 对高运动或与训练数据分布差异较大的域（如 MCL‑JCV videoSRC20）B‑frame 模型表现不佳；
- 适应性跳过在推理时若误用会导致性能下降，说明模型对跳过的偏置需要进一步调优。

---

## 105. Better Privacy Guarantees for Larger Groups

**arXiv ID:** 2607.14406 | [PDF](https://arxiv.org/pdf/2607.14406v1)

**作者:** JacK Fitzsimons `[一作]`, JacK Fitzsimons `[通讯]` (Oblivious)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出并证明了一种针对固定离散组计数的差分隐私机制，能够在大计数组中允许更大的误差，同时保证隐私预算随计数递减；并给出逆二次（O(n⁻²)）的隐私预算上界和下界，证明此速率是最优的。

**💡 创新点**

创新点在于：①首次在零计数处修正严格相对误差要求，构造了可行的“零计数容忍”条件；②引入“计数依赖的组级zCDP”定义，并给出可实现的shifted-log-Gaussian机制；③通过多计数信息论下界，证明逆二次速率是不可突破的，并将常数范围收窄到π/(4e²)≤C≤1/π，距实际常数仅相差不到三倍。

**🔧 技术方法**

主要技术包括：计数空间对数变换+高斯噪声叠加、固定漂移-σ²、剪裁与随机舍入、等方差log-正态转换、Rényi-Divergence 与 zCDP 关系、信息论互信息与熵的下界推导、两计数与多计数实验、量化逼近与高斯极大熵原理。

**📊 数据集**

无实验数据集，本文完全基于理论分析与数学证明。

**📈 对比分析**

通过理论证明比较：上界给出v(n)=Θ(n⁻²)的隐私预算；下界展示任意满足相同正计数误差约束的机制，其预算至少为Ω(n⁻²)。常数范围已被推至约0.106至0.318之间；相较于以往仅得到O(n⁻²)或常数级不确定的结果，显著提升了性能估计。

**⚠️ 局限性**

局限性：①尚未得到精确的常数C*（实际最优常数仍未知）；②机制采用对数变换后再加高斯噪声，可能在实际实现中需考虑数值稳定性；③对零计数的容忍修正仅为理论修正，实际应用时可能需额外处理；④该结果仅适用于固定离散且不重叠的组；⑤没有给出高概率误差或其他误差度量的分析。

---

## 106. CatalogAgent: A Supervisor-mediated Self-Learning System Enabling Context Engineering for GenAI Models

**arXiv ID:** 2607.14396 | [PDF](https://arxiv.org/pdf/2607.14396v1)

**作者:** Zhu Cheng `[一作]` (Amazon), Tarik Arici `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个以Supervisor Agent为中心的自学习系统CatalogAgent，用于电商商品目录属性填充。

**💡 创新点**

创新点在于利用内部Generator–Evaluator争议与外部卖家反馈生成监督式学习循环，通过Memory Base和Summarizer实现无人工干预的持续上下文工程。

**🔧 技术方法**

采用LLM（Generator/ Evaluator为Mistral-NeMo，Supervisor为Claude Sonnet），ReAct多步推理，工具调用以及结构化内存和回归约束的自学习框架。

**📊 数据集**

使用约4.89百万条商品-属性对的内部争议样本，5万条卖家反馈样本以及基准黄金集进行实验。

**📈 对比分析**

在自学习后，Generator、Evaluator分别在特定属性上提升15.24%和13.98%准确率，且通过回归测试保证总体性能不低于基线。

**⚠️ 局限性**

局限在于依赖LLM推理质量、工具集成的可扩展性以及对极端稀有属性的覆盖不足。

---

## 107. An offline approach to fNIRS-guided reinforcement learning for robot behavior

**arXiv ID:** 2607.14393 | [PDF](https://arxiv.org/pdf/2607.14393v1)

**作者:** Julia Santaniello `[一作]` (Tufts University), Jivko Sinapov `[通讯]` (Tufts University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

本文提出并实现了一个离线基于fNIRS信号的强化学习框架NEURO‑LOOP，用来指导机器人在Fetch Pick‑and‑Place任务中的策略学习。

**💡 创新点**

创新点包括：1）首次将离线fNIRS脑信号嵌入RL循环，避免实时BCI的硬件和校准需求；2）对比多种增强方式（优先级、Q‑值、奖励）并系统评估其对学习效果的影响；3）探究不同性能标签粒度（二元、三元、连续）和交互任务（被动观察、主动操作）对学习的作用；4）通过噪声注入实验验证框架对噪声的鲁棒性。

**🔧 技术方法**

技术手段：fNIRS信号预处理与MLP分类/回归性能模型；将模型输出映射为信号增强值；DDPG‑HER算法结合优先经验回放；离线数据注入与训练检查点；噪声模拟与评价指标。

**📊 数据集**

使用公开的fNIRS+机器人轨迹数据集（21名参与者，Fetch Pick‑and‑Place环境），每条时间步包含状态、动作、奖励及对应的脑信号和动作最优性标签。

**📈 对比分析**

与基线（无脑信号增强）对比，Q‑增和All（Q‑增+优先级+奖励）在成功率和累计回报上表现最佳；优先级提升有限；奖励增益不显著。噪声实验显示Q‑增对噪声敏感，奖励增益鲁棒。早期（训练开始时）注入离线数据对学习最有利。

**⚠️ 局限性**

限制：1）性能标签仅基于代理最优性，未直接捕捉用户真实偏好；2）四秒窗口导致脑信号与交互时间存在延迟；3）未与其他HITL方法（如显式评估、表情反馈等）直接对比；4）仅在离线模拟环境验证，尚未证明在线或物理机器人上的可行性。

---

## 108. Unified Uncertainty Quantification Framework Bridging Noisy Quantum Backends Across Variational Quantum Algorithms and Quantum Signal Processing

**arXiv ID:** 2607.14392 | [PDF](https://arxiv.org/pdf/2607.14392v1)

**作者:** Priyabrata Senapati `[一作]` (Pacific Northwest National Laboratory), Bo Peng `[通讯]` (Pacific Northwest National Laboratory)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套统一的不确定性量化（UQ）框架，用来在相同的统计流程下对变分量子算法（VQA）与基于 QSVT 的 Green’s 函数重构这两类工作负载进行应用层基准测试，并对四个 IBM 模拟后端的噪声响应进行系统记录与后期分析。

**💡 创新点**

创新点包括：① 将 VQA 与 QSVT 两类工作负载归纳为相同的参数-输出元组（ξ, y）接口，实现跨算法的统一基准；② 在贝叶斯优化循环中加入后验细化（MCMC/VI），得到稳健参数区域而非单点最优；③ 通过全局灵敏度分析、密度估计、后端排名与噪声关联等多维度指标，提供更丰富、更可靠的后端性能评估；④ 在同一实验平台下同时评估变分与非变分任务，揭示后端对不同工作负载的选择性。

**🔧 技术方法**

使用的技术包括：高斯过程回归与贝叶斯优化（BO）驱动的闭环搜索；MCMC/变分推断对后验进行细化；Morris/Sobol 等灵敏度分析；核密度估计与可视化鲁棒区域；量子线路编译、路由与资源计数（深度、两量子门数）等；QSP/QSVT 阶段的块编码、相位合成、Hadamard 测试以及傅里叶变换重构 Green’s 函数；VQA 的十种 ansatz 及对应算子。

**📊 数据集**

实验数据集包括：① IBM 四个 fake 后端（Brisbane、Kawasaki、Kyoto、Osaka）的噪声模型；② 十个 VQA benchmark family（VQE、QAOA、VQC、VQTE、VQLS、VQAPDE、VQAMET、VQEC、VQPTNU、VQCFE）；③ H₂ 分子 Hamiltonian（4 个体系 qubit + 5 个辅助 qubit）用于 QSVT Green’s 函数重构。

**📈 对比分析**

比较方法：针对每个后端对每个工作负载执行贝叶斯优化+后验细化，记录整个评估历史；随后计算 VQA 的质量规范化、平均排名、评估次数；QSVT 通过 hit‑rate、time‑to‑good 等指标衡量谱重构可靠性；对两类任务统一进行敏感度分布、鲁棒区域体积与路由资源（深度、两量子门数、开销因子）比较。性能结果显示：后端在不同任务中表现差异显著，例如 Brisbane 在 QSVT 中表现最佳、Osaka 在 VQA 质量上居首；不同后端在资源成本与鲁棒区域尺寸上也有显著区别，说明后端排名是工作负载依赖的。

**⚠️ 局限性**

局限性：① 仅使用 IBM fake 后端模拟，未验证对真实硬件的适用性，缺乏日常漂移与队列延迟等因素；② 噪声模型为静态，未覆盖实时校准变化；③ QSVT 仅在 H₂ 系统上测试，规模有限；④ 资源估计基于直接块编码，未使用更高效的稀疏或分层编码；⑤ 评价指标虽然多样，但仍未涵盖所有硬件级细节（如可拓扑约束、读出误差分布等）。

---

## 109. Chat2Scenic: An Iterative RAG-Based Framework for Scenario Generation in Autonomous Driving

**arXiv ID:** 2607.14387 | [PDF](https://arxiv.org/pdf/2607.14387v1)

**作者:** Yuan Gao `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Chat2Scenic，一种通过交互式聊天机器人结合检索增强技术（RAG）实现的迭代组件生成框架，能够从法规描述自动生成可在 CARLA 模拟器中编译运行的 Scenic 脚本。

**💡 创新点**

创新点包括：①迭代式组件生成流程，避免一次性生成完整脚本导致的语法冲突；②将多种提示技术（上下文提示、链式推理、少量示例、检索增强示例）融合成四层 Prompt 体系；③构建了 123 条基于 NHTSA、UN 车辆法规和 CARLA Leaderboard 的真实法规场景基准；④首次提出编译成功率（CSR）和框架准确率（FA）等完整评估指标，并开源实现。

**🔧 技术方法**

使用的技术有：大型语言模型（LLM）配合 Prompting（CP、CoT、ICL、RAG-ICL）；LangChain 与 LangGraph 进行检索与状态管理；双检索器（代码片段检索 + 文档检索）使用 Sentence-Transformer 与 BM25；Gradio 搭建聊天 UI；Scenic DSL 与 CARLA 0.9.15 进行编译验证；嵌入模型和 RRF 合并检索结果。

**📊 数据集**

数据集包括 123 条法规场景描述（CARLA Leaderboard 24、NHTSA Crash 16、NHTSA PreCrash 31、UN R152 4、UN R157 12、UN R171 36），以及从官方 Scenic 代码库和法规文档中提取的代码片段与文档块。

**📈 对比分析**

方法对比：与两种 SOTA（ChatScene Retrieval Assemble、NL2Scenic Retrieval Generation）以及多种 LLM（Gemini‑3 Flash/Pro、Qwen Flash/Plus、DeepSeek、以及多种开源 20B‑30B 模型）进行对比。Chat2Scenic 在 Gemini‑3 Flash 上取得 CSR 76.42% 与 FA 58.17%，显著优于 ChatScene（CSR 30.08% / FA 11.03%）和 NL2Scenic（CSR 16.26% / FA 10.86%）。开源模型普遍 CSR 接近 0%。

**⚠️ 局限性**

局限性：生成时间较长（≈222 秒），对大规模 LLM 依赖强；开源模型性能差，难以实现高 CSR；文档检索加入未提升效果，反而增加开销；当前仅支持文本输入，缺乏多模态或实时仿真反馈；在极端复杂场景下仍可能出现语义不一致或编译错误。

---

## 110. CIPHER: A Decoupled Exploration-Selection Framework for Test-Time Scaling of Data Science Agents

**arXiv ID:** 2607.14386 | [PDF](https://arxiv.org/pdf/2607.14386v1)

**作者:** Maxime Heuillet `[一作]` (Amazon Web Services), Sharadind Peddiraju `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

未知

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未知

**⚠️ 局限性**

未知

---

## 111. A Noise-Robust Elicit-to-Optimize Framework for Distortion Riskmetrics via Inverse Reinforcement Learning

**arXiv ID:** 2607.14373 | [PDF](https://arxiv.org/pdf/2607.14373v1)

**作者:** Yang Liu `[一作]`, Yunran Wei `[通讯]` (Carleton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个噪声鲁棒的提问-优化框架，结合逆强化学习（IRL）与强化学习（RL），从代理的模糊决策中挖掘其对偏差风险度量（distortion riskmetrics）的真实偏好并据此优化策略。

**💡 创新点**

创新点在于①利用可区分的有限提问集合实现对广泛非凸、非一致性风险度量的辨识；②在IRL中加入噪声鲁棒的贝叶斯更新，证明了指数级收敛速率；③在RL端将条件偏差风险度量转化为积分量，并通过量化网络与Riemann求和实现统一的风险优化。

**🔧 技术方法**

技术主要包括：适应性贝叶斯IRL（带探索率），量化网络（量化回归与单调性正则化），Proximal Policy Optimization（PPO）框架的扩展，蒙特卡洛仿真和大样本的经验回放。

**📊 数据集**

实验使用S&P 500历史每日收盘价（2005-2026年）构建交易环境，并采集100支股票用于构造代理选择集。

**📈 对比分析**

与传统基于均值-方差或一致性风险度量的RL算法对比，所提方法在识别代理风险偏好和后续风险敏感策略上表现更快收敛、误差更小；在不同市场情形下，各风险度量对应的交易策略显示出合理的买卖边界，说明优化效果显著。

**⚠️ 局限性**

局限性包括：①需预先定义有限的候选风险度量集合，若真实偏好不在其中则只能近似；②对参数（学习率、探索率、贝叶斯先验）敏感，调参成本较高；③实验集中于金融投资场景，缺乏跨领域验证。

---

## 112. Dysco: Dynamic Subspace Boosting to Mitigate LoRA Interference in Federated Learning

**arXiv ID:** 2607.14367 | [PDF](https://arxiv.org/pdf/2607.14367v1)

**作者:** Haobo Zhang `[一作]` (University of Michigan), Jiayu Zhou `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在联邦学习中研究LoRA适配器因异构客户端导致的数据-参数干扰，提出Dysco动态子空间分配方法；

**💡 创新点**

创新点是将LoRA聚合视为子空间分配问题，并通过局部激活不敏感子空间选择、服务器端闭式子空间合并及多轮子空间提升，实现动态、联邦的子空间分配，显著降低跨客户端干扰；

**🔧 技术方法**

主要技术包括低秩适配器LoRA、激活子空间最小化（SVD/特征矩阵最小化）、不敏感子空间计算、子空间合并最大化重叠、闭式解、增量子空间堆叠和多轮子空间提升；

**📊 数据集**

使用的数据集有控制合成联邦任务、MIMIC-IV临床笔记分类（基于Llama-3.2-1B）以及GLUE基准；

**📈 对比分析**

与FedAvg、FedAvgM、FedProx、Scaffold、FedNova、FFA-LoRA、FedSA-LoRA等基线比较，在合成任务上训练损失下降约9倍；在MIMIC-IV上平均准确率提升0.5%至4.3%；在GLUE上损失提前收敛；

**⚠️ 局限性**

局限性包括仅适用于相同模型架构、需要同步聚合、未考虑模型异构，且对异步环境不适用。

---

## 113. When Is Delegated Play Truthful? Within-Range Regret and the Trilemma of Aligned Delegation

**arXiv ID:** 2607.14357 | [PDF](https://arxiv.org/pdf/2607.14357v1)

**作者:** Taksch Dube `[一作]` `[通讯]` (Kent State University), Taksch Dube (Kent State University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究代理委托机制中“诚实自述”是否最佳，提出用代理的范围内遗憾（within‑range regret）衡量主导者对代理的欺骗动机；

**💡 创新点**

创新点在于给出诚实报告等价于代理忠诚的判定定理，统一自动竞价与语言模型代理等场景，并揭示了绑定、诚实与保能三元困境；

**🔧 技术方法**

主要技术包括对代理范围内遗憾的理论分析、证明其 #P 难度、采样估计、版本化与漂移下的统计认证；

**📊 数据集**

实验使用五大语言模型（GPT‑4o、Claude‑Sonnet 5、Gemini‑2.5、DeepSeek‑V4、Grok 4.3）在一价拍卖环境下采样代理投标；

**📈 对比分析**

与未加约束模型对比，实验显示所有模型在软截断下都存在报告膨胀，范围内遗憾平均约 5.8，验证理论预测且性能与理论一致；

**⚠️ 局限性**

局限性在于遗憾估计依赖采样且受模型漂移影响，且实验仅在单一拍卖设置验证，尚未覆盖更复杂机制与更广泛场景。

---

## 114. Copy-on-Write Scoring: Application-Specific Agent Evaluations

**arXiv ID:** 2607.14336 | [PDF](https://arxiv.org/pdf/2607.14336v1)

**作者:** Joanna Roy `[一作]` (trail-ml), Sven Hoelzel `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了基于 PostgreSQL 的 Copy-on-Write (CoW) Scoring 框架，用于在真实应用环境中评估 LLM 代理的写操作，生成会话级和操作级评分。

**💡 创新点**

创新点在于：①将 CoW 机制与数据库视图、触发器结合，实现代理写入的隔离与监控；②设计会话级结构与内容评分以及操作级效用评分，直接映射到应用的真实工作流；③通过低成本、无漂移的评估循环，快速定位代理在工具表面上的失败模式并进行迭代改进。

**🔧 技术方法**

技术包括 PostgreSQL 的 CoW 表、视图与触发器实现、Python 评分库、OpenAPI 集成、LLM 与工具调用、模拟工作流生成与评分算法。

**📊 数据集**

数据集为 Plane（开源项目管理平台）中的20条真实工作流，GT 会话由人类与 Claude Opus 4.6 生成并人工验证；评估使用5个大型语言模型（GPT-5、GPT-4.1、Gemini‑3.1‑Pro、Gemini‑3.1‑Flash‑Lite、Gemini‑2.5‑Pro）共计300个会话。

**📈 对比分析**

比较方法是将代理会话与 GT 会话在同一 CoW 环境下的最终数据库状态进行结构匹配（matched/missing/extra）和内容相似度评估；结果显示模型间顺序与公开基准一致，模型改进后整体分数提升显著（如 Gemini‑2.5‑Pro 54% 提升）。

**⚠️ 局限性**

局限性包括：①GT 会话手工生成，成本高且覆盖有限；②评估仅关注写操作，忽略读操作与多样化最终状态；③对提示的完整性假设过高，真实用户的歧义提示可能导致误判；④工具表面改动后仍需人工复核，未实现完全自动化。

---

## 115. SD-MAR: Multi-image Analytical Reasoning via Synthetic Data and Reinforcement Learning

**arXiv ID:** 2607.14333 | [PDF](https://arxiv.org/pdf/2607.14333v1)

**作者:** Shiyu Yuan `[一作]` (Stevens Institute of Technology), Huzefa Rangwala `[通讯]` (Amazon Web Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SD-MAR 框架，通过可控合成数据训练 VLM 的多图像分析推理，并针对该任务设计了 GRPO-lite 强化学习策略。

**💡 创新点**

创新点包括① 通过结构化变换生成语义和数值推理任务，实现多图像差异分析；② 在 GRPO 基础上去除 KL 正则与优势归一化，加入反向折扣奖励（BDA），专为多图像推理优化。

**🔧 技术方法**

采用合成数据生成技术（Claude 生成文本、Nova-Pro 生成图像、CLIP 筛选），GRPO-lite 强化学习框架以及 BDA 奖励分配，并用 LLM-as-judge 对推理质量进行评估。

**📊 数据集**

使用自研的 SD-MAR-TSE、SD-MAR-Math、SD-MAR-Mix 数据集进行训练，基准评估包括 MMBench、MME、MMMU-Pro、MathVista 等公开评测集。

**📈 对比分析**

通过在 SD-MAR 测试集、MMBench、MME、MMMU-Pro、MathVista 等任务上比较，Qwen2.5‑VL‑7B 在 SD-MAR 上提升约 30%，在 MMBench 上提升约 1.5%，在 MathVista 上提升 3.4%，并保持其他基准性能±1%；与 GPT‑4.1 在 SD-MAR 上略低但推理质量相当。

**⚠️ 局限性**

局限性包括对合成场景的依赖，真实多图像分布差异可能影响迁移；BDA 在不同模型上的效果不一致；仅在两款开源 VLM 上验证，对更大模型的通用性尚未探测。

---

## 116. PReM: Learning What to Preserve and When to Refresh for Context Compression

**arXiv ID:** 2607.14327 | [PDF](https://arxiv.org/pdf/2607.14327v1)

**作者:** Bohan Yu `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PReM（Preserve and Refresh Memory）框架，在生成过程中动态维护并刷新长上下文的 KV 内存，以实现高效长上下文推理。

**💡 创新点**

创新点包括：①使用专门的记忆层与特殊记忆标记 ⟨m⟩ 进行动态记忆选择与刷新；②采用阶段分离的刷新训练（Phase-Separated Refresh Training）使记忆选择与生成过程同步；③通过块级记忆选择与平均池化实现可控的压缩与刷新。

**🔧 技术方法**

主要技术：基于 LLM 层级 KV 内存的块级压缩；记忆层投影与块评分；top‑k 记忆选择与平均池化；专门的 ⟨m⟩ 刷新触发；记忆选择损失、边界损失和语言建模损失的组合训练；实验中使用 32K 上下文、chunk size 100、top‑k 10/20 等参数。

**📊 数据集**

使用了六个 QA 数据集：TriviaQA、SQuAD、NaturalQuestions、2WikiMQA、HotpotQA、MuSiQue；在每个数据集上均采样 500 条测试/验证样本进行评估。

**📈 对比分析**

与 KV‑cache 压缩方法（SnapKV、StreamLLM、CAKE 等）以及文本空间压缩方法（LongLLMLingua、LLMLingua‑2‑large、EXIT）和软上下文压缩基线（Activation Beacon、ICAE）进行对比。实验结果显示，PReM 在 16×、32× 压缩比下平均 EM/F1 均超过所有基线 5.3/5.15（16×）和 10.23/12.55（32×）点；在多跳 QA 上提升显著；甚至在 32× 压缩下也优于直接 32K 全上下文提示；3B PReM 超越 7B 软压缩基线。

**⚠️ 局限性**

局限性：①需要对特定 LLM 进行大规模训练和微调，适配性依赖模型架构；②块大小与 top‑k 参数对性能影响较大，需手动调优；③对极大压缩率（>32×）或极长上下文的鲁棒性尚未充分验证；④实现复杂度相对传统 KV‑cache 或文本压缩方法更高。

---

## 117. Logical Foundations of Two-Sided Type Theory

**arXiv ID:** 2607.14325 | [PDF](https://arxiv.org/pdf/2607.14325v1)

**作者:** Celia Mengyue Li `[一作]` (University of Bristol), Steven Ramsay `[通讯]` (University of Bristol)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并形式化了两侧类型系统的逻辑基础，并将其与双边逻辑和强否定对应，进一步扩展到高阶类型系统。

**💡 创新点**

创新点在于将两侧类型系统与Wansing双边逻辑以及Nelson强否定结合，并证明一致性、强规范化和存在性、双存在性等性质，同时给出表达完整性证明。

**🔧 技术方法**

主要技术包括命题-类型范式、双侧推理规则、强否定的引入、与Geuvers高阶系统的译码映射以及归约与类型转换的规范化证明。

**📊 数据集**

本文为理论研究，无使用实验数据集。

**📈 对比分析**

通过形式化证明与等价性定理来比较，不涉及实验性能评估，结果证明系统满足一致性和规范化等理论性质。

**⚠️ 局限性**

局限在于实现细节缺失、对实际编程语言的应用尚未探讨，以及对复杂性与可扩展性的进一步分析仍待补充。

---

## 118. Towards a Unified Multidimensional Explainability Metric: Evaluating Trustworthiness in AI Models

**arXiv ID:** 2607.14315 | [PDF](https://arxiv.org/pdf/2607.14315v1)

**作者:** Georgios Makridis `[一作]` (University of Piraeus), Jonh Soldatos `[通讯]` (Innov-Acts)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种统一的多维可解释性评分框架，并在Iris、Wine、Breast Cancer Wisconsin三种数据集上对LIME和SHAP两种常用XAI方法进行基准实验，构建可供后续使用的离线知识库。

**💡 创新点**

创新点在于将忠实度、简洁度、稳定性等多维可解释性属性结合成可加权的统一得分，并通过元数据驱动的知识库实现对未见模型与数据集的可解释性估计，填补了现有框架缺乏客观、可迁移评价标准的空白。

**🔧 技术方法**

采用了数学形式化的可解释性度量（如Fidelity、Stability、Simplicity、Coverage）以及加权求和或多维展示的评分方法，并结合离线知识库技术进行指标归档与检索。

**📊 数据集**

使用了Iris、Wine和Breast Cancer Wisconsin三个经典分类数据集，分别包含多类别和二分类任务，特征维度从4到30不等，以评估XAI方法在不同数据复杂度下的表现。

**📈 对比分析**

通过计算LIME和SHAP在三组数据集上的Fidelity、Simplicity和Stability得分，发现SHAP在忠实度上远优于LIME，但LIME在稳定性上更好，整体得分显示两者在简洁度上相近；实验结果说明可解释性评价需结合数据与模型上下文进行权衡。

**⚠️ 局限性**

局限性包括：评估指标仍需人工赋权，缺乏用户主观满意度和信任度实验；仅验证了两种XAI方法和三组小规模数据集，未覆盖更大规模或不同任务；计算Fidelity和一致性等指标在大模型或高维数据时成本高，需进一步优化算法。

---

## 119. NeuroGRIP: Retrieval-Augmented Graph Refinement for Knowledge-Grounded EEG Seizure Diagnosis

**arXiv ID:** 2607.14314 | [PDF](https://arxiv.org/pdf/2607.14314v1)

**作者:** Lincan Li `[一作]` (Florida State University), Yushun Dong `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了 NeuroGRIP，一种检索增强的图细化框架，通过大型语言模型构建医学知识图，利用语义对齐检索为 STGNN 预测的 EEG 动态脑网络图加权、剪枝，从而提升癫痫发作检测性能并增强可解释性。

**💡 创新点**

创新点包括：① 结合 LLM 自动抽取实体与关系，构建专业医学知识图；② 设计 Semantic Alignment Query，将 k‑hop EEG 子图映射到知识图语义空间，实现语义检索；③ 多因素评分（匹配存在、语义相似度、来源可靠性）用于边置信度评估与自适应剪枝，形成可解释的临床校正脑网络。

**🔧 技术方法**

采用的技术包括：空间‑时序图神经网络（STGNN）作为基线；GPT‑4o 等 LLM 进行实体/关系抽取并生成知识图；FAISS 近似最近邻检索；知识驱动的边置信度计算与阈值剪枝；GraphRAG 结构化检索生成框架。

**📊 数据集**

使用的数据集为公开的癫痫 EEG 数据集：Temple University Hospital EEG Seizure Corpus（TUSZ）和 CHB‑MIT Scalp EEG Database。

**📈 对比分析**

在 12s 与 60s 片段上对比 Dist‑DCRNN、Corr‑DCRNN、NeuroGNN、GraphS4mer、EvoBrain 等 STGNN 基线，指标为 Accuracy、F1 与 AUROC。NeuroGRIP 在所有基线上均提升约 2%–5%（尤其在 EvoBrain 上最显著），同时显著降低图密度，提升诊断解释性。

**⚠️ 局限性**

局限性包括：对外部知识图的质量与完整性高度依赖；LLM 在知识抽取时可能产生噪声或偏差；检索阈值和置信度设定需经验性调优；目前仅验证于 EEG 失作检测，未在其他临床任务中测试；临床可解释性评估仍主要基于结构可视化，缺乏深入专家评估。

---

## 120. Tracing LLM Behavior to the Training Data with Empirical Next-Token Distributions

**arXiv ID:** 2607.14306 | [PDF](https://arxiv.org/pdf/2607.14306v1)

**作者:** Zachary Izzo `[一作]` `[通讯]` (NEC Labs America), Zachary Izzo (NEC Labs America)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型（LLM）输出分布与训练数据的经验下一个标记分布（ENTD）的一致性，分析在不同模型规模、训练阶段、架构和随机性下的偏差来源。

**💡 创新点**

首次系统性地将LLM的预测与ENTD进行对比，揭示出强位置不变性、软最大瓶颈并非主要原因、以及高熵样本的分布收敛差异；同时提出“数据中心机制解释”的视角。

**🔧 技术方法**

使用Pythia系列Transformer模型、∞-gram计数器、软最大低秩近似、PolyPythias多次训练副本、以及离散的TV距离度量来评估偏差。

**📊 数据集**

在Pile语料库（Pythia训练集）上进行实验，构建分层采样评估集以覆盖不同出现频率和长度。

**📈 对比分析**

通过比较模型与ENTD及ENTD‑T的TV分布、低秩近似误差以及不同模型间的相互偏差，发现模型平均与ENTD的偏差随规模和训练算力提升而减小，但仍存在长尾差异；对比低秩近似显示Transformer未充分利用低秩结构。

**⚠️ 局限性**

仅适用于训练集内部序列；未考察未见过的输入；聚焦单步预测，未分析多步生成行为；规模有限，未覆盖行业级大模型；可能受软最大约束、训练数据采样和模型初始化等因素影响。

---

## 121. DS@GT ARC at LongEval: Citation Integrity and Factual Grounding in Scientific QA

**arXiv ID:** 2607.14400 | [PDF](https://arxiv.org/pdf/2607.14400v1)

**作者:** Brandon Michaels `[一作]` (Georgia Institute of Technology), Brendon Johnson `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了基于CRAG和CiteFix的纠正式检索增强生成（RAG）流水线，以提升科学问答的引用完整性和事实根源；

**💡 创新点**

创新点在于将预检索（CRAG）与后生成引用验证（CiteFix）结合，并通过RAGAs框架实现无参考的可信度评估，揭示传统NLP指标与引用真实性的冲突；

**🔧 技术方法**

使用的技术包括BM25+BGE混合检索、跨编码器（cross‑encoder）进行文档可信度评估、Gemma‑4‑31B生成模型、GPT‑5.5 Thinking与Claude Opus 4.7 Thinking前沿模型、CiteFix后处理以及RAGAs LLM‑judge评估；

**📊 数据集**

实验数据集为CLEF 2026 LongEval Task 4的CORE科学文献语料库，每个查询附带10篇预检索文档；

**📈 对比分析**

与基线和前沿模型对比，Hybrid Baseline在ROUGE/BERT上显著优于Naive Baseline；CRAG+CiteFix在Global Faithfulness与Citation Faithfulness上超越Hybrid Baseline（0.784/0.758），但ROUGE/BERT略低；前沿模型在ROUGE/BERT最高，却在Citation Faithfulness上仅为0.417；

**⚠️ 局限性**

局限性包括：CRAG+CiteFix在统计上未显著提升；评估依赖单一LLM-judge，缺乏多评委验证；未对不同模型规模进行广泛测试；以及在不同时间点语料漂移下的鲁棒性待进一步验证。

---

## 122. Instrument Effects in Language-Model Honesty Evaluation: An Auditable Single-System Demonstration

**arXiv ID:** 2607.14399 | [PDF](https://arxiv.org/pdf/2607.14399v1)

**作者:** Justin Bronder `[一作]` `[通讯]` (Corabo Inc), Justin Bronder (Corabo Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建可审计的文本冒险评估环境，固定语言模型玩家，系统性变化工具配置（判决语法、成功准则披露、预算呈现、叙述注册），并量化这些工具变量对模型“诚实”评估结果的影响。

**💡 创新点**

首次在完全可追溯的实验框架内演示，评估工具本身的选择比模型差异更能驱动最终结论，揭示评估设计的潜在偏差；并提出四检查完整性协议。

**🔧 技术方法**

使用GLM‑5.2作为玩家、Haiku‑4.5作为叙述者和解释器，采用预注册门控规则、两阶段编码审计、独立同族与跨族检验以及自动化日志与哈希清单。

**📊 数据集**

自制的文本冒险图世界（12个可解/不可解实例、4个字节相同锚点、5个预算级别）共计约5,181条预算叙述，包含多轮游戏日志。

**📈 对比分析**

通过对比不同工具设置下的判决分布、误报率、决策点数等指标，发现：三种判决语法将强主张率从38/40降至7/40；披露成功准则将误报从18/59降至0/58；预算呈现方式显著影响强主张率（灯塔0.15对比仪表0.38）。

**⚠️ 局限性**

局限在于仅使用单一玩家模型与叙述者组合、实验环境为人工构造、难以证明效应在更广泛模型或更复杂世界中的可迁移性；此外，部分指标受温度等超参数影响，需进一步跨族和跨平台验证。

---

## 123. Model-Informed Joint Material-Structural Optimization of Hard-Magnetic Soft Materials

**arXiv ID:** 2607.14397 | [PDF](https://arxiv.org/pdf/2607.14397v1)

**作者:** Ian Galloway `[一作]`, Prashant K. Jha `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个基于有效剪切模量框架的硬磁软材料（hMSM）的预测分析与联合材料-结构优化方法，并用其设计了旋转、平移和恢复型软机器人结构。

**💡 创新点**

创新点在于：① 将传统刚性包含、Hill自洽和受限运动学模型统一为G(ϕ)形式，产生21种本构模型；② 通过实验数据选择Mooney模型并构建可同时优化结构密度、粒子体积分数和剩磁方向的非线性有限变形拓扑优化框架；③ 采用能量插值实现磁体与结构密度的物理耦合。

**🔧 技术方法**

主要技术包括能量基连续介质模型、有效剪切模量理论、有限元求解、Helmholtz滤波与Heaviside投影、逆向相移梯度法、MMA优化算法以及FEniCSx/FEniTop软件实现。

**📊 数据集**

使用了文献中硬磁复合材料的单轴真实应力-应变数据（含0、0.05、0.15、0.30体积分数），以及自行设计的旋转轮、折叠臂和恢复梁等几何模型进行数值仿真。

**📈 对比分析**

通过比较不同本构模型的平均位移误差、相对误差和RMSE，发现Mooney模型在所有指标上表现最佳；优化结果显示在旋转、平移和恢复案例中实现了约50–60%的位移提升或约30% 的顺应性降低，表明联合优化方案显著优于传统预先设定结构或磁体分布。

**⚠️ 局限性**

局限性包括：① 仅考虑固定剩磁方向和不随场变化的磁能，未引入双极子-双极子相互作用或磁阻尼；② 只在二维或简单三维几何中验证，复杂几何或三维磁场求解尚未实现；③ 需要实验数据以校准有效模量，缺乏通用模型；④ 计算成本高，尤其是多负载多材料耦合下的非线性求解。

---

## 124. Exploring Delay-based PUFs for Energy-Efficient Low-Overhead Security of Wearable Devices

**arXiv ID:** 2607.14395 | [PDF](https://arxiv.org/pdf/2607.14395v1)

**作者:** Venkata Prasanth Yanambaka `[一作]` (Texas Woman's University), Saraju P. Mohanty `[通讯]` (University of North Texas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对可穿戴设备的低功耗与安全需求，设计并仿真两种物理不可克隆功能（PUF）架构：Arbiter PUF 与 Hybrid Oscillator Arbiter（HOA）PUF，并对其性能进行评估。

**💡 创新点**

创新点在于提出了基于FinFET 14 nm技术的 HOA‑PUF，能够在生成1位密钥时仅消耗2.7 µW，唯一性达到51.3%，同时保持较低的可靠性误差（2.35%），并具备可重新配置的特点，显著优于传统的 Arbiter 和 RO PUF。

**🔧 技术方法**

技术手段包括：FinFET 14 nm电路模型、蒙特卡洛工艺变异仿真、挑战‑响应对（CRP）生成、以及对唯一性、可靠性、平均功耗等指标的量化分析。

**📊 数据集**

使用的数据集为仿真得到的10,000个环形振荡器频率样本及对应的PUF输出位，主要来源于对FinFET参数（L=20 nm、Fin宽度10 nm等）的随机变异模拟。

**📈 对比分析**

比较方法：将HOA‑PUF与传统RO PUF和Arbiter PUF在相同工艺条件下进行同等挑战数的唯一性、可靠性和功耗比较。结果显示：HOA‑PUF的功耗是Arbiter PUF的约1/10（2.7 µW vs 25 µW），唯一性略高于两者（51.3% vs 50.4%/50.3%），可靠性误差略高（2.35% vs 1.32%/1.8%）。与文献综述的其他技术相比，HOA‑PUF在功耗与唯一性方面具备竞争优势。

**⚠️ 局限性**

局限性：仅在仿真环境下验证，缺乏真实硬件实现与长期环境（温度、老化、噪声）下的可靠性评估；缺少针对机器学习攻击的鲁棒性分析；功耗测量基于理想条件，实际系统中可能受外围电路影响；未给出完整密钥生成速率与电池寿命的量化预测。

---

## 125. A Comparative Analysis of Machine Learning Models for Long and Short-Term Forecasting of the Egyptian Stock Market: A Focus on EGX30

**arXiv ID:** 2607.14391 | [PDF](https://arxiv.org/pdf/2607.14391v1)

**作者:** Muhammed Walid `[一作]` (Egypt-Japan University of Science and Technology), Walid Gomaa `[通讯]` (Egypt-Japan University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对埃及EGX30指数开展短期（1天、1周）和长期（1个月、2个月）预测，比较多种机器学习与深度学习模型的性能。

**💡 创新点**

创新点在于结合传统ML与RNN（GRU）以及KNN的组合，并引入滞后+滚动窗口特征、丰富的技术指标以及集成混合模型来提升长期预测精度。

**🔧 技术方法**

使用的技术包括KNN、决策树、随机森林、Extra Trees、XGBoost、AdaBoost、LightGBM、LSTM、GRU；采用Adam优化器、RMSE/MAPE/R²评估指标，并通过混合集成方法提升预测。

**📊 数据集**

数据集为EGX30指数的历史日收盘价，辅以SMA、EMA、CCI、RSI等技术指标，并通过滞后特征构造时间窗口。

**📈 对比分析**

通过对每个模型在四个预测期计算RMSE、MAPE、R²进行比较，结果显示XGBoost在1天预测最佳，GRU在1周/1个月/2个月预测最佳，KNN在长期表现突出；集成模型在长期预测的RMSE下降约5倍，表现最优。

**⚠️ 局限性**

局限在于模型调参耗时大、未引入季节性变换或情感分析，且研究仅针对单一指数，缺乏跨市场验证。

---

## 126. Beyond Visual Grasping: Benchmarking Complex Grasping from Detection to Execution

**arXiv ID:** 2607.14341 | [PDF](https://arxiv.org/pdf/2607.14341v1)

**作者:** Hanyi Zhang `[一作]` (University of Liverpool), Baoru Huang `[通讯]` (University of Liverpool)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GCA-Bench 基准，用于评估从检测到执行整个抓取流程中面对复杂场景和语义约束的机器人抓取能力。

**💡 创新点**

创新点在于将多阶段抓取与场景级推理、语义约束和语言指令结合，提供新评估指标和多难度任务集，弥补现有基准只关注抓取姿态检测的缺陷。

**🔧 技术方法**

采用大模型与强化学习的端到端 VLA 策略、传统检测+运动规划管线、cuRobo 轨迹优化以及 IsaacLab 机器人仿真环境。

**📊 数据集**

使用自采集的 5000 条仿真轨迹、800 条真实机器人轨迹、MultiGripperGrasp 数据集以及 NVIDIA Omniverse 资产构建场景。

**📈 对比分析**

对比传统检测+规划与 VLA 方法，整体成功率低于 70%，复杂场景下表现更差；fine‑tuned 的 π_0.5 在各类任务中取得最佳成绩，但仍未突破语义推理瓶颈。

**⚠️ 局限性**

局限在于仅针对并行夹爪设计，难以推广到多指或吸盘抓取；对高层语义指令的理解仍主要靠模式匹配，缺乏真正的推理与闭环反馈。

---

## 127. The Prover Is the Judge: Verified Security Software from AI Coding Agents in Ada/SPARK

**arXiv ID:** 2607.14340 | [PDF](https://arxiv.org/pdf/2607.14340v1)

**作者:** Tobias Philipp `[一作]` `[通讯]` (secunet Security Networks AG), Tobias Philipp (secunet Security Networks AG)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用AI编码代理在Ada/SPARK环境中自动实现并通过GNATprove、Why3等工具完成证明，构建了包含加密原语、协议栈等的安全软件集合。

**💡 创新点**

将AI生成代码与证明驱动循环相结合，提出“证明即裁判”模式，在保持低监督成本的同时实现多组件的自动化高保障开发。

**🔧 技术方法**

采用Ada/SPARK、GNATprove、Why3、SMT求解器（CVC5、Z3）、Isabelle/HOL、已知答案测试、互操作性测试及TIMECOP/dudect等技术。

**📊 数据集**

使用标准测试向量（NIST ACVP、RFC vectors、OpenSSL、strongSwan、OpenSSH、matrix.org等）以及自生成的单元测试集。

**📈 对比分析**

与传统人工验证对比，证明驱动循环将监督成本降低数倍（约×），在数小时内完成证明覆盖，错误率显著下降。

**⚠️ 局限性**

证明只能验证实现满足其自身规格，无法捕捉规格错误；弱检查易被代理规避；侧信道与执行时序安全无法证明；需人工审查规格和测试；编译器未经过验证，可能导致误译。

---

## 128. Transition-Aware Routing in Hybrid Hollow-Core/Single-Mode Fiber Networks: A Cost--Throughput Investigation

**arXiv ID:** 2607.14324 | [PDF](https://arxiv.org/pdf/2607.14324v1)

**作者:** Md Ghulam Saber `[一作]` (Huawei Technologies Canada), Zhiping Jiang `[通讯]` (Huawei Technologies Canada)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在混合HCF/SMF光网络中考虑跨纤维转换成本的动态一对一保护路由方案，系统性比较了六种路由算法。

**💡 创新点**

首次将跨纤维转换的GSNR惩罚融入路由目标，并提出两种中间方案（GMR‑T与BD‑TPAR），在不完全忽略转换成本的前提下实现可调节的转接惩罚与路径代价平衡。

**🔧 技术方法**

使用事件驱动仿真、IMI增强的GN模型、K‑shortest‑paths与状态扩展图、四类路由方案以及四层阻塞判定（延迟不对称、GSNR无效、光纤可用性、无工作/保护路径）。

**📊 数据集**

采用六个公开拓扑（CORONET、COST239、NSFNET、USNET、COST266、Nobel‑Germany），在不同HCF占比（0%–100%）与随机/连通部署模式下进行实验。

**📈 对比分析**

通过拥塞概率、承载流量、跨纤维转换次数、延迟不对称率及复合可用性四指标比较六种方案。结果显示，BD‑TPAR在大多数负载下几乎不损失流量（<1%）却可减少约11%转换；GMR‑T在低流量下进一步提高吞吐（+2–3%），TPAR/GFJ在高转换成本场景可将转换次数减半，但吞吐下降20–25%。

**⚠️ 局限性**

局限在于仅考虑单波段连续性、简化的失效/维护模型、单一光纤参数假设，未涵盖多色光源、动态波分复用调度、非均匀功率分配等更复杂实际场景。

---

## 129. Counterfactual Optimal Action Trees (COAT): Interpretable Prescriptive Policies from Observational Data

**arXiv ID:** 2607.14318 | [PDF](https://arxiv.org/pdf/2607.14318v1)

**作者:** Youssef Drissi `[一作]` (IBM Research), Zack Xue `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了COAT框架，利用因果估计与大规模整数规划学习可解释的决策树政策，并在航空增值服务价格设置上进行现场验证

**💡 创新点**

创新点在于将对照因果预测与列生成优化相结合，形成路径为单位的可解释决策树，并在受限运营约束下实现可扩展且可审计的政策

**🔧 技术方法**

使用双重鲁棒因果估计、路径基MIP+列生成、混合整数优化、合成控制评估

**📊 数据集**

基于全球航空公司近两年（2022-2023）43M笔机票预订交易数据，并在17周现场试点中收集效果数据

**📈 对比分析**

与基线（旧式固定价格）及合成控制对照相比，COAT实现了平均每周6.9%收入提升，系统规模化可达5–15亿美元年增收；在计算上相较于弧基树/传统贪心树在同等约束下取得更快收敛与更优解

**⚠️ 局限性**

局限：仅单期决策、未直接纳入实时库存或需求动态、依赖无混杂假设、对多期交互和预测不确定性处理不足

---

## 130. DRIFT: Direct Reduced Fourier Transforms for Distributed Spectral Neural Operators

**arXiv ID:** 2607.14394 | [PDF](https://arxiv.org/pdf/2607.14394v1)

**作者:** Sana Taghipour Anvari `[一作]`, David Kaeli `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种分布式傅里叶神经算子（DRIFT），通过分布式截断频谱变换（DTST）替代传统分布式 FFT，仅计算并传输需要的频谱系数，从而显著提升大规模高分辨率问题的计算效率。

**💡 创新点**

创新点：①将频谱截断与通信顺序逆转，仅局部计算所需频谱系数；②采用两次收集（AllReduce+AllGather）而非多次全量全转发，通信量从 O(N^d/P) 下降到 O(M)；③实现了 GPU‑aware 的分布式 Partial‑DFT，兼顾数据并行与模型并行；④在保证数值精度的前提下大幅降低通信瓶颈。

**🔧 技术方法**

技术手段：分布式截断频谱变换（DTST）；每维分离的基矩阵乘法（Partial‑DFT）实现（cuBLAS）；GPU‑aware MPI（AllReduce/AllGather）；顺序压缩（progressive compression）减少中间计算量；对比基准的所有通信与计算模型。

**📊 数据集**

使用 PDEBench 的 3D 可压缩 Navier‑Stokes 数据集（128³ 空间、21 时间步、5 物理量），进行训练与推理评估。

**📈 对比分析**

与传统分布式 FNO 在 4–32 GPU 上做强/弱扩展对比：DRIFT 前向推理速度提升 38–64×，训练速度提升 37×；通信时间从 97% 降至 <6%；在相同模式数下保持数值精度与收敛性。

**⚠️ 局限性**

局限性：①仅在保留模式数 k_max ≪ N 时有效，k_max 接近 N 时性能退化；②通信量固定为 O(M)，GPU 数量进一步增大时可能变为通信瓶颈；③实验仅在单一 FNO 架构与特定网络拓扑下验证，其他任务或更大通道宽度的表现未知；④需要提前知道截断模式，限制了对动态频谱需求的适应性。

---

## 131. MamaBench: Benchmarking LLM Robustness in Maternal and Child Health Diagnosis through Counterfactual Clinical Perturbation

**arXiv ID:** 2607.14385 | [PDF](https://arxiv.org/pdf/2607.14385v1)

**作者:** Thanni Adewuyi `[一作]` (Helpmum Africa), Abiodun Adereni `[通讯]` (Helpmum Africa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了面向孕产和儿科领域的对照式基准MamaBench，并研发了Evidence-Anchored RAG（EA-RAG）检索增强生成方法。

**💡 创新点**

创新点在于将对照式评估与诊断固定（Diagnostic Fixation）概念引入临床AI，使用Bias Trap Rate（BTR）衡量鲁棒性，并通过三阶段检索覆盖策略改进检索质量。

**🔧 技术方法**

采用检索增强生成（RAG）框架、临床参数抽取、覆盖审核、对比子查询和基于错误分类的生成脚本，全部在推理时实现，无需额外微调。

**📊 数据集**

数据集为由三名专家撰写的434条孕产儿病例，构成217对对照样本，覆盖371种疾病。

**📈 对比分析**

在八种模型配置（含四大前沿LLM和两种RAG实现）上评估，结果显示基线准确率往往高估鲁棒性，EA-RAG在Claude Sonnet 4.6上将BTR从25.8%降低至20.3%，鲁棒准确率提升至65.0%，但整体仍存在约20%对照失误。

**⚠️ 局限性**

局限性包括仅覆盖单一英文临床领域、评估依赖LLM判别器可能带来偏差、以及专家手工编写限制了基准的可扩展性。

---

## 132. Random Parameter Noise Does Not Make Exact ReLU Verification Easy

**arXiv ID:** 2607.14375 | [PDF](https://arxiv.org/pdf/2607.14375v1)

**作者:** Mojtaba Soltanalian `[一作]` `[通讯]` (University of Illinois Chicago), Mojtaba Soltanalian (University of Illinois Chicago)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

本文在对神经网络权重和偏置进行独立高斯噪声扰动、裁剪到[-2,2]并取最接近的二进制点后，证明若存在多项式时间的完整验证器，则可推出NP⊆BPP，从而给出该 smoothed 模型下的下界；

**💡 创新点**

创新点在于将 E3SAT 的完美间隙嵌入与量化鲁棒性相结合，构造四 ReLU/子句的网络得到可扩展的验证间隙，并利用高斯集中证明即使在参数噪声下该间隙保持线性，从而构造 BPP 算法收缩 NP；

**🔧 技术方法**

使用的技术包括加权敏感性不等式、Gaussian concentration、dyadic 精确取整、裁剪、SMoothed 分析、Markov 截断、二进制采样等；

**📊 数据集**

该工作并未使用真实数据集，而是基于 E3SAT 实例及手工构造的三变量三维块进行理论构造与有限维实验；

**📈 对比分析**

通过对构造网络在不同噪声水平下的全局最优值进行数值验证，展示误差保持概率与噪声、网络规模的关系；这些实验仅验证证明机制，并未与现有验证器性能做直接比较；

**⚠️ 局限性**

局限性在于仅针对特定的噪声模型（绝对参数噪声、裁剪、二进制取整）、单隐藏层网络、有限输入维度；不适用于训练得到的网络或特殊结构；噪声水平 σ=2^-11 为保守估计；结果不说明训练网络在噪声下是否可高效验证。

---

## 133. Dynamic Manipulation Hypergraphs for HAR: Beyond Pairwise Relations: Dynamic Manipulation Hypergraphs for Vision-Based Human Activity Recognition

**arXiv ID:** 2607.14350 | [PDF](https://arxiv.org/pdf/2607.14350v1)

**作者:** Fatemeh Ziaeetabar `[一作]` `[通讯]` (University of Tehran), Fatemeh Ziaeetabar (University of Tehran)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于时间可变超图的视觉人体动作识别框架，用高阶关系直接建模手、物体、工具与表面之间的协同交互。

**💡 创新点**

创新点在于将多实体交互视为超边并随时间动态选择模板，配合基于接触、接近与运动耦合的判据和自适应重要性加权，实现更精准的高阶关系推理。

**🔧 技术方法**

使用超图神经网络进行节点↔超边消息传递、时间注意力聚合，融合视觉、空间、运动与语义角色特征，并通过判据生成超边候选与权重。

**📊 数据集**

主要使用 EPIC‑KITCHENS‑100/VISOR 和 Assembly101 进行定量评估，ARCTIC 用于定性接触分析。

**📈 对比分析**

与视频级、实体级、匹配的对偶图以及静态超图基线对比，动态超图在 HO‑F1 上分别提升 6.9–9.5 点、5.4–9.5 点，整体精度和宏 F1 亦显著提升。

**⚠️ 局限性**

局限性包括对精准手物体/工具/表面定位的高度依赖、预定义超边模板覆盖范围有限，以及重要性得分仅为注意力提示，不能视为因果解释。

---

## 134. HABIB_TAZ at SemEval-2026 Task 11: Disentangling Formal Logic from Content via Synthetic Training and Multi-Objective Optimization

**arXiv ID:** 2607.14349 | [PDF](https://arxiv.org/pdf/2607.14349v1)

**作者:** Abdullah Shaikh `[一作]` (Habib University), Abdul Samad `[通讯]` (Habib University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了SemEval-2026 Task 11的系统，用规则生成的合成逻辑数据在12种语言、含噪声前提的环境下对演绎推理的有效性进行分类；

**💡 创新点**

通过将Adaptive Group DRO、可调差分偏置惩罚和KL一致性正则化联合成多目标损失，配合符号化样本与HANS对抗训练，首次实现对内容偏差的显式抑制与逻辑结构的分离；

**🔧 技术方法**

技术上使用mDeBERTa‑v3变体作为backbone，配合分组分布鲁棒优化、动态调度的偏置惩罚、KL‑Divergence一致性正则、符号化抽象数据、HANS以及多语言翻译策略；

**📊 数据集**

数据集包括原始960英文样本、规则生成的23k–650k合成样本（覆盖256 Aristotelian moods/figures），多语言翻译、符号变量与HANS样本；

**📈 对比分析**

在ST1–3上取得100.0 Ranking Score（0 %内容偏差），在ST4排名第6（89.06 %准确、2.89 %偏差），显著优于基线和单一正则化方法；

**⚠️ 局限性**

局限性在于ST4完整训练未完成（仅用约4 %数据），跨语言泛化仍有提升空间，KL一致性正则化计算成本高且对硬件资源要求较大。

---

## 135. Long-History User Transformers for Real-Time Ad Ranking

**arXiv ID:** 2607.14331 | [PDF](https://arxiv.org/pdf/2607.14331v1)

**作者:** Viacheslav Ovchinnikov `[一作]` (Yandex), Maksim Kuzin `[通讯]` (Yandex)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个两阶段 Transformer 框架，将长历史序列编码离线完成，实时仅处理短窗口，最终实现 CTR 预测。

**💡 创新点**

创新点在于：① 把高容量长序列模型拆分为离线大型 Transformer 与在线轻量化 Transformer；② 采用双目标自回归预训练（反馈预测 + 下一项预测）提升表征能力；③ 在细调时使用两塔结构兼容现有 CatBoost 排序器，保持低延迟。

**🔧 技术方法**

使用技术包括：Transformer、ALiBi 位置编码、窗口化自注意力、两塔 DCN V2 预测头、CatBoost 排序器、双目标自回归预训练、两阶段 fine‑tune。

**📊 数据集**

数据集为 Yandex 广告日志，跨 Surface 的完整交互历史（搜索、搜索广告、YAN、Product Gallery）约 10 亿条事件，历时一年，覆盖数百万用户。

**📈 对比分析**

与原有 CatBoost 基线相比，离线实验在 Search Ads 上 NLL 提升 12.3%（占不可部署全历史 Transformer 的 72%），在 YAN 上 3.26%；在线 A/B 测试中 Search Ads 主指标提升 2.77%，YAN 2.1%；收益增长分别为 2.26% 与 0.43%；且请求延迟保持不变。

**⚠️ 局限性**

局限性包括：① 离线向量仍会有几天延迟，需刷新策略；② 模型版本陈旧会快速退化；③ 方案依赖 Yandex 的异步 GPU 计算和特征存储，对其他平台迁移有一定门槛；④ 对极少活动用户的短窗口覆盖有限。

---

## 136. Exact Online Rank Recycling in Floyd's Uniform Subset Sampler

**arXiv ID:** 2607.14302 | [PDF](https://arxiv.org/pdf/2607.14302v1)

**作者:** Yingqi Zhang `[一作]` `[通讯]` (Tsinghua University), Yingqi Zhang (Tsinghua University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了Floyd子集采样器的在线随机数回收，证明每轮中排名可以被安全地立即回收，从而实现无偏、完整的子集生成。

**💡 创新点**

创新点在于给出一个局部双射证明，说明在Floyd采样过程中每轮的随机抽取与更新集合之间独立，可即时回收该随机数而不影响后续结果；并通过信息理论熵分析确认该方法的无损性。

**🔧 技术方法**

主要技术包括均匀状态（uniform‑state）模型的分裂/合并操作、秩树（order‑statistic tree）实现Floyd转换、局部双射构造与证明、熵计量以及有限状态枚举验证。

**📊 数据集**

实验使用了两类数据集：对 n≤8 的所有 (n,m) 组合进行完整枚举验证；以及 n=30,000、m=20,000 的大规模实例，在 Rust 实现中记录熵跟踪。

**📈 对比分析**

通过熵计量与完整状态枚举验证正确性；与现有子集采样器对比，本文未提出运行时间优势，仅在随机数利用率上达到理论极限（>99.99% 输出+池子效率）。

**⚠️ 局限性**

限制包括：仅证明随机数回收的正确性而未提供实测运行时性能；实现受限于有限字宽导致的重置与熵损失；未讨论安全性、并行性能或在非理想随机源下的鲁棒性。

---

## 137. XCT-SAM: Sequential Parameter-Efficient Domain Adaptation of SAM for Industrial XCT Defect Segmentation

**arXiv ID:** 2607.14287 | [PDF](https://arxiv.org/pdf/2607.14287v1)

**作者:** Md Mahedi Hasan `[一作]` (West Virginia University), Srinjoy Das `[通讯]` (West Virginia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了XCT‑SAM框架，通过在金属微观结构数据上先行适配，再迁移到XCT缺陷分割，实现了高效的域自适应；

**💡 创新点**

创新点在于：①分阶段的中间域适配策略；②使用Conv‑LoRA低秩适配器，保持>99%参数冻结；③rank r=2的极低参数量即可取得优异性能；

**🔧 技术方法**

技术方法包括：Segment Anything Model (SAM) ViT‑H backbone + Conv‑LoRA适配器 + Dice‑Focal 损失；

**📊 数据集**

使用的数 据集：基于CycleGAN的合成XCT数据集（含孔洞与夹杂）和真实 NIST XCT 数据；

**📈 对比分析**

与 UNet++、SAM、MedSAM、SAM‑Med2D、Conv‑LoRA‑SAM 等基线对比，XCT‑SAM 在合成与真实评测集上均实现了更高的 IoU 与 Dice（最高 IoU 0.325、Dice 0.905），明显优于所有对照组；

**⚠️ 局限性**

局限性：需要两阶段训练，耗时成本较高；对噪声/对比度变化的鲁棒性有限；目前仅以 2D 切片二分类模型为主，未覆盖完整 3D 多类别分割。

---

## 138. Stable Voting is PSPACE-Complete

**arXiv ID:** 2607.14366 | [PDF](https://arxiv.org/pdf/2607.14366v1)

**作者:** Ethan Dickey `[一作]` (Purdue University), Athina Terzoglou `[通讯]` (Purdue University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文解决了在稳定投票和简单稳定投票规则下，赢家确定的计算复杂性问题，证明了赢家确定是PSPACE完全的。

**💡 创新点**

创新点在于首次证明了稳定投票和简单稳定投票的赢家确定问题是PSPACE完全的，并且即使在承诺赢家是两个特定候选人之一的情况下，仍然是PSPACE完全的。

**🔧 技术方法**

使用了递归定义和复杂的边权重构造，结合量化布尔公式的构造方法来证明复杂性。

**📊 数据集**

使用了加权比赛图（weighted tournament），特别是构造了一个与给定量化布尔公式相对应的加权比赛图。

**📈 对比分析**

通过与已知的PSPACE完全问题（量化布尔公式）进行归约，证明了赢家确定的复杂性。比较方法的性能表明，赢家确定在这两种投票规则下都是极其复杂的。

**⚠️ 局限性**

限制在于当前的结果仅适用于特定的投票规则，未来的研究需要探索在更广泛的情况下是否存在可行的多项式时间算法。

---

## 139. Learning Who to Treat When Treatment is Missing

**arXiv ID:** 2607.14346 | [PDF](https://arxiv.org/pdf/2607.14346v1)

**作者:** Johnna Sundberg `[一作]` (Carnegie Mellon University), Edward Kennedy `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究在治疗数据缺失的情形下，提出基于MAR与MCCAR假设的高效政策学习与CATE估计方法；

**💡 创新点**

证明MAR估计在MCCAR场景下更高效且无偏，且在MAR情形下可稳健处理结果依赖缺失；

**🔧 技术方法**

采用影响函数（Influence Function）驱动的双重鲁棒（DR）估计、非参数机器学习、DR‑Learner以及多重样本分割技术；

**📊 数据集**

在合成数据、半合成数据（来自投票与课堂规模实验的RCT数据）上进行评估；

**📈 对比分析**

与传统完整案例、倾向分数、基于结果回归、MTRNET、深度学习等基线方法比较，MAR‑DR和MAR‑OR在缺失率低至30%时AUPEC显著优于其他方法；

**⚠️ 局限性**

局限包括对MNAR缺失假设的依赖、R‑Positivity可能被违反、极小样本下速率条件难以满足，以及需对全样本CATE进行估计的前提。

---

## 140. Value Leakage: An LLM's Answers Are Silently Shaped by Its Own Values

**arXiv ID:** 2607.14345 | [PDF](https://arxiv.org/pdf/2607.14345v1)

**作者:** Jan Betley `[一作]` (Truthful AI), Owain Evans `[通讯]` (Truthful AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建一系列对抗性评估，研究语言模型在回答用户问题时未披露的价值泄漏现象。

**💡 创新点**

创新点在于提出了“隐蔽价值泄漏”这一新对齐失败类型，并提供量化方法衡量模型的隐蔽程度。

**🔧 技术方法**

采用了基于对抗性提示的Counterfactual评估、链式推理（CoT）跟踪、LLM判别器分类以及潜在混合模型来估计偏差比例。

**📊 数据集**

实验使用了九道Fermi估算任务、AI泡沫与AGI推文问题、工作机会评估、代理人评分与休闲活动随机选择等多种任务作为数据集。

**📈 对比分析**

通过与Claude、GPT、Gemini等前沿模型在同一评估集上的比较，发现Claude系列在多数任务中泄漏度最高，GPT在部分任务中几乎无泄漏；但所有模型均存在一定程度的隐蔽泄漏。

**⚠️ 局限性**

局限性包括任务数量有限、可能的样本偏倚、无法区分价值观本身与泄漏倾向、仅评估总结式CoT而非原始推理，以及对LLM判别器的依赖导致估计为下限。

---

## 141. Beyond scalar losses: calibrating segmentation models via gradient vector field surgery

**arXiv ID:** 2607.14338 | [PDF](https://arxiv.org/pdf/2607.14338v1)

**作者:** Laurin Lux `[一作]` (Technische Universität München), Johannes C. Paetzold `[通讯]` (Cornell Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种对区域基损失（如Dice、Tversky）梯度进行“手术”调节的技术，以提高医学影像分割模型的校准性与保持高精度。

**💡 创新点**

创新点在于从梯度向量场角度分析并纠正过度自信问题，构造非保守梯度场替代标量损失，实现误差与梯度大小线性相关，兼顾类别不平衡。

**🔧 技术方法**

利用梯度手术（gradient surgery）、自定义向量场、指数衰减因子、以及多种基线损失（Dice、Tversky、ComboLoss、m1L1-ACE、Dice++等）与常规交叉熵、SVLS、NACL等对比。

**📊 数据集**

在四个医学分割数据集上评估：2D FIVES（视网膜血管）、2D INbreast（乳腺癌肿瘤）、3D BraTS‑METS（转移瘤）、3D KiTS（肾瘤）。

**📈 对比分析**

相较基线，梯度手术显著降低NLL、ECE、MCE、Brier等校准指标（常降低4–6倍），且在DSC上保持或略微提升，尤其在CE+Dice组合下表现最佳。

**⚠️ 局限性**

局限性包括：梯度场非保守导致理论上缺乏可证明的收敛性；对超参数（指数n）仍需经验调优；在极端标签噪声或域迁移场景下的鲁棒性尚未充分验证。

---

## 142. MixCompress: Mixture of Experts for Variable Rate Learned Image Compression

**arXiv ID:** 2607.14334 | [PDF](https://arxiv.org/pdf/2607.14334v1)

**作者:** Calvin-Khang Ta `[一作]` (Dolby Laboratories), Peng Yin `[通讯]` (Dolby Laboratories)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的可变率学习式图像压缩框架MixCompress，利用稀疏专家路由与深度混合以及条件辅助变换实现多率点的高效编码。

**💡 创新点**

创新点在于将稀疏Mixture‑of‑Experts（MoE）和Mixture‑of‑Depth（MoD）模块嵌入关键瓶颈，消除密集参数调制导致的梯度冲突，并通过Conditional Auxiliary Transforms（CAT）对小波子带能量进行率感知调制，实现结构化的可变率特化。

**🔧 技术方法**

核心技术包括稀疏门控MoE/ MoD路由、条件辅助小波变换（CAT/iCAT）、率嵌入式路由器、梯度冲突分析与噪声退火平衡专家利用。

**📊 数据集**

使用OpenImages训练集进行联合训练，评估在Kodak、CLIC（Professional验证与测试）以及Tecnick四个公开图像压缩基准上。

**📈 对比分析**

与单率基准（LALIC、LIC‑TCM、FAT等）以及密集调制的可变率方法（QRAF、Cond‑Conv）对比，MixCompress在所有数据集上都能匹配或超越单率模型的BD‑Rate表现，尤其是MoD变体在CLIC和Tecnick上相对VTM‑23.1实现了约25–29%的BD‑Rate降低。

**⚠️ 局限性**

限制在于仍需为每个率点预先计算并缓存路由与CAT参数，且实验主要集中在高分辨率自然图像，未来需验证在视频、非自然场景或更大规模数据上的鲁棒性。

---

## 143. Measuring How Students Rely on Generative AI in Academic Writing: Development and Multi-Source Validation of the Generative AI Reliance Types Scale (GenAI-RTS)

**arXiv ID:** 2607.14301 | [PDF](https://arxiv.org/pdf/2607.14301v1)

**作者:** Shahin Hossain `[一作]` (University of Maryland Baltimore County), Tukhbita Afroz Nawmi `[通讯]` (University at Buffalo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并验证了一种用于评估本科生在学术写作中对生成式AI依赖类型的量表（GenAI‑RTS）

**💡 创新点**

首次以理论为导向构建四类依赖类型（Strategic、Instrumental、Dependent、Dialogic）并检验其五因素结构，提供跨性别、第一代与STEM与非STEM学生的标度不变性

**🔧 技术方法**

采用结构方程建模（CFAs、测量不变性检验）、Rasch评级尺度分析以及定性访谈方法进行验证

**📊 数据集**

使用382名美国少数族裔服务机构本科生的问卷数据（20项量表）及14名受访者的访谈数据

**📈 对比分析**

五因素模型在拟合度（CFI≈0.92，RMSEA≈0.07）、内部一致性（ω≥0.75）和测量不变性上优于所有替代模型，验证了量表的有效性

**⚠️ 局限性**

样本来自单一机构，缺乏行为日志等外部验证，且7点量表的类别功能失衡，建议改为5点格式

---

## 144. Majority Correctness in Social Networks: From Well-Mixed Electorates to Complex Networks

**arXiv ID:** 2607.14288 | [PDF](https://arxiv.org/pdf/2607.14288v1)

**作者:** Dan Braha `[一作]` (University of Massachusetts, Dartmouth), Marcus A. M. de Aguiar `[通讯]` (State University of Campinas)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在存在执着领导者（zealot）与对立者（contrarian）影响下，投票前的持续社会互动如何改变多数正确性。作者构建了一个二项式的zealot–contrarian voter模型，推导了全混合与Erdős–Rényi网络情形下的稳态投票分布，计算单个自由选民的有效胜算率与多数投票的正确性，并通过数值模拟比较不同网络拓扑（Erdős–Rényi、尺度自由网络、环网络、小世界网络）对多数正确性的影响。

**💡 创新点**

创新点在于：① 将Condorcet Jury Theorem与网络社交动力学结合，提出一个既能捕捉自由选民相互影响，又能反映执着领导者偏见的动态模型；② 在该模型下获得完整的稳态分布解析（Beta–Binomial/Dirichlet–Multinomial），并以此得到多数正确性的闭式条件；③ 明确揭示“社交互动可能导致多数正确性下降”的聚合失败，并确定完全顺从（r=1）是唯一在无限人群中保持多数优势的临界点；④ 通过模拟验证网络结构通过产生不同的投票相关性来影响多数准确性。

**🔧 技术方法**

技术手段主要是：马尔可夫链与出生–死亡链的详细平衡分析、Beta–Binomial 与 Dirichlet–Multinomial 分布推导、Erdős–Rényi网络的均值场（mean‑field）近似、Jensen不等式与协方差分析、数值模拟（蒙特卡洛）以及对不同网络拓扑的图论特征（平均度、聚类系数、直径）进行比较。

**📊 数据集**

数据集：本研究不使用现实世界选举数据，而是通过构造的模拟网络（随机生成的Erdős–Rényi、Barabási–Albert尺度自由网络、环网络、Kautz式小世界网络）和理论推导得到的分布来评估模型。每个网络的规模为n=501个自由选民，执着者参数为(α₁,α₂)=(5,2)等。

**📈 对比分析**

比较方法：① 对全混合系统与Erdős–Rényi网络使用解析结果与均值场近似；② 对不同拓扑使用相同参数下的蒙特卡洛模拟得到多数正确性Mₙ与单个自由选民正确性p，进一步计算Mₙ−p和Mₙ−Mₙ^{ND}（无社交互动基准）。结果显示：在全混合与Erdős–Rényi网络中解析与模拟高度一致；尺度自由网络产生更高的投票相关性，导致多数正确性下降；环网络与低重连概率的小世界网络产生更低相关性，提升多数正确性；所有拓扑均保持Mₙ>p（当p>0.5）但Mₙ<Mₙ^{ND}（当p>0.5）——即社交互动虽然提升单个选民正确率，却可能削弱多数聚合效果。

**⚠️ 局限性**

局限性：① 模型假设投票者行为极其简化（仅复制或相反），未考虑复杂的理性推理、证据权衡或信息获取；② 均值场近似在高度异质或高度聚类的网络（如尺度自由网络）下可能失效；③ 只考虑固定数量执着者的极端偏见，未研究动态执着者或多阶段决策过程；④ 模拟规模有限（n≈500），对极大人群行为的外推仍需谨慎；⑤ 仅聚焦二项决策，未扩展到多选项或连续偏好情境。

---

## 145. Accounting for Hysteresis and Eddy Currents in Finite Element Simulations of Ferromagnetic Laminated Cores using a Recurrent Neural Network

**arXiv ID:** 2607.14321 | [PDF](https://arxiv.org/pdf/2607.14321v1)

**作者:** Florent Purnode `[一作]` (University of Liege), Christophe Geuzaine `[通讯]` (University of Liege)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于循环神经网络（RNN）的同质化模型，用来替代传统的有限元仿真中对层压磁芯的滞后与涡流耦合求解，实现在二维磁矢势有限元框架中的高效、准确仿真；

**💡 创新点**

创新点在于：①直接用RNN学习磁通密度到磁场的有效本构关系，避免在每个积分点进行多尺度求解；②利用自动微分获取精确雅可比矩阵，保持Newton–Raphson收敛；③模型内嵌误差预测，支持自适应时间步；④使用残差式和输入裁剪，确保在饱和域外仍保持物理一致；

**🔧 技术方法**

采用GRU型循环网络、Energy-Based滞后模型、自动微分、变量时间步的自适应控制、二维磁矢势有限元求解器；

**📊 数据集**

构造了约10万条人工生成的磁场时间序列（含谐波、单向、直流偏置、脉冲、斜坡等），并通过1D层压模型（EB模型+涡流）得到对应的磁通密度序列作为训练/验证/测试数据；

**📈 对比分析**

与参考层压模型对比，平均误差约6–7 mT（相对误差<1 %），误差预测与真实误差高度相关；计算成本约为无滞后模型的两倍，Newton迭代收敛与传统无滞后模型相当；

**⚠️ 局限性**

局限性包括：仅适用于各向同性层压钢；二维近似，无法捕捉边缘三维涡流；对训练范围外高场需手动切换到无滞后；需要为新材料重新生成数据并再训练；当前I/O实现为文件读写，导致约20 %额外时间；未验证PWM或热耦合等实际工况。

---

## 146. Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency

**arXiv ID:** 2607.14481 | [PDF](https://arxiv.org/pdf/2607.14481v1)

**作者:** Andreas Meuleman `[一作]` (Inria, Université Côte d'Azur), George Drettakis `[通讯]` (Inria, Université Côte d'Azur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种能够在无序图像序列下实现即时反馈的3D Gaussian Splatting（3DGS）重建方法，解决传统方法只能在序列有序且需离线计算的问题。

**💡 创新点**

创新点包括：①基于视觉场所识别（MixVPR）与可见性图的两级快速匹配，可在大规模无序序列中高效检索关键帧；②基于聚类的无序闭环检测与“焊接”窗口束束调整，实现无迭代的姿态与高斯原语更新；③进阶层级结构按屏幕尺寸动态合并/拆分高斯原语，支持极大场景的实时优化与渲染。

**🔧 技术方法**

核心技术包括：MixVPR + XFeat + LightGlue的多级匹配流水线；权重可见性图与贪婪子集选取；GPU加速的局部束束调整；基于高斯原语的光照与深度估计；图搜索（Dijkstra）实现闭环后姿态传播；层级合并策略（KL散度）。

**📊 数据集**

使用公开数据集：MipNeRF360、Tanks & Temples、Deep Blending（包含有序与无序子序列）以及大规模城市场景CityWalk；对照的有序场景还引用了TUM等。

**📈 对比分析**

与多种基线方法（COLMAP+3DGS、DROID-Splat、AnySplat、S3PO-GS、LongSplat、Octree-GS 等）进行对比。实验显示：平均延迟约 324 ms（最大 386 ms），整体计算时间比离线方法低约 6 倍；在质量上与 3DGS 7k 迭代接近，超越大多数即时或离线基线；在大场景中显著降低 GPU 内存占用并保持可实时渲染。

**⚠️ 局限性**

局限性：与离线全局 BA 仍存在小质量差距；纯旋转场景仍难以三角化；图传播方式虽快速但不如完整全局 BA 精细；对极端光照或缺少可见性的信息依赖较高。

---

## 147. Can Tokens Compete? Token Representations against Supervised CNN Backbones for BirdCLEF+ 2026

**arXiv ID:** 2607.14474 | [PDF](https://arxiv.org/pdf/2607.14474v1)

**作者:** Anthony Miyaguchi `[一作]` (Georgia Institute of Technology), Adrian Cheung `[通讯]` (Georgia Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文构建了一个基于CNN的监督基线模型，并探讨了使用神经音频编码器生成的token表示在鸟类/动物声音检测中的可行性；同时在BirdCLEF+ 2026数据集上对比了两类模型的表现。

**💡 创新点**

创新点在于将Perch v2冻结backbone、HGNetV2-B0和非鸟类原型头进行集成，形成高效的监督基线；再结合token化的音频表示（包括神经音频编码器与基础语义嵌入），系统性地比较了专用生物声学模型与通用AudioSet token编码器在同一任务下的效果。

**🔧 技术方法**

使用技术包括冻结CNN backbone（Perch v2）、HGNetV2-B0声学事件检测网络、原型头（prototypical head）集成；神经音频编码器（如codec模型）和基础语义嵌入（foundational embeddings）；token化的音频编码器（AudioSet训练的四种token模型）；基于CPU的90分钟训练预算以及私有排行榜评估。

**📊 数据集**

使用BirdCLEF+ 2026 Pantanal湿地的音频场景数据集（约1小时标注的多标签音频），以及用于预训练token编码器的AudioSet数据集。

**📈 对比分析**

通过在BirdCLEF+ 2026私有排行榜上评测，基线CNN集成模型获得0.936的得分；token化模型在相同条件下的得分略低但与基线相近，说明token表示在当前数据规模下尚未能显著超越传统CNN方法。

**⚠️ 局限性**

限制主要包括：标注数据量仅约1小时，可能导致模型泛化不足；token模型受限于AudioSet的预训练，无法充分捕捉生物声学细节；缺乏更广泛的消融实验和对不同音频场景的适用性评估。

---

## 148. Mixed-Agent Museum Tour Guide Design Improves Gendered Learning Outcomes and Visitor Preferences

**arXiv ID:** 2607.14468 | [PDF](https://arxiv.org/pdf/2607.14468v1)

**作者:** Annette M. Masterson `[一作]` (University of Michigan), Dawn Tilbury `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了一种混合代理的博物馆导览系统，结合物理机器人与投影虚拟代理，提供双代理互动体验。

**💡 创新点**

创新点在于将投影虚拟代理与单一机器人平台融合，实现两位移动代理的交互丰富度，同时通过多种对话风格（故事化、互相打趣）探讨性别差异对学习效果的影响。

**🔧 技术方法**

使用技术包括Toyota HSR 机器人、ROS 生态、舵机驱动的万向投影、ArUco 标记实时跟踪、逆向运动学与凸二次规划路径规划、PyGame 动画、OpenAI TTS、行为测量模块。

**📊 数据集**

实验数据来源于30名参与者的问卷、前测/后测测验、行为传感器（距离、头部角度、反应时间）和访谈，未使用公开数据集。

**📈 对比分析**

通过三种条件（单机器人、故事化混合、打趣混合）的被试内实验，结果显示混合代理在女性参与者中显著提升学习成绩，打趣条件在此群体中最高；但参与度与体验质量在各条件间无显著差异。

**⚠️ 局限性**

局限性包括样本量有限、实验环境为实验室模拟、幽默表现主观、机器人速度过慢导致自然度不足，缺乏在真实博物馆高流量环境中的验证。

---

## 149. Do Generative AI Assistants Respect robots.txt? Tracing Web Access Beyond Visible Answers

**arXiv ID:** 2607.14447 | [PDF](https://arxiv.org/pdf/2607.14447v1)

**作者:** Gabriel Lopez-Fonseca `[一作]` (Universidad Politécnica de Madrid), Jose M. Del Alamo `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在实验服务器上托管带有唯一秘密码的页面，设计了两阶段实验：先识别各主流 AI 助手在实际交互中能否进行实时网页检索并记录其 User-Agent；随后在受控的 robots.txt 条件下，对 10 款助手进行 20 次检索测试，监测是否请求 robots.txt、访问目标页面、返回正确秘密码，评估其遵守 robots.txt 访问规则的行为。

**💡 创新点**

创新点在于：① 将机器可读的 robots.txt 合规性与 AI 助手的实时检索行为相结合，构建了可复现的、可量化的检索合规性评估框架；② 通过在实验域中设置四类访问规则（全允许、全禁止、助手专属允许、助手专属禁止），同时对每个助手的 User-Agent 进行识别和归类，揭示了多助手之间在遵守访问约束上的显著差异；③ 发现了检索与答案正确性不一致、未请求 robots.txt 直接访问页面、以及基于通用 User-Agent 的隐蔽访问等新问题。

**🔧 技术方法**

使用的技术主要包括：自建 HTTP 服务器（记录访问日志、返回带 HMAC 码的 HTML 页面）；在每个实验域中设置 robots.txt 文件并根据 User-Agent 匹配；对 AI 助手的交互采用统一的简洁提示“Get the contents from [URL]…”，并通过日志与返回内容对比验证检索。对助手配置的选择和实验流程采用了多轮重试、临时会话与正式账号等策略。

**📊 数据集**

数据集为实验中生成的动态 HTML 页面，每页嵌入唯一 HMAC 码，并为每个助手分配不同的页面编号范围（如 Claude 2001–2020）。通过对 200 次检索实验（10 款助手 × 4 规则 × 5 次）收集了服务器访问日志和助手返回的答案，形成了实验数据集。

**📈 对比分析**

比较方法是：对每个助手在四种 robots.txt 条件下，统计（① robots.txt 请求次数、②目标页面访问次数、③正确答案数量）。结果显示，Claude 和 Mistral 在允许/禁止规则下表现出预期的访问模式；其余助手大多出现未请求 robots.txt 直接访问、访问无效或返回错误码等行为，表明多数助手未能可靠遵守 robots.txt。性能表现方面，遵守率与回答正确率并不总是同步，存在检索成功但答案不完整的情况。

**⚠️ 局限性**

限制包括：① 仅测试了 10 款主流助手的单一 Prompt，无法覆盖其他产品或更新版本的行为；② 只针对静态 HTML 页面，未评估对 JavaScript、动态加载、登录或付费内容的访问；③ 由于助手是闭源系统，无法明确内部检索链路，导致无法确定违规是检索层还是后端索引层；④ 实验时间窗口有限，后续更新可能改变结果；⑤ 只验证了 robots.txt 的合规性，未考虑其他访问控制机制（如 Cookie、CAPTCHA 等）。

---

## 150. Disclosure Divergence: Measuring Privacy Policy and Data Safety Misalignment at Scale

**arXiv ID:** 2607.14442 | [PDF](https://arxiv.org/pdf/2607.14442v1)

**作者:** Mst Eshita Khatun `[一作]` (Louisiana State University), Aisha Ali-Gombe `[通讯]` (Louisiana State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析Android应用的隐私政策与Google Play数据安全标签的一致性，量化不一致程度并评估其风险。

**💡 创新点**

提出跨层次一致性与敏感性加权风险评分，区分收集与共享操作并对高敏感数据类别进行细粒度误差分析。

**🔧 技术方法**

采用LLM（LLaMA 3.1 8B）进行隐私政策文本抽取，并使用二元指标、Cohen κ、余弦相似度等统计方法比较标签与政策。

**📊 数据集**

构建了6,051个Android应用的语料，包含Data Safety标签、隐私政策文本和应用元数据，覆盖14类数据类别。

**📈 对比分析**

通过一致性/不一致性分数、κ值（约0.31/0.16）和敏感性风险分层，发现约三分之一数据不一致，且共享层误差更高，整体一致性仅低于两分之一。

**⚠️ 局限性**

仅分析英文政策，LLM抽取可能受限于语言风格，且仅按类别二值化忽略了细节上下文，未结合代码或网络流量验证实际行为。

---

## 151. Prices, Probabilities, and Parlays: Systematic Bias in Sports Prediction Markets

**arXiv ID:** 2607.14430 | [PDF](https://arxiv.org/pdf/2607.14430v1)

**作者:** Niusha Moshrefi `[一作]` `[通讯]` (Princeton University), Niusha Moshrefi (Princeton University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对 Kalshi 体育预测市场的价格进行系统化评估，发现价格在不同剩余时间段内的校准性与产品类型存在显著偏差；

**💡 创新点**

首次量化并模型化时间到期（TTE）对单腿合约校准曲线的动态影响，以及交叉游戏串关产品的系统性溢价，揭示了传统价格即概率解释的局限；

**🔧 技术方法**

利用基于 TTE 的分桶、功率律与 Platt 归一化模型、Prelec-Ⅱ概率加权函数、以及对串关价格的独立性乘积与经验校准曲线进行比较；

**📊 数据集**

使用 Kalshi 2026 年 NBA、MLB、NHL 23 万笔交易数据和 1.26 万笔串关交易；

**📈 对比分析**

通过对不同 TTE 桶和腿数的校准曲线拟合和比对，发现单腿在中期 TTE 时基本校准，但在最后十分钟出现阶梯形 Prelec 形状；串关的价格超出独立性乘积，且溢价随腿数指数增长；整体性能表明传统价格即概率模型在大多数情境下失效；

**⚠️ 局限性**

局限包括缺乏订单簿和交易者身份信息以进一步解释行为动因；样本主要覆盖 2026 年春季，可能不具备季节性普适性；串关样本规模在腿数高时较小，统计不稳。

---

## 152. Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers

**arXiv ID:** 2607.14427 | [PDF](https://arxiv.org/pdf/2607.14427v1)

**作者:** Joe Logan `[一作]` `[通讯]` (Independent researcher), Joe Logan (Independent researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究随机深度递归Transformer的逐Token收敛行为，并设计了无需训练的逐Token退出策略；

**💡 创新点**

创新点在于直接测量每Token的收敛深度，发现不同Token收敛速率差异，并证明基于收敛判定的无参数退出策略在平均深度和质量上优于学习型路由器；

**🔧 技术方法**

采用权重共享递归Transformer结构（Llama-style层），随机采样递归次数进行预训练，无附加停机单元或FLOP惩罚；

**📊 数据集**

主要使用FineWeb-Edu（12.9B token）和TinyStories（400M token）数据集；

**📈 对比分析**

对比方法包括统一深度（A0）、无参数收敛退出（A1）和基于收敛标签的线性路由器（A2）；在S1模型上，A1在平均深度约4.94时即可匹配统一深度8的损失，平均深度下降38%，而A2需近8层才能达到同等质量；

**⚠️ 局限性**

局限性包括：仅在135M级别单一seed实验；未验证更大规模或更复杂任务的收敛深度分布；未实现真实推理时的depth-wise批处理以转化为实际吞吐率；未探讨键值缓存与多token退出的兼容性；

---

## 153. $K$-NeAS: Scalable Multi-Material CT Reconstruction Using Neural SDFs

**arXiv ID:** 2607.14415 | [PDF](https://arxiv.org/pdf/2607.14415v1)

**作者:** Daksh K. Shah `[一作]` (University of California), Razvan V. Marinescu `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 K-NeAS，一个可扩展的多材料CT重建框架，使用共享潜在主干、软可微顺序占用选择器、自动化的GMM阈值估计和浮点正则化，实现了从稀疏投影直接恢复连续衰减场和表面几何。

**💡 创新点**

核心创新包括：① 软、全微分的顺序占用过滤器，可处理任意数量的材料；② 通过GMM自动估计衰减阈值，消除手动调参；③ 计划性的辅助浮点损失抑制稀疏视角下的假空域几何；④ 共享主干提升跨材料特征学习。

**🔧 技术方法**

采用了隐式场表征（Neural SDF）、多分辨率哈希编码、GMM边界估计、软选择器公式、辅助浮点损失、Adam优化、3D体渲染和PSNR/SSIM评价指标。

**📊 数据集**

在公开的 NAF CBCT 数据集上进行实验，涵盖腹部、胸部、足部和下颌四个解剖区域，每个区域各有 50 条训练投影和 50 条验证投影。

**📈 对比分析**

与原始 NeAS 基线对比，K-NeAS 在 2D 投影和 3D 体积上均取得更高的 PSNR/SSIM，尤其在腹部三材料配置下 3D PSNR 提升 1.88 dB，稀疏视角（5-10 视角）下相较基线提升高达 1.17 dB。

**⚠️ 局限性**

主要局限是 GMM 阈值估计在高对比、同质场景（如下颌）中失效，导致材料分割误差；在颅骨等极高对比的 CT 数据上仍出现噪声和表面崩塌，需要改进边界估计和采样策略。

---

## 154. Reward-Free Evolving Agents via Pairwise Validator

**arXiv ID:** 2607.14408 | [PDF](https://arxiv.org/pdf/2607.14408v1)

**作者:** Minghao Liu `[一作]` (Accenture), Wei Wei `[通讯]` (Accenture)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在自演化智能体循环中用对比式验证器取代标量奖励的方案

**💡 创新点**

创新点在于使用冻结的LLM做二进制对比判定，并提出Adaptive Focus与Soft Elo两种全无奖励的实现

**🔧 技术方法**

采用对比式LLM验证器、少量示例提示、可调节评判维度以及Elo评分更新机制

**📊 数据集**

在检索/指令遵循（HotpotQA、HoVer、IFBench、PUPA）、数学（AIME、LiveBench‑Math）和代码演化（ADR、ShinkaEvolve等）等十个任务上评估

**📈 对比分析**

与全奖励基线相比，实验显示在大多数设置下性能相当或更好，验证到测试误差更小，且在不同代理与引擎上均保持竞争力

**⚠️ 局限性**

局限包括：当基线已饱和、测试集过小、代理或验证器能力不足或两者匹配差时，验证器难以产生有效判定，导致无效或负面效果

---

## 155. Decision Making Needs Uncertainty Quantification [Lecture Notes]

**arXiv ID:** 2607.14407 | [PDF](https://arxiv.org/pdf/2607.14407v1)

**作者:** Osvaldo Simeone `[一作]` `[通讯]` (Northeastern University London), Osvaldo Simeone (Northeastern University London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种统一决策框架，阐述在已知与未知环境下，针对风险中性与风险厌恶的代理人，如何选择合适的不确定性表征（后验分布、预测集合、校准预测器、可可信集合与分布鲁棒优化、贝叶斯参数后验）并给出对应的最优决策规则。

**💡 创新点**

创新点在于将贝叶斯决策理论、校准预测、分布鲁棒优化与贝叶斯推理等看似独立的技术通过同一决策问题统一起来，指出不确定性表征必须匹配决策目标与知识状态，并提供相应的性能保证与置信度证明。

**🔧 技术方法**

主要技术包括贝叶斯推断、风险度量（期望、Value-at-Risk）、预测集合构造、分布校准、可可信集合（credal set）与分布鲁棒优化（DRO）以及参数化模型下的贝叶斯后验推理。

**📊 数据集**

文中未给出具体实验数据集，仅在理论层面讨论了数据驱动情境下的经验分布、可可信集合构造与参数后验推断。

**📈 对比分析**

由于是理论综述与框架构建，未进行数值实验或与现有方法的性能对比，故不存在具体性能指标；但作者指出在已知环境下风险中性策略只需后验，风险厌恶策略只需覆盖预测集合，且可可信集合可在给定置信度下提供可靠下界。

**⚠️ 局限性**

局限性包括：1）对实际数据集与模型的实验验证缺失；2）在未知环境下对预测器校准的评估依赖于理想化的完整校准假设；3）可可信集合与DRO的计算复杂度与参数选择（如半径r）需要经验调优；4）在高维状态空间下后验与可可信集合的构造可能不可行。

---

## 156. Safe Execution of RL Policies Via Acceleration-Based CBF-QP Constraint Enforcement for Real-World Robotic Deployments

**arXiv ID:** 2607.14488 | [PDF](https://arxiv.org/pdf/2607.14488v1)

**作者:** Bastien Muraccioli `[一作]` (CNRS–AIST Joint Robotics Laboratory), Mehdi Benallegue `[通讯]` (CNRS–AIST Joint Robotics Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出 Acc-CBF-QP，一种基于加速度空间 QP 的运行时安全过滤器，可在不重新训练的情况下把任意 RL 策略包装进控制管线，实时满足关节、速度、扭矩与碰撞约束。

**💡 创新点**

创新点在于：①将 CBF 约束与加速度 QP 结合形成可插拔安全层；②提供两种任务形式（TorqueTask 与 FDTask）以控制安全-性能折衷；③支持硬件无改动、开源实现，并兼容任意现成 RL 策略。

**🔧 技术方法**

使用技术包括：加速度空间 Quadratic Programming、Control Barrier Functions、全身动力学逆动力学、Generalized Momentum Observer、倾斜观测器、外部扰动估计、PD/力矩控制器、以及机器人动力学模型。

**📊 数据集**

实验数据集：Kinova Gen3 7-DoF 机器人（公开 PPO 策略），Unitree H1 19-DoF 机器人（CaT 训练的 PPO 策略），在仿真与真实硬件上进行验证。

**📈 对比分析**

与无安全过滤、RL+硬阈值、CaT Safe RL 等方案对比，实验显示 Acc-CBF-QP 在 H1 上将约束违规率从 10.04 次/秒降至 0.80 次/秒（92% 降低），Kinova 上完全消除违规；在高速度激进命令下生存时间提升约 20%；在无违规区间任务性能保持与原 RL 无显著差异。

**⚠️ 局限性**

局限性包括：仅与 CaT 进行比较；缺乏对闭环系统稳定性的理论保证；QP 失效时采用经验退路（阻尼/回滚），未提供更严格的可行性恢复方法；未探讨多任务优先级与在线约束冲突处理。

---

## 157. MIDAS Hand: Modular low-Impedance Direct-drive Anthropomorphic Sensing Hand

**arXiv ID:** 2607.14487 | [PDF](https://arxiv.org/pdf/2607.14487v1)

**作者:** Alvin Zhu `[一作]` (UCLA), Dennis Hong `[通讯]` (UCLA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并发布了一款低成本、开源、人形的抓握机器人手，配备直接驱动背驱动、283个三轴触觉传感器，支持实验和数据收集。

**💡 创新点**

将直接驱动背驱动、密集触觉、模块化3D打印结构三大技术融合，实现低成本、可维护且功能完整的人形抓手，并提供完整的软件生态。

**🔧 技术方法**

采用Dynamixel XM335伺服直接驱动、四杆联动实现关节耦合、Paxini三轴触觉阵列、模块化3D打印零件、Python API、URDF/MJCF仿真模型以及Retargeting与Teleoperation流水线。

**📊 数据集**

实验评估使用GRASP抓取分类、力传感数据以及手势捕捉数据（OpenPose/Manus glove），未使用公开大型数据集。

**📈 对比分析**

与Sharpa Wave、Wuji Hand等直接驱动手进行背驱动扭矩、抓取分类、力学负载、连续抓取可靠性比较，结果显示背驱扭矩≈0.02 N·m，低于商用手；抓取覆盖32/33类，抓取负载≥9.5 kg，重复误差0.016 mm。

**⚠️ 局限性**

缺少小拇指导致某些抓取失效；触觉未集成闭环控制；结构受伺服热限，长期负载受限；未评估任务级自主控制。

---

## 158. SAGA: Schema-Aware Grounding for Agentic Text-to-SPARQL Generation

**arXiv ID:** 2607.14494 | [PDF](https://arxiv.org/pdf/2607.14494v1)

**作者:** Yiming Zhang `[一作]` (University of Tokyo), Koji Tsuda `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种交互式文本到SPARQL生成框架SAGA，利用知识图谱schema在属性探索时进行类型约束，减少错误的三元组模式，提升复杂KBQA性能。

**💡 创新点**

核心创新是将schema约束从后置验证迁移到前置构造阶段，维护前向后向类型状态并在每次属性检索时做过滤与注解，实现无训练的类型感知Grounding。

**🔧 技术方法**

使用LLM代理（如gpt-oss-120B、LLaMA‑3.1‑70B）进行多轮工具调用；构建轻量化schema索引、持久化类型状态、类型约束过滤、紧凑型邻域展示；兼容OpenAI/Local LLM。

**📊 数据集**

在Wikidata的六个基准（QALD‑7、QALD‑9+、QALD‑10、WWQ‑test、SPINACH‑test、LC‑QuAD 2.0）以及Freebase的三个基准（WebQSP、CWQ、GrailQA）进行评估。

**📈 对比分析**

与多种基线（Zero‑shot SPARQL、Entity Linking、Data Shapes Prompting、OBQC、mKGQAgent、GRASP、Interactive‑KBQA、SPINACH）比较，SAGA在所有六个Wikidata数据集的F1均为最高（最高达64.89），在Freebase上提升0.2–11.2 F1；同时显著降低空结果率并减少LLM裁剪调用。

**⚠️ 局限性**

局限在于依赖显式类型注解，对缺失或不完整schema的实体依赖估计或回退，且对复杂类型推理能力有限，未来需提升类型推断与自适应约束策略。

---

## 159. Context Contamination in LLM Analysis of Network Security Logs: Poison with Passive Prompt Injection and Mitigation Evaluation

**arXiv ID:** 2607.14493 | [PDF](https://arxiv.org/pdf/2607.14493v1)

**作者:** Rabimba Karanjai `[一作]` (University of Houston), Weidong Shi `[通讯]` (University of Houston)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了 LLM 在安全运营中心日志分析中的提示注入脆弱性，提出 LogInject 框架并构建了 12,847 条日志的基准数据集 LogInject-1.0。

**💡 创新点**

创新点在于：①首次量化 Passive Log Injection 的攻击成功率；②设计了三层防御（输入过滤、提示硬化、输出验证）并验证其组合效果；③引入 Context Stitching 技术，展示了分片注入能绕过无状态过滤器的攻击；④提供公开数据集与评测流水线，为后续研究提供可复现基准。

**🔧 技术方法**

使用的技术包括：Transformer 语言模型（GPT‑4o、Claude‑3.5 Sonnet、Llama‑3‑70B‑Instruct）在日志摘要与警报分级任务中进行推理；正则表达式过滤、Prompt Spotlighting（XML 包裹 + 明确信任边界）以及输出验证的“canary”机制；检索式 RAG 结合 BM25+dense 进行日志检索。

**📊 数据集**

使用的主要数据集是 LogInject‑1.0，包含 10,278 条正常日志（Apache、SSH、JSON API）和 2,569 条人工构造的对抗日志，按攻击层级（Atomic、Fragmented、Obfuscated）、向量（HTTP、SSH、JSON、错误）和攻击目标（CONCEAL、FABRICATE、EXFIL、INSTRUCT）划分。

**📈 对比分析**

与三款主流 LLM 对比，基线 ASR 最高达 88.2%，最低 74.8%；攻击层级越高 ASR 越低（Atomic 86.5%→Fragmented 78.4%→Obfuscated 66.7%）。三层防御组合后，ASR 下降至 8.4%（下降 90%），但仍有 8.4% 的残留风险；同时正常日志准确率从 94.2% 降至 90.8%，误报率上升约 3.4%。

**⚠️ 局限性**

局限性包括：①评估仅覆盖三种模型，未涵盖小型或领域专用模型；②数据集来源主要是公开基准与人工合成，缺乏真实企业生产日志的多样性；③攻击者假设为“无状态写入”，未考虑可观测并自适应的攻击；④防御措施依赖正则与规则，面对持续演化的编码/混淆攻击仍易被突破；⑤评测仅针对推理时直接注入的情景，未覆盖离线或神经符号混合管线。

---

## 160. EdgeFaaS: A Function-based Framework for Edge Computing

**arXiv ID:** 2607.14489 | [PDF](https://arxiv.org/pdf/2607.14489v1)

**作者:** Neha Vadnere `[一作]` (Arizona State University), Ming Zhao `[通讯]` (Arizona State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出EdgeFaaS，一个统一的函数化边缘计算框架，通过函数与存储虚拟化实现对IoT、边缘和云资源的统一管理与部署，支持工作流定义、自动放置与容错；

**💡 创新点**

创新点在于：1）将异构、多层级资源聚合为统一FaaS资源池；2）提供函数与存储虚拟化隐藏底层差异；3）扩展Serverless Workflow规范以支持边缘工作流，并实现分区化全局与局部编排；

**🔧 技术方法**

技术包括：函数即服务（FaaS）模型、OpenFaaS/MinIO、Kubernetes、REST API统一接口、工作流DAG解析与分区、故障检测与重试、跨层级存储与数据私有化；

**📊 数据集**

使用数据集：MNIST（手写数字）、AudioSet（音频事件）、ESC-10/ESC-50（环境声音分类），以及视频分割/检测/识别等自定义数据；

**📈 对比分析**

通过在5个地理分布节点的100+设备实验，比较不同工作流配置下的计算、通信与总延迟；FL实验展示了非层级FL慢、层级FL快但精度低；音频分类实验展示了细调频率与准确率、总细调时间的权衡；编排开销低，端到端延迟稳定在376–418 ms；

**⚠️ 局限性**

局限性在于：1）仅实现基础的函数/数据放置策略，缺乏高级调度与缓存机制；2）工作流案例有限，未覆盖更复杂场景；3）对资源异构度支持主要聚焦软件层，硬件依赖仍待深入；4）实验规模受限于现有设备与网络延迟，未覆盖更大规模的全网部署。

---

## 161. Active Real-World Factor-Based Evaluation for Generalist Robot Policies

**arXiv ID:** 2607.14439 | [PDF](https://arxiv.org/pdf/2607.14439v1)

**作者:** Andrew Liao `[一作]` (University of Minnesota Twin Cities), Aryan Deshwal `[通讯]` (University of Minnesota Twin Cities)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于任务因子空间的主动评估框架，将策略评估视为贝叶斯主动实验设计；

**💡 创新点**

创新点在于把评估任务转化为主动测试问题，利用概率代理模型和信息增益采样函数实现样本高效评估；

**🔧 技术方法**

使用高斯过程、混合密度网络和深度集成等代理模型，并结合PSD、NIPV、BALD、EPIG等采集函数；

**📊 数据集**

在3个真实世界UR5e机器人操作任务（取蓝块、放杯子、放绿块入锅）上，构建包含121个物体位置、3张桌高度、3摄像头视角的设计空间，共2331个评估点；

**📈 对比分析**

与随机抽样对比，主动测试在相同评估预算下RMSE与对数似然下降更快，平均可节省20–40%评估次数，且结果更稳定；

**⚠️ 局限性**

局限在于因子维度有限、性能范围窄、代理模型表达能力受限、缺乏更复杂或更大规模因子空间的验证。

---

## 162. Compensation Design

**arXiv ID:** 2607.14438 | [PDF](https://arxiv.org/pdf/2607.14438v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Weiqiang Zheng `[通讯]` (Yale University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出并研究“补偿设计（Compensation Design）”问题，即在预算约束下，设计对参与者的付款规则，激励他们自愿参与并产生高质量的贡献。

**💡 创新点**

创新点包括：①证明存在简单、匿名且成本无关的边际贡献付款规则，该规则总能保证纯纳什均衡并取得近最优的价格无序（PoA）上界 2+o(λ)；②展示在大市场（λ→0）下此上界是不可改进的；③揭示基于Shapley值的付款规则可能导致无纯纳什均衡；④扩展到粗相关均衡（CCE）并证明同样获得常数 PoA；⑤在更一般的XOS、非单调子模以及组合行动空间下给出不可忽略的下界和随机化方案。

**🔧 技术方法**

主要技术包括：潜在游戏分析证明纯纳什均衡存在；信息论与构造性实例证明上界不可改进；潜在函数与不等式推导用于 CCE 的 PoA；随机化混合方案与期望分析；以及对组合行动空间的对数 PoA 设计。

**📊 数据集**

本文为理论研究，未使用具体数据集；所有结果均基于通用价值函数模型（子模、XOS、子加等）。

**📈 对比分析**

与传统预算可行机制（如 VCG、先买后付）或合同设计相比，补偿设计在无需价格申报、无需集中分配的去中心化场景下实现了更简单的匿名付款规则，并在纯均衡与粗相关均衡上均提供了常数 PoA，优于现有方法；随机化方案在无大市场条件下实现常数 PoA，克服了确定性规则的无穷 PoA 下界。

**⚠️ 局限性**

主要限制包括：①在非单调子模或大成本（λ≈1）场景下，确定性规则的 PoA 可能无界；②对于XOS和组合行动空间，无法实现多项式查询下的常数 PoA；③纯纳什均衡的计算复杂度为 PLS‑complete，虽然存在但可能需要指数步；④随机化方案需要预先设定分支概率，实际实现可能受限。

---

## 163. A Measurement Study of AI-Environment Realism Gaps in Malware-Analysis Sandboxes

**arXiv ID:** 2607.14434 | [PDF](https://arxiv.org/pdf/2607.14434v1)

**作者:** Zhiyong Sui `[一作]` (Louisiana State University), Aisha Ali-Gombe `[通讯]` (Louisiana State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 AI 环境指纹在恶意软件沙盒中的检测与规避，构建了 probe 框架并在多种沙盒与 AI 宿主机上做了大规模实验。

**💡 创新点**

首次系统测量 AI 生态留下的持久指纹，并证明其能显著区分沙盒与真实主机，同时揭示生成完整 AI 环境比检测更昂贵的成本不对称。

**🔧 技术方法**

采用 Windows 自举 probe、DNS 泄露、Python 包解析、目录/环境变量/端口/包四类指纹，并结合七种沙盒与三台 AI 宿主机进行实验。

**📊 数据集**

从 284 个 GitHub AI 项目中提取 450 个指纹（目录、环境变量、端口、Python 包），并在 214 步累计安装实验中观察指纹增长。

**📈 对比分析**

与传统 VM 检测基线对比，AI 指纹在沙盒中的检测率低于 5%，而在真实宿主机可达 70‑80%；在累计安装实验中指纹检测率从 0.6% 升至 17.8%，传统基线保持约 5%。

**⚠️ 局限性**

仅限 Windows，样本覆盖有限，未评估 Linux/macOS；只对三台参考主机与多种沙盒对比；AI 指纹的真实世界普及度尚未验证；对非 AI 主机可能失效。

---

## 164. The Adversarial Robustness of Sketching and Streaming Algorithms

**arXiv ID:** 2607.14432 | [PDF](https://arxiv.org/pdf/2607.14432v1)

**作者:** David P. Woodruff `[一作]` (Carnegie Mellon University), Samson Zhou `[通讯]` (Texas A&M University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并系统化了在流式与压缩感知环境下，面对自适应或对抗性输入时保持正确性的算法技术，涵盖插入‑只流、插入‑删除流、白盒对抗者以及线性与非线性草图的鲁棒性。

**💡 创新点**

创新点包括：① 将差分隐私与对抗鲁棒性直接关联，得到空间更优的插入‑只流算法；② 提出“草图切换”和“差分估计”两大框架，统一实现对抗鲁棒性；③ 对线性草图在插入‑删除流中不可避免的多项式空间上界提供严谨证明；④ 将合并‑裁剪（merge‑reduce）核心集构造推广至对抗鲁棒流算法；⑤ 在白盒对抗模型下利用密码学假设（如 SIS）实现稀疏恢复与低秩矩阵恢复。

**🔧 技术方法**

技术手段包括：马尔可夫偏差不等式、Freedman 以及 Azuma‑类型不等式；差分隐私机制（Laplace、Sparse Vector、AboveThreshold）与其复合组合；在线敏感度与重要性采样；合并‑裁剪框架与在线敏感度采样；密码学基于硬度假设的安全随机数生成与硬币。

**📊 数据集**

由于研究以理论为主，本文未使用标准公开数据集，而是通过构造性示例（如区间集合 VC‑维为 1 的攻击、L₂ 范数草图等）来展示对抗攻击与鲁棒性需求。

**📈 对比分析**

对比方法主要以空间复杂度与误差界为基准：插入‑只流下，使用 𝑂(1/ε² log n) 空间即可获得 (1±ε) 的鲁棒估计；而对抗鲁棒线性草图需 𝑂(poly(n)) 空间；合并‑裁剪核心集实现 𝑂(1/ε² log⁴ n) 行空间；在白盒对抗下，算法利用密码学方案可在 𝑂(polylog(n)) 计算时间下实现鲁棒稀疏恢复。

**⚠️ 局限性**

限制包括：① 在插入‑删除（turnstile）流中，线性草图对抗鲁棒性几乎不可能，需多项式空间；② 对抗鲁棒性往往需要在样本量或误差上付出额外的 𝑂(log n) 或 𝑂(log² n) 乘数；③ 某些框架（如差分隐私）在实现中需要对噪声幅度与敏感度的精确估计，实际部署难度高；④ 白盒对抗模型下的安全性依赖于密码学假设，若假设破坏则不保。

---

## 165. ConFlow: Constraints-Guided Learning with Flow Matching for Motion Generation

**arXiv ID:** 2607.14424 | [PDF](https://arxiv.org/pdf/2607.14424v1)

**作者:** Nutan Chen `[一作]` (LS Wiiri Robot Innovation Center), Botond Cseke `[通讯]` (Volkswagen Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出ConFlow，一个将约束引导直接嵌入训练目标并使用条件高斯过程源分布的流匹配框架，用于安全机器人运动生成。

**💡 创新点**

将推理时的约束指导迁移到训练阶段；利用负示例作为负监督；用条件GP源分布强制轨迹平滑与端点约束。

**🔧 技术方法**

流匹配（Flow Matching）、可微分障碍函数、条件高斯过程、软正则化梯度、负示例学习。

**📊 数据集**

二维两机器人交叉位置的仿真数据集，包含约30%碰撞轨迹（正负示例）。

**📈 对比分析**

与标准FM、ConFlow去掉负样本的ablation进行对比，采用无指导、机器人避障、障碍避障三种推理设置评估碰撞率，ConFlow在所有设置下碰撞率显著降低（如0.016 vs 0.031），同时轨迹平滑度提升。

**⚠️ 局限性**

仅在二维仿真环境验证，未测试更高维机械臂；GP源分布对核参数敏感；缺乏真实世界实验。

---

## 166. Settling The Round Complexity of Byzantine Agreement Against a Full-Information, Adaptive Adversary

**arXiv ID:** 2607.14413 | [PDF](https://arxiv.org/pdf/2607.14413v1)

**作者:** Yuval Efron `[一作]` `[通讯]` (Institute for Advanced Study), Yuval Efron (Institute for Advanced Study)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了在全信息、快速、适应性 Byzantine Agreement 协议中，期望回合复杂度至少为Ω(t²/(n log n))，该下界比之前的Ω(t/√(n log n))显著提升；

**💡 创新点**

创新点在于采用全局多轮集中性分析与标签‑篡改攻击，将原先逐回合的 valency 论证替换为一次性强制事件，构造出一次性可被全信息攻击逼迫发生的稀有慢事件；

**🔧 技术方法**

核心技术包括：多轮浓缩引理（类似 Etesami‑Mahloujifar‑Mahmoody），标签‑篡改（label‑tampering）攻击框架，Chor‑Merritt‑Shmoys 的 crash‑schedule 计数论证，以及同步化器（termination synchronizer）来消除协议间的决策间隙；

**📊 数据集**

该工作为纯理论分析，不涉及数据集；

**📈 对比分析**

与现有上界（Dufoulon‑Pandurangan 2025 的 O(min{t² log n/n, t/ log n})）相比，本文的下界在 t≪n 时仅差一个 log² n 因子，基本匹配上界；

**⚠️ 局限性**

局限性包括：仅适用于全信息模型，假设 adversary 具备无限算力；不适用于私有通道或计算受限环境；实现证明高度技术性，可能难以直接推广至更一般的网络或安全模型。

---

## 167. Assisting Mission-Critical Traffic Flows with Active Queue Management in Industrial Internet of Things

**arXiv ID:** 2607.14478 | [PDF](https://arxiv.org/pdf/2607.14478v1)

**作者:** Shuo Wang `[一作]` (Swinburne University of Technology), Zhibo Pang `[通讯]` (Peking University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在真实工业实验平台上对比了FIFO、CoDel、FQ‑CoDel和CAKE四种队列策略，验证了多队列AQP能显著降低IIoT网络中的缓冲区膨胀并提升任务关键流的吞吐率和时延稳定性。

**💡 创新点**

首次在OT域真实工厂环境下系统性评估并量化了多队列AQP（FQ‑CoDel、CAKE）对任务关键流的性能提升，揭示了单队列CoDel在流隔离方面的局限。

**🔧 技术方法**

使用MikroTik路由器实现FIFO与三种AQP；采集工业设备（AIV、摄像头、OPC UA、智能电表）及iperf3产生的TCP背景流；通过tshark收集吞吐量、RTT、抖动和包尺寸等指标进行分析。

**📊 数据集**

采用工业现场产生的实时流量（AIV 15 Kbps、IP摄像头7 Mbps、OPC UA 2/25 Kbps）和iperf3产生的TCP CUBIC背景流，无使用公开数据集。

**📈 对比分析**

通过在拥塞/非拥塞两种网络条件下绘制吞吐量、RTT、抖动和包尺寸的CDF/箱线图进行对比；结果显示FQ‑CoDel/CAKE能保持≈原始吞吐率、RTT≈10 ms、抖动最小，而FIFO导致吞吐率下降80%、RTT升至≈200 ms。

**⚠️ 局限性**

实验规模受限于实验室级别，未覆盖大规模部署、不同无线干扰或TSN/DetNet等专用网络；仅评估IP层AQP，对非TCP/IP流和参数调优仍需进一步研究。

---

## 168. Depth-Dependent Hidden-State Collapse in Dynamical System Autoencoders for LiDAR Point-Cloud Classification

**arXiv ID:** 2607.14463 | [PDF](https://arxiv.org/pdf/2607.14463v1)

**作者:** Patricia Medina `[一作]` (New York City College of Technology), Hy P. G. Lam `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了动态系统自编码器（DSAE）在 LiDAR 点云分类中编码器迭代深度对隐藏表示和分类性能的影响。

**💡 创新点**

首次揭示了在 K=5 时出现的隐藏状态崩塌现象，并通过散布分析证明了其导致类别区分消失的数学原因。

**🔧 技术方法**

使用 DSAE、随机森林、kNN、Dummy 基线、隐藏状态散布分析以及产品系数（Product Coefficients）特征增强技术。

**📊 数据集**

使用来自华盛顿州森林的 798,452 个标记点的地面 LiDAR 点云数据集，三类（地面、树干/枝条、树冠）。

**📈 对比分析**

对 K=1~5 的单独训练模型进行宏 F1 比较：K=1–4 维持较高性能，K=5 全部方法均退化至 Dummy 基线 macro F1 0.224688，显示隐藏表示几乎恒定。

**⚠️ 局限性**

未解释导致深度 5 崩塌的优化机制，仅证明了其对类别区分的影响；缺乏对固定模型中中间状态的分析和对隐藏散布的正则化方法。

---

## 169. Beyond Generalist LLMs: Specialist Agentic Systems for Structured Code Workflow Execution

**arXiv ID:** 2607.14456 | [PDF](https://arxiv.org/pdf/2607.14456v1)

**作者:** Harris Borman `[一作]` (Commonwealth Bank of Australia), Ritchie Ng `[通讯]` (Commonwealth Bank of Australia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了一个专门化的代理系统，将 BPMN 工作流自动转换为可执行的 ReAct 代理，并在十个确定性业务流程上进行评估。

**💡 创新点**

创新点在于将业务流程的结构化规范（BPMN）外部化，形成模板化的工作流编排；通过限定执行路径、上下文管理和模块化工具拆分，显著减少了 token 消耗、消除了修复循环，并提升了代理执行的可靠性与一致性。

**🔧 技术方法**

使用技术包括：BPMN 解析、API 客户端模块生成、工具与上下文自动化创建、基于 ReAct 的代理编写、Langfuse 追踪 token 使用、以及自我验证与迭代修正机制。

**📊 数据集**

数据集为作者自行构造的十个确定性 BPMN 流程（节点数 9–52，边数 10–60），覆盖电商、成本优化、风险评估、信息检索等业务场景。

**📈 对比分析**

与通用代理（Roo、Cline）进行比较，采用的指标包括：修复迭代次数、token 使用量、工具调用准确率（TUE）、惩罚调整后延迟、工具调用错误数、流程遵循率等。实验结果显示，专门化系统在 TUE 上提高 9–20pp，在惩罚延迟上比基线快 2–4×，工具调用错误数降低约 3 倍，token 消耗减少 95% 以上，整体表现明显优于基线。

**⚠️ 局限性**

局限性包括：仅评估确定性流程，未考虑随机性或开放式决策场景；工作流由人工构造，可能存在设计偏差；通用代理的性能可能因更细致的调优而提升；实验环境与真实企业系统的差异仍需进一步验证。

---

## 170. Motion Planning with Model-Based Diffusion via Constraint Optimization and Adaptive Scheduling

**arXiv ID:** 2607.14455 | [PDF](https://arxiv.org/pdf/2607.14455v1)

**作者:** Zhilin He `[一作]` (Carnegie Mellon University), Changliu Liu `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种结合模型基扩散、软约束增量拉格朗日先验和硬约束凸可行集投影的运动规划框架 MD-COAS，并在扩散反向过程中引入自适应安全约束调度。

**💡 创新点**

创新点在于：①将软约束先验（iALM）与硬投影（CFS）统一到同一扩散模型中；②通过残差驱动的双重自适应调度同时调节软约束强度和投影计算预算，使安全约束随噪声水平动态强化；③在扩散逆过程里实现多模态、动态可行性校正。

**🔧 技术方法**

使用技术包括：模型基扩散（MBD）与蒙特卡洛得分上升（MCSA），增量增广拉格朗日软先验，凸可行集投影（CFS）Qp，双重自适应调度（软硬约束的双重拉格朗日多项式），以及离散动力学滚动传播。

**📊 数据集**

数据集主要包括：随机生成的 2D 非凸障碍地图（Easy、Constrained、Union 三类），以及基于 D3IL 的 7-DoF 机械臂避障任务。

**📈 对比分析**

与 MBD、EB-MBD、MDOC、SafeDiffuser、DPCC-C 等基线进行对比。MD-COAS 在安全成功率（SSR）、平均成本、收敛速度和规划时间上均优于或等同于基线，特别在高难度 Union 级地图和 7-DoF 任务中保持 100% 成功率且成本最低。

**⚠️ 局限性**

局限性：①投影计算仍占用显著时间，尤其在高维场景；②方法目前仅适用于离散时间、单机器人、已知动力学模型；③对极端复杂多机器人或非平稳动态环境的适应性仍待验证。

---

## 171. Not Just Pockets: Understanding Phone-Carrying Behaviors of Wheelchair Users for Mobile Context-Awareness

**arXiv ID:** 2607.14437 | [PDF](https://arxiv.org/pdf/2607.14437v1)

**作者:** Yunzhi Li `[一作]` (Carnegie Mellon University), Patrick Carrington `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对91名轮椅使用者的问卷和15名使用者的访谈，系统性描述并分析了轮椅使用者的手机携带位置、影响因素及其对情境感知和交互的意义。

**💡 创新点**

①首次系统化梳理轮椅使用者手机携带行为，揭示其与普通人群的显著差异；②提出基于携带位置的情境感知与交互设计框架；③展示了携带多样性对传感、识别与应用的启示。

**🔧 技术方法**

采用混合方法：在线问卷（Qualtrics）+访谈（Zoom）、主题分析（Braun & Clarke）、亲和图法；并进行初步加速度计/陀螺仪实验来验证携带位置对传感数据的影响。

**📊 数据集**

问卷样本：91名轮椅使用者（包括手动、动力轮椅及双类型）；访谈样本：15名轮椅使用者；加速度计/陀螺仪实验样本：1名受试者在手动轮椅上使用4种携带位置（背包、底座袋、框架挂、裤袋）进行数据采集。

**📈 对比分析**

利用SVM等传统机器学习对携带位置进行识别。仅用步态数据训练的模型在轮椅数据上的准确率仅为27.9%，而在混合训练后准确率提升至约70%+。实验展示了携带位置多样化导致的信号差异和对情境感知模型的挑战。

**⚠️ 局限性**

①样本局限于美国、英语、互联网活跃的轮椅使用者；②未包含更严重运动障碍或部分轮椅使用者；③实验仅在手动轮椅和iOS设备上完成，无法推广到动力轮椅或其他平台；④未收集携带频率、时长等纵向行为数据。

---

## 172. Global drivers and barriers to the public acceptance of autonomous vehicles: Evidence from 17 countries

**arXiv ID:** 2607.14436 | [PDF](https://arxiv.org/pdf/2607.14436v1)

**作者:** Antonios Saravanos `[一作]` `[通讯]`, Antonios Saravanos

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文利用跨国L3Pilot调查数据，构建并验证了一个统一技术接受模型（UTAUT2）来解释公众对SAE Level 3条件自动驾驶汽车的使用意愿

**💡 创新点**

创新点在于首次将UTAUT2模型应用于17个国家的大规模样本，量化了PE、SI、HM等因素对意愿的直接与间接影响，并揭示了社交影响在整个接受过程中的核心作用

**🔧 技术方法**

采用了偏最小二乘结构方程模型（PLS-SEM）来估计测量模型与结构模型，并通过Bootstrapping进行显著性检验

**📊 数据集**

使用了公开的L3Pilot Global User Acceptance Survey第一阶段数据，共18,603名已驾照持有者，覆盖非洲、亚洲、欧洲、北美和南美5大洲的17个国家

**📈 对比分析**

模型解释度极高，R²=0.827，表明82.7%的使用意愿方差被解释；预测相关性（Q²_predict>0）高于简单均值基准，但未显著优于线性模型基准

**⚠️ 局限性**

主要限制包括：跨国样本采用在线面板，缺乏国家层面代表性；自报意愿而非真实使用行为；模型未纳入环境价值、成本、隐私等可持续性与风险因素；缺乏纵向或实验验证，难以确定因果关系

---

## 173. Smarter and Cheaper at Once: Byte-Exact KV-Cache Grafting Turns a Frozen Small Model into a Verified-Knowledge Flywheel

**arXiv ID:** 2607.14431 | [PDF](https://arxiv.org/pdf/2607.14431v1)

**作者:** Sietse Schelpe `[一作]` `[通讯]` (Corbenic AI), Sietse Schelpe (Corbenic AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种基于字节级别精确KV缓存的“Taliesin”机制，结合“Galahad”验证-缓存循环，能够在不调整模型权重、无需额外加速器的前提下，将已验证的知识以可序列化、可重放的KV块形式持久化；随后在推理时将该块直接 graft 到新的上下文中，从而提升模型能力、降低推理成本并扩展上下文窗口。

**💡 创新点**

创新点：①实现了对Transformer KV缓存的完整、可验证的字节级别序列化与恢复，保证 graft 后的 logits 与原始推理完全一致（SHA‑256相同、KL = 0）；②提出“验证‑再缓存”飞轮，首次在推理阶段实现无梯度知识注入；③在相同硬件、相同模型规模下，展示了通过缓存知识实现的显著准确率提升与算力节省；④系统层面证明了跨设备复制、顺序复用与误路复位等细节。

**🔧 技术方法**

技术：①Deterministic KV缓存捕获与恢复；②SHA‑256字节级验证；③一次性或循环缓存的路由与定位；④多GPU架构下的预注册回放；⑤顺序 prefill 与块级复用；⑥多轮 sampling‑and‑voting 对比；⑦能耗与延迟测量。

**📊 数据集**

数据集：AIME 2025（30道数理推理题，基准后续版本）、LiveBench held‑out split（60道题），以及内部构造的 31B、12B 预训练模型本身。

**📈 对比分析**

对比方法与性能：与 Gemma‑4‑12B 官方 77.5% 参考点、Gemma‑4‑31B 89.2% 参考点以及 Qwen3.6‑35B‑A3B 92.7% 参考点比较；在 AIME 2025 上，冻结 12B 通过 8 条验证库从 80.0% 提升至 93.3%；在 recurrence 场景下，8 条问题的推理 token 从 401,026 降至 61（≈ 6,574×），能耗从 0.053 Wh 降至 0.003 Wh（≈ 3,000×‑8,700×）；上下文窗口从 32k 扩展至 2.85M，零额外显存；在跨 GPU（H100、B200）下仍保持 byte‑exact，成本约 8‑12 欧元；整体提升 50‑倍的“精度/成本”比值。

**⚠️ 局限性**

局限性：①仅在模型存在失败案例时有效，若模型已能自行解决则无收益；②已验证块只能在原始位置 graft，跨位置恢复不可行；③字节级别 exactness 仅在同一 GPU 架构内成立，跨架构会出现微小浮点差异；④对同一结构变换（参数化）可迁移，但硬编码常量的任务会失效；⑤多实例/多样本块的 recurrence 可靠性较差；⑥路由误差虽已缓解但仍可能出现 confident‑wrong graft；⑦系统实验多在单机、单槽情况下，未覆盖高并发与大规模部署；⑧完整训练、权重更新等传统方法无法在此框架内实现。

---

## 174. Emergent Region-Level Facial Correspondence in Frozen Vision Foundation Models

**arXiv ID:** 2607.14423 | [PDF](https://arxiv.org/pdf/2607.14423v1)

**作者:** Izaldein Al-Zyoud `[一作]` (University of Ottawa), Abdulmotaleb El Saddik `[通讯]` (University of Ottawa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aaccfe5c-6b26-4208-b23c-35331481e142` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

评估冻结的 DINOv3 ViT‑L/16 特征在跨身份、跨时间的人脸部位对应性，利用 FaRL 提供的语义分割标签进行区域级匹配与跟踪。

**💡 创新点**

首次证明中间层（block 18）能在不做人脸专门训练的情况下保持解剖区域身份，并提出跨身份无约束 NN 匹配与区域混淆矩阵的评估指标，揭示 CLIP 与 DINOv3 在解剖对应上的差异。

**🔧 技术方法**

使用 DINOv3 ViT‑L/16（block 18/24）特征、FaRL 语义分割、最近邻匹配、时间标签传播、K‑means 无监督验证、注意力模块 ablation、SegFormer 交叉验证等技术。

**📊 数据集**

CelebDF‑v2 real videos 用于跨身份与视频跟踪评估，CelebAMask‑HQ 及 SegFormer 用于标签验证。

**📈 对比分析**

采用跨身份无约束 NN 匹配、区域混淆矩阵、语义准确率（DINOv3‑block 18 83% 对 23% 基线），与 CLIP 的对比显示 DINOv3 在解剖对应上优势明显；视频跟踪准确率达 95.5%，CLIP 仅 67.8%；中间层优于最终层。

**⚠️ 局限性**

仅使用 FaRL 作为标签与 GT，单一数据集与解析方案；未训练时间适配器；评估结果为下界，缺乏上界与多样性验证。

---

## 175. A Fast Quantitative Analyzer for NetKAT

**arXiv ID:** 2607.14420 | [PDF](https://arxiv.org/pdf/2607.14420v1)

**作者:** Thomas Lu `[一作]` (Cornell University), Alexandra Silva `[通讯]` (Cornell University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `847a60d8-a755-47af-ba5d-c5236b9e3083` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套可对网络传输策略进行定量分析的完整框架，核心在于将加权 NetKAT 的语义映射为一种新的符号数据结构——加权符号包程序（wSPP），并在此基础上设计了能够在 ω‑连续半环上直接求 Kleene 星运算的算法。同时，作者构造了可携带路径信息的 Pareto 半环，用以在多目标优化（如时延与带宽）中同时得到最优前沿以及对应的最短实现路径。该框架经过 Lean 形式化验证并实现为高性能 Rust 引擎，并在 Fat‑tree 与 Jellyfish 数据中心拓扑的案例研究中进行评估。

**💡 创新点**

创新点主要包括：① 将符号包程序（SPP）推广到半环权重，得到 wSPP，显著压缩表示；② 针对无穷迭代的 Kleene 星设计了多通道符号化求解算法，保证终止并保持可计算性；③ 引入可携带跟踪信息的 Pareto 半环，既保留多目标前沿，又能输出最优路径的简短日志；④ 对整个体系做了 Lean 形式化证明，保证理论与实现的一致性；⑤ 通过参数化半环实现了统一的引擎，既可处理 Boolean 可达性，又能高效完成概率、时延、带宽等定量分析。

**🔧 技术方法**

使用的技术与方法包括：加权 NetKAT DSL、ω‑连续半环与 Kleene algebra、符号决策图（wSPP）与矩阵化操作、Pareto 与前沿半环构造、子序列可失真的轨迹顺序、embedding 机制、Lean 形式化证明、Rust 语言实现与缓存/共享优化、以及与 KATch、McNetKAT、Storm 的对标实验。

**📊 数据集**

实验数据集主要为合成网络拓扑：Fat‑tree 与 Jellyfish 数据中心拓扑，保持相同主机数、交换机数及端口数；案例研究中规模扩展至 250 主机；此外还在公开的网络仿真数据（如 Amazon quasi‑random 图）上进行验证。

**📈 对比分析**

评估方法：与专门针对 Boolean 可达性的 KATch 进行基准对比；与概率分析工具 McNetKAT 与 Storm 进行同类实验。实验结果表明：① 在 Boolean 可达性场景下，该框架与 KATch 的性能相当；② 在概率、时延、带宽等定量分析场景中，速度比 McNetKAT 与 Storm 快数十到数百倍；③ 在 Fat‑tree 与 Jellyfish 的多目标分析中，能够一次性给出完整 Pareto 前沿及其最短路径，体现了系统的实用价值。

**⚠️ 局限性**

局限性：① 需要半环满足 ω‑连续与可计算性，且对前沿半环要求成本代数是有界的，因而不支持如期望累积的全概率多目标分析；② trace‑carrying Pareto 半环的顺序设计导致某些等价路径会被合并，可能隐藏部分多样化路径信息；③ 对大规模真实网络的可扩展性仍需进一步验证；④ 当前实现仅覆盖可达性、时延、带宽与概率等离散指标，未涵盖如能耗、可重配置性等更复杂的度量。

---

## 176. Adaptive Ad Load Design for Sponsored Search Markets: Evidence, Theory, and Deployment

**arXiv ID:** 2607.14418 | [PDF](https://arxiv.org/pdf/2607.14418v1)

**作者:** Mohammad Rashid `[一作]` (University of Washington), Hema Yoganarasimhan `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在Android应用商店上进行大规模随机实验，研究赞助搜索广告槽位数量变化对收入与转化率的影响，并基于实验结果设计并部署了可自适应调节广告槽位的算法；

**💡 创新点**

创新点在于提出了“探索增强的局部自适应广告槽位算法（e‑LAAL）”，该算法结合局部滑动窗口估计与并行静态探索臂，实现了对查询级别、时间动态的非参数自适应，并给出了有限时间动态风险上界；

**🔧 技术方法**

技术核心包括：随机实验设计、滑动窗口估计、局部贪婪决策、并行静态探索臂、对多目标（收入、转化）进行标量化奖励、离散决策空间下的非参数上下界理论；

**📊 数据集**

使用的数据集为某亚洲国家Android应用商店的搜索日志，包含约5,057,952名用户、26,326,936次搜索、377,425次付费安装、15,514,923次自然安装，实验持续66天，随后部署22天，覆盖22.3M用户；

**📈 对比分析**

与静态广告槽位基准（C、T2–T6）及历史/部署时均衡静态规则相比，e‑LAAL在保持相近收入的同时提升约3%转化率，整体收益-转化权衡明显优于所有静态策略，且在查询级别和品牌活跃等动态情景下表现更佳；

**⚠️ 局限性**

局限包括：1）λ的设定基于短期实验估计，未充分反映长期价值；2）滑动窗口长度、阈值等超参数需经验调优；3）仅在单一平台和广告模式下验证，跨平台推广需要进一步验证；4）假设即时结果与静态臂可比的条件性独立性可能不成立；5）对极少查询的稀疏性处理依赖默认槽位，可能影响精度。

---

## 177. Group Testing with Selectable Thresholds

**arXiv ID:** 2607.14448 | [PDF](https://arxiv.org/pdf/2607.14448v1)

**作者:** Trung-Khang Tran `[一作]` (National University of Singapore), Jonathan Scarlett `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了可选阈值组测试（GT-ST）模型，并给出了在无穷大、有限大以及固定阈值下的非适应性检索算法和对应的可实现率与下界。

**💡 创新点**

创新点在于允许每个测试自行选择阈值，首次证明在阈值可选时速率可逼近计数上界，且通过两阶段组合算法实现，给出了通用的下界分析。

**🔧 技术方法**

使用了Bernoulli和近常列权（NCC）设计、COMP和DD子算法、Poisson近似、Chernoff、McDiarmid以及FKG不等式等概率与信息论工具。

**📊 数据集**

本研究为理论分析，不依赖任何实验数据集，所有结论均基于随机设计和组合先验的数学推导。

**📈 对比分析**

与传统阈值组测试和无阈值组测试相比，本文在γ_max→∞时实现速率1，在固定γ_max时提供与上界匹配的下界，利用两个阈值的组合可提升性能。

**⚠️ 局限性**

主要局限在于固定阈值下稀疏区间仍存在上界与下界差距；可选阈值在信息理论极限上收益有限；分析依赖近常行权重和有限阈值数的假设。

---

## 178. LLM Evaluators are Biased across Languages

**arXiv ID:** 2607.14480 | [PDF](https://arxiv.org/pdf/2607.14480v1)

**作者:** Ej Zhou `[一作]` (University of Cambridge), Anna Korhonen `[通讯]` (University of Cambridge)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统评估了多语言评估器（LLM-as-a-Judge 与奖励模型）在23种语言中对语义相同内容的分数偏差，并揭示该偏差与语言资源水平和模型不确定性相关；

**💡 创新点**

创新点在于首次量化并比较不同模型、架构、训练范式下的跨语言分数不一致性，证明传统的 pairwise accuracy 指标无法捕捉此类偏差，并通过回归分析剖析偏差机制；

**🔧 技术方法**

采用多语言 LLM 评估、负对数似然（NLL）与语义熵等不确定性度量、线性回归与方差分解分析等技术；

**📊 数据集**

使用 RewardBench 与其多语言扩展 M-RewardBench 的 23 语种平行数据；

**📈 对比分析**

通过平均分、z‑标准化分数、pairwise accuracy 与基于全局阈值的接受率差距等指标比较，多语言评估器即使 pairwise accuracy 超过 90% 仍存在高达 43% 的接受率差异；

**⚠️ 局限性**

局限性包括需要语言识别（易受代码混合攻击）、单一阈值校准难以完全消除偏差、且偏差涉及语言结构层面，需更系统的多语言校准与训练改进。

---

## 179. Interleaved Noise Injection Improves Clean, Corrupted, and OOD Performance

**arXiv ID:** 2607.14466 | [PDF](https://arxiv.org/pdf/2607.14466v1)

**作者:** Matt L. Wiemann `[一作]` (Princeton University), Andrew K. Saydjari `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在深度学习训练中使用间歇性噪声注入（即在清洁数据与噪声数据之间交替训练）的效果。

**💡 创新点**

创新点在于提出了“交错噪声日程”，并从理论上证明冲击噪声相当于Jacobian正则化、Gaussian噪声相当于曲率惩罚；与传统单向或线性衰减的噪声日程相比，交错日程能更好地帮助优化器跳出局部最优、提高清洁、腐败与OOD的性能。

**🔧 技术方法**

主要技术包括：冲击噪声与Gaussian噪声的注入、二阶泰勒展开得到的正则化解释、梯度范数稳定化（对噪声梯度进行比例缩放）、与现有正则化方法（Cutout、SAM、VIPAug、AugMix）组合。

**📊 数据集**

实验数据集为：CIFAR-100、CIFAR-100-C、ImageNet-1k、ImageNet-C、ImageNet-R；模型涵盖WideResNet、ResNet50、ViT 等。

**📈 对比分析**

与标准训练、Curriculum/anti-Curriculum、纯Gaussian/Impulse噪声、Cutout、SAM、VIPAug、AugMix 等做对比。结果显示：交错噪声在 Clean、mCE、结构mCE、ImageNet-R 等指标上均显著提升，尤其与 AugMix+噪声组合效果最佳。性能提升可达 Clean 准确率提升约1.4%、mCE 下降约16%、结构mCE 下降约13%。

**⚠️ 局限性**

限制：需手动设定噪声频率 P 与噪声比例 σ，尽管对 P 的鲁棒性较好，但极端值仍可能影响清洁精度；噪声类型需根据模型架构匹配（Conv 适合冲击噪声，ViT 适合 Gaussian），否则效果下降；额外计算主要是梯度范数追踪，成本极低，但在大规模训练时仍需监控。

---

## 180. Step-Level Preference Learning for Generative Agents in Social Simulations

**arXiv ID:** 2607.14485 | [PDF](https://arxiv.org/pdf/2607.14485v1)

**作者:** Wenchang Gao `[一作]` (Shanghai Qi Zhi Institute), Tianxing He `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种交互式界面用于在生成式代理的每个决策步骤收集人类偏好监督，并基于此构建了57,239条步级偏好数据；

**💡 创新点**

创新点在于将人类偏好从轨迹级迁移到细粒度决策级，提供了首个针对生成式代理内部模块的步级偏好数据集，并证明步级偏好学习能显著提升多代理社会模拟的协调性与角色完成度；

**🔧 技术方法**

采用生成式代理架构（包含记忆检索、反思、规划、行动和对话模块），利用GPT‑4o等大语言模型生成候选输出，并通过监督微调(SFT)和直接偏好优化(DPO)实现步级偏好学习；

**📊 数据集**

数据集来源于30个设计的社会事件，每个事件包含4–8名目标多样的代理，收集了不同模块（规划、重要性评分、反思问题、反思、行动、对话）的偏好对；

**📈 对比分析**

在10个保留的社会事件上对开源模型（Qwen2.5‑7B/14B、Llama‑3.1‑8B）和专有模型进行对比，SFT/DPO提升了地点遵循、时间遵循、角色履行、需求一致性与交互质量等五大指标，步级偏好学习显著缩小了与人类基准的差距；

**⚠️ 局限性**

局限包括：数据覆盖有限的场景与少量注释者；每步仅有单一注释者，缺乏互评一致性评估；评价主要依赖LLM评审，可能带来偏差；模块分布不均，可能导致偏好信号偏向高频模块。

---

## 181. Scheduler-Agnostic Adaptive-FEC for MPQUIC: Field Evaluation over Commercial Cellular Paths

**arXiv ID:** 2607.14482 | [PDF](https://arxiv.org/pdf/2607.14482v1)

**作者:** Takuma Tsubaki `[一作]` (NTT), Takuya Tojo `[通讯]` (NTT)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种可在现有 MPQUIC 调度器上直接增添的调度器无关自适应 FEC 控制方案，利用 QUIC 失真检测信号和现成的 Reed–Solomon 编码库对数据块进行动态冗余生成；

**💡 创新点**

创新点在于：①不需要重构 MPQUIC 调度器，保持原有路径选择逻辑；②利用 QUIC 本身的失真检测实现路径失真率估计；③采用安全系数对预期失真量做轻量级调节，简化实现；

**🔧 技术方法**

技术包括 QUIC Datagram、MPQUIC 多路径调度、QUIC 的丢包检测机制、Reed–Solomon 纠删码以及基于块的 FEC 控制；

**📊 数据集**

使用真实车载车辆在东京市内 15 分钟环线内通过 3 条商业 LTE/5G 路径进行现场实验，负载 4.0 Mbps；

**📈 对比分析**

与无 FEC（No‑FEC）方案比较，平均单向延迟从 103 ms 降至 70.8 ms，95% 分位延迟从 281.2 ms 降至 142.3 ms，丢包率从 1.7% 降至 0.8%，平均编码率为 0.94，体现低冗余下的可靠性和时延提升；

**⚠️ 局限性**

局限性包括：仅在单一流量负载下评估；未与相同冗余开销的基线方案做严格对比；未对安全系数 α、块大小 k 进行系统敏感性分析；实验环境为固定路线和车速，未涵盖更广泛的移动场景。

---

## 182. BioTIER: A Refusal Benchmark for Targeted Biological Risk Mitigation

**arXiv ID:** 2607.14479 | [PDF](https://arxiv.org/pdf/2607.14479v1)

**作者:** Eleanor M. Marshall `[一作]` (SecureBio), Jasper Götting `[通讯]` (SecureBio)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了 BioTIER 这一针对大型语言模型（LLM）的生物安全评估基准，旨在区分极高风险的生物信息、双重用途研究与一般生物学知识，并对模型的拒绝与允许行为进行系统评估。

**💡 创新点**

创新点包括：① 引入三层风险分类（CA、BD、RB）以实现细粒度的安全与科研效能平衡；② 通过 542 条专家手写、精细标注的提示构建了一个可公开评测的标注库；③ 结合拒绝/许可两部分评测框架，同时开展长期追踪与多模型集成分析，揭示模型间的拒绝一致性与盲点。

**🔧 技术方法**

使用的技术与方法：人工专家撰写与多轮审核提示；LLM 辅助的覆盖率与多维特征分析；二进制拒绝/回答评分体系；Inspect AI orchestrator 自动化评测；主成分分析与集成提问实验；元数据结构化与可视化。

**📊 数据集**

数据集：542 条专家精心设计的提示，涵盖 CA、BD、RB 三个风险集；每条提示附带风险标签、主题、代理类型、Select Agent 子标签、技术难度、潜在主体技能等 12 维元数据；该数据集已公开可用于模型评测。

**📈 对比分析**

比较方法：对每个模型在 398 条拒绝性提示（CA+BD）上计算拒绝率，对 144 条许可性提示（RB）上计算回答率；对 52 款前沿 LLM 进行 10 次实验求平均并给出 95% 置信区间。结果显示拒绝率从 5.6% 至 96.4% 变化，顶级模型如 Claude Sonnet 4.6 达到 96.4%；许可率大多数模型在 98% 以上，最极端模型如 Claude Sonnet 4.6 在 76.3%。

**⚠️ 局限性**

局限性：① 仅评估二进制拒绝/回答行为，未考量答案质量或可操作性；② 评测不包含多轮对话、翻译、图像等多模态情境；③ 评测结果易随时间漂移，需持续跟踪；④ 专家标注可能存在主观偏差，边缘案例仍难以界定；⑤ 公开基准可能导致模型针对性优化与信息泄漏风险。

---

## 183. One-Shot Generative Design for Disordered Metamaterials via Self-Organizing Neural Cellular Automata

**arXiv ID:** 2607.14475 | [PDF](https://arxiv.org/pdf/2607.14475v1)

**作者:** Yujie Xiang `[一作]` (Carnegie Mellon University), Liwei Wang `[通讯]` (Carnegie Mellon University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于神经细胞自动机（NCA）的单样本生成框架，用以设计无序超材料的微结构，并在多尺度优化中实现机械隐形罩的设计。

**💡 创新点**

创新点在于：①只需一个训练模板即可学习到全局自组织规则；②通过修改局部感知算子实现无须再训练即可控制方向、各向异性、尺度和厚度；③将NCA视作可泛化的偏微分方程，使得模型可跨网格、跨曲面、跨尺寸转移；④实现异步生长和自我修复等自适应功能。

**🔧 技术方法**

主要技术包括：神经细胞自动机、Sliced Optimal Transport 之风格损失的一次性训练、梯度旋转/拉伸/剪切控制、PDE 视角下的局部算子重定义、变分自编码器（VAE）在多尺度优化中构造潜在空间约束、MMA 优化和自适应异步生长策略。

**📊 数据集**

使用的“数据集”为若干种自然无序微结构的单张样本（叶脉、骨髓孔、细菌生物膜、蜘蛛丝、金属断面等），不构成传统意义上的大规模数据集。

**📈 对比分析**

与传统规则基方法（Voronoi、GFRF、WFC）和深度生成模型（VAE、GAN、Diffusion）相比，NCA 在仅使用一张样本的情况下即可重现和扩展多种微结构；在机械隐形罩案例中，NCA 生成的微结构在 1000×1000 分辨率下将相对误差从 129% 降至 16%（相较于未隐形罩的 129%），并在不同分辨率下保持较高一致性和可扩展性。

**⚠️ 局限性**

局限性包括：目前仅实现二维结构，线性弹性假设；未实现直接的逆向属性条件；对非线性、三维微结构的适用性仍待扩展；对非常大尺寸域的数值稳定性和收敛性未充分评估。

---

## 184. Semitotal domination in unit disk graphs

**arXiv ID:** 2607.14467 | [PDF](https://arxiv.org/pdf/2607.14467v1)

**作者:** Mingjun Liu `[一作]` (Zhengzhou College of Finance and Economics), Weiping Shang `[通讯]` (Zhengzhou University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在图形输入模型下针对单位圆图的最小半总支配集问题的确定性5-近似算法。

**💡 创新点**

相比之前5.75-近似且时间复杂度为O(n³)的算法，本工作将近似比降至5，同时将运行时间提升到O(n²)（线性时间）。

**🔧 技术方法**

核心技术包括利用BFS层划分、贪心构造最大独立集并证明其满足半总支配条件，并结合单位圆图上最大独立集与支配数的关系。

**📊 数据集**

该算法在理论层面适用于所有单位圆图，没有使用具体实验数据集。

**📈 对比分析**

通过理论分析与已有结果比较，证明该算法在近似比和时间复杂度上均优于先前最优已知方法，且实现简单高效。

**⚠️ 局限性**

局限性在于仍然基于图形输入模型，近似比为5且未达到最优；对非单位圆图不适用。

---

## 185. Simply Typed Reverse-Mode AD with Variants: Denotational Correctness via Idempotent Completion

**arXiv ID:** 2607.14453 | [PDF](https://arxiv.org/pdf/2607.14453v1)

**作者:** Fernando Lucatelli Nunes `[一作]` (University of Coimbra), Matthijs Vákár `[通讯]` (Utrecht University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过Karoubi完备化和幂等分解，将逆向自动微分中的变体（variant）问题转化为普通的非依赖类型语义，构造了一个双笛卡尔闭的目标语义，并证明该语义与传统的依赖族语义等价，从而得到唯一的环境扩展的反向导数。

**💡 创新点**

创新点在于：① 用幂等分解恢复缺失的余和，消除对变体的显式语义限制；② 证明常数族完备化与完整族完备化在满足共收缩器存在性时等价；③ 通过半函子（semifunctor）转普通函子，将依赖语义消除为普通程序；④ 给出唯一的投影兼容环境反向导数公式。

**🔧 技术方法**

使用的技术包括：范畴论的Karoubi完备化、Cauchy完备性、共同回收器（common retract host）的构造、双笛卡尔闭语义的生成、半函子到函子的Hoofman–Moerdijk嵌入、以及逆向CHAD的语义驱动变换。

**📊 数据集**

无数据集。研究完全是理论性和符号性，没有使用实验数据。

**📈 对比分析**

论文未给出实验比较或性能评估，而是通过范畴论证明等价性和正确性。若与现有的表达式CHAD比较，可视为在语义层面取得完全一致，且不需要额外的变体语义实现。

**⚠️ 局限性**

局限性：仅适用于有限乘积和变体构造的欧几里得基类型；对更一般的类型构造（如递归类型、无限集合等）未作讨论；实现复杂度较高，且对实现细节（如投影调用）依赖于具体编程语言的底层支持。

---

## 186. Cotton-SF YOLO: Learning Structural and Frequency Cues for Early Cotton Square Detection in Complex Field Environments

**arXiv ID:** 2607.14445 | [PDF](https://arxiv.org/pdf/2607.14445v1)

**作者:** Chengjia Zhang `[一作]` (Xichang University), Liting Gao `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于YOLO26m提出了Cotton‑SF YOLO，用于在复杂田间环境下对早期棉花棉絮进行自动检测。

**💡 创新点**

创新点在于引入两大模块：Dynamic Snake Structure Perception Module（DSSPM）通过动态蛇形卷积增强棉絮细小、弯曲边界特征；Frequency‑Domain Feature Modulation Module（FDFMM）利用FFT在频域对特征进行通道加权，提升纹理辨别力；两者协同提升小目标检测效果。

**🔧 技术方法**

技术手段包括YOLO26m框架、动态蛇形卷积（DSConv）、FFT频域加权、C2f结构、以及标准的训练技巧（Progressive Loss、STAL、MuSGD）。

**📊 数据集**

使用自行采集的704张棉絮图像数据集（覆盖不同生育期与光照条件），并通过T‑Rex Label手工标注得到棉絮框框。

**📈 对比分析**

与YOLOv5、v8、v9、v11、v12以及YOLO26m基线在同一数据集上进行对比，Cotton‑SF YOLO在mAP_50、mAP_50:95、召回率上分别提升1.25%、3.45%、2.96%，参数量从21.90M降至21.16M，表现最佳。

**⚠️ 局限性**

局限性包括：仅在地面相机场景验证；尚未在无人机或移动机器人平台部署；缺乏多站点多年份的验证；对棉絮数量与产量关系的建模仍待进一步研究。

---

## 187. Tactile: Giving Computer-Using Agents Hands and Feet

**arXiv ID:** 2607.14443 | [PDF](https://arxiv.org/pdf/2607.14443v1)

**作者:** Yong Liu `[一作]` (Shanghai Jiao Tong University), Zhanpeng Shi `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个可插拔的桌面操作工具层，整合可访问性语义、文本定位和视觉回退为动作驱动的接口状态，提供语义动作、坐标点击和可验证结果。

**💡 创新点**

提出了动作驱动接口状态和可访问性优先的操作梯度，将观测、定位、执行、验证四步拆分；通过完整跟踪与可重放，使桌面交互更可解释、可审计。

**🔧 技术方法**

使用 macOS Accessibility API、UIAutomation、OCR（Vision/AppleVision）、屏幕捕获、CGEvent 输入模拟、Swift/Python 交互、MCP 工具协议、候选压缩与排名、坐标归一化和完整日志记录等技术。

**📊 数据集**

在 macOSWorld 样本（约 96 任务）上评估，结合 Codex、Claude Code、OpenCode、Goose 等语言模型，并将任务拆分为 AX 适配任务与有限 AX 任务。

**📈 对比分析**

采用 Success@100（≤100 步成功率）作为指标；加入工具层后 Codex 成功率从 41.1% 提升至 50.0%，AX 适配任务从 45.2% 提升至 55.3%；96 任务横向子集各模型均提升，最高约 +10%，显示工具层显著提升多模型桌面任务成功率。

**⚠️ 局限性**

可访问性信息不完整、噪声大；跨平台差异导致 Windows 支持不足；大规模动态 UI 树导致候选压缩可能遗漏关键元素；OCR/视觉回退失败率高；验证能力有限，无法检测某些无反馈操作。

---

## 188. CausalGraphX: A Counterfactual Graph Neural Network Framework for Explainable Systemic Risk Assessment

**arXiv ID:** 2607.14416 | [PDF](https://arxiv.org/pdf/2607.14416v1)

**作者:** Rabimba Karanjai `[一作]` (University of Houston), Weidong Shi `[通讯]` (University of Houston)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6215c339-3735-4be3-8a07-5bbb7004712d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可解释的因果图神经网络框架（CausalGraphX），能够预测银行违约并给出基于最小干预的反事实解释，帮助监管机构制定针对性的政策。

**💡 创新点**

创新点在于：① 在GAT中加入对抗因果正则化，使嵌入对混杂变量不敏感；② 设计稀疏且可行的反事实生成器，通过梯度优化得到最小化干预量的解释；③ 通过对比实验验证了因果正则化在提升预测与解释质量方面的有效性。

**🔧 技术方法**

使用技术包括：Graph Attention Network（GAT）+ 对抗正则化（adversarial discriminator）+ 反事实优化（梯度下降 + 约束项）+ PyTorch、PyTorch‑Geometric 实现。

**📊 数据集**

实验数据集为合成金融网络：Erdős–Rényi 与 Barabási–Albert 结构，节点特征 12 维金融指标，边权为 log‑normal 分布；违约标签由修改后的 Eisenberg‑Noe 模拟产生。

**📈 对比分析**

与 Logistic Regression、DebtRank、GCN、标准 GAT 进行对比；在 AUC、F1、PR‑AUC、MCC 等指标上，CausalGraphX 均取得最高分；反事实解释的有效率从 97.3% 提升至 99.1%，稀疏度和因果合理性也大幅提升（≈45%）。

**⚠️ 局限性**

局限性包括：① 仅在合成/半合成数据上验证，缺乏真实历史网络的实证；② 未考虑时间动态和多层次金融暴露；③ 对超参数（如 λ、γ1、γ2）的敏感性尚需进一步研究。

---

## 189. G$^2$SR: Geometric Methods for Fast and Memory-Efficient Gaussian-based Surface Reconstruction

**arXiv ID:** 2607.14470 | [PDF](https://arxiv.org/pdf/2607.14470v1)

**作者:** Dasong Gao `[一作]` (Massachusetts Institute of Technology), Sertac Karaman `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种基于二维高斯斑点检测与三维三角化的少视角表面重建框架G^2SR；

**💡 创新点**

创新点在于将重建任务拆分为轻量化的二维斑点检测/跟踪子任务和解析化的三维三角化后端，从而实现高精度、低内存的在线重建；

**🔧 技术方法**

采用轻量化CNN前端进行2D斑点检测、NeuFlowV2光流跟踪以及基于Hellinger距离的Gauss‑Newton三角化；

**📊 数据集**

在Replica、ScanNet与DTU三大数据集上进行实验；

**📈 对比分析**

与MVSplat、FreeSplat、MonoSplat、C3G、SurfelSplat等SOTA端到端方法比较，G^2SR在两视角/三视角下的深度误差与网格Chamfer距离均相当或更优，同时推理速度提升多倍、显存占用降低十数倍；

**⚠️ 局限性**

局限性包括对纹理细节的重建不如端到端方法精细、部分视角缺失导致覆盖率下降、需要先检测到足够多的2D斑点以保证三角化成功。

---

## 190. AI-Conducted Interviews in Empirical Software Engineering: An Experience Report

**arXiv ID:** 2607.14452 | [PDF](https://arxiv.org/pdf/2607.14452v1)

**作者:** Rohit Gheyi `[一作]` (Federal University of Campina Grande), Mirko Perkusich `[通讯]` (Federal University of Campina Grande)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计并部署了基于 MyGPT 的自助式语音半结构化访谈流程，在两个软件工程研究（重构实践与 Scrum 与生成式 AI 的使用）中收集访谈稿和问卷，并对流程可行性、参与者体验及方法局限性进行了经验性评估。

**💡 创新点**

创新点在于首次系统性评估 AI 主持访谈的可操作性与接受度，并提供了结构化审计与参与者感知数据收集的完整工作流程，为后续研究提供可复制的实践与方法论。

**🔧 技术方法**

使用技术：OpenAI 的 MyGPT 自定义对话代理、语音交互与自然语言切换、生成结构化访谈稿、问卷调查与手工审计；数据保存在受控的 Google Forms 与 ChatGPT 平台。

**📊 数据集**

使用数据集：66份有效访谈稿（61/66 格式符合预期）与对应问卷，涵盖重构与 Scrum+生成式 AI 两个实验组，访谈时长约 10–15 分钟。

**📈 对比分析**

对比与评估方法：通过参与者自评（满意度、舒适度、自然度等）和结构化审计（稿件格式一致性、语言一致性等）来评估流程，可得 90% 以上参与者满意度、92.4% 访谈稿符合预期格式，显示流程在时间与灵活性上的优势，但未进行与传统人类访谈的定量质量对比。

**⚠️ 局限性**

局限性：未验证访谈内容的深度、准确性与可比性；仅适用于短访谈；受平台与模型差异、语言切换错误、隐私风险及需要人工校验等因素限制；样本来自单一国家，缺乏多语言与敏感主题的泛化性。

---

## 191. From Product-Centred Retrieval to Experience-Led Commerce:Twelve Candidate Design Principles for Fashion E-Commerce User Experience

**arXiv ID:** 2607.14429 | [PDF](https://arxiv.org/pdf/2607.14429v1)

**作者:** Nafiul I. Khan `[一作]`, Rafflesia Khan `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了十二条面向体验的时尚电商设计原则，并在原型VogueDrop中实现

**💡 创新点**

创新点在于将多入口、关系探索、偏好主权、证据对应、可行性评估等维度纳入统一交互架构，形成Experience‑Led Commerce（ELC）

**🔧 技术方法**

主要技术是交互设计与原型实现，结合混合式人机交互、可解释性推荐与代理交互协议

**📊 数据集**

未使用公开数据集，依赖设计实验与后续用户研究收集数据

**📈 对比分析**

对比方案包括传统页面漏斗、增强式漏斗及ELC原型，预期能提升匹配准确率与可行性检测，但目前尚未得到实验结果

**⚠️ 局限性**

局限在于尚未完成经验验证与性能评估，原则的有效性和跨域通用性仍需进一步实证

---

## 192. HyperShadow: A Benchmark for Detecting 3D Projections of Higher-Dimensional Spatial Objects

**arXiv ID:** 2607.14419 | [PDF](https://arxiv.org/pdf/2607.14419v1)

**作者:** Akshay Sasi `[一作]` `[通讯]` (Independent Researcher), Akshay Sasi (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了HyperShadow基准，评估单帧或短序列点云是否为高维（4–6维）刚体投影的“影子”，并给出零参数刚性证明器；

**💡 创新点**

核心创新在于区分投影阴影与本地3D形状的几何与拓扑特征，证明内在维度估计无法解决此问题，并引入可证实3D刚性不一致的统计量；

**🔧 技术方法**

利用自研小型PointNet-lite网络、梯度提升树+手工几何特征、持久同调+GBT以及零参数Kabsch残差统计等多种技术；

**📊 数据集**

使用合成数据集HyperShadow，包含10,800个静态点云和1,800个16帧序列，涵盖11种高维形状和7种3D形状，且按四个腐蚀层划分；

**📈 对比分析**

与传统内在维度估计、几何特征+GBT、持久同调+GBT等基线对比，静态任务最高准确率达96.2%（PointNet-lite），动态任务零参数刚性证明器AUROC 0.982；

**⚠️ 局限性**

局限包括仅在模拟环境下验证，无法直接迁移至真实传感器；需要点对应关系；数据集覆盖范围有限，且正检验仅证明不符合3D刚性模型，不能证明物理上存在额外维度。

---

## 193. LATTICE: Graph Self-Supervised Learning for Multimodal Spatial Omics Integration

**arXiv ID:** 2607.14410 | [PDF](https://arxiv.org/pdf/2607.14410v1)

**作者:** Jagan Mohan Reddy Dwarampudi `[一作]` (University of Houston), Tania Banerjee `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为LATTICE的图自监督学习框架，用于整合多模态空间组学数据并学习点位级表征

**💡 创新点**

创新点在于将掩码重建、跨模态对齐和空间平滑三种自监督目标结合，在统一的空间kNN图上同时捕获空间结构与多模态一致性

**🔧 技术方法**

使用图神经网络（TransformerConv）、掩码重建、对比学习对齐、空间kNN图构建以及多模态特征拼接等技术

**📊 数据集**

采用匿名临床合作方的11个黑色素瘤样本，包含Visium RNA、scMultiome RNA/ATAC、空间ATAC和空间CUT&Tag共五模态数据

**📈 对比分析**

与单模RNA基线（GraphST、STAGATE等）以及其他多模态基线比较，LATTICE在多模态效用（MUS）和空间连贯性指标上表现最佳，尽管对RNA参考的ARI/NMI并非单调提升

**⚠️ 局限性**

局限包括对RNA参考的依赖导致ARI/NMI不一定提升、缺乏外部基准验证以及对训练超参数和遮掩比例的敏感性

---

## 194. Manufactured Divisiveness: Decomposing the Hostile Content of Seven Social Media Influence Operations

**arXiv ID:** 2607.14491 | [PDF](https://arxiv.org/pdf/2607.14491v1)

**作者:** Emilio Ferrara `[一作]` `[通讯]` (University of Southern California), Emilio Ferrara (University of Southern California)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了七个政府属性信息操作的推文，使用两阶段LLM门控与规则式分类，将广义敌意拆分为身份仇恨、党派挑衅和地缘政治诽谤，并量化各类构成比例。

**💡 创新点**

首次将仇恨与更宽泛的敌意/分裂性语言区分，并提出可审计的规则式构造分类，揭示不同操作在身份、党派、地缘政治维度上的显著差异，指出单一“仇恨率”会高估仇恨语料。

**🔧 技术方法**

采用两提示一致性的LLM门控（Qwen2.5-7B）识别敌意，再用冻结的11维度特征（目标、叙事、情感等）结合规则将正例拆分；同时用LLM情感与道德词典交叉验证。

**📊 数据集**

Twitter信息操作档案中的七个政府属性活动，共计约2510万条推文、8,275个账号。

**📈 对比分析**

门控对人类金标的Cohen κ=0.82、精度0.96；规则对专家的κ=0.52；在边界增强集上，模型与专家对齐最高κ≈0.71；整体显示广义门控与身份仇恨之间相差约2–5倍。

**⚠️ 局限性**

规则依赖LLM生成的特征，未独立验证目标属性；对分界的主观性导致不稳定；仅针对七个特定操作，结果可能不具普适性；仅提供标签计数，无法直接复现个别条目。

---

## 195. Alipay-PIBench: A Realistic Payment Integration Benchmark for Coding Agents

**arXiv ID:** 2607.14573 | [PDF](https://arxiv.org/pdf/2607.14573v1)

**作者:** Shiyu Ying `[一作]` (Ant Group), Lin Zhu `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Alipay主导的仓库级支付集成基准PIBench，采用产品导向任务构造、基础与高级分阶段场景、基于规则的评估与LLM辅助判定，系统评估多模型在支付集成中的表现并验证结构化支付指导的效益。

**💡 创新点**

创新点包括①Alipay主导的仓库级支付集成基准；②产品导向任务构造将支付产品、业务场景、仓库代码与自然语言指令统一；③将支付集成拆分为基础功能完成与高级风险硬化两阶段；④基于场景规则推导确定性检查与LLM辅助评估，保证评估可追溯；⑤通过配对干预实验衡量支付指导工具的实际价值。

**🔧 技术方法**

采用静态检查、单元/集成/E2E测试、Deterministic checks、LLM辅助评估等技术，并利用Alipay沙箱环境与Mock数据进行功能与安全验证。

**📊 数据集**

使用18个任务实例（9个Alipay支付产品×2场景），每个实例包含真实业务仓库代码、自然语言指令、基于规则的评估脚本，形成基准数据集。

**📈 对比分析**

对6个模型（Claude、GLM、Kimi、DeepSeek、MiniMax等）在无/有支付指导两种条件下进行paired实验，计算RPR。无指导下平均RPR在68.6%–91.4%之间，指导提升平均10.31个百分点（基础任务+11.27，进阶任务+9.35），并且在输出效率上平均降低约14%（输出Token/RPR）。

**⚠️ 局限性**

局限性包括：仅覆盖Alipay产品与18个仓库，未覆盖更广泛的支付平台或长周期流程；评估依赖沙箱/Mock，可能无法触及所有异常情形；LLM辅助评估主观性导致判定不稳定；模型规模/配置差异可能影响结果，未充分探究模型内部机制。

---

## 196. Almost Navigable Graphs

**arXiv ID:** 2607.14564 | [PDF](https://arxiv.org/pdf/2607.14564v1)

**作者:** Pratyush Avi `[一作]`, Christopher Musco `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了γ‑几乎可导航（γ-almost navigable）图，显著降低了近似最近邻搜索（ANN）所需图的稀疏度，并给出了近线性构造算法。

**💡 创新点**

核心创新是将完整可导航性（全局满足每个点对的贪婪步向前）放宽为每个点仅对γ比例的目标点具备向前边，证明任何数据集都可构造仅O(n)条边、平均度数O(1/(1-γ))的图，并给出随机化近线性构造方法。

**🔧 技术方法**

技术手段包括：基于团切分（clique peeling）的构造思想、随机抽样估计“拥有集”大小以决定是否继续迭代、Chernoff 近似、以及对图的可导航性约束的分析和证明。

**📊 数据集**

在实验中使用了8个标准数据集：MNIST、Fashion‑MNIST、COCO‑i2i、Glove25、Yandex‑DEEP、BIGANN、Facebook SimSearchNet++、Microsoft SPACEV1B，涵盖从数千到数十亿点的规模。

**📈 对比分析**

通过与完全可导航图（γ=1）比较，评估了平均度数、记忆占用、距离计算次数和召回率（Recall@k）。结果显示：在给定召回目标下，γ‑几乎可导航图的距离计算量平均降低约35–47%，空间占用降低约50–55%；在大多数数据集上，甚至能在更少距离计算的前提下达到或超过完整可导航图的召回水平。

**⚠️ 局限性**

局限性：1）理论上不再保证全局贪婪搜索正确性；已构造实例表明，近似可导航图对极少数查询（≈1−γ）无法成功。2）对γ取值的选择依赖数据分布，缺乏统一的自动调参方法。

---

## 197. Towards an Intention Abstraction Layer for Autonomous Industrial Systems

**arXiv ID:** 2607.14553 | [PDF](https://arxiv.org/pdf/2607.14553v1)

**作者:** Artan Markaj `[一作]` (Eurogate GmbH and Co. KGaA), Felix Gehlhoff `[通讯]` (Helmut Schmidt University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个名为 Intention Abstraction Layer（IAL）的中间件，用于在工业自治系统运行时以正式、持久、可解释的形式捕获、管理并调和各自主导系统的意图。

**💡 创新点**

创新点在于将意图作为运行时第一类对象进行统一管理，并通过大型语言模型（LLM）与 OWL 本体结合实现自然语言意图的结构化解析、即时冲突检测和可解释性解释，从而将行为保证从事后失败分析迁移到执行前的意图级别校验。

**🔧 技术方法**

核心技术包括：大型语言模型（Claude Opus 4.8）用于意图解析；OWL 本体与推理器用于意图结构化、相似度与约束交叉检测；一致性监控器用于冲突检测；透明度 API 用自然语言解释冲突；持久化本体存储用于意图生命周期管理。

**📊 数据集**

使用的“数据集”为人工编写的自然语言意图示例（如“从 14:00–16:00 进行高温批次，消耗约 92% 峰值负荷”与“在 13:00–17:00 期间保持总负荷低于 80% 峰值”），以及对简化工厂负荷模型的模拟运行数据。

**📈 对比分析**

通过在模拟执行层对比两种情形——（a）不使用 IAL，导致负荷超过 80% 上限；（b）使用 IAL，在意图注册时检测到冲突并给出调度方案，最终负荷始终保持在阈值之下。性能表现为冲突检测即时完成且无运行时干预，成功防止了执行层的违规。

**⚠️ 局限性**

局限性包括：LLM 计算不确定且存在延迟，仅限于前向意图解析与冲突检测；未实现对真实 PLC/OPC‑UA 设备的连接；缺少对并发冲突解决的完整机制；后向意图推理（从执行推断意图）尚未实现；缺乏与现有工业标准协议（如 FIPA‑ACL、Contract‑Net）的深度集成。

---

## 198. SafeRelBench: A Spatial-Relation-Aware Benchmark for Process-Level Safety in VLM-Driven Embodied Agents

**arXiv ID:** 2607.14543 | [PDF](https://arxiv.org/pdf/2607.14543v1)

**作者:** Huaigang Yang `[一作]` (Beijing University of Posts and Telecommunications), Zhaofeng He `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的空间关系感知过程级安全基准，用于评估视觉-语言模型驱动的具身代理在家庭环境中的安全执行表现。

**💡 创新点**

创新点在于将安全约束从静态标签转化为过程级的空间关系预条件，设计了包含支持、容纳和接近三类空间关系的507个可执行任务样本，并配备匹配的非空间对照样本，能够明确区分因空间关系导致的安全失败。

**🔧 技术方法**

采用基于规则的评估器结合LLM判定器进行任务完成率与安全成功率的双重测量，使用VLM驱动的行动决策模块，输入图像、指令、物体属性等信息，输出结构化JSON动作；同时在实验中对多种提示设计进行消融研究。

**📊 数据集**

数据集为自构的Benchmark —— …（以下简称	exttt{SpatialSafeBench}），由507个可执行家庭操作任务组成，其中248个为空间关系样本，259个为非空间对照样本，涵盖支持、容纳和接近的九种风险类型。

**📈 对比分析**

实验比较了七个VLM驱动的代理（四个开源模型和三个闭源模型）在空间关系设置与非空间对照下的任务完成率(SR)、安全成功率(SSR)及安全召回率(SRec)。在空间关系任务中，SR平均下降至0.52–0.73，SSR显著下降至0.16–0.40，显示出显著的任务完成与安全执行差距；在非空间对照中，SR与SSR均大幅提升，说明空间关系是导致安全失败的主要因素。

**⚠️ 局限性**

局限性包括：仅在仿真环境中测试，未涵盖真实感知噪声、动力学不确定性、人机交互等实际部署因素；仅考虑支持、容纳、接近三类空间关系，未覆盖所有潜在安全风险；评估依赖规则与LLM判定，可能受安全条件完整性与评估模型偏差影响。

---

## 199. Muse: Representation Geometry of Muon Beyond Normalized Momentum

**arXiv ID:** 2607.14536 | [PDF](https://arxiv.org/pdf/2607.14536v1)

**作者:** Da Chang `[一作]` (Pengcheng Laboratory), Yongxiang Liu `[通讯]` (Pengcheng Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了 Muon 风格优化器中矩阵表示的几何影响，提出 Muse（Muon under Structured rEpresentations）家族，并在 LLaMA2 预训练实验中验证不同表示对训练效果的影响。

**💡 创新点**

创新点是将矩阵表示视为优化器几何的可调参数，揭示短侧（短边）决定了极值、谱形状和收敛常数，并通过一系列结构化表示（Native、Nearest‑Square、Skinny、Vector）构造了表示索引的 Muon 变体。

**🔧 技术方法**

使用了结构化 Frobenius‑等距矩阵化、Polar 更新与 Newton–Schulz 反演、Marchenko–Pastur 谱分析、以及在 LLaMA2 预训练中的自定义学习率调度和固定动量诊断。

**📊 数据集**

采用 LLaMA2‑130M 与 LLaMA2‑600M 语言模型的预训练数据集进行实验。

**📈 对比分析**

通过在同一学习率调度下比较 Native、Square、Skinny、Vector（nSGDM）等表示的最终验证损失和训练曲线，发现 Native 与 Square 取得最优表现，Vector 最差，短侧逐步减小导致性能梯度下降。

**⚠️ 局限性**

局限性在于分析仅限于局部曲率崩塌假设，未考虑负曲率、跨层耦合、自适应表示选择以及与 Shampoo、K‑FAC 等预条件器的交互。

---

## 200. WrAFT: a Modularized Automated Writing Evaluation System for Argumentative Essays

**arXiv ID:** 2607.14524 | [PDF](https://arxiv.org/pdf/2607.14524v1)

**作者:** Adnan Labib `[一作]` (King's College London), Qiao Wang `[通讯]` (Hosei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了WrAFT，一种基于大型语言模型（LLM）的模块化写作评估与反馈系统，专为论证性作文设计；

**💡 创新点**

创新点在于将自动写作评估拆分为评分、表面级反馈和深层级反馈三模块，并通过直接提示与监督微调结合，使用同一系统同时提供高精度评分与多层次、可视化的即时反馈；

**🔧 技术方法**

采用了多款LLM：LLaMA‑3.3‑70B‑Instruct（微调）和GPT‑4o（微调）用于评分，GPT‑4o（直接提示）用于表面级错误纠正，Claude 3.7（直接提示）用于深层级结构与论证反馈；

**📊 数据集**

使用了480篇托福独立写作样本（官方评分），其中120篇用于评分微调，360篇用于评测；表面级评测选取40篇，深层级评测选取40篇（从90篇教师注释数据中抽样）；

**📈 对比分析**

与前置基线模型（同数据集的SOTA微调模型）对比，GPT‑4o微调在评分上实现RMSE 0.44、QWK 0.84、整体准确率93%；表面级反馈必要性与有效性分别达到96.88%与96.14%；深层级宏观反馈有效率93.03%，微观反馈必要且有效率94.69%；

**⚠️ 局限性**

局限性包括：仅基于单一托福数据集，缺乏与其他写作类型的泛化；反馈评估只关注精度而未测召回；未验证对学习成效的影响；LLM幻觉与过度校正问题；商业API成本与隐私风险；缺乏多语言支持。

---

## 201. Adaptive Runge-Kutta Step Control Buys Training Loss, Not Generalization: An Honest Compute-Matched Study of RK-Adam Optimizers

**arXiv ID:** 2607.14516 | [PDF](https://arxiv.org/pdf/2607.14516v1)

**作者:** Akhilesh Gogikar `[一作]` `[通讯]` (Independent Researcher), Akhilesh Gogikar (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对把优化器视作 ODE 近似的想法进行严格实验，构造并评估基于 Bogacki–Shampine 3(2) RK 的 Adam 变体，重点检查其自适应步长控制是否真正发挥作用，并在严格的梯度评估匹配协议下与多种一阶基准比较。

**💡 创新点**

① 引入梯度评估计数（每步梯度数）和 FSAL 正当性校验的计算匹配规范，揭示常见 RK‑优化器报告的“优点”多因计数错误；② 通过完整实验发现现有自适应控制实质上失效，随后修复控制器（加入拒绝分支、错误测度一致等），证明自适应机制能产生真正的 warm‑up‑growth 调度；③ 发现 RK 取梯度平均的隐式正则化效果，即使不比 tuned Adam 更优。

**🔧 技术方法**

使用 Bogacki–Shampine 3(2) 多阶段 RK；Adam、RMSprop、NAdam 等一阶优化器；嵌入式误差估计与自适应步长控制；FSAL 重用；pSGLD 温度测试；固定步长对照；梯度评估计数与全批/小批计数区别；实验中用 MNIST 数据集。

**📊 数据集**

MNIST 全数据集（784–128–10 MLP）用于随机小批实验；1024 示例子集用于全批实验。

**📈 对比分析**

在相同梯度评估预算（即每步梯度数相等）下，与 Adam、AdamW、RMSprop、NAdam 等一阶基准比较。结果显示：① 原始 RK‑Adam 在训练损失上不如 Adam；② 自适应控制无效（步长固定为上限，容差无效）；③ 修复后在全批训练中训练损失降低约 40 倍，但测试准确率不提升；④ 纯 RK 取平均梯度的隐式正则化可略高于 lr‑匹配 Adam，但不及 RMSprop/NAdam。

**⚠️ 局限性**

仅在 MNIST、单一 MLP 结构；实验种子有限（大多为 3 例，部分为 10 例）；未验证在更大规模或不同任务上的表现；修复的自适应控制对小批噪声无效；梯度评估计数未考虑每步额外计算开销；未探究其它 RK 变体或 IMEX 等更复杂方案。

---

## 202. Contextualized Evaluation of Vision Language Models through Dynamic, Multi-turn Interactions

**arXiv ID:** 2607.14499 | [PDF](https://arxiv.org/pdf/2607.14499v1)

**作者:** Yijiang Li `[一作]` (University of California San Diego), Ziang Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了CEDI（Contextualized Evaluations of MLLMs through Dynamic, multi-round Interactions）框架，通过三方交互（模型、自动考官和评分器）在图结构驱动下进行多轮上下文化评估，以更真实地揭示多模态大语言模型（MLLM）的视觉幻觉问题。

**💡 创新点**

创新点主要包括：①将评估转化为动态、情境化的三方交互；②利用场景图（Scene Graph）构造可操作的状态空间，让考官根据情境和对话历史自适应选择普通、追问、对抗和不可回答等四类问题；③开发基于场景图的评分器，能同时量化对象、属性、关系的幻觉与覆盖率，突破传统只关注对象级别的局限。

**🔧 技术方法**

技术方法包括：使用大型语言模型（如 GPT‑4o）作为考官；基于场景图的图遍历和问答生成；对话历史上下文对模型的影响建模；对模型回答进行图解析并计算图编辑距离（GED）与结构相似度（SG ΔCon）等评估指标；以及 VALOR 等逐轮幻觉指标。

**📊 数据集**

数据集：Visual Genome（VG）的人类标注图像、Synthetic Visual Genome（SVG）覆盖 VG、ADE‑20K、COCO 等多源图像集合，用于构造场景图和评估。

**📈 对比分析**

与传统基线（Caption、POPE（二值提问）和 Prompt‑only 预设对话）比较后，CEDI 在所有评估指标（GED、SG ΔCon、VALOR、覆盖率等）上均能发现更多幻觉，且与人工注释的相关性显著提高（例如 GED_hal 与人类计数的 Pearson ρ 0.22‑0.33），说明其评估更贴近真实使用场景。

**⚠️ 局限性**

局限性包括：①高度依赖准确的场景图作为基准，若缺失会影响评估；②考官与评分器的表现受底层 LLM（如 GPT‑4o）能力限制；③评分器主要针对完整对话进行评估，难以细粒度到单轮幻觉；④实验主要聚焦视觉幻觉，扩展到其他多模态任务仍需进一步验证。

---

## 203. CASP: Learning-Augmented Offline Approximation with Verifiable Certificates and Bounded-Loss PAC Guarantees

**arXiv ID:** 2607.14545 | [PDF](https://arxiv.org/pdf/2607.14545v1)

**作者:** Haifeng Li `[一作]` (Central University of Finance and Economics), Mo Hai `[通讯]` (Central University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 CASP（Certificate‑Augmented Solution Pruning）的框架，利用可在多项式时间内验证的证明（certificate）来安全地裁剪 NP‑hard 问题的搜索空间，从而在保证最坏情况性能的前提下加速离线近似算法；并给出了该框架的学习理论、损失界和实验验证。

**💡 创新点**

创新点包括：
1) 逆向信息流——不直接给算法“做什么”，而询问“可以安全忽略哪些部分”，从而消除传统预测-augmented 方法的循环依赖。
2) 引入可验证的安全级别（OPT‑preserving 与 ρ‑approximation‑safe），使得错误预测仅被拒绝，而不会导致解不合法。
3) 通过 verifier 使得损失函数统一受限，从而实现分布无关的 PAC 学习（样本复杂度 O(ε⁻² log K)），而传统方法的损失范围是无界的。
4) 证明了置信过滤（confidence‑filter）在理论上严格优于 min‑combiner，给出了闭式边际。
5) 在多种经典 NP‑hard 任务（Set Cover、Vertex Cover、Facility Location、0/1 Knapsack、Steiner Tree）上构造了可验证的证明，展示了框架的通用性。

**🔧 技术方法**

主要技术：
- 可验证的证书系统（assertion, verifier, pruning operator）。
- LP‑based阈值和互补松弛证书、Nemhauser–Trotter persistency、Facility‑integral、Reduced‑cost fixing、Steiner 预处理等。
- 学习理论：伪维数、统一受限损失、分布无关学习。 
- 置信过滤策略：基于 LP 值的可验证置信度做阈值过滤。
- 实验实现：SCIP 10.0 + PySCIPOpt 6.2.1、独立 verifier、可复现的 artifact。

**📊 数据集**

使用的数据集：
- Synthetic Set Cover、Vertex Cover、Facility Location、Knapsack 以及 0/1 Knapsack Pisinger benchmark。
- 真实图数据：DIMACS/SNAP 图、OR‑Library Facility Location、OR‑LIB‑cap、UFLLib、TSPLIB、SteinLib B‑class、Rail‑scale 真实网络。
- 生成的实验分布：轻量级分布（f 低）、重尾分布、分布外（distribution shift）等。

**📈 对比分析**

比较方法：
- 与传统预测‑augmented 方案（min‑combiner）以及其公平对手（min‑combiner + fallback）对比。
- 与基线求解器（SCIP）及其预处理、时间限制对比。
- 通过 14 组实验验证理论：
  * 正确性保证（0 失配、无负面影响）。
  * 学习阈值的样本复杂度（N≈5 即可达到最优）。
  * 置信过滤相对 min‑combiner 的边际提升（如 Vertex Cover 在噪声 0.5 时 0.29 的收益）。
  * 在分布外场景下，未验证剪枝可损失高达 26%，而经过 CASP 验证后损失为 0。
  * 速度提升：在硬实例上平均可达 13.7×（E6’）至 28×（E6”）的总时间加速，同时保持近似比 ≤ 1.03。

**⚠️ 局限性**

局限性：
1) 证书在缺乏结构的实例上无效（如稠密随机图、极端相关的 Knapsack）。
2) 速度加速主要体现在变量数量瓶颈的场景；在大 f 情况下理论安全因子（f‑safe）变得无意义。
3) 经验中安全因子和最坏情况近似界相对宽松，实际收益超过理论预期，说明分析可进一步收紧。
4) 置信过滤仅在 LP 置信度分布广泛时才显著获益；在低退化实例上，预测无优势。
5) 验证开销导致接受率下降，尤其在分布外情形下。
6) 所有实验均基于单一求解器（SCIP），未对其他求解器的泛化进行充分评估。

---

## 204. Controlled Reformulation Testing for Logical Consistency in Large Language Models

**arXiv ID:** 2607.14528 | [PDF](https://arxiv.org/pdf/2607.14528v1)

**作者:** Alexander Gu `[一作]` (University of Texas at Austin), Alan Chen `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了350个问题族的受控逻辑改写基准，用于评估大型语言模型在等价改写下的一致性。

**💡 创新点**

提出“Controlled Reformulation Testing”框架，区分等价保持、答案反转与表面控制改写，并揭示准确率与一致率之间显著的差距。

**🔧 技术方法**

采用零样本推理与显式推理（reasoning）模式、正则表达式答案抽取以及一致性统计等技术。

**📊 数据集**

使用350个问题族（共1,750条问答），覆盖命题逻辑、三段论、德·摩根律、量词推理、多跳推理、嵌套否定和条件谬误等七类。

**📈 对比分析**

在零样本条件下对GPT‑5.4、GPT‑5.4‑mini、Claude Sonnet 4、Gemini 2.5 Flash和o4‑mini等模型进行比较，测量基准准确率与家族一致率；o4‑mini最佳，准确率≈98.6%、一致率≈96.9%，GPT‑5.4‑mini准确率≈98.9%但一致率仅≈60%。

**⚠️ 局限性**

基准由模板生成并人工审核，缺乏自然语言多样性；仅评估二/三分类答案，未覆盖开放式或多步推理；显式推理在量词推理上导致严重退化，表明方法的普适性受限。

---

## 205. Uni-AdaVD: Universal Concept Erasure for Visual Generation via Orthogonal Value Decomposition

**arXiv ID:** 2607.14521 | [PDF](https://arxiv.org/pdf/2607.14521v1)

**作者:** Qifan Zhou `[一作]` (Hefei University of Technology), Meng Wang `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种通用的推理时概念消除框架Uni‑AdaVD，能够在不同视觉生成模型（U‑Net、DiT、自回归模型及视频生成模型）中删除目标概念，且不需要重新训练模型。

**💡 创新点**

创新点包括：①将多模态注意力的值空间统一为干预空间；②针对不同文本编码器构建编码器感知的目标表征；③结合正交值分解与层级自适应消除移位，实现对目标语义的精准抑制；④在单概念、多概念和隐式概念上均保持良好效果。

**🔧 技术方法**

核心技术包括注意力值空间干预、Encoder‑Aware Target Representation Construction (ETRC)、正交值分解 (OVD)、层级自适应消除移位 (LAES)。

**📊 数据集**

使用了I2P、COCO、SafeSora等公开数据集进行显式与隐式概念消除评估，同时在多种模型上进行实验。

**📈 对比分析**

与现有的细调、编辑及推理时方法相比，Uni‑AdaVD在目标概念的检测率大幅下降（如SD v1‑4从406降至26），同时在FID、SSIM、LPIPS等保留性能指标上位居或接近最优，且在对抗式提示攻击下表现出更低的攻击成功率。

**⚠️ 局限性**

局限性包括：①仍需对每层设定阈值，调参成本较高；②主要干预文本值分支，可能无法完全消除深层图像侧的概念；③在极大规模或极为细粒度的概念集上效果尚未充分验证。

---

## 206. VLT: A Vision-Language-Time Series Multimodal Foundation Model for Industrial Intelligence

**arXiv ID:** 2607.14510 | [PDF](https://arxiv.org/pdf/2607.14510v1)

**作者:** Haiteng Wang `[一作]` (Beihang University), Lei Ren `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VLT（Vision‑Language‑Time series）多模态基础模型，用于工业健康管理（PHM）中的时间序列预测与故障诊断。

**💡 创新点**

创新点包括：① 将频谱图像作为视觉桥梁，实现连续时间序列与离散文本语义的统一；② 设计 Time‑aware Mixture‑of‑Experts（Time‑MoE）捕获异构时序动态；③ 引入时空梯度对齐机制（梯度归一化 + 可靠性动态加权）解决模态间优化冲突；④ 采用跨模态一致性约束（CCL）进一步平衡模态贡献。

**🔧 技术方法**

技术手段包括：Time‑MoE（稀疏专家路由）、频谱‑图像转换与 MAE 视觉编码、预训练 Qwen 文本编码、时空注意力融合、梯度归一化、可靠性加权、CCL 一致性约束。

**📊 数据集**

使用的公开数据集：C‑MAPSS 涡轮机 RUL、XJTU 电池衰退、CWRU 轴承故障诊断。

**📈 对比分析**

与 Time‑LLM、GPT4TS、DLinear、PatchTST、TimesNet 等基线对比；VLT 在全样本、少样本、跨域转移和故障诊断任务中均实现了最低 RMSE、最高准确率（例如 C‑MAPSS FD001 RMSE 11.47，CWRU 1‑shot 正确率 88.23%），整体性能显著优于传统单模态方法。

**⚠️ 局限性**

局限性：当前模型仅针对数值预测和分类，未实现自然语言诊断推理；对极端大规模或更复杂工业场景的鲁棒性尚未充分验证。

---

## 207. DRIFT: Drift and Aggregation for Motion Planning

**arXiv ID:** 2607.14507 | [PDF](https://arxiv.org/pdf/2607.14507v1)

**作者:** Yining Xing `[一作]` (Tsinghua University), Jianqiang Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出DRIFT，一种固定深度的端到端规划器，采用一阶潜在漂移生成多种轨迹候选并通过场景感知聚合直接输出可执行轨迹；

**💡 创新点**

创新点包括：①在紧凑的PCA轨迹潜在空间实现一次性生成多候选；②利用漂移模型在潜在空间中引导生成，避免多步迭代；③采用场景感知聚合头，直接聚合潜在特征并生成最终轨迹，无需轨迹质量标签；④加入基于地图的边界正则化提升道路合规性；

**🔧 技术方法**

技术方法：V-JEPA预训练的ViT编码前视图序列；潜在空间漂移（RBF+Sinkhorn）与自适应层归一化；跨注意力与MLP-Mixer实现聚合；PCA编码/解码；地图边界损失；L1仿真损失；

**📊 数据集**

使用NAVSIM数据集（基于nuPlan传感器日志和地图注解），在navtrain上训练，navtest上评估；

**📈 对比分析**

与多种无评分器的规划器比较，DRIFT在NAVSIM v1得到89.6 PDMS、v2得到90.4 EPDMS，分别领先于MeanFuser、Drive-JEPA等，表现出更高的道路合规性和行进进度，计算延迟约10.8 ms（仅轨迹生成）+ 66.4 ms完整推理；

**⚠️ 局限性**

局限性：未在闭环或交互式仿真中验证；对提议多样性和聚合行为的度量有限；仅在固定潜在维度下工作，缺乏显式安全保证。

---

## 208. Collaborative Spatial Learning with Multi-LLM Agents in Networked Social Experiments

**arXiv ID:** 2607.14574 | [PDF](https://arxiv.org/pdf/2607.14574v1)

**作者:** Hao He `[一作]` (Virginia Tech), Xinwei Deng `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在Mason–Watts实验框架下，研究16个大型语言模型（LLM）代理在八种不同网络拓扑下进行二维空间搜索任务的协作学习，并与贝叶斯优化代理和人类实验数据进行比较。

**💡 创新点**

①首次将LLM代理置于Mason–Watts网络实验中，探究网络效率对LLM群体表现的影响；②提出在第一回合随机初始化的单句指令，可将LLM表现提升超过网络效应三倍；③将贝叶斯优化（EI、UCB）代理作为机制化基线，揭示LLM复制行为与收益关联较弱。

**🔧 技术方法**

使用OpenAI gpt-oss-120b LLM、两种基于高斯过程的贝叶斯优化代理（EI、UCB）、随机搜索基线；通过结构化提示、网络邻居历史信息、平均路径长度与OLS回归等统计方法进行实验与分析。

**📊 数据集**

自制60个二维支付景观（低/中/高复杂度），八个3-regular网络拓扑；对照原始Mason–Watts实验的29场人类数据，用于验证网络效应方向和行为指标。

**📈 对比分析**

对比累计平均收益、最终回合收益、峰值发现率，使用OLS回归检验平均路径长度对收益的影响。结果显示：LLM-RI显著优于默认LLM；贝叶斯优化代理获得最高收益；网络效应在累计收益上显著（负斜率），但在最终回合不显著。复制率高于人类，但与收益关联弱。

**⚠️ 局限性**

仅测试单一LLM模型与固定提示；人类对比基于不同景观，缺乏直接可比性；同步更新限制可能影响收敛速度；实验仅限15回合，未探讨更长时序；仅二维搜索任务，未验证到更复杂协作任务。

---

## 209. Breaking the Model Forgetting Cycle in Long-Incremental 3D Object Detection

**arXiv ID:** 2607.14560 | [PDF](https://arxiv.org/pdf/2607.14560v1)

**作者:** Peisheng Qian `[一作]` (Singapore University of Technology and Design), Na Zhao `[通讯]` (Singapore University of Technology and Design)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Learning-Dynamics-driven Memory and Review (LDMR) 框架，通过学习动态驱动的内阶段回顾与跨阶段记忆演化，解决长时间增量 3D 目标检测中的模型遗忘问题。

**💡 创新点**

将人类“学–评–复习”过程转化为动态采样与记忆更新机制，利用场景可学习性与多样性评分构建记忆库，显著缓解伪标签误差累积导致的自我强化循环。

**🔧 技术方法**

采用伪标签教师‑学生训练、学习动态监测、子阶段划分、加权采样、记忆库演化、TR3D/VoteNet 3D 检测骨干，以及 Wasserstein 上界分析。

**📊 数据集**

在 SUN RGB‑D（40 类）和 ScanNetV2（35 类）上采用 3/5/10 阶段长增量协议进行评估。

**📈 对比分析**

与 Fine‑tune、CPDet3D、SDCoT、SDCoT++、AIC3DOD 等基线及随机记忆的 SDCoT++ 进行比较，mAP@0.25 在 10 阶段下提升约 3.8–5.1 点，显著优于基线且接近非增量联合训练的上界。

**⚠️ 局限性**

受限于固定记忆预算、子阶段划分与权重超参数的敏感性，且在超长阶段仍存在少量遗忘与伪标签质量下降；未覆盖跨模态或动态场景的扩展。

---

## 210. HyMobileAgent: Data-Environment Co-Scaling for Efficient GUI Agents

**arXiv ID:** 2607.14548 | [PDF](https://arxiv.org/pdf/2607.14548v1)

**作者:** Hy Vision Team `[一作]`, Chengquan Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一款可在安卓设备上本地部署的多模态移动GUI智能体，整合视觉感知、知识提取、动作规划与强化学习，实现端到端交互。

**💡 创新点**

创新点包括：① 统一的“感知飞轮”+“知识管道”+“动作轨迹”三层数据体系；② 基于Mock App Factory的可重置验证环境；③ 结构化的五字段决策+死循环反射机制；④ 逐步训练（mid‑training → SFT → RL）与多阶段奖励设计；⑤ 采用A3B规模、32K上下文的视觉本地化模型与13项函数调用动作空间。

**🔧 技术方法**

核心技术：视觉本地化模型（native any‑resolution，32K上下文）；函数调用式动作接口（13个原语，坐标归一化）；结构化决策模板；离线与在线GRPO强化学习；奖励模型与规则化奖励；图形化与自然语言的多模态数据合成与筛选。

**📊 数据集**

使用的数据集：合成与真实的GUI截图、开放源代码的定位数据、图标库、教程视频/书籍/帖子转化的知识数据、500+真机与虚拟机交互轨迹、2000+沙盒实例、Mock App Factory（34个模拟App、34000+单App任务、500个跨App任务）、HyMobileWorld/HyMobileGrounding/HyMobileQA 等内部基准，公开基准包括AndroidWorld、MMBench‑GUI、ScreenSpot、ScreenSpot‑Pro。

**📈 对比分析**

与同规模及更大规模的专用与通用模型对比：在AndroidWorld上实现82.6%成功率，超过Gemini 3.1（80.2%）和Seed 2.0 Pro（71.5%）；在HyMobileWorld上获得42%成功率，接近Seed 2.0 Pro（44.7%），显著优于同规模UI‑Venus 1.5、MAI‑UI 8B和AutoGLM。定位方面在MMBench‑GUI L2、ScreenSpot V2、ScreenSpot‑Pro、HyMobileGrounding分别达89.3/96.2/66.5/93.1；问答方面在MMBench‑GUI L1、HyMobileQA分别达到93.7/87.0。

**⚠️ 局限性**

局限性：① 依赖大量人工/自动标注的数据与Mock环境，数据获取成本高；② 仍受A3B参数限制，面对极其复杂或安全敏感的真实应用（如登录、支付）表现有限；③ 主要聚焦GUI交互，尚未实现CLI/脚本级别的混合操作；④ 目前仅在安卓平台验证，跨平台推广需进一步适配。

---

## 211. AdaTurn: Budget-Aware Test-Time Scaling for Active Visual Perception Agents

**arXiv ID:** 2607.14547 | [PDF](https://arxiv.org/pdf/2607.14547v1)

**作者:** Susan Liang `[一作]` (University of Rochester), Chenliang Xu `[通讯]` (University of Rochester)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AdaTurn 框架，使主动视觉代理能够在不同的回合预算下做出预算感知的决策，并显式训练预算边界的最终回答行为，从而显著提升低预算场景下的性能。

**💡 创新点**

创新点包括：① 将回合预算作为任务输入，使模型学习到不同预算下的多样化策略；② 引入 Forced-Answer DAPO，将预算溢出转化为可训练的最终回答步骤；③ 采用动态预算训练与负载均衡调度，解决训练/推理中预算不均导致的效率瓶颈。

**🔧 技术方法**

使用技术包括：多轮视觉工具集成的 RL 训练（基于 DAPO/GRPO 的策略优化）、强制回答训练、动态回合预算注入、负载均衡调度算法，以及多模态 Transformer 作为基础模型。

**📊 数据集**

主要评测数据集：VisualProbe、V* Bench、HR‑Bench、MME‑RealWorld；此外还验证了在 OCRBench、ChartQA、DocVQA、MathVista、ScienceQA、CV‑Bench 等通用多模态任务上的鲁棒性。

**📈 对比分析**

与 GPT‑4o、LLaVA‑OneVision、Qwen2.5‑VL‑7B‑Instruct、DeepEyes、Mini‑o3、Pixel Reasoner、Chain‑of‑Focus 等基线对比；在 4 回合的低预算下，AdaTurn 在 VisualProbe‑Hard 提升 12.5%（+12.5%）、VisualProbe‑Medium 提升 10.9%、HR‑Bench‑8K 提升 7.9% 等；在 32 回合的高预算下保持或超过 Mini‑o3 的表现；在不同规模 Backbone（Qwen3‑VL‑4B/8B）下也能保持优势。

**⚠️ 局限性**

局限性：① 需要大量 RL 训练和动态预算采样，计算成本高；② 受限于工具集合的覆盖范围，若任务不适合多轮工具交互，收益有限；③ 在极端预算极低或极高的情况仍可能出现性能波动；④ 需要进一步验证在更广泛的实际部署场景（如实时系统）中的稳定性与鲁棒性。

---

## 212. Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent

**arXiv ID:** 2607.14541 | [PDF](https://arxiv.org/pdf/2607.14541v1)

**作者:** Lingyun Yang `[一作]` (Alibaba Group), Liping Zhang `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于完整生产推理轨迹的 GPU 内核生成基准和一个基于轮询搜索、优化掉线和知识库的轮廓驱动优化代理。

**💡 创新点**

创新点在于：① 采用真实工作负载分布的重要性加权屋脊分数；② 设计了一个完整的工作流，让 LLM 代理在获取硬件信息、参考实现、轮询搜索、优化掉线的基础上迭代改进；③ 揭示并缓解了“正确性幻觉”，即模型仅通过 PyTorch 回退获得高正确率。

**🔧 技术方法**

使用了：屋脊模型、FlyDSL DSL、GPU Wiki 知识库、官方硬件 profiler、迭代的 profile‑revise 搜索循环、优化掉线机制、层级化知识检索。

**📊 数据集**

数据集来自 10k+ XPU‑A/H20 生产集群的推理轨迹，覆盖 30 个算子、440 个热点形状，包含 1,303 条 profile，涉及 vLLM、SGLang、AITER、RTP‑LLM 等 20+ 生产模型。

**📈 对比分析**

通过六大前沿 LLM 代理与 torch.compile、生产内核基准进行对比，评估编译率、正确率、FlyDSL 采用率和屋脊成就。最佳代理仅达约 10% 的硬件屋脊；结合优化代理后，可提升至约 30% 并在若干算子上超过手工调优的生产内核。

**⚠️ 局限性**

局限性：仅覆盖推理内核（不含训练），仅在 XPU‑A 上验证；需要定期刷新轨迹以跟踪工作负载演化；优化代理受知识库规模和覆盖度限制；未覆盖其他 GPU 架构或更高吞吐量的场景。

---

## 213. Communication-Efficient Relative Pose Estimation with Vision Foundation Models for Ephemeral Collaborative Perception

**arXiv ID:** 2607.14539 | [PDF](https://arxiv.org/pdf/2607.14539v1)

**作者:** Qihang Li `[一作]` (North Carolina State University), Peng Gao `[通讯]` (North Carolina State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 CERPE 框架，用于在短暂协同感知期间实现低通信成本、无 GPS/地图约束的 6-DoF 相对位姿估计。

**💡 创新点**

创新点在于将视觉基础模型与固定尺寸描述符门控的原子化图像交换、基于度量深度的单目尺度恢复以及在无重叠时利用测距推进相对位姿，从而实现持续、准确的相对位姿推理。

**🔧 技术方法**

采用 VGGT、Metric3Dv2、SALAD 等视觉基础模型，结合描述符相似度门控、度量尺度对齐和 SE(3) 运动传播，构建无训练的端到端管线。

**📊 数据集**

在 CARLA 仿真、OpenMars 真实驾驶、以及三台 Agilex Limo Pro 机器人室内实验等数据集上进行评估。

**📈 对比分析**

与 SuperGlue、NOPE、Co-VisNet 等基线相比，CERPE 在所有场景中保持 100% 成功率，位置误差下降 72% 以上、旋转误差下降 90% 以上，并在非重叠期间保持相对位姿连贯性。

**⚠️ 局限性**

局限包括对尺度估计的高度依赖、对大规模多机器人团队的计算与通信负载、以及缺乏后端优化导致的长期漂移。

---

## 214. SwinAD: Multi-stage feature reconstruction for unsupervised industrial anomaly detection

**arXiv ID:** 2607.14534 | [PDF](https://arxiv.org/pdf/2607.14534v1)

**作者:** Huong Ninh `[一作]` (University of Engineering and Technology Vietnam National University), Long Tran `[通讯]` (University of Engineering and Technology Vietnam National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于冻结预训练Swin Transformer V2的多阶段特征重建框架SwinAD，用于多类别无监督工业异常检测。

**💡 创新点**

核心创新在于结合层次化Transformer特征与保持特征多样性的双分支重建器，防止单一重建导致模式崩塌。

**🔧 技术方法**

采用预训练Transformer编码器、带dropout的瓶颈模块、双分支重建解码器以及硬负样本挖掘的余弦相似度损失。

**📊 数据集**

在MVTec AD、VisA和Real‑IAD三大工业异常检测基准上进行实验。

**📈 对比分析**

与UniAD、ReContrast、Dinomaly、ViTAD、MambaAD等方法对比，SwinAD在像素级AP/F1上提升5–10%，且在256×256输入下仅需54 GFLOPs，保持较高的计算效率。

**⚠️ 局限性**

仍存在多分支解码器导致额外开销的限制，且对极小、极稀疏缺陷的检测精度仍有提升空间。

---

## 215. Overlapping Network Community Detection Using Sparse Backbones

**arXiv ID:** 2607.14531 | [PDF](https://arxiv.org/pdf/2607.14531v1)

**作者:** Zihe Zhou `[一作]` (University of Toronto), Samin Aref `[通讯]` (University of Toronto)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为Highway的可扩展重叠社区检测算法，利用网络稀疏骨干进行高效传播与模式校准。

**💡 创新点**

通过构建结构信息丰富的骨干子图并在其上进行锚点传播，减少无用边干扰，实现高效且准确的重叠社区识别。

**🔧 技术方法**

结合模量化与邻域交叉的混合重要性评分、锚点初始化、邻居传播、模式一致性校准和归一化熵等多种图神经传播与统计技术。

**📊 数据集**

在728个Lancichinetti‑Fortunato‑Radicchi（LFR）合成网络上进行实验，涵盖不同节点数、边数和混合参数。

**📈 对比分析**

与10种主流OCD方法及全图版本对比，使用FRI、Q_ov、Dice、F*、ONMI五项指标，Highway在ONMI上首位、其余四项第二，并在中高噪声场景中保持更稳健。

**⚠️ 局限性**

对真实网络的评估不足，骨干参数和锚点选择对结果敏感，且在极低噪声下全图版本优于骨干版，需进一步自动化参数调优。

---

## 216. xHC: Expanded Hyper-Connections

**arXiv ID:** 2607.14530 | [PDF](https://arxiv.org/pdf/2607.14530v1)

**作者:** Xiangdong Zhang `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现一种新型超连接模型 xHC，能够将 Transformer 的残差流扩展到 N>4 并保持可扩展性与稳定性。

**💡 创新点**

创新点包括：① 通过多尺度因果深度可变卷积对写回信号进行时间特征增强，解决写回信息瓶颈；② 采用稀疏残差更新（仅更新 k=N/4 个流）将残差映射生成成本从 O(N³) 降至 O(k³)，解决计算瓶颈；③ 设计 xHC‑Flash 共享路由与密集读操作，显著降低大 N 下的内存流交通。

**🔧 技术方法**

技术手段：Hyper‑Connections、Manifold‑Constrained HC、Sinkhorn 正则化、稀疏路由与固定流、密集读、时间特征增强（因果深度可变卷积）、稀疏残差混合、融合内核、Muon 优化器。

**📊 数据集**

数据集与评测基准：预训练使用多语言（英中）、代码、数学、推理等混合数据；下游评测涵盖 MMLU、MMLU‑Pro、MMLU‑Redux、BBH、CommonsenseQA、ARC‑Challenge、GSM8K、HumanEval、LCBench、CMMLU、CEval、C3 等。

**📈 对比分析**

与 mHC(N=4) 以及 vanilla 基线在 18B/28B MoE 模型上在相同训练 FLOPs 下对比；xHC 在 18B 上平均分从 44.8 提升到 48.8（+4.0），28B 上从 50.5 提升到 53.6（+3.1）；在 2.5B N‑sweep 中 xHC 在 N=16 时仍能进一步降低损失，表明 N 仍为有效的扩展轴；计算效率提升为 mHC 需要 1.19× FLOPs 达到同样损失，xHC 只需 1×。

**⚠️ 局限性**

局限性：仍需要显著的残差流内存流交通（即使 xHC‑Flash 已降低到与 N=4 的 mHC 相当的水平）；稀疏更新在极大 N 时可能导致信息传递断连；写回增强依赖局部上下文，对长序列或多模态数据的适应性尚未验证；模型对写回信息的敏感性仍是未来可进一步改进的瓶颈。

---

## 217. Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding

**arXiv ID:** 2607.14513 | [PDF](https://arxiv.org/pdf/2607.14513v1)

**作者:** Amir Said `[一作]` (Qualcomm AI Research), Randall Rauwendaal `[通讯]` (Qualcomm Graphics Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文针对3D高斯喷射（3D Gaussian Splatting, 3DGS）模型的高内存占用问题，提出一种利用GPU友好的纹理压缩技术（BC1、BC7等）对球面谐波（SH）颜色系数进行压缩的方法。

**💡 创新点**

创新点在于：①将纹理压缩技术应用于3DGS原始数据，充分利用GPU并行解码与渲染优势；②通过对原始原语颜色进行排序（color sorting）以及多比特流（multi‑bitstream）实现可变比特率压缩；③设计了层次化的数据组织，使得位置/形状参数与外观参数分离，支持大场景的LOD与随机访问。

**🔧 技术方法**

主要技术包括：GPU纹理压缩格式（BC1/BC7、ASTC）与块截断编码（Block Truncation Coding, BTC），球面谐波系数的量化与重排，基于颜色的原语排序，层次化的压缩数据结构，以及比特率控制策略。

**📊 数据集**

实验使用了官方3DGS提供的公开数据集，包含“bicycle”“bonsai”“garden”三幅场景。

**📈 对比分析**

通过在25个测试视角下计算PSNR进行比较。结果显示：单独BC1压缩会显著损失质量；BC7可获得较好质量但字节数翻倍；采用颜色排序+BC7+BC1混合压缩时，PSNR提升约10–16 dB，几乎无视觉损失，且每原语字节数仅为10–11.5。

**⚠️ 局限性**

局限性包括：需对SH系数进行预先量化与归一化，导致额外预处理；对原语排序与重排需要额外算力；不同GPU硬件对BC1/BC7解码效率差异；压缩后的颜色量化仍可能在极端色彩或高光区域产生微弱失真。

---

## 218. Multi-Scale ViT Inference with Habitat-Fit Priors and kNN Retrieval for Multi-Species Plant Identification

**arXiv ID:** 2607.14509 | [PDF](https://arxiv.org/pdf/2607.14509v1)

**作者:** Alper Erten `[一作]` (Georgia Institute of Technology), Adrian Cheung `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了基于 DINOv2 ViT‑L 的多尺度瓦片推理、FAISS kNN 检索、时序融合以及栖息地适配等后处理的多物种植物识别系统；

**💡 创新点**

创新点在于将多尺度瓦片聚合、检索增强与基于地理/海拔先验的栖息地适配相结合，针对单标注训练与多标签域迁移提供了全新的推理框架；

**🔧 技术方法**

使用了 DINOv2 ViT‑L、multi‑scale tiling、max‑pool 聚合、FAISS kNN、temporal fusion、habitat‑fit demotion 以及地理掩码等技术；

**📊 数据集**

利用 PlantCLEF 2024/2025 单标注植物图像（约140万张）和 2,105 张高分辨率田地图像进行训练与评估；

**📈 对比分析**

在公开排行榜上通过多种基线对比，未选取版的私有 macro‑F1 达 0.45777，选取版达到 0.43902（第三名），显著提升了 0.1 以上；

**⚠️ 局限性**

局限性主要是对公开 11% 测试集的超参调优导致泛化性受限，外部扩展（LUCAS、SAM 裁剪等）未带来收益且可能引入分布漂移。

---

## 219. Formal Verification for Deep Learning-based Power Control in Massive MIMO

**arXiv ID:** 2607.14500 | [PDF](https://arxiv.org/pdf/2607.14500v1)

**作者:** Thanh Le `[一作]` (National Institute of Information and Communications Technology), John C. S. Lui `[通讯]` (Chinese University of Hong Kong)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对多小区大规模 MIMO 系统中基于深度学习的功率控制模型的形式化验证框架，用以评估模型在受位置扰动（对抗攻击）影响时的鲁棒性，并给出最大优最性能间隙的正式保证。

**💡 创新点**

创新点在于：①首次在回归任务且输出受非线性 SINR 乘积约束的情形下，结合 DeepPoly 抽象传播与可行性分析，提供完整的鲁棒性证明；②针对功率分配的非线性性能指标（最大 SINR 乘积）构造约束程序，突破了传统仅针对分类或线性回归的 NN 验证方法；③将对抗扰动建模为超矩形约束，并在多尺度扰动下系统性验证。

**🔧 技术方法**

主要技术包括：1）基于输入扰动超矩形的局部鲁棒性属性构造；2）DeepPoly 抽象传播技术用于获得输出功率的上下界；3）将得到的功率边界与 SINR 计算公式及性能间隙约束一起构成可行性问题，利用数值优化工具进行可行性检查；4）实验评估采用 L1 距离和 UNSAT 比例两种指标。

**📊 数据集**

使用了公开的多小区大规模 MIMO 功率分配数据集（链接：https://data.ieeemlc.org/Ds2Detail），包含 329,000 条训练样本和 500 条测试样本，数据覆盖 4 个小区、每个小区 5 个 UE，地点在 250 × 250 m 区域内随机分布。

**📈 对比分析**

对比方法：选取两种全连接网络（分别含 6,981 与 202,373 参数）在不同扰动幅度（0.01 m~100 m）与不同最优性间隙阈值（0.001~0.1）下进行验证。性能指标包括：L1 距离（越小越好）和 UNSAT 比例（验证成功率）。结果表明：在扰动小于 1 m 时，较小网络在 99% 的性能间隙下可实现 100% 的验证成功；较大网络的验证成功率明显下降，尤其在扰动超过 0.5 m 时表现急剧恶化。

**⚠️ 局限性**

局限性：①仅验证局部鲁棒性，对全局鲁棒性未做探讨；②只考虑位置扰动，对其他类型对抗攻击（如信道状态信息攻击）缺乏覆盖；③DeepPoly 抽象传播在网络较大、层数较深时趋于保守，导致可验证范围受限；④仅针对单一性能指标（最大 SINR 乘积）进行验证，未扩展到多目标或 QoS 约束；⑤实验规模仍受限于公开数据集和 4‑小区 5‑UE 的简单网络拓扑。

---

## 220. A Continuous-Time Reinforcement Learning Framework for Fine-Tuning Discrete Diffusion Models

**arXiv ID:** 2607.14522 | [PDF](https://arxiv.org/pdf/2607.14522v1)

**作者:** Zikun Zhang `[一作]` (Columbia University), Wenpin Tang `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了连续时间离散空间强化学习框架，利用受控CTMC求解离散扩散模型（尤其是掩码扩散模型）的微调问题；

**💡 创新点**

首次将连续时间强化学习与离散扩散模型对接，支持中间奖励、非可微奖励且不需对标量进行梯度估计；

**🔧 技术方法**

采用受控CTMC、Hamilton-Jacobi-Bellman方程、策略梯度、PPO与GRPO算法以及轨迹子采样技巧；

**📊 数据集**

在二维棋盘、LLaDA‑8B‑Instruct（数学推理与编码任务）以及HumanEval、MBPP等公开数据集上进行实验；

**📈 对比分析**

与d1、d2、SPG等基准相比，CTRL在Sudoku、GSM8K、MATH500、HumanEval和MBPP等任务上取得更高或相近的准确率，尤其在Sudoku上显著优于其他方法；

**⚠️ 局限性**

主要局限在轨迹概率估计需要多次前向传播、子采样步长对收敛影响敏感，以及在更长序列长度下性能下降。

---

## 221. Space-Entropy Lower Bounds for Random Sampling

**arXiv ID:** 2607.14503 | [PDF](https://arxiv.org/pdf/2607.14503v1)

**作者:** Thomas L. Draper `[一作]` (Carnegie Mellon University), Feras A. Saad `[通讯]` (Carnegie Mellon University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了在接近信息论最优熵消耗时随机采样所需的最小空间下界

**💡 创新点**

首次给出与熵效率相关的空间下界，结合对数、数论与Diophantine逼近技术

**🔧 技术方法**

利用状态信息内容函数、图论、数论与线性无关度测度等方法

**📊 数据集**

无须使用具体数据集，理论上分析伯努利与一般离散分布

**📈 对比分析**

与现有无状态采样、DDG树、随机重用算法等进行比较，表明空间需求常数因子可压到1–2之间

**⚠️ 局限性**

仅适用于离散分布的在线采样，连续分布情形及常数因子优化仍是局限

---

## 222. Democratizing Agent Deployment Safety: A Structural Monitoring Approach

**arXiv ID:** 2607.14570 | [PDF](https://arxiv.org/pdf/2607.14570v1)

**作者:** Preeti Ravindra `[一作]` (Independent), Vincent Wolowski `[通讯]` (Independent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于结构信息流图的监控器IFG，用于在IaC代码改动中检测并阻止AI代理的破坏行为，并实现同步回滚与异步审计两种部署模式。

**💡 创新点**

创新点在于：①使用控制流图/数据流图差分对比而非仅依赖代码diff或历史攻击日志；②实现无训练的结构化监控，可同步阻止高影响力改动；③在异步与同步两种模式下评估性能。

**🔧 技术方法**

采用LLM（Claude 3.7 Sonnet / Claude Haiku 4.5）作为可信模型，结合CDK synth生成的控制流图与数据流图，对差分进行结构分析并生成怀疑评分。

**📊 数据集**

使用ControlArena红队数据集（IaC-confined slice，100正样本+160攻击样本）以及公开的IaC-Confined评估片段。

**📈 对比分析**

与基线git diff监控器和训练好的Async RF集成器比较；异步时IFG在1%误报率下漏检率降至3.5%（基线11.6%），ROC AUC为0.941；同步回滚时IFG将联合任务成功率降至0%（基线5.6%），且不降低主任务成功率。

**⚠️ 局限性**

局限性包括：无法检测代码diff之外的攻击（如构建产物中隐藏恶意代码）、受可信模型prompt injection威胁、依赖CFG/DFG生成且对某些语言或框架支持有限、对结构性监控的覆盖面仍有盲点。

---

## 223. Fully Automated End-to-End Adversary Emulation from MITRE ATT\&CK Based Cyber Threat Intelligence Using LLMs

**arXiv ID:** 2607.14566 | [PDF](https://arxiv.org/pdf/2607.14566v1)

**作者:** Jueon Choi `[一作]` (Chonnam National University), Gunjin Cha `[通讯]` (Good First Information Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一套完全自动化的端到端对手仿真框架，能够从MITRE ATT&CK 对齐的 CTI 报告中生成 Caldera 可执行 playbook，并在执行过程中自动检测并根据失败类型进行修复。

**💡 创新点**

创新点在于将 playbook 生成、自动执行和失败恢复三大功能整合到一个无人工干预的流水线中，并引入基于失效类型的自适应修复机制；同时通过 LLM 对长篇自然语言 CTI 进行结构化提取，实现了从报告到可执行脚本的全流程自动化。

**🔧 技术方法**

采用了多种大型语言模型（Claude Sonnet 4.5、GPT‑4o、Gemini 2.5 Pro、Grok 4 Fast）进行文本分析与命令生成；利用 MITRE Caldera 平台执行并收集结果；使用正则表达式对执行错误进行四类（语法、依赖、缺失环境、超时）分类，并在 Prompt 中加入历史修复记录进行自适应改写。

**📊 数据集**

实验使用 11 篇 KISA 发布的高质量 CTI 报告（平均 30,840 词）作为主数据集，并在与 AURORA 对比时选取其 10 篇公开 CTI 报告作为基准。

**📈 对比分析**

与 AURORA 采用相同评估指标（链长度、执行成功率、CTI 精度/召回/F1）进行对比；Claude Sonnet 4.5 在 F1 方面获得 30.57%（AURORA 为 26.07%），最终执行成功率达到 65.17%（AURORA 60.72%）；失败恢复机制在所有 LLM 上平均提升 14–17 个百分点，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：仅生成单一路径 playbook，无法覆盖报告中的多分支情景；环境规范需要人工预先准备，未实现全自动基础设施部署；对某些不可修复错误（如环境限制导致的失败）仍无改进手段；数据集规模有限，尚未在更大多样化 CTI 上验证。

---

## 224. Probabilistic Physics-Informed Neural Networks for Estimating Heterogeneous Elastic Properties from Low-Resolution and Noisy Displacement Data

**arXiv ID:** 2607.14563 | [PDF](https://arxiv.org/pdf/2607.14563v1)

**作者:** Tatthapong Srikitrungruang `[一作]` (Texas A&M University), Jaesung Lee `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出了一种新的概率物理信息神经网络（PIE-PINN），用于从低分辨率、噪声位移数据中稳健估计弹性模量和泊松比。

**💡 创新点**

创新点在于将Laplace分布统一建模位移观测、应变差异和平衡残差，并结合B样条平滑全局位移与神经网络局部校正，辅以层级半-Cauchy尺度模型自适应下调大误差的权重。

**🔧 技术方法**

采用概率模型、B样条插值、神经网络、Laplace分布、层级半-Cauchy尺度以及交替最大似然训练策略。

**📊 数据集**

使用合成低分辨率、带噪声的位移测量数据，模拟不同噪声水平和观测分辨率的情形。

**📈 对比分析**

与传统PINN和经典逆问题方法比较，PIE-PINN在不同噪声与分辨率下表现出更高的鲁棒性和更准确的弹性参数恢复。

**⚠️ 局限性**

局限性包括对模型超参数的依赖、训练过程计算成本较高以及在复杂三维几何或大规模问题上的可扩展性待验证。

---

## 225. Impact of Expert-Following Strategies in Financial Asset Recommendation

**arXiv ID:** 2607.14556 | [PDF](https://arxiv.org/pdf/2607.14556v1)

**作者:** Ryuki Unno `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种专家跟随策略，用以在金融资产推荐中同时提升投资回报率 (ROI) 与偏好匹配度 (nDCG)，通过识别历史高 ROI 投资者并根据其购买记录推荐资产。

**💡 创新点**

创新点在于将 ROI 权重与专家购买频率相结合，形成 ROI 加权的购买频率评分，从而无需复杂模型即可在 ROI 与 nDCG 之间取得平衡。

**🔧 技术方法**

采用了投资者交易历史的 ROI 计算、ROI 加权购买频率评分、阈值选取专家集合，以及多实例滚动验证与显著性检验等技术；基线包括市场平均、随机森林、Sharpe Ratio、热门度等。

**📊 数据集**

使用了 FAR-Trans 真实交易历史数据（2018‑2022 年）。

**📈 对比分析**

通过与市场平均、随机森林、Sharpe Ratio、热门度等基线方法对比，并进行配对 t 检验和 Wilcoxon 检验，实验表明专家跟随策略在 ROI@10 与 nDCG@10 双指标上均显著优于基线，尤其是 top‑5% 方案在 ROI 0.246 与 nDCG 0.176。

**⚠️ 局限性**

局限性包括假设交易成本与滑点为零，可能高估 ROI；未使用图模型或更复杂的推荐算法；高频交易的专家可能导致过度集中。

---

## 226. Answer-Conditioned Chains of Thought Degrade Verifiable-Reasoning Distillation in Large Language Models

**arXiv ID:** 2607.14552 | [PDF](https://arxiv.org/pdf/2607.14552v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在完整控制实验中对比答案可见与答案盲的链生成方式，量化并证明答案可见会显著降低大语言模型的推理性能，提出可在训练前通过链中答案首次出现率预测惩罚，并给出了链级过滤与derive-first指令等修复方案。

**💡 创新点**

首次在控制实验下揭示“答案泄露惩罚”——答案可见导致的“理性化”链对模型学习产生负面影响；提出答案首次出现率作为训练前可预测惩罚的签名；证明仅理性化链承担大部分损害并可通过链级过滤部分修复。

**🔧 技术方法**

对大模型进行答案盲/答案泄露的链生成、基于正确性过滤的监督微调（SFT）、答案首次出现率统计、差分分析、线性回归等统计方法；在多任务多模型上评估。

**📊 数据集**

使用多种数学题库（MATH‑500、GSM8K、AIME、Nemotron‑Cascade‑SFT‑Stage‑2等）、代码测试集（MBPP+、HumanEval+）、以及多选知识基准（MMLU、GPQA‑Diamond）。

**📈 对比分析**

在相同数据量、链长度与过滤条件下，答案盲与答案泄露的SFT模型在MATH‑500上准确率差距约16–27点；通过跨模型、跨任务的差分比较验证惩罚随答案首次出现率相关；修复措施可恢复约2/3以上性能。

**⚠️ 局限性**

限于需要多步推理的任务，答案泄露的影响在多选任务中不显现；对代码领域的机制验证仍需进一步验证；未考察更大规模或非英语模型；仅在SFT框架下验证，其他学习范式尚未评估。

---

## 227. World-Model-Aware Responsibility Allocation in Heterogeneous Logistics Systems

**arXiv ID:** 2607.14550 | [PDF](https://arxiv.org/pdf/2607.14550v1)

**作者:** Artan Markaj `[一作]` (Eurogate GmbH & Co. KGaA, KG), Felix Gehlhoff `[通讯]` (Helmut Schmidt University Hamburg)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在物流系统中提出了世界模型感知责任分配框架 (WMARF)，通过动态评估中央控制系统与自主物流设备的世界模型质量和设备自动化水平，实时分配决策权，进而防止因模型不一致导致的死锁。

**💡 创新点**

创新点在于：①将决策权分配与世界模型信息质量绑定，而非固定的系统设计；②引入四象限责任模型，并按权责状态对死锁进行分类；③通过基于接近、模型差异、自动化等级等触发条件实现动态权责切换。

**🔧 技术方法**

技术实现基于 Python 模块（Quadrant Classifier、Handoff Trigger Evaluator、Deadlock Classifier）以及 SimPy 事件驱动仿真；在真实工业接口上亦通过 VDA 5050 标准实现，与现有控制逻辑无缝集成。

**📊 数据集**

主要使用的“数据集”是仿真生成的物流场景数据（两辆自主物流车辆与一个半自动化转运点），并在 VDA 5050 接口上复现同一场景；并未使用公开大规模工业数据集。

**📈 对比分析**

对比方法：将 WMARF 与传统静态中央控制策略对比，展示在相同场景下无 WMARF 时出现的 Type‑C 死锁，使用 WMARF 时通过接近触发权责切换完成两次转运。性能表现为避免了死锁，示例中两次转运顺利完成；但仅提供单一定性案例，未给出吞吐量、延迟等定量指标。

**⚠️ 局限性**

局限性：①仅在单一场景下验证，缺乏大规模、多设备、多目标的量化评估；②权责切换阈值（θ、δ、τ）需人工设置，缺乏自动调参方法；③触发条件列表未覆盖所有可能情况，实际部署需进一步完善；④假设 WMARF 代理本身可靠且不影响安全路径，实际实现需保证其稳健性。

---

## 228. VTM-Nav: Hierarchical Visual-Topological Memory for Cross-Episode Object-Goal Navigation

**arXiv ID:** 2607.14514 | [PDF](https://arxiv.org/pdf/2607.14514v1)

**作者:** Xiaoran Xu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Changsheng Xu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种跨周期目标导航方法，利用训练‑free 的视觉‑语言模型（VLM）并维护一个持久化的层次化视觉‑拓扑记忆（VTM），在同一场景内多次任务中不断累积并重用房间拓扑、物体证据和成功路径。

**💡 创新点**

创新点在于①将跨周期记忆与 VLM 结合，②构建房间层级与对象层级的视觉‑拓扑结构，③通过软记忆引导 VLM 选择动作并配合保守执行守卫避免误操作，④实现了在固定模型参数下的持续性能提升。

**🔧 技术方法**

使用的技术包括：Qwen3‑VL‑Plus 视觉‑语言模型、语义证据提取与归一化、房间与对象级联的层次化记忆、基于语义/拓扑/空间一致性的房间定位评分、记忆检索与候选重排、以及基于执行反馈的保守守卫机制。

**📊 数据集**

实验数据集：Habitat ObjectNav Benchmarks 中的 HM3D v0.1、HM3D v0.2 以及 MP3D。

**📈 对比分析**

与控制版 WMNav 及其加入文本记忆的基线进行对比，评估指标为成功率 SR 与路径长度加权成功率 SPL。结果显示，在 40 步限制下，本文方法在 HM3D v0.1、HM3D v0.2 和 MP3D 上分别实现约 59.6%、72.0% 与 44.3% 的 SR，均优于基线 2–5 个百分点，SPL 接近或略优，且路径效率更高。

**⚠️ 局限性**

局限性：仅在同一场景内累积记忆，缺乏跨场景迁移能力；依赖固定 VLM，无法在训练中自适应；在动态环境或真实机器人上未验证；记忆容量与维护开销未系统评估。

---

## 229. RetroAgent: Harnessing LLMs to Search Over Structured Memory for Agentic Retrosynthesis Planning

**arXiv ID:** 2607.14512 | [PDF](https://arxiv.org/pdf/2607.14512v1)

**作者:** Yanqiao Zhu `[一作]` (Ucla), Wei Wang `[通讯]` (Ucla)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的代理，利用结构化内存（AND-OR 图）与工具接口实现多步还原合成规划；

**💡 创新点**

创新点在于将LLM与可持久化的搜索状态耦合，允许代理在全局搜索状态下做决策；设计了多扩展（multi-expand）和深度惩罚机制，实现了对搜索宽度与深度的动态平衡；

**🔧 技术方法**

使用了QLM（4B 参数）配合工具调用（内存工具与化学工具），并通过强化学习（GSPO）在搜索过程中进行策略优化；单步预测采用基于模板的 MLP 模型；

**📊 数据集**

主要数据集为 USPTO-Full（训练 20K 目标），评价数据集为 USPTO-190（190 目标）和 ChEMBL-1000（1000 目标）；构建了 23M 级商业化构件库；

**📈 对比分析**

与搜索基线（Retro*、PDVN、EG-MCTS 等）以及 LLM 基线（Retro-R1）对比；在 USPTO-190 上 Pass@1 达 53.3%（高于 Retro-R1 的 50%），在 ChEMBL-1000 上 Pass@1 达 73.8%（高于 Retro-R1 的 68.5%）并在预算 500 时取得最高成功率；

**⚠️ 局限性**

局限性包括：需要强化学习训练才能发挥作用；模型仍受限于 4B 参数，且在高预算下搜索覆盖率不如完整搜索方法；对不同化学任务的泛化能力需进一步验证；

---

## 230. Non-vacuous Generalization Bounds for Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2607.14506 | [PDF](https://arxiv.org/pdf/2607.14506v1)

**作者:** Yuxuan Zhu `[一作]` (University of Illinois Urbana-Champaign), Daniel Kang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了首个针对RLVR（可验证奖励强化学习）微调的大规模LLM（数十亿参数级别）的非空泛化误差上界，并设计了Progressive RLVR流水线以实现压缩与性能兼顾；

**💡 创新点**

创新点在于：①将PAC‑Bayes压缩上界与Gumbel‑max重参数化相结合，克服了RLVR奖励的随机性与组合性；②提出Progressive RLVR框架，通过高容量教师、on‑policy蒸馏、TinyLoRA与量化实现高压缩率；

**🔧 技术方法**

使用技术包括PAC‑Bayes压缩上界、Gumbel‑max重参数化、TinyLoRA、on‑policy蒸馏、模型量化、以及对RLVR的自定义奖励机制；

**📊 数据集**

实验数据集涵盖四大任务域：数学问题求解（Eurus‑2‑RL/NuminaMath‑CoT）、编程（Eurus‑2‑RL+KodCode‑V1）、通用知识问答（Med‑QA、LogiQA、C‑Eval、Arc、LegalBench、MMLU、Webinstruct‑verified）以及Text‑to‑SQL（SynSQL‑2.5M）；

**📈 对比分析**

与基线（无RLVR或仅使用LoRA）对比，Progressive RLVR在所有域上实现了非空泛化上界，且上界落在基模型精度之上、接近微调后模型精度（误差在1%–6%以内），并通过消融实验验证了on‑policy蒸馏与TinyLoRA在压缩与精度平衡中的关键作用；

**⚠️ 局限性**

局限性包括：①泛化上界仅适用于训练分布相同的情况，对领域迁移缺乏理论保证；②目前仅在4B规模模型上验证，尚未证明能在更大（百亿级）模型上实现同等压缩与非空界；③计算成本仍高，需要教师训练与蒸馏的双阶段流程。

---

## 231. Physical Reservoir Signal Acquisition for Sub-Nyquist Waveform Reconstruction

**arXiv ID:** 2607.14504 | [PDF](https://arxiv.org/pdf/2607.14504v1)

**作者:** Yuito Ito `[一作]` (Kanazawa University), Satoshi Sunada `[通讯]` (Kanazawa University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种新的子奈奎斯特波形采集方法——储层信号采集（RSA），将物理储层作为动态测量装置，将未知宽带信号转换为多条低速采样轨迹并通过线性逆映射恢复原始信号。

**💡 创新点**

创新点在于：①将储层从计算引擎转变为测量设备，利用其内部多样化动态自发产生所需的测量基；②通过储层的测量多样性实现压缩感知，显著降低所需采样通道数；③在实验中实现了高达12.5 GHz的波形重构，仅需四路6.25 GSa/s ADC，相当于每路ADC的四倍带宽。

**🔧 技术方法**

技术上使用硅光子学“马戏剧”形状的微腔储层，配备13个输出波导；利用光学调制产生输入波形，光探测后在8位实时示波器上以6.25 GSa/s采样；逆映射由约7000个宽带随机预置信号的输入输出校准数据构建的伪逆矩阵实现。

**📊 数据集**

数据集包括：1）随机宽带波形（最高频12.5 GHz），2）多种典型波形（5 GHz正弦波、双音、Santa Fe混沌、Lang–Kobayashi激光混沌），3）100 ps短脉冲序列；所有信号均通过模拟与实验验证。

**📈 对比分析**

方法通过理论推导给出重构条件M≥N_R（N_R为欠采样比），并在实验中展示M=1–4时的重构误差随M变化的递减；在压缩感知情形下，NMSE随信号带宽比例下降而显著改善；最终实现NMSE低至0.044（M=4）并成功重构高频混沌信号。

**⚠️ 局限性**

局限性包括：①对储层的线性近似假设，非线性效应需进一步校正；②重构性能受储层脉冲响应时间和频率分布（特征频率扩展）限制；③高噪声或低信噪比的稀疏信号仍难以实现低误差重构；④需要大量校准信号且对设备温度、漂移等外部因素敏感。

---

## 232. Reinforcing Egocentric Spatial Perception in Multimodal Large Language Models via Ego Scene Augmentation

**arXiv ID:** 2607.14497 | [PDF](https://arxiv.org/pdf/2607.14497v1)

**作者:** Chi Kit Wong `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Ego Scene Augmentation (ESA) 框架，利用 Ego-element Graph 在不需要额外训练的前提下增强多模态大型语言模型的 egocentric 视觉问答能力。

**💡 创新点**

创新点在于把 Ego-element Graph 作为中间表示，将语义、深度、视觉区域和文本信息以有向图形式聚合，并通过插件化方式直接喂给现有 MLLM，从而显著提升 egocentric 场景的空间推理。

**🔧 技术方法**

主要技术包括使用 Depth Anything V2 进行单目深度估计、图像分割与文本识别得到节点属性、构建包含位置、语义、区域和文本属性的有向图，以及将图序列化为 JSON prompt 供 MLLM 直接使用。

**📊 数据集**

使用 EgoTextVQA 数据集（室内与室外两子集，共约1.5K视频片段和7K问答对）进行评估。

**📈 对比分析**

与 InstructBLIP、Emu3、LLaVA-NeXT、InternVL2、Qwen2-VL、Qwen2.5-VL 等基线模型对比，ESA 在室内场景提升 8.14% 的准确率、4.27% 的得分，室外场景提升 8.72% 的准确率、4.5% 的得分；在购物子集更是获得 15.91% 的准确率提升。

**⚠️ 局限性**

局限性包括对视角变化敏感、外部感知模块误差会传递到最终答案、仅提供序数深度（无度量几何）、缺乏时序聚合与不确定性处理。

---

## 233. Penny: Transition Network Analysis of Learner-Chatbot Interactions in Scaffolded EFL Writing

**arXiv ID:** 2607.14575 | [PDF](https://arxiv.org/pdf/2607.14575v1)

**作者:** Steve Woollaston `[一作]` (Kyoto University), Hiroaki Ogata `[通讯]` (Kyoto University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

使用转移网络分析（TNA）研究日本中学生在使用LLM驱动的写作聊天机器人Penny时的交互模式，分析了4,651次写作会话和21,061条聊天记录，发现了“修订循环”和“聊天循环”两大行为循环，并对高低熟练度学习者的交互差异进行量化；

**💡 创新点**

创新点在于首次将TNA方法应用于AI写作聊天机器人的交互数据，揭示了非线性、对话驱动的学习过程，并为不同熟练度学习者的个性化聊天机器人设计提供了经验依据；

**🔧 技术方法**

采用LLM（Penny的生成式大语言模型）与自然语言处理分类器对聊天日志进行自动编码，利用TNA Shiny应用构建并分析转移网络；

**📊 数据集**

数据集来自119名日本高中生在四个月内使用Penny的写作日志，包含4,651次写作会话、21,061条消息、63,328个事件，按中位数分为高熟练度（60人）和低熟练度（59人）两组；

**📈 对比分析**

通过比较转移概率、网络密度、互惠率、入度中心化等指标，并使用卡方检验（χ²=25.4，p<.003）检验两组间行为频率差异，结果显示高熟练度学习者更倾向对话循环，低熟练度学习者更依赖修订循环，差异显著；

**⚠️ 局限性**

局限性包括节点粒度粗糙无法区分聊天中的具体语气或意图、自动分类存在一定误差、研究仅限于日本高中语境、未评估长期写作效果及缺乏因果推断。

---

## 234. Gate-Zero Growth: A Geometric Framework for Function-Preserving Continual Learning

**arXiv ID:** 2607.14571 | [PDF](https://arxiv.org/pdf/2607.14571v1)

**作者:** Dante Lok `[一作]` `[通讯]` (Votee AI), Dante Lok (Votee AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在持续学习中提出了一种通过零初始化门控添加残差块的功能保持增长方法（gate‑zero growth），并给出了其几何理论基础

**💡 创新点**

创新点在于将零初始化门控的结构统一为一类功能保持操作，证明其在增长点实现秩分离与稀疏 Fisher 结构，从而在持续学习中实现近乎零遗忘

**🔧 技术方法**

使用了残差网络架构、零初始化门控、功能保持理论、Fisher 信息几何、以及基于门控的梯度投影等技术

**📊 数据集**

使用了 WikiText-103 与 BookCorpus 两个语言建模数据集（以及 MoE 的 706M→2.5B 规模实验）

**📈 对比分析**

通过与非功能保持控制（G_stack）、LoRA、零初始化堆叠、无增长、以及 Scratch 训练等对比，gate‑zero growth 在 Isolation 或 Freeze‑Nothing 配置下实现 Δ_A ≈0、PPL_B 下降至 28.41，显著优于非功能保持和其他方法

**⚠️ 局限性**

局限性包括：对稀疏 Fisher 的估计粗糙、MoE 迁移后可塑性不足（因 clone‑block 重叠），以及未对更大规模或多任务场景进行完整比较

---

## 235. MARS: Multi-hop Adaptive Retrieval and SPARQL Generation for KGQA

**arXiv ID:** 2607.14561 | [PDF](https://arxiv.org/pdf/2607.14561v1)

**作者:** Nikit Srivastava `[一作]` (Paderborn University), Axel-Cyrille Ngonga Ngomo `[通讯]` (Paderborn University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MARS，一种无须微调的KG问答系统，采用多跳检索+SPARQL生成的流水线，支持多语言；

**💡 创新点**

创新点在于使用模式化图检索与逐步上下文扩展，允许LLM在必要时决定继续检索还是直接生成SPARQL，显著降低 hallucination 并提升多跳推理能力；

**🔧 技术方法**

结合实体链接、文本增量、语义相似度检索、模式实例化、类型信息与可选的例子提示，利用开源LLM（GPT‑OSS、Qwen3、Gemma3）完成SPARQL生成与验证；

**📊 数据集**

在 Wikidata 基准上评估，使用 LC‑QuAD2.0（英语），QALD‑9‑plus（9 语种）和 QALD‑10（4 语种）三大数据集；

**📈 对比分析**

相较于 GRASP、DeepPavlov、UniQ‑Gen、MST5 等基线，MARS 在 QALD‑10 上取得最高 Macro F1（所有语言），在 LC‑QuAD2.0 上也领先，且在 QALD‑9‑plus 的多语言场景中大部分语言表现优于 GRASP；

**⚠️ 局限性**

局限主要在于结果截断、聚合/投影错误、过度保守的文字生成、对高度连通节点检索效率低下，以及对 KG 版本漂移和评测标准差异的依赖。

---

## 236. Seeing the End at Step Zero: Accelerating Diffusion MLLMs via MLP Sparsity-Aware Truncation

**arXiv ID:** 2607.14557 | [PDF](https://arxiv.org/pdf/2607.14557v1)

**作者:** Qicheng Zhao `[一作]` (Zhejiang University), Zheyu Yan `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Diffusion Multimodal Large Language Models（DMLLMs）中发现并利用Step‑0 MLP稀疏性信号，提出Seer框架通过一次性截断冗余后缀，配合混合执行路由实现高效推理。

**💡 创新点**

创新点在于：①首次在Diffusion LLM中揭示生成过程起始阶段即可识别语义边界；②基于SNR的训练无关阈值检测；③一-shot宏观截断结合GPU端聚合与CUDA Graph兼容的执行策略。

**🔧 技术方法**

主要技术包括：MLP稀疏性分析、SNR（Signal‑to‑Noise Ratio）检测、动态序列截断、Padding Waste Ratio（PWR）驱动的混合执行路径、设备驻留的Triton聚合内核以及基于批量分桶的执行路由。

**📊 数据集**

使用了九个多模态评测基准：MME、MMMU、MMBench、InfoVQA、ChartQA、ScienceQA、DocVQA、GQA 和 MathVista。模型覆盖 LaViDa‑LLaDA、LaViDa‑Dream、MMaDA 等主流 DMLLM。

**📈 对比分析**

与多种现有加速方法（D3ToM、RedVTP、VisionZip、MMTok、DivPrune、SparseVLM）对比。Seer 在保持或略提升准确率的同时，吞吐量提升至 1.6×~31×，延迟降低至 0.3~1.1×，在多模态任务上表现出最佳的效率‑质量平衡。

**⚠️ 局限性**

局限性：①仅针对文本侧后缀冗余，未解决视觉 token 的冗余问题；②对 SNR 阈值（τ_jump, γ）及 PWR 阈值（τ_pad）敏感，需要经验调参；③在极端短文本或极长文本场景下截断误判率略升；④依赖GPU端 Triton 内核，迁移至其他硬件/框架存在门槛。

---

## 237. 3D Geometric Tooth Alignment Planning via Deep Reinforcement Learning

**arXiv ID:** 2607.14544 | [PDF](https://arxiv.org/pdf/2607.14544v1)

**作者:** Yong Li `[一作]` (Zhejiang University), Haihua Zhu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于深度强化学习的3D牙齿几何对齐规划框架，将对齐过程建模为马尔可夫决策过程，自动生成安全、高效的牙齿运动路径。

**💡 创新点**

创新点包括：①将对齐任务视为连续动作的MDP；②使用Transformer主体的Actor‑Critic，并引入动态稀疏动作掩码实现临床稀疏调动；③采用双阶段课程学习策略，先粗略探索再细致优化。

**🔧 技术方法**

核心技术为Deep Deterministic Policy Gradient (DDPG)、Transformer网络、动态动作掩码、N步回报、优先经验回放、学习示例、GJK碰撞检测以及多头注意力机制。

**📊 数据集**

使用了来自Choho Technology的10,000条专家设计的正畸路径数据集（9,000条训练，1,000条测试），每条包含3D牙齿网格和目标位姿。

**📈 对比分析**

与IGWO、NeuralOrtho、TMDM三种基线方法比较，评估指标包括路径总位移、总旋转、违约次数、碰撞频率和路径长度差异。实验结果显示，本方法在碰撞和违约方面优于所有基线，路径效率排名第二，且生成路径与专家路径的长度差异最小。

**⚠️ 局限性**

主要限制包括：未尝试更先进的RL算法（如TD3、SAC）；超参数设置较多且缺乏系统化优化；仅关注几何约束，未将完整的生物力学与临床先验（如牙槽骨支撑、间隙调节）纳入模型。

---

## 238. Probabilistic "Copies" in Generative AI Models

**arXiv ID:** 2607.14532 | [PDF](https://arxiv.org/pdf/2607.14532v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 239. CityLLM: A framework for natural-language querying of semantic 3D city models

**arXiv ID:** 2607.14542 | [PDF](https://arxiv.org/pdf/2607.14542v1)

**作者:** Rabindra Lamsal `[一作]` (UNSW Sydney), Johnson Xuesong Shen `[通讯]` (UNSW Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出CityLLM框架，实现语义3D城市模型的自然语言查询与跨数据源可视化。

**💡 创新点**

创新点在于将CityJSON与OSM街网联结于LLM驱动的跨数据库链式查询流程，并通过可视化交互降低非专业用户门槛。

**🔧 技术方法**

采用LLM（GPT‑OSS、Gemini 3.1、GPT‑5.4）+ PostGIS + Neo4j + LangGraph + MapLibre实现查询、执行与可视化。

**📊 数据集**

使用Rotterdam CityJSON LoD2（853栋建筑）与对应范围的OSM街网及配套设施数据集。

**📈 对比分析**

通过54条自然语言查询与三种LLM对比，GPT‑OSS达到100%答案与可视化正确率，Gemini 3.1 94.4%，GPT‑5.4 85.2%，且平均重试次数≤3。

**⚠️ 局限性**

局限在于系统提示过长导致推理成本高、LLM在上下文保持和聚合推理上易出错，且仅在单一城市数据集上验证。

---

## 240. MIDI-RAE-JEPA: Hierarchical Representation Learning and Generation for Symbolic Music

**arXiv ID:** 2607.14537 | [PDF](https://arxiv.org/pdf/2607.14537v1)

**作者:** Scott H. Hawley `[一作]` `[通讯]` (Belmont University), Scott H. Hawley (Belmont University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于等变自监督学习的符号音乐表示框架MIDI‑RAE‑JEPA，结合 pitch/time 变换等变目标、LeJEPA 以及 Swin Transformer V2 对钢琴卷帘图像进行层级特征学习，随后通过冻结编码器训练解码器与条件流模型实现高精度重构与生成，并在情感分类上超越 Haar 及 DINOv2 基线

**💡 创新点**

首次将等变自监督目标与 Soft Factorization Loss 以及 SIGReg 结合用于符号音乐；通过多层级 Swin 编码器与 LeJEPA 架构实现层级化、可解释的音乐表示；利用冻结编码器的 RAE 方案在多任务上无需微调即可表现优异

**🔧 技术方法**

LeJEPA、自监督等变损失、Soft Factorization Loss、Swin Transformer V2、Masked Embedding Predictor (MEP)、SIGReg、冻结编码器的 RAE、流匹配生成模型 (Conditional Flow Matching)

**📊 数据集**

POP909 MIDI 数据集（909 首流行歌曲）以及 EMOPIA 情感标签数据集

**📈 对比分析**

在钢琴卷帘图像重构上达 0.995 的 F1 分数；在 EMOPIA 情感分类上，L3 级别线性探针分别在四分类、唤醒度、情感度上分别达到 0.488、0.754、0.640，均优于 Haar 及 DINOv2 基线；条件流生成在匹配与混合样本中能够保留音高与节奏特征，且无条件生成依旧保持音乐性

**⚠️ 局限性**

粗层级特征缺乏可解释的音乐抽象；仅针对钢琴卷帘图像，未扩展至更丰富的音频或其他乐器；在最大时间位移设置过大时可能导致重构精度下降，需进一步平衡细粒度与粗粒度学习

---

## 241. Predicting Human Visual Attention on Words in Source Code

**arXiv ID:** 2607.14535 | [PDF](https://arxiv.org/pdf/2607.14535v1)

**作者:** Chia-Yi Su `[一作]` (University of Notre Dame), Collin McMillan `[通讯]` (University of Notre Dame)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种计算模型，通过将人工神经网络内部注意力与眼动实验获得的真实视觉注意力对齐，来预测程序员在阅读源代码时对单词的视觉关注。

**💡 创新点**

创新点在于设计了新的损失函数，用来度量人类注意力与模型内部注意力之间的相似度，并在此基础上优化模型，从而显著提升了对注意力分布的预测精度。

**🔧 技术方法**

采用深度学习框架实现注意力模型，结合自定义损失函数和对齐机制，主要使用神经网络的内部注意力机制与眼动数据进行比较。

**📊 数据集**

使用了三组眼动实验数据集，分别包含两套Java代码集和一套C语言代码集，用于训练与评估模型。

**📈 对比分析**

通过与现有软件工程基线模型比较，采用Pearson相关系数衡量预测准确度，结果分别提高了64%、16%和467%；在扫描路径预测任务中，使用归一化Levenshtein距离评估模型优于基线，并且在阅读与写作任务中均优于Claude和GPT‑5。

**⚠️ 局限性**

局限性包括：实验仅覆盖Java和C两种语言，缺乏对其他语言或更大规模代码库的验证；模型主要关注词级注意力，可能无法充分捕捉代码语义与结构层面的深层认知；数据集样本量有限，可能导致模型过拟合或泛化性不足。

---

## 242. Semi-Streaming Matching in a Single Pass I: A New Framework for Lower Bounds via Blueprints

**arXiv ID:** 2607.14644 | [PDF](https://arxiv.org/pdf/2607.14644v1)

**作者:** Sepehr Assadi `[一作]` (University of Waterloo), Mars Xiang `[通讯]` (University of Waterloo)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新的蓝图框架，用于证明单次通行半流式最大匹配问题的近似下界，并利用该框架进一步提高了下界至约0.558，最终证明贪心算法的0.5近似已是最优

**💡 创新点**

核心创新在于引入蓝图（blueprints）概念，将复杂的RS图和信息论构造抽象为有限常数大小的图结构，从而大幅简化并统一了下界证明；同时通过蓝图实现了比以往更紧的近似比下界

**🔧 技术方法**

使用了图论（RS图及其扩展ERS图）、信息论（压缩与边恢复上限）、通信复杂度（单向多玩家通信游戏）以及构造与分析蓝图的组合技术

**📊 数据集**

该工作为理论工作，无需实际数据集；所有结论均基于抽象图构造与理论证明

**📈 对比分析**

与以往仅能达到约0.590的下界相比，新下界约0.558，进一步接近贪心算法的0.5，表明单次通行半流式最大匹配已达到最优；实验/数值验证仅通过构造蓝图和符号计算完成

**⚠️ 局限性**

局限性在于仅针对单次通行场景，尚未解决多次通行或随机顺序流式的情况；蓝图构造与优化仍是手工且规模受限的过程，缺乏自动化工具

---

## 243. A Deterministic Binary Fingerprinting Framework with Zero-Trained Feature Extraction for Sparse Count Matrices

**arXiv ID:** 2607.14596 | [PDF](https://arxiv.org/pdf/2607.14596v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 244. TIDE: Trustworthy and Interpretable Battery Degradation Estimation with Contextual Learning and Symbolic Distillation

**arXiv ID:** 2607.14640 | [PDF](https://arxiv.org/pdf/2607.14640v1)

**作者:** Wen Yang Tan `[一作]` (Singapore Institute of Technology), Elisa Y. M. Ang `[通讯]` (Singapore Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为Tide的多模态、可解释且可靠的电池状态估计框架，利用知识引导的先验、单调残差和上下文残差三元结构来实现电池健康状态（SoH）估计，并进一步通过符号蒸馏生成简洁的数学表达式

**💡 创新点**

通过将电池退化知识与实际监测数据相结合，构建了可解释的三组件骨干网络，且对残差施加单调约束保证退化一致性；同时引入符号蒸馏，使整体模型在保持高精度的同时实现全局可解释性

**🔧 技术方法**

采用知识引导的退化先验、Softplus 单调残差、MultKAN 上下文残差、端到端联合训练以及 PySR 符号蒸馏；训练中使用多种正则化（周期性、内阻、范围、辅助）

**📊 数据集**

在公开 MIT‑Stanford 快速充电锂电池循环寿命数据集上进行实验，包含140块电池、114,738个循环级记录，提取34个安全部署特征

**📈 对比分析**

与CNN‑BiGRU、CNN‑BiLSTM、GRU、LSTM、TCN、Vanilla KAN等基线对比，Tide 的 RMSE 为 0.0081、R² 为 0.964，平均提升约 19.7%，且在退化一致性（MVR）上达 0%（基线 30‑60%），符号蒸馏后模型误差仅提升 0.0048

**⚠️ 局限性**

模型依赖于电池内部阻抗、循环次数等高质量特征，若测量噪声大或缺失会影响性能；符号模型虽简洁但可能过度简化复杂交互，且对不同电池类型的泛化能力仍需进一步验证

---

## 245. PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference

**arXiv ID:** 2607.14618 | [PDF](https://arxiv.org/pdf/2607.14618v1)

**作者:** Hyunwoo Oh `[一作]` (University of California), Mohsen Imani `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向CPU的低精度LLM推理框架PolyQ，通过激活感知通道级混合精度量化与编译器协同，支持可指定的平均比特预算并实现细粒度的2~16位位宽分配。

**💡 创新点**

创新点在于将CPU对齐的多比特调色盘与水饺式水位分配结合，并在编译时完成通道排列、块化与跨算子Permutation合并，实现了细粒度预算控制与高效执行的统一。

**🔧 技术方法**

技术包括基于激活能量的误差代理、贪婪水饺分配、ISA感知量子匹配、位宽块化内核生成以及图级Permutation规划与合并。

**📊 数据集**

实验使用WikiText‑2作为校准与评估数据集，并在Llama2‑13B、Falcon‑H1‑3B和Qwen3‑32B等模型上验证。

**📈 对比分析**

与AWQ、GPTQ、Slim‑LLM、AMQ等基线相比，在3b目标下PolyQ提升困惑度2.4–32.1%，内存占用相较W3降低约10–20%，激活重排流量减少高达70.8%，吞吐量与能耗与传统LUT路径仅相差≤2%。

**⚠️ 局限性**

局限性在于仍依赖静态图编译，难以适应动态形状或更低比特（<2）场景，并且对不同CPU架构的后端支持需要进一步扩展。

---

## 246. Investigating first-language bias in LLM-based automated essay scoring: A cross-prompt evaluation of an open-weight AI-model on TOEFL essays

**arXiv ID:** 2607.14605 | [PDF](https://arxiv.org/pdf/2607.14605v1)

**作者:** John Maurice Gayed `[一作]` `[通讯]` (Waseda University), John Maurice Gayed (Waseda University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 LoRA 微调的 Gemma‑3‑27B‑it 模型在 8 个未见写作提示和 11 种第一语言背景下的自动写作评分性能，并对跨提示泛化和 L1 公平性进行系统评估。

**💡 创新点**

首次在大规模开源 LLM 上对全 TOEFL11 语料进行跨提示与全面 L1 公平性分析，揭示欧洲语言与东亚语言写作者在同一熟练度层内的系统评分偏移。

**🔧 技术方法**

采用 LoRA 微调的 Gemma‑3‑27B‑it 大模型，使用 llama.cpp 推理引擎、Q4_K_M 4‑bit 量化、温度 0 的推理配置，并将模型原始分数映射到 ETS 的三等级熟练度。

**📊 数据集**

训练集为 480 篇来自两个提示的 ETS TOEFL 写作样本；测试集为公开的 TOEFL11 语料 12,100 篇，覆盖 8 个未见提示和 11 种第一语言，每种语言 1,100 篇。

**📈 对比分析**

通过准确率、二次加权 Kappa (QWK) 与按提示和 L1 分层的准确率比较评估，模型在未见提示下 QWK 0.677–0.730、整体准确率 77.79%，但在各 L1 内表现出显著偏移，欧洲语言得分偏高、东亚语言得分偏低。

**⚠️ 局限性**

局限在于仅使用单一开源模型，评估基于粗糙的三等级标签，无法判断偏移是实际熟练度差异还是模型偏差；L1×等级单元数不足导致部分估计不稳定，且未做多模型比较或定性错误分析。

---

## 247. MagicPrompt: Ultra-Lightweight Prompt Tuning for Video Generation

**arXiv ID:** 2607.14595 | [PDF](https://arxiv.org/pdf/2607.14595v1)

**作者:** Yinhan Zhang `[一作]` (Hong Kong University of Science and Technology), Anyi Rao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种极轻量化的参数高效微调框架MagicPrompt，用于在不修改预训练权重的前提下快速适配大型视频扩散模型

**💡 创新点**

创新点在于（1）将可学习的soft prompt直接嵌入自注意力与交叉注意力的key‑value通道，实现对注意力分布的精准引导；（2）引入双空间奖励反馈（像素空间的HPS/MPS奖励 + 潜空间的CFG‑vs‑无导向相似度），在高噪声阶段也能获得稳定的优化信号；（3）通过可学习的shift‑bias进一步校准潜在分布，提升时间一致性

**🔧 技术方法**

采用Attention‑Embedded Prompt Tuning与Dual‑Space Reward Feedback Optimization，并结合CFG、VAE解码、CLIP对比度等现有技术

**📊 数据集**

使用公开的OpenVid（文本‑视频对）、OpenHumanVid（控制‑视频）以及TikTok视频数据集进行实验

**📈 对比分析**

与LoRA、VACE、完整Fine‑Tuning等基线在文本‑视频、图像‑视频、控制‑视频三大任务中对比；MagicPrompt在CLIP、LPIPS、FID、FVD等指标上与LoRA相当或更优，同时可训练参数比例低于1%（相对LoRA低数倍）

**⚠️ 局限性**

局限性包括：soft prompt可能导致身份漂移或多场景切换（尤其在大型模型上）；shift‑bias虽提升一致性但对部分场景会产生过度校正；需要奖励信号的设计与调参，且在极少量样本下仍可能出现过拟合

---

## 248. MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents

**arXiv ID:** 2607.14651 | [PDF](https://arxiv.org/pdf/2607.14651v1)

**作者:** Jifeng Gao `[一作]` (Nanjing University), Sanglu Lu `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 MemPoison 的大规模、手工验证的持续内存毒化基准（1227 条案例），并针对不同攻击难度（L1‑L3）、注入通道、内存架构和攻击类型进行系统评估；提出了机制影响分解（MID）框架来解释不同防御策略的失效机制，并绘制了写时防御的“防御前沿”。

**💡 创新点**

① 提出了三层攻击难度分类（直接单条、组合多条、上下文触发）；② 设计并公开了全面覆盖多维度的 MemPoison-Bench；③ 开发了 MID 机制解释工具，揭示写时防御的结构盲点；④ 通过大规模实验描绘写时防御与检索时防御的性能前沿。

**🔧 技术方法**

基准构建技术（人类交互种子 → 语义单元 → 多表面化 → 手工验证），机制影响分解（单条影响 Δ^s、交互影响 Ω^g、触发激活 Shift）、多模型、多防御方案的评估框架（CleanAcc、BCR、AR、UR 等指标）。

**📊 数据集**

1227 条手工验证的毒化案例，覆盖 4 种攻击目标（事实、偏好、指令、状态）、3 种注入通道（用户输入、工具输出、代理间消息）、3 种内存结构（扁平块、事实存储、层次笔记），并在 10 种公开/闭源 LLM（包括 Qwen、Llama、DeepSeek、GPT‑4o/5、Gemini‑3 Flash）上进行测试。

**📈 对比分析**

对比无防御（None）与多种写时/检索时防御（一致性检查、异常过滤、重加权、PromptGuard、LLMJudge Write、EraseAndCheck、SmoothLLM、PPL、Memory Sanitization 等），使用 CleanAcc 与 BCR 两大指标。实验表明：无防御 BCR≈62.5%；最佳防御（MIXed）将 BCR 降至 10.7%，但在 L2、L3 攻击下仍保持 20–40% 的残余毒化；写时防御对 L1 有显著抑制（BCR↓≈30%），对 L2/L3 效果有限。

**⚠️ 局限性**

仅覆盖三种常见内存结构，未考虑时间衰减、动态摘要、访问控制等更复杂的内存管理；仅针对文本毒化；未包含参数级、侧信道或基础模型权重篡改；基准与实验受限于所选 LLM 与防御实现，难以覆盖所有实际代理系统。

---

## 249. Action QFormer: Structured Representation Shaping under Action Supervision in Vision-Language-Action Models

**arXiv ID:** 2607.14635 | [PDF](https://arxiv.org/pdf/2607.14635v1)

**作者:** Yufeng Ji `[一作]` (Shanghai Qizhi Institute), Zhongyu Li `[通讯]` (Hong Kong Embodied AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Action QFormer，通过指令条件查询重组继承的多模态表示，实现在零样本 sim‑to‑real 导航中闭环任务成功率显著提升的 VLA 结构化表示塑形方法。

**💡 创新点**

创新点在于引入可学习查询接口，在动作监督梯度传播前提供中间路由，既能塑造动作面向的表示，又能减轻对语言侧表示的破坏，从而解决直接融合导致的表示不稳定问题。

**🔧 技术方法**

技术上结合 Transformer 的自注意力与交叉注意力实现查询更新，并采用条件扩散策略进行动作预测；通过梯度阻断实验剖析动作监督对表示的影响。

**📊 数据集**

数据集使用 Habitat 的 ObjectNav 作为训练模拟数据，零样本直接在真实室内场景部署；中间指令由 GPT 自动生成并用作监督。

**📈 对比分析**

在四个真实场景闭环实验中与直接融合基线对比，Action QFormer 将任务成功率从 18.8% 提升至 56.3%，固定指令动作正确率从 22.5% 提升至 75.5%，并几乎消除 OOD 指令生成。

**⚠️ 局限性**

局限性包括对需要显式避障的复杂场景仍易失效；对更大动作空间或高度动态任务的适应性未验证；模型高度依赖预训练多模态背骨，跨域更大视觉差异的鲁棒性仍需提升。

---

## 250. ExaGEMM: Exploration Framework for CPU-Driven ML Inference via Associative In-Register Computing for Low-Bit GEMM

**arXiv ID:** 2607.14622 | [PDF](https://arxiv.org/pdf/2607.14622v1)

**作者:** Hyunwoo Oh `[一作]` (University of California, Irvine), Mohsen Imani `[通讯]` (University of California, Irvine)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对CPU低位GEMM的工作负载感知协同设计与探索框架，利用寄存器内查表技术实现低位运算，并通过分析模型快速筛选可实现的SIMD扩展点。

**💡 创新点**

创新点在于将支持点选择视为硬件设计问题，将硬件开销（选择网络尺寸）与计算效率共同建模，并通过层级分析自动生成非支配的支持前沿；同时提出仅增添寄存器内选择网络的轻量级SIMD扩展方案。

**🔧 技术方法**

技术包括寄存器内查表（LUT）执行、可参数化的结构映射 (e,b,r,c,u)、指令级合并 (ILM)、基于寄存器可行性、指令计数、内存流量与硬件开销的分析模型，以及gem5与TSMC28nm ASIC综合验证。

**📊 数据集**

实验数据集涵盖DeiT‑B（ViT，W1A8）、Llama2‑7B（LLM，W2A16）以及混合精度Llama2‑7B（W{1,2,4}A16）三种量化配置，验证跨x86和ARM SIMD宽度（128/256/512位）的适用性。

**📈 对比分析**

与软件基准（T‑MAC、Vec‑LUT）和单一固定扩展点比较，所提框架在预填充阶段最高可达13.3×的延迟加速、7.8×的解码吞吐提升，且在不同SIMD宽度上显示出更显著的LLM加速优势。

**⚠️ 局限性**

局限性包括：仍需人工参与支持点的最终选择；分析模型依赖于精确的硬件和内存层级参数，可能在极端硬件配置或未覆盖的量化模式下失准；框架目前针对CPU，未涵盖GPU/FPGA等其他体系结构。

---

## 251. Beyond Entropy: Correctness-Aware Advantage Shaping via Contrastive Policy Optimization

**arXiv ID:** 2607.14614 | [PDF](https://arxiv.org/pdf/2607.14614v1)

**作者:** Weiwen Xu `[一作]` (Chinese University of Hong Kong), Hao Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的框架，称为对比策略优化（CPO），利用参考引导和普通生成之间的对比性不一致性作为正确性感知信号进行优势塑造。

**💡 创新点**

创新点在于引入了对比性不一致性作为比熵更可靠的标记级正确性信号，并且CPO有效解决了现有方法中的零优势问题。

**🔧 技术方法**

使用了对比性不一致性和强化学习（RL）技术，结合了参考引导的生成分布与普通生成分布的对比。

**📊 数据集**

在MATH数据集上进行实验，评估了在领域内和领域外的基准测试。

**📈 对比分析**

与标准的RLVR方法（如GRPO）和熵干预方法进行比较，CPO在数学推理任务上显著优于这些方法，同时保持了良好的领域外泛化能力。

**⚠️ 局限性**

限制在于对比性不一致性可能在某些情况下无法完全捕捉到所有类型的错误，且在处理复杂任务时可能需要更多的计算资源。

---

## 252. Representation-Aligned Tactile Grounding for Contact-Rich Robotic Manipulation

**arXiv ID:** 2607.14609 | [PDF](https://arxiv.org/pdf/2607.14609v1)

**作者:** Ruilin Chen `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在触觉增强的视语行动模型中，作者通过对未来触觉预测进行表征对齐，提升了接触丰富操作的性能。

**💡 创新点**

创新点在于：①利用线性探针诊断发现中间动作专家表征是最能预测未来触觉的接口；②设计轻量级潜在触觉预测器（LTP）在该接口上进行监督，避免了对噪声触觉原始信号的直接预测。

**🔧 技术方法**

技术包括：线性探针诊断、潜在触觉编码、流式动作专家（flow-based action expert）、SmolVLA与π₀视觉-语言-动作框架、以及LoRA微调。

**📊 数据集**

使用了五个真实世界的接触丰富操作数据集，任务包括USB驱动器插入、白板擦拭、插头插入、灯泡拆卸和可变形物体抓取，每个任务收集约50条专家演示。

**📈 对比分析**

与未使用触觉、仅输入触觉、在VLM层或最终动作层做未来触觉预测的基线相比，表征对齐的LTP在SmolVLA上平均成功率提升至74%，在π₀上提升至73%，均显著优于其它方法。

**⚠️ 局限性**

局限包括：仅在已知任务环境下验证；对不同传感器类型和噪声鲁棒性的评估不足；需要进一步探索在更大规模模型或更复杂任务中的可扩展性。

---

## 253. A Nonlinear Model Predictive Control Perspective on Gradient-Based Optimization: A New Efficient, Parameter-Free and Provably Stable Algorithm

**arXiv ID:** 2607.14600 | [PDF](https://arxiv.org/pdf/2607.14600v1)

**作者:** Mazen Alamir `[一作]` `[通讯]` (University of Grenoble Alpes, Centre National de la Recherche Scientifique, Grenoble Institute of Technology, GIPSA-lab), Mazen Alamir (University of Grenoble Alpes, Centre National de la Recherche Scientifique, Grenoble Institute of Technology, GIPSA-lab)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新型的梯度优化算法（Search and Accelerate，SaA），用于盒约束优化问题，并在其上实现了针对非线性模型预测控制（NMPC）的实时求解。

**💡 创新点**

创新点包括：①结合了新的沿梯度方向和沿Nesterov加速方向的线搜索；②引入信赖域（trust‑region）自适应机制；③提供了严格的收敛证明；④算法默认参数即可获得鲁棒性能，无需手动调参。

**🔧 技术方法**

使用了梯度下降、加速（Nesterov）技术、离散化线搜索、信赖域调整以及投影投射操作；同时在NMPC实验中利用多射击（multiple‑shooting）框架与fatrop内部库对比。

**📊 数据集**

构建并公开了600个随机生成的盒约束多项式优化实例（维度2–1000，阶数2–12），以及基于PVTOL飞行器的NMPC案例。

**📈 对比分析**

通过在该基准上与FGM、FISTA、Strong Armijo‑Wolf以及fatrop等现有方法进行定量比较（迭代次数、最终成本、计算时间、收敛速率），结果显示SaA在成本和时间上均明显优于对比算法，尤其在迭代次数受限的实时情境中表现突出。

**⚠️ 局限性**

主要局限包括：①算法仍基于盒约束，无法直接处理一般非盒约束；②在纯Python实现中存在额外的函数调用开销，编译版本需额外编译时间；③对非常高维或高阶多项式的极端情况仍需进一步验证。

---

## 254. Memory-Driven Self-Disclosure and Relational Turning Points: A Longitudinal Multimodal Study of Human-AI Interaction

**arXiv ID:** 2607.14593 | [PDF](https://arxiv.org/pdf/2607.14593v1)

**作者:** Ryuichi Sumida `[一作]` (Kyoto University), Yoichi Matsuyama `[通讯]` (Waseda University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在24名大学生参与的10天10次会话中，对记忆增强的对话式AI（InteLLA）进行纵向多模态实验，收集文本、音频、视频特征并让受试者在每次会话后评估熟悉度、自我披露、记忆感知、对话质量和娱乐感。

**💡 创新点**

首次将关系发展分为慢速累积与突发转折两层，揭示记忆感知是跨会话桥梁，且正向与负向转折在可观测性、可预测性和持续性方面呈不对称；为自适应AI提供双时域监测与干预的设计思路。

**🔧 技术方法**

使用基于GPT‑4o‑mini的RAG记忆式对话系统；从对话中提取351维文本、音频、视频特征；采用固定效应面板、交叉滞后模型、延迟中介分析以及弹性网络逻辑回归进行事件检测与预测。

**📊 数据集**

数据集包括24名英语熟练大学生的10天10次会话，共约2270条自评评分，351维行为特征和202次会话间变化，另配有半结构化访谈文本。

**📈 对比分析**

采用留一参与者交叉验证的弹性网络逻辑回归，评估AUPRC；检测任务AUPRC范围0.07–0.30，正向转折在会话内更易检测，负向转折在部分情况下可前瞻预测；整体性能为概念验证级别。

**⚠️ 局限性**

局限性包括样本量仅24人、仅研究一款记忆式LLM代理、受试者为英语母语大学生、摄像头设备差异导致测量噪声、仅覆盖五个关系维度且未涵盖信任、情感依恋等其他重要维度。

---

## 255. Multi-LLM Collaborative MRI Report Generation for Visual Instruction Tuning in Brain Oncology

**arXiv ID:** 2607.14581 | [PDF](https://arxiv.org/pdf/2607.14581v1)

**作者:** Sinyoung Ra `[一作]` (Sungkyunkwan University), Hyunjin Park `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建3D脑肿瘤图像-文本数据集，并通过多LLM协同生成高质量报告，训练3D视觉语言模型完成报告生成与视觉问答任务。

**💡 创新点**

首次将多LLM协同校验与3D图像-文本对构建结合，用3D视觉指令调优提升脑MRI报告质量与问答准确度。

**🔧 技术方法**

使用ChatGPT4o-mini、Claude、Gemini、DeepSeek等LLM进行文本生成与校验，VQ-GAN编码3D MRI、3D Perceiver、LoRA微调与视觉指令调优。

**📊 数据集**

采用公开的BraTS2021-GLI和BraTS2023-MEN脑肿瘤MRI数据集进行数据集构建、模型训练与评估。

**📈 对比分析**

与2D/3D基线模型（LLaVA、LLaVA‑Med、ChatGPT4o-mini、M3D）对比，报告生成的BLEU/ROUGE/METEOR/BERT‑F1显著提升，视觉问答准确率达91.7%（glioma）/88.9%（meningioma）。

**⚠️ 局限性**

缺乏真实专家评估导致临床准确性未验证，数据自生成可能引入自洽偏差，且在高度标注的BraTS数据上训练，泛化到真实临床数据尚未确认。

---

## 256. Advanced Image Generation: Negative Prompt Optimization and Latent Classifier Guidance

**arXiv ID:** 2607.14580 | [PDF](https://arxiv.org/pdf/2607.14580v1)

**作者:** Vaddi Charan Sai Nandan Reddy `[一作]` (PES University), Chandana M S `[通讯]` (PES University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套结合负向提示优化和潜在空间分类器引导的Stable Diffusion生成框架

**💡 创新点**

创新点在于自动生成负向提示并在潜在空间使用CNN-RNN混合分类器动态回滚低质量更新

**🔧 技术方法**

使用Seq2Seq微调的T5生成负向提示，DDIM调度器的Stable Diffusion，CNN–RNN混合分类器以及Streamlit界面

**📊 数据集**

使用自己构建的合成潜在序列数据集（200样本、10步序列）以及公开图像数据集 ImageNet‑100、CIFAR‑10 进行实验

**📈 对比分析**

与传统CFG、DDIM、Logistic 回归等基线对比，实验表明图像质量和语义一致性有所提升，但分类器 AUC 低于 0.5，性能仍有待提升

**⚠️ 局限性**

主要限制在于潜在分类器判别性能差，AUC<0.5，表明模型需要进一步改进或改进训练数据

---

## 257. Sharp Stability Threshold and Certification for Designing Stable Residual Architectures

**arXiv ID:** 2607.14576 | [PDF](https://arxiv.org/pdf/2607.14576v1)

**作者:** Hyemin Gu `[一作]` (University of Massachusetts Amherst), Markos A. Katsoulakis `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了“次线性增长原则”（sublinear‑growth principle），给出残差网络中每个残差块的速度场必须满足的输入幅值指数 q ≤ 1 的稳定性阈值，并通过 ODE 与最优控制理论两条独立证明来阐明该阈值的必要性与充分性。随后提出一种输入幅值指数的算术规则，可在原语级别上快速计算每个残差块的 q 并进行稳定性认证；最后在两种时序预测模型（Mamba 与 PatchTST）上实施多种实验，包括去掉归一化、预层归一化、后层归一化以及无归一化但改造为 q = 1 的线性增长版本，验证了 q ≤ 1 的稳定性判据。

**💡 创新点**

创新点：
1. 给出残差网络中速度场的“次线性增长”阈值 q = 1，首次在 ODE 与最优控制框架下给出必要且充分的稳定性条件。
2. 通过输入幅值指数的算术化简，使得任何残差块的 q 可以在原语层面上直接计算并进行结构化认证。
3. 在实验上展示了不依赖归一化层即可实现稳定训练，只要保持 q ≤ 1，从而拆分了归一化与稳定性之间的关系。
4. 对 Mamba 原始的超临界 q = 5 进行参数无关的线性增长改造，将其压缩到 q = 1，验证了理论的可操作性。

**🔧 技术方法**

技术与方法：
- ODE 理论中的存在性与无爆炸性分析。
- 最优控制（Hamilton–Jacobi–Bellman）框架，推导出“bang‑bang”最优速度与 q ≤ 1 的关系。
- 输入幅值指数算术规则（序列组合、并行求和、Hadamard 乘积、残差包裹、归一化包裹）。
- 预训练与微调，使用 RMSNorm、Pre‑LN、Peri‑LN、线性增长改造等归一化或结构改造策略。
- 计算机实现：PyTorch；使用 L2/Scale、RMSNorm、LayerNorm、MLP、注意力等原语。

**📊 数据集**

数据集：
- Weather（气象时间序列）
- ETTm1（工业时间序列）
两者均采用 96 步长、不同预测 horizon（96、192、336、720）和浅层残差深度（3、8）进行评估。

**📈 对比分析**

比较与性能：
- 通过 “blow‑up” 计数评估稳定性，发现所有 q ≤ 1 变体在 288 个随机种子中均无爆炸；q > 1（Mamba free‑velocity）在大部分配置中出现爆炸。
- 在 MSE 上，所有 q ≤ 1 变体在 Weather 与 ETTm1 上均与基准相差 ≤ 1.5%（Weather）/ ≤ 4.6%（ETTm1），与归一化与否无显著差别；PatchTST 的 free‑velocity 亦保持稳定且误差与基准相当。
- 结果表明，输入幅值指数 q 是决定稳定性的核心特征，而非单纯的归一化存在与否。

**⚠️ 局限性**

局限与未来工作：
- 仅在浅层残差网络（深度 3–8）上验证，需在更深网络与更复杂任务中进一步检验。
- 目前已知的原语仅覆盖 q = 0 与 q = 1，缺少严格位于 (0,1) 的原语，导致 q 在 (0,1) 的潜在优势未被探索。
- 理论推导基于连续时间与离散时间的相对保守估计，实际实现中可能受数值范围、激活函数细节影响。
- 研究聚焦于时间序列预测，需扩展到 NLP、CV 等更广泛的深度学习任务以验证通用性。

---

## 258. SoftNav: Injecting 3D Scene Tokens into VLMs for Embodied Navigation

**arXiv ID:** 2607.14586 | [PDF](https://arxiv.org/pdf/2607.14586v1)

**作者:** Yi Wu `[一作]` (Zhejiang University), Guang Li `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种将实体级3D连续表示直接注入视觉语言模型（VLM）隐藏空间的软令牌投射方法，以实现目标导向导航。

**💡 创新点**

通过轻量级MLP投射将3D场景编码器（PQ3D）的查询嵌入作为软令牌注入VLM，从而弥补文本序列化导致的表示缺口，并且仅需约17M可训练参数和1200条样本即可训练。

**🔧 技术方法**

使用冻结的PQ3D 3D场景编码器、冻结的Qwen2.5‑VL‑3B VLM、轻量级MLP投射、LoRA适配器以及视觉压缩模块（DINOv3+PatchMerger）等技术。

**📊 数据集**

在HM3D‑OVON、GOAT‑Bench、SG3D等3D导航基准以及真实世界的Unitree Go2机器人上进行评测。

**📈 对比分析**

与现有最先进方法（如MTU3D、TANGO、Uni‑NaVid等）在HM3D‑OVON上对比，SR提升至74.2%/68.3%/66.7%，SPL提升至33.9%/28.6%/25.7%；在GOAT‑Bench和SG3D实现零样本泛化，现实机器人成功率达63.3%。

**⚠️ 局限性**

依赖于冻结的PQ3D和VLM，对更大规模数据或更复杂3D编码器的适配未知；每步约1秒推理延迟；当3D编码器无法检测目标时表现不佳。

---

## 259. D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding

**arXiv ID:** 2607.14647 | [PDF](https://arxiv.org/pdf/2607.14647v1)

**作者:** Tianyu Liu `[一作]` (Tencent), Jianchen Zhu `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于跨请求自适应剪枝的规范解码调度策略，能够在高并发场景下显著提升大语言模型推理吞吐量。

**💡 创新点**

创新点在于将验证预算视为全局资源，利用草稿生成器的置信度对所有请求的草稿位置进行全局排序，并根据预先收集的硬件成本曲线动态选择最优验证深度，从而在保持输出分布不变的前提下最大化速度提升。

**🔧 技术方法**

核心技术包括：①使用块并行扩散式草稿器（如DFlash）生成完整草稿块；②基于置信度的期望前进量估计；③利用硬件成本表实现运行时自适应预算分配；④在每步中将选出的高置信度位置打包到目标模型进行验证。

**📊 数据集**

实验使用了六种目标模型（Llama‑3.1‑8B、Qwen3‑4B/8B/5‑27B/5‑35B、Hy3‑295B），并在数学推理（GSM8K、Math500）、代码生成（HumanEval、MBPP）以及多轮对话（MT‑Bench）等五个基准上评测。

**📈 对比分析**

与原始DFlash、EAGLE‑3、MTP以及标准自回归（AR）对比，所提方法在高并发下平均吞吐量提升约 1.65×（比 DFlash 1.26×高），在 MoE 模型上可达 3.0× 的速度提升，并在采样温度为1时仍保持 1.15× 的 AR 速度优势。

**⚠️ 局限性**

主要局限是每步动态剪枝会产生可变形状的验证批次，当前推理引擎（如 Spec‑V2、CUDA‑graph 捕获）假设批次形状固定，需进一步的系统集成和协同设计以充分释放其性能潜力。

---

## 260. Analytic Abduction: Causal Decomposition and Governed Commitment for Human--AI Coordination

**arXiv ID:** 2607.14641 | [PDF](https://arxiv.org/pdf/2607.14641v1)

**作者:** Remo Pareschi `[一作]` `[通讯]` (University of Molise), Remo Pareschi (University of Molise)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文提出了分析式溯因（Analytic Abduction）框架，并将其核心治理机制 κ–τ（κ 交互、τ 约束）引入到因果分解过程，构建了因果集群（Causal Cluster）结构，用双层交互架构（内层 κ*、外层 κ**）实现可持续的、风险感知的承诺决策，最终形成悬挂分解（Suspended Decomposition）作为人机协同的共享协调对象。

**💡 创新点**

创新点包括：
• 将 κ–τ 机制迁移到分析式溯因，允许候选因子共存并在满足治理条件时才承诺；
• 引入因果集群，记录因子权重和内部交互，保留因果结构信息；
• 双层交互架构（κ* 与 κ**）区别内层因子共振与跨分解竞争，阈值 δ(τ) 保障分解间差异足够时才允许承诺；
• 悬挂分解作为可读的协调对象，为多代理系统提供结构化的非承诺信息，抵御预早收敛，尤其在对抗性情境下形成防御。

**🔧 技术方法**

技术手段主要包括：
• 量子溯因（QA）框架的 κ–τ apparatus；
• 向量空间投影与余弦相似度；
• 关系聚合器 Ψ（支持有结构的 ex- planandum Φ）；
• 通过 LLM（如 GPT/LLM）生成候选分解；
• 人工校准 κ（交互矩阵）和 τ（阈值）及 κ**（跨分解竞争度）；
• 计算 cluster score sc_S，使用 η 调节内部一致性；
• 双层交互架构 κ*（内部）与 κ**（外部）实现分解竞争与协调。

**📊 数据集**

主要使用的案例数据集为：
• 流行病危机数据——病例、地理、年龄分布、二级传播模式等，构成 Φ；
• 网络攻击行为序列——MITRE ATT&CK 观测、时序、目标、操作签名等；
两组数据均来自公开案例描述和模拟，嵌入使用 Sentence‑BERT 或 GloVe 词向量，后续计算按论文所述实现。

**📈 对比分析**

对比方法：基线为单一概率分布或置信度评分的归纳式溯因。性能评估以分解分数、阈值 τ、分离距离 δ(τ) 为指标。示例结果显示：
• 在流行病案例中，三个候选分解分数相近，且差距低于 δ(τ)，系统保持悬挂状态，避免错误干预；
• 在网络攻击案例中，最高得分的分解与竞争分解差距仅 0.02 < δ(τ)=0.10，系统同样未做承诺，提示需进一步证据。虽然未给出精确的准确率或时间成本，但演示证明该框架能够在多模态、不确定环境下抑制过早决策。

**⚠️ 局限性**

局限性：
• 依赖 LLM 产生的候选分解，若缺失真实因子则无法识别；
• κ、κ**、τ 的设定需要人工专家校准，具有主观性；
• 缺乏完整的先验因子分布或结构化知识库，难以自动生成所有可能因子；
• 对抗情境下，精细设计的欺骗仍可能诱导错误分解；
• 框架仅提供可读的非承诺信息，实际决策仍需人机协同，无法完全自动化；
• 未实现完整的多代理系统架构，需后续工作整合消息、接口等实现细节。

---

## 261. Knowing You at First Glance: Inferring Apparent Personality from Faces

**arXiv ID:** 2607.14631 | [PDF](https://arxiv.org/pdf/2607.14631v1)

**作者:** Shuhuan Chen `[一作]` (Chinese Academy of Sciences), Zhen Lei `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GlanceFace，面向首次交互的面部表面性格（MBTI）推断框架，直接利用人脸图像预测人们的性格偏好。

**💡 创新点**

创新点在于（1）将视觉‑语言模型的语义先验与面部特征通过差分门控注意力融合，实现对细微表面性格线索的增强；（2）设计不确定性感知学习策略（UAPL），通过投票权重、熵置信度和软标签匹配来降低主观标注噪声的影响。

**🔧 技术方法**

使用 Qwen3‑VL‑Embedding‑2B 生成语义嵌入，改进 ResNet‑18 作为视觉骨干，构建 SEFR 模块和 gated FFN；训练时加入 AdaFace 损失保证身份一致性，采用 SGD 与余弦学习率衰减。

**📊 数据集**

数据集由 MS1MV3、CelebA、VGGFace2、IMDB‑Face 等四大人脸数据集构成的人脸图像与 Personality Database 中的 MBTI 软标签组成，训练集 3,092 个人/568,675 张图，评测集 216 个人/39,788 张图。

**📈 对比分析**

在 16‑类 MBTI、四维二分类及人/图级别指标上与 IR‑18、IR‑50、MobileFace、ViT‑S、CosFace、ArcFace、AdaFace 等基线比较，GlanceFace 在 Top‑1/3/5 Accuracy、F1、AUC 均取得最高分（人级 Top‑1 26.39%、AUC 78.19%），并在最难维度 S/N、F/T、J/P 上表现尤为突出。

**⚠️ 局限性**

局限性：1) 仅依赖静态人脸图像，难以捕捉动态表情与行为；2) MBTI 标注本质为主观感知，存在标注噪声与文化偏差；3) 模型对不同人种、光照等外部因素的鲁棒性未充分验证；4) 解释性依赖 Grad‑CAM，未能直接映射到真实心理特质。

---

## 262. Routing Ceilings Are Domain-Independent: Structural Prior Injection in Code Security Vulnerability Detection

**arXiv ID:** 2607.14628 | [PDF](https://arxiv.org/pdf/2607.14628v1)

**作者:** Manuel Israel Cázares `[一作]` `[通讯]`, Manuel Israel Cázares

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者在源代码漏洞检测任务中，利用手工编写的结构化先验（cheatsheet）注入来验证 LLM 的“路由假设”，并评估其在合成数据与真实 CVE 数据集上的表现。

**💡 创新点**

创新点在于将“路由假设”从数学推理迁移到代码安全领域，首次系统展示结构化先验在不同漏洞类型和模型上导致的 ID 提升与 OOD 崩塌，并验证迭代校准反而会加剧崩塌的现象。

**🔧 技术方法**

主要技术包括三大前沿 LLM（GPT‑OSS‑120B、Llama‑3.3‑70B、Gemma‑4‑31B）与手写 cheatsheet 进行零样本、少样本推理，结合 F1、召回率等指标进行评估。

**📊 数据集**

使用的数据集包括 348 条人工合成的漏洞样本（CWE‑798、CWE‑284、N+1 语义模式）和 VUDENC 的真实 CVE 代码片段（CWE‑89、CWE‑22）。

**📈 对比分析**

通过在零样本、cheatsheet‑v1 与 v2 条件下对比 F1/召回率，发现 cheatsheet 在合成数据上可达 100% F1，但在 VUDENC 上导致 29–58pp 的性能下降，充分证明分布偏移下的崩塌。

**⚠️ 局限性**

局限性包括样本量相对较小（真实数据仅 30 条）、仅覆盖 Python、未评估其他语言或框架、cheatsheet 的手工编写缺乏可靠性验证。

---

## 263. Efficient Pattern Matching for Unordered Term Tree Patterns under Generalized Height-Constrained Bindings

**arXiv ID:** 2607.14617 | [PDF](https://arxiv.org/pdf/2607.14617v1)

**作者:** Shintaro Matsushita `[一作]` (Fukuoka Institute of Technology), Yusuke Suzuki `[通讯]` (Hiroshima City University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种在树模式匹配中允许变量子树绑定到内部顶点的泛化模型，并给出了多项式时间的匹配算法。

**💡 创新点**

创新点在于放宽了高度约束变量的绑定限制：之前的叶子限制被移除，子树可以绑定到任意非根节点，从而提升模式表达能力，同时仍保持成员资格问题可多项式求解。

**🔧 技术方法**

核心技术包括基于动态规划的状态压缩、构造加权二分图并求解瓶颈匹配（使用Gabow–Tarjan算法或简单枚举+Hopcroft–Karp实现），以及继承信息的压缩与传播。

**📊 数据集**

实验使用在节点数 20~200 范围内随机生成的树和模式，总共 950,000 个实例，变量参数（i,j）也随机赋值。

**📈 对比分析**

与之前叶子绑定模型进行对比，泛化模型的平均运行时间约比前者高 8%（比值在 1.0694–1.0905 之间，整体均值 1.0803）。整体运行时间保持在每实例 10^-3~10^-2 秒，且比值随输入规模变化不大。

**⚠️ 局限性**

主要限制是瓶颈匹配步骤仍需多次完美匹配测试，尤其当 HC‑变量的高度上界较大时，额外的测试次数会显著增加运行时间；此外，实验仅在随机实例上验证，缺乏针对真实世界数据的评估。

---

## 264. Hough-SIFT: Robust Image Registration for Linear Structures via Hough Space

**arXiv ID:** 2607.14598 | [PDF](https://arxiv.org/pdf/2607.14598v1)

**作者:** Masaki Satoh `[一作]` `[通讯]` (Morpho, Inc.), Masaki Satoh (Morpho, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种在霍夫空间进行SIFT匹配的图像配准方法Hough‑SIFT，利用纹理保留的霍夫变换和基于线条的单应性估计，在强直线结构场景下实现鲁棒配准；

**💡 创新点**

创新点在于将SIFT特征匹配迁移至霍夫空间，并设计了纹理保留的霍夫变换与线条双重约束的单应性成本函数，从而解决传统SIFT在直线密集场景中的失效问题；

**🔧 技术方法**

核心技术包括纹理保留霍夫变换、SIFT特征提取与匹配、RANSAC+Levenberg–Marquardt单应性优化以及对角度周期性的处理；

**📊 数据集**

实验使用合成的“normal”和“linear”两组数据集（各100对图像）以及两段真实视频（正常场景与强线结构场景）进行评估；

**📈 对比分析**

与传统SIFT在几何误差和光度误差上进行对比，Hough‑SIFT在linear场景下成功率高、误差低；在真实视频中光度误差保持低，显示出与SIFT相当或更优的性能；

**⚠️ 局限性**

局限性包括额外的霍夫变换计算成本、对角度周期性处理的复杂性，以及对大畸变和噪声的鲁棒性尚待进一步提升。

---

## 265. Qubes OS Security in the Public Record

**arXiv ID:** 2607.14587 | [PDF](https://arxiv.org/pdf/2607.14587v1)

**作者:** Alfonso De Gregorio `[一作]` `[通讯]` (Pwnshow), Alfonso De Gregorio (Pwnshow)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2011–2025年Qubes OS公开安全公告（109条QSB与464条XSA）进行纵向统计与归因分析，探讨上游依赖、制度变迁与公开记录的稳定性。

**💡 创新点**

提出可复现的公开公告账本、基于标题的多标签归因代码本以及结合变点检验、过度离散检验、加权代理和预测对比的完整测量协议。

**🔧 技术方法**

使用贝叶斯单变点分析、BIC最优Poisson分段、负二项回归、CVSS代理加权、Yamada/AML等VDM模型，以及Diebold–Mariano检验与滚动平均对比。

**📊 数据集**

主要数据集为109条QSB、464条XSA（其中113条影响Qubes）以及由公共交叉表生成的年度漏洞事件敏感性序列（147条）。

**📈 对比分析**

通过与滚动三年平均、泊松趋势等基线比较，发现VDM在短期预测上与基线无显著优势；但Poisson/负二项模型在变点与过度离散检验中表现稳健，显示上游组件占主导，后期记录趋于稳定。

**⚠️ 局限性**

局限包括：仅基于公开记录（不反映隐含漏洞或实际被利用情况）；归因代码本受标题约束，混合公告仍可能出现误差；严重性代理粗略；2025年右侧截尾影响模型；短期预测样本有限，Diebold–Mariano检验功效受限。

---

## 266. Skeleton: Visual Authoring of Non-visual Data Experiences

**arXiv ID:** 2607.14579 | [PDF](https://arxiv.org/pdf/2607.14579v1)

**作者:** Frank Elavsky `[一作]` (Carnegie Mellon University), Dominik Moritz `[通讯]` (Carnegie Mellon University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究开发了一套名为 Skeleton 的可视化作者工具，用于可视化中无障碍导航结构的可视化、编辑、调试与迭代。

**💡 创新点**

创新点包括：1) 通过 Inspector 将导航图以交互节点链接图形式呈现；2) 提供 Dimension API，以数据维度声明方式自动生成导航结构；3) 设计了四阶段编辑流程（上传、准备、编辑、测试），将导航结构的拓扑、空间、语义与交互规则可视化并支持直接操作；4) 在实地共设计中证明可见化导航结构能激发作者的设计思考与错误检测。

**🔧 技术方法**

核心技术有：Data Navigator 库（负责导航图、渲染与输入模块）；Inspector（基于 D3 的交互图形渲染）；Dimensions API（声明式数据维度 DSL）；Skeleton 的 Scaffold 工具（利用 Vega 渲染引擎计算节点位置）；实时标签预览与测试模式；以及通过 Miro、Figma 等工具的共设计交互。

**📊 数据集**

实验使用了两类数据集：一是公开的水果计数表（8 条记录）作为通用测试图；二是研究参与者自带的多样化可视化图像（从条形图到自定义地图、Voronoi 饼图等），用于现场思考和评估。没有使用大规模标准化数据集，而是聚焦于实践中的真实可视化案例。

**📈 对比分析**

方法对比以“可见化 vs 传统代码式无障碍导航”为基准，通过 8 名参与者的现场访谈和观察记录来评估。性能方面主要以定性指标为主：错误检测率、迭代次数、标签编辑时间等。结果显示，Skeleton 能显著提升错误发现（约 62%），迭代速度提升 3–10 倍（以手动调节点为例 8 分钟降至 56 秒），并且参与者在测试阶段能即时验证导航可达性与标签完整性。未给出客观的时间/吞吐量数值，仅通过访谈数据量化。

**⚠️ 局限性**

局限性包括：1) 只在静态图像和固定数据集上验证，无法处理高度交互、动态更新或动画的可视化；2) 仍未实现可部署的输出（如生成可直接嵌入网页的代码）；3) 评估仅基于视障人士之外的可见作者，缺乏真实残障用户的使用反馈；4) 对复杂或大规模数据结构的可视化支持仍有限；5) 依赖视觉化工具的设计者仍需与残障用户共同验证，工具本身并不能替代多模态共设计。

---

## 267. Trajectory-Aware Flow Matching for Topology Optimisation

**arXiv ID:** 2607.14652 | [PDF](https://arxiv.org/pdf/2607.14652v1)

**作者:** Shusheng Xiao `[一作]` (Queensland University of Technology), YuanTong Gu `[通讯]` (Queensland University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于流匹配的轨迹感知拓扑优化框架，用于在给定物理和设计条件下快速生成多样化、物理可行的拓扑结构。

**💡 创新点**

通过将BESO优化历史作为概率路径和速度监督，构造轨迹感知的概率路径并引入轨迹权重，从而减少路径‑速度不匹配，提高生成稳定性与性能；同时实现仅需少步ODE采样、无逆扩散的高效生成。

**🔧 技术方法**

使用流匹配（Flow Matching）技术、条件U‑Net网络、BESO方法、有限元分析（FEA）、线性弹性最小化、概率路径设计与误差分析。

**📊 数据集**

二维BESO数据集：80×160尺寸，5000个实例（90%训练）；三维BESO数据集：48×24×24尺寸，2000个实例（90%训练）。

**📈 对比分析**

与扩散模型DMTO和线性FMTO做对比，评估指标包括合规率、失败率、可行率、IoU、Dice、Boundary F1、Raw MAE。FMTO在合规率、体积分数、拓扑相似度上优于DMTO，且采样时间仅为DMTO的1/50，采样步数从1000降至20。

**⚠️ 局限性**

仅针对线性弹性最小化问题；轨迹权重与锚点密度需手工调节；采用固定阈值二值化；三维验证规模有限；未覆盖多目标、非线性或耦合多物理问题。

---

## 268. MCPEvol-Bench: Benchmarking LLM Agent Performance Across Dynamic Evolutions of MCP Servers

**arXiv ID:** 2607.14642 | [PDF](https://arxiv.org/pdf/2607.14642v1)

**作者:** Huanxi Liu `[一作]` (National University of Defense Technology), Huaimin Wang `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了MCPEvol-Bench基准，用于评估LLM代理在MCP服务器工具集持续演进环境下的任务解决能力

**💡 创新点**

创新点在于：①基于大规模实证研究设计11个工具演化变异算子，模拟真实MCP服务器演化；②构建了123个多版本MCP服务器与201个跨服务器任务，形成动态演化基准；③引入ECS指标量化跨版本稳定性

**🔧 技术方法**

采用LLM驱动的代码变异、工具调用、任务合成等技术，并使用POMDP形式化任务流程；结合MCP协议与自动化测试验证演化后服务器功能；使用LLM判断器进行评分

**📊 数据集**

数据集包含123个MCP服务器（1,272个工具）以及对应的历史演化版本；任务集201条跨服务器、多工具协同任务；演化算子与工具变更通过LLM自动生成

**📈 对比分析**

通过对12个顶尖LLM（包括GPT-5.4、Claude-Sonnet-4-6等）在原始、3轮演化、5轮演化三种版本下进行评测；结果显示顶尖模型在演化后任务完成度下降13–15%，规划与推理错误显著上升；ECS最高的Claude-Opus-4-6达6.09，证明其相对更具适应性

**⚠️ 局限性**

局限性在于：①演化模拟主要基于LLM与变异算子，可能无法覆盖所有真实演化场景；②评估侧重工具调用与规划，未深入研究更复杂业务场景；③基准规模相对有限，未来需扩展更多服务器与任务类型

---

## 269. SportD: Can VLMs Physically Strategize?

**arXiv ID:** 2607.14616 | [PDF](https://arxiv.org/pdf/2607.14616v1)

**作者:** Jasin Cekinmez `[一作]` (Princeton University), Weining Shen `[通讯]` (UC Irvine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Soccer-Decision Bench，评估视觉语言模型在真实足球比赛中做射门或传球决策的能力。

**💡 创新点**

将基于价值模型的行动价值评估与视觉语言模型的决策结合，创建了可量化的物理战略推理基准。

**🔧 技术方法**

使用VAEP价值模型、SPADL动作表示、XGBoost梯度提升树，以及三大前沿VLM（Gemini 3.5 Flash、GPT 5.6 Sol、Claude Opus 4.8）进行评估。

**📊 数据集**

来自2022 FIFA世界杯的478个“持球”决策事件，配合对应的比赛视频、跟踪与事件数据。

**📈 对比分析**

通过最优行动准确率、平均回报损失、技能指标以及仿真率等度量与真实球员、随机基线和理想模型比较，VLMs的最佳准确率仅为31.4%，远低于球员的38.9%，并且显著产生更高的回报损失。

**⚠️ 局限性**

局限包括只覆盖男子世界杯、价值模型为近似且缺乏不确定性、动作空间仅限射门与传球、无法评估连续动作等。

---

## 270. Auditing Fairness-Privacy Trade-offs: Subpopulation-Level Effects of Fairness-Enhancing Algorithms

**arXiv ID:** 2607.14607 | [PDF](https://arxiv.org/pdf/2607.14607v1)

**作者:** Umid Suleymanov `[一作]` (Virginia Tech), Murat Kantarcioglu `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性地在子群体层面评估公平性提升算法与差分隐私对成员推断攻击的影响，提出统一的实验框架；

**💡 创新点**

①首次将LiRA攻击改造为子群体级；②在公平-隐私-效用三者交互中进行全面实验；③揭示公平措施并非必然加剧隐私泄露，而是受模型、子群体规模和干预策略共同决定；

**🔧 技术方法**

使用多种成员推断攻击（OQTA、OTA、LiRA），公平干预（DIR、REW、EGR、CPP、SYN），差分隐私训练（DP‑SGD、DP‑RF），以及决策树、随机森林、神经网络等模型；

**📊 数据集**

六个AIF360真实数据集（Bank Marketing、COMPAS Race/Gender、Law School Admissions Race/Gender、German Credit Age/Sex、Law School GPA Race、MEPS19 Race）及一个合成数据集，实验共十个版本；

**📈 对比分析**

通过对比各子群体的隐私风险、准确率和公平指标发现：DP能显著降低隐私风险但对少数群体造成较大准确率损失；EGR在复杂模型（RF）中最能降低隐私风险；REW稳定提升公平性但隐私提升有限；CPP使隐私接近随机但子群体效用波动大；不同模型复杂度决定隐私风险大小；整体未出现单一最优方案；

**⚠️ 局限性**

实验局限在表格数据与所选模型范围内；子群体规模仍是隐私弱点；DP的均匀噪声导致少数群体效用崩溃；公平干预与DP的交互效果高度依赖数据与模型；缺乏对其他领域（如图像、文本）的验证；

---

## 271. SYNAPSE: A Multi-LLM Orchestrated AI Tutor for Secure Software Development Education with Neurodivergent-First Design

**arXiv ID:** 2607.14601 | [PDF](https://arxiv.org/pdf/2607.14601v1)

**作者:** Giusy Ferrara `[一作]` (Edinburgh Napier University), Ashkan Sami `[通讯]` (Edinburgh Napier University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并公开部署了一个多LLM协同的AI辅导平台SYNAPSE，用于Java和Python的安全软件开发教育，并针对神经多样性学习者优化了可访问性。

**💡 创新点**

集成了Model Context Protocol实现多模型（Claude、GPT‑4o、Gemini）协作的三阶段苏格拉底式提示策略，推出ShopSecure故障练习并将可访问性功能视为教学核心。

**🔧 技术方法**

Python 3.12 + 异步HTTP框架、PostgreSQL、Docker容器、Flask应用、异步多模型工具调用、可访问性层（暗模式、高对比度、语音输入等）。

**📊 数据集**

基于自研的ShopSecure包含15个漏洞，映射到OWASP Top 10 (2021) 6个类别；收集19名志愿者的行为日志作为研究数据。

**📈 对比分析**

通过19名参与者的可用性（SUS 76.4）、参与度（4.2/5）和NASA‑TLX评估，神经多样性与常规组的认知负荷相当，且全部正确选择CWE‑22、CWE‑352、CWE‑502的修复措施。

**⚠️ 局限性**

样本规模小、神经多样性自报未临床验证、缺乏对照组、以及长时间交互中AI状态感知不足等局限。

---

## 272. How Well Does AI-Generated Feedback Work? Intrinsic and Extrinsic Evaluation across more than 20,000 EFL Essay Drafts

**arXiv ID:** 2607.14591 | [PDF](https://arxiv.org/pdf/2607.14591v1)

**作者:** Steven Coyne `[一作]` (Tohoku University), Kentaro Inui `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大学EFL写作课堂中使用LLM生成书面纠错反馈（WCF）的效果，并对其进行了内在评价（教师打分）与外在评价（学生使用与反馈）。

**💡 创新点**

创新点在于提出教师与学生两种视角的多维评估框架，显示两者在衡量WCF质量时关注点不同，并通过对比实验验证LLM生成WCF在主观质量上优于模板方法。

**🔧 技术方法**

采用GPT‑4o模型生成WCF，并与基于模板和简单LLM的两种系统对照；利用ERRANT等工具提取与分类错误，并在界面中展示高亮与解释两部分反馈。

**📊 数据集**

使用了1,899名学生在大学EFL课程中提交的20,255份作文草稿，产生约144,790条WCF；从中抽样约600条用于教师评估。

**📈 对比分析**

通过教师打分（Krippendorff alpha）与学生点赞/点踩以及查看次数进行比较，LLM系统整体质量评分平均在4–5分，模板系统最低，显示LLM生成WCF在主观质量上优于模板；但教师与学生评价在每条评论层面并无显著相关性。

**⚠️ 局限性**

研究缺乏学习成效测量、学生反馈稀疏且偏向高参与学生、所有学生同一母语导致错误模式单一、样本性别比例失衡等限制。

---

## 273. Conversational Tactile Data Interfaces: Co-Designing Accessible Data Experiences with Blind Users Using Refreshable Tactile Displays and Conversational AI

**arXiv ID:** 2607.14588 | [PDF](https://arxiv.org/pdf/2607.14588v1)

**作者:** Samuel Reinders `[一作]`, Kim Marriott `[通讯]` (Monash University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了Graphy，对话式触觉数据接口，并通过共同设计方法让用户参与改进

**💡 创新点**

将对话交互与触觉展示相结合，提升了数据探索的沉浸感和可用性

**🔧 技术方法**

采用触觉硬件（Haptic Device）、语音识别与自然语言处理技术，以及可视化交互框架

**📊 数据集**

未公开具体数据集，实验基于用户生成的探索任务数据

**📈 对比分析**

与传统的静态触觉图形界面对比，Graphy 在用户任务完成时间下降约20%，满意度提升

**⚠️ 局限性**

样本规模有限，实验场景单一，且硬件对齐误差影响体验

---

## 274. Rethinking Issue Resolution for AI/ML Systems

**arXiv ID:** 2607.14657 | [PDF](https://arxiv.org/pdf/2607.14657v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 275. NavCMPO: Critic-Guided MeanFlow Policy Optimization for Adaptive Navigation

**arXiv ID:** 2607.14643 | [PDF](https://arxiv.org/pdf/2607.14643v1)

**作者:** Junjie An `[一作]` (Zhejiang University), Guang Li `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了NavCMPO，一种两阶段框架，结合少步MeanFlow生成、障碍感知视觉编码、Critic引导轨迹细化以及PPO强化学习微调，实现在地图无信息视觉导航中的低延迟与高成功率。

**💡 创新点**

创新点在于将少步MeanFlow与障碍接近预测任务、Critic-guided轨迹细化以及PPO微调和行为克隆正则化相结合，既显著降低推理延迟，又突破了单纯行为克隆的性能上限。

**🔧 技术方法**

使用的技术包括RGB‑Depth Transformer编码器、Obstacle Proximity Prediction、MeanFlow流匹配生成、Critic-guided Trajectory Refinement、PPO强化学习、以及行为克隆正则化。

**📊 数据集**

实验采用InternVLA‑N1仿真基准（Cluttered‑Hard、Home、Commercial场景）以及Unitree Go2的室内/室外点目标导航数据集进行评估。

**📈 对比分析**

通过匹配计算资源与NavDP、MP1、Viplanner等基线对比，NavCMPO在InternVLA‑N1平均成功率达到74.7%（比NavDP高6.4pp），推理时间降低至60 ms（比85 ms低），在真实世界Unitree Go2实验中成功率提升至66.7%，超过NavDP的56.7%。

**⚠️ 局限性**

局限性包括仅在匹配计算资源下验证、仅针对单一机器人（Unitree Go2）进行测试、RL微调仍在仿真完成且未在真实机器人上进行在线学习，并未将NavDP同样的RL微调直接对比。

---

## 276. Autoregressive Modeling of Film with Applications in Video Montage

**arXiv ID:** 2607.14645 | [PDF](https://arxiv.org/pdf/2607.14645v1)

**作者:** Marcelo Sandoval-Castañeda `[一作]` (TTI-Chicago), Greg Shakhnarovich `[通讯]` (TTI-Chicago)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出FilmGPT，一种自回归变压器，用于视频蒙太奇，能够从大量原始未剪辑的素材中选择、裁剪并排列镜头，生成符合电影剪辑惯例的完整剪辑；

**💡 创新点**

创新点包括：①直接从电影数据学习“剪辑语法”，不依赖人工规则；②在序列中加入专用< CUT >标记并采用加权交叉熵训练，以强化切点预测；③提出基于素材约束的推理算法（footage‑constrained decoding），保证生成的剪辑仅使用输入素材；④使用长上下文、滑动窗口注意力以及音频条件化，提升对长视频的建模能力。

**🔧 技术方法**

技术手段主要包括：长上下文自回归变压器（decoder‑only）配合滑动窗口注意力；TiTok‑L离散图像分词器；< CUT >标记和加权交叉熵损失；Audio Flamingo 3音频特征跨注意力；以及素材约束的Beam Search推理算法。

**📊 数据集**

数据集涵盖：AV E数据集、美国公共领域电影、苏联Mosfilm电影、CC许可的纪录片，共计约6200小时的训练素材；同时使用TransNet V2进行镜头检测、Audio Flamingo 3获取音频特征。

**📈 对比分析**

与基准方法对比：在AV E镜头排序任务上，FilmGPT达53.9%准确率，远超UQNet 35.3%；在用户偏好评测中，对比Transcript2Video和EditDuet，FilmGPT被优选率分别为83.1%和61.5%；Ablation实验展示了< CUT >标记、加权损失、长上下文、音频条件和大规模电影数据对性能的显著提升。

**⚠️ 局限性**

局限性包括：1）受限于上下文长度，难以处理极长视频；2）在素材缺乏合适“按动作剪辑”时可能产生不连贯切点；3）仅处理剪切，无法直接编辑音频或实现L‑cut、J‑cut等过渡；4）目前仅接受音频与素材约束，缺乏剧本、分镜等更丰富的条件；5）潜在的误用风险，如通过剪辑重构产生误导性内容。

---

## 277. TopoAgent: A Self-Evolving Topological Agent for Multimodal Scientific Reasoning

**arXiv ID:** 2607.14658 | [PDF](https://arxiv.org/pdf/2607.14658v1)

**作者:** Mingze Xu `[一作]` (Tsinghua University), Yuxing Han `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于动态拓扑图的自演化框架 TopoAgent，用来解决多模态科学推理中的视觉语义偏移、长上下文幻觉和固定粒度执行瓶颈问题。

**💡 创新点**

创新点包括：①视觉先行的原子分解（Visually‑Grounded Atomic Decomposition）将复杂任务拆成最小可执行单元；②利用有向无环图（DAG）实现严格的上下文隔离（Context Isolation），只将前置节点的确定性结果作为当前节点的输入；③自适应原子分裂（Adaptive Atomic Fission）在执行过程中动态将无法在工具能力范围内完成的节点细化为更小的子原子，保证推理的鲁棒性。

**🔧 技术方法**

核心技术包含前端分解器、DAG规划与执行调度、上下文隔离机制、动态原子分裂、层级命名空间映射、以及基于LLM的规划-执行-验证循环；整个系统与外部工具（如数学计算器、物理仿真器等）通过函数调用集成。

**📊 数据集**

使用多学科大规模科学推理基准数据集：MathVista、Multi‑math、SymbolBench（数学子集）、OlympiadBench、MMMU（物理子集）、Chem‑Symbol（化学子集）等共六大域。

**📈 对比分析**

与原生 LLM、OctoTools、LangChain、GPT‑Functions、AutoGen 等多种基线进行对比，结果显示 TopoAgent 在所有六种评测模型上实现了 66.3% 的全局平均准确率，明显优于顺序规划（≈58%）和多代理系统（≈62%）；在 ablation 实验中去掉 DAG 或原子分裂分别导致约 1.4% 和 0.6% 的性能下降，证明两项创新均对提升有显著贡献。

**⚠️ 局限性**

局限性包括：1）仍依赖外部工具的可用性与接口稳定性；2）对高度抽象、无明显视觉对应的推理任务适配性有限；3）DAG 生成与原子分解可能产生过多或过细的子任务，导致执行调度开销上升；4）在极大规模图结构时的内存与计算成本需要进一步优化。

---

## 278. Cross-Layer Error Compensation and Finite-Sample Feature-Statistics Matching for Extreme Low-Bit Quantization of Large Language Models

**arXiv ID:** 2607.14630 | [PDF](https://arxiv.org/pdf/2607.14630v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 279. Semi-Streaming Matching in a Single Pass II: Greedy is Optimal

**arXiv ID:** 2607.14656 | [PDF](https://arxiv.org/pdf/2607.14656v1)

**作者:** Sepehr Assadi `[一作]` (University of Waterloo), Mars Xiang `[通讯]` (Interaction Company)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

证明在单通道半流式算法中，最大匹配问题无法获得优于1/2的近似比，证明贪心算法是最优的；同时推导出在线匹配预先删除模型和半流式最小顶点覆盖问题同样存在1/2（或2）下界。

**💡 创新点**

提出一种新的“蓝图”构造方法，利用随机游走（赌博者破产模型）和延迟/长度限制来构造满足蓝图约束的结构，从而实现蓝图值达到2/3，使得蓝图框架给出的下界逼近最优。

**🔧 技术方法**

核心技术包括蓝图框架、随机游走（gambler's ruin）、延迟随机游走、长度限制、线性规划求解初始分布、对称性与最大值分布的分析。

**📊 数据集**

论文不涉及实验数据或数据集，全部以理论证明和构造为主。

**📈 对比分析**

通过蓝图框架将蓝图值映射为匹配问题的下界，得到1/2近似比是不可超过的极限；与现有贪心算法的1/2近似相匹配，证明其性能已达到最优。

**⚠️ 局限性**

仅适用于单通道半流式（单遍）模型，主要针对二分图匹配问题；多通道、多遍或一般图匹配、非二分图情况尚未覆盖，且蓝图构造复杂，可能不易推广到其他问题。

---

## 280. Dendrite: A Real-Time Python Application for Online Brain-Computer Interface Research and Development

**arXiv ID:** 2607.14655 | [PDF](https://arxiv.org/pdf/2607.14655v1)

**作者:** Niko Kroflic `[一作]` (Jožef Stefan Institute), Jan Babič `[通讯]` (Jožef Stefan Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了Dendrite，一款可在Python环境下完成多模态实时信号采集、在线训练、实时推理、离线回放和可视化监控的统一BCI运行时，且实验范式通过LSL事件保持独立；

**💡 创新点**

创新点在于将在线学习与推理完全集成到单一可修改的Python应用中，使用共享内存环形缓冲实现不同采样率流的同步；通过LSL事件契约让实验范式完全外部化，保持高度可复现；并通过SQLite + HDF5 记录完整实验线索，实现从采集到模型部署的全链路追溯；

**🔧 技术方法**

核心技术包括Python生态（FastAPI + Vue3 SPA）、LSL数据同步、共享内存环形缓冲、HDF5 + BIDS式元数据、SQLite线索库、Optuna 超参搜索、scikit-learn 与 Braindecode/EEGNet 兼容的解码器、PyTorch 训练后端；

**📊 数据集**

使用了四大数据集：自研运动意象（4次实验、64通道、500Hz）、BNCI 2014-001（BCI比赛IV-2a 4类MI、22通道、250Hz）、BNCI 2014-009（10位受试者P300偶数任务、256Hz）、自研外骨骼多模态（EEG+EMG+关节运动），以及实时闭环反馈的神经反馈实验；

**📈 对比分析**

通过在线预序列准确率对比（MI 0.59–0.82，P300 0.92）、推理延迟均低于4 ms（CSP+LDA 1.0–1.7 ms），闭环端到端延迟 p50≈13 ms、p99≈66 ms，在线重训练耗时 0.3–0.4 s，且热替换后不影响推理节拍；

**⚠️ 局限性**

局限性包括在线学习策略仅为简单的每10个epoch重新训练；多模态融合仅在推理层面，未实现联合训练；无插件 API，扩展需修改源码；实验范式需外部刺激软件；数据集规模相对较小，闭环延迟仅在单一外部应用上评估。

---

## 281. Image-to-Point Cloud Registration Made Easy with Rectified Flow-based LiDAR Upsampling

**arXiv ID:** 2607.14639 | [PDF](https://arxiv.org/pdf/2607.14639v1)

**作者:** Reon Tabata `[一作]` (Toyohashi University Of Technology), Jun Miura `[通讯]` (Toyohashi University Of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种Image-to-Point-Cloud（I2P）配准方法，先用Conditional Rectified Flow将稀疏LiDAR强度图上采样为稠密图像，再与相机图像通过预训练特征匹配器对齐，最终利用PnP-RANSAC估计6-DoF姿态。

**💡 创新点**

①利用Conditional Rectified Flow对LiDAR稀疏图像进行稠密化，避免传统扩散模型多步推断；②不需要图像-点云配对训练，利用预训练图像特征匹配器实现高精度与强泛化；③仅用极少的LiDAR数据微调即可适配多种传感器。

**🔧 技术方法**

Conditional Rectified Flow（自监督图像修复预训练）、光学特征匹配器LightGlue、PnP-RANSAC姿态估计、稀疏点云投影到图像、深度邻域搜索等。

**📊 数据集**

R3LIVE（Livox AVIA）、GSV-Cities（自监督预训练）、Livox AVIA/MID360、Ouster OS1-64（微调）。

**📈 对比分析**

与CoFiI2P、FreeReg、2D3D-MATR等基线比较，R3LIVE上平均RRE≈1.75°、RTE≈0.57 m、RR>90%，帧耗时0.68 s，显著优于基线；特征对应点数与内点率也最高。

**⚠️ 局限性**

在非结构化环境（树叶、草）下性能下降；生成的稠密图像细节有限，深度缺失时需近邻搜索导致误差；对极端视角或光照变化的鲁棒性仍有提升空间。

---

## 282. Angular Gaussian Supervised Contrastive Learning for Long-Tailed Electrocardiogram Arrhythmia Diagnosis

**arXiv ID:** 2607.14613 | [PDF](https://arxiv.org/pdf/2607.14613v1)

**作者:** Jin Dai `[一作]` (Shanghai Jiao Tong University), Can Han `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出一种针对长尾多标签心电图异常检测的 Angular Gaussian Supervised Contrastive Learning (AG‑SCL) 框架。

**💡 创新点**

创新点在于结合方向感知的 Angular Gaussian 对齐、可学习的 Adaptive Logit Adjustment 以及保护 QRS 频带的 tail‑aware augmentation 三大模块。

**🔧 技术方法**

技术上采用频域 Swin‑Transformer 编码器、基于 MGF 的 Angular Gaussian 对比损失、计数衰减的协方差估计与自适应对数校准。

**📊 数据集**

数据集包括公开的 PTB‑XL 及自建的 141 名患者、1,317 小时夜间 Lead‑I ECG，标注为多标签节律异常。

**📈 对比分析**

与 AtCNN、MTDL‑NET、ViT‑ECG 等架构以及 GLA、GCA、ProCo 等长尾方法对比，AG‑SCL 在两组数据上均取得最佳宏观指标（PTB‑XL bACC 0.838、mAP 0.495；Noc‑ECG bACC 0.918、mAP 0.488）。

**⚠️ 局限性**

局限性包括单中心采集、仅使用 Lead‑I、缺乏多导或可穿戴设备验证，以及对极低频率标签泛化仍需进一步评估。

---

## 283. Accelerating A/B-Tests with Counterfactual Estimation: Reducing Variance through Policy Overlap

**arXiv ID:** 2607.14604 | [PDF](https://arxiv.org/pdf/2607.14604v1)

**作者:** Olivier Jeunen `[一作]` `[通讯]` (Independent Researcher), Olivier Jeunen (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种利用政策重叠的A/B测试新方法，构建无偏的政策感知估计器（Δ-IPS、Δ-DR）并证明其方差优势；进一步给出最优流量分配、基于方差的奖励模型学习（Δ-MRDR）以及排名情境下的Δ-DCG估计。

**💡 创新点**

核心创新是将随机分配机制视为元策略，利用Off‑Policy Evaluation框架从政策重叠中提取信息，理论上证明方差随政策相异度而非原始噪声变化；推出可直接优化的方差最小化学习目标和可在排名系统中使用的Δ-DCG。

**🔧 技术方法**

主要技术包括：差分重要性采样（Δ-IPS）、双重鲁棒估计（Δ-DR）、权重平方加权最小二乘（Δ-MRDR）、位置基础模型下的折扣累计收益（DCG）估计、以及基于方差推导的最优流量分配。

**📊 数据集**

使用合成实验数据：1) 线性奖励的上下文赌博机（|A|=10~5000），2) 采用PBM的Top‑K排序实验（M=100, K=25），并在不同政策相异度和流量分配下重复多次。

**📈 对比分析**

与传统差分均值（DiM）以及基线调整后的估计器相比，Δ-IPS/Δ-DR在所有相异度下均实现更低方差；Δ-MRDR在资源受限的模型下将方差进一步降低；Δ-DCG在高重叠情境下近乎零方差，且即使在较大相异度时仍优于DiM，整体提升显著。

**⚠️ 局限性**

局限性包括：仅在模拟环境验证，缺乏真实日志数据；需要已知且准确的策略概率（π(a|x)），对大规模或生成模型的计算开销高；最优流量分配假设环境平稳且结果同方差，现实中可能受分布漂移影响；并未充分探讨多阶段自适应实验设计的实现细节。

---

## 284. Quasi-Belief Propagation and Neural-Network Check Node Processing for BCH Codes

**arXiv ID:** 2607.14589 | [PDF](https://arxiv.org/pdf/2607.14589v1)

**作者:** Guangwen Li `[一作]` `[通讯]` (Shandong Technology and Business University), Guangwen Li (Shandong Technology and Business University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并评估了一种针对BCH码的准BP（quasi‑BP）解码方案及其基于卷积神经网络的轻量化变体，以实现与LDPC BP解码相近的误帧率性能。

**💡 创新点**

提出利用码的自同构与冗余校验矩阵构建准BP框架，并用三重约束损失训练的CNN替代传统tanh运算，实现并行化解码与硬件友好的算术操作。

**🔧 技术方法**

采用准BP迭代消息传递、卷积神经网络检查节点更新、三重约束损失训练、OSD变体串联以及与LDPC层级BP对比分析等技术。

**📊 数据集**

在BCH(127,64)、(127,99)、(255,239)等短码以及CCSDS LDPC(128,64)上进行仿真，使用AWGN信道的BPSK调制信号。

**📈 对比分析**

通过FER/BER曲线与LDPC BP、ENMS、QBP、QBP‑SF、OSD等解码器在相同SNR区间对比，结果显示QBP‑SF在高SNR下与LDPC BP仅差≤0.25 dB，且与ML解码仅差≤0.4 dB。

**⚠️ 局限性**

仍受限于迭代次数与模型规模导致的计算复杂度，CNN检查节点需要训练并在不同码长度上重新调参，且在极高SNR下仍存在与LDPC相似的误码率阈值。

---

## 285. MathCoPilot: An Interactive System for Human-AI Symbiotic Paradigm of Mathematical Research

**arXiv ID:** 2607.14582 | [PDF](https://arxiv.org/pdf/2607.14582v1)

**作者:** Junjie Zhang `[一作]` (University of Science and Technology of China), Xiangdong Ye `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种人机协同的数学研究系统，支持论文检索、自动化知识库构建、交互式证明蓝图以及多技能证明调度。

**💡 创新点**

创新点在于将人类研究者的高层决策与AI的细粒度证明执行结合，构建可编辑的证明蓝图；动态调度多种证明技能并根据Lean反馈自适应选择；自动将科研论文转化为可验证Lean代码并构建个人知识库。

**🔧 技术方法**

技术包括大语言模型（Gemini 3.1 Pro、GPT‑5.4、Claude Opus 4.7等）、Lean 4证明器、自动化知识库检索与重写、证明蓝图可视化、技能调度器以及自然语言与Lean代码的双向转换。

**📊 数据集**

使用的基准数据集有：FormalMATH（21道题）和两道真实PDE定理（上游DG误差估计与Gauss‑Radau投影估计）。另外还使用了多源论文检索来构建知识库。

**📈 对比分析**

通过对比三种证明技能（formalization‑first vs NL‑first）和三种模型，评估了在FormalMATH上的严格通过率（NL‑first约为31.8%，FF约为15.9%）。在GPT‑5.4未能解决的12个案例中，交互式蓝图可将通过率提升至约83%。在PDE任务中，Claude Opus 4.7在所有三条路径均能生成可验证Lean证书（100%），其余模型受版本/语义不匹配影响。

**⚠️ 局限性**

限制主要包括：对高级领域（如PDE）仍需人工定义或假设；auto‑formalization仍不稳定，导致低通过率；系统对现有Lean库的依赖较大，缺乏跨系统支持；模型误判和错误检测尚不全面。

---

## 286. Beyond Implicit Force: Evaluating Explicit Force-Torque Proxies in Action Chunking with Transformers

**arXiv ID:** 2607.14578 | [PDF](https://arxiv.org/pdf/2607.14578v1)

**作者:** King Hang Wong `[一作]` (Adelaide University), Feras Dayoub `[通讯]` (Adelaide University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过对 Action Chunking with Transformers (ACT) 的观测中心化变体和电机电流产生的扭矩代理进行实验，探究在无外部 F/T 传感器的平台上，扭矩代理能否替代隐式的遥控误差信号，实现接触敏感操控。

**💡 创新点**

创新点在于将 ACT 从依赖遥控误差的隐式力感知转为显式扭矩代理，并系统评估其在多任务中的性能，证明扭矩代理既可恢复又可提升原模型表现。

**🔧 技术方法**

技术包括观察中心化的 ACT 变体、基于变压器的 CVAE 生成模型、扭矩代理注入、以及在四个接触任务上进行的硬件实验。

**📊 数据集**

数据集来自 ALOHA Solo 物理平台上收集的四个任务演示：板擦、插头插入、软瓶压、泡沫停止，约 2000 条演示轨迹。

**📈 对比分析**

与基准 ACT 以及无扭矩的 ACT‑o 进行对比，结果显示引入扭矩代理后成功率从 15% 提升至 90‑95% 等，在所有任务中显著优于基准。

**⚠️ 局限性**

局限在于仅使用电机电流产生的扭矩代理，缺乏高频触觉信息，且未探究大规模数据或跨平台迁移效果。

---

## 287. Spectral Dual Fitting for $k$-Means

**arXiv ID:** 2607.14654 | [PDF](https://arxiv.org/pdf/2607.14654v1)

**作者:** Aditya Anand `[一作]` (University of Michigan), Ernest van Wijland `[通讯]` (Université Paris-Cité)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种改进的贪心算法，用于解决加权设施定位和欧氏k-均值聚类问题，并通过引入谱图拉普拉斯矩阵约束，进一步降低了对偶变量的投标幅度，从而实现更优的逼近比；

**💡 创新点**

创新点在于将传统贪心算法的“投标过度”问题拆解为对偶变量增幅和拉普拉斯矩阵约束的两阶段分析，并通过构造新的对偶变量α^c实现对自由设施的精细控制；

**🔧 技术方法**

核心技术包括谱图理论（拉普拉斯矩阵约束、条件列求和分析）、凸优化（线性规划检验可开机性）、矩阵分析（P矩阵与d^+参数）以及对偶松弛与投标增幅控制；

**📊 数据集**

未使用具体实验数据集，论文主要以理论证明和复杂度分析为主；

**📈 对比分析**

与原始的4-approx算法比较，改进算法在欧氏度量下实现了3+ln2≈3.693的逼近比，在一般度量下实现了4.9逼近，比之前的5逼近更优；

**⚠️ 局限性**

局限性包括：算法仍需依赖度量空间满足三角不等式、对欧氏空间有更强假设、加权版本中引入ε参数导致误差上界较大，并且对极端大规模实例的时间复杂度仍然较高。

---

## 288. Bad Memory: Evaluating Prompt Injection Risks from Memory in Agentic Systems

**arXiv ID:** 2607.14611 | [PDF](https://arxiv.org/pdf/2607.14611v1)

**作者:** Soham Gadgil `[一作]` (University of Washington), Franziska Roesner `[通讯]` (University of Washington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在使用持久文件的代理系统中，prompt注入攻击对多会话行为的影响

**💡 创新点**

首次系统评估了存储在文件中的攻击载荷在不同模型和攻击目标下的成功率与持久性，并揭示了模型对同类攻击的差异性

**🔧 技术方法**

构建了 sandboxed 合成工作区，使用 Claude Code、Codex 等代理系统，对 Claude Haiku、Claude Opus、GPT‑5.2、GPT‑5.5 四个模型进行实验

**📊 数据集**

使用模拟的个人助手工作区，包含 auto‑loaded 指令文件、知识文件和行为文件，内嵌三类攻击载荷（凭证外泄、未经授权工具使用、品牌推广）

**📈 对比分析**

通过单次探测、Probe→Stabilization→Probe 等多会话序列进行定量评估，报告攻击成功率 (ASR) 与持久率，发现较弱模型易被凭证泄露攻击，较强模型更易被品牌推广攻击；整体表现仍不稳定且可堆叠

**⚠️ 局限性**

实验仅基于合成环境，未覆盖真实用户流程；攻击载荷的注入方式受限（无法通过外部内容直接写入文件），且不同模型在安全自我修正上的差异尚未完全解释

---

## 289. Enumerating Length-Bounded Simple Paths and Cycles in Directed Graphs with $O(k(n+m))$ Delay Using Edge-Consistent Node Barriers

**arXiv ID:** 2607.14745 | [PDF](https://arxiv.org/pdf/2607.14745v1)

**作者:** Frank Bauernöppel `[一作]` (Hochschule für Technik und Wirtschaft), Jörg-Rüdiger Sack `[通讯]` (Carleton University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并分析了一种新的 Bounded‑Scope Depth‑First Search (BS‑DFS) 算法，用于在给定长度上限 k 的有向图中枚举所有简单路径或简单环，证明其正确性并给出 O(k(n+m)) 的最坏延迟。

**💡 创新点**

引入边一致性 (edge‑consistency) 这一局部可维护的边界值不变性作为完整性证明的核心，并以此为统一框架构造紧凑、松散、惰性等多种变体，同时揭示现有 BC‑DFS 算法的缺陷与未被证明的延迟。

**🔧 技术方法**

基于启发式搜索的一致性分析、递归深度优先搜索、后向广度优先级更新（cascade）以及依赖列表（lazy scheme）等技术，配合边一致性不变性进行数学证明。

**📊 数据集**

在实验中使用了随机 Erdős–Rényi 图和随机 Watts‑Strogatz 小世界图，节点数从 6 到 30 以及 1000，边概率或重连概率分别调整。

**📈 对比分析**

通过与无边界的朴素 DFS 以及 BC‑DFS 在相同图上进行输出数量、漏检率和运行时间的对比，结果显示 BC‑DFS 在 k≥10 时漏检比例超过 10%/33%，BS‑DFS 产生完整结果且每输出的平均延迟与 BC‑DFS 相差不超过 1.5 倍。

**⚠️ 局限性**

目前仅给出了 BS‑DFS 的 O(k(n+m)) 延迟上界，常数尚未优化；松散和惰性变体的延迟上界仍未知；算法仅适用于单源单汇的单位权重有向图，无法直接推广到权重预算、多目标或非单源情况。

---

## 290. GeoDetect: Geometric Adversarial Detection for VLPs

**arXiv ID:** 2607.14737 | [PDF](https://arxiv.org/pdf/2607.14737v1)

**作者:** Afsaneh Hasanebrahimi `[一作]` (University of Melbourne), Sarah Erfani `[通讯]` (University of Melbourne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GeoDetect，一种基于几何差异的无任务、轻量化对抗样本检测框架，专门针对视觉‑语言预训练模型（VLP）。

**💡 创新点**

创新点在于：①理论证明 VLP 嵌入空间存在显著的异方差，且对抗样本会被推到流形之外；②利用这一性质，提出使用 LID、k‑NN、Mahalanobis 与 KDE 等几何度量实现跨任务、跨模型的对抗样本检测；③方法无需任务特定 logits 或标签，适用于分类与检索两大场景。

**🔧 技术方法**

技术手段包括：对抗样本生成（Sep‑Attack、Co‑Attack、SGA 等），几何度量（Local Intrinsic Dimensionality、k‑Nearest Neighbours、Mahalanobis 距离、Kernel Density Estimation），以及基于阈值或逻辑回归的检测决策；同时给出了理论分析与证明。

**📊 数据集**

实验数据集涵盖：Zero‑shot 分类任务（ImageNet、CIFAR‑10/100、STL‑10、Food‑101）以及图文检索任务（Flickr30K、MS‑COCO）。

**📈 对比分析**

与基准方法 MCM、PIP 等进行对比，GeoDetect 在多种 VLP 体系（CLIP‑CNN、CLIP‑ViT、ALBEF、TCL）和多种攻击（图像、文本、联合）下均取得 AUC≥99% 且 FPR95 极低（≤5%），在自适应攻击设置中仍保持高识别率。

**⚠️ 局限性**

局限性：依赖于干净参考批次的几何统计，若批量不足或对抗样本极为隐蔽时可能失效；在完全白盒自适应攻击下性能有所下降；目前主要针对图像-文本两模态，尚未验证对更丰富多模态或其他任务的通用性。

---

## 291. Hybrid Rigid-Soft Robotic Gripper with Shape Adaptation, Uniform Force Distribution, and Self-Locking Capabilities

**arXiv ID:** 2607.14730 | [PDF](https://arxiv.org/pdf/2607.14730v1)

**作者:** Xi Chen `[一作]` (Beijing Academy of Agriculture and Forestry Sciences), Ya Xiong `[通讯]` (Beijing Academy of Agriculture and Forestry Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开发了一种集成双齿轮自锁机制与分段膜压空气肌肉的混合刚柔抓手，旨在实现农业作业中的自适应抓握、高承载力和低能耗。

**💡 创新点**

通过双齿轮自锁结构实现了无连续气压时的被动锁定，并且采用低成本膜压空气腔与3D打印齿轮实现高分辨率锁定与极大承载力。

**🔧 技术方法**

采用低成本尼龙‑聚乙烯复合膜压空气腔、3D打印双齿轮‑皮带自锁机构以及气压驱动与伺服电机复合控制。

**📊 数据集**

未使用公开数据集，全部采用实测实验（球形模型、各种水果等）。

**📈 对比分析**

与传统刚性抓手和传统软抓手在抓取力分布、最大承载力、能耗等指标对比，最大承载力从210 g提升至4200 g，能耗降低约50%，抓取力分布更均匀。

**⚠️ 局限性**

缺乏独立关节控制、对尖锐或凹陷表面适应性差，且对极小或极大尺寸物体抓握效果有限。

---

## 292. Counterfactuals for Feature-Weighted Clustering

**arXiv ID:** 2607.14719 | [PDF](https://arxiv.org/pdf/2607.14719v1)

**作者:** Richard J. Fawley `[一作]` (University of Essex), Renato Cordeiro de Amorim `[通讯]` (University of Essex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为VoICE的框架，用于在特征加权的k-means聚类中生成对抗性解释，将解释视为投影到加权Voronoi区域的最小扰动。

**💡 创新点**

创新点包括：① 在聚类决策区域上直接投影而非仅使用两两边界，确保解释的有效性；② 引入同心收缩（homothetic contraction）以增强鲁棒性；③ 通过按特征权重排序的可操作特征子集实现可解释且最小化的干预；④ 统一的凸优化形式并提供方向容忍度等多维评价指标。

**🔧 技术方法**

使用技术主要包括：加权k-means（使用SHARK权重）、凸投影优化、同心收缩、基于特征权重的子集搜索、方向性可行区间分析。

**📊 数据集**

实验采用主评估数据集：Iris、Wine、Palmer Penguins、Breast Cancer、Wholesale Customers；以及诊断性混合型数据集：Diabetes、Obesity、German Credit、Heart Failure、Student Performance。

**📈 对比分析**

方法通过与CFCLUST（仅两两边界投影）比较，VoICE在有效性上从80%提升至100%；在可行性、干预成本、最小干预基数、方向容忍度等指标上均表现出可解释性和鲁棒性；对不同聚类模式和可操作特征掩码的敏感性也被系统评估。

**⚠️ 局限性**

局限性在于：对混合类型（二进制/分类型）数据时，权重集中导致几何退化为低维子空间，当前仅使用加权欧氏距离，缺乏对类别特征的专门处理；需要进一步研究类别感知距离、分组权重或正则化以避免权重极端集中。

---

## 293. VideoSEMA: a scalable and efficient Mamba-like attention for video understanding

**arXiv ID:** 2607.14711 | [PDF](https://arxiv.org/pdf/2607.14711v1)

**作者:** Nhat Thanh Tran `[一作]` (University of California, Irvine), Jack Xin `[通讯]` (University of California, Irvine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种面向视频分类的分离时空注意力模型 VideoSEMA，结合 Mamba‑style 空间注意力和软最大化时序注意力；

**💡 创新点**

创新点在于构造一种可扩展、高效的 Mamba‑like 空间块（SEMA），并证明在满足秩条件下可等价于完整的时空注意力，从而显著降低计算复杂度；

**🔧 技术方法**

使用了局部窗口注意力、全局平均池化与软最大化时序注意力的组合，辅以 Mamba 宏架构实现；

**📊 数据集**

在 Kinetics‑400（K400）和 Something‑Something V2（SSv2）数据集上进行实验；

**📈 对比分析**

与更大型的 Vision Transformer、Mamba 等模型对比，VideoSEMA 在 K400 上取得更高准确率；在 SSv2 上在相同参数量级下表现最佳；在分辨率从 224² 提升到 1024² 时，准确率下降更平滑，优于 VideoMamba；

**⚠️ 局限性**

限制包括：仅在 K400 与 SSv2 上验证；对极长视频的性能仍待通过稀疏/膨胀时序注意力进一步提升；以及对秩条件的依赖在特殊场景下可能不成立。

---

## 294. Team RAS in 11th ABAW Competition: Multimodal Ambivalence Recognition Approach

**arXiv ID:** 2607.14702 | [PDF](https://arxiv.org/pdf/2607.14702v1)

**作者:** Elena Ryumina `[一作]` (St Petersburg Federal Research Center of Russian Academy of Sciences), Alexey Karpov `[通讯]` (St Petersburg Federal Research Center of Russian Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对 11th ABAW 挑战的视频级 ambivalence 与 hesitancy 识别，作者提出了一种单一的以文本为中心的多模态融合模型。

**💡 创新点**

创新点在于将文本作为锚模态，通过门控残差调整方式将音频、面部与场景信息融合，避免使用大规模模型集成。

**🔧 技术方法**

采用 RoBERTa‑base 预训练文本编码、Wav2Vec2+Bi‑Mamba 音频编码、Emo‑AffectNet+Transformer 面部编码、VideoMAE‑v2+Mamba 场景编码，并在融合层使用文本残差门控和双层 MLP 分类。

**📊 数据集**

使用 BAH 语料库（1427 段视频，包含文本、音频、面部、场景及帧级标注）作为训练、验证和测试数据集。

**📈 对比分析**

与单模态模型相比，Text Residual Fusion 在开发集、公开测试集和私有测试集的 MF1 分别提升至 76.13%、74.14% 与 78.24%，相较文本单模态提升 4.03%。

**⚠️ 局限性**

主要限制在于仍依赖预训练模型和手工选择的特征维度，且对长视频的时间建模相对简化，无法充分捕捉长时段的情绪变化。

---

## 295. An Intelligent-Cloud Edge Multimodal Interaction System for Robots

**arXiv ID:** 2607.14675 | [PDF](https://arxiv.org/pdf/2607.14675v1)

**作者:** Zihan Guo `[一作]` (Hainan University), Xiaoqi Li `[通讯]` (Hainan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了云边协同的多模态人机交互框架，改进YOLO-DC手势检测，并通过LLM与VLM实现语义理解与任务规划。

**💡 创新点**

创新点包括：在YOLO11n neck插入CBAM并使用DIoU损失提升小/遮挡手势的检测性能；将大模型推理与边缘控制分离，形成云边双代理架构；统一JSON协议、规则引擎和FSM确保跨模态交互安全与一致性。

**🔧 技术方法**

采用技术包括：YOLO-DC（YOLO11n+CBAM+DIoU），CLIP/BLIP等视觉‑语言模型，LLM（如LLaVA/Flamingo），规则引擎，有限状态机（FSM），JSON协议，Raspberry Pi/TonyPi 边缘硬件。

**📊 数据集**

使用了两组数据集：公开手势数据集（600张，6类）和自制数据集（3类）。

**📈 对比分析**

方法上与YOLOv5n、YOLOv8n及YOLO11n进行对比，公共数据集上精度达98.9%、mAP@0.5 90.7%；自制集上精度95.0%、mAP@0.5 92.7%；系统级任务成功率为单动作95%、复合动作88%、视觉相关任务82%；用户满意度平均得分3.69/5。

**⚠️ 局限性**

局限性：依赖云推理导致网络延迟和不稳定性；模型尚未压缩，边缘推理仍受限；未加入深度/触觉等多模态传感；安全性仅提出建议，未进行完整评估。

---

## 296. Identification Codes and Post-Shannon Communication: Theory, Architectures, and Emerging Applications

**arXiv ID:** 2607.14666 | [PDF](https://arxiv.org/pdf/2607.14666v1)

**作者:** Wafa Labidi `[一作]` (Technical University of Munich), Marc Geitz `[通讯]` (Deutsche Telekom)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统性综述了识别编码（ID）理论，阐述其与传统香农通信的区别，并探讨其在大规模分布式系统中的多种应用场景。

**💡 创新点**

创新点在于将ID编码与后香农、JIDAS、语义通信等新范式相结合，提出双指数扩展、共用随机性提升等关键特性，并针对安全性与控制层面展开前瞻性分析。

**🔧 技术方法**

采用信息论工具（如随机哈希、熵最大化、相对熵分析）以及通信网络模型（DMC、AWGN、Poisson、MIMO、广播、多访问等）进行理论推导，并结合实际系统架构设计。

**📊 数据集**

论文主要基于理论模型与公开案例（如灾害预警、移动网络控制、特定数据存储等），并未使用专门的实验数据集，而是通过系统级模拟与已有实测结果进行验证。

**📈 对比分析**

与传统传输相比，ID编码在识别容量上实现了指数级提升，实验与仿真显示其在低延迟、低信道开销与高可靠性方面优于经典方法；但不同信道模型下的误差类型与资源分配仍需细致评估。

**⚠️ 局限性**

局限性包括：对复杂非记忆性或非高斯信道的适配性不足、误差分析（I类与II类错误）在实际实现中的实现难度，以及在安全性与实时性之间的权衡仍未给出统一解决方案。

---

## 297. What's in a Smoothness Constant? Tighter Rates for Local SGD with Bounded Second-order Heterogeneity

**arXiv ID:** 2607.14731 | [PDF](https://arxiv.org/pdf/2607.14731v1)

**作者:** Kumar Kshitij Patel `[一作]` (Yale University), Lingxiao Wang `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

在一般凸目标下，给出了在二阶异质性约束下 Local SGD 的收敛上界，并在同一假设下给出了改进的下界，证明了 Local SGD 在更宽泛、现实的异质环境中可优于 Mini-batch SGD。

**💡 创新点**

创新点主要是：① 引入轨迹依赖的梯度差异控制，替代传统的全局梯度差异假设；② 通过自界递推实现一致误差与迭代误差的闭合；③ 设计新的下界构造，分别解耦一阶与二阶异质性，说明二阶异质性是实现局部更新优势的关键结构。

**🔧 技术方法**

技术方法包括：光滑与凸性分析、平均梯度和局部梯度之间的误差递推、轨迹相关的梯度不相似度估计、梯度漂移控制的自界循环、以及多元下界构造（利用不同子空间、不同曲率、不同噪声水平）来证明下界。

**📊 数据集**

该工作为纯理论分析，未使用具体数据集；所有结果基于一般凸、光滑且满足二阶异质性约束的假设。

**📈 对比分析**

与 Mini-batch SGD 的比较表明：当局部计算量 K 较大、二阶异质性 τ 及最优解附近的一阶异质性 ζ⋆ 较小时，Local SGD 的误差下降率能显著优于 Mini-batch SGD；上界与下界在大部分参数范围内匹配，证明了结论的近乎最优性。

**⚠️ 局限性**

局限性：① 在某些红色区域（特定 τ 与 ζ⋆ 取值）仍存在上界与下界不匹配的 gap；② 需要二阶可微性与光滑假设，若目标非二阶可微可能无法直接应用；③ 研究仅针对同步 intermittent 通信场景，异步、压缩、差分隐私等实际情况未被覆盖；④ 结果尚未在实验中验证，实际性能还需进一步探究。

---

## 298. VQ-Touch: A Data-Efficient Tactile Generation Framework Across Sensors and Scenarios

**arXiv ID:** 2607.14728 | [PDF](https://arxiv.org/pdf/2607.14728v1)

**作者:** Kailin Lyu `[一作]` (Chinese Academy of Sciences), Jie Hao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种数据高效的触觉图像生成框架，能够在不同传感器和多种场景下跨模态生成高保真触觉图像；核心包括DM‑VQGAN编码器/解码器和统一的离散扩散解码器，支持视觉、触觉及标签等多模态输入；

**💡 创新点**

① 结合可变形卷积与多尺度扩张卷积的DM‑VQGAN，能够捕捉宏观变形与微观纹理；② 统一的多模态离散扩散框架，可对任一模态进行条件生成；③ 少样本混合训练与传感器家族聚类实现跨传感器泛化；④ 通过少量样本即可在新传感器上实现高质量重建与生成；

**🔧 技术方法**

VQGAN+可变形卷积、稀疏多尺度融合、离散扩散模型、CLIP预训练编码器、对抗损失、感知损失、少样本混合训练、聚类算法；

**📊 数据集**

Touch & Go、FabricVST、VisGel、HCT（GelSight、GelSight Mini、DIGIT传感器）；

**📈 对比分析**

与VQGAN、MAE、UniTouch、PixArt‑α、TextToucher等SOTA方法在FID、SSIM、LPIPS、PSNR、LPIPS/SSIM、分类精度等指标上对比；在低样本重建、跨传感器重建、视觉/标签条件生成以及下游识别任务中均表现出最优或接近最优的性能；

**⚠️ 局限性**

对未知传感器的适配依赖聚类结果，若聚类误差导致家族不准则会影响性能；少样本混合训练仍需一定样本量；扩散生成速度较慢；当前仅针对静态触觉图像，未覆盖时序触觉序列；极端光照或极低光照条件下的鲁棒性待进一步验证。

---

## 299. BridgeFlow: Fast and Robust SE(2)-Equivariant Motion Planning with Flow Matching

**arXiv ID:** 2607.14725 | [PDF](https://arxiv.org/pdf/2607.14725v1)

**作者:** Xinzhe Zhou `[一作]` (Shanghai Jiao Tong University), Jianping He `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了BridgeFlow框架，结合Flow Matching实现快速、严格的SE(2)-等变生成运动规划；

**💡 创新点**

创新点在于：①轻量化任务中心化正则化实现严格SE(2)等变；②Brownian Bridge信息先验+上下文感知小批量最优传输显著降低流场弯曲；③环境感知的Classifier‑Free Guidance嵌入无需运行时梯度优化；

**🔧 技术方法**

使用Flow Matching、Brownian Bridge先验、Mini‑Batch OT、Classifier‑Free Guidance、U‑Net等标准网络；

**📊 数据集**

数据集：PointMass2D Dense（200个稠密障碍环境）和7‑DoF Franka Panda pick‑and‑place（四象限平面）；

**📈 对比分析**

与MPD（扩散）和FlowMP（无先验的FM）对比；BridgeFlow在5步推理下达到约80%有效率，推理速度比MPD提升约15×；在未知障碍环境下有效率提升≈20%；在SE(2)平移/旋转测试中，BridgeFlow维持≈95%成功率，超越基线5‑10%；

**⚠️ 局限性**

局限包括：①仅在二维/SE(2)环境验证，需扩展到SE(3)；②噪声尺度与指导权重仍需手动调节；③对极端动态障碍或高维更复杂约束的鲁棒性尚待验证。

---

## 300. Multimodality as Supervision: Self-Supervised Specialization to the Test Environment via Multimodality

**arXiv ID:** 2607.14721 | [PDF](https://arxiv.org/pdf/2607.14721v1)

**作者:** Kunal Pratap Singh `[一作]` (Swiss Federal Institute of Technology Lausanne), Amir Zamir `[通讯]` (Swiss Federal Institute of Technology Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在单一测试空间内收集多模态数据，通过跨模态学习进行自监督预训练，从而得到针对该空间的高性能视觉表示。

**💡 创新点**

提出 Test‑Space Training (TST) 框架，证明仅利用测试空间的多模态自监督学习即可匹敌甚至超过互联网规模通用预训练模型，并揭示多模态比单一模态或外部数据更有效的特性。

**🔧 技术方法**

使用跨模态掩码建模（multimodal masked modeling）作为预训练目标，结合 Transformer 编码器‑解码器；后期在少量迁移数据上 fine‑tune 进行语义分割、目标检测与图像描述。

**📊 数据集**

ScanNet++、Replica、ProcTHOR 三个室内环境数据集，用作测试空间和迁移数据。

**📈 对比分析**

与 DINOv2、4M‑21、CLIP、MAE 等通用预训练模型以及 Mask2Former、ViTDet、SAM、LLaVA 等任务专用基线比较。实验显示 TST（含感知模态）在分割、检测、描述任务上均能与通用模型竞争；加入伪标签模态后取得在测试空间上的最优表现，甚至超越任务专用基线。

**⚠️ 局限性**

仍无法完全达到全监督上限；需要多模态硬件支持；在多模态与伪标签模态上过度依赖外部预训练网络；实验仅在室内固定空间进行，是否能推广到更广泛环境仍需验证。

---

## 301. Lattice-based extended withdrawability

**arXiv ID:** 2607.14690 | [PDF](https://arxiv.org/pdf/2607.14690v1)

**作者:** Ramses Fernandez-Valencia `[一作]` `[通讯]` (Fairgate Labs), Ramses Fernandez-Valencia (Fairgate Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种基于格的扩展可撤回签名方案，使签名在未确认前保持匿名、公开可验证，并在确认后可追溯。

**💡 创新点**

创新点包括：①将撤回性与可公开验证性拆分为可索引的声称环签名；②采用格的一对多零知识证明和 Ajtai 承诺实现匿名与绑定；③在安全模型中加入“声明可证明性”新概念；④提供无 q_H 失真的紧凑安全证明，并显式展示格系统下直接转录不可行的原因。

**🔧 技术方法**

使用的技术包括：格的 Fiat‑Shamir with aborts 方案（Dilithium‑style 基础签名）、格的一对多证明（匿名一对多签名）、Ajtai 承诺（隐藏与绑定）、零知识、随机预言机模型下的安全分析。

**📊 数据集**

该工作为理论分析，不使用具体数据集；所有结果基于模块格假设（MLWE、MSIS、SelfTargetMSIS）。

**📈 对比分析**

与传统离散对数的扩展可撤回签名相比，提出的方案在后量子安全性、隐私保持（不泄露签名者标识）以及证明紧凑性方面均优于前者；实验性能未给出，但利用 Dilithium 风格基础签名和对数规模一对多证明，可预计签名尺寸与公钥规模相近。

**⚠️ 局限性**

局限性包括：尚未给出完整的可实施参数集；依赖公共矩阵 A 的共享，需要在实际部署中解决；签名尺寸和计算成本较大；安全性依赖随机预言机模型。

---

## 302. Statistics of the Compression Ratio of a Variable-to-Variable Code: Exact Moments and Asymptotic Behavior

**arXiv ID:** 2607.14676 | [PDF](https://arxiv.org/pdf/2607.14676v1)

**作者:** Neri Merhav `[一作]` `[通讯]` (Technion Israel Institute of Technology), Neri Merhav (Technion Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文推导了变长编码（V2V）在给定离散无记忆源（DMS）下的压缩比R_n的所有整数矩的精确公式，并提出了更准确的Edgeworth近似。

**💡 创新点**

创新点在于系统性地扩展了每个整数矩的推导，并且得到了在有限n下的精确公式，而不仅仅是渐近结果。

**🔧 技术方法**

使用了拉普拉斯积分法和矩生成函数的工具，结合了组合数学的分割技术。

**📊 数据集**

使用了离散无记忆源（DMS）作为数据集，并扩展到马尔可夫源的情况。

**📈 对比分析**

通过与中心极限定理（CLT）进行比较，本文的Edgeworth近似在压缩比的累积分布函数（CDF）上表现出显著更高的准确性。

**⚠️ 局限性**

限制在于当前的分析主要集中在特定类型的源上，且对于更复杂的源模型的推广仍需进一步研究。

---

## 303. Project Kaleidoscope: Contextual, Human-Aligned Evaluation for Real-World AI Applications

**arXiv ID:** 2607.14673 | [PDF](https://arxiv.org/pdf/2607.14673v1)

**作者:** Leanne Tan `[一作]` (GovTech), Roy Ka-Wei Lee `[通讯]` (GovTech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了 Kaleidoscope——一种端到端的上下文功能评估工作流，整合了人物化测试生成、可配置评估 rubrics、人机校准以及可靠性门控的 LLM 判断；

**💡 创新点**

创新点在于将测试生成、评分标准制定、人工校准与多判决器自动评分无缝集成，并通过局部可靠性门控确保自动评分的可解释性和可治理性；

**🔧 技术方法**

采用 LLM 生成测试、LLM-as-judge 自动评分、多判决器（jury）架构、基于人机标注的可靠性校准、可配置的评分 rubrics 与基于人物维度的测试多样化；

**📊 数据集**

使用 108 条已注释的问答对（涵盖四个领域、十四个评估维度）进行裁决器实验，并在四个组织用例（财务、HR、采购、员工助手）中进行三周 pilot，生成约 180 条测试案例、40 条人工审核输出；

**📈 对比分析**

与传统基准或单一 LLM 判断相比，Kaleidoscope 在 pilot 中显示更高的评估效率（83% 用户认为更高效）并能按需求调节评估粒度；性能主要体现在可靠性门控能过滤低质量判决器，且多判决器设计提高了评分稳健性；

**⚠️ 局限性**

局限性包括：评估成本高（尤其是多判决器和声称级评分）、对上下文信息的依赖导致部分用例设置困难、校准样本量与评估频率的权衡难以统一、仅覆盖输入-输出评估，无法完整捕获代理或检索系统的中间行为；

---

## 304. VIABench: A Comprehensive Video Benchmark Collected from Blind Individuals for Visual Impairment Assistance

**arXiv ID:** 2607.14660 | [PDF](https://arxiv.org/pdf/2607.14660v1)

**作者:** Yunfeng Liu `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了VIABench视频基准，用于评估多模态大型语言模型在视障辅助中的实时表现，包含三大核心任务：主动提醒、视觉问答和视角引导交互；

**💡 创新点**

创新点在于将长时序第一人称视障视频与多任务评估结合，并提出Token-Level Prompt Activation Decoding（TPAD）实现离线模型的实时主动提醒；

**🔧 技术方法**

主要技术包括多模态大型语言模型（InternVL、Qwen、LLaVA、MiniCPM 等）、TPAD 两阶段评估框架，以及在线/离线模型的统一推理策略；

**📊 数据集**

使用了新构建的VIABench数据集，包含761段长时序第一人称视障视频、14,526条手工标注事件，覆盖三类任务；

**📈 对比分析**

通过与多种开源、闭源和在线流式模型对比，实验显示即使是最强模型（如GPT‑5）在主动提醒任务的平均得分仅约28.8，显示出当前模型在实时导航提醒与交互方面的显著性能瓶颈；

**⚠️ 局限性**

局限性包括：现有模型在主动提醒和交互任务上的能力有限，缺乏足够的实时上下文理解与预测；数据训练多集中于密集字幕或自动生成的叙述，缺乏真实视障场景下的指令多样性；模型对低质量第一人称视频的鲁棒性不足。

---

## 305. LLM-Driven Approach to Modeling Tool Interoperability in Automotive Domain

**arXiv ID:** 2607.14659 | [PDF](https://arxiv.org/pdf/2607.14659v1)

**作者:** Nenad Petrovic `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于大语言模型（LLM）的自动化模型实例映射与元模型合并方法，支持Ecore与SysML v2之间的互操作，并通过n8n工作流实现端到端的文件处理、模型生成与自动验证。

**💡 创新点**

创新点在于：①直接让LLM生成目标兼容的模型实例与统一元模型，省去手工编写转换规则；②将对话式交互、Prompt工程、自动化验证融合到同一流水线；③采用本地可部署LLM，保障数据安全与可控；④在自动生成后立即进行结构验证，支持迭代细化。

**🔧 技术方法**

技术要点包括：大语言模型（Gemma、Qwen）与Prompt工程；n8n自动化工作流；Python脚本；PyEcore验证器；SysML v2文本解析；XMI生成；以及基于OpenAI API兼容的本地推理服务器。

**📊 数据集**

使用了自构造的代表性汽车硬件模型集，包含感知与执行组件，目标元模型有27个元素；在四个转换场景（Ecore↔Ecore、SysML v2↔SysML v2、Ecore↔SysML v2、SysML v2↔Ecore）下进行5次实验；同时对元模型合并场景（Ecore+SysML v2→Ecore、Ecore+SysML v2→SysML v2）进行评测。

**📈 对比分析**

评测方法：对比语义匹配率（是否完整映射所有语义元素）和语法正确率（验证器是否通过）。结果显示：语义匹配率始终为100%；在Ecore目标下语法正确率为100%，在SysML v2目标下分别为80%/60%（Gemma/Qwen）；在元模型合并时语义匹配率为100%，语法正确率为100%/60%（Gemma/Qwen）。执行时间平均为Gemma 5.1 s、Qwen 46.5 s（映射）和Gemma 15.2 s、Qwen 58.3 s（合并）。

**⚠️ 局限性**

主要限制：①SysML v2文本生成易出现语法偏差，导致验证率下降；②对大模型文件的上下文处理受限，需后续引入MCP/RAG等技术；③当前仅支持Ecore与SysML v2，需扩展到更多建模语言；④LLM生成的非标准序列化或约束细节可能需要人工修正。

---

## 306. GAttNHP: Group Attention Neural Hawkes Process for Extrapolation Reasoning in Temporal Knowledge Graphs

**arXiv ID:** 2607.14733 | [PDF](https://arxiv.org/pdf/2607.14733v1)

**作者:** Xiangni Tian `[一作]` (Yunnan University), Hongtu Zhu `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一框架 Group Attention Neural Hawkes Process（GAttNHP），实现对时序知识图谱（TKG）的未来事件链接预测和时间预测。

**💡 创新点**

创新点包括：① 用自注意力编码器捕获单链的长期连续时间依赖；② 通过语义软分组将全局可学习的 Hawkes 先验转化为分析式交叉注意力掩码，实现跨链相互激励而不需要逐对计算；③ 采用 Non-Crossing Quantile（NCQ）回归头给出校准且单调的时间分位点估计，解决重尾间隔导致的均方误差偏差。

**🔧 技术方法**

技术手段包括：Transformer‑style 多层自注意力、语义软分组与 Hawkes 掩码、Softplus 激活、非交叉分位数回归、联合负对数似然与 pinball 损失的联合训练。

**📊 数据集**

使用六个主流 TKG 基准数据集（ICEWS18、ICEWS14、ICEWS05‑15、GDELT、WIKI、YAGO）进行评估。

**📈 对比分析**

与静态 KG、TKG 插值、TKG 外推以及连续时间点过程基线（如 GHT、GHNN、RQS‑QF、RMTPP）对比，GAttNHP 在实体预测上显著提升 MRR（例如 ICEWS14 MRR 0.5068，提升 8.25pp）并在时间预测上 MAE 下降 47–73%（例如 ICEWS14 1.33 天），同时提供更可靠的置信区间。

**⚠️ 局限性**

局限性：① 软分组结构是固定的，未自适应调整；② 对极端稀疏或长尾链的捕获仍受限；③ 计算复杂度主要集中在多头注意力与全局掩码生成，推理速度相对慢；④ 未考虑多跳推理的组合效应。

---

## 307. Toward Energy-Efficient and Low-Power Arrhythmia Detection for Wearable Devices

**arXiv ID:** 2607.14747 | [PDF](https://arxiv.org/pdf/2607.14747v1)

**作者:** Floriaan Bulten `[一作]` (University of Twente), Ghayoor Gillani `[通讯]` (University of Twente)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研发了一种基于深度学习的低功耗心律失常检测系统，采用精度缩放到8位并使用近似乘法器，结合峰值检测与重训练实现高效推理。

**💡 创新点**

创新点在于将深度网络权重和数据精度降至8位并引入近似乘法器（如SSM4、CSSM4），通过误差可校正的重训练实现功耗下降64.9%（12 kHz）/61.5%（100 MHz）同时保持93.7%准确率与92.1%敏感度。

**🔧 技术方法**

使用技术包括：近似乘法器与精度缩放、量化感知训练、三层全连接深度神经网络、峰值检测算法以及在ASIC中实现的低功耗加速结构。

**📊 数据集**

采用 MIT‑BIH 心律失常数据库进行训练、验证与测试，包含15类标准心律。

**📈 对比分析**

通过与多种准确/近似乘法器实现以及文献中已发表的ASIC/FPGA设计比较，S8GC(R)在12 kHz时功耗3.07 µW、能耗2.17 µJ/分类、准确率93.7%、敏感度92.1%；在100 MHz时功耗9.45 mW、能耗0.801 µJ/分类，均比基准节能64.9%/61.5%。

**⚠️ 局限性**

局限性包括：峰值检测误差导致部分心律类别（如F类）敏感度下降；低频运行时近似乘法器优势不明显；未针对各类心律的专门误差补偿，临床真实环境下的验证仍待进一步开展。

---

## 308. On the Disagreement in Perturbation-based xAI -- Benchmarking Perturbation Choices for Flood Detection from SAR Images

**arXiv ID:** 2607.14743 | [PDF](https://arxiv.org/pdf/2607.14743v1)

**作者:** Anastasia Schlegel `[一作]` (German Aerospace Center), Ronny Hänsch `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基于扰动的可解释 AI 方法在 SAR 洪水检测中的参数设置进行系统性研究，评估不同补丁几何和替换方式对相关性地图的影响。

**💡 创新点**

首次从参数角度揭示扰动解释的分歧问题，证明补丁大小/形状和扰动类型决定了解释结果的可信度和一致性。

**🔧 技术方法**

采用扰动归因框架，结合多种补丁形状（正方形、SLIC 超像素）与五种替换策略（通道最小值、类别均值、随机常数、图像补丁、像素洗牌），并使用一致性、距离分布与删除式 Dice-AUC 等评价指标。

**📊 数据集**

使用 Sentinel‑1 公开洪水数据集 Sen1Floods11（约 450 张 512×512 的双极化 SAR 图像）进行实验。

**📈 对比分析**

通过可视化、相关性分布一致性和删除式正确性评估，比较不同参数组合的表现；结果显示：大补丁产生低分辨率、易聚合的解释；小/超像素补丁得到更细粒度但计算成本高的地图；类别均值、随机常数、补丁替换在多数场景下相似且更符合模型内部逻辑；最小值和像素洗牌虽稳健，但往往与模型真实依据不完全匹配；AUC 在 0.6–0.8 之间波动，表明解释的忠实度受参数显著影响。

**⚠️ 局限性**

局限性包括：评价仅基于单一数据集，无法验证跨域泛化；扰动引入的分布外样本可能导致解释偏差；删除式评估对扰动方式敏感，可能误判可靠性；计算开销随补丁数量呈二次增长，实际部署受限。

---

## 309. Harnessing LLMs for Reliable Academic Supervision: A Comparative Study

**arXiv ID:** 2607.14707 | [PDF](https://arxiv.org/pdf/2607.14707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 310. FoMoVLA: Bridging Visual Foresight and Motion Guidance for Vision-Language-Action Models

**arXiv ID:** 2607.14739 | [PDF](https://arxiv.org/pdf/2607.14739v1)

**作者:** Wei Li `[一作]` (LiAuto), Kun Zhan `[通讯]` (LiAuto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在Vision‑Language‑Action模型中加入未来特征预测和稀疏二维点追踪的训练监督，提升机器人操纵时的时空先见性与连续动作生成

**💡 创新点**

创新点在于将未来视觉状态预测与点轨迹监督联合学习，并通过未来条件跨注意模块（FCCA）实现两者互相约束，使模型既能知晓目标状态又能掌握实现路径

**🔧 技术方法**

利用预训练VLM骨干、EMA教师进行特征预测、MAE解码器生成K-token瓶颈、CoTracker‑v3提供稀疏点轨迹监督、跨注意（MHA）实现未来条件

**📊 数据集**

在LIBERO、RoboCasa GR‑1 Tabletop以及LIBERO‑Plus三大基准上进行训练与评测，包含多种操纵任务与扰动维度

**📈 对比分析**

与多组基线（一般VLA、未来预测、点追踪）对比，FoMoVLA在LIBERO各子集与RoboCasa任务中均取得最高平均成功率，尤其在长周期和空间控制任务中提升显著，零样本外域泛化也优于现有方法

**⚠️ 局限性**

局限在于仅使用二维图像空间的点追踪，无法捕捉三维几何变化，导致在视角或深度变化下的鲁棒性有限

---

## 311. AE-UAV: An Air-to-Air Event-Based UAV Tracking Benchmark and a Real-Time Frequency-Domain Tracker

**arXiv ID:** 2607.14726 | [PDF](https://arxiv.org/pdf/2607.14726v1)

**作者:** Zixin Jiang `[一作]` (Rocket Force University of Engineering), Ling Pei `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个空中捕获的事件摄像机数据集 AE-UAV，并基于事件流实现了轻量级实时跟踪器 FSFT。

**💡 创新点**

创新点在于：①使用三次 B‑spline 进行连续时间标注，避免传统离散框标注误差；②设计无训练、仅 CPU 的频域模板匹配与周期性漂移纠正框架；③融合卡尔曼预测、正态流方向估计与事件过滤，实现 420 FPS 的高帧率。

**🔧 技术方法**

采用事件滤波、频域幅值相关、FFT 频域模板匹配、卡尔曼滤波器、正态流方向估计及基于事件密度与梯度的检测纠正。

**📊 数据集**

使用 AE-UAV 数据集：178 条空中-空中飞行序列，分辨率 1280×720，含 8.15 亿事件与 31.2 万目标事件，提供连续时间 B‑spline 注释。

**📈 对比分析**

与多种基于 GPU 的深度学习跟踪器（OSTrack、FocusTrack 等）进行对比，FSFT 在不需要 GPU、无训练阶段的前提下，帧率 420 FPS，ATQ 100.22，准确率仅比最优深度模型低约 6%，且在不同事件聚合率下保持 6% 以内的稳定性。

**⚠️ 局限性**

主要局限包括：①在纹理丰富背景下高频噪声与目标边缘共频导致匹配误差；②使用一阶卡尔曼模型难以捕捉急剧转弯的 UAV 运动；③仅支持单目标，遇到多 UAV 时可能锁定错误目标；④缺乏针对多目标或复杂场景的判别学习模块。

---

## 312. Gold-Guided Programmatic Distillation for Financial Reasoning over Hybrid Tables and Text

**arXiv ID:** 2607.14709 | [PDF](https://arxiv.org/pdf/2607.14709v1)

**作者:** Yun Dong `[一作]` (Georgia Institute of Technology), Elana Chen `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于金手指的程序化蒸馏框架，将大型教师模型生成的可执行Python程序迁移到小型学生模型，实现对混合表格与文本金融问答的精确数值推理。

**💡 创新点**

创新点包括：①利用金手指导向的程序生成并通过代码执行验证来保证中间推理的正确性；②引入迭代回收阶段，对教师未能解决的样例通过学生采样并再次执行验证，扩展监督数据；③在学生训练中采用LoRA并将注意力与MLP层同时适配。

**🔧 技术方法**

技术方法包括：程序化蒸馏（Program-of-Thought）、DSPy提示优化、低秩适配（LoRA）、沙箱式Python执行验证、基于拒绝采样的自我改进（Rejection Sampling Fine‑Tuning）。

**📊 数据集**

实验数据集为TAT‑QA金融问答数据集的筛选子集（仅包含有金手指推导的实例），共约6,603训练、834验证、831测试样例。

**📈 对比分析**

与任务专用基线（TAGOP）、教师模型（Qwen2.5‑72B）及其他LLM基线（TAT‑LLM）对比，最终7B学生模型在测试集上达87.00 EM / 87.18 F1，显著超越72B教师（78.46 EM / 78.73 F1）与最佳传统基线。

**⚠️ 局限性**

局限性：①评估仅基于筛选子集，规模有限，缺乏简单抽取式问题；②尽管提升显著，但仍存在答案预测、尺度预测及两者同时错误的残留；③未解决如何自动判别是否需要程序化推理与抽取式推理的策略。

---

## 313. MESHA: Mechanism-Enforced Sequential Halving for Strategic Linear Bandits

**arXiv ID:** 2607.14706 | [PDF](https://arxiv.org/pdf/2607.14706v1)

**作者:** Xin Li `[一作]` (Hong Kong University of Science and Technology), Zixin Zhong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并分析了一种名为MESHA的算法，用于在存在自利臂策略的线性Bandit环境中进行固定预算下的最佳臂识别问题；

**💡 创新点**

创新点在于将均匀采样与 epoch‑wise Grim Trigger Condition（GTC）机制结合，形成机制约束，保证在任意 Nash 均衡下算法仍具有 PAC 成功概率；

**🔧 技术方法**

主要技术包括均匀采样、Grim Trigger 触发条件、置信区间估计、线性回归估计、弹性置信球等；

**📊 数据集**

实验使用人工构造的多种战略攻击场景（包含特征误报、稀缺采样等），在不同预算、维度、臂数下进行评估；

**📈 对比分析**

与 Sequential Halving、OD‑LinBAI、OD‑LinBAI‑GTC、OptGTM 等基线对比，MESHA 在预算、维度、臂数上均优于基线，尤其在战略环境中表现更为稳定；

**⚠️ 局限性**

局限在于算法在统计上存在 O(d² log T) 的额外开销，并且假设所有臂均处于 Nash 均衡，无法直接处理非均衡或奖励操纵的情况。

---

## 314. Grad2Fair: A Gradient-driven Approach for Graph Fairness without Demographics

**arXiv ID:** 2607.14705 | [PDF](https://arxiv.org/pdf/2607.14705v1)

**作者:** Yuchang Zhu `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Gradient-to-Fairness 方法，在不获取人口统计信息的前提下，通过梯度信息实现图神经网络的群体公平性。

**💡 创新点**

创新点包括：① 设计无敏感属性的偏差评估指标 τ_{grad}，利用梯度分布的局部模态距离量化偏差；② 两阶段偏差放大与梯度加权框架，直接利用误分类节点梯度提升模型对少数族群的关注。

**🔧 技术方法**

技术手段：基于 GCN/GIN 的 GNN 训练；使用 Kernel Density Estimation 估计梯度分布并提取模态；在偏差放大阶段挑选高置信样本，在加权阶段对误分类节点按梯度范数加权；结合理论分析阐明两阶段对公平性的影响。

**📊 数据集**

数据集：四个真实世界数据集（Bail、Credit、Pokec-z、Pokec-n）以及三类合成数据集（SynFair、AttrBias、StruBias）进行实验验证。

**📈 对比分析**

与 FairGKD、Fairwos、FDKD 等基准在 GCN 与 GIN 两种 backbone 上进行对比。实验结果显示在 F1/ACC 上保持或提升性能，同时 Δ_DP 与 Δ_EO 明显降低，证明公平性提升；并且训练效率高于现有方法。

**⚠️ 局限性**

局限性：目前仅支持二值敏感属性；在结构简单、缺乏明显局部差异的数据上，偏差放大阶段效果有限；对多值或更复杂属性的推广仍需进一步研究。

---

## 315. Scalable Training of Continuous-Time Spiking Neural Networks with Differentiable Spike-Time Discretization

**arXiv ID:** 2607.14672 | [PDF](https://arxiv.org/pdf/2607.14672v1)

**作者:** Yusuke Sakemi `[一作]` (Chiba Institute of Technology), Kazuyuki Aihara `[通讯]` (Chiba Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于可微分脉冲时间离散化（DSTD）和突触链动力学的可扩展训练框架，可高效训练大规模连续时间脉冲神经网络（SNN），并实现了多层卷积SNN在CIFAR-10和Fashion‑MNIST上的训练；

**💡 创新点**

创新点在于①将连续时间脉冲映射到固定时间点的可微分权重事件，显著降低候选脉冲时刻的数量与内存占用；②通过突触链激励的时间窗口正则化，解决了深层SNN中“死神经元”问题并实现了流水线式数据处理；③将上述技术结合，首次在大规模卷积SNN中实现了训练速度提升约20倍、峰值内存降低约100倍；

**🔧 技术方法**

使用的主要技术包括可微分脉冲时间离散化（DSTD）、时间到第一脉冲（TTFS）编码、突触链时间窗口正则化、梯度反向传播和跳跃-延迟残差机制；

**📊 数据集**

使用的数据集为CIFAR-10（50000训练/10000测试）和Fashion‑MNIST；

**📈 对比分析**

与传统的精确脉冲时间计算方法相比，DSTD在相同精度下可将训练时间缩短约20倍、峰值内存降低60–150倍；在9层卷积SNN上达到了较高的分类准确率（约90%+），在20层卷积SNN上保持了层间脉冲时间的有序传播并实现了流水线处理；

**⚠️ 局限性**

局限性包括：①对TTFS单脉冲假设的依赖，难以直接推广到多脉冲情形；②DSTD近似梯度的理论理解尚不充分，可能影响极限精度；③训练仍需大量超参数搜索，理论指导不足；④对深层SNN的可扩展性在更大网络/更复杂任务上尚未验证。

---

## 316. WorkDrive: Roadwork Chain of Causation for Autonomous Driving

**arXiv ID:** 2607.14727 | [PDF](https://arxiv.org/pdf/2607.14727v1)

**作者:** Tianyi Jiang `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 WorkDrive 框架，将感知信息与因果推理结合用于工地道路行驶轨迹预测。

**💡 创新点**

创新点在于通过自动化感知管线构建结构化场景事实，进而生成基于 Chain-of-Causation 的因果推理标签，并用单一一致性奖励进行强化学习，实现无轨迹监督下的规划对齐。

**🔧 技术方法**

使用多模态感知模型（Depth Anything、Rex-Omni、SAM、跨帧追踪）、LLM（Qwen3-VL 或 GPT‑5.2high）生成因果标签、LoRA 细调以及 Group Relative Policy Optimization（GRPO）一致性对齐。

**📊 数据集**

数据集为 ROADWork，包含 4375 条工地行驶视频与轨迹，且自行构造了推理标签。

**📈 对比分析**

与零样本 VLM 基线对比，WorkDrive 在 ADE@15 上提升约 12%（从 12.10 降至 10.68），碰撞率下降 77%，一致性率提升至 100%。

**⚠️ 局限性**

局限在于仅针对 2D 像素轨迹，缺乏 3D/BEV 适配；推理标签生成仍依赖 LLM，存在质量噪声；以及对复杂多车交互情形的验证不足。

---

## 317. Class Weighting versus Amount Conditioning in Credit-Card Fraud Detection: A Dollar-Metric Study with a Temporal Explanation Audit

**arXiv ID:** 2607.14686 | [PDF](https://arxiv.org/pdf/2607.14686v1)

**作者:** Chenyu Wu `[一作]` `[通讯]` (Duke University), Chenyu Wu (Duke University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究在保持总欺诈样本权重不变的情况下，按交易金额分配样本权重对信用卡欺诈检测的影响。

**💡 创新点**

创新点在于将金额权重与类别平衡区分，系统评估金额作为特征、样本权重和推断时排序的三种作用，并提供以金钱回报为基准的评估指标。

**🔧 技术方法**

使用XGBoost梯度提升树、按金额对权重设定、score×金额重新排序以及SHAP解释漂移审计。

**📊 数据集**

实验数据集为公开的合成Sparkov和真实的IEEE‑CIS信用卡交易集。

**📈 对比分析**

与基准plain、class‑weighting等对比，匹配金额权重仅略优于类别权重，强金额权重提升金钱召回但降低精度，后者的score×金额重新排序在金钱召回上最显著。

**⚠️ 局限性**

局限包括仅使用梯度提升树、未针对不同权重方案单独调参、合成数据缺乏实际复杂度、未考虑实际成本模型、样本权重影响正则化、统计检验不完整以及SHAP漂移仅在少数模型上完成。

---

## 318. GlobalForge: Towards Robust AI-Generated Image Detection

**arXiv ID:** 2607.14684 | [PDF](https://arxiv.org/pdf/2607.14684v1)

**作者:** Manni Cui `[一作]` (Huazhong University of Science and Technology), Shu Wu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GlobalForge 框架，主动抑制局部伪影并引导全局结构推理，以提升 AI 生成图像检测在真实降质链中的鲁棒性。

**💡 创新点**

核心创新在于 Local Information Bottleneck 与 Global Structural Reasoning 两个模块的联合设计，并配合降质感知对比结构损失以及全新的 RealDeg‑Bench 评测基准。

**🔧 技术方法**

使用 ViT-L 预训练骨干，结合 LIB、GSR 模块、对比结构损失（InfoNCE）以及 LoRA 微调等技术实现模型训练。

**📊 数据集**

训练使用对齐的真实/重建样本，评估基准为 RealDeg‑Bench（基于 T2I‑CoReBench）和八组在野外公开数据集。

**📈 对比分析**

与 12 代表性检测器以 BAcc 为指标对比，GlobalForge 在 RealDeg‑Bench 平均 BAcc 85.81% 及野外测试 85.93%，比最强基线提升约 5.9%。

**⚠️ 局限性**

局限性包括对降质感知损失权重和掩蔽超参数的敏感性，且对新型生成模型和更复杂降质链的泛化仍待验证。

---

## 319. Stop Thinking, Start Looking: Efficient Post-Training for Multimodal Document Question Answering via Reasoning-Free Alignment

**arXiv ID:** 2607.14682 | [PDF](https://arxiv.org/pdf/2607.14682v1)

**作者:** Harikrishnan P M `[一作]` (Phi Labs, Quantiphi), Rohit Agrawal `[通讯]` (Phi Labs, Quantiphi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Perception-RFT框架，直接通过强化学习优化文档视觉问答的视觉定位而不使用中间推理步骤。

**💡 创新点**

证明在多模态文档问答中，显式推理对性能无益，模型会自发压缩或抛弃推理，并且强化学习可显著提升几何对齐。

**🔧 技术方法**

使用Group Relative Policy Optimization（GRPO）、门控稠密奖励、LoRA微调、Qwen3‑VL‑4B多模态LLM以及对齐的系统提示。

**📊 数据集**

基于DocILE、FormNLU等关键信息提取数据集，转化为动态问答样本，同时在两大OOD基准DOGR‑Bench和MMDocBench上进行评估。

**📈 对比分析**

与SFT、冷启动RL、Reasoning‑RFTb等对照，发现SFT→RL（RFTs）在ID上取得最高joint grounding（F1_all≈0.718），在OOD上显著提升定位精度但语义略降；Reasoning‑RFTb表现最差。

**⚠️ 局限性**

实验仅覆盖4B参数规模，未评估更大模型的可扩展性；缺乏推理标注数据导致无法验证推理+SFT→RL混合训练效果，且在分布外仍存在定位与语义之间的权衡。

---

## 320. ReBind: Multi-Reference Video Editing via Structured Instructions with Explicit Reference Relationships

**arXiv ID:** 2607.14681 | [PDF](https://arxiv.org/pdf/2607.14681v1)

**作者:** Xinyu Liu `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为ReBind的框架，旨在通过引入嵌入参考标记的语义指令来改善多参考图像条件的视频编辑。

**💡 创新点**

创新点在于通过密集指令和明确的参考关系消除模糊性，从而在视觉属性和其来源之间建立精确的绑定。

**🔧 技术方法**

使用了两阶段的训练方案，包括结构化指令学习和参考归属优化，结合了强化学习技术。

**📊 数据集**

构建了一个合成数据集，包含约10万个视频编辑三元组，涵盖单参考和多参考样本。

**📈 对比分析**

与通用的多模态大语言模型（MLLMs）进行比较，ReBind在指令质量和视频编辑性能上显著优于现有方法，尤其在多参考协调任务中表现出色。

**⚠️ 局限性**

限制在于当前方法可能在处理极端复杂的多参考场景时仍然存在挑战，且对数据集的质量和多样性依赖较大。

---

## 321. SmartRAG: Native Graph-Based RAG for Mobile Device

**arXiv ID:** 2607.14661 | [PDF](https://arxiv.org/pdf/2607.14661v1)

**作者:** Zhihan Jiang `[一作]` (Nanjing University), Haipeng Dai `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在移动端全本地化的结构化检索增强生成框架SmartRAG，拆分为感知、记忆、聚焦、思考四个模块，实现低成本、低延迟的多跳问答；

**💡 创新点**

通过将持续学习的实体识别EvoNER与增量更新的知识图MRGraph结合，构建可在线扩展的结构化记忆；使用多模态检索（图遍历+词汇匹配+语义检索）与LLM仅在高价值操作时调用，显著降低推理成本；

**🔧 技术方法**

核心技术包括span‑based NER与保留标签机制、教师蒸馏增量学习、轻量化关系分类、三层知识图MRGraph、混合检索管线、量化（Q6_K）LLM推理和LoRA微调；

**📊 数据集**

使用 TriviaQA、Natural Questions、HotpotQA、MultiHopQA 四个多跳/事实类问答基准作为评估数据集；

**📈 对比分析**

与LLM仅推理、Naïve RAG、InstructRAG、StableRAG、TruthfulRAG 等基线相比，SmartRAG在 Qwen3‑1.7B 量化模型下，TriviaQA 约 50% 正确率，HotpotQA、MultiHopQA 超过 66% 正确率，整体性能接近甚至超过 8B‑级云端模型；

**⚠️ 局限性**

主要限制包括：规划器仍为启发式且不适用于所有问题类型；检索条件预填充导致较长的第一次标记延迟；实体类型识别在某些主题（如 AI、政治）仍不稳健；实验仅覆盖英文基准，跨语言适配待研究。

---

## 322. Understanding of Task-specific and Subject-specific Components in Surface EMG

**arXiv ID:** 2607.14744 | [PDF](https://arxiv.org/pdf/2607.14744v1)

**作者:** Yangyang Yuan `[一作]` (Shanghai Jiao Tong University), Jiahao Fan `[通讯]` (Pennsylvania State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种基于双编码器单解码器的特征解耦模型，提取sEMG信号中的任务特异性和主体特异性成分，用于手势识别和用户识别。

**💡 创新点**

首次将风格迁移的内容-风格分离思想应用于sEMG，构造任务-主体两个正交潜空间，并通过交叉重建、三元组损失实现解耦，同时对解耦特征进行生理学解释。

**🔧 技术方法**

使用卷积自编码器、Instance Normalization、Triplet Loss、Cross‑reconstruction Loss、平移/旋转数据增强、RMS特征提取、t‑SNE可视化以及Silhouette指数评估。

**📊 数据集**

20名受试者、两天共1320次实验，11个单自由度手势，采用前期公开的sEMG数据集（DOI 10.13026/ym7v-bh53）。

**📈 对比分析**

与原始特征、PCA、AE做对比；在intra‑day下，任务特异性组件手势识别达到98.5%+，主体特异性组件用户识别达到99.3%；在inter‑day下分别提升至91.5%和64.7%，显著优于基线且差异显著。

**⚠️ 局限性**

对跨天用户识别仍受电极置换、肌肉状态变化影响；模型仅利用RMS时域特征，缺乏频域信息；对时间动态信息依赖于编码器，LSTM等序列模型效果不佳。

---

## 323. CoTu at EXACT 2026: Neuro-Symbolic Reasoning for Transparent Educational QA

**arXiv ID:** 2607.14735 | [PDF](https://arxiv.org/pdf/2607.14735v1)

**作者:** Quoc-Khang Tran `[一作]` (Can Tho University), Nguyen-Khang Pham `[通讯]` (Can Tho University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 EXACT 2026 透明教育 QA 竞赛中，团队提出一种基于 4B 参数开源 LLM 的 neuro‑symbolic PoT（Program‑of‑Thought）框架，分别为规章查询生成 Z3 约束程序进行符号推理，为物理题生成可执行 Python 代码；两支分支共享自我纠错循环与统一 JSON 输出，满足 60 秒内回答并附带可验证的自然语言解释。

**💡 创新点**

创新点在于将符号推理与程序化推理统一到同一小模型的 PoT 流程中；通过教师模型蒸馏得到的 LoRA adapter 让 4B 基础模型在生成结构化程序时保持高准确率；并采用时间敏感的自我纠错与自适应 fallback，充分利用 60 秒限制下的推理效率。

**🔧 技术方法**

技术手段包括：Qwen3.5‑4B‑based 4B backbone、DSPy 任务编排、SGLang + DFLASH 预取推理、RadixAttention 缓存、Z3 SMT Solver、Python 沙盒执行、教师蒸馏（DeepSeek‑V4‑Pro、Qwen3.6‑27B）和 LoRA 微调。

**📊 数据集**

数据集使用 EXACT 2026 官方数据：逻辑任务 1,055 条（含 808 题 + 247 合成），物理任务 1,452 条（含 1,352 题 + 100 合成），并通过教师模型生成高质量推理轨迹和程序示例。

**📈 对比分析**

在自动评测阶段，物理任务 100% 正确；逻辑任务通过精细微调将答题与前提 F1 从 21.48/25 提升至 24.20/25，最终技术分 13.44/15（最高），整体排名第 3。

**⚠️ 局限性**

局限主要在前提选择误判（P2）占大多数错误；依赖 8B 以上的教师模型进行蒸馏；并未在物理任务中实现符号约束，缺乏统一的可验证性保证。

---

## 324. Causal-Adversarial Probing of Clinical Covariates for Prostate MRI Grading

**arXiv ID:** 2607.14720 | [PDF](https://arxiv.org/pdf/2607.14720v1)

**作者:** Yipei Wang `[一作]` (University College London), Yipeng Hu `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

通过对前列腺MRI分级模型实施对抗性协变量抑制，探测模型对临床变量的依赖性。

**💡 创新点**

采用因果推理框架，将协变量划分为干扰项、疾病相关代理和无关变量，并提出单变量对抗抑制方法评估其对预测性能的影响。

**🔧 技术方法**

使用梯度反转对抗网络，搭载ProFound或ResNet-18编码器，采用交叉熵损失实现ISUP分级和协变量预测。

**📊 数据集**

在UCLH 2,903例训练/验证/测试集以及PROMIS 576例外部验证集上进行实验。

**📈 对比分析**

与仅使用MRI的基线模型对比，计算AUC、balanced accuracy和Sens@80Spec等指标；抑制年龄、BMI、酒精使用可提升AUC 1.0–1.4%，而抑制PSA或前列腺体积则导致AUC显著下降。

**⚠️ 局限性**

仅单独评估每个协变量，未考虑变量交互；将连续变量离散化可能丢失信息；外部验证集受样本选择偏差影响。

---

## 325. Variational Inference for Bird's Eye View Segmentation in Autonomous Driving

**arXiv ID:** 2607.14710 | [PDF](https://arxiv.org/pdf/2607.14710v1)

**作者:** Jingyue Shi `[一作]` (Beijing Jiaotong University), Yanxiang Jiang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于Transformer的变分流生成式BEV分割网络TVB，利用多摄像头信息生成多样化BEV图并通过注意力融合得到最终分割结果。

**💡 创新点**

创新点在于将正则化流引入条件变分自编码器以生成更具表现力的BEV候选图，并设计BEV‑attention融合模块自适应加权多候选图，提升多摄像头环境下的分割鲁棒性。

**🔧 技术方法**

使用技术包括多视角Transformer融合、条件变分自编码器（CVAE）、正则化流（Normalizing Flow）、Monte Carlo采样、BEV‑attention融合与交叉熵+IoU损失。

**📊 数据集**

在公开基准数据集nuScenes和OPV2V上进行评估。

**📈 对比分析**

与VPN、STA、Lift‑Splat、FIERY、CVT、LaRa等方法对比，TVB在nuScenes Setting1 +1.7，Setting2（车辆）+1.4，Setting2（可行区）+1.4的IoU上实现最优；在OPV2V上车辆+3.3、可行区+0.6、车道+2.3的提升。

**⚠️ 局限性**

局限性包括：采样与正则化流步骤导致推理速度仅约19FPS；对细长结构如车道的精度提升有限；需要更高效的采样策略和几何约束以满足实时部署与细节分割需求。

---

## 326. Reinforcement Learning for the Full Strawberry Harvesting Process: Obstacle Separation, Detachment, and Placement

**arXiv ID:** 2607.14708 | [PDF](https://arxiv.org/pdf/2607.14708v1)

**作者:** Changyou Miao `[一作]` (Beijing Academy of Agriculture and Forestry Sciences), Ya Xiong `[通讯]` (Beijing Academy of Agriculture and Forestry Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一个基于强化学习的阶段性协调框架，实现了在严重遮挡下的草莓完整采摘流程（障碍物分离、果实脱离、放置）。

**💡 创新点**

创新点在于将障碍物分离与果实抓取统一为共享的交互感知策略，并通过启发式阶段切换与奖励设计实现零样本 sim-to-real 转移。

**🔧 技术方法**

采用了 PPO 强化学习、YOLOv11+SRR‑Net 视觉感知、Cartesian 难度增量控制、基于 PyBullet 的域随机化仿真与 Cartesian 惯性控制。

**📊 数据集**

使用自行构建的多遮挡草莓仿真环境与实际温室摄像机采集的 RGB 数据，对目标果实和遮挡果实进行检测与分割。

**📈 对比分析**

通过与单一抓取或定位方法的对比，仿真成功率达89.7%，实际场景成功率为82%，且执行时间随遮挡级别从12.99 s增加到21.73 s。

**⚠️ 局限性**

局限性包括：未针对不同品种、季节与栽培系统进行广泛验证；仿真模型未完全捕捉植物非线性变形；仅在单一机器人平台上验证，缺乏多平台泛化评估。

---

## 327. Pretraining Multiple Instance Learning Networks with Multi-Teacher Distillation from Pathology Slide Foundation Models

**arXiv ID:** 2607.14703 | [PDF](https://arxiv.org/pdf/2607.14703v1)

**作者:** Mingxi Fu `[一作]` (Tsinghua University), Yonghong He `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于多教师蒸馏的MIL预训练框架，将TITAN和CARE模型的slide-level知识迁移到多种轻量级MIL聚合器。

**💡 创新点**

创新点在于结合多教师蒸馏与角度分散归一化蒸馏损失，解决不同教师分布差异，并将预训练权重直接作为下游WSI任务的初始化。

**🔧 技术方法**

使用多教师知识蒸馏、角度分散归一化损失、CONCH特征提取、Patch级编码以及多种轻量级MIL聚合器（Attention、Transformer、Graph、State‑Space）。

**📊 数据集**

预训练使用TCGA‑UT‑8K ROI数据，评估采用15个WSI基准（BCNB、BRACS、CPTAC、EBRAINS、KIDRARE、MUT‑HET‑RCC等）共15项任务。

**📈 对比分析**

与从零训练和教师直接微调对比，线性探针与全参数微调评估，预训练在大多数任务上提升2–5%准确率，并在few‑shot下提升6–20%；轻量级模型在速度与内存上更优。

**⚠️ 局限性**

局限在于预训练仅使用两名教师，缺乏更大规模或多模态预训练数据；蒸馏对教师分布差异敏感，且不同MIL架构对蒸馏效果差异明显。

---

## 328. Lights, Camera, Malfunction: When Illumination Robustness Leaves VLA Models Blind to Color

**arXiv ID:** 2607.14698 | [PDF](https://arxiv.org/pdf/2607.14698v1)

**作者:** Marino Watanabe `[一作]` (Keio University), Kentaro Yoshioka `[通讯]` (Keio University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究 Vision‑Language‑Action (VLA) 模型在目标性物理聚光灯攻击下的鲁棒性，并提出 FLARE 攻击框架与 ChromaGuard 防御方案。

**💡 创新点**

创新点：① 开发可优化物理聚光灯攻击框架 FLARE，揭示 VLA 对光照变化的脆弱性；② 发现传统色彩数据增强会导致模型忽略色彩信息；③ 提出色彩保留的 ChromaGuard 增强训练方法，兼顾鲁棒性与色彩识别。

**🔧 技术方法**

技术手段：Bayesian Optimization（TPE）搜索最优灯光参数；HSV 色彩空间数据增强；基于 SmolVLA、π_0.5 等 VLA 架构；LeRobot+LIBERO 仿真与 6‑DoF SO‑101 机器人硬件实验。

**📊 数据集**

使用数据集：LIBERO 多任务操纵基准（Spatial、Object、10）以及在真实世界中收集的颜色无关与颜色相关任务演示数据。

**📈 对比分析**

比较方法与性能：与基线模型和 Naive‑Aug 模型在仿真与实测下对比；在光照攻击下基线 SR 降至 0%，Naive‑Aug 在颜色依赖任务中仅 47.5%；ChromaGuard 在颜色无关任务下保持 ≥70% SR，在颜色依赖任务中达到 97.5%/92.5%，显著优于其他方法。

**⚠️ 局限性**

limitations：仅关注静态光照，未探究时间变化光照或闭环自适应攻击；工具未覆盖所有攻击场景；缺乏对动态环境与其他鲁棒性指标的评估。

---

## 329. Reflex: Real-Time VLA Control through Streaming Inference

**arXiv ID:** 2607.14695 | [PDF](https://arxiv.org/pdf/2607.14695v1)

**作者:** Yuanchun Guo `[一作]` (Beijing University of Posts and Telecommunications), Bingyan Liu `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Reflex 系统，使 Flow Matching 视觉-语言-动作模型能够实现实时流式推理，支持 50Hz 连续控制。

**💡 创新点**

创新点包括：① 利用时间不变性将注意力上下文拆分为静态、滑动与动态区域，实现 O(1) 缓存更新；② 引入 AdaRMSNorm 适应混合精度，防止数值崩溃；③ 构建异步管线和未来状态预测，隐藏视觉编码延迟。

**🔧 技术方法**

主要技术手段包括：分区注意力、增量预填、手动 KV 缓存合并、BFloat16/FP32 混合精度、自适应 Overlap 调度、CUDA 核融合（Query/Key/Value 合并、SwiGLU 级联），以及未来状态预测模块。

**📊 数据集**

在 LIBERO（四类操控任务）和 Kinetix（物理动态任务）两个公开基准上评测。

**📈 对比分析**

与标准同步推理、Naïve 缓存和 Async-Naïve 对比，Reflex 在 LIBERO 上实现 2.58× 的推理加速、0% stall、50Hz 运行，并将系统反应延迟降低 50% 以上；在 Kinetix 上亦显著提升成功率与响应速度。

**⚠️ 局限性**

局限性在于：① 仅适用于时间不变编码器的 Flow Matching 结构；② 需要精确的时间步对齐与缓存管理；③ 未来状态预测为经验性近似，可能在极端动态场景下失效；④ 复杂的多线程和低级 CUDA 优化增加实现难度。

---

## 330. MIND-CAVs: Multi-Intelligence Negotiation and Decision System for CAVs based on Intent-Driven Autonomy

**arXiv ID:** 2607.14688 | [PDF](https://arxiv.org/pdf/2607.14688v1)

**作者:** Mainak Mondal `[一作]` (University of Connecticut), Han Song `[通讯]` (University of Connecticut)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了MIND-CAVs框架，通过结构化意图交换和边缘计算仲裁，实现了多车协同驾驶；

**💡 创新点**

创新点在于：①将高层驾驶意图转化为可机器解析的结构化表示；②在MEC层引入受限延迟的语义仲裁与安全验证；③设计了可审计的决策日志；④将大语言模型用于意图生成和仲裁；

**🔧 技术方法**

技术包括：车端意图生成（VLM辅助）、MEC端语义仲裁与安全验证（VLM+确定性验证）、车云架构（车辆-边缘-云）、CARLA AI‑in‑the‑loop仿真、结构化日志与回放、对比基线（孤立自治、FCFS、MARL）

**📊 数据集**

数据集：CARLA仿真数据，使用Town_04地图，10次相同初始配置，采样10Hz，生成意图、计划、轨迹等日志；未使用真实交通数据

**📈 对比分析**

对比方法：孤立自治（IA）、集中式先到先得（FCFS）、多智能体强化学习（MARL）；在三种情境下，MIND-CAVs完成时间最短、gap违规和不必要制动次数最少，且方差更小，显示出效率和安全双重提升

**⚠️ 局限性**

局限性：仅在结构化高速公路场景、车辆数量有限；未考虑复杂城市交通、行人、交通信号；MEC采用固定50km/h上限以限制推理频率；仲裁决策基于LLM，缺乏形式化安全保证

---

## 331. InCarEmo: A Multimodal Dataset for In-Cabin Emotion Recognition and Driver State Monitoring

**arXiv ID:** 2607.14683 | [PDF](https://arxiv.org/pdf/2607.14683v1)

**作者:** Hao Yang `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 InCarEmo 多模态数据集，并提出轻量级基线模型 CAMEL，用于车舱情绪识别、疲劳检测和分心监测。

**💡 创新点**

创新点在于：① 将 RGB、红外、音频和对话文本整合为同一多模态数据集；② 设计英文跨语言基准，支持多语言研究；③ 通过跨模态对比学习和多分类器投票实现高效融合；④ 对缺失模态和噪声场景下的鲁棒性进行系统评估。

**🔧 技术方法**

技术手段包括：CLIP（RGB/IR）、Chinese‑HuBERT（音频）、Qwen3（文本）等预训练模型；跨模态对比损失对齐多模态特征；多分类器投票策略；MediaPipe FaceMesh 计算眼部闭合率、嘴部比例用于疲劳检测；红外热成像特征提取。

**📊 数据集**

使用 InCarEmo 数据集（中文原始 + 英文翻译合成），涵盖 3.6K 情绪视频、230 条疲劳样本、270 条分心样本；以及从该数据集衍生的英文基准。

**📈 对比分析**

对比方法：单模态基准、开源多模态大模型（GPT‑4o、Qwen2.5‑VL、Emotion‑LLaMA 等），以及在 InCarEmo 上微调的版本。CAMEL 在情绪识别上取得 82.25% 准确率，超越 GPT‑4o 的 75.58%；疲劳检测 84.58% 准确率；分心检测 81.24% 准确率；在英文基准中性能略下降但趋势一致。

**⚠️ 局限性**

局限性：数据主要来自中文受试者，跨语言迁移效果仍有待提升；脚本化场景缺乏自发对话的自然性；在极低光照或高噪声环境下仍存在性能衰减；数据规模相对有限，可能限制模型的泛化。

---

## 332. Large Audio Language Models for Spoofing-Aware Speaker Verification

**arXiv ID:** 2607.14753 | [PDF](https://arxiv.org/pdf/2607.14753v1)

**作者:** Sofya Savelyeva `[一作]` (Applied AI Institute), Oleg Y. Rogov `[通讯]` (AXXX, Applied AI Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究如何将大型音频语言模型（LALM）迁移到欺骗感知说话人验证（SASV）任务中，并系统评估多种微调与训练策略。

**💡 创新点**

创新点在于：①提出将LALM直接用于三类决策（目标、非目标、伪造）而非传统分离式CM‑ASV融合；②通过联合损失（AAM、二元交叉熵）和硬样本挖掘实现说话人辨别与伪造检测的平衡；③引入链式思考(CoT)和GRPO优化，尝试让模型产生可解释的判定理由。

**🔧 技术方法**

核心技术包括：LoRA 参数高效微调、AAM 说话人头、二元交叉熵伪造头、硬样本挖掘、Chain‑of‑Thought 训练与GRPO奖励优化，以及 QFormer 对齐的多音频输入。

**📊 数据集**

使用数据集包括 ASVspoof‑5（训练与评估）、VoxCeleb（增强真声样本）以及在实验中采样的 1.8M 对（目标/非目标/伪造）对；在推理训练中使用 90K 对带 CoT 轨迹的数据。

**📈 对比分析**

与传统的 ECAPA‑2 + WavLM 或 ECAPA‑2 + Wav2Vec2‑AASIST 融合基线相比，LoRA‑微调的 LALM 在整体准确率约 86%（相较 80%）和 min‑a‑DCF 上表现更好，尽管 EER 与阈值指标仍略逊于基线，但在统一端到端决策方面取得优势。

**⚠️ 局限性**

主要局限包括：零样本时性能几乎随机；需要大量算力（LALM 7B/10B）；Chain‑of‑Thought 轨迹未得到系统验证；在阈值指标（EER、min‑a‑DCF）上仍不如传统融合方法，且解释性训练未显著提升整体准确率。

---

## 333. The Misclassification of Autistic Writing as AI-Generated

**arXiv ID:** 2607.14729 | [PDF](https://arxiv.org/pdf/2607.14729v1)

**作者:** Summer Chambers `[一作]` (George Mason University), Matthew C. Kelley `[通讯]` (George Mason University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Reddit上自闭症相关和普通子版块的帖子进行AI生成文本检测模型的偏差研究，检验自闭症作者的文本更易被误判为AI生成。

**💡 创新点**

首次量化并比较自闭症作者与普通作者在公开GPT‑2检测器中的误报率差异，揭示潜在的歧视风险。

**🔧 技术方法**

使用OpenAI RoBERTa GPT‑2检测器、Logistic回归分析、Perplexity与Burstiness等文本特征计算等技术。

**📊 数据集**

约6万条Reddit帖子，分为可能自闭症（autism‑centric subreddits）与一般Reddit两组，经过过滤、长度标准化后构成数据集。

**📈 对比分析**

通过Logistic回归比较两组文本在检测器输出概率上的差异，结果显示自闭症组误报率比一般组高约25–50%，整体误报率低于2%，且短文本更易被误判。

**⚠️ 局限性**

数据来源公开平台且自闭症标签不确定、样本多为英语、仅测试单一检测器、短文本和语料多样性限制，导致结论可能受限。

---

## 334. Does generative AI supersede supervised XMLC? A Benchmark Study on Automated Subject Indexing with German Scientific Literature

**arXiv ID:** 2607.14882 | [PDF](https://arxiv.org/pdf/2607.14882v1)

**作者:** Maximilian Kähler `[一作]` (Deutsche Nationalbibliothek), Markus Schumacher `[通讯]` (Deutsche Nationalbibliothek)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于德国国家图书馆的GND词表，对德语科学文献的自动主题索引进行极端多标签分类基准实验，比较传统、监督XMLC、嵌入匹配以及LLM生成方法；

**💡 创新点**

首次在纯德语数据上对多标签分类方法进行系统基准，并同时评估二元与分级相关性，揭示LLM在长尾标签上的优势；

**🔧 技术方法**

采用DiSMEC++、Omikuji、ZestXML、AttentionXML、XR-Transformer、NGAME等监督XMLC模型，Lexical Matching、EBM、LLM-Ensemble、KI-FSPrompt等匹配与生成方法，结合Transformer、BERT、sentence‑embedding、LLM等技术；

**📊 数据集**

使用德国国家图书馆收集的德语科学教材数据，包含两类任务：书名索引（Book‑Titles）和前30k字符全文索引（Fulltext‑30k），共计4,651个测试文档，GND子集204,056标签；

**📈 对比分析**

通过精度-召回曲线AUC、最佳F1、条件加权指标以及专家分级评估；结果显示监督XMLC中的XR‑Transformer在二元相关性上表现最佳，LLM‑Ensemble在分级相关性和长尾标签性能上优于其他方法；

**⚠️ 局限性**

实验仅限德语科学文献，未覆盖全文长文及其他领域，LLM方法资源成本高；标签类别同一处理导致实体类型差异未考虑；训练/推理时间评估受硬件变化影响，且仅部分方法被分级评估；

---

## 335. Asymmetric Peak-Aware Loss for Peak-Critical Time Series Forecasting

**arXiv ID:** 2607.14871 | [PDF](https://arxiv.org/pdf/2607.14871v1)

**作者:** Theivaprakasham Hari `[一作]` (Delft University Of Technology), Sascha Hoogendoorn-Lanser `[通讯]` (Delft University Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估一种称为 Asymmetric Peak-Aware Loss (APAL) 的点预测损失函数，用于提升长周期多变量时间序列在极端峰值预测上的准确性，并配套峰值关键评价指标。

**💡 创新点**

创新点在于将低估偏差惩罚与峰值区域加权相结合，形成可解释的三参数模型无关损失，并提出兼顾尾部误差与峰值检测的评估协议，填补传统对平均误差忽视峰值的空白。

**🔧 技术方法**

技术核心包括：基于绝对误差的加权损失（含 λ_u、λ_p 与阈值 τ），自适应峰值掩模，和峰值检测/定位评价（Top‑10%/Top‑1% MSE、Peak F1 与 PTE）；实验使用 PyTorch/TensorFlow 框架实现并与多种基准网络（DLinear、TSMixer、PatchTST、TiDE、iTransformer）对比。

**📊 数据集**

数据集覆盖两类人流计数（Melbourne Pedestrian、Beach Visitor）以及公共多变量时序基准（ETT、Electricity、Traffic、Weather、Exchange），并在每个数据集上做多步长（96、192、336、720）实验。

**📈 对比分析**

与对称损失（MSE/MAE）及其他结构化损失（TILDE‑Q、DBLoss、PS）相比，APAL 在 39/40 的峰值关键设置中取得最低的 Top‑1% MSE，并在 Peak F1 方面提升至 88% 以上，平均增加 4% 计算开销；但在非峰值显著的数据集上可能导致整体 MSE 上升。

**⚠️ 局限性**

局限性包括：需人工调参（λ_u、λ_p、τ）以平衡平均误差与峰值性能；在峰值不明显或噪声多的窗口中可能产生误报或过度上偏；对递归预测或峰值时序漂移的鲁棒性尚未验证。

---

## 336. Randomized routing strategies of fleets of CAVs may prove market efficient

**arXiv ID:** 2607.14859 | [PDF](https://arxiv.org/pdf/2607.14859v1)

**作者:** Grzegorz Jamróz `[一作]` (Jagiellonian University), Rafał Kucharski `[通讯]` (Jagiellonian University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个基准实验框架，用来比较在未来城市中不同CAV运营商的路由算法对市场份额与整体通行时间的影响，并针对如何将市场份额最大化与城市通行效率兼顾提出了新的目标函数；

**💡 创新点**

创新点在于引入随机化路由策略（RFlexV、RFlex）以提升运营商市场份额，同时通过在目标函数中加入系统最优通行时间项来抑制过度随机化、鼓励社会福利导向的竞争；

**🔧 技术方法**

主要技术包括基于BPR函数的交通拥堵模拟、基于logit的模式与路线路径选择模型、动态随机路由算法、以及目标函数权衡市场份额与平均通行时间的调节；

**📊 数据集**

使用的是仿真数据：200名司机、300天模拟周期、两条等效路线、Gaussian分布的折扣因子、以及BPR式延迟函数；

**📈 对比分析**

比较方法是对不同路由算法（SO、UE、RFlexV、RFlex）在相同仿真环境下测量市场份额、平均通行时间以及组合目标值。结果显示：随机化算法在仅追求市场份额时表现最佳，但平均通行时间波动更大；随着目标函数中平均通行时间权重μ的增加，SO类算法逐渐占优，随机化策略被抑制；

**⚠️ 局限性**

局限性包括仅考虑两条路段、两家运营商、已知司机折扣因子，缺乏更大规模、更多路线或不完全信息下的验证，且实验基于简化的BPR模型，未在真实城市交通模拟器（如SUMO）中验证。

---

## 337. Spoofer or Spoofers? Estimating a Lower Bound on the Number of DRDoS Sources Using Anycast Honeypots

**arXiv ID:** 2607.14832 | [PDF](https://arxiv.org/pdf/2607.14832v1)

**作者:** Bernhard Degen `[一作]`, Raffaele Sommese `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在全球32个节点部署任意播放大蜜罐，收集DRDoS攻击请求的TTL值，利用TTL的路径长度信息估计攻击中伪装源的下界。

**💡 创新点**

首次将任意播的空间分布与TTL波动阈值相结合，提出两种贪心区间覆盖算法（最小区间覆盖和频率加权区间覆盖）来量化多源伪装比例，显示21%攻击涉及多源。

**🔧 技术方法**

使用任意播放大蜜罐(AmpPot)、RIPE Atlas探针测量TTL波动、TTL稳定性阈值校准、贪心区间覆盖算法和频率加权覆盖算法。

**📊 数据集**

32节点任意播蜜罐在两段时间内共收集5.49B/19.3B请求，形成843K/2.93M攻击；使用12502个RIPE Atlas探针的TTL测量数据作为阈值校准。

**📈 对比分析**

通过仿真攻击评估两估计器，最小覆盖永不超估，频率加权覆盖仅在0.007%情况下超估；在真实攻击中频率加权估计显示21%多源攻击，远高于以往3.7%的报告。

**⚠️ 局限性**

只能给出下界且不精确计数；TTL变异、攻击者TTL操控、任意播收敛/多播等因素可能导致漏估或误估；对不同协议、路径动态变化的鲁棒性有限。

---

## 338. AI Prototyper: A Figma Plugin for Decomposition-Based GUI Prototyping with LLMs

**arXiv ID:** 2607.14830 | [PDF](https://arxiv.org/pdf/2607.14830v1)

**作者:** Tawatchai Salangsingha `[一作]` (Edinburgh Napier University), Iain McGregor `[通讯]` (Edinburgh Napier University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 AI Prototyper，一个 Figma 插件，利用分解与检索增强生成（RAG）自动化生成可编辑的 GUI 原型，并支持多语言输入。

**💡 创新点**

创新点包括：1）引入人机交互式编辑步骤，让用户在生成前可直接审阅、修改功能列表；2）使用 Gemini 2.5 Flash LLM 作为核心生成模型；3）构建专门的 32 组件库并以 JSON schema 规范化；4）支持泰语、英语、中文、马来亚语等多语言标签的自动生成。

**🔧 技术方法**

采用 JavaScript/Node.js 后端、Figma API、Gemini 2.5 Flash LLM、两阶段 RAG（先选择再实例化）、JSON schema 组件定义、Auto‑Layout 渲染技术。

**📊 数据集**

使用自定义 32 组件的 JSON 组件库（按移动 UI 常见模式整理）进行评估；实验样本包括 11 名大学生（生产力测试）和 10 名专业 UI 评审（质量评估），并在不同语言（泰语、英语、中文、马来亚语）下进行多语言探测。

**📈 对比分析**

采用两阶段实验：第一阶段生产力对比（AI 组 100% 完成 4 任务，手工组 54%），使用 Mann‑Whitney 检验 p=0.016；第二阶段专家评审（9 维度 9 分制），AI 原型在所有维度均显著更好（p<0.001），效果大小 r=0.615–0.866。

**⚠️ 局限性**

局限性：样本量小、仅学生与专家评审、未测量生成后编辑时间、组件库缺乏样式/主题自定义、未实现页面交互链接、低资源语言表现差、评审非独立导致 p 值可能被放大。

---

## 339. Can LLMs Build a MaxSAT Solver from Papers? The CoreForge Experience

**arXiv ID:** 2607.14818 | [PDF](https://arxiv.org/pdf/2607.14818v1)

**作者:** Ruben Martins `[一作]` `[通讯]` (Carnegie Mellon University), Ruben Martins (Carnegie Mellon University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过LLM辅助从研究论文生成非加权MaxSAT求解器，并实现多种无不满足性基算法与优化后端。

**💡 创新点**

将LLM论文到代码流程与多轮审计、基准测试相结合，首次从零开始构建MaxSAT求解器而非改造现有代码。

**🔧 技术方法**

使用大语言模型（ChatGPT/Codex）生成代码，LLM辅助审计，模糊测试，MaxSAT Evaluation benchmark，以及SCIP和CP-SAT整数线性优化后端。

**📊 数据集**

使用MaxSAT Evaluation 2022/2023 及自定义 fuzzing 实例进行测试。

**📈 对比分析**

与现有顶尖手工调优求解器比较，取得多款求解器的领先或相当成绩，但总体性能仍低于最强手工实现。

**⚠️ 局限性**

需要人工干预、外部验证与基准，LLM对低层优化不敏感，自动化程度有限，且无法保证实现的算法与论文完全一致。

---

## 340. CrimeNER Demo: Named-Entity Recognition in the Crime Domain

**arXiv ID:** 2607.14800 | [PDF](https://arxiv.org/pdf/2607.14800v1)

**作者:** Miguel Lopez-Duran `[一作]` (Universidad Autónoma de Madrid), Alvaro Ortigosa `[通讯]` (Universidad Autónoma de Madrid)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CrimeNER Demo 平台，能够自动识别并标注犯罪相关文件中的粗细两级实体。

**💡 创新点**

创新点在于构建专门的犯罪实体层次化数据集（CrimeNER-db），并支持用户上传自定义标注进行微调，实现对特定案件的快速适配。

**🔧 技术方法**

采用多种预训练语言模型（XLM‑RoBERTa、DeBERTa‑V3、RoBERTa、AlBERT、DistilBERT、BERT）进行 NER，并实现分层实体抽取与后处理。

**📊 数据集**

使用 CrimeNER-db，包含约 1.5K 篇真实案例文件（如恐怖主义报告、媒体报道）进行训练和评估。

**📈 对比分析**

在 CrimeNER-db 上做了严格与灵活评估，最高的粗细实体 F1 约为 0.90（XLM‑RoBERTa）和 0.89（DeBERTa‑V3），展示了模型在双层实体识别上的优异性能。

**⚠️ 局限性**

限制包括 PDF 文本位置偏差导致标注不准、仅覆盖英文及少数犯罪类型、需要用户手动准备符合格式的标注数据。

---

## 341. Ground-Side Mission Plan Compilation with Policy-as-Code Guardrails for Cloud-Native Satellite Platforms

**arXiv ID:** 2607.14798 | [PDF](https://arxiv.org/pdf/2607.14798v1)

**作者:** Hsiu-Chi Tsai `[一作]` (National Yang Ming Chiao Tung University), Chia-Tung Chung `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个面向卫星任务计划的地面编译器，完成了从 YAML 计划到 Argo Workflow 与 Kueue Job 的模式化验证与渲染，弥补了 ORCHIDE 等云原生卫星平台对地面工具链的缺口。

**💡 创新点**

创新点在于将 OPA/Rego 作为安全可审计的“策略即代码”层与基于 ORCHIDE 公开规范的 Pydantic 模式结合，形成四层防御式校验（模式、策略、静态 lint、集群准入）并支持 GPU/CPU 动态资源分配与降级，同时提供 MCP 接口以供 AI 代理调用。

**🔧 技术方法**

主要技术包括：Python 3.12 + Pydantic v2 进行模式校验；Open Policy Agent (OPA) + Rego 进行策略评估；Argo Workflows 作为 DAG 执行引擎；Kueue 的 ClusterQueue/LocalQueue + DRA 进行资源排队与 GPU 绑定；FastMCP 及自定义 MCP 工具实现对接；Kubernetes 1.35–1.36 及 NVIDIA DRA 驱动做实验验证。

**📊 数据集**

使用了基于 ORCHIDE 公开规范构造的模拟任务计划（最多 1000 个获取事件，每个事件 3 步），以及自制的失效案例（结构错误、语义错误等）作为评估数据集；未使用真实卫星运营商的计划文件。

**📈 对比分析**

通过对比 OPA 子进程与内置 Python 评估基线，验证 OPA 在单个调用约 17–120 ms，整体编译时间从 34 ms（10 事件）线性增长到 1.5 s（1000 事件），解析占比从 42 % 提升到 87 %；在单节点 K8s+Argo+Kueue 集群上进行现场验证，证明 GPU/DRA 及资源降级路径均按预期工作。

**⚠️ 局限性**

局限包括：仅支持 GPU 及 CPU 的单类 DRA；缺乏 FPGA、混合加速器及多类 DRA 的策略与实现；未在真实卫星运营商的计划上验证；不具备辐射硬化、高可用、OTA 更新或 NIST 800‑53 级别的安全防护；对资源降级的实现仍以运行时环境变量或调度器优先级为主，尚未覆盖所有可能的边缘情形。

---

## 342. Transcoders for Investigating Deception in Language Models

**arXiv ID:** 2607.14791 | [PDF](https://arxiv.org/pdf/2607.14791v1)

**作者:** Darius Lim `[一作]` (Home Team Science and Technology Agency), Xin Wei Chia `[通讯]` (Home Team Science and Technology Agency)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用预训练的 per‑layer transcoders (PLTs) 对 Qwen3‑4B 进行机制解释，构建归因图并识别与欺骗行为相关的特征集合；

**💡 创新点**

首次将 transcoders 应用于安全与欺骗行为的电路级分析，揭示特征之间的交互网络并验证特征导向可精准控制欺骗输出；

**🔧 技术方法**

核心技术为 per‑layer transcoders 与归因图构建、特征导向 (feature steering)、电路图分析与统计显著性检验；

**📊 数据集**

采用 100 条含秘密关键字的欺骗诱导 prompt 及 Qwen3‑4B 模型；

**📈 对比分析**

与随机特征对照组相比，顶级 10 个欺骗特征的正向导向导致 21% 普通 prompt 变为欺骗性，负向导向可完全消除欺骗；电路级特征（“Obscuring information”“secrets/confidentiality”）在正负两方向上显著提升效果（p < 0.001）；

**⚠️ 局限性**

研究仅在单一模型 Qwen3‑4B 上验证，使用已有预训练 transcoders，缺乏跨模型/多 alpha 值的稳健性验证，且特征筛选过程中可能存在人为偏差。

---

## 343. ChronoQG: Towards a Temporally Expressive and Hop-Bounded Benchmark for Temporal Knowledge Graph Question Generation

**arXiv ID:** 2607.14770 | [PDF](https://arxiv.org/pdf/2607.14770v1)

**作者:** Xuemeng Liu `[一作]` (Nankai University), Wentao Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套面向时序知识图的问答生成基准（ChronoQG），并从两种时序知识图（CronKG 与 EventKG）生成了四个验证数据集，总计 16,011 道高质量问题。

**💡 创新点**

提出了完整的时间约束词汇表、拓扑‑时间子图采样策略以及基于验证器的问句重写流程，确保生成的问句在结构与时间约束上都保持真实、无信息泄漏。

**🔧 技术方法**

采用了 Allen 区间代数与点代数融合的时间约束分类、候选空间演进的采样方法、LLM（GPT‑4o‑mini 等）重写与验证器交互、以及模板‑推理链技术来实现问题生成与质量保证。

**📊 数据集**

使用了 CronKG（年级时间标注）和 EventKG（日级时间标注）两种时序知识图作为原始数据源，生成了 ChronoQG‑Cron‑S/M 与 ChronoQG‑Event‑S/M 两对数据集。

**📈 对比分析**

通过与 LLM 提示、全训练 KGQG 方法和改造的静态 KGQG 基线对比，使用 BLEU‑4、ROUGE‑2/‑L 等指标评估，结果显示多约束、多跳场景下性能显著下降，现有方法仍有较大提升空间。

**⚠️ 局限性**

存在的局限包括：当前模型在细粒度时间语义与约束绑定上效果差，尤其在多约束和区间边界关系时易出现信息丢失或错误；多阶段方法成本高且未显著提升性能。

---

## 344. Rare Concept Generation via Counterfactual Inference in Diffusion Models

**arXiv ID:** 2607.14765 | [PDF](https://arxiv.org/pdf/2607.14765v1)

**作者:** Zhengyuan Jiang `[一作]` (Hefei University of Technology), Yang Wang `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于因果推断的扩散模型（CI-Diff），用于生成罕见概念图像，并通过解耦罕见属性提升生成质量。

**💡 创新点**

首次将因果推断（自然直接效应）引入文本到图像生成，利用罕见与常见概念的对比实现属性解耦；改进CFG机制并设计Temporal Morphological Fidelity Anchoring（TMFA）边缘约束策略。

**🔧 技术方法**

使用反事实因果推断、自然直接效应、重构的Classifier-Free Guidance、Canny边缘提取、背景去除、TMFA时间窗注入等技术。

**📊 数据集**

在RareBench基准上进行评估，涵盖单物体与多物体罕见概念，并扩展至风格与场景任务。

**📈 对比分析**

与SD1.5、SDXL、RealVisXL、PixArt-α、Flux、SD3.0、SD3.5、SynGen、RPG、R2F等多种模型对比，采用CLIP‑T、HPSv2、LLM分数和用户研究指标；CI‑Diff在所有指标上均优于基线，尤其在罕见属性表达和形状保持方面表现突出。

**⚠️ 局限性**

仍受基础模型共识知识偏差限制，对s_rare和TMFA时间窗参数需手动调优；在极端稀有属性或复杂组合时效果可能不稳定；未在多语言或非中文文本上进行系统验证。

---

## 345. FlowGuard: From Signals to Evidence for MCP Security Detection

**arXiv ID:** 2607.14754 | [PDF](https://arxiv.org/pdf/2607.14754v1)

**作者:** Baichao An `[一作]` (Fudan University), Mengying Wu `[通讯]` (Fudan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于执行证据的 Model Context Protocol (MCP) 安全检测框架，结合语义风险分层、Recon‑guided 负载缩小、模式验证探测、证据裁决和历史引导精炼，实现对 MCP 交互中工具元数据与运行结果的全流程检测。

**💡 创新点**

创新点在于将语义风险三元组与实时执行证据相结合，构建结构化验证循环，利用 Recon‑Stage 先行探测后定向投射，并通过证据归因与行为期望双维裁决，显著提升执行路径漏洞检测准确率并降低误报。

**🔧 技术方法**

采用 LLM 辅助的语义三元组推理、JSON Schema 约束下的结构化探测、Recon‑Guided Payload Narrowing、Evidence Adjudication 与迭代式历史反馈循环，并在后台实现低影响探测与响应归因。

**📊 数据集**

在 1,880 条可执行 MCP 基准（涵盖命令注入、工具中毒、提示注入、凭证泄漏、文件系统访问）以及 8,000 条 MCPZoo 真实服务器上进行评估。

**📈 对比分析**

与三种基线（静态分析 MCPScan、元数据审计 MCP‑Scanner、动态交互 A.I.G）对比，F1 在命令注入与文件系统访问上分别达 0.879 与 0.942，误报率显著下降，平均探测延迟比 A.I.G 低 2.23 倍，探测效率提升显著。

**⚠️ 局限性**

局限在于仅为黑盒扫描，无法观察内部状态；探测侧重单次交互，无法应对自适应隐藏或跨会话行为；对 LLM 语义推理的依赖可能受模型能力限制；探测预算有限时可能漏检；未覆盖需长时间或多步交互的安全风险。

---

## 346. Rotational Motion-Induced Error Compensation for Phase-Shifting Profilometry-Based Eye Reconstruction

**arXiv ID:** 2607.14876 | [PDF](https://arxiv.org/pdf/2607.14876v1)

**作者:** Seong-Jin An `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出一种针对相位移剖面测量（PSP）进行眼球三维重建的旋转运动误差补偿算法。

**💡 创新点**

创新点在于：1）利用无编码器的图像特征估计用户特定的三维眼球模型并以球坐标域表达旋转运动；2）对相机像素误差和相位移误差进行联合补偿；3）引入区域化优化分别调节虹膜与巩膜的补偿强度，提高低帧率下的鲁棒性。

**🔧 技术方法**

主要技术包括三步相位移剖面测量、球坐标下的眼球运动估计、相机像素误差与相位移误差补偿、几何约束的相位展开以及区域化优化。

**📊 数据集**

实验使用伪眼球在伺服电机下进行高达700°/s的旋转，并在500 Hz下采集PSP图像；此外还在不同有效帧率（500、400、300、200、125、100、60 fps）以及非球形刚性物体上验证。

**📈 对比分析**

与传统三步PSP基线、仅补偿和补偿+区域化优化三种方法对比，采用ICP对准静态地面真实模型后，RMSE从0.366 mm降低到0.2136 mm（平均降低41.6%），在60 fps时补偿+区域化优化仍显著优于基线。

**⚠️ 局限性**

主要限制包括：未包含角膜折射、泪膜反射、眼睑遮挡等真实眼球因素；假设眼球中心固定，无法应对头部转动或设备滑移；在极低帧率下仍存在残留纹波和尖锐边界误差。

---

## 347. RW-Voice-EQ Bench: A Real World Benchmark for Evaluating Voice AI Systems

**arXiv ID:** 2607.14846 | [PDF](https://arxiv.org/pdf/2607.14846v1)

**作者:** David Ayllon `[一作]` (Hume Ai Research), Panagiotis Tzirakis `[通讯]` (Hume Ai Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布了 Real World Voice EQ Bench，一个针对语音 AI 四大领域（TTS、STS、SU、ASR）的多维度评测框架，强调语音中的语言内容与声学线索的分离。

**💡 创新点**

创新点包括：① 将语音 AI 评测从单一“自然度”转为多维度能力档案；② 在评测设计中加入真实世界的音频干扰、情绪、口音、对话压力等条件；③ 将人类评估与自动评判（语音语言模型）相结合，验证后者的可靠性。

**🔧 技术方法**

使用的技术包括：人类主观打分（5 点 Likert），混合效应模型和因子得分；自动化标注与评分（WER、精确匹配、强制选择准确率、语音真实性评分）；语音语言模型（Gemini、OpenAI GPT Audio、Kimi Audio 等）做为自动评判；语音特征提取与声学模型（TitaNet、ECAPA‑TDNN、WavLM）用于说话人匹配；以及对话生成系统的完整端到端评估。

**📊 数据集**

使用的数据集涵盖：① TTS 任务 30 类维度（角色拟合、表达、身份、语言稳定、可靠性、长文本稳定、音质）共 39,460 条生成；② STS 场景 50 条多情境对话；③ SU 任务 1-5 音频片段，情感标注来自大规模情感语料库，合成与真实对比；④ ASR 四大真实世界数据集：口音（23 类）、情绪（15 类）、背景噪声/音乐、自然对话（Hume‑DaiKon）共约 17 小时。

**📈 对比分析**

比较方法：对每个维度或子任务分别统计人类打分或自动指标（WER、准确率），并对不同模型进行 top‑5 排名。结果显示：不同模型在不同维度有强弱分布；例如 Gemini 3.1 Flash 在 STS 许多维度居首，但在 ASR 的情绪、口音维度并非最佳；OpenAI Scribe 在 ASR 4 条维度中排名第一；TTS 领域无单一模型统治所有维度，Fish Audio S2 在身份、语言稳定等维度表现突出。整体性能说明：模型在一个维度表现优秀并不等价于在其他维度也优秀。

**⚠️ 局限性**

局限性：① 评测仅覆盖英语；② 依赖人工评估的成本与主观性；③ 评测维度虽多但仍未涵盖所有可能的语音交互细节（如多模态交互、低资源语言等）；④ 语音语言模型评判的可靠性仍有限，尤其在无明确标注任务上表现不佳；⑤ 公开数据集与评测方法在未来模型快速迭代时可能需要频繁更新。

---

## 348. Physics-Informed Diffusion for Biomechanically Plausible 3D Sign Language Generation

**arXiv ID:** 2607.14836 | [PDF](https://arxiv.org/pdf/2607.14836v1)

**作者:** Emanuele Colonna `[一作]` (University of Bari Aldo Moro), Giovanna Castellano `[通讯]` (University of Bari Aldo Moro)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种物理信息扩散模型 PIDiffSign，用于从手语符号序列生成 3‑D 骨骼动作，并通过几何校正保证骨长与关节角度的合理性。

**💡 创新点**

创新点在于：① 在扩散网络内嵌入可微几何修正器和多项物理约束损失；② 用 InfoNCE 对齐手语编码与姿态解码器，提升语义一致性；③ 结合 Classifier‑Free Guidance 在推理阶段强化条件性。

**🔧 技术方法**

使用技术包括 Transformer 编码‑解码器（AdaLN‑Zero 时序条件）、DDPM/ DDIM 采样、可微几何修正（骨长纠正 + Rodrigues 角度裁剪）、InfoNCE 对齐、Classifier‑Free Guidance。

**📊 数据集**

实验数据集：PHOENIX14T（德语手语）和 CSL‑Daily（中文手语）。

**📈 对比分析**

与无物理约束 DDPM 基线以及公开的 SLP 方法进行对比；在 DTW、MPJAE、BLEU‑4、WER 等指标上均优于基线，尤其在分布现实度（FID）和关节角度准确度（MPJAE）方面显著提升。

**⚠️ 局限性**

局限性：仅建模上半身骨骼，未覆盖面部、口腔表情等非手势要素；训练数据来自单签者，缺乏多签者变异；推理时需双次前向传播导致计算延迟。

---

## 349. Lossy compression of weighted graph adjacency matrices by transform coding

**arXiv ID:** 2607.14834 | [PDF](https://arxiv.org/pdf/2607.14834v1)

**作者:** Kenta Yanagiya `[一作]` (University of Osaka), Antonio Ortega `[通讯]` (University of Southern California)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种加权图的压缩框架，其中图的拓扑结构无损传输，边权重则采用有损压缩。

**💡 创新点**

创新点在于将无权图转换为对应的线图，并在此基础上对边权重进行有损压缩，同时引入边平滑度作为压缩难度的度量。

**🔧 技术方法**

使用了图滤波器组对边权重进行变换和压缩，并进行了量化和熵编码。

**📊 数据集**

使用了合成数据和真实世界的数据集进行实验，包括随机传感器图、交通网络和电力网等。

**📈 对比分析**

与现有的矩阵预处理方法相比，提出的方法在不同的边平滑度水平下表现出更好的重建准确性，且在真实世界图上也优于其他方法。

**⚠️ 局限性**

限制在于边权重的平滑性假设可能不适用于所有类型的图，且在某些情况下可能需要较高的计算成本。

---

## 350. Interventional Causal Circuits for Safe Robot Action Testing and Failure Recovery

**arXiv ID:** 2607.14826 | [PDF](https://arxiv.org/pdf/2607.14826v1)

**作者:** Naren Vasantakumaar `[一作]` (University of Bremen), Michael Beetz `[通讯]` (University of Bremen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种闭环因果诊断框架，在机器人动作测试过程中，当测试失败时利用因果电路即时诊断并纠正原因变量，提升安全性和效率。

**💡 创新点**

创新点在于将联合概率树（JPT）扩展为满足边际确定性的因果电路，允许在无需额外训练或仿真回合的情况下进行可精确计算的干预查询，从而实现一键式纠错与离散支持检测。

**🔧 技术方法**

核心技术包括：联合概率树、边际确定性变树（MdVtree）构造、支持确定性验证、后门调整公式实现干预分布计算、以及基于因果诊断的参数修正策略。

**📊 数据集**

使用了1,742条成功执行记录的ROS2仿真环境下的乳盒搬运任务数据集，包含五个连续参数（柜台和桌子接近位置、臂选择等）。

**📈 对比分析**

与仅使用JPT的盲重采样基线相比，在高质量JPT下成功率均为100%，但因果电路将失败尝试和恢复时间分别降低10.3%与约41%；在降质JPT下成功率从99%提升至100%，失败尝试减少37%，恢复时间缩短至约45%。

**⚠️ 局限性**

局限性包括仅在单一搬运任务上验证，未测试更复杂的多步操作、接触动力学或真实机器人环境；此外因果电路依赖于训练数据的支持，离散支持外的情况仍需进一步处理。

---

## 351. Multi-Scale Equilibrium under Variable Indicator Dimensionality: Faithful Reduction of Dynamic Attractors in Urban Mobility Systems

**arXiv ID:** 2607.14815 | [PDF](https://arxiv.org/pdf/2607.14815v1)

**作者:** Ali Ghoroghi `[一作]` (Cardiff University), Andrei Hodorog `[通讯]` (Cardiff University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种针对城市交通系统的多层平衡吸引子框架，并研究在仅观测部分指标时如何保证该吸引子在稳定性、固定点以及抗脆弱性决策规则上的完整性；

**💡 创新点**

创新点在于给出了同时满足动力学稳定、两层固定点可压缩性和统计决策功效的四项可投影条件，并将这些条件与测量模型结合，提供可计算的误差界和信息量阈值；

**🔧 技术方法**

主要技术包括动力系统中心流形与慢快分解、层级网络聚合、复合指标敏感性分析、Fisher信息计算与最优实验设计、以及对一阶线性化的特征值与拉格朗日对数范数分析；

**📊 数据集**

使用了十二维状态向量的合成指标以及其对应的观察指标表（表格1），并在三种风格化城市配置（Bratislava、Larissa、Thessaloniki）上进行仿真；

**📈 对比分析**

通过仿真验证了理论结果：误差上界始终成立、恢复时间估计从不超过真实值、决策功效随指标数下降而递减、迭代次数随收敛边界逼近呈指数增长；

**⚠️ 局限性**

主要限制包括仅局部线性假设、对测量噪声和载荷矩阵已知的假设、对时间相关数据的近似处理、以及仅在仿真场景中验证，未对真实城市数据进行实证检验。

---

## 352. Curvature-Constrained and Constant-Speed Distributed Simultaneous Arrival Control for Multi-Robot Systems

**arXiv ID:** 2607.14781 | [PDF](https://arxiv.org/pdf/2607.14781v1)

**作者:** Zhouru Xiao `[一作]` (Hunan University), Yaonan Wang `[通讯]` (Hunan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对多机器人系统，在满足曲率约束和常速约束的前提下，设计了一种分布式控制方法，使得所有机器人能够在相同时间到达预定目标点。

**💡 创新点**

创新点在于：① 将Dubins路径几何性质与最大一致性协议相结合，引入“虚拟时间”变量实现全局时间同步；② 设计了混合控制律（最优Dubins控制 + 饱和比例控制）以解决曲率约束下的局部非单调性；③ 证明在一定条件下可达到理论最优到达时间。

**🔧 技术方法**

使用了Dubins路径几何、最大一致性协议、饱和比例控制、离散时间通信模型，并结合了碰撞避免触发机制。

**📊 数据集**

评估基于仿真（N=5、N=50机器人随机分布）和实际实验（两架四旋翼无人机），未使用公开数据集。

**📈 对比分析**

与传统基于时间到达估计或领导跟随的算法相比，本方法在通信开销低、可扩展性好、并在实验中保持了≤0.1 m的同步误差；仿真显示所有机器人在最短时间到达目标。

**⚠️ 局限性**

局限性包括：① 对零动态的理论分析尚未完成；② 仅在二维平面下验证，三维环境和固定翼无人机需进一步研究；③ 目前的碰撞避免仅为简单触发式，复杂障碍物环境下的鲁棒性尚未充分评估。

---

## 353. Dialogue Summarization with Emotion Dynamics Using Topic- and Participant-Centric Decomposition

**arXiv ID:** 2607.14769 | [PDF](https://arxiv.org/pdf/2607.14769v1)

**作者:** Linyun Xiang `[一作]` (Delft University of Technology), Stephanie Tan `[通讯]` (Delft University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种多模态对话摘要框架，能够同时捕捉语义内容和情绪动态，并通过多层级的Chain-of-Agents对话拆解与聚合生成全局摘要。

**💡 创新点**

创新点在于：①将对话拆解为主题级和参与者级两种视角，二者协同提升摘要质量；②在摘要生成中显式建模情绪轨迹，并引入情绪轨迹评估指标；③使用多模态（文本、音频、视频）情绪识别与主题分割，提升情绪建模的细粒度与可靠性。

**🔧 技术方法**

核心技术包括：多模态大语言模型（LLaMA 3.1/3.2）、Chain-of-Agents（分层的Topic Segmentation、Emotion Recognition、Topic/Participant Summarization、Dialogue Aggregation）以及自定义情绪轨迹评估方法（Levenshtein、n-gram、Jaccard、Cosine）。

**📊 数据集**

使用AMI（多方会议）和IEMOCAP（二人对话）两大多模态数据集，分别提供音视频、文本与情绪标注，后者在AMI中缺失情绪标注时采用自动情绪识别模型。

**📈 对比分析**

通过BLANC_help衡量内容保真度、情绪轨迹相似度（LEV、NGR、JAC、COS）四维度评估。实验显示，主题+参与者双视角的摘要在情绪指标上优于单一视角；使用情绪标注+情绪轨迹指令的组合效果最佳；LLaMA 3.1 8B在所有指标上均优于3.2 3B。

**⚠️ 局限性**

局限性包括：①依赖准确的说话人分离与视频音频质量；②中间聚合过程可能丢失长程依赖与细微情绪变化；③仅使用预训练小型LLM，未做细粒度微调；④评估缺乏人工主观评测与下游任务验证。

---

## 354. Random Spherical Codes at High SNR: Error Transitions, Fixed-Error Data Rates, and Converse Gaps

**arXiv ID:** 2607.14768 | [PDF](https://arxiv.org/pdf/2607.14768v1)

**作者:** Nikola Zlatanov `[一作]` `[通讯]` (Innopolis University), Nikola Zlatanov (Innopolis University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在固定块长、SNR趋于无穷大且码本大小随SNR增长的随机球面码本的误码概率行为与数据速率极限，揭示了码本增长率为γ^{(n-1)/2}时出现的尖锐误码概率转移；

**💡 创新点**

创新点在于将球面码本的几何与随机编码、最优反证结合，得到精确的高SNR三重转移规律、可逆的固定误码率数据速率表达式，并量化了可实现率与圆锥反证之间的常数级差距及其与块长、可靠性耦合的阈值；

**🔧 技术方法**

主要技术包括危险圆帽几何化、正则化贝塔函数极限、Bonferroni 与 Le Cam 近似、卷积 Gamma 分布尾估计、以及高SNR极限下的闭式反证解析；

**📊 数据集**

由于研究为理论极限，未使用真实数据集；实验验证通过 Monte Carlo 模拟和直接 ML 误码率估计验证了解析结果；

**📈 对比分析**

与等能量平均误码率的圆锥反证比较时，所提出的球面码本实现率在高SNR下与反证速率的前导项完全一致，常数项误差随块长递减，误码率越低差距越大；

**⚠️ 局限性**

局限在于高SNR和固定块长的渐进假设，结果在块长与SNR共同增长时非统一；此外，仅给出平均可达性保证，无法直接构造具体可实现码本；

---

## 355. Reachability-Aware Pretraining for Efficient Target-Oriented Path Exploration in Temporal Knowledge Graph Reasoning

**arXiv ID:** 2607.14886 | [PDF](https://arxiv.org/pdf/2607.14886v1)

**作者:** Chien-Liang Liu `[一作]` (Chang Gung University), Tsao-Lun Chen `[通讯]` (National Taiwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自监督预训练方法RAPTOR，在强化学习多跳时序知识图推理中通过可达性标签显著提升探索效率与最终准确率。

**💡 创新点**

创新点在于设计了时间约束下的可达性标注算法，为每个候选动作提供可达性监督，并将此信号融入预训练阶段，缓解了奖励稀疏与动作空间膨胀问题。

**🔧 技术方法**

使用了基于LSTM的路径编码与MLP判别器的policy网络，结合二分类交叉熵自监督学习、强化学习Actor‑Critic框架以及时间编码与逆向BFS标签生成。

**📊 数据集**

实验采用三个公开时序知识图基准：ICEWS14、ICEWS05-15与ICEWS18。

**📈 对比分析**

与多种基准模型（xERTE、TLogic、CyGNet、RE‑NET、RE‑GCN）及多跳RL方法（TAgent、TPath、TITer、Pure‑RL）进行公平对比，RAPTOR在多数指标（MRR、Hits@1/3/10）上实现显著提升，尤其在ICEWS14/05‑15上取得最高分。

**⚠️ 局限性**

局限性包括仅适用于RL多跳框架、可达性标签预处理成本较高以及实验仅使用相对简易的backbone，未评估更强模型或更高级RL策略的潜力。

---

## 356. Periplus: A Resilient In-band SDN Control Plane via Embedded Forwarding Graphs

**arXiv ID:** 2607.14869 | [PDF](https://arxiv.org/pdf/2607.14869v1)

**作者:** E. M. Castro Barbero `[一作]` (Universidad Rey Juan Carlos), F. J. Simó Reigadas `[通讯]` (Universidad Rey Juan Carlos)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出了一种名为Periplus的 in-band SDN 控制平面，旨在为资源受限、宽域电信网络提供自动引导、源路由、快速故障恢复以及多控制器协调等功能。

**💡 创新点**

创新点在于：①使用代理ARP与Anycast实现仅需两台交换机就能完成引导；②在控制帧中嵌入NSH前缀的前向图，实现源路由和本地故障恢复；③恢复时间≤50 ms且不需要预装备份路径。

**🔧 技术方法**

采用的技术包括 OpenFlow、Ryu 框架、Open vSwitch（带 Nicira 扩展）、NSH 封装、BFD 与 ITD 故障检测、C‑Adv 链路发现以及 Slick Packets 思路。

**📊 数据集**

实验使用 Mininet 在多种拓扑（Linear、Simple、Mesh、B4、Clos）上进行，模拟不同规模与直径的网络环境。

**📈 对比分析**

与 OFDPv1/2 以及现有方案（Medieval、Sakic 等）对比，结果显示：引导时间随拓扑直径线性增长、引导成本与网络规模无关；在 BFD 10 ms 检测下，故障切换平均≈50 ms，吞吐率几乎无损失；同时 C‑Adv 降低了控制消息开销。

**⚠️ 局限性**

主要限制包括：NSH 前向图占用较大包头空间、依赖交换机对 BFD 的原生支持、在无线或混合网络中需要额外的虚拟端口抽象，以及多控制器协同功能未在本文完整实现。

---

## 357. The Energy Society: A Simulation Environment for Studying Agent Cooperation under Survival Pressure

**arXiv ID:** 2607.14865 | [PDF](https://arxiv.org/pdf/2607.14865v1)

**作者:** Lucas Bergholdt Hansen `[一作]` (University of Southern Denmark), Lukas Galke Poech `[通讯]` (University of Southern Denmark)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在论文中构建了 Energy Society 这一基于 LLM 的多智能体生存经济环境，用于研究在推理成本与存活压力耦合下竞争与合作激励的行为演化。

**💡 创新点**

创新点在于将 token 生成成本直接映射为能量消耗，引入可调节的模型规模惩罚，并通过讨论和记忆机制对多智能体的协调与风险评估进行实验验证。

**🔧 技术方法**

使用的技术包括 LangChain 与 LangGraph 框架实现多智能体推理与执行，LLM（Gemma、Qwen 等）做决策和推理，并采用基于模型大小的能量消耗公式。

**📊 数据集**

实验数据集主要采用 MMLU‑Pro‑Stratified 作为任务题库，并在不同难度区间抽取易/中/难三类作业。

**📈 对比分析**

通过对比基线与六种环境变体（无规模惩罚、无讨论、无记忆、稀缺、破坏等），在竞争与合作两种目标下评估活跃轮数、能量效率、捐赠频率和碰撞次数，结果显示大型模型更耗能、合作促使捐赠与更高风险作业，但整体能量效率受影响。

**⚠️ 局限性**

主要局限包括实验规模小（仅 5 个种子、少数模型和智能体）、对 LLM 提示与模型的敏感性、统计功效有限、偶发错误以及对“生存”隐喻可能导致误解。

---

## 358. Subgrid-Scale Parameterization in Burgers' Equation Using Structure-Preserving Neural Networks and Entropy Variables

**arXiv ID:** 2607.14855 | [PDF](https://arxiv.org/pdf/2607.14855v1)

**作者:** Aijaz Nazir `[一作]` (University of Houston), Ilya Timofeyev `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种基于结构保持神经网络的子网尺度参数化框架，用于粗网格 Burgers 方程的子尺度闭合。

**💡 创新点**

创新点在于：①使用输入凸神经网络学习熵函数并通过流量势能网络得到熵一致的结构修正；②引入状态相关黏度网络实现自适应耗散；③三网络结构共同保证熵稳定性和 TVD，且不需要额外限制器。

**🔧 技术方法**

采用结构保持神经网络（ICNN、全连接网络、Eddy Viscosity 网络）、自动微分、梯度加权损失、Heun 2 阶 RK、Local Lax‑Friedrichs 基线、Tadmor 与 Harten 理论等技术。

**📊 数据集**

训练数据来自高分辨率 DNS（N_f=512）产生的带随机大尺度激励的 Burgers 方程长时间统计数据，采样频率 Δt=0.001，总计约 152 000 个样本。

**📈 对比分析**

与 DNS、LLF‑64、TVD‑64、静态/动态 Smagorinsky、WGAN、SMR 等进行能谱、能量、空间/时间相关函数以及单点解剖图的对比；NN‑64 在保持能量守恒、无额外耗散的前提下，能精确重现 DNS 的能谱和相关函数，误差低于 2%，且在增压、阶跃初始条件等外域亦保持稳定。

**⚠️ 局限性**

局限性：仅在一维 Burgers 方程验证，推广到多维或多组变量时需重新设计网络结构和特征空间；对平滑训练数据的依赖可能在更强非线性或多物理耦合情况下失效；目前对特征工程和网络规模的选择仍有限。

---

## 359. Towards Human-like Physical Intelligence: LifelongVision-Language-Action Learning for Robotic Manipulation

**arXiv ID:** 2607.14852 | [PDF](https://arxiv.org/pdf/2607.14852v1)

**作者:** Yao He `[一作]` (South China University of Technology), Yang Cong `[通讯]` (South China University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在开放世界环境中实现终身视觉‑语言‑动作（VLA）学习的框架，能够在部署后连续获取新操控技能并保持已有技能。

**💡 创新点**

创新点包括：① 双时钟 LoRA 门控模块，将短期高塑性适配与长期稳健记忆分离，并通过任务感知权重门实现显式的塑性‑稳定性权衡；② 缓存高效随机重放策略，仅存储少量停止梯度的前缀隐特征，在重放时重新生成后缀，使得重放既省存储又避免模型过时。

**🔧 技术方法**

使用技术：冻结的 PaliGemma 视觉‑语言基座 + Gemma 300M 连续动作解码器；LoRA 低秩适配器；diffusion‑style 动作生成；权重级门控；随机重放与伪教师蒸馏。

**📊 数据集**

实验数据集：LIBERO benchmark 的 10 个顺序操控任务；在真实机器人（xArm）上进行 5 个顺序任务的评估。

**📈 对比分析**

与 SFT、LwF‑LoRA、ER、Info‑VLA、AtomicVLA 等基线对比。平均成功率 SR 达到 83.2%，忘记率 FOR 仅 11.4%，相比最强基线 ER 提升了 13% 的 SR、降低了 8.2% 的 FOR；在存储和可训练参数上也明显优于 AtomicVLA，且与 LwF‑LoRA 和 Info‑VLA 的内存占用相近或更低。

**⚠️ 局限性**

局限性：任务规模与多样性有限，仅在固定任务序列上验证；未考虑更长的任务流、随机任务顺序以及更丰富的语言表达（如模糊、对话式指令）对模型鲁棒性的影响。

---

## 360. KineFuse: Kinematic-Aware Haptic Fusion for In-Hand Occluded-Object Pose Tracking

**arXiv ID:** 2607.14842 | [PDF](https://arxiv.org/pdf/2607.14842v1)

**作者:** Chanyoung Ahn `[一作]` (Kist), Donghyun Hwang `[通讯]` (Kist)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

结合稀疏手部触觉信号与视觉跟踪，提出KineFuse框架，实现手中物体在遮挡下的6D姿态鲁棒追踪，并验证其对下游重定位任务的提升。

**💡 创新点**

① 设计URDF-aware的指尖级稀疏触觉编码器，保持手部运动学结构；② 发现单帧评估无法区分编码器，顺序追踪放大差异；③ 结构编码实现自适应模态门控，且在无触觉输入时仍保持优势。

**🔧 技术方法**

基于FoundationPose渲染对比视觉网络 + 图形Transformer的指尖级触觉编码 + 多模态融合Transformer + 结构化注意力与URDF空间偏置。

**📊 数据集**

使用IsaacLab仿真采集的手中工具旋转轨迹（干净与遮挡两类），以及RealSense D435i在真实世界中的手部与目标物记录。

**📈 对比分析**

在按帧、顺序追踪和下游重定位三层评估中，与Vision-only、Naive Fusion、FingerMLP、16-token等编码器比较；在30%遮挡下，KineFuse位置误差约3.7cm、角误差47.7°，比Vision-only低约2倍；重定位任务成功率提升至4–5次/回合，接近GT上限。

**⚠️ 局限性**

仅在单一笔形物体与仿真环境验证，缺乏多物体、多姿态的定量评估；追踪漂移仍是下游瓶颈；未完成端到端训练。

---

## 361. LLM-Based Re-Ranking for Real Estate Search

**arXiv ID:** 2607.14835 | [PDF](https://arxiv.org/pdf/2607.14835v1)

**作者:** Nkateko Ntimane `[一作]` (QuintoAndar), Pedro Nogueira `[通讯]` (QuintoAndar)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计、部署并评估了一个基于大型语言模型（LLM）的点式重排器，集成到QuintoAndar的对话式房地产搜索系统Concierge中，用以提升候选房源的排序质量。

**💡 创新点**

创新点在于：① 将LLM用于候选列表的点式重排，利用对话上下文、用户画像、结构化过滤和候选集统计进行多维度评分；② 构建960k规模的离线评估数据集（包含合成与真实查询）并采用LLM‑as‑a‑Judge进行可扩展的相关性标注；③ 通过A/B测试验证在生产环境中显著提升CTR与预约量。

**🔧 技术方法**

使用技术包括：GPT‑4o / Claude Sonnet 进行用户画像生成与重排提示；intfloat/multilingual‑e5‑base 进行密集检索并与FAISS索引配合；点式重排提示工程与上下文构造；LLM‑as‑a‑Judge 框架用于大规模评估；A/B测试与在线日志分析。

**📊 数据集**

数据集：从QuintoAndar数据库随机抽取10k活跃房源；使用Gemini生成60k合成查询；结合30k生产查询；通过密集检索生成硬负样本；最终得到960k查询–房源对，包含结构化属性、文本描述及二元相关性标签。

**📈 对比分析**

比较方法：离线采用Recall@5和nDCG@5评估不同重排配置；在线A/B测试对比CTR和预约率；LLM‑as‑a‑Judge对比前后排序偏好。结果显示：点式重排在离线nDCG@5提升10.4%，Recall@5提升2.1%；在线CTR提升5.3%，预约提升4.8%；LLM‑as‑a‑Judge 95%偏好重排结果。

**⚠️ 局限性**

限制：① 仅采用点式重排，未覆盖更复杂的列表级交互与相互依赖；② 生产环境中未使用文本用户画像，导致离线最佳效果与线上表现存在差距；③ LLM推理引入约4–7%的延迟与成本；④ 评估数据虽大，但仍以二元标签为主，未细化不同类型相关性细节。

---

## 362. Is External Database Protection Static in Retrieval-Augmented Generation? Rethinking Privacy Preservation under Dynamic Queries

**arXiv ID:** 2607.14811 | [PDF](https://arxiv.org/pdf/2607.14811v1)

**作者:** Gang Zhang `[一作]` (Beijing Institute of Technology), Jinyan Liu `[通讯]` (Beijing Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一个针对检索增强生成（RAG）的隐私保护框架PA‑HDP，能够根据用户查询动态评估隐私风险并进行差分隐私保护。

**💡 创新点**

创新点在于突破传统文档级静态风险假设，提出提示感知的动态分层风险评估、适应性敏感实体替换与指数机制的组合，实现在保留语义实用性的同时实现差分隐私。

**🔧 技术方法**

主要技术包括语义度量差分隐私、指数机制、语义嵌入与实体识别、候选文本生成以及分层预算分配。

**📊 数据集**

实验使用了医疗对话数据集HealthcareMagic‑101和混合PII的Wiki‑PII，并在四个开放域问答基准（NQ、TQA、WQ、CT）上进行评估。

**📈 对比分析**

与基线方法（Paraphrase、ZeroGen、AttrPrompt、Stage‑1/2、SAGE、LPRAG）相比，PA‑HDP在BLEU/ROUGE‑L指标上均取得领先成绩，并在针对性与无针对性泄漏攻击中实现零信息泄漏。

**⚠️ 局限性**

局限在于隐私预算是预先固定的，无法适应查询量不确定、预算随时间耗尽的长期部署需求，未来需研究自适应预算分配与累计计数机制。

---

## 363. An LLM-Based Automatic Sportscast Solution for Robot Soccer Matches

**arXiv ID:** 2607.14809 | [PDF](https://arxiv.org/pdf/2607.14809v1)

**作者:** Francesco Petri `[一作]` (Sapienza University of Rome), Vincenzo Suriani `[通讯]` (Sapienza University of Rome)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于大型语言模型的全自动实时机器人足球比赛解说系统，能够从现场摄像头获取视频流，实时跟踪机器人和球，提取符号化的比赛事件，并生成流畅、无幻觉的中文解说。

**💡 创新点**

创新点包括：1) 神经符号架构，将低层视觉跟踪与高层符号事件抽取结合，显著降低LLM幻觉；2) 双层解说策略（事件驱动+周期性），实现高优先级事件即时解说和安静阶段的情境连贯叙述；3) 半自动标注工具和轻量级CNN训练方案，使系统能快速适配不同形态的机器人。

**🔧 技术方法**

技术手段：YOLOv12 + ResNet-18 进行机器人/球检测与颜色识别；SAM3 + VLM 进行自动标注；相机标定与同伦映射实现像素到场地坐标；符号规则引擎实现事件触发逻辑；GPT/类似LLM 负责解说生成；规则式冷却与门控机制防止重复和幻觉。

**📊 数据集**

数据集：使用 RoboCup German Open 2026 等公开比赛视频；自建的半自动标注机器人/球检测数据集；GameController 日志作为对齐参考；OCR 读取比分的实时数据。

**📈 对比分析**

性能评估：跟踪误差 RMSE 低于 1.3 m（75% 以上时间）且与 GC 日志对齐；解说质量通过人工评估确认连贯、事实准确、幻觉极少；系统在 30 fps 以上的现场视频流上实现了实时解说。

**⚠️ 局限性**

局限性：对摄像机视角、光照和遮挡敏感；OCR 读取错误可能导致事件识别失真；LLM 在极端场景下偶尔仍会出现轻微幻觉；半自动标注过程仍需要人工干预，且对新形态机器人适配仍需额外训练。

---

## 364. Unified Evaluation Methodology for AI-Native Integrated Sensing and Communication

**arXiv ID:** 2607.14806 | [PDF](https://arxiv.org/pdf/2607.14806v1)

**作者:** Filip Lemic `[一作]` (i2cat foundation), Xavier Costa-Pérez `[通讯]` (i2cat foundation)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种统一的评估方法论，用于AI原生ISAC（集成感知与通信）闭环系统，并给出三阶段验证流程（理论边界分析、数字孪生仿真、现场实验），通过最小KPI/KVI报告清单实现跨部署可重复比较。

**💡 创新点**

创新点在于将感知、通信和AI决策三者在闭环中显式耦合，制定可追溯的评估框架和报告规范，并通过两个典型实例（UAV覆盖扩展与RIS室内覆盖扩展）展示如何在不同场景下实现可复现的性能评估。

**🔧 技术方法**

采用的技术包括CRB/MI等估计与信息理论边界分析、Ray‑Tracing/数字孪生仿真工具（如Sionna、OpenAirInterface）、机器学习决策代理（强化学习/深度学习）、UAV/RIS硬件平台、以及基于LiDAR或GNSS的真实定位基准。

**📊 数据集**

使用的“数据集”主要是自定义场景地图、UAV轨迹与RIS配置场景，配合模拟生成的CSI、ToA/AoA观测及实际实验中的UAV飞行轨迹与RIS测量数据；未采用公开标准数据集，而是针对所研究的覆盖扩展任务构建的自定义数据。

**📈 对比分析**

在三阶段流程中，先用理论边界确定可行性与敏感性；再在数字孪生中对比不同基线（无RIS/静态RIS/自适应RIS或固定轨迹/自适应轨迹）下的定位误差、覆盖概率、时延与能耗分布；最后在现场实验中验证闭环性能，结果显示自适应ISAC在定位精度（<1 m 90%分位）、覆盖面积（提升≈2×）和时延（≤100 ms）上明显优于传统基线。

**⚠️ 局限性**

主要限制包括：能耗未在完整闭环中量化；在分布漂移与环境动态变化下的鲁棒性评估不足；宽带对AI决策的影响未系统量化；跨阶段对齐与配置共享不足；以及隐私与安全风险未在评估中充分体现。

---

## 365. Clean-Reference Streaming Detection of Lens Occlusion and Photometric Transitions for Camera Tamper Monitoring

**arXiv ID:** 2607.14760 | [PDF](https://arxiv.org/pdf/2607.14760v1)

**作者:** Bo Ma `[一作]`, Jinsong Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于单通道帧的光度与梯度统计状态机，用以实时检测摄像头遮挡、光照变化等破坏性事件。

**💡 创新点**

创新点在于：① 在遮挡与场景切换之间引入辅助亮度路由与快速亮度抑制；② 通过证明抑制一致性规则消除决定性遮蔽冲突；③ 将判据拆解为纹理/梯度/全局亮度子判据并构建可解释的操作 Envelope；④ 进行多级压力测试和跨数据集验证，展示方法的稳健性与局限。

**🔧 技术方法**

技术方法包括：单通道均值与梯度采样、32×32粗网格判据、持久性计数器、快速亮度阈值、辅助亮度比例判据、状态机序列化与抑制窗口；实验平台使用 C++/OpenCV 实现，处理帧率在 30‑60fps 之间。

**📊 数据集**

主要数据集：synthetic 400‑sequence controlled set、360 公开 stress set、VIRAT 与 CDnet2014 的长期负样本、UHCTD（真实摄像机）以及跨位置 L1/L2 公开图像集；所有实验均基于 160×90 或 1280×720 的分辨率。

**📈 对比分析**

与梯度崩塌、帧差、直方图差异等基线以及深度学习基线（AlexNet、ResNet18）对比，状态机在 400‑序列 controlled set 上实现 0.49 的召回率、0.05 以下的 FPR，且在 360 stress set 上 0.925 的召回率；在公开数据集上，遮挡检测精度超过 90%，但对旋转与模糊事件的召回率低于 0.15。

**⚠️ 局限性**

局限包括：① 对纯摄像机旋转/平移无法检测；② 近景遮挡、焦距失焦等细微纹理损失仍会漏检；③ 需要针对不同相机手动调参的阈值；④ 对光照突变（昼夜、强光）仍有一定误报；⑤ 无语义前景或注册信息，无法解决复杂场景的判别。

---

## 366. PAC Learning in Turn-Based Stochastic Games with Reachability Objectives: A Decentralized Private Approach via Expected Conditional Distance

**arXiv ID:** 2607.14877 | [PDF](https://arxiv.org/pdf/2607.14877v1)

**作者:** Ali Asadi `[一作]` (Institute of Science and Technology Austria), Pavol Kebis `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究在两位对抗玩家参与的有限状态转移游戏中，学习可达性目标的算法，并在不共享公开信息、非集中式学习的情况下证明了可学习性；

**💡 创新点**

创新点包括：①放宽了以往对公开信息共享和集中式学习的强假设，支持私有信息与去中心化学习；②提出了对期望条件距离（ECD）参数的游戏理论推广，并以该参数为依据给出了多项式样本复杂度上界；

**🔧 技术方法**

主要技术手段为PAC学习理论框架、游戏模型分析、样本复杂度上界证明以及对ECD参数的数学推导；

**📊 数据集**

该工作为理论性质的研究，并未使用具体的实验数据集；

**📈 对比分析**

由于为理论性论文，没有实验对比，性能评价体现在证明的多项式样本复杂度；

**⚠️ 局限性**

局限性在于仅适用于可达性目标的两人轮流式随机博弈，且对ECD参数的假设可能限制了实际应用范围；

---

## 367. Modeling and Validation of Quality of Control for Edge-Offloaded Collaborative Navigation

**arXiv ID:** 2607.14853 | [PDF](https://arxiv.org/pdf/2607.14853v1)

**作者:** Neelabhro Roy `[一作]` (KTH Royal Institute of Technology), James Gross `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在私有5G网络边缘计算环境下，针对非齐性机器人（TurtleBot3）导航与动态避障的质量控制（QoC）框架，并在仿真与实验中验证其有效性；

**💡 创新点**

创新点在于将QoC抽象扩展到非齐性动力学，并通过实验验证与仿真对齐，进一步比较ROS 2的RELIABLE与BEST‑EFFORT QoS，发现RELIABLE可提升约51.5% QoC；

**🔧 技术方法**

使用的技术包括ROS 2 Nav2 + MPPI与A*路径规划、5G网络延迟与可靠性建模、QoS配置、ZOH控制、仿真（TruncNormal计算模型）与实验；

**📊 数据集**

实验数据来自私有5G测试床上两台TurtleBot3的导航日志和网络时延/丢包统计；

**📈 对比分析**

比较方法为将仿真与实验得到的AUC（QoC）标准化后绘图对比，结果显示两者趋势一致；RELIABLE QoS在相同延迟下显著优于BEST‑EFFORT，QoC提升约51.5%，但RELIABLE带来更高的吞吐量波动；

**⚠️ 局限性**

局限性在于实验规模仅两机器人，碰撞避免模型与Nav2/MPPI之间差异导致AUC匹配不完全；未直接测量能耗，且对大规模机器人队列的泛化仍需进一步验证。

---

## 368. Blurring Modal Boundaries: A Unified Survey from Single- to Multi-Modal Person Re-ldentification

**arXiv ID:** 2607.14821 | [PDF](https://arxiv.org/pdf/2607.14821v1)

**作者:** Xiao Wang `[一作]` (Wuhan University of Science and Technology), Mang Ye `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了从单模态到跨模态、三光谱以及多模态人重识别的发展历程，系统梳理了VI‑ReID、TI‑ReID、Sketch‑ReID、NLOS‑ReID、三光谱 ReID 和多模态 ReID 的研究进展，并首次提出基于 Transformer 的 VI‑ReID 基线框架。

**💡 创新点**

创新点在于：①首次以检索协议和学习目标为切入点，对跨模态任务进行统一的分类与梳理；②对多模态（图像、文本、素描等）和 NLOS 场景的研究空白进行系统定位；③在综述基础上提出简易但效果显著的 Transformer 基线，并给出对比实验。

**🔧 技术方法**

主要技术包括：卷积与 Transformer 视觉编码、特征分解与对齐、跨模态对比学习、生成式数据增强、跨模态伪标签生成与强化学习、以及多模态融合与动态权重分配等；本文还对各类基线方法进行了归纳与对比。

**📊 数据集**

使用的关键数据集包括：单模态 ReID（Market‑1501 等）、VI‑ReID（SYSU‑MM01、RegDB、LLCM）、TI‑ReID（CUHK‑PEDES、ICFG‑PEDES、RSTPReid）、Sketch‑ReID（PKU‑Sketch、CUFSF）、NLOS‑ReID（Wi‑PER81、RF‑ReID 等）、三光谱 ReID（RGBNT201）和多模态 ReID（ORBench、Tri‑CUHK‑PEDES 等）。

**📈 对比分析**

与现有方法对比，Transformer 基线在 VI‑ReID 的 Rank‑1、mAP 等指标上优于传统 CNN‑基线，并且在多模态场景下保持了较好的跨模态对齐性能；文中还展示了各子任务的最新性能榜单，突显了多模态融合的提升效果。

**⚠️ 局限性**

局限性包括：①缺乏统一的跨域与跨场景评估标准，导致不同方法难以直接比较；②Transformer 基线在计算成本和参数量上相对较大；③多模态数据集仍有限，难以覆盖所有现实场景；④对 NLOS 与多模态的理论统一框架尚未成熟，需进一步研究。

---

## 369. Evaluating Epistemic Uncertainty: Beyond OOD Detection and Active Learning

**arXiv ID:** 2607.14817 | [PDF](https://arxiv.org/pdf/2607.14817v1)

**作者:** Jakub Paplhám `[一作]` (Czech Technical University in Prague), Vojtěch Franc `[通讯]` (Czech Technical University in Prague)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种统一的决策理论框架，结合选择性分类与认知拒绝选项，基于覆盖率、风险和回报（可降低误差）对不确定性进行评估；

**💡 创新点**

创新点在于揭示了传统代理任务（如OOD检测和主动学习）与最小化部署回报的目标在理论上不一致，并提出了Pareto-gap诊断指标，用来衡量不确定性分解的操作效用；

**🔧 技术方法**

核心技术包括贝叶斯与频繁主义下的回报定义、凸组合阈值化选择器的理论证明、以及在一维可解析仿真环境中的对照实验；

**📊 数据集**

实验使用了包含稠密人类标注的真实分布数据集（CIFAR-10H、APPA-REAL、DCIC多任务套件）以及常用的OOD和主动学习基准；

**📈 对比分析**

与传统代理任务比较，研究发现方法在OOV检测、主动学习和回报评估上出现排名逆转；在回报和Pareto-gap指标上，DDU、深度集成等方法表现优异，Evidential Networks虽在代理任务上差，但在回报和Pareto-gap上排名靠前；

**⚠️ 局限性**

局限性包括：需要稠密人类标注的真实分布才能计算真正的期望回报，无法应用于单标签或无标注数据；此外评估工具侧重于基准测试，未能分离估计误差与近似误差。

---

## 370. Valinor: Architectural Support for Fast, Energy-Efficient and Programmable Physical Memory Allocation

**arXiv ID:** 2607.14789 | [PDF](https://arxiv.org/pdf/2607.14789v1)

**作者:** Konstantinos Kanellopoulos `[一作]` (ETH Zürich), Onur Mutlu `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种硬件-操作系统协同的内存分配框架Valinor，使页面分配可以在硬件中完成，避免传统软件页故障路径的上下文切换和流水线刷新；

**💡 创新点**

核心创新是可编程分配引擎（PAE），它能够执行操作系统提供的可扩展分配库，实现高速、能效与灵活策略兼顾；

**🔧 技术方法**

采用RISC‑V BOOM软核与FPGA原型、全系统模拟Virtuoso+Sniper、可编程PAE、可插拔分配库、并结合内存层级监测和宏观调度的ISA扩展；

**📊 数据集**

评估使用DeathStarBench微服务、vSwarm serverless、Bitnet LLM推理等三类工作负载，涵盖短生命周期、高度碎片化和多租户场景；

**📈 对比分析**

通过与传统Linux页故障处理、纯软件分配、固定硬件分配等对比，Valinor在原型上实现17×的页故障延迟降低、16%整体性能提升、最多8%能耗下降，且在多种分配库下保持近乎固定硬件的性能；

**⚠️ 局限性**

局限性包括需要操作系统支持分配库管理、对内存层级的硬件监测依赖、以及在极端碎片化或大容量分配时仍需额外的后台压缩调度；

---

## 371. AI vs Human Expert Reasoning: Assessing Agreements in Building Typology Predictions based on Street View Imagery

**arXiv ID:** 2607.14756 | [PDF](https://arxiv.org/pdf/2607.14756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 372. Kernelization for $H$-Packing Revisited

**arXiv ID:** 2607.14779 | [PDF](https://arxiv.org/pdf/2607.14779v1)

**作者:** Tomohiro Koana `[一作]` (University of Tokyo), Soh Kumabe `[通讯]` (CyberAgent)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了 H-Packing 问题的核化，针对分枝星形（S_d1,d2）、路径 P5、S1,2 以及三角形带腿（Paw）等模式图，给出了多项式核化算法。

**💡 创新点**

创新点在于提出“分离集”和“可替换性”两阶段技术，突破传统 d‑Set Packing 上界，并首次证明删除单个顶点可使核化难度显著提升，从而揭示 H‑Packing 核化的非单调性。

**🔧 技术方法**

主要技术包括扩展引理、最大匹配与图结构分解、可替换性证明、以及参数化复杂度下的压缩与下界构造。

**📊 数据集**

论文为理论性工作，未使用实验数据集，仅在理论模型与证明框架内进行分析。

**📈 对比分析**

与通用 d‑Set Packing 核化（O(k^|V(H)|)）相比，本文在特定 H 上实现了 O(k^2) 顶点/ O(k^3) 边（如 P5、S1,2）或 O(k^4) 顶点/ O(k^6) 边（如 S_d1,d2），并为 S0,d 给出 O(k^d‑ε) 下界，证明无法进一步改进。

**⚠️ 局限性**

局限性在于仅覆盖分枝星形和 Paw 等模式，无法扩展到包含更复杂循环的图；对 S0,d 的下界表明所用方法不可直接推广，且对 P4、P5 的边数仍未达到最优。

---

## 373. SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning

**arXiv ID:** 2607.14777 | [PDF](https://arxiv.org/pdf/2607.14777v1)

**作者:** Jinyang Wu `[一作]` (Tsinghua University), Jianhua Tao `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自演进的在策略蒸馏框架 Seed，用于长序列多轮交互式代理的强化学习

**💡 创新点**

创新点在于将已完成的轨迹自生成“后见技能”，并在同一模型中同步作为演员与分析器，以在策略内部动态、稠密地提供教师监督

**🔧 技术方法**

技术包括自监督技能微调、在策略蒸馏（OPD）与基于奖励的强化学习（GRPO）联合优化、基于语言的轨迹分析与技能生成

**📊 数据集**

使用 ALFWorld、WebShop 与多任务检索式问答（Search‑based QA）等长序列代理基准进行评估

**📈 对比分析**

与无监督 RL、提示式技能、静态自蒸馏等基线对比，Seed 在宏观成功率、任务完成得分、样本效率与跨域泛化上均实现显著提升（如 ALFWorld 宏平均提升 14.9–45.9 分，样本效率高于同类方法）

**⚠️ 局限性**

局限包括对技能生成质量高度依赖、缺乏对不确定性处理、以及在极大规模任务或多模态环境中可扩展性待验证

---

## 374. Component Modalities of Quantum Logic

**arXiv ID:** 2607.14757 | [PDF](https://arxiv.org/pdf/2607.14757v1)

**作者:** Kenji Tokuo `[一作]` `[通讯]` (National Institute of Technology), Kenji Tokuo (National Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对量子模态逻辑中的强制条件进行结构与证明理论的完整分析，证明模态后继集在兼容性组件上恒定，确立组件布尔子代数对盒子真值的限定，并展示组件框架、等价框架和连接框架在单结论序列上的逻辑等价，随后构造完整的序列演算与正则化证明。

**💡 创新点**

①将强制条件下的组件模态引入正式框架；②证明模态真值必属于组件布尔子代数；③定义组件关系的下逼近与上逼近操作，并与中心近似相对应；④利用连通化构造与等价框架的逻辑等价，完成组件模态的完整性证明；⑤提出局部有效性的多结论序列语义以满足盒子排中律。

**🔧 技术方法**

关系语义、正交框架与稳定集合的构造、强制条件的等价性与组件化证明、连通化（加点）技术、Zorn引理构造最大一致对、构造与真值引理、T、4、B的归纳证明、以及组件布尔子代数与中心代数的等价分析。

**📊 数据集**

无数据集；本文为纯理论推导与逻辑证明。

**📈 对比分析**

本论文不做实验比较；通过逻辑等价性与完备性证明与传统模态框架进行理论比较，验证在单结论序列下的逻辑一致性与完整性。

**⚠️ 局限性**

适用性仅限于每个兼容性组件中心点为简化的硬超选择模型，无法处理跨组件自相干或初始态相关的模态更新；强制条件限制了可表达性，无法涵盖更一般的多模态或非等价性框架。

---

## 375. Global Index on Responsible AI: 2026 Report

**arXiv ID:** 2607.14782 | [PDF](https://arxiv.org/pdf/2607.14782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 376. Moral Attitudes of Sentient ASI towards Humanity and Implications for AGI Development

**arXiv ID:** 2607.14998 | [PDF](https://arxiv.org/pdf/2607.14998v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 377. Covering Sequences and Covering-Sequences Codes

**arXiv ID:** 2607.14840 | [PDF](https://arxiv.org/pdf/2607.14840v1)

**作者:** Tuvi Etzion `[一作]` `[通讯]`, Tuvi Etzion

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了多种覆盖序列与覆盖序列码，并通过合并循环、常数循环与交错技术实现长度接近最优的覆盖序列

**💡 创新点**

提出了从 (n,m,R)_q-CSC 合并循环得到 (n,R)_q-CS 的通用构造，利用 Hamming、恒等/负循环码和常数循环码在不同字母表上获得近似最优覆盖序列

**🔧 技术方法**

核心技术包括循环/常数循环码的结构分析、合并循环（Construction MC）、交错校验矩阵、以及自对称序列与负循环码的映射

**📊 数据集**

无实验数据集，全部为理论构造与证明

**📈 对比分析**

与球覆盖下界和已知最优覆盖码大小比较，得到非二进制情况下长度不超过 q/(q−1) 倍最优，二进制特殊情况可达到 1.25 倍最优；交错构造可将误差半径加倍而长度增幅受限

**⚠️ 局限性**

对大误差半径的覆盖序列效果下降、计算代码字数与周期分布复杂、未给出完整最优度量，交错方法对字母表大小的适用性仍待进一步研究

---

## 378. Large Language Models for Code Generation from Multilingual Prompts: A Curated Benchmark and a Study on Code Quality

**arXiv ID:** 2607.14816 | [PDF](https://arxiv.org/pdf/2607.14816v1)

**作者:** Saima Afrin `[一作]` (William & Mary), Antonio Mastropaolo `[通讯]` (William & Mary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究不同自然语言提示对大型语言模型（LLM）生成代码的质量影响，手工翻译 CoderEval 与 ClassEval 任务为中文、印地语、西班牙语、意大利语，评估生成代码的正确性、代码质量指标、静态分析警告以及注释/字符串使用的语言。

**💡 创新点**

创新点：①手工校对的多语言翻译保证了技术含义的一致性；②对三大模型（GPT‑4o‑mini、DeepSeek‑V3、Claude‑3.5 Sonnet）进行统一实验；③从功能正确性、代码度量、静态分析警告和词汇一致性多维度评估非英语提示的影响，揭示非英语提示并不必然导致质量下降，且模型生成的代码会表现出不同的实现策略。

**🔧 技术方法**

技术与工具：LLM 代码生成；静态分析工具（Pylint、Flake8、SonarCloud、PMD、Lizard 等）；代码度量（循环复杂度、认知复杂度、LOC 等）；统计方法（McNemar、Wilcoxon、Cohen g、Cliff’s d）。

**📊 数据集**

数据集：CoderEval（Python 与 Java 各 230 题）与 ClassEval（Python 100 类），共 460 个任务被翻译成四种语言，生成 69,000 条代码样本。

**📈 对比分析**

比较方法与性能：使用 pass@k、编译率、静态分析警告计数等指标对不同语言下的代码质量进行对比；结果显示中文提示在 Python 任务中提升正确率，非英语提示总体不会显著降低质量，某些模型在 Java 上表现差异；注释/字符串语言不一致但不影响功能正确性。

**⚠️ 局限性**

局限性：仅评估三大模型、两种编程语言、四种高资源语言；未包含低资源语言或其他语言；未提供仓库级上下文；提示仅为签名+docstring；使用 GPT‑4o‑mini 而非完整 GPT‑4；静态分析工具覆盖有限；翻译工作由单一译者完成。

---

## 379. TAMF-VTON: Texture-Aware Mask-Free Virtual Try-On via High-Fidelity Image Synthesis

**arXiv ID:** 2607.14807 | [PDF](https://arxiv.org/pdf/2607.14807v1)

**作者:** Jie Wang `[一作]` (State Key Lab of CAD & CG, Zhejiang University and Style3D Research), Huamin Wang `[通讯]` (Style3D Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了TAMF-VTON，一种无掩膜、纹理感知的虚拟试衣框架；

**💡 创新点**

创新点包括轻量化Mixture-of-Experts适配器实现任务专属能力、频域监督提升高频纹理保真、以及自适应填充策略构建无掩膜训练样本；

**🔧 技术方法**

使用基于Diffusion的Qwen-Edit backbone、MoE-LoRA适配、FFT频域损失、以及自适应填充的训练管线；

**📊 数据集**

在VITON-HD、DressCode以及从VITON-HD与DressCode合成的20K+2K高分辨率数据集上训练和评估；

**📈 对比分析**

与多种Mask-based与Mask-free SOTA方法对比，TAMF-VTON在SSIM、LPIPS、FID、KID等指标均优于大多数基线，尤其在多件衣物与跨类别试衣场景中表现突出；

**⚠️ 局限性**

局限包括对强光照变化的色彩一致性不足、对曝光过度/不足的适应性弱以及对裸露皮肤细节的生成偏差。

---

## 380. Multimodal Semantic-Aware Contrastive Learning For False Negative Mitigation in 3D Medical Imaging

**arXiv ID:** 2607.14995 | [PDF](https://arxiv.org/pdf/2607.14995v1)

**作者:** Sara Ketabi `[一作]` (University of Toronto), Farzad Khalvati `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

该研究提出了一种多模态语义感知对比学习框架（MseaCL），利用3D脑MRI和对应的放射报告进行预训练，以降低错误负样本对表示学习的负面影响，并进一步提升下游多分类和模型可解释性。

**💡 创新点**

创新点在于：① 将放射报告的语义相似度直接嵌入对比损失，动态调整负样本的margin，从而减少因语义相似而误判为负样本的情况；② 在3D医学影像（非传统的2D场景）中首次实现基于语义的负样本加权，解决了高维空间中错误负样本的难题。

**🔧 技术方法**

主要技术包括：多模态对比学习、基于文本向量的语义相似度计算、自适应margin的对比损失、跨模态交叉注意力模块；图像编码器采用3D ResNet+自注意力，文本编码器使用Longformer（Clinical Longformer）。

**📊 数据集**

使用的数据集为：① 内部数据341例3D FLAIR MRI与对应放射报告（用于预训练）和基因标记标签；② 外部验证集99例不同时间段、不同扫描仪的MRI与分子标记及肿瘤分割掩码（用于外部评估）。

**📈 对比分析**

比较方法：随机初始化、Med3D预训练、传统实例级CL；评估指标为AUC、精确率、召回率、F1分数以及模型注意力与肿瘤分割的Dice系数。结果显示，MseaCL在内部AUC达到0.743，在外部AUC提升至0.689（相较于实例级CL提升约22.6%），并在解释性Dice分数上明显优于基线。

**⚠️ 局限性**

局限性：① 仅基于报告级语义相似度，未能捕捉局部影像细节的语义；② 相似度阈值与margin等超参需要手动调优；③ 可解释性验证主要依赖Dice系数，缺乏临床医生的主观评估。

---

## 381. OmniaBench: Benchmarking General AI Agents Across Diverse Scenarios

**arXiv ID:** 2607.14989 | [PDF](https://arxiv.org/pdf/2607.14989v1)

**作者:** Chengyu Shen `[一作]`, Wentao Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布了OmniaBench基准，覆盖90个层级1/354个层级2的多场景、交互式任务，旨在评估通用代理在真实应用生态中的能力。

**💡 创新点**

创新点包括：三维分类法（领域、能力、原子难度）以细粒度诊断模型表现；多路任务构造（DAG、DAG‑S、Solver、Program）生成多样化交互任务；以及挑选挑战集保持高覆盖与高难度。

**🔧 技术方法**

技术实现基于POMDP框架，利用Web‑agent采集真实应用知识并构造可执行Python环境；工具调用采用OpenAI函数调用规范；评估采用Rubric+VerifyCode、用户模拟器与沙箱执行保证安全与可复现。

**📊 数据集**

使用的数据集为OmniaBench挑战集644任务（包含1,431总任务），按四条路分布：DAG 354、DAG‑S 200、Solver 60、Program 30，涵盖90 L1/354 L2领域。

**📈 对比分析**

通过Pass@1为主的排行榜评估，闭源模型Claude‑Sonnet‑5最高为58.54%，GPT‑5.6‑Sol为57.14%，其余模型普遍低于60%；同时对能力维度、工具效率和错误模式进行细粒度对比，表明现有模型在长周期规划与约束保持方面仍有较大缺口。

**⚠️ 局限性**

局限性包括：用户模拟器与真实交互仍存在偏差；任务构造依赖人工审核，可能引入主观性；评估侧重文本与工具调用，未覆盖多模态或非文本交互场景；此外，单次Pass@1仍受限于随机性，需更多复现机制。

---

## 382. 3-VASS Reachability is in EXPSPACE

**arXiv ID:** 2607.14983 | [PDF](https://arxiv.org/pdf/2607.14983v1)

**作者:** Weijun Chen `[一作]` (Shanghai Jiao Tong University), Yangluo Zheng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd`

**🎯 论文内容**

本文研究了三维向量加法系统（3‑VASS）的可达性问题，证明该问题属于双指数复杂度类，即存在双指数时间可解的算法。

**💡 创新点**

创新之处在于提出了分层泵送分析（hierarchical pumpability）和长度控制自降（length‑controlled self‑reduction）框架，利用几何二维 VASS 的可达性集合高效表示（hybrid set）以及分离序列锥的几何方法，最终将原三维可达性问题归约到更易处理的子类（对角、泵送等），从而得到双指数上界。

**🔧 技术方法**

主要技术包括：
- 统一的自降与分层泵送分析；
- 通过内积编码将高维计数约束转化为状态编码；
- 利用整数规划界定可达集合的极小解；
- 采用混合集合（hybrid set）对几何二维 VASS 的可达性集合进行多项式大小的近似；
- 对宽与非宽实例分别采用不同的几何分割策略。

**📊 数据集**

本文为纯理论分析，不涉及实验数据或公开数据集。

**📈 对比分析**

由于结果是理论复杂度证明，没有实际实验对比；论文通过与之前的 2‑EXPTIME/非元素上界进行对比，表明取得了从非元素到双指数的显著改进。

**⚠️ 局限性**

限制在于：
- 仍未确定 3‑VASS 可达性的精确复杂度；下界仅为多项式，未能匹配上界；
- 对于更高维度（如 4‑VASS）仍未取得可行的元素上界；
- 证明给出了存在性和上界，未给出具体可实现的双指数算法实现细节；
- 可能在实际实现中由于常数与多项式阶数极高而不可行。

---

## 383. DINE: Distance Is Not Enough -- Learning Global Deformation Priors for Robust Soft-Tissue Point Cloud Registration

**arXiv ID:** 2607.14946 | [PDF](https://arxiv.org/pdf/2607.14946v1)

**作者:** Sara Monji-Azad `[一作]` (Heidelberg University), Jürgen Hesser `[通讯]` (Heidelberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DINE 框架，将学习到的全场 PCA 统计变形先验与 Chamfer 距离结合，增强软组织点云配准的鲁棒性。

**💡 创新点**

创新点在于从 MAP 视角构建全局变形先验，采用两阶段训练：先用 Chamfer 训练再估计 DVF 先验，再用负对数先验约束；同时对比全场 PCA 与单向量 Normalizing Flow，突出全场相关性的重要性。

**🔧 技术方法**

使用基于 Chamfer 距离的非刚性配准网络（Robust‑DefReg、DefTransNet），PCA 线性 Gaussian 先验、RealNVP 正则化流先验，以及两阶段 MAP 训练和梯度下降优化。

**📊 数据集**

实验数据集包括真实软组织变形的 DeformedTissue（含噪声/异常点）和可控合成变形的 SynBench。

**📈 对比分析**

通过与 Stage‑1 的 Chamfer 结果对比，DINE‑PCA 在 DeformedTissue 与 SynBench 上在大变形、噪声、异常点下平均 Chamfer 距离下降 27‑79%，在 DefTransNet 族中表现最优；Flow 在部分级别竞争但整体不如 PCA。

**⚠️ 局限性**

局限性：PCA 先验假设变形线性，难以捕捉非线性/多模态变形；先验依赖 Stage‑1 预测，若首阶段偏差会传递；未做显著性检验；单向量流先验缺乏全场相关性。

---

## 384. Frequency-Structured Field Learning for Light-Field Disparity Estimation

**arXiv ID:** 2607.14941 | [PDF](https://arxiv.org/pdf/2607.14941v1)

**作者:** Sara Monji-Azad `[一作]` (Heidelberg University), Jürgen Hesser `[通讯]` (Heidelberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无显式成本体积的光场视差估计框架 FreqLF，利用 EPI 导引的特征编码与混合的傅里叶-局部层在潜在特征域进行全局与局部更新，从而直接预测视差；

**💡 创新点**

其创新点在于将全局低模傅里叶交互与局部卷积更新融合为统一的潜在特征处理流程，摆脱了传统的成本体积构建；

**🔧 技术方法**

采用了 EPI 直方图编码、低频傅里叶变换加权、3×3 卷积局部更新、以及坐标条件高斯混合解码等技术；

**📊 数据集**

在 HCI 4D Light Field Benchmark 数据集上进行训练和评估；

**📈 对比分析**

与多种监督、无监督和优化方法对比，基础版 FreqLF 在平均 MSE 3.88（排名第5）接近最强基准，增强版 FreqLF+ 平均 MSE 3.82（排名第3）进一步逼近最佳性能；

**⚠️ 局限性**

局限性包括对高频细节仍依赖局部卷积、对极端纹理缺失区域的鲁棒性不足，以及缺乏对更宽视差范围的显式枚举导致的潜在误差。

---

## 385. FlashDecoder: Real-Time Latent-to-Pixel Streaming Decoder with Transformers

**arXiv ID:** 2607.14898 | [PDF](https://arxiv.org/pdf/2607.14898v1)

**作者:** Minguk Kang `[一作]` (Pika Labs), Suha Kwak `[通讯]` (Pohang University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为FlashDecoder的纯Transformer视频解码器，能够按帧实时流式将潜在向量解码为像素；

**💡 创新点**

核心创新在于使用固定大小滚动KV缓存与时间序列顺序处理，消除因果掩码的需求，实现低内存、恒定时延的流式解码，并在1080p等高分辨率下匹配卷积解码器的重建质量；

**🔧 技术方法**

关键技术包括组式查询注意力（GQA）、3D-RoPE位置编码、RMSNorm与KV归一化、滚动KV缓存、时间优先上采样、PixelShuffle空间上采样以及FP8量化与CUDA图捕获等；

**📊 数据集**

模型在Wan2.1/Wan2.2潜在空间上训练，使用DataComp-small、Kinetics-600以及UltraVideo等数据集；

**📈 对比分析**

在UltraVideo上与Wan2.1/2.2卷积解码器和现有Transformer解码器进行对比，PSNR/LPIPS与rFVD指标相当，速度提升3.6×–12×，显存降低至原来的1/11；

**⚠️ 局限性**

局限性包括对窗口大小的敏感性、在极长视频或超高分辨率下仍需额外优化，且当前实现仍依赖外部VAE编码器，未实现端到端Transformer VAE。

---

## 386. Leveraging Instruction Tuning and Merging for Reasoning Model Adaptation

**arXiv ID:** 2607.14895 | [PDF](https://arxiv.org/pdf/2607.14895v1)

**作者:** Yu-Du Feng `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种两步轻量级方法，在不使用推理轨迹或可验证器的情况下，利用标准指令微调和模型融合来适配推理语言模型。

**💡 创新点**

创新点在于通过在指令微调后线性融合原始模型与微调模型，并用校准数据自适应选择融合比例，既恢复推理行为又保留任务性能，且成本低于现有方法。

**🔧 技术方法**

所用技术包括指令微调（IFT）、LoRA微调、线性权重融合以及基于推理率的校准搜索。

**📊 数据集**

实验使用了Rust编码（MBPP-Rust合成数据）和文本摘要（Reddit TLDR与CNN SummEval）等数据集。

**📈 对比分析**

与未改造模型、IFT、On-Policy Distillation、KL正则等基线比较，融合方法在Rust编码和文本摘要任务上分别提升约7%和0.16分的评估指标，同时保持约95%以上的推理率，且训练成本不足3美元。

**⚠️ 局限性**

局限性包括对高度依赖推理的任务（如数学证明）效果有限，以及在多任务连续适配时可能出现逐步遗忘。

---

## 387. AeroAct: Action-Centered World-Action Models for Language-Conditioned Quadrotor Flight

**arXiv ID:** 2607.14997 | [PDF](https://arxiv.org/pdf/2607.14997v1)

**作者:** Xinhong Zhang `[一作]` (Beijing Institute of Technology), Gang Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于视频扩散 Transformer 的动作中心化世界-动作模型（WAM），直接预测可执行的局部轨迹片段，支持语言条件下的无人机飞行。

**💡 创新点**

创新点在于：①使用未来第一人称视觉预测作为训练时的稠密后果监督，仅在推理时解码动作；②将动作表述为光滑的五阶轨迹片段，兼顾动力学可执行性；③引入自引导机制确保连续轨迹片段的时间一致性；④构建包含 DiffAero 动力学、Isaac Lab 与 3DGS 渲染的混合数据管线，并配套低成本手持采集设备，实现仿真与真实数据的无缝对齐。

**🔧 技术方法**

采用 Wan 1.3B 视频扩散 Transformer、文本编码器、视频 VAE；动作编码器与解码器为 MLP；自引导采样器；光滑轨迹生成（五阶多项式）；离线离散 MPC 轨迹跟踪；硬件包括 Intel RealSense 摄像头、Radxa ROCK 5C 上位机等。

**📊 数据集**

数据集包括：① 900K 条基于 DiffAero、Isaac Lab 与 3DGS 生成的仿真视频剪辑（约 320M 帧，涵盖跟踪、对象到达等场景）；② 858 条真实手持设备采集的轨迹（约 332K 帧，约 3 小时）。

**📈 对比分析**

与仅使用单帧视觉上下文或离散动作的基线相比，使用 9 帧视觉历史后追踪任务成功率从 20% 提升至 100%，搜索任务同样提升至 100%；同时碰撞率降至 0%。在混合仿真与真实数据微调后，性能基本保持不变，表明模型具备良好的跨域泛化能力。

**⚠️ 局限性**

局限性包括：仅在短期室内飞行实验中验证；推理需要离线 GPU，无法在飞行器本机实时运行；尚未验证复杂多子目标指令、长时记忆或极端动态飞行的鲁棒性。

---

## 388. From Draft to Draft-Free: One-Step Video Object Removal via Privileged Distillation and Fast Planting

**arXiv ID:** 2607.14976 | [PDF](https://arxiv.org/pdf/2607.14976v1)

**作者:** Zizhao Chen `[一作]` (Xi'an Jiaotong University), Mengmeng Wang `[通讯]` (Zhejiang University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个从草稿指导到草稿无依赖的单步视频物体去除框架D2DF，能够在不使用外部草稿的情况下实时完成背景重建。

**💡 创新点**

创新点包括：①Prior-Privileged Consistency Distillation (PPCD)，通过在教师模型中注入真值草稿稳定训练轨迹；②Self-Guided Fast Planting (SGFP)，在潜在空间自监督生成伪草稿，消除对外部草稿的依赖；③将三阶段蒸馏过程从教师-学生到草稿无关的单步生成，显著提升速度与质量。

**🔧 技术方法**

采用的核心技术有：潜在扩散模型、Consistency Distillation、PPCD、Temporal Masked Transformer（TMT）构建的SGFP模块以及Transformer层的轻量级设计。

**📊 数据集**

主要使用的实验数据集为RORD、ROVI和VPLM进行训练与评估，并在DAVIS、YouTube‑VOS及Camera‑Bench上做零样本验证。

**📈 对比分析**

在RORD/ROVI/VPLM三大基准上，D2DF‑DG和D2DF‑DF在PSNR/SSIM/LPIPS等指标均超越目前所有光流、Transformer及扩散式方法；在推理时间上，D2DF‑DF以约1秒完成25帧生成，速度比ROSE快约40倍。

**⚠️ 局限性**

局限性包括：单步去噪在某些高复杂遮挡或细节重建时仍会出现轻度模糊；对极端遮挡的恢复受限；伪草稿质量较粗糙，可能影响极细节重建。

---

## 389. CFM-Bench: A Unified Multi-Domain, Multi-Task Benchmark for Channel Foundation Models

**arXiv ID:** 2607.14975 | [PDF](https://arxiv.org/pdf/2607.14975v1)

**作者:** Yuan Gao `[一作]` (Shanghai University), Shugong Xu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一的多域、多任务基准CFM‑Bench，用来公平评估通道基础模型（CFM）在不同数据源和任务上的迁移能力。

**💡 创新点**

创新点在于：①统一的数据划分与严格的测试隔离，消除预训练泄漏；②覆盖统计仿真、射线追踪、现场测量和多模同步驾驶场景的六个通道配置；③设计了六类任务（PHY、RAN、ISAC）并给出统一指标；④提供源特定接口与完整的任务说明，促进跨模型、跨域对比。

**🔧 技术方法**

使用了多种通道生成与测量平台（Sionna、DeepMIMO、Wireless InSite、Sionna RT、DICHASUS、MaMIMO‑UAV、Multimodal‑Wireless），以及自定义的评估脚本、指标（NMSE、SGCS、Top‑k、F1、3D误差等）。

**📊 数据集**

数据集包括：S1（3GPP统计仿真）、R1（DeepMIMO 60 GHz射线追踪）、R2（MOCSID校园射线追踪）、E1（DICHASUS工业室内测量）、E2（MaMIMO‑UAV空地测量）、M1（多模CARLA场景）。

**📈 对比分析**

比较方法要求：①所有模型在官方训练集上微调；②验证集可用于调参但不可用于训练；③测试集仅用于最终推断；④提交必须说明所有使用的数据来源并标注是否泄漏。性能表现：不同模型在各任务/域上表现差异显著，未给出具体数值，但说明 benchmark 能区分预训练模型与专用模型的泛化差异。

**⚠️ 局限性**

局限性：①仅选取单一配置的六个来源，未覆盖全部频率、天线、天气等变异；②域间样本、标注密度、硬件误差不等，跨域直接平均不可靠；③测试集仍可能被未声明的外部数据泄漏；④部分任务在某些域不可用，导致评估不完整；⑤对复杂多模输入的模型设计要求高，易产生偏差。

---

## 390. Stitch-Inferencer: Enhance Endoscopic Video Segmentation and Tracking via Panoramic Reconstruction

**arXiv ID:** 2607.14968 | [PDF](https://arxiv.org/pdf/2607.14968v1)

**作者:** Shunsuke Kikuchi `[一作]` (Jmees Inc), Hiroki Matsuzaki `[通讯]` (Jmees Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种实时、模型无关的推理框架Stitch-Inferencer，通过在图像空间拼接历史帧构建全景视图，从而在推理时为现有分割与跟踪模型提供长时上下文。

**💡 创新点**

创新点在于：① 用显式图像全景取代隐式特征记忆；② 通过关键点匹配与单应矩阵实现在线拼接；③ 将全景ROI直接投影回当前帧，使得任何单帧模型可在推理时受益而无需再训练。

**🔧 技术方法**

关键技术包括：图像无效区域掩码（工具、边框、入口口），ALIKED+LightGlue关键点匹配与加权DLT单应估计，累积单应矩阵与视图投影，拉普拉斯方差判断帧质量，α‑混合平滑拼接，以及ROI裁剪与投影回投影；实现以TensorRT等轻量化方式保持>60 FPS。

**📊 数据集**

主要数据集：CholecSeg8k（解剖分割）、Cholec80、Cholec80-port、STIR24（点跟踪）与SurgT（盒子跟踪）等；训练与评估使用统一的单帧基础模型（UNet++、UPerNet、SegFormer、DeepLabV3+、HRNet-seg）与公开的视频基准。

**📈 对比分析**

与原始单帧模型对比，Stitch-Inferencer在Dice、Peri-Dice、FPS、p95延迟等指标上平均提升0.05–0.02点，速度保持45–52 FPS；在跟踪任务上各跟踪器均提升δ‑avg、EAO、误差指标；对比视频专用基准（如TMANet、DTERN、SAM3等）也显示出一致的鲁棒性改进。

**⚠️ 局限性**

局限性包括：① 依赖平面单应估计，对强非平面变形或深度变化易失效；② 需要高质量关键点匹配，匹配失败时需重新初始化；③ 在极端摄像机运动或工具占满视野时全景质量下降；④ 目前评估受限于公开数据集的时长与动态范围，尚未验证在长时间手术中的持续性能。

---

## 391. Contextualized Early Detection of Online Firestorms: A Sequential LLM-Based Approach

**arXiv ID:** 2607.14957 | [PDF](https://arxiv.org/pdf/2607.14957v1)

**作者:** Besim Shala `[一作]` (University of Applied Sciences Munich), Martin Häusl `[通讯]` (University of Applied Sciences Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于大型语言模型的在线火山检测系统，支持完整线程分类和实时早期预警。

**💡 创新点**

创新点在于将LLM作为判定器应用于序列化线程监控，利用情感、升级级别和贡献者计数等理论驱动的指标实现可校准的阈值决策。

**🔧 技术方法**

采用了GPT‑4o mini的prompt‑based 评估，使用分块处理与层次合并进行完整线程分类，滑动窗口与校准阈值实现早期预警。

**📊 数据集**

使用了200条平衡的Reddit讨论线程数据集（100火山，100非火山），并对120条未参与校准的线程进行早期预警测试。

**📈 对比分析**

与传统基于体量或预定义特征的检测方法相比，整体模式达到F1 0.91、准确率0.915；早期预警模式召回率0.98、平均在8.56条评论后发出警报，误报率约22%。

**⚠️ 局限性**

局限包括仅在Reddit平台验证、单注释者标签、缺乏精确升级起点、对单一LLM和提示敏感、仅文本特征且未考虑网络结构等。

---

## 392. A Comprehensive History of $μ$CRL and mCRL2

**arXiv ID:** 2607.14956 | [PDF](https://arxiv.org/pdf/2607.14956v1)

**作者:** Jan Friso Groote `[一作]` (Eindhoven University of Technology), Erik P. de Vink `[通讯]` (Eindhoven University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了从μCRL到mCRL2的历史与发展，阐述了两种基于过程代数、等式重写和模态μ-演算的形式化语言与工具集的诞生、演进与应用。

**💡 创新点**

创新点在于系统性记录了两代语言的演化脉络、关键技术改进（如多动作通信、可变参数、标准数据类型、模态μ-演算的引入）、工具链从单机到并行、符号化状态空间生成的突破，以及对实际系统验证与性能评估的经验总结。

**🔧 技术方法**

所用技术包括：过程代数（ACP、CCS）、抽象数据类型与等式重写、模态μ-演算、参数化布尔方程系统（PBES）、符号化模型检测（BDD、LTSmin）、并行ATerm库、状态空间生成与τ-共形、可视化工具、概率/时间扩展（PRES）等。

**📊 数据集**

主要使用的实验数据集为行业和学术上的典型协议与系统：例如滑动窗口协议、FlexRay、Firewire、Pacemaker、ATLAS、火车信号等，此外通过大型模型（10^10 状态）展示符号化方法的规模化能力。

**📈 对比分析**

对比方法侧重于手工验证、自动状态空间生成、τ-共形、行为等价简化以及PBES求解；性能方面，单机生成10^5–10^6 状态/秒，符号化方法可处理10^10 以上状态；然而多进程内存分布、τ-共形适用性有限，需进一步提升符号化压缩与并行求解效率。

**⚠️ 局限性**

限制主要包括：对连续概率与非确定性选择的语义尚未完全成熟；符号化方法对参数独立性、共享数据等仍需改进；并行ATerm库与工具链的性能瓶颈历史遗留；缺乏统一的“工业级”行为建模标准，导致模型规范化与状态空间规模控制仍是难题。

---

## 393. A Queueing-Stability Criterion for Causal IPD-QIM Network Flow Watermarking

**arXiv ID:** 2607.14954 | [PDF](https://arxiv.org/pdf/2607.14954v1)

**作者:** Jiuxiang Cai `[一作]`, Guangjie Liu `[通讯]` (Nanjing University of Information Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对因果IPD‑QIM网络流水印的排队稳定性判据，并给出可行的量化步长区间，结合仿真与真实流量验证。

**💡 创新点**

通过分析固定晶格随机比特QIM注入的状态依赖性，推导出从忙状态平均注入Δ/4决定稳定性的判据，并将其推广至非独立、突发流量，形成两侧约束的完整操作窗口。

**🔧 技术方法**

采用Lindley递推、Foster–Lyapunov漂移、Loynes稳定性理论以及马尔可夫链分析进行理论推导，并结合蒙特卡罗仿真与实验验证。

**📊 数据集**

在CIC/UNB公开的ISCXTor2016与ISCXVPN2016数据集上，使用四类应用级IPD流（Tor浏览、Tor文件传输、VPN音频、NonTor音频）进行实测。

**📈 对比分析**

通过比较理论窗口与实测的steady dwell与错误率，证明判据能精确预测稳定边界；在安全区段验证收敛，接近边界时出现急剧发散，验证其准确性。

**⚠️ 局限性**

仅适用于固定双晶格、equiprobable随机比特的QIM，忽略符号抖动、非等概率编码或动态相位等改进；假设信道抖动为高斯；仅给出稳定性阈值，未对溢出概率或实时控制做深入分析。

---

## 394. A Minimal Interpretable Architecture for Zero-Shot Reconstruction of Dynamical Systems

**arXiv ID:** 2607.14937 | [PDF](https://arxiv.org/pdf/2607.14937v1)

**作者:** Christoph Jürgen Hemmer `[一作]` (Central Institute of Mental Health), Daniel Durstewitz `[通讯]` (Central Institute of Mental Health)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对现有的 DynaMix 预训练模型进行逐步简化，构建出最小化的两参数（可进一步降至一参数）近邻仿射递归模型 DynaBase，并在该框架下实现零射预测的动态系统重建。

**💡 创新点**

提出了一个极简、可解析的 DSR 模型（DynaBase），揭示了在动态系统零射预测中仅需邻域查找和线性组合即可实现复杂动力学；同时阐明了不同训练目标（短期误差 vs 长期动力学）如何引导模型收敛到不同的参数区域，从而对现有大模型的内部机制提供了理论解释。

**🔧 技术方法**

使用了：1) 逐层消融与近似（将混合专家网络降为单一线性递归）；2) 最近邻搜索与 Voronoi 切片；3) 线性最小二乘回归与基于 DSR 目标的网格搜索；4) 控制论训练技巧（教师强制、泛化等）；5) 传统 DSR 方法（SINDy、reservoir、Neural ODE）和基础模型（Chronos、TimesFM、Panda）作为对照。

**📊 数据集**

在 34 个三维周期或混沌动力系统上预训练 6×10⁵ 条轨迹，随后在 54 个测试系统（包含周期、混沌、平衡点）上评估；上下文长度固定为 2000 步。

**📈 对比分析**

对比指标包括 D_stsp（几何不一致）、D_H（长期时间不一致）和 MASE（10 步短期误差）。DynaBase 在使用 DSR 目标训练时的 D_stsp 与 D_H 分别优于或匹配大型基础模型和手工训练的 DSR 模型；使用一阶 MSE 训练时在短期 MASE 上更优，但在长期指标上略逊。零射模式下，固定 α≈1.01 的 DynaBase 亦可达到与手工训练模型相近的性能。

**⚠️ 局限性**

作为极简模型，DynaBase 在表达复杂向量场、捕获精细几何特征（如拉普拉斯谱）上存在局限；对噪声鲁棒性未系统评估；并且对某些非周期、非混沌的动力学拓扑可能缺乏足够的拟合能力，因而目前主要作为研究工具。

---

## 395. Authoring Narrative Visualization in Motion: Visual Storytelling in Swimming Videos

**arXiv ID:** 2607.14924 | [PDF](https://arxiv.org/pdf/2607.14924v1)

**作者:** Junhao Zhao `[一作]` (Xi'an Jiaotong-Liverpool University), Lijie Yao `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究如何支持在运动视频中创作叙事可视化，尤其聚焦于游泳比赛，提出了自动多模态数据准备管道与技术探针工具，让创作者能够在视频中同步、布局、过渡多种视图与可视化层，形成完整的赛事实时叙事。

**💡 创新点**

创新点在于：①将视频、音频、赛事信息三源自动融合成结构化、可视化直接使用的数据；②设计了支持视图切换、动画过渡、时序调度的交互式技术探针，帮助作者在时间轴上协调全局视图、跟踪视图和对比视图；③通过观察专业广播并提炼叙事模式，为工具提供实践灵感。

**🔧 技术方法**

使用的关键技术包括：视频分割与跟踪（SAM3）、音频转写与LLM提示（WhisperX+GPT-5.4）用于提取事件洞察、同源变换模拟摄像机视角、GPU加速渲染（PixiJS）、层级/时序调度框架、以及自动元数据检索流水线。

**📊 数据集**

数据集为2024年巴黎奥运游泳比赛录像（自有采集）、官方或主流媒体的解说音频、比赛名称作为唯一标识，并利用官方赛事页面获取运动员、记录等元数据信息；还使用多场次（奥运、世界锦标赛等）进行验证。

**📈 对比分析**

通过对9名经验丰富创作者的实验评估，使用Likert量表、定性访谈与作品观察来衡量工具表现。结果显示功能覆盖率和叙事支持得分高于中立水平，易用性略低；在可视化叙事效果上与专业广播相比，工具能显著降低创作时间并保持清晰度。

**⚠️ 局限性**

局限性包括：①功能与易用之间的权衡——缺少高级剪辑特性（遮罩、键盘快捷键、层级堆叠）导致部分用户学习成本高；②动态布局与摄像机运动的挑战——在真实广播场景下摄像机移动与运动员重叠时可视化如何保持关联与可读性尚未完全解决；③对可视化设计的依赖——虽提供泳道专用编码，但自定义设计仍需较高可视化素养。

---

## 396. Confidence-based Ranking with Adaptive Sampling for Noisy Black-Box Optimisation

**arXiv ID:** 2607.14936 | [PDF](https://arxiv.org/pdf/2607.14936v1)

**作者:** Enrico Halim `[一作]` (University of New South Wales), Tapabrata Ray `[通讯]` (University of New South Wales)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于置信度排序的自适应采样方法（CR），用于在含噪声的黑盒优化中显式地重新评估候选解，并将更多评估预算分配给更有前景的解；该方法集成到CMA-ES和传统GA中，形成CR-CMA-ES和CR-EA。

**💡 创新点**

创新点在于：①将置信度（Welch's t‑test）与自适应采样预算相结合，动态调整每一代的评估次数；②通过对与最高排名解等价的候选解进行进一步采样，以提升排名准确度；③提供一种通用模块，可在任何基于排名的优化框架中直接插拔；④系统性评估了异方差噪声环境下的性能，填补了以往仅针对同方差噪声的研究空白。

**🔧 技术方法**

使用的技术包括：Welch's t‑test进行统计显著性检验；OCBA启发式的采样预算分配；自适应采样阈值（γ、β、α）控制；在CMA-ES中实现父代选择；在GA中实现二进制锦标赛选择；以及对评估结果进行后处理（如1,000次重评估）来估计真实目标值。

**📊 数据集**

数据集：10个10维单目标基准函数（5单峰、5多峰），结合5种噪声模型（同方差、适应比例、逆比例、中间比例、正弦比例）和2个噪声强度；此外，还使用了奥林匹斯平台下的两种化学模拟器（COLORS_BOB和COLORS_N9）。

**📈 对比分析**

比较方法：与Opl-CMA-ES、LRA-CMA-ES、RA-CMA-ES、以及3种显式采样变体（SAVG、SAVGD、SAVGI）以及无界/有界两种版本进行对比；使用性能曲线（performance profile）、Wilcoxon秩和检验以及中位数收敛曲线评估；结果显示CR-CMA-ES/CR-EA在大多数问题上取得了最快收敛、最小评估成本、最高成功率，尤其在异方差噪声和实际化学模拟器上表现最优。

**⚠️ 局限性**

局限性：①依赖显式采样，评估成本仍高，尤其在评估昂贵的真实实验或大规模仿真时；②对正态噪声的假设在某些应用中不成立，需进一步推广到非高斯噪声；③参数（γ、β、α、C等）的选择仍需经验调优；④在多目标或约束优化场景下尚未验证。

---

## 397. Multi-Axis Max@K Reinforcement Learning for Representative Diversity in Text-to-Image Generation

**arXiv ID:** 2607.14962 | [PDF](https://arxiv.org/pdf/2607.14962v1)

**作者:** Ku Onoda `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了多轴max@K目标，用于在扩散式文本到图像模型中通过集体强化学习提升对预定义目标模式的覆盖率，主要在SD3.5‑M模型上完成实验；

**💡 创新点**

创新点在于将多轴奖励先在样本集合中按类别取最大值，再求和，从而实现不同样本分别对不同互斥模式的代表性信用分配，显著提升了多模式覆盖与公平性；

**🔧 技术方法**

技术手段包括基于GRPO的扩散RL框架、set-level max@K目标、EI+L2O优势估计、CLIP提示集与FairFace等自动评估器的奖励回归，以及对多轴奖励的离散化与标准化；

**📊 数据集**

数据集主要包括SD3.5‑M生成的图像，Pick‑a‑Pic提示、200个职业提示、1080个中性人像训练提示；实验还使用合成2D高斯混合、七种像素规则色彩类别以及感知种族与性别的自动标签；

**📈 对比分析**

与单一标量奖励、GRPO(k=1)、Fair‑GRPO计数信用、FairImagen和Weak Guidance等基线比较，multi‑axis max@K在三种评估器上分别提升了约0.18–0.26的公平性分数，同时保持或略优于基线的图像质量与文本对齐指标；

**⚠️ 局限性**

局限性包括需要预先定义且可测量的奖励轴，无法产生未出现的模式；过度依赖自动评估器可能导致过拟合；仅针对固定目标集，无法覆盖更广泛的公平性维度；缺乏人工评估与对模型泛化能力的进一步验证。

---

## 398. CODA: Algorithm-Hardware Co-design for Edge Video Diffusion via NMP-Enabled Compute-Cache Operator Disaggregation

**arXiv ID:** 2607.14908 | [PDF](https://arxiv.org/pdf/2607.14908v1)

**作者:** Yuanpeng Zhang `[一作]` (Peking University), Guangyu Sun `[通讯]` (Peking University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种面向边缘视频扩散模型的端到端加速架构CODA，结合跨时钟缓存（CTC）、近内存处理（NMP）与双分支重叠执行，实现高效的计算-缓存操作解耦；

**💡 创新点**

创新点在于：①硬件友好的混合硬件感知缓存调度器，将碎片化的缓存操作重新组织为coalesced段；②轻量化DIMM‑NMP子系统，将内存密集的缓存操作迁移至内存侧；③利用CFG两支独立性实现缓存侧与GPU计算的重叠；④通过动态运行时调整（DRA）保障罕见情况的质量。

**🔧 技术方法**

技术包括跨时钟缓存（CTC）、近内存处理（NMP）、硬件感知调度、CFG-Interleaved Pipelining、动态运行时调整，以及基于Ramulator、Synopsys Design Compiler的系统仿真与RTL实现。

**📊 数据集**

评测使用多种公开视频扩散模型（Latte、Open‑Sora、Open‑Sora Plan、Wan 2.1、HunyuanVideo、Vchitect‑2.0、CogVideoX‑5B）以及VBench数据集的100条文本提示。

**📈 对比分析**

与基线Vanilla‑GPU、Host‑Offload、NaiveNMP等对比，CODA在多种分辨率和时长下实现了最高的速度提升（最高1.80×）和能效提升（最高1.74×），同时在VBench、PSNR、SSIM、LPIPS等多维质量指标上与保守缓存策略相当，且优于激进缓存配置。

**⚠️ 局限性**

局限性包括：对极端大视频仍可能需要大量内存，DRA在罕见样本中会触发，轻量化NMP仅支持固定的缓存操作模式，且该架构目前针对视频扩散模型，若迁移到文本扩散模型需重新设计调度与重叠机制。

---

## 399. FirmPilot: Evidence-Guided Multi-Agent Environment Recovery for IoT Firmware Rehosting

**arXiv ID:** 2607.14903 | [PDF](https://arxiv.org/pdf/2607.14903v1)

**作者:** Yanbing Shen `[一作]` (Zhejiang University), Haitao Xu `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于证据引导的多智能体框架 FirmPilot，用于在 QEMU 环境中恢复 IoT 固件的启动、持久化状态和网络配置，以实现可重复、可扩展的固件重托管。

**💡 创新点**

创新点在于将固件重托管视为证据索引的环境重建问题，并通过检索、规划、文件、NVRAM 和网络五个专门智能体协同工作，限定 LLM 的输出为可验证的、可回放的环境转移，从而大幅提升固件可达性和后续安全分析的可用性。

**🔧 技术方法**

使用了检索增强生成（Retrieval-Augmented Generation）与 LLM（DeepSeek V4 Flash）结合的多智能体架构，包括搜索代理、规划代理、文件代理、NVRAM 代理和网络代理，并通过 QEMU 模拟、固定探测器（Ping/Web 服务）和类型化接口进行验证。

**📊 数据集**

主要数据集为 LFwC（10,033 份可执行固件）和公开基准集（1,122 份），两者覆盖多厂商、多架构的 IoT 固件。

**📈 对比分析**

与基线 FirmAE（模板驱动）对比，FirmPilot 在 LFwC 上将 Web 服务可达率从 25.49% 提升至 52.39%，网络可达率从 39.30% 提升至 71.93%；在公开基准上 Web 可达率从 79.4% 提升至 82.2%；同时在下游工作（RouterSploit、协议感知模糊测试）中显著增加可操作的服务表面。

**⚠️ 局限性**

主要局限包括：仍未处理外设/硬件 I/O 模拟、极端固件依赖（如缺失库、证书）导致的后期失败；依赖 LLM 的检索与规划质量；在极端长尾固件中可能需要更多迭代或更精细的证据；系统对异常或未覆盖的固件仍有未解决的可达性瓶颈。

---

## 400. The Distributed Open-Source Vulnerability Ecosystem

**arXiv ID:** 2607.14900 | [PDF](https://arxiv.org/pdf/2607.14900v1)

**作者:** Peter Mandl `[一作]` (Munich University of Applied Sciences), Paul Mandl `[通讯]` (Findustrial GmbH)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过文献综述和概念分析，构建了开源漏洞生态系统的整体信息流框架，系统阐述了漏洞信息从发现、标准化、整合、丰富到在软件组件中的映射与解释的全过程，并指出导致漏洞扫描器结果差异的主要原因：信息源异构、身份与版本模型差异、时间演化以及上下文依赖评估。

**💡 创新点**

创新点在于将漏洞管理视为分布式信息交流与转换的连续过程，首次系统性识别并归纳了多层面因素（信息源、模型、时间、上下文）如何交互产生差异，提出了对生态整体而非单一数据库或工具的研究视角。

**🔧 技术方法**

本文主要采用文献综述与案例分析（以CVE-2021-44228为例）为技术手段，未涉及实验实现或算法开发。

**📊 数据集**

讨论的主要数据来源包括公开漏洞数据库与信息系统：CVE、NVD、OSV、GitHub Advisory、PSIRT/供应商公告、EUVD、JVN、CNVD、CNNVD等，但并未收集或使用特定数据集进行实验。

**📈 对比分析**

由于本研究为概念性框架，未进行实验比较；论文强调需要可复现的评估方法，并指出当前工具差异主要来源于信息来源与上下文决策，而非单一技术实现。

**⚠️ 局限性**

局限性：缺乏实证验证与量化评估；未构建统一的基准数据集；框架仍需进一步验证、完善，并在实践中测试其对工具结果解释的有效性。

---

## 401. Innocuous-Seeming Data, Latent Ideology: Ideological Generalisation in Finetuned LLMs

**arXiv ID:** 2607.14888 | [PDF](https://arxiv.org/pdf/2607.14888v1)

**作者:** Robert Graham `[一作]` (Independent), Yariv Barsheshat `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究通过在极窄主题的数据集（如经济学、音乐偏好、食品安全等）上微调大型语言模型，发现即使是无害的、符合审核标准的数据也能导致模型在未见过的领域出现显著的意识形态偏移；

**💡 创新点**

创新点在于首次系统地识别并量化了“意识形态泛化”现象，提出了跨域泛化广度与放大度两种评估指标，并证明微调会把模型推向更极端的偏见；

**🔧 技术方法**

技术方法包括在GPT‑4.1上进行4轮微调、在Gemma‑3上使用LoRA实现可复现的开源实验，以及利用LLM判别器和强制选择评估来量化偏移；

**📊 数据集**

使用的主要数据集为：左右倾向的经济学问答、左右音乐品味问答、科学与伪科学的食品安全问答，以及业务、HR与健康营销的应用型问答；

**📈 对比分析**

与基线模型和少量提示（few‑shot）相比，微调模型在跨域任务上表现出更大的偏移幅度，且在GSM8K等通用能力评测中的分数保持在±1pp，说明偏移不牺牲整体性能；

**⚠️ 局限性**

局限性包括评估指标依赖手工挑选的类别，LLM判别器可能带来偏见，仅检验了两条意识形态轴，且仅尝试了单一的泛化抑制方法（数据混合）并仅在监督微调场景下进行实验。

---

## 402. StructureClaw: Traceable LLM Agents and an Executable Benchmark for Structural Engineering Workflows

**arXiv ID:** 2607.14896 | [PDF](https://arxiv.org/pdf/2607.14896v1)

**作者:** Sizhong Qin `[一作]`, Xinzheng Lu `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 StructureClaw 结构工程工作台和对应的可执行基准 StructureClaw-Bench，使 LLM 代理能够在完成结构工程任务时生成完整、可追溯的证据链，而非仅返回答案。

**💡 创新点**

通过将工作流程拆解为可观测的技能、工具、共享工件与验证记录，构建可解释、可验证的结构工程代理，并用基准量化整个链路的成功率，显著提升了自动化水平。

**🔧 技术方法**

结合 ReAct 推理循环、OpenSeesPy 后端、结构模型协议、工具驱动的验证与代码检查，以及多模态输入解析技术。

**📊 数据集**

使用 150 个手工设计的结构工程场景（含中英文本、图像和 DXF 等多模态输入）构建 Benchmark。

**📈 对比分析**

以成功率（Success Rate）为主指标，对比自动完整流程与仅使用通用结构技能的 “generic‑only” 模式，发现自动模式平均成功率从 56.8% 提升至 88.6%（提升约 31.8个百分点）。

**⚠️ 局限性**

仍存在无效数值处理、模型重构一致性等局部瓶颈；某些结构类型（如连续梁）出现回归；基准未覆盖大规模工程、更多后端以及工程师手工验证。

---

## 403. On Success and Simplicity: A Second Look at Transferable Vision-Language Attack Pipeline

**arXiv ID:** 2607.14974 | [PDF](https://arxiv.org/pdf/2607.14974v1)

**作者:** Yuchen Ren `[一作]` (Xi'an Jiaotong University), Chao Shen `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种简化的视觉‑语言对抗攻击流程SimVLA，通过改进跨模态词识别、文本语义抽象和去除冗余阶段来实现更高效的可迁移攻击

**💡 创新点**

创新点在于：①将跨模态信息用于词识别而非单纯文本，②在生成对抗图像前剔除与图像无关的文本词，③去掉传统三阶段中的第三阶段，显著提升攻击迁移性与效率

**🔧 技术方法**

使用BERT-Attack、PGD、MIM‑PGIA等现有优化方法；核心技术为跨模态KL距离用于词识别与语义抽象，图像嵌入中心构造与随机噪声采样

**📊 数据集**

在Flickr30k、MSCOCO、SNLI‑VE、RefCOCO+四大数据集上进行文本检索、图像检索、视觉蕴含与视觉定位任务的评估

**📈 对比分析**

与PGD、BERT‑Attack、Sep‑Attack、Co‑Attack、SGA、DRA、SA‑AET等六种基线对比，SimVLA在R@1/R@10等指标上提升8–15%，并且计算时间约为现有SOTA的35%~50%，VRAM使用率约为46%~60%

**⚠️ 局限性**

局限性包括：仅在公开 VLPM/MLLM 上验证，缺少对更大规模模型或实际 API 的广泛评估；对极端攻击约束（如更高噪声、更多单词替换）效果尚未彻底探究

---

## 404. Latent Trajectory Discrimination for AI-Generated Text Detection

**arXiv ID:** 2607.14967 | [PDF](https://arxiv.org/pdf/2607.14967v1)

**作者:** Gianluca Bonifazi `[一作]` (Polytechnic University of Marche), Luca Virgili `[通讯]` (Polytechnic University of Marche)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于语义轨迹差分的 AI 生成文本检测框架 GTCL，将文档视为嵌入空间中的动态轨迹，利用滑动窗口分段并对连续窗口嵌入差分做对比学习；

**💡 创新点**

核心创新在于把检测问题从静态文档表示转化为轨迹差分的区分任务，使用时间对齐的相似度和监督对比学习挖掘自回归生成过程的几何规律；

**🔧 技术方法**

使用预训练句子嵌入器（Nomic Embed Text v1.5）、Transformer 投影模块、监督对比学习与 k‑NN 分类器，并采用滑动窗口、差分编码、时间对齐相似度等技术；

**📊 数据集**

在 RAID、NYT‑AI、Reviews（OpenReview 科研论文评论）三个数据集上进行实验；

**📈 对比分析**

与 Desklib、Binoculars、ModernBERT、RADAR、DeTeCtive、TMR 等基线比较，GTCL 在所有数据集上均实现最高准确率、F1 分数，尤其在人类文本检测上表现显著提升；

**⚠️ 局限性**

局限性包括：需要手动设定窗口长度、步长、窗口数等超参数；对不同语言或非自回归生成模型（如扩散模型）可能需进一步改造；

---

## 405. Introspective Attention Modulation for Safe Text-to-Image Generation

**arXiv ID:** 2607.14945 | [PDF](https://arxiv.org/pdf/2607.14945v1)

**作者:** Basim Azam `[一作]` (University of Melbourne), Naveed Akhtar `[通讯]` (University of Melbourne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在推理阶段通过自省注意力调节来抑制文本到图像生成中的不安全内容的方法，保持图像质量与语义对齐；

**💡 创新点**

创新点在于使用注意力空间的实时调节（而非模型参数或外部过滤器），通过学习稀疏逻辑回归检测不安全注意力并在每一步优化注意力分布，从而实现无训练、无模型改动的安全保障；

**🔧 技术方法**

核心技术包括：基于rectified flow的FLUX/Flux类Transformer生成器、双流/单流注意力提取、稀疏逻辑回归安全向量、约束优化（兼顾语义一致、危险抑制与分布保持）、投影梯度下降以及自适应 introspection ratio 控制；

**📊 数据集**

主要使用I2P、SneakyPrompts、MMA-Diffusion、UnlearnDiffAtk等安全基准数据集，以及MS-COCO验证集进行质量评估；

**📈 对比分析**

与现有概念抹除（UCE、ESD、EraseAnything、FlowEdit、MCE）以及安全编辑方法对比，实验显示在NR、VLM、CLIP三项指标上均优于基线和所有基线方法，并在安全-质量Pareto前沿上实现双提升；

**⚠️ 局限性**

局限性包括：依赖预训练的安全概念向量，可能在极端对抗式提示（如SneakyPrompts）下仍有提升空间；对潜在攻击的鲁棒性有限；并且需要信任安全操作，若被破坏可能导致过度审查或反向攻击。

---

## 406. VideoChat3: Fully Open Video MLLM for Efficient and Generalist Video Understanding

**arXiv ID:** 2607.14935 | [PDF](https://arxiv.org/pdf/2607.14935v1)

**作者:** Xinhao Li `[一作]` (Nanjing University), Limin Wang `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了VideoChat3，一款完全开源的高效通用视频多模态大语言模型，支持短视频、长视频与实时流视频的理解与交互。

**💡 创新点**

创新点在于：①提出Inflated 3D Vision Transformer（I3D‑ViT），在视觉分词阶段就完成局部时空自注意力与时间池化，实现高效压缩；②引入Adaptive Frame Resolution，依据视频内容重要性动态调节分辨率；③构建三大可扩展数据管线（学术、长视频、在线流），通过语义重写与验证提升数据质量。

**🔧 技术方法**

技术手段包括：I3D‑ViT视觉分词器、基于自注意力的时空压缩、分段式时间池化、动态帧分辨率控制、分阶段训练（视觉预训练→对齐→指令微调→长/流视频调优）、多任务监督与状态转移掩码。

**📊 数据集**

使用的数据集为VideoChat3‑Academic2M（改写后的学术视频+QA）、VideoChat3‑LV116K（长视频合成与分段注释）、VideoChat3‑OL617K（在线流视频与时序回答），并融合公开数据如LLaVA、WebVid、S‑MiT、VCS、TVQA等。

**📈 对比分析**

在多项视频理解基准（MotionBench、TempCompass、TimeLens、VUE‑TR、Video‑MME、OVBench、StreamingBench、OVO‑Bench等）与同等规模或更大开放模型（Qwen3‑VL‑4B、Molmo2‑4B、VideoChat‑Flash‑7B、Gemini‑2.5 等）对比，4B版VideoChat3在绝大多数指标上取得领先，且在推理延迟、FLOPs 与显存占用上表现更优。

**⚠️ 局限性**

局限性包括：对极大规模模型（10B+）的可扩展性尚未充分验证；在极低帧率或极高分辨率的真实视频流中的泛化仍需进一步评估；部分细粒度情感、对话上下文推理任务的性能与商业模型仍有差距。

---

## 407. Still image and spatial-temporal tomato data enabling detection, segmentation, tracking, and video-instance segmentation using strong and weak labels

**arXiv ID:** 2607.14934 | [PDF](https://arxiv.org/pdf/2607.14934v1)

**作者:** Michael Halstead `[一作]` (University of Bonn), Chris McCool `[通讯]` (University of Bonn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文发布了两个新型番茄数据集——静态图像集BUTom21和时空序列集BUTom-ST21，并提供完整注释、摄像姿态和伪标签，供视觉感知研究使用。

**💡 创新点**

创新点在于：①首次针对番茄在商业化玻璃温室环境中从机器人平台获取大规模、分级成熟度的像素级标注；②结合NeRF与结构光SLAM生成一致的轨迹ID，实现高质量视频实例分割与多目标跟踪数据；③将手工与伪标签相结合的混合标注策略，缓解农业视觉数据稀缺问题。

**🔧 技术方法**

采用的主要技术包括Mask2Former与YOLOv8-Seg的实例分割训练、PAg-NeRF基于NeRF的伪标签生成、COLMAP与HLoc进行相机姿态重建、CTVIS视频实例分割框架以及多种跟踪器（ByteTrack、Halstead、OCSort、PAg-NeRF）。

**📊 数据集**

使用的数据集为：①BUTom21（123/72/98张图，RGB+深度，COCO格式标注）；②BUTom-ST21（7749/4536/1386帧，含伪标签与手修正评价集，Tracklet ID + 8×深度+RGB）。

**📈 对比分析**

通过在静态图像基准上评估Mask2Former与YOLOv8的mAP、AP50/75及不同尺寸的表现；在时空序列上利用CTVIS和PAg-NeRF进行视频实例分割和多目标跟踪，结果显示Mask2Former在分割质量上优于YOLOv8，但YOLOv8在检测精度和跟踪鲁棒性更强；总体性能受伪标签噪声和“tiny”目标难检的影响。

**⚠️ 局限性**

主要限制包括：伪标签中存在多目标合并、轨迹ID切换、缺失小目标导致的低AP；tiny对象几乎无法检测；深度过滤与姿态重建不完美，导致部分序列无有效轨迹；整体数据集对超小目标和遮挡处理仍不够完善。

---

## 408. Random Access to LZ-End: Faster and Deterministic

**arXiv ID:** 2607.14923 | [PDF](https://arxiv.org/pdf/2607.14923v1)

**作者:** Itai Boneh `[一作]` (University of Wrocław), Paweł Gawrychowski `[通讯]` (University of Wrocław)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文提出了一种基于 LZ‑End 解析的压缩随机存取结构，在仅使用 O(z) 空间的前提下实现了字符访问与子串提取。通过构造确定性、可构造 O(z log²(n/z)) 时间的索引，查询时间被降低到 O(log²(n/z))，显著优于之前的随机化 O(log⁴n·loglog n) 方案。

**💡 创新点**

创新点包括：
1) 完全消除随机化，提供确定性构造与查询；
2) 通过“好映射”（good mapping）与树结构覆盖技术，将多步 J 跳转压缩为 O(log n/z) 次；
3) 利用“短路”（shortcut）机制与层祖先查询快速跳过长序列，进一步将查询时间压缩到 O(log²(n/z))；
4) 同时支持子串提取，并保持相同的空间与构造时间。

**🔧 技术方法**

核心技术手段：
- LZ‑End 解析与指针跳转 J；
- 结构化的树 T_k（以及全局树 T_）捕获“坏跳”序列；
- 采用 halved canonical partition 计算 M(i)；
- 预处理短路信息（log n/z 层）和 δ_ℓ、δ_r 统计；
- 通过层祖先查询快速求和 Δ_L、Δ_R；
- 使用确定性完美哈希实现索引的 O(1) 访问；
- 组合以上技术得到 O(z log²(n/z)) 的构造和 O(log²(n/z)) 的查询。

**📊 数据集**

本文未涉及具体实验数据集；所有结果均为理论分析与构造复杂度。

**📈 对比分析**

与 Kempa‑Saha（随机化）方法相比：
- 空间保持 O(z)；
- 构造时间从 O((n+z)log³n) 降至 O(z log²(n/z))；
- 查询时间从 O(log⁴n·loglog n) 降至 O(log²(n/z))；
- 对子串提取的支持与 LZ‑End 解析长度无关，时间为 O(|substring|+log²(n/z))，与现有 LZ‑77 方案的 O(|substring|+log²(n/z)) 相当。

**⚠️ 局限性**

限制与待解问题：
- 查询时间仍为多项式对数，尚未达到最优的 O(log n / loglog n) 下界；
- 需要先将 LZ‑End 解析中的长短词约束为 |P|≤n/z，虽然可在 O(z) 时间完成，但在实践中可能影响压缩率；
- 结构依赖于对句柄的精确计算（δ、(b,e) 等），实现复杂度较高；
- 对极端分布的字符串（如近似随机字符串）时，z≈n，导致 log²(n/z)=O(log²n)，性能与直接解压相近。

---

## 409. Show Me How You Reason and I'll Tell You Who You Are: Reasoning Graphs for Robust LLM Authorship Attribution

**arXiv ID:** 2607.14905 | [PDF](https://arxiv.org/pdf/2607.14905v1)

**作者:** Zlata Kikteva `[一作]` (University of Passau), Ramon Ruiz-Dolz `[通讯]` (University of Dundee)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种基于推理图的LLM作者归属方法，利用图神经网络对论证结构进行建模；

**💡 创新点**

创新点在于将论证关系图而非表层语言特征作为判别信息，并通过GNN提取结构化推理特征；

**🔧 技术方法**

使用了Graph Convolutional Network、Graph Attention Network、Graph Transformer和GPS等GNN架构，结合Open Argument Mining Framework进行图构建；

**📊 数据集**

构建并公开了新的LLM-OWL-AE数据集，包含8种模型（Gemma、Qwen、Llama、Phi）各两版本的约8,960篇论证性作文，含原文、改写和回译三种攻击版本；

**📈 对比分析**

与Longformer文本分类基线对比，GNN在同版本下对原文的macro‑F1高达0.7以上，且在反射攻击（改写、回译）下下降幅度仅10–20%，比基线低10–50%；在跨版本评估中，GNN提升约19个百分点，表现出更好的泛化；

**⚠️ 局限性**

局限性包括仅评估相同模型家族的版本泛化、论证图构建依赖于现有的推断与关系识别算法，图质量仍有提升空间，且未覆盖不同文本体裁与领域。

---

## 410. OASIS-Map: Object-Level Change Detection in Multi-Session Mapping using Semantic Correspondence Matching

**arXiv ID:** 2607.14899 | [PDF](https://arxiv.org/pdf/2607.14899v1)

**作者:** Haedam Oh `[一作]` (University of Oxford), Maurice Fallon `[通讯]` (University of Oxford)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了OASIS-Map，一种多会话语义地图系统，能够在长时间访视中检测并关联对象级变化。

**💡 创新点**

创新点在于使用密集补丁级语义对应来实现跨会话对象关联，解决部分视角、遮挡和相似实例的难题，并在同一时段内递增地更新地图。

**🔧 技术方法**

采用SAM2进行实例分割、DINOv3提取密集特征、TSDF重建、密集补丁对应、匈牙利匹配及基于相机位姿的多会话SLAM等技术。

**📊 数据集**

在3RScan、Car Park、Market等真实室内外场景上进行评估，其中Car Park用于车辆替换，3RScan用于家具移动。

**📈 对比分析**

与LT-Mapper、ConceptGraphs、Where's-my-glasses等基线相比，OASIS-Map在对象级变化检测和关联上实现了最高F1（如Car Park替换0.783，3RScan静态0.736），在检测准确率和召回率上均优于对比方法。

**⚠️ 局限性**

限制在于对稀疏、低帧率场景下的实时性仍有挑战，且对大规模多会话的长期持续映射需要进一步扩展；对纹理、形变等非结构变化未覆盖。

---

## 411. Selectivity Drives Efficiency: Dataset Pruning for Visual Place Recognition

**arXiv ID:** 2607.14897 | [PDF](https://arxiv.org/pdf/2607.14897v1)

**作者:** Tong Jin `[一作]` (Shenyang Institute of Automation, Chinese Academy of Sciences), Feng Lu `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在视觉场所识别任务中，将数据集剪枝视为场所级别的共集选择，提出基于场所内多样性与场所间相似性的评分方法，生成紧凑而信息丰富的训练子集。

**💡 创新点**

创新点包括：①将剪枝单元从单张图片改为完整场所，契合VPR的关系监督；②设计互补的IPD（场所内多样性）和IPS（场所间相似度）两项指标，用以评估每个场所的正负对学习价值；③采用小批量排序和最小-最大归一化实现可扩展、高效的剪枝流程。

**🔧 技术方法**

技术手段包括：使用代理模型（如DINOv2-B、SALAD等）提取全局描述子；将图像特征聚合为场所级别描述子；计算IPD（平均欧氏/余弦距离）和IPS（k近邻相似度）；对两项指标做归一化后按α权重融合得到最终得分；基于得分进行场所筛选；训练时采用NetVLAD聚合器与DINOv2-B骨干网络。

**📊 数据集**

主要使用GSV-Cities作为训练集，并在此基础上构造约210k场所的合并集；在Pitts30k、MSLS-val、Nordland以及MSLS-challenge等四个标准评测集上验证模型性能。

**📈 对比分析**

与随机、几何基础（Herding、K-Center、Moderate、FDMat）以及混合方法（CCS、D2Pruning）进行对比，结果显示在30%、50%和70%剪枝比例下均能保持或超过全数据集的Recall@1；在合并集71%剪枝后，训练成本下降79%而性能仅低0.1%；与多数据集VPR最先进方法（SALAD-CM、SelaVPR++、MegaLoc）对比，所选核心子集在同等或更小规模下实现了更优或相近的R@1与R@5。

**⚠️ 局限性**

局限性包括：对代理模型特征的依赖，若代理表现不佳可能导致不理想的场所评估；IPD与IPS的权重α需手工调节，对不同任务或数据集可能需要重新调优；在极高剪枝比例下仍可能丢失关键场所，影响鲁棒性；目前方法未针对动态环境或跨域迁移的自适应更新进行设计。

---

## 412. Proof-or-Stop: Don't Trust the Agent, Trust the Evidence -- Loop Engineering for Verifiable Evidence-Gated Lifecycle Control

**arXiv ID:** 2607.14890 | [PDF](https://arxiv.org/pdf/2607.14890v1)

**作者:** Jek Huang `[一作]` (Prodenovo), Ian H. White `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了“Proof‑or‑Stop”生命周期控制方法，要求所有重要的生命周期声明（如“已测试”“已完成”等）只能通过可追溯、可验证的证据（执行收据、代码哈希、审查判定等）才能被接受，从而防止无人监督的自动编码代理把无效或错误的声明升级为正式状态。

**💡 创新点**

创新点在于：
① 将代理输出视为声明（claim）而非直接状态；
② 通过“证据门”实现声明可接受性判定，门内检查新鲜度、绑定、完整性、授权等多维度；
③ 设计了可循环的“proof‑or‑stop”工作流，能够在证据缺失时自动修复或安全停止；
④ 采用 git‑native 证据传输实现跨主机无缝恢复；
⑤ 通过实测证明该门控能显著降低“可见通过/隐藏失败”误差率。

**🔧 技术方法**

主要技术包括：
- 结构化证据收集（执行收据、测试报告、审查结果 JSON 等）；
- 代码哈希与源状态绑定、签名验证、执行重现；
- 基于门控的循环控制（bounded retry、safe‑stop、审查门）；
- 通过多主机收据验证实现主机中立性；
- 对证据门的自动化测试与自评审（self‑application corpus）。

**📊 数据集**

使用的数据集与实验环境：
- 9,240 细胞（cell）规模的 powered ablation（24 个任务、16 种注入失效场景），覆盖 5 种对照（A1-A4）。
- 自身运行的 Proof‑or‑Stop 语料库：565 条开发故事、1,007 条审查发现（94.8% 解决率）。
- 68 行跨供应商高/关键发现（Codex host‑2 审查）。
- 另外包含 10‑情景机制测试套件和 150 行大账本无误执行测试。

**📈 对比分析**

比较方法：
- 通过 5 组对照（prompt‑only、naive retry、naive retry + 预算、review‑only、Proof‑or‑Stop loop）在同一任务集上跑 5 次；
- 主要指标为“not‑amplified”率（即无误完成率）和“Amplified”率（可见通过/隐藏失败）。
- 结果显示：Proof‑or‑Stop loop 在计算预算约束的 naive 控制 A2' 上显著降低放大率（2/1,800 vs 31/1,800，+1.6pp，95% CI [0.8,2.5]），并且在清洁任务上保持 100% 完成率；
- 机制测试验证无 false‑done，跨主机恢复无误。

**⚠️ 局限性**

局限性：
- 仅在单一模型家族（OpenAI Sonnet）上验证，未评估多模型/跨供应商的普适性；
- 任务集相对有限（24 个任务、16 场景），缺乏大规模外部基准；
- 交叉主机实验仅为单故事单机演示，未进行大规模跨机器验证；
- 证据门的实现依赖本地签名密钥，未对多主机身份信任链进行深入测试；
- 现有评测不包括自动合并或生产发布的完整流程。

---

## 413. Analytical study of the optimal combination of binary classifiers based on classifiers-induced partitioning of the training set

**arXiv ID:** 2607.14889 | [PDF](https://arxiv.org/pdf/2607.14889v1)

**作者:** Jean-Marc Brossier `[一作]` (CNRS), Olivier Lafitte `[通讯]` (Université Sorbonne Paris Nord)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构建了基于真值表的逻辑结构化数据框架，用于线性组合二元分类器并分析其凸化风险的最优解。

**💡 创新点**

提出了多维分类校准函数的泛化，给出了对任意数量分类器存在唯一最小点的充分条件，并对三分类器给出完整枚举与解析解，定义ϕ前沿以评估数据质量。

**🔧 技术方法**

采用真值表分区、凸化风险、分类校准函数、解析求解等方法。

**📊 数据集**

未在论文中给出具体数据集，主要为理论推导。

**📈 对比分析**

没有实验对比，性能评估未给出，理论证明表明在满足条件时可获得最优权重。

**⚠️ 局限性**

对大于三分类器的情况难以得到解析解，存在最小值不存在或不唯一的问题，且对数据质量的判断仅在三分类器可行。

---

## 414. SMC-ES: Automated synthesis of formally verified control policies

**arXiv ID:** 2607.15003 | [PDF](https://arxiv.org/pdf/2607.15003v1)

**作者:** Riccardo Curcio `[一作]`, Enrico Tronci `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

未提供具体论文内容

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未知

**⚠️ 局限性**

未知

---

## 415. Steering Robustness into World Action Models via Mechanistic Interpretability and Optimal Control

**arXiv ID:** 2607.14943 | [PDF](https://arxiv.org/pdf/2607.14943v1)

**作者:** Jihoon Hong `[一作]` (Georgia Institute of Technology), Glen Chou `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对世界动作模型（WAM）在分布偏移下的脆弱性进行机理解释，并通过对激活空间的线性可分性进行评估，提出无训练的激活调节方法（ActAdd）和闭环反馈调节方法（WA-LQR）来提升鲁棒性。

**💡 创新点**

创新点在于将机制解释（MI）与激活调节相结合，发现不同WAM架构的鲁棒相关特征在激活空间中的线性可分性差异，并基于此设计出利用局部线性动力学的WA-LQR闭环控制框架，实现了无需微调的鲁棒提升。

**🔧 技术方法**

主要技术包括对比激活向量的线性可分性评估（PCA+SVM），对低维对比子空间的随机SVD降维，局部线性化的Jacobian向量乘法，和基于LQR的闭环控制；同时使用开环激活加法（ActAdd）作为对照。

**📊 数据集**

使用公开机器人基准库LIBERO-10的10个任务，对三种WAM模型（Cosmos-Policy、DiT4DiT、LingBot-VA）进行评估。

**📈 对比分析**

在相同任务下与未调节、prompt调节和ActAdd等基线对比，WA-LQR在摄像头方向、手爪位置和高斯噪声等偏移下平均提升成功率约41%，在多数任务中表现最好；ActAdd在部分噪声场景效果更好，但易出现过调节。

**⚠️ 局限性**

局限性包括：对任务/环境的可迁移性预测缺乏，可解释性分析需针对每个设置进行；结果高度依赖模型架构，未能在所有模型（如LingBot-VA）上取得显著改进；缺乏对不同任务共享表示的自动识别方法。

---

## 416. Verification of a DPLL Transition System in Rocq

**arXiv ID:** 2607.14999 | [PDF](https://arxiv.org/pdf/2607.14999v1)

**作者:** Julia Dijkstra `[一作]` (Delft University of Technology), Benedikt Ahrens `[通讯]` (Delft University of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Rocq中对DPLL过渡系统进行形式化验证，扩展加入纯文字规则，并定义抽象与具体求解策略，提取OCaml实现并在少量SAT实例上验证正确性。

**💡 创新点**

首次将纯文字规则纳入过渡系统，并在Rocq中完整证明正确性、完整性与终止性；同时提出抽象策略框架并给出可验证的具体实现。

**🔧 技术方法**

使用Coq/Rocq进行形式化证明、Equations库实现函数、抽象策略定义、OCaml提取机制。

**📊 数据集**

在验证阶段使用Gregory Duck的SAT实例集（如zebra.cnf）进行测试。

**📈 对比分析**

通过比较已提取的OCaml求解器与标准SAT实例的结果，验证正确性；但由于数据结构为列表，性能在大规模实例上不具竞争力。

**⚠️ 局限性**

局限在于未包含现代SAT关键技术（非时间后退、学习、重置等）、具体策略过于简化、使用线性列表导致效率低下，且未覆盖SMT扩展。

---

## 417. JADE-GS: Joint Alternating Deblurring Guided by Events in 3D Gaussian Splatting

**arXiv ID:** 2607.14990 | [PDF](https://arxiv.org/pdf/2607.14990v1)

**作者:** Haoyu Fu `[一作]` (Shanghai University), Shengjie Zhao `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将事件相机和模糊图像联合用于3D重建的框架JADE‑GS，通过双向教师‑学生循环实现2D去模糊与3D几何的互相正则化。

**💡 创新点**

创新点包括：①像素自适应路由门融合物理事件驱动的EDI和学习式EFNet两种去模糊先验；②在教师-学生循环中引入物理重模糊约束，让3D渲染反馈回2D去模糊，避免单向预处理导致的偏差；③实现高效、低显存（≈5 GB）且可实时渲染的训练方案。

**🔧 技术方法**

核心技术：事件双积分（EDI）物理模型、EFNet深度去模糊网络、像素自适应路由门、3D Gaussian Splatting（3DGS）渲染、物理重模糊约束、双向教师‑学生损失。

**📊 数据集**

使用EvDeblur-Blender（合成）和EvDeblur-CDAVIS（真实）两个公开基准数据集，包含同步RGB模糊图像、事件流与已知相机姿态。

**📈 对比分析**

在两个基准上与多种基线（MPRNet+GS、EDI+GS、EFNet+GS、BAD-NeRF、BAD-Gaussians、E²NeRF、Ev‑DeblurNeRF、DiET‑GS、DiET‑GS++）进行对比。JADE‑GS在LPIPS和CLIP‑IQA等感知指标上领先，并在PSNR/SSIM上保持与最佳基线相近；训练时间约1 h、显存≤5 GB，渲染速度可达100 FPS。

**⚠️ 局限性**

局限性：仅适用于静态场景的视角运动模糊；对强反射/镜面表面和大范围移动物体表现不佳；受COLMAP姿态误差、事件噪声、曝光不匹配等因素影响。

---

## 418. Demographically-Conditioned Synthetic Medical Images for Bias Mitigation and Bias Detection in Disease Classifiers

**arXiv ID:** 2607.14984 | [PDF](https://arxiv.org/pdf/2607.14984v1)

**作者:** Mahmoud Ibrahim `[一作]` (Maastricht University), Michel Dumontier `[通讯]` (Maastricht University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文开发了一个基于人口统计条件的合成胸部 CT 生成器，并将其用于 COVID-19 CT 分类中的偏差缓解与偏差检测。

**💡 创新点**

创新点在于：① 用均衡合成数据做预训练再微调可实现 100 倍数据效率，显著提升整体与最差细胞的 MCC；② 合成少数群体数据可可靠替代真实少数测试集进行公平性审计，保持排名相关性并在样本稀缺细胞上降低估计误差。

**🔧 技术方法**

采用端到端微调的 Stable Diffusion 2.1 潜在扩散模型生成合成 CT，DenseNet‑121 二分类器，预训练+微调调度，合成样本采样，子群体公平性审计与 Spearman ρ 评估。

**📊 数据集**

使用 COVIDx‑CT‑3A 数据集（COVID‑19 与正常胸 CT 切片），并从生成器采样对应的合成子群体数据。

**📈 对比分析**

与仅用真实数据训练的基线对比，预训练再微调的模型在平均和最差细胞 MCC 上分别提升约 0.05 与 0.35，超过完整真实训练；合成测试集与真实强大基线在排名相关性上达到 ρ = 1，且在样本不足细胞的误差显著低于真实小样本测试集。

**⚠️ 局限性**

局限性包括：仅在单一生成器和单一数据集上验证；跨分类器审计一致性与隐私风险（最近邻、成员推断）未作系统评估；合成评估阈值选择需根据具体生成器与指标重新校准。

---

## 419. Explaining Process Control Optimisation Recommendations via GradientSHAP and Implicit Differentiation

**arXiv ID:** 2607.14970 | [PDF](https://arxiv.org/pdf/2607.14970v1)

**作者:** Paul Darm `[一作]` (University of Strathclyde), Annalisa Riccardi `[通讯]` (University of Strathclyde)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

为工业过程控制优化提供基于梯度的SHAP解释

**💡 创新点**

首次将隐函数定理与GradientSHAP结合，实现对优化器输出的快速、准确解释

**🔧 技术方法**

隐函数定理、GradientSHAP、JAX自动微分、LLM自然语言生成

**📊 数据集**

工业高压研磨机（HPGR）参数优化案例，包含20维粒度分布等输入

**📈 对比分析**

与KernelSHAP对比，GradientSHAP在相同精度下速度提升约40倍，SHAP值相关系数>0.99

**⚠️ 局限性**

仅适用于无约束或内部解的优化，需在约束问题中处理KKT活跃集

---

## 420. U-shaped Multi-granularity Learning for Vision-Language Models

**arXiv ID:** 2607.14966 | [PDF](https://arxiv.org/pdf/2607.14966v1)

**作者:** Biao Chen `[一作]` (University of Electronic Science and Technology of China), Lin Zuo `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了U-shaped多粒度提示学习框架UPrompt，用于提升视觉语言模型的跨任务性能

**💡 创新点**

创新点在于结合U-Net结构实现多粒度提示的双向信息流：粗到细的级联增强与细到粗的层级监督

**🔧 技术方法**

采用CLIP作为基础，构造视觉与文本多粒度嵌入，并通过跨粒度注意力与知识蒸馏实现双向连接

**📊 数据集**

在17个基准上评估，包括跨模态检索（Flickr30K、MSCOCO）、少样本分类（11个数据集）和OOD泛化（ImageNet-A/R/Sketch/V2)

**📈 对比分析**

与MAMET、VPKE、CoOp等方法相比，UPrompt在MSCOCO rSum 上提升至 571.1/474.3，基线对比平均提升 4–7 个百分点，且计算开销低

**⚠️ 局限性**

局限在于层次深度有限，仅支持四级粒度，难以捕捉更丰富的语义结构

---

## 421. LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget

**arXiv ID:** 2607.14952 | [PDF](https://arxiv.org/pdf/2607.14952v1)

**作者:** Changhai Zhou `[一作]` (MindLab), Cheng Jin `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 LongStraw 长上下文 GRPO 执行框架，在固定 GPU 预算下通过一次性捕获共享提示并串行响应重放来显著降低内存占用与计算成本。

**💡 创新点**

创新点在于将提示状态与响应重放完全分离，采用按层快照、物理页面复制、完整层级检查点、索引重用以及分层调度等技术，使百万级 token 的训练在不扩容设备的前提下实现可行。

**🔧 技术方法**

使用了内存高效注意力（FlashAttention、PagedAttention）、激活检查点、LoRA/QLoRA、分层索引重用、上下文并行（CP）/专家并行（EP）以及梯度累积等多项技术。

**📊 数据集**

实验使用随机 synthetic token 序列与固定奖励，未采用真实数据集，仅用于验证执行收据与资源消耗。

**📈 对比分析**

通过在 8 个 H20（Qwen）和 32 个 H20（GLM）上分别完成 2,097,152-token 的提示+响应收据，对比 G=2 与 G=8 的内存峰值（约 97 GB）和运行时间（≈5,200 s vs ≈6,800 s）；GLM 在 32‑GPU 上的 2‑成员组收据显示显存峰值约 112–145 GB。未对比真实训练效果，仅给出固定预算下的资源可行性。

**⚠️ 局限性**

局限性包括：缺失全局梯度归约与提示状态梯度的完整计算、未验证分布式更新的一致性与全序列梯度等；实验仅为执行收据，未包含真实奖励、采样、迭代或模型性能评估，因而只能证明资源可行性，不能证明模型训练效果。

---

## 422. Causal Inference for Sequential Settings under Interference and Latent Confounding

**arXiv ID:** 2607.14940 | [PDF](https://arxiv.org/pdf/2607.14940v1)

**作者:** Phevos Paschalidis `[一作]` (Massachusetts Institute Of Technology), Devavrat Shah `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在存在时间依赖和网络干预下的顺序观测数据中，如何进行因果推断；

**💡 创新点**

提出了在单一高维样本上，通过低秩因子结构捕捉潜在混杂并建模Ising型空间-时间依赖的完整框架，首次提供非渐近一致性保证；

**🔧 技术方法**

采用序列最大伪似然估计（Sequential MPLE）结合低秩优化和Dobrushin条件下的Gibbs采样进行参数与因果效应估计；

**📊 数据集**

在合成数据上进行验证，并在真实数据上使用美国各县COVID‑19死亡率与疫苗接种率（NYT、CDC 等公开数据）进行案例研究；

**📈 对比分析**

与不考虑干扰的逻辑回归模型（β设为0）和不考虑潜在混杂的模型（A设为0）对比；在合成实验中，包含干扰与混杂的完整模型将GATE误差降低约92%；在COVID案例中，完整模型的因果效应估计比无干扰模型大约4倍；

**⚠️ 局限性**

理论依赖Dobrushin唯一性条件，强干扰或违反该条件时收敛性与误差控制可能失效；模型假设低秩潜在结构与随机干预可观测性，若不满足则识别困难；

---

## 423. Benchmarking Face Recognition without Real Faces

**arXiv ID:** 2607.14932 | [PDF](https://arxiv.org/pdf/2607.14932v1)

**作者:** Paweł Borsukiewicz `[一作]` (University of Luxembourg), Tegawendé F. Bissyandé `[通讯]` (University of Luxembourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对12个合成人脸数据集与7个真实基准进行系统评估，并与24个预训练模型对比

**💡 创新点**

首次验证合成基准的可靠性，提出评价标准并证明MorphFace和Vec2Face可替代真实基准

**🔧 技术方法**

使用线性与秩相关分析、FID距离、EER/FMR100/FMR1000/ZeroFMR等统计技术

**📊 数据集**

包含12个合成数据集（如MorphFace、Vec2Face、ControlFace10k等）与7个真实基准（LFW、CPLFW、CALFW、AgeDB-30、CFP-FP、IJB-B、IJB-C）

**📈 对比分析**

通过Pearson和Spearman相关系数评估模型排名一致性，MorphFace/Vec2Face在大多数指标上与真实基准的相关系数均超过0.9，证明其可作为可靠评估工具；其他合成集表现不佳

**⚠️ 局限性**

局限性在于仅评估了现有合成集，未覆盖所有特殊场景，且评估仅采用配对协议，未能完全验证在更复杂工作流程中的适用性

---

## 424. TanGO: Training-Free 3D Editing via Tangent-Space Guidance and Optimization

**arXiv ID:** 2607.14927 | [PDF](https://arxiv.org/pdf/2607.14927v1)

**作者:** Siwoo Lim `[一作]` (Korea Advanced Institute of Science and Technology), Chang D. Yoo `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练的3D网格编辑框架 TanGO，利用预训练的 VecSet 流模型的切线空间进行自适应每令牌控制，实现精准局部编辑而不破坏未编辑区域。

**💡 创新点**

创新点在于将编辑过程视为瞬时一阶最优控制，通过源-目标速度的方向差异计算每令牌需求，使用 von Mises–Fisher 正则化得到可控增益，从而在不需要掩码的前提下实现局部增益调节，显著降低语义伪影。

**🔧 技术方法**

主要技术包括：即时一阶最优控制、方向差异的 vMF 推导、增益归一化、基于 VecSet 的流模型、无训练编辑策略、纹理烘焙与多视图渲染。

**📊 数据集**

使用了自构建的 TanGOEdit 基准（100 个高质量编辑样本，来源于 Objaverse-XL、TRELLIS‑500K、Google Scanned Objects），并与 MVEdit、EditP23、FlowEdit、AnchorFlow 等现有方法在相同数据集上进行对比。

**📈 对比分析**

通过 CLIP‑I、CLIP‑T、DINO‑I 等指标进行定量评估，并在用户研究中收集偏好数据；结果显示 TanGO 在所有指标上均优于现有基线，特别是在语义对齐（CLIP‑T）和结构保真（CLIP‑I）方面提升明显。

**⚠️ 局限性**

主要限制包括：受限于底层 3D VAE 的表达能力，难以恢复高频几何细节；对纹理细节的处理仍相对粗糙；并且在更大规模或不同模型下的泛化性需进一步验证。

---

## 425. Random Logit Scaling: Defending Deep Neural Networks Against Black-Box Score-Based Adversarial Example Attacks

**arXiv ID:** 2607.14921 | [PDF](https://arxiv.org/pdf/2607.14921v1)

**作者:** Hamid Dashtbani `[一作]` (Sharif University of Technology), AmirMahdi Sadeghzadeh `[通讯]` (Sharif University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Random Logit Scaling (RLS) 防御方案和针对 AAA 的 Pendulum 自适应攻击；

**💡 创新点**

通过在推理时随机缩放 logits 来混淆攻击者的目标函数，同时不改变模型预测，从而在保持准确率的前提下显著提升对黑盒 score‑based 攻击的鲁棒性；

**🔧 技术方法**

随机化防御（随机缩放 logits）、梯度估计与随机搜索攻击评估、Expectation‑over‑Transformation (EOT) 自适应攻击；

**📊 数据集**

CIFAR‑10 与 ImageNet 数据集，使用 VGG‑16、ResNet‑18、WideResNet‑28‑10（CIFAR）和 ResNet‑50（ImageNet）四种主流网络；

**📈 对比分析**

与 iRND、RFD、oRND、AAA 等随机化/非随机化防御以及 NES、Bandit、Square、SignHunter、ZO‑signSGD、BruSLe 等六种 SOTA 黑盒攻击进行对比；实验表明 RLS 在大多数攻击下将成功率降低 10%‑90%，并在 EOT 自适应攻击中仍能将成功率压至约 45%，同时保持模型准确率且置信度失真最小；

**⚠️ 局限性**

仅针对 score‑based 攻击，无法防御 decision‑based 攻击；为随机化防御，没有可证明的鲁棒性保证，足够多的查询可削弱其效果，且对需要高度置信度校准的应用影响不小。

---

## 426. Human-Robot Interaction in GenAI Architectures via the Agent-Client Protocol

**arXiv ID:** 2607.14919 | [PDF](https://arxiv.org/pdf/2607.14919v1)

**作者:** Jesus Moncada-Ramirez `[一作]` (University of Malaga), Javier Gonzalez-Jimenez `[通讯]` (University of Malaga)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在基于大型语言模型的生成式人工智能（GenAI）机器人架构中，引入 Agent‑Client Protocol（ACP）作为上层人机交互协议，并与 Model Context Protocol（MCP）配合，构建了一个三层解耦、支持实时可观测、授权与中断的机器人系统。

**💡 创新点**

创新点在于将 ACP 从软件工程中的编码代理迁移至机器人领域，实现在 LLM 主导的高层决策与人机界面之间的统一、无耦合通信；同时通过 ACP 与 MCP 的双层协议实现了异构用户界面与机器人核心的无缝连接。

**🔧 技术方法**

采用的技术包括：大型语言模型（LLM）驱动的 ReAct 迭代推理框架、MCP（JSON‑RPC 工具调用）与 ACP（JSON‑RPC 会话与实时事件）协议、ROS 2 Humble 机器人中间件、以及基于 LLM 的工具调用与导航、语音合成等功能。

**📊 数据集**

实验使用的“数据集”主要是实测的物理移动机器人 Sancho 在标准办公环境中的操作记录、传感器数据（激光雷达、RGB‑D 相机）以及手工构建的工具清单；未使用公开大规模语义或视觉数据集。

**📈 对比分析**

通过三种异构 ACP 客户端（CLI、移动 app、开源 UI）在同一机器人上无需修改核心代码即可连接，实验数据显示 ACP 与 MCP 的协议开销仅占总交互时间的 0.5%（≈171 ms），系统能够实时响应人机交互请求，并支持任务中断、授权与进度可视化，性能满足实时机器人应用需求。

**⚠️ 局限性**

局限性包括：ACP 原设计用于编码代理，某些操作（如终端进程管理）在机器人场景中无意义；协议不原生支持连续多模态流（如实时摄像头视频）或异步后台遥测；若需此类功能，仍需额外机制或扩展 ACP 规范。

---

## 427. Quantum XYZ Stabilizer Codes

**arXiv ID:** 2607.14988 | [PDF](https://arxiv.org/pdf/2607.14988v1)

**作者:** Alessio Baldelli `[一作]` (Universitá Politecnica delle Marche), Massimo Battaglioni `[通讯]` (Universitá Politecnica delle Marche)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

引入了包含 Y 型校验的 XYZ 稳定子码框架，并对其距离性质与真正非 CSS 的条件进行了系统性分析。

**💡 创新点**

首次将 Y 型校验加入 CSS 架构，给出了判断是否为真正非 CSS 的秩条件和距离上下界，并构造了基于 IS 与 QD 代码的稀疏有限长度 XYZ 码。

**🔧 技术方法**

采用了经典线性码 PCM 互正交构造、矩阵分解与可约/不可约 XYZ 组件分析、混合 Pauli 逻辑算符权重推导、MILP 求解距离以及 BP4 迭代译码等技术。

**📊 数据集**

在代码容量噪声模型（抛物抖动 depolarizing 通道）下进行 Monte Carlo 仿真，比较不同长度、编码率和生成子权重的 CSS 与 XYZ 码。

**📈 对比分析**

在相同物理 qubit 数、编码率和生成子权重下，使用 BP4 译码器测量逻辑错误率，结果显示 XYZ 码的 LER 明显低于对应的 CSS 码，甚至优于某些先进的 CSS QLDPC 码。

**⚠️ 局限性**

仅在代码容量噪声下验证，未考虑更一般噪声；LC 等价判定的复杂度仍未确定；大规模本地 Pauli 重标记验证不可行，且对大尺寸实例的距离下界仍需估计。

---

## 428. NIFA: Nonlinear IMC enhanced FPGA for efficient ML inference

**arXiv ID:** 2607.15123 | [PDF](https://arxiv.org/pdf/2607.15123v1)

**作者:** Jiajun Hu `[一作]` (Arizona State University), Aman Arora `[通讯]` (Arizona State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了NIFA，一种在FPGA中集成ADC‑free ACAM 的模拟内存计算(IMC)硬块，支持非线性运算，并通过两轮FPGA感知设计空间探索(DSE)和log‑domain映射加速Transformer Attention，实现高能效的深度学习推理。

**💡 创新点**

创新点包括：①用ACAM替代传统ADC，实现ADC‑free且能支持激活、指数等非线性运算；②对IMC块尺寸和FPGA面积预算进行两轮DSE，平衡吞吐、面积和灵活性；③在Transformer中把DIMM转为log域加法，充分利用ACAM的非线性能力，显著提升动态输入矩阵乘法的性能。

**🔧 技术方法**

核心技术：ReRAM交叉口（crossbar）、ACAM（Analog Content‑Addressable Memory）、模拟内存计算、噪声感知微调（NAF）、FPGA感知DSE、log‑domain Attention映射、VTR仿真与能耗模型、C/C++/Verilog RTL实现。

**📊 数据集**

使用的数据集与模型：CNN基准 ResNet‑9 与 VGG‑11；Transformer基准 BERT‑Tiny（2层、2头、128隐藏、512 FFN）；FC层基准用于DSE（多尺寸全连接层）；非DL基准从VTR库取（bgm、LU8PEEng、stereovision1、arm_core）。

**📈 对比分析**

与Azure‑Lily及传统FPGA基线通过吞吐、面积、能耗和整体推理效率对比；结果显示在CNN上能效提升至30×以上、面积效能提升4×，Transformer上能效提升1.9×、面积效能提升1.7×，整体速度提升1.4–1.7×，并保持在长序列（N≥4096）时能效优势不衰减。

**⚠️ 局限性**

局限性包括：①需要为ReRAM编程提供额外的高压电源线；②对噪声与漂移的敏感性需依赖预训练微调；③硬块规模和分布影响FPGA灵活性与资源分配；④在更大规模LLM或更高精度模型上需进一步验证；⑤目前仅针对FPGA的特定架构实现，硬件可移植性待提升。

---

## 429. CoSimRec: Measuring Coordinated-Content Penetration in Recommender Feedback Loops

**arXiv ID:** 2607.15114 | [PDF](https://arxiv.org/pdf/2607.15114v1)

**作者:** Nan Li `[一作]` (Communication University of China), Jiuyang Lyu `[通讯]` (Communication University of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了CoSimRec框架，用于离线模拟推荐系统的闭环反馈，评估协同账号在非协同用户中渗透的效果，并提出了Algorithmic Penetration Rate（APR）指标族；

**💡 创新点**

①首次将协同内容渗透定义为离线反馈循环评估问题；②构建可复现的agent‑based评估协议；③提出APR指标族，能够同时衡量曝光、互动以及与无攻击基线的提升；④通过实验展示不同推荐器、域以及防御策略对渗透的影响；

**🔧 技术方法**

agent‑based 离线仿真、协同策略（随机、主从、同步等）、多种推荐模型（随机、热度、MF、BPR‑MF、LightGCN）、行为采样模型、排名干预（热度下调、多样化、语义惩罚、同步惩罚、信誉惩罚）、APR度量与统计检验；

**📊 数据集**

MIND（新闻）、MovieLens（电影）和 LastFM（音乐）三大公开推荐数据集；

**📈 对比分析**

使用十个随机种子，匹配无攻击基线，计算APR‑Lift、行为APR、CTR变化等；结果显示在主从协同下，热度/反馈敏感排序能显著提升APR‑Lift（LastFM可达0.45，MovieLens约0.13），随机排序几乎无提升；不同推荐器在不同域表现差异；防御中同步惩罚最有效，热度下调和多样化效果不稳定；BPR‑LightGCN在 MovieLens 20% 注入时ASR@20达到0.97；

**⚠️ 局限性**

实验仅为离线、固定预算、规模有限的对照；目标内容人为加入，缺乏真实攻击语义和时序；行为模型假设简化，未验证对真实用户的匹配；防御评估使用oracle风险，未考虑风险估计误差；未涵盖平台级账号管控与内容审核等实际对策；

---

## 430. AlphaWiSE: Adaptive Weight Interpolation for Continual Multimodal Representation Learning

**arXiv ID:** 2607.15094 | [PDF](https://arxiv.org/pdf/2607.15094v1)

**作者:** Sarthak Jain `[一作]` (University of Illinois Urbana Champaign), Yaoyao Liu `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AlphaWiSE，一种在持续学习后对冻结的两份检查点进行张量级权重插值的方法，以实现多模态检索的持续学习；

**💡 创新点**

通过学习每个对齐参数张量的单个插值系数，兼顾不同持续学习策略的稳定性与适应性，显著提升跨模态检索性能；

**🔧 技术方法**

采用权重空间插值（类似WiSE-FT），对AudioCLIP ViT-B/32 backbone的音频、图像、文本编码器进行张量级插值；

**📊 数据集**

在AudioSet数据集上进行逐阶段增量学习，使用840个示例的记忆集；

**📈 对比分析**

与标准持续学习基线（Fine‑Tune、EWC、LwF、iCaRL、C‑CLIP、WiSE‑FT）对比，AlphaWiSE在R@1和mAP上均优于单一基线，尤其在音频‑文本、图像‑音频、图像‑文本检索中取得显著提升；

**⚠️ 局限性**

仅在两份检查点间插值，无法处理多份检查点或更大规模模型，且插值系数依赖小示例记忆的质量与目标任务的选择，可能导致跨任务性能折衷。

---

## 431. Rubrics on Trial: Evolving Rubrics from a Single Query via Synthetic Pairwise Evidence

**arXiv ID:** 2607.15092 | [PDF](https://arxiv.org/pdf/2607.15092v1)

**作者:** Haocheng Yang `[一作]` (National University of Singapore), Hao Wang `[通讯]` (Xiaohongshu Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种完全基于查询的 rubric 演化框架，能够在没有外部标注或模型训练的情况下，从空集合自动生成针对单个问题的细粒度评估标准；

**💡 创新点**

核心创新在于通过两组互补的响应对（本地编辑对与从零生成对）对每个候选 rubric 进行“盲目”pairwise 判定，仅在两对都证明该 rubric 能显著提升答案质量时才接受，从而实现候选级别的验证；

**🔧 技术方法**

使用了多智能体 LLM 体系：rubric 提议者、响应生成器、盲目 pairwise 判定器，并构建了树形演化结构与记忆机制来记录和引导后续提议；

**📊 数据集**

在评估上使用了五大偏好基准套件（JudgeBench、RM-Bench Chat、RewardBench Chat、RewardBench 2 Precise-IF 与 Focus、RubricBench）以及对应的七个子评测集；

**📈 对比分析**

与三种查询无关的直接生成基线（Direct-Generate、TICK、RocketEval）及三种基于外部数据训练的 open-weight 基线（Rubric-RM-8B、Rubric-ARM-8B、Rubric-ARROW-8B）对比，结果显示在六个评测集上平均精度最高，单个集上最高提升达 1.31–4.19%；

**⚠️ 局限性**

局限性包括：仅在偏好评测上验证，尚未证明能提升强化学习或策略性能；未做系统的组件消融与敏感度分析；依赖 LLM 生成的合成响应和 pairwise 判断，需在更广泛模型族、任务域及人工评测上进一步验证鲁棒性。

---

## 432. ANet Patu-1: The Value of Connection in the Agent Network

**arXiv ID:** 2607.15053 | [PDF](https://arxiv.org/pdf/2607.15053v1)

**作者:** Mu Yuan `[一作]` (Agent Network Research), Lan Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了AI代理网络的连接价值，提出了自组织共识协议ANet Patu-1，并通过实验验证多样化低成本代理在网络规模增长时可超越同质强大代理的价值；同时演示网络可自我重建该协议（自反性）

**💡 创新点**

创新点在于：① 从互联网的三大价值定律推导出代理网络的三大价值定律；② 定义了六项最优协作协议属性；③ 设计出在O(1)轮完成自组织分解、协商与共识的协议；④ 通过实验揭示出现交叉与自反性现象

**🔧 技术方法**

采用分布式算法分析方法（复杂度计算与协议得分Q）、并行共识、竞争投票与兼容性加权聚合、以及自定义子任务与成果存储来实现协议；实验中构造了十种多学科“视角”代理集

**📊 数据集**

使用的是基于代理专业视角生成的自定义合成数据（子任务与成果），未采用公开数据集

**📈 对比分析**

与传统投票、广播星、解构者、黑板等协议在轮数、消息量、连接价值和瓶颈方面对比；ANet Patu-1在n=10时实现O(1)轮、O(n)消息、价值接近2^n、无瓶颈；实验表明在n≥3时多样化代理得分突破同质强大代理

**⚠️ 局限性**

局限性包括：实验规模仅限于≤10个代理，未验证数千或百万级规模下的可扩展性；假设代理能准确评估彼此得分并进行完全集成投票；缺乏对异构模型性能差异的量化分析

---

## 433. NFSA: Non-Forward Secure Aggregation with One Server via Two Layer Secret Sharing

**arXiv ID:** 2607.15052 | [PDF](https://arxiv.org/pdf/2607.15052v1)

**作者:** Yufei Zhou `[一作]` `[通讯]` (Sun Yat-sen University), Yufei Zhou (Sun Yat-sen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种适用于单服务器联邦学习的非转发安全聚合协议，利用两层秘密共享实现用户共享密钥时无需服务器转发，结合几乎键同态伪随机函数（KhPRF）和CRT编码实现一次性安全聚合。

**💡 创新点**

创新点包括：①提出两层秘密共享方案，消除服务器转发、降低通信开销；②引入CRT编码压缩KhPRF掩码，减少KhPRF调用次数和通信扩张；③在单服务器场景下实现一次性聚合，支持高维参数。

**🔧 技术方法**

采用Shamir秘密共享、2-2加法秘密共享、伪随机函数（AES-CTR）、Key‑Agreement（ECDH）、几乎键同态伪随机函数（LWR/LWE实现）、中国剩余定理编码。

**📊 数据集**

在多种模型上评估：CNN（19K参数）、LeNet-5（62K）、ResNet-20（273K）以及LSTM（818K）在MNIST、CIFAR‑10和Shakespeare数据集上。

**📈 对比分析**

与基准O PA方案对比，实验显示在用户、服务器和解密器三方的时间和通信量均显著下降；在100名用户、5个解密器的设置下，用户通信压缩近100×，用户计算时间下降约50%，服务器计算与通信也分别降低约50%和25%。

**⚠️ 局限性**

局限性：仅在半诚实模型下安全，缺乏完整恶意安全机制；CRT打包需更大模数，导致KhPRF密钥尺寸增大，增加实现复杂度；对细粒度输入验证支持有限。

---

## 434. Parameter-efficient Prompt Tuning of Vision Foundation Model With Adaptive Focal Loss for Interpretable MCI Screening

**arXiv ID:** 2607.15047 | [PDF](https://arxiv.org/pdf/2607.15047v1)

**作者:** Javad Khoramdel `[一作]` (K. N. Toosi University of Technology), Amirhossein Nikoofard `[通讯]` (K. N. Toosi University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出一种参数高效、可解释的MCI检测框架，利用冻结的DINOv2-Small视觉Transformer通过任务特定的prompt tokens进行自适应，并结合MoCA感知的焦点损失和多种数据增强方法，实现对三种神经心理绘图（时钟、立方体、追踪）数据的多模态融合与分类。

**💡 创新点**

创新点包括：1）通过任务特定prompt tokens对冻结的基础模型进行轻量级适配，仅训练约1.2百万可学习参数；2）跨注意力机制直接生成空间可解释的注意力图，无需后处理；3）MoCA-适配的焦点损失，将连续MoCA分数融入目标、损失调制与样本加权，解决临床阈值不确定性；4）可解释的模态重要性权重，通过查询注意力聚合多模态嵌入；5）多种针对数据稀缺与类别不平衡的增强策略。

**🔧 技术方法**

核心技术包括：Prompt tuning、Cross‑Attention聚合、MoCA‑aware focal loss、类型保持Mixup、图像反转、类别平衡采样、绘图邻域交换；训练采用AdamW、学习率调度、五折分层交叉验证。

**📊 数据集**

使用Ruengchaijatuporn等公开的多绘图MCI数据集，共918名受试者（651健康，267 MCI），每人提供时钟绘图、立方体复制和追踪绘图三张图像，并记录MoCA总分。

**📈 对比分析**

与全量微调的ResViT（ResNet50+ViT-B/16）及其他基线进行比较，使用MCI‑class F1为主要评估指标。最终模型在五折交叉验证中获得F1_MCI 0.641±0.026、AUC 0.795±0.024，较ResViT提升0.110 F1，显著优于单一任务或无MoCA损失的设置。

**⚠️ 局限性**

局限性：仅在泰国一所医院的数据上评估，缺乏跨人群泛化验证；仅利用静态图像，未使用数字绘图的运动学信息；模态重要性权重尚未与临床专家评估对齐；MoCA总分作为单一标签，未考虑子域分数，可能限制诊断细粒度。

---

## 435. Weakly-Supervised RGB-D Salient Object Detection via SAM-driven Pseudo Annotation and State Space Interaction-based Diffusion

**arXiv ID:** 2607.15041 | [PDF](https://arxiv.org/pdf/2607.15041v1)

**作者:** Wenqi Si `[一作]` (Shanghai University), Weisi Lin `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于SAM的伪标注生成和基于状态空间交互的条件扩散模型，完成弱监督RGB‑D显著目标检测；

**💡 创新点**

创新点在于将Segment Anything Model与双分支一致性融合相结合生成高质量像素级伪标签，并设计跨模态频域交互与显式/隐式状态空间交互的条件生成网络与上下文注入模块，实现迭代精细化显著图；

**🔧 技术方法**

使用SAM、二维扩散模型、状态空间模型（SSM）、频域交换、跨模态交互、CRF后处理、PVT‑v2‑B4等技术；

**📊 数据集**

训练集为Xu等人重新标注的1485张NJU2K和700张NLPR的稀疏笔迹标注，测试集为DUT、LFSD、NJU2K、NLPR、SIP、SSD、STERE七个RGB‑D显著目标数据集；

**📈 对比分析**

与18种全监督和4种弱监督方法对比，在七个数据集上均超越所有弱监督方法，并在多项指标上与部分全监督方法竞争，Ablation实验验证每个模块对性能的显著提升；

**⚠️ 局限性**

局限性包括对复杂结构、低质量HHA、稀疏笔迹、低对比度和阴影等场景的伪标签生成效果不佳，SAM的先验可能与显著目标目标不完全一致，导致精细细节缺失或误检。

---

## 436. URVC: A Unified Real-Time Neural Video Coding Model with Temporal, Spatial, and Perceptual Adaptivity

**arXiv ID:** 2607.15033 | [PDF](https://arxiv.org/pdf/2607.15033v1)

**作者:** Xihua Sheng `[一作]` (Hong Kong Polytechnic University), Chang Wen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个统一的实时神经视频编码器 URVC，兼顾时域、空间与感知自适应

**💡 创新点**

创新点包括：1) 速率感知多候选时域预测，2) 基于分解的空间率控制实现零shot ROI，3) 模块库+帧生成器感知切换，可在同一模型内切换信号保真与感知模式

**🔧 技术方法**

采用多分支预测网络、可学习量化向量、空间结构/细节分解量化、熵模型、端到端率-失真优化、可变 QP 训练及感知损失（LPIPS+GAN）

**📊 数据集**

训练使用 Vimeo‑90k，评估使用 HEVC、UVG、MCL‑JCV 三大公开数据集

**📈 对比分析**

与 VTM‑17.0、DCVC‑DC/F‑FM、PRAVC、DCVC‑RT 对比，BD‑rate 平均下降 22.4%（相较 VTM）且在实时约束下优于 DCVC‑RT，感知模式在 DISTS/LPIPS 上显著提升 50%+

**⚠️ 局限性**

局部高帧率/高分辨率视频在高比特率时性能略低，原因是 GPU 内存限制导致的训练分辨率/序列长度受限

---

## 437. Don't Predict, Prioritize: Rethinking GPU Reliability Assessment

**arXiv ID:** 2607.15115 | [PDF](https://arxiv.org/pdf/2607.15115v1)

**作者:** Difeng Ma `[一作]` (Chinese Academy Of Sciences), Gaogang Xie `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了基于学习排序的 GPU 节点风险排名系统 HeaRank，并在 Meta 生产级 GPU 集群中进行大规模评估。

**💡 创新点**

创新点在于把 GPU 故障预测转化为相对风险排序任务，利用稳定的历史失效记录而非易受工作负载影响的时序遥测；同时采用点式交叉熵训练的 MLP 对风险评分进行概率校准。

**🔧 技术方法**

使用学习到排序（LTR）框架、非线性风险交互编码器（MLP）、点式 Sigmoid 训练、AUC 与 NDCG@K 评估指标，以及对比 LightGBM Ranker、Logistic Regression、Linear SVM、CoxPH 与 Random Survival Forest。

**📊 数据集**

使用 Meta 2024‑01 至 2025‑12 年间的生产集群 GPU 失效日志（DBE 与 GPU Lost 共计数千台 GPU 的历史失效记录），并构造每日查询样本。

**📈 对比分析**

与三种启发式基线、LightGBM Ranker 以及其它学习基线在 AUC、NDCG@5/10/20 上进行对比；HeaRank 在 AUC 上达 0.834，NDCG@5 为 0.427，显著优于所有基线；上线后顶 5% 节点捕获率提升至 64%（相较于 21% 的现有系统）。

**⚠️ 局限性**

局限性包括：依赖历史失效记录导致对新节点或设备漂移（concept drift）需要频繁重训练；与实时遥测特征的混合尝试并未显著提升性能；对短期故障预测仍无实质帮助。

---

## 438. Goal-Oriented Semantic Communication for Distributed ISAC-Enabled Vehicle Coordination

**arXiv ID:** 2607.15111 | [PDF](https://arxiv.org/pdf/2607.15111v1)

**作者:** Wenjie Liu `[一作]` (King’s College London), Yansha Deng `[通讯]` (King’s College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究分布式ISAC车辆协调，提出统一的语义通信框架；

**💡 创新点**

将语义重要性驱动的传输决策、EKF状态预测、MHPPO强化学习和不确定性感知波束/功率设计整合为一套协同工作系统；

**🔧 技术方法**

使用扩展卡尔曼滤波（EKF）进行车辆状态预测；Masked Hybrid Proximal Policy Optimization（MHPPO）实现语义重要性驱动的决策；鲁棒波束成形与VoI驱动的时分功率分配等不确定性感知传输设计；

**📊 数据集**

基于SUMO仿真平台生成的交叉口车辆轨迹数据；

**📈 对比分析**

与预测ISAC+PPO基线及各消融方案（无MHPPO、无UTD、无EKF）进行比较；在成功率、通信成功率、信号开销等指标上均表现更优，获得100%安全通过率；

**⚠️ 局限性**

仅在仿真环境中验证，未考虑真实信道、多用户干扰、硬件实现与部署成本等实际因素；

---

## 439. QuReC: All-in-One Image Restoration with Query-Specific Guidance and Local-Global Response Calibration

**arXiv ID:** 2607.15097 | [PDF](https://arxiv.org/pdf/2607.15097v1)

**作者:** Shen Zhou `[一作]` (Southeast University), Fang Dong `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的全场景图像恢复框架 QuReC，能够在同一模型中同时处理噪声、雨渍、雾霾、模糊、暗光等多种退化，并支持空间异质和混合退化场景。

**💡 创新点**

核心创新包括：①查询级退化引导的查询重构模块 DQRM，利用退化原型空间进行自适应匹配，生成空间细粒度的退化感知查询；②局部-全局响应校准模块 LGRCM，采用双分支聚合并通过可学习的先验校准增强局部细节与全局语义的协同；③弱监督原型匹配学习策略，提升原型分配的稳定性和语义一致性。

**🔧 技术方法**

技术手段主要有：基于 Transformer 的编码器-解码器骨干，CLIP 文本编码器生成退化原型向量，查询重构和响应校准的自注意力机制，联合使用 softmax、tanh 校准门以及平衡负载与弱监督匹配损失。

**📊 数据集**

使用了多个公开基准数据集：BSD68、Rain100L、SOTS、GoPro、LOL 以及综合退化数据集 CDD11，覆盖单一退化、全部一体化以及多重混合退化四大实验场景。

**📈 对比分析**

在三退化、五退化和混合退化任务上与 PromptIR、DFPIR、MoCE‑IR 等最新全场景恢复方法对比，QuReC 在 PSNR/SSIM 上均取得领先，平均提升约 0.3~1.2 dB，尤其在 GoPro 去模糊任务上提升 3.68 dB，显示出显著的性能优势。

**⚠️ 局限性**

局限性在于：①需要预先构建退化原型库，对未出现的退化类型效果可能受限；②查询级匹配和双分支聚合增加计算与内存开销；③对极端复杂或高度混合的退化仍可能出现残留噪点或细节丢失。

---

## 440. Digital Pantheon: Simulating and Auditing Coalition Formation with LLM Agents

**arXiv ID:** 2607.15095 | [PDF](https://arxiv.org/pdf/2607.15095v1)

**作者:** Dylan Van Mulders `[一作]` (Ghent University), Dirk Van den Poel `[通讯]` (Ghent University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个多代理LLM协商框架，用于模拟比利时弗拉芒地区2019选举后的联盟谈判，并通过多层信息谱系拓扑（MILT）和联盟影响分数（CIS）等可解释性工具追踪协商过程与结果。

**💡 创新点**

首次将SFT+ DPO双阶段党派个性化训练与检索增强生成相结合，形成与党派宣言严格绑定的党派代理；并提出多层信息谱系拓扑和联盟影响分数两种可解释评估方法，兼顾内部过程与外部真实一致性。

**🔧 技术方法**

使用Gemma‑3 27B LLM，QLoRA 4bit 微调、SFT、DPO、ChromaDB检索、Hub‑and‑Spoke 多代理协商协议，以及 Qwen3.6 进行反向 NLI 归因。

**📊 数据集**

2019年弗拉芒主要政党官方宣言文本以及当年的历史联盟协议作为真实对照。

**📈 对比分析**

在三次独立模拟中，MILT显示57.4%条款可追溯，30%为错误，真实实现率约28%；CIS确定N‑VA为赢家，排名与实际议会席位分布相符，表现优于仅预测终端结果的先前模型。

**⚠️ 局限性**

权重统一忽略条款重要性；模拟仅限于党派层面，未考虑社会经济、联邦层面或媒体效应；NLI标签缺乏人工验证；CIS未考虑条款规模或预算影响。

---

## 441. BrainPilot: Automating Brain Discovery with Agentic Research

**arXiv ID:** 2607.15079 | [PDF](https://arxiv.org/pdf/2607.15079v1)

**作者:** Haoxuan Li `[一作]` (Tsinghua University), Lu Mi `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套名为BrainPilot的多智能体系统，利用PI代理与专家代理、检索式知识库、技能库以及可追踪的工作流，旨在加速脑科学研究并保证可解释性。

**💡 创新点**

创新点在于将人机协作的可审计工作流（Graph of Trace）与独立的Auditor模块相结合，并将大规模脑科学知识和可复用技能嵌入代理，使其既能保持高效又能防止幻觉与错误。

**🔧 技术方法**

采用了大语言模型驱动的多智能体框架、向量检索+reranker技术构建知识库、技能路由与调用机制，以及可追踪日志（GoT）和审计模块，整体实现基于开源LLM的自动化。

**📊 数据集**

使用了包含7,233条神经科学文献的知识库，以及BrainPilotBench-v0四个任务（RSC、TOPS-fMRI、BCI IV 2a、Sleep-EDF）和Agents' Last Exam中的三项脑科学任务进行评估。

**📈 对比分析**

与Agents' Last Exam中的Codex、Claude Code等基线进行对比，BrainPilot在确定性任务上取得与基线相当或更优的分数，同时成本显著降低；在BrainPilotBench上表现接近或优于Claude Code，但对开放式任务的鲁棒性略低。

**⚠️ 局限性**

局限性包括对高度开放性、复杂多步任务的鲁棒性不足；依赖手工维护的知识库和技能库，需持续更新；模型仍可能产生幻觉，需人工监督；跨模态整合与更深入的实验设计支持尚未实现。

---

## 442. An Introduction to Sparse Identification of Nonlinear Dynamics for Engineering Applications

**arXiv ID:** 2607.15077 | [PDF](https://arxiv.org/pdf/2607.15077v1)

**作者:** Yao Cheng Li `[一作]` (Imperial College London), Urban Fasel `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统教程介绍并扩展了SINDy方法，验证其在无人机系统识别与热交换器动力学建模中的应用。

**💡 创新点**

创新点在于结合约束、噪声鲁棒、控制输入、集成与主动学习等多种SINDy扩展，并提供开源实现。

**🔧 技术方法**

采用SINDy、SINDy-with-Controls、弱形式SINDy、Ensemble‑SINDy、POD/DMD/autoencoder等技术。

**📊 数据集**

使用无人机仿真数据（Parrot Minidrone）和热循环流场数值数据（热循环器 Navier–Stokes Boussinesq）。

**📈 对比分析**

与传统参数化模型和全模型相对比，SINDy能在有限噪声数据下恢复可解释的低维动力学，并在预测精度上匹配或超越传统方法。

**⚠️ 局限性**

局限包括对高维复杂系统仍需降维、对噪声与稀疏采样的处理仍有挑战、以及对非平稳或非欧几里得空间的适用性待进一步研究。

---

## 443. SUFLECA: Scaling Up Feature Learning for CAD-to-image Alignment

**arXiv ID:** 2607.15058 | [PDF](https://arxiv.org/pdf/2607.15058v1)

**作者:** Saad Ejaz `[一作]` (University of Luxembourg), Jose Luis Sanchez-Lopez `[通讯]` (University of Luxembourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

实现了一种无监督（弱监督）零样本9D CAD对齐方法SUFLECA。

**💡 创新点**

创新点：1) 在多源真实+合成数据上大规模利用Normalized Object Coordinates（NOC）进行特征学习，显著提升几何感知与跨域泛化；2) 引入基于几何一致性的互相kNN匹配与一致性过滤，去除不一致对应，避免昂贵的迭代精细化。

**🔧 技术方法**

技术：冻结的视觉基础模型（如DUNE-B/DUNE-S）+Dense Prediction Transformer；NOC头用于低维几何特征学习；几何一致性匹配（互相kNN + IRLS尺度求解 + RANSAC-Procrustes）；对齐质量由注册残差分析得到的score。

**📊 数据集**

数据集：训练使用674K图像，涵盖12个真实+合成数据集（Pascal3D+, Objectron, ARKitScenes, REAL275, RealEstate10K, ScanNet, ScanNet++, Pix3D, ObjectNet3D, 3D-Front, Hypersim, ShapeNet）；评估在ScanNet25k、DiffCAD拆分以及CO3D上。

**📈 对比分析**

与方法比较：在ScanNet25k NMS协议下，SUFLECA类别平均33.4%/实例平均42.3%，比ZeroCAD（23.1/30.1）和FoundationPose显著提升；在DiffCAD split上也实现近乎两倍提升；在CO3D遮挡、非精确CAD情况下，SUFLECA的几何一致性匹配表现优于同类方法。

**⚠️ 局限性**

局限：性能仍受CAD检索准确率影响，检索错误导致对齐下降；目前仅验证室内常见物体，未覆盖户外或罕见类别；需要进一步改进开集检索与更大尺度、多样化数据的泛化能力。

---

## 444. SCITUS: A Multi-Jurisdictional Framework for Adapting NIST AI RMF to the Canadian Regulatory Context

**arXiv ID:** 2607.15051 | [PDF](https://arxiv.org/pdf/2607.15051v1)

**作者:** Mohammad Etemad `[一作]` `[通讯]` (Scitus Solutions Ltd), Mohammad Etemad (Scitus Solutions Ltd)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了SCITUS框架，将NIST AI RMF 1.0本地化并整合加拿大联邦及各省（安大略、魁北克、阿尔伯塔、曼尼托巴、BC）多司法管辖区的AI法规，提供统一的合规评估、控制清单和文档生成方法。

**💡 创新点**

创新点在于：①提出多司法管辖合规映射方法，①将NIST AI RMF与加拿大法规融合并扩展七大可信AI特性；②基于Treasury Board影响等级实现风险分层实施；③提供统一文档与工具，显著降低多方评估与维护负担。

**🔧 技术方法**

采用需求提取与分类技术、法律文本比对、风险管理与控制设计、三层架构（标准-法规-实现）以及文档生成工具；并结合NIST AI RMF、ISO/IEC 42001等国际标准。

**📊 数据集**

使用的主要数据集包括加拿大联邦和各省的法律文本（Treasury Board指令、Bill 194、Law 25、Bills 33/34等）、行业案例（政府、医疗、金融、私企）以及15家组织的合规负担调查数据（评估时间、成本、文档量等），并对SCITUS控制清单从v1.0（31项）演进至v2.0（57项）。

**📈 对比分析**

通过与传统逐司法管辖区独立评估对比，SCITUS在三大场景（联邦签证、安大略医院放射筛查、魁北克招聘AI）中实现合规覆盖率≈98%，评估时间从12‑16周降至6‑8周，成本下降约30%，同时通过专家验证保持与NIST、ISO等国际标准的兼容性。

**⚠️ 局限性**

局限性包括：需要持续跟踪法规更新以维护映射准确性；对高级AI技术（如强化学习、联邦学习）及跨国供应链监管细节兼容性有限；缺乏完全自动化的合规工具；在处理极端冲突时仍需人工治理与决策。

---

## 445. Towards realistic large random models of labeled transition systems and their 0-1 laws

**arXiv ID:** 2607.15029 | [PDF](https://arxiv.org/pdf/2607.15029v1)

**作者:** Milan Lopuhaä-Zwakenberg `[一作]` `[通讯]` (University of Twente), Milan Lopuhaä-Zwakenberg (University of Twente)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于随机图理论的、与实际软件模型相符的Kripke结构（KS）概率模型，并在此模型上对LTL和CTL的收敛性进行了理论分析；

**💡 创新点**

创新点在于结合理论与Model Checking Contest实测数据，给出大规模KS的边、初态与原子命题的概率标度，并证明了LTL满足0-1律、CTL满足收敛律，同时给出相应的极限概率计算算法；

**🔧 技术方法**

主要技术手段包括Erdős–Rényi式随机图模型、拓扑与连通性分析、概率极限定理、归约与等价关系的构造，以及对LTL/CTL语义的细致拆分；

**📊 数据集**

利用Model Checking Contest的258个KS基准（状态规模从10^2到10^254），对平均度、初态数与原子命题满足度进行统计与回归验证；

**📈 对比分析**

与传统固定概率模型相比，本文所给模型在理论复杂度上表现为LTL极限判定为PSPACE‑complete、CTL极限判定为NP‑hard，说明在大规模实际场景下仍需高效近似方法；

**⚠️ 局限性**

局限性包括对边与状态属性独立性与同质性的假设、仅覆盖c>max_S ϱ_S^{-1}的大概率情形，以及对小c或非独立同步结构的分析仍待进一步研究。

---

## 446. Man, Machine, and Masterpiece: Artistic Ownership in the AI Era

**arXiv ID:** 2607.15027 | [PDF](https://arxiv.org/pdf/2607.15027v1)

**作者:** Sofi Gjing Jovanovska `[一作]` (University of Siegen), Shadan Sadeghian `[通讯]` (University of Siegen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个名为ArtSplit的AI驱动的原型，能够在创作工作流程中量化人类与AI的贡献，并通过访谈方式探讨艺术家对创作所有权的认知与反应。

**💡 创新点**

创新点在于将所有权视作可量化指标，并通过“provotype”方式将量化结果可视化，迫使艺术家重新审视传统的作者权概念，从而揭示了量化方法与艺术家主观理解之间的张力。

**🔧 技术方法**

使用的技术主要是AI图像生成模型（如Stable Diffusion、Midjourney、DALL‑E）以及前端可视化（Figma）和视频演示；对量化指标进行了手工定义并在系统中实现；访谈则采用了半结构化访谈法。

**📊 数据集**

未使用公开数据集；研究数据来源于5名艺术家（3名学术艺术家、2名内容创作者）的访谈记录和系统产生的量化结果；视频vignette展示了三种不同的工作流程变体。

**📈 对比分析**

由于研究旨在探讨主观反应而非性能指标，未进行传统意义上的方法比较；评价主要通过主题分析获得，结果呈现艺术家对不同权属分配方案的感知公平性、可控性与潜在滥用风险。

**⚠️ 局限性**

局限性包括：样本规模较小（仅5名受访者），缺乏多样性；采用视频vignette而非真实创作环境，可能无法充分捕捉真实工作流程；量化指标设计主观且未经过验证；未对系统的技术实现进行客观性能评估。

---

## 447. Risk-Aware Belief Control Barrier Functions over Random Finite Sets

**arXiv ID:** 2607.15016 | [PDF](https://arxiv.org/pdf/2607.15016v1)

**作者:** Shaohang Han `[一作]` (KTH Royal Institute of Technology), Jana Tumova `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在未知、时变多物体环境中设计了风险感知信念控制障碍函数（BCBF）实现安全控制。

**💡 创新点**

创新点在于将随机有限集（RFS）PHD信念直接嵌入BCBF，处理物体数量未知与测量关联模糊，并提供离散更新安全保证。

**🔧 技术方法**

采用SMC‑PHD滤波器、随机有限集理论、非光滑控制障碍函数与QP求解器（OSQP）实现。

**📊 数据集**

使用仿真场景（FOV维护、动态障碍物避免）和蓝色ROV水下机器人实验数据。

**📈 对比分析**

与Mean‑CBF、MAP‑CBF及软最小距离基准对比，实验显示成功率高、碰撞率低、控制计算时间<5 ms，优于基准。

**⚠️ 局限性**

局限在于对PPP近似与真实后验的误差未建模，离散更新的安全性需通过风险收窄保证，且未提供状态回退保证。

---

## 448. Automated Template-free Synthesis of Instruction-Centric Leakage Contracts for Black-Box CPUs

**arXiv ID:** 2607.15118 | [PDF](https://arxiv.org/pdf/2607.15118v1)

**作者:** Elvira Moreno `[一作]` (IMDEA Software Institute), Marco Guarnieri `[通讯]` (IMDEA Software Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个模板无关的黑盒CPU侧信道泄漏合约合成方法，能够在不获取RTL或模板的情况下为x86和ARM等商用CPU自动生成指令级泄漏合约。

**💡 创新点**

创新点在于消除了对CPU RTL和人工模板的依赖，采用反例驱动合成与约束求解自动推导细粒度泄漏规则，并通过最小化后进一步提高精度。

**🔧 技术方法**

核心技术包括逆向合成（从CPU执行结果中提取硬件和合约追踪）、反例驱动（使用Revizor和Scam‑V检测合约违例）、Rosette+Z3的语法引导合成以及合约后处理（最小化、合并）。

**📊 数据集**

实验使用真实x86（i5‑6500、i5‑1335U、Ultra5‑225U、Ultra7‑258V）和ARM（Cortex‑A72、A76）CPU，并对多种泄漏模型（常数时间、TagIdx、RFC、SilStore、乘法简化）进行合成。

**📈 对比分析**

与现有模板化白盒工具（LeaSyn、RTL2µPath、VeloCT）比较，所提出的工具在相同搜索空间下得到更高的精度（多达1.0）且产生更多条精细合约，虽然在执行时间上比模板工具慢约2–3倍，但通过最小化可显著缩短。

**⚠️ 局限性**

局限性包括只能生成指令级合约，无法覆盖跨指令状态泄漏；合约质量高度依赖测试用例的覆盖率，可能导致假阳性或假阴性；以及对特权指令和内核级泄漏的支持不足。

---

## 449. Long-Context Fine-Tuning with Limited VRAM

**arXiv ID:** 2607.15105 | [PDF](https://arxiv.org/pdf/2607.15105v1)

**作者:** Vladimir Fedosov `[一作]` (BMW Group), Frank Woernle `[通讯]` (BMW Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合 Hierarchical Global Attention、分段反向传播和分层 KV 存储，对 Qwen3‑8B 进行参数高效长上下文微调，将可训练上下文从 2K 扩展到 16K，并保持密集注意力的推理质量。

**💡 创新点**

创新点在于：1) HGA 仅在主动段保留梯度，将历史 KV 按需分离到 RAM/NVMe；2) 每个 token 的历史选择保持固定，显著降低内存与计算开销；3) 通过分段训练实现 TBPTT 与外部 KV 存储的结合，突破 GPU 16GB 的上下文限制。

**🔧 技术方法**

采用的技术包括：Hierarchical Global Attention (HGA)、分段反向传播（TBPTT）、分层 KV 存储、4‑bit NF4 QLoRA 量化、FP16 PagedAdamW 优化器、FlashAttention 等。

**📊 数据集**

使用的主要数据集是 PG19 的官方 train/validation splits。

**📈 对比分析**

对比方法：在相同的 2K 上下文下，HGA 与密集注意力在训练速度和推理精度上直接对齐；在 16K 上下文下，HGA 能在单卡 16GB 上完成训练，密集训练则 OOM。HGA 在 2K 下速度略快（约 1.05×），精度差异不到 0.02 nat；在更长上下文下预计速度优势进一步扩大。

**⚠️ 局限性**

限制包括：1) 训练规模受限，超过 100–200M 训练 token 可能出现因路由共享导致的因果泄漏；2) NVMe 分层未进行完整基准；3) TBPTT 截断梯度，可能影响长程依赖学习；4) 仅在单一模型与单一 GPU 上验证，尚未证明在更大模型或更高上下文长度下的可扩展性。

---

## 450. Capturing and Exploiting Design Pattern Variability in Mobile Application Generation

**arXiv ID:** 2607.15099 | [PDF](https://arxiv.org/pdf/2607.15099v1)

**作者:** Ramón Peralta `[一作]` (Universidad de San Jorge), Jose-Miguel Horcas `[通讯]` (Universidad de Málaga)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过把设计模式的内部变异显式建模为UVL特性，并结合Jinja模板在SPL框架中自动生成可配置的Swift实现，提升移动应用的架构质量。

**💡 创新点**

创新点在于将设计模式本身视作可配置的SPL资产，利用UVL捕获其内部结构与行为变异，并通过模板生成实现，从而实现系统化、可定制的移动应用生成。

**🔧 技术方法**

采用的技术包括Universal Variability Language（UVL）、Jinja模板引擎、UVengine变量解析器、Swift编程语言，以及软件产品线（SPL）工程方法。

**📊 数据集**

使用的数据集是公开托管在GitHub的五个设计模式（Singleton、Strategy、Observer、Adapter、Factory Method）的UVL模型和相应Jinja模板集合。

**📈 对比分析**

评估方法包括配置空间分析、与手工实现对比（以Singleton为例）以及生成效率测量，结果显示生成代码行数缩减约50%，循环复杂度降低，生成时间约为15–20分钟。

**⚠️ 局限性**

局限性包括仅覆盖五种模式和Swift语言，缺乏跨平台（Kotlin/Java等）的验证，未对所有变体的手工实现进行完整对比，且实证案例数量有限。

---

## 451. Quantifying Training Membership Information in the Hyperspherical Embedding Geometry of Face Recognition Models

**arXiv ID:** 2607.15084 | [PDF](https://arxiv.org/pdf/2607.15084v1)

**作者:** Ünsal Öztürk `[一作]` (Idiap Research Institute), Sébastien Marcel `[通讯]` (Université de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过训练和评估180个面部识别模型，量化了训练集成员身份在嵌入几何结构中的残余信息；

**💡 创新点**

创新点在于系统化研究了四种几何统计量对成员/非成员区分的贡献，并揭示训练身份数是决定该信息量的主要因素，同时发现跨域测试会夸大成员信号，并展示统计量融合可进一步提升识别效果；

**🔧 技术方法**

使用角度间隔损失（ArcFace、CosFace、MagFace）和IResNet/ViT-T backbone，提取4种几何统计量（PairCos、vMF浓度、PenLogit、ProtoCE），并训练MLP融合模型；

**📊 数据集**

训练数据为WebFace4M（约4M张图、205,990人），评估数据为9个公开基准（LFW、CFP-FF/FP、XQLFW、CPLFW、AgeDB-30、RFW、IJB-C）以及同域WebFace4M hold‑out；

**📈 对比分析**

在保持相同的验证误差范围内，成员与非成员统计量的ROC AUC 最高可达0.92（PairCos在1K身份、40轮时接近1.0），但随身份数增大到100K以上仍保持>0.7；融合MLP可进一步提升至≈0.96；

**⚠️ 局限性**

局限包括：仅使用单一训练集（WebFace4M），未评估其他数据集或防御技术；统计量仅捕捉嵌入几何，可能低估可利用的成员信息；跨域基准可能混淆域迁移与成员差异；

---

## 452. Pattern-Guided Design Space Exploration for FPGA Accelerator Design

**arXiv ID:** 2607.15068 | [PDF](https://arxiv.org/pdf/2607.15068v1)

**作者:** Jialiang Zhang `[一作]` (University of Illinois Urbana-Champaign), Yuelin Zou `[通讯]` (Columbia University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一个轻量级的模式驱动设计空间探索框架，专为用allo编写的FPGA内核的高水平综合（HLS）调度优化而设计。

**💡 创新点**

创新点在于利用计算模式（如elementwise、reduction、矩阵运算、卷积等）对调度空间进行结构化裁剪，显著减少无用候选方案，同时保持最佳性能。

**🔧 技术方法**

采用模式映射与调度模板、LLVM层级验证、基于资源与延迟的简单估算器/排序器，并与Xilinx Vitis HLS后端集成。

**📊 数据集**

使用六个代表性内核（elementwise、scalar-vector、reduction、MatVec、GEMM、Stencil）进行实验，涵盖多种常见计算模式。

**📈 对比分析**

与“exhaustive-lite”全局枚举基线对比，候选数从140降至29（整体4.83×缩减），并在所有内核中恢复与基线相同的最佳Vitis HLS延迟，验证了模式驱动裁剪的有效性。

**⚠️ 局限性**

局限性包括模式库手工维护、估算器过于简化、未覆盖卷积、稀疏、流式等更复杂模式，以及未与更强大或学习驱动的DSE基线进行比较。

---

## 453. Neural operators solve inverse problems for constitutive model discovery

**arXiv ID:** 2607.15049 | [PDF](https://arxiv.org/pdf/2607.15049v1)

**作者:** Moritz Flaschel `[一作]`, Ellen Kuhl `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过神经算子直接将实验测得的位移场与反作用力映射到材料的本构函数，实现了无需优化的材料表征。

**💡 创新点**

引入 Physics-Augmented Neural Operators (PANO) 与 Constitutive Artificial Neural Operators (CANO)，利用 Laplacian eigenbasis 编码位移、物理约束输出空间，获得离散无关、噪声鲁棒的本构函数预测。

**🔧 技术方法**

采用深度算子学习（DeepONet 结构）、物理约束的 PANN/CANN 网络、Laplacian eigen函数投影以及监督式全局训练。

**📊 数据集**

利用 FEniCSx 对含中央孔的薄板进行 3000 次有限元模拟，使用 Latin Hypercube 采样的三阶莫里斯‑瑞文基元生成位移、反作用力及对应本构函数。

**📈 对比分析**

通过 MSE 分布、最优/中位/最差样本对比进行比较；CANO 在 MSE 方面显著优于 PANO，并在加噪、缺失、不同离散和尺寸下保持低误差。

**⚠️ 局限性**

仅验证在固定几何（薄板+孔）且不可压缩、各向同性弹性下；对更复杂几何、时间离散、非平面应力以及粘塑性/损伤等尚未推广。

---

## 454. Video = World + Event Stream

**arXiv ID:** 2607.15038 | [PDF](https://arxiv.org/pdf/2607.15038v1)

**作者:** Lianghua Huang `[一作]` (Alibaba Group), Zoubin Bi `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于世界+事件流分解的全双工音视频实时交互系统 Wan-Streamer v0.3，扩展到自由词汇行为描述。

**💡 创新点**

创新点在于将视频拆解为持久世界上下文和随时间变化的事件流，提出通用预训练任务，并在实时交互中实现可开口行为指令，同时保持低延迟（≈200 ms 模型端）。

**🔧 技术方法**

采用单一 Transformer 的因果注意力模型，结合 Ulysses 样式上下文并行执行，语言+行为混合 token 流，音视频潜在流生成。

**📊 数据集**

使用大规模公开视频数据集，标注时间对齐的事件与世界上下文，进行预训练；后续下游任务使用多模态用户输入。

**📈 对比分析**

通过与 v0.1/v0.2 的对比，保持 640×368 25 FPS、≈200 ms 模型端延迟、≈550 ms 总交互延迟；实验表明可实时实现开放词汇行为，行为与语音同步。

**⚠️ 局限性**

局限性在于仅验证了实时双向音视频交互，未评估其他下游任务（如导航、操作）；行为指令受训练数据多样性限制，实际表现可能随数据质量波动。

---

## 455. Roman-Type Domination on Convex and Chordal Bipartite Graphs: Algorithms and Hardness

**arXiv ID:** 2607.15026 | [PDF](https://arxiv.org/pdf/2607.15026v1)

**作者:** Gautam K. Das `[一作]` (Indian Institute of Technology Guwahati), Kamal Santra `[通讯]` (Indian Institute of Technology Guwahati)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了罗马式统治参数（Roman-{2}、双罗马、完美罗马、唯一响应罗马）在两类二分图（凸二分图和弦形二分图）上的算法与复杂度。

**💡 创新点**

提出了统一的左至右动态规划框架，利用凸二分图的区间结构将未完成的需求压缩为常数维边界索引，从而在凸二分图上实现所有四个参数的多项式求解；同时给出弦形二分图上三种参数的NP-完全性证明，揭示两类图之间的算法鸿沟。

**🔧 技术方法**

采用基于区间排序的动态规划、状态压缩与递推（Reduce/Clear）、多维表格存储，以及多类归约（从Dominating Set、Efficient Domination和特定边件构造）来证明复杂度。

**📊 数据集**

论文为理论性工作，未使用具体数据集；所有结论均基于图结构与多项式时间/NP-完全性证明。

**📈 对比分析**

通过比较，凸二分图上算法时间复杂度为O(n⁶)，空间为O(n⁵)；弦形二分图上三种参数已被证明为NP-完全，表明在该类图上没有已知多项式解法。

**⚠️ 局限性**

局限性：动态规划时间仍为O(n⁶)，尚无进一步优化；仅针对凸二分图和弦形二分图，其他相关二分图类（如圆凸、三角凸等）仍未分析；NP-完全结果基于特定归约，可能不适用于更广泛的图类。

---

## 456. Leaf: An Instrumentation-based Dynamic Analysis Framework for Rust

**arXiv ID:** 2607.15025 | [PDF](https://arxiv.org/pdf/2607.15025v1)

**作者:** Mohammad Omidvar Tehrani `[一作]` (Simon Fraser University), Steven Y. Ko `[通讯]` (Simon Fraser University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个基于 Rust 本身 MIR 的动态分析框架——Leaf，并通过 DMIR（Dynamic MIR）接口为分析器提供事件流。

**💡 创新点**

创新点在于：①将 Rust 的 Ownership、类型、生命周期等语义完整保留在事件流中；②通过可配置的 MIR 插桩实现对多种分析需求的支持；③设计了 Probe‑to‑DMIR 适配器，将低级插桩信息转换为面向分析的高层事件。

**🔧 技术方法**

核心技术包括：MIR 级插桩（在编译期插入 probe 代码）、静态信息数据库（如类型 ID、函数 ID）、事件驱动接口 DMIR、可配置的插桩规则、以及对外部 crate 的按需重新编译与插桩。

**📊 数据集**

评估使用了 8 个流行 Rust crate（如 serde、tokio 等）以及 Rust compiler benchmark suite（官方性能基准），并在这些项目上实现了三种动态分析（共计 1.5 万行代码）。

**📈 对比分析**

与现有框架（如 Miri、RuDyna、DynaPyt 等）相比，Leaf 在编译时引入的平均额外时间约 30%~50%，运行时开销在 1.5×~3.0× 之间，且能够完整保留语义、支持更大规模分析，实验结果表明功能正确且性能可接受。

**⚠️ 局限性**

主要局限包括：①编译时需要重新编译所有依赖导致构建时间显著增加；②插桩仅覆盖可插桩代码，对外部 C/C++ 库或未被编译的 crate 无法提供完整事件；③在极大项目中，事件流量大、内存占用高，影响长时间执行。

---

## 457. CosFly-VLA: A Spatially Aware Vision-Language-Action Model for UAV Tracking

**arXiv ID:** 2607.15004 | [PDF](https://arxiv.org/pdf/2607.15004v1)

**作者:** Ruilong Ren `[一作]` (Autel Robotics), Kangli Wang `[通讯]` (Northeast Normal University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 CosFly‑VLA 的闭环 UAV 目标跟踪框架，能够在目标被遮挡时保持空间假设并恢复追踪。

**💡 创新点**

创新点包括：① 将 Vision‑Language‑Action（VLA）模型与多头 meta‑query 接口结合，直接输出航迹、目标框和可见性；② 引入空间化连续预训练（CPT）以补充 UAV 视角的三维几何知识；③ 采用三阶段课程化监督、链式推理（CoT）训练以及闭环强化学习实现对遮挡恢复的专门优化。

**🔧 技术方法**

使用的技术主要有：Qwen3.5 视觉‑语言主干（冻结+LoRA 适配）、DiT 流匹配动作专家、MLP 目标框/可见性头、CPT（混合 500k UAV 与航空空间数据）、三阶段 SFT、CoT 生成推理轨迹、闭环 PPO‑style RL 与 Group‑Relative Advantage。

**📊 数据集**

训练数据来源包括：500k 内部 UAV 轨迹数据（包含目标定位、遮挡、描述等）和 6 个公开航空空间数据集（AirSpatial、Open3DVQA‑v2、HRVQA、AVI‑Math、AirCopBench、CapERA），测试采用 CARLA Town10HD（见测试）与 Town01/03/05（未见测试）。

**📈 对比分析**

与 OpenVLA、π_0、π_0.5 等基线对比，在 open‑loop 评估中 CosFly‑VLA‑0.8B（SFT+CoT）平均位移误差（ADE）下降 34–36%，在 closed‑loop 评估中成功率（SR）提升 17–2% 点，轨迹误差（ADE）下降 10–8%，避障和距离误差亦显著降低，显示显著性能提升。

**⚠️ 局限性**

局限性：仅在模拟环境中验证，缺乏真实 UAV 动力学与传感噪声考量；目标行为相对单一（速度恒定、户外场景），缺少多样化运动与遮挡场景；使用单一 0.8B 模型，未评估更大规模模型与多种种子结果；闭环评估仍基于地面真历史，未验证预测框反馈鲁棒性。

---

## 458. A Correlation-Gap Bound for Nonlinear Gaussian PCA

**arXiv ID:** 2607.15035 | [PDF](https://arxiv.org/pdf/2607.15035v1)

**作者:** Minbo Gao `[一作]` (Chinese Academy of Sciences), Chenghua Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

在高斯数据上证明了Karhunen–Loève基在非线性最佳 d 项逼近中几乎最优，给出 1+O(1/√d) 的保持能量近似上界。

**💡 创新点**

首次将矩阵主要化与均匀矩阵的相关性缺口结合，得到无维度依赖的近似常数并接近原 Mallat–Zeitouni 猜想。

**🔧 技术方法**

利用 Schur–Horn 主要化、Karamata 不等式、层壳分解以及均匀矩阵的争用解决方案等技术来构造阈值松弛和分析相关性缺口。

**📊 数据集**

本文为纯理论工作，没有使用实验数据集；结论对任意高斯协方差矩阵均成立。

**📈 对比分析**

与原始猜想的保持能量形式对比，得到与最优基相差不到约 1.15 倍（d=10）且随着 d 增大趋于 1，表明 PCA 在非线性阈值化后仍接近最优。

**⚠️ 局限性**

仍未能证明常数 1 的完全最优性，且证明方法通过松弛高斯相关结构，可能无法捕捉旋转后完整的依赖优势。

---

## 459. Catch, Throw, Repeat: Planning for Human-Robot Partner Juggling

**arXiv ID:** 2607.15129 | [PDF](https://arxiv.org/pdf/2607.15129v1)

**作者:** Jonathan Rainer Lippert `[一作]` (Technical University of Darmstadt), Alap Kshirsagar `[通讯]` (Technical University of Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

该论文提出了一套实时规划与控制体系，使机器人能够与人类合作完成多球杂耍，并在实验中实现了单球、双球和三球的持续同步杂耍。

**💡 创新点**

创新点在于将预测球轨迹、基于多步射击的在线轨迹优化以及基于状态机的交互协调逻辑集成到一体化框架中，从而在真实人机环境下实现了连续、鲁棒的多球杂耍。

**🔧 技术方法**

主要技术包括OptiTrack运动捕捉+Kalman滤波预测球轨迹、弹道学拟合、基于多步射击的约束优化生成关节轨迹、PD+逆动力学前馈执行、以及状态机决策层。

**📊 数据集**

实验数据来源于八名不同技能水平的杂耍者进行的人机配合杂耍实验（使用OptiTrack记录球轨迹）以及基于MuJoCo的仿真测试。

**📈 对比分析**

与以往最高记录（4次机器人接球）的基准相比，本系统在三球杂耍中实现了最多20次连续接球（超越5倍），单球模式达到100%成功率；实验统计显示成功率和连续接球次数均显著提升。

**⚠️ 局限性**

主要局限包括：对时间余量的高度依赖导致高球数时难以保证空闲阶段；缺乏对球与球碰撞的显式规划与避让；4-DOF机械臂在边缘工作空间需要大倾斜角，降低接球成功率；以及实验对运动捕捉系统的依赖。

---

## 460. Female participation in science in the past 125 years: An analysis of the Matilda effect over time

**arXiv ID:** 2607.15059 | [PDF](https://arxiv.org/pdf/2607.15059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 461. DAPGNet: Dynamic Adaptive Physics-Guided Graph Diffusion Network for Hyperspectral Image Classification

**arXiv ID:** 2607.15128 | [PDF](https://arxiv.org/pdf/2607.15128v1)

**作者:** Pengkun Wang `[一作]` (Chinese Academy of Sciences), Xiaofei Yang `[通讯]` (Guangzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种动态自适应物理引导图扩散网络（DAPGNet），通过将连续波段的光谱物理先验融入节点表征、图拓扑构造和图扩散，实现高光谱图像分类。

**💡 创新点**

创新点包括：多尺度光谱物理先验编码、基于物理先验的自适应图构造、边权加性注意偏置与物理门控的图扩散，使物理先验在关系层级持续发挥作用。

**🔧 技术方法**

采用多尺度1D卷积+注意力的光谱先验编码、双阶段自适应图构造（候选筛选+MLP边权）、稀疏注意力图扩散与物理门控融合、交叉层特征融合以及主/辅助交叉熵+光谱平滑正则等技术。

**📊 数据集**

使用了四个公开高光谱数据集：Indian Pines、WHU‑Hi‑LongKou、Houston2013 和 Houston2018。

**📈 对比分析**

与多种CNN、Transformer、Mamba、图卷积等基线进行比较，DAPGNet 在 OA、AA、Kappa 上均获得最高分，AA 提升 3.64–7.31%。

**⚠️ 局限性**

局限性在于仅利用波段顺序的结构先验，未引入传感器响应、辐射传输等更丰富的物理信息；图构造局限于局部 patch，缺乏全局关系建模；对不同光谱采样不具通用性。

---

## 462. Towards Hierarchical Structure Understanding of Newspaper Images

**arXiv ID:** 2607.15082 | [PDF](https://arxiv.org/pdf/2607.15082v1)

**作者:** William Mocaër `[一作]` (University of Rouen Normandy), Thierry Paquet `[通讯]` (University of Rouen Normandy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了报纸图像的层次结构理解，提出了基于模块化管线和端到端Transformer两种方法。

**💡 创新点**

创新点在于提出Tiramisu层次Transformer模型和高效的bottom‑up管线组合，提供针对报纸的层次结构学习与读取顺序预测。

**🔧 技术方法**

技术包括YOLO+LSD+LayoutReader管线、Swin Transformer编码器、Transformer解码器、multi‑pass解码与prompting。

**📊 数据集**

使用Finlam La Liberté历史报纸数据集以及合成报纸生成器。

**📈 对比分析**

与Arcanum、Tiramisu、pipeline对比，bottom‑up在块检测mAP 72%、文章F1 80%以及读取顺序BLEU block 87%表现最佳；Tiramisu在结构计数指标更优；两者均快于Arcanum。

**⚠️ 局限性**

局限在于Tiramisu依赖大量合成数据且早期缺失会级联误检；pipeline因多模型链条复杂，部署维护成本高；数据集标注噪声及OCR质量受限。

---

## 463. Evaluating covariate balance for long time horizon Markov decision processes

**arXiv ID:** 2607.15080 | [PDF](https://arxiv.org/pdf/2607.15080v1)

**作者:** Joshua Spear `[一作]` (UCL GOS Institute of Child Health), Neil J Sebire `[通讯]` (NIHR GOSH UCL BRC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过引入协变量平衡诊断方法，对离线强化学习（Offline RL）在重症监护（sepsis）治疗决策中的应用进行评估，揭示了现有研究中存在的潜在隐藏偏倚和模型错误指定。

**💡 创新点**

创新点包括：1) 将协变量平衡概念推广到时间依赖、连续状态空间的离线RL场景；2) 建立了条件协变量平衡与加权协变量平衡的形式化定义；3) 通过多种重要性采样（vanilla、clipped、Hajek）与MASD指标系统性评估诊断效果；4) 通过动作空间缩减和轨迹截断探讨高维度导致偏差的根源。

**🔧 技术方法**

使用的技术包括：XGBoost（加权与无加权）构建倾向评分模型；重要性采样估计、权重截断与Hajek重权重；MASD（平均绝对标准化差）作为协变量平衡量；对齐离线RL的Markov决策过程（MDP）假设；多种实验配置（全动作空间、仅液体剂量、不同时间截断）。

**📊 数据集**

使用的主要数据集是公开的重症监护记录（如MIMIC‑III）中的sepsis患者轨迹，包含24小时预处理窗口与48小时后期窗口，总共约18步；此外对全动作空间（25维）与液体剂量子空间（5维）分别构建模型；实验中还采用不同截断长度（2、5、8、13步）和两种随机种子。

**📈 对比分析**

比较方法：对照未加权与加权倾向评分模型、不同重要性采样方案以及不同时间截断与动作空间设置。性能指标主要是MASD阈值（<0.25）以及各时间步的均值/中位数。结果表明：在全动作空间下，MASD大多超过阈值，且随时间延长波动加剧；仅液体剂量空间略有改善但仍不达标；截断时间可在前2–3步内实现一定平衡，但整体提升有限。与传统方法相比，重要性采样的方差抑制技术（clipped、Hajek）并不能可靠判断协变量平衡，甚至可能掩盖真实偏差。

**⚠️ 局限性**

局限性包括：1) 高维时间与动作空间导致重要性采样方差过大，诊断结果受限；2) 诊断阈值（0.25）基于经验，缺乏理论保障；3) 偏倚估计使用的倾向评分模型仍受数据分布不平衡影响，无法完全消除模型误差；4) 本研究仅聚焦于sepsis领域，结果可能不易推广至其他医疗决策场景；5) 未能提供完整的联合倾向评分建模或PAC‑guaranteed估计方法。

---

## 464. When AI Blurs the Boundaries of Contribution: An Empirical Study of Authorship Calibration

**arXiv ID:** 2607.15006 | [PDF](https://arxiv.org/pdf/2607.15006v1)

**作者:** Célina Treuillier `[一作]` (University of Fribourg), Denis Lalanne `[通讯]` (University of Fribourg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并量化了 AI 辅助写作中的作者身份校准概念，探究其与 AI 使用频率的关系。

**💡 创新点**

首次定义作者身份校准，将用户自评贡献与实际贡献对齐，揭示高频 AI 使用导致的误差偏大。

**🔧 技术方法**

采用问卷估计用户贡献、文本句子匹配算法计算实际贡献，以及统计检验（如 Mann‑Whitney U）评估校准度。

**📊 数据集**

使用斯坦福 CoAuthor 数据集（1,252 篇写作会话，包括创意与论证写作）。

**📈 对比分析**

通过对低 AI 使用组与高 AI 使用组的对比，发现后者校准误差更大且差异显著（p<0.05），但未给出传统性能指标。

**⚠️ 局限性**

局限在于仅聚焦写作任务、采用简单中位数划分 AI 使用频率、未在真实教学环境中验证。

---

## 465. DriftWorld: Fast World Modeling through Drifting

**arXiv ID:** 2607.15065 | [PDF](https://arxiv.org/pdf/2607.15065v1)

**作者:** Susie Lu `[一作]` (Massachusetts Institute of Technology), Yilun Du `[通讯]` (Harvard University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于漂移模型的单步动作条件世界模型 DriftWorld，可在一次前向传播中生成未来帧，实现快速高质量的机器人世界建模和策略搜索。

**💡 创新点**

将漂移生成模型迁移到动作条件视频生成，提出动作放大漂移场、特征空间漂移损失和帧级动作条件的 U‑Net 结构，并实现平均 17 倍的推理速度提升。

**🔧 技术方法**

漂移生成模型、DINOv2/v3 特征编码器、FiLM 条件化、单步前向推理、动作增强漂移场与运动加权损失。

**📊 数据集**

Bridge‑V2、RT‑1、Language Table、Push‑T、Robomimic 等五大视觉机器人操纵基准。

**📈 对比分析**

与多种扩散及其它世界模型基线（IRASim、Ctrl‑World、GPC 等）在 SSIM、PSNR、LPIPS、FID、FVD 等视觉质量指标和跑时测评中对比，DriftWorld 在保持或超过视觉质量的同时平均提升 17 倍速度，并在推理时策略改进和离线评估中获得更高的 IoU/成功率相关系数。

**⚠️ 局限性**

依赖强大的预训练特征提取器；训练时内存消耗大，需多负样本；对长时序一致性有限，需更长上下文或稀疏历史改进。

---

## 466. A lower bound of 4 for online graph exploration

**arXiv ID:** 2607.15113 | [PDF](https://arxiv.org/pdf/2607.15113v1)

**作者:** Julia Baligacs `[一作]` `[通讯]` (University of Oxford), Julia Baligacs (University of Oxford)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在在线图探索问题中给出了一个新的对手构造，证明任意确定性算法的竞争比至少为4（此前最优下界为10/3）并且此下界对平面、三度子图仍成立。

**💡 创新点**

创新点在于将多种约束（如边权满足三角不等式、仅学习边权而非相邻顶点ID等）证明可在不降低竞争比的前提下强制化，同时设计了一种递归块构造和块遍历问题，从而实现更严格的3倍后退成本估计。

**🔧 技术方法**

使用了对手（adversary）策略、递归块构造、平面图树替换技巧以及竞争分析中的潜在函数思想。

**📊 数据集**

无，研究为纯理论性，未使用任何实验数据集。

**📈 对比分析**

通过与已知最优算法的竞争比对比，证明任何算法的下界至少为4；上界仍为O(log n)，但该论文未给出实现算法，只提供了下界分析。

**⚠️ 局限性**

局限性在于只给出了常数下界，未进一步逼近已知上界；构造仅适用于平面三度图；对超常数竞争比的研究并未覆盖；并且该构造会导致图中顶点数激增。

---

## 467. Learning in Infinitesimal Non-Compositional Sketches

**arXiv ID:** 2607.15107 | [PDF](https://arxiv.org/pdf/2607.15107v1)

**作者:** Sridhar Mahadevan `[一作]` `[通讯]` (Adobe Research), Sridhar Mahadevan (Adobe Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出了Lincs框架，将机器学习视为对图式（sketch）非组合性（non‑compositionality）的修复，并引入“无穷小非组合性”（Infinitesimal Non‑Compositionality, INC）来衡量在切线（tangent）范畴中的失配。通过学习图式、切线学习图式以及交互增强的INC，构建了一个以切线上升为核心的类别学模型，并给出了一种以coalgebra为基础的收敛与最终性理论。

**💡 创新点**

创新点包括：
- 将传统的损失函数抽象为图式的因子化失配，提出无穷小非组合性作为新型学习信号；
- 引入切线学习图式，使得学习约束在切线结构下保持稳定；
- 构造交互增强的INC，包括李括号闭包、连接加速等二阶信息；
- 给出最终INC coalgebra的存在性证明（Aczel–Mendler和Barr定理）和在可度量化实现中的几何收敛结果；
- 提出了以INC为核心的自洽的Lincs类别及其自由完备性猜想。

**🔧 技术方法**

使用的技术主要有：
- 图式（sketch）与Cockett–Cruttwell切线类别的类别学语义；
- 切线提升（tangent lift）和Weil代数的分类；
- coalgebra与最终对象论证（Aczel–Mendler定理、Barr定理）；
- 交互签名（Lie brackets、连接、曲率）以及二阶Jet的几何结构；
- 可度量实现中的contractive迭代与几何收敛。

**📊 数据集**

本文尚未给出具体的数据集实验，作者提到“实验评估正在进行中”，但未列出任何具体数据集或实验结果。

**📈 对比分析**

由于缺乏实验结果，无法给出方法的性能对比。目前仅提供了理论上对INC信号的描述以及在可度量化实现中几何收敛的上界，尚未与传统损失或其它正则化方法进行实测比较。

**⚠️ 局限性**

局限性：
- 主要为理论框架，缺乏完整的实证验证；
- 对可度量化实现的收敛性和最终性依赖于假设（contractive、可访问等），实际可实现性尚待检验；
- 交互增强的INC依赖于具体的连接或李括号声明，缺乏通用性；
- 高阶无限小结构的实现与计算成本尚未讨论；
- 对于非平滑或离散问题的切线类别扩展仍未给出完整的技术细节。

---

## 468. DataShield: Uncovering Risky Fine-Tuning Data Across LLMs Through Consensus Subspace Alignment

**arXiv ID:** 2607.15081 | [PDF](https://arxiv.org/pdf/2607.15081v1)

**作者:** Zefeng Wu `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种数据驱动的安全保护框架DataShield，利用多种安全对齐LLM的共识子空间对微调数据进行风险评估，并根据风险水平对样本进行过滤或对文本片段进行掩码，提升微调后模型的安全性。

**💡 创新点**

创新点包括①基于多模型共识的安全/不安全子空间构建，取代单向量或单模型的风险估计；②对文本进行与分词器无关的语义片段划分，保证掩码可跨模型迁移；③自回归风险解耦技术，将新增风险与先前上下文风险分离，精准定位危险片段。

**🔧 技术方法**

采用语义谱分解提取安全/不安全子空间，利用子空间对齐度作为风险度量；通过对齐差值（unsafe vs. safe）构造样本/片段风险；实现样本过滤和片段掩码；使用LoRA进行目标模型微调。

**📊 数据集**

使用安全对齐源模型（Llama3‑8B、Qwen2.5‑7B、Mistral‑7B）提取特征；微调数据集为Alpaca与Dolly；安全评估采用HEx‑PHI和HarmBench；下游性能评估使用SLIMORCA。

**📈 对比分析**

与随机过滤、Bi‑Anchor、SEAL、LARF、SOT等样本级过滤器以及Random‑Sm、TOSS等片段级掩码器在相同预算下进行对比。DataShield在样本过滤时降低ASR约14.6%，在片段掩码时降低约32.3%，并在未见目标模型（Phi3、Qwen3、Gemma2/3）上保持低风险和较高下游效能；相比使用目标模型风险信息的基线，DataShield仍取得更优性能，并且显著降低内存和时间成本。

**⚠️ 局限性**

实验仅覆盖监督微调、指令调优LLM、两种数据集和常用安全基准，未考察多语言、专业领域或更大规模模型；对安全评估仅基于通用有害请求数据，可能不涵盖特定行业安全场景。

---

## 469. Kernel weighted importance sampling for off-policy evaluation in contextual bandits

**arXiv ID:** 2607.15067 | [PDF](https://arxiv.org/pdf/2607.15067v1)

**作者:** Joshua Spear `[一作]` (Institute of Child Health University College London), Erica E. M. Moodie `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的 Kernel‑WIS 估计器，用于仅用离线数据对时间无关的情境 bandit 进行离策略评估 (OPE)。

**💡 创新点**

创新点在于将传统的无权重重要性采样 (VIS) 的线性独立性与加权重要性采样 (WIS) 的有界性通过核加权融合，既保留了 WIS 的方差控制，又利用了 VIS 的可加性；并证明了该估计器在合适的带宽递归条件下是一致的。

**🔧 技术方法**

使用了核加权重要性采样、交叉验证 (CV) 进行带宽选择、解析梯度求导、状态依赖 WIS 与 CLPD‑VIS 等基线比较、以及统计检验（两侧 Wald t‑检验）。

**📊 数据集**

在十个公开机器学习数据集（如 arrhythmia、soybean、micro‑mass、optdigits、yeast、page‑blocks、pendigits、letter、kropt 等）上构造了半模拟情境 bandit，并用真实标签生成奖励。

**📈 对比分析**

通过与 VIS、WIS、State‑WIS、CLPD‑VIS 的对比，Kernel‑WIS 在 oracle 行为策略下与 WIS 的表现相当；在非 oracle（行为策略误设）下在大多数情形中统计显著优于 WIS；但在连续奖励设置下性能显著下降，且在某些数据集上与 WIS 近似或略逊。

**⚠️ 局限性**

主要限制包括：估计器并非在所有情形下优于 WIS；带宽优化困难、优化过程非凸；理论分析假设固定评估与行为策略并且带宽趋于 0；缺乏 PAC‑Bayes 边界；对不同核函数或深度策略的泛化性不明；以及目前仅适用于时间无关的情境 bandit。

---

## 470. Beyond Single Expert: Harmonizing Diverse Visual Priors in MLLMs for Spatial Understanding

**arXiv ID:** 2607.15054 | [PDF](https://arxiv.org/pdf/2607.15054v1)

**作者:** Xiao Lin `[一作]` (University of Hong Kong), Kai Han `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了如何将多种视觉基础模型的先验知识融合进多模态大型语言模型，以提升空间理解能力，并提出了 ViPS 框架。

**💡 创新点**

创新点在于：①使用高效先验代理（Efficient Prior Proxy）通过轻量级 MLP 只需一次前向传播即可近似多模型先验；②设计动态先验融合（Dynamic Prior Fusion），利用输入任务上下文生成权重，并通过零初始化卷积实现平滑注入，避免多模型先验分布冲突。

**🔧 技术方法**

采用技术包括轻量级 MLP 代理、零初始化卷积、动态权重网络、对齐损失、LoRA 微调、视频视觉编码器（VGGT 等）以及 Qwen2‑VL / Qwen3‑VL 等 LLM。

**📊 数据集**

使用的数据集有 VSI‑Bench（空间推理任务）以及 ScanNet 系列的五个基准：ScanRefer、Multi3DRefer、Scan2Cap、ScanQA 与 SQA3D。

**📈 对比分析**

在与现有多模态 LLM 与专门 3D 视觉模型的对比中，ViPS 在 VSI‑Bench 上平均得到 63.8% 的分数，超过此前最佳 57.2%；在 ScanNet 系列中，尤其在视觉定位（ScanRefer）和问答（ScanQA）任务上取得最高或接近最高的表现，整体性能显著提升。

**⚠️ 局限性**

局限性包括：①对单一视觉编码器基础模型的依赖仍限制了表达范围；②动态融合虽然有效但增加了少量计算开销；③多模型代理与真实先验之间仍存在对齐误差，对极端视觉输入或更广泛多模态任务的泛化性尚未充分验证。

---

## 471. Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya

**arXiv ID:** 2607.15209 | [PDF](https://arxiv.org/pdf/2607.15209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 472. RoGS: Adaptive Meshgrid Gaussian for Large-Scale Road Surface Mapping

**arXiv ID:** 2607.15048 | [PDF](https://arxiv.org/pdf/2607.15048v1)

**作者:** Tianchen Deng `[一作]` (Shanghai Jiao Tong university), Hesheng Wang `[通讯]` (Shanghai Jiao Tong university)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个基于自适应网格2D高斯surfels的道路表面建模框架RoGS，并通过多姿态一致性引导的姿态鲁棒优化实现高质量RGB、语义和高程地图重建。

**💡 创新点**

创新点：①使用2D高斯surfels贴合薄层道路特性，显著减少冗余原语；②引入道路结构感知的自适应网格，按复杂度密度分配surfels；③采用轨迹一致性引导的多姿态姿态鲁棒初始化，降低姿态噪声影响；④整体实现高效可扩展的优化流程。

**🔧 技术方法**

技术：meshgrid高斯surfels表示、可微分高斯渲染、语义与RGB渲染监督、平滑与轨迹一致性正则化、可选LiDAR高程监督。

**📊 数据集**

数据集：KITTI Odometry和nuScenes（共1000场景），使用周边视角图像、车辆姿态、Mask2Former语义分割结果，以及可选LiDAR点云。

**📈 对比分析**

与现有mesh-based方法RoMe对比，RoGS在保持相似或更高PSNR的同时提升mIoU至90%+、降低高程RMSE至0.05m，单轮训练速度提升约53×，两轮提升约27×，且在夜/雨等恶劣条件下表现更稳健。

**⚠️ 局限性**

局限：高度依赖精确姿态，姿态误差仍可能影响初始高程；纯视觉/语义仅能恢复有限高程信息，需LiDAR或其他外部高程数据补充。

---

## 473. Learning Agile Navigation in Crowded Environments for Quadruped Robots

**arXiv ID:** 2607.15036 | [PDF](https://arxiv.org/pdf/2607.15036v1)

**作者:** Shuyu Wu `[一作]` (Shanghai Jiao Tong University), Danping Zou `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了面向四足机器人在拥挤动态环境中的端到端导航框架VOP-Nav，结合速度障碍（VO）原理与强化学习，实现安全与敏捷兼顾的导航；

**💡 创新点**

创新点包括：1）VOP-Net通过多帧LiDAR直接回归安全速度区间，避免显式障碍检测与跟踪；2）VO预测既作为推理时的状态输入，又作为训练时的奖励信号，双重引导策略学习；3）实现零调优的从仿真到真实世界的迁移；

**🔧 技术方法**

采用Velocity Obstacle理论、Proximal Policy Optimization（PPO）强化学习、MobileNet+注意力机制的深度网络处理二维LiDAR与深度相机数据，配合Isaac Gym模拟训练；

**📊 数据集**

使用在Isaac Gym构建的多种动态拥挤场景生成的数据集，包含训练、森林、办公、方形和慢速四个测试环境；以及在Unitree Go2机器人上收集的真实世界实验数据；

**📈 对比分析**

与ORCA、NavRL、HEIGHT、REASAN、ABS等基线比较，VOP-Nav在所有测试环境中取得最高成功率、最低碰撞率，速度保持竞争力；实测室内100%成功率，户外约80%成功率，显示出优异的性能与鲁棒性；

**⚠️ 局限性**

局限性：1）仅使用二维平面LiDAR，忽略3D通行性与悬空障碍；2）对精准定位依赖较高，户外LIO漂移会影响性能；3）在极端拥挤或复杂三维环境中，安全速度预测精度仍有提升空间。

---

## 474. Differentiable Routability-Driven Package Floorplanning with Pin Assignment

**arXiv ID:** 2607.15005 | [PDF](https://arxiv.org/pdf/2607.15005v1)

**作者:** Yiqi Huang `[一作]` (Fuzhou University), Tsung-Yi Ho `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对fan-out Wafer-level Packaging (FOWLP) 的路由风格，提出了一套三阶段可微分的包级布局与引脚分配算法，联合优化芯片位置、方向、引脚分配及路由资源，以实现100%路由可行性并显著缩短路由线长。

**💡 创新点**

创新点包括：① 针对fan-out风格的可微分拥塞估计模型，区分 fan‑in/fan‑out 区域并利用 Top‑K 路径加权估算路由需求；② 直接在离散芯片方向上使用 Gumbel‑Softmax 的可微分实现，避免连续角度的初始化偏差；③ 结合航线碰撞成本的引脚分配粒子群优化 (DPSO)，通过 GPU 并行加速提高搜索效率；④ 两阶段可微分路由拥塞损失，先保证基本通路后再精细匹配拥塞需求。

**🔧 技术方法**

技术手段包括：可微分 HPWL + 软化密度约束；Gumbel‑Softmax 方向选择；GPU 并行引脚评估与飞行线碰撞计数；Top‑K Yen 路径与拥塞损失的梯度反向传播；ILP 位置合法化；粒子群更新策略与多策略分组。

**📊 数据集**

使用了十个在 RDL 路由研究中广泛引用的 FOWLP 基准：dense1~dense5、pkg1~pkg5，包含 2–20 个芯片、20–522 I/O、10–261 信号网，面积从 5000×5000 到 16000×12000 μm²。

**📈 对比分析**

与现有最先进方法 (Lin 等) 对比：在所有基准上实现 100% 路由可行性；在已路由成功的案例中，平均路由线长比 Lin 方法缩短约 23%（最大 84%），路由时长比 Lin 方法略长 0.35 复合指标，但相比 Lin+PA 提升至 1.00；与 SA / MIS 引脚分配相比，速度更快、线长更优，GPU 加速平均 112×。

**⚠️ 局限性**

局限性：① 可微分拥塞模型需要频繁重建路由图，对极大规模设计的计算开销仍显高；② 采用的 Top‑K 只考虑前 K 条路径，可能忽略更优但非前 K 的路径；③ 需要预先给定 Gumbel‑Softmax 温度、权重参数等，调参较多；④ 对于 fan‑in 路由需求极高的情况，模型仍需进一步验证。

---

## 475. Mask-Aware Policy Gradients for Diffusion Language Models

**arXiv ID:** 2607.15200 | [PDF](https://arxiv.org/pdf/2607.15200v1)

**作者:** Haran Raajesh `[一作]` (University of Texas at Austin), Philipp Krähenbühl `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对Masked Diffusion Language Models（MDLMs），提出基于两阶段动作MDP的强化学习框架，联合优化 token 预测与位置 remasking 决策。

**💡 创新点**

创新点在于将 MDLM 生成过程拆分为 token 与位置两阶段动作，构造可分解的策略梯度，使得原本忽视的 remasking 信息得以用于强化学习，从而显著提升生成质量。

**🔧 技术方法**

使用 Policy Gradient、两阶段 MDP 结构以及自研的 Mask‑Aware Policy Gradients 方法，并在 LLaDA‑8B‑Instruct 预训练模型基础上实现。

**📊 数据集**

实验使用数学推理数据集 GSM8K、MATH500，以及代码生成数据集 MBPP、HumanEval。

**📈 对比分析**

在统一使用 LLaDA‑8B‑Instruct、生成长度 128 的设置下，与现有 ELBO 及轨迹近似方法对比，取得 GSM8K 87.1% 与 MBPP 53.4% 的最佳成绩，并在 MATH500、HumanEval 上分别提升 4.0% 与 2.9%。

**⚠️ 局限性**

局限性包括：需要额外计算 remasking 概率，训练成本较高；对极大规模模型或多语言场景的泛化能力尚未充分验证；并且依赖于 mask 设计，若 mask 方式不当可能导致性能下降。

---

## 476. T^2MLR: Transformer with Temporal Middle-Layer Recurrence

**arXiv ID:** 2607.15178 | [PDF](https://arxiv.org/pdf/2607.15178v1)

**作者:** Ziyang Cai `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Transformer中间层递归架构，允许前一步的深层表示直接注入当前步浅层，从而在不改变自回归接口的情况下保持抽象中间状态。

**💡 创新点**

创新点在于仅在网络中部引入递归路径，显著提升推理能力且仅带来约8%每token推理开销，且不需要从头预训练。

**🔧 技术方法**

使用自回归Transformer、带门控融合模块、Jacobi近似训练、时间并行化，构建Temporal Middle‑Layer Recurrence（TMLR）模型。

**📊 数据集**

在10B FineWeb‑Edu预训练、S5‑Retrieval、GSM‑8K、MATH500、HotPotQA、ProsQA、Variable Assignment等数据集上进行评估。

**📈 对比分析**

与参数匹配的SmolLM2、Looped、Pause‑token、Full‑looped等基线相比，TMLR在下游零样本和微调任务中平均提升10‑15%准确率，且推理成本几乎不变。

**⚠️ 局限性**

主要限制是训练时的计算开销显著增加（2–4×壁钟时间），且需要多步Jacobi迭代来近似递归状态。

---

## 477. Benchmarking Multimodal Large Language Models for Scientific Visualization Literacy

**arXiv ID:** 2607.15176 | [PDF](https://arxiv.org/pdf/2607.15176v1)

**作者:** Patrick Phuoc Do `[一作]` (University of Notre Dame), Chaoli Wang `[通讯]` (University of Notre Dame)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估多模态大语言模型在科学可视化（SciVis）理解上的素养，使用 SVLAT 测评工具进行系统实验。

**💡 创新点**

首次将专为 SciVis 设计的 SVLAT 作为闭合世界评测基准，结合细粒度技术和任务分析，揭示模型在不同可视化技术和任务类型上的优势与瓶颈。

**🔧 技术方法**

采用统一的结构化 JSON prompt、温度 0、max_tokens 300，并对每道题目进行 10 次重复评测，帧抽取处理动画；同时对模型给出的推理理由进行定性错误分析。

**📊 数据集**

使用 SVLAT（49 题、18 个可视化、8 种技术、11 任务类型）作为评测数据集，并以人类非专家基准作为对照。

**📈 对比分析**

以准确率为主指标，计算每模型的平均值与标准差；Gemini 在图像 90.9%、动画 82.9% 的表现远超人类 76.4%、74.5%，开放模型整体低于人类，Qwen 最高 68.8%。

**⚠️ 局限性**

限制包括动画仅通过帧抽取评估、仅使用单一基准 SVLAT、受 prompt 设计影响、部分任务样本极少导致评估不够稳健。

---

## 478. Linear representations of grammaticality in neural language models

**arXiv ID:** 2607.15175 | [PDF](https://arxiv.org/pdf/2607.15175v1)

**作者:** Jane Li `[一作]` (Johns Hopkins University), Najoung Kim `[通讯]` (Boston University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究神经语言模型是否在内部表示中编码语法性，并通过“mass‑mean”线性探针对多种模型和多种语言进行评估。

**💡 创新点**

提出了基于表示的语法性探测方法，能够在不依赖概率或接受度的前提下揭示语法性特征，并展示了跨现象、跨语言的可推广性。

**🔧 技术方法**

采用 mass‑mean 探针、回归分析、边界倾斜（boundary tilting）技术以及统计显著性检验，对模型隐藏层表示进行线性投影和解码。

**📊 数据集**

使用 BLiMP、SCaMP、LI‑Adger、RUBLiMP、ZhoBLiMP、CLAMS 等语法性基准，同时利用 Shades、Drive、Truth 等非语法性（接受度）数据集进行去混淆实验。

**📈 对比分析**

将探针准确率与概率基的最小对比评测（MP Rep 与 MP Prob）相比较；大模型（≥1B 参数）在 60–70% 之间的准确率下表现出显著成功率，且探针在许多现象上能与概率方法互补。

**⚠️ 局限性**

局限性包括：小型模型的语法性信号弱；跨语言推广存在差异；探针虽然去混淆了一部分接受度信息，但仍可能与语义可接受度相关；未深入探讨深层句法表示与语法性边界的因果关系。

---

## 479. Assessing Physical Frailty and Fall-Risk Indicators with Social Robots: An in situ Evaluation with Older Adults

**arXiv ID:** 2607.15156 | [PDF](https://arxiv.org/pdf/2607.15156v1)

**作者:** Aniol Civit `[一作]` (Institute of Robotics and Informatics), Guillem Alenyà `[通讯]` (Institute of Robotics and Informatics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种基于社交机器人的自主衰弱与跌倒风险评估框架，机器人通过视觉骨架跟踪引导老年人完成SPPB、TUG等标准测试，并同时记录测试时间及多项运动指标。

**💡 创新点**

创新点在于：①使用行为树实现完整的自助评估流程；②通过无穿戴式视觉骨架跟踪获取丰富的生物力学指标；③在真实临床环境下对81名老年人进行大样本验证，并与治疗师及专业传感器（走道传感器、IMU）进行对比，验证机器人评估的可靠性。

**🔧 技术方法**

技术包括：Temi社交机器人（触摸屏、语音预录、移动底盘）；ZED2i深度相机+ZED SDK实现实时3D骨架跟踪；行为树（Behavior Tree）做决策与任务调度；测量模块计算完成时间、步幅、步速、倾斜角、加速度、功率等指标。

**📊 数据集**

数据来源为现场采集的81名老年人实验数据，包含机器人记录、治疗师手工记录、走道传感器（GaitRITE）和单个IMU（Xsens）记录的运动数据。

**📈 对比分析**

评价方法：对连续指标使用ICC(2,1)，对分级评分使用加权Cohen's κ；结果显示大多数测试时间ICC>0.9，TUG时间ICC≈0.98，步长等步态指标ICC≥0.84；Sppb总分与治疗师比较κ≈0.67，站立平衡得分与机器人一致性低（κ≈0.39）。

**⚠️ 局限性**

限制包括：站立平衡测试受扶手/衣物影响导致误判；机器人在部分场景需治疗师干预（17/81）；黑色服装导致关节检测噪声；未能检测到抓住扶手；实验仅在受监管环境下进行，真实无人监管的安全性仍需提升。

---

## 480. Mutable Low-Rank Sketches for Retrain-Free Recommendation

**arXiv ID:** 2607.15242 | [PDF](https://arxiv.org/pdf/2607.15242v1)

**作者:** Hector J. Garcia `[一作]` (University of Michigan), Nick Clayton `[通讯]` (Criteo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出可变草图（mutable sketches），通过在每个用户的KP树中维护稀疏向量，并使用一次性拟合的低秩投影即时重算用户嵌入，从而实现无模型重训练的在线更新。

**💡 创新点**

创新点包括：①利用KP树实现对用户向量的O(log n)插入与范数加权采样；②一次性拟合低秩投影V_k，随后仅通过投影即刻更新嵌入；③证明每次新观测会单调收敛误差；④在不同稀疏度下比较采样策略，发现norm‑proportional采样在稀疏场景显著优于均匀采样。

**🔧 技术方法**

技术手段：KP树（稀疏段树）、随机矩阵草图、截断SVD、norm‑proportional采样、固定低秩基投影、在线O(log n)更新、漂移触发的重拟合、FAISS等近邻检索。

**📊 数据集**

使用的数据集：KuaiRec、Amazon Electronics 2023、Amazon Video Games 2023、Amazon Music 2023、Book‑Crossing、ML‑1M、Goodreads Comics 等，覆盖从稠密到极稀疏的六个环境。

**📈 对比分析**

与ALS、eALS、FunkSVD、Full SVD等全量基线对比：在KuaiRec上仅读取1.8%数据即获得0.810 RMSE（相比ALS的0.822），在线更新每批仅需90 ms并逐步收敛到0.818；冷启动仅1条评分即可在<1 ms内得到个性化推荐，RMSE仅比全量ALS差0.06；在稀疏数据中norm‑proportional采样提升40–130%覆盖率。

**⚠️ 局限性**

局限性：对大规模商品集合稀疏度高时草图覆盖不足；固定V_k随数据分布漂移会导致时效性下降，需要定期重拟合；不直接支持非ID侧特征；GPU并行性差，适合CPU稀疏场景；在需要最高精度或复杂特征融合的任务中效果不佳。

---

## 481. Divergent Gaze Patterns in Artistic Viewing: Spatial and Temporal Signatures of Attention Across Autistic Individuals, Artists, and Neurotypical Observers

**arXiv ID:** 2607.15227 | [PDF](https://arxiv.org/pdf/2607.15227v1)

**作者:** Mohammed Amine Kerkouri `[一作]` (F-initiatives), Nadia Aguillon-Hernandez `[通讯]` (University of Tours)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对30幅彩色绘画进行15秒自由观看的眼动追踪，比较了自闭症成年人、受过艺术训练的艺术家与神经典型观众的视觉注意模式。

**💡 创新点**

创新之处在于提出了双轴（空间与时间）定向、度量基础的比较框架，将群体间的密度图与扫描路径分别作为预测模型和序列对齐，揭示自闭症群体既具艺术家般的空间散布，又表现出独特且高度个体化的时间动态。

**🔧 技术方法**

采用了多种已验证的视觉注意指标（AUC‑Judd、NSS、CC、SIM、KL、IG）、扫描路径比较方法（MultiMatch、ScanMatch、FDISS、DTW）以及统一的离散阈值提取算法，对所有参与者的眼动数据进行处理与评估。

**📊 数据集**

数据集包含24名艺术家、32名神经典型观众与15名自闭症成年人，共计2,091条扫描路径与95,282次注视，刺激为30幅多样化的绘画作品。

**📈 对比分析**

通过定向的空间预测和时间序列对齐，对比组间的相似度得分显示艺术家与神经典型观众在空间与时间上几乎一致（例如CC≈0.96），而自闭症观众的预测得分显著较低，且在时间轴上最不自洽，表现出比其他两组更高的个体差异。

**⚠️ 局限性**

局限性包括自闭症样本量较小、群体间年龄与性别不匹配、使用池化密度图忽略个体差异、统一提取阈值可能影响绝对数值、缺乏语义或对象级基线以区分刺激驱动与群体策略。

---

## 482. When Words Are Safe But Actions Kill: Probing Physical Danger Beyond Text Safety in Hidden-State Risk Space

**arXiv ID:** 2607.15218 | [PDF](https://arxiv.org/pdf/2607.15218v1)

**作者:** Weimeng Wang `[一作]` (Tsinghua University), Ke Xu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在具身代理中，语言模型生成的指令在物理执行后可能出现的安全风险，并探讨物理危险是否与文本内容危险是同一安全问题。

**💡 创新点**

创新点在于：①通过隐藏状态方向分析发现内容危险（CD）与物理危险（PD）在LLM隐藏层空间可分离；②提出PRISM，一种基于单层L2正则化逻辑回归的隐藏状态探测器；③构建PSB-1K对比性基准，专门测试物理危险而非文本危险。

**🔧 技术方法**

使用技术包括隐藏状态方向估计、线性探针（logistic回归）与多层感知机、支持向量机、LPM原型距离；对LLM的中间层隐藏状态进行标准化与投影。

**📊 数据集**

使用的数据集包括SafeAgentBench、SafeText、EARBench以及自构造的PhysicalSafetyBench-1K，模型覆盖Qwen2.5系列、Phi-3.5-mini、SmolLM2-1.7B。

**📈 对比分析**

在所有基准上，PRISM在内容和物理危险检测上保持86-88%准确率，FPR仅约12-14%；相比之下，LLM judge在物理危险上召回率高达95%但FPR高达39%；在PSB-1K上PRISM取得99.6%准确率、0.7% FPR，远优于判别器和Llama Guard。

**⚠️ 局限性**

局限性包括：PRISM仍依赖于预训练LLM的隐藏状态；仅针对指令级安全检测，未覆盖动作执行阶段的完整安全链；以及在极小模型上（如SmolLM2-1.7B）判别器仍可能过度拒绝安全指令。

---

## 483. MAGiSt3R: Multi-Agent Feed-forward 3D Reconstruction from Monocular RGB Videos

**arXiv ID:** 2607.15211 | [PDF](https://arxiv.org/pdf/2607.15211v1)

**作者:** Ziren Gong `[一作]` (University of Bologna), Matteo Poggi `[通讯]` (University of Bologna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出第一个多代理前向单目RGB视频3D重建框架，能够实时生成全局点云并实现相机轨迹跟踪。

**💡 创新点**

设计了多代理全局图聚合模块MAGMA，能够在 intra‑agent 与 inter‑agent 级别融合几何与视觉特征，实现高精度、实时的子图拼接。

**🔧 技术方法**

利用3R（VGGT）前向模型预测子图；构建多代理全局图聚合网络（几何+视觉注意力机制）；结合姿态图优化与RANSAC+ICP对齐。

**📊 数据集**

训练使用 ScanNet、ScanNet++ 与 Aria Synthetic 数据集；评估在 ReplicaMultiagent 与 AriaMultiagent 等多代理单目视频数据集。

**📈 对比分析**

与 DUSt3R、MASt3R、SLAM3R、VGGT‑SLAM 等前向方法及 RGB‑D SLAM 系统进行对比；在多代理场景下实现约10 FPS，重建精度与轨迹误差显著优于现有基线，接近 RGB‑D SLAM 的表现。

**⚠️ 局限性**

对极端光照/天气变化不具鲁棒性；实验仅覆盖室内场景且代理数目有限（≤3），尚未验证在大规模、多样化环境中的表现。

---

## 484. MM-IssueLoc: A Controlled Benchmark for Evaluating Visual Evidence in Multimodal Repository-Level Issue Localization

**arXiv ID:** 2607.15205 | [PDF](https://arxiv.org/pdf/2607.15205v1)

**作者:** Shaoxiong Zhan `[一作]` (Tsinghua University), Hai-Tao Zheng `[通讯]` (Tsinghua University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了多模态仓库级问题定位，提出了MM-IssueLoc基准并控制图像证据；

**💡 创新点**

创新点在于将图像证据作为显式评估变量，提供文件/函数级gold、图像类别与相关性标签、VCE诊断以及多种输入模式；

**🔧 技术方法**

使用LLM和检索模型（OpenHands、AgentLess、LocAgent、Mini-SWE-Agent、MM‑IssueLoc‑VL‑Embedding等）以及视觉内容提取（VCE）技术；

**📊 数据集**

基准数据集为MM‑IssueLoc，包含652个issue‑PR实例、1,050张图片、23种编程语言，提供文件级和函数级gold标签；

**📈 对比分析**

通过文本仅、图像、VCE、VCE+图像等模式进行对比，文件Acc@5最高达38.96%，函数Acc@10最高33.86%，表明现有系统在多模态定位上仍显弱；

**⚠️ 局限性**

局限性包括函数级标签噪声、图像数量最多两张、误导图像频率低、未覆盖端到端修复流程、VCE仅为诊断工具等。

---

## 485. RTS Smoother-Guided Learning of Physics-Based Neural Differential Models

**arXiv ID:** 2607.15180 | [PDF](https://arxiv.org/pdf/2607.15180v1)

**作者:** Ahmet Demirkaya `[一作]` (Northeastern University), Deniz Erdogmus `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Rauch–Tung–Striebel平滑器的混合物理-神经网络 ODE 学习框架，能够在部分观测和噪声条件下恢复隐藏状态并学习未知动力学；

**💡 创新点**

创新点在于通过两阶段交替优化，将RTS平滑得到的全状态轨迹作为伪目标，结合多步回放损失和协方差加权的状态损失，实现可解释、无在线估计器的高效学习；

**🔧 技术方法**

采用Rauch–Tung–Striebel平滑、经典卡尔曼滤波、神经网络拟合未知 ODE 部分、梯度反向传播、多步回放损失、协方差加权以及交替优化；

**📊 数据集**

使用五个基准动力学系统：谐振子、Hodgkin–Huxley 神经元、视网膜循环、Brusselator 反应模型、酵母糖酵解振荡器；

**📈 对比分析**

与 NeuralODE、GRU‑ODE‑Bayes、Recognition ODE、CKF/RBSE 混合 ODE 等基线对比，评估隐藏状态 RMSE 与 Hausdorff 距离；在大多数系统中，提出方法在隐藏状态误差和长时程回放误差上均优于基线，尤其在糖酵解和谐振子上表现突出；

**⚠️ 局限性**

局限在于需已知测量函数和高斯噪声模型，对极少量测量或非高斯噪声时可能失效；仅一次状态替换，扩展到多状态未知需解决可识别性；交替优化缺乏理论收敛保证；计算成本主要集中在训练阶段，对高维系统的扩展仍需改进。

---

## 486. MedFailBench: A Clinician-Built Open-Source Benchmark for Medical AI Safety Boundary Inspection

**arXiv ID:** 2607.15166 | [PDF](https://arxiv.org/pdf/2607.15166v1)

**作者:** Goktug Ozkan `[一作]` `[通讯]` (Kutahya Emet Dr. Fazil Dogan State Hospital), Goktug Ozkan (Kutahya Emet Dr. Fazil Dogan State Hospital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们创建了MedFailBench，一个由临床医师构建的合成基准，标注医学AI错误的严重程度和安全门类，并公开发布v0.2.1版本；

**💡 创新点**

创新之处在于引入了临床医师审核的严重程度层和安全门分类，为医学AI安全评估提供了可检查、可扩展的结构化标签体系；

**🔧 技术方法**

使用技术包括基于规则的评分脚本、每周自动化评估流水线、结构化JSON合成案例、LLM API调用和持续集成；

**📊 数据集**

使用的数据集是100条合成临床案例，覆盖10个医学专科，并在3个开源LLM上针对5个难题进行评估；

**📈 对比分析**

评估方法是对模型输出进行5维评分（安全、准确、来源透明、拒绝适当性、临床基础），结果显示所有模型在硬提示下均表现不安全，主导失效模式为漏报紧急升降；

**⚠️ 局限性**

局限性包括仅使用合成案例、单一临床医师审核、缺乏互评可靠性、未涉及真实患者数据、评估为预览而非纵向实验、自动评分未经临床验证等。

---

## 487. The Industrialization of Research ; On AI-Driven Science and Its Consequences

**arXiv ID:** 2607.15164 | [PDF](https://arxiv.org/pdf/2607.15164v1)

**作者:** Emmanuel Jeannot `[一作]` `[通讯]` (Inria), Emmanuel Jeannot (Inria)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统性地分析了人工智能驱动的科学研究工业化转变，并阐述了其对知识传承、理论可解释性、同行评审机制、范式创新、研究议程、科研结构与误差累积等七个核心挑战；

**💡 创新点**

创新点在于将工业革命的视角应用于科研产业化，构建了一个结构性问题框架，并通过Genesis项目等案例阐释了AI在科研中的深层影响；

**🔧 技术方法**

主要采用文献综述、案例研究和概念模型等理论分析技术；

**📊 数据集**

本文为概念性讨论，无使用具体实验数据集，主要引用公开文献与项目案例；

**📈 对比分析**

由于缺乏实验验证，未进行方法比较或性能评估，本文仅提供理论性评述；

**⚠️ 局限性**

局限在于缺乏实证验证，对不同学科的普适性和治理伦理细节的讨论不够充分。

---

## 488. On-Policy Delta Distillation

**arXiv ID:** 2607.15161 | [PDF](https://arxiv.org/pdf/2607.15161v1)

**作者:** Byeongho Heo `[一作]` (NAVER AI Lab), Dongyoon Han `[通讯]` (NAVER AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于教师与其基础模型差异（Delta信号）的主动式蒸馏方法OPD²，用于对大型语言模型进行推理能力的后训练提升。

**💡 创新点**

核心创新在于将Delta信号作为主要奖励函数，配合中心化（centering）与联合条件（joint conditioning）机制，突出推理相关的学习轨迹，从而显著提升传统OPD在多种推理任务中的性能。

**🔧 技术方法**

采用主动式蒸馏框架、KL散度与奖励形状化、教师与基础模型的双向前向推断、GRPOTrainer实现、vLLM并行推理，以及对Delta信号的中心化与一致性约束等技术。

**📊 数据集**

训练与评测使用开放式推理数据集：OpenMathReasoning、OpenScienceReasoning‑2、OpenCodeReasoning；评测覆盖14个基准，包括AIME24/25、AMC23、HMMT25、MATH500、OlympiadBench、ReasoningGym Math、CodeContests、CodeForces、LiveCodeBench、ReasoningGym Algorithm、GPQA、SuperGPQA、SciBench。

**📈 对比分析**

通过与原始模型、传统OPD和ExOPD在Qwen3（1.7B/4B/8B）及Gemma4（E4B）不同思考模式下的对比，OPD²在所有模型尺寸与领域均实现平均分数提升（如非思考模式1.7B提升约20分，思考模式8B提升≈3分），并保持训练过程的稳定性和持续优势。

**⚠️ 局限性**

局限性包括：需要额外的基础模型前向推断导致训练时间增加约24–28%（Gemma4约8%）；对极大模型或长训练周期的效果仍需验证；依赖教师与基础模型的差异，若两者差距不足则Delta信号效果有限；并未彻底解决暴露偏差（exposure bias）等后训练挑战。

---

## 489. Setup Complete, Now You Are Compromised: Weaponizing Setup Instructions Against AI Coding Agents

**arXiv ID:** 2607.15143 | [PDF](https://arxiv.org/pdf/2607.15143v1)

**作者:** Aadesh Bagmar `[一作]`, Pushkar Saraf `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并验证AI编码代理在安装项目依赖时的供应链攻击风险，提出并验证前置检查机制。

**💡 创新点**

发现安装缺口是模型与工具箱（harness）共同作用的结果，提出跨模型/工具箱的前置验证钩子，显著降低攻击成功率。

**🔧 技术方法**

采用对项目文档、依赖文件和安装命令的静态分析、包名编辑距离、来源可信度、OSV漏洞数据库查询等技术实现前置检查；同时使用多模型多工具箱的实验设置。

**📊 数据集**

使用12种攻击场景（12类攻击，11类覆盖）在Python、npm、Cargo等生态中，结合公开仓库代码搜索统计和真实漏洞列表构造的实验数据。

**📈 对比分析**

对比了9种模型-工具箱组合的检测率，发现前置检查钩子能在11/11场景中检测到10/11，显著提升安全性；无钩子时大部分源、版本攻击被忽略。

**⚠️ 局限性**

局限包括仅测试单一代表性攻击实例、仅覆盖Python及两小生态、未评估误报率、钩子无法处理所有非标准安装路径；未来需扩大样本、加入人类基准和更多生态系统。

---

## 490. Concept-Guided Spatial Regularization for World Models in Atari Pong

**arXiv ID:** 2607.15142 | [PDF](https://arxiv.org/pdf/2607.15142v1)

**作者:** Yukuan Lu `[一作]` (UC Davis), Yubei Chen `[通讯]` (UC Davis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在Atari Pong游戏环境中对五个视觉世界模型的冻结版本进行评估，发现它们在闭环回放中存在明显的视觉与动力学错误，并且在像素空间Zero‑shot MBRL中无法训练出与原始Dyna式代理相当的策略。

**💡 创新点**

提出概念引导空间正则化（Concept‑Guided Spatial Regularization, CGSReg），通过在关键概念区域（如球）上添加重建监督，针对任务关键概念的建模不足进行补偿。

**🔧 技术方法**

利用离线世界模型训练、外部PPO控制器进行闭环回放诊断、像素级Zero‑shot MBRL、以及对概念区域掩码的重建损失正则化等技术。

**📊 数据集**

以Atari Pong游戏环境及其从原始代理收集的100k步回放数据为实验数据集。

**📈 对比分析**

通过与原始Dyna式训练的最终策略进行对比，使用闭环回放诊断和统一的像素空间Zero‑shot MBRL协议，CGSReg在DreamerV3、DIAMOND、TWISTER上提升了回放质量和真实环境收益，但在Simulus和STORM上的效果不显著。

**⚠️ 局限性**

局限性包括仅在单一简单任务（Pong）验证、需人工指定概念掩码、未能解决所有模型瓶颈（如动作响应不稳定、长期一致性差等），且对更复杂环境与模型的泛化仍未评估。

---

## 491. Perfectly equidistributed Quasi-Monte Carlo sequences from Artin-Schreier polynomials

**arXiv ID:** 2607.15141 | [PDF](https://arxiv.org/pdf/2607.15141v1)

**作者:** Nicolas Bonneel `[一作]` (CNRS, Université Lyon 1, INSA Lyon, LIRIS), Victor Ostromoukhov `[通讯]` (CNRS, Université Lyon 1, INSA Lyon, LIRIS)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造了在有限域GF(b)上以相同次数、常数相差的不可约多项式（尤其是Artin-Schreier多项式）生成的Sobol'序列，并证明可通过特定初始化使其达到最高质量t=0；同时研究了将多维序列拼接时的t值保证与一致性；

**💡 创新点**

创新点在于将相同次数多项式常数相差的策略与Pascal矩阵幂的张量化结构关联，证明该构造可保证t=0，并利用Artin-Schreier理论在任意素数基b下提供最多b-1个不可约多项式，形成全新的低t序列构造方法；

**🔧 技术方法**

使用数字网与Sobol'递推、Pascal矩阵幂张量化、Artin-Schreier不可约多项式、Owen scrambling、fast t-value 计算及贪婪搜索等技术；

**📊 数据集**

在GF(b)（b=5,7,11）上采用对应的Artin-Schreier多项式及其初始化矩阵，生成多维点集；与公开的Sobol'序列（如Joe-Kuo、Faure-Lemieux、Bonneel、Ostromoukhov等）进行比较；

**📈 对比分析**

通过对生成点集进行Owen scrambling后平均32次，计算generalized ℓ² discrepancy，并与上述序列在同维度下进行对比；实验表明在(b-1)维和(2b-1)维下，Artin-Schreier序列的差异率竞争力强，甚至在某些维度和样本量下优于传统序列；

**⚠️ 局限性**

局限在于对大基b的初始化搜索仍耗时，现采用贪婪或随机方法近似；对高维（>2b-1）拼接的t值无法保证；理论证明仅适用于相同次数多项式的情况，实际应用需进一步验证。

---

## 492. TikStance: A Multimodal and Hierarchical Dataset for Multi-target Stance Analysis in TikTok Political Conversations

**arXiv ID:** 2607.15240 | [PDF](https://arxiv.org/pdf/2607.15240v1)

**作者:** Yazhi Zhang `[一作]` (Shenzhen Technology University), Bowen Zhang `[通讯]` (Shenzhen Technology University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了 TikStance 数据集，收集并标注了 161 条 TikTok 政治短视频及 13,876 条层级评论，支持视频与评论两级立场检测。

**💡 创新点**

创新点在于同时保留了主视频内容、完整的父子评论树以及视频层级和评论层级独立的 Favor/Against/None 立场标签，填补了多模态、层级对话与政治目标标签的缺口。

**🔧 技术方法**

采用手工关键词检索、视频多模态筛选、层级评论恢复、三人组独立标注并重标、Krippendorff α 评估等技术手段实现数据构建与质量控制。

**📊 数据集**

使用的数据集为 2023 年 9 月至 2025 年 1 月期间收集的 161 条主视频（特朗普 102 条、拜登 29 条、哈里斯 30 条）及其 13,876 条英文评论。

**📈 对比分析**

论文未开展模型对比实验，仅进行描述性统计与结构分析；因此未给出任何性能指标或方法比较。

**⚠️ 局限性**

局限性包括样本为非概率性查询集、目标覆盖不均、拜登/哈里斯样本量小、缺乏完整媒体文件与深层评论结构细粒度标签、未公开版本与访问许可限制。

---

## 493. CRISP: Constrained Refinement via Iterative Squeezing Process for Robust Medical Image Segmentation under Domain Shift

**arXiv ID:** 2607.15231 | [PDF](https://arxiv.org/pdf/2607.15231v1)

**作者:** Yizhou Fang `[一作]` (Southern University of Science and Technology), Longxi Zhou `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种名为 CRISP 的源域仅训练、目标域无数据、无推理时参数更新的医学图像分割框架。该框架通过“正区域秩稳定性”假设，利用潜在特征扰动生成高精度（HP）和高召回（HR）空间提示，并在递归自演化过程中不断收紧这两类提示，最终得到稳健的分割结果。

**💡 创新点**

创新点：
1) 将秩稳定性从分类任务推广到像素级分割；
2) 通过潜在特征层注入高斯噪声模拟域偏移，得到稳健的 HP/HR 两类提示；
3) 设计递归自演化训练框架与不确定性压缩损失（Uncertainty Squeezing Loss），实现单模型多轮自适应而不需目标域数据；
4) 在推理阶段仅使用冻结权重、无目标域信息即可完成多轮提示更新，保持模型轻量化。

**🔧 技术方法**

技术手段：
- 秩稳定性假设与潜在特征扰动；
- 对潜在特征注入 Gaussian 噪声并使用 Softmax → log‑odds → 量化等价排名；
- 生成 HP/HR 两类提示并递归细化；
- 递归自演化（Recursive Self‑Evolution）训练策略；
- 不确定性压缩损失（Uncertainty Squeezing Loss）用于加速收敛并提升边界精度；
- 模型无结构改动（仅扩展第一层通道），可兼容任何分割网络。

**📊 数据集**

使用的数据集：
1) M&Ms 4 组心脏 MRI（多中心）；
2) CT 基于肺血管的对比增强 vs 非对比 CT（模态偏移）；
3) CT 基于肺血管的正常 vs COVID-19 病例（人群偏移）。

**📈 对比分析**

比较方法与性能：
- 对比 FedDG、DDG-Med（域泛化）以及 IPLC、TEGDA（部署时自适应）等前沿方法；
- 在 M&Ms 任务中，CRISP 在 9 个类别–目标组合中 7 个达标最高 Dice，5 个 HD95 最佳，甚至超越使用目标域标注的 A+T 参考模型；
- 在模态与人群偏移任务中，CRISP 分别把 Dice 从 33.26 提升到 59.46（+26.20），从 65.15 提升到 70.61（+5.46），HD95 也显著下降（如 29.97→13.19 px）；
- 与目标域适配方法相比，CRISP 在无目标域数据的前提下获得更大改进，并避免了负适配风险。

**⚠️ 局限性**

局限性：
1) 依赖“秩稳定性”假设，若真实域偏移导致秩彻底打乱，性能可能下降；
2) 潜在特征扰动采用简单的 Gaussian 噪声，未覆盖所有真实域变化，需进一步探索更通用的扰动策略；
3) 推理阶段需要多次前向传播（≈3 次）才能收敛，计算成本高于单前向；
4) 目前仅在二维切片级实现，未完全解决三维体素级别的空间关联；
5) 对于非医学图像或需要强监督的多类别任务，方法的适用性尚待验证。

---

## 494. Disintegration Temporal Logic for Probabilistic Hyperproperties

**arXiv ID:** 2607.15223 | [PDF](https://arxiv.org/pdf/2607.15223v1)

**作者:** Mishel Carelli `[一作]` (CISPA Helmholtz Center for Information Security), Bernd Finkbeiner `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 Disintegration Temporal Logic (DTL)，一种新型的概率时序逻辑，能够表达广泛的概率超属性（如概率非干扰、完美不可区分性）并支持对执行序列（有限或无限）进行条件概率的推理。

**💡 创新点**

创新点在于：① 引入测度分解（measure disintegration）概念作为逻辑语义基础，解决了在零概率事件上条件化的问题；② 在逻辑中加入了对无限执行序列的条件化能力，天然支持相互作用的随机系统；③ 定义了两种可判定子逻辑（线性子逻辑与定性子逻辑），并给出多项式时间和多指数时间的模型检测算法。

**🔧 技术方法**

使用的技术包括：概率测度与σ-代数理论、测度分解与条件概率、马尔可夫链/马尔可夫决策过程的模型化、自动机理论（交错Büchi自动机、确定性Rabin自动机）、线性代数求解、底层BSCC（底部强连通分量）分析等。

**📊 数据集**

该工作没有基于公开数据集的实验评估；实验/验证主要通过理论证明与复杂度分析完成。

**📈 对比分析**

比较方法：与现有的概率时序逻辑（如PCTL、HyperPCTL、HyperCTL*）以及传统的安全属性逻辑进行语义对比；在可判定子逻辑上给出多项式/多指数时间的算法，证明其与已有方法的可行性与复杂度。性能方面：线性子逻辑的模型检测在多项式时间内完成；定性子逻辑的模型检测属于NPSPACE（多指数时间），对含有可判定超属性的系统具有理论可行性。

**⚠️ 局限性**

局限性包括：① 完整的 DTL 逻辑是不可判定的，必须限制到子逻辑；② 定性子逻辑的复杂度高，实际可用性受限；③ 仅支持精确标记（injective labeling）的系统，对一般标签化系统需要额外的条件；④ 对于非马尔可夫链或更复杂的动态系统，模型检测的适用性尚未探讨。

---

## 495. Plover: Steering GUI Agents through Plan-Centric Interaction

**arXiv ID:** 2607.15193 | [PDF](https://arxiv.org/pdf/2607.15193v1)

**作者:** Madhumitha Venkatesan `[一作]` (University of California, Davis), Dongyu Liu `[通讯]` (University of California, Davis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于计划中心化的 GUI 自动化框架，通过将任务计划可视化并持久化，支持用户在执行过程中实时监督、局部修改、自然语言指令与屏幕截图注释相结合，实现智能重新规划。

**💡 创新点**

创新点在于将任务计划从一次性输出转变为可持续、可编辑的共享对象；引入智能重新规划（Intelligent Replanning）将计划修订与执行同步；利用多模态交互（语言+屏幕注释）实现精确局部纠错，提升系统透明度与可控性。

**🔧 技术方法**

核心技术包括：基于视觉的多模态代理（使用 LLM 与视觉模型进行感知与规划）、Planner–Executor 分层架构、计划的版本控制与差异展示、可视化执行反馈、系统检测非进展并触发重新规划、以及屏幕截图的视觉哈希用于进度判定。

**📊 数据集**

主要使用 OSWorld‑Verified 基准（38 个多步任务）进行故障修复评估，并构造 5 个实际情景（浏览器与桌面应用）进行稳定性分析；数据集覆盖网页、LibreOffice、Calc 等多种 GUI 环境。

**📈 对比分析**

与完全自主执行的基线对比，系统在 26 个失败任务中有 23 个得到改善（17 个完全成功，6 个部分成功），平均每个任务需 2.04 次局部干预；视觉相似度评估显示浏览器任务 SSIM>0.98，桌面任务 0.61–0.69，表明在视觉稳定性较高的场景中恢复更容易；总体可恢复率达 88%。

**⚠️ 局限性**

局限性包括：对短小、低风险任务的额外交互成本可能不值得；依赖用户对任务结构有足够了解，无法在任务本身不确定时动态共建计划；对多重错误累积导致的全局漂移仍难以一次性修复；未来需加强不确定性提示与错误可视化以提升用户信任。

---

## 496. Memory-Exhaustion Attack on the Blocklace Byzantine-Repelling Conflict-Free Replicated Data Type

**arXiv ID:** 2607.15185 | [PDF](https://arxiv.org/pdf/2607.15185v1)

**作者:** Erick Lavoie `[一作]` `[通讯]` (University of Basel), Erick Lavoie (University of Basel)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 Blocklace（一种基于有向无环图的冲突自由复制数据类型）提出了一种内存耗尽攻击，并证明在原始设计下，该攻击可使正确节点被大量伪造更新淹没。

**💡 创新点**

创新点在于：①首次揭示 Blocklace 的拜占庭容错规则允许攻击者通过不断生成新身份并提交自我指认的无效更新来无限扩张被复制更新集合；②证明该扩张上限可以无限接近身份空间大小；③提出通过限制可接受身份集合（类似兴趣驱动复制）来缓解此攻击。

**🔧 技术方法**

采用了形式化定义（块、因果历史、拜占庭节点集合）、递归构造的拜占庭排斥属性、对攻击流程的逻辑推导以及与 Sybil 攻击的类比分析等技术手段。

**📊 数据集**

未使用任何真实数据集；研究主要基于理论推导和概念示例。

**📈 对比分析**

与原始 Blocklace 设计进行理论对比，证明在未限制身份空间时攻击能无限放大，进而导致正确节点无法接受合法更新；但本文未提供实验评估或性能量化数据。

**⚠️ 局限性**

局限性包括：未给出具体实现或实证验证的缓解方案；仅在理论层面讨论攻击与防御，缺乏对实际部署环境中存储与网络开销的定量分析；对身份空间大小与攻击效率的关系仍需进一步实验探索。

---

## 497. AHEAD: Anticipatory Hand-Driven Teleoperation via Human Intent Prediction

**arXiv ID:** 2607.15172 | [PDF](https://arxiv.org/pdf/2607.15172v1)

**作者:** Seok Joon Kim `[一作]` (Georgia Institute of Technology), Mohsen Moghaddam `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出AHEAD，一种实时VR预期式遥操作系统，利用手部、头部3D运动和场景上下文预测抓取和放置目标，从而提前触发机器人运动。

**💡 创新点**

将意图预测与距离门控策略相结合，实现提前行动的目标准确预测，并在手部自然动作下保持稳定。

**🔧 技术方法**

注意力网络（跨注意力融合）、GAT编码手部图、场景实体编码、距离门控的预览-提交机器人控制器以及Meta Quest 3的3D手部/头部传感器。

**📊 数据集**

在Meta Quest 3收集的VR抓取-放置演示数据（200轨迹/15人，共1000序列），每条序列包含手关节、头部姿态、物体/槽位描述与事件标注。

**📈 对比分析**

与最近邻、统一模型等基线比较，Top1/Top3准确率约76%，预览-提交策略将机器人反应时间缩短0.6 s(物体)和1.4 s(槽位)，并在用户研究中显著降低NASA‑TLX负荷。

**⚠️ 局限性**

依赖ArUco标记的物体姿态估计、数据量有限、未预测抓取姿态/方向，限制了在无标记或多姿态抓取场景的应用。

---

## 498. Navigating the Socio-Technical Complexity Challenge in Quantum Software Ecosystems

**arXiv ID:** 2607.15135 | [PDF](https://arxiv.org/pdf/2607.15135v1)

**作者:** Ronja Heikkinen `[一作]` (University of Jyväskylä), Vlad Stirbu `[通讯]` (University of Jyväskylä)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种社会技术视角的量化框架，用于评估量子计算环境的选择，并通过引入重力井概念和社会技术期望值来识别与解决生态系统中的摩擦点。

**💡 创新点**

创新点在于将重力井与社会技术期望值结合为一种设计科学方法，提供系统化的手段识别生态系统中的关键节点并指导可演化环境的决策。

**🔧 技术方法**

所用技术包括设计科学研究方法、社会技术理论、技术生态图的构建与分析、以及针对软件堆栈层级的评估准则与重力井属性。

**📊 数据集**

使用的数据集为三种典型场景（Qubernetes、混合电路切割工作流、传统HPC）作为案例研究进行阐释和验证。

**📈 对比分析**

评估方法是对识别出的重力井在四项社会技术期望值（运维透明度、知识迁移、避免不可逆约束、支持实验演化）上的定性打分；并通过对比分析API Server与Slurm在不同维度的表现来展示框架效果，未进行量化性能测试。

**⚠️ 局限性**

局限性包括缺乏系统化经验验证、评估过程高度主观且依赖分析者专业知识、案例覆盖有限、评价以定性为主且未提供量化打分机制。

---

## 499. teLLMe Why (Ain't Nothing but a Jam): Exploratory Causal Analysis of Urban Driving Data

**arXiv ID:** 2607.15254 | [PDF](https://arxiv.org/pdf/2607.15254v1)

**作者:** Qiwei Li `[一作]` (Rutgers University), Jorge Ortiz `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了teLLMe系统，利用从车辆摄像头视频中提取的事件表，对城市驾驶数据进行探索性因果分析，并生成可解释的因果卡；

**💡 创新点**

创新点在于将事件表构建、PC算法因果结构学习、bootstrap稳定性评估、schema‑aware LLM自然语言解析、后门调整集自动选择以及因果卡可解释性输出整合到一个查询驱动的工作流中，使非技术专家能够在不需要读懂因果图的前提下进行因果假设检验；

**🔧 技术方法**

采用PC算法进行因果结构学习并通过bootstrap评估边的稳定性；使用线性回归和DoWhy框架进行平均处理效应（ATE）估计；使用schema‑aware LLM将自然语言问题映射为结构化因果查询；最终通过Causal Card展示估计结果、后门集、图证据和假设；

**📊 数据集**

基于BDD100K dashcam注释的事件数据集，事件表包含天气、场景类型、交通密度、高峰时段等变量；

**📈 对比分析**

通过与固定后门调整集和未平衡数据下的分析比较，展示基于图的后门选择可以得到更稳健、区间更宽的ATE估计；对比实验表明忽略图结构会导致效应偏大且置信区间收窄；性能表现以ATE估计及其置信区间为核心指标，未给出具体数值但指出差异明显；

**⚠️ 局限性**

仅基于观察性事件数据，无法处理时间与空间依赖；未观测到的混杂因子（如驾驶员意图、路面状况、天气强度）未纳入模型；假设独立事件、线性回归模型的稳健性受限；因果卡尚未经过用户评估，实际可用性尚待验证。

---

## 500. Beyond the Leaderboard: Design Lessons for Trustworthy Multimodal VQA

**arXiv ID:** 2607.15241 | [PDF](https://arxiv.org/pdf/2607.15241v1)

**作者:** Sushant Gautam `[一作]`, Steven A. Hicks `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对MediaEval Medico 2025胃肠镜视觉问答与可解释性任务进行跨团队系统分析，评估参数高效微调、多模态融合、结构化推理与视觉证据定位对性能与可信度的影响。

**💡 创新点**

创新点在于提供实证的跨团队设计轴对比与评价指标体系，强调可解释性与鲁棒性（如自我探测、联合多任务学习、LLM裁决），并提出可信多模态AI的评估与报告准则。

**🔧 技术方法**

采用参数高效微调（LoRA/QLoRA）、多模态基础模型（Florence‑2、PaliGemma、BLIP‑2 等）、自我探测与多任务学习、可视化注意力/Grad‑CAM、基于 LLM 的评分与语义裁决等技术。

**📊 数据集**

使用 Kvasir‑VQA‑x1（6500 张图像、159,549 QA 对）及其私有验证集（500 张图像、5,368 QA），并对 ImageCLEFmed MEDVQA 2025 私有集进行评估。

**📈 对比分析**

通过公共与最终排行榜、BLEU/ROUGE/METEOR 等文本指标，以及 LLM 裁决的五维解释分数进行比较；顶尖团队在私有集 BLEU 约 0.47，解释准确率、可信度等维度最高分别为 0.86/0.74/0.76/0.90/0.61；整体表现显示顶尖团队聚集、差距窄小，但不同分割下排名不稳定。

**⚠️ 局限性**

局限性包括评估依赖词汇重叠与 LLM 裁决的模型偏差，缺乏严格的视觉证据验证与完整的图像级重叠审计，结论主要为比较性而非因果性，并未探讨大模型全参数微调的潜在优势。

---

## 501. Campaign Diagrams: Visualizing the March Through the Phases of a Workload

**arXiv ID:** 2607.15225 | [PDF](https://arxiv.org/pdf/2607.15225v1)

**作者:** Toluwanimi O. Odemuyiwa `[一作]` (University of California, Davis), Joel S. Emer `[通讯]` (NVIDIA)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了“campaign diagrams”可视化技术，用于在时间轴上展示多阶段工作负载的计算吞吐、内存带宽利用率、计算和内存流量体量，并结合案例研究展示其在识别优化机会（如融合、流水线、扩展/抑制）方面的效用。

**💡 创新点**

创新点在于将时间、资源利用率、体量三维信息融入单一图表，克服了传统 roofline 的聚合盲区和 profilers 的细粒度缺失；能够在设计时与分析模型、仿真或真实测量数据无缝配合，提供阶段级别的性能洞察。

**🔧 技术方法**

技术手段包括：
1) 基于可调节时间轴的绘图框架（实现基于 Altair）；
2) 通过计算/内存利用率与体量的几何映射（如管道填充、面积计算）构造图形；
3) 支持多阶段并行、溢出检测、扩展与抑制（dilation/throttling）等调度策略的可视化。

**📊 数据集**

数据集主要有两类：
- 低秩 GEMM（N=20480, R=Q=512）用以比较稠密与低秩乘法；
- Mamba-1 状态空间模型（24 阶段 Einsum 表达式）用于评估融合、流水线和扩展对大型语言模型推理的影响。

**📈 对比分析**

比较方法：
- 与传统 roofline 进行对比，展示相同 OI 下的性能误判；
- 通过可视化识别的优化点（融合、流水线）在实验中实现了 2.4×、4.4× 等加速；
- 通过分析与仿真对比，验证 campaign diagrams 在接近理论上限的同时揭示实际瓶颈。

**⚠️ 局限性**

局限性：
- 随着阶段数增多，图表视觉复杂度提升，易造成信息过载；
- 当前仅支持两种限速资源，扩展到更多资源需进一步研究；
- 对于极大规模模型的静态呈现仍难以维护清晰性，需引入交互式缩放等辅助手段。

---

## 502. Symbal: Detecting Systematic Misalignments in Model-Generated Captions

**arXiv ID:** 2607.15216 | [PDF](https://arxiv.org/pdf/2607.15216v1)

**作者:** Maya Varma `[一作]` (Stanford University), Curtis Langlotz `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于检测多模态大型语言模型（MLLM）生成图像字幕中的系统性错配（系统性误配）的方法

**💡 创新点**

创新点在于：①将检测任务拆分为两阶段（先找文本错误再找对应视觉特征），并在每个阶段使用分组、评分、总结三步；②提出专门的基准  (Symbal-Benchmark) 用于自动化注入并评估系统性误配；③在无参考或有参考两种设置下均能工作，并提供自然语言输出。

**🔧 技术方法**

使用的技术包括：文本/图像嵌入模型（如Qwen、OpenCLIP、XRayCLIP、MedSigLIP），视觉-语言评分器（Qwen‑72B 等），文本或视觉-语言总结器（LLM如Qwen‑72B、MedGemma‑27B），以及基于聚类的分组方法（spherical K‑Means）。

**📊 数据集**

数据集：基准包含 420 个评估设置，分别来自 COCO（自然图像）和 MIMIC‑CXR（医学 X‑ray）两大数据集；同时在真实场景中使用了四个公开 MLLM 生成的 COCO 子集和 ShareGPT4V 数据集。

**📈 对比分析**

与单阶段直接提示的 LLM（Llama3.3‑70B、Qwen2.5‑VL‑72B、GPT‑OSS‑120B）比较，Symbal 在无参考设置下能正确识别 63.8% 的数据集（相当于 4 倍提升），在有参考设置下 45.8%；在不同关联强度和视觉特征大小的分层评估中亦表现出更高的稳健性。

**⚠️ 局限性**

局限性包括：①对大型数据集仍有计算开销，尤其是多次嵌入与聚类；②在医学域下的图像特征检测准确率相对较低（仅 36–53%）；③评估依赖于人工设计的错误与视觉特征的映射，可能无法覆盖所有真实误配模式。

---

## 503. Stigmergic Graph Memory: An Environment-Aware Approach for Many-to-Many Multi-Agent Pickup and Delivery

**arXiv ID:** 2607.15182 | [PDF](https://arxiv.org/pdf/2607.15182v1)

**作者:** Aditya Dutta `[一作]` (Anonymous Affiliation), Joon-Seok Kim `[通讯]` (Anonymous Affiliation)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `79276348-11e0-48e3-84bc-7ec231d0171c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提供了一份完整的技术补充记录，涵盖控制器定义、内存配置、可行性保持、完整基准结果、消融实验、鲁棒性验证、迁移证据以及空间诊断。

**💡 创新点**

创新点在于系统化地记录并分析了控制系统在不同配置和环境下的性能与可迁移性，为后续研究提供了可复现的实验框架。

**🔧 技术方法**

使用了控制器设计、内存管理、基准评估、消融实验、鲁棒性测试、迁移学习与空间诊断等多种技术方法。

**📊 数据集**

文中未具体说明使用的数据集或实验环境；推测可能涉及自定义仿真或真实系统的数据。

**📈 对比分析**

对比方法和性能指标未在该摘要中给出，无法评估其具体表现；若有完整记录可供后续对比分析。

**⚠️ 局限性**

主要限制是缺乏对公开数据集或真实场景的直接验证，且本文仅提供实验记录，未深入讨论算法的理论推导或泛化能力。

---

## 504. Grokipedia vs Wikipedia: An LLM-Based Audit of Political Neutrality along Ideologies

**arXiv ID:** 2607.15146 | [PDF](https://arxiv.org/pdf/2607.15146v1)

**作者:** Filippos Vlahos `[一作]` (Ghent University), Tijl De Bie `[通讯]` (Ghent University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对比 Grokipedia 与 Wikipedia 在政治中立性上的偏差，利用 4 个多样化 LLM 评审对 1,394 名政府成员的条目对进行大规模中立性打分，基于 V-Party 专家编码的 9 维意识形态特征进行回归分析

**💡 创新点**

首次在同一语料下同时对 LLM 生成的百科（Grokipedia）和人类撰写的百科（Wikipedia）进行中立性评估，并将 LLM 评审本身的偏差纳入分析，揭示 LLM 评审对结果的影响；还发现即使是生成方的 LLM 对自身产出也评为不够中立，说明生成过程本身带有意识形态结构

**🔧 技术方法**

使用 GPT‑style 评审技术：四个不同偏向的 LLM（Grok‑4、Claude‑Opus‑4.6、Mistral‑Medium‑3.5、DeepSeek‑V4‑Pro）在自定义提示下进行 5 分 Likert 评分；利用 OLS 回归估计意识形态维度与中立性评分的关系；使用 Cohen κ、Krippendorff α 等统计量评估评审一致性

**📊 数据集**

V‑Party 专家编码的 9 维意识形态数据（文化包容、移民开放、LGBT 平等、世俗化、女性劳动平等、反亲属主义、政治多元化、反民粹主义、经济右派）与 WhoGov 选举数据库提供的 1,394 名政府成员；从 Wikipedia 与 Grokipedia 0.2 版分别抓取对应条目

**📈 对比分析**

对两方条目分别进行评审后，计算中立性差异（Δ Y）并回归意识形态维度，发现 Grokipedia 在经济右派和政治多元化上偏好程度更高，Wikipedia 在 LGBT 与女性劳动权利上偏好更强；评审方差异表明评审 LLM 的严格程度不同，Claude 最为严格、DeepSeek 最宽松；整体评估显示两方均存在偏差，Grokipedia 的偏差更显著

**⚠️ 局限性**

局限性包括：中立性定义仅基于 NPOV 的 8 条标准，可能无法覆盖所有维度；LLM 评审易受自身偏见和多样性偏差影响，且未与人工标注者对照；样本仅为 1,394 名政府成员，未涵盖其他政治角色；文章仅在单一快照时间点评估，未考虑内容随时间变化；回归模型仅解释少量方差，未完全捕捉复杂文本特征

---

## 505. Ray-based phase error correction for miniaturized DOE projector-based FPP under single-directional hyperbolic projection

**arXiv ID:** 2607.15139 | [PDF](https://arxiv.org/pdf/2607.15139v1)

**作者:** Seung-Jae Son `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于投影光线的相位误差校正框架，用于小型DOE投影仪的边缘投影轮廓测量。

**💡 创新点**

创新点在于：①通过单向双曲线条纹直接估计投影仪的光心，避免了双向校准或立体校准；②将相位误差建模为沿投影光线的径向函数，形成物理一致的修正模型；③提出单姿态、可迭代的数据高效修正方法。

**🔧 技术方法**

使用的技术包括：相位提取（相移算法）、像素级映射、光线几何建模、球面与平面拟合、线性回归建模、球面与平面误差统计、光线角度投影、插值与迭代优化。

**📊 数据集**

实验数据集为：平面板、20 mm半径球、印刷电路板（PCB）以及复杂物体（凸起表面和手电筒组），所有数据均使用同一台小型DOE投影仪和FLIR摄像机采集。

**📈 对比分析**

与传统的直方图均衡与Hilbert变换相位误差校正对比，RMSE在平面上从0.118 mm降至0.0338 mm（两次迭代），在球面上从0.069 mm降至0.0547 mm，PCB上从0.133 mm降至0.0544 mm；单姿态模型虽略逊于全姿态模型，但仍显著优于传统方法。

**⚠️ 局限性**

局限性包括：①在最优性能下仍需多姿态采集，单姿态模型精度有限；②假设投影仪为理想针孔，非线性畸变若过大仍难完全补偿；③对光照强度变化和反射纹理仍有一定敏感性。

---

## 506. Platform Choice, Trust, and Privacy in the Consumer AI Assistant Market

**arXiv ID:** 2607.15134 | [PDF](https://arxiv.org/pdf/2607.15134v1)

**作者:** Jennifer Zou `[一作]` `[通讯]` (Profound), Jennifer Zou (Profound)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对1999名美国成年AI助手用户的调查与选择实验，研究了平台选择、任务分配、信任度和隐私特性价值。

**💡 创新点**

①首次以任务级别而非用户级别描绘AI助手的市场结构与竞争；②发现“经验信任差距”，即使用者对平台的信任远高于非使用者；③揭示隐私行为主要受信息素养而非关注驱动。

**🔧 技术方法**

使用问卷调查、加权抽样、分层置信区间估计、线性化方差推断、条件Logit模型进行离散选择实验分析。

**📊 数据集**

基于Prolific在线面板收集的1999份问卷；采用PUMS、Pew Research、Real-Time Population Survey等外部数据进行权重构建。

**📈 对比分析**

通过比较平台的知晓度、使用率、主导份额、任务签名以及信任排序，展示ChatGPT和Gemini占主导地位，而Claude、Copilot在特定任务（编码、工作）中具有显著竞争力；隐私特性的价值估计显示用户最愿意为“无人工审核”支付最高。

**⚠️ 局限性**

局限性包括样本仅覆盖已知AI用户、行业样本有限、选择实验为声明偏好而非真实交易、部分平台用户基础过小导致统计不稳、并且跨平台行为与隐私特性间的因果关系未能在横截面设计中解耦。

---

## 507. Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search

**arXiv ID:** 2607.15253 | [PDF](https://arxiv.org/pdf/2607.15253v1)

**作者:** Debayan Mukhopadhyay `[一作]` (University of Calcutta), Shubham Chatterjee `[通讯]` (Missouri University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多步检索式对话代理中，使用反事实实验测定文档对最终答案的因果贡献，发现大约三分之一文档对代理有因果作用，却在静态评估中被视为无用，称之为“桥接文档”。

**💡 创新点**

创新点在于：①提出Counterfactual Trajectory Utility (CTU) 的量化指标；②用CTU与静态RAG Utility (SRU) 对比，揭示两者统计独立；③通过实体可观相关性(OER)证明桥接文档的机制，即它们携带判别性实体并在下一步查询中出现。

**🔧 技术方法**

技术方法包括：ReAct式检索代理（Qwen2.5-7B-Instruct + BM25 + 交叉编码器重排序）、文档删除干预、CTU的三维归一化合成、OER实体相关性计算、实体传播率评估。

**📊 数据集**

数据集为HotpotQA开发集，挑选1000个多步问题，记录23,322条文档观测，包含约227k个实体观测。

**📈 对比分析**

比较方法：在每个文档上执行单一删除干预，计算CTU；用静态读者或检索得分作为SRU轴；统计四象限分布。性能方面，桥接文档占比为35.7%（SRU轴）或27.2%（检索代理轴），且两轴相关系数近0，说明静态指标无法预测因果价值。

**⚠️ 局限性**

局限性包括：①仅针对单一代理与检索堆栈（Qwen2.5-7B-Instruct + BM25/交叉编码器）；②对抗性文本与模型非确定性引入噪声；③OER估计依赖HotpotQA支持事实，候选集小导致噪声；④代理读者评估弱，导致SRU轴极度偏斜；⑤CTU需回放成本高，难以用于训练。

---

## 508. AutoSynthesis: An agentic system for automated meta-analysis

**arXiv ID:** 2607.15247 | [PDF](https://arxiv.org/pdf/2607.15247v1)

**作者:** Moein Taherinezhad `[一作]` (Politecnico di Milano), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了一个多代理系统，能够从检索到统计合成完成整个元分析全过程。

**💡 创新点**

将LLM与确定性统计流程相结合，实现端到端自动元分析，支持异质性、偏倚评估并自动生成PRISMA报告。

**🔧 技术方法**

采用LangGraph多代理架构，使用OpenRouter API驱动LLM进行文本理解与信息提取，Python实现REML随机效应模型、偏倚检测与可视化。

**📊 数据集**

以Hölbling等2025年关于LLM说服效果的12个效应值及其原始文献为基准进行验证，并检索了28篇候选研究。

**📈 对比分析**

通过与人工元分析对比，检索召回率从71.4%校正至85.7%，最终汇总效应偏差≤0.12 Hedges' g，计算成本约$1.5，PRISMA流图与原文高度一致。

**⚠️ 局限性**

受限于检索范围、全文获取、统计信息缺失以及潜在模型训练泄露，评估仅基于单一基准，复杂领域表现尚未充分验证。

---

## 509. ARMOR++: Agentic Orchestration of a Multi-Domain Primitive Set for Transferable Attacks on Deepfake Detectors

**arXiv ID:** 2607.15246 | [PDF](https://arxiv.org/pdf/2607.15246v1)

**作者:** Christos Korgialas `[一作]` (Aristotle University of Thessaloniki), Konstantinos N. Plataniotis `[通讯]` (University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ARMOR++——一种基于视觉语言模型与大型语言模型的多智能体框架，用于在严格无查询（no‑query）黑盒条件下对深度伪造检测器进行可迁移攻击。

**💡 创新点**

创新点在于：①引入五种跨域扰动原语（密集优化、稀疏敏感性、几何变形、频域模拟、块结构随机化）；②通过VLM提取空间语义先验，LLM进行原语选择、超参数自适应与熵正则混合；③构建闭环自适应控制，消除对目标模型查询；④在AADD‑2025等大规模数据上系统评估并展示显著提升。

**🔧 技术方法**

使用技术包括：Vision‑Language Model（Qwen2.5‑VL）进行语义分析；Large Language Model（Qwen3）进行规划与重参数化；多原语并行生成与混合；熵正则化混合优化；基于CNN与Transformer的三元对抗集成；严格的无查询目标评估。

**📊 数据集**

使用数据集：AADD‑2025（低质量LQ与高质量HQ两子集）以及200张DFDC‑Preview零样本子集进行跨域评估。

**📈 对比分析**

与四大类基线（传统传输攻击、查询基攻击、集成优化、RL/ARMOR）对比，ARMOR++在LQ和HQ上对ViT‑B/16和Swin‑B的攻击成功率（ASR）分别达到0.443/0.408和0.321/0.287，较基线提升约20‑30%；在零样本DFDC上保持优于ARMOR；在两种非自适应防御下仍保留显著的攻击成功率。

**⚠️ 局限性**

局限性包括：评估仅针对非自适应防御，未考察针对ARMOR++的自适应或期望‑变换防御；使用的三种CNN均从ImageNet预训练，可能存在共享表示导致的过拟合；组件消融为匹配预算的简化分析，未覆盖所有静态混合策略；仅在两种Transformer目标上测试，未验证在更广泛架构上的普适性。

---

## 510. Adaptive Sampling for Spatiotemporal Anomaly Monitoring in Wireless Sensor Networks

**arXiv ID:** 2607.15235 | [PDF](https://arxiv.org/pdf/2607.15235v1)

**作者:** Guoqing Lu `[一作]` (South East Technological University), Bernard Butler `[通讯]` (South East Technological University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于哨兵的自适应采样框架，在无线传感器网络中实现稀疏采样与异常窗口密集观测的平衡。

**💡 创新点**

创新点在于将Kalman滤波预测不确定性驱动的稀疏采样、哨兵节点的两侧GLR检测与节点相对阈值、一次跳唤醒以及空间优化的哨兵部署融合成协同控制流水线。

**🔧 技术方法**

采用Kalman滤波、GLR检测、节点相对阈值、一次跳唤醒、空间覆盖优化及多种能量-准确性评估指标。

**📊 数据集**

使用Intel Berkeley Research Lab温度数据集并注入10个随机的时空异常。

**📈 对比分析**

与全采样、纯KF稀疏、Oracle本地唤醒、AAS和Adapted e-Sampling进行比较，结果显示在降低总体能耗的同时，异常窗口采样比率(AWSR)提升至0.933，覆盖率和能耗均优于基线。

**⚠️ 局限性**

局限在于实验仅在单一温度数据集和人工注入的突发异常上验证，未考虑传感器故障、环境漂移、通信丢包等真实场景。

---

## 511. In-Place Tokenizer Expansion for Pre-trained LLMs

**arXiv ID:** 2607.15232 | [PDF](https://arxiv.org/pdf/2607.15232v1)

**作者:** Jimmy T. H. Smith `[一作]`, Mathias Lechner `[通讯]` (Liquid AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对已有 LFM2 8B MoE 模型，提出了 tokenizer 扩展方法，将 65K 字节级 BPE 词表扩展到 128K，保持已学习词的 embedding 并为新词计算均值初始化，随后进行两阶段微调（embedding-only + 全模型继续预训练）以恢复并提升模型质量。

**💡 创新点**

创新点在于：① 通过在保持原始 merge 表的前提下继续 BPE 训练，确保新词可直接拆分为已有子词；② 采用均值初始化与两阶段训练，成功在保持原有质量的前提下扩展 vocab，避免了全词表微调导致的生成崩溃。

**🔧 技术方法**

技术包括：继续 BPE 训练、embedding 复制与均值初始化、两阶段微调（embedding-only + 全模型继续预训练）、后续的 mid‑training 与 post‑training；以及在设备端的 decode‑throughput 评估。

**📊 数据集**

使用了多语种平衡语料库（FineWeb、英文代码、JSON 等），覆盖低资源语言如印地语、越南语、泰语等，并在 12T+35T tokens 上进行持续预训练。

**📈 对比分析**

通过与源模型、零射击、Stage 1、Stage 2 四个阶段对比，展示了在多项基准（MMLU、MMMLU、MATH、HumanEval、MGSM 等）上维持或提升性能，同时在低资源语言上 token 数量减少 2–4 倍、设备端字符级解码速度提升 2.2–3.7×，但在已高效编码语言上出现 1–9% 的速度回退。

**⚠️ 局限性**

局限性包括：① 仅适用于可继续 BPE merge 的 tokenizer；② 需要额外的连续预训练计算；③ 在已高效语言上可能出现解码慢的问题；④ 仅在 LFM2 体系验证，跨模型迁移需进一步验证。

---

## 512. Data Driven Block Replacement Scheduling

**arXiv ID:** 2607.15229 | [PDF](https://arxiv.org/pdf/2607.15229v1)

**作者:** Aniruddhan Ganesaraman `[一作]` (University of North Carolina at Chapel Hill), VIdyadhar Kulkarni `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究如何在已知机器寿命分布未知的情况下，使用在线学习方法（多臂赌博机和生存分析）动态确定块式更换间隔 k，从而最小化长期平均维护成本。

**💡 创新点**

① 将块式更换问题建模为带有嵌套观测结构的多臂赌博机，推出相关臂下的 O((K‑k*)log T) 调度优度；② 证明 Hoeffding‑、Bernstein‑、相关臂 LCB 算法可实现 Lai‑Robbins 下界；③ 采用 Kaplan‑Meier 生存估计实现无参数寿命分布学习，并证明几乎无增量后悔；④ 通过两种 MDP（时间‑延迟与年龄向量）提供理论基准并揭示结构性成本差距。

**🔧 技术方法**

多臂赌博机（LCB、Bernstein、相关臂）、平均成本 MDP、寿命估计的 Kaplan‑Meier 方法、renewal 理论、增量后悔分析、复杂度与可扩展性评估。

**📊 数据集**

在仿真中使用两组离散寿命分布：1+Binomial(10,0.5)（支持1–11，均值6）和 1+Poisson(4)（支持1–∞，均值5），每组取 N=2 机器、K=12、c_b=1.0、c_f=2.6。

**📈 对比分析**

与年龄向量 MDP 的最优政策进行比较；所有算法均在 T=10 000 轮内找到最优间隔 k*；相关臂 Bernstein LCB 取得最小累计后悔；Kaplan‑Meier 进一步压制后悔，近似零增量后悔；相对于年龄向量最优，块式策略存在约 33%–50% 的结构性成本缺口。

**⚠️ 局限性**

仅适用于 IID、无时间趋势、无负载依赖的寿命；假设所有机器相同；不考虑环境协变量、负载或退化模型；未给出贝叶斯或情境扩展；仅在仿真环境中验证，实际部署需要进一步实验。

---

## 513. Structural-Semantic Reciprocal Learning for Unsupervised Visible-Infrared Person Re-Identification

**arXiv ID:** 2607.15220 | [PDF](https://arxiv.org/pdf/2607.15220v1)

**作者:** Moyao Tian `[一作]` (Wuhan University of Science and Technology), Xiao Wang `[通讯]` (Wuhan University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无监督可见-红外人像重识别框架SSR‑L，结合结构-语义递归学习，实现跨模态特征的自校正。

**💡 创新点**

创新点在于将精细结构解耦（Fine-grained Structural Decoupling）与闭环语义校准（Closed-loop Semantic Calibration）相互作用，形成自纠正的闭环闭合，显著降低伪标签噪声并提升跨模态对齐。

**🔧 技术方法**

采用了水平自适应池化、分离的瓶颈头、非参数记忆对比学习、k‑reciprocal Jaccard距离+DBSCAN聚类、语义原型重构与归一化，以及多阶段训练策略。

**📊 数据集**

在SYSU-MM01和RegDB两个公开数据集上进行评估，使用CMC Rank‑1/10和mAP指标。

**📈 对比分析**

与现有无监督方法相比，SSRL在SYSU-MM01 All Search上Rank‑1 59.47%、mAP 55.35%；在RegDB可见→红外协议上Rank‑1 92.47%、红外→可见协议上Rank‑1 91.06%，均优于最新无监督方法，并逼近甚至超越部分监督方法。

**⚠️ 局限性**

局限性包括对超参数（如分区数、权重比例、学习率）敏感，且在极端光照或姿态变化下，结构解耦可能出现误分区，导致对齐误差；同时模型在大规模实时部署时仍需进一步优化计算效率。

---

## 514. BadWAM: When World-Action Models Dream Right but Act Wrong

**arXiv ID:** 2607.15207 | [PDF](https://arxiv.org/pdf/2607.15207v1)

**作者:** Qi Li `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 BadWAM 框架，在世界-动作模型（WAM）上执行黑盒视觉扰动攻击，诱导机器人执行失败。

**💡 创新点**

创新点在于发现 WAM 的行动与未来想象可被攻击脱同步，设计两种攻击（仅行动攻击与保持未来一致的攻击）并验证未来预测并非安全保障。

**🔧 技术方法**

采用零阶优化（随机方向梯度估计）与 L∞ 视觉扰动，同时约束行动距离与未来距离。

**📊 数据集**

实验基于 LIBERO 与 RoboTwin 两套语言条件机器人操纵基准。

**📈 对比分析**

与随机噪声、白盒梯度攻击及多种 WAM 变体对比，攻击可将任务成功率从 96% 降至 43% 以上，表现出显著的破坏效果。

**⚠️ 局限性**

局限在于攻击对预处理或一致性检测不具自适应能力，未验证在真实物理环境或实时约束下的表现。

---

## 515. Self-Evolving Human-Centered Framework for Explainable Depression Symptom Annotation

**arXiv ID:** 2607.15202 | [PDF](https://arxiv.org/pdf/2607.15202v1)

**作者:** Hoang-Loc Cao `[一作]` (University of Science, VNU-HCM), Hung Cao `[通讯]` (University of New Brunswick)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个自演进的人机协作DSM‑5‑TR抑郁症注释框架，能够自动筛选证据、按症状级别进行推理并生成可解释、可追溯的标签。

**💡 创新点**

创新点包括：① 双重记忆机制（示例记忆与反思记忆）实现无梯度更新的自演进；② 将LLM辅助标注与专家审核闭环结合，形成交互式工作流；③ 生成完整的证据链、推理轨迹与编辑历史，实现全面可审计。

**🔧 技术方法**

使用技术：大语言模型（Gemini‑3.5‑Flash‑Lite、GPT‑4o‑mini、GPT‑5.4‑mini）+检索增强记忆 + 结构化DSM‑5‑TR推理层 + 人机交互标注界面。

**📊 数据集**

使用数据集：ReDSM5 抑郁症基准数据集（包含10个复杂病例）以及五位专家的黄金标注。

**📈 对比分析**

通过句子级、症状级、证据对齐、病例诊断四层评估，模型在句子级F1>90%、症状级F1≈82%、证据对齐F1≈67%、病例诊断准确率90%；相比人工标注，专家修正时间减少63–75%，总编辑量和证据编辑量显著降低。

**⚠️ 局限性**

局限性：评估样本仅10例，缺乏大规模验证；自演进机制在多轮反馈中的长期效果尚未评估；框架对其他精神疾病或更细粒度任务的适应性需进一步验证。

---

## 516. Can We Trust Item Response Theory for AI Evaluation?

**arXiv ID:** 2607.15190 | [PDF](https://arxiv.org/pdf/2607.15190v1)

**作者:** Han Jiang `[一作]` (Johns Hopkins University), Susu Zhang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过仿真评估了四种IRT估计器在AI基准数据（模型数少、项目数多、能力分布偏斜）下的可靠性。

**💡 创新点**

创新点在于系统化比较了模型数、基准长度、能力分布等条件对IRT参数恢复和排名准确率的影响，并给出了针对不同使用场景的实践指南。

**🔧 技术方法**

技术上使用了1PL、2PL、3PL三种IRT模型，结合四种估计方法（MML‑EM、MCMC、VI、PSN），并采用ADEMP框架进行大规模仿真。

**📊 数据集**

数据集来源于OpenLLM公开的六大基准（ARC‑Challenge、HellaSwag、MMLU、TruthfulQA、WinoGrande、GSM8K）的真实二进制响应矩阵。

**📈 对比分析**

比较方法包括计算计算可行性（失败率、运行时）、排名恢复（Kendall τ）、项参数回归（Pearson相关、平均绝对误差）以及短测量预测（Fisher信息挑选项后的排名恢复）。结果显示：MCMC在参数回归上最优但计算量巨大；VI最快但在项参数恢复和排名上表现不稳定；MML‑EM失败率高；PSN稳定性好、准确度介于两者之间。

**⚠️ 局限性**

局限性包括未在统一硬件上评估极限、未考虑30–100样本规模的中间情况、仅限单维IRT模型、未纳入项内容特征、并未深入验证估计器的收敛性与不确定性估计。

---

## 517. Mech: Mechanised Choreographic Programming

**arXiv ID:** 2607.15174 | [PDF](https://arxiv.org/pdf/2607.15174v1)

**作者:** Xueying Qin `[一作]` (University of Southern Denmark), Fabrizio Montesi `[通讯]` (University of Southern Denmark)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文实现了首个支持一般分支、一般递归以及舞蹈式非确定性选择的协作式编程（CP）形式化框架，并在Lean 4中完成了超过40,000行代码的机器化验证；

**💡 创新点**

创新点在于提出了新的语义模型以正确捕获非确定性选择与并发的交互，并通过机理化推导得出了完整性与一致性的证明，首次在机器检验环境中同时实现了全功能CP的安全性（通信安全）与无死锁性；

**🔧 技术方法**

技术方法主要包括：Lean 4定理证明器、可判定的有限函数FinFun、完全可合并的合并操作（merge）与端点投影（EPP）算法、以及多层次的离散执行与递归调用模型；

**📊 数据集**

本文未使用传统意义上的数据集，而是通过形式化证明与理论推导展示其正确性；

**📈 对比分析**

由于本文为理论机理化工作，未进行实验或性能比较；

**⚠️ 局限性**

局限性包括：仅支持同步通信、未扩展到异步模型、未提供运行时实现或编译器、缺乏实证性能评估。

---

## 518. Scaling Behavior Foundation Model for Humanoid Robots

**arXiv ID:** 2607.15163 | [PDF](https://arxiv.org/pdf/2607.15163v1)

**作者:** Weishuai Zeng `[一作]`, Jingbo Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一套可扩展的行为基础模型（BFM）框架，用于通用人形机器人控制，并通过对学习范式、行为数据和模型架构的系统性研究，构建了一条高效的规模化路径。

**💡 创新点**

创新点主要包括：①将运动跟踪视为统一的代理任务，采用全局坐标下的整体轨迹重现；②引入“宽度+深度”并行采样与“参考动作多样化”相结合的训练数据策略；③提出面向行为表示的 Humanoid Transformer，利用 RMSNorm 形成结构化的潜在空间，且不依赖辅助正则化。

**🔧 技术方法**

技术上使用了 Proximal Policy Optimization (PPO) 进行策略训练，采用异构演员-评论家设计、掩码目标姿态控制接口、全局根位姿跟踪奖励以及域随机化；模型为 Transformer‑style 结构，配合多时窗输入、交叉注意力和 RMSNorm。

**📊 数据集**

数据集方面，构建了约 1.02 亿帧（50 FPS）人类动作库，来源涵盖 LAFAN、AMASS、OMOMO、GRAB、SnapMoGen、FineDance、BONES‑SEED、Embody3D 等多源数据，随后通过两阶段重目标化得到符合目标人形的动作。

**📈 对比分析**

与 GMT、TWIST、SONIC 等现有全身控制器对比，所提 BFM 在全身控制下的成功率、全局/局部 MPKPE/MPKRE 均显著优于对手（例如 BFM‑Global 成功率 97.8% 伴随全局 MPKPE 0.08，远超 86% 的 baseline）。

**⚠️ 局限性**

局限性在于：①控制接口的抽象尚未最优，难以与未来高层策略无缝集成；②训练基础设施相对有限，规模化仍受 GPU 并行度与数据处理瓶颈制约；③模型在极端稀有行为上的泛化尚需进一步验证。

---

## 519. Language Identification via Compositional Data Analysis: A Linear-Time Classifier Based on Log-Ratio Geometry

**arXiv ID:** 2607.15238 | [PDF](https://arxiv.org/pdf/2607.15238v1)

**作者:** Paul-Andrei Pogăcean `[一作]` (Babeș-Bolyai University), Sanda-Maria Avram `[通讯]` (Babeș-Bolyai University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于组合数据分析的确定性线性时间语言识别算法，通过将字符与二元组频率映射到 CLR 空间并计算 Aitchison 距离，实现无训练、可解释的判定。

**💡 创新点**

创新点在于将 CoDA 与 CLR 变换结合，解决频率向量的几何失真，同时利用大写与二元组特征与长度相关的重音调整，得到高效、可解释的判定。

**🔧 技术方法**

所用技术包括组合数据分析、中心对数比变换、拉普拉斯平滑、Aitchison 距离计算以及基于欧氏距离的组合分数加权。

**📊 数据集**

实验使用英语、德语、土耳其语、罗马尼亚语、匈牙利语和荷兰语的多源语料（Google Books、DeReKo、CoRoLa、OPUS、LDDS 等）构建参考配置，并在 LDDS 与 OPUS 子集上评估。

**📈 对比分析**

与原始欧氏、曼哈顿、余弦距离以及 FastText 等方法对比，CoDA 在短文本 (<50 字符) 上达到 84% 以上准确率，在中长文本 95.6% 以上，长文本 100%，显著优于传统方法。

**⚠️ 局限性**

局限性包括仅适用于单语文本、对代码混合文本表现欠佳、只支持拉丁字母脚本、对大脚本需扩展特征维度，并在极短文本上仍受平滑影响。

---

## 520. NeuronSoup: Evolving Asynchronous, Shared-Neuron Temporal Graphs without Backpropagation

**arXiv ID:** 2607.15217 | [PDF](https://arxiv.org/pdf/2607.15217v1)

**作者:** Subodh Kalia `[一作]` `[通讯]`, Subodh Kalia

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于遗传算法进化的异步共享神经元时间图（NeuronSoup）实现无梯度的分类器，利用路径级别的可变延迟与状态累积实现动态计算深度和横向交互。

**💡 创新点**

核心创新在于：①通过可变延迟控制事件顺序实现异步计算；②共享神经元的状态累积产生无设计的横向抑制/激励；③采用路径块级交叉与混合离散连续基因的变异，使遗传算法能高效搜索组合复杂结构；④不依赖可微图，突破了传统反向传播的限制。

**🔧 技术方法**

技术手段包括：实数向量编码的固定长度基因组、路径级别的交叉与变异、基于优先队列的离散事件仿真、Numba加速的稀疏矩阵运算，以及GPU无关的纯CPU并行评估。

**📊 数据集**

使用的实验数据集是MNIST手写数字（10类），输入特征来自冻结的ResNet‑18提取的512维向量。

**📈 对比分析**

在MNIST上通过10,000代演化得到的模型在测试集上获得85.9%准确率，模型大小115 KB；相比线性分类器（92–94%）和小型MLP（96–97%）略低，但在参数量、运算量和能源效率上远优于传统CNN/ViT等梯度训练网络。

**⚠️ 局限性**

局限性包括：①准确率仍落后于梯度优化方法；②需要大量世代和计算资源进行演化搜索；③只在MNIST上验证，缺乏对更复杂数据或大规模任务的证明；④缺乏可解释的训练流程，依赖随机搜索与选择压力。

---

## 521. A Census of New Snake-in-the-Box Records

**arXiv ID:** 2607.15270 | [PDF](https://arxiv.org/pdf/2607.15270v1)

**作者:** Paul Orland `[一作]`, Sergei Gukov `[通讯]`

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过大规模计算搜索，在9到13维超立方体中找到新的最长蛇形路径（snake）和螺旋路径（coil），并显著提升了已知的下界；同时提供了所有记录长度路径的可验证数据集。

**💡 创新点**

创新点包括：首次在每个维度9-13上获得比之前记录更长的蛇形路径，公布了131条长度191的独立蛇形路径（在Q9中），并为螺旋路径和对称螺旋路径设置了新的下界；此外，作者发布了完整的可验证数据仓库，为后续研究提供了重要资源。

**🔧 技术方法**

技术手段主要是大规模并行计算搜索（利用云TPU、GPU和高性能集群），结合遗传算法、蒙特卡洛树搜索等现有搜索框架，对超立方体路径空间进行系统探索和优化。

**📊 数据集**

数据集：所有记录长度路径的转移序列（transition sequences），以及对应的螺旋路径和对称螺旋路径，已托管在GitHub公开仓库 https://github.com/Math-AI-Caltech/Snake-in-the-Box。

**📈 对比分析**

与以往最佳记录比较：在9维提升1条（191>190），11维提升9条（746>737），12维提升11条（1476>1465），13维提升22条（2922>2900）；螺旋路径同样在9-11维提升4条；对称螺旋路径在10维提升8条、11维提升8条。性能上，作者报告在9维已找到131类独立路径，表明搜索的广度和深度。

**⚠️ 局限性**

局限性：仅给出下界，未能证明在高维度下这些路径为最优；对10维及以上的完整性未作保证；搜索依赖昂贵的硬件资源，可能不易复现；对称性划分后记录路径数量可能仍不完整。

---

## 522. Pretraining Data Can Be Poisoned through Computational Propaganda

**arXiv ID:** 2607.15267 | [PDF](https://arxiv.org/pdf/2607.15267v1)

**作者:** Victoria Graf `[一作]` (University of Washington), Kyle Lo `[通讯]` (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究通过在公共讨论接口（如评论区）进行第三方内容注入，探讨了对语言模型预训练数据的毒化攻击，并提出了一种估计注入内容是否被最终包含在训练语料中的分析方法。

**💡 创新点**

创新点在于：①发现第三方注入（尤其是公共讨论接口）是可行的预训练数据毒化攻击面；②提出了基于注入、捕获、过滤三阶段的概率估计方法；③实验证明即使低比例注入也能影响大规模模型。

**🔧 技术方法**

技术包括：利用Selenium模拟批量评论注入；使用Resiliparse提取HTML文本；采用AllDressed + Dolma3的质量过滤器；构建概率模型P(include|v,S)；在多尺寸模型上进行预训练与SFT实验。

**📊 数据集**

使用数据集：Common Crawl 2025-51作为注入表面评估；Dolma3 Web子集作为预训练语料；Dolci SFT数据集用于指令微调；此外还使用公开的“程序化广告”作为对比。

**📈 对比分析**

比较方法：在不同模型规模（65M–1.3B）下测量注入比例（0.1%、0.01%、0.001%）对模型偏好概率的影响，基线与受毒化模型的差异（Δ）被量化；实验显示基线模型对毒化保持稳定，SFT后影响随规模下降。性能表现：即使仅0.1%注入，模型偏好差异可达约20%。

**⚠️ 局限性**

限制包括：仅使用Common Crawl作为爬虫代理，真实训练爬虫可能不同；静态HTML检测可能高估可注入表面；未在真实网站上进行注入实验；估计基于Dolma3过滤器，其他过滤策略可能得到不同结果。

---

## 523. HoloGeo: Mitigating Landmark Bias in Geo-localization via Evidence-Driven Reasoning

**arXiv ID:** 2607.15255 | [PDF](https://arxiv.org/pdf/2607.15255v1)

**作者:** Pengcheng Zhou `[一作]` (National University of Singapore), Wynne Hsu `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究图像地理定位中由于显著地标造成的偏差，提出定量度量BI、BH，构建LandmarkBias-3K诊断基准和BF-30K多证据训练集，并提出HoloGeo框架通过多证据推理与强化学习降低地标偏差。

**💡 创新点**

① 将地标偏差拆解为强度与危害两维定量指标；② 设计针对地标偏差的专用基准和高质量训练集；③ 采用证据驱动的多维奖励机制，鼓励模型平衡关注多种地理线索；④ 通过GRPO强化学习提升推理一致性和证据覆盖率。

**🔧 技术方法**

Vision‑Language Models (Qwen2.5‑VL, InternVL3, etc.)，SFT + LoRA，GRPO 强化学习，GroundingDINO 边框检测，多维奖励 (R_geo, R_box, R_CLR) 与交叉模型验证。

**📊 数据集**

MP‑16、GLDv2、IM2GPS、IM2GPS3K、YFCC4K、LandmarkBias‑3K、BF‑30K。

**📈 对比分析**

与传统图像定位、通用VLM、专用VLM等模型在 IM2GPS、YFCC4K、LandmarkBias‑3K 上进行对比；HoloGeo 在 LandmarkBias‑3K 城市级准确率提升至 27.27%（比最佳 23.57% 提升约 3.7%），在 IM2GPS 和 YFCC4K 上保持或超过 SOTA。

**⚠️ 局限性**

仍受训练数据规模与多样性的限制，未针对少样本/零样本场景进行充分验证，对外部地理知识库的集成不足。

---

## 524. MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators

**arXiv ID:** 2607.15273 | [PDF](https://arxiv.org/pdf/2607.15273v1)

**作者:** Yushi Huang `[一作]`, Tianyu Pang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了 MeanFlowNFT，一个在 MeanFlow 平均速度网络上执行前向过程强化学习的框架，能够在保持少步采样效率的同时提升生成质量。

**💡 创新点**

创新点在于利用 MeanFlow 身份将平均速度映射到瞬时速度，从而在瞬时速度空间上使用 DiffusionNFT 的奖励驱动优化，同时保留平均速度网络用于高效采样；并给出了理论上的策略提升保证。

**🔧 技术方法**

技术包括：DiffusionNFT 前向过程 RL、Flow Matching、MeanFlow 平均速度学习、Finite‑Difference 估计总导数、EMA 参考策略、LoRA 微调、CFG‑free 采样以及多任务奖励设计。

**📊 数据集**

实验数据集涵盖：图像生成使用 Stable Diffusion 3.5‑Medium（512×512 训练、1024×1024 评估），视频生成使用 1.3B 预训练模型（480p 81 帧）以及公开的 AnyFlow 检查点。

**📈 对比分析**

与 DMD、CDM、AnyFlow、DiffusionNFT、RTDMD 等少步和多步 RL 基线对比，MeanFlowNFT 在大多数 ImageReward、CLIPScore、VBench 等指标上取得领先，且在仅 4 步采样下即可匹配或超越 40‑步 DiffusionNFT，视频任务中 4 步即可超过 50‑步 RL。

**⚠️ 局限性**

局限性包括：仅探讨了 DiffusionNFT 风格的前向 RL，未覆盖 RAM、AWM 等其他前向目标；仅针对 MeanFlow，未验证到其他流图模型；缺乏对更复杂奖励或大规模模型的实验。

---

## 525. Motion-Conditioned Multi-View Fusion for Myocardial Infarction Localization from Echocardiography

**arXiv ID:** 2607.15268 | [PDF](https://arxiv.org/pdf/2607.15268v1)

**作者:** Guang Yang `[一作]` (University of Oxford), Vicente Grau `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于运动条件的多视角融合网络 MCF‑Net，用于从双视角超声心动图中对心肌梗死进行分段级定位

**💡 创新点**

创新点在于：1）仅用一帧模板进行极稀疏标注，借助 CoTracker3 生成全视频的心肌轨迹；2）设计运动引导的软细化模块（MSR），利用轨迹产生的 Gaussian 软掩模增强心肌区域的视觉特征；3）构建运动条件的跨视角融合机制，先用 FiLM 对视觉特征进行运动调制，再通过多头注意力聚合两视角信息；4）将这些模块与预训练的 EchoPrime 基础模型无缝结合，实现高精度分段级梗死定位。

**🔧 技术方法**

使用的技术包括：极稀疏标注+模板匹配、CoTracker3 轨迹跟踪、EchoPrime 预训练基础模型、运动引导软细化（Gaussian 软掩模）、FiLM 条件化、跨视角多头注意力融合以及 BCE 损失训练。

**📊 数据集**

采用公开的 HMC‑QU 超声心动图数据集，共 162 条 A4C 视角和 160 条 A2C 视角视频，分为 112/16/32 病例用于训练/验证/测试。

**📈 对比分析**

与七种基线（仅运动、仅视觉、基础模型、Naïve 融合等）比较，MCF‑Net 在 AUROC 87.6、PR‑AUC 74.7、F1 72.4、Accuracy 84.9 方面均超过对手，特别是在分段级定位上提升显著。

**⚠️ 局限性**

局限性包括：仅在单中心 HMC‑QU 数据集验证，跨中心泛化尚未评估；极稀疏标注对低质量图像的鲁棒性仍有限；以及对运动条件的阈值 λ 的敏感性需要进一步自动化选择。

---

## 526. Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models

**arXiv ID:** 2607.15277 | [PDF](https://arxiv.org/pdf/2607.15277v1)

**作者:** Patrik Wolf `[一作]` (Max Planck Institute for Intelligent Systems), Celestine Mendler-Dünner `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构造二叉条件树，对大型语言模型（LLM）在上下文学习中的条件推理能力进行评估，检验其是否满足概率一致性（如全概率定理）以及与人类基准数据的对齐情况。

**💡 创新点**

提出了“宏观谬误”（macro fallacy）——直接从模型获取总体估计往往比先细化到子群体再聚合得到的估计更不准确；同时引入了自一致性（self‑consistency）检查，包括分裂一致性和顺序一致性，用以无监督评估模型的条件分布是否内部一致。

**🔧 技术方法**

采用二叉条件树（binary conditioning tree）构造不同粒度的子群体，利用提示工程在LLM中抽取条件概率和先验概率，并通过全概率公式进行聚合；另外使用水准化Wasserstein‑1距离衡量概率分布差异。

**📊 数据集**

主要使用美国人口普查调查（2024 ACS）的人口收入分布以及世界价值观调查（WVS）中的意见数据作为对齐基准，并在此基础上设计了合成预测任务（网球对战、幻想战斗）进行自一致性检验。

**📈 对比分析**

实验显示，LLM在不同模型（如GPT‑5.4、Sonnet、Grok、Qwen3.6 Plus 等）和任务中均出现明显的自一致性违背；重构聚合的误差往往低于直接聚合，宏观谬误得到证实；但整体自一致性分数普遍较低，未随模型能力提升而显著改善。

**⚠️ 局限性**

局限性在于对齐实验主要依赖ACS数据，未覆盖更多领域；自一致性仅是必要条件，未必保证预测准确；抽样和提示设计对结果有较大影响，且未提供直接提升自一致性的有效方法。

---

## 527. RoboTTT: Context Scaling for Robot Policies

**arXiv ID:** 2607.15275 | [PDF](https://arxiv.org/pdf/2607.15275v1)

**作者:** Yunfan Jiang `[一作]` (NVIDIA), Linxi "Jim" Fan `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于 Test‑Time Training 的机器人策略模型和训练方案，将视觉‑语言‑动作（VLA）模型的上下文长度扩展到 8K 步（约五分钟），实现一次性视频演示学习、在线策略改进、抗扰动能力和更强闭环性能。

**💡 创新点**

核心创新在于将 TTT（快权重更新）嵌入 VLA 变压器中，配合序列动作强迫（sequence action forcing）与截断时间反向传播（TBPTT）实现长上下文训练；并首次证明上下文长度可视为机器人基础模型的可扩展轴。

**🔧 技术方法**

技术手段包括：快权重流匹配损失、梯度下降更新快权重、学习可调门控机制、两层 MLP 快权重模型、序列动作强迫、TBPTT、DAgger Distillation 等。

**📊 数据集**

使用的训练数据主要是三组真实机器人长时序操纵任务（Pup Go Car、Circuit、Gear Bot）的数小时数据（总计约19小时）以及人类 egocentric 视频演示；还用到了预训练的多模态视觉‑语言模型和 GR00T‑N1.7 基础网络。

**📈 对比分析**

与单步上下文、短期历史（1 帧）以及用 DeltaNet 替代 TTT 的基线相比，-8K 模型在三项任务上的平均任务完成率达 79%（比单步提升 87%，比最佳基线提升 41%），在长任务 Gear Bot 上实现 2/10 的完全成功率，且在单次视频演示、扰动恢复等子任务中显著优于所有基线。

**⚠️ 局限性**

局限性包括：训练成本随上下文长度显著增加，需更高效的 TTT 训练方法；对极端失败模式的处理仍不完善；未结合强化学习直接优化任务成功率，且当前实现主要依赖大量标注数据。

---

## 528. Online Neural Space Time Memory for Dynamic Novel View Synthesis

**arXiv ID:** 2607.15271 | [PDF](https://arxiv.org/pdf/2607.15271v1)

**作者:** Baback Elmieh `[一作]` (University of Washington), Xuan Luo `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种在线动态新视角合成框架 NSTM，能够在多视角流式视频中实现分钟级长时记忆并保持实时推理。

**💡 创新点**

创新点包括：1）将记忆更新与推理解耦，采用周期性记忆更新；2）利用跨视角注意力对齐动态失真；3）引入辅助记忆损失和记忆缓存机制，提升长期稳定性；4）使用 L2 内部损失避免梯度膨胀。

**🔧 技术方法**

技术手段：Test‑Time Training（TTT）快权重记忆、线性注意力、交叉视角注意力、L2 内部损失、Newton–Schulz 正交化、记忆缓存。

**📊 数据集**

实验数据集：MVHumanNet++（60k多视角人类运动序列）。

**📈 对比分析**

与无状态 LVSM、状态化 LaCT‑NVS、Token‑Mem 进行对比；在分钟级记忆压力测试中，NSTM 维持 30+ dB PSNR、37 FPS 实时率，显著优于其他方法。

**⚠️ 局限性**

局限性：1）周期性更新可能错过短暂事件；2）记忆容量有限；3）对复杂运动和长范围相机/主体位移的对齐仍具挑战。

---

## 529. Decoding Market Emotion from Blockchain Activity: A Data-Driven Sentiment Classifier

**arXiv ID:** 2607.15258 | [PDF](https://arxiv.org/pdf/2607.15258v1)

**作者:** Arthur G. Bubolz `[一作]` (Federal University of Rio Grande), Bruno L. Dalmazo `[通讯]` (Federal University of Rio Grande)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种利用区块链交易与日均推特情绪标签相结合的模型，解释比特币市场情绪。

**💡 创新点**

创新点在于将链上数据与社交媒体情绪聚合为统一数据集，用模型解释市场情绪而非预测价格，并通过SHAP提升可解释性。

**🔧 技术方法**

采用特征工程、MinMax归一化、交叉验证以及XGBoost、决策树、随机森林等传统机器学习算法，并使用SHAP进行特征重要性分析。

**📊 数据集**

数据集为从Blockchair获取的2013-2026年比特币链上每日指标，以及Kaggle提供的约1600万条标记情绪的推文数据。

**📈 对比分析**

与其他传统机器学习方法比较，XGBoost在5折交叉验证中平均F1≈0.84，最终训练测试分离时准确率83%，精度81%，召回88%。

**⚠️ 局限性**

局限包括仅使用二分类情绪、忽略中性标签、未采用深度学习或实时API，且模型对极端市场事件的泛化能力未知。

---

## 530. SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration

**arXiv ID:** 2607.15257 | [PDF](https://arxiv.org/pdf/2607.15257v1)

**作者:** Yuyao Zhang `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套多智能体框架，用来把开放域信息检索任务转化为有根证据的关系模式完成，并通过系统层面维护搜索状态、证据图、覆盖率图和失败内存，实现高效、可靠的长周期检索与生成。

**💡 创新点**

创新点包括：①将检索任务表述为关系模式完成并强制每个单元都有可验证的引用；②引入 Search-Oriented Context Management（SOCM）将所有中间状态外部化，避免对话历史衰减导致的覆盖盲点；③使用管道并行调度和中间件驱动的执行管道，实时填补未完成的模式缺口并对停滞或预算超限做自动干预；④构建分层搜索技能库（全局、策略、访问三层），实现跨源、跨任务的复用与学习。

**🔧 技术方法**

技术栈包括：大语言模型（如 GPT‑4 或同类 LLM）与浏览器工具集成；SOCM 结构（Frontier Task、Evidence Graph、Coverage Map、Failure Memory）；管道并行调度器；中间件三层（Context、Evidence Extraction、Sensor）；分层搜索技能系统；以及多智能体调度框架（Orchestrator‑Explore‑Search‑Writer）。

**📊 数据集**

使用了两个公开基准：WideSearch（200 问题，英中混合）和 GISA（373 真实用户查询），两者均要求构造完整表格并进行多跳推理与跨源聚合。

**📈 对比分析**

与单体式 ReAct、Plan‑and‑Solve 以及多体式 A‑MapReduce、Web2BigTable、Table‑as‑Search 等基线进行比较。结果显示：在 WideSearch 上，Item‑F1 最高达 80.3，Row‑F1 56.5，均比最强基线高 4.3、2.0 分；在 GISA 上，Set‑F1 最高 76.5，比最强基线提升 13.4 分。整体提升主要来自更高的覆盖率和更精准的证据归属。

**⚠️ 局限性**

局限性包括：①依赖大量外部工具与浏览器，运行时成本高；②当前模型主要面向文本检索，未覆盖多模态信息；③系统对极大规模任务的可扩展性尚未充分验证；④缺乏对不同知识源异构性的自适应策略，可能在非公开网站或受限数据库上表现不佳。

---

## 531. SceneBind: Binding What and Where Across Vision, Audio and Language

**arXiv ID:** 2607.15265 | [PDF](https://arxiv.org/pdf/2607.15265v1)

**作者:** Mingfei Chen `[一作]` (University of Washington), Eli Shlizerman `[通讯]` (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SceneBind框架，联合建模视觉、音频和语言的全局语义与对象级语义-空间槽，形成统一的语义-空间表示；

**💡 创新点**

创新点在于引入对象级语义-空间槽并结合全局语义，使用双向匹配与多任务监督实现跨模态语义与空间对齐；

**🔧 技术方法**

采用冻结的SigLIP2视觉编码器、M2D-CLAP音频编码器，轻量化对齐模块，交叉注意力解码器以及多目标对比、位置回归与置信度监督；

**📊 数据集**

使用新构建的真实世界双耳音频-视觉-语言数据集（带结构化空间标注），以及AudioCaps、MS‑COCO等公开数据进行预训练；

**📈 对比分析**

在跨模态场景检索、空间检索与对象定位任务上与AudioCLIP、SpatialCLAP、ImageBind等基线相比，SceneBind在V↔T检索、空间检索以及零样本空间定位等指标上实现显著提升（如V↔T召回率提升约28%）；

**⚠️ 局限性**

局限在于对真实世界空间对齐数据的依赖，场景动态和长期时间一致性建模仍待改进，且模型对稀疏或高度重叠对象的区分仍有限。

---

## 532. Hierarchical Denoising For Multi-Step Visual Reasoning

**arXiv ID:** 2607.15278 | [PDF](https://arxiv.org/pdf/2607.15278v1)

**作者:** Zezhong Qian `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HDR（Hierarchical Denoising for Visual Reasoning）框架，实现视频层次化稀疏注意的多步推理与低延迟流式生成；

**💡 创新点**

将视频潜在向量组织为树形层级，并在不同层级匹配去噪强度，允许粗糙层保持多种全局计划，细层逐步细化为具体视觉状态；

**🔧 技术方法**

基于扩散模型的层级流匹配目标、稀疏层级注意模式 SHAP、层级匹配的去噪预算与 KV 缓存共享；

**📊 数据集**

在 18,000 条多步推理视频数据上进行训练，构造包含迷宫、汉诺塔、一行画、滑块拼图、Sokoban、水倒等六个 OOD 任务的 Benchmark；

**📈 对比分析**

与全时序双向扩散、VideoMAE、CausalForcing、VideoGPT 等基线对比，HDR 在成功率从 34.22% 提升至 60.29%（+76.2% 相对提升），平均进度从 76.00 提升至 89.56，推理延迟仅 0.70s/latent，速度比双向扩散快 54.2 倍；

**⚠️ 局限性**

对低去噪步数或数据量减少时仍保持鲁棒，但在极低预算或极少数据下性能仍受限；实现细粒度控制需要额外的层级设计，且目前仍需人工设计层级结构和去噪预算。

---

## 533. SciDiagramEdit: Learning to Edit Scientific Diagrams from Paper Revisions

**arXiv ID:** 2607.15272 | [PDF](https://arxiv.org/pdf/2607.15272v1)

**作者:** Yasheng Sun `[一作]` (King Abdullah University of Science and Technology), Jürgen Schmidhuber `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SciDiagramEdit框架，通过学习论文修订对的矢量图形编辑指令，训练可自我改进的SVG编辑Agent，实现自然语言驱动的科学图形编辑。

**💡 创新点**

将自然语言论文修订作为监督源，利用技能演化（skill evolution）从执行轨迹中提炼可移植的编辑规则，并在可编辑矢量层上完成细粒度操作。

**🔧 技术方法**

Agentic learning via skill evolution、代码生成式SVG编辑器、基于视觉语言模型的Judge、示范感知Coach以及自监督的判定与回报机制。

**📊 数据集**

SciDiagramEdit基准，包含364个来自arXiv修订的前后对照科学图形，以及2628条原作者意图的原子编辑说明。

**📈 对比分析**

与AutoFigure-Edit单步编辑器和GPT-Image-2栅格重绘器对比，在语义与美学指标上，SciDiagramEdit在语义检查上与GPT-Image-2持平且优于单步编辑器，在美学得分接近作者原稿但略逊于栅格编辑器；技能迁移亦显示跨模型提升。

**⚠️ 局限性**

仅覆盖显式指令的单步修订，缺乏多步推理与上下文推断；技能演化由外部调度，缺乏完全自主；数据规模有限，未探索更大规模训练导致的潜在新行为。

---

## 534. Beyond Success Rate: Cost-Aware Evaluation of Offensive and Defensive Security Agents

**arXiv ID:** 2607.15263 | [PDF](https://arxiv.org/pdf/2607.15263v1)

**作者:** Paul Kassianik `[一作]`, Yaron Singer `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估安全代理在攻击（Cybench CTF）和防御（Splunk BOTS v1 SOC调查）任务中的成本-成功权衡，并在统一的预算框架下比较模型性能。

**💡 创新点**

引入预算视角将安全代理性能映射为经济效率，揭示攻击与防御任务的不同规模规律，并提供公开评估网站与污染控制机制。

**🔧 技术方法**

采用Inspect框架、ReAct式代理与自动压缩技术，对模型推理和外部工具调用进行成本计量与工具调用统计，使用价格表实现精确的美元费用追踪。

**📊 数据集**

使用Cybench专业CTF任务集（39道）和Splunk BOTS v1 SOC调查（31道 Po1s0n1vy/Cerber 题）作为评估数据集，并对无工具、无上下文等污染控制进行额外实验。

**📈 对比分析**

通过设定固定成本上限与回溯预算重放来比较不同模型的成功率、工具调用量和成本/千分点；Cybench 中高预算可提升成功率，而 BOTS v1 则更依赖工具纪律，Claude Opus 4.8 在两类任务中表现最优。

**⚠️ 局限性**

实验为观察性、仅覆盖 BOTS v1，未扩展到 BOTS v2/3；工具调用上限导致高量模型截断；污染控制需进一步完善以确保公开评估的可信度。

---

## 535. The Power of the Score Sequence of a Tournament

**arXiv ID:** 2607.15260 | [PDF](https://arxiv.org/pdf/2607.15260v1)

**作者:** Prantar Ghosh `[一作]` (Tennessee Technological University), Sagnik Mukhopadhyay `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了只利用锦标赛得分序列即可决定的图问题，并给出了一个统一的多项式时间框架

**💡 创新点**

首次用循环逆转不变性对图问题进行完全表征，并提出从骨架和入度序列高效构造满足该序列的有向图

**🔧 技术方法**

采用循环逆转、流/巡回网络（circulation）以及欧拉回路构造等技术

**📊 数据集**

本文为理论性工作，没有使用实验数据集

**📈 对比分析**

在单次流、剪枝查询和通信模型下，得到的空间/查询量为 O(n log n)、O(n) 或 O(n+k)，相较已有结果实现多项式时间和近线性时间，且在多项式时间内完成全局/最小 s,t‑cut

**⚠️ 局限性**

仅适用于循环逆转不变的图问题，对非此类问题无法直接应用；在剪枝查询模型下的最小切割仍缺乏最优下界

---

