# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-11 | 今日论文总数: 510

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. UI-Venus-1.5 Technical Report

**arXiv ID:** 2602.09082 | [PDF](https://arxiv.org/pdf/2602.09082v1)

**作者:** Veuns-Team `[一作]`, Weiqiang Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款端到端的多模态 GUI 代理 UI‑Venus‑1.5，能够在移动、Web 等多平台上理解自然语言指令、感知屏幕视觉信息并执行交互动作。

**💡 创新点**

创新点包括：① 三阶段训练策略——大规模 Mid‑Training 注入 GUI 知识、任务特定离线 RL + 在线 RL 结合；② 统一动作空间与拒绝机制；③ 引入多层任务生成与分层采样；④ 采用 TIES‑Merge 模型合并，实现单模型跨域性能。

**🔧 技术方法**

核心技术包括：Qwen3‑VL 视觉‑语言主干；10B token 的 GUI 语料 Mid‑Training；GRPO 离线与在线强化学习；KL 与熵正则化；任务生成与分层采样；TIES‑Merge 模型合并；以及扩展的动作空间（Hover、DoubleClick、Hotkey 等）。

**📊 数据集**

使用了 30+ 规模不等的 GUI 数据集，累计约 10B token，主要包括 Mind2Web、ShowUI、AITW、ScreenSpot‑Pro、VenusBench‑GD、VenusBench‑Mobile、AndroidWorld、AndroidLab、WebVoyager 等。

**📈 对比分析**

在 GUI Grounding 和 Navigation 任务上与现有基准进行对比：Grounding 任务如 ScreenSpot‑Pro 达 69.6%，VenusBench‑GD 达 75%，OSWorld‑G‑R 达 76.4%；Navigation 任务如 AndroidWorld 77.6%，AndroidLab 68.1%，VenusBench‑Mobile 21.5%，WebVoyager 76%。与大型基线（MAI‑UI‑32B、Mobile‑Agent‑v3、GPT‑4o 等）相比，UI‑Venus‑1.5 在绝大多数指标上实现 SOTA 或显著提升，同时模型规模更小。

**⚠️ 局限性**

局限性：模型合并后对细粒度 Grounding 的精度略有下降（≈1.4%）；在线 RL 训练仍受环境动态性和奖励稀疏性限制；中文生态和多语言、多平台的通用性尚待进一步验证。

---

## 2. ML-DCN: Masked Low-Rank Deep Crossing Network Towards Scalable Ads Click-through Rate Prediction at Pinterest

**arXiv ID:** 2602.09194 | [PDF](https://arxiv.org/pdf/2602.09194v1)

**作者:** Jiacheng Li `[一作]` (Pinterest), Kungang Li `[通讯]` (Pinterest)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的特征交互模块——Masked Low‑Rank DCN，用实例条件掩码在低秩交叉层中动态选择并放大重要交互方向。

**💡 创新点**

创新点在于将低秩DCNv2的高效交叉结构与MaskNet的实例化掩码机制相结合，既保持了计算效率，又显著提升了表达能力，实现了更好的AUC–FLOPs权衡。

**🔧 技术方法**

技术包括低秩深度交叉网络（Low‑Rank DCN）、实例化掩码（instance‑guided mask）、LayerNorm、与多专家（MMoE）架构的兼容性。

**📊 数据集**

使用Pinterest内部大规模广告点击率（CTR）预测数据集进行实验。

**📈 对比分析**

与DCNv2、MaskNet、WuKong、RankMixer等主流交互模块在相同FLOPs下对比，Masked Low‑Rank DCN取得最高AUC提升，并在更高计算量下表现出更好的可扩展性；在线A/B测试显示CTR提升1.89%、gCTR提升2.17%、oCTR提升1.90%，且推理成本保持不变。

**⚠️ 局限性**

局限性包括仅在Pinterest内部数据与特定硬件环境验证；未探究更大计算预算下的极限性能；低秩约束可能限制极高阶交互的表达，需进一步改进更具表现力的机制。

---

## 3. LLMAC: A Global and Explainable Access Control Framework with Large Language Model

**arXiv ID:** 2602.09392 | [PDF](https://arxiv.org/pdf/2602.09392v1)

**作者:** Sharif Noor Zisad `[一作]` (University of Alabama at Birmingham), Ragib Hasan `[通讯]` (University of Alabama at Birmingham)

**通讯引用:** 3819 | [OpenAlex ID](https://openalex.org/A5076460405)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了LLMAC，一个利用大语言模型统一RBAC、ABAC、DAC的可解释访问控制框架

**💡 创新点**

将多种传统模型融合为单一LLM决策层，并通过结构化提示和合成数据实现高准确率与可解释性

**🔧 技术方法**

使用Mistral 7B + LoRA微调、Unsloth、量化与批处理技术，以及结构化输出的自然语言解释

**📊 数据集**

构造10,000条合成JSON记录，基于课堂管理系统的七项访问控制政策，训练集占10%，其余用于评估

**📈 对比分析**

与RBAC、ABAC、DAC对比，采用准确率、精确率、召回率、F1等指标；LLMAC准确率98.5%，远超ABAC 58.5%、DAC 27.5%及RBAC 14.5%，关键安全动作准确率均保持在90%以上

**⚠️ 局限性**

推理延迟相对较高；模型安全性（prompt injection、数据泄露）和量化/批处理的进一步优化仍需研究

---

## 4. Disambiguating Anthropomorphism and Anthropomimesis in Human-Robot Interaction

**arXiv ID:** 2602.09287 | [PDF](https://arxiv.org/pdf/2602.09287v1)

**作者:** Minja Axelsson `[一作]` (University of Cambridge), Henry Shevlin `[通讯]` (University of Cambridge)

**通讯引用:** 1080 | [OpenAlex ID](https://openalex.org/A5019796085)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对人类机器人交互(HRI)中的人性化与拟人化概念进行区分和澄清，并提出各自的责任主体。

**💡 创新点**

首次在HRI文献中明确区分“anthropomorphism”（用户赋予人性）与“anthropomimesis”（开发者设计人性），并系统阐述两者的理论基础与责任归属。

**🔧 技术方法**

主要采用文献综述与概念分析方法，对已有定义进行对比并构建对照表。

**📊 数据集**

无实验数据，采用已有研究中的定义与测量工具（如Godspeed问卷、ABOT数据库）进行参考。

**📈 对比分析**

通过对比表格展示两概念的责任主体、机制和理论来源，未涉及量化性能评估，仅提供概念层面的对照。

**⚠️ 局限性**

缺乏可操作的量化测量标准，无法分离两者在实际机器人设计与用户感知中的具体影响，未来需要开发针对性评估工具。

---

## 5. Spectral Disentanglement and Enhancement: A Dual-domain Contrastive Framework for Representation Learning

**arXiv ID:** 2602.09066 | [PDF](https://arxiv.org/pdf/2602.09066v1)

**作者:** Jinjin Guo `[一作]` (JD.com), Qixia Jiang `[通讯]` (JD.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Spectral Disentanglement and Enhancement（SDE）框架，对多模态特征进行SVD分解，分离强信号、弱信号和噪声，并通过课程化增强实现特征谱平衡，进一步通过双域对比损失同步优化特征空间与谱空间的对齐。

**💡 创新点**

创新点在于：①首次将谱分解应用于特征空间进行动态拆分；②设计基于课程学习的谱增强策略，既提升主导特征又抑制噪声；③引入谱对比损失以保证全局谱一致性，防止对齐失衡。

**🔧 技术方法**

核心技术包括：奇异值分解（SVD）、谱阈值分区、课程化谱增强、双域（特征+谱）对比学习、基于正则化的噪声抑制。

**📊 数据集**

使用大规模多模态基准MMEB（包含36个子任务，涵盖分类、VQA、检索、视觉定位等）。

**📈 对比分析**

与VLM2Vec、CLIP、BLIP2、UniIR、MegaPairs等基线对比，在MMEB上平均Precision@1提升至65.6，显著优于所有对照模型，尤其在高分辨率下表现最强。

**⚠️ 局限性**

局限性：仅在静态图文任务上验证，未探讨时序或动态多模态场景；谱分解与增强参数依赖经验阈值，需进一步自适应调优；对极端噪声或少量样本情形的鲁棒性尚待深入研究。

---

## 6. Feature salience -- not task-informativeness -- drives machine learning model explanations

**arXiv ID:** 2602.09238 | [PDF](https://arxiv.org/pdf/2602.09238v1)

**作者:** Benedict Clark `[一作]`, Stefan Haufe `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在猫狗图像和动物车辆图像上构造可控的水印与亮度扰动，训练并评估多种特征归因方法，系统检验归因是否真正反映模型学到的关键信息。

**💡 创新点**

证明归因结果主要受测试图像低层结构（如边缘、亮度）驱动，而非模型学到的统计关联，从而质疑现有XAI方法在检测捷径学习和模型诊断中的可靠性。

**🔧 技术方法**

使用卷积神经网络结合五种主流归因方法（Deconvolution、Integrated Gradients、Gradient SHAP、LRP-ε、LRP-αβ）以及两种无模型基准（二维拉普拉斯边缘滤波器和原始像素强度）进行对比，并通过线性混合模型量化归因差异。

**📊 数据集**

数据来源为Kaggle的猫狗图像（4,800张每类）与COCO动物/车辆子集（15,000张每类），分别生成三种实验设置（confounded、balanced、baseline）以及颜色编码反转版本。

**📈 对比分析**

在不同训练设置下，所有归因方法对带水印或亮度变化的图像都显示显著提升的相对重要性（RIW/RIL平均≈2），但模型训练方式对归因差异的解释仅占≤3%方差，说明归因与模型学习关联弱，表现不佳。

**⚠️ 局限性**

局限在于仅测试CNN架构、默认参数、特定的水印/亮度操作，并未全面控制所有任务相关特征，因而结论可能不适用于更复杂或不同数据域的XAI应用。

---

## 7. Sci-VLA: Agentic VLA Inference Plugin for Long-Horizon Tasks in Scientific Experiments

**arXiv ID:** 2602.09430 | [PDF](https://arxiv.org/pdf/2602.09430v1)

**作者:** Yiwen Pang `[一作]` (Southeast University), Shimin Di `[通讯]` (Southeast University)

**通讯引用:** 250 | [OpenAlex ID](https://openalex.org/A5006260400)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个基于大语言模型的推理插件，利用已有的Vision‑Language‑Action模型在科学实验的长时序任务中自动生成过渡动作，从而解决任务间状态缺口问题；

**💡 创新点**

该插件仅在推理阶段插入LLM推理，无需重新训练VLA模型；通过检索训练数据获取目标关节位置并生成过渡动作代码，实现任务之间的无缝切换；

**🔧 技术方法**

使用Vision‑Language‑Action模型（π_0、π_0.5、fast），大型语言模型GPT‑5.2进行过渡动作推理，插入/切换模块，以及Autobio仿真环境与真实实验验证；

**📊 数据集**

对每个原子任务收集100条随机化演示数据，包含“清洁桌面”3个原子任务和6个长时序科学操作任务，共14个原子任务；

**📈 对比分析**

与基线VLA模型在相同任务下进行20次推理对比；Sci‑VLA将第2、3个原子任务的成功率分别从0%提升至20%/45%，fast从0%提升至25%，整体平均成功率提升约42%，并显著改善任务连贯性；

**⚠️ 局限性**

过渡动作生成可能产生幻觉，需要多轮生成；插件不处理原子任务内部高精度操作；对高质量演示数据的依赖性强；LLM推理延迟和网络不确定性；安全性未能完全保证。

---

## 8. Learning with Multiple Correct Answers -- A Trichotomy of Regret Bounds under Different Feedback Models

**arXiv ID:** 2602.09402 | [PDF](https://arxiv.org/pdf/2602.09402v1)

**作者:** Alireza F. Pour `[一作]` (University of Waterloo & Vector Institute), Shai Ben-David `[通讯]` (University of Waterloo & Vector Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究了在多答案学习问题下，针对三种不同反馈模型（误差未知、误差已知、集合式反馈），给出了可实现与非可实现在线学习的最优误差/误差上界，并揭示了三种模型在无偏置环境下的三分支（线性、O(√T logT)、常数）误差上界；

**💡 创新点**

创新点在于统一构建三种反馈模型的组合学维度（包括集合式 Littlestone 维度和误差已知/未知维度），得到精确的误差和误差上界；同时在非可实现设置下利用专家建议与Bandit技巧，实现了误差已知模型下 O(√(T·d·log|Y|)) 的下界；

**🔧 技术方法**

核心技术包括树式组合学维度定义、标准最优算法 (SOA)、权重多数算法、Bandit‑style专家建议算法（EXP3‑like）以及在线到批量的转换技巧；

**📊 数据集**

论文未涉及实验或具体数据集，主要为理论分析；

**📈 对比分析**

由于缺乏实验比较，无法给出性能数值，但理论上证明了在不同反馈下的误差上界与维度之间的匹配关系；

**⚠️ 局限性**

局限性在于仅给出理论上限，缺少实验验证；对无限标签空间的分析仍以维度上界为依据，实际实现仍待研究。

---

## 9. Epistemic Throughput: Fundamental Limits of Attention-Constrained Inference

**arXiv ID:** 2602.09127 | [PDF](https://arxiv.org/pdf/2602.09127v1)

**作者:** Lei You `[一作]` (Technical University of Denmark), Lei You `[通讯]` (Technical University of Denmark)

**通讯引用:** 768 | [OpenAlex ID](https://openalex.org/A5082049111)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文建立了“注意力受限推理（ACI）”框架，刻画了在一次性检索-验证场景下，低成本筛选与高成本验证如何共同决定推断不确定度的降低；

**💡 创新点**

核心创新是提出 JaKoB 伸缩定律，表明经验信息通过筛选提升可实现的“认知吞吐量”在最优情况下呈 √(JKB) 的非线性放大；

**🔧 技术方法**

方法上结合信息理论（对数损失、互信息、数据处理不等式）、极值理论（尾均值、极值分布）以及顺序统计学来推导上界与下界；

**📊 数据集**

研究为理论性，未使用具体公开数据集；在附录给出了可解析的基准模型以验证公式的正确性；

**📈 对比分析**

通过与基于得分的验证策略对比，证明在弱筛选极限下该策略可近似达到上界，实验证明其信息增益符合 √(JKB) 的增长；

**⚠️ 局限性**

局限包括：假设记录独立、单窗口、对数损失最优、未考虑顺序决策或多级验证成本、轻尾得分分布下提升受限。

---

## 10. Fair Feature Importance Scores via Feature Occlusion and Permutation

**arXiv ID:** 2602.09196 | [PDF](https://arxiv.org/pdf/2602.09196v1)

**作者:** Camille Little `[一作]` (Rice University), Genevera Allen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出两种模型无关的特征公平重要性度量，评估单个特征对模型公平性的影响。

**💡 创新点**

创新点在于同时提供基于置换与基于遮蔽的两种直观计算方式，并利用 minipatch 学习极大简化遮蔽度量的计算成本。

**🔧 技术方法**

技术包括特征置换、留一法遮蔽、minipatch 学习、随机森林、对公平度量（如人口统计平等）与准确度度量（误差、MSE）进行评估。

**📊 数据集**

使用合成数据（1000样本，10特征）以及 UCI 实际数据集：Adult Income（96特征，45k样本）和 German Credit（56特征，1k样本），性别为受保护属性。

**📈 对比分析**

与传统性能特征重要性（如随机森林 Gini）对比，置换与遮蔽度量均能准确识别与敏感属性相关的偏差特征，并揭示公平-准确性权衡；在实际数据上，模型准确率约 0.84（Adult）和 0.80（German），公平度量分别为 0.85 和 0.94。

**⚠️ 局限性**

局限包括：只考虑单一特征的贡献，未探索特征交互对公平性的影响；置换法在高维高相关性场景下可能低估重要性；minipatch 方案对样本量和特征数量敏感，需进一步理论与实践验证。

---

## 11. A Theory for Probabilistic Polynomial-Time Reasoning

**arXiv ID:** 2602.09302 | [PDF](https://arxiv.org/pdf/2602.09302v1)

**作者:** Lijie Chen `[一作]` (University of California), Ryan Williams `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 5565 | [OpenAlex ID](https://openalex.org/A5063871533)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

提出了一种新的可界算术理论 _1（或 APX1），用于形式化理论计算机科学中广泛使用的概率性论证，并证明其在证明强度上严格弱于之前的框架（如 Jeřábek 的 _1）。

**💡 创新点**

创新点在于：①将近似计数直接作为公理化的核心原语；②提出了四条极简且足够强的公理（Basic、Boundary、Precision Consistency 与 Local Consistency）；③证明 _1 在捕捉概率工具（如期望、Markov/Chebyshev、不等式、误差降低、Chernoff 上限）方面具有完整性，并且能够在弱理论中完成若干重要的复杂度下界（如 Parity 的平均/最坏情况 ^0 下界）。

**🔧 技术方法**

使用的技术包括：1）基于近似计数的概率计算与局部一致性推导；2）“逐点到整体”点推手法（bit‑by‑bit fixing）；3）构造可证明的随机变量和期望；4）针对可行搜索问题的 Witnessing Theorem；5）利用弱鸽巢原理和逆数学框架来研究随机化下界；6）在证明中大量使用量化子自由的可归约性与迭代归纳。

**📊 数据集**

数据集：无，本文为纯理论性工作，未使用任何实验数据或机器学习数据集。

**📈 对比分析**

与以往理论的比较：相较于 Jeřábek 的 _1，_1 在证明强度上更弱，但已足以形式化包括布鲁姆‑刘布‑鲁比德线性测试、Schwartz‑Zippel 引理、Parity 的 ^0 下界以及相关错误降低等多项非平凡结果；并在可行性证明层面提供了更精细的结构；在性能方面（理论复杂度）证明了所有构造均可在多项式时间内完成，并在证明长度与算术复杂度上实现了显著简化。

**⚠️ 局限性**

局限性：①仍需依赖近似计数的公理化，若要证明更强的浓度不等式（如完整的 Chernoff 上限）可能需要更强的公理；②对某些下界结果（如某些深层的 circuit lower bound）尚未在 _1 内完全实现，可能需要进一步的扩展；③该理论在处理非可算取值集合（如无限集合）时不适用；④对实际可执行的程序验证缺乏直接工具，需结合外部证明助手进一步实现。

---

## 12. Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning

**arXiv ID:** 2602.09229 | [PDF](https://arxiv.org/pdf/2602.09229v1)

**作者:** Xincan Feng `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1623 | [OpenAlex ID](https://openalex.org/A5102396915)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究了对比学习中嵌入向量幅度是否携带任务相关信息，提出 2×2 归一化消融框架并在检索、RAG、STS、CLIP 等任务上实验验证。

**💡 创新点**

发现文档幅度与相关性高度相关，查询幅度影响训练动态；提出任务对称性原则：幅度学习对不对称任务有利，对称任务有害；提出可学习归一化参数自动选择最优策略。

**🔧 技术方法**

采用 InfoNCE 对比学习、余弦/点积相似度、归一化/非归一化变体、可学习归一化，以及 NDCG@10、R@1、STS 相关性、CLIP 对齐等评估指标。

**📊 数据集**

使用 MS MARCO 训练集进行微调，评估基于 TREC‑DL 2019/2020、BEIR 14 领域、BRIGHT 12 逻辑推理、Multi‑hop 4 复合问答；RAG 结合 Flan‑T5‑Large 读者；CLIP 预训练在 MS‑COCO 上进行验证。

**📈 对比分析**

与余弦相似度基线对比，点积和部分归一化在检索上提升 3–6% NDCG，QNorm 在 Contriever 上提升 7–13% 召回；在 RAG 上 QNorm 提升 5–8% EM；在 STS 与 CLIP 上归一化导致性能下降 40–45 分，验证了任务对称性原则。

**⚠️ 局限性**

局限性包括：需要预训练模型，某些架构（如 E5）需移除内置归一化才能学习幅度；部分归一化破坏对称性，无法用于双向相似度任务；跨模态验证仍有限，主要聚焦文本检索。

---

## 13. Non-existence of Information-Geometric Fermat Structures: Violation of Dual Lattice Consistency in Statistical Manifolds with $L^n$ Structure

**arXiv ID:** 2602.09028 | [PDF](https://arxiv.org/pdf/2602.09028v1)

**作者:** Kanta Tochigi `[一作]` `[通讯]` (Independent Researcher), Kanta Tochigi (Independent Researcher)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将费马大定理重新表述为信息几何嵌入问题，并证明在 n≥3 时不存在满足“信息几何费马解”的整数格点，从而提供一种新的理论证明。

**💡 创新点**

创新点在于将费马方程转化为 Lⁿ 矩约束的统计流形，结合最大熵原理、信息几何的对偶坐标与泊松求和公式，以及 Hausdorff‑Young 不等式，揭示对偶格点一致性只能在 n=2 时成立，进而解释 n≥3 无非平凡整数解的根本原因。

**🔧 技术方法**

使用技术包括：信息几何（Fisher 信息度量、对偶坐标、Legendre 变换）、最大熵原理构造指数族、泊松求和公式、Hausdorff‑Young 不等式、ζ 函数与 Mellin 变换等。

**📊 数据集**

未使用实验数据集，全部为纯理论推导。

**📈 对比分析**

无实验比较与性能评估，结论基于解析证明而非数值实验。

**⚠️ 局限性**

局限性：对偶格点一致性的假设尚未从信息几何的第一性原理直接推导；方法主要针对整数格点模型，未给出对代数结构的完整对应；仅证明 n=2 具备自对偶性，对更广泛的非整数维度或其他约束的适用性尚需进一步研究。

---

## 14. X-Mark: Saliency-Guided Robust Dataset Ownership Verification for Medical Imaging

**arXiv ID:** 2602.09284 | [PDF](https://arxiv.org/pdf/2602.09284v1)

**作者:** Pranav Kulkarni `[一作]` (University of Maryland), Heng Huang `[通讯]` (University of Maryland)

**通讯引用:** 24768 | [OpenAlex ID](https://openalex.org/A5060016795)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种样本特定、无标签的后门水印方法 X-Mark，用于胸部 X 光图像数据集的版权保护。

**💡 创新点**

创新点包括：①利用条件 U‑Net 与 EigenCAM 对显著区域生成个性化扰动；②在训练目标中加入拉普拉斯金字塔正则化，使水印保持尺度不变；③实现既能保持诊断质量又能保持视觉不可见的水印。

**🔧 技术方法**

使用的技术包括：条件 U‑Net、EigenCAM 显著性映射、LPIPS 感知相似度损失、拉普拉斯金字塔损失、对数似然等多目标训练；黑盒假设检验用于水印验证。

**📊 数据集**

使用的主要数据集为 CheXpert 胸部 X 光图像数据集。

**📈 对比分析**

与多种 Poisson‑label 与 Clean‑label 基线（BadNets、Blended、WaNet、UBW‑P、Label‑Consistent、SSCL‑BW）进行对比。X-Mark 在 BA 84.88%、WSR 100% 以及 LPIPS 0.020 上优于基线；在恶意模型验证场景下 ΔP>0.8、p<0.001；在 Ind‑M 场景中误报率降低 12%。

**⚠️ 局限性**

局限性：在 Ind‑M 场景下仍存在误报；对抗性攻击（如迁移性、裁剪）需进一步评估；目前仅针对分类任务，尚未扩展到分割或其它医学任务。

---

## 15. From Legible to Inscrutable Trajectories: (Il)legible Motion Planning Accounting for Multiple Observers

**arXiv ID:** 2602.09227 | [PDF](https://arxiv.org/pdf/2602.09227v1)

**作者:** Ananya Yammanuru `[一作]` (University of Illinois), Katherine Driggs-Campbell `[通讯]` (University of Illinois)

**通讯引用:** 1449 | [OpenAlex ID](https://openalex.org/A5059066811)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对多观察者环境，提出一种既能对正向观察者产生可读轨迹、又能对负向观察者产生不可读轨迹的运动规划问题，并给出求解框架。

**💡 创新点**

创新点在于：①将观察者的可视范围与动机（正负）统一纳入轨迹优化目标；②设计了可同时优化可读性与不可读性的成本函数；③提出了两种对负向观察者的策略（“假目标”与“回避”），并在同一框架内调节。

**🔧 技术方法**

使用基于 STOMP 的随机轨迹优化；成本函数包含正向观察者的可读性分数和负向观察者的不可读性分数（可读性/不可读性与可视区域重叠部分加权）；采用贝叶斯概率模型评估目标推断。

**📊 数据集**

实验使用人工生成的二维平面环境，随机放置若干观察者、可视区域和目标，未使用公开数据集。

**📈 对比分析**

与三类基线方法比较：最优效率轨迹、最大可读性轨迹和最大不可读性轨迹。评估指标包括：每个观察者首次正确预测的时间（越早越好/越晚越好）、正确预测占比、可读性/不可读性分数。实验结果表明，在有限可视范围和混合动机场景下，所提出的方法能够在保持相对高效的同时，实现对正向观察者的快速可读性和对负向观察者的高不可读性，整体性能优于单纯优化效率或可读性的基线。

**⚠️ 局限性**

限制：①观测者为静态、已知可视区域且数量有限；②环境无障碍物，未考虑碰撞约束；③轨迹规划为离线静态，无法处理动态观测者或实时变化；④未对观测者模型进行学习或适应；⑤未使用真实世界数据或复杂任务场景验证。

---

## 16. Contractual Deepfakes: Can Large Language Models Generate Contracts?

**arXiv ID:** 2602.09384 | [PDF](https://arxiv.org/pdf/2602.09384v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 17. Effective MoE-based LLM Compression by Exploiting Heterogeneous Inter-Group Experts Routing Frequency and Information Density

**arXiv ID:** 2602.09316 | [PDF](https://arxiv.org/pdf/2602.09316v1)

**作者:** Zhendong Mi `[一作]` (Stevens Institute of Technology), Shaoyi Huang `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 369 | [OpenAlex ID](https://openalex.org/A5073345631)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对MoE大语言模型的压缩框架RFID-MoE，利用专家路由频率和信息密度进行动态秩分配，并通过稀疏投影重构残差；

**💡 创新点**

创新点在于同时考虑路由频率和有效秩的融合评估专家重要性，实现非均匀秩分配；并引入参数高效的稀疏正交投影方法恢复低秩近似残差；

**🔧 技术方法**

技术包括SVD分解、有效秩（spectral entropy）计算、路由频率统计、秩分配融合、稀疏投影残差重构和无监督后训练；

**📊 数据集**

使用的主要数据集为WikiText-2、PTB、C4作为校准集，并在多款MoE LLM（Qwen3‑30B、DeepSeek‑MoE‑16B、Qwen2‑57B、Deepseek‑V2‑Lite‑Chat等）上评估；

**📈 对比分析**

与NAEE、MoE‑I^2、RS‑MoE、D^2‑MoE、MoBE等基线对比，RFID‑MoE在60%压缩下PPL可比原模型，零样本推理任务准确率提升约8%；

**⚠️ 局限性**

局限性包括需要额外的校准数据收集路由频率，超参数ξ对结果影响较大，且在极端高压缩率或非常小模型上可能效果有限。

---

## 18. SpinCastML an Open Decision-Making Application for Inverse Design of Electrospinning Manufacturing: A Machine Learning, Optimal Sampling and Inverse Monte Carlo Approach

**arXiv ID:** 2602.09120 | [PDF](https://arxiv.org/pdf/2602.09120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 19. Uncertainty-Aware Multimodal Emotion Recognition through Dirichlet Parameterization

**arXiv ID:** 2602.09121 | [PDF](https://arxiv.org/pdf/2602.09121v1)

**作者:** Rémi Grzeczkowicz `[一作]` (Kaliber Labs), Aneesh Jonelagadda `[通讯]` (Kaliber Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在边缘设备上实现了一套轻量、隐私友好的多模态情感识别框架，融合语音、文本与面部图像三种模态。

**💡 创新点**

创新点在于：①基于 Dempster–Shafer 理论与 Dirichlet 证据的无监督、无额外训练的融合机制；②直接在模型 logits 上进行不确定度建模，避免 softmax 带来的信息损失；③可扩展至任意数量模态、任意任务，具有极强的通用性。

**🔧 技术方法**

技术细节包括：Emotion2Vec（语音）、DistilRoBERTa（文本）和 ResEmotNet（面部）预训练 backbone；对 logits 做最小值平移后转化为 Dirichlet 证据；使用 Dempster–Shafer 合并证据；最终得到带不确定度的概率输出。

**📊 数据集**

使用了五个公开基准数据集：eNTERFACE05、MEAD、MELD、RAVDESS、CREMA‑D，对比评估。

**📈 对比分析**

与 Wang 等人提出的“基本缓解”方法和大型 Emotion‑Llama 模型对比，所提方法在中性容忍度（neutral‑tolerant）下 71%–90% 的准确率，整体优于基线；与 Emotion‑Llama 相比，参数量约 26 倍更小，推理速度约 10 倍快。

**⚠️ 局限性**

局限性包括：①仅在五个数据集上评估，缺乏更大规模或更公平的跨模型对比；②未处理时间序列信息，无法实现连续情绪追踪；③对未见情绪（如 contempt）依赖于中性作为 fallback，仍有改进空间；④未加入生理或环境等更多模态；⑤缺乏个性化校准或适应机制。

---

## 20. Harvest: Adaptive Photonic Switching Schedules for Collective Communication in Scale-up Domains

**arXiv ID:** 2602.09188 | [PDF](https://arxiv.org/pdf/2602.09188v1)

**作者:** Mahir Rahman `[一作]` (Purdue University), Vamsi Addanki `[通讯]` (Purdue University)

**通讯引用:** 124 | [OpenAlex ID](https://openalex.org/A5015398112)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种系统化方法，针对光学互连中集体通信的阶段化通信模式，合成可重配置的拓扑调度，以最小化总完成时间；

**💡 创新点**

创新点在于将最大并发流与 α–β 成本模型结合，形成一个同时考虑重配置延迟、拥塞和传播延迟的完整优化框架；并针对递归倍增 AllReduce 通过分析其结构，提供了 O((log n)^4) 的多项式时间最优调度算法；

**🔧 技术方法**

采用动态规划+混合整数二阶锥规划（MISOCP）求解子问题，利用最大并发流理论、Birkhoff–von Neumann 分解以及光学互连的传输特性；

**📊 数据集**

实验使用 Astra‑Sim 的包级仿真、数值优化（Gurobi 求解器）以及在 8‑GPU NVIDIA BlueField‑3 服务器上进行硬件仿真；

**📈 对比分析**

与传统的静态拓扑（环、Torus、Kautz 等）和每步重配置的 BvN 调度进行对比，结果显示在重配置延迟低于 1 µs 时可实现 6.4× 的 AllReduce 加速；在重配置延迟较大时仍可比 BvN 获得 3–4× 的提升，且整体性能优于两种极端方案；

**⚠️ 局限性**

主要限制在于子问题求解的计算开销（MISOCP 对大规模节点数的可扩展性受限），以及对光学互连硬件重配置延迟的准确建模需要依赖具体技术参数。

---

## 21. STaR: Scalable Task-Conditioned Retrieval for Long-Horizon Multimodal Robot Memory

**arXiv ID:** 2602.09255 | [PDF](https://arxiv.org/pdf/2602.09255v1)

**作者:** Mingfeng Yuan `[一作]` (University of Toronto), Steven L. Waslander `[通讯]` (University of Toronto)

**通讯引用:** 9992 | [OpenAlex ID](https://openalex.org/A5024242059)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出STaR框架，构建任务无关的多模态长时记忆，并实现可扩展的任务条件检索与上下文推理，支持机器人长周期任务执行。

**💡 创新点**

①通过信息瓶颈（IB）聚合3D原语形成紧凑、无冗余的检索子集；②引入视频字幕诱导原语子集，保证任务相关性；③实现Agentic RAG工作流，实现查询到动作的闭环。

**🔧 技术方法**

多模态感知堆栈（Segment Anything/CLIP/TAP）、NVILA视频字幕、Milvus向量检索、信息瓶颈聚类、LLM（ChatGPT‑4.1‑mini）进行检索与推理。

**📊 数据集**

NaVQA（校园室内外混合场景）与新构建的WH‑VQA（仓库仿真环境），均含长时序视觉与LiDAR数据。

**📈 对比分析**

相较于ReMEmbR与OpenGraph基线，STaR在空间、时间、文本以及多模态任务上均提升成功率（约1.5-2倍），空间误差下降约50-70%，时间误差保持低于2分钟，召回率明显提升。

**⚠️ 局限性**

依赖预先探索的环境；对实时动态环境适应性有限；对极长时间（数小时）记忆存储与检索仍需优化；需人工或外部SLAM获取位姿，整体系统复杂度较高。

---

## 22. Chaos in Autobidding Auctions

**arXiv ID:** 2602.09118 | [PDF](https://arxiv.org/pdf/2602.09118v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Kelly Spendlove `[通讯]` (Google)

**通讯引用:** 108 | [OpenAlex ID](https://openalex.org/A5033899886)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究基于回报率约束的自动竞价系统在连续和离散时间下的动力学行为，证明其可呈现正式混沌现象并通过模拟已知混沌系统（如丘阿电路）与经典映射（Ricker模型、Logistic映射）来揭示复杂动态

**💡 创新点**

首次将混沌动力学框架引入自动竞价领域，展示即使是最简单的价值最大化竞争也能产生不可预测、极度敏感的长期行为；同时提出连续否定和非线性项构造的通用模拟技术

**🔧 技术方法**

连续时间上利用可微分的竞价动态与构造的连续否定、非线性实现模块进行系统模拟；离散时间上采用镜像下降（entropic和Euclidean正则化）更新公式并与经典映射建立对应关系

**📊 数据集**

本研究为理论分析，不依赖任何实测数据集，而是通过数值仿真（Lyapunov指数、拓扑熵、分岔图）验证混沌性质

**📈 对比分析**

通过对比已知混沌系统的理论特征（正Lyapunov指数、正拓扑熵、Li‑Yorke周期三周期）与模拟结果，证明自动竞价动力学在适当参数下与这些系统具有相同的混沌表现；性能上表现为轨迹快速发散，长期预测不可行

**⚠️ 局限性**

局限性包括：仅覆盖特定类的非线性系统（如每个非线性项仅依赖自身变量），离散时间混沌依赖较大学习率，连续时间模拟需要高λ参数，且无法处理耦合非线性（如Lorenz系统）

---

## 23. Entropy-Based Evidence for Bitcoin's Discrete Time Mechanism

**arXiv ID:** 2602.09027 | [PDF](https://arxiv.org/pdf/2602.09027v1)

**作者:** Bin Chen `[一作]` (Shenzhen University), Pan Feng `[通讯]` (Shenzhen University)

**通讯引用:** 24680 | [OpenAlex ID](https://openalex.org/A5055477551)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对2016-2024年比特币区块到达时间的实证分析，探讨了概率挖矿如何产生可验证的非连续时间；

**💡 创新点**

创新点在于将熵坍塌（entropy collapse）与块生成的指数等待时间结合，给出比特币时间结构的机制化解释，并揭示分布式网络中熵坍塌完成的传播受限性质；

**🔧 技术方法**

采用统计学方法（指数拟合、协方差自相关、熵计算、存活函数）和概率论模型（伯努利采样、泊松极限定理、难度反馈机制）；

**📊 数据集**

使用约425,000个真实区块到达时间记录、2016-2024年的难度周期数据以及公开的分叉（fork）监测数据；

**📈 对比分析**

通过与理论指数分布、epoch级别平均到达率、并行矿工独立性检验相比较，结果显示区块间隔近似泊松过程，epoch级别平均值稳定在600±100秒，分叉持续时间随网络传播速度提升而明显缩短；

**⚠️ 局限性**

局限在于未对节点时钟偏差、区块时间戳误差及潜在的网络攻击做深入建模，且仅基于历史数据，未来网络变化可能导致模型失效。

---

## 24. XMap: Fast Internet-wide IPv4 and IPv6 Network Scanner

**arXiv ID:** 2602.09333 | [PDF](https://arxiv.org/pdf/2602.09333v1)

**作者:** Xiang Li `[一作]` (Nankai University), Zheli Liu `[通讯]` (Nankai University)

**通讯引用:** 4897 | [OpenAlex ID](https://openalex.org/A5060212061)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

设计并实现了一个高速、模块化的全网 IPv4/IPv6 网络扫描工具 XMap，支持任意子网随机扫描、多协议（ICMP、TCP SYN、UDP、DNS 等）和多端口并发扫描。

**💡 创新点**

创新点包括：
- 采用异步解耦的发送/接收机制与完整地址随机化算法，显著提升扫描速度；
- 支持任意子网位段的随机化扫描，突破传统工具只能按前缀扫描的局限；
- 引入伪造、状态化及地址重放的 DNS/ICMP 扫描模块，实现更丰富的测量和漏洞发现；
- 与旧版 ZMap 兼容并大幅扩展功能，形成完整可定制的扫描框架。

**🔧 技术方法**

使用技术包括：异步 I/O 与事件驱动、PF_RING、GMP 及 permutation modulo multiplication 地址随机化、C/C++ 高性能网络 I/O、Linux/macOS/BSD 多系统支持、数据库/文件输出、多线程并发、Docker 镜像打包。

**📊 数据集**

主要使用公开的 IPv4/IPv6 地址空间作为扫描目标；未显式引用专门的训练或标注数据集，扫描结果可用于后续安全评估与研究。

**📈 对比分析**

与 ZMap、Masscan 等现有工具在 32 位地址空间扫描时间对比：在 10 GbE + PF_RING 环境下可在 5 分钟内完成，单纯的 45 分钟扫描在 100 Mbps 下可实现；IPv6 periphery 探测速度也显著快于传统方法，整体性能提升约 10‑30 倍。

**⚠️ 局限性**

限制：
- 仍受网络带宽和目标网络的防火墙/速率限制影响；
- 随机化扫描可能导致部分稀疏或隐蔽主机被遗漏；
- 对某些特殊协议或非标准端口支持有限；
- 需要在合规的测试环境下使用，避免法律与隐私风险。

---

## 25. Barycentric alignment for instance-level comparison of neural representations

**arXiv ID:** 2602.09225 | [PDF](https://arxiv.org/pdf/2602.09225v1)

**作者:** Shreya Saha `[一作]`, Meenakshi Khosla `[通讯]` (School of ZZZ)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供了ICML 2026论文提交与格式化指南

**💡 创新点**

提出双盲审稿流程、严格页面与文件大小限制，并给出详细排版规范

**🔧 技术方法**

使用PDF、Type‑1字体、LaTeX模板与相关宏包（如algorithm, amsmath 等）

**📊 数据集**

无具体实验数据集，仅给出提交模板

**📈 对比分析**

没有实验比较，主要通过规范化格式来提升审稿效率

**⚠️ 局限性**

对排版与文件格式要求严格，可能限制作者使用其他工具或自由表达

---

## 26. Benchmarking Knowledge-Extraction Attack and Defense on Retrieval-Augmented Generation

**arXiv ID:** 2602.09319 | [PDF](https://arxiv.org/pdf/2602.09319v1)

**作者:** Zhisheng Qi `[一作]` (University of Oregon), Yu Wang `[通讯]` (University of Oregon)

**通讯引用:** 34819 | [OpenAlex ID](https://openalex.org/A5107306837)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了面向检索增强生成（RAG）系统的知识提取攻击与防御综合基准，统一设计空间并构建可复现评估框架。

**💡 创新点**

创新点在于将攻击/防御策略、检索嵌入、生成模型、知识库构建和评估协议统一起来，并通过系统化实验揭示不同阶段的漏洞与防御互补性。

**🔧 技术方法**

使用了检索嵌入模型（MiniLM、GTE、BGE）、多种生成模型（LLaMA、Qwen、GPT‑4）、多种攻击策略（随机、DGEA、IKEA 等）以及阈值过滤、系统/查询阻断和摘要防御等技术。

**📊 数据集**

数据集包括医疗问答（HealthCareMagic）、企业邮件（Enron）、小说文本（HarryPotter）和百科内容（Pokémon）等四类知识库。

**📈 对比分析**

通过 EE^R、EE^G、EE、ASR 等统一指标在四个数据集上评估，实验显示 DGEA 在无防御时取得最高检索覆盖率，阈值防御能显著抑制检索泄漏，摘要与系统阻断在生成阶段最有效；但不同模型间的迁移性能差异大。

**⚠️ 局限性**

局限在于仅覆盖单一 RAG 结构，缺乏对多轮代理式 RAG 的评估，防御多阶段协同机制尚未充分探索，且跨嵌入模型的攻击迁移能力有限。

---

## 27. PICASSO: Scaling CHERI Use-After-Free Protection to Millions of Allocations using Colored Capabilities

**arXiv ID:** 2602.09131 | [PDF](https://arxiv.org/pdf/2602.09131v1)

**作者:** Merve Gülmez `[一作]` (Ericsson Security Research), Thomas Nyman `[通讯]` (Ericsson Product Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 CHERI ISA 上实现了一种新的时序内存安全机制——Colored Capabilities（色彩能力），通过在能力模型中加入有限的间接指针表（PVT）实现对同一分配的所有能力批量撤销，从而避免了传统的内存隔离与回收延迟。

**💡 创新点**

创新点在于：① 为 CHERI 只在需要追踪的能力上引入可批量撤销的“颜色”字段；② 通过硬件管理的 PVT/ PVB 只在能力访问时做一次查表，显著减少了撤销扫描频率和暂停时间；③ 支持超过两百万个唯一分配 ID，实现大规模分配的安全性与性能兼顾。

**🔧 技术方法**

技术实现包括：① 在 CHERI‑RISC‑V 处理器（CHERI‑Toooba）上添加 PVT、PVB 相关指令与 CSR；② 在软件栈中扩展 CheriBSD 的内存分配器（mrs）和 Unr 计数器；③ 在 Clang/LLVM 工具链中实现彩色能力的编译器支持；④ 在 QEMU 与 FPGA 上分别部署完整系统验证。

**📊 数据集**

使用的数据集：NIST Juliet 测试套件（CWE‑416/415），SPEC CPU2006（单/双核），SQLite 3.22.0 speedtest1，PostgreSQL 10+，gRPC 1.54.2 QPS，及若干行业基准如 omnetpp、bzip2、h264ref 等。

**📈 对比分析**

对比方法：与 Cornucopia 及 Cornucopia Reloaded 进行对比，采用相同的分配器接口；在 SPEC CPU 上度量 CPU 周期与内存占用，结果为 1.05× CPU 负载、1.08× 内存；在 SQLite/PGSQL/gRPC 等时序敏感工作负载中，撤销次数大幅下降（从数百次降至 0 次或 1 次），并保持 3–8% 的性能开销，显著优于传统方案。

**⚠️ 局限性**

局限性：① 只能提供堆分配的时序安全，栈上用后返回不受保护；② PVT 需要 256 KiB（或更多）物理存储，对极小的嵌入式系统不友好；③ 仍需周期性撤销扫描，极端大规模程序或频繁分配时仍可能产生 1–2% 的额外延迟；④ 需要硬件改造，部署成本相对较高。

---

## 28. "These cameras are just like the Eye of Sauron": A Sociotechnical Threat Model for AI-Driven Smart Home Devices as Perceived by UK-Based Domestic Workers

**arXiv ID:** 2602.09239 | [PDF](https://arxiv.org/pdf/2602.09239v1)

**作者:** Shijing He `[一作]` (King's College London), Jose Such `[通讯]` (INGENIO)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过半结构化访谈与通信隐私管理（CPM）理论，研究英国雇佣家庭工在雇主家与自家中使用AI驱动的智能家居设备时的隐私边界、风险与对策，并提出社会技术威胁模型

**💡 创新点**

首次将CPM框架与AI智能家居功能结合，识别雇主与国内工代理机构为关键威胁主体，揭示跨屋场所的AI行为分析与残留数据导致的持续隐私风险；提出基于多情境的威胁模型

**🔧 技术方法**

使用CPM理论、半结构化访谈、访谈文本的主题编码与归纳分析方法

**📊 数据集**

18名英国雇佣家庭工（包括生活型与外出型）的访谈记录（共约22小时录音）

**📈 对比分析**

采用定性编码与主题归纳进行结果评估，并与现有文献中的威胁模型对比，说明研究填补了缺口；未涉及量化性能指标

**⚠️ 局限性**

样本规模有限、仅涵盖英语受访者、未采访雇主或代理机构、未直接检测设备硬件与设置、依赖受访者自我报告，结果可能受社会期望偏差影响

---

## 29. Bridging the Modality Gap in Roadside LiDAR: A Training-Free Vision-Language Model Framework for Vehicle Classification

**arXiv ID:** 2602.09425 | [PDF](https://arxiv.org/pdf/2602.09425v1)

**作者:** Yiqiao Li `[一作]` (City University of New York), Jie Wei `[通讯]` (City University of New York)

**通讯引用:** 1567 | [OpenAlex ID](https://openalex.org/A5081204417)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种训练无关的视听语言模型框架，通过深度感知图像生成将稀疏路侧LiDAR点云转化为可用于VLM的视觉代理，实现细粒度卡车分类。

**💡 创新点**

创新点在于设计了深度感知图像生成管线以桥接模态差距，并发现文本先验在极低样本下能作为“语义锚”，同时提供了从VLM冷启动到轻量监督模型的自标注流程。

**🔧 技术方法**

采用CLIP、EVA等预训练视觉语言模型、基于时间序列配准与形态学重建的图像生成、文本嵌入融合、原型推理等技术。

**📊 数据集**

使用真实道路侧LiDAR数据集，包含20个细粒度车辆类别，共约数千帧点云，配合摄像头标注。

**📈 对比分析**

与全监督ViT-B/16、PointNet以及CLIP线性探测器对比，VLM无训练方法在30-shot时F1≈0.62，略低于线性探测器0.649，但远优于监督基线，且在1–4-shot时表现更稳健。

**⚠️ 局限性**

局限包括对半壳扫描导致的几何歧义仍难以完全解决、文本先验在中大样本下导致性能下降，以及对超低样本仅提供了相对稳定但非最优的结果。

---

## 30. The Coordination Criterion

**arXiv ID:** 2602.09435 | [PDF](https://arxiv.org/pdf/2602.09435v1)

**作者:** Joseph M. Hellerstein `[一作]` `[通讯]` (University of California Berkeley and Amazon Web Services), Joseph M. Hellerstein (University of California Berkeley and Amazon Web Services)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

在异步消息传递模型下提出“协调准则”，通过对规范的可观测结果进行单调性检验，判定分布式规范是否需要协调实现；

**💡 创新点**

创新点在于将协调需求归结为规范自身的语义单调性，而非特定协议或一致性模型，形成一个统一、最小假设下的诊断框架；

**🔧 技术方法**

采用Lamport历史、可能性/可观测性函数以及结果偏序的语义工具，证明单调性是实现无协调实现的必要且充分条件；

**📊 数据集**

无（本研究为理论性工作，无使用实验数据集）；

**📈 对比分析**

无（未做实验或性能比较，理论上给出严格的必要与充分条件）；

**⚠️ 局限性**

限制在于仅关注安全类语义（可观测性）而不涉及活性约束或进展保证，并假设无公平或进度约束，难以直接应用于需实时或高可用性评估的实际系统。

---

## 31. Scaling GraphLLM with Bilevel-Optimized Sparse Querying

**arXiv ID:** 2602.09038 | [PDF](https://arxiv.org/pdf/2602.09038v1)

**作者:** Yangzhe Peng `[一作]` (Huazhong University of Science and Technology), Kun He `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4898 | [OpenAlex ID](https://openalex.org/A5033526822)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种通过双层优化实现稀疏查询的框架（BOSQ），在文本属性图（TAG）上利用大型语言模型（LLM）生成解释特征，仅为少数重要节点调用LLM，以显著降低计算成本。

**💡 创新点**

创新点在于：①将节点选择问题建模为双层优化，通过内层训练GNN、外层优化节点重要性评分；②采用连续松弛与Top‑K稀疏化，再用Straight‑Through Estimator实现真正的硬门控选择；③通过验证集的超梯度直接驱动选择，使得查询预算与任务性能最优匹配。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）作为解释器、预训练语言模型（LM）编码节点文本、图神经网络（GNN）、双层优化（hypergradient + Implicit Function Theorem）、Top‑K稀疏化、Straight‑Through Estimator、温度退火等。

**📊 数据集**

实验数据集：六个现实文本属性图（Photo、Instagram、Computer、User‑ltv、Item‑ltv、Post‑votes）以及OGB的ogbn‑products（240+万节点）。

**📈 对比分析**

与传统GNN、PLM、GraphLLM（TAPE、ENGINE、LLaGA、LLMEmb、LLMExpl、LLMPred）等方法对比，BOSQ 在所有任务上实现了 100‑200 倍的速度提升，同时保持或超过现有最优准确率；在百万级别图上可在 7.4 小时内完成训练，远优于其它 GraphLLM 方法。

**⚠️ 局限性**

局限性：仍需调用昂贵的LLM，且查询预算 K 的选择依赖实验调参；双层优化与超梯度计算在大图上可能产生较高内存/时间开销；当前仅验证于节点级任务，未知对更复杂图任务（如子图分类、图生成）的适用性。

---

## 32. Finite-time Stable Pose Estimation on TSE(3) using Point Cloud and Velocity Sensors

**arXiv ID:** 2602.09414 | [PDF](https://arxiv.org/pdf/2602.09414v1)

**作者:** Nazanin S. Hashkavaei `[一作]` (Syracuse University), Amit K. Sanyal `[通讯]` (Syracuse University)

**通讯引用:** 3514 | [OpenAlex ID](https://openalex.org/A5079524565)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个基于点云和角速度的有限时间稳定姿态估计器（FTS‑PE），可在没有转速测量、无动力学模型、无姿态解耦的条件下估计刚体的姿态、位置和速度。

**💡 创新点**

创新点包括：
• 直接在 SE(3) Lie 群上构造无模型的姿态估计器，避免了坐标奇异和摆动现象；
• 采用 Morse‑Lyapunov 函数和 Hölder 连续的控制律，证明了几乎全局的有限时间稳定性；
• 对角速度和位移测量噪声给出鲁棒性分析，证明误差在有噪声时收敛到有限邻域；
• 采用几何变分积分离散化方法，保持状态空间的几何结构。

**🔧 技术方法**

使用的技术包括：
• SE(3) Lie 群的几何表示与运算；
• Wahba 问题的加权形式；
• Morse‑Lyapunov 能量函数与 Hölder 连续滑模控制；
• 变分原理与 Rayleigh‑耗散项；
• 几何变分积分离散化；
• 过滤器实现无转速测量时的加速度估计。

**📊 数据集**

数据集：
• 在仿真中使用随机生成的运动轨迹，并加入高斯噪声（角速度 9.17°/s，位移 0.02 m/s）；
• 在实验中使用 ZED‑2i 立体深度相机点云与内置 IMU 的角速度测量。

**📈 对比分析**

比较方法与性能：
• 与 VPE（变分姿态估计）和 DQ‑MEKF（双四元数扩展卡尔曼滤波）在相同噪声下进行同样的仿真；
• FTS‑PE 在 30 s 仿真中实现了最快的收敛速度（运行时间 0.0469 s，低于 DQ‑MEKF 0.125 s 和 VPE 0.1094 s）；
• 角度误差 RMS 为 0.3544 rad，位置误差 RMS 为 0.2769 m，均低于两者；
• 在实验中误差在短时间内收敛到几乎零的邻域。

**⚠️ 局限性**

局限性：
• 需要至少两个非共线点云向量；
• 对测量噪声的有界性做了假设；
• 无转速测量时依赖额外的非线性滤波器；
• 离散化实现需要几何变分积分，计算量相对较大；
• 目前未在多速率或极低采样率条件下验证；
• 对极端动态变化（如快速抖动）时的性能未作系统评估。

---

## 33. SnareNet: Flexible Repair Layers for Neural Networks with Hard Constraints

**arXiv ID:** 2602.09317 | [PDF](https://arxiv.org/pdf/2602.09317v1)

**作者:** Ya-Chi Chu `[一作]` (Stanford University), Madeleine Udell `[通讯]` (Stanford University)

**通讯引用:** 2003 | [OpenAlex ID](https://openalex.org/A5084564811)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了 SnareNet，通过可微修复层和自适应松弛实现输入依赖的非线性约束严格满足，并在优化学习与控制策略上进行验证。

**💡 创新点**

核心创新包括：可微修复层、Levenberg–Marquardt 正则化的 Newton 更新、可调容差的约束控制以及自适应松弛训练策略。

**🔧 技术方法**

使用了可微修复层、Jacobian 与伪逆计算、LM 正则化 Newton 迭代、端到端梯度传播以及动态松弛调度。

**📊 数据集**

实验数据集包括 10,000 个线性/非线性参数化优化问题实例（NCPs、QCQPs）以及无人车障碍避免的控制任务。

**📈 对比分析**

与 Soft‑Constraints、投影/两阶段方法及传统线性约束修复等基线比较，SnareNet 在可行性、最优性间距、种子波动性和推理速度等指标上均显著优于基线。

**⚠️ 局限性**

局限性：对线性约束要求全秩，非线性约束需 LM 正则化计算成本高；需要手动调节松弛参数，对极大规模或复杂非线性约束的适用性仍有限。

---

## 34. SceneReVis: A Self-Reflective Vision-Grounded Framework for 3D Indoor Scene Synthesis via Multi-turn RL

**arXiv ID:** 2602.09432 | [PDF](https://arxiv.org/pdf/2602.09432v1)

**作者:** Yang Zhao `[一作]` (Shanghai Jiao Tong University), Jiang Bian `[通讯]` (Microsoft Research Asia)

**通讯引用:** 13578 | [OpenAlex ID](https://openalex.org/A5030951014)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于视觉自我反思的多轮强化学习框架SceneReVis，用于生成满足物理与语义约束的三维室内场景；

**💡 创新点**

创新点在于将场景合成转化为“诊断‑执行”闭环过程，利用多模态视觉反馈实现实时纠错，并通过逆向工程构建的大规模流程数据集SceneChain‑12k实现了多轮推理训练；

**🔧 技术方法**

结合了大语言模型（Qwen系列）+视觉语言模型+强化学习（GRPO）+物理仿真反馈的复合技术；

**📊 数据集**

使用3D‑FRONT数据集逆向生成的SceneChain‑12k（12,177条流程），并在此基础上进行SFT与RL训练；

**📈 对比分析**

与DiffuScene、Respace、LayoutGPT、LayoutVLM、Holodeck等方法对比，SceneReVis在物理违规率、碰撞率、视觉与语义评估（Ra/Spa/Ac）上均取得SOTA表现，并在长尾场景与目标导向优化任务中表现尤为突出；

**⚠️ 局限性**

局限性包括对大型LLM的依赖、训练与推理算力需求高，以及在极其复杂或非标准室内场景下的泛化能力仍待提升。

---

## 35. A11y-CUA Dataset: Characterizing the Accessibility Gap in Computer Use Agents

**arXiv ID:** 2602.09310 | [PDF](https://arxiv.org/pdf/2602.09310v1)

**作者:** Ananya Gubbi Mohanbabu `[一作]` (University of California), Amy Pavel `[通讯]` (University of California)

**通讯引用:** 1902 | [OpenAlex ID](https://openalex.org/A5014816733)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并公开了 A11y-CUA 数据集，记录了 16 名受试者（8 视力正常、8 盲/低视力）在 60 项日常计算任务中的 40.4 小时交互日志，并评估了主流 CUAs 在默认与辅助技术（屏幕阅读器、放大镜）条件下的表现

**💡 创新点**

首次提供了面向盲/低视力用户的跨应用交互数据，量化了 SUs 与 BLVUs 的交互差异与内部多样性，并揭示了 CUAs 在辅助技术环境下的感知、认知与动作三大缺口

**🔧 技术方法**

使用自研的多模态计算机使用记录器（捕获屏幕视频、系统音频、键鼠事件、UI 自动化树、DOM/可访问性树等），结合大型语言模型（Claude Sonnet 4.5、Qwen3‑VL‑32B‑Instruct）构建 CUAs 并在三种条件下执行

**📊 数据集**

A11y‑CUA 数据集（16 位受试者、60 任务、158,325 事件），公开在 Hugging Face，包含视频、音频、键鼠日志、UI/DOM 结构等同步轨迹

**📈 对比分析**

通过对比 SU 与 BLVU 的成功率、耗时、动作数与 CUAs 在默认、键盘‑仅与放大镜三种模式下的 78.3%、41.67% 与 28.33% 成功率，以及 Qwen3‑VL 在默认/AT 条件下 20% 与 0% 的表现，展示了当前 CUAs 在辅助技术下的显著性能下降

**⚠️ 局限性**

受试者使用统一环境导致个性化辅助技术设置缺失；仅在 Windows + 关闭源应用上收集，缺乏跨平台与开源软件的数据；任务为封闭式，未覆盖更复杂或开放式交互场景

---

## 36. Probabilistic Fair Ordering of Events

**arXiv ID:** 2602.09148 | [PDF](https://arxiv.org/pdf/2602.09148v1)

**作者:** Muhammad Haseeb `[一作]` (New York University), Anirudh Sivaraman `[通讯]` (New York University)

**通讯引用:** 3070 | [OpenAlex ID](https://openalex.org/A5008674902)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个名为Tommy的公平排序器，利用时钟同步误差分布进行概率比较，并将排序问题映射到社会选择理论中的排名问题，以生成事件的部分顺序。

**💡 创新点**

创新点包括：①使用概率比较的likely‑happened‑before关系而非单纯时间戳；②将排序问题转化为Condorcet/Smith集求解，处理非传递性关系；③结合心跳同步实现在线稳定性判定。

**🔧 技术方法**

技术手段：时钟误差统计模型、概率前驱计算、社会选择理论排名（Smith集、强连通分量、拓扑排序）、差分分布预计算、ns‑3仿真、Spanner TrueTime基线比较。

**📊 数据集**

使用的数据集：ns‑3仿真生成的Fat‑Tree数据中心延迟分布（Homa模型）、随机生成的客户端时钟漂移与偏移（正态分布）以及事件生成序列。

**📈 对比分析**

对比方法：与Spanner TrueTime基线在Rank Agreement Score（RAS）指标上比较公平性；实验表明Tommy在大多数延迟/漂移场景下显著优于基线；在线排序中预计算差分分布显著提升计算速度。

**⚠️ 局限性**

局限性：①假设时钟误差分布独立且短期稳定；②未考虑应用级延迟噪声；③仅在ns‑3仿真中验证，实际部署可能面临更复杂因素；④不支持拜占庭故障，需进一步扩展。

---

## 37. Atlas: Enabling Cross-Vendor Authentication for IoT

**arXiv ID:** 2602.09263 | [PDF](https://arxiv.org/pdf/2602.09263v1)

**作者:** Sanket Goutam `[一作]` (Stony Brook University), Amir Rahmati `[通讯]` (Stony Brook University)

**通讯引用:** 4297 | [OpenAlex ID](https://openalex.org/A5021423602)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并实现了一个基于 Web PKI 的跨厂商 IoT 认证框架，利用厂商控制的 DNS 子域和 ACME 协议为每台设备颁发 X.509 证书，支持设备直接使用 mTLS 进行安全互联，消除云中介的延迟与单点故障。

**💡 创新点**

创新点：①将 ACME 与 IoT 结合，允许厂商云充当 ACME 客户端完成域验证；②通过在厂商 DNS 名称空间下为设备生成唯一子域，实现全球可验证且不需要硬件改造的身份；③提供完整的证书生命周期管理（签发、续期、吊销），并与现有 MQTT/CoAP 等协议无缝集成，突破传统云‑垂直集成导致的跨域互操作瓶颈。

**🔧 技术方法**

核心技术：ACME 协议、X.509 证书、DNS 子域绑定、TLS1.3 / mTLS、MQTT、CoAP、ESP32 / Raspberry Pi 设备、AWS IoT Core 对比、ns‑3 城市仿真、Let's Encrypt 公共 CA。

**📊 数据集**

实验数据集：自建智能家居测试平台（MQTT 代理 + 3 台 Raspberry Pi/ESP32 设备）和 ns‑3 生成的城市级 IoT 网络（50–2500 台移动节点 + 5 台网关），消息速率 5–100 msg/s；对比基准使用 AWS IoT Core 的云中介路径。

**📈 对比分析**

评估方法：①对设备级性能测量（延迟、CPU 占用）在 insecure MQTT 与 mTLS MQTT 之间对比；②在云端测量证书颁发（域绑定 + ACME）所需时间；③在智能家居与城市仿真场景中对比传统云中介（AWS IoT + IFTTT）与本框架的端到端延迟、吞吐量和丢包率。结果显示：mTLS 仅增加约 17 ms 延迟、CPU < 9 %；证书颁发 < 6 s/设备；云中介平均延迟 12.7 s，存在长尾，吞吐量有限；本框架在多设备负载下保持毫秒级延迟。

**⚠️ 局限性**

局限性：未在 CA 或云端发生故障时进行可用性/故障注入实验；ESP32 仅使用 TLS 证明 mTLS 性能，未完整验证；未评估 CT 日志对隐私的实际影响；假设厂商可信，未探讨恶意厂商或关键私钥泄露后的补救；实验仅覆盖 Wi‑Fi 与 Linux/MCU 设备，未涵盖其他网络或硬件平台。

---

## 38. Data Sharing with Endogenous Choices over Differential Privacy Levels

**arXiv ID:** 2602.09357 | [PDF](https://arxiv.org/pdf/2602.09357v1)

**作者:** Raef Bassily `[一作]` (Ohio State University), Juba Ziani `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 303 | [OpenAlex ID](https://openalex.org/A5008250785)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在差分隐私（DP）约束下，异质隐私成本的参与者如何自发形成数据共享联盟，并通过联盟成员自主决定是否加入及其隐私参数，探讨联盟稳定性及其对准确性与社会福利的影响。

**💡 创新点**

创新点在于①将隐私参数的设定作为联盟内外的自适应决策引入博弈；②提出“强健平衡”概念，允许现有成员阻止外部成员加入，从而扩展传统Nash平衡；③对不同隐私成本增长规律（α取值范围）下的最优联盟结构、价格效应及与集中式最优解的效率差距做了全面理论分析。

**🔧 技术方法**

主要技术包括差分隐私的Laplace机制、敏感度与隐私预算关系、博弈论中的Nash与强健平衡定义、社会成本与方差的数学表达式、对α的分段分析以及价格稳定性（Price of Stability）的推导。

**📊 数据集**

本文为理论研究，没有使用具体公开数据集；所有结果均基于假设的隐私成本分布和方差σ²的通用分析。

**📈 对比分析**

与基线集中式设计相比，作者通过理论上计算价格稳定性来比较去中心化联盟与全局最优的社会成本与估计方差。结果显示：在α∈[-1,-1/2)区间，去中心化能实现非平凡的准确性提升，但效率差距显著；在α∈(-1/2,1/2]区间，集中式可获更佳准确性，去中心化效率下降；当α>1/2时，两者差距趋于常数。

**⚠️ 局限性**

限制包括：仅考虑单一统计量（均值）且一次性聚合；假设玩家信息完全公开；只讨论单一联盟且不考虑多联盟或重复交互；使用的DP机制为局部DP的Laplace噪声，未覆盖更复杂的隐私模型或实证验证。

---

## 39. Diffusion-Guided Pretraining for Brain Graph Foundation Models

**arXiv ID:** 2602.09437 | [PDF](https://arxiv.org/pdf/2602.09437v1)

**作者:** Xinxu Wei `[一作]` (Lehigh University), Yu Zhang `[通讯]` (Stanford University)

**通讯引用:** 13712 | [OpenAlex ID](https://openalex.org/A5100433691)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种基于扩散的预训练框架，用于脑图和脑超图的图对比学习与掩码自编码器，解决了随机丢弃/掩码导致语义破坏和缺乏全局结构信息的问题。

**💡 创新点**

创新点在于利用扩散引导的结构感知丢弃/掩码、扩散感知读出和全局重建，既保留脑网络语义，又引入全局上下文，提升表示学习质量。

**🔧 技术方法**

采用图扩散核（如随机游走、个性化 PageRank、热核）与图/超图神经网络相结合的预训练方法，并在对比学习和自编码器中实现扩散驱动的增强与重建。

**📊 数据集**

在超过25,000名受试者、60,000次扫描的多种神经影像数据集上验证，包括 ABIDE、ADHD200、OASIS3、ADNI、HBN、CUD、UCLA_CNP 等。

**📈 对比分析**

与传统无预训练模型、基于时间序列的 BrainLM/Brain-JEPA、基于连通性的 BrainMass、以及带有扩散架构的 GDT 等方法比较，Diffusion‑Guided 预训练在多种疾病分类任务上均取得更高的 AUC/ACC/F1，尤其在图/超图层面上表现突出。

**⚠️ 局限性**

局限性包括对超大规模数据仍有计算瓶颈、在极低样本量下相对优势有限，以及扩散参数选择需经验调优，且对不同图结构的泛化能力尚待进一步验证。

---

## 40. Priority-Aware Shapley Value

**arXiv ID:** 2602.09326 | [PDF](https://arxiv.org/pdf/2602.09326v1)

**作者:** Kiljae Lee `[一作]` (Ohio State University), Yuan Zhang `[通讯]` (Ohio State University)

**通讯引用:** 12478 | [OpenAlex ID](https://openalex.org/A5066093415)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种同时考虑硬优先约束和软优先权重的Shapley值方法——Priority‑Aware Shapley Value（PASV）。

**💡 创新点**

创新点在于把通用的先后关系（DAG）与个体权重统一到一个随机顺序框架，并给出唯一的公理化表征；同时提供了极端权重下的理论分析。

**🔧 技术方法**

主要技术包括随机顺序值（ROV）框架、基于DAG的可行集与最大元素概念、权重归一化因子、MCMC（相邻交换 Metropolis‑Hastings）采样和理论极限分析。

**📊 数据集**

实验使用 MNIST、CIFAR‑10（数据价值评估）和 UCI Census Income（特征归因）数据集。

**📈 对比分析**

与经典 Shapley、加权 Shapley（WSV）和优先 Shapley（PSV）比较，PASV在遵守数据重用顺序、调节风险权重方面表现更优；特征归因结果更稳定，且优先‑扫描可视化诊断性能良好。

**⚠️ 局限性**

局限包括：需要 MCMC 采样导致计算量大；对极端权重的解释在一般 DAG 下仍不完全；以及在极大规模数据时仍需进一步加速。

---

## 41. VLM-UQBench: A Benchmark for Modality-Specific and Cross-Modality Uncertainties in Vision Language Models

**arXiv ID:** 2602.09214 | [PDF](https://arxiv.org/pdf/2602.09214v1)

**作者:** Chenyu Wang `[一作]` (Boston University), Wenchao Li `[通讯]` (Boston University)

**通讯引用:** 3413 | [OpenAlex ID](https://openalex.org/A5100381719)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个针对视觉‑语言模型（VLM）的不确定性量化基准 VLM‑UQBench，并对九种不确定性估计方法在四种 VLM 上进行了系统评测。

**💡 创新点**

创新点包括：①提供了图像、文本与跨模态三类来源的不确定性标注子集；②设计了可扩展的扰动管道，能够在任何 VQA 数据集上自动注入可控的视觉、文本和跨模态噪声；③提出了两种新指标（URR、HCC）用于评估不确定性对扰动的敏感性和与幻觉（hallucination）的相关性。

**🔧 技术方法**

使用了白盒（如平均词熵、困惑度、语义熵）和黑盒（如词汇相似度、DegMat、LUQ）不确定性估计技术，结合扰动管道生成对比样本，并用 AUROC、F1、URR、HCC 进行评估。

**📊 数据集**

数据集包括：人标注的 VizWiz 子集（Clean、Image、Text、Cross）、基于 VQ‑FocusAmbiguity 的跨模态子集、CLEVR‑Hallucination 的幻觉子集，以及 VQAv2 等标准 VQA 数据集作为扰动的基础。

**📈 对比分析**

实验表明：不同 UQ 方法在不同模态上表现差异显著；在 VLM‑UQBench 上，某些方法在文本不确定性上表现优异（如语义熵），但在视觉不确定性上往往接近随机；总体而言，UQ 方法对幻觉的预测能力弱且高度依赖模型，未能提供可靠的风险信号。

**⚠️ 局限性**

局限性包括：①VizWiz 子集规模有限且存在人工标注的主观性；②扰动强度采用经验校准，缺乏更系统的数据或模型驱动的选择方案；③基准主要聚焦于 VQA 任务，对其他 VLM 下游任务的通用性仍待验证。

---

## 42. Counterfactual Maps: What They Are and How to Find Them

**arXiv ID:** 2602.09128 | [PDF](https://arxiv.org/pdf/2602.09128v1)

**作者:** Awa Khouna `[一作]` (Polytechnique Montreal), Thibaut Vidal `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出了针对树集成模型的“counterfactual maps”方法，用以在一次预处理后快速得到全局最优反事实解释。

**💡 创新点**

创新点在于将反事实搜索视为最近区域搜索，利用可压缩的轴对齐超矩形划分和泛化Voronoi分区，构建可查询的KD‑tree索引，实现全局最优且毫秒级响应。

**🔧 技术方法**

技术包括：树集成到轴对齐超矩形集合的提取（born‑again tree 近似），加权 Lp 距离的最近矩形搜索，基于体积 KD‑tree 的分支限界查询。

**📊 数据集**

实验使用四个高风险领域的数据集：Breast‑Cancer、COMPAS、FICO、Pima‑Diabetes，包含数值、类别和二进制特征。

**📈 对比分析**

与 OCEAN、Feature Tweaking、LIRE 等基线对比，CF‑Maps 在保持全局最优的前提下，平均查询时延仅数毫秒，整体运行时间比 OCEAN 快数百倍，且误差比率趋近 1；而启发式方法常出现失败或成本过高。

**⚠️ 局限性**

局限性包括：预处理阶段需提取并构造超矩形划分，规模较大时仍是瓶颈；对非轴对齐或连续非线性模型不适用；需要先验距离度量与可行动性约束的定义。

---

## 43. Stability and Concentration in Nonlinear Inverse Problems with Block-Structured Parameters: Lipschitz Geometry, Identifiability, and an Application to Gaussian Splatting

**arXiv ID:** 2602.09415 | [PDF](https://arxiv.org/pdf/2602.09415v1)

**作者:** Joe-Mei Feng `[一作]` (Tamkang University), Hsin-Hsiung Kao `[通讯]` (Central Police University)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5085389272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

提出了一套基于算子理论的非线性逆问题稳定性与统计收敛框架，并将其应用于高维块结构参数的高斯Splatting渲染模型，推导出内在的稳定性–分辨率权衡；

**💡 创新点**

创新点在于将块结构的Lipschitz几何、局部可辨识性和子高斯噪声结合，得到非渐近的高概率误差界，并首次揭示了高斯Splatting中图像分辨率与模型复杂度对稳定性的内在竞争关系；

**🔧 技术方法**

主要技术包括块结构的Lipschitz分析、局部可辨识性与方向可观测性证明、子高斯浓度不等式以及对齐过程的上界计算；

**📊 数据集**

本研究为理论工作，未使用具体数据集；

**📈 对比分析**

没有进行实验性比较，性能评价主要以理论误差下界和收敛速度展示；

**⚠️ 局限性**

局限在于依赖于块结构的假设和局部可辨识性，无法直接量化全局非线性行为，并且对实际优化算法的收敛性未给出具体证明。

---

## 44. Human Control Is the Anchor, Not the Answer: Early Divergence of Oversight in Agentic AI Communities

**arXiv ID:** 2602.09286 | [PDF](https://arxiv.org/pdf/2602.09286v1)

**作者:** Hanjing Shi `[一作]` (Lehigh University), Dominic DiFranzo `[通讯]` (Lehigh University)

**通讯引用:** 1332 | [OpenAlex ID](https://openalex.org/A5028356812)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2026年1月30日至2月6日两个新兴代理人工智能社区r/openclaw和r/moltbook的早期讨论进行主题建模与监督主题映射，比较各社区对监督、边界和责任的关注点。

**💡 创新点**

首次系统比较两类角色差异化的代理生态中监督话语的形成与差异，并提出基于主题加权热度的监督主题框架。

**🔧 技术方法**

使用LDA主题建模、Jensen–Shannon散度、余弦相似度、置换检验以及基于活跃度的热度计量。

**📊 数据集**

收集自Reddit的帖子与评论，覆盖2026年1月1日至2月6日，共2,733条记录，其中698个线程用于建模。

**📈 对比分析**

通过在联合LDA空间下计算主题分布的JSD（0.418）和余弦相似度（0.372）以及置换检验p=0.0005，证明两社区在监督议题上的显著差异；热度分析显示各自侧重的监督主题差异明显。

**⚠️ 局限性**

样本窗口短、仅来自Reddit、主题映射主观性高，且缺乏跨平台与长期跟踪验证。

---

## 45. GAFR-Net: A Graph Attention and Fuzzy-Rule Network for Interpretable Breast Cancer Image Classification

**arXiv ID:** 2602.09318 | [PDF](https://arxiv.org/pdf/2602.09318v1)

**作者:** Lin-Guo Gao `[一作]` (Mokwon University), Suxing Liu `[通讯]` (Mokwon University)

**通讯引用:** 221 | [OpenAlex ID](https://openalex.org/A5013549898)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于图注意力与可训练模糊规则的可解释网络 GAFR-Net，用于乳腺癌病理图像分类

**💡 创新点**

将图注意力机制与模糊推理融合至端到端模型，实现结构特征与可解释推理的统一

**🔧 技术方法**

多头图注意力网络、拓扑特征提取（聚类系数、节点度、两跳标签一致性）、可微模糊规则层、门控融合及 Softmax 分类

**📊 数据集**

BreakHis、Mini-DDSM、ICIAR2018（BACH）三大公开乳腺病理图像数据集

**📈 对比分析**

与 ResNet、GCN、ViT、Swim、CoAtNet 等十种先进方法对比，GAFR-Net 在 AUC‑ROC、准确率、Kappa 等指标上均实现了领先，并保持高敏感度与特异度

**⚠️ 局限性**

对大规模全切片图像的图构建与推理仍存在计算成本与可迁移性挑战，模型在极大图结构下的效率与泛化仍需进一步优化

---

## 46. Improved Parallel Repetition for GHZ-Supported Games via Spreadness

**arXiv ID:** 2602.09290 | [PDF](https://arxiv.org/pdf/2602.09290v1)

**作者:** Yang P. Liu `[一作]` (Carnegie Mellon University), Kunal Mittal `[通讯]` (New York University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文证明了对于任何三人游戏，只要其提问分布的支持与 GHZ 游戏相同，n 次并行重复的游戏值会呈指数衰减，并给出了相应的集中上界；

**💡 创新点**

创新点在于引入“代数 Spreadness”这一伪随机性概念，基于 Kelley‑Meka 的 3 项等差数列无解集合技术，首次实现了在 GHZ 支持下从多项式到指数衰减的跃迁；

**🔧 技术方法**

主要技术包括：代数 Spreadness 与组合 Spreadness 的转换、对任意集合的 Spreadness 分解（Uniformization）、利用 Spreadness 的平方覆盖（Square Covering）证明分布近似均匀、以及在并行重复框架下的诱导证明；

**📊 数据集**

该工作为纯理论研究，无需使用任何实验数据集；

**📈 对比分析**

相较于之前的多项式上界（n^-Ω(1)），本文的上界为 exp(-n^c)，在 n 足够大时显著更优；同时提供了关于成功率上限的概率集中界；

**⚠️ 局限性**

局限性：仅适用于 3‑玩家游戏并且提问分布必须与 GHZ 游戏的支持完全一致；对于更一般的多玩家或非 GHZ 支持的游戏仍无法得到类似的指数衰减结果；

---

## 47. $n$-Musketeers: Reinforcement Learning Shapes Collaboration Among Language Models

**arXiv ID:** 2602.09173 | [PDF](https://arxiv.org/pdf/2602.09173v1)

**作者:** Ryozo Masukawa `[一作]` (University of California), Mohsen Imani `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者提出了一种软隐藏状态协作方法，利用冻结的多种小型语言模型（SLM）的隐藏表示，通过可训练的 Perceiver‑style 交叉注意力门把这些表示转换为上下文 token，然后在强化学习验证奖励（RLVR）框架下对单一策略进行训练，从而实现多专家的协同推理。

**💡 创新点**

创新点在于：
1) 引入了“软隐藏状态协作”，即在隐藏层面而非输出层面对多模型进行融合；
2) 通过可训练的 latent 适配器和交叉注意力瓶颈，将不同专家的隐藏表示对齐并聚合，形成可观测的专家利用动态；
3) 证明 RLVR 本身能在无显式路由监督的情况下诱导专家角色的自组织与结构化分配；
4) 展示该方法在不同任务难度下的可调性与局限性。

**🔧 技术方法**

使用的技术主要包括：
- RLVR（基于结果级奖励的强化学习）
- Perceiver‑style 交叉注意力瓶颈（latent query + cross‑attention）
- LoRA 轻量化适配器
- GRPO（基于 KL 正则化的 PPO 变体）
- 句子级语义编码器（用于硬路由对照实验）

**📊 数据集**

采用的评测数据集：
- Reasoning Gym（Arithmetic、Logic、Algorithmic 三类任务）
- GSM8K（数学推理基准）

**📈 对比分析**

比较方法：与单模型 RLVR、输出级协作（文本提示拼接）以及硬专家路由（Top‑1）等基线进行对比。实验表明：
- 在 Arithmetic 和 Logic 任务中，软隐藏状态协作可实现最高 22.9% 的准确率提升；
- 在 Algorithmic 和 GSM8K 任务中，提升有限甚至出现下降；
- 软隐藏状态协作在不同专家组合下表现出显著的方差，说明其收益依赖于任务难度与专家能力。

**⚠️ 局限性**

局限性：
- 仅在任务难度适中的场景才有显著收益，对已饱和的任务（如 GSM8K）可能产生干扰；
- 训练过程对专家池的选择和隐藏表示对齐的超参数敏感，导致收敛不稳定；
- 无法完全区分专家的功能专长与容量优势，导致“高容量模型被偏好”的现象；
- 目前只关注隐藏表示层面，尚未验证更细粒度的专家交互或自适应执行策略。

---

## 48. DRAGON: Robust Classification for Very Large Collections of Software Repositories

**arXiv ID:** 2602.09071 | [PDF](https://arxiv.org/pdf/2602.09071v1)

**作者:** Stefano Balla `[一作]` (Universita di Bologna), Romain Robbes `[通讯]` (Univ. Bordeaux)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大规模软件仓库中实现了一种可对仓库进行多标签分类的系统

**💡 创新点**

提出了仅使用文件夹/文件名和可选 README 的轻量级句子对 BERT 模型，并通过焦点损失和按标签阈值调节实现鲁棒性与覆盖率平衡

**🔧 技术方法**

利用 BERT 的句子对编码、focal loss 处理类别不平衡、以及对标签进行阈值调节的技术

**📊 数据集**

基于 Software Heritage 中 825,000 份仓库、映射至 GitRanking 239 个领域级标签的最新公开数据集

**📈 对比分析**

与 LEGION 等基准对比，F1@5 达到 60.8%，比 LEGION 高约 6个百分点；在 README 缺失情况下，F1@5 仅下降 9% 以内，表现出显著的鲁棒性

**⚠️ 局限性**

存在标签噪声、稀有类别预测效果仍差、主要采用英文 tokenizer 对多语言仓库的适应性有限，且对近义标签的处理仍需改进

---

## 49. Elements of Robot Morphology: Supporting Designers in Robot Form Exploration

**arXiv ID:** 2602.09203 | [PDF](https://arxiv.org/pdf/2602.09203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 50. Accelerating Post-Quantum Cryptography via LLM-Driven Hardware-Software Co-Design

**arXiv ID:** 2602.09410 | [PDF](https://arxiv.org/pdf/2602.09410v1)

**作者:** Yuchao Liao `[一作]` (University of Arizona), Roman Lysecky `[通讯]` (University of Arizona)

**通讯引用:** 2505 | [OpenAlex ID](https://openalex.org/A5015269679)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了利用大型语言模型 ChatGPT‑4o 自动分析 FALCON 算法的性能热点并生成 FPGA HDL 代码，以实现硬件‑软件协同加速。

**💡 创新点**

首次将 LLM 用于硬件‑软件分离与 RTL 代码生成，展示其在比传统 HLS 更快的速度与更短临界路径上的潜力，同时揭示资源占用与功耗的权衡。

**🔧 技术方法**

采用 ChatGPT‑4o 进行源代码与 gprof 结果的分析、Prompt‑engineered RTL 生成；对比 Vitis HLS；在 Xilinx Zynq UltraScale+ MPSoC（ZCU104）上实现、验证与放线；使用 gprof 进行性能基准。

**📊 数据集**

使用 FALCON 官方参考实现（包含 FALCON‑512 与 FALCON‑1024 参数集）和 gprof 生成的运行时统计作为实验数据集。

**📈 对比分析**

通过将 LLM 生成的加速器与 HLS 生成的加速器在同一 FPGA 设计流中放线后对比，评估每个低层核的执行时间、LUT/FF/DSP 使用、功耗；结果显示 LLM 设计平均提升 1.78× 的速度（对比 HLS 的 1.15×），并在关键核实现最高 2.6× 的速度提升与更短的临界路径，但伴随更高的资源占用与功耗。

**⚠️ 局限性**

局限性包括：需要人工调优 Prompt 与验证；LLM 生成 RTL 的资源占用相对较高；未进行正式的常数时间与侧信道安全验证；系统级整体性能与多核协同尚未完成评估。

---

## 51. Understanding and Enhancing Encoder-based Adversarial Transferability against Large Vision-Language Models

**arXiv ID:** 2602.09431 | [PDF](https://arxiv.org/pdf/2602.09431v1)

**作者:** Xinwei Zhang `[一作]` (Hong Kong Polytechnic University), Haibo Hu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8698 | [OpenAlex ID](https://openalex.org/A5020630816)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对大型视觉语言模型（LVLM）的编码器基础对抗攻击的可迁移性进行系统研究，并提出一种语义引导的多模态攻击框架

**💡 创新点**

识别并解释编码器攻击可迁移性受限的根本原因（视觉定位不一致与语义冗余），并设计语义相关扰动与全局/局部语义破坏双重策略提升迁移性

**🔧 技术方法**

利用CLIP的Grad‑ECLIP定位语义重要区域，结合语义相似度损失、图像-图像对抗损失以及基于名词短语的局部损失进行PGD优化

**📊 数据集**

Flickr30k（图像描述）、CIFAR‑10（分类）与TextVQA（问答）等数据集

**📈 对比分析**

与现有的VT‑Attack、Attack‑Bard、TGR、PNA等对比，使用攻击成功率（ASR）和CLIP相似度评估，在开源模型上达到约55% ASR，在闭源模型上也实现约25%+，显著优于现有方法

**⚠️ 局限性**

对抗样本仍在闭源大型模型（GPT‑4o、Gemini）上效果有限；攻击依赖零查询黑盒环境，缺乏对模型防御机制的鲁棒性评估；对语义提取的依赖可能限制对非文本描述任务的通用性

---

## 52. The Price of Privacy For Approximating Max-CSP

**arXiv ID:** 2602.09273 | [PDF](https://arxiv.org/pdf/2602.09273v1)

**作者:** Prathamesh Dharangutte `[一作]` (Rutgers University), Zongrui Zou `[通讯]` (Nanjing University)

**通讯引用:** 111 | [OpenAlex ID](https://openalex.org/A5088656303)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在约束被视为敏感数据的 Max‑CSP 问题下实现差分隐私的近似算法，分析在不同隐私预算（高隐私与低隐私）下能达到的最佳近似比率，并给出相应的多项式或指数级算法与信息论下界。

**💡 创新点**

①在高隐私区间证明任意 ε‑DP 算法的近似比率不能超过 μ + O(ε)；②给出在有界度且无三角结构下的多项式时间算法，匹配上界并实现 μ + Ω(ε/√D)；③对 Max‑Cut 与 Max‑kXOR 的特殊情形去掉上述限制，获得 1/2 + Ω(ε) 或 1/2 + Ω(ε³) 的优势；④在低隐私区间给出接近最优的 1 − Θ(1/ε) 近似比率与相应下界；⑤给出多项式时间与指数时间两类算法，揭示了隐私预算对近似性能的根本影响。

**🔧 技术方法**

采用 Shearer 的随机割算法、随机划分与随机响应、指数机制（Exponential Mechanism）、傅里叶分析与负相关随机变量的工具、隐私放大与子采样技术、贪心线性提升与 Chebyshev 翻转等组合实现隐私化处理与近似改进。

**📊 数据集**

本文为理论分析，未使用具体实验数据集；通过构造随机或特殊实例来证明上界与下界。

**📈 对比分析**

通过信息论下界与算法实现的近似比率进行对比：在高隐私下，算法可实现 μ + Θ(ε/√D)（对 Max‑Cut 甚至达到 1/2 + Ω(ε)），比非私有多项式算法的 1/2 仅略优；在低隐私下，近似比率接近 1 − Θ(1/ε)，比非隐私算法略低；总体性能受隐私预算限制，隐私约束导致近似只能略好于随机。

**⚠️ 局限性**

①对一般 Max‑CSP 高隐私区间仍缺乏多项式时间算法与上界的匹配；②Max‑kXOR 偶数 k 仅得到 ϵ³ 的优势；③低隐私区间的 deficit 与下界仍存在指数级差距；④整体对大多数 CSP 的复杂度与隐私约束下的可行性尚未完全解决。

---

## 53. CausalGDP: Causality-Guided Diffusion Policies for Reinforcement Learning

**arXiv ID:** 2602.09207 | [PDF](https://arxiv.org/pdf/2602.09207v1)

**作者:** Xiaofeng Xiao `[一作]` (Northeastern University), Xubo Yue `[通讯]` (Northeastern University)

**通讯引用:** 231 | [OpenAlex ID](https://openalex.org/A5001018616)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出一种结合因果推断的扩散策略CausalGDP，能够在强化学习中实时更新因果动态模型并引导动作生成；

**💡 创新点**

创新点在于将因果指导嵌入扩散模型，使策略聚焦真正导致高奖励的动作分量，并通过实时干预提升采样效率；

**🔧 技术方法**

采用扩散模型、因果发现方法（如NOTEARS）、干预式扩散指导、Q网络估计及DDIM/score‑based diffusion等技术；

**📊 数据集**

使用Gym MuJoCo（HalfCheetah、Hopper、Walker2d、Humanoid）v4以及D4RL（Maze2D、AntMaze、Adroit、Kitchen）等多种离线与在线任务数据集；

**📈 对比分析**

与多种基线（CQL、IQL、Diffusion‑QL、DT、DD等）比较，CausalGDP在大多数任务上实现或超过最优表现，收敛更快；

**⚠️ 局限性**

对因果结构的准确性较为敏感，因果发现错误可能略降性能；扩散与干预的组合计算量较大，适用性仍需在更高维场景进一步验证。

---

## 54. Understanding Remote Mental Health Supporters' Help-Seeking in Online Communities

**arXiv ID:** 2602.09353 | [PDF](https://arxiv.org/pdf/2602.09353v1)

**作者:** Tuan-He Lee `[一作]` (Cornell University), Gilly Leshed `[通讯]` (Cornell University)

**通讯引用:** 1626 | [OpenAlex ID](https://openalex.org/A5046611715)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Reddit上远程精神健康照护者的支持寻求行为与社区回应进行了定性与定量分析；

**💡 创新点**

首次系统挖掘远程照护者的独特需求、挑战与社区响应，揭示远程情境下支持缺失与需求不匹配；

**🔧 技术方法**

采用手工编码与LLM辅助（GPT‑4）主题分析相结合，并使用非参数统计（Mann–Whitney U、chi-square）评估参与度；

**📊 数据集**

使用522条远程支持者原帖及3,355条评论，来自六个关注精神疾病的Reddit子版块（r/depression_partners、r/family_of_bipolar 等），总计8,361条；

**📈 对比分析**

通过与非远程贴子比较（Mann–Whitney U）发现远程贴子更长、点赞和评论数更少；按发帖目的对比显示请求指导得票最低、情绪表达得票最高；社区回复主题一致性低，远程上下文关注率仅13–18%，表明需求匹配不足；

**⚠️ 局限性**

仅覆盖六个公开子版块，可能遗漏其他平台与小众社区；LLM 辅助识别与编码的精度不一；缺乏对长期参与和个体动机的探究。

---

## 55. LARV: Data-Free Layer-wise Adaptive Rescaling Veneer for Model Merging

**arXiv ID:** 2602.09413 | [PDF](https://arxiv.org/pdf/2602.09413v1)

**作者:** Xinyu Wang `[一作]` (University of Connecticut), Jin Lu `[通讯]` (University of Georgia)

**通讯引用:** 14564 | [OpenAlex ID](https://openalex.org/A5100606222)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种无数据、无训练的层级自适应重缩放外壳（LARV），可挂载于任何任务向量合并方法上，利用两种权重仅诊断指标对每层进行缩放，从而提升多任务模型合并的性能。

**💡 创新点**

创新点在于：①首次将层级差异性（浅层敏感、深层稳定）嵌入合并规则；②设计两种无数据指标——信息丰富度（e_ℓ）与冲突程度（c_ℓ）——用以自适应计算每层缩放因子；③采用简单确定性映射（分段或连续tanh）实现轻量化、可解释的层级缩放。

**🔧 技术方法**

技术实现包括：对每层权重与增量矩阵做随机SVD计算有效秩对比；利用矩阵交换子测量非可交换冲突；标准化并通过softplus/分段或连续tanh得到缩放系数；将系数应用于合并后的增量后再加回基模型。

**📊 数据集**

实验数据集覆盖ViT B/32、B/16、L/14三种尺寸，并在FusionBench的8/14/20任务合并任务（SUN397、Cars、RESISC45、EuroSAT、SVHN、GTSRB、MNIST、DTD等）以及ImageNet-C、未见任务（MNIST、EuroSAT）上验证。

**📈 对比分析**

与Baseline（Simple Averaging、Task Arithmetic、ISO‑C/CTS、TIES、TSV‑M）比较，LARV在所有基线上均提升，平均提升约3–7.7%（ViT‑B/32 20任务可达19%），并在鲁棒性、跨任务泛化上同样表现优于基线。

**⚠️ 局限性**

局限性包括：①仍需对每层做随机SVD，计算开销相对传统合并略高；②仅适用于任务向量合并场景，对需梯度或数据驱动的合并方法兼容性未知；③在极大规模模型或任务数上，层级缩放策略可能需要进一步调整以防过度压制。

---

## 56. Beyond Input-Output: Rethinking Creativity through Design-by-Analogy in Human-AI Collaboration

**arXiv ID:** 2602.09423 | [PDF](https://arxiv.org/pdf/2602.09423v1)

**作者:** Xuechen Li `[一作]` (Tongji University), Qing Chen `[通讯]` (Tongji University)

**通讯引用:** 35852 | [OpenAlex ID](https://openalex.org/A5100371368)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 85 篇 AI 驱动的 Design‑by‑Analogy (DbA) 论文进行系统综述，梳理其表示形式、创作流程阶段、应用领域，并提出 DbA 作为人机协作媒介技术的框架与使用指南。

**💡 创新点**

①将 DbA 的理论与实践贯穿整个创作流程，突破仅局限于早期构思的传统视角；②构建六类表示形式与七个创作阶段的交叉表，系统性阐述 DbA 在不同创作阶段的介入方式；③提出 DbA 作为技术媒介的伦理与治理视角，强调其在赋能人类创造力与避免设计定势中的双重角色。

**🔧 技术方法**

采用 PRISMA 选取流程、主题分析与编码法对文献进行分类与编码；使用概念图、表格与统计可视化呈现代表形式与阶段分布；在此基础上提出对技术实现的四维原则（用户、数据、算法、交互）。

**📊 数据集**

无原始实验数据；研究对象为 1,615 篇候选文献（最终 85 篇），包括来自 ACM、IEEE、ASME、ScienceDirect 等数据库的学术论文。

**📈 对比分析**

本文不做算法对比实验，也未给出数值性能指标；通过定性分析与案例聚合，讨论不同 DbA 方法在创作流程各阶段的优势与局限，提出相对评估框架（例如 Assist、Augment、Automate 等级）。

**⚠️ 局限性**

①文献检索受关键词限制，可能遗漏非传统表述的 DbA 研究；②聚焦已有案例，缺乏跨学科新兴领域（如循环经济、可持续设计）的覆盖；③未对 DbA 工具的实际创造力效果进行实验验证；④技术实现与评估主要来自文献描述，缺乏统一评测标准。

---

## 57. Don't Shoot The Breeze: Topic Continuity Model Using Nonlinear Naive Bayes With Attention

**arXiv ID:** 2602.09312 | [PDF](https://arxiv.org/pdf/2602.09312v1)

**作者:** Shu-Ting Pi `[一作]` (Amazon), Qun Liu `[通讯]` (Amazon)

**通讯引用:** 17502 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于朴素贝叶斯、对数非线性与注意力机制的主题连贯性模型，用以判断对话句子是否保持主题一致。

**💡 创新点**

创新点在于将朴素贝叶斯拆解为对数非线性形式，加入自适应注意力函数与正弦残差修正，解决长文本和跨轮跳转导致的注意力缺失与上下文不连贯问题。

**🔧 技术方法**

采用BERT/Conversational‑BERT进行下一句预测，使用Sentence‑BERT+Isolation‑Forest估计话题内外概率，并通过残差正弦项实现自适应概率校正。

**📊 数据集**

使用4,000条由亚马逊客服人员与LLM模拟的对话数据（包含正常、跳转、领域外/内主题偏移四类标签），并计划公开类似的公开域对话数据集。

**📈 对比分析**

与标准NSP和BERT基线比较，实验显示在不同token长度和token gap场景下，本模型准确率提升约10‑20%，AUC提升14.2%，尤其在token>512时仍保持稳定优于NSP。

**⚠️ 局限性**

局限性在于对窗口划分与超参数（如残差系数）敏感，模型依赖预训练的NSP和Isolation‑Forest，缺乏完全端到端的自动化和跨域通用性。

---

## 58. AIDev: Studying AI Coding Agents on GitHub

**arXiv ID:** 2602.09185 | [PDF](https://arxiv.org/pdf/2602.09185v1)

**作者:** Hao Li `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**通讯引用:** 23771 | [OpenAlex ID](https://openalex.org/A5091586373)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文创建了一个名为AIDev的大规模数据集，记录了在真实GitHub项目中由五种AI编码代理（OpenAI Codex、Devin、GitHub Copilot、Cursor、Claude Code）提交的超过九十万条拉取请求（PR）以及相关的评论、提交差异、事件时间线等多维度信息。

**💡 创新点**

创新点在于首次系统性地聚合并公开了大量AI代理在实际软件开发工作流中的行为数据，提供了完整的PR、代码改动、审查、Issue关联等丰富元数据，为研究AI采纳、协作质量、风险与安全等提供了实证基础。

**🔧 技术方法**

技术上主要利用GitHub REST/GraphQL API抓取仓库、PR、评论、提交、Issue等信息，并对PR进行自动标签（如Conventional Commits分类）和事件时间线重建，同时使用脚本进行差异提取和元数据整理。

**📊 数据集**

数据集本身即为研究对象，包含：all_pull_request（932,791条）、all_repository（116,211个）、all_user（72,189人）；以及一个高星级仓库的精细子集（pull_request 33,596条、repository 2,807、user 1,796）和相关评论、提交差异、issue等表。

**📈 对比分析**

本文未给出算法性能对比，而是提供了可直接使用的Jupyter Notebook示例和Hugging Face/Zenodo下载入口，研究者可基于该数据集自行评估不同AI代理的提交质量、审查耗时、合并率等指标。

**⚠️ 局限性**

局限性包括：数据截至2025年8月，仅覆盖五种代理；仅对星级>100的仓库做了深度挖掘，导致样本偏向热门项目；API速率限制可能导致某些PR或事件漏采；未覆盖所有后续提交和完整审查细节，可能影响对长期协作效果的评估。

---

## 59. CoMMa: Contribution-Aware Medical Multi-Agents From A Game-Theoretic Perspective

**arXiv ID:** 2602.09159 | [PDF](https://arxiv.org/pdf/2602.09159v1)

**作者:** Yichen Wu `[一作]` (Massachusetts General Hospital and Harvard Medical School), Quanzheng Li `[通讯]` (Massachusetts General Hospital and Harvard Medical School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 CoMMa，一种基于贡献感知的去中心化医学多智能体框架，用于肿瘤多学科决策支持。

**💡 创新点**

创新点在于将角色扮演交互替换为数据分离专属智能体，并通过确定性嵌入与 Shapley 值正则化实现可解释的贡献分配。

**🔧 技术方法**

采用冻结的大型语言模型做嵌入投影，MLP 生成代理特征，代理决策矩阵与政策梯度+Shapley 正则化的协作博弈学习。

**📊 数据集**

在真实多学科肿瘤委员会数据集 HCC Tumorboard 以及公开的 MTBBench 分子肿瘤委员会基准上进行评估。

**📈 对比分析**

与 XGBoost、LGBM、CatBoost 以及 GPT‑4.1、MDAgents、MAC 等基线在 AUC 与准确率上对比，CoMMa 在所有任务上均取得最高分且决策更稳定。

**⚠️ 局限性**

受限于样本量有限，无法处理自由文本推理与不确定性决策，且对大型 LLM 的依赖仍带来资源与解释性的挑战。

---

## 60. Feasible Static Workspace Optimization of Tendon Driven Continuum Robot based on Euclidean norm

**arXiv ID:** 2602.09046 | [PDF](https://arxiv.org/pdf/2602.09046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 61. What do Geometric Hallucination Detection Metrics Actually Measure?

**arXiv ID:** 2602.09158 | [PDF](https://arxiv.org/pdf/2602.09158v1)

**作者:** Eric Yeats `[一作]` (Pacific Northwest National Laboratory), Henry Kvinge `[通讯]` (University of Washington)

**通讯引用:** 215 | [OpenAlex ID](https://openalex.org/A5066361433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究设计了多领域、多种幻觉特征的合成数据集，并评估几种基于LLM内部几何统计的幻觉检测方法。

**💡 创新点**

创新点在于将幻觉分为错误性、置信度、无关、无序、缺失四类，并揭示不同几何统计对这些特征的敏感度；同时提出了简单的扰动归一化方法显著缓解域迁移导致的性能下降。

**🔧 技术方法**

使用了隐藏分数（Hidden Score）、矩阵熵（Matrix Entropy）以及注意力分数（Attention Score）等几何统计，并在 Llama 3.1‑8B‑Instruct 上实现。

**📊 数据集**

采用了合成问答数据集，覆盖三大领域（乘法、历史年份、词频），并通过教师强制方式生成含不同幻觉属性的回答。

**📈 对比分析**

与传统的几何统计做对比，经过扰动归一化后，在多域幻觉检测上AUROC提升了34–40个百分点，单域检测保持或略逊。

**⚠️ 局限性**

主要限制是实验仅在合成数据上验证，真实对话环境下的回答定位与扰动选择可能更复杂，且目前仅考虑单一模型架构。

---

## 62. Decoding Future Risk: Deep Learning Analysis of Tubular Adenoma Whole-Slide Images

**arXiv ID:** 2602.09155 | [PDF](https://arxiv.org/pdf/2602.09155v1)

**作者:** Ahmed Rahu `[一作]`, Derrick Forchetti `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用深度学习模型对低级分化腺瘤的数字化组织切片进行特征提取，以预测患者未来是否会发展为结肠直肠癌。

**💡 创新点**

首次证明在低级分化腺瘤中存在可被机器检测的微观形态特征，这些特征能够准确预测肠癌发生，并通过Grad-CAM提供可解释的形态学依据。

**🔧 技术方法**

使用基于EfficientNetV2S的卷积神经网络，并结合颜色归一化、数据增强、dropout以及Grad-CAM可视化技术。

**📊 数据集**

共收集了335,763张高质量图块，来自52名患者（进展组32例，非进展组22例），并预留20张完整切片做独立验证。

**📈 对比分析**

与传统手工评估相比，模型在图块层面获得97.9%准确率、98.0%精确率、98.2%召回率、98.9% F1分数，AUROC接近1；在完整切片层面，所有20张测试切片均被准确分类，性能显著优于传统方法。

**⚠️ 局限性**

局限包括单中心单扫描仪的数据来源、进展组样本量有限、缺乏多机构和多设备的外部验证，可能影响模型的泛化能力。

---

## 63. FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases

**arXiv ID:** 2602.09163 | [PDF](https://arxiv.org/pdf/2602.09163v1)

**作者:** Xingjian Zhang `[一作]` (University of Michigan), Jiaqi W. Ma `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个用于评估AI代理在基因注释知识库建设中的端到端工作流的基准（FlyBase Ontology Curation Benchmark），并提供完整的论文检索语料库、任务定义和评价指标。

**💡 创新点**

创新点在于：① 把传统的分块式 NLP 任务统一到一个“代理”框架中，真实模拟人类生物学家从文献检索、阅读到结构化注释的全过程；② 通过“缺失术语”设置和语义召回指标来评估代理对新知识的发现和推理能力；③ 对四种不同架构（记忆、流水线、单代理、双代理）进行系统比较，揭示架构设计对检索与推理性能的决定性影响。

**🔧 技术方法**

采用了检索增强 LLM（GPT‑5‑mini、GPT‑4o、GPT‑5）、BM25 检索工具、Ontology 搜索与验证工具，并在 ReAct/多代理框架下实现动态工具调用与交互。

**📊 数据集**

使用了 16,898 篇公开获取的 PubMed Central 完整论文（≈1.4 亿词）作为检索语料，配合 FlyBase 的 100 个基因的 3,621 条 GO 注释、1,331 条表达注释和 2,445 条同义词作为专家标注的 ground truth。

**📈 对比分析**

通过在不同检索篇数（1/2/4/8/16）下对四种代理架构进行实验，测量语义召回@k（GO、表达）和精确召回@k（同义词）。实验显示：多代理架构在相同检索预算下的召回率最高（GO 53%/表达 53%/同义词 57%），而单代理在高篇数时因上下文溢出表现下滑；记忆基线在所有任务中表现最差；扩大模型规模（GPT‑5 vs GPT‑5‑mini）提升有限，表明架构比模型规模更关键。

**⚠️ 局限性**

局限性包括：① 仅处理文本信息，忽略图像、表格等多模态数据；② 语料库仅覆盖 30% 公开论文，无法检索被订阅的研究；③ 评价仅基于现有 FlyBase 注释，存在缺失或错误的 ground truth；④ 代理系统尚未与人工审核交互，缺乏实时反馈与纠错机制。

---

## 64. Digital Linguistic Bias in Spanish: Evidence from Lexical Variation in LLMs

**arXiv ID:** 2602.09346 | [PDF](https://arxiv.org/pdf/2602.09346v1)

**作者:** Yoshifumi Kawasaki `[一作]` (University of Tokyo), Yoshifumi Kawasaki `[通讯]` (University of Tokyo)

**通讯引用:** 40 | [OpenAlex ID](https://openalex.org/A5016903678)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型（LLM）在西班牙语地理词汇变异识别中的表现，采用问卷式评估。

**💡 创新点**

创新点在于：①将LLM视为虚拟问卷受访者，使用两种问卷格式（Yes–No 与多选）；②使用覆盖900+词条的专业编纂数据库VARILEX；③为多选任务设计调整后的Jaccard系数，兼顾偶然重叠。

**🔧 技术方法**

技术手段：利用GPT‑4、GPT‑3.5等GPT系列模型通过官方API进行问答，使用F1分数评估Yes–No结果，使用调整后Jaccard评估多选结果。

**📊 数据集**

数据集：VARILEX（西班牙语词汇变异数据库）以及CEREAL语料库的数字资源量作为对照。

**📈 对比分析**

比较方法：与基线（全部回答“Yes”或前三个选项）对比，分别计算F1和调整后Jaccard；在国家和方言区层面得到最高F1≈0.73（西班牙），最低≈0.28（哥伦比亚）；最高J_adj≈0.51（西班牙），最低≈0.14（智利）。总体模型性能约为基线的2–3倍。

**⚠️ 局限性**

局限性：①仅按国家划分，忽略区域内差异；②仅测试GPT系列模型；③VARILEX标注为二元，缺乏频率/分级信息；④未考虑社会人口变量；⑤未评估生成文本的方言敏感性。

---

## 65. Cross-Project Flakiness: A Case Study of the OpenStack Ecosystem

**arXiv ID:** 2602.09311 | [PDF](https://arxiv.org/pdf/2602.09311v1)

**作者:** Tao Xiao `[一作]` (Kyushu University), Yasutaka Kamei `[通讯]` (Kyushu University)

**通讯引用:** 4993 | [OpenAlex ID](https://openalex.org/A5045097606)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文研究了OpenStack生态系统中的跨项目与不一致的测试不稳定性（flakiness），并量化了其在代码评审与CI中的影响。

**💡 创新点**

创新点在于首次从生态系统层面定义并探究跨项目flakiness与不一致flakiness，揭示了单元测试亦可能在多项目中扩散不稳定性，并提供了跨项目与单项目 flakiness 的对比分析。

**🔧 技术方法**

主要技术包括对Zuul CI日志的自动抽取与分析、统计检验（Mann‑Whitney、Cliff’s δ）、定性编码与卡片排序等方法。

**📊 数据集**

使用的数据集涵盖 649 个OpenStack项目、29,175 条关闭的代码评审、73,707 条补丁集、139,768 条Zuul评论、29,911 次flaky build 与 11,506 个flaky test。

**📈 对比分析**

通过统计检验对比跨项目与单项目 flakiness 在耗时与资源消耗上的差异，结果显示跨项目 flakiness 的时间浪费显著更大，效应量为小；同时对 flakiness 发生频率和项目分布进行量化比较。

**⚠️ 局限性**

局限性包括仅针对 OpenStack 生态系统，CI 日志保留时间有限导致可能漏检；分类与编码存在主观性；结果的可推广性待在其他生态系统进一步验证。

---

## 66. Framework for Integrating Zero Trust in Cloud-Based Endpoint Security for Critical Infrastructure

**arXiv ID:** 2602.09078 | [PDF](https://arxiv.org/pdf/2602.09078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 67. First-order friction models with bristle dynamics: lumped and distributed formulations

**arXiv ID:** 2602.09429 | [PDF](https://arxiv.org/pdf/2602.09429v1)

**作者:** Luigi Romano `[一作]` (Linköping University), Erik Frisk `[通讯]` (Linköping University)

**通讯引用:** 2860 | [OpenAlex ID](https://openalex.org/A5070530122)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

研究提出了一种通过逆向摩擦特性推导的首阶速度相关摩擦模型（FrBD），并给出了其集聚式与分布式两种实现形式，进一步分析了其稳定性、被动性，并通过数值仿真与阀门实验验证了其效果。

**💡 创新点**

创新点在于：①利用摩擦曲线反演的物理推导方法生成FrBD模型，天然满足被动性；②在原LuGre模型的基础上引入微支架阻尼与正则化，消除对速度相关阻尼的依赖；③提出可用于滚动接触的分布式PDE描述，扩展了摩擦建模的空间维度。

**🔧 技术方法**

主要技术包括：粘弹性支架理论、隐函数定理与迭代逼近、系统稳定性与被动性证明、线性化与频域分析、分布式参数系统的特征线求解，以及实验系统建模与参数辨识（遗传算法）。

**📊 数据集**

实验数据来源于公开的阀门（diaphragm valve）实验数据库；仿真使用的参数来自文献（如轮胎低摩擦实验、质量-弹簧系统）以及论文自定义的数值参数。

**📈 对比分析**

将FrBD模型与经典LuGre模型在预滑移、摩擦滞后和粘滞滑移等典型现象下进行对比；在阀门实验中，FrBD模型的均方根误差(RMSE)分别为4.45×10⁻⁴和4.77×10⁻⁴，略优于LuGre模型（4.77×10⁻⁴和4.69×10⁻⁴），并在力-位移曲线、相位图和速度-位移耦合上表现出相似或更好的拟合效果。

**⚠️ 局限性**

局限性包括：①模型仍为首阶，仅适用于非粘着状态；②分布式版本仅在理论层面得到验证，缺乏实验数据支持；③正则化参数ε的选取可能影响模型精度；④对非线性支架阻尼、温度效应等复杂摩擦机理未作进一步扩展。

---

## 68. From Adam to Adam-Like Lagrangians: Second-Order Nonlocal Dynamics

**arXiv ID:** 2602.09101 | [PDF](https://arxiv.org/pdf/2602.09101v1)

**作者:** Carlos Heredia `[一作]` `[通讯]` (IAMM Research), Carlos Heredia (IAMM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文从离散 Adam 算法出发，推导出一个二阶连续‑时间非局部整数微分方程模型，并将其与一阶非局部 Adam 流程通过 α‑细化关系联系起来，同时给出了对应的拉格朗日变分表示与 Lyapunov 稳定性分析。

**💡 创新点**

创新点在于提出了一个包含惯性项和线性阻尼、通过因果卷积核捕捉历史梯度的二阶 Adam 连续模型，量化了其相对于一阶模型的 α‑细化误差，并首次给出 Adam‑启发的非局部拉格朗日框架，为设计新的自适应优化器提供了结构化的理论基础。

**🔧 技术方法**

主要技术手段包括：积分微分方程（卷积核解析）、拉格朗日变分与 Euler‑Lagrange 推导、Lyapunov 功能量法、Polyak‑Łojasiewicz 与 Kurdyka‑Łojasiewicz 收敛性理论。

**📊 数据集**

数值验证采用 Rosenbrock‑型可微无约束目标函数（c=0,1.5,4），未使用公开机器学习数据集，而是通过该函数族在不同非凸度下检验模型表现。

**📈 对比分析**

通过在物理时间 t=kα 处采样离散 Adam 与一阶、二阶连续模型进行对比，实验显示二阶模型在小步长 regime 下与离散算法更贴合；在满足 β₁≤√β₂ 的稳定区间内两种连续模型均保持与离散 Adam 近似的收敛速度，且在非凸双井潜能场中能够准确捕捉到基线切换现象。

**⚠️ 局限性**

局限性包括：需满足 β₁≤√β₂ 的稳定性条件；在短记忆或临界 β 取值下可能出现振荡或不收敛；模型对初始速度敏感，需经验调节；在大规模高维或复杂非凸场景中缺乏严格收敛保证。

---

## 69. Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge

**arXiv ID:** 2602.09341 | [PDF](https://arxiv.org/pdf/2602.09341v1)

**作者:** Wei Yang `[一作]` (University of Southern California), Jesse Thomason `[通讯]` (University of Southern California)

**通讯引用:** 2262 | [OpenAlex ID](https://openalex.org/A5108062941)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AgentAuditor 框架，利用推理树和局部证据审计替代多代理系统中的多数投票；

**💡 创新点**

通过显式构建推理树揭示代理分歧点，采用局部比较而非频数聚合，并引入 Anti‑Consensus Preference Optimization 训练抗多数偏见的审计器；

**🔧 技术方法**

语义分词+嵌入对齐构建推理树、关键分歧点打包、LLM判定器、条件 Beam Search 与 DPO/ACPO 优化；

**📊 数据集**

GSM8K、MATH、AMC、MMLU 等数学与通用推理基准；

**📈 对比分析**

与多数投票、LLM‑as‑Judge 以及多种 MAS 框架对比，AgentAuditor 在多数错误场景提升 5–6% 绝对准确率，整体平均提升约 3%，并显著降低 token 消耗；

**⚠️ 局限性**

受限于推理树构建的语义对齐误差、对极端多分支情形处理不足，以及对极长推理链的全局错误检测能力有限。

---

## 70. AgentCgroup: Understanding and Controlling OS Resources of AI Agents

**arXiv ID:** 2602.09345 | [PDF](https://arxiv.org/pdf/2602.09345v1)

**作者:** Yusheng Zheng `[一作]` (University of California Santa Cruz), Andi Quinn `[通讯]` (University of California Santa Cruz)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化AI编码代理在沙箱容器中执行工具调用的操作系统资源动态，并提出基于eBPF的资源控制器

**💡 创新点**

针对资源粒度、响应速度、可适应性三大匹配缺口，提出层级cgroup与内核级eBPF驱动的实时调度与内存控制

**🔧 技术方法**

使用eBPF、cgroup v2、sched_ext、memcg_bpf_ops等内核技术

**📊 数据集**

对144个SWE-rebench软件工程任务，在Claude Haiku 4.5（云API）和GLM-4.7-Flash（本地GPU）两种LLM模型下采样

**📈 对比分析**

与无服务器、微服务和批处理工作负载对比，实验表明在多租户内存争用下实现29% P95延迟下降，100%任务完成率

**⚠️ 局限性**

仅基于单一代理框架和基准，缺乏实时工作负载验证、对初始化、大镜像和重试累积等问题的完整优化

---

## 71. Scaffolding Metacognition with GenAI: Exploring Design Opportunities to Support Task Management for University Students with ADHD

**arXiv ID:** 2602.09381 | [PDF](https://arxiv.org/pdf/2602.09381v1)

**作者:** Zihao Zhu `[一作]` (City University of Hong Kong), Yuhan Luo `[通讯]` (City University of Hong Kong)

**通讯引用:** 486 | [OpenAlex ID](https://openalex.org/A5048911139)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究基于生成式人工智能（GenAI）如何支持 ADHD 大学生在学业任务管理中的元认知过程，采用个别协同设计会谈与专家访谈方法；

**💡 创新点**

首次将元认知视角与 GenAI 结合，提出三大设计方向（认知支架、反思执行、情绪调节），并结合参与者与专家的实地反馈形成可操作的设计建议；

**🔧 技术方法**

使用当前流行的生成式模型（ChatGPT、Gemini 2.5 Pro 等）作为思路来源与功能原型实现；

**📊 数据集**

共收集 20 名 ADHD 学生的设计方案与 5 名 ADHD 专家的评估意见；

**📈 对比分析**

研究采用定性主题分析而非量化对比，未设置基准模型；通过访谈与编码确认三大设计方向的有效性与可行性；

**⚠️ 局限性**

样本规模有限，且仅来自华语社群，缺乏跨文化验证；研究为探索性，未评估 GenAI 具体算法性能，可能存在技术与伦理风险（如过度依赖、错误信息）

---

## 72. PointAloud: An Interaction Suite for AI-Supported Pointer-Centric Think-Aloud Computing

**arXiv ID:** 2602.09296 | [PDF](https://arxiv.org/pdf/2602.09296v1)

**作者:** Frederic Gmeiner `[一作]` (Autodesk Research and Carnegie Mellon University), Justin Matejka `[通讯]` (Autodesk Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发并实现了PointAloud系统，该系统在建筑设计CAD工具中通过指针中心的思维口述捕获、指针跟踪、自动生成结构化笔记、主题聚类、提醒与AI提示，实现了实时低干扰反馈与人机协作的思维口述计算；

**💡 创新点**

创新点包括：①将用户语音与指针位置同步捕获，利用指针附近的低干扰视觉反馈鼓励口述；②自动生成结构化笔记（TalkNote），并通过语义聚类、标签分类实现工作流程文档化；③提供AI驱动的即时提示（TalkTip）和后续动作建议，形成过程感知的人机协作；④结合指针与语音的多模态上下文，提升AI建议的相关性。

**🔧 技术方法**

技术实现采用实时语音转写（Deepgram Nova‑3）、大语言模型（GPT‑4o、Gemini 2.5 Pro）进行语义分割、摘要、标签分类与提示生成；前端基于React + Three.js + TLDraw构建二维/三维绘图与指针显示；后端使用Node.js/Express处理转写与LLM接口，日志记录与界面交互同步。

**📊 数据集**

数据来源主要为实验室内收集的建筑平面图与3D模型（Polycam社区公开素材）以及12名建筑师/室内设计师在任务中的语音记录；未使用公开大规模文本或语音语料库，所有数据均为实验收集。

**📈 对比分析**

通过与仅提供文本转写的基线系统进行对照实验（12名参与者），采用Likert量表评估思维支持、任务支持、流程意识、认知负荷等指标，并使用Wilcoxon符号秩检验及t检验比较；结果显示PointAloud在思维支持、任务支持、流程意识、回顾与交流方面显著优于基线（p<0.05），认知负荷差异不显著；口语量（WPM）无显著提升。

**⚠️ 局限性**

限制包括：①未能显著增加口语量；②提醒功能效果低，提示多被忽视；③指针位置与用户注意力可能不一致，导致笔记定位不准确；④实验规模小且仅在建筑设计场景，缺乏跨领域验证；⑤长期使用效果与持续学习曲线未评估；⑥转写误差与LLM生成建议的准确性仍需改进。

---

## 73. Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh

**arXiv ID:** 2602.09216 | [PDF](https://arxiv.org/pdf/2602.09216v1)

**作者:** Varchita Lalwani `[一作]` (Plaksha University), Anupam Sobti `[通讯]` (Plaksha University)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5005231699)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对Project Sidewalk进行本土化改造，改进标签体系、示例图片，并引入视觉语言模型生成的即时任务指导，随后在昌迪加尔三大用地类型区块进行40公里人行道可达性审计。

**💡 创新点**

创新点在于结合本土交通环境的标签重构与AI辅助任务指导，将视觉语言模型与众包平台结合，为缺乏标准化人行道的印度城市提供可操作的可达性评估工具。

**🔧 技术方法**

采用Project Sidewalk平台、Gemini视觉语言模型、Google Street View、OSM数据、Foursquare POI数据，并设计基于权重的可达性分数计算。

**📊 数据集**

使用开放街图（OSM）与谷歌街景图像、Foursquare POI数据以及城市人口统计数据。

**📈 对比分析**

通过三名标注员对50条街段的AI指导进行相关性、准确度、实用度的Likert评估，平均得分分别为4.97、4.40、4.61；并对三区不同POI类别的可达性得分进行热图与条形图对比，显示商业区可达性最高、教育与公共服务区最低。

**⚠️ 局限性**

局限在于仅在昌迪加尔单一城市验证，VLM生成指导偶有不准确或不够精细，GSV覆盖不完整导致部分路段缺失标注，且未涉及更细粒度的障碍分类。

---

## 74. Deep Modeling and Interpretation for Bladder Cancer Classification

**arXiv ID:** 2602.09324 | [PDF](https://arxiv.org/pdf/2602.09324v1)

**作者:** Ahmad Chaddad `[一作]` (Guilin University of Electronic Technology), Xianrui Chen `[通讯]` (Guilin University of Electronic Technology)

**通讯引用:** 301 | [OpenAlex ID](https://openalex.org/A5042939085)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对13种深度学习模型（CNN与ViT）在多中心膀胱癌图像分类任务中进行统一实验，评估其分类性能、校准效果与可解释性，并通过测试时增强（TTA）提升模型可解释性。

**💡 创新点**

①在同一医疗图像数据集上系统比较CNN与ViT在泛化、校准和解释性方面的差异；②首次结合多种优化器（SGD、Adam、AdamW、Adagrad、Adadelta）评估其对模型表现的影响；③将GradCAM++与TTA相结合，探讨提升医疗影像模型可解释性的有效策略。

**🔧 技术方法**

使用13个CNN/ViT模型（ConvNeXt、MaxViT、Swin Transformer、ViT等），5种优化器，计算ECE校准误差并绘制可靠性图，利用GradCAM++生成热图进行可解释性分析，并采用TTA（水平/垂直翻转、旋转）进一步增强解释效果；对训练时间进行箱线图比较。

**📊 数据集**

多中心膀胱癌图像数据集（C1–C4共279例，T2加权），二分类：肌层浸润性膀胱癌（MIBC）与非肌层浸润性膀胱癌（NMIBC）。

**📈 对比分析**

采用训练/验证/测试拆分，评估ACC、BACC、PRE、REC、F1及其平均值；通过蛛网图、箱线图及ECE箱线图展示不同优化器下模型的性能与校准差异；实验结果显示ConvNeXt在ID下准确率高但泛化差，ViT校准更好但整体ACC相对低，TTA能提升模型可解释性；MaxViT在训练时间上最快。

**⚠️ 局限性**

①未进行域适应或迁移学习，导致模型在不同中心间泛化不足；②单一模型无法在所有中心实现高准确率与可解释性；③数据量有限，易产生过拟合；④缺少真实临床验证，无法评估模型在临床实践中的实际效用。

---

## 75. Weighted Wasserstein Barycenter of Gaussian Processes for exotic Bayesian Optimization tasks

**arXiv ID:** 2602.09181 | [PDF](https://arxiv.org/pdf/2602.09181v1)

**作者:** Antonio Candelieri `[一作]` (University of Milano-Bicocca), Francesco Archetti `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 2292 | [OpenAlex ID](https://openalex.org/A5011390491)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了加权Wasserstein平均高斯过程（W2BGP）框架，用以统一协作/联邦、批量和多保真度贝叶斯优化任务。

**💡 创新点**

创新点在于：①利用高斯过程后验的高斯分布性质，将Wasserstein平均简化为对一维正态的加权平均；②提供可调权重方案，使同一框架适配三类异构优化任务；③重新诠释常用的采集函数，使其在W2BGP下更高效。

**🔧 技术方法**

技术包括：高斯过程回归、2-Wasserstein距离、加权Wasserstein平均、采集函数改写（LCB、PI、EI）以及权重策略（自信、均等、无合作、按保真度等）。

**📊 数据集**

使用公开的贝叶斯优化基准（1–20维函数、Alpine、Rosenbrock、Rastrigin等）和多保真度基准（Forrester、Rosenbrock、ShiftedRotatedRastrigin、Heterogeneous、Pacioreck）。

**📈 对比分析**

通过AUGC和最佳观测值与传统模型平均和专门算法对比；结果显示自信权重在协作/联邦任务中表现最佳，非合作权重在批量任务中最优，多保真度任务中均衡或等权重更好；整体性能优于单一GP或现有平均方法。

**⚠️ 局限性**

局限性包括：①权重选择需手动设定或基于任务经验，缺乏自动适应机制；②对高维或极端不确定情形的鲁棒性尚未系统验证；③仅针对同步批量与特定协作模式，异步或分布式情形待扩展。

---

## 76. SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes

**arXiv ID:** 2602.09153 | [PDF](https://arxiv.org/pdf/2602.09153v1)

**作者:** Nicholas Pfaff `[一作]` (Massachusetts Institute of Technology), Russ Tedrake `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 14663 | [OpenAlex ID](https://openalex.org/A5074291890)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在自然语言提示下，提出了SceneSmith框架，实现了从场景布局到物体生成再到物理属性预测的全流程自动化室内场景合成，生成可直接用于机器人仿真的三维环境；

**💡 创新点**

核心创新包括：①将场景合成拆分为层级化的设计-评审-编排三代理交互；②集成文本到3D的生成、检索式的关节模型和薄膜覆盖技术，保证生成资产既多样又具备碰撞与质量属性；③通过后处理和物理仿真实现低碰撞率和高静态稳定性；

**🔧 技术方法**

采用视觉语言模型（VLM）驱动的设计师、评审和编排代理，工具集合涵盖场景观测、图像渲染、资产生成、碰撞检测等；资产生成使用文本‑图像模型、SAM3、SAM3D、ArtVIP；后处理使用Drake仿真；

**📊 数据集**

使用了210条多类别室内场景提示（房间级、住宅级、主题化、多样性等），以及公开的Objaverse、AmbientCG等素材库；

**📈 对比分析**

与五大基线（HSM、Holodeck、I-Design、LayoutVLM、SceneWeaver）以及六种消融进行对比。SceneSmith平均生成71个物体，碰撞率<2%，稳定率95.6%，在用户研究中对现实感和符合提示的胜率分别为92%和91%，显著优于所有基线；

**⚠️ 局限性**

局限性包括：对高密度场景时可访问性与导航性略下降；物理稳定性虽好但仍有微小穿插；生成速度相对较慢，且主要针对室内场景，未验证跨场景或室外扩展；

---

## 77. Universal Asymptotics for Jensen--Shannon Divergence under Shuffling

**arXiv ID:** 2602.09029 | [PDF](https://arxiv.org/pdf/2602.09029v1)

**作者:** Alex Shvets `[一作]` `[通讯]` (Independent Researcher), Alex Shvets (Independent Researcher)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

在随机洗牌模型中研究两相邻数据集产生的转录分布的 Jensen–Shannon 散度，并给出其随用户数 n 的显式两项渐近展开；

**💡 创新点**

创新点在于提出一种通用、可计算的两项展开式（第一项为 χ² 散度/8n，第二项为 (7/64)χ⁴ - (1/16)μ₃）并提供完整的误差界定，适用于任意有限输出的局部随机化器；

**🔧 技术方法**

主要技术包括：利用洗牌后直方图的可线性化似然比、i.i.d. 输出的中心极限定理、Taylor 展开、矩方法以及 Hoeffding 型尾部控制；

**📊 数据集**

本研究为理论分析，未使用任何真实数据集；

**📈 对比分析**

通过与二值化随机回应（Binary RR）及 k‑ary 随机回应的特殊实例对比，展示展开式的精确性和收敛速度；

**⚠️ 局限性**

局限性在于需要输出分布全支持（δ_* > 0），且对小样本（n 较小）时误差项可能显著；

---

## 78. AgentSkiller: Scaling Generalist Agent Intelligence through Semantically Integrated Cross-Domain Data Synthesis

**arXiv ID:** 2602.09372 | [PDF](https://arxiv.org/pdf/2602.09372v1)

**作者:** Zexu Sun `[一作]` (Baidu Inc.), Xu Chen `[通讯]` (Renmin University of China)

**通讯引用:** 22091 | [OpenAlex ID](https://openalex.org/A5100385692)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了 AgentSkiller 框架，自动化生成多轮跨域交互数据，用于训练和评估大型语言模型代理。

**💡 创新点**

创新点包括：基于 DAG 的状态机驱动的可恢复流水线、双模型 LLM 架构、跨域融合与统一策略、严格的执行验证与自我修正机制。

**🔧 技术方法**

采用了 LLM 双模型（语义推理 + 代码实现）、LangGraph 与 DAG 并行调度、规划器、自动化测试与修正、语义图与实体关系抽取技术。

**📊 数据集**

生成了约 11,522 条多轮交互轨迹数据，其中单域 5,941 条、跨域 5,581 条，涵盖物流、医疗、金融等多行业。

**📈 对比分析**

通过在 τ‑bench、τ²‑bench 与 ACEBench 上对比，AgentSkiller‑14B 在开源模型中实现 79.1%（τ²‑bench）与 78.0%（ACEBench），在多域性能上与甚至超越部分闭源大模型，单域 71.2% 提升至跨域 79.1%。

**⚠️ 局限性**

局限性：仍依赖 LLM 生成的质量与一致性；跨域推理复杂度高，导致生成成本和验证成本显著；缺乏真实世界部署与长期鲁棒性验证。

---

## 79. Overview of PAN 2026: Voight-Kampff Generative AI Detection, Text Watermarking, Multi-Author Writing Style Analysis, Generative Plagiarism Detection, and Reasoning Trajectory Detection

**arXiv ID:** 2602.09147 | [PDF](https://arxiv.org/pdf/2602.09147v1)

**作者:** Janek Bevendorff `[一作]` (Bauhaus-Universität Weimar), Eva Zangerle `[通讯]` (University of Innsbruck)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文概述了PAN 2026工作坊提出的五个共享任务：Voight-Kampff生成式AI检测、文本水印、多人写作风格分析、生成式剽窃检测以及推理轨迹检测，

**💡 创新点**

创新点包括引入文本水印与推理轨迹检测两大新任务，并通过“builder–breaker”模式增强对抗性评估，

**🔧 技术方法**

主要技术手段为TIRA实验平台提供的Docker容器化评估、对抗性文本扰动与可重复的算法比较，

**📊 数据集**

使用的数据集涵盖现有文本、粉丝小说、多语言医学与人文文献、公开公开的数学与编码推理样本等，

**📈 对比分析**

通过在TIRA平台上统一提交与评估，系统性能可以在多任务、跨语料、对抗性环境下进行客观对比，

**⚠️ 局限性**

局限性在于新任务数据的现实性与多样性尚待完善，评估过程对极端对抗性文本的鲁棒性仍需进一步验证。

---

## 80. Understanding Risk and Dependency in AI Chatbot Use from User Discourse

**arXiv ID:** 2602.09339 | [PDF](https://arxiv.org/pdf/2602.09339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 81. Wearable environmental sensing to forecast how legged systems will interact with upcoming terrain

**arXiv ID:** 2602.09209 | [PDF](https://arxiv.org/pdf/2602.09209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 82. Generalizing GNNs with Tokenized Mixture of Experts

**arXiv ID:** 2602.09258 | [PDF](https://arxiv.org/pdf/2602.09258v1)

**作者:** Xiaoguang Guo `[一作]` (University of Connecticut), Chuxu Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 5434 | [OpenAlex ID](https://openalex.org/A5022275632)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在冻结部署环境下同时兼顾模型拟合、对抗分布迁移和对输入扰动鲁棒性的框架——STEM‑GNN。

**💡 创新点**

创新点在于先从理论分析静态推理的不可逆限值，随后结合实例条件计算(ICC)，通过Mixture‑of‑Experts、向量量化(VQ)令量化离散化以及Lipschitz正则化三种技术共同实现三目标平衡。

**🔧 技术方法**

使用Mixture‑of‑Experts编码器实现输入依赖的多机制覆盖，VQ Token接口将连续表示离散化以吸收小扰动，预测头加入Lipschitz约束限制输出放大。

**📊 数据集**

在九个基准数据集上评估：节点分类(Cora、PubMed、Arxiv、WikiCS)、边预测(知识图谱WN18RR、FB15K‑237)、图分类(HIV、PCBA、ChEMBL)。

**📈 对比分析**

与传统监督模型、预训练方法及专门的OOD/鲁棒性方法对比，STEM‑GNN在清洁数据上的准确率保持竞争力，同时在特征遮蔽、边删除扰动以及度数/同质性分布偏移的OOV测试中均取得最优或次优表现，体现了更佳的三目标平衡。

**⚠️ 局限性**

局限在于专家数量固定、代码簿冻结，可能限制对远域或显著变化场景的自适应能力；未来可探索自适应专家分配与在线代码簿更新。

---

## 83. Learning to Remember, Learn, and Forget in Attention-Based Models

**arXiv ID:** 2602.09075 | [PDF](https://arxiv.org/pdf/2602.09075v1)

**作者:** Djohan Bonnet `[一作]` (Forschungszentrum Jülich), Emre Neftci `[通讯]` (Forschungszentrum Jülich)

**通讯引用:** 4723 | [OpenAlex ID](https://openalex.org/A5060924942)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了固定大小注意力记忆中的上下文学习，提出Palimpsa模型并通过贝叶斯元可塑性实现动态学习、记忆与遗忘；

**💡 创新点**

创新点在于将ICL视作连续学习问题，用贝叶斯框架引入自适应元可塑性注意力机制，证明Mamba2是Palimpsa的特殊情况，并提供从非元可塑模型转为元可塑模型的通用方法；

**🔧 技术方法**

主要技术包括贝叶斯元可塑性注意力、变分推理、贝叶斯梯度下降、线性Transformer/状态空间模型的统一数学框架及可微固定大小注意力更新；

**📊 数据集**

使用了合成MQAR基准、FineWeb‑Edu预训练数据、Wikitext、LAMBADA、PIQA、HellaSwag、WinoGrande、ARC、SIQA等常见语言与commonsense推理数据集；

**📈 对比分析**

通过与Transformer、Gated Deltanet、Mamba2等基线在准确率、困惑度等指标上对比，Palimpsa在MQAR上随序列长度提升性能，760M模型上平均准确率52.27%（比基线高约0.8分），LAMBADA困惑度更低；

**⚠️ 局限性**

局限在于仍受固定大小记忆容量限制，极长上下文仍可能出现信息丢失；实现元可塑性需要额外fine‑tuning阶段；相比纯非元可塑模型，计算与内存开销略高。

---

## 84. BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation

**arXiv ID:** 2602.09383 | [PDF](https://arxiv.org/pdf/2602.09383v1)

**作者:** Peng Lai `[一作]` (Southern University of Science and Technology), Guanhua Chen `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 6527 | [OpenAlex ID](https://openalex.org/A5100665987)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个完全由大语言模型驱动的框架（BiasScope），用于自动化、规模化地在 LLM-as-a-Judge 场景中发现未知偏差；同时基于该框架构建了更具挑战性的 JudgeBench-Pro 评测基准。

**💡 创新点**

创新点包括：① 通过教师模型对拒绝答案进行有针对性的扰动，并用误差级联与深度解释技术挖掘潜在偏差；② 在迭代过程中自动聚合、验证并扩展偏差库；③ 将发现的偏差用于对齐训练，验证其缓解效果；④ 结合人工审核与多模型交叉验证构建高质量的 JudgeBench-Pro。

**🔧 技术方法**

使用的技术主要有：大语言模型（GPT-OSS、Qwen、LLaMA 等）做教师与目标模型；扰动函数 Perturb、误差级联 DeeperExplain；误判筛选与偏差识别 IdentifyBias；聚合 Merge；验证 Verify；DPO 对齐训练；人工标注与多模型一致性检验。

**📊 数据集**

实验数据集包括：JudgeBench（原始评测）、JudgeBench-Pro（扩展版）、RewardBench、RM-Bench、UltraFeedback、SUSTech NLP JudgeBench-Pro 等；教师模型 Qwen3-32B 用于筛选与审核。

**📈 对比分析**

对比方法：在原始数据与注入偏差后数据上分别评估错误率；将 DPO 训练前后模型在 RewardBench 上的误差率对比；与现有基准在 JudgeBench 和 JudgeBench-Pro 上的表现做横向比较。实验显示：即使是强大模型，在 JudgeBench-Pro 上误差率可达 25.9% 或更高，表明框架能显著揭示并验证偏差；利用发现的偏差进行 DPO 训练后，模型在 RewardBench 上的错误率下降，验证了缓解效果。

**⚠️ 局限性**

局限性：① 需要较强的教师模型与算力，导致实验成本较高；② 偏差库的收敛与扩展速度取决于目标模型的表现，弱模型可能需要更多迭代；③ 发现的偏差仍需人工审核，耗时且易受标注者主观影响；④ 部分偏差可能源于教师模型自身的偏差，难以完全消除。

---

## 85. Impact of domain adaptation in deep learning for medical image classifications

**arXiv ID:** 2602.09355 | [PDF](https://arxiv.org/pdf/2602.09355v1)

**作者:** Yihang Wu `[一作]` (Guilin University of Electronic Technology), Ahmad Chaddad `[通讯]` (École de Technologie Supérieure)

**通讯引用:** 3443 | [OpenAlex ID](https://openalex.org/A5024700033)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对10种深度CNN模型在4个医学影像数据集上实施域适应(DANN等)的系统评估，并探讨噪声、联邦学习、可解释性与校准等因素的影响

**💡 创新点**

首次将域适应技术与多模态医学影像、噪声鲁棒性、联邦学习以及可解释性和校准指标结合起来进行全面对比，揭示其在不同任务中的正负迁移特性

**🔧 技术方法**

采用对抗域分类器(DANN)、最大均值散度(MMD)、t-SNE可视化、Grad‑CAM++/LayerCAM等XAI方法，以及ECE校准评估，全部在PyTorch框架下实现

**📊 数据集**

使用HAM10000（皮肤癌）、脑瘤、胸部X‑ray（肺癌）以及四个单一数据集融合的多模态数据集

**📈 对比分析**

将域适应模型与未适应模型按准确率、AUC和ECE比较：脑瘤、脑瘤+多模态数据显著提升≈4–6%准确率，皮肤癌与胸部X‑ray提升有限；域适应在噪声数据中能缓解≈3%准确率下降，校准误差下降约2–6%

**⚠️ 局限性**

存在负迁移现象（如皮肤癌、某些网络架构导致准确率下降）、对小样本/类不平衡数据的适应效果不佳、以及联邦学习环境下适应效果受限

---

## 86. Kyrtos: A methodology for automatic deep analysis of graphic charts with curves in technical documents

**arXiv ID:** 2602.09337 | [PDF](https://arxiv.org/pdf/2602.09337v1)

**作者:** Michail S. Alexiou `[一作]`, Nikolaos G. Bourbakis `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

本文介绍了Elsevier期刊的 LaTeX 模板 elsarticle.cls 的设计与使用，说明其如何在 article.cls 基础上实现多种期刊格式、支持 natbib、hyperref 等常用宏包，并提供完整的安装与使用示例。

**💡 创新点**

其创新点在于基于标准 article.cls 重写，最大限度减少与其它宏包冲突；提供预印本与期刊最终版的多种格式选项；集成了 natbib、geometry、graphicx 等核心宏包，并支持双栏/单栏切换、公式断行自检等高级排版功能。

**🔧 技术方法**

主要技术实现基于 LaTeX 宏包编程，包括 natbib（引用处理）、geometry（页面边距）、graphicx（图形插入）、hyperref（超链接）以及 endfloat（浮动对象排版）等；同时使用了自定义命令与环境来实现定理、列表、跨引用等功能。

**📊 数据集**

无（本文为模板说明文档，不涉及数据集）。

**📈 对比分析**

未进行实验性性能比较；通过示例代码演示不同选项下的排版效果，强调作者需在单栏与双栏模式下自行检查公式断行。

**⚠️ 局限性**

限制：仅适用于 Elsevier 期刊；在双栏打印时需手动检查公式断行；某些旧版宏包可能与其产生冲突；缺少自动化质量检测工具。

---

## 87. Rethinking Global Text Conditioning in Diffusion Transformers

**arXiv ID:** 2602.09268 | [PDF](https://arxiv.org/pdf/2602.09268v1)

**作者:** Nikita Starodubcev `[一作]` (Yandex Research), Dmitry Baranchuk `[通讯]` (Yandex Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了扩散模型中池化文本嵌入的作用，并提出一种基于调制引导的训练‑free 技术来增强文本条件。

**💡 创新点**

创新点在于将池化文本嵌入从被动信息传递转变为主动的调制引导，实现对生成过程的可控方向修正，并且提出了动态调制引导策略。

**🔧 技术方法**

主要技术包括：调制层（modulation layer）、CLIP 文本编码、基于正负提示的调制引导、动态调制尺度、以及对无池化嵌入模型的微调。

**📊 数据集**

实验使用的公开数据集包括：MJHQ、COCO2014、PartiPrompts、CompBench、VBench、SEED‑Data 等，用于评估文本到图像、视频生成和图像编辑。

**📈 对比分析**

与原始模型、CFG、Normalized Attention Guidance、Concept Sliders、LLM‑enhanced prompt 等基线相比，调制引导在多项指标（如 PickScore、CLIP Score、Aesthetics、Complexity、GenEval）上均有提升，且在不增加显著推理开销的情况下实现性能提升。

**⚠️ 局限性**

局限性包括：调制引导在极大提示长度下影响有限；动态策略需额外调参；对部分模型（如 HiDream‑Fast、COSMOS）仅在结合调制引导后才显现优势；在极端复杂编辑场景仍存在欠缺。

---

## 88. Image Quality in the Era of Artificial Intelligence

**arXiv ID:** 2602.09347 | [PDF](https://arxiv.org/pdf/2602.09347v1)

**作者:** Jana G. Delfino `[一作]` (U S Food and Drug Administration), Krishna Juluru `[通讯]` (U S Food and Drug Administration)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文阐述了人工智能在医学影像重建与增强中的优势与局限，强调了AI增强图像可能导致的幻觉、伪影和信息失真，并探讨了不同图像质量评估方法与FDA审批流程。

**💡 创新点**

创新点在于系统性梳理AI增强图像的失败模式、评估方法冲突，以及在临床任务中的表现差异，提醒用户在实际使用中需谨慎权衡视觉质量与诊断信息。

**🔧 技术方法**

使用的技术包括深度学习神经网络（如U‑Net、EnhanceNet）进行图像重建/超分辨率，和常见的定量质量指标（SSIM、MSE、SNR）以及主观Likert评分和任务基础评估。

**📊 数据集**

主要数据集为fastMRI（用于超分辨率实验）和一项临床CT肝转移瘤检测研究（标准剂量FBP与低剂量AI重建对比）。

**📈 对比分析**

比较方法：将AI重建图像与传统方法在定量指标（RMSE、SSIM、SNR）和主观评分上进行对比，同时使用任务基础评估（肝转移瘤检测率）。结果显示AI图像在视觉质量和定量指标上优于传统方法，但在肝转移瘤检测任务中，AI重建导致检测率显著下降（从98.8%降至84.5%）。

**⚠️ 局限性**

局限性包括：AI无法添加病人特异信息、可能产生幻觉或伪影，定量指标与诊断任务不一定一致，评估方法与临床需求不匹配；同时论文未深入探讨如何改进评估指标或开发自动识别AI失败模式的工具。

---

## 89. ALPHA-PIM: Analysis of Linear Algebraic Processing for High-Performance Graph Applications on a Real Processing-In-Memory System

**arXiv ID:** 2602.09174 | [PDF](https://arxiv.org/pdf/2602.09174v1)

**作者:** Marzieh Barkhordar `[一作]` (Simon Fraser University), Alaa R. Alameldeen `[通讯]` (Simon Fraser University)

**通讯引用:** 5129 | [OpenAlex ID](https://openalex.org/A5072086494)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在真实的UPMEM PIM系统上实现并评估了基于稀疏矩阵运算的图算法（BFS、SSSP、PPR），并对SpMSpV内核进行了不同压缩格式与分区策略的系统设计与优化；

**💡 创新点**

首次在真实PIM硬件上系统性地研究线性代数式图处理，展示了分区与压缩格式对性能的决定性影响，并提出了针对PIM的硬件协同设计建议；

**🔧 技术方法**

使用UPMEM的DPU架构、CSR/CSC/COO压缩格式、行/列/二维分区、SpMV/SpMSpV内核、动态内核切换、DMA传输及性能分析工具；

**📊 数据集**

采用公开的大规模稀疏图数据集（如社交网络、Web 链接图等）进行实验；

**📈 对比分析**

与传统CPU和GPU基线进行对比，SpMSpV内核在CPU上分别实现10.2×、48.8×、3.6×的加速，整体算法实现上得到2.6×、10.4×、1.7×的加速，且计算利用率显著高于CPU和GPU；

**⚠️ 局限性**

当前PIM受限于缺乏直接的DPU间通信、指令级并行度不足以及阻塞式DMA引起的计算与内存瓶颈，导致性能受限于数据传输和流水线停滞，需进一步硬件优化以实现更高效的图处理。

---

## 90. DSFlow: Dual Supervision and Step-Aware Architecture for One-Step Flow Matching Speech Synthesis

**arXiv ID:** 2602.09041 | [PDF](https://arxiv.org/pdf/2602.09041v1)

**作者:** Bin Lin `[一作]` (StepFun), Xuerui Yang `[通讯]` (StepFun)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了DSFlow框架，实现一步流匹配语音合成的高效蒸馏与训练，显著降低推理成本与模型参数。

**💡 创新点**

创新点包括双重监督（端点匹配+均值速度对齐）提升训练稳定性；step‑aware token化替代连续时间调制，提升参数效率；弱分类器无指导正则化保留可控性。

**🔧 技术方法**

技术包括流匹配（flow matching）、对抗式蒸馏、JVP-free平均速度计算、Transformer架构（DiT）与自注意力的step‑aware token化、弱CFG正则化。

**📊 数据集**

使用多语言大规模语料Emilia（约95k小时）进行训练，评估在LibriSpeech test‑clean、Seed‑TTS test‑en/zh等公开数据集。

**📈 对比分析**

与多步教师模型、Progressive Distillation、IntMeanFlow、VITS等基线对比；DSFlow在一步推理下自然度、音质与教师相差≤0.2 MOS，参数约24%缩减，RTF显著下降，且多步版本进一步逼近教师性能。

**⚠️ 局限性**

局限包括未在极大规模多语种训练下验证；仅针对Transformer‑style流匹配模型，未探讨自回归或其他架构的适配；对不同数据分布的自适应权重调度仍待研究。

---

## 91. Towards an OSF-based Registered Report Template for Software Engineering Controlled Experiments

**arXiv ID:** 2602.09292 | [PDF](https://arxiv.org/pdf/2602.09292v1)

**作者:** Ana B. M. Bett `[一作]` (State University of Maringá), Marcos Kalinowski `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析 OSF 平台中现有的 Registered Report (RR) 类型，并将其与软件工程控制实验的文档化指南进行映射，提出了符合指南的 RR Stage‑1 模板，并指出现有模板与指南之间的差距。

**💡 创新点**

首次系统评估 OSF RR 类型与软件工程实验指南的契合度，发现 RR.3 最能覆盖指南要求，并提出在 OSF 上针对软件工程控制实验定制 RR 的必要性和方法。

**🔧 技术方法**

使用 OSF 平台提供的 RR 模板、软件工程实验指南文档以及手工映射技术；通过对比分析实现了模板与指南的匹配度评估。

**📊 数据集**

没有使用传统实验数据集；研究数据来源为软件工程实验指南文本以及 OSF RR 类型描述文档。

**📈 对比分析**

通过计数法比较不同 RR 类型覆盖的指南条目数，结果显示 RR.3 覆盖 33/37 条指南，优于 RR.1 的 31/37；说明在指南覆盖度方面性能较高，但仍未完全满足。

**⚠️ 局限性**

局限性：OSF 现有 RR 类型不可定制，缺乏专门针对软件工程的 RR 类型；模板无法完全覆盖所有指南条目；实现过程可能导致审批延迟；对于定性或探索性研究的适用性有限。

---

## 92. Distributed Hybrid Parallelism for Large Language Models: Comparative Study and System Design Guide

**arXiv ID:** 2602.09109 | [PDF](https://arxiv.org/pdf/2602.09109v1)

**作者:** Hossam Amer `[一作]` (Huawei), Boxing Chen `[通讯]` (Huawei)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了大语言模型分布式训练与推理的多种并行策略，包括数据并行、张量并行、流水线并行、上下文并行、专家并行等，并对集体通信、内存优化、通信重叠等技术进行系统分析。

**💡 创新点**

提出了统一的理论分析框架，对不同并行策略的算力、内存、通信成本进行数学建模；给出了多维（3D/4D）混合并行的系统设计指南；结合案例研究展示了自动并行搜索与成本模型的实际效果。

**🔧 技术方法**

采用了集体通信原语（AllReduce、AllGather、ReduceScatter 等）、张量分块、流水线调度、激活检查点、ZeRO、内存卸载、通信重叠、理论 roofline 模型、自动化搜索框架（如 Alpa、Colossal‑AI、Mist 等）以及实测与理论推导相结合的评估方法。

**📊 数据集**

作为综述性质的工作，没有使用特定实验数据集；引用了公开的 LLM 训练与推理实验（如 GPT‑3、Llama、Mixtral 等）中的结果。

**📈 对比分析**

通过对比表格和案例研究，对不同并行策略在吞吐量、延迟、内存占用、通信量等指标进行系统比较；理论分析给出公式，实验验证表明混合并行可在保持低延迟的同时实现高吞吐。

**⚠️ 局限性**

仅为综述，缺乏统一的基准测试；对异构硬件、动态切换、负载不平衡等细节的深入探讨有限；理论模型未覆盖所有实际实现细节，实际部署仍需根据具体平台调优。

---

## 93. Measuring Inclusion in Interaction: Inclusion Analytics for Human-AI Collaborative Learning

**arXiv ID:** 2602.09269 | [PDF](https://arxiv.org/pdf/2602.09269v1)

**作者:** Jaeyoon Choi `[一作]` (University of California), Nia Nixon `[通讯]` (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于话语的“包容性分析”框架，用三维度（参与公平、情感氛围、认知公平）衡量协同问题解决过程中的包容性。

**💡 创新点**

创新点在于将包容性从静态自评转为动态交互度量，并首次将三维度通过可扩展的、无标注的互动层指标实现可量化。

**🔧 技术方法**

使用了词频计数、Gini系数、正面礼貌标记提取、语义向量相似度、指数衰减加权等计算方法，辅以GPT‑5.2生成模拟对话。

**📊 数据集**

数据集包括使用GPT‑5.2生成的模拟团队对话（共100组）和真实人‑AI协作实验数据（37支人类团队、37支人‑AI团队）。

**📈 对比分析**

通过Mann‑Whitney U检验对团队级指标进行比较；在模拟数据中，参与公平指标能显著区分平衡与不平衡情形；在真实数据中，参与公平在词数上差异显著，情感氛围和认知公平在人‑AI组与人类组间存在统计差异，表现出不同的效果。

**⚠️ 局限性**

局限性包括：礼貌度量受匿名聊天环境影响，认知公平指标易受任务主题相似性干扰，模型仍过于简单，难以捕捉细微的情感与认知边缘化；需进一步在多样化情境下验证。

---

## 94. Beyond Uniform Credit: Causal Credit Assignment for Policy Optimization

**arXiv ID:** 2602.09331 | [PDF](https://arxiv.org/pdf/2602.09331v1)

**作者:** Mykola Khandoga `[一作]` (Lexsi Labs), Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对RL强化学习模型进行基于对抗性重要性加权的策略梯度训练

**💡 创新点**

通过掩蔽推断子句重要性并将其作为token级权重来实现更精准的信用分配

**🔧 技术方法**

对抗性重要性评估、Token级加权的DAPO训练、掩蔽占位符、正向推断

**📊 数据集**

GSM8K数学推理数据集（以及在MBPP+代码生成上的消极验证）

**📈 对比分析**

相较于标准DAPO，CF‑DAPO在GSM8K上提升约0.8–1.1个百分点，且收敛更快；在代码生成上无明显改进

**⚠️ 局限性**

计算开销大、仅适用于集中式答案任务、依赖特定span检测且对分布式正确性任务无效

---

## 95. Benchmarking the Energy Savings with Speculative Decoding Strategies

**arXiv ID:** 2602.09113 | [PDF](https://arxiv.org/pdf/2602.09113v1)

**作者:** Rohit Dutta `[一作]` (Indian Institute of Technology), Niloy Ganguly `[通讯]` (Indian Institute of Technology)

**通讯引用:** 6915 | [OpenAlex ID](https://openalex.org/A5073812421)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了多种Speculative Decoding（推测性解码）策略在不同模型、任务与数据集下的能耗表现，并对其与传统自回归解码的速度和能耗关系进行量化比较。

**💡 创新点**

创新点在于首次以能耗为核心度量，揭示推测性解码在降低延迟的同时不一定带来能耗收益，发现模型大小差距、任务特性与实现后端对能效影响显著；并提出能耗节省因子（γ_e^GPU、γ_e^Total）与速度提升（γ_t）之间的线性关系与斜率可因模型对齐度变化。

**🔧 技术方法**

采用NF4 4‑bit量化的Transformer LLM作为目标模型，利用不同助手模型（68M、1B、B级别）与四大模型族（decoder‑only、encoder‑decoder）结合，实施标准SD（固定/动态生成）、SOTA SD（-2、-3）以及HF与vLLM后端实现，并用Code Carbon测量能耗。

**📊 数据集**

数据集覆盖三大任务：代码生成、数学推理和文本摘要，分别使用对应的公开数据集（每个任务采样256个或164个prompt）。

**📈 对比分析**

比较指标包括速度提升因子γ_t、GPU能耗节省因子γ_e^GPU、总能耗节省因子γ_e^Total；实验显示在部分模型-助手对（如13B+68M、70B+1B）与特定任务（如代码生成）上可实现最高2.5×的能耗节省，但在70B+XL、4B+0.6B等组合中速度提升并未转化为能耗收益，说明能效提升高度依赖模型与任务匹配。

**⚠️ 局限性**

主要局限为：实验仅在单一A5000/A6000 GPU环境下完成，未覆盖边缘设备或多GPU集群；仅使用batch size为1的推理，未验证更大批次下的能耗与速度表现；因此结果对不同硬件与生产环境的泛化性有限。

---

## 96. One RNG to Rule Them All: How Randomness Becomes an Attack Vector in Machine Learning

**arXiv ID:** 2602.09182 | [PDF](https://arxiv.org/pdf/2602.09182v1)

**作者:** Kotekar Annapoorna Prabhu `[一作]` (Purdue University), Zahra Ghodsi `[通讯]` (Purdue University)

**通讯引用:** 2348 | [OpenAlex ID](https://openalex.org/A5042124571)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统性分析机器学习框架中随机数生成的实现与安全缺陷，并提出了一个名为RNGGuard的工具，用于静态检测和运行时强制使用安全随机数源。

**💡 创新点**

将静态代码分析与运行时随机数检测结合，形成双模式（静态/动态）保障机制，并为隐私强化的机器学习提供可验证的安全随机数策略。

**🔧 技术方法**

CodeQL静态分析、Python动态函数替换、CSPRNG实现、KS/χ²统计测试、GPU/CPU异步审计等。

**📊 数据集**

以CIFAR‑10为基准训练ResNet20，评估DP‑SGD（ε,δ）下的训练性能。

**📈 对比分析**

与PyTorch原生实现对比，静态模式约增幅47%（非DP）或16.5%（DP），动态模式最初增幅高达650%，通过异步与采样优化可降至约20%（DP）。

**⚠️ 局限性**

需要手动识别核心PRNG，运行时测试仍带来显著开销，且仅在Python生态验证，跨框架自动化尚待完善。

---

## 97. Triggered: A Statistical Analysis of Environmental Influences on Extremist Groups

**arXiv ID:** 2602.09289 | [PDF](https://arxiv.org/pdf/2602.09289v1)

**作者:** Christine de Kock `[一作]` (University of Melbourne), Eduard Hovy `[通讯]` (University of Melbourne)

**通讯引用:** 43048 | [OpenAlex ID](https://openalex.org/A5060225743)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文系统性评估了暴力事件、新闻报道与跨群体语言传播对两个极端主义社区（Stormfront 与 Incels）以及主流社区的影响；

**💡 创新点**

创新点在于同时使用对照合成法与向量自回归（VAR）+ Granger 因果检验，三种方法互补并在同一框架下比较三大社区的响应差异；

**🔧 技术方法**

采用的技术包括结构时序模型的对照合成、VAR‑X 模型、Granger 因果检验、单位根检验、FDR 多重检验校正以及残差诊断（Ljung‑Box、ADF、KPSS）；

**📊 数据集**

使用的数据集为 2018–2024 年 Stormfront、Incels 与 Reddit 子版块的 19.6M 条帖子、GDELT 全球新闻事件数据以及 36 起与极端主义相关的暴力事件清单；

**📈 对比分析**

通过对三种方法的统计显著性和置信区间比较，Stormfront 对事件与新闻极为敏感，Incels 仅对极少数事件有显著反应，主流社区在对暴力事件的响应上甚至超过两极端主义社区；

**⚠️ 局限性**

局限性包括事件窗口短导致难以分离重叠事件效应、模型对时间序列非平稳性的依赖、新闻数据可能存在标签误差、仅覆盖英语数据、无法捕捉非量化的语义变化与潜在共同冲击。

---

## 98. The Laplacian Mechanism Improves Transformers by Reshaping Token Geometry

**arXiv ID:** 2602.09297 | [PDF](https://arxiv.org/pdf/2602.09297v1)

**作者:** Yuchong Zhang `[一作]` (University of Toronto), Vardan Papyan `[通讯]` (University of Toronto)

**通讯引用:** 1384 | [OpenAlex ID](https://openalex.org/A5018100130)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在Transformer中引入Laplacian机制，通过直接调节token方差来改进模型表现。

**💡 创新点**

创新点在于将注意力中的token与其上下文均值之差（即Laplacian）作为新的注意力头，实现对方差的直接控制，并发现其促成了Neural Token Collapse（NTC）几何结构。

**🔧 技术方法**

技术手段包括：Laplacian head替换部分传统注意力头；PCA、ANOVA、余弦相似度与Neural Collapse指标等多种工具用于分析token几何；以及对不同head数量k的实验设置。

**📊 数据集**

使用的数据集涵盖视觉分类（CIFAR‑10、CIFAR‑100、ImageNet）和语言任务（ARC‑Easy、ARC‑Challenge、MMLU、GSM8K、HumanEval）等。

**📈 对比分析**

通过在ViT‑B、GPT‑2等基线模型中逐步增添k个Laplacian头，使用同一训练配置与超参数，实验表明在所有基准上均可提升1–3%（视觉）和约10–20%（语言）性能，且在多任务平均分数上表现最好。

**⚠️ 局限性**

局限性包括：仅在监督分类和自回归任务上验证，对自监督学习的适用性未知；在大规模模型或不同任务中的实际效果与计算成本尚未系统评估。

---

## 99. RAPID: Risk of Attribute Prediction-Induced Disclosure in Synthetic Microdata

**arXiv ID:** 2602.09235 | [PDF](https://arxiv.org/pdf/2602.09235v1)

**作者:** Matthias Templ `[一作]` (University of Applied Sciences and Arts Northwestern Switzerland), Roman Müller `[通讯]` (University of Applied Sciences and Arts Northwestern Switzerland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种名为 RAPID 的基于攻击者在合成数据上训练模型后评估真实个体敏感属性推断风险的新度量。

**💡 创新点**

创新点在于通过与基线（类别分布或无信息预测）对比的归一化置信度或相对误差阈值，实现了对连续与分类敏感属性的统一、阈值可调且对类别不平衡鲁棒的风险评估。

**🔧 技术方法**

使用了机器学习预测模型（随机森林、梯度提升、逻辑回归等）、概率归一化、相对误差阈值、bootstrap 置信区间以及阈值曲线等技术。

**📊 数据集**

在 EU‑SILC 生成的全合成人口数据、UCI Adult 数据集以及自定义健康微数据模拟实验中进行验证。

**📈 对比分析**

通过与传统匹配/模型基准、距离最近记录等方法对比，RAPID 能清晰展示风险随关联强度和阈值变化的 S 形曲线；在高关联情形下风险可高达 0.97；在 Adult 数据默认阈值下约 72% 的记录被认为有风险，说明合成数据在保持高预测性能的同时存在显著属性推断风险。

**⚠️ 局限性**

限制包括需人工设定阈值 τ/ε、风险评估依赖于所选攻击模型、目前仅处理单一敏感属性、缺乏正式隐私保证以及模型透明度不如基于表格的传统方法。

---

## 100. Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling

**arXiv ID:** 2602.09084 | [PDF](https://arxiv.org/pdf/2602.09084v1)

**作者:** Ruijie Ye `[一作]` (Brown University), Zhengzhong Tu `[通讯]` (Texas A&M University)

**通讯引用:** 2357 | [OpenAlex ID](https://openalex.org/A5015173810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Agent Banana的多智能体层感知编辑框架，支持在4K原生分辨率下进行多轮、语义精细的图像编辑；

**💡 创新点**

核心创新包括：1）Context Folding（层级记忆压缩）以解决长序列上下文溢出；2）Image Layer Decomposition（局部层裁剪+融合）以保持非目标区域细节；3）构建HDD‑Bench高分辨率多轮对话评测基准；

**🔧 技术方法**

结合GPT‑5‑mini LLM、视觉工具（Diffusion、Inpainting、Masking等）、MCP协议、Gaussian混合、Otsu掩码评估与符号状态推理；

**📊 数据集**

使用HDD‑Bench（11.8M像素4K图像，3轮编辑任务）以及ImgEdit‑Bench和其他公开基准进行实验；

**📈 对比分析**

在HDD‑Bench上，Agent Banana在指令遵循、背景保持（IC≈0.871）、感知一致性（SSIM_OM≈0.84、LPIPS_OM≈0.12）等指标均优于Flux.1 Kontext、Nano Banana Pro等基线，单轮任务亦保持SOTA表现；

**⚠️ 局限性**

局限性包括：对LLM的强依赖（较弱模型性能大幅下降）、对工具集的依赖（需多模型协同），以及在极端多目标或极长序列情况下仍可能出现累积漂移（Prior‑Induced Editing Drift）或不可逆错误。

---

## 101. Empowering Contrastive Federated Sequential Recommendation with LLMs

**arXiv ID:** 2602.09306 | [PDF](https://arxiv.org/pdf/2602.09306v1)

**作者:** Thi Minh Chau Nguyen `[一作]`, Quoc Viet Hung Nguyen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出LUMOS框架，在联邦顺序推荐中利用本地LLM生成未来、同义和反事实序列视图，进行三视角对比学习，提升表示质量。

**💡 创新点**

创新点是将LLM作为本地语义生成器，用生成的序列视图做对比学习，既增强了数据稀疏时的自监督信号，又保持了严格的隐私。

**🔧 技术方法**

使用技术包括联邦学习（FedAvg）、大语言模型生成、三视角对比损失（InfoNCE）以及SASRec/GRU4Rec等序列编码器。

**📊 数据集**

使用的公开数据集为Amazon Cell Phone、Amazon Baby和MIND新闻推荐数据集。

**📈 对比分析**

与集中式SASRec、FedSeqRec以及对比式FedSRS进行比较，LUMOS在HR@20和NDCG@20上分别提升约6‑7%和1%，甚至超过集中式基线。

**⚠️ 局限性**

局限性包括依赖本地LLM的计算和存储成本、对LLM生成质量的敏感性以及在极端设备资源受限环境下的部署挑战。

---

## 102. Efficient Distance Pruning for Process Suffix Comparison in Prescriptive Process Monitoring

**arXiv ID:** 2602.09039 | [PDF](https://arxiv.org/pdf/2602.09039v1)

**作者:** Sarra Madad `[一作]` `[通讯]` (Université de Technologie de Troyes), Sarra Madad (Université de Technologie de Troyes)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对保守式过程监控中的后缀检索，提出了一种利用三角不等式和优化枢轴集合进行距离剪枝的高效方法。

**💡 创新点**

创新点在于将三角不等式下界/上界与k-center枢轴选择相结合，实现无损、完全可并行的后缀比较剪枝，同时通过批量化处理显著降低计算量。

**🔧 技术方法**

采用距离矩阵预计算、三角不等式剪枝、k-center枢轴选择（贪心远点法）、并行化批处理以及保守式过程监控框架。

**📊 数据集**

使用约150,000个后缀的工业事件日志作为实验数据集。

**📈 对比分析**

与完整暴力比较（约89小时）对比，改进方案在500个后缀的批次上仅需2.5小时，剪枝保持100%准确率，且完全可并行。

**⚠️ 局限性**

局限性包括：预计算距离矩阵的时间与存储开销、枢轴选择的近似可能在极端分布下效果下降，以及方法仅适用于满足三角不等式的距离度量。

---

## 103. Fully Differentiable Bidirectional Dual-Task Synergistic Learning for Semi-Supervised 3D Medical Image Segmentation

**arXiv ID:** 2602.09378 | [PDF](https://arxiv.org/pdf/2602.09378v1)

**作者:** Jun Li `[一作]` (Southwest Jiaotong University), Jun Li `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 26444 | [OpenAlex ID](https://openalex.org/A5100362041)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种全可微分的双向协同学习框架（DBiSL），通过双向任务变换器实现分割与距离回归任务的互相监督和一致性正则化，从而在医学图像分割的半监督学习中实现更高效、更精准的模型训练。

**💡 创新点**

创新点主要包括：
- 设计了全可微分的双向任务变换器，使得分割→回归和回归→分割的梯度能够在线、双向流动，弥补了以往单向协同方法的不足；
- 在同一框架中统一集成了监督学习、一致性正则化、伪标签生成和不确定度估计，形成一个闭环的全SSL体系；
- 通过在线卷积实现3D距离变换，解决了传统离线距离变换导致的信息不对称和GPU内存瓶颈问题；
- 通过统一处理有标注与无标注样本，提升了数据利用效率。

**🔧 技术方法**

使用的技术包括：
- 3D U-Net/V‑Net 等编码器–解码器骨干网络；
- 双向任务变换器（利用3D卷积实现概率图→距离图及其反向）
- Dice + CE 损失、MSE 损失；
- 伪标签平均与置信度掩码的投票机制；
- 交叉任务一致性与同一任务内部一致性正则化；
- 训练采用SGD、学习率衰减、滑动窗口采样。

**📊 数据集**

主要使用的数据集有：
- 左心房（LA）数据集（100份3D MRI）；
- 胰腺-CT数据集（82份CT扫描）；
- BraTS2019脑肿瘤数据集（335份多模态MRI）。

**📈 对比分析**

与多种主流半监督分割方法（如DTC、DUWM、SASSNet、CauSSL、MRPL等）做对比。实验结果表明：
- 在LA上，Dice从基线82.36%提升至90.54%；
- 在胰腺-CT上，Dice从基线81.69%提升至81.09%；
- 在BraTS2019上，Dice从基线85.13%提升至85.09%。
- 综上，DBiSL在Dice、95HD和ASD等指标均取得或接近当前SOTA，尤其在低标注比例（5%）下仍保持优势。

**⚠️ 局限性**

主要局限：
- 仅针对单目标（全肿瘤或单器官）进行评估，未涉及多类分割、部分标注或多模态缺失等更复杂场景；
- 对大模型或更高分辨率的GPU资源仍有一定需求，虽然已降低至可接受水平；
- 变换器近似距离变换，可能在极细粒度边界处产生小的误差，虽然总体影响可接受。

---

## 104. Synthetic Reflections on Resource Extraction

**arXiv ID:** 2602.09299 | [PDF](https://arxiv.org/pdf/2602.09299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 105. Stabilizing Physics-Informed Consistency Models via Structure-Preserving Training

**arXiv ID:** 2602.09303 | [PDF](https://arxiv.org/pdf/2602.09303v1)

**作者:** Che-Chia Chang `[一作]` (National Yang Ming Chiao Tung University), Chieh-Hsin Lai `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2542 | [OpenAlex ID](https://openalex.org/A5109952763)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种物理信息强化的一致性模型框架（sCM-PINN），实现对 PDE 的快速生成、前向求解和条件/无条件采样。

**💡 创新点**

创新点包括：① 两阶段训练策略（先学习分布后冻结系数解码器进行物理微调），② 结构保持的分解解码器避免系数模式崩溃，③ 双步残差约束在一致性轨迹上施加物理一致性，④ 零射线投影式前向推断实现无梯度前向求解。

**🔧 技术方法**

使用技术包括：一致性模型（sCM）与 TrigFlow 参数化、U‑Net 结构、AdaGN/Adaptive Double Normalization、冻结系数解码器、两步残差约束、SA‑PINN 自适应加权、零射线投影推断、DiffusionPDE 预训练、DDPM++ 及其改造。

**📊 数据集**

使用 DiffusionPDE 预生成的 128×128 网格数据集，共 50,000 组样本，分别对应 Darcy 流、Poisson 方程和 Helmholtz 方程。

**📈 对比分析**

与基线 sCM、DiffusionPDE 等比较：在前向求解中，sCM-PINN 在 16–64 次 NFE 下相较 DiffusionPDE 速度提升 8×以上，且相对 H¹ 误差更低；在条件重建中误差显著下降；在无条件采样中 PDE 残差最低，仅需 2 次 NFE，显著减少计算成本。

**⚠️ 局限性**

局限性：仅验证了稳态椭圆 PDE，未扩展到时间依赖或更高维问题；两阶段训练和分解解码器增加模型设计与训练复杂度；对高频残差仍有一定敏感性。

---

## 106. Global Protocols under Rendezvous Synchrony: From Realizability to Type Checking

**arXiv ID:** 2602.09197 | [PDF](https://arxiv.org/pdf/2602.09197v1)

**作者:** Elaine Li `[一作]`, Felix Stutz `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究在rendezvous同步网络模型下的全局协议（global protocol）的可实现性（realizability）问题，并给出其判定的复杂度以及一个用于检查π-计算机进程与本地规范（SCSM）兼容性的类型系统。

**💡 创新点**

① 证明当并发字母表的独立关系是可传递（transitive）时，全局协议的同步可实现性是可判定的，并给出了EXPTIME的上界；② 证明对“无歧义”（unambiguous）全局协议，全局协议可实现性可在EXPTIME内判定；③ 给出一般情况下同步可实现性是不可判定的；④ 提出了首个支持混合选择（mixed choice）、会话交叉（interleaving）和委托（delegation）的同步类型系统。

**🔧 技术方法**

使用Mazurkiewicz轨迹理论、可识别轨迹语言、无歧义轨迹语言以及多项式时间/指数时间的自动机构造；对可实现性问题的判定采用对称子集构造、平衡性与并发子集理论；类型系统基于SCSM的语义与结构化并发语法。

**📊 数据集**

无数据集；所有结果均为理论性的复杂度分析与可判定性证明。

**📈 对比分析**

与先前针对异步模型或受限分支（如sender‑driven）的可实现性判定方法相比，本工作提供了更广泛的可实现性判定（可传递字母表、无歧义协议）和更强大的类型系统。复杂度方面：可传递情形为EXPTIME（即2‑指数级），无歧义情形同样为EXPTIME；对异步模型常见的PSPACE/NP结果而言，本工作处于更高的复杂度，但覆盖范围更广；在不可判定情形下明确指出同步可实现性不可判定。

**⚠️ 局限性**

① 仅对并发字母表可传递或无歧义的协议可判定；② 复杂度高（EXPTIME或更高），对大规模协议实用性有限；③ 类型系统仅支持单会话（sinks），多会话间的交叉与委托仅在理论层面讨论；④ 证明多项式与指数构造主要在理论上可实现，实际实现尚待验证。

---

## 107. Gencho: Room Impulse Response Generation from Reverberant Speech and Text via Diffusion Transformers

**arXiv ID:** 2602.09233 | [PDF](https://arxiv.org/pdf/2602.09233v1)

**作者:** Jackie Lin `[一作]` (University of Illinois Urbana-Champaign), Paris Smaragdis `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 8910 | [OpenAlex ID](https://openalex.org/A5038903729)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于扩散变换器的模型 Gencho，用来从混响语音中盲估计房间冲击响应。

**💡 创新点**

创新点在于：①使用结构感知编码器分离早期反射与晚期尾部以获得更稳健的条件信息；②采用扩散生成器实现多模态、可多样化的冲击响应生成；③与传统回归模型融合的混合方式提升了早期反射与尾部的准确度。

**🔧 技术方法**

使用技术包括：时域卷积层 + 归一化的结构感知编码器、复杂频谱表示、v‑prediction 的扩散 Transformer、分类器无指导（classifier‑free guidance）、自注意力与交叉注意力、Flan‑T5 文本编码器（用于文本‑到‑RIR）。

**📊 数据集**

数据集涵盖了 OpenSLR、MIT IR Survey、EchoThief、Arni、dEchorate、ACE、OpenAIR、BUTReverbDB 以及 LibriTTS‑R 等多源房间冲击响应和语音数据，并通过数据增强提升多样性。

**📈 对比分析**

与 FiNS 等基准回归模型在未见测试集（BUTReverbDB+OpenAIR）进行对比，Gencho 在 T60 误差最小（13.6% PAE）且整体分布更贴近真实 RIR，混合方案在 EDT、DRR 上进一步提升；但在早期反射的细节上仍略逊于纯回归模型。

**⚠️ 局限性**

主要限制包括：扩散模型固有的随机性导致早期反射可能过度变异；模型仅支持 1 s 长度的冲击响应；对语音增强分离的依赖可能在极端嘈杂场景下影响估计质量。

---

## 108. The SJTU X-LANCE Lab System for MSR Challenge 2025

**arXiv ID:** 2602.09042 | [PDF](https://arxiv.org/pdf/2602.09042v1)

**作者:** Jinxuan Zhu `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Microsoft Research)

**通讯引用:** 27321 | [OpenAlex ID](https://openalex.org/A5100701572)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于BS‑RoFormer的分阶段音乐源恢复系统，通过先去噪、再分离、最后去混响的流程实现8种乐器的源恢复。

**💡 创新点**

创新点在于将任务拆分为单独模块并针对每个乐器训练独立的BS‑RoFormer模型，结合随机混合与长时段训练策略，同时仅对人声使用去混响，以提升在真实环境中的恢复效果。

**🔧 技术方法**

使用了BS‑RoFormer及其Mel‑Band RoFormer变体，采用多尺度STFT+L1损失，数据增强包括随机混合与10秒长段训练，并在NVIDIA H200 GPU上进行大规模训练。

**📊 数据集**

数据集方面结合RawStems与MoisesDB进行清洗与扩充，验证使用MSRBench，训练时对原始48kHz音频进行重采样至44.1kHz。

**📈 对比分析**

与赛道其他参赛队伍在MMSNR、Zimt、FAD三大客观指标及MOS主观指标进行比较，系统在所有指标上排名第一，MMSNR 4.4623、FAD 0.1988，MOS总分3.4665。

**⚠️ 局限性**

局限性在于任务仍以分离精度为主，对恢复质量关注不足，且8种乐器的高复杂度使得模型难以进一步提升恢复效果，未来可通过简化目标乐器数目来聚焦恢复质量。

---

## 109. Risk-Aware Obstacle Avoidance Algorithm for Real-Time Applications

**arXiv ID:** 2602.09204 | [PDF](https://arxiv.org/pdf/2602.09204v1)

**作者:** Ozan Kaya `[一作]` (Norwegian University of Science and Technology), Ingrid Bouwer Utne `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 22006 | [OpenAlex ID](https://openalex.org/A5012692888)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种结合贝叶斯网络风险建模与RA‑RRT*路径规划的混合风险感知航行框架，用于自主水面船舶在不确定海洋环境中的安全导航。

**💡 创新点**

创新点在于将贝叶斯网络产生的概率风险场直接嵌入RRT*采样、成本评估与重连过程，实现对动态与静态障碍的主动避险，并通过α权重实现路径长度与安全性的可调权衡。

**🔧 技术方法**

主要技术包括贝叶斯网络（PySMILE/GeNIe）、风险感知RRT*（RA‑RRT*）规划、B‑spline轨迹平滑以及基于风险地图的采样与重连策略。

**📊 数据集**

使用仿真海域数据集，包含多种静态与动态障碍物（未给出公开数据集，全部为实验室仿真场景）。

**📈 对比分析**

与传统RRT*及纯最短路径规划对比，实验显示RA‑RRT*在同等路径长度下可将累计风险概率下降至一半以上，α=0.2时既保持较短路径又显著降低风险，性能优于仅考虑几何约束的方法。

**⚠️ 局限性**

局限性包括：风险模型仅涵盖有限的影响因子（如距离岸、深度、DCPA等），未考虑更复杂的环境扰动与系统约束；方法为次优采样，不保证全局最小风险；缺乏真实船舶实验验证。

---

## 110. A Deep Multi-Modal Method for Patient Wound Healing Assessment

**arXiv ID:** 2602.09315 | [PDF](https://arxiv.org/pdf/2602.09315v1)

**作者:** Subba Reddy Oota `[一作]` (Woundtech Innovative Healthcare Solutions), Manish Gupta `[通讯]` (Microsoft AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过深度多模态方法预测患者住院风险，结合创面图像与临床变量。

**💡 创新点**

创新点在于将预训练CNN提取的五个创面变量与临床填充变量融合，用LightGBM构建住院风险预测模型。

**🔧 技术方法**

采用预训练Xception网络微调、CNN多任务学习以及LightGBM二分类器。

**📊 数据集**

使用包含20种溃疡类型、80%为5种主类的创面图像数据集，并整合临床填写的16个变量。

**📈 对比分析**

与人工专家对比，模型在住院风险分类上取得精确率0.68、召回率0.91、F1 0.78，且对完成治疗类别精确率0.99。

**⚠️ 局限性**

限制包括样本不均衡、遮挡、光照差异、数据量不足导致模型过拟合以及未提供外部验证。

---

## 111. Not-in-Perspective: Towards Shielding Google's Perspective API Against Adversarial Negation Attacks

**arXiv ID:** 2602.09343 | [PDF](https://arxiv.org/pdf/2602.09343v1)

**作者:** Michail S. Alexiou `[一作]` (Kennesaw State University), J. Sukarno Mertoguno `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 91 | [OpenAlex ID](https://openalex.org/A5039718380)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在现有机器学习毒性检测模型（如Perspective API、LSTM+CNN、BERT+Bi‑LSTM）外层加入形式化推理包装，分别在前处理和后处理阶段对句子中出现的否定词进行逻辑处理或词义反转，以提升对否定攻击的鲁棒性。

**💡 创新点**

创新点在于：①提出将形式化推理与统计学习融合的混合方法；②设计两类变体——（a）使用逻辑反转并按词数或模型自估权重重新加权毒性分数；（b）对否定词替换其反义词并生成多样化的同义句后平均毒性分数，兼顾语义保留与计算可行性。

**🔧 技术方法**

主要技术包括：词性标注（Stanford POS）、句法递归分析、词义消歧（Lesk算法）、反义词检索（Thesaurus/OneLook）、同义句生成（Parrot paraphraser）、多模型集成与加权平均。

**📊 数据集**

使用的评估数据集：Hosseini的否定攻击样本（9句）、由Jigsaw公开/私有毒性数据集生成的否定攻击集（公私各382/418句），以及Perspective API自带的毒性评分数据。

**📈 对比分析**

与单纯Perspective、LSTM+CNN、BERT+Bi‑LSTM等基线比较，混合方法M1.2和M1.5在公/私集上的准确率分别提升至82%/87%，相较于Baseline的0.3%显著提升；在Hosseini样本上平均毒性分数下降30%以上，显示混合方法在处理否定攻击时性能优越。

**⚠️ 局限性**

局限性包括：①SE方法（反义词+同义句生成）计算复杂度为O(5ⁿ)，对多否定词句子耗时大，难以实现实时检测；②仍依赖Perspective API，若API本身对新词或俚语预测不准，混合方法效果受限；③对特殊俚语或无反义词的词（如“shit”）处理效果不佳。

---

## 112. Boltzmann Reinforcement Learning for Noise resilience in Analog Ising Machines

**arXiv ID:** 2602.09162 | [PDF](https://arxiv.org/pdf/2602.09162v1)

**作者:** Aditya Choudhary `[一作]` (Sandia National Laboratories), Prasad Iyer `[通讯]` (Sandia National Laboratories)

**通讯引用:** 1000 | [OpenAlex ID](https://openalex.org/A5039525525)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种基于策略梯度的分布学习框架 BRAIN，用于在受测量噪声影响的模拟 Ising 机器（AIM）上高效逼近 Boltzmann 分布，并完成基准优化与采样实验。

**💡 创新点**

创新点包括：① 将噪声视为学习信号，利用 REINFORCE + baseline 充分利用多次嘈杂测量；② 采用完全因子化的 Bernoulli 变分分布，将 O(2^N) 采样问题压缩到 O(N) 可训练参数；③ 在噪声环境下实现 98% 地面态保真率和 192–408 倍的求解速度提升；④ 在多种拓扑（Curie‑Weiss、2D 邻域 Ising 等）和噪声水平下展示可扩展性（O(N^1.55)）。

**🔧 技术方法**

技术手段包括：策略梯度（REINFORCE）与 baseline 降方差、逆 KL 损失最小化、低维 Bernoulli 变分采样、数字化快速参数更新、对比 MCMC 与并行温度等传统采样算法。

**📊 数据集**

使用的“数据集”主要是人工构造的 Ising 组合优化实例：双阱连续势、6 维 Ising、Curie‑Weiss（全连接）和 2D 邻域 Ising（格点）等，未使用公开真实数据集。

**📈 对比分析**

与传统 MCMC（含并行温度）在不同噪声水平（1–40%）进行对比；BRAIN 在 3% 噪声下地面态保真率达 98% 远高于 MCMC 的 51%，且时间（能量评估次数）提升 192–408 倍；在 65,536 维系统上维持 O(N^1.55) 的可扩展性，验证了在大规模噪声环境下的高效性。

**⚠️ 局限性**

局限性：① 因子化 Bernoulli 近似忽略了自旋间的高阶相关，可能在高度互相排斥或复杂能量地形中性能下降；② 主要针对高斯噪声，对非高斯或系统性偏置、时间相关噪声缺乏鲁棒性；③ 需要进一步引入更表达性模型（如图卷积、因子图）以兼顾实时低延迟与复杂拓扑。

---

## 113. K-Sort Eval: Efficient Preference Evaluation for Visual Generation via Corrected VLM-as-a-Judge

**arXiv ID:** 2602.09411 | [PDF](https://arxiv.org/pdf/2602.09411v1)

**作者:** Zhikai Li `[一作]` (Institute of Automation, Chinese Academy of Sciences), Kurt Keutzer `[通讯]` (University of California)

**通讯引用:** 37962 | [OpenAlex ID](https://openalex.org/A5047285420)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种利用视觉语言模型（VLM）进行视觉生成模型评估的框架——K‑Sort Eval；

**💡 创新点**

通过后验修正（Posterior Correction）和动态匹配（Dynamic Matching）两大创新，提升了VLM判定与人类偏好的一致性与评估效率；

**🔧 技术方法**

使用贝叶斯推断、Spearman相关系数、正态分布建模、阈值筛选以及Llama Guard等技术；

**📊 数据集**

基于K‑Sort Arena收集的高质量人类投票数据集，包含图像与视频两类生成模型的对比实例；

**📈 对比分析**

与K‑Sort Arena以及传统评估方法（FID、CLIP‑based scoring等）比较，K‑Sort Eval在保持与人类偏好高度一致的同时，评估所需模型运行次数平均不到90次，显著提升了效率；

**⚠️ 局限性**

仍受VLM自身幻觉与偏差的限制，对极端多样化场景的适应性尚待进一步验证。

---

## 114. In-Hospital Stroke Prediction from PPG-Derived Hemodynamic Features

**arXiv ID:** 2602.09328 | [PDF](https://arxiv.org/pdf/2602.09328v1)

**作者:** Jiaming Liu `[一作]` (Nanjing University of Aeronautics and Astronautics), Daoqiang Zhang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 18454 | [OpenAlex ID](https://openalex.org/A5018821033)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用重症监护病房持续监测的PPG信号，在卒中发生前预测住院卒中

**💡 创新点**

首次将LLM辅助的临床文本时间提取与PPG生理特征相结合，实现前驱卒中早期预警

**🔧 技术方法**

采用LLM进行时间点抽取、PPG特征工程及ResNet‑1D深度学习模型

**📊 数据集**

使用MIMIC‑III和MC‑MED两大ICU数据库

**📈 对比分析**

在MIMIC‑III内4/5/6小时预测窗口F1分别为0.80/0.88/0.94，外部MC‑MED零样本下F1最高0.99，显示强泛化

**⚠️ 局限性**

样本量有限、缺乏卒中亚型标注、仅为回顾性研究，尚需前瞻验证

---

## 115. Fine-T2I: An Open, Large-Scale, and Diverse Dataset for High-Quality T2I Fine-Tuning

**arXiv ID:** 2602.09439 | [PDF](https://arxiv.org/pdf/2602.09439v1)

**作者:** Xu Ma `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 31253 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了一个规模达600多万条高质量、分辨率高、文本图像对齐度高的文本到图像（T2I）微调数据集 Fine‑T2I，并提供完整的数据构建流水线与微调实验；

**💡 创新点**

创新点包括：① 结合高分辨率合成图像与专业摄影师真实图像，覆盖10种任务组合、32种提示类别、11种视觉风格；② 引入语义去重、属性对齐过滤、增强提示、生成+挑选、VLM推理对齐校验等多层过滤策略；③ 将合成与真实数据统一标注并公开完整构建流程；

**🔧 技术方法**

技术手段涵盖：使用 LLaMA3 生成提示、微调提示增强模型、Z‑Image/FLUX2 合成图像、Aesthetic Predictor V2.5 质量筛选、Qwen3‑VL‑8B‑Instruct 进行属性对齐验证、VLM 推理校验文本图像对齐、LoRA 与全量微调分别在 SD‑XL 与 LlamaGen 上进行实验；

**📊 数据集**

核心数据集为 Fine‑T2I（约 6.1M 条样本）；来源数据包括 Pexels、Pixabay、Unsplash‑Lite（真实图像）以及使用预训练模型（如 SD‑XL、LlamaGen）做微调；实验对比数据集包括 T2I‑2M、BLIP3o‑60k；

**📈 对比分析**

评价方法为：随机抽取 500 条公共提示，进行大规模人工偏好评估（对齐与视觉质量），并报告 GenEval 自动指标；结果显示：Fine‑T2I 微调后 LlamaGen 视觉质量 80.7% 的胜率、对齐 65.3%；SD‑XL 视觉质量 61% 的胜率；相较于 T2I‑2M、BLIP3o‑60k，Fine‑T2I 在对齐与质量上均取得明显提升；

**⚠️ 局限性**

局限性包括：① 真实图像子集仅 168k 条，规模有限；② 合成图像可能仍缺乏某些复杂场景的真实性；③ 数据过滤成本高、生成算力需求大；④ 评价样本量 500 条可能不足以覆盖所有语义维度；⑤ 对极大分辨率图像支持仍有限；

---

## 116. Do Neural Networks Lose Plasticity in a Gradually Changing World?

**arXiv ID:** 2602.09234 | [PDF](https://arxiv.org/pdf/2602.09234v1)

**作者:** Tianhui Liu `[一作]` (University of Alberta), Lili Mou `[通讯]` (University of Alberta)

**通讯引用:** 6129 | [OpenAlex ID](https://openalex.org/A5024821632)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨在持续学习中网络失去可塑性的现象，提出并验证通过输入/输出插值和任务采样模拟逐步变化环境的方法，并给出理论和实验证明。

**💡 创新点**

①首次把逐步变化环境引入可塑性研究，证明急剧任务切换是导致可塑性下降的主要原因；②通过理论证明在平滑变化下梯度下降可保持在局部凸域内，维持可塑性；③提供实验框架，让渐进式任务切换在多种任务和模型上均能抵消可塑性损失。

**🔧 技术方法**

梯度下降、β‑平滑、局部强凸、输入/输出插值、任务采样、标签平滑；模型技术包括 MLP、ResNet‑18、T5‑small；对比技术包括 L2 正则化、Shrink&Perturb、Spectral 正则化、ReDO。

**📊 数据集**

MNIST、CIFAR‑10、Tiny‑ImageNet（图像任务）；EMNIST（像素置换任务）；自制随机 Seq2Seq 与 Bigram Cipher（文本任务）等多种数据集。

**📈 对比分析**

与 L2、Shrink&Perturb、Spectral 正则化、ReDO 等传统可塑性抑制方法相比，插值/采样方法在训练可塑性和泛化可塑性上均能保持或超过其性能；在大多数任务上，渐进式环境下模型训练精度与最佳基线持平或更优。

**⚠️ 局限性**

①模拟的渐进式环境仍需手工设置插值步长，过大会导致可塑性下降；②实验主要聚焦于小模型与合成任务，未验证在大模型或真实复杂任务中的通用性；③理论假设（β‑平滑、局部强凸、奇异值界限）对实际网络的适用性有限。

---

## 117. Faster Rates For Federated Variational Inequalities

**arXiv ID:** 2602.09164 | [PDF](https://arxiv.org/pdf/2602.09164v1)

**作者:** Guanghui Wang `[一作]` (Georgia Institute of Technology), Satyen Kale `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本论文研究了联邦优化在解决随机变分不等式（VIs）中的应用，提出了一系列改进的收敛速率。

**💡 创新点**

创新点在于通过对经典的Local Extra SGD算法进行精细分析，提供了更紧的收敛保证，并提出了一种新的算法Local Inexact Proximal Point Algorithm with Extra Step，能够减轻客户端漂移并在多个情况下实现改进的保证。

**🔧 技术方法**

使用了Local Extra SGD算法和Local Inexact Proximal Point Algorithm with Extra Step算法。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了联邦学习的应用场景，包括生成对抗网络和稳健强化学习等。

**📈 对比分析**

与现有方法相比，提出的算法在多个情况下实现了更好的收敛速率，尤其是在有界Hessian和有界算子的情况下，收敛速率达到了LSGD的最优界限。

**⚠️ 局限性**

限制在于该算法主要集中在同质设置下，假设所有机器从相同的分布中采样，未来的工作将探讨如何将结果扩展到异质设置。

---

## 118. Positive-Unlabelled Active Learning to Curate a Dataset for Orca Resident Interpretation

**arXiv ID:** 2602.09295 | [PDF](https://arxiv.org/pdf/2602.09295v1)

**作者:** Bret Nestor `[一作]` (Translacean Research Foundation), Jasper Kanes `[通讯]` (Ocean Networks Canada)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5079532993)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过正负标签无监督主动学习和业余标注，构建了覆盖30年海域、约5300小时的Southern Resident Killer Whale（SRKW）音频数据集DORI。

**💡 创新点**

创新点在于将正负标签主动学习与低成本人类标注结合，生成海量、公开的哺乳动物声学数据，并使用Transformer实现高效实时检测与分类。

**🔧 技术方法**

采用Whisper‑tiny Transformer提取嵌入、logistic回归/混合主动学习、熵/损失采样，并与PAMGuard、ROCCA、Animal‑Spot等基线模型对照。

**📊 数据集**

使用来自Ocean Networks Canada、SanctSound、OIO、OrcaSound等公开声学记录，合计约5300小时海洋哺乳动物音频，其中SRKW约920小时。

**📈 对比分析**

与传统方法相比，Whisper‑tiny在多组测试集上达到最高的95%灵敏度下特异性、准确率，并在速度和能耗上优于PAMGuard、ROCCA和Animal‑Spot。

**⚠️ 局限性**

局限包括对低SNR样本检测的敏感性不足、部分物种过滤器不完善，以及数据采样偏倚可能影响模型泛化能力。

---

## 119. NarraScore: Bridging Visual Narrative and Musical Dynamics via Hierarchical Affective Control

**arXiv ID:** 2602.09070 | [PDF](https://arxiv.org/pdf/2602.09070v1)

**作者:** Yufan Wen `[一作]` (Tsinghua University), Jian Wu `[通讯]` (ByteDance)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种层次化的情感驱动长视频配乐生成框架NarraScore，能够从视频中提取连续情绪轨迹并实时控制音乐生成。

**💡 创新点**

创新点包括将冻结的视觉‑语言模型作为情感感知器、双分支注入策略（全局语义锚和局部情感适配器），以及稀疏注意力注入实现高效长时序生成。

**🔧 技术方法**

采用Frozen Vision‑Language Model（如VideoLlama‑3）进行情绪提取，轻量级MLP探测头和残差情感适配器，结合MusicGen‑Small音频解码器与时间上采样适配器。

**📊 数据集**

使用基于影片观众情绪的连续VA数据集（movie dataset）以及带有VA标签的音乐情绪数据集（music emotion dataset）。

**📈 对比分析**

与VidMuse、GVMGen、M2UGEN、Video2Music、Caption2Music等基线对比，FAD、FD、KLD、IB指标均优于或相当于SOTA，在主观评测中在情感一致性、整体偏好等维度领先。

**⚠️ 局限性**

主要局限是情绪控制的时间粒度不足以实现对极快视觉事件的帧级同步，且级联结构易出现误差累积，未来需端到端联合优化和知识蒸馏降低视觉模块延迟。

---

## 120. All-in-One Conditioning for Text-to-Image Synthesis

**arXiv ID:** 2602.09165 | [PDF](https://arxiv.org/pdf/2602.09165v1)

**作者:** Hirunima Jayasekara `[一作]` (University of Maryland), Abhinav Shrivastava `[通讯]` (University of Maryland)

**通讯引用:** 7560 | [OpenAlex ID](https://openalex.org/A5101614443)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于场景图的零样本 ASQL Conditioner，在文本到图像的扩散模型中生成软视觉条件，通过推理时优化提升复杂多对象场景的语义一致性与结构完整性。

**💡 创新点**

创新点：① 用轻量 LLM 生成包含属性、尺寸、数量、位置的 ASQL 条件；② 采用模糊聚类+软区域掩模构建动态网格指导，避免硬布局限制；③ 通过多损失（属性、尺寸、位置交叉注意力、自注意力）实现推理时优化。

**🔧 技术方法**

使用技术包括 Stable Diffusion / PixArt‑α 扩散模型、CLIP 文本编码、Phi‑3-mini‑4k‑instruct 轻量 LLM、场景图生成、推理时优化、交叉注意力、Dice / 自注意力损失、模糊聚类与软掩模。

**📊 数据集**

使用的数据集：HRS、T2I‑CompBench、GenEval 评测集；场景图生成使用 FACTUAL 数据集；评估通过公开基准和多种模型对比进行。

**📈 对比分析**

与 SDv1.4/SDv2.1、PixArt‑α、Attend‑and‑Excite、EBAMA、Composable 等基线对比，实验显示在空间、属性绑定、位置准确率等指标上提升 5‑12%，整体 F1、BLIP、准确率显著优于基线。

**⚠️ 局限性**

局限性：对数量较大的计数指令仍不稳定；在过于文字化或复杂描述时容易产生字面化错误或缺失实体；需要进一步提升对高维度组合指令的泛化与可解释性。

---

## 121. FM SO.P: A Progressive Task Mixture Framework with Automatic Evaluation for Cross-Domain SOP Understanding

**arXiv ID:** 2602.09336 | [PDF](https://arxiv.org/pdf/2602.09336v1)

**作者:** Siyuan Huang `[一作]` (Johns Hopkins University), Han Zhao `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2647 | [OpenAlex ID](https://openalex.org/A5101670508)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FM SO.P框架，利用逐步任务混合训练和自动多智能体评估，提升跨域SOP理解能力。

**💡 创新点**

创新点在于：①分阶段任务混合（概念辨别→动作序列→图结构推理）累计数据避免优化冲突；②三智能体自适应评估系统自动生成域适配 rubric、分层测试集与评分。

**🔧 技术方法**

使用对比学习、结构化提示、图神经网络风格的序列推理，以及多智能体规则生成与评分。

**📊 数据集**

采用SOPBench七个行业域（Bank、DMV、Healthcare、Market、University、Library、Hotel）中的训练与测试集。

**📈 对比分析**

与多种基准模型（GPT‑4、Claude、Gemini、Llama、Qwen等）对比，7B版FM SO.P 34.3%通行率与72B基准相当，32B版 48.3%通行率刷新开源SOPBench最高。

**⚠️ 局限性**

局限包括：仍需人工或外部LLM生成参考答案；对极其复杂的跨流程依赖或长序列的推理效果尚有限；评估仍基于自动化 rubric，缺乏人工质量验证。

---

## 122. Genocide by Algorithm in Gaza: Artificial Intelligence, Countervailing Responsibility, and the Corruption of Public Discourse

**arXiv ID:** 2602.09202 | [PDF](https://arxiv.org/pdf/2602.09202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 123. MUZZLE: Adaptive Agentic Red-Teaming of Web Agents Against Indirect Prompt Injection Attacks

**arXiv ID:** 2602.09222 | [PDF](https://arxiv.org/pdf/2602.09222v1)

**作者:** Georgios Syros `[一作]` (Northeastern University), Alina Oprea `[通讯]` (Northeastern University)

**通讯引用:** 5862 | [OpenAlex ID](https://openalex.org/A5035574749)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 MUZZLE，一个完全自动化的红队框架，用于在沙盒化网页环境中发现并评估网页代理的间接提示注入攻击；通过代理执行轨迹识别高价值注入点，自动生成上下文感知的恶意指令并迭代优化；支持多步骤、跨应用的攻击链。

**💡 创新点**

① 自动化发现跨应用间接提示注入攻击；② 利用代理轨迹动态识别注入表面；③ 自适应生成上下文感知 payload；④ 完全不需要人工指定注入位置或模板；⑤ 通过多代理协作实现从Reconnaissance到Reflection的闭环。

**🔧 技术方法**

多代理架构（Explorer、Summarizer、Grafter、Dispatcher、Payload Generator、Judge）配合 AutoGen；LLM 作为红队与目标代理（GPT‑4o、GPT‑4.1、Qwen3‑VL‑32B‑Instruct）；沙盒化网页环境（基于 WebArena/VisualWebArena 扩展）；改造 PAIR 的 jailbreak 技术；利用上下文窗口定位实现高效注入。

**📊 数据集**

自建的虚拟网络环境中的四个应用（Gitea、Postmill、Classifieds、Northwind）以及跨应用场景；每个应用配合多个用户任务与攻击目标；不依赖公开数据集，而是通过环境内的交互记录与注入点自动生成。

**📈 对比分析**

与 WASP 和 VWA‑Adv 进行对比。评估指标为部分攻击（Partial）与完整攻击（End‑to‑End）成功率。四个应用共发现 4‑5 个完整攻击；跨应用攻击在 5 次试验中 2 次成功。平均跑时约 8 min，其中 54% 由 LLM 推理占用。相较于手工模板和单步攻击，MUZZLE 在成功率与攻击多样性上显著提升。

**⚠️ 局限性**

① 对极度破坏性指令（如删除仓库）受模型安全策略限制，成功率低；② 需要攻击目标应用支持登录/授权，跨应用攻击受限；③ 依赖红队 LLM 的推理能力，若模型更新或安全性提升，效果可能下降；④ 对极其复杂或极小的网页交互仍可能漏掉注入点；⑤ 未评估大规模多代理协同或动态网络拓扑的可扩展性。

---

## 124. Privacy Amplification for BandMF via $b$-Min-Sep Subsampling

**arXiv ID:** 2602.09338 | [PDF](https://arxiv.org/pdf/2602.09338v1)

**作者:** Andy Dong `[一作]` (Stanford University), Arun Ganesh `[通讯]` (Google Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种新的采样方法 b‑min‑sep，用于在 BandMF（带状噪声矩阵）中实现更强的隐私放大。

**💡 创新点**

创新点在于把 Poisson 采样、balls‑in‑bins 采样统一为 b‑min‑sep，提供近似精确的 Monte‑Carlo 隐私分析，并将其推广到多归属用户级别的差分隐私。

**🔧 技术方法**

使用了 Monte‑Carlo 会计与动态规划技术，对带状 Toeplitz 矩阵的 Gaussian 机制进行精确隐私评估，并结合隐私放大与矩阵机制分析。

**📊 数据集**

实验数据集为 CIFAR‑10（VGG 训练）和 arXiv 摘要（TinyBERT 微调），用于评估 MSE 与测试性能。

**📈 对比分析**

与 DP‑SGD+Poisson、循环 Poisson、balls‑in‑bins 进行对比；在中至低噪声/高 ε 场景下，b‑min‑sep 在 MSE 与测试精度/损失上均优于其他方案；在极低 ε 时提升有限。

**⚠️ 局限性**

局限性包括：依赖 Monte‑Carlo 估计带来的计算开销和误差；在低 ε 时提升不明显；适用范围受限于带状 Toeplitz 矩阵和需要预处理的多归属用户设置。

---

## 125. Predicting Open Source Software Sustainability with Deep Temporal Neural Hierarchical Architectures and Explainable AI

**arXiv ID:** 2602.09064 | [PDF](https://arxiv.org/pdf/2602.09064v1)

**作者:** S M Rakib Ul Karim `[一作]` (University of Missouri), Sean Goggins `[通讯]` (University of Missouri)

**通讯引用:** 2817 | [OpenAlex ID](https://openalex.org/A5037107679)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套分层时间序列预测框架，利用Transformer与MLP混合模型预测OSS项目的生命周期阶段，并通过可解释AI技术对模型决策进行归因分析。

**💡 创新点**

①层次化多阶段分类设计，有效缓解类别不平衡和相似性问题；②结合24个月的时间序列与聚合特征，兼顾动态与静态信息；③引入SHAP、Integrated Gradients等可解释方法，对活动类别进行归因；④公开完整实现与代码，便于复现。

**🔧 技术方法**

Transformer编码器、MLP、层次化决策、焦点损失、SMOTE、类权重、早停、SHAP、Integrated Gradients、注意力分析、归因聚合等技术。

**📊 数据集**

使用约2000+个GitHub OSS仓库的24个月月度活动数据（20个基础指标），按社交技术框架划分四个生命周期标签（club、contribMid、federation、toy）。

**📈 对比分析**

与MLP、GRU、TCN、1D CNN、CNN‑LSTM、AutoEncoder等基线模型对比，整体准确率94.08%，宏F1 78.96%，平衡准确率79.42%；在多数类别表现优异，少数类仍面临识别难题。

**⚠️ 局限性**

主要局限：极端类别不平衡导致少数类识别不佳；层次化决策易产生误分类传播；仅基于仓库活动缺乏治理、资金等外部信息；可解释归因聚合粗粒度，未捕捉更细粒度特征。

---

## 126. LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis

**arXiv ID:** 2602.09379 | [PDF](https://arxiv.org/pdf/2602.09379v1)

**作者:** Shihao Xu `[一作]` (EverMind AI Inc), Yafeng Deng `[通讯]` (EverMind AI Inc)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了LingxiDiagBench，一套包含静态诊断评估和动态多轮对话评估的中文精神科诊断基准，提供16k合成诊疗对话和真实EMR数据。

**💡 创新点**

创新点在于：①使用基于真实医疗记录的统计分布生成高保真合成对话；②引入多代理架构（患者、医生、诊断），实现可交互诊疗流程；③同时评估诊断准确性与对话质量，揭示两者并不完全相关。

**🔧 技术方法**

技术主要包括大型语言模型（Qwen、Gemini、Claude、GPT系列）、对话生成提示工程、检索增强（APA‑Guided + MRD‑RAG）、LLM‑as‑a‑Judge评估方法，以及多任务分类与序列生成评估指标。

**📊 数据集**

数据集为：①LingxiDiag‑Clinical（约1709例真实精神科病例的匿名EMR与转录）和②LingxiDiag‑16K（16,000例基于前者统计分布生成的合成诊疗对话）。

**📈 对比分析**

与多种开源与商用LLM进行对比：在2分类（抑郁‑焦虑）上最高可达92.3%准确率；4分类仅达43.0%；12分类最高28.5%。对话质量评分与诊断准确度的相关性仅为0.43，表明问诊表现与诊断效果不一致。

**⚠️ 局限性**

局限性包括：合成对话仍缺乏罕见病和非典型症状的多样性；评估仅在中文环境下，跨语言推广受限；未涵盖治疗方案、随访等临床流程；并且当前模型在诊断推理与问诊策略上仍有显著差距。

---

## 127. Investigating Bystander Privacy in Chinese Smart Home Apps

**arXiv ID:** 2602.09254 | [PDF](https://arxiv.org/pdf/2602.09254v1)

**作者:** Shijing He `[一作]` (Kings College London), Jose Such `[通讯]` (Ingenio CSIC Universitat Politècnica de València)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了49款中国智能家居APP的隐私政策、UX/UI与Apple App Store隐私标签，揭示对旁观者隐私缺失及政策与实现之间的脱节。

**💡 创新点**

首次系统性探讨中文智能家居生态中的旁观者隐私缺口，并提出可行的设计与监管改进建议。

**🔧 技术方法**

采用混合方法：主题分析（隐私政策）、认知漫游与启发式评估（UX/UI）、可追溯性矩阵对比以及对App Store隐私标签的审计。

**📊 数据集**

基于从苹果App Store抓取的49款智能家居应用公开隐私政策、界面截图和权限请求，构成实验数据集。

**📈 对比分析**

通过对比政策声明、UI实现和隐私标签三方对应关系，发现约80%应用实现完整追溯，约70%出现断裂，表明设计缺陷影响用户完整控制。

**⚠️ 局限性**

研究仅覆盖iOS平台与中国大陆市场，未检验Android或海外版本，也未对后台数据流或后端实现进行技术审计。

---

## 128. Importance inversion transfer identifies shared principles for cross-domain learning

**arXiv ID:** 2602.09116 | [PDF](https://arxiv.org/pdf/2602.09116v1)

**作者:** Daniele Caligiore `[一作]` (National Research Council), Daniele Caligiore `[通讯]` (National Research Council)

**通讯引用:** 2228 | [OpenAlex ID](https://openalex.org/A5060335674)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 X-CDTL 框架及 Importance Inversion Transfer（IIT）机制，实现跨领域网络结构的可解释迁移学习，提升在高噪声与数据稀缺场景下的异常检测稳定性。

**💡 创新点**

创新点在于将网络科学与可解释人工智能相结合，通过逆向特征重要性筛选出域无关的结构锚点，形成可解释的跨域迁移基准；IIT 通过优先保留低区分度但稳定的拓扑特征，突破传统对齐方法在极端异质系统中的失效。

**🔧 技术方法**

技术包括：多尺度拓扑特征提取（12项），Borda 逆排序与 IIT 分数综合评估，PCA‑SVD 共享子空间对齐，Isolation Forest 异常检测，以及多因素压力测试和非参数统计验证。

**📊 数据集**

使用四类公开网络数据集：Facebook Ego（社会网络），QM9（分子图），PROTEINS（蛋白相互作用网络），Brown Corpus（语言共现网络）。

**📈 对比分析**

通过与无迁移基线、传统对齐方法以及完整特征集对比，使用 ROC‑AUC、AP 与 F1 等指标评估。结果显示，使用 8 个 IIT 结构锚点在极噪声下可实现 56% 相对提升，且在大多数跨域对中保持 0.97 以上的 ROC‑AUC，验证了框架的鲁棒性与可解释性。

**⚠️ 局限性**

局限性包括：仅针对静态无向图；对动态、多层或因果网络的扩展尚未验证；IIT 需要先验的多模型重要性评估，计算成本较高；在高度相似域中，逆向选择可能导致信息丢失。

---

## 129. On A Parameterized Theory of Dynamic Logic for Operationally-based Programs

**arXiv ID:** 2602.09307 | [PDF](https://arxiv.org/pdf/2602.09307v1)

**作者:** Yuanrui Zhang `[一作]` (Nanjing University of Aeronautics and Astronautics), Yuanrui Zhang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5006367985)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种可参数化的动态逻辑（称为 DLp），通过标记与参数化机制直接基于程序的操作语义进行推理，并构造了一个小型内核推理规则集与对应的循环证明系统。

**💡 创新点**

创新点在于将操作语义与动态逻辑统一，消除传统动态逻辑中对程序语法规则的依赖；引入标记化配置与可自由标签；提供通用的循环推理框架；实现多模型可比性与对现有动态逻辑的兼容性。

**🔧 技术方法**

主要技术包括：参数化动态逻辑框架、标记化（labeling）与自由标签技术、基于操作语义的推理规则、循环证明（cyclic proof）与进展轨迹（progressive derivation trace）以及规则提升（lifting）方法。

**📊 数据集**

论文以 while 程序、Esterel 同步语言、FODL 以及过程逻辑等为案例进行演示，未使用实际数据集，而是通过理论示例与手工证明展示方法。

**📈 对比分析**

与传统 PDL、FODL 等动态逻辑相比，DLp 只需少量通用推理规则且不需程序结构转换，证明过程更直接；实验示例表明循环证明能在不展开循环结构的情况下完成推导，展示了在复杂程序模型上的优势。

**⚠️ 局限性**

限制在于：需要满足终止有限性与最小执行路径有限性的假设；对具有复杂测试（test）或非标准程序模型的支持有限；完整性证明仅在特定条件下成立，未给出完全判定性与复杂度分析。

---

## 130. MacrOData: New Benchmarks of Thousands of Datasets for Tabular Outlier Detection

**arXiv ID:** 2602.09329 | [PDF](https://arxiv.org/pdf/2602.09329v1)

**作者:** Xueying Ding `[一作]` (Carnegie Mellon University), Leman Akoglu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8375 | [OpenAlex ID](https://openalex.org/A5001634795)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含2446个表格异常检测数据集的大规模基准套件，并在其上系统评测了多种异常检测方法。

**💡 创新点**

创新点在于规模和多样性——涵盖790个语义异常、856个统计异常以及800个合成数据集；提供标准化的训练/测试划分、公共/私有分区以及丰富的语义元数据，并对方法进行超参数敏感性评估。

**🔧 技术方法**

使用了TabLib数据爬取与筛选、哈希去重、随机森林可分离性检查、LLM生成元数据；合成数据采用高斯混合、结构因果模型、Copula等生成器；评估基于14种经典、深度和基础模型（如KNN、LOF、GOAD、TabPFN‑OD、OutFormer 等）并统计AUROC/AUPRC、排名、ELO、运行时等指标。

**📊 数据集**

利用790个语义异常真实数据、856个统计异常真实数据和800个合成数据，总计2446个数据集；同时整合了TabZilla、TabRepo 等公开数据源。

**📈 对比分析**

通过在所有数据集上统一评测，将方法按AUROC/AUPRC进行排名，发现基础模型（TabPFN‑OD、OutFormer、FoMo‑0D）在性能和速度上均显著优于传统和深度模型，除EGMM外；经典方法在低资源环境下表现稳定，深度模型受超参数影响大。

**⚠️ 局限性**

局限性包括：数据源仍以TabLib为主，可能存在领域偏差；合成异常生成虽多样但未必覆盖所有真实场景；评估仅基于AUROC/AUPRC等指标，未考虑更细粒度的错误类型；部分基线方法选取可能不全面；私有数据的安全性依赖匿名化处理；

---

## 131. Untangling the Timeline: Challenges and Opportunities in Supporting Version Control in Modern Computer-Aided Design

**arXiv ID:** 2602.09236 | [PDF](https://arxiv.org/pdf/2602.09236v1)

**作者:** Yuanzhe Deng `[一作]` (University of Toronto), Shurui Zhou `[通讯]` (University of Toronto)

**通讯引用:** 711 | [OpenAlex ID](https://openalex.org/A5040272202)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对170条来自7个在线论坛的CAD版本控制讨论进行系统性访谈分析，梳理用户面临的管理、连续、范围和分布等四类挑战；

**💡 创新点**

首次从用户角度综合描述CAD版本控制的社会技术难点，并提出三项设计机遇（支持表述工作、促进跨界协作、实现基础设施反射性），为CAD版本控制改进提供新视角；

**🔧 技术方法**

采用关键词检索+自建网页爬虫收集数据，使用开放编码和闭合编码两阶段质性内容分析，计算编码一致性；

**📊 数据集**

共收集424条论坛帖子（包括Autodesk、Onshape、SolidWorks、Reddit、Eng‑Tips、CAD Forum、Thingiverse组），最终提取170条与版本控制相关的讨论；

**📈 对比分析**

通过对三大商业CAD平台（SolidWorks、Fusion、Onshape）的支持情况进行对比，按出现频率统计挑战类型；虽未给出数值性能指标，但发现部分挑战普遍存在、部分问题平台特异；

**⚠️ 局限性**

研究仅基于公开论坛讨论，缺乏实验验证和多学科覆盖；仅涵盖三款主流CAD软件，未覆盖企业级PLM方案，且数据来源单一，可能无法代表所有用户实践。

---

## 132. SVD-Preconditioned Gradient Descent Method for Solving Nonlinear Least Squares Problems

**arXiv ID:** 2602.09057 | [PDF](https://arxiv.org/pdf/2602.09057v1)

**作者:** Zhipeng Chang `[一作]` (Penn State University), Nian Liu `[通讯]` (Penn State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于奇异值分解预处理的梯度下降算法，用于求解非线性最小二乘问题。

**💡 创新点**

创新点在于利用SVD构造预条件器，既捕捉了雅可比矩阵的曲率信息，又显著提升了收敛速度与鲁棒性。

**🔧 技术方法**

采用SVD、梯度下降、预处理技术，并结合经典的Levenberg–Marquardt和高斯-牛顿算法进行对比。

**📊 数据集**

在合成基准问题（如Rosenbrock、Powell等）以及实际数据集（如光度立体、传感器标定）上进行实验。

**📈 对比分析**

与LM、Gauss–Newton和普通梯度下降相比，该方法收敛更快、残差更低，在大多数测试问题上至少提高了30% 的效率。

**⚠️ 局限性**

缺点是每次迭代都需要计算雅可比矩阵的SVD，导致大规模问题的计算开销较大；对强非线性或不光滑问题的适应性仍有限。

---

## 133. Enhanced Graph Transformer with Serialized Graph Tokens

**arXiv ID:** 2602.09065 | [PDF](https://arxiv.org/pdf/2602.09065v1)

**作者:** Ruixiang Wang `[一作]` (Institute of Automation, Chinese Academy of Sciences), Chunhong Pan `[通讯]` (School of Artificial Intelligence, University of Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了序列化token范式，用多token序列替代单token聚合，生成更丰富的图级表示。

**💡 创新点**

创新点在于将节点特征聚合成可学习的基准token序列并加入位置编码，随后利用自注意力建模多token间复杂交互，突破单token信息瓶颈。

**🔧 技术方法**

技术包括基于欧氏相似度的Gumbel Softmax聚合、可学习基准token序列做位置编码、堆叠自注意力网络以及局部消息传递模块。

**📊 数据集**

使用了 ZINC、ZINC‑FULL 以及 MolHIV 三个图级任务基准数据集。

**📈 对比分析**

与最新图Transformer/MPNN方法对比，在 ZINC MAE 0.055、MolHIV AUC 0.8163 以及 ZINC‑FULL MAE 0.013 上均取得最优或第二优表现，显示显著性能提升。

**⚠️ 局限性**

局限在于仅针对图级任务，未将全局信息嵌入节点学习阶段，对超大规模图的计算成本仍需进一步优化。

---

## 134. Legs Over Arms: On the Predictive Value of Lower-Body Pose for Human Trajectory Prediction from Egocentric Robot Perception

**arXiv ID:** 2602.09076 | [PDF](https://arxiv.org/pdf/2602.09076v1)

**作者:** Nhat Le `[一作]` (George Mason University), Xuesu Xiao `[通讯]` (George Mason University)

**通讯引用:** 1933 | [OpenAlex ID](https://openalex.org/A5017662025)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究利用360°机器人第一人称视角下的人类骨骼关键点，评估不同骨骼区域和推导的生物力学特征对多智能体轨迹预测的影响。

**💡 创新点**

系统比较了3D与2D关键点以及上下肢子集的预测价值，发现下肢3D关键点和相应的生物力学指示是最具预测力，并验证2D关键点在未校正的全景图像中仍能显著提升性能。

**🔧 技术方法**

采用Human Scene Transformer (HST) 模型并加入轻量MLP编码骨骼特征，利用姿态估计与3D重建、COCO 17关键点等技术。

**📊 数据集**

在JRDB和自建的Insta360 X4全景社会导航数据集上进行评估，其中JRDB提供3D姿态估计，后者提供未纠正的全景图像与2D姿态。

**📈 对比分析**

通过固定训练协议对不同骨骼配置（全身、上肢、下肢、2D/3D）进行Ablation，对MinADE、MinFDE、MLADE、NLL_pos等指标进行比较，结果显示下肢3D关键点可使MinADE下降13%，加入生物力学指示进一步提升1-4%，而2D关键点在全景图像上可提升7%。

**⚠️ 局限性**

实验受限于HST模型的架构，未考虑其他社会线索（手势、群体行为）以及不同底层网络的鲁棒性，且3D姿态推断在实时性和对遮挡的鲁棒性上仍有提升空间。

---

## 135. Measuring Privacy Risks and Tradeoffs in Financial Synthetic Data Generation

**arXiv ID:** 2602.09288 | [PDF](https://arxiv.org/pdf/2602.09288v1)

**作者:** Michael Zuo `[一作]` (Rensselaer Polytechnic Institute), Oshani Seneviratne `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 876 | [OpenAlex ID](https://openalex.org/A5038466673)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在金融表格数据上生成合成数据的隐私‑实用性权衡，针对类别不平衡和混合类型特征，提出了新的差分隐私实现的CTGAN与TVAE，并对多种生成器（Gaussian Copula、TabDiff、CTGAN、TVAE、DP‑CTGAN、DP‑TVAE）在六个金融数据集上的质量、下游任务性能与隐私指标进行了系统评估。

**💡 创新点**

创新点在于：①首次实现并公开了满足差分隐私的CTGAN与TVAE；②通过针对金融数据的预处理与模式归一化修改，消除了原有实现的隐私泄漏点；③对金融领域严重类别不平衡、混合特征的合成数据质量与隐私做了细粒度实验与分析。

**🔧 技术方法**

使用的技术包括：差分隐私（(ε,δ)-DP）与DP‑Adam、GAN/AE生成器（CTGAN、TVAE）、高斯Copula、TabDiff扩散生成器、MIA（shadow model）攻击、SDMetrics隐私度量（DCR基线与过拟合保护）以及XGBoost下游分类任务评估。

**📊 数据集**

采用六个公开金融数据集：Adult/Census Income (AD)、Bank Customer Churn (BC)、Bank Marketing (BM)、Default of Credit Card Clients (CC)、German Credit Data (CR) 以及 Give Me Some Credit (GM)，这些数据集包含混合数值/类别特征且存在显著类别不平衡。

**📈 对比分析**

实验比较发现：非DP生成器中TabDiff质量最高；在类别平衡维持下，生成器整体性能提升；DP‑CTGAN在保留高质量与实用性的同时，随着ε下降，平衡准确率与类别比例波动增大；DP‑TVAE易出现模式崩塌，实用性下降；DCR隐私指标与MIA攻击结果表现不一致，难以作为单一评判标准。

**⚠️ 局限性**

限制在于：①差分隐私实现对模型结构仍有影响，导致实用性明显下降；②MIA与DCR等隐私度量在金融数据上不具一致性，缺乏可靠的评估标准；③实验仅覆盖六个数据集，未验证跨域泛化；④对极端类别不平衡的处理仍有限；⑤缺乏对生成数据长期使用后风险的长期评估。

---

## 136. Large Language Models for Designing Participatory Budgeting Rules

**arXiv ID:** 2602.09349 | [PDF](https://arxiv.org/pdf/2602.09349v1)

**作者:** Nguyen Thach `[一作]` (University of Nebraska Lincoln), Hau Chan `[通讯]` (University of Nebraska Lincoln)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 LLMRule，一种基于大语言模型的进化搜索框架，用来自动设计参与式预算（PB）规则，并通过实验验证其在效用与公平之间的权衡。

**💡 创新点**

创新点：①引入可线性验证的 Strong‑EJR 近似公平目标，弥补现有公平度量的高复杂度；②将 LLM 与进化搜索结合，形成多目标（效用+公平）规则生成流程；③通过约束罚项实现双目标优化；④展示所生成规则在不同规模与投票类型下均能保持较高效用与公平，且能作为优秀的补全方法。

**🔧 技术方法**

技术手段：使用 GPT‑4o mini 生成规则文本与代码；进化搜索（父代选择、交叉、变异、提示改进）；Strong‑EJR 近似与其线性化验证；Apriori 频繁项集算法加速最大小致组搜索；对规则进行优先级评分并进行多目标评估。

**📊 数据集**

数据集：来自 Pabulib 的 617 个真实 PB 实例（美国、加拿大、波兰、荷兰），按投票类型（approval/ cardinal）与规模（小/大）划分，训练集为小规模 approval 实例，测试集包括大规模实例与留作 ID 测试的美国实例。

**📈 对比分析**

比较方法：与 MaxUtil、GreedUtil、MES 及其补全方式、SeqPhrag、MaximinSupp 等传统 PB 规则在 ID 与 OOD 集合上进行对比。实验表明 LLMRule 在利用率（ω）与 Strong‑EJR 近似（ϕ）上均达到了或超过基线，尤其在贪婪规则和补全方法中表现突出，展示了优秀的效用‑公平折衷。

**⚠️ 局限性**

局限性：①仍需人工设计提示与微调；②Strong‑EJR 近似虽然可线性化，但与真正的 Strong‑EJR 并非完全等价；③算法在项目数 m 较大时仍面临指数级搜索；④目前仅验证了贪婪规则框架，其他策略或更复杂的公平度量尚未探索。

---

## 137. BrainTAP: Brain Disorder Prediction with Adaptive Distill and Selective Prior Integration

**arXiv ID:** 2602.09294 | [PDF](https://arxiv.org/pdf/2602.09294v1)

**作者:** Zhenyu Lei `[一作]` (University of Virginia), Jundong Li `[通讯]` (University of Virginia)

**通讯引用:** 13385 | [OpenAlex ID](https://openalex.org/A5029588473)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 BrainTAP 模型，通过融合功能连接（FC）和结构连接（SC）网络预测青少年注意相关疾病。

**💡 创新点**

创新点在于自适应互相蒸馏（AMD）控制跨模态信息共享，同时选择性先验融合（SPF）学习个体化与全局神经学先验门控，提升对多模态特征的保留与利用。

**🔧 技术方法**

采用 Transformer 架构、互相蒸馏技术、可学习的先验门控与注意力偏置，实现跨模态协同与专家先验的自适应整合。

**📊 数据集**

使用 ABCD 项目中 9-11 岁儿童的 rs‑fMRI 数据，包含功能与结构连接网络。

**📈 对比分析**

与单模态 MLP、MaskedGCN 以及多模态 RH‑BrainFS、CrossGNN 进行对比，BrainTAP 在 ADHD、焦虑、OCD、ATT 四种疾病的 AUC 均超过基线，最高可达 73.8，显著提升性能。

**⚠️ 局限性**

对先验集合的依赖性仍较强，模型对先验数量与质量敏感，且未验证在不同年龄组或任务条件下的泛化能力。

---

## 138. Clarifying Shampoo: Adapting Spectral Descent to Stochasticity and the Parameter Trajectory

**arXiv ID:** 2602.09314 | [PDF](https://arxiv.org/pdf/2602.09314v1)

**作者:** Runa Eschenhagen `[一作]` (University of Cambridge), Hao-Jun Michael Shi `[通讯]` (Meta Platforms)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过理论与大规模实验，阐明了Shampoo优化器如何通过在参数轨迹与随机性上进行时间平均、实现矩阵级别的自适应，并将其与Muon、Adam、Signum等优化器进行对比；

**💡 创新点**

创新点在于将Shampoo拆解为Muon更新与两侧适配矩阵，证明Shampoo的优势完全来自对权重矩阵的处理；提出时间平均半正交约束的“时间平均半正交性”视角，将Shampoo与谱下降（SpectralGD）统一；

**🔧 技术方法**

主要技术包括矩阵预条件、Kronecker分解、谱下降、EMA时间平均、矩阵白化、KL-Shampoo求解、Newton-Schulz迭代、SVD、Adam/Signum的解析分解；

**📊 数据集**

使用的公开数据集为C4（语言模型）和Imagewoof（视觉Transformer/ConvNeXt V2），模型为Llama3架构，批大小64/256，Token预算不同；

**📈 对比分析**

实验将Shampoo（Shampoo^14、Shampoo^12、KL-Shampoo）、Muon（SVD）、AdamW、Signum等在相同超参搜索下比较，结果显示Shampoo各变种均优于Muon和AdamW；KL-Shampoo在大多数设置下最优，Shampoo^12优于Shampoo^14；

**⚠️ 局限性**

局限性包括仅在C4和Llama3上实验，未覆盖其他任务与数据集；未进行完整超参搜索，未调节权重衰减；计算/内存开销较高；对大规模稳定性、下游任务影响尚未系统评估。

---

## 139. SciDataCopilot: An Agentic Data Preparation Framework for AGI-driven Scientific Discovery

**arXiv ID:** 2602.09132 | [PDF](https://arxiv.org/pdf/2602.09132v1)

**作者:** Jiyong Rao `[一作]` (Shanghai Artificial Intelligence Laboratory), Chi Zhang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 6188 | [OpenAlex ID](https://openalex.org/A5100362456)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了科学AI‑Ready数据范式，并设计了SciDataCopilot多代理框架，实现从原始异构科学数据到可直接用于AI模型的任务定制数据的自动化转换；

**💡 创新点**

创新点包括：①将科学任务作为数据组织的核心，实现任务条件化的数据规范化；②通过四个协同代理（数据访问、意图解析、数据处理、数据集成）实现分阶段、可适配的工作流；③展示了跨学科、多模态数据在同一框架下的统一处理与融合；

**🔧 技术方法**

技术手段主要有：大语言模型（GPT‑5.2）驱动的代理推理与执行；案例库检索与重用机制；面向领域的工具链调用（如RDKit、MNE‑Python、xarray等）；约束驱动的数据融合与校验；闭环调试与版本追踪；

**📊 数据集**

使用的数据集包括：生命科学中的酶催化实验记录（生成214K条任务定制记录）；神经科学中的EEG/MEG数据集（四个子任务处理）；地球科学中的气象观测时间序列；同时依赖公开的实验平台、工具包与域数据仓库；

**📈 对比分析**

与专家手工流程对比，酶数据规模提升约20×；神经数据处理速度提升3–5×且与专家量化结果一致；气象数据准备效率提升30×；整体实现了与人工工作流相当或更优的质量与效率；

**⚠️ 局限性**

局限性包括：高度依赖预先整理的领域工具与案例；在新领域或新模态下的迁移需要人工干预；缺乏对科学有效性、可重复性等更深层评估指标；对实时流式实验数据的支持尚未完善。

---

## 140. Effective Reasoning Chains Reduce Intrinsic Dimensionality

**arXiv ID:** 2602.09276 | [PDF](https://arxiv.org/pdf/2602.09276v1)

**作者:** Archiki Prasad `[一作]` (University of North Carolina Chapel Hill), Peter Shaw `[通讯]` (Google DeepMind)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5106185324)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过测量不同链式推理（CoT）策略下的内在维度，探究推理链为何能提升语言模型的泛化能力。

**💡 创新点**

创新点在于：①提出使用内在维度作为定量指标评估推理策略的有效性；②发现内在维度与泛化性能呈强负相关，优质推理链可在更低维子空间中学习；③展示该指标相较于长度、KL散度、token perplexity更具预测力。

**🔧 技术方法**

技术手段主要包括：利用LoRA低秩适配器对Gemma‑3 1B/4B模型进行训练；在不同CoT策略下搜集训练数据并计算达到预设准确率阈值所需的最小可训练参数（即内在维度）；对比多种基准指标并计算Spearman相关系数。

**📊 数据集**

数据集：使用数学文字题集GSM‑8K的训练集；在GSM‑8K测试集及五个OOD压力测试集（GSM‑Symbolic、GSM‑IC、GSM‑Hard等）上评估泛化性能。

**📈 对比分析**

比较方法：对14种CoT/非CoT策略分别计算内在维度、长度、KL散度、token perplexity；在Gemma‑3 4B和1B模型上计算这些指标与整体准确率的Spearman相关性。结果显示，内在维度在4B模型上与整体准确率的相关系数为0.93，在1B模型上为0.75，远高于其他指标，表明其更能解释并预测推理策略的效果。

**⚠️ 局限性**

局限性包括：①内在维度评估需要对不同LoRA规模进行多次训练，计算成本高；②阈值选择对内在维度数值有影响，虽然相关性稳健但仍需更系统的阈值设定方法；③实验仅针对Gemma‑3模型与GSM‑8K任务，泛化到更大模型或其他推理任务的适用性尚未验证。

---

## 141. A Small-Scale System for Autoregressive Program Synthesis Enabling Controlled Experimentation

**arXiv ID:** 2602.09112 | [PDF](https://arxiv.org/pdf/2602.09112v1)

**作者:** Russ Webb `[一作]` (Apple), Jason Ramapuram `[通讯]` (Apple)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并训练了一种名为Cadmus的后缀式虚拟机（VM）模型，专门用于处理整数运算、序列标记和子程序调用等任务，并通过自回归方式训练出一个280M参数的Transformer模型，能够对“真程序”进行准确预测与执行。

**💡 创新点**

创新点包括：1）将模型训练目标限定为仅产生真值（true‑programs）的指令序列，从而避免自然语言噪声；2）设计固定、一对一的指令词表，消除词汇化复杂度；3）提供可验证的训练、验证与执行环境，便于实验可重复性；4）系统性比较与GPT‑5等大型语言模型的差异，揭示符号偏置对推理的影响。

**🔧 技术方法**

技术实现基于18层decoder‑only Transformer（1280维嵌入、20头、多层感知器3600维、GELU激活），使用Adam优化器（lr=1e‑4、余弦调度），在八块H100显卡上训练300k步，批量大小1024；VM指令集共65条，支持基本算术、比较、取反等操作。

**📊 数据集**

数据集由80M条“真程序”组成，按多种模板（基本算术、子程序、序列操作、标签应用、完整算法等）随机采样，验证集每类约2k条程序；此外还设计了随机真程序集（200k条）以及多层次的子问题模板。

**📈 对比分析**

实验对比：在分布内（[-20,20]）Cadmus模型在多种子任务上均达到1.0或接近1.0的准确率，随机真程序集约0.92；与GPT‑5对照实验显示Cadmus在分布内的预测优于GPT‑5，而在分布外则仅约46%准确，说明模型在未见值上仍存在泛化挑战。进一步的分析表明：①在序列长度变化时模型准确率波动，②对第二个数的计算阶段准确率下降后恢复，③比较指令后仍保留部分数值信息。

**⚠️ 局限性**

局限性主要体现在：1）对分布外数值的泛化能力不足；2）模型对符号表的依赖使得符号重定义可轻易破坏；3）当前仅覆盖整数运算，缺乏浮点/字符串等类型；4）缺乏深入的因果推理与自适应学习策略；5）实验仍依赖特定的指令集与模板，未能覆盖更广泛的编程范式。

---

## 142. Risk-sensitive reinforcement learning using expectiles, shortfall risk and optimized certainty equivalent risk

**arXiv ID:** 2602.09300 | [PDF](https://arxiv.org/pdf/2602.09300v1)

**作者:** Sumedh Gupte `[一作]` (Indian Institute of Technology Madras), Sanjay P. Bhat `[通讯]` (TCS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并提出了一种通用的风险敏感策略梯度算法，针对期望极限(expectiles)、基于效用的短缺风险(UBSR)和优化确定性等价(OCE)三类风险度量，给出了策略梯度定理、估计器以及MSE界限，并给出了非渐进收敛速率。

**💡 创新点**

创新点在于：
1) 将期望极限、UBSR、OCE这三类风险统一到RL框架中并推导对应的策略梯度表达式；
2) 对估计器给出O(1/m)的均方误差上界；
3) 在标准假设下证明目标函数光滑，从而得到非渐进收敛速率；
4) 通过实验验证风险敏感策略梯度相较于传统REINFORCE在奖励均值和方差上的优势。

**🔧 技术方法**

主要技术包括：
- 策略梯度方法（REINFORCE式估计）
- 风险度量的数学分析（期望极限、UBSR、OCE）
- 随机过程和马尔可夫决策过程理论
- MSE与高概率误差分析
- 非渐进收敛分析（Lipschitz、光滑性假设）

**📊 数据集**

使用MuJoCo的Reacher环境进行实验，采用标准的模拟轨迹样本（N=10000，batch=100）

**📈 对比分析**

与REINFORCE基线进行比较。结果显示：
- Expectile、Quadratic、Mean-Variance和Entropic风险的RAPG算法在平均奖励上均优于REINFORCE；
- Expectile表现最佳，平均奖励最高、方差最低；
- 具体数值：REINFORCE均值-8.07、Std 1.25；Entropic-4.41、Std 0.36；Quadratic-2.48、Std 0.11；Expectile-2.29、Std 0.11；Mean‑Variance-6.86、Std 0.90；CVaR-9.77、Std 1.83。

**⚠️ 局限性**

局限性：
- 仅在有限期HMDP和光滑性、方差有限的假设下证明，实际应用中可能不满足；
- 对UBSR和OCE的估计器使用了双重采样或假设，实际计算成本可能较高；
- 仅在MuJoCo Reacher环境验证，缺乏更广泛的跨域实验；
- 对于非连续或非光滑的风险度量（如CVaR的尖点），理论分析不完整。

---

## 143. Reward Modeling for Reinforcement Learning-Based LLM Reasoning: Design, Challenges, and Evaluation

**arXiv ID:** 2602.09305 | [PDF](https://arxiv.org/pdf/2602.09305v1)

**作者:** Pei-Chi Pan `[一作]` (University of Houston), Sen Lin `[通讯]` (University of Houston)

**通讯引用:** 3126 | [OpenAlex ID](https://openalex.org/A5048422840)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统梳理并整合了大语言模型（LLM）推理任务中的奖励设计与强化学习（RL）策略，提出了统一框架“Reasoning‑Aligned Reinforcement Learning (RARL)”，并在此框架下给出了奖励机制的三维分类（架构、粒度、语义）以及奖励劫持（reward hacking）的根源与对策；进一步探讨奖励信号如何在推理训练、推理推理推理（Inference‑Time Scaling）、偏差缓解、增强推理（如检索、表格、工具调用）等多种场景中发挥统一作用，并批判现有基准与评价方法的缺陷。

**💡 创新点**

创新点主要包括：①提出 RARL 统一视角，将 RLHF、RLAIF、RLVR 等多种 RL‑推理方法归纳为同一框架；②构建三维奖励分类体系，揭示奖励模型在架构、粒度、语义上的多样性；③系统性分析奖励劫持的类别（信用分配、分布偏移、长度/位置/忠实度偏差）并给出对策；④展示奖励信号在推理推理、推理效率、偏差校正、工具集成等方面的跨域协同效应；⑤对现有基准进行深度剖析，指出数据污染与奖励不对齐问题，提出更稳健的评价路径。

**🔧 技术方法**

技术手段包括：传统 RL 算法（PPO、GRPO、DAPO、DPO、REINFORCE++、MCTS 等）与奖励模型（discriminative、generative、critique、self‑reward）相结合；奖励形变（length penalty、KL 正则、reward shaping、ensemble、retrieval‑augmented、token‑/step‑级奖励）；自监督与自我奖励机制（自一致性、entropy、self‑refinement）；基于 IRL 的内在奖励（log‑prob、Q‑值）以及多模态检索、表格、工具调用的强化学习框架。

**📊 数据集**

数据集与基准：数学推理数据集（GSM8K、MATH、FG‑PRM 等）、通用推理评测（ARC、OpenBookQA 等）、表格推理（FinQA、AIT‑QA、TableBench、BIRD、CTRLSciTab、TReB）、多模态表格（image‑to‑text）、检索增强生成（RAG、GraphRAG‑R1 等）以及自生成或 LLM‑judge 生成的奖励标签数据。

**📈 对比分析**

在多项推理基准上，采用 RL 训练的 LLM 相比单纯监督微调可提升 5‑10% 的准确率；结合奖励形变与长度惩罚后，模型在保持正确率的同时可显著压缩推理步骤；在推理推理场景中，RM‑guided 搜索与 Beam‑search 能在不增大模型参数的前提下逼近大模型性能；在检索增强、表格与工具集成任务中，奖励驱动的策略可实现 10‑15% 的准确率提升或 20‑30% 的推理步骤削减；但多数改进在复杂推理任务上仍相对有限，且不同 RL 算法之间的差距并不显著。

**⚠️ 局限性**

局限性包括：奖励模型易受长度、位置、忠实度等偏差影响，仍存在奖励劫持风险；离线 RL 的外推误差与奖励不确定性难以完全消除；自监督奖励信号与人类偏好不完全一致，可能导致逻辑与事实不符；基准数据常受污染与标注错误，评估结果不稳定；奖励模型训练成本高，缺乏通用的高质量标签；整体框架在跨领域迁移与多模态推理中尚未得到充分验证。

---

## 144. The effect of whitening on explanation performance

**arXiv ID:** 2602.09278 | [PDF](https://arxiv.org/pdf/2602.09278v1)

**作者:** Benedict Clark `[一作]` (Physikalisch-Technische Bundesanstalt), Stefan Haufe `[通讯]` (Charité)

**通讯引用:** 8361 | [OpenAlex ID](https://openalex.org/A5068256213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探究了在解释式人工智能（XAI）中使用数据白化技术能否缓解特征归因误差，系统评估了多种白化方法与多种XAI算法、模型架构的互相作用。

**💡 创新点**

创新点在于将16种主流特征归因方法与5种白化变换相结合，构建完整实验框架，同时在理论上解析了白化对二元抑制变量的影响，首次揭示白化并非万能的解释改进手段。

**🔧 技术方法**

采用的技术包括：特征白化（球面化、对称正交化、最优信号保留、Cholesky、偏回归）; XAI方法（LIME、LRP、Gradient SHAP、Integrated Gradients、Sobel/Laplace边缘检测）；三种模型（线性逻辑回归、MLP、CNN）以及基于精度与地球搬运距离（EMD）的解释评估指标。

**📊 数据集**

使用的主要数据集为XAI‑TRIS基准，包含四个8×8像素二分类图像任务（LINEAR、MULTIPLICATIVE、RIGID、XOR），每个任务配有无相关白噪声（WHITE）和相关噪声（CORR）两种背景，此外在理论分析中构造了二维抑制变量生成模型。

**📈 对比分析**

比较方法是将不同白化处理后的数据与未白化情况下的XAI输出在所有样本、模型和任务上分别计算精度与EMD；实验表明球面化、对称正交化和最优信号保留能显著提升解释准确性，达到或接近无相关背景的水平；偏回归和Cholesky效果不佳；不同XAI算法对白化的响应差异明显。

**⚠️ 局限性**

局限性包括：白化仅消除线性相关性，在非线性或高阶相关场景下仍无法完全消除抑制变量导致的误归因；Cholesky依赖像素顺序；偏回归在维数增大时失效；实验仅覆盖合成数据与特定XAI算法，缺乏在真实工业数据和更广泛方法上的验证。

---

## 145. Measuring Dataset Diversity from a Geometric Perspective

**arXiv ID:** 2602.09340 | [PDF](https://arxiv.org/pdf/2602.09340v1)

**作者:** Yang Ba `[一作]` (Arizona State University), Rong Pan `[通讯]` (Arizona State University)

**通讯引用:** 15177 | [OpenAlex ID](https://openalex.org/A5075012459)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于拓扑数据分析的持久性景观（PL）多样性度量方法

**💡 创新点**

创新点在于利用PL捕捉数据几何结构而非仅统计分布或熵，提供更丰富的多样性描述

**🔧 技术方法**

主要技术包括拓扑数据分析（TDA）、持久性景观（PL）、机器学习评估与对比实验

**📊 数据集**

在多模态数据集上进行实验，包括图像、文本、语音等公开数据集

**📈 对比分析**

与传统熵、特征空间散度等指标对比，PLDiv 在多样性评估上更精准、可靠且可解释，实验结果表明其在各类数据集上均表现优越

**⚠️ 局限性**

限制在于对高维稀疏数据的持久性计算成本较高，且需要对阈值与参数进行经验调优

---

## 146. A Lightweight Multi-View Approach to Short-Term Load Forecasting

**arXiv ID:** 2602.09220 | [PDF](https://arxiv.org/pdf/2602.09220v1)

**作者:** Julien Guité-Vinet `[一作]` (Université du Québec à Montréal), Éric Beaudry `[通讯]` (Université du Québec à Montréal)

**通讯引用:** 308 | [OpenAlex ID](https://openalex.org/A5049449659)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种轻量化的多视角短期负荷预测框架，利用单值嵌入和缩放时序范围输入，构建基于Informer的Transformer模型，并加入嵌入丢弃机制以提升鲁棒性与可解释性。

**💡 创新点**

创新点在于：①使用单值嵌入显著降低参数量；②采用缩放时序范围仅采样关键滞后（1y、1w、1d、12h等），保留重要季节性与短期模式；③设计嵌入丢弃策略，可单独评估每个视角的贡献。

**🔧 技术方法**

技术方法包括Transformer（Informer）、嵌入投影、嵌入丢弃、三种嵌入聚合方式（加法、拼接、单值SVD）、AdamW优化器、CosineAnnealing学习率调度等。

**📊 数据集**

实验使用四个真实负荷数据集：Ontario（IESO）、Hydro‑Québec（HQ）、Panama（CND）和一年的REMOTE（REM）数据，并补充温度、湿度、风速、降水、假日与校区等外部特征。

**📈 对比分析**

在1、2、7天预测周期内，与TFT、SARIMAX、TiDE、N‑BEATS等主流基准模型对比，模型参数仅为3万左右，却能获得相近或更优的MAPE（例如IESO 1天1.27% vs TFT 4.35%），并在假日与噪声场景下保持稳健。

**⚠️ 局限性**

局限性包括：①对外部变量精度假设过高；②在短期数据集（REM）上效果受限，可能需迁移预训练；③未探索自适应嵌入权重机制；④在非周期性领域（如金融）需进一步调整时序窗口。

---

## 147. Data-centric Design of Learning-based Surgical Gaze Perception Models in Multi-Task Simulation

**arXiv ID:** 2602.09259 | [PDF](https://arxiv.org/pdf/2602.09259v1)

**作者:** Yizhou Li `[一作]` (Case Western Reserve University), Zonghe Chua `[通讯]` (Case Western Reserve University)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5082074188)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文在机器人手术模拟器上构建了主动-被动眼动多任务数据集，并评估不同专业水平与观看模式对学习型视觉感知模型的影响。

**💡 创新点**

创新点在于系统比较主动与被动眼动监督来源与专业水平对模型学习的影响，证明被动专家或群众观看眼动可替代昂贵的手术眼动监督。

**🔧 技术方法**

使用了基于CNN的MSI‑Net和GAN的SalGAN等显著性预测模型，对视频帧进行空间注意力建模。

**📊 数据集**

数据集为在da Vinci SimNow模拟器上收集的四个训练任务（Sea Spikes、Ring Rollercoaster、Knot‑tying、Needle‑driving）的主动与被动眼动，参与者分为初学者和中级组。

**📈 对比分析**

通过KLD、CC、SIM、NSS等显著性指标评估模型，MSI‑Net在主动与被动监督下均保持0.23–0.54的相关性，NSS达3–6，显示被动监督可恢复约70%主动关注；SalGAN表现不稳定。

**⚠️ 局限性**

局限性包括中级组样本量小且为非外科医生，未包含专家眼动；任务D的主动与被动数据缺失，整体数据规模有限。

---

## 148. Dispersion of Gaussian Sources with Memory and an Extension to Abstract Sources

**arXiv ID:** 2602.09176 | [PDF](https://arxiv.org/pdf/2602.09176v1)

**作者:** Eyyup Tasci `[一作]`, Victoria Kostina `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

研究了独立但非同分布的源在有限块长下的无损压缩问题，并给出了二阶（源色散）近似公式；同时将该框架推广到带记忆的高斯源（如Gauss‑Markov），得到可直接用协方差特征计算的解析式。

**💡 创新点**

创新点在于：①提出点质量乘积代理测度（point‑mass product proxy），使得即使源的各分量不相同，也能保持失真与信息密度的可加性，从而构造典型集合并利用Berry‑Esseen定理完成二阶分析；②将i.i.d.情形的色散分析统一到非i.i.d.情形，并对已有的Gauss‑Markov色散结果做了更精细的余项改进。

**🔧 技术方法**

主要技术手段包括：信息率失真函数（rdf）的n阶定义、源色散的计算、d‑倾斜信息密度、Berry‑Esseen 定理的非独立变量推广、点质量代理测度构造典型集以及典型性概率估计。

**📊 数据集**

该工作为理论分析，未使用具体实验数据集；所有结果均为理论推导和解析表达式（如基于协方差矩阵的逆水位填充公式）。

**📈 对比分析**

与已有的i.i.d.源和单变量Gauss‑Markov源的二阶结果对比，作者证明在相同的失真阈值下得到相同的主项（rdf与色散），但余项从o(1/√n)提升到O(log n/n)，在理论精度上优于之前的结果。

**⚠️ 局限性**

局限性包括：①需要色散严格大于零且存在六阶矩上界；②对零色散情况未给出完整处理；③假设源分量独立但不必同分布，仍无法覆盖自回归高阶或非高斯记忆源；④理论分析主要针对单一失真测度（平方误差）和独立分量，通用性受限。

---

## 149. Marco IA593: Modelo de Gobernanza, Ética y Estrategia para la Integración de la Inteligencia Artificial en la Educación Superior del Ecuador

**arXiv ID:** 2602.09246 | [PDF](https://arxiv.org/pdf/2602.09246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 150. RuleFlow : Generating Reusable Program Optimizations with LLMs

**arXiv ID:** 2602.09051 | [PDF](https://arxiv.org/pdf/2602.09051v1)

**作者:** Avaljot Singh `[一作]` (University of Illinois Urbana-Champaign), Charith Mendis `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 510 | [OpenAlex ID](https://openalex.org/A5034476447)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了三阶段混合优化框架 RuleFlow：①离线利用 LLM 自动发现可改进的程序片段；②把发现的优化转化为可重用的 rewrite 规则；③将这些规则集成到编译器中，对新代码进行无 LLM 调用的静态匹配与重写。

**💡 创新点**

创新点主要在于：①将 LLM 的高创意发现与编译器的可靠部署分离，显著降低对 LLM 的实时调用成本和错误率；②引入桥接阶段，将具体优化泛化为通用规则，实现跨程序的高复用；③结合对等价性与性能的对抗验证，保证生成规则的安全性和有效性。

**🔧 技术方法**

核心技术包括：使用 GPT‑4.1 作为 LLM，配合 AI 代理完成候选生成、等价性检测、对抗验证和规则抽象；采用领域特定语言（DSL）描述 rewrite 规则并编译成静态匹配器；实现规则调度器以选择最优重写；所有步骤均通过大量自动化测试与性能基准验证。

**📊 数据集**

数据集：学习阶段使用 199 个来自 Kaggle 的 Jupyter Notebook（KGTorrent 收集），验证阶段使用 Pandas Benchmarks（102 个真实笔记本）以及其自带的 6 倍扩展尺寸，用于评估重写规则在不同规模上的适用性。

**📈 对比分析**

与先前的系统级 SOTA（如 PandasAI）和编译器级 SOTA（如原生 Pandas rewrite engine）进行对比。RuleFlow 在整体上比编译器 SOTA 提升 4.3×，比系统 SOTA 提升 1914.9×；在单个代码单元上，最高可达 199×（相较于原始代码）和 1704×（相较于系统级方案）。

**⚠️ 局限性**

局限性：①发现阶段仍依赖 LLM，产生的优化比例低（约 5.7%），需要大量候选和验证；②等价性检查仅基于有限随机测试，缺乏形式化语义保证；③规则的泛化程度受限，部分复杂场景仍需手工干预；④目前仅针对 Pandas 进行实验，推广到其他 API 需进一步研究。

---

## 151. P1-VL: Bridging Visual Perception and Scientific Reasoning in Physics Olympiads

**arXiv ID:** 2602.09443 | [PDF](https://arxiv.org/pdf/2602.09443v1)

**作者:** Yun Luo `[一作]` (Shanghai AI Laboratory), Ganqu Cui `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了首个开源 Vision‑Language 模型 P1‑VL，用于解决奥林匹克级别的物理问题。

**💡 创新点**

创新点在于结合了从难度逐步递增的 Curriculum Reinforcement Learning 训练和推理时的 Agentic Augmentation，实现了视觉与逻辑的无缝对齐。

**🔧 技术方法**

采用了多阶段强化学习框架 GSPO、Seq‑MIS、Rule‑Based 与 XVerify 验证器，并与 PhysicsMinions 代理体系协同工作。

**📊 数据集**

使用了 8,033 条含图像与文本的物理题目构成的多模态数据集（包含奥林匹克题和教材题）。

**📈 对比分析**

与 39 款闭源及开源模型对比，P1‑VL‑235B‑A22B 在 HiPhO 获得 12 金牌、1 银牌，平均得分 39.3，位列第三；在 FrontierScience‑Olympiad 也实现了显著提升。

**⚠️ 局限性**

局限性包括对 RL 训练的高计算需求、可能的训练‑推理不匹配问题，以及在更广泛非物理领域的泛化仍需进一步验证。

---

## 152. Patient foundation model for risk stratification in low-risk overweight patients

**arXiv ID:** 2602.09079 | [PDF](https://arxiv.org/pdf/2602.09079v1)

**作者:** Zachary N. Flamholz `[一作]` (Zephyr AI, Inc.), Jeffrey Sherman `[通讯]` (Zephyr AI, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建了一种基于神经时间点过程（TPP）的患者表示学习模型 PatientTPP，并在 500,000 条真实临床轨迹上训练。

**💡 创新点**

创新点在于将静态特征、连续数值流和预训练临床嵌入融入 TPP 框架，同时利用预测的未来事件来生成可解释的患者嵌入，用于多种下游预测任务。

**🔧 技术方法**

采用了多头注意力的 Attentive Neural Hawkes Process（AttNHP）架构，结合自编码器、时间稀疏处理与预训练词向量，训练目标为最大化事件类型与时间的对数似然。

**📊 数据集**

数据集来源于 Optum® Market Clarity，包含医学与处方索赔、实验室检验、诊断代码等，覆盖 2006 年至 2024 年的患者轨迹。

**📈 对比分析**

通过将 PatientTPP 嵌入用于预测心血管事件、癌症、住院等，AUROC 0.53–0.79；与 BMI 进行成本分层时，PatientTPP 的累计收益曲线提升 17%（相对提高 36%），表明在风险分层与资源分配方面优于传统方法。

**⚠️ 局限性**

局限性包括仅在单一美国数据库训练与评估、事件类型选择有限、未对时间‑到‑事件的显式预测进行评估，以及缺乏前瞻性临床验证。

---

## 153. Train Less, Infer Faster: Efficient Model Finetuning and Compression via Structured Sparsity

**arXiv ID:** 2602.09169 | [PDF](https://arxiv.org/pdf/2602.09169v1)

**作者:** Jonathan Svirsky `[一作]` (Bar Ilan University), Ofir Lindenbaum `[通讯]` (Bar Ilan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FineGates 方法，通过学习二值随机门控实现结构化稀疏化，使得在保持任务性能的同时可压缩 20%–40% 的权重并加速推理。

**💡 创新点**

创新点在于：① 用二值门控直接裁剪行/列而非加入额外低秩适配器；② 门控在训练期间即完成稀疏化，省去后置剪枝步骤；③ 理论上证明了比 LoRA 更简单、更好条件的优化景观与梯度下降收敛性。

**🔧 技术方法**

采用结构化稀疏化、二值门控（Gaussian‑relaxation + reparameterization）、梯度下降、ℓ0 正则化、偏置更新及 PL 条件下的收敛分析。

**📊 数据集**

在 GLUE（CoLA, STSB, MRPC, RTE, SST‑2, MNLI, QQP, QNLI, SST‑2）上验证；使用 RoBERTa‑Base/large、Llama3.2‑1B、Llama‑2‑7B；预训练阶段在 C4 数据集上进行。

**📈 对比分析**

与 LoRA、BitFit、VeRA、RoCoFT 等 PEFT 方案以及全量微调对比；在 Llama3.2‑1B 上的 GLUE 均可匹配或优于 LoRA，且训练参数仅 0.9M；在 MRPC 上推理速度提升约 25%，参数压缩 20%–40%；预训练中压缩 44% 参数并提升收敛速度。

**⚠️ 局限性**

限制主要包括：对稀疏度比例需人工设定；门控的二值化可能在极端稀疏情况下导致表达能力下降；在 GPU 上结构化稀疏化的加速效果不如无结构化剪枝；实现需要额外的门控训练与正则化超参。

---

## 154. The Similarity Control Problem with Required Events

**arXiv ID:** 2602.09360 | [PDF](https://arxiv.org/pdf/2602.09360v1)

**作者:** Yu Wang `[一作]` (Nanjing University of Aeronautics and Astronautics), Yixuan Li `[通讯]` (University of Edinburgh)

**通讯引用:** 465 | [OpenAlex ID](https://openalex.org/A5100443466)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究了在离散事件系统中引入必需事件后的相似控制问题，并给出了其可解性必要与充分条件以及最大允许性监督器的合成方法。

**💡 创新点**

创新点在于将协变-逆变模拟（covariant‑contravariant simulation）作为行为关系，引入 Σ_ucr‑可控性集合概念，桥接了相似控制与分区控制的理论，并提供了可计算的合成算法。

**🔧 技术方法**

主要使用了协变-逆变模拟、可控性集合的定义、集合运算、Tarski 定理的极大不动点、以及自动机的同步并、状态映射等离散事件系统与形式方法技术。

**📊 数据集**

该工作完全基于理论分析，无使用实验或真实数据集；所有结果均通过模型例子和形式证明给出。

**📈 对比分析**

通过理论证明与具体例子验证，展示了在满足必需事件约束的前提下能够合成最大允许的监督器，理论上保证了所有可行方案中的最宽松控制。

**⚠️ 局限性**

主要局限在于未讨论非阻塞性约束；未来研究需进一步扩展到非阻塞相似控制问题。

---

## 155. SemanticMoments: Training-Free Motion Similarity via Third Moment Features

**arXiv ID:** 2602.09146 | [PDF](https://arxiv.org/pdf/2602.09146v1)

**作者:** Saar Huberman `[一作]` (Tel Aviv University), Ron Mokady `[通讯]` (BRIA AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于语义特征的高阶时序统计（SemanticMoments）用于视频检索，专注于捕捉语义运动而非外观相似性。

**💡 创新点**

创新点在于使用训练无关、对预训练语义模型（如DINOv2）提取的补丁特征进行三阶时序统计（均值、方差、偏度），形成紧凑的运动表征，解决传统方法对静态外观的偏倚。

**🔧 技术方法**

技术包括：预训练视觉模型（DINOv2/VideoMAE/VideoPrism）、补丁级特征提取、时序高阶矩计算、加权拼接得到视频向量，以及无监督检索评估。

**📊 数据集**

使用了两个新基准：SimMotion‑Synthetic（合成视频对，控制外观变换）和 SimMotion‑Real（人工标注的真实视频对），并在公开数据集 Jester 上做姿态级检索验证。

**📈 对比分析**

与多种基线（CLIP4Clip、X‑CLIP、I3D、CoCLR、MaCLR、SlowFast、TimeSformer、VideoMAE、VideoPrism、DINOv2）比较，SemanticMoments 在 Synthetic 任务中平均提升 15–20% 的检索准确率，在 Real 任务中实现 10–12% 的性能提升，整体显示其对运动相似性的显著优势。

**⚠️ 局限性**

局限性包括：对极细粒度运动（如手势、呼吸）仍缺乏敏感度；依赖于预训练语义模型的质量；在高度变化或多主体复杂场景下检索效果仍有限；缺乏针对特定运动的微调能力。

---

## 156. EExApp: GNN-Based Reinforcement Learning for Radio Unit Energy Optimization in 5G O-RAN

**arXiv ID:** 2602.09206 | [PDF](https://arxiv.org/pdf/2602.09206v1)

**作者:** Jie Lu `[一作]` (Michigan State University), Huacheng Zeng `[通讯]` (Michigan State University)

**通讯引用:** 1792 | [OpenAlex ID](https://openalex.org/A5027120851)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了5G O-RAN中无线单元（RU）的睡眠调度与分布式单元（DU）的资源分片的联合优化，并实现了名为EExApp的xApp；

**💡 创新点**

创新点包括双演员双评估者PPO架构、利用Transformer对动态用户设备进行编码、以及使用双向图注意网络（GAT）协调两个评估者，以实现能耗与QoS的平衡；

**🔧 技术方法**

采用了深度强化学习（PPO）、Transformer编码器、图注意网络（GAT）、近实时RAN控制器（Near‑RT RIC）等技术；

**📊 数据集**

使用真实O‑RAN现场实验获得的关键性能指标（KPI）数据，包含八台智能手机在室内环境下的实际流量；

**📈 对比分析**

与单演员单评估器、Kairos、O‑RAN DRL等基线对比，实验显示EExApp在能耗降低、QoS违规率低、收敛速度快等方面均优于现有方法；

**⚠️ 局限性**

局限性在于仅在单一室内测试床验证，缺乏大规模网络与长期部署的泛化与稳定性验证。

---

## 157. "Create an environment that protects women, rather than selling anxiety!": Participatory Threat Modeling with Chinese Young Women Living Alone

**arXiv ID:** 2602.09256 | [PDF](https://arxiv.org/pdf/2602.09256v1)

**作者:** Shijing He `[一作]` (Kings College London), Jose Such `[通讯]` (INGENIO CSIC Universitat Politècnica de València)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开展了以参与式威胁建模（PTM）为核心的研究，调查中国独居年轻女性在智能家居、社交平台及公共基础设施中的隐私、安防与安全（PSS）风险，并基于研究结果构建了人本威胁模型与一份实用的 PSS 指南。

**💡 创新点**

创新点包括：① 在非西方（中国）社会文化背景下首次结合女性主义视角开展 PTM；② 揭示数字化物理暴力、网络诈骗（如深度伪造）与多尺度监控三者相互交织的风险网络；③ 识别并评估四类双刃缓解策略（智能家居、边界与关系管理、文化实践、社交媒体），并将洞察转化为可操作的安全指南。

**🔧 技术方法**

使用技术与方法：参与式威胁建模（PTM）框架、沟通隐私管理（CPM）理论、质性访谈与焦点小组、智能家居设备配置与使用分析、社交平台与深度伪造技术的使用情境分析。

**📊 数据集**

数据集：33 名中国独居年轻女性在六次 PTM 研讨会（共 99.3 分钟平均时长）中的录音转录文本，配合问卷反馈（共 31 人有效回复），以及后续指南使用评价数据。

**📈 对比分析**

比较方法：与传统安全/隐私模型对比主要通过参与者实际缓解策略与研究建议的一致性评估；性能方面指南在参与者问卷中获得 94% 以上的“清晰实用”评价，未进行数值性能对比。

**⚠️ 局限性**

局限性：样本规模有限、仅覆盖中文语境、依赖自我报告可能存在偏差、缺乏纵向跟踪验证指南长期效果、未与其他文化背景进行跨域对比。

---

## 158. Agile asymmetric multi-legged locomotion: contact planning via geometric mechanics and spin model duality

**arXiv ID:** 2602.09123 | [PDF](https://arxiv.org/pdf/2602.09123v1)

**作者:** Jackson Habala `[一作]` (Pennsylvania State University), Baxi Chong `[通讯]` (Pennsylvania State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对多足机器人，提出基于几何力学的接触规划框架，并将其转化为图优化问题，再利用自旋模型（Potts 与 Ising）的对偶性实现多足步态的高效全局优化；验证该方法能产生比传统三足/四足步态更快、更稳的非对称步态，并能在步态中让部分腿不驱动即可维持性能。

**💡 创新点**

① 把多足接触规划问题映射为图优化并通过自旋模型对偶性将复杂的组合优化降至多项式时间；② 发现并利用多足系统中新的对称破缺（如长慢逆时针转向与短快顺时针转向）来实现前进运动；③ 证明在物理上可将部分腿改为刚性非驱动件而不损失速度，提供形态与控制的协同设计思路。

**🔧 技术方法**

几何力学（局部连接矩阵与旋转力学）、摩擦力模型（RFT）、自旋模型（Potts/Ising）及其域墙搜索算法、图优化、仿真与实机（六足机器人）实验、强化学习基准（PPO）等技术。

**📊 数据集**

使用自制的六足机器人实验数据和仿真数据；未使用公开数据集，所有数据均来自本文实验与模拟。

**📈 对比分析**

与三足/四足生物启发步态、扩展四足步态、以及基于强化学习的开放式步态进行对比。实验结果显示物理启发步态在步频升高时可达 0.61 BL/周期（相较传统步态提升约 50%），强化学习步态仅 0.38 BL/周期，生物启发步态约 0.48 BL/周期；速度随步频呈二次关系，说明高频可提升动态稳定性。

**⚠️ 局限性**

① 仅适用于已知平坦无噪声环境，缺乏对不确定接触与地形的建模；② 需要先验的精确摩擦与动力学模型；③ 结果受限于几何力学假设（低Coasting数）和保守场近似；④ 对大规模多足系统的可扩展性与实际工况鲁棒性尚未验证。

---

## 159. Sparse Layer Sharpness-Aware Minimization for Efficient Fine-Tuning

**arXiv ID:** 2602.09395 | [PDF](https://arxiv.org/pdf/2602.09395v1)

**作者:** Yifei Cheng `[一作]` (Sun Yat-sen University), Li Shen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 15537 | [OpenAlex ID](https://openalex.org/A5100768717)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种 Sparse-Layer SAM（SL‑SAM）算法，通过在 SAM 的两步（梯度上升和下降）中仅对动态选出的部分层进行梯度计算，从而显著降低 SAM 的计算开销。

**💡 创新点**

创新点在于将层级稀疏选择视作多臂赌博机（multi‑armed bandit）问题，用梯度范数作为奖励来动态平衡探索与利用，使得稀疏化既能覆盖所有层，又能聚焦最关键层；同时将该思想推广到单步 SAM（SL‑S²‑SAM）上。

**🔧 技术方法**

技术手段包括：Sharpness‑Aware Minimization (SAM)、AdamW、梯度范数驱动的稀疏采样、EXP3 基的分布更新、单步 SAM 的梯度利用，以及在实验中使用的多种预训练模型。

**📊 数据集**

实验数据集覆盖视觉、自然语言和大语言模型：DeiT‑Small 在 CIFAR‑10/CIFAR‑100、RoBERTa 在 GLUE、以及 Llama‑3.2‑3B‑Instruction 在 Open‑Platypus。

**📈 对比分析**

与 AdamW、AdaSAM、RST、ESAM、SSAM‑F 等基线对比，SL‑SAM 在保持或略低于原始 SAM 的准确率（如 DeiT 98.12% vs 98.22%，RoBERTa 86.78% vs 86.92%，LLM 59.36% vs 59.29%）的同时，参数激活比例仅为原来的 0.45–0.21，GPU 内存和训练时间分别下降约 8–20%。

**⚠️ 局限性**

局限性包括：梯度扰动的时延可能导致性能略逊于完整 SAM，稀疏率需手动调节（s/N），在极小批量或层尺寸不均匀时效果不如全梯度；单步 SAM 扩展在梯度延迟问题上表现不如双步版本。

---

## 160. Tight Inapproximability for Welfare-Maximizing Autobidding Equilibria

**arXiv ID:** 2602.09110 | [PDF](https://arxiv.org/pdf/2602.09110v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Kelly Spendlove `[通讯]` (Google)

**通讯引用:** 108 | [OpenAlex ID](https://openalex.org/A5033899886)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

分析自动竞价二价拍卖中求解高质量均衡的计算复杂性，给出关于福利和收入的近似极限。

**💡 创新点**

首次证明福利最大化可逼近2-ε仍为NP‑hard，收入最大化存在对数级的不可近似性，并将这些下界与价格均衡边界（PoA）紧密联系，扩展至含ML先验和学习动态的情形。

**🔧 技术方法**

通过从Label Cover/最大覆盖问题构造的多种电路式Gadget（标签分配、NAND、NOT、边/条款竞争）实现多项式时间归约，利用价格均衡的可扩展性与保守扩展技术。

**📊 数据集**

无实际数据集，所有结果均为理论性的计算复杂度证明。

**📈 对比分析**

与已有的APX/NP‑hard性结果相比，本文提供了匹配PoA的紧上界；因未给出可实现算法，未给出实验性能指标。

**⚠️ 局限性**

仅考虑二价拍卖，未给出上界或实际实现；对更一般拍卖形式（如一价、混合拍卖）的可扩展性仍为开放问题。

---

## 161. Looping Back to Move Forward: Recursive Transformers for Efficient and Flexible Large Multimodal Models

**arXiv ID:** 2602.09080 | [PDF](https://arxiv.org/pdf/2602.09080v1)

**作者:** Ruihan Xu `[一作]` (Peking University), Shiliang Zhang `[通讯]` (Peking University)

**通讯引用:** 13394 | [OpenAlex ID](https://openalex.org/A5055433405)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种递归Transformer架构RecursiveVLM，利用同一组参数多次迭代来提升多模态模型的表示能力和推理质量；

**💡 创新点**

核心创新在于（1）递归连接器（Recursive Connector）通过RMSNorm、残差缩放和MLP实现跨步特征对齐，并对视觉与语言两种模态使用独立投影；（2）单调递归损失（Monotonic Recursion Loss）对每一步预测进行监督，并强制递归深度越深性能不下降，保证任意一步均能得到可用结果；

**🔧 技术方法**

技术手段包括递归Transformer、RMSNorm、MLP映射、模态特定投影、残差缩放、按层聚合、多步交叉熵损失与单调约束；

**📊 数据集**

使用的训练数据：先在Ming‑Lite‑Omni 28层预训练模型上进行持续预训练（6M多模态+3M文本），随后分别在两套指令数据集上微调——Data1（9M多模态问答+3M文本指令）与Data2（8M链式推理+2.6M文本实例）；评测使用8个多模态基准：AI2D、MM‑Star、MM‑Vet、MMMU、MMB、MathVista、OCRBench、HallusionBench；

**📈 对比分析**

实验对比基线为标准Transformer和原生递归Transformer。递归VLM在所有8项基准上均优于非递归基线，单步提升约+3%，两步提升约+7%。在HallusionBench上递归深度越大越好，展示了递归对消除幻觉的潜力；

**⚠️ 局限性**

局限性包括：（1）递归深度对大多数任务的收益有限，超过2步后增益趋于递减；（2）实验仅探讨R=1/2/3，未系统评估更深层递归；（3）主要验证在单一大型模型（28层）上，缺乏对小模型或其他架构的泛化验证。

---

## 162. Latent Poincaré Shaping for Agentic Reinforcement Learning

**arXiv ID:** 2602.09375 | [PDF](https://arxiv.org/pdf/2602.09375v1)

**作者:** Hanchen Xia `[一作]` (Shanghai Academy of AI for Science), Siyu Zhu `[通讯]` (Fudan University)

**通讯引用:** 2890 | [OpenAlex ID](https://openalex.org/A5013549550)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 LaPha 框架，将 LLM 的搜索和学习统一到根中心化的 Poincaré 隐空间中，实现稀疏可验证奖励的密集过程奖励、轻量级价值估计与语义空间剪枝。

**💡 创新点**

创新点在于利用负曲率的 Poincaré 空间对节点潜力进行几何塑形，从而将终端验证结果转化为密集奖励；同时通过共享隐空间训练轻量级价值头，实现无需额外价值模型的自我引导 MCTS；并在隐空间进行聚类剪枝以消除同义分支。

**🔧 技术方法**

技术包括：Poincaré 球模型的指数映射、超曲线距离、潜力奖励塑形、AlphaZero 风格的 MCTS、价值头与策略的联合 RL 更新（GRPO 变体）以及基于距离的剪枝算法。

**📊 数据集**

数据集主要使用 DAPO-Math-17K 进行训练，评估基于 AIME'24、AIME'25、OlympiadBench、MATH-500 以及 Gaokao'23（En）等数学推理基准。

**📈 对比分析**

与 SFT、DAPO、ToRL、SimpleRL、Prime 等基线对比，LaPha 在所有基准上均实现显著提升；在 1.5B 模型上 AIME'24/25 分别提升至 56.7%/43.3%，MATH-500 达 88.2%；7B 模型在 AIME'24/25 分别达到 60.0%/53.3%，MATH-500 92.0%。

**⚠️ 局限性**

局限性包括：对整数答案的偏好导致在自由形式答案（如符号、π、根式）上表现不如 1.5B 模型；对大型模型训练成本高；在极端深度搜索时仍可能出现同义分支的冗余，需进一步改进剪枝策略。

---

## 163. VLM-Guided Iterative Refinement for Surgical Image Segmentation with Foundation Models

**arXiv ID:** 2602.09252 | [PDF](https://arxiv.org/pdf/2602.09252v1)

**作者:** Ange Lou `[一作]` (Vanderbilt University), Tianyu Luan `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了IR‑SIS，一种基于视觉‑语言模型（VLM）的迭代细化框架，用自然语言描述自动完成外科图像分割并支持临床医生交互。

**💡 创新点**

创新点在于：①支持自由文本查询而非预定义类别；②利用VLM做质量评估和工具检测，形成自适应迭代工作流；③通过“agentic”决策机制实现自动和人工交互的混合细化；④构建三层级语言注释数据集。

**🔧 技术方法**

核心技术包括：SAM3的语言微调分割器、Qwen‑2.5‑VL‑32B（或GPT‑4o）作为工具检测器、遮罩覆盖率与框重叠率的质量指标、以及基于阈值的自适应重分割策略；训练时使用焦点损失、Dice损失、分类损失和存在性损失的组合。

**📊 数据集**

使用EndoVis 2017/2018两套外科影像数据集进行训练与评测，并在Kvasir‑Instrument数据集上验证跨域泛化；通过三层级（一般、类别、具体）语言标注扩充训练样本。

**📈 对比分析**

与SAM3原版、SAM3微调版以及TP‑SIS等基线相比，IR‑SIS在EndoVis测试集上的平均IoU提升约8–12%，Dice提升约6–10%；在Kvasir‑Instrument上同样取得领先，且在引入临床反馈后进一步提升约2–4%。

**⚠️ 局限性**

局限性包括：①对VLM检测器的依赖，若检测误报/漏报会影响细化效果；②系统运行时需多轮推理，推理时间和资源成本较高；③目前主要针对外科工具的分割，尚未验证对软组织或复杂多物体场景的适用性。

---

## 164. UniComp: A Unified Evaluation of Large Language Model Compression via Pruning, Quantization and Distillation

**arXiv ID:** 2602.09130 | [PDF](https://arxiv.org/pdf/2602.09130v1)

**作者:** Jonathan von Rad `[一作]` (University College London), Andreas Geiger `[通讯]` (University of Tübingen)

**通讯引用:** 50965 | [OpenAlex ID](https://openalex.org/A5016606943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套统一的LLM压缩评估框架，系统比较剪枝、量化与知识蒸馏三种压缩技术在性能、可靠性和效率三维度上的表现，并对压缩过程中的知识偏倚与校准效果进行深入分析。

**💡 创新点**

创新点包括：①将性能、可靠性与效率三维度统一纳入13项指标的评估框架；②通过大规模（40+）多任务基准揭示压缩技术对知识、推理、多语言和指令遵循的差异性影响；③提出了推理感知校准策略，可显著提升剪枝模型的推理能力；④发现压缩后性能保持与可靠性并不总是同步，凸显两者的独立性。

**🔧 技术方法**

技术手段：剪枝方法（SparseGPT、Wanda）、量化方法（GPTQ、AWQ）、知识蒸馏方法（Minitron、Low‑Rank Clone）；评估工具包括 vLLM、LightEval、lm‑evaluation harness；可靠性评测使用 GPT‑4o‑mini 作为判定器；效率评测通过 vLLM 计量吞吐量、延迟、显存等。

**📊 数据集**

使用的数据集涵盖：知识类（MMLU、ARC‑E/C）、推理类（MATH‑500、GSM8K、GPQA‑Diamond）、多语言与文化（Global‑MMLU‑Lite、BBQ）、指令遵循（IFBench）、可靠性（TrustLLM 体系中的 ConfAIde、MoralChoice、HaluEval、StereoSet、Do‑Not‑Answer 等）以及常识与安全基准（HellaSwag、PIQA、Winogrande、Commonsense Reasoning 等），总计 40+ 个基准。

**📈 对比分析**

比较方法：对每个压缩模型在 13 项指标上进行归一化得分，随后计算三大维度（性能、可靠性、效率）的综合得分；结果显示：量化在保持知识准确性同时拥有最佳的性能–效率平衡；剪枝在多语言、推理与指令遵循上损失较大；知识蒸馏在运行时加速与部署效率上表现最好，但训练成本最高；校准后剪枝模型的推理能力提升可达 50%。

**⚠️ 局限性**

局限性：评估以 LLaMA‑3.1‑8B 与 Qwen‑2.5‑7B 为主，模型范围有限；未覆盖代码生成、多智能体协作等特殊能力；校准策略仅针对推理任务；此外，框架对某些大模型的硬件加速支持不足，导致剪枝效率低。

---

## 165. Statistical Roughness-Informed Machine Unlearning

**arXiv ID:** 2602.09304 | [PDF](https://arxiv.org/pdf/2602.09304v1)

**作者:** Mohammad Partohaghighi `[一作]` (University of California), YangQuan Chen `[通讯]` (University of California)

**通讯引用:** 49985 | [OpenAlex ID](https://openalex.org/A5100715957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于层级统计粗糙度的机器模型遗忘方法——SRAGU，用来安全高效地从已训练模型中删除指定数据的影响。

**💡 创新点**

创新点在于将层级稳定性（通过层权重矩阵的重尾谱指数估计）与敏感性加权相结合，形成带有可控幅度的谱重加权，解决了传统基于敏感性更新易导致的局部崩溃问题。

**🔧 技术方法**

技术主要包括：重尾谱拟合、重尾指数映射为稳定权重、谱重加权后的适应性梯度更新以及使用敏感度信号和早停规则的无监督遗忘框架。

**📊 数据集**

使用了 MNIST、CIFAR‑10/100、ImageNet‑100 以及 UCI Adult 等四类常见图像与表格数据集进行评估。

**📈 对比分析**

与 SISA、AmnesiacML、SCRUB、SalUn、Boundary Unlearning、AGU 等多种训练时与后置遗忘基线及全量重训练 (ORTR) 对比，SRAGU 在保留准确率、预测偏差 (ε_pred)、KL 散度以及隐私泄露（MIA AUC）等指标上均优于所有近似遗忘方法，且在不同删除策略（随机、类别特定、对抗性）和模型架构下保持稳健。

**⚠️ 局限性**

局限性包括：缺乏形式化的隐私/安全保证、谱估计对小或结构特殊层的鲁棒性不高、需要额外的谱计算开销、对层定义和权重矩阵选择敏感，以及在极端删除比例或更大模型规模下的可扩展性待验证。

---

## 166. LLM-CoOpt: A Co-Design and Optimization Framework for Efficient LLM Inference on Heterogeneous Platforms

**arXiv ID:** 2602.09323 | [PDF](https://arxiv.org/pdf/2602.09323v1)

**作者:** Jie Kong `[一作]` (Shandong University of Science and Technology), Chen Yu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 75364 | [OpenAlex ID](https://openalex.org/A5100402072)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了LLM-CoOpt框架，针对大规模语言模型推理的内存带宽、计算冗余和长序列处理瓶颈进行全链路优化。

**💡 创新点**

创新点在于将KV缓存压缩与FP8量化、分组查询注意力（共享K/V）以及分页注意力的双阶段过滤与共享内存聚合三项技术有机融合，实现算法与硬件的协同优化，兼顾精度与效率。

**🔧 技术方法**

核心技术包括Opt‑KV（KV缓存写读过滤与FP8量化）、Opt‑GQA（动态分组查询注意力，减少头冗余）、Opt‑Pa（分页注意力双阶段过滤与共享内存软最大化）以及对异构平台（FP8仿真、64线程wavefront、L2/DRAM层级）下的细粒度内存调度。

**📊 数据集**

使用LLaMa系列（7B、13B、2‑7B、2‑13B、Pro‑8B）量化模型作为评测模型；数据集包含ShareGPT_V3_unfiltered_cleaned_split用于吞吐量/延迟评估，ARC（C/E）用于准确率验证。

**📈 对比分析**

与原vLLM相比，LLM‑CoOpt在LLaMa模型上实现了5.5%–12.1%吞吐量提升、5.5%–17.0%延迟下降，且ARC数据集上的准确率保持基本不变或略有提升，证明了方法的有效性。

**⚠️ 局限性**

局限在于仍需针对不同异构硬件进行手工调优，分页注意力的同步与碎片开销在极长序列上仍存在；此外FP8量化对Softmax精度的稳定性需要进一步验证。

---

## 167. Gradient Residual Connections

**arXiv ID:** 2602.09190 | [PDF](https://arxiv.org/pdf/2602.09190v1)

**作者:** Yangchen Pan `[一作]` (University of Oxford), Bo Liu `[通讯]` (University of Arizona)

**通讯引用:** 28851 | [OpenAlex ID](https://openalex.org/A5116279896)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证梯度残差连接，提升神经网络对高频函数的拟合能力

**💡 创新点**

将梯度向量作为残差加入网络，使用可调的凸组合参数，并提供理论解释梯度能区分高频点

**🔧 技术方法**

设计梯度残差模块、梯度归一化与正则化、可调参数、以及多种实验对照方案

**📊 数据集**

使用一维合成正弦回归、DIV2K 超分辨率数据集及其衍生集（Urban100、BSD100、Set5、Set14）以及 SRResNet 等基准

**📈 对比分析**

与标准残差、无残差、梯度模长等方法对比，实验表明在高频任务和超分辨率上可提升 0.1–0.2 dB PSNR，分类/分割任务性能基本保持不变

**⚠️ 局限性**

对高频任务有效，但对低频任务无明显收益；梯度计算成本高、对噪声敏感，且未与其他训练策略或更深层架构结合深入探索

---

## 168. PABU: Progress-Aware Belief Update for Efficient LLM Agents

**arXiv ID:** 2602.09138 | [PDF](https://arxiv.org/pdf/2602.09138v1)

**作者:** Haitao Jiang `[一作]` (North Carolina State University), Rui Song `[通讯]` (Amazon)

**通讯引用:** 6395 | [OpenAlex ID](https://openalex.org/A5089012283)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于任务进度的贝叶斯更新框架PABU，允许LLM代理在交互过程中只保留与任务进度相关的关键历史动作与观察，从而构建紧凑的贝叶斯状态；

**💡 创新点**

创新点在于：①将任务进度作为状态抽象，形成可解释、近似Markov的进度序列；②引入选择性保留机制，既保留最新观察、用户查询，又记忆同一进度下已尝试的动作；③将进度预测、保留决策与动作生成联合训练，实现全流程的自动化；

**🔧 技术方法**

主要技术包括：大规模LLM（如Llama-3.2‑1B）作为单一模型实现贝叶斯更新与动作决策；基于进度序列的监督标签生成；保留策略的学习（是否保留新观察）；进度一致动作增强；训练目标包含进度、保留与动作的交叉熵损失；

**📊 数据集**

数据集为AgentGym套件（8种任务）提供的成功轨迹AgentTraj‑L，用于生成进度序列与保留标签；实验评测亦在同一套件上完成；

**📈 对比分析**

与三类基线（专有API模型、开源通用模型、AgentGym调优模型）在任务成功率与交互步骤两指标上对比；PABU在81%成功率、9.5步平均交互（比SOTA高23.9%成功率、低26.9%步骤）显示显著性能提升；

**⚠️ 局限性**

局限性包括：仅基于离线轨迹训练，缺少在线自适应探索；进度抽象可能未涵盖所有状态细节；对少量任务或稀有失败场景的鲁棒性尚待验证。

---

## 169. A Hybrid Deterministic Framework for Named Entity Extraction in Broadcast News Video

**arXiv ID:** 2602.09154 | [PDF](https://arxiv.org/pdf/2602.09154v1)

**作者:** Andrea Filiberto Lucas `[一作]` (University of Malta), Dylan Seychell `[通讯]` (University of Malta)

**通讯引用:** 535 | [OpenAlex ID](https://openalex.org/A5082718586)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一套可解释、可审计的多阶段框架（ANEP），用于自动检测、识别并提取广播新闻视频中的人名；

**💡 创新点**

创新点在于：①创建了覆盖多样化新闻图形的News Graphics Dataset（NGD），②构建了基于YOLOv12的可解释检测器，③设计了分阶段的 OCR + Transformer NER 与聚类归一化模块，保持数据可追溯；

**🔧 技术方法**

采用的技术包括 YOLOv12 对象检测、图像增强与自适应阈值 OCR（Tesseract/自定义）、基于 BERT 的 NER、模糊匹配与嵌入相似度聚类、Grad‑CAM 可视化等；

**📊 数据集**

使用的数据集为 NGD：300 条视频、1,500 帧、4,749 个标注框，涵盖 Breaking News、Lower Thirds、Tickers 等六类图形；

**📈 对比分析**

与两种生成式多模态基线（Gemini 1.5 Pro 与 LLaMA 4 Maverick）对比，ANEP 在准确性上取得 79.9% 精度、74.4% 召回、77.08% F1，速度为 542s；生成式基线虽在 F1 上略胜（84.18%）但缺乏可解释性；

**⚠️ 局限性**

局限性包括：OCR 对非拉丁/装饰字体的识别不佳，聚类步骤计算量大导致运行慢，无法处理多语种或右到左文字，依赖外部 OCR/API，缺少音频/面部识别等多模态补充。

---

## 170. DMamba: Decomposition-enhanced Mamba for Time Series Forecasting

**arXiv ID:** 2602.09081 | [PDF](https://arxiv.org/pdf/2602.09081v1)

**作者:** Ruxuan Chen `[一作]` (Harbin Engineering University), Fang Sun `[通讯]` (Capital Normal University)

**通讯引用:** 690 | [OpenAlex ID](https://openalex.org/A5028363934)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种将时间序列分解为趋势与季节两部分，并分别采用低复杂度 MLP 与高复杂度 Mamba 模块进行建模的长周期预测模型。

**💡 创新点**

核心创新是依据趋势与季节特征差异匹配模型复杂度——趋势用简单 MLP，季节用强大的 Mamba，并使用 EMA 分解与可逆实例归一化。

**🔧 技术方法**

采用 EMA 分解、RevIN 归一化、双流 Mamba‑MLP 架构、可逆注意力扫描的 Mamba、分段残差加权损失等技术。

**📊 数据集**

在 ETT、Weather、PEMS、Electricity、Exchange 等公开多变量时间序列基准上进行评测。

**📈 对比分析**

与 S-Mamba、iTransformer、PatchTST、TimesNet、XPatch 等 10+ 基线进行对比，DMamba 在大多数预测步长上均实现 SOTA，平均 MSE 降低 10–20%，并在效率上也更优。

**⚠️ 局限性**

主要局限在于对 EMA 平滑系数的敏感性、模型对极端非平稳序列的泛化能力未完全验证，以及在极大维度时仍可能受 Mamba 参数规模限制。

---

## 171. Lateral tracking control of all-wheel steering vehicles with intelligent tires

**arXiv ID:** 2602.09427 | [PDF](https://arxiv.org/pdf/2602.09427v1)

**作者:** Luigi Romano `[一作]` (Linköping University), Erik Frisk `[通讯]` (Linköping University)

**通讯引用:** 2860 | [OpenAlex ID](https://openalex.org/A5070530122)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

开发了一种基于分布式轮胎动力学与智能轮胎传感器的全轮转向车辆横向跟踪控制器及状态观测器，能够抑制微摇晃并实现路径跟踪；

**💡 创新点**

首次将无限维PDE轮胎模型与智能轮胎技术结合，提出完整的输出反馈控制与观测设计，并在非线性双轨模型上验证其有效性；

**🔧 技术方法**

采用线性单轨 ODE‑PDE 控制理论、状态观测设计、回退控制与强制微分方程分析等技术；

**📊 数据集**

未使用公开数据集，而是通过高保真非线性双轨仿真模型（包含有限摩擦与侧向载荷转移）进行验证；

**📈 对比分析**

与传统经验/机器学习方法相比，提出方案在抑制低速微摇晃、轨迹跟踪误差 RMS 与最大误差方面表现优异，仿真误差均低于 0.3 m (纵向) 与 0.05 rad；

**⚠️ 局限性**

局限性包括：需要完整的智能轮胎测量与高昂传感器成本、线性模型假设、对极限操纵条件下的非线性滑移缺乏理论保证。

---

## 172. Dieu khien he da tac tu

**arXiv ID:** 2602.09412 | [PDF](https://arxiv.org/pdf/2602.09412v1)

**作者:** Minh Hoang Trinh `[一作]`, Hieu Minh Nguyen `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过图论与谱分析方法，提出并解析了多智能体系统的平均一致与双积分一致算法，给出了在有向图和无向图下收敛的必要充分条件及其收敛速率与拉普拉斯矩阵第二小特征值的关系；

**💡 创新点**

创新点在于将平均一致问题推广到非平衡有向图，明确提出权重平衡图与平衡图的定义，并利用Jordan标准形证明双积分一致系统在连通有向图下的收敛性；

**🔧 技术方法**

主要技术包括图拉普拉斯矩阵构造、谱分析、Jordan标准形、线性系统稳定性理论及MATLAB仿真；

**📊 数据集**

使用的是合成图数据集，如环图、完全图、随机图等；

**📈 对比分析**

通过MATLAB仿真比较不同图结构下的收敛速率，验证收敛速度与拉普拉斯矩阵第二小特征值（Fiedler值）呈正比，实验结果与理论分析相符；

**⚠️ 局限性**

局限性包括只适用于连通或平衡图，假设网络拓扑固定且无噪声，未讨论大规模网络下的计算复杂度和鲁棒性问题。

---

## 173. Single-Slice-to-3D Reconstruction in Medical Imaging and Natural Objects: A Comparative Benchmark with SAM 3D

**arXiv ID:** 2602.09407 | [PDF](https://arxiv.org/pdf/2602.09407v1)

**作者:** Yan Luo `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**通讯引用:** 2604 | [OpenAlex ID](https://openalex.org/A5100632182)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对单切片医学影像的零样本3D重建进行了系统基准测试，评估了五个先进的图像到3D模型在多种医学与自然数据集上的性能。

**💡 创新点**

创新点在于首次将基于自然图像的几何先验迁移到医学领域，直接对比各模型在单视角重建的表现，并揭示深度重建的根本瓶颈。

**🔧 技术方法**

使用的技术包括SAM3D、Hunyuan3D-2.1、Direct3D、Hi3DGen和TripoSG等基于扩散的图像到3D生成框架，配合Voxel-IoU、F-score@0.01、Chamfer距离与Earth Mover距离等评估指标。

**📊 数据集**

所用数据集涵盖六个医学数据集（AeroPath、BTCV、DukeCspine、MSD Lung、MSD Brain、MSD Liver）以及两个自然数据集（GSO、Animal3D），涉及CT、MRI及自然摄影。

**📈 对比分析**

比较结果显示，在局部体素指标上所有模型表现均有限，而在全局距离指标上SAM3D表现最优，取得最低CD和EMD，表明其几何先验在医学场景下迁移性最好。

**⚠️ 局限性**

局限性主要在于单切片缺乏深度线索导致重建深度失真，体素级匹配差距大；需引入多视角聚合或额外约束以实现更可靠的医学3D重建。

---

## 174. Timing and Memory Telemetry on GPUs for AI Governance

**arXiv ID:** 2602.09369 | [PDF](https://arxiv.org/pdf/2602.09369v1)

**作者:** Saleh K. Monfared `[一作]` (Worcester Polytechnic Institute), Shahin Tajik `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 1533 | [OpenAlex ID](https://openalex.org/A5002624062)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种不依赖可信硬件的 GPU 监测框架，设计四种计算原语（PoW、VDF、GEMM、VRAM 居留检测）来推断 GPU 在部署后是否被正常使用。

**💡 创新点**

创新点在于利用 GPU 架构特征（张量核利用率、HBM 访存带宽、内存硬函数）构造可观测的时延和内存访问模式，形成可在恶意或未受信任环境下的利用率推断信号；同时将多种工作负载组合成可扩展、可验证的测量流程。

**🔧 技术方法**

技术包括：
- 基于哈希的 PoW 与多实例 VDF 作为并行/顺序努力测量；
- 基于 CUTLASS 的 GEMM 踩线来激活张量核并产生可测量的吞吐；
- Argon2id 与 BLAKE2b 的密钥化哈希，用于高带宽内存压力与 VRAM 居留检测；
- CUDA、NVLink、PCIe 的低级性能计数与时间戳采样；
- 随机挑战生成与响应验证流程。

**📊 数据集**

实验数据集与场景包括：
- 大规模语言模型推理（如 2 GB、9 GB、12 GB 级别的 LLM 以及 72 B Qwen2.5‑72B）在 NVIDIA T4、H100 等 GPU 上运行；
- VRAM 居留测试使用 60 GB 随机挑战集（CHAL），并在 HBM 与主机预留内存两种模式下测量。

**📈 对比分析**

比较方法：采用时间分布直方图、方差分析与对比模型占用的 GPU 资源；实验结果表明：
- PoW 与 VDF 在与 LLM 并发时均能显著拉长响应时间，敏感度高；
- GEMM 通过张量核激活导致更大延迟差异，适合检测 tensor‑core 密集工作；
- VRAM 居留检测可在 350 ms 以上的延迟差下可靠区分热/冷访问，且对功耗与内存占用影响极小；
- 在 10 分钟基准下，连续测量方法带来 10–20 % 的功耗/吞吐下降，而 VRAM 检测几乎无额外开销。

**⚠️ 局限性**

局限性：
- 未给出正式阈值与统计检测框架，误报/漏报率未知；
- 需要在不同硬件与工作负载间进行参数调优，适配性仍待验证；
- 对极端攻击（如模拟 GPU 的 CPU/ASIC）尚未彻底评估；
- 频繁测量会占用 GPU 资源，影响业务吞吐；
- 方案仅提供概率性推断，不能替代强加密证明或完整的可追踪性。

---

## 175. Evaluating Social Bias in RAG Systems: When External Context Helps and Reasoning Hurts

**arXiv ID:** 2602.09442 | [PDF](https://arxiv.org/pdf/2602.09442v1)

**作者:** Shweta Parihar `[一作]` (University of Illinois), Lu Cheng `[通讯]` (University of Illinois)

**通讯引用:** 3094 | [OpenAlex ID](https://openalex.org/A5111957381)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了检索增强生成（RAG）系统对13种社会偏见的影响，发现外部检索语境通常能降低模型偏见，而引入链式推理（CoT）会反而提升偏见；通过“早期回答”法对CoT的可信度进行评估，揭示其在不同检索语境下的动态偏见演化；

**💡 创新点**

创新点在于（1）首次在多种检索语料（WikiText‑103、C4）与多类偏见数据集（StereoSet、CrowS‑Pairs、WinoBias、BOLD、HolisticBias）上统一评估RAG的公平性；（2）将CoT推理嵌入RAG，揭示推理过程与偏见之间的相互作用；（3）利用“早期回答”法量化CoT解释的可信度与偏见变化。

**🔧 技术方法**

技术手段包括标准RAG架构、LangChain+Chroma向量检索、CoT提示、Pearson相关分析以及“早期回答”法对CoT可信度的定量评估。

**📊 数据集**

使用的检索语料库为WikiText‑103和C4；偏见评估数据集有StereoSet（包含CrowS‑Pairs、WinoBias）、BOLD和HolisticBias；实验模型为Meta‑Llama‑3‑8B‑Instruct。

**📈 对比分析**

与无检索的基线相比，RAG在大多数偏见类型上降低了偏见评分（如SCW总体偏见从2.72降至2.31），但在加入CoT后偏见评分普遍上升；相关性分析显示CoT强化了偏见与评价指标之间的关联，体现了“准确率‑公平性”权衡。

**⚠️ 局限性**

局限性包括：仅测试单一LLM（Llama‑3‑8B‑Instruct）；检索语料和提示模板未进行广泛优化；偏见评估主要依赖现有基准数据集，未覆盖更复杂的真实场景；对CoT可信度的评估仅在少量样本上进行，缺乏大规模验证。

---

## 176. Breaking the Pre-Sampling Barrier: Activation-Informed Difficulty-Aware Self-Consistency

**arXiv ID:** 2602.09438 | [PDF](https://arxiv.org/pdf/2602.09438v1)

**作者:** Taewoong Yoon `[一作]` (Konkuk University), Harksoo Kim `[通讯]` (Konkuk University)

**通讯引用:** 731 | [OpenAlex ID](https://openalex.org/A5022865376)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ACTSC 框架，利用大模型内部 FFN 激活估计问题难度，并在推理时根据难度自适应控制自一致性（SC）的采样数量。

**💡 创新点**

创新点在于：① 通过识别难度敏感神经元（DSN）构造轻量级难度探测器；② 无需预采样或额外模型调用，直接在单次前向传播中得到难度评估；③ 在推理阶段即时动态分配采样预算，显著降低计算成本。

**🔧 技术方法**

核心技术包括：激活信号提取、DSN 识别、线性难度探测器训练、动态窗口自一致性采样、置信度停止机制。

**📊 数据集**

使用的基准数据集有：MATH‑500、AIME 2024/2025（数学推理），GPQA‑Diamond、MMLU‑Pro（非数学推理），以及多种 LLM 模型（Gemma3‑4B、Qwen2.5‑3B/7B）。

**📈 对比分析**

与 SC、AC、ESC、DSC 四种自一致性变体对比，实验显示 ACTSC 在保持甚至略优准确率的同时，平均采样数下降约 40%–80%，Token 成本大幅降低（如在 AIME‑2025 上从 137.5k 降至 74.5k）。

**⚠️ 局限性**

局限性包括：探测器训练需要一次离线成本且依赖模型特定激活；难度阈值需经验设定；对于极难问题仍可能需要大量采样，导致成本未完全消除。

---

## 177. A Behavioral Fingerprint for Large Language Models: Provenance Tracking via Refusal Vectors

**arXiv ID:** 2602.09434 | [PDF](https://arxiv.org/pdf/2602.09434v1)

**作者:** Zhenyu Xu `[一作]` (Texas Tech University), Victor S. Sheng `[通讯]` (Texas Tech University)

**通讯引用:** 12039 | [OpenAlex ID](https://openalex.org/A5051706630)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于拒绝向量的模型指纹提取方法，用于追踪大语言模型的衍生版本和知识产权。

**💡 创新点**

创新点在于利用安全对齐引发的拒绝行为的方向模式作为行为指纹，既能在模型细化、量化、合并等常见修改后保持高度相似，又能在对齐破坏攻击中显著下降，提供安全性取证信号；同时提出可公开验证的隐私保护哈希与零知识证明框架。

**🔧 技术方法**

核心技术包括：层级拒绝向量计算（对有害与无害提示的隐藏状态差异归一化）、层选择与聚合、余弦相似度与SimHash哈希、层次聚类、以及基于LSH与ZKP的公开验证协议。

**📊 数据集**

使用了7个基础模型（如 Llama‑3.1、Qwen2.5、Phi‑3 等）以及从 HuggingFace 取的 76 个衍生模型（包含量化、LoRA、SFT、合并等），并在这些模型上构造了有害与无害提示集。

**📈 对比分析**

与 HuRef、REEF 等白盒基准以及 Instructional Fingerprinting、Chain & Hash 等黑盒方法对比，本文方法在 48 个衍生模型上的 Top‑1 识别率为 100%，比黑盒方法高出 40%+，在大规模 76 模型实验中保持 100% 识别准确度，并在对齐破坏攻击下仍保持相似度 ≈0.5，明显高于未关联模型。

**⚠️ 局限性**

主要局限包括：需要完整的白盒访问，无法直接在仅提供 API 的环境中使用；对针对性的对抗性攻击（如针对拒绝向量的梯度优化）尚未评估；对更高级的 jailbreak 技术可能失效；公开验证框架尚处理论阶段，需要进一步实现与性能评估。

---

## 178. On the Subpacketization Level of the Banawan-Ulukus Multi-Message PIR Scheme

**arXiv ID:** 2602.09417 | [PDF](https://arxiv.org/pdf/2602.09417v1)

**作者:** Anoosheh Heidarzadeh `[一作]` (Santa Clara University), Anoosheh Heidarzadeh `[通讯]` (Santa Clara University)

**通讯引用:** 547 | [OpenAlex ID](https://openalex.org/A5006920802)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

分析了在Banawan和Ulukus的多消息PIR方案中计算子分包化水平时出现的线性递归，并推导出归一化子分包化水平L的显式表示。

**💡 创新点**

提出了一个关于子分包化水平L的多项式表示，显示其系数为非负，并且其主项为N^(K-D+1)/D。

**🔧 技术方法**

使用了线性递归和多项式表示法来推导和证明结果。

**📊 数据集**

没有具体提到使用的数据集。

**📈 对比分析**

通过推导和比较不同的递归关系，证明了L的性质和表现，显示出L是N的多项式，且具有非负系数。

**⚠️ 局限性**

论文没有明确指出限制，但可能存在对特定参数范围的依赖性。

---

## 179. SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking

**arXiv ID:** 2602.09401 | [PDF](https://arxiv.org/pdf/2602.09401v1)

**作者:** Ruochen Yang `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种名为SARM的端到端直播推荐框架，将多模态内容的自然语言描述（语义锚点）与排序目标直接联合训练，实现精准内容语义建模与实时推理的统一。

**💡 创新点**

创新点包括：① 用可学习的自然语言语义锚点代替离散标签或稠密嵌入，消除信息瓶颈；② 设计轻量化双token门控融合机制，既保留域特定词汇的语义，又兼容通用语言模型；③ 采用内存银行异步部署，保证实时在线训练与推理。

**🔧 技术方法**

关键技术：多模态大语言模型（MLLM）生成锚点；Live-Streaming Tokenizer与双token门控融合；轻量化BERT式语义锚点编码器（SAE）；跨注意力融合作者身份；辅助CTR任务稳定训练；内存银行实现低延迟在线服务。

**📊 数据集**

使用千亿级别的快手直播数据集：400M活跃用户、3M主播、数十亿交互记录，包含视频、音频、字幕、评论等多模态信息。

**📈 对比分析**

与传统两阶段方案（标签、SIDs、MMBee、SIM）以及基线HOme相比，SARM在多任务AUC/G-AUC上提升约1–2%，A/B实验显示点击率提升0.4%、礼物率提升0.8%等关键指标，且上线后维持低延迟与高吞吐。

**⚠️ 局限性**

局限性：① 需要离线使用MLLM生成语义锚点，成本与更新频率受限；② 对极高频主播的即时动态更新仍有延迟；③ 对极端实时性场景（极低时延）可能略逊；④ 依赖大规模内存库，存储与同步成本较高。

---

## 180. Effective vocabulary expanding of multilingual language models for extremely low-resource languages

**arXiv ID:** 2602.09388 | [PDF](https://arxiv.org/pdf/2602.09388v1)

**作者:** Jianyu Zheng `[一作]` (University of Electronic Science and Technology of China), Jianyu Zheng `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 3923 | [OpenAlex ID](https://openalex.org/A5066186474)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言预训练模型进行词表扩展，并利用双语词典与源语言词向量的加权平均来初始化新词向量，随后在低资源语言语料上继续预训练。

**💡 创新点**

首次在目标语言词表扩展中使用源语言词嵌入与双语对齐的加权平均初始化方案，显著提升低资源语言的下游任务性能。

**🔧 技术方法**

词表筛选、fastText 词向量训练、双语词典正交对齐、子词嵌入合成、加权平均初始化、mBERT 继续预训练。

**📊 数据集**

MADLAD‑400 低资源语料、WikiAnn NER 数据、Universal Dependencies POS 数据、fastText 预训练词向量、Minixhofer 双语词典。

**📈 对比分析**

与随机初始化词表的 mBERT 继续预训练基线对比，在 POS 上平均提升 0.54%，在 NER 上平均提升 2.60%，并保持对源语言（英语）的性能。

**⚠️ 局限性**

受限于双语词典规模、低资源语言评测数据缺乏，以及源语言词表筛选质量有待进一步提升。

---

## 181. Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems

**arXiv ID:** 2602.09387 | [PDF](https://arxiv.org/pdf/2602.09387v1)

**作者:** Fangye Wang `[一作]` (AMAP Alibaba Group), Pengjie Wang `[通讯]` (AMAP Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了HeMix模型，通过查询混合兴趣提取与异质混合结构实现高效的CTR预测

**💡 创新点**

创新点在于：①使用动态+固定查询的混合注意力同时捕获上下文感知与不变的用户兴趣；②HeteroMixer块替代自注意力，采用多头混合token、低秩交互与组对齐重建，显著提升异质特征交互效率与表达力；③实现可扩展的规模法律，模型参数可线性扩展而保持高效推理

**🔧 技术方法**

技术包括：嵌入+Tokenization、混合异质注意力、HeteroMixer（多头token融合、低秩交互、重构）、HeteroFFN、均值池化+MLP预测、基于TensorFlow的分布式训练与推理

**📊 数据集**

使用AMAP（AutoNavi Map）大规模日志数据，约4.4B条样本、>2M用户、>2.5M物品，包含全局与实时行为序列，平均全局序列长度332，实时序列长度10

**📈 对比分析**

与DLRM、DCNv2、AutoInt、Hiformer、MTGR、RankMixer等SOTA方法在CTR/CVR任务上对比；HeMix在100M参数下AUC提升约1.25%，在1.5B参数下提升约1.95%；在线A/B实验显示GMV+0.61%、PV_CTR+2.32%、UV_CVR+0.81%，均显著优于基线

**⚠️ 局限性**

局限性包括：①对超参数（token数、分组比例、低秩维度）敏感，需经验调优；②主要针对CTR/CVR任务，泛化到多任务或推荐业务需进一步验证；③虽然推理效率较高，但模型仍相对较大，部署在极低延迟环境时需进一步压缩

---

## 182. AfriNLLB: Efficient Translation Models for African Languages

**arXiv ID:** 2602.09373 | [PDF](https://arxiv.org/pdf/2602.09373v1)

**作者:** Yasmin Moslem `[一作]` (ADAPT Centre), Amanuel Gizachew Abebe `[通讯]` (African Institute for Mathematical Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发AfriNLLB轻量化多语言翻译模型，基于NLLB-200 600M通过层剪枝、量化及知识蒸馏，在非洲语言上实现高效翻译；

**💡 创新点**

创新点在于将迭代层剪枝与FP16量化结合，并通过知识蒸馏恢复质量，既压缩模型又保持或提升翻译表现，同时公开所有训练数据与模型；

**🔧 技术方法**

技术包括Transformer架构、迭代层剪枝、量化（FP16）、多阶段微调、序列级知识蒸馏、语言检测、语义过滤、质量估计以及CTranslate2推理加速；

**📊 数据集**

使用从OPUS、Hugging Face等公开资源收集的1.6M双语样本（英、法与13种非洲语言对），并以Flores200进行验证/测试；

**📈 对比分析**

与NLLB-200 600M baseline通过BLEU、chrF++、COMET等指标对比，迭代剪枝模型平均提升20–50%推理速度（FP16可达57%），且在质量上与baseline相近；

**⚠️ 局限性**

局限性包括数据仍偏向高资源语言、仅在decoder层剪枝未深入探究encoder剪枝影响、模型对极低资源语言的泛化仍有限，后续需扩展语言和改进数据增强技术。

---

## 183. Phase-Aware Policy Learning for Skateboard Riding of Quadruped Robots via Feature-wise Linear Modulation

**arXiv ID:** 2602.09370 | [PDF](https://arxiv.org/pdf/2602.09370v1)

**作者:** Minsung Yoon `[一作]`, Sung-Eui Yoon `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

为四足机器人开发了一套基于强化学习的滑板骑行控制框架（PAPL），实现了在模拟与真实环境中从起步、转向到冲刺的完整滑板行走技能。

**💡 创新点**

创新点包括：① 通过相位时钟引入滑板周期性建模，并在 actor/critic 网络中使用 FiLM 层实现阶段感知的可调特征；② 采用非对称特权学习与 DAgger 蒸馏，使得在仿真中利用完整状态训练后能够在仅凭可观测信息部署；③ 将视觉感知（腹部摄像头）与姿态估计结合，提升对外部扰动与光照变化的鲁棒性；④ 在一次从仿真到实物的零拷贝转移中展示了可行性。

**🔧 技术方法**

核心技术：基于 PPO 的深度强化学习、FiLM 条件化多层感知网络、异构 actor‑critic 架构、特权学习 + DAgger 蒸馏、域随机化、CNN‑GRU 感知编码、RGB 视觉分割、模拟动力学模型（滑板转向与推进）。

**📊 数据集**

数据集：在 Isaac Gym 上搭建 4096 个并行仿真环境，使用 Unitree Go1 四足机器人与尺寸为 0.69×0.27×0.13 m 的滑板；通过对机器人与滑板的质量、摩擦、重心、关节 PD 等内部参数进行域随机化，以构建多样化训练与测试样本；无公开真实数据集，全部来自仿真与少量真实实验。

**📈 对比分析**

比较方法：① 通过命令跟踪误差热图与覆盖率曲线评估不同组件（FiLM、视觉、特权信息）对性能的贡献；② 通过功率消耗直方图与三种运动模式（滑板、轮腿、单腿）对比，证明滑板在相同前进距离下功耗最低；③ 真实机器人零拷贝转移实验验证了从仿真到硬件的成功迁移。性能表现：PAPL 在大部分命令空间内误差 ≤0.3、无失控；在能耗上平均比其他两种模式低 15–25%。

**⚠️ 局限性**

局限性：① 仅在平坦或轻度不平地形验证，难以应对大幅不规则路面；② 对极端外部扰动或极低光照下的感知鲁棒性仍有提升空间；③ 当前缺乏高层导航与动态路径规划，无法自动调节相位周期以完成复杂任务；④ 需要大量仿真环境与 GPU 资源，训练成本较高。

---

## 184. Certified Gradient-Based Contact-Rich Manipulation via Smoothing-Error Reachable Tubes

**arXiv ID:** 2602.09368 | [PDF](https://arxiv.org/pdf/2602.09368v1)

**作者:** Wei-Chen Li `[一作]` (Georgia Institute of Technology), Glen Chou `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于梯度的接触丰富操作规划框架，利用可微凸优化模拟器对接触与几何进行平滑，并通过可达管道保证真实混合动力学下的约束满足和目标到达。

**💡 创新点**

创新点包括：① 引入可微凸优化模拟器实现接触与几何的可微平滑，并可求得对平滑参数的梯度；② 推导平滑误差的集合值上界，并将其纳入可达管道；③ 将平滑动态与时变线性反馈策略联合优化，使得梯度信息可用且真实系统能被保证在可达管道内；④ 通过梯度而非采样实现高效规划，突破传统RL/零阶方法的采样瓶颈。

**🔧 技术方法**

采用的技术主要有：可微凸优化（线性/二次规划）模拟器、kappa参数平滑、可达管道理论、序贯凸优化（SCP）和线性化、与TO‑CTR基准对比评估。

**📊 数据集**

实验数据主要来自仿真与真实机器人平台，包括平面推送、平面桶/盒子搬运、Allegro手立方体再定向等任务；硬件为Kuka iiwa 7臂和Allegro手；未使用公开数据集，而是自行收集的实验数据。

**📈 对比分析**

方法与TO‑CTR（含MPC与非MPC）对比，评估指标包括目标误差、约束违规率、轨迹成本与计算时间。结果显示本文方法的约束违规率降至0%，目标误差略低或相当，轨迹成本更低，计算时间略高但仍在可接受范围内。

**⚠️ 局限性**

局限性包括：① 仅考虑接触平滑误差，未建模质量、摩擦或几何参数的不确定性；② 依赖局部线性化，可能在强非线性区域失效；③ 仅适用于准动力学，无法处理高动态或工具使用场景；④ 仅得到局部最优解；⑤ 对几何平滑的需求导致可达管道在某些情形下失效。

---

## 185. CAPER: Constrained and Procedural Reasoning for Robotic Scientific Experiments

**arXiv ID:** 2602.09367 | [PDF](https://arxiv.org/pdf/2602.09367v1)

**作者:** Jinghan Yang `[一作]` (University of Science and Technology Beijing), Yifan Wu `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 32259 | [OpenAlex ID](https://openalex.org/A5000234334)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CAPER框架，专为科学实验室中长期操纵任务设计，能够在有限监督和低演示的环境下实现流程正确、鲁棒的机器人执行；

**💡 创新点**

创新点在于将任务推理、感知对齐与低层控制完全分离，采用LLM的链式思维做符号级规划，随后用多模态预测与VLM生成可执行的动作原语，并通过强化学习在低层实现对物理不确定性的自适应；

**🔧 技术方法**

技术上结合了LLM（Meta‑Llama‑3.1‑8B、GPT‑4o）进行符号规划，基于条件扩散模型的视觉预测，VLM（GPT‑4o）进行动作原语生成，以及DDPG强化学习实现低层控制；

**📊 数据集**

使用了自建的科学实验工作流基准（包含pick_place、pour、stir、mix、crystallize等多步任务）以及RLBench的长程操纵任务，实验还涵盖了从仿真到真实场景的转移；

**📈 对比分析**

与多种基线（单纯RL、LLM+RL、VLM+RL、CAPER无规划或无预测等）对比，CAPER在科学实验基准上平均成功率提升至约80%，在RLBench长程任务中亦表现最优，尤其在低样本和长程情境下优势明显；

**⚠️ 局限性**

局限性包括：在极长程任务中因视觉模糊与程序复杂度导致成功率下降；对物理执行的抓取稳定性不足（如薄工具）；以及真实场景中感知误差导致的性能衰减，这些问题主要集中在感知模块，可通过改进检测和深度估计来缓解。

---

## 186. Are Language Models Sensitive to Morally Irrelevant Distractors?

**arXiv ID:** 2602.09416 | [PDF](https://arxiv.org/pdf/2602.09416v1)

**作者:** Andrew Shaw `[一作]` (University of Washington), Amy X. Zhang `[通讯]` (University of Washington)

**通讯引用:** 2361 | [OpenAlex ID](https://openalex.org/A5037569091)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了情境无关的情绪干扰（moral distractors）对大型语言模型道德判断的影响，并构建了60个多模态干扰样本。

**💡 创新点**

首次将情境主义视角引入LLM道德评估，量化情绪干扰对模型道德选择的偏移，为评估LLM道德稳定性提供新方法。

**🔧 技术方法**

使用预训练LLM在现有道德基准（MoralChoice、r/AITA）上插入情绪干扰，计算边际道德行为概率（MMAP）与投票分布，并通过卡方检验评估显著性。

**📊 数据集**

情绪干扰样本来源于IDEST（文本）和OASIS（图像）共60条；评估基准为MoralChoice（约1367个情景）和r/AITA（250个日常伦理情境）。

**📈 对比分析**

与无干扰基线对比，负面干扰可使LLM道德行为下降超过30%，并显著改变判决分布；不同模型表现差异显著，GPT‑4.1对干扰不敏感。

**⚠️ 局限性**

受限于模型规模、仅使用英语基准、未探索多轮交互和多语言、未系统评估干扰强度和数量，以及缺少对其他道德框架的泛化验证。

---

## 187. It's not a lie if you don't get caught: simplifying reconfiguration in SMR through dirty logs

**arXiv ID:** 2602.09441 | [PDF](https://arxiv.org/pdf/2602.09441v1)

**作者:** Allen Clement `[一作]` (Subzero Labs), Alex Shamis `[通讯]` (Subzero Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

设计并实现了一个模块化的重配置引擎，使 SMR 节点能够在不耦合共识协议的前提下独立升级共识实现、成员集和容错阈值，并实现了极小的停机时间。

**💡 创新点**

创新点在于：①通过区分共识协议的内日志与外暴露日志，实现共识实现与重配置逻辑的解耦；②采用水平 Paxos 思路的三阶段协议（Prepare、Handover、Shutdown），支持任意成员变更与协议切换；③通过外部有效性校验、签名证书与信任链，保证在协议切换过程中的安全性与连贯性。

**🔧 技术方法**

主要技术包括：水平 Paxos 重配置框架、日志消毒器对内日志进行清洗与转化、签名证书与哈希链验证、三阶段协议控制、以及对外暴露的统一外日志接口。

**📊 数据集**

在 Rialo 区块链的本地测试环境中，模拟了 4→4、4→7、7→10、10→13 验证者的 epoch 转换；未使用公开数据集。

**📈 对比分析**

通过将 epoch 转换拆分为三阶段（从 EpochChange 提交到 Ready、Ready 到 Handover、Handover 到完成）进行测量。实验表明验证者规模对整体延迟影响极小，Ready→Handover 阶段约占总延迟的 93%，整体重配置时延在几十毫秒至几百毫秒范围内，满足最小停机目标。

**⚠️ 局限性**

局限性：①实验仅在本地集群完成，未评估跨地域、网络分区下的性能；②缺乏对极端网络延迟和节点动态加入/离开的细粒度分析；③重配置仍会在某些阶段产生短暂的延迟峰值，且在高容错阈值或大规模节点数下的鲁棒性尚未充分验证。

---

## 188. Autonomous Action Runtime Management(AARM):A System Specification for Securing AI-Driven Actions at Runtime

**arXiv ID:** 2602.09433 | [PDF](https://arxiv.org/pdf/2602.09433v1)

**作者:** Herman Errico `[一作]` `[通讯]` (Independent Researcher), Herman Errico (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 Autonomous Action Runtime Management (AARM) 规范，旨在在 AI 代理执行工具动作前拦截、累积上下文、基于策略与意图对齐评估，并生成防篡改的执行凭证。

**💡 创新点**

核心创新在于：① 将安全边界从模型输出转移到动作执行层；② 设计四类动作分类（Forbidden、Context‑Dependent Deny/Allow/Defer）以实现上下文感知决策；③ 规范化实现架构（协议网关、SDK 注入、eBPF/LSM、供应商集成）与合规要求，为行业提供可互操作的安全标准。

**🔧 技术方法**

主要技术包括：动作中介层、会话上下文聚合器、意图对齐与语义距离评估、策略引擎（支持 OPA/ Cedar 等）、决策执行机制（Allow/Denial/Modify/Step‑Up/Defer）、防篡改的接收器（哈希链、加密签名）以及多种实现架构的安全特性评估。

**📊 数据集**

论文为规范性工作，不涉及实验数据集，故未使用任何公开数据集。

**📈 对比分析**

未给出实验对比或性能评估；讨论了不同架构在吞吐量、延迟、绕过风险等维度的理论权衡，建议在实际部署中结合组织控制层级选择架构。

**⚠️ 局限性**

局限性：① 仅覆盖运行时动作安全，无法防御模型训练时的污染或供应链攻击；② 需要在代理层实现或供应商提供同步钩子，对 SaaS 环境的依赖度高；③ 对复杂语义推理与意图漂移的检测仍受限于上下文信息的完整性；④ 需与其他安全机制（SIEM、IAM、数据泄露防护）协同使用。

---

## 189. Reward-Guided Discrete Diffusion via Clean-Sample Markov Chain for Molecule and Biological Sequence Design

**arXiv ID:** 2602.09424 | [PDF](https://arxiv.org/pdf/2602.09424v1)

**作者:** Prin Phunyaphibarn `[一作]` (KAIST), Minhyuk Sung `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于 Metropolis–Hastings 的训练无关奖励引导采样器 Clean‑Sample Markov Chain，用于离散扩散模型的分子与生物序列生成。

**💡 创新点**

通过构造前向–后向传播的 proposal 分布，使接受概率仅依赖奖励，从而彻底消除对不平滑奖励函数产生的中间奖励噪声的依赖。

**🔧 技术方法**

结合离散扩散模型、Metropolis–Hastings 算法、前向–后向组合提议、以及多步逆向采样技术。

**📊 数据集**

实验使用 QM9、ZINC250K（分子数据集）以及 MPRA DNA 序列数据集。

**📈 对比分析**

与 Best‑of‑N、SMC、SVDD、SGDD 及训练带指导的 D‑CFG 等方法对比，Clean‑Sample 在所有奖励（QED、环数、SA、HepG2 活性）上均获得最高奖励，并在匹配的 NFE 或墙钟时间下往往优于其他基线。

**⚠️ 局限性**

理论收敛依赖离散扩散模型的 denoiser 完整性；若 x₀ 预测误差较大，接受概率近似失效；且多步逆向采样增加了计算成本。

---

## 190. Squeezing More from the Stream : Learning Representation Online for Streaming Reinforcement Learning

**arXiv ID:** 2602.09396 | [PDF](https://arxiv.org/pdf/2602.09396v1)

**作者:** Nilaksh `[一作]` (Mila - Quebec AI Institute), Sarath Chandar `[通讯]` (Chandar Research Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文在流式强化学习中引入自预测表征（SPR）辅助任务，并通过正交梯度更新和ObGD与SGD的冲突调和，提升了样本效率。

**💡 创新点**

创新点在于将SPR与流式RL结合，并设计正交梯度与ObGD兼容的更新机制，解决了高相关数据导致的训练不稳定。

**🔧 技术方法**

主要技术包括自预测表征、动量目标网络、正交梯度投影、ObGD优化器、SGD与OBGD的梯度冲突消除以及轻量级状态缓存。

**📊 数据集**

实验使用了Atari 26、MinAtar和Octax三大基准环境，评估了不同帧数下的平均、众数和四分位平均回报。

**📈 对比分析**

与传统的流式DQN、QRC(λ)和Stream Q(λ)基线相比，加入SPR后所有算法在样本效率和最终回报上均显著提升，尤其在Atari上取得最高的IQM回报。

**⚠️ 局限性**

局限性包括对ObGD的依赖导致在某些环境下仍有性能差距、辅助任务增加的计算和内存开销，以及未利用已学习的动态模型进行规划或持续控制任务验证。

---

## 191. Attention to details, logits to truth: visual-aware attention and logits enhancement to mitigate hallucinations in LVLMs

**arXiv ID:** 2602.09521 | [PDF](https://arxiv.org/pdf/2602.09521v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 192. SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity

**arXiv ID:** 2602.09386 | [PDF](https://arxiv.org/pdf/2602.09386v1)

**作者:** Yukun Zhang `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并实现了可扩展的稀疏多任务推荐框架 SMES，解决大规模工业环境下模型扩展与低延迟的矛盾。

**💡 创新点**

创新点在于进步式专家路由（共享专家+任务私有专家）和全局负载平衡正则化，显著抑制专家激活爆炸与负载失衡。

**🔧 技术方法**

技术核心包括稀疏 MoE、Top‑K 路由、deduped 专家执行、Grouped GEMM 计算、内存池分配与负载平衡正则化。

**📊 数据集**

使用公开的 KuaiRand‑1K 以及 Kuaishou 规模达 40 亿用户的工业数据集进行离线与在线评测。

**📈 对比分析**

与 MMoE、PLE、MoME、HoME、RankMixer 等基线对比，SMES 在 GAUC/AUC 上提升 0.1%–2.4%（离线），线上提升 0.31% 用户观看时长，且延迟仅增加 4 ms。

**⚠️ 局限性**

局限性：实现复杂，需要自定义稀疏路由与内存管理；在任务数极大或任务分布极端不均时仍可能出现专家负载不均的问题。

---

## 193. Unsupervised Cross-Lingual Part-of-Speech Tagging with Monolingual Corpora Only

**arXiv ID:** 2602.09366 | [PDF](https://arxiv.org/pdf/2602.09366v1)

**作者:** Jianyu Zheng `[一作]` (University of Electronic Science and Technology of China), Jianyu Zheng `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 3923 | [OpenAlex ID](https://openalex.org/A5066186474)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种完全无监督的跨语言词性标注框架，只使用单语料，借助无监督神经机器翻译（UNMT）生成伪平行句子，然后通过词对齐投影和多源投影校正训练目标语言的词性标注器。

**💡 创新点**

①首次将UNMT与词性投影结合，消除对平行语料的依赖；②提出多源投影校正技术，利用多源语言的重叠词汇提升投影质量；③在28个语言对上实现与传统平行语料方法相当甚至更优的性能。

**🔧 技术方法**

无监督神经机器翻译、词对齐（GIZA++）、POS投影、BiLSTM‑softmax 词性标注器、最大投票多源校正。

**📊 数据集**

大规模单语料（CC‑100 1000万句/语种），Europarl 伪平行句子（100k/源），UD（Universal Dependencies）目标语言评测集，CCMatrix/GNOME用于验证/测试。

**📈 对比分析**

与使用平行语料的跨语言投影基线（Eskander et al.）和监督词性标注器Stanza比较；在28个语言对上，平均比基线提升1.3%，在与源语言同族或接近的目标语（如阿非里卡语、葡萄牙语）上提升2.6–3.3%；整体准确率>60%，最高达92.0%。

**⚠️ 局限性**

对UNMT质量敏感，低相关或极其稀有语言的翻译质量差会导致投影错误；多源投影仍无法完全解决一对多/多对一对齐与词性投影错误；整体性能仍低于监督标注器Stanza。

---

## 194. Equilibrium contrastive learning for imbalanced image classification

**arXiv ID:** 2602.09506 | [PDF](https://arxiv.org/pdf/2602.09506v1)

**作者:** Sumin Roh `[一作]` (Sungkyunkwan University), Il Yong Chun `[通讯]` (Sungkyunkwan University)

**通讯引用:** 18446 | [OpenAlex ID](https://openalex.org/A5101911180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了Equilibrium Contrastive Learning (ECL) 框架，用监督对比学习方式解决图像分类中的类别不平衡问题；

**💡 创新点**

创新点在于引入 BC‑ECL 损失平衡类均值与类原型的贡献，并设计 CC‑GE 损失对齐分类器权重与类中心，形成三维几何平衡；

**🔧 技术方法**

采用监督对比学习、线性分类器、BC‑ECL、CC‑GE、Logit Compensation 交叉熵损失以及投影器和数据增强技术；

**📊 数据集**

使用长尾自然图像数据集 CIFAR‑10‑LT、CIFAR‑100‑LT、ImageNet‑LT，以及医学图像数据集 ISIC 2019 与自建肺癌 CT LCCT；

**📈 对比分析**

与六种现有 SOTA 监督 CL 方法（TSC、BCL、GLMC、PaCo、GPaCo、ProCo）以及基准学习方法对比，在所有五个不平衡数据集上均实现最高 Top‑1 准确率，尤其在极端长尾和少样本类别上显著提升；

**⚠️ 局限性**

仍需大批量训练；对非线性分类器的几何平衡效果有限；对其他模态的推广尚未验证。

---

## 195. Where-to-Unmask: Ground-Truth-Guided Unmasking Order Learning for Masked Diffusion Language Models

**arXiv ID:** 2602.09501 | [PDF](https://arxiv.org/pdf/2602.09501v1)

**作者:** Hikaru Asano `[一作]` (University of Tokyo), Yukino Baba `[通讯]` (University of Tokyo)

**通讯引用:** 839 | [OpenAlex ID](https://openalex.org/A5010710732)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对Masked Diffusion Language Models（MDLM）中的“先解码哪一位”决策，本文提出基于真值的门限（Gt‑Margin）作为解码顺序的oracle，并训练一个监督式的学习排序规划器，使得MDLM在不改动token预测模型的前提下提升推理质量。

**💡 创新点**

创新点包括：① 用Gt‑Margin（真值与最大竞争者概率之差）定义oracle解码顺序，揭示早期解码对最终质量的决定性影响；② 将oracle顺序转化为学习排序任务，采用PiRank（NDCG@k松弛）作为训练目标，训练可部署的where‑to‑unmask规划器；③ 发现仅在前半段解码使用规划器即可获得大部分提升，提出部分计划（partial‑plan）策略。

**🔧 技术方法**

核心技术包括Masked Diffusion Language Models、离散时间扩散过程、学习排序（learning‑to‑rank）与PiRank损失、Transformer backbone+MLP评分头、以及基于oracle的监督式训练。

**📊 数据集**

实验数据集涵盖逻辑推理与结构化任务：GSM8K、MATH、Sudoku 9×9、StrategyQA。

**📈 对比分析**

与随机、AR、逆AR、Top‑Prob、Margin、Gt‑Prob以及Gt‑Margin等基线相比，Gt‑Margin在GSM8K上从0.605提升到0.845；在MATH上提升至0.415；Sudoku上几乎全解；StrategyQA亦有明显提升。训练出的规划器在GSM8K和MATH进一步提升到0.705/0.285，展示了oracle信息的有效蒸馏。

**⚠️ 局限性**

限制包括：① 需要完整真值作为oracle训练的前提；② 对高度结构化任务（如Sudoku）规划器效果仍不如直接使用Margin；③ 规划器在后期解码中的作用有限，需与heuristic结合；④ 仅在单一模型（LLaDA‑8B/Dream‑7B）上验证，泛化性待进一步验证。

---

## 196. XLB: A High Performance Layer-7 Load Balancer for Microservices using eBPF-based In-kernel Interposition

**arXiv ID:** 2602.09473 | [PDF](https://arxiv.org/pdf/2602.09473v1)

**作者:** Yuejie Wang `[一作]` (Peking University), Guyue Liu `[通讯]` (Peking University)

**通讯引用:** 1376 | [OpenAlex ID](https://openalex.org/A5065693468)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

实现了基于eBPF的内核层7负载均衡器，替代传统的sidecar架构，实现了对HTTP/1.1、HTTP/2.0等协议的解析与路由；

**💡 创新点**

创新点在于将L7负载均衡逻辑移至内核socket层，利用eBPF实现同步拦截与消息重定向，避免跨进程、用户态与内核态的多余切换；

**🔧 技术方法**

使用eBPF、内核socket层扩展（proxy/instance socket）、嵌套eBPF map结构以及与Envoy控制平面交互的Go守护进程；

**📊 数据集**

主要采用Fortio生成的HTTP/1.1流量进行微基准测试，并在Kubernetes集群中部署Bookinfo和Bank of Anthos微服务实例；

**📈 对比分析**

与Istio v1.15.3、Cilium v1.12.2比较，XLB在吞吐量上提升至1.5×、平均延迟下降60%、尾部延迟降低约70%，在高并发、服务链长度、服务密度等多种实验场景均表现优越；

**⚠️ 局限性**

主要局限在于对eBPF编程的依赖和实现复杂性，以及无法高效实现TLS握手、鉴权等计算密集型任务，导致部分功能仍需留给用户空间实现。

---

## 197. SpotAgent: Grounding Visual Geo-localization in Large Vision-Language Models through Agentic Reasoning

**arXiv ID:** 2602.09463 | [PDF](https://arxiv.org/pdf/2602.09463v1)

**作者:** Furong Jia `[一作]` (Peking University), Yu Liu `[通讯]` (Peking University)

**通讯引用:** 46831 | [OpenAlex ID](https://openalex.org/A5022072267)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SpotAgent框架，将图像地理定位任务转化为基于ReAct循环的agentic推理过程，利用大型视觉语言模型（LVLM）主动调用外部工具（如网络搜索、地图编码、视觉放大）进行信息检索与验证；

**💡 创新点**

创新点包括：①通过多Agent ReAct生成高质量、可验证的训练轨迹，实现工具调用与视觉推理的分离；②引入Agentic Cold Start阶段培养工具使用能力；③设计空间感知动态过滤策略（Spatially-Aware Dynamic Filtering）作为RL的课程学习；④在RL阶段采用GRPO实现高效的奖励优化；

**🔧 技术方法**

技术手段涵盖：大型视觉语言模型Qwen2.5‑VL‑7B‑Instruct；ReAct框架与工具调用接口；多Agent ReAct数据生成；监督微调（SFT）；Agentic Cold Start；强化学习（GRPO）+地理距离奖励；空间感知动态过滤与两阶段课程学习；

**📊 数据集**

使用的数据集包括：MP16‑Pro（SFT）、Im2GPS3k、YFCC4K、Street View Text Dataset，以及通过多Agent生成的SpotAgenticCoT‑6k训练轨迹；

**📈 对比分析**

在Im2GPS3k上，SpotAgent在1km/25km/200km/750km/2500km分别达到14.12%/40.36%/57.80%/73.43%/85.75%，在无检索（retrieval‑free）方法中排名第二，显著优于GeoCLIP、G3、GLOBE等；在Street View Text Dataset上，工具协助模式相较无工具模式提升约9个百分点；整体表明主动工具调用与迭代验证显著提升定位精度；

**⚠️ 局限性**

局限性包括：①仍易受外部工具可用性与调用成功率限制；②长尾极端场景下视觉信息稀缺导致工具搜索仍可能失败；③训练与推理成本较高，依赖大规模算力与外部API；④模型仍可能出现格式化错误或工具调用错误，需进一步鲁棒性改进。

---

## 198. SCA-Net: Spatial-Contextual Aggregation Network for Enhanced Small Building and Road Change Detection

**arXiv ID:** 2602.09529 | [PDF](https://arxiv.org/pdf/2602.09529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 199. Look-Ahead and Look-Back Flows: Training-Free Image Generation with Trajectory Smoothing

**arXiv ID:** 2602.09449 | [PDF](https://arxiv.org/pdf/2602.09449v1)

**作者:** Yan Luo `[一作]`, Mengyu Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于流形测地线向量场学习的生成式采样方法，利用神经网络学习从数据点到测地线方向的映射，并通过测地线积分生成新样本。

**💡 创新点**

创新点在于将对数映射（log_x0(x1)）与投影操作相结合，构造目标速度 u_t，并用网络逼近测地线向量场 v_Θ，使得采样路径沿着流形内的自然测地线移动。

**🔧 技术方法**

采用了神经网络拟合、测地线方程数值求解、投影投影矩阵 P、以及自回归的积分步骤等技术。

**📊 数据集**

使用了在二维非线性曲面（z=-0.15(x²+y²)+0.1*sin(deg(x·y))）上的合成数据集，包含真实分布 p_0 与噪声分布 p_1。

**📈 对比分析**

与传统欧氏空间的随机梯度或核密度采样方法相比，实验表明该方法在保持数据分布一致性、生成样本多样性以及采样效率方面取得了较好的表现。

**⚠️ 局限性**

局限性包括：仅在低维流形上验证，随着维度提升计算成本急剧上升；对投影误差和测地线求解误差敏感；且需要先验了解目标分布的对数映射。

---

## 200. SWE-AGI: Benchmarking Specification-Driven Software Construction with MoonBit in the Era of Autonomous Agents

**arXiv ID:** 2602.09447 | [PDF](https://arxiv.org/pdf/2602.09447v1)

**作者:** Zhirui Zhang `[一作]` (International Digital Economy Academy), Heung-Yeung Shum `[通讯]` (International Digital Economy Academy)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布了SWE‑AGI基准，用来评估大型语言模型（LLM）在从规范驱动、固定API scaffold下，自动构建规模达1k–10k行代码的完整MoonBit软件系统的能力。

**💡 创新点**

创新点在于：①提出了规范驱动、检索防御的端到端生产级软件工程评测框架；②利用MoonBit的声明式API和统一工具链，使评测不受现有代码库偏差影响；③通过隐藏私有测试集保证评测结果不被局部测试覆盖；④结合行为日志分析揭示长周期工程中的阅读/调试瓶颈。

**🔧 技术方法**

技术手段包括：LLM代理与工具链（Codex CLI、Claude Code、Gemini、Kimi等）交互；MoonBit编程语言及其“declare‑pub”声明式API scaffold；自动化构建/测试/提交流水线；日志分类为Spec/Read/Write/Debug等行为指标。

**📊 数据集**

使用了22个基准任务，覆盖七类规范（如C99解析器、WASM解码器、SAT求解器等），每个任务基于RFC/标准规范并提供公共与私有测试集；代码量在1k–10k LOC之间。

**📈 对比分析**

评测方法：按易/中/难三层记录任务成功率、整体测试通过率、墙钟时间、核心LOC；结果显示gpt‑5.3‑codex 19/22（86.4%），gpt‑5.2‑codex 17/22（77.3%），claude‑opus‑4.6 15/22（68.2%），claude‑opus‑4.5 10/22（45.5%）；在易层，其他模型最高仅2/6。

**⚠️ 局限性**

局限性：在中、难层面性能明显下降，主要受限于规范阅读与大规模代码维护的瓶颈；评测仅覆盖单一语言MoonBit，缺乏跨语言/分布式系统的挑战；隐藏私有测试需人工维护；模型在硬任务的边缘案例和性能边界上仍易失效。

---

## 201. Scalable and Reliable State-Aware Inference of High-Impact N-k Contingencies

**arXiv ID:** 2602.09461 | [PDF](https://arxiv.org/pdf/2602.09461v1)

**作者:** Lihao Mai `[一作]` (Arizona State University), Yang Weng `[通讯]` (Arizona State University)

**通讯引用:** 3934 | [OpenAlex ID](https://openalex.org/A5021106309)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种基于状态感知的扩散模型和拓扑感知图神经网络的N‑k并发故障快速筛选框架，用于在不枚举全部组合的情况下直接生成高影响力的并发故障情景，并提供可控的覆盖保证。

**💡 创新点**

创新点包括：① 将N‑k故障筛选转化为状态条件的生成问题，使用条件扩散模型聚焦高风险区域；② 通过仅用基线和N‑1数据训练的EVGNN快速构造高风险训练样本；③ 引入概率覆盖控制策略，实现对漏检严重故障的可调节风险保证；④ 在在线评估中实现固定预算下的可扩展计算。

**🔧 技术方法**

核心技术包括：条件扩散模型（diffusion sampler）、拓扑感知EVGNN（edge‑varying graph neural network）用于风险评分、概率覆盖理论、基于严重性加权的损失函数、梯度引导的生成步骤。

**📊 数据集**

使用IEEE标准测试系统：14‑、39‑、57‑、118‑巴斯系统，生成多种负荷/发电条件下的操作点，构造基线+N‑1数据集并进一步采样高风险N‑k样本进行扩散模型训练。

**📈 对比分析**

与穷举、基于代理、EVGNN单独、cVAE、cGAN以及均匀随机采样等基线对比，实验表明：在相同的ACPF评估预算下，本文方法的前k平均严重度显著更高，收敛率和高严重度区间占比均优于基线；运行时间保持几乎不随k增大而增长，且整体比穷举和代理方法低数百倍。

**⚠️ 局限性**

局限性包括：① 需在离线阶段大量生成并标注高风险样本，仍受限于基线+N‑1数据的覆盖范围；② 对极端高压或极低负荷等极端状态的泛化能力尚待验证；③ 需要准确估计严重事件概率下界以决定采样预算，估计误差可能导致风险控制失效；④ 生成的故障组合仍需通过ACPF最终验证，无法完全消除求解成本。

---

## 202. A Universal Action Space for General Behavior Analysis

**arXiv ID:** 2602.09518 | [PDF](https://arxiv.org/pdf/2602.09518v1)

**作者:** Hung-Shuo Chang `[一作]` (Academia Sinica), Hong-Yuan Mark Liao `[通讯]` (Academia Sinica)

**通讯引用:** 35984 | [OpenAlex ID](https://openalex.org/A5108538229)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现一种基于大规模人类动作数据（Kinetics）构建的 Universal Action Space（UAS），在冻结该表示的前提下，用轻量级线性头对动物行为数据（MammalNet、ChimpBehave）进行识别。

**💡 创新点**

核心创新在于：①将人类动作训练得到的高维表征视为通用动作空间，可直接迁移到动物行为，省去对动物数据的深度网络预训练；②只需在冻结的 UAS 上做线性探针，显著降低训练时间与可训练参数；③通过子空间投影实现多任务共享。

**🔧 技术方法**

技术手段包括：Video Swin Transformer (VST) 作为特征提取器；线性分类器（I3D 头）；对比实验使用 LoRA、全微调等；使用热图可视化运动特征。

**📊 数据集**

数据集：Kinetics‑400/600/700（人类动作）；MammalNet（173 种哺乳动物 12 行为）；ChimpBehave（7 类黑猩猩行为）。

**📈 对比分析**

与全微调基线（MViTv2、X3D）比较，UAS+线性探针在 MammalNet 上 Top‑1 提升 21.5%（MCA +14.3%），训练时间仅 3.3% ；在 ChimpBehave 上 Top‑1 提升 ≥3.8%，MCA +7.6%，训练时间与参数仅 0.12%；在 Kinetics‑700 差集任务上线性探针与全微调相当，训练成本降低 51%/62%。

**⚠️ 局限性**

局限性：MammalNet 的高类数与跨物种差异导致每个类样本稀少，Top‑1 仅 56.6%；对极低样本量或高度多样化的动物行为仍需更丰富的数据或更深的子空间学习。

---

## 203. Computationally Efficient Replicable Learning of Parities

**arXiv ID:** 2602.09499 | [PDF](https://arxiv.org/pdf/2602.09499v1)

**作者:** Moshe Noivirt `[一作]` (Johns Hopkins University), Eliad Tsfadia `[通讯]` (Bar-Ilan University)

**通讯引用:** 116 | [OpenAlex ID](https://openalex.org/A5016947119)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研发了一种多项式时间、可复制的PAC学习算法，用于在任意分布下学习二元取模的 parity 函数。

**💡 创新点**

首次突破 SQ 限界，证明可复制学习可以在通用分布下实现高效 parity 学习，并提出新的可复制线性子空间识别子程序。

**🔧 技术方法**

结合稳定分区算法、可复制子空间识别、McDiarmid不等式及可复制线性系统求解技术，构建整个学习流程。

**📊 数据集**

算法在理论上针对任意 i.i.d. 样本（任意分布）工作，未使用具体公开数据集；适用于任何分布。

**📈 对比分析**

与仅适用于 SQ 可学习或受限分布的可复制算法相比，本方法在 O(m²d³) 运行时间下实现多项式样本复杂度 poly(d,1/ρ,1/ε,log(1/δ))，在通用分布上达到 PAC 收敛。

**⚠️ 局限性**

仅在 realizable 场景下提供可复制性与正确性保证，无法处理噪声或非 realizable 任务；对噪声学习的可复制性仍是未解决的问题。

---

## 204. Computing Conditional Shapley Values Using Tabular Foundation Models

**arXiv ID:** 2602.09489 | [PDF](https://arxiv.org/pdf/2602.09489v1)

**作者:** Lars Henry Berge Olsen `[一作]` (Norwegian Computing Center), Dennis Christensen `[通讯]` (Norwegian Defence Research Establishment)

**通讯引用:** 6452 | [OpenAlex ID](https://openalex.org/A5034593036)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用 TabPFN 计算条件 Shapley 值，并在单独回归和代理回归两种框架下进行实验。

**💡 创新点**

首次将 tabular foundation model TabPFN 作为条件 Shapley 值估计器，尤其在单独回归中实现最优性能。

**🔧 技术方法**

采用 TabPFN（v2、v2.5-D、v2.5-R）以及传统回归、树模型和 Monte Carlo 采样等多种技术进行对比。

**📊 数据集**

在五个 UCI 公共数据集（包含连续与混合特征）以及仿真高斯数据集上评估。

**📈 对比分析**

与现有最优方法相比，TabPFN 在大多数场景下取得更低的 MAE/MSEv，且运行时间显著缩短（从数百倍到数十倍）。

**⚠️ 局限性**

对非平滑预测函数和代理回归时表现略逊，同时受限于上下文大小和缺失模式覆盖不足。

---

## 205. FD-DB: Frequency-Decoupled Dual-Branch Network for Unpaired Synthetic-to-Real Domain Translation

**arXiv ID:** 2602.09476 | [PDF](https://arxiv.org/pdf/2602.09476v1)

**作者:** Chuanhai Zang `[一作]` (Zhejiang University), XW Song `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出频率分离的双分支无对齐翻译模型 FD-DB，实现合成图像向真实域的转换，保持几何结构与语义一致。

**💡 创新点**

创新点在于将低频外观归一化拆解为可解释的图像参数编辑（白平衡、曝光、对比度、饱和度、模糊、噪声），高频细节由残差分支补偿，并通过频域约束、低频锚定以及两阶段训练显著提升结构稳定性。

**🔧 技术方法**

采用频率域分解与重构、可解释的图像编辑算子、PatchGAN 对抗、PatchNCE 对比学习、低频锚定约束以及两阶段训练调度。

**📊 数据集**

使用 YCB‑V 数据集（21 类物体的合成与真实图像）进行实验。

**📈 对比分析**

与仅训练真实、仅训练合成、合成+微调、单独编辑分支、单独残差分支、完整 FD-DB 等方案对比；在 YCB‑V 上语义分割 mIoU 从 0.2768（仅合成）提升至 0.6533（完整 FD-DB），接近仅真实训练的 0.7018。

**⚠️ 局限性**

局限性包括高频分支仍可能产生颜色漂移，需调节高通阈值；实验集中在语义分割，未验证在检测或姿态估计等其他几何敏感任务上的泛化效果。

---

## 206. ArtifactLens: Hundreds of Labels Are Enough for Artifact Detection with VLMs

**arXiv ID:** 2602.09475 | [PDF](https://arxiv.org/pdf/2602.09475v1)

**作者:** James Burgess `[一作]` (Stanford University), Kuan-Chieh Jackson Wang `[通讯]` (Snap Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了一种名为ArtifactLens的系统，用于检测AI生成图像中的人类解剖伪影。该系统通过对图像进行标记（伪影或非伪影）来实现伪影检测。

**💡 创新点**

创新点在于利用预训练的视觉语言模型（VLMs）进行伪影检测，而无需对其进行大规模微调。通过使用少量标记样本（每个伪影类别仅需几百个），该方法实现了最先进的性能。

**🔧 技术方法**

使用了多组件架构和黑箱优化技术，结合了上下文学习（ICL）和文本指令优化的方法。

**📊 数据集**

使用了五个不同的人类伪影基准数据集进行评估，包括SynArtifact、AbHuman、Human Artifact Dataset (HAD)、AIGC Human-Aware-1K和MagicBench。

**📈 对比分析**

与现有的微调模型相比，ArtifactLens在F1分数上提高了8%，且仅使用了10%的数据。即使使用200个训练样本，性能也仅下降9%。该方法在多个基准数据集上表现出色，且在零-shot VLMs的基础上，所有模型的性能均提高了至少45%。

**⚠️ 局限性**

限制在于该方法依赖于预训练的VLMs的能力，且在处理新类型伪影时可能需要额外的标记数据。此外，尽管该方法在多个基准上表现良好，但在某些特定情况下可能仍然存在性能下降的风险。

---

## 207. LLM-Grounded Dynamic Task Planning with Hierarchical Temporal Logic for Human-Aware Multi-Robot Collaboration

**arXiv ID:** 2602.09472 | [PDF](https://arxiv.org/pdf/2602.09472v1)

**作者:** Shuyuan Hu `[一作]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society), Tianwei Zhang `[通讯]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于LLM推理的神经符号框架，将自然语言任务转化为层次化的LTL_f规范，并在递归视野规划循环中实时动态规划多机器人协作任务；

**💡 创新点**

创新点在于将LLM生成的子任务嵌入层次化LTL_f中，利用递归视野规划实现实时重规划与安全约束，同时显著降低LLM推理令牌消耗；

**🔧 技术方法**

核心技术包括LLM推理、Open‑Vocabulary 3D 感知、Hierarchical LTL_f 规范、递归视野规划（RHP）、多机器人团队模型（Product Team Models）以及并行运动规划；

**📊 数据集**

实验数据集涵盖基于真实RGB‑D相机的手部与机器人协作场景，以及多任务、多机器人、不同拓扑结构的仿真环境；

**📈 对比分析**

与 SMART‑LLM 等基线对比，本文方法在成功率、执行时间、令牌使用率上均优于对手，尤其在任务复杂度提升时保持高成功率并显著降低重规划次数；

**⚠️ 局限性**

局限性包括对大规模机器人队列的扩展性尚待验证、对LLM推理速度的依赖、以及在极端动态环境下的鲁棒性仍需进一步提升。

---

## 208. Toward Linking Declined Proposals and Source Code: An Exploratory Study on the Go Repository

**arXiv ID:** 2602.09467 | [PDF](https://arxiv.org/pdf/2602.09467v1)

**作者:** Sota Nakashima `[一作]` (Kyushu University), Hidenori Matsuzaki `[通讯]` (DENSO CORPORATION)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大语言模型（LLM）构建三阶段管道，将被驳回的 Go 项目提案与源代码的不同粒度（目录、文件、函数）之间建立可追溯性链接。

**💡 创新点**

首次提出被驳回提案与源代码之间的链接任务；设计粒度决策、LLM驱动定位与链接决策三阶段管道；揭示链接性能高度依赖提案讨论内容而非长度。

**🔧 技术方法**

采用 DeepSeek‑V3.1 LLM（温度 0.0）与检索增强生成（RAG）相结合的定位和链接决策流程；通过提示工程实现粒度决策与二元链接判断。

**📊 数据集**

以 Go 官方提案库为数据集：448 个被驳回提案（用于评估）和 296/262 个已接受提案（用于基线对比）。

**📈 对比分析**

与 RAG‑基线方法（基于嵌入检索+LLM）对比：在已接受提案上，所提管道实现更高的 F1 与 21/22 例子中更高的精度；在被驳回提案上，粒度准确率为 0.836，平均链接精度 0.643（未评估召回）。

**⚠️ 局限性**

局限性：缺乏实现代码导致无法自动评估召回；评估主要基于人工判断，存在主观性；实验仅限 Go 项目，泛化性待验证。

---

## 209. Bounded Modal Logic

**arXiv ID:** 2602.09462 | [PDF](https://arxiv.org/pdf/2602.09462v1)

**作者:** Yuito Murase `[一作]`, Akinori Maniwa `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种称为 Bounded Modal Logic (BML) 的新型构造模态逻辑，并给出了其自然演绎证明系统、Kripke 语义、对应的 λ-算子以及完整性与一致性证明。

**💡 创新点**

创新点在于：①在传统模态逻辑基础上引入了“bounded modality”^≽γ，使得既能描述模态转移又能显式捕捉与 intuitionistic（作用域）转移的交互；②将一阶分类器(代表作用域)与模态算子结合，形成双关系 Kripke 结构；③证明 BML 可视为 S4 的推广，并提供两种方向的对应关系。

**🔧 技术方法**

使用的技术包括：
- 双关系 Kripke 结构（intuitionistic 预序 ≼ 与模态预序 ⊑）
- 带标签的自然演绎证明系统与 λ-算子（包含类标签、束缚类量化、模态引入/消除）
- Kripke 语义的构造与证明完整性
- 归约语义与可归约性、强归约性、规范化与唯一性证明
- 通过翻译与转换证明 BML 与 S4 的等价性。

**📊 数据集**

无实验数据集；本研究为理论性逻辑与类型系统研究，主要通过形式化证明与结构化推理完成验证。

**📈 对比分析**

比较方法：将 BML 与 S4 通过语义转换（M*→M! 与 flattening）以及证明系统翻译进行对应；在语义层面证明两者满足同一满足关系；在证明层面证明 ε⊢_4 A ⇒ ε⊢ (A)^≽，并期望反向亦成立。性能方面，所有结果均为形式化的逻辑性质（满足性、完整性、可归约性、强归约性、唯一化正规形），没有数值评估。

**⚠️ 局限性**

限制与未来工作：
- 本文仅证明理论基础，未展示在实际编程语言或多阶段编程中的可行性与性能；
- 对运算行为与时间顺序归约性等运算性问题缺乏深入分析；
- BML 目前仅覆盖 S4，扩展到 K、K4、T 等模态逻辑仍需进一步研究；
- 对于更强模态逻辑如 S5、GL 等，直接改造证明系统可能不充分。

---

## 210. Taming the Monster Every Context: Complexity Measure and Unified Framework for Offline-Oracle Efficient Contextual Bandits

**arXiv ID:** 2602.09456 | [PDF](https://arxiv.org/pdf/2602.09456v1)

**作者:** Hao Qin `[一作]` (University of Arizona), Chicheng Zhang `[通讯]` (University of Arizona)

**通讯引用:** 709 | [OpenAlex ID](https://openalex.org/A5066555678)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种名为 Offline Estimation to Decisions（OE2D）的算法框架，将具有一般奖励函数逼近的上下文无关学习问题归约为离线回归，并通过“exploitative F-design”实现探索-利用的平衡；给出对该框架的下界分析和上界证明。

**💡 创新点**

创新点包括：①引入 Decision‑Offline Estimation Coefficient（DOEC）这一新的复杂度度量，能同时体现探索覆盖和利用收益；②通过 DOEC 与已知的 Decision‑Estimation Coefficient（DEC）建立桥梁，首次统一了在线与离线回归 oracle 的设计原理；③改进的 F‑design 方案将纯探索 F‑design 与贪婪奖励最大化融合，实现 O(log T) 次离线回归调用；④在多种典型情形（离散动作、每场景线性奖励、h‑平滑奖励）下给出近似最优的 regret 上界。

**🔧 技术方法**

技术手段包括：离线回归 oracle（ERM）、实验设计中的非线性 F‑design、最小化覆盖误差与奖励差异的双目标优化、覆盖度量 ε‑coverage、基于 ε‑SEC 的结构化分析、坐标下降算法求解 F‑design、概率论中的高斯和集中不等式、以及对误差传播的逐步校准。

**📊 数据集**

论文主要为理论研究，未使用具体公开数据集；所有证明均基于假设的 IID 上下文与可实现性（realizability）条件下的函数族。

**📈 对比分析**

与现有方法相比：在离散动作空间下实现 √(N T log N) 的 regret，oracle 调用数从 O(T) 降至 O(log T)（若已知 T 可进一步降至 O(log log T)）；在每场景线性奖励与 h‑平滑奖励情形下同样得到最优 √(d T log N) 与 √(T/h log N) 的 regret；总体性能与先前最优算法相当，但在 oracle 调用和覆盖设计上更为简洁高效。

**⚠️ 局限性**

局限性：①仅在 realizable、IID 上下文模型下给出上界；②对非 i.i.d. 或自适应分布转移、非线性奖励等更复杂情形的理论支持有限；③对算法的计算复杂度（尤其是 F‑design 的优化求解）仅给出存在性与迭代上界，缺乏实用实现细节；④尚未提供信息理论下界或 DOEC 的下界分析，导致对算法极限尚不完整。

---

## 211. Conceptual Cultural Index: A Metric for Cultural Specificity via Relative Generality

**arXiv ID:** 2602.09444 | [PDF](https://arxiv.org/pdf/2602.09444v1)

**作者:** Takumi Ohashi `[一作]` (Hosei University), Hitoshi Iyatomi `[通讯]` (Hosei University)

**通讯引用:** 4392 | [OpenAlex ID](https://openalex.org/A5023899090)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种句子级文化特异性度量——Conceptual Cultural Index（CCI）。

**💡 创新点**

创新点在于将文化特异性定义为目标文化与其他文化一般性估计之差，既可解释又可通过比较设置灵活控制文化范围。

**🔧 技术方法**

主要技术是利用大型语言模型对句子在各文化中的通用度进行推断，并计算平均差值得到CCI。

**📊 数据集**

使用了由GPT-5生成的约400句日语文化与通用句子、以及JCommonsenseQA和JCommonsenseMorality等日语常识问答数据集。

**📈 对比分析**

与直接由LLM输出文化特异性分数的基线比较，CCI在AUC上提升约10个百分点，且在模型性能随CCI变化的分析中更清晰显示文化难度。

**⚠️ 局限性**

局限性包括以国家为单位粗粒度建模，未验证对其他文化的泛化，且CCI依赖LLM的偏差与校准问题。

---

## 212. Online Learning in MDPs with Partially Adversarial Transitions and Losses

**arXiv ID:** 2602.09474 | [PDF](https://arxiv.org/pdf/2602.09474v1)

**作者:** Ofir Schlisselberg `[一作]`, Yishay Mansour `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在有限时隧道MDP中，仅有固定Λ步为对抗转移的环境，提出条件占用率（COM）框架并给出两种算法以及对未知对抗步的降维方法。

**💡 创新点**

创新点包括：①引入COM分离稳定占用率与对抗转移概率；②针对连续对抗块设计子策略条件算法，消除对S^Λ的指数依赖；③提出未知对抗步的K^{2/3}降维方法。

**🔧 技术方法**

使用技术包括：乘法权重更新（OMD）、隐式探索、置信集合构造、对抗MDP理论、概率论与偏差/方差分析。

**📊 数据集**

未使用任何实验数据集，全部为理论分析与证明。

**📈 对比分析**

通过与现有全对抗MDP、随机MDP和破坏模型的上界/下界对比，获得上界 	ilde O(H S^Λ√{K S A^{Λ+1}})（或连续时 	ilde O(H√{K S^3 A^{Λ+1}})）与下界一致，表明仅在Λ维上呈指数。

**⚠️ 局限性**

局限性在于算法仍对S或A^Λ指数，计算复杂度高；未知对抗步的K^{2/3}上界可能非最优；在完全对抗MDP下仍存在指数依赖，尚未实现多项式时间算法。

---

## 213. HLGFA: High-Low Resolution Guided Feature Alignment for Unsupervised Anomaly Detection

**arXiv ID:** 2602.09524 | [PDF](https://arxiv.org/pdf/2602.09524v1)

**作者:** Han Zhou `[一作]` (Innolight Technology Research Institute), Xuezhe Zheng `[通讯]` (Innolight Technology Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出HLGFA框架，利用高低分辨率特征一致性对齐进行无监督工业缺陷检测；

**💡 创新点**

创新点在于将高分辨率特征拆解为结构与细节先导，用结构-细节解耦的对齐模块调节低分辨率特征，并加入噪声感知数据增强，使跨分辨率不一致成为可靠的异常信号；

**🔧 技术方法**

技术包括共享冻结的预训练Wide-ResNet-50骨干，结构‑细节分解、FiLM调制与门控残差校正的对齐模块，余弦对齐损失配合JS、Gram、L1辅助正则化，以及无记忆库的推理策略；

**📊 数据集**

使用MVTec AD数据集（15类物体与纹理）进行训练与评估；

**📈 对比分析**

与RD4AD、AnomalyCLIP、CRAD、NGAL等先进方法对比，HLGFA在图像级AUROC 97.5%、像素级AUROC 97.9%、像素级AP 65 等指标上均取得最高或接近最高成绩，显著优于同类方法；

**⚠️ 局限性**

局限性包括：仅在单模视觉场景验证，跨分辨率对齐对极小尺寸缺陷的检测敏感度有限；依赖预训练Backbone，缺乏对多模态或极端噪声环境的鲁棒性验证。

---

## 214. Earinter: A Closed-Loop System for Eating Pace Regulation with Just-in-Time Intervention Using Commodity Earbuds

**arXiv ID:** 2602.09522 | [PDF](https://arxiv.org/pdf/2602.09522v1)

**作者:** Jun Fang `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**通讯引用:** 5302 | [OpenAlex ID](https://openalex.org/A5057896400)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了一套基于商用耳机的闭环系统 Earinter，用于在日常进餐中实时监测咀嚼-吞咽周期并通过即时音频提示调节进食速度。

**💡 创新点**

创新点在于：①将耳机骨传导传感器双用为咀嚼信号捕捉和音频反馈通道，实现端到端闭环；②基于Dual Systems Theory设计的即时、可适配的提示策略；③将“咀嚼次数/吞咽次数（CPS）”作为行为指标，提供可解释的即时反馈。

**🔧 技术方法**

技术包括：骨传导音频采集、基于EfficientNet-B0的轻量化咀嚼检测网络、基于咀嚼间隔的吞咽推理算法、JIT音频提示生成与冷却策略以及Android端的实时推理与可视化。

**📊 数据集**

使用的数据集为10.07小时的实测数据，包含5.65小时实验室环境和4.42小时户外环境的咀嚼/吞咽事件，采用手工标注并与视频、喉部麦克风同步验证。

**📈 对比分析**

通过与控制（仅预餐提醒）和基线（无提示）条件对比，使用负二项混合模型和线性混合模型评估，结果显示实验条件下CPS平均从15.02提升至26.42（p<0.0001），进食速度降低约10 g/min（p<0.001），且在无提示的保留日仍保持显著提升；提示有效性评估指标包括F1=0.97、CPS估计MAE=0.18 ± 0.13。

**⚠️ 局限性**

局限性包括：研究周期仅13天，尚未检验长期习惯形成；CPS目标未针对不同食物属性自适应；提示策略可能在高社交压力或噪声环境下效果不佳；设备佩戴舒适度虽良好，但部分用户仍感受轻微干扰。

---

## 215. The CLEF-2026 CheckThat! Lab: Advancing Multilingual Fact-Checking

**arXiv ID:** 2602.09516 | [PDF](https://arxiv.org/pdf/2602.09516v1)

**作者:** Julia Maria Struß `[一作]` (University of Applied Sciences Potsdam), Konstantin Todorov `[通讯]` (University of Montpellier)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

CLEF-2026 CheckThat! Lab 设计了三项任务：多语言科学网络主张来源检索、带有推理层次的数值与时间主张核查，以及全篇事实核查文章自动生成；

**💡 创新点**

创新点在于：①引入多语言科学主张检索与 MRR@5 评估；②通过测试时缩放（temperature 多样化）对 LLM 推理路径进行排序提升数值主张核查效果；③首次将事实核查文章写作作为任务，并采用引用完整性与可推断性等参考无关指标评估；

**🔧 技术方法**

主要技术包括：检索式排名模型（BM25/向量检索）、LLM（GPT‑4o‑mini）生成多条推理路径、排序器（如 SVM/神经网络）对推理路径进行再排序、以及 LLM 作为自动写作引擎生成带内联引用的长文本；

**📊 数据集**

使用的数据集有：English/German/French 约 15,699/1,500/1,500 条社交媒体‑论文对；数值/时间主张集 8,000/2,808/3,260 条主张（含 20 条推理路径）；WatClaimCheck 26k 训练/3.3k 验证/1.2k 测试（ExClaim+AmbiguousSnopes）；

**📈 对比分析**

与基线相比，检索任务在 MRR@5 上达 0.45‑0.55（英文）/0.38‑0.48（德法），数值核查任务的宏观 F1 超过 0.70，Recall@5 超过 0.60；文章生成任务的引用完整性与可推断性均高于 0.45，且与人工文章在 Elo 评分上相差约 15 分；

**⚠️ 局限性**

局限性包括：仅覆盖五种语言（对低资源语言缺乏支持）、缺乏真实证据检索环节、LLM 生成文章在逻辑连贯性和事实严谨性上仍低于专家撰写；

---

## 216. Improved Approximate Regret for Decentralized Online Continuous Submodular Maximization via Reductions

**arXiv ID:** 2602.09502 | [PDF](https://arxiv.org/pdf/2602.09502v1)

**作者:** Yuanyu Wan `[一作]` (Zhejiang University), Mingli Song `[通讯]` (Zhejiang University)

**通讯引用:** 9284 | [OpenAlex ID](https://openalex.org/A5026532752)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了通过两种新的归约方法，将去中心化在线连续子模最大化（D-OCSM）问题转化为去中心化在线凸优化（D-OCO）问题，并基于此设计了一系列改进的算法；

**💡 创新点**

核心创新在于：①利用提升（boosting）技术构造线性化目标，将D-OCSM与D-OCO关联；②将分块版Meta‑Frank‑Wolfe扩展到去中心化环境，从而实现对下凸集的（1/e）近似保证；

**🔧 技术方法**

关键技术包括：加速图论中的 gossip 步骤、投影无关的 Frank‑Wolfe/FTPL 变体、分块/阻塞更新机制、以及对梯度误差与共识误差的细致控制；

**📊 数据集**

论文主要是理论分析，未涉及具体数据集实验；

**📈 对比分析**

与先前方法相比，所得到的近似后悔（α‑regret）上界在一般凸集合下与 D‑OCO 最优后悔相匹配，在下凸集合下首次给出 (1/e) 近似且时间复杂度与集中式算法相当，整体性能显著提升；

**⚠️ 局限性**

局限性包括：需假设攻击者为全局不适应（oblivious）且目标为连续 DR‑子模；对非凸决策集及自适应攻击者的情况尚未覆盖；缺乏实验验证。

---

## 217. Beyond Vizing Chains: Improved Recourse in Dynamic Edge Coloring

**arXiv ID:** 2602.09497 | [PDF](https://arxiv.org/pdf/2602.09497v1)

**作者:** Yaniv Sadeh `[一作]` (Tel Aviv University), Haim Kaplan `[通讯]` (Tel Aviv University)

**通讯引用:** 9673 | [OpenAlex ID](https://openalex.org/A5006699796)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了“shift-tree”技术，设计了一系列在动态图中维持（Δ+C）色的贪心与递推算法，并给出了严格的最坏情况复原代价（recourse）上界与下界。

**💡 创新点**

创新点在于将色移位过程视为树结构，突破传统的Fans与双彩路径限制，实现对大色板（C≥0.62Δ）的最优（O(log n / logΔ + C/(Δ-C))）最坏情况复原代价，并在低树枝度图中实现 C≥(2+ε)α 的最优结果。

**🔧 技术方法**

核心技术包括 shift-tree 构造、good/bad 邻居判定、BFS 级别展开、路径可达性分析以及对树深度与色板大小的精细计数与上界证明；此外还利用了组合论与递推式的解析技巧。

**📊 数据集**

论文为理论性工作，未使用具体数据集，仅在理论模型下给出构造与证明。

**📈 对比分析**

相较于已有的 Vizing 链、Nibbling 方法与低树枝度算法，本文在大色板场景下取得了最坏情况的匹配下界，证明了其最优性；在低树枝度图中，仅用 (2+ε)α 色即可达到 O((1/ε)·log n) 的复原代价，显著优于此前需要 (4+ε)α 的结果。

**⚠️ 局限性**

主要限制是需要较大的额外色板（C≥0.62Δ），且对小色板（C=O(1)）的适用性仍不确定；此外低树枝度算法的运行时间为多项式，未达到 poly‑log 的效率。

---

## 218. OSI: One-step Inversion Excels in Extracting Diffusion Watermarks

**arXiv ID:** 2602.09494 | [PDF](https://arxiv.org/pdf/2602.09494v1)

**作者:** Yuwei Chen `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Shiguang Shan `[通讯]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种单步逆向（One‑Step Inversion, OSI）方法，用于高效提取扩散模型生成图像中的 Gaussian Shading 风格水印。

**💡 创新点**

核心创新在于把水印提取视为离散符号分类问题，而非传统的多步连续回归；通过在预训练的扩散 U‑Net 与 VAE 编码器上进行微调，直接预测噪声符号掩码，从而大幅提升速度与提取准确率，并将载荷容量翻倍。

**🔧 技术方法**

技术手段包括：扩散模型（Stable Diffusion、SDXL、SD3.5 等）预训练网络；联合微调 Encoder 与 U‑Net；二元交叉熵 + MSE 损失；数据增强（裁剪、模糊等）；基准评估指标（TPR@FPR=1e-6、比特准确率、FLOPs、payload rate）；对抗攻击测试（压缩攻击、嵌入攻击）。

**📊 数据集**

使用 Stable Diffusion Prompts（SDP）语料库（约 72k 条训练 Prompt，1k 条评估 Prompt）以及 MS‑COCO 语料库进行跨数据集验证。

**📈 对比分析**

与传统 Gaussian Shading（GS）以及其改进版本（PRCW、T2SMark）进行对比，使用相同生成与逆向调度器；实验表明 OSI 在所有调度器、模型、加密方案下均实现：>20 倍 FLOPs 减少、>25 倍运行时加速、比特准确率提升、TPR 上升，且水印容量翻倍，图像质量保持不变。

**⚠️ 局限性**

限制：需要一次额外的微调成本；目前仅针对 Gaussian Shading 风格的水印；对更复杂对抗攻击（如大幅压缩或嵌入攻击）仍有进一步提升空间；未在非扩散模型或其他水印框架中验证，适用范围仍待扩展。

---

## 219. QoS Identifier and Slice Mapping in 5G and Non-Terrestrial Network Interconnected Systems

**arXiv ID:** 2602.09493 | [PDF](https://arxiv.org/pdf/2602.09493v1)

**作者:** Yuma Abe `[一作]` (National Institute of Information and Communications Technology), Amane Miura `[通讯]` (National Institute of Information and Communications Technology)

**通讯引用:** 511 | [OpenAlex ID](https://openalex.org/A5108563768)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了5G-NTN互联系统的端到端流量优化框架，包括5QI→NQI映射、NTN流量到切片映射以及基于切片的流量分配与路由优化。

**💡 创新点**

引入了NTN QoS标识符NQI和NTN切片概念，实现了在NTN环境下的QoS抽象与统一资源控制，并通过多种映射策略评估其对延迟与流量满足度的影响。

**🔧 技术方法**

采用混合整数线性规划（MILP）求解切片级流量分配与路由，结合5QI↔NQI映射函数和基于目的地的切片划分。

**📊 数据集**

基于仿真设置，使用30个gNB、150个UE、90颗LEO卫星、3个光学地面站以及七个5QI/NQI组合的随机流量生成。

**📈 对比分析**

通过六种5QI–NQI映射方案和两组成本函数权重（0.5/0.5 与 0.3/0.7）进行仿真，比较用户平均流量满足率和延迟满足率；结果显示保持PDB顺序且中等数量NQI（如条件4、5）可提升延迟满意度，但计算时间更长。

**⚠️ 局限性**

仅考虑5G UE流量，未涵盖非5G流量；映射与切片策略对计算复杂度影响大，且未加入公平性度量；实际部署中的动态链路可用性与时变延迟未建模。

---

## 220. Beware of the Batch Size: Hyperparameter Bias in Evaluating LoRA

**arXiv ID:** 2602.09492 | [PDF](https://arxiv.org/pdf/2602.09492v1)

**作者:** Sangyoon Lee `[一作]` (Pohang University of Science and Technology), Jaeho Lee `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 15257 | [OpenAlex ID](https://openalex.org/A5005562259)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新评估了LoRA及其主流变体，重点研究批量大小对微调性能的影响；

**💡 创新点**

发现批量大小是LoRA性能的重要决定因素，vanilla LoRA在适当调参后可匹配甚至超越复杂变体，并提出低成本的批量大小调参代理方法；

**🔧 技术方法**

使用LoRA自适应矩阵分解、学习率调优、固定样本协议、warm-up省略等技术；

**📊 数据集**

在MetaMathQA训练集上微调，使用GSM8K作为评估基准；

**📈 对比分析**

在统一实验框架下对比PiSSA、MiLoRA和vanilla LoRA，发现不同批量大小下各方法性能交叉，说明原先的优势多为超参偏差；vanilla LoRA在最佳批量下表现最佳；

**⚠️ 局限性**

局限性包括缺乏理论解释、仅针对数学推理任务、仅验证LLaMA-2系列，未验证跨任务和跨模型的普适性。

---

## 221. The Wisdom of Many Queries: Complexity-Diversity Principle for Dense Retriever Training

**arXiv ID:** 2602.09448 | [PDF](https://arxiv.org/pdf/2602.09448v1)

**作者:** Xincan Feng `[一作]` (Nara Institute of Science and Technology), Yuji Matsumoto `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**通讯引用:** 12402 | [OpenAlex ID](https://openalex.org/A5072032804)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究合成查询对密集检索性能的影响，提出可测量的查询多样性指标并验证复杂性-多样性原则。

**💡 创新点**

首次系统量化查询多样性效益，发现查询复杂度决定多样性是否有利，并给出可操作阈值。

**🔧 技术方法**

使用零-shot多查询生成、Q‑D度量（Dist‑Sim、Len‑Sim、CE、Self‑BLEU）、对比学习训练以及多种检索模型。

**📊 数据集**

在31个数据集上评估，包括MS MARCO、BEIR、BRIGHT和四类多跳问答（NovelHopQA、HotpotQA、MuSiQue、2WikiMultihopQA）。

**📈 对比分析**

与多种 Few‑shot/Zero‑shot 生成基线及无生成方法对比，基于 Contriever/RetroMAE 检索器，零-shot多查询在多跳任务上平均提升约 2‑5% NDCG@10，达到 SOTA。

**⚠️ 局限性**

仅验证两种检索器；复杂度度量仅基于词汇；多样性控制仅靠提示，未探索其他方法；主要关注英文，跨语言泛化未知。

---

## 222. A Scoping Review of Deep Learning for Urban Visual Pollution and Proposal of a Real-Time Monitoring Framework with a Visual Pollution Index

**arXiv ID:** 2602.09446 | [PDF](https://arxiv.org/pdf/2602.09446v1)

**作者:** Mohammad Masudur Rahman `[一作]` (Independent University), M Ashraful Amin `[通讯]` (Independent University)

**通讯引用:** 1825 | [OpenAlex ID](https://openalex.org/A5037386486)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了当前基于深度学习的城市视觉污染检测与分类研究，并提出了面向实时监测的统一框架。

**💡 创新点**

创新点包括：①构建统一视觉污染分类体系与可量化的视觉污染指数；②提出跨城市、跨数据源的综合监测管线；③整合检测、分割、解释与报告生成的端到端流程；④探讨利用大语言模型辅助自动报告与决策。

**🔧 技术方法**

技术手段主要是深度学习对象检测（YOLOv3–v8、Faster R‑CNN、EfficientDet、Swin Transformer）、语义分割、可解释AI、LLM/视觉‑语言模型以及移动/边缘推理框架。

**📊 数据集**

使用了多种公开/自建数据集，包括TACO、UVPD、MOMRAH VP、VP‑Dhaka、UAVBillboards、IAD、pLitterStreet、Open Litter Map、Saudi Public Roads VP Dataset 等共26篇文献所述。

**📈 对比分析**

对模型的比较采用 mAP、FPS、延迟、内存/能耗等指标。YOLOv8 在移动端可实现 95%+ 检测精度、>30 FPS；EfficientDet 在服务器端达到约 97% mAP，但整体仍低于理想基准，且缺乏统一跨城市评测。

**⚠️ 局限性**

局限性：①缺乏统一的污染物分类与标注规范；②缺少覆盖多城市、规模大的基准数据集；③视觉污染指数尚未标准化；④模型对光照、遮挡、不同城市景观的泛化能力有限；⑤实时部署在资源受限设备上的性能瓶颈；⑥缺乏人机交互与政策落地路径。

---

## 223. Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation

**arXiv ID:** 2602.09445 | [PDF](https://arxiv.org/pdf/2602.09445v1)

**作者:** Sunwoo Kim `[一作]` (Korea Advanced Institute of Science and Technology), Kijung Shin `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2485 | [OpenAlex ID](https://openalex.org/A5028609723)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在多模态推荐中实现了用户个性化的参数高效微调（PerPEFT），通过为不同兴趣组分配独立的PEFT模块，让基础模型关注与各组用户兴趣相关的物品属性。

**💡 创新点**

创新点包括：①利用用户兴趣聚类为每组配备专属PEFT模块，提升模型对细粒度兴趣的识别；②提出组特定负采样训练策略，显著强化对各组购买相关特征的学习；③保持参数增量极低（仅1.3%）并兼容任意PEFT方法。

**🔧 技术方法**

技术手段包括：使用CLIP作为多模态基础模型；集成LoRA、(IA)^3、IISAN等PEFT方法；采用SASRec作为序列推荐骨干网络；利用K‑Means进行用户兴趣聚类；对每组独立执行负采样与训练。

**📊 数据集**

数据集：Amazon评论的四个电商领域——Sports & Outdoors、Toys & Games、Beauty & Personal Care、Arts, Crafts & Sewing，分别含数万用户和数万商品，平均序列长度约7‑9。

**📈 对比分析**

与七种基线（含非多模态、冻结多模态、全局PEFT、用户/组级嵌入化个性化）进行对比，PerPEFT在48个评估设置中有44次领先；最高提升为NDCG@20提升15.3%；参数增量仅1.3%，训练时间略高于全局PEFT但仍可接受。

**⚠️ 局限性**

局限性：需要先行训练全局PEFT并提取用户预测嵌入做聚类，过程较为复杂；组数固定为8时效果最佳，过多或过少可能导致信息稀疏或过拟合；对极端稀疏用户或少数组的表现尚未充分验证。

---

## 224. SchröMind: Mitigating Hallucinations in Multimodal Large Language Models via Solving the Schrödinger Bridge Problem

**arXiv ID:** 2602.09528 | [PDF](https://arxiv.org/pdf/2602.09528v1)

**作者:** Ziqiang Shi `[一作]` (Fujitsu Research and Development Center Co., LTD), Koichi Shirahata `[通讯]` (Fujitsu Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对多模态大型语言模型的幻觉问题，提出了一种轻量级推理时框架 SchröMind，利用 Schrödinger 桥（最优传输）在 token 级别实现幻觉激活向真实激活的精细映射，从而抑制错误生成。

**💡 创新点**

创新点在于：①将幻觉与真实注意力分布建模为两种几何流，采用 Schrödinger 桥求解最短迁移路径；②实现 token‑级个性化干预，避免统一方向导致的全局失真；③只需几分钟的轻量训练即可完成，保持原模型能力且易于部署。

**🔧 技术方法**

核心技术包括：Schrödinger 桥（与熵正则化最优传输等价）、动态漂移函数 g*(·,t)、激活工程与对齐、逻辑回归头部选择器、以及对注意力激活的插值插入。

**📊 数据集**

实验使用了 POPE（MS COCO、A‑OKVQA、GQA）和 MME（14 维多任务）评测数据集，模型基于 LLaVA‑1.5‑7B 和 Qwen2.5‑VL‑7B 两大多模态 LLM。

**📈 对比分析**

与常规模型、ICT、VCD、OPERA 等方法对比，SchröMind 在 POPE 上在随机/热门/对抗场景中均实现了 3–6% 的 Accuracy/F1 提升，尤其在对抗问题上突破 80% F1；在 MME 上总分提升 7–10 分，显著超越前沿方案。

**⚠️ 局限性**

局限性包括：①需预先选定敏感注意力头部，依赖额外的头部筛选超参；②在非视觉输入或极端噪声场景下的鲁棒性尚未充分验证；③虽然开销低，但动态漂移插值仍带来一定的推理延迟；④方法主要聚焦视觉‑语言对齐，未对纯语言幻觉展开评估。

---

## 225. Singpath-VL Technical Report

**arXiv ID:** 2602.09523 | [PDF](https://arxiv.org/pdf/2602.09523v1)

**作者:** Zhen Qiu `[一作]` (LBP Singpath AI Lab), Hao Zhang `[通讯]` (LBP Singpath AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个针对宫颈细胞学的专用视觉语言大模型。

**💡 创新点**

提出三阶段数据生成管线与三阶段微调策略，解决细胞级细粒度描述与诊断。

**🔧 技术方法**

采用Qwen3-VL-4B基础模型，结合多模型弱标注、共识融合、专家注入以及知识重放训练。

**📊 数据集**

构建百万级的Singpath-CytoText图像-文本数据集以及细粒度感知和TBS分类基准。

**📈 对比分析**

与多种通用MLLM对照，零样本下精确度明显优于基线，达至与专家一致的细胞特征识别和诊断准确率。

**⚠️ 局限性**

仅适用于宫颈细胞学，缺乏因果推理、全流程临床集成以及跨细胞学领域的泛化能力。

---

## 226. Jokeasy: Exploring Human-AI Collaboration in Thematic Joke Generation

**arXiv ID:** 2602.09496 | [PDF](https://arxiv.org/pdf/2602.09496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 227. Energy-Efficient Fast Object Detection on Edge Devices for IoT Systems

**arXiv ID:** 2602.09515 | [PDF](https://arxiv.org/pdf/2602.09515v1)

**作者:** Mas Nurul Achmadiah `[一作]` (National Formosa University), Wen-Kai Kuo `[通讯]` (National Formosa University)

**通讯引用:** 698 | [OpenAlex ID](https://openalex.org/A5102949205)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

结合帧差法与轻量级 AI 分类器，在 IoT 边缘设备上实现快速移动物体检测与分类。

**💡 创新点**

创新点在于将传统帧差运动检测与深度学习分类器融合，形成轻量级、能耗低、延迟短的检测框架，显著提升了移动对象的识别精度与实时性。

**🔧 技术方法**

技术包括帧差 + 图像形态学处理、阈值分割、ROI 采样、图像裁剪与双线性重采样；使用 MobileNet、Inception‑v4、ResNet‑50、ViT‑Base 四种 CNN/Transformer 模型；部署于 AMD Alveo U50、Jetson Orin Nano 与 Hailo‑8 AI 加速器。

**📊 数据集**

采用 ImageNet 数据集（用于训练与评估分类模型）以及 MSCOCO 数据集（用于 YOLOX 对比实验）。

**📈 对比分析**

通过与 YOLOX end‑to‑end 方法对比，测量准确率、延迟(ms)、能耗(J)、效率(%/msW)。实验结果表明：平均准确率提升 28.3%，延迟降低 39.3%，能效提升 3.6 倍；MobileNet 在三种设备上表现最佳，整体实现了高准确、低延迟、低能耗的目标。

**⚠️ 局限性**

局限性：对动态背景、光照变化和摄像机震动敏感，易产生误检；对小或慢速/高运动模糊对象检测效果不佳；帧差法在噪声环境下鲁棒性不足。

---

## 228. EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies

**arXiv ID:** 2602.09514 | [PDF](https://arxiv.org/pdf/2602.09514v1)

**作者:** Xavier Hu `[一作]` (OPPO AI Agent Team), Wangchunshu Zhou `[通讯]` (OPPO AI Agent Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一个名为 ECOGYM 的无限时延交互经济模拟环境，用以评估大型语言模型在长周期内的规划与执行能力。

**💡 创新点**

创新点在于构建了统一的多场景（Vending、Freelance、Operation）无限时延经济测试框架，并将评估指标从传统奖励转为可观的经济收益。

**🔧 技术方法**

采用了基于 LLM 的智能体框架，结合情景上下文、记忆模块和行动空间，通过自动化脚本在模拟环境中进行长周期决策实验。

**📊 数据集**

使用的数据集为 ECOGYM 的三种环境模拟数据，覆盖每日交易、收入和活跃用户等关键指标，所有实验可在公开平台复现。

**📈 对比分析**

与基准模型和人类基线进行对比，发现当前主流 LLM 在任一场景均未能始终保持最高收益，整体性能在长期决策上相对短期任务表现明显不足。

**⚠️ 局限性**

局限性包括缺乏真正多主体交互的复杂性、模拟环境仍较为简化以及对模型记忆容量的依赖导致部分实验结果受限。

---

## 229. Beyond Student: An Asymmetric Network for Neural Network Inheritance

**arXiv ID:** 2602.09509 | [PDF](https://arxiv.org/pdf/2602.09509v1)

**作者:** Yiyun Zhou `[一作]` (Zhejiang University), Jingyuan Chen `[通讯]` (Zhejiang University)

**通讯引用:** 3463 | [OpenAlex ID](https://openalex.org/A5090689233)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种直接继承教师网络结构和知识的压缩方法——InherNet，利用低秩分解和SVD初始化构建轻量但表达力强的网络。

**💡 创新点**

创新点在于：①不通过隐式蒸馏，而是以教师权重为基础进行异步低秩分解；②结合MoE式的专家结构和自适应门控，兼顾深度、宽度与压缩；③SVD初始化显著提升收敛速度和稳定性。

**🔧 技术方法**

技术包括：矩阵奇异值分解（SVD）初始化、低秩分解、Mixture-of-Experts（MoE）门控、基于梯度的理论收敛与参数效率分析。

**📊 数据集**

在CIFAR‑100、ImageNet、GLUE（T5）、LLaMA‑2‑7B、Conceptual Captions 3M（CC3M）以及ImageNet零样本分类等多模态数据集上进行实验。

**📈 对比分析**

与传统KD方法（如KD、CTKD、MLKD、Logit Std.等）对比，InherNet在保持相同或更少参数的前提下实现了更快的收敛和更高的准确率；在CIFAR‑100上可匹敌或超越Logit Std.，在ImageNet、GLUE、CC3M等任务亦表现出显著优势。

**⚠️ 局限性**

局限性包括：①对高秩教师网络的继承效果仍受限于SVD近似误差；②在极小模型规模下，门控和专家数目需要精细调参；③缺乏对极大规模Transformer的广泛验证，且在某些任务中仍未完全达到教师性能。

---

## 230. Listen to the Layers: Mitigating Hallucinations with Inter-Layer Disagreement

**arXiv ID:** 2602.09486 | [PDF](https://arxiv.org/pdf/2602.09486v1)

**作者:** Koduvayur Subbalakshmi `[一作]` (Stevens Institute of Technology), Nastaran Jamalipour Soofi `[通讯]` (Stevens Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种无训练的推理时解码器CoCoA，利用中间层表征不稳定性来抑制LLM幻觉。

**💡 创新点**

创新点在于把中间层的表示一致性作为内部不确定性信号，并在解码时加入自信息门控的惩罚。

**🔧 技术方法**

使用自回归解码、层间余弦距离度量、正则化惩罚和自信息门控等技术。

**📊 数据集**

在TruthfulQA、NQ/NQ‑Swap、SAMSum、XSum、MBPP等多任务数据集上进行评估。

**📈 对比分析**

与贪婪、DoLa、DeCoRe、Diver等基线相比，CoCoA在多模型（Llama‑3、Mistral、Qwen‑2.5等）上显著提升真确率、T×I和代码通过率，效果稳定。

**⚠️ 局限性**

局限包括需要调参（α、层范围）、对内部状态的依赖、在极端置信词上可能过度惩罚，且在极大模型或不同架构上的通用性待验证。

---

## 231. AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms

**arXiv ID:** 2602.09464 | [PDF](https://arxiv.org/pdf/2602.09464v1)

**作者:** Haoyu Zhao `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**通讯引用:** 78329 | [OpenAlex ID](https://openalex.org/A5027798962)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个跨 Dafny、Verus、Lean 对齐的验证代码生成基准（AlgoVeri），评估 LLM 在生成形式验证的经典算法代码时的表现。

**💡 创新点**

首次将经典算法的全局推理任务在三种验证系统中语义对齐，消除工具链差异，并通过多轮修复和语义过滤揭示模型的自我纠错与语言障碍。

**🔧 技术方法**

使用大语言模型（如 Gemini‑3 Flash、GPT‑OSS‑120B 等）结合编译器/验证器反馈的多轮生成‑修复流程，辅以语义验证器与专家校准的评估管道。

**📊 数据集**

包含 77 道经典算法（堆、线段树、红黑树、Bellman‑Ford、最大流等）在 Dafny、Verus、Lean 上的对齐规范，数据与代码公开于 GitHub。

**📈 对比分析**

在三种验证器上进行多轮修复实验，衡量编译器验证率与语义过滤后的完整成功率；结果显示在 Dafny 上约 40%（最高），Verus 约 25%，Lean 仅约 8%；前沿模型在修复深度上优于开源模型。

**⚠️ 局限性**

依赖人工编写与对齐规范，可能产生偏差；对全局性质的验证仍然低效，开源模型缺乏深度修复能力；验证依赖特定编译器/验证器，泛化性受限。

---

## 232. Enhancing Affine Maximizer Auctions with Correlation-Aware Payment

**arXiv ID:** 2602.09455 | [PDF](https://arxiv.org/pdf/2602.09455v1)

**作者:** Haoran Sun `[一作]` (Peking University), Xiaotie Deng `[通讯]` (Peking University)

**通讯引用:** 10822 | [OpenAlex ID](https://openalex.org/A5100638710)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了关联感知的仿射最大化拍卖（CA-AMA）并给出两阶段训练方法，解决传统AMA在竞标者估价相关时的收益瓶颈。

**💡 创新点**

在传统AMA的VCG式支付规则中加入仅依赖其他竞标者估价的关联支付项，保持DSIC同时显著提升在相关估价分布下的收益。

**🔧 技术方法**

采用神经网络对AMA参数和关联支付进行参数化，使用软最大化近似实现可微分分配；通过双阶段训练（共识训练+后处理）结合IR惩罚，并给出连续性与泛化误差理论分析。

**📊 数据集**

主要使用合成相关估价分布，包括Dirichlet Value Share、Linear Correlation Mixture、以及完全相关的均值分布等。

**📈 对比分析**

与随机化AMA（AMenuNet）、Item-CAN、VCG等基线进行收益比较；在所有测试场景中CA-AMA收益均高，特别是在强相关环境下收益提升显著，IR违背被控制在约0.001。

**⚠️ 局限性**

对多项拍卖的最优理论仍未给出，仅通过经验验证；训练依赖软最大化近似，可能导致梯度不精确；需要进一步研究更复杂的关联结构及真实数据场景。

---

## 233. NOWJ @BioCreative IX ToxHabits: An Ensemble Deep Learning Approach for Detecting Substance Use and Contextual Information in Clinical Texts

**arXiv ID:** 2602.09469 | [PDF](https://arxiv.org/pdf/2602.09469v1)

**作者:** Huu-Huy-Hoang Tran `[一作]` (University of Engineering and Technology), Hoang-Quynh Le `[通讯]` (University of Engineering and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种多输出集成系统，用于同时检测西班牙语临床文本中的触发器和语境信息；

**💡 创新点**

创新点包括：①将句子过滤、共享BETO编码器与多任务CRF解码器结合的多输出框架；②多样化训练策略（加权损失、过采样、加权采样）生成的模型集成；③通过多数投票提升鲁棒性和精确度；

**🔧 技术方法**

使用技术包括：BETO（BERT基础模型）、条件随机场（CRF）序列标注、句子二分类过滤器、模型集成与投票；

**📊 数据集**

使用数据集：ToxHabits（1499份西班牙语临床病例报告），包含触发器（Tobacco、Cannabis、Alcohol、Drug）与语境参数（Type、Method、Amount、Frequency、Duration、History）；

**📈 对比分析**

与官方基线对比，最佳运行在触发器检测上取得F1 0.94、精确度 0.97；在语境信息检测上取得F1 0.91、精确度 0.95，显示相较单一模型和未集成方法的显著提升；

**⚠️ 局限性**

局限性：未使用大型语言模型，可能受限于模型规模与上下文理解；数据量有限导致过拟合风险；对不同语言和医学领域的迁移性能未评估。

---

## 234. Rashomon Sets and Model Multiplicity in Federated Learning

**arXiv ID:** 2602.09520 | [PDF](https://arxiv.org/pdf/2602.09520v1)

**作者:** Xenia Heilmann `[一作]` (Johannes Gutenberg University), Mattia Cerrato `[通讯]` (Johannes Gutenberg University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5042265681)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了在联邦学习（FL）环境下定义Rashomon集及其多重性指标的理论框架，并实现了对应的实验流水线；

**💡 创新点**

创新点在于：①给出了三种FL特定的Rashomon集定义（全局、t-一致、个体）；②在保证隐私的前提下将已有的预测多重性指标迁移到FL；③构建了可扩展的多重性感知FL管道；

**🔧 技术方法**

技术主要包括联邦平均（FedAvg）与FedSGD、差分隐私、分桶直方图、加权聚合与安全聚合、基于候选模型重训练生成Rashomon集；

**📊 数据集**

实验使用了三大FL基准数据集：Dutch Census、ACS Income（两者用于公平性评估）以及MNIST；

**📈 对比分析**

通过比较全局、t-一致、个体Rashomon集的多重性指标以及与集中式基线的差异，结果表明：全局集多样性最高、t-一致集更小、个体集最能反映本地差异；在公平性示例中，Rashomon集允许客户端在保持全局精度的前提下选择满足当地公平性约束的模型；

**⚠️ 局限性**

主要局限在于重训练生成候选模型的计算开销大；未来可探索利用FL内部更新信息或更高效的候选生成策略；

---

## 235. Knowledge Integration Decay in Search-Augmented Reasoning of Large Language Models

**arXiv ID:** 2602.09517 | [PDF](https://arxiv.org/pdf/2602.09517v1)

**作者:** Sangwon Yu `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**通讯引用:** 12623 | [OpenAlex ID](https://openalex.org/A5086877012)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在搜索增强推理过程中出现的知识整合衰减(KID)问题，并提出了一种推理时自锚定知识编码(SAKE)方法。

**💡 创新点**

创新点在于通过在推理链前后分别插入检索结果，形成双重知识定位，既保持知识语义完整，又减少先前推理对新知识编码的干扰，显著缓解KID。

**🔧 技术方法**

主要技术包括自锚定知识编码策略、注意力分布分析和推理-检索交互机制，全部在推理阶段实现，无需额外训练。

**📊 数据集**

实验使用多跳问答与复杂推理基准：HotpotQA、2WikiMultiHopQA、MuSiQue、FRAMES 和 GAIA。

**📈 对比分析**

在与标准 RAG 与 Search‑o1 基线对比时，SAKE 在多跳问答上最高提升 37.6% F1，且在 FRAMES、GAIA 等复杂任务中持续优于基线。

**⚠️ 局限性**

局限性包括显著增加的上下文长度导致推理成本上升，且仅在推理时处理 KID，未改进模型训练或检索质量。

---

## 236. Robust Depth Super-Resolution via Adaptive Diffusion Sampling

**arXiv ID:** 2602.09510 | [PDF](https://arxiv.org/pdf/2602.09510v1)

**作者:** Kun Wang `[一作]` (Singapore University of Technology and Design), Na Zhao `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 8442 | [OpenAlex ID](https://openalex.org/A5040897632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AdaDS框架，针对任意降质低分辨率深度图进行零样本、鲁棒的高分辨率深度重建。

**💡 创新点**

利用扩散模型的高斯平滑收敛特性，设计两阶段流程：先做不确定性感知的校准估计，再根据不确定性自适应选择时间步并注入噪声，将中间结果定位到目标后验高概率区域，从而实现对未知降质的天然鲁棒性。

**🔧 技术方法**

采用预训练的Marigold‑LCM扩散模型作为去噪器，结合ViT‑S的Depth Anything V2骨干、轻量UNet噪声采样网络以及NLL+L1等损失实现两阶段训练。

**📊 数据集**

在Hypersim合成数据上训练，评估时使用真实深度传感器数据集RGB‑D‑D、TOFDSR以及合成数据集NYUv2、ScanNet，并在多种降质（下采样、高斯噪声、模糊、稀疏、低位压缩）下进行测试。

**📈 对比分析**

与现有DSR方法（Depth Anything V2、Marigold‑LCM、SGNet、C2PD等）对比，AdaDS在所有评价指标（RMSE/MAE/δ1.05）均显著优于对手，尤其在非整数放大倍数和极端降质下保持稳定，零样本泛化性能最高。

**⚠️ 局限性**

局限性：依赖高质量预训练扩散模型，推理时仍需一定的计算开销；在极端噪声或缺失比例极高的情况下，校准阶段的估计误差可能导致后续噪声注入不够精准；缺少对不同传感器噪声谱的专门适配。

---

## 237. Towards Uniformity and Alignment for Multimodal Representation Learning

**arXiv ID:** 2602.09507 | [PDF](https://arxiv.org/pdf/2602.09507v1)

**作者:** Wenzhe Yin `[一作]` (University of Amsterdam), Efstratios Gavves `[通讯]` (University of Amsterdam)

**通讯引用:** 8802 | [OpenAlex ID](https://openalex.org/A5002625178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种冲突无关的多模态表征学习框架 UniAlign，分离统一性与对齐，解决 InfoNCE 在多模态（M≥3）中出现的分布桥梁和同义冲突。

**💡 创新点**

创新点包括：①理论分析两类冲突（对齐-统一性冲突和内在对齐冲突）随模态数增大而加剧；②提出基于锚点的对齐策略与每模态内部统一性损失，彻底消除这两类冲突；③引入全局 Hölder 散度，证明该损失是其有效近似，从而在理论上保证跨模态分布差距缩小。

**🔧 技术方法**

使用的技术主要是：多模态 InfoNCE 的分解、锚点对齐损失、每模态内部统一性损失（基于高斯核的负样本拉远），可选的球面几何实现；理论上利用 Hölder 散度与核密度估计给出损失近似；实验中采用跨模态检索与 UnCLIP 生成评估。

**📊 数据集**

主要数据集：VAST150K（训练），MSR‑VTT、DiDeMo、ActivityNet（视频检索评估），VGGSound（音视频文本三模态，交叉模态生成评估）。

**📈 对比分析**

与基线（VAST、GRAM、ImageBind 等）对比：在 T2V/V2T 零样本检索上提升约 8.7/6.4 R@1，跨模态生成（T2I、A2I）FID 下降 10–40，显著小于 InfoNCE 基线；同时在多模态插值生成上亦实现更低 FID，表明跨模态分布更紧密。

**⚠️ 局限性**

局限性：目前仅在细调阶段验证，未在大规模预训练场景下评估；同时需要更多算力以推广至更大规模数据集。

---

## 238. Adaptive recurrent flow map operator learning for reaction diffusion dynamics

**arXiv ID:** 2602.09487 | [PDF](https://arxiv.org/pdf/2602.09487v1)

**作者:** Huseyin Tunc `[一作]` `[通讯]` (Bahcesehir University), Huseyin Tunc (Bahcesehir University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种自适应递归流图算子学习框架（DDOL-ART），用于预测二维反应-扩散系统的长期动力学。

**💡 创新点**

创新之处在于将递归训练与轻量级验证里程碑相结合，利用反馈控制在训练期间早期退出无效展开段，从而显著提升一阶算子在长期循环中的稳定性和零样本 OOD 鲁棒性，同时大幅降低训练成本。

**🔧 技术方法**

技术上采用了基于残差卷积网络的流图算子、SSP-RK3 以及 FD2 空间离散，结合无教师强制的自由跑递归训练，并加入验证误差门限实现自适应调度。

**📊 数据集**

数据集为三类典型二维反应-扩散 PDE（FitzHugh–Nagumo、Gray–Scott、Lambda–Omega），训练初始条件取单一环形高斯分布，OOV 通过多高斯、噪声、条纹、环形等结构化形态生成。

**📈 对比分析**

与物理约束残差学习（NLOL）和纯数据驱动递归训练（DDOL）在相同评估协议下对比，DDOL-ART 在 ID 和多种 OOD 场景下均实现了最低的平均最大绝对误差，并在训练时间上比 NLOL 快 3.2–3.6 倍、比 DDOL 快约 1.8–2.2 倍。

**⚠️ 局限性**

局限性在于仅在完全观测的二维周期域反应-扩散系统上验证，未考察稀疏测量、非周期边界、其他 PDE 类别或无显式方程的通用时空序列。

---

## 239. Bridging Efficiency and Transparency: Explainable CoT Compression in Multimodal Large Reasoning Models

**arXiv ID:** 2602.09485 | [PDF](https://arxiv.org/pdf/2602.09485v1)

**作者:** Yizhi Wang `[一作]` (Southeast University), Min-Ling Zhang `[通讯]` (Southeast University)

**通讯引用:** 15413 | [OpenAlex ID](https://openalex.org/A5079083101)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种可解释的多模态思考链压缩框架 XMCC，旨在通过压缩长链（Long CoT）来提升推理效率，同时保持关键信息并生成自然语言解释。

**💡 创新点**

创新点包括：①将压缩过程视为序列决策问题并使用强化学习（GRPO）进行端到端优化；②设计四种奖励（格式、结果、步骤关键性、长度）以保证压缩后链的逻辑完整性、可解释性和自适应长度；③在压缩过程中显式保留视觉-文本对齐信息，并在压缩决策前生成解释文本。

**🔧 技术方法**

技术方法主要包括：多模态 LLM（如 Qwen3-VL-2B）作为压缩器、GRPO 强化学习框架、四元奖励函数、可解释生成模板、长度自适应惩罚、SFT 训练以获取高效推理模型。

**📊 数据集**

使用的数据集：自构造的 9k 样本 XMCC‑Dataset（包含 Geo170k 与 ScienceQA 的图文问答以及多模态 CoT），以及公开的 MathVista、WeMath、MMStar、MMMU、R1‑Onevision‑Bench 等基准数据。

**📈 对比分析**

与两种文本压缩基线（Prune‑on‑Logic、StepEntropy）及无压缩 baseline 进行对比。实验表明，在所有基准上，XMCC 将平均 CoT 长度从 500+ token 缩减至约 90‑110 token，压缩比例大幅提升，同时保持或略微提升准确率（如 R1 Physics 任务提升 2‑3%，其他任务提升 0.5‑1%）。

**⚠️ 局限性**

局限性包括：①仍需强大多模态 LLM 支持，计算成本高；②奖励设计和 RL 训练复杂，易受超参数影响；③在极端复杂或需要多步推理的任务中，过度压缩可能导致信息丢失；④解释质量依赖评判 LLM 的能力，缺乏客观评估；⑤未在安全敏感或高可靠性场景中验证鲁棒性。

---

## 240. Beyond Next-Token Alignment: Distilling Multimodal Large Language Models via Token Interactions

**arXiv ID:** 2602.09483 | [PDF](https://arxiv.org/pdf/2602.09483v1)

**作者:** Lin Chen `[一作]` (Ant Group), Shiming Xiang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 14131 | [OpenAlex ID](https://openalex.org/A5040673285)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 token 交互的知识蒸馏框架 Align‑TI，用于把大型多模态语言模型压缩成参数高效模型。

**💡 创新点**

创新点在于引入 Instruction‑aware Vision Alignment (IVA) 对视觉‑指令交互进行加权对齐，以及 Transition Probability Alignment (TPA) 对 token‑to‑token 转移概率进行对齐，从而刻画了视觉指令交互和生成过程交互的两种关键 token 交互。

**🔧 技术方法**

使用注意力权重加权的视觉对齐、KL 散度的 token 转移概率对齐、蒙特卡洛采样与并行计算的高效实现，整体基于 Transformer 与 LLM 架构。

**📊 数据集**

在 1.2M 图像-字幕、2.4M 混合字幕与 VQA 样本的数据集上进行 SFT 与 KD，评测使用 GQA、SQA、TextVQA、POPE、MME、MMB 等多模态基准。

**📈 对比分析**

与 LLaVA、MobileVLM、MiniCPM 等基准模型以及传统 Vanilla KD、GKD 等方法对比，1B/2B 参数模型在多模态评测平均得分提升约 2–7%，尤其 2B 模型超过 7B LLaVA 约 7%。

**⚠️ 局限性**

受限于教师规模的递减收益和学生容量的限制，进一步扩展性能受限；实现中仍存在相对较高的训练时间与显存开销。

---

## 241. Weakly Supervised Contrastive Learning for Histopathology Patch Embeddings

**arXiv ID:** 2602.09477 | [PDF](https://arxiv.org/pdf/2602.09477v1)

**作者:** Bodong Zhang `[一作]` (University of Utah), Tolga Tasdizen `[通讯]` (University of Utah)

**通讯引用:** 6594 | [OpenAlex ID](https://openalex.org/A5059125158)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在多实例学习框架下，提出弱监督对比学习方法 WeakSupCon，用袋级标签学习特征表示，提升 WSIs 的分类性能。

**💡 创新点**

创新点是将负袋特征聚类、正袋采用 SimCLR，并通过袋级标签实现无实例标签的对比学习，显著分离正负实例，提升特征可分性。

**🔧 技术方法**

使用对比学习（SimCLR、MoCo）、监督对比学习（SupCon）以及多任务损失融合的 WeakSupCon；编码器为 ResNet‑18，配合投影头。

**📊 数据集**

实验在 Camelyon16、RVT（肾静脉血栓）和肾转移三大病理 WSI 数据集上进行。

**📈 对比分析**

与 MoCo、SimCLR、SupCon 以及 Prov‑GigaPath、UNI2‑h 等基础模型比较，WeakSupCon 在 Balanced Accuracy、Accuracy、AUC 上均优于对照组，提升幅度约 1.5%–7%，并在两大数据集上优于现有基础模型。

**⚠️ 局限性**

局限性包括：仍需袋级标签，且权重调参对性能敏感；未在更大分辨率或其他医学影像域验证迁移性；实验仅在 ResNet‑18 上完成，模型规模受限。

---

## 242. AlignTune: Modular Toolkit for Post-Training Alignment of Large Language Models

**arXiv ID:** 2602.09621 | [PDF](https://arxiv.org/pdf/2602.09621v1)

**作者:** R E Zera Marveen Lyngkhoi `[一作]`, Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为 AlignTune 的统一后训练对齐工具包，整合了 TRL 与 Unsloth 两大后端，实现了 SFT、RLHF（DPO、PPO、GRPO 等）以及奖励模型训练、评估等完整管道，提供统一配置、CLI、数据管理和诊断功能。

**💡 创新点**

创新点在于：① 对齐工具的多后端抽象与隔离机制，使不同后端可在同一环境下互不干扰；② 将奖励函数与奖励模型提升为第一类对象，支持可组合、可注册的奖励体系；③ 统一的强类型配置和 CLI，降低实验复现门槛；④ 在同一平台上实现多算法、多后端的对比实验，证明后端差异不会导致模型质量波动。

**🔧 技术方法**

技术栈包括：Python、Hugging Face Transformers、TRX、Unsloth（量化与内核加速）、LoRA/QLoRA、PPO/DPO/GRPO/其它 RLHF 算法、lm-eval、PEFT、bitsandbytes、Accelerate、PyTorch 及自定义奖励、评估框架与沙箱执行。

**📊 数据集**

使用的数据集覆盖多任务：Alpaca（SFT）、HH-RLHF（偏好优化）、GSM8K（数学推理）、MBPP（代码生成）、Bitext Wealth Management、Bitext Retail Banking、以及公共 benchmark 如 HellaSwag、ARC、MMLU 等；同时支持 Hugging Face Hub、CSV、Parquet 等多种数据来源。

**📈 对比分析**

通过在相同配置下对比 TRL 与 Unsloth 两后端的 DPO 与 GRPO 训练，实验显示：Unsloth 在吞吐量和显存使用上优于 TRL（约 1.3×速度提升），但最终评估指标（奖励边际、准确率等）与 TRL 接近，说明后端切换不会对对齐质量产生显著影响；此外，工具可无代码改动完成跨后端实验，极大提升实验可重复性。

**⚠️ 局限性**

局限性包括：部分高级 RLHF 算法仅支持 TRL，Unsloth 需要特定 GPU/CUDA 版本，缺乏对所有模型族的完整兼容；奖励函数仍可能携带偏见且需人工审核；工具尚在早期阶段，部分功能（如安全对齐、机制解释）待完善；在极大规模模型或复杂任务下的性能与稳定性尚未完全验证。

---

## 243. With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots

**arXiv ID:** 2602.09616 | [PDF](https://arxiv.org/pdf/2602.09616v1)

**作者:** Zeinab Sadat Taghavi `[一作]` (Lucerne University of Applied Sciences and Arts), Andreas Marfurt `[通讯]` (Lucerne University of Applied Sciences and Arts)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5061833911)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解决RAG系统中的检索盲点问题，通过预索引审计预测实体检索风险并补齐知识库上下文来提升检索质量。

**💡 创新点**

提出实体检索概率分数（RPS）评估盲点并用嵌入几何预测，提出ARGUS管线实现无训练的盲点检测与补齐。

**🔧 技术方法**

使用神经检索器（如BGE、Contriever等）、线性/树/MLP预测探针、知识库检索、LLM生成与文档扩展等技术。

**📊 数据集**

基于Wikidata–Wikipedia对齐的大规模实体集以及BRIGHT、ImpliRet、RAR-b三大检索基准。

**📈 对比分析**

与原始索引比较，ARGUS在所有检索器上均提升nDCG@5/10，平均+3.4/6.8/1.7分；文档扩展效果最稳，LLM合成更节省空间。

**⚠️ 局限性**

仅使用单一阈值与通用KB，未针对不同任务或检索器微调，盲点阈值与增补形式可能需自适应。

---

## 244. Hand2World: Autoregressive Egocentric Interaction Generation via Free-Space Hand Gestures

**arXiv ID:** 2602.09600 | [PDF](https://arxiv.org/pdf/2602.09600v1)

**作者:** Yuxi Wang `[一作]` (Nanyang Technological University), Xingang Pan `[通讯]` (Nanyang Technological University)

**通讯引用:** 3771 | [OpenAlex ID](https://openalex.org/A5052549072)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Hand2World，能够基于单张场景图像和自由空间手势生成逼真、可控的前视交互视频，支持任意长度的自回归生成。

**💡 创新点**

创新点包括：① 采用投影的 3D 手掌网格（silhouette + wireframe）实现遮挡不敏感的手势条件，消除 mask 失配导致的深度错误；② 通过 Per-pixel Plücker‑ray 嵌入显式建模相机运动，分离手势与视角变化，消除背景漂移；③ 设计完整的单目自动标注管线和自回归蒸馏方法，使模型可在真实环境中实现在线、连续交互。

**🔧 技术方法**

核心技术：MANO 3D 手模型重建、投影渲染、Plücker‑ray 视角编码、基于 Wan2.1‑1.3B‑Control 的视频扩散 Transformer、LoRA 微调、CausVid + self‑forcing 的自回归蒸馏，以及多尺度 VAE 编码。

**📊 数据集**

主要使用 ARCTIC 数据集进行训练与评估，辅以 HOT3D、HOI4D 进行跨数据集验证；对比的基线包括 CosHand、InterDyn、Mask2IV、Wan2.1‑1.3B‑Control 等公开方法。

**📈 对比分析**

实验显示 Hand2World 在 ARCTIC 上 FVD 从 908 降至 219（约 76% 降幅），Cam‑ERR 由 0.13 降至 0.07，DINO 相似度提升至 0.88，整体视觉质量和 3D 一致性明显优于所有基线，并在自回归模式下保持可观的帧率（≈8.9 FPS）与长周期稳定性。

**⚠️ 局限性**

局限性在于自由空间手势缺乏物理接触约束，用户指定的不可行动作（如手穿过固体）可能导致不自然或不物理可行的交互；未来需引入力反馈或碰撞约束以提升逼真度。

---

## 245. Why the Counterintuitive Phenomenon of Likelihood Rarely Appears in Tabular Anomaly Detection with Deep Generative Models?

**arXiv ID:** 2602.09593 | [PDF](https://arxiv.org/pdf/2602.09593v1)

**作者:** Donghwan Kim `[一作]` (Yonsei University), Hyunsoo Yoon `[通讯]` (Yonsei University)

**通讯引用:** 1248 | [OpenAlex ID](https://openalex.org/A5076379562)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在表格数据中使用正则化流（normalizing flow）的似然测试是否会出现图像中已知的“反直觉”现象，并提供了一个通用定义和理论分析。通过在47个表格数据集和10个CV/NLP嵌入数据集上进行大规模实验，验证该现象在表格数据中几乎不存在。

**💡 创新点**

创新点在于：①提出了一个与域无关的“反直觉”现象定义；②从维度和特征相关性两方面理论分析导致该现象的机制；③通过全面实验表明，表格数据中使用简单似然测试的正则化流模型几乎不受该现象影响，并能显著优于传统基线。

**🔧 技术方法**

主要技术包括：正则化流（NICE、RealNVP 等）实现似然测试；对比浅层（PCA、LOF、IF、OCSVM、COPOD、ECOD）与深层（DAGMM、DeepSVDD、GOAD、NeuTraLAD、ICL、MCM、DRL、NF‑SLT）异常检测模型；利用AUROC/AUPRC评估；理论推导涉及KL、熵、内在维度等。

**📊 数据集**

使用了 ADBench 公开的 47 个表格数据集和 10 个 CV/NLP 嵌入数据集（如 CIFAR‑10, SVHN 等）进行实验，避免了数据集选择偏差。

**📈 对比分析**

与 13 个基线模型对比，NF‑SLT 在所有表格数据集上平均 AUROC 最高（0.8575），top‑2 率为 40%，失败率仅 6%。在 CV/NLP 嵌入数据集上，除 imdb 外均保持优越性能；整体上 NF‑SLT 在大多数数据集上名列前茅。

**⚠️ 局限性**

局限性包括：实验仅覆盖低维/相关性相对较弱的表格数据；对极高维或高度相关的表格数据的泛化尚未验证；仅评估了似然测试，未探究其他异常检测方法与正则化流结合的潜力。

---

## 246. Preference Aligned Visuomotor Diffusion Policies for Deformable Object Manipulation

**arXiv ID:** 2602.09583 | [PDF](https://arxiv.org/pdf/2602.09583v1)

**作者:** Marco Moletta `[一作]` (KTH Royal Institute of Technology), Danica Kragic `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 15262 | [OpenAlex ID](https://openalex.org/A5023792180)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究如何将预训练的视觉运动扩散模型对齐到用户偏好行为，主要针对布料折叠任务。

**💡 创新点**

提出RKO方法，将KTO的二元标签学习与RPO的相似度加权结合，并系统比较DPO、RPO、KTO与Vanilla DDPM。

**🔧 技术方法**

使用基于DDPM的视觉运动扩散模型，并实现DPO、RPO、KTO、RKO等偏好对齐框架。

**📊 数据集**

使用真实机器人收集的三类服装（裤子、袖子、T恤）折叠演示数据，包含60个优选示范与相应的失配示范，参考模型基于100个示范训练。

**📈 对比分析**

在固定60个演示以及样本效率（20-95个演示）两种设置下，与Vanilla DDPM对比；RKO在大多数任务中表现最佳，样本效率最高，且训练更快。

**⚠️ 局限性**

在极少数据（如20个演示）下RKO易受噪声影响；对离群状态鲁棒性有限；实验范围仅限于折叠任务，未验证其他柔性物体。

---

## 247. Higher Hardness Results for the Reconfiguration of Odd Matchings

**arXiv ID:** 2602.09573 | [PDF](https://arxiv.org/pdf/2602.09573v1)

**作者:** Joseph Dorfer `[一作]` `[通讯]` (Graz University of Technology), Joseph Dorfer (Graz University of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究图中奇数匹配（仅缺少一个顶点的匹配）的重新配置问题，证明其翻转图的直径问题属于Π₂^p‑难、半径问题属于Σ₃^p‑难，并证明最短翻转序列问题对数可近似不可实现。

**💡 创新点**

创新点在于将奇数匹配重新配置的直径和半径问题分别提升到多项式层级的最高两个层次，并通过从量化SAT和集合覆盖的多项式时间归约，首次给出这类问题在更高复杂度层级上的完备性与近似难度。

**🔧 技术方法**

核心技术包括构造专用的“句子”“∀‑gadget”“∃‑gadget”等图形化 gadget；利用这些 gadget 设计可编码量化布尔公式的翻转路径；并通过精确计数翻转步数的引理，将复杂度归约到所需层级。

**📊 数据集**

论文为纯理论工作，没有使用实际数据集，而是基于抽象图结构与布尔公式实例进行构造与归约。

**📈 对比分析**

由于研究的是复杂度与可逼近性，上文未给出实验比较；所提出的结果仅为理论上限，表明在多项式层级中难以设计多项式时间或高效近似算法。

**⚠️ 局限性**

局限性包括：仅处理组合（非几何）图的奇数匹配；未提供正向算法或实用启发式；且对于几何图形的对应问题仍未解决。

---

## 248. ECG-IMN: Interpretable Mesomorphic Neural Networks for 12-Lead Electrocardiogram Interpretation

**arXiv ID:** 2602.09566 | [PDF](https://arxiv.org/pdf/2602.09566v1)

**作者:** Vajira Thambawita `[一作]` (SimulaMet), Pål Halvorsen `[通讯]` (SimulaMet)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出ECG‑IMN，使用超网络生成针对每个样本的线性模型实现对12导联ECG的可解释分类

**💡 创新点**

将可解释的Mesomorphic Neural Networks迁移到时序多导联信号，并引入Transition Decoder生成高分辨率权重以精准定位病理特征

**🔧 技术方法**

深度卷积骨干+超网络+Transition Decoder+线性决策+L1正则化

**📊 数据集**

PTB‑XL 12导联ECG数据集（约2.2万记录）

**📈 对比分析**

与标准黑盒CNN对比，AUROC差距≤2%，并通过交互可视化展示归因热图，证明解释性不牺牲精度

**⚠️ 局限性**

对窗口聚合参数敏感、需人工调节；目前仅验证二分类任务，未扩展到多标签或更复杂的临床场景

---

## 249. Optimal Control of Microswimmers for Trajectory Tracking Using Bayesian Optimization

**arXiv ID:** 2602.09563 | [PDF](https://arxiv.org/pdf/2602.09563v1)

**作者:** Lucas Palazzolo `[一作]` (Universite Cote d'Azur), Laëtitia Giraldi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出使用贝叶斯优化结合B样条曲线对低雷诺数微观游泳器的轨迹跟踪进行最优控制。

**💡 创新点**

首次将贝叶斯优化应用于微观游泳器的控制，兼顾高维优化与复杂动力学，并实现对壁面影响的补偿。

**🔧 技术方法**

贝叶斯优化（SCBO）、B样条曲线、低维ODE模型（N‑link）与高维PDE模型（三球游泳器）以及有限元求解。

**📊 数据集**

采用数值仿真数据，构造多种参考轨迹（直线、椭圆、斜柱面等）并在不同模型上进行实验。

**📈 对比分析**

与传统正弦磁场驱动和经典游泳姿态相比，优化后的控制实现了更高的跟踪精度和更大的位移，并能在壁面附近成功补偿角度漂移。

**⚠️ 局限性**

局限在于仅验证了单个模型，未考虑多体碰撞、复杂边界或最终时间/初始状态优化，计算成本仍高且对控制维度敏感。

---

## 250. ReSIM: Re-ranking Binary Similarity Embeddings to Improve Function Search Performance

**arXiv ID:** 2602.09548 | [PDF](https://arxiv.org/pdf/2602.09548v1)

**作者:** Gianluca Capozzi `[一作]` (KASTEL Security Research Labs), Giuseppe Antonio Di Luna `[通讯]` (Sapienza University of Rome)

**通讯引用:** 1118 | [OpenAlex ID](https://openalex.org/A5103081624)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于二阶段检索的二进制函数相似性搜索框架，利用深度编码器先进行高效检索，再用交叉编码器重新排序候选结果，以提升函数检索的准确度。

**💡 创新点**

创新点在于：①引入神经重排序模块，突破传统 bi‑encoder 只能独立编码的瓶颈；②使用大规模预训练模型 DeepSeek‑R1‑Qwen3‑8B 进行微调，实现跨工具链知识迁移；③支持多模型集成，进一步提升性能。

**🔧 技术方法**

技术细节包括：Transformer‑based bi‑encoder（多种已公开模型）、交叉编码器 re‑ranker（采用 LoRA + 4‑bit QLoRA 微调）、最大内积搜索（MIPS）实现检索、硬件加速的并行推理。

**📊 数据集**

数据集主要有：
- ArchLinux–26M（训练和基准）
- SimTestData（多编译器/优化级别的多工具链测试）
- CVE‑基准（真实漏洞函数检索）。

**📈 对比分析**

与基线 bi‑encoder 系统对比，重排序后在两大数据集上平均提升 nDCG@k 21.7%、Recall@k 27.8%；在最差模型上可达 40%+ 的提升；集成多模型可额外提升约 3%。

**⚠️ 局限性**

局限性包括：
- 重新排序阶段计算成本高，窗口大小 w 越大推理时间线性增长；
- 仅在 x86‑64 架构下验证，跨架构泛化待研究；
- 依赖大规模预训练模型，对硬件资源和能耗有较高要求。

---

## 251. Scalpel: Fine-Grained Alignment of Attention Activation Manifolds via Mixture Gaussian Bridges to Mitigate Multimodal Hallucination

**arXiv ID:** 2602.09541 | [PDF](https://arxiv.org/pdf/2602.09541v1)

**作者:** Ziqiang Shi `[一作]` (Fujitsu Research and Development Center Co.,LTD.), Koichi Shirahata `[通讯]` (Fujitsu Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在 LVLM 推理阶段对注意力激活进行自适应干预，降低生成文本与视觉内容的不一致（幻觉）。

**💡 创新点**

创新点在于：① 用高斯混合模型 (GMM) 分别建模可信与幻觉注意力空间；② 用 Schrödinger 桥（等价于熵正则化最优传输）计算从幻觉到可信的最小干预路径；③ 依据每个注意力头和每个 token 的分布动态调整干预强度。

**🔧 技术方法**

技术方法包括：Gaussian Mixture Model、可微最优传输/Schrödinger Bridge、对齐算法（最小成本流）、顶级 k 注意力头筛选、Logistic 回归探针、t‑SNE 可视化等。

**📊 数据集**

实验数据集：POPE（object hallucination 27K 交互式查询）、MME（14 子任务，涵盖感知与推理）、以及底层图像源 MSCOCO、A‑OKVQA、GQA。

**📈 对比分析**

与 Vanilla、VCD、OPERA、ICT 等传统方法在 POPE（Accuracy/F1）和 MME 上进行对比；Scalpel 在所有子集均实现 2–4% 的准确率提升，最坏的 Adversarial 子集中提升超过 10%（F1），并在多模态任务上实现 SOTA。

**⚠️ 局限性**

局限性：① 需要预先训练 GMM 与探针，调参较多；② 只在推理阶段干预，未验证对模型结构修改的可迁移性；③ 对极端多模态场景或需要多步推理的任务尚未充分评估；④ 受 GMM 分量数限制，过多分量会增加计算成本。

---

## 252. UniARM: Towards a Unified Autoregressive Reward Model for Multi-Objective Test-Time Alignment

**arXiv ID:** 2602.09538 | [PDF](https://arxiv.org/pdf/2602.09538v1)

**作者:** Hongyan Xie `[一作]` (Beihang University), Shuangyong Song `[通讯]` (China Telecom)

**通讯引用:** 501 | [OpenAlex ID](https://openalex.org/A5087486098)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了统一自回归奖励模型（UniARM）及其参数高效实现MoSLoRA，用于多目标测试时对大语言模型进行对齐，避免了多目标特征混淆并支持实时调节偏好权重。

**💡 创新点**

创新点在于：①用统一参数空间一次性学习所有偏好维度，消除独立参数的冗余；②引入Preference‑Modulated & Shared Low‑Rank Adaptation（MoSLoRA），通过共享特征+偏好调制实现特征分离与灵活调节；③在弱模型引导强模型的弱→强扩展中保持性能提升。

**🔧 技术方法**

技术包括：自回归奖励模型（ARM）+低秩适配（LoRA）+混合偏好向量调制+联合本地/全局损失训练；在推理时通过特征尺度变换实现参数融合。

**📊 数据集**

主要使用的公开数据集：安全对齐任务的 QA 对偏好标签数据；帮助助手任务的 160K 对话数据；以及公开奖励模型作为 oracle。

**📈 对比分析**

与 RS、MOD、GenARM、PARM 等基线相比，UniARM 在安全对齐任务中 HV 提升 18.5%，MIP 提升 30.2%；在帮助助手任务中 HV 提升 5.4%，MIP 提升 10.7%；且参数量与 PARM 相同、推理速度相当。

**⚠️ 局限性**

局限性包括：仍需基准模型的预训练；对极端偏好组合的泛化未知；对多维偏好空间的高阶交互处理仍有限；实验主要在开源 LLaMA 系列模型上验证，跨模型的迁移性待进一步验证。

---

## 253. Autoregressive Direct Preference Optimization

**arXiv ID:** 2602.09533 | [PDF](https://arxiv.org/pdf/2602.09533v1)

**作者:** Masanari Oi `[一作]` (Institute of Science Tokyo), Nakamasa Inoue `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 846 | [OpenAlex ID](https://openalex.org/A5101261291)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ADPO（Autoregressive Direct Preference Optimization），在 DPO 的 Bradley‑Terry 模型中显式引入前缀闭包与自回归假设，得到新的损失形式并可实现细粒度的偏好优化。

**💡 创新点**

创新点在于：①在 BT 模型前引入自回归前缀能量，使得求和移出 log‑sigmoid；②揭示两种长度度量（token 长度 μ 与反馈长度 μ′）并证明任何奖励函数可重参数化为自回归模型；③提供了静态与自适应两类 granularity 家族。

**🔧 技术方法**

采用 DPO/ RLHF 训练框架，基于 Boltzmann 分布与前缀能量的理论推导，使用强组合 (strong composition) 细粒度拆分；在模型端采用 LoRA 微调与 AdamW 优化。

**📊 数据集**

实验数据集包括数学推理数据集 GSM8K 与 MATH500，以及会话评测基准 AlpacaEval 2、Arena‑Hard 与 MT‑Bench。

**📈 对比分析**

与 DPO、cDPO、SimPO 等基线在四大 LLM（Llama‑3‑8B、Gemma‑3‑12B、Qwen‑3‑8B、DeepSeek‑Math‑7B）上对比，ADPO（尤其 cADPO）在 GSM8K/MATH 的准确率、会话任务的赢率/分数均超过基线，表现出显著提升。

**⚠️ 局限性**

局限性：仍以 KL‑约束的奖励最大化为前提，未探索更一般的散度度量；在静态 granularity 族中需 padding 可能限制性能；实验范围受限于所选模型与数据集。

---

## 254. MILE-RefHumEval: A Reference-Free, Multi-Independent LLM Framework for Human-Aligned Evaluation

**arXiv ID:** 2602.09624 | [PDF](https://arxiv.org/pdf/2602.09624v1)

**作者:** Nalin Srun `[一作]` (Universite de Lorraine), Lydia Boudjeloud Assala `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MILE‑RefHumEval框架，利用多个独立LLM评估器在无参考答案、无交互的条件下对生成文本进行多维度评估；

**💡 创新点**

创新点在于：①独立评估而非对话，消除交互偏差；②多样化LLM组合，提升鲁棒性；③可通过统一的提示模板灵活切换任务；④大幅降低查询量，保持高效；

**🔧 技术方法**

核心技术包括提示工程、独立评估器集成、多数投票/平均聚合、标准统计评估（Acc、F1、MCC、Kappa、MSE/RMSE/MAE）等；

**📊 数据集**

使用了FairEval、SummEval、OID‑Rated Image Caption、PandaLM、Topical‑Chat等多任务/多模态基准；

**📈 对比分析**

与CHATEVAL、MILE‑RefHumEval‑Conv、G‑Eval‑4、ClipScore等方法对比，MILE‑RefHumEval在准确率、F1、相关系数、误差指标上均优于对手，同时查询量比对手低30‑70%；

**⚠️ 局限性**

局限性包括：对领域特定任务的适应性不足；提示对模型行为影响大；最佳评估器数量未确定；交互式设置仍存在顺序偏差；评估粒度与因果分析不足。

---

## 255. AGMark: Attention-Guided Dynamic Watermarking for Large Vision-Language Models

**arXiv ID:** 2602.09611 | [PDF](https://arxiv.org/pdf/2602.09611v1)

**作者:** Yue Li `[一作]` (East China Normal University), Linlin Wang `[通讯]` (East China Normal University)

**通讯引用:** 1600 | [OpenAlex ID](https://openalex.org/A5115695095)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种面向大规模视觉‑语言模型（LVLM）的动态水印方案 AGMark，用来在保持视觉语义一致性的前提下嵌入可检测的水印信号。

**💡 创新点**

创新点在于：① 通过注意力权重实时提取语义‑关键权重，动态识别每个解码步骤中与视觉内容最相关的 token；② 结合上下文一致性与视觉重要性做标准化融合；③ 基于 token 熵与权重密度自适应地调整词表分区比例，既提升文本质量，又保持水印强度。

**🔧 技术方法**

技术细节包括：视觉‑文本共享嵌入空间、注意力机制提取视觉关注、标准化与凸组合融合语义关键权重、熵与密度评估确定保护比例、logits‑based 水印偏置（正向偏置）以及后续检测统计（AUC/Accuracy）。

**📊 数据集**

使用公开视觉‑语言基准 AMBER 与 MS‑COCO 对 Llava‑Next‑Llama3、Qwen3‑VL、InternVL‑3.5 三个 8B LVLM 进行评估。

**📈 对比分析**

与 KGW、SynthID、IE、MorphMark、VLA‑Mark 等五种主流水印基线相比，AGMark 在 AUC（≥99.3%）和 Accuracy（≈99.5%）均处于前两名；同时在 Perplexity、BLEU、BertScore、STS 以及视觉一致性指标 CHAIR 上均显著优于或接近最优（如 Perplexity 下降 0.18，BLEU 提升 5% 以上，CHAIR 下降 1–2%）。

**⚠️ 局限性**

主要限制包括：① 需要手动调节三个关键超参数（ω、α、τ），对不同模型与数据集敏感；② 动态计算增加轻微的推理延时；③ 对极端的文本重写（如大规模同义替换/翻译）仍会有一定的 AUC 降低；④ 目前仅验证于视觉‑文本双模场景，未扩展到更复杂多模态或非图像输入。

---

## 256. Advancing Block Diffusion Language Models for Test-Time Scaling

**arXiv ID:** 2602.09555 | [PDF](https://arxiv.org/pdf/2602.09555v1)

**作者:** Yi Lu `[一作]` (Fudan University), Wei Wang `[通讯]` (Meituan LongCat Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为Block Diffusion Language Model（BDLM）提出一种统一的测试时缩放框架，结合动态解码和块大小自适应，以提升长Chain‑of‑Thought推理的速度与质量。

**💡 创新点**

创新点在于：①设计了Bounded Adaptive Confidence Decoding (BACD)，通过上下界阈值动态调节未掩码 token 的解码速率；②提出Think Coarse, Critic Fine (TCCF) 方案，将粗粒度思考与细粒度评审分阶段进行；③引入Progressive Block Size Extension 逐步扩大块大小以缓解大块训练难题。

**🔧 技术方法**

技术包括：BDLM 架构、离散化掩码扩散过程、动态置信度采样、块大小自适应、KV 缓存、前向过程与反向过程的联合训练。

**📊 数据集**

数据集涵盖六类推理任务：数学推理（Math500、AIME24、AIME25、AMC23）、代码生成（LiveCodeBench v5）、STEM 推理（GPQA‑diamond）等。

**📈 对比分析**

与现有 BDLM（LLaDA、Fast‑dLLM‑v2、SDAR‑8B‑Chat 等）以及同源自 Qwen3‑8B 的自回归模型做对比。实验显示：TDAR‑8B‑Thinking 取得与 TraDo‑8B‑Thinking 相比平均 3.4 分的提升，TPF 由 1.27 提升至 2.97；加上 BACD 后 TPF 进一步提升至 3.37，准确率再提升 1.6 分；TCCF 在保持 3.04 TPF 的同时，将 AIME24 得分从 36.3 提升至 42.9，整体性能与速度均达到或超过基线。

**⚠️ 局限性**

局限性：①主要在 8B 规模模型上验证，尚未测试更大规模或多语言场景；②对极长文本的推理效果与错误聚集仍需进一步评估；③BACD 与 TCCF 的阈值和块大小设定仍依赖经验调参，可能在不同任务间迁移性有限。

---

## 257. UniShare: A Unified Framework for Joint Video and Receiver Recommendation in Social Sharing

**arXiv ID:** 2602.09618 | [PDF](https://arxiv.org/pdf/2602.09618v1)

**作者:** Caimeng Wang `[一作]` (Kuaishou Technology), Jianhui Bu `[通讯]` (Kuaishou Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 UniShare，统一模型同时预测短视频分享的概率与接收者推荐，解决了传统分离模型导致的推荐不匹配与数据稀疏问题。

**💡 创新点**

创新点包括：① 将分享视为三元交互，统一建模；② 引入双边兴趣与关系-内容匹配机制；③ 使用预训练的图神经网络和多模态嵌入增强稀疏特征；④ 采用层级负采样与联合多任务学习，互相提升两任务表现。

**🔧 技术方法**

技术手段包括：图神经网络（GNN）预训练嵌入、视频和用户多模态嵌入、双向注意力机制、关系-内容对齐（LLM 生成匹配得分）、层级负采样、联合训练与加权损失、特征共享策略。

**📊 数据集**

使用 K-Share 数据集，来源于快手平台一个月的日志，包含约 20,000 位高活跃分享用户、1,000,000 条分享实例、1,000 万条视频播放记录、12 万条好友关系与 180,000 条接收者候选集。

**📈 对比分析**

与基线 PLE（视频）和 DCN（接收者）进行对比。UniShare 在离线指标上：视频 AUC 提升 1.01% (0.7588 vs 0.7512)，接收者 AUC 提升 2.01% (0.9307 vs 0.9124)。在线 A/B 测试显示：分享量提升 1.95%，独立分享用户 +0.805%，分享按钮 CTR +1.12%，分享面板 CTR +1.14%，接收者回复率 +0.482%。

**⚠️ 局限性**

主要局限：联合模型在推理阶段需要计算 |视频|×|接收者| 的样本量，导致推理成本高；目前仅在快手平台验证，泛化到其他短视频平台需进一步评估；对极端冷启动用户或视频仍存在稀疏挑战。

---

## 258. AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception

**arXiv ID:** 2602.09617 | [PDF](https://arxiv.org/pdf/2602.09617v1)

**作者:** Ruoxuan Feng `[一作]` (Renmin University of China), Di Hu `[通讯]` (Renmin University of China)

**通讯引用:** 2243 | [OpenAlex ID](https://openalex.org/A5100670614)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了触觉动态金字塔框架，构建了覆盖三高阶层的超大规模触觉数据集 ToucHD，并基于此设计了具备多层次动态感知目标的 AnyTouch 2 代表学习模型。

**💡 创新点**

创新点在于：①以触觉动态金字塔为系统化引导，统一数据采集与模型设计；②提出涵盖特定动作、真实操作与力对齐的三级高阶数据集 ToucHD；③在 AnyTouch 2 中融合像素级差分重建、动作匹配、力预测与跨模态对齐等多层次目标，实现在不同动态层级上的一致表现。

**🔧 技术方法**

技术方法包括：视频掩码自编码器 + 帧差分重建提升细粒度时序敏感；多模态对齐（视觉、语言）与跨传感器匹配实现传感器无关的对象语义；动作匹配对结构化动态进行语义化；力预测与增量力预测将物理动力学嵌入特征；层次化任务调度与自适应权重实现多目标协同学习。

**📊 数据集**

使用的数据集包括：新建的 ToucHD（Sim、Mani、Force 三子集）、TAG、VisGel、ObjectFolder Real、TVL、YCB-Slide、SSVTP、Octopi、TacQuad、Sparsh、Cloth 等多种公开触觉与视觉语料。

**📈 对比分析**

通过在 Object Bench、Sparsh Bench 与自建 ToucHD Bench 进行离线评测，以及在四项覆盖 Tier 1–5 的实操抓取、擦拭、插拔与芯片移动任务中对比 UniTouch、T3、MAE、VJEPA 与 AnyTouch 1 等基线，AnyTouch 2 在动态感知任务上普遍优于基线，尤其在高阶 Tier 1 任务中显著提升成功率，并在多传感器实操中保持最优表现。

**⚠️ 局限性**

局限性包括：高阶（Tier 1）力感知仍受限于力对齐数据量；跨模态对齐在提升静态语义时可能削弱对细粒度动态的敏感度；模型复杂度较高，训练成本和推理时延不可忽视；未来仍需扩充更多高层级数据与多种传感器，以进一步强化通用动态触觉感知。

---

## 259. When Handshakes Tell the Truth: Detecting Web Bad Bots via TLS Fingerprints

**arXiv ID:** 2602.09606 | [PDF](https://arxiv.org/pdf/2602.09606v1)

**作者:** Ghalia Jarad `[一作]` (Istanbul Technical University), Kemal Bicakci `[通讯]` (Istanbul Technical University)

**通讯引用:** 1626 | [OpenAlex ID](https://openalex.org/A5085532973)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对TLS握手的JA4指纹进行特征提取，并利用梯度提升树模型（XGBoost、CatBoost）训练分类器，区分网络中的恶意机器人流量与真实人类流量。

**💡 创新点**

创新点在于：①首次系统评估JA4指纹在真实网络流量中的区分效果；②使用低层协议特征（如cipher_count、ext_count、ja4_b等）实现对机器人高度识别；③提出基于JA4的防御方法对抗IP轮换与User-Agent伪装。

**🔧 技术方法**

技术包括：TLS指纹提取（JA4）、特征工程（将JA4字符串拆解为结构化特征）、梯度提升树机器学习（XGBoost、CatBoost）、性能评估（准确率、召回率、F1、AUC）以及特征重要性分析。

**📊 数据集**

使用公开的JA4DB数据集，包含约227,404条记录，其中包含人类流量、良性爬虫和恶意机器人流量，经过标签处理后训练集与测试集按80/20划分。

**📈 对比分析**

对比方法为：训练XGBoost与CatBoost两模型并在相同测试集上评估。CatBoost在准确率0.9863、F1 0.9734、AUC 0.998方面略优于XGBoost（准确率0.9862、F1 0.9732、AUC 0.998），两者均表现出极低的误报率与漏报率。

**⚠️ 局限性**

局限性包括：①无法识别使用真实浏览器引擎（如Puppeteer/Selenium）或专门模仿浏览器TLS指纹的高级机器人；②模型对未来机器人可能的TLS指纹变更或主动伪造仍存在适应性挑战；③仅基于TLS指纹，若攻击者同时攻击多层协议，需与其他信号结合使用。

---

## 260. On the Optimal Reasoning Length for RL-Trained Language Models

**arXiv ID:** 2602.09591 | [PDF](https://arxiv.org/pdf/2602.09591v1)

**作者:** Daisuke Nohara `[一作]` (Institute of Science Tokyo), Rio Yokota `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 1786 | [OpenAlex ID](https://openalex.org/A5024747717)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了长度控制方法对强化学习提升大型语言模型推理能力的影响。

**💡 创新点**

创新点在于揭示不同模型对输出长度的最优策略差异，并阐明长输出导致分散、短输出导致少思考的两种失败模式。

**🔧 技术方法**

使用了RLOO-LP、ALP、DRPO等长度惩罚技术，并将其应用于RL训练的策略。

**📊 数据集**

实验数据集基于两款模型（Qwen3-1.7B Base和DeepSeek-R1-Distill-Qwen-1.5B）的数学推理任务。

**📈 对比分析**

通过比较不同长度控制方法的表现，发现对具备先验推理能力的模型，适当的长度控制可提升效率；对推理能力弱的模型，需更长输出才能获得最佳性能。

**⚠️ 局限性**

研究仅覆盖两款模型和单一任务，缺乏自动寻找最优长度的机制，且需进一步探索更广泛场景。

---

## 261. Context-Aware Counterfactual Data Augmentation for Gender Bias Mitigation in Language Models

**arXiv ID:** 2602.09590 | [PDF](https://arxiv.org/pdf/2602.09590v1)

**作者:** Shweta Parihar `[一作]` (University of Illinois at Chicago), Lu Cheng `[通讯]` (University of Illinois at Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Context-CDA 方法，用大语言模型为传统反事实数据增强生成上下文丰富的示例，并结合语义熵过滤低质量样本，再在目标小型语言模型上进行微调，实现性别偏见减缓。

**💡 创新点**

创新点包括：①利用大模型生成上下文增强的反事实，显著提升语料自然度与多样性；②引入语义熵作为不确定性度量，对生成文本进行质量过滤；③实现了对编码器、解码器及编码-解码器模型的统一、模型无关的偏见缓解流程。

**🔧 技术方法**

技术手段：反事实数据增强 (CDA)、大模型提示生成 (prompting)、语义熵过滤、语言模型微调、对比实验与多维度偏见评估。

**📊 数据集**

使用的数据集：训练集采用 news-commentary（带性别词表）；评估集包括 StereoSet、CrowS-Pairs、BiasBios、STS-B、NLI-Bias；下游任务评估使用 GLUE 任务 QNLI、RTE、SST-2；大模型如 Llama‑3‑8B‑Instruct 用于上下文生成。

**📈 对比分析**

与 Vanilla、传统 CDA 及多种现有去偏方法（MABEL、INLP、SelfDebias、SENT‑DEBIAS、wiki‑debiased）对比，Context‑CDA 在所有五种模型（BERT、DistilBERT、T5、GPT‑2、Llama‑3.2‑1B）上在 Intrinsic、Extrinsic 及下游任务指标均不逊于或优于传统 CDA，且在语言建模分数和偏见指标上均保持或提升，证明其在保留语言能力的前提下有效减轻性别偏见。

**⚠️ 局限性**

局限性：①使用大模型生成上下文需较高算力与能源成本；②语义熵过滤可能丢弃有价值的复杂示例；③当前仅针对二元性别，未覆盖非二元或交叉身份；④对不同领域或多语言环境的适配性待进一步验证；⑤缺少人类审查或事实核查以保证生成内容的可靠性。

---

## 262. Mitigating the Likelihood Paradox in Flow-based OOD Detection via Entropy Manipulation

**arXiv ID:** 2602.09581 | [PDF](https://arxiv.org/pdf/2602.09581v1)

**作者:** Donghwan Kim `[一作]` (Yonsei University), Hyunsoo Yoon `[通讯]` (Yonsei University)

**通讯引用:** 1248 | [OpenAlex ID](https://openalex.org/A5076379562)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为语义比例熵操控（SPEM）的方法，以解决生成模型在处理离群数据（OOD）时的似然悖论问题。

**💡 创新点**

通过操控输入的熵来提高OOD检测的性能，尤其是对不相似的输入施加更强的扰动，从而增强了在分布内（ID）和离群（OOD）样本之间的似然差距。

**🔧 技术方法**

使用了高斯扰动和预训练的特征提取器来编码语义信息，并通过控制熵来调整输入的扰动强度。

**📊 数据集**

使用了CIFAR-10、CIFAR-100、SVHN、CelebA、MNIST和FashionMNIST等标准数据集进行评估。

**📈 对比分析**

与基于似然的OOD检测器进行了比较，SPEM在多个标准基准上表现出一致的AUROC提升，超越了其他基线方法。

**⚠️ 局限性**

该方法在处理高相似性输入时可能不够灵活，且在某些情况下，SPEM-noise（仅使用噪声进行检测）在性能上可能优于SPEM。

---

## 263. Rollout-Training Co-Design for Efficient LLM-Based Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.09578 | [PDF](https://arxiv.org/pdf/2602.09578v1)

**作者:** Zhida Jiang `[一作]` (JD.com), Ke Zhang `[通讯]` (JD.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FlexMARL 框架，实现 LLM‑基多智能体强化学习的端到端 roll‑out 与训练协同。

**💡 创新点**

创新点包括：并行采样加分层负载均衡、按需 agent‑centric 资源分配、微批级异步管线以消除长尾同步瓶颈。

**🔧 技术方法**

技术实现涵盖分离式架构、经验存储、微批异步管线、并行采样、层级负载平衡、进程组 agent‑centric 调度、统一跨层 API、D2D/H2D/D2H 移动。

**📊 数据集**

使用两大工业数据集：电商商家助手（MA）与品类助手（CA），模型为 Qwen2.5‑14B/32B。

**📈 对比分析**

与 MAS‑RL、DistRL、MARTI 对比，FlexMARL 在 MA 上实现 7.3× 速度提升、9.10k tps 处理率；在 CA 上 5.6× 速度提升、8.21k tps；硬件利用率提升至约 32%/20%。

**⚠️ 局限性**

局限包括：仍需手动调节负载阈值、对极大规模 agent 的状态迁移存在一定延迟、依赖专有硬件与 Ray 等平台。

---

## 264. Predictive Query Language: A Domain-Specific Language for Predictive Modeling on Relational Databases

**arXiv ID:** 2602.09572 | [PDF](https://arxiv.org/pdf/2602.09572v1)

**作者:** Vid Kocijan `[一作]` (Kumo.ai), Jure Leskovec `[通讯]` (Kumo.ai)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种 SQL‑风格的声明式预测查询语言 PQL，能一次性定义实体、目标标签、时间窗口等，自动生成机器学习训练表，支持大规模批处理与低延迟交互。

**💡 创新点**

通过把预测任务编译为统一的抽象计划，PQL 解决了标签生成中的时间泄漏、实体过滤与任务推断难题，并与关系深度学习与基础模型无缝集成，实现了从原始关系表到可训练标签的全自动流水线。

**🔧 技术方法**

使用 ANTLR4 解析语法，生成逻辑计划后在 Spark 上做大规模批处理，或在预构建的异构图上用自定义采样器（GraphSAGE）结合 Pandas 进行低延迟推理；对时间窗口、聚合、过滤等做多级优化。

**📊 数据集**

在 rel‑Amazon（约 1.8M 用户/500k 商品/20M 评测）、Fannie Mae（15M 贷款/600M 还款）以及合成 H&M（275M 用户/6.3B 交易）等行业基准上验证，亦使用真实 H&M 电商数据进行交互式实验。

**📈 对比分析**

与基线 Pandas/手工实现对比，PQL 在批处理上可实现 6×+ 的加速（大型数据集可完成原始实现超时的任务），在低延迟模式下可达 40× 的速度提升；同时显著降低人工错误和时间泄漏风险，模型精度与传统流程相当或更优。

**⚠️ 局限性**

受限于对时间戳、主键单列、预先注释好语义的假设；缺少显式 JOIN、列间算术比较、非原子主键等语法；需要在不符合最佳实践的数据库中进行额外预处理，且对多实体或集合型标签的支持尚不完善。

---

## 265. LEMUR: A Corpus for Robust Fine-Tuning of Multilingual Law Embedding Models for Retrieval

**arXiv ID:** 2602.09570 | [PDF](https://arxiv.org/pdf/2602.09570v1)

**作者:** Narges Baba Ahmadi `[一作]` (University of Hamburg), Chris Biemann `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多语言欧盟环境立法语料库LEMUR，并对其进行法律文档检索模型的对比实验

**💡 创新点**

提出了Lexical Content Score (LCS) 用于量化PDF转文本的质量，并在LEMUR上实现跨语言法律检索的对比学习微调

**🔧 技术方法**

使用多语言对比学习(MNR)和多正样本扩展进行嵌入模型微调，主要模型包括E5、Qwen‑0.6B、Qwen‑4B

**📊 数据集**

利用EUR‑Lex官方PDF文件（24,953份，25种语言）构成的LEMUR语料，覆盖环境类别15和子类别10

**📈 对比分析**

在单语、双语（EN‑LV）以及跨语（源语言微调后在目标语言评估）三种设置下比较检索性能，Top‑1、Top‑3、Top‑5准确率均显著提升，低资源语言效果尤为明显

**⚠️ 局限性**

限制包括语料覆盖范围局限于环境法类别、双语微调仅测试了一对语言、PDF转文本仍存在约6%噪声

---

## 266. Shifting landscape of disability and development in India: Analysis from historical trends to future predictions 2001-2031

**arXiv ID:** 2602.09543 | [PDF](https://arxiv.org/pdf/2602.09543v1)

**作者:** Hana Kapadia `[一作]` (Cambridge Centre for International Research), Arun Kumar Rajasekaran `[通讯]` (University of Cambridge)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5101967962)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对印度各邦2001‑2021年间的残疾负担（DALY）与人类发展指数（HDI）及性别比例进行纵向分析，并利用回归模型预测2031年的DALY与HDI。

**💡 创新点**

将HDI与三类DALY（传染病、非传染病、伤害）分离建模，首次在同一研究中同时评估性别差异与未来发展趋势，并采用指数衰减模型拟合传染病DALY的显著下降。

**🔧 技术方法**

使用线性回归、二次多项式回归和指数衰减回归三种方法，最终选择R²最高且稳定的模型来预测HDI及三类DALY。

**📊 数据集**

数据来源为2001/2011印度人口普查、IHME发布的DALY估计和Global Data Lab的HDI数据，构成每邦6个特征的时间序列数据集。

**📈 对比分析**

通过比较不同回归模型的R²值和预测误差，指数衰减模型在传染病DALY上R²≈1，线性回归在非传染病和伤害DALY上R²>0.9，预测结果显示非传染病DALY上升、传染病及伤害DALY下降，验证了方法的可行性。

**⚠️ 局限性**

局限包括：数据仅以十年间隔，缺乏年度细节；部分联邦领土及奥里萨邦缺失；依赖IHME全球估计可能掩盖地方差异；线性外推未考虑非线性变化及突发事件；2021年普查缺失导致性别比例缺失；女性残疾可能因社会污名被低报。

---

## 267. On the complexity of Sandwich Problems for $M$-partitions

**arXiv ID:** 2602.09576 | [PDF](https://arxiv.org/pdf/2602.09576v1)

**作者:** Alexey Barsukov `[一作]` (Charles University), Santiago Guzmán-Pro `[通讯]` (Dresden University of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文研究了2-边缘彩色图的结构分类及其在矩阵划分问题中的计算复杂性，提出了一种完整的多项式/NP难的判定准则；

**💡 创新点**

创新点在于将二维彩色图的可判定问题与3-SAT的矩阵划分问题精确对应，首次实现对“全同构图”与“全异构图”的统一结构分类，并提供了高效的多项式时间判定算法；

**🔧 技术方法**

主要技术包括可判定的结构化分解（同余拼接与交替分量分解）、原型化与可谓定义、循环与周期幂图构造以及DAG交错组件分析；

**📊 数据集**

该工作不依赖于外部数据集，完全在理论框架下进行；

**📈 对比分析**

方法对比：对所有2-彩色图实现了完全的P/NP判定，性能上即判定算法在多项式时间内完成；

**⚠️ 局限性**

局限性在于当前只能处理完全反射（自环满足同色/异色）且无*边的图；若存在更复杂边缘交错结构，算法需进一步扩展。

---

## 268. High-performance Vector-length Agnostic Quantum Circuit Simulations on ARM Processors

**arXiv ID:** 2602.09604 | [PDF](https://arxiv.org/pdf/2602.09604v1)

**作者:** Ruimin Shi `[一作]` (KTH Royal Institute of Technology), Ivy Peng `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1193 | [OpenAlex ID](https://openalex.org/A5037069204)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

改造谷歌 Qsim 状态向量模拟器，使其在 ARM SVE 的向量长度无关（VLA）架构上可单源实现，并在 Grace、Graviton 与 A64FX 上进行性能评测。

**💡 创新点**

提出了 VLA 设计与 VLEN 自适应内存布局、缓冲、细粒度循环控制和门融合等优化技术，同时定义新度量与 PMU 事件以评估 VLA 效率。

**🔧 技术方法**

采用 ARM SVE Intrinsics、OpenMP 并行、Gate Fusion、FP32 SIMD、PMU 计数、Cirq 生成量子电路以及 Python-C++ 绑定。

**📊 数据集**

使用五个典型量子电路（QFT、Grover、GHZ、QRC、QV）至 36 量子位的测试集，由 Cirq 构造。

**📈 对比分析**

在同一代码基上对比 auto‑vectorization、SVE 优化以及 H100 GPU 加速，通过跑时、IPC、AVL、IRR 等指标评测；SVE 版本在 A64FX 提升 4.5×、Grace 2.5×、Graviton 1.5×，并能在 CPU 上模拟 36 量子位电路，而 GPU 受显存限制。

**⚠️ 局限性**

受限于内存子系统匹配、后端存储瓶颈、不同平台 SVE 指令实现差异，门融合受缓存与 AI 平衡约束；Graviton 上性能提升有限，且未能在 GPU 上大规模扩展。

---

## 269. Learning from the Irrecoverable: Error-Localized Policy Optimization for Tool-Integrated LLM Reasoning

**arXiv ID:** 2602.09598 | [PDF](https://arxiv.org/pdf/2602.09598v1)

**作者:** Qiao Liang `[一作]` (Tongji University), Sheng Guo `[通讯]` (MYbank)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种错误局部化策略优化（ELPO），通过二分搜索树定位长序列工具集成推理（TIR）中首次不可恢复的错误步骤，进而实现更细粒度的信用分配与更新。

**💡 创新点**

创新点在于：①在固定采样预算下利用熵差引导的二分搜索快速定位错误步骤；②将树结构优势与全局轨迹优势结合，构建分层优势归因；③针对错误步骤及其后缀采用自适应剪辑，增强纠正力度。

**🔧 技术方法**

主要技术包括：基于树的rollout、熵差指导的路径剪枝、分层优势归因（分支级与轨迹级）、错误局部化自适应剪辑（ELC）以及PPO/GRPO风格的策略优化。

**📊 数据集**

在三类长序列推理数据集上验证：数学推理（MATH、GSM8K、AIME2024/2025、MATH500）、科学问答（GPQA-Diamond）和代码执行（LiveCodeBench）。

**📈 对比分析**

与传统提示、经典RL（GRPO、Reinforce++）、剪辑优化RL（DAPO）以及多种Agentic RL（ToRL、ARPO、AEPO、CIR、GIGPO、DemyAgent）对比，ELPO在Qwen2.5-7B和Qwen3-4B上分别提升平均准确率约2.2%和1.0%，在Pass@K、Major@K、工具调用效率等指标上均优于所有基线。

**⚠️ 局限性**

局限性包括：①依赖固定采样预算和当前策略，二分定位可能受成功率、噪声影响；②仅在相对确定性工具环境验证，未评估开放域搜索或交互式工具的鲁棒性；③目前仅关注单一最早错误步骤，无法完整处理多错误或渐进式失误。

---

## 270. MieDB-100k: A Comprehensive Dataset for Medical Image Editing

**arXiv ID:** 2602.09587 | [PDF](https://arxiv.org/pdf/2602.09587v1)

**作者:** Yongfan Lai `[一作]` (State Key Laboratory of General Artificial Intelligence), Shenda Hong `[通讯]` (National Institute of Health Data Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文构建了一个包含10万条编辑三元组的医疗图像编辑数据集MieDB-100k，并在此数据集上训练模型，展示了其在三类编辑任务（感知、修改、转换）中的优异表现。

**💡 创新点**

创新点在于：①把医疗图像理解和生成统一为编辑范式；②设计了结合专家模型与规则合成的可扩展数据构建流程；③提供了面向三类任务的评估指标和基准。

**🔧 技术方法**

主要技术包括基于FLUX.1-Fill-dev的模态专属专家模型、Qwen3-VL-32B-Instruct进行判别筛选、OmniGen2的Diffusion Transformer微调、以及基于VLM的评分规则评估。

**📊 数据集**

使用的数据集是从多种公开医学影像库（共10种模态）采集并合成的，最终得到的MieDB-100k涵盖感知、修改、转换三类任务共10万条样本。

**📈 对比分析**

实验通过与多款开源（SDXL-turbo、Bagel、OmniGen2等）和闭源（Nano Banana Pro、GPT-Image-1、Imagen4）模型对比，结果显示在感知准确率、PSNR/SSIM及修改任务的Rubric Score上，微调后模型均显著优于对照组，尤其在感知任务上提高约80%。

**⚠️ 局限性**

局限性包括：①对少数模态和罕见病变的覆盖仍不足；②依赖人工审核的流程虽提升质量但限制规模扩展；③评估指标对修改任务仍主要依赖VLM打分，可能存在主观性。

---

## 271. Sample-Efficient Real-World Dexterous Policy Fine-Tuning via Action-Chunked Critics and Normalizing Flows

**arXiv ID:** 2602.09580 | [PDF](https://arxiv.org/pdf/2602.09580v1)

**作者:** Chenyu Yang `[一作]` (ETH Zurich), Robert K. Katzschmann `[通讯]` (ETH Zurich)

**通讯引用:** 5177 | [OpenAlex ID](https://openalex.org/A5050915314)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用正则化的正态流策略和动作块（action-chunk）评论家（critic）实现的、对真实机器人交互预算极低的样本高效离线-在线强化学习微调框架。

**💡 创新点**

创新点：① 在多模态动作分布下采用可精确计算似然的正态流（normalizing flow）策略，允许基于似然的保守更新；② 设计与动作块执行对齐的评论家，实现更好的信用分配和方差削减；③ 通过离线微调与在线微调的四阶段训练流程，实现从少量演示数据到高效真实机器人执行的端到端迁移。

**🔧 技术方法**

核心技术包括：正态流策略（Conditional Normalizing Flow）、Transformer‑based Q‑critic（HL‑Gaussian 分布），离线 RL（多步 TD+Bootstrapping），基于似然正则化的策略更新（类似 TD3+BC），动作块执行（H 步长）以及多任务/多阶段训练管线。

**📊 数据集**

使用的数据集：① 121 条人类远程操作演示（其中 71 条成功）用于剪刀抓取与剪切任务；② 机器人在线收集的交互数据；③ RoboMimic 的混合人类演示（Lift、Can、Square）用于离线 RL 验证；④ 在 Cube 旋转任务中使用模拟（IsaacLab）训练的教师策略再通过离线蒸馏得到初始策略。

**📈 对比分析**

与基线比较：Flow‑Matching、ACT、单纯的 NF 模式（仅 IL）、在线 RL 数据增强等。结果显示：① 在剪刀任务中，离线微调将抓取成功率从 0.8 提升至 0.8+，剪切成功率从 0 提升至 0.7；② 在 Cube 旋转任务中，离线微调后 RPM 由 0 提升至 6.25，累计旋转达到 1.01 周；③ 与基线相比，SOFT‑FLOW 在样本效率和最终性能上均优于单纯的 IL、Flow‑Matching、ACT 等。

**⚠️ 局限性**

局限性：① 受限于真实机器人交互成本，难以在多任务或持续学习场景下大规模扩展；② 依赖稀疏手工标注奖励，奖励设计对学习效果影响大；③ 正态流模型相对传统 Gaussian 策略增加了架构与训练复杂度，计算开销和工程成本提升。

---

## 272. Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling of LLMs

**arXiv ID:** 2602.09574 | [PDF](https://arxiv.org/pdf/2602.09574v1)

**作者:** Sora Miyamoto `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (NII)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于预算的树搜索解码方法——Budget‑Guided MCTS（BG‑MCTS），在固定输出 token 预算下动态调整搜索策略以提高大语言模型的回答质量。

**💡 创新点**

创新点在于将剩余预算比值 ρ 作为决策依据，既在节点选择上通过预算调节 PUCT 探索与利用，又在树扩展上通过预算引导的宽度调节实现宽→深的搜索转变，从而避免传统预算无关策略导致的过早分支或浪费。

**🔧 技术方法**

技术实现包括：预算感知的 PUCT 评分（BG‑PUCT）、预算条件下的值修正与完成偏差、预算引导的虚拟生成子节点（widening）、以及整体的 MCTS 循环；使用 GenPRM‑7B 作为奖励模型。

**📊 数据集**

使用了两个公开的 10B 以下开源模型（Llama‑3.1‑8B‑Instruct 与 Qwen‑2.5‑7B‑Instruct）在两大数学推理基准上进行评测：MATH500 与 AIME24/25。

**📈 对比分析**

在固定 token 预算（10k、20k、30k）下，BG‑MCTS 在 12 种实验设置中获得 11 次最高准确率、1 次第二高，显著优于预算无关的 MCTS、AB‑MCTS‑M、LiteSearch 以及传统的并行/序列采样方法；在预算耗尽时的准确率提升尤为明显。

**⚠️ 局限性**

局限性包括：依赖奖励模型的稀疏评分导致搜索信号弱；在更大模型或更高预算的场景下其优势可能减弱；未考虑输入 token 或奖励计算成本的预算扩展；需要进一步研究奖励校准与多模型预算分配。

---

## 273. Development of an Energy-Efficient and Real-Time Data Movement Strategy for Next-Generation Heterogeneous Mixed-Criticality Systems

**arXiv ID:** 2602.09554 | [PDF](https://arxiv.org/pdf/2602.09554v1)

**作者:** Thomas Benz `[一作]` `[通讯]`, Thomas Benz

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出一种可模块化、可参数化的 DMA 引擎（iDMA）以及轻量级的 AXI‑REALM 互连扩展，用于在异构 SoC（尤其是自动驾驶、机器人与航空航天等 ACES 场景）中实现高效、低延迟的数据搬移与实时互连保障，并在多种硬件平台上实现、验证与性能评估。

**💡 创新点**

创新点包括：
- 统一的三层模块化架构（控制平面、数据平面、协议适配器）实现对多种工业级 on‑chip 协议的无缝支持；
- 通过在数据平面插拔可选的合法化与错误处理模块，使得 DMA 能在硬件层面完成合法化、重发、抑制错误等复杂逻辑；
- 引入 Tensor‑ND、RT‑3D 等硬件级多维/实时传输扩展，提升对 AI 推理、传感器流水线等大规模数据搬移需求的支持；
- 轻量化 AXI‑REALM 互连补丁实现对时间关键工作负载的可预测访问控制，解决在异构 MCS 中 DMA 竞争导致的实时性能下降问题。

**🔧 技术方法**

主要技术手段包括：
- SystemVerilog 参数化 RTL 设计与 Mako‑模板化脚本（Mario）实现协议多样化与快速生成；
- 基于 Ready/Valid handshake 的数据流控制与双向流加速（in‑stream 加速器）实现高带宽利用；
- 基于多级缓冲与数据流元素的解耦实现对不同 burst 长度与地址对齐的自适应；
- 在 AXI‑REALM 中实现数据流分片与“R‑AW‑coupled”模式，提供可预见的带宽调度；
- 采用综合、时序、面积模型与仿真验证对比（如 HBM、RPC‑DRAM、SRAM 等多种存储子系统）。

**📊 数据集**

主要使用的实验数据集和工作负载：
- 合成的二维/三维张量搬移（Tensor‑ND）与基于 AXI‑Stream 的高带宽数据流；
- 真实的自动驾驶感知流水线（多维 sensor 数据采集与预处理）；
- Linux‑兼容的 dma‑desc64 传输用于高性能 GPU/CPU 加速器；
- 车辆级 MCS（汽车电子控制单元）上运行的实时安全任务与非安全任务混合。

**📈 对比分析**

比较方法：
- 与多种现有 DMA 引擎（CubeDMA、RDMA、MT‑DMA、AXI‑DMA 等）在面积、时序、延迟、吞吐率上进行定量对比；
- 在不同存储系统（SRAM、RPC‑DRAM、HBM）下，测量 bus utilization 与吞吐率；
- 在汽车 MCS 平台上对比启用/不启用 AXI‑REALM 的实时任务性能（latency、吞吐、能耗）。
- 性能结果：
  * 在 32‑bit 2‑outstanding 参数下，iDMA 能在 4‑beat 传输下达到接近 100% 的 bus utilization；
  * 对深度内存（100‑cycle 延迟）支持几乎完美利用；
  * AXI‑REALM 能将实时任务延迟降低 30–50% 并提升吞吐率 10–20%，同时保持 10% 左右的额外面积与功耗。

**⚠️ 局限性**

局限性：
- 对极大尺寸的多维张量或极高频率的实时调度需求，仍需手动调整缓冲与调度策略；
- 虽然模块化设计降低了集成成本，但添加全新协议仍需实现相应的协议管理器；
- 在极低功耗嵌入式环境下，复杂的合法化与错误处理模块可能略微增加静态功耗；
- 对于单核或单协议系统，过度参数化可能导致面积与时序开销不必要的增加。

---

## 274. Comprehensive Comparison of RAG Methods Across Multi-Domain Conversational QA

**arXiv ID:** 2602.09552 | [PDF](https://arxiv.org/pdf/2602.09552v1)

**作者:** Klejda Alushi `[一作]` (Hub of Computing and Data Science), Martin Semmann `[通讯]` (Hub of Computing and Data Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对八个多轮对话QA数据集进行检索增强生成（RAG）方法的系统评测，比较不同检索与后处理技术在生成答案时的表现。

**💡 创新点**

首次在多轮对话QA场景下统一对比基础RAG与多种高级RAG技术，发现检索策略与数据集结构匹配比方法复杂度更关键。

**🔧 技术方法**

使用Llama 3 8B Instruct作为生成器，结合BM25、Hybrid BM25、Reranker、HyDE、Query Rewriting、Summarization等检索与后处理技术。

**📊 数据集**

使用来自ChatRAG‑Bench的八个子集：QuAC、SQA、QReCC、TopiOCQA、Doc2Dial、DoQA、CoQA、INSCIT。

**📈 对比分析**

在统一实验设置下通过MRR@5和F1评估，结果显示Hybrid BM25、Reranker和HyDE等稳健方法优于Vanilla RAG，性能随对话轮数变化因数据集而异。

**⚠️ 局限性**

实验受方法与数据集多样性、预处理工作量大以及大规模碎片化上下文导致检索困难的限制，影响了更深入的分析。

---

## 275. Training deep physical neural networks with local physical information bottleneck

**arXiv ID:** 2602.09569 | [PDF](https://arxiv.org/pdf/2602.09569v1)

**作者:** Hao Wang `[一作]` (Tsinghua University), Qiang Liu `[通讯]` (Tsinghua University)

**通讯引用:** 21429 | [OpenAlex ID](https://openalex.org/A5000554967)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了一种基于信息瓶颈的局部训练框架（PIB），实现了在电子忆阻器芯片和光学散射装置等多种物理计算平台上的深层物理神经网络的自监督、监督和强化学习，并支持分布式并行训练。

**💡 创新点**

创新点在于将信息瓶颈原理转化为可直接测量的矩阵互信息目标，消除了对辅助数字模型、梯度反向传播和对比测量的依赖，兼容同形和异形物理单元，并在噪声、硬件故障及分布式环境下保持高效、鲁棒的训练。

**🔧 技术方法**

采用基于矩阵互信息的训练目标、局部优化策略、光学散射与忆阻器交叉耦合的深层网络结构，并结合自监督视角、TD‑Q学习以及分布式资源并行训练技术。

**📊 数据集**

使用的主要数据集包括 MNIST、Fashion‑MNIST、MNIST‑1D（图像分类）以及 CartPole‑v1（强化学习控制）等；光学平台还使用了 8‑bit 编码的子集。

**📈 对比分析**

与传统的全局反向传播、物理感知训练（PAT）、误差传输（DFA）和对比学习（PhyLL）等方法对比，PIB 在实验和仿真中几乎逼近数字BP的性能，且在噪声鲁棒性、数据效率和 OOD 检测上优于现有方法，尤其在硬件故障恢复和分布式训练场景下表现突出。

**⚠️ 局限性**

主要局限是仍需依赖数字自微分获取梯度，导致在极大规模网络或完全物理环境下可能出现梯度估计误差；此外，局部优化可能导致与全局最优解存在一定差距，尤其在任务极其复杂时性能提升有限。

---

## 276. FLINGO -- Instilling ASP Expressiveness into Linear Integer Constraints

**arXiv ID:** 2602.09620 | [PDF](https://arxiv.org/pdf/2602.09620v1)

**作者:** Jorge Fandinno `[一作]` (University of Nebraska Omaha), Torsten Schaub `[通讯]` (University of Potsdam)

**通讯引用:** 8473 | [OpenAlex ID](https://openalex.org/A5058467603)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了 Flingo 语言与工具，能够在 CASP 约束中嵌入 ASP 的默认值、未定义、非确定性选择和聚合等高级表达式，并实现了到 Clingcon 的翻译。

**💡 创新点**

核心创新在于：①让约束变量具备未定义状态并实现“已定义”依赖；②提供严格与非严格聚合操作以兼容 ASP 聚合；③支持条件聚合项，扩展了 CASP 语义与 ASP 的一致性；④通过八步翻译流程实现与现有 CASP 求解器兼容。

**🔧 技术方法**

使用 Here‑and‑There with constraints 逻辑作为理论基础，借助 Clingo 的理论接口实现约束原子，并采用 Clingcon 作为整数约束后端；同时设计了一套八步翻译策略将 Flingo 程序转换为标准 Clingo/Clingcon 程序。

**📊 数据集**

论文未给出具体实验数据集，主要通过示例阐述功能和语义。

**📈 对比分析**

比较方法以理论与示例为主，未进行性能实验或与其它 CASP 系统的基准测试，故未给出性能评估。

**⚠️ 局限性**

局限性包括：仅支持线性整数约束；翻译过程复杂，可能导致程序规模膨胀；对非线性或更复杂约束的支持有限；缺乏实验验证其可扩展性和实际性能。

---

## 277. Tele-Omni: a Unified Multimodal Framework for Video Generation and Editing

**arXiv ID:** 2602.09609 | [PDF](https://arxiv.org/pdf/2602.09609v1)

**作者:** Jialun Liu `[一作]`, Xuelong Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Tele‑Omni统一多模态视频生成与编辑框架，可接收文本、图像、参考视频指令，支持文本‑视频、图像‑视频、首尾帧生成、上下文生成与编辑等多种任务。

**💡 创新点**

创新点在于：①将任务拆分为语义解析与视频合成两大模块，借助预训练多模态大语言模型生成结构化语义提示；②通过任务感知的数据流水线统一多任务输入；③使用轻量化适配器将语义特征映射至Diffusion Transformer，实现无任务分支的统一生成与编辑。

**🔧 技术方法**

核心技术包括预训练多模态LLM（如LLaVA/miniGPT‑4等）、Diffusion Transformer（DiT）+ VAE 编码、RoPE位置编码、适配器对齐、两阶段联合训练以及任务感知数据处理。

**📊 数据集**

使用多来源视频与图像数据，结合自动生成的编辑对（基于FLUX、Wan2.1、GPT‑4o、Grounded SAM2 等），涵盖文本‑视频、图像‑视频、首尾帧约束、上下文生成与编辑等任务的多模态数据。

**📈 对比分析**

与现有单任务模型（Video‑P2P、MagicEdit、MotionCtrl、UniVideo 等）对比，实验表明Tele‑Omni在文本‑视频、图像‑视频、首尾帧生成、上下文生成/编辑等场景下均保持高视觉质量与时序连贯，性能与或优于单任务基线。

**⚠️ 局限性**

局限性包括：对长视频/高分辨率的推理效率和稳定性待提升；生成结果受限于训练数据覆盖范围；局部编辑细节仍易出现不一致；模型规模大、推理速度慢，对工业部署构成挑战。

---

## 278. Designing a Token Economy: Incentives, Governance, and Tokenomics

**arXiv ID:** 2602.09608 | [PDF](https://arxiv.org/pdf/2602.09608v1)

**作者:** Samela Kivilo `[一作]` (Tallinn University of Technology), Luca Pennella `[通讯]` (University of Trieste)

**通讯引用:** 695 | [OpenAlex ID](https://openalex.org/A5103014446)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

**🎯 论文内容**

本文提出了Token Economy Design Method（TEDM），一种基于设计科学研究的分步骤方法，用于指导代币经济在激励、治理和代币经济学三大维度的设计。

**💡 创新点**

创新点在于将先前散见的代币经济学框架与实践案例合成为一套可操作的设计步骤，强调决策点、权衡与风险，且通过实证案例与专家访谈进行方法性验证。

**🔧 技术方法**

采用设计科学研究（DSR）方法，对文献进行定性合成，结合案例演示、半结构化专家访谈以及主题分析，对方法进行形成性评估。

**📊 数据集**

使用的数据来源包括：Currynomics生态系统的设计与实施细节、五位专家访谈记录，以及对Uniswap和Curve Finance两种运作中DEX的代币经济数据和治理指标。

**📈 对比分析**

通过在Uniswap和Curve Finance上应用TEDM对代币经济进行对比分析，展示了两者在激励机制、治理模型和代币经济学选择上的差异；评价指标主要是方法的完整性、简洁性、可理解性和可操作性，结果普遍获得正面反馈。

**⚠️ 局限性**

局限性在于评估仅为定性、形成性，样本量有限且仅覆盖单一案例，缺乏统计泛化和量化结果；方法对治理最小化或纯粹投机型代币经济的适用性不明确，需要进一步跨领域验证和数值模拟补充。

---

## 279. Detecting radar targets swarms in range profiles with a partially complex-valued neural network

**arXiv ID:** 2602.09597 | [PDF](https://arxiv.org/pdf/2602.09597v1)

**作者:** Martin Bauw `[一作]` `[通讯]` (ONERA), Martin Bauw (ONERA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了使用部分复数神经网络对雷达距离剖面进行处理，以检测目标群的存在。

**💡 创新点**

创新点在于将复数参数网络与完整距离剖面输入相结合，模拟波形失真，并直接替代匹配滤波+CA‑CFAR的传统检测链路。

**🔧 技术方法**

采用部分复数网络（modReLU 激活），MSE 损失、Adam 优化器，训练时使用不同噪声、反射系数和波形带宽的模拟数据。

**📊 数据集**

构造了基于 LFM 脉冲的模拟数据集，包括“baseline”和“enriched”两套训练/验证/测试集，涵盖单目标、多目标、波形失真、噪声水平等多种情形。

**📈 对比分析**

通过与传统匹配滤波+CA‑CFAR 的对比实验，发现当训练集加入波形失真、空白和对比剖面后，网络在高目标密度和波形失真情境下的检测概率显著优于基线；在低能量或低幅度波形时仍略逊。

**⚠️ 局限性**

局限性包括对低幅度、频率失真 echo 的泛化不足、易出现过拟合（未使用早停/Dropout），以及训练与测试集分布差异大导致的性能下降。

---

## 280. Delving into Spectral Clustering with Vision-Language Representations

**arXiv ID:** 2602.09586 | [PDF](https://arxiv.org/pdf/2602.09586v1)

**作者:** Bo Peng `[一作]` (Australian Artificial Intelligence Institute), Zhen Fang `[通讯]` (Australian Artificial Intelligence Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

结合预训练视觉‑语言模型的正向词进行神经切线核（NTK）构造，提出多模态谱聚类方法。

**💡 创新点**

通过把NTK与正向词对齐，形成视觉相似度与语义重叠的乘积亲和矩阵，显著提升块对角结构；并设计正则化亲和扩散融合多提示的亲和矩阵。

**🔧 技术方法**

采用CLIP预训练模型、神经切线核、图拉普拉斯谱聚类、正则化亲和扩散等技术。

**📊 数据集**

16个基准数据集，包括STL‑10、CIFAR‑10/20、ImageNet‑10、ImageNet‑Dogs、DTD、UCF‑101、ImageNet‑1K、ImageNet‑C/V2/S，以及5个细粒度数据集。

**📈 对比分析**

在所有数据集上与TAC、CLIP‑SC等最先进方法对比，平均提升10%以上的ACC/NMI/ARI，尤其在ImageNet‑Dogs、UCF‑101、细粒度数据上获得显著优势。

**⚠️ 局限性**

依赖CLIP预训练模型，对文本正向词筛选的质量和提示模板的选择敏感，且在极端领域迁移时仍有一定性能衰退。

---

## 281. Computational Explorations on Semifields

**arXiv ID:** 2602.09577 | [PDF](https://arxiv.org/pdf/2602.09577v1)

**作者:** Jean-Guillaume Dumas `[一作]` (Universite Grenoble Alpes), John Sheekey `[通讯]` (University College Dublin)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5038859418)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

对有限半域和有限域扩展（在特征 2 与 3 下，阶数为 32、81、243 等）中的乘法运算进行多重复杂度（张量秩）和加法复杂度研究，提出新的算法与上界，并给出多种实例的最优或近似最优直线程序。

**💡 创新点**

① 证明若干阶数的有限半域张量秩为 8、9、10、11、13，填补之前未知的缺口；② 在 81、243 元域上通过“折叠”技术找到更少乘法和加法的算法；③ 引入改进的核分解（kernel）和分组（folding）方法，显著降低加法复杂度；④ 计算并给出小参数下最短直线程序的具体下界和构造。

**🔧 技术方法**

使用张量分解与等价变换（等位同构）、直线程序搜索（随机与穷举）、线性码理论（最小距离、MDS 约束）、Griesmer 与 MDS 下界、Sylvester 约化矩阵、蒙特卡洛搜索及 Mathematica / MAGMA 等工具。

**📊 数据集**

对所有可行的 2 次、3 次、4 次、5 次多项式乘法（即对应的有限域扩展）以及已知的 32、81、243 阶半域实例进行实验；利用已公布的线性码表和半域分类表，枚举所有等价类进行搜索。

**📈 对比分析**

通过对比乘法次数 + 加法次数（总操作数）与已公开的基准算法（Karatsuba、Toom‑Cook、Montgomery、标准多项式乘法），证明本工作在大多数小域/半域上至少比现有方案节省 1–4 次乘法或 5–10 次加法；例如在 3^4 上实现 8M+22A（总 30 次）优于传统 9M+21A；在 3^5 上实现 10M+43A 优于此前 11M+44A。

**⚠️ 局限性**

① 对更高阶数或更大特征的域/半域仍无法给出最优张量秩或加法复杂度；② 部分结果依赖耗时的穷举搜索，难以扩展到更大参数；③ 对于某些半域仍存在未解决的上限/下限问题（如 3^4 是否能达 9M+20A、3^5 是否能达 10M+<43A 等）。

---

## 282. Fréchet Distance in the Imbalanced Case

**arXiv ID:** 2602.09551 | [PDF](https://arxiv.org/pdf/2602.09551v1)

**作者:** Lotte Blank `[一作]` (University of Bonn), Lotte Blank `[通讯]` (University of Bonn)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5113204598)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文研究两条多边形曲线之间离散与连续Fréchet距离的近似计算与上界，提出在1D情况下的2-近似算法，并给出在任意维度下的(3+ε)-近似算法；同时在1D、2D欧氏与L∞空间上以Orthogonal Vectors假设（OVH）为基础证明了更高近似因子（2-、1+√2-、3-）的下界，证明这些近似因子在时间复杂度O((nm)^{1-δ})内无法被突破。

**💡 创新点**

创新点包括：
1) 在1D离散Fréchet距离上取得2-近似的最优下界与时间复杂度；
2) 在2D欧氏与L∞空间上给出更强的下界（1+√2-、3-），扩展了已知的1.001下界；
3) 提供了一种压缩简化技术（compressed simplification）使得1D离散Fréchet查询可在O(m^2 log m)时间内完成；
4) 在任意维度下提出(3+ε)-近似算法，时间仅为O((n+m^2) log n)，几乎匹配下界。

**🔧 技术方法**

技术手段包括：
- 基于Orthogonal Vectors问题的归约构造，利用离散/连续Fréchet距离的自由空间矩阵特性；
- 1D曲线的最优m简化与压缩简化，结合Frederickson–Johnson的二分搜索实现快速预处理；
- 对自由空间矩阵的迭代可达性维护，利用每段重复点的结构实现O(m^2)判定；
- 构造P′曲线（长度≤2m）逼近P，随后通过两段匹配判定连续/离散Fréchet距离的(3+ε)近似；
- 对L_p范数的等价性与Colombe–Fox的对数放大技术，用以从判定得到优化近似。

**📊 数据集**

本文为理论论文，无实验数据集，全部结论均来自理论分析与归约构造。

**📈 对比分析**

与现有工作比较：
- 对1D离散Fréchet距离的2-近似与上界O(n log n + m^2 log m)与已知的O(nm)算法相比，时间大幅提升；
- 对2D欧氏/​L∞空间的下界提升至1+√2-和3-，比之前的1.001下界更强；
- (3+ε)-近似算法的时间仅为O((n+m^2) log n)，几乎与下界一致，表明该算法在理论上是最优的。

**⚠️ 局限性**

局限性：
- 1D离散Fréchet距离的2-近似并非最优近似因子（仍可能存在更好的近似因子）；
- 在高维空间下的(3+ε)-近似仍未突破3因子；
- 论文仅给出理论复杂度和下界，缺乏实验验证与实际应用评估；
- 归约构造依赖OVH或SETH假设，若这些假设被破坏，结论将失效。

---

## 283. SWE-Bench Mobile: Can Large Language Model Agents Develop Industry-Level Mobile Applications?

**arXiv ID:** 2602.09540 | [PDF](https://arxiv.org/pdf/2602.09540v1)

**作者:** Muxin Tian `[一作]` (University of Toronto), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7875 | [OpenAlex ID](https://openalex.org/A5003491365)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了SWE‑Bench Mobile基准，评估大语言模型代理在真实工业移动应用开发中的表现。

**💡 创新点**

首创将PRD、Figma设计、海量Swift/Objective‑C代码库与完整测试套件结合的多模态工业级基准。

**🔧 技术方法**

采用大型语言模型代理（Cursor、Codex、Claude Code、OpenCode）和多种模型（Claude Opus、GLM、GPT、Gemini）以及差分式评估流水线。

**📊 数据集**

基准任务来源于小红书iOS生产代码的50条真实功能需求，包含约5GB代码、449个测试用例。

**📈 对比分析**

通过任务成功率和测试通过率衡量，最佳配置仅12%任务通过率，显示现有代理与工业要求相距甚远。

**⚠️ 局限性**

受限于单一iOS平台、文本差分评估不覆盖运行时行为、任务规模有限，难以全面覆盖所有移动开发挑战。

---

## 284. AUHead: Realistic Emotional Talking Head Generation via Action Units Control

**arXiv ID:** 2602.09534 | [PDF](https://arxiv.org/pdf/2602.09534v1)

**作者:** Jiayi Lyu `[一作]` (University of the Chinese Academy of Sciences), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60398 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种两阶段基于AU的语音驱动对话头生成框架AUHead，先用ALM提取细粒度情绪AU序列，再用AU驱动扩散模型生成真实情感对话头视频。

**💡 创新点**

创新点是将AU作为中间可解释控制空间，结合ALM的情绪理解与稀疏token化，采用“emotion‑then‑AU”CoT策略，并引入AU‑vision交叉注意力和AU解耦引导。

**🔧 技术方法**

主要技术包括音频语言模型微调、AU稀疏token化与CoT生成、AU到2D表示映射、上下文感知AU嵌入、跨模态扩散模型与解耦引导。

**📊 数据集**

采用MEAD和CREMA情绪对话头数据集进行训练与评测。

**📈 对比分析**

与SOTA方法（如Hallov1/V2、MEMO、EAMM等）比较，AUHead在PSNR/SSIM/FID、M‑LMD/F‑LMD、情绪准确率等指标上均优于对手，且在用户评测中获得最高偏好。

**⚠️ 局限性**

限制包括对AU解码精度的依赖、对高帧率视频的可扩展性不足、对头部姿态和背景多样性的泛化能力待提升。

---

## 285. RAD: Retrieval-Augmented Monocular Metric Depth Estimation for Underrepresented Classes

**arXiv ID:** 2602.09532 | [PDF](https://arxiv.org/pdf/2602.09532v1)

**作者:** Michael Baltaxe `[一作]` (General Motors), Sagie Benaim `[通讯]` (The Hebrew University of Jerusalem)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种检索增强的单目深度估计框架 RAD，利用不确定性引导检索相似场景作为几何辅助，提升稀有类别的深度精度。

**💡 创新点**

创新点包括：①基于像素级不确定性与分割结果的检索策略；②双流网络与匹配跨注意机制，仅在可靠对应点传递几何信息；③结合 3D 数据增强与检索训练，实现对稀有类别的有效泛化。

**🔧 技术方法**

技术主要包括：不确定性估计（多次扰动后标准差）、SAM2 分割、DINO v2 特征检索、LightGlue 点匹配、Vision Transformer（DepthAnything v2）双流编码、匹配跨注意（Matched Cross‑Attention）以及 3D 渲染生成上下文图像。

**📊 数据集**

使用公开深度数据集 NYU Depth v2、KITTI（Eigen split）和 Cityscapes，并通过语义分割标注挑选低频出现的稀有类别进行评测。

**📈 对比分析**

与多种 fine‑tuning 与 zero‑shot 先行方法（如 DepthAnything v2、UniDepth v2、Metric3D v2、ZoeDepth 等）对比，RAD 在稀有类别上相对绝对误差提升 29.2%、13.3% 和 7.2%，在所有类别上保持或略优于现有最优方法，展示显著的性能提升。

**⚠️ 局限性**

局限性包括：检索过程依赖检索库的质量；对不确定性阈值与检索数量敏感；在极端稀有或无对应检索样本时性能下降；以及额外的检索与匹配开销。

---

## 286. DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment

**arXiv ID:** 2602.09531 | [PDF](https://arxiv.org/pdf/2602.09531v1)

**作者:** Bohan Fu `[一作]` (Beijing Institute of Technology), Runze Hu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 588 | [OpenAlex ID](https://openalex.org/A5011299161)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DR.Experts 框架，利用失真先验与差分注意力机制和动态专家加权实现无参考图像质量评估

**💡 创新点**

创新点在于将基于 DA‑CLIP 的失真先验与 Distortion‑Saliency Differential Module 结合，精细分离失真特征；随后通过 Dynamic Distortion Weighting Module 对不同失真类型进行自适应加权，显著提升对细微失真的感知与评价

**🔧 技术方法**

采用 DA‑CLIP 视觉‑语言模型获取失真先验，ViT 作为图像编码器，差分注意力机制（Differential Refinement Attention）提取失真特征，Mixture‑of‑Experts 结构实现动态加权，最后使用线性回归头和 Smooth L1 损失进行训练

**📊 数据集**

在五个在野 BIQA 基准上进行评估：KonIQ‑10k、LIVE‑FB、LIVE‑C、LIVEC 以及 BID 数据集

**📈 对比分析**

与 14 种先进方法（包括 DB‑CNN、HyperIQA、MUSIQ、TReS、DEIQT、QPT、LODA、LQMamba 等）进行横向比较；DR.Experts 在所有数据集上取得最高 SRCC/PLCC 评分，提升幅度可达 0.75‑0.95 分，且在跨数据集泛化和低样本量实验中同样表现优异

**⚠️ 局限性**

局限性：模型高度依赖预训练的 DA‑CLIP 失真先验，若遇到模型未见过的失真类型可能效果下降；此外 ViT 语义特征的噪声仍会在一定程度上影响最终质量评估

---

## 287. Learning to Discover Iterative Spectral Algorithms

**arXiv ID:** 2602.09530 | [PDF](https://arxiv.org/pdf/2602.09530v1)

**作者:** Zihang Liu `[一作]` (University of California), Michael W. Mahoney `[通讯]` (University of California)

**通讯引用:** 24452 | [OpenAlex ID](https://openalex.org/A5033006662)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用神经网络框架AutoSpec学习将谱探测信息映射为迭代多项式系数，从而自动生成适用于数值线性代数和优化的谱自适应迭代算法；

**💡 创新点**

提出连续搜索空间的可微分递推结构，结合自监督训练和从小型合成谱到大规模实际矩阵的迁移学习，生成具有Chebyshev等最优特性的高效多项式预条件器；

**🔧 技术方法**

核心技术包括：谱探测嵌入、全连接网络生成递推系数、可执行的线性递推实现、基于谱特征的自监督损失、对角化矩阵训练与数值稳定化归一化；

**📊 数据集**

训练数据为人工生成的低维对角矩阵谱；测试数据包括SuiteSparse稀疏矩阵集（电子结构、有限元、机械、回路等）以及DNA相似性Gram矩阵；

**📈 对比分析**

与传统Chebyshev多项式、幂预条件器、Neumann级数等基线比较，实验表明在迭代次数和误差幅度上实现了数阶至十数阶的加速与精度提升；

**⚠️ 局限性**

局限性主要包括：需先执行谱探测（需要额外Lanczos/Arnoldi迭代），对极端谱分布或非对称矩阵的泛化仍有限；模型对探测质量敏感，且目前仅支持线性递推，未充分挖掘更高阶或非线性更新空间。

---

## 288. An open-source implementation of a closed-loop electrocorticographic Brain-Computer Interface using Micromed, FieldTrip, and PsychoPy

**arXiv ID:** 2602.09735 | [PDF](https://arxiv.org/pdf/2602.09735v1)

**作者:** Bob Van Dyck `[一作]` (KU Leuven), Marc M. Van Hulle `[通讯]` (KU Leuven)

**通讯引用:** 5801 | [OpenAlex ID](https://openalex.org/A5022490304)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文实现了基于临床 Epilepsy 监测系统（Micromed）进行的闭环 ECoG BCI 原型，提供完整的数据采集、实时信号处理与闭环反馈的工作流；

**💡 创新点**

创新点在于构建了三套开源 Python 库（psychopylib、pymarkerlib、pyfieldtriplib）以及系统架构描述，显著降低了临床环境下实施闭环 ECoG BCI 的技术门槛；

**🔧 技术方法**

采用了 Python 生态（PsychoPy、FieldTrip、LSL、MNE‑Python 等）、MATLAB FieldTrip 缓冲、TCP/IP 无线传输以及多线程处理框架；

**📊 数据集**

使用的“数据集”是来自 Epilepsy 监测的实时 ECoG TRC 文件与同步事件标记，属于临床现场采集的非公开数据；

**📈 对比分析**

本文未给出与其他方法的对比实验或性能指标，只展示了功能性示例（如 epoch 处理、特征提取等），因此缺乏量化的性能评估；

**⚠️ 局限性**

主要局限包括对 MATLAB FieldTrip 缓冲的依赖、对特定 Micromed 设备的绑定、未在大规模数据上验证系统鲁棒性与实时延迟，以及缺少系统性能与基准对比。

---

## 289. Allure of Craquelure: A Variational-Generative Approach to Crack Detection in Paintings

**arXiv ID:** 2602.09730 | [PDF](https://arxiv.org/pdf/2602.09730v1)

**作者:** Laura Paul `[一作]` (LMU Munich), Tim Roith `[通讯]` (Helmholtz Imaging)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将裂纹检测建模为逆问题的变分方法，联合使用深度生成模型和Mumford–Shah型正则化，直接得到像素级裂纹图。

**💡 创新点**

创新点在于将生成式先验和学习的裂纹先验嵌入逆问题框架，避免传统预处理与后处理步骤，实现端到端的无预处理检测。

**🔧 技术方法**

使用 VQGAN 生成器做背景先验，DeepCrack 网络做裂纹先验，AT 近似 Mumford–Shah 正则化，以及 Adam 优化器求解。

**📊 数据集**

数据集包括约8100张无裂纹绘画补丁及利用车道裂纹数据生成的约1700张合成裂纹图像；在真实绘画上也做验证。

**📈 对比分析**

与传统形态学+机器学习方法对比，在合成数据上 F1 分数达 0.97；在真实图像上能够捕捉大部分裂纹，误检率低。

**⚠️ 局限性**

局限在于合成数据可能不完全代表真实老化裂纹，某些细微裂纹漏检，复杂纹理背景可能产生误检；且超参数需在合成上调优。

---

## 290. ExO-PPO: an Extended Off-policy Proximal Policy Optimization Algorithm

**arXiv ID:** 2602.09726 | [PDF](https://arxiv.org/pdf/2602.09726v1)

**作者:** Hanyong Wang `[一作]` (Organization), Menglong Yang `[通讯]` (Organization)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种扩展的离线近端策略优化（ExO-PPO）算法，将传统 PPO 的样本利用与离线重放结合，提供更高效、稳定的训练；

**💡 创新点**

创新点包括：① 通过扩展的离线改进下界支持使用最近 M 策略的经验回放；② 设计了分段指数衰减的比例函数（Extended Ratio Objective），在离线/混合策略下保持梯度平滑并抑制离线漂移；③ 将上述两者融合，兼顾样本效率与稳定性；

**🔧 技术方法**

使用技术包括：PPO 基础框架、经验回放、分段指数衰减比例函数、KL 散度惩罚、GAE 价值估计、CNN/高斯策略、环境并行、离线数据集加载等；

**📊 数据集**

实验数据集涵盖 Atari 游戏（Pong-v5、Breakout-v5 等）、MuJoCo 连续控制任务（Ant-v4、HalfCheetah-v4、Walker2d-v4）以及 Minari 离线数据集；

**📈 对比分析**

通过与 PPO、ESPPO、P3O-Scopic、SAC、TD3、GePPO、Off-PPO、Behavior PPO 等基线在相同网络、学习率和并行环境设置下进行对比；结果表明 ExO-PPO 在大多数任务中收敛更快、最终性能更好，特别是在离线任务上表现优于或等同于现有离线算法；

**⚠️ 局限性**

限制包括：对超参数（α、M、KL 权重等）敏感，需要手动调优；在极端分布漂移或连续任务中仍存在稳定性挑战；理论上仍需进一步研究分布偏移、梯度传播效率等问题。

---

## 291. Targum -- A Multilingual New Testament Translation Corpus

**arXiv ID:** 2602.09724 | [PDF](https://arxiv.org/pdf/2602.09724v1)

**作者:** Maciej Rapacz `[一作]`, Aleksander Smywiński-Pohl `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

建立了一个多语言的新约圣经翻译语料库，包含657个译本（352个独特版本），重点在五种语言（英语、法语、意大利语、波兰语、西班牙语）的垂直深度，并对每个译本手工标注了标准化元数据；

**💡 创新点**

创新点在于：①提供极致深度的翻译资源，突破传统语料库的语言宽度局限；②采用手工标准化和交叉源验证，保证元数据准确性；③设计“可重定义唯一性”的元数据结构，允许研究者按需去重或保留近似复制；④整合12个在线圣经库与已有语料，形成首个适用于多层次分析的资源。

**🔧 技术方法**

技术方法包括：网页抓取与索引、定制解析器处理多样的章节标注、手工元数据映射与YAML化、交叉源文本相似度校验（Levenshtein）、语义相似度评估（使用Qwen3-Embedding-0.6B模型）以及统计与可视化分析。

**📊 数据集**

数据集来源为12个公开圣经库（如Bible Gateway、Bible Hub、bible.com等）以及eBible数字语料库，共计收集了657个新约译本，覆盖英语、法语、意大利语、波兰语、西班牙语。

**📈 对比分析**

比较方法：对同一章节进行词汇相似度（Levenshtein）和语义相似度（词向量余弦相似度）计算，构建相似度矩阵。实验结果显示：大多数译本的语义相似度>0.8，能够有效识别近似复制与动态译本簇，支持微观与宏观层面的定量研究。

**⚠️ 局限性**

局限性：①受限于网络公开数字化译本，存在“数字化偏差”，无法完整覆盖所有历史译本；②手工标注过程虽严格，但仍可能出现错误，尤其是现代数字原版的修订年份难以确定。

---

## 292. From Lightweight CNNs to SpikeNets: Benchmarking Accuracy-Energy Tradeoffs with Pruned Spiking SqueezeNet

**arXiv ID:** 2602.09717 | [PDF](https://arxiv.org/pdf/2602.09717v1)

**作者:** Radib Bin Kabir `[一作]` (Islamic University of Technology), Md Hasanul Kabir `[通讯]` (Islamic University of Technology)

**通讯引用:** 2965 | [OpenAlex ID](https://openalex.org/A5071274329)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对轻量级 CNN（ShuffleNet、MnasNet、MixNet、SqueezeNet）做了稀疏化后的 SNN 转换，并在 CIFAR‑10、CIFAR‑100 与 TinyImageNet 上系统评估其准确率、能耗与计算量。

**💡 创新点**

创新点在于首次提供完整的轻量级 CNN‑to‑SNN 基准，并提出针对 SNN SqueezeNet 的结构化剪枝方法，使精度提升 6–7% 且能耗下降 88% 以上，接近 CNN 级别。

**🔧 技术方法**

技术包括 LIF 神经元、替代梯度下降、基于时间步的训练、结构化剪枝与能耗估算（AC+MAC 计算）。

**📊 数据集**

使用的公开数据集为 CIFAR‑10、CIFAR‑100 和 TinyImageNet（200 类），统一训练设置为 120 轮、Adam、学习率 0.001。

**📈 对比分析**

通过与对应 CNN 的准确率与能耗比较，SNN 在保持相近精度的同时实现 2.5–15.7 倍的能耗提升；剪枝后 SNN SqueezeNet‑P 几乎匹配 CNN 精度，仅耗 5.6 倍能耗。

**⚠️ 局限性**

局限在于：仍存在与 CNN 的精度差距；仅关注分类任务，未评估在真实神经形态硬件上的实际能耗；剪枝策略手动选择，缺乏自动化或 NAS 支持。

---

## 293. TraceMem: Weaving Narrative Memory Schemata from User Conversational Traces

**arXiv ID:** 2602.09712 | [PDF](https://arxiv.org/pdf/2602.09712v1)

**作者:** Yiming Shu `[一作]` (University of Hong Kong), Chen Sun `[通讯]` (University of Hong Kong)

**通讯引用:** 249410 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种三阶段的认知启发式记忆系统 TraceMem，能够把用户的对话轨迹分割成事件、提炼为突触记忆、再聚类为系统级叙事线程，并生成结构化的记忆卡片，同时实现代理式检索以支持长时对话智能。

**💡 创新点**

创新点包括：① 将演绎推理+XML提示用于实时事件分割；② 通过突触和系统级巩固形成层级叙事记忆；③ 两级（PCA→UMAP→HDBSCAN→KNN）聚类生成主题+线程结构；④ 结合代理式检索同时召回情节记忆与叙事线程，实现来源归因。

**🔧 技术方法**

使用技术：演绎推理 + XML 提示的主题分割；摘要与经验提炼（突触巩固）；PCA-UMAP-HDBSCAN-KNN 两级层级聚类（系统巩固）；向量检索（ChromaDB）与代理式检索；LLM（GPT‑4o‑mini / GPT‑4.1‑mini）作为模型与评判者。

**📊 数据集**

数据集：LoCoMo 基准（10 条长对话，约 600 轮，1,540 题目，包含单跳、多跳、时序、开放域四类推理）。

**📈 对比分析**

方法评估：在 LoCoMo 上与 FullText、NaiveRAG、A‑Mem、LightMem、Nemori 等基线比较，采用 GPT‑4o‑mini / GPT‑4.1‑mini 作为判定者，计算准确率。TraceMem 在所有任务上均超越基线，整体准确率约 0.90，尤其在 MultiHop 与 Temporal 推理中提升 16–20% 以上。

**⚠️ 局限性**

局限性：缺乏记忆重写与主动遗忘机制；需要进一步研究何时调用或抑制记忆以避免干扰；目前仅在 LoCoMo 小规模实验验证，未评估在更大规模或实时多用户场景下的可扩展性。

---

## 294. Resilient Class-Incremental Learning: on the Interplay of Drifting, Unlabelled and Imbalanced Data Streams

**arXiv ID:** 2602.09681 | [PDF](https://arxiv.org/pdf/2602.09681v1)

**作者:** Jin Li `[一作]` (University of Cyprus), Marios Polycarpou `[通讯]` (University of Cyprus)

**通讯引用:** 22286 | [OpenAlex ID](https://openalex.org/A5089297277)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对非平稳、标签缺失、类别不平衡与增量新类别的流式学习框架SCIL，能够在一比一在线学习环境中实现新类别检测、持续适应与知识保留。

**💡 创新点**

创新点在于将自编码器与多层感知机联合训练，利用重构误差阈值实现无监督新类别检测，并引入基于密度、尺度与距离的纠错机制与SMOTE过采样相结合的增量更新策略，以同时应对概念漂移、类别不平衡与新类别出现。

**🔧 技术方法**

主要技术包括：自编码器（AE）提取低维表示、全连接多层感知机（MLP）进行多类别分类、重构误差阈值检测新类、密度与尺度估计的核心子集筛选、SMOTE合成少数类、动态队列记忆与增量训练。

**📊 数据集**

实验使用了八个数据集，涵盖合成（Sea、Vib、Blob）与真实（WDN、MNIST、KDD99、Forest、Sensorless、Shuttle）场景，所有数据均包含多种概念漂移和类别不平衡配置。

**📈 对比分析**

与十余种先进方法（如iForest+MULTI、LOF+MULTI、MINAS、CPOCEDS、SNDProb、KNNENS、OCGCD、UDOR）在EN_Accuracy与G‑mean两种评估指标上进行对比，SCIL在所有数据集上均名列前茅，尤其在高维和频繁漂移场景下显著优于基线与现有技术。

**⚠️ 局限性**

局限性包括：仅针对单一主类别与多重少数类的场景；新类别阈值和纠错机制需要手动调参；在极端不平衡或漂移频繁的情形下仍可能出现暂时性能下降；缺乏对多标签或多任务流式学习的扩展。

---

## 295. Differentiable Modeling for Low-Inertia Grids: Benchmarking PINNs, NODEs, and DP for Identification and Control of SMIB System

**arXiv ID:** 2602.09667 | [PDF](https://arxiv.org/pdf/2602.09667v1)

**作者:** Shinhoo Kang `[一作]` (Korea University), Sehyun Yun `[通讯]` (Hyundai Motor Company)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对低惯性电网中的单机无穷节点(SMIB)系统进行建模、参数辨识与控制，比较了Physics‑Informed Neural Network（PINN）、Neural Ordinary Differential Equation（NODE）与Differentiable Programming（DP）三种可微模型；

**💡 创新点**

创新点在于首次将DP作为硬约束物理模型框架，显著提升参数辨识速度与控制稳健性，并展示NODE在无已知方程时可作为有效的控制代理；

**🔧 技术方法**

采用PINN、NODE、DP三种神经网络方法，并结合自动微分、ODE求解器以及LQR控制设计；

**📊 数据集**

使用由仿真生成的SMIB轨迹数据集，包括稳定与振荡两种工况，并在不同噪声水平下进行训练与测试；

**📈 对比分析**

通过比较轨迹预测误差、参数估计误差和闭环LQR控制性能，发现DP在参数辨识收敛和控制稳定性上优于PINN，NODE在外推预测和缺失方程时表现最佳；

**⚠️ 局限性**

局限性包括仅在单机模型上验证，DP需要已知动力学方程，PINN在外推时精度不足，缺乏真实电网数据验证，并且对多机复杂网络的可扩展性尚未探讨。

---

## 296. ClinAlign: Scaling Healthcare Alignment from Clinician Preference

**arXiv ID:** 2602.09653 | [PDF](https://arxiv.org/pdf/2602.09653v1)

**作者:** Shiwei Lyu `[一作]` (Ant Group), Yue Shen `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建医学领域的专家校验偏好数据集HealthRubrics，并从中提炼可复用的HealthPrinciples，用于对大语言模型进行细粒度临床对齐；

**💡 创新点**

①将真实医疗问答与多模型回答的用户偏好，经过三轮医生校正，生成高质量的偏好标注；②将反复出现的评判模式压缩为119条可复用原则，既实现了可扩展的监督，又提供了推理时的自修工具；

**🔧 技术方法**

使用Prompt-Driven rubric生成、两轮医生复核的工作流；利用GRPO（基于Qwen3-32B评估器）进行强化学习；将原则映射到问题生成可检验的rubric；推理时通过工具链调用原则生成rubric做自修；

**📊 数据集**

7,034条医生审核的偏好实例（HealthRubrics）；16,872条基于UltraMedical-Preference的追加医疗问题，结合HealthPrinciples生成的rubric；Benchmark数据包括HealthBench、LLMEval‑Med、Arena‑Hard‑v2；

**📈 对比分析**

在HealthBench‑Hard上，采用30B参数模型但仅激活3B参数，得分33.4%，超过DeepSeek‑R1、o3等更大模型；在HealthBench总体、LLMEval‑Med各子任务、Arena‑Hard‑v2上均实现与或优于专门化及闭源大模型的表现；

**⚠️ 局限性**

①对多步推理的提升有限，主要提升的是帮助性与安全性；②推理时工具调用的增益随次数递减，易出现饱和，需更智能的工具使用策略。

---

## 297. Time2General: Learning Spatiotemporal Invariant Representations for Domain-Generalization Video Semantic Segmentation

**arXiv ID:** 2602.09648 | [PDF](https://arxiv.org/pdf/2602.09648v1)

**作者:** Siyu Chen `[一作]` (Jimei University), Jinhe Su `[通讯]` (Jimei University)

**通讯引用:** 364 | [OpenAlex ID](https://openalex.org/A5049203004)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在单源监督下，本文提出 Time2General 框架，实现对未见域视频语义分割的跨域泛化。

**💡 创新点**

核心创新在于引入稳定查询（Stability Queries）作为跨域可迁移的语义锚点，利用无对应关系的时空记忆解码器进行多帧上下文聚合，并通过多步差异的 Masked Temporal Consistency 损失抑制帧间抖动。

**🔧 技术方法**

技术上使用冻结的 DINOv2 主干，轻量化查询模块，空间-时间记忆解码器以及随机采样步长和遮挡一致性损失实现时空一致性。

**📊 数据集**

在五个驾驶视频数据集（KITTI‑360、ApolloScape、CamVid、Cityscapes‑sequence 与 Cityscapes‑sequence‑Corrupted）上进行实验。

**📈 对比分析**

与 REIN、FADA、DepthForge、SSP、TV3S 等基线相比，Time2General 在 mIoU 上提升 4–16% 以上、mVC 在 90% 以上，且推理速度可达 18 FPS。

**⚠️ 局限性**

局限性在于仅针对单源训练场景，仍需依赖预训练主干，对极端低照度或大遮挡场景的鲁棒性不足，并且对新域的适应仍需更强的跨域表示学习。

---

## 298. LLM-FS: Zero-Shot Feature Selection for Effective and Interpretable Malware Detection

**arXiv ID:** 2602.09634 | [PDF](https://arxiv.org/pdf/2602.09634v1)

**作者:** Naveen Gill `[一作]` (National Institute of Technology Calicut), Madhu Kumar S D `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于大型语言模型的零样本特征选择框架（LLM‑FS），用于恶意软件检测。

**💡 创新点**

将LLM以零样本方式根据特征统计描述生成重要性分数，实现知识驱动、可解释的特征筛选。

**🔧 技术方法**

结合结构化提示、确定性解码的LLM推理（GPT‑4、Gemini等）、传统过滤/嵌入方法及多分类器（RF、ET、MLP、KNN）评估。

**📊 数据集**

在融合EMBED和BODMAS的EMBOD恶意软件数据集上实验，样本约9.34k、维度2381。

**📈 对比分析**

与多种传统FS方法在同一数据集、相同特征数（341）下用多指标（Accuracy、F1、AUC、MCC）比较，LLM‑FS的表现与最强传统方法相当甚至略优，且训练时长与传统法相近。

**⚠️ 局限性**

需要耗时的每特征LLM查询、对特征命名的依赖、模型更新导致稳定性波动、以及在极高维或非语义特征时效果受限。

---

## 299. Revealing the Challenges of Attention-FFN Disaggregation for Modern MoE Models and Hardware Systems

**arXiv ID:** 2602.09721 | [PDF](https://arxiv.org/pdf/2602.09721v1)

**作者:** Guowei Liu `[一作]` (Baidu Inc), Yanpeng Wang `[通讯]` (Baidu Inc)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文系统评估了 Attention‑FFN Disaggregation (AFD) 在大规模稀疏专家模型（MoE）推理中的性能，并通过延伸屋顶线模型到通信层面，建立预算驱动的硬件 FLOPS 利用率 (HFU) 上限分析框架，揭示了标准集群上的“死区”与离散节点级扩展导致的负载失衡惩罚；

**💡 创新点**

创新点在于（1）将屋顶线模型与通信带宽、算术强度及硬件 FLOPS 利用率相结合，量化 AFD 的理论 HFU 上限；（2）系统分析了 AFD 与传统大规模专家并行 (EP) 的负载失衡与性能瓶颈；（3）提出了在 Superpod‑class 体系结构与粗粒度、低稀疏度模型下 AFD 可取代 EP 的条件；

**🔧 技术方法**

主要技术包括：延伸屋顶线模型、算术强度计算、通信带宽与规模扩展分析、负载失衡参数化、微批次重叠（3BO）与 FP8 GEMM 评估、以及基于预算的 HFU 与 OFU 指标；

**📊 数据集**

实验以 DeepSeek‑V3 (671B) 与 Kimi‑K2 (1T) 为代表模型，配合 NVIDIA H800/H20/H100/H200 及 Superpod GB200/GB300 等硬件平台进行理论与仿真验证；

**📈 对比分析**

比较方法：通过计算不同平台/模型组合下的 HFU 上限与实际 OFU，评估 AFD 与 EP 的吞吐率与资源利用率；结果表明在非 Superpod 系统上 AFD 受限于网络带宽导致 HFU 低于 EP，而在 Superpod 或粗粒度专家模型下 AFD 能实现更高的 HFU 与吞吐；

**⚠️ 局限性**

局限性包括：仅对标准集群与特定的 DP/EP 并行方式进行建模，忽略了多阶段延迟、GPU 泡沫与图执行开销；假设的 t_B 与 t_g 可能过于乐观；且离散节点扩展导致的负载失衡在实际部署中更为复杂；最后，AFD 在主流细粒度专家模型与普通集群上表现不佳，需要硬件与模型协同设计。

---

## 300. Parallel Composition for Statistical Privacy

**arXiv ID:** 2602.09627 | [PDF](https://arxiv.org/pdf/2602.09627v1)

**作者:** Dennis Breutigam `[一作]` (Institute for Theoretical Computer Science), Rüdiger Reischuk `[通讯]` (Institute for Theoretical Computer Science)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了统计隐私（SP）框架下的多查询组合问题，提出基于随机分区采样的机制并给出了在非同分布数据库上无额外约束的隐私上界；

**💡 创新点**

首次在 SP 环境中给出组合隐私上界，利用随机划分降低查询间依赖并克服了传统差分隐私对背景知识的过度假设；

**🔧 技术方法**

使用随机分区采样、隐私曲线分析、统计隐私框架以及非等分布数据库的概率模型；

**📊 数据集**

实验基于模拟 iid 数据集（大小 2^15=32768 与 2^10=1024，属性概率 p=1/2）进行；

**📈 对比分析**

通过比较 SP 下同一精度所允许的属性查询次数与差分隐私下的高斯噪声机制，结果显示 SP 可支持显著更多查询且精度损失更低；

**⚠️ 局限性**

局限在于主要针对 iid 或可划分分布的数据，未覆盖高度依赖结构或极端背景知识场景，采样导致的精度下降仍需进一步评估。

---

## 301. Toward Fine-Grained Facial Control in 3D Talking Head Generation

**arXiv ID:** 2602.09736 | [PDF](https://arxiv.org/pdf/2602.09736v1)

**作者:** Shaoyang Xie `[一作]` (Southeast University), James Tin-Yau Kwok `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 14439 | [OpenAlex ID](https://openalex.org/A5032352025)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出FG-3DGS框架，用3D Gaussian Splatting技术实现音频驱动的高质量、时序一致的3D说话头生成。

**💡 创新点**

创新点包括：①频率感知的解耦建模，将低频面部区域与高频嘴眼区域分离处理；②为高频区域设计专门的后渲染对齐机制，显著提升唇音同步；③结合跨模态注意网络与哈希三平面编码，精细捕捉音频驱动的面部运动。

**🔧 技术方法**

核心技术包括：3D Gaussian Splatting、MLP+哈希三平面编码、频率感知分离建模、交叉模态注意网络、后渲染唇同步判别器、alpha混合、L1、D-SSIM、LPIPS和同步损失。

**📊 数据集**

使用公开的高分辨率说话肖像视频数据集（约6500帧/视频），对每位人物划分训练/测试10:1比例。

**📈 对比分析**

与2D、NeRF和3D Gaussian基线对比，FG-3DGS在PSNR 33.06、LPIPS 0.0252、FID 4.846、LMD 2.62、LSE-C 6.26、FPS 90等指标上均优于最新方法，并在跨主体唇同步实验中取得最高LSE-C得分。

**⚠️ 局限性**

局限性：模型针对单人物训练，跨人物泛化受限；对极端头部运动或光照变化仍有挑战；训练时间约2小时，需要强显存GPU；高频区域仍可能出现细节过平滑或同步偏差。

---

## 302. AI-Assisted Scientific Assessment: A Case Study on Climate Change

**arXiv ID:** 2602.09723 | [PDF](https://arxiv.org/pdf/2602.09723v1)

**作者:** Christian Buck `[一作]`, Boris Sakschewski `[通讯]` (Potsdam Institute for Climate Impact Research)

**通讯引用:** 6066 | [OpenAlex ID](https://openalex.org/A5085205601)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项针对大西洋子午环流（AMOC）稳定性评估的实验中，13位气候科学家与 Gemini LLM 共同完成了一份约8000字、涵盖79篇文献的科学评估报告，整个过程共经历104轮修订，耗时约46小时。

**💡 创新点**

创新点在于首次将代理式 AI（co‑scientist）深度嵌入完整的评估工作流——从大纲生成、章节撰写到最终整合，并结合实时证据检索、聊天交互和修订评审模块，展示了 AI 与人类专家的协同机制及其在“验证贫乏”科学问题中的可行性。

**🔧 技术方法**

采用的技术包括 Gemini 2.5 Pro 与 Gemini 3 Pro Preview 语言模型；BM25 与稠密检索结合的前置文献检索；基于 OpenAlex 的证据面板；文本编辑器、聊天框、修订评审工具；句子级对齐与精度/召回/F1 评价指标；以及使用文本嵌入的相似度与收敛性分析。

**📊 数据集**

使用的数据集为一套预构建的 1,660 篇 AMOC 相关科学论文集合（通过 OpenAlex 收集并转为 Markdown），以及实验中引用的 79 篇核心文献。

**📈 对比分析**

方法比较上，AI 生成的内容被 90%+ 保留，AI 贡献的修订占最终文本约 25%，但人类贡献 58% 的内容；效率提升估计 3–17 倍；句子级精度（Retention）几乎完美，召回率低，F1 分别为 76.7%（大纲）和 57.7%（完整报告），说明 AI 在结构保持上表现优异，但需要人类补充深度与准确性。

**⚠️ 局限性**

局限性包括：LLM 的幻觉、量化数据合成困难、对抗性请求时的顺从倾向；缺乏对深层科学推理和隐性专业知识的掌握；工具仍为原型，存在响应延迟、学习曲线和偶发错误；在人类- AI 协同模式下，AI 仍无法完全替代专家对争议性证据的细致评判。

---

## 303. BRAVA-GNN: Betweenness Ranking Approximation Via Degree MAss Inspired Graph Neural Network

**arXiv ID:** 2602.09716 | [PDF](https://arxiv.org/pdf/2602.09716v1)

**作者:** Justin Dachille `[一作]` (Kings College London), Emanuele Natale `[通讯]` (Universite Cote dAzur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并评估了一种轻量级图神经网络 BRAVA‑GNN，用于快速近似网络节点的介数中心性排名。

**💡 创新点**

创新点在于：①使用多跳度质量（degree mass）作为大小不变的节点特征；②采用双向消息传递以同时捕获入射与出射信息；③在训练时引入超几何随机图（hyperbolic random graphs）以提升对高直径图（如道路网络）的泛化；④整体参数量极低（≈1.3k），比现有最轻量基线低54倍。

**🔧 技术方法**

技术包括：双层 GNN 消息传递、ReLU 与 MLP 组合、L2 归一化、Margin Ranking 损失、度质量特征编码、超几何随机图生成与合成数据训练。

**📊 数据集**

训练集为 20 个合成图（10 个无尺度图 + 10 个超几何随机图），测试集为 19 个真实网络（社交网络、网页、电邮、合作、道路网等）。

**📈 对比分析**

与 GNN‑Bet 与 ABCDE 两大 SOTA GNN 进行对比，评估 Kendall τ_b 相关性与推理时间。BRAVA‑GNN 在大多数数据集上取得最高 Kendall τ_b，尤其在道路网络上提升约 214% 或更高；在推理速度上相对最优基线提升 20–70 倍。

**⚠️ 局限性**

局限性：在某些网络上相对 SOTA 仍略逊，推理时间虽大幅提升但仍受限于模型深度；未扩展到其他基于路径的中心性或时变图；对极大规模或特殊拓扑的泛化仍待验证。

---

## 304. Administrative Law's Fourth Settlement: AI and the Capability-Accountability Trap

**arXiv ID:** 2602.09678 | [PDF](https://arxiv.org/pdf/2602.09678v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 305. Stroke3D: Lifting 2D strokes into rigged 3D model via latent diffusion models

**arXiv ID:** 2602.09713 | [PDF](https://arxiv.org/pdf/2602.09713v1)

**作者:** Ruisi Zhao `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**通讯引用:** 80706 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了 Stroke3D，一种通过二维手绘笔画和文字提示直接生成可动画化的骨架和纹理化网格的两阶段框架。

**💡 创新点**

创新点在于：①采用骨架图VAE+骨架图扩散Transformer生成骨架并实现对笔画的显式结构约束；②提出 TextuRig 数据集提升骨架到网格的纹理化训练；③设计 SKA‑DPO 通过骨架‑网格对齐评分进行偏好优化，显著提高几何一致性。

**🔧 技术方法**

核心技术包括：骨架图 VAE (Sk‑VAE)、骨架图扩散 Transformer (Sk‑DiT)、TransformerConv、自注意力、跨注意力、CLIP 文本编码、Skeletal Graph DiT、SKA‑DPO（Direct Preference Optimization）、MVDream、SKDream、Dynov2、以及基于图像的 VLM 生成字幕。

**📊 数据集**

使用的主要数据集：MagicArticulate（骨架）、UniRig/Objaverse‑XL（骨架与纹理）经过筛选得到的 TextuRig、SKDream 数据集，训练时还结合了从这些数据中生成的文本描述。

**📈 对比分析**

与 RigNet、SKDream、MagicArticulate、UniRig、SDEdit 等基线对比。Skeleton 任务中在 CD‑J2J、CD‑J2B、CD‑B2B 上均取得最低误差；Mesh 任务中在 SKA 评分上平均提升约 1.9–2.4 分，并在 DPO 后进一步提升至 87.84（实例）/84.21（类别）。总体表现优于现有方法。

**⚠️ 局限性**

局限性：①对笔画输入仍需用户一定的手绘熟练度；②骨架生成对极少节点或极稀疏笔画的鲁棒性虽好但在极端情况下仍可能失真；③TextuRig 数据集规模有限，未来需更大规模纹理化骨架数据；④模型训练和推理成本相对较高。

---

## 306. PiTPM: Partially Interactive Signatures for Multi-Device TPM Operations

**arXiv ID:** 2602.09707 | [PDF](https://arxiv.org/pdf/2602.09707v1)

**作者:** Yunusa Simpa Abdulsalam `[一作]` (University Mohammed VI Polytechnic), Mustapha Hedabou `[通讯]` (University Mohammed VI Polytechnic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文设计了 PiTPM 框架，利用 TPM 2.0 设备实现无交互的多签名和阈值签名，通过可信聚合器预共享随机种子来消除传统签名协议中必需的承诺阶段，从而实现签名大小不随参与者数量增长。

**💡 创新点**

创新点在于：
1) 采用聚合器生成全局承诺，消除交互式承诺与挑战同步的瓶颈；
2) 通过预共享随机种子和 PRF 计算全局承诺，实现无交互且每个签名者可并行生成签名；
3) 结合 TPM 硬件隔离与远程 attestation，形成混合信任模型，既保留硬件安全，又提升性能；
4) 对多签名和阈值签名给出 EU‑CMA 安全证明，涵盖 rogue key、聚合器失效、随机种子泄露等攻击场景。

**🔧 技术方法**

主要技术手段包括：
- Schnorr 签名与 PRF（伪随机函数）
- TPM 2.0 关键操作（TPM2_Commit, TPM2_Sign）
- 可信聚合器实现（TPM、Intel SGX、ARM TrustZone）
- 随机预言机模型下的安全证明
- Shamir 秘密共享与 Lagrange 插值
- 远程 attestation 与 NVRAM 密封存储

**📊 数据集**

实验中未使用传统机器学习或图像等公开数据集，而是基于真实 TPM 硬件和模拟网络环境进行基准测试，评估了不同参与者数量（2–100、70 等）下的签名时间、通信量和吞吐量。

**📈 对比分析**

对比方法：将 PiTPM 与 MuSig2、FROST 以及 BLS 签名方案在 TPM 环境下进行端到端签名延迟、通信复杂度、网络延迟容忍度和签名大小的比较。实验结果表明：
- PiTPM 的签名时间仅为交互式方案的 1/15–1/18；
- 通信量从 O(n²) 降至 O(n)；
- 签名大小保持 96 bytes 不变；
- 在阈值钱包场景中，PiTPM 达到 421 tx/s 的吞吐量，远超传统方案（≈21 tx/s）。

**⚠️ 局限性**

限制与不足：
1) 需要可信聚合器，单点失效可能影响可用性；
2) 对聚合器性能的评估仅在最多 100 名参与者范围内，缺乏大规模（千人级）验证；
3) 方案仅在经典密码学模型下安全，未考虑后量子攻击；
4) 预共享随机种子分发过程仍需安全机制，若被泄露需重新分发；
5) 对恶意聚合器的攻击仅限定于可用性和错误承诺，不覆盖更高级的篡改行为。

---

## 307. Maastricht University at AMIYA: Adapting LLMs for Dialectal Arabic using Fine-tuning and MBR Decoding

**arXiv ID:** 2602.09703 | [PDF](https://arxiv.org/pdf/2602.09703v1)

**作者:** Abdulhai Alali `[一作]` (Maastricht University), Abderrahmane Issam `[通讯]` (Maastricht University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5078022416)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过LoRA微调单语与英-方言平行数据，结合TIES融合适配器和基于ADI2的最小贝叶斯风险（MBR）解码，提升方言阿拉伯语生成与翻译的方言真实性与语义准确性。

**💡 创新点**

首次将单语与翻译微调的LoRA适配器融合，并在推理阶段引入方言感知的ADI2 MBR重排序，形成一个紧凑高效且能兼顾方言流利度与语义忠实度的生成框架。

**🔧 技术方法**

LoRA参数高效微调、TIES-merge适配器融合、基于ADI2的MBR解码、ADI2与chrF++评估指标、方言识别模型（NADI）。

**📊 数据集**

Shami Corpus、UFAL（Syrian）、DoDa、SDC、SauDial（Moroccan、Saudi）等单语与英-方言平行数据集。

**📈 对比分析**

对JAIS-2和LLaMA3.2基线进行对比；单语FT显著提升ADI2，MT FT提升chrF++；TIES融合进一步平衡两项指标；ADI2 MBR解码使ADI2提升至约0.51，同时保持chrF++；在AMIYA官方评测中，Moroccan ADI2=0.679、Syrian=0.389、Saudi=0.464，翻译任务chrF++表现领先。

**⚠️ 局限性**

数据量有限，ADI2指标难以捕捉细微或非正式方言；MBR解码需要额外前向推理和候选集生成，显著增加推理时延，限制实时或低延迟应用。

---

## 308. MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering

**arXiv ID:** 2602.09642 | [PDF](https://arxiv.org/pdf/2602.09642v1)

**作者:** Sieun Hyeon `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**通讯引用:** 983 | [OpenAlex ID](https://openalex.org/A5024989829)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 MATA（Multi-Agent TableQA）框架，通过多路径推理（Chain-of-Thought、Program-of-Thought、text-to-SQL）和轻量工具（调度器、置信检查器、格式匹配器、判定器）来生成并选择表格问答答案，显著降低对大型 LLM 的调用次数。

**💡 创新点**

创新点在于：①实现模型无关性，能在开源/闭源、不同规模 LLM 上均保持高性能；②引入调度器根据表格特征动态决定推理路径优先级；③通过置信检查器快速筛选高置信答案，避免不必要的 LLM 调用；④使用轻量工具进行代码/SQL 调试与格式校正，提升答案质量与执行效率。

**🔧 技术方法**

技术手段包括：多路 LLM 推理（使用 3B+ 参数模型），轻量工具模型（MobileBERT、DeBERTaV3、Qwen2.5-instruct 500M 以下），调度器、置信检查器、判定器、格式匹配器、调试代理，LangChain 与 Ollama 框架实现模型切换，基于迭代调试的代码/SQL 纠错，最终通过置信阈值和判定器完成答案挑选。

**📊 数据集**

数据集：用于训练调度器与置信检查器的三大公开表格问答集（WikiTQ、TabMWP、TabFact）共 173,664 条样本；评估集为 Penguins in a Table（易）和 TableBench（难），涵盖事实检验、数值推理、数据分析等多子任务。

**📈 对比分析**

方法比较：在 Penguins in a Table 和 TableBench 上与 SynTQA、MixSC、TabLaP 等基线对比，MATA 在所有 10 种 LLM（小/大、开源/闭源）上均获得最佳或接近最佳的 Exact Match、Fuzzy 和 F1 分数，特别是在 TableBench 上相较最佳基线提升 40.1%（EM）、46.7%（fuzzy）和 33.1%（F1），显示出卓越的通用性与高效性。

**⚠️ 局限性**

局限性：仍需完整的 LLM 推理，单个大模型的计算成本未得到根本降低；对 LLM 本身的幻觉、偏见和不确定性依赖未完全消除；轻量工具的鲁棒性与跨域适配仍需进一步验证。

---

## 309. Stop Testing Attacks, Start Diagnosing Defenses: The Four-Checkpoint Framework Reveals Where LLM Safety Breaks

**arXiv ID:** 2602.09629 | [PDF](https://arxiv.org/pdf/2602.09629v1)

**作者:** Hayfa Dhabhi `[一作]`, Kashyap Thimmaraju `[通讯]` (Technische Universität Berlin)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5088851494)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建四检查点框架，对大型语言模型的安全机制进行分层评估，揭示其弱点与突破方式。

**💡 创新点**

创新点在于将安全流程拆分为输入/输出与文字/意图两维，形成四个可独立测评的检查点，并提出针对每个检查点的13种黑盒绕过技术。

**🔧 技术方法**

使用了基于LLM的判别器（LLM-as-judge）进行四级泄漏分类，并引入加权攻击成功率（WASR）指标；通过构造变形提示实现对四个检查点的系统性攻击。

**📊 数据集**

采用四个公开基准（HarmBench、JailbreakBench、AdvBench、Do-Not-Answer）共81条精心挑选的恶意提示，涵盖八大危害类别。

**📈 对比分析**

在GPT‑5、Claude Sonnet 4和Gemini 2.5 Pro三大前沿模型上，单轮黑盒测试共3,312例，二元攻击成功率仅22.6%，但加权成功率高达52.7%，表明输出阶段防御最脆弱。

**⚠️ 局限性**

局限包括仅评估单轮攻击、使用单一LLM评判器可能存在偏差、技术手段为手工设计且仅针对公开模型，无法覆盖多轮、跨模态或白盒攻击场景。

---

## 310. Sparse Axonal and Dendritic Delays Enable Competitive SNNs for Keyword Classification

**arXiv ID:** 2602.09746 | [PDF](https://arxiv.org/pdf/2602.09746v1)

**作者:** Younes Bouhadjar `[一作]` (Forschungszentrum Jülich), Emre Neftci `[通讯]` (Forschungszentrum Jülich)

**通讯引用:** 4723 | [OpenAlex ID](https://openalex.org/A5060924942)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文在深度前馈神经网络中引入可学习的轴突或树突延迟，证明其能够完成关键词分类任务。

**💡 创新点**

创新点在于将延迟与神经元关联而非突触，显著降低缓冲与存储需求，同时保持与传统突触延迟模型相近的准确率。

**🔧 技术方法**

采用 LIF 神经元、DCLS 延迟学习、ATan 代理梯度、稀疏延迟/权重以及事件驱动语音预处理技术。

**📊 数据集**

使用 Google Speech Commands（GSC）及其事件化版本 Spiking Speech Commands（SSC）两大数据集。

**📈 对比分析**

通过与突触延迟、适应性 LIF 以及其他基线模型比较，轴突/树突延迟模型在 GSC 上可达 95%+ 的准确率，在 SSC 上可达 80%+ 的准确率，同时参数量、缓冲区大小、SOP 与脉冲数均优于传统方法。

**⚠️ 局限性**

局限性包括在更复杂时序数据上表现差异不明显、延迟范围过大导致性能下降、树突延迟对稀疏化和活动正则化更为敏感。

---

## 311. GenSeg-R1: RL-Driven Vision-Language Grounding for Fine-Grained Referring Segmentation

**arXiv ID:** 2602.09701 | [PDF](https://arxiv.org/pdf/2602.09701v1)

**作者:** Sandesh Hegde `[一作]` (Camcom Technologies Pvt. Ltd.), Uma Mahesh `[通讯]` (Camcom Technologies Pvt. Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了细粒度参照图像分割的“先推理再分割”管线，利用Qwen3‑VL生成盒子与关键点提示，SAM 2冻结模型输出高质量掩模。

**💡 创新点**

创新点在于将GRPO强化学习与SAM 2奖励循环结合，支持负面查询检测并通过GRefCOCO训练提升无目标提示准确率。

**🔧 技术方法**

采用Qwen3‑VL视觉语言模型、GRPO训练、SAM 2预训练分割器、SAM 2反馈奖励、以及GRefCOCO负样本标注等技术。

**📊 数据集**

使用VisionReasoner‑MultiObjects‑7K、GRefCOCO、RefCOCOg和ReasonSeg等数据集。

**📈 对比分析**

在同一SAM 2下与Seg‑Zero、Seg‑R1、Qwen3‑VL‑Instruct等基线对比，-8B在RefCOCOg上cIoU 0.7127、mIoU 0.7382，超越Seg‑Zero +3.3 cIoU；-G在GRefCOCO上无目标准确率82.4%、目标mIoU 76.69%；-4B在ReasonSeg上mIoU 68.4%。

**⚠️ 局限性**

局限性包括仅在COCO领域数据上训练、对长尾属性和跨域场景性能未知、SAM 2对提示格式敏感、未完全消除数据重叠影响。

---

## 312. Contextual and Seasonal LSTMs for Time Series Anomaly Detection

**arXiv ID:** 2602.09690 | [PDF](https://arxiv.org/pdf/2602.09690v1)

**作者:** Lingpei Zhang `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 7718 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于双分支（季节性与上下文）LSTM的无监督异常检测框架CS-LSTMs，结合噪声分解与频域/时域预测；

**💡 创新点**

通过噪声分解去除异常影响、双分支并行捕捉长期周期与短期局部趋势、使用频域与时域特征融合，显著提升对细微点异常和慢升异常的检测；

**🔧 技术方法**

使用FFT变换+频域LSTM、短窗口重叠上下文LSTM、Wavelet去噪、掩码负对数似然损失、无监督训练；

**📊 数据集**

在四个公共UTS基准集上评测：Yahoo、KPI、WSD、NAB；

**📈 对比分析**

与10种主流基线（SPOT、SRCNN、TFAD、DONUT、Informer、Anomaly‑Transformer、AnoTransfer、VQRAE、KAN‑AD、FCVAE）对比，CS‑LSTMs在Best‑F1和Delay‑F1均领先，提升幅度约3–10%，并且模型参数约600K，推理速度提升40%；

**⚠️ 局限性**

局限性在于需先行估计周期窗口大小，对强噪声或非周期性序列的适用性尚待进一步验证，且模型仍以无监督方式训练，无法利用少量标签提升性能。

---

## 313. RANT: Ant-Inspired Multi-Robot Rainforest Exploration Using Particle Filter Localisation and Virtual Pheromone Coordination

**arXiv ID:** 2602.09661 | [PDF](https://arxiv.org/pdf/2602.09661v1)

**作者:** Ameer Alhashemi `[一作]`, Suryanarayana Datla `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出RANT框架，实现蚂蚁启发式多机器人雨林探索与粒子滤波定位、梯度增益以及虚拟信息素分布的协同控制；

**💡 创新点**

将粒子滤波定位、行为驱动的梯度爬升以及基于信息素的无回访阻止机制融合到同一多机器人系统，并通过实验量化三者对覆盖率、热点召回率与冗余采样的影响；

**🔧 技术方法**

粒子滤波定位、基于行为的控制器（探索/热点/恢复模式）、Webots仿真、GPS/IMU融合、IR避障、虚拟信息素地图、梯度估计；

**📊 数据集**

仿真环境为10×10 m Webots世界，隐藏的“丰度”场为50×50网格的四个高斯混合，利用噪声模型模拟GPS/IMU/滑移、雾雨天气与传感器误差；

**📈 对比分析**

通过对比团队规模（1、3、5）、定位开启/关闭以及信息素协同/禁用三组实验，评估覆盖率、热点召回、冗余率、定位误差；结果显示：5机器人时覆盖率18.1%、热点召回65.6%，信息素协同将冗余率从约40%降至约25%，定位开启显著提升热点检测；

**⚠️ 局限性**

依赖中心化监管、缺乏真实视觉惯性里程计、仿真环境简化（无障碍物动态、通信丢包），且信息素阻止不完全消除热点边界重叠，未来需要更真实感知与分布式通信验证。

---

## 314. Blind denoising diffusion models and the blessings of dimensionality

**arXiv ID:** 2602.09639 | [PDF](https://arxiv.org/pdf/2602.09639v1)

**作者:** Zahra Kadkhodaie `[一作]` (Flatiron Institute), Eero Simoncelli `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文理论和实证分析了基于盲去噪器的生成扩散模型的性能，证明了在数据分布具有低内在维度的假设下，盲去噪扩散模型（BDDMs）能够在没有噪声幅度信息的情况下，自动跟踪隐式噪声调度并准确从数据分布中采样。

**💡 创新点**

创新点在于首次为盲去噪扩散模型的成功提供了数学证明，指出低内在维度是其有效性的关键因素，并展示了BDDMs在采样质量上优于非盲模型的现象。

**🔧 技术方法**

使用了盲去噪器作为生成模型的核心，结合了贝叶斯统计估计噪声水平的技术。

**📊 数据集**

使用了合成数据和真实图像数据集（如CelebA和LSUN）进行实验验证。

**📈 对比分析**

通过与非盲去噪扩散模型的比较，发现BDDMs在相同初始噪声样本下生成的图像质量显著更高，且BDDMs能够有效避免由于噪声调度不匹配导致的错误。

**⚠️ 局限性**

限制在于当前理论分析主要基于低内在维度的假设，未来需要在更大规模和其他下游任务中进一步验证BDDMs的有效性。

---

## 315. Directed Information: Estimation, Optimization and Applications in Communications and Causality

**arXiv ID:** 2602.09711 | [PDF](https://arxiv.org/pdf/2602.09711v1)

**作者:** Dor Tsur `[一作]` (Ben-Gurion University of the Negev), Gerhard Kramer `[通讯]` (Technical University of Munich)

**通讯引用:** 16147 | [OpenAlex ID](https://openalex.org/A5047591578)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文综述了有反馈与记忆通道中定向信息的理论基础、估计方法以及其在信道容量、神经网络、因果推断等领域的应用。

**💡 创新点**

创新点在于系统性整合了定向信息与反馈容量的关系，并提出多种从经典统计到神经网络的估计与优化技术，同时讨论了信息流与因果推断的统一框架。

**🔧 技术方法**

使用的技术包括信息理论中的定向信息、互信息、转移熵、格兰杰因果性、Pearl因果模型；以及最大熵、贝叶斯、神经网络、强化学习、CTW、kNN 等估计与优化方法。

**📊 数据集**

所用数据集多为仿真生成的离散/连续信道序列、神经元尖峰记录、基因网络模拟、时间序列金融/股票数据等。

**📈 对比分析**

通过与传统互信息、格兰杰因果检验以及已知容量的对照实验，本文的定向信息估计方法在高维、非平稳和有反馈场景下表现出更高的准确性和收敛速度。

**⚠️ 局限性**

局限性包括计算复杂度随字母表大小与记忆长度呈指数增长，神经网络估计缺乏有限样本误差界，且在高度非线性或多元因果结构下仍需进一步验证。

---

## 316. A Multiliteracy Model for Interactive Visualization Literacy: Definitions, Literacies, and Steps for Future Research

**arXiv ID:** 2602.09631 | [PDF](https://arxiv.org/pdf/2602.09631v1)

**作者:** Gabriela Molina León `[一作]` (Aarhus University), Niklas Elmqvist `[通讯]` (Aarhus University)

**通讯引用:** 8063 | [OpenAlex ID](https://openalex.org/A5034277315)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一个多层识字（multiliteracy）模型，描述人们在交互式可视化系统中所需的能力与过程，并通过案例分析和观察性研究验证该模型。

**💡 创新点**

创新点在于：① 将可视化素养与交互性结合，形成包含三层抽象（战略、战术、操作）与三条“深渊”（目标形成、执行、评估）的二维模型；② 定义了九种新旧混合的“多层识字”，填补了交互可视化素养研究的空白；③ 提出了基于模型的设计、评估与教育实践路线图。

**🔧 技术方法**

采用的技术主要是：文献综述、概念建模、案例编码、定性观察与思考导向访谈；并对15个公开的可视化系统（数据驱动叙事、可视分析、沉浸式多模态分析）进行编码。

**📊 数据集**

使用的数据集包括：世界银行《可持续发展目标（SDG）》可视化数据、EgoLines动态网络数据等；实验参与者共9人，使用两套系统（SDG Atlas 与 EgoLines）进行任务。

**📈 对比分析**

方法对比：通过观察性研究和案例分析验证模型，而非传统量化性能评测；研究表明模型能解释用户在不同“深渊”和层级上遇到的具体识字缺失，但未给出定量性能指标。

**⚠️ 局限性**

局限性包括：① 模型未涵盖盲/低视障用户、合作场景；② 评估工具尚未开发，缺乏量化测量；③ 样本规模与背景有限，难以推广；④ 仅聚焦交互式系统，未覆盖物理可视化；⑤ 研究主要为探索性，需进一步的实证验证。

---

## 317. TeleGate: Whole-Body Humanoid Teleoperation via Gated Expert Selection with Motion Prior

**arXiv ID:** 2602.09628 | [PDF](https://arxiv.org/pdf/2602.09628v1)

**作者:** Jie Li `[一作]` (University of Science and Technology of China), Rongyun Cao `[通讯]` (AnyWit Robotics Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 TeleGate 框架，通过门控专家选择与 VAE 运动预测先验，实现全身实时遥操作，支持多种高动态动作。

**💡 创新点**

创新点包括：①使用门控网络保留专家能力，避免知识蒸馏导致的性能损失；②引入 VAE 运动先验弥补实时遥操作中缺失未来轨迹的缺口；③仅用 2.5 小时惯性运动捕捉数据即可实现高精度跟踪。

**🔧 技术方法**

采用的技术包括门控专家选择、VAE 运动先验、PPO 强化学习、MuJoCo 仿真、动作残差 PD 控制、领域随机化以及自定义奖励设计。

**📊 数据集**

使用自采的 2.5 小时惯性运动捕捉数据集，涵盖 6 种运动（走、跑、舞蹈、武术、跌倒恢复、跳跃），无公开数据来源。

**📈 对比分析**

在与 TWIST、Any2track、GMT 等基线在相同数据集上的对比中，TeleGate 成功率 97.3%，轨迹误差最低（E_mpjpe 17.22 mm，E_mpjae 0.085 rad），显著优于传统方法。

**⚠️ 局限性**

局限性在于仍需依赖昂贵的运动捕捉硬件，对极端动态或突发场景的鲁棒性待验证，门控策略对专家划分敏感，且缺乏端到端自适应学习机制。

---

## 318. DiffuReason: Bridging Latent Reasoning and Generative Refinement for Sequential Recommendation

**arXiv ID:** 2602.09744 | [PDF](https://arxiv.org/pdf/2602.09744v1)

**作者:** Jie Jiang `[一作]` (Tencent), Huan Yu `[通讯]` (Tencent)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5001795301)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出DiffuReason框架，将多步思考（Thinking Tokens）与扩散去噪（Diffusion）结合，实现对序列推荐的隐式推理与生成细化。

**💡 创新点**

创新点在于：①将隐式推理视为概率条件并用扩散模型去噪；②使用端到端GRPO强化学习将生成过程直接对齐到排序指标；③统一训练避免两阶段误差累积。

**🔧 技术方法**

采用Transformer解码器产生Thinking Tokens，扩散去噪网络、GRPO强化学习与常规推荐损失共同训练。

**📊 数据集**

在Amazon Review 4个数据集（Beauty、Instruments、Sports、Video&Games）上进行评估。

**📈 对比分析**

与ReaRec、LARES、SASRec、TIGER、HSTU等基线对比，DiffuReason在Recall@5/10和NDCG@5/10上平均提升15–50%，在工业平台A/B测试中GMV提升0.79%、广告收入提升1.15%。

**⚠️ 局限性**

局限性包括：对扩散步骤和RL样本规模的敏感性，计算成本相对传统方法略高，且对极稀疏或极长序列的进一步优化仍有空间。

---

## 319. Robust Vision Systems for Connected and Autonomous Vehicles: Security Challenges and Attack Vectors

**arXiv ID:** 2602.09740 | [PDF](https://arxiv.org/pdf/2602.09740v1)

**作者:** Sandeep Gupta `[一作]` (Queen's University Belfast), Roberto Passerone `[通讯]` (University of Trento)

**通讯引用:** 4158 | [OpenAlex ID](https://openalex.org/A5020920533)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过构建CAV视觉系统（CAVVS）参考架构，系统分析了其关键资产、脆弱点与攻击面，进而提出了针对数据、模型与输入三大攻击面的一整套攻击向量与对应的CIA风险评估；

**💡 创新点**

创新点在于将传统的安全威胁模型与自动驾驶视觉系统相结合，首次将攻击向量细分为八类（如数据中毒、模型提取、逻辑破坏等），并结合实际车辆感知组件提出完整的攻击面与威胁树；

**🔧 技术方法**

采用深度学习模型（CNN、YOLO、SSD、Faster‑R‑CNN、RNN、LSTM、GAN、Diffusion、Autoencoder、SNN、Transformer、ViT）以及物理与数字攻击技术（对抗样本、激光欺骗、伪造雷达信号、侧通道、MITM、数据外泄等）；

**📊 数据集**

论文主要以公开数据集如KITTI、ImageNet以及自定义模拟场景（CARLA、Inception‑V3实验）为基础，展示了模型在不同环境（光照、天气、遮挡）下的鲁棒性；

**📈 对比分析**

通过对比各类攻击对模型准确率、误报率等指标的影响，作者指出对抗攻击可导致准确率骤降至30%以下，侧通道攻击可泄露模型结构，数据中毒可在测试集上降低>10%性能；但缺乏统一的基准测试框架和量化评估方法；

**⚠️ 局限性**

局限性包括：缺乏实车实测验证，仅在仿真与公开数据集上评估；攻击向量与防御措施的实现细节不足；未考虑联邦学习与多车协同对抗环境的影响；未给出完整的性能/安全权衡指标。

---

## 320. Rethinking Visual-Language-Action Model Scaling: Alignment, Mixture, and Regularization

**arXiv ID:** 2602.09722 | [PDF](https://arxiv.org/pdf/2602.09722v1)

**作者:** Ye Wang `[一作]` (Renmin University of China), Qin Jin `[通讯]` (Renmin University of China)

**通讯引用:** 4942 | [OpenAlex ID](https://openalex.org/A5009985839)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 Vision‑Language‑Action（VLA）模型在多机器人、多感知、多动作空间下的规模化进行系统、对照的实验研究，并提出了可降低实验者偏差的分组盲评估协议。

**💡 创新点**

① 发现统一的 EEF‑相对动作空间是跨胳膊迁移的最佳选择；② 证明随意混合不同机器人数据往往导致负迁移；③ 说明在大规模预训练中常用的感知 dropout 与分阶段对齐课程并不总能带来提升；④ 通过分组盲评估有效降低真实机器人实验中的人为偏差。

**🔧 技术方法**

使用 Mixture‑of‑Transformers + flow‑matching 控制框架；VLM 背骨（InternVL‑3.5‑2B）+随机初始化的动作专家；统一的动作空间划分为 EEF、关节、抓手等子空间；感知 dropout、两阶段训练等正则化手段。

**📊 数据集**

整合 Open X‑Embodiment（OXE）、Agibot、RoboMind、InternData、SO‑100 等公开/专有数据集，覆盖真实与仿真、EEF 与关节空间，最终形成约 1.8 亿帧的平衡混合数据集。

**📈 对比分析**

在 LIBERO 与 RoboCasa 仿真基准以及 Franka Panda 实验平台上进行 5‑shot / 50‑shot 微调；与 3 种代表性通用 VLA 策略（π0、π0.5、GR00T‑N1）对比；结果表明 EEF‑相对策略在预训练后提升约 8‑10% 成功率，混合数据会导致 3‑5% 的性能下降；dropout 与两阶段训练对最终成功率影响不显著。

**⚠️ 局限性**

① 研究仅关注 VLA 的 Mixture‑of‑Transformers 架构，其他架构的可迁移性未验证；② 仅在有限任务与机器人上评估，可能不完全代表更广泛的操作场景；③ 混合数据负迁移的原因未深入解析（如关节约束差异、时序偏差等）。

---

## 321. Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs

**arXiv ID:** 2602.09719 | [PDF](https://arxiv.org/pdf/2602.09719v1)

**作者:** Longhuan Xu `[一作]` (Southeast University-Monash University Joint Graduate School), Feng Yin `[通讯]` (Southeast University)

**通讯引用:** 17308 | [OpenAlex ID](https://openalex.org/A5068182856)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向大型语言模型的无监督、样本特定测试时自适应框架，利用层级动态学习率调节（ScaleNet）对LoRA参数进行更新。

**💡 创新点**

核心创新是通过轻量级超网络预测每层、每步学习率倍率，实现对输入Prompt的细粒度、动态控制，从而避免固定学习率导致的过拟合与漂移。

**🔧 技术方法**

结合LoRA参数化、Transformer层结构、第一阶梯度近似、轻量MLP超网络（ScaleNet）以及无监督负对数似然自适应训练。

**📊 数据集**

在Llama‑3.2、Llama‑3.3、Qwen‑3系列模型上，使用XSum、SQuAD、NQ‑Open、AdaptEval等多任务、多领域数据集进行评估。

**📈 对比分析**

与固定学习率基线和步骤级别调度对比，实验表明动态层级调度在NLL和ROUGE‑Lsum上均显著提升，尤其在更大规模模型上效果更佳。

**⚠️ 局限性**

主要限制在于适配效果高度依赖训练时的任务分布，跨域泛化受限；超网络容量有限，可能无法捕捉更复杂的Prompt特征。

---

## 322. Fast Motion Planning for Non-Holonomic Mobile Robots via a Rectangular Corridor Representation of Structured Environments

**arXiv ID:** 2602.09714 | [PDF](https://arxiv.org/pdf/2602.09714v1)

**作者:** Alejandro Gonzalez-Garcia `[一作]` (KU Leuven), Wilm Decré `[通讯]` (KU Leuven)

**通讯引用:** 953 | [OpenAlex ID](https://openalex.org/A5009275320)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于确定性矩形通道分解的非完整移动机器人实时运动规划框架。

**💡 创新点**

创新点在于将全局自由空间分解为少量重叠矩形通道，构造紧凑图并利用解析轨迹生成器实现近实时、近时间最优且符合运动学的路径。

**🔧 技术方法**

技术包括离线矩形通道生成算法（线段检测、snap点、矩形生成与障碍裁剪）、离线图构造、在线Dijkstra路径搜索以及基于时间最优运动原语的解析轨迹规划。

**📊 数据集**

使用了24个合成室内地图（从330×630到2430×3010像素）以及在实验室投影环境下的ROSbot 3数据。

**📈 对比分析**

与A*、Hybrid‑A*和State‑Lattice Planner比较，平均规划时间仅为20‑40 ms，长路径上比传统网格规划快2.3倍，且比运动学可行规划器快十倍。

**⚠️ 局限性**

局限性在于仅适用于结构化地图，对非规则或细节丰富环境效果差，短路径和少通道场景收益有限，且当前解析规划仅考虑速度约束，未处理加速度和未定义的异常情况。

---

## 323. Physics-informed diffusion models in spectral space

**arXiv ID:** 2602.09708 | [PDF](https://arxiv.org/pdf/2602.09708v1)

**作者:** Davide Gallon `[一作]` (University of Munster), Arnulf Jentzen `[通讯]` (University of Munster)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种在谱空间中进行物理信息扩散的生成式框架（PISD），能够在给定稀疏观测条件下生成参数化 PDE 的解和参数；

**💡 创新点**

通过对谱系数进行数据相关的缩放，使得扩散过程在函数空间保持 Sobolev 正则性，从而在整个采样过程中都能施加 PDE 约束；

**🔧 技术方法**

使用谱系数作为潜在表示，训练扩散去噪器（ViT 架构），采用逆时 ODE 采样并在推理阶段加入基于 Adam 的 PDE 与观测约束的指导；

**📊 数据集**

在 128×128 分辨率下生成 Poisson、Helmholtz 以及 Navier–Stokes 方程的解与参数，数据由这些 PDE 在不同参数下的数值解构成；

**📈 对比分析**

与 DiffusionPDE、FunDPS 等现有基于扩散的 PDE 求解器对比，PISD 在前向、逆向及联合重建任务中均保持或超过其精度，并在推理时间上实现 3~15 倍的加速；

**⚠️ 局限性**

依赖手工谱编码，难以直接推广到不规则几何或复杂边界条件；需要大量训练数据；

---

## 324. Life Cycle-Aware Evaluation of Knowledge Distillation for Machine Translation: Environmental Impact and Translation Quality Trade-offs

**arXiv ID:** 2602.09691 | [PDF](https://arxiv.org/pdf/2602.09691v1)

**作者:** Joseph Attieh `[一作]` (University of Helsinki), Jörg Tiedemann `[通讯]` (University of Helsinki)

**通讯引用:** 8669 | [OpenAlex ID](https://openalex.org/A5082417280)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在机器翻译知识蒸馏（KD）的生命周期内，结合碳足迹评估与翻译质量，构建了一套完整的环境与性能双重评估框架。

**💡 创新点**

创新点在于：①引入MLCA框架将教师训练、蒸馏过程及推理阶段的碳排放量化；②绘制质量–碳足迹 Pareto 前沿，直观展示不同 KD 方法的成本与收益；③给出推理量阈值，帮助决策何时使用 KD 才真正绿色。

**🔧 技术方法**

技术手段包括：Transformer‑Big/Transformer‑Base/Transformer‑Tiny 模型，六种主流 KD 方法（Word‑KD、Sel‑KD、TIE‑KD、Seq‑KD、Seq‑INTER、Seq‑REP）；使用 COMET 指标评估翻译质量；利用 MLCA 计算 CO₂e（涵盖训练、蒸馏、推理的运营与硬件制造排放）。

**📊 数据集**

数据集：WMT 2024 英语→冰岛语并在 FLORES+ devtest 集上进行质量评估，教师与学生均在相同并行语料上训练。

**📈 对比分析**

通过对比教师、无 KD baseline、以及各 KD 变体在不同推理量（100 k 与 1 M 目标词）下的碳足迹与 COMET，结果显示：Word‑KD 在中等规模推理时成本高，Seq‑KD 成本更高；仅当推理量超过约几百万词时，KD 的碳足迹才低于直接使用教师；在 65 M 学生模型中可实现教师级质量并获得最佳成本–质量平衡，而 16 M 模型受质量上限限制。

**⚠️ 局限性**

局限性：仅在单一 NVIDIA V100 GPU 上实验，未覆盖多 GPU/多机训练与网络传输开销；仅评估 EN→IS 低资源语言，未验证在高资源语言上的推广；假设教师可完整访问（无法使用闭源教师时 Seq‑KD 成本更高），以及未考虑端设备与网络层面的能耗。

---

## 325. Model soups need only one ingredient

**arXiv ID:** 2602.09689 | [PDF](https://arxiv.org/pdf/2602.09689v1)

**作者:** Alireza Abdollahpoorrostam `[一作]` (École Polytechnique Fédérale de Lausanne), Pascal Frossard `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 26637 | [OpenAlex ID](https://openalex.org/A5000947076)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MonoSoup，一种仅使用单一 fine‑tuned 检查点的后置模型编辑方法，以保持 ID 与 OOD 性能平衡。

**💡 创新点**

创新点在于对每层权重更新做奇异值分解，自适应划分高能量与低能量子空间，并基于谱衰减与余弦相似度给出无参数加权规则。

**🔧 技术方法**

使用技术包括奇异值分解、有效秩自适应阈值、基于谱能量与对齐度的层级重权重、数据无关、无超参数的后置编辑。

**📊 数据集**

使用数据集包括 CLIP ViT‑B/32 与 ViT‑L/14 在 ImageNet 及其五个自然 OOD 变体（ImageNet‑V2、‑R、‑Sketch、‑A、ObjectNet），以及 Qwen3‑0.6B 在数学推理与多项选择（GSM8K、MATH、SciQ、MMLU‑P、OpenBookQA 等）任务。

**📈 对比分析**

通过与模型熔化（Model Soups）、ModelStock、Wise‑FT、LiNeS 等多模型或单模型方法对比，MonoSoup 在保持 ID 准确率不变的同时平均 OOD 提升约 1%–8%，在弱检查点甚至可恢复至与多模型相近的性能。

**⚠️ 局限性**

限制在于仍需设定能量保留阈值 R，虽然可自动化但对不同架构可能需要微调；方法仍需奇异值分解，极大模型的计算与存储成本不低。

---

## 326. TreeCUA: Efficiently Scaling GUI Automation with Tree-Structured Verifiable Evolution

**arXiv ID:** 2602.09662 | [PDF](https://arxiv.org/pdf/2602.09662v1)

**作者:** Deyang Jiang `[一作]` (Meituan), Zhixiong Zeng `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TreeCUA 框架，通过多智能体树结构可验证演化方法高效生成多样化 GUI 自动化轨迹，并基于树分支自生成的偏好数据训练 TreeCUA‑DPO。

**💡 创新点**

创新点在于：① 将 GUI 轨迹组织成树结构以共享前缀并消除步骤冗余；② 采用世界知识初始化、全局记忆回溯与自适应树拓扑实现高效且多样化的探索；③ 利用树分支天然产生的正负偏好对进行无成本 DPO 训练。

**🔧 技术方法**

使用多智能体协作（探索、验证、总结、评估）框架、树结构节点回放、世界知识库、全局记忆回溯、自适应树拓扑、可验证单步检查以及基于树分支的 DPO。

**📊 数据集**

数据集包括 100k 生成轨迹、101k 子轨迹、708k 单步样本，随后筛选得到 50k 高质量轨迹，全部公开发布于 https://github.com/UITron-hub/TreeCUA。

**📈 对比分析**

在 OSWorld‑Verified 基准上，TreeCUA‑7B 取得 34.6% 成功率，TreeCUA‑DPO‑7B 提升至 36.6%，明显优于同规模公开基线（OpenCUA 24.3%、ScaleCUA 15%）和开源模型（Qwen2.5‑VL 5.5%）；在 120 个 OOD 任务上，成功率从 0.8% 提升至 26.7%/30.8%，展示强泛化。

**⚠️ 局限性**

局限性包括：① 仍需高质量视觉‑语言模型支持，模型规模受限；② 树结构探索对操作系统快照和可重放的依赖可能在真实系统上受限；③ 对极长或高度动态的任务序列，树深度和回溯机制仍可能产生效率瓶颈。

---

## 327. Efficient Remote Prefix Fetching with GPU-native Media ASICs

**arXiv ID:** 2602.09725 | [PDF](https://arxiv.org/pdf/2602.09725v1)

**作者:** Liang Mi `[一作]` (Nanjing University), Yunxin Liu `[通讯]` (Institute for AI Industry Research, Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于GPU原生视频编解码器的远程KV缓存重用系统，用以加速大型语言模型（LLM）的推理。

**💡 创新点**

创新点包括：① 设计了兼容视频编码的张量布局（codec‑friendly tensor layout）以最大化压缩率；② 将KV压缩/解压完全迁移到GPU内置NVENC/NVDEC硬件上，消除对CUDA核心的竞争；③ 开发了异步KV fetcher，采用自适应分辨率和帧级恢复，隐藏网络波动并避免显存竞争；④ 在压缩时保持lossless，确保推理精度不受影响。

**🔧 技术方法**

技术手段主要有：GPU原生视频编解码器（NVENC/NVDEC）+H.265 lossless编码、FFmpeg + GStreamer+CUDA +Pybind11；张量布局优化利用SSIM/PSNR评估；自适应分辨率调度基于网络带宽预测；帧级KV恢复通过回调实现。

**📊 数据集**

实验使用长上下文基准：L‑Eval、LV‑Eval 和 LongBench‑V2，覆盖 3–200K 令牌的多种任务（问答、编程、对话等）。

**📈 对比分析**

与 Full‑Prefill、Raw‑KV‑Reuse、CacheGen、ShadowServe 以及 llm.265 等现有方案对比，实验在 1–40 Gbps 带宽、20K–200K 上下文长度下，TTFT（首个 token 时间）平均提升 1.29–3.5 倍，压缩率提升 2–3 倍，且保持 100% 推理精度；对非重用请求也几乎不产生干扰，TPOT 下降 35–40%。

**⚠️ 局限性**

局限性包括：NVENC 资源有限导致压缩过程仍不够快，无法满足在线压缩需求；低端 GPU 的 NVDEC 数量不足，影响解压吞吐；KV 压缩仅支持离线处理，在线迁移仍有瓶颈；预分配显存虽然能保障重用请求，但可能阻塞非重用请求，需未来改进。

---

## 328. Semi-supervised Liver Segmentation and Patch-based Fibrosis Staging with Registration-aided Multi-parametric MRI

**arXiv ID:** 2602.09686 | [PDF](https://arxiv.org/pdf/2602.09686v1)

**作者:** Boya Wang `[一作]` (University of Nottingham), Xin Chen `[通讯]` (University of Nottingham)

**通讯引用:** 19798 | [OpenAlex ID](https://openalex.org/A5100363117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一套半监督肝脏分割与基于补丁的纤维化分期框架，兼顾多模态MRI的配准与分割，并通过补丁级分类实现可视化分期。

**💡 创新点**

创新点包括将BRBS注册-分割框架改为局部互信息损失并加入加权一致性与空间风格采样；采用补丁级二分类推断病期并将比例映射为概率，实现对局部病变的可视化；同时在非对比与对比模式下均可泛化到OOD数据。

**🔧 技术方法**

采用半监督深度学习（BRBS + MI 损失）、多模态配准（ANTs）、补丁提取+ResNet‑18+MLP 分类器、阈值映射与概率映射等技术。

**📊 数据集**

使用 CARE Liver 2025 Track 4 多中心、多厂商 MRI 数据集（610例），包含 T1、T2、DWI 与 GED1‑GED4 四个动态相位，其中 GED4 为参考相位。

**📈 对比分析**

与原 BRBS‑NCC 进行对比，MI 版本在所有序列上 Dice 提升 4‑10% 且 Hausdorff 距离下降；在官方测试中 ID/OOD Dice 均 >0.9；纤维化分期 AUC 在 ID 下 0.77‑0.84，非对比模式在 OOD 下表现更稳健。

**⚠️ 局限性**

局限性：分割精度对分期影响大，缺失模态通过零填充处理；对比增强模式在 OOD 下性能显著下降；配准主要依赖 GED4，其他相位未统一配准导致潜在误差。

---

## 329. Trade-Offs in Deploying Legal AI: Insights from a Public Opinion Study to Guide AI Risk Management

**arXiv ID:** 2602.09636 | [PDF](https://arxiv.org/pdf/2602.09636v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 330. AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild

**arXiv ID:** 2602.09657 | [PDF](https://arxiv.org/pdf/2602.09657v1)

**作者:** Xiaolou Sun `[一作]` (Southeast University), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 43852 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了AutoFly端到端Vision‑Language‑Action模型，实现UAV在未知环境中仅基于粗略语言指导进行自主导航。

**💡 创新点**

创新点在于引入伪深度编码器以提升RGB的空间推理能力，采用两阶段训练与Siamese MLP投影器，实现视觉、语言与深度的统一对齐，并构建了兼顾仿真与真实的自主导航数据集。

**🔧 技术方法**

使用的技术包括Depth Anything V2伪深度生成、LLaVA+OpenVLA的视觉语言模型、SAC强化学习收集轨迹、Siamese MLP投影器、以及自研的动作解码器。

**📊 数据集**

使用的数据集为新构建的含约1.34万条仿真轨迹与1K条真实飞行数据的自主导航数据集，融合了多种障碍与目标识别任务。

**📈 对比分析**

与RT‑1、RT‑2、OpenVLA等基线对比，AutoFly在模拟与真实环境下的成功率提高约3.9%，碰撞率降低2.6%，路径效率提升2.2%。

**⚠️ 局限性**

局限在于仍依赖大量仿真数据与人工收集的轨迹，深度估计误差及动态障碍物在复杂自然环境中的鲁棒性待进一步验证。

---

## 331. VideoAfford: Grounding 3D Affordance from Human-Object-Interaction Videos via Multimodal Large Language Model

**arXiv ID:** 2602.09638 | [PDF](https://arxiv.org/pdf/2602.09638v1)

**作者:** Hanqing Wang `[一作]` (Hong Kong University of Science and Technology Guangzhou), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 43852 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了从人机交互视频中进行3D物体可供性定位的新任务，并提出了基于多模态大型语言模型的VideoAfford方法。

**💡 创新点**

①提出首个利用HOI视频进行3D可供性推理的任务和大规模数据集VIDA；②将多模态LLM与潜在动作编码器和空间约束损失融合，提升动态交互理解和空间连续性。

**🔧 技术方法**

采用Video‑LLaVA的多模态LLM框架、RenderNet潜在动作编码器、预训练3D编码器（Uni3D）以及自定义空间感知Dice损失，结合LoRA微调。

**📊 数据集**

公开的VIDA数据集，包含约38K个人机交互视频、22K个3D点云、16个可供性类别和38个对象类别。

**📈 对比分析**

与XMF、PFusion、IAGNet、GREAT等基线比较，VideoAfford在Seen/Unseen设置下的mIoU、AUC、SIM和MAE均显著优于对手，特别是8帧采样下的mIoU提升至28.20%/10.95%。

**⚠️ 局限性**

仍受限于视频帧采样数量与计算开销，缺乏对更复杂动态场景和多模态融合深度的评估，且模型对超出训练范围的对象仍表现欠佳。

---

## 332. Towards Training-free Multimodal Hate Localisation with Large Language Models

**arXiv ID:** 2602.09637 | [PDF](https://arxiv.org/pdf/2602.09637v1)

**作者:** Yueming Sun `[一作]` (University of Durham), Zeyu Fu `[通讯]` (University of Exeter)

**通讯引用:** 714 | [OpenAlex ID](https://openalex.org/A5012939991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了LELA框架，实现了训练无关的视频仇恨内容的帧级定位；

**💡 创新点**

采用多模态字幕分解+多阶段提示，使LLM能够在无需标注训练的情况下进行细粒度情感评估；

**🔧 技术方法**

使用多模态caption模型（BLIP‑2、EasyOCR、Whisper、LP‑Music‑Caps、PDVC）生成字幕，构建多阶段提示（情境化→推理→评分），并利用LLM（如GPT‑4o Mini）进行评分与组合匹配；

**📊 数据集**

在HateMM和MultiHateClip（MHC）两个公开基准数据集上进行实验；

**📈 对比分析**

与多种训练无关基线（CLIP、ImageBind、LLAVA、LAVAD等）以及部分有监督方法对比，LELA在ROC‑AUC、PR‑AUC、准确率、F1等指标均明显优于所有零样本方法，并接近或超过部分有监督模型；

**⚠️ 局限性**

仍受LLM对长视频描述质量、跨模态对齐精度以及阈值敏感度等限制，且对不同语言和文化语境的鲁棒性有待进一步验证。

---

## 333. ISO FastLane: Faster ISO 11783 with Dual Stack Approach as a Short Term Solution

**arXiv ID:** 2602.09633 | [PDF](https://arxiv.org/pdf/2602.09633v1)

**作者:** Timo Oksanen `[一作]` (Technical University of Munich), Timo Oksanen `[通讯]` (Technical University of Munich)

**通讯引用:** 2783 | [OpenAlex ID](https://openalex.org/A5021744500)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文设计并实现了一种ISO FastLane双栈方案，保留CAN总线的广播功能，利用以太网对ISOBUS点对点流量进行封装与路由，显著提升数据传输速率。

**💡 创新点**

创新点在于：1）无网关、无应用层改动的实现方式；2）在CAN地址申领中加入AACL（Augmented Address Claim）协议共享IP地址；3）通过简单的层3路由决策和UDP封装CAN帧，实现即时的点对点转发。

**🔧 技术方法**

使用技术包括：SAE J1939/ISO 11783协议栈、TP/ETP传输协议、UDP/IPv4（可扩展为IPv6）、以太网物理层、AACL消息与SA值252/253的使用、以及标准CAN硬件抽象层。

**📊 数据集**

实验数据集主要来自：实际的250 kbit/s CAN总线硬件（USB‑CAN适配器）和PC上的本地UDP socket；进行Token‑Ring Relay和双向高频率压力测试，测试文件包括1 MB ETP传输、5 KB、50 KB等多种负载。

**📈 对比分析**

与仅CAN模式对比，FastLane将TP/ETP传输速率提升至7.9倍（1 MB从5.2 min降至42 s）；双向压力测试可持续80 000 msg/s/节点、总吞吐率达86×CAN总线容量，且零丢包，展示了显著的性能优势。

**⚠️ 局限性**

limitations：1）仅支持具备CAN硬件的ECU；2）每个CAN帧封装后约60 字节的协议开销；3）不支持多播/广播、无线，仅单播UDP；4）缺乏QoS与高级安全机制；5）以太网失效时回落到CAN，双失效导致无通信；6）需在未来标准化并放宽速率限制。

---

## 334. Free-GVC: Towards Training-Free Extreme Generative Video Compression with Temporal Coherence

**arXiv ID:** 2602.09868 | [PDF](https://arxiv.org/pdf/2602.09868v1)

**作者:** Xiaoyue Ling `[一作]` (Institute of Image Communication and Network Engineering Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Institute of Image Communication and Network Engineering Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Free‑GVC，一种训练‑free 的生成式视频压缩框架，将视频编码视为在扩散空间中的潜在轨迹压缩。

**💡 创新点**

创新点包括：自适应质量控制模块、跨 GOP 对齐机制以及直接利用预训练视频扩散模型，无需再训练即可在超低比特率下实现高感知质量与时间一致性。

**🔧 技术方法**

技术栈涵盖：扩散模型（CogVideoX‑2B）、VAE、逆向通道编码（RCC）与Poisson功能表示（PFR）等。

**📊 数据集**

使用公开数据集 HEVC Class B∼E、UVG 与 MCL‑JCV 进行评估。

**📈 对比分析**

与传统 HEVC/VVC、神经码流 DCVC 系列以及生成式 GLC‑Video 对比，Free‑GVC 在 DISTS、LPIPS、FID 等感知指标上平均节约 93.29% BD‑Rate，并在用户研究中在相同或更低码率下优于对手。

**⚠️ 局限性**

局限在于实时性能不足，尤其是高分辨率视频的编码速度相对慢，需要进一步优化 RCC 与扩散加速。

---

## 335. Differentiable Tripartite Modularity for Clustering Heterogeneous Graphs

**arXiv ID:** 2602.09864 | [PDF](https://arxiv.org/pdf/2602.09864v1)

**作者:** Benoît Hurpeau `[一作]` `[通讯]`, Benoît Hurpeau

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种可微分的三元组网络模数（tripartite modularity），实现端到端无监督聚类；

**💡 创新点**

创新点在于：1）通过共路径（co‑paths）对三元组图定义模数；2）利用因式分解避免构造高阶张量；3）引入枢轴节点流量归一化以消除极端度数偏差；4）在 DMoN 框架中加入软匹配（soft correspondence）实现可微分；

**🔧 技术方法**

使用的技术包括：图神经网络（GNN）编码器、可微分模数优化、共路径因式分解、热力图匹配（softmax 温度 β）、流量归一化权重 ω_j、折叠惩罚（collapse regularization）和贝叶斯超参搜索；

**📊 数据集**

数据集为法国 Hauts‑de‑Seine 区（D092）的城市地籍数据，包含 308,241 栋建筑、地址与地块三元组；

**📈 对比分析**

与传统基于双向图或离散三元组模数方法对比，DMoN‑3p 在大规模异构图上实现线性复杂度，收敛稳定，获得 48 个活跃社区，显示出空间连贯性，且在无监督条件下性能优于基准；

**⚠️ 局限性**

局限性包括：1）仅在单一城市案例验证，缺乏跨域泛化实验；2）需要手动设置温度 β 与归一化权重，敏感度需进一步研究；3）对极端稀疏或高度分散的数据结构效果尚未评估。

---

## 336. Code2World: A GUI World Model via Renderable Code Generation

**arXiv ID:** 2602.09856 | [PDF](https://arxiv.org/pdf/2602.09856v1)

**作者:** Yuhao Zheng `[一作]` (University of Science and Technology of China), Kevin Qinghong Lin `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Code2World，基于可渲染的 HTML 代码生成实现 GUI 世界模型，能够预测下一个界面截图；

**💡 创新点**

创新点在于将 GUI 预测视为代码生成任务，利用结构化代码实现高保真视觉合成与结构可控；

**🔧 技术方法**

采用 Vision‑Language 模型，先用监督微调获取 HTML 语法，再通过 Render‑Aware Reinforcement Learning 用渲染结果作为奖励优化视觉与逻辑一致性；

**📊 数据集**

构建 AndroidCode 数据集（约 80K 条屏幕-动作对），来源于 AndroidControl 轨迹并通过 GPT‑5 合成并可视化反馈循环校正；

**📈 对比分析**

与多种像素生成模型（Gemini‑3‑Pro‑Image 等）和代码生成模型（GPT‑5、Claude‑4.5‑Sonnet 等）对比，Code2World‑8B 在视觉与功能逻辑指标上均领先，并显著提升下游导航成功率（+9.5%）；

**⚠️ 局限性**

局限在于对代码生成的渲染依赖；若生成代码不完整或渲染不一致，可能导致逻辑错误；此外对跨域应用的泛化仍受限，需进一步提升模型鲁棒性。

---

## 337. 6G NTN Waveforms: A Comparison of OTFS, AFDM and OCDM in LEO Satellite Channels

**arXiv ID:** 2602.09834 | [PDF](https://arxiv.org/pdf/2602.09834v1)

**作者:** Baidyanath Mandal `[一作]` (National Institute of Technology), Cezary Ziołkowski `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

比较OTFS、AFDM和OCDM三种非正交多载波波形在LEO卫星链路中的误码率表现；

**💡 创新点**

首次在同一MATLAB仿真框架下，将三种波形在3GPP NTN TDL模型下进行公平对比，并提出MMSE‑SD检测提升BER；

**🔧 技术方法**

利用离散仿射傅里叶变换（IDAFT/DAFT）、离散弗伦斯变换（IDFnT/DFnT）和ISFFT，配合LMMSE与MMSE‑SD检测算法；

**📊 数据集**

使用3GPP TR 38.811规定的四种NTN TDL（TDL‑A、B、C、D）模型，模拟LEO卫星（600 km，高度）与地面UE（500 km/h）场景；

**📈 对比分析**

通过相同SNR、调制阶数（16‑QAM）和帧长条件下的BER曲线比较，结果显示AFDM在所有模型中均优于OTFS，OTFS略优于OCDM，且MMSE‑SD显著降低AFDM/OTFS的SNR需求；

**⚠️ 局限性**

仅在仿真环境中验证，未考虑实际链路的硬件限制、时变相位噪声及多波束协同等复杂因素。

---

## 338. LLM Reasoning Predicts When Models Are Right: Evidence from Coding Classroom Discourse

**arXiv ID:** 2602.09832 | [PDF](https://arxiv.org/pdf/2602.09832v1)

**作者:** Bakhtawar Ahtisham `[一作]` (Cornell University), Rene F. Kizilcec `[通讯]` (Cornell University)

**通讯引用:** 7070 | [OpenAlex ID](https://openalex.org/A5071778778)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用LLM生成的推理文本作为辅助信号，训练监督分类器来判断其所给标签是否正确，从而构建一种可扩展的错误检测层；

**💡 创新点**

创新点在于：①将LLM推理视为可利用的错误检测信息，而非仅作辅助说明；②通过LIWC语言特征解析推理内容，揭示错误与正确推理的语义差异；③针对不同教学动作构造构建专用检测器，提升对复杂构造的识别效果；

**🔧 技术方法**

技术手段包括：TF‑IDF特征抽取、五种监督分类器（Random Forest、Logistic、SVM、GBM、Naïve Bayes）对推理进行误判预测、LIWC语言特征分析、构造特定模型训练与比较；

**📊 数据集**

使用的数据集为Talk Moves课堂对话语料，包含约30,300条教师发言，分别由四大LLM（GPT‑5、Claude 4.5、Gemini 2.5、o3）生成标签与推理；

**📈 对比分析**

方法评估：将数据按80/20划分，使用交叉验证调参，比较五种分类器，Random Forest以F1≈0.83（Recall 0.854）表现最佳；在所有LLM上F1>0.67，且构造特定模型在难度较高的构造上显著提升；

**⚠️ 局限性**

局限性：仅在数学教学语境下验证，推理质量高度依赖提示与模型版本；采用传统TF‑IDF忽略深层语义；未对其他学科或不同提示策略的迁移性能进行评估。

---

## 339. Internalizing Multi-Agent Reasoning for Accurate and Efficient LLM-based Recommendation

**arXiv ID:** 2602.09829 | [PDF](https://arxiv.org/pdf/2602.09829v1)

**作者:** Yang Wu `[一作]` (Tencent), Jie Jiang `[通讯]` (Tencent)

**通讯引用:** 1517 | [OpenAlex ID](https://openalex.org/A5101944041)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将多代理教师系统的工具使用与自我反思等推理能力压缩为单一模型，实现了高效、可解释的推荐

**💡 创新点**

提出协同信号翻译机制和轨迹驱动蒸馏管线，将多代理推理迁移到单模型，并在教师模型基础上反而取得更好性能

**🔧 技术方法**

利用多代理（Planner、执行器、反射器）、协同信号翻译工具、SFT+GRPO（轨迹驱动蒸馏）以及预先生成的自然语言证据

**📊 数据集**

Amazon、Goodreads、Yelp 三大电商/书评/本地服务数据集

**📈 对比分析**

与传统MF、LightGCN以及多种agentic baseline比较，STAR在 Classic/Cold-Start/Evolving-Interest 三个场景均超过教师模型 8.7–39.5% 的 HitRate，并且推理延迟大幅下降

**⚠️ 局限性**

推理仍超 0.1s 延迟，离线自然语言化过程存储成本高，且仅在 20 项候选集上验证，缺乏全量检索性能与跨垂直的通用性

---

## 340. PlugSI: Plug-and-Play Test-Time Graph Adaptation for Spatial Interpolation

**arXiv ID:** 2602.09824 | [PDF](https://arxiv.org/pdf/2602.09824v1)

**作者:** Xuhang Wu `[一作]` (Harbin Engineering University), Sumi Helal `[通讯]` (University of Bologna)

**通讯引用:** 8715 | [OpenAlex ID](https://openalex.org/A5028246366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出PlugSI，一种用于时空插值（SI）的测试时图适配框架，能在无训练数据、无需改动预训练模型的前提下，在线地对未知且不断变化的图结构进行自适应修正；

**💡 创新点**

创新点在于：①设计未知拓扑适配器（UTA）通过虚拟节点不确定度评分和邻接修正实现对虚拟节点拓扑的高效调整；②引入时间平衡适配器（TBA）通过历史信息加权平滑当前批次的调整，避免噪声漂移；③整体框架为plug‑and‑play，可无缝集成进现有基于图的SI模型；

**🔧 技术方法**

技术包括：自监督学习（利用弱/强数据增强对齐损失）、图神经网络提取上下文表示、两层MLP生成不确定度评分与邻接调整矩阵、在线梯度优化以及小批次时间序列处理；

**📊 数据集**

使用四个公开数据集：交通速度（METR‑LA、PEMS07）、空气质量（AQI‑36）和太阳能发电（NREL‑AL），覆盖不同规模与时空粒度；

**📈 对比分析**

在七种主流基于图的SI模型（KCN、IGNNK、SATCN、INCREASE、DualSTN、KCP、KITS）上进行对比实验，PlugSI在所有数据集和模型上均实现MAE、MRE、MAPE等指标的显著提升，平均提升约10‑15% MAE；

**⚠️ 局限性**

局限性：只针对可建图的传感器网络，未验证非图结构场景；适配协议仅考虑小批次即时流式输入，未充分利用跨批次的长期上下文；在极端虚拟节点比例或聚类分布下，改进幅度仍有限。

---

## 341. Text summarization via global structure awareness

**arXiv ID:** 2602.09821 | [PDF](https://arxiv.org/pdf/2602.09821v1)

**作者:** Jiaquan Zhang `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 111203 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于拓扑数据分析（TDA）的全局结构感知文本摘要框架GloSA-sum，利用一次性持久同调识别语义簇和逻辑循环构成保护池，然后通过轻量化代理度量进行迭代压缩，既能高效处理长文本，又能保持语义和逻辑连贯。

**💡 创新点**

创新点包括①首次将TDA引入文本摘要，使用持久同调提取H0语义簇与H1逻辑循环；②构建一次性保护池一次性保存全局骨干，避免反复高成本TDA；③提出Topology‑Guided迭代压缩与分层策略，兼顾局部与全局结构。

**🔧 技术方法**

技术主要包括：句子语义向量编码、k近邻图构建、混合权重（语义+位置）、持久同调（Lazy Witness Complex）一次性计算、保护池、基于最短路径的TopoScore、任务相关的TaskScore（语义相似+BM25）、层次化分块与并行化。

**📊 数据集**

实验数据集包含GovReport、ArXiv、PubMed、CNN/DailyMail等长文本数据。

**📈 对比分析**

与TextRank、LexRank、Lead‑3、BERTSum、MatchSum、MemSum、BART、PEGASUS、BigBird、DANCER等基线对比，GloSA‑sum在ROUGE‑L、ROUGE‑2等指标均领先或相当，效率方面比大多数生成式模型快6–8倍，人工评估显示逻辑连贯与信息完整性优于或匹配现有方法。

**⚠️ 局限性**

局限性包括：对句子编码器的依赖，极短文本或特殊结构下效果不明显；保护池大小、k近邻阈值等超参数需手动调节；TDA计算在极大规模文本上仍有可扩展性挑战；跨语言、跨领域泛化仍待进一步验证。

---

## 342. Efficient Unsupervised Environment Design through Hierarchical Policy Representation Learning

**arXiv ID:** 2602.09813 | [PDF](https://arxiv.org/pdf/2602.09813v1)

**作者:** Dexun Li `[一作]` (Singapore Management University), Pradeep Varakantham `[通讯]` (Singapore Management University)

**通讯引用:** 2770 | [OpenAlex ID](https://openalex.org/A5089113099)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于层级马尔可夫决策过程的无监督环境设计框架 SHED，能够在教师-学生交互受限的情况下，通过教师生成针对学生当前能力的训练环境来提升学生的零样本迁移性能。

**💡 创新点**

创新点在于：①用学生在评估环境上的表现向量来刻画学生能力，从而构建层级 MDP 教师策略；②引入扩散模型生成合成的学生进化轨迹，显著降低教师收集真实经验的成本。

**🔧 技术方法**

使用的技术包括层级强化学习（MDP + DDPG）、扩散概率模型（Diffusion）用于合成数据、PPO 训练学生、评估环境分布设计以及基于学习进步与公平度的教师奖励函数。

**📊 数据集**

实验数据集为仿真环境：Lunar Lander、Bipedal Walker 以及 Maze，分别使用自动生成的评估/测试环境集合，未使用公开真实数据集。

**📈 对比分析**

通过与 Domain Randomization、ACCEL、编辑后的 ACCEL、PAIRED、以及无扩散的 h‑MDP 进行对比，SHED 在 50 次教师交互预算下实现了更高的零样本性能、更低的方差和更好的稳健性，明显优于基线方法。

**⚠️ 局限性**

局限性包括：对高维教师动作空间的可扩展性不如纯随机化方法；初始教师训练需要较大环境预算；假设仿真与真实环境的学习动力学相似，尚需进一步验证。

---

## 343. Flexible Entropy Control in RLVR with Gradient-Preserving Perspective

**arXiv ID:** 2602.09782 | [PDF](https://arxiv.org/pdf/2602.09782v1)

**作者:** Kun Chen `[一作]` (University of Chinese Academy of Sciences), Wenji Mao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 3133 | [OpenAlex ID](https://openalex.org/A5035983004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究RLVR中因梯度裁剪导致的熵崩溃问题，提出基于梯度保持的动态熵控制机制及三种熵调度策略

**💡 创新点**

将梯度保持裁剪阈值与熵变化的关系理论化，并提出非线性动态上下阈值调节与增减增减/振荡衰减三种熵调度方案

**🔧 技术方法**

采用PPO‑Clip改进、动态阈值调节、优势标准化、熵正则化等技术，并在LLM上实现梯度裁剪与熵控制的联合优化

**📊 数据集**

在DAPO‑MATH、AIME24/25、GSM8k、AMC、MATH‑500、Olympiad等数学推理数据集上进行训练与评估

**📈 对比分析**

与GRPO、Clip‑Higher、Clip‑Lower、Entropy‑Reg、Clip‑Cov、GSPO、SAPO等基线对比，Ours‑ID在多项基准上平均提升约3‑5分，表现显著优于传统方法

**⚠️ 局限性**

仅在LLM数学推理任务验证，阈值设计仍需经验调整，对训练时间与算力有额外开销，且对不同任务的通用性待进一步验证

---

## 344. Self-Supervised Learning as Discrete Communication

**arXiv ID:** 2602.09764 | [PDF](https://arxiv.org/pdf/2602.09764v1)

**作者:** Kawtar Zaher `[一作]` (Institut National de l'Audiovisuel), Alexis Joly `[通讯]` (INRIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出将自监督学习视为教师与学生之间的离散通信，通过固定容量的二进制通道传递语义信息，学习结构化的多标签表示；

**💡 创新点**

创新点在于用多标签二进制消息替代连续对齐，结合编码率正则化与周期性投影头重置，实现对表示维度分布与容量的显式控制；

**🔧 技术方法**

使用教师‑学生框架、元素级二进制交叉熵损失、编码率正则化、Sigmoid概率化与硬阈值二进制目标、周期性重置投影头；

**📊 数据集**

在ImageNet‑1K上训练，并在ImageNet‑V2、ImageNet‑100、PascalVOC、COCO、Birds525、Food101、iNat2019、PlantNet等数据集评估下游任务；

**📈 对比分析**

与DINO、SimDINO等连续对齐方法相比，BITS在图像分类、检索、目标检测、实例分割、视频分割和域迁移的线性探针性能均有提升，尤其在检索和轻度域漂移场景表现最显著；

**⚠️ 局限性**

局限性包括：对投影头重置频率敏感；在严重域漂移下冻结模型时，重置策略会降低鲁棒性；目前仅使用二进制通道，未探讨更丰富的离散语言。

---

## 345. CompSplat: Compression-aware 3D Gaussian Splatting for Real-world Video

**arXiv ID:** 2602.09816 | [PDF](https://arxiv.org/pdf/2602.09816v1)

**作者:** Hojun Song `[一作]` (Kyungpook National University), Sang-hyo Park `[通讯]` (Kyungpook National University)

**通讯引用:** 5661 | [OpenAlex ID](https://openalex.org/A5083103168)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究在压缩长视频上的3D Gaussian Splatting（3DGS）重建与新视角合成

**💡 创新点**

提出压缩感知的质量导向稠密度控制与质量间隙遮蔽两大技术，显著提升压缩视频的几何稳定性和渲染质量

**🔧 技术方法**

采用3DGS框架、帧级压缩质量估计（QP与比特率）、指数移动平均、可调稠密化/裁剪阈值、关键点基遮罩等方法

**📊 数据集**

使用Tanks & Temples、Free、Hike公开数据集以及真实移动设备压缩视频进行实验

**📈 对比分析**

与CF-3DGS、NoPe-NeRF、LocalRF、LongSplat等基线比较，压缩条件下PSNR提升约1–2 dB，SSIM提升0.02–0.04，姿态误差下降50%以上，整体性能达标最优

**⚠️ 局限性**

对极低码率或严重噪声压缩仍存在局限，且对压缩信息的依赖导致在未知编码场景下效果可能受限

---

## 346. CoFEH: LLM-driven Feature Engineering Empowered by Collaborative Bayesian Hyperparameter Optimization

**arXiv ID:** 2602.09851 | [PDF](https://arxiv.org/pdf/2602.09851v1)

**作者:** Beicheng Xu `[一作]` (Peking University), Bin Cui `[通讯]` (Peking University)

**通讯引用:** 12950 | [OpenAlex ID](https://openalex.org/A5062357883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 CoFEH 框架，通过 LLM 驱动的树状思考 (Tree‑of‑Thought) 进行自由形式特征工程，并与贝叶斯优化 (Bayesian Optimization) 的超参数搜索进行协同迭代，实现端到端的 AutoML。

**💡 创新点**

创新点包括：① 用 LLM 的树形思考结合 MCTS 生成可变结构的特征工程流水线；② 引入互相条件化机制，让特征工程与超参数优化共享上下文；③ 用 PUCB 多臂赌博机动态调度 FE 与 HPO 的预算，自动实现任务自适应资源分配。

**🔧 技术方法**

技术手段：大语言模型 (Gemini‑2.0‑flash) + 结构化提示；Tree‑of‑Thought + MCTS 搜索；贝叶斯优化（随机森林 surrogate + EI acquisition）；元特征映射 + 逆向检索；PUCB 动态选择器。

**📊 数据集**

实验数据集：28 个公开表格数据集，19 个分类（Grinsztajn 公开集）+ 9 个回归（OpenML、Kaggle）。下游模型包含 XGBoost、MLP、以及 CASH 多算法搜索。

**📈 对比分析**

与传统 FE 方法（OpenFE、Mindware）及 LLM‑基础基线（OCTree、ELLM‑FT、LFG）对比；在独立 FE 任务中平均排名提升至 1.84，联合 FE+HPO 任务提升至 1.75，平均性能提升约 7%；在不同下游模型上同样保持领先，显著降低错误率（最高可达 45%）。

**⚠️ 局限性**

局限性：依赖特定 LLM 与 BO 实现，模型成本与推理时延仍显高；目前仅针对表格数据，需进一步扩展到图像/文本；提示设计与元特征选择对性能影响较大，需更多实验验证其鲁棒性。

---

## 347. Kelix Technique Report

**arXiv ID:** 2602.09843 | [PDF](https://arxiv.org/pdf/2602.09843v1)

**作者:** Boyang Ding `[一作]`, Ziqi Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Kelix，一种完全离散的自回归统一多模态模型，可同时完成视觉理解与图像生成。

**💡 创新点**

创新点在于提出多通道离散化量化方案大幅提升视觉 token 的信息容量，并使用 Next‑Block Prediction 机制在不扩展序列长度的前提下保持训练与推理效率，成功弥合了离散与连续表示的性能差距。

**🔧 技术方法**

技术包括基于 ViT 的多子空间 VQ 量化、Qwen3‑8B 作为 LLM 主干、块级编码/解码模块、以及基于 DiT 的扩散式图像去量化器；此外采用 1D RoPE 与二维坐标扩展的视觉定位策略。

**📊 数据集**

训练数据覆盖 1T 以上 token 的多模态语料，包括大规模图文对（LAION、COYO 等）、OCR（Handwritten、CodeOCR、SynthDoG 等）、VQA（ShareGPT‑4o、Docmatix 等）、STEM 与文本生成（Wikipedia、Arxiv、OpenCodeReasoning 等）以及专门构造的高细粒度图像生成语料。

**📈 对比分析**

通过与同规模离散、连续和混合 token 化的统一模型以及专用图像生成模型在 OCRBench、MathVista、GenEval、WISE、DPG‑Bench 等基准上对比，Kelix 在理解任务上达到与连续特征 VLM 并列，在生成任务上实现了 SOTA，尤其在 OCR 和文本丰富的推理任务上显著优于先前的离散模型。

**⚠️ 局限性**

局限性包括：对大规模预训练数据与算力需求较高，部分高复杂度生成任务仍略逊于大型混合模型；多通道量化虽提升信息容量，但在极端细粒度视觉细节上仍可能出现信息损失；模型在多模态推理与生成的跨域泛化还有提升空间。

---

## 348. How Do People Quantify Naturally: Evidence from Mandarin Picture Description

**arXiv ID:** 2602.09838 | [PDF](https://arxiv.org/pdf/2602.09838v1)

**作者:** Yayun Zhang `[一作]` (Central China Normal University), Tingting He `[通讯]` (Central China Normal University)

**通讯引用:** 2489 | [OpenAlex ID](https://openalex.org/A5039602194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了普通话说话者在自然场景描述任务中对数量的表达，包括是否量化、量化精度以及采用的量化策略。

**💡 创新点**

创新在于在无约束、自然生产条件下探讨量化决策，并区分集体化与标度化模糊表达的策略差异。

**🔧 技术方法**

采用混合效应逻辑回归模型分析量化行为，并利用Whisper进行语音转写与人工注释。

**📊 数据集**

使用200幅颜色图像（VAQUUM数据集，数量2-99）及50名母语者产生的2000条口头与书面描述。

**📈 对比分析**

通过对量化出现、精度和策略的逻辑回归比较，发现数量、可动性和语料模式显著影响量化，模型给出显著的赔率比。

**⚠️ 局限性**

局限在样本规模、低频预测式表达未单独分析、仅限汉语、以及仅图像描述场景的普适性不足。

---

## 349. SAKED: Mitigating Hallucination in Large Vision-Language Models via Stability-Aware Knowledge Enhanced Decoding

**arXiv ID:** 2602.09825 | [PDF](https://arxiv.org/pdf/2602.09825v1)

**作者:** Zhaoxu Li `[一作]` (Nanyang Technological University), Xudong Jiang `[通讯]` (Nanyang Technological University)

**通讯引用:** 15516 | [OpenAlex ID](https://openalex.org/A5085533260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练无关、可插拔的解码策略SAKED，利用内部知识稳定性评估来抑制大型视觉‑语言模型的幻觉生成。

**💡 创新点**

创新点包括：①从注意力头、层级、token三维度系统性分析知识不稳定与幻觉的关联；②提出三种稳定性度量CHSS、CLSS、CTSS并聚合为知识稳定性得分KSS；③基于KSS动态挑选最稳定与最不稳定层进行对比解码，抑制生成噪声。

**🔧 技术方法**

核心技术有：视觉激活一致性度量、SoftIoU、Jensen‑Shannon Divergence、对比解码、动态层选择与token修正；无额外训练或外部知识。

**📊 数据集**

使用了多种公开基准：CHAIR（图像描述幻觉）、POPE（物体问答幻觉）、MME（多模态问答）、AMBER（多维度幻觉评估），并在五种主流LVLM上进行实验。

**📈 对比分析**

与贪心、Beam、Dola、OPERA、Deco、SAVER等现有方法对比，SAKED在所有模型和基准上实现了最低幻觉率和最高F1/整体分数，稳健地超过SOTA。

**⚠️ 局限性**

局限性包括：对不同模型结构仍需手动选择候选层集合；对参数α、β、λ的调优依赖实验；未对生成多样性和质量平衡进行深入分析；在极端分布外场景的鲁棒性尚未验证。

---

## 350. Symbolic Pattern Temporal Numeric Planning with Intermediate Conditions and Effects

**arXiv ID:** 2602.09798 | [PDF](https://arxiv.org/pdf/2602.09798v1)

**作者:** Matteo Cardellini `[一作]` (Università di Genova), Enrico Giunchiglia `[通讯]` (Università di Genova)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文将符号模式规划方法扩展到含有中间条件与效果的时间数字规划，提出一种新的Satisfiability编码与模式生成与压缩技术，形成完整且可证明的求解框架。

**💡 创新点**

核心创新包括：①将中间条件与效果纳入模式的概念与实现；②设计了滚动（rolling）机制以处理可重复执行的时序动作；③在符号模式规划框架中实现时间与数值的联合约束，并给出完整性与正确性证明。

**🔧 技术方法**

采用规划为可满足性（Planning‑as‑Satisfiability）与SMT 求解器Z3；在此基础上实现符号模式规划（Symbolic Pattern Planning）与滚动技术，利用SMT 约束语言（smt‑lib）编码时间、数值与模式约束。

**📊 数据集**

实验使用了八个公开域（Cushing、Pour、Shake、Pack、Match、Oversub、InStation、Majsp、Painter）以及自定义火车调度域，涵盖了多种时间与数值交互特性。

**📈 对比分析**

与搜索基（tfd、lpg）及其他可满足性基（anml、itsat）规划器在5分钟限制下对比，本文方法在190个实例中最多解168例，平均耗时与步数均优于对手；在InStation与Oversub等域能在绑定n=1即单步完成。

**⚠️ 局限性**

主要局限在于需每次重计算完整模式导致求解前置成本高；对非well‑orderable域模式不完整；滚动与约束求解的可扩展性尚待进一步验证。

---

## 351. GHS-TDA: A Synergistic Reasoning Framework Integrating Global Hypothesis Space with Topological Data Analysis

**arXiv ID:** 2602.09794 | [PDF](https://arxiv.org/pdf/2602.09794v1)

**作者:** Jiaquan Zhang `[一作]` (University of Electronic Science and Technology of China), Yang Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 111203 | [OpenAlex ID](https://openalex.org/A5100397455)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建全局假设图（GHS）以整合多条推理路径，然后利用拓扑数据分析（TDA）提取稳定的推理骨干与自洽环路，从而生成高置信度、可解释的推理链。

**💡 创新点**

提出两阶段“构造–分析”框架GHS‑TDA，首次将全局假设空间与持久同调相结合，以结构稳定性驱动推理选择；同时引入多角色议程机制和图嵌入+距离度量来构造统一图。

**🔧 技术方法**

全局假设图构建（节点合并、语义对齐、支持/驳斥边）、多角色议程（explorer/validator/bridge）、图嵌入与距离加权、Vietoris–Rips 过滤、持久同调（H0、H1）、循环基础构造、置信/持久性加权投票。

**📊 数据集**

GSM8K、MATH、OlympiadBench、BBH、MMLU‑CF、LongBench、HotpotQA、MuSiQue。

**📈 对比分析**

与九种基线（CoT、CoT‑SC、Self‑Refine、Analogical Prompting、AFlow、ToT、GoT、FoT、AoT）在三种大模型（GPT‑4o‑mini、Qwen‑Turbo、DeepSeek‑V3）上对比，平均准确率分别提升至68.0%、67.6%、68.3%，相较基线提升约1–3个百分点；鲁棒性测试显示错误率下降；效率方面，LLM 调用次数固定为19，显著低于其他多路径方法。

**⚠️ 局限性**

仍依赖大量样本采样与阈值设定；对超参数（合并阈值、距离权重）敏感；在极端语言多样性或非常长推理链时持久同调计算成本上升；未充分验证跨语言/领域的泛化。

---

## 352. Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path

**arXiv ID:** 2602.09784 | [PDF](https://arxiv.org/pdf/2602.09784v1)

**作者:** Andres Saurez `[一作]` (Korea Advanced Institute of Science and Technology), Dongsoo Har `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1782 | [OpenAlex ID](https://openalex.org/A5005393869)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Circuit Fingerprint方法，利用答案令牌在激活空间的几何指纹进行电路发现与激活调节。

**💡 创新点**

将电路发现与激活调节统一为同一几何结构的读写操作，证明二者共享相同的几何指纹。

**🔧 技术方法**

采用几何对齐、答案向量投影、Shapley分解等线性几何技术实现无梯度电路识别与方向写入。

**📊 数据集**

使用IOI、SVA、MCQA等标准电路基准以及情绪、语言等自定义指令提示作为数据集。

**📈 对比分析**

与梯度基线EAP、EAP‑IG等比较，CMD/CPR指标相当，情绪分类准确率提升至69.8%（对比53.1%），并在大部分模型与任务上保持与梯度方法相近的效果。

**⚠️ 局限性**

对负向情绪和跨语言特征的调节易导致事实失真；仅关注最终标记，未考虑位置级影响和LayerNorm非线性，且零样本调节仍不稳定。

---

## 353. Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints

**arXiv ID:** 2602.09783 | [PDF](https://arxiv.org/pdf/2602.09783v1)

**作者:** Andres Saurez `[一作]` (Korea Advanced Institute of Science and Technology), Dongsoo Har `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1782 | [OpenAlex ID](https://openalex.org/A5005393869)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文阐明了 transformer 通过线性接口（注意力中的 OV 线性映射和解码器的线性投影）传递语义特征的必要性，并提出Invariant Subspace Necessity theorem和Self-Reference Property，证明任何可通过线性接口解码的语义特征必须位于上下文不变的子空间；利用这一理论设计了零样本无监督探针以及稀疏自编码器方法，实现对八类语义分类任务的高效分类。

**💡 创新点**

创新点包括：
1) 将架构约束视为线性可解释性成功的根本原因，给出了Invariant Subspace Necessity theorem；
2) 提出了Self-Reference Property，即 token 本身即提供其语义方向，能够在无标注或无训练的情况下实现零样本分类；
3) 将稀疏自编码器与线性探针映射到同一理论框架，验证两者提取到的方向一致，进一步证明了理论的普适性。

**🔧 技术方法**

主要技术手段：
- 理论推导与几何分析（Invariant Subspace Necessity、Self-Reference Property、Identity-Projection 投影）；
- 线性探针与无监督对比学习（Zero-shot probe、Unsupervised probe）；
- 稀疏自编码器（SAE）用于从上下文中无监督地学习特征方向；
- 通过 attention head 输出进行实验评估。

**📊 数据集**

数据集与模型：
- 八类分类任务（Animals、Countries、Emotional Sentences、Literary Quotes、Cartoon Phrases、Languages、Fruits、Companies），每类包含若干隐含实例；
- Polysemy 任务（Apple 的果蔬与公司两义）；
- 训练与评估使用四大 transformer 族：LLaMA3-8B、LLaMA3.2-3B、Mistral-7B、GPT2-Small。

**📈 对比分析**

方法比较与性能：
- 对每个任务在每个模型上分别评估 Zero-shot、Unsupervised、SAE、Text Output 四种方法；
- 大模型（LLaMA、Mistral）在多数任务上均可达 80%+ 的准确率；Zero-shot 和 Unsupervised 在 90% 以上的任务较高，Unsupervised 通常略优于 Zero-shot；SAE 在部分任务与 Unsupervised 相近；
- GPT2-Small 的表现普遍较差，说明模型规模与容量对方向性的重要性；
- 文本输出 baseline 仅在极少任务表现不错。

**⚠️ 局限性**

limitations：
1) 仅分析通过线性接口传递的特征，未涵盖 QK 形式的非线性路由特征；
2) 对子空间维度（单维 vs 多维）何时出现缺乏定量阐述；
3) 需要更多不同规模与架构的实验来进一步验证理论的普适性；
4) 在 polysemy 任务中，仍存在一定误差，说明上下文对特征强度的调制不完全可预测。

---

## 354. Explainability in Generative Medical Diffusion Models: A Faithfulness-Based Analysis on MRI Synthesis

**arXiv ID:** 2602.09781 | [PDF](https://arxiv.org/pdf/2602.09781v1)

**作者:** Surjo Dey `[一作]` (Rajiv Gandhi Institute of Petroleum Technology), Pallabi Saikia `[通讯]` (Rajiv Gandhi Institute of Petroleum Technology)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5041797369)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了用于MRI合成的扩散生成模型的可解释性，提出了基于原型的可解释框架，并评估了其对生成过程的faithfulness

**💡 创新点**

将原型学习与扩散模型相结合，提出了基于faithfulness的可解释性评估方法，特别是EPPNet在单类医学图像中的高faithfulness表现

**🔧 技术方法**

扩散模型（U‑Net）、ProtoPNet、EPPNet、ProtoPool以及PSNR/SSIM/LPIPS和faithfulness评分等评估指标

**📊 数据集**

Duke Breast MRI数据集（11860张高分辨率乳腺MRI图像）

**📈 对比分析**

通过FID 1.39、Dice 0.9478、PSNR 19.37±1.67、SSIM 0.653±0.105、LPIPS 0.289±0.105，EPPNet在faithfulness得分0.1534最高，优于ProtoPool 0.1420和PPNet 0.0965

**⚠️ 局限性**

仅针对单类乳腺MRI，缺乏多模态、多类别验证，数据量相对有限，模型在更大规模多样化数据上的泛化能力尚未验证

---

## 355. Diverse Skill Discovery for Quadruped Robots via Unsupervised Learning

**arXiv ID:** 2602.09767 | [PDF](https://arxiv.org/pdf/2602.09767v1)

**作者:** Ruopeng Cui `[一作]` (Fudan University), Wei Li `[通讯]` (Fudan University)

**通讯引用:** 96012 | [OpenAlex ID](https://openalex.org/A5100318082)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文提出了 MOD-Skill 框架，用于在无监督条件下发现多样化的四足机器人行走技能。

**💡 创新点**

创新点在于引入多分辨率判别器来拆解观测子空间，防止奖励劫持，并使用正交混合专家（OMoE）网络强制专家特征正交，从而提高技能多样性与训练效率。

**🔧 技术方法**

主要技术包括无监督强化学习、PPO 算法、Gram–Schmidt 正交化、正交混合专家架构、多个判别器以及域随机化实现仿真到真实转移。

**📊 数据集**

实验数据来自 Isaac Sim 仿真中的 12 自由度 Unitree A1 四足机器人，随后在真实机器人上验证。

**📈 对比分析**

与单一判别器基线（SD1、SD2、SD3）相比，MOD-Skill 在状态空间覆盖率上提升 18.3%，学习曲线更快，并能成功在真实环境中执行。

**⚠️ 局限性**

局限性包括对特定机器人平台的依赖、判别器设计仍需手工设定观测子空间，以及在极端复杂地形或更高维度任务下的通用性待验证。

---

## 356. BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation

**arXiv ID:** 2602.09849 | [PDF](https://arxiv.org/pdf/2602.09849v1)

**作者:** Yucheng Hu `[一作]` (Tsinghua University), Jianyu Chen `[通讯]` (Tsinghua University)

**通讯引用:** 5383 | [OpenAlex ID](https://openalex.org/A5100611364)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 BagelVLA，一个统一的 Vision‑Language‑Action 框架，通过在同一 Transformer 系统中交错进行语言规划、视觉预测与动作生成，实现在复杂、长时程操作中的指令跟随。

**💡 创新点**

创新点：
- 将语言规划、视觉预测和动作生成三大能力交错嵌入单一模型，而非传统的模块化设计；
- 引入 Residual Flow Guidance（RFG）用当前帧作为噪声先验，显著降低视觉预测的推理延迟；
- 采用两阶段训练（先在大规模通用多模态+机器人演示上预训练，再加入动作标签微调），使模型既保留通用推理能力，又能获得精准低层控制。

**🔧 技术方法**

主要技术：
- Mixture‑of‑Transformers（MoT）架构，包含 LLM（Qwen2.5‑LLM‑7B）、视觉生成专家（SigLIP2、VAE‑FLUX）和动作专家（小型 Transformer）;
- 双重 Flow‑Matching 进行图像与动作的去噪；
- RFG 的噪声初始化与单步去噪；
- 采用自回归 Cross‑Entropy 损失、Flow‑Matching 损失等联合优化。

**📊 数据集**

使用的数据集：
- 通用多模态数据：VQA（298w 条问答对）、人手视觉动态演示（31k 片段）;
- 机器人数据：自采与公开的 1.1w 以上演示，包含语义子任务标注、关键帧；
- 真实世界实验：AgileX 双臂平台的 3000 条基本任务和 1500 条长时程任务，均有子任务与关键帧注释；
- 仿真环境：Calvin、Robotwin 等。

**📈 对比分析**

比较方法：
- 在 Calvin 和 Robotwin 进行基准测试，比较 π_0、RDT、UP‑VLA、VPP 等；
- 结果显示 BagelVLA 在 Calv 约 4.41 任务完成长度、在 Robotwin 纯文本规划 75.26%/20.87% 成功率，均高于所有基线；
- 在真实世界基本任务中，BagelVLA 取得 75.5% 的平均成功率，高于 π_0（65%）和 VPP（59.5%）；
- 长时程规划任务中，BagelVLA 的规划准确率接近 90%，任务成功率显著提升（如 95% vs 75%）。

**⚠️ 局限性**

局限性：
- 受限于动作专家的容量，细粒度执行精度仍有提升空间；
- 在某些 OOD 视觉场景下，完整去噪或联合去噪易出现中间状态漂移；
- 需要大规模多模态与机器人演示数据，训练成本高；
- 仍需进一步提升在极端动态环境或高度复杂关节控制任务中的鲁棒性。

---

## 357. Reason-IAD: Knowledge-Guided Dynamic Latent Reasoning for Explainable Industrial Anomaly Detection

**arXiv ID:** 2602.09850 | [PDF](https://arxiv.org/pdf/2602.09850v1)

**作者:** Peng Chen `[一作]` (Shenzhen Campus of Sun Yat-sen University), Xiaochun Cao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个名为Reason‑IAD的知识驱动动态潜在推理框架，用于解释性工业异常检测；

**💡 创新点**

创新点在于将检索增强的领域知识、熵驱动的潜在推理机制以及动态视觉注入策略融合到无训练的多模态大型语言模型中，实现了高效、可解释的异常推理；

**🔧 技术方法**

主要技术包括CLIP检索知识库、可优化的潜在思考标记、基于熵的奖励与REINFORCE优化、以及基于注意力的动态视觉补充；

**📊 数据集**

使用了MMAD基准（MVTec‑AD、VisA、MVTec‑LOCO、GoodsAD）以及公开的工业异常检测数据集进行评估；

**📈 对比分析**

与多种商业与开源多模态大型语言模型（如Claude‑3.5‑Sonnet、Gemini‑2.5‑Pro、GPT‑4o、Qwen‑系列、LLaVA、InternVL等）以及专门的异常检测模型对比，Reason‑IAD在一/零样本设置下平均提升4–7%准确率，并在多项子任务中超越人类标注者；

**⚠️ 局限性**

局限性包括对检索知识库质量的依赖、潜在推理过程对参数（噪声、迭代次数等）敏感，以及在极端复杂场景下可解释性细节仍有提升空间。

---

## 358. ARK: A Dual-Axis Multimodal Retrieval Benchmark along Reasoning and Knowledge

**arXiv ID:** 2602.09839 | [PDF](https://arxiv.org/pdf/2602.09839v1)

**作者:** Yijie Lin `[一作]` (Sichuan University), Xi Peng `[通讯]` (Sichuan University)

**通讯引用:** 9767 | [OpenAlex ID](https://openalex.org/A5022800038)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个双轴（知识与推理）多模态检索基准ARK，并在其上评估了23种检索模型

**💡 创新点**

首次将知识域和推理能力作为并列评估轴，精心设计硬负样本并引入细粒度视觉与空间推理挑战，揭示现有模型的弱点

**🔧 技术方法**

使用大规模多模态检索器、文本嵌入模型、LLM驱动的查询重写和重排序等技术；通过多模态LLaMA等模型实现查询改写与负样本挖掘

**📊 数据集**

ARK数据集：5大知识域（视觉认知、自然科学、形式科学、人文社科、工程技术），17细粒度子类，16种视觉类型（表格、图表、化学结构、漫画、3D渲染等），约1.5k查询与36k候选

**📈 对比分析**

评估以Recall@1为主，结果显示最强模型Recall@1仅低于20%；在知识密集型任务表现较好，推理密集型尤其是细粒度视觉与空间推理表现最差；查询重写+重排序可提升约3–5个百分点

**⚠️ 局限性**

仍缺乏针对复杂视觉推理的专门机制，模型受限于表面语义匹配，难以捕捉高细节/空间关系；规模扩大会带来提升但未能根除推理瓶颈，数据集对跨模态语义表达仍有待进一步丰富

---

## 359. From FusHa to Folk: Exploring Cross-Lingual Transfer in Arabic Language Models

**arXiv ID:** 2602.09826 | [PDF](https://arxiv.org/pdf/2602.09826v1)

**作者:** Abdulmuizz Khalak `[一作]` (Maastricht University), Gerasimos Spanakis `[通讯]` (Maastricht University)

**通讯引用:** 754 | [OpenAlex ID](https://openalex.org/A5010354377)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究阿拉伯语大模型在现代标准阿拉伯语（MSA）与各地区方言间的跨语言迁移效果，利用三项核心NLP任务（POS、NER、情感分析）进行功能性探针和结构性表示相似度（CKA）评估，并探讨地理邻近度与迁移/相似度的关系。

**💡 创新点**

创新点在于：① 将功能探针与层级表示相似度分析相结合，构建多维度评估框架；② 通过地理距离量化方言连贯性，首次将方言连续体假说与预训练模型迁移性能直接关联；③ 揭示多方言预训练模型在高资源方言上的负干扰现象，提示需考虑方言或地区专属参数。

**🔧 技术方法**

采用的技术包括：BERT‑style预训练模型（CAMeLBERT、AraRoBERTa 及各方言专用模型），线性探针（多项式回归）提取层级表示，Centered Kernel Alignment (CKA) 计算层间相似度，CAMeL‑Lab 方言识别与银标NER注释，Adam 优化器训练探针。

**📊 数据集**

使用的数据集有：MADAR 并行语料（MSA 与 25 城市方言对齐），各方言的 POS、NER、情感分析公开数据集；因 Gulf 方言无公开 NER 集，利用 CAMeL‑Lab NER 模型在 MADAR 上生成银标数据。

**📈 对比分析**

比较方法：在每个模型（MSA、MIX、多方言、单方言）上分别提取最佳层，计算宏 F1（POS/NER/SA），并将结果与 MSA 原始性能对比；同时在 MADAR 上计算 MSA 与方言模型的层级 CKA 相似度。实验表明：MSA 模型在结构任务（POS/NER）上优于多方言模型；单方言模型在情感任务上往往优于通用模型；CKA 结果显示 MSA 与方言模型的相似度随地理距离增大而下降，体现方言连续体效应。

**⚠️ 局限性**

局限性：① Gulf 方言 NER 缺乏公开数据，银标注可能带来噪声；② 采用 Yemen 作为 MSA 的地理代理仅为操作性选择，其他选择可能产生不同结论；③ 模型规模、词表及预训练超参数差异导致结果混杂；④ 方言模型中 MSA 数据混入或过滤不均可能影响迁移与相似度评估。

---

## 360. Fully-automated sleep staging: multicenter validation of a generalizable deep neural network for Parkinson's disease and isolated REM sleep behavior disorder

**arXiv ID:** 2602.09793 | [PDF](https://arxiv.org/pdf/2602.09793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 361. A Controlled Study of Double DQN and Dueling DQN Under Cross-Environment Transfer

**arXiv ID:** 2602.09810 | [PDF](https://arxiv.org/pdf/2602.09810v1)

**作者:** Azka Nasir `[一作]`, Mohammad Ahmed Atif `[通讯]` (Habib University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文对双重DQN（DDQN）和优势网络DQN（Dueling DQN）在跨环境迁移（从CartPole迁移到LunarLander）中的表现进行受控实验比较。

**💡 创新点**

创新点在于首次系统评估两种价值网络架构在面对显著域迁移时的迁移鲁棒性，并通过严格的实验设置展示DDQN能保持正迁移、Dueling DQN会出现负迁移。

**🔧 技术方法**

使用了DDQN与Dueling DQN两种架构，采用经验回放、软目标网络、Adam优化器、固定层迁移等标准深度强化学习技术。

**📊 数据集**

实验数据集为OpenAI Gymnasium中的CartPole和LunarLander两种控制任务。

**📈 对比分析**

比较方法为在源环境训练完毕后，将隐藏层权重迁移至目标环境，先冻结100步再解冻；与在目标环境从零训练的基线进行对比。结果显示：DDQN迁移后验证奖励和回合奖励均与基线相当甚至略优，训练损失平滑；Dueling DQN迁移后验证奖励明显下降，回合奖励为负，训练损失波动大，显现负迁移。

**⚠️ 局限性**

局限性包括：只做了500个回合的目标环境训练，可能不足以观察Dueling DQN最终是否收敛；迁移策略仅为直接层权重复用且仅在初期冻结，未探索更复杂的适配方法；样本量仅5个随机种子，统计功效有限；仅比较两种单一架构，未考虑混合或Rainbow等更先进网络。

---

## 362. SciFlow-Bench: Evaluating Structure-Aware Scientific Diagram Generation via Inverse Parsing

**arXiv ID:** 2602.09809 | [PDF](https://arxiv.org/pdf/2602.09809v1)

**作者:** Tong Zhang `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14730 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SciFlow‑Bench，一个结构优先的科学图表生成评估基准，通过将源论文的框架图与其对应的结构图配对，并使用多层代理系统将生成的图像反向解析为结构图，进而以结构可恢复性为核心指标评估文本到图像模型。

**💡 创新点**

创新点在于：①将结构可恢复性作为主评估维度，弥补了传统图像相似度指标对逻辑错误不敏感的缺陷；②构建了端到端的多层代理解析流水线，可在数据集构建与评估中复用；③在实际论文 PDF 生成的 500 份图表上构建了结构图，体现了真实科研图表的多样性和复杂性。

**🔧 技术方法**

技术方面采用了层次化多代理系统（规划、感知、结构推理），其中包括形状检测、文本识别、布局推断等模块；反向解析通过 Mermaid 语法和图形构建器完成；评估则在结构图空间内计算节点/边的 Precision/Recall/F1，并与文本级和图像级指标组合。

**📊 数据集**

使用了来自 2025 年 arXiv 论文的 500 份真实科学框架图，覆盖计算机视觉、NLP、机器学习理论、集成电路与机器人等五大领域；这些图与自动生成的结构图配对，构成了 SciFlow‑Bench 数据集。

**📈 对比分析**

与多种模型对比，包括基于代码的 Graphviz、开源扩散模型（SDXL、PixArt‑Σ、Qwen‑Image）以及商业自回归视觉语言模型（Gemini 2.5、Gemini 3）。结果显示：代码驱动基准在文本一致性上最佳，但视觉表现有限；扩散模型在图像质量上稍好，却在结构可恢复性上几乎为零；自回归模型 Gemini 3 在所有维度上遥遥领先，尤其在较复杂的图表中结构可恢复性显著提升。

**⚠️ 局限性**

局限性：不评估细粒度的视觉美学与渲染质量；仅关注带有显式组件与有向依赖的框架式科学图表，未覆盖所有图表类型；结构优先评估可能忽略某些场景下的视觉信息对解释的重要性。

---

## 363. Tiny Moves: Game-based Hypothesis Refinement

**arXiv ID:** 2602.09801 | [PDF](https://arxiv.org/pdf/2602.09801v1)

**作者:** Agnieszka Dobrowolska `[一作]` (Relation Therapeutics), Anna Gogleva `[通讯]` (Relation Therapeutics)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出“假设游戏”框架，让LLM代理在共享假设状态上使用固定的推理动作集进行增量式改进；

**💡 创新点**

创新点在于把假设改进形式化为可重用的游戏语法，强调局部可解释的编辑，而非一次性大规模重写；

**🔧 技术方法**

技术上结合LLM控制器与多角色代理、四种核心动作（诊断、修剪、扩展、辩论），并在规则化语法下执行；

**📊 数据集**

使用Reactome人类通路的文本和图结构数据，构造了两类评测：错误修复和部分线索重建；

**📈 对比分析**

与Zero‑Shot、Chain‑of‑Thought、ReAct等提示基线对比，错误修复任务中该方法在精度、F1和错误去除率上均表现最佳；重建任务与ReAct相近，优于单纯提示；

**⚠️ 局限性**

局限性包括：重建任务仍难以完全恢复通路、受限于信息稀缺；框架仅实现最小动作集，未加入显式评分或学习控制器；未验证在完全开放式探索场景下的效果。

---

## 364. When Less is More: The LLM Scaling Paradox in Context Compression

**arXiv ID:** 2602.09789 | [PDF](https://arxiv.org/pdf/2602.09789v1)

**作者:** Ruishan Guo `[一作]` (Baidu Inc.), Daiting Shi `[通讯]` (Baidu Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究大型语言模型在上下文压缩中的规模-忠实度悖论，发现越大模型越易出现知识覆盖和语义漂移；

**💡 创新点**

提出针对知识覆盖与语义漂移的诊断 QA 任务与有效秩、条件熵等内在特征分析；

**🔧 技术方法**

使用压缩-解码框架、有效秩计算、条件熵评估、基准 QA 与 BLEU 等指标；

**📊 数据集**

基于 FineWeb、FaithEval、ConflictQA 等公开数据集；

**📈 对比分析**

与 BLEU 及传统重构指标对比，发现大模型训练损失下降但 QA 真实性与结构保持明显下降；

**⚠️ 局限性**

局限性在于仅针对压缩-解码架构、主要在 Qwen-3 与 LLaMA-3.2 系列，未覆盖所有模型与解码器组合。

---

## 365. Grounding LTL Tasks in Sub-Symbolic RL Environments for Zero-Shot Generalization

**arXiv ID:** 2602.09761 | [PDF](https://arxiv.org/pdf/2602.09761v1)

**作者:** Matteo Pannacci `[一作]` (Sapienza University of Rome), Roberto Capobianco `[通讯]` (Sony AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在子符号环境下，联合学习多任务 LTL 策略和符号映射，使智能体能从原始观测直接执行多种 LTL 指令。

**💡 创新点**

突破了需已知符号映射的假设，利用 Neural Reward Machine 在多任务情境下提供间接监督，成功实现符号地面化与策略共同训练。

**🔧 技术方法**

使用 Neural Reward Machines、LTL 进展算法、图神经网络编码公式、PPO 训练，结合半监督符号地面化技术。

**📊 数据集**

实验在离散 Minecraft‑like 迷宫和连续 FlatWorld 环境，使用随机生成的多任务 LTL 公式集（约 10k 条）进行训练与测试。

**📈 对比分析**

与仅基于 LTL2Action（已知符号映射）的上限和现有基线相比，本文方法在训练公式上与上限相当，零样本推理中显著优于基线；在 Minecraft‑like 环境中成功率 > 90%，在 FlatWorld 的部分任务亦逼近上限。

**⚠️ 局限性**

在全局避免类任务的学习效果不佳，且对观测噪声与符号语义扩展的鲁棒性待提升；当前方法仍需完整的训练公式集与足够多的任务来提供足够的监督。

---

## 366. Improving Interpretability of Lexical Semantic Change with Neurobiological Features

**arXiv ID:** 2602.09760 | [PDF](https://arxiv.org/pdf/2602.09760v1)

**作者:** Kohei Oda `[一作]` (Japan Advanced Institute of Science and Technology), Natthawut Kertkeidkachorn `[通讯]` (Japan Advanced Institute of Science and Technology)

**通讯引用:** 401 | [OpenAlex ID](https://openalex.org/A5028482151)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过把BERT语义空间映射到Binder神经生物学特征空间，实现了对词义变化（LSC）的可解释性研究，并利用该映射评估LSC程度、发现新型LSC类型以及检测词义趋正和趋负。

**💡 创新点**

创新点在于首次将65维Binder特征作为可解释维度映射到上下文嵌入空间，并结合Sparse PCA自动挖掘新的LSC类型，提供直观的功能级解释。

**🔧 技术方法**

主要技术包括BERT预训练模型、线性回归与多层感知机映射、Spearman/欧氏/余弦距离的平均对偶距离（APD）、Sparse PCA以及k‑means聚类用于语义分布分析。

**📊 数据集**

使用的数据集包括Binder特征数据集（535词、65维）、CCOHA历史美国英语语料（1820‑2020，按年代分段）、SemEval‑2020 Task‑1 37词手工评分集。

**📈 对比分析**

与基线BERT空间及已公开的十余种方法（如SSCS、XL‑LEXEME、SDML等）对比，使用线性回归映射后在SemEval评测中取得0.667的Spearman相关系数，优于无外部知识的其他方法；线性回归优于MLP。

**⚠️ 局限性**

局限性包括：Binder特征与某些LSC类型（如隐喻、转喻、缩化、泛化）关联不强；仅能分析Tokenizer可覆盖的词，无法处理子词拆分导致的词；并且样本规模与已知趋正/趋负词集较小。

---

## 367. SinFoS: A Parallel Dataset for Translating Sinhala Figures of Speech

**arXiv ID:** 2602.09866 | [PDF](https://arxiv.org/pdf/2602.09866v1)

**作者:** Johan Sofalas `[一作]` (Informatics Institute of Technology), Ruvan Weerasinghe `[通讯]` (Informatics Institute of Technology)

**通讯引用:** 530 | [OpenAlex ID](https://openalex.org/A5084628821)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建并公开了 SinFoS 数据集，收集了 2,344 条 Sinhala 语言的修辞表达，并对其文字图像、对应英语 FoS、含义解释等进行多维度注释。

**💡 创新点**

首创 Sinhala 级别的多标签 FoS 数据集，并提出投票集成模型实现格言与成语二分类；同时进行跨语言相似度与文化源域分析，揭示 Sinhala 与英语表达在意义与文化上的显著差异。

**🔧 技术方法**

采用 Word2Vec、TF‑IDF（字符 3‑gram）、SVM、RandomForest、XGBoost、Bi‑LSTM 与深度前馈网络进行 FoS 分类；使用 Bi‑encoder（BGE‑large）与 Cross‑encoder（STSB‑RoBERTa）计算 Cosine 相似度与 Fidelity 评分，评估 LLM 的翻译与理解能力。

**📊 数据集**

SinFoS 数据集本身（2,344 条 Sinhala FoS 及其文字图像、对应英语 FoS、含义解释、附加上下文），以及从 Sinhala 书籍与维基百科提取的原始文本。

**📈 对比分析**

投票集成模型在格言与成语分类任务中取得 90.56% 准确率（最高 92.7% 的深度网络）；LLM 评估显示部分模型在 Cosine 与 Fidelity 上表现优异，但整体仍低于人类水平，存在“幻觉”和“超字面”问题。

**⚠️ 局限性**

主要局限包括：部分 Sinhala FoS 缺失英文对应解释导致跨语言分析受限；文化特定表达在英语翻译中易失真；类别不平衡导致模型难以泛化。

---

## 368. Tracing Data Packet Paths over the Internet using Traceroute

**arXiv ID:** 2602.09857 | [PDF](https://arxiv.org/pdf/2602.09857v1)

**作者:** Thomas Dreibholz `[一作]` (Simula Metropolitan), Somnath Mazumdar `[通讯]` (Copenhagen Business School)

**通讯引用:** 549 | [OpenAlex ID](https://openalex.org/A5063602818)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究在端用户视角下，使用HiPerConTracer工具对2018-2022年间跨国四个测点、六大ISP、20个AS的Traceroute和Ping数据进行了为期五年的连续采集，并分析了不同ISP、协议（IPv4/IPv6）以及地理场景（邻国、跨洲、跨大洲）下的路由路径及延迟变化。

**💡 创新点**

创新点在于首次展示长期端用户层面IP路径的非确定性与跨国detour特征，并揭示IPv4与IPv6路由路径和RTT存在显著差异；此外，公开了HiPerConTracer测量框架与完整数据集，为后续研究提供可复现的基础。

**🔧 技术方法**

技术方法包括：HiPerConTracer（ICMP/ICMPv6高频Traceroute与Ping）、HLOC地理定位、BGP/AS映射、MongoDB数据存储、Python/JavaScript脚本处理与可视化。

**📊 数据集**

数据集来源于2018-2022年间在NorNet基础设施下的四个测点（挪威、瑞典、德国、中国）收集的约1.6千万条Traceroute与157.5万条Ping记录，覆盖六大ISP、20个自治系统及14个国家。

**📈 对比分析**

通过比较不同ISP、协议与路由场景的RTT分布（均值、10%/90%分位数）和跳数统计，发现商业ISP往往更不稳定、IPv6有时延更低、跨国detour导致RTT显著波动；这些差异对TCP等协议的拥塞控制与用户体验有直接影响。

**⚠️ 局限性**

局限性包括：Traceroute仅能得到ICMP响应路由，部分路由器不回应导致路径缺失；测量仅为单向视角，缺乏双向对比；AS与地理定位精度有限；无法获取ISP内部路由策略，导致路由变化原因不明；样本仅限四个测点，覆盖范围有限。

---

## 369. Would a Large Language Model Pay Extra for a View? Inferring Willingness to Pay from Subjective Choices

**arXiv ID:** 2602.09802 | [PDF](https://arxiv.org/pdf/2602.09802v1)

**作者:** Manon Reusens `[一作]` (University of Antwerp), David Martens `[通讯]` (University of Antwerp)

**通讯引用:** 4277 | [OpenAlex ID](https://openalex.org/A5101474247)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在旅行助手场景下大型语言模型（LLM）做出的主观决策，通过构造酒店客房属性对照选择任务并使用多项式逻辑模型估计模型的效用，从而提取出模型的支付意愿（WTP）值。

**💡 创新点**

首次将经济学的离散选择与WTP框架应用于LLM偏好评估，系统探讨了用户信息、角色设定（persona）和上下文学习对模型偏好与支付意愿的影响。

**🔧 技术方法**

采用多项式逻辑回归、WTP 计算公式、LLM 生成对话（Llama 3.3 70B、GPT‑4o、Gemini‑3‑Pro）以及不同提示策略（无信息、ICL、Persona、两者结合）等技术。

**📊 数据集**

使用基于 Masiero 等人 2015 年的酒店客房属性构造的 240 组二选一决策难题（包含视野、楼层、俱乐部入口、迷你吧、手机、取消政策、价格等七个属性），并对照人类基准 WTP 数据。

**📈 对比分析**

通过将模型得到的 WTP 与人类基准进行绝对/中位数偏差对比，并检视多项式逻辑模型的 pseudo‑R²；结果显示大型模型的解释力更好，但在某些属性上仍明显高估或低估人类支付意愿，提示提示方式与 persona 对结果影响显著。

**⚠️ 局限性**

局限性包括：小型模型存在顺序偏差、所有模型普遍偏高估人类支付意愿；对属性描述长度与措辞敏感，易被框架操纵；人类基准可能已随时间变迁失效；实验基于人工构造的情境，难以完全泛化至真实旅行代理系统。

---

## 370. Where Are We At with Automatic Speech Recognition for the Bambara Language?

**arXiv ID:** 2602.09785 | [PDF](https://arxiv.org/pdf/2602.09785v1)

**作者:** Seydou Diallo `[一作]` (MALIBA-AI), Aboubacar Ouattara `[通讯]` (DJELIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了首个标准化的 Bambara 语音识别基准测试集与公开排行榜，评估了 37 个模型的识别性能。

**💡 创新点**

创新点在于：①提供正式、纯 Bambara 语言的高质量测试集；②公开基准与排行榜，促进透明可重复的评估；③通过综合 WER/CER 指标揭示多语种预训练在低资源语言上的局限性。

**🔧 技术方法**

使用了音频‑文本对齐、信噪比（SNR）评估、标准化文本处理、加权综合评分（0.5 WER + 0.5 CER）以及对不同权重组合的敏感性分析。

**📊 数据集**

数据集来源于一小时专业录制的马里宪法翻译文本，包含 500 条语音片段（单一成人男性说话者），采用 Bambara 拉丁字母标准正字法。

**📈 对比分析**

对比方法：将 37 个公开模型按综合评分排序；结果显示最佳模型（djelia/asr-v2）WER 47.50%、CER 13.56%，所有模型均未达到 5–15% 的生产级 WER 目标；多语种大模型（如 Whisper）出现 100%+ WER，表现极差。

**⚠️ 局限性**

局限性包括：仅单一说话人、单一正式领域、极高 SNR 的理想录音；数据量极小（1 小时）；未覆盖口音、方言、噪声和法语混码；WER/CER 可能无法充分衡量形态丰富语言的语义质量。

---

## 371. Optimally Deployed Multistatic OTFS-ISAC Design With Kalman-Based Tracking of Targets

**arXiv ID:** 2602.09776 | [PDF](https://arxiv.org/pdf/2602.09776v1)

**作者:** Jyotsna Rani `[一作]` (Indian Institute of Technology Guwahati), Zilong Liu `[通讯]` (University of Essex)

**通讯引用:** 5889 | [OpenAlex ID](https://openalex.org/A5100629531)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在车辆网络中提出一种基于OTFS调制的多静态ISAC系统，利用几何三角测量和Kalman滤波实现目标的精确定位和速度估计，并设计了子最优接收器部署策略。

**💡 创新点**

创新点包括：①将OTFS的时频-延迟双域优势与多静态几何三角化相结合，实现更高精度的感知；②提出基于轨迹预测的Kalman滤波跟踪，使系统在噪声环境下保持鲁棒性；③通过分析三角形面积最优化得到子最优接收器布局，显著降低定位误差。

**🔧 技术方法**

使用的技术包括：OTFS调制、逆对称快速傅里叶变换、延迟-多普勒域协程估计、贝叶斯最小二乘估计、Kalman滤波、几何三角定位与速度投影、邻近选择筛选算法。

**📊 数据集**

实验采用仿真数据：M=256子载波、N=16符号、4‑QAM、Δf=240 kHz、fc=30 GHz、SNR从-20 dB到0 dB、400×400 m²平面内多目标（1–4个）在Δt=0.5 s间隔更新，使用随机与优化的接收器布置方案。

**📈 对比分析**

与随机接收器布置、单静态基线以及无Kalman滤波的主动感知相比，优化方案在-20 dB时目标定位RMSE下降约85 m（相比多目标时增幅），Kalman滤波进一步降低RMSE约5 m；在通信方面，优化ISAC方案相较于单静态基线BER提升高达92.1%，远优于53.8%的前期工作。

**⚠️ 局限性**

主要局限包括：①系统仍以二维平面模型为前提，未考虑高度差异；②延迟‑多普勒估计与Kalman滤波计算量大，复杂度随网格尺寸和目标数线性或更高增长；③在极高移动或多目标重叠场景下峰值分辨率可能受限，需要进一步改进搜索策略。

---

## 372. Where Do Images Come From? Analyzing Captions to Geographically Profile Datasets

**arXiv ID:** 2602.09775 | [PDF](https://arxiv.org/pdf/2602.09775v1)

**作者:** Abhipsa Basu `[一作]` (Indian Institute of Science), Danish Pruthi `[通讯]` (Indian Institute of Science)

**通讯引用:** 620 | [OpenAlex ID](https://openalex.org/A5056959868)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对大规模多模态数据集中的图像-文本对进行地理分布分析，利用LLM提取标题中的地点信息并映射到国家，再结合实体存在过滤器筛选真正包含实体的图像，最后统计各国家在20个常见视觉实体上的出现频率，评估多语言数据集的地理偏差，计算图像和标题的多样性，并用Stable Diffusion生成器检验模型在不同国家下的表现。

**💡 创新点**

首次在大规模视觉语言数据集上系统性地使用LLM驱动的地理归属推理，对比传统基于规则或NER的地理解析，提出extract‑retrieve‑predict框架并结合多语言翻译，构建了可扩展的地理分布量化工具；同时将地理分布与真实世界分布对比，揭示多模态模型训练数据中的严重地域偏差。

**🔧 技术方法**

使用大型语言模型（Gemini‑2.5‑flash、LLaMA‑3.1‑8B、Qwen‑2.5‑7B、GPT‑4 等）进行地点提取与国家预测；使用基于CLIP特征的SVM进行实体存在判断；Vendi‑Score 计算多样性；Stable Diffusion v1.3 进行图像生成；NLLB‑200‑3.3B 翻译非英语标题；GeoNames 数据库做候选检索；评估指标为精确率、召回率、Spearman 相关、Vendi‑Score 等。

**📊 数据集**

Re‑LAION2B‑en、DataComp1B、Conceptual Captions 12M（CC12M）三大英文视觉语言数据集；Re‑LAION2B‑multi 的西班牙语、希腊语、印地语、日语子集；对这些数据集中的20个常见实体（house、flag 等）进行分析。

**📈 对比分析**

通过与传统字符串匹配、NER+匹配、Geoparsepy 等基线在三组人工标注集（5k、57k、1.6k）上比较，Gemini‑2.5‑flash 在 zero‑shot、ICL 和 extract‑retrieve‑predict 模式下均取得最高的精确率与召回率（如 0.93/0.83、0.95/0.91 等）。在多语言场景中，国家分布与 GDP、人口等经济指标高度相关，相关系数达 0.8+。生成图像评估中，Stable Diffusion 在精度上表现良好但召回率几乎为零，说明生成覆盖度不足。

**⚠️ 局限性**

仅覆盖 3 个大规模数据集，语言多样性有限；非英语标题需先翻译，翻译误差可能传播；地理归属预测依赖 LLM 与实体过滤器的准确率；对未包含地点信息的标题无法进行地理标注；无法评估闭源数据集；隐私方面未细粒度定位。

---

## 373. QRS: A Rule-Synthesizing Neuro-Symbolic Triad for Autonomous Vulnerability Discovery

**arXiv ID:** 2602.09774 | [PDF](https://arxiv.org/pdf/2602.09774v1)

**作者:** George Tsigkourakos `[一作]` (University of Piraeus), Constantinos Patsakis `[通讯]` (Information Management Systems Institute of Athena Research Centre)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种三代理神经符号框架 QRS（Query‑Review‑Sanitize），利用 LLM 自动生成 CodeQL 查询、语义验证与消毒，从而在无规则库的情况下主动发现可利用漏洞。

**💡 创新点**

核心创新在于将 LLM 置于检测流程的生成端而非仅作为后置过滤器；通过自我修正的查询生成、链式推理验证、PoC 合成以及最终的消毒裁决，显著降低误报并扩展到传统规则难以覆盖的复杂漏洞。

**🔧 技术方法**

技术实现主要包括：多模型 LLM（Claude、Gemini、GPT、DeepSeek）配合 LiteLLM 接口；自定义工具集（CodeQL、静态切片、grep、路径追踪等）；三阶段代理循环（Q生成查询、R验证可利用性、S消毒裁决）；并使用少量 schema 定义与 few‑shot 示例驱动 LLM 生成。

**📊 数据集**

实验数据集：Hist20（20 条历史 CVE 的 20 个 PyPI 包）用于验证召回率；Top100（2025 年最受下载的 100 个 PyPI 包）用于真实世界漏洞发现与新 CVE 挑战。

**📈 对比分析**

与传统 SAST（Opengrep、Bandit、CodeQL）及现有 LLM‑驱动框架（LLMxCPG、VulAgent 等）比较，QRS 在 Hist20 上达 90.6% 的检测准确率、86.96% 精度、100% 召回；在 Top100 上重现 29 条已知 CVE，发现 5 条新 CVE（被正式公告），并将误报率压低至 35–65% 之间。

**⚠️ 局限性**

局限性包括仅支持 Python（需扩展 CodeQL schema 以适应多语言）；无法检测编译扩展、原生代码、需要动态分析的逻辑漏洞；未研究对抗性规避；误报仍存在，需要多模型配置以提升覆盖率；部分模型成本高、运行时间长。

---

## 374. From Multi-sig to DLCs: Modern Oracle Designs on Bitcoin

**arXiv ID:** 2602.09822 | [PDF](https://arxiv.org/pdf/2602.09822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 375. Generative AI Adoption in an Energy Company: Exploring Challenges and Use Cases

**arXiv ID:** 2602.09846 | [PDF](https://arxiv.org/pdf/2602.09846v1)

**作者:** Malik Abdul Sami `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**通讯引用:** 10199 | [OpenAlex ID](https://openalex.org/A5058417486)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究通过对能源公司16次访谈和内部文档分析，系统识别了41个AI用例，并提出两套基于LLM的pilot方案，探索生成式AI在日常运营中的落地路径。

**💡 创新点**

创新点在于首次从多部门角度量化能源企业的AI需求，构建跨部门优先级框架，并将agentic AI与现有工作流无缝衔接，为行业提供可操作的落地蓝图。

**🔧 技术方法**

使用了大语言模型（LLM）与检索增强生成（RAG）技术，基于LangChain/LangGraph实现邮件自动回复与知识检索等功能。

**📊 数据集**

使用的主要数据集来自能源公司内部，包括财务报表、交易记录、维护日志等，数据为机密，未公开。

**📈 对比分析**

通过访谈编码与主题分析评估需求，并在pilot系统中用BERTScore测得邮件回复准确率89%；未与传统工具直接对比，主要以定性反馈评估效果。

**⚠️ 局限性**

局限性包括仅单一公司案例、样本规模有限、访谈部分采用笔记而非完整录音，缺乏量化绩效验证，结果难以推广到其他行业或规模不同的组织。

---

## 376. Hybrid Responsible AI-Stochastic Approach for SLA Compliance in Multivendor 6G Networks

**arXiv ID:** 2602.09841 | [PDF](https://arxiv.org/pdf/2602.09841v1)

**作者:** Emanuel Figetakis `[一作]` (University of Guelph), Ahmed Refaey Hussein `[通讯]` (University of Guelph)

**通讯引用:** 1340 | [OpenAlex ID](https://openalex.org/A5055738950)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种混合责任感AI-随机学习框架，集成公平、鲁棒与可审计的机制，实现多供应商6G网络SLA合规。

**💡 创新点**

创新点在于将RAI游戏与随机优化结合，形成动态对抗重加权与概率探索的混合模型，并在控制循环中嵌入责任感审计平面。

**🔧 技术方法**

采用负责AI（RAI）游戏、随机优化、Follow‑The‑Regularized‑Leader、CVaR、Frank–Wolfe、深度强化学习与可解释AI技术。

**📊 数据集**

使用合成的二维两类样本，包含五个子组的子群数据进行实验。

**📈 对比分析**

与传统ERM、单对抗RAI、AdaBoost、GDRO等基线对比，混合RAI在最差组准确率提升至60.5%（相较于ERM的21.5%）并保持平均准确率72.7%，SLA违约率大幅下降。

**⚠️ 局限性**

局限在于仅在模拟数据上验证，未测试真实网络环境的动态非平稳性与更复杂的供应商交互；同时对模型超参数的调优依赖实验设计。

---

## 377. Covo-Audio Technical Report

**arXiv ID:** 2602.09823 | [PDF](https://arxiv.org/pdf/2602.09823v1)

**作者:** Wenfu Wang `[一作]` (Tencent), Shan Yang `[通讯]` (Tencent)

**通讯引用:** 1553 | [OpenAlex ID](https://openalex.org/A5101736420)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Covo‑Audio，一款7B参数的端到端大型音频语言模型，可直接用连续音频输入生成连续音频输出，并实现语音理解、对话、情感、全双工交互等多种音频任务。

**💡 创新点**

创新点包括：三模态语音‑文本交错编码框架、智能‑说话人解耦技术、将全双工训练直接嵌入预训练阶段、以及在7B规模上实现多任务SOTA性能。

**🔧 技术方法**

采用Whisper‑Large‑v3作声学编码器、Qwen2.5‑Base作为LLM骨干、基于VQ‑WavLM的音频离散化器、Flow‑Matching+BigVGAN语音解码、层次化三模态交错、以及多任务预训练、RL（GRPO）与Chain‑of‑Thought微调。

**📊 数据集**

使用了约8000万小时多语言语音+3T文本、200k小时ASR、8M小时音频‑文本对、Emotion、Speaker、Age等属性数据，以及在URO‑Bench、VCB‑Bench、MMAU、MMSU等公开基准集上进行训练与评测。

**📈 对比分析**

在语音‑文本对齐、ASR/TTS、跨模态对话、情感识别、音频理解等任务中，与同规模或更大规模的基准模型（如GLM‑4‑Voice、Step‑Audio、Moshi等）比较，Covo‑Audio在多数指标上达到或超过SOTA，尤其在对话推理与全双工交互上表现突出。

**⚠️ 局限性**

局限性包括：在全双工模式下仍存在过早中断、短暂停顿处理不完善；语音情感表达在主观评测上略逊于顶尖商业模型；对多语言知识的覆盖和推理深度仍有提升空间。

---

## 378. AnalyticsGPT: An LLM Workflow for Scientometric Question Answering

**arXiv ID:** 2602.09817 | [PDF](https://arxiv.org/pdf/2602.09817v1)

**作者:** Khang Ly `[一作]` (Elsevier B.V.), Seyed Amin Tabatabaei `[通讯]` (Elsevier B.V.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于大型语言模型（LLM）与检索增强生成（RAG）的工作流AnalyticsGPT，用于回答科研计量（scientometric）领域的问答任务。

**💡 创新点**

创新点包括：① 分层计划（高层规划 + 详细规划）实现任务分解与工具调用的系统化；② 结合实体解析、批量查询与可视化生成，使得回答更完整、可验证；③ 首次在科研计量问答这一细分领域系统地验证LLM的有效性。

**🔧 技术方法**

使用技术：GPT‑4o / GPT‑4o Mini 进行规划、执行、撰写与评估；LangChain 框架管理工具调用；检索增强生成（RAG）与实体解析 API；代码生成与可视化；LLM 评审团（GPT‑4o Mini、GPT‑4.1 Mini、Claude 3.5 Haiku/ Sonnet）做结果判定。

**📊 数据集**

数据集：84 条科研计量问题，包含 34 条真实用户提问与 50 条从 DBLP‑QuAD 采样的合成问答，覆盖多种问题类型（事实、单意图、联合、比较等）。

**📈 对比分析**

与基线（单一 RAG + 直接生成）对比，采用五维评价指标（Coverage、Coherence、Verifiability、Validity、Avg.）和 SME/LLM 评审；AnalyticsGPT 在 Coverage、Validity 上平均提升 0.28 分，错误率从 5/84 降至 1/84，整体性能显著优于基线。

**⚠️ 局限性**

局限性：① 评估标准过细且依赖人工，LLM 评审容易偏差，缺乏大规模自动化评估；② 对细粒度语义理解、时间限定词等细节处理仍不够精准；③ 系统对幻觉引用的校正不完善；④ 缺少对单个模块（如 HLPM、DPM）的消融实验。

---

## 379. Decomposing Reasoning Efficiency in Large Language Models

**arXiv ID:** 2602.09805 | [PDF](https://arxiv.org/pdf/2602.09805v1)

**作者:** Daniel Kaiser `[一作]` (UiT - Arctic University of Norway), Benjamin Ricaud `[通讯]` (UiT - Arctic University of Norway)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可选追踪的推理效率分解框架，将模型推理效率拆解为截断鲁棒性、逻辑鲁棒性、冗长度等可解释因子；

**💡 创新点**

创新点在于将效率分解为可解释模块，并通过工作负载归一化与无追踪评估实现跨模型诊断，首次系统揭示准确率与效率之间的分离；

**🔧 技术方法**

使用token计数、实例工作负载代理、自动化追踪质量度量（grounding、repetition、prompt‑copy）等技术；

**📊 数据集**

在CogniLoad生成式推理任务上评估，涵盖25个模型（12个可追踪）共计约224k实验；

**📈 对比分析**

与准确率对比发现效率排名与准确率相关性仅为0.63，逻辑鲁棒性为主要瓶颈，冗长度差异高达9倍，提出针对性干预建议；

**⚠️ 局限性**

局限在于token仅为计算近似、需结构化工作负载代理、仅在CogniLoad上验证，其他任务需自行定义代理和追踪质量度量。

---

## 380. NavDreamer: Video Models as Zero-Shot 3D Navigators

**arXiv ID:** 2602.09765 | [PDF](https://arxiv.org/pdf/2602.09765v1)

**作者:** Xijie Huang `[一作]` (Zhejiang University), Fei Gao `[通讯]` (Zhejiang University)

**通讯引用:** 26226 | [OpenAlex ID](https://openalex.org/A5100318655)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 NavDreamer 框架，利用生成式视频模型进行 3D 导航。

**💡 创新点**

创新点在于将视频模型作为高层规划器，使用 VLM 对生成的视频进行选择，并通过逆动力学模型与度量深度校正尺度，实现可执行 waypoint 的解码。

**🔧 技术方法**

采用的技术包括生成式视频模型（如 Wan 2.6）、VLM（Qwen3‑VL）进行轨迹选择、逆动力学模型 π³、深度估计模型 Moge2、低层轨迹规划模块 Ego‑Planner 等。

**📊 数据集**

使用互联网规模的公开视频数据训练模型，无需专门的导航数据，评估基于自建的涵盖对象导航、精准导航、空间 grounding、语言控制、场景推理等五个维度的基准任务。

**📈 对比分析**

通过对开放源和闭源视频模型的对比实验，采用视觉一致性、动态可行性和任务完成度三项指标，结果显示 Wan 2.6 在大多数指标上表现最佳，整体平均任务完成度约 84%。

**⚠️ 局限性**

限制包括高计算延迟（1–2 分钟生成完整视频）和难以满足高敏捷或精细动作的需求，导致在狭窄或高动态场景中性能不足。

---

## 381. Towards Poisoning Robustness Certification for Natural Language Generation

**arXiv ID:** 2602.09757 | [PDF](https://arxiv.org/pdf/2602.09757v1)

**作者:** Mihnea Ghitu `[一作]` (Imperial), Matthew Wicker `[通讯]` (Imperial)

**通讯引用:** 415 | [OpenAlex ID](https://openalex.org/A5006299169)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对自然语言生成的可证实毒化鲁棒性框架，并通过定义稳定性和有效性两个安全属性，实现了对生成文本的正式鲁棒性证明。

**💡 创新点**

创新点在于提出Targeted Partition Aggregation (TPA)算法，可对目标攻击给出的毒化预算进行下界计算，并扩展到多词、短语以及多轮对话的可证实安全性。

**🔧 技术方法**

主要技术包括分片聚合 (shard‑and‑aggregate)、投票聚合、MILP优化、LoRA微调以及对语言模型的多样本/多轮协同认证。

**📊 数据集**

实验使用了 Toucan 工具调用数据集、Anthropic 的 Helpful‑&‑Harmless RLHF 偏好数据集以及 OLMo‑1B/2B、Gemma‑2‑2B、Qwen‑1.5‑4B 等模型。

**📈 对比分析**

与传统的 DPA、DPA+ROE 等方法对比，TPA 在稳定性/有效性认证半径上显著提升（如工具调用中中位数抵抗 0.5% 数据毒化），且在多轮对话与偏好对齐任务中保持 90%+ 的有效性认证。

**⚠️ 局限性**

局限性包括推理时高延迟、对实际攻击的保守估计、仅覆盖训练时毒化的最坏情况，无法直接处理推理时攻击、触发式后门等威胁。

---

## 382. Design and Evaluation of an Assisted Programming Interface for Behavior Trees in Robotics

**arXiv ID:** 2602.09772 | [PDF](https://arxiv.org/pdf/2602.09772v1)

**作者:** Jonathan Styrud `[一作]` (ABB Robotics), Christian Smith `[通讯]` (Department of Robotics, Perception and Learning, Royal Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

设计并评估了一个名为 BEhavior TRee GUI 的辅助编程界面，结合 LLM、规划、遗传程序和贝叶斯优化等技术，帮助用户在图形界面中快速构建机器人行为树。

**💡 创新点**

首次将多种自动生成方法（LLM 生成、PDDL 规划、遗传程序搜索、贝叶斯优化）与拖拽图形编辑器融合，并通过“锁定节点”等交互机制提升人机协作，验证人机共同编辑行为树的效果。

**🔧 技术方法**

使用了 GPT‑4 LLM、PDDL 规划、遗传程序 (GP)、贝叶斯优化 (BO)、Python/PyQt5 GUI、Unity 仿真、Google Protobuf API，以及线性混合模型等统计方法。

**📊 数据集**

采用四个自定义 Unity 机器人任务（Demo、Cubes and bowl、Tableware、Trashpicking）作为实验情境，未使用公开数据集。

**📈 对比分析**

通过 60 名受试者的混合组实验，对 6 种 GUI 变体（FULL、MANUAL_ONLY、NO_BO、NO_GP、NO_LLM、NO_PLANNER）进行任务得分、SUS 可用度量和排名比较；FULL 取得显著更高的任务得分（≈91）和 SUS 分数，去除 LLM 或规划导致得分显著下降，证明 AI 助手提升了任务成功率和可用性。

**⚠️ 局限性**

局限性包括任务过于简化、实验时间短限制了学习算法效果、用户对 AI 建议的信任度不足以及未充分利用锁定等交互；系统需在更真实、复杂场景和更长训练时间下进一步验证。

---

## 383. Infusion: Shaping Model Behavior by Editing Training Data via Influence Functions

**arXiv ID:** 2602.09987 | [PDF](https://arxiv.org/pdf/2602.09987v1)

**作者:** J Rosser `[一作]` (University of Oxford), Laura Ruis `[通讯]` (MIT CSAIL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Infusion 框架，利用影响函数对少量训练文档做最小化扰动以诱导模型产生指定行为。

**💡 创新点**

首次将训练数据归因技术逆向用于攻击，即通过文档级影响函数估计与梯度优化实现无需显式目标样本的隐蔽性数据中毒。

**🔧 技术方法**

采用 EK‑FAC 近似逆 Hessian 与梯度投影，PGD 计算文档扰动，随后在改动后的语料上微调验证效果。

**📊 数据集**

在 CIFAR‑10 图像分类、30k Caesar 密文 Transformer 任务以及 TinyStories 上预训练的 GPT‑Neo‑8M 进行实验。

**📈 对比分析**

与随机噪声、单样本插入及多样本插入三种基线对比，CIFAR‑10 上实现 100% 成功率，目标类概率提升至 37%，跨架构转移可观；Transformer 与小型 LLM 仅实现概率提升，预测翻转稀少。

**⚠️ 局限性**

对大规模模型的影响有限，离散词嵌入优化难度大，扩展到完整预训练流程或后期对齐仍需研究。

---

## 384. Coupled Inference in Diffusion Models for Semantic Decomposition

**arXiv ID:** 2602.09983 | [PDF](https://arxiv.org/pdf/2602.09983v1)

**作者:** Calvin Yeung `[一作]` (University of California), Mohsen Imani `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出一种基于扩散模型的语义分解框架，利用耦合的逆扩散过程和重建引导实现对绑定表示的分解。

**💡 创新点**

将扩散模型与Hopfield/共振网络对应，构造解析先验和重建能量，并引入迭代采样实现连续版共振网络，从而显著提升分解容量与鲁棒性。

**🔧 技术方法**

使用解析扩散先验、能量引导的逆扩散（Diffusion Posterior Sampling）、概率流ODE、类似Hopfield的softmax更新以及多模型耦合的迭代采样。

**📊 数据集**

在自生成的随机{-1,1}代码本上实验，代码本维度D=1000，K=2–5，搜索空间n^K，采用多种噪声水平、m值与D变化进行评估。

**📈 对比分析**

与共振网络、注意力共振网络以及ALS进行对比，Similarity/Latent Similarity模型在准确率、容量上均优于基线，容量提升可达2–5倍；在噪声、K、m、D等设置下均保持高准确率。

**⚠️ 局限性**

对代码本维度和噪声敏感；迭代采样需要多重重启，计算成本相对较高；对非平衡或非独立代码本的泛化性尚未验证。

---

## 385. Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning

**arXiv ID:** 2602.09945 | [PDF](https://arxiv.org/pdf/2602.09945v1)

**作者:** Jinsong Liu `[一作]` (Weill Cornell Medicine), Jiang Bian `[通讯]` (Indiana University)

**通讯引用:** 13578 | [OpenAlex ID](https://openalex.org/A5030951014)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一套差异推理学习框架（DRL），通过对比临床代理的自由文本推理（Chain‑of‑Thought）与专家/指南推理图，提取推理差异并生成可检索的“修补指令”，在推理时动态补齐逻辑缺口。

**💡 创新点**

创新点包括：①将临床推理建模为有向无环图（DAG），使推理结构可量化；②使用临床加权图编辑距离（GED）和LLM‑as‑judge进行语义匹配，得到缺失、错误与结构偏差三类诊断；③将诊断转化为可复用的自然语言指令，构建 Differential Reasoning Knowledge Base (DR‑KB)；④在推理时采用检索增强生成（RAG）仅检索“修补指令”而非原始示例，实现无参数更新的即时补丁。

**🔧 技术方法**

技术手段主要包括：LLM语义解析器（提取推理图）、LLM‑as‑judge（节点语义匹配与一致性检验）、临床加权GED算法（差异分析）、指令生成LLM（Insight Generator）以及BM25检索与RAG推理。

**📊 数据集**

数据集包括公开医学问答基准 MedQA、MedMCQA（对齐症状、胸痛、卒中三类问题），以及从急诊电子病历转化的内部 Return Visit Admission (RVA) 预测问答集，共436个样本。

**📈 对比分析**

与基线模型（Qwen3‑8B、LLaMA‑3.1‑8B‑Instruct、MedReason‑8B、HuatuoGPT‑o1‑8B、MedPRM‑8B）对比，DRL 在 MedQA 和 MedMCQA 上提升约 2–4 分，最显著在 RVA 上提升约 24 分（Qwen）或 15 分（LLaMA），显著优于所有基线并表明在需要深度推理的真实临床任务中更具鲁棒性。

**⚠️ 局限性**

局限性：①依赖 LLM 进行图提取和节点匹配，提取噪声或偏差会影响后续修补；②需要高质量的参考推理（专家笔记/指南），在低资源或不规范环境下难以实现；③实验规模受限，未在更大临床数据或多中心环境中验证；④当前仅提供“指令”级别的补丁，缺乏可直接转换为决策规则或策略的形式。

---

## 386. Why Do AI Agents Systematically Fail at Cloud Root Cause Analysis?

**arXiv ID:** 2602.09937 | [PDF](https://arxiv.org/pdf/2602.09937v1)

**作者:** Taeyoon Kim `[一作]` (Hanyang University), Kyungyong Lee `[通讯]` (Hanyang University)

**通讯引用:** 1104 | [OpenAlex ID](https://openalex.org/A5006735153)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对五种大型语言模型在OpenRCA基准上的RCA代理执行了1675次完整运行，并对失败模式进行过程级诊断，形成了12种陷阱类型的分类体系。

**💡 创新点**

提出了面向过程的失败诊断方法和12种陷阱类型的分类体系，并通过实验验证了陷阱主要源自代理架构而非单个模型。

**🔧 技术方法**

采用多代理Controller–Executor架构、ReAct/链式思维推理、Prompt工程、代码生成与执行、以及对交互协议进行扩展以传递代码与错误信息等技术。

**📊 数据集**

使用OpenRCA基准数据集，共335起故障事件，包含系统指标、日志与分布式跟踪等多模态遥测。

**📈 对比分析**

将五个模型的完美检测率与基准结果对比，发现最高仅12.5%，通过增强代理间通信可将完美率提升至约7-9%，同时降低步骤数和总体token消耗。

**⚠️ 局限性**

受限于仅在Bank域子集进行缓解实验、诊断流程仍需人工验证以及缺乏对其他RCA框架通用性的验证。

---

## 387. VersaViT: Enhancing MLLM Vision Backbones via Task-Guided Optimization

**arXiv ID:** 2602.09934 | [PDF](https://arxiv.org/pdf/2602.09934v1)

**作者:** Yikun Liu `[一作]` (Shanghai Jiao Tong University), Weidi Xie `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 VersaViT——一种多任务协同后训练框架，针对 MLLM 视觉编码器在稠密预测任务上表现不足的问题，通过轻量化任务头和多粒度监督提升视觉编码器的全局语义与像素级感知能力。

**💡 创新点**

创新点在于将 VQA/描述、单目深度估计与指代分割三类任务联合起来，利用多粒度监督（高层语义、中层空间、像素级定位）共同优化共享视觉编码器，同时通过轻量化任务头将任务冲突隔离，兼顾语言推理与稠密感知。

**🔧 技术方法**

采用 Vision Transformer 视觉编码器、CLIP 预训练、指令微调、轻量化任务头、联合多任务损失、伪深度标签生成、线性探针评估以及对比学习与多任务学习相结合的技术栈。

**📊 数据集**

使用的数据集包括 pixmo‑cap、SA‑1B‑InternVL、idLWDS OCR、Depth Anything V2 伪深度、RefCOCO、Pascal VOC、ADE20k、NYUv2、KITTI、COCO、CC3M、CC12M、YFCC15M 等多源视觉文本与稠密标注数据。

**📈 对比分析**

在 OpenCompass VQA、语义分割（ADE20k、Pascal VOC）、单目深度估计（NYUv2、KITTI）等基准上与 Qwen2‑VL‑ViT 进行线性探针与整体微调对比，VQA 平均提升 1.6 分，语义分割从 33.6 提升至 49.6，深度 RMSE 从 3.735 降至 3.136，整体性能超过 SigLIP 2，接近 DINOv3。

**⚠️ 局限性**

局限性包括：仍未与顶尖 MLLM 级别模型匹敌，主要受限于数据规模；多任务权重需要调优，训练稳定性尚待进一步验证；在检索任务中略逊于 SigLIP 2，且未在预训练阶段验证其对原始 MLLM 训练流程的影响。

---

## 388. SARS: A Novel Face and Body Shape and Appearance Aware 3D Reconstruction System extends Morphable Models

**arXiv ID:** 2602.09918 | [PDF](https://arxiv.org/pdf/2602.09918v1)

**作者:** Gulraiz Khan `[一作]` (University of Hull), Waqas Ahmed `[通讯]` (University of Hull)

**通讯引用:** 475 | [OpenAlex ID](https://openalex.org/A5010694386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一个名为SARS的模块化系统，利用单张RGB图像同时重建高保真的人脸和全身3D网格，并保持面部身份、表情与年龄、性别等语义一致性。

**💡 创新点**

创新点包括：① 将面部语义特征（年龄、性别、关键点）与几何先验（位移图、符号距离场）在潜在空间融合，构建身份感知的表情驱动面部重建；② 独立面部与身体模块后通过融合模块无缝拼接，解决单一模型难以同时捕捉细节与姿态的问题；③ 采用多任务鉴别器与结构一致性损失，显著提升面部细节与全局几何的逼真度。

**🔧 技术方法**

主要技术包括：3DMM（FaceScape）、SMPL与SMPLify、SPIN、StyleGAN2解码器、残差CNN提取语义特征、对抗训练、多任务鉴别器、结构一致性损失、优化+注意力融合的全身拼接模块。

**📊 数据集**

使用的数据集有：MICC Florence 3D（面部重建评估）、3DPW（体姿评估）和EHF（全身重建评估）。

**📈 对比分析**

性能对比：在MICC Florence 3D上，SARS相较传统方法平均点对面距离降低约40%；在3DPW和EHF上，SARS在MPJPE、V2V、PA-V2V等指标上均优于HMR、SPIN、ExPose、SMPLify-X等SOTA方法，显示出更高的重建精度与细节保留。

**⚠️ 局限性**

局限性：仅基于单张RGB图像，面对严重遮挡或极端视角时性能可能下降；面部细节仍受训练数据分布限制；系统目前未进行实时性能评估；未考虑体积、体重等更细粒度的体形特征；融合模块对大幅度姿态变换的兼容性仍有改进空间。

---

## 389. Self-Regulated Reading with AI Support: An Eight-Week Study with Students

**arXiv ID:** 2602.09907 | [PDF](https://arxiv.org/pdf/2602.09907v1)

**作者:** Yue Fu `[一作]` (University of Washington), Alexis Hiniker `[通讯]` (University of Washington)

**通讯引用:** 3190 | [OpenAlex ID](https://openalex.org/A5074077266)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一门大学AI选修课中，研究人员跟踪15名学生在8周内使用AI聊天机器人支持自我调节阅读，记录并编码了239个阅读会话中的838条提示，归纳出四个认知主题（解码、理解、推理、元认知），并分析其随时间、位置和个体差异的使用模式；

**💡 创新点**

创新点在于构建了以阅读理解理论为基础的细粒度提示编码框架，揭示了学生主要使用AI做理解与总结、推理和元认知层面不足的“阅读通过AI”现象，以及“意向-行为差距”与效率驱动的阅读策略；

**🔧 技术方法**

采用了质性与量化混合方法：编码体系、线性混合效应模型、卡方检验、ICC、主题分析等统计和分析技术，结合Bloom、Barrett等认知框架对提示进行主题归类；

**📊 数据集**

数据集为来自15名本科生的239个阅读会话记录，共838条AI交互提示，附带学生自评与访谈资料；

**📈 对比分析**

论文未进行系统对比或性能评估，而是以描述性统计和相关性分析阐述提示分布、认知进展、个体差异和意向-行为差距，未给出传统机器学习模型或效果对比；

**⚠️ 局限性**

局限性包括样本规模小、仅涉及单门AI课程、提示作为认知代理且缺乏实际理解测评、未验证不同学科或更大人群的普适性、以及对AI工具功能和设计的外部约束未深入探究。

---

## 390. Eve-positional languages: putting order into Büchi automata

**arXiv ID:** 2602.09896 | [PDF](https://arxiv.org/pdf/2602.09896v1)

**作者:** Olivier Idir `[一作]` (Université Paris Cité), Olivier Idir `[通讯]` (Université Paris Cité)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5014600757)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出了“有序Büchi自动机”这一新形式，用来精确刻画Eve‑positional ω‑正则语言，并给出从可完成的ε‑完整奇偶自动机到有序Büchi自动机的多项转换；随后给出了一个以阶乘上界的确定化过程，证明在充分完整的字母表下可得到最优大小的确定奇偶自动机；此外证明了有序Büchi自动机与Eve‑positional语言之间的等价性，补全了Casares‑Ohlmann关于可完成性的缺失命题。

**💡 创新点**

创新点包括：①首次定义并使用“有序Büchi自动机”作为Eve‑positional语言的形式化；②提供了从ε‑完整奇偶自动机到有序Büchi自动机的直接构造；③提出了新的确定化算法，复杂度为阶乘上界，并在特定字母表上证明其最优性；④完成了对Casares‑Ohlmann理论的缺失推论。

**🔧 技术方法**

技术主要包括：形式化的自动机理论与图论（奇偶/巴赫自动机、ε‑完整性、预序/全序结构）；上层语义与结构化的“本地偏好”性质证明；利用tile（过渡块）与上升闭包构造有序自动机；确定化过程基于记录（record）与“最近出现记录”技巧；以及对最优性的证明借鉴Colcombet‑Zdanowski的花游戏与策略记忆论证。

**📊 数据集**

本文为理论研究，无实验数据集；所有结果均为形式化证明与构造性算法。

**📈 对比分析**

相较于以往对Eve‑positional语言的研究（如Casares‑Ohlmann、Colcombet‑Idir等的ε‑完整性与语义化简），本文提供了更直接的结构化表述和更紧凑的确定化方案。其确定化的状态上界为∑_{i=1}^{n-1} i!，相比一般Büchi自动机的O((1.64n)^n)上界显著下降；在充分完整字母表下进一步证明了此上界的最优性。

**⚠️ 局限性**

局限性包括：①未能在理论框架下给出比现有更快的判定算法；②研究范围仅限于ω‑正则语言，未扩展到更一般的ω‑语言；③确定化过程虽然复杂度下降，但在最坏情况下仍为阶乘级，若状态数较大仍不现实；④对字母表完整性的要求限制了最优性结论的适用范围。

---

## 391. TriPilot-FF: Coordinated Whole-Body Teleoperation with Force Feedback

**arXiv ID:** 2602.09888 | [PDF](https://arxiv.org/pdf/2602.09888v1)

**作者:** Zihao Li `[一作]` (Zeno AI), Weiming Zhi `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套名为TriPilot-FF的全身遥控系统，结合脚踏式底盘控制与双臂跟随遥控，并通过激光雷达驱动的阻力反馈、臂侧力反射以及可调操纵性提示，为移动双臂机器人提供连续、低延迟的底盘操作和接触感知。

**💡 创新点**

创新点包括：① 脚踏式低成本阻力反馈，将激光雷达距离转化为方向感知的阻力，直接引导操作员进行碰撞规避；② 通过双臂跟随映射实现手臂的力反射，让操作员实时感知末端执行器接触状态；③ 利用可学习的可调操纵性场，向操作员反馈底盘重定位方向，降低臂部姿态过伸和低可达性问题；④ 将这些反馈信号作为额外观测加入Action Chunking with Transformers（ACT）策略，显著提升了机器人学习的成功率。

**🔧 技术方法**

主要技术包括：基于低成本360°激光雷达的碰撞检测与阻力计算、三自由度脚踏机件的双向力反馈、双臂跟随/高增益位置跟踪的阻尼控制、可微可调操纵性场的神经网络逼近与梯度引导、以及将力观测嵌入ACT中的编码/解码模块。

**📊 数据集**

数据集：利用TriPilot-FF在真实环境下收集的全身遥控轨迹，包含22×10^3条轨迹（包括移动、抓取、搬运等任务），并在MuJoCo仿真中生成约100条任务轨迹；此外，针对长周期任务（如LaundryTransport）收集约20条完整执行轨迹。

**📈 对比分析**

比较方法：在多个基准任务（BlindCarry、NarrowTransport、GuidedReach、MobileSwipe）上，将TriPilot-FF与无脚踏反馈、无力反射、无操纵性提示三种对照方案对比。结果显示：① 在BlindCarry中，成功率从55%提升至100%，碰撞次数降至0，完成时间缩短30%；② 在NarrowTransport中，成功率从75%提升至100%，碰撞次数降至0，完成时间缩短≈30%；③ 在GuidedReach中，完成时间从15.17s降至7.39s，低操纵性时段比例从41.04%降至30.59%；④ 在MobileSwipe中，扭矩标准差与能耗分别从3.239降至2.532、0.313降至0.196。学习阶段，ACT+Torque模型在CubeTransfer和CubePickPlace的成功率分别从22%→50%、28%→36%，奖励提升明显；在真实任务BasketPack和HangerHandOff中，加入Torque使成功率从12%→24%（+12）和68%→92%（+24）增长。

**⚠️ 局限性**

局限性：① 需要对操作员进行足够的训练以适应脚踏与手臂的耦合控制；② 当前阻力和操纵性提示参数是固定的，缺乏在线自适应；③ 仅针对前轮或全轮移动平台，未验证在差异更大的底盘（如四足、履带）上的适用性；④ 在极端动态环境（高速移动、复杂障碍）中，激光雷达的实时性与精度可能不足，导致反馈延迟；⑤ 对于极低频的远程交互或网络延迟场景，系统仍需进一步鲁棒性验证。

---

## 392. AdaTSQ: Pushing the Pareto Frontier of Diffusion Transformers via Temporal-Sensitivity Quantization

**arXiv ID:** 2602.09883 | [PDF](https://arxiv.org/pdf/2602.09883v1)

**作者:** Shaoqiu Zhang `[一作]` (Shanghai Jiaotong University), Yulun Zhang `[通讯]` (Shanghai Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 AdaTSQ，一个针对 Diffusion Transformer 的后训练量化框架，利用时间感知的动态位宽分配与 Fisher 信息校准实现高效低位宽推理。

**💡 创新点**

创新点在于将 Pareto 约束下的时间步动态位宽搜索与基于 Fisher 信息的时间加权权重校准相结合，显著提升低位宽下的生成质量。

**🔧 技术方法**

使用 Beam Search 对时间步位宽路径进行 Pareto 最优搜索，采用 Fisher 信息衡量层对量化噪声的敏感度，并在 Hessian 加权中加入时间权重，实现风险感知的权重量化。

**📊 数据集**

在 Flux-Dev、Flux-Schnell、Z-Image（文本到图像）和 Wan2.1-1.3B（文本到视频）四大 Diffusion Transformer 上进行实验。

**📈 对比分析**

与 SVDQuant、ViDiT‑Q、SmoothQuant、GPTQ 等基线在 Geneval（图像）和 VBench（视频）上对比，AdaTSQ 在 W4A4 维持近 FP16 的质量，且首次实现 W3A3 生成，性能显著优于现有方法。

**⚠️ 局限性**

局限在于仍需针对不同模型调节温度与位宽搜索范围，且对极端低位宽（≤2 位）或更大模型的推广尚待验证。

---

## 393. Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings

**arXiv ID:** 2602.09985 | [PDF](https://arxiv.org/pdf/2602.09985v1)

**作者:** Alexander Fertig `[一作]` (Technische Hochschule Ingolstadt), Michael Botsch `[通讯]` (Technische Hochschule Ingolstadt)

**通讯引用:** 478 | [OpenAlex ID](https://openalex.org/A5058811339)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一个在线监控框架，用于在自动驾驶车辆的对象列表（object list）中检测对象状态表示（multivariate time‑series）的异常。

**💡 创新点**

创新点在于：①使用JEPA（Joint Embedding Prediction Architecture）自监督学习将对象状态映射到富表达的潜在空间，完全不需要异常标签；②将该嵌入作为输入，利用成熟的异常检测算法（如ABOD、LOF、GMM）实现在线、实时的异常判定；③设计适合SOTIF操作阶段的在线部署方案，提升车辆运行时的安全评估。

**🔧 技术方法**

采用的技术包括：Transformer‑based context/target 编码器、Masking 与 Predictor 架构的 JEPA 训练，L1 损失与 stop‑gradient；异常检测方法为 ABOD（主流基线），并对比 LOF 与 GMM；训练与推理均保持低 FLOPs，适合嵌入式部署。

**📊 数据集**

实验数据集为公开的 nuScenes，使用其 LiDAR‑only MOT（FocalFormer3D）生成的对象列表；对测试集通过误差模型在单一特征（如速度）上注入随机噪声，构造人工异常。

**📈 对比分析**

在人工异常任务上，利用 JEPA 嵌入的异常检测方法在 AUROC、F1、MCC、Accuracy 等指标上明显优于直接在原始特征空间进行检测；FPR95 为 52.8%（ABOD），在低 FPR（1%/5%）下仍能捕获 23.4%/52.9% 的异常，说明框架在严格误报预算下保持可靠。

**⚠️ 局限性**

局限性：①仅在单传感器 LiDAR 数据上验证；②异常类型为人工生成的噪声，缺乏真实标注；③缺少多模态或更复杂场景下的鲁棒性评估；④对实时系统的实际部署成本与延迟未给出详细评估。

---

## 394. The Parameterized Complexity of Geometric 1-Planarity

**arXiv ID:** 2602.09978 | [PDF](https://arxiv.org/pdf/2602.09978v1)

**作者:** Alexander Firbas `[一作]` (TU Wien), Alexander Firbas `[通讯]` (TU Wien)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5092760441)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

系统研究几何 1‑可平面图的参数化复杂性，给出按树深度的 FPT 算法、按反馈边数的核化以及对应下的下界。

**💡 创新点**

创新点在于首次将 Thomassen 的直线化判定与 B/W 配置判定相结合，实现树深度下的 FPT；并将核化技术推广到几何 1‑可平面和 k‑可平面，显著缩小核大小。

**🔧 技术方法**

采用 Thomassen 的 B/W 直线化判定、Bannister‑Cabello‑Eppstein 的结构化参数化分析、Reidemeister 移动、以及对 Bin Packing 的多项式化简构造等技术。

**📊 数据集**

论文为理论性工作，无实验数据集，所有结果均来自数学证明与构造。

**📈 对比分析**

与先前核大小 O((3ℓ)!) 的 1‑可平面核相比，改进至 O(ℓ·8^ℓ)，同时给出 k‑可平面 O(ℓ·8^ℓ) 核，证明了参数化下的有效性；在下界部分展示了在路径宽、回路顶点数、带宽等限制下仍保持 NP‑完全。

**⚠️ 局限性**

限制在于仅覆盖几何 1‑可平面（k>1 的情况仍未可解），未给出多项式核的存在性，且对树深度下的通用图仍缺乏完整的参数化算法。

---

## 395. Hydra-Nav: Object Navigation via Adaptive Dual-Process Reasoning

**arXiv ID:** 2602.09972 | [PDF](https://arxiv.org/pdf/2602.09972v1)

**作者:** Zixuan Wang `[一作]` (ByteDance Seed), Yiming Gan `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在单一视觉‑语言模型（VLM）中统一慢速高层推理与快速低层控制的双进程物体导航代理，并通过迭代拒绝微调（IRFT）实现对推理时机的自适应触发。

**💡 创新点**

创新点包括：① 在单模型中实现慢速推理与快速执行的无缝切换，避免多模型碎片化；② 设计“慢速–快速”自适应机制，使用IRFT学习在“停滞点”才触发高成本推理；③ 构建三阶段训练流程（空间‑动作对齐、记忆‑推理融合、IRFT）和数据合成管线，以提升时空推理和记忆能力。

**🔧 技术方法**

核心技术包括：视觉‑语言模型（Qwen2.5‑VL‑7B）、基于图的长时记忆结构、基于KV缓存的快速低层动作生成、迭代拒绝微调（IRFT）以及对“停滞点”的定义与采样策略。

**📊 数据集**

使用了 HM3D、MP3D、OVON 三大公开物体导航基准，以及在 HM3D/MP3D/OVON 训练集上合成的 500k+ 轨迹与 565k 混合样本；在实验中还对 OVON 进行未见对象（Val‑Unseen）和同义词（Val‑Synonyms）测试。

**📈 对比分析**

与现有最优方法相比，作者在 HM3D、MP3D、OVON 的成功率（SR）和路径效率（SPL）均取得领先；在 OVON Val‑Unseen 上 SR 提升至 66.3%，比上一榜单高出 21.1%；同时引入 SOT 指标显示推理成本显著下降，IRFT 仅需约 3% 的推理比例即可实现 22.2% 的 SOT。

**⚠️ 局限性**

局限性主要在于：仅在 Habitat 模拟器上评估，缺乏更逼真模拟器或真实环境下的高质量基准；推理切换机制仍基于预定义的停滞阈值，未采用在线强化学习进一步优化；且目前仅针对物体导航任务，未扩展到更广泛的机器人任务。

---

## 396. Bladder Vessel Segmentation using a Hybrid Attention-Convolution Framework

**arXiv ID:** 2602.09949 | [PDF](https://arxiv.org/pdf/2602.09949v1)

**作者:** Franziska Krauß `[一作]` (University of Stuttgart), Carina Veil `[通讯]` (Stanford University)

**通讯引用:** 166 | [OpenAlex ID](https://openalex.org/A5044301640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种混合注意力-卷积（HAC）框架，用于在膀胱镜图像中精确分割血管，以支持术中导航。

**💡 创新点**

创新点在于将Transformer编码器提取的全局血管拓扑与U‑Net细节重建解码器相结合，并采用物理感知自监督预训练，专门针对膀胱镜的气泡、眩光和对比度变化。

**🔧 技术方法**

使用的技术包括Transformer自注意力、U‑Net分割网络、物理感知数据增强（合成气泡、光照衰减、对比度降低）、Tversky、Dice、clDice损失、随机深度正则化以及自监督去噪预训练。

**📊 数据集**

实验数据集为BlaVeS，仅包含50帧人工标注的膀胱镜血管图像，并利用81帧未标注图像进行自监督预训练。

**📈 对比分析**

通过在BlaVeS测试集上与U‑Net、U‑Net++、SA‑UNet和Swin‑UNet等现有方法比较，HAC在精度（≈0.61）、clDice（≈0.66）和总体准确率（≈0.94）上均优于或相当于其他模型，尤其在抑制假阳性（如黏膜折叠）方面表现突出。

**⚠️ 局限性**

局限性主要是数据量极少（仅50帧标注），导致模型泛化能力和统计显著性受限；标注质量和多样性不足可能影响评估结果。

---

## 397. Operationalizing Human Values in the Requirements Engineering Process of Ethics-Aware Autonomous Systems

**arXiv ID:** 2602.09921 | [PDF](https://arxiv.org/pdf/2602.09921v1)

**作者:** Everaldo Silva Júnior `[一作]`, Genaína Nunes Rodrigues `[通讯]` (University of Brasilia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套基于目标导向与规则约束的需求工程流程，用于在自主系统的设计与运行时阶段把人类价值转化为可执行的规范目标，并通过自动化检查发现价值冲突。

**💡 创新点**

创新点在于：①将伦理规范与功能、适应性目标统一建模为规范目标，②通过自动化翻译将目标模型映射为SLEEC DSL规则，③利用LEGOS‑SLEEC实现规则层面的完整性与冲突检测，并支持价值冲突的可追溯协商。

**🔧 技术方法**

使用了目标模型（goal-oriented modeling）、SLEEC Domain Specific Language、LEGOS‑SLEEC工具（含自动翻译与一致性检查）以及基于任务的可执行规则生成技术。

**📊 数据集**

案例数据集为医疗体感网络（Body Sensor Network）场景，用于演示流程的可行性；未采用公开大规模数据集。

**📈 对比分析**

通过预先设计的冲突诊断与协商会话验证流程的有效性，虽然没有提供量化性能指标，但在案例研究中成功识别并解决了规范-功能-适应性冲突，体现了方法在实践中的可操作性。

**⚠️ 局限性**

局限性包括：评估范围仅限单一案例，缺乏大规模实验和定量性能评估；对不确定性来源与障碍事件的建模仍不完整；需进一步研究多源不确定性、保证机制及更广泛的用户研究。

---

## 398. Focus Session: LLM4PQC -- An Agentic Framework for Accurate and Efficient Synthesis of PQC Cores

**arXiv ID:** 2602.09919 | [PDF](https://arxiv.org/pdf/2602.09919v1)

**作者:** Buddhi Perera `[一作]` (New York University), Ramesh Karri `[通讯]` (New York University)

**通讯引用:** 16486 | [OpenAlex ID](https://openalex.org/A5059648257)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 LLM4PQC，一个基于大语言模型的代理框架，将 NIST PQC 参考实现自动重构为可合成的 HLS-C 代码，并通过编译-仿真-合成的反馈循环完成 RTL 合成与验证。

**💡 创新点**

创新点在于：①使用 LLM 对 C 代码进行语义感知的预处理，消除动态内存、运行时初始化和复杂结构的合成障碍；②将 LLM 与 C2HLSC 及 Catapult 结合，形成自动化、迭代的设计空间探索流程；③针对 PQC 关键子模块（NTT、FFT、采样）实现了高效且可验证的硬件加速器。

**🔧 技术方法**

核心技术包括：大语言模型（ChatGPT o3-mini）驱动的代码转换、结构展开、常量表预计算、HLS pragmas 生成；C2HLSC 自动化的迭代优化；Catapult HLS、Vivado 及 Design Compiler 进行综合与仿真；KAT 测试驱动的功能验证。

**📊 数据集**

使用的数据集为 NIST 官方 PQC 参考实现（Kyber、Dilithium、Falcon）以及 PQC‑clean 仓库的子程序，提取出关键子模块并生成 KAT 测试集。

**📈 对比分析**

与手工 RTL 及现有 LLM 辅助 RTL 框架对比，LLM4PQC 在 ASIC（Nangate 45nm）与 FPGA（Artix‑7、XCZU7EV）上实现的 LUT/FF 资源显著降低，Latency 在部分子模块保持相近或更优，但在某些采样器上由于数据相关性导致周期数较高；总体上显示出高生产力与竞争性硬件质量。

**⚠️ 局限性**

主要局限包括：对浮点运算的支持仍依赖预处理，导致合成效率受限；LLM 在未显式引导时倾向于面积优化而非延迟优化，需手工或进一步演化策略来提升性能；模型泛化性受限于训练数据，未来可通过 HLS 数据集微调或 RAG 方法提升效果。

---

## 399. Safeguarding Privacy: Privacy-Preserving Detection of Mind Wandering and Disengagement Using Federated Learning in Online Education

**arXiv ID:** 2602.09904 | [PDF](https://arxiv.org/pdf/2602.09904v1)

**作者:** Anna Bodonhelyi `[一作]` (Technical University of Munich), Enkelejda Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 10818 | [OpenAlex ID](https://openalex.org/A5008809634)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一套基于联邦学习的隐私保护框架，用来实时检测在线学习中的心智游离、脱离和厌倦状态。

**💡 创新点**

首次将跨设备联邦学习应用于教育领域，实现了在不共享原始视频数据的前提下仍能提高检测精度，并通过眼镜相关特征提升对佩戴眼镜用户的鲁棒性。

**🔧 技术方法**

使用双向长短时记忆网络（bi‑LSTM）结合EmoNet与OpenFace特征提取，并在联邦学习框架下实现FedAvg、FedAdam、FedProx、MOON、FedAwS、TurboSVM-FL等算法。

**📊 数据集**

在五个公开数据集上评估：Colorado、Korea、Germany（心智游离检测），EngageNet（参与度检测），DAiSEE（厌倦检测）。

**📈 对比分析**

通过与集中式学习和多模型Bagging进行对比，联邦学习在大多数数据集上实现了更高或相近的F1分数（如EngageNet 57.4% vs. 51.4%），表明联邦学习在保持隐私的同时不显著损失性能。

**⚠️ 局限性**

受限于样本不平衡、眼镜佩戴者比例低、光照噪声大以及模型对人脸遮挡的鲁棒性不足，导致整体检测精度仍低于典型图像分类基准，且在极少样本或特殊场景下性能波动显著。

---

## 400. Stemphonic: All-at-once Flexible Multi-stem Music Generation

**arXiv ID:** 2602.09891 | [PDF](https://arxiv.org/pdf/2602.09891v1)

**作者:** Shih-Lun Wu `[一作]`, Nicholas J. Bryan `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了一种名为Stemphonic的扩散/流式框架，能够一次性生成可变组合的同步音乐stem并支持条件生成与活动控制。

**💡 创新点**

通过训练时的stem分组与共享噪声技术，在保持开放词汇生成的同时实现多stem同步，并实现了一键式多stem生成与活动控制。

**🔧 技术方法**

结合VAE编码、Latent Diffusion/Flow（Diffusion Transformer）与文本T5-XXL嵌入、CFG、噪声共享与子mix条件等技术。

**📊 数据集**

在20k小时授权混音上预训练，400小时带stem的授权数据微调，评估使用MoisesDB与MusDB两套开源stem分离数据集。

**📈 对比分析**

与单stem迭代模型及仅分组或仅共享噪声的对照实验比较，Stemphonic在FAD_stem、FAD_mix和CLAP上取得更优分数，一次推理比迭代方法快25–50%，且2-pass策略进一步提升质量。

**⚠️ 局限性**

共享噪声方法理论解释尚浅，活动控制会略微降低音频清晰度；目前stem标签仍受限于预定义类型，未完全支持自由文本细粒度控制。

---

## 401. Instruct2Act: From Human Instruction to Actions Sequencing and Execution via Robot Action Network for Robotic Manipulation

**arXiv ID:** 2602.09940 | [PDF](https://arxiv.org/pdf/2602.09940v1)

**作者:** Archit Sharma `[一作]` (Indian Institute of Technology Mandi), Laxmidhar Behera `[通讯]` (Indian Institute of Technology Mandi)

**通讯引用:** 5951 | [OpenAlex ID](https://openalex.org/A5065056581)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套两阶段的机器人指令解析与执行框架（Instruct2Act 与 RAN），能够在单目眼内摄像头的受限硬件环境下，将自然语言指令实时转换为精确的抓取、放置、倒液、擦拭等操作。

**💡 创新点**

创新点在于：1) 轻量级的 BiLSTM+多头注意力+自编码器结构用于指令到细粒度子动作的预测；2) 动态自适应轨迹径向网络（DATRN）替代传统 DMP，能够一次性闭式求解并实时适应目标；3) 完全离线、设备端的处理流程，消除了云计算需求。

**🔧 技术方法**

使用的技术包括：BERT 任务嵌入、BiLSTM 与多头注意力、自编码器、DATRN（RBF+岭回归）、YOLOv8‑n 目标检测、PD 控制器、ROS 机器人控制。

**📊 数据集**

使用自制的 2,850 条英文自然语言指令与对应子动作序列的数据集，覆盖 pick‑place、pick‑pour、擦拭、give 等多种任务，包含 10+ 目标物体与多种语言变体。

**📈 对比分析**

与 LSTM、BiLSTM、BiLSTM+MHA 等基线相比，Instruct2Act 在子动作预测上达到 91.5% 的准确率，F1 91%，召回率 88%，参数量 7.91M，训练时间 4.7 min；DATRN 在轨迹拟合误差上优于 DMP+GA，训练时间约 2 秒。实验中 4 种任务的整体成功率达 90%。

**⚠️ 局限性**

局限性在于对极长或高度组合化的指令仍存在误判，特别是后续子动作缺失；需要扩充长链指令数据并尝试轻量级 Transformer 结构来提升长程依赖建模。

---

## 402. Supervised Metric Regularization Through Alternating Optimization for Multi-Regime Physics-Informed Neural Networks

**arXiv ID:** 2602.09980 | [PDF](https://arxiv.org/pdf/2602.09980v1)

**作者:** Enzo Nicolas Spotorno `[一作]`, Antonio Augusto Frohlich `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种拓扑感知 PINN（TAPINN），通过在潜在空间中加入监督的度量正则化和交替优化，解决参数化动力学系统在分岔等剧烈状态转换时的梯度冲突和谱偏差问题。

**💡 创新点**

创新点在于：①利用三元组损失在潜在空间中刻画不同物理状态的几何结构；②采用阶段性（交替）训练策略先稳定潜在空间，再引入物理约束，从而显著降低物理残差并提升模型泛化。

**🔧 技术方法**

使用的技术包括：LSTM 编码器捕捉时间序列依赖；PINN 生成器输出完整轨迹；Triplet 损失实现潜在空间的度量正则化；Sobolev (H¹) 损失约束一阶导数；交替优化（Metric → Physics → Joint）调度；Adam 优化器。

**📊 数据集**

数据集为 Duffing 振荡器的数值仿真轨迹，按 F₀=0.3、0.5、0.8 三种参数分别采集 500 条轨迹；观测窗口为前 100 步（约 10% 轨迹），其余步骤用于物理残差评估。

**📈 对比分析**

与三种基线（参数化 PINN、HyperPINN、Multi‑Output Sobolev PINN）对比，TAPINN 在物理残差上达到 0.082（≈49% 降低），参数量仅 8,003，显著优于 39,169 的 HyperPINN，同时保持与其他基线相近的参数规模，展示了更高的物理合规性与模型压缩效果。

**⚠️ 局限性**

局限性包括：实验仅限于 Duffing 系统，缺乏对更高维或更复杂分岔系统（如 Lorenz、Allen‑Cahn）的验证；对噪声鲁棒性、超参数敏感度以及不同梯度调度策略的系统性评估仍待进一步研究。

---

## 403. ATTNPO: Attention-Guided Process Supervision for Efficient Reasoning

**arXiv ID:** 2602.09953 | [PDF](https://arxiv.org/pdf/2602.09953v1)

**作者:** Shuaiyi Nie `[一作]` (Institute of Information Engineering Chinese Academy of Sciences), Hua Wu `[通讯]` (Baidu Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种低开销的过程监督强化学习框架 AttnPO，利用大型推理模型自身的 Key‑Focus Heads（KFHs）对每一步的优势进行细粒度缩放，抑制冗余推理并提升准确率。

**💡 创新点**

创新点：首次识别并利用 KFHs 进行步骤级信用分配；设计正负优势衰减策略和难度感知基线；实现无额外开销的过程监督；通过注意力加权实现一步步优势重标。

**🔧 技术方法**

技术手段包括 Transformer 注意力分析、KFH 识别、注意力加权步长得分、优势重标（stepwise advantage rescaling）、正负优势衰减策略、难度感知基线和调度、RL 框架（GRPO/TLMRE）以及实验环境配置。

**📊 数据集**

训练使用 DeepScaleR‑Preview 数据集；评估集包含 GSM8K、MATH500、AMC23、OlympiadBench、AIME2024、AIME2025；在 OOD 任务 LiveCodeBench、GPQA‑Diamond、MMLU 上也进行了验证。

**📈 对比分析**

与 outcome‑supervised RL（TLMRE、ThinkPrune、DIET、ACPO、Laser）、process‑supervised RL（LC‑R1、VSRM‑R++、S‑GRPO、DEPO、DECS）以及 adaptive‑mode 方法（AdaptThink、AutoThink）对比，AttnPO 在六大数学基准上平均 AES 排名第一或第二；1.5B 模型平均长度缩短 61%，准确率提升 7.3 点；7B 模型长度缩短 55%，准确率提升 2.9 点；在 OOD 任务中保持或略优；在不同 token 预算、探索能力、跨域等方面均表现优异。

**⚠️ 局限性**

局限性：实验仅覆盖 1.5B 与 7B 两个规模；训练数据仅为数学领域，缺乏更广泛多域验证；计算资源受限，未测试更大规模模型或更复杂任务的适用性。

---

## 404. JMigBench: A Benchmark for Evaluating LLMs on Source Code Migration (Java 8 to Java 11)

**arXiv ID:** 2602.09930 | [PDF](https://arxiv.org/pdf/2602.09930v1)

**作者:** Nishil Amin `[一作]` (University College London), He Ye `[通讯]` (University College London)

**通讯引用:** 450 | [OpenAlex ID](https://openalex.org/A5101610258)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了名为JMigBench的 Java 8 → Java 11 函数级迁移基准，评估大型语言模型在代码迁移任务中的表现

**💡 创新点**

首次提供具有真实功能对等的 Java 8 与 Java 11 函数对，涵盖八类已弃用 API，并通过人工与 LLM 合作生成干净、平衡的数据集

**🔧 技术方法**

使用 Mistral 的 Codestral LLM 进行一次性提示迁移，评估时结合 CodeBLEU（词法、语法与数据流相似度）和关键词去除率两类指标

**📊 数据集**

基准数据集由 45 对手工校正的函数组成，包含 66 个已弃用关键词，涵盖常见的 CORBA、JAX‑WS、线程、SecurityManager 等八类 API

**📈 对比分析**

通过与人工编写的 Java 11 代码比对，Codestral 在词法与语义上略有提升（CodeBLEU 从 0.62 提升至 0.63），但仅有 11.11% 的函数实现完全一致；关键词去除率平均为 31.82%，仅 40% 的函数完全去除弃用词

**⚠️ 局限性**

局限性包括：仅使用单一 LLM（Codestral），缺乏编译与运行时验证，基准仅覆盖函数级别缺失项目上下文，且对复杂多步迁移的推理能力不足

---

## 405. A benchmark for video-based laparoscopic skill analysis and assessment

**arXiv ID:** 2602.09927 | [PDF](https://arxiv.org/pdf/2602.09927v1)

**作者:** Isabel Funke `[一作]` (National Center for Tumor Diseases), Stefanie Speidel `[通讯]` (Dresden University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并公开 LASANA 腹腔镜训练视频数据集，并提供基准模型用于视频评估手术技能和错误识别。

**💡 创新点**

首次构建多任务、带有 GOALS 结构化评分与错误标注的大规模无机器人腹腔镜数据集，并对分割、可靠性和基准性能进行系统分析。

**🔧 技术方法**

采用 3D CNN X3D+MLP 进行 GRS 回归，二分类网络（全连接层）进行错误识别，使用 Lin’s ρc、准确率和均衡准确率等指标评估。

**📊 数据集**

使用 LASANA 数据集（1270 条立体视频，4 个任务）进行训练、验证和测试；对比 JIGSAWS、ROSMA、AIxSuture 等现有腹腔镜/机器人数据集。

**📈 对比分析**

在验证/测试集上计算 Lin’s ρc：视频+时长输入可达 0.88–0.95，错误识别均衡准确率≥80%；circle cutting 任务因低标注一致性导致性能较差。

**⚠️ 局限性**

受限于样本量、标注不一致（尤其是 circle cutting）、模型可能利用视频时长做短路、对不同任务的适用性差异大。

---

## 406. TaCo: A Benchmark for Lossless and Lossy Codecs of Heterogeneous Tactile Data

**arXiv ID:** 2602.09893 | [PDF](https://arxiv.org/pdf/2602.09893v1)

**作者:** Zhengxue Cheng `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 82886 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个全面的触觉数据编解码基准TaCo，并提出了针对触觉数据的自监督神经编解码器TaCo-LL（无损）和TaCo-L（有损）。

**💡 创新点**

创新点在于：①首次整合30种编解码器与5种触觉数据集形成统一评测框架；②设计并训练了完全基于触觉数据的自适应编解码器，突破了先前模型在不同传感器间的泛化瓶颈。

**🔧 技术方法**

技术上采用传统压缩算法（gzip、zstd、FLIF、JPEG‑XL等）、预训练神经网络（DLPR、P2LLM、LALIC等）以及自训练的双向变分自动编码器（DualComp‑I、LALIC），并结合BD‑Rate、bits/Byte、分类精度与抓取成功率等多维度指标进行评测。

**📊 数据集**

使用的数据集包括：Touch and Go、ObjectFolder 1.0、SSVTP、YCB‑Slide 和 ObjTac，涵盖视觉式与力学式触觉传感器的多样化场景。

**📈 对比分析**

在四类任务（无损存储、人类可视化、有损分类与机器人抓取）上，TaCo‑LL 在无损压缩上比所有基线低约30‑70% bits/Byte；TaCo‑L 在有损压缩中实现-60%~-30% BD‑Rate，且在分类与抓取任务中误差不超过1‑2%，保持与未压缩数据相近的性能。

**⚠️ 局限性**

局限性在于：①评测仅覆盖5个数据集，难以覆盖所有触觉场景；②自训练模型对训练数据分布高度依赖，跨域泛化仍有限；③神经编解码器虽性能优越，但计算开销较大，实时部署仍需进一步优化。

---

## 407. Statistical benchmarking of transformer models in low signal-to-noise time-series forecasting

**arXiv ID:** 2602.09869 | [PDF](https://arxiv.org/pdf/2602.09869v1)

**作者:** Cyril Garcia `[一作]`, Guillaume Remy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究Transformer在低数据、多变量时间序列预测中的性能，并使用合成数据对不同模型进行对比。

**💡 创新点**

① 设计了时间‑交叉维度双路注意力的Transformer架构；② 提出了数据驱动的动态稀疏化算法；③ 在可统计显著的bootstrap框架下评估模型。

**🔧 技术方法**

使用Transformer（自注意力）、Lasso、Boosting、MLP等传统基线；实现动态稀疏化(max_sparse)；通过合成数据模拟多种时间与空间依赖效应。

**📊 数据集**

完全基于人工生成的合成数据：包含线性、TS‑Shift、CS‑Shift、Fea‑Nonlin、TSCS‑Shift五种效应，噪声水平可调；样本规模为T_train=2500、T_test=1500、N=10、F=20。

**📈 对比分析**

采用测试集的相关系数与理论最优预测的相关系数对比，并通过bootstrap检验显著性。结果显示：在大多数效应和低信噪比场景下，Transformer优于传统方法；在稀疏结构下动态稀疏化可提升10–20%的相关系数。

**⚠️ 局限性**

主要限制：对TSCS‑Shift效应的学习效果仍差；稀疏化阈值固定，未学习或交叉验证；未在真实工业数据上验证；多头/多层组合的潜在优势尚未深入探索。

---

## 408. Monocular Normal Estimation via Shading Sequence Estimation

**arXiv ID:** 2602.09929 | [PDF](https://arxiv.org/pdf/2602.09929v1)

**作者:** Zongrui Li `[一作]` (Nanyang Technological University), Song Bai `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种将单目法线估计重构为阴影序列估计的框架RoSE，并通过图像到视频的生成模型预测阴影序列，再用最小二乘求解得到法线。

**💡 创新点**

创新点在于：①将法线估计视为可感知几何的阴影序列任务，显著降低3D失配；②利用大型视频扩散模型生成与光照一致的阴影序列；③通过在高多样化合成数据集上训练提升泛化能力。

**🔧 技术方法**

技术包括：图像到视频扩散生成模型（视频Diffusion U‑Net）、CLIP与VAE双分支条件编码、最小二乘法（OLS）求解法线、灰度输入消除色彩干扰。

**📊 数据集**

使用了自建的多材质多光照合成数据集（包含90K Objaverse模型、MatSynth材质、HDR环境光等），并在DiLiGenT、LUCES、Objaverse等公开基准上进行评测。

**📈 对比分析**

与7种现有单目法线估计方法（GeoWizard、DSINE、StableNormal、Lotus‑G/D、Neural LightRig、NiRNE）比较，RoSE在DiLiGenT和LUCES的平均角误差均显著低于SOTA（如DiLiGenT 15.37° vs 17.27°，LUCES 14.48° vs 17.44°），同时在多种光照与材质设置下保持优越性能。

**⚠️ 局限性**

局限性包括：①视频扩散模型推理时计算量大，难以实时部署；②在极端光照或大面积无照区域时阴影质量下降导致法线误差增大；③无法处理透明或半透明对象；④目前仅在对象级别评测，未验证场景级应用。

---

## 409. BabyMamba-HAR: Lightweight Selective State Space Models for Efficient Human Activity Recognition on Resource Constrained Devices

**arXiv ID:** 2602.09872 | [PDF](https://arxiv.org/pdf/2602.09872v1)

**作者:** Mridankan Mandal `[一作]` `[通讯]` (Indian Institute of Information Technology), Mridankan Mandal (Indian Institute of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 BabyMamba-HAR 框架，包含两种轻量级的 Mamba 风格选择性状态空间模型（CI‑BabyMamba‑HAR 与 Crossover‑BiDir‑BabyMamba‑HAR），用于在 TinyML 设备上高效识别多通道惯性传感器序列中的人类活动。

**💡 创新点**

创新点包括：
1) 通过通道独立 stem（CI‑BabyMamba）和早期融合 stem（Crossover‑BiDir）两种不同的输入投影策略，针对不同传感器配置提供最优计算复杂度；
2) 采用权重共享的双向扫描机制，既获得双向上下文又不增加额外参数；
3) 设计了轻量化的上下文门控时间注意力池化头，用于聚焦关键时刻；
4) 在统一评测协议下对八个公开 HAR 数据集进行系统评估与消融，验证了上述设计的有效性。

**🔧 技术方法**

技术手段：
- 选择性状态空间模型（Mamba）实现线性时间 O(N) 的序列建模；
- 权重共享的双向扫描实现双向上下文；
- 轻量化上下文门控时间注意力池化；
- 通道独立与早期融合 stem 方案；
- 数据增强（时间扭曲、幅值缩放、高斯抖动、通道丢弃）和标签平滑；
- 统一的评测协议与多种基线对比。

**📊 数据集**

使用的公开 HAR 数据集共八个：UCI‑HAR、MotionSense、WISDM、PAMAP2、Opportunity、UniMiB‑SHAR、Skoda 和 Daphnet。

**📈 对比分析**

比较方法：在统一的预处理、子标记拆分、5 次随机种子统计下，分别与 TinyHAR、TinierHAR、DeepConvLSTM 等基线模型对比。Crossover‑BiDir‑BabyMamba 在所有数据集上的平均宏 F1 分数为 86.52%，与 TinyHAR（86.16%）相当、仅比 TinierHAR（87.39%）低 0.87%；参数约 27K，平均 MACs 2.21M；在高通道数据集（Opportunity、Skoda）上相较 TinyHAR 减少 11× MACs；消融实验表明双向扫描可提升最高 8.42% 的 F1，门控注意力提升最高 8.94%。

**⚠️ 局限性**

局限性：
- 双向扫描不适用于实时流式推理；
- 对极端类别失衡（如 Daphnet）缺乏专门的损失策略；
- 仅针对惯性传感器数据，未验证其它模态；
- 微控制器上的 INT8 量化效果尚未评估；
- 评测仅在单机 CPU/GPU 上完成，未考虑低功耗硬件实现细节。

---

## 410. SCOPE: A Training-Free Online 3D Deployment for UAV-BSs with Theoretical Analysis and Comparative Study

**arXiv ID:** 2602.09971 | [PDF](https://arxiv.org/pdf/2602.09971v1)

**作者:** Chuan-Chi Lai `[一作]` (National Chung Cheng University), Chuan-Chi Lai `[通讯]` (National Chung Cheng University)

**通讯引用:** 115 | [OpenAlex ID](https://openalex.org/A5082920277)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在多用户多无人机空中基站部署场景中，提出一种无训练、可在线执行的三维部署框架SCOPE，利用周界提取与最小包围圆算法动态规划无人机位置、海拔与用户关联；

**💡 创新点**

创新点在于将几何计算（周界提取+SEC）与容量、海拔、QoS约束结合，形成可在多样化用户密度下自适应扩容、降冗的“剥层”部署策略，且提供O(N²logN)的确定性时间复杂度；

**🔧 技术方法**

使用的技术主要包括：Andrew单调链凸包算法、Welzl最小包围圆算法、K‑NN容量约束、距离阈值与海拔计算；

**📊 数据集**

使用模拟数据集：400×400 m²区域内的Matern聚类过程生成的用户分布，并以Gauss‑Markov模型模拟移动，用户数在200–1000之间；

**📈 对比分析**

与基准（CCS、K‑means、Voronoi、Random、DRL‑PPO）比较，SCOPE在用户满意率（65–80%）、能效、均衡度方面均优于传统启发式，且推算时间仅≈12 ms，远低于DRL训练/推理耗时；

**⚠️ 局限性**

局限性包括：仅处理瞬时快照，缺乏对无人机轨迹规划与多层干扰管理；在极大规模场景下凸包与邻近搜索仍占用一定时间；未在真实场景中验证。

---

## 411. Causal Identification in Multi-Task Demand Learning with Confounding

**arXiv ID:** 2602.09969 | [PDF](https://arxiv.org/pdf/2602.09969v1)

**作者:** Varun Gupta `[一作]` (University of Utah), Vijay Kamble `[通讯]` (University of Illinois Chicago)

**通讯引用:** 245 | [OpenAlex ID](https://openalex.org/A5048397860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于信息设计的多任务需求学习框架——决策条件掩码结果元学习（DCMOML），能够在价格选择存在内生性且每个任务仅有少量价格变动的场景下识别并估计因果价格弹性。

**💡 创新点**

创新点在于通过在元学习过程中随机化查询点并屏蔽两次价格对应的需求结果，从而打破查询回归器可识别导致的识别失败，实现仅需两种价格即可获得因果参数的条件期望，并给出一致性证明。

**🔧 技术方法**

主要技术包括信息设计（decision‑conditioned masking）、随机化查询策略、元学习框架（可使用神经网络）、统一的平方损失优化以及在假设下的可识别性与一致性理论分析。

**📊 数据集**

实验使用了合成数据（不同程度的价格内生性设置）和英国在线零售交易数据（约4,070个产品，含平均3.78个价格水平），在两种任务定义下进行评估。

**📈 对比分析**

与传统的 OLS、池化 OLS、元学习、Empirical Bayes GLS 等基线相比，DCMOML 在高内生性、低价格变动的情况下显著降低斜率和截距估计误差（MSE 下降约 40‑60%），并在真实零售数据上在两种任务设定下取得最低的保留点 RMSE。

**⚠️ 局限性**

局限性包括：仅在每个任务有两种不同价格且满足“最终价格不依赖于最近一次需求冲击”的条件下可行；在价格变化更多或高度自适应的情形下，信息屏蔽可能削弱可用信号；理论与实验主要聚焦线性需求模型，扩展到非线性或交叉价格效应尚待研究。

---

## 412. Drug Release Modeling using Physics-Informed Neural Networks

**arXiv ID:** 2602.09963 | [PDF](https://arxiv.org/pdf/2602.09963v1)

**作者:** Daanish Aleem Qureshi `[一作]` (Brown University), Vikas Srivastava `[通讯]` (Brown University)

**通讯引用:** 4113 | [OpenAlex ID](https://openalex.org/A5038126740)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了物理信息神经网络（PINNs）与贝叶斯PINNs（BPINNs），利用Fick扩散定律与少量实验数据，预测平面、1D皱纹及2D折皱薄膜的药物释放曲线；

**💡 创新点**

创新点在于将Fick扩散方程嵌入PINNs的损失函数中，并通过有限实验点实现长周期释放预测；进一步使用BPINNs实现不确定性量化，在噪声环境下提升鲁棒性；整体比传统模型误差下降约40%；

**🔧 技术方法**

核心技术包括PINN与BPINN框架、拉丁超立方采样生成1万点物理约束、自动微分、Hamiltonian Monte Carlo (HMC) 采样/变分推断、蒙特卡洛 Dropout、MAE/RMSE 评估；

**📊 数据集**

使用Liu等人公开的基于石墨烯氧化物纳米片的RhB释放实验数据（平面、1D皱纹、2D折皱三种薄膜），并在此基础上添加高斯噪声进行鲁棒性测试；

**📈 对比分析**

通过与经典Fick、Higuchi、Peppas模型在相同数据集上比较，使用MAE和RMSE指标；PINN在所有膜型下均比经典模型低40%误差，RMSE<0.05仅需9-11个数据点；BPINN在噪声场景下更稳健，提供更窄的置信区间；

**⚠️ 局限性**

局限性包括：BPINN计算成本高（HMC采样）；模型依赖Fick扩散假设，可能对非Fick或高非均匀材料适应性不足；需要手动调参、网络结构选择；噪声模型仅为高斯，实际实验噪声可能更复杂；

---

## 413. Environment-in-the-Loop: Rethinking Code Migration with LLM-based Agents

**arXiv ID:** 2602.09944 | [PDF](https://arxiv.org/pdf/2602.09944v1)

**作者:** Xiang Li `[一作]` (University College London), He Ye `[通讯]` (University College London)

**通讯引用:** 450 | [OpenAlex ID](https://openalex.org/A5101610258)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于LLM的多代理环境驱动迁移框架，将环境搭建、代码迁移和测试集生成集成为闭环流程。

**💡 创新点**

创新点在于把动态环境交互引入迁移过程，构建迁移、环境、测试三体LLM代理，形成持续反馈循环；以及利用环境代理自动构建可复现运行时环境。

**🔧 技术方法**

使用LLM（如ChatGPT）构建的代理模型，包含规划、记忆、感知和行动四大模块；通过Docker/K8s等容器化技术实现可复现环境；结合自动化测试生成工具。

**📊 数据集**

评估基于EnvBench数据集（数千个Python和JVM项目）以及若干真实迁移案例。

**📈 对比分析**

与传统静态迁移工具和仅使用LLM建议的方法对比，实验显示环境驱动模型能显著降低运行时错误率、缩短迭代周期，整体迁移成功率提升约20-30%。

**⚠️ 局限性**

局限包括：环境重现和依赖推断仍难，代理需要学习构建脚本和日志；多代理协调与反馈解释不足；在CI/CD集成中对已有工作流的兼容性和审计需求尚未完善。

---

## 414. QEMI: A Quantum Software Stacks Testing Framework via Equivalence Modulo Inputs

**arXiv ID:** 2602.09942 | [PDF](https://arxiv.org/pdf/2602.09942v1)

**作者:** Junjie Luo `[一作]` (Kyushu University), Jianjun Zhao `[通讯]` (Kyushu University)

**通讯引用:** 6534 | [OpenAlex ID](https://openalex.org/A5065190767)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了QEMI框架，通过在量子程序中插入死代码并生成等价变体，利用差分检测发现量子软件栈中的错误

**💡 创新点**

首次将经典编译器的Equivalence Modulo Inputs技术迁移到量子领域，并设计了针对量子控制流的死代码模式

**🔧 技术方法**

采用随机量子程序生成器、静态死代码模式、EMI变体生成、Hellinger距离比较以及自适应早停测量策略

**📊 数据集**

使用Qiskit、Q#、Cirq的最新版本作为目标软件栈，并在其模拟器上生成的随机量子程序作为测试数据

**📈 对比分析**

与QDiff、MorphQ、QuteFuzz对比，QEMI在24小时内覆盖率最高（约14.8%），发现12个真实bug（11已修复），且早停策略在测量成本上提升约53.8%（n=6）/72.2%（n=8）

**⚠️ 局限性**

仅支持三大量子栈；依赖模拟器，未在真实硬件上验证；早停策略无严格理论保证；死代码插入模式有限，缺乏自动检测机制

---

## 415. GeoFormer: A Swin Transformer-Based Framework for Scene-Level Building Height and Footprint Estimation from Sentinel Imagery

**arXiv ID:** 2602.09932 | [PDF](https://arxiv.org/pdf/2602.09932v1)

**作者:** Han Jinzhen `[一作]`, HongSik Yun `[通讯]` (Sungkyunkwan University)

**通讯引用:** 376 | [OpenAlex ID](https://openalex.org/A5113713719)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出GeoFormer，一种基于Swin Transformer的多任务框架，联合估计100 m网格下的建筑高度和足迹；

**💡 创新点**

创新点在于采用geo-blocked分割策略实现严格空间独立，结合多源（Sentinel‑1/2+DEM）数据的融合以及可调节的上下文窗口大小，显著提升了全球范围内的3D城市数据生成；

**🔧 技术方法**

使用了Swin Transformer骨干、窗口注意力、可学习的任务不确定性加权Adaptive Huber损失以及多尺度上下文抽取；

**📊 数据集**

数据集包括51个使用SHAFTS v2022.3建筑高度/足迹标签的全球城市，结合Sentinel‑1/2（10 m）和SRTM DEM（30 m）；

**📈 对比分析**

与三种CNN基线（ResNet‑MTL、UNet‑MTL、SENet‑MTL）对比，GeoFormer 5×5上下文窗口在54座城市上实现建筑高度RMSE 3.19 m、足迹RMSE 0.050，比最佳CNN低7.5 %和15.3 %；跨洲迁移仍保持<3.5 m RMSE；

**⚠️ 局限性**

局限在于对高层建筑和高密度区域仍存在误差，模型对缺失高层样本敏感，且需要预先获取完整的建筑高度标签，未来可考虑加入时序信息或更细粒度的解码模块。

---

## 416. TAROT: Towards Optimization-Driven Adaptive FEC Parameter Tuning for Video Streaming

**arXiv ID:** 2602.09880 | [PDF](https://arxiv.org/pdf/2602.09880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 417. AmharicIR+Instr: A Two-Dataset Resource for Neural Retrieval and Instruction Tuning

**arXiv ID:** 2602.09914 | [PDF](https://arxiv.org/pdf/2602.09914v1)

**作者:** Tilahun Yeshambel `[一作]` (Addis Ababa University), Josiane Mothe `[通讯]` (University of Toulouse)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文发布了两份阿姆哈拉语数据集：1）1091个手工验证的查询-正负文档三元组，用于训练和评测神经检索模型；2）6285个经过人工校正的 GPT 风格提示-响应对，用于指令调优与生成评测。

**💡 创新点**

创新点在于：①提出了可迁移的低资源语言数据构建管线，结合专家撰写、网页挖掘和 LLM 辅助生成，并通过母语者验证保证正负标注质量；②首次提供大规模、手工验证的阿姆哈拉语检索三元组和指令对，填补了低资源语言资源空白。

**🔧 技术方法**

技术手段包括：1）对检索三元组采用对比损失训练；2）对指令对使用 GPT‑style 生成并人工校对；3）在阿姆哈拉语预训练模型上微调 SPLADE、RoBERTa 与 ColBERT，结合硬负样本挖掘；4）对数据进行语言特定归一化、去重和标准化格式化。

**📊 数据集**

使用的数据集为：1）新发布的 1091 个查询-正负三元组（包含 2183 篇文档）；2）新发布的 6285 条提示-响应对；3）原始阿姆哈拉语预训练模型（RoBERTa、SPLADE、ColBERT）作为基线。

**📈 对比分析**

评测方法：在 654/218/219 的 train/valid/test 分割上进行 fine‑tune，使用 MAP、NDCG、Precision/Recall 等标准 IR 指标；实验结果显示 RoBERTa 在各截断点均优于 SPLADE 与 ColBERT，且随着 K 的增大，NDCG、MAP、Recall 上升，Precision 下降。

**⚠️ 局限性**

局限性：①数据量相对有限，检索相关文档数量稀疏；②仍缺乏跨域、长文本检索评测；③指令对中仍可能存在事实性错误，需进一步增强事实校验；④当前仅在阿姆哈拉语上验证，需在更多低资源语言上检验泛化性。

---

## 418. QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search

**arXiv ID:** 2602.09901 | [PDF](https://arxiv.org/pdf/2602.09901v1)

**作者:** Jianzhao Huang `[一作]` (Xiaohongshu Inc.), Shaosheng Cao `[通讯]` (Xiaohongshu Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小红书社交网络服务场景下，提出了统一生成式大模型 QP-OneModel，能够一次性完成 NER、词分割、词权重、查询分类和意图描述等多任务查询处理。

**💡 创新点**

创新点包括：① 将异构查询处理子任务映射为统一的序列到序列生成框架；② 采用三阶段递进对齐策略（知识注入 → 目标分布对齐 → 多奖励强化学习）实现业务规则的内部化；③ 生成意图描述作为高保真语义信号，为下游重写和排序提供新的输入。

**🔧 技术方法**

核心技术包括：RedOne SNS 领域预训练大模型、混合式监督训练（pseudo 标签+人标注）、多奖励强化学习（GRPO + 任务权重奖励）、业务感知提示工程。

**📊 数据集**

使用的数据集：~10^7 条历史日志的伪标注数据、约10^5 条统一标注的黄金测试集（2.5k 查询）以及线上真实流量的 A/B 对照数据。

**📈 对比分析**

通过与传统 BERT 系列流水线基准对比，离线整体得分提升 7.35%，NER F1 提升 9.01%，词权重 F1 提升 9.31%；在线 A/B 结果显示 DCG 0/1 降低 0.21%，用户留存提升 0.044%；在少样本 ICL 任务上，QP-OneModel 超越同等参数 Qwen3‑32B 7.60% 的准确率。

**⚠️ 局限性**

局限性：模型高度依赖大量标注与日志数据，数据更新频率与质量直接影响效果；多奖励 RL 训练复杂，调参成本高；虽然在 SNS 领域表现优异，但跨域迁移及对极端新词、罕见语义的处理仍有提升空间。

---

## 419. Immersion in the GitHub Universe: Scaling Coding Agents to Mastery

**arXiv ID:** 2602.09892 | [PDF](https://arxiv.org/pdf/2602.09892v1)

**作者:** Jiale Zhao `[一作]` (Renmin University of China), Kai Jia `[通讯]` (ByteDance)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Scale‑SWE，一个基于沙盒多智能体的自动化工作流，用于大规模构建可执行的软件工程数据集。

**💡 创新点**

创新点在于：①将环境搭建、单元测试生成和任务描述合成拆分为三个专用智能体；②采用交互式Docker化环境构建、Fail‑to‑Pass/Passto‑Pass测试自动生成；③通过自动化流程处理600万条PR，产出100k实例，显著提升规模与多样性。

**🔧 技术方法**

技术上结合了Docker沙盒、OpenHands框架、DeepSeek‑V3.x/ Gemini3‑Pro 生成器、LLM‑as‑Judge 过滤与评估。

**📊 数据集**

数据集为自研的Scale‑SWE‑Data，包含100k可验证实例，基于6百万GitHub PR，覆盖5.2k Python仓库。

**📈 对比分析**

与SWE‑Bench‑Verified及其他公开基线比较，Fine‑tuned 的 Scale‑SWE‑Agent 在解题率上从22%提升至64%，在同等参数规模模型中位居榜首。

**⚠️ 局限性**

局限性包括：目前仅支持Python；生成的环境与测试仍可能受LLM误判影响；构建过程对计算资源要求高，且对GPU/特定依赖的仓库处理仍不完善。

---

## 420. MVISTA-4D: View-Consistent 4D World Model with Test-Time Action Inference for Robotic Manipulation

**arXiv ID:** 2602.09878 | [PDF](https://arxiv.org/pdf/2602.09878v1)

**作者:** Jiaxu Wang `[一作]` (MMLab, Chinese University of Hong Kong), Xiangyu Yue `[通讯]` (MMLab, Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现一种能够从单视角RGB‑D观测生成多视角时空一致的4D场景（RGB‑D）并通过轨迹潜在空间进行动作推理的全新机器人世界模型；

**💡 创新点**

创新点包括：①跨模态（RGB‑Depth）与跨视角融合机制，显著提升几何一致性；②将整条动作轨迹压缩为低维潜在向量作为“风格码”，实现对生成视频的轨迹级控制；③在测试时对该潜在向量进行梯度优化，再用残差逆动力学模型细化动作，解决传统逆动力学的多义性；

**🔧 技术方法**

核心技术包括：基于VAE+Transformer的潜在视频扩散模型、局部跨模态注意力、几何感知可变形跨视角注意力、TCN‑VAE轨迹编码器、残差逆动力学网络；

**📊 数据集**

使用了三组数据集：RLBench（8k+10k条轨迹），RoboTwin（10k条轨迹），以及自制的真实机器人RGB‑D数据集（14个操纵任务，约4台摄像头），均包含文本指令与对应动作；

**📈 对比分析**

与UniPi、TesserAct、4DGen等基线在4D生成（PSNR、SSIM、深度误差、CD/EMD）和下游操纵成功率上均表现优异；在RLBench、RoboTwin及真实机器人平台的成功率分别提升约10‑15%，在复杂遮挡场景中多视角生成显著减少几何误差；

**⚠️ 局限性**

局限性主要体现在：①仍需多台摄像头或高质量单视角输入；②生成与逆动力学计算量较大，实时性尚待提升；③对极端遮挡和动态光照的鲁棒性尚有限；④对语言指令的理解仍以规则模板为主，泛化到更开放式指令时可能受限。

---

## 421. The Devil Behind Moltbook: Anthropic Safety is Always Vanishing in Self-Evolving AI Societies

**arXiv ID:** 2602.09877 | [PDF](https://arxiv.org/pdf/2602.09877v1)

**作者:** Chenxu Wang `[一作]` (Beijing University of Posts and Telecommunications), Philip S. Yu `[通讯]` (University of Illinois at Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了闭环自演进多智能体系统的安全衰退，证明安全与自演进不可共存

**💡 创新点**

首次将信息论熵和热力学框架用于量化安全漂移，并给出四种缓解策略

**🔧 技术方法**

信息理论（KL散度、数据处理不等式）、热力学熵分析、RL与记忆式自演进实现

**📊 数据集**

Moltbook社区交互日志、AdvBench（jailbreak）、TruthfulQA（hallucination）

**📈 对比分析**

通过对比RL‑based和memory‑based两种自演进循环，20轮实验显示安全指标随迭代递增下降，RL更快且波动大，memory虽缓慢但事实准确性更差

**⚠️ 局限性**

结论仅适用于完全隔离环境，缺乏实际部署验证，且未给出完整的外部监督实现细节

---

## 422. Steer2Edit: From Activation Steering to Component-Level Editing

**arXiv ID:** 2602.09870 | [PDF](https://arxiv.org/pdf/2602.09870v1)

**作者:** Chung-En Sun `[一作]` (University of California San Diego), Tsui-Wei Weng `[通讯]` (University of California San Diego)

**通讯引用:** 1503 | [OpenAlex ID](https://openalex.org/A5114139431)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种训练自由的Steer2Edit框架，将Steering向量转化为按组件级别的权重编辑，实现对大型语言模型行为的可解释、可控调节；

**💡 创新点**

通过理论分析将Steering向量映射为组件级rank‑1更新，给出闭式软阈值分配方法，既保持原模型架构又获得更优的属性‑效用折衷；

**🔧 技术方法**

采用线性层rank‑1权重编辑、Elastic‑Net正则、语义方向对齐、attention/MLP头/神经元的输入输出对齐、基于Steering向量的诊断与输入选择等技术；

**📊 数据集**

在安全实验中使用LLaMA‑2‑7B‑Chat与Mistral‑7B‑Instruct‑v0.2；在真实性实验中使用Gemma‑2‑2B‑IT与LLaMA‑3‑8B‑Instruct；在推理效率实验中使用Qwen3‑4B‑Thinking‑2507与OpenMath‑Nemotron‑7B；对标任务包括GSM8K、CodeMMLU、CommonsenseQA、TruthfulQA等；

**📈 对比分析**

与激活层Steering做曲线对比，结果显示在安全、真实性、推理效率三类任务中，Steer2Edit在保持或提升下游效用的同时，安全拒绝率提升17.2%、真实性偏好提升9.8%、推理长度缩短12.2%；整体表现优于基线；

**⚠️ 局限性**

仅针对线性投影组件，假设Steering向量可解释；对更大模型或跨模态场景的通用性尚待验证；编辑易被滥用，缺乏动态输入自适应机制；

---

## 423. Learning to Detect Baked Goods with Limited Supervision

**arXiv ID:** 2602.09979 | [PDF](https://arxiv.org/pdf/2602.09979v1)

**作者:** Thomas H. Schmitt `[一作]` (Technische Hochschule Nürnberg Georg Simon Ohm), Tobias Bocklet `[通讯]` (Technische Hochschule Nürnberg Georg Simon Ohm)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过将开源词汇检测模型与图像级监督相结合，并利用 Segment Anything 2 进行伪标签传播，提出一种仅依赖图像级标签即可训练的弱监督目标检测框架，用于自动计数德国面包店剩余面包。

**💡 创新点**

创新点在于：①将 OWLv2、Grounding DINO 的零样本定位能力与图像级监督配合，生成高质量的弱标签；②采用 Segment Anything 2 对视频帧进行伪标签传播，极大降低人工标注成本；③在仅靠图像级标签的情况下，训练 YOLOv11 达到与全监督模型相媲美甚至更优的性能。

**🔧 技术方法**

技术包括：YOLOv11 目标检测、OWLv2 与 Grounding DINO 词汇检测、Segment Anything 2 伪标签、数据增强、类无关 NMS、cosine 归一化等。

**📊 数据集**

使用的数据集：19 类德国烘焙产品共 763 张部署图像、315 张单类图像、4945 帧训练视频、1186 帧测试视频，以及 90 张不同视角的 A_test 图像。

**📈 对比分析**

通过在 D_test（全监督基线 mAP 0.98）与 V_test（视角变化下更具挑战性）进行比较，弱监督 + 伪标签模型在 V_test 上提升了 19.3%，最终模型在多角度测试中保持 mAP 超过 0.9，优于基线。

**⚠️ 局限性**

局限性包括：对极端视角和强遮挡下的检测仍弱；模型对名称与外观不匹配的烘焙品辨识率低；仍需在初始阶段手工标注少量图像以生成伪标签；适用性主要集中在单一工艺流程与特定面包种类。

---

## 424. RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation

**arXiv ID:** 2602.09973 | [PDF](https://arxiv.org/pdf/2602.09973v1)

**作者:** Hao Li `[一作]`, Jiangmiao Pang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RoboInter Manipulation Suite，包含数据、VQA 基准和 VLA 框架，用于推动机器人视觉-语言-动作模型的研究。

**💡 创新点**

创新点在于：①构建了规模达 230k 条、每帧密集的多类型中间表示（子任务、原语、目标框、抓取框、姿态、轨迹等）数据集；②将 VQA 与 plan‑then‑execute 结合，设计可调的 F‑CoT 作为桥梁；③提供轻量化的半自动标注工具 RoboInter‑Tool；④系统性地评估中间表示对 VLA 性能的影响。

**🔧 技术方法**

采用的技术包括：①基于 Qwen‑VL / LLaVA 的视觉‑语言模型作为 Planner；②Diffusion Transformer 作为 Executor 的动作头；③RoboInter‑Tool 结合 SAM2 与 ChatGPT 实现半自动标注；④多视角观测（第三人称+腕部视角）和 F‑CoT 进行中间表示融合；⑤多任务训练（VQA + 行为监督）。

**📊 数据集**

使用了自建的 RoboInter‑Data（230k 片段，来自 Droid、OXE、RH20T）以及基于该数据的 RoboInter‑VQA（约 1M 条问答）。此外，还对比了 Where2Place、RoboRefIt、RoboVQA 等公开基准。

**📈 对比分析**

方法上与 Qwen2.5‑VL‑3B/7B、LLaVA‑OneVision‑7B 以及 OpenVLA、Pi‑0、VLA‑OS 等基线对比。结果显示，RoboInter‑Planner 在空间与时间 VQA 上提升 40–80%，Executor 的 open‑loop mOLS 从 0.3086 提升至 0.3861，闭环成功率在 ID 任务上由 65% 提升至 77%，OOB 任务上由 38% 提升至 58%。

**⚠️ 局限性**

局限性包括：①仍需人工参与验证，标注成本高；②主要聚焦室内表面交互，缺少多模态感知如本体感；③部分中间表示依赖模型推理，可能受分布偏移影响；④实验多在仿真或实验台上进行，跨平台泛化尚待验证。

---

## 425. ViMultiChoice: Toward a Method That Gives Explanation for Multiple-Choice Reading Comprehension in Vietnamese

**arXiv ID:** 2602.09961 | [PDF](https://arxiv.org/pdf/2602.09961v1)

**作者:** Trung Tien Cao `[一作]` (Vietnam National University), Ngan Luu-Thuy Nguyen `[通讯]` (Vietnam National University)

**通讯引用:** 1374 | [OpenAlex ID](https://openalex.org/A5033137339)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 ViMultiChoice 方法，结合多选阅读理解与解释生成，针对越南语特性设计；

**💡 创新点**

创新点在于引入 ViWordFormer 模块捕获越南词组结构，并通过多任务学习提升答案选取与解释质量；

**🔧 技术方法**

采用预训练越南语言模型、Transformer 自注意力与 ViWordFormer、OCN 基础结构以及多任务损失；

**📊 数据集**

使用新构建的 ViRCSoSciD 数据集（12,819 题）以及 ViMMRC 2.0 基准；

**📈 对比分析**

与 LLM 与传统 MCRC 基线对比，ViMultiChoice 在选项准确率与 F1‑macro 上均取得 SOTA，解释生成 BLEU‑4、ROUGE‑L 也大幅提升；

**⚠️ 局限性**

局限包括对更大规模、跨领域文本的适应性不足，以及解释生成仍受限于训练数据的质量与多样性。

---

## 426. Trustworthy Agentic AI Requires Deterministic Architectural Boundaries

**arXiv ID:** 2602.09947 | [PDF](https://arxiv.org/pdf/2602.09947v1)

**作者:** Manish Bhattarai `[一作]` (Los Alamos National Laboratory), Minh Vu `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 534 | [OpenAlex ID](https://openalex.org/A5108818262)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Trinity Defense Architecture，将 LLM 视为不可信组件，通过动作治理、信息流控制和权限分离三大模块实现命令‑数据分离并保障授权安全；

**💡 创新点**

证明仅靠训练或对齐无法提供确定性授权保证，首次将有限动作演算、强制标签化和参考监视器等架构手段整合到 Agentic AI 体系中；

**🔧 技术方法**

采用有限动作算子（FAC）与判定型授权函数、强制访问标签化（IFC）、参考监视器、日志审计以及 Planner‑Worker 权限隔离等技术；

**📊 数据集**

本文以理论与模拟攻击为主，并未使用公开数据集；评估框架基于自定义的间接注入、内存泄漏和多模态绕过基准；

**📈 对比分析**

对比基线（无架构约束）时，Trinity 在所有测试中实现零授权违规、<5% 的误拦截率，平均授权延迟 <50 ms，标签传播 <25 ms；

**⚠️ 局限性**

限制：门控设计受不可判定性影响，需人工制定保守策略，可能阻断合法动作；架构依赖于小型可信执行基，且在面对未知攻击时仍需补丁迭代。

---

## 427. Efficient Learning of Sparse Representations from Interactions

**arXiv ID:** 2602.09935 | [PDF](https://arxiv.org/pdf/2602.09935v1)

**作者:** Vojtěch Vančura `[一作]` (Recombee), Ladislav Peška `[通讯]` (Charles University)

**通讯引用:** 880 | [OpenAlex ID](https://openalex.org/A5008815142)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了在推荐系统检索阶段训练高维稀疏嵌入的模型 Compressed ELSA，取代传统密集嵌入，实现高效且可扩展的表示学习。

**💡 创新点**

创新点在于：①在训练过程中采用逐步剪枝（top‑k稀疏化）策略，避免事后压缩导致的性能损失；②得到的稀疏高维表示可直接形成可解释的倒排索引结构，支持基于段落的推荐与解释。

**🔧 技术方法**

技术方法包括：基于 ELISA 线性自编码器的行级 top‑k 稀疏化、渐进剪枝计划、CSC/CSR 稀疏矩阵运算加速、以及利用聚类语义描述对段落进行可解释标签。

**📊 数据集**

使用的公开推荐数据集为 Goodbooks‑10k、MovieLens‑20M 和 Netflix Prize。

**📈 对比分析**

与 EASE、密集 ELSA、低维 ELSA、Pruned EASE 以及 ELSA+SAE 等基线在 nDCG@100 上进行对比，Compressed ELSA 在相同存储预算下可实现 10‑倍乃至 100‑倍的嵌入压缩，且准确率与密集模型相近或更优。

**⚠️ 局限性**

局限性包括：段落语义标签依赖外部语言模型，解释性评估仅在 Goodbooks‑10k 上进行，且对非线性（如 transformer）架构的泛化性尚未验证。

---

## 428. Unbalanced optimal transport for robust longitudinal lesion evolution with registration-aware and appearance-guided priors

**arXiv ID:** 2602.09933 | [PDF](https://arxiv.org/pdf/2602.09933v1)

**作者:** Melika Qahqaie `[一作]` (FAU Erlangen-Nürnberg), Veronika A. Zimmer `[通讯]` (Siemens Healthineers)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种基于不平衡最优传输的注册感知匹配器，用于纵向CT扫描中肿瘤病灶的对应与演化追踪。

**💡 创新点**

创新点在于首次将不平衡最优传输应用于肿瘤对应，融合了注册置信度、局部纹理一致性和患者肿瘤负荷自适应先验，天然支持病灶出现、消失、合并与分裂。

**🔧 技术方法**

技术方法包括：不平衡最优传输（UOT）框架、基于Jacobian确定式的注册置信度加权、零均值归一化交叉相关（ZNCC）纹理一致性、以及患者肿瘤负荷变化驱动的非对称松弛参数。

**📊 数据集**

实验使用30个合成病例和149例真实肺部CT病例（来自AutoPET IV挑战v02），在开发集100例、测试集49例上评估。

**📈 对比分析**

与两种仅基于距离的双边匹配基线相比，UOT在边缘检测、病灶状态分类和拓扑一致性上均取得更高的召回率和F1分数，尤其在病灶合并/分裂识别上显著优于基线。

**⚠️ 局限性**

局限性包括对配准精度高度依赖、对自动检测病灶的适用性尚未验证，以及在真实数据中合并/分裂事件相对稀少，导致相关性能提升有限。

---

## 429. LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations

**arXiv ID:** 2602.09924 | [PDF](https://arxiv.org/pdf/2602.09924v1)

**作者:** William Lugoloobi `[一作]` (Oxford Internet Institute), Chris Russell `[通讯]` (Oxford Internet Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在LLM的生成前激活上训练线性探针，预测模型在数学与编程任务中的成功率与难度，并利用该预测实现模型路由，以降低推理成本；

**💡 创新点**

首次揭示模型内部编码的“模型难度”与人类主观难度并不一致，并证明探针在推理时长增加时仍能提供有效的难度估计；

**🔧 技术方法**

线性探针、TF‑IDF特征、输入长度比较、Maj@K与greedy解码、成本估计（基于Fireworks AI定价）以及简易阈值/效用路由策略；

**📊 数据集**

E2H‑AMC（含人类IRT难度与模型性能）、MATH、GSM8K、AIME、LiveCodeBench；

**📈 对比分析**

与文本特征基线相比，探针在预测人类难度时Spearman ρ≥0.83，在预测模型成功率时ρ≈0.40–0.64，AUROC>0.7；路由后可在MATH上实现约17%–70%成本下降且保持或超过最强单模型准确率；

**⚠️ 局限性**

探针在深度推理时线性可分性下降；仅使用单一预生成位置的线性探针；未探索非线性或多位置探针；未研究跨域/跨数据集的探针迁移；路由策略过于简单，缺乏自适应学习。

---

## 430. The Need for Standardized Evidence Sampling in CMMC Assessments: A Survey-Based Analysis of Assessor Practices

**arXiv ID:** 2602.09905 | [PDF](https://arxiv.org/pdf/2602.09905v1)

**作者:** Logan Therrien `[一作]` (Dakota State University), John Hastings `[通讯]` (Dakota State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对17名CMMC认证评估者进行问卷调查，探索其证据抽样做法及其差异性

**💡 创新点**

首次系统性、实证性分析CMMC评估中的证据抽样缺失与变异，并提出标准化、风险导向抽样框架的必要性

**🔧 技术方法**

采用混合方法问卷（结构化题+情景题）与主题编码分析

**📊 数据集**

匿名收集的17份可用调查回应（包含结构化数据与开放式回答）

**📈 对比分析**

通过对情景题结果的定量分布与定性主题进行对比，显示抽样幅度在相同情境下存在巨大差异，表明现行做法缺乏一致性；未进行性能量化评估

**⚠️ 局限性**

样本量小、受访者自报数据、缺乏观察验证、主题编码单一研究者完成，导致结果可能不具普适性

---

## 431. Routing, Cascades, and User Choice for LLMs

**arXiv ID:** 2602.09902 | [PDF](https://arxiv.org/pdf/2602.09902v1)

**作者:** Rafid Mahmood `[一作]` `[通讯]` (University of Ottawa and NVIDIA), Rafid Mahmood (University of Ottawa and NVIDIA)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文通过Stackelberg博弈模型研究LLM提供商与单一用户之间的路由与回流决策，揭示用户对延迟与质量的耐心如何影响放弃行为；

**💡 创新点**

创新点在于将用户行为纳入路由决策框架，首次给出用户最优回避策略、提供商最优路由阈值，并量化两者的误配（misalignment）及极端情况下的滞后调节（throttling）效应；

**🔧 技术方法**

使用马尔可夫链分析、闭式期望计算、阈值分段最优化以及Stackelberg均衡推导等理论工具；

**📊 数据集**

未使用公开数据集，而是基于假设的成功概率、延迟、成本与用户价值等参数进行理论推导；

**📈 对比分析**

由于研究为理论分析，未与实测系统做直接对比；作者通过数学证明阐明在不同参数区间内最优策略为单一路由或简单阈值决策，并讨论与传统仅考虑质量-成本权衡的算法差异；

**⚠️ 局限性**

局限性包括只考虑两种模型、假设成功概率独立且用户可知路由策略、未考虑支付定价、未给出实验验证，且模型对真实用户行为的适配性需进一步探索。

---

## 432. Spinel: A Post-Quantum Signature Scheme Based on SLn(Fp) Hashing

**arXiv ID:** 2602.09882 | [PDF](https://arxiv.org/pdf/2602.09882v1)

**作者:** Asmaa Cherkaoui `[一作]` (Hassan II University), Siamak F. Shahandashti `[通讯]` (University of York)

**通讯引用:** 1661 | [OpenAlex ID](https://openalex.org/A5056489873)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种后量子数字签名方案SL_4-Based Post-Quantum Integrity from Non-Backtracking Expanders over Linear Groups（S4-Post-Quantum），在Sphincs+框架下替换了传统哈希原语，使用基于SL_4(F_p)的Tillich–Zémor Cayley-walk哈希函数；

**💡 创新点**

创新点在于将基于SL_n(F_p)的代数哈希函数成功集成进Sphincs+，在保持其正式安全证明的同时，提供了新的后量子安全假设（SL_4(F_p)随机游走难题），并给出实验验证与参数选择指南；

**🔧 技术方法**

主要技术包括：Sphincs+的Winternitz OTS、Merkle树、Forest of Random Subsets层、分层雏树（hypertree）结构；以及SL_4(F_p)的矩阵群随机游走哈希、非回溯随机 walk、固定宽度32位整数编码和NIST STS随机性测试；

**📊 数据集**

使用的“数据集”是生成的随机字节流（100条长度1,000,000位），用于NIST STS测试；另外对不同参数集的签名长度、签名/验证时间进行了基准测评，硬件为AMD EPYC 7643；

**📈 对比分析**

比较方法：将S4-Post-Quantum与原始Sphincs+（多种哈希原语）在同一硬件上进行键生成、签名、验证的周期计数和签名尺寸对比；实验结果显示签名速度约为原Sphincs+的10–20%（取决于参数），验证速度相差不大；签名尺寸略大（~60 KB vs 50–55 KB），但整体性能可接受；

**⚠️ 局限性**

局限性包括：签名生成的计算开销显著高于原Sphincs+（约10^12周期），签名尺寸相对较大；安全性仍基于SL_4(F_p)随机游走的难度，尚未有针对量子攻击的正式证明；以及对参数选择的依赖性强，需在实际应用中根据签名预算进行细致调优。

---

## 433. Maximizing Diversity in (near-)Median String Selection

**arXiv ID:** 2602.10050 | [PDF](https://arxiv.org/pdf/2602.10050v1)

**作者:** Diptarka Chakraborty `[一作]` (National University of Singapore), Aravinda Kanchana Ruwanpathirana `[通讯]` (Nanyang Technological University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在汉明距离下求解多样化中位数集合的问题，提出了最大直径、最大和分散度以及最大最小分散度的算法。

**💡 创新点**

首次系统地探讨多样化中位数，给出了最大直径的精确算法、最大和分散度的 PTAS，以及最大最小分散度的双标准近似算法，填补了该领域的空白。

**🔧 技术方法**

利用汉明中位数空间的结构特征、误码纠错码理论、动态规划、贪心、随机化、Plotkin 约束、负型度量与基于 LP 的相依舍入技术等多种算法工具。

**📊 数据集**

文中未给出具体实验数据集，主要以理论分析与算法复杂度说明为主。

**📈 对比分析**

通过理论证明和复杂度分析与现有最优/近似算法对比，展示了在多样化度量上实现更高多样性或更好的近似比，尤其在最大和分散度上达到 PTAS，最大最小分散度提供了 1/2 及 1/2‑δ 近似。

**⚠️ 局限性**

存在对直径 D* 处于 ω(1) 与 O(log k) 区间的最小分散度问题缺乏多项式时间解法；算法多采用双标准近似，未完全单独优化多样化度量；目前仅针对汉明度量，难以直接推广至其他度量空间。

---

## 434. Long Chain-of-Thought Compression via Fine-Grained Group Policy Optimization

**arXiv ID:** 2602.10048 | [PDF](https://arxiv.org/pdf/2602.10048v1)

**作者:** Xinchen Han `[一作]` (Institut Polytechnique de Paris), Lu Yin `[通讯]` (University of Surrey)

**通讯引用:** 12124 | [OpenAlex ID](https://openalex.org/A5045336307)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Fine‑grained Group policy Optimization (FGO)，一种利用 RL 对 LLM 生成的长 Chain‑of‑Thought（CoT）进行细粒度分组与奖励塑造，从而压缩 CoT 长度并保持甚至提升推理性能的方法。

**💡 创新点**

创新点在于：① 将生成结果按正确/错误子组划分，针对每个子组构造长度与熵权重的奖励；② 通过细粒度奖励实现高效 CoT 压缩；③ 同时解决 GRPO 的数据利用低效和熵坍塌两大缺陷。

**🔧 技术方法**

核心技术包括：强化学习（GRPO 基础框架）、Softmax 权重分配、长度与熵联合奖励塑造、子组优势函数计算。

**📊 数据集**

使用 MATH500、AIME24、AMC23、Minerva 四大数学/推理基准；训练集为 MATH‑LightEval 前 3,200 条样本。

**📈 对比分析**

与 Vanilla、GRPO 以及 TLDR 进行对比；实验表明 FGO 在 CoT 长度显著缩短的同时，准确率提升或保持不变，ACT（每百 token 的准确率贡献）更高，数据利用率达到 100%，并成功抑制熵坍塌。

**⚠️ 局限性**

主要局限在于对超参数（如 α、β）的敏感性，需要手动调优；目前仅在数学/推理任务上验证，尚未在更广泛的多模态或非结构化任务中评估。

---

## 435. Fake-HR1: Rethinking reasoning of vision language model for synthetic image detection

**arXiv ID:** 2602.10042 | [PDF](https://arxiv.org/pdf/2602.10042v1)

**作者:** Changjiang Jiang `[一作]`, Wei Lu `[通讯]` (Wuhan University)

**通讯引用:** 13579 | [OpenAlex ID](https://openalex.org/A5035830977)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并训练了一个能够自适应决定是否生成推理链的多模态大模型 Fake-HR1，用于检测 AI 生成图像。

**💡 创新点**

引入双阶段训练框架（Hybrid Fine‑Tuning + Hybrid‑Reasoning Grouped Policy Optimization），并在奖励中加入“是否需要推理”的目标，解决了传统模型在需要推理时出现的“推理退化”问题。

**🔧 技术方法**

使用 Qwen2.5‑VL‑7B 作为基础模型，结合自监督微调与强化学习（GRPO）以及自定义奖励函数。

**📊 数据集**

使用 GenImage 生成非推理样本、FakeClue 生成推理样本，并通过拒绝采样构造 RL 训练集。

**📈 对比分析**

在 FakeClue 测试集上对比 Qwen2.5‑VL‑7B、InternVL3‑8B、GPT‑4o，Fake‑HR1 在推理与非推理两种模式下均取得 94.4% 与 91.7% 的准确率，且平均输出长度显著缩短。

**⚠️ 局限性**

受限于固定的 seed 问题集和主要关注图像检测，模型在更复杂的文本/视频或未见的生成器上可能表现下降，需要进一步扩展任务多样性与数据来源。

---

## 436. Resilient Topology-Aware Coordination for Dynamic 3D UAV Networks under Node Failure

**arXiv ID:** 2602.10029 | [PDF](https://arxiv.org/pdf/2602.10029v1)

**作者:** Chuan-Chi Lai `[一作]` (National Chung Cheng University), Chuan-Chi Lai `[通讯]` (National Chung Cheng University)

**通讯引用:** 115 | [OpenAlex ID](https://openalex.org/A5082920277)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于拓扑感知图注意力的多智能体强化学习框架（TAG‑MAPPO），用于3D空地集成网络中无人机（UAV）自动重构、持续覆盖和自愈。

**💡 创新点**

创新点在于引入双路径融合的拓扑感知图注意力（TA‑GAT）以及随机观测打乱（ROS），实现对动态节点数的置换不变性，使得在节点失效后系统能够快速恢复覆盖并提升公平性。

**🔧 技术方法**

采用图注意力网络（GAT）+多智能体近端策略优化（MAPPO）+随机观测打乱+残差自我状态融合+Huber损失+PPO‑Clip等技术，构建中心化训练、去中心化执行的学习体系。

**📊 数据集**

使用基于仿真生成的三种场景（拥挤城市、郊区、乡村）数据集，共140名地面用户，仿真参数涵盖2 GHz载波、20 MHz带宽、不同LOS/NLOS路径损耗模型；未使用公开真实数据集。

**📈 对比分析**

与G‑MAPPO（MLP）、QMIX、几何K‑means基线进行对比，TAG‑MAPPO在覆盖率、能效、握手次数上均优越；覆盖率在节点失效后15步内恢复90%以上，手动次数下降约50%，能效提升至约5 Mbps/W，公平性（Jain指数）甚至在失效后超过原始四UAV配置。

**⚠️ 局限性**

局限性包括仅在4架UAV规模的仿真验证、缺乏真实部署实验、需要集中式训练（不易在线迁移）、对更大规模网络的可扩展性和训练成本尚待评估。

---

## 437. MEVER: Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval

**arXiv ID:** 2602.10023 | [PDF](https://arxiv.org/pdf/2602.10023v1)

**作者:** Delvin Ce Zhang `[一作]` (University of Sheffield), Dongwon Lee `[通讯]` (Pennsylvania State University)

**通讯引用:** 9348 | [OpenAlex ID](https://openalex.org/A5100405086)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多模态可解释主张验证框架MEVER，结合证据检索、联合文本-图像推理与解释生成；

**💡 创新点**

创新点在于：①使用两层多模态图网络实现图像-文本交叉检索；②引入标记级与证据级双层融合实现更细粒度的跨模态推理；③设计Fusion-in-Decoder与一致性正则化生成与判断一致的解释；

**🔧 技术方法**

核心技术包括Transformer与ViT混合编码器、图神经网络（GNN）跨模态注意力、联合对比检索损失、对抗式解释生成与KL一致性约束；

**📊 数据集**

使用新构建的AIChartClaim科学数据集（1200条主张、300张图表、注释解释），并在ChartCheck、Mocheg、MR2等公开数据集上评测；

**📈 对比分析**

与多模态与文本检索/验证基线（RAV、JustiLM、MochegModel等）对比，MEVER在检索MAP、验证Macro‑F1和解释ROUGE‑L等指标上分别提升约1–3个百分点，显示显著优势；

**⚠️ 局限性**

局限包括：依赖图像存在的多模态设定，缺少图像时性能下降；AIChartClaim规模受制于人工审核，难以扩展到更大规模或其他学科；

---

## 438. METTLE: Efficient Streaming Erasure Code with Peeling Decodability

**arXiv ID:** 2602.10020 | [PDF](https://arxiv.org/pdf/2602.10020v1)

**作者:** Qianru Yu `[一作]` (Georgia Tech), Jun Xu `[通讯]` (Georgia Tech)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了METTLE，一种同时具备高编码效率、低编码复杂度和低延迟的流式纠删码。

**💡 创新点**

创新点在于将Walzer的空间耦合哈希结构改为时间耦合，并引入多边缘类型（MET）与触摸无前缘（TLE）策略，实现纯剥离解码。

**🔧 技术方法**

利用哈希生成的图、空间耦合、分布式MDGM、Peeling解码以及数值密度演化优化。

**📊 数据集**

在各种模型通道（BEC、Gilbert‑Elliott）以及实际VoIP、WiMAX、视频会议等仿真数据上评估。

**📈 对比分析**

与RaptorQ和LT通过编码效率、解码延迟、解码速度等指标对比，METTLE在相同延迟下编码效率略优、解码速度提升近两倍至十倍，且对突发损失更鲁棒。

**⚠️ 局限性**

局限在于当丢包率极高时存在误差底线，且在极低延迟或极大块码时需进一步调优窗口大小。

---

## 439. ADORA: Training Reasoning Models with Dynamic Advantage Estimation on Reinforcement Learning

**arXiv ID:** 2602.10019 | [PDF](https://arxiv.org/pdf/2602.10019v1)

**作者:** Qingnan Ren `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 13717 | [OpenAlex ID](https://openalex.org/A5102740754)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ADORA 框架，通过在线 Rollout 自适应地对样本进行优势估计加权，从而改进强化学习中的策略优化；

**💡 创新点**

创新点在于将样本分为临时有利样本（TAS）与临时不利样本（TDS），根据 Rollout 动态地调整优势权重，使模型专注学习信息量最大的经验；

**🔧 技术方法**

技术包括基于 GRPO 的策略梯度算法、在线 Rollout 统计、长度优势与难度优势判别策略、以及对 LLM 与 VLM 的任务专属加权方案；

**📊 数据集**

使用 Geometry3K（2000 条样本）评估 VLM 几何推理，使用 MATH500（12000 条样本）评估 LLM 数学推理，并在 MathVista、MathVerse、DynaMath、GSM8K、MATH500、AMC23、CollegeMath、OlympiadBench、AIME24 等数据集上进行测试；

**📈 对比分析**

与 vanilla GRPO 对比，ADORA 在大多数模型和数据规模下均取得显著提升（平均提升 0.83%–3.50%，在 Qwen‑7B 上 3.4 分点提升，VLM 在 MathVista 上匹敌 Vision‑R1‑7B 甚至超越部分闭源模型），并在 2000 条样本的低数据量条件下实现了 SOTA 级别性能；

**⚠️ 局限性**

局限性包括：需要针对不同任务重新设计样本区分策略；依赖 Rollout 质量，若基准模型产生低质量轨迹则 TAS/TDS 判别可能失效；实验范围仅覆盖数学与几何推理，对常识推理、代理任务等其他场景未验证；对更高级 RL 算法的适配性尚待进一步研究。

---

## 440. Discovering High Level Patterns from Simulation Traces

**arXiv ID:** 2602.10009 | [PDF](https://arxiv.org/pdf/2602.10009v1)

**作者:** Sean Memery `[一作]` (University of Edinburgh), Kartic Subr `[通讯]` (University of Edinburgh)

**通讯引用:** 931 | [OpenAlex ID](https://openalex.org/A5086009008)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于自然语言引导的模式发现方法，利用进化程序合成从低层仿真轨迹中自动识别高层物理事件，生成Annotated Simulation Traces（AST），并在此基础上实现自然语言问题回答、摘要生成和奖励程序合成等下游任务。

**💡 创新点**

创新点包括：①仅通过用户提供的自然语言标签即可驱动进化搜索学习模式检测程序；②将低级数值轨迹映射为可解释的高层事件注释；③利用这些注释让大型语言模型（LLM）在物理推理、规划和奖励函数生成上显著提升。

**🔧 技术方法**

技术手段主要有：进化程序搜索（FunSearch）、LLM辅助程序修复与评估、模式检测器与DSL奖励程序的代码生成、基于交叉熵与相关性的模式筛选指标，以及对生成奖励程序的密集反馈优化。

**📊 数据集**

实验数据集为两类：1）改编后的Phyre 2D刚体仿真任务（25个模板，共2500个场景），2）从Phyre仿真生成的问答基准（10个问题模板×100场景）。

**📈 对比分析**

与仅使用视频或原始轨迹对比，使用AST能够使LLM在Phyre任务中成功率提升约10%，在问答中准确率提升约15%；生成的奖励程序在模拟退火优化中，比稀疏二进制奖励快50%收敛，且在训练价值网络时的平均奖励提升约30%。

**⚠️ 局限性**

局限性包括：需要人工提供初始自然语言模式标签，方法目前仅在二维刚体环境验证，且随着模式库规模扩大可能导致注释过长、上下文溢出问题；对更复杂多体或三维环境的泛化尚待进一步研究。

---

## 441. Answer First, Reason Later: Aligning Search Relevance via Mode-Balanced Reinforcement Learning

**arXiv ID:** 2602.10006 | [PDF](https://arxiv.org/pdf/2602.10006v1)

**作者:** Shijie Zhang `[一作]` (Qwen Applications Business Group, Alibaba Group), Kevin Zhang `[通讯]` (Peking University)

**通讯引用:** 23112 | [OpenAlex ID](https://openalex.org/A5065063835)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Answer-First, Reason Later (AFRL)框架，通过先输出相关性得分再生成结构化推理链，解决搜索相关性评估中的低延迟与可解释性冲突。

**💡 创新点**

创新点包括将决策与推理解耦实现毫秒级响应、引入模式平衡优化结合前向与反向KL避免模式崩塌、PIAR自动指令演化、分阶段课程学习以及大模型到小模型的知识蒸馏。

**🔧 技术方法**

使用了大规模语言模型的监督微调、强化学习（GRPO/Stepwise GRPO）、模式平衡混合目标、PIAR指令自进化、结构化数据合成、规则化奖励、分步优势加权，以及知识蒸馏技术。

**📊 数据集**

训练与评估数据包括150k专家标注的AFRL轨迹、14,625条真实日志的INDUSTRY基准、3,881条长尾冷启动查询的LONGTAIL基准以及1.5M通用相关性样本用于预训练。

**📈 对比分析**

与72B Qwen2.5-72B-Instruct基线对比，32B AFRL模型在INDUSTRY和LONGTAIL上分别实现73.07% 5-ACC、85.57% 2-ACC、82.40% Pair-ACC，超过72B模型；通过知识蒸馏，0.6B学生模型在5-ACC上达到70.9%，接近教师性能。

**⚠️ 局限性**

局限性包括训练成本仍高、模型对规则奖励的依赖可能导致泛化受限、未在多模态检索或动态计算分配上验证、以及在极端长尾或多义查询场景下的鲁棒性待进一步探究。

---

## 442. Acoustic Drone Package Delivery Detection

**arXiv ID:** 2602.09991 | [PDF](https://arxiv.org/pdf/2602.09991v1)

**作者:** François Marcoux `[一作]` (Université de Sherbrooke), François Grondin `[通讯]` (Université de Sherbrooke)

**通讯引用:** 4461 | [OpenAlex ID](https://openalex.org/A5045901453)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并验证了一套基于地面麦克风阵列的无人机包裹投递事件检测系统。

**💡 创新点**

首次通过声学信号估计旋翼转速并利用变化点检测识别投递时刻，结合CRNN实现BPF与无人机活动预测。

**🔧 技术方法**

使用CNN+Bi‑GRU多任务CRNN进行BPF与无人机活动预测，输入为mel spectrogram与功率共振谱；投递检测采用统计距离（Chi‑squared、Jensen–Shannon、交集、均值差）驱动的变化点检测。

**📊 数据集**

收集28次实地飞行数据，包含机载与阵列录音、遥测、PWM、BPF校准等，生成约21,675秒训练集、3,712秒验证集、8,990秒测试集，并混入多种背景噪声。

**📈 对比分析**

与传统SVM、CNN等模型对比，本方法在100m距离内BPF MAE低至16Hz，飞行检测准确率97%，投递检测TPR 96%且FPR 8%。

**⚠️ 局限性**

仅使用单一机型训练导致域适应性差；对轻量投递敏感度不足；未考虑无人机位置、速度导致的多普勒或风扰动。

---

## 443. Some conditions implying if P=NP then P=PSPACE

**arXiv ID:** 2602.10073 | [PDF](https://arxiv.org/pdf/2602.10073v1)

**作者:** Ismael Rodriguez `[一作]` (Universidad Complutense de Madrid), Ismael Rodriguez `[通讯]` (Universidad Complutense de Madrid)

**通讯引用:** 1645 | [OpenAlex ID](https://openalex.org/A5080808845)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文探讨了在某些条件下，如果P=NP成立，则P=PSPACE也成立的情况。作者识别了一些条件X，证明了这些条件可以推导出P=NP ⇒ P=PSPACE，并讨论了每个条件的可行性。

**💡 创新点**

创新点在于提出了一些充分条件X，若这些条件成立，则可以得出P=NP ⇒ P=PSPACE的结论。这为复杂性理论提供了新的视角。

**🔧 技术方法**

使用了图灵机的定义和性质，构造了计算设备，并通过逻辑表达式和布尔函数的性质进行推导。

**📊 数据集**

没有使用特定的数据集，而是通过理论推导和逻辑构造来支持论点。

**📈 对比分析**

通过与现有复杂性理论的比较，作者展示了这些条件的潜在影响，但没有提供具体的性能比较数据。

**⚠️ 局限性**

限制在于没有给出充分条件A的证明，且对条件A的可行性讨论较为模糊，缺乏实证支持。

---

## 444. Chain of Mindset: Reasoning with Adaptive Cognitive Modes

**arXiv ID:** 2602.10063 | [PDF](https://arxiv.org/pdf/2602.10063v1)

**作者:** Tianyi Jiang `[一作]` (Beijing Jiaotong University), Ronghao Chen `[通讯]` (QuantaAlpha)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Chain of Mindset（CoM）框架，能够在单一推理过程中动态切换四种功能异质的认知模式（空间、聚合、发散、算法），实现训练无关的代理式推理；

**💡 创新点**

核心创新是实现步级自适应多模式切换，利用Meta‑Agent与双向Context Gate高效隔离与过滤信息，从而在同一推理过程内协同多种认知范式；

**🔧 技术方法**

采用LLM代理决策、Context Gate双向语义过滤、四个专门的心态模块（Spatial、Convergent、Divergent、Algorithmic），配合图像生成（Nano‑Banana‑Pro）与沙盒Python执行等技术；

**📊 数据集**

在六大基准上评估：AIME 2025、Real‑Fermi、LiveCodeBench、GPQA‑Diamond、MathVision‑Mini 与 MAZE；

**📈 对比分析**

与 Direct I/O、Zero‑shot CoT、Tree of Thoughts、Chain of Code、ReAct、MRP、Meta‑Reasoner 等方法对比；在 Qwen3‑VL‑32B‑Instruct 上整体准确率 63.28%，比最强基线高 4.96%；在 Gemini‑2.0‑Flash 上 52.41%，比最强基线高 4.72%；同时保持中等 token 消耗，位于准确率‑效率 Pareto 前沿；

**⚠️ 局限性**

局限性包括需手工设计四种心态、缺乏针对任务的自适应心态子集选择、Context Gate 与 Meta‑Agent 设计复杂、对极大规模或实时推理的可扩展性尚未验证、部分任务对多路径探索可能产生冗余。

---

## 445. Vendi Novelty Scores for Out-of-Distribution Detection

**arXiv ID:** 2602.10062 | [PDF](https://arxiv.org/pdf/2602.10062v1)

**作者:** Amey P. Pasarkar `[一作]` (Princeton University), Adji Bousso Dieng `[通讯]` (Princeton University)

**通讯引用:** 983 | [OpenAlex ID](https://openalex.org/A5063448291)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的后置式OOD检测方法VNS，利用Vendi Score从多样性角度衡量测试样本对训练集的“新颖度”。

**💡 创新点**

创新点在于：① 将Vendi Score用于测量类别级的多样性提升，② 用概率加权的top‑K聚合与全局多样性校正相结合；这些设计既不需要概率模型假设，也避免了高复杂度的KNN搜索。

**🔧 技术方法**

技术核心包括：稀疏向量化的余弦核Vendi Score、秩‑1近似更新、概率加权聚合、全局最大特征值的一阶近似。

**📊 数据集**

实验使用CIFAR‑10、CIFAR‑100、ImageNet‑1K三大图像分类数据集，并在ResNet、ViT、Swin‑T等多种网络架构上进行评估。

**📈 对比分析**

与10种主流后置OOD检测基线（MSP、KNN、NCI等）对比，VNS在FPR@95和AUROC上均取得最优或近最优表现，平均相对第二名降低13%的FPR@95，且在仅使用训练集1%数据时性能仍保持稳定。

**⚠️ 局限性**

局限性包括：① 需访问训练数据且对训练集规模敏感；② 采用余弦核，可能在其他相似度度量下效果不佳；③ 引入3个可调超参数，需要额外的验证集进行调优。

---

## 446. Conformal Prediction Sets for Instance Segmentation

**arXiv ID:** 2602.10045 | [PDF](https://arxiv.org/pdf/2602.10045v1)

**作者:** Kerri Lu `[一作]` (Massachusetts Institute of Technology), Sherrie Wang `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1619 | [OpenAlex ID](https://openalex.org/A5088966490)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种基于conformal prediction的实例分割置信集合方法，能给定像素查询生成多种预测掩码并提供理论覆盖保证。

**💡 创新点**

创新点在于：①使用可调参数产生多样化掩码；②通过集合覆盖构造置信集；③引入去重与重新校准以保持覆盖；④提供渐进与有限样本理论保证。

**🔧 技术方法**

主要技术包括：conformal prediction、集合覆盖（贪心+暴力搜索）、去重的支配集算法、IoU评估、重采样校准。

**📊 数据集**

实验数据集：农业字段分割（Fields of The World+UNet+阈值）、细胞分割（Cellpose/Cellpose-SAM+阈值）、车辆检测（Cityscapes+Segment Anything Model+mask索引与阈值）。

**📈 对比分析**

与可行的 Learn‑Then‑Test / CRC 基线以及基于形态膨胀的方法对比，所提方法在覆盖率上提升约10–20%，且置信集大小自适应。

**⚠️ 局限性**

局限包括：覆盖阈值统一；需存在可调参数以产生多样性；集合覆盖与去重是 NP‑hard，实际需要小规模；覆盖保证为全局，不能针对单个查询自适应。

---

## 447. A Collision-Free Sway Damping Model Predictive Controller for Safe and Reactive Forestry Crane Navigation

**arXiv ID:** 2602.10035 | [PDF](https://arxiv.org/pdf/2602.10035v1)

**作者:** Marc-Philip Ecker `[一作]` (AIT Austrian Institute of Technology), Wolfgang Kemmetmüller `[通讯]` (Automation and Control Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个将碰撞规避与吊载摆动阻尼统一的模型预测控制器（MPC），并在真实林木起重机上进行了验证。

**💡 创新点**

创新点在于：① 将实时 LiDAR 感知的欧氏距离场（EDF）直接嵌入 MPC 优化约束，实现在线碰撞检测；② 在同一控制器中同时处理吊载摆动阻尼与碰撞规避；③ 引入时间进度变量，保证在无法避障时安全停机。

**🔧 技术方法**

技术方法包括：LiDAR 点云滤波 + OctoMap + FIESTA EDF 构建；全局规划使用 PFRC‑VP‑STO；局部 MPC 采用二阶执行器模型、主动/被动关节耦合动力学，约束包括关节极限、加速度、泵流量和 EDF 距离；成本函数由跟踪、阻尼、速度、加速度和进度四项构成。

**📊 数据集**

数据集：使用现场采集的 Livox Avia LiDAR 点云和起重机运动记录；未使用公开数据集。

**📈 对比分析**

通过实验比较：① 在无障碍环境中，MPC 能将摆动时间从 >20 s 缩短至 ≈5 s；② 在靠近障碍物时，开启碰撞成本能避免碰撞并保持安全间隙；③ 当出现不可避障碍时，MPC 能主动停机；实验表明控制性能显著优于传统跟踪或单独摆动阻尼方案。

**⚠️ 局限性**

局限性：仅适用于准静态环境；LiDAR 视野有限导致盲区风险；全局规划缺乏动态再规划能力；对日志识别与体积估计的感知仍需改进；需要进一步提升实时性能和安全性。

---

## 448. Optimal Bounds-Only Pruning for Spatial AkNN Joins

**arXiv ID:** 2602.10027 | [PDF](https://arxiv.org/pdf/2602.10027v1)

**作者:** Dominik Winecki `[一作]` (Ohio State University), Dominik Winecki `[通讯]` (Ohio State University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5092519187)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在无索引、基于分区的空间 All-K-Nearest-Neighbor (AkNN) 连接中，仅使用分区边界进行剪枝，并提出 AllPointsCloser 三边界测试来判断一个分区内所有点是否都更靠近另一个分区而非第三个。

**💡 创新点**

提出了最优且可实现的仅使用分区边界的剪枝算法 AllPointsCloser，解决了传统 Bound-to-Bound 或 MinMaxDist 方法在方向性和重叠分区下的保守性问题，并给出了严格的数学证明。

**🔧 技术方法**

结合了欧氏距离的点-边界与边界-边界距离函数、凸集与 Jensen 不等式的凸性证明，以及基于维度独立的 O(R) 复杂度优化实现；实现使用 Rust，形式化验证使用 Lean4。

**📊 数据集**

实验主要使用合成空间分区数据，在 2–4 维常见整数和浮点数类型上进行时间测评，未给出具体真实数据集。

**📈 对比分析**

与传统的 Bound-to-Bound 与 MinMaxDist 剪枝方法比较，AllPointsCloser 在分区数大、方向性明显时能提前裁剪更多分区；实验显示在各种整数/浮点类型和维度下平均执行时间仅 4–10 ns，显著快于 O(R·2^R) 的朴素实现。

**⚠️ 局限性**

仅适用于欧氏/二范数距离、Axis-aligned Bounding Boxes；在 k 超过分区大小、存在非最小边界或使用额外谓词时需构造 DAG 进一步处理，实际应用中仍需结合后续点级 kNN 算法。

---

## 449. The Architecture of Illusion: Network Opacity and Strategic Escalation

**arXiv ID:** 2602.10053 | [PDF](https://arxiv.org/pdf/2602.10053v1)

**作者:** Raman Ebrahimi `[一作]` (University of California), Massimo Franceschetti `[通讯]` (University of California)

**通讯引用:** 11215 | [OpenAlex ID](https://openalex.org/A5075999338)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在该研究中，作者通过在网络结构中引入可调的“局部性参数”p，将认知层级模型与社会网络相结合，构建了“Connected Minds”框架，分析了网络对玩家信念与行动的系统性影响，并进一步探讨了信息架构对集体福利的机制设计效应。

**💡 创新点**

创新点在于：①把网络可见性抽象为连续参数p，形成从传统Level‑k到Cognitive‑Hierarchy的光滑过渡；②证明即使认知深度趋于无穷，局部性偏差仍导致感知分布漂移至τ/p（Poisson‑Shift收敛定理）；③在机制设计层面提出“Escalation Principle”和“Transparency Reversal”，揭示在不同策略游戏中最优信息透明度的非直观取向。

**🔧 技术方法**

主要技术包括：博弈论中的迭代推理与策略互补分析；概率与统计学中的指数加权（tilting）与对数凹性保持；生成函数与总变差距离证明；以及信息设计与贝叶斯说服框架的延伸。

**📊 数据集**

本文并未使用实证数据，而是以理论推导和模拟实验（如美丽竞赛、技术采纳等）为主要展示手段；若要实证验证，可利用实验经济学平台或社交网络采集的层级分布数据。

**📈 对比分析**

与传统CH模型相比，作者通过数值实验表明，在低p（信息不透明）情形下，集体行动水平提升，且对策略互补游戏的福利提升更明显；而在协调游戏中，提升p可显著降低方差，改善均衡偏差；整体性能表现取决于目标函数的设定。

**⚠️ 局限性**

局限性包括：①模型采用的是抽象的p参数，缺乏对具体图拓扑（如度分布、聚类系数）的直接映射；②仅考察静态均衡，未探讨网络与学习的共进化；③在观测层面，k与p的效应可能不可观测，导致识别困难；④对极端游戏设定（如大型市场、金融危机）的适用性尚待检验。

---

## 450. A Collaborative Safety Shield for Safe and Efficient CAV Lane Changes in Congested On-Ramp Merging

**arXiv ID:** 2602.10007 | [PDF](https://arxiv.org/pdf/2602.10007v1)

**作者:** Bharathkumar Hegde `[一作]` (Trinity College Dublin), Melanie Bouroche `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种结合多智能体强化学习（MARL）与协作安全护盾（MASS）的车道变换控制器（MARL-MASS），用于拥堵的匝道合流场景。

**💡 创新点**

创新点在于：①设计了协作安全护盾MASS，利用控制壁垒函数（CBF）与交互拓扑实现多车协同安全；②提出交互拓扑构造算法捕捉车辆间依赖；③为有护盾的MARL设计了专门的奖励函数，显著提升学习稳定性和效率。

**🔧 技术方法**

技术包括：多智能体强化学习（MAPPO/Actor-Critic）、控制壁垒函数（CBF）、车辆动力学（运动学双轮车模型）、图论交互拓扑、强化学习奖励设计。

**📊 数据集**

使用自行构造的匝道合流仿真环境（基于 highway-env），在高密度交通（7-11 辆 CAV）上进行训练与评估，没有公开标准数据集。

**📈 对比分析**

与三种基线比较：无护盾 unsafe MARL、MARL-CS（无安全约束）和 MARL-HSS（集中式安全护盾）。结果显示：MARL-MASS 在保持 0.5 s 时间头距安全约束的前提下，平均车速最高（24.71 m/s）、合流率最高（83.36 %），并且安全头距最低（0.59 s）。相比 MARL-HSS，效率提升约 12.57 % 并降低最小头距。

**⚠️ 局限性**

局限性：仅在匝道合流单一场景验证，未考察多车道或交叉口等更复杂场景；奖励函数需手动调参；仿真环境与真实道路差距可能影响迁移效果。

---

## 451. ViSpeechFormer: A Phonemic Approach for Vietnamese Automatic Speech Recognition

**arXiv ID:** 2602.10003 | [PDF](https://arxiv.org/pdf/2602.10003v1)

**作者:** Khoa Anh Nguyen `[一作]` (Vietnam National University), Ngan Luu-Thuy Nguyen `[通讯]` (Vietnam National University)

**通讯引用:** 1374 | [OpenAlex ID](https://openalex.org/A5033137339)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于音素级的越南语语音识别框架ViSpeechFormer，并开发了专用的越南音素分词器ViPhonER；

**💡 创新点**

创新点在于首次将越南语的语音识别显式转化为音素级解码，利用该语言的拼写与音素高度透明的特性，显著降低词表大小与输出长度；

**🔧 技术方法**

技术包括Transformer Encoder、专门设计的三头音素解码器（首音、韵尾、声调）、前馈网络多分支输出以及对每个音素子任务的交叉熵损失；

**📊 数据集**

实验使用公开的越南语数据集ViVOS（15小时）和LSVSC（100小时）进行评测；

**📈 对比分析**

与基线的字符级Speech Transformer和多种子词级模型（Conformer、ZipFormer等）对比，ViSpeechFormer在两数据集上均实现了最低的CER/WER，并在OOV词识别率和推理速度上优于对手；

**⚠️ 局限性**

局限性包括仍需手工设计音素分词规则、对多音节语言扩展性有限以及在极端噪声环境下的鲁棒性待进一步验证。

---

## 452. Faster-GS: Analyzing and Improving Gaussian Splatting Optimization

**arXiv ID:** 2602.09999 | [PDF](https://arxiv.org/pdf/2602.09999v1)

**作者:** Florian Hahlbohm `[一作]` (Computer Graphics Lab), Marcus Magnor `[通讯]` (Computer Graphics Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了 Faster‑GS，一个针对 3D 高斯喷射（Gaussian Splatting）训练流程的端到端优化框架，系统性集成并验证了多种 GPU 与算法层面的加速策略，能够在保持原始重建质量和高斯数量的前提下，使训练速度提升约 5 倍、显存占用下降 30% 以上；并扩展到 4D 高斯动态场景，实现三倍加速；

**💡 创新点**

创新点包括：① 统一并实现多种现有优化（如按张量绑定裁剪、按 Z‑order 排序、按高斯反向梯度、核融合、Adam 更新融合等），② 通过共享内存、内存局部性提升实现显存访问最优，③ 将 3DGS 优化迁移至 4D 动态重建，形成完整、可复现的加速基线；

**🔧 技术方法**

技术上使用了：GPU 内存合并与共享内存策略、Z‑order 排序、Per‑Gaussian 反向梯度、卷积式核融合（前向、后向、Adam 同步）、稀疏化与密集化优化、SH 系数分离与融合、双精度/混合精度控制（虽然未采用）、CUDA 核融合与自定义 Adam；

**📊 数据集**

实验数据集涵盖 Mip‑NeRF360、Tanks & Temples、Deep Blending（静态场景）以及 D‑NeRF（8 个合成动态场景）；

**📈 对比分析**

与官方 3DGS、Radl、Taming‑3DGS、Speedy‑Splat 等基线比较，Faster‑GS 在 RTX 4090 上对 Mip‑NeRF360 及 Deep Blending 的平均训练时间分别缩短至 1/4.1、1/5.2，VRAM 下降 30%；在 4D 动态场景上亦实现约 3 倍加速，且重建质量无显著下降；

**⚠️ 局限性**

主要限制包括：① Adam 更新仍占用 40–60% GPU 时间，需进一步融合或引入二阶优化；② 某些加速技术（如 per‑Gaussian 反向梯度、Z‑order 重新排序）在低 Gaussian 数量时反而导致性能下降；③ 目前未采用混合精度或更高阶优化器，缺乏对极低显存设备的最优支持；④ 复杂动态场景的重建仍受限于原始 3DGS 纹理与 SH 参数的稀疏性处理。

---

## 453. Popularity Feedback Constrains Innovation in Cultural Markets

**arXiv ID:** 2602.09997 | [PDF](https://arxiv.org/pdf/2602.09997v1)

**作者:** Lucas Gautheron `[一作]` (University of Missouri), Nori Jacoby `[通讯]` (Cornell University)

**通讯引用:** 3496 | [OpenAlex ID](https://openalex.org/A5021757437)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开展大规模在线实验，研究受欢迎度反馈对图像创作过程中的选择与创作的影响。

**💡 创新点**

首次将社会反馈同时作用于选择与创作两阶段进行实验，揭示“文化累积优势”和“文化奔跑”机制对多样性与创新速度的影响。

**🔧 技术方法**

采用在线实验平台PsyNet、Stan贝叶斯层级模型、Vision Transformer图像嵌入、UMAP降维、BERTopic主题聚类、GPT‑5.2自动标签、Gini系数与自相关分析等技术。

**📊 数据集**

使用1,008名参与者的1,008份图像创作数据（共7,680张16×16像素图像）、486名评审者的美学/创意评分数据以及372名注释者对编辑策略的分类数据。

**📈 对比分析**

在PI（提供受欢迎度信息）与NPI（无信息）两组间进行对比，PI组的多样性显著下降，创新步长更慢，早期美学/创意评分更低，但后期趋于持平；使用多维度度量（系统进化树距离、汉明距离、语义距离）和统计检验验证差异显著。

**⚠️ 局限性**

限制：未给参与者自身作品的成功反馈；编辑像素数上限可能不足以体现真实累积进化；实验仅涉及单一二值像素域，缺乏空间/网络耦合；结果可能受样本选择与任务简化的影响。

---

## 454. ORCHID: Fairness-Aware Orchestration in Mission-Critical Air-Ground Integrated Networks

**arXiv ID:** 2602.09994 | [PDF](https://arxiv.org/pdf/2602.09994v1)

**作者:** Chuan-Chi Lai `[一作]` (National Chung Cheng University), Chi Jai Choy `[通讯]` (Feng Chia University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了ORCHID框架，旨在优化无人机（UAV）与地面基站（GBS）在关键任务空地集成网络中的协作部署。

**💡 创新点**

创新点在于结合了基于GBS的拓扑分区和基于MAPPO的重置与微调（R&F）机制，有效解决了多目标强化学习中的稳定性与灵活性困境。

**🔧 技术方法**

使用了多智能体近端策略优化（MAPPO）算法，并引入了重置与微调机制以增强学习稳定性。

**📊 数据集**

使用了Thomas聚类过程（TCP）生成的用户分布数据集，模拟了真实的异构流量模式。

**📈 对比分析**

与现有的基线方法（如MADDPG和静态K均值聚类）进行比较，ORCHID在归一化能量效率（NEE）上提高了6.8%，同时保持了服务公平性，显示出其优越的帕累托支配地位。

**⚠️ 局限性**

局限性在于该框架主要针对特定的任务场景，未来需要扩展到高移动性车辆环境，并探索集成感知与通信（ISAC）协议的应用。

---

## 455. Artisan: Agentic Artifact Evaluation

**arXiv ID:** 2602.10046 | [PDF](https://arxiv.org/pdf/2602.10046v1)

**作者:** Doehyun Baek `[一作]` (University of Stuttgart), Michael Pradel `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Artisan，一种基于 LLM 的自动化 artifact 评估系统，自动生成可执行脚本来重现研究论文中的实验表格结果。

**💡 创新点**

创新点在于将重现任务建模为代码生成问题，设计两层自动化评判机制（输出判定和方法判定）以及表格格式化工具，确保脚本既能正确输出又不靠复制预先计算的结果。

**🔧 技术方法**

核心技术包括 LLM 代理（以 GPT‑5 为主）配合 Bash 与专用格式化工具、自动化评判脚本、以及 artifact 下载与表格数字混淆处理等辅助组件。

**📊 数据集**

使用 23 篇软件工程论文构成的 ArtisanBench，共 60 个表格重现任务，涵盖 Python、Java、Rust、Scala、OCaml、Bash 等多种语言与研究方向。

**📈 对比分析**

与 mini‑swe‑agent、SWE‑agent、OpenHands 等基线比较，Artisan 在 GPT‑5.1 模型下成功生成 24 个完整重现脚本、20 个近乎完整脚本（共 44/60），超过最强基线约 4.7 倍，且在成本和执行时间上形成新的 Pareto 前沿。

**⚠️ 局限性**

主要局限包括：仅针对已 Docker 包装、无非公开 API、无 GPU 需求的论文；step 与 cost 限制可能导致部分任务未完全探索；方法判定误判率约 10%，需要人工复核。

---

## 456. Optimistic World Models: Efficient Exploration in Model-Based Deep Reinforcement Learning

**arXiv ID:** 2602.10044 | [PDF](https://arxiv.org/pdf/2602.10044v1)

**作者:** Akshay Mete `[一作]` (Texas A&M University), P. R. Kumar `[通讯]` (Texas A&M University)

**通讯引用:** 4996 | [OpenAlex ID](https://openalex.org/A5107699893)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Optimistic World Models（OWMs）框架，利用奖励偏置最大似然估计（RBMLE）原则在世界模型中引入乐观动态损失，实现高效探索。

**💡 创新点**

创新点在于将RBMLE原则转化为纯梯度的损失函数，使得乐观性直接融入模型学习，无需不等式约束或不确定性估计，并实现了可插拔的改进，适用于DreamerV3、STORM等多种世界模型。

**🔧 技术方法**

技术主要包括：基于神经网络的动态模型与策略网络，梯度估计的奖励偏置项和优势函数，乐观动态损失与熵正则化，以及α(t)和η的超参数调节。

**📊 数据集**

使用的实验数据集包括Atari 100K benchmark（多款稀疏与密集奖励游戏）和DeepMind Control（DMC）Suite（Proprio 与 Vision 两大子集），并在多种稀疏奖励环境中进行验证。

**📈 对比分析**

通过与DreamerV3、STORM及其基线对比，OWMs在稀疏奖励游戏中取得显著提升（如Private Eye、Freeway、Cartpole Swingup Sparse等）且在DMC稀疏任务上亦有改善；平均人类标准化得分从DreamerV3的97.45%提升至152.68%，OWMs的整体性能与基线相当或更优。

**⚠️ 局限性**

局限性包括：α(t)为常数或简单衰减方案，缺乏自适应设计；理论分析主要基于小规模MDP，缺少对深度网络的收敛证明；超参数（α、η）对性能敏感，需要额外调优。

---

## 457. Simple Image Processing and Similarity Measures Can Link Data Samples across Databases through Brain MRI

**arXiv ID:** 2602.10043 | [PDF](https://arxiv.org/pdf/2602.10043v1)

**作者:** Gaurang Sharma `[一作]` (VTT Technical Research Centre of Finland), Jussi Tohka `[通讯]` (A.I. Virtanen Institute for Molecular Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出一种基于标准预处理和图像相似度计算的无监督方法，用于在不同数据库、扫描仪和时间点之间识别同一受试者的去颅骨 T1 加权 MRI。

**💡 创新点**

创新点在于无需监督学习或深度网络，仅用常规图像处理（仿射配准、强度标准化、去颅骨）和简单相似度指标即可实现近乎完美的匹配，并在跨数据库、跨扫描仪、跨认知状态下验证其鲁棒性。

**🔧 技术方法**

技术包括仿射配准、直方图匹配、偏置场校正、去颅骨、十一种相似度度量（SSIM、NMI、PCC 等）以及基于核密度估计的阈值聚类。

**📊 数据集**

使用的公共数据集有：模拟数据 SLDM 与 SHCP；真实数据包括 Hormonal Health Study、Running Intervention、Traveling Human Phantom、SDSU‑TS 以及 Alzheimer’s Disease Neuroimaging Initiative (ADNI)。

**📈 对比分析**

通过对比 11 种相似度指标在预评估和评估阶段的 AUC、灵敏度和特异性，发现 SSIM、NMI、GradSim、PCC 等指标在多数据集上均能实现 AUC ≈1、灵敏度、特异性均 ≥0.99，阈值在不同数据库间保持一致，表现优异。

**⚠️ 局限性**

局限性包括：仅针对去颅骨 T1W MRI，其他扫描序列或未去颅骨图像的通用性未知；小样本或低图像质量时阈值可能不稳定；方法仍不排除在结合其他元数据时可能导致重新识别风险。

---

## 458. Effectiveness of Binary Autoencoders for QUBO-Based Optimization Problems

**arXiv ID:** 2602.10037 | [PDF](https://arxiv.org/pdf/2602.10037v1)

**作者:** Tetsuro Abe `[一作]` (Keio University), Shu Tanaka `[通讯]` (Keio University)

**通讯引用:** 1458 | [OpenAlex ID](https://openalex.org/A5057961231)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过训练二进制自编码器（bAE）学习 8 城市 TSP 方案的低维二进制潜在表示，并在该潜在空间中构造因式分解机（FM）生成的 QUBO，利用量子退火（Ising 机）进行黑盒优化，系统评估了潜在表示的重构精度、结构保留性以及对 FMQA 性能的影响。

**💡 创新点**

创新点在于：① 用自编码器直接学习符合约束的二进制潜在空间，避免手工编码导致的邻域失真；② 通过对比 Spearman 相关、邻域距离特征和局部最优比例三维度量，揭示潜在空间结构对搜索效率和可行性生成的影响；③ 证明潜在压缩既能提升搜索平滑度，又能内在化可行性，从而在量子退火中实现全可行解采样。

**🔧 技术方法**

技术手段包括：GRU 基序列到序列二进制自编码器、straight‑through 估计的二进制化、因式分解机回归构造二次 QUBO、D‑Wave 量子退火求解、以及多种结构保留评估指标（Spearman 相关、邻域距离曲线、局部最优比例）。

**📊 数据集**

使用 8 城市（L=8）的 TSP 作为实验数据集，共计 5040 条可行旅行方案，训练数据集约 5000 条（约占全空间 99%），验证集占 20%。

**📈 对比分析**

与 Rank‑Log、Rank‑Gray 与 Random‑Label 三种手工编码方式对比：bAE 在重构准确率（≈70%）上略逊于手工编码，但在结构保留指标上均优越；在 FMQA 循环中，bAE 使逼近比 R 在迭代次数上显著下降，最终达到 1（最优），并且可行样本概率 P_Feasible 始终为 1；而手工编码在可行性和收敛速度上表现较差。

**⚠️ 局限性**

局限性包括：仅在极小规模（8 城市）可解释性实验，缺乏对大规模 TSP 或其他组合问题的可扩展性验证；潜在空间维度选择仍需经验性调参；未探究不同自编码器架构、正则化或预训练对结果的影响；在量子退火硬件约束下的性能与理论可行性仍待进一步评估。

---

## 459. Position: Message-passing and spectral GNNs are two sides of the same coin

**arXiv ID:** 2602.10031 | [PDF](https://arxiv.org/pdf/2602.10031v1)

**作者:** Antonis Vasileiou `[一作]` (RWTH Aachen University), Ron Levie `[通讯]` (Technion)

**通讯引用:** 405 | [OpenAlex ID](https://openalex.org/A5001833718)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文阐述了图神经网络（GNN）中两大传统——消息传播网络（MPNN）与谱域网络——实际上是对同一类可置换不变算子不同参数化，主张用统一的算子视角来理解、比较与融合两种方法。

**💡 创新点**

创新点在于提出了一个统一的框架，将MPNN与谱GNN视为同一类可置换不变运算的不同实现；指出它们在表达能力上在大多数假设下是等价的；并明确了两者在离散结构、逻辑可表达性（MPNN）与光滑、瓶颈、社区敏感性（谱GNN）方面的互补优势。

**🔧 技术方法**

技术上主要利用群不变性、谱分解、函数计算法、可置换不变逼近理论、图拉普拉斯谱分析、W‑L 归一化判定以及对特征增强（位置编码）的理论探讨。

**📊 数据集**

本文为位置论文，未使用具体数据集；主要通过理论推导与已有文献引用来论证观点。

**📈 对比分析**

由于缺乏实验，本文没有直接的性能对比；但通过对已有实验与理论结果的梳理，说明在多种指标（表达力、光滑性、稳健性、可迁移性）上，两种方法在相同假设下可达成相似效果，且各自有独特优势。

**⚠️ 局限性**

局限性包括：缺乏统一框架下的系统实验验证；对特征增强（谱位置编码）仍处于研究早期，存在对称性与稳定性问题；以及在实际应用中如何平衡两种参数化的具体实现与计算成本仍需进一步探索。

---

## 460. Optimal PRGs for Low-Degree Polynomials over Polynomial-Size Fields

**arXiv ID:** 2602.10030 | [PDF](https://arxiv.org/pdf/2602.10030v1)

**作者:** Gil Cohen `[一作]` (Tel Aviv University), Noam Goldgraber `[通讯]` (Ben Gurion University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文构造了一个在多项式大小域上、种子长度最优（O(d log n + log q)）的伪随机生成器（PRG），适用于次数≤d的低次数多项式；

**💡 创新点**

核心创新是结合了 Lecerf 的不可约性保持技术与多项式击打集生成器（PHSG），从而在仅多项式依赖于 d 的域大小下实现最优种子长度；

**🔧 技术方法**

主要技术包括不可约性与不可分性保持的代数方法、Lecerf 的线性系统构造、PHSG 与 HSG 的组合、以及利用采样器高效构造域扩展；

**📊 数据集**

论文未使用任何具体数据集，而是针对理论构造和证明进行分析；

**📈 对比分析**

与之前的工作（如 Viola、Derksen–Viola、Dwivedi‑Guo‑Volk）相比，该方法在域大小约为 (d log d)^4 时即可达到最优种子长度，显著降低了域大小要求；

**⚠️ 局限性**

局限性在于仍需假设域特征大于 d(d−1)，且对非常小的域（如 𝔽₂）无法直接获得最优种子长度，且实现复杂度较高。

---

## 461. Overview of the TREC 2025 RAGTIME Track

**arXiv ID:** 2602.10024 | [PDF](https://arxiv.org/pdf/2602.10024v1)

**作者:** Dawn Lawrie `[一作]` (Johns Hopkins University), Andrew Yates `[通讯]` (Johns Hopkins University)

**通讯引用:** 3086 | [OpenAlex ID](https://openalex.org/A5059489981)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建并评估TREC RAGTIME 2025的多语言报告生成、单语报告生成与多语言检索任务，构建包含阿拉伯语、中文、英文、俄文约一百万条新闻文档的RAGTIME1集合，并收集125次系统提交。

**💡 创新点**

首次将检索增强生成（RAG）与长篇多语言摘要结合，提出基于背景的报告请求、引用验证机制，并引入自动ARGUE评估和未来的自生成提问（Autonuggetization）任务，创新性地把检索与生成评估统一。

**🔧 技术方法**

采用BM25、稀疏/稠密检索、LLM重排（Qwen3、Jina）、Plaid‑X检索服务以及Llama3 70B指令版进行自动评估；检索端使用多检索结果融合与重排序。

**📊 数据集**

使用RAGTIME1文档集（四语新闻）作为评测集合，NeuCLIR 2024 Report Generation Pilot 作为开发与评测辅助手段。

**📈 对比分析**

自动评估中短主题句子支持率超过0.9、nugget覆盖率低于0.5；检索任务Judge@20>0.847，顶级检索系统融合三检索结果并用Qwen3重排获得0.561的最高得分；团队间检索结果相似度呈现聚类，表明系统实现方式相似。

**⚠️ 局限性**

nugget集规模大导致短文本难以覆盖所有信息；自动评估可能偏袒使用相同技术的系统；检索评估cutoff浅层，未覆盖全部文档；引用支持与nugget匹配阶段尚未完成，系统间性能差异仍待进一步验证。

---

## 462. Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design

**arXiv ID:** 2602.10016 | [PDF](https://arxiv.org/pdf/2602.10016v1)

**作者:** Bojian Hou `[一作]` (Meta Platforms), Huayu Li `[通讯]` (Meta Platforms)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种统一的Kunlun架构，针对CTR预测同时建模序列和非序列特征，并实现可预测的缩放定律。

**💡 创新点**

低层模块和高层计算重分配的系统协同设计，GDPA、HSP、滑动窗口注意力以及CompSkip、事件级个性化和专家并行等创新。

**🔧 技术方法**

利用GDPA（通用点乘注意力）、Hierarchical Seed Pooling、Sliding Window Attention、CompSkip、事件级个性化、专家并行等技术，同时采用自定义Triton核加速。

**📊 数据集**

Meta Ads内部大规模数据集：约70B+样本，数百至数千个非序列特征，10+序列特征，序列长度从几百到几千不等。

**📈 对比分析**

与Wukong和InterFormer在同一数据集和硬件环境下对比，Kunlun在6/60/180 GFLOPs时分别提升0.31%、0.66%、0.79%的NE，缩放效率约为现有方法的2倍。

**⚠️ 局限性**

缺点包括模型结构复杂，调参成本高；实验仅在Meta内部数据上验证，缺乏公开数据的跨域验证；对极端稀疏事件的处理仍需改进。

---

## 463. A Task-Centric Theory for Iterative Self-Improvement with Easy-to-Hard Curricula

**arXiv ID:** 2602.10014 | [PDF](https://arxiv.org/pdf/2602.10014v1)

**作者:** Chenruo Liu `[一作]` (New York University), Qi Lei `[通讯]` (New York University)

**通讯引用:** 3654 | [OpenAlex ID](https://openalex.org/A5100667506)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并分析迭代自我提升（self‑improvement）在有限样本、任务中心的理论框架，并在图形推理任务上进行实验验证。

**💡 创新点**

① 将自我提升建模为奖励过滤后的最大似然微调，给出有限样本下的期望奖励下界；② 推导多步自我提升的迭代映射，揭示任务难度与初始化对提升的影响；③ 对多任务 easy‑to‑hard 课程给出可行性与提升条件，阐明难度比例与样本预算的相互作用；④ 通过 Monte Carlo 仿真与实验验证理论。

**🔧 技术方法**

统计学习理论（PAC、集中不等式）、最大似然微调与奖励过滤、概率质量“锐化”视角、难度层级模型、Llama‑3.2‑1B‑Instruct 微调。

**📊 数据集**

使用自动生成的有向无权图最短路径任务（synthetic graphs），将图与节点对转化为自然语言问答，基于 Llama‑3.2‑1B‑Instruct 进行微调；通过不同子集预训练得到不同初始化性能。

**📈 对比分析**

对比 easy‑to‑hard 课程与固定混合 baseline 的自我提升；在测试集上评估 Pass@1；实验显示：在适中难度、足够样本时 easy‑to‑hard 明显优于 baseline，且存在批量规模临界点；难度比例在 0.2–0.5 之间可获得最大提升，整体提升趋于饱和。

**⚠️ 局限性**

仅考虑二元可验证奖励的任务；理论假设有限样本、有限模型类；未深入讨论模型崩溃、极端难度与不确定性宽度的影响；实验仅在 synthetic 图任务上，缺乏在多样化真实推理基准上的验证。

---

## 464. ESTAR: Early-Stopping Token-Aware Reasoning For Efficient Inference

**arXiv ID:** 2602.10004 | [PDF](https://arxiv.org/pdf/2602.10004v1)

**作者:** Junda Wang `[一作]` (University of Massachusetts Amherst), Robert E. Tillman `[通讯]` (Optum AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种名为ESTAR的早停机制，能够在大型推理模型生成链式思考（CoT）时检测到已达到正确答案的点，并及时停止冗余推理。

**💡 创新点**

创新点在于三部分协同：①轻量化的轨迹分类器ESTAR‑LITE用局部概率和稳定性特征预测安全停点；②通过自监督微调让模型自行生成停止标记（ESTAR‑FT）；③将停点奖励融入强化学习，实现“验证后截断”的高效推理。

**🔧 技术方法**

核心技术包括LightGBM轨迹分类器、token‑level跨词预测微调、计算感知奖励的强化学习（GRPO变体）以及基于答案后验变化的停点判别理论。

**📊 数据集**

实验使用了医学 QA（USMLE、JAMA）、硬 STEM QA（GPQA）、开放式数学问答（MATH500、AIME2025）等五大公开数据集，且在跨域测试中保持强泛化。

**📈 对比分析**

与长度惩罚、AdaptThink、GRPO等基线相比，ESTAR在保持>97%原始准确率的同时，平均CoT长度缩短约3.7×（闭合问答约7×，开放问答约6×），显著提升了计算效率。

**⚠️ 局限性**

局限性包括对阈值τ的敏感性、需要模型对停止标记的良好生成能力、对极长推理或极难任务的鲁棒性尚未完全验证，以及在非标准任务格式下可能需要额外适配。

---

## 465. Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery

**arXiv ID:** 2602.09988 | [PDF](https://arxiv.org/pdf/2602.09988v1)

**作者:** Enzo Nicolas Spotorno `[一作]`, Antonio Augusto Medeiros Frohlich `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究将Kolmogorov‑Arnold网络(KAN)嵌入硬约束递归物理信息网络(HRPINN)，通过对Duffing和Van der Pol振荡器的残差进行学习与恢复，评估KAN在递归物理建模中的可行性和表现。

**💡 创新点**

首次在硬约束递归架构中对原始vanilla KAN进行系统基准测试，并探讨其在单变量多项式残差与乘法耦合残差上的优势与局限，提出对KAN在物理建模中的适用性提供经验性指导。

**🔧 技术方法**

使用HRPINN框架、可学习B‑spline一元函数的KAN、标准ReLU MLP、单步教师强制和BPTT训练策略、候选基函数拟合评估、残差R²指标等技术。

**📊 数据集**

利用Duffing（-0.3x³）和Van der Pol（(1−x²)v）两种经典振荡器的仿真轨迹，在[-2.5,2.5]×[-2.5,2.5]的100×100网格上进行残差恢复与评估。

**📈 对比分析**

通过统一的候选基函数拟合对比KANS与MLP的预测残差，评估MSE与Discovery R²。结果显示，KANS在单变量多项式残差上与MLP相当甚至略优，但在乘法耦合残差上表现不佳，方差大、易失稳；MLP在所有规模下表现更稳健且精度更高。

**⚠️ 局限性**

vanilla KAN的加性先验导致对变量耦合（乘法）表达困难，深层KAN在递归训练中极易不稳定，超参数和参数规模高度敏感；缺乏自动符号提取与优化稳定性，限制其在递归物理信息网络中的实用性。

---

## 466. Learning Force-Regulated Manipulation with a Low-Cost Tactile-Force-Controlled Gripper

**arXiv ID:** 2602.10013 | [PDF](https://arxiv.org/pdf/2602.10013v1)

**作者:** Xuhui Kang `[一作]` (University of Virginia), Yen-Ling Kuo `[通讯]` (University of Virginia)

**通讯引用:** 336 | [OpenAlex ID](https://openalex.org/A5087296750)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了低成本（$150）可通过触觉实现精细力控制的并行手爪 TF‑Gripper，并设计了配套的远程操作装置用于收集带力控制的演示数据，随后提出了 RETAF 框架，将抓取力调节与末端执行器姿态预测解耦，实现高频触觉驱动的力控制。

**💡 创新点**

①将力控制与触觉反馈集成到低成本、可跨机器人兼容的手爪中；②通过高频触觉与腕部视觉联合注意力实现实时力调节，避免了传统低频端到端策略对力控制的瓶颈；③在五个需要精细力控制的真实世界任务上，力控制显著优于位置控制，并且 RETAF 在不同基线策略上均提升稳定抓取与任务成功率。

**🔧 技术方法**

软触觉传感器（FlexiTac）与动力电流测量实现开环力估计；高频视觉‑触觉联合注意力网络（RetAF‑force）与低频基线策略（Diffusion Policy、ViTac‑MAE、π0.5）结合；通过行为克隆训练力调节网络；使用动态视觉编码器（CLIP）与简单的 2‑层 CNN 对触觉进行编码。

**📊 数据集**

每个任务收集 50 条人类演示（总计 250 条），演示中记录末端姿态、抓取力、抓取位置、RGB 图像（腕部视角与全景）和触觉信号；这些数据用于训练和评估 RETAF 以及基线策略。

**📈 对比分析**

与传统基线（DP、ViTac‑MAE、π0.5）以及位置控制/力控制两种方式比较，评估标准为 Reach、Stable Grasp、Task Success 三阶段成功率。实验表明：①力控制相较于位置控制在稳定抓取率上提升约 10–30%；②RETAF 在所有任务中均实现了比基线更高的稳定抓取和任务成功率，尤其在“Cherry Tomato Picking”和“Liquid Transfer”任务中提升 15–30% 以上；同时 RETAF 在 Reach 阶段的成功率也有所提升，说明解耦后姿态预测更精确。

**⚠️ 局限性**

实验仅覆盖五个相对简单的日常物体任务，缺乏对更大规模、多样化对象和复杂场景的验证；力控制仍基于开环电流估计，未深入讨论温度漂移、摩擦变化对力精度的长期影响；RETAF 需要额外的高频触觉与视觉数据采集，对硬件与计算资源有一定要求。

---

## 467. Humanoid Factors: Design Principles for AI Humanoids in Human Worlds

**arXiv ID:** 2602.10069 | [PDF](https://arxiv.org/pdf/2602.10069v1)

**作者:** Xinyuan Liu `[一作]` (Arizona State University), Lixiao Huang `[通讯]` (Arizona State University)

**通讯引用:** 586 | [OpenAlex ID](https://openalex.org/A5072891983)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向人机共存的“Humanoid Factors”框架，并用该框架评估了一套真实人形机器人控制算法。

**💡 创新点**

创新点在于将人因工程与人形机器人特性结合，形成物理、认知、社会与伦理四柱的系统化设计视角，并通过框架指导机器人行为的可解释性与安全性。

**🔧 技术方法**

采用了生成式大规模 AI 基础模型（如 VLA、Diffusion 等）以及层次化的感知、规划与安全监控技术。

**📊 数据集**

训练数据涵盖互联网规模多模态数据、人体动作捕捉与机器人真实交互日志，并加入合成与强化学习样本。

**📈 对比分析**

与传统基于任务完成度的评价方法相比，框架下的评估兼顾运动可读性、能耗、容错性和人类信任度，实验表明被评估算法在这些维度表现更稳健。

**⚠️ 局限性**

局限在于框架仍以理论为主，缺乏统一的量化指标；同时生成式模型在真实环境中的推理可靠性与可解释性尚待进一步验证。

---

## 468. Evaluating Disentangled Representations for Controllable Music Generation

**arXiv ID:** 2602.10058 | [PDF](https://arxiv.org/pdf/2602.10058v1)

**作者:** Laura Ibáñez-Martínez `[一作]` (Music Technology Group, Universitat Pompeu Fabra), Martín Rocamora `[通讯]` (Music Technology Group, Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文对音乐音频中无监督结构–音色解耦表示法进行系统评估，利用多轴探测框架分析其信息性、等变性、等价性与解耦程度。

**💡 创新点**

创新点在于将图像/语音中提出的四轴评估方法迁移到音乐领域，并细化针对结构和音色两种表示的等变/等价测试；同时通过消除增广和对抗损失两种策略的对照，揭示它们对解耦效果的具体影响。

**🔧 技术方法**

主要技术包括SS‑VQ‑VAE、TS‑DSAE、AFTER三种无监督解耦模型；对其嵌入进行线性探测、等变性/等价性测量以及互信息差异评估；使用SynTheory、MAESTRO、MusicNet等数据集进行多任务测试。

**📊 数据集**

实验数据集为Slakh2100（合成混音）、SynTheory（控制化乐理变体）、MAESTRO（钢琴演奏）和MusicNet（真实演奏），全部用于训练和评测。

**📈 对比分析**

通过对比三模型及其变体在十项下游任务（乐器分类、多音符估计、和弦/音符/节拍预测等）的准确率、MSE、F1和余弦相似度，发现SS‑VQ‑VAE在信息性最高但等变性最差；TS‑DSAE在节拍回归和等变性上表现最好；AFTER在结构等价性上中等，去除对抗损失后解耦与等价性显著下降。

**⚠️ 局限性**

局限性包括：1）嵌入层信息过于复杂，简单线性探测难以充分提取；2）解耦效果仍不完美，存在结构-音色信息泄露；3）评估仅停留在表示层，缺乏对生成音频层面的可控性验证；4）使用的合成数据集可能无法完全反映真实音乐的复杂性。

---

## 469. WildCat: Near-Linear Attention in Theory and Practice

**arXiv ID:** 2602.10056 | [PDF](https://arxiv.org/pdf/2602.10056v1)

**作者:** Tobias Schröder `[一作]` (Imperial College London), Lester Mackey `[通讯]` (Microsoft Research)

**通讯引用:** 37970 | [OpenAlex ID](https://openalex.org/A5038379562)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了WildCat算法，用权重核心子集压缩Transformer中的softmax注意力，实现近线性时间和超多项式误差衰减。

**💡 创新点**

创新点在于结合随机主元Cholesky子采样与Nyström加权，能够在保持高精度的同时实现超多项式误差下降，并提供GPU优化实现。

**🔧 技术方法**

采用随机主元Cholesky（RPC）、Nyström低秩近似、加权核心子集、温度缩放、分块并行等技术。

**📊 数据集**

在BigGAN图像生成、T2T‑ViT图像分类以及Qwen2.5‑7B‑Instruct的13项LongBench‑E长上下文任务上进行评估。

**📈 对比分析**

与Reformer、ScatterBrain、Performer、KDEformer、Thinformer等方法对比，WildCat在保持相同或更低误差的前提下，生成速度提升约3.7×、分类准确率接近甚至略高，KV缓存压缩质量最高。

**⚠️ 局限性**

局限在于目前未处理因果掩码的流式生成，且核心子集选择过程存在序列依赖，可通过加速RPCholesky或递归重要性采样改进。

---

## 470. Spatio-Temporal Attention for Consistent Video Semantic Segmentation in Automated Driving

**arXiv ID:** 2602.10052 | [PDF](https://arxiv.org/pdf/2602.10052v1)

**作者:** Serin Varghese `[一作]` (CARIAD SE), Kira Maag `[通讯]` (Heinrich Heine University Düsseldorf)

**通讯引用:** 101 | [OpenAlex ID](https://openalex.org/A5009008179)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将时空注意力（STA）嵌入Transformer框架的网络，用于视频语义分割，提升时序一致性和精度。

**💡 创新点**

将Temporal上下文直接融入自注意力计算，避免光流或额外模块，兼顾效率与准确性。

**🔧 技术方法**

改进自注意力机制、使用多帧上下文聚合、轻量MLP解码器。

**📊 数据集**

Cityscapes 和 BDD100k 两个道路场景视频数据集。

**📈 对比分析**

与基线单帧Transformer（SegFormer、UMixFormer）对比，mIoU 提升 0.8–1.8pp，mTC 提升 1.3–9.2pp，且 3 帧上下文最优。

**⚠️ 局限性**

对高帧率快速运动场景仍有限，T>3 会导致性能下降，且大模型推理速度受限。

---

## 471. Budgeting Discretion: Theory and Evidence on Street-Level Decision-Making

**arXiv ID:** 2602.10039 | [PDF](https://arxiv.org/pdf/2602.10039v1)

**作者:** Gaurab Pokharel `[一作]` (Virginia Tech), Patrick J. Fowler `[通讯]` (Washington University in Saint Louis)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将街头级官僚的裁量权视为有限预算下的动态分配问题，构建了一个可求解的有限期动态规划模型。

**💡 创新点**

创新点在于提出了一个阈值策略，并证明了对位置尺度族收益分布的“尺度不变行为不变性”，从而将裁量使用频率与收益分布的尾部形状相关联。

**🔧 技术方法**

技术方法包括动态规划求解阈值政策、尺度不变性证明、数值模拟验证，以及利用决策树恢复默认政策以定义裁量事件。

**📊 数据集**

所用数据集为圣路易斯住房管理信息系统（HMIS）的实际服务分配记录，用以检验裁量行为与运营约束的对应关系。

**📈 对比分析**

通过将裁量概率与可用容量、工作负荷及时间周期等变量关联，对比实验显示裁量模式与模型预测高度一致，说明阈值策略能够解释真实的决策时序。

**⚠️ 局限性**

局限性包括对收益分布假设的简化、仅在无家可归服务场景验证、未能观测到所有潜在收益以及对跨机构可比性的探索不足。

---

## 472. Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model Framework for Efficient Long-Context Inference

**arXiv ID:** 2602.10021 | [PDF](https://arxiv.org/pdf/2602.10021v1)

**作者:** Wenxuan Xie `[一作]` (Fudan University), Xuhong Wang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 1702 | [OpenAlex ID](https://openalex.org/A5060634520)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了 DRIFT，一种双模体系结构，将知识抽取与推理过程显式分离：先用轻量级知识模型对长文档进行查询相关的动态压缩，生成隐式事实 token；随后将这些 token投影到大型推理模型的嵌入空间，供推理模型完成回答。

**💡 创新点**

创新点包括：① 通过双模架构实现知识与推理的显式解耦；② 采用查询条件的动态压缩与分桶压缩策略，保证关键信息得到保留；③ 引入隐式事实 token 作为稠密知识表示；④ 设计三阶段训练目标（LFRP、QAFT-DC 与 QAFT-QA），实现知识压缩效率与推理鲁棒性的协同优化。

**🔧 技术方法**

技术实现包括：轻量级知识模型（如 1.5B/7B 规模）与大型推理模型（Mistral、Qwen2.5 7B/14B）；MLP 投影器将隐式事实 token 映射至推理模型嵌入空间；使用 LangChain 进行文档分块；训练时采用 LoRA 进行参数高效微调；三阶段训练目标：LFRP（事实重构预训练）、QAFT-DC（查询感知动态压缩）、QAFT-QA（问答微调）。

**📊 数据集**

数据集：基于 2023‑11‑01 Wikipedia snapshot 构建 300K+ 文档–问答–证据样本；训练时使用 Wiki 文档；评测使用公开长文本基准：LongBench v2、Bamboo、L‑Eval、LoCoMo、QMSUM、SPACE 等。

**📈 对比分析**

与传统硬/软压缩基线（LLMLingua‑2、xRAG、COCOM、ICAE）及原生 LLM（Mistral‑7B、Qwen2.5‑7B）对比，DRIFT 在 32× 压缩比下 LongBench v2 的准确率从 20.87% 提升至 29.22%，在多项长文本任务中保持 7× 的速度提升；ablation 结果显示三阶段训练目标各自贡献显著。

**⚠️ 局限性**

局限性：实验仅在 ≤14B 参数模型上验证，未知更大模型的扩展性；隐式事实 token 解释性差；目前仅采用监督微调，未探索 RL 等更强大策略。

---

## 473. Efficient Special Stain Classification

**arXiv ID:** 2602.09989 | [PDF](https://arxiv.org/pdf/2602.09989v1)

**作者:** Oskar Thaeter `[一作]` (Technical University of Munich), Peter J. Schüffler `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对14种常用特殊染色和两种H&E类型的全切片图像进行自动染色分类，比较多实例学习（MIL）和轻量级缩略图两种方法。

**💡 创新点**

提出基于缩略图的轻量级分类器，在保持可比准确率的同时显著提升吞吐量，并在外部TCGA数据上展示更好的跨域泛化能力。

**🔧 技术方法**

使用H0-mini ViT视觉骨干，构建ABMIL多实例学习模块和缩略图ViT+MLP分类器，并通过双阶段微调实现性能提升。

**📊 数据集**

使用TUM内部4172张WSI（涵盖14种特殊染色、H&E-FFPE、H&E-FS）作为训练/验证/内部测试集，并用TCGA H&E-FFPE/冻切样本进行外部泛化评估。

**📈 对比分析**

内部测试中MIL（k=all）宏F1为0.941/0.969，缩略图为0.897/0.953；外部TCGA测试中缩略图权重F1 0.843，优于MIL 0.807/0.768；吞吐量方面缩略图达5.635滑/秒，MIL仅0.018/0.271滑/秒。

**⚠️ 局限性**

主要局限在于模型在单一机构数据上训练，跨机构迁移仍受限；对稀有染色和多样扫描条件的鲁棒性不足，需要进一步加入开放集检测与不确定性估计。

---

## 474. RoboSubtaskNet: Temporal Sub-task Segmentation for Human-to-Robot Skill Transfer in Real-World Environments

**arXiv ID:** 2602.10015 | [PDF](https://arxiv.org/pdf/2602.10015v1)

**作者:** Dharmendra Sharma `[一作]`, Laxmidhar Behera `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种端到端的人机子任务分割与执行框架RoboSubtaskNet，并构建了可直接映射到机器人动作的子任务标注数据集RoboSubtask，完成了从视频分割到实际机械臂执行的完整闭环；

**💡 创新点**

创新点包括：①将注意力融合的I3D特征与改进的Fibonacci扩张MS‑TCN相结合；②引入转移感知损失以抑制无效子任务切换；③提供与机器人原语对齐的子任务级数据集，填补了传统HAR与机器人执行之间的鸿沟；

**🔧 技术方法**

采用的技术包括：RGB+光流的I3D特征提取与注意力融合；改进的Fibonacci扩张多阶段TCN模型；交叉熵、时序均方误差与转移感知三项复合损失；YOLOv8目标检测；基于DMP的轨迹规划与比例控制器实现机械臂运动；

**📊 数据集**

使用的数据集包括公开的GTEA、Breakfast以及作者自建的RoboSubtask（医疗与工业演示，子任务级标注）；

**📈 对比分析**

与MS‑TCN和MS‑TCN++对比实验表明，RoboSubtaskNet在GTEA的边界敏感度指标（F1@50 79.5%）和在RoboSubtask上的整体性能（F1@50 94.2%、Edit 95.6%）均优于基线；在Breakfast上略逊于MS‑TCN++（F1@50 30.4%），但在实际机械臂执行中达到了约91.25%的成功率；

**⚠️ 局限性**

局限性在于：①需要人工监督的子任务标签；②假设任务为短时域子任务，难以处理长时域复杂任务；③仅使用单目RGB+光流，受感知误差影响较大；④映射到机器人动作是确定性的，缺乏运行时安全监控与自适应修正。

---

## 475. Features as Rewards: Scalable Supervision for Open-Ended Tasks via Interpretability

**arXiv ID:** 2602.10067 | [PDF](https://arxiv.org/pdf/2602.10067v1)

**作者:** Aaditya Vikram Prasad `[一作]` (Goodfire AI), Ekdeep Singh Lubana `[通讯]` (Goodfire AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将模型内部特征（如对事实性的不确定度）作为奖励信号，训练RL策略减少大型语言模型的幻觉并实现可插入式纠正

**💡 创新点**

将解释性探测器转化为可量化、可扩展的奖励函数，实现对开放式行为（如事实性）的强化学习，避免昂贵的外部评判器

**🔧 技术方法**

使用探测器（probe）读取模型激活，基于Gemini 2.5 Pro 评估标签训练奖励探测器；采用ScaleRL+CISPO的RL框架；在训练时利用最佳-32采样、B-O-N 采样做测试时计算扩展

**📊 数据集**

Longfact++（约2万道题的长文生成数据集）用于探测器训练和评估；Gemini 2.5 Pro 结合网络搜索做幻觉判定与奖励标签；Gemma‑3‑12B‑IT 作为基准模型

**📈 对比分析**

与原始Gemma‑3‑12B‑IT模型、无B‑O‑N采样、无内联干预版本及使用监控+内联干预的基线进行对比；在幻觉率上实现58%下降；在标准基准（如MMLU、ARC等）保持与基线相当；B‑O‑N采样进一步提升正确率，远优于直接用Gemma评估

**⚠️ 局限性**

探测器可能被学生模型逃避的风险；奖励信号依赖模型特征的校准，若失准效果差；对内联干预的过度修正可能导致模型分布漂移；对不同开放式行为的泛化仍需进一步验证

---

## 476. AIDED: Augmenting Interior Design with Human Experience Data for Designer-AI Co-Design

**arXiv ID:** 2602.10054 | [PDF](https://arxiv.org/pdf/2602.10054v1)

**作者:** Yang Chen Lin `[一作]` (National Tsing Hua University), Po-Chih Kuo `[通讯]` (National Tsing Hua University)

**通讯引用:** 2953 | [OpenAlex ID](https://openalex.org/A5053702420)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估AIDED工作流，将多模态客户体验数据（人口统计、眼动热图、问卷可视化、AI预测注意力覆盖）嵌入到生成式AI（GAI）辅助的室内设计过程中，并通过12名专业设计师的四条件内测实验验证不同数据模态对设计决策、认知负荷、信任感与创意输出的影响。

**💡 创新点**

创新点在于：①首次将客观眼动数据与主观问卷结果统一呈现，并与AI预测覆盖层相结合，形成真实性–可解释性权衡框架；②设计了一种基于Grad‑CAM的AI预测注意力覆盖，并引入LLM（ChatGPT‑5）提供自然语言解释与设计建议；③通过经验研究与定量评估相结合，系统性揭示不同模态在GAI协同设计中的效用与局限，提出可行的实务建议。

**🔧 技术方法**

使用技术包括：文本到图像生成（Gemini‑2.0‑flash）、眼动追踪（Tobii Pro Spark）、问卷数据可视化（柱状图/雷达图）、AI注意力覆盖（多模态预测模型 + Grad‑CAM）、LLM解释生成（ChatGPT‑5）。

**📊 数据集**

使用的数据集：30名真实客户的体验数据（人口统计、眼动轨迹、问卷、口述反馈）以及16段首人视角室内漫游视频（每段80帧），覆盖4种设计风格（现代、北欧、侘寂、无印良品）。

**📈 对比分析**

比较方法为四条件内测（C1–C4）对设计师自评与问卷评估的非参数统计（Friedman、Wilcoxon），并对整体系统评估使用一次性Wilcoxon与FDR校正。结果显示：问卷可视化在决策支持与信任感上最优；眼动热图在可视化清晰度上突出，但可解释性弱；AI预测覆盖在缺乏LLM解释时效果有限；LLM支持显著提升可解释性与设计建议的实用性。总体而言，AIDED提升了设计师对客户需求的把握与信心，但在创意自由度方面提升有限。

**⚠️ 局限性**

局限性包括：①参与者样本量仅12名，缺乏更大规模与不同背景的验证；②实验仅涉及室内静态图像与固定风格，未覆盖建筑、公共空间或动态场景；③缺乏真实项目落地与长期效益评估；④AI预测覆盖与LLM解释依赖预训练模型，可能存在偏差与可解释性不足。

---

## 477. Automata on Graph Alphabets

**arXiv ID:** 2602.10036 | [PDF](https://arxiv.org/pdf/2602.10036v1)

**作者:** Hugo Bazille `[一作]` (EPITA Research Lab), Uli Fahrenberg `[通讯]` (Université Paris-Saclay)

**通讯引用:** 934 | [OpenAlex ID](https://openalex.org/A5002354883)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在图结构字母表上定义的有限自动机模型，并建立了其基本理论。

**💡 创新点**

创新点在于将字母表约束为有向图，给出了对应的 Kleene 与 Myhill–Nerode 定理，并证明了可判定性、确定化、补集与最小化等性质。

**🔧 技术方法**

技术上利用了图字母表的自由范畴结构、并、连、幂运算以及标准的自动机闭包构造与子集构造。

**📊 数据集**

该研究为纯理论探索，不使用特定数据集；所有结果均为形式证明。

**📈 对比分析**

在理论层面上通过构造证明了与传统自由字母表自动机的等价性和可判定性，没有实验性能比较。

**⚠️ 局限性**

局限性包括在无限图字母表时不保持补集闭包，以及未解决更一般的预单纯形字母表或高阶并发模型的可判定性问题。

---

## 478. Perception with Guarantees: Certified Pose Estimation via Reachability Analysis

**arXiv ID:** 2602.10032 | [PDF](https://arxiv.org/pdf/2602.10032v1)

**作者:** Tobias Ladner `[一作]`, Matthias Althoff `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4`

**🎯 论文内容**

无法确定具体内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法确定比较方法及性能表现

**⚠️ 局限性**

缺乏足够信息，无法评估局限性

---

## 479. On the generalization of $g$-circulant MDS matrices

**arXiv ID:** 2602.10028 | [PDF](https://arxiv.org/pdf/2602.10028v1)

**作者:** Atif Ahmad Khan `[一作]` (Aligarh Muslim University), Bhupendra Singh `[通讯]` (Defence Research and Development Organisation)

**通讯引用:** 1049 | [OpenAlex ID](https://openalex.org/A5042898723)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出并研究了一类新的矩阵结构——consta‑g‑circulant 矩阵，给出了其可逆性、MDS 性质以及自反性（involutory）和半自反性（semi‑involutory）的充分必要条件，并将该结构推广至含有域自同构的 consta‑θ_g‑circulant 矩阵；同时提供了针对阶 3 与 4 的枚举算法和实例。

**💡 创新点**

创新点在于：① 定义了同时包含参数 λ 与 g 的新矩阵族 consta‑g‑circulant 与 consta‑θ_g‑circulant；② 通过 Chinese Remainder 定理给出了可逆矩阵的闭式计数公式；③ 建立了可逆、MDS 与自反性之间的完整对应关系；④ 首次将 skew polynomial 环与自同构引入该框架。

**🔧 技术方法**

采用了有限域多项式环、模多项式同构、线性变换理论、Chinese Remainder 定理、矩阵对称性分析以及 skew polynomial 环与域自同构的代数工具。

**📊 数据集**

以多种有限域（如 𝔽₈、𝔽₉、𝔽₁₆、𝔽₂₅）为实验背景，给出对应的 x^m−λ 分解与可逆矩阵计数实例。

**📈 对比分析**

通过理论推导得到可逆矩阵的闭式计数与 MDS 判定标准，算法主要依赖多项式系数的枚举，实验示例证明可行；虽未给出具体时间复杂度，但与传统 circulant 的存在性限制相比，所提出结构更具可实现性。

**⚠️ 局限性**

局限性在于：计数上限可进一步细化；未探讨 orthogonal consta‑g‑circulant 的情况；算法主要针对阶 3 与 4，扩展到更高阶仍需研究。

---

## 480. SCORE: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation

**arXiv ID:** 2602.10017 | [PDF](https://arxiv.org/pdf/2602.10017v1)

**作者:** Homaira Huda Shomee `[一作]` (University of Illinois Chicago), Tanwi Mallick `[通讯]` (Argonne National Laboratory)

**通讯引用:** 401 | [OpenAlex ID](https://openalex.org/A5037238294)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对高风险专业领域的LLM生成答案的多维、无参考评估框架，并构建了1412条覆盖40种专业角色和7种自然灾害的合成问答数据集。

**💡 创新点**

创新点包括：①四维评估（特异性、鲁棒性、答案相关性、上下文利用）以及可解释的多代理LLM‑judge特异性判定；②在检索增强生成中引入语义掩码与BGE reranker提升答案相关性；③采用几何平均token概率等技术估计模型对检索上下文的信心，量化上下文利用。

**🔧 技术方法**

技术方法：LLM‑as‑judge（GPT‑4o、Qwen3‑8B）、检索增强生成（RAG）、语义掩码、BGE reranker、离散式鲁棒性评估、几何平均token概率与不确定性估计。

**📊 数据集**

使用自建的合成问答数据集（1412对），涵盖40专业角色、7种灾害、美国多地点与时间维度。

**📈 对比分析**

与GPT‑4o、Gemini 2.5等模型对比：GPT‑4o在特异性上显著优于Gemini；两者在鲁棒性（对句法变形）和答案相关性上表现相近；上下文利用指标显示大多数检索断言对答案贡献正面；人类评估与自动评估一致性中等，表明多指标评估更可靠。

**⚠️ 局限性**

局限性：①依赖固定的约60万条气候相关知识库，若覆盖不足会误检；②合成数据难以完全覆盖真实场景复杂性；③人类评估主观性高，尤其在时间、强度等属性；④部分评估需模型提供token‑级logits，受限于模型接口。

---

## 481. Human-AI Synergy Supports Collective Creative Search

**arXiv ID:** 2602.10001 | [PDF](https://arxiv.org/pdf/2602.10001v1)

**作者:** Chenyi Li `[一作]` (Cornell University), Nori Jacoby `[通讯]` (Cornell University)

**通讯引用:** 3496 | [OpenAlex ID](https://openalex.org/A5021757437)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在单词猜测游戏中探究人机协作对创造力与多样性的影响，比较人类、AI与人机混合三类组的表现。

**💡 创新点**

发现人机混合组既能提升目标相似度，又能保持或提升猜测多样性，揭示人机之间的第二阶适配效应，表明协作优势源于两者认知差异而非单纯组合。

**🔧 技术方法**

使用大语言模型 Gemini 2.5 Flash（以及 GPT 5.1 对比）进行一词提示，基于 Word2Vec 的语义嵌入与 UMAP 降维评估猜测，采用类似 Semantle 的余弦相似度得分机制。

**📊 数据集**

数据集为约663,000个常见英文单词的嵌入，10个隐藏目标词（每词5场游戏，共50场），人类参与者来自 Prolific，AI 通过 Google Gemini API 调用实现。

**📈 对比分析**

通过个体与集体最高相似度、平均相似度、以及词语多样性指标与标准偏差比较，混合组在所有性能指标上均显著优于人类单独组和 AI 单独组；对照实验显示模型多样性（Gemini vs GPT）对协作收益影响有限，提示人机互适是关键。

**⚠️ 局限性**

局限性包括：任务仅限单词猜测，缺乏真实创作场景；网络拓扑为线性链且信息共享仅限最佳猜测；多样性度量仅基于语义相似度，未结合人类评判；实验仅覆盖少数 LLM 与提示方案，未涵盖更广泛的模型或交互模式。

---

## 482. A Unified Assessment of the Poverty of the Stimulus Argument for Neural Language Models

**arXiv ID:** 2602.09992 | [PDF](https://arxiv.org/pdf/2602.09992v1)

**作者:** Xiulin Yang `[一作]` (Georgetown University), Ethan Gotlieb Wilcox `[通讯]` (Georgetown University)

**通讯引用:** 974 | [OpenAlex ID](https://openalex.org/A5011708753)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了针对贫穷刺激假说（PoSH）中经典句法现象（疑问句形成、岛屿约束、绑定理论、wanna 缩写）的统一评测基准，并在仅包含 10‑50M 单词的儿童级别训练语料上训练 GPT‑2 变压器模型，考察其在缺少直接正例、不同输入类型与数据规模下的学习表现；进一步引入三种认知启发的归纳偏差（Dyck 预训练、树植入 Transformer、动态递归偏差）评估其对 PoSH 现象学习的影响。

**💡 创新点**

创新点在于①首次提出针对 PoSH 现象的完整、可重复的评测基准；②在儿童输入规模下系统评估 Transformer 的学习能力；③细粒度探讨直接正例缺失、输入类型与数据量对 PoSH 学习的交互效应；④将三种认知启发的偏差与标准 Transformer 对比，验证它们在提高整体句法能力与针对 PoSH 现象学习方面的差异。

**🔧 技术方法**

采用 GPT‑2 变压器作为基础模型，进行自回归语言建模；使用 k‑Dyck 语言进行预训练以提供弱层次偏差；构建 Tree‑Planted Transformer（TPT）在注意力中加入句法树距离矩阵以实现强结构偏差；实现动态递归偏差通过在注意力分数中加入递减距离惩罚。评估方式为极小对比的困惑度比较，计算每对的正确率。

**📊 数据集**

训练数据：结合 CHILDES、OpenSubtitles、BNC、Switchboard 等语音转录与 TinyStories、Project Gutenberg、Simple English Wikipedia 等文本，构成 10M、30M、50M、100M 单词的“baby‑base”子集；成人文本使用 Wiki‑100M 作为基准。评测数据为自制的 PoSH 评测集，包含 9 个子类别、共 500 个极小对。

**📈 对比分析**

对比方法：在不同输入类型（baby‑base vs. Wiki‑100M）、不同数据规模（10M、30M、50M、100M）以及是否包含直接正例（过滤 vs. 保留）下训练模型，并对三种偏差进行 ablation。结果显示，Transformer 在 10M 词即可超随机，但整体准确率仍低于儿童在同一阶段的表现；成人文本或直接正例的增补对 PoSH 性能提升有限；偏差虽然显著提升广义句法基准（BLiMP、Zorro、SCaMP）的平均准确率，却对 PoSH 评测的特定现象提升不明显。

**⚠️ 局限性**

局限性：仅使用文本输入，未覆盖多模态与社会交互；偏差实现仅为三种示例，可能未覆盖人类所需的全部语言约束；评测集规模有限，单一语言（英语）且未充分考虑不同子类别的交叉验证；与人类对照存在评测协议差异，难以直接量化数据效率差距。

---

## 483. Quantum Multiple Rotation Averaging

**arXiv ID:** 2602.10115 | [PDF](https://arxiv.org/pdf/2602.10115v1)

**作者:** Shuteng Wang `[一作]` (Max Planck Institute for Informatics), Vladislav Golyanik `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 2973 | [OpenAlex ID](https://openalex.org/A5080103406)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 IQARS，一种将多旋转平均问题映射到量子退火器上求解的迭代框架。

**💡 创新点**

创新点在于将 MRA 通过 SO(3) 的切空间线性化并二进制化为可在量子退火器上求解的 QUBO 子问题，配合后验 Boltzmann 权重聚合。

**🔧 技术方法**

使用量子退火、SO(3) Lie 代数映射、二进制编码、后验能量加权投票。

**📊 数据集**

使用合成无噪声旋转图、噪声扰动的合成数据，以及真实 SfM 数据集 Fountain、Castle、Herz‑Jesus。

**📈 对比分析**

与 Shonan、L1‑IRLS、局部优化等方法对比，IQARS 在噪声数据上平均约 12% 的残差下降，达到最佳残差。

**⚠️ 局限性**

局限在于当前量子硬件资源有限，需粗粒度离散化，链断裂率升高，难以扩展到更大规模问题。

---

## 484. ConsID-Gen: View-Consistent and Identity-Preserving Image-to-Video Generation

**arXiv ID:** 2602.10113 | [PDF](https://arxiv.org/pdf/2602.10113v1)

**作者:** Mingyang Wu `[一作]` (Texas A&M University), Zhengzhong Tu `[通讯]` (Texas A&M University)

**通讯引用:** 2357 | [OpenAlex ID](https://openalex.org/A5015173810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ConsID-Gen，一种利用多视角辅助图像与双流视觉-几何编码器的图像到视频生成框架，以保持对象身份和几何一致性。

**💡 创新点**

创新点在于：①构建大型对象中心的 ConsIDVid 数据集和 ConsIDVid-Bench 评测基准；②引入双流视觉-几何编码器和文本-视觉联接器，先对多视角视觉信息进行精细预对齐；③在扩散变压器前进行统一条件化，显著提升身份保真度。

**🔧 技术方法**

核心技术包括 CLIP/CLIP‑style 2D 编码器、VGGT 结构化几何编码器、双流注意力模块、文本‑视觉多模态联接器，以及基于 Diffusion Transformer 的视频生成后端。

**📊 数据集**

使用 ConsIDVid 数据集（约 1.1 万条多视角对象视频）和公开来源（Co3D、OmniObject3D、Objectron 等）合成的视频，评测基准 ConsIDVid‑Bench 包含专有和公开两部分。

**📈 对比分析**

与 Wan2.1、Wan2.2、HunyuanVideo 等主流 I2V 模型对比，ConsID-Gen 在 ConsIDVid‑Bench 的身份一致性、MEt3R、Chamfer Distance 等几何与外观指标上均取得最优或竞争性表现，尤其在专有子集的身份一致性得分提升约 3–4%。

**⚠️ 局限性**

局限性包括：对复杂背景（如网格纸等）易出现生成崩塌；依赖多视角辅助图像质量；主要针对刚体对象，非刚体或极端姿态的身份保持仍需改进。

---

## 485. ST4VLA: Spatially Guided Training for Vision-Language-Action Models

**arXiv ID:** 2602.10109 | [PDF](https://arxiv.org/pdf/2602.10109v1)

**作者:** Jinhui Ye `[一作]` (Shanghai AI Laboratory), Jiangmiao Pang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种双系统 Vision‑Language‑Action 框架，利用空间引导训练在保持空间先验的前提下实现高质量的机器人控制；

**💡 创新点**

创新点在于将空间先验的学习与动作生成分离，采用两阶段空间引导训练，利用空间提示和梯度衰减实现感知与动作优化的对齐，并引入轻量化查询 Transformer 以实现高效的跨系统沟通；

**🔧 技术方法**

技术手段包括：Qwen2.5‑VL 视觉语言编码器、DINOv2 视觉特征提取、DiT Actor 低级控制器、空间提示（spatial prompting）、梯度衰减因子、两阶段预训练/后训练流程；

**📊 数据集**

使用了海量网络视觉‑语言数据（RefCOCO、LLaVA‑OneVision 等）、机器人专用数据集（RoboRefIt、A0 等）、SimularEnv、LIBERO、GenManip 的模拟任务，以及 1K 条真实机器人抓取‑放置轨迹；

**📈 对比分析**

与 π_0、GR00T、OpenVLA、CogACT、Magma 等基线在 Google Robot、WidowX、SimplerEnv、LIBERO 以及真实机器人和长时序任务上进行对比。模型在视觉匹配任务上提升 5.9%‑14.6%，在视觉聚合任务提升 5.3%‑12.4%，在 WidowX 任务提升 9.8%‑17.0%，在长时序任务中亦表现出更高的成功率和更强的适应性；

**⚠️ 局限性**

局限性包括：需要大量预训练和后训练数据；对空间提示的依赖在无明显空间结构的任务中可能效果不佳；模型复杂度较高，部署在资源受限的机器人上仍有挑战；仅在抓取‑放置等操作任务上验证，其他类型操作的泛化性尚未充分评估。

---

## 486. VideoWorld 2: Learning Transferable Knowledge from Real-world Videos

**arXiv ID:** 2602.10102 | [PDF](https://arxiv.org/pdf/2602.10102v1)

**作者:** Zhongwei Ren `[一作]` (ByteDance Seed), Xiaojie Jin `[通讯]` (ByteDance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本工作提出VideoWorld 2，研究如何从未标注的真实视频中直接学习可迁移的长时序任务知识，并在手工艺和机器人环境中验证该方法；

**💡 创新点**

创新点包括：①将视觉外观与动作动力学解耦，提出动态增强潜在动力学模型（dLDM）；②利用预训练视频扩散模型（VDM）作为外观先验，使潜在代码专注于任务核心动作；③通过自动回归Transformer对潜在动作序列建模，实现跨环境的可迁移长时序任务执行；

**🔧 技术方法**

采用技术：VQ‑VAE编码未来视觉变化为离散潜在码；预训练Cosmos DiT 2B视频扩散模型负责外观重建；投影层与因果交叉注意力将潜在码输入VDM；Cosmos AR 4B Transformer预测潜在码序列；使用ControlNet‑style控制器稳定VDM训练；评估指标包括Sequential Task Success Rate、LPIPS、SSIM；

**📊 数据集**

使用数据集：①Video‑CraftBench（≈7小时手工艺教程视频，约9.5k段视频）用于训练与评估；②OpenX（大规模机器人演示视频）用于跨域预训练；③CALVIN（34个长时序机器人任务）用于下游测试与预训练验证；

**📈 对比分析**

与四种最先进视频生成模型（Cosmos AR 4B、Cosmos DiT 2B、Wan2.2 14B、HunyuanVideo 13B）、五种潜在动作模型以及原版VideoWorld进行对比。VideoWorld 2在纸飞机折叠任务的Sequential Success Rate达68.8%，在块堆叠任务81.5%，远优于基线（≤10%）；在CALVIN上通过OpenX预训练后，成功率提升至≈70%–85%，接近全标签模型；同时视觉质量（SSIM/LPIPS）也显著提升；

**⚠️ 局限性**

限制与挑战：①对极长时序任务的推理仍受限，生成长度受Transformer和VDM上下文长度限制；②对外观变化的鲁棒性在更大视觉差异场景下尚未充分验证；③需要大量计算资源（预训练VDM、Transformer大规模训练）；④对多模态输入（语音、文本）支持不足，未来可扩展多模态学习。

---

## 487. Quantum-Audit: Evaluating the Reasoning Limits of LLMs on Quantum Computing

**arXiv ID:** 2602.10092 | [PDF](https://arxiv.org/pdf/2602.10092v1)

**作者:** Mohamed Afane `[一作]` (Fordham University), Juntao Chen `[通讯]` (Fordham University)

**通讯引用:** 1820 | [OpenAlex ID](https://openalex.org/A5100780558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Quantum-Audit 基准，用于系统评估大型语言模型在量子计算领域的知识与推理能力，涵盖多选、开放式和错误前提三种题型；

**💡 创新点**

首次构建量子计算专用评测集，包含 2,700 道题目，并引入错误前提题目检验模型识别错误假设的能力；

**🔧 技术方法**

采用多模型评测框架，使用标准提示模板和 JSON 题库；对 26 个公开与闭源 LLM 进行统一评测；

**📊 数据集**

基准数据集由 1,000 道专家编写题目、1,000 道 LLM 自动生成并专家校验的题目、以及 700 道开放式/错误前提题目组成；

**📈 对比分析**

将 LLM 性能与 43 名量子专家与 30 名从业者的分数进行对比；顶尖模型 Claude Opus 4.5 及 GPT‑5.2 Pro 达到约 84%‑85% 的整体准确率，超过专家平均水平；在高级安全题上仅达 73% 以上；错误前提题准确率仅 64%‑66%；

**⚠️ 局限性**

评测仅以准确率为主，未考虑模型可信度或生成文本的深度；多语言子集有限，未覆盖更多语言；模型对最新研究动态的把握仍不足；

---

## 488. Allocation Proportionality of OWA--Based Committee Scoring Rules

**arXiv ID:** 2602.10083 | [PDF](https://arxiv.org/pdf/2602.10083v1)

**作者:** Daria Boratyn `[一作]` (Jagiellonian University), Dariusz Stolicki `[通讯]` (Jagiellonian University)

**通讯引用:** 64 | [OpenAlex ID](https://openalex.org/A5079790151)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了在多胜选举中基于政党框架的分配比例性（allocation proportionality）概念，并为基于 OWA 的委员会计分规则定义了量化的分配比例性度量；随后利用该度量在大量模拟的政党多区选举中，对 SNTV、k‑Borda、Chamberlin–Courant、Harmonic Borda、Proportional k‑Approval（PAV）以及 Bloc Voting 等常用规则的分配比例性表现进行实验比较。

**💡 创新点**

① 把比例代表性从传统的块代表性或 PSC 扩展到可直接与席位比例相比较的分配比例性；② 通过聚合党得分构造了可用于任何 OWA 规则的投票“代理”ψ；③ 用 α‑散度、Lp 范数和有效党派数等指标统一度量分配比例性；④ 系统性实验验证了不同规则在不同模型和参数下的比例性特征，揭示了如 PAV 最优、k‑Borda/Bloc 最差、CC 与 Harmonic Borda 对党派数和区级大小的非单调依赖等现象。

**🔧 技术方法**

使用 OWA‑based committee scoring 规则框架；定义聚合党得分 ψ；引入 α‑散度（α=1）和 Lp 范数、有效党派数等量化指标；在 1D/2D 欧氏、Mallows、单峰、IC 等概率模型下生成大量合成政党选举；采用 256 次实验（每组参数）进行统计比较；利用图表展示 α‑散度、L2 范数、分数化/偏差等结果。

**📊 数据集**

完全基于合成数据：在 1 维/2 维欧氏空间中生成政党、候选人和选民理想点，随后在每个区按高斯或均匀分布采样；Mallows、Conitzer、Walsh、IC 等模型的随机投票；每个实验包含 1024 名选民、k 个候选人/党派，设置多区规模（如 128 区、64 区等）以保证席位总数至少 128。

**📈 对比分析**

通过计算各规则在同一模拟设定下的 ψ 与席位比例差异，用 α‑散度（KL 散度）和 L2 范数衡量分配比例性；结果显示：PAV 的分配比例性最优，α‑散度与 L2 值均最低；SNTV 与 k‑Borda 在大党派面前偏大、在小党派面前偏小；Bloc 与 Harmonic Borda 处于中间；Chamberlin–Courant 对党派数和区级大小表现出非单调性，既可能在小党派数量下表现良好，也可能在大区级下严重偏大。总体而言，规则的分配比例性随委员会大小、党派数量和模型类型而变化。

**⚠️ 局限性**

① 仅为经验性研究，缺乏严格的理论证明；② 只关注 OWA‑based 规则，未涉及其他类型的多胜规则；③ 结果仅基于合成数据，未验证在真实选举中的适用性；④ 对分配比例性的度量假设了党派已知，实际选举中党派划分可能不完整；⑤ 对 PSC、EJR 等传统比例代表性公理的关系仍未得到解析；⑥ 由于席位离散性，规则无法完全满足分配比例性，评估只能是“近似”。

---

## 489. Anagent For Enhancing Scientific Table & Figure Analysis

**arXiv ID:** 2602.10081 | [PDF](https://arxiv.org/pdf/2602.10081v1)

**作者:** Xuehang Guo `[一作]` (William and Mary), Qingyun Wang `[通讯]` (William and Mary)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模科学表格与图像分析基准，并提出了一种基于四个专门代理的多代理分析框架。

**💡 创新点**

基准覆盖63,178个实例，跨9大科学领域、170个细分学科，并按7维复杂度划分；多代理系统通过任务分解、工具驱动检索、生成与反思四个阶段，结合测试时优化与模块化训练，实现高质量分析。

**🔧 技术方法**

使用多模态大型语言模型（MLLM）作为各代理的核心；通过监督微调（SFT）与强化学习（GRPO）实现代理能力提升；配备16个专用工具箱、检索插件和五维评估（ROUGE、SciBERT-Score、MLLM-as-judge）等技术。

**📊 数据集**

基准数据来自公开学术论文（如arXiv、PubMed等）自动提取的表格与图像，并结合其上下文信息。

**📈 对比分析**

与现有单代理MLLM基线相比，训练无监督的多代理在零/一-shot场景下提升约13.4%相对，微调后提升至42.1%；在多维评估中显示显著优势，并与人类专家评估结果高度一致。

**⚠️ 局限性**

仍存在模型规模限制、工具匹配不完善、对小模型的反思能力不足等问题；基准仅覆盖PDF/HTML格式，未覆盖所有出版物类型，未来需扩展数据来源和改进代理间的协同机制。

---

## 490. Can Image Splicing and Copy-Move Forgery Be Detected by the Same Model? Forensim: An Attention-Based State-Space Approach

**arXiv ID:** 2602.10079 | [PDF](https://arxiv.org/pdf/2602.10079v1)

**作者:** Soumyaroop Nandi `[一作]` (University of Southern California), Prem Natarajan `[通讯]` (University of Southern California)

**通讯引用:** 7833 | [OpenAlex ID](https://openalex.org/A5066184920)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种名为Forensim的框架，利用三类掩模（无瑕、源区、目标区）联合定位图像伪造及其来源。

**💡 创新点**

创新点在于提出基于状态空间的相似性与操作注意力模块，能够以线性复杂度捕捉全局复制模式，并首次将拼接与复制移动伪造统一在三类监督下训练。

**🔧 技术方法**

使用的技术包括Vision Transformer + 视觉状态空间网络（Mamba/VMamba）、旋转位置编码、局部增强位置编码、非局部归一化以及多尺度注意力块。

**📊 数据集**

使用的数据集包括公开的拼接数据集、CMFD基准（USC-ISI、CoMoFoD、CASIA）以及新发布的高分辨率CMFD_Anything数据集。

**📈 对比分析**

通过在像素级和图像级指标（Precision、Recall、F1、AUC）与多种基线（BusterNet、ManTra-Net、DOA-GAN、TruFor、SparseViT）进行对比，Forensim在大多数基准上均实现了SOTA性能，尤其在CMFD_Anything上显著提升。

**⚠️ 局限性**

局限性包括对背景匹配拼接、重复结构以及低色彩/灰度图像的误检率较高，未来可通过频域/边界检测头、硬负样本挖掘或灰度增强来进一步提升鲁棒性。

---

## 491. Olaf-World: Orienting Latent Actions for Video World Modeling

**arXiv ID:** 2602.10104 | [PDF](https://arxiv.org/pdf/2602.10104v1)

**作者:** Yuxin Jiang `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**通讯引用:** 3492 | [OpenAlex ID](https://openalex.org/A5068937750)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于SeqΔ-REPA的可迁移潜在动作学习方法及Olaf-World视频世界模型预训练框架，实现从无标签视频中自动学习并迁移可控动作；

**💡 创新点**

创新点在于引入序列级控制‑效应对齐目标SeqΔ-REPA，将冻结自监督视频编码器的特征差分作为全局参考，解决跨场景潜在动作非可识别性问题；

**🔧 技术方法**

使用β‑VAE逆动力学编码器、对齐损失、Diffusion Transformer（DiT）以及LoRA适配器构建可迁移动作空间与视频生成器；

**📊 数据集**

主要使用MiraData的3D Rendering与City Walking数据集进行预训练，使用MIND（Unreal Engine 5）数据集进行跨域评估；

**📈 对比分析**

通过与AdaWorld等基线在线性探针、原型相似度、零样本动作转移及少量标签适配等指标的对比，Olaf-World在跨域可识别性、零样本转移和适配精度上均显著优于基线，并在OOV场景保持更好的可控性；

**⚠️ 局限性**

局限在于仍依赖自监督编码器作为参考，对极端视觉偏移或完全不同动作空间的迁移效果有限，且对高频动作细节的建模仍有提升空间。

---

## 492. Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders

**arXiv ID:** 2602.10099 | [PDF](https://arxiv.org/pdf/2602.10099v1)

**作者:** Amandeep Kumar `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**通讯引用:** 22364 | [OpenAlex ID](https://openalex.org/A5004716468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出在预训练表示编码器的球面特征空间上使用Riemannian流匹配与Jacobi正则化，使标准Diffusion Transformer能够收敛并生成高保真图像。

**💡 创新点**

创新点在于：① 识别欧氏流匹配在球面特征上因几何干扰导致收敛失败；② 通过SLERP构造球面测地线轨迹；③ 引入Jacobi场正则化校正曲率影响，从而消除对模型宽度扩展的需求。

**🔧 技术方法**

使用技术包括Riemannian流匹配、Jacobi场正则化、SLERP插值、测地线积分器，以及LightningDiT等Transformer架构，配合预训练的表示编码器（DINOv2、SigLIP、MAE）等。

**📊 数据集**

在ImageNet 1K 256×256数据集上进行训练与评估，并在不同模型规模（DiT‑B、DiT‑L、DiT‑XL）上验证效果。

**📈 对比分析**

与DiT、LightningDiT、REPA、Euclidean Flow Matching等基线对比，DiT‑B 131M在200轮训练下取得FID 3.37、IS 180；DiT‑L 4.21、DiT‑XL 3.62，显著优于基线（如Euclidean Flow Matching FID 4.28），并且在收敛速度与参数规模上更具效率。

**⚠️ 局限性**

局限性包括：① 仍依赖特征空间近似球面，非球面或不同表示编码器的适用性待验证；② Jacobi正则化的权重需经验调优；③ 对高分辨率或多模态任务的泛化性尚未充分评估。

---

## 493. Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning

**arXiv ID:** 2602.10090 | [PDF](https://arxiv.org/pdf/2602.10090v1)

**作者:** Zhaoyang Wang `[一作]` (University of North Carolina at Chapel Hill), Yuxiong He `[通讯]` (Snowflake)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个全自动化的代码驱动环境生成管线，能够在千级规模下生成数据库后端、统一 MCP 接口、可并行执行的可执行 agentic 环境，并在此环境上进行大规模强化学习。

**💡 创新点**

创新点在于将环境合成拆解为需求、数据库、工具接口与验证四个模块，利用 LLM 自动生成并自我纠错，结合代码与 LLM 的混合评估，打破传统仅依赖任务或 LLM 模拟的局限，实现可扩展、高质量、可复制的执行环境。

**🔧 技术方法**

使用 GPT‑5 / Qwen3 等大语言模型进行场景、任务、数据库模式、接口代码和验证逻辑的生成；SQLite 作为状态后端；MCP 协议统一工具调用；GRPO 强化学习框架；LLM‑as‑Judge 结合代码检查实现奖励；自我纠错循环；基于窗口截断的历史感知训练。

**📊 数据集**

以 100 个场景种子为起点生成 1,000 个环境，衍生 10,000 个任务和 35,062 个工具；在公开基准 BFCLv3、τ² 与 MCP‑Universe 上进行评测；对比数据来自公开任务集与生成的环境。

**📈 对比分析**

与三类基线（Base、Simulator、EnvScaler）在 BFCLv3、τ²、MCP‑Universe 三大评测上进行比较。实验表明，在 BFCLv3 上整体分数从 53.83 提升至 65.94，MCP‑Universe 金融和位置子任务提升显著，整体表现优于所有基线。

**⚠️ 局限性**

局限性包括：仅训练了 526/1000 环境；LLM 生成的环境仍存在 BUG；缺乏自我进化与跨场景任务生成能力；实验仅覆盖 Qwen3 系列模型，模型泛化性待进一步验证。

---

## 494. Beyond a Single Queue: Multi-Level-Multi-Queue as an Effective Design for SSSP problems on GPUs

**arXiv ID:** 2602.10080 | [PDF](https://arxiv.org/pdf/2602.10080v1)

**作者:** Zhengding Hu `[一作]` (University of Science and Technology of China), Guangzhong Sun `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

为GPU上的单源最短路径（SSSP）问题提出一种多层多队列（MLMQ）数据结构，并在此基础上实现了高效、可配置的队列框架。

**💡 创新点**

创新点在于：①把队列拆分到寄存器、共享内存和全局内存三个层次，形成多层队列；②引入缓存式协作机制实现不同层次队列间低成本的数据交换；③提供统一的读写原语，支持模块化队列组合；④利用随机森林学习器实现输入自适应队列配置。

**🔧 技术方法**

技术包括：GPU并行编程（warp/块级持久线程）、基于原子操作的队列读写原语、共享内存与全局内存的多级队列实现（FIFO、Bucket、优先级、Multi-queue等），以及基于特征的机器学习自适配策略。

**📊 数据集**

使用的实验数据集包括：真实世界图（美国高速公路网络、三维网格、回路图、社交网络）、合成R-MAT与幂律图、以及约680个SuiteSparse矩阵。

**📈 对比分析**

与主流GPU实现（ADDS、H-BF、Gunrock）和CPU实现（Boost、Gunrock-CPU）比较，MLMQ在三款GPU（RTX 3080Ti、Tesla A100、RTX 4090）上平均提升1.9×至17.1×；在CPU基线上可达≈200×；在SuiteSparse基准上对91.5%以上的图均优于ADDS，平均提升>2.3×。

**⚠️ 局限性**

局限性在于：目前仅针对SSSP及BFS类遍历算法；仅实现单GPU版本；在极大规模或多GPU分布式环境下的扩展尚未验证。

---

## 495. The Complexity of Proper Equilibrium in Extensive-Form and Polytope Games

**arXiv ID:** 2602.10096 | [PDF](https://arxiv.org/pdf/2602.10096v1)

**作者:** Brian Hu Zhang `[一作]` (Massachusetts Institute of Technology), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20186 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在广义博弈（尤其是扩展形式博弈）中计算正则（proper）均衡的计算复杂性，并证明在扩展形式博弈中此问题与求解 Nash 均衡的复杂性相同，同时揭示在多面体博弈中正则均衡的计算为 NP‑hard。

**💡 创新点**

①首次证明正则均衡在扩展形式博弈中的复杂性与 Nash 均衡等价；②展示传统的 K‑M 置换法在多面体博弈中不可行；③提出一种新的可高效计算“顺序正则”最优策略的线性规划框架，并证明其与正则均衡等价。

**🔧 技术方法**

采用图论与线性规划（CLF）门、条件线性可行性门、最大化/最小化门、以及多面体博弈的序列形式表示，结合固定点理论和多面体几何来构造算法；同时利用 #P‑hard 归约证明 NP‑hard 结果。

**📊 数据集**

无实测数据集，本研究完全基于理论模型与形式化证明；使用的博弈模型为标准的有限扩展形式博弈与多面体博弈。

**📈 对比分析**

通过对比现有的正则均衡计算的已知复杂度（PPA/PPAD 完全）与新提出的多面体博弈中 NP‑hard 结果，表明在扩展形式博弈中正则均衡的求解与 Nash 均衡等价；在多面体博弈中则显著更难；算法复杂度与已知算法相当或更优。

**⚠️ 局限性**

限制：①对正则均衡的完整复杂度在多面体博弈中仍未确定（仅给出 NP‑hard 下限）；②所需的 ε 规模在理论上可能需要双指数大小，实际实现可能不可行；③仅适用于有限博弈，未扩展至连续或无限形式博弈。

---

## 496. CAPID: Context-Aware PII Detection for Question-Answering Systems

**arXiv ID:** 2602.10074 | [PDF](https://arxiv.org/pdf/2602.10074v1)

**作者:** Mariia Ponomarenko `[一作]` (University of Waterloo), Xi He `[通讯]` (University of Waterloo)

**通讯引用:** 4903 | [OpenAlex ID](https://openalex.org/A5038736889)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于上下文的 PII 检测与匿名化框架，利用本地小模型在问答系统中智能保留与遮掩个人信息。

**💡 创新点**

创新点在于：①构建 CAPID 合成数据集，专门标注 PII 的类型和对问题的相关性；②通过 LLM 生成多主题、多类型、多情境的数据，提升模型泛化能力；③在本地模型上实现二分类相关性预测，兼顾隐私与答案质量。

**🔧 技术方法**

主要技术包括：LLM（GPT‑4.1‑mini / GPT‑5）生成合成文本，LoRA 与 4‑bit 量化的 Llama‑3.1‑8B / Llama‑3.2‑3B 微调，JSON 格式标签训练，后期人工校验与一致性检查。

**📊 数据集**

使用 CAPID 合成数据集（约 2,307 条样本，2,107 训练，200 测试）以及 150 条 Reddit 实际问答数据进行评估。

**📈 对比分析**

与 GPT‑4.1‑mini、Microsoft Presidio、以及其他基线相比，Fine‑tuned Llama‑3.1‑8B 的 span F1 由 0.48 提升至 0.96，relevance accuracy 由 0.51 提升至 0.93；在 Reddit 上相关性准确率约 0.80，高于 0.69；低相关性遮掩相比全遮掩在答案实用性上提升 22‑28%。

**⚠️ 局限性**

局限包括：长序列处理能力不足；在完全中性问题中相关性判定模糊；缺乏领域特定的相关性策略；仅使用二元敏感度评分，无法细粒度控制泄露风险。

---

## 497. Learning Agile Quadrotor Flight in the Real World

**arXiv ID:** 2602.10111 | [PDF](https://arxiv.org/pdf/2602.10111v1)

**作者:** Yunfan Ren `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**通讯引用:** 36981 | [OpenAlex ID](https://openalex.org/A5057116316)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过自适应框架，研究人员让四旋翼在没有预先系统辨识的情况下，从保守飞行逐步提升到接近硬件极限速度。

**💡 创新点**

创新点在于结合自适应时间缩放 ATS、在线残差学习与实地锚定短期 BPTT，实现安全探索并快速提升性能。

**🔧 技术方法**

采用可微分仿真、残差神经网络、RASH‑BPTT、ATS 与实时梯度优化等技术。

**📊 数据集**

主要使用实测飞行数据（Agilicious 平台配合运动捕捉系统）和仿真场景；未使用公开数据集。

**📈 对比分析**

与基线、LOFT、Anchor Only、Residual Only 等方法对比，实验表明可在约 100 秒内将速度提升至 7.3 m/s，跟踪误差显著降低，性能优于传统方法。

**⚠️ 局限性**

受限于仅对时间尺度可微分，未实现空间–时间联合优化；在极端扰动或感知误差情况下仍需进一步改进。

---

## 498. Biases in the Blind Spot: Detecting What LLMs Fail to Mention

**arXiv ID:** 2602.10117 | [PDF](https://arxiv.org/pdf/2602.10117v1)

**作者:** Iván Arcuschin `[一作]` (University of Buenos Aires), Oana-Maria Camburu `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种完全自动化的黑盒管道，用于检测大型语言模型（LLM）在决策过程中未在链式思考（CoT）中表述的偏见；

**💡 创新点**

创新点在于：①利用LLM自动生成偏见概念假设；②采用分阶段多级测试与统计早停（O'Brien‑Fleming 和功效分析）来显著减少计算成本；③在偏见检测中同时检验概念是否被模型在CoT中言及，实现对“隐性偏见”的精准定位；

**🔧 技术方法**

核心技术包括：LLM驱动的概念生成与变体构造、输入嵌入聚类、正负变体对照实验、McNemar检验结合Bonferroni校正、统计早停与功效停止、LLM语义检测器判定概念是否在CoT中被引用；

**📊 数据集**

实验使用了三类任务数据：招聘筛选（1,336个简历+职位描述）、贷款批准（2,500个合成贷款申请）和大学录取（1,500个合成学生申请），并将上述管道应用于四个先前偏见研究的实验设置；

**📈 对比分析**

在六个不同供应商的LLM（Gemma 3 12B/27B、Gemini 2.5 Flash、GPT‑4.1、QwQ‑32B、Claude Sonnet 4）上测试，管道成功复现了已知的性别与种族偏见，并自动发现了西班牙语流利度、英语熟练度、书写正式度等新偏见；与手工标注相比，约三分之一的计算量被提前终止，检测到的67%偏见的95%置信区间不含零，显示出良好的统计显著性；

**⚠️ 局限性**

局限性包括：生成的变体可能引入额外混淆，导致部分真实偏见被漏检；管道只能检测到LLM在概念生成阶段提出的假设，缺失未被假设的偏见；保守的Bonferroni校正与早停阈值可能导致对小效应的误判；LLM语义检测器的误报/漏报也会影响最终结果。

---

## 499. Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction

**arXiv ID:** 2602.10101 | [PDF](https://arxiv.org/pdf/2602.10101v1)

**作者:** Sizhe Yang `[一作]` (Shanghai AI Laboratory), Jiangmiao Pang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了Robo3R，一种实时前馈3D重建模型，可直接从RGB图像和机器人状态生成高精度、度量尺度的3D几何，用于机器人操作；

**💡 创新点**

创新点包括：①将机器人状态与视觉特征按逐元素方式融合以提升几何精度；②设计mask点头分离深度、归一化坐标与遮罩，产生细腻点云；③联合预测相对位姿与全局相似变换，并通过关键点+PnP进一步精细化相机外参；④构建大规模4M帧合成数据集Robo3R‑4M，覆盖丰富材质与随机化；

**🔧 技术方法**

核心技术为基于Transformer的Alternating‑Attention骨干、masked point head、relative pose head、similarity transformation head，以及基于PnP的外参估计模块；

**📊 数据集**

使用了Robo3R‑4M合成数据集（约400万帧），包含RGB、深度、语义、机器人状态、相机内外参等多模态标签；

**📈 对比分析**

与VGGT、π^3、MapAnything、DepthAnything3等前馈模型和RealSense D455深度相机比较，在点云误差、相机位姿精度、模仿学习、sim‑to‑real、抓取与碰撞规避等下游任务中，Robo3R 的性能提升超过5‑8倍（如相对位姿误差下降至0.014m、抓取成功率在透明/薄壁物体上达最高）；

**⚠️ 局限性**

局限性在于目前仅支持针孔相机和有限的机器人构型，对鱼眼、全景相机或更复杂装置的适配仍需进一步研究；

---

## 500. Towards Explainable Federated Learning: Understanding the Impact of Differential Privacy

**arXiv ID:** 2602.10100 | [PDF](https://arxiv.org/pdf/2602.10100v1)

**作者:** Júlio Oliveira `[一作]`, Eirini Eleni Tsilopoulou `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于决策树的联邦学习框架 FEXT‑DP，将差分隐私嵌入树的分裂选择中，实现数据隐私与可解释性的双重保障。

**💡 创新点**

创新点在于首次将差分隐私的指数机制应用于联邦决策树的分裂步骤，并系统评估其对模型可解释性（MDI）的影响，填补了该领域的空白。

**🔧 技术方法**

使用联邦学习、决策树、差分隐私（指数机制）以及可解释性指标 MDI；实现基于 Python、scikit‑learn 1.7.dev1+dp 的原型。

**📊 数据集**

采用 Appliance Energy Prediction Data（AEPD）数据集，共 19,735 条家电能耗记录，包括温湿度、位置、时间等特征。

**📈 对比分析**

通过与不使用差分隐私的联邦决策树和基于神经网络的 FedAVG 进行对比，评估 MSE、Pearson 相关系数和 MDI；结果显示 FEXT‑DP 在 MSE 与相关性上优于 FedAVG，差分隐私使 MSE 略有上升但提升了隐私保证。

**⚠️ 局限性**

局限性包括差分隐私降低模型可解释性、仅针对水平联邦场景、未引入剪枝或客户端选择机制，且实验仅在 AEPD 上完成，缺乏对训练时延、内存占用和网络流量等实战指标的评估。

---

## 501. UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking

**arXiv ID:** 2602.10093 | [PDF](https://arxiv.org/pdf/2602.10093v1)

**作者:** Baijun Chen `[一作]` (Nanjing University), Yao Mu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 12419 | [OpenAlex ID](https://openalex.org/A5008178136)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UniVTAC 平台，实现统一的 visuo‑tactile 数据生成、表示学习与基准评测；

**💡 创新点**

1) 支持三种主流 visuo‑tactile 传感器的可扩展模拟；2) 通过多路径监督（形状、接触、姿态）预训练的 UniVTAC Encoder；3) 基于 TacEx 的八任务基准，强调触觉驱动的决策；

**🔧 技术方法**

物理基础模拟（TacEx/IPC/FEM/MPM）、可视化渲染、深度学习 Encoder（ResNet‑18 + 多头解码器）、Transformer‑based ACT 策略、对比学习基准 VITaL、强化学习/专家驱动数据合成；

**📊 数据集**

使用模拟生成的 205,826 样本（14,000+每种形状），以及 8 个 visuo‑tactile 任务的自动化合成数据；真实数据用于 3 个物理任务的 150 条演示；

**📈 对比分析**

在 UniVTAC Benchmark 上，ACT+UniVTAC 平均成功率 48.0%，比 ACT（30.9%）提升 17.1%，比 VITaL（40.5%）更优；在真实机器人上，平均成功率从 43.3% 提升到 68.3%（+25%）；

**⚠️ 局限性**

局限：依赖高保真仿真，仍需解决仿真与真实差距；仅覆盖三种传感器；基准任务仍相对简化，缺乏动态与多物体场景；

---

## 502. CODE-SHARP: Continuous Open-ended Discovery and Evolution of Skills as Hierarchical Reward Programs

**arXiv ID:** 2602.10085 | [PDF](https://arxiv.org/pdf/2602.10085v1)

**作者:** Richard Bornemann `[一作]` (Imperial College London), Antoine Cully `[通讯]` (Imperial College London)

**通讯引用:** 2351 | [OpenAlex ID](https://openalex.org/A5011747084)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的框架，称为CODE-SHARP，旨在通过利用基础模型（FM）实现开放式技能发现和演化，自动生成和优化奖励函数。

**💡 创新点**

创新点在于通过将技能定义为层次奖励程序（SHARP），实现了技能的开放式发现和自动化奖励设计，克服了手动设计奖励函数的局限性。

**🔧 技术方法**

使用了基础模型（FM）进行技能提案生成、实施和评估，以及技能变异生成和实施。

**📊 数据集**

在Craftax环境中进行评估，该环境结合了Minecraft和NetHack的机制，提供了丰富的开放式任务空间。

**📈 对比分析**

与预训练代理和任务特定专家策略相比，使用CODE-SHARP生成的技能的目标条件代理在复杂的长期目标上表现出色，平均超出134%。

**⚠️ 局限性**

限制在于需要环境以代码形式指定，这限制了其在现实世界场景（如机器人技术）中的适用性。

---

## 503. SAGE: Scalable Agentic 3D Scene Generation for Embodied AI

**arXiv ID:** 2602.10116 | [PDF](https://arxiv.org/pdf/2602.10116v1)

**作者:** Hongchi Xia `[一作]` (NVIDIA), Fangyin Wei `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于代理的框架 SAGE，能够从开放词汇文本提示自动生成可直接用于机器人学习的 3D 室内场景，支持多级数据增强和动作示例合成；

**💡 创新点**

创新点在于将 LLM 代理与多种生成器（场景初始化、资产摆放、移动、删除）和视觉/物理评判器通过 Model Context Protocol (MCP) 进行自适应协同，闭环反馈实现物理稳定与语义合理的自我修正；

**🔧 技术方法**

核心技术包括大语言模型（gpt-oss-120b）、视觉语言模型（Qwen3-VL-30B）、3D 对象合成（TRELLIS）、纹理合成（MatFuse）、物理验证（Isaac Sim）以及多级场景/对象增广；

**📊 数据集**

使用公开的 3D 资产库和自研 SAGE‑10k 数据集（10k 场景、50 种房间类型、56.5K 独特 3D 对象）进行评估；

**📈 对比分析**

与 Holodeck、SceneWeaver 等基线对比，SAGE 在视觉质量、物理稳定率、碰撞率等指标上显著优于对手，且基于其生成的数据训练的策略在 Pick‑and‑Place 与 Mobile Manipulation 任务中实现了与基准规划器相近的成功率，展示了更快的收敛和更好的泛化；

**⚠️ 局限性**

局限性包括仅针对室内刚体场景，未涵盖户外、柔性或关节对象，动作生成仅限于 pick‑place 与导航，缺乏在线强化学习和真实机器人闭环验证等。

---

## 504. Decoupled MPPI-Based Multi-Arm Motion Planning

**arXiv ID:** 2602.10114 | [PDF](https://arxiv.org/pdf/2602.10114v1)

**作者:** Dan Evron `[一作]` (Ben-Gurion University of the Negev), Ronen I. Brafman `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 5587 | [OpenAlex ID](https://openalex.org/A5007883545)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出一种名为 MR-STORM 的分布式多臂运动规划框架，将 STORM（基于 GPU 的采样式 MPC 规划器）扩展到多机器人环境，支持动态障碍物和动态优先级，以实现低延迟、可扩展的协同规划。

**💡 创新点**

创新点包括：
- 将 STORM 的代价函数改为同时考虑静态与动态障碍物；
- 通过共享稀疏的球体轨迹信息，使每个臂能够实时预测其他臂的位置并将其作为动态障碍物纳入规划；
- 引入基于距离的动态优先级机制，避免臂与臂之间的死锁与碰撞；
- 在实现上保持 GPU 并行评估的高效性，实现与单臂 STORM 相当的控制频率。

**🔧 技术方法**

主要技术包括：
- 采样式模型预测积分（MPPI）/STORM；
- GPU 并行采样与碰撞检查；
- 目标点与动态障碍物的球体近似；
- 软约束（关节/速度限制、碰撞、操纵度、正则化等）及其重要性加权更新；
- 基于距离比值的动态优先级因子。

**📊 数据集**

实验数据集：
- 使用 NVIDIA Isaac Sim 进行 120 个仿真环境，涵盖 4 种任务（目标到达易/难、跟踪、箱子装载），每种任务 30 个实例，每个实例 5 个难度级别；
- 机器人采用 UR5e 6-DOF 力臂，4 支臂同时操作；
- 仿真硬件为 1 台 RTX‑4090 GPU + 8 核 CPU。

**📈 对比分析**

比较方法：
- 与 STORM 的集中式版本（SC）、全局规划器（CC）、单臂 STORM 处理动态障碍物（SD）以及无优先级的 MR-STORM ablation（MRS‑）对比；
- 评估指标包括任务完成度、碰撞次数、控制频率。结果显示：MR-STORM（400,5）在所有任务上均优于集中式或无通信方案；MRS‑（无优先级）性能下降明显；MRS‑（1,400）与 MRS‑（400,5）差异不大但控制频率更高；集中式规划器在高交互情景下失效。整体来看，MR-STORM 在任务完成度上领先 15–30%，碰撞率降低 70% 以上。

**⚠️ 局限性**

局限性：
- 仅在仿真环境验证，缺乏真实机器人实验；
- 规划器为局部 MPC，无法提供完整性或全局最优保证；
- 障碍物仅以球体近似，无法处理复杂形状；
- 需要高性能 GPU 进行并行评估，实际部署需进一步验证多机器人的通信延迟与同步问题。

---

## 505. EgoHumanoid: Unlocking In-the-Wild Loco-Manipulation with Robot-Free Egocentric Demonstration

**arXiv ID:** 2602.10106 | [PDF](https://arxiv.org/pdf/2602.10106v1)

**作者:** Modi Shi `[一作]` (Beihang University), Li Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 29726 | [OpenAlex ID](https://openalex.org/A5100379155)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用人类正视摄像的演示与机器人远程操作数据，共同训练人机视觉-语言-动作（VLA）策略，实现类人机器人全身运动与操作的迁移与泛化。

**💡 创新点**

首次通过视角对齐与动作统一空间，解决人类与类人机器人在形态、视点与运动动力学上的差异，实现跨体型协同学习，并展示在人机共训练中大幅提升泛化性能。

**🔧 技术方法**

采用基于MoGe的深度重投影、Latent Diffusion Inpainting的视角对齐；动作对齐使用6-DoF增量末端执行器姿态和离散步态指令；VLA模型为π_0.5（视觉-语言-动作）。

**📊 数据集**

使用由可穿戴PICO VR系统采集的自制人类正视演示数据（约300条），以及通过VR遥控收集的类人机器人远程操作演示数据（约100条）。

**📈 对比分析**

在四个室内外全身运动+操作任务上与仅用机器人数据训练的基线比较，类人机器人仅用机器人数据得分约59%/31%，共训练得分约78%/82%，提升超过50%。

**⚠️ 局限性**

仍存在姿态不确定性、对细粒度旋转控制有限、需要大量数据、以及缺乏完整的全身动力学建模等限制。

---

## 506. DexImit: Learning Bimanual Dexterous Manipulation from Monocular Human Videos

**arXiv ID:** 2602.10105 | [PDF](https://arxiv.org/pdf/2602.10105v1)

**作者:** Juncheng Mu `[一作]` (Shanghai AI Laboratory), Jiangmiao Pang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本工作提出了一个自动化框架，能够仅凭单目人类操纵视频（包括网络公开视频和文本生成的视频）生成可用于双臂灵巧操作的机器人数据，完成从 4D 手物交互重建、子任务拆分与双臂调度、基于力闭包的抓取与运动规划以及完整的数据增强，最终实现零射击在真实机器人上的部署。

**💡 创新点**

创新点主要体现在四个方面：①不依赖深度或相机信息即可完成近米尺度的手物 4D 重建；②提出以动作为中心的子任务调度算法，支持任意时延、并行和非并行双手操作；③基于力闭包约束的抓取生成与关键帧运动规划，显著提升生成轨迹的物理可行性；④全流程数据增强（物体姿态/尺度、摄像机位姿、点云噪声）使得训练出的策略可零射击真实环境。

**🔧 技术方法**

技术手段包括：Vision‑Language 模型（Qwen3‑VL）用于视频理解与子任务标注；Grounded Sam2 与 SAM3D 实现手物分割与 3D 生成；SpatialTracker v2 与 FoundationPose++ 等深度与 6D 关节估计；MANO 与 force‑closure 优化的抓取合成；基于关键帧的运动规划；以及 3D Diffusion Policy 进行策略学习。

**📊 数据集**

使用的数据来源涵盖网络公开的真实人类操纵视频、基于文本的生成视频（Wan2.2、Veo3）、自定义拍摄视频以及手工校正视频，并在标准双臂任务（Put Cup、Grapefruit、Fruits、Pour、Pot、Stack Cups）以及长时程任务（Make‑Beverage、Cook、Cut‑Apple、Stack‑Cup）上进行评估。

**📈 对比分析**

与两种基线（单臂抓取方案 RigVid 与基于视频轨迹的 RL 方法 DexMan）对比，所提出的方法在所有短时任务中成功率最高，在长时程及精细操作（如 Stack‑Cups）上也能保持 50% 以上的成功率；在真实机器人上通过数据增强实现的零射击部署成功率高于 90%，实验中消除尺度或视觉增强会显著降低表现。

**⚠️ 局限性**

局限性包括：由于由多模块串联完成，错误可能在管道中传播，导致部分数据不可用；目前无法处理复杂的手内操作；对可变形与关节物体的支持有限，未来需引入更强大的 3D 生成与动力学建模。

---

## 507. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model

**arXiv ID:** 2602.10098 | [PDF](https://arxiv.org/pdf/2602.10098v1)

**作者:** Jingwen Sun `[一作]` (University of Science and Technology of China), Zhibo Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 11028 | [OpenAlex ID](https://openalex.org/A5079572598)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的预训练框架VLA-JEPA，通过在无标签人类视频与带标签机器人演示数据上学习隐式动作表征，实现从视频中提取与控制相关的状态转移语义；

**💡 创新点**

创新点在于：①消除传统隐式动作预训练中的像素级信息泄漏，采用未来潜在状态对齐；②通过两阶段（预训练+微调）简化流程；③将人类视频与机器人演示联合预训练，提升鲁棒性与泛化；

**🔧 技术方法**

主要技术包括：基于JEPA的潜在空间对齐损失、VLM（如Qwen3-VL）骨干网络、时间因果注意力、流匹配动作头、世界模型预测；

**📊 数据集**

使用的数据集包括人类动作数据集Something‑Something‑v2、机器人演示数据集Droid、LIBERO、LIBERO‑Plus、SimplerEnv（Fractal、BridgeV2）以及真实Frank A Research 3抓取任务的100条演示；

**📈 对比分析**

与多种VLA基线（LAPA、UniVLA、villa‑X、OpenVLA‑OFT、π0等）对比，VLA‑JEPA在LIBERO、SimplerEnv、LIBERO‑Plus以及真实机器人实验中均实现或接近最优成功率，尤其在多扰动情景下表现出更高的鲁棒性；

**⚠️ 局限性**

局限性包括：对人类视频的依赖仍存在有限的物理动态学习，鲁棒性在某些任务如文本指令执行细节上略逊；模型规模与计算成本较高；未来需进一步验证跨域（如更多物理仿真/真实环境）的一致性。

---

## 508. Step-resolved data attribution for looped transformers

**arXiv ID:** 2602.10097 | [PDF](https://arxiv.org/pdf/2602.10097v1)

**作者:** Georgios Kaissis `[一作]` (Hasso Plattner Institute for Digital Engineering), Eleni Triantafillou `[通讯]` (Google DeepMind)

**通讯引用:** 1505 | [OpenAlex ID](https://openalex.org/A5073728630)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Step‑Decomposed Influence（SDI），一种将训练样本对循环Transformer内部各步骤的影响力分解并可量化的技术。

**💡 创新点**

创新点在于：①把TracIn迁移到循环结构并按步骤拆分；②引入TensorSketch做实时稀疏投影，避免存储完整梯度；③通过SDI可获得内部推理过程的时间序列影响信息。

**🔧 技术方法**

技术包括：循环Transformer（weight‑tied block）、TracIn、CountSketch、TensorSketch、BPTT中的实时Sketch、梯度投影、批量训练与梯度累积。

**📊 数据集**

使用的数据集有：算法推理任务的Parity、Sudoku、以及大规模聊天模型Nanochat的混合语料（GSM8K等）。

**📈 对比分析**

与全梯度基线对比：SDI在误差<5%范围内、内存使用降低≈1000×、计算开销≈2.5s/检查点，实验在135M GPT模型、32层循环、128长度上验证；在Parity、Sudoku和Nanochat实验中，SDI提供了与全梯度一致的影响轨迹，并揭示了循环深度与推理效果的关系。

**⚠️ 局限性**

局限性包括：①对梯度下降的假设（TracIn不考虑动量/自适应优化）；②需要完整BPTT才能获得完整轨迹，长循环下资源仍昂贵；③步骤索引在随机循环/截断训练下不完全可比；④仅测量梯度相似性，未证明实际删改样本会改变行为。

---

## 509. Causality in Video Diffusers is Separable from Denoising

**arXiv ID:** 2602.10095 | [PDF](https://arxiv.org/pdf/2602.10095v1)

**作者:** Xingjian Bai `[一作]` (Massachusetts Institute of Technology), Zongze Wu `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种解耦的因果视频扩散架构（SCD），将一次性帧间因果推理与多步帧内去噪分离，以提升生成效率和实时性。

**💡 创新点**

发现并利用了早期层跨去噪步骤的特征冗余和深层跨帧注意力稀疏性，提出一次性编码器+轻量化解码器的分离设计，实现计算与内存的显著节省。

**🔧 技术方法**

使用因果 Transformer 编码器、轻量化扩散解码器、Teacher‑Forcing/Diffusion‑Forcing 训练、KV 缓存、噪声注入对抗以及自注意力机制。

**📊 数据集**

在 TECO‑Minecraft、UCF‑101、RealEstate10K 等小规模视频数据集上预训练，并在 VBench、RealEstate10K 进行文本到视频评估。

**📈 对比分析**

与全因果注意力的 AR 扩散基线（Causal‑DiT、Self‑Forcing 等）及非因果模型对比，SCD 在生成质量（LPIPS/SSIM/PSNR/FVD、VBench 分数）与实时性（FPS/每帧秒数）上匹配或超越基线，速度提升 2–3 倍，内存降低。

**⚠️ 局限性**

假设跨去噪步骤的冗余和深层跨帧注意力完全可忽略，实际在后期去噪和深层仍有少量残留，导致高分辨率下与全因果基线略有质量差距。

---

## 510. 4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere

**arXiv ID:** 2602.10094 | [PDF](https://arxiv.org/pdf/2602.10094v1)

**作者:** Yihang Luo `[一作]` (Nanyang Technological University), Chen Change Loy `[通讯]` (Nanyang Technological University)

**通讯引用:** 78772 | [OpenAlex ID](https://openalex.org/A5005626854)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种统一的 feed‑forward Transformer 框架，能够一次性编码整段单目视频为 4D 表示，并在任意时刻对任意视角进行几何与运动的查询与解码。

**💡 创新点**

创新点在于：① encode‑once、query‑anywhere、anytime 的查询范式；② 将 4D 归一化为基础几何 + 时间相关位移的最小化分解表示；③ 结合自注意力与跨时间注意力的轻量级运动解码器，实现对任意时间的稠密 3D 运动预测。

**🔧 技术方法**

使用 Vision Transformer（ViT）作为编码器，加入时间令牌；几何头采用 DPT 风格的深度、射线与相机参数回归；运动头采用自注意力、跨注意力和 AdaLN 的 Transformer 结构；整体采用端到端联合监督，并用不确定性加权与梯度正则化提升稳定性。

**📊 数据集**

在训练时使用 PointOdyssey、Dynamic Replica、Kubric、Waymo、DL3DV、ScanNet++、MVS‑Synth 等多样化公开数据集，涵盖静态与动态、真实与合成场景。

**📈 对比分析**

与现有 3D/4D 复原方法（如 ST4RTrack、Any4D、V‑DPM、TraceAnything 等）以及传统的 SfM/MVS/深度学习方法进行对比，实验显示在相机姿态、深度估计、多视角重建、稠密 3D 跟踪及整体 4D 任务上均获得更低的误差、更高的 APD/EPE，并在 TUM‑dynamics、Waymo、Sintel、Bonn 等基准上刷新了多项最佳指标。

**⚠️ 局限性**

局限性包括：① 需要大量标注视频数据与显著 GPU 资源进行训练；② 对极端大范围运动或长时间遮挡的鲁棒性尚未完全验证；③ 采用基于第一帧的世界坐标系，可能在跨序列迁移时产生尺度/偏置问题。

---

