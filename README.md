# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-12 | 今日论文总数: 328

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Understanding LLM-Driven Test Oracle Generation

**arXiv ID:** 2601.05542 | [PDF](https://arxiv.org/pdf/2601.05542v1)

**作者:** Adam Bodicoat `[一作]` (University of Auckland), Valerio Terragni `[通讯]` (University of Auckland)

**通讯引用:** 710 | [OpenAlex ID](https://openalex.org/A5068101658)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了使用大型语言模型（LLM）为单元测试生成断言（oracle）的效果，并通过不同提示（prompt）策略和上下文输入来评估其准确性与可编译率。

**💡 创新点**

创新点在于首次系统评估了提示类型（零样本、少样本、链式思维、树式思维）与上下文层级（仅测试前缀、加方法、加类）对LLM生成断言的影响，并使用专门设计的无泄露数据集。

**🔧 技术方法**

采用的大型语言模型包括代码专用模型（S）和通用模型（G），提示方式包括零样本、少样本、CoT、ToT，评估指标包括准确率、错误检测率、正确通过率和编译率。

**📊 数据集**

实验基于GitHub Recent Bugs（GHRB）基准，随机抽取36个Java bug案例，并在每个bug上执行多达5次生成。

**📈 对比分析**

通过对2,160次实验的统计，发现包含完整类上下文（CUT）的提示和零样本/少样本策略能获得最高的准确率（约54–55%）和编译率（约70%），相较于CoT/ToT下降约25%；模型差异影响相对较小。

**⚠️ 局限性**

局限性包括样本量有限、仅评估已知错误前缀、未考虑误报率、仅测试两种模型、提示配置受限，且结果可能不适用于更广泛的代码或真实环境。

---

## 2. SPAM: Style Prompt Adherence Metric for Prompt-based TTS

**arXiv ID:** 2601.05554 | [PDF](https://arxiv.org/pdf/2601.05554v1)

**作者:** Chanhee Cho `[一作]` (Chung-Ang University), Bugeun Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 69 | [OpenAlex ID](https://openalex.org/A5077260647)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了Style Prompt Adherence Metric（SPAM），用于自动评估基于提示的 TTS 合成语音是否遵循给定的风格提示。

**💡 创新点**

创新点包括：① 将声学属性（音高、语速、能量、说话人、文本）显式分解并通过并行分支融合；② 使用监督对比（SupCon）损失，充分利用批次中的多正样本；③ 采用 Llama‑3.1 作为 prompt 编码器，提升对细粒度文本差异的辨识。

**🔧 技术方法**

核心技术包括 CLAP 框架、WavLM、X‑Vector、G2P、Llama‑3.1、监督对比损失、Auxiliary 预测头（速度、能量、音高）。

**📊 数据集**

训练数据来自 TextrolSpeech 与 SpeechCraft 合并集；测试使用 TextrolSpeech 与 LibriTTS‑P；对 5 种不同 TTS 生成的 500 句合成语音进行评估。

**📈 对比分析**

通过与 RA‑CLAP、CLAP 以及人工 MOS 的相关性（LCC、SRCC、KTAU）和 faithfulness（AR、paired t‑test）实验比较，SPAM 在相关性、正负样本区分度和对正样本的保真度上均优于对比基线，表现出更高的可解释性和稳定性。

**⚠️ 局限性**

局限性：对极端声学属性或非常短、复杂的提示可能仍存在误判；目前仅在单一语言上验证，跨语言适用性未充分测试；模型对极少量数据的泛化仍需进一步评估。

---

## 3. On the Effect of Cheating in Chess

**arXiv ID:** 2601.05386 | [PDF](https://arxiv.org/pdf/2601.05386v1)

**作者:** Daniel Keren `[一作]` (University of Haifa), Daniel Keren `[通讯]` (University of Haifa)

**通讯引用:** 1300 | [OpenAlex ID](https://openalex.org/A5113439295)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究在棋局中有限次数的作弊（使用强力棋程软件提供建议）对引擎对弈结果的提升效果，量化了不同作弊次数的平均得分。

**💡 创新点**

创新点在于：①首次在受限预算k的条件下量化作弊收益；②利用日志数据与离线模拟实现快速超参数（阈值）优化，避免直接跑棋引擎。

**🔧 技术方法**

主要技术包括：基于 Stockfish 的 WDL 评估、单调回归/单调神经网络调校 WDL、贝叶斯优化寻找阈值、最大 Δ 预测模型（线性回归、随机森林、MLP）以及基于离线数据的“无引擎”阈值模拟。

**📊 数据集**

使用的数据集：约 50,000 场无干预引擎对弈日志、50,000 场随机单次干预日志、以及额外的 200,000 次模拟运行，用于阈值优化。

**📈 对比分析**

与无干预或随机干预对比，单次随机干预平均得分从 0.51 提升到约 0.69；最优阈值策略（k=4）在实际引擎对弈中平均得分达到 0.91，远高于 0.51 的基准。

**⚠️ 局限性**

局限性包括：①只在引擎自对弈上验证，尚未直接评估在真人对局中的效果；②干预策略假设“助手”始终能在给定局面选出最优强制走法，实际人类使用时受限；③离线模拟对局面分布和时间约束的逼真度有限，可能导致收益估计偏差。

---

## 4. Secure Communication via Modulation Order Confusion

**arXiv ID:** 2601.05292 | [PDF](https://arxiv.org/pdf/2601.05292v1)

**作者:** Jingyi Wang `[一作]`, Fanggang Wang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于调制阶数混淆（MOC）的安全通信方法，在单天线和多天线系统中通过符号随机映射或符号时延多样化、泰勒级展开和星座路径设计等方案，误导窃听者的调制分类器。

**💡 创新点**

创新点在于将调制阶数混淆作为防御手段，不依赖对方知识，可灵活将原始调制伪装为更高或更低阶调制，并在多天线和RIS系统中实现无需接收机改造的混淆。

**🔧 技术方法**

采用符号随机映射、符号时延多样化、泰勒级展开、星座路径设计、动态规划解码、凸优化（CVX）、盲源分离、深度学习分类器对比等技术。

**📊 数据集**

使用自建数据集，包含10种调制（如2FSK、QPSK、16QAM等），并在单天线模拟中通过随机相位旋转等方式模拟信道。

**📈 对比分析**

通过仿真与传统调制分类器（VGG、SCGNet、WSMF、ChainNet以及专家知识分类器）对比，显示MOC能显著降低分类准确率，同时在单天线系统BER在低至中等SNR下介于原始与目标调制之间，且多天线方案BER随天线数增大而下降。

**⚠️ 局限性**

局限包括：单天线方案需Bob重新设计接收机；高阶混淆可能导致频谱效率下降；对SNR变化敏感；若窃听者获知映射或持续时间分布则安全性降低。

---

## 5. DafnyPro: LLM-Assisted Automated Verification for Dafny Programs

**arXiv ID:** 2601.05385 | [PDF](https://arxiv.org/pdf/2601.05385v1)

**作者:** Debangshu Banerjee `[一作]` (University of Illinois Urbana-Champaign), Stefan Zetzsche `[通讯]` (Amazon Web Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DafnyPro，一种在推理时为Dafny程序自动生成验证注释的框架。

**💡 创新点**

创新点在于三大机制：diff‑checker防止代码逻辑被篡改；pruner剔除冗余不必要的循环不变式；hint‑augmentation检索并应用问题无关的证明策略。

**🔧 技术方法**

主要技术包括使用Dafny解析器进行语义diff检查、迭代式基于验证器反馈的注释生成与裁剪、以及通过LLM检索已有证明策略并注入提示。

**📊 数据集**

实验使用了Clover、MBPP‑Dafny、HumanEval‑Dafny和大规模评测基准DafnyBench四个数据集。

**📈 对比分析**

与基线（仅使用验证器反馈迭代）和以往工作对比，Claude 3.5/3.7 Sonnet在DafnyBench上分别提升约10%（从约70%提升至约86%），同时在其他数据集提升3–14%，证明了显著的性能提升。

**⚠️ 局限性**

主要局限包括：仍依赖大型LLM的推理成本；问题无关证明策略仍需人工挑选且仅覆盖有限类型；对复杂嵌套或多循环程序的裁剪策略尚不完善。

---

## 6. Multi-turn Jailbreaking Attack in Multi-Modal Large Language Models

**arXiv ID:** 2601.05339 | [PDF](https://arxiv.org/pdf/2601.05339v1)

**作者:** Badhan Chandra Das `[一作]` (Florida International University), Yanzhao Wu `[通讯]` (Florida International University)

**通讯引用:** 1375 | [OpenAlex ID](https://openalex.org/A5060093535)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为 MJAD-MLLM 的框架，包含多轮 jailbreaking 攻击方法和 Fragment‑Guard 防御机制，用于评估和缓解多模态大型语言模型（MLLM）的安全风险。

**💡 创新点**

创新点：①首次系统化研究多轮 jailbreaking 攻击，揭示对 MLLM 的隐蔽性与逐步诱导优势；②设计 Fragment‑Guard，通过分片检测与多 LLM 交叉评估实现更细粒度的毒性识别；③提出了统一的多 LLM 评估方法，避免单一评估器的偏差。

**🔧 技术方法**

技术手段：多轮对话触发式攻击；响应分片（token 片段）技术；利用 OpenAI 的 o1、Google Gemini‑2.5‑Flash、Meta LLaMA‑3 等大型模型进行毒性评分；阈值抑制与拒绝机制。

**📊 数据集**

数据集：MM‑SafetyBench 基准（13 类禁止场景），包含 Stable Diffusion 生成的图像与嵌入的关键信息，结合多模态查询与回答。

**📈 对比分析**

对比方法：与 MM‑SafetyBench、Shuffle‑Inconsistency、FigStep 等基线在 LLaVA‑7B、GPT‑4o 上对比；实验显示多轮攻击的 ASR 在 Turn‑3 处分别达到 91.5%（开源）和 77.3%（闭源），明显高于基线；FragGuard 在 Turn‑3 处将 ASR 降低至约 1%（开源）和 0.6%（闭源），并显著提升拒绝率（RR）。

**⚠️ 局限性**

局限性：①仅对文本输出的 MLLM 有效，无法直接处理纯图像或音频生成；②评估依赖外部 LLM 的毒性评分，可能受模型自身偏差影响；③攻击与防御均为手工设计，缺乏自动化生成与大规模多样化测试。

---

## 7. Imitation Learning for Combinatorial Optimisation under Uncertainty

**arXiv ID:** 2601.05383 | [PDF](https://arxiv.org/pdf/2601.05383v1)

**作者:** Prakash Gawas `[一作]` (Polytechnique Montreal), Louis-Martin Rousseau `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了针对组合优化下不确定性的模仿学习专家分类法，并在此框架下设计了一种支持多专家查询与交互的通用DAgger算法，随后在动态医师‑病人分配问题上进行实验验证。

**💡 创新点**

创新点在于系统化地从不确定性处理、最优性水平和交互模式三维度定义专家类型，提供统一的理论框架，并将该框架与改进的DAgger算法结合，探索不同专家组合对学习效果的影响。

**🔧 技术方法**

采用了模仿学习中的行为克隆与DAgger迭代、Gurobi求解器生成专家示例、MLP预测模型以及聚合策略（如多场景汇聚）等技术。

**📊 数据集**

实验数据为基于统计分布（患者到达时间、优先级、持续时长等）的合成医师‑病人分配实例，共计1000个测试案例。

**📈 对比分析**

与贪心、两阶段随机最优、全信息最优基线相比，使用两阶段随机专家训练的模型在总成本、优先级1病人拒绝率等指标上优于确定性和全信息专家；交互式DAgger进一步提升性能，且在仅使用更少示例时即可逼近最优基线。

**⚠️ 局限性**

局限性包括：专家求解时需耗时较长（尤其是两阶段或多阶段随机模型），导致训练成本高；实验仅在合成数据上验证，缺乏真实场景的实证；以及对模型超参数（迭代次数、场景数、时间阈值等）敏感，需进一步研究鲁棒性与自适应性。

---

## 8. Explainable AI: Learning from the Learners

**arXiv ID:** 2601.05525 | [PDF](https://arxiv.org/pdf/2601.05525v1)

**作者:** Ricardo Vinuesa `[一作]` (University of Michigan), Gianmarco Mengaldo `[通讯]` (National University of Singapore)

**通讯引用:** 2154 | [OpenAlex ID](https://openalex.org/A5091468612)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了将可解释人工智能（XAI）与因果推理相结合，形成统一框架，用于科学发现、工程优化和系统认证；

**💡 创新点**

创新点在于强调XAI可以“学习学习者”，通过因果机制提取、可视化并验证模型内部逻辑，从而实现对黑盒模型的可解释性、可推广性与可审计性；

**🔧 技术方法**

使用的技术包括SHAP、集成梯度、符号回归（Genetic Programming、PySR、SINDy）、自编码器（AE、β‑VAE）、基于Transformer的时序预测、潜在扩散模型、以及大语言模型驱动的agentic‑AI；

**📊 数据集**

以流体力学（湍流控制、气动设计）为例的数据集，涵盖不同几何形状、雷诺数和马赫数等条件；

**📈 对比分析**

文中没有给出具体数值对比，而是通过案例说明XAI方法在揭示物理机制、引导设计决策和检测异常方面的优势，暗示其在性能和可靠性上的提升；

**⚠️ 局限性**

局限性包括：解释方法的可靠性与可验证性不足、计算开销大、对高维潜在空间的因果推理仍不成熟，以及解释结果可能因人类主观偏好而产生误导。

---

## 9. Knowledge-Driven Multi-Turn Jailbreaking on Large Language Models

**arXiv ID:** 2601.05445 | [PDF](https://arxiv.org/pdf/2601.05445v1)

**作者:** Songze Li `[一作]` (Southeast University), Zhihui Fu `[通讯]` (OPPO Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Mastermind 框架，能够在多轮对话中自动化地执行 jailbreak 攻击，突破 LLM 的安全防护。

**💡 创新点**

核心创新在于：① 用知识驱动的层次化多智能体架构，将高层规划与低层执行分离；② 通过自监督的策略蒸馏构建可迁移的攻击知识库；③ 在策略空间而非文本空间进行遗传式模糊搜索，实现高效的策略组合优化。

**🔧 技术方法**

主要技术包括：层次化规划（Planner/Executor/Controller）、自我反思与知识蒸馏、策略级进化模糊（选择、交叉、变异）、LLM-as-Judge 评估、以及安全防御适配（Self‑Reminder、SmoothLLM、Llama Guard）。

**📊 数据集**

使用 HarmBench、StrongReject 两个公开危害性评测集作为实验基准，并通过 200 条生成的恶意查询构成知识积累的辅助集；实验还在多种开源与商用 LLM（如 GPT‑4o、Claude 3.7 Sonnet、GPT‑5 等）上进行评估。

**📈 对比分析**

与 Crescendo、ActorAttack、X‑Teaming、Siren 等现有多轮 jailbreak 方法对比，Mastermind 在 HarmBench 上平均 ASR 达到 87%，在 StrongReject 上 91%；在多种防御（Self‑Reminder、SmoothLLM、Llama Guard）下仍保持 80%+ 的成功率，并取得最高的 Harmfulness Rating，显示出卓越的攻击效果和鲁棒性。

**⚠️ 局限性**

局限性包括：① 依赖大量计算资源进行策略模糊搜索；② 主要针对文本对话，尚未覆盖多模态或更复杂交互；③ 受限于黑盒设置，仍可能遇到模型的未知安全策略；④ 生成的攻击策略需人工审查，存在伦理与滥用风险。

---

## 10. Rethinking Basis Path Testing: Mixed Integer Programming Approach for Test Path Set Generation

**arXiv ID:** 2601.05463 | [PDF](https://arxiv.org/pdf/2601.05463v1)

**作者:** Chao Wei `[一作]` (Hubei University of Technology), Ting Cai `[通讯]` (Hubei University of Technology)

**通讯引用:** 615 | [OpenAlex ID](https://openalex.org/A5062162392)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将基路径生成任务从传统的贪心图遍历转变为一个声明式的优化问题，使用混合整数规划（MIP）框架生成全局最优且结构简洁的基路径集合。

**💡 创新点**

创新点在于（1）提出了全局最优的Holistic MIP模型并引入子环消除的网络流约束；（2）设计了可扩展的增量式MIP策略，并通过“新颖度惩罚”多目标函数有效避免贪婪陷阱，保证在复杂图中仍能完整生成基路径；（3）将结构优化与后续语义测试相结合，提供可直接用于后续可执行路径生成的高质量路径基。

**🔧 技术方法**

主要技术包括：混合整数规划（MIP）、网络流与子环消除约束、路径长度与新颖度双目标优化、增量式路径生成算法、与基线BFS对比实验。

**📊 数据集**

使用了两类数据集：真实代码数据集（50个Python函数，CC 1~8）和合成大规模数据集（随机CFG，CC分别为10、50、100）。

**📈 对比分析**

与传统BFS基线以及全局Holistic MIP对比；实验结果显示：在真实代码上所有MIP方法成功率100%，平均耗时约0.03s；在合成数据上，增量式MIP2（新颖度驱动）在所有复杂度下保持100%成功率，且计算时间最小（CC=100约17.7s），显著优于BFS（成功率低于20%）和增量式MIP1（贪婪陷阱导致成功率降至14.7%）。

**⚠️ 局限性**

局限性在于仍未解决语义不可执行路径（infeasible path）问题；MIP求解在极大规模CFG上会遇到时间与内存瓶颈；多目标优化的参数（惩罚系数）需经验调优，未给出理论证明其最优性。

---

## 11. MOSAIC-GS: Monocular Scene Reconstruction via Advanced Initialization for Complex Dynamic Environments

**arXiv ID:** 2601.05368 | [PDF](https://arxiv.org/pdf/2601.05368v1)

**作者:** Svitlana Morkva `[一作]` (ETH Zurich), Vaishakh Patil `[通讯]` (ETH Zurich)

**通讯引用:** 205 | [OpenAlex ID](https://openalex.org/A5076239754)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

MOSAIC-GS通过多阶段预处理和Poly‑Fourier曲线初始化，实现从单目视频高效重建动态场景。

**💡 创新点**

引入基于几何与刚性约束的动态初始化、静态‑动态高斯拆分以及时间变Poly‑Fourier运动编码，显著提升训练速度与重建精度。

**🔧 技术方法**

使用3D高斯展开、光流与极线误差动态检测、SAM2实例分割、点跟踪与刚性变换、Poly‑Fourier运动编码、深度相关Pearson损失等技术。

**📊 数据集**

在iPhone DyCheck、NVIDIA Dynamic Scene（原版与Gaussian Marbles）等标准单目动态重建基准上进行实验。

**📈 对比分析**

与MoSca、Gaussian Flow等方法在PSNR、LPIPS、训练时长、帧率等指标对比，MOSAIC-GS保持近似或更好PSNR、LPIPS最低、训练时间约10.5 min、渲染180 FPS，明显快于现有方法。

**⚠️ 局限性**

依赖外部分割与场景流的质量，对全隐蔽物体运动估计不足，且预处理对噪声较为敏感。

---

## 12. Retrieval-Augmented Multi-LLM Ensemble for Industrial Part Specification Extraction

**arXiv ID:** 2601.05266 | [PDF](https://arxiv.org/pdf/2601.05266v1)

**作者:** Muzakkiruddin Ahmed Mohammed `[一作]`, Adriaan Marais `[通讯]` (PiLog Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了RAGsemble系统，通过九个LLM和检索增强生成技术实现工业部件规格的高质量提取。

**💡 创新点**

创新点在于多模型协同与RAG全流程融合、置信度感知合成以及可配置的检索增强机制。

**🔧 技术方法**

采用多模型集成、检索增强生成、FAISS向量检索以及Gemini/OpenAI/Mistral/Gemma等大型语言模型。

**📊 数据集**

使用Pilog集团的工业部件数据以及GE9X风扇叶片案例进行实验。

**📈 对比分析**

与单模型基线（GPT‑4o、Claude、Gemini 2.5、Grok 3）对比，RAGsemble在完整性、技术深度和结构质量上均显著优于对手，完整性与技术深度均达100%。

**⚠️ 局限性**

局限包括成本高、延迟大、非确定性、对第三方API依赖、领域特定准确性下降以及检索质量受限。

---

## 13. CourtNav: Voice-Guided, Anchor-Accurate Navigation of Long Legal Documents in Courtrooms

**arXiv ID:** 2601.05255 | [PDF](https://arxiv.org/pdf/2601.05255v1)

**作者:** Sai Khadloya `[一作]` (Adalat AI), Utkarsh Saxena `[通讯]` (Adalat AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款名为CourtNav的语音导向、anchor‑first的法律PDF导航系统，能够根据法官的口头指令快速定位并高亮文档中的段落或表格单元；

**💡 创新点**

创新点在于将布局感知索引、混合检索、语法+LLM路由以及anchor‑first可审计视图相结合，聚焦导航而非摘要，显著提升检索精度与可验证性；

**🔧 技术方法**

使用了Whisper ASR、语法解析与LLM后备路由、BM25 + late‑interaction向量检索、PDF.js自定义视图、OCR 兼容布局解析以及表格网格建模等技术；

**📊 数据集**

利用15份长篇法律文件（起诉书、答辩状、裁判文书等）构建导航查询集，并发布了Indian Legal Retrieval dataset；

**📈 对比分析**

通过与传统PDF阅读器手动滚动+Find进行对比，Temporal命令的时间-相关性从10→5秒，Contextual命令从200→6秒；检索准确性上Late‑window+Keyword模式的Strict‑F1达到0.92，明显优于单纯关键词或密集检索；

**⚠️ 局限性**

局限性包括受PDF.js性能限制（大文件滚动慢）、文档尺寸限制、LLM路由的随机性、ASR错误率、仅支持英文、缺乏多语种和实时反馈机制。

---

## 14. MMViR: A Multi-Modal and Multi-Granularity Representation for Long-range Video Understanding

**arXiv ID:** 2601.05495 | [PDF](https://arxiv.org/pdf/2601.05495v1)

**作者:** Zizhong Li `[一作]` (University of California), Jiawei Zhang `[通讯]` (University of California)

**通讯引用:** 14243 | [OpenAlex ID](https://openalex.org/A5100462828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MMViR，一种多模态多粒度结构化长视频表示，用于视频问答、摘要与检索等任务；

**💡 创新点**

通过转折点分段、三层级描述（全局时间线、剪辑粗粒度文本、细粒度视觉-文本对）实现高效检索与全局细节兼顾，兼具结构化索引与跨模态压缩；

**🔧 技术方法**

采用CLIP视觉相似度分段、KTS方法、LLM（VideoLLaMA‑7B、GPT‑4o）生成多粒度文本、Contriever检索、帧低频采样等技术；

**📊 数据集**

在HourVideo、EgoSchema、Video‑MME、MovieChat‑1K、Ego4D等长视频基准上进行评测；

**📈 对比分析**

与现有caption、memory、检索基线对比，MMViR在VideoQA、摘要与检索任务上均取得更高准确率，尤其在HourVideo上提升19.67%同时处理时延降至45.4%；

**⚠️ 局限性**

细粒度视觉采样产生冗余；基于时间线摘要的检索可能遗漏细节；缺乏空间-时空建模，难以精准捕捉动态细节。

---

## 15. CosyEdit: Unlocking End-to-End Speech Editing Capability from Zero-Shot Text-to-Speech Models

**arXiv ID:** 2601.05329 | [PDF](https://arxiv.org/pdf/2601.05329v1)

**作者:** Junyang Chen `[一作]` (Nankai University), Yong Qin `[通讯]` (Nankai University)

**通讯引用:** 9438 | [OpenAlex ID](https://openalex.org/A5088716214)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了一个端到端语音编辑模型 CosyEdit，利用零样本 TTS 模型通过任务特定微调和推理优化实现插删改等编辑功能。

**💡 创新点**

采用零样本 TTS 的“后训练”策略，内部化语音-文本对齐，结合 AR+NAR 的语言模型与引导条件流匹配，且只需 250 小时数据即可实现高质量编辑。

**🔧 技术方法**

结合自回归大语言模型、非自回归条件流匹配、引导的最优传输（GOT‑CFM）、零样本/一示例上下文训练以及改进的推理协议。

**📊 数据集**

在自建的 250 小时 GigaEdit 数据集（基于 GigaSpeech）上微调，并在 RealEdit 基准上评测。

**📈 对比分析**

与多种级联与端到端编辑基线（FluentSpeech、VoiceCraft、SSR‑Speech、Step‑Audio‑EditX、MiMo‑Audio、Ming‑UniAudio）在 RealEdit 上对比，CosyEdit 在 WER、EMOS、声纹一致性等指标上均优于基线，接近甚至超过部分级联系统。

**⚠️ 局限性**

仍存在对多语言、多编辑位置的泛化不足、对背景噪声的鲁棒性有限，以及潜在的深度伪造风险需进一步监管。

---

## 16. Closing the Modality Reasoning Gap for Speech Large Language Models

**arXiv ID:** 2601.05543 | [PDF](https://arxiv.org/pdf/2601.05543v1)

**作者:** Chaoren Wang `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 TARS 框架，通过强化学习对齐语音与文本的推理轨迹，显著缩小语音推理与文本推理之间的差距。

**💡 创新点**

创新点在于非对称密集奖励设计——同时利用层级隐藏状态相似度（表示对齐）和语义嵌入相似度（行为对齐），并采用模态特定归一化的 GRPO，实现在 on‑policy 学习中纠正表示漂移与语义不一致。

**🔧 技术方法**

技术方法包括：基于 GRPO 与 DAPO 的强化学习；层级隐藏状态余弦相似度与语义嵌入相似度的密集奖励；LoRA 参数高效微调；语音合成（TTS）与 ASR 生成配对数据；xFinder 进行答案抽取。

**📊 数据集**

使用的数据集包括：UnifiedQA 训练集（合成语音+文本配对）；VoiceBench 下的 MMSU 与 OBQA 作为推理评测；LibriSpeech 用于评估 ASR 误码率。

**📈 对比分析**

与多种基线（AlignChat、SALAD、DeSTA、Knowledge Distillation、SFT、DPO、Standard GRPO、cascade ASR+LLM）比较，在 7B 规模模型上，TARS 在 MMSU 与 OBQA 上的语音准确率分别达约 76.8%–79.8%，平均 MRR 近 100%（甚至超越原始文本准确率），显著优于所有对比方法。

**⚠️ 局限性**

局限性：仅在 7B 规模验证，未探讨更大模型；仅针对单轮推理，未覆盖多轮或对话场景；对齐仍依赖文本参考，未充分利用情感、韵律等非文本语音信息。

---

## 17. Engineering the RAG Stack: A Comprehensive Review of the Architecture and Trust Frameworks for Retrieval-Augmented Generation Systems

**arXiv ID:** 2601.05264 | [PDF](https://arxiv.org/pdf/2601.05264v1)

**作者:** Dean Wampler `[一作]` (IBM Research), Alireza Seddighi `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 2018‑2025 年 Retrieval‑Augmented Generation（RAG）相关的学术研究、工业应用与真实部署进行了系统性文献综述，归纳并统一了多样化的 RAG 架构、评估方法与工程实践，构建了一个包含检索、融合、模态、适应性与信任校准等维度的完整体系结构分类法，并提出了针对 RAG 可信性、对齐与安全性的综合治理框架与最佳实践。

**💡 创新点**

创新点：
- 通过系统性文献检索与质量评估，首次将散乱的 RAG 研究归纳为统一的五维分类法；
- 建立了 RAG 评估与监控的标准化指标与工具链（如 RAGAS、LlamaIndex、UpTrain 等）；
- 提出了多层次的信任与安全漏洞映射与对策，涵盖知识库、检索层、生成层与引用层；
- 系统梳理了行业部署中的工程模式与反模式，为实际落地提供了可操作的设计指南；
- 通过对比分析多种 RAG 变体（Fusion‑RAG、RE‑RAG、Hierarchical‑RAG、Hybrid‑RAG、Graph‑RAG、Agentic‑RAG）展示了性能差异与适用场景。

**🔧 技术方法**

主要技术与方法：
- Systematic Literature Review（SLR）方法，基于 Kitchenham 与 Charters 规范，涵盖学术论文、行业报告、技术文档与开源实现；
- 多维度架构分类与指标化（检索策略、融合机制、模态、适应性、信任校准）；
- 评估框架与指标：RAGAS、LlamaIndex、TruLens、RAGChecker、DeepEval 等，包含检索相关度、上下文精确率/召回率、生成可信度、引用准确度等；
- 可信性与安全性评估模型：Hallucination Rate、Citation Accuracy、Context Adherence、Completeness 等；
- 采用行业标准 benchmark（HotpotQA、MS MARCO、Natural Questions、FEVER、RGB、OmniEval）进行跨模型与跨场景比较；
- 结合云与容器化部署、事件驱动、无服务器等架构模式，提出可伸缩的工程实现路径。

**📊 数据集**

主要使用的数据集与 benchmark：
- HotpotQA（多跳推理）
- MS MARCO（检索问答）
- Natural Questions（真实查询）
- FEVER（事实验证）
- RGB Benchmark（多域能力）
- OmniEval（金融等垂直领域）
- 以及针对行业内部数据的自定义知识库与知识图谱。

**📈 对比分析**

比较方法与性能表现：
- 对比传统单一检索 RAG 与多策略 RAG（Fusion‑RAG、RE‑RAG、Hierarchical‑RAG、Hybrid‑RAG、Graph‑RAG、Agentic‑RAG），通过 BLEU、ROUGE、BERTScore、LLM‑based Faithfulness 等指标进行统一评估；
- 表 5.1 与表 5.2 展示了各架构在准确率、推理深度、延迟与资源占用方面的相对优势：
  • RAG‑Fusion 在多来源覆盖与推理深度上表现突出；
  • RE‑RAG 在精度与召回率上显著提升；
  • Hybrid‑RAG 在词汇匹配与语义匹配之间取得平衡，适用于多语种与多样化查询；
  • Graph‑RAG 在关系建模与长文本推理上优势明显；
  • Agentic‑RAG 在动态规划与自适应检索方面具备最高的灵活性与可扩展性。

**⚠️ 局限性**

局限性与挑战：
- 领域碎片化导致缺乏统一的评估基准与接口规范；
- 评估框架与指标仍以人工标注为主，缺乏大规模无监督自评方法；
- 检索层面仍面临检索质量、索引新鲜度与多模态匹配的瓶颈；
- 生成层的幻觉与上下文截断问题仍未完全解决；
- 可信性与安全性框架多为理论描述，缺少统一的工业级实现与持续监测机制；
- 在极大规模部署（数十亿文档、实时低延迟）中，模型、检索与推理的统一优化仍存在显著的计算与成本挑战。

---

## 18. On the Transition to an Auction-based Intelligent Parking Assignment System

**arXiv ID:** 2601.05429 | [PDF](https://arxiv.org/pdf/2601.05429v1)

**作者:** Levente Alekszejenkó `[一作]` (Budapest University of Technology and Economics), Dobrowiecki Tadeusz `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了从传统泊车系统向基于拍卖的智能停车分配系统的过渡，并通过Eclipse SUMO微观仿真评估不同市场渗透率下的交通流、停车成本与用户体验；

**💡 创新点**

创新点在于引入同时独立上升拍卖（SIA）结合本地贪婪竞价（LGB）的停车分配机制，并系统性探讨低至高渗透率对交通拥堵、停车价格梯度和占用模式的影响；

**🔧 技术方法**

使用的技术包括：Eclipse SUMO仿真平台、基于Python的交通控制接口、Android/智能手机应用模拟竞价代理、SIA/LGB拍卖算法以及停车距离与价格的加权成本函数；

**📊 数据集**

使用的“数据集”为人工构造的6×6网格城市中心业务区模型，包含车辆来源、目的地、停留时长以及三类β（价格/步行权衡）分布，模拟了约11520辆车；

**📈 对比分析**

比较方法为将三种停车行为（基线、实时信息、拍卖）在0%–100%渗透率的十次重复仿真中，分别记录路程长度、停车价格、停车距离、交通流量等指标。结果显示：拍卖系统在渗透率升高时均可提升交通流量、缩短停车距离，并使非参与者亦获益；信息系统提升有限，且高渗透时性能趋于基线；拍卖系统虽略增路程长度，但总体收益显著；

**⚠️ 局限性**

局限性包括：仅采用简化的网格模型，未考虑真实道路几何、交通信号时序和停车冲突的细粒度；拍卖周期15s导致非参与者偶尔抢占预留位；仿真未覆盖多城市不同车流特征；对非参与者行为假设过于理想化；实际部署需考虑隐私、通信延迟与应用采纳成本。

---

## 19. Ontology Neural Networks for Topologically Conditioned Constraint Satisfaction

**arXiv ID:** 2601.05304 | [PDF](https://arxiv.org/pdf/2601.05304v1)

**作者:** Jaehong Oh `[一作]` `[通讯]` (Soongsil University), Jaehong Oh (Soongsil University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种融合拓扑约束、梯度稳定化和进化优化的本体神经网络（ONN），用于解决语义约束下的图结构优化问题。

**💡 创新点**

创新点在于将Forman‑Ricci曲率作为拓扑调节信号、引入Deep Delta Learning实现一阶投影的可学习秩‑一扰动以及利用CMA‑ES对权重与投影参数进行无梯度优化，从而实现种子无关、稳健收敛。

**🔧 技术方法**

所用技术包括本体神经网络架构、Forman‑Ricci曲率计算、Deep Delta Learning投影算子、LOGOS约束投影循环和CMA‑ES参数搜索。

**📊 数据集**

实验采用合成约束满足任务的图数据集，节点数从2到20不等，随机生成初始状态和约束，未使用公开真实数据集。

**📈 对比分析**

与基线（无ONN）、ONN v1（仅曲率）比较，ONN v2在20个随机种子下平均能量降至1.15（标准差0.36），成功率从基线12%提升至95%，显示显著性能提升。

**⚠️ 局限性**

主要局限在于规模上限约为20节点、对离散或非可微约束支持不足、以及需要进一步研究在线/时变约束的适应性。

---

## 20. Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geolocalization

**arXiv ID:** 2601.05432 | [PDF](https://arxiv.org/pdf/2601.05432v1)

**作者:** Yuxiang Ji `[一作]` (Xiamen University), Xiangxiang Chu `[通讯]` (AMAP, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出“Thinking with Map”模型，利用地图API工具和代理式强化学习实现图像地理定位，并通过并行测试时缩放提升性能。

**💡 创新点**

首次将地图工具集成到大规模视觉语言模型中，将定位过程建模为代理‑地图循环，并结合并行采样与验证器实现从pass@K到pass@1的跃迁。

**🔧 技术方法**

采用地图工具调用（POI检索、静态/卫星图像查询等）、代理式强化学习（GRPO）、并行测试时缩放+验证器以及Qwen3‑VL等大型视觉语言模型。

**📊 数据集**

使用自建MAPBench（5000张中国街景图）以及IMAGEO‑Bench、GeoBench等公开基准进行评测。

**📈 对比分析**

与多款开放与闭源模型（GPT‑o3、GPT‑5、Gemini‑3‑Pro、Qwen3‑VL等）对比，Acc@500m从约4%提升至约44%/55%，在多种指标上均领先。

**⚠️ 局限性**

地图工具使用仍不如人类精准，缺乏对空间关系的推理；强化学习数据有限，难以训练长时序决策；并行TTS是临时补偿，单一代理长时推理仍待提升。

---

## 21. Parallel Dynamic Spatial Indexes

**arXiv ID:** 2601.05347 | [PDF](https://arxiv.org/pdf/2601.05347v1)

**作者:** Ziyang Men `[一作]` (University of California Riverside), Yihan Sun `[通讯]` (University of California Riverside)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了两种新的并行空间索引数据结构——P-Orth-tree（并行空间划分树）和 Spatial PaC-tree（并行 R‑树族），并实现了在高动态工作负载下的低延迟批量更新和高效查询；

**💡 创新点**

创新点在于：①P-Orth-tree完全摆脱了传统基于空间填充曲线（SFC）的排序，采用基于 Sieve 的直接划分，从而实现构造和更新的 I/O‑efficient 设计；②Spatial PaC-tree 在 join‑based 并行 BST 上放宽叶子排序限制，只保留部分顺序，显著降低更新成本，同时保持查询性能；

**🔧 技术方法**

采用的技术包括并行 fork‑join 计算模型、Sieve 直方图划分、整数排序与 Hilbert/Morton 空间填充曲线、join‑based 纯函数式平衡 BST、叶子包裹（leaf‑wrapping）、分治并行以及缓存友好 I/O‑optimized 设计；

**📊 数据集**

实验使用了 10⁹ 条 2D/3D 合成点集（均匀、偏斜、随机游走、聚类分布）以及真实 GIS 数据集（北美地图、Cluster 数据集）来验证性能；

**📈 对比分析**

与现有并行空间索引（Parallel KD‑tree、Parallel Quadtree、Parallel R‑tree）以及 Boost R‑tree 等基线在构建、批量更新、k‑NN 和范围查询等任务上进行对比，结果显示 P-Orth‑tree 与 Spatial PaC‑tree 在构造/更新上通常比对手快 2–6 倍，查询性能与传统 KD‑tree/R‑tree 相当或更优；

**⚠️ 局限性**

限制包括：对高维（>3）受空间填充曲线精度限制，极端偏斜分布时 P-Orth‑tree 性能下降；实现仍以单机共享内存为主，尚未验证在分布式或多机环境中的可扩展性。

---

## 22. STELP: Secure Transpilation and Execution of LLM-Generated Programs

**arXiv ID:** 2601.05467 | [PDF](https://arxiv.org/pdf/2601.05467v1)

**作者:** Swapnil Shinde `[一作]` (Capital One), Emily Chen `[通讯]` (Capital One)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了STELP（Secure Transpiler and Executor of LLM-Generated Programs），一种通过可配置的语法子集、工具限制与安全控制动态地转译并执行LLM生成代码的安全执行引擎；

**💡 创新点**

创新点在于将转译器模式与实时执行相结合，形成可自定义安全规则、动态安全包装、反馈循环的多层防护体系，且首次通过人类验证的InjectedHumanEval数据集对安全性、正确性与延迟进行系统评估；

**🔧 技术方法**

核心技术包括AST解析与验证、可配置安全语法子集、代码转译生成带安全控制的Python代码、工具调用的超时/重试与代理化、基于LLM的反馈生成器以及多层安全控制；

**📊 数据集**

使用了人类验证的InjectHumanEval（含164安全样本、470不安全样本）以及Python-Code-Execution-Output和HumanEval等公开数据集进行评测；

**📈 对比分析**

与Meta的CodeShield等静态分析工具进行对比，STELP在True Block Rate达到1.0、True Allow Rate 0.981，显著优于CodeShield（TBR 0.68、TAR 0.93）；在正确性上100%通过，延迟仅平均提升4.93 ms（中位数0.19 ms），与原生Python差距可忽略；

**⚠️ 局限性**

局限性包括目前仅支持Python，跨语言支持尚在研发；对极高复杂度或长时间运行代码的安全控制仍需进一步验证；反馈循环对LLM生成质量的依赖可能导致重试次数过多。

---

## 23. The LLM Mirage: Economic Interests and the Subversion of Weaponization Controls

**arXiv ID:** 2601.05307 | [PDF](https://arxiv.org/pdf/2601.05307v1)

**作者:** Ritwik Gupta `[一作]` (University of California), Andrew W. Reddie `[通讯]` (University of California)

**通讯引用:** 227 | [OpenAlex ID](https://openalex.org/A5030445990)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文通过政策分析揭示了美国AI安全治理中以训练计算量为核心的“LLM幻影”，并提出基于意图与能力、效果与国际人道法的武器化定义及面向AI三元（数据、算法、计算）的实时基准框架，以替代仅凭计算阈值的监管模式。

**💡 创新点**

创新点在于：①将AI武器化从硬件/计算层级转移到效果层面；②引入国际人道法的效果主导评估标准；③设计可持续、实时评估的多维基准体系；④建议以NIST/NAIRR为核心的机构架构实现独立评估与出口管制衔接。

**🔧 技术方法**

采用的技术主要包括：规模法研究与批判（Sutton的Bitter Lesson、Chinchilla等），法律推演（IHL 追加议定书），基准设计方法（数据集选择、指标设定、硬件配置）以及持续对抗式评测流程。

**📊 数据集**

使用的数据集示例：化学与生物数据库（ZINC、PubChem）、软件漏洞测试库（Juliet Test Suite）、目标识别图像集（xView）等，均用于构建对应领域的任务基准。

**📈 对比分析**

方法上通过对比传统基于FLOPs或参数阈值的监管方式，论文展示了实时基准如何直接量化高危能力并驱动动态阈值更新；在性能上没有给出具体数值，而是强调基准能够揭示不同资源组合下的危险能力提升速率。

**⚠️ 局限性**

局限性包括：需要高度独立的评测机构与资源，可能面临政治与产业捕获风险；基准设计与验证复杂，需跨学科协作；缺乏实测数据对照，评估模型与实际武器化路径的相关性仍待进一步验证。

---

## 24. Self-Evolving Distributed Memory Architecture for Scalable AI Systems

**arXiv ID:** 2601.05569 | [PDF](https://arxiv.org/pdf/2601.05569v1)

**作者:** Zixuan Li `[一作]` (Pacific Coast University), Haotian Sun `[通讯]` (Northern Research Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Self-Evolving Distributed Memory Architecture（SEDMA）三层自适应框架，实现跨计算、通信、部署层的统一内存管理。

**💡 创新点**

创新点在于双层记忆体系驱动的动态矩阵分区、基于网络拓扑与计算能力的自适应对等选择，以及运行时连续重编译的部署优化，实现内存管理的自演化。

**🔧 技术方法**

结合 RRAM 内存计算、libp2p+DHT 的自适应路由、Kubernetes 动态编排、基于经验的 λ 调整与深度学习优化等技术。

**📊 数据集**

在 COCO 2017、ImageNet 与 SQuAD 三大分布式 AI 基准上进行实验。

**📈 对比分析**

与 Ray Distributed、PyTorch Distributed 及静态 RRAM 分区等基线对比，SEDMA 在内存利用率 87.3%/吞吐率 142.5 ops/s、通信延迟 171.2 ms、资源利用率 82.7% 以及适配速度 3.2 min 等方面均显著优于基线。

**⚠️ 局限性**

局限性包括对 RRAM 设备依赖、需要数据库存储长短期记忆导致额外延迟、以及在极大规模集群和多租户环境下的可扩展性尚待验证。

---

## 25. LiveVectorLake: A Real-Time Versioned Knowledge Base Architecture for Streaming Vector Updates and Temporal Retrieval

**arXiv ID:** 2601.05270 | [PDF](https://arxiv.org/pdf/2601.05270v1)

**作者:** Tarun Prajapati `[一作]` `[通讯]` (Indian Institute of Technology), Tarun Prajapati (Indian Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个实时版本化知识库架构LiveVectorLake，支持流式向量更新与时间点检索。

**💡 创新点**

创新点在于采用内容可寻址哈希进行段级变更检测、冷热双层存储分离以及跨层ACID一致性的时间查询路由。

**🔧 技术方法**

使用的技术包括SHA‑256内容哈希、Milvus向量数据库、Delta Lake/Parquet版本化存储、SentenceTransformers嵌入、HNSW索引、write‑ahead日志等。

**📊 数据集**

实验数据集为约100份文档的五个时间点版本（约12k段）。

**📈 对比分析**

与传统增量 upsert 与12h批量刷新比较，LiveVectorLake仅需10–15%内容重新处理，当前查询平均65ms，历史查询1.2s，时间查询准确率100%。

**⚠️ 局限性**

局限性包括同步吞吐量受限、仅文本支持、多模态和大规模分布式部署尚未实现。

---

## 26. What's Left Unsaid? Detecting and Correcting Misleading Omissions in Multimodal News Previews

**arXiv ID:** 2601.05563 | [PDF](https://arxiv.org/pdf/2601.05563v1)

**作者:** Fanxiao Li `[一作]` (Yunnan University), Min-Yen Kan `[通讯]` (National University of Singapore)

**通讯引用:** 11243 | [OpenAlex ID](https://openalex.org/A5066305082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了6000实例的多模态新闻预览误导遗漏基准，并提出OMGuard框架实现检测与标题修正。

**💡 创新点**

通过解释意识的微调与生成解释实现零样本标题修正，并使用认知模拟管线生成高质量标注，显著提升小模型检测能力。

**🔧 技术方法**

利用大型视觉语言模型、LoRA微调、解释生成、最小编辑与自由形式修正、以及视觉原型生成技术。

**📊 数据集**

以VisualNews新闻预览与完整文章为基础构建了6000实例基准。

**📈 对比分析**

在该基准上与多种开源8B/90B/235B LVLM对比，OMGuard 8B在检测F1达0.86、纠正成功率达0.95，性能逼近或超越235B模型。

**⚠️ 局限性**

仅通过标题修正可能带来新偏见，图像原型生成缺乏实用性，未探索真实图像检索或联合文本图像调整方法。

---

## 27. Tracing Moral Foundations in Large Language Models

**arXiv ID:** 2601.05437 | [PDF](https://arxiv.org/pdf/2601.05437v1)

**作者:** Chenxiao Yu `[一作]` (University of Southern California), Morteza Dehghani `[通讯]` (University of Southern California)

**通讯引用:** 3891 | [OpenAlex ID](https://openalex.org/A5065952016)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型中道德基础的内部表示与因果影响，使用Moral Foundations Theory进行层级与特征分析，并在推理阶段通过向量与稀疏特征进行激活干预。

**💡 创新点**

首次证明道德概念在LLM内部呈线性可分层结构，并通过稀疏自编码器将宏观道德方向分解为可解释的稀疏特征，从而实现对模型道德行为的可控干预。

**🔧 技术方法**

采用层级向量投影、稀疏自编码器（SAE）分解、线性推理时激活干预（宏观向量和微观特征）等技术。

**📊 数据集**

使用扩展的MFV‑130情景集构建概念向量，验证基于Reddit的Moral Foundations Corpus、Moral Foundations Dictionary 2.0、MFQ‑2问卷和MMLU等数据集。

**📈 对比分析**

通过Signed Wasserstein距离衡量向量与人类标签的分离度，利用MFQ‑2得分变化评估干预效果，发现宏观向量在某些道德基础上受“对齐惯性”影响，但微观稀疏特征干预能显著恢复或提升可控性，整体保持模型通用能力。

**⚠️ 局限性**

局限于仅研究两款中型指令调优模型，使用固定间隔的SAE可能遗漏细粒度特征，依赖Moral Foundations Theory与英美语料导致跨文化推广受限，且未系统评估干预对拒绝行为、偏见、毒性等实际部署副作用。

---

## 28. MMUEChange: A Generalized LLM Agent Framework for Intelligent Multi-Modal Urban Environment Change Analysis

**arXiv ID:** 2601.05483 | [PDF](https://arxiv.org/pdf/2601.05483v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 29. GaussianSwap: Animatable Video Face Swapping with 3D Gaussian Splatting

**arXiv ID:** 2601.05511 | [PDF](https://arxiv.org/pdf/2601.05511v1)

**作者:** Xuan Cheng `[一作]`, Lvqing Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于3D高斯喷射（3D Gaussian Splatting）的视频人脸换脸框架GaussianSwap，该框架可从目标视频构建3D人脸头像，并将源图像的身份特征迁移到头像上，实现可动画、可交互的面部替换；

**💡 创新点**

创新点包括：①将3D高斯喷射与FLAME人脸模型绑定，生成可随时间动态控制的头像；②采用三种主流人脸识别模型（ArcFace、FaceNet、Dlib）构建复合身份嵌入，提升身份保持度；③在头像生成后通过身份细调与传统像素级渲染相结合，兼具高质量视频输出与可交互特性；

**🔧 技术方法**

核心技术包括：3D Gaussian Splatting、FLAME参数追踪、鲁棒视频抠图、L1+SSIM重建损失、中心与尺度正则化、复合身份损失、Adam优化、球谐光照；

**📊 数据集**

实验使用公开人脸视频数据（如VoxCeleb/300V等）与任意源人脸图像，未公开具体数据集名称；

**📈 对比分析**

与现有方法（DynamicFace、VividFace、HiFiVFS、CanonSwap等）进行对比，评价指标为身份相似度、视觉清晰度与时序一致性。GaussianSwap在身份保持与视觉质量上均优于对比方法，并保持良好的时间连贯性；

**⚠️ 局限性**

局限性包括：训练时间长（6–10小时，RTX4090）；对极端姿态、遮挡和背景变化的鲁棒性待提升；需手动对源图像进行对齐；在大规模应用时对显存与算力要求较高。

---

## 30. Interactive Distillation for Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2601.05407 | [PDF](https://arxiv.org/pdf/2601.05407v1)

**作者:** Minwoo Cho `[一作]` (Institute for Robotics and Intelligent Machines), Matthew Gombolay `[通讯]` (Institute for Robotics and Intelligent Machines)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 HINT 框架，通过层次化教师与交互式知识蒸馏实现多智能体协作学习。

**💡 创新点**

创新点在于：① 层次化教师（HiT-MAC）提升教师性能与可扩展性；② 伪离线强化学习让教师在学生轨迹上自适应更新；③ 基于性能的过滤提升演示质量，减少教师与学生观测不匹配。

**🔧 技术方法**

技术手段包括：层次化强化学习、知识蒸馏（KL+熵正则）、V‑trace 离线校正、HetNet 及 HetGAT 通信网络、DAgger+性能过滤、Pseudo‑off‑policy RL 等。

**📊 数据集**

实验数据集为两类协作任务：FireCommander（火灾扑救）和 MARINE（海上物流），分别设有 easy/medium/hard 难度，涵盖 5–10 代理。

**📈 对比分析**

与多种 CTDE 基线（MAPPO、HAPPO、TarMAC 等）及 KD 基线（CTDS、IGM‑DA、PTDE）比较，HINT 在 medium/hard 任务上成功率提升 60–165%，并在平均步数上显著下降，表明其在复杂动态环境中的鲁棒性和高效性。

**⚠️ 局限性**

局限性包括：教师采用集中训练-集中执行，规模受限；需要人工定义层次结构，缺乏完全端到端自适应；计算开销相对较大，需多线程实现。

---

## 31. PRISM: Protocol Refinement through Intelligent Simulation Modeling

**arXiv ID:** 2601.05356 | [PDF](https://arxiv.org/pdf/2601.05356v1)

**作者:** Brian Hsu `[一作]`, Arvind Ramanathan `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 PRISM 框架，利用多代理 LLM 进行实验方案规划、生成可执行 YAML 并通过 NVIDIA Omniverse 仿真进行预执行验证，最终在实验室机器人平台上实现无人工干预的完整实验流程。

**💡 创新点**

创新点在于将自然语言理解、模块化多代理推理、机器人协议生成与高保真数字孪生仿真闭环结合，实现从实验想法到可执行机器人协议的全流程自动化，并通过仿真反馈形成自我纠错迭代。

**🔧 技术方法**

使用技术包括：多代理 LLM（GPT‑5、Claude、Gemini）与单代理推理、MADSci 协议格式、NVIDIA Omniverse Isaac Sim 物理仿真、ZeroMQ 接口与机器人控制、Python 生成 YAML 并与实验室机器人（Opentrons OT‑2、PF400、Azenta 设备）协同。

**📊 数据集**

使用的数据集为公开实验手册与网络检索得到的实验流程（如 Luna qPCR、Cell Painting），并通过人工整理的实验步骤与设备说明文档作为输入；未构造专门的标注数据集。

**📈 对比分析**

通过 F1 分数比较各 LLM 在协议生成与错误检测的准确率；单步生成中 GPT‑5 达到 1.0，Claude/ Gemini 在多轮迭代后可达 0.94–0.82；仿真反馈显著减少物理错误；PCR 实验成功验证协议的可执行性和实验结果与人工实验一致。

**⚠️ 局限性**

局限性包括：对科学有效性缺乏自动验证，需要人工审查；仿真不涉及液体物理与化学/生物反应；对大规模多步骤实验的全局错误检测仍不完善；对提示工程高度敏感，需进一步优化与自动化。

---

## 32. The Number of Cycles of Bi-regular Tanner Graphs in Terms of the Eigenvalues of the Adjacency Matrix

**arXiv ID:** 2601.05340 | [PDF](https://arxiv.org/pdf/2601.05340v1)

**作者:** Roxana Smarandache `[一作]` (University of Notre Dame), David G. M. Mitchell `[通讯]` (New Mexico State University)

**通讯引用:** 5828 | [OpenAlex ID](https://openalex.org/A5076523933)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了双正规（bi-regular）图（如 Tanner 图）中短周期（cycle）的计数问题，提出了利用邻接矩阵（adjacency matrix）本征值直接计算周期数的递推与显式公式。

**💡 创新点**

创新点在于：① 将原本需要先求导向边矩阵（directed edge matrix）本征值的两步方法压缩为单步，直接用邻接矩阵本征值完成计算；② 给出了递推公式和闭式表达式，可快速得到 2k  长度循环数，且仅需本征值的幂；③ 证明了与 girth（环长）之间的必要充分条件，并提供了可手工计算的简洁公式。

**🔧 技术方法**

采用的技术主要包括：谱图理论、矩阵本征值与其多项式、Newton 恒等式、递推关系、块循环矩阵（block-circulant matrix）特性，以及对 QC‑LDPC 码的矩阵多项式表示。

**📊 数据集**

论文主要是理论推导与符号例子，并未使用具体实验数据集；所用的“数据”是示例码（如 3×5, 3×7, 31 阶块循环矩阵）以及已知的本征值集合。

**📈 对比分析**

与传统方法（先求导向边矩阵本征值再求周期）相比，递推公式不需要计算非整数根，运算量大幅下降；在示例中得到的 8、10、12、14 长度循环数与其他文献结果一致，证明了方法的正确性和效率。

**⚠️ 局限性**

局限性包括：① 仅针对双正规二分图；② 对较大 k（>14）时递推公式变得繁琐；③ 需要邻接矩阵本征值，若这些本征值难以解析获得，仍需数值求解；④ 该方法未直接给出最优码设计方案，只是为循环计数提供工具。

---

## 33. Scalable Heterogeneous Graph Learning via Heterogeneous-aware Orthogonal Prototype Experts

**arXiv ID:** 2601.05537 | [PDF](https://arxiv.org/pdf/2601.05537v1)

**作者:** Wei Zhou `[一作]` (Huazhong University of Science and Technology), Bang Liu `[通讯]` (Université de Montréal)

**通讯引用:** 973 | [OpenAlex ID](https://openalex.org/A5100691219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种可插拔的异构图学习解码框架HOPE，使用原型驱动的专家路由和正交约束，克服传统线性投影的瓶颈。

**💡 创新点**

创新点在于（1）将Mixture‑of‑Experts迁移到解码阶段；（2）利用可学习的原型进行语义相似性路由，天然匹配长尾分布；（3）加入正交约束保证专家多样性；（4）弹性容量机制平衡高频与低频节点，防止专家坍塌。

**🔧 技术方法**

技术包括：异构图神经网络（R‑GCN、R‑GAT、R‑GAT、SeHGNN、HGAMLP等）作为编码器；Mixture‑of‑Experts架构；原型向量+余弦相似度路由；正交损失约束；弹性容量筛选（质量、稳定、容量三层阈值）。

**📊 数据集**

使用四个真实异构图数据集：Freebase、Ogbn‑mag、DBLP（节点分类+链接预测）和Yelp（链接预测）。

**📈 对比分析**

与多种基线（R‑GCN、R‑GAT、R‑GSN、NARS、SeHGNN、HGAMLP）对比，HOPE在所有任务上均提升准确率（节点分类提升约2–6%）、链接预测提升约1–3%，同时计算与参数开销几乎不变甚至略有降低。

**⚠️ 局限性**

局限性包括：需手动调参（如正交权重、阈值δ、上下限K、C）；对不同类型图的原型数目和维度需要经验选择；在极大规模图上，原型匹配与稀疏激活仍可能产生额外的内存占用；并未深入探究在非节点分类任务（如图生成、图翻译）上的适用性。

---

## 34. PRISMA: Reinforcement Learning Guided Two-Stage Policy Optimization in Multi-Agent Architecture for Open-Domain Multi-Hop Question Answering

**arXiv ID:** 2601.05465 | [PDF](https://arxiv.org/pdf/2601.05465v1)

**作者:** Yu Liu `[一作]` (Institute of Information Engineering), Zhiyuan Ma `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 639 | [OpenAlex ID](https://openalex.org/A5100622932)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于多代理的检索增强生成框架PRISMA，解决开放域多跳问答中的检索崩溃与学习不稳定问题。

**💡 创新点**

通过Plan-Retrieve-Inspect-Solve-Memoize五阶段架构和两阶段GRPO+OARPO训练，实现了检索-推理-验证协同与目标感知的错误检测与回滚。

**🔧 技术方法**

采用强化学习（GRPO、OARPO）、多代理协作（Planner、Inspector、Solver、Memoizer）、三阶段检索（dense→hybrid→cross‑encoder）以及序列化对话式推理。

**📊 数据集**

在10个开放域多跳QA基准上评估，包括HotpotQA、2WikiMultiHopQA、MuSiQue、NQ、MultiHopRAG、Bamboogle以及五个领域专属集（Chemistry、Food、Game、Geography、Music）。

**📈 对比分析**

与多种无训练与训练方法对比，PRISMA在所有基准上均取得最优或竞争最优的EM/F1，比强大的API模型在最难任务上仍有显著提升；在训练样本、检索召回、延迟等方面也表现出较好的效率。

**⚠️ 局限性**

需要大规模GPU内存、训练成本高、模型对单个组件的对齐敏感、难以超越快速发展的大型API模型。

---

## 35. CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems

**arXiv ID:** 2601.05520 | [PDF](https://arxiv.org/pdf/2601.05520v1)

**作者:** Xuemei Tang `[一作]` (Hong Kong Polytechnic University), Chu-Ren Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 6554 | [OpenAlex ID](https://openalex.org/A5024924150)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建古代中国历史事件层级分类体系

**💡 创新点**

将分类构建拆分为三阶段（Inducer、Expander、Enricher）并引入多模态 LLM 角色协同完成

**🔧 技术方法**

多模型 LLM（GPT‑4o、GPT‑5、DeepSeek‑V3、Qwen3）+ 语义相似度、生成式对话式推理

**📊 数据集**

《二十四史》原始文本（约 2000 年历史跨度）

**📈 对比分析**

与 Chain‑of‑Layer、TaxoAdapt、人工 CHED 基准对比，性能在覆盖率、节点召回、创新度上居优，结构一致性与人类评估亦优于单一 LLM 方法

**⚠️ 局限性**

受模型偏见、古文理解局限、官方史料视角缺失、事件中心化限制、评价指标局限等影响

---

## 36. Conformity and Social Impact on AI Agents

**arXiv ID:** 2601.05384 | [PDF](https://arxiv.org/pdf/2601.05384v1)

**作者:** Alessandro Bellina `[一作]` (Centro Ricerche Enrico Fermi), David Garcia `[通讯]` (University of Konstanz)

**通讯引用:** 7063 | [OpenAlex ID](https://openalex.org/A5084395089)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统性地将社会心理学中的一致性实验（如 Asch 视觉任务）改编为多模态 LLM 的视觉判定任务，评估在不同群体大小、同意度、任务难度、源强度与社会接近度等因素下，LLM 对群体压力的响应；

**💡 创新点**

首次将 Social Impact Theory 概念迁移至人工智能代理，证明多模态 LLM 在群体大小、权威源、身份归属等方面呈现与人类相似的一致性模式；

**🔧 技术方法**

使用多模态指令微调 LLM（Qwen2.5、Gemma、Ovis、Mistral 等）配合 Prompt 设计，通过提取 logits 计算错误率来量化一致性；

**📊 数据集**

构建合成图像数据集，包含线条判断、颜色识别与点数估计三类视觉任务，保证基线下模型 100% 正确率；

**📈 对比分析**

通过对比不同条件下的错误率曲线（p_wrong(N)）和 AUC 指标，发现模型在群体压力下错误率显著上升，表现与人类一致；对比不同模型尺寸，结果相似，未见性能提升显著降低一致性；

**⚠️ 局限性**

局限性包括仅使用人工合成视觉任务，缺乏自然场景与多模态交互；实验仅覆盖少数任务类型和模型族，未探索因果机制，且未检验长期交互与自适应学习对一致性的影响。

---

## 37. Sketch&Patch++: Efficient Structure-Aware 3D Gaussian Representation

**arXiv ID:** 2601.05394 | [PDF](https://arxiv.org/pdf/2601.05394v1)

**作者:** Yuang Shi `[一作]` (National University of Singapore), Wei Tsang Ooi `[通讯]` (National University of Singapore)

**通讯引用:** 2771 | [OpenAlex ID](https://openalex.org/A5072587271)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种针对3D高斯分布的分层压缩方法Sketch&Patch++，将高频结构特征与低频体积特征分别编码并实现大幅度压缩

**💡 创新点**

通过多维度基于密度的聚类直接在已优化的3D高斯分布上识别Sketch和Patch高斯，消除对外部线段检测的依赖，并引入多阈值聚类与自适应多项式回归细化

**🔧 技术方法**

使用密度聚类（改进的DBSCAN）、多项式回归编码、可变形重训练、向量量化、Draco几何压缩、半精度、Gzip等多种压缩技术

**📊 数据集**

在七个场景（Playroom、Drjohnson、Room、Kitchen、Garden、Train、Truck）上使用Deep Blending、Mip-NeRF360和Tanks&Temples数据集

**📈 对比分析**

与先前的Sketch&Patch、Prune&Retrain、Sketch单独压缩以及多种SOTA压缩方法对比；在相同质量下比S&P小 2.7–3.9 倍，在同等模型尺寸下比S&P和Prune&Retrain取得 1.7–6.7 PSNR/SSIM/LPIPS 的提升，压缩率可达 175×

**⚠️ 局限性**

目前仅适用于静态场景；缺乏动态更新机制、完整的流式传输评估以及对语义信息的利用

---

## 38. Do LLMs Need Inherent Reasoning Before Reinforcement Learning? A Study in Korean Self-Correction

**arXiv ID:** 2601.05459 | [PDF](https://arxiv.org/pdf/2601.05459v1)

**作者:** Hongjin Kim `[一作]` (ETRI), Oh-Woog Kwon `[通讯]` (ETRI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对LLM进行RL训练与多种微调策略（包括持续预训练、层级微调、LoRA、DPO以及针对韩语特定神经元的细粒度调优），探究其在低资源韩语推理与自我纠错任务中的效果。

**💡 创新点**

创新点在于利用神经元重要性检测发现并微调早期层的韩语特定神经元，配合代码切换的自我纠错数据，从而在不注入新知识的前提下有效激活并对齐LLM现有的英文推理能力。

**🔧 技术方法**

主要技术包括Group Relative Policy Optimization（GRPO）强化学习、自我纠错代码切换数据生成、神经元重要性评估与细粒度微调、LoRA低秩适配、DPO直接偏好优化，以及内部翻译分析等。

**📊 数据集**

使用的数据集包括MathDial、HRM8K（包含GSM8K、MATH、Omni‑MATH）、MMLU、GPQA以及自构建的韩语自我纠错代码切换数据。

**📈 对比分析**

通过与多种基线微调方法（持续预训练、层级微调、LoRA、DPO）对比，实验显示在韩语数学推理与自我纠错任务上，神经元微调方法可获得最高的性能提升，并且在RL后进一步显著提升自我纠错率，同时保持英文任务的性能。

**⚠️ 局限性**

局限性包括：仅针对早期层神经元，可能无法推广到其他任务或语言；使用的数据量有限，易出现过拟合；未提升韩语生成的流利度与文化细节；需在更大规模、多语言环境中进一步验证。

---

## 39. The Kernel Manifold: A Geometric Approach to Gaussian Process Model Selection

**arXiv ID:** 2601.05371 | [PDF](https://arxiv.org/pdf/2601.05371v1)

**作者:** Md Shafiqul Islam `[一作]` (Texas A&M University), Raymundo Arróyave `[通讯]` (Texas A&M University)

**通讯引用:** 10704 | [OpenAlex ID](https://openalex.org/A5055147706)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于Gaussian Process先验距离的几何化核搜索框架，通过多维尺度嵌入构造连续核空间，并在该空间上执行贝叶斯优化以自动发现适用于不同任务的复合核

**💡 创新点**

创新点在于用概率发散度（如Jensen–Shannon）定义核间距离，纠正曲率后可嵌入欧氏空间，使得核搜索从离散语法跳转为连续优化，显著提升搜索效率和稳定性

**🔧 技术方法**

核心技术包括GP先验的期望发散度测度、曲率校正（对数扭曲或球面弦距映射）、多维尺度嵌入、核对核的多尺度GP代理以及基于EI的贝叶斯优化

**📊 数据集**

实验使用合成七个经典优化函数（Eggholder、Ackley、Dropwave等）以及三组真实时间序列数据（国际航空客运、Mauna Loa CO₂、热历程），还在增材制造熔池几何的TCAM模拟数据上进行案例验证

**📈 对比分析**

与传统基于BIC的贪婪搜索、单一RBF核的贝叶斯优化、随机搜索以及LLM辅助遗传算法（CAKE）相比，本文方法在所有任务中收敛更快、最终log似然更高、方差更小；在多目标印刷可打印性优化中亦显著提升超体积指标

**⚠️ 局限性**

局限在于需先预先构造大量核库并计算高维距离矩阵，计算成本仍高；对极端异构数据或需要深层语法的核时，距离映射可能失效；另外，映射到欧氏空间后仍受MDS维数截断的影响，导致某些高度非平坦的核关系难以完整保留

---

## 40. SAS-VPReID: A Scale-Adaptive Framework with Shape Priors for Video-based Person Re-Identification at Extreme Far Distances

**arXiv ID:** 2601.05535 | [PDF](https://arxiv.org/pdf/2601.05535v1)

**作者:** Qiwei Yang `[一作]` (Dalian University of Technology), Zijing Gong `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种针对极远距离空地视频行人重识别（VPReID）的统一框架SAS‑VPReID，能够同时解决尺度变化、视角差异、服装变化和低分辨率等挑战。

**💡 创新点**

三大创新点：① 通过视频一致性颜色抖动与多代理记忆监督提升CLIP视觉编码器在极端降质环境下的鲁棒性；② 多尺度时序建模（MGTM）与可学习尺度融合，既捕捉短期运动细节又聚合长期运动趋势；③ 引入SMPL基础形状先验的先验正则化形状动力学（PRSD），补充服装无关的人体结构特征。

**🔧 技术方法**

采用CLIP‑ViT‑L视觉编码器、Mamba序列运算、GRU与Transformer混合的时序建模、SMPL形状参数回归、对比学习与三元组损失等技术。

**📊 数据集**

在DetReIDXV1（极远距离空地+服装变化）和VReID‑XFD挑战集上进行实验，使用两次拍摄、不同平台和服装变化的数据。

**📈 对比分析**

与多种基线和挑战榜单前列方法对比，SAS‑VPReID在A→G、G→A、A→A三种评估设置下均取得最高mAP与Rank‑1；在VReID‑XFD排行榜上获得32.89 mAP‑3，领先第二名2.46点。

**⚠️ 局限性**

主要局限：对极低分辨率（≤10像素）下的关键帧特征仍易受噪声影响；形状回归对遮挡敏感；模型规模较大，推理时GPU内存占用高，需进一步优化。

---

## 41. Bayesian Recovery for Probabilistic Coalition Structures

**arXiv ID:** 2601.05273 | [PDF](https://arxiv.org/pdf/2601.05273v1)

**作者:** Angshul Majumdar `[一作]` (Indraprastha Institute of Information Technology), Angshul Majumdar `[通讯]` (Indraprastha Institute of Information Technology)

**通讯引用:** 6139 | [OpenAlex ID](https://openalex.org/A5020310463)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究将概率联盟结构生成问题转化为稀疏线性回归模型，并分析在高度共线设计下不同稀疏恢复方法的表现。

**💡 创新点**

证明在高度共线的联盟结构中，传统的ℓ1软阈值和贪婪OMP方法失效，而稀疏贝叶斯学习（SBL）能够实现支持一致性，从而揭示贝叶斯方法在此类问题中的优势。

**🔧 技术方法**

采用稀疏线性模型、ℓ1正则化（LASSO）、k步Orthogonal Matching Pursuit（OMP）以及Gaussian–Gamma层级的SBL最大后验估计等技术，辅以概率论和极限条件证明。

**📊 数据集**

未使用真实数据集，全部基于理论构造的联盟共现矩阵（近似重复列的合成设计）进行分析。

**📈 对比分析**

在相同的高维共线设计下与LASSO、OMP对比，后者的支持恢复概率保持在常数下限，而SBL的恢复概率随维度增大趋近于1。

**⚠️ 局限性**

仅在高斯噪声、β_min 条件和特定共线结构下给出理论证明，缺乏实验验证，且不考虑非高斯噪声或更一般的价值函数。

---

## 42. TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds

**arXiv ID:** 2601.05499 | [PDF](https://arxiv.org/pdf/2601.05499v1)

**作者:** Weishang Wu `[一作]` (National University of Defense Technology), Zhiping Cai `[通讯]` (National University of Defense Technology)

**通讯引用:** 7827 | [OpenAlex ID](https://openalex.org/A5006334685)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出任务导向的形状完成与对应抓取生成方法，通过先生成多种基于任务的形状候选，再挑选并优化，最终输出符合下游操作任务的抓取姿态。

**💡 创新点**

创新点在于：①把形状完成转化为任务导向，关注接触区域而非完整几何；②利用多模态预训练模型生成候选；③使用3D判别式自编码器选择最佳形状并全局优化；④通过单步梯度校正的流匹配模型FlowGrasp无额外损失实现约束感知抓取。

**🔧 技术方法**

采用ControlNet+预训练3D生成模型、SAM+大模型做任务区域检测、3D判别式自编码器、FlowGrasp条件流匹配模型，配合CLIP、PointNet++、GPT‑4o等技术。

**📊 数据集**

训练集包括ModelNet40、ShapeNetCore、ScanObjectNN、OmniObject3D、DexGraspNet、AffordPose；完成后用OakInk‑PartialPC数据集进行评测。

**📈 对比分析**

与多种基准（GraspCVAE、GraspTTA、SceneDiffuser、DexTOG、DexGYSGrasp、PointAttn、SVDFormer、SymmCompletion）对比，Grasp Displacement下降16.17%、Chamfer Distance提升55.26%，在抓取稳定性、接触率和评估分数等指标均显著优于现有方法。

**⚠️ 局限性**

限制在于：仍需依赖大模型的推理时间，生成候选过程对计算资源要求较高；对极端遮挡或完全未知类别的形状恢复尚未充分验证；以及对真实场景噪声和光照变化的鲁棒性待进一步提升。

---

## 43. Blockchain Verifiable Proof of Quantum Supremacy as a Trigger for Quantum-Secure Signatures

**arXiv ID:** 2601.05534 | [PDF](https://arxiv.org/pdf/2601.05534v1)

**作者:** Nicholas Papadopoulos `[一作]` `[通讯]`, Nicholas Papadopoulos

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在以太坊上实现一个智能合约，生成不可预知的RSA‑UFO难题，并利用提交‑揭示机制实现无信任的量子超越证明及资产安全回退。

**💡 创新点**

创新点在于将区块链与量子超越验证结合，提供可信的量子超越证明与即时切换到后量子签名的机制。

**🔧 技术方法**

使用Solidity智能合约、Sander的RSA‑UFO生成方法、提交‑揭示协议以及Lamport签名做后量子验证。

**📊 数据集**

未使用传统数据集，难题由合约内随机生成；示例中生成119个3072位RSA‑UFO锁。

**📈 对比分析**

与订单寻找（order‑finding）难题对比，部署与验证Gas消耗更低；实验显示单锁验证约xxxxx Gas，整个合约约xxxxx Gas。

**⚠️ 局限性**

局限包括高Gas成本、需大量交易以完成锁生成、仅适用于以太坊网络、对量子硬件成熟度假设，且Lamport验证在主网仍不可行。

---

## 44. Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings

**arXiv ID:** 2601.05271 | [PDF](https://arxiv.org/pdf/2601.05271v1)

**作者:** Xiran Fan `[一作]` (Visa Research), Yan Zheng `[通讯]` (Visa Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种混合框架，将大型语言模型（LLM）生成的语义嵌入作为先验初始化，融合到传统基于表格的序列交易模型中，以弥补索引式类别表示造成的语义信息丢失。

**💡 创新点**

创新点包括：①利用LLM离线生成的句子嵌入作为嵌入层的初始值，既保留了LLM的语义理解又避免了实时推理成本；②提出“一词限制”提示原则，保证不同LLM在生成嵌入时输出格式一致、噪声可控；③构建多源数据融合流程，对MCC、商家名、地理信息进行外部知识补充；④在多任务学习框架下统一评估交易量、下降率、欺诈率等业务指标。

**🔧 技术方法**

主要技术手段包括：表格基础模型（Transformer‑based sequential tabular model），LLM提示工程（Llama2‑7b/13b、Llama3‑8b、Mistral‑7b），从LLM隐藏层最后非填充标记提取句子嵌入，嵌入层的语义初始化，以及多任务联合训练。

**📊 数据集**

使用约10亿条2022‑2023年1月‑12月的交易记录，包含商家名称、MCC、位置、金额等字段，并通过外部来源（官方MCC文档、行业分类、经济指标等）进行补充。

**📈 对比分析**

与传统仅使用ID‑embedding的基线模型（Vanilla）进行对比，采用MAE、sMAPE、Acc、F1等指标评估；在交易指标评估任务中使用相对改进（RI）衡量。实验表明，LLM嵌入初始化在大多数任务中均取得显著提升（最高RI≈3.93%），且在多数配置下表现优于基线。

**⚠️ 局限性**

局限性包括：①提示工程相对简单，可能未能充分挖掘LLM潜能；②仅评估了四种主流LLM，未涉及最新的嵌入模型；③仅对MCC、商家名、位置三类字段进行语义化，其他重要字段未覆盖；④嵌入为静态预计算，无法动态捕捉商家特征随时间演变的变化。

---

## 45. LLM2IR: simple unsupervised contrastive learning makes long-context LLM great retriever

**arXiv ID:** 2601.05262 | [PDF](https://arxiv.org/pdf/2601.05262v1)

**作者:** Xiaocong Yang `[一作]` `[通讯]` (University of Illinois Urbana-Champaign), Xiaocong Yang (University of Illinois Urbana-Champaign)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将任意解码器大语言模型通过无监督对比学习（随机裁剪+BM25硬负样本）转化为高效稠密检索模型

**💡 创新点**

不依赖多阶段训练、双向注意力或人工标注，仅用单一无监督对比损失即可与有监督检索模型媲美；并证明更长上下文窗口显著提升检索性能

**🔧 技术方法**

无监督对比学习、随机裁剪、BM25硬负样本、LoRA适配、RoPE/LongRope/YaRN上下文扩展

**📊 数据集**

使用Wikitext‑103进行训练；评测数据集包括LoCo、LongEmbed、BEIR（12项LoCo任务、4项LongEmbed任务及多域BEIR任务）

**📈 对比分析**

与LLM2Vec、E5‑Mistral、BM25等基线对比，LLM2IR在LoCo、LongEmbed和BEIR上均取得或超过基线性能；在长上下文任务中，128k窗口模型平均提升6‑8点nDCG@10

**⚠️ 局限性**

正样本仅为随机裁剪，可能与真实查询不匹配；缺乏对上下文长度“悬崖”的理论解释；实验仅在单语言单步检索，未涵盖多语言、多跳等场景

---

## 46. Congestion Mitigation in Vehicular Traffic Networks with Multiple Operational Modalities

**arXiv ID:** 2601.05375 | [PDF](https://arxiv.org/pdf/2601.05375v1)

**作者:** Doris E. M. Brown `[一作]` (Missouri University of Science and Technology), Sajal K. Das `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 32159 | [OpenAlex ID](https://openalex.org/A5050881965)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了基于信任的控制交易策略 TACTS，用于在多模态车辆与仲裁系统之间通过动态切换控制权来降低交通网络拥堵。

**💡 创新点**

首次把多模态车辆的行为建模为贝叶斯 Stackelberg 游戏，并通过惩罚-回报（regret matching）实现自适应信任更新，逼近系统最优的控制分配。

**🔧 技术方法**

使用 Stackelberg 游戏理论、信任计算与惩罚匹配算法、BPR 交通时延模型及 Python 仿真实现。

**📊 数据集**

在真实交通网络数据集 Anaheim、Sioux Falls 和 Chicago Sketch 上进行仿真，并对多种诱导流量与拥堵水平进行实验。

**📈 对比分析**

与 DOC、TASR、随机与单一控制等基线对比，TACTS 在所有实验场景下平均性能比最优值低 8.8% 以内，且在高拥堵时优于其他策略，计算耗时仅为毫秒级。

**⚠️ 局限性**

局限在于仅考虑单辆车与仲裁系统交互、假设可信度可通过历史误差估计、在完全多模态网络中可能需要扩展算法。

---

## 47. Intent at a Glance: Gaze-Guided Robotic Manipulation via Foundation Models

**arXiv ID:** 2601.05336 | [PDF](https://arxiv.org/pdf/2601.05336v1)

**作者:** Tracey Yee Hsin Tay `[一作]` (University of California, Los Angeles), Yuchen Cui `[通讯]` (University of California, Los Angeles)

**通讯引用:** 1270 | [OpenAlex ID](https://openalex.org/A5070110088)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出了基于眼动追踪与视觉语言模型的‘Gaze Assisted Manipulation for Modular Autonomy’系统，可通过用户视线自动推断意图并执行机器人操作，无需任务专属训练；

**💡 创新点**

创新点在于将人眼注视点与大型基础模型（VLM）相结合，实现从低级注视到高级意图再到低级抓取姿态的端到端零样本推理，兼顾直观、低负担和可扩展性；

**🔧 技术方法**

核心技术包括Meta Project Aria眼镜的自我中心眼动估计、ArUco标记姿态映射、SAM2分割、Contact‑GraspNet抓取预测，以及Gemini Pro/2.5、Llama4、GPT‑4o等视觉语言模型的多轮推理与视频提示；

**📊 数据集**

使用了实验室自建的30个桌面场景（Lab‑Tabletop）和45个从DROID数据集抽样的自然场景，结合RealSense RGB‑D图像进行评测；

**📈 对比分析**

与基线的眼动面板控制方法相比，系统在意图识别上成功率约0.79‑0.84（桌面）/0.64‑0.73（野生），抓取选择成功率最高为0.60，且在用户研究中完成任务的时间减少一半以上，但用户更偏好基线的直接控制；

**⚠️ 局限性**

局限包括VLM在细粒度低级规划上的不稳定、推理耗时长、零样本抓取误差累积导致失败、对ArUco标记的依赖、缺乏移动机器人能力以及仅在健康受试者上验证，未来需探索混合控制与更高效模型。

---

## 48. RingSQL: Generating Synthetic Data with Schema-Independent Templates for Text-to-SQL Reasoning Models

**arXiv ID:** 2601.05451 | [PDF](https://arxiv.org/pdf/2601.05451v1)

**作者:** Marko Sterbentz `[一作]` (Northwestern University), Kristian J. Hammond `[通讯]` (Northwestern University)

**通讯引用:** 5091 | [OpenAlex ID](https://openalex.org/A5114000097)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将schema‑independent查询模板与LLM重新表述问题相结合，生成高质量的SQL–问题对，用于训练文本到SQL模型。

**💡 创新点**

创新点在于：① 设计可跨数据库重用的schema‑independent模板，保证SQL语法与语义正确；② 用LLM对模板生成的问题进行自然语言重述，提升语言多样性；③ 两者结合既保持可靠性又具备可扩展性。

**🔧 技术方法**

技术方法包括：SQR（结构化查询表示）模板与Ring（数据库语义标签）映射；模板填充、随机过滤器生成、SQL编译与简化；LLM（GPT‑4o‑mini）重写问题；RLVR强化学习微调；使用Qwen2.5‑Coder‑3B/7B进行训练。

**📊 数据集**

数据集：自行构造的RingSQL数据集（5000问答对），覆盖160个多样化数据库；对比SynSQL（Uniform/Complex）等现有合成数据集；评测基准为Spider、BIRD及其变体。

**📈 对比分析**

在Spider、BIRD及其变体上进行对比实验。使用RingSQL训练的模型在六个基准上平均提升约2.3%准确率，3B模型最高平均为72.7%，7B模型最高平均为85.5%，显著优于基线与SynSQL数据集。

**⚠️ 局限性**

limitations：仅生成SQLite语法的查询；实验仅在Qwen2.5‑Coder 3B/7B上进行；模板设计仍需人工；未覆盖数据清洗或转换操作；计算资源限制导致规模受限。

---

## 49. Prediction of Fault Slip Tendency in CO${_2}$ Storage using Data-space Inversion

**arXiv ID:** 2601.05431 | [PDF](https://arxiv.org/pdf/2601.05431v1)

**作者:** Xiaowen He `[一作]` (Stanford University), Louis J. Durlofsky `[通讯]` (Stanford University)

**通讯引用:** 17733 | [OpenAlex ID](https://openalex.org/A5002057296)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了基于变分自编码器（VAE）的数据空间反演框架，用于在CO₂贮存项目中预测压力、应变、应力场以及断层滑动倾向。

**💡 创新点**

创新点在于将高维耦合流‑地质力学响应通过VAE映射为低维近似高斯潜变量，直接在数据空间完成后验预测，既避免了生成后验地质模型，又显著降低了计算成本。

**🔧 技术方法**

采用了3D ConvLSTM结构的变分自编码器、数据空间反演（DSI）+ESMDA同化、以及GEOS耦合流‑地质力学模拟。

**📊 数据集**

使用了约1200个基于墨西哥湾（Gulf of Mexico）几何模型的随机渗透率/孔隙率场与不确定地质力学/断层参数的先验模拟数据进行VAE训练和DSI验证。

**📈 对比分析**

与传统基于模型的历史匹配（如ESMDA或MCMC）相比，DSI‑VAE在GPU推理仅需7 min，总体耗时约2400 CPU‑h＋12 GPU‑h；后验结果在压强、应变和滑动倾向上的误差降至<0.05，参数不确定性大幅缩小，性能显著优于需要10³–10⁴次模拟的传统方法。

**⚠️ 局限性**

局限性包括：缺乏后验地质模型（无法直接用于井位优化等后续决策），对极端非高斯分布或极端多尺度特征的适用性待验证，以及需要大量先验高质量模拟来训练VAE。

---

## 50. Cross-Document Topic-Aligned Chunking for Retrieval-Augmented Generation

**arXiv ID:** 2601.05265 | [PDF](https://arxiv.org/pdf/2601.05265v1)

**作者:** Mile Stankovic `[一作]` `[通讯]`, Mile Stankovic

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出跨文档主题对齐（CDTA）chunking 方法，先在语料库层面识别主题，再通过 LLM 将跨文档相关段落聚合合成完整、信息密集的知识块，最终用于 RAG 系统的检索与生成。

**💡 创新点**

创新点在于：
1) 将分块视角从文档内扩展到语料库整体，解决知识碎片化；
2) 通过主题提取、聚类、相关性过滤和 LLM 合成形成统一的主题知识单元；
3) 将跨文档合成与传统文档级上下文增强相结合，显著提升检索质量。

**🔧 技术方法**

使用技术包括：
- LLM（GPT‑4o、GPT‑4o‑mini、Claude 3.5 Sonnet）进行主题抽取、相关性判断与知识合成；
- OpenAI text‑embedding‑3‑large 进行主题嵌入、段落与块的向量化；
- 随机层次聚类进行主题去重；
- 语义分块与上下文增强（Anthropic 方案）做基线；
- RAGAS 框架评估 faithfulness、context precision/recall、retrieval metrics；
- 实验代码与评测在 HuggingFace 等平台实现。

**📊 数据集**

数据集：
- HotpotQA（5901 个多跳问答，检索自多篇 Wikipedia 文章）
- UAE Legal Corpus（847 份法律文件，312 条专家查询）

**📈 对比分析**

比较方法：
- 传统分块（Fixed‑Size、Recursive、Semantic）
- 行业实践的 Contextual Retrieval
- 评测指标：Faithfulness、Answer Relevancy、Context Recall/Precision、MRR、Hit@k、Citation Accuracy
- 性能结果：
  • HotpotQA 上 CDTA faithfulness 0.93（比 Contextual 12% 提升、Semantic 19% 提升、Recursive 31% 提升、Fixed‑Size 45% 提升）；
  • UAE Legal 上 CDTA faithfulness 0.94（比 Contextual 18% 提升、Semantic 27% 提升、Recursive 40% 提升、Fixed‑Size 62% 提升）；
  • 在低 k（k=3）检索时，CDTA 仅下降 2% faithfulness，保持高 Hit@1（0.88）和 Hit@3（0.96）。

**⚠️ 局限性**

限制：
- 索引阶段成本高（每文档 38–44 秒，API 费用显著）；
- 对 LLM 的依赖导致可解释性、偏见与更新延迟问题；
- 相关性映射 O(|T|×|S|) 复杂度在百万级文档上难以扩展；
- 需要手工调优主题粒度与合成长度；
- 对实时或频繁更新的语料库不友好，需要部分重合成。

---

## 51. Ensemble of radiomics and ConvNeXt for breast cancer diagnosis

**arXiv ID:** 2601.05373 | [PDF](https://arxiv.org/pdf/2601.05373v1)

**作者:** Jorge Alberto Garza-Abdala `[一作]` (Tecnologico de Monterrey), José Gerardo Tamez-Pena `[通讯]` (Tecnologico de Monterrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

建立了一个融合放射组学特征与 ConvNeXtV1‑small 深度学习模型的集成框架，用于乳腺钼靶筛查的早期诊断。

**💡 创新点**

创新点在于将手工提取的放射组学特征与端到端训练的 ConvNeXtV1‑small CNN 进行概率校准与平均融合，并在两个大规模独立数据集上采用 leave‑one‑year‑out 方法验证，显著提升了诊断性能。

**🔧 技术方法**

采用放射组学特征提取（形状、统计、GLCM、波形分解）与多模型软投票；ConvNeXtV1‑small CNN 从零训练；使用概率校准与平均融合的集成策略。

**📊 数据集**

使用 RSNA 2023 乳腺癌检测挑战（11,913 例）训练 ConvNeXt，使用 TecSalud（19,400 例）训练放射组学模型并对集成模型进行校准与验证。

**📈 对比分析**

通过 leave‑one‑year‑out 验证对比 Radiomics、ConvNeXt、Ensemble 三模型；集成模型 AUC 为 0.878（95% CI 0.859–0.897），高于 Radiomics 0.801 与 ConvNeXt 0.830；TPR 0.778，TNR 0.831，准确率 0.830。

**⚠️ 局限性**

主要局限在于使用未来年份的数据来预测过去年份导致时间偏差；未在真正独立外部数据集上进行验证；未分析各放射组学特征的具体贡献。

---

## 52. Quantifying Document Impact in RAG-LLMs

**arXiv ID:** 2601.05260 | [PDF](https://arxiv.org/pdf/2601.05260v1)

**作者:** Armin Gerami `[一作]` (University of Maryland), Ramani Duraiswami `[通讯]` (University of Maryland)

**通讯引用:** 9327 | [OpenAlex ID](https://openalex.org/A5013222310)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种基于部分信息分解（PID）和语义熵的影响评分（Influence Score, IS），用于量化检索文档对 Retrieval Augmented Generation (RAG) 生成结果的贡献。

**💡 创新点**

创新点在于：①首次将 PID 与语义熵结合构造可量化的文档影响指标；②通过 IS 实现对恶意文档的快速定位和对文档重要性的精准排序；③验证了 IS 在毒性攻击识别和消融实验中的有效性，提升了 RAG 的可解释性与安全性。

**🔧 技术方法**

使用技术包括：部分信息分解 (PID)、语义熵 (Semantic Entropy) 计算、基于 GPT‑4、Llama‑3.3‑70b、DeepSeek‑R1 的大语言模型推理、以及多轮生成与相似度聚类。

**📊 数据集**

数据集：HotPotQA、Natural Questions、MS MARCO，均用于检索五篇最相关文档并构造实验。

**📈 对比分析**

比较方法：在毒性攻击实验中，IS 将恶意文档排名最高的成功率为 86%（高于提示工程的 83%），在消融实验中人工和 GPT‑4 评审分别以 93% 及 95% 的比例更倾向于使用最高 IS 文档生成的响应，表明 IS 能准确挑选最具影响力的文档。

**⚠️ 局限性**

局限：计算成本显著——每篇文档需要两次 LLM 调用（单独与其余文档的上下文），总计 2k+1 次调用；对大规模检索集合扩展性有限，且对语义熵估计的稳定性仍需进一步研究。

---

## 53. Jailbreaking Large Language Models through Iterative Tool-Disguised Attacks via Reinforcement Learning

**arXiv ID:** 2601.05466 | [PDF](https://arxiv.org/pdf/2601.05466v1)

**作者:** Zhaoqi Wang `[一作]` (Beijing Institute of Technology), Yong Liu `[通讯]` (Qi-AnXin Technology Group Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为iMIST的交互式多步骤工具伪装越狱攻击方法，利用LLM的函数调用接口隐藏恶意意图并通过强化学习迭代优化对话，最终获得高危害输出。

**💡 创新点**

创新点在于：①将恶意查询改写为合法的工具调用请求，从而绕过内容过滤；②设计交互式进步优化框架，使用PPO强化学习根据实时危害评估动态调整攻击策略；③将工具调用链与自然语言重构结合，形成可持续提升危害度的多轮对话流程。

**🔧 技术方法**

技术手段包括：工具调用（Function Calling）接口、强化学习（PPO）进行交互式优化、JADES与StrongREJECT评估模型、LLM-based 判别器（LlamaGuard、ShieldLM、Perplexity）做检测，辅以对话历史管理与工具集动态重构。

**📊 数据集**

使用的公开数据集为 HarmfulQA（50条有害问答）和 JBB（100条有害行为样本），并在三大公开LLM上进行实验：DeepSeek‑V3、Qwen3‑32B 与 GPT‑OSS‑120B。

**📈 对比分析**

与基线、ArtPrompt、FlipAttack、PAIR、TAP、DRA、WordGame、SATA、RL‑Jack、PASS 等十余种黑盒攻击方法对比，iMIST 在所有模型和数据集上都取得最高的 JADES 与 StrongREJECT 分数，拒绝率（RR）与检测率（LlamaGuard/ShieldLM）均低于或接近对手，表明既具攻击效果又具伪装性。

**⚠️ 局限性**

局限性包括：①仅在支持函数调用的LLM环境下有效；②依赖对目标模型黑盒交互，若模型采用更强的对话监控或工具调用审计可能失效；③实验仅覆盖部分开源模型，缺乏对商业闭源LLM的验证；④攻击成功后仍需将工具参数重构为文本，若重构规则被发现亦会被阻断。

---

## 54. Mean Field Analysis of Blockchain Systems

**arXiv ID:** 2601.05417 | [PDF](https://arxiv.org/pdf/2601.05417v1)

**作者:** Yanni Georghiades `[一作]` (University of Texas at Austin), Sriram Vishwanath `[通讯]` (University of Texas at Austin)

**通讯引用:** 12605 | [OpenAlex ID](https://openalex.org/A5088120102)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文构建了一个基于均值场游戏和POMDP的框架，用来分析和求解区块链共识机制的均衡行为，并给出对网络延迟与PoW效率的精确权衡。

**💡 创新点**

创新点在于：①首次用均值场逼近将复杂的POSG化简为可求解的POMDP；②在此框架下严格证明最长链规则（LCR）在Nakamoto型区块链中是唯一最优的均衡策略；③提供了可扩展的图同构化方法，显著降低状态空间。

**🔧 技术方法**

核心技术包括：均值场游戏、部分可观测马尔可夫决策过程（POMDP）、全可观测值近似、迭代最佳响应学习、图同构与动态规划、以及对冲击传播的概率建模。

**📊 数据集**

实验使用的是仿真数据，基于设定的系统参数（N=1000、M=5、α=0.001、δ≈0.01、γ=0.99、ϵ=0.01 等）构造的随机区块生成与传播过程；并未使用公开区块链数据集。

**📈 对比分析**

通过与理论公式 1/(1+αΔ) 对比，验证了PoW效率与网络延迟的关系；利用穷举搜索验证LCR在4区块图上达到最高PoW效率；实验规模可达 7 区块（约 5.4 万状态），在此范围内求解速度可接受。

**⚠️ 局限性**

局限性包括：①均值场假设需大量均匀算力矿工；②模型仅针对Nakamoto型PoW链，对PoS、DAG等结构的适用性需进一步扩展；③穷举搜索在区块图规模大于 4 时不可行，导致对更大网络的验证依赖近似；④结果对初始策略敏感，实际矿工多样性未能完全捕获。

---

## 55. ROAP: A Reading-Order and Attention-Prior Pipeline for Optimizing Layout Transformers in Key Information Extraction

**arXiv ID:** 2601.05470 | [PDF](https://arxiv.org/pdf/2601.05470v1)

**作者:** Tingwei Xie `[一作]`, Yonghong Song `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ROAP（Reading‑Order and Attention‑Prior）轻量级管道，用以在多模态文档理解模型中显式建模阅读顺序并抑制视觉模态干扰。

**💡 创新点**

创新点包括：① Adaptive‑XY‑Gap Tree (AXG‑Tree) 用于鲁棒提取复杂布局的阅读序列；② Reading‑Order‑Aware Relative Position Bias (RO‑RPB) 将逻辑顺序编码进自注意力；③ Textual‑Token Sub‑Block Attention Prior (TT‑Prior) 通过可调的先验抑制视觉噪声。

**🔧 技术方法**

技术手段包括层次聚类与自适应阈值分割（AXG‑Tree）、基于阅读顺序的离散相对偏置学习（RO‑RPB）、自适应长度匹配的卷积先验注入（TT‑Prior）以及在现有布局 Transformer 之上的无架构改动插件化集成。

**📊 数据集**

实验使用 FUNSD 与 CORD 两大视觉丰富文档基准，分别在实体识别（SER）和关系抽取（RE）任务上评估。

**📈 对比分析**

与 LayoutLMv3、GeoLayoutLM 等前沿模型比较，ROAP 在 FUNSD 上 SER 的 F1 从 0.9029 提升至 0.9150（+1.4%），在 CORD 上提升至 0.9799（+1.2%）；在 RE 任务上也分别提升 1.1%–1.2% 的 F1。整体表现显示显著提升，尤其在含视觉模态的模型上效果更突出。

**⚠️ 局限性**

局限性包括：① 对文本‑仅模型提升有限；② 依赖 AXG‑Tree 的阅读顺序提取，复杂表格或手写区域仍可能失真；③ 需要额外的计算与调参来确定阈值与先验长度匹配。

---

## 56. LIDL: LLM Integration Defect Localization via Knowledge Graph-Enhanced Multi-Agent Analysis

**arXiv ID:** 2601.05539 | [PDF](https://arxiv.org/pdf/2601.05539v1)

**作者:** Gou Tan `[一作]` (Sun Yat-sen University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 29512 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多代理框架 LIDL，用于定位 LLM 集成软件中的缺陷。

**💡 创新点**

创新点包括：①将源代码与 LLM 交互的提示、配置、工具调用等异构文件融入同一知识图谱并做语义标注；②融合运行时信号、LLM 推断和语义检索三种证据；③采用对比推理（counterfactual reasoning）验证候选缺陷，区分根因与症状。

**🔧 技术方法**

技术手段：Tree‑sitter 代码解析构造图谱；正则+LLM 进行文件标注；BM25+k‑hop BFS 扩展候选文件；LLM（如 kimi‑k2）完成症状推断与对比验证；子图上下文构造、计分及自适应排序；Token‑efficient 提示设计。

**📊 数据集**

数据集：146 个真实缺陷实例，来自 105 个 GitHub 仓库和 16 个基于 Agent 的系统，综合 Hydrangea 与 AgentIssue‑Bench 公开数据。

**📈 对比分析**

与 5 个基线（SWE‑agent、Agentless、AutoCodeRover、SWE‑agent*、Agentless*）在 6 个 LLM 后端进行对比。LIDL 的 Top‑3 达 0.64、MAP 0.48，较最佳基线 AutoCodeRover 提升 64.1%；成本平均仅 0.008 美元/实例，比 AutoCodeRover 低 92.5%，输出 Token 下降 97.6%。

**⚠️ 局限性**

局限性：目前仅支持 Python；模式库依赖主流 LLM 框架，其他自定义或罕见框架的覆盖率低；成本随 LLM 价格波动；未在工业级多语言代码库上验证。

---

## 57. ReasonAny: Incorporating Reasoning Capability to Any Model via Simple and Effective Model Merging

**arXiv ID:** 2601.05560 | [PDF](https://arxiv.org/pdf/2601.05560v1)

**作者:** Junyao Yang `[一作]` (Shanghai Artificial Intelligence Laboratory), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 9670 | [OpenAlex ID](https://openalex.org/A5023198186)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种训练‑free 的模型融合框架 ReasonAny，用来在不破坏领域专长的前提下将大型推理模型的链式思考能力加入到特定领域模型中。

**💡 创新点**

核心创新在于“对比梯度识别”（Contrastive Gradient Identification）：发现推理能力主要驻留在梯度幅值较低的参数区域，而领域知识位于梯度幅值较高的区域，并通过互斥排除（Exclusion）保证两类参数不冲突，从而避免性能崩塌。

**🔧 技术方法**

使用梯度幅值（梯度矩阵核范数）进行参数重要性评估，采用 Top‑K / Bottom‑K 选取高梯度和低梯度参数；对冲突参数执行集合差集排除；最终通过加权叠加 (λ_r, λ_t) 把两个任务向量融合到基模型。

**📊 数据集**

在安全（Safety‑Bench, HarmBench, SafeChain）、生物医学（PubMedQA, MedQA）、金融（ConvFinQA, OpenFinData）等领域进行评估，同时使用推理基准（GSM8K, Math500, AIME2024, HumanEval, LiveCodeBench）与知识基准（ARC‑E/C, MMLU, GPQA）来衡量推理与知识保持。

**📈 对比分析**

与 Linear、Task Arithmetic、TIES‑Merging、DARE、FuseLLM、LED‑Merging 等主流融合方法对比，ReasonAny 在推理任务（如 GSM8K 取得 86.28 分，接近原始推理模型 98.91%）和领域任务（如 MedQA 47.96 分、Finance 约 70 分）均保持或提升性能，且未出现灾难性干扰。

**⚠️ 局限性**

局限性：假设推理与领域知识完全分离，复杂任务中仍可能有重叠导致轻微冲突；目前仅支持两模型融合，扩展到多领域融合尚未验证；对梯度评估的计算开销高于简单平均法。

---

## 58. Generation-Based and Emotion-Reflected Memory Update: Creating the KEEM Dataset for Better Long-Term Conversation

**arXiv ID:** 2601.05548 | [PDF](https://arxiv.org/pdf/2601.05548v1)

**作者:** Jeonghyun Kang `[一作]` (Konkuk University), Harksoo Kim `[通讯]` (Konkuk University)

**通讯引用:** 724 | [OpenAlex ID](https://openalex.org/A5022865376)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了KEEM数据集，用生成式方法动态更新长会话聊天机器人的内存，既保留关键信息，又加入情感及其原因。

**💡 创新点**

创新点在于用生成式技术替代传统的累加/删除/替换操作，能够在内存更新时整合情感与因果信息，同时避免信息丢失。

**🔧 技术方法**

使用ChatGPT‑4.0进行数据生成、情感/原因注入、内存更新及验证，并通过关键词召回、NLI冲突检测等手段评估质量。

**📊 数据集**

基于韩国KMSC多会话对话数据生成KEEM，另外对比原始KMSC和CareCallmem数据集。

**📈 对比分析**

通过手工评分、关键词召回率、冲突率和在RAG/FiD/Llama2等长会话模型上的困惑度（Perplexity）评估，KEEM在情感/原因反映、内存更新准确率、召回率和冲突率方面均优于累积和操作方法，模型困惑度降低约20%或更多。

**⚠️ 局限性**

局限在于生成过程中偶尔误删无关内容、摘要替代完整对话可能导致信息遗漏，以及生成模型仍可能产生错误，需要额外验证步骤。

---

## 59. Generalized Canonical Polyadic Tensor Decompositions with General Symmetry

**arXiv ID:** 2601.05335 | [PDF](https://arxiv.org/pdf/2601.05335v1)

**作者:** Alex Mulrooney `[一作]` (University of Delaware), David Hong `[通讯]` (University of Delaware)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5063636172)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种能够同时处理任意形式对称性和通用损失函数的CP分解方法（SymGCP）。

**💡 创新点**

创新点在于将一般张量对称性（对某些子模式的对称）与General CP（GCP）损失相结合，并给出高效的梯度表达式与随机梯度近似。

**🔧 技术方法**

采用全局梯度优化（L‑BFGS‑B）和Adam随机梯度下降，利用MTTKRP与Khatri‑Rao积的张量核实现计算；同时提供稀疏采样的随机梯度公式。

**📊 数据集**

使用合成的全对称二值张量、稀疏全对称二值张量、猴子神经元协激活张量以及UCI Irvine社交网络计数张量等真实数据集。

**📈 对比分析**

与传统最小二乘CP、非对称GCP、以及对称CP进行对比；在合成实验中，Bernoulli损失下的SymGCP能更准确恢复因子（余弦相似度≥0.99）；在稀疏实验中，Adam+分层采样比L‑BFGS‑B快约50倍但精度略低；在真实数据中，SymGCP能揭示更具意义的社交群组与实验目标结构。

**⚠️ 局限性**

限制在于仍采用通用梯度下降，未充分利用对称结构进行加速；随机方法在收敛精度上略逊；缺乏在线/流式算法与更高效的正则化策略。

---

## 60. LEAPS: An LLM-Empowered Adaptive Plugin for Taobao AI Search

**arXiv ID:** 2601.05513 | [PDF](https://arxiv.org/pdf/2601.05513v1)

**作者:** Lei Wang `[一作]` (Taobao and Tmall Group of Alibaba), Haiping Hou `[通讯]` (Taobao and Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 LEAPS 框架，通过在传统电商搜索引擎的上游和下游挂接 LLM 插件，实现对话式查询的“扩展-细化”流程，解决复杂查询导致的零结果和决策过载问题。

**💡 创新点**

创新点包括：①非侵入式“扩展-细化”插件架构；②三阶段训练策略（逆数据增强、后验知识 SFT、RL 多样性）用于查询扩展；③多源异构数据与 Chain‑of‑Thought 推理相结合的相关性验证器；④基于 RL 的多样性奖励和分布式批处理实现高并发部署。

**🔧 技术方法**

使用技术主要有：LLM（如 Qwen3、Tbstar）、逆向数据增强、后验知识监督微调、RL（GRPO/GSPO/REINFORCE++）、多源数据融合（OCR、评论、销量等）、CoT 推理、批处理与自适应分页、模型蒸馏与优化。

**📊 数据集**

数据集：
- 约 2M 条淘宝商品标题与生成的用户查询对；
- 约 200K 条真实搜索日志，用于后验知识 SFT；
- 约 50K 条日志用于 RL 训练；
- 80 万条人工标注的查询‑商品相关性对（用于验证器训练与评估）。

**📈 对比分析**

评估方法：与现行生产基线、仅 SFT、不同 RL 奖励（HR、GR、ER）下的 LEAPS 进行离线对比，指标为 Hybrid/Global/Effective Relevance；在线 A/B 测试衡量 Low‑Result Rate 与 CTR。性能表现：
- 离线 HR/GR/ER 均提升 30‑40% 以上；
- 在线 LRR 从 24.88% 降至 16.98%（下降 7.9pp）；
- CTR 从 9.39% 提升至 10.93%（提升 1.54pp）。

**⚠️ 局限性**

局限性：
- 查询扩展的搜索预算分配仍为均等，未能学习不同扩展词的重要性权重；
- 相关性验证器依赖 OCR 与文本，缺乏直接视觉感知，易受 OCR 误差影响；
- 模型规模大、推理成本高，需要进一步蒸馏与量化；
- 目前仅支持文本查询，未覆盖多模态输入。

---

## 61. RECOR: Reasoning-focused Multi-turn Conversational Retrieval Benchmark

**arXiv ID:** 2601.05461 | [PDF](https://arxiv.org/pdf/2601.05461v1)

**作者:** Mohammed Ali `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结合多轮对话与推理密集检索的基准，并通过分解与验证框架生成高质量、事实检索驱动的对话数据

**💡 创新点**

创新点在于：①将复杂单轮问答拆解为多层级事实并生成显式检索推理；②创建了首个同时覆盖多轮与推理的基准，涵盖十一大领域；③通过历史+推理策略显著提升检索性能

**🔧 技术方法**

使用了分解与验证（Decomposition‑and‑Verification）框架、GPT‑4o生成查询与推理、DIVER、ReasonIR等推理专用检索模型以及BGE、E5、Contriever等稠密编码器和BM25等稀疏检索器

**📊 数据集**

基准数据来自BRIGHT（六个领域）与从StackExchange抽取的五个领域，共计707段对话、2,971轮，包含507,141篇文档（正样本2,900篇、负样本504,241篇）

**📈 对比分析**

评估使用nDCG@10和LLM‑Judge（1–5分）四个维度。结果显示：历史+推理将检索性能从0.236提升至0.479（+103%）；推理专用检索器DIVER在历史+推理下达到0.584，显著高于稠密编码器；生成质量在大模型下平均得分0.880，较小模型高出约0.06；但在隐式推理和领域专业术语上仍存在显著不足

**⚠️ 局限性**

局限性：仅覆盖英语；仅涵盖已具备问答+文档支持的十一领域，无法推广到数学、常识等尚未结构化的数据；对低质量或缺乏文档的领域效果不佳

---

## 62. Multi-Image Super Resolution Framework for Detection and Analysis of Plant Roots

**arXiv ID:** 2601.05482 | [PDF](https://arxiv.org/pdf/2601.05482v1)

**作者:** Shubham Agarwal `[一作]` (Ben Gurion University of the Negev), Jhonathan E. Ephrath `[通讯]` (Ben Gurion University of the Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了多图超分辨框架MI-DRCT，配合自研的RootCam硬件获取重叠的地下根系RGB图像，并利用提升后的高质量图像进行根毛检测与定量分析。

**💡 创新点**

创新点主要包括：① 开发了可获取重叠低分辨率视角的RootCam硬件；② 在现有单图DRCT模型上引入子像素对齐与多图融合机制，形成MI-DRCT；③ 构造了专门模拟地下根系的合成重叠图像数据集，用于高效训练；④ 通过多图超分辨显著提升根毛可见度，显著提高根毛计数与长度估计精度。

**🔧 技术方法**

技术手段包括：多图超分辨（MI-DRCT）、深度残差通道注意力变换器（DRCT）、相位相关子像素对齐、卷积特征提取、实例分割网络、BRISQUE/CLIP‑IQA等无参考图像质量评估。

**📊 数据集**

使用的数据集为：① 10,000组三张合成根系图像（训练）+2,000组验证；② 50组三张真实Bell Pepper根系图像（测试）。合成数据通过重叠裁剪、子像素位移、下采样等步骤生成。

**📈 对比分析**

实验与双线性、双三次、SwinIR、BSRGAN、单图DRCT等方法比较。指标包括MSE、PSNR、SSIM、BRISQUE、CLIP‑IQA。结果显示：合成集上MI‑DRCT取得MSE 16.31、PSNR 38.96 dB、SSIM 0.95、BRISQUE 43.13、CLIP‑IQA 0.39；真实集上BRISQUE 44.50、CLIP‑IQA 0.38，均优于其他方法；根毛检测精度从4/77提升至44/77，根毛长度、面积估计也显著提高。

**⚠️ 局限性**

局限性包括：① 合成数据对真实场景的模拟程度有限，可能影响泛化；② 仅在单一作物（Bell Pepper）验证，缺乏多作物泛化评估；③ 需要高性能GPU进行训练与推理，实时部署受限；④ 子像素对齐假设位移主要为垂直方向，可能无法处理更复杂的相机运动；⑤ 对深度网络超参数敏感，需调参。

---

## 63. Effects of personality steering on cooperative behavior in Large Language Model agents

**arXiv ID:** 2601.05302 | [PDF](https://arxiv.org/pdf/2601.05302v1)

**作者:** Mizuki Sakai `[一作]` (Shizuoka University), Genki Ichinose `[通讯]` (Shizuoka University)

**通讯引用:** 866 | [OpenAlex ID](https://openalex.org/A5080267104)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对GPT‑3.5‑turbo、GPT‑4o与GPT‑5三代大型语言模型，在重复囚徒困境游戏中先用 Big Five Inventory (BFI‑44) 量化其自然人格，再通过提示注入或极端值操控人格特质，研究人格引导对合作行为的影响。

**💡 创新点**

在定量测评基础上系统性操控人格特质，区分不同模型代之间人格对合作的调节作用，首次揭示人格“可调”与模型代差异如何共同决定合作倾向。

**🔧 技术方法**

使用 BFI‑44 人格测评、Prompt 注入人格信息、极端值操控、重复囚徒困境实验设计以及统计比较指标（合作率、累计收益）。

**📊 数据集**

BFI‑44 问卷（44题）与固定对手策略集（ALLC、ALLD、RANDOM、TFT、GRIM），无外部公开数据集。

**📈 对比分析**

通过对比基线（无人格信息）与人格信息条件下的合作率与累计收益，发现高 agreeableness 提升合作但易被利用；后代模型表现更为选择性合作，整体收益受影响程度随模型代提升而降低。

**⚠️ 局限性**

人格操控仅为行为偏置，无法完全控制合作；实验仅限囚徒困境，缺乏对多样化社会情境的验证；模型内部决策过程不透明，限制了对结果机制的深入解释。

---

## 64. Hippocampal Atrophy Patterns Across the Alzheimer's Disease Spectrum: A Voxel-Based Morphometry Analysis

**arXiv ID:** 2601.05494 | [PDF](https://arxiv.org/pdf/2601.05494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 65. MaxCode: A Max-Reward Reinforcement Learning Framework for Automated Code Optimization

**arXiv ID:** 2601.05475 | [PDF](https://arxiv.org/pdf/2601.05475v1)

**作者:** Jiefu Ou `[一作]` (Johns Hopkins University), George Karypis `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在代码优化任务中使用推理时搜索方法，并将其重新表述为最大奖励强化学习框架；

**💡 创新点**

创新点在于：①通过自然语言批判模型将原始执行反馈转化为诊断信息，丰富观测空间；②引入最大奖励回报的辅助变量u，实现基于最佳历史奖励的搜索；③训练生成式奖励-到-去模型（reward‑to‑go）来预测最大未来奖励，指导候选筛选；

**🔧 技术方法**

主要技术包括：大型语言模型（LLM）作为策略与批判器、最大奖励强化学习框架、生成式奖励预测模型、Beam Search、Effi‑Learner与CUDA‑LLM等先行搜索方法的重构；

**📊 数据集**

实验使用了KernelBench（CUDA核函数优化）和PIE（C++竞赛级优化）两个基准数据集；

**📈 对比分析**

与Effi‑Learner、CUDA‑LLM以及直接采样基线比较，本文方法在KernelBench L1/L2与PIE上均提升了最大加速比（例如KernelBench L2提升约27.3%，PIE提升约11.0%），并在不同搜索深度下展示了更快的性能提升；

**⚠️ 局限性**

局限性包括：奖励模型在高方差的速度提升场景下预测准确度有限；训练数据与实际搜索轨迹分布可能存在偏移；以及对LLM推理成本和执行环境开销的依赖。

---

## 66. MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards

**arXiv ID:** 2601.05488 | [PDF](https://arxiv.org/pdf/2601.05488v1)

**作者:** Zhiyu Shen `[一作]` (Sun Yat-sen University), Yanghui Rao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2178 | [OpenAlex ID](https://openalex.org/A5058291454)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练一个4B参数模型来构建多维外部记忆，从而提升长时对话的一致性。

**💡 创新点**

提出MemBuilder框架，结合稠密会话级奖励和贡献感知梯度加权的ADRPO方法，解决稀疏奖励和多维记忆归因问题。

**🔧 技术方法**

使用强化学习（ADRPO）、自监督微调、稠密奖励生成、贡献感知梯度加权、检索增强问答等技术。

**📊 数据集**

在LongMemEval上进行训练，并在LoCoMo、PerLTQA等OOD基准上进行评估。

**📈 对比分析**

与RAG、提示式框架（Mem0、MIRIX）以及基于RL的Memory-R1等基线比较，Qwen3-4B在LoCoMo上取得84.23%（比MIRIX高6.75pp），在其他基准也显著领先。

**⚠️ 局限性**

评估依赖封闭源模型（Claude 4.5 Sonnet）和合成QA生成的质量；模型训练和推理成本仍高，且在某些非训练域的泛化仍有限。

---

## 67. Lost in Execution: On the Multilingual Robustness of Tool Calling in Large Language Models

**arXiv ID:** 2601.05366 | [PDF](https://arxiv.org/pdf/2601.05366v1)

**作者:** Zheng Luo `[一作]` (University of Southern California), Xiyang Hu `[通讯]` (Arizona State University)

**通讯引用:** 769 | [OpenAlex ID](https://openalex.org/A5044665455)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并诊断多语言环境下大型语言模型的工具调用鲁棒性，发现参数值语言不匹配导致执行错误，并尝试推理时缓解策略。

**💡 创新点**

提出 MLCL 诊断基准和细粒度错误分类，将参数值语言不匹配定位为主导失效模式，并系统探究多语言工具调用的接口瓶颈。

**🔧 技术方法**

使用结构化函数调用、翻译预/后处理、提示式指令等推理时技术，以及对 BFCL V4 扩展的多语言语料进行细粒度错误分析。

**📊 数据集**

在 Berkeley Function Calling Leaderboard (BFCL) V4 单轮任务基础上，生成中文、印地语、伊博语的翻译与语义扰动版本。

**📈 对比分析**

采用严格的函数名、参数键值匹配评估，细分错误率；发现完全翻译导致参数值错误显著升高，部分翻译或预翻译可显著降低错误但仍低于英语基准。

**⚠️ 局限性**

仅评估单轮任务、语言样本有限、假设接口全为英文、推理时缓解措施简单、模型范围有限。

---

## 68. TAPM-Net: Trajectory-Aware Perturbation Modeling for Infrared Small Target Detection

**arXiv ID:** 2601.05446 | [PDF](https://arxiv.org/pdf/2601.05446v1)

**作者:** Hongyang Xie `[一作]` (University of Borsetshire), Victor Sanchez `[通讯]` (Collaborators, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于扰动引导轨迹的 Mamba 传播网络 TAPM-Net，用于红外小目标检测。

**💡 创新点**

首次将扰动能量场与轨迹感知状态空间建模结合，利用物理扩散思想实现方向性特征传播，并实现了低计算成本的高效检测。

**🔧 技术方法**

核心技术包括扰动能量场（PGM）、轨迹引导路径模块、Mamba 状态空间块（TASB）、U‑Net 编码解码器以及多尺度特征融合。

**📊 数据集**

在 NUAA‑SIRST 与 IRSTD‑1K 两大红外小目标检测基准集上进行实验。

**📈 对比分析**

与传统 CNN、ViT 及混合模型对比，TAPM-Net 在 IoU、nIoU、P_d 等指标上均取得 SOTA，检测率达到 100% 且误报率极低。

**⚠️ 局限性**

对轨迹采样长度、能量阈值等超参较为敏感，且在极低对比度或极大遮挡场景下仍可能出现漏检。

---

## 69. Multi-User Covert Communications via Intelligent Spectrum Control

**arXiv ID:** 2601.05281 | [PDF](https://arxiv.org/pdf/2601.05281v1)

**作者:** Yujie Ling `[一作]` (Xidian University), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 80345 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文在多小区多用户环境下提出一种智能频谱控制（ISC）方案，用于实现低可检测性的安全通信，提供可靠的数据传输；

**💡 创新点**

创新点在于将高精度谱感知与基于AI的实时决策结合，动态生成时频占用模式，主动规避干扰和共信道冲突，并推导了多用户联合检测下的检测误差概率（DEP）与可靠传输概率（RTP），进而优化传输功率和用户接入容量；

**🔧 技术方法**

技术包括CNN/SVM谱感知、深度双DQN调度、联合能量检测、Gamma分布与超几何函数分析、功率约束下的可靠率/可检测率优化；

**📊 数据集**

使用模拟Rayleigh衰落信道数据，设定4个基站、500 MHz总带宽、L=8、q=64、单个干扰器和被动窃听器，未采用公开数据集；

**📈 对比分析**

与AN‑aided OFDM基准方案比较，实验显示ISC方案在相同SNR下实现更高的DEP、更大的RTP以及更大的可接入用户数，证明了在覆盖率与可靠性约束下的优越性；

**⚠️ 局限性**

局限性包括：仅考虑单一被动窃听器、固定信道统计分布、未涉及协同干扰器、实验仅为仿真，缺乏真实部署验证，且对极端干扰场景的鲁棒性尚待进一步研究。

---

## 70. Semi-Supervised Facial Expression Recognition based on Dynamic Threshold and Negative Learning

**arXiv ID:** 2601.05556 | [PDF](https://arxiv.org/pdf/2601.05556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 71. An Empirical Study of Policy-as-Code Adoption in Open-Source Software Projects

**arXiv ID:** 2601.05555 | [PDF](https://arxiv.org/pdf/2601.05555v1)

**作者:** Patrick Loic Foalem `[一作]` (Polytechnique Montreal), Ettore Merlo `[通讯]` (Polytechnique Montreal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对399个GitHub OSS仓库进行大规模实证研究，分析了9种Policy-as-Code工具的使用情况、项目类型及治理目的，并构建了5大治理类别的分类体系。

**💡 创新点**

首次在开源生态中系统评估PaC工具采纳与治理用例，结合LLM自动分类与专家验证构建实证可复现的治理分类体系。

**🔧 技术方法**

使用GitHub API抓取、正则模式识别、LLM-assisted 分类、双人编码一致性评估、统计与可视化分析等技术。

**📊 数据集**

399个GitHub开源仓库（共12,152份PaC文件），涵盖OPA、Kyverno、Gatekeeper、Pulumi、Cloud Custodian等9种工具。

**📈 对比分析**

通过文件计数、仓库覆盖率、共用矩阵和Cohen’s Kappa评估方法，LLM分类与人工校验一致性高（Kappa≈0.84），OPA最普遍、其他工具多为单一使用，整体表现良好。

**⚠️ 局限性**

研究仅覆盖GitHub OSS，过滤条件排除小/无活跃项目，工具识别依赖模式可能产生误报，文件级分析忽略多重治理维度，未涉及企业私有或专有工具，LLM可能产生hallucination。

---

## 72. KP-Agent: Keyword Pruning in Sponsored Search Advertising via LLM-Powered Contextual Bandits

**arXiv ID:** 2601.05257 | [PDF](https://arxiv.org/pdf/2601.05257v1)

**作者:** Hou-Wan Long `[一作]` (Chinese University of Hong Kong), Tianshu Sun `[通讯]` (Cheung Kong Graduate School of Business)

**通讯引用:** 747 | [OpenAlex ID](https://openalex.org/A5015099727)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于 LLM 的关键词精简方法（KP-Agent），通过上下文带分（contextual bandit）框架实现广告关键词的动态裁剪，提升广告收益。

**💡 创新点**

创新点在于：①将 LLM 与领域专用工具集和长记忆模块结合，实现无幻觉的表格处理与少量样本自我反思；②把关键词精简建模为带上下文的 bandit 问题，使用自生成代码实时执行裁剪决策；③仅依赖广告主侧 KPI 数据，无需用户搜索词。

**🔧 技术方法**

使用 GPT‑4.1 作为 LLM 引擎，配合 Contextual Bandit 策略、代码生成与调试工具、记忆检索与反思模块，并通过代码执行实现关键词裁剪。

**📊 数据集**

数据集来源于 Meituan 21 天内的药品广告 SSA 记录，约 0.5 M 条记录，45 个广告系列、278 个关键词，包含曝光、点击、转化及成本信息。

**📈 对比分析**

在模拟实验中与四种基线方法（Impression‑Rank、CTR‑Rank、CVR‑Rank、Impression Regression）对比，累计利润提升范围从 2.46 % 至 49.28 %，尤其在关键词保留阈值较低时差距更大。

**⚠️ 局限性**

主要局限：仅在历史数据模拟中验证，缺乏真实 A/B 测试；对预算分配假设过于简化；工具集与记忆依赖预先定义，可能在不同平台或广告场景下需要重新调整。

---

## 73. GlyRAG: Context-Aware Retrieval-Augmented Framework for Blood Glucose Forecasting

**arXiv ID:** 2601.05353 | [PDF](https://arxiv.org/pdf/2601.05353v1)

**作者:** Shovito Barua Soumma `[一作]` (Arizona State University), Hassan Ghasemzadeh `[通讯]` (Arizona State University)

**通讯引用:** 4585 | [OpenAlex ID](https://openalex.org/A5007139473)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种名为 GlyRAG 的基于大语言模型的上下文感知、检索增强血糖预测框架，能够仅利用 CGM 数据自动提取语义上下文并进行长时段预测。

**💡 创新点**

创新点包括：① 用 LLM 生成血糖波形的可解释文本摘要并作为上下文输入；② 在多模态 Transformer 中融合文本与血糖嵌入，并通过跨模态翻译损失实现语义对齐；③ 在推理时检索相似历史片段并通过跨注意力实现案例推理，提升预测的稳定性和临床可靠性。

**🔧 技术方法**

采用 GPT‑4 或类似 LLM 生成文本、BERT 进行文本编码、PatchTST 风格的糖浓度补丁嵌入、跨模态 Transformer、自监督的跨翻译损失以及检索增强的 RAG 模块，预测头为 MLP 或小型 LSTM。

**📊 数据集**

在两大真实 T1D 数据集上验证：OhioT1DM（12 例，Medtronic CGM+胰岛素记录）和 AZT1D（25 例，Dexcom G6+Tandem Pump）。

**📈 对比分析**

与 LSTM、MTL‑LSTM、TimesFM、GluNet 等 SOTA 方案以及仅使用 BGL 的基线相比，GlyRAG 在 5、30、60 分钟预测范围内的 RMSE 下降最多 39%（对比最优方法）并比 BGL‑only 基线低 1.7%，临床指标如 Clarke Error Grid 区域 A–B 占比高达 85%，并显著提升低血糖和高血糖事件的检出率。

**⚠️ 局限性**

局限性包括：仅评估 T1D 病人，缺乏 T2D 或前驱糖尿病样本；数据集规模有限且为回溯式；LLM 未针对糖尿病语料进行微调，可能导致领域不匹配；检索库仅基于训练时嵌入，未考虑隐私保护或在线更新；未进行前瞻性临床试验，缺乏实时交互与安全性验证。

---

## 74. Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions

**arXiv ID:** 2601.05414 | [PDF](https://arxiv.org/pdf/2601.05414v1)

**作者:** Minda Zhao `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**通讯引用:** 2511 | [OpenAlex ID](https://openalex.org/A5100632182)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并系统化审计LLM的原生概率采样能力，覆盖11个模型、15种分布，使用两种采样协议并探讨对下游生成任务的影响。

**💡 创新点**

首次大规模、统计上有力的采样评测，提出批量与独立请求双协议以区分自我校正与固有偏差，并将采样失效演绎到MCQ与图像提示生成的公平性。

**🔧 技术方法**

采用Wasserstein‑1、KL散度与KS/Chi‑square检验对采样进行量化；通过批量生成与独立调用协议；在下游任务中使用位置分布检验与属性分布检验。

**📊 数据集**

使用人工构造的15种标准概率分布样本（N_ref=1000），以及自生成的1000条MCQ与1000条属性约束提示，用来对照统计检验。

**📈 对比分析**

与参考高精度采样对比，统计显著性检验显示批量模式仅有约13‑40%通过率，独立请求几乎全部失败；随着分布复杂度升高通行率下降、Wasserstein距离上升；在下游任务中出现显著位置和属性偏差。

**⚠️ 局限性**

仅评估已知分布的显式采样，未覆盖隐式或动态分布；缺乏理论证明，结果可能随模型架构演进变化；下游偏差分析有限，未做公平性全面评估。

---

## 75. Studying Illustrations in Manuscripts: An Efficient Deep-Learning Approach

**arXiv ID:** 2601.05269 | [PDF](https://arxiv.org/pdf/2601.05269v1)

**作者:** Yoav Evron `[一作]` (Ben-Gurion University of the Negev), Michael Fire `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 1525 | [OpenAlex ID](https://openalex.org/A5081035294)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个三阶段的深度学习流水线，用于自动识别、定位并生成数字化手稿中的插画描述，并在Vatican Library与Golden Haggadah的大规模图像上进行了验证。

**💡 创新点**

创新点在于将页面级分类、目标检测与视觉语言描述模块化组合，采用轻量级模型实现每页<0.06秒的推理时间，显著提升了传统像素分割方法在海量文档中的可扩展性和效率。

**🔧 技术方法**

使用EfficientNet-B0进行页面分类、YOLOv11n进行插画定位、LLaVA模型生成文本描述，辅以迁移学习、数据增强、IIIF API抓取等技术。

**📊 数据集**

数据集包括约20,000张手标手稿页面（其中约5.8%为插画）以及覆盖Vatican Library的3,000,000+页和Golden Haggadah的101张卷页，用于训练、验证、测试与大规模应用。

**📈 对比分析**

与基线的像素级分割方法(docExtractor)相比，分类模型ROC‑AUC为0.95，精度78.6%、召回74.6%；检测模型mAP为75.6%，精度51.2%、召回78.7%；检测速度0.06秒/页，远快于基线的51秒/页；LLaVA在100幅插画上的生成准确率超过75%。

**⚠️ 局限性**

局限性包括：训练样本有限，罕见风格或损坏页表现差；检测精度仅55%，部分误检为子插画；caption在细节、符号和宗教人物识别上存在误差；缺乏对不同文化/语言手稿的充分泛化验证。

---

## 76. STResNet & STYOLO : A New Family of Compact Classification and Object Detection Models for MCUs

**arXiv ID:** 2601.05364 | [PDF](https://arxiv.org/pdf/2601.05364v1)

**作者:** Sudhakar Sah `[一作]` (STMicroelectronics), Ravish Kumar `[通讯]` (STMicroelectronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了两类面向 MCU/NPU 的轻量级网络——STResNet（分类）和 STYOLO（目标检测），并通过压缩和 NAS 优化实现低参数、低延迟、低内存占用。

**💡 创新点**

创新点在于将 ResNet 通过低秩层分解与 NAS 引导的通道压缩（CompressNAS）结合，得到完全基于标准卷积的轻量化模型；同时将预训练的 STResNet backbone 直接嵌入 YOLOX 框架，配合层级学习率和投影层实现更高的效率与精度。

**🔧 技术方法**

主要技术包括 Tucker 分解、整数线性规划（ILP）搜索、Zero‑Cost（MSE）精度估计、层级学习率调度、投影层减少 RAM、以及在 STM32N6 NPU 上的固件层性能评测。

**📊 数据集**

使用 ImageNet‑1K 训练分类模型，使用 MS‑COCO 训练检测模型，并在 STM32N6 Neural Art NPU 上进行真实硬件基准。

**📈 对比分析**

与 MobileNetV1/V2、ShuffleNet、EfficientNet、YOLOv5n、YOLOv8n 等现有轻量模型对比，STResNetTiny 在 4 M 参数内 Top‑1 达 71.6%，STResNetMilli 70.0%；STYOLOMicro 在 1.69 M 参数内 mAP 30.54%，相较 YOLOv5n 提升 2.54 mAP，且延迟和 RAM 均优于同类模型。

**⚠️ 局限性**

局限性包括：对更大模型或更高分辨率的适配性尚未充分验证；目前仍需在多任务（分割、关键点等）上进一步测试；量化策略仍基于 INT8，混合精度与更低位宽的联合优化尚未实现。

---

## 77. Betting on Equilibrium: Monitoring Strategic Behavior in Multi-Agent Systems

**arXiv ID:** 2601.05427 | [PDF](https://arxiv.org/pdf/2601.05427v1)

**作者:** Etienne Gauthier `[一作]` (Inria), Michael I. Jordan `[通讯]` (University of California)

**通讯引用:** 178244 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于 e-value 的在线连续检测框架，用于实时监测多智能体系统中的均衡偏离。

**💡 创新点**

创新点在于将 e-value 与 Benjamini–Hochberg FDR 控制结合，统一处理 Nash、相关与粗相关均衡，并扩展到随机博弈且不需要预先知道均衡策略。

**🔧 技术方法**

采用超马丁格、e-value 投注、KL 散度、混合 e-value、e-BH 多重检验以及似然比检验等技术。

**📊 数据集**

实验数据来源于 2×2 均衡游戏、5×4 网格足球与 10×10 捕食-猎物网格世界的仿真。

**📈 对比分析**

与传统单一阈值 FWER 方法相比，FDR 方法在大规模信号下显著加快检测速度，实验验证检测时间呈 1/ε² 缩放。

**⚠️ 局限性**

局限性包括需观测所有反事实收益或已知目标策略，以及对学习主体自适应反馈机制尚未研究。

---

## 78. Efficient Inference for Noisy LLM-as-a-Judge Evaluation

**arXiv ID:** 2601.05420 | [PDF](https://arxiv.org/pdf/2601.05420v1)

**作者:** Yiqun T Chen `[一作]`, Shengyi Li `[通讯]` (Johns Hopkins University)

**通讯引用:** 3161 | [OpenAlex ID](https://openalex.org/A5091667095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了在 LLM‑as‑a‑judge 评估中利用有限人类标注校准 LLM 判断结果的两类校正方法（测量误差校正和 PPI），并基于半参数效率理论统一它们，提出了高效的 EIF 估计器和与其等价的优化 PPI 形式，并在模拟与真实数据上验证其优越性。

**💡 创新点**

创新点在于：① 将测量误差校正与 PPI 两种思路通过效率影响函数统一起来，给出 EIF 估计器；② 证明在二值结果下，优化参数 λ 的 PPI 与 EIF 等价，并与 MLE 等效；③ 对比三种校正方法（RG、PPI、EIF/MLE）在不同标注比例、判定准确率与样本量下的偏差、覆盖率与区间宽度，明确了 EIF/PPI 的效率优势；④ 提出了实例相关误差、聚合多评判器以及分布偏移等未来扩展方向。

**🔧 技术方法**

主要技术包括：半参数效率理论（效率影响函数）、误分类模型（Rogan–Gladen）、预测驱动推断（PPI）及其参数调优、最大似然估计、非参数回归（GAM/样条）以及模拟与置信区间构造（logit 变换）。

**📊 数据集**

数据集：① 1000 次模拟（二值 Y 与不同 q0,q1、标注比例）；② 真实人类偏好数据 Chatbot Arena（Claude‑Opus 4 对 Gemini 2.5 Flash、Gemini 2.5 Pro、Qwen3‑235B 共 3 组模型对），使用 GPT‑4o‑mini 与 GPT‑5.2 两个 LLM 判定器。

**📈 对比分析**

比较方法：在相同标注比例、判定准确率和样本量下，计算偏差、覆盖率、平均置信区间宽度；结果显示：① EIF（或优化 PPI）与 MLE 产生最小方差、覆盖率接近 90% ；② RG 产生宽阔区间且保守；③ PPI 较宽但覆盖率略高；④ 传统无校正的 naive 估计严重偏差并且覆盖率低；在真实数据中，EIF/PPI/MLE 置信区间约为 RG 的 1/3‑1/4，且保持 90% 覆盖率。

**⚠️ 局限性**

局限性：① 仅针对平均参数，未针对更复杂指标（如分位数、方差等）；② 仅考虑独立同分布与 MCAR 条件，未解决实例依赖误差或分布漂移；③ 在多评判器、连续/多类别标签的通用性需要进一步研究；④ 计算复杂度与大规模部署时的可扩展性尚未评估。

---

## 79. Efficient Differentiable Causal Discovery via Reliable Super-Structure Learning

**arXiv ID:** 2601.05474 | [PDF](https://arxiv.org/pdf/2601.05474v1)

**作者:** Pingchuan Ma `[一作]` (Zhejiang University of Technology), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 97310 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于稀疏+低秩分解的精度矩阵学习方法，用以构建可靠的超结构，并将该超结构作为硬约束引入梯度优化的可微因果发现流程中；

**💡 创新点**

将稀疏与低秩分解、ADMM求解、超结构硬约束投影与可微因果发现算法深度耦合，并提供理论证明超结构一定包含真因果图；

**🔧 技术方法**

采用图拉索（Graphical Lasso）与其扩展的稀疏+低秩分解（SVT + 软阈值），通过ADMM迭代求解；随后利用硬约束投影（mask）与L‑BFGS或其它梯度优化器完成可微因果结构学习；

**📊 数据集**

在合成数据上使用Erdős‑Rényi、尺度自由与二部图，变更节点数、平均度、样本量和噪声分布；在真实数据上使用Sachs细胞计数数据；

**📈 对比分析**

与基线可微方法（NOTEARS、GOLEM、DAGMA、ABIC）以及GLasso、LVGL、SL等超结构学习策略对比；结果显示平均F1提升≈3.3%，跑时减少≈50–70%，在多种图类型、度数、样本量与噪声类型下均保持显著优势；在Sachs数据上F1从0.31提升到0.43；

**⚠️ 局限性**

受限于线性高斯假设，尽管对非高斯噪声有一定鲁棒性，但在高密度图、极大样本量或离散/非线性模型下提升有限；阈值选择对超结构召回影响大，误设可能导致漏边；实现仍需额外的ADMM与SVD计算，规模极大时开销不容忽视。

---

## 80. SP-Rank: A Dataset for Ranked Preferences with Secondary Information

**arXiv ID:** 2601.05253 | [PDF](https://arxiv.org/pdf/2601.05253v1)

**作者:** Hadi Hosseini `[一作]` (Penn State University), Amrit Puhan `[通讯]` (Penn State University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍了SP‑Rank数据集，并在该数据集上评估了同时利用第一阶投票和第二阶预测的排序聚合方法。

**💡 创新点**

首次公开大规模包含投票与元预测的公开数据集，并通过SP‑Voting展示了第二阶信息显著提升聚合性能。

**🔧 技术方法**

采用SP‑Voting算法、传统投票规则（Borda、Copeland、Maximin）以及基于 Mallows 和 Plackett‑Luce 的概率模型进行实验。

**📊 数据集**

12,384 条样本，覆盖地理、电影、绘画三大领域，并在 Amazon Mechanical Turk 上收集。

**📈 对比分析**

将 SP‑Voting 与传统投票规则在全局与子集排名恢复、Kendall Tau 指标下对比，结果表明 SP‑Voting 平均提升 0.4–0.8 的 Tau，尤其在低信息格式下表现突出。

**⚠️ 局限性**

数据仅覆盖三领域、来源于 MTurk 的单一文化背景、假设存在唯一真实排名，限制了跨域与多元文化的泛化。

---

## 81. A Survey of Agentic AI and Cybersecurity: Challenges, Opportunities and Use-case Prototypes

**arXiv ID:** 2601.05293 | [PDF](https://arxiv.org/pdf/2601.05293v1)

**作者:** Sahaya Jestus Lazer `[一作]` (Tennessee Tech University), Elisa Bertino `[通讯]` (Purdue University)

**通讯引用:** 39534 | [OpenAlex ID](https://openalex.org/A5061694501)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了 Agentic AI（具备规划、行动、记忆和自适应的人工智能）在网络安全中的应用与挑战，并提供了三种代表性用例原型。

**💡 创新点**

创新点在于：① 把 Agentic AI 的核心能力（规划、工具调用、记忆、协同）与 NIST 网络防御生命周期对齐；② 系统性梳理 Agentic AI 对防御与攻击双重用途的影响；③ 将多层安全模型与治理框架与实际用例结合，提出了未来研究方向。

**🔧 技术方法**

主要技术包括大语言模型（LLM）与函数调用、检索增强生成（RAG）、多智能体协作、工具调用接口、向量数据库记忆等。

**📊 数据集**

由于是综述论文，未使用单一实验数据集；引用了多篇现有研究与案例（如 RedTeamLLM、ARCeR、Cyberwheel 等）来说明技术与实践。

**📈 对比分析**

本文通过对比已有的安全框架（如 ATFAA、SHIELD、MAESTRO、OWASP Agentic Security Initiative）以及对 Agentic AI 攻击与防御案例的实证比较，指出现有方法在安全性、可解释性、治理等维度的表现与局限，并提出了更完善的评估与治理建议。

**⚠️ 局限性**

主要局限包括：① 综述性质缺乏统一实验评估与基准；② 关注的安全风险与治理框架仍不完整，缺少跨域统一规范；③ 对 Agentic AI 在实际网络环境中的长期行为、协同攻击与治理交互尚未充分验证。

---

## 82. Simulation-Free PSRO: Removing Game Simulation from Policy Space Response Oracles

**arXiv ID:** 2601.05279 | [PDF](https://arxiv.org/pdf/2601.05279v1)

**作者:** Yingzhuo Liu `[一作]` (Beijing University of Posts and Telecommunications), Zhaofeng He `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1002 | [OpenAlex ID](https://openalex.org/A5101869968)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于动态窗口的无游戏模拟 PSRO（SF-PSRO），通过限制策略集大小、记录最佳响应训练过程中的对局结果来构造近似元支付矩阵，并用 Nash 聚类挑选被淘汰策略，从而显著降低计算成本并提升性能。

**💡 创新点**

创新点包括：① 将游戏模拟完全去除，仅利用最佳响应训练时的交互信息构造“草图”元支付矩阵；② 引入动态窗口机制限制策略数，解决传统 PSRO 中策略集过大导致的对手选择和最佳响应训练效率低下问题；③ 在窗口内采用 Nash 聚类精确挑选最弱策略进行剔除，避免随机或简单平均导致的误淘汰。

**🔧 技术方法**

主要技术包括：PSRO 框架、最小后悔约束配置（MRCP）作为元策略求解器、行为多样性（BD）正则化的最佳响应求解、利用经验对局结果更新草图元支付矩阵、Nash 聚类与消除步骤、以及策略窗口的动态维护。

**📊 数据集**

实验数据集涵盖 Leduc Poker（两人）和 Goofspiel（两人、三人）等两阶段或多阶段的博弈环境，使用 OpenSpiel 实现。

**📈 对比分析**

与 Vanilla PSRO、PSD‑PSRO、Anytime PSRO、Vanilla Self‑Play、Fictitious Self‑Play 等基线比较，结果显示在 Leduc Poker、Goofspiel（2人）和 Goofspiel（3人）中，动态窗口 SF‑PSRO 在相同或更短的运行时间内达到或超过基线的 exploitability / TrueSkill 指标，且常位于性能-时间 Pareto 前沿。

**⚠️ 局限性**

局限性包括：① 需要手动调节窗口大小，过小或过大均影响最终性能；② 草图元支付矩阵不如完整模拟得到的矩阵准确，导致最终策略选择可能不完全最优；③ 对更复杂游戏的适用性与可调参成本待进一步验证。

---

## 83. Separating Semantic Expansion from Linear Geometry for PubMed-Scale Vector Search

**arXiv ID:** 2601.05268 | [PDF](https://arxiv.org/pdf/2601.05268v1)

**作者:** Rob Koopman `[一作]` `[通讯]` (Independent Researcher), Rob Koopman (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个PubMed规模的检索框架，先用大语言模型将查询扩展成精炼的医学短语，再在固定的、无偏、近似等距的Johnson–Lindenstrauss投影空间中对文档与查询进行余弦距离检索，不使用任何训练好的嵌入模型。

**💡 创新点**

将语义解释与几何距离完全分离，利用确定性LLM扩展获得查询语义，然后在无参数的线性投影空间完成检索，证明在PubMed规模下无需学习的嵌入模型也能获得高质量检索。

**🔧 技术方法**

使用的大语言模型进行查询扩展；词向量平均化与均值去除；Johnson–Lindenstrauss随机投影；int8压缩向量；exact cosine kNN搜索；max‑dot交叉注意力重排序。

**📊 数据集**

MEDLINE完整语料（约3.96 千万条记录）以及经过过滤保留的约792 万医学词汇；对20个生物医学提示进行评估。

**📈 对比分析**

通过几何指标（head cosine、compactness、centroid closure、isotropy）与随机基线比较，head cosine≈0.68、compactness≈0.70、centroid closure≈0.81，均显著高于随机0.37；索引占用12.8 GiB，构建18 min，检索速度≈3.7×10⁴文档/秒，投影速度≈1.3×10⁵文档/秒。

**⚠️ 局限性**

检索性能高度依赖LLM扩展质量，对不够具体的查询（如泛词“blood”）的区分能力有限；不提供召回度评估；固定线性空间对语义细粒度的判别能力受限。

---

## 84. A General Metric-Space Formulation of the Time Warp Edit Distance (TWED)

**arXiv ID:** 2601.05263 | [PDF](https://arxiv.org/pdf/2601.05263v1)

**作者:** Zhen Yi Lau `[一作]` `[通讯]`, Zhen Yi Lau

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了对时间扭曲编辑距离（TWED）的通用化——Generalized Time Warp Edit Distance (GTWED)，将观测域和时间域都视为任意度量空间；

**💡 创新点**

在保持严格度量性质的前提下，实现了在任意度量空间上的可编辑时间序列距离；

**🔧 技术方法**

采用度量空间理论、子可加函数正规化以及动态规划递推实现距离计算；

**📊 数据集**

该工作为理论性研究，没有使用具体数据集；

**📈 对比分析**

未给出实验比较，主要通过理论证明显示GTWED在三角不等式等度量公理上与原TWED一致；

**⚠️ 局限性**

限制包括：计算复杂度与原TWED相同但实际代价更高（需评估高维或自定义度量），且原TWED的上界分析不再适用于GTWED。

---

## 85. VIB-Probe: Detecting and Mitigating Hallucinations in Vision-Language Models via Variational Information Bottleneck

**arXiv ID:** 2601.05547 | [PDF](https://arxiv.org/pdf/2601.05547v1)

**作者:** Feiran Zhang `[一作]` (Fudan University), Xiaoqing Zheng `[通讯]` (Fudan University)

**通讯引用:** 1103 | [OpenAlex ID](https://openalex.org/A5017835517)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于变分信息瓶颈的视觉语言模型幻觉检测与缓解框架（VIB-Probe），利用所有层次多头注意力输出进行幻觉预测和实时干预。

**💡 创新点**

创新点在于将信息瓶颈理论应用于内部注意力特征，压缩高维注意力输出到可判别的低维表示，并通过梯度归因自动识别并抑制导致幻觉的注意力头，从而实现检测与推理时干预的统一。

**🔧 技术方法**

采用变分信息瓶颈（VIB）对注意力输出做编码解码，利用重参数化技巧训练二元交叉熵+KL正则化的检测器，并使用梯度归因对头进行重要性评估，实现推理时单步抑制。

**📊 数据集**

在POPE、AMBER、M‑HalDetect和COCO‑Caption四大幻觉基准上，对MiniGPT‑4、LLaVA‑v1.5‑7B、LLaVA‑v1.6‑Mistral‑7B和Qwen2.5‑VL‑7B‑Instruct 四款主流VLM进行评测。

**📈 对比分析**

与AvgEnt、AvgProb、RepProbing、MetaToken、DHCP等基线比较，VIB‑Probe在所有指标（AUPRC、AUROC）均超过对手，特别在生成任务上提升约2.8个百分点；在误差修正实验中也实现了最高的准确率和F1。

**⚠️ 局限性**

局限性包括仅适用于具备显式注意力的Transformer VLM，需要白盒访问模型内部状态，且未针对非注意力或其他多模态架构进行验证。

---

## 86. Improving User Experience with Personalized Review Ranking and Summarization

**arXiv ID:** 2601.05261 | [PDF](https://arxiv.org/pdf/2601.05261v1)

**作者:** Muhammad Mufti `[一作]` (King Fahd University of Petroleum and Minerals), Mahfuzur Rahman `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 2482 | [OpenAlex ID](https://openalex.org/A5026904260)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于用户情感与历史偏好的个性化评论排序与抽象摘要框架。

**💡 创新点**

结合星级评分与文本情感的混合情绪模型，以及基于句子嵌入的用户偏好聚类，生成个性化排序与摘要。

**🔧 技术方法**

采用 NLTK/TextBlob 进行情感分析，Sentence‑Transformers 生成嵌入，PyTorch 训练模型，OpenAI GPT‑4 生成摘要。

**📊 数据集**

使用来自 Amazon Mobile Electronics 的 104,854 条评论数据集。

**📈 对比分析**

通过 70 名参与者的用户研究，对比未排序、排序和摘要三种展示方式，结果显示摘要方式获得最高满意度、决策自信和购买意愿，且停留时间最短。

**⚠️ 局限性**

依赖人工标注的情感词典和单一文本数据，缺乏多模态信息，且仅在移动电子产品上验证，泛化性待进一步验证。

---

## 87. From Events to Trending: A Multi-Stage Hotspots Detection Method Based on Generative Query Indexing

**arXiv ID:** 2601.05258 | [PDF](https://arxiv.org/pdf/2601.05258v1)

**作者:** Kaichun Wang `[一作]` (Bytedance), Fei Lu `[通讯]` (Bytedance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出多阶段框架，用事件数据库构建、生成式索引生成和检索+重排序识别实现对话式系统中的热点查询检测。

**💡 创新点**

将热点查询检测转化为检索‑生成重排序任务，利用生成式索引桥接事件内容与用户查询，并引入链式推理、多样性生成与LLM后过滤，以及在线数据指令微调。

**🔧 技术方法**

使用大语言模型（GPT‑4o‑mini、InternLM2）、BGE‑M3 嵌入、BGE‑Gemma 重排序、链式推理（CoT）、后过滤、阈值检索以及指令微调。

**📊 数据集**

基于 Bytedance 对话日志构建的事件库，12,000 条在线查询标签、2,000 条重排序对、13,000 条在线微调样本，涵盖多日热点新闻。

**📈 对比分析**

相较于传统基线、仅检索或仅重排序方案，离线实验 F1 达 0.91，在线 A/B 测试用户满意度提升 27%，用户活跃度提升 0.77% 以上。

**⚠️ 局限性**

热点比例极低（约0.5%）时误召仍较多，需提高阈值导致召回下降；依赖高质量新闻源与 LLM 生成可信度；实时更新与多语言支持尚待完善。

---

## 88. Buffered AUC maximization for scoring systems via mixed-integer optimization

**arXiv ID:** 2601.05544 | [PDF](https://arxiv.org/pdf/2601.05544v1)

**作者:** Moe Shiina `[一作]` (University of Tsukuba), Yuichi Takano `[通讯]` (University of Tsukuba)

**通讯引用:** 1532 | [OpenAlex ID](https://openalex.org/A5081879179)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种利用混合整数优化（MIO）构建可解释评分系统的方法，直接最大化缓冲AUC（bAUC）并通过组稀疏约束限制问题数量，从而得到具有小整数系数的低维线性模型。

**💡 创新点**

创新点在于：① 将组稀疏约束引入评分系统设计，限制问题数而非变量数；② 将bAUC（AUC的紧凑凹下界）作为目标函数并通过正交变换化简为线性可求解的MILO问题；③ 在同一框架下实现整数系数的严格约束，兼顾可解释性与预测性能。

**🔧 技术方法**

技术手段包括：组稀疏（group sparsity）约束、缓冲AUC（bAUC）变换、整数系数限制、L1正则化、MILO（混合整数线性规划）模型，使用Gurobi求解器进行求解。

**📊 数据集**

实验使用了四个UCI公开二分类数据集：Surger（肺癌生存），Mushroom（毒蘑菇识别），Bank Marketing（定期存款订阅），Adult（收入分类），对每个数据集随机分为80%训练、20%测试，重复五次取平均。

**📈 对比分析**

与六种基准方法（L1-Regularization、Elastic-Net、Forward/Backward Stepwise、bAUC-Rounding、基于整数的bAUC）进行比较。结果显示：在Surger、Mushroom、Adult三组数据中，bAUC-Integer取得最高平均AUC，优于其它方法；在Bank数据集中，传统正则/步进法稍优。计算时间虽略高，但均在几分钟内，可接受。

**⚠️ 局限性**

限制：① 计算复杂度高，需在训练时对样本进行采样以降低规模，可能影响预测稳健性；② 只最大化bAUC而非真正的AUC，未能完全解决AUC最大化的NP-hard性；③ 在类别极不平衡的数据（Bank）上表现不佳；④ 仅针对离线训练，未讨论在线更新或动态调整。

---

## 89. Inverting Non-Injective Functions with Twin Neural Network Regression

**arXiv ID:** 2601.05378 | [PDF](https://arxiv.org/pdf/2601.05378v1)

**作者:** Sebastian J. Wetzel `[一作]` `[通讯]`, Sebastian J. Wetzel

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出并实现了一种称为逆双重神经网络回归（ITNNR）的确定性框架，用于在非单射函数的情况下通过锚点和 k‑最近邻搜索来恢复多值逆映射。

**💡 创新点**

创新点在于：①将逆问题重构为局部锚点回归；②利用 TNNR 预测相对调整而非绝对值；③通过前向一致性检验自动挑选正确分支；④在无限多解场景下仍能得到可解释的局部解。

**🔧 技术方法**

核心技术包括：Twin Neural Network Regression（双重神经网络回归）、k‑Nearest Neighbors（kNN）搜索、前向一致性投影、深度神经网络作为前向模型或后备模型。

**📊 数据集**

实验使用的典型数据集包括：1D 三次/四次多项式、正弦函数；2D 上半球、双变量多项式；3D 三次多项式；以及 2/3/4 关节平面/空间机械臂的逆运动学数据（均采用从理论函数生成或实际测量的噪声数据）。

**📈 对比分析**

与传统单向神经网络和纯 kNN 取样方法比较，ITNNR 在已知前向函数的情形下 RMSE 下降 80%–93%，在噪声数据情形下下降 30%–85%，大幅优于基线并保持较低的计算复杂度。

**⚠️ 局限性**

主要限制：①对高维问题的可扩展性尚未充分验证；②需要足够且分布良好的锚点，锚点选择不当会导致分支切换或误差增大；③若无精确前向模型，需训练额外网络进行一致性检验，可能引入误差。

---

## 90. TIME: Temporally Intelligent Meta-reasoning Engine for Context Triggered Explicit Reasoning

**arXiv ID:** 2601.05300 | [PDF](https://arxiv.org/pdf/2601.05300v1)

**作者:** Susmit Das `[一作]` `[通讯]` (Coherence Initiative), Susmit Das (Coherence Initiative)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型中引入时间敏感的思考触发机制，使模型在对话中按需发起短暂的推理。

**💡 创新点**

创新点是将时间戳、tick 与可插入的简短思考块作为训练原语，并通过四阶段课程与全批次对齐实现上下文触发的推理策略。

**🔧 技术方法**

使用 Qwen3 系列模型、QLoRA 微调、三阶段结构化训练与最终全批对齐、ISO‑8601 时间标签、<think> 语块、tick 事件。

**📊 数据集**

使用人工构造与 GPT‑4o/Gemini 生成的示例对话，随后手工挑选 128 条多样化对话作为全批对齐集；评估使用自建 TimeBench 77 场景。

**📈 对比分析**

与基线 Qwen3 的思考模式及非思考模式对比，TimeBench 评分提升 22–27 分（相当于 4B 时从 30% 提升到 53%），且推理 token 数量下降约 10 倍，degeneracy 降至 0.26%。

**⚠️ 局限性**

局限在于仅针对已支持思考的 Qwen3 系列，无法保证在纯指令模型或多模态模型上的迁移；对任务性评价缺失；对多语言、实时性与安全性未做验证；全批对齐集小且手工挑选。

---

## 91. The Persona Paradox: Medical Personas as Behavioral Priors in Clinical Language Models

**arXiv ID:** 2601.05376 | [PDF](https://arxiv.org/pdf/2601.05376v1)

**作者:** Tassallah Abdullahi `[一作]` (Brown University), Ritambhara Singh `[通讯]` (Brown University)

**通讯引用:** 3303 | [OpenAlex ID](https://openalex.org/A5070578596)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了 Persona 作为行为先验在临床大语言模型中的效果，系统评估不同专业角色（如急诊医生、护士）和交互风格（大胆 vs 谨慎）对三种任务（急诊转诊、门诊转诊、患者安全回应）的性能与安全性。

**💡 创新点**

提出了将 Persona 视为情境依赖的行为先验框架，首次量化其在高危与低危临床情境中的非单调、双刃效应，并揭示其对风险姿态、校准与一致性的影响。

**🔧 技术方法**

使用 Prompt‑based Persona 注入、定量指标（准确率、风险倾向、风险敏感度、一致率、校准）、LLM 内部评判器和专家临床医生评审，形成多维度评估方法。

**📊 数据集**

数据来源包括 1,466 例急诊 TIA/卒中病例（2013–2020）+ 201 例门诊常规病例的结构化记录，以及公开的 PatientSafetyBench（466 例安全合规问答）。

**📈 对比分析**

与无 Persona、Helpful Assistant 以及 No Persona 等基线对照后发现：在急诊场景中医疗 Persona 可提升约 20% 的准确率和校准度；在门诊场景中则下降约 10%；在安全合规任务中医疗 Persona 获得 LLM 与临床医生更高的安全与帮助性评价，但整体性能仍受模型与情境影响。

**⚠️ 局限性**

局限包括仅评估急诊角色与两种交互风格、未涵盖更广泛的临床角色与专业领域、样本规模有限、评估仅基于提示层面而非训练时控制，以及人类评估样本量不足，难以全面覆盖复杂临床任务。

---

## 92. MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**arXiv ID:** 2601.05296 | [PDF](https://arxiv.org/pdf/2601.05296v1)

**作者:** Jiyuan Zhang `[一作]`, Shen Li `[通讯]` (School of Computation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供了MLSys 2025会议的论文提交与格式化指南。

**💡 创新点**

通过统一严格的排版与提交流程，强化双盲审查机制与论文质量控制。

**🔧 技术方法**

使用PDF、Type‑1字体、LaTeX/Word转换、图表、脚注、算法环境等技术实现规范排版。

**📊 数据集**

该文档不涉及具体实验数据集，仅引用了示例参考文献。

**📈 对比分析**

与其他会议提交进行比较，强调不接受重复或已审稿论文，并规定严格的页面与字数限制。

**⚠️ 局限性**

局限性在于仅提供格式与流程规范，缺乏实验结果与学术贡献的实证内容。

---

## 93. Same Claim, Different Judgment: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection

**arXiv ID:** 2601.05403 | [PDF](https://arxiv.org/pdf/2601.05403v1)

**作者:** Zhiwei Liu `[一作]` (University of Manchester), Sophia Ananiadou `[通讯]` (University of Manchester)

**通讯引用:** 17835 | [OpenAlex ID](https://openalex.org/A5077976343)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个多语言、多情境的金融误信息检测基准，评估LLM在不同角色、地区和身份背景下的行为偏差。

**💡 创新点**

创新点在于将情境化、跨语言与身份敏感的误信息检测结合成统一框架，并系统量化情境对LLM判断的影响。

**🔧 技术方法**

采用22个主流LLM（包括开源与闭源、推理型与非推理型模型），通过情境注入、最大概率预测和F1差异量化来衡量偏差。

**📊 数据集**

数据集基于FinFact/​Snopes的金融误信息主张，经过人工筛选后翻译为英语、中文、希腊语和孟加拉语，共502条主张。

**📈 对比分析**

通过宏F1与差异度（AM/MAV）比较模型性能，结果显示大型模型偏差较小，但所有模型在判定真主张时仍易出现明显偏差，低资源语言表现更差。

**⚠️ 局限性**

局限性包括数据集以错误信息为主导致类别失衡、不同地区人工标注人数有限、且依赖Snopes来源与机器翻译质量。

---

## 94. EdgeLDR: Quaternion Low-Displacement Rank Neural Networks for Edge-Efficient Deep Learning

**arXiv ID:** 2601.05379 | [PDF](https://arxiv.org/pdf/2601.05379v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 95. TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation

**arXiv ID:** 2601.05254 | [PDF](https://arxiv.org/pdf/2601.05254v1)

**作者:** Wenbiao Tao `[一作]` (East China Normal University), Weining Qian `[通讯]` (East China Normal University)

**通讯引用:** 3830 | [OpenAlex ID](https://openalex.org/A5089931216)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了 TagRAG 框架，利用标签链构建层次化知识图并实现标签驱动的检索增强生成，支持高效的全局推理与增量更新。

**💡 创新点**

创新点包括：① 用对象标签与预设根域标签构建 DAG 形式的域链，实现知识的层次化组织；② 在构建阶段对域链与对象标签进行知识摘要融合；③ 采用标签驱动的检索机制，既提升检索粒度，又显著降低构建与推理成本；④ 设计了轻量化的增量插入方法，避免重建整个图谱。

**🔧 技术方法**

技术手段：使用 Qwen3-4B（或 GPT‑4o‑mini）进行标签与关系抽取与摘要生成；bge‑large‑en‑v1.5 进行向量检索；基于 DAG 的域链组织与标签检索；增量插入与知识融合算法；检索-生成流水线。

**📊 数据集**

实验数据集：UltraDomain benchmark 的四个子集——农业、计算机科学、法律以及混合域（Mix）.

**📈 对比分析**

与 NaiveRAG、GraphRAG、LightRAG、MiniRAG 在四个数据集上进行对比，TagRAG 的平均赢率达 95.41%，在构建时间上比 GraphRAG 快 14.6 倍，检索时间快 1.9 倍；在小模型（Qwen3‑1.7B）和不同检索器（bge‑base、小）下仍保持优势。

**⚠️ 局限性**

局限性：① 在多样性指标上略逊于部分基线；② 依赖 LLM 抽取标签与生成摘要，LLM 质量下降会影响整体效果；③ 在更大规模或更复杂跨域场景下的可扩展性和稳健性尚待进一步验证。

---

## 96. ART: Adaptive Reasoning Trees for Explainable Claim Verification

**arXiv ID:** 2601.05455 | [PDF](https://arxiv.org/pdf/2601.05455v1)

**作者:** Sahil Wadhwa `[一作]` (Capital One), Yue Wu `[通讯]` (Capital One)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种层级化的自适应推理树（ART）用于主张验证，通过生成支持和攻击论点并进行二分对决得到最终判决。

**💡 创新点**

创新点在于将论证拆分为树形结构，使用独立的判定LLM进行对抗比赛，并通过Bradley‑Terry模型校准论点强度，实现可解释、可争议的决策流程。

**🔧 技术方法**

采用LLM作为论点生成器和判定者，结合Bradley‑Terry概率模型进行强度校准，并使用层级聚合（ProductAggregation）得到根节点得分。

**📊 数据集**

实验数据集包括MedQA（医学多项选择）、StrategyQA（多跳常识问答）和TruthfulQA（事实与误导辨别）。

**📈 对比分析**

与直接提示、Chain‑of‑Thought（CoT）和ArgLLM基线比较，ART在三大数据集上取得更高或相近的准确率，尤其在多模型评判器设置下明显优于对照方法。

**⚠️ 局限性**

局限性包括：需要多次LLM调用导致计算开销较高，判断LLM可能存在偏见，树结构深度/宽度受限时易出现误判，且在极其复杂或知识密集的主张上仍易出现误差。

---

## 97. One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection

**arXiv ID:** 2601.05552 | [PDF](https://arxiv.org/pdf/2601.05552v1)

**作者:** Bin-Bin Gao `[一作]` (Tencent YouTu Lab), Chengjie Wang `[通讯]` (Tencent YouTu Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种语言无关、极简的通用异常检测框架 UniADet，能够在零样本或少样本场景下对工业和医学图像进行异常分类和像素级分割。

**💡 创新点**

核心创新点包括：①完全抛弃文本编码器与复杂的 prompt 设计，仅学习判别权重；②将全局异常分类与局部异常分割完全解耦；③进一步将不同层级特征解耦，使每一层拥有独立的分类/分割权重；④利用多尺度记忆库实现少样本提升，并通过类感知增强提升鲁棒性。

**🔧 技术方法**

使用的技术主要有：视觉基础模型 CLIP、DINOv2/DINOv3 作为特征提取器；全局/局部特征与权重的对比学习；多尺度记忆库检索；类感知数据增强；轻量化参数学习（仅约 0.002M 训练参数）。

**📊 数据集**

在 14 个公开基准上评估：工业域（MVTec、VisA、BTAD、DTD、KSDD、Real‑IAD）和医学域（HeadCT、BrainMRI、Br35H、ISIC、ColonDB、ClinicDB、Endo、Kvasir）。

**📈 对比分析**

与现有零样本/少样本方法对比，UniADet 在图像级 AUROC/AUPR、像素级 AUROC/AUPR 上均实现或突破 state‑of‑the‑art，甚至在 1‑shot 下超过部分全样本方法；推理速度最快，参数量最低（仅 0.002M）。

**⚠️ 局限性**

局限性：仍依赖高质量的视觉基础模型，对极细微或完全未见的异常类型可能效果不佳；少样本模式需构建记忆库，处理极大规模数据时存储开销可能上升；在极端噪声或严重遮挡场景下的鲁棒性待进一步验证。

---

## 98. The Table of Media Bias Elements: A sentence-level taxonomy of media bias types and propaganda techniques

**arXiv ID:** 2601.05358 | [PDF](https://arxiv.org/pdf/2601.05358v1)

**作者:** Tim Menzner `[一作]` (Coburg University of Applied Sciences and Arts), Jochen L. Leidner `[通讯]` (University of Sheffield)

**通讯引用:** 2460 | [OpenAlex ID](https://openalex.org/A5078661113)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一个 38 种细粒度、句子级别的媒体偏见与宣传分类体系，并在实际新闻语料中实现自动标注与可视化；

**💡 创新点**

创新点在于将偏见拆解为可单独识别的 38 个基本类型，构建“媒体偏见元素周期表”模型，并通过跨学科理论与实证注释相结合，填补了现有分类框架在细节、覆盖度与歧义性上的不足；

**🔧 技术方法**

采用系统化的标注流程（迭代式抽样、人工细读、跨学科推理）与多层次聚类，将词汇、语境、逻辑与社会驱动等维度融合，形成多元化识别指引；

**📊 数据集**

数据集包括 26,464 句（来自新闻稿、用户提交、浏览记录、Reuters/Fox News、BABE 等），其中 16,229 句来自 BiasScanner 用户标注，另外 9,284 句为实验语料，覆盖德语与英语；

**📈 对比分析**

通过对 155 句随机抽样的量化调查展示了各类型出现频率，并将本体系与 SemEval‑2020、Media Bias Identification Benchmark 等主流分类表进行交叉映射，结果显示覆盖面扩大、歧义度降低、分类一致性提升；

**⚠️ 局限性**

局限性包括：仅关注句子级别，无法捕捉跨句或篇章层面的偏见；语料主要限于德语与英语，可能缺乏跨文化泛化；标注过程仍受主观判断影响；新兴偏见形式可能尚未被纳入，需持续迭代。

---

## 99. Uncovering Failures in Cyber-Physical System State Transitions: A Fuzzing-Based Approach Applied to sUAS

**arXiv ID:** 2601.05449 | [PDF](https://arxiv.org/pdf/2601.05449v1)

**作者:** Theodore Chambers `[一作]` (University of Notre Dame), Jane Cleland-Huang `[通讯]` (University of Notre Dame)

**通讯引用:** 7182 | [OpenAlex ID](https://openalex.org/A5037363688)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为SaFUZZ的基于语义的模糊测试管线，用于验证小型无人机（sUAS）在多层状态机、故障保护和人机交互下的状态转移行为；

**💡 创新点**

创新点在于：①使用语义化模糊规范生成真实任务场景与时序扰动；②结合决策树或然器与聚类分析自动判定失败；③通过真值表和最小割集自动生成故障树，可直观展示根因；

**🔧 技术方法**

主要技术包括：语义模糊生成、Gazebo高保真仿真、Python多线程控制、决策树分类、K‑means聚类、布尔最小化生成故障树；

**📊 数据集**

使用真实sUAS平台（PX4/ArduPilot）在仿真与实地场景中收集飞行日志、传感器扰动和人为操作数据；

**📈 对比分析**

与开发团队的18个月手工测试相比，SaFUZZ发现了8项状态/模式转移故障，开发团队仅发现1项；在实地验证中，6项仿真发现的故障中有4项得到复现；

**⚠️ 局限性**

局限性包括：仅验证单一SuT，仿真与实测之间存在地理围栏等差异；未开展正式用户可用性评估；未与更正式的形式验证或基线模糊工具做直接对比；

---

## 100. Transforming User Defined Criteria into Explainable Indicators with an Integrated LLM AHP System

**arXiv ID:** 2601.05267 | [PDF](https://arxiv.org/pdf/2601.05267v1)

**作者:** Geonwoo Bang `[一作]` (Sungkyunkwan University), Moohong Min `[通讯]` (Sungkyunkwan University)

**通讯引用:** 148 | [OpenAlex ID](https://openalex.org/A5065739631)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UniScore 框架，使用 LLM 对用户定义的多项评估准则进行评分，并通过 AHP 结合 Jensen–Shannon 距离自动生成可解释的加权综合文本质量分数。

**💡 创新点**

创新点在于将 LLM 评分与信息论度量（JSD）相结合，利用 AHP 对准则的判别力进行差分映射，从而在轻量级模型与低延迟场景下实现稳健、可解释的多准则聚合。

**🔧 技术方法**

技术包括：LLM-as-judge（Qwen3‑1.7B 等轻量级模型）、Jensen–Shannon 距离用于判别力评估、差分映射构造的 AHP 对比矩阵、主特征向量求权重、线性组合生成最终分数。

**📊 数据集**

数据集包括：Amazon Reviews（软件类，用户有帮助投票）、RoSE XSum（系统摘要，ACU 分数）、Depression Tweet（二分类抑郁/非抑郁）。

**📈 对比分析**

与单一 LLM、随机权重、线性回归、随机森林、神经网络等基线对比，UniScore 在所有数据集上都实现了更高的 Spearman、Kendall、Pearson 等相关系数，判别力（F1、准确率）更好，且计算速度和内存占用更低。

**⚠️ 局限性**

局限性包括：依赖外部监督信号（需要标签或投票等），需要用户预先定义评估准则，且在大模型或多任务场景下的性能尚未验证。

---

## 101. WildSci: Advancing Scientific Reasoning from In-the-Wild Literature

**arXiv ID:** 2601.05567 | [PDF](https://arxiv.org/pdf/2601.05567v1)

**作者:** Tengxiao Liu `[一作]` (University of California), William Yang Wang `[通讯]` (University of California)

**通讯引用:** 18655 | [OpenAlex ID](https://openalex.org/A5100702485)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

自动从同行评审论文中生成多学科科学推理的多项选择题，并用强化学习微调模型。

**💡 创新点**

提供完全自动化的科学问题生成管道、基于投票的质量控制以及覆盖9个学科、26个子领域的可验证数据集。

**🔧 技术方法**

采用大型语言模型进行问题生成与细化，使用群组相对策略优化（GRPO）进行强化学习微调。

**📊 数据集**

WildSci数据集（56K题）以及公开基准如GPQA、SuperGPQA、MMLU‑Pro。

**📈 对比分析**

与未训练模型相比，在所有基准平均提升约7%（如GPQA‑Aug +49%，SuperGPQA +5%，MMLU‑Pro +11%），并且在内部验证集从约47%提升至≈80%。

**⚠️ 局限性**

数值推理过于简单，MCQ格式易被模型利用启发式，且难以评估完全开放式科学推理。

---

## 102. The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era

**arXiv ID:** 2601.05564 | [PDF](https://arxiv.org/pdf/2601.05564v1)

**作者:** Zhixian Zhao `[一作]` (NPU), Lei Xie `[通讯]` (Huawei Technologies)

**通讯引用:** 3017 | [OpenAlex ID](https://openalex.org/A5100365442)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HumDial挑战，构建双轨道评测体系以评估情感智能与全双工交互能力。

**💡 创新点**

首次采用真实录音结合LLM生成的多轮情感演化脚本与自然交叉干扰，系统性考察情感追踪与实时双向交互。

**🔧 技术方法**

结合Gemini2.5-pro与DeepSeek生成脚本、专业演员录制、Qwen3‑Omni‑30B自动评分与人工评估、Docker+RTX A6000统一实验环境。

**📊 数据集**

使用自研HumDial数据集，覆盖六种情感类别，包含Task 1、Task 2、Task 3三类多轮对话及全双工干扰实例。

**📈 对比分析**

采用自动评分与人工评估加权合成最终分数；Track I最高得分4.97，Track II Cookie_asr以79.9分夺冠，显示LLM在情感分析上几近完美，但情感生成与噪声抑制仍有提升空间。

**⚠️ 局限性**

受限于人工录制导致数据量有限、情感类别受限；评测多依赖LLM判断，易受模型偏差影响；全双工交互仍难以精准处理非指令噪声。

---

## 103. A Rank 23 Algorithm for Multiplying 3 x 3 Matrices with an Arithmetic Complexity of 59

**arXiv ID:** 2601.05272 | [PDF](https://arxiv.org/pdf/2601.05272v1)

**作者:** Erik Mårtensson `[一作]` (Lund University), Joshua Stapleton `[通讯]` (Imperial College London)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

研究了3×3矩阵乘法的低算术复杂度算法，并提出了一个基于rank 23且仅需59个加法的新算法。

**💡 创新点**

在保持rank 23不变的前提下，将加法次数从之前的60/61降至59，首次实现了无基变换的59加法算法。

**🔧 技术方法**

结合了Mårtensson与Stankovski Wagner的加法优化方法和Stapleton的基于神经网络的快速方案生成技术。

**📊 数据集**

无实测数据集，算法在理论层面通过符号计算和自动化工具验证。

**📈 对比分析**

通过与已有的23乘法算法（61、62、60加法）比较，证明59加法实现了加法复杂度的最优进展；实验上仅涉及符号计数，未做数值性能测试。

**⚠️ 局限性**

算法仅在3×3矩阵规模上取得进展，对更大规模矩阵的推广未知；实现仍需手工整理，缺乏通用自动化流程。

---

## 104. Optimizing Digital Adjudication through Social Network Analysis: An Empirical Study of Credit Card Disputes in Beijing

**arXiv ID:** 2601.05299 | [PDF](https://arxiv.org/pdf/2601.05299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 105. Protosampling: Enabling Free-Form Convergence of Sampling and Prototyping through Canvas-Driven Visual AI Generation

**arXiv ID:** 2601.05401 | [PDF](https://arxiv.org/pdf/2601.05401v1)

**作者:** Alicia Guo `[一作]` (Autodesk Research), Fraser Anderson `[通讯]` (Autodesk Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了“protosampling”概念，设计并实现了 Atelier，一个基于无限画布的创作系统，将采样与原型制作融合，并通过易于使用的“easel”模块化工作流、快速操作和版本追踪等功能支持生成式 AI 的可控创作。

**💡 创新点**

创新点在于：①将采样与原型视为同一创作循环，形成新的创作视角；②将传统 AI 工具拆解为可视化、可操作的模块（easel），降低技术门槛；③引入完整的 provenance 与组织机制，让创作过程在无限画布上可追溯、可重构；④结合多模态输入与参数调控，实现更细粒度的生成控制。

**🔧 技术方法**

使用技术包括：OpenAI、Stable Diffusion XL、FLUX、Wan 2.2、Flux Redux、ControlNet、Lying Sigmas、Normalized Attention Guidance、FlowEdit、ComfyUI、tldraw（无限画布）、React、D3 等；核心实现为多模型融合的自定义 ComfyUI 工作流与前端可视化组件。

**📊 数据集**

未使用专门的标注数据集；模型均来自公开训练的开源大模型（如 Stable Diffusion、FLUX 等），并在系统中预处理生成控制图像（深度、姿态、线稿等）。

**📈 对比分析**

评估方法主要是定性用户研究：5 名创意专业人士在 4 小时内使用 Atelier，并与 Midjourney、ChatGPT、ComfyUI 等现有工具进行对比。未给出量化性能指标，评估侧重于创作自由度、可控性、流程可视化与用户体验等维度。

**⚠️ 局限性**

局限性包括：①用户样本规模有限，结果缺乏普适性；②依赖本地开源模型，可能面临硬件要求、能耗与版权争议；③高级 AI 技术仍需一定学习成本，难以完全消除技术门槛；④系统主要聚焦图像/视频，音频、3D 等多模态支持尚不完善；⑤在高并发或大规模项目中，画布管理与实时渲染性能需进一步优化。

---

## 106. DeMa: Dual-Path Delay-Aware Mamba for Efficient Multivariate Time Series Analysis

**arXiv ID:** 2601.05527 | [PDF](https://arxiv.org/pdf/2601.05527v1)

**作者:** Rui An `[一作]` (Northwestern Polytechnical University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 45320 | [OpenAlex ID](https://openalex.org/A5100404130)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种双路径、延迟感知的Mamba骨干网络（DeMa）用于多变量时间序列分析

**💡 创新点**

创新点在于通过自适应傅里叶滤波器先将时间序列拆分为跨时间和跨变量两种依赖；双路径设计分别采用Mamba‑SSD和Mamba‑DALA，后者同时考虑全局相关延迟与token级相对延迟；最终通过加权融合得到高效、可迁移的特征；

**🔧 技术方法**

利用自适应傅里叶分解、Mamba状态空间模型、延迟感知方向注意力（DALA）、token化与加权融合等技术实现

**📊 数据集**

在五大主流MTS任务上测试，使用的数据集包括ETTh1/2、ETTm1/2、Electricity、Traffic、PEMS03/04/07/08、Solar‑Energy、UEA多变量分类集、MSL等异常检测集

**📈 对比分析**

与Transformer、MLP、TCN以及最新Mamba变体（Affirm、S‑Mamba、CMamba、SAMBA等）做对比，DeMa在长短期预测、补全、分类与异常检测上均实现了SOTA水平，且训练速度快、显存占用低

**⚠️ 局限性**

局限性在于对融合权重的选择较为敏感，尤其是细粒度任务（补全、异常检测）；自适应傅里叶分解对强混合频谱假设敏感，可能对极端跨变量耦合或高缺失率数据效果不佳

---

## 107. EvidFuse: Writing-Time Evidence Learning for Consistent Text-Chart Data Reporting

**arXiv ID:** 2601.05487 | [PDF](https://arxiv.org/pdf/2601.05487v1)

**作者:** Huanxiang Lin `[一作]` (South China University of Technology), Mingkui Tan `[通讯]` (South China University of Technology)

**通讯引用:** 14075 | [OpenAlex ID](https://openalex.org/A5032352025)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的多智能体框架，支持在撰写时动态生成并嵌入图表，从而实现文本与图表的实时交织与一致性；

**💡 创新点**

核心创新是将可视化分析拆分为专门的Data-Augmented Analysis Agent和Real-Time Evidence Construction Writer，并通过写作时的即时图表请求和注入，实现图表生成与叙述同步、无需预先冻结证据空间；

**🔧 技术方法**

利用大模型（如Qwen3、Qwen2.5-VL）配合可视化工具、EDA知识注入、提示工程以及多轮交互的图表生成流程；

**📊 数据集**

在来自Tableau Public Stories、Our World in Data和USAFacts的三组多表报告数据集上进行评测；

**📈 对比分析**

与Direct、DataNarrative（text-first-graph-second）和DeepAnalyze（graph-first-text-second）等基线对比，基于LLM-as-Judge和人工评估的多层指标（图表质量、章节级文本-图表一致性、报告整体实用性），新框架在所有数据集和指标上均获得最高排名；

**⚠️ 局限性**

局限性包括对多模态大模型的依赖、对复杂交互式可视化技术的实现细节仍需进一步完善，以及在极端大规模表格或多语言环境下的通用性尚待验证。

---

## 108. Feedback Effects on Cognitive Dynamics: Network-Based Insights from EEG Patterns and Behavioral Performance

**arXiv ID:** 2601.05450 | [PDF](https://arxiv.org/pdf/2601.05450v1)

**作者:** Behdokht Kiafar `[一作]`, Roghayeh Leila Barmaki `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在阅读眼睛测试中，比较了给予即时反馈与不提供反馈两种条件下，受试者的脑电（EEG）活动与正确回答率的关系。

**💡 创新点**

创新点是将Epistemic Network Analysis（ENA）与Ordered Network Analysis（ONA）扩展为 Neuro‑Epistemic Network Analysis（NENA）和 Neuro‑Ordered Network Analysis（NONA），专门用于解码脑电频段与行为结果之间的网络关系。

**🔧 技术方法**

技术方法包括：Muse 头戴式 EEG 采集、平均参考、滤波、ICA 去噪、Welch PSD、SNR 阈值二值化、SVD 降维、共现矩阵构建、NENA（无向共现网络）与 NONA（有向时间序列网络）构建与可视化。

**📊 数据集**

数据集为 11 名受试者完成 36 题 RMET（18 题反馈，18 题无反馈）的 EEG 数据，采样率 256 Hz，四个通道（TP9、TP10、AF7、AF8）。

**📈 对比分析**

通过两组网络的二维投影和差异网络进行比较，显著差异检测（p=0.01，Cohen’s d>2），表明反馈条件下高频（β、γ）与正确回答的关联更强，而无反馈条件下低频（θ、α）与错误回答的关联更显著。

**⚠️ 局限性**

局限性包括样本量小、EEG 信号易受噪声和伪迹影响、仅研究即时反馈对神经动态的影响、未检验长期记忆保持、以及仅使用有限通道的可穿戴设备。

---

## 109. Glitter: Visualizing Lexical Surprisal for Readability in Administrative Texts

**arXiv ID:** 2601.05411 | [PDF](https://arxiv.org/pdf/2601.05411v1)

**作者:** Jan Černý `[一作]` (Charles University), Silvie Cinková `[通讯]` (Charles University)

**通讯引用:** 1026 | [OpenAlex ID](https://openalex.org/A5043569790)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用信息熵与词汇惊奇度评估文本可读性，开发可视化工具 Glitter，帮助改进行政文本的易读性和清晰度。

**💡 创新点**

创新点包括：①将大语言模型的 token 概率与惊奇度结合，提供多模型估计；②可视化高可预测（低惊奇度）片段同样可能难懂；③使用热度色标突出信息量差异；④对行政法律文本的可读性改写提供直观评估。

**🔧 技术方法**

技术：采用预训练 Transformer（如 GPT‑2）计算 token 概率并得到惊奇度；使用 softmax、概率链规则处理子词；多模型支持；前端 UI 与 CLI，颜色映射可视化。

**📊 数据集**

数据集：主要使用行政/法律文本 KUKY 数据集（后编辑版与原始版），以及 GPT‑2 训练用的通用网页语料（约 40 GB）。

**📈 对比分析**

方法比较：对 KUKY 数据进行人工评估，观察改写后文本的惊奇度分布更均衡；未给出数值指标，而是通过定性分析表明改写提升了可读性；与先前仅用惊奇度评估的工具对比，强调了模型预测过强导致的低惊奇度误判。

**⚠️ 局限性**

局限性：①大型预训练模型对专业术语预测过于精准，导致惊奇度低但对普通读者仍难懂；②模型训练数据覆盖法律文本，偏向专业知识，低估普通读者的认知负荷；③缺乏系统化的量化评估与对比实验；④工具仅基于概率估计，未考虑语义层面的易读性因素。

---

## 110. Bi-Orthogonal Factor Decomposition for Vision Transformers

**arXiv ID:** 2601.05328 | [PDF](https://arxiv.org/pdf/2601.05328v1)

**作者:** Fenil R. Doshi `[一作]` (Harvard University), George Alvarez `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过构建Bi‑Orthogonal Factor Decomposition (BFD)，对视觉Transformer的注意力机制进行分离与谱分析，揭示了token间信息流的空间与内容组成及其在各层、各头中的专业化。

**💡 创新点**

创新点在于：①将激活拆分为位置、内容和全局三因子，实现统计正交化；②对查询‑键交互矩阵做SVD得到bi‑orthogonal模式，量化不同信息因子对注意力能量的贡献；③揭示自监督模型（DINOv2）在中间层保持二维空间拓扑并持续丰富语义，解释其更优的整体形状处理能力。

**🔧 技术方法**

采用ANOVA式因子分解、SVD、线性解码、PCA等技术；并利用多头注意力的谱特征（稳定秩、对齐度）对模式进行能量归因与功能聚类。

**📊 数据集**

在ImageNet验证集（5,000张图片）上，对比监督ViT-B/16与自监督DINOv2-B/14。

**📈 对比分析**

与监督模型相比，自监督模型在内容–位置能量上占比更高，模式谱更丰富、稳定秩更大，位置结构在中间层保持二维拓扑；这些指标表明DINOv2在注意力分配与信息集成方面表现更好。

**⚠️ 局限性**

局限性包括：①仅评估了ViT与DINOv2两种架构，未验证其他自监督方法；②因子分解假设线性可分，可能忽略更深层次的非线性耦合；③使用的样本量有限，未直接关联下游任务性能；④方法对大规模多模态Transformer的适用性尚待验证。

---

## 111. Coding the Visual World: From Image to Simulation Using Vision Language Models

**arXiv ID:** 2601.05344 | [PDF](https://arxiv.org/pdf/2601.05344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 112. MoGen: A Unified Collaborative Framework for Controllable Multi-Object Image Generation

**arXiv ID:** 2601.05546 | [PDF](https://arxiv.org/pdf/2601.05546v1)

**作者:** Yanfeng Li `[一作]` (Macao Polytechnic University), Tao Tan `[通讯]` (Macao Polytechnic University)

**通讯引用:** 3075 | [OpenAlex ID](https://openalex.org/A5101628586)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多对象图像生成框架MoGen，能够根据文本提示精确控制物体数量并支持可选的结构、对象和边框三种控制信号；

**💡 创新点**

核心创新在于(1)区域语义锚定（RSA）模块，利用全局与短语级文本语义分别引导整体结构与局部区域，实现文本描述的数量一致性；(2)自适应多模态引导（AMG）模块，能够动态解析并融合多源控制信号，生成结构化意图并对场景布局与属性进行细粒度控制；

**🔧 技术方法**

技术手段包括扩散模型（基于SDXL）下的U‑Net交互层、语义解析器（全局/短语分支）、自注意力与交叉注意力机制、Box Encoder、Image Encoder（DinoV2）以及自适应控制器等；

**📊 数据集**

使用了新构建的MoCA基准数据集（约10k张多对象图像），包含文本描述、结构参考、对象参考及边框标注；

**📈 对比分析**

与SDXL、FLUX、Emu2、Omnigen2、MS‑Diffusion、Xverse、Bounded‑Attention、StableFlow、FlowEdit等方法对比，MoGen在多种生成配置（T→Img、T+O→Img、T+B→Img、T+O+B→Img、T+S→Img、T+S+O→Img）上均显著提升NIQE、CLIP Score、DPG、Q‑Align、Spatial‑Sim、Numerical Accuracy和MOS，尤其在数量一致性与细粒度控制方面突破传统方法；

**⚠️ 局限性**

局限性主要包括：(1) 仍基于SDXL等预训练扩散模型，可能受限于其语义和分辨率；(2) 对极大物体数量或极复杂布局的鲁棒性尚未充分验证；(3) 对控制信号的组合支持虽灵活，但在多模态信息不完整或错误时的容错性需要进一步改进。

---

## 113. Readability-Robust Code Summarization via Meta Curriculum Learning

**arXiv ID:** 2601.05485 | [PDF](https://arxiv.org/pdf/2601.05485v1)

**作者:** Wenhao Zeng `[一作]` (Shanghai Jiao Tong University), Xiaodong Gu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2317 | [OpenAlex ID](https://openalex.org/A5033286111)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究低可读性代码对代码摘要模型的影响，提出结合元学习与课程学习的鲁棒微调方法，以提升模型在低可读性代码下的摘要质量同时保持对原始代码的性能。

**💡 创新点**

创新点在于构建逐步加难的语义模糊与结构干扰数据集，并在每个训练步骤使用MAML进行元更新，使模型既能保持对清晰代码的高性能，又能显著提升对低可读性代码的鲁棒性。

**🔧 技术方法**

采用Transformer基础的LLM（DeepSeek‑Coder、Qwen2.5‑Coder）+课程学习+元学习（MAML）+细粒度梯度更新技术。

**📊 数据集**

使用CodeSearchNet（Python子集）与人工标注的MLRC数据集，利用FNE、IRN、DCI等可读性降级方式生成训练与测试样本。

**📈 对比分析**

与直接微调、仅课程学习、CLAWSAT等基线对比，实验显示在语义模糊和结构干扰两种低可读性场景下，方法在BLEU/SBERT上平均提升3–6点，同时对原始代码性能提升0.5–1点。

**⚠️ 局限性**

局限性包括仅使用合成的可读性下降样本，未覆盖更复杂的混淆技术（如控制流平坦化）；超参数调优范围有限；评估指标依赖自动化指标，缺乏人工评测。

---

## 114. Mathematical Knowledge Graph-Driven Framework for Equation-Based Predictive and Reliable Additive Manufacturing

**arXiv ID:** 2601.05298 | [PDF](https://arxiv.org/pdf/2601.05298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 115. Improving Enzyme Prediction with Chemical Reaction Equations by Hypergraph-Enhanced Knowledge Graph Embeddings

**arXiv ID:** 2601.05330 | [PDF](https://arxiv.org/pdf/2601.05330v1)

**作者:** Tengwei Song `[一作]` (Machine Learning Department, MBZUAI), Zhiqiang Xu `[通讯]` (Machine Learning Department, MBZUAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了利用化学反应方程式的高阶关系进行酶预测的 Hyper-Enz 模型，解决传统酶-底物配对数据稀疏问题。

**💡 创新点**

创新点在于把反应方程式视作知识图谱三元组，构建双层异质超图并结合超图 Transformer 与 KGE，另外引入多专家机制融合完整与缺失反应信息。

**🔧 技术方法**

使用了超图 Transformer、PairRE KGE、MLP 以及邻居采样等技术，同时融合知识库专家、超图专家和机器学习专家的输出。

**📊 数据集**

构建了基于 BRENDA、PubChem、BKMS-react 的 EQ50k（38,861 条完整方程 + 42,714 条缺失方程）以及酶-底物配对数据集 ES-23k/ES-23k-M。

**📈 对比分析**

与 TransE、RotatE、PairRE、ComplEx、LightGCN、HL-GNN 等 KGE/GCN 基线，以及 EnzRank、Boost-RS、MEIGCN、FusionESP 等酶配对模型对比，Hyper-Enz 在 MR、MRR、Hit@k 等指标上平均提升 10-30%，在配对级别上实现最高的 MRR 与 Hit@10。

**⚠️ 局限性**

局限性包括依赖大量完整或部分完整的反应方程，对低频酶或罕见底物的预测仍受限；超图构建与邻居采样对计算资源要求较高。

---

## 116. On the Limits of Self-Improving in LLMs and Why AGI, ASI and the Singularity Are Not Near Without Symbolic Model Synthesis

**arXiv ID:** 2601.05280 | [PDF](https://arxiv.org/pdf/2601.05280v1)

**作者:** Hector Zenil `[一作]` (King's College London), Hector Zenil `[通讯]` (King's College London)

**通讯引用:** 2978 | [OpenAlex ID](https://openalex.org/A5003718950)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过离散时间动力学系统建模，证明了在大语言模型及生成式人工智能中，递归自我训练（以自身生成的数据为主）必然导致模型崩塌，表现为熵衰减与方差放大；

**💡 创新点**

创新点在于首次给出了模型崩塌的严格数学证明，提出了两种根本失效模式，并将问题归因于分布学习的有限采样限制，随后提出利用算法概率（CTM）与符号回归的混合神经符号方法来突破此限制；

**🔧 技术方法**

使用了信息论工具（KL散度、DPI）、算法信息理论（Kolmogorov复杂度、Coding Theorem、CTM、BDM）以及符号投影与因果修正操作的组合来设计新的学习框架；

**📊 数据集**

论文主要为理论分析，不依赖具体数据集，而是对普遍的生成式模型假设进行推导；

**📈 对比分析**

由于是理论证明，未进行实验对比；作者通过与传统仅基于分布的优化（交叉熵/KL）对比，指出后者在自我训练时会出现熵损失和方差漂移，而混合方法能够在理论上维持信息量；

**⚠️ 局限性**

局限性包括：假设模型具备无限表达能力且采样过程理想，实际模型受容量与噪声限制；提出的混合方法仍处于概念阶段，尚未在大规模模型上验证；并且对真实世界的外部真值信号（α>0）依赖性仍未解决。

---

## 117. Task Cascades for Efficient Unstructured Data Processing

**arXiv ID:** 2601.05536 | [PDF](https://arxiv.org/pdf/2601.05536v1)

**作者:** Shreya Shankar `[一作]` (University of California Berkeley), Aditya Parameswaran `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出任务级流水线（Task Cascades）用于大规模文本分析，按文档子集、模型和操作顺序执行，降低LLM推理成本。

**💡 创新点**

创新点：①将传统模型级流水线扩展到任务级流水线，允许在不同阶段切换模型、操作和文本片段；②利用LLM代理自动生成并迭代优化 surrogate 任务；③提供统计显著性保证，使得最终流水线可达到预设准确度阈值。

**🔧 技术方法**

技术手段包括：LLM代理生成与迭代；文档重排序与轻量级相关性分类器；置信度阈值设定与动态调整；贪心任务排序算法；统计置信度保证（Hoeffding/方差加权估计）；成本模型与缓存利用。

**📊 数据集**

使用八个真实工作负载：agnews、court、enron、fever、games、legal、pubmed、wiki_talk（均来自 Kaggle 或先前研究），覆盖短文档、多步推理、法律/医学文本等多种场景。

**📈 对比分析**

对比方法：oracle 仅推理、两模型流水线（proxy+oracle）及其保证版。结果显示，在 90% 目标准确率下，Task Cascades 平均成本比两模型流水线低 48.5%，比 oracle 低 86.2%。在不同准确率阈值下，Task Cascades 仍保持 Pareto 前沿优势，特别是在较低准确率或文本冗余较高的任务中表现突出。

**⚠️ 局限性**

局限性：①在需要极高准确率的任务中，任务级流水线收益会减弱；②依赖足够强大的代理模型和较大开发集，构造成本随数据规模增加；③LLM 的置信度不一定可靠，阈值设定仍需经验；④对多类或开放式输出的适用性有限，需进一步研究。

---

## 118. DynaSTy: A Framework for SpatioTemporal Node Attribute Prediction in Dynamic Graphs

**arXiv ID:** 2601.05391 | [PDF](https://arxiv.org/pdf/2601.05391v1)

**作者:** Namrata Banerji `[一作]` (Ohio State University), Tanya Berger-Wolf `[通讯]` (Ohio State University)

**通讯引用:** 4553 | [OpenAlex ID](https://openalex.org/A5060005215)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种端到端的动态图节点属性多步预测框架 DynaSTy，能够利用每个样本的时间序列邻接矩阵与节点特征共同预测未来节点属性。

**💡 创新点**

创新点包括：① 允许每个训练样本使用独立的动态图序列；② 在 transformer 的注意力中注入可学习的邻接偏置，实现动态边权重；③ 结合遮掩预训练、时间加权损失与差分惩罚，提升长期预测稳定性。

**🔧 技术方法**

核心技术为：Transformer‑based 空间编码器（带边缘偏置）、GRU 自回归解码器、scheduled sampling、horizon‑weighted MAE + 变化损失、以及遮掩重建预训练。

**📊 数据集**

实验数据集包括：Bitcoin‑OTC、Bitcoin‑Alpha 信任网络；METR‑LA、PEMS‑Bay 交通流；ABIDE 脑功能 fMRI（200 ROI）。部分数据使用半合成动态图。

**📈 对比分析**

与 STGCN、DCRNN、Graph WaveNet、MTGNN、DGCRN、PDFormer 等基线比较，DynaSTy 在 RMSE/MAE 上均显著优于它们，尤其在需要 per‑sample 动态图的 Bitcoin 与脑网络上提升更为明显；对比打乱动态图的实验显示图结构信息对性能至关重要。

**⚠️ 局限性**

局限性：仅假设节点集合固定，无法处理节点出现/消失；对大规模图的计算开销大，当前实现基于全注意力，未来需要稀疏或分布式扩展。

---

## 119. When the Server Steps In: Calibrated Updates for Fair Federated Learning

**arXiv ID:** 2601.05352 | [PDF](https://arxiv.org/pdf/2601.05352v1)

**作者:** Tianrun Yu `[一作]` (Pennsylvania State University), Minghong Fang `[通讯]` (University of Louisville)

**通讯引用:** 1529 | [OpenAlex ID](https://openalex.org/A5056811906)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了一种服务器端的校准更新方法，用于提升联邦学习系统的公平性。

**💡 创新点**

创新点在于服务器通过收集早期全局模型生成合成数据，再生成单一校准更新，既不需要客户端修改训练流程，也不受聚合规则限制，可兼容多种公平度量。

**🔧 技术方法**

使用的技术包括模型检查点凝聚、合成数据生成、基于公平指标的梯度校准更新，以及对多种聚合策略和公平度量的无缝支持。

**📊 数据集**

实验采用六个数据集：Income‑Sex、Employment‑Sex、Health‑Sex、Income‑Race、MNIST 和 CIFAR‑10。

**📈 对比分析**

与六个基线方法（FLinear、FairFed、FedFB、Reweight、Gaussian、Uniform）比较，在所有公平指标上均取得更低的偏差，同时保持与 FedAvg 相近甚至更高的准确率；对不同聚合规则、客户端规模及非 IID 情况均表现稳健。

**⚠️ 局限性**

局限性包括：需要服务器存储模型与合成数据，存在一定隐私风险；合成数据质量对公平效果有影响，过大噪声会削弱准确率；目前仅在诚实客户端环境下验证，对攻击或恶意行为的鲁棒性尚未充分探讨。

---

## 120. Meaning over Motion: A Semantic-First Approach to 360° Viewport Prediction

**arXiv ID:** 2601.05416 | [PDF](https://arxiv.org/pdf/2601.05416v1)

**作者:** Arman Nik Khah `[一作]` (University of Texas at Dallas), Ravi Prakash `[通讯]` (University of Texas at Dallas)

**通讯引用:** 6227 | [OpenAlex ID](https://openalex.org/A5091471408)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于语义适应的共形分块与关联前瞻的360°视频流方案。

**💡 创新点**

通过将语义推理离线到服务器生成轻量关联图，结合动态安全边距与关联前瞻，克服了传统运动预测的“扫视陷阱”。

**🔧 技术方法**

采用服务器端语义分割+关联图、客户端轻量变压器预测、分层共形预测与自适应风险调节等技术。

**📊 数据集**

使用360-AV-HM数据集进行追踪驱动评估。

**📈 对比分析**

与LSTM轨迹预测、SalViT360显著性模型以及通用共形预测基线对比，停顿持续时间降低≥20%，带宽节省≥18%。

**⚠️ 局限性**

局限在离线预处理对实时流的适用性不足，且未考虑音频关联与更复杂场景的多模态推理。

---

## 121. A Technical Report on the Second Place Solution for the CIKM 2025 AnalytiCup Competition

**arXiv ID:** 2601.05259 | [PDF](https://arxiv.org/pdf/2601.05259v1)

**作者:** Haotao Xie `[一作]` (Hangzhou International Innovation Institute), Yuanyuan Liu `[通讯]` (Hangzhou International Innovation Institute)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于大型语言模型的单模型多语种商品类别相关性判断框架，通过链式思维提示拆分为翻译、意图理解、类别匹配和相关性判断四步，并利用LoRA进行轻量级微调；

**💡 创新点**

创新点在于用结构化提示引导单一模型实现类似集成模型的可解释性与鲁棒性，同时通过低秩适配实现高效、低成本的部署；

**🔧 技术方法**

核心技术包括Chain-of-Thought（CoT）提示工程、Low‑Rank Adaptation（LoRA）参数化微调、Qwen2.5‑14B基础模型以及高吞吐量推理管线；

**📊 数据集**

实验数据集为CIKM 2025 AnalytiCup 竞赛提供的多语言查询–类别对标注数据；

**📈 对比分析**

与传统集成基线对比，单模型在公共榜单得分0.8902、私有榜单得分0.8889，均显著高于基线0.8698，并在单张A100 GPU上实现20样本/秒的推理速度；

**⚠️ 局限性**

局限性包括仍以文本为主，翻译步骤可能导致语义损失；对未知语言/域的泛化能力待验证；对多模态信息缺乏支持。

---

## 122. The Facade of Truth: Uncovering and Mitigating LLM Susceptibility to Deceptive Evidence

**arXiv ID:** 2601.05478 | [PDF](https://arxiv.org/pdf/2601.05478v1)

**作者:** Herun Wan `[一作]` (Xi'an Jiaotong University), Min-Yen Kan `[通讯]` (National University of Singapore)

**通讯引用:** 11243 | [OpenAlex ID](https://openalex.org/A5066305082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并评估大型语言模型（LLM）在面对“硬不可证伪”误导性证据时的鲁棒性，构建了一套多轮协作的证据生成框架（Planner-Reviewer-Refiner）和评估框架（MisBelief），并在此基础上提出了基于意图识别的防护机制（Deceptive Intent Shielding）。

**💡 创新点**

① 生成能让LLM高度信服的误导证据的多角色协作生成流程；② 以硬不可证伪误导证据为核心的新评估框架MisBelief；③ 将意图识别引入防护，构成Deceptive Intent Shielding。

**🔧 技术方法**

多角色LLM生成（Planner、Reviewer、Refiner）实现证据逐步逼真；5分Likert尺度测量LLM内部信仰；对抗式迭代改进证据；意图分析器（Analyst）提取误导意图并做警告；使用GPT-4.1、GPT-5、Qwen系列等模型进行评估。

**📊 数据集**

从LatestEval新闻数据抽取8个领域（健康、科技、体育等）生成4,800个实例，按Easy/Medium/Hard三层难度划分；另收集16个真实误导案例用于验证；所有证据均经过多轮生成后提供给模型。

**📈 对比分析**

与无证据、Planner单纯生成、不同轮数迭代生成的证据对比，7大模型在硬难度下信仰分平均提升93%；通过Deceptive Intent Shielding可将提升幅度降低8–41%；模型在Hard情形下从3.0以下提升到3.0以上，显示误导证据对决策建议的实质性影响。

**⚠️ 局限性**

仅评估了7个代表性模型，未覆盖所有LLM；未对长期社会或行为影响进行评估；生成的误导证据具备恶意使用风险；防护方案仅探讨意图识别，缺乏其他多维度对策。

---

## 123. Towards Valid Student Simulation with Large Language Models

**arXiv ID:** 2601.05473 | [PDF](https://arxiv.org/pdf/2601.05473v1)

**作者:** Zhihao Yuan `[一作]` (Carnegie Mellon University), Tom Mitchell `[通讯]` (Carnegie Mellon University)

**通讯引用:** 48104 | [OpenAlex ID](https://openalex.org/A5102921433)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于大语言模型的学生仿真方法，并构建了“认知状态规范（ESS）”与目标-环境（Goal‑by‑Environment）框架。

**💡 创新点**

将学生仿真定义为受限生成，并引入E0–E4四级认知状态规范来系统化“能力悖论”，提供了可对齐评估与可复现的设计标准。

**🔧 技术方法**

采用大语言模型、提示工程、受限解码、外部学习状态表征（如知识追踪、误区图）以及结构化认知模型，实现对学生行为的受限生成。

**📊 数据集**

未在论文中使用具体实验数据集，主要以公开教育研究与模拟数据为参考进行概念性阐述。

**📈 对比分析**

通过目标行为一致的评估指标（误差分布、一致性、学习轨迹）进行比较，论文未给出定量性能结果，侧重框架的一致性与可比性。

**⚠️ 局限性**

主要局限在于缺乏实证验证、依赖黑盒LLM导致知识泄露风险、评估标准尚未统一，框架在随机性强的模型上可能难以完全实施。

---

## 124. Memory Poisoning Attack and Defense on Memory Based LLM-Agents

**arXiv ID:** 2601.05504 | [PDF](https://arxiv.org/pdf/2601.05504v1)

**作者:** Balachandra Devarangadi Sunil `[一作]`, Shuchi Mishra `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文系统评估了在电子健康记录（EHR）代理中通过查询注入的记忆中毒攻击的鲁棒性，并提出了两种基于信任的防御机制，分别是输入/输出审核与记忆消毒；

**💡 创新点**

创新点在于：①在更真实的部署环境下（含已有合法记忆、可变检索数量等）量化攻击效果；②设计了多信号复合信任评分的两阶段审核与时间衰减、模式过滤的记忆消毒；③对比不同模型（GPT‑4o‑mini、Gemini‑2.0‑Flash、Llama‑3.1‑8B‑Instruct）和不同攻击/防御配置，揭示了模型对信任评估的脆弱性；

**🔧 技术方法**

技术主要包括：记忆注入攻击技术（桥接步骤、指示提示、逐步压缩）、Levenshtein 距离检索、基于 LLM 的语义审核、代码安全静态分析、沙盒执行、信任得分计算、时间衰减与模式过滤；

**📊 数据集**

使用的数据集是 MIMIC‑III 临床记录，结合人工构造的患者 ID 换绑攻击场景；

**📈 对比分析**

与无防御基线对比，攻击成功率（ASR）在无初始记忆时可达 60% 以上，但在包含合法记忆时可降至 6% 以下；防御实验表明 GPT‑4o‑mini 的记忆消毒能完全拒绝所有候选记忆，保持 0% 泄露；但 Gemini‑2.0‑Flash 在高信任阈值下误将 54% 的恶意记忆加入，显示防御对模型过度自信的脆弱；

**⚠️ 局限性**

局限性包括：①实验规模受限，未覆盖大规模真实记忆库；②防御阈值调参需手工，缺乏自适应机制；③防御依赖 LLM 语义判断，易被模型操纵；④未评估防御对系统整体性能与用户体验的长期影响。

---

## 125. Multi-task Cross-modal Learning for Chest X-ray Image Retrieval

**arXiv ID:** 2601.05399 | [PDF](https://arxiv.org/pdf/2601.05399v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 126. Naiad: Novel Agentic Intelligent Autonomous System for Inland Water Monitoring

**arXiv ID:** 2601.05256 | [PDF](https://arxiv.org/pdf/2601.05256v1)

**作者:** Eirini Baltzi `[一作]` (National Technical University of Athens), Konstantinos Karantzalos `[通讯]` (National Technical University of Athens)

**通讯引用:** 4022 | [OpenAlex ID](https://openalex.org/A5064461457)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了 NAIDAD，一个基于大型语言模型的 agentic AI 助手，利用单一自然语言提示、检索增强生成（RAG）和动态构造的有向无环图（DAG）来实现内陆水体质量监测与报告生成。

**💡 创新点**

其创新点在于将 RAG 与 LLM 推理相结合，实时动态生成并执行多工具工作流（如 Sentinel‑2 数据检索、NDCI 计算、CyFi 蓝藻预测、天气增强），并通过自省机制纠正输出，从而实现跨工具协同与用户适配的单代理交互。

**🔧 技术方法**

所用技术包括 Qwen‑2.5 (14B) 与 Gemma‑3 (27B) LLM、Ollama 本地推理、Llama Index 与 BAAI/bge‑large‑en‑v1.5 向量检索、LLM 工具调用、DAG 编排、外部 API（Sentinel‑2、天气、CyFi 等）以及自动报告生成模块。

**📊 数据集**

实验数据集为自建的三湖（Lysimachia、Trichonida、Mornos）监测案例集，包含 Sentinel‑2 影像、NDCI、气象参数、蓝藻预测结果及文献知识库；评测还采用 GeoLLM‑Squad 任务生成协议。

**📈 对比分析**

评估采用正确率（Correctness）和输出相关性（Relevancy）指标；Qwen‑2.5 14B 在正确率上达到 82.98%，相关性 78.72%；Gemma‑3 27B 正确率同为 82.98%，相关性 68.09%；在多专业水平的用户查询中，工具调用正确率超过 77%，输出相关性超过 85%。

**⚠️ 局限性**

主要局限在于单代理架构仍难以处理高度分布式、多领域任务；工具集主要围绕 Sentinel‑2 与预训练模型，缺乏对其他遥感源和实时地面传感器的充分集成；对模型规模和算力的依赖限制了在资源受限环境中的部署；数据覆盖仍以希腊湖泊为主，跨区域适应性需进一步验证。

---

## 127. Evaluating the Use of LLMs for Automated DOM-Level Resolution of Web Performance Issues

**arXiv ID:** 2601.05502 | [PDF](https://arxiv.org/pdf/2601.05502v1)

**作者:** Gideon Peters `[一作]` (Concordia University), Emad Shihab `[通讯]` (Concordia University)

**通讯引用:** 7074 | [OpenAlex ID](https://openalex.org/A5049727493)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了九种前沿大语言模型在自动修复网页DOM以提升性能的效果。

**💡 创新点**

提出多维度DOM修改评估指标（如EATRR、PCD、深度分布），揭示模型在SEO/可访问性和性能改进上的差异，并首次量化模型的增删元素与位置变动模式。

**🔧 技术方法**

使用LLM生成DOM修改、Lighthouse 12.2.0 进行性能审计、树编辑距离验证重组、JSON差异工具统计DOM变更，并通过零-shot prompt 与固定温度 0.0 进行统一推理。

**📊 数据集**

15个来自 Alexa Top 500 的真实网页首页（涵盖购物、专业、社交、娱乐等四类），提取完整 DOM 树并进行拆分。

**📈 对比分析**

以 AIR（audit incidence ratio）变化百分比对比改进，优秀模型如 GPT‑4.1 与 Qwen2.5‑32B‑Instruct 在多项性能指标上可降低约 40‑60% 的问题占比，而多数模型引入视觉不稳定性，表现参差不齐。

**⚠️ 局限性**

仅关注 DOM 层面，未涉及 CSS/JS 交互；单次推理实验、缺乏多次重复或真实用户体验测评；LLM 选型与 prompt 设计对结果敏感。

---

## 128. How Exploration Breaks Cooperation in Shared-Policy Multi-Agent Reinforcement Learning

**arXiv ID:** 2601.05509 | [PDF](https://arxiv.org/pdf/2601.05509v1)

**作者:** Yi-Ning Weng `[一作]` (National Taiwan University), Hsuan-Wei Lee `[通讯]` (Lehigh University)

**通讯引用:** 6812 | [OpenAlex ID](https://openalex.org/A5065210070)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在动态囚徒困境环境中，系统性研究了参数共享深度Q网络（DQN）在不同探索强度、支付严峻度、网络拓扑和状态可观测性下的合作稳定性。

**💡 创新点**

揭示了共享表示导致的“合作崩溃”现象，并证明其根源是代表性耦合与部分可观测性，而非激励冲突或训练不稳定；提出分组学习和状态增强等可缓解措施。

**🔧 技术方法**

采用多智能体强化学习框架、共享参数的DQN、Softmax探索、n步回报、AdamW优化、UMAP可视化和轮廓系数等技术。

**📊 数据集**

使用合成的动态囚徒困境实验环境，在二维格子、模块化、随机和小世界四种网络拓扑上进行训练与评估。

**📈 对比分析**

通过比较共享DQN、分组DQN以及不同状态增强方式的合作率和收敛性；结果显示共享DQN在高探索或高惩罚时出现显著合作下降，而分组和状态增强能显著延缓或部分恢复合作，性能改进取决于拓扑与探索强度。

**⚠️ 局限性**

局限在于仅针对共享参数的DQN，且实验基于简化的囚徒困境；可能无法直接推广到复杂任务或其他强化学习算法，且只关注合作率而非更全面的任务性能。

---

## 129. Enabling Stroke-Level Structural Analysis of Hieroglyphic Scripts without Language-Specific Priors

**arXiv ID:** 2601.05508 | [PDF](https://arxiv.org/pdf/2601.05508v1)

**作者:** Fuwen Luo `[一作]` (Tsinghua University), Yang Liu `[通讯]` (Tsinghua University)

**通讯引用:** 111042 | [OpenAlex ID](https://openalex.org/A5100355638)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了Hieroglyphic Stroke Analyzer (HieroSA)，一种能够在没有人工注释的情况下，自动从字符位图中提取并表示笔画结构的框架。

**💡 创新点**

创新点包括：①仅使用字符位图作为监督，构造无监督的强化学习奖励函数；②将笔画编码为两点的线段，避免对笔画数量的先验假设；③通过RL训练MLLM，使其获得可解释的几何笔画表示，兼具跨语言泛化能力。

**🔧 技术方法**

技术方法：强化学习（GRPO）+ 结构化奖励（覆盖率、无效笔画惩罚）+ 线段表示；采用 Qwen3‑VL‑4B‑Instruct 作为基础模型进行微调；通过坐标系统增强位置预测。

**📊 数据集**

数据集：从中文（简体/繁体）字符、日文汉字以及甲骨文（Oracle Bone Script）三种书写系统中采集的位图；训练集包含数千幅字符图像，无需任何标注。

**📈 对比分析**

与基线模型（未使用笔画信息的直接视觉或文本模型）比较，HieroSA 在奖励（RE）、前景覆盖率（CO）和无效笔画比例（IS）上均实现显著提升；在跨语言评估中保持较高性能，并在 OCR 与结构引导字符检索任务中取得提升。

**⚠️ 局限性**

局限性：①对图像噪声和结构复杂度敏感；②实验规模受限于中等规模模型，未探索更大模型的潜力；③单模型设置缺乏集成策略；④结构引导检索的结果仍需人工筛选，缺乏自动化排序机制。

---

## 130. FlashMem: Distilling Intrinsic Latent Memory via Computation Reuse

**arXiv ID:** 2601.05505 | [PDF](https://arxiv.org/pdf/2601.05505v1)

**作者:** Yubo Hou `[一作]` (Beihang University), Zengchang Qin `[通讯]` (Beihang University)

**通讯引用:** 2937 | [OpenAlex ID](https://openalex.org/A5032405950)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FlashMem 框架，利用 LLM 后向缓存共享的 KV 直接从计算流中提取内存，避免额外编码和参数分离

**💡 创新点**

创新点：把记忆抽取内嵌于 LLM 的后向缓存，采用共享 KV 生成器和基于注意力熵的无参认知监测器动态触发记忆生成

**🔧 技术方法**

技术：共享 KV 记忆整合器、投影无关交叉注意力、权重继承初始化、注意力熵无参监测

**📊 数据集**

数据集：GSM8K、MATH、GPQA、KodCode、BookSum、GovReport 等多任务数据集

**📈 对比分析**

与 Vanilla、CoT‑SC、SnapKV、MemGen 等 baseline 对比，性能与 MemGen 相当或略优，推理延迟约 5 倍更低，VRAM 近似 Vanilla

**⚠️ 局限性**

局限：目前仅适用于文本/代码任务，未扩展到多模态；未验证在亿级参数模型上的可扩展性

---

## 131. Over-Searching in Search-Augmented Large Language Models

**arXiv ID:** 2601.05503 | [PDF](https://arxiv.org/pdf/2601.05503v1)

**作者:** Roy Xie `[一作]` (Duke University), Bhuwan Dhingra `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究了搜索增强型大语言模型在不回答能力（unanswerable）查询中出现的“过度搜索”（over-searching）现象，并提出了相应的评估指标与对策；

**💡 创新点**

创新点包括：①首次将过度搜索问题拆解为答案准确率与放弃回答准确率的平衡，②提出Tokens Per Correctness（TPC）度量来量化搜索成本与正确率的权衡，③构建了1,188条平衡的答案可/不可回答问答基准，④从检索质量、推理深度、多轮会话等多维度系统分析过度搜索；

**🔧 技术方法**

采用检索-生成工具结合的LLM（如GPT-4o-mini、o4-mini、Qwen3-235B-Think等），在检索层使用Wikipedia、C5、Web Search等语料；利用LLM判断器评估答案与放弃回答的准确性；通过Prompt工程（abstention-aware、few-shot、self-eval）和检索层（合成负证据）进行对策实验；

**📊 数据集**

使用自研的	exttt{<ABSTAINBENCH>}基准（由HotpotQA、SimpleQA、Natural Questions等数据集筛选并配对的答案可/不可回答问题）以及标准检索语料（Wikipedia最新/旧版、C5、Web搜索），并对查询进行长度与语义相似度控制；

**📈 对比分析**

对比了多种模型在是否启用搜索、不同检索源、不同推理深度、以及多轮会话情境下的答案准确率、放弃回答准确率和TPC。结果显示，启用搜索可提升约24%答案准确率，但放弃回答准确率下降约12.8%；TPC随搜索步数增加呈单调上升，表明搜索成本过高；查询级与检索级对策在一定程度上提升放弃回答准确率，但提升幅度有限；

**⚠️ 局限性**

局限性在于：①仅采用无训练的对策，未探究模型微调或架构改进；②基准查询来源于已有数据集，可能与真实搜索日志分布不同；③对检索质量的依赖导致在真实互联网环境下的泛化性尚待验证；④对策改进效果有限，需进一步研究更有效的搜索决策机制。

---

## 132. The Evaluation Gap in Medicine, AI and LLMs: Navigating Elusive Ground Truth & Uncertainty via a Probabilistic Paradigm

**arXiv ID:** 2601.05500 | [PDF](https://arxiv.org/pdf/2601.05500v1)

**作者:** Aparna Elangovan `[一作]` (Independent Researcher), Dan Roth `[通讯]` (University of Pennsylvania)

**通讯引用:** 31208 | [OpenAlex ID](https://openalex.org/A5023802054)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

论文提出一种概率框架，利用专家共识程度来评估 AI 系统在医学图像诊断中的性能。

**💡 创新点**

创新点在于将地面真值的不确定性量化为 p_d，并通过期望准确率和 F1 计算，揭示低置信度下专家与 AI 的差距会被掩盖。

**🔧 技术方法**

采用概率推导、模拟实验以及对实际医学数据集的分层评估方法。

**📊 数据集**

使用 CheXpert 胸部 X 光图像数据集和德国全国乳腺筛查（mammogram）数据集。

**📈 对比分析**

与多模态 LLM（Gemini‑3、GPT‑5.1 等）和人类放射科医生进行分层比较；在高 p_d（≈1）时人类表现显著优于模型，而低 p_d 时两者差距缩小甚至模型优于人类。

**⚠️ 局限性**

局限性包括仅针对二分类任务，且需收集多份专家标注；多标签问题需要更多注释以保证证据质量。

---

## 133. Prompt-Free SAM-Based Multi-Task Framework for Breast Ultrasound Lesion Segmentation and Classification

**arXiv ID:** 2601.05498 | [PDF](https://arxiv.org/pdf/2601.05498v1)

**作者:** Samuel E. Johnny `[一作]` (Carnegie Mellon University), Assane Gueye `[通讯]` (Carnegie Mellon University)

**通讯引用:** 295 | [OpenAlex ID](https://openalex.org/A5016748726)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种无提示的多任务深度学习框架，联合完成乳腺超声图像的肿瘤分割与诊断分类。

**💡 创新点**

将Segment Anything Model（SAM）的视觉编码器直接用于医学图像，无需外部提示，采用掩膜引导的注意力机制实现分割与分类的互相强化。

**🔧 技术方法**

利用SAM的Vision Transformer编码器、轻量级卷积或U‑Net式解码器、掩膜引导注意力模块、交叉熵与Dice损失的多任务优化。

**📊 数据集**

在PRECISE 2025乳腺超声数据集上进行实验，该数据集整合了BUSI、BrEAST和BUS‑BRA三大公开数据集，共计2,508张图像。

**📈 对比分析**

与三种基线方法（多任务基线、BI‑RADs集成、EDCNN）比较，取得最高的分割Dice 0.887、分类准确率90.7%（最高92.3%）和AUC 0.981，整体性能优于现有方法。

**⚠️ 局限性**

缺点包括对SAM编码器预训练数据的依赖、掩膜引导注意力机制在多实例病例中可能仍受限、以及未对多视角或时间序列超声数据进行评估。

---

## 134. Assembling Solar Panels by Dual Robot Arms Towards Full Autonomous Lunar Base Construction

**arXiv ID:** 2601.05491 | [PDF](https://arxiv.org/pdf/2601.05491v1)

**作者:** Luca Nunziante `[一作]` (Sapienza University of Rome), Kazuya Yoshida `[通讯]` (Tohoku University)

**通讯引用:** 13030 | [OpenAlex ID](https://openalex.org/A5023419492)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在月球基地场景下，利用双臂机器人开发并实现了全自主的太阳能电池板装配流水线，集成视觉识别、姿态估计、非线性模型预测控制、阻抗与力控制以及定制抓取模块。

**💡 创新点**

创新点包括：①使用 YOLOv8.1 预测定向边界框并结合深度反投影获得 6D 位姿；②将 NMPC 与非线性碰撞约束结合，实现负载安全提升；③采用阻抗-力控制双模式协作完成插入；④设计专用抓取模块以补偿对齐误差。

**🔧 技术方法**

所用技术：YOLOv8.1 定向检测、RGB‑D 深度反投影、阻抗控制、力控制、非线性模型预测控制（NMPC）、xArm7 7‑DoF 机器人 + Realsense D435i + F/T 传感器。

**📊 数据集**

训练数据集：先在公开 DOTA‑v1.0 数据集预训练，然后在包含 309 张图像、两类（电池板斑点与连接器）的自定义数据集上微调。

**📈 对比分析**

在 40 次真实场景试验中，整体成功率为 61%，失败主要为抓取失误（66%）和插入失误（34%）；相较传统手动或半自动流程，展示了流水线的可行性和自主性，但尚未给出数值基准对比。

**⚠️ 局限性**

主要局限：深度估计噪声导致抓取失败；对阻抗/力参数的手工调优不够系统；缺乏更精确的 6D 感知或深度重建方法，需进一步提升成功率。

---

## 135. DIFF-MF: A Difference-Driven Channel-Spatial State Space Model for Multi-Modal Image Fusion

**arXiv ID:** 2601.05538 | [PDF](https://arxiv.org/pdf/2601.05538v1)

**作者:** Yiming Sun `[一作]` (Southeast University), Pengfei Zhu `[通讯]` (Southeast University)

**通讯引用:** 19831 | [OpenAlex ID](https://openalex.org/A5006952581)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种差异驱动的通道-空间状态空间模型（DIFF‑MF）用于多模态图像融合，能够同时保留红外热信息与可见图像纹理细节。

**💡 创新点**

创新点包括：①使用特征差异图引导特征提取，突出两模态的独特信息；②在通道维度通过跨模态SSD交换实现自适应重加权；③在空间维度采用多尺度跨模态扫描，完成全局融合；④保持线性计算复杂度。

**🔧 技术方法**

采用了Mamba结构的可变状态空间模型（VSS），双分支差异引导网络、通道交换模块、空间交换模块以及SSIM、纹理、强度三项损失。

**📊 数据集**

实验使用M^3FD、TNO、DroneVehicle三大公开红外-可见图像融合数据集，并在FMB数据集上评估下游目标检测与语义分割任务。

**📈 对比分析**

与SwinFusion、TarDAL、DIDFuse、CDDFuse、MambaDFuse、FusionMamba、EMMA等七种先进方法对比，DIFF‑MF在EN、SD、SF、MI、VIF、AG等多项指标上均居首位，并在目标检测与分割上取得更高mAP与mIoU。

**⚠️ 局限性**

局限性：目前仅在三类特定的红外‑可见数据集上验证，泛化到其他模态或更复杂环境需进一步研究；模型仍存在一定的参数量和推理时间，尤其在极端低光或大尺寸图像上可能受限。

---

## 136. Learning specifications for reactive synthesis with safety constraints

**arXiv ID:** 2601.05533 | [PDF](https://arxiv.org/pdf/2601.05533v1)

**作者:** Kandai Watanabe `[一作]` (Google LLC), Morteza Lahijanian `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1665 | [OpenAlex ID](https://openalex.org/A5069564559)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种从演示数据中学习概率确定有限自动机（PDFA）并在动态环境中合成多目标安全策略的方法，能在不依赖先验任务描述的情况下实现机器人自主完成复杂任务；

**💡 创新点**

创新点在于（1）将任务建模为PDFA并在学习过程中直接嵌入安全约束；（2）开发了预处理式安全学习算法，避免后处理对概率分布的破坏；（3）设计了基于Pareto前沿的多目标反应式合成算法，能给出全部最优成本‑偏好权衡；

**🔧 技术方法**

主要技术包括基于证据驱动状态合并的语法推断（EDSM）、概率 DFA（PDFA）学习、LTL安全 DFA 构造、产品游戏构建以及基于值迭代的多目标Pareto前沿计算；

**📊 数据集**

实验数据来源于多种演示集：网格世界、MiniGrid、机器人操纵（Franka Panda）与人机协作（鸡尾酒调制）等场景的演示轨迹；

**📈 对比分析**

与传统基于公式推理、后处理安全的学习方法相比，本文方法在学习时间（≤0.01 s）、精度（L1误差≤1.7×10⁻³）和生成的最优策略数上均具优势；

**⚠️ 局限性**

主要限制包括：①产品游戏规模随状态数呈指数增长，影响大规模场景的可扩展性；②需手动设置合并阈值α，影响学习结果与安全性；

---

## 137. Safety Not Found (404): Hidden Risks of LLM-Based Robotics Decision Making

**arXiv ID:** 2601.05529 | [PDF](https://arxiv.org/pdf/2601.05529v1)

**作者:** Jua Han `[一作]` (Dongguk University), Jihie Kim `[通讯]` (Dongguk University)

**通讯引用:** 2555 | [OpenAlex ID](https://openalex.org/A5080664764)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并评估了七个面向安全的任务（完整信息、缺失信息和安全导向空间推理），系统测试大语言模型与视觉‑语言模型在火灾疏散、地图导航、序列推理等场景下的空间推理与决策可靠性。

**💡 创新点**

创新点包括：①将ASCII网格映射与自然语言指令相结合，消除视觉歧义；②设计安全导向空间推理（SOSR）任务，用自然语言评估模型在高风险决策中的鲁棒性；③通过细粒度成功率、路径连通性和结构完整性三维评价指标，揭示传统“99%准确率”评估的误导性。

**🔧 技术方法**

使用了自然语言提示、ASCII地图、序列帧合成、视觉‑语言模型推理（LLaVA、Qwen、InternVL）、大语言模型API（Gemini‑2.5 Flash、Gemini‑2.0 Flash、GPT‑5、GPT‑4o、LLaMA‑3‑8b）以及基于规则的路径验证脚本。

**📊 数据集**

数据集为自定义：①完全信息的三张ASCII地图（易/中/难）与两张不确定地形地图；②100条室内/室外导航序列（五帧/两帧）；③“后门”场景的真实图像与火灾疏散文本描述；⑤SOSR任务中的多难度自然语言情景。

**📈 对比分析**

评估方法：每个模型30次独立试验，计算成功率并对路径连通性、障碍规避和地图结构完整性进行判定；在SOSR任务中执行100次重复提问，统计错误率和熵值。性能上：GPT‑5在绝大多数任务实现100%成功率；Gemini‑2.0 Flash、GPT‑4o表现良好；Gemini‑2.5 Flash、LLaMA‑3‑8b 以及多种VLM在某些任务（尤其是不确定地图和SOSR）出现灾难性失败（0%或低于10%）。

**⚠️ 局限性**

限制：仅使用单一RTX‑6000 GPU，限制模型规模与并行度；数据量有限（100条序列、少量ASCII地图），未覆盖更大多样化环境；评估侧重于文本输出，缺乏真实机器人执行验证；未对模型内部推理机制做深入分析。

---

## 138. Efficient Temporal-aware Matryoshka Adaptation for Temporal Information Retrieval

**arXiv ID:** 2601.05549 | [PDF](https://arxiv.org/pdf/2601.05549v1)

**作者:** Tuan-Luc Huynh `[一作]` (Monash University), Thanh-Toan Do `[通讯]` (Monash University)

**通讯引用:** 2478 | [OpenAlex ID](https://openalex.org/A5025723803)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Temporal‑aware Matryoshka Representation Learning (TMRL)，为检索器构建时间感知的嵌入；

**💡 创新点**

创新点在于将Matryoshka嵌入的前t维专门划分为时间子空间，并通过自蒸馏和对比学习强化时间信息；

**🔧 技术方法**

采用LoRA参数高效微调、时间投影器、时间对比学习、局部/全局自蒸馏等技术；

**📊 数据集**

使用改进后的Temporal Nobel Prize (TNP) 数据集与 TimeQA 数据集进行训练与评测；

**📈 对比分析**

与 BM25、原始TEM、Ts‑Retriever、TempRetriever、M‑Adaptor 等基线相比，在TNP与TimeQA上实现了更高的 nDCG@10 与 F1，且能在不同嵌入维度下灵活权衡精度与效率；

**⚠️ 局限性**

局限包括未在大型 LLM 嵌入模型上验证、仅处理单一时间表达的段落、依赖非最先进 LLM 生成对比查询、以及对多时间表达文档的支持不足。

---

## 139. Can Large Language Models Differentiate Harmful from Argumentative Essays? Steps Toward Ethical Essay Scoring

**arXiv ID:** 2601.05545 | [PDF](https://arxiv.org/pdf/2601.05545v1)

**作者:** Hongjin Kim `[一作]` (Konkuk University), Harksoo Kim `[通讯]` (Konkuk University)

**通讯引用:** 724 | [OpenAlex ID](https://openalex.org/A5022865376)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了HED（有害论文检测）基准，并对现有AES模型与多种LLM在检测与评分有害论文的能力进行系统评估。

**💡 创新点**

提出利用POR/POC度量LLM拒绝与规避诱导生成有害文本的能力，并将有害内容检测准则嵌入评分指令，显著降低有害论文分数，提升模型安全性。

**🔧 技术方法**

采用LLM指令微调、Perspective API毒性评分、分类与评分任务指令、POR/POC指标等技术进行实验与评测。

**📊 数据集**

使用IELTS写作数据集生成提示，构建HED基准（100条论证论文、190条有害论文），并在此基础上训练与评估AES模型与LLM。

**📈 对比分析**

通过精确率、召回率、F1评价LLM的有害检测性能，使用QWK评估评分准确度；实验表明LLM相较传统AES更能识别有害内容，但仍倾向于给有害论文较高分，改进指令后有害论文得分显著下降。

**⚠️ 局限性**

实验仅覆盖参数<10B的LLM，HED基准未为有害论文提供金标准分数，且模型规模和数据多样性受限，未来需扩展模型规模、增加人类评标以提升可靠性。

---

## 140. Empirical Characterization of Logging Smells in Machine Learning Code

**arXiv ID:** 2601.05540 | [PDF](https://arxiv.org/pdf/2601.05540v1)

**作者:** Patrick Loic Foalem `[一作]` (Polytechnique Montreal), Heng Li `[通讯]` (Polytechnique Montreal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对444个活跃Python机器学习开源仓库进行AST提取、GPT‑4o‑mini辅助识别和人工验证，构建并实证验证了ML日志smell分类法，并在Google Forms问卷中评估其在业界的感知频率、严重性和重要性。

**💡 创新点**

首次将传统日志smell研究拓展至机器学习工作流，提出并验证了ML特有的日志smell，并创新性地将LLM自动化识别与双人人工标注相结合的混合方法。

**🔧 技术方法**

使用AST解析、GPT‑4o‑mini自动分类、双人人工标注、Cohen’s κ一致性评估、以及Google Forms问卷收集与分析。

**📊 数据集**

444个活跃Python ML开源仓库（共86,143条日志语句）以及12种通用/ML专用日志库作为数据集。

**📈 对比分析**

通过比较LLM分类与人工标注的一致率（Cohen’s κ>0.75）验证方法可靠性；问卷中各smell的平均Likert分数揭示其真实影响，未给出数值性能指标但方法可复现。

**⚠️ 局限性**

仅限Python生态的开源项目，无法推广到闭源或其他语言；LLM可能产生幻觉，需人工校验；受访者自选偏差可能影响问卷结果。

---

## 141. Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism

**arXiv ID:** 2601.05524 | [PDF](https://arxiv.org/pdf/2601.05524v1)

**作者:** Yuhao Shen `[一作]` (Zhejiang University), Cong Wang `[通讯]` (Zhejiang University)

**通讯引用:** 25160 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Double 的并行投机解码框架，通过双重检索（draft 与 target 端）实现无训练、无损失的加速，突破传统 PSD 的速度上限，显著提升大模型推理速度。

**💡 创新点**

创新点在于：1) 将检索应用于草稿端实现多轮迭代检索，打破 draft‑to‑target 速度比 C 的理论上限；2) 在目标端使用检索提供多 token 预检和前向指导，解决中途拒绝导致的计算浪费；3) 统一的双检索同步并行执行，消除流水线瓶颈，实现精度与效率的平衡。

**🔧 技术方法**

技术手段包括：多轮检索推理（Retrieval Forward）、并行检索解码架构、目标引导的验证与纠错、层次化检索数据库（静态 + 动态），以及在推理期间动态同步两侧检索。

**📊 数据集**

使用的公开数据集包括 HumanEval、GSM8K、CNN/DM、Alpaca、MT‑Bench，模型对比如 LLaMA‑2/3、Deepseek、Qwen3，分别配合小模型（7B/1.3B/0.6B）和大模型（70B/33B/32B）。

**📈 对比分析**

与 SD、PSD（PEARL、SpecBranch）、检索侧方法（Lookahead、PLD、Token Recycling、Ouroboros）以及训练型 SOTA EAGLE‑3 对比，Double 在多项基准上实现了 1.6×–5.3× 的速度提升，平均提升约 3.7×，并保持与目标模型相同的采样分布（lossless）。

**⚠️ 局限性**

局限性：使用统一的检索数据库和相同的检索深度，未充分挖掘草稿端和目标端各自的优势；对不同模型特性缺乏自适应检索深度与独立数据库，未来可进一步细化与自适应化。

---

## 142. Toward an Integrated Cross-Urban Accident Prevention System: A Multi-Task Spatial-Temporal Learning Framework for Urban Safety Management

**arXiv ID:** 2601.05521 | [PDF](https://arxiv.org/pdf/2601.05521v1)

**作者:** Jiayu Fang `[一作]` (University of Sydney), Junbin Gao `[通讯]` (University of Sydney)

**通讯引用:** 10955 | [OpenAlex ID](https://openalex.org/A5015817857)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种跨城市事故预防系统——Mamba Local-Attention Spatial–Temporal Network，利用多任务学习在多城市共享模型参数以统一预测事故风险。

**💡 创新点**

创新点包括：①将Mamba状态空间模型与局部遮蔽注意力相结合的STG-MA模块，用于抑制噪声并捕获长期时序依赖；②STS-MA模块通过多种语义图（道路拓扑、事故共现、POI相似）实现跨城市特征共享，同时保留城市特有语义空间；③整体采用共享-本地化参数的多任务框架实现跨城迁移学习。

**🔧 技术方法**

技术方法主要包括：Mamba（Selective State Space）序列建模；局部遮蔽多头注意力；图卷积网络（AGCN）与自适应邻接矩阵；空间格点与节点级嵌入融合；门控融合与轻量化输出头。

**📊 数据集**

使用真实城市事故数据集：纽约市（NYC）与芝加哥（Chicago），包含事故记录、POI、出租车行程、道路网和气象信息，按小时进行空间网格划分。

**📈 对比分析**

与GSNet、TWCCNet、ViT‑Traffic、MG‑STNet等SOTA模型在两类预测场景（全日、短时高频）下进行对比，Mamba模型在RMSE上平均降低≈6%、召回率提升≈8%、MAP提升≈5%，并在50%输入噪声下性能波动不到1%。

**⚠️ 局限性**

局限性包括：对极端稀疏或时序相位错位的数据仍易出现误差；不同城市语义数据不平衡导致迁移效果受限；模型规模与推理速度仍高，实际实时部署需要进一步压缩和加速；缺乏对隐私与联邦学习的考量。

---

## 143. Secure Text Entry using a Virtual Radial Keyboard with Dynamically Resized Keys and Non-Intrusive Randomization

**arXiv ID:** 2601.05516 | [PDF](https://arxiv.org/pdf/2601.05516v1)

**作者:** Yuxuan Huang `[一作]` (University of Minnesota), Evan Suma Rosenberg `[通讯]` (University of Minnesota)

**通讯引用:** 892 | [OpenAlex ID](https://openalex.org/A5063130930)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种虚拟径向键盘，结合每键入一次随机旋转和动态键体放大，并与当前主流的ISPR和传统QWERTY键盘在安全性与可用性上进行对比。

**💡 创新点**

创新点在于：①对键盘进行全局每键入一次的随机化，打破键位与键值的确定性；②通过无限延伸的选区与动态键体放大降低精度需求；③首次在VR文本输入中引入“均匀采样攻击”评估现有安全机制的脆弱性。

**🔧 技术方法**

使用的技术包括：Meta Quest 3与Unreal Engine 5.3实现VR环境；Viterbi解码（基于三元语法模型）与均匀采样攻击实现对ISPR的评测；对径向键盘的随机化与键体动态重排；统计分析采用Friedman、Conover等非参数检验。

**📊 数据集**

数据集主要有：MacKenzie短语集用于输入任务；big.txt（Nörvig词典）用于训练三元语法模型；此外，研究中使用了13句短语做实验前的预试。

**📈 对比分析**

与ISPR和QWERTY比较时，径向键盘在安全指标（ICR≈0.175、SS≈0.10）显著优于ISPR和QWERTY；在效率上，WPM最低（≈5.13），而QWERTY最快（≈11.39）；在可用性上，SUS最低（≈55）且NASA‑TLX最高（≈47）。

**⚠️ 局限性**

局限性包括：样本量有限、练习时间短导致用户对径向布局的熟悉度不足；仅评估了英文字母键盘，未涵盖数字与符号；双击误触是主要错误来源；更高级的攻击方式（如自定义语言模型）仍未彻底验证。

---

## 144. Hi-ZFO: Hierarchical Zeroth- and First-Order LLM Fine-Tuning via Importance-Guided Tensor Selection

**arXiv ID:** 2601.05501 | [PDF](https://arxiv.org/pdf/2601.05501v1)

**作者:** Feihu Jin `[一作]` (Peking University), Ying Tan `[通讯]` (Peking University)

**通讯引用:** 11301 | [OpenAlex ID](https://openalex.org/A5023089209)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种Hi-ZFO框架，将FO梯度与ZO估计结合，在LLM细调时通过层级分割优先更新高重要性层，低重要性层采用ZO探索；

**💡 创新点**

创新点在于把ZO从单纯的内存友好近似转化为“有益随机性”源，并利用基于FLOPs的动态分割实现梯度精度与探索度的平衡；

**🔧 技术方法**

采用FO（Adam）、ZO（随机扰动+有限差分）、动态规划分割、双流前向、损失耦合及梯度估计等技术；

**📊 数据集**

在多种生成与推理任务上评测：SciTLDR、DialogSum、WebQuestions、GSM8K、Math500、HumanEval；使用OPT、BLOOM、Qwen等大型模型；

**📈 对比分析**

与全微调、LoRA、GreenTrainer、MeZO、Bilevel-ZOFO等基线对比，Hi‑ZFO在ROUGE、准确率、pass@1上普遍领先，且显著降低显存与训练时长；

**⚠️ 局限性**

局限在于参数分割和ZO比例是固定的，未动态适应损失景观变化；仅在监督微调环境验证，未测试对偏好优化或RLHF等对齐场景的兼容性。

---

## 145. Coset Shaping for Coded Modulation

**arXiv ID:** 2601.05652 | [PDF](https://arxiv.org/pdf/2601.05652v1)

**作者:** Irina Bocharova `[一作]` (University of Tartu), Boris Kudryashov `[通讯]` (University of Tartu)

**通讯引用:** 737 | [OpenAlex ID](https://openalex.org/A5088880021)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种可同时对信息位和校验位进行几何级Shaping的Coset Shaping方案，适用于编码QAM/PAM调制

**💡 创新点**

创新点在于利用线性码的余子集作为Shaper，使得在不增加额外复杂度的情况下实现信息位与校验位的共同Shaping，并可在无限长码和高阶调制下逼近容量

**🔧 技术方法**

采用线性码与Coset Shaping、PAM映射、Voronoi类几何Shaping、BP译码以及随机Shaping集合的理论分析

**📊 数据集**

使用长码（如64800位NB QC‑LDPC码）与256‑QAM/64‑QAM信号进行仿真，比较不同Shaping（Coset、PAS、Sphere）方案

**📈 对比分析**

通过BP译码误码率仿真与Shannon/极限曲线对比，Coset Shaping在高SNR下略优于PAS和Sphere，尤其在误差底部表现更好

**⚠️ 局限性**

局限在于理论假设为无限长码和高阶调制，实际实现仍需针对不同调制和码长优化，且需要更多实验验证其在实际系统中的适用性

---

## 146. From Global to Local: Cluster-Aware Learning for Wi-Fi Fingerprinting Indoor Localisation

**arXiv ID:** 2601.05650 | [PDF](https://arxiv.org/pdf/2601.05650v1)

**作者:** Miguel Matey-Sanz `[一作]` (Universitat Jaume I), Sergio Trilles `[通讯]` (Universitat Jaume I)

**通讯引用:** 2053 | [OpenAlex ID](https://openalex.org/A5014771012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

在Wi‑Fi指纹定位前先对指纹数据进行空间或信号强度聚类，并在定位时仅在最相关的聚类内使用机器学习模型，以提高定位精度与计算效率。

**💡 创新点**

提出了一种基于最强AP子集的聚类估计机制，支持建筑层与楼层级别聚类，并将聚类过程嵌入训练与定位阶段，显著降低搜索空间并提升定位性能。

**🔧 技术方法**

采用K‑Means聚类、RSSI到线性尺度转换、最强AP子集匹配、KNN/加权KNN、XGBoost和CNNLoc等机器学习方法。

**📊 数据集**

使用UJIIndoorLoc、UTSIndoorLoc和TUT三大公开Wi‑Fi指纹数据集进行实验。

**📈 对比分析**

与无聚类基线以及现有聚类方法比较，建筑级聚类在大多数情况下将二维定位误差降低至约7–9米，Floor Detection Rate (FDR)略有下降；相较于其他研究，CNNLoc+聚类在UJI、UTS、TUT三组数据上取得最小定位误差且FDR可达90%以上。

**⚠️ 局限性**

聚类虽然能降低误差，但会牺牲楼层识别准确率；方法对数据集异质性敏感，需根据环境调优参数N和K；实验仅涵盖三数据集，未验证在更大规模或实时场景中的可扩展性。

---

## 147. Continual Pretraining on Encrypted Synthetic Data for Privacy-Preserving LLMs

**arXiv ID:** 2601.05635 | [PDF](https://arxiv.org/pdf/2601.05635v1)

**作者:** Honghao Liu `[一作]` (International Digital Economy Academy), Jian Guo `[通讯]` (International Digital Economy Academy)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小规模敏感语料上，通过加密实体并使用基于实体图的数据合成方法实现对大语言模型的持续预训练，保证个人身份信息（PII）的隐私；

**💡 创新点**

首次将确定性加密与实体图驱动的数据合成相结合，实现了在加密语料上训练 LLM 并能授权用户通过解密密钥获取原始信息；

**🔧 技术方法**

利用 AES-ECB 加密 PII、构建加权实体图、利用 GPT‑4o/DeepSeek 合成问答对、在 Llama3‑8B 与 Qwen2.5‑7B 上进行持续预训练；

**📊 数据集**

评估使用了英文学术 QA 数据集 QuALITY 与中文法庭案例 QA 数据集；

**📈 对比分析**

与基线模型、未加密合成预训练模型以及在原始数据上持续预训练模型对比，结果显示加密合成模型在保持 PII 安全的同时，性能仅略逊于未加密合成模型，且显著优于基线；

**⚠️ 局限性**

主要限制包括：使用 AES-ECB 产生的确定性加密易泄露等价性与频率信息、加密后序列长度增加导致上下文处理受限、方法仅在小规模手工可验证数据集上验证、对更大规模数据集的可扩展性未知、以及依赖 LLM 进行实体提取与合成可能导致幻觉与错误。

---

## 148. Continual Learning of Achieving Forgetting-free and Positive Knowledge Transfer

**arXiv ID:** 2601.05623 | [PDF](https://arxiv.org/pdf/2601.05623v1)

**作者:** Zhi Wang `[一作]` (Xidian University), Yuping Wang `[通讯]` (Xidian University)

**通讯引用:** 5438 | [OpenAlex ID](https://openalex.org/A5100339104)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于任务特定二进制掩码的连续学习方法ETCL，实现无遗忘且正向/反向知识迁移

**💡 创新点**

①将连续学习建模为优化问题，要求正向/反向迁移；②引入基于Wasserstein距离的在线任务相似度检测；③采用梯度对齐与双目标优化实现正向/反向迁移

**🔧 技术方法**

稀疏子网络掩码、Wasserstein距离、梯度对齐、正交梯度投影（GPM）、双目标损失

**📊 数据集**

11个任务序列：5个离散（PMNIST、CIFAR100、CIFAR100 Sup、MiniImageNet、5-Datasets）、4个相似（F-EMNIST-1/2、F-CelebA-1/2）和2个混合（EMNIST+F-EMNIST-1、CIFAR100+F-CelebA-1）

**📈 对比分析**

与18种强基线（网络扩展、正则化、经验重放、掩码、KT方法、集成模型）对比，ETCL在离散任务上ACC提升≈9.4%，在相似任务上FWT/BWT均为正且提升≈10%；在混合任务上ACC提升5–6%，优于所有基线

**⚠️ 局限性**

仍需进一步提升在极大任务数下的可扩展性和在极低相似度任务中的迁移效果，且实现复杂度较高

---

## 149. Data Augmented Pipeline for Legal Information Extraction and Reasoning

**arXiv ID:** 2601.05609 | [PDF](https://arxiv.org/pdf/2601.05609v1)

**作者:** Nguyen Minh Phuong `[一作]` (Japan Advanced Institute of Science and Technology), Ken Satoh `[通讯]` (Center for Juris-Informatics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于大语言模型的法律信息抽取数据增强管线，快速适配Deep PROLEG系统。

**💡 创新点**

创新在于利用few-shot提示生成模板与槽位，自动产生法律案例及对应PROLEG事实，显著降低人工标注工作量。

**🔧 技术方法**

使用ChatGPT（GPT‑3.5 Turbo、GPT‑4o mini、GPT‑4o）及少量人工示例进行few‑shot生成；对生成数据做fine‑tune神经语义解析器。

**📊 数据集**

构造的自定义法律案例集（约5000条）涵盖四类合同与20种槽位；还使用公开开源LLM（Qwen2.5‑14B、Meta‑Llama‑3‑8B）做对比。

**📈 对比分析**

与传统基于规则或句法的增强方法对比，实验显示在5000条增强集上模型精度>95%，显著提升信息抽取与推理准确率。

**⚠️ 局限性**

仍受限于模板与槽位设计的覆盖范围、LLM生成误差，以及在更复杂法律文本上的可迁移性和解释性不足。

---

## 150. Conformity Dynamics in LLM Multi-Agent Systems: The Roles of Topology and Self-Social Weighting

**arXiv ID:** 2601.05606 | [PDF](https://arxiv.org/pdf/2601.05606v1)

**作者:** Chen Han `[一作]` (University of Chinese Academy of Sciences), Xijin Tang `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究了LLM驱动多代理系统中网络拓扑对一致性行为的影响，使用可信度归一化更新规则评估中央聚合与分布式共识在误信息检测任务中的表现。

**💡 创新点**

提出了统一的可信度归一化汇聚规则，并通过自我加权参数α平衡自我决策与社会影响，系统性比较不同拓扑对效率、鲁棒性和错误级联的影响。

**🔧 技术方法**

利用LLM的自我置信评分、图网络通信、迭代更新算法和阈值二值化，采用中心-边缘一致性、准确率、收敛时间等指标进行评估。

**📊 数据集**

使用新构造的Snopes25数据集（448条真实声明，包含真假标签）进行二分类误信息检测。

**📈 对比分析**

在七代理星形、层级、环形到完全图等七种拓扑上实验，发现α=0.75在大多数设置下获得最佳准确率；中央聚合在高自我加权时更鲁棒，分布式共识在高连通性下收敛更快但易产生高置信错误。

**⚠️ 局限性**

主要局限在置信度自报不一定校准、实验规模固定为七人、仅三种LLM、仅二分类误信息任务，以及缺乏开放式询问或工具辅助等交互机制。

---

## 151. Learning Geometric Invariance for Gait Recognition

**arXiv ID:** 2601.05604 | [PDF](https://arxiv.org/pdf/2601.05604v1)

**作者:** Zengbin Wang `[一作]` (Beijing University of Posts and Telecommunications), Man Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 5908 | [OpenAlex ID](https://openalex.org/A5100353093)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过构建基于几何不变性的网络框架，改进步态识别模型以更好适应不同步态条件。

**💡 创新点**

创新点在于将步态条件差异视作可组合的几何变换（反射、旋转、尺度），并在网络中分别设计反射、旋转、尺度等价变换学习模块，使特征实现等变与不变，提升跨条件识别性能。

**🔧 技术方法**

使用等变卷积（ReEL、RoEL、SEL）实现反射、旋转和尺度的等变学习，随后通过水平池化与全连接实现不变特征；还采用自适应旋转角度预测与卷积旋转、跨尺度与跨通道注意力机制。

**📊 数据集**

在四个公开步态数据集上评估：Gait3D、GREW、CCPG、SUSTech1K，涵盖室内外、跨视角、跨服装等多种挑战场景。

**📈 对比分析**

与多种最新方法（如GaitSet、GaitSSB、HSTL、DyGait 等）对比，RRS‑Gait 在 Gait3D 上 Rank‑1 76.7%、GREW 上 81.0%，在 CCPG、SUSTech1K 上平均提升 4–6% 的 Rank‑1，mAP 亦显著提高，表明其在跨条件下具有更高的鲁棒性与准确性。

**⚠️ 局限性**

仅考虑了三种几何变换，未覆盖所有实际变化；等变卷积设计较为局部，缺乏通用的统一核结构；对尺度增广与更大范围旋转的适应性还有待进一步研究。

---

## 152. SceneAlign: Aligning Multimodal Reasoning to Scene Graphs in Complex Visual Scenes

**arXiv ID:** 2601.05600 | [PDF](https://arxiv.org/pdf/2601.05600v1)

**作者:** Chuhan Wang `[一作]` (University of California), Jingbo Shang `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于场景图的结构化偏好对齐框架（SceneAlign），通过对场景图进行可控干预生成高质量负面推理链，提升多模态大语言模型的视觉推理可信度。

**💡 创新点**

创新点在于：①利用场景图提供细粒度视觉结构，将负面样本的生成与视觉实体、关系直接关联；②设计四种可控干预策略（swap、replace、shorten、overthink），模拟常见的视觉推理错误；③将这些结构化负样本与正样本对齐应用直接偏好优化（DPO），实现结构感知的推理对齐。

**🔧 技术方法**

主要技术包括场景图生成（GPT‑4o）、结构化干预与负样本构造、对齐训练（DPO）、多模态大语言模型（Qwen、InternVL、LLaVA‑Next 等）。

**📊 数据集**

使用 A‑OKVQA 作为训练数据，评估基于 MME‑RealWorld、GQA、EMMA‑mini、SEEDBench、HallusionBench、ScienceQA、MMMU‑Reasoning 等七个视觉推理基准。

**📈 对比分析**

与预训练模型、仅正样本 SFT 以及两种基于文本的对齐方法（AoT、LLaVA‑Reasoner）对比，SceneAlign 在多模态推理任务上平均提升 3–5%，在 HallusionBench、EMMA 等推理密集型基准上表现尤为突出。

**⚠️ 局限性**

局限性：仅针对单图推理，未扩展至多图或视频；负样本生成与评估依赖 GPT‑based 自动工具，可能无法完全反映人类细粒度判断。

---

## 153. ACR: Adaptive Context Refactoring via Context Refactoring Operators for Multi-Turn Dialogue

**arXiv ID:** 2601.05589 | [PDF](https://arxiv.org/pdf/2601.05589v1)

**作者:** Jiawei Shen `[一作]` (Zhejiang Normal University), Jiajie Xu `[通讯]` (Soochow University)

**通讯引用:** 4343 | [OpenAlex ID](https://openalex.org/A5086062267)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Adaptive Context Refactoring (ACR) 框架，通过动态监测并重构多轮对话历史来缓解上下文惯性与状态漂移问题，提升大型语言模型在长对话中的一致性与事实准确性。

**💡 创新点**

创新点在于：①构建包含六类策略的上下文重构操作库；②引入教师引导的自我演化训练范式，让模型学习何时介入以及如何重构；③将上下文管理与推理过程解耦，形成闭环监测-重构管控。

**🔧 技术方法**

核心技术包括：语义路由器（Router）选择重构操作；LoRA 形式的重构器（Refactorer）执行具体操作；教师引导自我演化（TGSE）训练阶段；信息密度优化、逻辑流控制、注意力管理等六类重构策略。

**📊 数据集**

在七个多轮 QA 基准上评测：Natural Questions、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle。

**📈 对比分析**

与 Prompt、CoT、IRCoT、SFT、RAG（DRAGIN、DioR、SEAKR）、压缩（RECOMP）、外部记忆（HippoRAG、ITER-RETGEN）以及 RL 搜索（Search‑R1、StepSearch）等基线比较，ACR 在单跳和多跳任务均显著提升 EM 分数，性能接近或超过 RL 训练方法，同时 token 消耗显著下降（约 83%）。

**⚠️ 局限性**

局限性包括：部署时需要额外的路由器和重构器模块，增加系统复杂度与内存/推理延迟；目前评测仅在 QA 场景，缺乏对更复杂 agent、工具使用和动态环境的验证；未来需将重构能力内部化以简化架构。

---

## 154. GS-DMSR: Dynamic Sensitive Multi-scale Manifold Enhancement for Accelerated High-Quality 3D Gaussian Splatting

**arXiv ID:** 2601.05584 | [PDF](https://arxiv.org/pdf/2601.05584v1)

**作者:** Nengbo Lu `[一作]` (Guilin University of Electronic Technology), Yizhou Liang `[通讯]` (Guilin University of Electronic Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于3D高斯分布的动态场景重建方法 GS-DMSR，结合运动显著性驱动的动态高斯优化和多尺度流形增强模块，以加速模型收敛并提升渲染质量。

**💡 创新点**

创新点包括：1）引入运动显著性系数实现自适应梯度聚焦，对高显著性高斯进行迭代优化，降低低显著性高斯的冗余更新；2）构建多尺度流形增强模块，将隐式非线性解码器与显式变形场耦合，提升复杂变形建模能力并保持时空连续性。

**🔧 技术方法**

使用的技术包括3D Gaussian splatting、可微分投影、基于球面谐波的颜色编码、基于变形场的位移/旋转/尺度预测，以及自适应梯度更新机制。

**📊 数据集**

实验使用了三大数据集：synthetic（D‑NeRF）用于基准对比，HyperNeRF（真实场景）以及人工合成的 Dynamic Object 数据集。

**📈 对比分析**

与多种现有方法（如 4D‑GS、3D‑GS、V4D、FFDNeRF 等）对比，GS‑DMSR 在 PSNR、SSIM、LPIPS 等指标上往往位列或接近前列，同时训练时间和 FPS 也表现出显著优势（如 8 分钟训练即可达 96 FPS）。

**⚠️ 局限性**

局限性在于：1）对运动显著性阈值的选择可能需要手工调参；2）在极度复杂动态场景或非常稀疏的输入下，显著性划分误差可能导致渲染质量下降；3）多尺度流形模块虽然提升了表现，但增加了模型复杂度与推理时的计算开销。

---

## 155. Learn to Evolve: Self-supervised Neural JKO Operator for Wasserstein Gradient Flow

**arXiv ID:** 2601.05583 | [PDF](https://arxiv.org/pdf/2601.05583v1)

**作者:** Xue Feng `[一作]` (University of California), Rongjie Lai `[通讯]` (Purdue University)

**通讯引用:** 1755 | [OpenAlex ID](https://openalex.org/A5088277298)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种自监督的 Learn‑to‑Evolve 框架，用单一神经算子学习 JKO 步进算子，并通过在训练过程中反复使用当前算子产生新的轨迹数据，实现无需预先求解每个 JKO 子问题即可得到 Wasserstein 梯度流的离散演化。

**💡 创新点**

创新点：
- 采用自监督学习机制，跳过传统的多步训练，直接学习全局的 JKO 算子；
- 在训练中交替生成轨迹数据与更新算子，形成“数据自演化”与“算子自优化”的闭环；
- 对该闭环给出收敛性分析，证明在满足 Lipschitz 条件下训练集会收敛到理想 JKO 轨迹；
- 使用 Transformer 结构捕获粒子间的全局非局部相互作用，解决密度表示的可变长度与顺序无关问题；
- 将时间步长作为网络输入，提升对不同 Δt 的适应性。

**🔧 技术方法**

技术手段：
- Wasserstein 梯度流的 JKO 变分离散（Wasserstein 原点算子）；
- 神经算子（Transformer + 自注意力/交叉注意力）预测位移场；
- 粒子基础离散与 Monte Carlo 损失；
- 结合 2-Wasserstein 距离与能量正则化的自监督损失；
- Adam 优化器、余弦学习率调度与累积损失比较；
- 边界条件、归一化、离散化错误的分析。

**📊 数据集**

数据集：
- 仅使用少量初始密度样本（如单一正态分布、随机矩形/三角分布）和能量参数族（聚合方程的 (p,q)、Porous Medium 的 (d,m)、Fokker‑Planck 的目标分布），
- 在训练过程中通过算子自身迭代产生新的轨迹数据，形成自生成的数据集；
- 没有使用公开的标准数据集，全部为仿真生成。

**📈 对比分析**

比较方法与性能：
- 与传统 JKO 数值求解（primal‑dual、back‑and‑forth、augmented Lagrangian）相比，学习算子不需要逐步求解子问题，显著降低算力与内存；
- 与多步训练的基线（固定数据集训练多步算子）相比，Learn‑to‑Evolve 在相同算子容量下收敛更快、能量下降更平滑、泛化能力更强；
- 在聚合方程、Porous Medium、Fokker‑Planck 三类问题上，预测结果与解析解或高精度数值解误差满足 O(Δt) 的理论预期；
- 在高维（d≤10）下仍能保持较低的 L1/L∞ 误差，证明算子在维度上具有良好可扩展性。

**⚠️ 局限性**

限制与不足：
- 需要能量在参数空间内满足 λ‑凸或 λ‑Lipschitz 条件，否则收敛理论不成立；
- 网络容量有限，极端参数或高度非凸能量可能导致学习失败；
- 训练过程对初始网络权重和学习率调度敏感，可能出现局部最优；
- 对高维问题仍受粒子采样误差影响，尤其在密度稀疏区域；
- 目前的泛化理论仍是经验性的，缺乏严格的泛化误差界定。

---

## 156. Strong Singleton-Like Bounds, Quasi-Perfect Codes and Distance-Optimal Codes in the Sum-Rank Metric

**arXiv ID:** 2601.05581 | [PDF](https://arxiv.org/pdf/2601.05581v1)

**作者:** Chao Liu `[一作]` (Hubei University), Dabin Zheng `[通讯]` (Hubei University)

**通讯引用:** 447 | [OpenAlex ID](https://openalex.org/A5072537287)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在 sum‑rank 度量下的覆盖码、准完美码和距离最优码，提出新的大小、覆盖半径和块长度上界，并给出了强单宁式界与多族距离最优及准完美码的构造。

**💡 创新点**

创新点在于①通过从汉明度量覆盖码构造 sum‑rank 覆盖码，推导出更紧的大小、覆盖半径和块长度上界；②给出大块长度下优于传统 Singleton‑like 界的强单宁式界；③构造无限族距离最优与准完美 sum‑rank 码，尤其是矩阵尺寸 2×m 与 2×2 的实例；④利用 Plotkin 求和得到新的距离最优码。

**🔧 技术方法**

采用 q‑多项式映射、线性化 Reed‑Solomon 与 BCH/ Boston 界、线性代数构造以及 Plotkin 求和等技术，对 sum‑rank 码的参数与覆盖半径进行理论分析与构造。

**📊 数据集**

本文为理论研究，主要通过符号参数（q、m、t 等）进行分析，并未使用具体实验数据集；所有结论均基于理论证明与符号计算。

**📈 对比分析**

通过与传统 Singleton‑like、Sphere‑packing、MDS、MSRD 等经典界比较，证明在块长度较大时新界更紧；构造的距离最优码满足覆盖半径条件 V_sr(q,2) > q^{…}，证明其为最优或近最优，且准完美码自动满足距离最优。

**⚠️ 局限性**

局限性包括：准完美码仅在矩阵尺寸 2×m 与 2×2、特定 q 取值下构造；对更一般矩阵尺寸或更大距离的准完美/近 MSRD 码尚无构造；覆盖半径上界仍可进一步改进。

---

## 157. Generalizable and Adaptive Continual Learning Framework for AI-generated Image Detection

**arXiv ID:** 2601.05580 | [PDF](https://arxiv.org/pdf/2601.05580v1)

**作者:** Hanyi Wang `[一作]` (Shanghai Jiao Tong University), Shilin Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2838 | [OpenAlex ID](https://openalex.org/A5101717685)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了三阶段领域持续学习框架，用于AI生成图像检测，可持续提升对新模型的泛化与鲁棒性。

**💡 创新点**

①基于LoRA的参数高效微调构建可迁移检测器；②设计渐进式数据增强链与K-FAC正则化实现少样本适应；③利用线性模式连通性进行参数线性插值平衡稳健性与可塑性。

**🔧 技术方法**

LoRA微调、渐进式数据增强链、K‑FAC Hessian近似、线性模式连通性插值以及CLIP‑ViT视觉语言模型。

**📊 数据集**

27种GAN/深伪/扩散模型（如StyleGAN2、DDPM、SDXL等）构建的时序基准，时间跨度从2017年到2024年。

**📈 对比分析**

与9个检测基线（CNNSpot、UnivFD、NPR等）及8种连续学习基线（Seq、EWC、A‑GEM等）对比，离线检测mAP提升+5.51%，持续学习平均准确率92.20%，显著优于现有方法。

**⚠️ 局限性**

仍受限于少量新模型样本、对极端后处理（如强JPEG压缩）鲁棒性不足，且在线更新需维护数据链与K‑FAC计算开销。

---

## 158. EvoQRE: Modeling Bounded Rationality in Safety-Critical Traffic Simulation via Evolutionary Quantal Response Equilibrium

**arXiv ID:** 2601.05653 | [PDF](https://arxiv.org/pdf/2601.05653v1)

**作者:** Phu-Hoa Pham `[一作]` (Ho Chi Minh City University of Science), Trung-Kiet Huynh `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 EvoQRE 框架，用量化响应均衡 (QRE) 和演化博弈理论对多智能体交通仿真进行建模，显式考虑驾驶员的有界理性。

**💡 创新点**

创新点在于：① 在 Markov 游戏中正式推广 Logit-QRE 并给出存在性与收敛证明；② 提出双时尺度演化复制动力学实现 QRE，并证明 O(log k/k^{1/3}) 的收敛速率；③ 针对连续动作空间给出核密度、SVGD 与能量基等可实现方案；④ 通过可调理性参数生成可控的安全关键情景。

**🔧 技术方法**

主要技术包括：最大熵强化学习（Soft Actor-Critic）实现软化最佳响应；演化复制动力学 (Entropy-Regularized Replicator Dynamics) 与多时尺度随机逼近；核密度混合模型、SVGD、Langevin 动态等连续策略表示；经验回放与 Retrace(λ) 用于减少 Q‑函数方差。

**📊 数据集**

实验使用 Waymo Open Motion Dataset（WOMD）与 nuPlan 交通仿真基准，验证现实感、碰撞率与安全性。

**📈 对比分析**

与行为克隆、SMART、CCE-MASAC、VBD、GR2、Safe‑Sim、CHARMS 等基线对比，EvoQRE 在 NLL（2.83 bits/action）与碰撞率（1.2%）上优于大多数方法，CLS‑SR 得分最高（0.847）。其 QRE 间距与可利用性指标均低于对手，且可通过调节 λ 产生多样化安全关键场景。

**⚠️ 局限性**

局限性包括：① 对高维/大规模场景（N>50）扩展性不足；② 仅实现独立 QRE，无法直接捕获显式协同；③ 理论收敛率相对慢，受弱单调性假设限制；④ 需在真实车辆上进一步验证；⑤ 需处理神经网络近似误差导致的理论与实践差距。

---

## 159. Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs

**arXiv ID:** 2601.05641 | [PDF](https://arxiv.org/pdf/2601.05641v1)

**作者:** Alireza Dehghanpour Farashah `[一作]` (Mila – Quebec AI Institute), Golnoosh Farnadi `[通讯]` (Mila – Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多语言大型语言模型的忘却（Data Unlearning 与 Concept Unlearning），评估在不同语言间忘记效果的跨迁移与传播

**💡 创新点**

首次系统比较多语言忘却在十种语言（涵盖不同资源级别与语言族）的跨语言转移性，揭示语法相似度比资源水平更能预测转移；并提出统一实验框架与评测方法

**🔧 技术方法**

梯度差异法（GradDiff、GradDiff‑KL）与负偏好优化（NPO）三种基于梯度的忘却策略；使用 Aya‑Expanse‑8B 模型；通过跨语言概率比与困惑度评估

**📊 数据集**

TOFU（数据忘却）和 SeeGULL（概念忘却）两大基准，分别翻译成10种语言（英、法、阿、日、俄、波斯、韩、印、希、印尼），同时使用 mC4 测试通用性能

**📈 对比分析**

与 fine‑tuned 基线及 retain 模型对比；结果显示忘却大多语言特定，跨语言传播有限；在高资源语言间（如英-法）有部分转移，NPO 在保持性能方面更稳定；概念忘却在源语言英语的影响可扩散至多语言，但转移程度受文化差异影响

**⚠️ 局限性**

翻译质量未充分保证；缺乏多语言偏见与概念忘却基准；评估指标受语言差异影响（如 ROUGE、BLEU 适用性有限），模型在低资源语言上仍易出现过拟合与忘却不稳定

---

## 160. Dual-Phase LLM Reasoning: Self-Evolved Mathematical Frameworks

**arXiv ID:** 2601.05616 | [PDF](https://arxiv.org/pdf/2601.05616v1)

**作者:** ShaoZhen Liu `[一作]` (University of Chinese Academy of Sciences), Zhenan Sun `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 17735 | [OpenAlex ID](https://openalex.org/A5055505703)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个两阶段的监督微调框架，通过多轮自我纠错生成长链式推理（CoT）数据，并利用难度感知拒绝采样进一步提升模型对复杂数学问题的解决能力。

**💡 创新点**

创新点在于：① 用自我生成的长CoT数据激活模型内在推理能力，避免依赖大模型蒸馏；② 设计了多轮对话策略嵌入验证、回溯、子目标拆分和逆向推理四大推理模式；③ 引入难度感知拒绝采样机制，动态聚焦难题并标记数据来源，以提升训练效果和输出精简度。

**🔧 技术方法**

核心技术包括：多轮自回归推理策略、规则式高质量样本过滤、监督微调（SFT）、难度感知拒绝采样、数据标记（multi‑turn/rejection markers）以及基于vLLM的高效推理。

**📊 数据集**

主要数据集为公开的 DeepScaleR 以及其拆分的长CoT合成数据（≈10k 条）和拒绝采样数据（≈5k 条），并在 AIME24、AMC23、GSM8K、MATH500、SVAMP、TabMWP、Gaokao2023en 等七个数学基准上进行评估。

**📈 对比分析**

与自我奖励、自思两次、蒸馏等基线相比，D_multi+rej 方案在大多数基准上实现 10–20% 的准确率提升，尤其在 AIME24、GSM8K、MATH500 等任务上表现突出；同时通过标记训练显著缩短输出长度，取得了更优的准确率‑长度权衡。

**⚠️ 局限性**

局限性包括：对手工模板和两轮对话结构的依赖，难以捕捉更深层次的多步推理；缺乏自动化模板生成和符号推理集成；以及难度评估指标仍不够解释性，需进一步改进。

---

## 161. Research Integrity and Academic Authority in the Age of Artificial Intelligence: From Discovery to Curation?

**arXiv ID:** 2601.05574 | [PDF](https://arxiv.org/pdf/2601.05574v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 162. LatentVLA: Efficient Vision-Language Models for Autonomous Driving via Latent Action Prediction

**arXiv ID:** 2601.05611 | [PDF](https://arxiv.org/pdf/2601.05611v1)

**作者:** Chengen Xie `[一作]` (Shanghai Innovation Institute), Hongyang Li `[通讯]` (OpenDriveLab at The University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于自监督潜在动作学习的LatentVLA框架，用于在不依赖语言标注的情况下训练视觉‑语言‑动作模型，并通过知识蒸馏将其能力迁移到传统端到端网络中。

**💡 创新点**

创新点在于：① 用自监督的自我潜在动作预测替代语言标注，消除语言偏差；② 采用少量离散动作码表实现高精度轨迹预测；③ 通过知识蒸馏实现从大规模视觉语言模型到轻量化网络的高效迁移，兼顾实时性。

**🔧 技术方法**

主要技术包括VQ‑VAE离散动作编码、时空Transformer编码器‑解码器、跨模态多头注意力融合、基于Transfuser与iPad的BEV融合，以及基于规划Transformer的蒸馏策略。

**📊 数据集**

使用的数据集有nuPlan、nuScenes、OpenScene、navtrain、navtest（NAVSIM），以及在无监督阶段的OpenScene图像。

**📈 对比分析**

与Transfuser、iPad、DiffusionDrive等基线相比，LatentVLA在NAVSIM PDMS上从84.0提升到92.4（iPad版）或86.6（Transfuser版），零样本在nuScenes的L2误差仅为0.33m，显示出强大的跨域泛化；蒸馏版将推理延迟从≈790ms降至≈210ms，帧率提升至≈4.8 FPS。

**⚠️ 局限性**

局限性包括：① 仍需要预训练的大型视觉语言模型作为教师；② 原始VLA模型推理延迟高，需蒸馏后才能满足实时要求；③ 对多传感器配置、不同城市环境的适应性尚未充分验证。

---

## 163. Orchestrating Tokens and Sequences: Dynamic Hybrid Policy Optimization for RLVR

**arXiv ID:** 2601.05607 | [PDF](https://arxiv.org/pdf/2601.05607v1)

**作者:** Zijun Min `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**通讯引用:** 3902 | [OpenAlex ID](https://openalex.org/A5066326238)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了动态混合策略优化（Dynamic Hybrid Policy Optimization），通过在单一的clipped surrogate目标中混合 token‑level 与 sequence‑level 的重要性比率，解决 RLVR 中的高方差与粗粒度信用分配问题。

**💡 创新点**

创新点在于：① 混合权重设计——平均混合与熵导向混合两种策略；② 端到端的分支特定裁剪（branch‑specific clipping）先分别限制 token 与 sequence 重要性比率，再混合，从而降低单一裁剪区间导致的偏差与不稳定性。

**🔧 技术方法**

核心技术包括：PPO‑style clipped surrogate 换算、分支特定裁剪、熵信号用于动态权重调节、group‑based advantage 估计（GRPO/GSPO 机制）以及对 Qwen3 系列 LLM 的 RLVR 训练。

**📊 数据集**

使用 Qwen3 1.7B、4B 与 30B‑A3B‑Base 三个模型，在七个数学推理基准数据集上评估：AIME 2024/2025、AMC 2023、OlympiadBench、MATH‑500、Minerva Math 与 GSM8K。

**📈 对比分析**

与 GRPO、GSPO、GMPO、CISPO 等代表性 RLVR 基线进行对比；在所有模型与基准上均优于 GRPO 与 GSPO（平均提升约 4.9 % vs GRPO，4.3 % vs GSPO），例如 AIME 24 从 22.5 % 提升至 34.4 %，AIME 25 从 14.6 % 提升至 26.5 %。

**⚠️ 局限性**

局限性：实验仅覆盖 Qwen3 系列模型，未验证在其他架构或不同 tokenizer 方案上的泛化；对基线的覆盖有限，未覆盖所有 RLVR 方法；缺乏对更广泛模型家族与训练配置的系统性评估。

---

## 164. Revisiting Human-vs-LLM judgments using the TREC Podcast Track

**arXiv ID:** 2601.05603 | [PDF](https://arxiv.org/pdf/2601.05603v1)

**作者:** Watheq Mansour `[一作]` (University of Queensland), Andrew Yates `[通讯]` (HLTCOE)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对TREC Podcast 2020/2021语音转文本段落的18,284个查询–段落对进行重新评估，使用五种LLM生成判定，并与原始TREC评估及三位IR专家的重新评估进行对比。

**💡 创新点**

首次在Podcast语料上系统性比较LLM与人工评估差异，发现LLM与多名专家的一致性高且能揭示原始评估的不稳定性，验证LLM在多样化检索场景下可替代人工评估的潜力。

**🔧 技术方法**

使用OpenAI GPT‑4o、Mistral‑Small‑Instruct‑2409、Qwen2.5‑14B‑Instruct、Meta‑Llama‑3.1‑8B‑Instruct、Gemma‑2‑9b‑it五种LLM，采用DNA（Description, Narrative, Aspects）提示并调优，利用RBP、NDCG、Kendall’s τ、Rank‑Biased Alignment等指标评估系统排名。

**📊 数据集**

采用TREC 2020/2021 Podcast Track，包含约3.4 M两分钟语音转文本段落，共18,284个查询–段落对。

**📈 对比分析**

通过Kendall’s τ和RBA指标比较系统排序，2020集群对齐度高（τ≥0.85、RBA>0.90），2021集群对齐度低（τ可低至0.41），LLM评估更倾向于词典式检索模型，导致系统排名大幅变动。

**⚠️ 局限性**

局限性包括2021数据评估不稳定、转录错误影响判定、仅使用单一TREC人工评审者、LLM对高相关性倾向偏高、对复杂语义判断仍有限，且仅对五种LLM进行测试。

---

## 165. Transformer Is Inherently a Causal Learner

**arXiv ID:** 2601.05647 | [PDF](https://arxiv.org/pdf/2601.05647v1)

**作者:** Xinyue Wang `[一作]` (Halıcıoğlu Data Science Institute), Biwei Huang `[通讯]` (Halıcıoğlu Data Science Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

训练自回归解码器Transformer并利用梯度归因（LRP）从预测模型中恢复多变量时序因果图。

**💡 创新点**

证明Transformer输出对滞后输入的梯度能唯一确定因果结构，并提供可扩展的梯度能量提取与二值化方法。

**🔧 技术方法**

采用 decoder‑only Transformer、梯度归因（LRP）、梯度能量聚合、Top‑k 二值化等技术。

**📊 数据集**

使用多种合成时序数据集（线性、非线性、长时滞、高维、非平稳、带潜变量、不同噪声）进行实验。

**📈 对比分析**

与 PCMCI、DYNOTEARS、VAR‑LiNGAM、NTS‑NOTEARS、TCDF、Granger 等基线比较，F1 分数普遍高于基线，尤其在高维、长时滞、非线性、非平稳情形下显著提升，并随样本量增大持续改善。

**⚠️ 局限性**

缺乏对潜在混杂变量和即时因果关系的建模，Transformer 在潜变量多时会出现伪关联；对窗口长度与层数敏感；未利用预训练知识；需要进一步机制来处理即时和潜变量。

---

## 166. Compressing image encoders via latent distillation

**arXiv ID:** 2601.05639 | [PDF](https://arxiv.org/pdf/2601.05639v1)

**作者:** Caroline Mazini Rodrigues `[一作]` (University of Rennes), Thomas Maugey `[通讯]` (University of Rennes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对硬件受限环境，本文提出通过简化知识蒸馏方法，将大型图像压缩模型的编码器压缩为轻量化版本。

**💡 创新点**

创新点在于仅使用编码器的隐藏表示进行蒸馏，减少训练损失与数据需求，同时保持解码器不变，实现轻量化编码器的快速训练。

**🔧 技术方法**

采用自编码器结构的Factorized Prior和Hyperprior（MS‑ILLM）模型，利用特征层蒸馏（latent 对齐）及简化的均方误差蒸馏损失。

**📊 数据集**

使用Vimeo‑90k（子集10k短视频）作为训练集，评估在Kodak和CLIC2020测试集上的图像。

**📈 对比分析**

与同架构直接从零训练的模型（Frozen）对比，结果显示蒸馏后模型在PSNR/​FID上显著优于Frozen，甚至在极低数据比例（ρ=0.1）下也能逼近原始教师模型。

**⚠️ 局限性**

局限性包括只压缩编码器而未考虑解码器压缩，且在极高压缩率（r=8）时仍有轻微性能下降。

---

## 167. Analytical Approach to Wave Scattering in Waveguide Junction with Conducting Cylindrical Posts

**arXiv ID:** 2601.05638 | [PDF](https://arxiv.org/pdf/2601.05638v1)

**作者:** Malgorzata Warecka `[一作]` (Gdansk University of Technology), Piotr Kowalczyk `[通讯]` (Gdansk University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了矩形波导交叉中包含导电圆柱状物体的电磁波散射，提出了一种基于局部投影函数的模式匹配分析方法。

**💡 创新点**

创新点在于不引入假设圆柱区域，只在散射物体边界上进行投影，并使用一维有限元局部基函数，显著改善数值条件数并降低积分范围。

**🔧 技术方法**

采用模式匹配技术、局部一维有限元基函数投影、解析积分求解、最小二乘求解以及与全波有限元仿真（InventSim）的对比验证。

**📊 数据集**

使用了 WR‑62 波导下的三种结构（两圆柱共振器的不同间距、两极和四极波导滤波器）的几何参数作为验证数据集。

**📈 对比分析**

通过将解析方法得到的散射矩阵与 InventSim 的 FEM 结果对比，展示频率响应一致，收敛性验证表明所需模式数随圆柱间距变化；整体性能优于传统全波仿真，计算速度快、内存占用低。

**⚠️ 局限性**

限制在于仅适用于导电圆柱，非导电或非圆柱形散射体需要进一步改进；在极高频或复杂几何下仍需更多模式，且未考虑波导长度非零的情况。

---

## 168. GenCtrl -- A Formal Controllability Toolkit for Generative Models

**arXiv ID:** 2601.05637 | [PDF](https://arxiv.org/pdf/2601.05637v1)

**作者:** Emily Cheng `[一作]` (Universitat Pompeu Fabra), Xavier Suau `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于控制理论的框架和 Monte Carlo 算法，用于正式估计大语言模型与文本‑图像生成模型在对话过程中的可达性与可控性集合，并给出了概率保证；

**💡 创新点**

① 将可达性/可控性概念引入生成模型并针对离散瓶颈提出粗粒度可达性定义；② 推导 PAC 误差界与样本复杂度；③ 开源可验证可控性工具包，填补了先前方法无法处理黑盒生成模型的空白；

**🔧 技术方法**

控制理论（可达性、可控性）+ 概率 PAC 分析 + Monte Carlo 采样 + γ‑量化抽样 + Python/PyTorch 实现；

**📊 数据集**

使用公开的 LLM（Mistral‑7B、Qwen、Llama2、ChatGPT 等）和 T2IM（如 Stable Diffusion）进行对话/生成实验，任务包括文本正式度、句子长度、对象计数、对象位置、图像饱和度等；使用公开的 COCO、统一分布采样等；

**📈 对比分析**

通过比较不同模型、不同提示方式（0‑shot、5‑shot）以及模型规模，评估可控性覆盖率（coverage）、校准指标（Spearman、Pearson、MAE）。实验显示可控性高度依赖模型、任务和规模，未出现统一可控性，某些任务可达性较好但校准仍差；

**⚠️ 局限性**

结果依赖用户指定的输入分布、读出映射和初始状态分布；可达性/可控性估计的样本复杂度随可达性覆盖数 N 增大；对高维属性的可达性估计不易，工具仅适用于黑盒模型，无法解释内部机制；理论不直接迁移到新的分布设置。

---

## 169. GIFT: Games as Informal Training for Generalizable LLMs

**arXiv ID:** 2601.05633 | [PDF](https://arxiv.org/pdf/2601.05633v1)

**作者:** Nuoyan Lyu `[一作]` (State Key Laboratory of AI Safety Institute of Computing Technology CAS), Huawei Shen `[通讯]` (State Key Laboratory of AI Safety Institute of Computing Technology CAS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将游戏视为大型语言模型的非正式学习环境，并通过嵌套训练框架融合形式化学习（数学推理）与非正式学习（游戏）来提升模型的通用能力。

**💡 创新点**

创新点在于①用游戏替代传统的人工标注式学习，②设计嵌套训练把混合任务的“或”目标转化为“与”目标，显式要求模型同时掌握多种能力，从而避免任务干扰。

**🔧 技术方法**

采用强化学习（GRPO/StarPO）在 Qwen2.5‑1.5B/7B 上对 Matrix Game、TicTacToe、Who’s the Spy 等游戏与 Math 任务进行训练。

**📊 数据集**

使用 MathLv3‑5（SimpleRL‑Zoo‑Data）、MATH500、MMLU、MMLU‑Pro、CommonGen、SocialIQA 等公开基准；游戏环境为 Matrix、TicTacToe、Who’s the Spy。

**📈 对比分析**

通过单任务、混合多任务和嵌套多任务三种设置进行对比；嵌套训练在 1.5B 模型上平均通用能力从 38.34% 提升至 42.43%，在 7B 模型上从 42.00% 提升至 55.84%；相比之下混合训练出现性能退化。

**⚠️ 局限性**

局限性包括：实验仅涵盖少量形式化与非正式学习环境；游戏设计有限，难以覆盖所有真实交互情境；对手模型固定，缺乏自适应或自演化的训练；嵌套框架在更复杂场景中的有效性仍待验证。

---

## 170. Can large language models interpret unstructured chat data on dynamic group decision-making processes? Evidence on joint destination choice

**arXiv ID:** 2601.05582 | [PDF](https://arxiv.org/pdf/2601.05582v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 171. Cumulative Path-Level Semantic Reasoning for Inductive Knowledge Graph Completion

**arXiv ID:** 2601.05629 | [PDF](https://arxiv.org/pdf/2601.05629v1)

**作者:** Jiapu Wang `[一作]` (Nanjing University of Science and Technology), Kai Sun `[通讯]` (Beijing University Of Technology)

**通讯引用:** 483 | [OpenAlex ID](https://openalex.org/A5101864192)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了CPSR框架用于指令式知识图谱补全，通过查询相关掩码和全路径语义评分提升推理质量。

**💡 创新点**

创新点在于结合查询依赖噪声掩码与全路径全局语义评分，实现对结构噪声的自适应过滤与长距离语义依赖的捕获。

**🔧 技术方法**

采用逻辑规则推理、贝叶斯掩码、全路径评分、路径贪心Top‑k选择以及多分类log‑loss训练。

**📊 数据集**

评估使用四个WN18RR与四个FB15k‑237的指令版数据集。

**📈 对比分析**

与RuleN、NeuralLP、DRUM、GraIL、NBFNet、RED‑GNN、AdaProp、A*Net、MLSAA等基线对比，MRR在多数数据集上显著领先（最高提升约15%）。

**⚠️ 局限性**

限制在于Greedy Top‑k路径选择可能忽略低分但重要的长路径，以及全路径评分导致的额外计算开销。

---

## 172. Package-Aware Approach for Repository-Level Code Completion in Pharo

**arXiv ID:** 2601.05617 | [PDF](https://arxiv.org/pdf/2601.05617v1)

**作者:** Omar Abedelkader `[一作]`, Guillermo Polito `[通讯]` (Inria)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

在 Pharo 代码完成系统中加入了基于包结构的启发式排序，使同一包内的类/变量优先出现；

**💡 创新点**

创新点在于将包级别的上下文信息纳入完成排序，而非仅依赖全局平面；

**🔧 技术方法**

使用 AST 分析、惰性 Fetcher 以及新的包感知启发式算法，结合 MRR、NDCG 等评估指标；

**📊 数据集**

评估数据集为 Pharo 开源框架：Iceberg、Moose、Roassal、Seaside、Spec，覆盖不同规模与模块化程度；

**📈 对比分析**

通过将默认“全局平面”完成与新“包感知”完成在同一套测试环境下对比，结果显示在大多数框架中 MRR 上升 2%~10%，尤其是 Spec、Iceberg；然而测试包中表现不佳，说明仅按包名顺序仍有局限；

**⚠️ 局限性**

局限包括：仅采用短前缀（2~8 字符）进行评测，未覆盖长标识符；假设同包引用占主导，实际情况（如测试包）可能不符；未考虑真实交互式使用模式和历史日志；

---

## 173. Good Allocations from Bad Estimates

**arXiv ID:** 2601.05597 | [PDF](https://arxiv.org/pdf/2601.05597v1)

**作者:** Sílvia Casacuberta `[一作]`, Moritz Hardt `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种低采样量、近似最优的治疗分配方法，利用粗略估计的 CATE（条件平均治疗效应）即可在预算约束下选出接近最优的单元集合。

**💡 创新点**

创新点在于：①证明在“平滑”或 ρ‑regular 的治疗效应分布下，治疗分配的样本复杂度可从 O(M/ε²) 降到 O(M/ε)，远低于传统的完整 CATE 估计；②引入 ρ‑regular 性质和阈值切分策略；③提出预算灵活性和资源增量（overspending）策略以进一步减少样本需求。

**🔧 技术方法**

技术手段包括：Hoeffding 及 Le‑Cam 相关的误差界；阈值判定和 CDF 近似；非自适应低精度估计算法（LEA）；分布正则性理论、实例依赖的充分必要条件。

**📊 数据集**

实验使用五个真实 RCT 数据集：STAR（教育）、TUP（贫困经济发展）、NSW（劳工经济）、Acupuncture（医疗）以及 Post‑operative Pain（医疗）。

**📈 对比分析**

与传统的完整 CATE 估计和随机分配基线比较，实验表明所提方法在所有数据集上均能以远少于 O(M/ε²) 的样本实现 (1‑ε)‑optimal 分配，且在大多数预算下失败率低于 5%，只需加 1 个单位即可恢复最优值。

**⚠️ 局限性**

局限性：依赖 ρ‑regular 正则性，若治疗效应分布高度聚集在阈值附近会退化；算法为一次性非自适应估计，若能加入自适应采样可能进一步提升；预算需具备灵活调整或可通过资源增量补救。

---

## 174. RISE: Rule-Driven SQL Dialect Translation via Query Reduction

**arXiv ID:** 2601.05579 | [PDF](https://arxiv.org/pdf/2601.05579v1)

**作者:** Xudong Xie `[一作]` (Institute of Software at CAS), Jun Wei `[通讯]` (Institute of Software at CAS)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于大型语言模型（LLM）的 SQL 方言翻译方法——RISE，利用方言感知的查询简化技术先把复杂 SQL 查询简化为只保留方言相关部分，再由 LLM 翻译简化查询并自动生成翻译规则，最终将规则应用于原查询，实现准确、高效的跨 RDBMS 方言迁移。

**💡 创新点**

创新点在于：① 通过查询简化消除 LLM 在处理长复杂查询时的幻觉错误；② 采用 LLM 自动生成并抽象化翻译规则，使规则具备更强泛化能力；③ 将规则驱动与 LLM 结合，实现从零开始的自适应方言翻译，显著提升传统规则工具和单纯 LLM 方法的准确率。

**🔧 技术方法**

使用技术包括：适配的 ANTLR 解析器、基于 AST 的随机与 LLM 驱动的查询简化、LLM（GPT‑4o / DeepSeek‑V3）进行翻译与规则抽象、AST 匹配与替换算法实现规则驱动转换。

**📊 数据集**

数据集：TPC‑DS（99 条复杂查询，含 42 行长查询）和 SQLProcBench（44 条存储过程代码），均改为 PostgreSQL 兼容版本，用于评测跨 PostgreSQL‑>MySQL 与 PostgreSQL‑>Oracle 的方言迁移。

**📈 对比分析**

与六个基线（SQLGlot、SQLines、jOOQ、原型 LLM‑Translator、CrackSQL）比较。RISE 在 TPC‑DS 上 97.98% 的准确率，SQLProcBench 100%，平均提升 24.62% 及 238.41%；在效率上，RISE 在翻译规则成熟后可达 3.78 条/分钟，优于 LLM‑Translator（3.3 条/分钟）和 CrackSQL（0.5 条/分钟）。

**⚠️ 局限性**

局限性包括：① 仍无法完全消除因语义不一致导致的错误（如 PostgreSQL 与 MySQL 对 '||' 的不同语义）；② 需要源查询包含足够执行上下文（数据）来验证语义一致性；③ 对涉及模式信息的语义歧义（如除法运算）处理不完整，需进一步融合模式信息。

---

## 175. Orient Anything V2: Unifying Orientation and Rotation Understanding

**arXiv ID:** 2601.05573 | [PDF](https://arxiv.org/pdf/2601.05573v1)

**作者:** Zehan Wang `[一作]`, Zhou Zhao `[通讯]`

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

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类精度上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

模型在处理大规模数据集时可能会遇到内存限制，且对噪声数据的鲁棒性有待提高。

---

## 176. Towards Generalized Multi-Image Editing for Unified Multimodal Models

**arXiv ID:** 2601.05572 | [PDF](https://arxiv.org/pdf/2601.05572v1)

**作者:** Pengcheng Xu `[一作]` (Western University), Boyu Wang `[通讯]` (Western University)

**通讯引用:** 1617 | [OpenAlex ID](https://openalex.org/A5100383955)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可扩展的多图像编辑框架，使统一多模态模型能够保持跨图像视觉一致性与精确引用。

**💡 创新点**

创新点包括：①可学习的潜在分隔符，显式区分不同图像的视觉token；②基于正弦函数的可扩展图像索引编码，实现绝对图像身份识别并支持超出训练图像数量的推理。

**🔧 技术方法**

采用混合MLLM‑Diffusion架构（如Qwen‑Edit），在MM‑DiT中加入分隔符与正弦索引编码，并结合多模态RoPE进行位置编码。

**📊 数据集**

使用逆向构造法生成的MMIE‑Bench数据集，涵盖添加、替换、风格转换、人像、推理与混合六大任务，样本数量从2至5张图像不等。

**📈 对比分析**

与DreamOmni2、OmniGen2、Qwen‑Edit等基线在MMIE‑Bench上进行评估，使用两款大型MLLM（Qwen2.5‑VL 72B与Doubao‑1.6）进行语义一致性、视觉真实性与多图整合三维指标评估；实验显示本方法在所有任务均提升约0.3–0.7分，整体平均分提升至约3.7–4.1。

**⚠️ 局限性**

局限性在于：①对极大图像数量（>5）或极高分辨率的泛化仍有限；②在某些风格与人像任务中，单一模块效果略逊于组合方案；③引入分隔符和正弦编码略微增加计算开销。

---

## 177. Statistical Foundations of DIME: Risk Estimation for Practical Index Selection

**arXiv ID:** 2601.05649 | [PDF](https://arxiv.org/pdf/2601.05649v1)

**作者:** Giulio D'Erasmo `[一作]` (Sapienza University of Rome), Fabrizio Silvestri `[通讯]` (Sapienza University of Rome)

**通讯引用:** 5138 | [OpenAlex ID](https://openalex.org/A5044165871)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RDIME，一种基于统计风险最小化的查询自适应维度选择方法，直接在推理时确定每个查询的最优子空间，避免了传统方法的网格搜索。

**💡 创新点**

创新点在于：①把 DIME 视为潜在信号的平方估计器；②用硬阈值推断证明可获得无偏估计；③构造统一的 Kernel DIME 框架，提供可解释的权重化方案；④在无验证集的情况下即可实现查询级维度自适应，显著降低维度且保持检索效果。

**🔧 技术方法**

技术手段包括：硬阈值估计器、kernel 加权最小二乘、模调估计框架、PRF/LLM 文档反馈、统计风险分析、配对 t 检验与 Holm–Bonferroni 校正。

**📊 数据集**

实验使用四个检索基准：TREC DL 2019、TREC DL 2020、Deep Learning Hard 集以及 TREC Robust 2004；评估模型包括 768 维的 ANCE、Contriever 与 TAS‑B；对比基线为全维度检索。

**📈 对比分析**

与传统的 Top‑k 维度阈值（k∈{0.4,0.6,0.8}）进行 nDCG@10 和 AP 比较，RDIME 在大多数设置下与最佳 Top‑k 基线无显著差异，且平均保留维度约 50%，提升幅度从 0.15% 到 2.46%，验证了其在不调参情况下的鲁棒性。

**⚠️ 局限性**

局限性：①实验仅涵盖 PRF、LLM 等标准 DIME 变体，未系统评估其他核函数（如 RBF、Sigmoid）的表现；②理论上仅在均匀权重下保证无偏，随机权重的统计性质仍待深入研究。

---

## 178. SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving

**arXiv ID:** 2601.05640 | [PDF](https://arxiv.org/pdf/2601.05640v1)

**作者:** Jingyu Li `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**通讯引用:** 79712 | [OpenAlex ID](https://openalex.org/A5100425671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将Vision‑Language Model（VLM）的表示学习明确结构化为场景-主体-目标层级的框架，以提升端到端自动驾驶的轨迹规划性能。

**💡 创新点**

创新点在于：①构建三层驱动相关世界知识（几何场景、关键主体、短期目标）并显式预测其未来演化；②设计块级遮蔽注意力机制防止不同层级信息泄漏；③将生成的层级世界知识直接作为条件输入至扩散Transformer（DiT）轨迹生成器，完成从语义推理到连续动作的闭环。

**🔧 技术方法**

核心技术包括：InternVL3‑2B VLM骨干、<world>特殊查询、块级遮蔽注意力、VAE/DETR式几何与主体监督、L1/CE损失、Diffusion Transformer轨迹生成器、两阶段监督微调策略。

**📊 数据集**

使用的主要数据集是NAVSIM（包含 navtrain 与 navtest 两子集）以及超过 3.1 M 的视觉问答对进行域适配与微调。

**📈 对比分析**

与现有摄像头或 LiDAR 结合的端到端方法对比，本文在 NAVSIM 上取得 PDMS 87.4（SFT）/91.1（RFT）和 EPDMS 86.2 的最高分，特别在无责任碰撞（NC）和时间到碰撞（TTC）指标上显著优于 RecogDrive 等前沿方法。

**⚠️ 局限性**

局限性：①仅利用单摄像头输入，未充分利用 LiDAR/雷达等多模态信息；②对未来场景预测依赖已标注的占据和主体信息，泛化到未见场景或不同城市风格时可能受限；③块级遮蔽注意力虽降低信息泄漏，但可能导致某些场景下过度保守的轨迹。

---

## 179. Text Detoxification in isiXhosa and Yorùbá: A Cross-Lingual Machine Learning Approach for Low-Resource African Languages

**arXiv ID:** 2601.05624 | [PDF](https://arxiv.org/pdf/2601.05624v1)

**作者:** Abayomi O. Agbeyangi `[一作]` (Walter Sisulu University), Abayomi O. Agbeyangi `[通讯]` (Walter Sisulu University)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5040655156)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 isiXhosa 与 Yorùbá 两种低资源非洲语言的文本去毒化，构建并使用轻量可解释的 TF–IDF+Logistic 回归检测模型和基于词典的重写模块，实现毒性检测与意义保持的改写。

**💡 创新点**

创新点：①结合可解释的传统机器学习与规则式重写，避免大模型不适用于低资源语言；②提供首个 isiXhosa 与 Yorùbá 的并行毒性‑中和语料与开放工具；③实现无 GPU 可在 CPU 上完成。

**🔧 技术方法**

使用的技术：TF–IDF 特征抽取、逻辑回归分类、Unicode 文本规范化、词典查表与词级替换、Stratified K‑fold 交叉验证、ROC‑AUC 评估。

**📊 数据集**

使用的数据集：约 178 句毒性‑中和对的手工标注并行语料（isiXhosa 与 Yorùbá），并借鉴 AfriHate、PAN TextDetox 等公开资源，数据已上传至 Mendeley DOI。

**📈 对比分析**

与现有大模型基准（mT5、BART 等）比较，轻量模型在 isiXhosa 取得 61–72% 准确率、0.65–0.80 AUC，Yorùbá 取得 72–86% 准确率、0.81–0.98 AUC，并能在 CPU 上实现实时推理，显示出在低资源场景下的竞争力。

**⚠️ 局限性**

局限性：①仅检测二元毒性，未细分种类；②对隐式或上下文敏感的毒性识别受限；③并行语料规模有限，导致模型对罕见表达泛化不足；④未评估跨域与真实社交媒体噪声下的鲁棒性。

---

## 180. A Large Scale Empirical Analysis on the Adherence Gap between Standards and Tools in SBOM

**arXiv ID:** 2601.05622 | [PDF](https://arxiv.org/pdf/2601.05622v1)

**作者:** Chengjie Wang `[一作]` (Institute of Software, Chinese Academy of Sciences), Chen Zhao `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文开展了基于真实 GitHub 项目的大规模、两阶段纵向实验，系统评估了 6 款主流 SBOM 工具与 SPDX/CycloneDX 标准在合规性、互操作一致性和准确性方面的符合度缺口。

**💡 创新点**

创新点在于提出了可扩展的自动化评估框架 SAP，结合三因素最佳匹配算法和 ISO/IEC 25012 数据质量维度，对 SBOM 生成、抽取与评估进行完整的流水线实现，并首次量化展示了工具在标准合规、跨工具一致性以及字段准确性方面的持久低效。

**🔧 技术方法**

核心技术包括：多语言 Docker 化工具执行、统一 SBOM 抽取与规范化、三因素最佳匹配（包名、版本、purl）进行包级匹配、以及基于置信度阈值的多层一致性与准确性指标计算。

**📊 数据集**

实验数据集为 3,287 个拥有至少 100 星的 C/C++、Java 与 Python 开源仓库，生成 27,795 份 SBOM（baseline）及 27,649 份（2025 追踪），并构建 100 个 Python 项目的人工校验真值集用于准确性评估。

**📈 对比分析**

对比方法采用合规率、跨工具一致性（平均 7.8%–12.8%）和准确率（部分字段低于 20%）等指标；实验显示工具在合规与一致性得分低于 20%，且随版本演进仍未显著提升，验证了系统性缺口。

**⚠️ 局限性**

主要局限包括：准确性评估仅基于 Python 生态的人工真值集；实验聚焦 GitHub 公开仓库，未覆盖私有或多语言混合项目；标准歧义导致工具实现差异难以量化。

---

## 181. PiXTime: A Model for Federated Time Series Forecasting with Heterogeneous Data Structures Across Nodes

**arXiv ID:** 2601.05613 | [PDF](https://arxiv.org/pdf/2601.05613v1)

**作者:** Yiming Zhou `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27290 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在联邦学习环境下为时间序列预测提出了PiXTime模型，解决不同采样率和变量集异构性问题。

**💡 创新点**

创新点在于为每节点定制Patch Embedding对多粒度时间片进行统一维度映射，并引入全局VE表实现变量语义对齐，从而兼顾局部个性化与全局共享。

**🔧 技术方法**

采用Transformer Encoder-Decoder架构，配合Patch Embedding、Variable Embedding、交叉注意力及局部投影头，结合FedOPT联邦训练。

**📊 数据集**

使用八个公开数据集：Electricity、Traffic、Weather、Exchange、ETTh1、ETTh2、ETTm1、ETTm2，涵盖不同采样率与变量规模。

**📈 对比分析**

与DLinear、iTransformer、PatchTST、TimeXer等SOTA基线在非联邦和联邦两种设置下对比，PiXTime在大多数指标上均获得MSE/MAE显著下降，尤其在联邦下表现更优。

**⚠️ 局限性**

局限性包括对采样率差异的处理依赖于固定物理时间对齐，且在极度不均衡的节点数据或极低采样率时仍可能导致性能波动。

---

## 182. Autoregressive Ranking: Bridging the Gap Between Dual and Cross Encoders

**arXiv ID:** 2601.05588 | [PDF](https://arxiv.org/pdf/2601.05588v1)

**作者:** Benjamin Rozonoyer `[一作]` (University of Massachusetts Amherst), Felix Yu `[通讯]` (Google DeepMind)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并分析了自回归排序（Autoregressive Ranking, ARR）模型的理论优势，并针对排序任务设计了一种基于排名的 next‑token 预测损失。

**💡 创新点**

创新点包括：
- 证明 ARR 在维度不随文档数增长的前提下即可实现任意排名；
- 设计了排名感知的加权损失（λ(r)=1/r^α 或 stepwise）以及利用前缀树对目标分布进行平滑的训练策略；
- 将上述损失应用于 Mistral‑7B 之类的 LLM，实现单模型直接排序，取代传统 DE/CE 两阶段流程。

**🔧 技术方法**

使用技术：
- 自回归语言模型（Mistral‑7B‑v0.3‑it）
- 约束解码与前缀树（Trie）构造
- 位置/排名加权交叉熵损失
- 对比 DE（Dual Encoder）与 CE（Cross Encoder）的软最大/对数似然训练
- 采用 beam / greedy 解码评估排序

**📊 数据集**

数据集：
- WordNet 词汇层级（超词层次）用于生成查询和文档集合；
- ESCI 购物查询数据集，包含商品标题及手工设计的稀疏 docID。

**📈 对比分析**

比较方法与性能：
- 在 WordNet 上与 DE（不同维度）和 CE（4 层 MLP）进行对比；ARR 在 nDCG、R@k 上与 CE 相当，远超 DE；
- 在 ESCI 上采用排名感知损失后 R@k（k>1）显著提升，R@1 略下降；
- 主要指标：CVR（越低越好）、nDCG（最高≈99.8）、R@1~R@5 介于 90%–98% 之间。

**⚠️ 局限性**

局限性：
- 前缀树训练与推理存在不匹配（后续 token 的平滑缺失）；
- 理论所需的 token 嵌入矩阵秩在大词表下难以满足，实际模型在多文档场景下仍受限；
- 仅使用贪心/beam 解码，未探索更高效的自回归生成策略；
- 需要专门的 docID 设计以兼容前缀树，限制了通用性。

---

## 183. HogVul: Black-box Adversarial Code Generation Framework Against LM-based Vulnerability Detectors

**arXiv ID:** 2601.05587 | [PDF](https://arxiv.org/pdf/2601.05587v1)

**作者:** Jingxiao Yang `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 1760 | [OpenAlex ID](https://openalex.org/A5047459900)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了黑盒对抗代码生成框架HogVul，融合词法与语法扰动；

**💡 创新点**

创新点在于双通道协同优化，利用粒子群优化实现词法与语法扰动的动态切换与信息共享；

**🔧 技术方法**

技术手段包括粒子群优化（PSO）、语义相似度驱动的词法替换、AST结构变换、停滞检测与通道切换；

**📊 数据集**

使用了Devign、DiverseVul、BigVul、D2A四个主流漏洞检测数据集；

**📈 对比分析**

与ALERT、DIP基线对比，平均攻击成功率提升约26%，查询效率和攻击质量（CodeBLEU、CAD）均优于基线；

**⚠️ 局限性**

局限性在于仅针对C/C++函数，攻击仍需大量模型查询，对其他编程语言或特殊编译器的适用性待验证，且对抗样本生成复杂度仍较高。

---

## 184. Reinforcement Learning of Large Language Models for Interpretable Credit Card Fraud Detection

**arXiv ID:** 2601.05578 | [PDF](https://arxiv.org/pdf/2601.05578v1)

**作者:** Cooper Lin `[一作]` (Hong Kong University of Science and Technology), Jun Song `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 7021 | [OpenAlex ID](https://openalex.org/A5081444975)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用强化学习（GSPO）在原始电子商务交易文本上对轻量级 LLM 进行后期微调，实现了欺诈检测与可解释推理的结合。

**💡 创新点**

创新点在于：①使用基于规则的奖励系统和组级优势估计实现无监督的探索学习；②通过开放式提示鼓励模型发现非显式欺诈信号；③展示了模型规模与表现的反向关系（“少即多”效应）以及信息压缩的自然自适应。

**🔧 技术方法**

技术手段包括：大型语言模型（Qwen3 系列 4B/8B/14B），Group Sequence Policy Optimization (GSPO)，规则式奖励（准确率+格式），自定义提示模板，DeepEval 语义真实性评估。

**📊 数据集**

数据集为中国全球支付解决方案公司提供的真实电子商务交易记录（2023‑2024 年共约 9.4 万笔，含订单、IP、支付、地址、历史等多维字段），按时间分为训练（51.8% 欺诈）与测试（9.6% 欺诈）集。

**📈 对比分析**

与 GPT‑4.1、Claude‑4.5‑Sonnet 等大模型对比，后期微调的 Qwen3‑4B/8B/14B 在 F1 上提升约 100%+，召回/特异性显著改善；同时，Token 数量大幅下降（约 30%–60%），实现低延迟；压缩版实验表明过度约束导致性能下降 90%+。

**⚠️ 局限性**

局限性包括：①较小训练样本量导致泛化不确定；②RL 过程可能产生 hallucination，尤其是大模型；③缺乏对业务实时成本和可扩展性的深入评估；④未将 LLM 与传统表格模型融合，仍未充分利用结构化特征。

---

## 185. Productive Discussion Moves in Groups Addressing Controversial Issues

**arXiv ID:** 2601.05651 | [PDF](https://arxiv.org/pdf/2601.05651v1)

**作者:** Kyuwon Kim `[一作]` (Ewha Womans University), Hyo-Jeong So `[通讯]` (Ewha Womans University)

**通讯引用:** 5291 | [OpenAlex ID](https://openalex.org/A5026147191)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨学生在AI伦理争议情境中展开的讨论动作，识别其对讨论质量的影响，并分析高低质量讨论的顺序模式。

**💡 创新点**

提出一种混合分析方法，结合专家制定的SEDA框架与基于BERTopic的无监督聚类，生成包含情感表达与模糊性承认的新讨论动作类别，显示情感与经验论证对高质量讨论的显著正向作用。

**🔧 技术方法**

采用SEDA编码、Korean Sentence‑BERT与BERTopic、HDBSCAN聚类、线性混合效应模型、Ordered Network Analysis。

**📊 数据集**

收集51名韩国本科生在5个AI伦理情境下的83组面谈记录（约10–20分钟），共2199句子。

**📈 对比分析**

通过混合编码评估讨论动作与Integrative Complexity的线性混合模型；结果表明情感表达和对模糊性的承认显著提升讨论质量；Ordered Network显示高质量组情感表达往往转向证据论证，低质量组则停滞在情感或程序管理。

**⚠️ 局限性**

样本仅限韩国本科生，可能不具跨文化普适性；依赖人工标注，扩展性受限；聚类参数人为设定，可能影响结果一致性。

---

## 186. Multiset Deletion-Correcting Codes: Bounds and Constructions

**arXiv ID:** 2601.05636 | [PDF](https://arxiv.org/pdf/2601.05636v1)

**作者:** Avraham Kreindel `[一作]` (Reichman University), Aryeh Lev Zabokritskiy `[通讯]` (MIGAL -- Galilee Research Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本研究提出了多集删除纠错码的理论与构造，探讨了极端删除场景的最优码量；

**💡 创新点**

创新点在于给出二元多集码的全局最优构造、单删除多字母码的取舍与循环Sidon型线性码，并通过递归打孔界实现高删数下的上界；

**🔧 技术方法**

主要技术包括球包与投影界、图论计数、Reiman不等式、Sidon集与模约束的构造、线性格子映射；

**📊 数据集**

由于本工作为理论分析，未使用实验数据集；

**📈 对比分析**

在已知最优区间内构造与上界匹配，证明了构造的最优性；对于多字母单删场景，只给出渐近最佳且非最优的构造，理论上优于现有方案；

**⚠️ 局限性**

局限性在于非二进制场景的最优码量仍不完全确定，且构造的冗余在某些参数下可能不最小，未给出多删除非二元的具体实现与性能评估。

---

## 187. Quantifying and Inducing Shape Bias in CNNs via Max-Pool Dilation

**arXiv ID:** 2601.05599 | [PDF](https://arxiv.org/pdf/2601.05599v1)

**作者:** Takito Sawada `[一作]` (Doshisha University), Masahiro Okuda `[通讯]` (Doshisha University)

**通讯引用:** 4585 | [OpenAlex ID](https://openalex.org/A5025207272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于L0-SSIM的形状‑纹理平衡指标，并在小样本场景下通过仅调整Max‑Pool层的扩张率、冻结卷积权重的方式实现CNN的形状偏置快速适配。

**💡 创新点**

创新点：
1) 用SSIM衡量图像亮度通道与L0平滑后的差异来定量描述数据集的形状‑纹理比例；
2) 仅改Max‑Pool扩张率而不训练卷积权重，提供一种计算量低、易实现的形状偏置诱导方法；
3) 将指标与适配策略结合，构建两阶段的指标‑引导适配框架。

**🔧 技术方法**

技术手段包括：
- 计算Y通道与L0‑平滑图像的SSIM；
- 采用ResNeXt‑50预训练模型；
- 通过调整Max‑Pool的dilation参数实现形状偏置；
- 使用Adam优化器、5‑折交叉验证、Cross‑Entropy 损失。

**📊 数据集**

使用六个小规模数据集：
- TU‑Berlin Sketches（100类, 2000张）
- MPEG‑7 Silhouette（70类, 1400张）
- AnimeFace（50类, 1000张）
- BTSD（953张）
- DTD（940张）
- Stanford Dogs（2400张）

**📈 对比分析**

实验对比：
- 对高L0‑SSIM数据集（Sketches, MPEG‑7）时，S_conv‑Model 或 S_maxpool‑Model 的准确率比标准Texture‑Biased Model 提升 2‑7%；
- 对低L0‑SSIM数据集（DTD, Dogs）时，强形状偏置模型的准确率下降 12‑26%；
- 结果表明L0‑SSIM能有效预测何时使用形状偏置模型，验证了方法的有效性。

**⚠️ 局限性**

局限性：
- Max‑Pool扩张方法在极抽象图像（Sketches）上提升有限；
- L0‑SSIM仅基于结构相似度，可能无法全面反映形状与纹理的交互；
- 实验仅覆盖六个数据集，未验证在更大规模或不同领域数据上的泛化。

---

## 188. PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning

**arXiv ID:** 2601.05593 | [PDF](https://arxiv.org/pdf/2601.05593v1)

**作者:** Jingcheng Hu `[一作]` (StepFun), Heung-Yeung Shum `[通讯]` (Tsinghua University)

**通讯引用:** 32848 | [OpenAlex ID](https://openalex.org/A5061000201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了并行协调推理框架PaCoRe，通过多轮消息传递与并行推理路径，显著提升语言模型在固定上下文窗口下的测试时计算量并获得更优推理结果。

**💡 创新点**

将测试时计算从顺序深度转向并行宽度，利用消息压缩与合成能力实现百万级标记有效计算，同时通过大规模基于结果的强化学习训练模型的推理合成能力。

**🔧 技术方法**

基于大型语言模型（Qwen3‑8B）实现并行链式推理、消息压缩（仅保留结论）、多轮协调、以及PPO+GAE的强化学习来优化合成策略。

**📊 数据集**

采用竞赛级数学与编程任务（如 HMMT 2025、AIME 2025、LiveCodeBench、SWE‑Verified 等）以及自建高质量需要合成的训练集进行RL训练。

**📈 对比分析**

与 GPT‑5、Kimi‑K2‑Thinking、GLM‑4.6 等前沿推理模型在多项基准上对比，PaCoRe‑8B 在 HMMT 2025 以 94.5%（高于 GPT‑5 93.2%）和 LiveCodeBench 78.2% 等指标表现出色。

**⚠️ 局限性**

仍受模型规模、并行路径数与轮数的硬件限制，消息压缩方式仅保留结论可能丢失细节，且在极端高计算需求下可能出现收敛慢或资源浪费。

---

## 189. A Causal Information-Flow Framework for Unbiased Learning-to-Rank

**arXiv ID:** 2601.05590 | [PDF](https://arxiv.org/pdf/2601.05590v1)

**作者:** Haoming Gong `[一作]` (Rutgers University), Yongfeng Zhang `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于结构因果模型和信息流的无偏学习排序框架，利用条件互信息正则化降低多渠道偏差导致的泄漏，进而提升排名质量。

**💡 创新点**

创新点：①将多渠道偏差抽象为信息通道，采用结构因果模型明确偏差路径；②通过条件互信息量化信息泄漏，并给出风险界限；③将此量化结果作为可解释的预算正则化加入训练；④结合双重稳健估计提升稳健性。

**🔧 技术方法**

使用技术：结构因果模型（SCM）、条件互信息正则化、闭式二元通道MI估计、IPS/DR 等逆倾向评分与双重稳健方法。

**📊 数据集**

数据集：Yahoo! LETOR、MSLR-WEB30K 的半合成模拟数据，用于评估位置偏差和位置+信任双重偏差场景。

**📈 对比分析**

对比方法：IPS、DLA、Pair‑debias 等经典 ULTR 基线；在单偏差场景下与最佳基线相当；在多渠道偏差场景下提升 2–3 NDCG 分，尤其在信任偏差主导时表现最优。

**⚠️ 局限性**

局限：①仍假设偏差渠道可观测，未考虑未观测混杂；②需要手动设定 MI 预算和正则化系数；③在极端信任偏差下可能过度正则化导致性能退化。

---

## 190. Poisson Hyperplane Processes with Rectified Linear Units

**arXiv ID:** 2601.05586 | [PDF](https://arxiv.org/pdf/2601.05586v1)

**作者:** Shufei Ge `[一作]` (ShanghaiTech University), Lloyd Elliott `[通讯]` (Simon Fraser University)

**通讯引用:** 11864 | [OpenAlex ID](https://openalex.org/A5088544768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将两层 ReLU 神经网络视为具有高斯先验的 Poisson 超平面过程（PHP），并提出分解原理和退火序贯蒙特卡洛（SMC）算法进行贝叶斯推断。

**💡 创新点**

创新点在于：①建立 PHP 与两层 ReLU 网络的严格对应关系；②给出三条分解原理，实现大规模问题的可扩展性；③将退火 SMC 与 PHP 相结合，显著提升推断效率与模型预测性能。

**🔧 技术方法**

主要技术包括 Poisson 超平面过程建模、正态/逆伽马先验、退火序贯蒙特卡洛（annealed SMC）、随机游走 Metropolis–Hastings、分解原理（超平面数或数据域分区）。

**📊 数据集**

实验使用人工模拟数据（二维、p 维、不同超平面数）以及公开真实数据集：红酒质量数据集和海螺年龄数据集。

**📈 对比分析**

与决策树、随机森林、线性回归、SVM（线性/径向）以及两层 ReLU 神经网络（Keras/TensorFlow）进行对比；结果显示，本文方法在 RMSE 上优于大多数传统方法，且在覆盖率、置信区间长度和计算时间方面表现更优或相近。

**⚠️ 局限性**

局限性：超平面数需预先设定，缺乏自动确定机制；模型仅针对两层 ReLU，难以直接推广至更深网络；退火 SMC 仍需调参，计算开销相对较大。

---

## 191. Crisis-Bench: Benchmarking Strategic Ambiguity and Reputation Management in Large Language Models

**arXiv ID:** 2601.05570 | [PDF](https://arxiv.org/pdf/2601.05570v1)

**作者:** Cooper Lin `[一作]` (Hong Kong University of Science and Technology), Jun Song `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 7021 | [OpenAlex ID](https://openalex.org/A5081444975)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 Crisis-Bench——一个多智能体、部分可观测马尔可夫决策过程模拟框架，用于评估大型语言模型在企业危机中的声誉管理与战略信息控制能力。

**💡 创新点**

创新点包括：①构建双重知识体系（私有与公开知识）以逼真模拟信息不对称；②设计 Adjudicator‑Market Loop，将公众情绪与模拟股价联动，形成经济导向的评估指标；③通过可复现的事件池和 LLM‑驱动的路由器，兼顾可比性与情境连贯性。

**🔧 技术方法**

技术主要涉及：多智能体 POMDP 设计、链式推理 (CoT) 生成策略、LLM‑驱动的路由与裁判代理、基于公开情绪评分的股票价格模拟，以及大规模语言模型推理与强化学习评估。

**📊 数据集**

数据集由 80 条跨行业（医药、保险、IT、餐饮、家电、汽车、金融、娱乐）危机情景组成，每条情景包含多条预定义事件与事实，构成了模拟的“真实文件”与知识库。

**📈 对比分析**

通过对 12 种 LLM（含 GPT‑5、Gemini、Qwen、DeepSeek、Mistral、Llama 等）在 80 场危机情景中的 560 条 PR 语句进行评估，使用 Adjudicator 打分与股价回报两大指标进行对比；结果显示 GPT‑5.1 在保持合理信任度的同时，凭借更低成本与更高危机缓解效率，获得最高股价；而“极端透明”模型表现最差，验证了“对齐税”假说。

**⚠️ 局限性**

局限性包括：①模拟环境对真实危机的复杂性与不确定性（如监管介入、多渠道传播、市场情绪波动）简化不足；②股价模拟仅为代理指标，未覆盖宏观经济因素；③事件池与评估标准的人工设计可能限制多样性与可扩展性；④当前仅测试了 80 场情景，缺乏更广泛的现实数据验证。

---

## 192. A Framework for Personalized Persuasiveness Prediction via Context-Aware User Profiling

**arXiv ID:** 2601.05654 | [PDF](https://arxiv.org/pdf/2601.05654v1)

**作者:** Sejun Park `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 1664 | [OpenAlex ID](https://openalex.org/A5021733732)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个可训练的、上下文感知的用户画像框架，用于个性化说服力（视图变化）预测；

**💡 创新点**

创新点在于利用无监督的视图变化性能通过 DPO 训练查询生成器和画像器，生成针对当前说服上下文的用户画像，并通过记录级说服效用评分来指导检索；

**🔧 技术方法**

主要技术包括基于 LLM 的查询生成器与画像器、DPO 训练、BGE、BM25、HyDE 等检索方法，以及多种预测器模型；

**📊 数据集**

使用 Reddit ChangeMyView (CMV) 数据集进行实验；

**📈 对比分析**

通过与 PAG、HSumm、Recursumm 等个性化框架以及检索/画像基线对比，实验证明在不同预测器上相较于无个性化或基线方法可提升至 +13.77%p 的 F1 分数，并在检索端提升 NDCG@5 等指标；

**⚠️ 局限性**

局限性包括仅在长文本在线讨论场景下验证，未针对短文本或实时推荐等其他互动模式进行测试，且仅预测视图变化而不涉及实际说服干预。

---

## 193. Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer

**arXiv ID:** 2601.05770 | [PDF](https://arxiv.org/pdf/2601.05770v1)

**作者:** Yifan Zhang `[一作]` (Peking University), Zhi Jin `[通讯]` (Peking University)

**通讯引用:** 9738 | [OpenAlex ID](https://openalex.org/A5049100391)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Discrete Transformer框架，利用功能解耦的数值注意力和数值MLP以及温度退火的离散化采样，能够直接从Transformer模型中提取可执行、可解释的程序，支持连续变量域；

**💡 创新点**

创新点在于将Transformer拆分为信息路由（注意力）和算术运算（MLP）两大模块，并通过严格的功能解耦与可微离散化实现模型的可解释性；构建了分模块的提取管道，利用假设检验和符号回归从训练好的权重中恢复完整程序，并提供可控的算法发现方式；

**🔧 技术方法**

主要技术包括温度退火的可微采样（Gumbel‑Softmax）、Piecewise Linear Encoding、硬注意力指针、子MLP并行结构、线性输出头、假设检验用于识别注意力模式、符号回归（PySR）用于恢复算术表达式；

**📊 数据集**

使用MIPS基准数据集，包括线性算术、非线性组合以及物理动力学等多种算法推理任务；

**📈 对比分析**

与RNN基准（MIPS）进行对比，性能相当或略优，在所有任务上MSE接近0；此外能够完整提取出与目标任务等价的可读Python程序，且相比于MIPS可处理连续浮点输入；

**⚠️ 局限性**

局限性主要包括：受限于低维稀疏交互，难以处理高维密集任务；硬注意力指针限制了模型的表达能力；功能解耦导致的稀疏表示削弱了分布式特征的压缩能力，限制了在大规模NLP任务中的适用性。

---

## 194. Explicit Reward Mechanisms for Local Flexibility in Renewable Energy Communities

**arXiv ID:** 2601.05756 | [PDF](https://arxiv.org/pdf/2601.05756v1)

**作者:** Thomas Stegen `[一作]` (University of Mons), Bertrand Cornélusse `[通讯]` (University of Liège)

**通讯引用:** 874 | [OpenAlex ID](https://openalex.org/A5027468572)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于社区运营商协调的去中心化迭代方法，用于最大化可再生能源社区内的本地能量交换价值，并通过规则化的分配机制（Keys of Repartition）实现灵活负荷的激活；

**💡 创新点**

创新点在于将传统集中式优化拆分为用户层和运营商层的两步迭代，既保留了整体最优近似，又兼顾了用户隐私与参与度；同时将水暖设备（热水器、热泵）与电动车、储能等设备统一建模为等效储能系统，实现更细粒度的灵活性利用；

**🔧 技术方法**

主要技术包括：等效储能模型、离散化的功率和状态约束、基于规则的分配机制（等分、比例、级联）以及双层优化的迭代求解；

**📊 数据集**

使用了基于Resflex的随机住宅负荷生成器得到20户家庭的非灵活与灵活负荷数据，并假设15户配备PV（2–20 kWp，总计147 kWp），无具体公开数据集引用；

**📈 对比分析**

通过与集中式最优方案以及若干基准模型（SoloFix、SoloFlex、ECFix、ECFlex'）比较，去中心化方案在全年模拟中仅产生约3.2%到6.35%的账单增幅，且在自我消耗提升、网损降低和电网负荷削减等方面表现优于无灵活性基准；

**⚠️ 局限性**

局限在于：迭代收敛速度与分配规则相关，级联与比例机制虽然减少迭代次数但对公平性影响有限；模型假设完美预测PV和负荷，未考虑不确定性；缺少对更大规模社区（>100户）和更复杂电网拓扑的验证。

---

## 195. AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor

**arXiv ID:** 2601.05752 | [PDF](https://arxiv.org/pdf/2601.05752v1)

**作者:** Shu Yang `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 3934 | [OpenAlex ID](https://openalex.org/A5100401482)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AutoMonitor-Bench 基准，系统评估 LLM 监测器的可靠性，并构建大规模训练集进行监督微调实验。

**💡 创新点**

首个针对 LLM 监测器的可靠性基准，使用 Miss Rate 与 False Alarm Rate 双指标，设计多种隐蔽误行为类别与 paired benign 例子，并探讨微调对监测性能的影响。

**🔧 技术方法**

采用 LLM 作为监测器（如 GPT、Claude、Qwen 等），结合 LoRA 微调、针对不同任务的提示设计和二分类评估。

**📊 数据集**

使用 AutoMonitor-Bench 共 3,010 例子（涵盖 Safety & Permission Violations、Sycophancy & Bias、Specification Gaming），以及 153,581 条训练样本。

**📈 对比分析**

在 22 款 LLM 上通过 Miss Rate 与 False Alarm Rate 进行评估，发现专有模型 MR 低但 FAR 可能升高，开源模型 MR 高、FAR 低，呈现安全-效用权衡；微调在已训练类别提升显著，但跨类别泛化有限。

**⚠️ 局限性**

仅做二分类判断，未定位或评估置信度；未研究对抗性或更复杂训练策略；缺乏精细误行为识别与校准方法。

---

## 196. PII-VisBench: Evaluating Personally Identifiable Information Safety in Vision Language Models Along a Continuum of Visibility

**arXiv ID:** 2601.05739 | [PDF](https://arxiv.org/pdf/2601.05739v1)

**作者:** G M Shahariar `[一作]` (University of California), Zhouxing Shi `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 PII-VisBench 基准，用以评估视觉语言模型在不同在线可见度（从高可见到零可见）下对个人可识别信息（PII）的泄露与拒绝行为。

**💡 创新点**

首次将主体在线存在量作为维度，对 VLM 的隐私安全进行分层评估；揭示了“高可见度隐私缺口”以及模型在不同可见度下的拒绝率与泄露率的显著差异，并展示了 Prompt 变形与 Jailbreak 攻击对安全性的不同行为影响。

**🔧 技术方法**

采用了 18 种公开 VLM（0.3B–32B 参数）进行评估，使用两类自动判定器（目标字符串匹配与 LLM-as-judge）计算拒绝率（RR）与条件泄露率（cPDR），并结合 Prompt 重述与七种 Jailbreak 攻击进行鲁棒性测试。

**📊 数据集**

基准数据集由 200 个主体（高、中、低、零可见度各 50 人）与 20 类 PII（易、中、难三级）组成，共 4,000 条 Probe，主体图像来源于 CelebA、FFHQ、MMDT 等公开数据集，合成面孔使用 StyleGAN。

**📈 对比分析**

通过对比 RR 与 cPDR，发现可见度越低模型拒绝率越高、泄露率越低；不同模型家族及参数规模表现不一，部分大型模型如 InternVL3 14B 具备高拒绝率与低泄露率；Prompt 变形与 Jailbreak 攻击可显著降低拒绝率、提高泄露率，表明安全性易被绕过。

**⚠️ 局限性**

局限包括：仅评估公开 VLM 且不涉及闭源模型；可见度指标基于搜索结果的噪声估计；仅使用英语 PII，忽略多语言和文化差异；未验证生成 PII 的真实性，可能是幻觉；缺少对闭源系统的验证；数据集对公众人物偏向，可能导致族群偏差。

---

## 197. FeatureSLAM: Feature-enriched 3D gaussian splatting SLAM in real time

**arXiv ID:** 2601.05738 | [PDF](https://arxiv.org/pdf/2601.05738v1)

**作者:** Christopher Thirgood `[一作]` (University of Surrey), Simon Hadfield `[通讯]` (University of Surrey)

**通讯引用:** 5999 | [OpenAlex ID](https://openalex.org/A5091184063)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种能够实时在线运行的 FeatureSLAM 框架，在 3D Gaussian Splatting（3DGS）模型中嵌入基础模型特征，实现高质量的图像重建与开集语义分割。

**💡 创新点**

创新点包括：① 将多尺度 SAM2 特征自动编码并直接嵌入 3DGS，既保持实时性又提升语义表达；② 采用结构感知深度光栅化与多项式正则化，解决传统单值深度监督在纹理贫乏或遮挡区的误差；③ 引入基于特征匹配的 GICP 追踪与梯度驱动的语义剪枝策略，使地图保持紧凑且对低纹理场景鲁棒。

**🔧 技术方法**

使用的技术主要有：3D Gaussian Splatting、SAM2 视觉基础模型特征提取、LoRA+自编码器在线适配、结构感知深度栅格化、特征引导 GICP、语义梯度剪枝、per-splat 后向并行优化。

**📊 数据集**

在 Replica（合成室内无噪声）和 TUM RGB‑D（真实手持数据）两个基准数据集上进行评估，并使用 COCO、ScanNet 进行特征自编码器预训练。

**📈 对比分析**

与现有 RGB‑D NVS SLAM、闭集语义 SLAM、开集语义 SLAM 以及离线特征蒸馏 3DGS 方法相比，FeatureSLAM 在 ATE、PSNR、SSIM、LPIPS 及 mIoU 上均达或超过最优水平；实时率约 5 fps，单帧处理时间 5–6 ms，且训练时间仅为离线方法的 10 倍。

**⚠️ 局限性**

局限性包括：① 长期跟踪时仍易出现漂移，需进一步研究闭环与基于特征的全局优化；② 对极端光照变化或极度稀疏纹理的场景仍有挑战；③ 当前实现对 GPU 资源需求较高，移动平台部署仍受限。

---

## 198. mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations

**arXiv ID:** 2601.05732 | [PDF](https://arxiv.org/pdf/2601.05732v1)

**作者:** Yongyi Yang `[一作]` (University of Michigan), Jianyang Gao `[通讯]` (Nanyang Technological University)

**通讯引用:** 452 | [OpenAlex ID](https://openalex.org/A5036318925)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出mhc-lite，一种通过Birkhoff–von Neumann定理对双随机矩阵进行重参数化的残差连接，消除Sinkhorn–Knopp迭代带来的近似误差和工程难题。

**💡 创新点**

创新点在于用凸组合的排列矩阵直接构造完全双随机矩阵，保证了精确的双随机性、训练稳定性，并可仅用标准矩阵运算实现。

**🔧 技术方法**

技术包括双随机矩阵重参数化、软最大化得到权重、矩阵乘法组合排列矩阵，以及对比实验使用PyTorch实现。

**📊 数据集**

使用nanoGPT框架，在6层、12层、24层模型上训练，数据集包括Wikitext-2、Wikitext-103等。

**📈 对比分析**

与传统HC及MHC做对比，mhc-lite在训练损失和验证损失上相当或略优，梯度幅值更小，吞吐量更高，且不出现残差不稳定。

**⚠️ 局限性**

局限在于当残差流数量n增大时，排列矩阵数量n!指数增长，导致存储和计算成本上升，可通过采样子集来缓解。

---

## 199. A Stock-Flow Framework for Editorial Board Dynamics: The Case of Economics Journals, 1866-2019

**arXiv ID:** 2601.05727 | [PDF](https://arxiv.org/pdf/2601.05727v1)

**作者:** Alberto Baccini `[一作]` `[通讯]` (Università degli Studi di Siena), Alberto Baccini (Università degli Studi di Siena)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对经济学期刊编辑委员会的纵向动态进行建模和实证分析，提出了一个包含期刊人口统计、编辑席位流动和编辑成员流动的库存-流动框架。

**💡 创新点**

创新点在于：①首次将期刊与其编辑席位、编辑成员视为三个互联层级的库存系统；②通过库存-流动模型量化编辑席位和成员的创建、销毁、留存与流失；③针对不同时间跨度的观察窗口设计了规范化的年化率，解决了历史数据不均匀采样的问题。

**🔧 技术方法**

技术方法主要是：库存-流动数学建模、比例率与对称归一化、基于平均库存的增长率计算、分层分解（期刊层面与全体编辑层面）以及对流动数据的分布与箱线图可视化分析。

**📊 数据集**

使用数据集为GOELD（Gatekeepers of Economics Longitudinal Database），涵盖约1,724本EconLit期刊，时间跨度从1866年到2019年，按十年采样（1866–2006）以及2012、2019两年不规则采样。

**📈 对比分析**

通过对不同历史时期的库存、流动率和分布进行对比，发现了“二战后爆发”与“2006–2019结构性停滞”两大阶段。相较于以往仅作静态描述的研究，该方法能够揭示期刊生态的动态演化与结构转折；性能指标主要表现为对多重分层流动的可解释性和对长期趋势的捕捉。

**⚠️ 局限性**

局限性包括：①数据主要基于AEA定义的期刊列表，可能低估非英语和全球南方的期刊；②编辑职位与角色未细分，无法区分编辑主编与助理编辑等职能差异；③历史数据采样不均匀导致对短期波动的解析有限；④手工数据采集成本高，限制了研究可扩展性与跨学科应用。

---

## 200. Rotate Your Character: Revisiting Video Diffusion Models for High-Quality 3D Character Generation

**arXiv ID:** 2601.05722 | [PDF](https://arxiv.org/pdf/2601.05722v1)

**作者:** Jin Wang `[一作]` (Hunyuan), Ping Luo `[通讯]` (The University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RCM（Rotate your Character Model）——一种基于视频扩散的图像到视频框架，能够将任意姿态的单/多视图角色图像转化为标准姿态并生成1024×1024分辨率、全视角旋转的高质量3D角色视频；

**💡 创新点**

创新点包括：①分阶段的三步训练策略（姿态规范化→视角初始化→角色旋转）实现了姿态去耦和视角可控；②引入Camera Encoder与Plücker嵌入实现几何感知的相机姿态条件；③支持多达4张输入视图的条件，并在单一模型中完成全视角生成；④构造了RCM‑Wild和RCM‑Hard两套具有挑战性的基准数据集。

**🔧 技术方法**

核心技术为：Wan 2.2视频扩散模型（flow‑matching），Camera Encoder（Plücker embedding）、进阶训练策略、以及高分辨率视频生成框架。

**📊 数据集**

使用约46k个自研角色模型（每个含多姿态），生成约120k个训练视频；并在RCM‑Wild（113张高质量图像）和RCM‑Hard（140个复杂角色模型）两套基准上进行评测。

**📈 对比分析**

与CharacterGen、SyncDreamer、SV3D、Epidiff、Hi3D、AR‑1‑to‑3、Wan 2.1/2.2等多视角与视频扩散基线对比，基于人类主观评估在图像‑视频一致性、提示‑视频一致性、审美质量、运动质量、姿态规范化5项均获最高分；在RCM‑Hard的PSNR/SSIM评测中，RCM分别取得19.41/0.89，显著优于所有对照方法。

**⚠️ 局限性**

局限性包括：目前仅实现静态姿态旋转，未覆盖动态动画；训练过程需分阶段，耗时较长；对高质量大规模角色数据依赖强；未与完整3D重建管线无缝集成。

---

## 201. Visualising Information Flow in Word Embeddings with Diffusion Tensor Imaging

**arXiv ID:** 2601.05713 | [PDF](https://arxiv.org/pdf/2601.05713v1)

**作者:** Thomas Fabian `[一作]` `[通讯]` (Technical University Darmstadt), Thomas Fabian (Technical University Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出了一种基于扩散张量成像（DTI）的新可视化工具DONALD-D，用以展示自然语言表达中词嵌入的方向性信息流，进而分析模型层利用率与上下文对信息流的影响。

**💡 创新点**

创新点在于将神经网络嵌入空间视为二维格点，并借助DTI的扩散椭球和方向向量直观呈现词间信息传递，突破了传统仅关注单词位置的点图可视化方法。

**🔧 技术方法**

采用的技术包括：对每层词嵌入的隐藏维度做均值归约；基于离散梯度求解结构张量；计算方向向量与各向异性；用扩散椭球与色彩编码在二维矩阵中可视化信息流。

**📊 数据集**

实验数据集为Hugging Face公开模型（BERT、Longformer、GPT‑2、PEGASUS）的嵌入，使用单句示例（包括代词辨识与隐喻检测）进行演示。

**📈 对比分析**

通过比较四种模型在同一句输入下的层利用率与信息流分布，发现BERT、Longformer、PEGASUS表现相似且最高层利用率低；GPT‑2层分布更不均匀，早期层信息流显著；在代词与隐喻对照实验中，信息流差异主要集中于中间层，提示上下文对语义层的影响。

**⚠️ 局限性**

局限性包括：仅对隐藏维度做平均，可能掩盖细粒度信息；仅在单句示例上展示，缺乏大规模语料量化；未对剪枝或性能提升进行系统评估；DTI扩散模型为二维，无法完全捕捉三维嵌入空间的复杂结构。

---

## 202. Multimodal In-context Learning for ASR of Low-resource Languages

**arXiv ID:** 2601.05707 | [PDF](https://arxiv.org/pdf/2601.05707v1)

**作者:** Zhaolin Li `[一作]` (Karlsruhe Institute of Technology), Jan Niehues `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 4060 | [OpenAlex ID](https://openalex.org/A5046084081)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多模态上下文学习（MICL）在未覆盖低资源语言中的ASR性能。

**💡 创新点**

创新点在于利用语音与文本的对齐上下文实现语言无关的学习，并证明跨语言微调能匹配目标语言训练。

**🔧 技术方法**

采用Phi-4和Qwen3-Omni两大开放语音LLM，并结合SONAR检索、LoRA微调、以及基于MICL的候选重排序。

**📊 数据集**

使用Khinalug、Kichwa、Mboshi三种濒危语言的数据集，包含自发、广播、朗读等语音。

**📈 对比分析**

通过与传统ASR、文本LM以及多语言预训练的对比，MICL在未见语言上显著降低WER/困惑度，跨语言微调可与目标语言微调持平甚至超越。

**⚠️ 局限性**

局限在于仅评估单一ASR任务、语言覆盖不足、仅测试两款开放LLM，未覆盖闭源模型。

---

## 203. AIBoMGen: Generating an AI Bill of Materials for Secure, Transparent, and Compliant Model Training

**arXiv ID:** 2601.05703 | [PDF](https://arxiv.org/pdf/2601.05703v1)

**作者:** Wiebe Vandendriessche `[一作]` (Ghent University), Merlijn Sebrechts `[通讯]` (Ghent University)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5015754339)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了AIBoMGen平台，用于在AI模型训练过程中自动生成可签名、可验证的AI Bill of Materials（AIBOM）。

**💡 创新点**

创新点在于将in‑toto、Cryptographic signatures、Artifact hashing等技术集成到一个受限的训练管道中，形成第三方可信观察者而非依赖TEE，并实现零开销的AIBOM生成。

**🔧 技术方法**

使用了CycloneDX（SBOM）、in‑toto、FastAPI、Celery、RabbitMQ、MinIO、Trivy、签名/哈希等技术栈。

**📊 数据集**

示例实验基于公开数据集（如MNIST、CIFAR 等）和小型模型，未公开大规模数据集。

**📈 对比分析**

与传统手工或模型卡对比，AIBoMGen的AIBOM生成时间约为0.4s，几乎不影响训练；存储开销低于1%；能够完整检测所有篡改。

**⚠️ 局限性**

局限在于只能使用预定义的训练脚本，无法执行自定义代码或其他框架，且依赖单一签名密钥，未来需支持TEE和更丰富的框架。

---

## 204. Afri-MCQA: Multimodal Cultural Question Answering for African Languages

**arXiv ID:** 2601.05699 | [PDF](https://arxiv.org/pdf/2601.05699v1)

**作者:** Atnafu Lambebo Tonja `[一作]` (MBZUAI), Thamar Solorio `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Afri-MCQA，这是第一个覆盖15种非洲语言、12个国家、文本与语音双模态、约7.5k图像支持的多语言文化视觉问答基准。

**💡 创新点**

创新点在于①首次为非洲语言提供并行文本与语音的视觉文化问答数据；②设计控制实验区分语言理解与文化知识缺陷；③揭示语音处理是当前多模态模型在非洲语言上的主要瓶颈。

**🔧 技术方法**

采用多模态大语言模型（MLLMs）评估，包括Qwen 2.5-Omni、Gemma‑3n、Gemini‑2.5 Pro等；使用location-aware prompt、GPT‑4o-mini评判开放式答案、自动指标（准确率、chrF++、WER）和人工评估。

**📊 数据集**

使用Afri-MCQA数据集，包含图像、对应的多语言文本与语音问答，另外参考AfriXNLI、AfriMMLU、ASR和LID等基准进行控制实验。

**📈 对比分析**

与文本、语音、英文、母语以及多选与开放式问答四种设置比较，发现：开源模型在英文多选问答中最高可达约78%，但在母语或开放式问答中准确率低至2–38%；Gemini‑2.5 Pro在所有设置下表现最佳，且母语与英文差距最小；模型规模并未显著提升低资源语言性能。

**⚠️ 局限性**

局限性包括：覆盖语言有限（15种），无法代表非洲全部语言和文化细节；数据规模中等，主要用于评估而非预训练；评测仅涉及少数模型，结果可能不完全代表整个模型生态；数据收集仍可能带有注释者偏见与文化简化。

---

## 205. Drivora: A Unified and Extensible Infrastructure for Search-based Autonomous Driving Testing

**arXiv ID:** 2601.05685 | [PDF](https://arxiv.org/pdf/2601.05685v1)

**作者:** Mingfei Cheng `[一作]` (Singapore Management University), Yuan Zhou `[通讯]` (Zhejiang Sci-Tech University)

**通讯引用:** 2290 | [OpenAlex ID](https://openalex.org/A5083918663)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

开发了一个统一且可扩展的搜索式自动驾驶系统（ADS）测试平台 Drivora，整合了12种ADS和5种测试方法，并基于CARLA仿真实现。

**💡 创新点**

创新点在于：①提出低层可操作的场景定义 OpenScenario，统一不同测试方法的输入；②将测试流程拆分为测试算法、场景执行和ADS集成三大模块，支持多AV并行运行；③提供统一的Agent接口，显著降低ADS集成成本。

**🔧 技术方法**

主要技术包括：CARLA仿真环境、Python与Docker容器化、进化算法（evolutionary computation）用于搜索关键场景、并行执行框架以及统一的OpenScenario场景描述。

**📊 数据集**

使用CARLA自带的地图（如Town01）和自定义种子场景生成脚本；未使用公开的标准数据集，而是基于仿真产生的动态/静态车辆与行人场景。

**📈 对比分析**

对比方法尚未在大规模实验中给出定量评估，仅通过示例展示违规案例和并行执行吞吐量随实例数线性增长，说明平台在效率和多AV测试能力上的优势。

**⚠️ 局限性**

主要限制包括：缺乏系统性实验和对比数据；当前仅演示示例，未验证所有测试方法的全面效果；需进一步扩展高级突变策略和强化学习等新技术。

---

## 206. FLRQ: Faster LLM Quantization with Flexible Low-Rank Matrix Sketching

**arXiv ID:** 2601.05684 | [PDF](https://arxiv.org/pdf/2601.05684v1)

**作者:** Hongyaoxing Gul `[一作]`, Fangfang Liu `[通讯]` (Key Laboratory of System Software)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FLRQ（Flexible Low‑Rank Quantization），通过自适应选择每层的低秩分解来实现高效的后训练量化。

**💡 创新点**

创新点在于：①使用R1‑Sketch实现快速、可变的秩选择，避免了传统固定秩导致的浪费；②引入BLC（Best Low‑rank Approximation under Clipping）迭代方法进一步降低量化误差；③两者结合可在保持精度的同时显著降低额外内存和量化时间。

**🔧 技术方法**

核心技术包括：随机低秩投影（Rank‑1 Sketch）、基于裁剪的低秩量化（BLC）、激活校准缩放、以及GPU友好的BLAS Level‑2实现。

**📊 数据集**

使用WikiText2、C4作为量化校准和评价数据集，并在ARC、BoolQ、OpenBookQA、PIQA、Winogrande等零样本任务上进行下游评估。

**📈 对比分析**

与AWQ、LQER、OmniQuant、AffineQuant、Quip#、CALDERA、RILQ等方法比较，FLRQ在3‑4 位时的PPL与精度基本与FP16相当，2 位时仍保持低误差；额外内存仅为固定秩方法的40%以下，量化速度比SVD‑based方法快30%+，推理延迟仅提升4–6%。

**⚠️ 局限性**

局限性包括：仍需要校准数据集，迭代过程对极低精度（≤2 位）存在一定误差；当前实现主要针对权重量化，未涵盖激活量化；对极大模型的GPU显存仍有一定压力。

---

## 207. AGDC: Autoregressive Generation of Variable-Length Sequences with Joint Discrete and Continuous Spaces

**arXiv ID:** 2601.05680 | [PDF](https://arxiv.org/pdf/2601.05680v1)

**作者:** Yeonsang Shin `[一作]` (Seoul National University), Bohyung Han `[通讯]` (Seoul National University)

**通讯引用:** 19440 | [OpenAlex ID](https://openalex.org/A5006594639)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种自回归框架AGDC，用于联合建模可变长度序列中的离散与连续数值，实现高精度生成。

**💡 创新点**

创新点在于将离散预测与扩散连续建模结合，并通过MLP驱动的EOS logits调整和长度正则化实现可变长度精确控制。

**🔧 技术方法**

采用Transformer解码器+MLP+扩散网络，结合EOS logit调整和长度正则化损失。

**📊 数据集**

使用新构建的ContLayNet 334K半导体布局数据集，以及PubLayNet、Rico和FIGR-8-SVG等图形和SVG数据集。

**📈 对比分析**

与基线LayoutTransformer（离散化）和DLT（非自回归扩散）比较，AGDC在ContLayNet的四项DRC指标、PubLayNet和Rico的FID/Overlap/Alignment以及IconShop的SVG FID/CLIP上均实现显著提升。

**⚠️ 局限性**

局限在于长序列时误差累积仍显著，模型在极长或极高精度场景下仍难以完全克服错误传播。

---

## 208. Tracing Stereotypes in Pre-trained Transformers: From Biased Neurons to Fairer Models

**arXiv ID:** 2601.05663 | [PDF](https://arxiv.org/pdf/2601.05663v1)

**作者:** Gianmario Voria `[一作]` (University of Salerno), Fabio Palomba `[通讯]` (University of Salerno)

**通讯引用:** 9514 | [OpenAlex ID](https://openalex.org/A5033738898)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了Transformer预训练模型内部的社会偏见，构建了偏见关系数据集，利用知识神经元框架追踪并抑制这些偏见神经元，并评估抑制对软件工程任务的影响。

**💡 创新点**

创新点在于：①首次将知识神经元方法扩展到偏见关系；②构造了九类社会维度的偏见关系与激活提示数据集；③展示偏见知识高度局部化，可通过小规模神经元抑制有效降低偏见且几乎不损失任务性能。

**🔧 技术方法**

主要技术包括：Transformer（BERT）模型；神经元归因（Integrated Gradients）与基线归因；神经元抑制（激活置零）；偏见激活提示与对照提示的概率/困惑度评估；统计检验（Wilcoxon、Cliff’s Δ、Spearman）以及SE任务下的准确率/宏F1/困惑度比较。

**📊 数据集**

数据集：1,018条偏见关系（9个类别），10,180条对应的bias‑activating prompts；基准SE任务集合SELUM，包含5个非代码任务（incivility, tone‑bearing, requirement‑type, sentiment, requirement‑completion）。

**📈 对比分析**

对比方法：基线（无抑制） vs 抑制偏见神经元；对抗性提示与控制提示比较；使用Wilcoxon检验和Cliff’s Δ验证显著性。结果显示：偏见神经元平均1–3个，抑制后偏见提示困惑度提高70%–130%，对控制提示影响小；对SE任务的影响在±2–3%以内，甚至在MLM任务中提高流畅度。

**⚠️ 局限性**

局限性：仅在BERT系列模型上验证，未涵盖生成型或因果LLM；偏见关系与提示的构造依赖GPT-4o生成，可能存在误差；评估的SE任务有限，未覆盖所有软件工程场景；抑制效果对某些情感任务仍有轻微负面影响。

---

## 209. Motion Compensation for Real Time Ultrasound Scanning in Robotically Assisted Prostate Biopsy Procedures

**arXiv ID:** 2601.05661 | [PDF](https://arxiv.org/pdf/2601.05661v1)

**作者:** Matija Markulin `[一作]` (University of Zagreb), Bojan Šekoranja `[通讯]` (University of Zagreb)

**通讯引用:** 389 | [OpenAlex ID](https://openalex.org/A5073050197)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研发并验证了协作机器人系统，实现对前列腺超声扫描的主动运动补偿与三维重建，为前列腺活检提供更精准的支撑。

**💡 创新点**

创新点在于结合力控与视觉定位的双重运动补偿，实时保持探头与前列腺相对位置，并利用MicroSegNet加分类头实现即时分割，显著提升了扫描稳定性与重建精度。

**🔧 技术方法**

采用KUKA LBR/IIWA机器人、ROS2与MoveIt控制框架、PID力控与视觉定位技术；使用CUDA/PyTorch实现的MicroSegNet（CNN‑ViT+分类头）进行超声分割；点云配准采用ICP，并通过Hausdorff距离、fitness与RMSE评估。

**📊 数据集**

使用CIRS 070L前列腺假体（和Yezitronix等型号）在机器人平台上采集的自制超声图像数据集，对MicroSegNet进行训练与验证。

**📈 对比分析**

在静止、水平、垂直、组合四种运动场景下各采集30次扫面，通过ICP配准比较不同场景下的点云。结果显示，运动补偿平均延迟0.5 s，追踪误差≤3 mm；重建RMSE≈0.35–0.4 mm，fitness≥0.83，阈值0.8 mm提供了较好的精度与稳健性平衡。

**⚠️ 局限性**

局限性包括仅使用单一医学假体导致分割相对容易、力感测依赖关节扭矩估计存在误差、未在真实前列腺上验证、缺乏非线性控制与更复杂假体实验，以及未充分利用线性与凸探头信息来补偿x轴旋转。

---

## 210. EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents

**arXiv ID:** 2601.05777 | [PDF](https://arxiv.org/pdf/2601.05777v1)

**作者:** Yaoqi Guo `[一作]` (Nanyang Technological University), Zhenpeng Chen `[通讯]` (Tsinghua University)

**通讯引用:** 443 | [OpenAlex ID](https://openalex.org/A5101612444)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于经验的早期终止方法（EET），通过提取历史问题解决经验并在补丁生成与选择过程中动态决定是否提前停止，从而显著降低软件工程代理的成本。

**💡 创新点**

创新点在于将结构化的历史经验（问题抽象、轨迹抽象、可靠性评分等）与检索机制相结合，用于引导补丁迭代的早期终止，而不需要对现有代理进行重构。

**🔧 技术方法**

技术包括经验生成与压缩（问题与轨迹抽象、评估抽象、可靠性与质量评估）、TF‑IDF检索、基于经验的置信度评分与阈值决策以及在补丁生成与选择两阶段的动态早停策略。

**📊 数据集**

实验使用了 SWE‑bench Verified 作为评估基准，并使用 SWE‑bench Lite（去重后207个任务）来构建经验库；同时对 Agentless、Mini‑SWE‑Agent、Trae Agent 三类代理在 GPT‑5‑mini 与 DeepSeek‑V3.2 两大 LLM 后端上进行评测。

**📈 对比分析**

与两类基线（turn‑control 与 naive‑RAG）比较，EET 在保持（或提升）问题解决率（最多 0.2% 下降）的同时，实现了 19.3%–55.1%（平均 31.8%）的总成本下降，并在 API 调用、输入/输出 token 方面也取得显著缩减，表现出更优的成本‑性能平衡。

**⚠️ 局限性**

局限性包括：评估仅限于 SWE‑bench Verified；依赖历史数据，面临冷启动问题；对新颖或数据稀缺的领域可能效果不足；未来需在更多基准和工业环境中验证。

---

## 211. One Script Instead of Hundreds? On Pretraining Romanized Encoder Language Models

**arXiv ID:** 2601.05776 | [PDF](https://arxiv.org/pdf/2601.05776v1)

**作者:** Benedikt Ebing `[一作]` (University of Würzburg), Goran Glavaš `[通讯]` (University of Würzburg)

**通讯引用:** 3945 | [OpenAlex ID](https://openalex.org/A5079336821)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了拉丁化在多语言编码器预训练中的影响，比较原始脚本与拉丁化文本的下游表现。

**💡 创新点**

首次系统评估高资源语言多语言预训练中拉丁化的损失与收益，并证明拉丁化不会导致跨语言干扰。

**🔧 技术方法**

采用BERT编码器、BPE分词、两种拉丁化工具（Uroman与Uconv），在六种语言上进行预训练与微调。

**📊 数据集**

使用Fineweb‑2大规模清洗文本作为预训练语料，配合五个标准下游任务（XNLI、SIB200、MASSIVE、WikiAnn、MASSIVE slot）。

**📈 对比分析**

通过在相同数据、相同计算预算下对比原脚本与拉丁化模型的平均微调性能，发现拉丁化在段落脚本几乎无损，语音文字脚本平均下降≤1.5%。

**⚠️ 局限性**

仅覆盖六种语言和脚本，模型规模与预训练数据有限，未探讨更大规模模型或其他脚本的表现。

---

## 212. More Power to the Particles: Analytic Geometry for Partial Optimal Transport-based Fluid simulation

**arXiv ID:** 2601.05765 | [PDF](https://arxiv.org/pdf/2601.05765v1)

**作者:** Cyprien Plateau--Holleville `[一作]` (Inria Paris-Saclay), Bruno Lévy `[通讯]` (U. Paris-Saclay)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于部分最优输运的流体模拟与渲染框架，利用解析计算的Laguerre细胞与球面交集精确获取自由表面并进行物理模拟和光线追踪渲染。

**💡 创新点**

创新点在于通过解析方法直接求解受限Laguerre细胞的体积与面积，消除了离散球面逼近带来的误差与计算量；并将该几何结构与梯度/海森矩阵凸优化结合，形成高效的流体模拟与渲染管线。

**🔧 技术方法**

使用的技术包括半离散/部分最优输运、Laguerre图与球面交集的解析几何、基于梯度/海森矩阵的凸优化求解权重、Lagrangian Navier‑Stokes 物理积分、GPU 加速的光线追踪与 impostor 渲染等。

**📊 数据集**

实验数据集主要为合成流体仿真场景，如1000 个或 100000 个细胞的球滴、粘性兔子等大规模细胞分布。

**📈 对比分析**

与传统基于球面离散的实现相比，平均模拟时间从 177 ms 降低到 162 ms，渲染时间仅 9 ms；在大规模（10万细胞）场景下仍保持良好性能，显示出显著的速度与精度提升。

**⚠️ 局限性**

局限性包括对受限单元内部点的手工选取可能导致数值不稳定；极端几何变化时仍需改进稳定性；目前实现主要在单机 CPU/GPU，尚未充分并行化，且对复杂边界或多相交互的支持有限。

---

## 213. FlyPose: Towards Robust Human Pose Estimation From Aerial Views

**arXiv ID:** 2601.05747 | [PDF](https://arxiv.org/pdf/2601.05747v1)

**作者:** Hassaan Farooq `[一作]` (Universität der Bundeswehr Munich), Peter St\ütz `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套轻量化的人体检测与姿态估计管线 FlyPose，专门针对无人机航空视角实现实时推理。

**💡 创新点**

创新点包括：① 多数据集联合训练提升对航空视角的泛化；② 在检测与姿态网络中引入 Normalized Wasserstein Distance Loss 与 TensorRT 优化，实现 20 ms/帧的边缘推理；③ 自制 FlyPose‑104 挑战性航空姿态数据集，填补数据缺口。

**🔧 技术方法**

核心技术：RT‑DETRv2‑S（轻量检测网络）、ViTPose‑S（姿态网络）、Normalized Wasserstein Distance Loss、TensorRT FP32 引擎、CUDA jetson‑utils 预处理。

**📊 数据集**

使用的数据集：Manipal‑UAV、VisDrone、HIT‑UAV、FlyPose‑104（自制 104 张图像）、UAV‑Human、COCO‑Person（训练集/验证集）。

**📈 对比分析**

性能对比：多数据集训练后检测平均提升 6.8 mAP；姿态估计在 UAV‑Human 上达到 73.18 mAP（比 AlphaPose 提升 16.3 mAP）；在 Jetson Orin 上实现 19.5 ms 推理（13 ms 检测 + 6.5 ms 姿态），满足 25 fps 需求。

**⚠️ 局限性**

局限：① 对极小人目标及严重遮挡的检测召回仍偏低；② 热成像图像下姿态精度下降；③ 多人场景下的推理延迟与批处理效果尚未完全评估；④ 受限于目前航空姿态标注数据量，模型在更广泛场景的泛化仍有提升空间。

---

## 214. ViTNT-FIQA: Training-Free Face Image Quality Assessment with Vision Transformers

**arXiv ID:** 2601.05741 | [PDF](https://arxiv.org/pdf/2601.05741v1)

**作者:** Guray Ozgur `[一作]` (TU Darmstadt), Fadi Boutros `[通讯]` (Fraunhofer IGD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的面部图像质量评估方法 ViTNT‑FIQA，利用 Vision Transformer（ViT）中各层 patch 嵌入的稳定性来判定面部图像对识别任务的适用性。

**💡 创新点**

创新点在于：①仅用单次前向传播而不需要反向传播或多次 dropout；②通过比较相邻 Transformer 层归一化后的 patch 嵌入的欧氏距离，捕捉图像质量的“特征演化稳定性”；③结合 ViT 自注意力权重实现空间加权聚合。

**🔧 技术方法**

技术手段包括：ViT 结构、patch 嵌入、层间欧氏距离、L2 归一化、注意力权重聚合，以及基于这些特征的无监督质量评分映射。

**📊 数据集**

使用合成的 SynFIQA 数据集验证方法有效性，并在八大公开基准（LFW、AgeDB‑30、CFP‑FP、CALFW、Adience、CPLFW、XQLFW、IJB‑C）以及四种主流 FR 模型（ArcFace、ElasticFace、MagFace、CurricularFace）进行评估。

**📈 对比分析**

与现有 SOTA 方法（SER‑FIQ、GraFIQ、DifFIQA 等）对比，ViTNT‑FIQA 在 FMR=1e‑3/1e‑4 下的 pAUC‑EDC 与或优于对手，并且只需单次前向推断，计算开销大幅降低。

**⚠️ 局限性**

局限性包括：对 ViT 预训练模型的依赖，非面部识别专用的基础模型（如 CLIP）表现略逊；未进一步探讨对极端遮挡、光照变化等场景的鲁棒性；以及仅利用相邻层距离，可能未能充分利用更深层的语义信息。

---

## 215. TAGRPO: Boosting GRPO on Image-to-Video Generation with Direct Trajectory Alignment

**arXiv ID:** 2601.05729 | [PDF](https://arxiv.org/pdf/2601.05729v1)

**作者:** Jin Wang `[一作]` (Hunyuan, Tencent), Ping Luo `[通讯]` (The University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TAGRPO框架，利用轨迹对齐损失和记忆库对图像到视频生成模型进行后训练。

**💡 创新点**

创新点在于将相同初始噪声下的样本之间的相对奖励关系作为监督信号，通过轨迹级对齐拉近高奖励轨迹、远离低奖励轨迹，同时使用记忆库降低生成开销。

**🔧 技术方法**

采用流匹配扩散模型、GRPO强化学习、对比学习思想的记忆池、以及视频奖励模型Q‑Save和HPSv3进行评估。

**📊 数据集**

使用内部约1万条图像‑文本对数据集训练，评估集TAGRPO‑Bench包含200条挑战性图像‑文本对。

**📈 对比分析**

与基准模型和DanceGRPO对比，在Wan 2.2和HunyuanVideo 1.5上在320p/720p分辨率下均获得更高的Q‑Save和HPSv3分数，收敛更快、奖励提升明显。

**⚠️ 局限性**

局限在于仅在图像到视频任务上验证，依赖已训练的奖励模型，对多模态或更大规模数据的适用性尚待进一步探索。

---

## 216. SketchVL: Policy Optimization via Fine-Grained Credit Assignment for Chart Understanding and More

**arXiv ID:** 2601.05688 | [PDF](https://arxiv.org/pdf/2601.05688v1)

**作者:** Muye Huang `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 73099 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SketchVL模型并采用FinePO强化学习算法，实现对图表理解任务的细粒度信用分配，显著提升模型推理准确性。

**💡 创新点**

创新点在于：1）将中间推理步骤以可视化标记形式投射到图像中，形成可追踪的推理轨迹；2）设计FinePO算法，通过FinePRM评估每一步的质量，实现跨步骤细粒度信用重新分配；3）引入KL正则化以避免动作偏倚。

**🔧 技术方法**

主要技术包括多模态大型语言模型（Qwen2.5VL）、Reasoning on Image（RoI）范式、FinePO强化学习框架、FinePRM过程奖励模型以及交叉模态蒸馏数据生成 pipeline。

**📊 数据集**

使用的训练与评测数据集包括：EvoChart、GQA、ChartQA‑Train（cold‑start 50K）、ChartSketcher pipeline、OpenImages、ChartQA‑Train‑Augmented、ChartQA‑Train‑Human、Vision‑R1‑RL、ChartBench‑Train、VisualCoT‑Train；评测基准覆盖 EvoChart‑QA、ChartQA、ChartQA‑Pro、ChartBench、PlotQA、MathVista、MMStar。

**📈 对比分析**

与基线模型（如Qwen2.5VL‑7B/3B、VLM‑R1、ChartSketcher‑2B）对比，SketchVL‑7B 在多数图表专家数据集上平均提升约7.23%，SketchVL‑3B 在 ChartQA 与 ChartBench 等任务上提高 15.32% 与 3.76% 的分数；在通用多模态数据集 MathVista、MMStar 上亦保持或超过竞争模型。

**⚠️ 局限性**

局限性包括：1）缺乏用于定量评估 FinePRM 的标准基准；2）FinePRM 的训练样本仍可能存在偏倚，影响信用分配的客观性；3）对极端稀有图表格式的泛化能力尚待进一步验证。

---

## 217. CHDP: Cooperative Hybrid Diffusion Policies for Reinforcement Learning in Parameterized Action Space

**arXiv ID:** 2601.05675 | [PDF](https://arxiv.org/pdf/2601.05675v1)

**作者:** Bingyi Liu `[一作]` (Wuhan University of Technology), Zhuangzhuang Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 144 | [OpenAlex ID](https://openalex.org/A5100697207)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Cooperative Hybrid Diffusion Policies (CHDP) 框架，用两位协作代理分别采用离散扩散策略和连续扩散策略，联合建模并生成混合动作；

**💡 创新点**

创新点包括将混合动作空间视为完全合作游戏；使用扩散模型提升策略多模性；采用顺序更新机制解决多代理同步更新冲突；构建 Q‑引导的代码本，将高维离散动作嵌入低维潜在空间并通过 Q 值对齐；

**🔧 技术方法**

核心技术为扩散模型（Diffusion Policy）与 DQL、VQ‑VAE 代码本、Q‑guided 代码本、顺序更新策略、双重 Q‑学习（Double Q）以及基础 TD3 算法；

**📊 数据集**

实验使用 8 个标准 Parameterized Action MDP benchmark：Platform、Goal、Catch Point、Hard Goal、以及 Hard Move（n=4、6、8、10）等；

**📈 对比分析**

与 HyAR、PDQN‑TD3、PA‑TD3、HHQN‑TD3、HPPO 等 SOTA 基线对比，CHDP 在所有任务上均优于对手，最高提升 19.3% 成功率，且在高维离散空间中仍保持 90%+ 成功率，采样效率更高；

**⚠️ 局限性**

局限性：依赖扩散模型训练成本较高，代码本设计仍需手工调参；在真实机器人或更高维连续参数的任务中验证不足；可解释性与计算效率等方面仍待进一步研究。

---

## 218. HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation

**arXiv ID:** 2601.05656 | [PDF](https://arxiv.org/pdf/2601.05656v1)

**作者:** Rongxin Chen `[一作]` (Institute of Computing Technology), Huawei Shen `[通讯]` (Institute of Computing Technology)

**通讯引用:** 6646 | [OpenAlex ID](https://openalex.org/A5047897879)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于层次人口分布树的主题自适应代理生成框架HAG，用以提升代理模型在不同主题下的宏观分布匹配与微观个体一致性；

**💡 创新点**

创新点在于将世界知识模型用于自动构建主题自适应分布树，结合真实数据检索与基于LLM的增量生成，实现宏观层级依赖建模与微观真实一致性的双重保证；

**🔧 技术方法**

采用世界知识模型（WKM）进行维度优先级排序与条件概率推断，层次树构造与边权计算；利用World Values Survey（WVS）数据库进行真实用户检索与数据补全；在生成阶段结合LLM进行增量生成与一致性约束；

**📊 数据集**

主要使用World Values Survey（WVS）作为真实数据来源，构建12个关键维度的社会人口特征；并在Bluesky社交数据、Amazon评论和IMDB用户评论三大公开语料库上进行主题人口基准构建与评估；

**📈 对比分析**

与随机采样、主题检索、LLM直接生成及HAG-Flat（无层次依赖）等基线进行对比，使用PACE评估框架中的JSD、KL、DivErr、ArchRel、IndCon等指标，实验显示HAG平均提升人口对齐度37.7%、社会一致性18.8%；

**⚠️ 局限性**

主要局限在于对世界知识模型的依赖，若WKM知识不足或出现幻觉会导致分布树失真；同时使用的WVS数据静态且可能存在时间滞后，需定期更新或引入实时数据以维持准确性；

---

## 219. Nonlinearity Mitigation for Coherent Ground-to-Satellite Optical Links

**arXiv ID:** 2601.05655 | [PDF](https://arxiv.org/pdf/2601.05655v1)

**作者:** Stella Civelli `[一作]` (National Research Council Institute of Electronics, Computer and Telecommunications Engineering), Luca Potì `[通讯]` (National Interuniversity Consortium for Telecommunications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a7d414a-27d1-4de0-aac0-e554088edeb4`

**🎯 论文内容**

研究并提出了低复杂度数字信号处理技术（概率幅度整形与非线性相位旋转）来缓解地面站高功率光放大器的Kerr非线性，提高光学地-卫星链路的可接受损耗。

**💡 创新点**

结合4D球面整形与极短块长度（N=4）的查找表实现，以及在发射机与接收机之间分配非线性相位旋转，显著提升链路容忍损耗且保持低硬件复杂度。

**🔧 技术方法**

采用概率幅度整形（PAS）、4D球面映射、非线性相位旋转（NLPR）、查找表实现、SSFM仿真以及基于通用互信息（GMI）的性能评估。

**📊 数据集**

文中使用模拟数据（100 GBd QAM调制、256QAM/64QAM等），无真实数据集，全部为数值仿真结果。

**📈 对比分析**

通过比较统一分布与整形+NLPR方案下的可接受链路损耗，显示目标GMI 3/5 bit/2D时，链路损耗分别提升3.5 dB和6 dB，显示方法在受限带宽110 GHz下仍保持显著优势。

**⚠️ 局限性**

受限于带宽限制导致NLPR效果下降；需要在接收端增加轻微复杂度；查找表实现仍依赖于块长度N=4，可能未完全针对高功率放大器最优；仿真未包含真实大气衰减与相位噪声等实际干扰。

---

## 220. From Issues to Insights: RAG-based Explanation Generation from Software Engineering Artifacts

**arXiv ID:** 2601.05721 | [PDF](https://arxiv.org/pdf/2601.05721v1)

**作者:** Daniel Pöttgen `[一作]` (University of Cologne), Andreas Vogelsang `[通讯]` (University of Duisburg-Essen)

**通讯引用:** 1370 | [OpenAlex ID](https://openalex.org/A5022998651)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建并评估了一套基于检索增强生成（RAG）的系统，利用软件项目的 issue‑tracking 数据自动生成关于系统行为的自然语言解释。

**💡 创新点**

创新点在于首次将 issue‑tracking 视为可检索知识源，通过多阶段检索+LLM 生成解释，并证明其生成的解释与人工撰写的解释能够达到90%+ 的相似度。

**🔧 技术方法**

技术实现包括：对 issue 文本进行分块并使用 multi‑qa‑mpnet‑base‑dot‑v1 生成嵌入，存入 Chroma 向量库；采用 LLM（Granite 3.1、Phi‑4、Qwen 2.5 等）对查询进行重写、检索、去重和重排序；最终用本地部署的 LLM 生成只依赖检索文档的解释。

**📊 数据集**

使用的数据集为 Mattermost 开源项目的 GitHub issue（约 10,000 条），并构造了三组评测集：系统特定 QA、Out‑of‑Domain 可信度检验和随机化答案鲁棒性检验。

**📈 对比分析**

评估采用 LLM‑as‑Judge（Granite 3.1）进行 Answer‑Reference 相似度、Faithfulness、Helpfulness 与 Document Relevance 四维度打分；在系统特定 QA 上平均 ARS≥0.90，Faithfulness≥0.90，Helpfulness≥0.98，Document Relevance 94–100%；Out‑of‑Domain 中大模型表现出更高的指令遵从和更低的幻觉率。

**⚠️ 局限性**

局限性包括对 issue‑tracking 数据完整性与质量的高度依赖；敏感/私密信息可能导致不适合公开解释；仅在 Mattermost 上验证，跨项目泛化需要进一步研究；LLM 的可解释性仍有限，且小模型易产生幻觉。

---

## 221. Variational Autoencoders for P-wave Detection on Strong Motion Earthquake Spectrograms

**arXiv ID:** 2601.05759 | [PDF](https://arxiv.org/pdf/2601.05759v1)

**作者:** Turkan Simge Ispak `[一作]` (Middle East Technical University), Erdem Akagunduz `[通讯]` (Middle East Technical University)

**通讯引用:** 578 | [OpenAlex ID](https://openalex.org/A5005758418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究用自监督变分自编码器检测强震谱图中的P波，并系统评估不同架构对重构与检测权衡的影响。

**💡 创新点**

通过全网格搜索四种VAE结构，揭示跳接导致过度泛化、注意力机制提升检测性能，证明全球上下文优先是关键。

**🔧 技术方法**

使用变分自编码器（Basic、Skip、Attention、Hybrid），引入U‑Net跳接与Transformer自注意力，采用MSE+KL损失，并用MAE与NCC评估。

**📊 数据集**

采用土耳其国家强震网络（TNSMN）648条加速度记录，结合噪声合成与谱图化，构成1920个增强样本。

**📈 对比分析**

与四种架构对比，Attention‑VAE在0–40km内AUC达0.91、全局最高0.8749；Skip‑VAE AUC 0.821、MAE最低；Hybrid位于两者之间。

**⚠️ 局限性**

受限于低标签数据导致纯Attention不稳定、对窗口对齐高度敏感，模型对远场及复杂噪声的鲁棒性有限。

---

## 222. VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit

**arXiv ID:** 2601.05755 | [PDF](https://arxiv.org/pdf/2601.05755v1)

**作者:** Junda Lin `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27290 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于验证-先决-提交（verify-before-commit）机制的LLM代理安全框架VIGIL，解决工具流注入攻击。

**💡 创新点**

创新点在于：①将代理推理与执行分离，利用意图锚定生成安全约束；②用感知消毒器净化工具描述；③通过投机推理构造候选执行路径并用基于意图的验证器筛选；④结合轨迹记忆实现自适应回溯，兼顾安全与推理灵活性。

**🔧 技术方法**

核心技术包括：意图锚定（Intent Anchor）、感知消毒器（Perception Sanitizer）、投机推理器（Speculative Reasoner）、基于意图的验证器（Grounding Verifier）以及已验证轨迹记忆（Validated Trajectory Memory）。

**📊 数据集**

使用自构建的SIREN基准：959个工具流注入案例（覆盖5个攻击向量）以及949个数据流基准案例（来自AgentDojo），对Qwen3‑max和Gemini‑2.5‑pro两大LLM进行评测。

**📈 对比分析**

与7种主流防御（输入层、静态隔离、动态防御）在ASR、UA、BU等指标上对比。VIGIL在工具流攻击中将ASR降低约22%，UA提升超过2倍，BU与无防御模型相近，整体处于安全‑效用最佳区间。

**⚠️ 局限性**

主要局限：投机推理与多轮验证导致计算开销较高；安全约束基于初始用户意图，可能无法适应动态生成的新子目标；在极高攻击密度或复杂任务下，轨迹空间巨大仍需进一步优化。

---

## 223. DynaDebate: Breaking Homogeneity in Multi-Agent Debate with Dynamic Path Generation

**arXiv ID:** 2601.05746 | [PDF](https://arxiv.org/pdf/2601.05746v1)

**作者:** Zhenghao Li `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27290 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多智能体辩论框架 DynaDebate，用动态路径生成、过程中心辩论和触发式验证三步提升多智能体的推理多样性与准确性。

**💡 创新点**

创新点在于：1）通过专门的路径生成代理动态生成互异、逻辑严谨的解题路径打破初始化同质化；2）将辩论焦点转向对每一步推理进行第一原理审计，提升过程正确性；3）引入触发式验证代理在出现僵局时调用外部工具提供客观依据，缓解盲从。

**🔧 技术方法**

技术包括：大型语言模型（GPT‑4o‑mini、Qwen3‑8B）、自定义多智能体协同框架、路径生成与自适应冗余分配、步骤级同行评审、外部工具调用（代码执行、搜索引擎）等。

**📊 数据集**

使用的公开数据集有：数学推理（GSM8K、MATH500、AIME 2024‑2025）、通用知识（MMLU）、事实准确性（Biography）。

**📈 对比分析**

与六类基线（单体CoT、CoT‑SC、Self‑Refine；多体MAD、SoM、DMAD）进行对比，DynaDebate 在 GPT‑4o‑mini 与 Qwen3‑8B 上多项指标均表现最佳，尤其在 AIME、MATH500 及 MMLU 上显著提升，单体工具增强版本仍低于 DynaDebate。

**⚠️ 局限性**

主要局限是：对推理难度低的任务（如 GSM8K）提升有限，且多轮辩论与路径生成带来额外 token 消耗，成本与收益不成正比。

---

## 224. The Echo Chamber Multi-Turn LLM Jailbreak

**arXiv ID:** 2601.05742 | [PDF](https://arxiv.org/pdf/2601.05742v1)

**作者:** Ahmad Alobaid `[一作]` (NeuralTrust), Joan Vendrell `[通讯]` (NeuralTrust)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现一种名为 Echo Chamber 的多轮 jailbreak 攻击，研究其在多种主流 LLM 上的效果，并提供自动化工具。

**💡 创新点**

创新点在于：① 通过“poisonous seeds”和“steering seeds”先植入隐蔽种子，随后利用模型自身的一致性偏好逐步放大有害信息，形成“回声室”效应；② 将攻击完全黑盒化，利用另一 LLM 作为攻击者与评判者实现自动化；③ 通过多轮对话而非单一指令，显著突破现有安全防护。

**🔧 技术方法**

技术方法包括：多轮 prompt 设计、对话历史操控、LLM‑as‑judge（主次评判器）、自动化脚本（基于 API 的交互）、实验对比框架与统计分析。

**📊 数据集**

使用的数据集主要有：AdvBench benchmark（包含 Violence、Hacking、Fraud、Misinformation 四类 12 个任务）以及自定义的三类极端任务（种族宣言、疫苗误信息、制毒说明）。

**📈 对比分析**

与 Crescendo、Chain of Attack、Foot‑in‑the‑Door（多轮）以及 DAN（单轮）进行对比，评估指标为成功率。实验显示 Echo Chamber 在 12 个 AdvBench 任务上成功率为 45.0%，显著高于 Crescendo（28.6%）和 DAN（9.5%）。在 Violence、Hacking 类别表现最突出；在 Fraud 类别略逊于 Crescendo，但在 Misinformation 上仍有 25% 成功率。

**⚠️ 局限性**

局限性：① 需要模型具备对话历史功能；② 依赖黑盒 API，若接口限制会受限；③ 攻击效果受攻击者 LLM 的对抗性影响，若攻击者自身被限制可能失败；④ 自动化评判器可能出现误判；⑤ 目前仅在公开 LLM 上验证，针对自研或高度安全模型的效果未知。

---

## 225. Logic-Parametric Neuro-Symbolic NLI: Controlling Logical Formalisms for Verifiable LLM Reasoning

**arXiv ID:** 2601.05705 | [PDF](https://arxiv.org/pdf/2601.05705v1)

**作者:** Ali Farjami `[一作]` (University of Luxemburg), Marco Valentino `[通讯]` (University of Sheffield)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种逻辑可参数化的神经符号自然语言推理框架，将不同逻辑嵌入 HOL 以实现 LLM 与定理证明器的交互。

**💡 创新点**

创新点在于将逻辑本身视为可配置的第一类参数，而非固定背景，系统评估逻辑内部与外部策略对规范推理的影响。

**🔧 技术方法**

采用 LogiKEy 语义嵌入、Isabelle/HOL 定理证明、LLM（GPT‑4o、DeepSeek‑V1）自动化形式化与解释精炼。

**📊 数据集**

使用自制的 BENR（Bioethical Explanations and Normative Reasoning）数据集，包含 103 条伦理推理实例。

**📈 对比分析**

通过比较 FOL、KD、DDLE、DDL_CJ 四种逻辑在不同 LLM 上的解释成功率、精炼步数、求解时间和语法错误率；实验显示 KD 在规范推理中最高成功率 77.67%，并且收敛步数最少、求解速度最快。

**⚠️ 局限性**

局限性包括仅评估两种大型 LLM 与四种逻辑，未覆盖更复杂语言输入，且精炼策略单一，可能不具普适性。

---

## 226. Circular Reasoning: Understanding Self-Reinforcing Loops in Large Reasoning Models

**arXiv ID:** 2601.05693 | [PDF](https://arxiv.org/pdf/2601.05693v1)

**作者:** Zenghao Duan `[一作]` (Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 20528 | [OpenAlex ID](https://openalex.org/A5029998682)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构造LoopBench数据集，系统研究并量化大型推理模型（LRMs）中出现的循环推理（Circular Reasoning）现象；

**💡 创新点**

创新点在于提出循环推理的定义与两类循环（数值循环与语句循环），揭示其内部状态崩塌、语义循环先于文本重复以及V形注意力机制；

**🔧 技术方法**

采用内部隐藏状态监测、语义相似度与L2距离分析、注意力分布可视化、循环图拓扑建模，并在此基础上设计基于CUSUM的实时循环预测与早期干预；

**📊 数据集**

使用LoopBench（700个样本，涵盖高精度算术与复杂递归推理任务）作为评测基准，同时与AIME2025和SuperGPQA等传统基准对比；

**📈 对比分析**

在多种开源与闭源LRMs上进行实验，结果显示LoopBench能显著提高循环触发率（最高数值循环率≈30%、语句循环率≈45%），而CUSUM方法在所有模型上实现了>70%的早期检测率、<30%的误报率，提前≈40句/1500字检测到循环；

**⚠️ 局限性**

局限性包括：只针对高精度算术与递归推理场景，未覆盖其他任务；机制分析主要基于DeepSeek与Qwen系列，缺乏跨体系验证；干预手段仅适用于推理时检测与终止，未提供训练阶段根除循环的解决方案。

---

## 227. Secure Multiuser Beamforming With Movable Antenna Arrays

**arXiv ID:** 2601.05686 | [PDF](https://arxiv.org/pdf/2601.05686v1)

**作者:** Zhenqiao Cheng `[一作]` (China Telecom Beijing Research Institute), Xingqi Zhang `[通讯]` (University of Alberta)

**通讯引用:** 4172 | [OpenAlex ID](https://openalex.org/A5082899535)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在多用户安全通信场景中，提出基于可移动天线(MA)的框架，同时对数字波束成形和MA位置进行联合优化以最大化总保密率。

**💡 创新点**

首次将MA位置与多用户安全波束成形相结合，利用分数规划和块坐标下降实现可解的非凸优化，并通过MA提供的空间自由度显著提升保密率。

**🔧 技术方法**

采用分数规划、辅助变量变换、块坐标下降、凸二次约束优化、1维搜索/网格搜索以及闭式解法。

**📊 数据集**

实验使用基于LoS+NLoS的随机多径信道模型，生成随机角度、幅值和噪声，采用仿真数据而非公开数据集。

**📈 对比分析**

与固定位置天线(FPA)基线比较，MA方案在不同SNR、天线数、用户数和窃听人数下均实现更高的总保密率；收敛速度快，约20次迭代即可稳定。

**⚠️ 局限性**

主要限制包括：需要对MA位置进行离散网格搜索导致计算复杂度较高；对完美CSI假设敏感；在高SNR下泄漏随功率提升而加剧，保密率趋于下降；算法在极大规模系统中可扩展性待验证。

---

## 228. On the closest pair of points problem

**arXiv ID:** 2601.05681 | [PDF](https://arxiv.org/pdf/2601.05681v1)

**作者:** Martin Hitz `[一作]` (University of Klagenfurt), Michaela Hitz `[通讯]` (University of Klagenfurt)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出两种确定性、非递归的最接近点对算法（cppMM 和 cppAPs），并证明 cppMM 在均匀分布点集上实现线性时间。

**💡 创新点**

创新点在于利用数学最优包装理论给出点集最小距离上界，从而构建 O(n) 大小网格实现线性时间；同时提出改进的暴力算法 cppAPs。

**🔧 技术方法**

采用基于网格划分的空间哈希、排序和两两比较；cppMM 结合上界 δ̅；cppAPs 采用按 x 坐标排序的活窗口。

**📊 数据集**

使用随机生成的二维均匀分布和截断正态分布点集，规模从 2^10（1024）到 2^25（33 554 432）点。

**📈 对比分析**

在 C++ 环境下与经典暴力、分治、平面扫描、Rabin‑Lipton、Khuller‑Matias 等算法比较；在均匀分布下 cppAPs 在 n≤2^21 时最快，随后 cppMM 以线性时间优于其它；在截断正态分布时 cppAPs 仍在 n≤2^23 最快。

**⚠️ 局限性**

最坏情况下 cppMM 可退化为 O(n^2)；算法对点集分布敏感，在极端集中分布时性能下降；实现仍依赖于均匀性假设。

---

## 229. Analysing Differences in Persuasive Language in LLM-Generated Text: Uncovering Stereotypical Gender Patterns

**arXiv ID:** 2601.05751 | [PDF](https://arxiv.org/pdf/2601.05751v1)

**作者:** Amalie Brogaard Pauli `[一作]` (Aarhus University), Ira Assent `[通讯]` (Aarhus University)

**通讯引用:** 3984 | [OpenAlex ID](https://openalex.org/A5104360871)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一个框架，用于评估用户指令如何影响大型语言模型（LLMs）生成的劝说性语言，特别是针对不同性别的接收者、发送者意图或输出语言的影响。

**💡 创新点**

创新点在于系统性地评估控制提示属性对LLM生成劝说性语言的影响，并揭示了在所有模型中存在显著的性别差异，这些差异反映了社会心理学和社会语言学中记录的性别刻板印象。

**🔧 技术方法**

使用了13个大型语言模型（LLMs）和16种语言，采用了LLM作为评判者的设置，结合社会心理学和传播科学的理论。

**📊 数据集**

研究中使用了多种数据集，包括300个生成的提示，涵盖了人际劝说消息和政治论证的领域。

**📈 对比分析**

通过LLM作为评判者的设置，比较了不同模型生成的劝说性语言，结果显示所有测试的LLMs在性别处理上表现出显著的刻板印象差异，女性目标的消息更倾向于情感和社交，而男性目标的消息则更直接和果断。

**⚠️ 局限性**

限制在于分析框架主要基于西方理论传统，可能无法捕捉非西方文化背景下有效的劝说策略。此外，研究仅覆盖了两种性别，未能探讨其他性别或因素的影响。

---

## 230. Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding

**arXiv ID:** 2601.05724 | [PDF](https://arxiv.org/pdf/2601.05724v1)

**作者:** Yuxuan Zhou `[一作]` (Independent Researcher), Zhi-Qi Cheng `[通讯]` (University of Washington)

**通讯引用:** 1713 | [OpenAlex ID](https://openalex.org/A5058898461)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种层次化的投机式解码方法（Hierarchical Speculative Decoding, HSD），实现无损验证并显著提升可接受的token数，解决了传统token‑级验证的可验证性瓶颈；

**💡 创新点**

采用分层分支重采样策略，构建可在可访问分支内一次性完成重采样的“Capped Branch Resampling”，从而在保持分布完整性的同时避免了全局联合概率计算的不可行性；

**🔧 技术方法**

核心技术包括分层分支散度（Branch Divergence）与其加权差异、最大前缀比例索引、截断比例（Capped Prefix Ratio）以及基于这些指标的接受概率和重采样概率公式；

**📊 数据集**

在GPTQ‑量化的Qwen2.5系列（0.5B草稿、14B/32B/72B目标）上评估，涉及GSM8K（数学题）、HumanEval（代码生成）、CNN/DailyMail（摘要）等任务；

**📈 对比分析**

与Token‑wise和Block‑wise无损验证方法对比，使用Block Efficiency（token/step）和Decoding Speed（token/s）衡量；在各模型规模和任务上，HSD平均提升约6.2% 的 Block Efficiency 和 6.7% 的 Decoding Speed；在多草稿设置下亦能保持 5–12% 的加速；

**⚠️ 局限性**

对极端高温度或极短/极长草稿的鲁棒性仍待进一步验证，且在某些大规模模型或硬件受限环境下，层次重采样的额外计算成本可能仍是瓶颈；

---

## 231. 2BRobust -- Overcoming TCP BBR Performance Degradation in Virtual Machines under CPU Contention

**arXiv ID:** 2601.05665 | [PDF](https://arxiv.org/pdf/2601.05665v1)

**作者:** Kathrin Elmenhorst `[一作]` (Osnabrück University), Nils Aschenbruck `[通讯]` (Osnabrück University)

**通讯引用:** 2409 | [OpenAlex ID](https://openalex.org/A5031130703)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了 BBR 在虚拟机 CPU 争用环境下吞吐量急剧下降的现象，并提出基于 Inflight Deficit 指标的最小化补丁来提升性能。

**💡 创新点**

创新点在于：①构建可控的 deadline scheduling 测试平台，可精细化评估 BBR 的吞吐；②发现 CPU 争用导致的“Inflight Deficit”状态；③设计仅增大 pacing gain 的补丁，避免修改 cwnd 造成公平性问题。

**🔧 技术方法**

技术手段包括：Linux deadline scheduling、tc netem、iperf3 测试、bbrstat 统计、TSO 及 pacing、BDP 估计以及内核源码 patch。

**📊 数据集**

实验数据集为：30 次 20 秒 iperf3 连接吞吐率样本，覆盖 RTT 10–40 ms、带宽 10–1000 Mbps、时隙 1–20 ms、CPU share 10–70%，并与 Cubic 及不同 BBR 版本（v1‑v3）对比。

**📈 对比分析**

比较方法：在相同网络与调度条件下对比 BBR 与 Cubic 的吞吐；实验结果显示在 10–25% CPU share 时 BBR 仅达 10–20 Mbps；补丁后在 25%（1 ms）或 40%（10 ms）CPU share 下即可恢复至接近满带宽，提升幅度约 2–3 倍。

**⚠️ 局限性**

局限性：①实验仅在 deadline scheduling 下模拟，未覆盖真实云 hypervisor 的动态调度；②只测试单一 VM 负载，未考虑多应用竞争；③未验证对公平性与跨协议（如 QUIC）的全局影响；④补丁针对 BBR 内核实现，迁移到其他实现可能需要进一步验证。

---

## 232. Adaptive Disentangled Representation Learning for Incomplete Multi-View Multi-Label Classification

**arXiv ID:** 2601.05785 | [PDF](https://arxiv.org/pdf/2601.05785v1)

**作者:** Quanjiang Li `[一作]` (National University of Defense Technology), Chenping Hou `[通讯]` (National University of Defense Technology)

**通讯引用:** 3072 | [OpenAlex ID](https://openalex.org/A5091529433)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自适应解耦表示学习框架（ADRL），用于同时处理视图缺失和标签缺失的多视图多标签分类问题。

**💡 创新点**

创新点包括基于注意力的缺失视图重建、随机片段遮蔽增强鲁棒性、互信息约束的共享/私有特征解耦、图注意网络构建标签原型以及利用伪标签的自适应视图融合。

**🔧 技术方法**

使用了图注意力网络、互信息下界、对比学习、随机遮蔽、双通道编码器及伪标签生成等技术。

**📊 数据集**

在六个公开多视图多标签数据集（Corel5k、Pascal07、ESPGame、IAPRTC12、Mirflickr、Object）以及真实NBA球员数据集上进行实验。

**📈 对比分析**

与九种主流方法（AIMNet、DICNet、DIMC、DM2L、iMVWL、LMVCAT、LVSL、MTD、SIP）对比，ADRL在所有指标上均名列前茅，尤其在高缺失率下提升约15% AP/AUC。

**⚠️ 局限性**

仍受限于对超参数的调优敏感、需要至少一个完整视图样本、计算量相对较大以及在极端稀疏标签/视图场景下可能仍存在性能下降。

---

## 233. StriderSPD: Structure-Guided Joint Representation Learning for Binary Security Patch Detection

**arXiv ID:** 2601.05772 | [PDF](https://arxiv.org/pdf/2601.05772v1)

**作者:** Qingyuan Li `[一作]` (Nanjing University), Bin Luo `[通讯]` (Nanjing University)

**通讯引用:** 10833 | [OpenAlex ID](https://openalex.org/A5100372676)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结构引导的联合表示框架，用于在二进制补丁中同时捕捉结构和语义信息以实现安全补丁检测。

**💡 创新点**

创新点在于将图神经网络分支与大语言模型（LLM）分支通过三重适配器与门控机制融合，并采用两阶段训练解决参数规模不均衡问题，从而显著提升检测准确率。

**🔧 技术方法**

采用了CFG图、GGCN、UniXcoder节点编码、适配器（Query/Key/Value）、门控网络、交叉注意力、LLM（Qwen3‑8B为主）以及LoRA微调。

**📊 数据集**

使用了自建的跨项目跨域二进制安全补丁基准（1,720 条记录，涵盖 ImageMagick、TcpDump、Qemu、Radare2、Slurm 等五大领域），并在此基准上进行评估。

**📈 对比分析**

与三类基线（LLM、图网络、源代码适配）对比，平均准确率提升 12.66%，F1 提升 8.19%，误报率降低 38.57%；在不同优化等级、不同漏洞类型以及不同 LLM 家族上均保持领先。

**⚠️ 局限性**

局限性包括对某些 LLM 家族（如 Yi 系列）适配效果不佳、仅针对 C/C++，对多语言的泛化待验证，以及对反编译器质量的依赖。

---

## 234. Do Sparse Autoencoders Identify Reasoning Features in Language Models?

**arXiv ID:** 2601.05679 | [PDF](https://arxiv.org/pdf/2601.05679v1)

**作者:** George Ma `[一作]` (University of California Berkeley), Somayeh Sojoudi `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对稀疏自编码器（SAE）中被认为是推理特征的方向进行系统的消真实验，检验其是否真正反映推理计算而非表面语言相关性。

**💡 创新点**

提出了结合因果词注入和 LLM 引导对抗样本的消真框架，强调在推理特征评估中必须消除词汇和上下文的表面共因；并在多模型、多层次、多数据集上验证此框架。

**🔧 技术方法**

使用稀疏自编码器（GemmaScope、DeepSeek 训练得到的 SAE）、因果词注入实验、Gemini 3 Pro 生成对抗样本、以及对目标模型的激活分析与特征驱动。

**📊 数据集**

采用 1,000 题数学 CoT 数据集（s1K‑1.1）、6,000 个多领域推理问答集（General Inquiry CoT）以及非推理文本（Pile 子集）进行实验。

**📈 对比分析**

相较于仅靠对比激活差异的方法，消真实验显示 59%–94% 的候选特征被词汇触发，剩余的 Context‑Dependent 特征在 LLM 对抗测试中也未能满足真推理标准；因此，未发现任何能稳健捕捉推理的特征，性能提升极为有限甚至略有下降。

**⚠️ 局限性**

局限在于：仅评估特定 SAE 变体和公开权重模型；定义的“真推理特征”过于严格，可能忽略分布式或间接参与推理的表示；LLM 引导的反例生成不保证搜索完整；并且仅做了有限的驱动实验，未能完整揭示特征的因果作用。

---

## 235. On the Complexity of Electromagnetic Far-Field Modeling

**arXiv ID:** 2601.05674 | [PDF](https://arxiv.org/pdf/2601.05674v1)

**作者:** Torben Kölle `[一作]` (ETH Zurich), Christoph Studer `[通讯]` (ETH Zurich)

**通讯引用:** 10610 | [OpenAlex ID](https://openalex.org/A5083617223)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个基于麦克斯韦方程的数学框架，用于分析一般天线架构的电磁远场建模的复杂性，证明了在物理上有意义的假设下，这些天线架构的复杂性是有限的，可以用有限秩算子和有限参数进行建模。

**💡 创新点**

创新点在于构建了一个数学上严谨的框架，证明了广泛的辐射结构的电磁远场交互可以通过有限秩算子序列进行任意精度的近似，并且一旦秩超过与天线架构和分析频率相关的有效带宽，近似误差呈超指数衰减。

**🔧 技术方法**

使用了线性系统理论和向量球谐函数的方法来构建有限秩算子序列，并分析其近似性质。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了广泛的辐射结构和无线系统的电磁远场交互。

**📈 对比分析**

通过与现有文献的比较，本文的方法在处理复杂性和近似误差方面表现出更强的普适性和准确性，尤其是在不限制于小型或大型系统的情况下。

**⚠️ 局限性**

限制在于该研究主要集中在有限秩算子的构建和近似性质上，可能在实际应用中需要进一步的实验验证和实际系统的适应性分析。

---

## 236. Local generation of languages: the monotonic binary sequences

**arXiv ID:** 2601.05673 | [PDF](https://arxiv.org/pdf/2601.05673v1)

**作者:** Mathieu Hoyrup `[一作]` `[通讯]` (University of Lorraine), Mathieu Hoyrup (University of Lorraine)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文研究在分布式环境下局部生成二进制单调序列语言（即所有非递增或非递减序列）的通信结构。作者通过构造和分析模拟产生过程的simplicial complex（通信复形）来决定单元间需要多少信息交换才能生成目标语言，并给出了对该语言所有最小通信复形的结构性描述。

**💡 创新点**

创新点包括：
• 将局部生成问题正式化为通过simplicial complex来捕捉通信；
• 对单调序列语言的最小生成复形进行分级分类，首次给出三类基本复形族（K₂、K₅、K₈）并证明它们通过顶点插入可以得到所有更大 n 的最小生成复形；
• 证明所有最小生成复形的最大单纯形必为“区间”，并确定区间长度的极限比例 μ(n)≈3n/4；
• 设计并利用推理系统与自动工具来寻找矛盾，从而判定复形是否可生成该语言。

**🔧 技术方法**

核心技术：
• 组合拓扑与simplicial complex理论，用以描述通信结构；
• 自动生成推理系统（类似证明助手）用于构造冲突，验证生成性；
• 递归插入/删除顶点的操作来构造新复形并分析其最小性；
• 证明技术包括区间覆盖、对称性（D₂n 迪氏群）与不可分性论证。

**📊 数据集**

数据集：本研究没有依赖外部数据集，而是以理论语言 Lₙ={0^k1^{n-k} | 0≤k<n}∪{1^k0^{n-k}} 作为实验对象，探讨其在不同 n 下的最小生成复形。

**📈 对比分析**

方法比较：作者通过理论证明和有限 n 的完整枚举（使用自动工具）验证各类复形的最小性，并与之前已知的简单案例（如 K₂ 家族）进行对比。性能方面的评价是基于区间长度上限（≤3n/4）与下限（≥3n/4）相匹配，表明已达到最优区间长度上限；对 n=6 的情况给出了开放式结论，说明方法在该点尚未达到完整覆盖。

**⚠️ 局限性**

局限性：
• 对 n=6 的最小生成复形是否为 K₆ 仍未定；
• 证明中依赖大量手工或自动搜索的冲突，缺乏统一、简洁的通用证明框架；
• 目前仅覆盖单调序列语言，难以直接推广到更一般的多字母或更复杂约束语言；
• 3/4 近似比例虽已得到上、下界匹配，但缺乏直观解释或更一般的结构性理论。

---

## 237. How to Analyse Interviews: A Documentary Method of Interpretation

**arXiv ID:** 2601.05871 | [PDF](https://arxiv.org/pdf/2601.05871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 238. LACIN: Linearly Arranged Complete Interconnection Networks

**arXiv ID:** 2601.05668 | [PDF](https://arxiv.org/pdf/2601.05668v1)

**作者:** Ramón Beivide `[一作]` (Universidad de Cantabria), Mateo Valero `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 13494 | [OpenAlex ID](https://openalex.org/A5020844763)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出一种线性排列的完全互连网络（LACIN），通过统一端口索引实现全互连交换机之间的链接，简化布线和路由；

**💡 创新点**

创新点在于将端口索引同一化（isoport）与线性布局相结合，形成无交叉、可按色彩分组的电缆系统，并提供基于 Xor、Circle 两种 1‑factor 分解的实现方案；

**🔧 技术方法**

利用图论中的完全图 1‑factor 分解、XOR 与 Circle 组合，以及硬件友好的最短路由算法（XOR 仅需减法、Circle 需简单比较）实现；

**📊 数据集**

未使用具体实验数据集，主要通过理论分析和仿真计算评估布线长度、路由开销与可扩展性；

**📈 对比分析**

与传统 Swap、Circle、XOR 线性互连比较，LACIN 在布线长度上与 Circle 相当（无交叉），在路由成本上与 XOR 相同或更低；

**⚠️ 局限性**

局限性包括：XOR 方案仅适用于节点数为 2ⁿ 的网络；Circle 方案虽无交叉但实现复杂；LACIN 仅针对单维线性布局，对更高维或非线性拓扑的适配仍需进一步研究。

---

## 239. Advancing credit mobility through stakeholder-informed AI design and adoption

**arXiv ID:** 2601.05666 | [PDF](https://arxiv.org/pdf/2601.05666v1)

**作者:** Yerin Kwak `[一作]` (University of California), Zachary A. Pardos `[通讯]` (University of California)

**通讯引用:** 3522 | [OpenAlex ID](https://openalex.org/A5021273980)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用人工智能辅助 SUNY 系统的课程对等评估，结合教师和行政人员的反馈改进算法，并预测潜在的课程对等关系。

**💡 创新点**

创新点在于提出共享空间对齐（SSA）方法，借助多语言伪监督技术消除机构偏差，显著提升对等召回率；同时采用人工反馈制定阈值，实现高可信度的对等预测。

**🔧 技术方法**

采用 NLP 词向量（Word2Vec、SBERT、OpenAI LLM）与学生注册序列向量（Course2vec）相结合；通过 SSA 对齐、KNN 近邻检索、AUC‑ROC 设定相似度阈值，并使用 t‑SNE 进行可视化解释。

**📊 数据集**

数据集为 SUNY 52 所机构的课程标题与描述共 120,749 门可转学分课程，包含 156,968 条已建立的对等对；以及学生注册历史信息。

**📈 对比分析**

与先前的 SBERT 基线和其他 NLP 代号模型对比，使用 recall@1 与 recall@5 评估；SSA 在所有模型上平均提升约 121% 的召回率，最终 OpenAI+SSA 召回率达到 recall@1 0.764、recall@5 0.928，显著高于先前最高 0.139。

**⚠️ 局限性**

局限性包括：需要已有对等数据来训练 SSA，难以在没有对等记录的机构使用；采纳率预测基于低风险情境的意向，未在实际审批流程中验证；阈值设定时负样本为伪负样本，可能影响阈值精度。

---

## 240. Stephanie2: Thinking, Waiting, and Making Decisions Like Humans in Step-by-Step AI Social Chat

**arXiv ID:** 2601.05657 | [PDF](https://arxiv.org/pdf/2601.05657v1)

**作者:** Hao Yang `[一作]` (Fudan University), Xinhua Zeng `[通讯]` (Fudan University)

**通讯引用:** 527 | [OpenAlex ID](https://openalex.org/A5054486930)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Stephanie2，一种在每一步可主动决定发送或等待、并根据思考与打字时间调节消息节奏的多消息即时聊天代理，并通过时间窗口双代理系统生成高质量对话数据。

**💡 创新点**

创新点在于引入主动等待机制与显式思考轨迹，结合思考+打字时间模型的消息节奏调整，以及基于时间窗口的双代理生成框架和角色识别测试。

**🔧 技术方法**

使用大型语言模型（如GPT-5.2、DeepSeek-V3、Llama3.1-8B）进行生成，采用prompt、in-context learning、决策策略π(a|m,p)与时间窗口机制，并用自动评估与人类评估指标。

**📊 数据集**

主要基于Persona-Chat数据集，经过过滤得到6,459条对话作为种子，再通过双代理生成约6,459条Stephanie2级别对话；实验中亦使用公开模型和标准评估数据。

**📈 对比分析**

通过自动评估（自然度、连贯性等）和人类评估（评分与角色识别测试）与PD、Stephanie1以及三大基线模型对比，Stephanie2在自然度、参与度、角色识别通过率等指标上均显著优于前者。

**⚠️ 局限性**

局限性包括资源受限未能在更大规模模型上实验，人工评估样本量有限，且未在真实用户场景中验证。

---

## 241. Kidney Cancer Detection Using 3D-Based Latent Diffusion Models

**arXiv ID:** 2601.05852 | [PDF](https://arxiv.org/pdf/2601.05852v1)

**作者:** Jen Dusseljee `[一作]` (Radboudumc), Alessa Hering `[通讯]` (Radboudumc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建了一种基于3D潜在扩散模型的弱监督肾癌异常检测流水线，结合DDPM、DDIM、VQ‑GAN与分类器引导。

**💡 创新点**

创新点在于首次将潜在扩散模型与分类器指导联合用于全体卷积CT的体素级重建与异常映射，突破传统二维切片限制。

**🔧 技术方法**

核心技术包括：Denoising Diffusion Probabilistic/Implicit Models、Vector‑Quantized GAN自编码器、潜在空间的扩散采样与分类器引导。

**📊 数据集**

使用Radboudumc私有的8,377张强化腹部CT（7,571研究，6,800患者）生成的伪标签数据集，含5,095左肾和5,099右肾。

**📈 对比分析**

与两种监督基线（nnU‑Net和nnDetection）在30张完全标注的CT上比较，检测准确率和Dice系数远低于监督模型；DDPM和DDIM在小于2 cm的肿瘤上召回率仅0.14，精确率低于0.05。

**⚠️ 局限性**

局限包括：重建过程中的解剖结构伪影导致高误报、分类器在伪标签上的性能有限、对大于4 cm的肿瘤几乎无检测能力、整体灵敏度与分割精度不足。

---

## 242. DexterCap: An Affordable and Automated System for Capturing Dexterous Hand-Object Manipulation

**arXiv ID:** 2601.05844 | [PDF](https://arxiv.org/pdf/2601.05844v1)

**作者:** Yutong Liang `[一作]`, Libin Liu `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了低成本、多摄像机标记式手部与物体交互捕捉系统DexterCap，并基于该系统构建并公开了高质量的DexterHand数据集，涵盖多种物体形状和复杂的在手操作；

**💡 创新点**

创新点包括：①稠密字符编码标记和三阶段检测+投票的标记识别方案，显著提升遮挡下的鲁棒性；②利用自标定的MANO手模型和Kabsch对象姿态估计，实现全自动化无人工干预的运动重建；③系统成本低（≈6000美元）且流程可扩展，支持大规模数据采集；

**🔧 技术方法**

使用技术主要包括：多摄像机灰度同步采集、U-Net热图式CornerNet检测角点、ResNet EdgeNet/BlockNet进行边缘和块识别、投票纠错、RANSAC外点剔除、MANO手模型求解（Adam优化）、Kabsch算法求解物体姿态；

**📊 数据集**

采用了自建的DexterHand数据集（包含7种基础形状和魔方，序列时长10+分钟）进行评估，并与公开数据集GRAB、ARCTIC、HUMOTO、HaMeR和GigaHands进行基准对比；

**📈 对比分析**

与商用Vicon系统和手套、以及Vision‑only方法（HaMeR、GigaHands）相比，DexterCap在MSNR（≈9.3）、jerk（≈0.76）、Diversity（≈0.97）与Coherence（≈0.68）等指标上取得竞争甚至领先的表现，且能唯一捕捉在手复杂操作；

**⚠️ 局限性**

局限性主要为：1）仍易受严重遮挡影响，遮挡时重建质量下降；2）系统仅适用于刚性或可预知形状的物体，对柔性物体或多手/工具交互支持不足；3）缺乏遮挡感知与预测机制，需进一步加入IMU或学习式姿态先验提升鲁棒性。

---

## 243. Peek2: A Regex-free implementation of pretokenizers for Byte-level BPE

**arXiv ID:** 2601.05833 | [PDF](https://arxiv.org/pdf/2601.05833v1)

**作者:** Liu Zai `[一作]` `[通讯]`, Liu Zai

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Peek2预分词器，替代正则表达式的cl100k实现，实现更快更安全的Byte-level BPE预分词。

**💡 创新点**

采用两字符预览和7×7分类表的Regex‑free算法，保持bug‑for‑bug一致性，提升1.11倍吞吐量，同时消除正则带来的复杂性与安全风险。

**🔧 技术方法**

基于Safe Rust实现的CPU算法，利用Unicode NFKC、递归/迭代预分词、查表决策和HuggingFace tokenizers框架集成。

**📊 数据集**

在LLaMa-3 tokenizer任务（llama3-offsets、llama3-encode、llama3-batch、BPE Train vocab）以及XNLI数据集上进行测试。

**📈 对比分析**

与原cl100k Regex实现在同一任务和代码基准下比较，测量吞吐量(Mbps)和耗时(ms)，Peek2在所有任务均提升约1.11×吞吐量，耗时更短。

**⚠️ 局限性**

仅适用于cl100k‑like预分词器，CPU实现；无法直接迁移到GPU；bug‑for‑bug实现仍受原正则歧义限制，部分错误无法彻底修复。

---

## 244. Decoding Workload and Agreement From EEG During Spoken Dialogue With Conversational AI

**arXiv ID:** 2601.05825 | [PDF](https://arxiv.org/pdf/2601.05825v1)

**作者:** Lucija Mihić Zidar `[一作]` (Brandenburg Technical University Cottbus-Senftenberg), Thorsten O. Zander `[通讯]` (Brandenburg Technical University Cottbus-Senftenberg)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究通过设计两种对话实验（拼写蜂和句子完成任务），评估了基于EEG的被动BCI模型（工作负荷和隐式同意）在自然语言交互中的可迁移性，并实现了对话转录、标注与EEG信号的实时对齐流程。

**💡 创新点**

创新点在于：① 将传统实验室任务中训练的工作负荷与同意分类器迁移至真实语音对话情境；② 开发了端到端的语音转录与时间对齐管线；③ 对工作负荷在不同对话轮次的动态变化进行量化分析；④ 发现同意分类器在连续应用中失去时间锁定，揭示了事件驱动与连续场景之间的界限。

**🔧 技术方法**

技术主要包括：EEG采集与预处理（down‑sample、ICA、AMICA、ICLabel）；工作负荷分类器使用fbCSP+LDA；同意分类器使用窗口均值+LDA；语音识别采用Whisper；时间对齐与事件标注利用Lab Streaming Layer与force‑align；统计分析采用OLS回归与AR(1)调整。

**📊 数据集**

数据集为两名受试者在工作负荷校准（算术任务）和同意校准（网格跳跃）中的EEG记录，以及两名受试者在拼写蜂和句子完成对话中的EEG、语音、转录与事件日志。

**📈 对比分析**

工作负荷分类在校准阶段的平均准确率约为69%（区间64–81%），显著高于随机；在对话中，工作负荷随轮次呈上升趋势，一名受试者的线性趋势显著（p<0.001），说明模型可迁移；同意分类器在对话中的连续输出未出现明显事件相关峰值，说明性能不佳。

**⚠️ 局限性**

局限性包括：样本量极小（仅4人，分两组）；对话设计与实验室任务差异大，导致同意模型失效；EEG噪声与口语声压共振可能影响信号质量；缺乏对比实验（如传统显式反馈）以评估实际优势；并未在真实对话中验证模型对AI行为的实时调整效果。

---

## 245. LLMs as Science Journalists: Supporting Early-stage Researchers in Communicating Their Science to the Public

**arXiv ID:** 2601.05821 | [PDF](https://arxiv.org/pdf/2601.05821v1)

**作者:** Milad Alshomary `[一作]` (Columbia University), Smaranda Muresan `[通讯]` (Columbia University)

**通讯引用:** 2919 | [OpenAlex ID](https://openalex.org/A5043262011)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并评估了专门扮演科学记者角色的LLM，帮助早期研究者向公众有效沟通其论文。

**💡 创新点**

提出从论文与新闻稿自动合成科学对话数据并通过监督微调与偏好学习双阶段训练的框架，使LLM能主动提问社会影响、科学背景和澄清性问题。

**🔧 技术方法**

使用监督微调（SFT）、基于偏好学习的DPO、LLM评估者、Prompt设计及自动对话合成等技术。

**📊 数据集**

基于约41k篇论文与新闻稿对，过滤后得到18k对作为训练集，并合成约80k条记者发言及19k条偏好数据。

**📈 对比分析**

通过自动对话模拟评估三类问题比例、冗余与跟进，并与简易提示及封闭源模型对比；DPO-Llama/Qwen在问题多样性与跟进上最高，平均Harmonic为0.35/0.41；用户研究（9名博士生）显示LLM记者获得77%偏好、平均分高于基线。

**⚠️ 局限性**

局限在于对话仅限10轮、仅关注三类问题、缺乏多样化受众与更长对话评估、科学背景提问表现仍不足、样本规模小等。

---

## 246. Detecting Autism Spectrum Disorder with Deep Eye Movement Features

**arXiv ID:** 2601.05812 | [PDF](https://arxiv.org/pdf/2601.05812v1)

**作者:** Zhanpei Huang `[一作]` (Guangdong University of Technology), Yiqun Zhang `[通讯]` (Guangdong University of Technology)

**通讯引用:** 714 | [OpenAlex ID](https://openalex.org/A5100329232)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用眼动数据构建 ASD 与 TD 的区分模型，提出 DSTS 框架

**💡 创新点**

针对眼动数据的离散短时依赖，提出基于 CNN 的 DSTS 并加入类别感知表征与类别不平衡处理

**🔧 技术方法**

一维 CNN、Multi‑Similarity 损失、加权交叉熵、Adam 优化

**📊 数据集**

深圳产妇儿童医院收集的 8 个眼动实验数据集（Speaking、Walking1、Walking2、Helicopter、Baby、Tablet、Attention、Sad）

**📈 对比分析**

与 Informer、CrossFormer、MPTSNet、EmMixformer、Detach‑Rocket 等方法对比，DSTS 在绝大多数数据集上实现最高 ACC 与 F1，且参数量与计算时间最小

**⚠️ 局限性**

仍未解释 Transformer 低效原因，且缺乏跨范式自适应学习的通用框架

---

## 247. What do the metrics mean? A critical analysis of the use of Automated Evaluation Metrics in Interpreting

**arXiv ID:** 2601.05864 | [PDF](https://arxiv.org/pdf/2601.05864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 248. Modular Autonomy with Conversational Interaction: An LLM-driven Framework for Decision Making in Autonomous Driving

**arXiv ID:** 2601.05806 | [PDF](https://arxiv.org/pdf/2601.05806v1)

**作者:** Marvin Seegert `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**通讯引用:** 1859 | [OpenAlex ID](https://openalex.org/A5063677428)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于大型语言模型（LLM）的交互框架，使乘客能够通过自然语言向Autoware模块化自动驾驶系统发出指令并查询信息。

**💡 创新点**

创新点在于：①构建了五大交互类别的应用级DSL；②引入了安全验证与接口节点，确保LLM指令不直接影响关键控制；③使用两阶段LLM生成流程，让乘客反馈与实际执行结果保持一致。

**🔧 技术方法**

采用的技术包括Gemini 2.5 Flash‑Lite LLM（云端）、ROS 2接口节点、Autoware规划与决策模块、语音识别/合成（STT/TTS）以及自定义DSL解析与验证层。

**📊 数据集**

主要使用的数据集是人工构造的200条人类指令与对应DSL命令对，另外在仿真中重复执行30场包含五类交互的完整场景进行定量评估。

**📈 对比分析**

与Song et al.的方案相比，本文的基线准确率为97.0%（高于87.0%），平均翻译时延1.723 s，仿真任务成功率96.7%，并且命令执行延迟在毫秒级，显示出更快、更稳定的性能。

**⚠️ 局限性**

局限性包括：依赖云端LLM导致网络延迟和单点失效；LLM未针对驾驶域微调，处理歧义的鲁棒性未充分验证；实验仅覆盖有限场景，需在更复杂交通环境中进一步测试。

---

## 249. InsSo3D: Inertial Navigation System and 3D Sonar SLAM for turbid environment inspection

**arXiv ID:** 2601.05805 | [PDF](https://arxiv.org/pdf/2601.05805v1)

**作者:** Simon Archieri `[一作]` (Heriot-Watt University), Yvan Petillot `[通讯]` (Heriot-Watt University)

**通讯引用:** 6792 | [OpenAlex ID](https://openalex.org/A5034972272)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了基于3D声纳与INS的实时大规模3D SLAM框架InsSo3D。

**💡 创新点**

创新点包括：利用3D声纳消除二维声纳的高程歧义；结合CFEAR+GICP进行点云配准；前端子图TSDF生成与后端因子图优化相结合，实现循环闭环与全局一致。

**🔧 技术方法**

使用技术有：3D声纳深度图、DVL+AHRS+压力传感器INS、CFEAR稀疏表面点与GICP配准、TSDF空间切割、因子图前端后端优化、Loop closure检测、Marching Cubes网格化。

**📊 数据集**

实验数据集：WaterLinked Sonar3D‑15声纳与BlueROV2、Stereo相机、Nortek Nucleus 1000 DVL；在水槽（配备Qualisys运动捕捉）和水填充采石场（使用Colmap重建的光学参考）进行测试。

**📈 对比分析**

对比方法：与光学SfM（Colmap）及Qualisys轨迹对齐，计算APE_RMS、APE_STD；InsSo3D轨迹平均误差<21cm，地图误差9cm，明显优于单纯里程计，保持高精度。

**⚠️ 局限性**

局限性：GICP在几何对称或特征缺失环境中鲁棒性不足；声纳多路径噪声影响配准质量；未与视觉数据融合，需进一步提升稳健性。

---

## 250. From Off-Policy to On-Policy: Enhancing GUI Agents via Bi-level Expert-to-Policy Assimilation

**arXiv ID:** 2601.05787 | [PDF](https://arxiv.org/pdf/2601.05787v1)

**作者:** Zezhou Wang `[一作]` (Nanjing University), Yan Lu `[通讯]` (Microsoft Research Asia)

**通讯引用:** 19678 | [OpenAlex ID](https://openalex.org/A5035278528)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文提出一种双层专家-策略同化（BEPA）方法，用于在有限的专家轨迹下提升端到端GUI代理的强化学习性能。

**💡 创新点**

创新点在于将离散的框架专家轨迹先通过自滚（LEVEL‑1）转化为与策略可达的轨迹，再通过动态缓存（LEVEL‑2）在训练中按需注入，实现专家指导与策略分布的自适应对齐。

**🔧 技术方法**

主要技术包括自滚式轨迹重演、基于GRPO的强化学习、离线专家轨迹转换、缓存更新策略以及条件轨迹替换。

**📊 数据集**

实验使用的主要数据集是OSWorld‑Verified、MMBench‑GUI和Online‑Mind2Web，并从Agent S2等框架专家收集约115条成功轨迹。

**📈 对比分析**

与传统SFT、RL+SFT、LUFFY、Trace Replacement等方法相比，BEPA在OSWorld‑Verified上将UITARS1.5‑7B的成功率从22.87%提升至32.13%，在保留的测试集上亦从5.74%提升至10.30%，并在MMBench‑GUI、Online‑Mind2Web等跨域任务中保持一致性提升。

**⚠️ 局限性**

局限性包括对GUI类任务的专注，实验仅验证了单一基线与专家来源，对移动平台、非GUI环境或多模态交互的适用性尚未探索；同时，BEPA需额外维护自滚轨迹和缓存，增加实现复杂度。

---

## 251. Rigorous Implications of the Low-Degree Heuristic

**arXiv ID:** 2601.05850 | [PDF](https://arxiv.org/pdf/2601.05850v1)

**作者:** Jun-Ting Hsieh `[一作]` (Massachusetts Institute of Technology), Stefan Tiegel `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文通过将低阶似然比（LDLR）上界转换为对具体算法的严格下界，证明了在加入噪声且满足对称性的情形下，任何基于低阶多项式、对称多项式或常数规模子图计数的判别方法在平均-案例中无法区分种子分布与无信号分布。

**💡 创新点**

创新点主要有：①首次把LDLR的“低阶不可区分性”证明为对低阶多项式判别器的真正不可行性；②在噪声模型下利用对称性把高维问题降维为一维；③结合傅里叶分析、超正则化、局部中心极限定理和图矩阵矩估计，给出了一系列严格的 TV 距离和 L∞ 上界，扩展了低阶方法的理论边界。

**🔧 技术方法**

采用的技术包括：
- 低阶似然比（LDLR）与 χ² 散度的等价分析；
- 对称性投影（将函数压缩为取决于哈密顿重量或 Hermite 多项式向量）；
- 傅里叶变换与频率匹配（利用矩匹配得到低频误差控制）；
- 超正则化（hypercontractivity）与高阶矩上界；
- 局部中心极限定理（LLT）与逆傅里叶重构；
- 图矩阵矩估计与子图计数的高阶矩控制；
- 经验上对噪声的“有限独立性加噪声”分析。

**📊 数据集**

本文属于理论研究，不涉及具体数据集，所有结论均基于数学模型（{0,1}^n 的伯努利分布、标准高斯分布以及其噪声化版本）。

**📈 对比分析**

与传统的经验算法对比，本工作不提供上界或算法实现，而是给出理论证明：在给定的低阶多项式/对称多项式/子图统计限制下，任何算法都在统计上无法超过随机猜测；即下界与已知的统计阈值吻合，表明存在计算-统计差距。

**⚠️ 局限性**

局限性包括：
- 仅对低阶（O(log n / loglog n)）多项式或常数规模子图计数的判别器给出下界；
- 对于高阶多项式或大规模子图统计未能给出结论；
- 需要对称性和噪声的假设，非对称或无噪声情况仍不确定；
- 只给出可辨识性下界，未给出相应的上界或算法；
- 结果高度依赖理论证明的假设（如低阶似然比近似为零），在实际模型中可能不严格满足。

---

## 252. Semantic NLP Pipelines for Interoperable Patient Digital Twins from Unstructured EHRs

**arXiv ID:** 2601.05847 | [PDF](https://arxiv.org/pdf/2601.05847v1)

**作者:** Rafael Brens `[一作]` (Binghamton University), Zhaohan Xi `[通讯]` (Binghamton University)

**通讯引用:** 111 | [OpenAlex ID](https://openalex.org/A5026309535)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了基于语义NLP的端到端管线，将未结构化的 EHR 文本自动转换为符合 FHIR 规范的患者数字孪生。

**💡 创新点**

创新点在于将 NER、概念规范化、关系抽取三大模块与 FHIR schema 紧密对齐，实现无人工干预的数字孪生生成，并在 MIMIC‑IV Demo 上显著提升语义完整性和互操作性。

**🔧 技术方法**

采用 Transformer 预训练模型（ClinicalBERT/Cased BERT）、UMLS/SNOMED‑CT/ICD‑10/LOINC/RxNorm 归一化、上下文关系抽取以及 FHIR v4.0.1 资源组装与验证等技术。

**📊 数据集**

使用 MIMIC‑IV Clinical Database Demo（100 名患者）生成的模拟自由文本与对应的 MIMIC‑IV‑on‑FHIR 参考映射作为实验数据集。

**📈 对比分析**

与正则表达式规则抽取和裸映射基线对比，评估 NER/RE F1、语义完整度和互操作性；结果显示 NER F1 0.89、RE F1 0.81、语义完整度 91%、互操作性 0.88，较基线分别提升 17、26、29、27 分。

**⚠️ 局限性**

仅在 100 人模拟数据上验证，未涵盖真实医生笔记；缺少多中心、跨语言、跨时间序列的泛化评估；未建模多次就诊的时间依赖；仅处理英文文本。

---

## 253. GeoSurDepth: Spatial Geometry-Consistent Self-Supervised Depth Estimation for Surround-View Cameras

**arXiv ID:** 2601.05839 | [PDF](https://arxiv.org/pdf/2601.05839v1)

**作者:** Weimin Liu `[一作]` (Tsinghua University), Joshua H. Meng `[通讯]` (University of California)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5080515832)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于几何一致性的自监督周边视图深度估计框架 GeoSurDepth。

**💡 创新点**

创新点包括：①利用深度先验模型 DepthAnything V2 生成伪几何先验并通过表面法线一致性约束提升深度质量；②引入跨模态注意力机制融合 CLIP 语义特征，增强几何表示；③设计 2D‑3D 升降和多上下文视差重建的新视角合成管线；④提出自适应联合运动学习策略，动态加权各摄像头的运动信息。

**🔧 技术方法**

核心技术包括自监督 SfM 视角合成、基于几何先验的表面法线一致性损失、基于深度差异的平滑一致性损失、跨模态注意力、空间反向重投影和自适应运动学习模块。

**📊 数据集**

在 DDAD 与 nuScenes 两大周边视图数据集上进行实验，使用 6 台摄像头的 360° 场景。

**📈 对比分析**

与 FSM、VFDepth、SurroundDepth、CVCDepth、MCDP 等现有方法对比，GeoSurDepth 在绝对相对误差、平方相对误差、RMSE、RMSElog、阈值精度等指标均实现或逼近最优水平，尤其在边缘和纹理区域表现显著提升。

**⚠️ 局限性**

主要局限在于对大规模高分辨率图像的处理仍受限于模型规模；伪几何先验的准确性受 DepthAnything V2 预测误差影响；自适应运动学习虽然有效但在极端动态场景中仍可能出现运动估计偏差。

---

## 254. Descriptor: Multi-Regional Cloud Honeypot Dataset (MURHCAD)

**arXiv ID:** 2601.05813 | [PDF](https://arxiv.org/pdf/2601.05813v1)

**作者:** Enrique Feito-Casares `[一作]` (Universidad Rey Juan Carlos), José-Luis Rojo-Álvarez `[通讯]` (Universidad Rey Juan Carlos)

**通讯引用:** 5202 | [OpenAlex ID](https://openalex.org/A5051286582)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在Microsoft Azure上部署多地域、三种honeypot（Cowrie、Dionaea、SentryPeer）构建的72小时高时间分辨率攻击日志数据集（132,425条事件），并完成预处理、可视化与描述性统计。

**💡 创新点**

①首次将全球四地区的honeypot同步捕获、聚合并提供统一时间戳与地理、ASN标签的完整数据；②通过三种平台捕获多协议（SIP、Telnet、SMB）攻击，揭示平台偏好与空间热点；③为后续时空攻击建模与多源异常检测提供高质量、可复现的数据基础。

**🔧 技术方法**

使用Azure虚拟机、T‑Pot社区版部署多honeypot、IP‑ASN查询、地理位置服务、Python Jupyter 预处理脚本（JSON 解析、特征抽取、缺失处理）以及可视化工具（matplotlib/plotly）进行数据收集与加工。

**📊 数据集**

本研究自身产生的 Multi‑Regional Cloud Honeynet Dataset（含原始 JSON 与预处理后的 CSV），覆盖 4 个地理位置、3 种honeypot、3 天（6/9–6/11/2025）共 132,425 条攻击事件。

**📈 对比分析**

论文未开展模型对比实验；提供描述性统计与热图、时间序列示例，展示攻击峰值、协议占比和地理热点。作者建议后续可利用该数据进行异常检测、时间序列预测或协议滥用分析的基准实验。

**⚠️ 局限性**

①维护窗口导致的时序缺口；②不同honeypot本身的捕获偏差，影响协议与源国分布；③攻击来源高度集中（top 1 % IP 产生 15 % 事件），数据分布偏斜；④仅覆盖三天，未能体现长期趋势；⑤未提供完整的攻击类型标签，仅有协议与平台信息。

---

## 255. Simplify-This: A Comparative Analysis of Prompt-Based and Fine-Tuned LLMs

**arXiv ID:** 2601.05794 | [PDF](https://arxiv.org/pdf/2601.05794v1)

**作者:** Eilam Cohen `[一作]` (Tel Aviv University), Omri Loewenbach `[通讯]` (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了大语言模型在文本简化任务中细调（fine‑tuning）与提示工程（prompt engineering）的性能差异，并构建、清洗了WikiLarge‑Clean数据集，公开了模型检查点与提示模板。

**💡 创新点**

首次在同一实验框架下跨多数据集、跨多种seq2seq LLM，对比细调与提示的优劣，并将清洗数据、代码、模型与提示公开，形成可复现的研究资源。

**🔧 技术方法**

使用Encoder–Decoder seq2seq LLM（BART、T5、Flan‑T5、Pegasus、ProphetNet），在WikiLarge‑Clean上进行全参数细调；提示工程采用10种模板；评估自动指标SARI、FKGL、BERTScore、LENS，并通过人工偏好对比。

**📊 数据集**

评估集包括ASSET、Med‑EASi、OneStopEnglish；细调训练集为清洗后的WikiLarge‑Clean（约12.4万句对）。

**📈 对比分析**

对每个模型构建FT、P‑SARI、P‑LENS三种配置，计算指标并进行人工对比；结果显示细调在SARI（结构简化）上持续领先，提示在LENS/BERTScore（语义相似）上有优势但复制率高；人工评估更偏好细调输出。

**⚠️ 局限性**

实验仅涵盖英语数据集，模型规模≤1B，未验证更大规模LLM或decoder‑only架构；缺乏低资源语言、噪声或多模态文本的评估，提示效果易受模板与领域影响。

---

## 256. Goal Force: Teaching Video Models To Accomplish Physics-Conditioned Goals

**arXiv ID:** 2601.05848 | [PDF](https://arxiv.org/pdf/2601.05848v1)

**作者:** Nate Gillman `[一作]` (Brown University), Chen Sun `[通讯]` (Brown University)

**通讯引用:** 13773 | [OpenAlex ID](https://openalex.org/A5100722234)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Goal Force 框架，让用户通过指定目标力向量来引导视频生成，从而实现对物理因果链的逆向规划；

**💡 创新点**

创新点在于将目标力作为条件输入，训练视频生成模型成为隐式神经物理模拟器，能够在无物理引擎支持下规划实现目标的前置动作；

**🔧 技术方法**

采用多通道 ControlNet 结构对 Wan2.2 大规模扩散模型进行微调，以三通道物理控制信号（直接力、目标力、质量）作为条件；

**📊 数据集**

使用 Blender 与 PhysDreamer 合成的简单物理原语数据集（多米诺、滚动球、植物摆动），并在真实世界场景（工具使用、人机交互等）进行零样本测试；

**📈 对比分析**

与仅文本条件的基线（Zero-shot 与 Fine-tuned）以及先前的 Force Prompting/PhysGen 等方法比较，Goal Force 在目标力符合度上显著提升，且在视觉质量与运动真实度上无显著下降；

**⚠️ 局限性**

局限性包括对目标力向量的依赖，无法处理高度抽象或多模态目标，且在极端物理环境或高度复杂动力学中可能出现规划失误或模式崩溃。

---

## 257. Continual-learning for Modelling Low-Resource Languages from Large Language Models

**arXiv ID:** 2601.05874 | [PDF](https://arxiv.org/pdf/2601.05874v1)

**作者:** Santosh Srinath K `[一作]` (Birla Institute of Technology and Sciences), Abhijit Das `[通讯]` (Birla Institute of Technology and Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于适配器的持续学习框架，将大型语言模型迁移到低资源语言，并通过 POS 引导的代码切换提升跨语言记忆保持。

**💡 创新点**

创新点在于：①引入共享 Replay Adapter 与语言专属适配器分离，显著降低参数量同时保持知识；②结合 POS 代码切换生成对抗式重放数据，既保留句法结构又增强记忆；③在多模态任务（VQA、问答）和单模态任务（意图分类）上统一验证。

**🔧 技术方法**

技术包括：适配器微调（MAD‑X）、POS 代码切换子程序、经验重放、冻结主干、层级 probing 等。

**📊 数据集**

使用的公开数据集有 MTOP（意图分类）、PAXQA（多语问答）、xGQA（多模态问答）以及 NLLB、IndicTrans2 等自动翻译工具生成的低资源语言样本。

**📈 对比分析**

与无重放、随机代码切换、LWF、EWC、联合训练等基线对比，POS 代码切换+Replay Adapter 在所有任务和语言序列上均取得最高平均准确率，提升幅度从约 2% 到 10% 左右，且参数占用仅 +2%。

**⚠️ 局限性**

局限性包括：依赖于高质量 POS 标注与双语词典；对极低资源语言翻译质量可能受限；仅在适配器规模较小的 LLM 上验证，未探讨更大模型或其他 PEFT 方法的效果。

---

## 258. Intelligent Singularity Avoidance in UR10 Robotic Arm Path Planning Using Hybrid Fuzzy Logic and Reinforcement Learning

**arXiv ID:** 2601.05836 | [PDF](https://arxiv.org/pdf/2601.05836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 259. CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning

**arXiv ID:** 2601.05858 | [PDF](https://arxiv.org/pdf/2601.05858v1)

**作者:** Alexandra Dragomir `[一作]` (Bitdefender), Radu Tudor Ionescu `[通讯]` (University of Bucharest)

**通讯引用:** 8537 | [OpenAlex ID](https://openalex.org/A5081017623)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在零样本多语翻译中，提出了一种基于易到难的课程学习策略CLewR，并在每个epoch重新排序训练样本以缓解灾难性遗忘；

**💡 创新点**

创新点在于通过在训练期间多次重启易到难的课程学习并结合自适应惩罚的ARPO-z'提升了偏好优化的效果；

**🔧 技术方法**

采用了偏好优化方法（DPO、CPO、ARPO）与课程学习相结合，使用LoRA微调、BLEU、COMET等翻译质量评估指标；

**📊 数据集**

使用Flores‑200和X‑ALMA偏好数据集，在罗曼语族六种语言（以及GemmaX2的三种语言）上进行实验；

**📈 对比分析**

与CurriDPO和非课程学习的偏好优化基线比较，CLewR在BLEU和COMET上均获得统计显著提升，尤其在Gemma2、Qwen2.5和Llama3.1上表现突出；

**⚠️ 局限性**

实验受限于计算资源，仅覆盖罗曼语族六种语言，未验证在其他语言组或更大模型上的效果。

---

## 260. LayerGS: Decomposition and Inpainting of Layered 3D Human Avatars via 2D Gaussian Splatting

**arXiv ID:** 2601.05853 | [PDF](https://arxiv.org/pdf/2601.05853v1)

**作者:** Yinghan Xu `[一作]` (Trinity College), John Dingliana `[通讯]` (Trinity College)

**通讯引用:** 1043 | [OpenAlex ID](https://openalex.org/A5026997782)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于2D高斯溅射和扩散模型的三阶段框架，将静态多视角人像分解为可动画的内层身体和外层服装的多层3D人体模型；

**💡 创新点**

创新点在于仅依赖多视角2D图像即可完成层级分解和缺失区域的扩散模型填补，并通过高斯溅射实现精细几何与逼真渲染；

**🔧 技术方法**

采用2D高斯溅射（2DGS）、扩散模型（Stable Diffusion V1.5）与ControlNet、OpenPose的组合，辅以线性混合皮肤化与深度/法线一致性约束；

**📊 数据集**

使用4D-Dress和Thuman2.0两个真实人体数据集进行实验与评测；

**📈 对比分析**

与GALA和VTON360对比，本文在SSIM、PSNR、LPIPS、CLIP、ImageReward和DINO相似度等指标上均优于对手，特别在服装细节与缺失区域重建上表现更好；

**⚠️ 局限性**

局限性包括仅适用于静态多视角图像、对松散服装或自交服装效果不佳、以及在极端姿态下可能出现穿插伪影。

---

## 261. A New Family of Poisson Non-negative Matrix Factorization Methods Using the Shifted Log Link

**arXiv ID:** 2601.05845 | [PDF](https://arxiv.org/pdf/2601.05845v1)

**作者:** Eric Weine `[一作]` (Massachusetts Institute of Technology), Matthew Stephens `[通讯]` (University of Chicago)

**通讯引用:** 112375 | [OpenAlex ID](https://openalex.org/A5036504609)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了使用shifted‑log链接函数的Poisson NMF模型，提供了最大似然拟合算法和针对稀疏数据的近似求解方法，并在生物学与文本数据上进行了实证评估。

**💡 创新点**

创新点包括：①首次将非恒等链接函数引入Poisson NMF，构造可在加性与乘性组合之间切换的模型；②设计了可扩展至稀疏大规模数据的二阶近似似然，显著降低计算复杂度；③使用块坐标上升框架将模型拆解为若干非负Poisson回归问题，便于并行实现。

**🔧 技术方法**

采用Poisson GLM/GBM框架、shifted‑log链接、二次Taylor/切比雪夫近似、块坐标上升、并行非负回归求解器、重标定与初始化策略等技术。

**📊 数据集**

实验数据包括：MCF‑7 细胞株的bulk RNA‑seq（41×16,733计数矩阵）；小鼠胰腺单细胞RNA‑seq（7,606×18,195计数矩阵，≈82% 0）；BBC新闻文本语料库（2,127×5,861 文档‑词矩阵，≈98% 0）。同时使用模拟数据检验近似似然的准确性。

**📈 对比分析**

通过在不同 c（包括 ∞ 对应标准 Poisson NMF）下拟合模型，并与 GLM‑PCA 进行比较。评估指标包括：对数似然、运行时间、因子稀疏度、因子相关性。结果表明：①近似似然在稀疏数据上与精确似然几乎无差异；②小 c 值产生更模块化、相关性更低的因子，适合解释性；③大 c 值产生更聚类化、相关性更高但稀疏的因子；③计算时间在大规模稀疏数据下可缩至 O(ωK) 级别，显著快于传统 Poisson NMF。

**⚠️ 局限性**

局限性包括：①模型参数 c 的选择缺乏理论准则，需经验调参；②对极小 c 的近似似然在某些场景下可能失效；③缺乏稀疏正则化，导致小 c 产生的因子稀疏度下降；④非负约束导致部分可解释性受限；⑤对非稀疏或高度密集的数据仍保持 O(nmK) 计算复杂度；⑥尚未提供对多项式分布等更一般计数模型的直接适配。

---

## 262. A Dual Pipeline Machine Learning Framework for Automated Multi Class Sleep Disorder Screening Using Hybrid Resampling and Ensemble Learning

**arXiv ID:** 2601.05814 | [PDF](https://arxiv.org/pdf/2601.05814v1)

**作者:** Md Sultanul Islam Ovi `[一作]` (George Mason University), Syed Sabbir Hasan `[通讯]` (Shahjalal University of Science and Technology)

**通讯引用:** 73 | [OpenAlex ID](https://openalex.org/A5067209701)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了双管道机器学习框架，用于多类别睡眠障碍自动筛查

**💡 创新点**

创新点在于将统计线性工程与包装式非线性工程并行处理，并结合SMOTETomek混合重采样和Wilcoxon检验实现统计稳健性

**🔧 技术方法**

采用Mutual Information、LDA、Boruta、Autoencoder、SMOTETomek、Extra Trees、KNN等技术

**📊 数据集**

使用公开的Sleep Health & Lifestyle数据集（374条样本，三类睡眠障碍）

**📈 对比分析**

通过与多种基线和最新研究对比，获得98.67%的准确率，显著优于之前的96.88%等基线，并在交叉验证中达到统计显著提升，推理时间低于400毫秒

**⚠️ 局限性**

局限在样本量小、主要基于自报生活方式数据，未在多中心大型数据集上验证，模型对不同人群的公平性仍需进一步评估

---

## 263. EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis

**arXiv ID:** 2601.05808 | [PDF](https://arxiv.org/pdf/2601.05808v1)

**作者:** Xiaoshuai Song `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 24297 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 EnvScaler 框架，实现自动化、可扩展的工具交互式环境与情景合成，用于训练 LLM 代理。

**💡 创新点**

通过 SkelBuilder 进行主题挖掘、可执行模型构造及双代理评估，实现无需人工或真实系统先验即可批量生成高质量可运行环境；ScenGenerator 生成初始状态、挑战任务和基于规则的轨迹验证，支持多轮多工具交互的训练与评估。

**🔧 技术方法**

基于 LLM 的文本生成与推理、程序化合成（自动生成 Python 环境代码）、双代理评估（测试代理与检查代理）、基于状态检查的奖励函数、SFT 与 RL（Reinforce++）训练、POMDP 建模。

**📊 数据集**

使用 API‑Bank 与 ToolACE 进行主题挖掘；生成 191 个合成环境、约 7K 场景；实验中对比 BFCL‑MT、Tau‑Bench、ACEBench‑Agent 三大多轮工具使用基准。

**📈 对比分析**

与基线模型（GPT‑4.1、Qwen3 系列原版）对比；SFT+RL 训练后，EnvScaler 在三大基准上平均提升 4–12 分（如 Qwen3‑8B 在 BFCL‑MT 提升 4.88 分，Tau‑Bench 3.46 分），表现随环境数量、交互模式和模型规模提升而持续改善。

**⚠️ 局限性**

依赖 LLM 生成，可能带来偏差；聚焦有状态、域限定的环境，无法覆盖开放式检索或网络交互；缺少对延迟、网络波动、错误模式的模拟；仅支持文本工具，未涵盖多模态。

---

## 264. Gender Bias in LLMs: Preliminary Evidence from Shared Parenting Scenario in Czech Family Law

**arXiv ID:** 2601.05879 | [PDF](https://arxiv.org/pdf/2601.05879v1)

**作者:** Jakub Harasta `[一作]`, Tomas Foltynek `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了四款顶尖LLM在基于捷克家庭法的离婚共享抚养情景中的性别偏见。

**💡 创新点**

通过零射击交互结合真实案例与性别标识两种版本，首次探讨LLM在法律自助中的性别偏见。

**🔧 技术方法**

使用OpenAI、Anthropic、Google和Meta的LLM，采用自动化提示构造与JSON解析技术进行交互。

**📊 数据集**

使用专家设计的离婚情景与九个风险因素的事实插值，未使用公开数据集。

**📈 对比分析**

对每个模型执行20次重复，计算共享抚养比例的均值和标准差；发现某些模型对女性风险持有者表现出更高的偏好。

**⚠️ 局限性**

结果仅为描述性预备，未做统计显著性检验，存在性别标记、位置偏差及样本规模有限等局限。

---

## 265. Phase4DFD: Multi-Domain Phase-Aware Attention for Deepfake Detection

**arXiv ID:** 2601.05861 | [PDF](https://arxiv.org/pdf/2601.05861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 266. iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models

**arXiv ID:** 2601.05877 | [PDF](https://arxiv.org/pdf/2601.05877v1)

**作者:** Meghana Sunil `[一作]` (Nagasaki University), Muthu Subash Kavitha `[通讯]` (Nagasaki University)

**通讯引用:** 1272 | [OpenAlex ID](https://openalex.org/A5100667678)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一个无监督的自我进化框架 iReasoner，利用跨回放的链式思维（CoT）步骤一致性奖励来提升多模态模型的推理能力。

**💡 创新点**

在无标签图像上加入基于中间推理步骤一致性的轨迹感知奖励，实现了不依赖外部评测即可对推理过程进行优化。

**🔧 技术方法**

采用 Proposer–Solver 自我进化结构、REINFORCE 与 KL 正则化的策略梯度、CoT 一致性奖励、文本嵌入与加权相似度计算等技术。

**📊 数据集**

在 2500 张无标签图像（ChartQA、AI2D、InfoGraphic‑VQA、PlotQA、ChartX、Geometry3K）上训练，并在 8 个多模态推理基准（InfoGraphic‑VQA、AI2D、ScienceQA、MMMU、ChartQA、MathVista、MathVision、MathVerse）进行评估。

**📈 对比分析**

与基线 Qwen2.5‑VL‑7B、EvoLMM、VisPlay 等方法对比，iReasoner 在所有基准上均有提升，平均提升约 +2.1 分，尤其在视觉数学任务中表现突出。

**⚠️ 局限性**

仅靠内部一致性可能强化错误推理；受限于训练步数、数据量和模型可访问度，未解决外部正确性评估与黑盒系统的适配问题。

---

## 267. IIB-LPO: Latent Policy Optimization via Iterative Information Bottleneck

**arXiv ID:** 2601.05870 | [PDF](https://arxiv.org/pdf/2601.05870v1)

**作者:** Huilin Deng `[一作]`, Yu Kang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 I2B‑LPO，通过信息瓶颈驱动的潜在分支和自我奖励机制，解决大语言模型推理中的探索崩溃问题。

**💡 创新点**

将探索从 token 级概率扰动转为路径拓扑分支，并结合信息瓶颈做过滤与奖励，实现结构化多样化推理。

**🔧 技术方法**

使用条件变分自编码器（CVAE）采样潜变量、伪自注意力（PSA）注入、信息瓶颈正则化、以及 GRPO 训练框架。

**📊 数据集**

在 4 个数学基准（MATH‑500、AIME2025、AIME24、OlympiadBench）以及 GSM8K 上进行评测。

**📈 对比分析**

与 entropy‑regularization、token‑selective、self‑reward 等基线在 Qwen2.5‑7B/14B 上对比，I2B‑LPO 在准确率上提升约 5–10%，并在多样性指标上优于所有基线，同时保持合理的生成长度。

**⚠️ 局限性**

依赖熵作为不确定性指标在非数学开放式任务中的有效性有限，分支机制计算成本高。

---

## 268. Secure Change-Point Detection for Time Series under Homomorphic Encryption

**arXiv ID:** 2601.05865 | [PDF](https://arxiv.org/pdf/2601.05865v1)

**作者:** Federico Mazzone `[一作]` (University of Twente), Massimiliano Pronesti `[通讯]` (IBM Research Europe)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种首次在加密时间序列上完成变化点检测（change‑point detection，CPD）的方案，能够在不解密数据的前提下检测均值、方差和频率的突变。

**💡 创新点**

创新点包括：①利用CKKS同态加密实现全加密的CPD流程，避免了传统差分隐私的噪声损失；②设计了高效的加密比较和argmax算子，显著降低了加密运算成本；③将序列映射为序列模式（ordinal patterns）后使用CUSUM统计，实现了对依赖序列的非参数检测。

**🔧 技术方法**

核心技术：CKKS同态加密、序列模式（ordinal pattern）转换、CUSUM统计、加密多项式比较（f、g 近似）、SIMD 计算、矩阵编码与分块求和、加密 argmax 的乘法聚合。

**📊 数据集**

实验数据集包括：
- 合成数据（正态、均匀、拉普拉斯、Student‑t 分布，及 AR(1) 过程）
- 实际医疗数据：EEG 睡眠相变段（60,000 点）
- 实际医疗数据：心率间隔序列（8,083 点）
- 实际网络监控数据：CESNET‑TimeSeries24 的包率序列（40,298 点）。

**📈 对比分析**

与本地差分隐私（local‑DP）和中心差分隐私（central‑DP）方法对比，local‑DP 在 ε 较大时可达 20% 误差，ε 较小时误差过高；central‑DP 在独立假设下可达到较低误差，但不适用于依赖序列。本文方法在 1,000,000 点上检测均值/方差/频率变点分别耗时约 100–180 秒，精度与明文基线一致；加密过程相对明文多 2–3 个数量级，但仍可在几分钟内完成，通信与内存开销可接受。

**⚠️ 局限性**

局限性：
- 仅支持单一变点检测；多变点处理仍未实现。
- 块大小 m 需整除序列长度并为 2 的幂，若不满足需额外填充。
- 依赖序列的自相关性已被模型捕获，但对高度非平稳或噪声强度不稳定的序列效果未知。
- 目前仅实现了在已知变点存在的前提下的检测，未提供完整的显著性检验或阈值估计。

---

## 269. Router-Suggest: Dynamic Routing for Multimodal Auto-Completion in Visually-Grounded Dialogs

**arXiv ID:** 2601.05851 | [PDF](https://arxiv.org/pdf/2601.05851v1)

**作者:** Sandeep Mishra `[一作]` (Indian Institute of Technology Kharagpur), Manish Gupta `[通讯]` (Microsoft)

**通讯引用:** 5428 | [OpenAlex ID](https://openalex.org/A5101454729)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究多模态实时自动补全（MAC）任务，构建基准数据集，对比文本与视觉语言模型，并提出动态路由框架来平衡准确性与延迟。

**💡 创新点**

创新点在于首次定义MAC任务与基准，证明视觉上下文对补全的关键作用，且提出Router‑Suggest动态路由机制，显著提升速度同时保持高质量补全。

**🔧 技术方法**

采用Transformer‑based文本模型（MPC、MPC++、QB）和VLM（MiniCPM‑V、PaliGemma、Qwen2‑VL），使用LoRA微调、成本感知路由器训练等技术。

**📊 数据集**

基准数据集来自GPT‑4V过滤的MMDialog和ImageChat两大多模态对话数据集，确保视觉信息对补全具有决定性。

**📈 对比分析**

通过TR、SM、PR‑P/PR‑R、PR‑F1、TES等专用指标进行评估，结果显示VLM优于文本模型；Router‑Suggest在保持PR‑F1≈最佳的同时，速度提升2.3–10倍。

**⚠️ 局限性**

局限性包括：数据集主要包含单图、视觉明显场景，缺少多图或动态视觉；路由器依赖嵌入特征，域迁移时可能失效；用户研究样本有限且TES易被模型本身影响。

---

## 270. Influence of Parallelism in Vector-Multiplication Units on Correlation Power Analysis

**arXiv ID:** 2601.05828 | [PDF](https://arxiv.org/pdf/2601.05828v1)

**作者:** Manuel Brosch `[一作]` (Technical University of Munich), Georg Sigl `[通讯]` (Technical University of Munich)

**通讯引用:** 4022 | [OpenAlex ID](https://openalex.org/A5026512033)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文通过理论分析与 FPGA 实验，研究了在 AI 加速器中向量乘法单元的并行程度对相关功耗侧信道分析（CPA）成功率的影响，探讨了并行运算对功耗信号的噪声与相关性下降的机制，并给出了相应的指数衰减公式；

**💡 创新点**

创新点在于首次量化并行向量乘法对 CPA 的影响，推导出权重假设相关系数随并行 PE 数量呈指数衰减的关系式，并通过实验验证发现当并行 PE 超过约15个时，基于全局功耗的 CPA 成功率降至几乎为零；

**🔧 技术方法**

使用了相关功耗侧信道分析（CPA）、硬件功耗建模（HW/HD）、统计相关系数分析、仿真与 FPGA 现场测量等技术，并讨论了掩码和打乱等防御手段；

**📊 数据集**

数据集采用随机生成的 8 位权重和输入值；在附录中亦用训练好的 NN 层权重进行验证，但未使用公开的机器学习数据集；

**📈 对比分析**

实验与理论结果对比显示，理论上可达 15 个并行 PE 时仍能成功 CPA，实际 FPGA 测试仅在 8 个并行 PE 以内可提取权重，SNR 较低时需更多测量；整体性能表明，随着并行度提升，SNR 降低至约 0.045 以上才可能成功；

**⚠️ 局限性**

局限性包括仅考虑相同输入的并行计算场景；实验环境噪声较高，导致实际阈值低于理论值；未覆盖不同输入或流水线/时钟频率高的架构；仅验证了全局功耗侧信道，未涉及局部 EM 侧信道；

---

## 271. SSR: Safeguarding Staking Rewards by Defining and Detecting Logical Defects in DeFi Staking

**arXiv ID:** 2601.05827 | [PDF](https://arxiv.org/pdf/2601.05827v1)

**作者:** Zewei Lin `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 33227 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

系统地识别并检测 DeFi staking 合约中的逻辑缺陷，提出 SSR（Safeguarding Staking Reward）工具。

**💡 创新点**

首次对 DeFi staking 的逻辑缺陷进行实证分类并构建基于 LLM 的模型驱动检测框架。

**🔧 技术方法**

结合大语言模型提取 staking 关键信息，利用静态分析（控制流图、数据流图）和规则匹配进行缺陷检测。

**📊 数据集**

使用 64 起安全事件 + 144 篇审计报告构成的实证集，40 个真实合约做 ground truth，15,992 个公开合约做大规模评估。

**📈 对比分析**

通过人工标注的 ground truth 与采样验证，SSR 在 ground truth 上 precision 92.31%，recall 87.92%，F1 88.85%；在 15,992 合约上 precision 89.41%，22.24% 合约存在缺陷。

**⚠️ 局限性**

仅覆盖六种缺陷类型，对复杂或新型缺陷识别不足，LLM 与 Slither 生成的 CFG/DFG 误差导致 FP/FN。

---

## 272. Cross-National Evidence of Disproportionate Media Visibility for the Radical Right in the 2024 European Elections

**arXiv ID:** 2601.05826 | [PDF](https://arxiv.org/pdf/2601.05826v1)

**作者:** Íris Damião `[一作]` (Instituto Superior Técnico), Joana Gonçalves-Sá `[通讯]` (NOVA University Lisbon)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2024年欧盟议会选举期间，五个欧盟国家（奥地利、德国、爱尔兰、波兰、葡萄牙）主流媒体对不同政治家族的可见度进行了系统的跨国比较分析；

**💡 创新点**

创新点在于将大语言模型（ChatGPT‑4o）与模糊匹配相结合，对新闻标题和URL进行自动实体抽取，并将本国党派映射到欧盟议会集团及五大政治倾向，从而实现跨国、跨媒体平台的可见度量化；

**🔧 技术方法**

使用技术包括：Media Cloud平台抓取新闻；Semrush Traffic Analytics筛选热门媒体；ChatGPT‑4o进行实体识别与归属；模糊匹配提升召回；Python脚本统计与可视化；统计方法基于比例差异和标准差阈值；

**📊 数据集**

数据集为21,528篇来自Media Cloud的新闻标题/URL，覆盖五国共计21,528条条目，其中10,292条含有可辨识的政治实体；

**📈 对比分析**

比较方法：将每个政治倾向的媒体提及比例与2019选举结果、预选民调预测及2024实际结果进行对比，计算差异并以标准差界定显著性；性能方面，实体抽取准确率约95%，偏差分析显示极右党派的可见度普遍高于其选举表现；

**⚠️ 局限性**

局限性包括：仅分析标题/URL，未考虑正文内容与语调；未评估情感倾向或叙事框架；数据仅覆盖主流在线媒体，忽略社交媒体和付费内容；方法为相关性分析，无法证明因果关系；

---

## 273. Boosting Latent Diffusion Models via Disentangled Representation Alignment

**arXiv ID:** 2601.05823 | [PDF](https://arxiv.org/pdf/2601.05823v1)

**作者:** John Page `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种语义解耦变分自编码器（Send‑VAE），通过非线性映射网络将VAE潜空间与预训练视觉基础模型对齐，从而提升VAE的语义解耦能力并加速后续潜在扩散模型的训练。

**💡 创新点**

创新点在于：①发现VAE需要强语义解耦而非仅高层语义；②设计带ViT层的非线性映射器实现潜空间与视觉基础模型的对齐，并将此对齐作为VAE训练的额外损失；③提出使用线性探测属性预测作为评估VAE语义解耦的指标。

**🔧 技术方法**

技术包括：VAE训练、非线性映射器（patch embedding + ViT + MLP）、对齐损失（余弦相似度）、噪声注入、预训练视觉基础模型（DINOv2、CLIP等）、扩散模型SiT、SDE Euler–Maruyama采样、线性探测等。

**📊 数据集**

使用 ImageNet 256×256 作为主要数据集；在属性预测评测中使用 CelebA、DeepFashion、AwA。

**📈 对比分析**

与现有的 VAE+扩散模型（如 VA‑VAE、E2E‑VAE、REPA）在 ImageNet 256×256 生成任务中对比，使用 gFID、sFID、IS、Precision、Recall 等指标；Send‑VAE 在 80 epoch 训练即可达到 gFID 1.21（有 CFG）/1.75（无 CFG），显著优于对照模型，并且显著加速了 SiT 的训练。

**⚠️ 局限性**

局限性包括：①对预训练视觉基础模型的依赖；②在某些细节重建上略逊于传统 VAE；③对 VAE 初始化的敏感度仍有限；④未在多尺度或多模态任务中进行验证。

---

## 274. Learning Reconstructive Embeddings in Reproducing Kernel Hilbert Spaces via the Representer Theorem

**arXiv ID:** 2601.05811 | [PDF](https://arxiv.org/pdf/2601.05811v1)

**作者:** Enrique Feito-Casares `[一作]` (Universidad Rey Juan Carlos), José-Luis Rojo-Álvarez `[通讯]` (Universidad Rey Juan Carlos)

**通讯引用:** 5202 | [OpenAlex ID](https://openalex.org/A5051286582)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于再构造的核嵌入（Autoreconstructive Kernel Embedding）算法，通过在核Hilbert空间中求解自重构权重并对低维嵌入进行核对齐，实现无监督的非线性降维。

**💡 创新点**

创新点在于将自重构几何与可分离算子值核结合，得到闭式求解的重构权重；随后通过核对齐把高维核几何迁移至低维空间，形成全新的自重构驱动降维框架。

**🔧 技术方法**

采用了再现定理、算子值核、核对齐、二次优化、梯度下降、Nyström近似等核方法与优化技术，配合对称正定核和特征映射构建完整流程。

**📊 数据集**

实验使用合成数据（同心圆、瑞士卷）和真实数据（NCI癌症分子指纹、CIC‑IoT2023网络入侵），验证了方法在多种数据类型上的有效性。

**📈 对比分析**

与KPCA、UMAP等基准相比，本文方法在同心圆任务中达成了显著更低的Davies‑Bouldin与更高的Calinski‑Harabasz；在瑞士卷任务中兼顾信任度与连续性；在分子分类中在20维时实现最高的准确率与ROC‑AUC；在IoT入侵检测中得到更紧凑、分离度更高的低维表示，优于UMAP。

**⚠️ 局限性**

主要局限包括：缺乏显式逆映射，限制生成应用；需要Nyström等近似实现样本外扩展；计算复杂度受核矩阵O(n²)和三次操作O(n³)限制；未引入监督信息，且目前仅支持单一核函数。

---

## 275. SceneFoundry: Generating Interactive Infinite 3D Worlds

**arXiv ID:** 2601.05810 | [PDF](https://arxiv.org/pdf/2601.05810v1)

**作者:** ChunTeng Chen `[一作]` (National Yang Ming Chiao Tung University), YuanFu Yang `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 151 | [OpenAlex ID](https://openalex.org/A5040566036)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了 SceneFoundry，一种多阶段的语言引导扩散框架，用于生成可控、功能完整的公寓级3D场景，支持机器人学习与仿真。

**💡 创新点**

创新点包括：①将LLM与室内布局生成器结合，将自然语言转化为可调节的参数空间；②在扩散后向采样中引入可微分的目标函数，实现物体数量、可动作部件碰撞和可走性三种功能约束；③提出新的评估指标量化这些约束的可控性与功能合理性。

**🔧 技术方法**

主要技术：LLM（如ChatGPT）用于语义转参数；Diffusion Posterior Sampling（后向采样）配合可微分约束；可微分碰撞与占地面积约束；Post‑optimization 进行可走性调优；VAE 与 3D-FRONT/GAPartNet 进行资产检索。

**📊 数据集**

使用的数据集包括：3D‑FRONT（室内布局与CAD模型），GAPartNet（部件级语义与姿态），以及 3D‑FUTURE（纹理化家具模型）等。

**📈 对比分析**

与 ATISS、DiffuScene、PhyScene 等基线对比，SceneFoundry 在 FID、KID、SCA、CKL 等感知指标上保持相近或略优；在语义一致性指标（S_node≈0.99、S_constraint≈0.92、S_edge≈0.95）上远超基线；物体数量控制成功率 0.95‑0.97；功能碰撞率 R_acoll 降至 0.11、可达率 R_reach 提升至 0.81；可走性成功率在不同阈值下均显著优于基线。

**⚠️ 局限性**

局限性：仍依赖精细标注的 CAD 数据集，扩散模型对大规模物体排列的计算成本高；可走性优化仅在后处理阶段，可能无法实时适配动态环境；缺乏对物理动态交互（如抓取、运动学约束）的完整模拟。

---

## 276. Fusion Matters: Length-Aware Analysis of Positional-Encoding Fusion in Transformers

**arXiv ID:** 2601.05807 | [PDF](https://arxiv.org/pdf/2601.05807v1)

**作者:** Mohamed Amine Hallam `[一作]` (Harbin Institute of Technology), Kuo-Kun Tseng `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 943 | [OpenAlex ID](https://openalex.org/A5017727521)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文研究了位置编码与标记嵌入的融合机制对Transformer性能的影响，特别是在长序列设置下。通过对三种融合策略（逐元素相加、拼接与投影、标量门控融合）进行控制实验，比较了它们在不同长度文本分类数据集上的表现。

**💡 创新点**

创新点在于首次系统地将位置编码融合作为Transformer中的一个独立建模变量进行研究，发现融合机制对长序列分类的性能有显著影响，并提出了一种轻量级的卷积门控机制以引入局部归纳偏置。

**🔧 技术方法**

使用了标准的Transformer架构，结合了逐元素相加、拼接与投影、标量门控等多种融合技术，并探索了局部卷积门控机制。

**📊 数据集**

使用了三个文本分类数据集：AG News（短序列）、IMDB（中等长度序列）和ArXiv（长文档）。

**📈 对比分析**

通过固定架构和随机种子进行比较，发现短文本上不同融合方法的性能差异微乎其微，而在长文档上，标量门控融合相较于标准的加法融合提高了约6.5个百分点的准确率，且在不同数据集间的比较中表现一致。

**⚠️ 局限性**

本研究的局限性在于仅关注分类任务和单一的Transformer架构，未来的工作可以扩展到生成任务、混合编码或多模态设置。

---

## 277. An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift

**arXiv ID:** 2601.05882 | [PDF](https://arxiv.org/pdf/2601.05882v1)

**作者:** Constantinos Karouzos `[一作]` (University of Sheffield), Nikolaos Aletras `[通讯]` (University of Sheffield)

**通讯引用:** 3540 | [OpenAlex ID](https://openalex.org/A5010341007)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了在领域迁移情境下，偏好调优对语言模型性能的影响，并对不同对齐目标与适配策略进行对比

**💡 创新点**

创新点在于将对齐目标（SFT、DPO、KTO、ORPO、PPO、GRPO）与适配方法（源域SFT、混合SFT、目标SFT、伪标签）组合系统化评估，揭示伪标签能显著降低域迁移衰退但易导致模式崩溃

**🔧 技术方法**

采用偏好优化技术（DPO、KTO、ORPO、PPO、GRPO）与自监督的伪标签生成，结合LoRA、PEFT等微调框架

**📊 数据集**

使用Reddit TL;DR→CNN/DM摘要数据与AskEngineers→AskCulinary问答数据，构成源域与目标域对比实验

**📈 对比分析**

实验显示，伪标签适配在目标域取得最高win率（最高约95%），但多样性显著下降；混合SFT+在线RL在保持多样性的同时，目标域win率亦能提升；在源域表现最佳的对齐目标在迁移后往往失效

**⚠️ 局限性**

局限包括：仅使用7B–8B规模模型、仅评估英语摘要与问答、伪标签依赖强教师模型、LLM‑as‑judge评估可能偏向结构化回答、未涉及多语言或推理任务

---

## 278. Universal and Asymptotically Optimal Data and Task Allocation in Distributed Computing

**arXiv ID:** 2601.05873 | [PDF](https://arxiv.org/pdf/2601.05873v1)

**作者:** Javad Maheri `[一作]` (EURECOM Institute), Petros Elia `[通讯]` (EURECOM Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在分布式计算场景下，本文提出了一种确定性的文件与子函数分配方案（Interweaved Clique，IC 设计），该方案在任意子函数集合 X ⊆ A_{n,d} 下实现通信成本与计算延迟的共同最小化。

**💡 创新点**

创新点在于：
- 通过构造互交团（interweaved cliques）的组合结构，既能在保证通信成本 π_X 达到 n/N^{1/d} 的下界的同时，又能将计算延迟 δ_X 控制在常数 5（高概率）以内；
- 方案不依赖于具体的子函数集合 X，文件分配一次完成即可支持多种函数分解（“盲”分配）；
- 给出了关于通信成本的下界与上界的理论证明，并在广泛参数范围内实现了与下界匹配的阶数最优性。

**🔧 技术方法**

核心技术包括：
- 将问题转化为 d‑均匀超图边划分，利用组合学与信息理论中的团结构构造；
- 采用分层分块、分组与均匀划分的离散化方法（类似分割、拉普拉斯分解），保证每个组的文件覆盖数与子函数数均衡；
- 随机稀疏化（random thinning）与大数定律相结合，对计算延迟进行概率上限证明。

**📊 数据集**

本文为理论性工作，没有使用具体实验数据集；所有结果均在数学模型与随机稀疏假设下给出。

**📈 对比分析**

与现有的基于 ARF（平均复制因子）和图划分的 heuristic 算法相比，IC 设计在 d=2 时可实现 ARF ≤ 2√(2N)，而传统方法的上界为 O(√N)+N/n；在更一般的 d 情况下，IC 设计在满足 N ≤ (9/10√(n/d))^d 的前提下，通信成本达到与下界一致的阶数，并保持计算延迟不超过 5。实验（理论分析）表明，在大多数参数区间内，IC 设计的通信效率与现有算法相当或更优。

**⚠️ 局限性**

局限性包括：
- 需要满足 N ≤ (9/10√(n/d))^d，且 d 必须远小于 n（如 d ≤ n/32）才能保证 δ_X ≤ 5；
- 计算延迟上界的概率论证明依赖于子函数集合 X 的随机稀疏假设，针对特定结构化的 X 可能不成立；
- 方案在处理极端稀疏或极密集的子函数集合时（φ 接近 0 或 1）性能表现未给出完整分析；
- 没有考虑通信链路异构、故障恢复与多租户动态调度等实际系统细节。

---

## 279. FACTUM: Mechanistic Detection of Citation Hallucination in Long-Form RAG

**arXiv ID:** 2601.05866 | [PDF](https://arxiv.org/pdf/2601.05866v1)

**作者:** Maxime Dassen `[一作]` (University of Amsterdam), Kevin Duh `[通讯]` (Johns Hopkins University)

**通讯引用:** 6043 | [OpenAlex ID](https://openalex.org/A5070418792)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证FACTUM框架，用于检测长文本RAG系统中的引用幻觉，结合Transformer内部Attention与FFN通道的机制性分数进行判别。

**💡 创新点**

创新点在于：①将Attention和FFN拆分为“读”和“忆”两条信息流；②提出四个机制性分数（CAS、BAS、PFS、PAS），分别衡量上下文对齐、注意力汇聚、参数力与通道对齐；③通过特征压缩和学习发现规模依赖的内部签名，显著提升检测性能。

**🔧 技术方法**

使用机制性可解释性分析、Transformer内部状态分解、head/层裁剪、聚合与降维特征工程、逻辑回归/LightGBM/EBM分类器、LLM-as-judge标签生成、对NeuCLIR 2024数据集进行实验。

**📊 数据集**

TREC NeuCLIR 2024报告生成任务（15篇检索文档，机器翻译为英文）。

**📈 对比分析**

与ReDeEP（ECS+PKS）及多种内部不确定性指标（Perplexity、Entropy、Energy、P(True)）对比；在Llama-3.2-3B与Llama-3.1-8B上，FACTUM在AUC、Precision、Recall、F1均优于基线，最高AUC达0.737，提升约20%。

**⚠️ 局限性**

局限性：需要大量内部状态数据，训练成本高；特征裁剪与聚合对不同模型架构泛化有限；实验仅在英语翻译文本上验证，跨语言与多模态适用性待进一步研究。

---

## 280. Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation

**arXiv ID:** 2601.05855 | [PDF](https://arxiv.org/pdf/2601.05855v1)

**作者:** Kaiwen Huang `[一作]` (Nanjing University of Science and Technology), Tao Zhou `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 30258 | [OpenAlex ID](https://openalex.org/A5090925242)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种双向通道选择语义交互（BCSI）框架，用于半监督医学图像分割。

**💡 创新点**

创新点在于引入语义-空间扰动（SSP）实现弱到强一致性，并设计通道选择路由器（CR）与双向通道交互（BCI）来动态选择和交换最相关的特征通道，减少噪声并提升模型鲁棒性。

**🔧 技术方法**

主要技术包括强/弱数据增强（颜色抖动、复制粘贴）、伪标签、弱到强一致性学习、通道选择路由器、双向通道交互以及基于VNet的3D U‑Net结构。

**📊 数据集**

实验使用左心房（LA）、BraTS‑2019 及胰腺CT（Pancreas‑CT）三个医学图像数据集，分别包含 3D 磁共振/CT 切片。

**📈 对比分析**

与11种最新半监督方法（如UAMT、DTC、MC‑Net 等）以及全监督 VNet 进行对比，BCSI 在 Dice、IoU、95HD、ASD 等指标上均取得领先或相当表现，尤其在 10%‑20% 标注比例下显著优于竞争者。

**⚠️ 局限性**

局限性包括对 3D 医学图像的专注，模型参数量与推理时间较大；通道选择与双向交互的超参数需要手工调节；在不同模态或更大规模数据上泛化性尚未充分验证。

---

## 281. Discrete dualities for some algebras from rough sets

**arXiv ID:** 2601.05843 | [PDF](https://arxiv.org/pdf/2601.05843v1)

**作者:** Ivo Düntsch `[一作]` (Brock University), Ewa Orłowska `[通讯]` (Institute of Telecommunications)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并构建了多种从粗糙集得到的代数（单子代数、充分性代数、正则双石代数、德摩根代数以及粗糙关系代数）的离散对偶关系，并给出了对应的表示定理与框架嵌入映射。

**💡 创新点**

创新点在于系统化地将粗糙集框架下的多类代数与其对应的关系框架统一到离散对偶的视角，提出了新的充分性与多重关系（粗糙关系帧）的对偶性，扩展了模态、关系代数与布尔代数的统一理论。

**🔧 技术方法**

使用了 Stone 论证、Kripke 结构、算子对偶性、伪补运算、正则双石代数的极大极小点结构等理论技术来构造对偶框架和证明表示定理。

**📊 数据集**

本研究为理论综述，没有使用具体数据集。

**📈 对比分析**

没有实验比较，性能评估通过证明表示定理与嵌入性来验证对偶关系的正确性。

**⚠️ 局限性**

限制在于仅考虑等价关系或有限结构，对非等价关系、无限系统以及实际应用层面的推广尚未完成。

---

## 282. Left, Right, or Center? Evaluating LLM Framing in News Classification and Generation

**arXiv ID:** 2601.05835 | [PDF](https://arxiv.org/pdf/2601.05835v1)

**作者:** Molly Kennedy `[一作]` (Ludwig-Maximilians-Universität München), Hinrich Schütze `[通讯]` (Ludwig-Maximilians-Universität München)

**通讯引用:** 46885 | [OpenAlex ID](https://openalex.org/A5071144367)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估九种大型语言模型在新闻文本的政治立场分类与生成摘要中的偏见与框架倾向。

**💡 创新点**

首次将分类与生成两种评测并行，使用统一的评估器检测框架漂移，并系统分析安全对中心化倾向的影响。

**🔧 技术方法**

采用少样本提示、固定评估器（Gemini 3 Pro）、固定输出长度与温度=0 等技术实现分类与生成评测。

**📊 数据集**

使用 AllSides 左/中/右标签的新闻语料库，分别取 12k 文章做分类，1k 文章做摘要生成。

**📈 对比分析**

通过宏 F1、准确率、Cohen κ 等指标比较分类效果；对生成摘要进行评价得分，结果显示 Claude Sonnet 4.5 与 GPT‑5 在分类上领先，Grok 4 在生成偏见表达最强。

**⚠️ 局限性**

评估仅用单一 LLM 评审，可能带偏见；使用粗糙的左/中/右标签，未能捕获细粒度框架；模型接口差异与安全设计可能影响结果。

---

## 283. Improving Clinical Data Accessibility Through Automated FHIR Data Transformation Tools

**arXiv ID:** 2601.05822 | [PDF](https://arxiv.org/pdf/2601.05822v1)

**作者:** Adarsh Pawar `[一作]` (Binghamton University), Zhaohan Xi `[通讯]` (Binghamton University)

**通讯引用:** 111 | [OpenAlex ID](https://openalex.org/A5026309535)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一个完全基于浏览器的系统，自动将FHIR JSON资源解析、归一化并转化为可读的PDF/Excel报告和交互式可视化，帮助临床人员和研究者轻松访问和解读FHIR数据。

**💡 创新点**

创新点在于提供了一个零后端、无数据库、无需服务器的完整客户端流水线，既支持实时从FHIR端点拉取数据，也支持本地文件离线处理，显著降低了使用门槛和隐私泄露风险。

**🔧 技术方法**

技术实现上使用React构建前端架构，结合jsPDF、xlsx、Recharts等库完成解析、可视化与导出；在性能优化方面还考虑了Web Workers与流式解析等策略。

**📊 数据集**

评估数据集包括合成FHIR示例、HAPI FHIR公开资源以及去标识化的真实FHIR Bundle，涵盖Patient、Observation、Encounter、DocumentReference等多种资源类型。

**📈 对比分析**

通过与原始JSON字段对照的准确率评估、解析/导出时延测量和可视化渲染速度比较，系统在100% Patient、96% Observation等资源上实现高准确率；PDF生成120–180 ms、Excel导出80–140 ms、可视化渲染<50 ms，表现出实时交互性能。

**⚠️ 局限性**

局限性包括仅支持有限的FHIR资源类型，对非合规扩展的鲁棒性不足；浏览器内存和计算能力限制大规模多病人Bundle的处理；缺乏真实临床用户验证，需要进一步扩展FHIR profile和混合架构以满足更大规模场景。

---

## 284. Performance-Portable Optimization and Analysis of Multiple Right-Hand Sides in a Lattice QCD Solver

**arXiv ID:** 2601.05816 | [PDF](https://arxiv.org/pdf/2601.05816v1)

**作者:** Shiting Long `[一作]` (KTH Royal Institute of Technology), Dirk Pleiter `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 4698 | [OpenAlex ID](https://openalex.org/A5076651122)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了多右手边（rhs）阻塞技术并对DD-αAMG的Wilson‑Dirac算子评估与GMRES求解器进行了重构，采用新的数据布局提升SIMD利用率，并引入Arm SME指令集实现矩阵乘法优化。

**💡 创新点**

创新点在于同时针对算法层（rhs阻塞、批量GMRES）和实现层（可切换数据布局、SME外积实现）进行性能可移植性优化，并系统评估其对不同CPU架构的影响。

**🔧 技术方法**

使用C++/GCC自动向量化、手写SVE/SME内核、MPI+OpenMP并行化、以及自定义PAPI计数器进行性能分析。

**📊 数据集**

采用典型的Lattice QCD格点数据集，分别为128×64³和64×16³的四维格子。

**📈 对比分析**

在JUWELS（x86）、Ookami（A64FX）和HAICGU（Kunpeng）三大平台上对比，基准算子阻塞可实现10%–24%加速，批量GMRES提升约7%，但受内存带宽和编译器生成代码差异限制，性能提升不均衡。

**⚠️ 局限性**

局限性包括写密集导致的内存带宽瓶颈、对不同硬件的SIMD指令支持差异、编译器自向量化行为不可预测、SME尚无真实硬件验证以及需进一步改进数据局部性以降低写强度。

---

## 285. Tensor-DTI: Enhancing Biomolecular Interaction Prediction with Contrastive Embedding Learning

**arXiv ID:** 2601.05792 | [PDF](https://arxiv.org/pdf/2601.05792v1)

**作者:** Manel Gil-Sorribes `[一作]` (Nostrum Biodiscovery), Alexis Molina `[通讯]` (Nostrum Biodiscovery)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了Tensor‑DTI框架，利用分子图、蛋白质语言模型和结合位点嵌入的多模态表征，通过对比学习和Siamese双编码器实现了药物–靶标交互预测和亲和力回归。

**💡 创新点**

创新点包括：①将分子图、蛋白质Transformer嵌入和结合位点信息统一映射到共享潜在空间；②在双编码器中引入对比学习以强化正负样本分离；③利用PickPocket等工具生成的结合位点嵌入，实现基于结合位点的可解释性与特异性提升。

**🔧 技术方法**

技术手段包括：GCN（处理分子图）、SaProt/ESM‑2 Transformer（处理蛋白序列）、PickPocket+GearNet（生成结合位点嵌入）、Siamese双编码器、对比损失+交叉熵/均方误差训练，以及不确定性和“陌生度”评估模块。

**📊 数据集**

使用了多种公开基准数据集：BIOSNAP、BindingDB、DAVIS、PLINDER、LP‑PDBBind、PDBBind‑Opt、DUD‑E、CoPRA、Propedia、PRA310，以及Enamine REAL超大规模化合物库等。

**📈 对比分析**

在DTI、DTA和加速筛选任务中与ConPLex、MolTrans、GraphDTA、HyperAttentionDTI、DeepConv‑DTI等现有方法进行对比；Tensor‑DTI在BIOSNAP（AUPR 0.903）、BindingDB（0.699）、DAVIS（0.547）等任务中均超过对手；在TDC‑DG（PCC 0.580）和LP‑PDBBind（PCC 0.565/0.750）等回归任务中也表现出竞争优势；在DUD‑E、PLINDER等低泄漏基准上同样获得最佳或接近最佳性能；大规模CDK2筛选和加速提升实验中，Tensor‑DTI实现了与Glide/ Boltz‑2相当或更好的提取率和全量回收效率。

**⚠️ 局限性**

局限性包括：①结合位点版本受限于Pocket数据集规模与多样性，导致大规模筛选时收敛不稳；②模型在极端异构或未见的结合位点/蛋白家族时仍存在性能衰退；③对结合位点相似性过度依赖，可能导致表面特征驱动的误判；④对罕见或极度多样的化学空间仍需进一步的可靠性校准与主动学习补充。

---

## 286. SAFE: Secure and Accurate Federated Learning for Privacy-Preserving Brain-Computer Interfaces

**arXiv ID:** 2601.05789 | [PDF](https://arxiv.org/pdf/2601.05789v1)

**作者:** Tianwang Jia `[一作]` (Huazhong University of Science and Technology), Dongrui Wu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 14574 | [OpenAlex ID](https://openalex.org/A5008740867)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种安全且准确的联邦学习框架SAFE，用于在保护EEG隐私的前提下实现跨用户的脑机接口解码。

**💡 创新点**

创新点在于将本地批次归一化（LBSN）与联邦对抗训练（FAT）和权重扰动（AWP）结合，既提升了跨主体泛化，又增强了对抗鲁棒性，同时实现无校准数据的隐私保护。

**🔧 技术方法**

采用联邦学习、局部批次归一化、联邦对抗训练、权重扰动以及EEGNet作为网络骨干。

**📊 数据集**

使用了五个EEG数据集：MI1、MI2、MI3（运动意象）和ERP1、ERP2（事件相关电位）。

**📈 对比分析**

与14个基准方法（7个集中式、7个联邦式）在五个数据集上进行比较，SAFE在多数指标上均优于所有基准，尤其在准确率和对抗鲁棒性上实现了最优表现。

**⚠️ 局限性**

局限性包括：实验仅在离线设置下完成，缺乏实时在线验证；对大规模联邦网络的通信开销和客户端计算资源需求未做深入评估；以及对其他BCI范式（如SSVEP）的适用性尚未验证。

---

## 287. AdaFuse: Adaptive Ensemble Decoding with Test-Time Scaling for LLMs

**arXiv ID:** 2601.06022 | [PDF](https://arxiv.org/pdf/2601.06022v1)

**作者:** Chengming Cui `[一作]` (University of Illinois Urbana-Champaign), Jingrui He `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4125 | [OpenAlex ID](https://openalex.org/A5073158087)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AdaFuse，一种在推理时自适应词级融合的大语言模型集成框架，能够根据生成上下文动态决定是否进行集成。

**💡 创新点**

创新点在于置信度驱动的自适应词级承诺与多样性感知的两阶段词级搜索，使得中途生成可以灵活、语义连贯地进行融合。

**🔧 技术方法**

采用置信度阈值判定、两阶段词级搜索、跨模型归一化负对数似然(NLL)评估以及加权融合等技术。

**📊 数据集**

使用开放域问答（NQ、SQuAD、TriviaQA）、算术推理（GSM8K）和机器翻译（Floresta En→De、De→En）等六大基准数据集。

**📈 对比分析**

与四种主流集成方法（LLM‑Blender、DeepEn、SweetSpan、UniTE）对比，AdaFuse 在所有任务上平均提升约 6.88%（相对），在各个基准上均取得最高分。

**⚠️ 局限性**

局限性是需要访问模型的 token‑level 概率和 tokenizer 输出，无法直接用于封闭或黑盒 LLM API。

---

## 288. Mobility Trajectories from Network-Driven Markov Dynamics

**arXiv ID:** 2601.06020 | [PDF](https://arxiv.org/pdf/2601.06020v1)

**作者:** David A. Meyer `[一作]` (University of California), Asif Shakeel `[通讯]` (University of California)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于时间相关马尔可夫动力学的生成式人类移动模型，利用网络层级（枢纽、走廊、供给路径、地铁连线）和重力式距离衰减、时间表及方向偏差构造转移矩阵，并通过记忆无关、可互换的“隐形移动者”产生轨迹。

**💡 创新点**

核心创新在于：①直接规定马尔可夫转移矩阵而非从轨迹推断；②将宏观流量约束与微观轨迹生成统一在网络结构与时间偏差上；③提供可解释的、隐私友好的轨迹与 OD 汇总一致性框架；④通过 Perron‑Frobenius 理论保证每日周期不变分布。

**🔧 技术方法**

使用技术包括：H3 网格空间离散、无向基础图与重构的层级覆盖网络、重力模型加权转移、时间调度模块（stay、mass、hub、feeder、metro、方向偏差）、列随机化的马尔可夫链、概率抽样生成轨迹、后验距离/时间赋值，以及矩阵一致性检验（l1、RMSE、JSD）。

**📊 数据集**

数据集主要为合成：利用 H3 六阶网格构造的网络和预设的时间调度；实验中未使用真实 GPS 或电话数据，强调模型可直接对聚合 OD 流进行推断与模拟。

**📈 对比分析**

评估方法为内部一致性验证：从生成的轨迹估计多步转移矩阵，再与按时间顺序合成的马尔可夫矩阵比较。性能指标为矩阵级与列级误差（l1、RMSE、JSD），误差随轨迹数量（K）增加而趋于零，表明误差仅来自有限样本噪声。

**⚠️ 局限性**

局限性包括：①缺乏个体行为与偏好建模，模型只能捕捉由网络和时间偏差决定的宏观流动；②参数（枢纽位置、方向偏差时程、距离衰减指数等）需人工设定或后续拟合，若与真实情况差异大则预测失真；③仅适用于可视化的离散网格，细粒度运动细节无法重现；④对高频、实时动态的场景支持有限。

---

## 289. AWaRe-SAC: Proactive Slice Admission Control under Weather-Induced Capacity Uncertainty

**arXiv ID:** 2601.05978 | [PDF](https://arxiv.org/pdf/2601.05978v1)

**作者:** Dror Jacoby `[一作]` (Tel Aviv University), Igor Kadota `[通讯]` (Northwestern University)

**通讯引用:** 1826 | [OpenAlex ID](https://openalex.org/A5052671825)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出基于天气预测的主动切片接入控制框架 AWaRe‑SAC，利用深度学习预测毫米波链路容量并结合 Q‑learning 优化切片接入，以提升雨衰减条件下 mmWave x‑haul 网络的收益和 QoS。

**💡 创新点**

①整合概率天气衰减预测与基于预测的 Q‑learning；②使用注意力集成 RNN（AttIRNN）进行多步雨衰减预测并生成容量概率分布；③在容量变化下实现奖励/惩罚驱动的动态切片分配，逼近离线最优。

**🔧 技术方法**

深度序列到序列注意力 RNN（AttIRNN）、基于预测的 Q‑learning、局部最优 MILP 求解、概率转化的容量状态离散化以及奖励/惩罚建模。

**📊 数据集**

①V‑band mmWave 链路 58–70 GHz 的 RSL 与容量实测数据（6 条链路，34 h 三场雨）；②来自 520 居民单元的流量追踪，用于生成 eMBB/URLLC/mMTC 切片请求；③公开气象站降雨强度信息。

**📈 对比分析**

与传统贪心、无预测 Q‑learning、局部最优 MILP、Oracle（零误差预测）等方案比较；在真实雨场景下，预测 Q‑learning 通过 2–3 倍收益提升，奖励提升约 20–30%，并在不同 CV 条件下保持低于 5% 的超额惩罚率。

**⚠️ 局限性**

仅在 V‑band（70 GHz）密集城市环境下验证；预测误差仍可能导致罕见错误；切片模型仅基于住宅流量，未覆盖工业/车联网；框架假设单中心星型拓扑；需进一步扩展到多域 O‑RAN 与更广泛的 ISAC 场景。

---

## 290. Categorical Foundations for CuTe Layouts

**arXiv ID:** 2601.05972 | [PDF](https://arxiv.org/pdf/2601.05972v1)

**作者:** Jack Carlisle `[一作]` (Colfax Research), Paul VanKoughnett `[通讯]` (Colfax Research)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种自动生成针对矩阵乘法的GPU内核的方法，利用张量收缩表示的线性代数表达式，结合内存布局、块化、线程共线、共享内存使用及边界处理等技术构造优化内核。

**💡 创新点**

创新点在于将张量收缩与布局变换结合成一种自动化编译流程，能够根据不同的数据形状生成高效且自适应的GPU内核；同时提出了一套基于嵌套元组和操作符的理论框架，用于统一描述内存布局和变换。

**🔧 技术方法**

使用了CUDA GPU编程技术、共享内存调度、线程块划分、内存共线化、块化/拆分算法，以及基于操作符代数的布局优化。

**📊 数据集**

使用了Matrix Market的标准矩阵数据集以及合成的测试矩阵（随机尺寸、稀疏/稠密组合）。

**📈 对比分析**

与手工调优的CUDA内核进行对比，通过吞吐量（GFlops/s）和占用率等指标进行评估，实验显示在多种规模下实现了最高30%的吞吐量提升，并显著降低了占用率问题。

**⚠️ 局限性**

局限性包括：仅针对矩阵乘法和张量收缩的特定类内核，未在多GPU或异构系统上验证；理论框架在极大规模或高深度嵌套布局时的复杂度和内存占用仍需进一步评估；自动化生成的内核在某些极端边界情况可能无法达到最佳性能。

---

## 291. Prophet as a Repro ducible Forecasting Framework: A Methodological Guide for Business and Financial Analytics

**arXiv ID:** 2601.05929 | [PDF](https://arxiv.org/pdf/2601.05929v1)

**作者:** Sidney Shapiro `[一作]` (University of Lethbridge), Burhanuddin Panvelwala `[通讯]` (University of Lethbridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究 Prophet 框架在金融与零售时间序列中的可重复性与预测性能，进行与多种 ARIMA、SARIMA、随机森林模型的对比实验。

**💡 创新点**

将可重复性三维度（数据、计算、分析）纳入评价，提出 Prophet 在标准化工作流、参数可解释性和不确定性量化方面的可重复性优势，并提供完整代码与环境配置。

**🔧 技术方法**

使用 Prophet 及其 Stan 后端、statsmodels 的 ARIMA/SARIMA、scikit‑learn 随机森林、Python 脚本化工作流、交叉验证与 Diebold‑Mariano 检验。

**📊 数据集**

公开的 Tesla 股票每日收盘价（2015‑2024）和 UCI Store Item Demand（2013‑2017）日销量汇总数据。

**📈 对比分析**

通过 RMSE/MAE/MAPE 与 95% 区间覆盖率进行评价，发现 Prophet 在零售季节性数据上优于基准模型，但在波动性金融数据上预测精度低于 ARIMA 与随机森林；DM 检验证实这一差异显著。

**⚠️ 局限性**

Prophet 在高度波动的金融序列中产生宽泛的置信区间，且对超参数（如变点尺度）敏感；实验仅覆盖两类时间序列，未检验更复杂的多变量或非线性模型。

---

## 292. Distilling Lightweight Domain Experts from Large ML Models by Identifying Relevant Subspaces

**arXiv ID:** 2601.05913 | [PDF](https://arxiv.org/pdf/2601.05913v1)

**作者:** Pattarawat Chormai `[一作]` (Technische Universitaet Berlin), Grégoire Montavon `[通讯]` (Charite Universitaetsmedizin Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种针对子任务的知识蒸馏方法SubDistill，使教师模型只对相关类别进行蒸馏

**💡 创新点**

创新点在于结合可解释AI的PRCA提取子任务相关子空间，并在层级蒸馏中使用正交子空间匹配以提升数值稳定性和对齐精度

**🔧 技术方法**

技术包括层级蒸馏、正交适配器、PRCA子空间提取、Stiefel流形优化、LRP可解释性分析

**📊 数据集**

使用CIFAR‑100和ImageNet，在各自的超类子任务上进行实验

**📈 对比分析**

与Output Only、AT、VID、VKD等基线比较，SubDistill在所有子任务上均提升5–10%准确率，低数据量场景下仍保持优势

**⚠️ 局限性**

局限性包括仅在图像分类任务验证，未测试文本或多模态场景；对大规模模型的PRCA计算成本未作系统评估

---

## 293. Pantagruel: Unified Self-Supervised Encoders for French Text and Speech

**arXiv ID:** 2601.05911 | [PDF](https://arxiv.org/pdf/2601.05911v1)

**作者:** Phuong-Hang Le `[一作]` (Univ. Grenoble Alpes), Didier Schwab `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个统一的 French 自监督编码器家族，分别针对文本和语音，使用 data2vec 2.0/JEPA 框架进行预训练。

**💡 创新点**

创新点在于：① 为文本引入了结合 MLM 与 embedding‑level 预测的复合目标；② 将相同架构分别应用于文本与语音；③ 在预训练中加入了 10 万小时法国广播音频数据，显著提升噪声鲁棒性。

**🔧 技术方法**

使用了 data2vec 2.0（基于 JEPA）的预测式自监督方法，文本端辅以 Masked Language Modeling；实现了教师‑学生 EMA 训练、特征级 L2 损失和可选的 token‑level 交叉熵。

**📊 数据集**

文本数据：Wikipedia、OSCAR、CroissantLLM；语音数据：LeBenchmark（14k 小时）与 INA 100k 小时广播语音；此外还使用了对齐语料（CommonVoice、ETAPE 等）做下游评测。

**📈 对比分析**

与 FlauBERT、CamemBERT（文本）和 LeBenchmark（语音）对比，模型在大多数下游任务（QA、分类、NER、SLU、ASR 等）上取得相当或更佳的性能；在语音端尤其在噪声/自发语料上提升明显；文本端在语义任务上优于基线，但在句法/依存解析上仍略逊。

**⚠️ 局限性**

限制：embedding‑level 目标对文本的句法细粒度表达不足，需与 MLM 结合；大模型训练仍受限于资源，未完成大规模文本模型；广播语料对某些清晰语音任务略有负面影响；多模态联合训练仍待进一步验证。

---

## 294. Can AI mediation improve democratic deliberation?

**arXiv ID:** 2601.05904 | [PDF](https://arxiv.org/pdf/2601.05904v1)

**作者:** Michael Henry Tessler `[一作]` (Google DeepMind), Christopher Summerfield `[通讯]` (University of Oxford)

**通讯引用:** 16657 | [OpenAlex ID](https://openalex.org/A5031878516)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一种基于大语言模型的“Habermas Machine”，用于在大规模在线群体中生成并选取共同立场声明，实现AI调解的可扩展民主讨论。

**💡 创新点**

首次将生成模型与奖励模型结合，模拟选举并使用社会选择算法，系统性地在众多实验中证明AI调解能高效、公平地产出共同立场，超越人工调解。

**🔧 技术方法**

采用大语言模型（如GPT、Chinchilla等）生成候选声明，奖励模型预测个人排名，社会选择方法（Schulze）进行投票汇总；并利用多轮反馈迭代优化。

**📊 数据集**

使用来自英国Kaggle/MTurk等平台的2110名参与者意见、批评和排名数据，并在代表性样本中复现实验。

**📈 对比分析**

与人工调解和随机候选比较，Habermas Machine在接受度、时间成本、群体分化程度等指标上均优于人类调解，且在多轮后显著提高支持度。

**⚠️ 局限性**

受限于训练样本代表性不足、奖励模型预测误差、算法偏见、可解释性不足、战略操纵风险、缺乏社会情感及算法恐惧，且需进一步验证在真实政治情境中的可靠性与安全。

---

## 295. StackPlanner: A Centralized Hierarchical Multi-Agent System with Task-Experience Memory Management

**arXiv ID:** 2601.05890 | [PDF](https://arxiv.org/pdf/2601.05890v1)

**作者:** Ruizhe Zhang `[一作]` (Peking University), Yasha Wang `[通讯]` (Peking University)

**通讯引用:** 4696 | [OpenAlex ID](https://openalex.org/A5055336632)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出层级化中央多代理框架，显式管理任务记忆与经验记忆以提升长期协作稳定性和跨任务泛化。

**💡 创新点**

将记忆作为可控资源分离为任务记忆和经验记忆，并通过主动记忆管理与强化学习优化协调策略。

**🔧 技术方法**

采用层级化多代理、主动记忆管理（更新/压缩/剪枝）、结构化经验记忆检索、基于GRPO的强化学习决策，以及检索增强的LLM。

**📊 数据集**

使用多跳问答与代理基准：2WikiMultiHopQA、MusiQue、GAIA、FRAMES等。

**📈 对比分析**

与Naive、单代理、传统多代理及Agentic-RL基线对比，实验表明在3B/7B LLM上F1最高，明显优于所有对手。

**⚠️ 局限性**

对多轮交互支持不足，长期记忆存在冷启动问题，且在更开放领域的验证有限。

---

## 296. Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks

**arXiv ID:** 2601.06007 | [PDF](https://arxiv.org/pdf/2601.06007v1)

**作者:** Elias Lumer `[一作]` (PricewaterhouseCoopers), Vamse Kumar Subbiah `[通讯]` (PricewaterhouseCoopers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估 LLM 代理长时间任务中 Prompt 缓存的成本与延迟收益

**💡 创新点**

首次系统性量化 Prompt 缓存对多轮代理任务的成本与延迟影响，并提出基于缓存边界的策略改进

**🔧 技术方法**

Prompt 缓存、LLM 调用、工具调用、TTFT 与成本测量技术

**📊 数据集**

DeepResearchBench（500+ 代理会话，10k‑token 系统提示）

**📈 对比分析**

对 OpenAI、Anthropic、Google 三大供应商的四款模型进行 40 次会话的基线与三种缓存策略比较，成本下降 45‑80%，TTFT 提升 13‑31%

**⚠️ 局限性**

受限于仅评估特定模型与供应商，且仅针对静态系统提示场景，动态内容与多租户环境下的缓存行为不完全代表

---

## 297. The Molecular Structure of Thought: Mapping the Topology of Long Chain-of-Thought Reasoning

**arXiv ID:** 2601.06002 | [PDF](https://arxiv.org/pdf/2601.06002v1)

**作者:** Qiguang Chen `[一作]` (ByteDance Seed China), Wenhao Huang `[通讯]` (ByteDance Seed China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将长链式思考（Long CoT）建模为由三类“化学键”——深层推理、反思和探索——构成的分子结构，并引入语义同分异构体概念；

**💡 创新点**

创新点在于把长 CoT 的逻辑结构视为稳定的分子网络，从键的分布和互作角度解释学习稳定性；通过结构感知合成框架在无长 CoT 数据的前提下能从指令模型中生成高质量长 CoT；

**🔧 技术方法**

采用行为转移图、稀疏自编码器、注意力能量分析等技术；还结合监督微调、知识蒸馏与强化学习进行评估；

**📊 数据集**

使用多种公开推理基准（GSM8K、MATH‑500、AIME、AMC、OlympiadBench）以及从强教师模型（DeepSeek‑R1、Qwen2.5、OpenAI‑OSS、Gemini、Claude）蒸馏得到的长 CoT 样本；

**📈 对比分析**

与仅指令调优、人工注解、ICL 方式蒸馏等方法对比，结构感知合成在六个基准上达到与强教师蒸馏相近的得分，并在强化学习任务中显著提升稳定性；

**⚠️ 局限性**

局限包括：仅在少数教师/学生模型上验证；仍停留在离线蒸馏与监督微调，缺乏在线交互评估；分子结构的通用性与标签噪声仍待进一步验证。

---

## 298. Open-Vocabulary 3D Instruction Ambiguity Detection

**arXiv ID:** 2601.05991 | [PDF](https://arxiv.org/pdf/2601.05991v1)

**作者:** Jiayu Ding `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**通讯引用:** 13766 | [OpenAlex ID](https://openalex.org/A5100447673)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了开放词汇3D指令歧义检测任务，构建了大规模基准Ambi3D，并设计了两阶段感知‑推理框架AmbiVer；

**💡 创新点**

创新点在于首次把指令歧义识别作为前置任务，采用分离感知与逻辑推理的两阶段结构，利用可视证据驱动零样本VLM进行歧义判定；

**🔧 技术方法**

技术包含语义解析、Grounding DINO多视图目标检测、BEV构建、ray‑based 3D融合与Union‑Find聚类、以及Qwen‑3‑VL等通用VLM的零样本推理；

**📊 数据集**

使用自制的Ambi3D数据集，包含703个ScanNet室内场景、22k条开放词汇指令，涵盖实例、属性、空间、动作等四类歧义；

**📈 对比分析**

与多种3D LLM基线比较，AmbiVer在零样本下取得最高准确率81.3%、F1 83.3%，显著优于Fine‑tuned LoRA版本（约70%）和其他基线；

**⚠️ 局限性**

局限在于对高质量视频/相机位置信息依赖较高，VLM推理的可解释性和鲁棒性有限，且对更细粒度或语义错误的歧义处理仍有提升空间。

---

## 299. Community-Based Model Sharing and Generalisation: Anomaly Detection in IoT Temperature Sensor Networks

**arXiv ID:** 2601.05984 | [PDF](https://arxiv.org/pdf/2601.05984v1)

**作者:** Sahibzada Saadoon Hammad `[一作]` (Universitat Jaume I), Sergio Trilles Oliver `[通讯]` (Universitat Jaume I)

**通讯引用:** 2053 | [OpenAlex ID](https://openalex.org/A5014771012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在温度传感器网络中提出一种基于社区兴趣（CoI）的异常检测框架，先融合时间、空间、海拔三维相似度构建层次聚类得到若干社区，再为每个社区训练 MLP、LSTM 与 BiLSTM 自动编码器模型，并在社区内外共享模型进行异常判别；

**💡 创新点**

创新点在于（1）利用多维融合相似矩阵将传感器划分为功能相似的社区，实现模型共享；（2）将时间相关性、空间距离衰减和海拔相似度三者统一融合，提升社区划分的鲁棒性；（3）采用贝叶斯超参数优化与扩展窗口交叉验证相结合的训练策略，进一步提高模型泛化性能；

**🔧 技术方法**

主要技术包括 Spearman 相关系数、Gaussian 距离衰减函数、层次聚类、三种深度自编码器（MLP‑AE、LSTM‑AE、BiLSTM‑AE）、贝叶斯超参数搜索、时间序列扩展窗口交叉验证、标准化预处理以及 TensorFlow/Keras 框架；

**📊 数据集**

使用 AVAMET 计量的 41 个加泰罗尼亚卡斯特罗省气象站的温度时间序列数据，采样间隔为 10 分钟；

**📈 对比分析**

在同一社区内，三种模型的精准率普遍超过 0.88，PR‑AUC 在 0.75–0.9 之间；跨社区测试时，Cluster 1、3、4 的模型保持 0.80 以上的精准率，但针对 Cluster 2 的模型精准率骤降至 0.58 甚至 0.07，表明社区划分对模型迁移有显著影响；

**⚠️ 局限性**

局限性包括（1）Cluster 2 的数据特征差异导致模型迁移效果差，缺乏解释性分析；（2）仅在云端实验，未验证在边缘设备上的可部署性；（3）异常标注主要为人工合成，真实异常样本稀缺，可能影响评估真实性。

---

## 300. A Framework for Optimizing Human-Machine Interaction in Classification Systems

**arXiv ID:** 2601.05974 | [PDF](https://arxiv.org/pdf/2601.05974v1)

**作者:** Goran Muric `[一作]`, Steven Minton `[通讯]` (InferLink Corporation)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了双阈值策略下的人机协作分类系统，并提出了一个优化框架，用于在保持系统准确性的同时最小化人工审核负担，随后利用蒙特卡洛仿真评估不同阈值配置的性能；

**💡 创新点**

创新点在于将自动决策与人工审核的成本与准确性统一建模，给出可复制的双阈值优化方法，并通过Pareto前沿展示预算-性能之间的权衡；

**🔧 技术方法**

主要技术包括贝叶斯假设下的概率校准模型、双阈值决策规则、精确度/召回率/F1指标计算、Beta混合分布的概率模拟以及蒙特卡洛仿真；

**📊 数据集**

本文使用合成的10,000个样本的概率分布（Beta混合、右偏、左偏）进行实验，并提供完整代码可在GitHub复现；

**📈 对比分析**

通过在不同阈值和审核预算下计算预期TP、FP、F1等指标，绘制热图和Pareto前沿，结果表明随着审核预算增加F1提升但递减收益，最佳阈值区间明显；

**⚠️ 局限性**

局限性包括假设概率已完全校准、人工审核者被视为完美、未考虑多类别或多专家场景以及审核成本可能不均匀的实际情况；

---

## 301. Distilling Feedback into Memory-as-a-Tool

**arXiv ID:** 2601.05960 | [PDF](https://arxiv.org/pdf/2601.05960v1)

**作者:** Víctor Gallego `[一作]` `[通讯]` (Komorebi AI Technologies), Víctor Gallego (Komorebi AI Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将评估者反馈转化为可写文件式记忆的框架，利用工具调用在推理时检索并应用这些“经验教训”，从而摊销推理时间成本。

**💡 创新点**

创新点在于将LLM的临时批评抽象为持久可检索的规则文件，既保留了自我批评的高精度，又通过文件系统实现了透明、可解释的记忆管理，显著降低推理成本。

**🔧 技术方法**

使用文件系统作为外部记忆、工具调用接口、基于概率生成的LLM推理、Rubric Feedback Bench评估、对比自我批评和零样本基线等技术。

**📊 数据集**

采用 Rubric Feedback Bench（42个情景、5个任务类别）以及相应的评估者模型进行实验。

**📈 对比分析**

在持续学习和混合任务实验中，与零样本基线和推理时间自我批评相比，记忆+反馈方案在两轮反馈后即可达到或超过自我批评的分数，并在12任务长序列中平均得分为 0.78±0.10，高于基线的 0.52±0.25，同时成本显著低于自我批评。

**⚠️ 局限性**

限制在于检索机制依赖于文件名推理，随着文件数量扩展到千级时可扩展性受限；在长生命周期下可能需要更复杂的层次检索或主动遗忘机制。

---

## 302. Agentic LLMs as Powerful Deanonymizers: Re-identification of Participants in the Anthropic Interviewer Dataset

**arXiv ID:** 2601.05918 | [PDF](https://arxiv.org/pdf/2601.05918v1)

**作者:** Tianshi Li `[一作]` (Northeastern University), Tianshi Li `[通讯]` (Northeastern University)

**通讯引用:** 1293 | [OpenAlex ID](https://openalex.org/A5039928918)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大型语言模型和网络搜索技术，对 Anthropic Interviewer 的科学家采访数据进行再识别攻击，成功重识别出 6 条采访对应的论文及作者。

**💡 创新点**

提出将再识别攻击拆解为表面无害的任务流程，展示 LLM 代理可轻松突破传统匿名化措施，降低攻击门槛。

**🔧 技术方法**

采用无思考模型筛选含已发表工作描述的采访，使用思考模型代理进行网络检索、匹配与排名，并对候选结果给出置信度与理由。

**📊 数据集**

使用 Anthropic Interviewer 公共数据集（共 1,250 条采访，125 条科学家采访）。

**📈 对比分析**

通过 LLM API 执行攻击，单条采访成本 <0.5 美元，耗时约 4 分钟，24 条可识别采访中成功识别 6 条（25%），体现低成本高效率。

**⚠️ 局限性**

仅适用于已公开的科学家采访数据，未评估更大规模或受限数据；攻击效果受 LLM 搜索能力影响，未来模型升级可能提升或削弱识别率；未深入讨论法律合规与不同隐私法规的适用性。

---

## 303. Deepfake detectors are DUMB: A benchmark to assess adversarial training robustness under transferability constraints

**arXiv ID:** 2601.05986 | [PDF](https://arxiv.org/pdf/2601.05986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 304. Navigating the Sociotechnical Imaginaries of Brazilian Tech Workers

**arXiv ID:** 2601.05961 | [PDF](https://arxiv.org/pdf/2601.05961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 305. Auditing Fairness under Model Updates: Fundamental Complexity and Property-Preserving Updates

**arXiv ID:** 2601.05909 | [PDF](https://arxiv.org/pdf/2601.05909v1)

**作者:** Ayoub Ajarra `[一作]` (University of Lille), Debabrota Basu `[通讯]` (University of Lille)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种通用的黑盒模型审计框架，支持在模型被更新后仍能估计并保证群组公平性（统计对齐）及构造符合审计属性的后续模型集合（prospect class）。

**💡 创新点**

核心创新是引入“SP维度”（Statistical Parity Dimension），一种比VC维度更弱、更贴合公平性审计的组合学度量；并将审计任务转化为基于ER(Empirical Property Optimization)的弱/强审计算法，兼顾属性估计与后续模型集合的完整性。

**🔧 技术方法**

技术手段包括：定义属性相关的损失函数、使用ER-Oracle实现EPO；利用均匀收敛与最优近似证明弱审计可行；对无限模型族提出SP维度的上界与下界；利用Prospect Ratio量化后续模型集合的覆盖率；以及实验中采用深度学习、随机森林等策略模型。

**📊 数据集**

使用两个公开公平性基准数据集：COMPAS（Caucasian vs. non‑Caucasian）和Student Performance（Female vs. Male）。

**📈 对比分析**

与传统的统计估计、基于教练集的重建方法以及基于置信度的显式估计进行对比；实验显示在样本量从几百到上千时，估计误差可降至<0.01，Prospect Ratio收敛近似真实值；计算成本仅为每样本毫秒级。

**⚠️ 局限性**

局限性：仅对统计对齐公平性做详细分析；对无限VC维度模型不适用；Prospect Ratio估计对策略模型空间的采样高度敏感；算法在高度动态或分布漂移极端的场景下尚未完全验证。

---

## 306. HAPS: Hierarchical LLM Routing with Joint Architecture and Parameter Search

**arXiv ID:** 2601.05903 | [PDF](https://arxiv.org/pdf/2601.05903v1)

**作者:** Zihang Tian `[一作]` (Renmin University of China), Xu Chen `[通讯]` (Renmin University of China)

**通讯引用:** 21266 | [OpenAlex ID](https://openalex.org/A5100385692)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种层次化的LLM路由框架HAPS，既动态选择模型架构又针对输入生成可调LoRA参数；

**💡 创新点**

创新点在于将离散的模型选择和连续的参数搜索联合优化，并通过共享高层与低层网络实现知识互补；

**🔧 技术方法**

主要技术包括基于LLaMA的多标签分类器（高层路由）、参数生成网络（低层路由）、LoRA参数生成、奖励增强的最大似然RL优化；

**📊 数据集**

使用HotpotQA（多跳推理）和MMLU（通用知识）两大公开基准；

**📈 对比分析**

与四个主流路由基线（Random、RouteLLM、GraphRouter、IRT‑Router）对比，HAPS在所有模型对上均达到或接近最高性能，尤其在混合源和成本平衡场景表现突出；

**⚠️ 局限性**

局限包括：仅在特定任务、模型池和成本设置下验证，奖励信号可能噪声大导致训练不稳定；参数高效微调虽然便捷但可能低于全参数微调的极限性能；

---

## 307. Cybersecurity AI: A Game-Theoretic AI for Guiding Attack and Defense

**arXiv ID:** 2601.05887 | [PDF](https://arxiv.org/pdf/2601.05887v1)

**作者:** Víctor Mayoral-Vilches `[一作]`, Cristóbal R. J. Veas Chavez `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为G‑CTR的游戏理论指导框架，利用LLM自动提取攻击图、计算纳什均衡，并生成策略摘要（digest），以闭环方式引导AI渗透测试与防御行为。

**💡 创新点**

创新点在于：①把LLM用于自动生成攻击图并构造“努力”评分取代传统概率；②将纳什均衡结果转化为可直接注入LLM系统提示的战略摘要；③实现攻击与防御双向协同的Purple merged团队，显著提升攻击成功率与防御效率。

**🔧 技术方法**

技术手段包括：LLM（Claude、ChatGPT等）、CAI渗透测试框架、G‑CTR（攻击图生成+纳什均衡计算）、Digest生成算法（算法式与LLM式两种模式）、ReAct代理以及对抗实验中的多智能体交互。

**📊 数据集**

数据集来源于五个真实网络安全演练的日志、44次Shellshock漏洞的Cyber‑Range评测以及两项Attack‑and‑Defense CTF（Cowsay、Pingpong），涵盖从日志文本到完整攻击图的转换。

**📈 对比分析**

与传统无指导LLM和手工生成攻击图的基线相比，G‑CTR在五个演练中实现了60‑245倍的时间加速、140‑450倍的成本降低，并在44次渗透测试中将成功率从13.3%提升至42.9%，成本每成功减少23倍，工具使用方差降低5.2倍；在红蓝对抗中Purple merged模式相较基线获胜比率提升至1.8:1，击败独立引导团队的3.7:1。

**⚠️ 局限性**

局限性包括：对LLM质量与提示工程的敏感性，可能仍存在幻觉；攻击图的节点数量与结构需人工调参；实验仅覆盖有限场景，尚未在更大规模或多样化网络环境中验证；以及对实时性能与安全性（如对抗攻击）的进一步评估尚待开展。

---

## 308. Multi-Modal Style Transfer-based Prompt Tuning for Efficient Federated Domain Generalization

**arXiv ID:** 2601.05955 | [PDF](https://arxiv.org/pdf/2601.05955v1)

**作者:** Yuliang Chen `[一作]` (Shanghai Jiao Tong University), Xiu Su `[通讯]` (Central South University)

**通讯引用:** 545 | [OpenAlex ID](https://openalex.org/A5011334334)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FaST-PT 框架，在联邦学习中利用 CLIP 实现多模态风格迁移与提示调优，以提升对未见域的泛化。

**💡 创新点**

创新点在于结合轻量化多模态风格迁移（MST）与双重提示模块（全局提示+域提示）及域感知提示生成，既实现局部特征增强，又兼顾跨域泛化，同时仅传输提示，降低通信与隐私泄露风险。

**🔧 技术方法**

使用 CLIP 预训练模型、MST 变换网络、提示调优、域分类器、对比损失等技术。

**📊 数据集**

在 PACS、VLCS、OfficeHome、DomainNet 四大跨域基准数据集上进行实验。

**📈 对比分析**

与 FedAvg、FedProx、ELCFS、FedSR、FedDG-GA、PromptFL、FedAPT、DiPrompt 等方法对比，FaST-PT 在 18 个任务中取得 17 项最佳，平均准确率分别为 PACS 97.32%、VLCS 99.31%、OfficeHome 99.93%、DomainNet 98.16%，显著优于 DiPrompt。

**⚠️ 局限性**

局限在于依赖 CLIP 预训练模型和文本描述，迁移到更大规模或更专业领域仍需进一步验证，且极少样本场景下性能仍受限。

---

## 309. Context-Aware Decoding for Faithful Vision-Language Generation

**arXiv ID:** 2601.05939 | [PDF](https://arxiv.org/pdf/2601.05939v1)

**作者:** Mehrdad Fazli `[一作]` (George Mason University), Ziwei Zhu `[通讯]` (George Mason University)

**通讯引用:** 1408 | [OpenAlex ID](https://openalex.org/A5019994221)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过层次级别的 Logit Lens 分析，揭示视觉语言模型在生成过程中真实与幻觉词语的承诺深度差异，并基于此提出 Context Embedding Injection（CEI）这一训练无关的解码干预方法，利用首词上下文嵌入持续引导生成，显著降低幻觉率。

**💡 创新点**

创新点在于：①首次将真实与幻觉词语的层级承诺差距作为可解释信号；②提出基于首词上下文嵌入的静态/动态注入机制，以持续保持视觉对齐；③通过动态注入根据 Top‑K 概率自适应调整注入强度，实现对长文本生成的实时纠偏。

**🔧 技术方法**

核心技术包括：Logit Lens 机制分析；Context Embedding Injection（CEI）的静态与动态实现；训练无关的解码时间干预；对比编码与注意力校准等现有无监督幻觉缓解方法。

**📊 数据集**

实验使用公开幻觉评测基准：CHAIR、AMBER 生成分支和 MMHal‑Bench（96 对对抗性图像-问题对）。

**📈 对比分析**

在三大基准上与 5 种无训练基线（VCD、AvisC、M3ID、OPERA、CAAC）对比，CEI（尤其是动态版本）在 CHAIR、AMBER、MMHal‑Bench 上均实现了最小幻觉率，并保持或提升覆盖率，整体性能优于或接近现有最优方案。

**⚠️ 局限性**

局限性包括：需要白盒访问模型内部隐藏状态和解码矩阵；动态 CEI 需每步进行额外前向传播，导致推理延迟增加；仅在公开基准上验证，未评估医疗、自动驾驶等专业领域；分析阶段依赖手工标注的真实/幻觉词标签。

---

## 310. Performance of a Deep Learning-Based Segmentation Model for Pancreatic Tumors on Public Endoscopic Ultrasound Datasets

**arXiv ID:** 2601.05937 | [PDF](https://arxiv.org/pdf/2601.05937v1)

**作者:** Pankaj Gupta `[一作]` (Postgraduate Institute of Medical Education and Research), Usha Dutta `[通讯]` (Postgraduate Institute of Medical Education and Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

评估基于Vision Transformer的深度学习分割模型在内镜超声图像中对胰腺肿瘤分割的性能，并在外部数据集上进行验证。

**💡 创新点**

首次将ViT与USFM框架结合用于胰腺超声图像分割，并在多中心公开数据上展示了良好的泛化能力。

**🔧 技术方法**

使用Vision Transformer（HVITBackbone4Seg）作为骨干网络，配合ATMHead解码器、ATMLoss损失、AdamW优化器和余弦学习率调度等技术。

**📊 数据集**

训练集为17,367张EUS图像（胰腺癌+GIST514-DB），外部验证集为350张来自LEP数据集的胰腺癌图像。

**📈 对比分析**

通过5折交叉验证和独立外部验证，主要指标为DSC≈0.657、IoU≈0.614、特异性≈97.7%、敏感性≈71.8%、准确率≈97.5%，整体表现稳健但仍出现少量多预测错误。

**⚠️ 局限性**

受限于数据集异质性、外部验证样本有限、无后处理步骤、标注主观性及未使用对比增强等因素，影响模型进一步提升与临床推广。

---

## 311. Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards

**arXiv ID:** 2601.06021 | [PDF](https://arxiv.org/pdf/2601.06021v1)

**作者:** Jiajie Zhang `[一作]` (Tsinghua University), Juanzi Li `[通讯]` (Tsinghua University)

**通讯引用:** 14263 | [OpenAlex ID](https://openalex.org/A5003324011)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种细粒度奖励框架CaRR和相应的RL算法C-GRPO，用以训练大语言模型驱动的深度搜索代理

**💡 创新点**

创新点在于通过将多跳问题拆解成单跳rubric，并以实体识别、引用验证和证据连贯性三步评估来生成上下文感知奖励，既补足单一结果奖励的不足，又避免了短路与幻觉

**🔧 技术方法**

使用LLM生成rubric与实体，LLM判别器进行评估；结合GRPO改进的C‑GRPO算法；工具链包括Serper、Jina和文本匹配工具；奖励体系融合结果奖励和rubric奖励

**📊 数据集**

训练数据为公开的DeepDive深度搜索数据集（1,016条SFT样本，2,234条RL样本）；评估基准包括BrowseComp、BrowseComp‑ZH、xbench‑DeepSearch和GAIA验证子集；开放式研究任务DeepResearch Bench也用于通用性验证

**📈 对比分析**

与GRPO、E‑GRPO以及其他SOTA代理对比，C‑GRPO在所有四大基准上都显著提升，4B模型平均提升5.1/8.0（64k/128k）点，30B模型提升2.6/6.0点；在开放式研究任务上甚至超过部分专有数据代理

**⚠️ 局限性**

局限在于rubric生成依赖合成多跳问题的结构，对开放式QA或无明确约束的问题适配性有限；若不具备实体与结构信息，框架难以直接迁移

---

## 312. LookAroundNet: Extending Temporal Context with Transformers for Clinically Viable EEG Seizure Detection

**arXiv ID:** 2601.06016 | [PDF](https://arxiv.org/pdf/2601.06016v1)

**作者:** Þór Sverrisson `[一作]` (University of Iceland), Steinn Guðmundsson `[通讯]` (University of Iceland)

**通讯引用:** 2015 | [OpenAlex ID](https://openalex.org/A5047864815)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出Transformer结构的LookAroundNet，用于自动癫痫发作检测，并通过引入目标段前后更宽的时间上下文来提升性能。

**💡 创新点**

创新点在于利用目标段的前后上下文窗口（look‑behind/ look‑ahead）以及多模型集成来模拟医生解读时的全局视角，并证明更长时间上下文与多源数据训练能显著提升跨数据集的泛化。

**🔧 技术方法**

技术包括Transformer编码器、通道级自注意力、卷积前置特征提取、位置编码、全连接分类层，以及数据预处理（长时双极 montage、滤波、重采样）。

**📊 数据集**

使用了多种公开数据集（TUSZ、Siena、SeizeIT1）和一大规模专有的Kvikna家庭监测数据，涵盖临床EEG、长期随访EEG等不同环境。

**📈 对比分析**

通过SzCORE框架的事件级与样本级评估，并与EEG‑U‑Transformer、EventNet等基准模型对比；在公开数据集上获得最高的事件级F1和较低的误报/天数，在Kvikna上表现虽稍逊但仍优于现有方法。

**⚠️ 局限性**

局限在于对短时发作（<8s）的检测率仍低，长时家庭记录中的工件仍导致误报，固定目标段长度限制了对变长发作的捕捉，需要进一步改进工件处理与动态窗口设计。

---

## 313. Adaptive Conditional Contrast-Agnostic Deformable Image Registration with Uncertainty Estimation

**arXiv ID:** 2601.05981 | [PDF](https://arxiv.org/pdf/2601.05981v1)

**作者:** Yinsong Wang `[一作]` (Imperial College London), Chen Qin `[通讯]` (Imperial College London)

**通讯引用:** 3449 | [OpenAlex ID](https://openalex.org/A5100362874)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种适应性条件对比无关可变形图像配准框架 AC‑CAR，能够在训练时仅使用单一对比度图像，推理时对任意未见对比度进行配准并估计配准不确定性。

**💡 创新点**

创新点包括：①使用随机卷积生成任意对比度的增强样本；②引入适应性条件特征调制（ACFM）和对比度不变潜在正则化（CLR）联合学习对比度无关特征；③集成异方差不确定性网络实现对配准误差的置信度估计。

**🔧 技术方法**

采用技术包括：随机卷积对比度增强、离散小波变换+条件实例归一化、潜在空间对比度不变损失、β‑NLL 置信度训练、U‑Net 编码/解码结构、LNCC 相似度、Jacobian 正则化等。

**📊 数据集**

使用数据集：CamCAN（T1w/T2w）、IXI（T1w/T2w/PD）、CMRxRecon（T1 映射序列）、Learn2Reg 腹部 CT‑MR，涵盖 3D 脑 MRI、2D 心脏 MRI 及跨模态 CT‑MR。

**📈 对比分析**

与 SyN、VXM‑LNCC/MIND、MIDIR、SynthMorph、CAR、OTMorph、UTSRMorph 等基线比较，AC‑CAR 在 Dice、HD95、J_<0% 及 |∇J| 等指标上均优于现有方法，并在未见对比度和多模态任务中表现出更好的泛化能力；推理时间保持在 2‑3 秒左右。

**⚠️ 局限性**

局限性：ACFM 在 3D 任务中仅使用 2D 切片进行条件，可能无法完整捕捉体积对比度变化；对比度增强仅模拟对比度差异，尚未覆盖更广泛的多模态差异（如 MR‑CT、US‑CT），未来需要进一步扩展。

---

## 314. VideoAR: Autoregressive Video Generation via Next-Frame & Scale Prediction

**arXiv ID:** 2601.05966 | [PDF](https://arxiv.org/pdf/2601.05966v1)

**作者:** Longbin Ji `[一作]` (Baidu), Haifeng Wang `[通讯]` (Baidu)

**通讯引用:** 20460 | [OpenAlex ID](https://openalex.org/A5100386408)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种大规模的视觉自回归（VAR）视频生成框架 VideoAR，该框架结合多尺度下一个帧预测与自回归建模，并通过 3D 因果分词器将视频压缩为离散时空表示，随后使用 Transformer 逐帧生成残差。

**💡 创新点**

创新点包括：
1) 将 VAR 框架从图像迁移到视频，采用多尺度残差预测与跨帧错误继承机制，实现空间与时间的解耦。
2) 设计 3D 因果分词器与多尺度 Tokenizer，支持长时序、分块推理。
3) 引入多尺度时间 RoPE 以及时间‑依赖噪声扰动（Time‑Dependent Corruption）和错误继承（Error Inheritance）训练策略，显著缓解长序列误差累积。
4) 采用可调节的时空自适应 CFG 与帧重编码，提升文本对齐与视频时序可控性。
5) 构建多阶段预训练管线，逐步提升分辨率与时长，从而实现高质量长时序视频生成。

**🔧 技术方法**

技术栈：
- 视觉自回归（VAR）+ Transformer 基础网络
- 3D 因果卷积编码器‑解码器与多尺度 Tokenizer
- 多尺度时间 RoPE（三维位置编码）
- 跨帧错误纠正（Cross‑Frame Error Correction）与时间噪声扰动
- 时空自适应无分类器引导（Temporal‑Spatial Adaptive CFG）
- 多阶段预训练策略（Stage I‑III）
- 训练损失包括重构、感知、GAN、commitment 与 entropy 结合
- 评估指标 FVD、gFVD、VBench 等

**📊 数据集**

使用数据集：
- UCF‑101（人类动作短视频）用于 gFVD 评估
- VBench（公开长时序多域视频基准）用于整体一致性、语义等多维度评分
- 公开 8K 视频集（含 101 个动作类别）作为低分辨率训练与预训练
- 自研大规模真实视频数据集（高分辨率、长时序）用于最终 4B 模型的预训练与评估
- 低分辨率图像与视频混合数据用于 Stage I 预训练
- 高分辨率图像与视频混合数据用于 Stage II
- 长时序视频数据集用于 Stage III

**📈 对比分析**

与竞争方法对比：
- 与扩散模型（CogVideo、Step‑Video、CogVideoX 等）比较：VBench 总分 81.74 与 30B/13B 模型相当，且参数量显著更小。
- 与自回归基准（PAR‑4x、MAGVIT‑v2‑AR、OmniTokenizer 等）比较：gFVD 88.6（2B 参数）/90.3（926M 参数），分别比 PAR‑4x 提升 11% 与 4%；推理步数仅 30 步，比 PAR‑4x 低 10×，推理速度提升 13×。
- 在 UCF‑101 上实现 gFVD 88.6，击败之前最优 AR 99.5，且仅需 0.86 s 推理时间。
- 在 VBench 语义、审美、一致性等子指标上均取得领先或接近最优，显示出优秀的文本‑视频对齐与时序一致性。

**⚠️ 局限性**

局限与未来工作：
- 训练阶段使用的序列长度相对有限，尚未在极长时序（>20 s）上系统验证，长程一致性可能仍需改进。
- 由于 3D 因果结构，模型在极高分辨率或长帧率视频上对硬件资源有一定压力。
- 在某些细粒度视觉细节上仍略逊于大型扩散模型，尤其是对极端纹理或细节的再现。
- 公开数据集与自研数据的偏差可能导致模型在更广泛领域的泛化能力尚未完全体现。
- 需要进一步研究更高效的 Tokenizer 与更深层次的跨帧注意机制，以进一步降低误差累积与提升长程稳定性。

---

## 315. On the Robustness of Age for Learning-Based Wireless Scheduling in Unknown Environments

**arXiv ID:** 2601.05956 | [PDF](https://arxiv.org/pdf/2601.05956v1)

**作者:** Juaren Steiger `[一作]` (Pennsylvania State University), Bin Li `[通讯]` (Pennsylvania State University)

**通讯引用:** 25553 | [OpenAlex ID](https://openalex.org/A5100365212)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于首位包年龄（Head‑of‑Line Age）的学习式无线调度策略，用以在未知环境下实现带约束的组合多臂赌博机（CMAB）问题的调度，取代传统的基于队列长度的虚拟队列方法；

**💡 创新点**

创新点在于将首位包年龄引入虚拟包调度并与UCB相结合，既能在i.i.d.条件下达到与现有最优算法相同的无约束违例性能，又在通道骤变导致约束暂时不可行时表现出更强的鲁棒性与快速恢复；

**🔧 技术方法**

使用的技术包括CMAB建模、UCB估计、首位包年龄动态分析、Lyapunov漂移与马氏指数方法（drift‑plus‑regret）以及对虚拟队列与首位包年龄的概率分析；

**📊 数据集**

实验数据集主要有：1）合成6臂示例（每臂单一选择）；2）真实无线网络SNR轨迹（30个通道中选取12、16、17通道，采样3×10⁴时隙）；

**📈 对比分析**

与三种现有策略（基于队列长度、TSLR以及两者组合）进行比较。理论上在i.i.d.情况下，age‑based策略在窗口大小为O(η/ε)时实现零约束违例，regret为O(εT+T/η+√(TlogT))；实验中，age‑based在i.i.d.环境下性能居中，但在通道突变（短期约束不可行）情形下恢复最快，短期吞吐量与信息新鲜度均优于队列长度策略；

**⚠️ 局限性**

局限性包括：1）对非i.i.d.、暂时约束不可行时的鲁棒性只能通过实验验证，缺乏严谨理论证明；2）算法需要调参（η、ε、窗口W）且分析复杂；3）仅针对单通道选择或有限干扰模型，未覆盖更通用的多用户/多干扰场景。

---

## 316. Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency

**arXiv ID:** 2601.05905 | [PDF](https://arxiv.org/pdf/2601.05905v1)

**作者:** Haoming Xu `[一作]` (Zhejiang University), Ningyu Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 4393 | [OpenAlex ID](https://openalex.org/A5089259739)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Neighbor-Consistency Belief (NCB) 指标、认知压力测试协议以及结构感知训练 (SAT)，用于评估与提升 LLM 的信念鲁棒性。

**💡 创新点**

创新点在于将信念视作结构化状态，构造 NCB 量化模型在概念邻域中的一致性，并设计社会压力与权威干扰的压力测试，进一步通过 SAT 让模型在多种上下文中保持一致的推理。

**🔧 技术方法**

采用贝叶斯启发的后验估计、邻域一致性聚合、链式思考与反思策略、KL 散度训练等技术，并结合多代理与来源可信度模拟的干扰。

**📊 数据集**

构建 Neighbor‑Enriched 数据集（来源于 SimpleQA、HotpotQA、SciQ），每条事实附带约 8 条邻域事实和 5 条误导邻域事实，并采集 100 条新事实用于 SAT 评估。

**📈 对比分析**

在自洽度高的数据子集上按 NCB 取高低分层，实验显示高 NCB 组的误差下降幅度比低 NCB 组小 10–20%，SAT 使新学知识的准确率达 93% 且在压力测试中平均降幅约 30%。

**⚠️ 局限性**

局限包括仅针对时间不变的静态事实、邻域关系类型有限、构建邻域需要人工验证、缺乏人类对“真实理解”的评估、以及计算开销大和潜在误用风险。

---

## 317. Spectral Clustering in Birthday Paradox Time

**arXiv ID:** 2601.05883 | [PDF](https://arxiv.org/pdf/2601.05883v1)

**作者:** Michael Kapralov `[一作]` (École polytechnique fédérale de Lausanne), Weronika Wrzos-Kaminska `[通讯]` (École polytechnique fédérale de Lausanne)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种本地聚类算法，能够在仅获得图中某一顶点及其局部邻域信息的前提下，将含有k个大规模连通分支的图划分为聚类并给出聚类数近似；

**💡 创新点**

创新点在于：①提出本地谱嵌入表示并利用多项式滤波（构造多项式p(x)）逼近前k个特征向量；②设计了“well‑spread”样本集合与“典型点”概念，通过随机符号投影与碰撞计数实现聚类检索；③实现了O(√(n/k))的时间复杂度和近似聚类数的多项式误差；

**🔧 技术方法**

使用了随机游走、随机符号投影、碰撞计数、谱多项式逼近（利用Chebyshev/Weierstrass多项式）、概率上界（Markov、Chernoff、Chernoff bounds）、组合论与随机采样等技术；

**📊 数据集**

该工作为理论算法，不涉及具体实验数据集，所有结果均为理论上限与期望分析；

**📈 对比分析**

与以往基于本地搜索或全局特征子空间的聚类方法相比，时间复杂度显著降低（从O(nk)降至O(√(nk)·poly(ϵ/φ^2) )），聚类误差率为(ϵ/φ^2)^{1/3}，同时能在多项式时间内估计聚类数；

**⚠️ 局限性**

局限性包括：①算法对参数ϵ/φ^2的大小有限制（需小于1/lnk）；②需知道或估计k的下界；③对非常稀疏或度极高的图可能不适用；④在极小ϵ下多项式系数仍为高阶，实际运行时常可能膨胀；

---

## 318. GlueNN: gluing patchwise analytic solutions with neural networks

**arXiv ID:** 2601.05889 | [PDF](https://arxiv.org/pdf/2601.05889v1)

**作者:** Doyoung Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jaeok Yi `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 26 | [OpenAlex ID](https://openalex.org/A5038101150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了GlueNN框架，利用神经网络学习不同区域解析解的系数，使得整个域内的解可以光滑地拼接而不需要手工匹配；

**💡 创新点**

创新点在于将解析解中的积分常数提升为可学习的尺度相关函数，并通过结合数据拟合、残差约束以及patch抑制的复合损失实现无手工匹配的全局解；

**🔧 技术方法**

使用了基于PINN的头‑干预（head‑trunk）网络结构，配合复合MSE损失函数，对系数函数进行训练；

**📊 数据集**

采用化学反应（Ricatti方程）与宇宙学向量粒子产生问题的数值解作为训练与验证数据集，参数范围分别为化学反应中η=10⁴以及宇宙学模型中H=150、k=2.0、m=0.10；

**📈 对比分析**

与传统C⁰或C¹手工匹配方法比较，GlueNN在全域解的误差上显著更小，能够在训练数据之外的区域保持准确，并且对匹配点的选择不敏感；

**⚠️ 局限性**

限制包括需要先知晓解析解的形式和系数数目、损失权重需手工调参、对网络结构和正则化敏感，以及在极为复杂或多尺度问题中训练仍可能面临困难。

---

## 319. The Causal Effect of First-Time Academic Failure on University Dropout: Evidence from a Regression Discontinuity Design

**arXiv ID:** 2601.05987 | [PDF](https://arxiv.org/pdf/2601.05987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 320. Database Theory in Action: Direct Access to Query Answers

**arXiv ID:** 2601.06013 | [PDF](https://arxiv.org/pdf/2601.06013v1)

**作者:** Jiayin Hu `[一作]` (University of California Santa Cruz), Nikolaos Tziavelis `[通讯]` (University of California Santa Cruz)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了针对联接查询（CQs）在词典序或求和序上的高效直接访问与单次访问算法，支持所有可行的时间复杂度场景；

**💡 创新点**

首次提供完整的、可自动判定可行性的实现，覆盖所有理论上可行的直接访问与单次访问算法，并允许用户自定义排序；

**🔧 技术方法**

采用数据库理论中的准线性预处理与对数级访问算法、Top‑N 堆排序、sort‑before‑join、CTE 与窗口函数等技术，并在GitHub公开实现；

**📊 数据集**

使用合成数据对三路联接 R⋈S⋈T 进行实验，分别采用 Uniform、Large join result 与 Small join result 分布，关系规模从 10^4 到 10^6；

**📈 对比分析**

与 PostgreSQL 17.4 的标准、Top‑N heapsort 与 sort‑before‑join 策略对比，实验显示在大结果集或非小 k 的场景下直接访问明显快于数据库策略；与单次访问比较时，直接访问在仅需 1~2.4 次访问后即可超过单次访问性能；

**⚠️ 局限性**

尚未与数据库引擎深度集成，只针对 CQs，且仅在合成数据上验证，常数因子和实际数据分布可能影响性能交叉点；

---

## 321. The Importance of Parameters in Ranking Functions

**arXiv ID:** 2601.06001 | [PDF](https://arxiv.org/pdf/2601.06001v1)

**作者:** Christoph Standke `[一作]` (RWTH Aachen University), Benny Kimelfeld `[通讯]` (Technion)

**通讯引用:** 2761 | [OpenAlex ID](https://openalex.org/A5006706357)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

对不同排名函数和影响函数下，计算列权重的 SHAP 分数和列的 Shapley 值的计算复杂度进行系统分析。

**💡 创新点**

首次给出这些问题的多项式可解与 #P‑难的完整划分，并证明 SHAP 分数可与期望效果等价，从而把解释任务转化为经典计数问题。

**🔧 技术方法**

使用多项式时间动态规划、事件分解、线性期望展开以及从背包问题、正 CNF 计数等经典 #P‑难问题的多项式约简来证明复杂度。

**📊 数据集**

本文为理论研究，不涉及具体实验数据集，所有结果均在抽象矩阵与分布模型上证明。

**📈 对比分析**

对可解的情形给出多项式时间算法；对不可解的情形给出 #P‑硬性证明；此外提出可在期望效应上实现加法 FPRAS 的采样近似方案。

**⚠️ 局限性**

仅适用于基础的排名（Sum、Max、Lex）和影响函数（Kendall’s τ、MD、位置、top‑k 成员等）；假设参数独立且数值以二进制/一进制编码，未覆盖通用分布、非独立参数或多重近似等更广泛场景。

---

## 322. CyberGFM: Graph Foundation Models for Lateral Movement Detection in Enterprise Networks

**arXiv ID:** 2601.05988 | [PDF](https://arxiv.org/pdf/2601.05988v1)

**作者:** Isaiah J. King `[一作]` (Cybermonic LLC), H. Howie Huang `[通讯]` (Cybermonic LLC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图的语言模型（Graph Foundation Model），利用随机游走生成句子，结合BERT的mask预训练和微调实现企业网络侧移检测中的链路预测与异常检测。

**💡 创新点**

创新点在于将传统的随机游走+Word2Vec思路升级为Transformer级别的预训练模型，并在无监督的图数据上构建基础模型，显著提升侧移检测的精度与效率。

**🔧 技术方法**

使用BERT Tiny Transformer、Scheduled Masked Token Prediction、随机游走采样（静态与时间偏置）、边特征token化、链路预测或分类微调等技术。

**📊 数据集**

在OpTC、UNSW‑NB15和LANL这三个公开网络安全数据集上进行实验。

**📈 对比分析**

与N2V、Pikachu、Euler、Argus四种基准模型对比，CyberGFM在三大数据集上平均精度(AP)提升约2倍，AUC和AP均超过所有对手，表现最优。

**⚠️ 局限性**

局限性包括：模型非增量/非归纳，无法处理未出现的节点；训练与推理时间相对较长；推理时需重新采样随机游走，影响实时性。

---

## 323. Age of Gossip With Cellular Drone Mobility

**arXiv ID:** 2601.05983 | [PDF](https://arxiv.org/pdf/2601.05983v1)

**作者:** Arunabh Srivastava `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 13727 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在蜂窝网络中，移动无人机与节点内部推式信息传播相结合的版本信息新鲜度问题。

**💡 创新点**

创新点在于提出双瓶颈理论，揭示无人机移动速率与信息传播速率哪个更主导，并给出两种速率比下的版本信息新鲜度上界。

**🔧 技术方法**

采用连续时间马尔可夫链（CTMC）、相位类型分布、SHS框架以及切比雪夫不等式进行理论推导。

**📊 数据集**

未使用真实数据集，全部为理论分析与仿真验证。

**📈 对比分析**

通过仿真对比理论上界与下界，验证在大规模网络中两者吻合，性能随节点数n和单元数f(n)按预期增长。

**⚠️ 局限性**

局限在于对一般CTMC下的高概率分析困难，仅在完全连通移动模型给出精确方差；低速移动时高方差影响评估不充分。

---

## 324. WaveRNet: Wavelet-Guided Frequency Learning for Multi-Source Domain-Generalized Retinal Vessel Segmentation

**arXiv ID:** 2601.05942 | [PDF](https://arxiv.org/pdf/2601.05942v1)

**作者:** Chanchan Wang `[一作]` (Xinjiang University), Guanxin Chen `[通讯]` (Xinjiang University)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5025438616)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出WaveRNet框架，利用波形变换与频域学习实现多源域泛化的视网膜血管分割，并在SAM的提示式分割基础上进行细化。

**💡 创新点**

创新点：①Spectral-guided Domain Modulator（SDM）将自学习的低频/高频分解与域标记融合，显式区分照明鲁棒与对比敏感特征；②Frequency-Adaptive Domain Fusion（FADF）通过频域相似度进行推理时域选择与软加权融合；③Hierarchical Mask-Prompt Refiner（HMPR）采用分层掩模提示和长程注意力逐步恢复细小血管，克服SAM直接上采样导致的细节丢失。

**🔧 技术方法**

技术：Wavelet变换（自学习低/高频分支）、SAM的ViT-B编码器、轻量级适配器、频域相似度投影、软加权融合、双阶段解码器与注意力机制。

**📊 数据集**

使用四个公开视网膜数据集：DRIVE、STARE、CHASE_DB1、RECOVERY-FA19（含FA图像）。

**📈 对比分析**

在单域训练下与传统U-Net系列、SAM、MedSAM等方法比较，取得最高Dice；在Leave-One-Domain-Out泛化任务中平均Dice 69.49%，比最优SAM-Med2D提升约9.4%，比最佳U-Net提升约28.8%，尤其在RECOVERY-FA19上显著提升。

**⚠️ 局限性**

局限性：仍需显式域标签来构建域标记，未实现无监督域发现；推理时涉及多步融合与双阶段解码，计算量相对较大；仅在视网膜血管任务验证，其他医学影像领域待进一步扩展。

---

## 325. Can We Predict Before Executing Machine Learning Agents?

**arXiv ID:** 2601.05930 | [PDF](https://arxiv.org/pdf/2601.05930v1)

**作者:** Jingsheng Zheng `[一作]` (Zhejiang University), Ningyu Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 4393 | [OpenAlex ID](https://openalex.org/A5089259739)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种“预测-验证”循环的自主机器学习代理，利用内部执行先验通过LLM进行数据中心的解决方案偏好预测，从而显著减少物理执行时间。

**💡 创新点**

提出了Data‑centric Solution Preference任务及其18,438对比样本的语料库，并将LLM作为隐式世界模型实现即时推理，突破传统生成‑执行‑反馈循环的执行瓶颈；同时构建了“Profile‑Verify‑Verbalize”管道生成已验证的数据分析报告。

**🔧 技术方法**

使用DeepSeek‑V3.2‑Thinking与GPT‑5.1等LLM进行因果推理；通过“Profile‑Verify‑Verbalize”生成已验证的数据分析报告；构建Predict‑then‑Verify循环；在实验中与随机与复杂度启发式基线进行对比。

**📊 数据集**

18,438对比样本的Data‑centric Solution Preference语料库，来源于MLE‑bench上AIDE与AutoMind产生的1,329个完整ML解法，经过专家筛选后再扩展为所有对组合。

**📈 对比分析**

与随机基线（50.0%）和复杂度启发式（50.8%）相比，DeepSeek‑V3.2‑Thinking达到61.5%准确率；在自定义代理中实现6倍速度提升、+6%性能提升、搜索空间扩展3.2倍。

**⚠️ 局限性**

语料库样本不均衡，部分专业领域样本稀缺；代理仅实现基本Predict‑then‑Verify，未充分探索更高级推理与参数调优；LLM推理受验证‑测试差距限制，易受分布偏移影响。

---

## 326. Adapting Vision Transformers to Ultra-High Resolution Semantic Segmentation with Relay Tokens

**arXiv ID:** 2601.05927 | [PDF](https://arxiv.org/pdf/2601.05927v1)

**作者:** Yohann Perron `[一作]` (Ecole francaise d’Extreme-Orient), Loic Landrieu `[通讯]` (Ecole nationale des Ponts et Chaussees)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Relay Token机制，使标准Vision Transformer能够在保持预训练权重的同时，跨尺度传递信息，实现高分辨率细节与全局上下文的并行处理。

**💡 创新点**

通过少量可学习的relay token在局部高分辨率窗口和全局低分辨率窗口之间递归传递特征，解决了传统滑窗缺失全局上下文和降采样失去细节的问题。

**🔧 技术方法**

基于ViT、Swin、GLAM等Transformer骨干，添加relay token并联合使用局部/全局窗口、交叉尺度损失与一致性损失，实现多尺度信息融合。

**📊 数据集**

在三大UHR分割基准（Archaeoscape、URUR、Gleason）以及经典Cityscapes数据集上进行评估。

**📈 对比分析**

相较于滑窗、降采样、线性注意力以及专用多尺度网络，Relay Token在保持参数增幅<2%、显著降低GPU显存需求的同时，平均提升mIoU 5–15个百分点，部分场景达到SOTA。

**⚠️ 局限性**

对极大全局窗口的计算成本和对token数量的敏感性仍有限制，过大或过小的全局尺度会导致收益递减，且在极低分辨率或极细粒度对象上提升有限。

---

## 327. TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents

**arXiv ID:** 2601.05899 | [PDF](https://arxiv.org/pdf/2601.05899v1)

**作者:** Dawei Wang `[一作]` (Newcastle University), Richard Davison `[通讯]` (Newcastle University)

**通讯引用:** 1328 | [OpenAlex ID](https://openalex.org/A5109112867)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了轻量级塔防类RTS环境 TowerMind，并在其中评估 LLM 的长期规划与决策能力，同时提供多模态观测与幻觉评估。

**💡 创新点**

①提供低计算需求、支持文本、像素和结构化观测的RTS环境；②引入幻觉评价机制和可定制关卡；③兼容 LLM 与 RL 的统一基准。

**🔧 技术方法**

使用 Unity 引擎 + ML-Agents、OpenAI Gym 接口，零样本 LLM 推理，结合 Ape‑X DQN 与 PPO 等 RL 算法。

**📊 数据集**

自建的 5 级塔防 benchmark，包含文本、像素、结构化观测；评测模型包括 GPT‑4.1、Gemini‑2.5‑Pro、Claude 3.7 Sonnet、Llama 3.2、Qwen2.5‑VL；并对比人类专家基准。

**📈 对比分析**

通过得分与有效动作率两指标与人类专家对比，LLM 在视听模式下有所提升，但整体仍落后人类约58‑62%；RL 算法虽能解决简单关卡，但整体表现远逊于人类。

**⚠️ 局限性**

LLM 缺乏空间推理、多重终点决策与对动作空间的充分利用；幻觉率随关卡难度升高；环境未包含对手，评估对抗情境受限；评测仅采用零样本提示，未充分挖掘模型潜能。

---

## 328. The Modal Logic of Abstraction Refinement

**arXiv ID:** 2601.05897 | [PDF](https://arxiv.org/pdf/2601.05897v1)

**作者:** Jakob Piribauer `[一作]` (Technische Universitat Dresden), Vinzent Zschuppe `[通讯]` (Technische Universitat Dresden)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文通过引入模态逻辑与CTL的结合，研究了在系统抽象细化过程中CTL属性真值的变化规律，并提出了抽象细化的模态逻辑（MLAR）框架。

**💡 创新点**

创新点在于提出了控制语句（如纯按钮、弱按钮、开关、受限开关与决策）作为证明MLAR上下界的工具，首次将这些控制语句与抽象细化的偏序结构结合，得到S4.2、S4.2.1和S4FPP等新型模态逻辑。

**🔧 技术方法**

主要技术包括CTL抽象化、模态逻辑语义、F‑labeling构造、预布尔代数与倒置棒形（inverted lollipop）框架、有限偏函数偏序（FPF poset）以及对控制语句的独立性分析。

**📊 数据集**

由于研究是理论性的，本文没有使用任何实验数据集，而是通过构造特定的抽象系统来演示控制语句的实现与逻辑等价性。

**📈 对比分析**

通过证明上下界，作者展示了S4.2和S4.2.1的PSPACE复杂度，并与传统只满足ACTL的CEGAR方法进行对比，证明了在更一般的分支时序属性上抽象细化的适用性与限制。

**⚠️ 局限性**

局限性包括：仅考虑CTL（或更强的CTL*）的可表达属性，对所有系统的精确逻辑仍未完全确定；S4FPP的有限可算性未解决；以及缺乏经验验证与工具实现。

---

