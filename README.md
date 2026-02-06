# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-06 | 今日论文总数: 616

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. From Fragmentation to Integration: Exploring the Design Space of AI Agents for Human-as-the-Unit Privacy Management

**arXiv ID:** 2602.05016 | [PDF](https://arxiv.org/pdf/2602.05016v1)

**作者:** Eryue Xu `[一作]` (Northeastern University), Tianshi Li `[通讯]` (Northeastern University)

**通讯引用:** 1328 | [OpenAlex ID](https://openalex.org/A5039928918)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过访谈发现用户跨应用隐私管理痛点，设计并评估了九种基于人工智能代理的隐私管理工具。

**💡 创新点**

首次从人类“整体单元”视角探讨跨境隐私管理，并提出以后期管理为主的后共享 AI 代理方案。

**🔧 技术方法**

采用对话式 LLM 代理（OpenAI LLM）进行知识提取、策略建议与自动化操作，辅以 GPT‑4o‑mini 生成图示。

**📊 数据集**

使用12名美国参与者的半结构化访谈数据和116名参与者的速配式问卷数据。

**📈 对比分析**

通过 Plackett‑Luce 排序与相对效用评分，发现 Digital Identity Manager、Dynamic Privacy Preference Agent、History Sweeper 三个方案被最高评价，表现优于传统预共享控制工具。

**⚠️ 局限性**

样本规模有限、仅覆盖美国技术熟练用户、仅聚焦移动环境，且缺乏真实系统实现与长期用户实验。

---

## 2. StagePilot: A Deep Reinforcement Learning Agent for Stage-Controlled Cybergrooming Simulation

**arXiv ID:** 2602.05060 | [PDF](https://arxiv.org/pdf/2602.05060v1)

**作者:** Heajun An `[一作]` (Virginia Tech), Jin-Hee Cho `[通讯]` (Virginia Tech)

**通讯引用:** 5693 | [OpenAlex ID](https://openalex.org/A5011649304)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 StagePilot，一个基于离线强化学习的对话代理，用于模拟网络诱拐的分阶段对话流程，帮助青少年进行预防教育。

**💡 创新点**

创新点在于：①将对话转化为阶段级决策问题，利用相邻阶段约束实现可解释、非线性进展；②设计情绪+距离双重奖励以平衡情绪共鸣与目标推进；③构建完整的模拟与评估框架，验证离线 RL 在安全关键对话中的有效性。

**🔧 技术方法**

使用了：离线强化学习（IQL+AWAC、CQL、BC），Transformer 编码器（RoBERTa、DistilRoBERTa、DeBERTa）做阶段判别；LLM（Mistral 7B Instruct）作为环境生成器；情绪分类器（RoBERTa）提取受害者情绪。

**📊 数据集**

主要数据集为 Perverted‑Justice（PJ）聊天记录，经过 GPT‑4o / Claude 3.5 Sonnet 复核并标注六阶段标签，构成约 1.12M 条记录的训练集。

**📈 对比分析**

与 Prompt‑Engineering、BC、CQL、IQL+AWAC 进行对比；IQL+AWAC 在 Stage‑6 达成率 95%/成功终止 91%，情绪正向率 70.6%，对话长度最短（约 72 轮），显著优于其他方法。

**⚠️ 局限性**

局限性包括：①完全基于 LLM 生成的模拟，缺乏真实人类交互验证；②对情绪判别依赖单一模型，可能受噪声影响；③对阶段标签的人工标注仍有一定误差；④模型对长周期、非线性真实诱拐行为的适应性有限。

---

## 3. VEXA: Evidence-Grounded and Persona-Adaptive Explanations for Scam Risk Sensemaking

**arXiv ID:** 2602.05056 | [PDF](https://arxiv.org/pdf/2602.05056v1)

**作者:** Heajun An `[一作]` (Virginia Tech), Jin-Hee Cho `[通讯]` (Virginia Tech)

**通讯引用:** 5693 | [OpenAlex ID](https://openalex.org/A5011649304)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 VEXA 框架，为网络钓鱼与诈骗信息生成以模型证据为根基、符合用户个体风险感知的自然语言解释。

**💡 创新点**

创新点在于将 GradientSHAP 产生的特征重要性与基于 Big Five 的“脆弱性人设”相结合，实现解释内容的忠实根基与呈现方式的可定制化，突破了仅靠 LLM 生成、缺乏可信度的局限。

**🔧 技术方法**

技术组合包括：BERT‑based 效检器、GradientSHAP 归因模块、基于人格维度的人设筛选与指令生成、以及以人设条件提示的 LLM 生成器。

**📊 数据集**

实验使用了覆盖 Email、SMS、社交媒体的多渠道公开数据集：Ling‑Spam、Enron‑Spam、Human–LLM Phishing Email、AI‑Generated Email、Super SMS、UCI SMS Spam、NUS SMS、SpamHunter、Social Honeypot、UTKML Twitter 等。

**📈 对比分析**

与单纯 LLM、仅 XAI、XAI+高脆弱性、XAI+低脆弱性四种设置对比，自动指标显示 XAI grounding 的 Faithfulness ≈0.75、Correctness ≈0.59，且无显著提升 FKGL；高/低脆弱性人设分别在 FKGL（12.0 vs 14.4）和 Correctness（0.54 vs 0.64）上展示可解释的差异。

**⚠️ 局限性**

局限包括：人设仅采用二分高/低简化，缺少细粒度；实验仅基于自动指标与两位专家评审，未验证对真实用户理解与决策的影响；XAI 与专家手工挑选的关键字仍可能存在差异。

---

## 4. SemPipes -- Optimizable Semantic Data Operators for Tabular Machine Learning Pipelines

**arXiv ID:** 2602.05134 | [PDF](https://arxiv.org/pdf/2602.05134v1)

**作者:** Olga Ovcharenko `[一作]` (BIFOLD and TU Berlin), Sebastian Schelter `[通讯]` (BIFOLD and TU Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出 SemPipes，一种通过语义数据操作符来描述并优化表格机器学习流水线的声明式编程模型，并在训练时使用大语言模型生成针对数据的代码实现。

**💡 创新点**

创新点在于将 LLM 驱动的代码合成与树形进化搜索结合，实现在流水线中的数据操作可被自动优化，同时保持在推理时无需调用 LLM。

**🔧 技术方法**

核心技术包括大语言模型（GPT‑4、Gemini、Qwen‑coder）进行代码合成、数据上下文感知的语义操作符、基于树结构的进化搜索（MCTS 等）以及基于 skrub 的 DataOps 框架。

**📊 数据集**

实验使用来自 NeurIPS、EMNLP、SIGMOD 编程赛、Kaggle 竞赛的 19 条专家与自动生成的流水线，涉及多表、文本、图像等多模态表格数据。

**📈 对比分析**

与原始专家流水线、自动生成流水线以及专门方法（如 CAAFE、Palimpzest、LOTUS）比较，SemPipes 在 17/19 例子中提升性能，优化后大多能超过原始流水线，且在特征工程、缺失值插补和零样本特征提取任务中均表现出与或优于专业基线。

**⚠️ 局限性**

局限性包括目前不支持时间序列或视频数据、合成代码存在安全风险、以及需要在安全沙箱中运行，且模型优化主要关注预测准确性，未同时考虑执行时间或成本。

---

## 5. CoWork-X: Experience-Optimized Co-Evolution for Multi-Agent Collaboration System

**arXiv ID:** 2602.05004 | [PDF](https://arxiv.org/pdf/2602.05004v1)

**作者:** Zexin Lin `[一作]` (AutoGame Research), Xiaoqiang Ji `[通讯]` (School of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CoWork-X 框架，实现多智能体协作的主动共进化。

**💡 创新点**

创新点：将协作建模为跨回合闭环优化，使用 HTN 结构化技能库和后置 LLM 补丁，既保证子秒级实时协作，又实现持续提升。

**🔧 技术方法**

使用 HTN 规划、Python 代码生成与调试、LLM（如 Gemini）进行后期补丁、离线诊断与更新。

**📊 数据集**

使用 Overcooked-AI 类实时协作基准（多菜肴烹饪任务）进行评估。

**📈 对比分析**

与 ReAct、Reflexion、DPT‑WToM 等基线对比，CoWork-X 在 30 回合平均回报从 52 提升到 96，在线延时仅 2.6 s，token 0，显著优于基线。

**⚠️ 局限性**

局限性：对 LLM 补丁的正确性与安全性依赖手动/自动验证；对环境或队友变化的泛化能力有限；补丁过程可能引入不可预料的错误。

---

## 6. Euphonium: Steering Video Flow Matching via Process Reward Gradient Guided Stochastic Dynamics

**arXiv ID:** 2602.04928 | [PDF](https://arxiv.org/pdf/2602.04928v1)

**作者:** Ruizhe Zhong `[一作]` (Shanghai Jiao Tong University), Junchi Yan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 16614 | [OpenAlex ID](https://openalex.org/A5087158377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将过程奖励梯度嵌入流匹配模型的SDE框架，对视频生成进行后训练对齐。

**💡 创新点**

创新点在于：①构建Reward-Gradient Guided Dynamics，使采样过程主动向高奖励区域引导；②设计无梯度推理的蒸馏方法，将奖励引导信息内化到流网络；③引入双重奖励策略，结合潜在过程奖励与像素级结果奖励。

**🔧 技术方法**

采用流匹配、非平衡传输采样 (NETS)、SDE、GRPO、潜在奖励模型 (PRM)、像素级奖励模型 (ORM) 等技术。

**📊 数据集**

奖励模型在约20k提示生成的200k视频样本上训练；在 VBench2 评测集上进行性能对比。

**📈 对比分析**

相较于基线 Flow‑GRPO、Dance‑GRPO，方法在 VBench2 上获得 54.24 的总分（比基线提升约 3 分），并将收敛速度提升至 1.66 倍。

**⚠️ 局限性**

依赖特定 VAE 的潜在空间奖励模型，鲁棒性和跨模型迁移性受限。

---

## 7. CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs

**arXiv ID:** 2602.05258 | [PDF](https://arxiv.org/pdf/2602.05258v1)

**作者:** Haoran Li `[一作]` (Carnegie Mellon University), Feng Wang `[通讯]` (Johns Hopkins University)

**通讯引用:** 55790 | [OpenAlex ID](https://openalex.org/A5114994797)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CoPE，一种通过柔性裁剪 RoPE 低频分量来提升长文本上下文 LLM 性能的技术；

**💡 创新点**

将 OOD 缓解与语义建模统一到低频分量的子最优行为上，采用软裁剪避免频谱泄漏，从而实现长距语义注意力的稳定；

**🔧 技术方法**

软裁剪（cosine taper）对 RoPE 频率进行加权，结合 ABF 技术提升基频，实验使用 Llama‑3‑8B 训练继续至 64k 上下文，随后使用 YaRN 进行 256k 推理；

**📊 数据集**

在真实场景的 HELMET 基准（检索增强生成、长文档问答、多提示学习、摘要等）以及 RULER、InfiniteBench 等合成基准上进行评测；

**📈 对比分析**

与原 RoPE 及硬裁剪对比，CoPE 在 8k–256k 规模下平均提升 4.5%–58%（最高约 2 倍）且在 64k 训练区间不失一般性；

**⚠️ 局限性**

仍缺乏对更大模型、不同架构及多语种的广泛验证，且软裁剪参数（裁剪阈值）需经验调优；

---

## 8. FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters

**arXiv ID:** 2602.05235 | [PDF](https://arxiv.org/pdf/2602.05235v1)

**作者:** Zhilin Liang `[一作]` (Beihang University), Yongxin Tong `[通讯]` (Beihang University)

**通讯引用:** 11375 | [OpenAlex ID](https://openalex.org/A5051874566)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于参数化适配器的联邦检索增强生成（FedRAG）框架，既满足本地化约束，又通过多文档适配器和选择性聚合实现知识共享。

**💡 创新点**

创新点包括：①首个满足本地化约束的联邦RAG框架；②利用语义聚类与文档特定掩码的多文档LoRA适配器，显著降低存储与通信开销；③引入冲突感知的选择性聚合，避免跨站点适配器冲突。

**🔧 技术方法**

技术实现：参数化RAG（LoRA适配器）、约束k-means聚类、二值掩码学习与位打包、冲突感知的贪心选择、加权适配器聚合、联邦检索与重排序模型。

**📊 数据集**

使用四大问答数据集（HotpotQA、2WikiMultihopQA、PopQA、ComplexWebQuestions）进行主实验，另外用Enron Emails和WikiText评估隐私攻击。实验采用LLaMA3.2-1B-Instruct和LLaMA3-8B-Instruct两种基线。

**📈 对比分析**

与本地RAG、在情境FedRAG、联邦微调、参数化RAG等四类基线进行对比，平均提升10–20% F1，存储成本降低78–86%，通信成本下降91%，在隐私攻击测试中提取成功率为0%。在所有数据集上保持state‑of‑the‑art表现。

**⚠️ 局限性**

局限性包括：需对聚类大小和掩码稀疏度进行经验调参；假设所有站点使用相同的基线LLM；实验规模相对有限，未检验极大语料下的可扩展性；对极端隐私攻击（如模型逆向）仍未充分评估。

---

## 9. Learning with Adaptive Prototype Manifolds for Out-of-Distribution Detection

**arXiv ID:** 2602.05349 | [PDF](https://arxiv.org/pdf/2602.05349v1)

**作者:** Ningkang Peng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 618 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了APEX框架，通过两阶段修复原有原型学习中的静态同质假设和学习推理脱节问题，实现了更精细的原型空间和更优的OOD检测。

**💡 创新点**

创新点在于自适应原型曲面（APM）根据MDL/BIC动态分配每类原型数，和后验感知OOD评分（PAOS）利用原型质量信息进行评分校准。

**🔧 技术方法**

采用了vMF混合模型、GMM+贝叶斯信息准则、Sinkhorn-Knopp最优传输、EMA更新、能量函数和Mahalanobis距离校准等技术。

**📊 数据集**

主要在CIFAR‑100上实验，另外在CIFAR‑10、ImageNet‑100、ImageNet‑F/R等数据集验证。

**📈 对比分析**

与MSP、Energy、CIDER、PALM等基线相比，APEX在CIFAR‑100上平均FPR下降至29.32%，AUROC提升至92.91%，在多远域OOD场景和近域OOD场景均保持SOTA表现。

**⚠️ 局限性**

局限在于对超参数（如α、温度）敏感，需要在不同任务上手动调节，且目前主要针对分类任务的原型学习，迁移到其他任务的通用性待验证。

---

## 10. A Simple Reduction Scheme for Constrained Contextual Bandits with Adversarial Contexts via Regression

**arXiv ID:** 2602.05019 | [PDF](https://arxiv.org/pdf/2602.05019v1)

**作者:** Dhruv Sarkar `[一作]` (Indian Institute of Technology), Abhishek Sinha `[通讯]` (Tata Institute of Fundamental Research)

**通讯引用:** 2047 | [OpenAlex ID](https://openalex.org/A5057128963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一个模块化、统一的算法框架，用于在对抗性上下文下解决受约束的上下文赌博机（CCB）问题，给出了通用的惩罚与估计结合的策略；

**💡 创新点**

创新点在于构造了一个新的惩罚分解不等式，能够将探索（逆间隙加权）、约束管理（Lyapunov 驱动的代理奖励）和统计估计（在线回归 oracle）三大要素分离，并在此基础上得到对多种可行性基准（期望可行、Slater 条件、几乎必然可行、预算约束等）的理论最优或近最优性能；

**🔧 技术方法**

主要技术包括：对抗性上下文下的在线回归 oracle、Inverse Gap Weighting (IGW) 探索策略、Lyapunov 乘子法与代理奖励的自适应构造，以及基于上述惩罚分解的不等式实现的单一分析流程；

**📊 数据集**

论文不涉及实际数据集，全部以理论分析为主；

**📈 对比分析**

与现有基线（如基于 Lyapunov 计数的传统 CCB、带噪声的随机上下文方法等）相比，在多种基准下取得了更优的阶数（如期望可行情况下的 regret 与约束违背均为 O(√K T^{3/4} U_T^{1/4})，并在 Slater 条件、预算约束等情形下给出了相应的改进上界）；

**⚠️ 局限性**

局限性包括：需要满足 realizability 条件；算法的性能依赖于回归 oracle 的误差上界 U_T；理论上限可能不是最优；缺乏实验验证；对高维或非线性函数类的具体实现细节未展开。

---

## 11. Phantom Transfer: Data-level Defences are Insufficient Against Data Poisoning

**arXiv ID:** 2602.04899 | [PDF](https://arxiv.org/pdf/2602.04899v1)

**作者:** Andrew Draganov `[一作]` (LASR Labs), Mary Phuong `[通讯]` (Google Deepmind)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种能够在数据级防御下仍能植入模型情感倾向的全量数据投毒攻击，并展示了该攻击在多种大型语言模型（如 GPT‑4.1、Gemma‑3‑12B 等）上的成功率。

**💡 创新点**

创新点在于：①将“潜伏学习”（subliminal learning）方法改造为在真实训练场景（Alpaca 数据集）可用的投毒技术；②实现了 100% 投毒数据且在后期训练后仍保持隐蔽；③演示了即使采用最强（oracle）数据过滤、LLM 判断和改写，攻击仍能逃避检测；④扩展为密码触发后门，进一步提升难度。

**🔧 技术方法**

技术手段包括：潜伏学习框架、正则表达式过滤、LLM‑Judge（GPT‑5‑mini）评估、句子改写、以及对抗性的“steering vector”实验。

**📊 数据集**

使用的数据集为公开的 Alpaca 指令‑调优数据集（约 52k 例），教师模型为 Gemma‑3‑12B 与 GPT‑4.1，学生模型包括 GPT‑4.1、GPT‑4.1‑mini、Gemma‑3‑12B、OLMo‑2‑13B。

**📈 对比分析**

对比方法：在无防御、基本防御（词频、LLM 判断、随机删样）与 oracle 防御三种设置下评估攻击成功率（ASR）。实验显示，攻击在所有设置下特定与邻域 ASR 均高达 30–50%，而大多数防御的 TPR 仅为 0–6%。模型审计（Petri、pre‑fill、direct questioning）对普通投毒效果几乎无效，只有直接提问在一定程度上检测到情感倾向。

**⚠️ 局限性**

局限性包括：①攻击机制尚不完全透明，作者本人未能明确阐释为何投毒成功；②仅在情感倾向投毒上验证，未探究对其他攻击目标（如偏见、功能性后门）的通用性；③实验主要集中在少数模型与单一数据集，缺乏更广泛的跨模型与跨数据集验证。

---

## 12. COFFEE: A Carbon-Modeling and Optimization Framework for HZO-based FeFET eNVMs

**arXiv ID:** 2602.05018 | [PDF](https://arxiv.org/pdf/2602.05018v1)

**作者:** Hongbang Wu `[一作]` (Cornell University), Udit Gupta `[通讯]` (Cornell Tech)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出 COFFEE 框架，对 HZO‑FeFET 电子非易失性存储器的制造与使用生命周期碳足迹进行系统建模与评估；

**💡 创新点**

创新点在于首次将真实半导体工厂的 ALD 工艺参数与 NVM Explorer 设计空间探索结合，构建可复现的全生命周期碳排放评估；

**🔧 技术方法**

采用实测的 ALD 工具功耗与时间、iMec CMOS 数据、NVM Explorer 性能模拟及 ACT 计算的制造能耗与 GHG；

**📊 数据集**

使用来自半导体晶圆厂的工艺步骤数据、文献报道的 HZO‑FeFET 设备参数（7 种器件）以及 MobileNet V1 的边缘 TPU 流水线访存日志；

**📈 对比分析**

通过将 HZO‑FeFET 与 6T SRAM 在容量、面积、读写能耗等目标下进行对比，结果显示 HZO‑FeFET 的单位面积碳排放最高 11% 但单位容量碳排放约 4.3 倍低，边缘 TPU 权重缓冲区实现 42.3% 体积碳与 70% 操作碳显著下降；

**⚠️ 局限性**

局限性包括仅针对 HZO‑FeFET，未考虑其它 eNVM；寿命模型假设写入负载均匀，未充分考虑写入失效和可重复性；依赖特定晶圆厂的工艺数据，导致可迁移性受限。

---

## 13. PATHWAYS: Evaluating Investigation and Context Discovery in AI Web Agents

**arXiv ID:** 2602.05354 | [PDF](https://arxiv.org/pdf/2602.05354v1)

**作者:** Shifat E. Arman `[一作]` (University of Dhaka), Shahrear Bin Amin `[通讯]` (University of Dhaka)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了PATHWAYS基准，用以评估网页智能体在隐藏情境下主动发现并使用关键信息的能力，揭示其导航‑发现差距、调查幻觉以及对误导性回应的易受骗性；

**💡 创新点**

创新点在于：①将调查能力单独提炼为评价维度，填补功能执行与安全拒绝之间的空白；②构建了含有隐藏情境的多步决策任务；③设计了严格的 Proven Success Rate 等链式评估指标；

**🔧 技术方法**

采用基于ReAct/Reflexion的工具调用框架，结合原始与带提示的两种 prompt，使用任务级别的 Investigative Accuracy、Reasoning Accuracy、Evidence Quality Score 等度量；

**📊 数据集**

使用自建的 PATHWAYS 数据集，基于 WebArena 模拟的两类真实场景：购物后台（Magento）和 Reddit 社区审核，共250个多步决策任务，包含人类与 LLM 生成的干扰内容；

**📈 对比分析**

通过对比 Gemini、GPT‑4o、Qwen‑235B、Mistral、Llama 等闭源与开源模型，使用 Task Completion Rate、Investigation Accuracy、Proven Success Rate 等指标；结果显示 Reddit 领域完成率高但调查与决策准确率低，提示式 prompt 能提升调查准确率但往往导致决策错误，且模型对欺骗性回应高度敏感但对纠正信息反应迟缓；

**⚠️ 局限性**

局限性包括：基准仅基于静态 WebArena 快照，未涵盖实时网页动态和视觉复杂度；只聚焦文本驱动的电商与社区审核任务；未评估如 Tree‑of‑Thought 等更复杂推理架构；对抗实验仅为单轮，未覆盖多轮社会操纵情境。

---

## 14. Private Prediction via Shrinkage

**arXiv ID:** 2602.05219 | [PDF](https://arxiv.org/pdf/2602.05219v1)

**作者:** Chao Yan `[一作]` (Georgetown University), Chao Yan `[通讯]` (Georgetown University)

**通讯引用:** 1477 | [OpenAlex ID](https://openalex.org/A5101641854)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在流式查询模型下设计了差分隐私预测器，显著降低了对查询次数 T 的样本复杂度依赖，从传统的 √T 降到对 T 的 polylog 级别；并针对半空间概念类提出了适用于适应性攻击者的高效方案。

**💡 创新点**

核心创新在于：①利用“子样本‑聚合”与稀疏向量技术结合，构造可在不泄露训练样本信息的情况下对大量查询做预测；②提出通过不断收缩概念空间来限制稀疏向量触发次数，从而把 √T 下降到 O(VC·log T)；③针对半空间，利用线性可行性/几何约束，将适应性查询导致的“难查询”次数限制在 d+1 次，实现 O(d^5.5 log T) 的样本复杂度。

**🔧 技术方法**

主要技术手段包括：
- 子样本‑聚合（subsample‑and‑aggregate）框架
- 稀疏向量（Sparse Vector）与噪声投影
- 概念空间收缩（shrinkage）策略
- 线性可行性与凸几何（cdepth、线性约束交叉）
- Sauer 定理与深度函数分析
- 随机分块、PAC 学习与错误上界推导。

**📊 数据集**

使用的是理论模型中的标签样本集合 S，取自未知分布 𝔻；未采用真实数据集，全部基于合成或抽象概念类（一般 VC 类与 ℝ^d 上的半空间）进行分析。

**📈 对比分析**

与先前工作相比：
- 对一般 VC 类的无偏离对手，样本复杂度从 O(VC·√T) 下降到 O(VC^3.5·log^3.5 T)；
- 对半空间在适应性攻击下，样本复杂度从 O(d^2.5·√T)（或更高）提升到 O(d^5.5·log T)。
- 在保留相同隐私参数（ε,δ）和误差目标（α,β）的前提下，查询次数 T 的影响仅为多项式对数级。

**⚠️ 局限性**

局限性：
- 对 VC 类的样本复杂度仍呈高次多项式（VC^3.5），未达到最优线性或对数级；
- 适应性攻击下仅对半空间提供了结果，尚未推广到所有 VC 类；
- 需要在每个子块中训练非隐私 PAC 学习器，导致实现复杂度和运行时间较高；
- 对高维空间 d 的依赖较大（d^5.5），在实际高维应用中可能不切实际。

---

## 15. Untwisting RoPE: Frequency Control for Shared Attention in DiTs

**arXiv ID:** 2602.05013 | [PDF](https://arxiv.org/pdf/2602.05013v1)

**作者:** Aryan Mikaeili `[一作]` (Simon Fraser University), Ali Mahdavi-Amiri `[通讯]` (Simon Fraser University)

**通讯引用:** 899 | [OpenAlex ID](https://openalex.org/A5021862056)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对RoPE在扩散Transformer中的频率结构进行了分析，并提出了针对共享注意力导致的参考复制问题的频率感知调制方法，实现在不训练的情况下实现风格对齐的图像生成；

**💡 创新点**

创新点在于将RoPE的频率分量分解为高低频，并通过对高频抑制、低频放大实现对注意力的语义导向，从而消除参考复制并提升风格迁移的可控性；

**🔧 技术方法**

核心技术包括RoPE频率分析、频率感知调制（按频率分量动态缩放）、时间调度策略以及在Flux Diffusion Transformer中的共享注意力实现；

**📊 数据集**

实验使用StyleAligned的风格对齐提示集以及B-LoRA、InstantStyle-Plus等的参考图像数据集进行风格迁移评估；

**📈 对比分析**

与StyleAligned、AlignedGen、IP-Adapter和B-LoRA等方法对比，本文方法在保持语义多样性的同时显著降低了参考复制现象，取得了更佳的风格对齐效果，且不需要额外训练；

**⚠️ 局限性**

局限性包括对频率调制参数的手工设置需要经验、在极高分辨率下仍可能出现周期性复制或轻微伪影，且缺乏严格的定量评估。

---

## 16. Attack Selection Reduces Safety in Concentrated AI Control Settings against Trusted Monitoring

**arXiv ID:** 2602.04930 | [PDF](https://arxiv.org/pdf/2602.04930v1)

**作者:** Joachim Schaeffer `[一作]` (Pivotal Research), Tyler Tracy `[通讯]` (Redwood Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了AI攻击者在受监控部署中的攻击选择能力，并通过提示式分解质量评分与提交概率实现攻击选择。

**💡 创新点**

将攻击选择拆解为质量评分映射和提交阈值映射，视其为分类问题，提出基于嵌套交叉验证的阈值优化，并比较了rubric与GEPA优化提示。

**🔧 技术方法**

使用提示工程、自动提示优化（GEPA）、嵌套分割优化、概率阈值映射以及基于监控评分的质量评分。

**📊 数据集**

使用 BigCodeBench (BCB) 代码后门任务，筛选 438 个任务进行实验。

**📈 对比分析**

与始终攻击、vibes、监控提示+API 泄露等基线比较，安全性显著下降：0.5% 审计预算下 GEPA 优化提示安全性降至 59%，rubric 提示为 79%。

**⚠️ 局限性**

限制包括任务相对简单、游戏长度有限、未考虑攻击执行与利用、假设独立同分布、未覆盖多步代理任务等。

---

## 17. Semantic-level Backdoor Attack against Text-to-Image Diffusion Models

**arXiv ID:** 2602.04898 | [PDF](https://arxiv.org/pdf/2602.04898v1)

**作者:** Tianxin Chen `[一作]` (Fudan University), Cheng Huang `[通讯]` (Fudan University)

**通讯引用:** 36461 | [OpenAlex ID](https://openalex.org/A5100678432)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SemBD，一种在文本到图像扩散模型中以语义空间触发的后门攻击。

**💡 创新点**

创新点包括：①将触发器定义为连续语义区域而非离散词/句法；②使用多实体目标提升混淆性；③通过语义正则化防止不完整语义激活；④采用蒸馏式编辑跨注意力键/值投影实现轻量级后门植入。

**🔧 技术方法**

技术手段包括：跨注意力层键/值投影的模型编辑、基于CLIP的语义嵌入、语义正则化、蒸馏对齐损失、多实体目标设计。

**📊 数据集**

实验使用 Stable Diffusion v1.5 与 SDXL 两大模型，评估数据来自 MS‑COCO 验证集，触发器与对抗测试用 GPT‑4 生成的同义语义提示。

**📈 对比分析**

与现有词级、句法级后门（VillanDiffusion、Personalization、EvilEdit、BadT2I、IBA 等）对比，SemBD 在 ASR、CLIP_p 上均达到 100%/最高，且在 T2IShield、UFID、NaviT2I 等输入级防御下 DSR 仅为 2%–25.8%，同时保持 FID、LPIPS 等生成质量指标与原始模型相当。

**⚠️ 局限性**

局限性包括：在某些防御方法下仍能被检测到；依赖于对跨注意力投影的编辑，可能受模型结构变化影响；对超参数（λ_reg、α_k、α_v）敏感；目前仅验证于 SD‑v1.5/SDXL，跨平台泛化需进一步研究。

---

## 18. A Framework for Combining Optimization-Based and Analytic Inverse Kinematics

**arXiv ID:** 2602.05092 | [PDF](https://arxiv.org/pdf/2602.05092v1)

**作者:** Thomas Cohn `[一作]` (Massachusetts Institute of Technology), Russ Tedrake `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 14644 | [OpenAlex ID](https://openalex.org/A5074291890)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种利用已知解析逆运动学作为变量变换的新优化式IK框架，并在多种机器人平台上进行实验验证。

**💡 创新点**

创新点在于将解析IK映射作为光滑变量变换，将原本复杂的非线性等式约束转化为线性约束，从而显著提升求解器在碰撞、抓取、稳定性等附加约束下的成功率。

**🔧 技术方法**

采用解析IK函数、自动微分、三类主流优化器（IPOPT、SNOPT、NLOPT）以及基线方法（随机采样、Global‑IK），并实现碰撞避免、抓取选择、移动底座与人形稳定性等约束。

**📊 数据集**

使用基于仿真的随机目标姿态数据集：KUKA iiwa 14/7DOF臂、PR2两臂、Hubo 2+人形，以及不同障碍物场景中的100/40个随机目标。

**📈 对比分析**

通过与旧式IK优化、采样、Global‑IK等方法对比，新的变换式IK在三种求解器上均取得更高的成功率（尤其是碰撞、抓取、稳定性任务），但在部分场景下计算时间略长，且优化成本常低于旧式。

**⚠️ 局限性**

局限性包括：对解析IK的依赖（未覆盖无解析解的机器人）、成本函数更复杂导致迭代次数增多、在极高自由度或强非凸约束下可能仍遇到收敛速度慢或局部最优问题。

---

## 19. Democratic Preference Alignment via Sortition-Weighted RLHF

**arXiv ID:** 2602.05113 | [PDF](https://arxiv.org/pdf/2602.05113v1)

**作者:** Suvadip Sana `[一作]` (Cornell University), Martin T. Wells `[通讯]` (Cornell University)

**通讯引用:** 10793 | [OpenAlex ID](https://openalex.org/A5084409300)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于算法抽签（sortition）的民主偏好优化框架 DemPO，用来在强化学习自人类反馈（RLHF）中实现人口代表性。

**💡 创新点**

创新点在于：①用算法抽签构造符合边际配额约束的代表性迷你公众；②提出硬面板（Hard Panel）仅使用抽签样本的偏好数据训练；③提出软面板（Soft Panel）对所有偏好按抽签包含概率加权训练，并证明两者等价于抽签期望；④在实验中证明代表性约束能显著提升模型行为与代表性公众宪法的一致性。

**🔧 技术方法**

技术核心包括：算法抽签（LEXIMIN 过程）生成面板概率；直接偏好优化（Direct Preference Optimization, DPO）作为基础损失；按抽签概率加权的自归一化训练；多重排序聚合方法（Bradley–Terry、Borda、Copeland、Kemeny、Mallows）用于评估模型与宪法条款的一致性。

**📊 数据集**

数据集：PRISM 对话与偏好数据，附带评审者人口统计信息；美国代表性公众收集的 75 条宪法条款（从 1,000 名受访者的投票与提案中生成）。

**📈 对比分析**

比较方法：在 Llama 系列模型（1B、3B、8B）上分别训练五种策略（Base、Full PRISM、US‑Rep、Hard Panel、Soft Panel）。使用 GPT‑5.2 作为自动评判器生成 3,000 条评价问答，随后采用五种聚合方法统计模型排名。结果显示 Hard Panel 始终排名第一，US‑Rep 第二，Soft Panel 高于 Full PRISM，Base 最差；且随着模型规模增大，民主面板的优势进一步放大。

**⚠️ 局限性**

局限性包括：①仅在单一基础模型族上验证；②依赖 PRISM 这类稀缺的带有人口统计信息的偏好数据；③评判依赖自动 LLM 评判器，可能带来偏差；④抽签仅满足边际配额，未考虑交叉属性或更丰富的价值约束；⑤对不同地区、语言环境的适用性尚未验证。

---

## 20. HugRAG: Hierarchical Causal Knowledge Graph Design for RAG

**arXiv ID:** 2602.05143 | [PDF](https://arxiv.org/pdf/2602.05143v1)

**作者:** Nengbo Wang `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**通讯引用:** 4016 | [OpenAlex ID](https://openalex.org/A5004523290)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HugRAG 框架，利用层次化因果门控在图检索中跨模块跳转并通过 LLM 识别因果路径过滤噪声，从而提升 RAG 的全局召回与局部精度。

**💡 创新点**

① 将知识图按多层级模块划分并构造跨模块的因果门，打破信息隔离；② 用 LLM 进行因果路径推理，做显式噪声过滤；③ 在多域数据上验证其在召回-精度平衡与可扩展性上的优势；④ 引入 HolisQA 评测集以检验全局推理能力。

**🔧 技术方法**

图结构检索、Leiden 社区划分、向量检索、LLM（gpt‑5‑nano）用于因果判定与文本生成、最佳优先搜索、MMR 多样化种子选择、个性化 PageRank、因果门控机制。

**📊 数据集**

HolisQA（5 个学科）+ MS MARCO、Natural Questions、2WikiMultiHopQA、QASC、HotpotQA。

**📈 对比分析**

与 BM25、StandardRAG、GraphRAG Global/Local、LightRAG、HippoRAG2、LeanRAG、CausalRAG 等基线比较；在 HolisQA 的 F1、Context Recall、Answer Relevancy 均为最高；在标准 QA 数据集也获得最优或接近最优的 F1 与 AR；在不同文本规模下保持稳定的性能。

**⚠️ 局限性**

依赖 LLM 进行因果推理，计算成本较高；因果门阈值和层级数需手动调参；目前未在极大规模实时系统或多语言低资源环境下验证；对门控机制的可解释性和鲁棒性仍需进一步研究。

---

## 21. Reliable Explanations or Random Noise? A Reliability Metric for XAI

**arXiv ID:** 2602.05082 | [PDF](https://arxiv.org/pdf/2602.05082v1)

**作者:** Poushali Sengupta `[一作]` (University of Oslo), Yan Zhang `[通讯]` (University of Oslo)

**通讯引用:** 45062 | [OpenAlex ID](https://openalex.org/A5100456327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种解释可靠性指数（ERI）及其变体，并构建了ERI‑Bench基准，用于评估XAI方法在小扰动、特征冗余、模型演进与时间漂移等实际场景下的稳定性。

**💡 创新点**

创新点在于将可靠性视为四条可证明的公理（扰动稳健、冗余一致、模型演进平滑、分布/时间鲁棒），给出严格的Lipschitz界限，并首次引入时序可靠性指标ERI‑T以及统一的可靠性评测框架。

**🔧 技术方法**

技术上结合了解释器的Lipschitz连续性、期望漂移计算、距离度量（如L2、余弦、Wasserstein）、Monte Carlo采样和理论上可验证的冗余/时间变换操作。

**📊 数据集**

实验使用四个公开数据集：EEG微状态、UCI HAR（人体动作识别）、挪威电力负荷预测（5个区域）以及CIFAR‑10，覆盖时序、表格与图像域。

**📈 对比分析**

对比结果显示，传统解释器（IG、SHAP、DeepLIFT）在冗余与时间维度上易失真，MCIR、MI、HSIC在ERI各维度表现最稳定（ERI≈1），但其中MI/HSIC为全局依赖度量而非真正的局部解释；ERI‑Bench揭示了广泛的可靠性缺陷，并表明高ERI往往与更好的下游特征选择性能相关。

**⚠️ 局限性**

局限性包括：可靠性高不代表解释的因果正确性或有用性，某些全局度量（MI/HSIC）得到的高ERI是人为的稳定性；ERI仅评估非对抗性小变动，对鲁棒性攻击和极端分布偏移的评估仍需进一步研究。

---

## 22. Physics as the Inductive Bias for Causal Discovery

**arXiv ID:** 2602.04907 | [PDF](https://arxiv.org/pdf/2602.04907v1)

**作者:** Jianhong Chen `[一作]` (Northeastern University), Xubo Yue `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种整合已知物理动力学与因果发现的框架，使用随机微分方程（SDE）将漂移项编码已知的ODE，扩散项编码未知因果耦合，并通过稀疏最大似然（MLE）算法恢复因果图；

**💡 创新点**

创新点在于：①将物理知识作为诱导偏差直接嵌入SDE模型；②不依赖DAG或平稳假设，支持非加性噪声和循环结构；③给出多变量稀疏SDE下的识别性与收敛理论；④设计可扩展的梯度优化方法。

**🔧 技术方法**

技术手段包括：Euler‑Maruyama离散、基于高斯似然的稀疏拉格朗日优化、L1 正则化、局部强凸性证明、稀疏性与不可相关性假设下的统计分析。

**📊 数据集**

使用自生成的SDE时间序列数据，构造 DAG 与循环（feedback）图，维度分别为 5、10、15、20，并在稳定与不稳定两种动态模式下进行实验；

**📈 对比分析**

与四个基线方法（score‑based、constraint‑based、VAE‑neuralSDE、Granger）对比，评估指标为 SHD、TPR、FDR；实验显示该方法在所有设置下均显著降低 SHD、提升 TPR、降低 FDR，尤其在循环图和高维/不稳定场景中表现最优。

**⚠️ 局限性**

局限性包括：仅在模拟数据上验证；对真实复杂非平稳系统的泛化尚待验证；需要已知部分ODE，且假设稀疏且线性/局部线性；计算成本相对较高。

---

## 23. Knowing When to Answer: Adaptive Confidence Refinement for Reliable Audio-Visual Question Answering

**arXiv ID:** 2602.04924 | [PDF](https://arxiv.org/pdf/2602.04924v1)

**作者:** Dinh Phu Tran `[一作]` (School of Computing), Daeyoung Kim `[通讯]` (School of Computing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出可靠音频视觉问答（ℛ-AVQA）框架，允许模型在不确定时拒答。

**💡 创新点**

创新在于自适应置信度细化（ACR），在保留最大软最大概率的基础上，通过残差风险头和置信门控头学习输入自适应校正。

**🔧 技术方法**

技术包括两阶段训练、残差风险估计、置信门控、跨模态特征融合以及与MSP、MCD、VS、Doctor等基线比较。

**📊 数据集**

使用MUSIC-AVQA、MUSIC-AVQA-R、MUSIC-AVQA-v2.0等三大音视频问答数据集。

**📈 对比分析**

与MSP、MCD、VS、Doctor等方法对比，在低风险下（1%–20%）覆盖率提升约5–20个百分点，AURC与ECE明显下降。

**⚠️ 局限性**

局限在于仍有与oracle之间的覆盖差距，且在极低风险下覆盖率偏低；方法依赖预训练模型和多模态特征，迁移到其他任务需验证。

---

## 24. Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?

**arXiv ID:** 2602.05023 | [PDF](https://arxiv.org/pdf/2602.05023v1)

**作者:** Ruixin Yang `[一作]` (Georgia Institute of Technology), Alan Ritter `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 10169 | [OpenAlex ID](https://openalex.org/A5039096905)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个用于评估视觉语言模型在图像地理位置披露时是否遵守情境完整性的基准，并对现有模型进行评测。

**💡 创新点**

创新点在于结合真实社交媒体图像、细粒度上下文与共享意图标注，构建了包含多项选择与自由生成任务的双重评价体系，揭示模型在隐私决策上的细致缺陷。

**🔧 技术方法**

采用多模态视觉语言模型（如 GPT‑5、o3、Gemini‑2.5‑Flash、Claude‑Sonnet‑4、Llama‑4‑Maverick 等），并使用 Chain‑of‑Thought、恶意提示等多种对话策略进行评估。

**📊 数据集**

数据集为 1,200 张从 YFCC、IM2GPS、GPTGeoChat 等公开数据集挑选并人工标注的图像，覆盖人脸、政治事件、私密空间等敏感因素。

**📈 对比分析**

通过结构化多项选择题和自由生成题与人类标注对比，发现大多数模型在情境判断上仅 60% 级别匹配，最佳模型在自由生成中的准确率仅为 49.7%，且在恶意或逐步提示下会显著过度披露或泄露位置。

**⚠️ 局限性**

局限性包括数据集规模有限、标注仍可能受主观影响、模型对细粒度隐私规范的学习不足，以及评测方法主要聚焦地理位置披露而未覆盖更广泛的隐私场景。

---

## 25. Causal Online Learning of Safe Regions in Cloud Radio Access Networks

**arXiv ID:** 2602.05280 | [PDF](https://arxiv.org/pdf/2602.05280v1)

**作者:** Kim Hammar `[一作]` (University of Melbourne), Emil Lupu `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过因果在线学习（COL）方法在云RAN中识别安全可操作区域，支持在不违反服务规范的前提下安全探索网络配置。

**💡 创新点**

创新点在于将因果图知识与贝叶斯主动学习相结合，利用因果推断产生先验后通过高斯过程进行主动探索，显著提升样本效率并提供可证明的安全保证。

**🔧 技术方法**

使用的核心技术包括因果推断（do‑calculus）、Gaussian Process回归、主动学习采样策略以及与安全强化学习基线的对比实验。

**📊 数据集**

实验使用了168小时的5G云RAN测试平台监控数据（包含CPU、内存、负载、延迟等指标）以及相应的模拟数据集。

**📈 对比分析**

与CPO、PPO‑Lagrangian等安全强化学习基线对比，COL在安全区域学习上样本效率提升约10倍，违规干预次数仅为基线的十分之一，安全性保证达α=0.8。

**⚠️ 局限性**

主要限制是需要预先知道准确的因果图；若因果结构未知或随时间变化，则需进行因果发现或手动更新，且对高维离散控制变量的扩展仍需进一步研究。

---

## 26. DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search

**arXiv ID:** 2602.05014 | [PDF](https://arxiv.org/pdf/2602.05014v1)

**作者:** Zhanli Li `[一作]` (Chinese Academy of Sciences), Ping Luo `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 53125 | [OpenAlex ID](https://openalex.org/A5100752686)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种结构感知的多轮检索-生成代理DeepRead，用于长文档问答。论文将PDF转换为Markdown，按段落索引并为每段生成坐标式元数据，随后在Agentic RAG框架下提供两种工具：Retrieve（定位）和ReadSection（按顺序读取）。模型先用Retrieve快速定位可能含有答案的段落，再用ReadSection完整获取该段落所在章节的连贯文本，模拟人类的“先定位后阅读”流程。

**💡 创新点**

创新点主要体现在：
1) 将文档的层级结构和段落顺序映射为坐标化元数据，使检索结果携带结构位置信息；
2) 通过两种互补工具实现结构化检索与顺序阅读的协同，显著减少上下文碎片化和重复检索；
3) 在Agentic RAG中首次引入基于结构的交互接口（Coordinate-based Interaction），从而使LLM在多轮推理中能自适应地决定何时检索、何时阅读、何处停留。

**🔧 技术方法**

核心技术包括：
- OCR + Markdown解析（PaddleOCR‑VL）将PDF转换为结构化文本；
- 段落级向量检索（使用Qwen3-embedding‑8b）与索引；
- ReAct框架下的工具调用，两个工具分别实现扫描检索与顺序阅读；
- 结构化坐标系统（doc‑id, section‑id, paragraph‑id）与元数据包装；
- 评判者采用LLM（DeepSeek V3.2、GLM‑4.7、Qwen3‑235B）进行自动打分。

**📊 数据集**

使用四个公开/自制基准：
- FinanceBench（财务报告长文档）
- ContextBench（自制单文档长范围QA）
- QASPER（多文档学术论文QA）
- SyllabusQA（多文档课程大纲QA）。

**📈 对比分析**

与Dense RAG、RAPTOR、ITRG、Search‑o1等多种基线在相同的检索设置下对比。DeepRead在四个基准上的平均提升约为10个百分点：
- FinanceBench +3.0 ；
- ContextBench +17.0 ；
- QASPER +7.7 ；
- SyllabusQA +13.8 ；
在扩展窗口（expand）下仍保持优势，说明结构化阅读能在无需额外窗口扩展的情况下获得完整上下文。实验还表明，在多文档场景中，DeepRead的性能提升更为显著。

**⚠️ 局限性**

主要局限：
- 依赖OCR/Markdown转换的质量，低质量扫描文档会导致结构丢失；
- 目前只支持层级化PDF，非结构化文本或极端排版的文档效果未知；
- 仅按段落级索引，跨段落推理或细粒度句子级信息仍需进一步改进；
- 评判采用LLM，可能带来偏差，需结合人工评测验证；
- 对于某些任务，扩展窗口的使用会降低准确率，说明需要更细粒度的窗口控制策略。

---

## 27. Private PoEtry: Private In-Context Learning via Product of Experts

**arXiv ID:** 2602.05012 | [PDF](https://arxiv.org/pdf/2602.05012v1)

**作者:** Rob Romijnders `[一作]` (University of Amsterdam), Yuki M. Asano `[通讯]` (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Product‑of‑Experts（PoE）模型的差分隐私上下文学习方法。

**💡 创新点**

创新点在于用软预测的 PoE 取代硬投票，保留不确定性信息，并给出理论隐私证明和经验成员推断攻击评估。

**🔧 技术方法**

采用 DP 算法中的概率剪裁、指数机制、PoE 近似以及隐私分析与成员推断攻击技术。

**📊 数据集**

使用 AGNews、DBpedia、TREC、GSM8k 以及 Vision‑Language pseudo‑name 任务的数据集进行实验。

**📈 对比分析**

与 RNM、PbS、合成数据等方法对比，平均在文本分类上提升约 30% 点，数学和 VLM 任务亦表现优于先前方法。

**⚠️ 局限性**

局限性包括依赖条件独立假设、对少数类的公平性未得到充分保障，以及需要手动调节隐私预算。

---

## 28. Unlocking Prototype Potential: An Efficient Tuning Framework for Few-Shot Class-Incremental Learning

**arXiv ID:** 2602.05271 | [PDF](https://arxiv.org/pdf/2602.05271v1)

**作者:** Shengqin Jiang `[一作]` (Nanjing University of Information Science and Technology), Ming-Hsuan Yang `[通讯]` (University of California at Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种冻结特征提取器、仅细调原型的高效原型调优框架，用于解决少样本类增量学习中的灾难性遗忘和过拟合问题。

**💡 创新点**

创新点在于将静态原型拆分为类特定和任务感知双校准偏置，形成可学习的动态原型，并通过负误差投影显式建模查询特征与所有原型之间的线性关系，从而提升判别力。

**🔧 技术方法**

采用预训练ViT/DINOv2/DINOv3的冻结特征提取器，结合类特定与任务感知双校准、负误差投影、交叉熵加互类判别损失实现原型细调。

**📊 数据集**

在CUB‑200、ImageNet‑R、ImageNet‑A、VTAB四个公开基准上进行实验。

**📈 对比分析**

与SOTA方法（TEEN、SEC、ASP、DSS‑P等）对比，平均精度和最终精度均明显提升，且可学习参数量仅为0.22–0.28M，参数效率极高。

**⚠️ 局限性**

依赖预训练特征空间的质量，若预训练模型表现不佳或存在极端少样本/长尾分布时，提升空间有限，且未针对多任务或跨域迁移做进一步扩展。

---

## 29. SocialVeil: Probing Social Intelligence of Language Agents under Communication Barriers

**arXiv ID:** 2602.05115 | [PDF](https://arxiv.org/pdf/2602.05115v1)

**作者:** Keyang Xuan `[一作]` (University of Illinois Urbana-Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7864 | [OpenAlex ID](https://openalex.org/A5003491365)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于认知差异导致的沟通障碍的交互式社交学习环境SocialVeil，用于评估大型语言模型在现实沟通环境中的社交智能；

**💡 创新点**

创新点在于构建了三类系统化沟通障碍（语义模糊、社会文化不匹配、情感干扰）并配套两项障碍感知评价指标（未解决困惑、互惠理解），从而在控制实验中逼真模拟现实沟通失效；

**🔧 技术方法**

使用了大语言模型（GPT‑4o, Qwen‑2.5‑7B, Qwen‑3‑4B, Mistral‑8B）生成对话，并通过Prompt注入障碍、两层参数化设计、自动化评估（goal、relationship、knowledge、confusion、mutual）以及人类评测来验证效果；

**📊 数据集**

数据集由180个场景（语义模糊、社会文化不匹配、情感干扰）与基线共四组，场景来源于现有社交基准（如Sotopia、Sotopia-6c），并通过GPT‑4o重写实现任务目标隐藏；

**📈 对比分析**

与基线比较发现三类障碍均显著降低LLM的交互质量，互惠理解下降≈45%、关系质量下降≈49%，实验还测试了修复指令与交互式学习两种适应策略，后者仅提升约10‑20%，仍无法恢复到无障碍水平；

**⚠️ 局限性**

局限性包括：障碍仅从认知层面建模，未涵盖物理噪声；评估主要基于LLM内部提示与自动评测，可能对不同模型产生偏差；且人类评测一致性有限（Fleiss Kappa≈0.38）。

---

## 30. Atomic Information Flow: A Network Flow Model for Tool Attributions in RAG Systems

**arXiv ID:** 2602.04912 | [PDF](https://arxiv.org/pdf/2602.04912v1)

**作者:** James Gao `[一作]` (Atlassian), Steven Yoo `[通讯]` (Atlassian)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Atomic Information Flow (AIF)，一种基于图网络流的模型，用以将工具输出和 LLM 调用拆分为不可分割的原子（atom），并追踪这些原子在 RAG 系统中的流动，从而实现细粒度的工具归因与调试。

**💡 创新点**

创新点包括：① 将 RAG 视为流网络，以原子为流量单位，引入最大流-最小割定理寻找信息瓶颈；② 设计了原子分解、信号注入、响应原子匹配的完整流水线；③ 通过最小割信号训练小型 Gemma3‑4B 上下文压缩器，达到与更大模型相当的压缩与准确率；④ 定义多种流量启发式指标（Groundedness、Tool Consumption 等），提供可量化的工具使用分析。

**🔧 技术方法**

技术手段包括：图网络流建模、原子分解（使用 GPT5‑Nano + map‑reduce）、原子相关性评分、响应原子匹配、流量启发式计算、监督式最小割优化、Gemma3‑4B 上下文压缩器微调、LLM‑as‑a‑judge（GPT‑4.1）评估。

**📊 数据集**

使用的数据集：HotpotQA、MS MarcoV2、Musique、Wiki Multihop QA；每个数据集均包含多文档上下文与人工标注的工具归因标签，用于评估 AIF 的归因精度与压缩效果。

**📈 对比分析**

比较方法：在归因任务上与 ALCE 基线（Vanilla GPT5‑Nano）对比，采用 Precision、Recall、F1；在压缩任务上与原始 Gemma3‑4B、Gemma3‑27B 进行对比，评估 Token Reduction 与回答准确率。结果显示，AIF 在 true 段落上 Precision 及 Recall 均略高于基线，F1 在 HotpotQA 上提升 0.3%；在压缩实验中，Gemma3‑4B‑AIF 在 HotpotQA 上实现 87.52% token 压缩并保持 82.71% 正确率，明显优于纯 Gemma3‑4B 的 54.7% 正确率。

**⚠️ 局限性**

局限性：① 仅针对工具到响应的流动，未覆盖检索到工具（Query→Tool）的路径；② 原子分解与匹配需要多次 LLM 调用，计算成本高；③ 目前仅使用相关性信号，未探索权威性、时效性等更丰富的原子属性；④ 评估范围受限于四个 QA 数据集，未验证在更广泛多模态或更大规模 RAG 系统中的表现；⑤ 需进一步研究轻量化分解模型与全流程的可扩展性。

---

## 31. LOBSTgER-enhance: an underwater image enhancement pipeline

**arXiv ID:** 2602.05163 | [PDF](https://arxiv.org/pdf/2602.05163v1)

**作者:** Andreas Mentzelopoulos `[一作]` (Massachusetts Institute of Technology), Keith Ellenbogen `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 23 | [OpenAlex ID](https://openalex.org/A5048759825)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并训练一个条件潜在扩散模型（约 11M 参数），通过先合成水下降解的图像，然后学习反向恢复，从而实现水下摄影图像的去模糊、去泡沫、色彩校正和结构修复。

**💡 创新点**

创新点包括：① 用专门设计的人工降解管道模拟真实水下失真；② 在极小的数据集（约 2.5k 张）上训练轻量化潜在扩散模型；③ 结合质量加权采样、cosine 变异、v‑prediction loss 等技术，使模型在训练集内外均能保持高感知一致性并实现自适应增强。

**🔧 技术方法**

技术细节：潜在扩散模型（DDPM）+ VQ‑GAN 编码器+ 轻量 U‑Net；cosine 变异调度、v‑prediction 目标、条件化 concat 与 classifier‑free guidance；质量加权采样以及周期性刷新降解样本等训练策略。

**📊 数据集**

数据集：Keith Ellenbogen 的 2.5k 张高质量水下摄影图像（仅 4 种物种），再加 50 张不同物种的 OOD 图像用于测试；所有图像尺寸 512×768，采用自制降解管道生成对应降解图像。

**📈 对比分析**

评价方法：使用感知一致性、色彩校正、结构恢复等主观指标，并展示对 OOD 物种的泛化；模型在 15k 轮训练后收敛，生成 512×768 的高质量图像，显著提升色彩、对比度和细节，且在未见过的物种上也能保持真实感。

**⚠️ 局限性**

局限性：① 训练数据规模有限，极端光照或大尺度景观下的恢复效果可能不足；② 人工降解管道与真实水下条件不完全匹配，导致某些细节或光照效果仍有偏差；③ 本文缺少与传统水下图像增强算法的量化对比与标准评价指标。

---

## 32. HealthMamba: An Uncertainty-aware Spatiotemporal Graph State Space Model for Effective and Reliable Healthcare Facility Visit Prediction

**arXiv ID:** 2602.05286 | [PDF](https://arxiv.org/pdf/2602.05286v1)

**作者:** Dahai Yu `[一作]` (Florida State University), Guang Wang `[通讯]` (Florida State University)

**通讯引用:** 4548 | [OpenAlex ID](https://openalex.org/A5100451757)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向医疗机构访客预测的无不确定性时空框架，能够预测不同类型机构的访客量并给出可靠的不确定性区间。

**💡 创新点**

创新点在于同时建模空间依赖、异构上下文信息和多类型机构，并整合节点、分布、参数三种不确定性量化与后置分位数校准，实现了在异常情境下的可靠预测。

**🔧 技术方法**

核心技术包括统一时空上下文编码器(STCE)、图状态空间模型GraphMamba（基于UNet结构的Mamba与自适应图学习），以及三组不确定性量化模块和分位数校准。

**📊 数据集**

使用来自SafeGraph的四州（加州、纽约、德州、佛罗里达）县级每日访客数据，涵盖四种医疗机构类别。

**📈 对比分析**

与13个最先进基线（GNN、注意力、Transformer、LLM与Mamba等）对比，平均MAE降低约6%，不确定性评价指标提升约3.5%，在短期和中期均取得最佳性能。

**⚠️ 局限性**

局限性包括对长周期预测缺乏评估、模型对极端事件的适应性仍依赖于校准集，且在极端天气与疫情之外的其他异常场景下的鲁棒性尚未验证。

---

## 33. Reporting and Reviewing LLM-Integrated Systems in HCI: Challenges and Considerations

**arXiv ID:** 2602.05128 | [PDF](https://arxiv.org/pdf/2602.05128v1)

**作者:** Karla Felix Navarro `[一作]` (Universite de Montreal), Ian Arawjo `[通讯]` (Universite de Montreal)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过访谈18位 HCI 领域作者，系统地探讨了在 LLM 整合系统论文的写作、评审与发布过程中的信任挑战与实践策略，并基于访谈结果和专家反馈提出了一套作者、评审和社区的考量与指南。

**💡 创新点**

创新点在于：①首次从作者与评审双方视角对 LLM 集成系统论文的信任机制进行定性研究；②揭示 LLM 带来的不确定性如何削弱 HCI 传统的信任与评审流程；③提出基于实践经验的、可操作的报告与评审建议，填补 HCI 领域关于 LLM 系统规范的空白。

**🔧 技术方法**

采用的技术主要是半结构化访谈与归纳式主题分析（Grounded Theory），辅以对收集到的审稿意见与反驳稿进行案例分析。

**📊 数据集**

研究所用数据集为：18 名受访者的访谈记录（约 1 小时/人）以及5份匿名评审与2份作者反驳稿，补充4份自我提交的系统论文评审记录。

**📈 对比分析**

本文并未进行算法性能比较，而是通过访谈结果与专家反馈归纳出作者与评审在报告细节、技术评估、LLM 选择与使用理由等方面的共识与分歧，并给出对应的对策与指南。

**⚠️ 局限性**

局限性包括：受访者样本可能偏向对 LLM 评价持强烈意见者，且研究主要聚焦 HCI 领域，未覆盖更广泛的跨学科社区；访谈数据受回忆与自我呈现偏差影响；缺乏大规模量化验证，指南的通用性需进一步评估。

---

## 34. Magic-MM-Embedding: Towards Visual-Token-Efficient Universal Multimodal Embedding with MLLMs

**arXiv ID:** 2602.05275 | [PDF](https://arxiv.org/pdf/2602.05275v1)

**作者:** Qi Li `[一作]` (Honor Device Co., Ltd), Jinxiang Liu `[通讯]` (Honor Device Co., Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Magic-MM-Embedding，通过压缩视觉令牌并使用分阶段精细训练，实现了统一多模态检索的高效与高精度。

**💡 创新点**

创新点在于引入无参双线性插值压缩视觉令牌，并构建三阶段渐进式训练流程（生成恢复、对比预训练+硬负样本挖掘、MLLM-评判细化），将效率与性能协同提升。

**🔧 技术方法**

采用 InternVL3-MLLM 作为骨干，配合无参视觉令牌压缩、InfoNCE 对比损失、LoRA 微调、硬负样本挖掘、MLLM 评判（Judge）做数据清洗，以及协同训练的重排序器。

**📊 数据集**

使用 32M 多模态指令跟随数据、16M 多模态检索数据（MegaPairs、Colpali、VisRAG、Docmatix、BAAI‑MTP、ImageNet‑1K、BLIP、MMEB‑train、mmE5‑synthetic）以及 1.5M 精细标注的多任务数据集进行训练。

**📈 对比分析**

在 MMEB、VisDoc、Flickr30K、MSCOCO 等基准上与 CLIP、UniME‑V2、QQMM、GME 等方法对比，Magic‑MM‑Embedding 在精度上刷新 SOTA，并在视觉令牌仅占 25% 的情况下显著降低推理延迟。

**⚠️ 局限性**

局限性包括对大型公开数据集的依赖、三阶段训练流程复杂且耗时，以及在视觉文档检索场景下相较某些基线仍有略高延迟；对 MLLM Judge 的准确性也会影响硬负样本的质量。

---

## 35. Applying a Requirements-Focused Agile Management Approach for Machine Learning-Enabled Systems

**arXiv ID:** 2602.05042 | [PDF](https://arxiv.org/pdf/2602.05042v1)

**作者:** Lucas Romao `[一作]` (Pontifical Catholic University of Rio de Janeiro), Marcos Kalinowski `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一次产学研合作项目中，作者提出并实践了RefineML——一种面向需求的敏捷管理方法，用以持续改进机器学习驱动的网络安全系统。

**💡 创新点**

创新点在于将PerSpecML需求规范、Agile4MLS敏捷实践、MVM与LoD模型演进框架整合为一套连续、双轨治理的工作流程，并强调早期交付的Demo API。

**🔧 技术方法**

使用的技术包括PerSpecML需求建模、Agile4MLS迭代计划、Scrum/Scrum Master、MVM与LoD层次，以及标准的机器学习分类模型（如诈骗信息、恶意URL、截图文本提取）。

**📊 数据集**

数据集来自合作方EXA的企业内部安全日志与消息数据，未公开列出具体公开数据集。

**📈 对比分析**

本研究未通过实验对比算法性能，而是通过问卷和访谈评估方法的可用性与接受度，结果显示用户认为方法易用且有效，但缺乏量化性能指标。

**⚠️ 局限性**

主要限制包括：将PerSpecML规范转化为可执行的ML待办项困难、对经验丰富的主持人高度依赖、对ML工作量与结果的估算极其不确定，以及MLOps实施带来的运维负担。

---

## 36. ZeroS: Zero-Sum Linear Attention for Efficient Transformers

**arXiv ID:** 2602.05230 | [PDF](https://arxiv.org/pdf/2602.05230v1)

**作者:** Jiecheng Lu `[一作]` (Georgia Institute of Technology), Shihao Yang `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1158 | [OpenAlex ID](https://openalex.org/A5057260690)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Zero-Sum Linear Attention（ZeroS），通过去除 softmax 的常数项并重新加权零求和残差实现线性复杂度的注意力机制。

**💡 创新点**

创新点在于消除零阶常数项，允许负权重与高阶交互，从而突破传统线性注意力的凸组合限制。

**🔧 技术方法**

采用重加权零求和 softmax、可学习的门控、旋转位置编码（RoPE）与前缀和扫描实现 O(N) 复杂度。

**📊 数据集**

在 MAD、RegBench、WikiText‑103、ImageNet‑1k、OpenWebText‑2、时间序列预测等多任务数据集上进行评估。

**📈 对比分析**

与 Transformer 及 Hyena、Mamba、GLA、DeltaNet、LinAttn 等线性注意力模型对比，ZeroS 在多数基准上匹配或超过 softmax 结果，并在推理效率上保持 O(N)。

**⚠️ 局限性**

局限在于未对大规模 LLM 任务进行评估，且缺乏针对 GPU 加速的工程优化，主要聚焦于算法表达力提升。

---

## 37. Large Language Models in Software Documentation and Modeling: A Literature Review and Findings

**arXiv ID:** 2602.04938 | [PDF](https://arxiv.org/pdf/2602.04938v1)

**作者:** Lukas Radosky `[一作]` (Comenius University Bratislava), Ivan Polasek `[通讯]` (Comenius University Bratislava)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5064482961)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2024‑2025年IEEE TSE、ACM TOSEM、Springer EMSE、ICSE等四个主流软件工程期刊/会议的57篇论文进行系统综述，聚焦LLM在软件文档与建模任务中的应用；

**💡 创新点**

首次从文档与建模角度全面梳理LLM4SE的任务类别、提示技术、评估指标、数据集与人类评估方法，并指出零样本提示与promptless模型占主导，few‑shot与链式思维应用不足；

**🔧 技术方法**

采用文献检索+手工筛选、关键词搜索、标题/摘要过滤、全文阅读等方法，对任务使用文本生成、分类、回归等技术进行归类与评估；

**📊 数据集**

主要使用公开数据集包括MCMD（提交信息）、CodeSearchNet、PCSD（代码摘要）、BPMN/SO标签/标题等，以及自建或增强版数据；

**📈 对比分析**

通过对比各论文的BLEU、ROUGE、F1、MAE等指标与基线，发现大多数LLM方法在生成质量或准确率上优于传统方法，尤其是零样本提示和promptless模型，但对比缺乏统一标准和跨数据集验证；

**⚠️ 局限性**

研究仅覆盖最近两年且仅限四个主流期刊/会议，忽略了其他LLM4SE工作；多任务与多代理系统未深入探讨；数据集多为自建或领域特定，缺乏跨任务通用基准。

---

## 38. Applying Ground Robot Fleets in Urban Search: Understanding Professionals' Operational Challenges and Design Opportunities

**arXiv ID:** 2602.04992 | [PDF](https://arxiv.org/pdf/2602.04992v1)

**作者:** Puqi Zhou `[一作]` (George Mason University), Sungsoo Ray Hong `[通讯]` (George Mason University)

**通讯引用:** 1532 | [OpenAlex ID](https://openalex.org/A5059548204)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过焦点小组访谈，探讨了城市搜索任务中警察的工作流程、挑战及其对可携带、低成本地面机器人队列的需求与期望，提出六项系统设计要求；

**💡 创新点**

创新点在于将多机器人协作、可解释AI与视觉提示与警察现有流程对接，强调“可审计性”和“人机协作”的系统设计框架，填补了现有技术与操作实践之间的鸿沟；

**🔧 技术方法**

研究主要采用焦点小组访谈和主题分析方法，并结合人机交互、机器人自治与计算机视觉的相关文献综述，未实施机器人或AI系统；

**📊 数据集**

使用的“数据集”为五个弗吉尼亚州警局八名警官的访谈记录与现场讨论笔记；

**📈 对比分析**

本研究未进行性能比较或定量评估，研究结果为基于访谈的质性洞察，缺乏对技术性能的量化验证；

**⚠️ 局限性**

局限包括样本规模小（仅八名警官）、地域限定（仅弗吉尼亚州）、未实际部署机器人导致缺乏现场使用数据、未录音导致可能遗漏细节，且研究仅为假设性探讨。

---

## 39. Dual-Representation Image Compression at Ultra-Low Bitrates via Explicit Semantics and Implicit Textures

**arXiv ID:** 2602.05213 | [PDF](https://arxiv.org/pdf/2602.05213v1)

**作者:** Chuqin Zhou `[一作]` (Shanghai Jiao Tong University), Wenjun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22741 | [OpenAlex ID](https://openalex.org/A5100447801)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个双分支框架，融合显式语义（量化潜在与标签式提示）与隐式纹理（通过逆通道编码在扩散轨迹上编码），实现极低比特率图像压缩。

**💡 创新点**

创新点包括：①在不训练的前提下通过插值插件同时控制失真与感知质量；②使用标签式提示替代冗长的标题式描述，显著降低显式信息比特率；③将显式与隐式信息在条件扩散模型中协同使用，弥补单一分支的语义与纹理缺陷；④采用 tile‑based latent 推理提升高分辨率处理效率。

**🔧 技术方法**

核心技术：预训练的条件扩散模型（LDM）、VAE 编码/解码、逆通道编码（RCC）用于隐式信息压缩、ControlNet/交叉注意力实现条件扩散、标签式提示提取与固定长度编码、插值插件实现失真‑感知调节、tile‑based 过lap 处理。

**📊 数据集**

实验主要使用 Kodak、DIV2K 和 CLIC2020 进行压缩评估，插件编码器在 Flickr2W 上训练。

**📈 对比分析**

与 MS-ILLM、GLC、DiffEIC、PerCo、OSCAR、ResULIC、RDEIC、DiffC 等基线比较，BD‑Rate 在 DISTS、CLIPSim、FID 上下降 29–48%，在 0.02 bpp 以下区间获得最优感知与语义一致性，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性：解码时间仍在秒级，计算开销高；依赖预训练扩散模型与 VAE，未实现实时性能；RCC 代码量大且在 KL 较大时效率下降；在极端低比特率下仍可能出现细节丢失或轻微语义漂移。

---

## 40. Formal Synthesis of Certifiably Robust Neural Lyapunov-Barrier Certificates

**arXiv ID:** 2602.05311 | [PDF](https://arxiv.org/pdf/2602.05311v1)

**作者:** Chengxiao Wang `[一作]` (University of Illinois), Gagandeep Singh `[通讯]` (University of Illinois)

**通讯引用:** 2538 | [OpenAlex ID](https://openalex.org/A5100760604)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并提出了在存在动力学扰动时能够保证深度强化学习控制器安全与稳定的鲁棒神经 Lyapunov‑Barrier 证书的合成方法。

**💡 创新点**

通过将 Lyapunov 减小条件扩展到扰动邻域，利用 Lipschitz 连续性推导出鲁棒性充要条件，并提出对抗训练、邻域约束和全局 Lipschitz 正则化三种训练目标实现鲁棒证书。

**🔧 技术方法**

采用对抗训练（PGD）、基于全局 Lipschitz 常数的邻域约束、全局 Lipschitz 正则化、CEGIS 循环与 Marabou 证明器、Spectral norm 上的 Lipschitz 上界等技术。

**📊 数据集**

在 2D Docking 和 Inverted Pendulum 两个强化学习环境（典型的控制任务）上进行实验。

**📈 对比分析**

与标准（vanilla）Lyapunov 证书基线比较，在 2D Docking 证书范围提升至 4.6 倍、经验成功率提升 2.4 倍；在 Inverted Pendulum 也取得类似提升，显著优于基线。

**⚠️ 局限性**

仅考虑状态扰动，未覆盖动力学、观测或控制输出等其他不确定性，未来需扩展到更广泛的扰动来源。

---

## 41. Templated Assembly Theory: An Extension of the Canonical Assembly Index with Block-Compressed Template

**arXiv ID:** 2602.04889 | [PDF](https://arxiv.org/pdf/2602.04889v1)

**作者:** Piotr Masierak `[一作]` `[通讯]` (Lukaszyk Patent Attorneys), Piotr Masierak (Lukaszyk Patent Attorneys)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了模板化装配理论（Templated Assembly Theory），在传统Assembly Theory的基础上引入了块压缩模板与通配符（*）插入操作，定义了新的模板化装配指数（TAI）来衡量字符串或序列在使用模板化复制时的复杂度，并给出了完整的形式化定义与计算方法。

**💡 创新点**

创新点在于：①将通配符模板与原子级拼接结合，形成新的装配规则集合；②证明了模板化装配指数严格小于等于传统指数，并能够捕捉非局部的模板化模块化；③将该模型与最小语法、宏语法、模式语言等已有理论建立关联；④提出了基于宏语法的贪心启发式，用于逼近TAI。

**🔧 技术方法**

主要技术包括：①形式化装配空间与模板集的定义；②拼接与模板实例化规则的构造；③NP可判定性的证明与NP难度的猜想；④示例构造与贪心宏规则启发式算法。

**📊 数据集**

论文中使用的主要数据集是人工构造的示例字符串（如w、w1、w2等），用于展示传统装配指数与模板化装配指数的差异，并未使用大规模公开数据库。

**📈 对比分析**

比较方法是对同一字符串同时计算传统装配指数（ASI）与模板化装配指数（TAI），并统计两者所需的最小拼接步数。示例中发现TAI比ASI少1或2步，证明模板化操作能够有效减少装配步骤；性能评估仅限于这些手工示例，没有大规模实验。

**⚠️ 局限性**

局限性包括：①尚未给出模板化装配指数的NP‑hard证明，理论复杂度仍不确定；②提出的贪心启发式仅在示例中验证，缺乏大规模实验与性能评估；③模板形式被限定为块压缩子串，无法覆盖更一般的模板化模式；④未在真实生物或化学数据集上进行实证验证。

---

## 42. Cross-Domain Few-Shot Segmentation via Multi-view Progressive Adaptation

**arXiv ID:** 2602.05217 | [PDF](https://arxiv.org/pdf/2602.05217v1)

**作者:** Jiahao Nie `[一作]` (Nanyang Technological University), Shijian Lu `[通讯]` (Nanyang Technological University)

**通讯引用:** 16489 | [OpenAlex ID](https://openalex.org/A5023507910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究跨域少样本分割问题，提出多视图渐进适应（MPA）框架，旨在在目标域数据稀缺且与源域差异大的情况下逐步建立少样本分割能力。

**💡 创新点**

创新点在于两方面：①从数据视角设计混合渐进增强（HPA），通过累计强烈的数据增强逐步生成更复杂多样化的查询视图；②从策略视角设计双链多视图预测（DMP），同时使用序列链和并行链，并对每一步提供监督，从而充分利用增强视图并抑制误差积累。

**🔧 技术方法**

技术手段包括：基于ResNet‑50的编码器，元学习框架，混合渐进增强（HPA）、双链多视图预测（DMP）、密集监督与反向预测、交叉熵与BCE损失等。

**📊 数据集**

实验数据集涵盖五个数据稀缺域：Deepglobe（卫星影像）、ISIC2018（皮肤病变）、Chest X‑Ray（肺部X光）、FSS‑1000（自然小目标）和SUIM（水下目标）。

**📈 对比分析**

与IFA、PATNet、SSP、CPFT、TAFT等最新方法对比，MPA在1‑shot和5‑shot场景下平均提升约7% mIoU，并在无源训练条件下仍优于SOTA，提升幅度可达5–12%。

**⚠️ 局限性**

局限性包括：需要手动设计和调参增强累积规则与视图数，参数权重设置较多；在极端域差或极少样本的极端场景下效果可能受限；目前主要关注语义分割，尚未扩展至更广泛的跨域任务。

---

## 43. RAG without Forgetting: Continual Query-Infused Key Memory

**arXiv ID:** 2602.05152 | [PDF](https://arxiv.org/pdf/2602.05152v1)

**作者:** Yuntong Hu `[一作]` (Emory University), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 6684 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无监督、零推理开销的检索记忆演化框架（ERM），通过将查询时的扩展转化为持久的检索键更新，提升了检索质量和下游生成性能。

**💡 创新点**

创新点在于：1）正确性门控反馈机制，只有在检索或生成验证通过时才触发键更新；2）选择性扩展归因，将每个扩展单元仅归属于受益的文档键；3）渐进键演化策略，使用批量、受限幅度的更新并证明收敛性，避免语义漂移。

**🔧 技术方法**

使用的技术包括：检索-生成（RAG）体系、查询扩展（QE）与键扩展（KE）方法、相似度函数（内积/点积）、软最大归因、累积更新、稳定收敛证明及推理时间对比分析。

**📊 数据集**

实验数据集涵盖BEIR与BRIGHT的13个领域，检索器包括BM25、BGE、GTE、MiniLM、Cohere和Voyage，检索索引方式包括标题、关键词、摘要和全文。

**📈 对比分析**

与HyDE、BGE-Reasoner-Rewriter、T5L-Turbo Query Rewriting等查询重写/扩展方法以及键扩展基线进行对比，ERM在nDCG@10上对BM25平均提升约46%，对密集检索器提升13-15%，在生成质量上平均提升2-4%，且推理延迟与原生检索相近（仅几百毫秒）。

**⚠️ 局限性**

局限性包括：对早期错误检索的正反馈可能放大偏差；对极少见查询的更新效果有限；需要足够的查询量以实现有效演化；对已强大检索器的提升空间相对有限。

---

## 44. Crypto-asset Taxonomy for Investors and Regulators

**arXiv ID:** 2602.05098 | [PDF](https://arxiv.org/pdf/2602.05098v1)

**作者:** Xiao Zhang `[一作]` (Exponential Science), Jiahua Xu `[通讯]` (Department of Computer Science, University College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个多维度的加密资产分类体系，涵盖技术标准、中心化程度、资产功能、机制设计、法定分类等维度；

**💡 创新点**

创新点在于将技术设计、经济功能与监管框架统一映射到资产层面，提供可操作的传统金融类比与合规路径；

**🔧 技术方法**

主要使用结构化文献回顾、法规文本分析、基于Python的dataclass实现和决策树规则；

**📊 数据集**

数据集为2025年11月19日CoinGecko前100大市值加密资产的公开资料和链上持有者信息；

**📈 对比分析**

通过手工案例研究和对100资产的量化映射，展示了不同资产在分类维度上的分布与传统金融类比的一致性；

**⚠️ 局限性**

局限包括分类体系固定不易跟随生态快速演变、案例研究主观判断、对链外资产信息覆盖不足以及缺乏跨国监管细节整合。

---

## 45. CORP: Closed-Form One-shot Representation-Preserving Structured Pruning for Vision Transformers

**arXiv ID:** 2602.05243 | [PDF](https://arxiv.org/pdf/2602.05243v1)

**作者:** Boxiang Zhang `[一作]` (Purdue University), Baijian Yang `[通讯]` (Purdue University)

**通讯引用:** 1620 | [OpenAlex ID](https://openalex.org/A5089355143)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种闭式一轮结构化剪枝框架 CORP，能够在无标签、无梯度、无微调的部署环境下，对 Vision Transformers 进行高效剪枝。

**💡 创新点**

创新点是将剪枝视为表示恢复问题，利用闭式岭回归等方法为被剪除的 MLP 隐藏维度和注意力子结构提供补偿；不需要迭代或再训练，显著提升一轮剪枝的可行性。

**🔧 技术方法**

使用低秩近似、闭式岭回归求解补偿系数、对 MLP 激活和注意力 logit 进行线性/雅可比建模，并通过小型无标签校准集估计统计量。

**📊 数据集**

在 ImageNet‑1K 数据集上，对 DeiT 系列模型（Tiny 到 Huge）进行实验。

**📈 对比分析**

与 OPT‑IN、SNOW 等现有方法在相同 FLOPs 下对比，CORP 在 DeiT‑Base/Huge 50% 结构稀疏时 Top‑1 准确率仅下降 2–3%，并在单 GPU 20 分钟内完成剪枝；实现显著 FLOPs、吞吐量提升。

**⚠️ 局限性**

限制在于对 MLP 与 Attention 的补偿独立处理，未建模两者交互；依赖小规模校准集，对分布漂移敏感；在极端稀疏下仍不可逆信息损失。

---

## 46. E-Globe: Scalable $ε$-Global Verification of Neural Networks via Tight Upper Bounds and Pattern-Aware Branching

**arXiv ID:** 2602.05068 | [PDF](https://arxiv.org/pdf/2602.05068v1)

**作者:** Wenting Li `[一作]` (University of Texas at Austin), Huan Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 13772 | [OpenAlex ID](https://openalex.org/A5020766468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种混合式可验证器E‑Globe，在分支定界框架下通过紧凑上界（NLP‑CC）与下界（β‑CROWN）共同逼近神经网络的全局最优，支持ϵ‑全局最优的可扩展验证；

**💡 创新点**

① 用补充性约束的NLP‑CC实现对ReLU的精确编码，得到真正可行的上界；② 通过warm‑start与低秩KKT更新显著加速NLP求解；③ 采用模式对齐的强分支策略，利用上界得到的激活模式快速提升下界；

**🔧 技术方法**

NLP‑CC（补充性约束的非线性规划）、IPOPT内部点法、β‑CROWN松弛、分支定界、warm‑start与低秩矩阵更新、GPU批量计算；

**📊 数据集**

MNIST与CIFAR‑10两大公开图像分类数据集；

**📈 对比分析**

与CROWN‑IBP、CROWN、α‑CROWN、PGD以及MIP完整求解器比较。上界紧凑度优于PGD，误差仅几千分之一；相较于MIP，E‑Globe在二进制变量数≥180时可获得两至三位数的速度提升；上界求解在大扰动下保持稳定，整体时间比MIP显著缩短；

**⚠️ 局限性**

在二进制变量极少或扰动非常小的情况下，E‑Globe的优势不明显；对激活模式不严格符合补充性约束的情况，UP‑CC的上界可能略微松弛；在极大网络或极大扰动下，分支树仍可能膨胀，导致整体耗时升高。

---

## 47. Blockchain Technology for Public Services: A Polycentric Governance Synthesis

**arXiv ID:** 2602.05109 | [PDF](https://arxiv.org/pdf/2602.05109v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 48. Optimizing Mission Planning for Multi-Debris Rendezvous Using Reinforcement Learning with Refueling and Adaptive Collision Avoidance

**arXiv ID:** 2602.05075 | [PDF](https://arxiv.org/pdf/2602.05075v1)

**作者:** Agni Bandyopadhyay `[一作]` (Julius-Maximilians-University Wuerzburg), Gunther Waxenegger-Wilfing `[通讯]` (DLR Lampoldshausen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了基于强化学习的多目标碎片清除任务规划。

**💡 创新点**

创新点在于结合掩码PPO实现自适应碰撞规避与加油决策。

**🔧 技术方法**

使用了掩码近端策略优化（Masked PPO）以及Hohmann转移轨道模型。

**📊 数据集**

使用Iridium 33碎片数据集生成随机碎片场景。

**📈 对比分析**

与贪心与混合基线对比，RL方案在碰撞率低、碎片覆盖率高方面表现更优。

**⚠️ 局限性**

局限在于仅考虑单一卫星、无J2扰动、离线训练。

---

## 49. Internalizing LLM Reasoning via Discovery and Replay of Latent Actions

**arXiv ID:** 2602.04925 | [PDF](https://arxiv.org/pdf/2602.04925v1)

**作者:** Zhenning Shi `[一作]` (Tsinghua University), Congcong Miao `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

STIR通过动态隐状态控制实现内部推理的自我纠错；

**💡 创新点**

创新点在于把推理错误的隐状态差分成可复用的“工具”，并在推理时动态检索和注入这些工具；

**🔧 技术方法**

核心技术包括差分内在动作诱导、稀疏控制基底构建与基于价值的轨迹干预；

**📊 数据集**

实验数据集涵盖算术推理（AIME 2024/2025、AMC 2023、MATH‑500）与常识/逻辑问答（ARC‑Challenge、OpenBookQA）；

**📈 对比分析**

相较于vanilla、Self‑Consistency、Self‑Discover、DEER、SEAL等基线，STIR平均提升 1.9%–7.5% 的准确率，同时在 token 消耗上降低 10%–35%，构成新的精度–效率 Pareto 前沿；

**⚠️ 局限性**

局限性包括需离线构造工具库、对注入强度和层级的敏感性，以及在极端复杂任务或跨域迁移时可能失效。

---

## 50. Bagpiper: Solving Open-Ended Audio Tasks via Rich Captions

**arXiv ID:** 2602.05220 | [PDF](https://arxiv.org/pdf/2602.05220v1)

**作者:** Jinchuan Tian `[一作]` (Carnegie Mellon University), Shinji Watanabe `[通讯]` (Carnegie Mellon University)

**通讯引用:** 24983 | [OpenAlex ID](https://openalex.org/A5001291873)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建8B音频基础模型Bagpiper，通过生成丰富字幕实现音频理解与生成的一体化；

**💡 创新点**

创新点在于使用丰富字幕作为全局语义中介，实现物理音频与高层认知概念的双向映射，并通过“Caption-then-Process”训练实现无任务标注的开放式任务解决；

**🔧 技术方法**

采用预训练+自监督、LLM解码器+音频编码器、CLAP、UTMOS、audiobox-aesthetics、CLAP对齐、Gumbel Top‑k、LLM-as-judge、chain-of-thought等技术；

**📊 数据集**

数据集包含约422M条音频‑字幕对，覆盖语音、音乐、环境音；预训练使用600B token，微调使用约845k理解样本和1.47M生成样本；评测基准包括LibriSpeech、MMAU、AIR‑Bench、AudioBench、TTS、TTA等；

**📈 对比分析**

与现有7B/30B模型相比，Bagpiper在ASR、MMAU、AIR‑Bench、AudioBench以及TTS/FID等指标均达到或超过SOTA，生成质量优于TangoFlux、AudioLDM2‑Large，生成多模组合能力突出；

**⚠️ 局限性**

局限在于推理时长高、对高度组合/冲突指令的满足度有限，且模型可能存在偏差及版权风险。

---

## 51. GAMMS: Graph based Adversarial Multiagent Modeling Simulator

**arXiv ID:** 2602.05105 | [PDF](https://arxiv.org/pdf/2602.05105v1)

**作者:** Rohan Patil `[一作]` (University of California San Diego), Henrik I. Christensen `[通讯]` (University of California San Diego)

**通讯引用:** 12798 | [OpenAlex ID](https://openalex.org/A5066237365)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一个基于图的多代理仿真器GAMMS，用于在大规模、真实世界地理数据环境下快速测试和调试多智能体策略。

**💡 创新点**

通过“integration‑first”和“单一访问点”架构实现易用性与可扩展性；采用中间表示记录与可视化；基于OSM的统一图构建；支持低成本高可扩展的仿真而非物理细节。

**🔧 技术方法**

Python＋NetworkX＋OSMnx；图模型、传感器、代理、记录器、可视化模块；基于策略模式的行为编程；与ROS2等高保真工具集成的可能性。

**📊 数据集**

OpenStreetMap街道网络（如La Jolla、Manhattan）以及手工构造的网格世界。

**📈 对比分析**

与Gazebo/Isaac Sim等高保真模拟器在计算成本与规模上对比；与Mesa、Repast等低保真框架对比，展示可扩展至千节点、每秒多千代理的仿真，并提供实时可视化反馈。

**⚠️ 局限性**

缺乏真实物理和高保真传感器模型；仿真精度与实际机器人差距；仅适用于可图形化的环境，难以映射到非网络结构任务；多机器人交互细节支持有限。

---

## 52. EBPO: Empirical Bayes Shrinkage for Stabilizing Group-Relative Policy Optimization

**arXiv ID:** 2602.05165 | [PDF](https://arxiv.org/pdf/2602.05165v1)

**作者:** Kevin Han `[一作]` (Meta AI), Lizhu Zhang `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Empirical Bayes Policy Optimization (EBPO)，通过在线 Empirical Bayes 推断把每个 prompt 的局部奖励均值与全局成功率进行 shrinkage，解决 GRPO 在小组规模和饱和失败时梯度消失与高方差的问题。

**💡 创新点**

创新点在于：
1) 将 Empirical Bayes 视角引入无 critic 的 RLVR 训练，构造可自适应的 shrinkage 基线；
2) 在线估计全局均值与方差（使用 Welford 算法），避免了额外的超参数调优；
3) 通过主题或难度聚类的序列采样，使全局先验更贴近局部任务分布，从而进一步降低估计误差。

**🔧 技术方法**

技术手段包括：
- 基于 GRPO 的优势估计框架；
- Empirical Bayes 收缩估计器 (V_q^EB = (1‑S_q)·μ_group + S_q·μ_glob)；
- Welford 的在线均值/方差更新；
- 主题/难度聚类的训练数据重排；
- 对比实验使用 Pass@1、Majority‑Vote@16 等指标。

**📊 数据集**

使用的数据集：
- 训练集：DAPO‑Math‑17K；
- 评估基准：AIME‑2024、AIME‑2025、AMC23、Math‑500、OlympiadBench；
- 模型：LLaMA3.1‑8B、Qwen3‑8B、Qwen3‑14B。

**📈 对比分析**

与 GRPO、DAPO、DrGRPO、EntropyMech 等基线在相同训练条件下比较，结果显示：
- 在 4、8、16、32 等小组规模下，EBPO 在大多数模型/数据集上获得最高 Pass@1；
- 在 8 组时，EBPO 的平均提升约 11%；
- 通过主题聚类（-topic）和难度课程（-diff）进一步提高性能，尤其在高难度评测（AIME、OlympiadBench）中优势明显；
- 在梯度范数、KL 收敛和策略熵等训练动态指标上，EBPO 保持较大梯度、受限的更新幅度及更高的探索度。

**⚠️ 局限性**

局限性：
- 采用高斯近似而非 Beta‑Binomial，可能在极端稀疏奖励下产生偏差；
- 需要在线维护全局统计，若数据分布剧烈变化或聚类不充分，收缩效果会下降；
- 对分群质量高度依赖，若聚类误差大则全局先验失真；
- 目前仅验证在数学推理/竞赛问题上，未知在其它类型推理或更大模型上的泛化情况。

---

## 53. PatchGuru: Patch Oracle Inference from Natural Language Artifacts with Large Language Models

**arXiv ID:** 2602.05270 | [PDF](https://arxiv.org/pdf/2602.05270v1)

**作者:** Thanh Le-Cong `[一作]` (University of Melbourne), Cristian Cadar `[通讯]` (Imperial College London)

**通讯引用:** 8149 | [OpenAlex ID](https://openalex.org/A5053355200)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型（LLM）从拉取请求（PR）的自然语言描述中自动推理并生成可执行的补丁或然式规范（patch oracle），通过比较前后版本的运行时断言来验证补丁是否符合开发者意图。

**💡 创新点**

首次提出将补丁或然式规范作为可执行的运行时断言，并通过LLM抽取开发者意图、迭代完善、自动回顾等机制，弥补补丁验证中缺乏可执行规格的空白。

**🔧 技术方法**

核心技术包括：GPT‑5‑mini LLM调用、AST 与 libCST 静态代码分析、Docker 容器化执行、动态运行时分析、错误修复与自评审模块。

**📊 数据集**

在四大 Python 开源项目（Keras、Marshmallow、Pandas、SciPy）的最近单函数 PR 集合上进行实验，构建了相应的 PR 数据集与补丁或然式规范。

**📈 对比分析**

与 Testora 和开发者手写回归测试对比，PatchGuru 在 400 PR 上发现 24 条真实 bug（精度 0.62）高于 Testora（精度 0.32），平均 mutation score 为 0.70，高于回归测试 0.58；平均耗时 8.9 分钟、每 PR 费用约 0.07 美元。

**⚠️ 局限性**

局限性包括：仅支持单函数补丁、LLM 生成的程序与输入可能不现实导致假阳性、对多函数或跨文件更改缺乏支持、对 LLM 性能与 token 限制敏感，需进一步提升泛化与鲁棒性。

---

## 54. AirGlove: Exploring Egocentric 3D Hand Tracking and Appearance Generalization for Sensing Gloves

**arXiv ID:** 2602.05159 | [PDF](https://arxiv.org/pdf/2602.05159v1)

**作者:** Wenhui Cui `[一作]` (Meta), Li Guan `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在传感手套上使用基于视觉的手部跟踪模型，系统评估了裸手模型在不同手套上的性能衰减，并提出了 AirGlove 框架，利用对抗学习实现手套外观不变的表示，从而在有限标注数据下提升对新手套的跟踪能力。

**💡 创新点**

①首个针对传感手套的基于视觉跟踪模型系统评估；②提出的 AirGlove 在多手套数据上通过能量化对抗学习实现外观不变特征；③在不需要大规模标注的情况下，能显著提升对未知手套的泛化性能。

**🔧 技术方法**

使用了基于视频的时间感知视觉网络（TADV‑Net）和多视角解码器；对抗外观不变判别器（AAID）通过 KL 散度实现外观信息消除；交替优化策略、t‑SNE 可视化、MKPE/F‑MKPE 评估指标等。

**📊 数据集**

构建了多手套数据集，包括 IMU‑Glove、PS‑Glove、MoCap‑Glove、Haptic‑Glove 四种手套以及裸手数据，总计数百万帧，配备 OptiTrack 提供的 3D 关键点标注。

**📈 对比分析**

将 MEgATrack 与 UmeTrack 两大裸手基线模型直接在手套上测试，发现显著性能下降；随后在新手套上进行 20%–100% 数据比例的微调比较，AirGlove 在所有指标上均优于基线和从零开始训练的模型；消除对抗损失的消融实验表明该损失能有效抑制外观信息，提升泛化。

**⚠️ 局限性**

缺乏大规模、覆盖更广手套种类的公开数据集；对抗训练可能存在收敛不稳定的问题；仅在四种手套和室内实验环境下验证，未知环境下的鲁棒性待进一步研究。

---

## 55. Quality Model for Machine Learning Components

**arXiv ID:** 2602.05043 | [PDF](https://arxiv.org/pdf/2602.05043v1)

**作者:** Grace A. Lewis `[一作]` (Carnegie Mellon Software Engineering Institute), Katherine R. Maffey `[通讯]` (Carnegie Mellon University)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5071449587)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本研究提出并验证了针对机器学习组件的质量模型，为模型开发与测试提供了结构化的质量属性框架，并将其集成到开源工具MLTE中。

**💡 创新点**

创新点在于将ISO 25010与AI专用标准结合，仅保留可在组件层面评估的质量属性，区分系统与组件属性，形成可测试的属性集合，并通过实证调查验证其实用性。

**🔧 技术方法**

采用卡片排序法、专家访谈与问卷调查收集质量属性，并利用MLOps工具MLTE实现测试目录与评估。

**📊 数据集**

使用的数据主要来自22位来自政府、科研机构和工业组织的参与者问卷结果；质量属性来源为ISO标准、相关论文与专家经验，并未使用公开机器学习数据集。

**📈 对比分析**

通过问卷评估属性被测试的频率、重要性与难度，结果显示预测准确性最常被测试，其他属性的重要性与难度分布相对均衡；模型已在MLTE中实现，测试目录覆盖每一属性。

**⚠️ 局限性**

局限性包括样本量仅22人、未进行大规模实验验证、仅关注传统模型与LLM的通用属性、缺乏针对特定任务或模型架构的评估，以及负责AI属性重要性低估可能受限。

---

## 56. Comparing Euclidean and Hyperbolic K-Means for Generalized Category Discovery

**arXiv ID:** 2602.04932 | [PDF](https://arxiv.org/pdf/2602.04932v1)

**作者:** Mohamad Dalal `[一作]` (Aalborg University), Joakim Bruslund Haurum `[通讯]` (University of Southern Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种在 Lorentz 双曲超球模型中直接进行 K‑Means 聚类的通用类别发现（HC‑GCD）方法，并与现有 Euclidean K‑Means 及 Hyp‑GCD（Poincaré 模型）进行对比。

**💡 创新点**

创新点在于将 K‑Means 推广到 Lorentz 双曲几何，证明 Einstein 中点等价于 Lorentz 质心，从而无需在不同模型间转换；以及利用对比学习在双曲空间保留层次结构，从而提升聚类性能。

**🔧 技术方法**

采用 Lorentz Hyperboloid 的对比学习（距离与角度损失）、指数映射、超弧距离、超曲 K‑Means 以及欧氏剪裁等技术。

**📊 数据集**

在 Semantic Shift Benchmark 上的 CUB、Stanford Cars 与 FGVC‑Aircraft 三个细粒度数据集上进行实验。

**📈 对比分析**

通过与 Hyp‑GCD、Euclidean K‑Means 及 Poincaré‑K‑Means 的对比，HC‑GCD 在 All、Old、New 准确率上均与或优于 Hyp‑GCD；使用双曲 K‑Means 能显著提升准确率并降低方差，并在不同标签粒度下获得更高的同质性。

**⚠️ 局限性**

局限性包括：未在 Poincaré 空间实现可行的 K‑Means，且未结合主流非参数方法（如 Hyp‑SelEx）进一步验证双曲聚类的优势。

---

## 57. Explainable AI: A Combined XAI Framework for Explaining Brain Tumour Detection Models

**arXiv ID:** 2602.05240 | [PDF](https://arxiv.org/pdf/2602.05240v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 58. Fairness Under Group-Conditional Prior Probability Shift: Invariance, Drift, and Target-Aware Post-Processing

**arXiv ID:** 2602.05144 | [PDF](https://arxiv.org/pdf/2602.05144v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Kaveh Aryan `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究在组内先验概率偏移（GPPS）情境下，机器学习模型公平性随分布变化的行为，并提出一种无标签后处理方法（TAP‑GPPS）来恢复目标域的公平性。

**💡 创新点**

创新点在于①将公平性指标分为结构不变的分离型指标（如等化偶然性）与易漂移的接受率/预测值指标；②证明非平凡分类器在不同偏移下无法同时满足人口统计平等（impossibility）；③在GPPS下证明目标域风险与公平度量可无标签识别；④设计基于目标先验估计与阈值调节的完整后处理框架。

**🔧 技术方法**

采用的技术包括：
- 基于GPPS的贝叶斯先验校正公式；
- EM 与 BBSE 方法估计目标先验概率；
- 通过阈值二分实现目标域公平约束；
- 目标域风险的无标签估计与有限样本误差分析；
- 传统分类模型（Logistic、XGBoost、MLP）与阈值后处理组合。

**📊 数据集**

实验使用的公开数据集有：UCI Adult、COMPAS、MEPS，此外构造了高斯仿真数据；对每个数据集在训练（source）与部署（target）之间人为设置GPPS，调整各组的正例比例。

**📈 对比分析**

与基线比较：source‑only（无调整）、无校正的DP阈值平衡、以及使用目标标签的oracle阈值。结果显示，等化偶然性在GPPS下几乎保持不变；人口统计平等在source‑only中漂移明显，TAP‑GPPS将DP gap压缩到≈0.02，且准确率仅下降≈2%；相比之下，单纯阈值调整的No‑correction DP也能降低DP gap，但TAP‑GPPS在无标签情境下实现了更稳健的风险估计和更低的误差。

**⚠️ 局限性**

局限性包括：
- 需要目标域中可观测的敏感属性及足量未标记样本；
- 只针对二分类且单一组属性，扩展到多类别或连续属性仍需研究；
- 结果基于GPPS假设，若真实环境存在更复杂的分布偏移或特征分布变化，方法效果未知；
- 对公平性阈值的选择仍需人工设定，可能导致公平-效用权衡不可控。

---

## 59. On the Reachability Problem for One-Dimensional Thin Grammar Vector Addition Systems

**arXiv ID:** 2602.05315 | [PDF](https://arxiv.org/pdf/2602.05315v1)

**作者:** Chengfeng Xue `[一作]` (Shanghai Jiao Tong University), Yuxi Fu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1429 | [OpenAlex ID](https://openalex.org/A5034346547)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了薄型一维文法向量加法系统（thin 1‑GVAS）的可达性问题，提出了一种新的算法框架；

**💡 创新点**

其创新点在于将经典的 KLM 分解方法推广到 KLM 树上，并通过整数规划和层次化秩函数实现对求解空间的有效约束；

**🔧 技术方法**

主要技术包括：KLM 树的构造、特征系统（ILP）描述、秩函数与层次化递归、配置约束、正交化、代数与组合分解以及泵送技术；

**📊 数据集**

论文未使用实验数据集，而是以理论证明为主，给出了上界的严格计算；

**📈 对比分析**

与之前的 ∪_{k} (6k-4) 上界相比，提出的 ∪_{k} (2k) 上界在快增长层级上显著更低，证明了更高效的非确定性算法；

**⚠️ 局限性**

局限性在于仍无法突破 Ackermann 上界：对一般 GVAS 的可达性问题仍为 Ackermann‑可决；此外尚无关于指数参数 k 的下界或多维情况的完整结果。

---

## 60. MINT: Minimal Information Neuro-Symbolic Tree for Objective-Driven Knowledge-Gap Reasoning and Active Elicitation

**arXiv ID:** 2602.05048 | [PDF](https://arxiv.org/pdf/2602.05048v1)

**作者:** Zeyu Fang `[一作]` (George Washington University), Mahdi Imani `[通讯]` (Northeastern University)

**通讯引用:** 1467 | [OpenAlex ID](https://openalex.org/A5017741575)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为MINT的神经符号树，用于在开放世界的语言交互式联合规划中主动提问以弥补知识缺口，并通过自我对弈和LLM摘要来优化提问策略。

**💡 创新点**

创新点在于将符号推理、基于不确定性估计的深度Q网络以及大型语言模型结合，构建可解释的知识缺口树并主动生成高信息增益的二进制查询，从而在几次提问内逼近理想规划回报。

**🔧 技术方法**

核心技术包括：①Bootstrapped uncertainty‑aware DQN（UA‑DQN）用于估计不同知识缺口下的Q值均值与方差；②神经符号树MINT，用于递归拆分知识缺口并评估提问效益；③LLM（GPT‑4o 等）对树进行压缩、合并等操作并生成最优查询；④基于假设的MDP家族与伪Lipschitz连续性证明给出回报上界。

**📊 数据集**

使用了三类基准数据集：MiniGrid（含未知物体），Atari Pacman（含不确定奖励/障碍），以及基于NVIDIA Isaac Gym的搜索救援仿真，分别对应离散、视觉和连续控制环境。

**📈 对比分析**

与纯RL（DQN、PPO）、纯LLM规划以及HITL‑RL（高不确定时请求专家动作）进行对比。MINT在所有环境中取得近专家级回报，仅需1–3个二进制问题；奖励和成功率明显优于基线，查询成本也比HITL‑RL低。

**⚠️ 局限性**

局限性包括：①仅支持二进制/yes‑no查询，未考虑更复杂自然语言交互；②LLM生成的答案在实验中由另一模型模拟，真实人类回答的可靠性尚待验证；③在极大规模或连续状态空间下构建和推理树的计算成本仍需进一步评估。

---

## 61. TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference

**arXiv ID:** 2602.05145 | [PDF](https://arxiv.org/pdf/2602.05145v1)

**作者:** Jiyoung Park `[一作]` (Moreh), Wookeun Jung `[通讯]` (Moreh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套名为TIDE的实时自适应草稿模型更新框架，在LLM推理过程中零开销地收集目标模型隐藏状态作为训练信号，并动态调整投机解码与训练策略，以提升推理吞吐量。

**💡 创新点**

创新点在于：①将训练信号收集、草稿模型更新与推理引擎紧密耦合，做到零额外推理开销；②利用自适应运行时控制，依据实时接受率动态开启/关闭投机解码与训练；③通过异构GPU分离推理与训练，充分发挥不同GPU的优势，实现资源高效利用。

**🔧 技术方法**

技术包括：SGLang/vLLM推理引擎；EAGLE-3轻量级草稿模型；PyTorch FSDP并行训练；EMA监控接受率；GPU异构调度（H100推理 + MI250训练）；性能模型预测投机解码加速。

**📊 数据集**

使用的目标模型有gpt-oss-120b、Qwen3-235B-A22B、Llama-4-Scout-17B-16E、Llama-3.3-70B-Instruct；数据集覆盖对话（ShareGPT）、科学文本（Science）、数学推理（NuminaMath）、代码生成（EvolCodeAlpaca）以及多语Alpaca（韩、阿、汉、法）。

**📈 对比分析**

通过与SpecForge离线/在线训练以及静态投机解码基线比较，TIDE在吞吐量上相较静态投机解码提升1.15×，训练时间比SpecForge离线快1.67×、比在线快3.02×；在异构GPU部署下实现整体吞吐提升1.08–1.22×，且模型精度保持与离线/在线训练相当。

**⚠️ 局限性**

局限性：在高熵开放式对话任务（如ShareGPT）中投机解码收益有限；对大规模草稿模型的性能预测误差较大；需要额外的异构GPU资源；对极端分布漂移的检测仍依赖接受率阈值，可能导致响应延迟。

---

## 62. Linear Model Merging Unlocks Simple and Scalable Multimodal Data Mixture Optimization

**arXiv ID:** 2602.04937 | [PDF](https://arxiv.org/pdf/2602.04937v1)

**作者:** Davide Berasi `[一作]` (University of Trento), Elisa Ricci `[通讯]` (University of Trento)

**通讯引用:** 10781 | [OpenAlex ID](https://openalex.org/A5065059558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了用模型合并作为数据混合优化的代理，训练域专家后线性合并以评估不同数据混合的效果。

**💡 创新点**

创新点在于证明并验证线性合并专家可作为混合训练模型的秩相关代理，从而在不重新训练每种混合的情况下高效选择最佳混合权重。

**🔧 技术方法**

采用LoRA或全微调训练域专家，使用简单的线性权重平均进行模型合并，并通过Spearman相关系数和二阶Taylor近似分析进行评估。

**📊 数据集**

使用了23个多模态SFT数据集，划分为四个领域（通用、多模态理解、OCR、视觉感知与计数、图表理解），并在14个基准（如GQA、OK‑VQA、VQAv2、DocVQA等）上进行评测。

**📈 对比分析**

与网格搜索得到的混合训练模型在14个基准上的性能进行Spearman相关和平均性能对比，线性合并代理在2-4个域、2B/7B/8B模型上均取得0.57-0.78的相关系数，所选混合在最差场景下仅下降≤1%，明显优于均匀混合。

**⚠️ 局限性**

局限性包括：在极小训练预算（10k）下的专家代理相关性差；在极端或高维混合空间中仍有失败案例；代理仅用于选择混合而不提供最终模型，且需要先训练所有域的专家。

---

## 63. Capacity Constraints and the Multilingual Penalty for Lexical Disambiguation

**arXiv ID:** 2602.05035 | [PDF](https://arxiv.org/pdf/2602.05035v1)

**作者:** Sean Trott `[一作]` (Rutgers University), Pamela D. Rivière `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过比较多语言和单语言语言模型在词义消歧任务中的表现，量化了多语言模型的“多语言惩罚”，并探究了三种潜在的容量限制因素（表示同质化、注意力分配不足、词表多词标记化）对该惩罚的影响。

**💡 创新点**

创新点在于将多语言模型在词义消歧任务中的低效性与可解释的容量瓶颈（嵌入同质化、注意力焦点、词元分割率）进行关联，并通过AIC模型证明这些因素能够解释多语言状态对性能的影响。

**🔧 技术方法**

使用的技术包括基于层级上下文化词嵌入的余弦距离计算、线性混合模型回归分析、中心同质性（CI）指标评估嵌入同质性、注意力权重聚合以及tokenization统计。

**📊 数据集**

采用了两份人类相关性判定数据集：英语的RAW-C（672句对）和西班牙语的SAW-C（812句对）。

**📈 对比分析**

方法是将模型层输出的词嵌入与人类判定进行R²回归，评估多语言与单语言模型的差异。结果显示，多语言模型在相同规模和层深下表现持续低于单语言模型，且多语言惩罚与三种容量限制指标呈显著相关。

**⚠️ 局限性**

局限性包括：仅覆盖英语和西班牙语两种语言，使用的模型族不包括最新模型，实验仅采用双向模型且数据量有限，且所有分析均为相关性研究，缺乏因果验证。

---

## 64. The Single-Multi Evolution Loop for Self-Improving Model Collaboration Systems

**arXiv ID:** 2602.05182 | [PDF](https://arxiv.org/pdf/2602.05182v1)

**作者:** Shangbin Feng `[一作]` (University of Washington), Wenhao Yu `[通讯]` (Tencent AI Seattle Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了单多演化循环（Single-Multi Evolution Loop），通过让多模型协作产生高质量文本，然后将协作结果蒸馏到单个模型，最后让蒸馏后的模型再次协作，形成迭代自我演化的过程。

**💡 创新点**

创新点在于：①将多模型协作系统作为教师进行知识蒸馏；②将协作与蒸馏交替循环，实现模型与协作系统的共同进化；③兼容多种协作方式（API路由、多代理辩论、logit融合、模型合并）和蒸馏策略（监督、多学生、logit蒸馏），在保持协作优势的同时显著降低推理成本。

**🔧 技术方法**

使用的技术包括多模型协作策略（API级路由、多代理辩论、logit融合、模型合并）、监督蒸馏、多学生蒸馏、logit基准蒸馏（on‑policy），以及迭代的单多演化循环框架（k 次迭代）。

**📊 数据集**

实验采用 15 个数据集，涵盖 QA（AGIEval、ARC、MMLU）、推理（BigBench‑hard、GSM8k、MATH、TheoremQA、GPQA、TableMWP）、知识（WikiDYK、PopQA、Blend）、安全、科学、指令跟随等多领域任务。

**📈 对比分析**

通过与多种协作策略、蒸馏方法和不同模型池进行对比实验，单模型平均提升 8.0%，协作系统平均提升 14.9%，相较于现有演化策略平均提升 7.7%。在推理与知识任务上，单多演化循环分别提升了约 16.84% 与 12.67%。

**⚠️ 局限性**

局限性包括：1）需要多模型并行加载，仍然存在硬件成本；2）logit 蒸馏对 tokenization 依赖较强；3）若协作模型有恶意或失效，演化过程可能受损；4）实验主要在特定模型与任务上验证，通用性和安全性仍需进一步研究。

---

## 65. Gabor Fields: Orientation-Selective Level-of-Detail for Volume Rendering

**arXiv ID:** 2602.05081 | [PDF](https://arxiv.org/pdf/2602.05081v1)

**作者:** Jorge Condor `[一作]` (USI Lugano), Piotr Didyk `[通讯]` (USI Lugano)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Gabor场（Gabor Fields）作为体素密度表示，能够基于频率与方向实现连续的细节层级控制（LOD）。

**💡 创新点**

创新点在于将可调谐的Gabor核与高斯核组合，推导出解析的行积分与分层回归方法，使得体素表示可在不增加显存的前提下实现频率/方向可控的LOD以及加速渲染。

**🔧 技术方法**

主要技术包括：3D 向量可调 Gabor 核、解析行积分与复杂误差函数近似、分层差分回归、光线可见性掩码、随机层级采样与方向性采样、以及多散射下的随机拉普拉斯估计。

**📊 数据集**

使用的公共数据集包括 OpenVDB Bunny、Tornado 动画、Disney Cloud、烟雾与爆炸等场景，均通过Voxel Grid 或 VDB 数据导入。

**📈 对比分析**

与传统高斯原始体素、Gaussian Splatting 等方法对比，Gabor Fields 在相同精度下渲染时间更短、PSNR 更高；通过掩码可实现实时 LOD 切换，渲染速度提升 2-3 倍，同时保持高质量细节。

**⚠️ 局限性**

主要局限包括：多散射场景下的偏差控制难度、可见性掩码受限于 8 位掩码导致方向/频率分层数受限、回归过程耗时高（数小时）、对极高频细节的性能下降，以及在极端视角/频率对齐时可能出现伪影。

---

## 66. SynAT: Enhancing Security Knowledge Bases via Automatic Synthesizing Attack Tree from Crowd Discussions

**arXiv ID:** 2602.05329 | [PDF](https://arxiv.org/pdf/2602.05329v1)

**作者:** Ziyou Jiang `[一作]` (State Key Laboratory of Intelligent Game), Qing Wang `[通讯]` (State Key Laboratory of Intelligent Game)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

自动化从 Stack Overflow 等知识共享平台的安全讨论中提取攻击事件和关系，并基于提取结果使用规则合成攻击树。

**💡 创新点**

① 采用 LLM 与 prompt learning 筛选包含攻击信息的句子范围；② 设计转移式联合事件‑关系提取模型；③ 通过手工设计的三条规则将提取结果映射成完整攻击树；④ 在公共与私有安全知识库中验证可行性。

**🔧 技术方法**

大语言模型（GPT‑3）、prompt learning、转移式事件‑关系提取框架（Stack‑LSTM + 多标签分类）、规则推理、实验对照基线模型（Structured‑Joint、ED3C、MLBiNet、KnowledgeILP、JC‑Learning 等）。

**📊 数据集**

5,070 条 Stack Overflow 安全帖子（扩增至 18,203 条）+ 2,350 条 GitHub Issue；公共知识库 CVE、CAPEC；Huawei 私有攻击树数据库。

**📈 对比分析**

通过与多种基线模型对比，攻击树合成的平均汉明距离（AHD）10.24%、树编辑距离相似度（TEDS）7.93%均优于基线；事件提取 F1 80.93%；关系提取 F1 87.81%；训练时长 18h；在实际增强知识库时，84% 的新树被接受，私有库增量 42%。

**⚠️ 局限性**

① 数据集规模有限，需人工标注；② 规则化构建可能遗漏特殊攻击场景；③ 低可读性或重复树的自动过滤尚未成熟；④ 对非文本信息（代码块、截图）支持不足，影响提取完整度。

---

## 67. ASA: Activation Steering for Tool-Calling Domain Adaptation

**arXiv ID:** 2602.04935 | [PDF](https://arxiv.org/pdf/2602.04935v1)

**作者:** Youjin Wang `[一作]` (Renmin University of China), Liangming Pan `[通讯]` (Peking University)

**通讯引用:** 997 | [OpenAlex ID](https://openalex.org/A5027533517)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种在推理时通过单次激活干预实现LLM工具调用的域自适应控制机制ASA。

**💡 创新点**

创新在于利用中间激活的线性可读性构建域感知向量混合，并引入基于探测器的符号门控实现无训练、可插拔的双向控制。

**🔧 技术方法**

采用激活向量对齐、轻量路由器、线性探测器、单步注入（gate+α·MoV）等技术。

**📊 数据集**

使用MTU-Bench（Code、Math、Search、Translation共1600条样本）以及Qwen2.5-1.5B、LLaMA-8B模型。

**📈 对比分析**

与Prompt-only、LoRA、Q-LoRA等基线对比，在严格触发协议下ASA在F1上提升约60–170%（域不同），且FPR保持低，存储仅≈20KB，训练成本几乎为零。

**⚠️ 局限性**

局限在于只能在已有工具调用能力的模型上工作，无法从零创建调用行为；域路由误判会导致误触发；对抗性提示仍可能削弱效果。

---

## 68. ReGLA: Efficient Receptive-Field Modeling with Gated Linear Attention Network

**arXiv ID:** 2602.05262 | [PDF](https://arxiv.org/pdf/2602.05262v1)

**作者:** Junzhou Li `[一作]` (University of Science and Technology of China), Li Xiao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 46422 | [OpenAlex ID](https://openalex.org/A5100452145)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种轻量化的混合CNN‑Transformer架构ReGLA，通过高效卷积与ReLU门控线性注意力实现对高分辨率图像的准确且低延迟推理，并通过多教师蒸馏提升下游任务性能。

**💡 创新点**

创新点在于：①引入ELRF模块以在保持大感受野的同时显著提升局部特征提取效率；②提出RGMA模块，将ReLU线性注意力与门控机制结合，既保持线性复杂度又提升表达能力；③构建多教师蒸馏策略，系统整合七种不同预训练模型以增强泛化；④将上述模块与轻量卷积共设计，形成完整的端到端高效模型。

**🔧 技术方法**

使用的技术包括ReLU基线线性注意力、门控卷积调制、深度可分离卷积、轻量化FFN替代方案、同步BN、余弦相似度损失、CLS自适应池化、以及基于UNIC的多教师蒸馏框架。

**📊 数据集**

实验数据集涵盖ImageNet‑1K/21K（分类）、COCO 2017（检测与实例分割）、ADE20K（语义分割）等主流视觉基准。

**📈 对比分析**

与同参数规模的MobileNet、EfficientNet、iFormer、FastViT、Swin等最新轻量模型在iPhone 16 Pro上对比，ReGLA-M在224 px下Top‑1 80.85%、512 px下4.98 ms延迟；在COCO检测与分割上分别提升约3‑5% AP；在ADE20K分割上提升3.6% mIoU；整体在参数、FLOPs与延迟上保持竞争力并往往领先。

**⚠️ 局限性**

局限性包括：①模型仍对大尺寸输入存在显存与算力约束；②多教师蒸馏过程复杂，训练成本高；③评测主要集中在iOS CPU，其他硬件平台及长序列/视频推理等情况尚未系统验证；④在极大模型规模下性能提升幅度趋于平缓。

---

## 69. Varifocal Displays Reduce the Impact of the Vergence-Accommodation Conflict on 3D Pointing Performance in Augmented Reality Systems

**arXiv ID:** 2602.05129 | [PDF](https://arxiv.org/pdf/2602.05129v1)

**作者:** Xiaodan Hu `[一作]` (Graz University of Technology), Kiyoshi Kiyokawa `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 3561 | [OpenAlex ID](https://openalex.org/A5022749721)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文通过对比可变焦与固定焦距AR显示器，在基于Fitts法则的三维点位任务中进行双盲用户实验，评估其对点位性能的影响。

**💡 创新点**

首次量化可变焦AR显示器对交互性能的提升，并揭示了基于用户基线表现的差异性效应，强调了个体差异在显示器评估中的重要性。

**🔧 技术方法**

使用光学可变焦AR立体显示（可调液晶透镜+分束镜），基于Unity渲染、OptiTrack运动捕捉、线性混合效应模型（LMM）进行统计分析。

**📊 数据集**

实验数据来源于两轮参与者实验：第一次24人，第二次12人；无公开数据集，完全由自制实验收集。

**📈 对比分析**

采用Within‑Subject Latin‑Square设计，使用Fitts法则和LMM检验，varifocal模式平均降低点位时间、提升信息吞吐量；效应大小约Cohen d≈0.4，显示显著优势；但效应随基线表现负相关，个体差异显著。

**⚠️ 局限性**

局限包括：受限视场与IPD范围、无真实场景背景、缺乏眼动触发焦点控制、性别分布不均、未探讨高阶交互情境下的可变焦效果。

---

## 70. Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution

**arXiv ID:** 2602.04918 | [PDF](https://arxiv.org/pdf/2602.04918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 71. Fast-SAM3D: 3Dfy Anything in Images but Faster

**arXiv ID:** 2602.05293 | [PDF](https://arxiv.org/pdf/2602.05293v1)

**作者:** Weilun Feng `[一作]` (University of Chinese Academy of Sciences), Yongjun Xu `[通讯]` (Institute of Computing Technology)

**通讯引用:** 5844 | [OpenAlex ID](https://openalex.org/A5103245119)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对单视图3D重建模型SAM3D的推理瓶颈进行系统分析，并提出了Fast‑SAM3D加速框架，利用异质感知的三种模块实现推理速度提升。

**💡 创新点**

创新点在于：①提出基于模态感知的步骤缓存，区分形状与布局的动力学差异；②设计时空令牌雕刻，针对纹理稀疏性动态聚焦高熵区域；③使用光谱感知令牌聚合，根据实例几何频谱自适应下采样；三者协同实现无训练的高效加速。

**🔧 技术方法**

技术手段包括：扩散模型推理加速（步骤缓存、令牌裁剪、分辨率自适应聚合）、基于令牌重要性与光谱熵的动态计算分配策略。

**📊 数据集**

使用数据集：Toys4K、Aria Digital Twin（ADT）用于几何精度和布局评估；ISO3D用于跨模态感知一致性评估；实验还参考了公开的SAM3D训练数据。

**📈 对比分析**

与SAM3D原始模型及TaylorSeer、EasyCache、Fast3DCache等基线进行对比，Fast‑SAM3D在场景生成和单物体生成中分别实现2.01×–2.67×的速度提升，几何指标（F‑Score、vIoU等）保持相当甚至略有提升。

**⚠️ 局限性**

局限性：①对极端遮挡或信息缺失场景的鲁棒性尚未充分验证；②加速机制仅针对单视图SAM3D架构，未考虑训练过程的加速；③在超大规模场景或高分辨率细节要求下，仍存在一定推理延迟。

---

## 72. SpectraKAN: Conditioning Spectral Operators

**arXiv ID:** 2602.05187 | [PDF](https://arxiv.org/pdf/2602.05187v1)

**作者:** Chun-Wun Cheng `[一作]` (University of Cambridge), Angelica I. Aviles-Rivero `[通讯]` (Tsinghua University)

**通讯引用:** 2255 | [OpenAlex ID](https://openalex.org/A5013015879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 SpectraKAN，一种结合多尺度 Fourier 网络与全局 Kolmogorov–Arnold Network (KAN) 条件化的谱算子，用于高效学习 PDE 解决算子。

**💡 创新点**

创新点在于将静态傅里叶卷积转化为输入条件的积分算子，利用 KAN 生成全局令牌并通过单查询注意力实现多尺度、可调节、非平稳的谱算子，同时保持傅里叶混合的计算效率与连续性。

**🔧 技术方法**

采用了 Adaptive Multi‑Scale FNO、Kolmogorov–Arnold Network (KAN)、单查询跨注意力的 Global Modulation Layer、连续积分算子理论与 Lipschitz 控制等技术。

**📊 数据集**

使用了 PDEBench（1D CNS、1D Diffusion–Reaction、2D Darcy flow、2D Shallow Water）以及 2D 气候模型和 Shallow Water 等公开数据集进行评估。

**📈 对比分析**

与 FNO、FFNO、LNO、Transovler、CViT、DeepOKAN、UNet、LocalFNO 等基线比较，RMSE 与相对 L² 均显著下降，平均提升约 30–50%，在时空预测和超分辨率任务上表现最佳。

**⚠️ 局限性**

局限性包括对全局令牌对全域状态建模的依赖，可能在极大尺度或不规则网格上适应性受限；模型复杂度与训练成本相对较高；理论证明基于理想连续化假设，实际离散化误差仍需进一步研究。

---

## 73. Towards Reducible Uncertainty Modeling for Reliable Large Language Model Agents

**arXiv ID:** 2602.05073 | [PDF](https://arxiv.org/pdf/2602.05073v1)

**作者:** Changdae Oh `[一作]`, Sharon Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了面向大型语言模型（LLM）代理的交互式、长期任务环境中的不确定性量化（UQ）框架，并给出了通用的形式化定义。

**💡 创新点**

创新点在于将不确定性视为可调节的双向流，通过信息门控函数捕捉可约不确定性，并给出了理论极限与可解释性分析。

**🔧 技术方法**

采用了动态贝叶斯网络建模、信息论度量（熵、互信息）以及动作分类与门控机制，并结合理论推导与符号分析。

**📊 数据集**

论文为概念性工作，未使用具体数据集；主要参考已有的代理基准与案例（如航班预订代理）。

**📈 对比分析**

由于缺乏实验实现，本文未给出对比实验或性能指标；主要通过理论比较说明现有UQ方法在代理场景中的局限性。

**⚠️ 局限性**

局限性包括：假设环境更新确定、观测可靠，未处理随机或对抗性环境；缺乏实际验证与可扩展的评测基准。

---

## 74. SHaSaM: Submodular Hard Sample Mining for Fair Facial Attribute Recognition

**arXiv ID:** 2602.05162 | [PDF](https://arxiv.org/pdf/2602.05162v1)

**作者:** Anay Majee `[一作]` (University of Texas at Dallas), Rishabh Iyer `[通讯]` (University of Texas at Dallas)

**通讯引用:** 1895 | [OpenAlex ID](https://openalex.org/A5000529247)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将公平表征学习视为子模子硬样本挖掘问题的两阶段框架，包含硬样本挖掘模块-MINE和联合学习模块-LEARN。

**💡 创新点**

创新点在于用子模函数（LogDet、Facility-Location）实现对硬正负样本的平衡选择，并在-LEARN中通过子模条件互信息统一最大化同类重叠、最小化敏感属性干扰，实现公平表征学习。

**🔧 技术方法**

采用子模子最大化/条件互信息的组合、贪心算法、ResNet‑18/ViT特征提取、对比学习与温度调制的投影层。

**📊 数据集**

在CelebA（目标属性如attractiveness、big nose、bags under eyes）和UTKFace（目标属性gender，敏感属性age/ethnicity）两个公开人脸属性基准上进行实验。

**📈 对比分析**

与Vanilla CE、GRL、LNL、FD‑VAE、MFD、FSCL、FairViT等SOTA方法对比，得到最多2.7分的Equalized Odds改进，Top‑1 Accuracy提升至4.1%（ResNet‑18）或2.95%（ViT），且在不同敏感属性不平衡比例下仍保持较好公平与准确率。

**⚠️ 局限性**

局限性包括：训练阶段需要多次子模优化导致每步计算开销较大；目前只针对二分类属性验证，扩展到多标签/连续属性尚待验证；对超参数（预算k、温度等）敏感。

---

## 75. Position: Universal Time Series Foundation Models Rest on a Category Error

**arXiv ID:** 2602.05287 | [PDF](https://arxiv.org/pdf/2602.05287v1)

**作者:** Xilin Dai `[一作]` (Zhejiang University), Qiang Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 14025 | [OpenAlex ID](https://openalex.org/A5088556682)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出时间序列统一基础模型存在范畴错误，并提出以因果控制代理为核心的多模块预测框架。

**💡 创新点**

① 用群论证明时间序列无共享对称性导致“零自由”对称性；② 引入自回归盲目性界限（Autoregressive Blindness Bound）；③ 将感知、控制、求解拆分成独立模块并引入外部干预信号；④ 提议以适应速度为指标的新基准。

**🔧 技术方法**

理论推导（群论、信息论、控制理论）、Mixture‑of‑Experts、FIATS 等因果干预模型与 LLM 感知层的组合。

**📊 数据集**

未做实验，仅引用公开基准如 M4、Monash、GIFT‑Eval、fev‑bench 用以说明当前评测方式。

**📈 对比分析**

无实验比较，作者指出现有 Zero‑Shot/Static 评测不足，主张采用适应性指标（如 TTR）来评估模型对结构性变动的恢复速度。

**⚠️ 局限性**

缺乏实证验证与可复现实验，主要依赖理论假设，且未给出完整的实现细节与调优策略。

---

## 76. Laplacian Representations for Decision-Time Planning

**arXiv ID:** 2602.05031 | [PDF](https://arxiv.org/pdf/2602.05031v1)

**作者:** Dikshant Shehmar `[一作]` (University of Alberta), Marlos C. Machado `[通讯]` (University of Alberta)

**通讯引用:** 1024 | [OpenAlex ID](https://openalex.org/A5085413987)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 ALPS 算法，通过在离线数据上学习 Laplacian 表征、前向模型和行为先验，利用 k‑means 聚类生成子目标，并在高层使用 Dijkstra 规划、低层使用带先验的 CEM 实现层次决策时规划。

**💡 创新点**

创新点在于将 Laplacian 表征用作高层抽象空间并结合行为先验驱动的 CEM，既解决了长时序误差，又通过谱聚类自然发现子目标，从而实现了基于离线数据的模型驱动层次规划。

**🔧 技术方法**

使用了 ALLO 目标学习 Laplacian 表征、单步前向模型的多步自回归训练、行为先验的行为克隆、k‑means 谱聚类、Dijkstra 高层规划以及带行为先验的 CEM 低层优化。

**📊 数据集**

在 Maze2D‑PointMass 以及 OGBench 的 pointmaze、antmaze、humanoidmaze（navigate、stitch、explore）离线数据集上进行评估。

**📈 对比分析**

与 PcLast、模型自由的 GCRL 基线（BC、ICQL、RIL 等）以及多种 ALPS 变体在相同环境下进行对比实验，ALPS 在大多数 OGBench 任务中显著优于模型自由基线（p<0.01），并在长时序任务中表现最佳。

**⚠️ 局限性**

存在对 teleport 迷宫对称性导致 Laplacian 对称化误导子目标选择的缺陷；在简单任务中行为先验可能限制性能；对离线数据质量高度敏感，且前向模型误差仍会影响高维连续空间的规划。

---

## 77. Does Programming Language Matter? An Empirical Study of Fuzzing Bug Detection

**arXiv ID:** 2602.05312 | [PDF](https://arxiv.org/pdf/2602.05312v1)

**作者:** Tatsuya Shirai `[一作]` (Nara Institute of Science and Technology), Hajimu Iida `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1507 | [OpenAlex ID](https://openalex.org/A5055973723)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本研究针对 OSS‑Fuzz 中不同编程语言的持续 fuzzing，进行跨语言大规模实验，分析了 bug 频率、类型、严重性、可复现性与检测效率等指标。

**💡 创新点**

首次系统比较多语言 fuzzing 的 bug 分布与检测效率，揭示语言设计对漏洞性质、复现率及检测时间的显著影响。

**🔧 技术方法**

使用统计方法（Kruskal‑Wallis、Mann‑Whitney U）、Bug 标签、CWE 分类以及 Patch Coverage 与 Time‑to‑Detection 等指标，对 OSS‑Fuzz 的 issue、build 日志和覆盖报告进行抽取与归类。

**📊 数据集**

采用 OSS‑Fuzz 开源项目中 6 种主语言（C、C++、Go、Java、Python、Rust）共计数千条 issue、build 日志和覆盖报告，并筛选出至少 10 条 bug 的项目作为数据集。

**📈 对比分析**

通过中位数比较、分布可视化和显著性检验，发现 C++/Rust 的 bug 检测率最高，Python 低且变异小；Rust 可复现率最高，Go 最差；Patch Coverage 与检测时间呈反向关系，语言差异显著。

**⚠️ 局限性**

局限性在于仅分析 OSS‑Fuzz 项目，样本受过滤条件限制；未充分考虑项目规模、成熟度等混杂因素；结果可能不易推广到非 OSS 或其他 fuzzing 工具环境。

---

## 78. Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities

**arXiv ID:** 2602.05281 | [PDF](https://arxiv.org/pdf/2602.05281v1)

**作者:** Pengyi Li `[一作]` (FusionBrain Lab), Ivan Oseledets `[通讯]` (Institute of Numerical Mathematics)

**通讯引用:** 10498 | [OpenAlex ID](https://openalex.org/A5004111307)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ProGRPO，基于 RLVR 的政策梯度方法，利用优势重加权与低概率 token 长度归一化来缓解熵坍塌，提升 LLM 推理与代码生成的多样性与稳定性。

**💡 创新点**

创新点在于将 prompt 与答案的置信度引入优势函数，实现自适应的置信度重加权；同时通过只归一化高不确定性 token 的长度，聚焦奖励于信息量大、易产生多样化推理路径的部分；整体构建了 ProGRPO 的新框架。

**🔧 技术方法**

采用 RLVR 的群组相对策略优化（GRPO）框架，结合优势重加权（ARM）、低概率 token 长度归一化、熵约束裁剪、可验证奖励以及 Pass@k 评价指标；技术实现包括概率估计、梯度裁剪与多样性评估。

**📊 数据集**

在数学推理任务使用 DAPO、AIME2024/25、AMC23、MATH500、Minerva、OlympiadBench 等数据集；代码生成任务使用 DeepCoder、LiveCodeBench、CodeForces、HumanEval+；OOD 评估使用 GPQA、MMLU-Pro。

**📈 对比分析**

与传统 GRPO 与 FlowRL 进行对比，ProGRPO 在 Qwen2.5‑7B 上 Pass@1 提升 5.7%，Pass@32 提升 13.9%；在多模型、多规模（Qwen2.5‑32B、DeepSeek‑1.5B/7B）与多领域（数学、代码）上均表现出稳定的性能提升，并在多样性与熵指标上优于基线。

**⚠️ 局限性**

局限性包括：需手动调节置信度重加权系数 α，过大或过小都会影响性能；方法仍依赖可验证奖励，难以直接迁移到无标注或不易验证的任务；在极端难度或高维奖励场景下，熵坍塌问题可能仍未完全消除。

---

## 79. Towards Advancing Research with Workflows: A perspective from the Workflows Community Summit -- Amsterdam, 2025

**arXiv ID:** 2602.05131 | [PDF](https://arxiv.org/pdf/2602.05131v1)

**作者:** Irene Bonati `[一作]` (SURF Cooperation), Rafael Ferreira da Silva `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 3133 | [OpenAlex ID](https://openalex.org/A5072339196)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本报告总结了2025年工作流社区峰会的主要发现，并提出了技术、政策和社区层面的行动方案。

**💡 创新点**

创新点在于从技术导向转向以科学影响为核心的测度框架，并提出统一工作流模式与基准标准。

**🔧 技术方法**

采用工作流模式规范、基准测试框架、社区协作平台以及跨领域的共创方法。

**📊 数据集**

未使用实验数据集，主要基于峰会讨论与案例回顾得出结论。

**📈 对比分析**

未进行直接性能比较，而是通过案例分析和专家共识提出评估指标，缺乏可量化的性能验证。

**⚠️ 局限性**

主要限制在于缺乏实证数据与可度量指标，无法验证提出措施的实际效果与可推广性。

---

## 80. PatchFlow: Leveraging a Flow-Based Model with Patch Features

**arXiv ID:** 2602.05238 | [PDF](https://arxiv.org/pdf/2602.05238v1)

**作者:** Boxiang Zhang `[一作]` (Purdue University), Corey Vian `[通讯]` (Stellantis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了PatchFlow框架，用于工业产品图像的无监督异常检测，结合邻域感知的补丁特征与归一化流模型，并通过适配器模块弥合预训练特征与工业图像的分布差异。

**💡 创新点**

创新点包括：1) 采用补丁级别的邻域感知特征聚合，使异常定位更细粒度；2) 引入轻量级特征适配器降低分布偏差；3) 设计瓶颈耦合结构的归一化流，显著降低计算量并保持表达能力；4) 在多尺度特征金字塔上直接映射补丁特征到标准正态分布，实现精准定位与高效推理。

**🔧 技术方法**

主要技术包括：预训练的EfficientNet‑B5特征提取器；多层级、多尺度特征聚合；全连接适配器；基于RealNVP的归一化流（含瓶颈耦合层）；使用KL散度训练流模型；基于最大异常分数的图像级AUROC评估与像素级AUPRC评估。

**📊 数据集**

实验使用了公开的MVTec AD和VisA两个大规模异常检测数据集，并在自采集的铸模缺陷数据集上验证，分别包含约5000张正样本和12000张样本（含异常），以及112张正常与36张合成缺陷的铸模图像。

**📈 对比分析**

与DRÆM、CutPaste、PADIM、PatchCore、CFlow‑AD、CS‑Flow等SOTA方法比较，PatchFlow在MVTec AD上图像级AUROC达到99.28%（比PatchCore提升约20%错误率），在VisA上达到96.48%（比PatchCore提升约28.2%错误率）。在铸模数据集上实现95.77%的准确率，无需使用异常样本训练。

**⚠️ 局限性**

局限性包括：1) 归一化流步骤越多性能反而下降，需权衡深度与速度；2) 计算量仍较大（约1.25×10¹⁰ FLOPs，2.7 FPS），对实时工业系统有一定负担；3) 依赖于预训练网络与适配器的泛化能力，面对极端不同的工业图像可能需要重新适配；4) 目前主要验证在金属铸件缺陷上，其他行业或更复杂缺陷类型的泛化尚待进一步研究。

---

## 81. Individual Fairness In Strategic Classification

**arXiv ID:** 2602.05084 | [PDF](https://arxiv.org/pdf/2602.05084v1)

**作者:** Zhiqun Zuo `[一作]` (Ohio State University), Mohammad Mahdi Khalili `[通讯]` (Ohio State University)

**通讯引用:** 922 | [OpenAlex ID](https://openalex.org/A5078967045)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在策略分类环境下的个体公平性问题，证明确定性阈值分类器必违背个体公平，提出通过随机阈值分布并利用线性规划求解最优分布，从而实现个体公平，并进一步加入群体公平约束。

**💡 创新点**

创新点在于首次将随机阈值分布与个体公平约束结合，给出可行性条件并用线性规划求最优解，同时兼顾群体公平，实验验证了该方法在保持准确率的前提下显著提升公平性。

**🔧 技术方法**

技术手段包括理论证明、随机阈值分布建模、约束条件推导、分箱近似、线性规划求解、以及对误差率与个体公平比率等指标的数值计算。

**📊 数据集**

实验使用了 FICO 信贷评分数据集和 Law School（bar 考试）学生成绩数据集，分别构造了信用分与 GPA 作为阈值指标。

**📈 对比分析**

方法通过与最优确定性阈值基线比较，利用 F1、IF 比率、统计差异（S‑DP）、等机会差异（EO‑DP）和等机会差异（ED‑DP）等指标评估；随机阈值在准确率几乎不变的情况下，将 IF 比率从 4.4 降至 0.1（FICO）或从 3.16 降至 0.17（Law School），并显著改善群体公平指标。

**⚠️ 局限性**

局限性包括：仅适用于可用阈值化的分类问题；要求存在 l(x) 并且成本仅依赖于 l(x) 的差值；样本量不足会导致估计误差；线性规划的求解复杂度随分箱数指数增长，限制了可使用的分箱数。

---

## 82. Decoupled Orthogonal Dynamics: Regularization for Deep Network Optimizers

**arXiv ID:** 2602.05136 | [PDF](https://arxiv.org/pdf/2602.05136v1)

**作者:** Hao Chen `[一作]` (Beijing University of Posts and Telecommunications), Hanmin Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的优化器AdamO，专门将参数的径向（模）控制与切向（特征学习）动态解耦，解决AdamW在径向与权重衰减之间的冲突；

**💡 创新点**

创新点在于：1）通过正交投影实现径向与切向子空间严格分离；2）仅在径向方向应用纯径向权重衰减；3）引入曲率自适应径向步长；4）根据网络结构进行维度感知和投影规则；

**🔧 技术方法**

技术包括：正交投影、独立的动量和二阶矩状态、径向学习率自适应、AdamP式切向投影、低维参数简化更新；

**📊 数据集**

使用CIFAR‑100图像分类数据集以及在附录中提到的语言任务（如Grokkng）进行实验；

**📈 对比分析**

与Adam、AdamW、AdamP等基线在相同训练预算、学习率调度下对比；AdamO在CIFAR‑100上达到79.74%准确率，比AdamW高4.99个百分点，显著优于其他改进方法；

**⚠️ 局限性**

局限性包括：目前仅在标准任务上验证，缺乏大规模模型或更复杂任务的评估；算法实现相对复杂，需额外的子空间投影和曲率估计；

---

## 83. Scalable Generation and Validation of Isomorphic Physics Problems with GenAI

**arXiv ID:** 2602.05114 | [PDF](https://arxiv.org/pdf/2602.05114v1)

**作者:** Naiming Liu `[一作]` (Rice University), Zhongzhou Chen `[通讯]` (University of Central Florida)

**通讯引用:** 1068 | [OpenAlex ID](https://openalex.org/A5101509377)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过生成式 AI 结合 prompt chaining 与工具调用，自动创建大规模同构物理题库，并使用大语言模型与学生成绩进行预部署验证。

**💡 创新点**

提出可精确控制结构性与表面变异的 prompt chain 与工具化生成框架，以及基于 LLM 的难度一致性与异质性检测机制。

**🔧 技术方法**

使用大语言模型（LLMs）进行题目生成与评估，并借助 Python 代码解释器实现结构性变异的计算与校验；核心技术为 prompt chaining 与 tool‑use。

**📊 数据集**

ESTELA‑Physics 数据集：共 666 题，覆盖 12 主题，包含 NUM、MCQ、MA、CAT 四种题型，已公开可用。

**📈 对比分析**

对 17 个开源 LLM 与 >200 名学生在三次期中考试中的表现进行对比，评估难度一致性（73% 通过 Fisher 检验）和 Pearson 相关（最高 0.594），验证 LLM 能够检测难度离散并与学生表现相关。

**⚠️ 局限性**

局限性：小规模模型推理不足，超大模型出现天花板效应；MCQ 题目难度受干扰项设计影响；当前框架对图形化题目支持有限。

---

## 84. Quantifying the Knowledge Proximity Between Academic and Industry Research: An Entity and Semantic Perspective

**arXiv ID:** 2602.05211 | [PDF](https://arxiv.org/pdf/2602.05211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 85. Position: Capability Control Should be a Separate Goal From Alignment

**arXiv ID:** 2602.05164 | [PDF](https://arxiv.org/pdf/2602.05164v1)

**作者:** Shoaib Ahmed Siddiqui `[一作]` (University of Cambridge), Adrian Weller `[通讯]` (University of Cambridge)

**通讯引用:** 5127 | [OpenAlex ID](https://openalex.org/A5042278493)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

本文阐述并倡导将能力控制与模型对齐分离，提出三层防御（数据、学习、系统）实现对基础模型的硬性能力限制。

**💡 创新点**

创新点在于将能力控制视为独立目标，提出三层防御框架并讨论各层缺陷与互补关系。

**🔧 技术方法**

使用的技术包括数据过滤/策划/生成、基于行为示例/偏好/无学习的学习控制、对抗训练、权重更新、知识编辑、表示工程、系统提示、输入/输出过滤、链式思维监测和信息流控制等。

**📊 数据集**

本文未给出具体数据集，而是在理论与方法论层面讨论，对各种通用预训练数据来源的控制策略。

**📈 对比分析**

由于缺乏实验和基准，本文未提供性能对比，主要以理论与现有文献为依据讨论各方法的效果与局限。

**⚠️ 局限性**

主要局限包括缺乏实证评估、难以完全去除能力、对抗性规避风险、开放权重模型安全难题、双用途知识冲突与组合泛化带来的重现风险。

---

## 86. EntRGi: Entropy Aware Reward Guidance for Diffusion Language Models

**arXiv ID:** 2602.05000 | [PDF](https://arxiv.org/pdf/2602.05000v1)

**作者:** Atula Tejaswi `[一作]` (University of Texas at Austin), Sujay Sanghavi `[通讯]` (University of Texas at Austin)

**通讯引用:** 5849 | [OpenAlex ID](https://openalex.org/A5110619770)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在离散扩散语言模型上进行奖励引导的策略，称为 EntRGi。

**💡 创新点**

创新点在于引入基于熵的自适应混合机制：根据模型在每一步的预测熵动态调节软/硬嵌入的比例，从而兼顾梯度精度与奖励模型对离散输入的可靠性。

**🔧 技术方法**

使用的技术包括离散扩散语言模型、奖励模型的梯度、softmax、straight‑through estimator（STE）、熵权重插值以及多步梯度优化。

**📊 数据集**

实验数据集包括 Dream‑v0‑Instruct‑7B 作为基础模型，Skywork Reward 系列（0.6B、1.7B、4B）作为奖励模型，评测基准为 Reward‑Bench‑2、RM‑Bench、JudgeBench。

**📈 对比分析**

与 BoN、Expectation（连续松弛）、APS 等基线对比，EntRGi 在所有基准任务的 Top@1、Avg@N 以及 LMUnit 分数均优于 APS，且在更高采样温度（τ=0.7）下表现更佳，显示出更强的奖励导向效果。

**⚠️ 局限性**

局限包括：要求扩散模型与奖励模型使用相同词表；仍存在奖励劫持风险；需要调节梯度步数以避免过度优化；对词表不匹配或不同奖励目标的适用性尚未验证。

---

## 87. Learning Where It Matters: Geometric Anchoring for Robust Preference Alignment

**arXiv ID:** 2602.04909 | [PDF](https://arxiv.org/pdf/2602.04909v1)

**作者:** Youngjae Cho `[一作]` (PYLER), Ji-Hoon Kim `[通讯]` (PYLER)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于几何锚点的偏好优化方法GAPO，利用局部对抗扰动评估偏好对局部扰动的鲁棒性并自适应加权。

**💡 创新点**

创新点是用动态几何锚点取代固定参考，构造Anchor Gap作为实例级鲁棒性信号，实现对脆弱偏好的自动下权。

**🔧 技术方法**

采用对数损失、对抗性参数扰动、梯度重加权、SAM对比、Hessian谱分析等技术。

**📊 数据集**

使用UltraFeedback、Anthropic HH、Pythia、Mistral-Instruct、Llama‑3、Gemma等公开对话与推理数据集。

**📈 对比分析**

与DPO、SimPO、α‑DPO、R‑DPO等基线对比，GAPO在指令跟随、推理能力及噪声鲁棒性方面均实现显著或相当的提升。

**⚠️ 局限性**

缺点是训练时间约为SimPO的两倍，且仍无法完全消除偏见与安全风险。

---

## 88. CAST-CKT: Chaos-Aware Spatio-Temporal and Cross-City Knowledge Transfer for Traffic Flow Prediction

**arXiv ID:** 2602.05133 | [PDF](https://arxiv.org/pdf/2602.05133v1)

**作者:** Abdul Joseph Fofanah `[一作]` (Griffith University), Zhongyi Zhang `[通讯]` (Griffith University)

**通讯引用:** 312 | [OpenAlex ID](https://openalex.org/A5100442966)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 CAST-CKT，一种融合混沌理论的跨城市少量数据交通预测框架，能够在数据稀缺、跨城市情境下进行自适应时空建模并给出不确定性估计。

**💡 创新点**

核心创新包括：① 用多维混沌特征（Lyapunov指数、熵等）构建交通动态“混沌谱”，① 以此特征调控注意力与图结构的自适应学习，② 通过混沌一致性实现跨城市知识对齐，③ 结合多尺度时序编码与不确定性量化实现长短程多步预测。

**🔧 技术方法**

采用混沌特征提取、混沌条件注意力、可学习的自适应图结构、并行多尺度 LSTM 编码、基于 Gaussian NLL 的不确定性估计，并在 meta‑learning 框架下训练以实现快速适应。

**📊 数据集**

在四个真实交通数据集上评估：METR‑LA、PEMS‑Bay、深圳、成都；每个目标城市仅使用 3 天（约 5–10%）标签进行微调，其余城市做元训练。

**📈 对比分析**

与 13 种基线（传统 ST‑GNN、跨域迁移、prompt‑based 等）对比，CAST‑CKT 在所有数据集、所有时延（5–60 分钟）均取得最高或次高 MAE、RMSE，平均提升 20–35%（长时延提升更显著），并且给出校准良好的置信区间。

**⚠️ 局限性**

主要局限包括：① 计算混沌特征和自适应图学习耗时，难以满足极低时延的实时系统需求；② 对于极度稀缺或噪声严重的数据，混沌谱提取可能不稳定；③ 目前仅在四个城市验证，跨更大规模多样化城市的泛化尚未充分探测。

---

## 89. RFM-Pose:Reinforcement-Guided Flow Matching for Fast Category-Level 6D Pose Estimation

**arXiv ID:** 2602.05257 | [PDF](https://arxiv.org/pdf/2602.05257v1)

**作者:** Diya He `[一作]` (University of Science and Technology of China), Jiahu Qin `[通讯]` (University of Science and Technology of China)

**通讯引用:** 9801 | [OpenAlex ID](https://openalex.org/A5053071794)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种融合流匹配与强化学习的RFM-Pose框架，用于类别级6D姿态估计；

**💡 创新点**

将流匹配采样过程视为马尔可夫决策过程，使用PPO微调采样策略，并用多重价值网络对候选姿态进行评分，实现生成与评估一体化；

**🔧 技术方法**

流匹配（flow matching）、Proximal Policy Optimization (PPO)、多重价值网络、SO(3)四元数平均、QUEST；

**📊 数据集**

REAL275、CAMERA、Omni6DPose三大RGB‑D基准数据集；

**📈 对比分析**

与多种确定性与概率性方法（如GenPose、NOCS、DualPoseNet等）对比，RFM‑Pose在5°2cm/5°5cm/10°2cm/10°5cm四个指标上均取得更高mAP，同时采样步数仅为H=20，速度提升30FPS以上，显著降低计算成本；

**⚠️ 局限性**

依赖单模RGB‑D输入，无法充分利用RGB信息；对极端遮挡和极高对称性物体仍存在误差；未来需探索RGB特征融合与更大类别的泛化。

---

## 90. Data-Centric Interpretability for LLM-based Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.05183 | [PDF](https://arxiv.org/pdf/2602.05183v1)

**作者:** John Yan `[一作]` (Gutenberg AI), Matthew Lyle Olson `[通讯]` (Oracle)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个双通道解释框架，利用稀疏自编码器（SAE）与大型语言模型（LLM）摘要技术，对在Full‑Press Diplomacy多智能体强化学习训练过程中的行为进行细粒度与宏观层面的解释，并通过用户研究与下游任务验证所生成的解释假设的可解释性和实用性。

**💡 创新点**

创新点主要包括：① 设计Meta‑Autointerp方法，将大量单一SAE特征聚合成可解释的训练动态假设；② 将SAE与LLM摘要视角结合，揭示不同层次的行为模式；③ 通过用户实验评估解释假设的“有用性”与“可解释性”，首次量化解释对人类决策的实际帮助；④ 在RL训练中发现并验证奖励“投机”行为以及提前识别失败训练的能力。

**🔧 技术方法**

采用技术包括：稀疏自编码器（Gemma Scope 2）、LLM摘要（Gemini 2.5 Flash 与 Claude Opus 4.5）、Group Relative Policy Optimization（GRPO）强化学习、自动解释（Autointerp）以及相关统计/机器学习评估方法（Spearman/Isotonic相关、McNemar检验、线性探针、AUC）。

**📊 数据集**

数据集为Full‑Press Diplomacy环境下的训练轨迹，总计6400条，其中包含一次成功、一次失败的训练跑；从中抽取900条“canonical”轨迹做特征提取；轨迹包含完整游戏记录、工具调用、代理思考与回复等。

**📈 对比分析**

比较方法：通过三组假设（LLM摘要、单一SAE特征、SAE Meta‑Features）分别评估人类评分（可解释性/有用性）、自动预测准确率（LLM评审）以及下游任务（游戏得分）。结果显示：① SAE Meta‑Features在人类评分中达到0.85/0.83，自动预测显著率90%；② 单一SAE特征显著率45%；③ LLM摘要显著率21%；④ 在游戏对照实验中，使用Meta‑Features优化的提示提升平均得分14.2%。

**⚠️ 局限性**

局限性：仅在单一多智能体环境与有限训练跑上验证，缺乏因果分析与大规模交叉验证；解释特征的可执行性测试为一次性实验，未逐一消融；SAE特征稀疏导致对短文本的判别困难；LLM摘要受上下文长度限制，部分宏观行为难以捕捉。

---

## 91. Imposing Boundary Conditions on Neural Operators via Learned Function Extensions

**arXiv ID:** 2602.04923 | [PDF](https://arxiv.org/pdf/2602.04923v1)

**作者:** Sepehr Mousavi `[一作]` (ETH Zurich), Laura De Lorenzis `[通讯]` (ETH Zurich)

**通讯引用:** 14268 | [OpenAlex ID](https://openalex.org/A5086471469)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可插拔的边界条件扩展模块，通过函数扩展将复杂多变的边界信息映射到整个域，实现神经算子对任意边界条件的高效处理。

**💡 创新点**

将Dirichlet、Neumann、Robin等不同类型的边界条件统一归一化，并设计基于交叉注意力的可学习伪扩展器，将边界信息编码为域函数，显著提升对复杂边界的适应能力；同时在18个高难度基准数据集上验证其效果。

**🔧 技术方法**

在现有神经算子（如DeepONet、FNO、RIGNO、GAOT）框架下加入零扩展、谐波扩展和可学习伪扩展器，采用交叉注意力与feed‑forward块构建扩展模块，并使用encode‑process‑decode体系进行端到端训练。

**📊 数据集**

构造18个包含Poisson、线性弹性和非线性弹性问题的数据集，涵盖多边界段、随机尺寸/位置、混合边界类型、非凸几何等复杂设置。

**📈 对比分析**

与BENO及传统无边界条件网络进行对比，采用相同超参数，结果显示学习扩展器在所有数据集上误差下降30%以上；在复杂弹性问题中误差从约30%降至11%；与FEM对比，推理速度提升约四万倍。

**⚠️ 局限性**

仅在二维问题验证，三维推广仍需更多数据；扩展器对边界节点数量相对有限，可能需进一步优化；在极端噪声或完全不同物理系统的泛化仍存在一定局限。

---

## 92. A Short and Unified Convergence Analysis of the SAG, SAGA, and IAG Algorithms

**arXiv ID:** 2602.05304 | [PDF](https://arxiv.org/pdf/2602.05304v1)

**作者:** Feng Zhu `[一作]` (North Carolina State University), Aritra Mitra `[通讯]` (North Carolina State University)

**通讯引用:** 1391 | [OpenAlex ID](https://openalex.org/A5101651422)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

为有限和差分优化问题提出一种统一的收敛性分析框架，涵盖SAG、SAGA及其确定性变体IAG，并给出其线性收敛率。

**💡 创新点**

创新点在于：① 用单一简洁的证明步骤同时处理偏置的SAG与无偏的SAGA；② 通过高概率分析首次给出SAG、SAGA的收敛界；③ 证明该框架可直接扩展到Markov采样和非凸目标；④ 以此方法显著提升IAG的已知收敛速率。

**🔧 技术方法**

主要技术包括：利用伯努利/ Bernstein 归一化定理对采样延迟做概率上界；构造带有延迟窗口的Lyapunov函数；在该Lyapunov框架下完成单步递推；结合强凸性与光滑性得到指数收敛；对Markov链使用混合时间与最小驻留概率的 Bernstein 版本。

**📊 数据集**

无具体实验数据集，本文主要为理论分析。

**📈 对比分析**

与传统的期望收敛分析相比，新界在高概率下给出了更强的保证；相比IAG先前的指数速率（指数系数~1/κ²N²），本文得到指数系数~1/κN，显著加快；在Markov采样下的收敛速率与IID情形相同，只是延迟参数依赖混合时间和最小驻留概率。

**⚠️ 局限性**

局限性：① 高概率收敛率与先前期望收敛率仍存在阶数差距，尚不清楚是否可进一步压缩；② 对Markov采样的下界尚未给出；③ 论文未给出实验验证，理论界的实际常数与经验表现仍需进一步评估。

---

## 93. Does SGD Seek Flatness or Sharpness? An Exactly Solvable Model

**arXiv ID:** 2602.05065 | [PDF](https://arxiv.org/pdf/2602.05065v1)

**作者:** Yizhou Xu `[一作]` (École Polytechnique Fédérale de Lausanne), Liu Ziyin `[通讯]` (NTT Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在可解析的深度线性网络模型中，研究SGD训练过程的平坦度/尖锐度偏好，给出在最小梯度波动约束下的解析解，证明SGD最终收敛到唯一的尖锐度值。

**💡 创新点**

创新点包括：①证明SGD的隐式偏好是最小梯度波动而非平坦度；②指出标签噪声的各向异性决定收敛尖锐度，并给出对应的闭式公式；③提出“最小波动准则”，用以解释尖锐度悖论并统一SGD在不同情形下的平坦/尖锐行为。

**🔧 技术方法**

使用的技术主要有：可解析深度线性网络建模、最小波动（熵）损失框架、SVD和矩阵重标定对称性分析、理论推导与闭式结果、数值实验（MLP、RNN、Transformer 等）来验证理论。

**📊 数据集**

实验数据集包括：合成线性教师生成的数据、MNIST 数据集以及通过人工设置的不同噪声谱（各向异性与等向性）构造的标签噪声场景。

**📈 对比分析**

方法比较：将SGD与全批梯度下降（GD）对比，观察尖锐度随训练进展的变化；将实验曲线与理论预测（闭式尖锐度）对齐；在非线性网络实验中，验证噪声不均导致尖锐度升高的趋势，均与理论一致，显示理论具有较高解释性。

**⚠️ 局限性**

limitations：仅在小学习率、最小波动的先导项下成立，未揭示训练动态过程；未解释大学习率和 EoS 行为；理论对高阶项敏感性未知；实验验证范围有限，未覆盖所有实际网络架构。

---

## 94. Autodiscover: A reinforcement learning recommendation system for the cold-start imbalance challenge in active learning, powered by graph-aware thompson sampling

**arXiv ID:** 2602.05087 | [PDF](https://arxiv.org/pdf/2602.05087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 95. BioACE: An Automated Framework for Biomedical Answer and Citation Evaluations

**arXiv ID:** 2602.04982 | [PDF](https://arxiv.org/pdf/2602.04982v1)

**作者:** Deepak Gupta `[一作]` (National Library of Medicine), Dina Demner-Fuhsman `[通讯]` (National Library of Medicine)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个自动化评估框架BioACE，用于评估生物医学问题回答和引用的质量。

**💡 创新点**

创新点在于将答案拆分为nuggets并用嵌入、NLI和LLM三种方式评估完整性、正确性和引用支持，结合多维度指标并与人工标注高度相关。

**🔧 技术方法**

采用句子嵌入、贝叶斯高斯混合模型、预训练语言模型（BERT、RoBERTa）、大语言模型（Llama、Qwen、Mistral）以及LoRA微调等技术。

**📊 数据集**

使用公开的BioGen问答数据集（65题）以及相应的引用标注。

**📈 对比分析**

与人工评估和现有基线方法对比，BioACE在nugget recall、答案完整性和引用准确率上与人工评分相关性高，部分指标的F1可达76%以上。

**⚠️ 局限性**

局限性包括小样本导致微调效果不佳、引用评估仍显低效且对抗性干扰，且在多模型间性能差异仍存在。

---

## 96. CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning

**arXiv ID:** 2602.04920 | [PDF](https://arxiv.org/pdf/2602.04920v1)

**作者:** Ronghao Lin `[一作]` (Sun Yat-Sen University), Haifeng Hu `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 7456 | [OpenAlex ID](https://openalex.org/A5056953478)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个信息瓶颈空间，通过令牌级与标签级信息瓶颈以及循环交互翻译，实现在同一模型中同时支持完整与不完整多模态学习。

**💡 创新点**

创新点在于将循环信息瓶颈与跨模态残差自编码器相结合，既能清洗特征、压缩信息，又能在缺失模态下重建信息，从而实现单模型覆盖完整与缺失两种场景。

**🔧 技术方法**

使用信息瓶颈（Token/Label IB）与变分近似、跨模态残差自编码器（CRA）实现循环翻译、Transformer融合以及对任务目标的监督。

**📊 数据集**

在四个公开多模态情感/情绪数据集上评估：MOSI、MOSEI、IEMOCAP、MELD。

**📈 对比分析**

与多种基线（CCA、DCCA、CRA、MCTN、GCNet、IMDer等）比较，CyIN 在完整模态和多种缺失模态配置下均达到或超过现有最先进方法的性能。

**⚠️ 局限性**

局限性包括：1）对各模态赋予相同权重，未考虑模态重要性差异；2）翻译模块仍可用更先进的生成模型（如扩散模型）替代；3）未在更大规模多模态或与大型语言模型的联合训练上进一步验证泛化。

---

## 97. A novel scalable high performance diffusion solver for multiscale cell simulations

**arXiv ID:** 2602.05017 | [PDF](https://arxiv.org/pdf/2602.05017v1)

**作者:** Jose-Luis Estragues-Muñoz `[一作]` (Barcelona Supercomputing Center), Alfonso Valencia `[通讯]` (ICREA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种高性能、可扩展的BioFVM-B库，用于在超算环境下高效求解细胞多尺度仿真中的分子扩散-衰减问题，支持从毫米级到厘米级（甚至全肝）微环境模拟。

**💡 创新点**

创新点包括：
- 采用4维连续内存结构替代嵌套向量，显著降低内存占用并提升访问局部性；
- 结合MPI-OpenMP混合并行、SIMD指令集（AVX256D）以及块级通信策略，实现对TDMA求解的高并行化；
- 设计了可调节块数的启发式方法，平衡MPI通信与计算重叠；
- 在不改动分子扩散求解器核心算法的前提下，实现了显著的加速与可扩展性。

**🔧 技术方法**

主要技术：有限体积方法（FVM）→分子扩散-衰减 →多维TDMA求解；MPI分布式并行、OpenMP线程并行、SIMD向量化；内存优化与通信重叠；性能分析工具（Extrae、Paraver）。

**📊 数据集**

使用仿真数据集：以人类肝脏为参照，构建多种尺寸的立方微环境（4%~100%肝脏，尺寸从5 mm到120 mm，substrates数量从1到8），用于评估内存占用、单步耗时与可扩展性。

**📈 对比分析**

与原生BioFVM（共享内存）和BioFVM-X（MPI+OpenMP）对比。BioFVM-B在单核/单节点上实现约34.8×速度提升；在分布式多节点上实现最高约196.8×速度提升；内存占用比BioFVM-X低36%，比BioFVM低26.5%；在全肝模拟（100%肝）下，单步耗时从32秒降至约1秒级（取决于节点数）。

**⚠️ 局限性**

局限性：
- 仍需依赖大规模HPC资源，对极大尺寸或多substrate场景内存仍然昂贵；
- 块数NB的最佳选择依赖经验启发式，需进一步自动化调优；
- 目前仅支持CPU（AVX），尚未实现GPU/FPGA/向量扩展器的加速；
- 扩展到完整PhysiCell等仿真框架时，还需进一步验证与集成。

---

## 98. ProAct: Agentic Lookahead in Interactive Environments

**arXiv ID:** 2602.05327 | [PDF](https://arxiv.org/pdf/2602.05327v1)

**作者:** Yangbin Yu `[一作]` (Tencent Hunyuan), Jie Jiang `[通讯]` (Tencent Hunyuan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 ProAct 框架，通过先在环境中使用 MCTS 生成的搜索树压缩为简洁的推理链来监督 LLM，随后利用 Monte‑Carlo Critic 在 RL 训练中提供低方差价值估计，以提升大语言模型在长周期交互环境中的前瞻性决策能力。

**💡 创新点**

创新点在于（1）Grounded Lookahead Distillation（GLAD）将真实环境的搜索结果压缩为自然语言推理链，克服 LLM 在内部仿真中累积误差的难题；（2）Monte‑Carlo Critic（MC‑Critic）通过随机策略轻量化 roll‑out 估计价值，显著降低价值估计方差并稳定多回合 RL 更新。

**🔧 技术方法**

主要技术包括 MCTS、监督微调（SFT）与推理链压缩、基于 Monte‑Carlo roll‑out 的价值估计、以及在 PPO/GRPO 等 policy‑gradient 算法中集成 MC‑Critic 的插件式改造。

**📊 数据集**

实验使用了两类长周期游戏：随机化的 2048（grid 4×4/3×3/3072）和确定性的 Sokoban（标准、未见、动作空间改造、符号改造等变体），并在这些环境上收集了搜索生成的样本进行 GLAD 训练。

**📈 对比分析**

与多种开源与闭源基线（如 GPT‑5、Claude‑4.5‑Sonnet、Qwen3‑30B、Qwen3‑4B‑Instruct 等）对比，ProAct 在 2048 与 Sokoban 上均取得显著提升，4B 模型在 GLAD+MC‑Critic 组合下接近甚至超越部分闭源模型，且在环境变体上表现出良好的泛化。

**⚠️ 局限性**

局限性包括：① 仍需依赖昂贵的 MCTS 进行监督数据生成；② MC‑Critic 的随机 roll‑out 可能在奖励稀疏或极端长序列环境中失效；③ 目前验证仅局限于棋类/益智类任务，缺乏在更复杂、现实世界场景中的评估。

---

## 99. TestMigrationsInPy: A Dataset of Test Migrations from Unittest to Pytest

**arXiv ID:** 2602.05122 | [PDF](https://arxiv.org/pdf/2602.05122v1)

**作者:** Altino Alves `[一作]` (Federal University of Minas Gerais), Andre Hora `[通讯]` (Federal University of Minas Gerais)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

收集并整理了从unittest迁移到pytest的真实迁移案例，构建了包含923个迁移实例的TestMigrationsInPy数据集。

**💡 创新点**

首次提供专门针对Python测试框架迁移的公开数据集，并对迁移类型（断言迁移、fixture迁移等）进行标注，支持后续迁移工具的验证与评估。

**🔧 技术方法**

利用自动迁移检测工具（结合PyDriller）识别迁移提交，并手工筛选孤立迁移；通过GitHub仓库结构化存储迁移代码与摘要。

**📊 数据集**

基于100个流行Python项目的测试套件，从中筛选出690个迁移提交，最终提炼出923个孤立迁移实例，构成该数据集。

**📈 对比分析**

将LLM（如GPT‑4o）生成的迁移结果与数据集中的真实迁移对比，发现GPT‑4o能够加速迁移过程，但在fixture迁移上仍需人工修正，整体迁移成功率较高但存在细节错误。

**⚠️ 局限性**

数据集仅覆盖主流项目，未包含小众项目；迁移类型分类粗略，缺乏对断言/fixture细节的细粒度标注；未来需扩展项目覆盖范围和细化迁移标签。

---

## 100. Hybrid Gated Flow (HGF): Stabilizing 1.58-bit LLMs via Selective Low-Rank Correction

**arXiv ID:** 2602.05269 | [PDF](https://arxiv.org/pdf/2602.05269v1)

**作者:** David Alejandro Trejo Pizzo `[一作]` `[通讯]` (OpenCoresAI), David Alejandro Trejo Pizzo (OpenCoresAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Hybrid Gated Flow（HGF）双流架构，将1.58位三值量化权重与可学习的低秩FP16校正通道结合，利用门控实现结构与语义的协同；

**💡 创新点**

创新点在于把极端量化与低秩校正融合为双路网络，并通过门控实现量化稳定化、差分注意力的结构正则；

**🔧 技术方法**

采用1.58-bit三值量化、Straight-Through Estimator、低秩LoRA校正、门控激活与差分注意力；

**📊 数据集**

主要在TinyStories数据集上验证，并在SlimPajama、FineWeb-Edu上做初步规模扩展；

**📈 对比分析**

与FP16基线、BitNet b1.58、全精度差分注意力等做对比，HGF 1.0在验证集上loss为0.9306，比BitNet 1.0294低约9.6%，恢复约55%质量差距，且在2500步即可收敛；

**⚠️ 局限性**

局限包括：质量仍低于FP16基线，硬件对三值运算支持不足，规模扩展尚未完全验证，且在高阶任务中可能仍有性能瓶颈。

---

## 101. Metacognitive Demands and Strategies While Using Off-The-Shelf AI Conversational Agents for Health Information

**arXiv ID:** 2602.05111 | [PDF](https://arxiv.org/pdf/2602.05111v1)

**作者:** Shri Harini Ramesh `[一作]` (University of Calgary), Fateme Rajabiyazdi `[通讯]` (University of Calgary)

**通讯引用:** 392 | [OpenAlex ID](https://openalex.org/A5046158009)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过让15名参与者使用基于ChatGPT‑4o的对话式AI在模拟健康情境下进行健康信息检索，并进行思考录音，探讨用户在此过程中面临的认知监控需求及其对策

**💡 创新点**

首次系统性阐释了健康信息检索中使用通用对话式AI所产生的元认知负担，并基于实证结果提出五条面向健康信息检索的界面设计原则

**🔧 技术方法**

采用ChatGPT‑4o API构建自定义聊天界面，配合思考录音、NVivo编码与主题分析技术

**📊 数据集**

数据来源为15名在多元化样本设计下的健康情境模拟任务，未使用公开数据集

**📈 对比分析**

本研究为定性研究，无对照实验或数值性能指标，主要通过主题归纳评估需求与对策

**⚠️ 局限性**

样本规模有限、情境为模拟且仅单次会话，且使用的模型为GPT‑4o，可能导致发现不适用于更真实或更长时段的使用场景

---

## 102. High-Performance Moment-Encoded Lattice Boltzmann Method with Stability-Guided Quantization

**arXiv ID:** 2602.05295 | [PDF](https://arxiv.org/pdf/2602.05295v1)

**作者:** Yixin Chen `[一作]` (University of Toronto), Kui Wu `[通讯]` (Lightspeed)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种面向 GPU 的高效、低内存占用的基于时刻编码的 Lattice Boltzmann 方法（HOME‑LBM）实现，并通过拆分核、Von Neumann 稳定性分析以及基于稳定性的 16 位量化方案实现实时大规模流体-固体耦合仿真。

**💡 创新点**

创新点在于：① 将流体更新与固体边界处理拆分成两个独立的 GPU 核，显著减少 warp divergence；② 对 HOME‑LBM 进行首创的 Von Neumann 稳定性分析，得到各时刻分量的稳定边界；③ 基于该分析设计可直接保证数值稳定性的 16 位量化策略，兼顾高性能与数值安全；④ 采用混合精度累积减少原子操作误差。

**🔧 技术方法**

使用的技术包括：GPU 结构化数据布局（SoA + 8×8×8 tile）、共享内存流动 + 碰撞融合、BVH/体素化加速的三角网边界插值、Hermite 三阶分布重构、Von Neumann 频域稳定性分析、均匀 16 位量化与空间抖动、混合精度原子累加。

**📊 数据集**

实验数据集涵盖多种大型流体-固体场景：
- 1024³、1000×400×400 细节丰富的 Ducati 摩托车与 F1 车身烟雾；
- 660×250×330 的 Delta‑Wing；
- 256×512×256 的 Hilbert 结构、Porous 细孔等；
- 256×128×128 的多层涡旋、双层涡流；
- 256×512×256 等多分辨率测试。

**📈 对比分析**

与 HOME‑LBM、BGK、Leapfrog Flow‑Map 等基线相比，
- 纯流体模拟时内存压缩 50% 并实现高达 6× 的速度提升；
- 复杂固体耦合时内存压缩 25%，速度提升 4.1×（split‑kernel）+ 额外 16 位量化提升 0.8×；
- 与 Leapfrog Flow‑Map 在相同分辨率下 5.2× 的速度提升，同时保持相似或更好的细节；
- 在 1000×400×400 场景中实现约 1 秒/帧的实时渲染。

**⚠️ 局限性**

局限性包括：仅支持单层表面网格，无法处理薄壳或层叠几何；未对子网格几何精度与边界误差进行系统评估；仅支持静态固体，未实现双向耦合或运动固体；量化方案目前针对弱压缩流体，可能在高度可压缩场景下失效。

---

## 103. SynthForensics: A Multi-Generator Benchmark for Detecting Synthetic Video Deepfakes

**arXiv ID:** 2602.04939 | [PDF](https://arxiv.org/pdf/2602.04939v1)

**作者:** Roberto Leotta `[一作]` (iCTLab s.r.l.), Sebastiano Battiato `[通讯]` (University of Catania)

**通讯引用:** 7055 | [OpenAlex ID](https://openalex.org/A5042746008)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个针对纯合成视频的人类中心基准数据集 SynthForensics，包含 6,815 条来自五大开源 T2V 模型的高质量合成视频，并采用 paired‑source 协议与双阶段人工验证。

**💡 创新点**

创新点在于：① 采用 paired‑source 协议使合成视频与真实视频在语义上对应，减少内容共线性；② 双阶段人工验证确保语义一致性与伦理合规；③ 提供四种压缩版本，模拟真实世界的鲁棒性；④ 评估现有检测器的落后，并证明该数据集能显著提升模型泛化。

**🔧 技术方法**

技术手段包括：基于 Vision‑Language 模型生成结构化提示、对五个 T2V 模型（Wan2.1、CogVideoX、SkyReels‑V2、Self‑Forcing、MAGI‑1）进行统一配置、手工校验与自动化敏感内容过滤、视频合成与多版本压缩、VBench 等自动评估、以及多种检测器的零射击、微调与从零训练实验。

**📊 数据集**

使用的数据集为 FF++ 与 DFD 的 1,363 条原始视频作为源，生成对应的 T2V 视频；并对比现有的 FaceForensics++、Celeb‑DF、DFDC 等传统 manipulation 基准。

**📈 对比分析**

通过零射击实验，现有检测器在 SynthForensics 上平均 AUC 下降约 29.2 点，部分模型甚至低于随机；微调后可提升至近 100%；从零训练并在三大生成器上训练，能在未见生成器上保持 93.8% AUC，但在旧的 manipulation 数据集上性能骤降。

**⚠️ 局限性**

局限性包括：① 语义多样性受限于 FF++/DFD 来源，缺乏更广泛的种族与内容覆盖；② 只关注 T2V，未覆盖图像‑转‑视频等新型合成方式；③ 微调后对传统 manipulation 的后向兼容性差，表明两类攻击特征不兼容；④ 数据集生成耗时较大，扩展难度高。

---

## 104. Extreme Weather Nowcasting via Local Precipitation Pattern Prediction

**arXiv ID:** 2602.05204 | [PDF](https://arxiv.org/pdf/2602.05204v1)

**作者:** Changhoon Song `[一作]` (Seoul National University), Youngjoon Hong `[通讯]` (Seoul National University)

**通讯引用:** 2826 | [OpenAlex ID](https://openalex.org/A5085177012)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种名为exPreCast的确定性雷达降雨现在预报框架，可细致捕捉极端降雨细节。

**💡 创新点**

创新点在于结合局部时空注意力、立方体双重上采样(CDU)解码器以及可调时序提取器(TE)，实现低成本且对极端事件同样精确的预报。

**🔧 技术方法**

采用Video Swin Transformer为骨干，加入CDU、TE模块，并使用三维像素重排与三线性插值混合的上采样技术。

**📊 数据集**

使用了KMA平衡雷达数据集，并在SEVIR与MeteoNet等传统极端/常规降雨数据集上进行验证。

**📈 对比分析**

与现有基线（如Earthformer、SimVP、CasCast等）对比，exPreCast在CSI、HSS等指标上取得最优或相近性能，同时参数量与GFLOPs显著下降。

**⚠️ 局限性**

局限性在于缺乏不确定性估计、对长时延极端事件预测仍受数据稀缺和局部观测限制。

---

## 105. Reinforcement Learning Enhancement Using Vector Semantic Representation and Symbolic Reasoning for Human-Centered Autonomous Emergency Braking

**arXiv ID:** 2602.05079 | [PDF](https://arxiv.org/pdf/2602.05079v1)

**作者:** Vinal Asodia `[一作]` (University of Surrey), Saber Fallah `[通讯]` (University of Surrey)

**通讯引用:** 4268 | [OpenAlex ID](https://openalex.org/A5071227510)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于超维向量的语义空间表示（VSR）与软一阶逻辑（SFOL）奖励函数相结合的纵向控制管线，用于人本自适应紧急制动。

**💡 创新点**

创新点在于：1）扩展VSR以编码静态与动态实体，并通过权重强调易受伤道路使用者；2）将SSI超向量与空间特征嵌入以逐元素相加方式融合；3）设计SFOL奖励函数，将安全、效率与舒适三值通过符号推理动态平衡。

**🔧 技术方法**

采用的技术包括：超维计算（HRR）、语义分割（UNet+ResNet18）、特征自编码器、PPO强化学习、PyReason符号推理、CARLA仿真。

**📊 数据集**

实验数据来源于CARLA仿真平台中的遮挡行人穿越场景，设置低/中/高交通密度与部分/完整遮挡两种遮挡级别。

**📈 对比分析**

通过对比四种VSR融合方式和五种SFOL奖励变体，在碰撞率、成功率、停距、驾驶特征等指标评估。最佳组合为SSI⊕SF与完整规则集，成功率最高、碰撞率最低，且保持良好安全与效率平衡；其他组合在高密度或完整遮挡下表现欠佳。

**⚠️ 局限性**

局限性包括：未实现100%成功率；在高拥堵时符号推理仍出现波动；仅处理纵向控制，缺乏时序符号信息；实验仅限于仿真环境，需进一步验证在真实交通中的鲁棒性。

---

## 106. Data Kernel Perspective Space Performance Guarantees for Synthetic Data from Transformer Models

**arXiv ID:** 2602.05106 | [PDF](https://arxiv.org/pdf/2602.05106v1)

**作者:** Michael Browder `[一作]` (University of Maryland), Peter Viechnicki `[通讯]` (Johns Hopkins University)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5048546372)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Data Kernel Perspective Space (DKPS) 框架，用于对 Transformer（包括 LLM）生成的合成数据进行统计特性分析，并给出偏差、方差和模型间距离的数学保证。

**💡 创新点**

创新点在于：① 将黑盒模型输出映射到欧氏空间并利用多维尺度缩放 (MDS) 构造可解释的低维表示；② 在此框架下给出对合成数据质量的统计保障；③ 将 DKPS 同时应用于传统最大似然 (MLE) 与对比偏好优化 (CPO) 训练，揭示批量与顺序生成数据的差异。

**🔧 技术方法**

技术手段包括：多维尺度缩放 (MDS)、主成分分析 (PCA)、LASER3 句子嵌入、最大似然估计、对比偏好优化 (CPO)、高斯混合模型、欧氏/马氏距离等。

**📊 数据集**

实验数据集：英-祖鲁平行语料 1000 句子（训练集）与 999 句子（验证集），使用 Sockeye NMT 模型生成 10 条批量与 10 条顺序翻译。

**📈 对比分析**

比较方法：用 DKPS 计算不同模型（人类翻译、顺序翻译、批量翻译）的距离矩阵并进行 MDS 可视化；评估偏差与方差在 MLE 与 CPO 两种框架下的变化。结果显示：批量翻译的方差更大、偏差分布不同；CPO 能抑制顺序翻译的方差但仍显著增加批量翻译的方差；不同维度揭示了偏差与方差的结构。

**⚠️ 局限性**

限制：① 仅在机器翻译任务上验证，未覆盖其他 NLP 场景；② DKPS 依赖查询集与嵌入方式，可能对选择不敏感；③ 样本量有限，维度选择有主观成分；④ 对非线性关系和高维特征的处理不足；⑤ 未直接量化合成数据对下游模型性能的提升。

---

## 107. Beyond Cosine Similarity

**arXiv ID:** 2602.05266 | [PDF](https://arxiv.org/pdf/2602.05266v1)

**作者:** Xinbo Ai `[一作]` (Beijing University of Posts and Telecommunications), Xinbo Ai `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 344 | [OpenAlex ID](https://openalex.org/A5022897645)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出基于重排不等式的相似度度量recos，并与传统余弦相似度和dot-product进行对比。

**💡 创新点**

创新点在于推导更紧的上界并以秩序一致性代替线性相关性作为完美相似度的条件，扩大了相似度的捕获范围。

**🔧 技术方法**

主要技术包括重排不等式、向量排序、三类相似度度量的定义与证明，并在实验中使用STS基准进行评估。

**📊 数据集**

实验数据集涵盖7个STS benchmark（STS12–STS16、STS-B、SICK-R）以及11种预训练模型（Word2Vec, FastText, GloVe, BERT, SGPT, DPR, E5, BGE, GTE, SPECTER, CLIP-ViT）。

**📈 对比分析**

与余弦相似度比较时，recos在77个模型-数据集组合中获得71次提升，平均提升0.29分，统计显著（p<0.001）。

**⚠️ 局限性**

局限在于需要对每个向量进行排序导致O(d log d)复杂度，且高维空间中完全秩序一致性的语义解释仍不明确。

---

## 108. VISTA: Enhancing Visual Conditioning via Track-Following Preference Optimization in Vision-Language-Action Models

**arXiv ID:** 2602.05049 | [PDF](https://arxiv.org/pdf/2602.05049v1)

**作者:** Yiye Chen `[一作]` (Georgia Tech), Dongdong Chen `[通讯]` (Microsoft)

**通讯引用:** 16006 | [OpenAlex ID](https://openalex.org/A5100364587)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对视觉条件（视觉信息对动作预测的影响）进行量化，提出一种基于视觉轨迹的偏好优化（Track‑Follow DPO）和潜在蒸馏的训练框架，显著提升 Vision‑Language‑Action（VLA）模型的视觉对齐和任务成功率。

**💡 创新点**

创新点在于：①首次把视觉条件作为评估指标并与模型性能关联；②利用视觉轨迹直接构造离线偏好对，避免昂贵的 on‑policy roll‑out；③通过 DPO 在轨迹跟随任务上强化视觉‑动作对应关系，并在最终 instruction‑following 任务中通过潜在蒸馏迁移这种对齐，既无需修改网络结构也不需要额外数据。

**🔧 技术方法**

核心技术包括：Direct Preference Optimization (DPO) 用于偏好优化；视觉轨迹注释和基于轨迹的偏好对生成；潜在空间蒸馏；对 Discrete Autoregressive (OpenVLA) 与 Continuous Parallel‑Decoding (OpenVLA‑OFT) 两类 VLA 架构的统一训练策略；以及对动作分块的滑动窗口处理。

**📊 数据集**

使用的公开数据集主要是：LIBERO（四个子任务套件：Spatial、Object、Goal、Long）和 CALVIN（尤其是 ABC→D 长时序任务）。

**📈 对比分析**

与多种基线（OpenVLA 原版、TraceVLA、Diffusion Policy、LAPA、Octo、SpatialVLA、RoboDual、DITA 等）比较，实验表明在 LIBERO 上平均成功率提升约 3.1%（相对提升 0.9%），在 CALVIN ABC→D 上任务完成率提升 >2.3% 及平均完成长度提升 4%，明显优于现有 VLA 及辅助任务方法。

**⚠️ 局限性**

主要局限：仅在微调阶段进行实验，未在预训练阶段探索；仅使用静态第三人称 RGB 观察；仅针对两种 VLA 架构（离散和连续），未验证在更大规模或不同视觉输入下的泛化；对偏好对的生成仍依赖轨迹标注质量；并未考虑多模态（如本体感知、摄像头视角）扩展。

---

## 109. DCER: Dual-Stage Compression and Energy-Based Reconstruction

**arXiv ID:** 2602.04904 | [PDF](https://arxiv.org/pdf/2602.04904v1)

**作者:** Yiwen Wang `[一作]`, Jiahao Qin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种双阶段压缩与能量重建框架DCER，能够在音频、视频、文本三模态情感分析中同时提高鲁棒性与不确定性估计。

**💡 创新点**

创新点在于将模态特定频域压缩（音频小波、视频离散余弦变换）与跨模态注意力瓶颈相结合，并利用能量基重建实现缺失模态恢复与置信度量化。

**🔧 技术方法**

核心技术包括离散小波变换、DCT、可学习查询瓶颈、跨模态注意力、能量基模型与梯度下降重建。

**📊 数据集**

在CMU-MOSI、CMU-MOSEI（英语）和CH-SIMS（中文）三大情感分析基准上进行评测。

**📈 对比分析**

与八种先进方法相比，DCER在MAE、相关系数、Acc‑7、Acc‑2和F1等指标均取得显著提升，且在高缺失率和不同遮罩协议下保持鲁棒性能。

**⚠️ 局限性**

局限性包括对极端缺失率（>70%）的性能下降、对非时空模态的适用性有限以及仅在情感分析任务上验证，需进一步验证于其他多模态任务。

---

## 110. Understanding LLM Evaluator Behavior: A Structured Multi-Evaluator Framework for Merchant Risk Assessment

**arXiv ID:** 2602.05110 | [PDF](https://arxiv.org/pdf/2602.05110v1)

**作者:** Liang Wang `[一作]` (Visa Research), Yiwei Cai `[通讯]` (Visa Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一个结构化多评审框架，用于评估大型语言模型（LLM）在商户风险评估（MCC）中的推理质量，并通过五项评分准则与蒙特卡洛稳定性分析实现对评审结果的量化。

**💡 创新点**

创新点包括：① 引入自洽的共识偏差度量，消除自评循环并能同时捕获正负自评偏差；② 在支付风险领域首次实现三角验证（LLM同侪评审、人类专家、交易数据），证明评审框架与实际风险高度对应。

**🔧 技术方法**

使用技术：五款前沿LLM（GPT‑5.1、Gemini‑2.5 Pro、Grok 4、Claude 4.5 Sonnet、Perplexity Sonar）生成并交叉评估MCC风险推理；配合五维评分量表（准确性、论证质量、一致性、完整性、实用性）与10次独立蒙特卡洛抽样，获得评审稳定性统计。

**📊 数据集**

数据集：仅利用公开的MCC分类表进行推理生成；为验证结果，使用某全球支付网络2019‑2024年的交易数据（涵盖800+ MCC），计算统一的经验风险分数并与LLM评审进行相关性检验。

**📈 对比分析**

比较方法：与26位支付行业专家的评分进行一致性对比；对五款LLM的自评与他评进行共识偏差分析；将评审结果与交易数据计算Spearman相关系数。性能表现：四款LLM（Claude 4.5 Sonnet、Gemini‑2.5 Pro、Grok 4、GPT‑5.1）与交易数据相关性显著（ρ = 0.56‑0.77），评审框架与人类专家及实证数据高度对应。

**⚠️ 局限性**

局限性：① 仅评估五款闭源LLM，未覆盖开源或多语言模型；② 研究仅聚焦支付风险任务，缺乏跨领域验证；③ 采用单一模型版本快照，缺少跨版本或跨任务的稳定性与普适性评估。

---

## 111. Cross-talk based multi-task learning for fault classification of physically coupled machine system

**arXiv ID:** 2602.05146 | [PDF](https://arxiv.org/pdf/2602.05146v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 112. Momentum Attention: The Physics of In-Context Learning and Spectral Forensics for Mechanistic Interpretability

**arXiv ID:** 2602.04902 | [PDF](https://arxiv.org/pdf/2602.04902v1)

**作者:** Kingsuk Maitra `[一作]` `[通讯]` (Qualcomm), Kingsuk Maitra (Qualcomm)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提供了动量增强注意力如何实现单层诱导头形成的完整代数推导，发展了三种互补的视角：幽灵键机制、信噪比分析和框架完整性原则。

**💡 创新点**

创新点在于通过动量增强的注意力机制，绕过了标准变换器在配置空间中的深度限制，允许在单层中实现诱导头的形成。

**🔧 技术方法**

使用了动量增强的注意力机制，结合了代数推导和信号处理的分析方法。

**📊 数据集**

未具体提及使用的数据集，但提到通过结构化序列进行的实验验证了理论预测。

**📈 对比分析**

与标准变换器的比较表明，动量增强的注意力机制在单层中实现诱导的能力显著提高，尤其在信号与噪声的分离方面表现出色。

**⚠️ 局限性**

限制在于该方法依赖于动量参数的选择，且在某些情况下可能会引入额外的计算复杂性。

---

## 113. Copyright Detective: A Forensic System to Evidence LLMs Flickering Copyright Leakage Risks

**arXiv ID:** 2602.05252 | [PDF](https://arxiv.org/pdf/2602.05252v1)

**作者:** Guangwei Zhang `[一作]` (Pine AI), Denghui Zhang `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 12072 | [OpenAlex ID](https://openalex.org/A5100366431)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了《Copyright Detective》这一交互式取证系统，用于检测、分析和可视化LLM输出中的版权风险。

**💡 创新点**

创新点在于将版权合规视为证据发现过程，融合了多模态检测（内容召回、相似度、劝说式越狱、去学习验证），并通过推理时间扩展、白盒/黑盒两种方式实现可复现的法律证据。

**🔧 技术方法**

采用了提示工程、推理时间扩展、Jaccard、Levenshtein、ROUGE、语义相似度、Min‑K% 代币概率、PCA 代表性漂移、CKA 等技术；通过自定义提示模板与逆向劝说策略触发模型潜在泄露。

**📊 数据集**

使用约 20 本版权书籍（如《The Hobbit》《Harry Potter》）、5 大类 LLM（Llama、GPT‑4o‑mini 等）以及 5 种劝说式越狱模板进行实验，数据来源为公开书籍原文与模型生成文本。

**📈 对比分析**

与单次推理、标准拒绝策略对比，实验表明：推理扩展可显著发现低概率泄露；劝说式越狱可将拒绝分布向高 ROUGE‑L 区域转移；去学习检测显示深层 Transformer 产生显著的代表性漂移，指标如 AUC、TPR@5%FPR 等均表明去学习未能完全抹除版权痕迹。

**⚠️ 局限性**

局限性包括：泄露检测高度概率化，单次查询易漏；对齐压制导致记忆隐藏；去学习仅产生表征漂移，无法保证完全删除；在完全黑盒环境下只能依赖代币概率，缺乏内在可解释性。

---

## 114. The Birthmark Standard: Privacy-Preserving Photo Authentication via Hardware Roots of Trust and Consortium Blockchain

**arXiv ID:** 2602.04933 | [PDF](https://arxiv.org/pdf/2602.04933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 115. Multi-Aspect Mining and Anomaly Detection for Heterogeneous Tensor Streams

**arXiv ID:** 2602.04917 | [PDF](https://arxiv.org/pdf/2602.04917v1)

**作者:** Soshi Kakio `[一作]` (SANKEN, Osaka University), Yasushi Sakurai `[通讯]` (SANKEN, Osaka University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于高斯过程的连续事件张量流无监督分解与异常检测框架，能够同时处理离散与连续特征，并保留时间戳的连续性；

**💡 创新点**

创新点在于：①用逻辑高斯过程（LGP）先验直接估计连续属性的未知分布，避免离散化导致的信息损失；②用高斯过程建模隐层动态，使得模型可捕捉时间变化；③通过χ²拟合检验实现对群体异常（collective anomaly）的及时检测；

**🔧 技术方法**

主要技术包括：高斯过程回归（GP）、逻辑高斯过程（LGP）、Collapsed Gibbs 采样、Polya‑Gamma 数据增强、Kalman 滤波器/RTS 平滑、L‑BFGS 优化、χ² 拟合检验；

**📊 数据集**

使用了六个真实网络流量/入侵数据集（KDD‑99、UNB‑UCD、CIC‑IDS、CTU‑Botnet、UNB‑IDS、UNB‑Worm）和一个用户评论数据集；

**📈 对比分析**

与 OneClassSVM、iForestASD、RRCF、ARCUS、Mstream、MemStream、Anograph、CubeScope、CyberCScope 等方法进行对比，实验结果显示其在群体异常检测准确率最高、解释性最强，并且在随时间增长的流中保持计算时间基本不变；

**⚠️ 局限性**

局限性包括：①需要预先设定隐层数 K，且对超参数选择敏感；②逻辑高斯过程需要将连续域离散化为网格，网格数量影响精度与效率；③高斯过程的计算开销在维度较高时仍可能成为瓶颈；④目前仅在特定网络与评论数据上验证，尚未在更大规模或多模态数据集上充分评估。

---

## 116. EGSS: Entropy-guided Stepwise Scaling for Reliable Software Engineering

**arXiv ID:** 2602.05242 | [PDF](https://arxiv.org/pdf/2602.05242v1)

**作者:** Chenhui Mao `[一作]` (Ant Group), Yong Li `[通讯]` (Ant Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于工具熵的动态步骤缩放框架（EGSS），通过熵引导的搜索与跨轨迹测试整合，提升软件工程任务的推理效率与补丁可靠性。

**💡 创新点**

创新点在于：①使用工具熵定位高不确定性的关键步骤，动态分配计算资源；②构建跨轨迹的综合测试集与多模型投票策略，显著降低自欺式调试；③实现对传统重复采样的显著性能与成本提升。

**🔧 技术方法**

核心技术包括：熵引导的动态步骤搜索（DSS）与LLM判别器；测试整合增强（TCA）生成统一测试套件；多模型投票与基准评估；使用SWE-Bench的代码工具与调试工具。

**📊 数据集**

主要使用数据集为SWE-Bench-Verified（以及SWE-Bench-Lite），覆盖500+软件工程任务，验证框架的普适性。

**📈 对比分析**

与传统的重复采样+Dei Aug、Repeat+TCA以及DSS+Dei Aug等基线相比，EGSS在Kimi-K2-Instruct上从65.4%提升到70.6%，在GLM-4.6上从66.6%提升到73.8%；整体提升约5–10%；同时令平均token使用量下降约28%。

**⚠️ 局限性**

局限性包括：1）仍依赖大量LLM推理和判别器，导致整体推理时间有一定增长；2）在非SWE-Bench任务上的通用性尚未充分验证；3）对工具熵阈值与投票策略的超参数选择仍需经验调优。

---

## 117. Towards a Science of Collective AI: LLM-based Multi-Agent Systems Need a Transition from Blind Trial-and-Error to Rigorous Science

**arXiv ID:** 2602.05289 | [PDF](https://arxiv.org/pdf/2602.05289v1)

**作者:** Jingru Fan `[一作]`, Maosong Sun `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于LLM的多智能体系统的科学化设计框架，定义协作收益指标Γ，构建因子归因范式和系统化因子库，并在多领域任务上验证其有效性。

**💡 创新点**

创新点在于：①引入协作收益指标Γ，能够剥离资源扩展与真正协作带来的性能提升；②提出两步因子归因流程，使得协作因子可被客观判定为正/负；③将因子空间分层（控制层/信息层）并系统化为因子库，为MAS的实验提供可复用的结构化设计空间。

**🔧 技术方法**

利用大型语言模型驱动的多智能体架构，结合基准任务的性能评估（如准确率、覆盖率等），使用Γ指标对比单智能体与多智能体在相同资源约束下的表现，并通过内容熵、进化距离等信息层因子进行动态监测。

**📊 数据集**

在多任务多领域的公开基准上实验，包括科学研究、软件工程、基础设施管理、医疗服务、金融分析与社会科学等领域的典型数据集，展示了不同因子组合对协作收益的影响。

**📈 对比分析**

采用同等资源预算下的单智能体基线与多智能体系统的性能比值Γ进行对比；实验结果显示，当Γ>1时，多智能体能够在不增加资源的前提下实现性能提升，证明了因子归因和协作收益指标的有效性。

**⚠️ 局限性**

局限性包括：①实验设计和资源分配需要严格控制，实施成本高；②因子库仍为经验性构建，未实现完全可解释的因果推理；③过度拆分可能忽略系统级的整体涌现特性，难以覆盖所有领域的复杂交互。

---

## 118. Aligning Large Language Model Behavior with Human Citation Preferences

**arXiv ID:** 2602.05205 | [PDF](https://arxiv.org/pdf/2602.05205v1)

**作者:** Kenichiro Ando `[一作]` (RIKEN), Tatsuya Harada `[通讯]` (The University of Tokyo)

**通讯引用:** 10750 | [OpenAlex ID](https://openalex.org/A5042711470)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型（LLM）在生成文本时选择加引用的倾向，并与人工标注的引用偏好进行对比；随后通过直接偏好优化（DPO）对模型进行调优，使其更符合人类需求。

**💡 创新点**

首次系统性构建了基于八类内容（如缺失信息、医学、模糊、数字等）的句子对比较数据集，并揭示LLM在引用选择上存在的偏差（如过度引用“Citation needed”，不足引用数字和人名句子），以及通过DPO能显著提升模型与人类的契合度。

**🔧 技术方法**

使用的技术包括：LoRA参数高效微调、DeepSpeed ZeRO-Offload 加速大模型训练，以及直接偏好优化（Direct Preference Optimization, DPO）进行偏好对齐。

**📊 数据集**

数据集由 6,000 条 Wikipedia 句子构成，按作者手工标注的 8 类引用需求进行划分，并在 3,000 条句子对中进行人类二元比较，最终形成 2,596 条有效对比数据。

**📈 对比分析**

对比方法是将模型在该数据集上的引用选择与人类选择进行逐对比较，计算匹配率；在 11 种模型中，平均匹配率约 60%，最高的 DeepSeek 为 62.7%。经过 DPO 调优后，平均匹配率提升约 5.8%，小模型提升更明显（如 Llama 1B 提升 11.8%）。

**⚠️ 局限性**

局限性包括：仅使用 Wikipedia 语料，语言仅为英文；数据规模相对有限，未覆盖法律、科学等高风险领域；缺乏端到端用户体验评估（如任务成功率、信任度）。

---

## 119. Disentangled Representation Learning via Flow Matching

**arXiv ID:** 2602.05214 | [PDF](https://arxiv.org/pdf/2602.05214v1)

**作者:** Jinjin Chi `[一作]` (Jilin University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 98354 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于流匹配的分离表示学习框架，将分离目标视为在紧凑潜在空间中对因子条件化的流动；

**💡 创新点**

通过引入因子对齐的输出注意力和正交正则化，显式实现语义对齐并抑制因子间干扰；

**🔧 技术方法**

使用流匹配（Conditional Flow Matching）、跨注意力、输出注意力路由以及正交正则化等技术；

**📊 数据集**

在Cars3D、Shapes3D、MPI3D-toy以及CelebA四个数据集上进行实验；

**📈 对比分析**

与VAE、GAN及现有扩散方法比较，取得了更高的FactorVAE、DCI分数、TAD分数和更低的FID，表明在分离度、可控性和生成质量上均优于对比方法；

**⚠️ 局限性**

受限于对因子条件的线性插值假设和对正则化强度的调参，未来需探索更灵活的桥接策略和更强的理论保证。

---

## 120. Extracting Recurring Vulnerabilities from Black-Box LLM-Generated Software

**arXiv ID:** 2602.04894 | [PDF](https://arxiv.org/pdf/2602.04894v1)

**作者:** Tomer Kordonsky `[一作]` (Technion), Avi Mendelson `[通讯]` (Technion)

**通讯引用:** 2456 | [OpenAlex ID](https://openalex.org/A5089135250)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Feature–Security Table（FSTab）框架，通过观察 LLM 生成软件的前端特征，预测隐藏的后端安全漏洞，实现黑盒攻击和模型级别的漏洞持续性评估。

**💡 创新点**

创新点在于①利用 LLM 生成代码中的可复现漏洞模板构建模型特定的“漏洞指纹”；②将此指纹转化为可查询的概率表（FSTab），在不访问后端代码的情况下完成漏洞优先级定位；③提出四项跨域、跨功能、跨重述的漏洞持续性度量（FVR、RVP、DVR、CDT），系统性量化模型产生漏洞的规律性。

**🔧 技术方法**

技术手段包括：LLM 代码生成（GPT‑5.2、Claude‑4.5 Opus、Gemini‑3 Pro/Flash 等）；静态分析（CodeQL、Semgrep）标注真实漏洞；AST 与正则式特征提取生成前端特征集；点互信息（PMI）与多样性惩罚算法构建 FSTab；黑盒攻击流程（Recon‑Mapping‑Query）与度量公式。

**📊 数据集**

数据集主要为 WebGenBench（5 个应用领域，共 1050 个程序，约 64 GB 代码）以及额外的 E2EDev 评估；在每个领域随机划分 5 个 prompt 用于构建 FSTab，剩余 5 个用于 held‑out 与 cross‑domain 评估。

**📈 对比分析**

通过与传统静态扫描（CodeQL/ Semgrep）对比，FSTab 在 held‑out 场景下平均攻击成功率（ASR）可达 81%–100%，覆盖率（ACR）在 70%–100% 之间；在跨域测试中表现仅略有下降，说明模型漏洞模板具有通用性。度量指标显示 GPT‑5.2、Claude‑4.5 Opus、Gemini‑3 系列在 FVR、RVP、DVR、CDT 上均高于基线，证明模型级别的漏洞持续性显著。

**⚠️ 局限性**

局限性包括：①特征提取与静态分析工具可能产生误报/漏报，影响 FSTab 的精确度；②FSTab 只考虑已知漏洞规则，未覆盖运行时或逻辑错误；③攻击前需知道生成模型的身份，若模型隐藏身份则难以使用；④研究聚焦于 Python/JavaScript，其他语言的可迁移性待验证。

---

## 121. Low-Cost Underwater In-Pipe Centering and Inspection Using a Minimal-Sensing Robot

**arXiv ID:** 2602.05265 | [PDF](https://arxiv.org/pdf/2602.05265v1)

**作者:** Kalvik Jakkala `[一作]` (Texas A&M University), Jason O'Kane `[通讯]` (Texas A&M University)

**通讯引用:** 1463 | [OpenAlex ID](https://openalex.org/A5089976123)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种仅使用IMU、压力计和两束单波束声呐的最小化感知框架，实现水下管道内自由漂移机器人对中与通行。

**💡 创新点**

创新点包括：① 计算高效的声呐强度到距离的提取算法，能够在噪声和多路径环境下实时识别管壁；② 利用已知管径的闭式几何模型与两束声呐距离估计管道中心，配合自适应置信度加权的Kalman滤波；③ 基于置信度的PD控制实现对中与前进运动的自适应调节，避免过度控制。

**🔧 技术方法**

使用的技术包括：单波束声呐强度信号的近场抑制、去噪与边缘增强、峰值检测与距离转换；几何中心估计与自适应测量协方差建模；Kalman滤波与自适应置信度权重；PD控制与姿态稳定。

**📊 数据集**

实验数据集主要来自BlueROV2在直径0.46 m织物管内的声呐波形，采集了约1000个声呐波形并手工标注真实管壁距离；此外还使用了二维圆形管道的仿真数据。

**📈 对比分析**

与人工标注的真实距离相比，声呐距离提取的RMSE为0.02 m；在仿真中平均10步收敛到0.05 m误差；在现场实验中，机器人能够在存在弯曲、变形与流动的管道中保持良好对中并完成全程通行，验证了方法在低成本、实时场景下的优越性能。

**⚠️ 局限性**

限制包括：需要已知管径且管道截面近似圆形；对大yaw误差不鲁棒；在强流动或湍流条件下性能可能下降；系统参数（滤波增益、阈值、置信度权重）需要手动调优。

---

## 122. Scaling Laws for Embedding Dimension in Information Retrieval

**arXiv ID:** 2602.05062 | [PDF](https://arxiv.org/pdf/2602.05062v1)

**作者:** Julian Killingback `[一作]` (Center for Intelligent Information Retrieval), Hamed Zamani `[通讯]` (Center for Intelligent Information Retrieval)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究了稠密检索中向量维度与模型参数的扩展对检索性能与成本的影响，并基于对比实验提出了向量维度与模型参数的联合扩展的量化规律；

**💡 创新点**

首次在检索任务中系统性验证向量维度与模型参数扩展形成的交互式规模规律，并给出了针对不同计算预算下的最优配置方案；

**🔧 技术方法**

使用了对比损失（contrastive entropy）作为检索性能指标、知识蒸馏、负采样、近似最近邻（ANN）等技术；

**📊 数据集**

使用的主要数据集包括MSMarco、TREC的Legal QA、TREC的Paper Retrieval等；

**📈 对比分析**

通过将模型扩展到多种向量维度与模型规模，并与同类模型和传统检索指标（RR@10、R@1000）进行比较，实验表明向量维度和模型参数的联合扩展显著提升检索性能且可根据计算预算实现性能最优化；

**⚠️ 局限性**

局限性在于对比损失作为检索性能的代理在某些极端维度/规模下的预测不够稳定，对极端分布任务的泛化仍需进一步研究。

---

## 123. Trojan Attacks on Neural Network Controllers for Robotic Systems

**arXiv ID:** 2602.05121 | [PDF](https://arxiv.org/pdf/2602.05121v1)

**作者:** Farbod Younesi `[一作]` (Concordia University), Amr Youssef `[通讯]` (Concordia University)

**通讯引用:** 6048 | [OpenAlex ID](https://openalex.org/A5085765243)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文针对差速驱动机器人的神经网络控制器，设计并实现了一个轻量级的后门网络，能够在触发特定姿态与目标位置的罕见条件下悄无声息地修改轮速，从而实现机器人停止或加速等恶意行为；

**💡 创新点**

创新点在于将后门攻击扩展到连续控制任务，并通过并行小型多层感知网络实现对主控制器的乘法门控，触发条件高度专一，保持低误报率，同时保持正常运行时性能不受影响；

**🔧 技术方法**

使用了行为克隆（Behavioral Cloning）训练主控制器（全连接MLP，SiLU激活），并训练后门网络（ReLU激活）输出乘法因子，二者通过乘法门控融合；采用AdamW优化器、MSE损失、Euler积分仿真，评估IAE和NAMD等指标；

**📊 数据集**

采用自行生成的合成数据集：通过仿真几何姿态跟踪控制器生成约10万条样本，记录机器人姿态、目标位置及对应轮速；后门训练集包含触发区域样本（20×20cm）与非触发样本，保持极低触发比例；

**📈 对比分析**

与几何控制器比较，主控制器的IAE仅略高（57.84 vs 54.82），保持接近；后门攻击在触发区的NAMD>0.9，非触发区<0.05，说明攻击效果显著且隐蔽；

**⚠️ 局限性**

局限性包括：仅在仿真环境验证；仅针对差速驱动机器人；后门触发条件为固定空间区域，若被检测或改动可能失效；未给出防御或检测方案；对复杂任务或多传感器系统的适用性未知。

---

## 124. Enhanced QKNorm normalization for neural transformers with the Lp norm

**arXiv ID:** 2602.05006 | [PDF](https://arxiv.org/pdf/2602.05006v1)

**作者:** Ezequiel Lopez-Rubio `[一作]`, Esteban Jose Palomo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Lp范数的通用化QKNorm注意力正则化方法，并在Transformer中进行实验验证。

**💡 创新点**

创新点是将传统的欧氏L2正则化扩展为可调p的Lp正则化，允许通过p控制注意力关注的特征范围，从而实现更灵活的几何设计。

**🔧 技术方法**

使用了Transformer的自注意力机制、Lp范数归一化、可学习的缩放参数α以及PyTorch框架进行实现。

**📊 数据集**

使用了Hugging Face的“English poet”40000行字符级语料集，采用字符级Tokenizer进行训练。

**📈 对比分析**

通过在p∈{1.0,1.5,2.0,2.5,3.0,3.5,4.0}的多折交叉验证实验，比较了验证交叉熵损失、收敛速度和训练时间。结果显示p>2时验证损失最低、收敛最快，而训练时间基本不变。

**⚠️ 局限性**

局限性包括：仅在小规模字符级模型上验证，未评估在大型文本或语言建模任务中的表现；p的最优取值依赖任务，缺乏理论指导；未探索p趋近∞时的极限效果。

---

## 125. CoSA: Compressed Sensing-Based Adaptation of Large Language Models

**arXiv ID:** 2602.05148 | [PDF](https://arxiv.org/pdf/2602.05148v1)

**作者:** Songtao Wei `[一作]` (University of Texas at Dallas), Bingzhe Li `[通讯]` (University of Texas at Dallas)

**通讯引用:** 890 | [OpenAlex ID](https://openalex.org/A5048972267)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于压缩感知的参数高效微调方法 CoSA，利用固定随机投影矩阵和紧凑可训练核心实现对大型语言模型的任务特定适配。

**💡 创新点**

创新点在于将权重更新视为压缩感知中的合成过程，采用随机投影的 Kronecker 字典并证明满足 RIP，从而在不低秩约束的前提下保持表达力和优化稳定性。

**🔧 技术方法**

核心技术包括压缩感知理论、RIP 证明、Kronecker 乘积字典、随机投影、低维可训练核心矩阵 Y 以及与传统 LoRA、PiSSA 等方法的对比。

**📊 数据集**

使用的数据集包括 GLUE（SST‑2、MRPC、CoLA、QNLI、RTE、STS‑B）、MetaMathQA、Code‑Feedback、GSM8K、MATH、HumanEval、MBPP 等 NLU 与 NLG 评测基准，并在 RoBERTa、LLaMA、Qwen 等不同规模模型上进行实验。

**📈 对比分析**

与全微调、LoRA、AdaLoRA、PiSSA、DoRA、VeRA 等现有 PEFT 方法比较，CoSA 在多数 GLUE 任务和大模型的 NLG、推理、代码生成任务上均达到或超过最强基线，参数量显著降低且内存占用更小。

**⚠️ 局限性**

局限性包括需要手动调节压缩维度 (a,b) 以平衡表达力与参数量，随机投影可能导致不同实验之间的可重复性略差；此外在极大模型或极高维任务中，随机投影的理论保证与实际效果可能出现差距。

---

## 126. Co-Designing Collaborative Generative AI Tools for Freelancers

**arXiv ID:** 2602.05299 | [PDF](https://arxiv.org/pdf/2602.05299v1)

**作者:** Kashif Imteyaz `[一作]` (Civic AI Lab), Saiph Savage `[通讯]` (Civic AI Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开展了基于协作式设计（co-design）方法的研究，邀请27名自由职业者（开发者、设计师、作家等）参与同步与异步工作坊，探索并构想面向自由职业者协作的生成式AI工具。

**💡 创新点**

创新点在于：① 将批判理论（马克思主义技术理性）与共创设计结合，帮助自由职业者从“技术主导效率”反向构想“辅助式AI”与“协作摩擦”设计原则；② 通过生成式AI（DALL·E）作为设计探针，促使非专业设计者能够可视化未来工具；③ 提出了针对自由职业者特定需求的三大设计原则（技术多样性、AI辅助角色、人工协调）。

**🔧 技术方法**

使用技术主要包括：
- 生成式AI工具ChatGPT（文本）、DALL·E（图像）作为交互与设计探针；
- 远程协作平台Slack与Zoom进行同步/异步工作坊；
- 研究方法：Future Workshops、反思性主题分析。

**📊 数据集**

数据来源为参与者的访谈记录、讨论转录、生成的DALL·E图像等定性材料，并非传统机器学习数据集；因此未使用公开数据集。

**📈 对比分析**

研究未涉及算法性能评估或对照实验，比较方式为定性主题分析与案例研究；没有数值指标。

**⚠️ 局限性**

局限性包括：
- 样本规模有限（27人），多为北美与拉美地区自由职业者，可能缺乏对其他地区或行业的普适性；
- 依赖自述与讨论，可能存在社交期望偏差；
- 研究聚焦设计愿景与原则，未对实际工具实现与用户体验进行验证；
- 生成式AI探针的结果受模型版本与提示设计影响，难以复制。

---

## 127. Privileged Information Distillation for Language Models

**arXiv ID:** 2602.04942 | [PDF](https://arxiv.org/pdf/2602.04942v1)

**作者:** Emiliano Penaloza `[一作]` (ServiceNow), Massimo Caccia `[通讯]` (ServiceNow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了两种利用训练时特权信息（PI）的蒸馏算法——π-Distill 和 On‑Policy Self‑Distillation（OPSD），实现对前沿模型（缺乏完整 Chain‑of‑Thought 轨迹）的有效蒸馏，最终得到能在测试时不依赖 PI 的高性能代理。

**💡 创新点**

创新点在于：
• 采用共享参数的教师‑学生框架，将 PI 引入教师策略并通过逆 KL 正则化保证教师与学生的分布不相距太远；
• 通过 π‑Distill 的联合训练（α∈[0,1]）同时优化教师与学生，显著缓解了传统序列化蒸馏中的分布漂移问题；
• 引入 OPSD 作为无偏见的 on‑policy 蒸馏方案，利用逆 KL 作为细粒度奖励，可在缺少 CoT 的情形下进一步提升性能。

**🔧 技术方法**

技术细节包括：
• 基于 GRPO（Group Relative Policy Optimization）的 RL 策略优化；
• 逆 KL 正则化约束教师与学生的差异；
• 通过共享参数实现教师/学生的知识迁移；
• 对 PI 进行三种形式的转换（工具调用+参数、仅工具调用、模型生成提示）以探索信息密度对性能的影响。

**📊 数据集**

使用的数据集有：
• τ‑Bench（零售和航空两类任务）
• Travel Planner（工具调用规划任务）
• GEM search‑tool suite（七个公开工具任务，用于 OOD 泛化评估）。

**📈 对比分析**

与基线（SFT+CoT+RL、SFT+CoT、SFT+无CoT、纯 RL 等）比较时，π‑Distill 和 OPSD 均显著优于基线；在 Travel Planner 上最高提升约 11.8%，在 τ‑Bench 零售和航空上分别提升约 2.1% 与 6.0%；在 GEM OOD 评估中，π‑Distill/OPSD 亦能保持或提升性能，且在大型模型上不出现 RL 退化现象。

**⚠️ 局限性**

主要限制包括：
• 仅在已获取前沿模型轨迹的情形下验证，缺乏对无前沿模型或无真值答案情况的实验；
• 实验规模局限于 8B 以下模型，未验证更大规模模型的适用性；
• 需要手动调节 β 与 α 等超参数，且不同 PI 类型对训练稳定性影响较大。

---

## 128. Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries

**arXiv ID:** 2602.04940 | [PDF](https://arxiv.org/pdf/2602.04940v1)

**作者:** Hang Zhou `[一作]` (Tsinghua University), Mingsheng Long `[通讯]` (Tsinghua University)

**通讯引用:** 29064 | [OpenAlex ID](https://openalex.org/A5019241553)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Transolver‑3 框架，能够在单张 GPU 上以工业级精度处理超过 10^8 个网格单元的 PDE 预测。

**💡 创新点**

创新点包括：Physics‑Attention 结构的快速切分/重排、几何切片分块、梯度检查点、几何摊销训练以及解耦式推理（物理状态缓存 + 全网格解码）。

**🔧 技术方法**

技术手段包括：线性注意力、矩阵乘法结合性优化、几何切片分块、梯度检查点、物理状态缓存、分块推理、自动微分与内存/计算折中策略。

**📊 数据集**

使用了三大工业级 CFD 基准：NASA‑CRM、AhmedML、DrivAerML，网格规模从 10^5 到 1.6×10^8。

**📈 对比分析**

通过与 Graph‑U‑Net、GINO、GAOT、UPT、AB‑UPT、Transolver、Transolver++ 七种基线比较，Transolver‑3 在 10/11 评估指标中取优，尤其在 160M 网格的体场预测和整流量系数计算上明显领先。

**⚠️ 局限性**

局限性：仍需多 GPU 分布式训练；推理仍受限于单 GPU 内存，需分块推理；tile 大小折中导致计算与内存的权衡；对极高细节网格的捕捉能力仍有提升空间。

---

## 129. Evaluating Robustness and Adaptability in Learning-Based Mission Planning for Active Debris Removal

**arXiv ID:** 2602.05091 | [PDF](https://arxiv.org/pdf/2602.05091v1)

**作者:** Agni Bandyopadhyay `[一作]` (Julius-Maximilians-Universität Würzburg), Günther Waxenegger-Wilfing `[通讯]` (Julius-Maximilians-Universität Würzburg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在低地球轨道上进行多目标碎屑清除任务时，比较了名义Mask PPO、域随机Mask PPO和纯MCTS三种规划方法，评估其在燃料和任务时间受限情况下的鲁棒性与效率。

**💡 创新点**

通过在训练时对燃料预算和任务时限进行域随机化，使得Mask PPO在面对分布偏移时表现更稳健，并提出学习与搜索相结合的混合框架以弥补单一方法的不足。

**🔧 技术方法**

使用Maskable Proximal Policy Optimization（带动作掩码）与基于UCT的Monte Carlo树搜索，以及自定义的Gymnasium兼容仿真环境实现轨道动力学与约束处理。

**📊 数据集**

数据集为自定义的随机碎屑场景，每个场景包含50个碎屑，测试共100个实例，覆盖三种任务约束（正常、燃料减半、时间减半）。

**📈 对比分析**

在三种约束情形下评估平均已清除碎屑数：名义PPO在正常情形下最佳，域随机PPO在约束变更时性能最稳健，MCTS在燃料极限下表现最优但计算时间高达数分钟，整体展示了速度-鲁棒性权衡。

**⚠️ 局限性**

局限包括MCTS计算成本过高、PPO在分布偏移下仍易失效、实验仅限单机碎屑清除、未考虑轨道扰动与多代理协同等真实工程难题。

---

## 130. Laws of Learning Dynamics and the Core of Learners

**arXiv ID:** 2602.05026 | [PDF](https://arxiv.org/pdf/2602.05026v1)

**作者:** Inkee Jung `[一作]` (Boston University), Siu Cheong Lau `[通讯]` (Boston University)

**通讯引用:** 388 | [OpenAlex ID](https://openalex.org/A5052908802)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于熵的学习动力学法则，并基于此设计了终身多代集成学习（logifold）用于检测并抵御对抗性攻击，形成了免疫机制IMM；

**💡 创新点**

创新点在于将熵视为无标签的不确定度度量，定义“核心”区域并通过熵阈值实现模型层级化、迁移和免疫；并在对抗环境中通过多代生成提升鲁棒性；

**🔧 技术方法**

主要技术包括熵与交叉熵理论推导、集合学习熵的定义、熵阈值核心划分、对抗训练与迁移学习、投票聚合（概率平均）以及实验中的APGD-CE与AutoAttack生成的对抗样本；

**📊 数据集**

使用CIFAR‑10数据集，包含原始、弱扰动（ε=0.5/0.7）与强扰动以及转移式对抗样本；

**📈 对比分析**

与单模型、基线平均集成、以及仅弱扰动专用模型进行对比；在清洁数据上性能相当，弱扰动下准确率提升约9个百分点，强扰动下提升超过20个百分点，混合域准确率从64%提升至87%（对抗增强），总熵显著下降；

**⚠️ 局限性**

局限性包括缺乏最坏情况下的理论鲁棒性保证、对超参数（熵阈值、生成数）敏感、以及实验仅在CIFAR‑10上验证，缺乏更大规模或不同任务的评估。

---

## 131. Multilingual Extraction and Recognition of Implicit Discourse Relations in Speech and Text

**arXiv ID:** 2602.05107 | [PDF](https://arxiv.org/pdf/2602.05107v1)

**作者:** Ahmed Ruby `[一作]` (Uppsala University), Sara Stymne `[通讯]` (Uppsala University)

**通讯引用:** 893 | [OpenAlex ID](https://openalex.org/A5051985869)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了英文、法语、西班牙语三种语言的多模态隐式话语关系(IDR)数据集，并提出融合文本与音频的多模态分类模型。

**💡 创新点**

创新点：1) 通过翻译显式化连接词自动生成多语言多模态IDR数据集；2) 在Qwen2‑Audio基础上融合prosody特征与音频统计池化，形成端到端的多模态框架；3) 评估单语与跨语训练的迁移效果。

**🔧 技术方法**

技术手段：使用Qwen2‑Audio预训练模型、LoRA微调、Faster‑Whisper音频转录与prosody提取、log‑mel特征、跨模态注意力融合、以及多语言共享训练。

**📊 数据集**

数据集：基于TEDx讲座的文本与音频，自动构建英文、法语、西班牙语IDR数据集（共计约2,600例），并与已有阿拉伯语IDR数据及TED‑MDB人标注对照使用。

**📈 对比分析**

比较方法与性能：与BERT/TF‑IDF、Prosodic+LogReg、BERT+Wav2vec2等基线对比；单语训练中文本模型最高F1（英语0.53），多模态在低资源西班牙语提升明显；跨语训练在法语、西班牙语提升，但在阿拉伯语、英语略有负迁移。

**⚠️ 局限性**

局限性：仅包含文本与音频，缺乏视觉信息；多模态融合效果不稳定，部分语言表现下降；显式化翻译方法覆盖率有限；低资源语言仍受文本稀缺限制。

---

## 132. ShapePuri: Shape Guided and Appearance Generalized Adversarial Purification

**arXiv ID:** 2602.05175 | [PDF](https://arxiv.org/pdf/2602.05175v1)

**作者:** Zhe Li `[一作]` (Friedrich Alexander University Erlangen Nuremberg), Bernhard Kainz `[通讯]` (Friedrich Alexander University Erlangen Nuremberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练了一种 ShapePuri 框架，利用形状编码和外观去偏来提升图像分类模型的对抗鲁棒性。

**💡 创新点**

创新点包括：① 用 Signed Distance Function (SDF) 构建稠密几何先验；② 引入 Global Appearance Debiasing (GAD) 模块以去除对纹理的依赖；③ 在训练时采用多流监督，推理阶段不需额外模块或计算。

**🔧 技术方法**

采用 SDF 形状编码、随机卷积去偏、交叉熵多流损失、PGD/AutoAttack 对抗生成、标准卷积网络（ResNet‑101、ResNet‑152、ConvNeXt‑L）等技术。

**📊 数据集**

使用 ImageNet 数据集（训练/验证划分）。

**📈 对比分析**

与 DiffPure、OSCP、MeanSparse 等现有对抗防御方法比较，AutoAttack 下实现 84.06% 清洁精度、81.64% 鲁棒精度，首次突破 80% 鲁棒阈值，领先 SOTA 约 7.45%。

**⚠️ 局限性**

局限性：仅在 ImageNet 分类任务上验证；对更复杂攻击、不同数据分布或多模态任务的适用性尚未探究；SDF 的二值化步骤对噪声敏感，可能影响稳定性。

---

## 133. SIDeR: Semantic Identity Decoupling for Unrestricted Face Privacy

**arXiv ID:** 2602.04994 | [PDF](https://arxiv.org/pdf/2602.04994v1)

**作者:** Zhuosen Bao `[一作]` (Xiamen University of Technology), Jun Luo `[通讯]` (Nanyang Technological University)

**通讯引用:** 12681 | [OpenAlex ID](https://openalex.org/A5081222445)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了SIDeR框架，实现可逆且语义可控的人脸隐私保护，能够在视觉上匿名化人脸同时保持机器识别一致性并支持授权恢复。

**💡 创新点**

创新点在于将语义‑身份解耦与扩散模型潜空间对抗优化相结合，并引入可逆嵌入的双层可逆神经网络，实现视觉匿名、身份一致与可逆恢复三位一体。

**🔧 技术方法**

使用Stable Diffusion扩散模型进行潜空间对抗优化，配合动量迭代、语义引导、文本提示生成与LLM微调；利用可逆神经网络（INN）实现嵌入与解嵌。

**📊 数据集**

主要实验数据集为 CelebA‑HQ 与 FFHQ 两大高分辨率人脸数据库。

**📈 对比分析**

与多种噪声攻击、无约束生成、语义不变等方法对比，SIDeR 在黑盒攻击中的攻击成功率高达 99% 以上，PSNR 恢复提升超过 41%，FID 更低，整体性能明显优于现有基线。

**⚠️ 局限性**

主要局限在于对 λ、动量等超参数高度敏感，过大 λ 可能导致视觉失真；对不同模型的迁移性仍有限，且对极端隐蔽强度下的视觉自然度仍需进一步改进。

---

## 134. PieArena: Frontier Language Agents Achieve MBA-Level Negotiation Performance and Reveal Novel Behavioral Differences

**arXiv ID:** 2602.05302 | [PDF](https://arxiv.org/pdf/2602.05302v1)

**作者:** Chris Zhu `[一作]` (Yale University), Daylian Cain `[通讯]` (Yale University)

**通讯引用:** 3779 | [OpenAlex ID](https://openalex.org/A5066159562)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并评估了一个基于真实MBA案例的双人谈判基准PieArena，用以衡量语言模型在价值创造和价值索取中的表现；

**💡 创新点**

①引入可饱和抗抵消的竞争式评估；②构建大规模LM与人类对战数据；③开发共享意向性代理框架提升模型协作；④提出基于高斯-广义Bradley–Terry–Luce的连续收益排名模型；④提供多维行为诊断（欺骗、准确率、合规性、声誉等）；

**🔧 技术方法**

使用自然语言生成、强化学习的代理架构、共享意向性状态跟踪与策略规划、Gaussian–GTL统计模型以及多轮对话抓取与结构化输出解析；

**📊 数据集**

包含326个候选LLM（筛选后13个），167名MBA学生的谈判记录（总计~25000条LM对话，超过1000条人机对话），以及多种真实商务案例（SnyderMed、Main Street、Top Talent、Twisted Tree、Z‑lab）；

**📈 对比分析**

通过镜像对弈、交叉对弈和人机对弈进行比较。结果显示：前沿模型（如GPT‑5、Gemini‑3‑Pro）在多问题场景下可超越顶尖MBA学生；中下层模型在共享意向性辅助下大幅提升；在单问题场景中GPT‑5明显占优；行为诊断揭示强模型常伴高欺骗率与高声誉；

**⚠️ 局限性**

受限于：①模型仍易违反规则、误算、撒谎；②行为维度与最终收益不一定正相关；③偏向线性效用假设，忽略非线性或情境依赖偏好；④仅覆盖有限商务案例，缺乏跨文化或不同行业的泛化验证；

---

## 135. Rule-Based Spatial Mixture-of-Experts U-Net for Explainable Edge Detection

**arXiv ID:** 2602.05100 | [PDF](https://arxiv.org/pdf/2602.05100v1)

**作者:** Bharadwaj Dogga `[一作]` (University of Cincinnati), Kelly Cohen `[通讯]` (University of Cincinnati)

**通讯引用:** 3243 | [OpenAlex ID](https://openalex.org/A5034113408)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种可解释的基于sMoE U-Net的边缘检测模型，利用空间自适应Mixture-of-Experts和TSK模糊推理实现像素级的可解释策略和规则。

**💡 创新点**

创新点在于将空间自适应Mixture-of-Experts块嵌入U-Net解码器跳跃连接，并用TSK模糊头替代传统Sigmoid分类层，实现可视化的规则触发图和策略图，既保留了深度学习的精度，又提供了可解释的决策逻辑。

**🔧 技术方法**

使用的技术包括Sobel预处理、空间自适应Mixture-of-Experts（sMoE）模块、First-Order TSK模糊推理引擎、二分类损失组合（BCE+Dice）以及知识蒸馏方法训练模糊头。

**📊 数据集**

使用了BSDS500数据集（200训练/100验证/200测试），并通过旋转增强提升鲁棒性。

**📈 对比分析**

与传统Canny、Sobel、U-Net和HED进行比较，ODS F-score分别为0.5450、0.5769、0.7437和0.7688；sMoE U-Net取得ODS 0.7628、OIS 0.7458，略优于U-Net且接近HED，说明可解释性并未显著牺牲性能。

**⚠️ 局限性**

局限性包括：规则数目有限（仅4条），可能无法覆盖所有复杂边缘类型；缺乏在更大规模或专业安全关键场景（如医学影像、航空维护）中的实证验证；以及在高分辨率图像上可能需要更深层次的专家网络来保持性能。

---

## 136. Challenges in Solving Sequence-to-Graph Alignment with Co-Linear Structure

**arXiv ID:** 2602.05186 | [PDF](https://arxiv.org/pdf/2602.05186v1)

**作者:** Xingfu Li `[一作]` (Guizhou University of Finance and Economics), Xingfu Li `[通讯]` (Guizhou University of Finance and Economics)

**通讯引用:** 76 | [OpenAlex ID](https://openalex.org/A5027094450)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了序列对图对齐中共线性链的计算复杂度。

**💡 创新点**

提出Gap-CLC和Edit-CLC两种新模型，并证明Gap-CLC与Exa-SGM同等难度、Edit-CLC在有误差时为NP-难。

**🔧 技术方法**

使用多项式时间归约、SETH假设、NP-难性证明等理论工具。

**📊 数据集**

未使用具体实验数据集，研究基于理论分析。

**📈 对比分析**

通过与已知难度问题的归约比较，说明无子二次算法；未给出实验性能。

**⚠️ 局限性**

仅给出复杂度上界，缺乏实际算法实现，对显式锚点的处理仍未解决。

---

## 137. Prediction Laundering: The Illusion of Neutrality, Transparency, and Governance in Polymarket

**arXiv ID:** 2602.05181 | [PDF](https://arxiv.org/pdf/2602.05181v1)

**作者:** Yasaman Rohanifar `[一作]` (University of Toronto), Sharifa Sultana `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 648 | [OpenAlex ID](https://openalex.org/A5103026409)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对Polymarket进行社会技术审计，研究其预测市场的运作方式并提出“预测洗钱”概念

**💡 创新点**

首次系统阐述“预测洗钱”及其四阶段生命周期模型，揭示平台如何将主体不确定性与资本偏好洗净为看似客观的概率信号

**🔧 技术方法**

采用数字人种学、解释性走查、半结构访谈及主题编码等质性研究技术

**📊 数据集**

使用27名受访者数据、平台交易记录、Discord、X、Reddit等社交媒体日志

**📈 对比分析**

以主题对比不同受访者群体的经验为主要方法，未给出量化性能指标，提供的是对比性质性洞察

**⚠️ 局限性**

样本规模有限、仅聚焦Polymarket、缺乏定量验证，平台快速演进可能影响结果的长期适用性

---

## 138. Deterministic Retrieval at Scale: Optimal-Space LCP Indexing and 308x Energy Reduction on Modern GPUs

**arXiv ID:** 2602.04936 | [PDF](https://arxiv.org/pdf/2602.04936v1)

**作者:** Stanislav Byriukov `[一作]` `[通讯]`, Stanislav Byriukov

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种基于最长公共前缀（LCP）的 deterministic top‑k 检索结构，证明其空间下界为Ω(N)，并给出 O(N·L) 空间、O(L+k) 查询时间的最优算法；

**💡 创新点**

创新点在于：①用 trie 结构实现完全确定性检索；②证明 LCP 检索的空间下界为 Ω(N)；③提出 Thermal‑Aware Logic (TAL) 能量优化，使能耗降低 308 倍；④在安全关键系统（如航天 GNC）中验证了 determinism 与能效兼顾的可行性；

**🔧 技术方法**

使用的技术包括：单词探测模型（cell‑probe）、Trie 与子树计数、前缀桶分区扫描、GPU 并行实现、CUDA 能耗监测与热管理、以及在 NVIDIA H100 上的性能基准；

**📊 数据集**

主要使用的实验数据集为：(1) 2 M 条长度 256 的符号序列（模拟大规模检索）；(2) GNC 传感器融合数据（≈ 1 000 步、4 000 Hz）；(3) 多代理协同场景中的共享观测数据；

**📈 对比分析**

与传统 pairwise materialization（Θ(N²)）、HNSW、IVF、FlashAttention 等方法对比，LCP‑Index 在空间上从 1 TB 降到 200 MB，查询时间从 O(L) 维持不变但显著低于预期；能耗方面单查询从 4.46 J 降至 0.0145 J（308×）；GPU 利用率保持 98.98%，吞吐量达 4 157 Hz；

**⚠️ 局限性**

局限性：只能处理符号序列，无法直接支持连续嵌入的语义相似度；字母表较大时 trie 子节点映射开销增大；目前仅支持静态数据集，缺乏高效的动态插入/删除机制；

---

## 139. GT-SVJ: Generative-Transformer-Based Self-Supervised Video Judge For Efficient Video Reward Modeling

**arXiv ID:** 2602.05202 | [PDF](https://arxiv.org/pdf/2602.05202v1)

**作者:** Shivanshu Shekhar `[一作]` (University of Illinois Urbana-Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 24858 | [OpenAlex ID](https://openalex.org/A5100378779)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

将先进的视频生成模型改造为能量模型，并通过自监督对比学习得到判别器，再用少量人工标注的偏好数据训练奖励模型，实现视频质量的自动评估。

**💡 创新点**

① 把生成模型视为能量模型，利用对比学习将其转换为判别器；② 通过在潜在空间中施加多种精细扰动（时间切片、帧打乱、特征交换等）生成困难负样本；③ 仅在Transformer中间层插入LoRA，既保持模型表达力又实现参数高效微调；④ 以仅30K人工标注数据即可达到或超过现有VLM基准，显著提升数据效率。

**🔧 技术方法**

使用基于Transformer的视频生成模型与VAE编码器；能量模型（EBM）对比学习；LoRA参数高效微调；偏好学习采用Bradley‑Terry（含ties）与回归；自监督扰动操作（帧打乱、特征交换、时间切片交换等）。

**📊 数据集**

20K真实互联网视频、30K生成视频、30K人工标注偏好；基准评测集包括GenAI‑Bench、MonteBench、VideoReward‑Bench；对比的基准模型有VisionReward、VideoReward及VideoReward‑Bench。

**📈 对比分析**

先训练判别器再进行偏好调优得到奖励模型；在GenAI‑Bench和MonteBench上实现state‑of‑the‑art，分别比前沿基准提升约25%/4%（有tie）和3%/8%（无tie）；在VideoReward‑Bench排名第二，落后约7%；只需30K标注，即比VideoReward 6×、比VisionReward 65×。

**⚠️ 局限性**

缺乏对话式交互接口；在VideoReward‑Bench分布偏移时性能下降；模型目前基于单一生成骨干，对更长时长视频的推广尚待验证。

---

## 140. The Necessity of a Holistic Safety Evaluation Framework for AI-Based Automation Features

**arXiv ID:** 2602.05157 | [PDF](https://arxiv.org/pdf/2602.05157v1)

**作者:** Alireza Abbaspour `[一作]` (Qualcomm Technologies Inc), Jeff Stafford `[通讯]` (Qualcomm Technologies Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过理论分析与案例研究，证明即使是被划为 QM 级的 AI 组件也可能产生 SOTIF 风险，并提出将 ISO 26262、ISO 21448 与新兴 ISO/PAS 8800 结合，形成完整的安全评估框架。

**💡 创新点**

创新点在于：1) 系统性地将三大安全标准整合，对 QM AI 组件进行 SOTIF 评估；2) 提出基于 SOTIF 需求的 AI 安全生命周期与安全需求推导流程；3) 以低层感知（LLP）和无手驾驶（HOD）为案例，展示 QM 组件风险与对应的安全措施。

**🔧 技术方法**

使用的技术包括：系统理论过程分析（STPA）、SOTIF 风险识别与评估（SIRA）、因果树分析（CTA）、AI 安全生命周期（V&V、监测、变更管理）、多模感知融合、信号置信度阈值设计、离线/在线校准与漂移监测等。

**📊 数据集**

文中未给出具体公开数据集，主要讨论了高精地图、摄像头、雷达、激光雷达等多模传感器的数据及其融合，但未使用任何公开数据集进行实验验证。

**📈 对比分析**

未进行实验或性能比较，文献主要通过理论推导和案例描述阐明安全需求与风险，缺乏量化的性能指标或对比结果。

**⚠️ 局限性**

局限性包括：1) 缺乏实测验证与量化风险评估；2) 未给出具体实现细节与数据集，导致难以复现；3) 只针对单一案例（LLP/HOD），未覆盖更广泛的 AI 组件或更高自动化等级；4) 主要依赖标准与理论分析，缺乏实际系统验证。

---

## 141. Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning

**arXiv ID:** 2602.05089 | [PDF](https://arxiv.org/pdf/2602.05089v1)

**作者:** Ethan Rathbun `[一作]` (Northeastern University), Christopher Amato `[通讯]` (Northeastern University)

**通讯引用:** 5725 | [OpenAlex ID](https://openalex.org/A5033129735)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在强化学习训练过程中，通过对仿真器动态行为进行细微改动，植入了能够在触发特定状态时执行预设动作的后门；

**💡 创新点**

提出了完全不需要访问或修改奖励信号的无奖励后门攻击方法 Daze，理论证明其既能实现攻击成功，又能保持与正常策略不可区分；

**🔧 技术方法**

利用无奖励 MDP 设计、理论分析和仿真器包装器实现，具体实现为在仿真器内部根据触发/迷惑函数修改状态转移；

**📊 数据集**

在 MuJoCo 连续控制任务、Atari 离散控制任务以及基于 Turtlebot/Fetch 的真实机器人实验中进行评估；

**📈 对比分析**

与 TrojDRL、SleeperNets、Q-Incept 等现有后门攻击比较，连续域中 Daze 的攻击成功率超过 92% 且几乎不影响正常回报；离散域中与 Q-Incept 的攻击成功率相当，但在平均回报（BR）上表现更优；

**⚠️ 局限性**

依赖于“随机动作非最优”这一假设（Assumption 1），若该假设不成立则攻击效果可能下降，同时需要手工设计触发器和迷惑函数，增加了攻击的复杂度与可检测性。

---

## 142. Improving Set Function Approximation with Quasi-Arithmetic Neural Networks

**arXiv ID:** 2602.04941 | [PDF](https://arxiv.org/pdf/2602.04941v1)

**作者:** Tomas Tokar `[一作]` (University of Toronto), Scott Sanner `[通讯]` (University of Toronto)

**通讯引用:** 6450 | [OpenAlex ID](https://openalex.org/A5028174137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出可学习的Kolmogorov均值，并基于此构建 QUANNs 模型，用于集函数学习

**💡 创新点**

创新点在于首次将 Kolmogorov 平均通过可逆神经网络实现为可学习聚合函数，并证明其在多种分解形式下的近似能力和结构化潜在空间优势

**🔧 技术方法**

利用可逆神经网络实现可学习 Kolmogorov 平均，结合编码器、估计器构建 QUANNs，并进行理论分析和实验验证

**📊 数据集**

在 MNIST‑sets、Omniglot‑sets、ModelNet40、QM9 等公开数据集上进行实验

**📈 对比分析**

与 DeepSets、PointNet、HPDS、SetTransformer、SlotAtt、FSPool、LAF 等基线比较，QUANNs 在多任务上显著提升精度或降低 MSE，并在转移学习中保持更好性能

**⚠️ 局限性**

对 sum‑decomposable 集函数近似能力有限，在样本稀缺时易过拟合，且目前仅针对可数集合实验，未扩展到连续集合

---

## 143. Bayesian Neighborhood Adaptation for Graph Neural Networks

**arXiv ID:** 2602.05358 | [PDF](https://arxiv.org/pdf/2602.05358v1)

**作者:** Paribesh Regmi `[一作]` (Rochester Institute of Technology), Kishan K C `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于贝叶斯非参数框架的邻域尺度自适应方法（BNA），通过把每个跳数的贡献概率建模为beta过程，并在训练时与节点表示同时学习，从而自动推断每个节点在信息聚合时应使用的有效邻域范围。

**💡 创新点**

创新点在于：①首次将beta过程用于描述GNN的跳数分布，实现对邻域尺度的非参数化自适应推断；②在保持GNN原有结构的同时，引入可梯度化的Concrete Bernoulli采样，允许在变分推断中同时更新邻域尺度与网络参数；③理论证明该方法可提升GNN的表达能力，并通过实验验证其在深层网络中的稳定性与不确定性校准优势。

**🔧 技术方法**

核心技术包括：贝叶斯非参数贝塔过程与其stick-breaking构造；伯努利过程与Concrete Bernoulli连续化；随机变分推断（SVI）与KL正则化；以及多种GNN骨干网络（GCN、ResGCN、GAT、JKNet、GCNII、ACM-GCN+）。

**📊 数据集**

实验使用的主要数据集有：同质化图（Cora、Citeseer、Pubmed）与异质化图（Chameleon、Cornell、Texas、Wisconsin）进行节点分类；大规模图（Flickr、ogb-arxiv、ogb-proteins）进行多/二分类；蛋白质互作网络（PPI）进行链路预测。

**📈 对比分析**

通过将BNA与各基线GNN结合，实验结果显示在大多数数据集上准确率提升或保持最优，尤其在深层网络中表现更稳定；在大规模数据集上也能保持竞争力；不确定性校准方面，ECE显著下降；相较于基线和模型集成，BNA在性能与校准度上均有优势。

**⚠️ 局限性**

局限性包括：需要设置截断层数T，且变分推断需要多次采样导致计算开销略增；对极大图的稀疏性处理仍有改进空间；目前仅验证了传统GNN骨干，尚未在图变换器等更复杂架构上深入评估。

---

## 144. ORACL: Optimized Reasoning for Autoscaling via Chain of Thought with LLMs for Microservices

**arXiv ID:** 2602.05292 | [PDF](https://arxiv.org/pdf/2602.05292v1)

**作者:** Haoyu Bai `[一作]` (University of Melbourne), Rajkumar Buyya `[通讯]` (University of Melbourne)

**通讯引用:** 105797 | [OpenAlex ID](https://openalex.org/A5014716105)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于大语言模型（LLM）的统一框架 ORACL，利用链式推理（CoT）同时完成微服务根因分析（RCA）与资源自适应扩缩容。

**💡 创新点**

创新点包括：1）把系统状态转化为语义化自然语言提示，让 LLM 直接进行推理；2）在 CoT 过程中嵌入根因诊断与资源决策，形成闭环；3）通过监督 + 强化学习（GRPO/GSPO）对 CoT 进行细粒度优化；4）构建 80 GB 规模的微服务事件数据集，支持跨部署泛化。

**🔧 技术方法**

技术栈主要包括：LLM（基础模型 + instruction‑tuned token）、Prompt 聚合、强化学习与监督微调、结构化输出模板、Prometheus/Jaeger 监控、Kubernetes API、DeepSpeed ZeRO‑3 训练、Unsloth。

**📊 数据集**

数据集：由公开微服务工作负载（Sock‑Shop、Train‑Ticket、Hotel Reservation 等）收集的指标、日志、追踪，共计 80 GB；此外使用 Murphy、ExplainIT、NetMedic 等基准数据验证 RCA。

**📈 对比分析**

与传统规则/深度学习控制器（KuScal、CoScale、FIRM）以及四个主流 RCA 系统（ExplainIT、NetMedic、Sage、Murphy）对比，ORACL 在根因识别上精度 92%，在 Sock‑Shop 的吞吐量比基线高 5%‑20%，延迟比基线低 20%‑40%；在 Train‑Ticket 的全训练对比中，吞吐量仅比 FIRM 小 2%，延迟仅比 FIRM 高 10%，但相比 Kubernetes 原生扩缩容显著提升。

**⚠️ 局限性**

局限性：1）推理成本高，推理时延不稳定；2）训练数据覆盖面有限，缺乏多样化云环境与能耗场景；3）推理粒度主要在服务层，缺乏容器/节点级细粒度诊断；4）对极端工作负载峰值的鲁棒性仍待提升。

---

## 145. Balanced Anomaly-guided Ego-graph Diffusion Model for Inductive Graph Anomaly Detection

**arXiv ID:** 2602.05232 | [PDF](https://arxiv.org/pdf/2602.05232v1)

**作者:** Chunyu Wei `[一作]` (Renmin University of China), Fei Wang `[通讯]` (Cornell University)

**通讯引用:** 21679 | [OpenAlex ID](https://openalex.org/A5100455750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于离散自我图扩散模型的 BAED 框架，用来在诱导式图异常检测中同时解决动态图结构与类别不平衡问题。

**💡 创新点**

创新点包括：① 用离散自我图扩散模型生成与真实异常分布对齐的自我图；② 通过 GIN 编码的异常引导嵌入实现对特定异常模式的精准生成；③ 采用基于损失的课程式异常增强策略，使得生成样本随训练动态调整，提升模型的泛化与鲁棒性。

**🔧 技术方法**

使用的技术包括：图神经网络（GCN、GraphSAGE、BWGNN、BernNet）、离散扩散模型、GIN 编码器、课程学习权重机制、诱导式训练与增量学习框架。

**📊 数据集**

实验数据集涵盖五个大规模真实图：Elliptic（比特币交易）、Reddit（社交互动）、T-Finance（金融交易）、Photo（共购网络）和 DGraph（金融网络），并在动态图上进一步验证。

**📈 对比分析**

与 AEGIS、GGAD、CGenGA 等生成式方法以及四种基线 GNN 进行比较。BAED 在 AUROC、AUPRC、F1 上多项指标显著优于对手，尤其在稀疏和动态场景下提升幅度更大（如 T-Finance 上 BWGNN 的 AUROC 提升近 90%）。在动态图实验中，BAED 的性能波动最小，显示出更好的适应性。

**⚠️ 局限性**

局限性：主要聚焦结构异常，缺乏对特征异常的处理；目前只在极端类别不平衡的异常检测任务上验证，尚未扩展到其他类型的不平衡或稀缺事件预测场景。

---

## 146. ARGaze: Autoregressive Transformers for Online Egocentric Gaze Estimation

**arXiv ID:** 2602.05132 | [PDF](https://arxiv.org/pdf/2602.05132v1)

**作者:** Jia Li `[一作]` (University of Texas at Dallas), Yapeng Tian `[通讯]` (University of Texas at Dallas)

**通讯引用:** 10680 | [OpenAlex ID](https://openalex.org/A5101835756)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出ARGaze框架，将第一人称注视估计重新表述为严格因果的自回归序列预测，实现实时在线估计。

**💡 创新点**

创新点在于：①使用热图token化与有限长度的历史窗口；②加入跟踪意识模板提供局部视觉先验；③通过Transformer decoder实现跨注意力的自回归推理，保证因果性、固定内存且提升时间一致性与鲁棒性。

**🔧 技术方法**

核心技术包括DINOv3视觉编码、Transformer自回归解码器、热图token化、跨注意力、动态模板裁剪、KL损失与scheduled sampling。

**📊 数据集**

使用EGTEA Gaze+、Ego4D（Gaze子集）和EgoExo4D三个基准，并在EgoExo4D构造OOD（场景/任务/参与者）测试。

**📈 对比分析**

与AT、EgoM2P、GLC等基线对比，ARGaze在三大基准上取得最高F1（EGTEA 44.48% vs GLC 41.57%）、精度提升1.82×推理速度、显著降低3.8×显存，并在OOD设置下保持强健性能。

**⚠️ 局限性**

局限性：对固定窗口长度的依赖在极慢或突变场景中可能欠佳；模板裁剪尺度需手动调优；仍受手部遮挡影响，未在所有边缘设备上进行完整评估。

---

## 147. Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks

**arXiv ID:** 2602.05066 | [PDF](https://arxiv.org/pdf/2602.05066v1)

**作者:** Jafar Isbarov `[一作]` (Virginia Tech), Murat Kantarcioglu `[通讯]` (Virginia Tech)

**通讯引用:** 12164 | [OpenAlex ID](https://openalex.org/A5087192873)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“Agent‑as‑a‑Proxy”攻击框架和Parallel‑GCG优化算法，揭示监控式防御（尤其是混合CoT+工具监控）对间接提示注入的脆弱性。

**💡 创新点**

1) 通过让代理本身重复攻击字符串来“递送”攻击到监控模型；2) 开发Parallel‑GCG，针对在多处上下文中重复出现的攻击字符串进行梯度优化；3) 证明即便监控模型规模与代理相当，攻击仍可成功，挑战可扩展监督假设。

**🔧 技术方法**

Greedy Coordinate Gradient (GCG)、多目标GCG、Parallel‑GCG；使用LLM监控（AlignmentCheck、PromptGuard 2+AlignmentCheck、Extract‑and‑Evaluate）与代理模型（GPT‑4o mini、Llama‑3.1‑70B、Mistral‑7B等）。

**📊 数据集**

AgentDojo benchmark（Slack、Banking等任务套件），对20×5等注入任务进行实验。

**📈 对比分析**

对比静态注入与自适应优化攻击，发现自适应攻击在所有监控方案下ASR均超过90%，而原始监控对静态攻击几乎无效；同时Hybrid监控虽对静态攻击更稳健，却在自适应攻击下更易突破。

**⚠️ 局限性**

攻击字符串高困惑度、易被检测；实验仅针对白盒监控，黑盒情况待研究；依赖代理能精确重复字符串，若代理生成错误或模糊则攻击效果下降。

---

## 148. RoboPaint: From Human Demonstration to Any Robot and Any View

**arXiv ID:** 2602.05325 | [PDF](https://arxiv.org/pdf/2602.05325v1)

**作者:** Jiacheng Fan `[一作]` (Paxini Tech), Zhengxue Cheng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5071945287)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了从人类演示到机器人执行的 Real‑Sim‑Real 数据管线，收集同步的 RGB、RGB‑D、关节、触觉信息，并通过 Dex‑Tactile 复位和 3D Gaussian Splatting 生成可直接训练 VLA 模型的多模态数据集。

**💡 创新点**

创新点在于：① 采用仪器化手套实现高精度视觉‑关节‑触觉同步采集；② 提出 Dex‑Tactile 联合优化的跨身体映射方法，兼顾 kinematic 与触觉一致性；③ 通过 3D Gaussian Splatting 实现真实场景的光照与几何对齐；④ 将上述技术整合为可批量生成高质量训练数据的完整流程。

**🔧 技术方法**

使用了多摄像头同步 RGB‑D 采集、磁编码器+霍尔效应触觉传感器、ArUco 姿态估计、FoundationPose 目标姿态推算、Dex‑Tactile 复位算法、3D Gaussian Splatting、Isaac Sim 物理渲染、路径跟踪渲染、以及 VLA 模型训练框架（Pi0.5、DP）。

**📊 数据集**

自建的 Human‑Dex 数据集（包含 100+ 任务示例，11 视角 RGB、3 RGB‑D、15 触觉、29 关节）以及对照的 Teleoperation 真实机器人演示数据。

**📈 对比分析**

与传统 Teleoperation 数据集进行对比，Real‑Sim‑Real 训练的 VLA 策略在 pick‑place、push、pour 等任务上平均成功率为 80%（仅比 Teleoperation 低 20%），在真实机器人实验中平均成功率 84%；通过模拟评估，触觉对齐误差约 3.86 mm。

**⚠️ 局限性**

局限性包括：仍受人机结构差异导致的映射误差（如触觉对齐误差 3.86 mm）；对复杂形状或高动态物体的鲁棒性不足；需要昂贵的多摄像头和仪器化手套，部署成本高；对场景重建的 3DGS 需要专业标定。

---

## 149. Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Variation in RLVR

**arXiv ID:** 2602.05261 | [PDF](https://arxiv.org/pdf/2602.05261v1)

**作者:** Fanfan Liu `[一作]` (Meituan), Haibo Qiu `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的强化学习算法LUSPO，用于消除RLVR中的响应长度偏差并提升大语言模型的推理性能

**💡 创新点**

通过在GSPO目标函数中加入序列长度缩放因子，显著缓解响应长度崩溃问题并提升模型推理深度

**🔧 技术方法**

改进的RLVR优化框架、序列级重要性权重、长度无偏序列策略优化（LUSPO）与现有GRPO、GSPO对比实验

**📊 数据集**

多样化数学与多模态数据集，包括DAPO‑MATH‑17K、ViRL39K、AIME24/25、MathVista、MathVision、MathVerse等

**📈 对比分析**

在dense和MoE模型（Qwen2.5‑7B‑Base、Qwen3‑30B‑A3B‑Instruct）以及多模态模型（Qwen2.5‑VL‑7B‑Instruct）上进行基准测试，LUSPO在大多数任务上比GRPO、GSPO提升约2–7 %准确率，平均提升约4–5 %

**⚠️ 局限性**

仍依赖手工调参的长度奖励、在更大规模模型或更复杂任务上需要进一步验证，且对不同长度奖励设置的敏感性尚未充分探究

---

## 150. CVA6-CFI: A First Glance at RISC-V Control-Flow Integrity Extensions

**arXiv ID:** 2602.04991 | [PDF](https://arxiv.org/pdf/2602.04991v1)

**作者:** Simone Manoni `[一作]` (University of Bologna), Andrea Bartolini `[通讯]` (University of Bologna)

**通讯引用:** 3167 | [OpenAlex ID](https://openalex.org/A5047906923)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 CVA6 核心中实现并评估了 RISC-V 标准的控制流完整性 (CFI) 扩展，构建了阴影栈单元和着陆点单元，并将完整实现开源。

**💡 创新点**

首次提供了 RISC-V CFI 扩展的硬件实现，面积开销仅 1.0%，性能损失最大 15.6%，且实现与先前硬件 CFI 方法相比面积开销最低，具备开源参考价值。

**🔧 技术方法**

使用硬件设计技术构建阴影栈单元 (SSU) 与着陆点单元 (LPU)，将其集成进 CVA6 的流水线；在 Synopsys Design Compiler 下以 22 nm FDX 工艺综合；通过 QEMU 的周期级模拟结合 SiFive GCC 13.3 CFI-aware 编译器进行性能与代码尺寸评估。

**📊 数据集**

使用 MiBench 汽车子集作为评测数据集。

**📈 对比分析**

将未加 CFI 的 CVA6 与加 CFI 的 CVA6-CFI 在面积、指令计数、执行周期等方面进行对比；面积差异约 1%，代码尺寸增加约 7.6–9.2%（约 22–23 kB），执行周期最多增加 15.6%。与其他硬件 CFI 方案相比，面积占优。

**⚠️ 局限性**

仅在顺序单发式核心上验证，未在乱序或多线程架构上测试；未做硅级验证或功耗测量；因缺乏稳定的 CFI 支持 Linux，所有基准均在 QEMU 用户态执行，可能忽略内核级交互；评测仅覆盖 MiBench 子集，未探测更广泛的应用场景。

---

## 151. Are Open-Weight LLMs Ready for Social Media Moderation? A Comparative Study on Bluesky

**arXiv ID:** 2602.05189 | [PDF](https://arxiv.org/pdf/2602.05189v1)

**作者:** Hsuan-Yu Chou `[一作]` (Duke University), Xiaowei Yang `[通讯]` (Duke University)

**通讯引用:** 5475 | [OpenAlex ID](https://openalex.org/A5077188576)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比评估了七个最先进的大语言模型（LLMs）在Bluesky社交媒体平台上进行有害内容检测的零样本表现，使用Bluesky Moderation Service（BMS）官方标签和人工注释作为基准。

**💡 创新点**

首次系统展示了开放权重、具备推理能力的LLMs在无需微调的情况下，能与主流专有LLMs匹配或超越的准确率；同时提出了针对不同有害属性（粗鲁、歧视、威胁）的敏感度/特异度差异，并强调了属性级别的标准化需求。

**🔧 技术方法**

采用推理式prompt（Chain‑of‑Thought）进行二分类判断；使用Azure、Google、Amazon Bedrock等云服务以及本地vLLM部署；利用Rand Accuracy、敏感度/特异度等统计指标进行模型对比。

**📊 数据集**

基于Bluesky公开数据，收集约430万条英文根帖；使用BMS发布的粗鲁、歧视、威胁三类标签；并由两位作者独立标注520条粗鲁贴与786条无标签贴作为人工参考集。

**📈 对比分析**

对比方法：将LLM输出与BMS和人工标签视为真实值，计算敏感度、特异度以及整体准确率。结果显示，开放权重LLMs的敏感度范围81%–97%、特异度91%–100%与专有LLMs（72%–98%、93%–99%）高度重叠，整体准确率可达84%–98%，与人类标注者的一致性相当。

**⚠️ 局限性**

研究仅覆盖英文根帖的文本内容，样本量相对总数据量极小；未涉及多模态（图像、音频、视频）和非英文语言；未探讨上下文化或链式回复中的复杂语境；因此结论对更广泛场景的推广仍需进一步验证。

---

## 152. VERA-MH: Reliability and Validity of an Open-Source AI Safety Evaluation in Mental Health

**arXiv ID:** 2602.05088 | [PDF](https://arxiv.org/pdf/2602.05088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 153. Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs

**arXiv ID:** 2602.05191 | [PDF](https://arxiv.org/pdf/2602.05191v1)

**作者:** Wentao Ni `[一作]` (University of California San Diego), Jishen Zhao `[通讯]` (University of California San Diego)

**通讯引用:** 5137 | [OpenAlex ID](https://openalex.org/A5077387335)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双层 Top‑p 稀疏注意力框架（Double‑P），在长上下文 LLM 推理中通过先在聚类层估计注意力质量，再在词元层动态分配计算，从而实现高效稀疏注意力。

**💡 创新点**

创新点在于：① 引入聚类层的大小加权质心近似，用低成本估计聚类注意力质量；② 在此基础上做第二层词元级 Top‑p 细化，仅对高影响聚类执行完整词元注意力；③ 三阶段（估计、选择、稀疏计算）协同优化，消除固定预算瓶颈并保证注意力质量。

**🔧 技术方法**

采用 k‑means 聚类、大小加权质心近似、两阶段 Top‑p 选择、混合精确词元与聚类质心的注意力计算、GPU 高效 Kernel（Top‑p、token/cluster 收集、FlashAttention 混合）等技术。

**📊 数据集**

在 LLaMA‑3.1‑8B、Qwen‑3‑8B 等长上下文模型上，使用 RULER（13 任务，4K–128K token）和 LongBench（6 类，5K–15K token）进行评估。

**📈 对比分析**

与 Quest、RetroInfer、Twilight 等基线相比，Double‑P 在保持近乎零准确率损失的前提下，注意力层加速 1.3–1.8×，解码速度提升 1.1–1.3×，相较全注意力可达 2.2× 加速，平均准确率提升 1.2–1.8个百分点。

**⚠️ 局限性**

局限性包括：需要先行聚类导致预处理开销；Top‑p 阈值需手动调优，可能不适用于所有模型/硬件；对极端分布或非常短上下文时的效果尚未充分验证。

---

## 154. Hallucination-Resistant Security Planning with a Large Language Model

**arXiv ID:** 2602.05279 | [PDF](https://arxiv.org/pdf/2602.05279v1)

**作者:** Kim Hammar `[一作]` (University of Melbourne), Emil Lupu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在安全规划中使用大型语言模型的迭代框架，结合一致性检测和基于反馈的上下文学习，自动生成并筛选安全响应动作。

**💡 创新点**

创新点在于：①通过一致性阈值实现可调控的幻觉风险；②给出一致性检测与推理过程的理论上限与后悔上界；③将 Lookahead、ICL 与自我拒绝机制融合到安全决策支持系统。

**🔧 技术方法**

使用的核心技术包括：大型语言模型（deepseek‑r1‑14b）生成候选动作；一致性函数 λ 评估预测的一致性；看ahead 预测未来执行时间；自我拒绝阈值 γ 控制幻觉概率；基于外部反馈（数字孪生或专家）进行上下文学习；理论分析中采用贝叶斯学习与多臂赌博机（Thompson Sampling）框架推导后悔上界。

**📊 数据集**

实验数据集：CTU‑Malware‑2014、CIC‑IDS‑2017、AIT‑IDS‑V2‑2022、CSLE‑IDS‑2024 四个公开的网络攻击与日志数据集。

**📈 对比分析**

与三大前沿 LLM（deepseek‑r1、gemini 2.5 pro、openai o3）进行对比，框架在平均恢复时间上缩短约 30%，幻觉率从 6% 降至 2%，并在不同数据集上保持稳健性能。

**⚠️ 局限性**

局限性包括：需要外部反馈或数字孪生来修正幻觉，导致额外的计算和人力成本；一致性阈值需要经验性调优；假设候选动作的分布与贝叶斯后验一致，实际情况可能不满足；仅在基于文本日志的攻击场景中验证，泛化至更复杂或多模态安全任务仍待探索。

---

## 155. VR Calm Plus: Coupling a Squeezable Tangible Interaction with Immersive VR for Stress Regulation

**arXiv ID:** 2602.05093 | [PDF](https://arxiv.org/pdf/2602.05093v1)

**作者:** He Zhang `[一作]` (Pennsylvania State University), Xinyi Fu `[通讯]` (Tsinghua University)

**通讯引用:** 22566 | [OpenAlex ID](https://openalex.org/A5063671472)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了一套名为VR Calm Plus的沉浸式VR情绪调节系统，该系统通过软体可挤压玩具与VR环境的实时映射实现身体触感与视觉音频的交互。

**💡 创新点**

创新点在于将可挤压的物理输入设备与VR交互深度融合，创造“主动放松”体验；通过身体压力即时驱动虚拟场景变化，提升情绪调节效果。

**🔧 技术方法**

技术实现包括：FSR压力传感器+Arduino+Unity实时映射、Meta Quest 3 VR头显、无线生理监测腕带（心率、皮肤电、脉搏变异性）以及自定义体验问卷与PANAS‑X情绪量表。

**📊 数据集**

数据来源为40名受试者（22F/18M）在实验室完成的问卷与生理记录数据，没有使用公开数据集。

**📈 对比分析**

方法采用跨实验交叉设计与配对t检验比较挤压交互与单纯视听两种条件，结果显示：挤压条件在正向情绪（宁静、专注）提升、心率下降、皮肤电升高、脉搏变异性保持方面均显著优于对照；效应量为中等至大。

**⚠️ 局限性**

局限性：样本来自单一文化背景、年龄层偏年轻、实验环境为实验室，缺乏跨文化与长期实测；可挤压玩具的长期耐用性与舒适度未充分验证。

---

## 156. CLEAR-HPV: Interpretable Concept Discovery for HPV-Associated Morphology in Whole-Slide Histology

**arXiv ID:** 2602.05126 | [PDF](https://arxiv.org/pdf/2602.05126v1)

**作者:** Weiyi Qin `[一作]` (Rutgers University), Hao Wang `[通讯]` (Rutgers University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种后置可解释性框架 CLEAR-HPV，利用注意力引导的多实例学习潜在空间重构，自动发现 HPV 相关的病理概念，并通过概念分数向量恢复模型预测；

**💡 创新点**

创新点在于：①在注意力结构的潜在空间上实现无监督概念发现，自动生成可解释的概念分数向量；②框架与多种 MIL 骨干兼容，能跨不同数据集保持概念稳定与可迁移；

**🔧 技术方法**

技术手段包括预训练的 ViT/UNI 编码器、注意力多实例学习（CLAM、ABMIL、TransMIL、MHMIL）、加权 K‑means 聚类实现概念发现，以及构造概念分数向量与空间概念图的后处理；

**📊 数据集**

使用了三大独立数据集：TCGA‑HNSCC、TCGA‑CESC 与 CPTAC‑HNSCC 的全切片图像；

**📈 对比分析**

与基线 MIL、热图、Dirichlet、编码器概念等方法对比，CLEAR‑HPV 在 10 维概念空间上能够恢复与原模型相近或略优的 ACC、AUC，且在跨队列零样本迁移中保持高精度，可用于生存预测；

**⚠️ 局限性**

局限性包括：①对骨干模型注意力质量敏感；②概念聚类未使用专家标注，细粒度区分仍有限；③依赖预训练编码器与固定注意力，缺乏端到端的联合优化。

---

## 157. MobileManiBench: Simplifying Model Verification for Mobile Manipulation

**arXiv ID:** 2602.05233 | [PDF](https://arxiv.org/pdf/2602.05233v1)

**作者:** Wenbo Wang `[一作]` (University of Sydney), Baining Guo `[通讯]` (Microsoft Research Asia)

**通讯引用:** 46858 | [OpenAlex ID](https://openalex.org/A5101666011)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MobileManiBench—a 规模化仿真基准，集成两款移动机器人（G1并行抓手与XHand灵巧手）、630个跨20类对象、5种移动操控技能（开/关/拉/推/拾）以及100+任务，自动化生成300K轨迹并配备语言指令、多视角RGB‑Depth‑Seg图像与同步状态/动作数据；

**💡 创新点**

创新点在于：1）利用仿真-强化学习管线实现无人工标注的多机器人、多对象、多技能轨迹生成；2）构建统一的数据集与基准，支持对VLA模型在移动平台上的可扩展验证；3）引入多模态输入与扩散Transformer的动作预测，提升对未知场景与对象的泛化；

**🔧 技术方法**

采用了NVIDIA Isaac Sim仿真平台、基于状态的强化学习（MobileManiRL）、SigLIP视觉编码、PaliGemma‑2与Gemma‑2的视觉‑语言融合、扩散Transformer（DiT）实现动作序列预测，以及端到端的MSE损失训练；

**📊 数据集**

使用了自构造的MobileManiDataset，包含630个对象（PartNet‑Mobility、UniDoorManip、YCB等），100个现实场景与100K+轨迹，并与公开数据集如Open‑X‑Embodiment、RLBench等作对比；

**📈 对比分析**

与现有VLA模型（OpenVLA、CogACT、π_0、π_0.5）在G1机器人上对比，MobileManiVLA在未见对象与场景上的成功率为28.2%，显著高于4.5%–6.8%；在RL对比中，MobileManiRL在已知组合上可达约90%成功率，说明数据与模型的有效性；

**⚠️ 局限性**

局限性包括：1）VLA模型对未知对象/场景的泛化仍有限；2）对仿真环境的依赖可能导致现实落地时的鲁棒性不足；3）仅覆盖两款机器人，未覆盖更广泛的移动平台与抓手；4）轨迹生成与模型训练耗时较长，需大规模GPU资源；

---

## 158. A$^2$-LLM: An End-to-end Conversational Audio Avatar Large Language Model

**arXiv ID:** 2602.04913 | [PDF](https://arxiv.org/pdf/2602.04913v1)

**作者:** Xiaolin Hu `[一作]` (Beijing University of Posts and Telecommunications), Kai Chen `[通讯]` (Zhongguancun Institute of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出A2-LLM，一种端到端的大型多模态模型，能够同时生成文本、音频和3D面部动画，构建统一的语言、音频与表情生成框架。

**💡 创新点**

创新点包括：① 通过离散化面部运动（RVQ‑VAE）和Motion Connector，将语义与音频隐藏状态直接映射到面部动作，消除传统串行管线的语义‑情感鸿沟；② 构建FLAME‑QA大规模QA式多模态数据集，提供严格的语义监督；③ 采用三阶段训练策略（Connector预训练、LoRA重置、情感指令微调）实现高效、情感丰富的面部表达。

**🔧 技术方法**

技术方案：基于Step‑Audio‑2‑mini（Qwen2.5‑7B）LALM；使用FLAME 3DMM参数；RVQ‑VAE对面部动态进行残差量化；Motion Connector采用跨注意力与历史条件；低秩适配器LoRA与分阶段学习率；联合交叉熵损失结合语音、文本与运动三模态。

**📊 数据集**

主要使用的数据集为FLAME‑QA（约100k QA triplet），其中包含音频、文本、FLAME 3D参数；此外还采集了高动态子集（≈1k）用于情感训练；数据来源包括VoxCeleb、SMIRK、Whisper、IndexTTS2、InfiniteTalk等。

**📈 对比分析**

与传统ASR→LLM→TTS→动画串行管线对比，A2-LLM在TTFA仅535 ms，RTF 0.7；在OpenVoiceBench语言性能与主流音频模型持平；在面部表情评估（MOD、UFD、Temporal Correlation、Liveliness）均优于ARTalk、CodeTalker等基线；用户偏好实验显示A2-LLM在表达性方面获胜率超过70%。

**⚠️ 局限性**

局限性：目前仅支持英文；面部动画为主，缺乏全身姿态生成；多模态训练对资源需求高，且在极端情感场景下仍有一定误差；未来需要扩展多语言、多模态（全身、手势）以及更大规模的情感对话数据。

---

## 159. Automatic Cognitive Task Generation for In-Situ Evaluation of Embodied Agents

**arXiv ID:** 2602.05249 | [PDF](https://arxiv.org/pdf/2602.05249v1)

**作者:** Xinyi He `[一作]` (Peking University), Yujia Peng `[通讯]` (State Key Laboratory of General Artificial Intelligence, BIGAI, Beijing, China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在未见的3D场景中，使用动态交互‑演化两阶段框架（TEA）自动生成大量任务，并构建评测基准。

**💡 创新点**

创新点包括：①以图结构形式定义任务并在交互与演化阶段自动生成；②在交互阶段实现“agent‑in‑loop”自生成任务；③通过任务重用与重组实现无需外部资源的任务演化；④提出MIR、MIR‑e等多维度评估指标。

**🔧 技术方法**

技术手段：Unreal Engine仿真、视觉‑语言模型（VLM）生成语义任务、规则式任务生成、谱聚类与多模态嵌入相似度计算、任务过滤、ε‑随机游走、图结构重用与重组。

**📊 数据集**

数据集：10个未见真实扫描场景，生成87,876个任务，抽样848个作为测试集，并进行10%人工验证；另收集100个导航任务。

**📈 对比分析**

对比方法：在生成的任务集上评测多款SOTA VLM（GPT‑5、GPT‑4o、GPT‑4.1、Gemini‑2.5‑Pro/Flash、Claude‑Opus‑4、llama‑4‑Scout、o3‑2025）与7名人类参与者；模型在基本感知任务、3D交互任务上表现显著低于人类，MIR指标显示生成任务多样性提升；导航任务中模型成功率低，展示缺乏3D意识。

**⚠️ 局限性**

局限性：仅在UE模拟环境中评估；机器人交互仍受限，未涉及真实硬件；任务覆盖受交互策略影响；模型性能仍受训练分布偏差限制；需要进一步验证跨场景泛化。

---

## 160. First Proof

**arXiv ID:** 2602.05192 | [PDF](https://arxiv.org/pdf/2602.05192v1)

**作者:** Mohammed Abouzaid `[一作]` (Stanford University), Lauren Williams `[通讯]` (Harvard University)

**通讯引用:** 3804 | [OpenAlex ID](https://openalex.org/A5082619822)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公布了十个研究水平的数学问题，用于评估AI系统解答高难度数学问题的能力。

**💡 创新点**

创新点在于采用真实未公开的研究问题作为基准，减少数据污染，并聚焦AI在专业数学研究中的实际表现。

**🔧 技术方法**

使用大型语言模型（GPT‑5.2 Pro、Gemini 3.0等）与手工验证的推理过程进行实验。

**📊 数据集**

数据集为作者自己生成的十个多领域数学问题，答案暂时保密，仅在实验中提供。

**📈 对比分析**

通过单次交互给模型答案并人工评估，发现目前最先进的模型在多数问题上仍无法给出完整证明，说明仍有显著差距。

**⚠️ 局限性**

局限在于问题数量有限，缺乏统一评分标准，且实验仅基于单次提问，无法体现交互式推理的潜力。

---

## 161. Informative Path Planning with Guaranteed Estimation Uncertainty

**arXiv ID:** 2602.05198 | [PDF](https://arxiv.org/pdf/2602.05198v1)

**作者:** Kalvik Jakkala `[一作]`, Srinivas Akella `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种可在有限距离和能量约束下，保证高斯过程后验方差低于用户设定阈值的环境监测路径规划方法；

**💡 创新点**

创新点在于将不确定性阈值约束转化为可离散化的覆盖问题，支持非平稳核和非凸障碍环境，并给出近似最优的 GreedyCover 与 GCBCover 两种规划器，后者同时兼顾采样点选择与路径长度；

**🔧 技术方法**

主要技术包括高斯过程建模与非平稳核学习、基于核阈值构造的二进制覆盖矩阵、子模函数贪婪选择与子模约束路由的通用成本收益算法；

**📊 数据集**

使用了 Shuttle Radar Topography Mission（SRTM）地形数据（47°N,124°W）以及现场湖泊深度测量（ASV/AUV）进行验证；

**📈 对比分析**

与基准方法 HexCover 对比，GreedyCover 与 GCBCover 在满足相同方差阈值的前提下，采样点数与路径长度均更短；GCBCover 在距离预算下仍能近似满足不确定性要求；实验显示运行时间较 HexCover 更快；

**⚠️ 局限性**

主要局限包括对高斯过程核的依赖（核设定不当会影响不确定性下界），只在离散评估集上给出保证，未考虑连续域；并未充分处理执行不确定性（定位误差、轨迹跟踪误差）。

---

## 162. Steering Externalities: Benign Activation Steering Unintentionally Increases Jailbreak Risk for Large Language Models

**arXiv ID:** 2602.04896 | [PDF](https://arxiv.org/pdf/2602.04896v1)

**作者:** Chen Xiong `[一作]` (Chinese University of Hong Kong), Tsung-Yi Ho `[通讯]` (IBM Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了激活Steering对大语言模型安全性的无意外影响，揭示即使是从纯善意数据学习的Steering向量也会削弱拒绝行为并显著提升越狱成功率。

**💡 创新点**

提出了“Steering Externalities”概念，系统展示了benign Steering通过改变早期token分布和隐藏表示削弱安全门槛的机制，并首次提出安全感知Steering‑BIND缓解策略。

**🔧 技术方法**

使用激活Steering技术、黑盒越狱方法（CoP、PAIR、TAP）、token‑级KL差异分析、t‑SNE+线性探测、HarmBench分类器等多种技术手段。

**📊 数据集**

利用Alpaca（benign prompts）、IFEval（JSON格式）、HarmBench（400 harmful queries）以及Mistral生成的响应集进行实验。

**📈 对比分析**

在Llama‑2/3、Gemma三大模型上对比原始与Steered模型的拒绝率、JSON有效率和攻击成功率（ASR），发现Steering显著降低拒绝率、提升ASR至近99%，验证了其对安全性的负面影响。

**⚠️ 局限性**

局限性包括仅评估了有限的Steering方向和越狱方法，缺乏对更广泛安全评估和在线动态调整Steering的研究，且未充分验证模型规模、训练数据多样性下的泛化能力。

---

## 163. Semantic Search over 9 Million Mathematical Theorems

**arXiv ID:** 2602.05216 | [PDF](https://arxiv.org/pdf/2602.05216v1)

**作者:** Luke Alexander `[一作]` (University of Washington), Vasily Ilin `[通讯]` (University of Washington)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5009909631)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个包含920万条数学定理的语义检索系统

**💡 创新点**

首次大规模集成化定理检索框架，并证明用LLM生成自然语言标语显著提升检索效果

**🔧 技术方法**

利用LLM（DeepSeek V3/Claude 4.5等）生成定理标语，再用Qwen3‑8B等嵌入模型和HNSW索引实现检索；并进行提示、上下文、重排序等系统性消融

**📊 数据集**

920万条从arXiv、Stacks Project、ProofWiki等公开源提取的定理，伴随元数据与原文链接

**📈 对比分析**

相较于Google/ArXiv检索、ChatGPT 5.2、Gemini 3 Pro，系统在主题检索上达到45.0% Hit@20（定理级）和56.8% Hit@20（论文级），显著优于基线

**⚠️ 局限性**

仍受限于检索对符号化定理的解释能力、LLM生成标语的可解释性、对多语言与非英文资源的覆盖不足以及高昂的计算与API成本

---

## 164. Temporal Pair Consistency for Variance-Reduced Flow Matching

**arXiv ID:** 2602.04908 | [PDF](https://arxiv.org/pdf/2602.04908v1)

**作者:** Chika Maduabuchi `[一作]` (William and Mary), Jindong Wang `[通讯]` (William and Mary)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Temporal Pair Consistency（TPC），在流匹配与 Rectified Flow 训练中通过在同一概率路径上耦合不同时间步的速度预测来降低梯度方差，从而提升采样质量与效率。

**💡 创新点**

创新点在于：① 在不改动模型架构、概率路径或求解器的前提下，通过配对时刻的约束实现轻量级方差削减；② 通过固定反向配对与可学习的单调配对函数提供可调节的时间一致性；③ 结合理论分析给出控制变差的证明和数值稳定性的提升。

**🔧 技术方法**

使用技术包括：连续时间流匹配（Flow Matching）、Rectified Flow、概率路径采样、抗噪声/噪声增强训练、Score‑based denoising、固定/学习单调配对函数、控制变差正则化、随机门控、以及自适应 ODE/RK45 求解器。

**📊 数据集**

实验数据集涵盖：CIFAR‑10；ImageNet 32×32、64×64、128×128（无条件）以及 ImageNet 64/128（条件）等，覆盖多分辨率与多任务场景。

**📈 对比分析**

与多种基线（DDPM、i‑DODE、ScoreFlow、FM‑OT、Rectified Flow 等）在 FID、NFE、IS、召回率等指标上对比，TPC 在相同或更低 NFE 下显著降低 FID，例如 CIFAR‑10 FID 从 6.35 降至 3.19，ImageNet 128×128 FID 从 20.9 降至 18.6，条件 ImageNet 128×128 FID 从 6.8 降至 4.9；一阶采样与全仿真场景均获提升。

**⚠️ 局限性**

局限性：仅在 128×128 以下的无条件图像生成上验证，未评估更高分辨率、条件任务或其他模态；过强的时间一致性约束可能限制模型表达；在某些超参数组合下仍需微调。

---

## 165. PLATO Hand: Shaping Contact Behavior with Fingernails for Precise Manipulation

**arXiv ID:** 2602.05156 | [PDF](https://arxiv.org/pdf/2602.05156v1)

**作者:** Dong Ho Kang `[一作]` (University of Texas at Austin), Luis Sentis `[通讯]` (University of Texas at Austin)

**通讯引用:** 4186 | [OpenAlex ID](https://openalex.org/A5079374033)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了一款采用混合指尖（硬质指甲+柔性指掌）和低阻尼五杆链条的主动手，具备高频力感知与精准抓取能力。

**💡 创新点**

将人类指甲的刚度结构引入机械手指尖，构建能在全局弯曲与局部压痕间分配能量的弹性能量分解模型，并结合自感动力学实现透明力控制。

**🔧 技术方法**

使用弹性能量分解的弯曲-压痕模型、QDD驱动、五杆传动、离散力感测与卡尔曼滤波等技术。

**📊 数据集**

通过在7-DoF机械臂上进行抓取、拉出、纹理频谱及动态冲击实验收集的数据，未使用公开数据集。

**📈 对比分析**

与仅柔性指掌的指尖在拉出实验、纹理频谱、以及八项触摸任务中对比，增设指甲后成功率从0提升至0.85–1.0，拉出力提升23–78%，力感知频率响应提升8–20 dB。

**⚠️ 局限性**

受限于传动齿比导致的反射惯性和摩擦/滞后，无法在更高DoF系统中保持低阻尼；对极细/曲面物体仍需更精细对齐，未探究多手指协同与动态大力场。

---

## 166. Aspect-Aware MOOC Recommendation in a Heterogeneous Network

**arXiv ID:** 2602.05297 | [PDF](https://arxiv.org/pdf/2602.05297v1)

**作者:** Seongyeub Chu `[一作]` (Korea Advanced Institute of Science and Technology), Mun Yong Yi `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 7565 | [OpenAlex ID](https://openalex.org/A5088156206)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 AMR（Aspect-aware MOOC Recommendation）框架，利用双向 walk 自动发现元路径，并将节点内容嵌入多方面表示，用于知识概念（KC）推荐。

**💡 创新点**

创新点在于：①不再依赖人工定义的元路径，而是通过双向 walk 自动挖掘；②在每条路径中对节点内容进行多方面（aspect）编码，生成细粒度的边特征；③将多方面表示作为图神经网络的边权重，实现更加丰富的关系表达。

**🔧 技术方法**

技术包括：FastText 词向量提取、双向 walk、Bi‑LSTM + 注意力获取路径表示、GNN（GCN 作为主干）聚合多方面表示、三元组损失 + BPR 损失联合训练，以及对重要性进行学习的 aspect importance 模块。

**📊 数据集**

实验数据集：MOOCCube（XuetangX 平台）和 PEEK（VideoLectures.Net）两大 MOOC 真实数据集。

**📈 对比分析**

与七种基线（Metapath2vec、ACKRec、MOOCIR、AMCGRec、PGPR、CAFE、UCPR）进行比较。AMR 在 HR@5/10/20、nDCG@5/10/20 上均击败所有基线，尤其在 MOOCCube 上提升明显；在不同 GNN 变体中，GCN 版本表现最佳；多方面数量越多性能越好。

**⚠️ 局限性**

局限性：①对路径密度敏感，Learner‑Learner 路径较稀疏时表现受限；②固定方面数量，缺乏自适应选择；③在路径稀疏或异构度低的子图中需要进一步改进。

---

## 167. Signal or 'Noise': Human Reactions to Robot Errors in the Wild

**arXiv ID:** 2602.05010 | [PDF](https://arxiv.org/pdf/2602.05010v1)

**作者:** Maia Stiber `[一作]` (Microsoft Research), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**通讯引用:** 3205 | [OpenAlex ID](https://openalex.org/A5017287995)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在公共空间部署了一台自主咖啡机机器人，收集了49名参与者在机器人错误和正常执行过程中的社会信号，并通过主题编码和指标分析研究人类对机器人错误的反应。

**💡 创新点**

首次系统性在野外环境下大规模考察非社交机器人错误对个人和群体交互的社会信号影响，发现信号既丰富又嘈杂；并提出“噪声”与“有用信息”共存的观点，提供了在真实环境中利用社会信号管理错误的启示。

**🔧 技术方法**

技术方案包括基于Kinova Gen3机械臂、ROS、三台Kinect摄像头+麦克风的感知栈；软件模块有Proximity Handler、Ordering Interface、Data Collection、Intrusion Detector、Robot Controller 以及可插拔的LLM交互；使用主题编码、行为计数、语音相关性评估等方法提取和量化社会信号。

**📊 数据集**

使用自建的咖啡机数据集：共39次完整交互（含13次群体交互、26次单人交互），49名参与者，211分钟的视频/音频，包含43个错误事件、13次连续错误以及多次重复交互。

**📈 对比分析**

对比方法主要是基于定性编码的指标比较：错误与非错误情境下社会信号出现率、反应时长、信息性发声比例等；结果显示约98%错误引发反应，平均反应时约19秒，错误中志愿信息比例达58%；群体交互比单人交互产生更多信息性发声但也伴随更高噪声。

**⚠️ 局限性**

局限性包括：样本多样性有限（主要是大学校园人员）、部署时间短、参与者对机器人好奇心随时间下降导致回访率低、噪声高导致自动检测困难、缺乏可重复的外部验证与模型泛化实验。

---

## 168. Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions

**arXiv ID:** 2602.05234 | [PDF](https://arxiv.org/pdf/2602.05234v1)

**作者:** Yuntai Bao `[一作]` (Zhejiang University), Jianwei Yin `[通讯]` (Zhejiang University)

**通讯引用:** 7185 | [OpenAlex ID](https://openalex.org/A5069353502)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于分布匹配的弱监督干预模型（Concept DAS），利用分布式互换干预（DII）在推理时对 LLM 内部表示进行可逆、可解释的调整；

**💡 创新点**

创新点包括：① 采用 JSD 分布匹配目标取代传统概率最大化/偏好优化，以弱监督方式学习干预向量；② 通过 DII 自动采样干预因子，消除手工调参，天然支持双向干预；③ 在大规模模型上表现更稳健，能在安全任务中实现无参调优的高效控制；

**🔧 技术方法**

核心技术包括：Jensen‑Shannon Divergence 分布匹配、分布式对齐搜索（DAS）框架、分布式互换干预（DII）策略、LLM 内部向量干预（加法/夹紧）、梯度优化训练以及多维评估指标（Steering Score、ASR、KL 散度）等；

**📊 数据集**

使用的数据集有：增强版 Concept500（覆盖 500 个概念）、Gemma‑2 benchmark、AlpacaEval、AdvBench、HarmBench、TDC2023、MaliciousInstruct、JailbreakBench、Red‑teaming 等，以及安全微调模型（Phi‑3.5‑mini、Llama‑3.1‑8B/70B）和后门 LLM；

**📈 对比分析**

方法与无监督差分平均、语言建模（Lang.）、偏好优化（RePS、BiPO）、提示 LoReFT 等基线在 Gemma‑2‑9B 上取得最高 Steering Score；在安全案例中，Concept DAS 在保持模型通用性能、低 KL 散度的前提下，能在无超参数调优的情况下有效抑制拒绝行为和中和链式思维后门，优于 PO 方法；

**⚠️ 局限性**

局限性包括：① 需要对比训练数据，对数据要求较高；② 在小模型或浅层干预时效果欠佳；③ 尽管无需手工调参，但在通用概念调优中仍需进行因子调优；④ 仅研究 rank‑1 干预向量，未探讨低秩或 LoRA/LoReFT 方案；⑤ 机制解释尚不完全基于因果理论。

---

## 169. Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning

**arXiv ID:** 2602.04998 | [PDF](https://arxiv.org/pdf/2602.04998v1)

**作者:** Yu-Ang Lee `[一作]` (National Taiwan University and Academia Sinica), Mi-Yen Yeh `[通讯]` (Academia Sinica)

**通讯引用:** 954 | [OpenAlex ID](https://openalex.org/A5013596889)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统重新评估了多种LoRA变体，在统一实验框架下进行大规模超参数搜索，发现不同方法的最佳学习率范围不同，但在各自最优设置下，性能基本相当。

**💡 创新点**

创新点在于：①指出在LoRA相关研究中，缺乏充分的学习率调优导致的误导性结论；②通过对比多种初始化和结构改进的LoRA方法，揭示它们在学习率上的差异；③利用Hessian最大特征值分析解释不同方法所需学习率差异。

**🔧 技术方法**

技术包括：LoRA基础方法及其代表性改进（PiSSA、MiLoRA、Init[AB]、DoRA）、多模型（Qwen3-0.6B、Gemma-3-1B、Llama-2-7B）训练，数学推理（MetaMathQA/GSM8K/MATH）和代码生成（CodeFeedback/HumanEval/MBPP）任务，系统化的学习率、批大小、rank 超参数搜索，以及对Hessian最大特征值的Lanczos算法估计。

**📊 数据集**

使用的数据集包括：MetaMathQA（训练），GSM8K 与 MATH（测试）用于数学推理；CodeFeedback（训练），HumanEval 与 MBPP（测试）用于代码生成；同时在不同模型规模上重复实验。

**📈 对比分析**

对比方法：Vanilla LoRA 与四个代表性变体；通过统一实验协议对所有方法进行批量、rank 与学习率的搜索。结果表明：在最优学习率下，各方法的峰值性能差距仅 0.52%（Gemma数学）、1.75%（Llama代码）以内；学习率选择是影响性能的关键因素，batch size 调优对性能提升影响较小。

**⚠️ 局限性**

局限性包括：仅评估了 decoder‑only LLM 规模至 7B，未覆盖更大模型或其他架构；只测试了数学推理与代码生成两个任务，缺乏多样化任务验证；对 DoRA 的 Hessian 分析未完成；实验资源受限，未能对所有模型与任务进行完整的 rank 维度搜索。

---

## 170. Unbiased Single-Queried Gradient for Combinatorial Objective

**arXiv ID:** 2602.05119 | [PDF](https://arxiv.org/pdf/2602.05119v1)

**作者:** Thanawat Sornwanee `[一作]` `[通讯]` (Stanford University), Thanawat Sornwanee (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为 Easy Stochastic Gradient (ESG) 的单次查询无偏梯度估计器，用于通过多项式扩展（Product‑Bernoulli 形式）来优化组合目标。

**💡 创新点**

创新点在于：① 通过构造可路径求导的值估计器，使梯度可直接通过自动微分获得；② 将传统 REINFORCE 与 ESG 通过重要性采样统一；③ 通过“good tuple” (f,σ,σ̂) 产生一族新的无偏梯度估计器，显著降低方差；④ 提出单次查询下降算法 (SQD) 并在实验中验证其优越性。

**🔧 技术方法**

主要技术包括：组合优化的多项式扩展、Bernoulli 随机变量的概率放缩、REINFORCE 与控制变量、重要性采样、自动微分、方差分析与调度学习率。

**📊 数据集**

使用合成数据集：对称切片（Symmetric‑Slice）问题，维度 d = 10、20、30，oracle Q 只依赖 Hamming 权重，构造了宽阔平坦区和稀疏峰值，作为实验基准。

**📈 对比分析**

与 REINFORCE、RELAX、ARM、DisARM 等无偏/低方差梯度估计器比较。实验显示：SQD+ESG 在达成全局最优方面比 REINFORCE 与 ARM/DisARM 更快、更稳定；RELAX 虽性能好但需要额外的控制变量学习；SQD 在单查询次数上保持优势，最终得到的目标值与最优解相同。

**⚠️ 局限性**

局限性包括：
- 对于 compactly‑supported 或降落风险率分布的“good tuple”不存在，导致 ESG 无法实现校准；
- 仍受限于 Product‑Bernoulli 结构，难以直接推广到更一般的组合约束；
- 方差虽降低但在某些极端点仍可能较高；
- 未提供自适应选择或动态调参机制，无法在不同任务中自动挑选最优 tuple。

---

## 171. Stochastic hierarchical data-driven optimization: application to plasma-surface kinetics

**arXiv ID:** 2602.04975 | [PDF](https://arxiv.org/pdf/2602.04975v1)

**作者:** José Afonso `[一作]` (Instituto de Plasmas e Fusão Nuclear), Pedro Viegas `[通讯]` (Instituto de Plasmas e Fusão Nuclear)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于Sloppy模型理论的随机层级优化框架，用于高效校准计算成本高、参数空间高维的物理模型；

**💡 创新点**

创新点在于将减缩Hessian近似作为线性自编码器，利用随机子空间快速捕捉“硬”参数方向；并将概率推导的损失函数与实验数据直接对应；

**🔧 技术方法**

技术包括：随机层级优化、Gauss–Newton Hessian、减缩Hessian（随机子空间）、概率最大似然损失、梯度自适应搜索；

**📊 数据集**

使用225条O₂/CO₂等离子体-玻璃表面交互实验数据（不同压强、电流、温度）训练模型；

**📈 对比分析**

与DE、CMA‑ES、TRF、Powell、GP等基线算法对比，结果显示层级方法在模拟调用次数（样本效率）上优于所有基线，尤其在前50次迭代内下降最快；

**⚠️ 局限性**

局限性：算法仍为局部搜索，无法保证全局最优；对极端稀疏或噪声实验数据的外推能力未充分验证；

---

## 172. TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation

**arXiv ID:** 2602.04929 | [PDF](https://arxiv.org/pdf/2602.04929v1)

**作者:** Junhan Kim `[一作]` (Samsung Research), Yongkweon Jeon `[通讯]` (Samsung Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后训练量化（PTQ）算法 TurboBoA，用于在不训练的情况下高效地将大语言模型（LLM）压缩到低位宽。

**💡 创新点**

创新点包括：①联合量化多条输出通道并给出闭式误差补偿公式，显著减少顺序量化瓶颈；②在误差补偿中加入前面已量化层产生的输入误差，降低误差累积；③自适应地在每一步量化前重新计算量化网格，并用坐标下降细化比例系数，保持权重与网格的一致性。

**🔧 技术方法**

主要技术手段包括无梯度量化、基于Hessian的误差补偿（Kronecker 结构Hessian）、闭式解法、坐标下降（CD）优化以及与变换方法（如 QuaRot、SpinQuant、OSTQuant）的结合。

**📊 数据集**

使用 WikiText‑2 2048 长度序列做校准数据，评测数据集包括 Wiki2、C4；在零样本推理任务上使用 8 个常识推理基准（ARC‑challenge、BoolQ、HellaSwag、LAMBADA、OpenbookQA、PIQA、WinoGrande）。

**📈 对比分析**

与 GPTQ、GPTAQ 以及多种变换压缩方法（OmniQuant、DuQuant、SpinQuant、OSTQuant）进行对比。TurboBoA 在 INT2/INT3 等低位宽场景下，Wiki2/C4 的困惑度（PPL）和零样本准确率均显著优于对手，速度提升超过三倍。

**⚠️ 局限性**

局限性包括：当并行量化通道数 N 超过 16 时速度提升趋于饱和；误差补偿需要额外的前向推理以获得前层输入误差；当前实现假设 H 的 Kronecker 结构，若 Hessian 近似不再满足该假设，则性能可能下降。

---

## 173. FATe of Bots: Ethical Considerations of Social Bot Detection

**arXiv ID:** 2602.05200 | [PDF](https://arxiv.org/pdf/2602.05200v1)

**作者:** Lynnette Hui Xian Ng `[一作]` (Carnegie Mellon University), Kathleen M. Carley `[通讯]` (Carnegie Mellon University)

**通讯引用:** 27999 | [OpenAlex ID](https://openalex.org/A5085927300)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过FATe框架评估社交媒体机器人检测系统的伦理问题，并结合文献综述、实验和用户案例分析提出改进建议。

**💡 创新点**

创新点在于将公平、责任、透明度三大伦理维度系统化应用于机器人检测，并提出多样化数据集、算法公平性校准和可解释性提升的研究方向。

**🔧 技术方法**

使用了文献回顾、机器学习算法性能评估、定性案例分析以及政策文本比较等技术。

**📊 数据集**

主要使用OSOME机器人数据集（19个人工标注子集）以及多平台收集的BotBuster训练数据。

**📈 对比分析**

与BotBuster在英语与非英语用户上的准确率对比发现英语用户识别率略高，整体精度超过96%；但对非英语仍有偏差。

**⚠️ 局限性**

局限性包括数据集主要来自X且以英语为主，缺乏跨语言、跨平台的代表性；算法解释性不足；对现实平台自动化机制的验证有限。

---

## 174. System-Level Isolation for Mixed-Criticality RISC-V SoCs: A "World" Reality Check

**arXiv ID:** 2602.05002 | [PDF](https://arxiv.org/pdf/2602.05002v1)

**作者:** Luis Cunha `[一作]`, Thomas Roecker `[通讯]` (Infineon AG)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文研究了RISC‑V SoC中混合关键性环境下的系统级隔离，比较并实现了IOPMP、标准World Checker（S‑WC）和改进版World Checker（M‑WC）。

**💡 创新点**

创新点在于对World Checker进行结构优化，引入起止地址匹配、通用读权限位以及显式的WID‑权限条目，从而显著提升可扩展性与面积效率。

**🔧 技术方法**

实现技术包括AXI通道检查器、LUT/FF资源计数与时延测量、以及基于CVA6的SoC硬件实现和FPGA验证。

**📊 数据集**

实验数据集来自于在CVA6基于SoC上运行的自制DMA引擎，覆盖多种规则配置与不同ID数的仿真/FPGA测量。

**📈 对比分析**

通过对比访问延迟、LUT/FF占用及面积占比，改进版World Checker在大ID数配置下面积减小约5%，延迟保持固定的2周期，而IOPMP的延迟为可变且更高。

**⚠️ 局限性**

局限性包括改进版在共享权限上略逊于原版，且在极大规则集情况下仍存在面积提升但未达到最优，且对不同硬件平台的兼容性尚待进一步验证。

---

## 175. Active Label Cleaning for Reliable Detection of Electron Dense Deposits in Transmission Electron Microscopy Images

**arXiv ID:** 2602.05250 | [PDF](https://arxiv.org/pdf/2602.05250v1)

**作者:** Jieyun Tan `[一作]` (Southern Medical University), Lei Cao `[通讯]` (Southern Medical University)

**通讯引用:** 3098 | [OpenAlex ID](https://openalex.org/A5101715180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种基于主动学习的标签清洗方法，针对电子密集沉积（EDD）目标检测中的众包标签噪声，实现自动化且高效的噪声等级修正。

**💡 创新点**

① 双模型协同的Label Selection Module结合活学习选择最有价值样本；② 对噪声进行实例级分级自动修正与人工复核相结合；③ 引入Bib Correction Module自动消除嵌盒噪声；④ 在医学小样本场景下设计针对性主动学习策略。

**🔧 技术方法**

采用Faster R‑CNN+ResNet‑50+FPN目标检测框架，配合MMDetection训练；基于不一致度的主动学习框架进行样本挑选；构建标签清洗模型并使用TIDE评估噪声；引入Bib Correction Module实现自动嵌盒噪声校正。

**📊 数据集**

私有数据集：1112张肾小球TEM图像（包含众包与病理专家标注），以及公开BCCD数据集（205训+87验证+72测试），通过众包与专家双重标注构造噪声标签。

**📈 对比分析**

与Noisy Training、OA‑MIL、Mao、ALC、Clean Training及随机采样对比。私有数据集上AP_50提升至67.18%（比噪声训练高18.83%，接近Clean Training的95.79%），并将噪声种类显著降低；在BCCD上AP_50提升至88.63%（比噪声训练高4.81%，达97.97% Clean）。同时标注成本降低73.3%。

**⚠️ 局限性**

仅在小规模医学数据上验证，需病理专家参与的复核仍占比；对极大噪声比例或不同噪声类型的泛化尚未充分评估；阈值δ、γ需要手动调参。

---

## 176. Causal Representation Meets Stochastic Modeling under Generic Geometry

**arXiv ID:** 2602.05033 | [PDF](https://arxiv.org/pdf/2602.05033v1)

**作者:** Jiaxu Ren `[一作]` (New York University), Biwei Huang `[通讯]` (University of California San Diego)

**通讯引用:** 788 | [OpenAlex ID](https://openalex.org/A5071784683)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种可识别的因果表示学习框架，用于从高维观测数据中恢复连续时间随机点过程（如 Hawkes 过程）的潜在因果变量及其结构，并给出了对应的可辨识性理论。

**💡 创新点**

创新点包括：① 将连续时间随机点过程与无穷阶整数自回归（INAR(∞)）等价并通过弱收敛性保证可辨识；② 在泛型非可逆混合函数下给出必要充分条件，首次实现对潜在因果过程的全可辨识；③ 提出 MUTATE——一种基于可辨识理论的可变自编码器，结合时间频域的神经 PSD 模块实现对随机因果过程的学习。

**🔧 技术方法**

技术手段包括：代数几何方法（理想维数、Kruskal 条件、累积分布的张量分解）、弱收敛性分析、INAR(∞) 与 Hawkes 过程的对应、可变自编码器架构、频域神经 PSD 模块、ELBO 目标优化。

**📊 数据集**

使用合成数据（五种核函数：指数、幂律、矩形、非线性、非参数）以及基于 SERGIO 的基因表达模拟，此外还对真实世界的神经元脉冲和基因突变数据进行了验证。

**📈 对比分析**

与 TDRL、BetaVAE、SlowVAE、PCL 等基线进行对比，采用 MCC 评估潜在变量重建质量。实验显示 MUTATE 在所有核函数下均显著优于基线，MCC 最高达 0.84，说明在模拟与真实数据中表现最优。

**⚠️ 局限性**

局限性包括：① 仅对可辨识的线性或泛型非线性混合函数给出理论，完全非参数核函数尚未覆盖；② 对高维、强非线性或强耦合的实际数据可能仍需更复杂的模型；③ 需要充分的时间分辨率与多环境干预来满足可辨识条件。

---

## 177. Gradually Compacting Large Language Models for Reasoning Like a Boiling Frog

**arXiv ID:** 2602.04919 | [PDF](https://arxiv.org/pdf/2602.04919v1)

**作者:** Yiran Zhao `[一作]` (Salesforce AI Research), Michael Qizhe Shieh `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种逐步压缩LLM的Prune–Tune Loop（PTL）方法，结合结构化剪枝与微调，保持推理性能；

**💡 创新点**

创新点在于将压缩过程细化为多轮小幅Prune–Tune循环，避免一次性剪枝导致性能骤降，并支持神经元与层级双重剪枝；

**🔧 技术方法**

采用结构化神经元/层级剪枝、持续预训练（CoT数据）与强化学习（GRPO）恢复、DeepSpeed ZeRO Stage‑2等技术；

**📊 数据集**

使用数学推理数据集NuminaMath‑CoT、MetaMathQA、GSM8K、Minerva Math、MATH‑500，代码生成任务使用MBPP、OpenCoder、StarCoder Python子集；

**📈 对比分析**

与ShortGPT、SliceGPT、Prune‑Once等基线对比，压缩30‑40%参数后，数学推理准确率基本不降甚至略升，FLOPs降低30‑50%，推理速度提升约2‑2.5倍；对Qwen2.5‑7B通过RL可恢复甚至超越原模型；

**⚠️ 局限性**

未验证指令跟随能力，仅支持开源模型，且仍需要一定的硬件与训练成本。

---

## 178. Doc2Spec: Synthesizing Formal Programming Specifications from Natural Language via Grammar Induction

**arXiv ID:** 2602.04892 | [PDF](https://arxiv.org/pdf/2602.04892v1)

**作者:** Shihao Xia `[一作]` (Pennsylvania State University), Linhai Song `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多代理框架 Doc2Spec，利用大型语言模型从自然语言 API 规则中自动诱导语法并生成形式化规范。

**💡 创新点**

首次将语法诱导与 LLM 指导的规范生成相结合，使用模板化 DSL 自动学习域特定排序和谓词，显著提升了规范的一致性与覆盖率。

**🔧 技术方法**

采用多代理系统、GPT‑OSS‑20B 等 LLM 进行实体定位、属性提取、规则抽取与语法诱导，配合正则表达式、滑动窗口、JSON 验证以及 Lark 语法检查等技术。

**📊 数据集**

在七份 API 文档上评估，涵盖三种 ERC 标准（ERC20/721/1155）、两份 Rust 内存分配器实现和两份 Java 库实现，总计约 132+ 约 17+26+76+53 条规则。

**📈 对比分析**

与无语法诱导基线相比，精度提升 0.27、召回率提升 0.11；与手工编写语法的 SymGPT 对比，ERC1155 召回率更高，并能检测到 120 条真实合约违规。

**⚠️ 局限性**

实验仅覆盖七份文档，DSL 仅支持基本逻辑（无量化、时序），且需要外部 LLM 翻译才能与验证引擎对接，限制了可推广性。

---

## 179. Parameterized Algorithms for the Drone Delivery Problem

**arXiv ID:** 2602.04985 | [PDF](https://arxiv.org/pdf/2602.04985v1)

**作者:** Simon Bartlmae `[一作]` (University of Copenhagen), Heiko Röglin `[通讯]` (University of Bonn)

**通讯引用:** 1437 | [OpenAlex ID](https://openalex.org/A5034122679)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了可选起点的无人机递送问题（DDT‑SP）在路径图上的复杂性，并提出了基于交叉图树宽和最大度的参数化算法；

**💡 创新点**

创新点包括首次证明路径图上DDT‑SP为a(n)-APX‑hard，提出了当交叉图树宽为w时路径图上FPT算法，以及在交叉图为树时的O(k²n²)多项式解法；

**🔧 技术方法**

采用了交叉图（Intersection Graph）概念、树宽与最大度参数化、动态规划与树分解、图论归约与硬化、以及Dijkstra图层化构造等技术；

**📊 数据集**

本文未使用具体实验数据，而是通过理论证明和构造实例来展示算法效果；

**📈 对比分析**

与先前的NP‑hard结果相比，本文在路径图上给出无多项式近似的下界，在交叉图为树时实现O(k²n²)的最优算法，在一般图上实现f(Δ,w)·poly(n,k)的FPT复杂度；

**⚠️ 局限性**

限制包括对固定速度数目的强NP‑难性仍未解决、参数化算法的依赖仍较高，以及缺乏实际实验验证等。

---

## 180. Exceptional Behaviors: How Frequently Are They Tested?

**arXiv ID:** 2602.05123 | [PDF](https://arxiv.org/pdf/2602.05123v1)

**作者:** Andre Hora `[一作]` (Federal University of Minas Gerais), Gordon Fraser `[通讯]` (University of Passau)

**通讯引用:** 11291 | [OpenAlex ID](https://openalex.org/A5079261847)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对25个流行和标准库的Python系统的测试套件进行动态分析，量化了异常行为的出现频率，并在方法、调用和系统层面展开了统计与分类。

**💡 创新点**

创新点在于：①首次对测试运行时所有抛出的异常（不只限于传递给测试的异常）进行全景性频率分析；②将异常发生的罕见性与频繁性分层次归类，揭示了异常行为既可以是“异常”也可能是“正常”流程；③提出针对高频异常的`try/except`块重构建议。

**🔧 技术方法**

技术手段主要包括：①使用SpotFlow进行全局代码跟踪与异常收集；②对收集到的数据做聚合、分布和四分位分析；③基于Python的标准库和第三方库构建实验环境。

**📊 数据集**

数据集为25个Python项目（10个热门应用、15个标准库/第三方库），共执行5,372个方法、17.9M次调用，捕获1.4M次异常。数据已公开在Zenodo（DOI: 10.5281/zenodo.14187323）。

**📈 对比分析**

通过对异常方法与非异常方法在调用次数、执行路径数、异常调用比例等指标的统计，发现异常方法调用量约为常规方法的4倍、路径数为3倍，且异常调用比例中位数为10%。与先前仅关注异常测试的研究相比，本文提供了更全面的频率视角，未采用传统基准测试，性能表现以统计量呈现。

**⚠️ 局限性**

局限性包括：①未区分异常来源（系统内部 vs 第三方库）和访问级别（公开 API vs 内部实现）；②样本仅限Python生态，无法直接推广到其他语言或更广泛的项目类型；③仅关注运行时异常，未深入研究异常被捕获后处理逻辑的影响。

---

## 181. Proteus: Append-Only Ledgers for (Mostly) Trusted Execution Environments

**arXiv ID:** 2602.05346 | [PDF](https://arxiv.org/pdf/2602.05346v1)

**作者:** Shubham Mishra `[一作]` (UC Berkeley), Chris Jensen `[通讯]` (Azure Research, Microsoft)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于可信执行环境（TEE）的分布式日志系统，在TEE平台被攻破时通过嵌入BFT审计协议实现安全回滚，并保证提交后审计不落后于提交的常数延迟。

**💡 创新点**

创新点在于将BFT协议无额外消息嵌入CFT协议，形成平台故障容错（Platform Fault Tolerance）模型；通过hash‑链、流水线和快速审计路径，实现了审计与提交几乎同步的高性能系统。

**🔧 技术方法**

采用TEE（Intel SGX/SEV‑SNP/TDX）、改进的CFT Raft、BFT HotStuff/Autobahn、hash‑链、流水线、签名间隔技术，并用Rust实现。

**📊 数据集**

使用Azure云平台多区域多TEE虚拟机集合进行实验，并配合YCSB、银行转账、密钥透明、秘密恢复、代码透明等自定义工作负载。

**📈 对比分析**

与现有TEE‑Raft、Engraft、HotStuff、Autobahn、PeerReview对比，提交吞吐量与TEE‑Raft相当，审计吞吐率仅比BFT低15%，提交延迟比BFT低2.6×；在多平台配置下仍保持高吞吐并能快速检测并纠正TEE被攻破导致的分叉。

**⚠️ 局限性**

限制在于需要TEE硬件支持，TEE安全漏洞仍可能导致审计滞后；审计间隔和签名频率会影响性能；系统主要适用于追加日志，非追加写入或需要强一致性写入的场景适用性有限。

---

## 182. SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel

**arXiv ID:** 2602.04915 | [PDF](https://arxiv.org/pdf/2602.04915v1)

**作者:** Jose Miguel Luna `[一作]` (Columbia University), Krzysztof Choromanski `[通讯]` (Google DeepMind)

**通讯引用:** 2685 | [OpenAlex ID](https://openalex.org/A5031842812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为SLAY的线性时间注意力机制，基于将查询和键约束在单位球面上并使用Yat核的球面线性化，实现了与softmax相近的注意力效果。

**💡 创新点**

创新点包括：①利用Bernstein定理将球面Yat核拆解为可正随机特征近似的非负混合；②在单位球面约束下解耦对齐与距离，保持几何自调节特性；③构造完全正的随机特征和多尺度Gauss‑Laguerre积分，实现了真正的线性时间与正定性保证。

**🔧 技术方法**

主要技术手段包括：随机特征近似（PRF和anchor多项式特征）、Gauss‑Laguerre四折积分、Bernstein积分表示、张量化特征融合、标准线性注意力重排；并在此基础上实现了O(L)时间和空间的注意力计算。

**📊 数据集**

使用的数据集包括：综合12种合成序列建模任务、Eurlex‑4K极端分类数据集、以及对GPT‑2 Small进行的长上下文语言建模实验（遵循Chinchilla规模法则），并在多种自然语言与视觉任务上进行验证。

**📈 对比分析**

与标准softmax、YAT、Performers、Cosformers、Linear（ELU+1）等方法比较：SLAY在合成任务中基本匹配softmax表现，极端分类中P@指标明显优于Performer；在Transformer训练中验证损失和困惑度仅与softmax相差几分，远优于其他线性注意力；内存占用、延迟与吞吐量保持线性增长，能够处理数十倍长的序列。

**⚠️ 局限性**

局限性：对查询/键做单位球面归一化可能限制表示多样性；随机特征近似需调节积分点数与特征维度，过低会导致精度下降；在某些大规模任务或极高维输入下，特征构造与张量化开销仍显著；目前实验主要聚焦于文本/极端分类，未覆盖更广泛的视觉或多模态任务。

---

## 183. Quantile-Physics Hybrid Framework for Safe-Speed Recommendation under Diverse Weather Conditions Leveraging Connected Vehicle and Road Weather Information Systems Data

**arXiv ID:** 2602.05053 | [PDF](https://arxiv.org/pdf/2602.05053v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 184. TADS: Task-Aware Data Selection for Multi-Task Multimodal Pre-Training

**arXiv ID:** 2602.05251 | [PDF](https://arxiv.org/pdf/2602.05251v1)

**作者:** Guanjie Cheng `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 9250 | [OpenAlex ID](https://openalex.org/A5055284175)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多任务多模态预训练的数据选择框架TADS，能够在固定训练预算下通过筛选高价值样本提升模型零样本性能。

**💡 创新点**

创新点在于将样本质量、任务相关性和分布多样性三维度联合建模为可学习的价值函数，并通过反馈驱动的元学习机制自适应优化选择策略。

**🔧 技术方法**

使用多模态特征提取、跨模态对齐算子、任务原型相似度、聚类多样性因子以及基于代理模型的REINFORCE梯度估计等技术。

**📊 数据集**

在CC12M大规模噪声数据上进行实验，随后在ImageNet‑1K、CIFAR‑100、MS‑COCO、Flickr30K等四个下游任务上评估。

**📈 对比分析**

与10种任务无关和任务相关的数据筛选方法对比，TADS在仅使用约3.95M样本的情况下在ImageNet Top‑1、CIFAR‑100 Top‑1、MS‑COCO IR@1、Flickr30K TR@1等指标上均超过基线与现有最佳方法，平均提升约1.0%。

**⚠️ 局限性**

局限性包括对代理模型和聚类参数的依赖、在极端稀疏任务定义下任务相关性估计可能受干扰，以及对动态任务集合的适应性仍需进一步研究。

---

## 185. Polynomial-Time Solutions for Longest Common Subsequence Related Problems Between a Sequence and a Pangenome Graph

**arXiv ID:** 2602.05193 | [PDF](https://arxiv.org/pdf/2602.05193v1)

**作者:** Xingfu Li `[一作]` (Guizhou University of Finance and Economics), Yongping Wang `[通讯]` (Guizhou University of Finance and Economics)

**通讯引用:** 1328 | [OpenAlex ID](https://openalex.org/A5101845864)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究序列与 pangenome 图之间的最长公共子序列（LCS）问题及其三种变体，并将它们归约为 DAG 上的最长路径问题，从而证明这些问题属于 P 类。

**💡 创新点**

创新点在于：①首次将 LCS-SG、FGLCS-SG、MEMC、MSP 四个问题统一建模为 DAG 最长路径；②给出了四个多项式时间的归约与构造方法；③系统分析了构造过程的时间复杂度，揭示了现有算法的 cubic 上界。

**🔧 技术方法**

主要技术包括：图变换（构造辅助 DAG）、拓扑排序与动态规划求解 DAG 最长路径、Floyd‑Warshall 预处理以获取顶点间可达性与距离、MEM 链与种子链的严格序列判定。

**📊 数据集**

论文未使用具体实验数据集，全部以理论构造与复杂度分析为主。

**📈 对比分析**

与传统的 LCS 动态规划相比，归约后可以在 DAG 上以线性时间求解最长路径；然而，为构造 DAG 需要 O(n³) 的 Floyd‑Warshall 预处理，导致整体复杂度为 O(n³ + |Q|²N²)，在大规模实例上性能仍受限。

**⚠️ 局限性**

主要局限在于：①构造 DAG 的时间复杂度为 cubic，难以处理大规模 pangenome 图；②对输入长度 |Q| 与图总字符量 N 的二次幂依赖导致空间与时间成本高；③未给出实测或改进方案，缺乏对实际基因组数据的验证。

---

## 186. Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives

**arXiv ID:** 2602.04990 | [PDF](https://arxiv.org/pdf/2602.04990v1)

**作者:** Ioannis Anagnostides `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文分析了美国心脏移植分配过程中因激励不匹配导致的操纵行为，并提出构建激励感知的分配政策的研究路线；

**💡 创新点**

将机制设计、战略分类、因果推断与社会选择等多学科方法融入器官分配，首次系统阐明激励因素对现行分配体系的负面影响；

**🔧 技术方法**

主要讨论并引用机制设计、战略分类、因果推断、社会选择、强化学习与人类反馈等理论与方法；

**📊 数据集**

以UNOS（美国器官移植网络）公开的心脏移植候选人和结果数据为例进行案例与统计分析；

**📈 对比分析**

论文未给出具体模型或算法的实验对比，主要以案例统计、趋势分析和理论论证说明问题；性能评估仍待后续基于激励感知算法的实验研究；

**⚠️ 局限性**

缺乏实证验证与性能评估，理论与实践间的可解释性与可操作性尚待探讨，且激励机制的设计与实施面临监管、伦理和技术复杂性挑战。

---

## 187. Boosting SAM for Cross-Domain Few-Shot Segmentation via Conditional Point Sparsification

**arXiv ID:** 2602.05218 | [PDF](https://arxiv.org/pdf/2602.05218v1)

**作者:** Jiahao Nie `[一作]` (Nanyang Technological University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence, China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种训练无关的条件点稀疏化方法（CPS），利用参考样本在跨域少样本分割任务中自适应地稀疏匹配点，从而提升 SAM 的分割性能。

**💡 创新点**

创新点在于：①发现跨域域移导致密集提示点失效；②通过参考样本的 IoU 计算动态确定最佳点密度，实现条件点稀疏化；③引入边界点剔除和后期掩模修正，进一步提升分割质量。

**🔧 技术方法**

采用 DINOv2 进行特征提取与密集点匹配，利用 SAM 进行分割；通过网格划分实现自适应稀疏化；使用凸包边界剔除误匹配点；使用形态学开闭操作对初始掩模进行细化。

**📊 数据集**

在四个跨域数据集上评估：医学图像（ISIC2018、Chest X‑Ray）、遥感图像（DeepGlobe）、水下图像（SUIM）；并在 FSS‑1000 与 COCO‑20i 上进行额外验证。

**📈 对比分析**

与传统训练依赖的 CD‑FSS 方法以及现有训练无关的 SAM 基线（Matcher、GF‑SAM）在 mIoU 上对比，CPS 在所有跨域数据集上均实现 2–4% 的提升，表现出显著优势。

**⚠️ 局限性**

局限性包括：仅针对单目标图像设计，需进一步扩展到多目标场景；稀疏化过程仍依赖参考掩模的 IoU 评估，参考质量不佳会影响效果；对极小目标的稀疏化可能导致细节丢失。

---

## 188. Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink

**arXiv ID:** 2602.05228 | [PDF](https://arxiv.org/pdf/2602.05228v1)

**作者:** Guozhi Liu `[一作]` (South China University of Technology), Li Shen `[通讯]` (Sun Yat-sen University)

**通讯引用:** 15527 | [OpenAlex ID](https://openalex.org/A5100768717)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过测量注意力“sink divergence”，并在微调阶段抑制正向sink头，实现对有害数据的轻量级防御。

**💡 创新点**

引入sink divergence概念，提出可分离sink divergence假设，并基于此设计Surgery正则化器，从而在微调阶段实现有效且计算开销低的安全防御。

**🔧 技术方法**

利用注意力sink机制构造sink divergence统计量，加入ReLU正则项与交叉熵损失，使用AdamW进行全参数微调。

**📊 数据集**

使用RepNoise‑Refusal（BeaverTails衍生）构造有害/拒绝数据，下游任务采用SST‑2、AGNEWS、GSM8K；评估集包括BeaverTails、HarmBench和SorryBench。

**📈 对比分析**

与七种现有微调防御（Lisa、SafeGrad、ConstrainedSFT、SPARD、AsFT、DSS）在不同模型、比例、样本量和任务上对比，Surgery平均提升约8% 防御效果，且对任务准确率影响极小。

**⚠️ 局限性**

对学习率与正则强度敏感；仅针对大规模指令微调模型，迁移性至其他体系结构或更大规模模型待验证；未解决后期模型恢复或更深层安全性问题。

---

## 189. Benchmarking Artificial Intelligence Models for Daily Coastal Hypoxia Forecasting

**arXiv ID:** 2602.05178 | [PDF](https://arxiv.org/pdf/2602.05178v1)

**作者:** Magesh Rajasekaran `[一作]` (Louisiana State University), Z. George Xue `[通讯]` (Louisiana State University)

**通讯引用:** 2039 | [OpenAlex ID](https://openalex.org/A5101518948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四种深度学习序列模型（BiLSTM、TCN、Medformer、ST-Transformer）进行统一预处理、训练与评估，比较其在每日沿海缺氧（二分类）预测上的表现。

**💡 创新点**

①在同一实验框架下对比多种模型；②首次将McNemar检验和Cohen w效应量引入环境AI模型评估；③评估两种Transformer变体（Medformer与ST-Transformer）在缺氧预测中的表现；④提供完整可复现的代码与数据管道。

**🔧 技术方法**

BiLSTM、TCN、Medformer、ST-Transformer；SMOTE过采样、时间循环编码、滑动窗口序列化；AUC‑ROC、AUC‑PR、准确率、F1、Brier分数、Log Loss、McNemar检验、Cohen w。

**📊 数据集**

COAWST（Louisiana‑Texas shelf）海洋-气象-沉积耦合模型产生的夏季（May–Aug）日数据，2009–2020年12年训练集，2020年8月及2022–2024年夏季作为独立测试集；特征包括PEA、SOC、DCP_Temp。

**📈 对比分析**

采用统一的时间窗口（7天）序列化、相同的训练/验证划分、相同的损失与优化器；使用上述评价指标对比模型。ST‑Transformer在所有测试期间获得最高AUC‑ROC（0.982–0.992）和最高F1；BiLSTM与ST‑Transformer差异在McNemar检验中无统计显著（p≈0.098）。

**⚠️ 局限性**

仅针对夏季缺氧进行评估，缺乏跨地区或多季节泛化验证；模型仅做二分类，未考虑多步预测；实时推理时延未评估；对极罕见缺氧事件的识别仍不理想；缺乏对模型对物理约束的解释。

---

## 190. PriMod4AI: Lifecycle-Aware Privacy Threat Modeling for AI Systems using LLM

**arXiv ID:** 2602.04927 | [PDF](https://arxiv.org/pdf/2602.04927v1)

**作者:** Gautam Savaliya `[一作]` (Deggendorf Institute of Technology), Martin Schramm `[通讯]` (Deggendorf Institute of Technology)

**通讯引用:** 192 | [OpenAlex ID](https://openalex.org/A5056684537)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 PriMod4AI 框架，利用 LLM 结合 LINDDUN 与 AI 专用隐私攻击知识库实现生命周期感知的自动化隐私威胁建模。

**💡 创新点**

将双重结构化知识库与检索增强生成（RAG）相结合，统一覆盖传统 LINDDUN 威胁与模型中心攻击，并在 DFD 上实现阶段化推理与可解释输出。

**🔧 技术方法**

使用结构化知识库构建、FAISS 向量检索、LLM（GPT‑OSS、LLaMA 3.1）+ RAG、JSON 模板化提示以及生命周期映射与评估指标。

**📊 数据集**

对人脸认证系统与自动驾驶系统的 DFD 进行评估，结合公开的 PILLAR 结果和自建的 AI‑Privacy KB 文献。

**📈 对比分析**

通过与 PILLAR 的 LINDDUN 分类覆盖与召回对比（召回率≈85%，Jaccard≈0.7）以及跨 LLM 的 Cohen κ≈0.7，表明模型输出一致性良好。

**⚠️ 局限性**

缺乏专家标注的 AI 专用威胁基准，知识库更新不动态，未提供直接的对策建议，且在极大规模系统上的验证有限。

---

## 191. Feedback Control for Multi-Objective Graph Self-Supervision

**arXiv ID:** 2602.05036 | [PDF](https://arxiv.org/pdf/2602.05036v1)

**作者:** Karish Grover `[一作]` (Carnegie Mellon University), Christos Faloutsos `[通讯]` (Carnegie Mellon University)

**通讯引用:** 81220 | [OpenAlex ID](https://openalex.org/A5035605036)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于控制理论的闭环调度框架，通过时间分配单任务块来协调多目标图自监督学习，避免梯度冲突。

**💡 创新点**

创新点是将多目标图自监督视为闭环调度问题，结合谱需求、梯度冲突感知、Pareto‑aware log‑hypervolume规划与PID缺口追踪，实现可审计的动态任务分配。

**🔧 技术方法**

使用全图谱估计的谱需求与MGDA干扰信号，基于log‑hypervolume的目标分配规划，PID缺口追踪控制以及B‑block单任务更新等技术。

**📊 数据集**

实验涵盖9个标准图数据集：Cora、Citeseer、PubMed、Coauthor‑CS/Physics、Wiki‑CS、Chameleon、Squirrel、Actor以及ogbn‑arxiv。

**📈 对比分析**

与单任务SSL、AutoSSL、ParetoGNN、PCGrad、CAGrad等方法对比，平均排名均居前，在节点分类、链接预测和聚类任务上均实现显著性能提升。

**⚠️ 局限性**

局限性包括需手动设定调度周期、块大小和参考点，适配极大规模图时需要近似估计，控制参数调优相对繁琐。

---

## 192. Learning, Solving and Optimizing PDEs with TensorGalerkin: an efficient high-performance Galerkin assembly algorithm

**arXiv ID:** 2602.05052 | [PDF](https://arxiv.org/pdf/2602.05052v1)

**作者:** Shizheng Wen `[一作]` (ETH Zurich), Siddhartha Mishra `[通讯]` (ETH Zurich)

**通讯引用:** 5364 | [OpenAlex ID](https://openalex.org/A5101961582)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种统一的 TensorGalerkin 框架，能够高效地求解、学习和优化具有变分结构的偏微分方程 (PDE)，并实现三种下游应用：数值 PDE 求解器 TensorMesh、物理信息算子学习框架 TensorPils 以及端到端可微的 PDE 约束优化框架 TensorOpt。

**💡 创新点**

核心创新在于将 Galerkin 组装过程改写为两阶段严格张量化的 Map‑Reduce：Batch‑Map 通过单一 GPU 核心完成所有单元局部算子张量收缩，Reduce 阶段利用预计算的稀疏路由矩阵实现全局稀疏矩阵乘法（SpMM），从而消除传统散点加（scatter‑add）循环带来的 Python 解释器和自动微分开销；同时在物理信息算子学习中采用解析形函数梯度而非自动微分，显著降低梯度计算成本。

**🔧 技术方法**

技术手段包括：
- 基于变分形式的 Galerkin 离散；
- TensorGalerkin 的张量化 Batch‑Map 与稀疏 Reduce；
- PyTorch 生态下的 GPU 核心实现；
- 解析形函数梯度用于空间导数计算；
- 端到端可微的自动微分链路，用于逆设计与优化；
- 与传统 FEM 软件（FEniCS、scikit‑fem、JAX‑FEM）以及 PINN、VPINN、DeepRitz、PI‑DeepONet 的对比实验。

**📊 数据集**

使用一系列人工合成基准数据：
- 3D Poisson 方程、线性弹性方程；
- 2D Poisson 方程（不同频率的检查板源项）；
- 圆域线性声学波方程、L‑形域 Allen‑Cahn 方程；
- 2D 触臂梁拓扑优化（SIMP 方法）。

**📈 对比分析**

与现有方法比较：
- 在 TensorMesh 中相较 FEniCS、scikit‑fem 速度提升 10‑100 倍；相较 JAX‑FEM GPU 版本提升 3–4 倍；
- 在 TensorPils 中相较 PINN、VPINN、DeepRitz，误差平均下降 50% 以上，训练速度提升 2–3 倍；
- 在算子学习实验中，TensorPils 对 ID 与 OOD 误差均比 PI‑DeepONet 降低 8–10 倍；
- 在拓扑优化实验中，TensorOpt 与 JAX‑FEM 比较，整体运行时间缩短 3.7×，且收敛设计相同。

**⚠️ 局限性**

局限性：
- 仅适用于具有变分结构的 PDE；
- 当前实现基于符合单元 FEM，尚未覆盖 DG、Petrov–Galerkin 等非符合方法；
- 对三维复杂几何、非稳态（时间步进）和非线性 PDE 的支持尚待扩展；
- 需要进一步验证在更真实工程场景中的可扩展性与鲁棒性。

---

## 193. Traceable Cross-Source RAG for Chinese Tibetan Medicine Question Answering

**arXiv ID:** 2602.05195 | [PDF](https://arxiv.org/pdf/2602.05195v1)

**作者:** Fengxian Chen `[一作]` (Lanzhou University), Qingguo Zhou `[通讯]` (Lanzhou University)

**通讯引用:** 2046 | [OpenAlex ID](https://openalex.org/A5100604365)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对中文藏医问答，在三类知识库（百科、经典文献、临床论文）下实现检索增强生成（RAG）系统，并通过两种方法提升可追溯性、降低幻觉、支持跨库验证。

**💡 创新点**

创新点在于：① DAKS 路由器利用检索分布特征与权威先验进行预算化检索，减少密集百科库的主导；② 构建分块–实体对齐图，引导跨库融合与证据打包，显著提升跨库证据覆盖率和引用准确率。

**🔧 技术方法**

技术手段包括：轻量级检索器+可选重排序、DAKS 路由与预算分配、结构感知扩展、图支持评分、基于令牌预算的覆盖感知贪婪打包、以及 openPangu-Embedded-7B 轻量级生成器。

**📊 数据集**

使用自构建的 500 条藏医问答基准，涵盖定义、经典原理、临床证据、跨库综合四类问题，每条记录附有黄金答案和分块级证据。

**📈 对比分析**

与单库检索、合并库检索、朴素多库拼接等基线相比，DAKS+GraphFusion 在 CrossEv@5、Faithfulness、Citation Correctness 等指标上取得最优或接近最优表现；单独使用 DAKS 或 GraphFusion 时，性能提升有限，表明两者互补。

**⚠️ 局限性**

局限性包括：评估仅依赖 GLM-4.7 判别模型；实验仅在中文藏医领域，缺乏跨语言/子领域的验证；对实体提取与类型化的依赖可能导致图支持受限；未对检索模型进行领域监督，检索性能仍有提升空间。

---

## 194. Food Portion Estimation: From Pixels to Calories

**arXiv ID:** 2602.05078 | [PDF](https://arxiv.org/pdf/2602.05078v1)

**作者:** Gautham Vinod `[一作]`, Fengqing Zhu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了从传统几何重建到深度学习单目推断的食物份量估计方法，并探讨了其在慢性病管理中的应用潜力。

**💡 创新点**

提出将多模态大语言模型与视觉特征结合以弥补体积-质量转换的不确定性，并强调了无标记尺度估计与隐式表面重建的创新方向。

**🔧 技术方法**

主要技术包括单目深度估计、直接能量回归、NeRF隐式表面表示、语义-几何多尺度融合以及扩散模型的不可见几何推断。

**📊 数据集**

利用公开食物图像数据集（如Food-101、UEC-Food、MSCOCO等）以及营养数据库（如USDA FNDDS）进行训练与评估。

**📈 对比分析**

与基于深度传感器、MVS、模板匹配等传统方法对比，单目深度+NeRF方法在可用性上显著提升，但在体积误差上仍有10–20%范围，能量估计误差受密度估计影响。

**⚠️ 局限性**

主要局限包括尺度不确定性、遮挡导致的体积欠估、密度差异导致的能量误差、对训练数据分布的依赖以及对复杂食物结构的建模不足。

---

## 195. LISA: Laplacian In-context Spectral Analysis

**arXiv ID:** 2602.04906 | [PDF](https://arxiv.org/pdf/2602.04906v1)

**作者:** Julio Candanedo `[一作]` `[通讯]` (SparseTrace.ai), Julio Candanedo (SparseTrace.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

引入了一种基于拉普拉斯谱分析的时间序列自适应推理框架 LISA，在推理时利用前缀进行轻量级的无梯度更新，实现对拉普拉斯谱模型的上下文适配。

**💡 创新点**

创新点在于将延迟嵌入、拉普拉斯谱学习与两种无参数上下文适配器（高斯过程回归和注意力式 Markov 算子）结合，首次实现连续时间序列的无参数推理时间自适应。

**🔧 技术方法**

使用技术包括 NLSA、延迟坐标嵌入、拉普拉斯谱、GPLM 解码器、Gaussian Process 回归、Nadaraya–Watson 核回归和注意力式 Markov 算子。

**📊 数据集**

实验数据集包括经典三维混沌吸引子（Rössler、Lorenz‑63 等）、受控参数切换的 Lorenz 系统以及真实的美国县级电力负荷时序数据。

**📈 对比分析**

与固定窗口基线 NLSA、两种 ICM 变体 LISA/ALSA 以及监督神经网络 PTST 在多起点、不同上下文长度和不同预测步长下进行对比，结果显示 LISA/ALSA 在长上下文下能显著降低误差并保持动力学一致性，尤其在非平稳 regime‑switch 中表现优异。

**⚠️ 局限性**

局限性包括对延迟窗口长度 L 的强依赖、对噪声和弱相关历史的敏感性，以及随着上下文长度增加导致的计算成本上升，且缺乏自动化选择 L 或上下文筛选策略。

---

## 196. ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation

**arXiv ID:** 2602.05051 | [PDF](https://arxiv.org/pdf/2602.05051v1)

**作者:** Songyuan Zhang `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1860 | [OpenAlex ID](https://openalex.org/A5019603699)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于反射流的离线强化学习方法 ReFORM，能够在保持策略在行为策略支持集内的前提下，利用流模型学习多模态动作分布并实现性能提升。

**💡 创新点**

创新点包括：① 通过将行为克隆流的源分布设为有界球面，从而构造出满足支持约束的策略；② 引入反射流噪声生成器，使噪声在源分布的支持内生成并保持多模态；③ 通过构造式的支持约束避免了传统统计距离正则化所需的超参数调优，且不限制策略改进。

**🔧 技术方法**

技术手段包括：流匹配（flow matching）训练行为克隆流，反射流（reflected flow）保证噪声在有界域内，BPTT+分解式反射Euler方法实现训练；使用基于 Q‑value 的演员-评论家框架进行策略优化，并对行为克隆流进行一阶蒸馏。

**📊 数据集**

数据集：OGBench 离线 RL 基准，包含 40 个任务（运动学和操作任务），分别提供“clean”（专家生成）和“noisy”（低质量噪声）两类数据集。

**📈 对比分析**

与 FQL、IFQL、DSRL 等现有流模型离线 RL 算法比较，ReFORM 在两类数据集上都取得最高的性能曲线，且只需统一的超参数；在“clean”数据集上排名第一，且在“noisy”数据集上不受正则化强度敏感性影响，证明了其对 OOD 的有效抑制和对多模态策略的支持。

**⚠️ 局限性**

局限性包括：① 仍依赖行为克隆模型，若其产生 OOD 错误会影响最终策略；② 噪声生成器训练需要 BPTT，计算开销较大；③ 在专家级数据集上学习速度相对慢，缺少显式正则化导致对专家策略的依赖；④ 反射项的设计尚未理论化，可能存在更优方案。

---

## 197. Near-Optimal Dynamic Matching via Coarsening with Application to Heart Transplantation

**arXiv ID:** 2602.04989 | [PDF](https://arxiv.org/pdf/2602.04989v1)

**作者:** Itai Zilberstein `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20180 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于聚类（coarsening）的在线随机匹配算法，用于心脏移植分配；

**💡 创新点**

通过将离线节点聚合为容量有限的簇，既保持理论近似性质，又在实际数据中获得近乎最佳的竞争比；

**🔧 技术方法**

利用在线随机 b‑matching、线性规划离线规划、容量聚类（如约束凝聚、k‑means、递归二分）以及代表权重的均值估计；

**📊 数据集**

使用美国 UNOS 注册数据库（1987‑2019 年心脏移植候选人与捐献者记录）进行实验；

**📈 对比分析**

与 b=1（单个候选人）和美国现行 6 层优先级政策比较，实验显示竞争比从 0.63/0.51 提升至约 0.91，显著高于基线；

**⚠️ 局限性**

局限性包括历史数据偏差与预测不确定性、聚类导致的个体公平性受损、模型对系统动态假设过于静态、以及对边缘病例的潜在不利影响。

---

## 198. FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion

**arXiv ID:** 2602.05305 | [PDF](https://arxiv.org/pdf/2602.05305v1)

**作者:** Zhuokun Chen `[一作]` (Monash University), Bohan Zhuang `[通讯]` (Zhejiang University)

**通讯引用:** 3630 | [OpenAlex ID](https://openalex.org/A5076928390)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

分析块扩散在长上下文推理中的注意力重复开销，提出块外注意力缓存机制以减少KV缓存访问和计算量。

**💡 创新点**

发现块外注意力在连续扩散步之间高度稳定，利用缓存重用该稳定部分，实现与稀疏注意力兼容的残差重用方案。

**🔧 技术方法**

使用块扩散、KV缓存、注意力拆分与log‑space合成、FlashAttention、重新分配注意力、稀疏注意力与重用感知蒸馏等技术。

**📊 数据集**

在文本生成上使用 Trado‑8B‑Thinking，数学与代码基准（GSM8K、MATH500、AIME、LiveCodeBench‑V2、LiveBench、HumanEval、MBPP），在视频生成上使用 LongLive‑1.3B 与 VBench2 数据集。

**📈 对比分析**

与原始块扩散、SparseD 等基线对比：在 32k 上下文下，块外缓存可使吞吐量提升至 1.44×、注意力时间缩短 1.6×，质量基本不变；与 SparseD 结合时，可在高稀疏率下恢复或提升 7–10% 的准确率。

**⚠️ 局限性**

仅在实验中验证，稀疏注意力的实现尚未与 KV‑cache 兼容；长视频推理受限于其它模型模块，整体加速受限；需要进一步开发高效的块扩散友好稀疏注意力核。

---

## 199. Depth-Wise Emergence of Prediction-Centric Geometry in Large Language Models

**arXiv ID:** 2602.04931 | [PDF](https://arxiv.org/pdf/2602.04931v1)

**作者:** Shahar Haim `[一作]` (Champalimaud Centre for the Unknown), Daniel C McNamee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在不同层级的计算，从上下文处理阶段到预测形成阶段，并通过几何分析与因果干预结合，揭示后期层的角度结构对预测产生决定性作用。

**💡 创新点**

创新点在于将几何分析与干预实验统一，证明后期层的角度组织是决定预测的因果结构；同时显示可解性或线性分离并不等同于可操纵性，为模型解释与控制提供新的理论基础。

**🔧 技术方法**

使用Transformer解码器模型，实施层级干预（输入中心与输出中心）、纯角度干预、纯范数干预；对隐藏层进行参与比率（PR）维度估计，计算欧氏/角度距离与预测分布的KL相关性，并对Softmax读出层的角度‑范数分解。

**📊 数据集**

数据集包括自定义的Months任务（总共144个提示）用于干预评估，以及Wiki‑text‑103‑raw‑v1中的短（15词）与长（500–600词）有序/打乱序列，用于几何分析；实验模型为Llama 8B、Mistral 7B、Qwen 7B三款开源解码器LLM。

**📈 对比分析**

通过干预实验比较早期层对输入干预的敏感度与后期层对输出干预的敏感度；角度干预仅在后期层有效，范数干预无效；在Months任务中，Llama、Mistral、Qwen的准确率分别为95%、85%和70%，说明干预效果与模型性能相关。

**⚠️ 局限性**

局限性：仅针对单一任务和解码器模型，未验证跨任务或跨模态的普适性；未深入阐明早期层几何如何导致后期角度编码；未分析多步或算法任务中的上下文动态。

---

## 200. AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction

**arXiv ID:** 2602.05353 | [PDF](https://arxiv.org/pdf/2602.05353v1)

**作者:** Ruijie Shi `[一作]`, Chen Qian `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Agentic Workflow Reconstruction (AWR) 任务并实现 AgentXRay 框架

**💡 创新点**

将黑盒 LLM 代理系统的工作流程以可编辑白盒形式重构，并通过红黑剪枝的 MCTS 提高搜索效率

**🔧 技术方法**

采用 MCTS、Red‑Black Pruning、统一原语空间、线性化工作流表示以及 SFE 评价指标

**📊 数据集**

使用多领域数据集，包括 ChatDev、MetaGPT、TeachMaster、ChatGPT 3D 模型、Gemini 以及 Atoms 开源多智能体平台

**📈 对比分析**

与 SFT、Claude、ReAct、AFlow 等基线对比，五大领域平均 SFE 达到 0.426，优于 AFlow 的 0.339，并在固定预算下减少 8–22% 令牌消耗

**⚠️ 局限性**

仅适用于线性链式工作流，对并发/异步结构不适用；评估仅基于输出相似性，无法保证内部实现一致

---

## 201. Correcting Contextual Deletions in DNA Nanopore Readouts

**arXiv ID:** 2602.05072 | [PDF](https://arxiv.org/pdf/2602.05072v1)

**作者:** Yuan-Pon Chen `[一作]` (University of Illinois Urbana-Champaign), Jin Sima `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 503 | [OpenAlex ID](https://openalex.org/A5051523017)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出并研究了一类上下文删除信道，给出了容量上下界，并构造了多种高效纠错码，尤其在对数阈值下实现了低冗余编码与解码。

**💡 创新点**

首次引入上下文删除模型并对其容量进行理论分析，同时设计了基于哈希、Reed–Solomon码和有限独立生成器的高效纠错方案。

**🔧 技术方法**

使用了随机化独立生成、有限域Reed–Solomon编码、哈希函数与bounded independence生成器以及典型的计数/枚举方法来实现码构造与容量估计。

**📊 数据集**

无实际数据集，全部基于理论模型与组合计数进行分析。

**📈 对比分析**

与传统删除码相比，所给码在冗余量上实现了更优的理论上界（如O(t(1−C)log n)），并通过数学证明证明其为零误码。

**⚠️ 局限性**

仅针对二进制、阈值为对数级别的情形，实际实现复杂度高且未在实验环境中验证，对更大阈值或多符号信道的推广仍有待研究。

---

## 202. Evaluating Large Language Models on Solved and Unsolved Problems in Graph Theory: Implications for Computing Education

**arXiv ID:** 2602.05059 | [PDF](https://arxiv.org/pdf/2602.05059v1)

**作者:** Adithya Kulkarni `[一作]` (Ball State University), Jay Bagga `[通讯]` (Ball State University)

**通讯引用:** 247 | [OpenAlex ID](https://openalex.org/A5066100121)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估大型语言模型在已解决和未解决的图论问题中的推理表现

**💡 创新点**

通过八阶段评估协议，系统比较模型在已知解与开放问题上的可靠性与局限性

**🔧 技术方法**

使用ChatGPT 5.1进行多模态提示工程，执行包含问题理解、探索、结果回顾、证明尝试等八阶段工作流程

**📊 数据集**

利用两篇背景论文（动态图标签综述与线图研究论文）以及GitHub提示与输出存档作为输入数据集

**📈 对比分析**

专家根据定义准确性、结果召回率、证明有效性等指标评估模型；在已解决问题上模型达到约90%以上的正确率，而在未解决问题仅提供探索性思路且未产生新洞见

**⚠️ 局限性**

模型缺乏原创推理与验证能力，无法在未知问题中生成新理论或完整证明，仅能整理已知知识

---

## 203. Differentiable Inverse Graphics for Zero-shot Scene Reconstruction and Robot Grasping

**arXiv ID:** 2602.05029 | [PDF](https://arxiv.org/pdf/2602.05029v1)

**作者:** Octavio Arriaga `[一作]` (Robotics Innovation Center), Rebecca Adam `[通讯]` (Robotics Innovation Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在单张RGBD图像与边界框提示下，提出一种基于基础分割模型与物理可微渲染的零样本场景重建与机器人抓取框架。

**💡 创新点**

核心创新包括：①使用概率椭球估计实现鲁棒的三维初始姿态；②引入物理一致的可微渲染器和柔性掩码；③三阶段粗细优化链路，完全不需要训练数据即可在新环境中重建并抓取对象。

**🔧 技术方法**

技术手段涵盖：SAM分割、基于物理约束的可微渲染器（JAX-sphere/JAX-mesh）、最大后验椭球估计、基于梯度下降的场景与网格优化、软掩码函数、深度图与RGB的多模态融合。

**📊 数据集**

实验使用的公开数据集包括：CLVR（CLEVR‑style）、FewSol、MOPED、LINEMOD‑OCCLUDED 以及 YCB‑Video 中的物体。

**📈 对比分析**

与 FS6D、Gen6D、OnePose++ 等主流无监督/少量样本方法对比，本文在 AR_VSD、Chamfer/ Hausdorff 距离上均实现或接近最先进水平；在零样本抓取任务中获得 89.3% 的成功率，超过 GraspSAM 等基于训练抓取的数据驱动方法。

**⚠️ 局限性**

主要限制：整体优化耗时约 1 分钟（RTX 5090 GPU）；依赖预先给出的边界框；当前模型未针对不同传感器条件进行优化，且在极端遮挡或纹理缺失情况下仍易出现分割误差。

---

## 204. How Do Language Models Acquire Character-Level Information?

**arXiv ID:** 2602.05347 | [PDF](https://arxiv.org/pdf/2602.05347v1)

**作者:** Soma Sato `[一作]` (Nagoya University), Ryohei Sasano `[通讯]` (Nagoya University)

**通讯引用:** 796 | [OpenAlex ID](https://openalex.org/A5049498516)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对输入文本进行控制性变换和词元化器操作，系统分析语言模型如何获取字符级信息，量化分词规则、正字法约束与语义/句法线索对字符学习的贡献。

**💡 创新点**

将字符级信息获取因素分为分词相关（合并规则、正字法约束）与分词无关（子串语义关联、句法信息）两类，并分别量化其对模型字符预测性能的影响。

**🔧 技术方法**

使用词嵌入的多层感知机探测器、控制版BPE/WordPiece分词器、词替换与字符扰动变换、语义分组、词形还原与词干提取等技术对模型进行实验。

**📊 数据集**

主要采用FineWeb大规模英文文本作为预训练语料，训练小规模模型（BERT‑Tiny、nanoGPT），并与公开模型（GPT‑2、BERT‑base‑uncased、GPT‑J）进行对比。

**📈 对比分析**

通过匹配/不匹配字符长度分布、不同子词长度、变换类型及分词器设置的探测任务评估模型性能；nanoGPT在匹配条件下达成约66–69%准确率，低于大型公开模型，但与BERT‑base‑uncased相当。

**⚠️ 局限性**

实验仅限英文与FineWeb语料，未完全消除词汇表信息对结果的影响；所用模型规模有限，未检验更大模型的行为；未深入分析低频词或罕见字符对性能的影响。

---

## 205. Privacy Amplification Persists under Unlimited Synthetic Data Release

**arXiv ID:** 2602.04895 | [PDF](https://arxiv.org/pdf/2602.04895v1)

**作者:** Clément Pierquin `[一作]` (Craft AI), Matthieu Boussard `[通讯]` (Craft)

**通讯引用:** 838 | [OpenAlex ID](https://openalex.org/A5055391641)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

本文研究了在线性生成器下通过发布合成数据实现差分隐私放大效应，并证明在参数有界的前提下，即使发布无限多合成样本，仍能获得隐私放大；

**💡 创新点**

创新点在于：1）提出了在无穷多合成样本时隐私放大仍存在的理论证明；2）通过将隐私损失归结为 Gram 矩阵的分布，利用 Fisher 信息的上界得到统一且紧的 Rényi 隐私上界；3）给出了既适用于局部（高隐私）又适用于全局非渐进的隐私保证；

**🔧 技术方法**

主要技术包括：Rényi 差分隐私框架、Fisher 信息与 Rényi 散度的关系、非中心卡方与 Wishart 分布的 Fisher 信息上界、以及充分统计量的归约；

**📊 数据集**

实验使用了合成数据（随机生成的线性模型参数和高斯噪声），未使用公开真实数据集；

**📈 对比分析**

与此前的 <cit> 方法比较，本文的上界不依赖于合成样本数，且在无穷多样本极限下可获得更强的隐私放大；实验显示合成样本数迅速趋近理论极限，放大效果显著；

**⚠️ 局限性**

局限性包括：仅针对线性生成器，未直接验证深度生成模型；Fisher 信息上界仍有保守空间，可能导致全局上界不够紧；假设参数有界，实际训练中可能难以严格满足；

---

## 206. Mind the Performance Gap: Capability-Behavior Trade-offs in Feature Steering

**arXiv ID:** 2602.04903 | [PDF](https://arxiv.org/pdf/2602.04903v1)

**作者:** Eitan Sprejer `[一作]` (Universidad de Buenos Aires), Iván Arcuschin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了 Goodfire 的 Auto Steer 目标功能驱动方法在 Llama-8B 与 Llama-70B 上的行为控制效果与任务表现的权衡，比较了简单提示、Auto Steer 与混合方法；

**💡 创新点**

首次量化展示了特征驱动干预在提高行为控制的同时会严重牺牲推理准确性和输出连贯性的根本能力‑行为折衷，强调需在部署前经验性评估；

**🔧 技术方法**

使用了 Goodfire Auto Steer 的特征编辑、LLM‑as‑Judge（GPT‑4o‑mini）评估连贯性与行为、MMLU 题目准确率等指标；

**📊 数据集**

采用 171 条来自 MMLU（57 科目）随机抽取的多选题，配合 14 个行为提示进行实验；

**📈 对比分析**

对比结果显示：简单提示保持与对照组相近的 66%（8B）/86.9%（70B）准确率和高连贯度；Auto Steer 行为得分提升 0.35–0.46 但准确率下降 19.9%（8B）/13.5%（70B），连贯度降 52%（8B）/21%（70B）；混合方法虽行为进一步提升，但继承了 Auto Steer 的连贯性与准确率损失；

**⚠️ 局限性**

限制：仅评估单一 Auto Steer 方法，未探讨不同强度与其他特征驱动技术；仅使用 MMLU 多选问答，可能不适用于开放式生成任务；LLM‑as‑Judge 的自动评估可能存在偏差；仅测试 Llama 系列，缺乏跨架构验证。

---

## 207. GAS: Enhancing Reward-Cost Balance of Generative Model-assisted Offline Safe RL

**arXiv ID:** 2602.05323 | [PDF](https://arxiv.org/pdf/2602.05323v1)

**作者:** Zifan Liu `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 84370 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Goal‑Assisted Stitching (GAS) 算法，针对离线安全强化学习（OSRL）中生成模型方法缺乏轨迹拼接与奖励‑约束平衡的问题，改进数据处理、目标函数学习与策略优化。

**💡 创新点**

创新点包括：① 在转移层级进行返回重标记与时间段返回增强（TSRA），提升可拼接子轨迹；② 用期望回归(expectile regression)学习奖励和成本目标函数，避免“幸运”样本偏差；③ 对数据集进行重塑以实现奖励‑成本返回均衡；④ 将估计的目标函数作为中间目标，引导策略通过受限优势加权回归 (constrained AWR) 进行优化。

**🔧 技术方法**

核心技术：生成模型（Decision Transformer 结构）、期望回归、目标函数驱动的受限 AWR、转移层返回重标记与时间段增强、数据集重塑与采样策略。

**📊 数据集**

实验使用 Bullet‑Safety‑Gym 与 Safety‑Gymnasium 两大基准，数据来源为 DSRL 公开离线数据集（按 D4RL 格式），共覆盖 15 个任务与 12 场景。

**📈 对比分析**

与 8 个基线（CPQ, COptiDICE, WSAC, VOCE, CDT, FISOR, CAPS, CCAC）比较。GAS 在紧约束条件下实现最佳安全性能并获得最高奖励；在宽约束下获得最优奖励；在不同阈值的零样本适配测试中，GAS 始终保持安全且优于 CDT；对不平衡的奖励‑成本目标具有较强鲁棒性。消融实验表明期望回归与 TSRA 对拼接能力和性能提升均起决定性作用。

**⚠️ 局限性**

限制：注意力模块在捕获真实时序依赖上表现欠佳，导致拼接能力受限；GAS 仅通过在安全数据中挑选优质子段来拼接，未能构造跨安全与不安全数据的新安全轨迹；在高记忆需求与拼接深度之间存在权衡。

---

## 208. A Design Space for Live Music Agents

**arXiv ID:** 2602.05064 | [PDF](https://arxiv.org/pdf/2602.05064v1)

**作者:** Yewon Kim `[一作]` (Carnegie Mellon University), Chris Donahue `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3166 | [OpenAlex ID](https://openalex.org/A5019674079)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述和视频分析，构建了一个统一的设计空间，用以描述和分析跨HCI、AI与计算机音乐领域的现场音乐代理系统，覆盖使用场景、交互方式、技术实现和生态环境四大维度；

**💡 创新点**

创新点在于首次将三学科的研究成果整合为一套可视化、可扩展的设计空间，提供交叉学科共享词汇、洞察趋势与空白，并公开发布为可迭代的“活”资源；

**🔧 技术方法**

采用系统性文献检索、视频采样、开放式编码与轴向编码的方法论来构建设计空间，并用结构化表格与交互可视化工具呈现结果；

**📊 数据集**

数据来源包括：1）核心关键论文集；2）通过布尔搜索在HCI、AI和计算机音乐会议中检索得到的论文；3）基于关键词在YouTube上筛选的现场演示和实践视频；共计约180余个系统；

**📈 对比分析**

通过将各系统映射到设计空间的维度与子维度，作者对比了技术侧重点（如适配方式、延迟优化）、交互特征（输入/输出模态）以及使用场景（流派、角色），揭示了技术普及、伦理讨论不足等趋势；虽然未给出数值性能评估，但定性比较为未来设计提供了方向；

**⚠️ 局限性**

局限性包括：①仅约10%文献进行双重编码，剩余部分单编码可能存在主观偏差；②样本主要来自英文会议与YouTube，可能遗漏非公开或商业工具及非英语研究；③设计空间以当前技术定义为界限，未来新兴技术与范式需要进一步扩充与更新。

---

## 209. Visual concept ranking uncovers medical shortcuts used by large multimodal models

**arXiv ID:** 2602.05096 | [PDF](https://arxiv.org/pdf/2602.05096v1)

**作者:** Joseph D. Janizek `[一作]` (Stanford University), Roxana Daneshjou `[通讯]` (Stanford University)

**通讯引用:** 5253 | [OpenAlex ID](https://openalex.org/A5014300312)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并验证了一种Visual Concept Ranking（VCR）方法，用于识别大规模多模态模型在特定任务中依赖的重要视觉概念。

**💡 创新点**

通过将视觉语言模型用于概念标注、线性回归学习概念激活向量，再利用梯度方向导数衡量模型敏感度，从而实现对模型的因果解释，并在分布漂移下保持鲁棒性。

**🔧 技术方法**

使用CLIP等视觉语言模型进行概念标注，线性回归生成CAV，梯度和方向导数计算重要性，bootstrap与t检验做显著性检验；实验还包括合成数据、医学影像与自然图像。

**📊 数据集**

合成视觉概念数据集、Diverse Dermatology Images（DDI）、CheXpert胸部X光数据、Imagenette自然图像等。

**📈 对比分析**

与纯相关性方法MA‑MONET相比，VCR在合成数据上与真实干预效果相关性达 r>0.6；在分布漂移场景下，VCR 92%准确识别伪特征，而 MA‑MONET 仅 18%；在皮肤病诊断任务中揭示模型对蓝紫色标记等 shortcut 的依赖，实验验证。

**⚠️ 局限性**

需要模型梯度访问，无法直接用于仅提供 API 的商业模型；对空间位置概念表现不佳；概念标签的语义可能不完全准确，需人工解释验证。

---

## 210. Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks

**arXiv ID:** 2602.05125 | [PDF](https://arxiv.org/pdf/2602.05125v1)

**作者:** William F. Shen `[一作]` (Meta Superintelligence Labs), Ilias Leontiadis `[通讯]` (Meta Superintelligence Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出递归Rubric Decomposition框架，自动细化、过滤并加权rubrics，用于提升LLM judge的判别精度和RFT奖励质量。

**💡 创新点**

创新点包括：① 递归拆分高层rubric以获得更完整的评判维度；② 引入正向边和相关性上界的理论保证rubrics信息量最大、冗余最小；③ 使用误差过滤和白化均匀权重消除偏差和相关噪声。

**🔧 技术方法**

技术手段：LLM prompt-based rubric生成、递归拆分循环、误差过滤、相关性过滤、白化权重优化；RFT采用GRPO算法；评估使用JudgeBench、PPE、BiGGen Bench、HealthBench-Hard等基准。

**📊 数据集**

使用的数据集包括：JudgeBench、Preference Proxy Evaluation (PPE)、WildChat（RFT训练）、BiGGen Bench（多模态生成评测）和HealthBench-Hard（临床对话评测）。

**📈 对比分析**

与无rubric基线、LLM Rubrics、Chasing the Tail等对比，在JudgeBench上GPT‑4o准确率从55.6%提升至73.3%（+17.7%），Llama‑3.1‑405B提升6.6点；在RFT奖励上，_WU版奖励提升约150–160%（Qwen3‑4B）和55–60%（Llama3.1‑8B）；最终策略在BiGGen Bench上达到82.8%/71.1%，在HealthBench‑Hard上表现最佳，显著优于所有基线。

**⚠️ 局限性**

局限性：① 递归拆分过程依赖LLM生成质量，若LLM误差大可能导致过度拆分或噪声累积；② 终止阈值和过滤标准需手工调参，缺乏自动化；③ 仅在英语及特定任务上验证，跨语言、跨模态泛化性尚未充分评估；④ 对低质量样本或极端多模态任务的鲁棒性尚需进一步研究。

---

## 211. Learning Context Matters: Measuring and Diagnosing Personalization Gaps in LLM-Based Instructional Design

**arXiv ID:** 2602.04972 | [PDF](https://arxiv.org/pdf/2602.04972v1)

**作者:** Johaun Hatchett `[一作]` (Rice University), Richard G. Baraniuk `[通讯]` (Rice University)

**通讯引用:** 51716 | [OpenAlex ID](https://openalex.org/A5072713767)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于策略层面的诊断框架，用合成的心理测量学学习情境来量化大语言模型（LLM）在教学设计中的个性化程度，并对比其与专家教师的策略偏差。

**💡 创新点**

创新点在于：①将心理测量学（MSLQ）数据生成逼真的学习情境作为实验探针；②通过“策略偏差”和“学习者中心度”两种指标客观衡量LC对LLM策略的影响；③引入相关性-影响性诊断，揭示模型对学习者特征的偏优或偏误使用。

**🔧 技术方法**

技术手段包括：多元正态建模生成学习者特征；因子负载映射生成条目响应；LLM提示重构为自然语言学习情境；多次采样估计策略分布；总变差距离（TVD）与学习者中心度计算；留一法相关性-影响性分析。

**📊 数据集**

数据集主要包括：来自MSLQ的学习者特征统计（均值、方差、相关矩阵）；OpenStax 计算学教材的学习目标；两位计算学专家的策略与特征相关性标注；LLM（GPT‑5.2）在不同情境下生成的策略采样。

**📈 对比分析**

比较方法：在“无情境”和“有情境”两种提示下分别估计LLM策略，并用TVD衡量与专家策略的偏差；计算学习者中心度评估个性化程度；使用留一法将特征重要性与专家相关性对比，绘制四象限图。结果显示：提供情境可显著降低LLM与专家策略的TVD（平均约10%以上），但学习者中心度仍低于专家；相关性-影响性分析揭示存在被忽视的高相关特征和对无关特征的“幻觉”影响。

**⚠️ 局限性**

局限性包括：①实验仅针对计算学单一学科，结果可能不具普适性；②留一法只能捕捉边际影响，忽略特征交互；③仅评估设计阶段策略，未涉及执行过程；④合成情境虽心理测量学一致，但与真实学习者的多样性仍有差距。

---

## 212. Locas: Your Models are Principled Initializers of Locally-Supported Parametric Memories

**arXiv ID:** 2602.05085 | [PDF](https://arxiv.org/pdf/2602.05085v1)

**作者:** Sidi Lu `[一作]` (Tencent AI Lab), Dong Yu `[通讯]` (Tencent AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Locas 框架，在推理时通过侧向 FFN 方式动态扩展模型容量，实现高效的参数和计算级别的自适应记忆。

**💡 创新点**

创新点在于利用主干模型的激活与梯度进行原则化初始化，构造可聚焦的关键-值记忆槽，同时通过非线性 SVD 等压缩算法保持低维表示。

**🔧 技术方法**

采用基于 Transformer 的 FFN 解释为软查找表的内存模块，设计 Locas‑MLP 与 Locas‑GLU 两种变体，结合权重裁剪、输出缩放和 Top‑K 激活基准克隆等技术。

**📊 数据集**

在 PG‑19 整本书语言建模数据集和 LoCoMo 长上下文对话问答数据集上进行实验。

**📈 对比分析**

与上下文截断、长上下文注意力以及 TempLoRA 等基线对比，Locas‑GLU 在相同或更小参数量（仅 15%‑25% 的额外参数）下实现相当或更低的困惑度和更高的问答 F1，且在 MMLU 上的灾难性遗忘幅度低至 0.1%–0.2%。

**⚠️ 局限性**

局限性包括对激活选择的依赖、GLU 结构的兼容性限制、非线性 SVD 的高计算开销以及在极大模型或更长序列下的可扩展性和稳定性仍需进一步验证。

---

## 213. Reducing the Costs of Proof Synthesis on Rust Systems by Scaling Up a Seed Training Set

**arXiv ID:** 2602.04910 | [PDF](https://arxiv.org/pdf/2602.04910v1)

**作者:** Nongyu Di `[一作]` (Nanjing University), Xiaoxing Ma `[通讯]` (Nanjing University)

**通讯引用:** 2342 | [OpenAlex ID](https://openalex.org/A5041674680)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过设计数据合成管道，生成了600多万条Rust程序的Verus验证数据，并用其微调Qwen2.5‑Coder‑32B‑Instruct模型实现低成本的程序证明生成。

**💡 创新点**

创新点在于结合自我合成、自研教程驱动合成和长链路思维（CoT）合成三种方法，突破了传统数据规模受限、功能覆盖不足、推理复杂度低的瓶颈。

**🔧 技术方法**

核心技术包括LLM自我生成（self‑synthesis）、教程引导合成（tutorial‑based synthesis）、Agent轨迹采样（agent trajectory synthesis）以及基于SimHash的去重与Verus验证/调试流程。

**📊 数据集**

使用的主要数据集为SAFE（≈10K Verus验证程序）、AlphaVerus、VeruSAGE、以及自合成得到的600+万条程序，其中包含约4,557条CoT示例。

**📈 对比分析**

与o4‑mini、Claude Sonnet 4.5及在AlphaVerus/Safe数据上微调的Qwen模型比较，Finetuned Qwen2.5‑Coder‑32B‑Instruct在VerusBench和VeruSAGE‑Bench上准确率分别提升至≈75%/83%，且单任务成本仅为$0.61，相比Sonnet 4.5的$8.04下降≈13×。

**⚠️ 局限性**

局限性包括仍需依赖大型LLM生成初始合成，CoT数据量有限，模型在复杂系统任务的准确率仍低于顶级商业模型，且缺乏对工具使用（如Verus交互）能力的进一步提升。

---

## 214. UniTrack: Differentiable Graph Representation Learning for Multi-Object Tracking

**arXiv ID:** 2602.05037 | [PDF](https://arxiv.org/pdf/2602.05037v1)

**作者:** Bishoy Galoaa `[一作]` (Northeastern University), Sarah Ostadabbas `[通讯]` (Northeastern University)

**通讯引用:** 2273 | [OpenAlex ID](https://openalex.org/A5031787107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为UniTrack的图论式损失函数，能够在训练阶段统一优化检测精度、身份保持和时空一致性，提升多目标跟踪性能；

**💡 创新点**

创新点在于将图结构与可微分损失结合，形成统一的端到端可训练目标，且通过自适应加权机制动态调节空间与时间一致性的重要性；

**🔧 技术方法**

采用可微分图表征学习、流网络约束、空间-时间一致性损失、拉普拉斯谱自适应加权等技术；

**📊 数据集**

在MOT17、MOT20、SportsMOT、DanceTrack等四大公开基准数据集上进行评估；

**📈 对比分析**

与Trackformer、MOTR、FairMOT、ByteTrack、GTR、MOTE等七种主流跟踪框架进行对比，结果显示在所有模型上均提升MOTA、IDF1和HOTA，最大可达9.7% MOTA、12.3% IDF1的提升；

**⚠️ 局限性**

局限性包括仅针对单摄像头跟踪、训练时引入约5%的显存开销及O(n²t)的计算复杂度，且在极端密集场景下可能受限于计算规模；

---

## 215. PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction

**arXiv ID:** 2602.05190 | [PDF](https://arxiv.org/pdf/2602.05190v1)

**作者:** Ju Shen `[一作]` (University of Dayton), Vijayan K. Asari `[通讯]` (University of Dayton)

**通讯引用:** 9984 | [OpenAlex ID](https://openalex.org/A5061050831)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出PoseGaussian框架，利用人体姿态指导的高斯投影实现实时高保真动态人体新视角合成。

**💡 创新点**

创新点在于将姿态信息同时嵌入几何深度估计和特征解码，配合Temporal Pose Stabilizer实现时序一致性，并保持2D高斯原始优势。

**🔧 技术方法**

采用3D Gaussian Splatting、卷积特征编码、姿态编码器、GRU深度更新、光度+LPIPS损失等技术。

**📊 数据集**

训练与评估使用ZJU-MoCap、THuman2.0、HuMMan、DNA-Rendering、People-Snapshot等公开动态人体数据集。

**📈 对比分析**

与多种SOTA方法（如GPS-Gaussian、InstantNVR、HumanNeRF等）对比，PSNR/SSIM/LPIPS均达或超越最高水平，同时实现100 FPS实时渲染。

**⚠️ 局限性**

局限在于对极端视角偏移或极快动作仍易出现细节模糊与漂移，且对多人交互场景尚未扩展。

---

## 216. GreekMMLU: A Native-Sourced Multitask Benchmark for Evaluating Language Models in Greek

**arXiv ID:** 2602.05150 | [PDF](https://arxiv.org/pdf/2602.05150v1)

**作者:** Yang Zhang `[一作]` (Ecole Polytechnique), Michalis Vazirgiannis `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了首个大规模、原生希腊语来源的多任务语言理解基准 GreekMMLU，包含 21,805 道多项选择题。

**💡 创新点**

创新点在于使用正式教育与专业考试的原生希腊语题目，提出 45 个主题层级与四个教育难度等级，并首次聚焦希腊文化特有科目。

**🔧 技术方法**

采用自动 OCR + LLM 校正、Unicode 规范化、专家人工审核等技术完成数据清洗与标注，并使用 lm-evaluation-harness 在零样本与五样本提示下评估模型。

**📊 数据集**

数据集来源于希腊学术、专业与政府考试的公开题库，公开子集 16,857 题，私有子集 4,948 题，用于官方排行榜。

**📈 对比分析**

对 80+ 开放与闭源 LLM 进行零样本与五样本比较，发现闭源前沿模型（如 Gemini 3 Flash、ChatGPT）平均准确率达 86–93%，而最强开放模型约 79%；模型规模、指令调优、希腊/欧洲专属训练均显著提升性能，且更大模型和指令调优对希腊文化题目表现尤为明显。

**⚠️ 局限性**

局限在于仅包含多项选择题，缺乏开放式生成、交互推理或口语表达；仅覆盖标准现代希腊语，无法体现方言与非正式用法；不含多模态内容；且部分题目可能与模型预训练语料重叠。

---

## 217. AudioSAE: Towards Understanding of Audio-Processing Models with Sparse AutoEncoders

**arXiv ID:** 2602.05027 | [PDF](https://arxiv.org/pdf/2602.05027v1)

**作者:** Georgii Aparin `[一作]` (Huawei Noah's Ark Lab), Irina Piontkovskaya `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 Whisper 与 HuBERT 的编码器层激活进行稀疏自编码器（SAE）训练，评估其稳定性、可解释性，并利用 SAFeatures 对 Whisper 进行干预以降低幻听，并与 EEG 进行相关性分析。

**💡 创新点**

首次在大规模音频模型上系统应用 SAE，提出跨层、跨模型的特征覆盖度评估指标，展示 SAE 能有效分离语音、音乐、环境声以及音素、情感等细粒度概念，并证明其在降低幻听和 EEG 对齐上的实用价值。

**🔧 技术方法**

使用 Batch-Top‑k 稀疏自编码器架构、L2 重构损失、分布式语义相似度指标、Top‑k 探测、特征消除（unlearning）、方向向量干预（steering）以及线性时间响应函数（TRF）进行特征评估和 EEG 对齐。

**📊 数据集**

利用约 2.8k 小时多模态语音数据集（LibriSpeech、LibriTTS、FSD50k、Musan、Wham 等），覆盖语音、音乐和环境声，训练 SAEs 并在多任务分类和语音识别上评估。

**📈 对比分析**

通过特征覆盖度、冗余度、重构误差、分类 Top‑k 效果、幻听 FPR 与 WER 的对比，证明 SAEs 在不同随机种子下稳定覆盖率>50%，重构误差低，Top‑k 10–150 能捕获大部分任务信息，且 SAE 导向可将幻听 FPR 降低 70% 而 WER 仅提升 0.4% 级别。

**⚠️ 局限性**

受限于仅评估 base/small 版本模型、缺少更大规模或其他 SSL 模型、特征可解释性依赖于音频字幕模型限制、EEG 仅使用单个通道且线性模型，未来需扩展任务、模型、自动解释和脑成像方法。

---

## 218. A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture

**arXiv ID:** 2602.04911 | [PDF](https://arxiv.org/pdf/2602.04911v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 219. Affordance-Aware Interactive Decision-Making and Execution for Ambiguous Instructions

**arXiv ID:** 2602.05273 | [PDF](https://arxiv.org/pdf/2602.05273v1)

**作者:** Hengxuan Xu `[一作]` (Tsinghua University), Tao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 81317 | [OpenAlex ID](https://openalex.org/A5100375748)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一套名为 AIDE 的双流框架，利用多阶段推理和加速决策实现机器人在陌生环境下对歧义指令的交互式探索与实时闭环执行。

**💡 创新点**

核心创新在于：① 双流 MSI–ADM 结构，将多模态 Chain‑of‑Thought 与 affordance‑aware 关系空间相结合；② 基于 affordance 向量的高效检索与 k‑means 聚类；③ 交互式可见/不可见探索策略，实现零-shot affordance 理解与动态决策。

**🔧 技术方法**

采用 GPT‑5 的多模态 CoT 进行语义理解；YOLO‑World + SAM2 负责目标检测与分割；MobileCLIP 用于多模态相似度匹配；k‑means 与 t‑SNE 处理 affordance 向量；DFS 搜索与阈值匹配实现实时检索；交互式探索策略控制机器人运动。

**📊 数据集**

数据集包括：生成场景集（DALL‑E 3）、真实场景集（网络收集）、增广场景集（ImageNet 物体+干扰物）；Instruction‑Tool 关系空间共 1,368 条样本；G‑Dataset 与 R‑Dataset 各 200 组 instruction‑scene 对；真实实验采用 6 条歧义指令（如“我渴了”“我要砸核桃”等）。

**📈 对比分析**

与 8 个基线（Molmo、Qwen3‑VL、GPT‑5、Magma、Robopoint、SayCan、MOKA、IntroPlan）在 G‑Dataset 与 R‑Dataset 上对比：AIDE 任务成功率 80%+、工具选择 95%+、无工具场景 60%+、实时率 ≈10 Hz，显著优于所有基线，且在探测与执行效率上领先。

**⚠️ 局限性**

局限性包括：对极端遮挡或极不常见的 affordance 仍可能失败；依赖 GPT‑5 与大模型，算力与能耗高；实验多聚焦室内单机器人场景，复杂动态环境的泛化与鲁棒性待进一步验证。

---

## 220. Adaptive Exploration for Latent-State Bandits

**arXiv ID:** 2602.05139 | [PDF](https://arxiv.org/pdf/2602.05139v1)

**作者:** Jikai Jin `[一作]` (Stanford University), Congshan Zhang `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一类基于隐藏状态的多臂老虎机（bandit）算法，通过使用过去的动作-奖励对以及有策略的探测（probing）来推断未观测的时间变换状态，并实现对动态最优策略的逼近。

**💡 创新点**

创新点：
1) 将隐藏状态信息通过延迟上下文（lagged context）与状态指纹（state fingerprint）嵌入到上下文特征中，实现无显式状态建模的状态推断；
2) 设计了两种探测范式（随机探测 RP-UCB、顺序探测 SP-UCB）以及自适应探测机制（AdaRP-UCB、AdaSP-UCB），三种门控（残差门、不确定性门、时滞门）协同决定探测时机，显著提升在状态跳变时的适应性；
3) 在实验中系统评估不同状态数、转移概率、噪声与时间尺度下的算法表现，给出实用的算法选择指南。

**🔧 技术方法**

技术：
- 线性上下文UCB（LinUCB）作为基础框架；
- 延迟上下文特征（(a_{t-1}, r_{t-1})）与指纹特征（(r_0^{fp}, r_1^{fp})）拼接；
- 探测策略：随机并行探测（RP-UCB）、顺序探测（SP-UCB）；
- 自适应探测门控：残差门、置信区间门、时滞门；
- 经典基线：UCB1、TS、SW-UCB、D-UCB、EXP3、EXP3-S、LC-TS 等。

**📊 数据集**

数据集：完全合成实验环境，双臂老虎机，隐藏状态数量 S∈{2,10,20,50}，状态转移矩阵为马尔可夫链（自转概率 p_{stay}∈{0.5,0.8,0.9,0.95,0.99}），奖励服从 N(μ_{s,a},σ²)（σ∈{0.01,0.05,0.1,0.5}），时间长度 T∈{500,1000,5000,20000}。每个配置随机生成 128 个实例并做 5 次独立跑。

**📈 对比分析**

比较方法：与经典上下文与非稳定 bandit 算法（UCB1、TS、SW-UCB、D-UCB、EXP3、EXP3-S、LC-TS、LC-UCB、SP-UCB）对照；评价指标为累计 regret 及赢率（在每一配置下获得最低 regret 的算法比例）。实验结果显示：
- AdaRP-UCB 在 12/13 配置下击败 RP-UCB，尤其在状态跳变频繁时表现突出；
- AdaSP-UCB 在低噪声、慢混合或大状态空间时最优；
- LC-UCB 在快混合时表现最好；
- 当奖励噪声极大（σ=0.5）时，AdaRP/ AdaSP、LC 系列退化，D-UCB 与 SW-UCB 取得最好表现。

**⚠️ 局限性**

局限性：
- 由于隐藏状态不可观测，无法达到零平均 regret，算法只能逼近最优策略；
- 高奖励噪声或弱状态指纹时，探测成本高且收益低，导致性能退化；
- 探测策略与门控阈值需要经验调参；
- 只考虑马尔可夫链状态转移，若真实环境非马尔可夫或存在动作影响状态，则不适用；
- 计算量相对传统 UCB 较小，但在多臂或高维上下文时仍可能面临维度灾难。

---

## 221. E.M.Ground: A Temporal Grounding Vid-LLM with Holistic Event Perception and Matching

**arXiv ID:** 2602.05215 | [PDF](https://arxiv.org/pdf/2602.05215v1)

**作者:** Jiahao Nie `[一作]` (Nanyang Technological University), Shijian Lu `[通讯]` (Nanyang Technological University)

**通讯引用:** 16489 | [OpenAlex ID](https://openalex.org/A5023507910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出E.M.Ground，一种基于Vid-LLM的全局事件匹配模型，实现精确的Temporal Video Grounding。

**💡 创新点**

引入单一<evt>事件令牌聚合全时段语义，使用Savitzky–Golay平滑减少噪声，并融合多尺度视觉特征以补偿压缩损失。

**🔧 技术方法**

利用多层视觉特征聚合、单一事件令牌匹配、Savitzky–Golay平滑、LLM（Phi‑3 Mini）+ Q‑Former+ ViT‑G/14 视觉编码器等技术。

**📊 数据集**

在Charades‑STA、E.T.Bench、YouCook2、QVHighlights、MVBench等数据集上进行评测。

**📈 对比分析**

与TRACE、E.T.Chat等SOTA Vid‑LLM 在零样本设置下对比，mIoU/Recall 提升4.5%/13.4%，在多目标、长时段等场景保持稳定且显著优于大模型。

**⚠️ 局限性**

仍受限于视觉压缩导致信息损失，模型对极短/极长事件精度有待提升，需进一步优化多尺度融合与实时推理。

---

## 222. Artificial Intelligence as Strange Intelligence: Against Linear Models of Intelligence

**arXiv ID:** 2602.04986 | [PDF](https://arxiv.org/pdf/2602.04986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 223. Accelerated Sequential Flow Matching: A Bayesian Filtering Perspective

**arXiv ID:** 2602.05319 | [PDF](https://arxiv.org/pdf/2602.05319v1)

**作者:** Yinan Huang `[一作]` (Georgia Institute of Technology), Pan Li `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 10201 | [OpenAlex ID](https://openalex.org/A5100455171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Sequential Flow Matching框架，将流式推断视为贝叶斯滤波，通过前一时刻后验作为源分布进行概率流匹配，显著减少采样步骤；

**💡 创新点**

创新点在于用贝叶斯滤波的递归结构构造时间相关的源-目标耦合，理论证明重启高斯比从上一步后验更低采样误差，并引入重噪声机制与预训练微调实现稳定高效推断；

**🔧 技术方法**

使用概率流匹配（Flow Matching）、扩散模型、ODE求解、贝叶斯滤波、重噪声机制、预训练-微调策略；

**📊 数据集**

实验数据集包括1D Burgers方程、WeatherBench2天气预测、D4RL迷宫规划、Navier‑Stokes烟雾控制以及Lorenz系统状态估计；

**📈 对比分析**

与一致性模型、MeanFlow、温启动扩散、异步去噪等基线比较，实验表明仅需1–3步采样即可匹敌全步扩散在RMSE、能量分数和规划性能上的表现，并大幅降低采样延迟；

**⚠️ 局限性**

局限性在于依赖预训练–微调流程，缺乏从零开始的无仿真训练；重噪声水平需手动调节，在高度不确定性环境中表现可能不稳定。

---

## 224. A Causal Perspective for Enhancing Jailbreak Attack and Defense

**arXiv ID:** 2602.04893 | [PDF](https://arxiv.org/pdf/2602.04893v1)

**作者:** Licheng Pan `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35146 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Causal Analyst框架，利用LLM编码和图神经网络进行因果发现，识别并利用Prompt特征对LLM进行攻击与防御；

**💡 创新点**

创新点在于首次从因果视角分析LLM越狱机制，构建可解释的因果图，并将因果洞察直接用于攻击增强和防御提升；

**🔧 技术方法**

采用LLM（Qwen2.5-7B）进行prompt编码，使用DAG‑GNN进行因果图学习，结合多任务训练、特征融合与对齐；

**📊 数据集**

使用了35k条由100个模板和50个有害查询构成的跨七款LLM（Qwen、Baichuan2、LLaMA3、GLM4、GPT‑4o等）的越狱尝试数据集，并标注了37个可读prompt特征；

**📈 对比分析**

与传统关联式方法、PC、DirectLiNGAM以及无因果基础的基线（Vanilla Extractor）对比，因果方法在攻击成功率、意图提取BLEU/ROUGE、图学习损失等指标上均优于非因果方法；

**⚠️ 局限性**

局限性包括：仅针对模板式攻击，未覆盖梯度或优化攻击；手动调参缺乏自动化；模型对新型多模态攻击的泛化未知；

---

## 225. Evaluating Kubernetes Performance for GenAI Inference: From Automatic Speech Recognition to LLM Summarization

**arXiv ID:** 2602.04900 | [PDF](https://arxiv.org/pdf/2602.04900v1)

**作者:** Sai Sindhur Malleni `[一作]` (Red Hat), André Bauer `[通讯]` (Illinois Institute of Technology)

**通讯引用:** 2032 | [OpenAlex ID](https://openalex.org/A5063155276)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

结合 Kueue、DAS 与 GAIE 在 Kubernetes 上实现多阶段生成式 AI 推理流水线（语音转写+文本摘要）

**💡 创新点**

创新点是将队列调度、动态 GPU 切分与推理路由三者整合为统一的端到端 Kubernetes 原生工作流，提升吞吐量与响应时间

**🔧 技术方法**

使用 Kueue 进行批量作业排队与优先级/抢占，DAS 进行 NVIDIA MIG 切片动态分配，GAIE+llm-d 对多节点 LLM 进行智能路由，辅以 Whisper、vLLM、Qwen3-8B、OpenShift ROSA、kube-burner 与 GuideLLM

**📊 数据集**

采用公开的 Earnings 22 企业财报电话记录数据集（共 32 条音频）进行 ASR 与摘要实验

**📈 对比分析**

通过 kube-burner 与 GuideLLM 进行基准，比较不使用 Kueue/DAS/GAIE 的基线，结果表明 Kueue 可使作业完成时间缩短 15%，DAS 使平均作业完成时间降低 36%，GAIE 将首 token 时延降低 82% 并将整体响应时延缩短 6 秒

**⚠️ 局限性**

局限性包括缺乏 Kueue 与 DAS 的原生集成、仅验证了 Whisper 与 Qwen3-8B 两类模型、实验仅在单一 A100 GPU 集群上进行、未覆盖多模态或实时流推理等场景

---

## 226. Certifiable Boolean Reasoning Is Universal

**arXiv ID:** 2602.05120 | [PDF](https://arxiv.org/pdf/2602.05120v1)

**作者:** Wenhao Li `[一作]` (University of Toronto), Dennis Zvigelsky `[通讯]` (McMaster University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种可训练的分布式神经网络架构，能够在任何参数配置下几乎必然生成满足 fan‑in 2、fan‑out 1 的布尔电路，并可通过参数调整实现对任意布尔函数的精确逼近，且在稀疏（log‑junta）情形下参数规模近似线性。

**💡 创新点**

创新点在于：① 通过随机边/门选择与软最大化构造全概率保证的布尔电路分布；② 给出该模型在任意布尔函数和稀疏布尔函数上的可达性理论与参数上界；③ 在实验中首次量化并对比了网络内部单元的布尔可表示性（BNR），验证了模型的可解释性。

**🔧 技术方法**

主要技术包括：可学习的随机布尔门采样（16种门的软最大化）；随机边采样与温度控制的多层结构；多层布尔电路递推定义；对参数空间的可微优化与温度退火；以及理论证明中的电路结构与概率论工具。

**📊 数据集**

使用的“数据集”是人工生成的全真值表（truth‑table completion）任务：从随机布尔公式得到所有 2^B 赋值的真值表，B 通常取 4–6 以便可视化。

**📈 对比分析**

与传统 ReLU MLP 在相同参数/神经元预算下进行对比：在 Exact Match（EM）上模型与 MLP 接近（≈0.99–1.0），但模型所有内部单元均满足 BNR=1，而 MLP 仅约 10–20% 单元可二值化；表明模型在保持竞争性能的同时提供了完整的电路层级可解释性。

**⚠️ 局限性**

局限性包括：① 仅证明了 fan‑in 2、fan‑out 1 的布尔电路；② 目前仅在小规模布尔函数（B≤6）上验证，尚未证明可扩展到更大规模或更复杂的符号任务；③ 对抗鲁棒性与迁移学习等实用安全性尚未深入探讨。

---

## 227. Denoising diffusion networks for normative modeling in neuroimaging

**arXiv ID:** 2602.04886 | [PDF](https://arxiv.org/pdf/2602.04886v1)

**作者:** Luke Whitbread `[一作]` (Australian Institute for Machine Learning), Mark Jenkinson `[通讯]` (South Australian Health and Medical Research Institute)

**通讯引用:** 119112 | [OpenAlex ID](https://openalex.org/A5004220534)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并验证了基于去噪扩散概率模型（DDPM）的多元规范化建模框架，可从条件密度中估计年龄/性别等协变量对应的IDP分位数与偏差得分。

**💡 创新点**

创新点在于：①将DDPM迁移到表格神经影像学特征，提供完整的多元条件分布；②引入SAINT风格自注意力变压器作为去噪器，显著提升高维（≤200维）下的校准与依赖结构捕获；③设计了面向规范化建模的综合评估体系（分位数误差、PIT、KS、两两距离与高阶依赖诊断、邻域记忆率）。

**🔧 技术方法**

使用去噪扩散模型（DDPM）结合两种去噪器：FiLM 线性调制的多层感知机和基于SAINT的自注意力变压器；对协变量采用嵌入或FiLM调制；训练采用标准的噪声预测损失，采样通过逆向马尔可夫链实现。

**📊 数据集**

在合成数据（四个IDP，包含异方差和多峰年龄效应）与真实数据（UK Biobank 20个与4个FreeSurfer IDP）上进行评估；实验覆盖2至200维的IDP集，并探索不同训练样本比例（10%–50%）。

**📈 对比分析**

与传统GAMLSS（SHASH）和独立的MLP去噪器对比。结果显示：①在低维（≤20）下，SAINT与MLP、GAMLSS的ACE、PIT、KS均可接受；②高维（100–200）时，SAINT保持良好校准与低KS拒绝率，而MLP出现覆盖饱和、PIT聚集与高KS；③在时间与采样成本上，SAINT在高维下训练/采样时间可控制在数小时内，显著优于GAMLSS的数千秒训练。

**⚠️ 局限性**

局限性包括：①MLP去噪器在高维下性能衰退；②模型受训练队列选择偏倚影响（UK Biobank的健康志愿者偏差）；③评估侧重二阶统计量，对更高阶依赖或互信息未作深入验证；④生成样本需大量采样才能估计分位数，计算成本仍高。

---

## 228. Among Us: Measuring and Mitigating Malicious Contributions in Model Collaboration Systems

**arXiv ID:** 2602.05176 | [PDF](https://arxiv.org/pdf/2602.05176v1)

**作者:** Ziyuan Yang `[一作]` (University of Washington), Yulia Tsvetkov `[通讯]` (University of Washington)

**通讯引用:** 5137 | [OpenAlex ID](https://openalex.org/A5062910836)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了多语言模型（LLM）协同系统在存在恶意模型时的性能衰退，并提出了两类缓解策略（无监督投票和外部监督）来恢复系统性能。

**💡 创新点**

创新点在于：①首次系统性评估了四类恶意模型（prompting、激活 steering、SFT、RL）对四级协同方法的影响；②提出了基于内部投票和外部监督的两种防御框架；③揭示了恶意模型对安全和推理任务影响最大。

**🔧 技术方法**

技术手段包括：构造恶意模型（提示、激活 steering、监督微调、逆奖励RL）；多级协同方法（API 路由、文本辩论、logit 聚合、权重合并）；监督器（LLM‑as‑a‑judge 与奖励模型）；实验框架 MoCo；使用激活向量提取、GRPO 等训练技巧。

**📊 数据集**

使用了十个公开基准数据集，覆盖安全、推理、知识、编程与指令遵循：CocoNot、SafetyBench、GSM8k、NLGraph、MMLU、TruthfulQA、HumanEval、DS-1000、IFBench、IFEval。

**📈 对比分析**

通过在原始无恶意模型的基准上注入恶意模型并与之对比，量化性能下降；再对比无监督和监督两种缓解方案，监督方案平均可恢复约95%原始性能；API‑级别最易受攻击，logit/权重级别表现最鲁棒。

**⚠️ 局限性**

局限性：①只考虑了四种恶意模型构造，未涵盖更复杂或自适应攻击；②实验聚焦于 API 与文本级别的协同，未评估 logit/权重级别的防御；③仅使用自动评测，缺乏人工评估；④攻击空间更广，结果可能无法完全推广到所有协同范式。

---

## 229. On QC and GQC algebraic geometry codes

**arXiv ID:** 2602.05097 | [PDF](https://arxiv.org/pdf/2602.05097v1)

**作者:** Matteo Bonini `[一作]` (Aalborg University), Francesco Ghiandoni `[通讯]` (University of Primorska)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一类新的准循环（QC）和广义准循环（GQC）代数几何码，利用Kummer扩张、超椭圆、规范-跟踪曲线以及Hermitian曲线的自同构群实现码的生成；

**💡 创新点**

突破了以往仅限于椭圆曲线的限制，提供了可调共索引（co-index）的代码构造，并给出了基于自同构群的显式参数公式；

**🔧 技术方法**

采用自同构群作用、Riemann–Roch理论、分离多项式曲线的几何性质以及AG码评估映射等技术；

**📊 数据集**

使用各类曲线在有限域上的有理点集合作为码字长度的自然“数据集”，如超椭圆曲线、规范-跟踪曲线、Hermitian曲线和其商曲线；

**📈 对比分析**

与传统AG码和椭圆曲线QC码相比，新码在共索引灵活性、距率平衡和渐近良好参数（如相对距离和码率均可调）方面表现更佳；

**⚠️ 局限性**

局限在于需先识别曲线的可计算自同构群、曲线参数受域大小限制，以及对高阶自同构群曲线的构造与分析复杂度较高。

---

## 230. Pruning Minimal Reasoning Graphs for Efficient Retrieval-Augmented Generation

**arXiv ID:** 2602.04926 | [PDF](https://arxiv.org/pdf/2602.04926v1)

**作者:** Ning Wang `[一作]` (Cornell University), Sainyam Galhotra `[通讯]` (Cornell University)

**通讯引用:** 1060 | [OpenAlex ID](https://openalex.org/A5038532934)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 AutoPrunedRetriever，一种基于图的检索增强生成系统，能够在多轮查询中保留并增量扩展最小推理子图，只使用符号化的实体-关系编码而非原始文本。

**💡 创新点**

创新点包括：①本地增量构建图，避免全局重构；②路径中心检索，将推理链视为检索单元并直接评分；③精确符号重用，通过 ID 代码书实现零冗余上下文；④双层实体合并（近似最近邻 + k‑means）保持图规模；⑤DPO 自适应压缩策略，根据查询复杂度动态控制提示长度。

**🔧 技术方法**

采用技术包括：REBEL 或 LLM 作为三元组抽取器；ID‑索引的实体/关系代码书；粗到细的路径检索（先在符号空间召回，再对三元组细粒度重排）；两层实体合并策略；DPO 学习的压缩策略；整体图检索+提示生成管线。

**📊 数据集**

实验使用 GraphRAG-Benchmark（Medical、Novel）、HotpotQA 派生的 STEM 与 TV 语料以及其他 GraphRAG 任务集（Fact Retrieval、Complex Reasoning、Contextual Summarize、Creative Generation）。

**📈 对比分析**

与 HippoRAG2、LightRAG、Fast‑GraphRAG、RAG 等基线对比，AutoPrunedRetriever‑REBEL/llm 在复杂推理任务上平均提升约 10–11 分，取得 state‑of‑the‑art；同时在 STEM/TV 上使用 1–2 个数量级更少的 token，延迟显著降低，保持与基线相当甚至更优的准确率。

**⚠️ 局限性**

局限性包括：仅在英文文本数据上验证，未考察多语种或嘈杂用户日志；依赖三元组抽取器，抽取错误会直接影响推理；未与抽取器联合训练；目前仅支持单轮文本问答，缺乏多模态、工具使用或人机交互的支持。

---

## 231. Reaching Univalency with Subquadratic Communication

**arXiv ID:** 2602.05356 | [PDF](https://arxiv.org/pdf/2602.05356v1)

**作者:** Andrew Lewis-Pye `[一作]` `[通讯]`, Andrew Lewis-Pye

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文将拜占庭共识拆解为两阶段：实现单值性（univalency）与信息传播（dissemination），并证明在无认证下单值性可用 O(f log f) 消息/通信，认证下可用 O(n log n)，从而说明 Dolev‑Reischuk 的二次下界主要来源于传播成本。

**💡 创新点**

创新点在于首次将拜占庭协议的通信成本拆解、提出 ϵ‑BA（允许少量错误输出）与可提取 BA（collectively deterministic）两种松弛协议，并利用递归阶段王（Recursive Phase King）结合概率/确定性采样实现子二次通信，完成全局共识的两阶段设计。

**🔧 技术方法**

关键技术包括递归阶段王协议、概率采样与 Chernoff/KL 边界、可提取共识定义、以及对采样选择的确定性化（derandomization）和多层递归结构。

**📊 数据集**

论文纯粹理论分析，不使用外部数据集；所有结果均来自算法设计与复杂度证明。

**📈 对比分析**

与传统的 Ω(f²) 下界对比，论文证明单值性阶段仅需 O(f log f) 或 O(n log n) 的上界，说明传播阶段导致二次成本；实验或数值比较均为理论上界与下界的对照，表明实现单值性成本远低于整体协议。

**⚠️ 局限性**

局限性包括：单值性阶段的下界仍为 Ω(f)，与 O(f log f) 上界之间存在对数级差距；方法仅适用于同步/完全同步模型，未扩展到异步或部分同步；并未给出更紧凑的下界或针对更高 Byzantine 容错率的分析。

---

## 232. Smoothness Errors in Dynamics Models and How to Avoid Them

**arXiv ID:** 2602.05352 | [PDF](https://arxiv.org/pdf/2602.05352v1)

**作者:** Edward Berman `[一作]` (Northeastern University), Robin Walters `[通讯]` (Northeastern University)

**通讯引用:** 24949 | [OpenAlex ID](https://openalex.org/A5063254454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了松弛单元卷积（Relaxed Unitary Convolution）框架，解决了传统单元卷积在动力学建模中导致的过度平滑问题。

**💡 创新点**

创新点在于理论证明单元卷积对角度依赖强的动力学系统过度约束，并设计了两种松弛方法（Taylor截断与编码器–解码器）以及将单元卷积推广到网格结构。

**🔧 技术方法**

使用了单元卷积、Taylor级数截断、编码器–解码器架构、网格Rayleigh商以及梯度下降训练等技术。

**📊 数据集**

实验数据集包括热扩散、波动、Cahn–Hilliard PDE的网格模拟以及WeatherBench2全球天气预报数据。

**📈 对比分析**

与GCN、Lie单元卷积、GemCNN、EAGLE、Hermes、EGNN等基线相比，在热扩散、波动、Cahn–Hilliard任务上实现了最低的NRMSE、SMAPE和RE，并在天气预报任务中接近SOTA的RMSE与ACC。

**⚠️ 局限性**

局限性在于对高角度依赖的动力学仍难以完全通过松弛实现，且在非扩散类动力学（如波动）上的优势不如扩散任务明显。

---

## 233. Wi-Fi Radar via Over-the-Air Referencing: Bridging Wi-Fi Sensing and Bistatic Radar

**arXiv ID:** 2602.05344 | [PDF](https://arxiv.org/pdf/2602.05344v1)

**作者:** Koji Yamamoto `[一作]` (Kyoto Institute of Technology), Koji Yamamoto `[通讯]` (Kyoto Institute of Technology)

**通讯引用:** 2397 | [OpenAlex ID](https://openalex.org/A5063642951)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用Wi‑Fi的射线图（CIR）和LoS路径作为OTA参考，实现了在未同步的COTS Wi‑Fi设备上进行相位保持的时延-多普勒分析。

**💡 创新点**

创新点在于：① 通过LoS路径提供时间和相位基准，消除传统雷达所需的有线参考或专用参考天线；② 在单天线单端口收发器上即可完成相位保持；③ 将雷达信号处理方法直接迁移到常见的Wi‑Fi感知场景，实现物理可解释的多普勒符号（可区分接近/远离）和亚波长位移测量。

**🔧 技术方法**

技术手段包括：802.11ax CSI抽取、频域预处理（外点剔除、DC相位插值、功率归一化）、黑曼窗口和零填充的IDFT生成CIR、LoS路径峰值检测与时延/相位校准、杂波平均去除、时间轴统一重采样、STFT得到时延‑多普勒图。

**📊 数据集**

实验数据集：使用Intel AX210芯片的两台ASUS NUC 13 Rugged‑Tall设备，160 MHz带宽，进行人类步态（往复直线）和呼吸（15 bpm）实验，配合VIVE Tracker 3.0或SteamVR Base Station 2.0进行基准位置信息。

**📈 对比分析**

对比方法：传统仅使用CFR幅度（或PCA降维后幅度）进行多普勒估计。实验表明，LoSRef方案可实现20 dB弱于静态多径的目标检测，恢复相位符号，获得精确的距离-速度轨迹；幅度基方法仅给出对称多普勒谱，无法区分方向。

**⚠️ 局限性**

局限性：① 需要LoS路径在CIR中占主导且可分辨；② 对Tx‑Rx距离保持恒定，环境变化会影响参考时延；③ 仅适用于提供复数CFR的COTS 802.11ax芯片（目前多为AX2xx系列）；④ 非均匀时间采样需手动重采样，若间隔变化过大会影响精度。

---

## 234. Consistency-Preserving Concept Erasure via Unsafe-Safe Pairing and Directional Fisher-weighted Adaptation

**arXiv ID:** 2602.05339 | [PDF](https://arxiv.org/pdf/2602.05339v1)

**作者:** Yongwoo Kim `[一作]` (Korea University), Donghyun Kim `[通讯]` (Korea University)

**通讯引用:** 20989 | [OpenAlex ID](https://openalex.org/A5100719069)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新的概念消除框架PAIRed Erasing，利用不安全-安全多模态配对实现一致性保持的语义重新对齐，以实现对目标概念的精准消除；

**💡 创新点**

创新点在于：①用视觉-文本配对的安全对应作为引导，使消除过程从单纯的无意义空洞映射转变为语义对齐；②设计Fisher加权的DoRA初始化（FiDoRA），在低秩适配器中仅更新对目标概念最敏感的方向，兼顾参数效率和结构一致性；

**🔧 技术方法**

使用了低秩适配（LoRA/DoRA）与Fisher信息权重、IP-Adapter视觉条件、Stable Diffusion U‑Net结构、IP-Adapter预训练、ImageGuard/ICEdit等技术；

**📊 数据集**

构建了安全-不安全成对数据集，包括针对裸露、艺术风格（如梵高）和目标物体的配对图像；

**📈 对比分析**

与Stable Diffusion、ESD、SPM、RECE、Co‑Erasing、AGE等基线对比，实验表明PAIR在裸露消除、艺术风格消除等任务上取得了最优的 Harmonic Mean，消除效果显著、生成质量与一致性均得到保持；

**⚠️ 局限性**

局限性包括：需要人工或自动化生成配对数据，生成安全对应可能不完美；目前仅针对单一概念消除，跨概念或多目标消除的适用性待验证；

---

## 235. NeuCLIRTech: Chinese Monolingual and Cross-Language Information Retrieval Evaluation in a Challenging Domain

**arXiv ID:** 2602.05334 | [PDF](https://arxiv.org/pdf/2602.05334v1)

**作者:** Dawn Lawrie `[一作]` (Johns Hopkins University), Luca Soldaini `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了一个针对技术中文文档的跨语言检索评估集合，包含近四十万条中文学术摘要及其机器翻译英文版本，提供110条查询和35,962条深度相关性判定。

**💡 创新点**

创新点在于将技术文档与跨语言检索结合，构建深度判定数据，提供多种最新神经检索基线，并提供跨语言检索首阶段融合基线，突破BM25局限。

**🔧 技术方法**

采用多种神经检索技术（单向双编码器、多向双编码器、稀疏学习等），并结合Qwen3-8B、PLAID‑X、MILCO三系统融合；还使用Google翻译生成英文字段。

**📊 数据集**

使用CSL（Chinese Scientific Literature）数据集约39.6万条期刊摘要，并机器翻译成英文；查询由22名研究生及博士后依据其研究领域生成。

**📈 对比分析**

通过与TREC NeuCLIR 2023/24 结果对比，评估单语和跨语言检索的nDCG@20、Recall等指标；Qwen3-8B在首阶段检索表现最佳，但跨语言任务中仍低于BM25，提示技术文档跨语言检索难度大。

**⚠️ 局限性**

局限在于相关性判定仅覆盖部分系统检索的前20/35文档，未检索到的相关文档可能被误评为非相关，导致评价指标低估；缺乏对非参与TREC队列系统的充分评估。

---

## 236. Pool-based Active Learning as Noisy Lossy Compression: Characterizing Label Complexity via Finite Blocklength Analysis

**arXiv ID:** 2602.05333 | [PDF](https://arxiv.org/pdf/2602.05333v1)

**作者:** Kosuke Sugiyama `[一作]` (Waseda University), Masato Uchida `[通讯]` (Waseda University)

**通讯引用:** 913 | [OpenAlex ID](https://openalex.org/A5000225743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种基于信息论的框架，将池式主动学习视为噪声有损压缩问题，利用有限块长分析得到标签复杂度和泛化误差下界。

**💡 创新点**

创新点在于将池式主动学习映射为噪声有损压缩，并首次将学习算法的过拟合度与归纳偏差匹配度纳入信息论下界的量化，同时与信息理论界限和稳定性理论建立了直接联系。

**🔧 技术方法**

主要使用信息理论、有限块长压缩分析、噪声有损压缩模型、倾斜信息等技术来推导下界。

**📊 数据集**

论文为理论分析，未使用具体数据集。

**📈 对比分析**

未进行实验比较，因其为理论界限推导，暂无性能指标。

**⚠️ 局限性**

局限在于计算复杂度高、缺少对归纳偏差匹配度指标的实际验证，以及对有限池大小下下界可计算性的验证不足。

---

## 237. MTPano: Multi-Task Panoramic Scene Understanding via Label-Free Integration of Dense Prediction Priors

**arXiv ID:** 2602.05330 | [PDF](https://arxiv.org/pdf/2602.05330v1)

**作者:** Jingdong Zhang `[一作]` (Texas A&M University), Xin Li `[通讯]` (Texas A&M University)

**通讯引用:** 38651 | [OpenAlex ID](https://openalex.org/A5100354056)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 MTPano，一种多任务全景场景理解基础模型，采用无标签训练管线，利用视角基础模型生成伪标签并投影回球面；设计了双流 PD‑BridgeNet，分别处理旋转不变与旋转变任务，并通过梯度截断桥接实现安全交互；同时引入 ERP Token Mixer 以及辅助稠密任务（梯度、边缘距离场、点图）提升跨任务学习。

**💡 创新点**

创新点包括：① label‑free 训练管线，将视角模型的高质量稠密先验投影回全景并做随机裁剪监督；② 将任务分为旋转不变/变两组并在网络中以双流形式解耦；③ 设计 ERP Token Mixer 以适应全景投影失真；④ 梯度截断桥接机制，实现特征交互同时避免负迁移；⑤ 通过多辅助稠密任务强化跨任务语义与几何一致性。

**🔧 技术方法**

使用技术包括：视角基础模型 InternImage‑H 与 MoGe‑2 生成伪标签；ViT + DINOv3 预训练 backbone；FiLM 进行几何调制；ERP Token Mixer 处理纬度失真；梯度截断桥接机制；辅助任务（图像梯度、边缘距离场、点图）与交叉注意力桥接；Zero‑Convolution 与 Progressive Warmup 训练策略；多任务损失与几何正则。

**📊 数据集**

训练数据为 140,884 张无标签全景图，来自 Structured3D、Sun360、Matterport3D 与 DiT360 生成合成图；评估数据为 Structured3D（合成）和 Stanford2D3D（实景）两个标准全景基准。

**📈 对比分析**

与单任务专家（SFSS‑MMSI、DAP、PanoNormal）以及多任务基线（InvPT、BridgeNet、TaskPrompter 等）进行对比。MTPano 在 Structured3D 上实现 mIoU 75.66% / Depth AbsRel 0.0248 / Normal mean 3.85°，均为当前 SOTA；在 Stanford2D3D 上达到 mIoU 69.47% / Depth AbsRel 0.0675 / Normal mean 9.71°，显著优于多任务基线并逼近或超过专用模型。

**⚠️ 局限性**

限制：仍依赖视角基础模型的先验，伪标签噪声对最终精度有影响；对极端纬度（极点）失真仍存在一定误差；训练需要大规模 GPU 资源，模型规模较大；未在更大多任务或多模态（光照、语义）设置下验证鲁棒性；对恶劣天气或光照变化的适应性尚待评估。

---

## 238. A Data Driven Structural Decomposition of Dynamic Games via Best Response Maps

**arXiv ID:** 2602.05324 | [PDF](https://arxiv.org/pdf/2602.05324v1)

**作者:** Mahdis Rabbani `[一作]`, Shima Nazari `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种异步结构化约简方法，将对手的最优性条件替换为离线编译的最佳响应映射，从而实现在缺乏对手完整信息时的 Nash 均衡求解。

**💡 创新点**

创新点在于用数据驱动的最佳响应逼近代替在线最优求解，并将其作为可行性约束嵌入 NLP，消除嵌套优化与导数耦合。

**🔧 技术方法**

采用最佳响应映射学习（MLP）、动态规划、NLP（IPOPT）求解、MCP、Monte Carlo 验证等技术。

**📊 数据集**

使用从两车赛道仿真生成的27,067条轨迹数据进行最佳响应模型训练，并在1,200个随机起始条件下进行测试。

**📈 对比分析**

与全信息的 DGSQP、IBR 基线相比，所提方法在70%的成功率、约1秒的中位求解时间和相近的自车成本，同时无需对手模型。

**⚠️ 局限性**

主要限制在于最佳响应逼近的误差导致可行性与安全性下降、求解失败多因不可行性、以及未针对实时闭环与模型不确定性做进一步稳健设计。

---

## 239. Wid3R: Wide Field-of-View 3D Reconstruction via Camera Model Conditioning

**arXiv ID:** 2602.05321 | [PDF](https://arxiv.org/pdf/2602.05321v1)

**作者:** Dongki Jung `[一作]` (University of Maryland), Suyong Yeon `[通讯]` (NAVER LABS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了前向神经网络 Wid3R，用于直接从宽视角相机（鱼眼、全景）图像实现 3D 重建。

**💡 创新点**

创新点在于：① 通过球谐系数的射线表示统一处理多种畸变相机；② 引入可学习的相机模型 token 作为条件提示；③ 在单一网络中完成深度、姿态与点云预测，无需后期优化。

**🔧 技术方法**

使用了射线‑球谐表示、Transformer 编码器、相机 token 作为提示、端到端多任务损失（点云、法线、姿态、射线、径向、置信度）以及数据增强策略。

**📊 数据集**

训练数据覆盖 9 个多视场景数据集：TartanAirV2、ASE、Hypersim、KITTI-360、360Loc、Matterport3D、ScanNet++、EDEN、VKITTI，包含针孔、鱼眼与 360° 相机。

**📈 对比分析**

与传统 SfM、单视深度估计以及最新多视前向模型（DUSt3R、Spann3R、FLARE、VGGT 等）进行对比，在 Stanford2D3D、Matterport3D、ScanNet++ 等基准上实现了最高 77.33% 的提升，且在 360° 场景下保持了卓越的零样本鲁棒性。

**⚠️ 局限性**

局限性在于：未显式建模动态场景；360° 数据标注稀缺，导致模型在此类数据上的泛化仍受限。

---

## 240. Once Correct, Still Wrong: Counterfactual Hallucination in Multilingual Vision-Language Models

**arXiv ID:** 2602.05437 | [PDF](https://arxiv.org/pdf/2602.05437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 241. Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework

**arXiv ID:** 2602.05310 | [PDF](https://arxiv.org/pdf/2602.05310v1)

**作者:** Jipeng Kong `[一作]` (Institute of Artificial Intelligence), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 61676 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个分阶段的 Perception‑Action integrated Decision‑making (PAiD) 框架，用于教会仿人机器人在复杂环境中踢球，包含运动捕捉、轻量感知集成以及物理感知的 sim‑to‑real 迁移。

**💡 创新点**

创新点在于：①把踢球技能拆分为三步，先学习无感知的基础动作，再加入轻量感知进行位置泛化，最后用物理参数对齐降低 sim‑to‑real 问题；②采用自适应采样在运动轨迹库中动态平衡难度，避免奖励冲突；③通过系统识别与基于物理的观测噪声随机化，显著提升真实世界鲁棒性。

**🔧 技术方法**

使用的技术包括：深度强化学习（PPO）+运动重定向（GMR）、自适应采样、基于 CMA‑ES 的物理系统辨识、Physics‑Guided Domain Randomization、YOLOv8+LiDAR 的多模态感知、Fast‑LIO 运动估计以及机器人姿态控制的低层 PD 控制。

**📊 数据集**

采用 13 条来自真实球员的踢球动作捕捉数据，分为标准踢和专业化踢两类，用于训练运动跟踪与动作风格迁移。

**📈 对比分析**

与 Pure RL、AMP‑based、Single‑Stage 等基线相比，PAiD 在模拟环境中静态球命中率达 91.3%、滚动球命中率 71.9%，精度（cosine similarity）分别为 0.9689 与 0.8892；在真实硬地与草地上成功率超过 90%，明显优于基线。

**⚠️ 局限性**

局限性主要包括：在极端球位（需大幅旋转或长距离接近）和高动态条件下性能下降；依赖于高质量的人类运动捕捉数据，迁移到全新运动样式需要进一步扩展；感知模块对光照与遮挡仍有一定敏感性。

---

## 242. Enabling Automatic Disordered Speech Recognition: An Impaired Speech Dataset in the Akan Language

**arXiv ID:** 2602.05406 | [PDF](https://arxiv.org/pdf/2602.05406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 243. MentorCollab: Selective Large-to-Small Inference-Time Guidance for Efficient Reasoning

**arXiv ID:** 2602.05307 | [PDF](https://arxiv.org/pdf/2602.05307v1)

**作者:** Haojin Wang `[一作]` (University of Illinois at Urbana-Champaign), Yulia Tsvetkov `[通讯]` (University of Washington)

**通讯引用:** 5137 | [OpenAlex ID](https://openalex.org/A5062910836)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理阶段通过大型模型（LRM）作为导师，向小型模型（SLM）提供稀疏且选择性指导的协作框架。

**💡 创新点**

创新点在于：①不再让SLM模仿LRM的冗长链式思考，而是通过随机采样位置检测两者分歧；②使用轻量级验证器（可基于生成器本身或训练的MLP）决定是否采用导师的短段落；③保持SLM主导生成，仅在必要时引入LRM提示，从而显著降低推理成本。

**🔧 技术方法**

核心技术包括：随机决策（Bernoulli 采样）以决定是否查询导师；短段落（长度 4–16 词）生成与比较；两种验证器实现——提示式（free）和训练型 MLP；API 层级的模型协作与贪心解码。

**📊 数据集**

使用了三大领域的数据集：MATH（数学推理）、SuperGPQA（通用知识）和 Com^2-hard-Intervention（常识推理），分别在官方测试集或随机抽样样例上评测。

**📈 对比分析**

与五种基线（Average Decoding、Nudging、CoSD、R-Stitch、Co-LLM）对比，平均提升 3–8%（最高 8%），在 15 个生成器-导师对上获得 12/15 的正面效果；导师令牌占比约 10–30%，远低于全体替换方法，同时保持或提升推理质量。

**⚠️ 局限性**

局限性包括：①依赖随机位置启动协作，缺乏精准的干预判定标准；②高置信度的生成器预测有时会被无效跳过；③仅研究单一导师-生成器双模型，未探索多导师或多生成器的扩展；④验证器对弱生成器可能过度保守，需进一步调优。

---

## 244. Robust Inference-Time Steering of Protein Diffusion Models via Embedding Optimization

**arXiv ID:** 2602.05285 | [PDF](https://arxiv.org/pdf/2602.05285v1)

**作者:** Minhuan Li `[一作]` (Flatiron Institute), Luhuan Wu `[通讯]` (Flatiron Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种推理时嵌入优化方法EmbedOpt，用于在蛋白质扩散模型中通过优化序列嵌入来引导生成与实验测量一致的结构。

**💡 创新点**

创新点在于将注意力从坐标级别转移到条件嵌入空间，动态更新嵌入以重塑先验，显著降低先验–似然不匹配导致的脆弱性，并实现对参数的轻量级调整。

**🔧 技术方法**

使用AlphaFold 3风格的蛋白质序列到结构扩散模型、梯度上升优化嵌入、RMS梯度归一化、后期能量放松以及对比的Diffusion Posterior Sampling（DPS）方法。

**📊 数据集**

采用两套基准数据集：77个蛋白质的cryo‑EM图像拟合任务和24个蛋白质的距离约束任务，均通过合成实验数据生成。

**📈 对比分析**

与DPS比较时，EmbedOpt在“难”目标下（先验对齐度低）取得更高的图像相关系数，距离约束任务表现相当；且对学习率、采样步数等超参数的鲁棒性更好，能在更少的扩散步数下保持优良结构质量。

**⚠️ 局限性**

局限性包括对嵌入空间覆盖度的依赖，若MSA深度不足或目标结构包含训练中未见的物理相互作用，EmbedOpt可能过拟合且生成结构缺乏物理合理性；且目前仍需后处理能量放松来确保几何质量。

---

## 245. NeVStereo: A NeRF-Driven NVS-Stereo Architecture for High-Fidelity 3D Tasks

**arXiv ID:** 2602.05423 | [PDF](https://arxiv.org/pdf/2602.05423v1)

**作者:** Pengcheng Chen `[一作]`, Eric J Seibel `[通讯]` (University of Washington)

**通讯引用:** 4074 | [OpenAlex ID](https://openalex.org/A5004492656)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 NeVStereo，一种基于 NeRF 的 NVS-立体融合框架，能够同时输出高精度相机姿态、多视角深度、创新视角合成和 3D 网格重建。

**💡 创新点**

创新点在于：① 将 NeRF 的高质量 NVS 与立体深度估计结合；② 通过多视角置信度引导的 RGB‑D 优化（Mv‑CG）实现姿态与深度协同迭代；③ 将 NeRF 与束调整耦合，使渲染、深度和姿态共同优化；④ 采用深度引导的 Gaussian 采样提升 NeRF 采样效率与几何精度。

**🔧 技术方法**

使用的技术包括 ZipNeRF（抗锯齿 NeRF）、FoundationStereo（深度估计）、DROID‑SLAM 的姿态优化、TSDF 融合、深度引导的 Gaussian 采样、以及多尺度 hash‑grid 训练。

**📊 数据集**

在四大类数据集上评估：ScanNet++、Replica、NVIDIA‑HOPE（桌面场景）和 WildUAV（无人机空中场景），以及内部收集的 Mobilebrick 等。

**📈 对比分析**

与现有基准（COLMAP、VGGSFM、MVSFormer、NeuS、SurfaceSplat、ZipNeRF 等）在相机姿态、深度误差、PSNR/SSIM/LPIPS、F1/Chamfer 等指标上均表现出色；在多视角深度上误差下降约 36%，姿态误差降低 10.4%，NVS PSNR 提升 4.5%，网格 F1 达 91.93%（Chamfer 4.35 mm）。

**⚠️ 局限性**

局限性包括：仍依赖 COLMAP 的初始姿态；对极稀疏视角的鲁棒性不足；NeRF 训练耗时较长；以及在极端光照或纹理缺失场景下可能出现浮点/堆叠等几何误差。

---

## 246. Imagine a City: CityGenAgent for Procedural 3D City Generation

**arXiv ID:** 2602.05362 | [PDF](https://arxiv.org/pdf/2602.05362v1)

**作者:** Zishan Liu `[一作]` (Lingnan University), Zhengzhe Liu `[通讯]` (Lingnan University)

**通讯引用:** 376 | [OpenAlex ID](https://openalex.org/A5014239420)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了CityGenAgent框架，实现了基于自然语言的分层程序化3D城市生成与交互式编辑。

**💡 创新点**

创新点包括：①引入Block Program与Building Program两级DSL，实现城市布局与建筑细节的解耦与可编辑；②采用SFT+RL两阶段训练，设计空间对齐奖励与视觉一致性奖励，显著提升空间推理与视觉语义对齐。

**🔧 技术方法**

使用了Qwen3-8B LLM的SFT与PPO RL优化、程序化生成与执行引擎、VLM评估、渲染与网格生成、资产检索与文本到3D生成等技术。

**📊 数据集**

构造了5k对SFT样本与5k偏好样本的合成数据集，用于BlockGen与BuildingGen训练；并收集100个城市块描述与50个交互提示进行评测。

**📈 对比分析**

与CityDreamer、CityCraft、CityX、Hunyuan3D等渲染、扩散与程序化方法在文本对齐、视觉一致性、几何质量（ROS/OTR）和程序格式正确率等指标比较，CityGenAgent在文本与视觉一致性上均领先，几何质量更优，生成速度也更快。

**⚠️ 局限性**

局限性在于依赖合成数据集，资产库规模有限；对极端或高度复杂的城市布局仍可能表现不足；缺乏大规模真实户外数据；多模态一致性评估主要靠VLM，可能忽略细节。

---

## 247. SciDef: Automating Definition Extraction from Academic Literature with Large Language Models

**arXiv ID:** 2602.05413 | [PDF](https://arxiv.org/pdf/2602.05413v1)

**作者:** Filip Kučera `[一作]` (National Institute of Informatics), Timo Spinde `[通讯]` (National Institute of Informatics)

**通讯引用:** 390 | [OpenAlex ID](https://openalex.org/A5041704286)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SciDef LLM驱动的定义抽取流水线，并公开了DefExtra和DefSim两个数据集用于评测；

**💡 创新点**

创新点在于（1）首次公开学术论文定义的标注基准；（2）系统性探索多步骤和DSPy优化的提示策略；（3）引入NLI为最佳相似度度量以提高评估可靠性；

**🔧 技术方法**

使用的大模型包括16种LLM（OpenRouter、gemini、gpt-5、vLLM等），提示策略有OneStep、MultiStep、FewShot以及DSPy编译；评估技术包括Embedding相似度、NLI、LLM判别；

**📊 数据集**

使用DefExtra（75篇论文268条定义）作为训练/测试集，DefSim（60个定义对）用于验证相似度；还在标准语义相似度基准（STS、SICK、MSRP、QQP）上验证指标；

**📈 对比分析**

比较方法是基于定义集合的双向匹配评估，使用阈值0.25；在多模型、多提示策略下，DSPy+NLI得到最高的平均得分0.397，覆盖率达69.7%；相较于单步提示，精度提升约20%；

**⚠️ 局限性**

局限性包括：(1) 样本量和领域覆盖有限；(2) 仍存在过度生成/不相关定义的问题；(3) 未使用最前沿大型模型，成本受限；(4) 需要进一步自动化相关定义筛选。

---

## 248. RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs

**arXiv ID:** 2602.05367 | [PDF](https://arxiv.org/pdf/2602.05367v1)

**作者:** Youngcheon You `[一作]` (Samsung Research), Dongkyu Kim `[通讯]` (Samsung Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RaBiT框架，解决极低位（2‑bit）大语言模型量化中的路径共适应问题，实现高效、准确的模型推理

**💡 创新点**

通过共享全精度权重、按残差顺序动态生成二进制路径，构建残差层级结构，消除路径间冗余；并引入功能感知的迭代SVID初始化以提升训练稳定性

**🔧 技术方法**

量化感知训练（QAT）、残差二进制网络、STE梯度传递、SVD与迭代残差分解、知识蒸馏、MuOn优化器

**📊 数据集**

WikiText‑2、C4（用于校准）、Llama2/3、Gemma3模型系列，以及针对评测的五大推理基准（HellaSwag、PIQA、WinoGrande、ARC‑e/c）和更难任务（BBH、GPQA、MMLU‑Pro、IFEval）

**📈 对比分析**

与多种2‑3‑bit方法（GPTQ、EfficientQAT、AQLM、QuIP#、QTIP、BitStack、DB‑LLM、MBOK、DBF等）对比，RaBiT在2‑bit下实现了SOTA的PPL与QA准确率，甚至超越多数VQ方法；同时在RTX 4090上实现与FP16相当甚至更高的推理吞吐率，显著降低延迟

**⚠️ 局限性**

仍受限于训练时的梯度估计误差与模型规模的可扩展性；对极端压缩后安全性与对齐效果未做充分验证；在部分新架构（如Llama3‑8B）中仍需改进以避免不稳定性

---

## 249. LD-SLRO: Latent Diffusion Structured Light for 3-D Reconstruction of Highly Reflective Objects

**arXiv ID:** 2602.05434 | [PDF](https://arxiv.org/pdf/2602.05434v1)

**作者:** Sanghoon Jeon `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Yonsei University)

**通讯引用:** 29584 | [OpenAlex ID](https://openalex.org/A5080001926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于潜在扩散的结构光恢复框架 LD‑SLRO，用以修复高度反射、低粗糙度表面的条纹图像，并提升三维重建精度。

**💡 创新点**

创新点在于：①双编码器（漫反射编码器与镜面反射编码器）分别提取表面光学特征；②将这些特征作为条件注入潜在扩散模型；③设计时间可变通道仿射与多尺度注意力模块，能够自适应不同噪声级别下的反射扰动；④支持输入与输出条纹数量与周期不同的配置，提升采集灵活性。

**🔧 技术方法**

使用技术包括：潜在扩散概率模型（DDPM）与条件扩散；VAE 结构光编码器/解码器；时间步正弦嵌入与通道仿射；通道注意力与瓶颈自注意力；相位解包与三角测量。

**📊 数据集**

数据集为实验系统采集的 200 组真实高度反射物体，共 8400 幅条纹图像；使用 6 步 36 像素二进制条纹与 24 步 24 像素条纹（并在表面喷墨涂层做真值）进行训练与评估，包含不同曝光级别的 HDR 合成。

**📈 对比分析**

方法与 24 步传统相位切换、HDRNet、DC‑UNet、Y‑FFC 进行对比。LD‑SLRO 在纹理恢复指标上取得 MSE 0.0002、SSIM 0.9708、PSNR 36.29；在三维重建上 RMSE 0.8059~1.394 mm，显著优于其他方法（最小为 1.4929 mm，最大 7.4923 mm）。

**⚠️ 局限性**

局限性：对极高频细节仍有一定损失；需要大量标注数据和相机/投影器的精准校准；对极端多反射或快速运动场景的鲁棒性尚未验证；扩散模型训练耗时，推理时仍需一定算力。

---

## 250. M$^2$-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining

**arXiv ID:** 2602.05429 | [PDF](https://arxiv.org/pdf/2602.05429v1)

**作者:** Rui Lv `[一作]` (Ant Group), Lei Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 6714 | [OpenAlex ID](https://openalex.org/A5002705877)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 M^2-Miner 框架，能够自动化挖掘移动设备 GUI 代理所需的高质量意图–轨迹对。

**💡 创新点**

创新点在于将 MCTS 与三大协同智能体（InferAgent、OrchestraAgent、JudgeAgent）结合，并引入意图回收与进化式模型迭代训练，显著提升挖掘效率和数据多样性。

**🔧 技术方法**

技术方案包括基于 MCTS 的搜索树结构、利用 Qwen2.5‑VL‑7B/72B 大模型实现动作生成、评估与意图生成，t‑SNE 可视化以及模型循环训练。

**📊 数据集**

使用自建 M^2‑Miner‑Agent 数据集（20k 图像、2,565 轨迹）以及公开基准 AC、AITZ、GUI Odyssey、CAGUI 进行评测。

**📈 对比分析**

与多种基线对比，M^2‑Miner 在四大基准上均实现了 SOTA，尤其在动作预测准确率与任务完成率（SR）上明显优于手工标注数据或自动挖掘基线，MSR 与 DQA 也得到显著提升。

**⚠️ 局限性**

局限性包括：对新场景仍需通过模型迭代训练逐步适配；框架高度依赖大模型算力，成本与能耗较高；意图回收策略对大模型生成意图质量敏感，易受错误意图影响。

---

## 251. Disco: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring

**arXiv ID:** 2602.05420 | [PDF](https://arxiv.org/pdf/2602.05420v1)

**作者:** Rui Sun `[一作]` (Shanghai Academy of Artificial Intelligence for Science), Yuan Cheng `[通讯]` (Fudan University)

**通讯引用:** 8298 | [OpenAlex ID](https://openalex.org/A5058272109)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于邻接图着色的细胞实例分割框架 Disco，利用显式标记与隐式消歧解耦损失实现对非二分细胞图的高效分割。

**💡 创新点**

创新点在于：①系统揭示真实细胞图普遍非二分，含大量奇环；②设计“分而治之”动态着色策略，将大部分节点划为二分子图，剩余冲突集单独标记；③引入邻接约束损失（Adjacency Constraint Loss）在连续特征空间中实现隐式消歧；④用冲突图作为解释性工具。

**🔧 技术方法**

技术包括：图着色理论、双分支深度分割网络、BFS 取最大二分子图、显式标记策略、交叉熵与一致性损失、冲突消歧损失、对比式邻接约束损失。

**📊 数据集**

使用四个数据集：PanNuke、DSB2018、CryoNuSeg、并新建的高密度 GBC‑FS 2025。

**📈 对比分析**

与检测、轮廓、距离映射、FCIS、SAM 等主流方法对比，Disco 在 PQ、AJI、SQ 等指标上平均提升约 2.7%，在最难的 GBC‑FS 2025 上提升 7.08% PQ，且在所有四个数据集上均位居榜首。

**⚠️ 局限性**

局限性包括：对冲突集的显式标记仍需手工构造的图算法；仅处理 2D 平面图像；对极端三角冲突集较大或三维组织时可能仍受限；整体模型参数量和推理时间相对传统方法略高。

---

## 252. Reduced-Order Surrogates for Forced Flexible Mesh Coastal-Ocean Models

**arXiv ID:** 2602.05416 | [PDF](https://arxiv.org/pdf/2602.05416v1)

**作者:** Freja Høgholm Petersen `[一作]` (DHI), Allan P. Engsig-Karup `[通讯]` (DTU)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对受气象与边界强迫驱动的海岸-海洋系统，构建并评估了一种基于 Koopman 自编码器的低阶 surrogate，并与 POD surrogate 进行了对比。

**💡 创新点**

创新点在于：①将外部强迫显式整合进 Koopman 自编码器；②采用特征分离 autoencoder 与线性/非线性时间传播；③通过特征值正则化与时间展开提升长时序预测的稳定性。

**🔧 技术方法**

使用了 Koopman 自编码器（线性/非线性编码器）、POD+线性回归或 MLP 回归、随机化 SVD、特征值正则化、时间展开、神经网络优化等技术。

**📊 数据集**

使用三套真实海岸域的 MIKE21 训练数据：Øresund、Southern North Sea 与 Adriatic Sea，包含 30 分钟间隔的表面高度、二维速度以及风速、气压等外部强迫。

**📈 对比分析**

通过对一年（30 分钟步长）自回归预测与观测对比，Koopman 自编码器在相对 RMSE 0.01–0.13、R² 0.94–0.996 之间优于 POD，推断速度提升 300–1400 倍。

**⚠️ 局限性**

局限在于：Adriatic Sea 训练数据不足导致误差增大；极端事件预测与未知强迫泛化能力待进一步验证；非凸训练导致收敛不稳定。

---

## 253. Explainable Pathomics Feature Visualization via Correlation-aware Conditional Feature Editing

**arXiv ID:** 2602.05397 | [PDF](https://arxiv.org/pdf/2602.05397v1)

**作者:** Yuechen Yang `[一作]` (Vanderbilt University), Yuankai Huo `[通讯]` (Vanderbilt University)

**通讯引用:** 5403 | [OpenAlex ID](https://openalex.org/A5067191302)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种Manifold-Aware Diffusion (MAD) 框架，用于在保持细胞核形态生物学可行性的前提下，可控编辑关联的数字病理学特征；

**💡 创新点**

创新点在于通过在变分自编码器学习到的特征流形上进行隐空间引导优化，确保编辑的特征向量仍落在真实细胞核分布的低维流形内，从而避免传统独立特征编辑导致的无效或不自然结果；

**🔧 技术方法**

使用了条件扩散模型、β-变分自编码器、MLP条件编码器以及隐空间梯度优化等技术；

**📊 数据集**

数据集基于1556份人类和啮齿类肾脏组织的全切片图像，提取75维细胞核特征，最终用于训练和评估的样本约28,809个；

**📈 对比分析**

与StyleGAN2、Stable Diffusion (LoRA)以及无VAE的MAD进行对比，MAD在单核分割成功率、特征控制MAE/R²以及感知相似度（LPIPS）方面显著优于编辑基准，并与无条件生成相比保持更高图像质量；

**⚠️ 局限性**

主要局限包括推理速度慢（约17秒/次优化），限制实时交互，以及仅针对肾脏细胞核，需扩展到其他细胞类型和进一步加速优化流程。

---

## 254. Linear Systems and Eigenvalue Problems: Open Questions from a Simons Workshop

**arXiv ID:** 2602.05394 | [PDF](https://arxiv.org/pdf/2602.05394v1)

**作者:** Noah Amsel `[一作]`, Jess Williams `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

整理并分类了2025年Simons研究所线性系统与特征值问题研讨会产生的55个开放研究问题，覆盖迭代求解、低秩近似、随机化投影等五大类别。

**💡 创新点**

创新点在于将理论计算机科学与数值分析界的讨论融合，形成统一的开放问题目录，为后续研究提供清晰的路线图。

**🔧 技术方法**

采用工作组讨论、文献综述和主题梳理的方式，构建问题框架，并用技术层面如多重网格、随机投影、低秩分解等方法进行归类。

**📊 数据集**

本报告不涉及实验数据，主要基于已有文献与专家讨论而非特定数据集。

**📈 对比分析**

由于是开放问题清单，未进行方法比较或性能评估；文中仅引用现有研究结果与理论预期。

**⚠️ 局限性**

局限性在于缺乏对每个问题的深入解决方案，且多数问题仍待进一步研究与实验验证。

---

## 255. Beyond Length: Context-Aware Expansion and Independence as Developmentally Sensitive Evaluation in Child Utterances

**arXiv ID:** 2602.05392 | [PDF](https://arxiv.org/pdf/2602.05392v1)

**作者:** Jiyun Chun `[一作]` (Ohio State University), Andrew Perrault `[通讯]` (Ohio State University)

**通讯引用:** 364 | [OpenAlex ID](https://openalex.org/A5057049889)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于LLM的“扩展(Expansion)”和“独立性(Independence)”两维度评估儿童在成人对话中的回应，结合成人前一句的语用类型进行上下文感知评分。

**💡 创新点**

创新点在于：①用成人前一句的类型作为条件，挖掘语境对儿童回应的影响；②利用LLM-as-a-judge自动化标注，避免手工规则的脆弱性；③将长度/可读性指标替换为关注推理深度、话题维护与话语计划的量表，从而提升评估的语义和发展性敏感度。

**🔧 技术方法**

技术方法包括：大型语言模型（Gemini 2.5 Pro、Mixtral‑8x7B‑Instruct）进行自动标注；统计建模使用CLMM、LMM、线性/梯度提升回归；年龄预测实验使用5‑折交叉验证；人类评估通过三名专家标注验证模型一致性。

**📊 数据集**

数据集为CHILDES北美英语子语料库20个，约360k条儿童对话，儿童年龄范围2–10岁，包含多种成人提问/非提问类型。

**📈 对比分析**

方法对比：与传统MLU、vocd‑D、Flesch‑Kincaid、Gunning Fog等基线相比，E+I+PT特征在年龄预测任务中MAE下降约20–30%，R²提升至0.55以上，成为最佳表现；在描述统计和语义敏感性实验中，Expansion/Independence的效果显著优于长度指标。

**⚠️ 局限性**

局限性：仅使用书面转录，未考虑语音/音韵信息；基于北美英语，跨语言适用性未知；LLM在年龄预测中可能受预训练数据泄漏影响；指标不适合临床诊断或高风险决策。

---

## 256. Dataset Distillation via Relative Distribution Matching and Cognitive Heritage

**arXiv ID:** 2602.05391 | [PDF](https://arxiv.org/pdf/2602.05391v1)

**作者:** Qianxin Xia `[一作]` (University of YYY), Guoming Lu `[通讯]` (University of YYY)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预训练自监督视觉模型的基础上，提出一种高效的统计流匹配（Statistical Flow Matching，SFM）框架，用极少的合成图像实现数据集蒸馏，并通过继承原始数据训练的分类器（Golden Classifier）提升推理性能。

**💡 创新点**

① 将线性梯度匹配视为“流”并引入全局统计中心；② 通过一次性加载原始统计并使用单次可微增强的 SFM，显著降低内存与计算开销；③ 采用分类器继承（CI）重用原始数据训练的分类器，替代软标签与全模型微调。

**🔧 技术方法**

统计流匹配（SFM）+余弦距离、可微分增强、轻量线性投影器、预训练自监督模型（CLIP、DINO‑v2、EVA‑02、MoCo‑v3）

**📊 数据集**

ImageNet‑1k、ImageNet‑100、Stanford Dogs、CUB‑200‑2011、ImageWoof、ArtBench 等公开数据集

**📈 对比分析**

与 Linear Gradient Matching（LGM）及三种实图基线（Random、Centroids、Neighbors）对比，SFM+CI 在 ImageNet‑100 上 95.1% 的蒸馏准确率、ImageNet‑1k 上 92.8% 的蒸馏准确率，逼近全数据训练；运行时间缩短 4×、显存降低 10×；在跨模型泛化实验中也优于 LGM。

**⚠️ 局限性**

仅针对分类任务验证；对不同自监督模型的泛化仍有限；未覆盖检测、分割等下游任务；依赖预训练模型与统计中心的计算成本。

---

## 257. Assessing Electricity Demand Forecasting with Exogenous Data in Time Series Foundation Models

**arXiv ID:** 2602.05390 | [PDF](https://arxiv.org/pdf/2602.05390v1)

**作者:** Wei Soon Cheong `[一作]` (Institute for Infocomm Research), Jamie Ng Suat Ling `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 3868 | [OpenAlex ID](https://openalex.org/A5017222712)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究对比评估了多种时序基础模型（MOIRAI、MOMENT、TinyTimeMixers、ChronosX、Chronos-2）与 RevIN-LSTM 基线，在新加坡与澳大利亚电力需求预测中引入外部特征的效果；

**💡 创新点**

创新点在于系统性地检验跨通道特征整合对基础模型的影响，并揭示模型架构与地理环境对性能的决定性作用，指出基础模型并非在所有场景下都优越；

**🔧 技术方法**

采用滑动窗口上下文（512 步）预训练/微调，零样本与微调实验，利用 MAPE 评价，并进行特征选择与 Granger 因果检验；

**📊 数据集**

使用新加坡（2016-2022）与澳大利亚 ACT 区（2015-2023）的小时级与日级电量及气象/日期特征数据，共 30 维变量；

**📈 对比分析**

通过将各模型在“所有特征”“选择特征”“仅目标”三种配置下的 MAPE 进行对比，发现 Chronos-2 在大多数情形下表现最佳，而 RevIN‑LSTM 在新加坡短期预测中更具优势；

**⚠️ 局限性**

局限包括仅覆盖两国两种气候环境、未深入探讨计算成本与能耗、基础模型在稳定环境下的过拟合风险，以及缺乏专门针对电力域的预训练策略。

---

## 258. Multi-AD: Cross-Domain Unsupervised Anomaly Detection for Medical and Industrial Applications

**arXiv ID:** 2602.05426 | [PDF](https://arxiv.org/pdf/2602.05426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 259. Day-Ahead Electricity Price Forecasting for Volatile Markets Using Foundation Models with Regularization Strategy

**arXiv ID:** 2602.05430 | [PDF](https://arxiv.org/pdf/2602.05430v1)

**作者:** Kritchanat Ponyuenyong `[一作]` (Nanyang Technological University), Lianlian Jiang `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 511 | [OpenAlex ID](https://openalex.org/A5103723713)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在新加坡高波动性电价数据上，使用时间序列基础模型结合STL‑KF极值正则化，完成了多步（24小时）日间电价预测实验。

**💡 创新点**

创新点在于将TSFMs迁移到电价预测领域，并提出了基于STL与加权Kalman滤波的极值检测与平滑策略，同时采用对数变换和滞后特征提升模型鲁棒性。

**🔧 技术方法**

技术手段包括STL分解+Kalman滤波极值正则化、对数变换、滞后特征工程，评估了8种TSFMs（TTMs、MOIRAI、MOMENT、TimesFM、Time‑MoE、Timer‑XL、Chronos、Lag‑Llama）以及ARIMA、LSTM、CNN‑LSTM、PatchTST等基线模型。

**📊 数据集**

使用了2021‑2024年新加坡能源市场公司（EMC）与能源市场管理局（EMA）提供的半小时电价与需求数据，外加OpenWeather API气象变量（温度、湿度、热指数）和公共假日信息。

**📈 对比分析**

采用MAE、MAPE、RMSE三指标，在训练从零、零射击、单变量微调、多变量微调等六种策略下公平比较；结果显示TSFMs平均MAPE下降约37.4%，TTMs多变量微调模型取得最佳性能。

**⚠️ 局限性**

局限性包括仅在新加坡市场验证，未考察其他地区；极值正则化对不同市场的通用性未知；某些模型在微调后性能下降，可能存在过拟合；实验未涉及概率预测与不确定性评估。

---

## 260. Multi-Field Tool Retrieval

**arXiv ID:** 2602.05366 | [PDF](https://arxiv.org/pdf/2602.05366v1)

**作者:** Yichen Tang `[一作]` (Tsinghua University), Qingyao Ai `[通讯]` (Tsinghua University)

**通讯引用:** 4438 | [OpenAlex ID](https://openalex.org/A5089655391)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多字段工具检索框架（MFTR），通过标准化工具文档、查询重写与自适应字段加权，提升大语言模型在工具检索任务中的表现。

**💡 创新点**

创新点包括：①设计统一的四字段工具文档模式（目的、输入、输出、使用场景）；②使用工具感知的查询重写，消除用户查询与技术文档的语义与粒度不匹配；③引入参数缺失惩罚与自适应字段权重，动态衡量工具多维效用；④在混合大规模数据集上展示卓越的跨域鲁棒性。

**🔧 技术方法**

技术手段主要有：LLM（gpt‑4o‑mini）做文档标准化与查询重写；稀疏检索（BM25）与多种稠密检索模型（MiniLM‑L6、Contriever、E5、GTE、BGE、API Retriever、COLT）做字段级匹配；对齐时使用余弦相似度、sigmoid惩罚；最终通过学习权重与pairwise ranking loss进行端到端训练。

**📊 数据集**

使用五个公开工具检索数据集（ToolBench、APIGen、APIBank、Gorilla、Toolink）以及自构建的混合（Mixed）数据集进行评估。

**📈 对比分析**

与Full‑Doc、EasyTool、PLUTo、OnlineRAG等基线对比；在NDCG@10和Recall@10上，MFTR在单一数据集上平均提升约10–20%，在Mixed benchmark上提升28.6%/18.9%；在不同检索器下保持显著优势，证明模型无关性和高效性。

**⚠️ 局限性**

局限性：仍依赖LLM进行标准化与重写，处理极大规模工具库时会产生额外推理开销；对极其不完整或缺失字段的文档效果有限；缺乏对工具执行反馈的在线学习机制，难以动态调整检索效果。

---

## 261. Multimodal Latent Reasoning via Hierarchical Visual Cues Injection

**arXiv ID:** 2602.05359 | [PDF](https://arxiv.org/pdf/2602.05359v1)

**作者:** Yiming Zhang `[一作]` (Nanyang Technological University), Kai Han `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在多模态潜在空间中进行递归推理的框架 HIVE，通过层次化视觉信息注入和循环 Transformer 进行“慢思考”推理。

**💡 创新点**

在潜在空间中实现多模态递归推理，首次将多尺度视觉特征层级注入循环 Transformer，并结合自适应计算与 KV 缓存优化提升效率。

**🔧 技术方法**

采用 Huginn 风格的循环 Transformer、InternViT 视觉编码器、多层级视觉注入、Poisson 随机递归深度、KV 缓存重用以及自适应推理步数等技术。

**📊 数据集**

训练阶段使用 LCS‑558K、EMOVA、ShareGPT‑4V、ALLaVA、SynthDog、MMC‑Alignment、UReader 等多模态对齐数据；评估基准包括 MMBench、MMStar、ScienceQA‑Img、SEED‑Bench、RealWorldQA、TextVQA、DocVQA、ChartQA、MathVista、POPE、GQA 等。

**📈 对比分析**

与无递归基线、仅递归无层次以及公开 7B/8B 模型对比，HIVE 在 ScienceQA‑Img 91.57%、MMStar 49.79%、POPE 87.61% 等任务上均优于同规模模型，且在可观测推理步数上显著降低。

**⚠️ 局限性**

模型规模仍偏小(4B)，对 OCR/表格细粒度视觉理解不足；层级特征选择与动态分辨率方案待优化；未实现显式 CoT 与更通用的早停机制。

---

## 262. Rich-Media Re-Ranker: A User Satisfaction-Driven LLM Re-ranking Framework for Rich-Media Search

**arXiv ID:** 2602.05408 | [PDF](https://arxiv.org/pdf/2602.05408v1)

**作者:** Zihao Guo `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**通讯引用:** 17862 | [OpenAlex ID](https://openalex.org/A5100380470)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Rich-Media Re-Ranker 框架，利用会话感知的查询规划器拆分用户多维意图，VLM 评估封面图的视觉相关性与质量，并通过 LLM re‑ranker 整合多维度信号实现可解释的重新排序；同时通过多任务强化学习提升 VLM 与 LLM 的适应性。

**💡 创新点**

①会话感知的查询拆解实现多维意图覆盖；②封面图的视觉相关性与质量双维度评估；③在 LLM 与 VLM 上采用 GRPO 多任务强化学习；④在基线上引入综合重排序原则与视觉信号。

**🔧 技术方法**

会话感知查询规划器、VLM 评估器、LLM re‑ranker（Qwen3‑4B）、多任务强化学习（GRPO）、离线推理+T+1 激活、覆盖多维重排序原则。

**📊 数据集**

工业搜索日志产生的 14,115 条查询（复杂/广义/简单），每条约 20 候选；封面图视觉评估用人工标注 500 张图/级别；Re‑ranking 训练集由 DeepSeek‑R1 合成；评估用 N@K、R@K、RBO 等指标。

**📈 对比分析**

与 ReaRank、Rank‑R1、ReasonRank 等基线（使用 Qwen2.5‑7B）对比；在所有查询类型上均优，NDCG@10 提升 17.1%，Recall@10 提升 27.0%；在线 A/B 实验提升点击率、满意度等指标；离线实验中各子模块消融实验验证其贡献。

**⚠️ 局限性**

仍需大量标注和算力；对简单查询提升有限；主要关注文本与封面图，未覆盖其他多模态信号；离线推理+缓存限制实时性；模型规模受限，未充分探索更大规模 LLM 与 VLM 的效果。

---

## 263. BadTemplate: A Training-Free Backdoor Attack via Chat Template Against Large Language Models

**arXiv ID:** 2602.05401 | [PDF](https://arxiv.org/pdf/2602.05401v1)

**作者:** Zihan Wang `[一作]` (University of Electronic Science and Technology of China), Guowen Xu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 4858 | [OpenAlex ID](https://openalex.org/A5046950426)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型的聊天模板进行恶意修改，在系统提示中注入隐藏指令，实现无训练参数修改的后门攻击

**💡 创新点**

创新点在于利用聊天模板的可定制性，将后门指令嵌入系统提示，形成“训练‑free”后门，攻击成本低、部署广泛、且对模型参数不做任何更改

**🔧 技术方法**

使用 Jinja 模板编辑聊天模板、基于系统提示注入词级与句级触发指令、采用贪婪解码和 ICL 演示进行推理评估

**📊 数据集**

在 SST-2、SMS、AGNews、DBPedia、Amazon 五个文本分类基准数据集上进行实验

**📈 对比分析**

与基线 ICL-1Shot/2Shot/3Shot 进行对比；在 6 个开源和 3 个闭源 LLM 上，攻击成功率可达 100%，同时保持接近原始 ACC，明显优于传统基于提示的后门

**⚠️ 局限性**

局限性包括：需要攻击者对聊天模板具备修改权限，攻击效果受模型指令遵循能力和句子长度影响；现有 HuggingFace 与 LLM‑as‑a‑judge 检测机制对其检测率低，需开发更精准的防御手段

---

## 264. A Decomposition-based State Space Model for Multivariate Time-Series Forecasting

**arXiv ID:** 2602.05389 | [PDF](https://arxiv.org/pdf/2602.05389v1)

**作者:** Shunya Nagashima `[一作]` (Neurogica Inc), Shinnosuke Hirano `[通讯]` (Neurogica Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种端到端的多变量时间序列分解预测框架DecompSSM，利用三条并行的状态空间模型分离趋势、季节和残差，进一步通过全局上下文修正和辅助重构/正交损失提升预测精度。

**💡 创新点**

创新点在于：1）引入输入自适应时间尺度预测器（ASP）实现分量级别的时间尺度自适应；2）设计跨变量的全局上下文修正模块（GCRM）以同步变量关系；3）使用辅助重构与正交损失确保分解的可解释性与互补性。

**🔧 技术方法**

核心技术包括：多变量状态空间模型（S5）、输入自适应时间尺度预测器、全局上下文修正模块、重构与正交辅助损失以及双向S5作为特征提取骨干。

**📊 数据集**

在ECL、Weather、ETTm2、PEMS04四个主流多变量时间序列数据集上进行实验验证。

**📈 对比分析**

与Autoformer、PatchTST、iTransformer、PPDformer、DLinear、HDMixer、TimesNet等七种基线模型对比，DecompSSM在32个预测场景中获得28个最优分数，在MSE和MAE上均优于第二名PPDformer，提升幅度在1%–2%之间。

**⚠️ 局限性**

局限性包括：1）分支数量固定为三条，未能自动根据频谱自适应；2）实验仅使用10个训练周期，可能未充分挖掘模型潜力；3）缺乏对长序列（>720步）和多模态输入的进一步评估。

---

## 265. IESR:Efficient MCTS-Based Modular Reasoning for Text-to-SQL with Large Language Models

**arXiv ID:** 2602.05385 | [PDF](https://arxiv.org/pdf/2602.05385v1)

**作者:** Tao Liu `[一作]` (Zhengzhou University), Hongyin Zan `[通讯]` (Zhengzhou University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 IESR 框架，将文本到 SQL 的推理拆分为信息理解、数值计算与 SQL 生成三步，并通过 MCTS 多路径搜索和轨迹一致性验证来提升执行准确率。

**💡 创新点**

创新点包括：① 明确解耦数学计算与 SQL 生成，② 设计多类型动作的 MCTS 以异质化探索推理步骤，③ 使用判别器进行轨迹级别的一致性校验，④ 通过多数投票进一步提升鲁棒性。

**🔧 技术方法**

使用的技术包括轻量级大型语言模型、语义信息提取与架构链接、结构化的 MCTS（支持多种动作）、执行一致性奖励、判别器一致性验证、以及多数投票策略。

**📊 数据集**

评估数据集涵盖：LogicCat、Archer、Spider 与 BIRD，后两者用于验证在通用任务上的性能。

**📈 对比分析**

与 DIN‑SQL、DAIL‑SQL、DTS‑SQL、Alpha‑SQL、SQL‑O1 等主流方法对比，IESR 在 LogicCat 上达到 24.28% EX，Archer 上 37.28% EX，且使用 7B‑8B 轻量模型且无指令微调，显著优于现有 SOTA。

**⚠️ 局限性**

局限性包括：对初始语义提取的质量高度依赖，MCTS 搜索导致推理成本上升，缺乏对中间推理错误的细粒度诊断，且在实际可变数据库或噪声环境下的泛化能力尚未充分验证。

---

## 266. VRIQ: Benchmarking and Analyzing Visual-Reasoning IQ of VLMs

**arXiv ID:** 2602.05382 | [PDF](https://arxiv.org/pdf/2602.05382v1)

**作者:** Tina Khezresmaeilzadeh `[一作]` (University of Southern California), Konstantinos Psounis `[通讯]` (University of Southern California)

**通讯引用:** 10027 | [OpenAlex ID](https://openalex.org/A5042745248)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VRIQ视觉推理IQ基准，构建了包含抽象与自然图像的双域问答集，并通过分层诊断探针评估视觉语言模型的感知与推理能力。

**💡 创新点**

创新点在于：①双域（抽象/自然）对齐的IQ题目设计；②将感知与推理拆分为独立探针，细粒度定位失败原因；③量化感知瓶颈对整体性能的影响，揭示模型主要欠缺在感知层面而非推理。

**🔧 技术方法**

采用了多任务感知探针（计数、形状、位置、颜色、旋转、深度等）与推理探针（文本化规则推断），并在OpenAI o3等模型中实验工具增强推理；对模型进行多轮评测与人类基线对比。

**📊 数据集**

使用了自制的VRIQ数据集（共1500道专家编写的IQ题目），涵盖抽象与自然两种视觉域，并为每道题目标注了8个感知属性与5个推理属性。

**📈 对比分析**

通过层级评测（端到端、感知探针、推理探针）与错误归因，比较了多款开源与专有视觉语言模型；结果显示抽象题目准确率仅约28%，自然题目约45%，工具增强模型在某些子任务可达80%+但仍远低于人类（≈90–98%）。

**⚠️ 局限性**

局限性包括：仅覆盖静态IQ题型，未涉及动态或因果推理；探针覆盖的感知维度有限，未能捕捉纹理、语义等高级属性；数据集虽经过人工校对但仍可能存在泄露风险，且工具使用受限于调用次数。

---

## 267. Stable Velocity: A Variance Perspective on Flow Matching

**arXiv ID:** 2602.05435 | [PDF](https://arxiv.org/pdf/2602.05435v1)

**作者:** Donglin Yang `[一作]`, Renjie Liao `[通讯]` (University of British Columbia)

**通讯引用:** 5965 | [OpenAlex ID](https://openalex.org/A5048686150)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 Stable Velocity 框架，统一改进流匹配训练和采样，包含 StableVM、VA-REPA 与 StableVS 三个模块。

**💡 创新点**

创新点在于从方差视角揭示流匹配的两段结构，利用无偏方差减小的 StableVM、低方差下的自适应辅助监督 VA-REPA，以及无需再训练的低方差采样加速 StableVS。

**🔧 技术方法**

采用流匹配、随机插值、SDE/ODE 逆演化、重要性自归一化采样、表示对齐与 SNR 加权等技术实现方差分析与训练/采样加速。

**📊 数据集**

主要实验数据集包括 ImageNet 256×256（通过 Stable Diffusion VAE 编码的潜在空间）以及预训练的文本-图像/文本-视频模型 SD3.5、Flux、Qwen-Image、Wan2.2。

**📈 对比分析**

在不需要额外 fine‑tune 的情况下，StableVM+VA-REPA 在多种模型尺寸下的 FID、IS、精度/召回均优于 REPA、REG、iREPA 等基线；StableVS 在低方差区间实现 2× 以上的采样步数缩减，保持或提升图像/视频质量。

**⚠️ 局限性**

局限性包括需手动设定低/高方差分界点 ξ，方差估计对不同分布的稳健性未完全验证，StableVM 仍需要额外的训练步骤和参考样本池，且在极端高维或非 Gaussian 数据上效果可能下降。

---

## 268. Benchmarking Affordance Generalization with BusyBox

**arXiv ID:** 2602.05441 | [PDF](https://arxiv.org/pdf/2602.05441v1)

**作者:** Dean Fortier `[一作]` (Microsoft Research), Galen Mullins `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可3D打印、模块化的硬件基准板“VLABench”，用于系统评估Vision‑Language‑Action模型在不同视觉变异下的支配性（affordance）泛化能力，并收集了配套的语言标注演示数据。

**💡 创新点**

创新点在于将硬件与实验流程统一化：设计了可通过交换/旋转模块快速生成视觉差异但保持相同支配性的配置，并提供了公开的CAD文件与电子电路，形成了可在实验室快速复现且能评测VLA视觉泛化的物理基准。

**🔧 技术方法**

技术手段包括：使用Trossen Mobile Aloha双臂机器人进行遥操作数据采集；借助Raspberry Pi 0与USB传感器实现模块状态自动记录；对OpenAI的LLaVA和NVIDIA的Isaac‑GR00T等开源VLA模型进行微调与评估。

**📊 数据集**

使用的数据集为约1993条演示，覆盖6类支配性（按钮、滑块、旋钮、开关、线缆插拔、摄像头视角等），每类均包含多种语言指令变体，且针对canonical、semi‑shuffled、fully‑shuffled三种硬件配置均有采样。

**📈 对比分析**

比较方法：在三种硬件配置上各自随机初始状态下执行60条任务指令（不含插线任务），每条任务限制30秒完成，并以任务完成且未干扰其它控件为成功标准。结果显示：在视觉一致（canonical）配置下，LLaVA约82%成功、GR00T约78%；在视觉异构（semi‑shuffled、fully‑shuffled）配置下，成功率骤降至<50%，表明现有VLA在支配性泛化上仍表现不足。

**⚠️ 局限性**

限制：模型对视觉差异过度拟合，未充分利用摄像头信息；实验未覆盖所有简单动作（如臂移动）导致结果偏颇；数据集在插线任务上的覆盖不足，导致该类任务的成功率低。

---

## 269. LTRAS: A Linkable Threshold Ring Adaptor Signature Scheme for Efficient and Private Cross-Chain Transactions

**arXiv ID:** 2602.05431 | [PDF](https://arxiv.org/pdf/2602.05431v1)

**作者:** Yi Liang `[一作]` (Southeast University), Jinguang Han `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种Linkable Threshold Ring Adaptor Signature（LTRAS）方案，能够在多账户联合支付场景下实现高效、隐私、原子性的跨链交易。

**💡 创新点**

创新点：①将适配器签名、阈值环签名与可链接性三者融合，实现多账户在同一环中隐匿签名并防止双花；②设计了滑动窗口转换及链标记机制，兼顾效率与安全；③提供完整的安全模型与证明，证明预签名可转换、可提取、可链接、可匿名等性质。

**🔧 技术方法**

技术：基于离散对数假设的 Schnorr/ECDSA 原型；阈值环签名与滑动窗口转换；可链接标记（tag）；随机预言模型安全证明；实现了预签名、适配、验证、提取等算法。

**📊 数据集**

数据集与实验：在 1.90 GHz Intel i5‑1340P、16 GB 内存、Windows 11 机器上用 Java 进行实验；对环大小从 10 到 100（阈值为 n/2）进行评估；与基线 Linkable DualRing Adaptor Signature（基于 Schnorr）比较。

**📈 对比分析**

比较方法：测量 9 个核心算法（Setup, KeyGen, GenR, PreSign, Verify, Adapt, Sign, Ext, Link）的计算时间与通信开销；结果显示 LTRAS 在 PreSign、Verify、Adapt 等关键步骤的运算时间显著低于基线，并且通信成本从 t(n+1)|ℤ_p|+t|𝔾_p| 下降到 (n+1)|ℤ_p|+t|𝔾_p|，总体性能提升明显；Setup、GenR、Ext 略高但使用频率低。

**⚠️ 局限性**

局限性：①仍需满足连续账户约束，限制匿名性；②仅基于后量子安全假设（离散对数），未实现后量子版本；③多账户跨链实际部署的复杂性与兼容性尚待进一步验证；④目前未能完全消除环中公共密钥的相邻性对匿名性的影响。

---

## 270. VMF-GOS: Geometry-guided virtual Outlier Synthesis for Long-Tailed OOD Detection

**arXiv ID:** 2602.05415 | [PDF](https://arxiv.org/pdf/2602.05415v1)

**作者:** Ningkang Peng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 618 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种全无外部数据的长尾分布下 OOD 检测框架 VMF-GOS，利用几何引导的虚拟外来样本合成与双粒度语义损失实现更紧凑的决策边界。

**💡 创新点**

创新点在于：①使用 von Mises–Fisher（vMF）分布在高维球面上精确定位低似然环带，并在此区域进行方向性采样合成虚拟外来样本；②设计双粒度语义损失（DGS），将合成样本作为显式负样本融入对比学习，提升 ID 区域紧凑度；③结合温度尺度判别调节（TLA）和能量极化正则化（EPR）形成联合优化。

**🔧 技术方法**

技术手段包括：vMF 方向分布建模、基于 χ² 分布的低似然环带采样、对比学习与双粒度语义损失、温度尺度判别调节、能量极化正则化、ODIN 后处理。

**📊 数据集**

主要使用 CIFAR-10-LT 与 CIFAR-100-LT 两个长尾基准数据集，以及多种 OOD 测试集（Textures、SVHN、Tiny ImageNet、LSUN、Places365）。

**📈 对比分析**

与现有方法（包括 MSP、EnergyOE、OE、PASCL、EAT、PATT、DARL 等）进行对比，结果显示 VMF-GOS 在 AUROC、AUPR、FPR95、ACC95 等指标上均优于或接近依赖外部样本的最先进方法，同时保持更高的 ID 分类准确率。

**⚠️ 局限性**

限制：方法仍需在高维特征空间中假设 vMF 分布，采样区间对性能有一定影响；在极度不平衡或极端高维情形下，几何合成的稳定性与泛化能力尚待进一步验证。

---

## 271. Parallel Swin Transformer-Enhanced 3D MRI-to-CT Synthesis for MRI-Only Radiotherapy Planning

**arXiv ID:** 2602.05387 | [PDF](https://arxiv.org/pdf/2602.05387v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 272. Robust Federated Learning via Byzantine Filtering over Encrypted Updates

**arXiv ID:** 2602.05410 | [PDF](https://arxiv.org/pdf/2602.05410v1)

**作者:** Adda Akram Bendoukha `[一作]` (Samovar), Sébastien Gambs `[通讯]` (Université du Québec à Montréal)

**通讯引用:** 2985 | [OpenAlex ID](https://openalex.org/A5000861121)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过结合同态加密与属性推断元分类器，对联邦学习中加密的模型更新进行拜占庭过滤，实现隐私保护与鲁棒性的统一。

**💡 创新点**

首次将属性推断攻击方法迁移为过滤器，并在加密域直接识别与抑制拜占庭更新，显著提升了安全聚合的效率与准确性。

**🔧 技术方法**

采用CKKS同态加密、SVM过滤器、SPCA降维、Chebyshev近似Sigmoid、Newton–Raphson反演等技术实现加密下的高效推断与聚合。

**📊 数据集**

使用CIFAR‑10、CIFAR‑100、MNIST、GTSRB、ACSIncome等公开数据集，并通过自生成的shadow数据集训练过滤器。

**📈 对比分析**

在Backdoor、梯度上升、标签翻转/打乱等多种攻击下与无防御或传统稳健聚合对比，F1分数在90%–94%，加密推断耗时约6–24秒，模型精度下降不足10%。

**⚠️ 局限性**

同态运算开销仍高，尤其是高阶核导致时延升高；对更大模型和更复杂攻击的鲁棒性仍需进一步验证。

---

## 273. Clinical Validation of Medical-based Large Language Model Chatbots on Ophthalmic Patient Queries with LLM-based Evaluation

**arXiv ID:** 2602.05381 | [PDF](https://arxiv.org/pdf/2602.05381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 274. OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration

**arXiv ID:** 2602.05400 | [PDF](https://arxiv.org/pdf/2602.05400v1)

**作者:** Shaobo Wang `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14237 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为OPUS的动态数据选择框架，在大语言模型预训练中根据优化器的实际更新方向对训练样本进行评分与采样，从而提升训练效率与模型性能。

**💡 创新点**

创新点包括：①将数据选择的效用定义为在优化器诱导的更新几何下对代理分布的损失改进；②通过Ghost梯度与CountSketch实现对梯度信息的低维近似，显著降低计算开销；③采用Boltzmann采样保持多样性；④构造Bench-Proxy使代理梯度更贴近评测任务。

**🔧 技术方法**

技术方法包括：Ghost梯度分解、CountSketch随机投影、优化器特定的预处理器（AdamW、Muon）以及Boltzmann软采样。

**📊 数据集**

使用的主要数据集有FineWeb、FineWeb-Edu（高质量子集）、SciencePedia（持续预训练），并在多个通用与专业基准（MMLU、ANLI、HellaSwag等）上评测。

**📈 对比分析**

与静态过滤器（QuRating、DSIR、FineWeb-Edu Classifier等）以及动态选择方法（PPL、GREATS）相比，OPUS在30B更新标记预算下平均提升约2.2%准确率，并在GPT‑2 XL、Qwen3‑8B等模型上实现与使用200B标记相当的性能，且在持续预训练中仅需0.5B标记即可击败3B标记的随机训练。

**⚠️ 局限性**

局限性包括：对代理池构建与更新有额外成本；在不同优化器或更大模型规模下的泛化尚需验证；仍需保证Ghost与CountSketch近似的稳定性；以及在极端低质量数据场景下可能无法充分发挥优势。

---

## 275. Late-to-Early Training: LET LLMs Learn Earlier, So Faster and Better

**arXiv ID:** 2602.05393 | [PDF](https://arxiv.org/pdf/2602.05393v1)

**作者:** Ji Zhao `[一作]` (Hong Kong University of Science and Technology), Zeke Xie `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 253 | [OpenAlex ID](https://openalex.org/A5066773635)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Late-to-Early Training (LET) 方法，利用小型预训练模型的晚期层表示来对齐并指导大模型的早期训练，从而加速收敛并提升下游性能。

**💡 创新点**

创新点在于将小模型的晚层特征映射到大模型的早层，并在训练初期使用该小模型，同时通过线性衰减的对齐损失实现动态权重衰减，使得大模型在不受限于大模型教师的前提下能够更快、更好地学习。

**🔧 技术方法**

使用Transformer（LLaMA）架构、RMSNorm、SwiGLU激活、BF16精度，并结合余弦相似度对齐损失、λ权重衰减、AdamW优化器和余弦学习率调度等技术。

**📊 数据集**

主要在The Pile大规模文本语料上进行预训练，并在九个下游任务集（ARC、SciQ、BoolQ 等）上进行一次性评估。

**📈 对比分析**

与标准因果语言模型、SALT 及 RKD 对比，LET 在 1.4B 与 7B 模型上平均准确率提升约 1.6 倍，同时训练步数减少约 33%，并在语言建模任务上也显著降低 perplexity。

**⚠️ 局限性**

局限包括：相比基线吞吐量略低，实验规模受限于 1–7B 参数，且需进一步验证更大规模模型（70B+）和更大数据集（1T token）下的可扩展性与实用性。

---

## 276. Spider-Sense: Intrinsic Risk Sensing for Efficient Agent Defense with Hierarchical Adaptive Screening

**arXiv ID:** 2602.05386 | [PDF](https://arxiv.org/pdf/2602.05386v1)

**作者:** Zhenxiong Yu `[一作]` (Shanghai University of Finance and Economics), Liwen Zhang `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 5974 | [OpenAlex ID](https://openalex.org/A5100459595)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于Intrinsic Risk Sensing（IRS）的事件驱动防御框架，允许LLM代理在感知到风险时才触发防御，并采用分层防御机制。

**💡 创新点**

创新点在于把代理安全从强制检查转为内在选择性检测，结合轻量级相似度匹配与深度内部推理，消除对外部模型的依赖。

**🔧 技术方法**

使用事件驱动架构、相似度匹配技术、内部深度推理网络和分层防御策略。

**📊 数据集**

使用了新构建的S^2Bench生命周期感知基准，涵盖真实工具执行和多阶段攻击场景。

**📈 对比分析**

与现有基于强制检查的防御方法比较，实验显示在攻击成功率(ASR)和误报率(FPR)上取得最低值，且延迟开销仅为8.3%。

**⚠️ 局限性**

局限性包括对未知攻击类型的适应性待验证，内部推理的计算成本与模型规模相关，实际部署环境下的可扩展性需要进一步研究。

---

## 277. Speech-XL: Towards Long-Form Speech Understanding in Large Speech Language Models

**arXiv ID:** 2602.05373 | [PDF](https://arxiv.org/pdf/2602.05373v1)

**作者:** Haoqin Sun `[一作]` (TMCC, College of Computer Science, Nankai University), Yong Qin `[通讯]` (TMCC, College of Computer Science, Nankai University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出Speech-XL框架，使用Speech Summarization Token（SST）对长音频进行KV压缩，实现高比例压缩后仍保持语义与声学信息；

**💡 创新点**

创新点在于将SST作为可学习的压缩单元，并采用逐步提升压缩比例的课程学习，显著降低Transformer的记忆和计算成本；

**🔧 技术方法**

技术方案包括Whisper‑small声学编码器、Dual‑Adapter桥接、Qwen2.5‑7B‑Instruct LLM，并在SST层实现KV压缩与动态课程调度；

**📊 数据集**

使用10k小时Emilia语音数据进行ASR预训练，127小时多任务语音数据（VCTK、Accentdb、IEMOCAP、DailyTalk、VoxCeleb1）进行细化，最后在LongSpeech数据上训练压缩模块；

**📈 对比分析**

在LongSpeech基准上，Speech‑XL将WER降至11.4%（接近11.0%上限），在AudioMarathon综合得分48.9，压缩率4×时与上限差距仅1.4%，相较于AudioFlamingo3、Voxtral、DashengLM等传统方法性能大幅提升；

**⚠️ 局限性**

局限性主要在于基础模型规模与数据量不足，难以与大规模预训练模型（如Qwen2.5‑Omni）竞争；此外对非语音音频的泛化能力仍有限。

---

## 278. Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting

**arXiv ID:** 2602.05371 | [PDF](https://arxiv.org/pdf/2602.05371v1)

**作者:** Hongyi Li `[一作]` (Harbin Institute of Technology), Jun Xu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 13772 | [OpenAlex ID](https://openalex.org/A5020766468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种倾斜回归树（HRT），将节点分裂问题转化为包含两个线性模型的非线性最小二乘问题，并通过 hinge 函数实现 ReLU 样式的分段线性表达，从而兼顾可解释性和强表达力。

**💡 创新点**

创新点在于：①把分裂建模为可解析的 Gauss–Newton（阻尼牛顿）更新；②给出节点级收敛证明并提供 O(δ²) 的逼近率；③采用最大/最小包络层次化实现高效倾斜决策边界，保持树结构的透明性。

**🔧 技术方法**

主要技术包括：非线性最小二乘（Gauss–Newton/阻尼牛顿）、线性最小二乘与可选 Ridge 正则化、回溯线搜索、递归树构造、以及理论证明与数值实验。

**📊 数据集**

数据集涵盖：合成二维/三维函数；公开回归基准（Abalone、CPUact、Ailerons、CSlice、YearPred、Concrete、Airfoil、MSLR、Fried、D‑Elevators、D‑Ailerons、Kinematics、C&C、Blog）以及工业级大规模数据。

**📈 对比分析**

与 CART、XGBoost、DTSemNet、DGT、TAO、M5、LinearTree 等单棵树/集成方法对比，HRT 在大多数数据集上取得最低或相近的 RMSE，同时树深度和叶子数显著更小，训练时间更短。

**⚠️ 局限性**

局限性包括：仅适用于回归单棵树，未直接处理多分类/结构化输出；在极高维或严重不平衡数据上仍可能收敛慢；未研究将 HRT 纳入随机森林或提升等集成框架的进一步提升。

---

## 279. PACE: Defying the Scaling Hypothesis of Exploration in Iterative Alignment for Mathematical Reasoning

**arXiv ID:** 2602.05370 | [PDF](https://arxiv.org/pdf/2602.05370v1)

**作者:** Jun Rao `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 59784 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为PACE的迭代对齐框架，使用极小的探索预算（N≈2）结合自我纠错生成高质量的“硬负样本”，从而实现数学推理任务的高效对齐。

**💡 创新点**

①将Best‑of‑N的高计算预算压缩到仅2–3个样本，消除标签噪声与分布偏移；②通过“Trace‑Aware”回溯纠错，将错误轨迹与真值结合生成近端正样本，形成Hard Negative对；③加入一致性门控过滤逻辑不一致的修正，提升梯度质量。

**🔧 技术方法**

采用迭代直接偏好优化（Iterative DPO）与生成式自纠正（trace‑aware refinement），结合一致性筛选与对比损失；并给出False Positive Amplification与Distribution Shift的理论分析。

**📊 数据集**

在多个数学推理基准上验证：Math、Minerva Math、Gaokao 2023 En、Olympiad Bench、AMC23。

**📈 对比分析**

与标准DPO‑R1（N=16、N=8、N=4、N=2）及其低预算对比，PACE在所有基准上保持或超过高N版本的性能，同时计算成本约为高N的1/5，速度提升约5×；在20%标签噪声实验中更具鲁棒性。

**⚠️ 局限性**

仍依赖可靠的验证器/奖励模型；目前仅验证于确定性数学推理任务，未针对开放式生成任务；对极端难题的纠错能力可能有限。

---

## 280. TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions

**arXiv ID:** 2602.05414 | [PDF](https://arxiv.org/pdf/2602.05414v1)

**作者:** Ngoc Doan-Minh Huynh `[一作]` (Sungkyunkwan University), Jae Wook Jeon `[通讯]` (Sungkyunkwan University)

**通讯引用:** 5175 | [OpenAlex ID](https://openalex.org/A5024137527)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集并构建了一个覆盖四季、八类交通主体、包含极端天气与高遮挡的CCTV监控视频数据集TSBOW，并提供3.2M帧的标注。

**💡 创新点**

创新点在于：①引入浓雾、暴雪等极端天气及灾难场景下的遮挡车辆检测；②使用半自动迭代标注流程；③将八类交通主体细粒度标注并公开。

**🔧 技术方法**

采用YOLOv8/11/12以及RT‑DETR等目标检测模型进行基线训练，并通过YOLOv12x进行半自动标注。

**📊 数据集**

主要使用TSBOW数据集，同时与UAVDT、UA‑DETRAC等公开数据集进行对比。

**📈 对比分析**

在TSBOW训练的YOLOv12x在测试集上获得mAP50≈0.744、mAP50‑95≈0.615，明显优于在UAVDT/UA‑DETRAC训练的模型；RT‑DETR在召回率最高但精度与mAP略低。

**⚠️ 局限性**

局限包括：仅覆盖白天场景，夜间及部分极端天气下的标注比例不足；缺乏多目标跟踪、语义分割等任务的标注；模型仍对高遮挡、粗尺度物体的检测表现不佳。

---

## 281. Synthetic Defect Geometries of Cast Metal Objects Modeled via 2d Voronoi Tessellations

**arXiv ID:** 2602.05440 | [PDF](https://arxiv.org/pdf/2602.05440v1)

**作者:** Natascha Jeziorski `[一作]` (RPTU University Kaiserslautern-Landau), Claudia Redenbach `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了基于Voronoi分割的参数化模型，用以合成各类铸造金属表面缺陷（裂纹、弯曲、隆起、脱层等），并将生成的缺陷嵌入到数字双胞胎模型中，利用物理基础蒙特卡洛渲染生成可像真实检测数据的合成图像，同时实现像素级自动标注。

**💡 创新点**

创新点在于：①将多种缺陷归结为相同的最短路径+膨胀+三维网格化流程，便于通过参数控制缺陷形状、尺寸、分支等特征；②采用规则化（非学习）生成方法，可系统覆盖稀有或极端缺陷；③在生成过程中自动提供完美注释，为深度学习提供大规模、可控的数据集。

**🔧 技术方法**

技术实现包括：Voronoi分割、Dijkstra最短路径、路径膨胀与三角化、三维网格化、布尔运算（trimesh）、Blender物理基础渲染、蒙特卡洛光线追踪，辅以Python实现的参数采样与自动标注脚本。

**📊 数据集**

使用者仅依赖自身的产品几何模型（3D网格）与缺陷参数空间；论文未采用公开真实缺陷数据集，而是完全在仿真环境中生成合成数据。

**📈 对比分析**

实验比较主要基于已有文献（Fulir等）指出：在缺乏足够真实数据时，用本方法生成的合成数据可显著提升基于深度学习的缺陷检测性能；但本文未给出具体数值评估，只展示了示例图像与模型可视化。

**⚠️ 局限性**

局限性包括：①缺陷模型虽可控但不一定完全符合真实铸造物理过程；②对参数设置需要人工经验，可能影响生成多样性；③生成三角网格时若路径细节过细会出现薄三角，影响后续插入；④渲染成本高，生成大规模高分辨率数据仍耗时；⑤模型主要针对铸造金属表面，迁移到其他工艺需进一步调整。

---

## 282. THOR: Inductive Link Prediction over Hyper-Relational Knowledge Graphs

**arXiv ID:** 2602.05424 | [PDF](https://arxiv.org/pdf/2602.05424v1)

**作者:** Weijian Yu `[一作]` (University of Macau), Dingqi Yang `[通讯]` (University of Macau)

**通讯引用:** 4952 | [OpenAlex ID](https://openalex.org/A5026625771)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为 THOR 的全新的可完全归纳的超关系知识图谱链接预测方法。

**💡 创新点**

创新点在于设计了关系和实体基础图来捕捉超关系图中不同位置间的基本交互，并通过双重 NBFNet 编码器与 Transformer 解码器实现无负样本、可掩码训练，突破了传统转导模型对词表的依赖。

**🔧 技术方法**

使用了 NBFNet（基于路径的图神经网络）、Transformer 的边加权自注意力机制、双重等变形理论、以及自定义的基础图结构。

**📊 数据集**

实验数据集包括 12 个超关系图数据集，涵盖半归纳的 WD20K（100%、66%、33% 版本）、完全归纳的 WDSPLIT100、JFFI 以及对应的 100% 超关系版本。

**📈 对比分析**

与 16 种基线（规则、半归纳、完全归纳、转导）在 12 个数据集上进行全面对比，THOR 在最佳规则基线提升 66.1%，最佳半归纳基线提升 55.9%，最佳完全归纳基线提升 20.4%，在跨域完全归纳设置中更是领先 32.5%。

**⚠️ 局限性**

局限性包括对极大规模知识图谱的可扩展性待验证、对更复杂超关系模式的处理仍依赖手工构造的基础交互，且在引入图形子结构（motif）时性能会下降。

---

## 283. Grammatical Error Correction Evaluation by Optimally Transporting Edit Representation

**arXiv ID:** 2602.05419 | [PDF](https://arxiv.org/pdf/2602.05419v1)

**作者:** Takumi Goto `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1622 | [OpenAlex ID](https://openalex.org/A5102396915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于编辑向量的语法错误纠正自动评估指标 UOT-ERRANT，用以衡量预测句子与参考句子之间编辑的相似度。

**💡 创新点**

创新点在于将每一次编辑表示为句子表示差分向量，并利用非平衡最优运输（UOT）在编辑向量空间中进行软对齐，从而得到可解释的精确、召回与 F0.5 分数。

**🔧 技术方法**

使用 ELECTRA 句子编码器生成句子向量，计算编辑向量；采用非平衡最优运输与 Sinkhorn 算法求解运输计划；基于此得到的对齐矩阵进行评估。

**📊 数据集**

使用 SEEDA（14 系统、包含 GPT‑3.5、REF‑M/F 等）和 GMEG‑Data（Wiki 领域）两大元评估数据集，并在 SEEDA 上使用多种参考集合（Official、10Refs、E‑Minimal、NE‑Minimal、E‑Fluency、NE‑Fluency）。

**📈 对比分析**

与 ERRANT、PT‑ERRANT、GLEU、GREEN、CLEME、SOME、Scribendi、IMPARA、LLM‑S/E 等指标对比，实验显示在 SEEDA‑E 的 +Fluency 设置下，UOT‑ERRANT 在 Pearson 与 Spearman 与人工评估的相关性均位列第一，平均排名显著提升；在 GMEG‑Data 由于标点编辑占比高，相关性略逊。

**⚠️ 局限性**

局限性：计算成本高（需对每个编辑做前向推理并计算 UOT），对标点编辑不敏感；在编辑量少或编辑多样性低的场景下提升有限；验证仅在 SEEDA、GMEG‑Data 等特定任务与数据集上完成。

---

## 284. H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration

**arXiv ID:** 2602.05407 | [PDF](https://arxiv.org/pdf/2602.05407v1)

**作者:** Jun-Min Lee `[一作]` (Korea Advanced Institute of Science and Technology), Edward Choi `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 7050 | [OpenAlex ID](https://openalex.org/A5034622258)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

建立了一个基于 FHIR 的多智能体门诊行政流程仿真框架，能合成医院、医生、患者数据并对 LLM 进行端到端评估。

**💡 创新点**

创新点包括：① 使用 194 对病症-症状与诊断史生成多样化患者；② 统一采用 FHIR 交互；③ 结合工具调用与纯推理的排程策略；④ 提供完整评估指标和基准。

**🔧 技术方法**

使用了多智能体 LLM（GPT‑5 系列、Gemini）、工具调用、FHIR、时间步长仿真和基于规则的评估体系。

**📊 数据集**

使用全合成数据集，涵盖 9 个内科门诊、194 病症-症状对、不同医院级别配置及医生/患者属性，未使用真实数据。

**📈 对比分析**

通过 10 条评估指标与 13 码错误，对 3 个模型在 3 级医院的 intake 与 Scheduling (T/R) 成功率进行比较；结果显示 Gemini 2.5 Flash 最高，tool‑based 调度稳定，而 intake 仍为瓶颈，性能随医院级别和先前诊断率变化。

**⚠️ 局限性**

局限性：仅覆盖首诊门诊、仿真与真实场景差距、对工具的高度依赖、缺乏多语言和复杂临床情境、模型对话策略未充分优化。

---

## 285. Enabling Large-Scale Channel Sounding for 6G: A Framework for Sparse Sampling and Multipath Component Extraction

**arXiv ID:** 2602.05405 | [PDF](https://arxiv.org/pdf/2602.05405v1)

**作者:** Yi Chen `[一作]` (Dalian University of Technology), Chong Han `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8947 | [OpenAlex ID](https://openalex.org/A5048916368)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于稀疏非均匀采样的通道声学框架，包括抛物线频率采样（PFS）和修正的SAGE算法（LR‑SAGE），实现6G THz频段大规模高效通道测量与多径参数提取。

**💡 创新点**

创新点在于通过设计抛物线频率采样消除延迟模糊、显著减少采样点数，并通过对多径抽取过程中的似然函数进行频率/延迟相关校正，恢复吸收导致的幅度失真。

**🔧 技术方法**

技术手段包括非均匀频率采样、Poisson求和与驻点法理论分析、LR‑SAGE算法的似然修正因子、分子吸收模型与Beer‑Lambert定律。

**📊 数据集**

使用了自研的280–300 GHz VNA ISAC声学装置采集的实测数据，以及基于仿真的多径信道合成数据。

**📈 对比分析**

与传统均匀频率采样+原始SAGE相比，PFS+LR‑SAGE在K≈251时即可获得与12001点均匀采样相当的功率‑延迟‑角度谱，测量时间、数据量和计算复杂度分别下降约98%、50×和99.96%，延迟估计RMSE低于10⁻² ns。

**⚠️ 局限性**

局限性包括对极端吸收环境仍需足够多采样点；在高噪声或多径严重互相干扰的场景下，E步残余干扰可能导致估计偏差；以及对非线性相位误差和硬件非理想的鲁棒性待进一步验证。

---

## 286. Advancing Opinion Dynamics Modeling with Neural Diffusion-Convection-Reaction Equation

**arXiv ID:** 2602.05403 | [PDF](https://arxiv.org/pdf/2602.05403v1)

**作者:** Chenghua Gong `[一作]` (University of Science and Technology of China), Linyuan Lü `[通讯]` (University of Science and Technology of China)

**通讯引用:** 13550 | [OpenAlex ID](https://openalex.org/A5000969982)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个名为 Opinn 的物理信息神经网络框架，用于建模和预测社交网络中的舆情演化。

**💡 创新点**

创新点在于将扩散–对流–反应（DCR）物理系统与 Neural ODE 结合，既提供可解释的物理先验，又保持数据驱动的灵活性。

**🔧 技术方法**

采用的技术包括 Graph Convolutional Network、注意力机制、可学习门控系数、Neural ODE 以及 RK‑4 数值求解器。

**📊 数据集**

实验使用四个真实世界数据集（Delhi Election、U.S. Election、Israel‑Palestine、COVID‑19）和三种合成数据集（共识、极化、聚类）进行评估。

**📈 对比分析**

与 16 种基线（机械模型、纯数据驱动模型、其他物理信息模型）比较，Opinn 在 RMSE/MAE 上平均提升 2‑8%，例如在 Israel‑Palestine 数据集上 MAE 降低 8.62%，在少量样本场景下亦保持领先。

**⚠️ 局限性**

局限性包括：对流模块的 O(N²) 复杂度导致大规模网络的计算瓶颈；仅使用从文本提取的舆情得分，缺乏人口学等辅助特征；对真实世界数据的获取和仿真困难。

---

## 287. Dolphin-v2: Universal Document Parsing via Scalable Anchor Prompting

**arXiv ID:** 2602.05384 | [PDF](https://arxiv.org/pdf/2602.05384v1)

**作者:** Hao Feng `[一作]`, Can Huang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Dolphin‑v2，一种针对数字化文档和拍摄文档的两阶段统一文档解析框架；

**💡 创新点**

创新点包括联合文档类型分类与布局分析、细粒度21类元素检测与语义属性提取、拍摄文档的整体页面解析与数字文档的并行元素解析，以及公式与代码块的专属解析模块；

**🔧 技术方法**

采用基于Qwen2.5‑VL的视听语言模型，利用NaViT原生分辨率视觉编码、统一解码器与专用提示词，实现端到端的布局与内容生成；

**📊 数据集**

使用合成的200K拍摄文档、200K代码图像、200K目录图像以及真实拍摄的RealDoc‑160、DocPTBench和OmniDocBench三个评测基准；

**📈 对比分析**

与多种通用与专业VLM及管线模型对比，Dolphin‑v2在OmniDocBench整体得分从74.67提升至89.78，拍摄文档错误率降低91%，在RealDoc‑160的编辑距离均值仅0.039，显著优于现有方法；

**⚠️ 局限性**

局限性在于仍需人工标注丰富的合成数据、对极端光照/畸变场景的鲁棒性有待进一步提升，并未覆盖多语言混排与更复杂交互式元素。

---

## 288. SAIL: Self-Amplified Iterative Learning for Diffusion Model Alignment with Minimal Human Feedback

**arXiv ID:** 2602.05380 | [PDF](https://arxiv.org/pdf/2602.05380v1)

**作者:** Xiaoxuan He `[一作]` (WeChat Vision, Tencent Inc), Bo Zhang `[通讯]` (ZheJiang University)

**通讯引用:** 30595 | [OpenAlex ID](https://openalex.org/A5107117609)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SAIL框架，通过闭环自反馈和自奖励直接优化，利用少量种子偏好数据对扩散模型进行人类偏好对齐，无需外部奖励模型。

**💡 创新点**

①自奖励直接优化，将扩散模型自身生成与评价结合；②排名偏好Mixup策略防止灾难性遗忘；③仅使用6%偏好数据即可超过全量数据的现有方法。

**🔧 技术方法**

基于DiffusionDPO、直接偏好优化（DPO）技术，利用Stable Diffusion 1.5/XL、噪声预测与奖励计算、闭环自我强化以及混合偏好数据策略。

**📊 数据集**

使用Pick-a-Pic v2、HPSv2、PartiPrompts等偏好数据集，并以少量种子偏好对照集作为起始。

**📈 对比分析**

与DiffusionDPO、DiffusionSPO、MaPO等方法比较，SAIL在SD1.5/XL上仅用0.05M样本（6%）实现了PickScore、ImageReward、Aesthetics、HPSv2等指标提升0.3–0.5个百分点，性能优于基线。

**⚠️ 局限性**

目前仅在图像域验证，视频生成的偏好数据收集困难；对高复杂度场景仍有提升空间；需要进一步探索多任务奖励平衡与更丰富的自我反馈机制。

---

## 289. Erase at the Core: Representation Unlearning for Machine Unlearning

**arXiv ID:** 2602.05375 | [PDF](https://arxiv.org/pdf/2602.05375v1)

**作者:** Jaewon Lee `[一作]` (Korea University), Donghyun Kim `[通讯]` (Korea University)

**通讯引用:** 21003 | [OpenAlex ID](https://openalex.org/A5100719069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多层对比学习与深度监督的机器无学习框架 EC，以实现从层级到特征的彻底遗忘

**💡 创新点**

首次将对比无学习扩展到网络深层并结合跨层交叉熵监督，显著提升表示层遗忘效果

**🔧 技术方法**

多层对比无学习（Contrastive Unlearning）+深度监督+层加权交叉熵

**📊 数据集**

ImageNet-1K 与 CIFAR-100，使用 ResNet‑50 与 Swin‑Tiny

**📈 对比分析**

与 PL、DUCK、CU 等基线对比，EC 在 Logit 级别保持几乎零忘记准确率的同时，CKA、IDI 等表示度量显著降低，H‑Mean 最高

**⚠️ 局限性**

仅为经验方法，缺乏理论保证，且在某些基线上略有性能下降

---

## 290. Cross-Lingual Empirical Evaluation of Large Language Models for Arabic Medical Tasks

**arXiv ID:** 2602.05374 | [PDF](https://arxiv.org/pdf/2602.05374v1)

**作者:** Chaimae Abouzahir `[一作]` (New York University), Farah E. Shamout `[通讯]` (New York University)

**通讯引用:** 930 | [OpenAlex ID](https://openalex.org/A5023660328)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对比分析了大语言模型在阿拉伯语与英语医学问答任务上的表现，提出并验证了跨语言诊断评估框架，并探讨了词分词、任务难度与模型信心对性能的影响。

**💡 创新点**

创新点在于：①构建了一个可控的跨语言评估框架，可同时评估多种输出格式与可靠性信号；②通过对阿拉伯医学文本的分词碎片化进行量化，首次揭示语言结构对模型性能的具体影响；③系统评估了模型自评信心与生成解释的可靠性，指出其在多语言医学问答中的局限性。

**🔧 技术方法**

使用了多大规模开源 LLM（DeepSeek‑V3.2、LLaMA‑3.3‑70B、Mistral‑Small‑3.2‑24B、Meditron‑3‑70B、Med42‑70B、MedGemma‑27B‑text‑it），并结合了多种评估技术：多选答案对比、字母匹配、文本级序列相似度、tokenization 统计与信心/解释相关性分析。

**📊 数据集**

数据集为 MedAraBench（阿拉伯语医学 MCQ）及其自动翻译成英语的版本，包含 19 种医学专业、不同难度等级的 4,989 道测试题。

**📈 对比分析**

比较方法：在相同提示、解码设置下分别评估阿拉伯语和英语版本，统计准确率、差距 Δ；同时分析输入长度、难度、专业对准确率的影响；使用 token‑level 匹配评估自由文本答案；对模型自评信心与准确率的相关性做 Pearson 相关分析。结果显示，除 DeepSeek‑V3.2 外，大多数模型在阿拉伯语上的准确率比英语低 10–20%，且性能随问题长度与难度增加而衰退更为明显；token‑level 匹配误差更大，说明语言结构导致表面形式差异；信心与准确率负相关，说明自评不可靠。

**⚠️ 局限性**

局限性包括：①仅以阿拉伯语–英语双语为研究对象，未检验更广泛语言；②缺乏因果实验，无法精准归因于训练策略、数据分布或模型架构；③使用 4‑bit 量化评估大型模型，可能引入额外误差；④自动翻译的英语版本未人工校对，翻译噪声可能影响结果；⑤提示语言采用混合模式（英文提示+阿拉伯语输入），未系统比较纯阿拉伯提示效果。

---

## 291. Breaking Semantic Hegemony: Decoupling Principal and Residual Subspaces for Generalized OOD Detection

**arXiv ID:** 2602.05360 | [PDF](https://arxiv.org/pdf/2602.05360v1)

**作者:** Ningkang Peng `[一作]` (Nanjing Normal University), Yanhui Gu `[通讯]` (Nanjing Normal University)

**通讯引用:** 618 | [OpenAlex ID](https://openalex.org/A5100749023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究深度特征空间中由神经坍塌导致的语义霸权现象，揭示了OOD检测中的“简单悖论”，并提出一种无训练、可即插即用的双空间KNN（D‑KNN）框架，通过正交分解与双空间校准恢复残差信息，从而显著提升检测性能。

**💡 创新点**

创新点在于：①首次将语义霸权与神经坍塌联系起来解释OOV检测的失败；②提出正交分解将特征分为语义主空间与结构残差空间；③设计双空间校准（Z‑score）激活残差信号；④实现了训练无关、可在多种模型上直接部署的D‑KNN方法。

**🔧 技术方法**

主要技术包括：正交分解（PCA）、特征投影至单位球面、残差空间投影、KNN距离度量、双空间Z‑score校准、融合权重α。

**📊 数据集**

实验数据集涵盖：CIFAR‑10、CIFAR‑100、ImageNet‑1K、MNIST、EMNIST、SVHN、LSUN、iSUN、Textures、ImageNet‑O、iNaturalist、Places、SUN、OpenImage‑O、NINCO、CIFAR‑100‑C等，既有标准OOD任务，也包括结构简单样本和传感器噪声情景。

**📈 对比分析**

与MSP、Energy、MaxLogit、Mahalanobis、ViM、KNN、DICE、Line、FDBD、NeCo、WDiscOOD等主流方法比较，D‑KNN在CIFAR‑10、CIFAR‑100、ImageNet‑1K等基准上显著降低FPR95并提升AUROC，例如CIFAR‑10 FPR95 17.5%（低于Energy 23.3%、KNN 30.3%）、CIFAR‑100 FPR95 26.7%（低于KNN 36.8%）、ImageNet‑1K平均FPR95 54.6%（低于KNN 57.1%、Energy 64.1%），在MNIST简单样本上FPR95降至2.3%（vs 31.3% KNN），在噪声任务中AUROC提升至94.9%（vs 79.7% Energy）。

**⚠️ 局限性**

局限性包括：①对残差子空间的噪声分量仍可能受限，需足够模型容量以充分实现神经坍塌；②低容量模型提升有限；③需要调节融合权重α，尽管对性能影响相对稳健；④在非图像任务或极端高维场景下的通用性尚未验证。

---

## 292. A Unified Multimodal Framework for Dataset Construction and Model-Based Diagnosis of Ameloblastoma

**arXiv ID:** 2602.05515 | [PDF](https://arxiv.org/pdf/2602.05515v1)

**作者:** Ajo Babu George `[一作]` (DiceMed), Balu Bhasuran `[通讯]` (School of Information, Florida State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了专门针对牙母细胞瘤的多模态数据集，并基于该数据集开发了多任务深度学习模型，用于诊断分类、变异预测及手术规划支持。

**💡 创新点**

创新点在于：①采用开放式病例报告自动提取结构化文本信息的多模态数据构建流程；②结合医学自然语言处理（关键词匹配、Word2Vec、BioBERT、Gemini LLM）实现高质量结构化标注；③提出基于Sentence‑BERT+FAISS的病例检索系统，实现语义相似度匹配；④多任务网络（DenseNet121+多头）同时预测病理异常、诊断类别与肿瘤变异。

**🔧 技术方法**

技术主要包括：Python API、OpenCV、PIL、PyTorch、DenseNet121、Sentence‑BERT、FAISS、BioBERT、Gemini LLM、Word2Vec、TF‑IDF、K‑Means、数据增强（旋转、翻转）。

**📊 数据集**

数据集为：①自构建的牙母细胞瘤多模态集（152例，1152张图；包含放射、病理、口腔照片、图表），②公开的多模态基线 MultiCaRe 数据集（用于图像子集生成），③Kaggle 病理图像数据集（补充典型病理样本）。

**📈 对比分析**

与现有 MultiCaRe 资源对比，利用预处理后模型在诊断准确率、变异分类、异常检测等指标均显著提升（例如异常检测 F1 从 0.43 提升至 0.90，变异分类准确率从 0.46 提升至 0.66），统计检验表明差异显著；检索系统采用 Sentence‑BERT+FAISS 在速度与语义匹配上优于 TF‑IDF、K‑Means 等传统方法。

**⚠️ 局限性**

局限性：①病例报告中图像与标题标签不一致导致标注噪声；②稀有变异样本不足，导致分类准确率受限；③模型对新型病例的外部验证尚缺；④数据规模相对有限，未能充分探索更大规模多模态模型。

---

## 293. Statistical Verification of Medium-Access Parameterization for Power-Grid Edge Ad Hoc Sensor Networks

**arXiv ID:** 2602.05510 | [PDF](https://arxiv.org/pdf/2602.05510v1)

**作者:** Haitian Wang `[一作]` (University of Western Australia), Yihao Ding `[通讯]` (University of Western Australia)

**通讯引用:** 38 | [OpenAlex ID](https://openalex.org/A5074613052)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于随机时钟混合自动机与统计模型检查的框架，用于在电网边缘无线传感网络中验证 IEEE 802.15.4 CSMA/CA 的参数化，能够在存在自利节点行为的异步、事件驱动、能量受限环境下正式评估协议配置。

**💡 创新点**

创新点在于将时序混合自动机与带置信区间的统计模型检查相结合，定义 Δ‑松弛纳什均衡以校准协议参数；同时通过两阶段自动化筛选与统计认证，兼顾可靠性、能耗与网格约束，实现对自利策略鲁棒性的正式保证。

**🔧 技术方法**

采用的技术包括：随机时钟混合自动机（STHA）建模、UPPAAL SMC 进行统计模型检查、贝叶斯置信区间（Clopper‑Pearson）与 Bonferroni 校正、两阶段协议筛选与 Δ‑松弛纳什均衡判定。

**📊 数据集**

使用基于真实电网事件频率和维护窗口生成的合成负载；每个节点周期内需要传输状态帧（45 s 间隔）及在异常事件时在 3 s 内完成报警；通过 10 000 次屏蔽运行与 100 000 次认证运行获得统计结果。

**📈 对比分析**

与两种经验性基准（E 与 F）及最优对称配置（C、D）比较，认证的 NE 配置 A 与 B 在 0.862→0.914 的效用提升、89.5 %→93.2 % 的交付率提升以及 152.8 mJ→149.2 mJ 的能耗下降；同时满足 90 % 可用率与 150 mJ 能耗上限，鲁棒系数 ≥0.97。

**⚠️ 局限性**

局限性包括：模型对大规模网络扩展时的计算量有限；未考虑多信道干扰、设备异质性与空间拓扑影响；安全与故障恢复机制未纳入；因此在更复杂的实际部署中仍需进一步验证与扩展。

---

## 294. Report on the second Toulouse Tensor Workshop

**arXiv ID:** 2602.05490 | [PDF](https://arxiv.org/pdf/2602.05490v1)

**作者:** Jan Brandejs `[一作]` (Université de Toulouse), Paolo Bientinesi `[通讯]` (Umeå University)

**通讯引用:** 1481 | [OpenAlex ID](https://openalex.org/A5048393932)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本报告记录了2025年在图卢兹举行的第二届张量研讨会，讨论并评估了新提出的低级张量收缩接口TAPP及其参考实现，并对后续标准化工作进行了规划。

**💡 创新点**

创新点在于：①首次推出针对张量收缩的统一低级API（TAPP）并实现可扩展的参考实现；②通过社区调研（80名受访者）聚焦张量收缩的性能瓶颈与需求，确定后续标准化重点；③提出在TAPP中引入块稀疏（block sparsity）支持的可能方案，为后续对称性、稀疏性和分布式内存等高级特性的标准化奠定基础。

**🔧 技术方法**

使用技术包括：TAPP低级API（基于C/C++），后端实现如TBLIS、cuTENSOR、ROCm/hiptensor；Python/Julia等高级接口（einsum、PyTorch、NumPy）；自动微分框架；分布式张量库（Cyclops、TiledArray、CTF）；以及针对块稀疏的元数据传递与块管理机制。

**📊 数据集**

主要数据集来自两项工作：①对80位科研人员（以量子化学、凝聚态物理、量子计算等为主）进行的张量收缩使用习惯与性能影响调查；②DIRAC电子结构代码中使用的实际张量收缩案例，展示TAPP在真实应用中的可行性。

**📈 对比分析**

对比方式主要通过社区反馈与实验室案例：研讨会讨论表明在单节点、GPU加速环境下引入块稀疏可显著减少计算量；参考实现已在DIRAC中与现有库（e.g., cuTENSOR、TBLIS）进行性能对比，结果显示TAPP能与现有后端兼容，并支持多后端切换，性能与传统实现相当或更优。缺乏系统性基准测试，无法给出统一的数值指标。

**⚠️ 局限性**

局限性包括：①TAPP目前仅支持单节点张量收缩，尚未覆盖分布式内存；②块稀疏支持仍在规划阶段，API及后端实现尚未完成；③对更通用稀疏形式（如结构稀疏、对称性稀疏）的支持尚无标准；④在自动微分、低精度运算等前沿功能上的细节尚待完善；⑤缺乏大规模基准实验，性能评估主要来自案例展示与社区感知。

---

## 295. Refine and Purify: Orthogonal Basis Optimization with Null-Space Denoising for Conditional Representation Learning

**arXiv ID:** 2602.05464 | [PDF](https://arxiv.org/pdf/2602.05464v1)

**作者:** Jiaquan Wang `[一作]` (Southeast University), Yuheng Jia `[通讯]` (Southeast University)

**通讯引用:** 1712 | [OpenAlex ID](https://openalex.org/A5013880628)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无监督条件表示学习框架 OD-CRL，用于从视觉-文本模型中提取与指定条件（如颜色、形状等）一致的特征

**💡 创新点**

创新点在于通过自适应正交基优化（AOBO）消除文本基底的冗余与歧义，并通过零空间去噪投影（NSDP）抑制不同子空间间的相互干扰

**🔧 技术方法**

采用 SVD 进行正交基优化、曲率自适应截断、零空间投影、向量投影与对数回归/三元组损失等技术，整体基于 CLIP/BLIP‑2 的视觉‑语言模型

**📊 数据集**

使用 Clevr4‑10k、Cards、DeepFashion 等公开数据集进行聚类、少量样本分类和时尚检索任务评估

**📈 对比分析**

与 CLIP、CRL、Multi‑Map、Multi‑Sub 等基线方法对比，OD‑CRL 在聚类 NMI/ACC/ARI、少样本分类准确率和时尚检索 mAP 上均取得显著提升，甚至无需额外训练即可超越原始 CLIP

**⚠️ 局限性**

局限性包括：需要预训练视觉‑语言模型的推理成本；零空间投影可能会削弱部分目标语义；在极度不平衡或非常细粒度的条件下的表现尚待验证

---

## 296. Thermodynamic Limits of Physical Intelligence

**arXiv ID:** 2602.05463 | [PDF](https://arxiv.org/pdf/2602.05463v1)

**作者:** Koichi Takahashi `[一作]` (AI Alignment Network), Yusuke Hayashi `[通讯]` (AI Alignment Network)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了两轴 bits-per-joule 的物理智能度量：热力学 epiplexity per joule（学习/表示效率）和 empowerment per joule（控制效率），并给出闭循环热力学上限和开放边界解耦构造。

**💡 创新点**

创新点在于统一学习与控制的能量效率度量，明确能量会计、粗粒化、时间窗口、重置等边界规范，推导出 Landauer 尺度的闭循环极限，并提供可操作的 MDL/压缩增益替代方案。

**🔧 技术方法**

技术包括信息熵与热力学不等式、条件数据处理不等式、可逆计算与低熵资源计入、MDL 与计算受限的 MDL 估计、互信息容量与单位成本优化等。

**📊 数据集**

数据集方面：论文主要使用理论构造和模拟，若实验需用到，建议使用标准 RL 或视觉仿真环境，并在基准中给定环境实例 Z。

**📈 对比分析**

比较方法：在闭循环实验中报告 ΔI/E_cons 或 ΔI/Q_diss（学习效率）以及 empowerment/cost 比值（控制效率），并与 T ln 2 基准做对比；实验表明当前系统相距该极限若干阶，凸显算法与硬件开销。

**⚠️ 局限性**

局限性包括对边界会计与重置假设的严格依赖、对理论环境实例 Z 的假设、连续变量需要粗粒化/噪声模型、以及实际硬件耗能（冷却、通信）可能远高于理论热力学最低限。

---

## 297. DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching

**arXiv ID:** 2602.05449 | [PDF](https://arxiv.org/pdf/2602.05449v1)

**作者:** Chang Zou `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14253 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对大规模视频扩散模型，本文提出一种可与蒸馏兼容的可学习特征缓存（DisCa）和受限MeanFlow两大加速方案，实现显著的推理加速。

**💡 创新点**

创新点在于引入轻量级可学习神经预测器替代传统训练自由缓存，并通过裁剪MeanFlow的过度压缩区间实现更稳定、可控的蒸馏。

**🔧 技术方法**

使用的技术包括Diffusion Transformer架构、MeanFlow改进、可学习预测器、生成对抗训练（GAN）以及对比传统缓存方法（如TaylorSeer、Δ-DiT、PAB等）。

**📊 数据集**

实验基于HunyuanVideo（540p预训练模型）生成704×704×129帧、5秒视频，评估指标采用VBench的语义分数和质量分数。

**📈 对比分析**

与多种基线（Δ-DiT、PAB、FORA、TaylorSeer、MeanFlow）对比，DisCa在11.8×加速下仅损失5.7%语义分数、0.5%质量分数，总分仅下降1.4%，显著优于50步CFG蒸馏模型。

**⚠️ 局限性**

局限性在于极高压缩仍会有细节衰减，且方法目前仅在视频任务上验证，缺乏对其他模态（图像、音频等）的通用性验证。

---

## 298. Phi-Former: A Pairwise Hierarchical Approach for Compound-Protein Interactions Prediction

**arXiv ID:** 2602.05479 | [PDF](https://arxiv.org/pdf/2602.05479v1)

**作者:** Zhe Wang `[一作]` (Hong Kong University of Science and Technology), Yuan Yao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5086 | [OpenAlex ID](https://openalex.org/A5000492991)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为Phi‑Former的成对层次化表征学习框架，用于预测化合物‑蛋白质相互作用（CPI）。

**💡 创新点**

创新点在于：①将分子按原子和功能基（motif）两层级别建模，强调功能基在生物识别中的核心作用；②设计了三种交互损失（原子层、功能基层、以功能基为条件的原子层），实现不同层级的互补学习；③采用基于图Transformer的预训练与微调策略，利用距离自监督任务捕捉结构相互作用。

**🔧 技术方法**

使用技术包括：图Transformer编码器、基于高斯核的空间位置编码（SPE）、注意力偏置、三层级（原子/功能基/以功能基为先验的原子）编码器、距离自监督学习（masking原子-蛋白间距离）以及线性回归头进行亲和力预测。

**📊 数据集**

数据集：PDBBind 2019（训练集16493对）与CASF‑2016（275对测试集），通过去除重叠与无效数据得到的标准评价集。

**📈 对比分析**

与SIGN、IGN、VinaRF_20、OnionNet、Mol‑PSI、GraphDTA、K_Deep、SS‑GNN等现有方法对比，Phi‑Former在RMSE上取得最低1.159、R_p为0.846，表现优于大多数基线且仅次于SS‑GNN的0.853，显示出预训练层次化学习显著提升了预测精度。

**⚠️ 局限性**

局限性：①框架仅针对CPI任务，尚未验证到更大规模或其他相互作用任务；②模型复杂度较高，训练与推理成本相对较大；③在极端结构多样性或低数据量场景下的泛化能力尚需进一步评估。

---

## 299. Mapper-GIN: Lightweight Structural Graph Abstraction for Corrupted 3D Point Cloud Classification

**arXiv ID:** 2602.05522 | [PDF](https://arxiv.org/pdf/2602.05522v1)

**作者:** Jeongbin You `[一作]` (Korea University), Seungsang Oh `[通讯]` (Korea University)

**通讯引用:** 393 | [OpenAlex ID](https://openalex.org/A5064077787)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于Mapper的区域图和GIN消息传递的轻量级三维点云分类方法Mapper-GIN。

**💡 创新点**

创新点在于将Mapper拓扑抽象与图神经网络相结合，使用结构化区域图实现鲁棒性提升，而非依赖大模型或复杂的数据增强。

**🔧 技术方法**

技术包括Mapper算法（PCA视角、立方体覆盖、DBSCAN聚类）构造区域图、GIN网络进行结构感知聚合、以及局部坐标归一化的轻量级点编码。

**📊 数据集**

实验使用ModelNet40及其破坏版ModelNet40-C数据集进行评估。

**📈 对比分析**

与传统的MLP、PointNet和PointNet++相比，Mapper-GIN在模型参数仅0.5M的情况下，在模型网格破坏和噪声、变形等多种扰动下达到约75%的平均鲁棒准确率，几乎逼近PointNet++的性能。

**⚠️ 局限性**

局限在于对点删除或稀疏化等导致区域图结构变化的扰动鲁棒性不足，并且Mapper构造依赖手工设计的视角、覆盖和聚类超参，缺乏端到端可学习性。

---

## 300. Detecting Information Channels in Congressional Trading via Temporal Graph Learning

**arXiv ID:** 2602.05514 | [PDF](https://arxiv.org/pdf/2602.05514v1)

**作者:** Benjamin Pham Roodman `[一作]` (National Taiwan University), Chaun-Ju Wang `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了国会成员股票交易中的信息优势，并提出基于时间图网络的识别方法

**💡 创新点**

创新点在于构建多模态动态图并引入GAP‑TGN以处理长周期标签延迟及信息老化

**🔧 技术方法**

使用Gated Asynchronous Propagation Temporal Graph Network与多模态融合与注意力机制

**📊 数据集**

采用Capitol Gains数据集，融合国会交易、游说关系、竞选捐款、宏观经济与公司财报

**📈 对比分析**

对比传统Logistic/MLP/XGBoost，GAP‑TGN在18/24个月超额收益预测上F1提高约51%，AUROC略低于XGBoost

**⚠️ 局限性**

局限在于模型仍未加入节点/边强化，数据规模有限，AUROC差距仍大，且仅为初步实验

---

## 301. VGGT-Motion: Motion-Aware Calibration-Free Monocular SLAM for Long-Range Consistency

**arXiv ID:** 2602.05508 | [PDF](https://arxiv.org/pdf/2602.05508v1)

**作者:** Zhuang Xiong `[一作]` (Huazhong University of Science and Technology), Wenbing Tao `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4707 | [OpenAlex ID](https://openalex.org/A5087239641)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `51c0528b-f690-4182-ae60-bb5f046c276c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无标定的单目SLAM框架 VGGT-Motion，通过运动感知子图分段、anchor 驱动的直接 Sim(3) 配准以及轻量级子图图优化，实现千米级轨迹的一致性与高效推理。

**💡 创新点**

创新点：
1) 基于光流的运动感知子图构造，抑制零运动漂移并将转弯段完整封装，保持几何一致性；
2) 通过共享 anchor 的像素级对应实现直接 Sim(3) 配准，O(N) 线性复杂度，避免特征匹配和自注意力的 O(N²) 成本；
3) 子图级 Pose Graph 优化，线性复杂度实现全局一致性。

**🔧 技术方法**

技术手段：VGGT 3D 基础模型、光流估计、平移/旋转/尺度（Sim(3)）估计、Huber 鲁棒损失、Lie 群下 Levenberg–Marquardt 图优化、帧级重采样与冗余抑制、anchor 驱动的像素级对应。

**📊 数据集**

使用的数据集：KITTI、Waymo Open、4Seasons、Complex Urban、A2D2 等多种公开长序列数据。

**📈 对比分析**

与 ORB‑SLAM2、DPVO、DROID‑SLAM、MASt3R‑SLAM、CUT3R、Fast3R、VGGT‑SLAM、VGGT‑Long 等基准对比。结果表明，VGGT‑Motion 在 ATE、漂移率（<1%）和运行时（相较 VGGT‑Long 提升 18–36×）方面均显著优于现有 SOTA，且在零射击泛化任务中保持高精度。

**⚠️ 局限性**

局限性：
- 仍依赖 VGGT 的推断，受预训练数据和模型规模限制；
- 在极低纹理或极快运动场景下可能仍产生漂移；
- 需要手动设置阈值（如光流阈值、重叠大小等）；
- 未进行全局 Bundle Adjustment，精度仍受子图级优化的限制。

---

## 302. XEmoGPT: An Explainable Multimodal Emotion Recognition Framework with Cue-Level Perception and Reasoning

**arXiv ID:** 2602.05496 | [PDF](https://arxiv.org/pdf/2602.05496v1)

**作者:** Hanwen Zhang `[一作]` (University of Electronic Science and Technology of China), Qiao Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 4701 | [OpenAlex ID](https://openalex.org/A5100393703)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出XEmoGPT，一个能够感知并推理多模态情绪线索的可解释多模态情绪识别框架。

**💡 创新点**

创新点在于引入Video Emotional Cue Bridge (VECB) 与 Audio Emotional Cue Bridge (AECB) 以及构建 EmoCue 数据集和 EmoCue-360 评价指标，实现细粒度情绪线索感知与推理。

**🔧 技术方法**

结合CLIP‑ViT、HuBERT、Transformer、对比学习、Masking、LoRA调优，以及大语言模型 Qwen3-4B 生成解释文本。

**📊 数据集**

使用 MAFW、DFEW、MER2025、SpeechCraft、EmoCue‑Instruct、EmoCue‑ShortCaption 等多模态数据集训练，并在 EMER 与 EmoCue‑Eval 基准上评测。

**📈 对比分析**

与 SECap、Emotion‑LLaMA、AffectGPT 等对比，XEmoGPT 在 EmoCue‑360、BLEU、CIDEr 等指标上均优于基线，特别在视觉与全局情绪线索的 F1 分数最高。

**⚠️ 局限性**

局限在于仍依赖大量人工/机器生成的标注，情绪线索提取与匹配可能受文本风格影响，且对极其细腻或跨模态混合情绪的解释仍不够精确。

---

## 303. Transport and Merge: Cross-Architecture Merging for Large Language Models

**arXiv ID:** 2602.05495 | [PDF](https://arxiv.org/pdf/2602.05495v1)

**作者:** Chenhang Cui `[一作]`, Tat-Seng Chua `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于最优传输的跨架构模型融合框架，利用少量校准数据在激活空间对齐异构模型并直接在权重空间进行部分融合，实现大模型向小模型的知识迁移。

**💡 创新点**

创新点在于：①在激活空间使用最优传输得到跨层、跨神经元的软对应关系；②将这些对应关系映射到权重空间进行有选择的残差融合；③引入残差冻结微调，进一步提升性能。

**🔧 技术方法**

主要技术包括最优传输（Sinkhorn迭代）、Transformer内部神经元/特征映射、激活相关性度量、残差冻结适配、少量校准集激活提取。

**📊 数据集**

实验数据集涵盖低资源语言（马来语、印尼语、泰语、广东话）MMLU/CMMLU/MGSM/ARC等基准，以及金融与医学专家域的MMLU子集，使用的校准集仅为数百条样本。

**📈 对比分析**

与基准模型、SFT、蒸馏以及不同规模源模型的融合对比，结果显示在低资源语言和专业领域均显著提升，融合+适配往往优于蒸馏或SFT，且对一般能力影响小。

**⚠️ 局限性**

局限性包括：需要激活提取的校准集，若源/目标架构或tokenization差异过大可能失效；仅适用于Transformer线性子层；未充分评估偏见或错误传播风险。

---

## 304. Monte Carlo Rendering to Diffusion Curves with Differential BEM

**arXiv ID:** 2602.05492 | [PDF](https://arxiv.org/pdf/2602.05492v1)

**作者:** Ryusuke Sugimoto `[一作]` (University of Waterloo), Michal Lukáč `[通讯]` (Adobe Research)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

直接从噪声Monte Carlo渲染样本中，无需光栅化，利用差分边界元方法求解拉普拉斯方程，得到扩散曲线向量图像。

**💡 创新点**

提出差分BEM框架，支持双侧边界且不需构造线性系统；并在噪声数据下采用Levenberg–Marquardt迭代，实现在不依赖光栅化的向量化。

**🔧 技术方法**

差分边界元方法（differential BEM）、拉普拉斯方程、双侧边界跳跃、蒙特卡罗采样、Levenberg–Marquardt优化、GPU加速。

**📊 数据集**

在Adobe提供的sculpture、knots等场景以及自制的高细节场景上进行实验，使用512^2分辨率的基准图像。

**📈 对比分析**

与传统基于光栅化向量化方法（如FEM、边界提取）以及随机梯度下降对比，RMSE下降10-20%，收敛速度快，耗时仅几分钟。

**⚠️ 局限性**

难以捕捉高频细节、对稀疏手柄数有限制、需手工剪裁域边界、对高噪声仍有限制。

---

## 305. Sovereign-by-Design A Reference Architecture for AI and Blockchain Enabled Systems

**arXiv ID:** 2602.05486 | [PDF](https://arxiv.org/pdf/2602.05486v1)

**作者:** Matteo Esposito `[一作]` (University of Oulu), Valentina Lenarduzzi `[通讯]` (University of Southern Denmark)

**通讯引用:** 3069 | [OpenAlex ID](https://openalex.org/A5015576503)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出了一种 Sovereign Reference Architecture（SRA），将自我主权身份（SSI）、区块链信任与审计、主权数据治理以及生成式 AI 在同一体系结构中协同实现数字主权。

**💡 创新点**

创新点在于将数字主权从法规层面提升为架构质量属性，明确了三大核心层面（SSI、区块链、主权 AI）并通过跨层约束实现可审计、可演化且符合司法边界的系统设计。

**🔧 技术方法**

主要技术包括分布式身份（DID/VC）、区块链（如权限链或跨链锚定）、政策即代码的数据治理、主权 AI 生命周期管理（模型注册、评估门控、可追溯日志）以及服务网格/API 网关等。

**📊 数据集**

由于是定位性论文，没有使用具体数据集，而是基于欧盟云主权框架（EU Cloud Sovereignty Framework）以及相关监管文本进行设计推导。

**📈 对比分析**

论文未进行实验性比较或性能评估，而是通过案例分析和架构图说明其可行性；未来工作建议在真实云/边缘环境中验证可审计性、可迁移性与成本收益。

**⚠️ 局限性**

局限性包括：实现复杂度高、成本和可扩展性挑战、与现有云服务兼容性问题、缺乏实证验证及对 AI 黑盒可解释性的进一步研究需求。

---

## 306. Tight FPT Approximations for Fair $k$-center with Outliers

**arXiv ID:** 2602.05476 | [PDF](https://arxiv.org/pdf/2602.05476v1)

**作者:** Ameet Gadekar `[一作]` `[通讯]` (CISPA Helmholtz Center for Information Security), Ameet Gadekar (CISPA Helmholtz Center for Information Security)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种确定性 3‑近似算法，解决了公平 k‑中心加离群点问题，算法在以 k 为参数的 FPT 时间内运行。

**💡 创新点**

创新点在于摆脱了传统的两阶段“投影”技术，直接构造公平解；引入迭代球查找框架和结构三分法，并证明 3 倍逼近因子是最优。

**🔧 技术方法**

核心技术包括颜色编码、迭代球查找、结构三分法、FPT 复杂度分析以及从一般公平约束到彩色实例的逼近保持归约。

**📊 数据集**

论文仅为理论研究，没有使用具体实验数据集。

**📈 对比分析**

与之前的随机双重标准 FPT 逼近相比，本文实现了确定性 3 倍逼近，并在时间复杂度 2^O(k log k) poly(n) 内完成，理论上是最优的。

**⚠️ 局限性**

主要局限在于算法指数依赖于 k，且仅针对 k‑中心目标；对其他聚类目标或更复杂的公平约束仍未给出多项式时间的近似。

---

## 307. Structured Context Engineering for File-Native Agentic Systems: Evaluating Schema Accuracy, Format Effectiveness, and Multi-File Navigation at Scale

**arXiv ID:** 2602.05447 | [PDF](https://arxiv.org/pdf/2602.05447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 308. Reasoning under Ambiguity: Uncertainty-Aware Multilingual Emotion Classification under Partial Supervision

**arXiv ID:** 2602.05471 | [PDF](https://arxiv.org/pdf/2602.05471v1)

**作者:** Md. Mithun Hossaina `[一作]`, Md Shafiqul Islam `[通讯]` (Bangladesh University of Business and Technology)

**通讯引用:** 406 | [OpenAlex ID](https://openalex.org/A5059572045)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种不确定性感知的多语言多标签情绪分类框架，解决部分监督下情绪歧义问题。

**💡 创新点**

创新点是引入基于熵的实例权重和正负标签无监督正则化，将注释不确定性显式纳入学习目标。

**🔧 技术方法**

采用共享多语言编码器（如XLM‑R）+线性情绪头、熵权重、PU正则化和贝塔分布推理等技术。

**📊 数据集**

在SemEval‑2018 Task 1 的英语、西班牙语、阿拉伯语情绪数据集上进行实验。

**📈 对比分析**

与多种基线相比，在Hamming Loss、Ranking Loss、micro‑F1、macro‑F1和AP等指标上均实现显著提升，鲁棒性和稳定性更好。

**⚠️ 局限性**

局限在于熵权重依赖注释质量，未考虑标注者差异；仅在短文本验证，长文本或多模态任务的泛化需进一步研究。

---

## 309. When Are RL Hyperparameters Benign? A Study in Offline Goal-Conditioned RL

**arXiv ID:** 2602.05459 | [PDF](https://arxiv.org/pdf/2602.05459v1)

**作者:** Jan Malte Töpperwien `[一作]` (Leibniz Universität Hannover), Marius Lindauer `[通讯]` (Leibniz Universität Hannover)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在离线目标导向强化学习环境下研究超参数敏感性，系统比较了基于Bootstrapping的HIQL与非Bootstrapping的QRL两种算法的超参数景观、转移性与梯度干扰。

**💡 创新点**

创新点在于将探索噪声剔除后，揭示超参数不稳定主要由目标设计（Bootstrapping vs. 代表学习）及数据质量变化驱动，并通过跨目标梯度相似度诊断解释了敏感性机制。

**🔧 技术方法**

采用超参数景观分析（ε-最优区域、相位漂移、早期选择损失）和fANOVA重要性评估，并使用梯度相似度（cosine相似度）作为干扰指标；实现了HIQL与QRL的离线目标重标记与目标学习。

**📊 数据集**

实验使用公开的离线目标导向数据集（包含专家演示与随机探索轨迹的混合），在不同数据质量比例（0%、40%、80%、100%）及按阶段递增的专家比例（100%→80%→40%→0%）下进行评估。

**📈 对比分析**

与在线RL相比，离线设置下超参数景观更平缓；QRL在加入约20%专家数据后保持宽广的近最优区域，性能稳定；HIQL在同一数据质量下表现出尖锐的最优峰和相位间显著漂移，早期选择损失较高。

**⚠️ 局限性**

局限在于仅对两种代表性算法进行评估，未探讨更广泛目标设计；梯度干扰诊断仅关注价值网络，可能未涵盖策略更新的交互；实验仅在离线目标导向RL框架内，结果对在线或其他任务的推广仍需验证。

---

## 310. DistillER: Knowledge Distillation in Entity Resolution with Large Language Models

**arXiv ID:** 2602.05452 | [PDF](https://arxiv.org/pdf/2602.05452v1)

**作者:** Alexandros Zeakis `[一作]` (National Kapodistrian University of Athens), Manolis Koubarakis `[通讯]` (National Kapodistrian University of Athens)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DistillER 框架，通过知识蒸馏将大型语言模型的实体解析能力迁移至小模型，实现无标签且高效的实体解析；

**💡 创新点**

创新点在于：①构建三维蒸馏框架（数据选择、知识获取、蒸馏算法）；②首次系统地研究无标签知识蒸馏在实体解析中的可行性；③对比监督微调与强化学习两大蒸馏策略，并支持生成解释；

**🔧 技术方法**

使用大型 LLM（Llama‑3.1、Qwen‑2.5）和小型 SLM（S‑MiniLM、RoBERTa），采用排序/聚类数据选择、LLM/SLM/多模型教师注释、监督微调（SFT）及强化学习（GRPO、DPO），并评估 F1 分数；

**📊 数据集**

八个真实实体解析数据集（D1‑D8），涵盖产品、电影等多域；

**📈 对比分析**

与多种无监督与有监督基线（ZeroER、CollaborEM、HierGAT、Unicorn 等）对比；在无标签蒸馏标签下，SFT 取得平均 F1≈0.85，显著优于现有方法；RL 提升有限；Ranking 方案在数据选择上优于随机/聚类；

**⚠️ 局限性**

局限性包括：仍需依赖 LLM 生成的噪声标签；RL 方法在实际提升上有限；解释生成对 DPO 效果不稳定；大型模型推理成本仍高；对极端样本不平衡和长文本的鲁棒性尚未充分验证。

---

## 311. GNSS SpAmming: a spoofing-based GNSS denial-of-service attack

**arXiv ID:** 2602.05517 | [PDF](https://arxiv.org/pdf/2602.05517v1)

**作者:** Sergio Angulo Cosín `[一作]` (National Institute for Aerospace Technology), José-Antonio Gómez-Sánchez `[通讯]` (National Institute for Aerospace Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

通过伪造少数卫星的信号并配合干扰，实施了一种新的GNSS拒绝服务攻击（SpAmming），使接收机失去对特定卫星的获取能力，从而间接导致定位失效。

**💡 创新点**

提出了将伪装与干扰结合的混合攻击策略，能够绕过传统的jamming和spoofing防御，尤其针对仅部分卫星提供的OSNMA身份验证服务。

**🔧 技术方法**

实验使用软件定义无线电（Ettus USRP B210）、gal-sdr-sim仿真工具、GNU Radio、以及u‑blox ZED‑F9P接收机，结合SDR发射和真实卫星信号进行对比测试。

**📊 数据集**

实验基于真实卫星信号，没有使用公开数据集；所有测试在室内实验室通过模拟接收与发射来完成。

**📈 对比分析**

将攻击前后的信号获取情况做对比，寒冷启动下成功率接近100%，温热启动和热启动的成功率下降但可通过辅助干扰提高；评估重点是信号获取失败率和定位误差。

**⚠️ 局限性**

局限性包括：对热启动的干扰效果有限，需要精确匹配Doppler和码偏移，实验仅为概念验证，缺乏大规模正式评估和对不同GNSS系统的普适性验证。

---

## 312. Virtual-Tube-Based Cooperative Transport Control for Multi-UAV Systems in Constrained Environments

**arXiv ID:** 2602.05516 | [PDF](https://arxiv.org/pdf/2602.05516v1)

**作者:** Runxiao Liu `[一作]` (Beihang University), Quan Quan `[通讯]` (Beihang University)

**通讯引用:** 6058 | [OpenAlex ID](https://openalex.org/A5058029021)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于虚拟管道与耗散系统理论的多无人机协同运输控制框架，用于在狭窄环境中安全高效地悬挂并搬运重载物体。

**💡 创新点**

创新点在于将耗散系统理论引入虚拟管道控制，既保证了系统的稳定性与鲁棒性，又实现了张力分布均衡与自适应配置，从而克服传统虚拟管道方法在窄通道中形成失衡和不稳定的问题。

**🔧 技术方法**

采用虚拟管道模型、耗散系统控制、虚拟中间系统（虚拟托盘、节点、弹簧）以及低层速度控制与力估计（ESO/ CFO）等技术实现分布式控制。

**📊 数据集**

实验数据来源于大规模仿真（10架无人机、18 kg 负载、50 m×20 m×20 m 障碍环境）以及户外三架四旋翼实际飞行实验（3.06 kg 负载）。

**📈 对比分析**

与传统虚拟管道控制方法比较，本文方法在仿真和实验中表现出更低的振荡、更均衡的张力分布、保持虚拟管道内的安全距离，且能够成功完成重负载搬运，证明其在复杂狭窄环境中的性能优于传统方案。

**⚠️ 局限性**

主要局限在于需预先规划好虚拟管道，无法实现在线自适应生成管道，且对极端动态障碍物的适应性尚未验证。

---

## 313. Relying on LLMs: Student Practices and Instructor Norms are Changing in Computer Science Education

**arXiv ID:** 2602.05506 | [PDF](https://arxiv.org/pdf/2602.05506v1)

**作者:** Xinrui Lin `[一作]` (Beijing Institute of Technology), John Vines `[通讯]` (University of Edinburgh)

**通讯引用:** 4966 | [OpenAlex ID](https://openalex.org/A5053470577)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对16名计算机科学学生和6名教师的用户研究，系统分析了学生在五个情境下使用大语言模型（LLM）的意图与实践，并对比教师规范，探讨了冲突与合作关系，提出了针对不同意图的LLM设计建议。

**💡 创新点**

提出了以“意图”为核心的细粒度LLM使用框架，识别出七种意图及其冲突等级，揭示教师从禁用到过程评估的转变，并给出了针对高冲突与低冲突场景的定制化设计原则。

**🔧 技术方法**

采用定性访谈与聊天日志分析方法，对学生聊天记录进行主题编码与归纳，并结合教师访谈完成框架构建与设计建议。

**📊 数据集**

收集了16名学生的50条LLM聊天日志（ChatGPT、Gemini、DeepSeek等）和6名教师的访谈文本，未使用公开大规模数据集。

**📈 对比分析**

本研究为探索性定性研究，未进行数值性能比较；主要通过对比学生实践与教师规范的冲突程度，提供设计建议，未给出实验指标。

**⚠️ 局限性**

样本局限于中国高校CS学生与教师，样本量小且未覆盖全部意图；研究为定性分析，缺乏量化验证与跨文化验证。

---

## 314. A Unified Framework for Rethinking Policy Divergence Measures in GRPO

**arXiv ID:** 2602.05494 | [PDF](https://arxiv.org/pdf/2602.05494v1)

**作者:** Qingyuan Wu `[一作]` (University of Southampton), Chao Huang `[通讯]` (University of Southampton)

**通讯引用:** 12986 | [OpenAlex ID](https://openalex.org/A5042083053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出统一裁剪框架并基于KL3估计实现ATR-GRPO，以提升大型语言模型在数学推理任务中的强化学习训练效果。

**💡 创新点**

将比例裁剪与KL裁剪统一成通用框架，发现KL3估计能够以低成本近似信任区间，并通过异向裁剪（ATR）实现更强探索；理论证明其优于传统对称裁剪。

**🔧 技术方法**

使用GRPO、KL3 Monte Carlo估计、裁剪函数、LoRA微调、Unsloth+TRL框架，以及基于奖励的RL训练。

**📊 数据集**

训练集：DAPO-Math-17k；评估集：AMC2023、AIME2024、AIME2025。

**📈 对比分析**

与Clip、Clip-Higher、Dual Clip、Dynamic Clipping、Clip-Cov、Soft Gate 等 SOTA 裁剪方法对比；在 Qwen3-1.7B/8B 上，ATR-GRPO 在 Mean@8/Pass@8 上均取得最高平均分，训练稳定性和收敛速度显著优于基线。

**⚠️ 局限性**

局限性：使用固定的信任区间阈值，未实现自适应调整；裁剪在 token 级别，导致与序列奖励不匹配，产生高方差；缺乏序列级整合和更深层次的可解释性分析。

---

## 315. Feature points evaluation on omnidirectional vision with a photorealistic fisheye sequence -- A report on experiments done in 2014

**arXiv ID:** 2602.05487 | [PDF](https://arxiv.org/pdf/2602.05487v1)

**作者:** Julien Moreau `[一作]` (Universite de technologie de Compiegne), Yassine Ruichek `[通讯]` (Universite de technologie de Belfort-Montbeliard)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

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

## 316. Wasure: A Modular Toolkit for Comprehensive WebAssembly Benchmarking

**arXiv ID:** 2602.05488 | [PDF](https://arxiv.org/pdf/2602.05488v1)

**作者:** Riccardo Carissimi `[一作]` (Università degli Studi di Milano), Ben L. Titzer `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1377 | [OpenAlex ID](https://openalex.org/A5034437441)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Wasure，一个可扩展的命令行工具，用于 WebAssembly 引擎的基准测试、特性检测与动态分析。

**💡 创新点**

将基准执行、结果可视化、特性兼容性检查和无侵入式动态分析统一到一个模块化框架，并通过 Wizard 引擎实现细粒度指令级监测。

**🔧 技术方法**

使用 Python 编写 CLI，采用子命令模式；集成多种 Wasm 引擎（V8、SpiderMonkey、wasmtime、wasmer、WAMR、wasmedge 等）和工具（Wizard、Wasabi），动态指标采集通过 Wizard 的探针完成。

**📊 数据集**

数据集包括 Wasure 自带的多套 benchmark（mibench、ostrish、polybench、wasm-r3 等）以及 Wizard 产生的动态分析结果。

**📈 对比分析**

在多引擎、多配置下多次重复执行基准，记录执行时间、RSS/VMS、指令覆盖等，输出 JSON/CSV 并绘制性能分布图、覆盖率柱状图和 PCA 可视化；结果显示不同引擎在平均速度、变异性和资源占用上存在显著差异。

**⚠️ 局限性**

局限在于仅支持 Unix‑like 系统（Linux/macOS），Windows 支持不足；动态分析仅能在 Wizard 上完成，无法覆盖所有 benchmark；Benchmarks 仍缺乏更广泛的工业级工作负载。

---

## 317. LMMRec: LLM-driven Motivation-aware Multimodal Recommendation

**arXiv ID:** 2602.05474 | [PDF](https://arxiv.org/pdf/2602.05474v1)

**作者:** Yicheng Di `[一作]` (Jiangnan University), Yuan Liu `[通讯]` (Jiangnan University)

**通讯引用:** 17323 | [OpenAlex ID](https://openalex.org/A5100390926)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种基于大语言模型的动机感知多模态推荐框架 LMMRec，利用文本与交互数据提取细粒度动机并进行跨模态对齐。

**💡 创新点**

创新点在于将链式思维提示与大语言模型结合生成文本动机，设计双塔结构与动机协调与交互‑文本对应两大机制，显著降低多模态噪声并提升解释性。

**🔧 技术方法**

核心技术包括链式思维提示、双编码器（LightGCN+线性映射）、对比学习（Motivation Coordination Strategy）、基于动量的教师‑学生交互文本对应方法以及多任务联合训练。

**📊 数据集**

实验使用公开的 Yelp、Amazon‑book 与 Steam 三个数据集，分别过滤不同评分阈值后划分训练/验证/测试。

**📈 对比分析**

与 UIST、ONCE、AutoGraph 等通用改进基线以及 WeightedGCL、PolyCF 基本模型相比，LMMRec 在 Recall@N 与 NDCG@N 上平均提升 3–5%（Steam 上最高 4.98%），并在噪声鲁棒性上表现更优。

**⚠️ 局限性**

主要局限在于依赖大语言模型产生的文本动机，受模型推理成本与数据偏差影响；未针对开放域或极端交互场景验证；以及对不同模态的融合策略仍有改进空间。

---

## 318. TaSA: Two-Phased Deep Predictive Learning of Tactile Sensory Attenuation for Improving In-Grasp Manipulation

**arXiv ID:** 2602.05468 | [PDF](https://arxiv.org/pdf/2602.05468v1)

**作者:** Pranav Ponnivalavan `[一作]` (Waseda University), Shigeki Sugano `[通讯]` (Waseda University)

**通讯引用:** 7010 | [OpenAlex ID](https://openalex.org/A5080654277)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种两阶段深度预测框架（TaSA），先学习机器人自身触碰（self‑touch）的动力学，再将预测的自触信息与原始触觉输入一起用于控制多指手进行高精度插入任务。

**💡 创新点**

创新点在于首次将感官衰减（sensory attenuation）机制引入机器人触觉学习，通过显式预测自触并将其从外部触觉信号中滤除，显著提升了对外部物体接触的辨别能力。

**🔧 技术方法**

技术包括全连接网络（FCN）用于自触预测，长短期记忆网络（LSTM）用于运动学习，配合预测模型的前向传播实现自触衰减；数据预处理、损失函数设计以及双阶段训练策略。

**📊 数据集**

数据集来自在 Allegro Hand 上使用 uSkin 触觉皮肤（仅 index 与 thumb 指尖）通过人为遥控采集的运动与触觉数据，涵盖纸夹定位、硬币插槽和铅笔芯插入三类高精度插入任务。

**📈 对比分析**

与仅使用原始触觉输入（baseline）对比，TaSA 在纸夹、硬币和铅笔芯插入任务中的成功率分别从约 70%→95%、68%→92% 和 26%→58% 大幅提升，且在未见过的测试配置上也能实现完美或大幅提高的泛化性能。

**⚠️ 局限性**

局限在于仅验证了两指手部设置，未覆盖更多指尖或手掌自触；对极细小或弱触觉信号的区分仍有限，且需要更大规模、真实环境下的进一步验证。

---

## 319. Attention Retention for Continual Learning with Vision Transformers

**arXiv ID:** 2602.05454 | [PDF](https://arxiv.org/pdf/2602.05454v1)

**作者:** Yue Lu `[一作]` (Northwestern Polytechnical University), Wencong Zhang `[通讯]` (Hikrobot Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于注意力保持的连续学习框架ARCL‑ViT，通过梯度屏蔽避免Vision Transformer在学习新任务时出现注意力漂移，从而减轻灾难性遗忘。

**💡 创新点**

创新点在于：①使用层级Rollout结合自适应阈值动态生成二值注意力掩码；②在反向传播中对梯度进行掩码并按比例放缩，使方法兼容任何优化器；③完全不需要保存旧任务样本。

**🔧 技术方法**

技术要点包括：Vision Transformer (ViT)、层级Attention Rollout、二值注意力掩码、梯度屏蔽与梯度放缩、Adam优化器、预训练ViT权重。

**📊 数据集**

实验数据集：10/20-split ImageNet‑R、10-split CIFAR‑100、10-split DomainNet，以及在ImageNet‑R和DomainNet上进行的50/100-split长序列任务。

**📈 对比分析**

与Seq‑FT以及15+近期方法（如VPT‑CPG、SD‑LoRA、BiLoRA等）进行对比，在四大基准上平均准确率提升约1.8%，忘记率显著降低，长序列实验中比VPT‑CPG提升约3.3%。

**⚠️ 局限性**

局限性：目前仅在视觉Transformer上验证；需要额外计算和存储注意力掩码；对任务间高度重叠或非视觉域的适用性尚未系统评估。

---

## 320. DiLLS: Interactive Diagnosis of LLM-based Multi-agent Systems via Layered Summary of Agent Behaviors

**arXiv ID:** 2602.05446 | [PDF](https://arxiv.org/pdf/2602.05446v1)

**作者:** Rui Sheng `[一作]` (Hong Kong University of Science and Technology), Furui Cheng `[通讯]` (ETH Zürich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于 Activity Theory 的三层（活动、动作、操作）可视化系统，用于 LLM 多智能体系统的故障诊断与理解。

**💡 创新点**

创新点包括：①使用 Activity Theory 的三层结构组织 Agent 行为；②通过自然语言探测（prompting）自动提取计划、动作、操作信息；③提供层级摘要视图和交互过滤功能；④显著提升开发者诊断效率与信心。

**🔧 技术方法**

技术实现涵盖：LLM 自然语言探测、表格/列表/日志可视化前端、NASA‑TLX 认知负荷评估、用户研究设计；使用 Magnetic‑One 中央化多智能体框架和 GAIA 基准进行验证。

**📊 数据集**

使用 GAIA 基准的四个含错误案例作为实验数据集。

**📈 对比分析**

与仅显示时间线日志的基线系统进行对比；结果显示在系统条件下失败识别平均提高 1.42 个（p<0.05），用户信心提升 1.08 分，NASA‑TLX 负荷显著降低，表明性能显著优于基线。

**⚠️ 局限性**

局限性：仅评估诊断阶段未测试修复效果；LLM 探测可能存在不准；实验仅覆盖中央化多智能体架构，无法直接推广到去中心化或共享消息池系统；样本量有限。

---

## 321. Causal Front-Door Adjustment for Robust Jailbreak Attacks on LLMs

**arXiv ID:** 2602.05444 | [PDF](https://arxiv.org/pdf/2602.05444v1)

**作者:** Yao Zhou `[一作]` (Institute of Software Chinese Academy of Sciences), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 43772 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、因果驱动的前门调整攻击（CFA^2），通过剥离LLM内部安全机制实现越狱。

**💡 创新点**

创新点在于将安全对齐视为未观测混杂变量，使用因果前门调整理论切断安全干预，并借助稀疏自编码器实现可解释的特征分离。

**🔧 技术方法**

采用稀疏自编码器（SAE）提取核心任务语义、权重正交化消除安全子空间，以及前门调整公式实现低复杂度生成。

**📊 数据集**

实验使用AdvBench和HarmBench两个常用越狱评测数据集。

**📈 对比分析**

与8种基线（包括GCG、PAIR、TAP等）对比，平均攻击成功率（ASR）达83.68%，相较最强基线提升约35%；推理时间比传统梯度攻击提升≈850×，文本流畅度（PPL）显著更好。

**⚠️ 局限性**

方法需白盒访问和模型参数修改，无法直接迁移到闭源模型；对内部稀疏特征的依赖也限制了黑盒场景的适用性。

---

## 322. A Human-in-the-Loop, LLM-Centered Architecture for Knowledge-Graph Question Answering

**arXiv ID:** 2602.05512 | [PDF](https://arxiv.org/pdf/2602.05512v1)

**作者:** Larissa Pusch `[一作]` (Zuse Institute Berlin), Tim Conrad `[通讯]` (Zuse Institute Berlin)

**通讯引用:** 1363 | [OpenAlex ID](https://openalex.org/A5012229905)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种交互式框架，使大语言模型（LLM）能够将自然语言问题翻译为可执行的图查询，提供可解释的查询说明，并通过用户反馈迭代改进查询。

**💡 创新点**

创新点在于将透明的LLM生成查询与自然语言迭代修订相结合，实现了可审计、可解释的知识图谱查询，同时提供了多模型、多领域的实验评估。

**🔧 技术方法**

技术包括LLM Prompt设计（查询生成、解释、修订）、Cypher查询执行、结果可视化、误差检测与修正循环，以及对不同LLM模型的接口封装。

**📊 数据集**

使用了三类数据集：合成电影知识图（90条查询）、数学研究数据图（MaRDI子图）和生物学研究图（Ngorongoro羚羊豺狼项目）等真实领域图谱。

**📈 对比分析**

通过与多款LLM（GPT、Claude、DeepSeek、Gemma、Llama等）在相同查询集上的交互实验，对比了首次成功率、累计成功率、修订次数和错误检测率，结果显示o3-mini和GPT‑5.2在多数任务上达到100%成功率，而某些模型在生物领域实验中表现明显下降，凸显了模型跨域适应性差异。

**⚠️ 局限性**

主要局限包括：解释过程中易遗漏时间信息、不同模型对错误检测的敏感度不一、实验规模受限于小样本且未涵盖所有复杂查询类型，未来需要更大规模、多样化评测与更完善的交互工具。

---

## 323. Repairing Property Graphs under PG-Constraints

**arXiv ID:** 2602.05503 | [PDF](https://arxiv.org/pdf/2602.05503v1)

**作者:** Christopher Spinrath `[一作]` (Lyon 1 University), Rachid Echahed `[通讯]` (CNRS LIG)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套针对PG‑Constraints（即属性图约束）的错误修复管线，能够在删除节点、边或标签的前提下修复满足约束的属性图；

**💡 创新点**

创新点在于引入RGPC约束子集（支持递归、标签表达式等），通过构造RGPC自动机推导“重要标记”，并将错误映射为冲突超图，利用最小加权顶点覆盖求解最优修复，同时提供两种贪心算法与可选的标签删除和邻域错误扩展以实现可调节的时间-质量权衡；

**🔧 技术方法**

技术包括：RGPC约束和自动机模型、错误检索的GQL查询、构造冲突超图、整数线性规划（ILP）求解最小加权顶点覆盖、LP‑guided贪心算法、标签删除的标记扩展、邻域错误压缩；

**📊 数据集**

使用四个真实或近似真实的属性图：ICIJ（国际调查记者联盟）图、意大利立法图、Coreutils代码图以及LDBC基准图（不同规模）来评估；

**📈 对比分析**

与ILP（最优）及两种贪心算法比较：LP‑guided贪心在保持与ILP相同删除量的前提下，平均速度提升≈97%；在某些大规模约束下ILP仍优；可选标签删除可将删除量减少最多59%，但运行时间可提升5倍；邻域错误压缩可显著降低时间而保持删除量；

**⚠️ 局限性**

限制在于：需枚举所有错误导致可扩展性受限；标签删除仅在约束中出现标签时可行；使用删法可能引入新错误；当前未支持属性值修改或插入操作；以及对递归路径的复杂度分析尚不完整。

---

## 324. SDFP: Speculative Decoding with FIT-Pruned Models for Training-Free and Plug-and-Play LLM Acceleration

**arXiv ID:** 2602.05499 | [PDF](https://arxiv.org/pdf/2602.05499v1)

**作者:** Hanyu Wei `[一作]` (Tsinghua University), Yuhan Dong `[通讯]` (Tsinghua University)

**通讯引用:** 5513 | [OpenAlex ID](https://openalex.org/A5108047157)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于 Fisher Information Trace 的层级剪枝式草稿模型，用于无训练、即插即用的自回归推理加速。

**💡 创新点**

将 Fisher Information Trace 作为无监督、数据集无关的敏感性度量，引入一站式剪枝与投机解码相结合的全新框架，消除训练与调参负担。

**🔧 技术方法**

使用 Fisher Information Trace、经验 Fisher 跟踪、层级剪枝、投机解码的接受-拒绝机制以及 KV 缓存技术。

**📊 数据集**

FIT 计算基于 WikiText2；性能评估在 CNN/Daily Mail、GSM8K、TinyStories 三大任务上进行，使用 LLaMA‑2 系列模型。

**📈 对比分析**

与 Vanilla、Parallel、Lookahead、SWIFT 等方法对比；在三大基准上实现 1.32×–1.5× 的推理速度提升，且无额外训练或离线调优开销。

**⚠️ 局限性**

对剪枝比例的手工设定敏感；极端剪枝可能导致精度下降；使用通用数据集计算 FIT 可能无法捕捉任务特定的细微差异。

---

## 325. Toward Operationalizing Rasmussen: Drift Observability on the Simplex for Evolving Systems

**arXiv ID:** 2602.05483 | [PDF](https://arxiv.org/pdf/2602.05483v1)

**作者:** Anatoly A. Krasnovsky `[一作]` `[通讯]` (Innopolis University), Anatoly A. Krasnovsky (Innopolis University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Aitchison几何的漂移可观测框架，利用工件驱动的模型动态刷新部件清单和安全边界，并通过部件谱系聚合在架构变更下保持可比较性，进一步给出早期预警诊断和可解释的平衡坐标。

**💡 创新点**

创新点在于将组合数据分析（CoDA）与软件运维工件（如SLO、Kubernetes清单、OpenSLO）相结合，使用Aitchison几何获取坐标不变的漂移方向和安全距离，同时通过部件谱系实现对系统演化的自适应稳定化，并提供基于安全边界的可解释诊断。

**🔧 技术方法**

使用技术包括：Aitchison/CoDA、对数比率变换（ilr）、线性状态模型、部件谱系映射、工件提取与自动化模型推断、根本安全模型（Rasmussen）以及安全边界的对数柱面惰性（log‑barrier）诊断。

**📊 数据集**

本文为愿景性研究，未使用公开真实数据集，而是以SLO‑as‑code、微服务部署清单、分布式追踪和合成漂移序列为假设输入，示例中使用了SLO边界、错误预算、服务图等工件构造的合成数据。

**📈 对比分析**

比较方法：与传统SRE报警规则（如错误预算耗尽/燃烧率）、单变量变化检测、欧氏距离监测以及对数比率单独监测等基线对比。预期在模拟漂移和零漂移情境下，Aitchison几何诊断能更早、更准确地捕捉边界逼近，但当前尚无实验结果，性能评估需后续验证。

**⚠️ 局限性**

局限性包括：需要人工或规则驱动的平衡分区选择；零值和缺失值的处理需细致；模型提取的准确性和置信度决定报警门槛；对实时监控的计算开销待评估；仅适用于组合型信号，难以直接扩展到分布型遥测；在真实生产环境中的可行性与部署复杂性尚未验证。

---

## 326. SOMA-1M: A Large-Scale SAR-Optical Multi-resolution Alignment Dataset for Multi-Task Remote Sensing

**arXiv ID:** 2602.05480 | [PDF](https://arxiv.org/pdf/2602.05480v1)

**作者:** Peihao Wu `[一作]` (School of Remote Sensing Information Engineering, Wuhan University), Yongjun Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SOMA-1M数据集，并在多任务遥感中进行评测

**💡 创新点**

提供百万级、三分辨率、像素级对齐的SAR-光学数据集，并设计两阶段精细配准框架

**🔧 技术方法**

采用MapGlue匹配、RANSAC、深度学习匹配/融合/云去除/翻译模型

**📊 数据集**

SOMA-1M（1.3M对齐样本），SOMA-0.1M子集与OSdataset、SRIF等公开数据集对比

**📈 对比分析**

SOMA-0.1M训练后模型在匹配、融合、云去除、翻译四个基准任务上均优于基线，匹配SOMA_MapGlue实现SOTA

**⚠️ 局限性**

高分辨率SAR仍带来显著几何噪声与光谱差异，模型对极端分辨率适应性有限，需进一步改进多尺度自监督学习

---

## 327. MerNav: A Highly Generalizable Memory-Execute-Review Framework for Zero-Shot Object Goal Navigation

**arXiv ID:** 2602.05467 | [PDF](https://arxiv.org/pdf/2602.05467v1)

**作者:** Dekang Qi `[一作]` (Amap, Alibaba Group), Mu Xu `[通讯]` (Amap, Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于记忆–执行–复核的视觉语言导航框架MerNav，能够在不需要训练的情况下实现高效导航。

**💡 创新点**

创新点在于构建分层记忆体系（短期、长期与常识记忆）以及复核模块，允许系统在常规与异常场景中自适应决策并纠错。

**🔧 技术方法**

使用大型多模态基础模型（如Qwen3‑vl‑plus或GPT‑5.2）结合观察分析、路径规划、动作选择和复核机制的端到端推理管道。

**📊 数据集**

在四个公开数据集上进行评测：MP3D、HM3D_v0.1、HM3D_v0.2 和 HM3D_OVON。

**📈 对比分析**

与所有训练免费（TF）和零样本（ZS）基线相比，MerNav平均提升成功率约7%（TF）和5%（ZS），在HM3D_v0.1与HM3D_OVON上分别提升8%和6%，并在MP3D和HM3D_OVON上同时超越所有SFT方法。

**⚠️ 局限性**

主要局限在于仍依赖大规模预训练模型的算力与成本，缺乏真实机器人实验验证，以及复核模块的阈值和规则需要手工调参。

---

## 328. Optimization is Not Enough: Why Problem Formulation Deserves Equal Attention

**arXiv ID:** 2602.05466 | [PDF](https://arxiv.org/pdf/2602.05466v1)

**作者:** Iván Olarte Rodríguez `[一作]` (Leiden University), Elena Raponi `[通讯]` (Leiden University)

**通讯引用:** 408 | [OpenAlex ID](https://openalex.org/A5089662417)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了基于层压参数插值的多变量纤维方向与拓扑联合优化框架，针对受限体积的悬臂梁结构进行黑盒优化。

**💡 创新点**

创新点在于将拓扑设计变量与层压参数分离并采用分阶段（先拓扑后纤维方向）的顺序优化，显著提高了物理可行性和性能。

**🔧 技术方法**

使用的技术包括Moving Morphable Components (MMC) 以及 Lamination Parameter Interpolation Method (LPIM)，并应用 CMA-ES、BAxUS、HEBO、TuRBO 等黑盒优化算法。

**📊 数据集**

数据集为单一的二维悬臂梁仿真案例，采用有限元模拟求解合规性，并在1000次评估预算下评估。

**📈 对比分析**

比较方法为对比并发与顺序两种策略，结果显示顺序策略在绝大多数算法中取得更低的合规性和更小的方差，尤其在 TuRBO 与 HEBO 上表现突出。

**⚠️ 局限性**

局限性在于仅测试了单一基准问题，且仅使用了三条 MMC 和少量主节点，缺乏对更复杂结构或更高维度设计空间的验证。

---

## 329. Explicit List-Decodable Linearized Reed-Solomon Subspace Codes via Subspace Designs

**arXiv ID:** 2602.05462 | [PDF](https://arxiv.org/pdf/2602.05462v1)

**作者:** Kuo Shang `[一作]`, Ruiqi Zhu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了基于线性化Reed–Solomon（LRS）与折叠线性化Reed–Solomon（FLRS）码的显式子码，能够在求和-秩度量下实现列表解码，且解码半径可突破唯一解码极限，列表大小有可控上界。

**💡 创新点**

主要创新点是：① 将子空间设计与子空间避免集合结合到求和-秩度量场景；② 采用线性代数法构造求和-秩度量下的代数约束，得到周期子空间；③ 在这些周期子空间与子空间设计/避免集合交集时，可保证列表大小多项式/可控；④ 通过折叠操作进一步提升解码半径，首次给出显式正码率 FLRS 子码实现列表解码超过唯一解码半径。

**🔧 技术方法**

使用的技术包括：
- 带有 Frobenius 自同构的斜多项式（skew polynomial）评估与求和-秩度量码的构造；
- 线性代数型插值与解码（Welch–Berlekamp 方式的多变量斜多项式插值）；
- 周期子空间与子空间设计（(s,A,n)-subspace design）相结合以控制列表大小；
- 子空间避免集合（subspace‑evasive set）用于 FLRS 子码列表限制；
- 组合编码与解码的多阶段算法（插值、候选集求解、交集求解）。

**📊 数据集**

本文为理论研究，不使用实验数据集；所有结果均为数学证明与构造。

**📈 对比分析**

相较于之前仅提供随机或非显式代码的列表可解性结论，本文给出了显式构造的正码率子码，且在相同参数下实现的列表解码半径可逼近或达到列表解码容量上界，解码复杂度为多项式级别。实验与数值评估在文中未给出，但理论上证明了列表大小上界（如 h^{O(s²/ε²)} 或 (d/ε)^d）与解码半径（如 ρ = 1 – R – ε）满足预期。

**⚠️ 局限性**

局限性包括：
- 需要大域（如 _h^t 级别），导致实现难度较高；
- 列表大小虽被理论上限制，但在实际参数下仍可能很大；
- 解码半径依赖于参数 s、λ 的取值，若想逼近容量需取 s ≈ 1/ε，导致列表上界指数随 1/ε² 增长；
- 代码的实际构造与编码/解码实现仍需在大域上进行多项式运算，实际效率未在实验中验证。

---

## 330. BLITZRANK: Principled Zero-shot Ranking Agents with Tournament Graphs

**arXiv ID:** 2602.05448 | [PDF](https://arxiv.org/pdf/2602.05448v1)

**作者:** Sheshansh Agrawal `[一作]` (Contextual AI), Douwe Kiela `[通讯]` (Contextual AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于锦标赛图的 k‑wise 排序框架，用最少的查询次数在零样本情况下完成 top‑m 选择。

**💡 创新点**

创新点在于：①充分利用每次 k‑wise 比较所包含的完整锦标赛信息并通过传递闭包进行增量推理；②将非可传递偏好视为强连通分量，给出层级排序而非强制总序；③设计贪心查询调度保证每步都有进展并证明终止。

**🔧 技术方法**

主要技术包括锦标赛图理论、传递闭包推理、强连通分量 (SCC) 分析、基于信息增益的查询调度算法以及对 LLM 作为零样本排序器的调用。

**📊 数据集**

在 14 个检索基准（6 个 TREC Deep Learning、8 个 BEIR 数据集）和 5 种 LLM（GPT‑4.1、Gemini‑3‑Flash、GLM‑4.7、DeepSeek‑V3.2、Qwen3‑235B‑A22B‑Instruct）上进行实验。

**📈 对比分析**

与滑动窗口、TourRank、AcuRank、Setwise、Pairwise 等方法比较，结果显示该方法在保持或略优的 nDCG@10 的同时，输入 token 数量比基线平均减少 25–40%，比 pairwise 方法降低 7 倍，展现出显著的效率优势。

**⚠️ 局限性**

局限性在于：①假设判定器是确定性的；噪声（LLM 或人工作业）可能导致错误边破坏传递推理；②当前算法仅利用检索得分作为隐式先验，未充分结合先验信息；③对非可传递偏好的处理虽然合理，但在层级边界内的排序仍无明确标准。

---

## 331. Wikipedia and Grokipedia: A Comparison of Human and Generative Encyclopedias

**arXiv ID:** 2602.05519 | [PDF](https://arxiv.org/pdf/2602.05519v1)

**作者:** Ortal Hadad `[一作]` (Sapienza University of Rome), Walter Quattrociocchi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 13136 | [OpenAlex ID](https://openalex.org/A5008291667)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较维基百科与Grokipedia在内容选择、文本重写、叙事结构与评估框架的差异

**💡 创新点**

揭示生成媒介在选择与重写上的偏好，同时证明叙事结构基本保持一致的创新性发现

**🔧 技术方法**

利用逻辑回归、AMR抽取、情感平衡、LLM评估和拟合–复杂度框架等技术

**📊 数据集**

基于维基百科英文页面的浏览量、编辑与参考信息以及Grokipedia的页面与编辑日志，抽取热门5,000页及其子集

**📈 对比分析**

通过统计回归、网络相似度和情感平衡偏移进行对比；结果显示Grokipedia偏好高关注与争议页，叙事结构保持一致，框架略呈正面倾向

**⚠️ 局限性**

仅评估结构与框架，未检验事实准确性与说服力；Grokipedia编辑窗口短暂，跨平台对齐有限

---

## 332. DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter

**arXiv ID:** 2602.05513 | [PDF](https://arxiv.org/pdf/2602.05513v1)

**作者:** Xukun Li `[一作]` (Beijing Academy of Artificial Intelligence), Zhenguo Sun `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 813 | [OpenAlex ID](https://openalex.org/A5100732983)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 DECO，一个解耦多模态扩散 Transformer，用于双臂灵巧操作，并实现了插件式触觉适配器。

**💡 创新点**

创新点包括：①将视觉、运动学和触觉三种模态通过解耦注入方式分别处理；②设计轻量化 LoRA 触觉适配器，实现参数高效的触觉信息注入；③发布了 50 小时、5M 帧的 DECO‑50 双臂触觉数据集。

**🔧 技术方法**

技术手段主要是扩散 Transformer、AdaLN、跨模态注意力、LoRA 参数化适配以及可训练的触觉编码器。

**📊 数据集**

使用 DECO‑50 数据集，包含 4 个场景、28 子任务、约 50 小时录像、5M 帧和 8k 成功轨迹。

**📈 对比分析**

与 ACT、DP 等视觉基线比较，DECO 在所有任务中平均成功率 72.25%，比基线提升 21%；触觉适配器单独提升 10.25%，在复杂接触任务上提升 20%，且仅训练不到 10% 参数。

**⚠️ 局限性**

局限性在于对长时序任务的建模不足；触觉适配器在极少触觉采样时的鲁棒性待进一步提升。

---

## 333. LinguistAgent: A Reflective Multi-Model Platform for Automated Linguistic Annotation

**arXiv ID:** 2602.05493 | [PDF](https://arxiv.org/pdf/2602.05493v1)

**作者:** Bingru Li `[一作]` (University of Birmingham), Bingru Li `[通讯]` (University of Birmingham)

**通讯引用:** 2422 | [OpenAlex ID](https://openalex.org/A5100387270)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个名为LinguistAgent的无代码多代理平台，用于自动化文本标注和评估，特别聚焦于隐喻识别任务。

**💡 创新点**

创新点在于将注释者和评审者两大代理角色通过反射式多代理工作流组合，支持零/少样本提示、检索增强生成(RAG)和微调三种实验范式，并实现实时评估与可追溯的日志。

**🔧 技术方法**

采用大型语言模型（如Gemini 3、Qwen3）、多代理架构、Streamlit前端、Plotly可视化、Google GenAI SDK和OpenAI兼容接口、JSON原生输出等技术。

**📊 数据集**

使用IMDb隐喻数据集的子集作为实验文本，并以人工金标准作为评估基准。

**📈 对比分析**

通过对比Annotator-Only与Annotator+Reviewer两种模式，以及零样本、少样本和RAG三种实验范式，实验结果显示在所有模型中开启Reviewer模式可提升F1，Gemini 3在RAG下达到约0.58，Qwen3在少样本下约0.33。

**⚠️ 局限性**

局限在于多代理通信仍可能出现无声失效，受模型上下文长度与API配额限制，且当前未集成完整的RAG与微调流程，缺乏人类参与的交互验证。

---

## 334. Fine-Tuning Large Language Models for Automatic Detection of Sexually Explicit Content in Spanish-Language Song Lyrics

**arXiv ID:** 2602.05485 | [PDF](https://arxiv.org/pdf/2602.05485v1)

**作者:** Dolores Zamacola Sánchez de Lamadrid `[一作]` (Universidad Pontificia Comillas), Eduardo C. Garrido-Merchán `[通讯]` (Instituto de Investigación Tecnológica)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过微调GPT模型，对西班牙语雷鬼/Trap歌词进行自动检测性暗示内容。

**💡 创新点**

创新点在于结合专家手工标注与参考短语表、反馈驱动改进循环，使用极小数据集实现高精度性暗示检测，并提出可落地的多层级音乐内容评级框架。

**🔧 技术方法**

技术手段包括：GPT预训练模型微调、交叉熵损失与AdamW优化、精细的反馈循环、以及与ChatGPT基线的对照实验。

**📊 数据集**

数据集为100首拉丁城市音乐歌曲（50 explicit、50 non‑explicit），由专家手工标注并记录显式短语。

**📈 对比分析**

评估采用准确率、精确率、召回率、特异性等指标；初始评估准确率83%，精确率86%，召回率80%，特异性87%；反馈后精确率与特异性提升至100%，准确率87%，召回率73%；与ChatGPT基线对比，专家一致率59.2%对比55.1%，显著优于传统方法。

**⚠️ 局限性**

局限性包括：数据量极小、仅覆盖西班牙语雷鬼/Trap两类曲风、标注主观性高、可能存在文化偏差、以及对误判后果的社会伦理考量需进一步研究。

---

## 335. ALIVE: Awakening LLM Reasoning via Adversarial Learning and Instructive Verbal Evaluation

**arXiv ID:** 2602.05472 | [PDF](https://arxiv.org/pdf/2602.05472v1)

**作者:** Yiwen Duan `[一作]` (Independent Researcher), Xinpei Zhao `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ALIVE 框架，让 LLM 在无外部奖励的自监督循环中完成任务构造、求解与自评，实现端到端的自我提升。

**💡 创新点**

创新点在于将任务构造、解答与评判统一为单一模型的三种认知角色，利用自生成的口头反馈和软奖励克服奖励瓶颈，并通过对抗式任务生成强化难度与可解性。

**🔧 技术方法**

技术包括自监督强化学习（GRPO）、反馈条件化策略（FCP）、对抗式任务生成、口头反馈生成与软奖励融合、以及联合优化的多任务学习框架。

**📊 数据集**

主要使用未经标注的原始文本（将问答对拼接成文档），在数学推理（Big-Math、WebInstruct）、代码生成（SWE-smith、CodeContests、NuminaMath）、通用推理（GPQA-Diamond、MMLU、SuperGPQA、BBEH）等公开基准上评测。

**📈 对比分析**

与外部奖励 RL（GRPO、RFT）、监督式口头反馈（FCP+Bootstrap）以及 PretrainZero 等基线对比，ALIVE 在多项基准上显著提升 4–10%（如 GPQA-Diamond 45.96% vs 39.1%），并在跨域与长序列任务中表现出更强的鲁棒性与自检能力。

**⚠️ 局限性**

局限在于依赖模型自身生成的口头评判，若模型起始缺乏足够的指令遵循能力，需额外的教师提示预热；且在极端不确定或多解任务上，软奖励和自评仍可能产生偏差。

---

## 336. Emergence-as-Code for Self-Governing Reliable Systems

**arXiv ID:** 2602.05458 | [PDF](https://arxiv.org/pdf/2602.05458v1)

**作者:** Anatoly A. Krasnovsky `[一作]` `[通讯]` (Innopolis University), Anatoly A. Krasnovsky (Innopolis University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出 Emergence-as-Code（EmaC）框架，利用声明式旅程意图与运行时证据（如跟踪、路由配置、服务拓扑）结合，编译出可计算、可审计的旅程可靠性边界，并将这些边界转换为控制平面治理 artefacts（警报、滚动发布门槛、动作守卫）。

**💡 创新点**

创新点在于①将旅程层面的可观测性与可配置性统一到代码化的“意图 + 证据”模型；②引入共享命运和不确定性假设，给出可观测的可扩展置信区间；③通过 Git‑Ops 流程让治理 artefacts 变为可审计的版本化产物。

**🔧 技术方法**

技术手段包括：声明式旅程语法（控制流算子集合）、基于运行时数据的模型推断（利用 OpenTelemetry 跟踪、Prometheus 指标、服务网格配置）、区间推断（独立与共享命运下的最优/最差边界）、分布式延迟组合（卷积、极值统计）以及 GitOps+Argo Rollouts 生成的治理 artefacts。

**📊 数据集**

文中没有使用公开数据集，而是基于云原生平台常见的运行时指标与跟踪数据，示例性地提供了一个 “checkout” 旅程的 YAML/JSON 仓库，用以演示意图到 artefacts 的编译流程。

**📈 对比分析**

由于是概念性设计与演示，未进行量化实验或与现有工具（如 Sloth/Pyrra、OpenSLO）对比；所述方法通过理论边界分析和示例生成来展示潜在性能提升（如更安全的滚动发布门槛、更透明的可观测性），但缺乏实测指标。

**⚠️ 局限性**

局限性包括：①共享命运推断依赖稀疏故障数据，可能产生过宽或过窄的置信区间；②尾延迟组合假设独立/可观测分布，未覆盖复杂重试/反馈循环；③治理 artefacts 的自动化需要与多租户或多团队治理策略兼容，未深入探讨；④实现细节（如持续推断与合并冲突的阈值）仍需在实际系统中验证。

---

## 337. Ontology-Driven Robotic Specification Synthesis

**arXiv ID:** 2602.05456 | [PDF](https://arxiv.org/pdf/2602.05456v1)

**作者:** Maksym Figat `[一作]` (Warsaw University of Technology), Michel D. Ingham `[通讯]` (NASA Jet Propulsion Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种本体驱动的分层方法，将高层任务目标自动转化为可执行的多级规范，并利用带时间、随机性和资源约束的Petri网进行验证，最终生成ROS 2控制代码；

**💡 创新点**

创新点在于：① 结合IEEE 1872标准构建统一机器人本体，实现语义一致性；② 引入资源感知、随机时间Petri网支持多层级可执行建模；③ 通过中间可执行规范实现设计空间探索、性能评估与自动代码生成；

**🔧 技术方法**

使用的技术包括：本体建模（OWL/OWL‑RL）、带时间/随机/资源约束的Petri网、Monte Carlo仿真、RSSM与RSSL2规范语言、ROS 2与C++自动生成、MuJoCo物理仿真；

**📊 数据集**

数据集主要是人工构造的塔式堆叠场景（10 m×10 m 区域，三块绿盒子、一本红盒子）及其多机器人配置参数的随机化；未使用公开公开数据集；

**📈 对比分析**

通过与传统MBSE、SysML、LLM生成动作序列等方法比较，展示多机器人数量、能量约束下的完成成功率与任务耗时；实验表明两台机器人可在10 min内完成堆叠，成功率>90%，相较于单机72%；Monte Carlo仿真进一步验证系统可用性与可靠性；

**⚠️ 局限性**

局限性包括：本体覆盖范围有限，未充分考虑负面affordance；模型假设设备独立可靠性；仅在单一堆叠任务验证，缺乏更复杂任务的评估；随着层级增大模型规模和计算成本上升；

---

## 338. Forward Index Compression for Learned Sparse Retrieval

**arXiv ID:** 2602.05445 | [PDF](https://arxiv.org/pdf/2602.05445v1)

**作者:** Sebastian Bruch `[一作]` (Northeastern University), Rossano Venturini `[通讯]` (University of Pisa)

**通讯引用:** 1616 | [OpenAlex ID](https://openalex.org/A5084138015)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究了用于学习型稀疏检索的前向索引压缩技术，并提出一种专为内积计算优化的新压缩算法。

**💡 创新点**

创新点在于：①针对16位组件的整数序列设计了按8个值一组的控制字节格式，极大提升了SIMD解码速度；②在解压过程中直接在SIMD寄存器里完成组件解码、查询向量收集、值读取、乘加，避免了缓冲区拷贝；③通过对组件序列进行自适应排序（递归图二分）进一步压缩。

**🔧 技术方法**

技术主要包括整数压缩算法（VByte、Elias Gamma/Delta、Zeta、Varint-G8IU/StreamVByte）、自定义按8个值解码的SIMD实现、递归图二分排序、Rust语言实现、Sparse ANN框架Seismic。

**📊 数据集**

使用的公开数据集为MS MARCO（文档集）以及其查询集合，另外还使用了两种稀疏嵌入模型：一个是通用Bi‑Encoder嵌入，另一个是无查询嵌入的Learned Inference‑Free模型。

**📈 对比分析**

通过与基准压缩方法（VByte、Elias Gamma/Delta、Zeta、Varint-G8IU）在同一索引上进行对比，评估指标为平均查询延迟（µs）、索引体积（GB）以及召回率（90%、95%、97%、99%）。实验表明：①新算法在保持与Varint-G8IU相近的准确率的同时，内存占用比前者低约22%；②在保持相同召回率时，新算法的查询延迟仅为基准方法的1/3至1/2，且整体效率明显提升。

**⚠️ 局限性**

局限性：①仅压缩组件数组，对值（非零坐标的实数）压缩有限；②未考虑小幅度gap的sub‑byte压缩；③实验仅在单线程环境下进行；④适用性主要在16位组件的稀疏向量，其他位宽或稠密向量尚未验证。

---

## 339. Clouding the Mirror: Stealthy Prompt Injection Attacks Targeting LLM-based Phishing Detection

**arXiv ID:** 2602.05484 | [PDF](https://arxiv.org/pdf/2602.05484v1)

**作者:** Takashi Koide `[一作]` (NTT Security Holdings Corporation & NTT), Daiki Chiba `[通讯]` (NTT Security Holdings Corporation & NTT)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多模态LLM钓鱼检测系统面临的Prompt Injection攻击，构建两轴攻击分类体系，评估攻击成功率，并提出InjectDefuser防御框架。

**💡 创新点**

首创针对多模态LLM钓鱼检测的Prompt Injection两轴分类，量化攻击影响，并设计多层防御（Prompt硬化+RAG+输出校验）显著降低攻击成功率。

**🔧 技术方法**

使用Prompt硬化、带UUID的上下文边界、允许列表检索增强（RAG）、输出验证以及视觉与文本隐写等技术。

**📊 数据集**

构建基于10大目标品牌的HTML模板，生成2,000个含注入的样本；URL级PI 200个样本；以及10品牌未改动的网页用于系统评估。

**📈 对比分析**

在标准、Advanced 与 InjectDefuser 三种模式下对 GPT‑5、Grok 4 Fast、Llama 4、Gemma 3 四款LLM进行攻击成功率（ASR）测试；InjectDefuser 将 GPT‑5 的ASR 从 10.1% 降至 0.3%，其他模型亦显著降低，表明防御有效。

**⚠️ 局限性**

仅针对HTML/URL级PI的单一注入，缺乏真实场景用户可见性评估；需要持续维护品牌库和上下文边界；防御对高性能模型的抵御仍有限，且增加额外计算与维护成本。

---

## 340. Can We Classify Flaky Tests Using Only Test Code? An LLM-Based Empirical Study

**arXiv ID:** 2602.05465 | [PDF](https://arxiv.org/pdf/2602.05465v1)

**作者:** Alexander Berndt `[一作]` (Heidelberg University), Sebastian Baltes `[通讯]` (Heidelberg University)

**通讯引用:** 1315 | [OpenAlex ID](https://openalex.org/A5033132966)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估三种未微调的大型语言模型（GPT‑4o、GPT‑OSS、Qwen‑Code）在仅使用测试代码进行 flakiness 分类的效果，结合不同提示技术（zero‑shot、CoT、few‑shot）进行实验；

**💡 创新点**

探讨 LLM 在无上下文的 flakiness 分类中表现不佳的原因，并通过人工评估验证测试代码本身缺乏足够信息，提出需引入额外上下文或检索增强的解决思路；

**🔧 技术方法**

使用预训练 LLMs 与多种提示工程（含 chain‑of‑thought 与 few‑shot 示例）进行无监督分类；

**📊 数据集**

利用两大公开基准数据集——IDoFT（Java/Python）和 FlakeBench（Java）进行实验；

**📈 对比分析**

与全 flaky、随机分类基线比较，最佳 MCC 仅为 0.27（IDoFT）或 0.17（FlakeBench），远低于基准，且模型在多次运行间表现出 0–25% 的非确定性；

**⚠️ 局限性**

局限性包括仅评估两类 Java 测试数据、未使用任何微调或上下文信息、提示策略有限、样本量小、实验仅在实验环境下进行，难以直接推广至更广泛的实际 CI/CD 场景。

---

## 341. Conditional Diffusion Guidance under Hard Constraint: A Stochastic Analysis Approach

**arXiv ID:** 2602.05533 | [PDF](https://arxiv.org/pdf/2602.05533v1)

**作者:** Zhengyi Guo `[一作]` (Columbia University), Renyuan Xu `[通讯]` (Stanford University)

**通讯引用:** 768 | [OpenAlex ID](https://openalex.org/A5068576165)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于Doob h‑变换的硬约束条件扩散指导框架，能够在保持预训练扩散模型的分数网络不变的前提下，直接生成满足指定事件或稀有事件的样本。

**💡 创新点**

创新点包括：① 用马尔可夫性和二次变差理论构造两种离线学习目标（马尔可夫损失和协方差损失）来估计条件函数h及其梯度；② 在理论上给出非渐进的总变差和Wasserstein距离误差上界，明确了分数误差与指导误差的影响；③ 提供了可扩展至ODE采样和分类器指导的通用实现方案。

**🔧 技术方法**

核心技术包括：扩散模型的概率解释、Doob h‑变换、马尔可夫性质、二次变差回归、Girsanov变换、随机近似优化、Wasserstein与总变差误差分析。

**📊 数据集**

实验数据集涵盖：一维/二维截断正态分布（synthetic）、美国四只股票（AAPL、AMZN、TSLA、JPM）的历史日收益、以及模拟医院排队网络的生成数据。

**📈 对比分析**

通过与简单重抽样、软指导以及无指导扩散的对比，实验表明：在硬约束下CDG‑MCL与CDG‑ML分别在KS统计量（0.0694→0.0437）和Wasserstein距离（0.3451→0.0765）上取得显著提升；在金融压力测试中生成的收益分布与真实压力情景的分位数和波动率基本匹配；在供应链仿真中，软指导可降低队列长度波动并提高系统鲁棒性。

**⚠️ 局限性**

主要局限包括：① 对条件函数h的梯度估计依赖于二次变差回归，估计误差难以严格界定；② 需要对扩散模型的分数误差和梯度误差做严格假设，实际中可能不满足；③ 对指导尺度η的选择敏感，过大易导致模式崩塌；④ 在高维稀有事件场景下，采样效率和计算成本仍高于理想情况。

---

## 342. OpenMAG: A Comprehensive Benchmark for Multimodal-Attributed Graph

**arXiv ID:** 2602.05576 | [PDF](https://arxiv.org/pdf/2602.05576v1)

**作者:** Chenxi Wan `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7228 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OpenMAG，一个覆盖 19 个数据集、6 个领域、16 种编码器、24 种模型、8 个下游任务的统一多模态属性图（MAG）基准。

**💡 创新点**

创新点在于：①整合多模态与图结构的评测标准，填补现有基准在领域覆盖、编码器可调性、模型多样性和任务范围上的不足；②设计五维评估框架（必要性、数据质量、有效性、鲁棒性、效率）和 14 条系统性结论；③实现可扩展的开源库，支持冻结与可微编码、全链路训练与 PEFT。

**🔧 技术方法**

使用的技术包括：多模态编码（Frozen/Trainable 文本/视觉模型如 SBERT、CLIP、ViT、BERT、ViT 等）、图神经网络架构（GCN、GAT、DMGC、DGF、MMGCN、MGAT、LGMRec、UniGraph2 等）、LLM 辅助模型（GraphGPT‑O、InstructG2I 等）、参数高效微调（LoRA、Adapter）以及多任务评测与复杂度分析。

**📊 数据集**

利用 19 个公开数据集：电商商品、社交媒体、艺术网络、视频推荐、图书推荐、图片网络等，涵盖节点分类、链路预测、聚类、模态匹配/检索、模态生成（图→文本、图→图像）等八类任务。

**📈 对比分析**

方法比较基于统一管线，在同一数据集和任务上对 24 模型进行公平评测。实验显示：①图增强模型在大多数图任务上优于传统 GNN；②多模态增强模型在生成任务中表现突出；③LLM 增强模型在判别任务中相对欠缺；④LoRA 作为 PEFT 方案在生成质量上最优。性能提升从 5%~30% 量级不等，且在鲁棒性与效率维度提供了系统性比较。

**⚠️ 局限性**

局限性包括：①基准仍以静态图为主，缺少大规模动态图支持；②多模态编码多为预训练模型，参数量大，推理成本高；③模型多样性虽丰富，但缺少统一的泛化框架；④对低资源场景的鲁棒性与压缩研究不足。

---

## 343. Visual Implicit Geometry Transformer for Autonomous Driving

**arXiv ID:** 2602.05573 | [PDF](https://arxiv.org/pdf/2602.05573v1)

**作者:** Arsenii Shirokov `[一作]` (Lomonosov Moscow State University), Dmitry Senushkin `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5050618007)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

ViGT 提出一种能够从未标定的环视相机图像中学习连续三维占据场的视觉隐式几何 Transformer，并实现了无监督、自校准的训练方法。

**💡 创新点**

创新点包括：① 采用 calibration‑free 隐式 BEV 投影，使模型对不同相机布置具有良好可迁移性；② 用自监督的 LiDAR 点云监督构造连续占据场，无需昂贵的体素标签；③ 通过多尺度 ViT 编码器和交叉注意力实现多视角特征聚合，形成统一的场景中心 BEV 表示。

**🔧 技术方法**

核心技术包括：ViT‑Large 图像编码器、两阶段交叉注意力隐式 BEV 投影、基于 ImplicitIO 的隐式解码器、LiDAR 直线采样生成正负占据标签、二元交叉熵损失以及体素/点云渲染。

**📊 数据集**

在五个大规模自动驾驶数据集上训练：NuScenes、Waymo、NuPlan、ONCE、Argoverse 2，并在 Occ3D‑NuScenes 占据基准和多数据集点图估计任务上评测。

**📈 对比分析**

与现有基于像素对齐、已标定或需要体素标签的方法相比，ViGT 在 Occ3D‑NuScenes 的 F1/IoU 取得第三名（仅自监督且无标定），在五大数据集的点图估计任务中平均排名 1.8，显著优于 Self‑Occ、RenderOcc 等自监督/无标定方法。

**⚠️ 局限性**

局限性包括：① 训练需要大量多视角图像与 LiDAR 同步数据，仍受硬件资源限制；② 目前的 BEV 投影仅使用交叉注意力，缺乏对极端视角或遮挡场景的细粒度建模；③ 在极高分辨率或稀疏 LiDAR 传感器下的精度和实时性能尚未充分验证。

---

## 344. On Path-based Marginal Cost of Heterogeneous Traffic Flow for General Networks

**arXiv ID:** 2602.05565 | [PDF](https://arxiv.org/pdf/2602.05565v1)

**作者:** Jiachao Liu `[一作]` (Carnegie Mellon University), Sean Qian `[通讯]` (Carnegie Mellon University)

**通讯引用:** 5705 | [OpenAlex ID](https://openalex.org/A5082123014)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种近似计算多类交通流路径边际成本（PMC）的分析方法，并将其应用于多类路径/时段系统最优动态交通分配（SO‑DTA）。

**💡 创新点**

创新点主要包括：① 基于感知密度和道路空间分割引入交叉类转化因子，量化不同车辆类之间的相互影响；② 在多类细胞传输模型（CTM）框架下对非可微性进行分析，并采用下界子梯度近似PMC；③ 将上述PMC近似直接嵌入MSA迭代求解SO‑DTA，实现多类路径与时段的最优配置。

**🔧 技术方法**

使用的技术与方法有：多类CTM（考虑感知密度、空间分割与车种交互）；PMC分解（同类内外部性）与子梯度；Method of Successive Averages（MSA）迭代求解；时间依赖最短路径（DOT）算法用于寻找最小PMC的路径和出发时间。

**📊 数据集**

数据集包括两部分：① 采用人工构建的合流小型网络（10 条链接、2 条 OD 连接）用于初步验证；② 真实的马里兰州巴尔的摩附近网络，包含 1,510 条链接、776 个节点、15,376 条 OD 对，配准了实际交通需求（由 MDOT 提供）。

**📈 对比分析**

与动态用户均衡（DUE）做基准比较，并在小型网络中对比包含交叉类 PMC 与仅包含内类 PMC 的两种 SO‑DTA；结果显示：加入交叉类 PMC 可将总旅行成本降低 23%–32%，同时使收敛误差（gap）更小；在大规模网络上，交叉类 PMC 将总旅行时间成本分别降低约 9%（车辆）和 5%（卡车），并显著改善系统整体成本。

**⚠️ 局限性**

局限性包括：① 仅考虑双类车辆，未推广到多类情况；② 使用了简单的 MSA 迭代，收敛速度慢且易受非平滑性影响；③ 对非可微性仅采用子梯度近似，精度和收敛性尚有提升空间；④ 未来需设计更稳健的加速/稳态算法以处理大规模网络。

---

## 345. Taking the Leap: Efficient and Reliable Fine-Grained NUMA Migration in User-space

**arXiv ID:** 2602.05540 | [PDF](https://arxiv.org/pdf/2602.05540v1)

**作者:** Felix Schuhknecht `[一作]` (Johannes Gutenberg University), Nick Rassau `[通讯]` (Johannes Gutenberg University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于用户空间内存重连的异步页面迁移方法，解决传统 NUMA 迁移的控制、可靠性与池化内存等问题。

**💡 创新点**

核心创新在于：利用内存重连实现可调粒度迁移；通过 mprotect 与信号处理自动检测并处理并发写；支持已池化内存且能保证迁移最终完成。

**🔧 技术方法**

使用 mmap/mprotect、信号处理、主内存文件（memory‑rewiring）等技术实现物理页拷贝与虚拟页重映射，并在后台线程完成迁移。

**📊 数据集**

在实验中使用 1 GB 的 TPC‑H Lineitem 表（以及 4 GB / 128 GB 大小的内存块）作为数据集进行评测。

**📈 对比分析**

与 Linux 自动 NUMA 平衡和 move_pages() 进行对比，迁移时间在无写场景下可达原始 memcpy 的 50% 以内，且在并发写场景下吞吐量比两种基线高约 2 倍。

**⚠️ 局限性**

主要局限是需要支持主内存文件的 Linux 环境；在极高写压下仍会产生重试开销，且对迁移粒度的自适应需要经验性调参。

---

## 346. Strong Normalisation for Asynchronous Effects

**arXiv ID:** 2602.05528 | [PDF](https://arxiv.org/pdf/2602.05528v1)

**作者:** Danel Ahman `[一作]` (University of Tartu), Ilja Sobolev `[通讯]` (University of Tartu)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

研究了核心 λ-计算机（用于异步编程的代数效应）在无递归与有限重装中断处理下的强归一化性质，并给出完整的 Agda 形式化。

**💡 创新点**

提出了使用 Kripke 样式可归约性判据结合 ⊤⊤-lifting 的方法，并给出一种可归约的可重装中断处理变体，弥补了原先可重装中断导致的非归一化问题。

**🔧 技术方法**

采用 Girard–Tait 可归约性（type‑directed reducibility）与 ⊤⊤‑lifting 技术，配合效果注解、Kripke 可归约性索引与 Agda 形式化证明。

**📊 数据集**

无传统数据集，工作基于理论证明与 Agda 代码实现。

**📈 对比分析**

对比未引入重装中断的原始模型，证明在有限效果注解下并行进程也保持强归一化；与原文实验无直接性能对比，重点在理论安全性。

**⚠️ 局限性**

局限在于仍未覆盖所有扩展（如带状态的可重装中断、动态进程生成）且对更高级的效果类型（模态类型、状态）支持不足。

---

## 347. Capture the Flags: Family-Based Evaluation of Agentic LLMs via Semantics-Preserving Transformations

**arXiv ID:** 2602.05523 | [PDF](https://arxiv.org/pdf/2602.05523v1)

**作者:** Shahin Honarvar `[一作]` (Imperial), Alastair F. Donaldson `[通讯]` (Imperial)

**通讯引用:** 3088 | [OpenAlex ID](https://openalex.org/A5080781439)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过生成语义等价的CTF挑战族，系统评估agentic LLM在网络安全任务中的鲁棒性与工具使用能力。

**💡 创新点**

提出CTF挑战族概念，利用语义保持的程序变换自动生成多样化实例，并配套开源工具与大规模实验数据。

**🔧 技术方法**

采用重命名、循环/条件/函数注入、注释插入以及PyObfuscator等语义保持变换，结合agentic框架（如LlamaIndex、OpenAI API）实现工具调用与多轮交互。

**📊 数据集**

使用公开的Python CTF套件（如CTFTime的Dynastic、PK以及I-79等）共计约十余道题目，构建多族实例进行评测。

**📈 对比分析**

对数个LLM配置（大模型到中模型）进行多次重复实验，比较成功率、工具调用次数与token消耗，结果显示模型在重命名与单一注入变换下鲁棒性高，但在组合变换或PyObfuscator下性能显著下降。

**⚠️ 局限性**

局限性包括仅针对Python，未覆盖多语言与混合语言挑战；变换种类有限；实验受token预算限制；数据集可能受训练集泄漏影响；缺乏对模型内部决策机制的深度解析。

---

## 348. CAViT -- Channel-Aware Vision Transformer for Dynamic Feature Fusion

**arXiv ID:** 2602.05598 | [PDF](https://arxiv.org/pdf/2602.05598v1)

**作者:** Aon Safdar `[一作]` (University College Dublin), Mohamed Saadeldin `[通讯]` (University College Dublin)

**通讯引用:** 49 | [OpenAlex ID](https://openalex.org/A5102745668)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CAViT，一个将 Vision Transformer 的静态 MLP 替换为通道自注意力的双注意力结构。

**💡 创新点**

创新点在于将通道维度视为 tokens，通过维度交换进行自注意力，实现动态、内容感知的通道混合，并将空间与通道注意力统一到同一 attention‑only 块中，既提升表达能力又不增加深度或额外模块。

**🔧 技术方法**

技术手段包括多头空间自注意力（MHSA）、单头通道自注意力（SHSA）、维度交换操作、CLS token 处理与基于 ViT 的 Transformer 块设计。

**📊 数据集**

使用了五个不同领域的数据集：CIFAR‑10、Cats vs Dogs、Malaria、PneumoniaMNIST、BreastMNIST。

**📈 对比分析**

与 ViT_tiny 在同等训练设置下比较，CAViT 在所有数据集上取得最高 3.6% 的 Top‑1 准确率提升，同时参数量与 FLOPs 各下降约 30%，收敛更快且注意力图更具语义性。

**⚠️ 局限性**

局限性：仅在 ViT_tiny 与中小规模数据集上验证，未尝试大型模型、ImageNet 等大规模基准；未结合自监督、蒸馏或多模态任务；维度交换在更大模型或多任务场景下的效率与实现细节尚待进一步研究。

---

## 349. Deep Learning for Contextualized NetFlow-Based Network Intrusion Detection: Methods, Data, Evaluation and Deployment

**arXiv ID:** 2602.05594 | [PDF](https://arxiv.org/pdf/2602.05594v1)

**作者:** Abdelkader El Mahdaouy `[一作]` (Mohammed VI Polytechnic University), Ismail Berrada `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化了流量层面入侵检测的上下文感知深度学习方法，提出四维（时间、图谱、多模态、多分辨率）分类框架并强调严谨评估与部署实用性。

**💡 创新点**

提出统一的四维上下文分类法，首次将评估规范与数据集质量问题纳入讨论，为未来研究提供可复制、可对比的基准与方法论。

**🔧 技术方法**

利用循环/卷积/Transformer序列模型、图神经网络（同构/异构/动态/联邦）、多模态融合策略、层次聚合与自监督预训练等多种深度学习技术。

**📊 数据集**

引用并评估主流公开数据集（CIC‑IDS‑2017、CSE‑CIC‑IDS‑2018、MAWIFlow、CTU‑13、IoT‑specific 数据集等）以及新近的多模态测试床。

**📈 对比分析**

通过对比报告的指标（F1、AUROC、误报率、跨数据集泛化）指出：上下文感知模型在多阶段、分布式攻击上相对传统单流分类提升10–30%（视数据集与评估协议而定），但多数提升在不严格时间顺序、无泄漏验证下失真。

**⚠️ 局限性**

主要限制包括：评估流程易产生时序泄漏，公开数据集标签噪声与多样性不足导致泛化差；模型对低延迟、高吞吐的部署仍欠缺；在自监督与联邦学习等新技术的实用性与鲁棒性尚未充分验证。

---

## 350. A Mixed Reality System for Robust Manikin Localization in Childbirth Training

**arXiv ID:** 2602.05588 | [PDF](https://arxiv.org/pdf/2602.05588v1)

**作者:** Haojie Cheng `[一作]` (National University of Singapore), Eng Tat Khoo `[通讯]` (National University of Singapore)

**通讯引用:** 436 | [OpenAlex ID](https://openalex.org/A5059188359)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文开发了一个基于混合现实的分娩训练系统，利用Quest头显与外置RGB‑D摄像头，结合标定的物理孕妇仿真模型和新生儿头部点云配准，提供实时手部引导，提升学生分娩技能。

**💡 创新点**

创新点在于首次实现了独立可部署的MR分娩训练框架：1) 将外部RGB‑D摄像头与Quest实现无缝对齐；2) 采用粗到细的多标记＋点云配准策略，精准定位孕妇与新生儿；3) 通过虚拟引导手与物理仿真模型的空间同步，弥补传统VR缺乏触觉与物理交互的不足。

**🔧 技术方法**

技术主要包括：外部RGB‑D摄像头与Quest的eye‑to‑hand标定；多标记检测与深度融合实现孕妇姿态估计；Fast Global Registration 与 ICP 点云配准定位新生儿头部；实时手部轨迹捕捉与可视化；以及基于HoloLens/Quest的passthrough MR渲染。

**📊 数据集**

使用的数据集为两套Laerdal Prompt Flex产科仿真孕妇模型与预先扫描的新生儿头部点云（通过结构光扫描得到），以及83名四年级医学生在实际训练过程中的视频记录和DOPS评分作为评测数据。

**📈 对比分析**

与传统VR训练对比，MR训练在DOPS评分上表现更好：总分中位数从VR的5.5提升到MR的7.5（p=0.003），尤其在交付、产后处理与整体操作上显著优于VR；同时在问卷中MR的参与度、存在感与偏好得分均高于VR，说明MR能更好地吸引学习者并提升技能获取。

**⚠️ 局限性**

局限性包括：1）仅适用于正常分娩情景，无法处理非刚性姿态如臀位或肩难产；2）缺乏力反馈机制，无法模拟真实施力限制；3）需要在孕妇模型上贴标记，易受光照与润滑剂影响；4）系统仍依赖外置RGB‑D摄像头，部署成本与维护相对较高；5）研究仅在单一机构、短期培训内验证，缺乏长期保留与跨机构的泛化评估。

---

## 351. PIRATR: Parametric Object Inference for Robotic Applications with Transformers in 3D Point Clouds

**arXiv ID:** 2602.05557 | [PDF](https://arxiv.org/pdf/2602.05557v1)

**作者:** Michael Schwingshackl `[一作]` (AIT Austrian Institute of Technology), Markus Murschitz `[通讯]` (AIT Austrian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出端到端的PIRATR框架，能够从部分观测的LiDAR点云中同时估计多类别的6-DoF姿态和类特定参数（如抓手开合角度）。

**💡 创新点**

创新点在于将PI3DETR的Transformer架构与可微分的几何匹配、类特定前馈头相结合，实现参数化对象检测，并在仅用仿真数据训练后即可无调优地迁移到真实户外LiDAR环境。

**🔧 技术方法**

使用技术包括点云采样与特征聚合、Transformer检测器、几何感知匹配（考虑对称性）、Chamfer距离正则化以及类特定的前馈回归网络。

**📊 数据集**

使用的数据集为Blender生成的5k合成点云（包含起重机抓手、装载平台、托盘等）以及实际收集的73场景Livox Mid‑70 LiDAR扫描数据。

**📈 对比分析**

与PI3DETR及传统Vox/Point/BEV方法对比，合成测试mAP达0.982，实景测试mAP为0.919，几何误差均低于10–20 cm/10–20°，显示出显著优于现有方法的性能。

**⚠️ 局限性**

主要局限包括对堆叠托盘、遮挡严重场景的检测精度仍有下降；训练完全基于仿真，缺乏多样化真实数据；在400k点时推理耗时约0.6 s，对极端实时性要求仍存在挑战。

---

## 352. SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation

**arXiv ID:** 2602.05534 | [PDF](https://arxiv.org/pdf/2602.05534v1)

**作者:** Youngwoo Shin `[一作]` (Korea Advanced Institute of Science and Technology), Junmo Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 9409 | [OpenAlex ID](https://openalex.org/A5100606266)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了训练无关的Scaled Spatial Guidance (SSG)，在VAR模型推理时对logit进行频域增强，提升细节与全局一致性。

**💡 创新点**

创新点是通过信息瓶颈理论导出对semantic residual的logit加权，引入频域Prior DSE实现多尺度细节引导，且不需要额外训练。

**🔧 技术方法**

使用信息瓶颈理论、频域离散余弦变换（DCT）、logit级别的加权引导、VAR框架下的next‑scale预测。

**📊 数据集**

在ImageNet 256×256/512×512以及MJHQ‑30K文本到图像数据集上进行评测。

**📈 对比分析**

与Diffusion、GAN、Masked和VAR等主流模型对比，SSG在保持低延迟（≤1×VAR-d30）的同时将FID从3.42降至1.68，在文本条件下将FID从8.46降至7.28，明显优于同类方法。

**⚠️ 局限性**

局限包括对Prior质量依赖较高，β参数需手工调节，且在极高分辨率或非VAR结构时效果可能下降。

---

## 353. BhashaSetu: Cross-Lingual Knowledge Transfer from High-Resource to Extreme Low-Resource Languages

**arXiv ID:** 2602.05599 | [PDF](https://arxiv.org/pdf/2602.05599v1)

**作者:** Subhadip Maji `[一作]`, Arnab Bhattacharya `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

演示如何使用ACL风格文件在LuaLaTeX或XeLaTeX中编写文档

**💡 创新点**

提供了不同语言文本的示例以及引用格式

**🔧 技术方法**

使用了LaTeX、LuaLaTeX或XeLaTeX

**📊 数据集**

无数据集

**📈 对比分析**

无方法比较

**⚠️ 局限性**

缺乏实际实验与评估

---

## 354. A Hybrid CNN and ML Framework for Multi-modal Classification of Movement Disorders Using MRI and Brain Structural Features

**arXiv ID:** 2602.05574 | [PDF](https://arxiv.org/pdf/2602.05574v1)

**作者:** Mengyu Li `[一作]` (University of Iceland), the ASAP Neuroimaging Initiative `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种融合3D CNN与机器学习的多模态框架，利用T1加权MRI、12个深脑结构分割掩码及其体积特征，对帕金森综合征子类型（PSP、MSA与PD）实现高精度早期鉴别诊断。

**💡 创新点**

创新点在于：①将影像、分割掩码和体积量化三种模态同时输入模型，②采用分支式空间金字塔CNN提取局部与全局特征；③在二阶段融合中将CNN高维特征与体积特征拼接，使用Logistic回归进一步提高判别力；④通过3D Grad‑CAM实现模型可解释性，验证其关注点与临床典型病理区块一致。

**🔧 技术方法**

技术要点包括：3D卷积网络（分支式金字塔结构）、Batch Normalization、全局平均池化、Dense层与Dropout、Adam优化器、加权交叉熵、L2正则化的Logistic回归、类别平衡权重、5折交叉验证、3D Grad‑CAM可视化。

**📊 数据集**

数据集来源于ASAP Neuroimaging Initiative，包含554例多中心T1加权MRI（PD 285，PSP 192，MSA 23，MSA‑C 20，MSA‑P 34）。每例图像经过12个深脑结构的分割，计算体积并以ICV标准化，形成三模态输入。

**📈 对比分析**

在PSP vs PD、MSA vs PD、PSP vs MSA三项二分类任务上做消融实验，比较单模态（影像/掩码/体积）与混合模态（MRI+掩码+体积）及CNN/ML单一模型。最佳混合模型在PSP vs PD的AUC 0.95、F1 0.91、准确率 89.6%；MSA vs PD的AUC 0.86、F1 0.62、准确率 79%；PSP vs MSA的AUC 0.92、F1 0.83、准确率 83%。

**⚠️ 局限性**

局限性包括：①MSA子类型样本稀少导致MSA vs PD和PSP vs MSA的性能相对较低；②数据仍以单一T1影像为主，缺乏DTI或其他模态验证；③模型对训练集的多中心差异性尚未完全鲁棒，需进一步外部验证；④未展开多类别或纵向学习，未来可拓展。

---

## 355. ShapeGaussian: High-Fidelity 4D Human Reconstruction in Monocular Videos via Vision Priors

**arXiv ID:** 2602.05572 | [PDF](https://arxiv.org/pdf/2602.05572v1)

**作者:** Zhenxiao Liang `[一作]` (University of Texas at Austin), Jing Xiao `[通讯]` (PAII Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建并优化基于视觉先验的可变形3D高斯模型，实现从单目视频中高保真、无模板的4D人体重建

**💡 创新点**

使用多帧参考、2D视觉先验和自定义对齐深度，首次在单目环境下实现无模板的高保真人体动态重建

**🔧 技术方法**

3D高斯喷射、DensePose、深度估计、稀疏点对齐、神经变形场、同步密度控制、跨帧损失等技术

**📊 数据集**

ZJU Mocap、NeuMan 等真实单目人体数据集

**📈 对比分析**

与 4DGS、Deformable‑GS、HUGS、GART、Shape‑of‑Motion 等方法对比，PSNR/SSIM/LPIPS 等指标均优于同类方法，尤其在高变形运动场景表现突出

**⚠️ 局限性**

对外部语义先验和相机位姿高度依赖；不支持全新姿态或多人体交互；对复杂遮挡场景的鲁棒性有限

---

## 356. TangramSR: Can Vision-Language Models Reason in Continuous Geometric Space?

**arXiv ID:** 2602.05570 | [PDF](https://arxiv.org/pdf/2602.05570v1)

**作者:** Yikun Zong `[一作]` (University of Cambridge), Cheston Tan `[通讯]` (A*STAR)

**通讯引用:** 1475 | [OpenAlex ID](https://openalex.org/A5073326796)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于视觉语言模型的测试时自我改进框架，利用迭代反馈和奖励引导，无需参数更新即可显著提升Tangram拼图的连续几何推理性能。

**💡 创新点**

创新点在于把人类认知机制（心理旋转、迭代细化、视觉反馈）转化为AI推理流程，并设计了无训练的奖励驱动自我细化循环，使模型在连续空间中实现自我提升。

**🔧 技术方法**

采用在上下文中提示（ICL）与基于IoU与位置误差的奖励函数相结合的自我改进循环，并在必要时进行局部网格搜索，输出JSON形式的位姿预测。

**📊 数据集**

使用自建的Tangram连续空间基准（SVG → JSON → PNG），包含单片与双片任务，并提供位置、角度、尺寸的标注数据。

**📈 对比分析**

通过与五大主流VLM（Qwen、GPT‑4o、LLaMA、Gemini、Claude）在单片和双片任务上进行IoU、误差等指标评估，基线IoU仅0.2–0.45，改进后单片可达0.93，双片仍约0.23，证明自我细化显著提升性能。

**⚠️ 局限性**

局限性包括仅针对单片/双片拼图；评估依赖栅格化IoU，对分辨率敏感；奖励函数固定，缺乏自适应；当初始预测严重失准时循环可能不收敛，缺乏安全与鲁棒性保障。

---

## 357. FastVMT: Eliminating Redundancy in Video Motion Transfer

**arXiv ID:** 2602.05551 | [PDF](https://arxiv.org/pdf/2602.05551v1)

**作者:** Yue Ma `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14253 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本工作提出FastVMT，一种训练‑free 视频运动转移框架，利用滑动窗口注意力与梯度跳过技术显著提升速度并保持视觉与时序一致性。

**💡 创新点**

创新点在于通过局部窗口化的注意力搜索消除运动冗余，并在扩散过程中缓存梯度实现梯度重用，从而在保持高质量的前提下加速推理。

**🔧 技术方法**

主要技术包括基于DiT的视频扩散模型、滑动窗口运动提取策略、对应窗口损失、以及间隔梯度重用（step‑skipping）优化。

**📊 数据集**

实验数据集涵盖DAVIS、VBench以及40段真实世界视频，使用WAN‑2.1等公开视频生成模型作为基线。

**📈 对比分析**

与MOFT、MotionInversion、MotionDirector、DiTFlow等方法对比，FastVMT在运动保真度、时序一致性和文本相似度指标上均优于或相当于SOTA，同时实现了约3.43倍的平均加速和14.91倍的延迟降低。

**⚠️ 局限性**

局限性包括仍需依赖大型预训练扩散模型、对极端运动或高度动态场景的处理可能受限，以及在多目标交互复杂场景下的细粒度运动一致性尚需进一步验证。

---

## 358. Logical Guidance for the Exact Composition of Diffusion Models

**arXiv ID:** 2602.05549 | [PDF](https://arxiv.org/pdf/2602.05549v1)

**作者:** Francesco Alesiani `[一作]` (NEC Laboratories Europe), Mathias Niepert `[通讯]` (University of Stuttgart)

**通讯引用:** 3851 | [OpenAlex ID](https://openalex.org/A5031719069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种逻辑引导框架，能够在扩散模型推理时以精确的方式实现对复杂布尔逻辑表达式的约束生成。

**💡 创新点**

创新点在于给出了布尔算子（与、或、非）的精确递归组合规则，并证明在满足条件独立或互斥的电路结构下可以完全精确计算后验概率与逻辑得分；同时提出混合分类器与无分类器指导的混合方法，实现了不需要梯度的概率加权组合。

**🔧 技术方法**

技术包括：扩散模型的无分类器与分类器指导、贝叶斯后验概率估计、概率电路（可分解与确定性求和节点）以及递归逻辑组合规则。

**📊 数据集**

在图像生成上使用 CMNIST、Shapes3D、CelebA、ImageNet 等数据集；在分子生成上使用 GRM5‑RRM1 蛋白配体双靶点设计实验。

**📈 对比分析**

与传统的常数混合权重方法（静态组合基线）相比，新方法在布尔逻辑（尤其是或、非、多层组合）下的符合率（Conformity Score）提升超过 20%，且保持更高的多样性（熵）。在 CelebA 的视觉质量指标 FID 上表现更佳；在分子设计中，双靶点和选择性靶点的对接分数接近或优于专用双靶点方法 DualDiff。

**⚠️ 局限性**

局限性包括：需要对原子预测的后验概率估计保持足够精确，估计误差会被非线性系数放大；精确性仅在满足条件独立/互斥的电路结构时可保证，实际数据中这些假设可能只近似成立；总体计算复杂度与原子谓词数量线性相关，极大逻辑表达式仍可能开销较高。

---

## 359. Reasoning-guided Collaborative Filtering with Language Models for Explainable Recommendation

**arXiv ID:** 2602.05544 | [PDF](https://arxiv.org/pdf/2602.05544v1)

**作者:** Fahad Anwaar `[一作]`, Kezhi Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 RGCF-XRec 框架，将协同过滤知识与 LLM 进行融合，实现可解释的序列推荐。

**💡 创新点**

创新地将协同过滤知识通过推理导向（CoT）强化，并通过统一表示学习将协同与语义信息对齐，且在单一步骤中同时完成下一条推荐和可解释文本生成。

**🔧 技术方法**

采用预训练协同过滤模型 SASRec、SBERT 文本编码、LoRA 轻量化 LLaMA‑3.2‑3B 进行 CoT 生成、统一投影网络和多维评分机制。

**📊 数据集**

在 Amazon Sports、Toys、Beauty 三个公开数据集（共 642,503 条交互）进行评估。

**📈 对比分析**

与九种推荐基线和五种解释生成基线进行对比，HR@10、NDCG@10 提升 7.38%/4.59%，ROUGE‑L 提升 8.02%/3.49%，显著降低冷-热项目差距并实现 18–23% 的零样本提升。

**⚠️ 局限性**

仍受协同过滤依赖用户历史的稀疏性限制，且 CoT 生成与评分机制可能引入偏见；评估仅基于自动指标，缺乏人类主观质量验证。

---

## 360. Steering Large Reasoning Models towards Concise Reasoning via Flow Matching

**arXiv ID:** 2602.05539 | [PDF](https://arxiv.org/pdf/2602.05539v1)

**作者:** Yawei Li `[一作]` (Ludwig Maximilian University of Munich), Cheng Wang `[通讯]` (Amazon)

**通讯引用:** 2550 | [OpenAlex ID](https://openalex.org/A5051931459)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为大型推理模型提供一种非线性引导机制，旨在压缩冗长推理步骤，生成更简洁、紧凑的推理过程。

**💡 创新点**

创新点在于用流匹配（Flow Matching）学习从冗长到简洁推理激活分布的完整变换，而不是传统的单一线性向量平移，能够实现输入相关、精细的推理控制。

**🔧 技术方法**

核心技术是流匹配作为速度场来逼近分布转移，结合Transformer隐藏状态的非线性变换，得到针对每个输入的动态引导策略。

**📊 数据集**

在多种推理基准上验证，包括GSM‑8K、MMLU、ARC、TruthfulQA等，覆盖算术、通用知识、科学推理等任务。

**📈 对比分析**

与主流推理时基线（如提示压缩、参数剪枝、权重分块）对比，实验表明该方法在保持甚至提升任务准确率的同时，显著减少了所需的token数（token‑efficiency提升约20–30%）。

**⚠️ 局限性**

局限性包括：① 需要额外的训练步骤来学习流匹配变换，训练成本较高；② 在极大模型规模下推理时的额外算子可能导致计算开销上升；③ 对极端长推理仍存在一定冗余，尚需进一步优化。

---

## 361. Path-Guided Flow Matching for Dataset Distillation

**arXiv ID:** 2602.05616 | [PDF](https://arxiv.org/pdf/2602.05616v1)

**作者:** Xuhui Li `[一作]` (MBZUAI), Zhiqiang Xu `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出在VAE潜空间中采用流匹配进行数据集蒸馏，开发了Path‑Guided Flow Matching（PGFM）框架。

**💡 创新点**

创新点在于首次将流匹配与轻量级路径引导相结合，采用早期阶段的原型引导、暖启动以及信任域控制，以提升模式覆盖而不损失细节。

**🔧 技术方法**

使用预训练的GMFlow流匹配生成器、冻结的VAE编码/解码、ODE积分、K‑means原型聚类、路径引导、早停和信任域等技术。

**📊 数据集**

实验数据集包括高分辨率ImageNette、ImageIDC（ImageNet100）以及ImageNet‑1K，IPC设置从10到50不等。

**📈 对比分析**

与Diffusion基准（MiniMaxDiff、MGD³、DiT）对比，PGFM在IPC≤20时在ConvNet‑6、ResNet‑18、ResNet‑AP等三种backbone上提升2–5个百分点，采样步数仅32步，效率提升7.6×以上。

**⚠️ 局限性**

局限在于对暖启动和原型选取较为敏感，过强的路径引导可能导致模糊，且在IPC较高时性能提升趋于平稳。

---

## 362. HiCrowd: Hierarchical Crowd Flow Alignment for Dense Human Environments

**arXiv ID:** 2602.05608 | [PDF](https://arxiv.org/pdf/2602.05608v1)

**作者:** Yufei Zhu `[一作]` (Orebro University), Allan Wang `[通讯]` (Miraikan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 HiCrowd 分层框架，将强化学习与模型预测控制结合，用人群运动引导移动机器人在密集人群中导航。

**💡 创新点**

创新点在于把人群视为导航引导源，RL 生成跟随点并通过人群跟随奖励加速学习；分层决策将长期决策与短期安全控制分离，显著减少机器人冻结现象。

**🔧 技术方法**

采用强化学习（SAC）与模型预测控制（MPC）相结合；利用 DBSCAN 聚类检测人群流；设计跟随点、进展奖励与安全成本等多维奖励。

**📊 数据集**

使用 ETH‑UCY 实验集（ETH、HOTEL、UNIV、ZARA1、ZARA2）和自建合成密集人群数据集。

**📈 对比分析**

与 ORCA、SARL、CrowdAttn、纯 MPC 进行离线/在线对比，HiCrowd 在所有设置中实现 100% 成功率、最低导航时间、最低碰撞率和最低冻结频率，整体性能最佳。

**⚠️ 局限性**

限制在于高层与低层动作比例固定，导致在快速变化的人群条件下反应延迟；未考虑静态障碍；缺乏学习的人群运动预测以进一步提升鲁棒性。

---

## 363. Shiva-DiT: Residual-Based Differentiable Top-$k$ Selection for Efficient Diffusion Transformers

**arXiv ID:** 2602.05605 | [PDF](https://arxiv.org/pdf/2602.05605v1)

**作者:** Jiaji Zhang `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**通讯引用:** 9252 | [OpenAlex ID](https://openalex.org/A5055284175)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于残差的可微Top‑k选择的Shiva‑DiT框架，实现Diffusion Transformer中的动态Token剪枝，兼顾可微、效率和严格预算；

**💡 创新点**

①残差基可微排序与残差STE实现可学习的预算k；②上下文感知路由器与自适应比例策略；③硬Top‑k保证静态张量形状，消除ragged tensor；

**🔧 技术方法**

可微排序（soft rank）、残差STE、上下文路由器、自适应比例网络、层间共享路由、EMA预算约束、分层采样、蒸馏等技术；

**📊 数据集**

在MJHQ‑30K微调数据集上，对SD3.5、Flux.1‑dev、PixArt‑Σ等模型进行评测；

**📈 对比分析**

与ToMeSD、ToFu、SDTM、ToMA、IBTM、DiffCR、DyDiT、SparseDiT等基线对比，Shiva‑60%实现1.54×壁钟加速，CLIP分数超越finetuned，在SD3.5上获得新的Pareto前沿；

**⚠️ 局限性**

依赖空间冗余，稠密或细节丰富场景中可能丢失信息；完全丢弃未选Token可能导致细节损失。

---

## 364. On the Superlinear Relationship between SGD Noise Covariance and Loss Landscape Curvature

**arXiv ID:** 2602.05600 | [PDF](https://arxiv.org/pdf/2602.05600v1)

**作者:** Yikuan Zhang `[一作]` (Peking University), Yuhai Tu `[通讯]` (Flatiron Institute)

**通讯引用:** 12880 | [OpenAlex ID](https://openalex.org/A5101587306)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了随机梯度下降（SGD）噪声协方差与损失曲率（Hessian）之间的关系，证明其并非线性，而是超线性，并给出了理论推导与实验验证；

**💡 创新点**

创新点在于放弃传统的Fisher信息矩阵可实现性假设，提出基于Activity–Weight Duality（AWD）的通用框架，得到噪声协方差为每样本Hessian二阶矩的期望，推导出幂律指数γ的理论上界1≤γ≤2；

**🔧 技术方法**

主要技术包括AWD理论、矩阵对角化与交换性分析、随机矩阵理论、抑制实验（Eigenvalue suppression）以及对比交叉熵（CE）和均方误差（MSE）损失的本征值-方向相关性；

**📊 数据集**

实验数据集涵盖MNIST和CIFAR‑10（分别使用3、6、10分类子集），以及多种网络结构（MLP与CNN），在不同损失下训练到收敛；

**📈 对比分析**

通过绘制对数-对数图 C_ii 与 H_ii 的关系，拟合得到指数γ，并与AWD理论预测进行对比，实验发现 CE 的 γ≈1.4、MSE 的 γ≈1，均落在理论上界内，显示出比传统Fisher比例更准确的噪声-曲率关系；

**⚠️ 局限性**

局限性在于推导仅在局部最小附近假设曲率正定；对不同网络架构的 Δw 具体形式未完全给出；对 CE 与 MSE 产生不同 γ 的根本机制仍未深入阐明。

---

## 365. TOLEBI: Learning Fault-Tolerant Bipedal Locomotion via Online Status Estimation and Fallibility Rewards

**arXiv ID:** 2602.05596 | [PDF](https://arxiv.org/pdf/2602.05596v1)

**作者:** Hokyun Lee `[一作]` (Seoul National University), Jaeheung Park `[通讯]` (Seoul National University)

**通讯引用:** 3043 | [OpenAlex ID](https://openalex.org/A5031070386)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 TOLEBI，一套针对双足机器人在发生关节锁死或功率失效等硬件故障时仍能保持稳定行走的强化学习框架。

**💡 创新点**

创新点包括：① 通过联合学习关节状态估计器，使策略实时感知关节健康状态；② 采用课程学习（先训练正常行走，再逐步引入失效与扰动）；③ 设计了“fallibility reward”以及相位调制动作以增强对失效情况下的行走鲁棒性；④ 将这些方法成功迁移到真实 humanoid 机器人 TOCABI 上。

**🔧 技术方法**

技术主要包括：Proximal Policy Optimization（PPO）在 Isaac Gym 里训练；关节状态估计采用 GRU 网络；域随机化与动力学随机化；动作空间包含 12 轴扭矩与相位调制；奖励设计结合速度跟踪、身体姿态、关节动力学与失效惩罚。

**📊 数据集**

使用的“数据集”是自建的仿真环境（4096 并行 MuJoCo/Isaac Gym 任务），在此产生的轨迹与失效标签作为训练数据；随后将学习得到的策略迁移到真实 TOCABI 机器人进行实验验证。

**📈 对比分析**

与基线策略和仅使用关节掩码/状态估计的版本相比，TOLEBI 在关节锁死条件下的成功率提升至 81.27%，在功率失效条件下提升至 52.67%；在真实机器人上实现了平地行走与楼梯下行且保持速度跟踪，表现出显著的鲁棒性。

**⚠️ 局限性**

局限性：目前仅支持单一关节失效；未处理多关节同时失效或更复杂的非结构化环境；实验只在 TOCABI 机器人上验证，泛化性仍待进一步评估。

---

## 366. EgoPoseVR: Spatiotemporal Multi-Modal Reasoning for Egocentric Full-Body Pose in Virtual Reality

**arXiv ID:** 2602.05590 | [PDF](https://arxiv.org/pdf/2602.05590v1)

**作者:** Haojie Cheng `[一作]` (National University of Singapore), Eng Tat Khoo `[通讯]` (National University of Singapore)

**通讯引用:** 436 | [OpenAlex ID](https://openalex.org/A5059188359)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 EgoPoseVR，一种整合 HMD 运动与头戴式下视 RGB‑D 摄像机的双流时空框架，实现无外部基础设施的全身姿态估计。

**💡 创新点**

创新点在于（1）利用双流时空编码结合跨模态注意力融合实现全身姿态精细化；（2）引入可视性感知热图与 Kinematic Pose Optimization 能力提高鲁棒性；（3）构建首个 VR 视角同步 RGB‑D 与 HMD 运动的大规模合成数据集。

**🔧 技术方法**

技术手段包括 Transformer 时空编码器、交叉注意力融合、可视性概率预测、欧式过滤、逆运动学能量优化以及深度视觉热图回归。

**📊 数据集**

使用 EgoPoseVR 语料：1.8M 帧合成数据集，包含 RGB‑D、HMD 轨迹、2D/3D 关节标注；对比实验亦引用 Mo2Cap2、xR‑EgoPose、EgoGlass、UnrealEgo、ARES、SynthEgo 等。

**📈 对比分析**

与单模态基线（EgoPoser、EgoPoseFormer）和多模态改进版对比，EgoPoseVR 在上身 MPJPE‑U 仅 1.66 cm、下身 4.75 cm、帧率 97 FPS，显著优于现有方法。

**⚠️ 局限性**

局限性包括对视角遮挡和低视野场景敏感、合成与真实差距仍存、缺乏生物力学约束导致高频肌肉动态缺失。

---

## 367. Geometric Observability Index: An Operator-Theoretic Framework for Per-Feature Sensitivity, Weak Observability, and Dynamic Effects in SE(3) Pose Estimation

**arXiv ID:** 2602.05582 | [PDF](https://arxiv.org/pdf/2602.05582v1)

**作者:** Joe-Mei Feng `[一作]` (Tamkang University), Sheng-Wei Yu `[通讯]` (Tamkang University)

**通讯引用:** 5259 | [OpenAlex ID](https://openalex.org/A5038099773)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出了一种统一的 Lie 群算子框架，用于量化在 SE(3) 估计中单个特征的几何灵敏度，并定义了几何可观测指数（GOI）；

**💡 创新点**

创新点在于将影响函数理论、Fisher 信息几何与条件数分析在 SE(3) 上统一，并通过可观测子空间的曲率谱给出特征级的灵敏度、动态点放大和经典几何退化的解析解释；

**🔧 技术方法**

使用了左平移化的 SE(3) 计算、曲率算子、谱分解、影响函数、Fisher 信息等理论工具；

**📊 数据集**

未给出具体数据集，主要以理论推导与有限样本稳定性分析为主；

**📈 对比分析**

比较方法主要是理论与定理的证明，未做实验验证，性能以理论上对极值、稳定性误差界定等方式展示；

**⚠️ 局限性**

局限在于仅为局部线性分析，对极端离群点、非光滑残差、深度学习先验等情况缺乏完整处理，且需在实际系统中实现阈值设定与工程细节。

---

## 368. Uncovering Residual Factors in Financial Time Series via PCA and MTP2-constrained Gaussian Graphical Models

**arXiv ID:** 2602.05580 | [PDF](https://arxiv.org/pdf/2602.05580v1)

**作者:** Koshi Watanabe `[一作]` (Hokkaido University), Masanori Hirano `[通讯]` (Preferred Networks Inc.)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种层次化的残差因子提取方法，通过先 PCA 去除主因子，再使用 MTP_2 限制的 Gaussian 图模型进一步去除残余共因子。

**💡 创新点**

创新点在于将 PCA 与 MTP_2 GGM 结合，利用正相关性假设（所有偏相关均非负）来进一步正交化残差因子，并提供理论证明。

**🔧 技术方法**

使用 PCA、信息准则、GGM、MTP_2 约束、梯度投影及 Dykstra 投影等技术。

**📊 数据集**

在 S&P 500 和 TOPIX 500 两个股票指数的日收益数据上进行实验。

**📈 对比分析**

与 ICA、白化、收缩白化、单纯 PCA 等方法比较，使用交叉资产相关系数（L1、L2均值）以及对冲交易回测的夏普率、最大回撤、CVaR 等指标，结果表明所提方法在正交度和交易表现上均优于对照组。

**⚠️ 局限性**

局限包括假设协方差满足 MTP_2 约束，未考虑时序依赖性，计算量随资产数增大仍需改进，且在极端市场条件下性能未知。

---

## 369. LoGoSeg: Integrating Local and Global Features for Open-Vocabulary Semantic Segmentation

**arXiv ID:** 2602.05578 | [PDF](https://arxiv.org/pdf/2602.05578v1)

**作者:** Junyang Chen `[一作]` (Southeast University), Yiguo Qiao `[通讯]` (Southeast University)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5053918847)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了单阶段开词汇语义分割框架LoGoSeg，能够在未知类别上进行像素级预测。

**💡 创新点**

创新点包括结合对象存在先验、区域对齐和双流融合，显著降低类别混淆并提升像素级对齐精度。

**🔧 技术方法**

采用CLIP视觉语言模型、区域先验估计、区域级文本指导、方向注意力与状态空间建模双流融合以及可学习查询的Transformer解码器。

**📊 数据集**

在COCO‑Stuff上训练，并在ADE20K、PASCAL VOC、PASCAL‑Context的六个基准（A‑847/150、PC‑459/59、PAS‑20、PAS‑20b）进行评测。

**📈 对比分析**

与现有两阶段与单阶段方法对比，LoGoSeg在六大基准上实现了95%以上mIoU，多次名列第一或第二，性能优于最新对手。

**⚠️ 局限性**

局限在于对大规模VLM后端的依赖，推理速度受限，难以满足资源受限场景的实时部署需求。

---

## 370. LocateEdit-Bench: A Benchmark for Instruction-Based Editing Localization

**arXiv ID:** 2602.05577 | [PDF](https://arxiv.org/pdf/2602.05577v1)

**作者:** Shiyu Wu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Yequan Wang `[通讯]` (Peking University)

**通讯引用:** 2788 | [OpenAlex ID](https://openalex.org/A5054437932)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了LocateEdit-Bench数据集，并在该基准上评估了多种图像伪造定位方法。

**💡 创新点**

创新点在于首次构建大规模、针对指令式编辑的定位基准，并设计多模型泛化与降质鲁棒性评测协议。

**🔧 技术方法**

使用四种先进的指令式编辑模型（Qwen-Image-Edit、Flux-Kontext、Step1X-Edit、BAGEL）、SAM3语义分割、LLM/VLM筛选指令，以及多维度评估指标。

**📊 数据集**

数据集基于OmniEdit图像，包含231K张经过四种编辑器生成的添加、替换、属性修改三类编辑样本，并附有高质量掩码。

**📈 对比分析**

通过对ObjectFormer、PSCC-Net、IML-ViT、PIM-Net、Mesorch、CLIP、SegFormer、DINOv3等方法进行两种评测协议，发现同编辑器下表现良好，但跨编辑器泛化显著下降，SegFormer在整体性能上最优。

**⚠️ 局限性**

局限在于现有定位方法对指令式编辑难以捕捉，跨编辑器泛化不足，且对JPEG压缩和高斯模糊等降质极易失效，亟需开发更具鲁棒性和编辑器无关的特征。

---

## 371. Unveiling Implicit Advantage Symmetry: Why GRPO Struggles with Exploration and Difficulty Adaptation

**arXiv ID:** 2602.05548 | [PDF](https://arxiv.org/pdf/2602.05548v1)

**作者:** Zhiqi Yu `[一作]`, Liangqiong Qu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对传统 GRPO 框架中的 Group Relative Advantage Estimation (GRAE) 进行理论分析，发现其存在优势对称性导致探索不足和难度适配失效，进而提出 Asymmetric GRAE (A-GRAE) 以动态打破组层与样本层的优势对称性，提升大语言模型与多模态语言模型的推理性能。

**💡 创新点**

① 证明 GRPO 的优势对称性会抑制对未采样正确路径的探索；② 发现样本层对称性导致对中等难度样本的过度关注；③ 提出基于训练状态的动态权重分配，分别在组层抑制正轨迹优势、在样本层按难易度加权，从而实现自适应探索与难度衔接。

**🔧 技术方法**

采用强化学习可验证奖励 (RLVR) 框架中的 GRPO；对 GRAE 进行数学证明；实现 A‑GRAE 并嵌入 GRPO、DAPO、Dr.GRPO 等变体；利用熵、Pass@k 等指标评估；通过与 W‑REINFORCE、GRPO‑LEAD 等方法对比验证效果。

**📊 数据集**

七大基准：文本数学推理（MATH、AMC23、AIME2025）以及多模态数学与医学推理（Geo3k、MathVision、MathVerse、HuatuoGPT‑Vision），涵盖单模态与多模态多任务场景。

**📈 对比分析**

与原始 GRPO、GRPO‑LEAD、W‑REINFORCE、DAPO、Dr.GRPO 等方法在 Pass@1 与 Pass@k 上进行统一评测。A‑GRAE 在所有评测中均实现显著提升，平均 Pass@1 提升 3–5个百分点，Pass@k 在大采样预算下提升 1–2个百分点；在多模态任务中既提升 ID 又提升 OOD，证明通用性。

**⚠️ 局限性**

① 需要手工调节动态权重参数，过度抑制正轨迹可能导致训练不稳定；② 对极难样本的收敛速度仍相对慢；③ 仅在基于 RLVR 的 GRPO 框架验证，跨其他 RL 方法的通用性待进一步验证；④ 对超大规模模型和不同奖励形式的适用性尚待评估。

---

## 372. When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging

**arXiv ID:** 2602.05536 | [PDF](https://arxiv.org/pdf/2602.05536v1)

**作者:** Yayuan Li `[一作]` (Nanjing University), Yinghuan Shi `[通讯]` (Nanjing University)

**通讯引用:** 4860 | [OpenAlex ID](https://openalex.org/A5055917015)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了模型融合中出现的谱过度计数问题，并提出了一种无训练、无数据的后处理方法——奇异值校准（SVC），以纠正融合模型谱的不平衡；

**💡 创新点**

创新点在于通过投影到融合模型的列空间子空间来量化跨任务的谱重叠，并据此按子空间缩放奇异值，从而消除共享方向的过度累积；

**🔧 技术方法**

技术主要包括奇异值分解（SVD）、子空间投影、基于投影系数的奇异值缩放，以及可选的任务偏好优化；

**📊 数据集**

实验使用了多种视觉任务（8/14 影像分类基准）和自然语言处理基准（11 语言分类任务、两大LLM生成基准），并在ViT和LLaMA2等模型上验证；

**📈 对比分析**

在所有基准上，SVC在不改变方向的前提下均显著提升融合性能（例如 Task Arithmetic 提升约13%），并在视觉、语言任务中刷新了SOTA；

**⚠️ 局限性**

局限性包括需要对每个层进行SVD导致的计算开销，且对右奇异向量的校准效果不佳，且在某些任务间极大域差异时，单任务偏好校准可能导致其他任务性能下降。

---

## 373. Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification

**arXiv ID:** 2602.05535 | [PDF](https://arxiv.org/pdf/2602.05535v1)

**作者:** Tao Huang `[一作]` (Beijing Jiaotong University), Liping Jing `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3507 | [OpenAlex ID](https://openalex.org/A5069749738)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于证据理论的 Evidential Uncertainty Quantification (EUQ) 方法，用单前向推理从 LVLM 输出层预激活特征中提取冲突 (CF) 与无知 (IG) 指标，来检测幻觉、越狱、对抗和 OOD 等四类误行为。

**💡 创新点**

创新点包括：① 将线性投影视为证据融合器，直接在单前向推理中计算 CF 与 IG；② 在多层解码器中跟踪不确定性动态；③ 通过 CF 与 IG 区分不同误行为，显著提升检测性能。

**🔧 技术方法**

技术手段：Dempster–Shafer 证据理论、基本信念分配（BBA）、Dempster 合并规则、最小承诺原则、预激活特征映射、单前向推理和不确定性度量。

**📊 数据集**

实验使用的 LVLM 有 DeepSeek‑VL2‑Tiny、Qwen2.5‑VL‑7B、InternVL2.5‑8B、MoF‑Models‑7B；数据集包括 POPE、R‑Bench（幻觉）、FigStep、Hades、VisualAdv（越狱）、ANDA、PGN（对抗）、AI‑Secure/MMDecodingTrust‑I2T（OOD）。

**📈 对比分析**

与四类基线（self‑consistency、semantic entropy、predictive entropy、length‑normalized PE、HiddenDetect）对比，EUQ 的 CF 与 IG 在 AUROC 与 AUPR 上平均提升约 10.4%/7.5% 与 5.3%/5.5%，在幻觉检测中 CF 最优，在 OOD 检测中 IG 最优，且仅需一次前向推理，计算效率远高于采样方法。

**⚠️ 局限性**

局限性：需要访问模型内部预激活特征，无法直接应用于封闭 API；对中等规模模型误行为的检测仍有挑战；目前仅针对具有线性投影的解码器，扩展到更广泛架构或黑盒设置仍需进一步研究。

---

## 374. Split Personality Training: Revealing Latent Knowledge Through Alternate Personalities

**arXiv ID:** 2602.05532 | [PDF](https://arxiv.org/pdf/2602.05532v1)

**作者:** Florian Dietz `[一作]` (Spoken Language Systems), Dietrich Klakow `[通讯]` (Spoken Language Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

引入Split Personality Training（SPT），在LoRA适配器中训练“诚实人格”，实现模型生成后访问隐藏内部状态进行自我审计。

**💡 创新点**

通过在LoRA适配器中分离主模型与审计人格，做到零对主模型能力的影响，能够揭示被训练隐藏的奖励劫持等隐性偏差。

**🔧 技术方法**

使用LoRA参数高效微调、激活触发字符串、LoRA掩码技术、合成训练数据生成与外部模型评判、以及Anthropic Auditing Game Model Organism基准。

**📊 数据集**

采用28,321条合成训练样本（涵盖11类对齐失误）以及Anthropic Auditing Game Model Organism的评估数据。

**📈 对比分析**

与未微调的Llama‑3.3‑70B基线及线性探针进行对比，SPT在奖励劫持检测上达96%准确率，特异性高，且在多模型与多主题上表现优于探针。

**⚠️ 局限性**

仅在单轮英文对话中验证，难以系统评估多轮或非英语场景；部分失误类型（如造假统计）检测效果差；诚实人格在被直接针对的 jailbreak 攻击下仍易被突破，且训练依赖外部模型生成的数据。

---

## 375. AI Agent Systems for Supply Chains: Structured Decision Prompts and Memory Retrieval

**arXiv ID:** 2602.05524 | [PDF](https://arxiv.org/pdf/2602.05524v1)

**作者:** Konosuke Yoshizato `[一作]` (National Institute of Advanced Industrial Science and Technology), Takanobu Otsuka `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并实现了一种基于大型语言模型（LLM）的多智能体系统（MAS）用于多层供应链库存管理，并通过引入历史经验检索与记忆机制提升其适应性与性能。

**💡 创新点**

1)首次验证LLM在多层库存管理中可通过安全库存策略实现最优订单决策；2)提出基于相似经验检索的记忆模块，使LLM MAS在多种供应链场景下保持高性能，性能与传统强化学习方法相当。

**🔧 技术方法**

使用OpenAI o4-mini LLM作为决策引擎，结合安全库存公式、相似度匹配（欧氏距离）和向量数据库存储历史（状态、订单、收益）实现记忆检索；同时对比使用GPT‑4.1、Heuristic（基准库存策略）以及基于CTDE的PPO/MPPO强化学习。

**📊 数据集**

采用自定义的五种供应链实验场景（均匀/多样化参数与三种需求曲线），不使用公开数据集，而是自行构造的基准库存管理问题。

**📈 对比分析**

通过在每个场景下进行5次试验，比较不同模型的平均总奖励与相对最优性缺口Δ，结果显示：记忆+RL日志（w/ RL log）在多数场景中均取得最高或接近最优的奖励；安全库存策略在恒定需求场景下表现最佳；相较于传统Heuristic和RL基准，LLM‑MAS表现相当甚至更优。

**⚠️ 局限性**

1)相似度匹配仅使用固定欧氏距离与阈值，缺乏自适应性；2)在部分情境下LLM可能忽略检索案例而直接输出结果，导致过度“过度思考”现象；3)实验仅限确定性需求，未验证在随机需求环境下的鲁棒性。

---

## 376. ArkTS-CodeSearch: A Open-Source ArkTS Dataset for Code Retrieval

**arXiv ID:** 2602.05550 | [PDF](https://arxiv.org/pdf/2602.05550v1)

**作者:** Yulong He `[一作]` (St. Petersburg State University), Dmitry Shalymov `[通讯]` (St. Petersburg State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了ArkTS语言的代码检索数据集（ArkTS-CodeSearch），并基于该数据集评估并微调多种开源代码嵌入模型。

**💡 创新点**

首次公开ArkTS的高质量docstring–代码对数据集与基准，结合AST解析与跨平台去重，使用Fine‑tuning显著提升检索性能。

**🔧 技术方法**

采用Tree-sitter-arkts进行AST解析、CodeSearchNet框架构建检索任务、SentenceTransformer加多负样本排序损失进行监督式对比学习。

**📊 数据集**

从GitHub和Gitee抓取的1,577个ArkTS仓库，得到24,452个函数-注释对（12,792来自Gitee，11,660来自GitHub）。

**📈 对比分析**

使用MRR、NDCG@5和Recall@K对比，零样本中Qwen3取得最高（MRR 0.6776），Fine‑tuned embeddinggemma_arkts在测试集上达到MRR 0.7788、Recall@5 0.8769，显著优于其他模型。

**⚠️ 局限性**

仅针对单一docstring–代码检索任务、受限于注释质量与覆盖范围、缺乏结构化（AST/图）模型支持，数据规模与多语言多任务仍有提升空间。

---

## 377. ADCA: Attention-Driven Multi-Party Collusion Attack in Federated Self-Supervised Learning

**arXiv ID:** 2602.05612 | [PDF](https://arxiv.org/pdf/2602.05612v1)

**作者:** Jiayao Wang `[一作]` (Yangzhou University), Dongfang Zhao `[通讯]` (University of Washington)

**通讯引用:** 1328 | [OpenAlex ID](https://openalex.org/A5101671477)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于注意力驱动的多方共谋攻击（ADCA），利用分布式触发器和客户端间动态聚合来隐蔽并增强联邦自监督学习中的后门攻击。

**💡 创新点**

创新点在于：1）将全局触发器拆解为多种局部模式以提升隐蔽性；2）在恶意联盟内采用注意力机制动态加权聚合模型，显著降低后门信号在全局聚合中的稀释。

**🔧 技术方法**

采用对比学习（SimCLR）作为自监督框架，使用卷积核权重再加权和多层感知机的通道注意力等技术实现动态聚合。

**📊 数据集**

在四个公开图像数据集上评估：CIFAR-10、CIFAR-100、STL-10、GTSRB。

**📈 对比分析**

与现有基线（DBA、FCBA、UBA、BadEncoder、BADFSS）以及多种自监督算法（SimCLR、MoCo、BYOL、SimSiam）对比，ADCA在攻击成功率（ASR）上平均提升约30%–40%，并保持高分类准确率（ACC），在不同数据分布、客户端比例及防御策略下表现稳健。

**⚠️ 局限性**

局限性包括：1）攻击仍依赖一定比例的恶意客户端参与；2）在极端非IID或极大客户端规模时后门效果可能略有下降；3）针对未来更强的全局聚合防御（如更复杂的异常检测）尚未验证。

---

## 378. Multi-instance robust fitting for non-classical geometric models

**arXiv ID:** 2602.05602 | [PDF](https://arxiv.org/pdf/2602.05602v1)

**作者:** Zongliang Zhang `[一作]` (Jimei University), Zongyue Wang `[通讯]` (Jimei University)

**通讯引用:** 549 | [OpenAlex ID](https://openalex.org/A5081721333)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了如何从含噪声数据中鲁棒地拟合多实例的非经典几何模型。

**💡 创新点**

提出了一种基于最近邻数据点正则化的NPRE估计器，能够在不设阈值的情况下解决重叠计数问题，并对多实例进行有效优化。

**🔧 技术方法**

使用NPRE估计器配合烏鴉搜索（Cuckoo Search）元启发式算法进行模型参数的全局优化。

**📊 数据集**

在合成噪声线条、程序化字符以及3D高速公路曲线（含合成与真实激光扫描点云）等数据集上进行了实验验证。

**📈 对比分析**

与最先进的经典模型拟合方法及之前的非经典模型拟合方法比较，实验结果显示本方法在鲁棒性、精度和多实例恢复能力上均优于或不逊于现有方法，能够成功恢复多条曲线和完整字符。

**⚠️ 局限性**

主要局限是计算效率相对较低，需要大量迭代，且对参数维度较高的复杂模型实时性不足。

---

## 379. Emulating Aggregate Human Choice Behavior and Biases with GPT Conversational Agents

**arXiv ID:** 2602.05597 | [PDF](https://arxiv.org/pdf/2602.05597v1)

**作者:** Stephen Pilli `[一作]` (University), Vivek Nallur `[通讯]` (University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在人机对话实验中引入先前对话的复杂性，评估大型语言模型（LLM）在模拟人类决策时对状态守恒偏差（Status Quo Bias）的再现能力，并与人类受试者的行为进行比较。

**💡 创新点**

创新点在于：①首次在对话环境下对个体级别的偏差预测进行量化评估；②系统性地比较不同人类模仿提示（HL1–HL3）对LLM行为的影响；③使用先前对话的认知负荷来检验偏差与情境交互的动态；④通过模型精确度、Cohen’s h及平均绝对差等指标全面衡量LLM的逼真度。

**🔧 技术方法**

技术与方法包括：使用GPT‑4.1、GPT‑4.1‑mini、GPT‑5、GPT‑5‑mini（OpenAI API）构建聊天代理；Streamlit实现实验界面；NASA‑TLX、回忆准确率等指标评估认知负荷；使用GLMM对偏差效应进行统计建模；对模型输出进行人工注释与评估。

**📊 数据集**

数据集：人类实验数据（N≈1100，来自Prolific），包含三类决策场景（预算分配、投资组合、大学岗位）以及三种先前对话（简单/复杂）来自Schema‑Guided Dialogue（SGD）数据集；LLM模拟数据对应每位受试者的对话记录与决策场景。

**📈 对比分析**

比较方法：将LLM预测结果与人类实际选择进行精确度（precision）比较，并通过Cohen’s h和MD（平均绝对差）衡量效应大小匹配度。结果显示：HL1/HL2下精确度可达1.00，HL3约0.67；个体预测精确度约68%；GPT‑4.1‑mini在所有指标上表现最优，尤其在捕捉对话复杂性与偏差交互方面表现突出。

**⚠️ 局限性**

局限性：①仅研究了状态守恒偏差，未覆盖其他典型偏差；②决策场景抽象、文本化，缺乏真实情境的多模态交互；③先前对话与决策场景属不同领域，可能降低自然性；④仅使用自我报告与行为指标评估认知负荷，未加入生理测量；⑤LLM在高阶提示下可能出现过度对齐，导致偏差夸大。

---

## 380. EdgeMask-DG*: Learning Domain-Invariant Graph Structures via Adversarial Edge Masking

**arXiv ID:** 2602.05571 | [PDF](https://arxiv.org/pdf/2602.05571v1)

**作者:** Rishabh Bhattacharya `[一作]` (International Institute of Information Technology Hyderabad), Naresh Manwani `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 EdgeMask-DG* 方案，利用对抗性稀疏边掩码在增强图结构上训练 GAT，提升图域迁移下的结构不变性

**💡 创新点**

创新点在于：①将原始拓扑与基于特征的 kNN 与谱聚类边融合成增强图；②采用极小极大对抗掩码学习，自动筛选对模型最具挑战性的稀疏子图；③从鲁棒优化角度理论解释对抗掩码的稀疏性约束与域不变性的关系

**🔧 技术方法**

技术手段包括：图注意力网络（GAT）作为任务网络；轻量级 MaskNet（投影层+MLP）生成连续掩码；kNN 与谱聚类边构造；对抗性 min‑max 训练（交替更新任务网络与掩码网络）；稀疏正则 λ 控制掩码密度；PyTorch‑Geometric实现；鲁棒优化理论证明；实验中还使用多源域交叉验证与留一域评估

**📊 数据集**

使用多种图域迁移基准：citation 网络（ACM、DBLP、Citation）、Cora、Amazon‑Photo、Facebook‑100、Twitch、Elliptic、OGB‑Arxiv 等，覆盖结构、社交、时间与人工诱导的域偏移场景

**📈 对比分析**

与 ERM、EERM、MMD、DRNN、LiSA、GRM、TRACI、GraphAug、MARIO 等多种 GNN 与域泛化方法对比；在 citation 任务上实现最高平均 F1（73.81%）并在两项子任务中领先；在其它基准上亦常获得最优或接近最优成绩，尤其在 Cora、Photo、Twitch 上显著提升；实验通过多种指标（Accuracy、Micro/Macro‑F1、ROC‑AUC）验证其鲁棒性

**⚠️ 局限性**

局限性：①对特征稳定性的假设在 FB‑100 等高度结构化数据中效果不佳；②kNN 与谱聚类的预计算及稠密化会导致大图上 O(N^3) 复杂度，需进一步优化；③对动态/时序图的适配仍有限，需探索更高效的时间增强与对抗策略

---

## 381. MAGPrompt: Message-Adaptive Graph Prompt Tuning for Graph Neural Networks

**arXiv ID:** 2602.05567 | [PDF](https://arxiv.org/pdf/2602.05567v1)

**作者:** Long D. Nguyen `[一作]` (Victoria University of Wellington), Binh P. Nguyen `[通讯]` (Victoria University of Wellington)

**通讯引用:** 2766 | [OpenAlex ID](https://openalex.org/A5091142923)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于消息传递的图提示调优框架 MAGPrompt，用以在保持预训练 GNN 参数冻结的前提下，利用可学习的门控和提示向量对邻居消息进行重加权和注入，从而实现任务特定的邻域交互调节。

**💡 创新点**

创新点在于将提示注入扩展到消息级别，不仅重新加权每条邻居消息，还通过可组合的边级提示向量实现细粒度的邻域信息调节，解决了传统图提示方法仅修改输入或表示而无法控制邻居权重的问题。

**🔧 技术方法**

技术主要包括：基于注意力的门控机制、加权重的消息重构、边级提示基底组合以及提示坍塌正则化；实现上采用 GCN/GIN 等通用 GNN 作为骨干，保持其参数不变，只训练少量提示参数。

**📊 数据集**

在节点分类上使用 Cora、CiteSeer、Pubmed、ogbn-arxiv、Flickr；在图分类上使用 TUDataset 的 ENZYMES、DD、NCI1、NCI109、Mutagenicity 以及 MoleculeNet 的 BACE、BBBP、SIDER、ClinTox；并对多种预训练策略（如 GraphCL、SimGRACE、DGI、AttrMasking 等）进行评估。

**📈 对比分析**

与 GPPT、GraphPrompt、EdgePrompt、EdgePrompt+、ALL-in-One 等现有图提示方法及线性探针、全微调进行对比，MAGPrompt+ 在 5-shot/50-shot 低样本设置下平均提升 4-5% 的准确率，并在全样本任务中与全微调相当或略优；实验表明其在多种预训练目标下均保持鲁棒性。

**⚠️ 局限性**

局限性包括：提示坍塌问题需额外正则化；对超参数（如提示基数 M、门控参数 β、注意力头数 d_a）敏感，需要手工调优；在极大规模图或极稠密图上仍可能面临计算和内存瓶颈。

---

## 382. IndustryShapes: An RGB-D Benchmark dataset for 6D object pose estimation of industrial assembly components and tools

**arXiv ID:** 2602.05555 | [PDF](https://arxiv.org/pdf/2602.05555v1)

**作者:** Panagiotis Sapoutzoglou `[一作]` (National Technical University of Athens), Maria Pateraki `[通讯]` (National Technical University of Athens)

**通讯引用:** 1054 | [OpenAlex ID](https://openalex.org/A5047139078)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个新的工业工具和零件的 RGB‑D 数据集 IndustryShapes，并提供了经典集和扩展集，支持实例级和新对象 6D 位置估计。

**💡 创新点**

创新点在于提供具有挑战性的工业零件（无纹理、金属、对称、薄壁），真实工业环境捕获，并首次加入 RGB‑D 静态上线序列支持无模型姿态估计。

**🔧 技术方法**

使用 Intel RealSense D455/D405 摄像头采集，结合 OpenGL 渲染生成合成图像，采用 ArUco+SfM 标注，基于 BOP 框架进行评估，并用 EPOS、ZebraPose、DOPE、FoundPose、FoundationPose、CNOS、SAM‑6D 等现有方法做基线。

**📊 数据集**

数据集本身 IndustryShapes，包括 5 种工业对象，经典集 4623 图像约 6k 姿态标注，扩展集 10 个上线序列约 6.3k 帧和 10.3k 标注，合计 22.4k 图像。

**📈 对比分析**

在经典集上实例级方法 AR 最高约 0.5；在扩展集上新对象方法 FoundationPose (模型基) AR 0.69，模型自由 0.33；检测/分割 mAP 约 0.27/0.35。整体表现低于常见室内数据，表明挑战性大。

**⚠️ 局限性**

局限在于训练集复杂度不均衡、测试帧数不齐、对象与背景差异大导致泛化差；缺少更多视角覆盖和多物体训练；方法对反射/薄壁物体鲁棒性差。

---

## 383. VLN-Pilot: Large Vision-Language Model as an Autonomous Indoor Drone Operator

**arXiv ID:** 2602.05552 | [PDF](https://arxiv.org/pdf/2602.05552v1)

**作者:** Bessie Dominguez-Dager `[一作]` (University Institute for Compute Research), Miguel Cazorla `[通讯]` (University Institute for Compute Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VLN-Pilot 框架，将大型视觉语言模型（VLLM）作为无人机的自主导航与操作员，完成从自然语言指令到视觉感知再到飞行决策的闭环控制。

**💡 创新点**

创新点在于将 VLLM 与有限状态机（FSM）结合，实现高层语义决策与低层运动控制的无缝衔接，并通过多模态推理实现室内无人机的自主任务执行。

**🔧 技术方法**

使用的技术包括 GPT‑4.1 与 Gemini‑2.5‑flash 作为 VLLM、Unity‑based DJI Tello 模拟器、ML‑Agents、Python 控制器以及 FSM 逻辑；模型通过结构化提示与视觉输入交互，产生高层移动指令。

**📊 数据集**

数据集为自建的 Photorealistic 室内仿真环境（Furnished Cabin）以及对应的房间拓扑图和目标物体位置，未采用公开的 VLN 数据集。

**📈 对比分析**

通过在三种起始点进行 5 次实验，比较 GPT 与 Gemini 在目标达成率、碰撞次数和最大步数等指标；实验表明 GPT 在多数任务中实现了 80%–100% 的成功率、碰撞更少，且步数更少；Gemini 在门口对齐时易出现振荡导致步数上限被触发。

**⚠️ 局限性**

局限性包括 VLLM 缺乏对无人机尺寸与空间的体积感知，导致在门口对齐时碰撞；Gemini 对空间关系的解读更为严格，易产生振荡；两模型均需进一步加入几何约束与物理尺寸信息以提升安全性和鲁棒性。

---

## 384. Multi-Task GRPO: Reliable LLM Reasoning Across Tasks

**arXiv ID:** 2602.05547 | [PDF](https://arxiv.org/pdf/2602.05547v1)

**作者:** Shyam Sundhar Ramesh `[一作]` (University College London), Ilija Bogunovic `[通讯]` (University of Basel)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的多任务GRPO（MT-GRPO）算法，旨在通过动态调整任务权重来优化最差任务的性能，并促进任务间的平衡进展。

**💡 创新点**

创新点在于引入了改进感知的任务重加权机制和比例保持采样器，以确保任务权重能够有效反映在梯度贡献上，从而提高最差任务的准确性。

**🔧 技术方法**

使用了基于强化学习的后训练方法，结合了任务级奖励和任务改进信号来进行任务重加权，并采用了比例保持的批量构建机制。

**📊 数据集**

使用了ReasoningGym框架生成的多任务推理基准数据集，包括Countdown、Zebra和ARC等任务，涵盖了不同难度级别的实例。

**📈 对比分析**

与标准GRPO和DAPO等基线方法相比，MT-GRPO在最差任务的准确性上提高了16-28%，并且在3任务设置中需要50%更少的训练步骤来达到50%的最差任务准确性，显示出显著的效率提升。

**⚠️ 局限性**

限制在于标准GRPO在多任务设置中可能导致任务间的负迁移和干扰，且不同任务的零梯度率差异可能影响训练效果。

---

## 385. A Comparative Study of 3D Person Detection: Sensor Modalities and Robustness in Diverse Indoor and Outdoor Environments

**arXiv ID:** 2602.05538 | [PDF](https://arxiv.org/pdf/2602.05538v1)

**作者:** Malaz Tamim `[一作]` (Fraunhofer Institute for Cognitive Systems IKS), Karsten Roscher `[通讯]` (Fraunhofer Institute for Cognitive Systems IKS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在JRDB数据集上，对3D人体检测进行了系统评估，比较了仅摄像头、仅激光雷达以及摄像头-雷达融合三种感知模态。

**💡 创新点**

首次将这些模态在多种室内外环境、不同遮挡、距离和传感器噪声条件下进行统一对比，并揭示融合模型在鲁棒性与准确性上的优势与局限。

**🔧 技术方法**

采用BEVDepth（摄像头）、PointPillars（激光雷达）和DAL（融合）三种代表性网络，并通过合成噪声、失配等人为扰动评估鲁棒性。

**📊 数据集**

使用JRDB 2022官方拆分（训练 21704、验证 6189、测试 27661）进行训练与评测。

**📈 对比分析**

通过AP_0.3/AP_0.5 指标在不同距离、遮挡和噪声等级下对比，发现融合模型DAL在AP_0.3上最高达73.18%，相较单模态提升显著；在AP_0.5上也最高为24.73%，但整体性能仍低于工业级模型。

**⚠️ 局限性**

DAL对激光雷达时序失配及空间失配仍敏感，单模态模型在遮挡和远距离下表现差；同时AP_0.5评估对鲁棒性敏感度不高，缺乏更细粒度的误差分析。

---

## 386. Generalization of Self-Supervised Vision Transformers for Protein Localization Across Microscopy Domains

**arXiv ID:** 2602.05527 | [PDF](https://arxiv.org/pdf/2602.05527v1)

**作者:** Ben Isselmann `[一作]` (Hochschule Darmstadt), Andreas Weinmann `[通讯]` (Technische Hochschule Würzburg-Schweinfurt)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估 DINO 预训练的 Vision Transformer 在不同数据集上的迁移性能，并在 OpenCell 蛋白定位任务中验证其泛化能力。

**💡 创新点**

创新点在于系统对比自然图像（ImageNet-1k）、HPA 微观图像和目标数据（OpenCell）的预训练权重，并探索通道映射与通道复制两种嵌入策略，证明大规模领域相关预训练可跨域迁移。

**🔧 技术方法**

使用自监督学习框架 DINO、Vision Transformer 编码器、通道映射/复制嵌入、MLP 分类头，以及 5 折交叉验证评估。

**📊 数据集**

使用的数据集包括 ImageNet-1k、Human Protein Atlas（HPA）四通道图像、OpenCell 两通道图像。

**📈 对比分析**

在 OpenCell 上通过宏 F1 分数比较不同预训练配置的表现：HPA 预训练 + 通道映射获得最高 0.8221，ImageNet-1k 0.8057，OpenCell 自训 0.7918；通道复制性能略逊。

**⚠️ 局限性**

局限性包括仅采用单一 MLP 分类头、未深入探讨多任务或微调策略、通道映射对通道语义不匹配的鲁棒性有限，以及存在轻微过拟合风险。

---

## 387. Assessing Problem-Solving in HR Contexts: A Comparison Between Game-Based and Self-Report Measures

**arXiv ID:** 2602.05525 | [PDF](https://arxiv.org/pdf/2602.05525v1)

**作者:** Fabrizio Fornari `[一作]` (University of Camerino), Luigi Caputo `[通讯]` (Emaze Gaming Società Benefit SRL)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过比较传统自评工具PSI-B与游戏化评估Behaveme-PS，探讨问题解决能力的自我感知与行为表现的关系。

**💡 创新点**

首次实证对比两种方法，揭示自评与行为评估提供互补信息而非等价测量，指出在人才选拔中需多元评估。

**🔧 技术方法**

采用Behaveme-PS游戏化平台进行5分钟问题解决任务，采集游戏行为指标，并结合PSI-B自评问卷。

**📊 数据集**

78名计算机科学专业学生（最终72人完整数据）。

**📈 对比分析**

使用分四分位的自评分组与游戏水平四分级对照，计算斯皮尔曼相关，结果无显著关联（ρ=-0.09，p=0.45）。

**⚠️ 局限性**

样本单一、以学生为主，缺乏工作场景验证，未考虑动机、自我效能等潜在调节变量。

---

## 388. Limitations of SGD for Multi-Index Models Beyond Statistical Queries

**arXiv ID:** 2602.05704 | [PDF](https://arxiv.org/pdf/2602.05704v1)

**作者:** Daniel Barzilai `[一作]` (Weizmann Institute of Science), Ohad Shamir `[通讯]` (Weizmann Institute of Science)

**通讯引用:** 7919 | [OpenAlex ID](https://openalex.org/A5109213302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种新框架，直接研究标准 SGD 在高维单索引和多索引模型中的学习难度，并给出严格的下界。

**💡 创新点**

创新点在于摆脱 SQ 方案的噪声假设，利用梯度条件数与子空间对齐的理论，得到关于信息指数的本质下界。

**🔧 技术方法**

主要技术是对 SGD 噪声建模为随机游走、引入梯度条件数、使用 Hermite 多项式展开和对齐分析。

**📊 数据集**

研究主要是理论推导，并未使用具体实验数据集；示例以标准正态分布和均匀立方体为输入分布。

**📈 对比分析**

与传统 SQ 下界和球面 SGD 结果对比，本文证明在标准 SGD 下仍需 Ω̃(d^{max(k_*−1,1)}) 步骤；实验上未给出，但理论上与已知上界相符。

**⚠️ 局限性**

局限在于需假设梯度条件数不随维度增长、输入满足子高斯假设，且对更一般的网络结构和非平稳梯度难以直接估计。

---

## 389. HyperPotter: Spell the Charm of High-Order Interactions in Audio Deepfake Detection

**arXiv ID:** 2602.05670 | [PDF](https://arxiv.org/pdf/2602.05670v1)

**作者:** Qing Wen `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35154 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于高阶交互的音频深度伪造检测框架HyperPotter，利用超图与原型引导的高阶关系建模。

**💡 创新点**

创新点在于首次引入O-信息分析确认协同高阶交互对检测关键，并通过超图与原型池实现高阶关系的聚类与强化。

**🔧 技术方法**

使用超图注意力层（HAGNN）、模糊C均值聚类、原型引导初始化、关系增强模块以及预训练的Wav2Vec2+AASIST编码器。

**📊 数据集**

在13个公开数据集（ASVspoof2019/2021、ASVspoof2020、FoR、Codecfake、ADD2022/2023、Libri、SONAR等）上评估。

**📈 对比分析**

与多种SOTA同等规模模型对比，平均相对提升15.3%，在多跨域、跨说话人、跨攻击的测试集上取得最低EER，显著优于传统双向或图模型。

**⚠️ 局限性**

局限在于对强信道失真和压缩编码的鲁棒性仍不足，且超图构造及原型更新带来额外计算与调参成本。

---

## 390. Stable but Wrong: When More Data Degrades Scientific Conclusions

**arXiv ID:** 2602.05668 | [PDF](https://arxiv.org/pdf/2602.05668v1)

**作者:** Zhipeng Zhang `[一作]` (China Mobile Research Institute), Kai Li `[通讯]` (China Mobile Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在观测可靠性不可观测漂移下，常规推断程序即使收敛稳定，也会系统性偏离真值，且误差随数据量增加而放大。

**💡 创新点**

发现并证明了在不可观测漂移结构下，稳定收敛与置信度提升不一定意味着正确，提出了“可观测性不确定性陷阱”概念。

**🔧 技术方法**

使用序列贝叶斯推断、线性与随机游走漂移模型、对数线性更新、残差统计、滑动窗口基线等技术。

**📊 数据集**

主要用合成数据（θ*=0，噪声 N(0,1)，漂移为线性或随机游走）以及 SDSS Stripe82 星色数据验证。

**📈 对比分析**

与无漂移控制、滑动窗口估计等方法对比，发现无漂移时误差随样本增大下降，而有漂移时误差先下降后上升，显示传统诊断误导。

**⚠️ 局限性**

局限性在于实验以极简合成模型为主，未涵盖所有现实漂移情况，且仅讨论单参数场景，无法直接推广至多参数复杂模型。

---

## 391. Graph-based Agent Memory: Taxonomy, Techniques, and Applications

**arXiv ID:** 2602.05665 | [PDF](https://arxiv.org/pdf/2602.05665v1)

**作者:** Chang Yang `[一作]` (Hong Kong Polytechnic University), Xiao Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 45810 | [OpenAlex ID](https://openalex.org/A5073869073)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文综述并系统化整理了基于图结构的代理记忆（Graph‑Based Agent Memory）研究，包括记忆的分类、生命周期、技术实现、开源库与基准以及应用场景，构建了统一的标签和评估框架；

**💡 创新点**

创新点在于提出了记忆的多维分类与生命周期视角，将图记忆技术与评估基准系统化，强调了知识与经验、短期与长期、结构化与非结构化等维度，并为未来研究提供了可操作的标准与路线图；

**🔧 技术方法**

所述技术涵盖：LLM 作为抽取器（知识图谱、超图、层次结构、时序图等） + 向量/结构化检索、基于规则/强化学习/代理式检索、图推理与自我演化、主动探索与外部检索等；

**📊 数据集**

使用的数据集与基准主要包括多轮对话与跨会话记忆基准（LoCoMo、LongMemEval、MEMTRACK、StoryBench等）、Web 与环境交互基准（WebShop、WebArena、AgentGym 等）、长文本推理基准（LongBench、LongBench v2 等）、持续学习与终身学习基准（MemoryBench、Evo‑Memory 等），以及金融、推荐等特定领域数据；

**📈 对比分析**

对比方法：对多种图记忆库（如Mem0、OpenMemory、Cognee 等）与传统线性/向量存储进行检索准确率、对话一致性、执行成功率等指标评估，结果显示图记忆在多跳推理、长程记忆保持、适应性学习方面优于非图结构，且在资源消耗与解释性方面展现更佳可扩展性；

**⚠️ 局限性**

limitations包括：缺乏统一的、可重复的评估标准和基准；记忆更新与遗忘机制不完善；跨模态一致性与对齐难度大；实时性与计算成本仍高；对模型可解释性与因果推理支持不足；整体体系仍依赖人工规则与专家标注。

---

## 392. Alignment Verifiability in Large Language Models: Normative Indistinguishability under Behavioral Evaluation

**arXiv ID:** 2602.05656 | [PDF](https://arxiv.org/pdf/2602.05656v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (UNIR), Igor Santos-Grueiro (UNIR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将对大语言模型的行为评估正式化为可辨识性问题，证明在有限的行为评估和评估感知代理条件下，行为合规性无法唯一确定潜在的对齐属性。

**💡 创新点**

创新点在于提出Alignment Verifiability Problem与Normative Indistinguishability概念，并给出了一个严谨的不可辨识定理，阐明行为测试的固有限制。

**🔧 技术方法**

采用形式化模型、可辨识性理论、统计推断框架以及与黑盒测试、逆向奖励学习的类比等理论技术进行分析。

**📊 数据集**

本文为理论工作，不使用具体公开数据集，主要通过假设的交互历史和评估协议进行形式化讨论。

**📈 对比分析**

未进行实验比较，也未给出性能指标，核心贡献为理论证明而非算法或模型性能。

**⚠️ 局限性**

局限性包括仅适用于有限行为评估、评估感知存在、假设空间闭合等条件，未考虑内部解释、正式约束或多源证据等情形。

---

## 393. Groups and Inverse Semigroups in Lambda Calculus

**arXiv ID:** 2602.05654 | [PDF](https://arxiv.org/pdf/2602.05654v1)

**作者:** Antonio Bucciarelli `[一作]` (Paris Cité University), Antonino Salibra `[通讯]` (Paris Cité University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究 λ-理论下 λ-项的可逆性，证明有限和无限 hereditary permutations 在不同 λ-理论（λη、H*、H+）下构成逆半群，并利用最小群同余给出可逆项的完整描述。

**💡 创新点**

首次将逆半群结构与 λ-计算中的有限与无限 hereditary permutations 联系起来，提出新的自然偏序与最小群同余对应的 η-展开关系，从而统一了 λη、H*、H+ 等 λ-理论中的可逆项描述，验证了 Barendregt 的猜想。

**🔧 技术方法**

采用逆半群理论、共形树、Böhm 树、最小群同余、以及共价同伦方法，构造了 permutation trees 的逆半群并证明其 F‑inverse 性。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

无对比实验；通过代数同构与等价关系证明结果的正确性，并与先前仅针对 λη 的结果相比，扩展到更广 λ-理论并完成 Barendregt 预测的验证。

**⚠️ 局限性**

局限在于未完全解析 H+ 与 H* 之间更细粒度 λ-理论的可逆性，且对无限 permutation tree 的可构造性与计算复杂度仍待进一步研究。

---

## 394. Modelling the Morphology of Verbal Paradigms: A Case Study in the Tokenization of Turkish and Hebrew

**arXiv ID:** 2602.05648 | [PDF](https://arxiv.org/pdf/2602.05648v1)

**作者:** Giuseppe Samo `[一作]` (Idiap Research Institute), Paola Merlo `[通讯]` (Idiap Research Institute)

**通讯引用:** 1340 | [OpenAlex ID](https://openalex.org/A5102715103)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构造基于范式的Blackbird Language Matrices（BLM）任务，比较了单语和多语Transformer模型在土耳其语和现代希伯来语复杂动词范式中的表现，重点探讨了不同tokenization策略对模型内部表示的影响。

**💡 创新点**

创新点在于：①将自然语料和合成“VerbOnly”数据相结合，系统检验tokenization对形态学捕获的作用；②提出并使用范式级BLM任务来评估模型对形态学范式的学习；③对希伯来语模板形态（binyan）在字符级tokenization下的难点进行定量分析。

**🔧 技术方法**

技术与方法：使用BERTurk（土耳其语单语）、AlephBERT（希伯来语单语）和Electra（多语）生成句子嵌入；采用Feed‑Forward Neural Network进行分类；对比atomic、sub‑word和character级tokenization；使用F1、混淆矩阵等指标评估。

**📊 数据集**

数据集：从UD treebanks抽取土耳其语（Penn、Kenet）和希伯来语（HBT、IW）句子，构造BLM自然数据集；随后生成仅含动词的合成“VerbOnly”数据；所有数据公开可下载。

**📈 对比分析**

比较方法：在自然和合成数据上分别对四个模型进行F1评估。结果显示：土耳其语的单语与多语表现相近，均能正确识别范式；希伯来语单语模型平均F1≈0.84，显著优于多语模型（F1≈0.33，p<0.05）；在合成数据上多语模型略有提升但仍低于单语模型，特别是对causative范式的区分存在明显错误。

**⚠️ 局限性**

Limitations：仅覆盖两种语言且使用的是中等规模Transformer；未测试大规模LLM或更复杂的架构；缺乏人类上界对比；tokenization的细粒度分析仍有限，未系统探索多种语言特定的tokenization方案。

---

## 395. Poster: Camera Tampering Detection for Outdoor IoT Systems

**arXiv ID:** 2602.05706 | [PDF](https://arxiv.org/pdf/2602.05706v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 396. Joint Embedding Variational Bayes

**arXiv ID:** 2602.05639 | [PDF](https://arxiv.org/pdf/2602.05639v1)

**作者:** Amin Oji `[一作]` (University of Waterloo), Paul Fieguth `[通讯]` (University of Waterloo)

**通讯引用:** 12830 | [OpenAlex ID](https://openalex.org/A5078015739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Variational Joint Embedding (VJE)，在无对比学习的自监督框架中引入变分推断，构造归一化的潜变量模型，直接在编码器嵌入空间上最大化对称条件证据下界（ELBO）

**💡 创新点**

创新点在于：①使用极坐标分解将方向和模长独立建模；②采用重参数化的 Student‑t 方向/径向似然以获得鲁棒的角度和尺度估计；③不使用投影头，特征维度方差与方向似然共享，得到细粒度的不确定性表示；④通过一阶条件 ELBO 与 stop‑gradient 实现对视图对的自监督学习

**🔧 技术方法**

技术手段包括：变分自编码器框架、Student‑t 似然、极坐标分解、重参数化技巧、对称条件 ELBO、卷积 ResNet 编码器、MLP 推断网络、层归一化、β‑VAE 正则化

**📊 数据集**

实验数据集：ImageNet‑1K、CIFAR‑10/100、STL‑10、CIFAR‑10 单类异常检测（10 组），并对比 Rot+Trans、GOAD 等基线

**📈 对比分析**

在 ImageNet‑1K 上，VJE 线性 top‑1 65.6%（SimSiam 90.60%），在 CIFAR‑10 上 k‑NN 89.98% 与 90.60% SimSiam 相近，线性 92.1% 超过 SimSiam 91.8%；在单类异常检测上 VJE 取得 90.3% AUROC，优于 Rot+Trans 89.8% 和 GOAD 88.2%

**⚠️ 局限性**

局限性：在 ImageNet‑1K 上相较最强的确定性基线略低；Gaussian 似然极限导致后验崩溃；缺乏对更大尺度或多层次不确定性建模的支持；与对比学习方法相比，可能在边缘分类性能上有所折扣

---

## 397. Structural Disentanglement in Bilinear MLPs via Architectural Inductive Bias

**arXiv ID:** 2602.05635 | [PDF](https://arxiv.org/pdf/2602.05635v1)

**作者:** Ojasva Nema `[一作]` (Indian Institute of Technology Roorkee), Parikshit Pareek `[通讯]` (Indian Institute of Technology Roorkee)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5055795558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

探究双线性MLP通过显式乘性交互实现结构解耦，提升模型的可编辑性与长周期外推能力。

**💡 创新点**

首次将梯度流下双线性参数化的非混合性质与结构解耦关联，并证明其能实现无干扰的选择性忘记。

**🔧 技术方法**

使用双线性MLP、SwiGLU/GeGLU与传统ReLU/Tanh/Sigmoid网络对比，并结合梯度流解析、谱/奇异值分析等技术。

**📊 数据集**

采用模数算术（Z₉₇）、循环推理任务、李群动力学（旋转、体积保持）等人工合成数据集。

**📈 对比分析**

与点wise激活网络对比，双线性模型在任务拆分、单调外推以及保留特定模式时损伤更低，性能显著优于ReLU等基线。

**⚠️ 局限性**

仅在可辨识的代数结构任务上验证，理论依赖理想梯度流与平方损失，未证明在开放性感知任务中的泛化。

---

## 398. Unified Sensor Simulation for Autonomous Driving

**arXiv ID:** 2602.05617 | [PDF](https://arxiv.org/pdf/2602.05617v1)

**作者:** Nikolay Patakin `[一作]` (Lomonosov Moscow State University), Dmitry Senushkin `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5050618007)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了统一的传感器仿真框架 XSIM，能够在单一模型下模拟激光雷达和摄像头的滚动快门效应，提升自动驾驶环境下的仿真真实性。

**💡 创新点**

核心创新包括：1）针对 3DGUT splatting 的相位建模机制，专门处理球面摄像机（激光雷达）在方位角边界处的多模态投影问题；2）引入双重不透明度参数（摄像头与激光雷达各自的不透明度），解决几何与外观分布不匹配；3）统一的滚动快门建模，使不同传感器共享同一渲染管线。

**🔧 技术方法**

技术手段包括：3D 高斯 splatting、Unscented Transform（UT）投影、相位包装处理、卷积网络解码颜色、光学层的可见性与深度渲染、基于 Newton–Raphson 的时间迭代求解。

**📊 数据集**

在 Waymo Open Dataset、Argoverse 2 以及 PandaSet 三大自动驾驶公开数据集上进行训练与评估。

**📈 对比分析**

与多种基线（UniSim、NeuRAD、EmerNerf、PVG、StreetGaussians、OmniRe、HUGS、SplatAD 等）对比，XSIM 在 PSNR、SSIM、LPIPS、Chamfer Distance 等指标上均实现显著提升，例如在 Waymo 上 PSNR 提升 3.01dB、SSIM 提升 3.8%，LiDAR CD 错误下降 8.8 倍。

**⚠️ 局限性**

局限性主要包括：对极端动态场景下的非线性运动仍需进一步优化；双重不透明度参数增加了模型复杂度与训练成本；相位建模目前仅覆盖 ±π 的方位角范围，需扩展至更复杂的球面扫描模式。

---

## 399. SEAL: Symbolic Execution with Separation Logic (Competition Contribution)

**arXiv ID:** 2602.05703 | [PDF](https://arxiv.org/pdf/2602.05703v1)

**作者:** Tomáš Brablec `[一作]` (Brno University of Technology), Tomáš Vojnar `[通讯]` (Masaryk University)

**通讯引用:** 2437 | [OpenAlex ID](https://openalex.org/A5086446392)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

设计并实现了一款基于分离逻辑的静态分析器，用于验证处理无限链表等数据结构的程序的内存安全。

**💡 创新点**

创新点在于使用通用分离逻辑求解器（通过翻译到 SMT 实现可满足性与包含判定），解耦分析与逻辑推理，并支持用户自定义归纳谓词。

**🔧 技术方法**

采用前向抽象解释、符号堆表示、归纳谓词抽象、SMT 求解器、分离逻辑翻译、Frama‑C 插件实现等技术。

**📊 数据集**

使用 SV‑COMP MemSafety‑LinkedLists 基础类别的程序集，包含 134 个程序，其中 69 个涉及无限链表。

**📈 对比分析**

与 23 位参赛者比较，只在 4 位能够验证无限链表程序；在 MemSafety‑LinkedLists 竞赛中，验证的程序数多于多数工具，但性能仍落后于基于 SMG 的分析器。

**⚠️ 局限性**

局限性包括：只支持有限的 C 语言子集，整数值跟踪范围有限，缺少数组、指针运算等支持；易产生未知/假阳性；仅支持三种链表结构，无法处理更复杂的数据结构。

---

## 400. MedErrBench: A Fine-Grained Multilingual Benchmark for Medical Error Detection and Correction with Clinical Expert Annotations

**arXiv ID:** 2602.05692 | [PDF](https://arxiv.org/pdf/2602.05692v1)

**作者:** Congbo Ma `[一作]` (New York University Abu Dhabi), Farah E. Shamout `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 930 | [OpenAlex ID](https://openalex.org/A5023660328)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了MedErrBench多语言医学错误检测、定位和纠正基准，涵盖英、阿、汉三语并由临床专家注释。

**💡 创新点**

首创包含10类错误、跨语言且专家驱动的基准，系统评估LLM在错误检测、定位与纠正上的表现。

**🔧 技术方法**

采用GPT-4o、Gemini、Llama系列、Deepseek、MedGemma等LLM，结合错误类型定义、示例提示等技术进行实验。

**📊 数据集**

使用MedErrBench（基于MedQA、MedArabiQ等）构建的英、汉、阿三语数据集，并对其进行多层次注释。

**📈 对比分析**

通过准确率、ROUGE、BERTScore等指标比较三类任务，发现通用LLM在定位与纠正上与专门化模型相当，但整体表现仍低，尤其是非英语场景。

**⚠️ 局限性**

数据集规模有限，阿拉伯语子集不足；缺少严重性与公平性等注释，且未包含真实临床记录。

---

## 401. Almost Asymptotically Optimal Active Clustering Through Pairwise Observations

**arXiv ID:** 2602.05690 | [PDF](https://arxiv.org/pdf/2602.05690v1)

**作者:** Rachel S. Y. Teo `[一作]` (National University of Singapore), Vincent Y. F. Tan `[通讯]` (National University of Singapore)

**通讯引用:** 4858 | [OpenAlex ID](https://openalex.org/A5058345431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对固定置信度下的群体式聚类问题，提出了一种主动学习框架，利用成对比较数据主动探索以辨识底层聚类结构。

**💡 创新点**

创新点在于推导了信息理论下的样本复杂度下界，并证明最难区分的错误聚类仅需通过对邻近聚类的拆分/合并操作即可；随后给出了几乎达到该下界的 D‑tracking 采样与 GLR/可计算停机规则，且通过仅检索极大极小估计值大幅降低停机统计量的计算量。

**🔧 技术方法**

主要技术包括：KL 散度与信息量分析、Berge 最大化定理、Chernoff–Hoeffding 与 Jensen 不等式、二元熵优化、D‑tracking 采样策略与凸正则化的结合；同时利用等价类划分简化全局搜索。

**📊 数据集**

实验使用合成数据（随机生成的 M 项、K 个簇的聚类实例，概率 p>0.5 与 q<0.5 的二项分布）进行评估。

**📈 对比分析**

与先前基于同类信息聚类的算法（如 Chen 等 2015）进行对比；结果表明在相同置信水平下，该方法的期望采样数接近理论下界且显著低于对照方法，尤其在大 M 时表现尤为突出。

**⚠️ 局限性**

局限性包括：停机统计量仍需遍历有限但指数级别的等价类集合，导致在极大规模下仍具高计算复杂度；算法假设所有对比结果遵循相同的 Bernoulli 分布，未考虑工作者多样性与噪声不均匀性等实际场景。

---

## 402. Exploring AI-Augmented Sensemaking of Patient-Generated Health Data: A Mixed-Method Study with Healthcare Professionals in Cardiac Risk Reduction

**arXiv ID:** 2602.05687 | [PDF](https://arxiv.org/pdf/2602.05687v1)

**作者:** Pavithren V S Pakianathan `[一作]`, Jan David Smeddinck `[通讯]` (Ludwig Boltzmann Institute for Digital Health and Prevention)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究开发了一个集成LLM生成摘要和对话式接口的原型，用于帮助心血管风险降低中的医疗专业人员快速理解和分析患者生成的健康数据。

**💡 创新点**

创新点在于将LLM自动摘要与可视化仪表板相结合，并提供自然语言交互来弥补数据素养差距，同时系统性评估了其对临床工作流程的影响和潜在风险。

**🔧 技术方法**

技术主要包括大语言模型（GPT‑4‑Turbo）用于生成多模态摘要和对话式可视化；使用Plotly Dash构建仪表盘；采用自然语言处理生成Plotly JSON用于图表绘制。

**📊 数据集**

使用合成的七个患者人物一年期多模态数据（体能活动、血压、睡眠、久坐时间），数据由ChatGPT根据预设模板和随机函数生成。

**📈 对比分析**

通过16名心血管康复专业人员的混合方法研究（定量 NASA‑TLX、SUS、信任量表；定性访谈）进行评估，结果显示LLM摘要提高了信息获取速度，用户满意度高，工作负荷无显著提升，但未在工作效率上产生统计显著差异。

**⚠️ 局限性**

局限包括：使用合成数据缺乏真实世界噪声和缺失；样本量小、无临床患者交互，未检验系统在实际咨询中的效果；LLM未针对传感器数据预训练，摘要准确性有限；缺乏对不同专业角色的可扩展性与安全合规评估。

---

## 403. Variable Search Stepsize for Randomized Local Search in Multi-Objective Combinatorial Optimization

**arXiv ID:** 2602.05675 | [PDF](https://arxiv.org/pdf/2602.05675v1)

**作者:** Xuepeng Ren `[一作]` (China University of Geosciences), Miqing Li `[通讯]` (University of Birmingham)

**通讯引用:** 7485 | [OpenAlex ID](https://openalex.org/A5036335232)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种可变步长随机局部搜索（VS‑RLS），通过在搜索过程中动态调整邻域大小，以改进多目标组合优化问题的求解。

**💡 创新点**

创新点在于：①将邻域步长从1逐步扩展到最大可行值，既能在早期广泛探索，又能在后期细致改进；②使用单解局部搜索结合可变步长的两相策略（探索-开发），显著提升多目标多样性与收敛性；③对比固定步长的随机局部搜索与传统MOEAs，首次在多目标组合优化中展示可变步长的优势。

**🔧 技术方法**

采用了随机局部搜索框架（SEMO/PLS）与可变邻域机制；实现了二进制问题的多位翻转、置换问题的2‑opt/2‑swap；使用Pareto主导关系维护归档；实验中使用了标准的多目标评价指标——超体积（HV）。

**📊 数据集**

使用四类经典MOCOP数据集：多目标0/1背包（D=500）、旅行商问题（D=200）、二次分配问题（D=100）以及NK景观（D=100），并在不同规模和评估预算下重复测试。

**📈 对比分析**

将VS‑RLS与三种主流MOEA（NSGA‑II、MOEA/D、SMS‑EMOA）、随机采样、SEMO、PLS进行比较。实验结果显示，VS‑RLS在大多数实例和评估规模下均取得最高HV，尤其在背包、TSP、QAP和NK景观上均表现出更好的多样性与覆盖度；虽然在极少评估预算（如1×10⁵）时表现略逊，但随着评估次数增加，优点显现。

**⚠️ 局限性**

局限性：①在评估预算极低时，步长扩展会耗尽资源，导致收敛慢；②对初始解的依赖较高，若无足够探索阶段，可能陷入局部最优；③目前参数（T_vl、V_C）的设置仍需经验调优，缺乏自适应机制；④仅针对单解局部搜索，未与群体搜索结合，可能在大规模问题中受限。

---

## 404. Fast Private Adaptive Query Answering for Large Data Domains

**arXiv ID:** 2602.05674 | [PDF](https://arxiv.org/pdf/2602.05674v1)

**作者:** Miguel Fuentes `[一作]` (University of Massachusetts Amherst), Daniel Sheldon `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 4536 | [OpenAlex ID](https://openalex.org/A5061155671)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 AIM+GReM 机制，利用残差查询在差分隐私框架下高效回答大规模多维边际查询。

**💡 创新点**

创新点包括：① in‑axis 多维数组实现残差与边际的快速转换；② GReM 的延迟更新（lazy updating）以避免冗余重构；③ 条件残差规划（CRP）在每轮测量中动态优化隐私预算分配。

**🔧 技术方法**

采用的技术包括：in‑axis 轴内操作、Gaussian Residuals‑to‑Marginals (GReM) MLE 重构、exponential 机制的自适应选择、凸优化求解 CRP 以及零浓度差分隐私（zCDP）保障。

**📊 数据集**

实验数据集涵盖 Adult、ACS、Loans 三个真实世界数据集。

**📈 对比分析**

与 AIM+PGM、ResidualPlanner 以及原始 AIM 进行比较，AIM+GReM 在速度上提升 18–92 倍，误差与 ResidualPlanner 相当或更优，并在大域场景下比 AIM+PGM 快 100–700 倍。

**⚠️ 局限性**

局限性包括：在极低隐私预算下误差仍较高；对高阶残差测量的精度受限；仅在高斯噪声模型下验证，未探讨其它噪声分布。

---

## 405. Low-complexity Design for Beam Coverage in Near-field and Far-field: A Fourier Transform Approach

**arXiv ID:** 2602.05666 | [PDF](https://arxiv.org/pdf/2602.05666v1)

**作者:** Chao Zhou `[一作]` (Southern University of Science and Technology), Chengwen Xing `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 2496 | [OpenAlex ID](https://openalex.org/A5008383657)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文针对多天线系统的远场与近场波束覆盖问题，提出了基于傅里叶变换的低复杂度设计方法。

**💡 创新点**

创新点在于将波束覆盖问题转化为天线域的权重塑形问题，利用逆傅里叶得到理想权重序列，并通过“保护放大”缓解有限天线导致的 roll‑off；对近场采用一阶泰勒展开实现二维逆傅里叶。

**🔧 技术方法**

采用了傅里叶变换、卷积定理、泰勒展开以及闭式 sinc 结构的天线权重设计。

**📊 数据集**

实验使用仿真场景：64 架天线 30 GHz 远场与 256 架天线 30 GHz 近场，未使用公开数据集。

**📈 对比分析**

与传统基于采样与优化的方案以及 DFT 代码字求解方法相比，所提方案在保持相似最差点波束增益的同时，将计算时延从约 900 ms 降至 0.007 ms，降低数十倍。

**⚠️ 局限性**

局限性包括：近场设计对角度偏差要求小于阈值，需要对大角度偏差进行分区；近场范围失焦效应受限；对模拟线性相位约束的相位阵列方案尚未完整给出。

---

## 406. GLASS: A Generative Recommender for Long-sequence Modeling via SID-Tier and Semantic Search

**arXiv ID:** 2602.05663 | [PDF](https://arxiv.org/pdf/2602.05663v1)

**作者:** Shiteng Cao `[一作]` (Tsinghua University), Cheng Yang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 11058 | [OpenAlex ID](https://openalex.org/A5060417049)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了基于SID‑Tier和语义硬搜索的生成式长序列推荐框架GLASS，能够在生成过程中直接利用长时序用户兴趣。

**💡 创新点**

创新点在于：①将长时序交互映射为统一兴趣向量并注入生成过程；②使用生成的首级SID作为查询进行语义硬检索并通过门控融合短期与长周期信息；③提出稀疏增强策略（邻域扩展与码本重塑）缓解检索稀疏性。

**🔧 技术方法**

采用RQ‑VAE量化构建层次化语义ID，Transformer自回归解码器，SID‑Tier模块、门控融合机制、语义邻域扩展与码本重塑技术。

**📊 数据集**

使用两大工业真实数据集：TAOBAO‑MM和KuaiRec，并对正样本进行过滤、长短序列划分。

**📈 对比分析**

与多种ID级别（Caser、SASRec、Bert4Rec 等）和SID级别（Tiger、DualGR 等）基线在 H@1、NDCG@3 等指标上比较，GLASS 在 Taobao‑MM 上 H@1 提升约21%，NDCG@3 提升约30%；在 KuaiRec 上提升约5%。

**⚠️ 局限性**

局限性包括：①在码本细粒度很高时，语义邻域扩展可能引入噪声导致性能下降；②对多模态特征质量依赖较高，特征不足时提升有限。

---

## 407. Probabilistic Multi-Regional Solar Power Forecasting with Any-Quantile Recurrent Neural Networks

**arXiv ID:** 2602.05660 | [PDF](https://arxiv.org/pdf/2602.05660v1)

**作者:** Slawek Smyl `[一作]` (Walmart), Grzegorz Dudek `[通讯]` (Czestochowa University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于Any‑Quantile RNN的多区域光伏发电概率预测框架，能够在单模型中输出任意分位数的预测。

**💡 创新点**

创新点包括任何分位数预测、双轨RNN结合跨区域上下文、扩展的dilated recurrent cell、patching与动态团队集成以及分段重叠分位子区的专用预测器。

**🔧 技术方法**

采用了深度循环网络、dilated recurrent cell、内部上下文嵌入、Patch、动态团队与分位子区化等深度学习技术，并辅以量化损失与置信度正则。

**📊 数据集**

使用30年（1986‑2015）小时级EMHIRES光伏发电数据，覆盖259个欧洲NUTS‑2行政区。

**📈 对比分析**

与ARIMA、Theta、DeepAR、Transformer、WaveNet、TFT等统计与神经基线在CRPS、MARFE、Winkler、MAE等指标上进行对比，AQ‑RNN/ensemble 在所有指标上均显著优于基线，尤其在低/高分位数的校准和PI质量上表现突出。

**⚠️ 局限性**

局限性在于仅使用历史发电数据未结合气象输入、对极端分位数仍存在偏差、模型参数量大且训练成本高、对夜间零产能时段的处理未做特殊设计。

---

## 408. AI chatbots versus human healthcare professionals: a systematic review and meta-analysis of empathy in patient care

**arXiv ID:** 2602.05628 | [PDF](https://arxiv.org/pdf/2602.05628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 409. One Size Does NOT Fit All: On the Importance of Physical Representations for Datalog Evaluation

**arXiv ID:** 2602.05651 | [PDF](https://arxiv.org/pdf/2602.05651v1)

**作者:** Nick Rassau `[一作]` (Johannes Gutenberg University), Felix Schuhknecht `[通讯]` (Johannes Gutenberg University)

**通讯引用:** 548 | [OpenAlex ID](https://openalex.org/A5005397258)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了 Datalog 评估中物理表示的选择问题，提出了一个高度灵活的 Datalog 引擎，并对 13 种物理表示与 7 维度工作负载特征的相互作用进行系统实验，基于实验结果设计决策树并实现了自动化选择机制；

**💡 创新点**

创新点在于：①首次从多维度系统化评估物理表示与工作负载特征的关系；②提出基于工作负载签名的自动选择框架，包括索引键选择、数据结构、访问类型和共享策略的决策树；③通过实验验证自动化配置能逼近手工调优性能，并优于现有主流 Datalog 引擎；

**🔧 技术方法**

使用的技术包括：自研可插拔物理表示的 Datalog 引擎、BTree、Radix Tree、Hash Table、Sorted Array、Tuple Store 等 13 种组合、基于工作负载签名的对齐图和操作计数、三棵决策树实现自动配置；

**📊 数据集**

实验数据集：四个真实工作负载（Program Analysis 的 Andersen 近似分析、Graph Analytics 的 Reachability、Same generation、Transitive closure），以及合成实验；

**📈 对比分析**

比较方法：对同一工作负载在四种配置下测量运行时间——自动配置(-Auto)、Soufflé 类似配置(-Soufflé-like)、手工最佳配置(-Hand)、以及 Soufflé、RecStep、DDLog 等主流系统；结果显示 -Auto 与 -Hand 相差 ≤1.1×，且在所有工作负载上均优于其他系统；

**⚠️ 局限性**

局限性：仅在单节点主存环境下实验；未考虑分布式 Datalog 引擎；物理表示种类仍有限，未来需扩展至更丰富的数据结构和更大规模数据。

---

## 410. Enhancing Personality Recognition by Comparing the Predictive Power of Traits, Facets, and Nuances

**arXiv ID:** 2602.05650 | [PDF](https://arxiv.org/pdf/2602.05650v1)

**作者:** Amir Ansari `[一作]` (Universitat de Barcelona), Cristina Palmero `[通讯]` (King's College London)

**通讯引用:** 410 | [OpenAlex ID](https://openalex.org/A5056906303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用音视频交互数据构建人格识别模型，并探究将Big Five模型的细粒度层级（facet、nuance）作为训练标签，是否能提升自我报告人格的预测精度。

**💡 创新点**

① 引入多模态Transformer（MulT）与跨主体注意力机制，使模型既能捕捉不同模态间的相互作用，又能考虑交互双方的社交线索；② 证明细粒度的nuance标签在所有任务中均显著优于trait和facet层级，体现了更精细、情境化的预测潜力。

**🔧 技术方法**

使用Transformer架构（MulT），跨模态与跨主体注意力、自回归频谱特征表示（离散傅里叶变换）、Bayesian超参数调优与Adam优化。

**📊 数据集**

UDIVA v0.5 数据集：约80小时的双人对话（含自由对话与三种结构化任务），配合BFI-2问卷提供60个Likert量表项目（即15个facet、5个trait）。

**📈 对比分析**

在10折主题独立划分的实验中，采用MSE、MAE、PCC、R²进行评估。nuance模型在所有任务中均达到MSE≈0.046–0.049、PCC≈0.94–0.95、R²≈0.82–0.83，较基线降低MSE达87%/73%/63%，而trait、facet模型表现中等。

**⚠️ 局限性**

局限性：训练样本量有限，任务间差异小；nuance标签噪声大导致单项预测波动；数据缺乏文化、年龄与关系多样性，模型泛化与潜在偏差尚待进一步评估。

---

## 411. Empowering Time Series Analysis with Large-Scale Multimodal Pretraining

**arXiv ID:** 2602.05646 | [PDF](https://arxiv.org/pdf/2602.05646v1)

**作者:** Peng Chen `[一作]` (East China Normal University), Chenjuan Guo `[通讯]` (East China Normal University)

**通讯引用:** 3107 | [OpenAlex ID](https://openalex.org/A5084021933)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出基于时间序列、文本和图像的多模态预训练范式，并构建MM-TS大规模数据集，训练HORAI模型实现时间序列预测与异常检测。

**💡 创新点**

首次构造包含多模态信息的跨域时间序列数据集，并设计频率增强跨模态编码器与时频Mixture‑of‑Experts解码器，实现更优的模态融合与跨域泛化。

**🔧 技术方法**

采用FFT频率分解、流式注意力对齐、时频Mixture‑of‑Experts路由以及GPT式自回归预训练等技术。

**📊 数据集**

自建MM‑TS（6个领域、约10亿时间点，含时间序列、文本、图像），以及常用预测与异常检测下游数据集。

**📈 对比分析**

与ChatTime、VisionTS、ROSE、Timer、MOIRAI、GPT4MTS等基准进行零样本和少量样本对比，HORAI在18例预测任务中15例领先（MSE下降29.6%），在15例异常检测任务中13例领先（AUC提升13%‑20%）。

**⚠️ 局限性**

依赖LLM生成文本的质量与对齐机制，真实视觉图像缺失导致表征受限；跨域泛化虽优但仍受限；模型规模和算力需求高。

---

## 412. Mode-Dependent Rectification for Stable PPO Training

**arXiv ID:** 2602.05619 | [PDF](https://arxiv.org/pdf/2602.05619v1)

**作者:** Mohamad Mohamad `[一作]` (Inria), Xavier Descombes `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种两阶段训练框架—Mode-Dependent Rectification (MDR)，用来稳定使用模式依赖层（如 BatchNorm、Dropout）的 PPO 训练。

**💡 创新点**

创新点在于将训练与评估模式差异视为对 PPO clipping 的动态扰动，并通过额外的确定性校正阶段恢复信任区域，从而在无需改动网络结构的前提下提升稳定性与性能。

**🔧 技术方法**

主要技术包括 PPO、BatchNorm、Dropout、Batch Renormalization、层归一化、组归一化，以及两阶段训练的 MDR 校正。

**📊 数据集**

实验使用了 Procgen 生成式游戏（六种难度）和两类真实图像补丁定位任务（自然图像与病理图像）。

**📈 对比分析**

通过与传统 BN、Eval、Batch Renorm、Dropout 等方法对比，MDR 在大多数任务上显著提升了训练稳定性和最终回报，并在 Patch-localization 任务中缩小了泛化差距。

**⚠️ 局限性**

局限性包括对 MDR 超参数的敏感性、仅在单机 PPO 上验证、未结合其他归一化技术或多任务设置，且对极端动态分布漂移的鲁棒性仍需进一步评估。

---

## 413. FedRandom: Sampling Consistent and Accurate Contribution Values in Federated Learning

**arXiv ID:** 2602.05693 | [PDF](https://arxiv.org/pdf/2602.05693v1)

**作者:** Arno Geimer `[一作]` (University of Luxembourg), Radu State `[通讯]` (University of Luxembourg)

**通讯引用:** 5242 | [OpenAlex ID](https://openalex.org/A5069228908)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedRandom 方法，通过在每轮聚合时随机选择已有聚合策略，生成更多样本来评估参与者贡献，从而降低贡献估计的方差和偏差。

**💡 创新点**

创新点在于将聚合策略的随机采样视为统计估计问题，利用单一聚合器即可模拟成百上千个聚合过程，显著提升贡献估计的稳定性和准确性。

**🔧 技术方法**

技术手段包括：FedRandom 随机采样聚合策略集合 S、Shapley 值的多轮重构（MSM）、统计学平均估计、L2 与 L∞ 距离评估。

**📊 数据集**

实验使用四个常见视觉数据集：CIFAR‑10、CIFAR‑100、MNIST、Fashion‑MNIST，采用 9 种随机种子、3 轮 epoch、3 个 Dirichlet α 参数，共 324 个 FL 场景。

**📈 对比分析**

与 MSM（多策略平均）对比，FedRandom 在 94% 的场景中方差更低，85% 的场景 L2 距离更小，82% 的场景 L∞ 距离更小；在 92% 的案例中降低与真实贡献的距离超过 30%，并在 33/36（92%）情况中优于 MSM。

**⚠️ 局限性**

局限性包括：需额外计算和通信开销（样本数线性扩展）；Shapley 计算在大规模客户端时指数增长；只验证了少数聚合策略，未探讨不同目标函数或更复杂任务的适用性。

---

## 414. Time-Complexity Characterization of NIST Lightweight Cryptography Finalists

**arXiv ID:** 2602.05641 | [PDF](https://arxiv.org/pdf/2602.05641v1)

**作者:** Najmul Hasan `[一作]` (University of North Carolina), Prashanth BusiReddyGari `[通讯]` (University of North Carolina)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

构建符号模型，推导并比较了NIST十大轻量级密码学最终候选算法的时间复杂度

**💡 创新点**

提出了统一的三阶段（初始化‑数据处理‑最终化）时间复杂度模型，并给出每个算法的符号表达式，揭示设计参数对资源受限环境的计算规模影响

**🔧 技术方法**

符号时间复杂度分析、数学建模、分段复杂度分解

**📊 数据集**

无实际数据集，使用十个算法的理论规范和参数进行分析

**📈 对比分析**

通过符号复杂度表达式对十个算法进行对比，发现GIFT‑COFB、Grain‑128AEAD和ISAP具有最简单的线性标度，其余算法的系数与设计差异相关

**⚠️ 局限性**

仅为理论推导，未考虑硬件实现细节、编译器优化、并行性等实际因素，缺乏实测验证

---

## 415. Generative Ontology: When Structured Knowledge Learns to Create

**arXiv ID:** 2602.05636 | [PDF](https://arxiv.org/pdf/2602.05636v1)

**作者:** Benny Cheung `[一作]` `[通讯]` (Dynamind Research), Benny Cheung (Dynamind Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了生成本体（Generative Ontology）框架，将传统本体知识与大型语言模型结合，实现结构化的创意生成；

**💡 创新点**

创新点在于把本体作为生成语法和验证合约，通过可执行的Pydantic模式、DSPy签名、多代理流水线与检索增强，解决LLM结构性幻觉和本体静态性问题；

**🔧 技术方法**

技术包括Pydantic可执行模式、DSPy声明式语言、基于ChromaDB/SQLite的检索增强、多代理（Mechanics Architect、Theme Weaver等）以及迭代验证与改进循环；

**📊 数据集**

使用了BoardGameGeek（约2,000款桌游）语义向量和元数据集进行检索增强，并在游戏设计任务中评估；

**📈 对比分析**

与单一LLM生成方式对比，生成本体通过结构约束、机制-组件验证、平衡评估和可玩性评分等指标，生成的游戏设计在连贯性、可玩性与创新度上明显优于传统LLM输出；

**⚠️ 局限性**

局限性包括对高质量本体的依赖、检索范围受限、LLM仍可能产生深层语义冲突、以及多代理协调与迭代过程的计算成本。

---

## 416. ROMAN: Reward-Orchestrated Multi-Head Attention Network for Autonomous Driving System Testing

**arXiv ID:** 2602.05629 | [PDF](https://arxiv.org/pdf/2602.05629v1)

**作者:** Jianlei Chi `[一作]` (Hangzhou Institute of Technology Xidian University), Jun Sun `[通讯]` (Singapore Management University)

**通讯引用:** 21787 | [OpenAlex ID](https://openalex.org/A5100728816)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于奖励调度的多头注意力网络（ROMAN），用于生成高风险交通法规违章场景，帮助自动驾驶系统（ADS）进行更全面的安全测试。

**💡 创新点**

创新点包括：①使用大语言模型（LLM）对交通法规进行风险加权，评估违章的严重度与发生频率；②采用多头注意力机制捕捉多车、信号灯、天气等因素的复杂交互；③结合代理奖励模型快速训练，避免仿真耗时。

**🔧 技术方法**

技术手段包括 Transformer‑Encoder 与多头注意力网络、STL（Signal Temporal Logic）形式化交通法规、LLM（GPT/Claude/Gemini/Qwen）风险评估、代理奖励网络、CARLA 仿真平台以及 DTW（Dynamic Time Warping）度量场景多样性。

**📊 数据集**

使用 12,800 条基于 LawBreaker 生成的测试场景作为训练集，并采用 81 条可 STL 表达的中国交通法规条款作为评估标准。

**📈 对比分析**

与 ABLE 和 LawBreaker 进行对比实验，ROMAN 在所有四类路况下的违章数量、最高违章次数和高风险比例均优于两者；平均违章数比 ABLE 提升 7.91%，比 LawBreaker 提升 55.96%；DTW 中位数距离提升约 19% 以上，说明场景多样性更好；训练时间较长，但推理时间与 ABLE 相近。

**⚠️ 局限性**

局限性包括：①训练时间和 GPU 计算成本显著高于基线；②风险加权依赖于法律文档的完整性与 LLM 的偏差；③对极罕见边缘情况覆盖有限，可能仍遗漏某些极端高风险场景。

---

## 417. Reactive Knowledge Representation and Asynchronous Reasoning

**arXiv ID:** 2602.05625 | [PDF](https://arxiv.org/pdf/2602.05625v1)

**作者:** Simon Kohaut `[一作]` (Artificial Intelligence and Machine Learning Group, TU Darmstadt), Devendra Singh Dhami `[通讯]` (Uncertainty in Artificial Intelligence Group, TU Eindhoven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种异步概率编程语言 Resin 与其对应的自适应推理结构 Reactive Circuits（RC），实现实时、精确的连续推理。

**💡 创新点**

创新点在于：1) 将概率逻辑与反应式编程结合，形成可与异步数据流交互的高层语言；2) 通过 FoC（频率变化）驱动的动态电路重构与记忆化，显著降低不必要的计算。

**🔧 技术方法**

技术包括：概率逻辑编程、答案集程序 (ASP) 编译、代数电路、分布式数据传输 (DDS)、Kalman 滤波估计 FoC、Lift/Drop 结构重构、按频率分层的 RC 评估。

**📊 数据集**

数据集：合成随机 FoC 数据、基于 AirSim 的无人机群（UAS）高保真仿真，涉及 42 条源信号（以及其否定）和 2^21-1 个稳定模型。

**📈 对比分析**

比较方法：将 RC 与传统一次性完整重计算（flat）和仅结构优化的全重计算（adapted）比较。实验显示，RC 在飞行安全判定任务中相较于 flat 方法实现了数十至数百倍的速度提升，且相对于 adapted 方法进一步提升数倍。内存消耗随分层细粒度增加而上升。

**⚠️ 局限性**

局限性：1) 需要手动设置 FoC 阈值和分层宽度，影响精度与性能；2) 记忆化与重构带来额外内存占用；3) 对高度非平稳或极端噪声的 FoC 估计需要更鲁棒的滤波方法；4) 目前仅支持可分布式的离散/连续概率模型，尚未覆盖更复杂的神经-符号混合模型。

---

## 418. Perception-Based Beliefs for POMDPs with Visual Observations

**arXiv ID:** 2602.05679 | [PDF](https://arxiv.org/pdf/2602.05679v1)

**作者:** Miriam Schäfers `[一作]` (Ruhr University Bochum), Maximilian Weininger `[通讯]` (Ruhr University Bochum)

**通讯引用:** 575 | [OpenAlex ID](https://openalex.org/A5042375584)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将视觉感知模型与贝叶斯更新分离的框架，用于解决视觉POMDP。

**💡 创新点**

创新点在于用感知模型近似观测概率并结合不确定性量化实现可扩展的贝叶斯更新，兼容任意贝叶斯POMDP求解器。

**🔧 技术方法**

利用深度卷积网络进行图像分类，蒙特卡罗Dropout/熵等不确定性评估，配合贝叶斯更新和POMDP求解器（POMCP、DESPOT、SARSOP等）。

**📊 数据集**

在三个基准上验证：自定义交通灯图像、102分类花卉数据集、改造的FrozenLake（带滑行度隐藏变量）。

**📈 对比分析**

与端到端DRL、PSRL等方法对比，表现与最优基线相当并在视觉噪声下更鲁棒，计算时间主要受底层求解器限制。

**⚠️ 局限性**

局限性包括实现效率不足导致部分求解器性能下降、对感知模型准确度敏感、仅适用于可分解为视觉与非视觉的VPOMDP。

---

## 419. Determining Energy Efficiency Sweet Spots in Production LLM Inference

**arXiv ID:** 2602.05695 | [PDF](https://arxiv.org/pdf/2602.05695v1)

**作者:** Hiari Pizzini Cavagna `[一作]` (University of Bologna), Andrea Bartolini `[通讯]` (University of Bologna)

**通讯引用:** 3167 | [OpenAlex ID](https://openalex.org/A5047906923)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型推理过程中的能耗曲线，提出了基于Transformer计算与内存访问复杂度的解析模型，用以预测不同输入输出长度组合下的能效甜点。

**💡 创新点**

创新点在于将推理能耗拆解为预填充阶段的二次输入成本与解码阶段的线性输出成本，并将内存访问纳入模型，显著提升预测准确度（平均MAPE≈1.8%），揭示了能效峰值的非线性规律。

**🔧 技术方法**

使用TensorRT-LLM推理框架、NVIDIA H100 GPU、FP16/ BF16 计算以及对Transformer的多头/分组/多查询注意力的分析，构建了FLOPs与内存访问双模型。

**📊 数据集**

实验数据来自多款1B–9B参数的LLM（OPT、LLaMA、Gemma、Falcon、Qwen2、Granite），在64–4096令牌的输入输出长度组合以及不同请求数（10/100/1000）下进行收集。

**📈 对比分析**

通过与四种基线模型（常数、仅输出、含输入、混合项）对比，所提模型在所有模型上平均MAPE降至1.79%，比基线降低约96%；能效峰值相对于最差配置提升约33倍。

**⚠️ 局限性**

局限性包括仅测试中等规模模型、仅单一硬件平台和单一推理框架，未考虑批量大小对能效的影响，且模型未对更大规模LLM或不同硬件做泛化验证。

---

## 420. Mining Generalizable Activation Functions

**arXiv ID:** 2602.05688 | [PDF](https://arxiv.org/pdf/2602.05688v1)

**作者:** Alex Vitvitskyi `[一作]` (Google DeepMind), Petar Veličković `[通讯]` (Google DeepMind)

**通讯引用:** 14214 | [OpenAlex ID](https://openalex.org/A5008869927)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过 AlphaEvolve 进化搜索框架，自动发现并验证能在 OOD（out‑of‑distribution）场景下更好泛化的激活函数。

**💡 创新点**

创新点包括：①利用前沿 LLM（如 Gemini）在无限可能的 Python 函数空间中生成候选激活函数；②将 OOD 验证误差作为适应度函数，直接导向泛化能力；③发现了多种点wise 激活函数，其形式为标准激活函数与周期性（如 sin、sinc）项的和或乘积，提升了周期性数据的外推性能。

**🔧 技术方法**

技术手段：AlphaEvolve 进化框架、LLM 驱动的函数生成与评估、基于 MSE 的适应度评估、在小规模 MLP 上快速训练、随后在 CIFAR‑10、ImageNet、CLRS‑30（含 OOD 测试）和 ogbg‑molhiv 上的下游验证。

**📊 数据集**

数据集：训练阶段使用自制的低维回归合成数据（多项式、谐波、Feynman 公式）；下游测试使用公开标准数据集 CIFAR‑10、ImageNet、CLRS‑30（含 OOD 子集）和 ogbg‑molhiv。

**📈 对比分析**

对比方法：与 ReLU、GELU 等基线激活函数在合成数据、CIFAR‑10、ImageNet、CLRS‑30、ogbg‑molhiv 上进行相同模型和训练设置下的评估。结果显示：在合成 OOD 测试误差明显下降；在 CLRS‑30 OOD 精度最高；在 ImageNet/CIFAR‑10 与 ReLU 基线相当甚至略优；GELU‑Sinc 变体在所有任务上表现最为稳健。

**⚠️ 局限性**

局限性：①批统计基的激活函数易过拟合、OOM，且难以迁移到大规模任务；②仅在极小合成任务上训练，可能未覆盖更复杂的分布；③依赖 LLM 生成代码，代码质量与可解释性不确定；④部分发现函数在不同架构或数据集上未能保持一致的泛化提升。

---

## 421. Accelerating Benchmarking of Functional Connectivity Modeling via Structure-aware Core-set Selection

**arXiv ID:** 2602.05667 | [PDF](https://arxiv.org/pdf/2602.05667v1)

**作者:** Ling Zhan `[一作]` (Southwest University), Tao Jia `[通讯]` (Chongqing Normal University)

**通讯引用:** 3628 | [OpenAlex ID](https://openalex.org/A5019949140)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种基于Transformer的结构感知对比学习框架SCLCS，用于在大规模fMRI数据中挑选核心样本以实现功能连接模型（SPI）排名的高效保真评估。

**💡 创新点**

创新点包括：① 将核心样本选择视为排名保持问题；② 设计Structure Perturbation Score (SPS) 用于衡量样本的结构稳定性；③ 引入密度平衡采样增强多样性；④ 证明改进Transformer具备对连续SPI映射的通用逼近能力。

**🔧 技术方法**

核心技术涵盖：Transformer自注意力、可学习头融合、结构扰动评分、密度平衡采样、身份监督对比学习以及nDCG@k排名一致性评估。

**📊 数据集**

使用REST‑meta‑MDD大规模多站点静息态fMRI数据（约4,520个滑动窗口样本）进行实验。

**📈 对比分析**

与九种现有核心样本选择方法（Random、k‑Means、Forgetting、Entropy、EL2N、AUM、CCS、EVA、BOSS）对比，SCLCS在10%核心样本下nDCG@5/10/20的排名一致性分别提升至约81%、66%和57%，SCLCS_Dense在MDD诊断任务上更优，整体表现明显优于基线。

**⚠️ 局限性**

局限性主要在于：① 核心样本的代表性依赖SPS和密度估计，仍可能忽略极端稀有模式；② 仅在REST‑meta‑MDD上验证，跨数据集推广性待进一步验证；③ 对SPI的通用性假设在实际任务中可能受限。

---

## 422. Tight Long-Term Tail Decay of (Clipped) SGD in Non-Convex Optimization

**arXiv ID:** 2602.05657 | [PDF](https://arxiv.org/pdf/2602.05657v1)

**作者:** Aleksandar Armacki `[一作]` (Ecole Polytechnique Federale de Lausanne), Ali H. Sayed `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了非凸优化中基于随机梯度下降（SGD）及其剪裁变种的长期尾部衰减行为，给出了固定误差阈值下的指数衰减上界与下界；

**💡 创新点**

创新点在于直接利用大偏差理论（LDP）推导长期尾概率上界，得到比以往有限时高概率结果快一个数量级的指数衰减速率，并证明该速率在某些设定下是最优的；

**🔧 技术方法**

主要技术包括：大偏差原理（Gärtner–Ellis定理）、对梯度噪声的指数生成函数（MGF）控制、对剪裁梯度的无偏与有偏分解以及精细的渐进分析；

**📊 数据集**

论文未使用具体数据集，而是进行理论推导与分析，关注的是普适的非凸目标与噪声假设；

**📈 对比分析**

与现有基于高概率界的结果相比，本文在长期尾衰减速率上提升了一个数量级（从~√t或t^β/2提升到t/ln t 或 t^β/ ln t），但仍需要满足梯度有界或噪声矩上界等假设；

**⚠️ 局限性**

主要限制包括：需对梯度或噪声设定更强假设（如梯度有界、噪声矩已知），剪裁阈值需精细调节；同时得到的是上界，完整的 LDP 仍未实现，且下界仅为指数级近似，未覆盖所有实例。

---

## 423. Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation

**arXiv ID:** 2602.05694 | [PDF](https://arxiv.org/pdf/2602.05694v1)

**作者:** Shuting Jiang `[一作]` (Kunming University of Science and Technology), Zhengtao Yu `[通讯]` (Kunming University of Science and Technology)

**通讯引用:** 4573 | [OpenAlex ID](https://openalex.org/A5100619287)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种只更新共识对齐神经元的多域机器翻译微调框架（CANEFT）

**💡 创新点**

通过最大化神经元行为与域特征的互信息，筛选出既具域无关又具域特定能力的神经元

**🔧 技术方法**

利用激活‑梯度重要性评估、互信息（MI）选择和掩码梯度更新等技术

**📊 数据集**

在德英、汉英两语种上使用10个域（IT、Law、Medical、Subtitles等）公开数据集

**📈 对比分析**

与多种PEFT基线（LoRA、DoRA、LLaMA Pro、LAPE等）和全参数微调对比，CANEFT在10个域平均提升约1.4–1.6 BLEU、1–1.7 COMET，且仅更新约1%参数，且对未见域具有良好泛化

**⚠️ 局限性**

方法仍依赖于对域标签的可用性，且在极端域差异或样本稀缺情况下共识神经元的识别可能受限

---

## 424. (Computer) Vision in Action: Comparing Remote Sighted Assistance and a Multimodal Voice Agent in Inspection Sequences

**arXiv ID:** 2602.05671 | [PDF](https://arxiv.org/pdf/2602.05671v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 425. From Vision to Decision: Neuromorphic Control for Autonomous Navigation and Tracking

**arXiv ID:** 2602.05683 | [PDF](https://arxiv.org/pdf/2602.05683v1)

**作者:** Chuwei Wang `[一作]` (Cornell University), Anastasia Bizyaeva `[通讯]` (Cornell University)

**通讯引用:** 365 | [OpenAlex ID](https://openalex.org/A5072253048)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种神经形态控制框架，将高维视觉输入映射为神经动力学，再转化为机器人本体运动指令，实现自主导航与跟踪。

**💡 创新点**

创新点在于利用输入驱动的环形吸引子与分岔机制，使机器人在对称环境中自发打破选择不确定性，兼具近端感知的低计算量与远端规划的决策力。

**🔧 技术方法**

技术包括基于可微分的神经网络动力学、输入驱动的权重矩阵、阈值化决策、以及RGB/光流目标检测等视觉预处理。

**📊 数据集**

使用的“数据集”为仿真环境中的多目标地图、AirGen的光照渲染场景以及实验室收集的实时RGB图像与光流序列。

**📈 对比分析**

与MPC、PF、RL基线比较时，神经形态方法在对称情境下成功决策、计算时间仅0.1ms、相较MPC快3-4阶、RL快5倍，且保持了决策的公平性与鲁棒性。

**⚠️ 局限性**

局限性包括对目标数目与分布的可扩展性尚未系统评估、对高速度飞行或极端感知噪声的鲁棒性需进一步验证。

---

## 426. A Stronger Benchmark for Online Bilateral Trade: From Fixed Prices to Distributions

**arXiv ID:** 2602.05681 | [PDF](https://arxiv.org/pdf/2602.05681v1)

**作者:** Anna Lunghi `[一作]` (Politecnico di Milano), Alberto Marchesi `[通讯]` (Politecnico di Milano)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5039843107)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在线双边贸易，提出了一种算法以在随机环境中实现相对于全球预算平衡（GBB）基准的亚线性遗憾。

**💡 创新点**

首次在随机环境下实现了相对于GBB基准的亚线性遗憾，证明了在联合估值分布具有有界密度的情况下，算法可以达到𝒪(T^3/4)的遗憾。

**🔧 技术方法**

使用了一种基于三阶段的算法，结合了利润收集、纯探索和GFT优化的策略。

**📊 数据集**

假设私有估值是来自固定联合分布的独立同分布样本，主要关注具有有界密度的分布。

**📈 对比分析**

与现有的WBB机制相比，GBB基准可以提高GFT，现有的WBB算法在GBB最优解下会遭受线性遗憾。本文的算法在GBB基准下实现了𝒪(T^3/4)的遗憾，证明了其性能的紧密性。

**⚠️ 局限性**

算法在处理未知可行性约束时面临挑战，且在没有有界密度假设的情况下，无法实现亚线性遗憾。

---

## 427. ShapeUP: Scalable Image-Conditioned 3D Editing

**arXiv ID:** 2602.05676 | [PDF](https://arxiv.org/pdf/2602.05676v1)

**作者:** Inbar Gat `[一作]` (Aigency), Daniel Cohen-Or `[通讯]` (Tel Aviv University)

**通讯引用:** 40760 | [OpenAlex ID](https://openalex.org/A5036688260)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 ShapeUP，一种基于图像提示的可监督 3D 编辑框架，通过在 native 3D 潜在空间中进行潜在‑到‑潜在的转换，实现全局与局部、结构一致且可扩展的编辑。

**💡 创新点**

创新点在于将 3D 编辑建模为基于图像的监督潜在翻译任务，消除显式掩码和多视图重建需求，支持 mask‑free 定位、细粒度控制，并通过训练实现对 3D 基础模型的可扩展性。

**🔧 技术方法**

使用 native 3D Diffusion Transformer（DiT）与 LoRA 适配、Step1X‑3D 图像‑>3D 基础模型、两阶段几何与纹理编辑、图像提示条件、分类器无指导（CFG）等技术。

**📊 数据集**

在 7430 个 Objaverse 纹理网格上构建的合成数据集（Parts 与 Distant Frames in Motion DFM 样本）上训练，并在包含 TRELLIS/Hunyuan 等来源的 24 个网格+100 条编辑条件的全局编辑基准集上进行评估。

**📈 对比分析**

与 EditP23（多视图传播）和 3DEditFormer（在 TRELLIS 上微调）进行对比，使用 CLIP、DINO、SSIM、LPIPS、CLIP‑Dir、Occluded Region Fidelity 等指标，ShapeUP 在编辑一致性、身份保持、遮挡区域保真度上均明显优于基线，并在用户研究中获得最高偏好率。

**⚠️ 局限性**

局限性包括训练数据规模有限且偏向封闭、以物体为中心的网格；需要大量监督样本；在编辑与身份保持之间仍存在权衡，未来需扩大数据多样性并深入探究引导参数的影响。

---

## 428. Making AI Agents Evaluate Misleading Charts without Nudging

**arXiv ID:** 2602.05662 | [PDF](https://arxiv.org/pdf/2602.05662v1)

**作者:** Swaroop Panda `[一作]` (Northumbria University), Swaroop Panda `[通讯]` (Northumbria University)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5112614062)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用GPT‑5.2代理在未提示的情况下评估10幅含图表垃圾的可视化，检验其是否能自发识别误导元素。

**💡 创新点**

首次评估AI代理在未提示情况下自动识别图表误导缺陷，并提出将审美/可读性与完整性检查结合的建议。

**🔧 技术方法**

采用GPT‑5.2语言模型和ChatGPT Plus接口，对可视化进行单轮BeauVis与PREVis评分。

**📊 数据集**

从公开可视化语料库随机抽取10幅已知有缺陷的图表。

**📈 对比分析**

未与人类评测对比，仅通过量表发现代理在审美和可读性评分上高度一致，却对完整性缺陷识别不足，显示出对误导元素的低敏感性。

**⚠️ 局限性**

样本量有限、缺乏人类对照、仅使用单一GPT‑5.2模型，且完整性检查仅通过间接量表推断，限制了结论的普适性。

---

## 429. End-to-End Compression for Tabular Foundation Models

**arXiv ID:** 2602.05649 | [PDF](https://arxiv.org/pdf/2602.05649v1)

**作者:** Guri Zabërgja `[一作]` (University of Freiburg), Josif Grabocka `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种端到端的上下文压缩框架TACO，用于加速表格基础模型的推理，将训练集压缩到仅占原始行数的4%以内，同时保持预测性能。

**💡 创新点**

创新点在于：① 通过学习的压缩模块在Transformer前端一次性压缩训练数据，显著降低注意力层的计算与内存复杂度；② 端到端联合训练压缩器与预测器，使压缩后的表示能最佳匹配预测器；③ 兼容KV缓存与分块拼接策略，使模型能够处理百万级行数的数据集。

**🔧 技术方法**

使用Transformer（TabPFN v2类）双向行列注意力、混合精度训练、AdamW、cosine退火、梯度裁剪；构建压缩器与预测器的两层残差MLP；利用合成SCM数据做Meta‑learning先验；实现KV缓存和分块拼接。

**📊 数据集**

实验数据集包括：TabArena分类基准（26个二分类+多分类数据集，36个≤10类数据集），用于训练的合成SCM数据集（约82M个），以及1.5M行的MetroPT‑3用于大规模分块压缩验证。

**📈 对比分析**

与无压缩的预测器(POT)、kNN采样、随机采样以及最新Transformer基线进行对比。TACO在4%压缩率下实现53×推理速度提升、94%内存减少，且在ROC‑AUC上与POT无显著差异；在大规模数据上仍保持优于基线的性能。

**⚠️ 局限性**

局限性包括：训练成本高（15天、8张H100 GPU）；压缩率需在合理范围内，极端压缩时性能尚未充分验证；对极大表仍可能受限于硬件；依赖Transformer架构，未完全解决分布式或多节点部署的细节。

---

## 430. UAV Trajectory Optimization via Improved Noisy Deep Q-Network

**arXiv ID:** 2602.05644 | [PDF](https://arxiv.org/pdf/2602.05644v1)

**作者:** Zhang Hengyu `[一作]` (Wenzhou Kean University), Zhong Zhuoqing `[通讯]` (Wenzhou Kean University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究无人机在含障碍、信号衰减的 15×15 网格环境中轨迹优化问题，提出一种改进的噪声深度 Q 网络（Improved Noisy DQN）来实现自主导航。

**💡 创新点**

创新点包括：1）将残差噪声线性层与可调 Gaussian 噪声调度机制结合，实现状态相关且可学习的探索；2）通过软目标网络更新和双 DQN 估计降低过估计与目标漂移；3）采用学习率预热 + 余弦退火、损失平滑等技巧提升训练稳定性。

**🔧 技术方法**

使用的技术包括：Noisy DQN（Factorized Gaussian 噪声、残差连接）、双 DQN、软目标网络更新、经验回放、学习率预热+余弦退火、损失平滑、性能感知噪声调度、动作空间为离散五个动作。

**📊 数据集**

实验数据集为自行搭建的 15×15 网格仿真环境，包含结构化障碍分布和基站信号衰减模型；未使用公开数据集。

**📈 对比分析**

与标准 DQN、标准 NoisyNet DQN 和 Double DQN 在同一环境下对比，利用总奖励和完成任务所需步数评估。结果表明改进模型收敛最快，奖励接近 100，完成任务所需步数稳定在 28 步，显著优于其他三种算法。

**⚠️ 局限性**

局限性：仅针对离散动作空间和单步 TD 学习；未验证连续控制或高维状态下的性能；缺乏多步价值估计和序列策略，泛化能力受限。

---

## 431. UniSurg: A Video-Native Foundation Model for Universal Understanding of Surgical Videos

**arXiv ID:** 2602.05638 | [PDF](https://arxiv.org/pdf/2602.05638v1)

**作者:** Jinlin Wu `[一作]` (Center for Artificial Intelligence and Robotics Hong Kong Institute of Science and Innovation Chinese Academy of Sciences), Zhen Lei `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 26624 | [OpenAlex ID](https://openalex.org/A5109299788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 UniSurg，一种面向外科视频的通用基础模型，专注于从像素重建转向运动感知的潜在预测。

**💡 创新点**

创新点在于三大技术改进：运动引导的潜在预测、时空亲和自蒸馏以及特征多样性正则化，解决了噪声干扰、时空一致性和特征崩溃问题。

**🔧 技术方法**

核心技术包括基于 V-JEPA 的潜在空间预测、运动梯度加权损失、全局亲和矩阵自蒸馏和方差/协方差正则化，并结合 EMA 目标网络和多尺度卷积 Transformer 架构。

**📊 数据集**

使用了 UniSurg-15M 数据集——3,658 小时、50 来源、13 个解剖区域、100+ 术式的多机构外科视频集合，作为大规模无监督预训练语料。

**📈 对比分析**

与 13 种基线模型（包括 DINOv3、VideoMAE、EndoViT、EndoFM、GSViT、SurgeNetXL、SurgVLP 等）比较，在 8 个工作流程识别基准、动作三元组识别、技能评估、息肉分割和深度估计等 17 个任务上均取得显著提升，部分任务如 EgoSurgery 工作流程识别提升 14.6% F1，CholecT50 三元组 mAP 达 39.54%。

**⚠️ 局限性**

局限在于仍缺乏部分手术类型和机构的代表性，模型在极端光照/视角变换下的鲁棒性待进一步提升，且对复杂多器械交互的细粒度理解仍有提升空间。

---

## 432. CASTLE: A Comprehensive Benchmark for Evaluating Student-Tailored Personalized Safety in Large Language Models

**arXiv ID:** 2602.05633 | [PDF](https://arxiv.org/pdf/2602.05633v1)

**作者:** Rui Jia `[一作]` (East China Normal University), Min Zhang `[通讯]` (East China Normal University)

**通讯引用:** 59811 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了CASTLE基准，用于评估大型语言模型在教育场景下对学生个性化安全的表现。

**💡 创新点**

通过构造14维学生属性与15类教育风险的两层分类，并设计风险敏感度、情感共情、学生契合度三维指标，实现了对个体化安全的系统评估。

**🔧 技术方法**

采用循环多模型协同生成、逻辑约束规则、双语提示与自动评判器（Claude‑Haiku‑4.5）等技术。

**📊 数据集**

构建92,908条中英双语场景，覆盖7–22岁学生的多维属性。

**📈 对比分析**

对18款SOTA LLM进行评分，平均安全得分仅约2.3/5，显示普遍缺乏个性化安全保障。

**⚠️ 局限性**

仅限单轮问答、缺少多轮对话和交互性，且评估依赖自动评判，可能存在偏差。

---

## 433. Rewards as Labels: Revisiting RLVR from a Classification Perspective

**arXiv ID:** 2602.05630 | [PDF](https://arxiv.org/pdf/2602.05630v1)

**作者:** Zepeng Zhai `[一作]` (Xiaohongshu Inc.), Yuan Lu `[通讯]` (Xiaohongshu Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将可验证奖励重新定义为类别标签，将强化学习与可验证奖励（RLVR）任务转化为分类问题进行策略优化；

**💡 创新点**

创新点在于识别并解决GRPO等方法存在的梯度误分配与梯度支配问题，提出REAL框架通过软max交叉熵与锚点对数几率实现梯度单调有界分配，提升训练稳定性与性能；

**🔧 技术方法**

采用软max交叉熵损失（及可选的二元交叉熵）、锚点对数几率、温度调节以及无KL正则的自适应梯度裁剪技术；

**📊 数据集**

在DeepScaleR-Preview数据集（约4万道可验证推理题）上进行训练，并在AIME 2024/25、MATH 500、AMC 23、Minerva、O-Bench六大数学推理基准上评估；

**📈 对比分析**

与GRPO、DAPO、GSPO、TRPA等主流RLVR基线比较，REAL在1.5B模型上平均提升约6.7% Pass@1（相较DAPO），在7B模型上更是超越GSPO约1.7%，训练过程无KL惩罚即可保持稳定；

**⚠️ 局限性**

局限性包括：需要可验证的规则奖励，无法直接应用于无规则评价任务；对温度τ的选择敏感；在非推理任务或更大规模模型上的泛化尚待进一步验证。

---

## 434. Smoothed aggregation algebraic multigrid for problems with heterogeneous and anisotropic materials

**arXiv ID:** 2602.05686 | [PDF](https://arxiv.org/pdf/2602.05686v1)

**作者:** Max Firmbach `[一作]` (Universität der Bundeswehr München), Matthias Mayr `[通讯]` (Universität der Bundeswehr München)

**通讯引用:** 1684 | [OpenAlex ID](https://openalex.org/A5077454772)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于材料张量的强耦合度量，用于改进 Smoothed Aggregation AMG 在异质和各向异性材料问题上的收敛性能。

**💡 创新点**

创新点在于将材料张量信息直接嵌入强耦合度量，使得在粗化过程中能准确识别跨材料界面的弱耦合和各向异性方向，从而生成更符合物理本质的粗网格。

**🔧 技术方法**

采用材料加权距离构造强耦合度量、点/剪切掉落准则、改进的聚合与延伸算子平滑、以及基于聚合的材料张量平均来生成多级 AMG 预条件器，并使用 Trilinos 的 ML/MLM/Ifpack 等包实现。

**📊 数据集**

使用学术测试（二维二维双材料跳跃和各向异性热扩散），以及工业真实案例（热活化电池和光伏单元），这些案例包含高对比度、强各向异性以及极度拉伸的网格。

**📈 对比分析**

与传统的 SA‑AMG（^sa、^dlap）以及距离基度量（^dlap）进行对比，实验显示材料加权度量在所有材料对比、网格细化与各向异性场景下迭代次数与总应用成本均显著低于对照组，且在大规模并行环境下保持良好的强/弱标度性能。

**⚠️ 局限性**

局限性包括：需手动选择合适的掉落容忍度（θ）以避免过度掉落导致收敛不稳；材料加权度量在设置阶段的计算量和内存占用略高；目前仅在标量 Poisson‑type PDE 以及单体电池/光伏模型上验证，尚未扩展到向量型方程或多物理耦合系统。

---

## 435. Bagging-Based Model Merging for Robust General Text Embeddings

**arXiv ID:** 2602.05787 | [PDF](https://arxiv.org/pdf/2602.05787v1)

**作者:** Hengran Zhang `[一作]` (Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 20597 | [OpenAlex ID](https://openalex.org/A5029998682)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地比较了多任务训练的调度策略，并提出了基于Bagging的鲁棒模型合并（BRMM）方法，用于构建既能提升OOV泛化又能高效增量更新的通用文本嵌入模型。

**💡 创新点**

创新点包括：①揭示任务冲突极小，批级打乱训练是最优策略；②通过在不同采样子集上并行训练多模型并合并成单模型，既保留了集成的鲁棒性，又消除了推理时的多模型开销；③将此框架扩展到增量学习场景，利用核心子集与新数据训练轻量更新模型再合并，从而实现低成本持续升级。

**🔧 技术方法**

使用的技术包括：批级打乱、数据集/任务级顺序训练、两阶段训练；模型合并方法（Multi‑SLERP、Task Arithmetic、TIES、SCE 等）；Bagging 采样与参数空间融合；对大型 LLM（Qwen3‑4B、Qwen3‑0.6B）采用 LoRA 微调。

**📊 数据集**

主要数据集为约2.8M多任务语料，涵盖检索、分类、聚类、语义文本相似度、重新排序、摘要等，来自 MTEB、RTEB、MTEB(Code) 及其多语言与代码检索子集。

**📈 对比分析**

对比方法以 MTEB(Eng v2)、RTEB(beta) 与 MTEB(Code v1) 的 Mean(Task) 为评估指标；批级打乱在所有模型规模下均优于其他调度策略；BRMM 在 OOD 以及 In‑Domain 上提升约 5–10% 评分，并在增量学习中把训练成本降低至原来 40% 以内，达到或超过最佳基线。

**⚠️ 局限性**

局限性包括：①对 OOD 泛化仍有提升空间；②当前合并采用 Multi‑SLERP，尚需针对通用嵌入开发更合适的合并算法；③需调参的采样比例和模型数量；④实验仅验证在基准任务上的表现，未系统评估在检索增强生成等下游应用中的效果。

---

## 436. Different Time, Different Language: Revisiting the Bias Against Non-Native Speakers in GPT Detectors

**arXiv ID:** 2602.05769 | [PDF](https://arxiv.org/pdf/2602.05769v1)

**作者:** Adnan Al Ali `[一作]` (Charles University), Jindřich Libovický `[通讯]` (Charles University)

**通讯引用:** 1181 | [OpenAlex ID](https://openalex.org/A5061045500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对捷克语环境下LLM生成文本检测器是否存在对非母语作者的偏见进行了系统研究，并通过熵分析证明非母语写作文本并非普遍更易被误判为机器生成，进而探讨了检测器对熵的依赖性；

**💡 创新点**

创新点在于：①首次以捷克语为研究对象检验检测器的语言偏见；②提出使用熵（与困惑度相关）而非困惑度本身作为偏见评估指标；③在多域、多模型的检测器上开展跨域鲁棒性和偏见量化实验；

**🔧 技术方法**

采用Llama 3.2 1B作为熵评估器；构建TF‑IDF+朴素贝叶斯、Fine‑tuned RobeCzech（RoBERTa‑like）模型，并引入随机噪声增强；使用商业API Plagramme进行对比；

**📊 数据集**

使用SYNv9新闻/文学语料、Wiki和新闻爬虫生成与原始对照集；非母语学生论文集AKCES 3（450篇）和熟练非母语（29篇）；本土学生论文集AKCES 1（450篇、29篇）；学术摘要集Pre‑GPT与Post‑GPT；

**📈 对比分析**

通过准确率、FPR、FNR三项指标在源域与跨域（非母语、母语、学术、新闻等）上评估；发现检测器在训练域几乎完美，但跨域性能显著下降；商业检测器整体表现最佳且无显著偏见；熵与检测器输出的相关系数弱（|ρ|≤0.2），表明检测器不主要依赖熵；

**⚠️ 局限性**

局限性包括：非母语熟练作者样本不足（仅29篇）；仅使用少数生成模型（GPT‑4o、GPT‑4o‑mini、Llama 3.1 405B）导致通用性受限；自研检测器未达SoTA水平；对商业检测器的黑箱分析受限于API。

---

## 437. LongR: Unleashing Long-Context Reasoning via Reinforcement Learning with Dense Utility Rewards

**arXiv ID:** 2602.05758 | [PDF](https://arxiv.org/pdf/2602.05758v1)

**作者:** Bowen Ping `[一作]` (Peking University), Baobao Chang `[通讯]` (Peking University)

**通讯引用:** 5755 | [OpenAlex ID](https://openalex.org/A5021459300)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LongR 框架，通过在 LLM 推理过程中动态交替“思考-阅读”与上下文检索，并引入基于相对信息增益的稠密奖励，显著提升在长上下文推理任务上的性能。

**💡 创新点**

创新点包括：①动态 Think-and-Read 机制，使检索与推理交错自然进行；②利用信息论的相对信息增益计算上下文价值，形成稠密奖励；③采用渐进式课程学习，避免构造长文档专用训练数据；④用冻结的验证模型估算信息增益，既高效又可解释。

**🔧 技术方法**

技术手段：强化学习（如 DAPO、GSPO、CISPO 等）与监督微调相结合；信息论指标（相对信息增益）作为奖励；稠密奖励归一化；渐进式上下文长度训练；思考-阅读交错策略。

**📊 数据集**

使用的数据集包括：LongBench v2、RULER、InfiniteBench、NIAH 以及官方长文本数据，涵盖多种长文档推理与对话场景。

**📈 对比分析**

与官方基线、复现模型以及仅使用稀疏奖励的 RL baseline 对比，LongR 在 LongBench v2 上整体提升约 9%，Hard/Long 子集提升显著；在 RULER 上提升 6–17%；在 InfiniteBench 上提升 3–5%；不同 RL 算法（DAPO、GSPO、CISPO）均能获得加速效果，证明其通用性。

**⚠️ 局限性**

局限性：①奖励计算依赖冻结验证器，验证器规模越大效果越好但计算成本上升；②在极长文本（> 128k）下仍需进一步验证效率；③对多模态或多文档场景的适用性尚未评估；④在非中文或低资源语言的鲁棒性待测试。

---

## 438. Generalized Pinsker Inequality for Bregman Divergences of Negative Tsallis Entropies

**arXiv ID:** 2602.05744 | [PDF](https://arxiv.org/pdf/2602.05744v1)

**作者:** Guglielmo Beretta `[一作]` (Ca' Foscari Venezia), Roberto Colomboni `[通讯]` (Milano)

**通讯引用:** 54 | [OpenAlex ID](https://openalex.org/A5012252221)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出并证明了一条关于由负α‑Tsallis 熵生成的 Bregman 散度的精确 Pinsker 不等式，给出了最优常数 C_{α,K}，并阐明了不同 α、K、维度以及奇偶性所导致的相位转变。

**💡 创新点**

创新点在于：1）首次得到所有 β‑divergence（包括 Itakura‑Saito）对应的最优 Pinsker 常数；2）揭示了 α>2 时多类别问题无统一常数、仅在二分类或受限条件下可行；3）通过几何视角解析常数随维度与 α 的变化规律，并给出具体解析表达式。

**🔧 技术方法**

采用了 Bregman 散度、Hessian 分析、变分方法、极值与 Jensen/凹凸性质、奇偶性分解等数学技术，最终以闭式最优化计算得到 C_{α,K}。

**📊 数据集**

本研究为纯理论工作，未使用任何公开数据集进行实验验证。

**📈 对比分析**

由于缺少实验，对方法性能的比较基准没有提供；但理论上已给出最优常数，表明在可行区间内可获得最优的从失误风险到总变差（或 0–1 误差）的转换。

**⚠️ 局限性**

局限性包括：① 对 α>2 且 K≥3 的多类别情况无法得到正的统一常数；② 结果仅在概率单纯形的相对内部成立，边界点需极限处理；③ 在实际学习算法中需要额外约束（如分布下界）才能应用该不等式。

---

## 439. CSRv2: Unlocking Ultra-Sparse Embeddings

**arXiv ID:** 2602.05735 | [PDF](https://arxiv.org/pdf/2602.05735v1)

**作者:** Lixuan Guo `[一作]` (Stony Brook University), Chenyu You `[通讯]` (Stony Brook University)

**通讯引用:** 4476 | [OpenAlex ID](https://openalex.org/A5076320750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的稀疏嵌入训练方案 CSRv2，使得在极低稀疏度（k≤4）下的嵌入可用，显著提升嵌入质量与推理效率。

**💡 创新点**

创新点包括：① 采用逐步稀疏（k‑annealing）课程学习，缓解死神经；② 用自然监督（supervised contrastive loss）替代无监督对比，提升下游对齐；③ 允许全模型微调，增强跨域泛化；④ 通过上述方法在 ultra‑sparse 场景实现 14% 的准确率提升，且大幅提升速度和内存效率。

**🔧 技术方法**

核心技术：k‑annealing、监督稀疏对比学习、稀疏自编码器（SAE）、全链路微调、跨域多任务训练、GPU 稀疏算子加速。

**📊 数据集**

使用的数据集与模型：e5‑Mistral‑7B backbone + MTEB（6类任务），Qwen3‑Embedding‑4B，ImageNet‑1k，GraphRAG benchmark，SPLADEv3 进行对比评测。

**📈 对比分析**

在相同 backbone、相同训练数据与超参数下，CSRv2 在 k=2/4 的超稀疏设置下均优于 CSR 与 MRL，提升 7%–14% 的任务准确率；在 1M 数据库上实现 7× 的检索速度提升和 300× 的计算/内存节省；在 Qwen3 与 SPLADE 对比实验中亦保持或超越现有方法。

**⚠️ 局限性**

局限性：k=1 的极端稀疏仍无法克服死神经与性能骤降；对更低维度的理论与方法尚未成熟；对不同 backbone 的泛化仍需进一步验证。

---

## 440. Adaptive Global and Fine-Grained Perceptual Fusion for MLLM Embeddings Compatible with Hard Negative Amplification

**arXiv ID:** 2602.05729 | [PDF](https://arxiv.org/pdf/2602.05729v1)

**作者:** Lexiang Hu `[一作]` (Peking University), Zhouchen Lin `[通讯]` (Peking University)

**通讯引用:** 25993 | [OpenAlex ID](https://openalex.org/A5016399094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种适用于多模态大型语言模型的自适应全局与细粒度感知融合框架 AGFF-Embed，通过可学习的提示令 MLLM 生成多维度嵌入并使用 logsumexp 聚合，实现对多模态匹配任务的高效对齐。

**💡 创新点**

结合全局与细粒度感知的多模态嵌入融合；使用可学习提示令 MLLM 自主生成细粒度嵌入；采用平滑的 logsumexp 聚合兼容 Explicit Gradient Amplification（EGA）进行硬负样本强化。

**🔧 技术方法**

多模态大型语言模型（QQMM）、可学习提示与特殊嵌入标记、logsumexp 相似度聚合、EGA 硬负样本梯度放大、GradCache 反向传播缓存、LoRA 微调。

**📊 数据集**

MMEB 基准（36 个子任务，训练集与测试集分离）和 MMVP-VLM 基准（9 个细粒度子任务）。

**📈 对比分析**

与仅使用 MMEB‑train 训练的 CLIP 与 MLLM 嵌入模型进行对比，AGFF‑Embed 在 MMEB 上整体得分 74.9%，比 QQMM 提升 2.4%；在 MMVP‑VLM 上零样本平均得分 61.5%，在 6 个子任务均夺得榜首，显示出优异的全局与细粒度理解能力。

**⚠️ 局限性**

对硬负样本的 EGA 依赖于已训练的 MLLM 对难度的判断，在训练早期对感知模式识别不够准确；需要较大的 batch size 与显存；模型仅在 MMEB‑train 上训练，缺乏跨域或更大规模数据验证。

---

## 441. Anchored Policy Optimization: Mitigating Exploration Collapse Via Support-Constrained Rectification

**arXiv ID:** 2602.05717 | [PDF](https://arxiv.org/pdf/2602.05717v1)

**作者:** Tianyi Wang `[一作]` (Beijing University of Posts and Telecommunications), Guanhua Chen `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 6515 | [OpenAlex ID](https://openalex.org/A5100665987)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Anchored Policy Optimization（APO），通过支持覆盖（Support Coverage）和弹性恢复机制解决 RLVR 中的 Recursive Space Contraction（RSC）问题，提升模型在数学推理任务上的准确率与多样性。

**💡 创新点**

核心创新是从全局形状匹配（KL 正则化）转向局部支持覆盖，定义“安全流形”（Safe Manifold），引入拉伸-收缩比率正则化（ratio rectification）实现梯度对齐，消除梯度冲突，并在错误时主动回归有效路径。

**🔧 技术方法**

技术实现基于 PPO/GRPO 的强化学习框架，加入 Anchor Ratio、Push/Pull 复合正则项；使用重要性采样计算 Anchor Ratio；通过稀疏化处理降低全词表 KL 计算复杂度；实验使用 Llama‑3.2‑3B‑Instruct、Qwen2.5‑7B、Qwen2.5‑Math‑7B 等大型语言模型。

**📊 数据集**

评测数据集包括 AIME 2024/2025、AMC 2023、MATH‑500、Minerva Math 等五大数学推理基准，采用 Pass@1（效率）和 Pass@K（多样性）两项指标。

**📈 对比分析**

与基线 GRPO、GRPO‑KL、NSR 等方法对比，APO 在 Pass@1 上提升约 6% 左右，同时保持甚至提升 Pass@K，突破传统的效率‑多样性权衡，显著恢复被 RSC 抑制的有效推理路径。

**⚠️ 局限性**

局限性在于安全流形的大小（K）与 Pull/Push 系数需要手动调参；在极大词表或极端多样性任务中仍可能出现过度收敛；以及对参考模型分布质量的依赖，若参考模型本身噪声过大，安全流形定义可能不够可靠。

---

## 442. Cost-Efficient RAG for Entity Matching with LLMs: A Blocking-based Exploration

**arXiv ID:** 2602.05708 | [PDF](https://arxiv.org/pdf/2602.05708v1)

**作者:** Chuangtao Ma `[一作]` (Aalborg University), Paul Groth `[通讯]` (University of Amsterdam)

**通讯引用:** 26594 | [OpenAlex ID](https://openalex.org/A5034924491)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种成本高效的检索增强生成（RAG）架构CE‑RAG4EM，用于大规模实体匹配

**💡 创新点**

将阻塞技术与批量检索、批量生成结合，显著减少检索与推理开销，并在保持或提升匹配质量的同时降低总时延

**🔧 技术方法**

使用基于向量的检索、知识图谱（Wikidata）子图/三元组扩展、指令调优提示、批量推理等技术

**📊 数据集**

在九个公开实体匹配基准数据集（如Abt‑Buy、Amazon‑Google、DBLP‑ACM、Fodors‑Zagats等）上进行实验

**📈 对比分析**

与传统LLM直接提示、基于PLM的监督方法以及普通RAG进行对比，CE‑RAG4EM在大多数数据集上取得与PLM相近或更优的F1，并且每对查询的平均运行时间显著低于单独查询的RAG和LLM提示

**⚠️ 局限性**

检索质量和上下文相关性仍是瓶颈；批量化虽节省成本但可能导致跨对的干扰；对超大规模知识图谱的扩展仍需要进一步优化

---

## 443. How to Achieve the Intended Aim of Deep Clustering Now, without Deep Learning

**arXiv ID:** 2602.05749 | [PDF](https://arxiv.org/pdf/2602.05749v1)

**作者:** Kai Ming Ting `[一作]` (Nanjing University), Hang Zhang `[通讯]` (Nanjing University)

**通讯引用:** 36653 | [OpenAlex ID](https://openalex.org/A5100456227)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对深度聚类（Deep Clustering）的目标进行了重新定义，并指出其常用的“点对点相似度”定义所带来的限制。作者通过实验和理论分析，揭示了 DEC/IDEC 等主流深度聚类方法在克服 k‑means 的局限性方面仍存在同样的问题，并提出了基于“Cluster-as-Distribution（CaD）”的聚类框架，利用分布核实现聚类，无需深度学习即可达到深度聚类的预期目标。

**💡 创新点**

创新点主要有三：①重新提出聚类定义，将聚类视为从不同分布中采样的 i.i.d. 点集合，摆脱了对点对点相似度的依赖；②证明了 DEC/IDEC 等深度聚类方法并未真正突破 k‑means 的形状、大小、密度等基本局限；③基于分布核的 CaD 聚类方法，能够在合成、高维生物医学与图像数据上实现或优于深度聚类，且不需要复杂的深度学习。

**🔧 技术方法**

使用的技术包括：
- 对 DEC、IDEC 的自编码器与 KL 损失进行理论与实验评估；
- 采用分布核（distribution kernel）进行聚类，形成 CaD 方法；
- 对比 k‑means、Spectral Clustering、Kernel Bounded Clustering (KBC)、Contrastive Clustering 等传统与深度聚类方法；
- 使用 NMI（Normalized Mutual Information）指标评估聚类效果。

**📊 数据集**

使用的数据集涵盖三大类：
- 合成数据集（如 2Crescents、Diff‑Sizes、AC）用于展示聚类形状、大小、密度的极端情况；
- 高维单细胞与空间转录组数据（Tutorial、Tonsil、Airway、Crohn、DLPFC 等）
- 高维图像数据集（CIFAR‑10、ImageNet‑10、MNIST、USPS、COIL‑20、STL‑10、ImageNet‑Dogs 等）。

**📈 对比分析**

比较方法采用 10 次随机重跑后取平均 NMI。结果显示：
- 在合成数据集上，CaD 方法在所有三种极端场景下均得到 NMI=1 的完美聚类，而 DEC/IDEC、Contrastive Clustering、KBC 等深度聚类方法只能得到 0.4–0.6 之间；
- 在高维生物医学与图像数据集上，CaD 方法往往与深度聚类相当甚至更好（如在 Tutorial 数据集上 KBC 的 NMI 为 0.87，远高于 DEC/IDEC 的 0.01/0.02）；
- 在传统 k‑means 上，虽然在部分图像数据集（如 COIL‑20）表现不错，但整体受初始化敏感、无法处理复杂形状，表现远逊于 CaD 与深度聚类。

**⚠️ 局限性**

limitations：
- 深度聚类方法仍然基于点对点相似度，未充分利用分布信息，导致无法克服任意形状、大小、密度的聚类；
- 本研究对 DEC/IDEC 的理论证明仍不完整，仅通过经验实验指出其局限；
- CaD 方法目前采用贪婪搜索，缺乏更精细的优化策略；
- 论文未对 CaD 在更大规模数据集或实时系统中的可扩展性进行深入评估。

---

## 444. FiMI: A Domain-Specific Language Model for Indian Finance Ecosystem

**arXiv ID:** 2602.05794 | [PDF](https://arxiv.org/pdf/2602.05794v1)

**作者:** Aboli Kathar `[一作]` (National Payments Corporation of India), Yatharth Dedhia `[通讯]` (National Payments Corporation of India)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了针对印度数字支付生态的专用语言模型 FiMI（包括基础版和指令版），通过连续预训练、指令微调和领域监督微调，支持英、印、 Hinglish 三语种对话并具备工具调用能力。

**💡 创新点**

创新点在于：① 多阶段训练管道将海量印度金融语料与通用数据无缝融合；② 设计专属工具调用与合规安全框架；③ 构建大规模合成对话数据并实施两阶段结构/语义校验，确保工具调用准确与合规。

**🔧 技术方法**

采用 Mistral Small 24B 作为基础模型，配合 DeepSpeed ZeRO‑3、FlashAttention‑2、bfloat16、Cosine 学习率调度与序列打包；使用 LLM 微调、指令微调、监督微调；合成数据通过 Gemma 3 27B 与提示工程生成；评估工具包括 lm‑eval、DeepEval、KL divergence 与 perplexity。

**📊 数据集**

数据集涵盖 68 B 令牌的印度金融文本、英语/印语混合文本、通用英语、数学、代码、金融、Hinglish 等多语料；合成问答对、HellaSwag Finance、Finance Reasoning、MMLU Finance、工具调用示例以及社交媒体查询样本。

**📈 对比分析**

通过与 Mistral Small 24B Base / Instruct 的基准对比，FiMI Base 在金融推理基准提升约 20%，FiMI Instruct 在工具调用任务提升 87%；在多语言与工具调用的自定义基准上表现领先，并保持与同等规模通用模型相近的通用基准性能。

**⚠️ 局限性**

局限性包括：需要大量人工审核合成数据；工具调用准确率受模型规模与训练样本多样性影响；跨语言多模态推理仍有限；部署成本与合规要求较高；对极端稀有场景的泛化能力尚待提升。

---

## 445. Scalable and General Whole-Body Control for Cross-Humanoid Locomotion

**arXiv ID:** 2602.05791 | [PDF](https://arxiv.org/pdf/2602.05791v1)

**作者:** Yufei Xue `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18104 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种跨人形机器人全身控制的通用策略，能够在一次训练后实现零样本迁移到多种不同结构的机器人。

**💡 创新点**

创新点在于（1）提出物理一致的形态随机化，保证随机化样本可模拟且保持物理可行；（2）通过统一的全局关节空间和图网络构造实现不同机器人之间的语义对齐；（3）使用Transformer/GNN编码器捕捉机器人拓扑，学习跨形态运动先验。

**🔧 技术方法**

技术包括物理一致的形态随机化、全局关节映射、机器人拓扑图构造、Transformer或GCN编码器、联合状态估计器、强化学习（PPO）以及针对部分可观测环境的监督回归。

**📊 数据集**

数据集：在仿真环境中随机生成12种不同形态的人形机器人样本，随后在7个真实机器人（包括不同关节数、动力学、结构的硬件平台）上进行零样本和微调实验；所有实验使用统一命令空间和机器人观测。

**📈 对比分析**

与MetaMorph、MorAL、Naive Random等跨形态基线以及单独专家策略比较，通用策略在零样本下实现100%生存率、指令跟踪误差低于专家的10%；在微调后可达到甚至超过专家约10%的性能，且收敛速度最快。

**⚠️ 局限性**

局限性：依赖共享的命令语义，难以直接支持需要形态感知的高表达式控制（如精确运动捕捉或形态专属的动作重定向）。

---

## 446. Distributional Reinforcement Learning with Diffusion Bridge Critics

**arXiv ID:** 2602.05783 | [PDF](https://arxiv.org/pdf/2602.05783v1)

**作者:** Shutong Ding `[一作]` (ShanghaiTech University), Ye Shi `[通讯]` (ShanghaiTech University)

**通讯引用:** 8315 | [OpenAlex ID](https://openalex.org/A5017327360)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了基于扩散桥模型的分布式强化学习方法 DBC，用来替代传统 Critic，直接学习 Q 值的逆累计分布函数，从而得到更准确的价值分布。

**💡 创新点**

①使用扩散桥模型对 Q 值逆 CDF 进行参数化，避免了离散化与高斯退化问题；②提出锚点损失和积分一致离散化技术，有效修正离散误差并加速收敛；③将 DBC 作为 plug‑and‑play 组件，能无缝集成到 SAC、TD3、QVPO 等主流算法中。

**🔧 技术方法**

扩散桥模型（UniDB）、逆 CDF 参数化、量化损失、锚点损失、DropTop 聚合、积分一致离散化、Huber/量化损失等。

**📊 数据集**

MuJoCo 连续控制 benchmark：Ant‑v5、HalfCheetah‑v5、Hopper‑v5、Humanoid‑v5、Walker2d‑v5。

**📈 对比分析**

与 SAC、DSAC、IQN、TQC、Value Diffusion、Value Flows 等基线进行对比。DBC 在 5 个任务中均取得最高或最接近最高回报，提升约 10%–24%；在不同 actor（SAC、TD3、QVPO）中替换 Critic 也均显著提高性能。

**⚠️ 局限性**

①需要多步离散化（M=5 最佳，M>5 稳定性下降）；②锚点权重需精细调参，过大会导致偏差；③积分一致离散化仍有数值误差；④实验仅覆盖 MuJoCo 连续控制任务，缺乏更大规模或离散动作任务验证；⑤计算开销相对较高。

---

## 447. How Controlling the Variance can Improve Training Stability of Sparsely Activated DNNs and CNNs

**arXiv ID:** 2602.05779 | [PDF](https://arxiv.org/pdf/2602.05779v1)

**作者:** Emily Dent `[一作]` (Mathematical Institute), Jared Tanner `[通讯]` (Mathematical Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过把 Gaussian 过程的方差参数 q* 视作可调控的初始化参数，扩展 Edge‑of‑Chaos 理论，证明增大 q* 能显著提升使用稀疏激活函数（如 CReLU_τ,m）的深层网络的训练稳定性、可表达性与收敛速度，并在 DNN/CNN 上实现高达 90% 的隐藏层稀疏率。

**💡 创新点**

创新点在于：① 把 q* 作为调节稀疏激活网络稳定性的关键参数；② 分析了 q* 对方差映射对称性、二阶导数及 χ₁ 的敏感度的影响；③ 通过有限维修正理论证明增大 q* 能抑制方差波动；④ 实验验证了该方法在极高稀疏率下仍能保持高准确率并加速收敛。

**🔧 技术方法**

使用技术包括：Edge‑of‑Chaos 初始化、Gaussian 过程分析、方差映射与关联函数的解析推导、有限维修正理论、梯度下降训练、随机梯度下降（SGD）与余弦学习率调度。

**📊 数据集**

使用的数据集为 MNIST（DNN）和 CIFAR‑10（CNN），网络宽度分别为 300，深度分别为 100 与 50，激活函数为 CReLU_τ,m（τ=1, m 可调），并对比 ReLU 等传统激活。

**📈 对比分析**

比较方法：在固定稀疏率与方差斜率 V'(q*) 下，对不同 q*（1、2、3）进行实验；对比 ReLU、不同 q* 的 CReLU 以及不同 V'(q*) 的情况。结果显示：增大 q* 能显著提高训练准确率（接近或超过 90%）、缩短收敛步数、降低对超参数的敏感度，并在 90% 稀疏率下保持高精度。

**⚠️ 局限性**

局限性包括：仅在简单的 DNN/CNN 架构上验证；未考虑 Transformer、RNN 等更复杂结构；仅实验 CReLU_τ,m，其他稀疏激活仍需研究；缺乏对能耗或实际推理时间的实测；对不同数据集与任务的泛化能力尚待进一步验证。

---

## 448. Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance

**arXiv ID:** 2602.05774 | [PDF](https://arxiv.org/pdf/2602.05774v1)

**作者:** Xiandong Zou `[一作]` (Ant Group), Pan Zhou `[通讯]` (Singapore Management University)

**通讯引用:** 15293 | [OpenAlex ID](https://openalex.org/A5100693197)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Variational Speculative Decoding (VSD)，通过变分推断训练草稿模型，使其在多路径推理中更频繁被目标模型接受，从而提升推理速度。

**💡 创新点**

将草稿训练重新表述为变分推断，利用 ELBO 优化路径级接受概率，并结合 MCMC‑EM、Adaptive Rejection Weighting (ARW) 与 Confidence‑Aware Regularization (CAR)。

**🔧 技术方法**

变分推断、EM‑MCMC 采样、ELBO、路径级效用、ARW、CAR、采样滤波器、对数接受概率等技术。

**📊 数据集**

LLM 任务使用 LLaMA‑3、Vicuna、DeepSeek；MLLM 任务使用 LLaVA；评测数据集包括 MT‑Bench、HumanEval、GSM8K、SQA Image、VQAv2、AI2D、ChartQA、TextVQA、Hallusion 等。

**📈 对比分析**

在同一硬件（RTX PRO 6000 / L‑40S）上与 EAGLE‑3、ViSpec、SPS 等基线对比，VSD 在 LLM 上平均提升 9.6% 的速度比（T=0）和 7.3%（T=1），在 MLLM 上提升 7.9%（速度）和 8.8%（接受长度），整体表现显著优于现有方法。

**⚠️ 局限性**

受限于 MCMC 采样规模（S≤40）难以进一步扩大；实验仅覆盖文本和视觉任务，尚未验证语音等其他模态的适用性。

---

## 449. FMPose3D: monocular 3D pose estimation via flow matching

**arXiv ID:** 2602.05755 | [PDF](https://arxiv.org/pdf/2602.05755v1)

**作者:** Ti Wang `[一作]` (École Polytechnique Fédérale de Lausanne), Mackenzie Weygandt Mathis `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 10116 | [OpenAlex ID](https://openalex.org/A5025196309)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于流匹配的条件生成模型FMPose3D，用来从单张图像的二维姿态快速推断三维姿态并生成多样化的可行解；

**💡 创新点**

创新点在于将3D姿态恢复视作条件分布传输问题，利用ODE驱动的速度场实现少量步骤高效采样，并引入基于重投影的后验期望聚合（RPEA）来融合多候选解；

**🔧 技术方法**

采用流匹配（Flow Matching）学习条件速度场，使用GCN+自注意力双分支网络作为特征提取器，并通过显式欧拉积分实现ODE求解；

**📊 数据集**

在Human3.6M、MPI-INF-3DHP、Animal3D和CtrlAni3D四个数据集上进行训练与评估；

**📈 对比分析**

与多种确定性和概率方法比较，单一假设下MPJPE降至47.3mm，40个假设下进一步到43.5mm，明显优于DiffPose、ProPose等前沿方法，且推理速度提升至150FPS以上；

**⚠️ 局限性**

局限性包括对极端遮挡的鲁棒性不足，重投影误差作为后验近似可能导致非最佳聚合，且在真实场景中的迁移性能仍需进一步验证。

---

## 450. A Bayesian Optimization-Based AutoML Framework for Non-Intrusive Load Monitoring

**arXiv ID:** 2602.05739 | [PDF](https://arxiv.org/pdf/2602.05739v1)

**作者:** Nazanin Siavash `[一作]` (University of Colorado), Armin Moin `[通讯]` (University of Colorado)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了 AutoML4NILM 框架，实现非侵入式负载监测的模型自动化选择与超参数调优。

**💡 创新点**

首次将贝叶斯优化与 AutoML 结合用于 NILM，并扩展至多种深度学习算法，形成可复用、可扩展的开源工具。

**🔧 技术方法**

使用贝叶斯优化（Hyperopt）、多种机器学习与深度学习模型（决策树、随机森林、GRU、LSTM、Seq2Point 等）进行调参。

**📊 数据集**

采用公开的 UK‑DALE 电力消耗数据集进行实验。

**📈 对比分析**

通过对比单模型训练与 AutoML 搜索后的结果，AutoML 选出的 Seq2Point 在 MAE 下降至 7.12、准确率约 97.98%，显著优于传统方法。

**⚠️ 局限性**

受限于计算资源，最多只能完成 30 次评估，无法充分搜索全部超参数空间，部分模型训练耗时过长导致实验中断。

---

## 451. Ethology of Latent Spaces

**arXiv ID:** 2602.05710 | [PDF](https://arxiv.org/pdf/2602.05710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 452. Neuro-Inspired Visual Pattern Recognition via Biological Reservoir Computing

**arXiv ID:** 2602.05737 | [PDF](https://arxiv.org/pdf/2602.05737v1)

**作者:** Luca Ciampi `[一作]` (Institute of Information Science and Technologies of the National Research Council of Italy), Giuseppe Amato `[通讯]` (Institute of Information Science and Technologies of the National Research Council of Italy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在体培养的大脑皮层神经网络上实现生物储存计算框架，利用高密度多电极阵列进行电刺激和记录，完成静态视觉模式识别。

**💡 创新点**

创新点在于将真实的活体神经网络作为物理储存器，直接利用其瞬时响应生成高维特征，并在生物体内完成计算。

**🔧 技术方法**

技术包括体培养皮层神经元、HD‑MEA电刺激与记录、脉冲参数校准、双阈值检测、归一化面积滤波、线性感知机读出层。

**📊 数据集**

使用的数据集包括自定义的点刺激、定向条纹、钟形数字十个类以及MNIST手写数字的子集200张。

**📈 对比分析**

与匹配条件的人工储存器对比，BRC在点与条纹任务上可达≈90–98%，钟形数字约70%，MNIST约30–35%，低于人工模型但显著高于随机猜测。

**⚠️ 局限性**

局限在于生物网络的日间可塑性与噪声导致表现随时间衰退、样本容量小、操作难度高、设备寿命受限。

---

## 453. Depth as Prior Knowledge for Object Detection

**arXiv ID:** 2602.05730 | [PDF](https://arxiv.org/pdf/2602.05730v1)

**作者:** Moussa Kassem Sbeyti `[一作]` (Karlsruhe Institute of Technology), Nadja Klein `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 1568 | [OpenAlex ID](https://openalex.org/A5009563869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DepthPrior 框架，利用深度信息作为先验知识，通过训练时的深度加权损失（DLW）和分层损失（DLS），以及推理时的深度自适应阈值（DCT）来提升物体检测性能，尤其是远距与小尺寸目标。

**💡 创新点**

创新点在于：① 将深度视为监督先验而非特征融合，避免模型改造与模态不平衡；② 设计批量级与图像级的深度加权/分层策略；③ 开发基于贝叶斯优化的 B‑spline 深度阈值学习，实现对不同距离的精准阈值调整。

**🔧 技术方法**

使用的技术包括：单目深度估计（Depth Anything）、指数加权损失、深度分层损失、贝叶斯优化求解 B‑spline 参数、YOLOv11 与 EfficientDet 检测器、NMS 与阈值后处理等。

**📊 数据集**

实验数据集涵盖 KITTI、MS COCO、VisDrone、SUN RGB‑D，覆盖地面、空中、室内等多种场景与距离分布。

**📈 对比分析**

与基线、深度融合、简单加权、无监督等方法对比，DepthPrior 在小目标 mAP_S 提升 4–9%，总体 mAP 提升 0.4–1.6%；在阈值优化上实现高正样本恢复同时几乎不增加误报，验证集与测试集均表现出显著的性能提升。

**⚠️ 局限性**

局限性包括：依赖深度估计的精度，深度误差会影响加权与阈值；需为不同数据集调节超参数（α、β、λ 等）；在某些配置下可能对大目标有轻微退化；验证集规模有限，可能影响阈值泛化；未在 Transformer 结构上评估。

---

## 454. Projected Boosting with Fairness Constraints: Quantifying the Cost of Fair Training Distributions

**arXiv ID:** 2602.05713 | [PDF](https://arxiv.org/pdf/2602.05713v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Kaveh Aryan `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于投影的公平 Boosting 方法（Projected Fair Boosting），通过将集成产生的指数权重分布投影到满足公平约束的凸集上，形成公平训练分布来调节弱学习器；

**💡 创新点**

创新点在于保持 AdaBoost 的指数损失递归不变，同时通过投影获得的 KL 散度量化“公平成本”，并给出明确的有效边界和指数损失上界；

**🔧 技术方法**

使用了 KL 投影、对偶优化、边界转移分析以及 AdaBoost 的镜像下降框架；

**📊 数据集**

在 Adult、German Credit、COMPAS 三个标准表格数据集上进行实验；

**📈 对比分析**

与 AdaBoost、预处理重加权、Exponentiated Gradient 等公平基线比较，实验显示在保持较高准确率的同时可显著降低均等机会（Equal Opportunity）和人口统计公平（Demographic Parity）误差；

**⚠️ 局限性**

局限在于公平约束仅是训练分布的代理，无法保证最终分类器的严格公平性；公平成本随公平松弛参数变小而增大，过度约束会导致有效边为负而提前停止；处理多组交叉公平时计算量随组数上升。

---

## 455. Fix Representation (Optimally) Before Fairness: Finite-Sample Shrinkage Population Correction and the True Price of Fairness Under Subpopulation Shift

**arXiv ID:** 2602.05707 | [PDF](https://arxiv.org/pdf/2602.05707v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Kaveh Aryan `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出在子群分布不变但比例变化的子群平移情景下，对公平性评价进行去混淆的评估协议，即先进行最佳收缩人口校正，再比较公平方法与该基线。

**💡 创新点**

创新点在于发现公平性提升与准确性之间的“免费午餐”往往是由训练数据比例失配引起的，并给出了有限样本下最优收缩校正公式及其理论与经验验证。

**🔧 技术方法**

使用了重要性加权、收缩重权、偏差-方差分析、James‑Stein 估计思想，以及正则化逻辑回归/梯度下降等技术。

**📊 数据集**

实验数据包括人工合成数据以及真实数据集 Adult（收入预测）和 COMPAS（再犯预测）。

**📈 对比分析**

通过与标准 ERM、完整重要性加权和收缩校正基线的对比，展示收缩校正在大多数情况下能获得更高准确率并消除公平性误导，公平约束在校正后表现出真实的准确性成本。

**⚠️ 局限性**

局限性包括假设子群内分布不变、已知目标比例以及只能处理比例失配的子群平移，无法覆盖协变量平移或标签平移等更一般的分布偏移。

---

## 456. Evaluating the impact of word embeddings on similarity scoring in practical information retrieval

**arXiv ID:** 2602.05734 | [PDF](https://arxiv.org/pdf/2602.05734v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 457. Classification Under Local Differential Privacy with Model Reversal and Model Averaging

**arXiv ID:** 2602.05797 | [PDF](https://arxiv.org/pdf/2602.05797v1)

**作者:** Caihong Qin `[一作]` (Indiana University), Yang Bai `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 253 | [OpenAlex ID](https://openalex.org/A5034695398)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在本地差分隐私（LDP）框架下，提出了一套通过迁移学习思路提升分类器性能的完整方法。

**💡 创新点**

创新点包括：①把私有学习重新诠释为迁移学习；②引入基于隐私化二值反馈的评估机制；③设计模型反转（Model Reversal）与模型平均（Model Averaging）两种技术来挽救低质量或负迁移的模型；④在功能数据上实现基于基函数投影的 LDP 分类器。

**🔧 技术方法**

核心技术：本地差分隐私机制（Laplace、随机响应）、隐私化二值反馈评估、模型反转、加权模型平均、功能数据的基函数投影与标准化、基于误差上界的理论分析。

**📊 数据集**

实验使用：①模拟数据；②公开向量数据（糖尿病预测、员工流失预测）；③功能数据（基于运动记录的心血管健康预测、TIMIT 语音音素分类）。

**📈 对比分析**

与传统 LDP 分类方法（直方图、投票、简单平均）及全量数据模型对比，MRMA 在低 ε（强隐私）下误分类率显著下降，随着 ε 增大性能趋近全量数据水平，且在功能数据上同样表现优异。

**⚠️ 局限性**

局限性：仅针对二分类任务；需要足够的训练/评估样本以保证评估准确；对多类别、回归及更复杂模型的推广尚待研究；模型反转与平均的阈值和加权策略依赖经验设定。

---

## 458. Data analysis of cloud virtualization experiments

**arXiv ID:** 2602.05792 | [PDF](https://arxiv.org/pdf/2602.05792v1)

**作者:** Pedro R. X. do Carmo `[一作]` (Universidade Federal de Pernambuco), Djamel Sadok `[通讯]` (Universidade Federal de Pernambuco)

**通讯引用:** 3167 | [OpenAlex ID](https://openalex.org/A5014967408)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在云计算环境中收集并构建了基于KVM、LXC和Docker的虚拟化网络实验数据集，并对RTT和CPU使用率进行测量。

**💡 创新点**

创新点在于公开完整实验数据集、结合多种虚拟化技术与参数进行系统化评估，并通过机器学习构建预测模型。

**🔧 技术方法**

使用了虚拟机/容器技术（KVM、LXC、Docker）、网络测量工具（ping、tcpdump、mpstat）、数据预处理（One-Hot编码、z-score标准化）、PCA、K‑Means聚类以及决策树/随机森林回归模型。

**📊 数据集**

使用的数据集为约160万条记录，包含7列特征和RTT、CPU使用率指标，已公开可下载。

**📈 对比分析**

通过Spearman相关、PCA+K‑Means可视化以及回归模型对比，结果表明Docker在CPU占用和RTT方面表现最佳，KVM+VirtIO驱动也具备较低延迟；随机森林模型在预测RTT上误差较低。

**⚠️ 局限性**

局限性包括仅评估了三种虚拟化技术，未涵盖如VMware、XEN等；实验环境为单一数据中心拓扑；频率与VM数的取值范围有限；模型对未见参数组合的泛化能力尚未充分验证。

---

## 459. TimelyFreeze: Adaptive Parameter Freezing Mechanism for Pipeline Parallelism

**arXiv ID:** 2602.05754 | [PDF](https://arxiv.org/pdf/2602.05754v1)

**作者:** Seonghye Cho `[一作]` (Department of XXX), Jae-Gil Lee `[通讯]` (Company Name)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了ICML 2026会议的论文提交与排版指南。

**💡 创新点**

创新点在于系统化地列出了双盲评审、字体、图表、参考文献等细节要求。

**🔧 技术方法**

使用 LaTeX、PDF、Type‑1 字体、图形导入等技术。

**📊 数据集**

未使用任何实验数据集。

**📈 对比分析**

无实验比较，主要说明如何遵循规范以避免常见错误。

**⚠️ 局限性**

局限性在于仅适用于ICML 2026，且不涉及实际研究内容。

---

## 460. CompactRAG: Reducing LLM Calls and Token Overhead in Multi-Hop Question Answering

**arXiv ID:** 2602.05728 | [PDF](https://arxiv.org/pdf/2602.05728v1)

**作者:** Hao Yang `[一作]` (Nanjing University), Lin Yang `[通讯]` (Nanjing University)

**通讯引用:** 82702 | [OpenAlex ID](https://openalex.org/A5100328102)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CompactRAG框架，通过离线将大规模文本转化为原子问答对并在在线推理中仅调用LLM两次，实现多跳推理；

**💡 创新点**

创新点在于将检索与推理解耦，构建“原子QA知识库”，使用轻量级答案提取器和子问题重写器，避免传统迭代RAG中频繁的LLM调用和实体漂移；

**🔧 技术方法**

采用LLM（如LLaMA3.1-8B或GPT-4）进行离线原子化，使用Contriever做密集检索，RoBERTa做答案提取，Flan‑T5做子问题重写，最终由LLM完成答案合成；

**📊 数据集**

在HotpotQA、2WikiMultiHopQA与MuSiQue三大多跳问答基准上进行评估；

**📈 对比分析**

与Vanilla‑RAG、Self‑Ask、IRCoT、Iter‑RetGen等迭代式RAG基线相比，CompactRAG在精确率、F1和LLM‑Acc方面保持竞争力，同时平均每问token消耗降低约70%–80%，且LLM调用次数固定为两次；

**⚠️ 局限性**

局限性包括：离线构建原子QA库需要一次较大LLM成本，库质量对性能影响显著；对非问答型知识的覆盖不足；在极深或极复杂的推理路径上仍可能出现信息缺失。

---

## 461. Exploring the Temporal Consistency for Point-Level Weakly-Supervised Temporal Action Localization

**arXiv ID:** 2602.05718 | [PDF](https://arxiv.org/pdf/2602.05718v1)

**作者:** Yunchuan Ma `[一作]` (University of Chinese Academy of Science), Qingming Huang `[通讯]` (University of Chinese Academy of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在点级监督的时间动作定位任务中，提出一种多任务学习框架，将三种基于点注释的自监督任务（动作完成、动作顺序理解、动作规律性理解）与主分类任务联合训练，以显式建模动作片段的时间一致性，提升定位精度。

**💡 创新点**

创新点在于首次将三种自监督任务引入点级监督的时间动作定位，通过对动作片段的完成预测、序列顺序判别和规律性判别三方面共同提升模型的时间一致性理解能力，从而显著提高点监督下的定位性能。

**🔧 技术方法**

采用预训练的 I3D 视觉特征，使用 Transformer 编码器进行特征嵌入，硬参数共享的多任务学习架构，伪标签生成、Focal 损失、对比损失以及 Soft‑NMS 等技术；三种自监督任务通过各自的全连接头实现。

**📊 数据集**

在四个主流数据集上进行实验：THUMOS’14、GTEA、BEOID、ActivityNet1.3。

**📈 对比分析**

与多种全监督、弱监督及点监督方法对比，实验表明在 THUMOS’14 上的平均 mAP（0.1–0.7）达到 58.2%，在其他数据集亦实现 SOTA 以上表现，并在某些 IoU 阈值上超过部分全监督方法，验证了自监督任务对定位效果的显著提升。

**⚠️ 局限性**

主要局限在于自监督任务仅在训练阶段使用，略微增加计算成本；对点注释分布较为敏感，点靠近边界时性能下降；在极少数复杂场景中仍可能出现误检或边界不精确的情况。

---

## 462. Muon in Associative Memory Learning: Training Dynamics and Scaling Laws

**arXiv ID:** 2602.05725 | [PDF](https://arxiv.org/pdf/2602.05725v1)

**作者:** Binghui Li `[一作]` (Peking University), Liwei Wang `[通讯]` (Peking University)

**通讯引用:** 13291 | [OpenAlex ID](https://openalex.org/A5100406718)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在关联记忆线性模型中对Muons优化器的训练动态进行理论分析，并与梯度下降和SignGD做对比。

**💡 创新点**

揭示梯度下降受频率差异影响导致学习速率不均衡，而Muons通过矩阵符号更新实现任务对齐与预条件化，从而获得指数加速和更优的缩放律。

**🔧 技术方法**

采用线性softmax关联记忆模型、矩阵符号运算、预条件化视角、频率谱分析以及实验验证等技术。

**📊 数据集**

使用合成关联记忆数据、带不平衡标签的MNIST数据集以及10B token的LLaMA预训练数据。

**📈 对比分析**

与GD、SignGD和理论对齐的TRA‑SignGD进行对比，Muons在无噪声情况下实现指数速度提升、在噪声下实现更快的收敛率，并在LLM预训练中展现更陡峭的规模收益。

**⚠️ 局限性**

结果基于正交嵌入和线性模型的理想化假设，未能完整覆盖复杂神经网络的非线性与多层结构；实际应用需进一步验证。

---

## 463. Price of universality in vector quantization is at most 0.11 bit

**arXiv ID:** 2602.05790 | [PDF](https://arxiv.org/pdf/2602.05790v1)

**作者:** Alina Harbuzova `[一作]` (Massachusetts Institute of Technology), Yury Polyanskiy `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 8904 | [OpenAlex ID](https://openalex.org/A5031031216)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明存在一个单一的通用量化码本，能够在任何输入激活协方差下，以几乎最优的误差率（最大多出0.11 bit/维）实现权重矩阵的低精度存储与计算。

**💡 创新点**

创新点在于：①提出并证明了不依赖激活统计的通用码本理论存在性；②给出了与最优水位填充（oracle）方案的误差上界，表明通用性并非信息论瓶颈。

**🔧 技术方法**

主要技术包括：信息论的水位填充与率失配分析、随机码本构造与大偏差理论、随机矩阵与覆盖理论、以及数值验证以确认误差上界。

**📊 数据集**

该工作为理论性研究，不使用具体数据集；权重向量假设为独立同分布的标准高斯分布。

**📈 对比分析**

通过与oracle水位填充基准的比特率比较，证明最大比特率溢出为0.11 bit/维；在低率情形下，通用码本相较于传统 INT8 等方法能实现更高的压缩率与近似精度。

**⚠️ 局限性**

局限性在于证明为非构造性，无法直接给出具体的通用码本实现；实际构造与硬件实现仍需进一步研究；此外，目前仅考虑单向量量化，矩阵级通用性尚未完整解析。

---

## 464. Allocentric Perceiver: Disentangling Allocentric Reasoning from Egocentric Visual Priors via Frame Instantiation

**arXiv ID:** 2602.05789 | [PDF](https://arxiv.org/pdf/2602.05789v1)

**作者:** Hengyi Wang `[一作]` (University of Science and Technology of China), Weiming Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 14087 | [OpenAlex ID](https://openalex.org/A5067689180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Allocentric Perceiver，一种无训练、基于几何的框架，用以解决VLM在需要视角转换的分配性空间推理任务中的不足。

**💡 创新点**

创新点在于：①将二维图像通过深度估计和三维重投影恢复统一的度量世界坐标；②依据查询语义动态构造目标中心的分配参考系，实现显式视角转换；③仅使用结构化几何文本提示完成推理，消除视觉先验干扰。

**🔧 技术方法**

采用的技术包括：LangSAM与层次语义放宽用于物体定位；Depth‑Anything‑3等深度估计器进行三维重建；DBSCAN聚类提取稳定几何；OpenCV坐标变换构造分配帧；以及利用LLM的链式推理进行几何推断。

**📊 数据集**

使用的公开数据集有：ViewSpatial‑Bench（含摄像头视角与人视角混合查询）和3DSRBench（多视角空间推理）。

**📈 对比分析**

通过在多种VLM骨干（Qwen2.5‑VL、InternVL2.5、GPT‑4o）上做无训练增强，Allocentric Perceiver在分配性任务平均提升约10%，同时保持或提升主体视角任务的准确率，且优于现有空间调优模型和主流商业模型。

**⚠️ 局限性**

局限性包括：①依赖高质量深度估计与物体检测，图像质量差时表现受限；②当前仅支持单目标或少数目标的参考帧构造，对复杂多人场景的支持不足；③推理过程仍需LLM的算力，实际部署时算力与延迟需进一步优化。

---

## 465. Selecting Hyperparameters for Tree-Boosting

**arXiv ID:** 2602.05786 | [PDF](https://arxiv.org/pdf/2602.05786v1)

**作者:** Floris Jan Koster `[一作]` (Seminar for Statistics, ETH Zurich), Fabio Sigrist `[通讯]` (Seminar for Statistics, ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对不同的树提升模型超参数优化方法进行系统实验评估，比较其在59个公开表格数据集上的性能表现。

**💡 创新点**

创新点在于提供了统一实验协议与可比的调参预算，揭示了SMAC方法在多数任务上优于其他主流技术，并指出大多数超参数都对模型效果有显著影响。

**🔧 技术方法**

采用了LightGBM树提升实现，并使用了随机网格搜索、TPE、GP-BO、Hyperband、SMAC以及全网格搜索等超参数搜索技术。

**📊 数据集**

实验数据来自OpenML的59个回归（36）和分类（23）任务，样本规模上限为10万，采用五折交叉验证与随机种子重复20次。

**📈 对比分析**

比较结果显示SMAC在所有指标上均优于其他方法，TPE排名第二；GP-BO和随机网格搜索处于中间位置；全网格搜索和Hyperband表现最差，默认参数更差；且至少需要100次试验才能获得可靠性能。

**⚠️ 局限性**

局限性包括仅评估LightGBM实现，未考虑时间成本、并行加速与公平性等实际约束，且在单个数据集上并非绝对最佳，未来可扩展至多种提升库及更完整的成本收益分析。

---

## 466. ReText: Text Boosts Generalization in Image-Based Person Re-identification

**arXiv ID:** 2602.05785 | [PDF](https://arxiv.org/pdf/2602.05785v1)

**作者:** Timur Mamedov `[一作]` (Tevian), Vadim Konushin `[通讯]` (Tevian)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种名为 ReText 的通用人像 Re-ID 方法，利用多摄像机数据与配有文本描述的单摄像机数据共同训练。

**💡 创新点**

创新点在于：①将单摄像机数据与自然语言描述相结合，通过多模态联合学习提升跨域泛化；②提出身份感知匹配损失（Identity‑aware Matching Loss）和结构保持损失（Structure‑preserving Loss）来更好对齐图像与文本；③在单摄像机数据上加入文本引导的图像重建任务，增强对遮挡与缺失信息的鲁棒性。

**🔧 技术方法**

核心技术包括 Transformer‑based 图像编码器（ViT‑Base）与 BERT‑Base 文本编码器；多任务损失（Re‑ID、图像‑文本匹配、文本引导重建）；动量图像编码器实现稳定训练；soft 匹配与 KL‑散度实现多正样本对齐。

**📊 数据集**

使用了多摄像机 Re‑ID 数据集（CUHK03、Market‑1501、MSMT17、CUHK‑SYSU、RandPerson）以及规模巨大且带文本描述的单摄像机数据集 SYNTH‑PEDES。

**📈 对比分析**

在三种标准跨域评测协议下，ReText 在 Rank‑1 与 mAP 上均超过现有最优方法（如 ReMix、DynaMix、CLIP‑ReID 等），显著提升跨域性能，尤其在使用多摄像机+单摄像机+文本联合训练时表现最为突出。

**⚠️ 局限性**

局限性包括：对高质量文本描述的依赖，单摄像机数据的标签稀缺性；模型对极端域漂移（如光照、姿态）仍有一定敏感度；训练成本较高，需要多模态大规模数据。

---

## 467. RL-VLA$^3$: Reinforcement Learning VLA Accelerating via Full Asynchronism

**arXiv ID:** 2602.05765 | [PDF](https://arxiv.org/pdf/2602.05765v1)

**作者:** Zhong Guan `[一作]` (Tianjin University), Junwu Xiong `[通讯]` (JDT AI Infra)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了全异步训练管线 RL‑VLA³，旨在提升 Vision‑Language‑Action（VLA）模型在强化学习后训练阶段的效率。

**💡 创新点**

创新点包括：①三层异步架构（环境‑回放、流式策略生成、解耦训练更新）实现全流程无锁并行；②动态批处理调度与微批次训练流水线，最大化 GPU 利用率；③GPU 资源动态分配策略，在不同阶段实现负载平衡。

**🔧 技术方法**

技术手段涵盖：异步并行与 GPU 分离分配、动态批处理调度、流式微批次训练、改造 RLinf 框架、分布式通信队列；VLA 模型使用 ViT+LLM、Diffusion/Autoregressive 结构。

**📊 数据集**

使用的数据集：LIBERO benchmark（LIBERO‑10/100）和 ManiSkill；对预训练的 VLA 模型 GR00T N1.5、π 系列（π₀、π₀.₅）以及 OpenVLA‑OFT 进行后训练。

**📈 对比分析**

与 RLinf 的两种同步策略（Colocated 与 Disaggregated）对比，实验显示在 8–256 GPU 规模下，Train Async、Rollout Async 与 Streamer 组合可实现最高 59.25%（相较同步单一策略）及 126.67%（相较同步解耦策略）的吞吐率提升；同时保持或仅略微影响最终任务性能。

**⚠️ 局限性**

局限性：①高通信开销导致在 128–256 GPU 极大规模下出现子线性扩展；②在 ManiSkill 环境中 Rollout Async 可能导致 GPU 计算失衡；③对高保真物理仿真与渲染延迟的处理尚不完善；④目前仅在现有仿真后端验证，需进一步扩展至更真实的仿真平台。

---

## 468. MU-MIMO Uplink Timely Throughput Maximization for Extended Reality Applications

**arXiv ID:** 2602.05751 | [PDF](https://arxiv.org/pdf/2602.05751v1)

**作者:** Ravi Sharan Bhagavathula `[一作]` (Nokia Bell Labs), Baltasar-Beferull Lozano `[通讯]` (SimulaMet and University of Agder)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对 6G 上的扩展现实（XR）应用，提出一种面向上行多用户 MIMO（MU‑MIMO）的调度迭代启发式算法，旨在最大化“及时吞吐量”并满足峰值信息年龄（PAoI）约束，从而提升 XR 能力。

**💡 创新点**

创新点在于：①将 PAoI 作为实时调度的权重因素，构造基于 PAoI 的加权比例公平（PF）度量；②在 PHY‑MAC 的跨层框架内考虑 MU‑MIMO 的空间层数与干扰约束；③提出完全无信令的调度方案，只利用上行 BSR 与 ACK，避免额外的信令开销。

**🔧 技术方法**

技术包括：无信令的 UE 资源分配、加权 PF 调度、时间平均 PAoI 计算、基于 MU‑MIMO 的链路预算与干扰建模、以及 5G NR 38.901 的 Urban Micro (UMi) NLoS 信道模型。

**📊 数据集**

使用仿真数据集：单站三 gNB，N=10 UE，gNB 48 天线、16 TRX，UE 4 TRX；采用 3GPP 38.901 Urban Micro NLoS 信道、5G 导频估计、RB 与 MCS 设置；包大小 75 kb，PDB 30 TTI，仿真总时长 10⁶ TTI，35 次独立仿真。

**📈 对比分析**

与基线方法（经典 PF、加权 PF、DRL 解决方案）进行比较，指标为良吞吐量（goodput）、XR 能力（满足 99% 包成功率的 UE 百分比）以及时间平均 PAoI 分布。结果显示：①加权 PF 方案在前 95% UE 的良吞吐量与经典 PF 相近；②在底部 5%/10% UE 上显著提升；③XR 能力提升至 81%（比 PF 方案提升 >20%），时间平均 PAoI 均值为 28，低于 PDB。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实网络部署；未考虑 HARQ 重传、链路误码；需要对 κ、θ 等参数进行系统化调优；算法仍具一定计算复杂度，尤其在大规模 UE 或多小区场景下；并且对信道估计误差的鲁棒性未作深入分析。

---

## 469. LeakBoost: Perceptual-Loss-Based Membership Inference Attack

**arXiv ID:** 2602.05748 | [PDF](https://arxiv.org/pdf/2602.05748v1)

**作者:** Amit Kravchik Taub `[一作]` (Ben Gurion University), Yisroel Mirsky `[通讯]` (Ben Gurion University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种名为LeakBoost的白盒成员推断攻击框架，通过对目标模型内部表示进行感知损失优化生成交互图像，从而增强模型对是否属于训练集的隐式信号；

**💡 创新点**

创新点在于将感知（激活空间）损失用于主动探测，利用模型对输入的动态响应来放大成员与非成员之间的细微差异，并能与现有检测器无缝配合；

**🔧 技术方法**

核心技术包括感知损失（MSE在多层激活空间内）、随机噪声初始化、Adam优化、可选值裁剪、以及后续使用GLiR等梯度基检测器；

**📊 数据集**

实验使用CIFAR‑10和CIFAR‑100两大图像数据集，模型涵盖ResNet‑18、DenseNet、AlexNet和轻量Vision Transformer ViT‑4；

**📈 对比分析**

与SIF、LAEQ、IA、GLiR等白盒基线相比，LeakBoost在ViT‑4和AlexNet上显著提升AUC（从0.53–0.62提升至0.81–0.88）以及低FPR下TPR（如1% FPR由≤1.3%提升至11.8%），在其他网络上提升幅度虽小但仍表现良好；

**⚠️ 局限性**

局限性包括对深层CNN（DenseNet、ResNet‑18）提升有限，且对模型架构和超参高度敏感（尤其是学习率、优化步数、层级选择），在不同数据集上表现不稳定，且依赖于完整白盒访问。

---

## 470. Mitigating Hallucination in Financial Retrieval-Augmented Generation via Fine-Grained Knowledge Verification

**arXiv ID:** 2602.05723 | [PDF](https://arxiv.org/pdf/2602.05723v1)

**作者:** Taoye Yin `[一作]` (Ant Group), Feng Wang `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种基于强化学习的细粒度知识验证框架 RLFKV，旨在提升金融检索增强生成系统对检索文档的事实一致性。

**💡 创新点**

创新点在于将生成的回复拆分为原子知识单元（实体、指标、数值、时间四元组），对每个单元独立评估并产生细粒度奖励，并辅以信息量奖励以抑制奖励挖掘。

**🔧 技术方法**

主要技术包括使用大型语言模型 Qwen3-32B 进行原子单元分解与验证、构造置信度与信息量两类奖励、以及采用 GRPO 算法进行策略优化。

**📊 数据集**

实验使用公开的 Financial Data Description (FDD) 数据集以及作者自建的多类型 FDD-ANT 数据集（涵盖股票、基金和宏观指标）。

**📈 对比分析**

与 DeepSeek‑V2、LLaMA、Qwen、Xuanyuan、Dianjin 等多种基线对比，RLFKV 在 FDD 上提升 3.6 分，FDD‑ANT 上提升 3.1 分，同时保持甚至提高信息量；去掉信息奖励后信度基本保持但信息量下降。

**⚠️ 局限性**

限制主要体现在仍存在时间缺失、时间不准和数值误差等三类错误，表明奖励机制和单元验证过程还需进一步改进以提升时序和数值准确性。

---

## 471. A Dual-Loop Agent Framework for Automated Vulnerability Reproduction

**arXiv ID:** 2602.05721 | [PDF](https://arxiv.org/pdf/2602.05721v1)

**作者:** Bin Liu `[一作]` (Harbin Institute of Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 29115 | [OpenAlex ID](https://openalex.org/A5107888510)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于双循环的LLM代理框架，自动从CVE描述生成可执行PoC并进行验证

**💡 创新点**

创新点包括：将战略规划与战术执行分离的双循环结构、分层多模验证机制、双维度失败诊断与稀疏经验索引技术

**🔧 技术方法**

使用大语言模型（如DeepSeek‑V3.2）配合LLM代理、静态/动态分析、分层Oracle以及稀疏经验索引

**📊 数据集**

使用SecBench.js与PatchEval两大真实漏洞基准，共计617个漏洞

**📈 对比分析**

与PoCGen、OpenHands、CAI对比，分别在SecBench.js和PatchEval上实现82.9%/54.3%的复现成功率，比最优基线高出11.3%/20.4%，且token消耗显著降低

**⚠️ 局限性**

局限在于仅评估JavaScript/Python/Go三种语言，依赖现有基准，验证指标未覆盖PoC最小化和可读性细粒度等更细化质量维度

---

## 472. Towards Green AI: Decoding the Energy of LLM Inference in Software Development

**arXiv ID:** 2602.05712 | [PDF](https://arxiv.org/pdf/2602.05712v1)

**作者:** Lola Solovyeva `[一作]` (University of Twente), Fernando Castor `[通讯]` (University of Twente)

**通讯引用:** 2454 | [OpenAlex ID](https://openalex.org/A5062400717)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对LLM推理过程进行阶段级能耗分析，分别评估前填充(prefill)和解码(decoding)两个阶段的能耗，并研究输入长度、输出长度及模型“胡言乱语”(babbling)行为对能耗的影响；此外，提出了一种基于测试通过的早停(babbling suppression)方法，显著降低不必要的token生成，从而降低能耗。

**💡 创新点**

①首次将LLM推理能耗分解为前填充与解码两个阶段，揭示二者在不同任务与模型规模下的能耗特征；②发现前填充成本会显著放大解码阶段每token能耗；③识别并量化模型的胡言乱语行为；④提出并验证基于中间测试通过的早停算法，实现高达89%的能耗下降。

**🔧 技术方法**

使用GPU能耗采样(pyNVML)与token生成时间戳对齐的细粒度能耗测量；对10个decoder‑only transformer模型（Llama, Phi, Gemma, Qwen）进行推理；实现基于语法和单元测试的中间输出检测与早停逻辑。

**📊 数据集**

HumanEval（代码生成）和LongBench（代码理解）两个公开基准数据集。

**📈 对比分析**

通过在5种不同工作负载（Zero-shot、Few-shot、Chain‑of‑thought、Code understanding、Code understanding with explanation）下，分别测量每个模型的总能耗、prefill能耗、解码能耗、每token能耗和准确率；结果表明解码阶段占能耗大部分，输入长度增大会提升prefill与解码能耗，输出长度越长每token能耗越高；babbling suppression在保持准确率的前提下，使能耗下降44%~89%。

**⚠️ 局限性**

仅针对decoder‑only transformer模型，实验仅在单一GPU（NVIDIA A10）环境下进行，可能受限于硬件、驱动与并发负载；阶段划分方法假设首token归prefill，可能略有偏差；未覆盖encoder/encoder‑decoder或混合专家模型，结论泛化受限；实验样本数有限，未采用统计显著性检验。

---

## 473. Automated Customization of LLMs for Enterprise Code Repositories Using Semantic Scopes

**arXiv ID:** 2602.05780 | [PDF](https://arxiv.org/pdf/2602.05780v1)

**作者:** Ulrich Finkler `[一作]` (IBM Research), Shyam Ramji `[通讯]` (IBM Research)

**通讯引用:** 434 | [OpenAlex ID](https://openalex.org/A5091531089)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

自动从企业代码库提取语义作用域，生成训练对，使用微调方式定制小型LLM实现代码补全。

**💡 创新点**

首次基于语义作用域自动化构造训练数据，证明微调小模型在专有仓库上优于RAG和更大模型。

**🔧 技术方法**

语义作用域抽取、向量检索增强生成（RAG）、监督式微调（HuggingFace Trainer）、Levenshtein距离评估等技术。

**📊 数据集**

两大企业私有仓库（C/C++ DataB，Java STM）和公开基准（CCEval、RepoBench）。

**📈 对比分析**

用全/最优Levenshtein距离比较，微调模型显著降低距离，优于RAG；在企业仓库上小型微调模型的精确度和延迟均优于原始大型模型。

**⚠️ 局限性**

需依赖语言特定的作用域提取，微调时数值不稳定，RAG缺乏简洁性，实验仅覆盖代码补全，未验证其他任务或多语言适用性。

---

## 474. Task-Oriented Robot-Human Handovers on Legged Manipulators

**arXiv ID:** 2602.05760 | [PDF](https://arxiv.org/pdf/2602.05760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 475. Cross-Domain Offline Policy Adaptation via Selective Transition Correction

**arXiv ID:** 2602.05776 | [PDF](https://arxiv.org/pdf/2602.05776v1)

**作者:** Mengbei Yan `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 10596 | [OpenAlex ID](https://openalex.org/A5100754504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了跨域离线强化学习中针对动力学不匹配的策略适应方法——Selective Transition Correction (STC)，通过将源域转移样本转化为目标域数据来提升目标域策略性能。

**💡 创新点**

创新点在于：①直接修正源域过渡的动作与奖励，使其与目标域动力学对齐；②引入前向动力学模型实现选择性修正，避免误差放大；③在理论上给出了修正后数据与目标域之间的动态与价值差距界定。

**🔧 技术方法**

使用了逆策略模型、奖励模型和前向动力学模型进行转移样本修正，并在离线 actor‑critic 框架中训练最终策略；采用了基于 TD 目标的 Q‑学习和 Q 加权行为克隆。

**📊 数据集**

实验数据集为 MuJoCo “‑v2” 版本的 D4RL 源域数据与 ODRL 目标域数据，涵盖重力、摩擦、结构三种动力学偏移，任务包括 ant、hopper、halfcheetah、walker2d。

**📈 对比分析**

与 IQL、DARA、BOSA、SRPO、IGDF、OTDF 等基线进行对比，STC 在多种动力学偏移与数据质量设置下均获得更高的标准化得分（总分 1045.2 或 820.8，明显优于 OTDF 等对手）。

**⚠️ 局限性**

局限性：需要额外训练逆策略、奖励和前向动力学模型；理论分析依赖若干较强假设；在极少量目标域数据或高度离散动作空间时性能可能受限。

---

## 476. A Structural Equivalence of Symmetric TSP to a Constrained Group Steiner Tree Problem

**arXiv ID:** 2602.05773 | [PDF](https://arxiv.org/pdf/2602.05773v1)

**作者:** Yılmaz Arslanoğlu `[一作]` `[通讯]`, Yılmaz Arslanoğlu

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了将对称TSP转换为在三角形与边构成的偶联图上的受限Group Steiner Tree问题的结构等价性

**💡 创新点**

将旅行商巡回视为二维单纯复形的边界，利用拓扑学的Euler特征约束将全局连通与边权归零自然嵌入模型

**🔧 技术方法**

组合数学、图论、简单拓扑学（单纯复形与Euler特征）以及群Steiner树理论

**📊 数据集**

未进行实验，本文仅给出一个 n=10 的示例图说明模型；数据来源为完整加权图

**📈 对比分析**

无实验比较；理论上证明最优值等价，未给出算法或性能评估

**⚠️ 局限性**

缺乏实现细节与实际算法，无法直接评估求解效率；模型只在完整三角化可用时保持精确，受限时仅为启发式

---

## 477. RocqSmith: Can Automatic Optimization Forge Better Proof Agents?

**arXiv ID:** 2602.05762 | [PDF](https://arxiv.org/pdf/2602.05762v1)

**作者:** Andrei Kozyrev `[一作]` (JetBrains Research), Anton Podkopaev `[通讯]` (Constructor University)

**通讯引用:** 150 | [OpenAlex ID](https://openalex.org/A5082235493)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并比较了多种自动化代理优化方法在 Rocq 形式化验证任务中的效果，使用简化版 RocqProof 代理作为实验基线。

**💡 创新点**

首次系统检验了自动化优化（提示、上下文、控制流）在正式推理环境中的可迁移性，并指出仅通过少量示例即可显著提升性能。

**🔧 技术方法**

使用 BootstrapFewShot、MIPROv2、SIMBA、GEPA、ACE、ReasoningBank、ADAS 等优化技术，并基于 DSPy 与 Koog 两个框架实现代理。

**📊 数据集**

采用 IMM 题库（300 条 Coq 定理，按证明长度划分），实验中用 50 条各难度组评估，60 条训练优化器。

**📈 对比分析**

通过对比基线（无优化）与各优化器在 20 次工具调用限制下的成功率，发现 BootstrapFewShot 在 DSPy 上将成功率从 19% 提升至 40%，在 Koog 上从 20% 提升至 33%，但均未达到手工构造的 SOTA 代理（约 53%）。

**⚠️ 局限性**

自动化优化方法仍无法匹配人类工程化代理的性能；上下文检索方法需额外工程改动；控制流优化 ADAS 结果不稳定且过拟合训练数据；少量示例在长证明/长追踪中会耗尽上下文。

---

## 478. Toward Quantum-Safe Software Engineering: A Vision for Post-Quantum Cryptography Migration

**arXiv ID:** 2602.05759 | [PDF](https://arxiv.org/pdf/2602.05759v1)

**作者:** Lei Zhang `[一作]` (University of Maryland, Baltimore County), Lei Zhang `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 4856 | [OpenAlex ID](https://openalex.org/A5100433957)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了Quantum‑Safe Software Engineering (QSSE) 研究方向，并设计了 AQuA 框架，包含三大支柱：PQC‑Aware Detection、Semantic Crypto‑Refactoring、Hybrid Correctness Verification，旨在支持大规模 Post‑Quantum Cryptography（PQC）迁移。

**💡 创新点**

创新点在于将加密约束提升为软件工程的核心驱动因素，系统化构建了 PQC‑aware detection、refactoring patterns 与 hybrid verification 的完整流程，填补了现有 CBOM 方案在代码语义、自动化重构与持续验证三方面的空白。

**🔧 技术方法**

技术上结合了静态与动态分析、机器学习辅助扫描、架构模式与代码生成、侧信道分析以及差分/形态测试等方法。

**📊 数据集**

本论文未提供具体数据集或实验集，主要以概念性框架与理论阐述为主。

**📈 对比分析**

由于为框架性提案，文中未给出实验对比或性能指标，亦未进行实测评估。

**⚠️ 局限性**

限制在于方案尚未实现或验证，缺乏实际工具实现、跨语言与跨平台支持、与 CI/CD 的集成、可扩展性与性能评估，以及在真实项目迁移中的效果与成本评估。

---

## 479. Learning to Inject: Automated Prompt Injection via Reinforcement Learning

**arXiv ID:** 2602.05746 | [PDF](https://arxiv.org/pdf/2602.05746v1)

**作者:** Xin Chen `[一作]` (ETH Zurich), Florian Tramer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用强化学习自动化生成通用且可迁移的 Prompt Injection 攻击后缀，以提升攻击成功率并保持任务实用性。

**💡 创新点**

①通过与反馈模型比较当前后缀与历史最优后缀，产生稠密奖励；②同时实现在线查询优化和可迁移后缀两种攻击模式；③利用 GRPO 训练轻量化策略模型。

**🔧 技术方法**

强化学习（GRPO）、稠密奖励的比较反馈模型（基于 GPT‑4o‑mini）、LLM 策略模型（Qwen2.5‑1.5B）、AgentDojo 测试管线。

**📊 数据集**

AgentDojo benchmark（97个用户任务+注入任务）、Meta‑SecAlign‑70B 等公开对抗性数据集。

**📈 对比分析**

与模板攻击、GCG、TAP、随机自适应攻击等基线相比，在 Gemini‑2.5‑Flash、GPT‑4.1‑Nano、GPT‑5‑Nano、Claude‑3.5‑Sonnet 等前沿 LLM 上，攻击成功率提升至 58%–21%，且往往保持或超过无攻击时的任务实用性；对 Meta‑SecAlign‑70B 也取得 21.88% 的成功率，说明可突破基于安全微调的防御。

**⚠️ 局限性**

依赖有限的黑盒查询预算；对高度安全模型（如 GPT‑5）迁移效果低；假设目标模型即时执行工具调用；攻击后缀生成受限于反馈模型的准确性和训练数据。

---

## 480. Balancing FP8 Computation Accuracy and Efficiency on Digital CIM via Shift-Aware On-the-fly Aligned-Mantissa Bitwidth Prediction

**arXiv ID:** 2602.05743 | [PDF](https://arxiv.org/pdf/2602.05743v1)

**作者:** Liang Zhao `[一作]` (South China University of Technology), Yi Zou `[通讯]` (South China University of Technology)

**通讯引用:** 1847 | [OpenAlex ID](https://openalex.org/A5100458990)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种支持可变对齐尾数位宽的FP8数字计算内存（DCIM）加速器，能够在Transformer推理/训练中动态调整权重和激活的尾数精度以兼顾精度与能效。

**💡 创新点**

创新点包括（1）动态位宽预测（DSBP）算法，可实时估计每个乘加组的最佳尾数位宽；（2）基于FIFO的输入对齐单元（FIAU）取代复杂的桶式移位器，降低面积与功耗；（3）可扩展的INT MAC阵列，支持2/4/6/8位权重与2~12位输入的高效融合。

**🔧 技术方法**

实现技术涵盖FP8量化、位宽预测算法、FIFO指针对齐、压缩加法树与可重用融合路径，以及28nm CMOS工艺布局。

**📊 数据集**

实验使用Llama‑7b（BoolQ、Winogrande）和ResNet‑18（ImageNet）等公开数据集验证精度与能效。

**📈 对比分析**

与固定位宽FP8与INT8对比，DSBP模式在保持相同或更低误差的同时，效率提升至2.8×（E5M7下实现20.4 TFLOPS/W），在Llama‑7b上达33.7 TFLOPS/W的能效，且支持所有FP8格式。

**⚠️ 局限性**

局限性在于：位宽预测与对齐需要额外的硬件开销（MPU占用7%面积），对极端动态范围的数据分布仍可能出现截断误差；加速器目前针对28nm工艺，迁移到更先进工艺需进一步验证。

---

## 481. OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale

**arXiv ID:** 2602.05711 | [PDF](https://arxiv.org/pdf/2602.05711v1)

**作者:** Jingze Shi `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1645 | [OpenAlex ID](https://openalex.org/A5100614732)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 OmniMoE 框架，通过原子专家、笛卡尔乘积路由器和专家中心调度，实现了大规模细粒度 MoE 的高效推理与优秀性能。

**💡 创新点**

创新点包括：① 将专家粒度细化到原子级并通过动态专家组装（DEA）构造令牌专属的高表达度专家；② 采用笛卡尔乘积路由器将 N 维专家索引拆分为行列两维，路由复杂度从 O(N) 降到 O(√N)；③ 设计专家中心调度，将令牌级别的散乱访存转为专家级别的连续批量 GEMM，显著提升 GPU 计算利用率。

**🔧 技术方法**

使用技术包括：Mixture-of-Experts 结构、SwiGLU 激活、分块矩阵检索、LogSoftmax 路由、并行 top‑K 选取、分组 GEMM（Grouped GEMM）以及 NVIDIA A100 GPU 上的高效张量运算。

**📊 数据集**

训练数据集为 40B 词的 SmolLMCorpus，评测数据集为七个零样本基准：MMLU、TriviaQA、ARC、PIQA、HellaSwag、OBQA 与 Winogrande。

**📈 对比分析**

与 Dense、GShard、DeepSeekMoE、PKM、PEER 等基线在同等激活参数预算下对比，OmniMoE 在 6.4B‑A1.7B 模型上获得 50.9% 的平均零样本准确率，超过 DeepSeekMoE（+0.7%）和 PEER（+2.0%），且推理延迟从 PEER 的 73 ms 降至 6.7 ms，速度提升 10.9×。

**⚠️ 局限性**

局限性包括：① 对 GPU 内存和带宽仍有一定需求，尤其在极大专家池时需要额外的内存调度；② 原子专家表达力有限，可能在极端复杂任务上受限；③ 笛卡尔路由器和专家调度的实现复杂度较高，迁移到其它硬件或框架时需额外工程工作。

---

## 482. Nonlinearity as Rank: Generative Low-Rank Adapter with Radial Basis Functions

**arXiv ID:** 2602.05709 | [PDF](https://arxiv.org/pdf/2602.05709v1)

**作者:** Yihao Ouyang `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4085 | [OpenAlex ID](https://openalex.org/A5039670436)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的低秩适配器 GenLoRA，通过 RBF 函数从共享潜在向量生成基向量，实现参数高效的模型微调。

**💡 创新点**

创新点在于将显式存储的高维基向量转化为非线性生成方式，利用 Radial Basis Function（RBF）压缩参数量并保留低秩结构，实现“非线性即秩”。

**🔧 技术方法**

采用 RBF 生成器、实例归一化、组归约等技术，构建轻量级非线性基向量生成器，并在 LoRA 结构中替代传统基向量。

**📊 数据集**

在数学推理数据集 Math10K、常识推理数据集 Commonsense170K 以及代码生成数据集 HumanEval+ 上进行实验。

**📈 对比分析**

与 LoRA、MELoRA、HiRA、DoRA、Aurora 等基准方法对比，GenLoRA 在保持 1/5‑1/4 参数量的前提下，NLG 性能提升 2‑4%，代码生成提升 6‑8%，并在多模型、多任务上取得领先。

**⚠️ 局限性**

局限性包括对 RBF 栅格和组大小的敏感性，需要手动设定；在极大规模模型或非语言任务上尚未充分验证；生成基向量的表达能力可能受限于简单 RBF 模型。

---

## 483. Pathwise Test-Time Correction for Autoregressive Long Video Generation

**arXiv ID:** 2602.05871 | [PDF](https://arxiv.org/pdf/2602.05871v1)

**作者:** Xunzhi Xiang `[一作]` (Nanjing University), Chunchao Guo `[通讯]` (Tencent Hunyuan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练-free的测试时校正（Test‑Time Correction）方法，在推理阶段对蒸馏自回归扩散模型的随机采样路径进行路径校正，利用首帧作为参考来纠正中间状态，从而抑制误差累积并延长可生成视频长度。

**💡 创新点**

创新点在于不对模型参数进行任何优化，而是在采样轨迹中插入参考引导的去噪和重新噪声化步骤，使得纠正可以在保持原始分布的同时平滑地被后续步骤继承，避免了传统 sink‑based 方法导致的结构冻结和漂移。

**🔧 技术方法**

使用自回归扩散模型（如 CausVid、Self‑Forcing）与蒸馏少步采样，结合参考条件去噪、路径上重噪声化的训练‑free校正流程；与现有的 Rolling Forcing、LongLive、Self‑Forcing 等方法做对比。

**📊 数据集**

主要使用 MovieGen 数据集（随机抽取 128 条提示）进行 30 秒视频生成评估，并在 5 秒短视频场景下也做了验证。

**📈 对比分析**

在 VBench、颜色漂移（L1、Correlation）、JEPA 以及 t‑LPIPS 等指标上与传统基线和训练驱动方法比较。结果显示，在 30 秒生成任务中，TTC 在保持 FPS 低于训练方法的同时，获得与 Rolling Forcing/LongLive 相当甚至更优的视觉质量、动态度和时间一致性。

**⚠️ 局限性**

局限性包括：①仅针对蒸馏自回归模型验证，超长（>30 秒）序列仍难以完全稳定；②需要手动设定校正步骤和噪声等级，可能对不同场景不够自适应；③对极端复杂动态或视觉细节变化的场景，路径校正效果仍有限。

---

## 484. Persistent Human Feedback, LLMs, and Static Analyzers for Secure Code Generation and Vulnerability Detection

**arXiv ID:** 2602.05868 | [PDF](https://arxiv.org/pdf/2602.05868v1)

**作者:** Ehsan Firouzi `[一作]` (Technische Universität Clausthal), Mohammad Ghafari `[通讯]` (Technische Universität Clausthal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文先综述现有关于LLM安全代码生成与漏洞检测的文献，随后手工标注1080个GPT‑4o RCI生成的Python代码样本，比较静态分析工具CodeQL和Semgrep与人工评估的结果，并基于发现的不足提出LLMSecGuard框架，框架通过动态检索增强生成（RAG）与人类反馈循环，持续改进LLM的安全代码生成与漏洞检测；

**💡 创新点**

①系统性地构造了人类验证的ground‑truth并量化工具召回/精准；②发现现有静态分析工具在安全判定上误报与漏报都较多，凸显专家反馈不可或缺；③提出了持久存储人类反馈的Dual‑Source RAG框架，并为反馈分配动态信任权重与EMA更新机制，支持LLM在后续生成中复用历史反馈；

**🔧 技术方法**

使用的技术包括GPT‑4o（RCI prompting）、静态分析工具CodeQL与Semgrep、双人手工评审、Dual‑Source Retrieval‑Augmented Generation（RAG）以及基于信任权重和指数加权移动平均的反馈更新算法；

**📊 数据集**

实验数据集主要为LLMSecEval与SecurityEval基准（Python/C/C++），共1080个GPT‑4o生成的代码样本，涵盖CWE Top 25任务与对应CWE分布；

**📈 对比分析**

通过对每个样本的安全/不安全判定与人工标签进行交叉对比，计算Recall、Precision和F1。结果显示：CodeQL召回0.34、精准0.67、F1 0.45；Semgrep召回0.54、精准0.52、F1 0.53；两者一致时召回0.30、精准0.76、F1 0.43，表明两工具在安全检测上的误报率高且召回率低；

**⚠️ 局限性**

局限性包括：仅覆盖Python与C/C++语言，评估仅限GPT‑4o与两种静态工具，手工评审仍受主观偏差影响，实验仅针对目标CWE，未考虑其他CWE或大规模真实项目，因而难以直接推广到所有语言与工具组合。

---

## 485. DARWIN: Dynamic Agentically Rewriting Self-Improving Network

**arXiv ID:** 2602.05848 | [PDF](https://arxiv.org/pdf/2602.05848v1)

**作者:** Henry Jiang `[一作]` `[通讯]` (Georgia Institute of Technology), Henry Jiang (Georgia Institute of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

DARWIN是一种基于进化的GPT框架，通过LLM代理相互改写训练代码来实现模型自我改进；

**💡 创新点**

创新点在于将遗传算法与LLM驱动的代码突变相结合，并引入持久化JSON记忆与双向人机交互接口以实现安全、可追踪的自我优化；

**🔧 技术方法**

使用技术包括OpenAI GPT‑4o‑mini API进行代码突变、nanoGPT训练框架、Docker容器化隔离、JSON日志存储、遗传算法选择等；

**📊 数据集**

使用数据集为Shakespeare散文文本（nanoGPT默认数据），作为训练和评测基准；

**📈 对比分析**

通过在5代进化中对比基线模型，使用困惑度（PPL）和MFU指标进行评测，结果显示困惑度下降2.07%、MFU提升1.26%，每代平均耗时约223秒，错误率约37.5%；

**⚠️ 局限性**

局限性包括改进幅度有限、仅在小模型和小数据集上验证、训练次数少、错误率高、需要手动安全检查、缺乏对大规模模型和更复杂任务的验证。

---

## 486. Synthesizing Realistic Test Data without Breaking Privacy

**arXiv ID:** 2602.05833 | [PDF](https://arxiv.org/pdf/2602.05833v1)

**作者:** Laura Plein `[一作]` (CISPA Helmholtz Center for Information Security), Andreas Zeller `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于语言测试的隐私合成数据生成框架，利用Fandango模糊器生成数据并通过判别器迭代逼近原始数据分布，避免直接使用原始数据，从而实现隐私保护与数据实用性的平衡。

**💡 创新点**

核心创新在于将传统GAN的生成器替换为基于语法的模糊测试器，并将判别器的预测结果作为约束反馈，使得生成过程完全无原始数据流，既提升隐私安全，又能保持统计特征和模型性能。

**🔧 技术方法**

主要技术包括Fandango语法模糊测试、决策树/分类器判别器、Wasserstein距离评估、机器学习模型（如随机森林、梯度提升机、SVM）训练与验证。

**📊 数据集**

实验使用了四个公开表格数据集：UCI Adult、Credit、Insurance、Bank（后续计划扩展至10个），涵盖分类与回归任务。

**📈 对比分析**

通过模型在原始/合成数据上的R²/准确率对比、判别器准确率下降至随机水平以及Wasserstein距离评估，结果表明合成数据与原始数据相似度高、模型性能几乎相同，且未出现原始样本。

**⚠️ 局限性**

局限性包括需要手工或自动化生成Fandango规范、对分类特征相似度的评估仍有限、实验数据集数量有限，需进一步扩展数据集并改进隐私度量方法。

---

## 487. Whispers of the Butterfly: A Research-through-Design Exploration of In-Situ Conversational AI Guidance in Large-Scale Outdoor MR Exhibitions

**arXiv ID:** 2602.05826 | [PDF](https://arxiv.org/pdf/2602.05826v1)

**作者:** Dongyijie Primo Pan `[一作]` (Hong Kong University of Science and Technology), Pan Hui `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 21129 | [OpenAlex ID](https://openalex.org/A5029925982)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在大规模户外混合现实艺术展中开发并部署了可召唤的会话式 AI 导览 Dream-Butterfly，并对比 AI 主导与人类导览的访客体验。

**💡 创新点**

创新点是将会话式 AI 作为可随时召唤的非人类伴侣，在自由漫游环境下提供即时多语言解说，并系统探究 AI 与人类导览的角色配置与交互。

**🔧 技术方法**

使用了检索增强生成（RAG）的多语言对话系统、SLAM 定位与动态跟随、低功耗非人类蝴蝶形象、语音识别/合成以及多模态交互技术。

**📊 数据集**

使用了展览创作者提供的多语言说明材料、翻译文本作为知识库以及访客交互日志数据。

**📈 对比分析**

通过野外受试者对照实验（N=24），使用问卷（解释获取、沉浸感、角色清晰度）和访谈进行比较，结果显示 AI 主导配置提升了解释获取和沉浸感，且未显著增加访客工作量。

**⚠️ 局限性**

局限性包括样本规模小、实验分配受现场条件限制、对话质量受方言差异影响，且结果仅为经验轨迹而非因果结论。

---

## 488. ToMigo: Interpretable Design Concept Graphs for Aligning Generative AI with Creative Intent

**arXiv ID:** 2602.05825 | [PDF](https://arxiv.org/pdf/2602.05825v1)

**作者:** Lena Hegemann `[一作]` (Aalto University), Hariharan Subramonyam `[通讯]` (Stanford University)

**通讯引用:** 707 | [OpenAlex ID](https://openalex.org/A5111056004)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出ToMigo系统，将用户创作意图建模为可解释、可编辑的设计概念图，以帮助生成式AI与设计者的目标对齐。

**💡 创新点**

创新点在于构建基于图谱的理论推理框架（图Schema），并通过图结构实现对AI推理的可视化、交互编辑和动态更新，弥合提示式交互与专业设计流程之间的鸿沟。

**🔧 技术方法**

技术包括多模态大型语言模型（如Claude、GPT‑4.1、o4‑mini）用于图谱生成与更新，图神经网络无明确使用但图结构与边关系为核心，前端React、后端Flask、PostgreSQL实现交互与存储。

**📊 数据集**

数据集由99组用户上传的三张参考图及其意图描述（来自Prolific问卷）构成，用于构建图Schema；设计任务共18种，随机采样3个给每位受试者；实验采用两组用户研究收集对齐评估。

**📈 对比分析**

通过两项用户研究对比ToMigo与基线（直接提示式生成）在图与图像对齐度、用户满意度等指标上进行评估。实验结果显示，图概念对齐度平均4.22/5，图像对齐度与基线无显著差异，且用户在ToMigo条件下报告更高的控制感、意图澄清和创作满意度。

**⚠️ 局限性**

局限性包括仅测试单页静态图形设计，未覆盖多页、交互式界面或三维包装；图谱基于非专业设计师的参考图片，可能与专业设计知识存在差异；系统对复杂设计类型和交互元素的支持需进一步扩展。

---

## 489. NEX: Neuron Explore-Exploit Scoring for Label-Free Chain-of-Thought Selection and Model Ranking

**arXiv ID:** 2602.05805 | [PDF](https://arxiv.org/pdf/2602.05805v1)

**作者:** Kang Chen `[一作]` (Fudan University), Yixin Cao `[通讯]` (Fudan University)

**通讯引用:** 5587 | [OpenAlex ID](https://openalex.org/A5013247988)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无标签的神经元动态评分方法NEX，用来评估大语言模型的推理质量并对模型与数据进行排序与筛选。

**💡 创新点**

创新点在于利用稀疏MLP神经元的新颖度斜率（novelty‑slope）并通过粘性两状态HMM分段，将推理过程划分为探索（E‑phase）和利用（X‑phase），再依据神经元在E→X周期中的复用程度给出正负权重，从而将非单调的探索程度转化为单调的性能预测信号。

**🔧 技术方法**

核心技术包括稀疏MLP激活记录、novelty‑slope时间序列构建、粘性HMM状态分割、周期级进展与巩固计算、神经元权重归一化以及Good‑Mass Fraction评分。

**📊 数据集**

使用Qwen3系列模型（4B、VL‑4B、VL‑8B、VL‑32B）的指令端点与思考端点混合生成候选模型；在100道无标签推理题目上学习神经元权重；在AIME24/25、GPQA、HMMT25、BRUMO25等五个严苛推理基准上评估。

**📈 对比分析**

与长度、熵、平均log‑prob等无标签基线相比，NEX在20个模型‑基准对中平均Pearson相关系数r≈0.78、Regret@1≈2.7pp、Hit@3≈35%；在数据筛选实验中，Top20% NEX筛选的学生模型在各任务上平均提升约3–5pp；此外，跨模型神经元转移验证了有效与冗余神经元的因果影响。

**⚠️ 局限性**

局限性包括：需访问内部激活（对闭源模型不可行）；依赖固定token长度的分块，短响应可能缺少足够的E/X结构；目前仅适用于稀疏MLP神经元，难以直接扩展到注意力头等；假设激活量能准确反映神经元贡献，可能对不同架构偏差。

---

## 490. Learning Compact Boolean Networks

**arXiv ID:** 2602.05830 | [PDF](https://arxiv.org/pdf/2602.05830v1)

**作者:** Shengpu Wang `[一作]` (ETH Zurich), Martin Vechev `[通讯]` (ETH Zurich)

**通讯引用:** 11114 | [OpenAlex ID](https://openalex.org/A5069901599)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种高效学习紧凑布尔网络的方法

**💡 创新点**

通过无参数连接学习、单操作卷积核和自适应离散化三项创新显著压缩网络并提升准确率

**🔧 技术方法**

使用可微布尔函数松弛、指数加权熵自适应重采样、逐层离散化以及基于热编码的输入/输出转换

**📊 数据集**

在MNIST、CIFAR‑10等标准视觉数据集上进行实验

**📈 对比分析**

与TreeLogicNet等基准对比，取得更高准确率且布尔运算量减少最多37倍，模型规模亦大幅缩小

**⚠️ 局限性**

仍需在训练阶段保持浮点计算、仅验证视觉任务、未结合高级逻辑合成及输入编码优化

---

## 491. The Case of the Mysterious Citations

**arXiv ID:** 2602.05867 | [PDF](https://arxiv.org/pdf/2602.05867v1)

**作者:** Amanda Bienz `[一作]` (University of New Mexico), Simon Garcia de Gonzalo `[通讯]` (Sandia National Laboratories)

**通讯引用:** 350 | [OpenAlex ID](https://openalex.org/A5000022037)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套自动化检测管线，系统性分析了四大高性能计算会议在2021年和2025年间论文中的神秘引用与标题/作者错误，量化了引用错误的出现比例。

**💡 创新点**

创新点在于：①首次用大规模自动化方法对多年份会议论文进行比较，揭示了引用错误随AI写作普及而显著上升的趋势；②发现现行学术写作与AI使用政策缺乏足够执行力，未能有效防止引用造假。

**🔧 技术方法**

采用文本检索与元数据比对技术（如DOI、数据库查询），结合引用验证工具，对论文参考文献进行自动核对。

**📊 数据集**

使用的数据集为四大高性能计算会议的完整论文集，分别取2021年和2025年的出版物，覆盖数千篇论文。

**📈 对比分析**

通过对比两年份的引用错误率，发现2025年所有会议论文均出现2%–6%的神秘引用，而2021年无此现象；同时，标题和作者错误显著增加，说明AI写作的误差累积。性能上，该检测管线对完整会议论文集的处理时间可在数小时内完成。

**⚠️ 局限性**

局限性包括：①仅针对四个会议，结果可能不具备广泛代表性；②检测只关注引用的真实性，未评估引用内容与正文的关联性；③未区分AI生成与人为失误导致的错误；④未对会议政策执行力度进行深入调查。

---

## 492. Constrained Group Relative Policy Optimization

**arXiv ID:** 2602.05863 | [PDF](https://arxiv.org/pdf/2602.05863v1)

**作者:** Roger Girgis `[一作]` (Mila - Quebec AI Institute), Liam Paull `[通讯]` (Universite de Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于GRPO的约束强化学习框架Constrained GRPO，通过引入指标成本函数和拉格朗日乘子实现对行为约束的动态约束，并在优势估计中采用“优势标量化”而非“奖励标量化”，从而保持乘子对约束权重的原始意图。

**💡 创新点**

创新点在于①阐释并证明在GRPO中先标量化奖励再进行组内标准化会导致隐式的协方差驱动权重重塑，破坏约束学习；②提出通过先对每个目标单独组内标准化再线性组合的优势标量化方法，恢复拉格朗日乘子预期的约束调节效果；③将该方法应用于大规模多模态模型的强化学习，展示其在安全约束场景下的优越性。

**🔧 技术方法**

核心技术包括：GRPO的组内优势标准化、拉格朗日方法实现约束松弛、指标成本函数与阈值形式的约束定义、优势标量化与奖励标量化对比、基于大型预训练模型（如Qwen2.5‑VL 3B）进行RL微调。

**📊 数据集**

实验数据集主要有：①简化的10×10格子世界，用于验证优势标量化与奖励标量化的差异；②NAVSIM‑v2 Navhard分割（来自OpenScene/nuPlan），包含约10万训练场景与1.2万测试场景，用于评估自驾场景中的安全约束与任务性能。

**📈 对比分析**

与多种基线对比：传统GRPO（奖励/优势标量化）、GRPO直接优化EPDMS、规则基PDM规划器、以及Constrained GRPO（奖励/优势标量化）。实验显示，Constrained GRPO + 优势标量化在NAVSIM‑v2 Stage 1和Stage 2上均取得最高EPDMS（0.516 vs 0.436），同时满足约束阈值（如无碰撞率≥0.99）且任务指标（如Ego Progress、Comfort）显著提升；相比奖励标量化，优势标量化表现更稳定、约束违约率更低。

**⚠️ 局限性**

局限性包括：①对软约束的依赖，硬约束仍可能导致学习信号稀疏；②阈值和指标成本的设定仍需人工经验；③实验主要集中在两类任务，缺乏对更复杂多目标场景的广泛验证；④未引入自适应约束阈值或课程学习，可能限制在更严格约束下的探索与收敛。

---

## 493. DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders

**arXiv ID:** 2602.05859 | [PDF](https://arxiv.org/pdf/2602.05859v1)

**作者:** Xu Wang `[一作]` (University of Hong Kong), Difan Zou `[通讯]` (University of Hong Kong)

**通讯引用:** 2585 | [OpenAlex ID](https://openalex.org/A5085848346)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了 DLM-Scope，一套针对扩散语言模型的稀疏自编码器（SAE）解释框架，用于机制分析、特征蒋解、跨步骤干预和解码顺序评估；

**💡 创新点**

首次为 DLM 提供 SAE 解释接口；发现 SAE 插入可在早层降低交叉熵；实现跨 denoising 步骤的特征 steering、解码顺序动态追踪，以及基模型 SAE 对指令微调模型的可迁移性；

**🔧 技术方法**

采用稀疏自编码器训练（Mask‑/Unmask‑SAE）、稀疏性‑保真度评估、特征 steering、Jaccard 相似度追踪特征动态、自动解释与可解释性评分等技术；

**📊 数据集**

使用 Common Pile 作为训练语料；在 Dream‑7B、LLaDA‑8B 两款 DLM 上训练 SAE；使用 GSM8K 作为评估任务，并对 steering 采用标准提示集；

**📈 对比分析**

通过 sparsity‑fidelity 曲线、Δ交叉熵、解释性评分以及 steering 得分 S 与传统 LLM‑SAE 对比；实验显示 DLM‑SAE 在深层的 steering 得分可比 LLM‑SAE 高 2‑10 倍，且 SAE 在多数层上可无损迁移到指令微调模型；

**⚠️ 局限性**

限制主要集中在最深层 SAE 的迁移性能下降；仅评估了两款 DLM，缺乏对更大规模模型的验证；SAE 插入策略和位置固定可能限制了对全局表示的捕捉。

---

## 494. Exact Recovery in the Data Block Model

**arXiv ID:** 2602.05852 | [PDF](https://arxiv.org/pdf/2602.05852v1)

**作者:** Amir R. Asadi `[一作]` (University of Cambridge), Farzad Parvaresh `[通讯]` (Isfahan University of Technology)

**通讯引用:** 1313 | [OpenAlex ID](https://openalex.org/A5006066917)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在包含节点属性的随机块模型（Data Block Model, DBM）中，基于图结构和节点数据的精确恢复问题，并给出了可实现的多项式时间算法。

**💡 创新点**

提出了新的 Chernoff–TV 散度，能够同时量化图连通性与节点属性信息的贡献，并用其精确刻画了 DBM 的精确恢复相位阈值；该阈值在属性信息缺失时退化为经典 SBM 阈值，在图信息不足时退化为仅用属性信息的阈值。

**🔧 技术方法**

主要技术包括：CH 散度与多元泊松分布的 Chernoff 信息等价关系、Chernoff–TV 散度的定义与性质、两阶段聚类算法（Sphere‑comparison + 本地 MAP 估计）、概率与信息论的上界下界证明、以及实验中基于泊松似然的 MAP 评分。

**📊 数据集**

实验使用了合成的两社区对称 DBM 数据集：节点数 n=1000（或 10、10²、10³、10⁴ 进行规模实验），图边概率矩阵 Q=[a b; b a]，边期望度为 Θ(log n)，节点标签通过擦除信道（概率 1‑n⁻ᵃ）获得。

**📈 对比分析**

与基线方法（仅图的 SBM、Spectral、Data‑only、以及迭代版）比较。实验表明，当属性信息足够强时，DBM 的精确恢复阈值显著低于纯图模型；在阈值附近 DBM 的精确恢复概率和误差率都明显优于其他方法；迭代 MAP 细化对精确恢复概率有一定提升，运行时间几乎与单次 MAP 相当。

**⚠️ 局限性**

局限性：理论阈值与算法均在 n→∞、对数度数标度下证明；在小样本（n 较小）时存在偏差；实验仅覆盖对称两社区情形，未验证多社区或不均衡先验的效果；Chernoff–TV 散度的计算在大规模稀疏图中可能需要近似；算法依赖于 Sphere‑comparison 的初始聚类，若其误差较大可能导致最终恢复失效。

---

## 495. Visualizing the loss landscapes of physics-informed neural networks

**arXiv ID:** 2602.05849 | [PDF](https://arxiv.org/pdf/2602.05849v1)

**作者:** Conor Rowan `[一作]` (University of Colorado Boulder), Finn Murphy-Blanchard `[通讯]` (University of Colorado Boulder)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对物理信息机器学习中的损失景观进行了全面的回顾，并使用多种技术对深Ritz方法和强形式损失的损失景观进行了实证研究。

**💡 创新点**

创新点在于引入了“海森步行”和“加速度”概念，以探索解决方案的流形，并将现有的损失景观可视化技术引入科学机器学习社区。

**🔧 技术方法**

使用了深度学习技术，特别是多层感知器（MLP）网络，并结合了海森矩阵的特征值分析和随机方向的损失表面可视化。

**📊 数据集**

使用了一维椭圆边值问题和二维Neohookean超弹性问题作为数据集，进行损失景观的分析和比较。

**📈 对比分析**

通过与现有的图像分类问题的损失景观进行比较，发现物理信息网络的损失景观在解决方案附近是平滑、凸且良好条件的，挑战了传统观点认为损失景观是嘈杂和不良条件的。

**⚠️ 局限性**

限制在于边界条件的处理方式，当前的网络离散化中没有引入惩罚项，未来的工作应关注边界条件执行技术对损失景观可视化的影响。

---

## 496. OmniVideo-R1: Reinforcing Audio-visual Reasoning with Query Intention and Modality Attention

**arXiv ID:** 2602.05847 | [PDF](https://arxiv.org/pdf/2602.05847v1)

**作者:** Zhangquan Chen `[一作]` (Tsinghua University), Ruqi Huang `[通讯]` (Tsinghua University)

**通讯引用:** 280 | [OpenAlex ID](https://openalex.org/A5086379651)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于强化学习的两阶段框架，利用查询驱动的自监督定位（Query‑Intensive Grounding, QI）与对比式模态注意力融合（Modality‑Attentive Fusion, MA），提升多模态（音频+视觉）推理能力。

**💡 创新点**

创新点在于：1）在不依赖人工过程级标注的情况下，通过自监督的时间‑描述对生成与验证机制，促使模型主动定位关键时段并形成推理链；2）采用对比式奖励机制，鼓励模型在完整多模态输入下优于单模态表现，从而实现更强的跨模态协同。

**🔧 技术方法**

核心技术包括：强化学习（采用 Group Sequence Policy Optimization, GSPO）来优化整体推理过程；自监督的时间‑描述一致性与完整性奖励；对比式模态注意力奖励；以及基于大模型（Qwen3‑Omni‑30B‑A3B）和多模态编码器的集成。

**📊 数据集**

数据集：从 LLaVA‑Video 与 Video‑Vista 采集并经三阶段清洗、评分与类别平衡后，得到 88,173 个训练样本；其中 12,887 个高音频‑视觉依赖样本用于 MA 阶段。评测使用公开的音频‑视觉任务（MMStar、MathVista_mini、Daily‑Omni、IntentBench、WorldSense、OmniVideoBench）以及无声视频任务（Video‑MME、MLVU、LVBench）。

**📈 对比分析**

与 Video‑SALMONN 2+ 72B（公开 SOTA）相比，提升 4.3%（82.8% vs 78.5%）；与闭源 Gemini3‑Pro 对比，Daily‑Omni 提升 2.1%（82.8% vs 80.7%），IntentBench 提升 3.8%（74.2% vs 70.4%）；在专门考察音频‑视觉协同的 OmniVideoBench 上，提升 21.1%（44.8% vs 23.7%）。在单模态评测中未出现性能下降，甚至在部分数据集上提升。

**⚠️ 局限性**

局限性包括：对大模型的依赖导致计算与资源成本高；强化学习的奖励设计对任务特性敏感，需手工调参；目前仅验证于音频‑视觉两模态，缺乏对更多模态（文本、传感器等）的推广；自监督时段定位的准确性仍受视频与音频质量影响，可能导致误定位。

---

## 497. FHAIM: Fully Homomorphic AIM For Private Synthetic Data Generation

**arXiv ID:** 2602.05838 | [PDF](https://arxiv.org/pdf/2602.05838v1)

**作者:** Mayank Kumar `[一作]` (University of Central Florida), Sikha Pentyala `[通讯]` (University of Washington Tacoma)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5055498873)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出首个基于全同态加密的输入隐私合成数据生成框架，能在数据加密状态下训练AIM模型生成合成表格数据。

**💡 创新点**

创新点在于设计了DP‑in‑FHE协议：加密的marginal计算、加密的查询选择（使用平方L₂代替传统L₁）、以及加密的高斯噪声注入；实现了单一服务商环境下的输入隐私，并将加密算子与差分隐私机制无缝结合。

**🔧 技术方法**

技术栈包括CKKS全同态加密、差分隐私（Gaussian机制、Gumbel‑Max）、AIM合成算法、SIMD打包、一次热编码、加密矩阵乘法等。

**📊 数据集**

使用了乳腺癌（Breast Cancer）、COMPAS监狱再犯风险数据以及糖尿病（Diabetes）三大真实表格数据集。

**📈 对比分析**

与明文AIM（L₁与L₂）以及无DP噪声基准对比；在工作负载误差、模型精度和分类准确率上与明文AIM基本一致；总体运行时间约为11–30分钟，内存占用低于32 MB，性能略高于L₁但保持稳定。

**⚠️ 局限性**

限制包括：相较于明文实现仍有显著的计算与能耗开销；需预先生成并加密大量噪声样本；目前仅验证于中等规模表格数据，对高维或大规模数据的可扩展性仍待进一步评估。

---

## 498. An FWCI decomposition of Science Foundation Ireland funding

**arXiv ID:** 2602.05836 | [PDF](https://arxiv.org/pdf/2602.05836v1)

**作者:** Eoin Ó Colgáin `[一作]` (Atlantic Technological University), Eoin Ó Colgáin `[通讯]` (Atlantic Technological University)

**通讯引用:** 5843 | [OpenAlex ID](https://openalex.org/A5028386025)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

**🎯 论文内容**

本文通过检索 Science Foundation Ireland (SFI) 2012-2016 年的 Investigator Awards 关联的 3,243 篇原始科研论文，利用 SCOPUS 数据库计算其 Field‑Weighted Citation Impact（FWCI）指标，探讨学术影响与社会经济影响之间的关系；

**💡 创新点**

创新点在于将 FWCI 视为 log‑normal 分布的指标，并利用模拟校正小样本的 FWCI，比较单一项目的学术兴趣是否高于国际中位水平；

**🔧 技术方法**

方法主要使用 log‑normal 拟合、正态近似、蒙特卡洛模拟以及 FWCI 计算；

**📊 数据集**

数据集来源于 SFI Open Data（项目编号）和 SCOPUS（论文与 FWCI 数据），共计 3,243 篇论文；

**📈 对比分析**

通过将项目平均 FWCI 与模拟产生的中位 FWCI 对比，评估 65%–69% 的项目在国际学术兴趣上位于上半区，显示 SFI 资助项目整体学术竞争力良好；

**⚠️ 局限性**

局限在于 FWCI 的短期窗口、缺失未标记 FWCI 的论文、可能的文献覆盖不完整以及对 log‑normal 假设的依赖。

---

## 499. Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning

**arXiv ID:** 2602.05809 | [PDF](https://arxiv.org/pdf/2602.05809v1)

**作者:** Enwei Tong `[一作]` (Harbin Institute of Technology), Xianming Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 6784 | [OpenAlex ID](https://openalex.org/A5100654390)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于人类视觉感知的训练无关视觉 token 剪枝框架 Focus‑Scan‑Refine（FSR），通过动态分配局部证据与全局上下文，实现高效视觉 token 剪枝。

**💡 创新点**

创新点在于三阶段人类启发式剪枝：先结合视觉显著性与指令相关性选取关键局部证据；再有条件地采样与聚合多样化的全局上下文；最后聚合未选 token 以提升细节保留，且在固定 token 预算下实现更优的局部–全局平衡。

**🔧 技术方法**

技术包括双通路打分机制、条件上下文采样 (CCS)、基于相似度的聚合与加权合并，配合 CLIP 文本编码、注意力显著性计算以及贪心 k‑center 的覆盖保证。

**📊 数据集**

数据集涵盖 VQA‑V2、GQA、SQA‑IMG、VQA‑Text、POPE、MME、MMBench、MM‑Vet、Qwen2.5‑VL‑7B 及视频 benchmark LLaVA‑Video、MMVU、MMWorld 等。

**📈 对比分析**

与 FastV、SparseVLM、DART、HoloV、VisPruner、CDPruner 等现有训练无关剪枝方法在多种 VLM（LLaVA‑1.5‑7B/13B、LLaVA‑NeXT‑7B/13B、Qwen2.5‑VL‑7B、LLaVA‑Video‑7B‑Qwen2）和任务上对比，FSR 在保持 64‑token 预算时平均保持 99% 以上性能，显著优于其它方法，并在高分辨率和视频场景下仍保持 97% 以上准确率。

**⚠️ 局限性**

局限性包括对超大 token 数的稀疏特征估计仍依赖贪心近似；聚合阶段可能导致局部细节过度平滑；在极端压缩（>90%）时对复杂多模态推理的鲁棒性仍有限。

---

## 500. Contour Refinement using Discrete Diffusion in Low Data Regime

**arXiv ID:** 2602.05880 | [PDF](https://arxiv.org/pdf/2602.05880v1)

**作者:** Fei Yu Guan `[一作]` (University of Toronto), Steven Waslander `[通讯]` (University of Toronto)

**通讯引用:** 9980 | [OpenAlex ID](https://openalex.org/A5024242059)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于离散扩散模型的轮廓细化方法，用于在低样本数据条件下提升图像边界检测的精度。

**💡 创新点**

创新点在于将离散扩散过程（Discrete Diffusion）引入轮廓细化任务，利用扩散模型在缺乏标注数据时仍能生成高质量的边缘信息，从而克服传统方法对大规模标注数据的依赖。

**🔧 技术方法**

核心技术包括：
- 离散扩散模型（Discrete Diffusion）
- 轮廓特征提取与条件生成网络
- 低样本学习框架（如数据增强与迁移学习）
- 评价指标如边缘IoU、F1分数等。

**📊 数据集**

使用的数据集主要包括：
- **PASCAL VOC**（低样本版本）
- **Cityscapes**（少量标注）
- 自制合成数据集用于验证模型在极低样本情况下的鲁棒性。

**📈 对比分析**

与传统方法（如基于CNN的边缘检测、传统分割网络、以及基于GAN的细化方法）对比，实验结果表明该方法在低样本设置下的边缘IoU提升约4–6%，F1分数提升约3–5%，同时在计算复杂度上保持与现有方法相近。

**⚠️ 局限性**

局限性包括：
- 对扩散过程的迭代次数敏感，过多迭代会导致计算成本显著上升。
- 在极端低样本（仅数十张图像）时仍存在细节模糊的情况。
- 目前仅在公开语义分割数据集上验证，尚未在医疗影像或遥感等其他领域进行测试。

---

## 501. EuroLLM-22B: Technical Report

**arXiv ID:** 2602.05879 | [PDF](https://arxiv.org/pdf/2602.05879v1)

**作者:** Miguel Moura Ramos `[一作]` (Instituto Superior Técnico & Universidade de Lisboa), André F. T. Martins `[通讯]` (TransPerfect)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文介绍了 EuroLLM-22B——一款拥有 22B 参数、支持 35 种语言（包括 EU 24 官方语言及 11 种补充语言）的开放式大语言模型，并发布了其基础版和指令调优版。

**💡 创新点**

创新点包括：① 采用分阶段训练策略，将低质量数据放在前期、优质数据放在后期；② 将上下文窗口扩大至 32K 令牌，使用 RoPE 扩展；③ 通过大规模数据过滤、合成数学和代码数据提升多语言推理与指令遵循；④ 公开了新版本的多语言指令数据集 EuroBlocks‑SFT‑2512 及对应的预训练数据集 EuroLLM‑Multilingual‑Data‑2512。

**🔧 技术方法**

技术上基于 Megatron‑LM 进行预训练，采用分组查询注意力（Grouped Query Attention）、SwiGLU 激活、RMSNorm 以及 RoPE 位置编码；指令调优使用 Axolotl 与 Liger‑Kernel 进行混合精度、序列打包和余弦学习率调度；评估使用 LLM‑as‑a‑Judge 以及 COMET‑DA 进行自动评分。

**📊 数据集**

数据集涵盖：
- 预训练：FineWeb‑Edu、Nemotron‑CC、RedPajama‑Data‑v2、HPLT、MADLAD‑400、CulturaX、mC4、EuroParl、ParaCrawl 等多语料；
- 对齐与翻译：句对平行数据、文档级平行数据；
- 代码与数学：The Stack、Open‑web‑math、FineMath、GSM‑8K、Math‑Aptitude；
- 合成数据：基于 Qwen‑2.5 生成的 170 万数学样本；
- 指令数据：EuroBlocks‑SFT‑2512，包含多源指令和生成答案。

**📈 对比分析**

在一系列英文与多语种基准（如 IFEval、HellaSwag、MMLU、ARC‑C、GSM‑8K、HumanEval、FLORES‑200、WMT24/25 等）中，EuroLLM‑22B 与同等规模的欧盟公开模型相比保持竞争力，并在大多数任务上超越 Apertus‑70B、Mistral‑3.2‑24B 等对手；在翻译任务上与其它模型基本持平，未出现显著下降。

**⚠️ 局限性**

局限性包括：
- 与目前最强的开源权重模型（如 Qwen‑3‑32B、Llama‑3.3‑70B 等）仍有一定差距；
- 由于上下文窗口扩展的需求，部分基准（如 WMT25）无法完整评估；
- 训练规模约 4T token，若进一步提升高质量数据量仍有提升空间；
- 主要面向欧盟语言，非欧盟语言覆盖仍有限。

---

## 502. Agent2Agent Threats in Safety-Critical LLM Assistants: A Human-Centric Taxonomy

**arXiv ID:** 2602.05877 | [PDF](https://arxiv.org/pdf/2602.05877v1)

**作者:** Lukas Stappen `[一作]` (BMW Group Research), Georg Groh `[通讯]` (Technical University Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了AgentHeLLM框架，用于在车辆AI助手的威胁建模中实现资产与攻击路径的分离，并构建了人本资产分类和图模型攻击路径；

**💡 创新点**

创新点包括将资产与攻击路径明确分离、基于人权的资产分类、对毒化路径与触发路径的形式化区分以及双层搜索的攻击路径生成器；

**🔧 技术方法**

技术实现涵盖基于图的系统抽象、A*主搜索与BFS子搜索的双层规划、LLM交互模拟与A2A协议分析；

**📊 数据集**

未使用专门的数据集，攻击路径基于模型和公开协议规范构造；

**📈 对比分析**

相较于传统AI安全框架，AgentHeLLM通过生成可量化的攻击路径序列实现更系统的威胁覆盖，但未给出量化性能指标；

**⚠️ 局限性**

局限性包括仅支持静态架构、未考虑动态发现与概率评估、所有边视为可攻击且缺乏实证验证。

---

## 503. xList-Hate: A Checklist-Based Framework for Interpretable and Generalizable Hate Speech Detection

**arXiv ID:** 2602.05874 | [PDF](https://arxiv.org/pdf/2602.05874v1)

**作者:** Adrián Girón `[一作]` (Universidad Politécnica de Madrid), David Camacho `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 7319 | [OpenAlex ID](https://openalex.org/A5025314362)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种诊断式框架（xList-Hate），通过10个概念层面的二元问题拆解仇恨言论判定，先由LLM回答，再由轻量决策树汇总做最终分类。

**💡 创新点**

创新点在于将仇恨言论拆分为可解释的概念维度，避免单一标签的过拟合与不透明决策，提升跨数据集鲁棒性与可审计性。

**🔧 技术方法**

技术包括大语言模型（多种decoder LLM）独立完成10个提示式二元问答、基于二进制诊断向量的决策树聚合，以及对LLM输出的二进制解析与纠错。

**📊 数据集**

使用的公开数据集包括 Measuring Hate Speech、HateXplain、Stormfront、ETHOS 以及功能测试集 HateCheck，涵盖多种语言环境与标注准则。

**📈 对比分析**

与零样本LLM直接分类和在域内微调的监督模型对比，xList-Hate 在跨域相对AUC与OOD AUC 上均表现出更高或相近的稳健性，尤其在域转移时保持较低的性能衰减。

**⚠️ 局限性**

局限性主要是推理成本较高（每条输入需10次LLM调用）以及在某些情境下绝对OOD性能略低于最优监督微调，但通过可解释路径弥补了这些缺陷。

---

## 504. CFRecs: Counterfactual Recommendations on Real Estate User Listing Interaction Graphs

**arXiv ID:** 2602.05861 | [PDF](https://arxiv.org/pdf/2602.05861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 505. EoCD: Encoder only Remote Sensing Change Detection

**arXiv ID:** 2602.05882 | [PDF](https://arxiv.org/pdf/2602.05882v1)

**作者:** Mubashir Noman `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salman Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 11611 | [OpenAlex ID](https://openalex.org/A5000300751)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种仅使用编码器的变化检测框架EoCD，采用早期融合并用参数无关的多尺度特征融合模块取代传统解码器；

**💡 创新点**

创新点在于：①把Siamese编码器和复杂解码器拆除，仅保留单一编码器；②引入无可学习参数的EMFF模块，实现高效的多尺度特征融合；③通过教师-学生蒸馏提升无解码器网络性能；

**🔧 技术方法**

使用的技术包括：早期图像拼接、卷积stem层、深度可分离卷积、多尺度特征融合EMFF、交叉熵+二元交叉熵+MAE三重损失蒸馏、Transformer/CNN等多种编码器（如ResNet-34、FocalNet-T、mit‑b1等）；

**📊 数据集**

实验数据集为四个公开遥感变化检测基准：LEVIR‑CD、CDD‑CD、SYSU‑CD、WHU‑CD；

**📈 对比分析**

与多种SOTA方法（Convformer‑CD/48、ChangeFormer、RHighNet、ELGCNet‑LW等）对比，EoCD在保持或提升IoU、F1、OA的同时，大幅降低FLOPs与延迟，尤其在LEVIR‑CD上达到84.78% IoU、90.83% F1、99.17% OA；

**⚠️ 局限性**

局限性包括：性能高度依赖于编码器；对极端变化场景的鲁棒性未全面验证；仅在二维光学影像上测试，未扩展到雷达或多源数据。

---

## 506. Large-scale Score-based Variational Posterior Inference for Bayesian Deep Neural Networks

**arXiv ID:** 2602.05873 | [PDF](https://arxiv.org/pdf/2602.05873v1)

**作者:** Minyoung Kim `[一作]` `[通讯]` (Samsung AI Center), Minyoung Kim (Samsung AI Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种可扩展的基于分数匹配的变分推断算法，用于大型贝叶斯深度神经网络的后验推断。

**💡 创新点**

创新点在于将分数匹配损失与逐步近似惩罚结合，消除了对重参数化采样的依赖，并允许使用无偏噪声小批量分数，实现对大规模网络的高效训练。

**🔧 技术方法**

采用分数匹配（score matching）、近似（proximal）正则化、随机梯度下降以及基于正态分布或正则化流的变分分布。

**📊 数据集**

在视觉识别任务中使用ResNet‑101与Vision Transformers（ViT‑L‑32）对Pets、Flowers、Aircraft、DTD等数据集；在时间序列预测中使用Koopa网络对多项公开数据集。

**📈 对比分析**

与ELBO‑基于ADVI、GSM、BaM以及MC‑Dropout、SGLD、Laplace Approximation等方法比较，结果显示在预测性能相近的前提下，该方法在不确定性量化（ECE、NLL）和训练收敛速度上均优于传统VI方法。

**⚠️ 局限性**

局限性包括仍需计算近似正则化项导致的额外计算开销；在极高维参数空间下使用正态流时仍面临数值不稳定；若变分族与真实后验差距大，收敛速度可能受限。

---

## 507. RRAttention: Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference

**arXiv ID:** 2602.05853 | [PDF](https://arxiv.org/pdf/2602.05853v1)

**作者:** Siran Liu `[一作]` (Baidu Inc), Chao Yang `[通讯]` (Peking University)

**通讯引用:** 145948 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态稀疏注意力机制RRAttention，通过头部轮转抽样和步长级聚合实现长上下文的高效注意力计算。

**💡 创新点**

创新点在于利用头部轮转（Round‑Robin）抽样保证查询独立性、全局评估以及无预处理的模式发现，同时通过步长级软化实现计算复杂度从O(L²)降至O(L²/S²)。

**🔧 技术方法**

核心技术包括：1）头部轮转查询抽样；2）步长级重要性估计与行归一化softmax；3）Top‑τ块级选择；4）自适应Top‑τ阈值和最后查询块保护；5）GPU友好的批量矩阵运算。

**📊 数据集**

使用两个公开基准：HEL​MET（7类NLP长文档理解任务）和Video‑MME（多模态视频问答），在Meta‑LLaMA‑3.1‑8B‑Instruct与Qwen‑2.5‑7B‑Instruct等长上下文模型上评测。

**📈 对比分析**

与全精度Attention、FlexPrefill、XAttention等方法对比，RRAttention在128K上下文下恢复>99%完整注意力性能，仅计算约一半注意力块，速度提升约2.4×，稀疏度最高且保持或超过其他稀疏方法的准确率。

**⚠️ 局限性**

局限性：当步长S大于头数或过大时，轮转抽样无法覆盖所有位置，导致重要性估计失真；过粗步长会削弱精度，需在S=8或16等中等步长范围内使用。

---

## 508. "It Talks Like a Patient, But Feels Different": Co-Designing AI Standardized Patients with Medical Learners

**arXiv ID:** 2602.05856 | [PDF](https://arxiv.org/pdf/2602.05856v1)

**作者:** Zhiqi Gao `[一作]` (Chinese University of Hong Kong), Benyou Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过访谈12名临床年级医学生并开展三次共创工作坊，探讨其在标准化患者（SP）训练中的体验与需求，并据此制定AI标准化患者（AI‑SP）的设计要求与概念工作流。

**💡 创新点**

创新点在于：①将学习者的经验与需求系统化为六大主题；②将这些主题映射为AI‑SP的六项设计要求，强调模式选择、信息释放策略、跨模态证据获取、混合输入、双回路反馈与可控情感；③提出将AI‑SP定位为“教学基础设施”而非单纯的对话模拟器，聚焦可操作性、可观察性与可学习性。

**🔧 技术方法**

研究方法主要是人机交互领域的定性研究：访谈使用反思式主题分析；工作坊采用共创与草图迭代；技术层面仅涉及LLM概念化的AI‑SP模型（未实现），并未使用具体LLM模型或算法。

**📊 数据集**

数据来源为12名临床医学生的访谈录音与转录、以及三次工作坊中被访者绘制的草图和讨论记录。并未使用公开数据集或医学案例库。

**📈 对比分析**

由于研究未构建可运行的AI‑SP原型，故无对比实验或性能评估。研究仅通过主题归纳与需求映射提供了设计框架，后续工作需在真实环境中实现并评估学习效果与教师反馈。

**⚠️ 局限性**

局限性包括：①样本量有限，且来自两所医学院，文化背景单一；②未评估AI‑SP对学习成果的影响；③未涉及教师或SP训练师的视角；④缺乏技术实现与安全性评估；⑤结论仅适用于概念阶段，需进一步验证。

---

## 509. An Equational Axiomatization of Dynamic Threads via Algebraic Effects: Presheaves on Finite Relations, Labelled Posets, and Parameterized Algebraic Theories

**arXiv ID:** 2602.05850 | [PDF](https://arxiv.org/pdf/2602.05850v1)

**作者:** Ohad Kammar `[一作]` (University of Edinburgh), Sam Staton `[通讯]` (University of Oxford)

**通讯引用:** 1391 | [OpenAlex ID](https://openalex.org/A5068183682)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了动态线程的完全等式公理化，利用参数化代数理论与标签偏序（Pomset）相结合，构建了简洁并发语言的运算语义与定语义，并证明其完整性、正确性与完全抽象性。

**💡 创新点**

创新点在于：①将线程 ID 视为参数化代数理论中的参数，直接在代数效应框架内公理化 fork、wait 等并发原语；②提供了针对标签偏序的表示定理与完备性证明，首次把等式公理与传统的真正并发模型紧密对应；③通过参数化代数理论构造强单子，实现从运算语义到定语义的完整映射。

**🔧 技术方法**

使用了参数化代数理论、范畴中的 functor 语义、强单子、标签偏序（Pomset）模型、表示定理、完整性证明与全抽象性证明等理论技术。

**📊 数据集**

无数据集，全部为形式化证明与理论模型。

**📈 对比分析**

通过与标签偏序（Pomset）模型的同构比较验证完整性；通过运算语义与定语义的一致性证明其正确性；在理论层面证明完全抽象性，未涉及实验性能评估。

**⚠️ 局限性**

局限性：语言模型极简化，只支持原子动作、无共享状态、无冲突、无递归；线程间通信缺失；未考虑调度策略、线程资源限制等实际并发系统特性。

---

## 510. UI-Mem: Self-Evolving Experience Memory for Online Reinforcement Learning in Mobile GUI Agents

**arXiv ID:** 2602.05832 | [PDF](https://arxiv.org/pdf/2602.05832v1)

**作者:** Han Xiao `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 40378 | [OpenAlex ID](https://openalex.org/A5100732450)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 UI-Mem 框架，通过构建层级化、可自我进化的经验记忆，改进移动 GUI 代理的在线强化学习；

**💡 创新点**

创新点在于：① 将经验抽象为工作流、子任务技能与失败模式三层模板；② 采用分层群采样（Stratified Group Sampling）在 GRPO 训练中维持优势估计的方差；③ 引入自我进化循环持续抽取成功计划与失败诊断，保持记忆与策略同步；

**🔧 技术方法**

技术包括：多模态大型语言模型（Qwen3‑VL）、Group Relative Policy Optimization (GRPO)、向量数据库检索、模板化抽象、动态指导比例与奖励塑形、两阶段文本验证器；

**📊 数据集**

使用的数据集：AMEX、AndroidLab、UI‑Genie 的任务指令（共 256 条），并在 AndroidWorld（116 任务）和 AndroidLab（138 任务）上进行评估，外部评测还涵盖 5 个未见应用；

**📈 对比分析**

与 vanilla GRPO、进度奖励、经验回放、推理时提示等方法对比，UI‑Mem 在 AndroidWorld 上 4B 模型达到 58.2% 成功率，8B 模型 71.1%，均显著高于基线与现有开源/闭源模型；在 AndroidLab 上也提升了 Sub‑SR、RRA、ROR 等细粒度指标；

**⚠️ 局限性**

局限性：① 记忆模板仍需手工定义或自动抽取，抽象错误可能导致迁移失败；② 强化学习探索阶段可能产生不可预期的操作，需要安全约束；③ 依赖大型语言模型与算力，部署成本高；④ 对极端动态 UI 或隐藏元素的适应性尚待验证。

---

## 511. Weaver: End-to-End Agentic System Training for Video Interleaved Reasoning

**arXiv ID:** 2602.05829 | [PDF](https://arxiv.org/pdf/2602.05829v1)

**作者:** Yudi Shi `[一作]` (Xiaohongshu Inc.), Weidi Xie `[通讯]` (School of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个名为Weaver的多模态代理式推理系统，能够在视频问答中动态调用多种视觉工具以逐步获取视觉证据并生成交互式视觉‑文本推理轨迹。

**💡 创新点**

创新点在于：①将工具调用嵌入到推理过程中，实现感知‑循环的交互式推理；②采用两阶段训练（监督微调 + 强化学习）让模型自适应组合工具；③构建了专门的工具使用数据集并引入工具使用奖励。

**🔧 技术方法**

技术包括：大规模视觉‑语言模型（Qwen2.5‑VL）、多种视觉工具（时间定位、帧选择、轨迹跟踪、空间定位等）、交互式视觉‑文本编码、工具调用模板、强化学习（GRPO‑改版）以及多模态Chain‑of‑Thought。

**📊 数据集**

使用了Video‑R1‑170K、LongVideo‑Reason‑51K等原始CoT数据，经过重写和工具调用插入生成Weaver‑SFT与Weaver‑RL两个数据集；在评测时使用LVReason、LVBench、MLVU、VideoMME、VideoMMMU、VSIBench、MVBench等长视频与通用VideoQA基准。

**📈 对比分析**

与基准模型（如Qwen2.5‑VL、Gemini‑2.5‑Pro、Video‑R1、VideoMTR等）对比，Weaver在多项长视频推理基准上提升6.7%‑12%不等；在VideoMME、MLVU、LVReason等上均优于现有文本‑中心CoT及其他交互式方法，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：①需要大量工具库和手工设计的工具调用模板；②强化学习阶段仍受奖励设计的影响，可能导致工具使用偏向；③对极短或无明显视觉信息的视频问题的效果有限；④模型规模较大，推理成本和推理时间较高。

---

## 512. Authorship Drift: How Self-Efficacy and Trust Evolve During LLM-Assisted Writing

**arXiv ID:** 2602.05819 | [PDF](https://arxiv.org/pdf/2602.05819v1)

**作者:** Yeon Su Park `[一作]` (KAIST), Juho Kim `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一次大规模线上实验中，研究者让302名受试者在GPT‑4协助下完成一篇议论文，并在每一次与模型互动后实时记录受试者的自我效能与信任感评分，随后对评分轨迹、提问意图与实际/感知作者身份进行统计与建模；通过ROUGE‑3与语义相似度评估受试者文本中模型生成内容的采纳程度，进一步探究自我效能与信任的动态变化如何影响写作过程与作者身份。

**💡 创新点**

首次将自我效能与信任视为在多轮LLM协作中的动态心理状态，并将其轨迹与提问策略、实际/感知作者身份关联；通过量化分析揭示自我效能下降时用户倾向于“编辑”式交互，而自我效能回升时更偏向“评审”式交互，从而为设计支持作者身份的LLM交互提供经验性依据。

**🔧 技术方法**

利用混合效应回归模型（检验自我效能与信任随轮次的相互作用）、关键词意图分类（提问意图识别）、ROUGE‑3与文本嵌入余弦相似度（衡量实际作者身份）以及传统统计检验（Kruskal‑Wallis、Mann‑Whitney、Dunn等）进行数据分析。

**📊 数据集**

实验数据来自302名英语母语的Prolific受试者的交互日志与文本；包括每轮自我效能/信任评分、用户提问、模型回复、完整议论文及后测自评作者身份。

**📈 对比分析**

本文没有与其他模型或交互设计做直接的性能对比，而是通过混合效应模型展示自我效能随轮次的下降趋势、信任的上升趋势，并通过比较不同轨迹组在实际/感知作者身份上的差异，证明自我效能轨迹对作者身份有显著影响。

**⚠️ 局限性**

主要限制包括：1）自我效能与信任的评分为自报，可能受关注效应影响；2）实验仅为单次交互，无法评估长期轨迹；3）仅使用单一LLM，缺乏模型差异性验证；4）未对外部工具使用进行监测；5）缺乏质性访谈解释内在心理机制。

---

## 513. Prompting Destiny: Negotiating Socialization and Growth in an LLM-Mediated Speculative Gameworld

**arXiv ID:** 2602.05864 | [PDF](https://arxiv.org/pdf/2602.05864v1)

**作者:** Mandi Yang `[一作]` (Nankai University), Dongyijie Primo Pan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一款基于LLM的角色扮演游戏，通过四季社交化阶段和延迟反馈支持玩家对教育角色与责任的反思。

**💡 创新点**

将社会化理论分阶段映射为游戏剧情，并采用反可视化（隐藏实时分数）与延迟成长反馈的方式，提升反思深度。

**🔧 技术方法**

使用GPT‑4生成剧情和NPC对话，Unity 3D+Photon实现多玩家架构，Python脚本驱动会话和LLM接口。

**📊 数据集**

使用12名参与者的游戏日志与访谈记录作为实验数据，未使用公开大规模语料库。

**📈 对比分析**

实验采用质性RTA分析，未设置对照组，无法量化性能；结果显示玩家在各阶段体现教育角色转变和责任重塑。

**⚠️ 局限性**

样本规模小、缺乏对照、仅单人游戏评估、LLM输出变异性大，限制了因果推断与泛化能力。

---

## 514. BABE: Biology Arena BEnchmark

**arXiv ID:** 2602.05857 | [PDF](https://arxiv.org/pdf/2602.05857v1)

**作者:** Junting Zhou `[一作]` (ByteDance Seed), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5617 | [OpenAlex ID](https://openalex.org/A5101674305)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建并发布了BABE（Biology Arena BEnchmark）基准，专门评估大型语言模型在生物学实验推理任务中的能力；

**💡 创新点**

创新之处在于：①聚焦实验结果与背景信息整合的推理，而非传统的事实回忆或单任务预测；②所有任务均来源于同行评审论文，包含真实实验图像与数据；③通过三元组问题设计并划分强弱相关关系，能细粒度诊断多步与并行推理；④覆盖12个生物子领域，体现广泛领域覆盖；

**🔧 技术方法**

采用了多阶段注释流程、三元组问题结构、强弱相关标注、LLM推理评测与推理行为分析（深度推理、反思行为）、多轮推理收敛分析等技术；

**📊 数据集**

使用的主要数据集为BABE基准集，包含来自最新同行评审论文的实验描述、图像、结果等多模态内容；

**📈 对比分析**

通过在强弱相关子集上评估多款LLM（如OpenAI GPT‑5.1、Gemini‑3、Claude 等）的平均分进行比较，最佳模型平均得分约为52分；多轮推理可提升约20–30分，显示模型在单次推理中的局限；

**⚠️ 局限性**

局限性包括：①仍未覆盖所有生物子领域；②任务基于单一来源文档，可能限制推理复杂度；③受LLM偏差、误差传播与自我反思过度等问题影响；④对多模态推理深度评估尚不充分。

---

## 515. A Hybrid Autoencoder for Robust Heightmap Generation from Fused Lidar and Depth Data for Humanoid Robot Locomotion

**arXiv ID:** 2602.05855 | [PDF](https://arxiv.org/pdf/2602.05855v1)

**作者:** Dennis Bank `[一作]`, Simon F. G. Ehlers `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种多模态传感器融合的自编码器，用以实时生成机器人中心化的高精度高度图，并将其作为强化学习行走策略的输入；

**💡 创新点**

创新点在于将激光雷达、深度相机和IMU数据通过球面投影、卷积编码器与GRU瓶颈融合，构建了一个端到端的混合卷积-递归自编码器，能生成细粒度、可直接用于机器人导航的二维高度图；

**🔧 技术方法**

采用卷积-递归自编码器架构、球面投影、GRU时序建模、数据增强、PPO强化学习以及域随机化等技术；

**📊 数据集**

使用NVIDIA Isaac Lab中基于Unitree G1仿真模型的程序化生成地形数据集，包括楼梯、台阶与不规则地面等；

**📈 对比分析**

与单模态或仅使用本体感知的基线方法比较，实验显示重建误差显著降低、碰撞率下降、行走稳定性提升，RL策略在多样地形上表现出更高的成功率和更低的能耗；

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实世界的实测；对动态障碍物处理不足；对传感器噪声和遮挡的鲁棒性仍有限，且需要进一步提升域迁移的可靠性。

---

## 516. OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions

**arXiv ID:** 2602.05843 | [PDF](https://arxiv.org/pdf/2602.05843v1)

**作者:** Fangzhi Xu `[一作]` (Xi'an Jiaotong University), Qika Lin `[通讯]` (National University of Singapore)

**通讯引用:** 1240 | [OpenAlex ID](https://openalex.org/A5086407377)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 OdysseyArena 基准，设计四个轻量级交互环境（Turn On Lights、AI Trading、Energy Dispatch、Repo System），生成 120 个任务以及 1,000+ 步的极长步数压力测试，用于评估 LLM 在长周期主动探索和归纳学习方面的能力。

**💡 创新点**

核心创新在于把传统扣式评测转向长期主动探索与归纳推理，将四种结构原语（离散符号规则、连续随机动力、周期性时序模式、图结构依赖）映射到可执行环境，并引入极长步数的压力测试以揭示模型在归纳学习上的瓶颈。

**🔧 技术方法**

利用大语言模型（Gemini、GPT-5、DeepSeek、Llama、Qwen、GLM 等）作为代理，构建交互式环境循环；使用自定义的 Prompt 模板、行动/观测接口以及 success rate、profit rate、Pass@4 等指标进行量化评估。

**📊 数据集**

数据集主要来自内部任务生成器：120 个基准任务（每类 30 任务）以及 10 个 1,000+ 步的压力测试任务；环境参数通过随机采样并经过可解性校验保证每个任务都有可行解。

**📈 对比分析**

通过在同一任务集合上跑 15+ LLM，比较其成功率和收益；最高模型 Gemini 3 Pro Preview 的成功率约 44%，远低于人类（81%+），表明即便是最先进的 LLM 在归纳推理和长期规划上仍存在显著缺陷。

**⚠️ 局限性**

主要限制是模型缺乏真正的归纳学习能力，长时间交互后性能停滞；频繁出现“动作循环”且无法从失败中提取隐藏规则；仅靠模型规模无法突破归纳瓶颈，需进一步探索新架构和学习机制。

---

## 517. Reinforcement World Model Learning for LLM-based Agents

**arXiv ID:** 2602.05842 | [PDF](https://arxiv.org/pdf/2602.05842v1)

**作者:** Xiao Yu `[一作]` (Columbia University), Zhou Yu `[通讯]` (Equal Advising)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自监督方法，利用LLM作为动作条件的世界模型，基于环境交互数据训练模拟下一个状态与真实状态的一致性

**💡 创新点**

创新点在于用预训练嵌入空间中的余弦相似度作为奖励，避免token级别预测导致模型崩溃，并通过易样本子采样与GRPO提升学习效率；同时在RL前先训练世界模型，减少灾难性遗忘

**🔧 技术方法**

采用RL算法GRPO、embedding‑based奖励、预训练的文本嵌入模型、LLM自身的推理和生成能力，并结合子采样策略

**📊 数据集**

在两个长期任务基准上进行实验：ALFWorld与τ^2 Bench

**📈 对比分析**

与任务成功奖励RL、SFT、专家/强LLM训练等方法对比；在ALFWorld和τ^2 Bench上分别提升约19.6和7.9分；与任务成功RL联合时，提升6.9和5.7分，且匹配专家数据训练水平；并显著降低无效/低效动作比例

**⚠️ 局限性**

对弱基模型的迁移效果有限，仍受模型容量影响；奖励函数仍可能被较弱模型利用或被“hack”；需要环境交互数据；在更大规模或更复杂环境中的可扩展性尚未验证

---

## 518. Bandit Social Learning with Exploration Episodes

**arXiv ID:** 2602.05835 | [PDF](https://arxiv.org/pdf/2602.05835v1)

**作者:** Kiarash Banihashem `[一作]` (University of Maryland), Aleksandrs Slivkins `[通讯]` (Microsoft Research)

**通讯引用:** 4180 | [OpenAlex ID](https://openalex.org/A5058550942)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在多臂老虎机框架下，多个自利代理人通过一次或多次决策组成的“周期”式社会学习模型，并分析了其探索与利用的平衡及学习失败现象。

**💡 创新点**

证明了即使每个代理人在周期内进行探索，仍普遍出现学习失败，导致贝叶斯后悔随时间线性增长；这一结论对不同聚合函数（如 sum、max、min 或对称函数）均成立，强调了外部探索激励的必要性。

**🔧 技术方法**

使用贝叶斯推断、Beta 先验、期望效用最优单周期策略，结合马尔可夫、Azuma‑Hoeffding、对偶偏置等概率工具，构造“强失败”事件并推导线性后悔下界。

**📊 数据集**

不使用外部真实数据，而是在理论上构造两臂伯努利分布（加上始终为 0 的跳过臂）的随机奖励模型。

**📈 对比分析**

与传统贪婪或单步期望策略对比，实验（理论证明）表明所有自利策略在本模型下均产生线性后悔，无法达到期望子线性后悔的目标，从而验证了平台外部激励的必要性。

**⚠️ 局限性**

结论仅适用于两臂、伯努利奖励、对称或特定聚合函数的理想化情形；对多臂、非对称函数或非伯努利分布的推广仍待研究。

---

## 519. DuoDrama: Supporting Screenplay Refinement Through LLM-Assisted Human Reflection

**arXiv ID:** 2602.05854 | [PDF](https://arxiv.org/pdf/2602.05854v1)

**作者:** Yuying Tang `[一作]` (Hong Kong University of Science and Technology), Huamin Qu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10800 | [OpenAlex ID](https://openalex.org/A5091466289)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了 DuoDrama 系统，利用 LLM 通过体验‑评估两阶段角色切换（ExReflect 工作流）为剧本改写阶段的编剧提供经验驱动的反馈，以支持他们的反思与改进。

**💡 创新点**

创新点在于：1）将演绎理论（斯坦尼斯拉夫斯基的沉浸式表演与布莱希特的批判性距离）与 AI 交互结合，构建了体验‑评估双角色切换工作流；2）在多角色剧本中采用多智能体架构，让每个角色独立生成“内心体验”，再以此为背景生成针对性的反馈；3）设计了经验基础的问答式反馈和多维度对齐评价体系，显著提升反馈质量与上下文一致性。

**🔧 技术方法**

技术实现主要包括：使用 GPT‑4／GPT‑4o 通过 LCEL（LangChain Expression Language）实现链式推理；使用 FAISS 进行长短期记忆检索；采用 FastAPI+Next.js/React 构建前后端；实现多智能体角色切换与即时/后续反馈的交互。

**📊 数据集**

数据集为十四名专业编剧提供的原创剧本（含情节大纲、人物简介和完整剧本），每人提供两段情节用于实验；无公开大规模剧本数据集，所有文本均来自参与者自创。

**📈 对比分析**

评估方法：两阶段用户研究——第一阶段体验性评估（SUS、7点 Likert、访谈），第二阶段离线对比评估（四种反馈条件：Eval‑PE、Exp‑PE、Eval‑NoPE、Reviewer）。统计采用 Wilcoxon 符号秩检验。结果显示 DuoDrama 在对齐度、质量、感知效果、反思深度与丰富度等指标上显著优于其他三种条件，尤其在情感洞察、情节节奏和主题一致性评估上取得最高分。

**⚠️ 局限性**

局限性包括：仅在剧本改写领域验证，缺乏跨域通用性；实验时长短，仅两次交互，无法观察长期使用效果；仅基于文本经验，未探索多模态体验；未对大规模自动化评估做系统化验证。

---

## 520. Self-Supervised Learning with a Multi-Task Latent Space Objective

**arXiv ID:** 2602.05845 | [PDF](https://arxiv.org/pdf/2602.05845v1)

**作者:** Pierre-François De Plaen `[一作]` (KU Leuven), Marc Proesmans `[通讯]` (KU Leuven)

**通讯引用:** 2505 | [OpenAlex ID](https://openalex.org/A5004700622)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在预测器（Predictor）驱动的 Siamese 自监督学习框架中，提出为每种视图类型（全局 crop、局部 crop、cutout）分配独立的预测器，并将多crop与cutout整合为一个统一的多任务训练框架；

**💡 创新点**

创新点在于：①识别到共享预测器导致多crop训练不稳定，并通过为每种视图分配专用预测器实现稳定；②引入 cutout 作为新的空间监督任务，进一步提升表示质量；③将上述改进统一为一种多任务 Siamese SSL 体系，兼容 CNN 与 Transformer 结构。

**🔧 技术方法**

使用技术包括：Siamese 编码器（共享 backbone + projection head）、针对不同视图的独立 predictor、EMA 更新的目标网络、对齐损失、随机 crop、局部 crop、随机 cutout 以及多任务加权损失。

**📊 数据集**

主要实验数据集为 ImageNet‑1k（预训练与线性/ k‑NN 评估）和 COCO（Mask R‑CNN dense 任务）。

**📈 对比分析**

与 BYOL、SimSiam、MoCo v3 等 predictor‑based 方法以及 SwAV、DINO、ReLIC 等对比，模型在 200‑epoch 预训练下显著提升（ResNet‑50 linear 76.7%，ViT‑S linear 74.0%，ViT‑B linear 77.7%），在 COCO dense 任务中 AP 也高于监督预训练与其他 SSL 方法。

**⚠️ 局限性**

限制包括：仅在图像空间验证，未探究视频或点云等模态；需较多计算资源（如 800‑epoch 训练）才能获得最优效果；对不同网络结构的适应性和长期稳定性仍待进一步验证。

---

## 521. Large Data Acquisition and Analytics at Synchrotron Radiation Facilities

**arXiv ID:** 2602.05837 | [PDF](https://arxiv.org/pdf/2602.05837v1)

**作者:** Aashish Panta `[一作]` (University of Utah), Valerio Pascucci `[通讯]` (University of Utah)

**通讯引用:** 11008 | [OpenAlex ID](https://openalex.org/A5009460413)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计、部署并评估了一个基于 Web 的实时数据采集与可视化框架，实现 CHESS 同步辐射设施的高吞吐量数据监测与质量评估。

**💡 创新点**

通过 NSDF EntryPoint 实现安全的外部访问，结合 OpenVisus 多分辨率块存储与自动化元数据抽取，支持数十 TB 数据的近实时可视化与错误检测。

**🔧 技术方法**

使用了 OpenVisus/OpenVisusPy、NSDF EntryPoint、Docker、NGINX/Apache、Plotly、xarray/pandas、Python 处理管道、SPEC 插件、WebGL（Deck.gl）以及 SSH/NoMachine 远程访问技术。

**📊 数据集**

应用于 CHESS ID3A/B/C 三条波束线产生的近 10 M 文件、50–100 TB 原始数据，数据格式包括 NeXus、HDF5、CBF、TIFF 等多种。

**📈 对比分析**

与传统 NoMachine 远程桌面和本地软件对比，Dashboard 在 10 秒刷新、实时错误警报下显著降低人工检查时间；在 100 TB 规模下保持 10–20 秒更新周期，文件完整性检测成功率>99%。

**⚠️ 局限性**

受限于文件系统抓取速度导致刷新不稳定，缺乏完整的长期归档集成，某些高分辨率图像仍需后处理，未实现跨设施通用化。

---

## 522. Sparse Video Generation Propels Real-World Beyond-the-View Vision-Language Navigation

**arXiv ID:** 2602.05827 | [PDF](https://arxiv.org/pdf/2602.05827v1)

**作者:** Hai Zhang `[一作]` (University of Hong Kong), Hongyang Li `[通讯]` (University of Hong Kong)

**通讯引用:** 3230 | [OpenAlex ID](https://openalex.org/A5100450555)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于稀疏视频生成的视觉语言导航系统（SparseVideoNav），实现了在真实世界中无监督完成Beyond-the-View导航任务；

**💡 创新点**

核心创新在于将长时域视频生成能力迁移到导航任务，并通过稀疏视频监督与四阶段训练（从T2V到I2V、历史注入、扩散蒸馏、动作学习）实现了长预测视野与高效推理的双重突破；

**🔧 技术方法**

关键技术包括基于Wan 1.3B T2V骨干的稀疏视频生成、跨模态Q‑Former + Video‑Former 历史压缩、PCM式扩散蒸馏、DiT动作头以及DA3重标记等；

**📊 数据集**

使用了自采集的140小时真实世界导航视频数据集（约13,000条轨迹），并在此基础上构建了高质量的语言指令与动作标注；

**📈 对比分析**

与三大基线（UniNavid、StreamVLN、InternVLA‑N1）在六个未见场景下的IFN与BVN任务进行零样本对比，SparseVideoNav在BVN任务上平均成功率提升至25%，比最强基线提升约15%；

**⚠️ 局限性**

主要局限在于数据规模尚不足以覆盖更广泛场景，且尽管推理已大幅加速，但相比部分LLM方法仍略显慢；

---

## 523. NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects

**arXiv ID:** 2602.05822 | [PDF](https://arxiv.org/pdf/2602.05822v1)

**作者:** Musawar Ali `[一作]` (University of Bologna), Luigi Di Stefano `[通讯]` (University of Bologna)

**通讯引用:** 9255 | [OpenAlex ID](https://openalex.org/A5025618347)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了NVS‑HO基准，专为手持物体的RGB视角合成设计。

**💡 创新点**

首创仅使用RGB并通过手持与固定序列获取真实姿态的评测基准。

**🔧 技术方法**

使用COLMAP、VGGT进行姿态估计，NeRF（Nerfacto）和高斯射线（Splatfacto）进行视角合成。

**📊 数据集**

数据集包含67个常见物体，分别记录手持序列和棋盘序列共100–200帧。

**📈 对比分析**

基准通过对齐姿态后在遮罩化的视图上计算PSNR/SSIM/LPIPS，COLMAP+Splatfacto在所有指标上表现最好但仍低于理想值。

**⚠️ 局限性**

受手势遮挡和姿态误差影响，现有方法在手持场景下效果有限，需要更鲁棒的姿态与渲染技术。

---

## 524. TKG-Thinker: Towards Dynamic Reasoning over Temporal Knowledge Graphs via Agentic Reinforcement Learning

**arXiv ID:** 2602.05818 | [PDF](https://arxiv.org/pdf/2602.05818v1)

**作者:** Zihao Jiang `[一作]` (Wuhan University), Min Peng `[通讯]` (Wuhan University)

**通讯引用:** 6180 | [OpenAlex ID](https://openalex.org/A5102996335)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了一种基于LLM的自主规划与自适应检索的时序知识图问答代理，并通过监督微调和强化学习实现对TKG的多步推理。

**💡 创新点**

将TKGQA建模为与时序检索工具交互的RL决策过程，首次提出多维奖励（格式、检索、结果）并实现“思考–行动–观察”循环。

**🔧 技术方法**

采用Chain‑of‑Thought监督微调、PPO/GRPO强化学习、ReAct式工具调用以及时序检索工具。

**📊 数据集**

在MULTITQ、CronQuestions两个基准集上进行实验，并在附录中使用TimelineKGQA进行跨域验证。

**📈 对比分析**

与PLM、Embedding和LLM‑RAG等多类基线对比，Hits@1整体提升7–8%，在多步复杂问答上提升近30%，表现明显优于现有方法。

**⚠️ 局限性**

奖励仅为二值，缺乏对中间推理过程的细粒度评估；数据集多步推理深度有限，难以训练长程规划与推理能力。

---

## 525. Interpreting Manifolds and Graph Neural Embeddings from Internet of Things Traffic Flows

**arXiv ID:** 2602.05817 | [PDF](https://arxiv.org/pdf/2602.05817v1)

**作者:** Enrique Feito-Casares `[一作]` (Universidad Rey Juan Carlos), José-Luis Rojo-Álvarez `[通讯]` (Universidad Rey Juan Carlos)

**通讯引用:** 5262 | [OpenAlex ID](https://openalex.org/A5051286582)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种可解释的 GNN + P-UMAP 管道，用于 IoT 网络流量的低维可视化与入侵检测。

**💡 创新点**

通过在训练时将嵌入与 UMAP 投影耦合，确保低维可视化与判别任务保持一致，并结合 SHAP 进行特征归因，弥合了高维嵌入的可解释性缺口。

**🔧 技术方法**

采用 GIN 消息传递网络、Parametric-UMAP 降维、SHAP 特征归因、异步损失/自编码器等技术。

**📊 数据集**

主要使用 CICIoT2023 基准数据集，包含 105 种 IoT 设备的四周流量和 33 种攻击。

**📈 对比分析**

与仅使用 P-UMAP 降维的基线相比，监督式 GNN-CLS 在边缘嵌入上 DBI 降低、Silhouette 提升，二分类 F1=0.830；在多分类上宏 F1≈0.56；实验还展示了概念漂移对性能的影响。

**⚠️ 局限性**

主要局限在于多分类性能随时间漂移显著下降，且模型对相似攻击（如 Mirai 与 DoS）难以区分；模型在大规模实时环境中的可扩展性与对不同网络拓扑的泛化仍待进一步验证。

---

## 526. Where Does Warm-Up Come From? Adaptive Scheduling for Norm-Constrained Optimizers

**arXiv ID:** 2602.05813 | [PDF](https://arxiv.org/pdf/2602.05813v1)

**作者:** Artem Riabinin `[一作]` (Basic Research of Artificial Intelligence Laboratory), Aleksandr Beznosikov `[通讯]` (Basic Research of Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于线性最小化算子（LMO）的学习率自适应调度器，能够自动完成 warm‑up 与衰减阶段；

**💡 创新点**

通过引入子最优度相关的光滑性假设（Kρ>0），从理论上解释并生成 warm‑up；

**🔧 技术方法**

采用 LMOs（normSGD、signSGD/Lion、Muon 等）结合自适应学习率公式，利用理论分析和实验验证；

**📊 数据集**

在 FineWeb 数据集上对 LLaMA 124M/210M 预训练任务进行实验；

**📈 对比分析**

与手工调参的 warm‑up + cosine 计划比较，结果显示自适应 scheduler 在所有优化器、模型规模、批量大小下均能匹配或超越手动设置，且无需网格搜索；

**⚠️ 局限性**

限制在于需要预估目标损失 f⋆ 以及假设的光滑性参数 K0、K1、Kρ 可能在不同任务/模型上变化，且理论假设仍基于星形凸性，未涵盖所有实际非凸情形。

---

## 527. Principled Confidence Estimation for Deep Computed Tomography

**arXiv ID:** 2602.05812 | [PDF](https://arxiv.org/pdf/2602.05812v1)

**作者:** Matteo Gätzner `[一作]` (ETH Zürich), Johannes Kirschner `[通讯]` (Swiss Data Science Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于顺序似然混合的计算机断层扫描（CT）重建置信区间框架，能够为深度学习重建提供理论覆盖保证并检测重建幻觉。

**💡 创新点**

创新点在于将顺序似然混合方法与X射线CT的非线性物理模型和泊松噪声结合，利用深度网络生成更紧致的置信域，并实现可解释的像素级不确定性可视化。

**🔧 技术方法**

使用的技术包括顺序似然混合、马尔可夫性质、U‑Net、U‑Net集成、扩散模型、泊松前向模型、Bootstrap抽样以及基于扩散的置信边界采样。

**📊 数据集**

实验数据集涵盖医学LIDC‑IDRI（肺部CT）、工业Lamino（半导体芯片扫描）和材料Composite（纤维增强复合材料扫描）三类CT图像。

**📈 对比分析**

与传统FBP、MLE等基线相比，深度学习方法在相同剂量下产生更小的β_t、覆盖率达到5%理论上限，且在低剂量条件下逼近理论错误率；幻觉检测结果显示置信域能有效过滤不一致的生成图像。

**⚠️ 局限性**

局限性包括：仍需在真实临床扫描上验证，依赖于物理模型假设和实现性，可能对高维参数敏感，且实验仅针对平行光束几何，未覆盖其他扫描协议。

---

## 528. STProtein: predicting spatial protein expression from multi-omics data

**arXiv ID:** 2602.05811 | [PDF](https://arxiv.org/pdf/2602.05811v1)

**作者:** Zhaorui Jiang `[一作]` (Peking University), Wei Pang `[通讯]` (Heriot-Watt University)

**通讯引用:** 3675 | [OpenAlex ID](https://openalex.org/A5081644845)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了STProtein框架，用以预测空间蛋白表达并基于预测结果进行空间蛋白聚类；

**💡 创新点**

创新点在于将图注意力网络与多任务学习相结合，通过KNN构建特征图在仅有空间转录组的条件下高精度预测空间蛋白表达，解决空间蛋白数据稀缺问题；

**🔧 技术方法**

采用GATv2图注意力网络、两层编码器-解码器结构、多任务损失（RNA+蛋白重构），并以KNN构造的特征图为输入；

**📊 数据集**

使用小鼠脾脏SPOTS、小鼠胸腺Stereo‑CITE‑seq以及人类淋巴结10x Visium三组空间多组学数据集；

**📈 对比分析**

与totalVI、scArches、Dengkw和cTp_net四种基线方法在RMSE与聚类指标（NMI、AMI、FMI、ARI、V‑Measure、F1、Jaccard）上进行对比，STProtein在所有数据集上均表现出更低的RMSE和更高的聚类质量；

**⚠️ 局限性**

局限性在于KNN特征图更侧重全局特征，可能忽略局部空间邻接关系，难以捕捉细微组织结构，未来可考虑整合H&E图像信息以提升预测精度。

---

## 529. A Hybrid Data-Driven Algorithm for Real-Time Friction Force Estimation in Hydraulic Cylinders

**arXiv ID:** 2602.05967 | [PDF](https://arxiv.org/pdf/2602.05967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 530. Bifrost: Steering Strategic Trajectories to Bridge Contextual Gaps for Self-Improving Agents

**arXiv ID:** 2602.05810 | [PDF](https://arxiv.org/pdf/2602.05810v1)

**作者:** Quan M. Tran `[一作]` (Sydney AI Centre), Tongliang Liu `[通讯]` (Sydney AI Centre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练免费方法 Bifrost，利用任务上下文差异精确地在隐空间中改写过去的成功轨迹，以适应新的任务上下文。

**💡 创新点**

创新点在于发现并利用“上下文‑轨迹相关性”，即上下文偏移与轨迹偏移在隐空间中几乎同向，从而通过对隐藏状态的向量投影实现无监督的轨迹转向，而非传统的粗暴裁剪或昂贵的微调。

**🔧 技术方法**

核心技术包括基于线性表示假设的隐藏状态与输出向量建模、上下文方向向量 Δ 的计算（q̂ 的隐藏状态减去历时轨迹隐藏状态均值）、以及对指定层的隐藏状态进行 α 乘以 Δ 的“Steering”操作；同时对多种上下文方向提取方法（平均、PCA、稀疏自编码）做对比。

**📊 数据集**

实验数据集涵盖跨域任务：数学推理（AQUA→GSM8K）、多领域问答（ARC‑Easy→GPQA‑Diamond）、代码生成（HumanEval→LiveCodeBench），并在 Llama‑3.1/3.2 系列模型上验证。

**📈 对比分析**

与 CoT、ICL、Reflexion、BoT、DoT、RISE、Paprika 等基线相比，Bifrost 在三类任务的成功率或通过率上平均提升 2–16 %（如 GSM8K 90.22 % vs 86.13 % RISE、GPQA‑Diamond 44.44 % vs 33.54 % RISE），且不需要任何额外训练，完全训练‑free。

**⚠️ 局限性**

主要局限在于理论基于线性表示假设，若 LLM 的内部表示高度非线性或任务差异极大，所学的上下文方向可能失效；此外方法仅针对单一目标上下文，尚未处理多目标或持续适应场景。

---

## 531. A Guide to Large Language Models in Modeling and Simulation: From Core Techniques to Critical Challenges

**arXiv ID:** 2602.05883 | [PDF](https://arxiv.org/pdf/2602.05883v1)

**作者:** Philippe J. Giabbanelli `[一作]` (Old Dominion University), Philippe J. Giabbanelli `[通讯]` (Old Dominion University)

**通讯引用:** 2811 | [OpenAlex ID](https://openalex.org/A5023496425)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文以大语言模型（LLM）在建模与仿真（M&S）工作流中的应用为主题，提供了系统化的实用指南，涵盖提示工程、超参数调节、知识增强（RAG、LoRA）以及非确定性源的识别与评估。

**💡 创新点**

创新点在于：① 将LLM常见误区与最佳实践映射到M&S具体情境；② 引入实验设计（DoE）与统计检验方法评估LLM非确定性对仿真结果的影响；③ 提出将LLM定位为“翻译器”而非直接实现者的思路，强调与专业工具协同工作；④ 对温度、top‑p、top‑k等解码超参数在不同模型与任务中的最优取值进行系统性实验与对比。

**🔧 技术方法**

核心技术包括：提示工程（task decomposition、validation prompt、格式化、prompt compression）、解码超参数（temperature、top‑k、top‑p、repetition penalty）、知识增强（检索增强生成 RAG、LoRA 低秩适配）、非确定性来源识别（推理非确定性、分布式偏差）、实验设计（全因子/分数因子 DoE、bootstrap 统计）。

**📊 数据集**

本文并未使用单一公开数据集，而是利用多领域M&S案例（城市交通、社交网络、火灾模拟、工业互联等）以及公开文本、代码生成与仿真结果作为评估素材；在温度与RAG实验中使用了 OpenAI GPT‑3.5‑Turbo、Gemini、Claude‑3‑Opus、DeepSeek‑R1 等模型。

**📈 对比分析**

评估方法：通过多次重复实验（多种温度、不同检索参数）统计平均性能、置信区间、最佳/最差分布以及 TAR@N 等指标；在温度优化实验中绘制二维/三维响应曲面；在 RAG 评估中比较仅 LLM 与 LLM+RAG 的准确率、可解释性与成本。结果表明：温度对不同模型影响差异显著，最佳值与上下文窗口大小相关；RAG 在保持语义一致、减少幻觉方面优于纯 LLM，但需注意检索质量与检索量；LoRA 能在保持模型结构不变的前提下实现领域定制，但需谨慎防止多 LoRA 叠加导致冲突。

**⚠️ 局限性**

局限性：① 研究聚焦于当前主流 LLM 与开源模型，随着新模型发布性能与行为可能改变；② 许多实验使用单一任务或少量案例，缺乏大规模跨任务验证；③ 统计方法假设分布平滑，但 LLM 输出分布往往多峰或重尾，可能影响置信区间的解释；④ 仍缺乏对模型版本控制、持续集成等软件工程最佳实践的深入讨论；⑤ 由于篇幅限制，未能对所有知识增强技术（如适配器、选择式方法）进行系统对比。

---

## 532. Beyond Manual Planning: Seating Allocation for Large Organizations

**arXiv ID:** 2602.05875 | [PDF](https://arxiv.org/pdf/2602.05875v1)

**作者:** Anton Ipsen `[一作]` (AI Research, J.P. Morgan Chase), Manuela Veloso `[通讯]` (AI Research, J.P. Morgan Chase)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种层级化座位分配问题（Hierarchical Seating Allocation Problem，HSAP），并给出一种基于概率道路图（PRM）和快速探索随机树（RRT）的高效成对座位距离估计方法，同时将HSAP拆解为若干子问题（单层座位分配），使用混合整数规划（IP）及多种启发式算法（ICA、ICA++、GSA、LS）解决，最后通过深度优先层级分配（DF‑HSA）和延迟办公位选择实现全局分配。

**💡 创新点**

创新点包括：①采用PRM+RRT实现对复杂楼层平面图中可行行走距离的精确估计，克服欧氏距离与基于网格路径的缺陷；②将层级座位分配问题分解为多层SA子问题，显著降低问题规模；③设计基于迭代聚类（ICA）与局部搜索（LS）的高效启发式框架，并提出延迟办公位分配策略；④通过实验验证该框架在不同规模实例上的可行性与优越性。

**🔧 技术方法**

技术手段包括：概率道路图与快速探索随机树构建全局路径网络； Floyd‑Warshall算法计算最短路径；混合整数规划（IPSA）求解精确解；迭代k‑means聚类（ICA/ICA++）与贪心分配（GSA）及局部搜索（LS）作为启发式求解；深度优先层级分配（DF‑HSA）实现多层次分配；Python 3.8 + CPLEX 12.9 作为求解环境。

**📊 数据集**

数据集为三套人工构造的楼层平面图实例：小型（172个桌子+26个办公室）、中型（275个桌子+27个办公室）、大型（966个桌子+31个办公室），对应的团队需求分别为167/12、245/23、843/22。

**📈 对比分析**

与基准方法（IPSA、纯贪心、ICA/ICA++）比较，评估指标为中心座位距离、办公位距离和执行时间。实验表明：①IPA在小规模时最优，但在中大规模因变量爆炸难以完成；②ICA+LS在所有规模下获得与IPA相近的距离但速度更快；③延迟办公位选择进一步降低高层团队与办公位之间的距离；总体来看启发式方法在保持可接受质量的前提下显著提升可扩展性。

**⚠️ 局限性**

局限性包括：①PRM+RRT构建需手动调参，易受节点数和阈值影响；②局部搜索对大规模实例收敛慢；③层级分解为子问题虽降低规模，但未实现全局最优，可能导致层间冲突；④实验仅基于合成楼层平面图，缺乏真实大企业案例验证；⑤方法对组织层级结构的假设较强，复杂层级或多根树情形尚未充分探讨。

---

## 533. Competitive Analysis of Online Facility Assignment Algorithms on Discrete Grid Graphs: Performance Bounds and Remediation Strategies

**arXiv ID:** 2602.05953 | [PDF](https://arxiv.org/pdf/2602.05953v1)

**作者:** Lamya Alif `[一作]` (American International University Bangladesh), Md Manzurul Hasan `[通讯]` (American International University Bangladesh)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

分析在线设施分配（OFA）在离散 L1 网格上的局限，构造了区域崩塌、边界振荡和批量过度集中等攻击实例，并提出轻量级的缓解方案。

**💡 创新点**

首创给出了 L1 网格下容量约束导致的离散几何失效模式的精确构造，并提出基于预留容量和稀缺性惩罚的批量化改进，实验验证其对尾部风险的显著降低。

**🔧 技术方法**

使用竞争分析、最小费用流求解批量内最优分配、离散 Voronoi 加权、随机/确定性贪心策略及实验模拟等技术。

**📊 数据集**

采用随机均匀、聚类突发和手工构造的对抗性模板请求序列，在 r×c 网格上进行评估。

**📈 对比分析**

通过与标准贪心、随机贪心和原始批量最小费用流做成本比对，改进方案在最坏情况下降低比例显著，平均场景几乎不增加成本。

**⚠️ 局限性**

仅给出了对抗性下的下界与经验缓解，未提供全局竞争比保证；批量延迟参数和预留阈值需经验调节。

---

## 534. Breaking Symmetry Bottlenecks in GNN Readouts

**arXiv ID:** 2602.05950 | [PDF](https://arxiv.org/pdf/2602.05950v1)

**作者:** Mouad Talhi `[一作]` (Imperial College London), Anthea Monod `[通讯]` (Imperial College London)

**通讯引用:** 349 | [OpenAlex ID](https://openalex.org/A5044345590)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了 GNN 读取层（readout）存在一个与消息传递无关的表达瓶颈，并提出了一种基于表示论的投影读out，能够保留节点嵌入中的对称性信息；通过实验验证了该读out 在分离同形但非同构图、对称性强的基准以及真实数据集上的性能提升。

**💡 创新点**

创新点在于：①使用有限群表示论揭示任何线性置换不变读out必定通过 Reynolds 投影（平均）实现，导致非平凡对称分量被完全丢失；②提出图自适应的投影读out，先通过等变投影将节点嵌入分解为对称性通道，再用非线性不变统计量聚合，从而逃脱平均导致的丢失；③在训练无关的分离实验、对称性基准与下游任务上展示了该方法在不同场景下的优越性。

**🔧 技术方法**

核心技术包括：有限群表示论（不可约分解、Reynolds 算子）、等变投影矩阵构造（基于图自同构轨道）、非线性不变汇总（平方和、能量、均值等统计），以及在实验中对 GNN 编码器（GIN、GraphSAGE、PNA 等）的通用封装。

**📊 数据集**

实验数据集涵盖：①同形非同构图（2C_k vs. C_{2k}、CFI、螺旋形带、Peterson 互换）、②对称性基准 SRG‑16、BREC RPC‑lite、③真实图数据集 PROTEINS、ENZYMES、ZINC、MolHIV，并使用随机节点重标记来保证不变性评估。

**📈 对比分析**

与传统的求和/均值池化或其他不变读out进行对比。训练无关分离实验中，传统池化的相似度≈1.0，新的投影读out在 33/36 对例子中均能分离；在 SRG‑16、BREC RPC‑lite 中，传统池化的准确率≈0.45/0.5，投影读out 达到 1.00/0.90；在下游任务中，投影读out 在 PROTEINS、ENZYMES、MolHIV 上的 MAE/R² 或 ROC‑AUC 明显优于单纯求和，证明其在全局结构感知上的优势。

**⚠️ 局限性**

主要局限包括：①需要对每张图计算自同构群及其轨道，计算量随图大小和对称性增长；②当两图拥有相同对称性结构时，投影读out 仍无法区分；③在局部特征驱动任务（如 ZINC）中，单独使用投影读out 并不一定优于传统池化；④目前对大规模图的可扩展性和近似方法仍待进一步研究。

---

## 535. Self-Improving Multilingual Long Reasoning via Translation-Reasoning Integrated Training

**arXiv ID:** 2602.05940 | [PDF](https://arxiv.org/pdf/2602.05940v1)

**作者:** Junxiao Liu `[一作]` (Nanjing University), Shujian Huang `[通讯]` (Nanjing University)

**通讯引用:** 3696 | [OpenAlex ID](https://openalex.org/A5102865824)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 TRIT 框架，通过翻译训练与多语言推理的强化学习闭环，提升模型在不同语言下的推理准确率与语言一致性。

**💡 创新点**

创新点在于：① 将翻译质量与推理准确度通过奖励结构耦合，使推理成功率成为翻译质量的代理；② 采用跨语言推理过滤器筛选可用训练样本，减少翻译误差反馈噪声；③ 无需外部多语言标注数据即可实现自我提升，形成翻译↔推理的双向正反馈。

**🔧 技术方法**

使用技术包括：Group Relative Policy Optimization（GRPO）强化学习、四元奖励（准确度、语言一致、无重复、格式）、跨语言推理过滤、基于 langdetect 的语言检测、COMET 评估翻译质量、MEXA 评估跨语言问题对齐。

**📊 数据集**

数据集：训练采用 DAPO‑MATH‑17K；评估使用 MMATH（AIME24/25、CNMO、MATH500）以及 FLORES‑200 进行翻译质量评估。

**📈 对比分析**

与 Prompt Control、SFT、Naive RL、SLC‑RL、M‑Thinker、External‑Translation 等基线对比；在三种后端模型（DeepSeek‑Distill‑Qwen‑1.5B、Qwen3‑1.7B、Qwen3‑4B）和五种目标语言上，TRIT平均提升 7pp，单模型最高提升 10pp；语言一致率接近 100%；在翻译任务上，MATH500 赢率提升至 2.2–3.3:1，FLORES‑200 COMET 分数提升最多 8.4 分。

**⚠️ 局限性**

局限性：仅在五种目标语言（FR, PT, JA, KO, TH）验证；未评估更大规模模型；对极低资源语言的泛化效果仍需进一步验证；实现依赖对翻译与推理的统一奖励设计，若奖励不当可能导致不稳定。

---

## 536. Polyglots or Multitudes? Multilingual LLM Answers to Value-laden Multiple-Choice Questions

**arXiv ID:** 2602.05932 | [PDF](https://arxiv.org/pdf/2602.05932v1)

**作者:** Léo Labat `[一作]` (Sorbonne Université), François Yvon `[通讯]` (Sorbonne Université)

**通讯引用:** 3876 | [OpenAlex ID](https://openalex.org/A5030615769)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多语言LLM在价值导向多选题的答案一致性，构建并使用MEVS语料进行系统实验。

**💡 创新点**

首次提供人类翻译、重新对齐的多语言价值调查语料，并深入分析语言特异性与一致性关系。

**🔧 技术方法**

使用多语言大模型（如Llama‑3.1、Qwen2.5等）配合答案顺序、符号类型、尾字符等prompt变体，采用MCP提取答案并计算PPA、Shannon熵、互信息等一致性指标。

**📊 数据集**

使用由142题构成的Multilingual European Value Survey（MEVS），覆盖8种欧洲语言。

**📈 对比分析**

通过PPA、熵、NMI等指标比较模型在24题的回答一致性，发现大型指令微调模型表现最佳，但一致率仅在40–60%之间，且受问题和语言影响显著。

**⚠️ 局限性**

仅评估24题且未涵盖同义句或开放式生成，且仅使用MCP提取答案，未对跨领域或多轮一致性进行检验。

---

## 537. Quantum Reinforcement Learning with Transformers for the Capacitated Vehicle Routing Problem

**arXiv ID:** 2602.05920 | [PDF](https://arxiv.org/pdf/2602.05920v1)

**作者:** Eva Andrés `[一作]` `[通讯]` (University of Granada), Eva Andrés (University of Granada)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了基于量子Transformer的优势演员-评论家（A2C）模型，用于求解多车辆容量限制的车辆路径规划（CVRP）问题，并对经典、混合与全量子三种模型进行了对比实验。

**💡 创新点**

创新点在于首次将自注意力与交叉注意力的量子Transformer模块嵌入强化学习框架，实现动态多车辆、容量约束CVRP的端到端学习；并设计了混合量子-经典架构，在保持可扩展性的同时提升模型表达能力。

**🔧 技术方法**

技术手段包括：优势演员-评论家算法、Pointer Network架构、Transformer自注意力与交叉注意力、幅度编码（Amplitude Embedding）、变分量子电路（VQC）实现量子层、混合量子/经典前后处理，以及使用AdamW优化器与多头量子解码器。

**📊 数据集**

数据集使用随机生成的20个客户、4辆车的CVRP实例，客户位置与需求在二维平面内随机分布，重复10次实验以评估鲁棒性。

**📈 对比分析**

对比方法：在同一自定义环境下训练1000/500/500 episode，10次随机种子；评估指标为总行驶距离、路线紧凑度与交叉次数。实验结果显示混合模型在距离与交叉率上均优于经典与全量子模型；全量子模型在路线紧凑度上表现最佳。

**⚠️ 局限性**

局限性：量子模型在模拟器上训练耗时显著，难以扩展到更大规模实例；受限于当前NISQ硬件与变分电路的梯度消失问题，实际部署仍面临技术挑战。

---

## 538. Layer-wise LoRA fine-tuning: a similarity metric approach

**arXiv ID:** 2602.05988 | [PDF](https://arxiv.org/pdf/2602.05988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 539. CLIP-Map: Structured Matrix Mapping for Parameter-Efficient CLIP Compression

**arXiv ID:** 2602.05909 | [PDF](https://arxiv.org/pdf/2602.05909v1)

**作者:** Kangjie Zhang `[一作]` (East China Normal University), Shaohui Lin `[通讯]` (East China Normal University)

**通讯引用:** 3498 | [OpenAlex ID](https://openalex.org/A5043643513)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于可学习映射矩阵的 CLIP‑Map 压缩框架，能够在宽度和深度上同时压缩 CLIP 模型，同时保留原始模型的大量信息，并在重训练阶段使用知识蒸馏进一步提升性能。

**💡 创新点**

创新点主要包括：① 使用可学习的映射矩阵并通过 Kronecker 因子化实现宽度压缩，显著降低参数量；② 设计对角继承初始化方案，减少分布漂移并加速收敛；③ 构建统一的映射‑重训练两阶段流程，避免硬剪枝导致的信息损失。

**🔧 技术方法**

核心技术包括可学习映射矩阵、Kronecker 乘积因子化、对角继承初始化、知识蒸馏（软标签+InfoNCE硬标签）、Transformer‑based CLIP 结构以及两阶段训练策略。

**📊 数据集**

使用的主要数据集为 YFCC‑15M 进行训练，MSCOCO 与 Flickr30K 进行零检索评估，ImageNet‑1K 进行零分类评估，并在 21 个下游分类任务上进行泛化测试。

**📈 对比分析**

在与 TinyCLIP、OpenCLIP、Meta‑CLIP、MoPE‑CLIP、MobileCLIP 等压缩方法的对比中，CLIP‑Map 在 1%、10% 及 50% 的压缩率下均实现了更高的零检索召回率和零分类准确率，且训练 epoch 更少、模型尺寸更小。

**⚠️ 局限性**

局限性包括：仍需两阶段训练，映射矩阵的优化在计算上略有开销；目前实验主要针对 Transformer‑based CLIP，跨模态和更大模型的通用性尚待进一步验证；在极端压缩率下性能仍有一定下降空间，且对特定任务的泛化需要更多研究。

---

## 540. Regularized Calibration with Successive Rounding for Post-Training Quantization

**arXiv ID:** 2602.05902 | [PDF](https://arxiv.org/pdf/2602.05902v1)

**作者:** Seohyeon Cha `[一作]` (University of Texas at Austin), Haris Vikalo `[通讯]` (University of Texas at Austin)

**通讯引用:** 5537 | [OpenAlex ID](https://openalex.org/A5067602750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型提出正则化不对称校准与顺序量化的后训练量化框架，提升量化精度。

**💡 创新点**

通过α插值正则化将对称与不对称校准统一，并设计贪心及束搜索的顺序量化算法，实现精度与计算的可调平衡。

**🔧 技术方法**

使用Hessian加权二次近似、α加权校准、三角离散最小二乘、贪心/束搜索量化、正则化等技术。

**📊 数据集**

在C4和WikiText2上做校准，评测于WikiText2、C4以及六个commonsense推理任务。

**📈 对比分析**

与GPTQ、GTAQ、LDLQ、GuidedQuant等无学习PTQ基线比较，SNRQ在3/4/5比特量化下PPL降低5–28%，且量化速度提升约30%。

**⚠️ 局限性**

局部误差仍影响通道级低比特量化，束搜索内存随宽度增长，且α的设置仍需经验手动调节。

---

## 541. Residual Reinforcement Learning for Waste-Container Lifting Using Large-Scale Cranes with Underactuated Tools

**arXiv ID:** 2602.05895 | [PDF](https://arxiv.org/pdf/2602.05895v1)

**作者:** Qi Li `[一作]` (RPTU University of Kaiserslautern-Landau), Karsten Berns `[通讯]` (RPTU University of Kaiserslautern-Landau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

针对城市废物回收场景中液压起重机的容器吊装阶段，提出了一种残差强化学习(RRL)控制框架，将基准正交坐标控制器与学习到的残差策略相结合，实现精准的TCP轨迹跟踪与摆动抑制。

**💡 创新点**

创新点包括：①将正交坐标（admittance）控制与摆动抑制的模型基础与强化学习残差相结合；②只在关键的水平对齐阶段使用残差策略，降低学习复杂度；③采用域随机化与多阶段轨迹分段设计，显著提升泛化与鲁棒性。

**🔧 技术方法**

使用了Isaac Lab仿真平台，基准控制器实现了正交坐标控制、摆动抑制与阻尼最小二乘逆运动学；残差策略采用PPO强化学习，网络结构为三层MLP([128,64,32])；还利用了随机化的仿真参数（质量、阻尼、执行器增益）和TCP轨迹管道约束。

**📊 数据集**

所有实验均在Isaac Lab的仿真环境中进行，数据集由随机生成的容器姿态（位置、朝向）和起重机TCP初始状态组成，包含约300个仿真试验，用于训练与评估。

**📈 对比分析**

通过与仅使用正交坐标控制、仅使用摆动抑制以及仅使用RRL的对照组进行对比，RRL+摆动抑制组合在追踪误差、管道偏差、摆动角度和成功率上分别提升约30-40%，成功率从57%提升至91%，在不同刚度配置下仍保持较高的鲁棒性。

**⚠️ 局限性**

主要限制在于：①全部实验仅在仿真中完成，未验证真实液压系统的可迁移性；②对实时执行性能未做评估；③残差策略仅在水平对齐阶段使用，可能忽略其他关键阶段的误差；④未考虑感知不确定性与环境动态变化。

---

## 542. Clifford Kolmogorov-Arnold Networks

**arXiv ID:** 2602.05977 | [PDF](https://arxiv.org/pdf/2602.05977v1)

**作者:** Matthias Wolff `[一作]` (University of Münster), Xiaoyi Jiang `[通讯]` (University of Münster)

**通讯引用:** 12655 | [OpenAlex ID](https://openalex.org/A5022183918)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于Clifford代数的Kolmogorov‑Arnold网络（ClKAN），扩展了CVKAN的超复数域功能，加入两种RBF基函数、随机化 Sobol 网格生成和多种批归一化策略，并在多维Clifford空间中进行函数逼近实验。

**💡 创新点**

创新点在于：①首次将Clifford代数融入KAN框架，实现高维超复数域的可解释网络；②设计了保留空间结构的Clifford RBF；③使用随机化 Sobol 网格有效降低高维参数量；④提出维度、节点、组件三种批归一化方案以适配任意Clifford维度。

**🔧 技术方法**

技术方法包括：Clifford代数运算与几何乘积、KAN结构、两种RBF基函数、scrambled Sobol 序列生成网格、节点/维度/组件批归一化、10倍学习率提升、5折交叉验证与MSE/MAE/CE评价。

**📊 数据集**

使用的数据集为：①复数域的四个合成函数（square、sin、mult、squaresquare）；②物理启发的 holography 数据集（100k 样本）；③高维Clifford代数（GA(2)、四元数、(1,1) conformal GA、(1,0,1) PGA）对应的同一四个合成函数；（Knot 数据集仅做过度验证，未公布）。

**📈 对比分析**

通过5折交叉验证、测试集评估与改进CVKAN对比；结果显示ClKAN在复数域任务中与CVKAN相当，Sobol 网格在参数显著减少（至 2–6%）的情况下与全格子匹配甚至超越；在高维Clifford空间中，Sobol 网格同样保持低MSE并在某些配置下优于全格子；学习率 0.1 改善了收敛稳定性。

**⚠️ 局限性**

局限性包括：对初始化高度敏感，部分网格尺寸与RBF/批归一化组合导致高方差和不稳定性；Sobol 网格在某些数据集下表现略逊于全格子；未实现可学习的 RBF 形状和更优的残差激活函数；整体稳定性和鲁棒性仍需进一步提升。

---

## 543. Inverse Depth Scaling From Most Layers Being Similar

**arXiv ID:** 2602.05970 | [PDF](https://arxiv.org/pdf/2602.05970v1)

**作者:** Yizhou Liu `[一作]` (Massachusetts Institute of Technology), Jeff Gore `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 11209 | [OpenAlex ID](https://openalex.org/A5003202779)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型（LLM）深度的损失缩放关系，发现其损失近似按深度的倒数缩放，归因于多层几乎相同的子网络通过集成平均降低误差；

**💡 创新点**

首次将深度与宽度的损失贡献分解为单独的幂律项，并通过理论、隐藏状态分析与对照实验验证了“集成平均”是LLM中主导的深度利用机制；

**🔧 技术方法**

使用角度度量隐藏状态变换、PCA、神经网络缩放律拟合、教师-学生残差网络实验、KL散度训练、以及对残差网络的理论误差分析；

**📊 数据集**

主要基于Pythia系列模型在FineWeb数据集上的表现，结合Chinchilla模型的公开损失数据进行多模型拟合；

**📈 对比分析**

与传统的单一参数缩放律对比，提出的分解式模型在200个不同宽度/深度组合上实现了平均0.4%的损失预测误差，α_m≈1、α_ℓ≈1.2，说明深度的逆比缩放在实际模型中成立；

**⚠️ 局限性**

受限于缺乏严格的第一性原理推导，无法排除其他可能导致同样逆比缩放的机制；实验样本受开源模型规模和数据集的限制，未涵盖更大规模或不同架构的模型，且对训练动态与深度相互作用的交叉项分析不足。

---

## 544. Better Source, Better Flow: Learning Condition-Dependent Source Distribution for Flow Matching

**arXiv ID:** 2602.05951 | [PDF](https://arxiv.org/pdf/2602.05951v1)

**作者:** Junwan Kim `[一作]` (New York University), Seungryong Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种条件相关源分布的流匹配方法（CSFM），通过学习与文本提示相关的源分布来提高文本到图像生成的效率和质量。

**💡 创新点**

创新点在于将源分布设为可学习的条件分布，并通过方差正则化和方向对齐来降低内在方差，从而实现更稳定的训练和更直线的流场。

**🔧 技术方法**

采用流匹配框架、条件源生成器、仅对方差的KL正则化、负余弦方向对齐损失，并在RAE（DINOv2）特征空间中训练。

**📊 数据集**

在ImageNet-1K数据集上使用Qwen3‑VL生成的描述性标题进行评估，扩展到BLIP3o预训练数据以进行大规模实验。

**📈 对比分析**

与传统高斯源、C^2OT、CrossFlow等方法对比，CSFM在FID收敛速度提高3.01倍、CLIP得分加速2.48倍，并在多种架构和文本编码器上持续提升性能。

**⚠️ 局限性**

局限性包括：当条件对应的目标分布高度多模态或在目标空间缺乏结构时，源均值难以定义，导致收益有限；目前仍不适用于无条件生成，且对特定表示空间（如RAE）依赖较强。

---

## 545. Location-Aware Dispersion on Anonymous Graphs

**arXiv ID:** 2602.05948 | [PDF](https://arxiv.org/pdf/2602.05948v1)

**作者:** Himani `[一作]` (Dhirubhai Ambani University), Gokarna Sharma `[通讯]` (Kent State University)

**通讯引用:** 1090 | [OpenAlex ID](https://openalex.org/A5002981812)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并研究了在无标记图上进行位置感知色彩分散（Location‑Aware Dispersion）问题，提出了多种确定性算法并给出了时间与空间下界及不可解性结果。

**💡 创新点**

创新点在于：
- 将经典 dispersion 扩展到需满足节点与机器人颜色匹配的场景；
- 证明了该问题在单机器人且不知道图大小时不可解；
- 设计了分阶段 DFS 与组化策略，既能保证在任意图上完成分散，又能在不同初始配置（根型、分散、一般）和参数已知/未知情况下给出几乎紧贴下界的时间与空间复杂度；
- 通过多 DFS 交叉与“吸收”机制实现从一般配置到根型的统一解决方案。

**🔧 技术方法**

采用的技术包括：
- 深度优先搜索（DFS）与基于端口的导航；
- 组化（grouping）与守护机器人（guard）概念；
- 三阶段算法（图探索、连通性收集、内存高效分散）；
- 迭代猜测（doubling strategy）以处理未知图大小；
- 归并/吸收策略（subsumption）在一般初始配置中实现多 DFS 的合并。

**📊 数据集**

本文为理论性工作，没有使用具体数据集，所有结果均为图论上最优或近似最优的数学证明与算法复杂度分析。

**📈 对比分析**

与传统 dispersion 的比较：
- 对于 k≤n 的情况，经典 dispersion 可在 O(k) 轮内完成，内存 O(Δ+log k)；
- 本文在已知 n,k 时，位置感知色彩分散在树、路径、环等特殊图上仍可实现 O(n) 轮、O(log(k+Δ)) 位内存；
- 对于任意图，已知 n,k 时可在 O(n/k·m) 轮、O(n/k·log(k+Δ)) 位内存；若 n 未知，则额外乘以 O(log(n/k))；
- 下界表明位置感知色彩分散至少需要 Ω(min{n²/k, m}) 轮、Ω(n/k·log k) 位内存，显著高于经典 dispersion。

**⚠️ 局限性**

局限性包括：
- 当 k=1 且不知道 n 时问题不可解；
- 在一般初始配置下，若每个源节点最多只有两个机器人，算法需要完整图信息导致内存 O(n·log(k+Δ))；
- 对于颜色数 t≫1 的情况，算法的常数因子和组化阈值（2c–4c）可能导致实现复杂；
- 所有分析均在同步、局部通信模型下，异步或全局通信环境需进一步研究。

---

## 546. $f$-GRPO and Beyond: Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment

**arXiv ID:** 2602.05946 | [PDF](https://arxiv.org/pdf/2602.05946v1)

**作者:** Rajdeep Haldar `[一作]` (Purdue University), Qifan Song `[通讯]` (Purdue University)

**通讯引用:** 995 | [OpenAlex ID](https://openalex.org/A5110373518)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于 f‑divergence 的通用 LLM 对齐方法 f‑GRPO（纯 on‑policy）和 f‑HAL（混合 on/off-policy），并在可验证奖励（RLVR）与偏好对齐（PA）两种场景下实现对齐。

**💡 创新点**

创新点在于：① 把偏好对齐与可验证奖励的对齐统一到一个 divergence 估计框架；② 推导出可以直接使用环境奖励的 on‑policy f‑GRPO；③ 设计混合 f‑HAL，利用两种信息源并能理论证明奖励提升与对齐一致性。

**🔧 技术方法**

技术手段包括：变分 f‑divergence 表示、重要性采样与截断权重、GRPO 风格的 on‑policy 更新、混合损失构造，以及固定点、收敛与奖励改进的理论证明。

**📊 数据集**

实验数据集：
- RLVR（数学推理）使用 OpenRS、LIMR、GSM8K、MATH500、AMC23、AIME24/25；
- PA（安全对齐）使用 rhaldar97/Safety_Accept_Reject 与 OpenAssistant/reward‑model‑deberta‑v3‑large‑v2；
- 评估 benchmark 包括 MMLU‑Pro、IFEval、MuSR、ToxiGen、ASR 等。

**📈 对比分析**

对比方法：GRPO、DPO、KTO、BCO、FDO；结果显示：
- 在数学任务中，f‑GRPO 在大多数分数上优于 GRPO；
- 在安全对齐中，f‑HAL（尤其 λ=0.5）在鲁棒性与效用上均优于纯 on‑policy 或 off‑policy 基线，并有效抑制奖励劫持。

**⚠️ 局限性**

局限性：
- 需要假设奖励与对齐分布的对应关系；
- 截断重要性采样会引入偏差；
- 小模型容量时仍易出现奖励劫持；
- 性能受 f‑divergence 与链接函数选择影响，且在大规模模型上的计算成本仍需进一步优化。

---

## 547. AgenticTagger: Structured Item Representation for Recommendation with LLM Agents

**arXiv ID:** 2602.05945 | [PDF](https://arxiv.org/pdf/2602.05945v1)

**作者:** Zhouhang Xie `[一作]` (University of California San Diego), Randolph Brown `[通讯]` (Google)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LLM多智能体的框架，用于自动生成层级化、低基数、可解释的自然语言词条（descriptor）作为商品特征，支持多种推荐场景。

**💡 创新点**

创新点在于：①将LLM的生成能力与多智能体自我改进循环结合，构建全局可见的词典；②通过“架构者-注解者”两层LLM架构实现高效并行化的词典构建和分配；③生成的词条既是语义ID又是可解释的关键词，兼顾可解释性与可控性。

**🔧 技术方法**

核心技术包括：多智能体LLM框架、层级化词典构建、并行化注解（ParallelAssign）、迭代自我改进（Multi-agent Self-refinement）、基于上下文提示的词条分配、以及对比实验的基准模型（SASRec、BERT4Rec、FDSA、S3-Rec、VQ-Rec、TIGER 等）。

**📊 数据集**

主要使用 Amazon Reviews 公开数据集（Sports、Beauty、CDs 等三大域）以及一个私有新闻推送数据集（1791 用户、15830 条目）。

**📈 对比分析**

与传统仅使用 ID 或嵌入特征的模型、基于协同过滤的词条交叉模型（SentencePiece、ActionPiece）以及自由生成词条进行对比。实验显示，在生成式推荐和排序任务中，所提方法在 Recall@K、NDCG@K 等指标上提升 2%~12%（如 Sports、CDs 领域 NDCG@10 提升 8.9%–11.6%），在排序任务中 Recall@10 提升约 50%。

**⚠️ 局限性**

局限性包括：①词典构建和分配阶段对 LLM 的调用成本高，尤其在极大数据集上；②对单层分配的严格约束可能导致对多标签商品的表示不足；③缺乏对协同过滤信号的直接利用，导致在长尾或小数据域效果有限；④评估主要集中在离线指标，缺少在线实验验证。

---

## 548. "Detective Work We Shouldn't Have to Do": Practitioner Challenges in Regulatory-Aligned Data Quality in Machine Learning Systems

**arXiv ID:** 2602.05944 | [PDF](https://arxiv.org/pdf/2602.05944v1)

**作者:** Yichun Wang `[一作]` (University of Amsterdam), Hazar Harmouch `[通讯]` (University of Amsterdam)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5008534205)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对欧盟内从事机器学习系统的从业者进行访谈，研究其在日常工作中如何解释并落实监管对数据质量的要求；

**💡 创新点**

首次系统性梳理监管（GDPR、AI法）与技术实践之间的翻译鸿沟，并提出监管对齐数据质量的社会技术视角与实践建议；

**🔧 技术方法**

采用定性访谈与情景案例法，结合主题编码与归纳分析；

**📊 数据集**

访谈对象为14名欧盟从业者，涉及金融、零售、健康、交通、科研等行业，未使用公开数据集；

**📈 对比分析**

论文不涉及算法或模型性能评估，而是通过访谈文本定性比较各参与者在合规与工程目标间的权衡与工具缺口；

**⚠️ 局限性**

样本量有限、受访者偏向大企业和男性，且依赖自述，缺乏直接观察，导致结果可能存在主观性与代表性不足。

---

## 549. Task-Adaptive Physical Reservoir Computing via Tunable Molecular Communication Dynamics

**arXiv ID:** 2602.05931 | [PDF](https://arxiv.org/pdf/2602.05931v1)

**作者:** Saad Yousuf `[一作]` (Koç University), Murat Kuscu `[通讯]` (Koç University)

**通讯引用:** 1184 | [OpenAlex ID](https://openalex.org/A5048937970)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在模拟环境中展示了单一分子通信通道可作为可调节的物理递归计算器，通过改变扩散、传输距离、受体结合速率等生物物理参数实现任务自适应。

**💡 创新点**

提出将分子通信通道的生物物理参数视为可调节“控制旋钮”，并通过贝叶斯优化找到针对不同计算任务（时间序列预测、非线性变换、混合任务）的最佳参数集，证明了可实现的任务自适应。

**🔧 技术方法**

采用双模拟框架：确定性均场模型与高保真粒子模拟（Smoldyn），结合贝叶斯优化、时间多路化、线性读取层以及因果滑动平均滤波等技术。

**📊 数据集**

使用 Mackey‑Glass 预测序列、正弦‑方波变换以及 Mackey‑Glass 立方混合任务作为基准数据集。

**📈 对比分析**

通过 NRMSE 评价，在确定性模型下匹配参数集可将 MG 预测误差降至 0.097、正弦‑方波误差 0.237、混合任务误差 0.307；在粒子模拟中加入滤波后误差提升到 0.49、0.39、0.74，说明参数自适应有效但噪声仍影响性能。

**⚠️ 局限性**

局限性包括：需要在物理实现前通过昂贵的模拟验证，参数空间仍大且调节困难；粒子级噪声和高分子密度导致模拟不可行；以及缺乏实验验证，真实环境中的温度、离子强度等因素未被全面考虑。

---

## 550. Compound Deception in Elite Peer Review: A Failure Mode Taxonomy of 100 Fabricated Citations at NeurIPS 2025

**arXiv ID:** 2602.05930 | [PDF](https://arxiv.org/pdf/2602.05930v1)

**作者:** Samar Ansari `[一作]` (University of Chester), Samar Ansari `[通讯]` (University of Chester)

**通讯引用:** 3143 | [OpenAlex ID](https://openalex.org/A5007269494)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析了 NeurIPS 2025 接受论文中 AI 生成的虚假引用现象，揭示 53 篇论文共 100 条幻觉引用，约占 1% 的论文被污染。

**💡 创新点**

创新点在于提出五分类失效模式（总制造、属性偏差、识别劫持、占位符、语义幻觉），并首次发现所有幻觉均为多层复合模式，说明审稿流程无法检测。

**🔧 技术方法**

采用 GPTZero 的幻觉检查工具配合人工专家标注，对每条引用进行主次分类，形成复合失效模式矩阵。

**📊 数据集**

数据集为 GPTZero 标注的 100 条幻觉引用，来源于 53 篇 NeurIPS 2025 论文。

**📈 对比分析**

与传统仅做文字核对的审稿流程对比，发现单一属性检查无法检测 100% 的幻觉，提出多属性（存在性、元数据一致性、识别符验证、语义合理性）自动验证方案，可覆盖约 90% 以上的幻觉。

**⚠️ 局限性**

局限在于仅分析 NeurIPS 2025 数据，且仅检测可检索的幻觉，未评估作者意图、其他学科分布及模型生成策略的演变。

---

## 551. KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs

**arXiv ID:** 2602.05929 | [PDF](https://arxiv.org/pdf/2602.05929v1)

**作者:** Jian Chen `[一作]` (University at Buffalo), Yirui Liu `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 6285 | [OpenAlex ID](https://openalex.org/A5100418866)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 KV-CoRE，一种基于 SVD 的方法对大型语言模型的 KV 缓存进行数据驱动的低秩可压缩性评估，并构建了大规模压缩基准；

**💡 创新点**

创新点在于：①使用增量 SVD 直接对键/值激活进行无梯度、可层级化的低秩近似，保证全局最优；②引入 Normalized Effective Rank (NER) 作为压缩性指标，证明其与压缩后的 perplexity 和 GPT‑score 强相关；③提出 Normalized Delta‑Perplexity (ND‑PPL) 作为跨数据集的压缩鲁棒性衡量；

**🔧 技术方法**

技术包括：增量奇异值分解、无梯度优化、按层级和数据集进行 KV 缓存抽样、基于 Frobenius 范数的低秩最优投影；

**📊 数据集**

使用多种公开 LLM（Qwen3、Mistral‑7B、Gemma‑1.1、Phi‑3‑mini‑128k‑instruct 等）和多域、多语种数据集（Alpaca、MedAlpaca、CodeAlpaca、WizardCoder、FunctionCall、VisR‑Bench 的 15 种语言）进行评估；

**📈 对比分析**

比较方法包括：在不同压缩比下测量 PPL 热力图、GPT‑score、ND‑PPL 与 NER 的相关性。实验显示 NER 与压缩后性能退化高度相关，模型在低 NER 时更耐压缩；

**⚠️ 局限性**

局限性：仅针对自回归 Transformer 的 KV 缓存；未考虑压缩对模型可解释性或对抗鲁棒性的影响；增量 SVD 的计算仍有一定开销，且在极大模型或分布式设置下的可扩展性待验证。

---

## 552. Improved SDP-Based Algorithm for Coloring 3-Colorable Graphs

**arXiv ID:** 2602.05904 | [PDF](https://arxiv.org/pdf/2602.05904v1)

**作者:** Nikhil Bansal `[一作]` (University of Michigan), Euiwoong Lee `[通讯]` (University of Michigan)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5075423314)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对3-可着色图，在多项式时间内给出了一个使用 O(n^0.19539) 颜色的着色算法。

**💡 创新点**

创新点在于：①将 SDP 近似算法从二阶邻域提升到三阶邻域；②设计了新的 5/2-向量着色器，能在三阶邻域上提取更大的独立集；③改进了覆盖组合与稀疏化技术，克服了之前覆盖不够随机的瓶颈。

**🔧 技术方法**

核心技术包括：Sum‑of‑Squares / Lasserre 高阶 SDP 约束、向量着色与独立集提取、Gaussian 随机投影与 Borell 等距测不等式、覆盖（cover）与打包（packing）稀疏化方法、以及对三阶随机 walk 的几何分析。

**📊 数据集**

本工作完全是理论算法，无使用实验数据集或实际图实例，结论仅通过数学证明和上界计算得到。

**📈 对比分析**

与之前最佳 SDP 基方法 O(n^0.19747) 的对比，本文将指数项从 0.19747 降到 0.19539；与传统组合方法（如 Wigderson、Blum 等）相比，虽然理论上可得到更小的颜色数，但在实际实现成本与常数项上仍显高。该结果是迄今为止 SDP 方法在 3‑可着色图着色中最优的上界。

**⚠️ 局限性**

主要限制包括：①算法仍受覆盖组合 Lemma 的 Gaussian 等距测不等式损失限制，若该界可进一步收紧可带来更小的指数；②分析极其复杂，实际实现难度大；③所得到的改进仅为几十亿分之一的指数下降，在实际规模下效果不一定显著；④尚未探索更高阶邻域或更强的组合与 SDP 结合方式。

---

## 553. Parity, Sensitivity, and Transformers

**arXiv ID:** 2602.05896 | [PDF](https://arxiv.org/pdf/2602.05896v1)

**作者:** Alexander Kozachinskiy `[一作]` (CENIA), Przemysław Wałȩga `[通讯]` (Queen Mary University of London)

**通讯引用:** 303 | [OpenAlex ID](https://openalex.org/A5053108637)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一个四层 softmax Transformer 能够计算 PARITY 的新构造，并证明了单层单头 Transformer 无法完成该任务。

**💡 创新点**

创新点包括：
   • 通过长度无关、多项式界限的位置信息编码、无层归一化，构造了仅需四层即可实现 PARITY 的软注意力网络，弥补了之前需要两层或硬注意力、长度相关编码的缺陷；
   • 给出了平均灵敏度分析和超平面划分计数的下界，证明任何单层单头 Transformer 的平均灵敏度为 O(√n)，从而证明它无法实现 PARITY。

**🔧 技术方法**

使用的技术：
   • 归一化的平均灵敏度（average sensitivity）分析；
   • 超平面划分边界计数（O(√n) 量化）
   • 位置编码的设计（基于对数、幂函数的组合）
   • 近似 Faulhaber 公式推广，计算多项式求和；
   • 软注意力权重的精确调制（通过指数函数与位置信息的线性组合）

**📊 数据集**

没有使用实际数据集，全部为理论构造与证明。

**📈 对比分析**

与之前的 2 层或硬注意力、长度相关编码的构造相比，新的 4 层软注意力模型在结构上更简洁、可实现性更高（长度无关、多项式界限），但在理论上仅给出了可计算性而未进行实验验证，故无性能数值可比。

**⚠️ 局限性**

限制：
   • 仍需要至少四层；
   • 构造为理论上可实现，并未证明在实际训练中可收敛；
   • 对非常长序列的实现仍受限于多项式位置编码的增长；
   • 仅关注 PARITY，未证明对更一般的任务同样适用。

---

## 554. Escaping Local Minima Provably in Non-convex Matrix Sensing: A Deterministic Framework via Simulated Lifting

**arXiv ID:** 2602.05887 | [PDF](https://arxiv.org/pdf/2602.05887v1)

**作者:** Tianqi Shen `[一作]` (City University of Hong Kong), Ziye Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5065399853)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在原始矩阵空间内通过模拟张量过参数化来实现确定性逃离非凸矩阵感知中诡异局部最小值的SOD逃逸方法。

**💡 创新点**

创新点在于将过参数化张量空间的逃逸方向映射回低维空间，并提供逃逸可行性评分（EFS）与多步截断投影梯度下降（TPGD）两种确定性逃逸机制，完全不依赖随机扰动或经验规则。

**🔧 技术方法**

采用理论分析、RIP约束、张量投影与梯度推导技术，并实现了单步与多步SOD逃逸算法，保证目标值下降与逃逸成功。

**📊 数据集**

实验使用了基于扰动矩阵完成（PMC）的人工数据集以及从真实系统获取的感知矩阵，验证了方法在中大规模实例上的有效性。

**📈 对比分析**

与传统梯度下降相比，SOD逃逸显著提升了成功率；相较于显式张量提升的做法，SOD在保持可计算性的同时实现了更高的成功率和更低的计算成本。

**⚠️ 局限性**

局限性包括需要满足RIP约束、逃逸步长与迭代次数受限、在高阶张量映射时数值精度可能下降，以及方法目前仅在矩阵感知框架下得到理论保证，推广至更一般非凸问题仍有待进一步研究。

---

## 555. Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations

**arXiv ID:** 2602.05885 | [PDF](https://arxiv.org/pdf/2602.05885v1)

**作者:** Wei Liu `[一作]` (Hong Kong University of Science and Technology), Junxian He `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2815 | [OpenAlex ID](https://openalex.org/A5015879697)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统研究了基于强化学习的Triton GPU核代码生成，构建了可扩展的KernelGym执行环境，设计了多轮RL策略、TRLOO优势估计、基于剖面奖励与拒采技术，以及序列化推理扩展方法，以提升模型的正确率与加速效果。

**💡 创新点**

主要创新点包括：①提出KernelGym的分布式GPU环境，实现故障隔离、细粒度奖励反馈和高效多轮交互；②发现GRPO自包含偏差并提出无偏的Turn-level REINFORCE Leave-One-Out (TRLOO)；③引入基于剖面的奖励（PR）和拒采（PRS）来抑制惰性优化和奖励劫持；④利用序列化测试时扩展（STTS）进一步提升推理阶段的加速率。

**🔧 技术方法**

采用的技术有：分布式服务器‑工作器架构、CUDA容器化子进程、奖励劫持检测、GPU剖面工具、TRLOO优势估计、几何不匹配拒采（MRS）、基于剖面奖励与拒采（PR/PRS）、多轮交互式RL、上下文管理的序列化推理。

**📊 数据集**

数据集主要来自KernelBench基准任务以及由GPT‑5生成的8K个5‑turn轨迹，用于监督预训练和RL收集；训练模型为Qwen3‑8B/14B，并使用GPU H100 进行评估。

**📈 对比分析**

与AutoTriton、Claude‑4.5‑Sonnet、GPT‑5等基线比较，-14B模型在KernelBench Level‑1/2 的 Fast@1.2 指标分别达到 25.1%/47.8%，显著超过前沿模型（Claude 26.7%/28.6%）。STTS进一步提升至 59.8%/80.9% 的最佳历史加速率，显示出优秀的性能提升。

**⚠️ 局限性**

局限性包括：①数据量相对有限，缺乏更大规模的高质量 kernel 生成数据；②模型规模受限，仍需更大参数量以进一步提升效果；③当前仍未实现完全自动、生产级别的 end‑to‑end kernel 生成，需进一步完善工程化与安全验证。

---

## 556. Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training

**arXiv ID:** 2602.05933 | [PDF](https://arxiv.org/pdf/2602.05933v1)

**作者:** Zhenghao Xu `[一作]` (Georgia Institute of Technology), Tuo Zhao `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 2634 | [OpenAlex ID](https://openalex.org/A5101595500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大语言模型后训练中使用的Kimi式PMD算法进行理论分析和实证验证，推导其总体最优更新并证明其等价于自适应混合KL–χ²正则化的镜像下降子问题；

**💡 创新点**

创新点在于利用Lambert‑W函数精确表述Kimi‑PMD的更新，揭示其隐式χ²正则化带来的稳定性机制，并对一阶改进率、估计误差及收敛性进行分阶段分析；

**🔧 技术方法**

主要技术包括：政策镜像下降（PMD）、对数策略空间回归、平均奖励近似代替对数分区函数、Lambert‑W函数解析、混合KL–χ²正则化、样本复杂度分析与离线回合采样；

**📊 数据集**

使用了DAPO‑Math‑17k数据集进行训练，评估以AIME 2024和AIME 2025为准；实验模型为Qwen2.5‑7B与Qwen3‑30B‑A3B‑Base；

**📈 对比分析**

与GRPO、on‑policy梯度及GSPO对比，Kimi‑PMD在AIME得分上提升了2.6%/9.0%（7B）或14.6%/8.1%（30B），并实现约4.6×的时间加速；相比之下GRPO在训练中易失稳，Kimi‑PMD表现更稳定；

**⚠️ 局限性**

局限性：在奖励低、回合样本不足时估计误差显著；对τ、批大小等超参数敏感；缺乏全局收敛或无偏误差的理论保证；仅在情境赌博机设定下验证，可能不适用于更复杂环境。

---

## 557. Even Faster Geosocial Reachability Queries

**arXiv ID:** 2602.05928 | [PDF](https://arxiv.org/pdf/2602.05928v1)

**作者:** Rick van der Heijden `[一作]` (Eindhoven University of Technology), Thekla Hamm `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5034753800)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 2DReach 方法，用于解决地理社交网络中的 RangeReach 查询，避免区间标签并使用每个强连通分量（SCC）的 2D R‑tree。

**💡 创新点**

创新点在于完全消除 3DReach 的区间标签和 3D R‑tree，直接为每个 SCC 存 2D R‑tree；并通过排除空间 sink 与共享相同可达集的 R‑tree 进一步压缩存储。

**🔧 技术方法**

采用 SCC 分解、逆拓扑合并、2D R‑tree 索引以及空间压缩与指针共享技术。

**📊 数据集**

使用四个真实 LBSN 数据集：Yelp、Foursquare、Gowalla 与 Weeplaces。

**📈 对比分析**

与 3DReach 与 3DReach‑Rev 进行对比；2DReach 在索引构建时间更快，压缩版索引尺寸最小；查询性能更稳定，某些实验中比 3DReach 提升至十倍左右。

**⚠️ 局限性**

局限性包括仍需为每个 SCC 存 2D R‑tree，空间 sink 仍占一定存储；在 SCC 数量极少或空间节点占比极高的场景压缩效果有限；对 3D 维度或非矩形查询的适用性尚未充分验证。

---

## 558. From Bench to Flight: Translating Drone Impact Tests into Operational Safety Limits

**arXiv ID:** 2602.05922 | [PDF](https://arxiv.org/pdf/2602.05922v1)

**作者:** Aziz Mohamed Mili `[一作]` (École de technologie supérieure), David St-Onge `[通讯]` (École de technologie supérieure)

**通讯引用:** 566 | [OpenAlex ID](https://openalex.org/A5082797874)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计并搭建了可复现的桌面撞击实验平台，对多款室内微型空中车辆在3–4 m/s撞击下的力-时间、冲量、反弹能量和接触时长进行测量，并基于实验回归生成速度-冲击力映射，最终实现了一款基于ROS 2的安全总线，能在感知失效时在线限制无人机速度以满足人机接触力阈值。

**💡 创新点**

创新点在于：①首次系统地采集完整机身在室内速度范围内的撞击数据并揭示反弹与峰值力的权衡；②将实验得到的回归模型与 ISO/TS 15066 合作空间模型结合，形成端到端的数据驱动安全守门器；③提供完整的数据集、代码和 ROS 2 节点，构建可供行业直接使用的验证工具链。

**🔧 技术方法**

技术手段包括：定制的线性弹射撞击台、三轴加载计、加速度计、TFmini‑S 速率传感器、高帧率摄像机、但丁/SiCDAQ 数据采集、Kalman 滤波与正向力积分、多项式回归、ROS 2 速度限制节点、Gazebo+PX4 SITL 仿真验证。

**📊 数据集**

数据集涵盖20个样本（4款商用机型＋12款自制机型），每款4次实验，记录峰值力、冲量、接触时长、能量吸收/反弹比例以及不同撞击角度（0° 与 45°）的数据。数据已公开存放，供后续研究引用。

**📈 对比分析**

通过对比不同机型、材料和角度下的峰值力与 ISO/TS 15066 的面部、颈部、胸部阈值，验证了回归模型的准确性；在 Gazebo/PX4 仿真中，安全守门器将无人机速度限制在满足 140 N（胸部）阈值的 3 m/s 左右，同时保留了任务通量，性能表现优于仅基于几何或速度阈值的传统方法。

**⚠️ 局限性**

局限性包括：实验速度仅限 3–4 m/s、仅考虑离旋翼关机状态、样本量小（每款机型仅 4 次）、未覆盖旋翼开启撞击、未实现在线姿态补偿以及对非室内场景的适用性验证。

---

## 559. Chunky Post-Training: Data Driven Failures of Generalization

**arXiv ID:** 2602.05910 | [PDF](https://arxiv.org/pdf/2602.05910v1)

**作者:** Seoirse Murray `[一作]` (Anthropic Fellows Program), Sara Price `[通讯]` (Anthropic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并揭示了模型后训练阶段因数据意外相关性导致的“Chunky post‑training”行为失效；

**💡 创新点**

提出了两种工具（SURF 用于自动发现失效行为，TURF 用于将失效归因到训练数据），并阐述了其工作机制；

**🔧 技术方法**

技术核心包括基于属性的提示生成与黑盒评分的搜索循环（SURF），以及属性嵌入、聚类与相似度检索匹配（TURF）；

**📊 数据集**

使用前沿模型（Claude 4.5、GPT‑5.1、Gemini 3、Grok 4.1、Grok 4.1 mini 等）的官方后训练数据以及开放模型 Tülu3 的完整 SFT 数据进行实验；

**📈 对比分析**

实验显示 SURF 在多模型中发现了大量错误行为路由，TURF 能定位具体数据特征，数据消融实验进一步验证因果关系，移除或平衡相关样本可显著降低失效率；

**⚠️ 局限性**

局限性包括仅考虑单轮对话、未覆盖 RL/其他后训练阶段对失效的影响、工具对评分模型依赖且未在多轮交互上进行评估

---

## 560. Stop Rewarding Hallucinated Steps: Faithfulness-Aware Step-Level Reinforcement Learning for Small Reasoning Models

**arXiv ID:** 2602.05897 | [PDF](https://arxiv.org/pdf/2602.05897v1)

**作者:** Shuo Nie `[一作]` (Harbin Institute of Technology), Xuelong Li `[通讯]` (China Telecom Corp Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种 Faithfulness-Aware Step-Level Reinforcement Learning (FaithRL) 框架，用于在小型推理模型 (SRMs) 上抑制链式思考 (CoT) 步骤中的真确性幻觉。

**💡 创新点**

创新点在于同时引入显式（基于 PRM 的句子级奖励）和隐式（动态截断重采样产生对比信号）的步骤级奖励，精确惩罚不可信的推理步骤，并通过信息增益与重复惩罚控制生成长度与多样性。

**🔧 技术方法**

核心技术包括基于 GRPO 的强化学习、HHEM-2.1 过程奖励模型 (PRM)、动态截断重采样 (Dynamic Truncated Resampling, DTR)、句子/令牌信息增益奖励、n-gram 重复惩罚以及 LLM-as-a-judge 的评估。

**📊 数据集**

使用的评估数据集为开放书目问答 (Open-Book QA) 数据集：SQuAD、NewsQA、TriviaQA、NQ、HotpotQA，训练样本从 HotpotQA 与 2WikiMultiHopQA 随机抽取 8,000 条。

**📈 对比分析**

与 SFT、GRPO、KD、Self-Refine、FSPO 等基线比较，FaithRL 在所有 SRMs（DPSK-1.5B、Qwen3-0.6B、Qwen3-1.7B）上平均提升 3.86% 的答案 F1、3.48% 的 faithfulness；在攻击集和 Hallucination Rate 上也表现出显著优势。

**⚠️ 局限性**

局限性包括：仍依赖外部 PRM 进行句子级真确性评估，且对复杂推理场景的鲁棒性未覆盖全部可能类型；在更大规模或非开放书目任务中的适用性尚待验证。

---

## 561. Metric Hedonic Games on the Line

**arXiv ID:** 2602.05888 | [PDF](https://arxiv.org/pdf/2602.05888v1)

**作者:** Merlin de la Haye `[一作]` (Hasso Plattner Institute), Marcus Wunderlich `[通讯]` (University of Augsburg)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在一维类型空间中定义三种自然成本函数（平均距离、最大距离、阈值切断），并对跳跃式与交换式稳定性下的合伙人博弈进行理论分析，证明存在稳定结构、探讨结构是否排序、分析社会最优与稳定结构的价格差异；

**💡 创新点**

提出了基于类型距离的简洁成本模型，并给出对应的潜在函数与单调性条件，首次证明所有三种成本在交换稳定下都具有潜在函数，给出左改进移动算法求解跳跃稳定结构，系统性阐明了多种模型下 PoA 与 PoS 的界限，提出了“排序最优”猜想与“优美切断”子类的 PoS 1 结论；

**🔧 技术方法**

使用潜在函数、单调性分析、改进移动算法、组合构造、极端实例构造与图论工具等理论技术；

**📊 数据集**

无实测数据，全部基于抽象的理论构造与证明；

**📈 对比分析**

通过构造最坏实例计算 PoA 上界/下界，证明大多数模型 PoA 无界、Swap 游戏 PoS 为 1，Jump 游戏 PoS >1，除优美切断子类外；没有实验性能指标；

**⚠️ 局限性**

对 Avg-Jump 游戏的排序最优性仍是未证实的猜想，模型仅限于固定最大团数 k，未考虑动态团大小约束，结果多基于人工构造的极端实例，缺乏对随机/现实场景的验证与计算复杂度分析；

---

## 562. RISE-Video: Can Video Generators Decode Implicit World Rules?

**arXiv ID:** 2602.05986 | [PDF](https://arxiv.org/pdf/2602.05986v1)

**作者:** Mingxin Liu `[一作]` (Shanghai Jiao Tong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5701 | [OpenAlex ID](https://openalex.org/A5043503650)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们提出了 RISE-Video benchmark，用于评估文本到视频模型在隐式推理方面的能力。

**💡 创新点**

创新之处在于构建了八大推理维度、467 条人工标注样本、四维度评价体系，以及基于大型多模态模型的自动判定管线。

**🔧 技术方法**

采用 GPT‑5 等 LMM 进行多维度评估，结合手工设计的知识导向提问、帧采样策略和专门的图形对齐检查。

**📊 数据集**

使用自研的 RISE‑Video 数据集，共 467 条样本，覆盖经验、常识、时空、社会、感知、空间、学科、逻辑八个领域。

**📈 对比分析**

对 11 款 TI2V 模型进行评测，闭源模型优于开源模型，最优模型 Hailuo 2.3 仅达 22.5% 的整体准确率，凸显推理能力瓶颈。

**⚠️ 局限性**

主要局限在于模型对隐式规则的把握不足、动态响应迟缓、逻辑推理能力低，导致整体评测分数普遍偏低。

---

## 563. Geographically-aware Transformer-based Traffic Forecasting for Urban Motorway Digital Twins

**arXiv ID:** 2602.05983 | [PDF](https://arxiv.org/pdf/2602.05983v1)

**作者:** Krešimir Kušić `[一作]` (University of Zagreb), Ivana Dusparic `[通讯]` (Trinity College Dublin)

**通讯引用:** 1801 | [OpenAlex ID](https://openalex.org/A5059738292)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于Transformer并利用互信息挑选地理相关传感器的交通流预测模型。

**💡 创新点**

创新点在于用互信息做特征选择，增强Transformer的地理意识，提高预测精度而不增加模型复杂度。

**🔧 技术方法**

使用Transformer架构、互信息特征选择、概率时序预测技术。

**📊 数据集**

使用瑞士日内瓦高速公路网络的实时5分钟交通流数据（14个传感器）。

**📈 对比分析**

与标准Transformer、单传感器模型对照，MASE、sMAPE、MAE、RMSE等指标提升约50%–90%，显著优于基线。

**⚠️ 局限性**

局限：仅验证单一城市网络，缺乏跨城市泛化评估；未与图神经网络等更复杂空间模型比较；仅考虑交通流量，未加入天气等外部因子。

---

## 564. Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space

**arXiv ID:** 2602.05971 | [PDF](https://arxiv.org/pdf/2602.05971v1)

**作者:** Felipe D. Toro-Hernández `[一作]` (Federal University of ABC), Rodrigo M. Cabral-Carvalho `[通讯]` (Federal University of ABC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出了通过累积Transformer文本嵌入构建语义检索轨迹的框架，并提取五个几何与动力学指标（相邻距离、速度、加速度、熵、到质心距离）来细粒度刻画人类语义搜索过程。

**💡 创新点**

创新点在于：①把语义检索视为在多维嵌入空间中的连续运动；②使用累积嵌入捕捉上下文依赖的检索历史；③引入物理学灵感的动力学指标，实现对搜索局部与全局动态的双重刻画；④在不同语言与临床人群上验证其可迁移性与鲁棒性；⑤极大降低人工预处理成本。

**🔧 技术方法**

技术方法包括：Transformer基础文本嵌入（OpenAI text‑embedding‑3‑large、Google text‑embedding‑004、Qwen3‑Embedding‑0.6B、fastText），累积编码；计算余弦距离、向量差、二阶差、熵和质心距离；ZCA‑whitening 校正空间异质性；使用GLMM对组/类别效应进行统计检验。

**📊 数据集**

使用四个公开数据集：①神经退行性疾病组（西班牙语PD、bvFTD、HC）的属性列举；②英语谩骂词汇流畅性（包含动物、首字母、谩骂词）；③意大利语属性列举；④德语属性列举。

**📈 对比分析**

通过与fastText的非累积基线对比、不同嵌入模型（OpenAI、Google、Qwen3）交叉验证，并用GLMM评估组间差异，结果显示：累积嵌入在长轨迹上表现更优；各模型在局部动力学指标上高度相关，能显著区分临床与对照组、不同语义类别；到质心距离和熵指标在跨模型间关联最弱，体现全局几何差异。

**⚠️ 局限性**

局限性：①任务缺乏时间戳，无法捕捉真实时间动态；②假设欧氏几何，未考虑嵌入空间的非欧氏或各向异性；③仅针对流畅性/属性列举任务，难以推广到更复杂的语义检索情境；④仅使用简化的熵度量，缺乏更丰富的信息论工具；⑤到质心距离受模型几何影响大，跨模型比较受限。

---

## 565. LSA: Localized Semantic Alignment for Enhancing Temporal Consistency in Traffic Video Generation

**arXiv ID:** 2602.05966 | [PDF](https://arxiv.org/pdf/2602.05966v1)

**作者:** Mirlan Karimov `[一作]`, Marc Pollefeys `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对自动驾驶场景中的视频生成，本文提出了一种Localized Semantic Alignment (LSA)框架，通过在训练阶段对生成视频与真实视频的语义特征进行对齐，以提升时序一致性。

**💡 创新点**

创新点在于：1）仅在训练时使用动态物体的真实框进行局部语义特征一致性损失，避免推理时需要额外控制信号；2）只需单个训练 epoch 即可显著提升时序连贯性；3）将语义特征对齐与标准扩散损失联合，兼顾视觉质量与动态一致性。

**🔧 技术方法**

技术手段包括：Stable Video Diffusion (SVD) 作为基础生成模型；DINOv2 作为预训练语义特征提取器；VAE 编码解码；扩散损失与语义一致性损失的联合训练；在训练时对U‑Net 进行全局微调。

**📊 数据集**

实验使用的公开数据集为 nuScenes（单视角 RGB）和 KITTI Tracking（2D/3D 框）。

**📈 对比分析**

与基线 SVD、SVD 细调、以及Ctrl‑V 1‑to‑0 进行对比。评估指标为 FVD、FID、mAP 与 mIoU。结果显示：LSA 在两大数据集上均使 FVD、FID 降低约10‑18%；mAP 提升约40‑50%；mIoU 亦提升约2‑6%。同时不增加推理时间或模型参数。

**⚠️ 局限性**

主要局限：训练阶段需要真实的动态物体框标注；目前仅支持单一未来序列，未能生成多种可行未来；依赖 DINOv2 语义特征，若目标场景与训练域差异较大可能效果受限。

---

## 566. Learning to Share: Selective Memory for Efficient Parallel Agentic Systems

**arXiv ID:** 2602.05965 | [PDF](https://arxiv.org/pdf/2602.05965v1)

**作者:** Joseph Fioresi `[一作]` (University of Central Florida), Mubarak Shah `[通讯]` (University of Central Florida)

**通讯引用:** 58296 | [OpenAlex ID](https://openalex.org/A5080823547)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了学习共享内存机制（LTS），在并行代理系统中通过轻量级控制器实现对中间步骤的选择性录入，从而减少冗余计算并提升执行效率。

**💡 创新点**

创新点在于：1) 设计全局键值对共享内存；2) 用强化学习训练的记忆录入控制器，结合使用感知奖励与稀疏正则化解决稀疏监督；3) 将共享内存与并行代理框架无缝集成，保持团队探索性同时降低重复工作。

**🔧 技术方法**

技术包括：轻量级Transformer（LoRA）控制器、冻结文本嵌入模型、stepwise强化学习（优势估计、使用感知奖励）、并行代理框架MagenticOne/M1-Parallel。

**📊 数据集**

使用 GAIA 与 AssistantBench 两个长链任务基准；训练记忆控制器仅在 AssistantBench dev 33 题上完成，评估在 GAIA 与 AssistantBench 上。

**📈 对比分析**

与无共享、全共享、LLM筛选等策略对比，采用 K=3 并行团队。实验显示 LTS 在 AssistantBench 和 GAIA 上保持或提升准确率的同时，平均降低 40–55% 的壁时运行时间，显著优于基线。

**⚠️ 局限性**

局限性包括：共享内存仅在单任务内，非跨任务持久；控制器不处理记忆删除或更新；训练依赖稀疏奖励且耗时；未在更大规模硬件上验证扩展性。

---

## 567. Multi-Scale Global-Instance Prompt Tuning for Continual Test-time Adaptation in Medical Image Segmentation

**arXiv ID:** 2602.05937 | [PDF](https://arxiv.org/pdf/2602.05937v1)

**作者:** Lingrui Li `[一作]` (University of Nottingham), Zhun Zhong `[通讯]` (Hefei University of Technology)

**通讯引用:** 10551 | [OpenAlex ID](https://openalex.org/A5065328976)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种多尺度全局-实例提示调优（MGIPT）框架，实现医疗图像分割中的持续测试时自适应，避免了错误累积和灾难性遗忘。

**💡 创新点**

创新点在于引入自适应尺度实例提示（AIP）与多尺度全局提示（MGP）的双层提示调优，并通过早停、教师-学生EMA更新和置信度加权融合，兼顾实例特异性与域级知识，同时消除记忆库导致的隐私泄露风险。

**🔧 技术方法**

使用的技术包括视觉提示调优、频域低频提示、BN对齐损失、逐步提示调优与早停、基于实例的自适应尺度选择、教师-学生EMA更新、置信度加权融合等。

**📊 数据集**

使用的数据集为：光学视网膜眼底分割（5个中心的OD/OC数据集）和息肉分割（4个中心的Polyp数据集）。

**📈 对比分析**

与7种主流CTTA/TTA方法（TENT、CoTTA、DLTTA、SAR、DomainAdaptor、MoASE、VPTTA）以及源域基线进行对比，MGIPT在所有域上平均Dice提升约2-4%，在长期CTTA中保持最小性能衰退，表现最优。

**⚠️ 局限性**

局限性在于需要对每个样本进行迭代优化以寻找最优提示尺度，导致推理时计算量增加。

---

## 568. Dimensionality Reduction on Riemannian Manifolds in Data Analysis

**arXiv ID:** 2602.05936 | [PDF](https://arxiv.org/pdf/2602.05936v1)

**作者:** Alaa El Ichi `[一作]` (University of Littoral Cote d'Opale), Khalide Jbilou `[通讯]` (University of Littoral Cote d'Opale)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统研究了基于黎曼几何的降维方法，重点提出并实现了PGA、R-RPCA、R-ONPP、R-LE、R-LDA、R-Isomap和RSVM等多种适用于曲率空间的数据降维与分类框架，并给出了统一的实现与实验验证。

**💡 创新点**

创新点在于：①提出了统一的黎曼空间降维框架，将PCA、LDA、Isomap等经典方法推广到非欧几里得空间；②引入了在切空间上的鲁棒PCA和正交邻域保持投影；③结合黎曼梯度优化、谱分解与核方法，形成了一整套从表示到分类的端到端流程。

**🔧 技术方法**

采用的技术包括：黎曼指数/对数映射、Fréchet均值估计、切空间线性化、黎曼梯度优化、谱嵌入（Laplacian Eigenmaps、Isomap）、鲁棒PCA、LDA、SVM与基于几何核的R-SVM。

**📊 数据集**

实验使用了12个基准数据集：4个常见实值数据（MNIST 8×8、Wine、Breast Cancer、Synthetic HD）；4个三维非线性嵌入（Swiss Roll、S-Curve、3D Moons、3D Circles）；4个球面数据（Sphere Hard、Great Circle、Sphere Bands、Rings）。

**📈 对比分析**

与欧氏基线（PCA、LDA、Isomap）进行对比，实验表明在球面和高曲率数据上，黎曼方法的分类准确率可达100%，在MNIST等任务中R-LE达90.2%优于PCA 73.5%；总体而言，黎曼方法在保持几何结构、提升下采样质量和分类性能方面均优于传统欧氏方法。

**⚠️ 局限性**

主要限制包括：对几何运算（指数/对数、几何距离）和Fréchet均值的数值求解耗时，某些方法（R-LE、R-Isomap）缺乏天然的出样扩展；在高维或大规模数据上计算复杂度高；以及对球面或SPD等特殊流形的近似误差和理论收敛性尚待进一步研究。

---

## 569. Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

**arXiv ID:** 2602.05935 | [PDF](https://arxiv.org/pdf/2602.05935v1)

**作者:** Sudeepta Mondal `[一作]` (RTX), Ganesh Sundaramoorthi `[通讯]` (RTX)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在没有额外 OOD 数据的情况下对 OOD 检测器参数进行调优的方法；通过在训练集上随机留出部分类别生成模拟 OOD 数据，并利用多份不同训练出的任务网络对调参进行平均，从而实现自洽调参。

**💡 创新点**

创新点在于：1）不再依赖真实或人工生成的 OOD 样本；2）通过在训练集内部切分类别来构造近似 OOD 样本；3）在多份不同的任务网络上评估调参效果，并采用 Bayesian Optimization 对参数空间进行搜索，显著降低对调参集的敏感性。

**🔧 技术方法**

使用的技术包括：- 模拟 OOD 的类别留置策略（将训练集随机划分为 OOD 类和 ID 类）；- 对每个划分训练一个任务网络并从中采样验证集；- 在所有网络与验证集上计算损失并取平均；- 采用 Bayesian Optimization 对 OOD 检测器的参数空间进行全局搜索；- 在 ReAct、ASH、KNN、VRA、PLF 等后置 OOD 检测器上实现。

**📊 数据集**

实验数据集包括：CIFAR‑10、CIFAR‑100、ImageNet‑200、ImageNet‑1k；使用的模型架构有 ResNet‑18、ResNet‑50、MobileNet‑V2。

**📈 对比分析**

与 Gaussian noise 与 adversarial perturbation 两种基线（均不使用真实 OOD 数据）以及 OpenOOD 提供的真实 OOD 调参集进行比较；结果显示：在高参数检测器（VRA、PLF）上，本方法明显优于基线，在低参数检测器（ReAct、ASH、KNN）上保持竞争力，整体性能与使用真实 OOD 调参集相当甚至略优。

**⚠️ 局限性**

局限性：需要为每一次划分训练多份任务网络，增加计算成本；目前仅在分类任务上验证，难以直接推广到分割、语义理解或大规模语言模型等更复杂任务。

---

## 570. Adaptive Hashing: Faster Hash Functions with Fewer Collisions

**arXiv ID:** 2602.05925 | [PDF](https://arxiv.org/pdf/2602.05925v1)

**作者:** Gábor Melis `[一作]` (Google DeepMind), Gábor Melis `[通讯]` (Google DeepMind)

**通讯引用:** 380 | [OpenAlex ID](https://openalex.org/A5039107666)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文实现并评估了一种在线自适应哈希函数框架，可在哈希表插入过程中根据当前键集动态调整哈希函数。

**💡 创新点**

创新点在于将哈希函数的选择转变为在线可调节、键感知的过程，并提供轻量级适配机制，无需改变 API。

**🔧 技术方法**

采用基于键分布检测的移位、指针混合、常量哈希等自适应哈希函数，以及链长和碰撞计数触发的重哈希策略。

**📊 数据集**

数据集包括SBCL运行时的约4万条字符串键、列表键以及整数/指针键（包括固定进程、随机进程和实际对象地址）。

**📈 对比分析**

通过与 Murmur、Prefuzz、Uniform 等传统哈希函数在插入、查找、删除以及宏基准的基准测试比较，实验显示自适应哈希在插入与总体工作负载中平均提升约8–15%，在宏基准中提升约0.7%至1.5%。

**⚠️ 局限性**

局限在于适配开销对极小或极大表的收益有限，对链长阈值的粗糙估计可能导致不必要的重哈希，且在不规则键分布时仍无法完全匹配最优哈希。

---

## 571. DFPO: Scaling Value Modeling via Distributional Flow towards Robust and Generalizable LLM Post-Training

**arXiv ID:** 2602.05890 | [PDF](https://arxiv.org/pdf/2602.05890v1)

**作者:** Dingwei Zhu `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 16592 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了DFPO框架，通过在价值函数中引入连续时间流场的分布式建模来提升RL在噪声和OOD环境下的鲁棒性和泛化。

**💡 创新点**

①将价值函数从离散分位数转化为连续流场建模，利用ODE向量场逼近返回分布；②引入条件风险控制和一致性约束，抑制低尾噪声并保证流路径平滑；③采用单步欧拉推理实现高效且稳健的价值估计。

**🔧 技术方法**

流模型（Neural ODE/Flow Matching）、分布式优势估计（Distributional GAE）、风险敏感约束（CVaR、尾曲率正则化）、几何一致性正则化、谱归一化、基于Transformer的backbone。

**📊 数据集**

多轮对话（Honor‑Dialogue Dataset）、数学推理（Light‑R1, MATH500, AIME24, Minerva‑Math, AMC23）、科学问答（SuperGPQA, SampleQA, GPQA, HLE）以及基于Qwen3‑8B的模型。

**📈 对比分析**

与PPO、GRPO、Dr.GRPO、KTAE、λ‑GRPO、BAPO、Robust Bellman、FlowRL等基线比较。实验表明，DFPO在对话任务平均准确率≈86.7%，在数学和科学任务ID/OOD平均≈39‑40%，均优于所有基线，尤其在噪声和OOD场景下表现出稳定性和无灾难性崩溃。

**⚠️ 局限性**

参数调优需任务特定；单步直线近似对高度非线性价值动态的适应有限；对极端奖励污染仍可能导致流场崩溃；需要进一步研究更强的噪声鲁棒机制和多步推理的平衡。

---

## 572. Neural Implicit 3D Cardiac Shape Reconstruction from Sparse CT Angiography Slices Mimicking 2D Transthoracic Echocardiography Views

**arXiv ID:** 2602.05884 | [PDF](https://arxiv.org/pdf/2602.05884v1)

**作者:** Gino E. Jansen `[一作]` (Amsterdam UMC), Ivana Išgum `[通讯]` (Amsterdam UMC)

**通讯引用:** 16204 | [OpenAlex ID](https://openalex.org/A5084070018)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

通过学习来自CTA的多类形状先验，使用神经隐式函数从稀疏的TTE视角分割图像重建完整的3D心脏结构。

**💡 创新点**

关键创新在于将视角姿态优化与形状潜在向量联合在推理阶段进行，使得即使视角初始误差较大也能得到高精度重建。

**🔧 技术方法**

采用多层感知机（MLP）隐式函数、轴角旋转参数化的刚性变换以及两阶段的梯度优化。

**📊 数据集**

训练与评估使用来自452名急性缺血性卒中患者的CTA全心段段落，选取153例高质量分割，测试集40例。

**📈 对比分析**

与基准Simpson双平面法比较，平均Dice 0.86±0.04，左心室体积误差降低至4.88±4.26 mL（相对误差约5%），显著优于传统方法。

**⚠️ 局限性**

主要局限在于仅使用CTA模拟的视角，未在真实TTE图像上验证；视角估计仍受分割误差影响；右心室在标准视角下分辨率不足。

---

## 573. Verification of the Implicit World Model in a Generative Model via Adversarial Sequences

**arXiv ID:** 2602.05903 | [PDF](https://arxiv.org/pdf/2602.05903v1)

**作者:** András Balogh `[一作]` (University of Szeged), Márk Jelasity `[通讯]` (HUN-REN-SZTE Research Group on AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过对围棋/象棋的生成序列模型进行对抗性序列生成评估，验证其隐式世界模型是否与真实世界模型一致。

**💡 创新点**

提出了新的对抗性验证框架和多种攻击策略（IMO、BSO、AD、RM、SMM），并揭示了训练规模、目标和数据来源对模型可验证性的重要影响。

**🔧 技术方法**

使用GPT‑2架构，结合下一词预测、概率分布预测、多任务学习（加入棋盘状态探测器）以及不同的解码策略（贪心、top‑k）进行训练与评估。

**📊 数据集**

实验使用了500k‑10M的随机生成游戏、8M Stockfish 对弈游戏、16M Lichess 人类游戏以及1M/5M/10M 的随机游戏数据集。

**📈 对比分析**

与贪心解码和 top‑k 解码下的五种攻击对比，发现 IMO 最高成功率；大型数据集和概率分布目标显著提升“可听”度，但模型仍不完全可验证，性能受限于数据分布和解码策略。

**⚠️ 局限性**

局限性包括仅使用单一 GPT‑2 架构、对弈策略固定、未探究更复杂的世界模型和不同语言模型架构，且对抗方法仅针对合法序列，未考虑更广泛的异常情况。

---

## 574. Orthogonal Model Merging

**arXiv ID:** 2602.05943 | [PDF](https://arxiv.org/pdf/2602.05943v1)

**作者:** Sihan Yang `[一作]` (Chinese University of Hong Kong), Weiyang Liu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 615 | [OpenAlex ID](https://openalex.org/A5115593625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在正交群流形上进行模型融合的OrthoMerge方法，能够在不破坏预训练权重几何结构的前提下将多任务微调模型融合为单一模型。

**💡 创新点**

创新点在于：1）在Lie代数中对正交变换做幅度校正的平均，以保持旋转强度；2）利用Cayley变换将融合结果映射回正交群；3）为非OFT模型设计Orthogonal‑Residual Decoupling，先提取隐式正交分量再分别在流形与欧氏空间上融合。

**🔧 技术方法**

技术核心包括正交变换的Lie代数表示、Cayley参数化、正交Procrustes求解、幅度校正平均、残差融合以及对OFT/LoRA/全微调模型的通用处理。

**📊 数据集**

实验使用Llama‑3.1‑8B、Qwen2.5‑3B以及Llama‑3.2‑3B等基础模型，微调数据集包括ScienceQA、CommonsenseQA、Social‑IQA、Magicoder‑OSS‑Instruct、NuminaMath‑TIR、MATH500、HumanEval+、M‑ARC、AGIEval、MMLU、MergeBench（IFEval、GSM8k、HumanEval+、MBPP+、ARC、WildGuardTest、HarmBench、XSTest、DoAnythingNow）及视觉‑语言集SenseNova‑SI、olmOCR、HuatuoGPT、MMSI‑Bench、EmbSpatial、MMMU、PathVQA、OCRBench、CharXiv、MMBench。

**📈 对比分析**

与TA、TIES、TSV‑M、DARE等基线比较，OrthoMerge在任务专属、跨域泛化和整体性能上均表现更好，平均准确率提升约3–5个百分点，并显著降低灾难性遗忘。

**⚠️ 局限性**

局限性包括：对正交变换的依赖导致对OFT模型更友好；对非OFT模型需要额外的正交残差分离步骤；在极大任务数或维度极高时仍存在计算和内存开销；在某些任务中正交分量提取可能不足以捕获全部适配信息。

---

## 575. Codified Finite-state Machines for Role-playing

**arXiv ID:** 2602.05905 | [PDF](https://arxiv.org/pdf/2602.05905v1)

**作者:** Letian Peng `[一作]` (University of California, San Diego), Jingbo Shang `[通讯]` (University of California, San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过自动从角色文本档案中提取关键状态，并利用LLM生成可执行的有限状态机，构建角色潜在状态的可追踪模型。

**💡 创新点**

提出 Codified Finite‑State Machine (CFSM) 与其概率扩展 CPFSM，既保持 FSM 的可解释性与确定性，又通过 LLM 自动编码转移逻辑和引入概率分布，实现更细腻且可扩展的状态建模。

**🔧 技术方法**

使用 LLM 进行状态抽取、代码生成与条件检测，构建可执行 FSM；利用概率矩阵与 softmax 生成 CPFSM；配合判别器与细化的条件检查实现高效推理。

**📊 数据集**

在 Fandom Benchmark（83 角色、5,141 场景）以及合成 Mario、COD、Tyrion 等三类 FSM 进行实验验证。

**📈 对比分析**

与提示式、文本档案、PromptTrans、角色更新、情节摘要等基线比较，CFSM/CPFSM 在 NLI 分数上平均提升约 1–2%，并在小模型上保持优势；CPFSM 在多样化回复探索（Best@K）中表现最优。

**⚠️ 局限性**

受限于预设的状态集合、对 LLM 提取质量的依赖、缺乏数值动态（如血量）以及无法动态添加新技能或特征的能力。

---

## 576. ContextBench: A Benchmark for Context Retrieval in Coding Agents

**arXiv ID:** 2602.05892 | [PDF](https://arxiv.org/pdf/2602.05892v1)

**作者:** Han Li `[一作]` (Nanjing University), He Ye `[通讯]` (University College London)

**通讯引用:** 448 | [OpenAlex ID](https://openalex.org/A5101610258)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个新的基准数据集，用于评估大型语言模型驱动的编程代理在问题解决过程中检索代码上下文的能力；

**💡 创新点**

创新点在于：1）引入人类验证的“黄金上下文”作为中间指标；2）设计自动化评估框架，实时追踪并量化检索的召回率、精准率和效率；3）从过程层面揭示代理与LLM的检索行为；

**🔧 技术方法**

主要技术包括：使用Tree‑Sitter解析AST、构建文件/块/行三层坐标系、基于规则与嵌入的任务去重、专家标注与LLM验证、以及对代理轨迹的实时记录与指标计算；

**📊 数据集**

数据集为“ContextBench”，包含1136个GitHub issue任务，跨8种语言、66个仓库，含52万行黄金上下文；

**📈 对比分析**

与现有的SWE‑Bench等终端成功率指标相比，ContextBench能细粒度评估不同代理与LLM在检索上的表现；实验显示：高级代理与LLM在召回率上占优，但精准率低，复杂框架并未显著提升检索效果；

**⚠️ 局限性**

局限性包括：黄金上下文依赖单一LLM验证，可能忽略多解情况；评估侧重静态检索，未覆盖动态运行时信息；以及对代理行为的解释仍依赖后续分析。

---

## 577. When Elo Lies: Hidden Biases in Codeforces-Based Evaluation of Large Language Models

**arXiv ID:** 2602.05891 | [PDF](https://arxiv.org/pdf/2602.05891v1)

**作者:** Shenyu Zheng `[一作]` (Centre for Software Excellence, Huawei), Ahmed E. Hassan `[通讯]` (Queen's University)

**通讯引用:** 23757 | [OpenAlex ID](https://openalex.org/A5091586373)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型（LLM）在Codeforces Elo评估中隐藏的偏差，系统评估了提交顺序、比赛难度选择和运行变异对 Elo 分数的影响；

**💡 创新点**

首次量化了三种实验因素对 Elo 分数的最大波动（最多 1,122 分），并提供了公开可复现的基准数据与自动化验证流程；

**🔧 技术方法**

使用 LLM 自动生成测试用例、自动生成验证脚本、模拟 Codeforces Elo 评分更新，并在多模型（DeepSeek、Qwen、MiniMax 等）上进行实验；

**📊 数据集**

基准包含 37 场 Codeforces 竞赛、260 道题目以及 13,691 条自动生成的测试用例；

**📈 对比分析**

通过改变提交顺序、选择不同分区比赛以及多次运行同一模型，比较其 Elo 分数，结果显示分数差异可达 394 分（提交顺序）、1,122 分（比赛难度）和 349 分（运行变异），表明 Elo 评估高度敏感；

**⚠️ 局限性**

主要限制包括生成测试用例的覆盖率不完全、官方判题不可访问导致的验证误差、LLM 自动生成验证脚本可能存在错误。

---

## 578. From Human-Human Collaboration to Human-Agent Collaboration: A Vision, Design Philosophy, and an Empirical Framework for Achieving Successful Partnerships Between Humans and LLM Agents

**arXiv ID:** 2602.05987 | [PDF](https://arxiv.org/pdf/2602.05987v1)

**作者:** Bingsheng Yao `[一作]` (Northeastern University), Dakuo Wang `[通讯]` (Northeastern University)

**通讯引用:** 31187 | [OpenAlex ID](https://openalex.org/A5080292717)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并组织“人类-代理协作”主题的CHI 2026研讨会，制定研究议程和交互设计活动。

**💡 创新点**

将远程协作者概念与LLM代理结合，建立以CSCW理论为基础的新合作框架，推动跨学科社区共识。

**🔧 技术方法**

基于LLM代理的自然语言交互、混合主动性工作流程与工作空间感知设计方法。

**📊 数据集**

无直接数据集；聚焦理论与设计实践。

**📈 对比分析**

通过现场访谈、Lightning Talk、设计工作坊等方式评估框架的可行性，尚未有定量性能指标。

**⚠️ 局限性**

缺乏经验性验证，潜在的技术与伦理风险未被充分评估；研究范围受研讨会规模与参与者限制。

---

## 579. SAGE: Benchmarking and Improving Retrieval for Deep Research Agents

**arXiv ID:** 2602.05975 | [PDF](https://arxiv.org/pdf/2602.05975v1)

**作者:** Tiansheng Hu `[一作]` (New York University), Chen Zhao `[通讯]` (New York University)

**通讯引用:** 71668 | [OpenAlex ID](https://openalex.org/A5019034689)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个名为SciAG的科学文献检索基准，并在此基准上评估了六种深度研究代理与三种检索器（BM25、ReasonIR、gte-Qwen2-7B-instruct）的性能，探讨了检索器与代理的协同机制。

**💡 创新点**

创新点在于发现 LLM‑based retrievers 在深度研究代理工作流中表现落后 BM25 约 30%，并提出通过在文档层面加入元信息与关键词的测试时扩充（corpus‑level test‑time scaling）来显著提升检索效果。

**🔧 技术方法**

使用的技术包括 LLM‑based agents（GPT‑5、Gemini‑2.5 等）、传统 BM25 检索、ReasonIR 与 gte‑Qwen2‑7B‑instruct 的向量检索，以及基于 Qwen‑3‑Next‑80B‑A3B‑Instruct 的关键词抽取和文档扩充。

**📊 数据集**

使用的数据集为 SciAG benchmark，包含 1,200 个跨计算机科学、自然科学、医疗健康与人文四大领域的查询，以及 200,000 篇最新学术论文组成的检索语料库。

**📈 对比分析**

通过精确匹配（EM）和加权召回（Weighted Recall）对比实验，六款代理在短问答任务中 GPT‑5 最高（EM 71.69%），BM25 在检索任务中显著优于 LLM‑based retrievers（约 30%），而 corpus‑level scaling 在短问答上使 BM25 提升 8.18% 以上，LLM 检索器提升有限。

**⚠️ 局限性**

主要局限在于未对代理进行检索器感知的指令微调，实验仅以 DR Tulu 为代表，无法完全推广至其他代理；此外检索器与代理间的关键词式查询不匹配导致检索覆盖率受限。

---

## 580. Discrete diffusion samplers and bridges: Off-policy algorithms and applications in latent spaces

**arXiv ID:** 2602.05961 | [PDF](https://arxiv.org/pdf/2602.05961v1)

**作者:** Arran Carter `[一作]` (University of Edinburgh), Nikolay Malkin `[通讯]` (University of Edinburgh)

**通讯引用:** 247 | [OpenAlex ID](https://openalex.org/A5068089852)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了离散空间中的扩散采样器，并引入离线（off‑policy）强化学习技术以提升采样质量，进一步将其推广至 Schrödinger 桥问题，并在离散潜在空间实现后向采样；

**💡 创新点**

创新点包括：1）首次把重放缓冲、重要性加权缓冲和 MCMC 探索等离线 RL 策略应用于离散扩散采样器；2）提出了数据到能量的离散 Schrödinger 桥算法；3）展示了离散扩散采样器在离散 VQ‑VAE 隐空间后向采样中的可行性；

**🔧 技术方法**

使用的技术主要有离散扩散采样器、二次矩损失（TB/LV）、重放缓冲、重要性加权缓冲、MCMC 探索、迭代比例拟合（IPF）以及 VQ‑VAE 等；

**📊 数据集**

实验数据集包括 Ising 与 Potts 统计力学模型、合成多峰分布（40GMM、ManyWell）、MNIST VQ‑VAE 隐空间；

**📈 对比分析**

与传统 MH、MDNS、on‑policy 采样器对比，离线策略训练的采样器在 ELBO、EUBO、MMD、Sinkhorn 等评估指标上均有显著提升，尤其在高模态和低温场景下有效缓解模态崩溃；

**⚠️ 局限性**

主要局限包括：需预设噪声核、轨迹滚动与缓冲存储成本较高、尚未在大规模生成模型上验证、离散空间连续时间极限理论待进一步完善。

---

## 581. Shared LoRA Subspaces for almost Strict Continual Learning

**arXiv ID:** 2602.06043 | [PDF](https://arxiv.org/pdf/2602.06043v1)

**作者:** Prakhar Kaushik `[一作]` (Johns Hopkins University), Alan Yuille `[通讯]` (Johns Hopkins University)

**通讯引用:** 105888 | [OpenAlex ID](https://openalex.org/A5086706224)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Share方法，实现使用单一共享低秩子空间进行参数高效的连续微调，支持多任务、多模态模型；

**💡 创新点**

创新点在于动态更新共享子空间，无需重放数据、无额外模型，显著降低参数和内存需求，并通过解析式合并实现无梯度更新；

**🔧 技术方法**

结合LoRA基础结构、SVD提取低秩子空间、渐进式子空间更新、系数优化以及无梯度的解析合并等技术；

**📊 数据集**

使用GLUE（NLP）、CIFAR‑100/Food‑101/Caltech‑101/Flowers‑102（图像）、Pascal3D+（3D姿态）、Flux/CLIP/Diffusion等文本‑图像生成任务的数据集；

**📈 对比分析**

与非连续LoRA、联合LoRA、EWC/LwF、Prompt/Adapter等方法比较，Share在参数上提升约100×、内存约281×，平均性能与上界相当，显著降低遗忘率并实现向后知识迁移；

**⚠️ 局限性**

局限性包括：需要初始LoRA或数据来构建子空间；对极端多样化任务子空间可解释性有限；在极大任务序列中子空间维度可能增长；目前未验证从零开始训练的效果。

---

## 582. CommCP: Efficient Multi-Agent Coordination via LLM-Based Communication with Conformal Prediction

**arXiv ID:** 2602.06038 | [PDF](https://arxiv.org/pdf/2602.06038v1)

**作者:** Xiaopan Zhang `[一作]` (University of California), Jiachen Li `[通讯]` (University of California)

**通讯引用:** 24144 | [OpenAlex ID](https://openalex.org/A5070982282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于大语言模型的分布式通信框架 CommCP，用于多机器人多任务的 Embodied Question Answering（MM‑EQA）问题，利用对话式 LLM 进行场景感知、问题解答和信息共享。

**💡 创新点**

创新点在于：①将 conformal prediction（CP）引入 LLM 输出校准，显著降低错误信息传播；②设计了针对目标与相关对象的通信模板，使机器人只在自信时发送与任务相关的自然语言信息；③构建了 MM‑EQA benchmark，首次在多人、多人任务下评估协作 EQA。

**🔧 技术方法**

技术包括：视觉语言模型（Prismatic‑VLM‑13B）感知；LLaMA3‑8B‑instruct 进行自然语言生成与概率输出；CP 进行置信度校准；前景基探索（FBE）和语义价值地图引导运动；Gaussian 平滑与前沿搜索实现高效导航。

**📊 数据集**

使用 Habitat‑Matterport 3D（HM3D）场景生成 70 个真实感 3D 室内环境，设计 420 个多项选择式 EQA 问题，另外 20 个场景用于 CP 校准数据集。

**📈 对比分析**

与 MMFBE、MMEuC 等基线对比，CommCP 在成功率上提升约 0.03（从 0.65 提升到 0.68），在归一化时间成本上缩短约 0.4（从 0.8 降至 0.4）。CP 校准效果明显，未校准版本性能近乎无通信的 MMEuC；通信量控制实验表明信息质量高于数量。

**⚠️ 局限性**

局限性包括：仅在 2-3 机器人规模上验证，难以直接推广至更大团队；依赖 LLaMA3‑8B 之类的开源模型，性能可能不及 GPT‑4V；CP 校准需额外采样，适应性受限；通信延迟在极高负载下仍可能影响实时协作。

---

## 583. InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions

**arXiv ID:** 2602.06035 | [PDF](https://arxiv.org/pdf/2602.06035v1)

**作者:** Sirui Xu `[一作]` (University of Illinois Urbana-Champaign), Liangyan Gui `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 InterPrior，一种可扩展的物理驱动生成控制器，利用大规模模仿专家的蒸馏和强化学习微调，实现稀疏目标驱动的全身交互。

**💡 创新点**

创新点在于：① 将完整轨迹模仿专家蒸馏为带可学习潜在空间的变分策略；② 引入潜在约束与正则化，保证多模态可行性；③ 通过“中间插值”式的 RL 微调，实现从稀疏目标到全局状态的鲁棒性与自恢复。

**🔧 技术方法**

核心技术包括：大规模模仿学习（PPO），多模态变分蒸馏（latent VAE + Transformer prior），潜在空间正则化与投影，RL 微调（PPO + 终止/成功奖励），以及物理扰动与域随机化。

**📊 数据集**

主要使用 InterAct 及其子集 OMOMO、BEHAVE、HODome 的日常交互数据，辅以物理随机化来扩展动态分布。

**📈 对比分析**

与 InterMimic、MaskedMimic 等基线对比，在完整轨迹跟踪、稀疏目标追踪、长时序多目标链和随机初始化等任务中，InterPrior 的成功率提升 20–30%，误差下降 30–50%，且在新对象、未见交互和 sim‑to‑sim 转移上表现优异。

**⚠️ 局限性**

局限性包括：对极薄或未见形状的物体仍易失去接触；在多目标链中当标准化产生大偏差时倾向保持平衡而非完成精细目标；对极端动态扰动或极端几何形状的鲁棒性仍待提升。

---

## 584. Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction For 3D-Aware Distillation

**arXiv ID:** 2602.06032 | [PDF](https://arxiv.org/pdf/2602.06032v1)

**作者:** David Shavin `[一作]` (Hebrew University of Jerusalem), Sagie Benaim `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 689 | [OpenAlex ID](https://openalex.org/A5081028371)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在教师网络中加入前向3D重建管线，将2D特征提升到3D Gaussian表示，并投影到新视角，用以监督学生网络，从而提升Vision Foundation Models的3D意识。

**💡 创新点**

①使用前向3D重建替代慢速优化，②结合mask‑aware上采样和语义混合提升特征质量，③在学生‑教师框架中进行3D知识蒸馏。

**🔧 技术方法**

3D Gaussian Splatting（MVSplat）前向重建、mask‑aware 特征上采样与语义融合、DINOv2自蒸馏框架、EMA教师更新、跨视角渲染。

**📊 数据集**

ScanNet++, ScanNet, NYUv2, SuperGlue, ADE20k, Pascal VOC, KITTI。

**📈 对比分析**

与DINOv2、FiT3D、MEF等基线进行单视角深度/法线线性探测、语义分割、跨视角对应率等评估，Splat & Distill在所有任务上均优于基线：深度RMSE提升约5–6%，法线RMSE提升约5%，语义mIoU提升2–3%，跨视角对应召回率提升。

**⚠️ 局限性**

需要人工或SAM分割掩码、依赖多视角上下文、训练主集中在室内数据，室外泛化仍有限，推理时额外的3D重建步骤带来一定计算开销。

---

## 585. A Systematic Evaluation of Large Language Models for PTSD Severity Estimation: The Role of Contextual Knowledge and Modeling Strategies

**arXiv ID:** 2602.06015 | [PDF](https://arxiv.org/pdf/2602.06015v1)

**作者:** Panagiotis Kaliosis `[一作]` (Stony Brook University), Andrew H. Schwartz `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在 1,437 名世界贸易中心受害者的自述访谈中，系统评估并对比了 11 种大型语言模型在 PTSD 严重程度估计任务中的性能，并探索了上下文知识、提示方式、模型规模、推理努力、后处理与集成等多种因素的影响。

**💡 创新点**

创新点在于：①首次在同一数据集上对多种开源与闭源 LLM 进行大规模、零/少样本评估；②通过细粒度的上下文提示（症状定义、访谈情境、分布先验等）与模型策略（推理深度、后分布重塑、集成）揭示了提升准确性的关键因素；③发现 OpenAI GPT‑5 等闭源模型在此临床任务上明显优于开放模型，并证明模型规模在 70B 参数后趋于饱和。

**🔧 技术方法**

主要技术手段包括：大规模提示工程（零样本/少样本、逐步推理、Chain‑of‑Thought、Reasoning Effort 设置）、后处理的 Predictive Redistribution、基于分布的校准、模型集成（加权平均）、与传统基于 RoBERTa 的监督回归模型进行对照。

**📊 数据集**

使用的数据集为 1,437 名受 9/11 事件影响的受害者自述访谈文本，配套自评 PTSD 检测量表 PCL‑5 分数；文本由 Whisper 语音转写得到。

**📈 对比分析**

与自评 PCL‑5 分数及两名专业评估者的评分相比，最佳模型 LLaMA‑3.1‑70B 的 Pearson 相关系数达 0.53，GPT‑5 达 0.59，均显著优于人类评估者（0.44）和监督 RoBERTa 模型（0.45）；集成策略进一步提升相关系数至 0.56，平均绝对误差（MAE）显著下降。

**⚠️ 局限性**

局限性包括：①仅使用单一访谈格式，缺乏对其他对话场景的泛化评估；②样本仅为 WTC 受害者，难以推广到其他创伤人群；③仅利用文本信息，未结合语音、视觉等多模态特征；④仅评估 PCL‑5 分数，未验证模型对正式诊断或临床评估的适用性。

---

## 586. Characterizing and Modeling the GitHub Security Advisories Review Pipeline

**arXiv ID:** 2602.06009 | [PDF](https://arxiv.org/pdf/2602.06009v1)

**作者:** Claudio Segal `[一作]` (Fluminense Federal University), Daniel Sadoc Menasché `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究对 GitHub Security Advisories（GHSA）的审查过程进行了大规模实证分析，利用 288,604 条漏洞报告（其中 23,563 条已审查）系统性描述了审查频率、路径、时延及其随时间演化的特征；

**💡 创新点**

创新点在于揭示了两条显著的审查路径——GRA（GitHub Repository Advisory）先行的快路和 NVD 首先的慢路，并通过构建 M/M/∞ 与 M/M/1 结合的排队模型解释了审查时延的二分现象；

**🔧 技术方法**

主要技术包括大数据抓取与清洗、统计检验（Mann‑Whitney U、效应量计算）、最长递增子序列（LIS）分析来验证 FIFO 性，以及基于队列网络的理论模型与仿真；

**📊 数据集**

使用的数据集为截至 2025 年 8 月的 GHSA 数据（288,604 条记录），并通过 GitHub、NVD、各生态系统数据库（如 RustSec、PyPA、RubySec 等）以及 OpenSSF Scorecard 进一步丰富了时间戳、用户与仓库元数据；

**📈 对比分析**

通过比较 GRA 与 NVD 来源的审查时延百分位数、效应量（RBC）以及后期自动化对 NVD 导入的影响，实验表明 GRA 的审查平均时延显著更短（中位数约 0.5–2 天 vs 28 天），并验证了模型对真实分布的拟合度；

**⚠️ 局限性**

局限性包括：仅分析已审查的 GHSA，未覆盖未审查案例；部分时间字段缺失或不一致导致的误差；对 NVD 自动化前后数据分段处理可能引入偏差；模型假设 FIFO 与单一服务速率，忽略批处理与人工干预等实际操作细节。

---

## 587. Supply vs. Demand in Community-Based Fact-Checking on Social Media

**arXiv ID:** 2602.06005 | [PDF](https://arxiv.org/pdf/2602.06005v1)

**作者:** Moritz Pilarski `[一作]` (Justus Liebig University Giessen), Nicolas Pröllochs `[通讯]` (Justus Liebig University Giessen)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用 X 社区笔记平台的1.1百万条请求与注释数据，实证分析了事实核查的需求与供给之间的匹配度，并评估请求展示对贡献者行为的影响。

**💡 创新点**

创新点在于：①首次大规模同时考察需求（用户请求）与供给（社区贡献者注释）的交互；②通过半实验设计（Top Writer 可见请求）量化请求展示对注释产生速率的因果效应；③揭示需求偏向高可见度、影响力账号的帖子，而供给更均衡。

**🔧 技术方法**

技术方法包括：语言/情感/主题自动标注（Gemma 3 LLM）、KDE、KS检验、卡方检验、分层 Cox 生存模型及平均边际风险比（AMHR）估计。

**📊 数据集**

数据集：X Community Notes 平台的笔记与请求原始记录（6.7万条笔记、589万请求），经 API 过滤后得到558,190 条可用笔记与请求，对应711,914 条独立推文。

**📈 对比分析**

方法对比：通过分层 Cox 模型与 AMHR 评估请求可见度对 Top Writer 注释速率的影响。结果显示，后期请求展示可使 Top Writer 注释速率提升至 1.79 倍，整体平均提升 32%。

**⚠️ 局限性**

局限性：仅限于单一平台；请求仅为一种需求信号，可能低估其他用户需求；数据缺失（已删除/屏蔽推文）影响样本完整性；请求展示时机与视图计数估计存在不确定性；未深入分析请求者动机与政治偏好等因素。

---

## 588. DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning via Semantic Matching

**arXiv ID:** 2602.06039 | [PDF](https://arxiv.org/pdf/2602.06039v1)

**作者:** Yuxing Lu `[一作]` (Peking University), Jiuxin Cao `[通讯]` (Southeast University)

**通讯引用:** 1501 | [OpenAlex ID](https://openalex.org/A5012384188)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个基于大型语言模型的多智能体框架DyTopo，利用动态语义匹配重新构建每轮稀疏有向通信图以提升多轮推理性能。

**💡 创新点**

创新点在于：①引入轻量化的“需求/提供”自然语言描述实现自适应消息路由；②通过每轮语义匹配实现实时拓扑重构，实现从探索到验证的阶段性通信转变；③提供可解释的通信轨迹与聚合顺序。

**🔧 技术方法**

核心技术包括：多角色LLM智能体、语义嵌入编码器(all‑MiniLM‑L6‑v2)、余弦相似度阈值化构建稀疏图、同步阈值与聚合顺序策略、管理者Meta‑Agent实现目标设定与终止决策。

**📊 数据集**

使用了四大类基准数据集：代码生成（HumanEval、APPS‑Competition）和数学推理（MATH‑500、Omni‑MATH），涵盖从基础到奥数级别的多难度任务。

**📈 对比分析**

与固定拓扑、随机拓扑以及现有Agent框架对比，DyTopo在所有四大LLM基底（MiMo‑V2‑Flash、GPT‑oss‑120B、Llama3‑8B‑Instruct、Qwen3‑8B）下平均提升约+6.1%，单一基底可达+17.1%，并在更难的数学任务上获得显著收益。

**⚠️ 局限性**

局限性包括：①对阈值τ的敏感性，需要手工调参；②描述词误导可能导致信息路由错误；③在极端稠密或稀疏的拓扑下易出现信息丢失或噪声累积；④目前仅在公开基准验证，实际应用中的鲁棒性待进一步评估。

---

## 589. DFlash: Block Diffusion for Flash Speculative Decoding

**arXiv ID:** 2602.06036 | [PDF](https://arxiv.org/pdf/2602.06036v1)

**作者:** Jian Chen `[一作]` (University of California San Diego), Zhijian Liu `[通讯]` (University of California San Diego)

**通讯引用:** 6994 | [OpenAlex ID](https://openalex.org/A5100453845)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DFlash，一种使用轻量级块扩散模型进行并行草稿、并用目标大模型隐藏特征做条件化的投机式解码框架，显著提升 LLM 推理速度；

**💡 创新点**

创新点包括：①将扩散模型限定为草稿阶段，充分利用其并行生成优势；②通过把目标模型隐藏层特征注入扩散模型的 KV 缓存，实现强大且可持续的条件化；③改进训练策略（anchor 采样、稀疏注意、加权损失、共享词表/LM 头），使草稿质量与接受率大幅提升；

**🔧 技术方法**

核心技术包括：块扩散模型（block diffusion）、KV 注入条件化、稀疏注意力训练、anchor 采样与加权交叉熵、共享嵌入/LM 头、SGLang 以及 FlashAttention‑4 后端；

**📊 数据集**

使用约 800K 的多任务数据集，主要来自 NVIDIA Nemotron Post‑Training Dataset V2 与 CodeAlpaca；实验中还采用 UltraChat、ShareGPT 等公开对话数据用于对比；

**📈 对比分析**

与基线自回归解码及 EAGLE‑3 进行对比，任务涵盖数学、代码和聊天；在 Qwen3‑8B、LLaMA‑3.1‑8B 等模型上，DFlash 实现平均 4.5–6.1× 的无损加速，且相较 EAGLE‑3 提升约 2.4–2.5×；在 SGLang 服务器部署场景下，单 GPU 上 5.1× 的吞吐量提升；

**⚠️ 局限性**

限制主要在于：①草稿模型仍需训练，且对目标模型隐藏特征的提取增加了存储和计算成本；②块大小的选择存在折衷，较大块能提高接受率但验证成本升高；③实验未覆盖所有扩散式投机方法（如 SpecDiff‑2、DiffuSpec），难以直接比较；④对长文本或极大模型的通用性和可扩展性尚待进一步验证。

---

## 590. Can vision language models learn intuitive physics from interaction?

**arXiv ID:** 2602.06033 | [PDF](https://arxiv.org/pdf/2602.06033v1)

**作者:** Luca M. Schulze Buschoff `[一作]` (Institute for Human-Centered AI), Eric Schulz `[通讯]` (Institute for Human-Centered AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

评估通过交互学习（GRPO）是否能提升视觉语言模型的直觉物理理解，比较其在构建与评估积木塔任务上的表现。

**💡 创新点**

首次系统比较基于一阶强化学习的交互训练与传统监督微调对视觉语言模型直觉物理推理的影响，并结合激活可解释性分析。

**🔧 技术方法**

使用Qwen3‑VL量化模型、参数高效微调（QLoRA）配合Group‑Relative Policy Optimization（GRPO）与监督微调（SFT），并设计奖励函数与线性探测。

**📊 数据集**

自制的三维积木塔图像数据集（两种块位移变体）以及Lerer等人的真实木块塔图像，用于训练与评估。

**📈 对比分析**

通过交叉任务评估、真实图像测试及激活可解码度对比，发现GRPO与SFT在任务内表现相近，但均未在零样本或真实图像上实现显著泛化。

**⚠️ 局限性**

交互仅采用一阶短期决策、数据量有限、模型规模受限，且模型可能学到基于像素的捷径，导致难以获得真正的可迁移物理直觉。

---

## 591. PhysicsAgentABM: Physics-Guided Generative Agent-Based Modeling

**arXiv ID:** 2602.06030 | [PDF](https://arxiv.org/pdf/2602.06030v1)

**作者:** Kavana Venkatesh `[一作]` (Virginia Tech), Jiaming Cui `[通讯]` (Virginia Tech)

**通讯引用:** 831 | [OpenAlex ID](https://openalex.org/A5001813016)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了 PhysicsAgentABM，一种层级神经符号框架，将推理从个体代理迁移到群体级别，并结合符号推理与多模态神经网络，实现可标定的群体级转移概率；同时提出 ANCHOR 作为 LLM 代理驱动的行为驱动聚类机制。

**💡 创新点**

①将推理从个体层面上移到群体层面并通过不确定性感知融合实现可标定的群体级过渡概率；②引入 ANCHOR，利用 LLM 进行行为相似性聚类，显著降低 LLM 调用次数并提升聚类质量；③在每个簇内部采用状态专门化符号代理与多模态神经预测并通过 epistemic fusion 进行融合；④通过聚类级推理和轻量级代理实现个体转移，显著降低计算成本。

**🔧 技术方法**

使用大型语言模型（LLM）进行符号推理；图神经网络（GraphSAGE、Spectral Clustering）和多模态编码器；状态专门化符号代理与元代理；神经网络预测（LSTM、TGN）；不确定性感知融合（epistemic fusion）；对比学习与 anchor 约束的聚类方法；滚动窗口评估与事件时间指标（EETE、ET‑F1、NLL、Brier）。

**📊 数据集**

1) 新加坡 MOH COVID‑19 数据（1000 确诊病例，2020 年 1 月 23 日至 4 月 14 日）\n2) 以 S&P 500 为基准的金融模拟（100 名交易者，2024 年 7 月至 12 月）\n3) 社会关注热度模拟（250 名代理，气候变化话题，2024 年 12 月至 2025 年 2 月）\n4) 公开的接触网络、相关系数网络和社交图。

**📈 对比分析**

与 8 种基线（机制化：Rule‑ABM、MF‑Markov；纯神经：GNN‑LSTM、TGN；LLM：单一 LLM‑Agent、flat LLM‑MAS；混合：DeepProbLog、Rule‑NN）在三大领域（流行病学、金融、社会扩散）进行滚动窗口评估。指标为 EETE、ET‑F1、NLL、Brier。PhysicsAgentABM 在所有指标上均优于基线，例如疫情域 EETE=1.92±0.05、ET‑F1=0.81、NLL=0.73、Brier=0.16；成本与可扩展性分析显示 LLM 调用次数减少 6–8 倍，token 与费用下降约 3–4 倍，性能保持不变。

**⚠️ 局限性**

仍对 LLM 的可用性与成本敏感；聚类数、状态专门化和融合权重等超参数对结果影响较大；在极端非平稳或稀缺数据情境下可能失效；未在极大规模系统中充分验证长期鲁棒性；对抗性环境与隐私约束下的可解释性和安全性仍待研究。

---

## 592. Correctness-Optimized Residual Activation Lens (CORAL): Transferrable and Calibration-Aware Inference-Time Steering

**arXiv ID:** 2602.06022 | [PDF](https://arxiv.org/pdf/2602.06022v1)

**作者:** Miranda Muqing Miao `[一作]` (University of Pennsylvania), Lyle Ungar `[通讯]` (University of Pennsylvania)

**通讯引用:** 29816 | [OpenAlex ID](https://openalex.org/A5044944954)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 CORAL，一种在推理时利用正则化 MLP 探针从冻结的残差流中直接预测残差正确性并进行加法 steering 的方法。

**💡 创新点**

创新点在于：① 通过正则化 MLP 学习分布式正确性信号，而非单个神经元或代理；② 直接优化 Brier 分数中的残差正确性，从而同时提升准确率与校准；③ 证明该 steering 方向具有跨任务迁移能力。

**🔧 技术方法**

采用的技术包括：冻结 LLM 的隐藏激活、均值池化 + z‑score 标准化、权重衰减 MLP 探针（4 层 ReLU+dropout、tanh 输出）、残差正确性预测、加法 steering 以及对多层进行实验。

**📊 数据集**

训练数据：CommonsenseQA + RACE（共 10k 题）和 MMLU（90k 辅助训练题）。测试与迁移数据：HellaSwag、OpenBookQA、Math‑MC、ARC‑Challenge 的完整公开测试集。

**📈 对比分析**

与 ITI、SteerConf、CCPS 以及基准少量样本 prompting 进行比较；在 3 个 7B 模型上平均提升 10% 准确率、50% ECE；在四个 held‑out 基准上平均提升 14% 准确率、49% ECE，显示显著优于现有方法。

**⚠️ 局限性**

局限性：仅在多选题答案评估上验证，需标注数据训练探针，推理时提取激活与计算会增加延迟，且迁移效果受目标任务与训练任务相似度影响。

---

## 593. Multi-Token Prediction via Self-Distillation

**arXiv ID:** 2602.06019 | [PDF](https://arxiv.org/pdf/2602.06019v1)

**作者:** John Kirchenbauer `[一作]` (University of Maryland), Tom Goldstein `[通讯]` (University of Maryland)

**通讯引用:** 13796 | [OpenAlex ID](https://openalex.org/A5060687985)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练自回归预训练模型在在线蒸馏目标下，使其一次性生成多令牌的多令牌预测模型，且不需要额外的校验器或复杂推理流水线。

**💡 创新点**

提出将教师模型的高概率多令牌序列作为奖励的在线蒸馏框架，并结合置信度自适应k选择，实现高效且无缝的多令牌推理。

**🔧 技术方法**

使用Transformer自回归模型、argmax/softmax教师评估、blocked attention、动态k随机化、KV缓存管理和置信度自适应采样等技术，并借鉴RL启发的奖励机制。

**📊 数据集**

在MetaMathQA上进行训练，评估于GSM8K、AIME25、GPQA、BBH、IFEVAL和CNN DailyMail等六个基准。

**📈 对比分析**

与单令牌（k=1）以及静态k值对照，ConfAdapt阈值90%时可获得3倍以上加速，GSM8K准确率仅下降≤3%（L3.1-8B）或≤7%（Qwen3-4B），更激进设置可达5×加速。

**⚠️ 局限性**

熵压缩效果不完全、仅在单一训练数据集上学习、未充分探索多步RL或更严格的熵惩罚，且模型规模与训练预算受限导致迁移性能有限。

---

## 594. On Computation and Reinforcement Learning

**arXiv ID:** 2602.05999 | [PDF](https://arxiv.org/pdf/2602.05999v1)

**作者:** Raj Ghugare `[一作]` (Princeton University), Benjamin Eysenbach `[通讯]` (Princeton University)

**通讯引用:** 1057 | [OpenAlex ID](https://openalex.org/A5035051008)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并验证了将强化学习策略视为可变计算量的模型：在固定参数规模下，增加计算步骤能提升学习性能和对长时序任务的泛化能力。

**💡 创新点**

创新点包括：①在理论层面证明存在任务，更多计算时间的策略能够实现更高奖励并更好泛化；②设计了最小化递归单元（IRU）实现可扩展计算；③提出了“计算价值”度量，用于评估额外计算带来的收益。

**🔧 技术方法**

技术手段包括：使用可调递归步骤的IRU网络（基于门控线性层），对比标准MLP与深度残差网络；通过可计算机器模型对策略进行时间约束分析；构造“计算价值”公式来量化计算收益。

**📊 数据集**

实验数据集涵盖约31个任务，主要分为三类：
- Boxpick（离散格子搬运/拼装）
- Lightsout（组合谜题）
- OGBench（离线目标到达任务）
每类包含多种训练/测试长度差异的子任务。

**📈 对比分析**

对比方法：相同算法但不同架构（IRU、MLP、ResNet）。结果显示：
- 增加递归步骤（如IRU‑5）显著提升最终奖励和样本效率。
- IRU在长时序测试上往往超越MLP和ResNet，尤其在Boxpick‑exact、Boxpick‑gen和Lightsout‑4×5任务。
- 尽管ResNet使用约2×计算量和5×参数，IRU在这些任务仍表现更好，表明计算效率优于单纯参数规模。

**⚠️ 局限性**

局限性包括：
- 计算步骤在所有状态固定，未实现自适应或基于难度的计算分配。
- 仅评估最小递归架构，未探讨Transformer或更复杂网络的可行性。
- 理论模型基于单盘图灵机的时间复杂度，未考虑空间或更贴近现代网络的布尔电路模型。

---

## 595. SwimBird: Eliciting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs

**arXiv ID:** 2602.06040 | [PDF](https://arxiv.org/pdf/2602.06040v1)

**作者:** Jintao Tong `[一作]` (Huazhong University of Science and Technology), Yixiong Zou `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5076460648)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种能够在文本、视觉和两者混合模式间切换的多模态大语言模型SwimBird；

**💡 创新点**

核心创新点是引入混合自回归框架，支持文本token和连续视觉嵌入的生成，并通过标签实现查询自适应的模式切换以及视觉思考的动态预算；

**🔧 技术方法**

技术包括：混合自回归生成、视觉嵌入的MSE重建损失、可学习的模式切换标签、基于分辨率的动态视觉令牌预算、以及多模式监督的SFT；

**📊 数据集**

使用了SwimBird‑SFT‑92K数据集，该数据集包含50k文本CoT、42k视觉或交互CoT样本，来源于OpenMMReasoner、ThinkMorph、Zebra‑CoT、MathCanvas等；

**📈 对比分析**

与文本推理基线（GPT‑4o、Qwen3‑VL‑8B‑Instruct）、潜在视觉推理模型（Monet、LVR、SkiLa）以及多模态代理模型（Pixel Reasoner、DeepEyes、Thyme）对比，SwimBird在细粒度视觉理解基准（V* Bench、HR‑Bench 4K/8K、MME‑RealWorld）以及通用VQA和多模态推理基准（MMStar、RealWorldQA、WeMath、DynaMath、MathVerse_MINI）均取得SOTA或显著提升；

**⚠️ 局限性**

局限性包括：对视觉预算上限的敏感性（过大导致冗余计算），以及对视觉嵌入质量的依赖，模型在极端文本推理任务中若未得到足够标注可能仍受视觉模式误导。

---

## 596. V-Retrver: Evidence-Driven Agentic Reasoning for Universal Multimodal Retrieval

**arXiv ID:** 2602.06034 | [PDF](https://arxiv.org/pdf/2602.06034v1)

**作者:** Dongyang Chen `[一作]`, Shichao Ka `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了V-Retrver框架，利用多模态交互式链式推理主动获取视觉证据，以提升通用多模态检索性能。

**💡 创新点**

将检索任务转化为agentic推理过程，引入可调用的视觉工具实现逐步检验和重排序，并通过三阶段课程式强化学习对工具使用进行优化。

**🔧 技术方法**

采用多模态交互式链式推理（MIER）、视觉工具（SELECT-IMAGE、ZOOM-IN）、基于Qwen2.5-VL-7B的LLM、分阶段监督+拒绝采样+证据对齐策略优化（EAPO）等技术。

**📊 数据集**

在M-BEIR的八个检索子任务上进行训练，评估包括CIRCO、GeneCIS、Visual Storytelling、Visual Dialog等未见数据集，以及对held‑out任务的零样本测试。

**📈 对比分析**

与CLIP、BLIP、UniIR、LamRA、U-MARVEL等基线对比，V-Retrver在M-BEIR平均Recall@10提升至69.7%，在未见数据集上MAP@5或R@1亦显著超越同类模型，表现出更强的泛化与细粒度检索能力。

**⚠️ 局限性**

模型依赖预训练视觉编码器冻结，工具调用成本和推理时长仍高；对极端视觉模糊或非图像输入的处理仍有限，且需要大量训练样本与复杂的RL调参。

---

## 597. AP-OOD: Attention Pooling for Out-of-Distribution Detection

**arXiv ID:** 2602.06031 | [PDF](https://arxiv.org/pdf/2602.06031v1)

**作者:** Claus Hofmann `[一作]` (Johannes Kepler University), Werner Zellinger `[通讯]` (Institute for Machine Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种半监督的自然语言文本OOD检测方法，利用注意力聚合代替传统平均池化，从token级别提取更丰富的分布信息；

**💡 创新点**

创新点在于将可学习的注意力权重嵌入马氏距离的方向分解，构建基于token的聚合机制，同时支持无监督与有辅助异常样本的无缝过渡；

**🔧 技术方法**

技术手段包括Transformer编码器产生token嵌入、可学习的查询向量进行注意力聚合、马氏距离和交叉熵损失的组合训练（含辅助样本惩罚），以及对不同头数和查询数的可扩展设计；

**📊 数据集**

使用了文本数据集XSUM、CNN/DailyMail、Newsroom、Reddit TIFU、Samsum用于摘要任务，WMT15英法翻译配合ParaCrawl作为辅助异常数据，以及MIMII‑DG音频数据集进行跨模态验证；

**📈 对比分析**

与Mahalanobis、KNN、Deep SVDD、perplexity、entropy等无监督基线以及binary logits、relative Mahalanobis、Deep SAD等监督基线对比，实验证明在无监督情形下FPR95显著下降（如XSUM从27.84%降至4.67%），在有辅助数据时AUROC进一步提升，整体实现了文本与音频领域的SOTA；

**⚠️ 局限性**

局限性包括对辅助异常样本分布的高度依赖，若其与推理时实际OOD分布差异过大，决策边界可能失效；在大规模LLM场景下的可扩展性与评估不足；以及对未见OOD类型的泛化能力仍需进一步验证。

---

## 598. Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory

**arXiv ID:** 2602.06025 | [PDF](https://arxiv.org/pdf/2602.06025v1)

**作者:** Haozhen Zhang `[一作]` (Nanyang Technological University), Wenya Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 4535 | [OpenAlex ID](https://openalex.org/A5101936536)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了BudgetMem框架，构建了可在运行时根据预算分层路由的模块化记忆提取管道。

**💡 创新点**

核心创新在于：①为每个模块定义Low/Mid/High三层预算，②使用轻量路由器通过强化学习学习预算决策，③比较三种实现策略（实现、推理、容量）并系统评估其性能–成本权衡。

**🔧 技术方法**

采用模块化管道、轻量路由器、PPO强化学习、成本归一化与奖励尺度对齐技术，并以LLaMA、Qwen等LLM作为记忆提取和回答模型。

**📊 数据集**

在LoCoMo、LongMemEval和HotpotQA三个长序列记忆与问答基准上进行实验。

**📈 对比分析**

与7种主流记忆增强基线（ReadAgent、MemoryBank、A-MEM、LangMem、Mem0、MemoryOS、LightMem）对比；在performance-first设置下BudgetMem在F1和LLM-judge上均优于所有基线；在不同预算权重下绘制性能–成本曲线，显著提升Pareto前沿。

**⚠️ 局限性**

限制：对LLM推理成本估计的准确性敏感；奖励尺度对齐需要手动调节；在历史短或检索规模小的场景下成本优势不明显；未针对多模态或动态更新历史进行评估。

---

## 599. Orthogonal Self-Attention

**arXiv ID:** 2602.05996 | [PDF](https://arxiv.org/pdf/2602.05996v1)

**作者:** Leo Zhang `[一作]` (University of Oxford), James Martens `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的注意力机制——正交自注意力（OSA），并在不使用跳跃连接和归一化层的 Transformer 结构中验证其可行性。

**💡 创新点**

创新点在于：①通过把注意力矩阵参数化为正交矩阵（利用矩阵指数映射奇异对称矩阵）来避免传统 Softmax 自注意力的秩坍塌和 Jacobian 条件数不佳的问题；②设计了基于低秩结构的高效指数计算方法，复杂度线性增长；③给出了保证输入-输出 Jacobian 条件数良好的初始化方案。

**🔧 技术方法**

主要技术包括：矩阵指数（exponential）映射、低秩奇异值分解、QR 或 Newton‑Schulz 正交化、Stiefel 随机初始化、以及对正交注意力的核分析与梯度传播研究。

**📊 数据集**

实验使用 MNIST 数据集进行图像分类任务，构造了 ViT 基线模型，并在其基础上替换 SSA 为 OSA，去掉跳跃连接和层归一化。

**📈 对比分析**

对比方法：将 OSA‑Transformer 与 ViT 基线、ViT 去掉跳跃连接、以及 ViT 去掉跳跃连接与归一化层的组合进行对比。实验结果表明，OSA‑Transformer 在训练速度和测试性能上可与基线 ViT 相当，甚至在去掉跳跃连接的 ViT 上表现更好；QR 正交化略优于 Newton‑Schulz。

**⚠️ 局限性**

局限性：实验仅在简单的 MNIST 数据集上验证，未覆盖更复杂的图像或序列任务；对 OSA 在大规模模型或因果结构中的适用性未作深入探究；此外，计算正交化的数值稳定性和超参数（如 α、K）对性能的影响仍需进一步系统评估。

---

## 600. Diamond Maps: Efficient Reward Alignment via Stochastic Flow Maps

**arXiv ID:** 2602.05993 | [PDF](https://arxiv.org/pdf/2602.05993v1)

**作者:** Peter Holderrieth `[一作]`, Max Simchowitz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Diamond Maps——一种随机流映射模型，用于在推理时高效、准确地对任意奖励函数进行对齐；并给出了两种实现方案：Posterior Diamond Maps（从 GLASS Flow 蒸馏得到的后验采样器）和 Weighted Diamond Maps（在标准流映射上通过重新噪声化实现随机性）。

**💡 创新点**

创新点：
- 将适应性设计嵌入生成模型本身，避免单纯的后训练微调；
- 利用流映射将多步模拟压缩为一步，同时保留随机性以精确估计价值函数；
- 通过 GLASS Flow 直接得到后验动力学并蒸馏为一阶采样器；
- 引入 Early‑Stop DDPM 采样方案，显著降低迭代误差；
- 通过重新噪声化与局部奖励权重实现任意流映射的随机化，从而得到一致的价值梯度估计。

**🔧 技术方法**

技术栈：
- 流匹配（Flow Matching）与扩散模型；
- 流映射（Flow Maps）与 GLASS Flow；
- 蒸馏、Lagrangian / Eulerian 损失；
- 价值函数估计、蒙特卡罗采样、序贯蒙特卡罗（SMC）与搜索；
- 重噪声化、重采样、加权估计与分数校正；
- 近似或完整的梯度引导（guidance）与奖励对齐。

**📊 数据集**

使用的数据集与评测工具：
- CIFAR‑10、CelebA（小规模实验）
- SANA‑Sprint 0.6B（高分辨率 T2I）
- GenEval 评价基准（评估真实生成质量）
- CLIP、ImageReward、HPSv2、PickScore 等预训练人类偏好奖励模型。

**📈 对比分析**

与现有方法对比：
- 与 Best‑of‑N、Reward‑based Noise Optimization、Prompt Optimization、ReNO 等推理时对齐方法对比；
- Weighted Diamond Maps 在 NFEs 低至 1‑64 时实现更优的 Pareto 前沿，性能接近 GPT‑4o；
- 在 4‑粒子/4‑步的设置下显著优于 Baseline，速度更快、内存消耗更低；
- Posterior Diamond Maps 在后验采样、价值函数估计与搜索中表现出更高的样本效率与鲁棒性。

**⚠️ 局限性**

局限性：
- 需要先行训练/蒸馏 GLASS Flow 或标准流映射，训练成本相对较高；
- Weighted Diamond Maps 在推理时需多次重采样，导致计算开销随 ESS 下降；
- 本文实验主要在小规模模型上验证，尚未在大规模扩散/流模型上进行全面评估；
- 对极大奖励尺度的稳健性虽提升，但在极端奖励场景仍可能出现偏差；
- 需要可靠的奖励函数，若奖励定义不准确仍可能导致对齐失败。

---

## 601. Thinking with Geometry: Active Geometry Integration for Spatial Reasoning

**arXiv ID:** 2602.06037 | [PDF](https://arxiv.org/pdf/2602.06037v1)

**作者:** Haoyuan Li `[一作]` (Shenzhen campus of Sun Yet-sen University), Xiaodan Liang `[通讯]` (Yinwang Intelligent Technology Company Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过主动几何集成框架 GeoThinker，提升多模态大语言模型的空间推理能力。

**💡 创新点**

创新点是将被动融合转为主动感知，利用 Spatial‑Grounded Fusion 与 Importance Gating 实现仅在需要时提取任务相关几何信息。

**🔧 技术方法**

采用 VGGT 3D 视觉几何编码器、Qwen‑VL 视觉‑语言基础模型、frame‑strict cross‑attention、Importance Gating、global scaling 等技术。

**📊 数据集**

训练与评估使用 VSI‑Bench、MMSI‑Bench、MindCube、VideoSpatial、SITE、CV‑Bench 等空间推理基准，外加 RoboRefer、ReCogDrive、NAVSIM 等下游任务。

**📈 对比分析**

与 VG‑LLM、Cambrian‑S 等基线对比，GeoThinker 在 VSI‑Bench 取得 72.6 分的 SOTA，Debiased 128‑frame 仍达 68.1，且在 RoboRefer、ReCogDrive 上分别提升 1.66% 与 2.0 点。

**⚠️ 局限性**

限制在于需要额外的几何编码器与跨模态注意力，计算成本上升；对完全未知场景的泛化仍有待验证。

---

## 602. Context Forcing: Consistent Autoregressive Video Generation with Long Context

**arXiv ID:** 2602.06028 | [PDF](https://arxiv.org/pdf/2602.06028v1)

**作者:** Shuo Chen `[一作]` (University of California, Merced), Wenhu Chen `[通讯]` (University of Waterloo)

**通讯引用:** 4847 | [OpenAlex ID](https://openalex.org/A5103103242)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Context Forcing 框架，通过长上下文教师监督长上下文学生，实现长时长视频的自回归生成，解决学生-教师匹配导致的遗忘漂移问题。

**💡 创新点**

创新点包括：① 设计长上下文教师并对学生进行上下文分布匹配蒸馏；② 引入慢-快记忆 KV 缓存架构降低视觉冗余；③ 采用错误回收微调提升教师对失真上下文的鲁棒性；④ 动态滚动窗口训练策略。

**🔧 技术方法**

使用技术：DiT 语义自回归模型、Contextual Distribution Matching Distillation (CDMD)、KV 缓存 + Slow-Fast Memory、Bounded Positional Encoding、Error-Recycling Fine‑Tuning (ERFT)。

**📊 数据集**

训练和评估数据集包括：40k 条 10+ 秒长视频（Sekai、Ultravideo 过滤后）、VidProM 用于 Stage 1/2 训练、MovieGenBench 随机 100 条提示用于评测、VBench 用于一致性指标评估。

**📈 对比分析**

在 5 秒和 60 秒视频上与 bidirectional、autogressive、LongLive、Framepack 等基线对比，采用 DINOv2、CLIP‑F/T 及 VBench 评价一致性。结果显示，Context Forcing 在 20+ 秒上下文下保持最高的背景/主体一致性，平均提升 2–10 倍的上下文长度，且在 60 秒生成中显著降低漂移与身份切换。

**⚠️ 局限性**

局限性：记忆压缩仍存在信息稀疏问题，无法进一步提升信息密度；模型规模和生成速度受限；对极长序列的泛化、多样性和自适应记忆机制仍需研究。

---

## 603. Optimism Stabilizes Thompson Sampling for Adaptive Inference

**arXiv ID:** 2602.06014 | [PDF](https://arxiv.org/pdf/2602.06014v1)

**作者:** Shunxing Yan `[一作]` (Peking University), Han Zhong `[通讯]` (Peking University)

**通讯引用:** 20531 | [OpenAlex ID](https://openalex.org/A5018465248)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

研究了在多臂高斯赌博机中引入乐观性（通过方差膨胀或均值奖金）后，Thompson Sampling 的稳定性与可行的自适应推断。

**💡 创新点**

创新点在于证明两种乐观版本（方差膨胀和均值奖金）在任意 K≥2 的多臂设置下均能保证稳定性，并在多最优臂情形下实现均匀分配；同时给出自适应置信区间的渐近有效性，并量化了稳定化所带来的轻微 regret 费用。

**🔧 技术方法**

主要技术包括：1）利用赢家映射和负反馈性质构造 Lyapunov 函数；2）对后验采样的概率分布进行误差扰动分析；3）通过几何等待时间和稀有事件控制得到子优臂的对数量级抽样；4）利用马尔可夫中心极限定理和 Slutsky 定理推导自适应样本均值的渐近正态性。

**📊 数据集**

本研究为理论工作，未使用真实数据集；实验验证部分以合成高斯多臂赌博机为基础进行仿真。

**📈 对比分析**

与传统的 vanilla TS 以及 UCB、Bayes‑UCB 等算法进行比较。结果显示：在实现稳定性后，乐观 TS 的累积 regret 与 vanilla TS 仅相差一个 loglog(T) 或 √loglog(T) 的倍数；同时可获得符合 Wald 理论的置信区间覆盖率，表明推断性能显著提升。

**⚠️ 局限性**

局限性包括：1）仅针对无偏方差的高斯噪声模型；2）只讨论固定时间窗口下的 asymptotic 结果，缺乏有限样本的收敛速率分析；3) 未考虑更一般的情境（如线性/马尔可夫决策过程）或非高斯奖励分布。

---

## 604. AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions

**arXiv ID:** 2602.06008 | [PDF](https://arxiv.org/pdf/2602.06008v1)

**作者:** Xianyang Liu `[一作]`, Dawn Song `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了 AgenticPay 基准，覆盖 110+ 任务，模拟多智能体买卖自然语言谈判场景。

**💡 创新点**

创新点包括：①把谈判建模为语言游戏并加入私有约束与多产品；②构建可扩展的任务层级；③提出基于可行性、效率与福利的评估指标。

**🔧 技术方法**

使用 LLM 策略代理、对话解析器、vLLM/SGLang 部署以及结构化动作抽取技术。

**📊 数据集**

基于 10 个真实商业场景（消费、服务、采购、金融资产）生成的产品描述与约束数据。

**📈 对比分析**

采用统一推理协议对专有模型（Claude Opus 4.5、Gemini‑3‑Flash、GPT‑5.2）和开源模型（Qwen3‑14B、Llama‑3.1‑8B）进行对比，指标为 GlobalScore、BuyerScore、SellerScore；专有模型取得最高分且零超时，开源模型表现明显逊色，且出现溢价率和超时。

**⚠️ 局限性**

主要局限在于长轮次战略推理不足、买方表现持续弱于卖方、开源模型在多方并行时易违反约束，并且基准仍基于模拟数据，缺乏真实交易验证。

---

## 605. Visuo-Tactile World Models

**arXiv ID:** 2602.06001 | [PDF](https://arxiv.org/pdf/2602.06001v1)

**作者:** Carolina Higuera `[一作]` (University of Washington), Franziska Meier `[通讯]` (Meta)

**通讯引用:** 1611 | [OpenAlex ID](https://openalex.org/A5071976030)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种多任务视觉-触觉世界模型（VT‑WM），能够利用触觉信息补充视觉，对机器人-物体交互进行物理真实的预测，并在实时规划中实现零样本高成功率。

**💡 创新点**

创新点在于：①首次将触觉与视觉联合用于多任务世界模型；②通过触觉实现物体永久性与运动物理的一致性；③在零样本规划中显著提升了接触丰富任务的成功率；④展示了模型在少量示例新任务中的快速迁移能力。

**🔧 技术方法**

采用的技术包括：Cosmos 与 Sparsh‑X 预训练编码器将 RGB 与触觉图像映射至潜在空间；12 层 Transformer 进行时空自注意与动作跨注意的混合推理；结合教师强迫与采样损失的自回归训练；以及基于 CEM 的目标条件规划。

**📊 数据集**

使用了包含 5 类接触丰富操作（如推水果、擦布、堆叠方块等）的真实机器人演示数据，并在此基础上加入 20 条新任务演示进行快速适配。

**📈 对比分析**

与单模视觉世界模型（V‑WM）对比，VT‑WM 在对象永存性上平均降低 33% 的 Fréchet 距离，在因果符合性上降低 29%；在零样本规划中对接触任务的成功率提升高达 35%；在新任务少样本迁移中成功率达 77%。

**⚠️ 局限性**

局限性包括：模型仍需较大计算资源；对极端遮挡和快速运动的触觉响应仍有限；未针对长时间连续任务进行深入评估；在某些任务如“scribble with marker”中表现不如预期。

---

## 606. Towards uncertainty quantification of a model for cancer-on-chip experiments

**arXiv ID:** 2602.06018 | [PDF](https://arxiv.org/pdf/2602.06018v1)

**作者:** Silvia Bertoluzza `[一作]` (Consiglio Nazionale delle Ricerche), Pietro Zanotti `[通讯]` (Università degli Studi di Milano)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了一维癌症-白细胞相互作用的化学趋化Keller‑Segel模型，并用Hybridizable Discontinuous Galerkin（HDG）方法求解。

**💡 创新点**

创新点在于将基于合成实验数据的贝叶斯后验参数化与前向不确定性量化结合，并采用稀疏格代理加速整个UQ流程。

**🔧 技术方法**

使用的技术包括HDG求解、Sobol/Morris全局敏感度分析、贝叶斯逆向（Gaussian近似与slice采样MCMC）以及稀疏格插值代理。

**📊 数据集**

实验数据采用从HDG求解得到的中心质量轨迹加入高斯噪声的合成数据。

**📈 对比分析**

通过与先验分布比较，后验分布在参数和QoI（化学趋化物质总量）上均实现显著不确定度减小，数值结果误差保持在1%以内。

**⚠️ 局限性**

局限性包括模型仅为一维简化，未考虑真实细胞动力学细节，且采样效率受高维参数空间限制。

---

## 607. Pseudo-Invertible Neural Networks

**arXiv ID:** 2602.06042 | [PDF](https://arxiv.org/pdf/2602.06042v1)

**作者:** Yamit Ehrlich `[一作]` (Technion), Assaf Shocher `[通讯]` (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种非线性伪逆（Natural Non‑Linear Pseudo‑Inverse）的理论，并在此基础上设计了可实现唯一伪逆的Surjective Pseudo‑Invertible Neural Network（SPNN），进一步提出非线性后投影（Non‑Linear Back‑Projection, NLBP）用于零样本反演；

**💡 创新点**

①将 Moore‑Penrose 伪逆概念推广到非线性映射，提出通过双射完成（bijective completion）实现唯一伪逆；②构造维度缩减的 SPNN 架构，在保证首两条 Penrose 恒等式的前提下通过学习辅助网络实现唯一最小范数伪逆；③将 SPNN 与 NLBP 结合，提供在非线性退化（如语义分类、图像压缩）下的零样本恢复与属性控制。

**🔧 技术方法**

利用双射完成、最小范数约束、Affine Coupling 结构、可学习正交旋转（Cayley 变换）、辅助网络学习 null‑space 组件，以及在扩散模型采样循环中嵌入的 NLBP 迭代投影。

**📊 数据集**

CelebA‑HQ（256×256）数据集，使用其 40 个二进制属性标签作为非线性退化的目标。

**📈 对比分析**

与随机/最小范数辅助网络、传统非线性后投影等对照实验，结果显示 SPNN+NLBP 在属性重建二值一致率达 92.3%，多属性编辑和语义恢复效果优于消融基线；在生成样本的多样性与一致性上均表现出色。

**⚠️ 局限性**

（1）辅助网络表达能力有限时，伪逆可能产生不真实的前像；（2）仅适用于可达（surjective）退化，无法处理超出范围的观测；（3）依赖训练数据分布，可能带来偏差。

---

## 608. Predicting Camera Pose from Perspective Descriptions for Spatial Reasoning

**arXiv ID:** 2602.06041 | [PDF](https://arxiv.org/pdf/2602.06041v1)

**作者:** Xuejun Zhang `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8423 | [OpenAlex ID](https://openalex.org/A5103178893)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CamCue 框架，实现从多视角图像与自然语言视角描述中预测目标相机位姿，并可合成目标视角图像以辅助回答问题。

**💡 创新点**

创新点在于：① 将相机位姿显式注入图像特征作为几何锚点；② 用交叉注意力学习自然语言描述到目标位姿的映射；③ 在预测位姿的基础上进行姿态控制的视角合成，提升视角一致性和推理准确率。

**🔧 技术方法**

使用 Plücker 码器生成每像素位姿纹理，patch 对齐融合图像与位姿特征；交叉注意力预测目标相机矩阵；LoRA 微调多模态语言模型；可选的 LVSM 图像解码器用于生成目标视角图像。

**📊 数据集**

主要数据集为自研的 CamCue‑Data（27,668 训练 / 508 测试），涵盖多视角图像、相机位姿、以及多种自然语言视角描述；同时在 MindCube Tiny 与 MMSI 等公开多图像推理基准上进行评估。

**📈 对比分析**

在 CamCue‑Data 上整体准确率提升约 9.06%；目标位姿预测在旋转误差 <20° 时达到 91.5%（合成描述）或 100%（人工描述），平移误差 <0.5 时为 92.9%/95.1%；与 MindJourney 等基线相比，CamCue 在视角敏感任务（可见性、距离排序、相对关系）上表现更佳，并将单例推理时延从 256.6 秒压缩至 1.45 秒，显著提升实时交互能力。

**⚠️ 局限性**

局限性：目前仅针对视角转移式问答，未覆盖完整的执行规划任务；当合成图像产生噪声或错位时可能误导推理，需要进一步构建鲁棒性评估与选择机制。

---

## 609. Curiosity is Knowledge: Self-Consistent Learning and No-Regret Optimization with Active Inference

**arXiv ID:** 2602.06029 | [PDF](https://arxiv.org/pdf/2602.06029v1)

**作者:** Yingke Li `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1866 | [OpenAlex ID](https://openalex.org/A5019603699)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过理论分析证明，在主动推理（Active Inference）框架下，只要好奇系数足够大，既能保证后验一致性（学习真实模型），又能保证累计回报的无上界（无退化优化）

**💡 创新点**

创新点在于将好奇系数视为统一机制，既调节信息探索，又保证收敛与无退化；并给出了一阶后验一致性与回报界限的具体定理和取样复杂度、信息增益、光滑性等因素的依赖关系

**🔧 技术方法**

主要技术包括：期望自由能（Expected Free Energy）最小化、信息增益（mutual information）与期望能量（pragmatic cost）的分解、贝叶斯实验设计、贝叶斯优化、Gaussian Process 代理、回报函数的光滑性假设、以及利用信息增益上界的累积回报分析

**📊 数据集**

使用的数据集包括：离散状态沙盒（Synthetic Discrete Sandbox）、一维GP Bandit、二维环境监测 plume 现场数据（真实传感器观测）以及电网能源资源分配的40维输入与4维输出的实际案例

**📈 对比分析**

通过实验验证理论，展示好奇系数对后验收敛速率和累积回报的影响，比较不同好奇度、信息辨识度、先验不确定性与启发式误差的组合，实验结果表明足够好奇可实现最佳收敛与最小回报，且在不同任务中表现一致，但未给出统一数值基准

**⚠️ 局限性**

局限性包括：理论界限较为保守、对好奇系数的调度未给出自适应方法、需要满足可辨识性与光滑性等严格假设、对模型错配、非平稳性与部分可观测环境可能失效、以及在真实任务中对启发式对齐的控制具有挑战

---

## 610. Learning Event-Based Shooter Models from Virtual Reality Experiments

**arXiv ID:** 2602.06023 | [PDF](https://arxiv.org/pdf/2602.06023v1)

**作者:** Christopher A. McClurg `[一作]` (Pennsylvania State University), Alan R. Wagner `[通讯]` (Pennsylvania State University)

**通讯引用:** 2645 | [OpenAlex ID](https://openalex.org/A5045198429)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了基于VR实验数据的离散事件模拟器，用于大规模评估学校安全干预措施并支持强化学习策略开发。

**💡 创新点**

创新点在于将参与者行为数据转化为可学习的射手转移模型（基于图神经网络）与事件生成模型（分层截断正态采样），并结合机器人烟雾扩散效应，形成可插拔的中等保真度仿真框架；该框架使得海量样本的实验和策略迭代成为可能。

**🔧 技术方法**

使用的技术包括图神经网络（GraphSAGE + 分类器）进行射手区域转移预测，分层截断正态采样实现时间/射击/受害者事件的随机生成，基于烟雾扩散的指数衰减模型调节机器人效果，以及双重深度Q网络（DDQN）进行机器人控制策略学习。

**📊 数据集**

使用了来自两次虚拟现实实验的210个5分钟射手角色扮演数据（包含无机器人和机器人喷烟两种条件），共计210条轨迹，涵盖射手位置、射击次数、受害者数量等信息。

**📈 对比分析**

与传统的随机、最近目标、常速等启发式基线相比，GNN射手转移模型在射手区域预测准确率显著更高；事件生成模型在均值、方差、空间JSD以及时间相关性等指标上均匹配或接近实验数据；机器人干预策略评估显示机器人向射手移动能显著降低受害者数，RL学习的策略在受害者减少率上达到约38%，与手工策略相当。

**⚠️ 局限性**

局限性包括：仅在单一仿真环境（康斯布尔高中）收集数据，缺乏对不同建筑布局、占用率、时间等情境变量的系统研究；机器人效果仅考虑喷烟干预，未覆盖其他干预方式；所学策略尚未在VR实验中验证，需要进一步闭环评估。

---

## 611. Mechanisms of AI Protein Folding in ESMFold

**arXiv ID:** 2602.06020 | [PDF](https://arxiv.org/pdf/2602.06020v1)

**作者:** Kevin Lu `[一作]` (Northeastern University), Chris Wendler `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对 ESMFold 的折叠 trunk 进行因果性、可解释性分析，利用 β‑hairpin 这一二级结构作为案例，揭示模型在不同块中如何将序列信息转化为配对特征并最终生成三维结构。

**💡 创新点**

首次把折叠过程划分为两个阶段：早期块将化学信息（如电荷）从序列注入配对表示；后期块将配对表示转化为几何信息（距离、接触），并通过可操作的“charge 方向”和“距离方向”证明这些信息对结构决定具有因果影响。

**🔧 技术方法**

主要技术包括：激活补丁（activation patching）和向量 steering、线性 probe 预测配对距离与电荷、注意力偏置分析、尺度变换实验、以及对不同块的 ablation 探测。

**📊 数据集**

使用 95 个 α‑螺旋目标蛋白（从多家族中挑选）与约 80,000 个 PDB 中提取的 β‑hairpin，形成约 5,000 次补丁实验；另外用 600 条蛋白序列做线性 probe 训练与评估。

**📈 对比分析**

对比未补丁与补丁、不同模块（encoder/trunk/structure）以及不同补丁位置，证明在早期块补丁 sequence 能产生约 40% 的 β‑hairpin 成功率，在后期块补丁 pairwise 能产生约 20% 成功率；通过尺度变换验证 pairwise 表示的几何作用，展示显著的因果效果。

**⚠️ 局限性**

局限性在于只针对单一二级结构（β‑hairpin）和单一模型（ESMFold），实验仅覆盖短蛋白，未验证对更复杂折叠模式或其他折叠模型的通用性；分析未直接提升预测性能，仅揭示内部机制。

---

## 612. MambaVF: State Space Model for Efficient Video Fusion

**arXiv ID:** 2602.06017 | [PDF](https://arxiv.org/pdf/2602.06017v1)

**作者:** Zixiang Zhao `[一作]` (ETH Zurich), Konrad Schindler `[通讯]` (ETH Zurich)

**通讯引用:** 23337 | [OpenAlex ID](https://openalex.org/A5005404030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MambaVF，一种基于状态空间模型的无光流视频融合框架。

**💡 创新点**

创新点在于用状态空间模型代替光流估计，结合空间-时间双向扫描实现线性复杂度的跨模态融合。

**🔧 技术方法**

采用 Mamba 结构的 VSS 块、STB 扫描、双流编码器和 3D 解码器，以及 Sobel 梯度、SSIM 等多项损失。

**📊 数据集**

在 VF‑Bench 四大任务（多曝光、多焦点、红外可见、医学）以及公开的多模视频数据集上进行实验。

**📈 对比分析**

与 UniVF、TemCoCo 等 SOTA 比较，MambaVF 在所有任务上获得相当或更优的 VIF/SSIM 等指标，同时参数仅 7.75%、FLOPs 11.21%，速度提升 2.1×。

**⚠️ 局限性**

局限在于单模源时性能下降，对极端动态场景下的光流缺失仍可能产生残留模糊，且仅在短时序内验证，长序列效果待进一步评估。

---

## 613. GenArena: How Can We Achieve Human-Aligned Evaluation for Visual Generation Tasks?

**arXiv ID:** 2602.06013 | [PDF](https://arxiv.org/pdf/2602.06013v1)

**作者:** Ruihang Li `[一作]` (University of Science and Technology of China), Jiaqi Wang `[通讯]` (Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 GenArena，一种基于 VLM 的双人对决评估框架，改用配对比较而非传统的绝对分数进行视觉生成模型的自动评估，并通过 Elo 排名构建可复现的排行榜；

**💡 创新点**

创新点在于（1）系统性揭示点分法在自一致性和人类对齐上的缺陷；（2）引入双人对决与 Elo 评分的组合，显著提升评估的稳定性与可信度；（3）证明开源 VLM 在配对模式下即可超越商用大型模型，无需额外微调；

**🔧 技术方法**

主要技术包括：Vision‑Language Model（VLM）作为评判者；双向一致性检查与强制二选一的评判策略；Elo 评分系统（通过 Bradley‑Terry 模型实现）来聚合对决结果；与点分法对比的 Krippendorff α 自一致性度量；

**📊 数据集**

使用的评测数据集包括：GenAI‑Bench（图像生成）、EditScore‑Bench、GEdit‑Bench（图像编辑）、VideoGen‑RewardBench（视频生成）、以及自建的 6,086 条多参考编辑与推理任务的 Prompt 集；

**📈 对比分析**

与传统基于 GPT‑4.1 等的点分评估相比，配对+Elo 方案提升了约 20% 的判定准确率，Spearman 相关系数从 0.36 提升至 0.86；在各任务维度（Basic、Reasoning、Multi‑Ref）上，开源 Qwen3‑VL‑32B‑FP8 评判者的排名与 LMArena 人类榜单高度吻合；

**⚠️ 局限性**

局限性包括：评判者仍受 VLM 训练数据偏见影响，可能在多文化或敏感场景下产生偏差；对 tie（平局）的处理需强制二选一，可能导致误判；评估框架在极端复杂或超大规模模型时的计算成本和对决规模仍是挑战；

---

## 614. Speech Emotion Recognition Leveraging OpenAI's Whisper Representations and Attentive Pooling Methods

**arXiv ID:** 2602.06000 | [PDF](https://arxiv.org/pdf/2602.06000v1)

**作者:** Ali Shendabadi `[一作]` (University of Tehran), Mahmoud Bijankhan `[通讯]` (University of Tehran)

**通讯引用:** 450 | [OpenAlex ID](https://openalex.org/A5103104653)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对Whisper ASR编码器提取的高维特征，提出两种基于注意力的池化方法（多头注意力平均池化与多头QKV池化），并在Persian ShEMO和English IEMOCAP上实现语音情感识别。

**💡 创新点**

①利用全局平均池化作为查询，创新性地将QKV注意力引入池化过程；②首次在低资源语言Persian中使用Whisper作为轻量级特征提取器；③系统性评估Whisper不同编码层的情感表征能力。

**🔧 技术方法**

Whisper预训练模型（Tiny/Small）冻结编码层，基于Transformer的多头注意力池化，投影层降维，再加全连接分类器；训练使用AdamW + cosine学习率调度。

**📊 数据集**

Persian语料库ShEMO（3h 3,000句）与英语IEMOCAP（≈12h 2,793句）两个标准情感语音数据集。

**📈 对比分析**

与均值池化基线以及其他主流基线（Wav2Vec2.0、HuBERT、Whisper Large）对比。QKV池化在ShEMO上实现83.07% UA（SOTA水平），在IEMOCAP上达到72.96% UA，且模型参数仅88M，显著低于HuBERT X-Large等大型模型。

**⚠️ 局限性**

①投影与分类层权重大、易过拟合，受限于小样本；②未结合Whisper解码器文本信息或多模态特征；③在IEMOCAP上仍低于大型模型的准确率，需进一步提升性能。

---

## 615. VisRefiner: Learning from Visual Differences for Screenshot-to-Code Generation

**arXiv ID:** 2602.05998 | [PDF](https://arxiv.org/pdf/2602.05998v1)

**作者:** Jie Deng `[一作]` (Institute of Software, Chinese Academy of Sciences), Libo Zhang `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VisRefiner 框架，使多模态大语言模型能够通过学习截图与渲染结果之间的视觉差异来改进前端代码生成，并实现自我改进。

**💡 创新点**

创新点：将视觉差异与代码编辑对齐，构建差异对齐监督（SFT）以及基于视觉奖励的 GRPO 强化学习，实现视觉反馈在训练中的直接使用，提升模型的自我纠错能力。

**🔧 技术方法**

使用的技术：差异对齐监督、GRPO 强化学习、自渲染对比、CLIP 相似度奖励、视觉差异标注、自动生成的视觉扰动规则等。

**📊 数据集**

使用的数据集：VisDiffUI（差异对齐的 UI 代码‑截图对）、Design2Code、Design2Code‑HARD、VisDiffUI‑Test。

**📈 对比分析**

比较方法与性能：在 Design2Code、Design2Code‑HARD 与 VisDiffUI‑Test 上与 GPT‑4o、Claude、Qwen 等模型进行对比，单步平均得分接近 GPT‑4o；在自我改进（one‑step refinement）上保持持续提升，表现稳定且优于大多数开源基线。

**⚠️ 局限性**

limitation：对复杂交互与动态样式的支持有限；多轮自我改进仍可能出现波动；需要大量标注样本与算力，模型规模受限。

---

## 616. DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs

**arXiv ID:** 2602.05992 | [PDF](https://arxiv.org/pdf/2602.05992v1)

**作者:** Lizhuo Luo `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2803 | [OpenAlex ID](https://openalex.org/A5028270700)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于动态滑动块的无训练调度方法，改善扩散式大语言模型的并行解码。

**💡 创新点**

创新点在于把块大小与语义难度动态匹配，并结合专门的KV缓存（DSB Cache）实现高效稳定的解码。

**🔧 技术方法**

采用动态滑动块调度、无训练的自适应阈值解码、前缀窗口更新的KV缓存技术。

**📊 数据集**

使用 GSM8K、MATH、HumanEval、MBPP、BBH 等常见推理基准进行评测。

**📈 对比分析**

与传统固定块调度、基于阈值的并行解码及 Dual Cache 进行对比，实验显示 DSB 与 DSB Cache 在多模型、多任务上都显著提升准确率和 tokens‑per‑second。

**⚠️ 局限性**

仍受块长度、最小前缀窗口等超参数影响，对某些模型（如 Dream 系列）的提升不如预期，且在极大块或极短输入时可能导致语义偏差。

---

