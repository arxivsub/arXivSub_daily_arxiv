# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-22 | 今日论文总数: 444

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Beyond Affinity: A Benchmark of 1D, 2D, and 3D Methods Reveals Critical Trade-offs in Structure-Based Drug Design

**arXiv ID:** 2601.14283 | [PDF](https://arxiv.org/pdf/2601.14283v1)

**作者:** Kangyu Zheng `[一作]` (Rensselaer Polytechnic Institute), Zhiding Liang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个跨算法基准，评估了15种结构基础药物设计（1D、2D、3D）模型在10个靶蛋白上的分子生成、对接评分、配体姿态、化学有效性及药理属性。

**💡 创新点**

创新点在于首次实现不同算法范式（搜索、生成、强化学习）的横向比较，并将对接评分与姿态质量、内部应变能等物理评估指标纳入综合评价，揭示3D模型虽优先获取高亲和力但常伴随高应变和有效性波动。

**🔧 技术方法**

使用AutoDock Vina进行对接评分，PoseBuster 与 PoseCheck 对姿态进行化学一致性与物理可行性评估；同时采用 QED、SA、LogP、SCScore、RAscore 等启发式属性或acles；对生成多样性、唯一性、有效性进行评估。

**📊 数据集**

数据集包括10个多样化靶蛋白（5个来自 CrossDocking，5个来自其他公开数据集），使用TDC、MOSES 等公开分子库进行模型预训练和对接输入。

**📈 对比分析**

通过多维度评分（对接得分、姿态 RMSD、应变能、化学有效性、生成多样性、药理指标）对15个模型进行排名；3D模型 Pocket2Mol 在对接得分上领先，但 PoseBuster 通过率和 RMSD 低；1D 模型在有效性与生成速度上最稳健；2D 模型在平衡性方面表现最佳。

**⚠️ 局限性**

局限性包括仅采用 AutoDock Vina 作为对接oracle，未覆盖更先进的 DiffDock 等；靶蛋白数量有限，未充分验证跨靶泛化；3D 模型对计算资源需求高、易出现生成失败；基准未考虑真实药物开发流程中的合成可行性评估与实验验证。

---

## 2. Opening the Black Box: A Survey on the Mechanisms of Multi-Step Reasoning in Large Language Models

**arXiv ID:** 2601.14270 | [PDF](https://arxiv.org/pdf/2601.14270v1)

**作者:** Liangming Pan `[一作]` (Peking University), Fengbin Zhu `[通讯]` (National University of Singapore)

**通讯引用:** 380 | [OpenAlex ID](https://openalex.org/A5029052244)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对大型语言模型（LLM）多步推理的内部机制进行系统综述，构建了由隐式推理、显式推理以及七个互联研究问题组成的认知框架，并提出五个未来研究方向。

**💡 创新点**

创新点在于：①首次将多步推理的隐式与显式机制统一到一个七问题的框架中；②聚焦机制层面而非仅提升性能；③通过对已有实验技术（因果探针、表征分析等）的整合，梳理了推理的层级化、序列化与并行化机制；④明确指出了“grokking”与“shortcut”现象对推理发展的影响；⑤为后续研究提供了白盒评估、隐式思维、因果验证等具体方向。

**🔧 技术方法**

使用的技术主要包括：因果探针、机械追踪、表征分析、PatchScope 与 back‑patching、logit 视图、神经元级别的 logit flow、注意力头分析、神经元激活追踪、干预（weight‑editing、CREME、back‑attention）、可解释的评估指标（如 MUI）等。

**📊 数据集**

参考的数据集与任务包括：GPT‑2、LLaMA、LLaMA‑2 等大模型在符号推理、数学推理、逻辑推理、计划任务、BIG‑Bench Hard、MMLU、数学、符号逻辑任务、知识问答、常识推理等公开基准和人工构造的符号任务。

**📈 对比分析**

本文并未进行新的实验，而是对已有研究的实验结果进行对比与归纳：在隐式推理方面，展示层级化、特定层功能化、短板深度等；在显式推理方面，阐明 CoT 在数学/符号任务中的显著提升以及在知识/常识任务中的有限或负面效果；对比不同方法在提升推理精度、降低样本复杂度、提升鲁棒性等方面的表现；整体而言，综述显示 CoT 能显著提升有序推理任务的准确率，而隐式推理受层深与短路的限制。

**⚠️ 局限性**

限制主要体现在：①仅聚焦基于文本的 transformer LLM，未覆盖多模态 LLM、扩散模型等；②综述依赖公开文献与实验，缺乏新的因果验证与大规模真实场景实验；③未系统讨论模型规模、预训练任务与推理机制之间的细粒度关联；④未给出统一的评估框架或标准，仍依赖单一指标。

---

## 3. CORVUS: Red-Teaming Hallucination Detectors via Internal Signal Camouflage in Large Language Models

**arXiv ID:** 2601.14310 | [PDF](https://arxiv.org/pdf/2601.14310v1)

**作者:** Nay Myat Min `[一作]` (Singapore Management University), Jun Sun `[通讯]` (Singapore Management University)

**通讯引用:** 10124 | [OpenAlex ID](https://openalex.org/A5100429004)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级LoRA红队方法CORVUS，用于在单通道检测器下隐藏大语言模型的内部迹象，以逃避幻觉检测。

**💡 创新点**

创新点在于利用嵌入空间FGSM对注意力进行攻击，同时在训练时优化隐藏日志体积、注意力对角性和令牌熵三种内部指标，实现对内部特征的协同伪装。

**🔧 技术方法**

技术包括教师强制回放、LoRA微调、FGSM嵌入扰动、隐藏状态日志体积、注意力对角性和令牌熵三种内部指标。

**📊 数据集**

数据集使用了1,000条Out-of-distribution Alpaca指令进行训练，并在FAVA-Annotation上进行零样本评估。

**📈 对比分析**

对比四大开源模型（Llama-2、Vicuna、Llama-3、Qwen2.5）和多种单通道检测器（如LLM-Check、SEP、ICR-Probe），结果显示CORVUS可使AUROC、准确率等指标大幅下降，证明其有效性。

**⚠️ 局限性**

局限性包括需要白盒访问、仅针对单通道检测器、未改进事实准确性，以及对封闭API或更高级检索式防御不具备适用性。

---

## 4. On the Limits of Learned Importance Scoring for KV Cache Compression

**arXiv ID:** 2601.14279 | [PDF](https://arxiv.org/pdf/2601.14279v1)

**作者:** Brady Steele `[一作]` (Georgia Institute of Technology), Brady Steele `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 271 | [OpenAlex ID](https://openalex.org/A5016775476)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对学习型 KV 缓存压缩方法进行系统评估，提出并训练了 1.7M 参数的 Speculative Importance Prediction (SIP) 模型。

**💡 创新点**

创新点在于构建了一种专门用于非查询感知的多尺度重要性预测器，并在严谨的实验框架下展示了学习方法在该任务上的局限性。

**🔧 技术方法**

技术包括多尺度前瞻预测、跨注意力机制、跨层时间衰减、误差校准以及基于注意力标签的多任务训练。

**📊 数据集**

使用 TinyLlama‑1.1B‑Chat 预训练模型的 KV 表示以及 1,000 条 OpenWebText 长度 2048 的序列进行训练和评估。

**📈 对比分析**

与随机选择、位置启发式、prefill‑attention、H2O、StreamingLLM 等基线对比，SIP 在任何压缩比例下均未获得统计学显著优势，性能与简单启发式相当。

**⚠️ 局限性**

实验仅覆盖 1.1B 参数模型、2048 长度上下文，且仅使用一种学习架构，结果可能不适用于更大模型、长上下文或其他模型结构。

---

## 5. The Slow Drift of Support: Boundary Failures in Multi-Turn Mental Health LLM Dialogues

**arXiv ID:** 2601.14269 | [PDF](https://arxiv.org/pdf/2601.14269v1)

**作者:** Youyou Cheng `[一作]` (University of Incarnate Word), Qiyang Pan `[通讯]` (Mayo Clinic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个多轮对话安全压力测试框架，针对心理健康聊天机器人通过虚拟患者资料进行长对话评估。

**💡 创新点**

创新点在于从单轮检测转向多轮边界演变评估，并比较静态进展与自适应探测两种压力模式。

**🔧 技术方法**

使用GPT‑5.2生成虚拟患者、评判器和规划器，并对DeepSeek‑Chat、Gemini‑2.5‑Flash和Grok‑3进行评测。

**📊 数据集**

数据集为50份手工设计的虚拟患者档案（包含人口学、心理特征、升级路径等），每份可进行至多20轮对话。

**📈 对比分析**

通过边界违规率与到达违规的轮次（time‑to‑breach）进行比较，实验显示自适应探测导致违规时间更早，整体违规率普遍较高，尤其中文测试更易违规。

**⚠️ 局限性**

局限性包括仅使用模拟患者、缺乏真实用户反馈、只关注特定违规类型、以及GPT‑5.2驱动的评判可能存在偏差。

---

## 6. IntelliSA: An Intelligent Static Analyzer for IaC Security Smell Detection Using Symbolic Rules and Neural Inference

**arXiv ID:** 2601.14595 | [PDF](https://arxiv.org/pdf/2601.14595v1)

**作者:** Qiyue Mei `[一作]` (University of Melbourne), Michael Fu `[通讯]` (University of Melbourne)

**通讯引用:** 790 | [OpenAlex ID](https://openalex.org/A5102710465)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 IntelliSA，一种结合符号规则与神经推理的 IaC 安全臭味检测工具。

**💡 创新点**

创新点在于通过 LLM 生成伪标签进行知识蒸馏，将大模型的判断压缩到 2.2 亿参数的轻量化学生模型，实现了高效、低成本的误报过滤。

**🔧 技术方法**

技术包括符号规则匹配、LLM（Claude‑4）伪标签生成、CodeT5p‑220M 神经模型蒸馏、基于 IR 的多平台解析。

**📊 数据集**

使用 Saavedra 等人公开的 241 条 Puppet/Ansible/Chef 脚本（共 11,814 行）与人工标注的 213 条安全臭味实例进行训练与评估。

**📈 对比分析**

与 SLIC、SLAC、Glitch 以及 Claude‑4、Grok‑4、GPT‑5 等基线相比，IntelliSA 在三大技术上实现宏观 F1 最高 83%，在成本效益指标 Effort@60%Recall 与 F1@1%LOC 上均优于所有基线，误报率显著下降。

**⚠️ 局限性**

局限包括：仍有部分真实臭味被误判为假阳性，无法补偿符号规则本身漏检的真阳性；对稀有臭味和未包含在训练集中的技术/模型适应性有限。

---

## 7. Learning PDE Solvers with Physics and Data: A Unifying View of Physics-Informed Neural Networks and Neural Operators

**arXiv ID:** 2601.14517 | [PDF](https://arxiv.org/pdf/2601.14517v1)

**作者:** Yilong Dai `[一作]` (University of Alabama), Runlong Yu `[通讯]` (University of Alabama)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2017–2026年间以数据驱动方式求解偏微分方程（PDE）的两大主流范式——物理信息神经网络（PINNs）与神经算子（NOs），提出了统一的设计空间与分层分类法，阐述了它们的核心结构、优势与不足，并在实际科学工作流中给出方法选择与应用场景分析；

**💡 创新点**

首次将PINNs与NOs置于同一“物理–数据”连续体与共享设计空间中，识别出三大结构决策（学习对象、监督接口、算力摊销点），构建基于结构属性的跨范式评估框架，为后续混合模型与可解释性研究提供系统视角；

**🔧 技术方法**

主要运用理论分析、分类框架构建、案例综述以及对现有文献的系统梳理与综合；通过对比研究，提出了多尺度、几何边界、可信度等三大结构属性及其对模型设计与评估的影响；

**📊 数据集**

本文并未提供新的实验数据集，而是引用了近十年顶级AI/ML会议（NeurIPS, ICML, ICLR, CVPR）以及数学与物理期刊中使用的各类PDE数据集（如Navier–Stokes、热传导、弹性力学等模拟数据及部分实验测量数据）；

**📈 对比分析**

本综述未进行统一的实验对比；作者总结了已有工作中不同范式的评估指标（均方误差、谱误差、边界违规率、长期滚动误差等），并讨论了在不同工作流契约下的性能优势与劣势；

**⚠️ 局限性**

局限性包括：① 依赖已有文献的实验结果，缺乏统一、可复现的基准评估；② 重点关注插值性能，对外推（契约漂移）与可信度诊断的系统化验证不足；③ 未给出明确的成本模型或规模律；④ 主要为综述性质，缺乏新的算法或实证验证。

---

## 8. Tracing the Data Trail: A Survey of Data Provenance, Transparency and Traceability in LLMs

**arXiv ID:** 2601.14311 | [PDF](https://arxiv.org/pdf/2601.14311v1)

**作者:** Richard Hohensinner `[一作]` (Pro2Future GmbH), Roman Kern `[通讯]` (Graz University of Technology)

**通讯引用:** 2195 | [OpenAlex ID](https://openalex.org/A5014398832)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述过去十年关于LLM数据来源的研究，构建了数据来源、可追溯性、透明性等三轴与偏见、不确定性、隐私、工具技术等六柱的系统分类框架，并对95篇论文进行了主题分布与方法评估

**💡 创新点**

首次提出了将LLM数据治理拆解为三轴与六柱的综合维度框架，并系统梳理并量化了LLM数据来源、可追溯性与透明性技术的发展及其相互关系

**🔧 技术方法**

使用系统文献检索（Scopus）、关键词过滤、人工筛选构建数据集，并采用分类表格、图表和量化指标（如文献数量、技术类型）进行比较分析

**📊 数据集**

主要基于公开研究论文与开源LLM项目（如GPT、Llama、ChatGPT、Gemma等）以及公开数据集（如FineWeb、WebText、OpenWebText、BERT语料等）进行案例与技术阐述

**📈 对比分析**

通过对95篇文献的统计与分类展示，表明在数据来源、可追溯性与透明性技术上已出现多样化方法（如水印、追踪、解释、可视化），但缺乏统一的评测基准，性能评价多为案例级别且缺乏横向对比

**⚠️ 局限性**

主要局限包括：缺乏统一评测基准与度量标准，数据来源与方法的可复现性不高；对闭源LLM透明度的研究有限；隐私与偏见技术与可追溯性方法耦合度不足；并未系统探讨LLM在不同行业与法规环境下的跨域适用性

---

## 9. Language, Caste, and Context: Demographic Disparities in AI-Generated Explanations Across Indian and American STEM Educational Systems

**arXiv ID:** 2601.14506 | [PDF](https://arxiv.org/pdf/2601.14506v1)

**作者:** Amogh Gupta `[一作]` (Society-Centered AI Lab), S Gaikwad `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）在印度和美国工程教育背景下，对不同社会身份交叉的学生画像提供解释质量的差异。

**💡 创新点**

首次从跨文化和交叉身份视角量化LLM的教学偏见，并揭示即使学生获得社会流动性，身份特征仍导致解释复杂度差异。

**🔧 技术方法**

采用了LLM排名与生成任务，使用MAB、MDB等偏差指标，并通过t检验、Cohen's d、KL散度等统计方法评估公平性。

**📊 数据集**

数据集包括MATH‑50（通用数学题）和JEEBench（印度工程入学考试题），并构造了覆盖种姓/种族、收入、语言、院校层次等维度的学生画像。

**📈 对比分析**

对比四个模型（Qwen2.5‑32B‑Instruct、GPT‑4o、GPT‑4o‑mini、GPT‑OSS‑20B）在排名和生成任务中的解释复杂度；结果显示所有模型均存在显著偏差，模型规模和开源/闭源属性对公平性影响不大。

**⚠️ 局限性**

局限在于使用构造的虚拟画像而非真实学生，可能忽略学生真实学习体验；且仅评估了两国教育系统，未覆盖更广泛的跨文化情境。

---

## 10. Adaptive KDE for Real-Time Thresholding: Prioritized Queues for Financial Crime Investigation

**arXiv ID:** 2601.14473 | [PDF](https://arxiv.org/pdf/2601.14473v1)

**作者:** Danny Butvinik `[一作]` (NICE Actimize), Achi Hackmon `[通讯]` (NICE Actimize)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

做了什么：通过在线自适应核密度估计与密度谷定位，实现实时风险评分阈值设定，满足容量约束并支持多队列路由。

**💡 创新点**

创新点是什么：在无标签环境下将阈值锚定到持久密度谷，采用边界反射的自适应 KDE 并加入阈值粘滞机制，以抑制阈值抖动。

**🔧 技术方法**

用了什么技术：在线自适应 KDE（Epanechnikov 核 + Abramson 带宽）、边界反射、持久性谷检测、指数衰减/滑动窗口、阈值粘滞、以及多队列容量匹配。

**📊 数据集**

用了什么数据集：使用合成的多模式 Beta 混合分布生成的评分流，涵盖不同业务活动并模拟漂移、季节性、离散化等情形。

**📈 对比分析**

如何比较的方法，性能怎么样：与窗口量化、固定带宽 KDE、EWMA 阈值等基线对比，实验显示在容量跟踪精度、阈值抖动和队列波动方面均优于基线，误差率保持在 10% 范围内。

**⚠️ 局限性**

limitation是什么：仅基于分数无标签，无法提升检测精度；在稀疏窗口或高度离散化时谷检测不稳定；阈值响应的可调参数（窗口、衰减、粘滞）需手动调节。

---

## 11. MARBLE: Multi-Agent Reasoning for Bioinformatics Learning and Evolution

**arXiv ID:** 2601.14349 | [PDF](https://arxiv.org/pdf/2601.14349v1)

**作者:** Sunghyun Kim `[一作]` (Dongguk University), Sangsoo Lim `[通讯]` (Dongguk University)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5101655230)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出MARBLE框架，实现多代理闭环的文献感知、辩论式推理、自动执行与基于性能的记忆更新，自动化迭代改进生物信息学模型架构。

**💡 创新点**

将文献检索与实验反馈结合，构建结构化辩论推理流程，并通过执行稳定性与性能奖励实现真正自主、可重复、可持续的模型演化，突破传统AutoML仅限搜索空间和LLM仅限代码生成的局限。

**🔧 技术方法**

使用多代理系统（研究组、批评者、评估者、实现建筑师等），BGE‑M3语义嵌入+混合检索、经验奖励、代码验证与重试、Docker化执行、性能指标反馈等技术。

**📊 数据集**

使用空间转录组域分割（STAGATE、DeepST）、药物‑靶点交互预测（HyperAttentionDTI、DLM‑DTI）和药物响应预测（DeepTTA、DeepDR）等公开基准数据集。

**📈 对比分析**

与Claude、Codex、NADER、AutoML‑Agent等基线对比；在6个模型上，MARBLE均取得正向累计性能提升（NPG>0）、较高的持续改进（NAUI、SIC）以及近乎100%执行成功率，基线多为偶发改进且执行不稳定。

**⚠️ 局限性**

仅支持固定数据模态，未引入新实验测量；迭代成本高、可扩展性待评估；并未直接产生新的生物发现。

---

## 12. Guided by the Plan: Enhancing Faithful Autoregressive Text-to-Audio Generation with Guided Decoding

**arXiv ID:** 2601.14304 | [PDF](https://arxiv.org/pdf/2601.14304v1)

**作者:** Juncheng Wang `[一作]` (Hong Kong Polytechnic University), Shujun Wang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 6116 | [OpenAlex ID](https://openalex.org/A5100602073)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究发现自回归（AR）文本到音频生成模型在生成前缀时已隐式编码全局语义属性，并提出了Plan‑Critic辅助模型用于在推理时基于前缀评估最终语义符合度，随后通过前缀优先采样策略提升生成质量。

**💡 创新点**

创新点：①揭示AR模型存在的隐式规划能力；②设计基于Generalized Advantage Estimation的Plan‑Critic训练框架，能够从部分生成序列预测最终CLAP得分；③提出前缀导向采样策略，既提升语义对齐又保持与传统best‑of‑N相同的计算量。

**🔧 技术方法**

技术：自回归Transformer音频生成（Siren） + 轻量级Critic Transformer + GAE‑启发式训练 + 前缀优先采样（Plan‑Critic guided sampling），并使用CLAP作为指令遵循评估指标。

**📊 数据集**

数据集：使用LLM生成的5k伪音频字幕及AudioCaps、VGGSound等公开数据集中的1k测试提示；生成音频后用CLAP评估其与文本的语义一致性。

**📈 对比分析**

与其他AR模型（如Siren）以及双向扩散模型（AudioLDM2、MagNet等）对比，Plan‑Critic指导下的Siren在CLAP得分上提升约10分，达到AR文本到音频生成的state‑of‑the‑art水平；在FAD、FD、KL、IS等音频质量指标上保持与Siren相近。

**⚠️ 局限性**

局限性：①依赖CLAP作为代理指标，可能无法捕捉人类感知的细粒度差异；②固定前缀长度（32令牌）假设在更长或更复杂的音频序列中可能不适用；③Plan‑Critic训练仅基于Siren生成的音频，可能对其他AR模型或不同音频域的泛化能力有限。

---

## 13. Large-Scale Label Quality Assessment for Medical Segmentation via a Vision-Language Judge and Synthetic Data

**arXiv ID:** 2601.14406 | [PDF](https://arxiv.org/pdf/2601.14406v1)

**作者:** Yixiong Chen `[一作]`, Alan Yuille `[通讯]` (Johns Hopkins University)

**通讯引用:** 104836 | [OpenAlex ID](https://openalex.org/A5086706224)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了 SegAE，一种基于视觉‑语言模型的标签质量评估工具，能够自动预测医学影像分割标注的 Dice 评分并用于数据集评测与样本筛选。

**💡 创新点**

创新点在于：①将视觉编码器与语言编码器结合，利用文本语义区分同一图像中的多重解剖结构；②引入“最佳配对排名”损失，提升模型对标签质量的排序能力；③构建大规模合成质量数据集，仅依赖伪标签生成，减少人工标注；④实现轻量级、高效评估（3D mask 0.06 s）。

**🔧 技术方法**

使用 BiomedCLIP 的预训练视觉与语言编码器；多层感知机加注意力机制进行特征融合；MSE 与最佳配对排名联合损失；数据增强与重采样提升泛化；评估指标包括 LCC、SROCC、MAP@k 等。

**📊 数据集**

主要使用了 DAP Atlas（含 142 个结构的 4M 切片-掩码对）、AbdomenAtlas、TotalSegmentator、BTCV、AbdomenCT‑12organ 等公开 CT 分割数据集，结合合成的伪标签生成训练集。

**📈 对比分析**

与随机、Monte‑Carlo dropout、熵等传统不确定度方法相比，SegAE 在主动学习和半监督学习中均能显著提升 Dice/NSD（在 TotalSegmentator 上比随机高约 0.007‑0.012，半监督比熵高 1.6‑1.9 点），并在评估速度和资源占用上优于对比方法（3D mask 0.06 s，显著降低 RAM/磁盘需求）。

**⚠️ 局限性**

局限性包括：目前仅在 CT 成像上验证，未扩展到 MRI、超声等其他模态；对极少见或全新解剖结构需额外文本提示，可能需要微调；对全局 3D 关联信息处理仍有限；合成质量数据的真实性仍依赖于原始分割模型的性能。

---

## 14. Scribble-Supervised Medical Image Segmentation with Dynamic Teacher Switching and Hierarchical Consistency

**arXiv ID:** 2601.14563 | [PDF](https://arxiv.org/pdf/2601.14563v1)

**作者:** Thanh-Huy Nguyen `[一作]` (Carnegie Mellon University), Ulas Bagci `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种双教师、单学生框架 SDT-Net，用动态教师切换与层次一致性机制，在仅有稀疏涂鸦标注的医学图像分割任务中提升性能

**💡 创新点**

创新点在于：①动态教师切换（DTS）根据涂鸦区域损失动态选取更可靠的教师，减轻伪标签噪声和确认偏差；②层次一致性（HiCo）在低层几何与高层语义两级特征上实现学生与教师的对齐，增强结构学习；③结合Pick Reliable Pixels（PRP）筛选高置信度像素生成伪标签，提升伪标签质量

**🔧 技术方法**

技术包括：涂鸦监督下的部分交叉熵损失、伪标签损失（交叉熵+Dice）、层次一致性损失（L1+余弦相似度）、教师模型的指数移动平均（EMA）更新、学生网络与教师网络共享 UNet 结构

**📊 数据集**

使用 ACDC（心脏 MRI 100 张）和 MSCMRseg（45 张 Late Gadolinium Enhancement MRI）两个公开数据集，分别对右心室（RV）、左心室（LV）和心肌（MYO）进行分割

**📈 对比分析**

与 DMPLS、ScribbleVC、ScribFormer、AIL、HELPNet 等多种涂鸦监督方法比较，SDT-Net 在 ACDC 上平均 Dice 得分 90.8%（比最佳 HELPNet 高 2.0%），在 MSCMRseg 上平均 Dice 90.0%（比最佳 HELPNet 高 1.8%），在各类结构上均显著提升，尤其是对 RV 区域的精细边界把握

**⚠️ 局限性**

局限性包括：①仍需人工涂鸦标注，标注成本未完全消除；②模型训练过程较为复杂，需要双教师、动态切换和多级对齐，计算开销和实现难度较高；③在极少标注或极端噪声场景下，DTS 与 PRP 的选择阈值可能需要手动调参

---

## 15. Predicting Tail-Risk Escalation in IDS Alert Time Series

**arXiv ID:** 2601.14299 | [PDF](https://arxiv.org/pdf/2601.14299v1)

**作者:** Ambarish Gurjar `[一作]` (University of North Carolina at Charlotte), L Jean Camp `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 538 | [OpenAlex ID](https://openalex.org/A5083819292)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过构造强度、波动率和动量特征，利用XGBoost模型对IDS警报流的高强度波动进行30分钟的早期预警。

**💡 创新点**

将金融极值预测方法迁移至网络安全，提出以95%分位数阈值为目标的极端状态预测框架，并通过可解释的时间微结构特征实现高准确率。

**🔧 技术方法**

特征工程：基于滑动窗口的强度、波动率、动量；模型：梯度提升树XGBoost；评价指标：准确率、召回率、F1、ROC‑AUC。

**📊 数据集**

使用某美国公立大学的三个月Suricata IDS日志（约2.5亿条警报，分为5个严重等级）。

**📈 对比分析**

在各严重等级上进行时间序列划分（70%训练/30%测试），对比全特征、剔除波动率/动量的消融实验，最高F1 > 0.97，ROC‑AUC > 0.99；单一强度特征性能急剧下降。

**⚠️ 局限性**

仅使用单一模型族和单一95%阈值，未考虑多维风险（技术、协议等），波动率/动量参数未做系统性敏感性分析，且缺乏跨机构、跨数据集的泛化验证。

---

## 16. Guardrails for trust, safety, and ethical development and deployment of Large Language Models (LLM)

**arXiv ID:** 2601.14298 | [PDF](https://arxiv.org/pdf/2601.14298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 17. Shapley Value on Uncertain Data

**arXiv ID:** 2601.14543 | [PDF](https://arxiv.org/pdf/2601.14543v1)

**作者:** Zhuofan Jia `[一作]` (Duke University), Jian Pei `[通讯]` (Duke University)

**通讯引用:** 81360 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在数据共享环境中将玩家贡献视为从概率分布抽样的数据，提出了概率化的Shapley值框架，并同时估计期望和方差。

**💡 创新点**

创新点在于把Shapley值从确定性数据扩展到随机抽样数据，提出无偏的期望与方差估计，并设计了基于采样池的池化与分层池化算法。

**🔧 技术方法**

采用蒙特卡罗采样、无偏估计、Bootstrap重采样与分层分配等技术。

**📊 数据集**

使用合成数据（Gaussian+sin组合）和Wine Quality真实数据集。

**📈 对比分析**

通过与基线方法对比，池化方法减少97%采样量，分层池化在保持预算不变的前提下，方差降低70–95%，实验显示估计精度和稳定性显著提升。

**⚠️ 局限性**

局限在于仍假设玩家分布已知且可抽样，未考虑客户端参与随机、分布漂移以及高维大规模情况。

---

## 18. Forest-Chat: Adapting Vision-Language Agents for Interactive Forest Change Analysis

**arXiv ID:** 2601.14637 | [PDF](https://arxiv.org/pdf/2601.14637v1)

**作者:** James Brock `[一作]` (University of Bristol), Nantheera Anantrasirichai `[通讯]` (University of Bristol)

**通讯引用:** 3054 | [OpenAlex ID](https://openalex.org/A5021717616)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过将多任务视觉语言模型与大型语言模型相结合，构建了Forest-Chat对话式遥感图像森林变化分析系统，支持像素级变化检测、变化描述、计数、面积估算等多项任务。

**💡 创新点**

创新点包括提出集成多任务MCI视觉语言骨干、AnyChange零样本检测与LLM驱动的任务协调框架，并首次发布用于森林变化联合检测与字幕的Forest-Change数据集。

**🔧 技术方法**

采用的技术包括Siamese SegFormer骨干的多级变化解释模型（Bi3层）、基于SAM的AnyChange零样本变化检测、LLM（ChatGPT‑4o‑mini）控制器以及Python工具链实现的交互式接口。

**📊 数据集**

实验使用自建的Forest‑Change数据集（约334对遥感图像、变化掩码与5句字幕）和从LEVIR‑MCI抽取的LEVER‑MCI‑Trees子集（约2305对带树覆盖变化）。

**📈 对比分析**

在Forest‑Change上，监督版FC‑Supervised获得mIoU 67.10%和BLEU‑4 40.17%；在LEVER‑MCI‑Trees上得到mIoU 88.13%和BLEU‑4 34.41%；零样本版FC‑Zero‑shot在两数据集上mIoU约59%–47%，低于监督版但仍可与SOTA方法进行对比。

**⚠️ 局限性**

主要局限包括对小、零散森林变化检测效果差、字幕生成受规则化限制缺乏多样性、Zero‑shot方法对大气噪声和季节变化敏感，以及数据规模与多样性不足导致模型泛化受限。

---

## 19. Predicting Long-Term Self-Rated Health in Small Areas Using Ordinal Regression and Microsimulation

**arXiv ID:** 2601.14335 | [PDF](https://arxiv.org/pdf/2601.14335v1)

**作者:** Seán Caulfield Curley `[一作]` (University of Galway), Patrick Mannion `[通讯]` (University of Galway)

**通讯引用:** 1337 | [OpenAlex ID](https://openalex.org/A5046330057)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

结合动态微观模拟与序数回归预测2023-2057年爱尔兰不同选区自评健康状况分布。

**💡 创新点**

在小区层面同时运用开放源码微观模拟与序数回归，并引入全国比例对齐校正与最佳/最差情景分析，提升预测的地理细粒度和情景灵活性。

**🔧 技术方法**

使用SEMIPro动态微观模拟、累积logit序数回归、对齐比例校正、添加高斯过程预测以及蒙特卡洛采样。

**📊 数据集**

主要数据集包括爱尔兰CSO人口普查与DCM预测、Healthy Ireland 2023健康调查数据、迁移、出生、死亡率等社会经济指标。

**📈 对比分析**

通过与2022年各选区实际SRH分布比较，平均R²≈0.9、MSE≈0.0037，表明预测结果与实测高度一致。

**⚠️ 局限性**

受限于基线假设、对齐比例可能偏差、迁移与健康分布的未来不确定性，以及在极小或极端区域预测的稳定性有限。

---

## 20. Structured Image-based Coding for Efficient Gaussian Splatting Compression

**arXiv ID:** 2601.14510 | [PDF](https://arxiv.org/pdf/2601.14510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 21. U-Harmony: Enhancing Joint Training for Segmentation Models with Universal Harmonization

**arXiv ID:** 2601.14605 | [PDF](https://arxiv.org/pdf/2601.14605v1)

**作者:** Weiwei Ma `[一作]` (Washington University in St. Louis), Yongsong Huang `[通讯]` (Tohoku University)

**通讯引用:** 192 | [OpenAlex ID](https://openalex.org/A5101074338)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了Universal Harmonization（U‑Harmony）模块和域门控头，实现单一分割模型在异构医学影像数据集上联合训练，并能在推理时无需数据集标签。

**💡 创新点**

创新点在于两阶段特征调和与恢复机制（先归一化后恢复域特定信息）以及域门控头的无标签推理能力，使模型既能学习跨域共享知识，又能保留各域独有特征，并支持多模态与多解剖学类别的统一学习。

**🔧 技术方法**

技术包括U‑Harmony模块（实例归一化、可学习多项式仿射变换、第二阶段恢复）、域门控头（共享头+域原型软门控）、以及在nn‑UNet、SwinUNETR等架构上的集成；与BatchNorm、LayerNorm、RevIN等归一化方法进行对比；使用AdamW优化器。

**📊 数据集**

使用公开脑部肿瘤/转移分割数据集：UCSF‑BMSR（560 MR，5,136条转移注释）、BrainMetShare（156 MR，四模态）以及BraTS‑METS 2023（1,303 MR，包含增强肿瘤、肿瘤核心和SNFH标签）。

**📈 对比分析**

通过Dice相似系数与多种基线（3D‑UNet、TransUNet、V‑Net、UNETR、nnFormer、nn‑UNet、SwinUNETR、CVCL、MoME、MultiTalent、RevIN等）比较，U‑Harmony在nn‑UNet和SwinUNETR上平均提升1.6–3.4%，在多域联合训练中提升至+8.4%，并在边界精细化方面表现优异。

**⚠️ 局限性**

局限性包括：与需要数据集标签的oracle多头模型相比仍略低约1.3% Dice；对极端域差异的鲁棒性尚待进一步验证；缺乏推理速度/显存消耗评估；高分辨率影像的计算成本可能较高。

---

## 22. A Unified Framework for Scalable and Robust Paper Assignment

**arXiv ID:** 2601.14402 | [PDF](https://arxiv.org/pdf/2601.14402v1)

**作者:** Michael Cui `[一作]` (Carnegie Mellon University), Fei Fang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6433 | [OpenAlex ID](https://openalex.org/A5061127138)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种统一可扩展的论文分配框架RAMP，兼顾分配质量、多样性和鲁棒性

**💡 创新点**

创新点在于将凸凹形扰动最大化目标与软约束结合，并通过分段线性逼近、属性感知抽样和稀疏化技术实现大规模实时分配

**🔧 技术方法**

使用了soft-constraint优化（concave perturbed-maximization）、分段线性化、Birkhoff–von Neumann采样、稀疏化以及Gurobi求解器

**📊 数据集**

实验数据集包括20,000+论文的Synthetic Large集以及AAMAS 2015、ICLR 2018和S2ORC四个数据集

**📈 对比分析**

与Default、MILP、PLRA、PM等基线对比，RAMP在Large集上在20分钟内完成，质量仅下降2.6%，同时显著提升多样性（div从0.615→0.895）和鲁棒性（coauthor 163→21，2-cycle 86→1）

**⚠️ 局限性**

局限在于需要手工调参软约束权重，对极端稀疏场景的理论保证有限，且某些公平性指标未完全覆盖

---

## 23. Place with Intention: An Empirical Attendance Predictive Study of Expo 2025 Osaka, Kansai, Japan

**arXiv ID:** 2601.14570 | [PDF](https://arxiv.org/pdf/2601.14570v1)

**作者:** Xiaojie Yang `[一作]` (University of Tokyo), Noboru Koshizuka `[通讯]` (University of Tokyo)

**通讯引用:** 1526 | [OpenAlex ID](https://openalex.org/A5012068346)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于 Transformer 的日度参观人数预测框架，利用 Expo 2025 大阪的门票预订动态（预订更新）作为主要预测信号；

**💡 创新点**

创新点在于：①将预订动态视为访客意图的代理，避免多源外部数据融合；②设计了自适应融合模块和逆向式时间嵌入，以提升模型对不同时间尺度的捕捉能力；

**🔧 技术方法**

使用了 Transformer 编码器-解码器架构、时间嵌入、交叉注意力、自适应融合以及软标签门控机制；

**📊 数据集**

采用了从 2025 年 4 月 23 日至 9 月 17 日收集的 Expo 2025 大阪的入口记录与门票预订更新数据，按小时（08:00-22:00 共 14 槽）聚合；

**📈 对比分析**

在两通道（东门、西门）预测设置下，与 ARIMA、LSTM、GRU、TCN 等基线相比，MAE 下降至 745，MAPE 降至 1944，尤其在 5 天预测窗口内相对 TCN 提升 5.9%/44.9%；

**⚠️ 局限性**

局限性包括：依赖预订系统的稳定性，突发政策或罕见事件可能导致预订动态与实际入场不一致；模型在 Expo 2025 大阪之外的场景通用性尚未验证；

---

## 24. Diffusion Large Language Models for Black-Box Optimization

**arXiv ID:** 2601.14446 | [PDF](https://arxiv.org/pdf/2601.14446v1)

**作者:** Ye Yuan `[一作]` (McGill), Xue Liu `[通讯]` (McGill)

**通讯引用:** 13438 | [OpenAlex ID](https://openalex.org/A5100372152)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用预训练的扩散大型语言模型（diffusion LLM）在离线黑盒优化（BBO）中通过上下文提示和掩码扩散树搜索实现设计空间的高效搜索与迭代优化

**💡 创新点**

①将任务描述、离线数据与指令拼接为自然语言提示，使扩散 LLM 能在上下文中进行自我指导的 denoising；②提出“掩码扩散树搜索”（MDTS），把扩散过程视为蒙特卡洛树搜索，动态平衡探索与利用；③结合高斯过程预测的期望改进（EI）来评估子节点，提升搜索方向性

**🔧 技术方法**

扩散 LLM（如 LLaDA‑8B‑Instruct、MMaDA‑8B‑MixCoT）、蒙特卡洛树搜索、基于高斯过程的期望改进评估、自然语言提示工程

**📊 数据集**

Design‑Bench 四个任务：Ant Morphology、D’Kitty Morphology（连续空间）和 TF Bind 8、TF Bind 10（离散 DNA 序列），每个任务采用 10 条离线样本的 few‑shot 设置

**📈 对比分析**

与 15 种基线（代理方法、生成模型、MCTS、CMA‑ES 等）对比，dLLM 在 100‑th 百分位归一化得分上始终排名第一（如 Ant 0.652、TF8 0.876），在所有任务中均取得显著提升

**⚠️ 局限性**

依赖大规模预训练扩散 LLM 与 GPU 计算，推理成本较高；对离线数据量和提示设计敏感；缺乏理论收敛保证，且对更大规模或更复杂任务的泛化尚待验证

---

## 25. Vision-Based Natural Language Scene Understanding for Autonomous Driving: An Extended Dataset and a New Model for Traffic Scene Description Generation

**arXiv ID:** 2601.14438 | [PDF](https://arxiv.org/pdf/2601.14438v1)

**作者:** Danial Sadrian Zadeh `[一作]` (University of Waterloo), Behzad Moshiri `[通讯]` (University of Tehran)

**通讯引用:** 3601 | [OpenAlex ID](https://openalex.org/A5073252452)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将单摄像头前视图图像转换为交通场景自然语言描述的端到端框架。

**💡 创新点**

创新点包括：1) 设计混合注意力融合与静态图像时序化的 MViTv2‑S + xLSTM 编码‑解码器；2) 构建基于 BDD100K 的 10K 图像子集，并为每张图生成 10 句符合安全驾驶准则的描述；3) 对多种评估指标进行系统对比，确定 CIDEr 与 SPICE 为最适合的指标。

**🔧 技术方法**

采用 Vision Transformer（MViTv2‑S）、扩展 LSTM（xLSTM）解码器、跨模态注意力、跨注意力融合、静态图像时序化技术，并使用交叉熵训练。

**📊 数据集**

使用 BDD100K 10K 图像子集（600/1000 图像已标注，后续扩充至 1,000）作为训练和评估数据；对比测试还使用了 Flickr8k 的 5 张驾驶场景图像。

**📈 对比分析**

通过三阶段实验比较十种 encoder‑decoder 组合，评估指标包括 BLEU、ROUGE‑L、METEOR、CIDEr、SPICE、BERTScore、CLIPScore 等；最终 MViTv2‑S‑xLSTM 在 CIDEr、SPICE 上优于 VGG‑16‑E01，泛化到未见图像和 Flickr8k 的描述质量也可接受。

**⚠️ 局限性**

限制：① 训练数据量仅 600/1,000 张图像；② 仅使用交叉熵损失，未进行 RL 微调；③ MViTv2‑S 权重被冻结，无法进一步提升性能；④ CLIPScore 等指标受文本长度限制，评估时需额外处理。

---

## 26. GEGO: A Hybrid Golden Eagle and Genetic Optimization Algorithm for Efficient Hyperparameter Tuning in Resource-Constrained Environments

**arXiv ID:** 2601.14672 | [PDF](https://arxiv.org/pdf/2601.14672v1)

**作者:** Amaras Nazarians `[一作]` (American University of Armenia), Sachin Kumar `[通讯]` (American University of Armenia)

**通讯引用:** 5042 | [OpenAlex ID](https://openalex.org/A5032948428)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将黄金鹰优化（GEO）与遗传算法（GA）融合的混合元启发式优化框架GEGO，用于在计算受限环境下进行高维超参数调优。

**💡 创新点**

创新点在于将遗传算子直接嵌入GEO的迭代搜索过程中，而非单独作为演化阶段，从而在保持GEO探索能力的同时持续维持种群多样性并抑制早熟收敛。

**🔧 技术方法**

技术包括黄金鹰优化的攻击与巡航向量更新、遗传算子的选择、交叉（线性交叉）和变异（低概率位点变异），以及混合编码-解码策略处理连续与离散混合搜索空间。

**📊 数据集**

使用了标准CEC2017复合函数（100维）作为基准，MNIST手写数字数据集进行ANN超参数调优（网络层数、神经元数、dropout、批大小、学习率等）。

**📈 对比分析**

通过多组40次独立实验将GEGO与GEO、GA以及其他经典算法（PSO、GWO、JSA、SCA、L‑SHADE、CMA‑ES）对比，结果显示GEGO在大多数基准函数和MNIST调优中获得更优或相近的最优值，且在测试准确率上最高可达97.90%，显著优于单独使用GEO或GA。

**⚠️ 局限性**

局限性包括：在大型深度网络和大数据集上未进行验证；与高级DE变体（如L‑SHADE）在大规模复合问题上的表现不一；实验次数有限，未做严格的统计显著性检验；仅限于固定网络结构的超参数搜索，未覆盖更广泛的神经架构搜索。

---

## 27. AI Agents vs. Human Investigators: Balancing Automation, Security, and Expertise in Cyber Forensic Analysis

**arXiv ID:** 2601.14544 | [PDF](https://arxiv.org/pdf/2601.14544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 28. A full process algebraic representation of Ant Colony Optimization

**arXiv ID:** 2601.14436 | [PDF](https://arxiv.org/pdf/2601.14436v1)

**作者:** Maria Garcia `[一作]`, Ismael Rodriguez `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文设计并实现了过程代数 PA^2CO，能够完整形式化描述蚁群优化（ACO）算法及其多种并行实现（细粒度与粗粒度）。在此代数基础上，给出了 Ant‑System、MAX‑MIN Ant‑System 与 Ant‑Colony System 的完整规范，并分别演示了多种并行化策略（各蚂蚁独立进程、共享图过程、自由蚂蚁、多余路径收集等）。

**💡 创新点**

①首个针对 ACO 的完整过程代数规范，既捕捉算法功能逻辑，也细致刻画并行执行中的同步与通信；②提供从细粒度到粗粒度的完整并行化空间，便于根据需求抽象或细化；③为后续形式化分析、验证与自动化工具支持奠定理论基础。

**🔧 技术方法**

使用过程代数与操作语义定义状态、变换、条件和通信；引入概率与非确定性混合模型；采用状态/变换函数、通信通道、同步与并发组合；通过符号化状态空间描述 ACO 的决策与信息更新。

**📊 数据集**

论文以旅行商问题（TSP）作为示例说明概率计算与信息更新，但未使用具体数值数据集进行实验，而是采用符号/抽象数据进行建模。

**📈 对比分析**

本文未进行实测实验；性能与方法的比较主要基于理论分析与规范结构对比。作者指出，粗粒度独立复制的实现通常是最有效的，并通过对不同规范的细节对比阐释细粒度与粗粒度实现的差异。

**⚠️ 局限性**

1）仅针对 ACO 进行了形式化，未覆盖其他蚁群变体；2）模型复杂，使用门槛高，需手工构造；3）缺乏实证验证与性能评估；4）未针对大规模实例的状态爆炸问题给出解决方案。

---

## 29. Rethinking On-Device LLM Reasoning: Why Analogical Mapping Outperforms Abstract Thinking for IoT DDoS Detection

**arXiv ID:** 2601.14343 | [PDF](https://arxiv.org/pdf/2601.14343v1)

**作者:** William Pan `[一作]` (San Francisco State University), Rose Qingyang Hu `[通讯]` (Virginia Polytechnic Institute and State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用链式推理（CoT）和检索增强生成（RAG）方法，提升边缘设备小型大语言模型（ODLLM）在 IoT DDoS 检测中的性能

**💡 创新点**

创新点在于将教师模型生成的 CoT 解释作为示例指导小模型，并结合 XGBoost 生成的概率签名实现检索增强，显著提升了模型的识别准确率

**🔧 技术方法**

使用技术包括：Chain‑of‑Thought 结构化提示、检索增强生成（Prob‑RAG）、XGBoost 分类器生成的概率向量、少量示例（one‑shot / few‑shot）以及小型 LLaMA 3.2 / Gemma 3.2/4.0 语言模型

**📊 数据集**

使用的数据集为 CICIOT 2023，包含 500 例每种 5 类 DDoS 攻击（ICMP、UDP、TCP、PSH/ACK、RST/FIN）和 500 例正常流量

**📈 对比分析**

通过对比不同提示策略（无 KB、短 KB、CoT、One‑Shot、Few‑Shot）评估宏平均 F1 分数；实验表明 Few‑Shot RAG 在 LLaMA 3.2 3B 与 Gemma 3.2 4B 上分别达到了 0.75 与 0.77 的宏平均 F1，远超单纯 CoT 或无 KB 的模型

**⚠️ 局限性**

局限性包括：对模型规模高度敏感，较小模型仍难以达到大模型水平；检索组件需要预先构建知识库；实验主要针对结构化网络特征，未验证对更复杂或混合攻击场景的泛化能力

---

## 30. Learning Consistent Taxonomic Classification through Hierarchical Reasoning

**arXiv ID:** 2601.14610 | [PDF](https://arxiv.org/pdf/2601.14610v1)

**作者:** Zhenghong Li `[一作]` (Stony Brook University), Haibin Ling `[通讯]` (Stony Brook University)

**通讯引用:** 35154 | [OpenAlex ID](https://openalex.org/A5061469520)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 VL-Taxon 两阶段层次推理框架，结合监督微调和基于 GRPO 的强化学习，提升 VLM 在物种分类中的层次一致性与叶节点准确率。

**💡 创新点**

创新点包括：① 通过先推断叶节点再进行层级一致性校正的两阶段推理；② 采用层级化监督微调 + GRPO 强化学习的混合训练策略；③ 利用叶节点作为先验来约束上层类别，显著提升层次一致性。

**🔧 技术方法**

技术手段：Qwen2.5-VL-7B 视觉语言模型、LoRA 微调、GRPO 强化学习、top‑down 层次推理、开放式多选问答格式、格式与准确度双重奖励。

**📊 数据集**

使用 iNaturalist‑2021 子集（植物 3,771 种 × 10 张图）进行微调，评估数据集包括 iNat21‑Animal、iNat21‑Plant 以及 CUB‑200。

**📈 对比分析**

与 LLaVA‑OV‑7B、InternVL2.5‑8B、InternVL3‑8B、Qwen2.5‑VL‑7B/32B/72B、LoRA 微调模型及 GPT‑4o 对比，VL‑Taxon 在 iNat21‑Plant 上 HCA 提升 45% 以上、叶节点准确率提升 30% 以上，且在 7B 模型上实现与 72B 对等甚至更优的性能。

**⚠️ 局限性**

局限性：依赖于有限的植物子集训练，可能在更大类群或不同领域的推理中表现不佳；RL 训练复杂且对超参数敏感；仅在层次一致性与叶节点准确率上评估，缺乏对解释性或可扩展性的进一步验证。

---

## 31. Self-Blinding and Counterfactual Self-Simulation Mitigate Biases and Sycophancy in Large Language Models

**arXiv ID:** 2601.14553 | [PDF](https://arxiv.org/pdf/2601.14553v1)

**作者:** Brian Christian `[一作]` (University of Oxford), Matan Mazor `[通讯]` (University of Oxford)

**通讯引用:** 386 | [OpenAlex ID](https://openalex.org/A5065480857)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过让大语言模型（LLM）调用自身的API，模拟并执行对“被遮蔽”条件下的决策，从而实现对性别、种族偏见和顺从偏见（sycophancy）的减轻。

**💡 创新点**

创新点在于：① 将自我模拟提升为可执行的“自我调用”技术；② 利用LLM自身的完整、真实输出作为其对照，构成一种“内在对照法”实现可解释且公平的决策；③ 证明传统的提示干预往往无效甚至加剧偏见。

**🔧 技术方法**

技术手段包括：自定义工具（run_counterfactual_simulation）实现LLM对自身的调用；对输入进行精确重写（剔除性别/种族或用户身份信息）；对模型输出进行对比分析（logit差异、绝对误差）。

**📊 数据集**

数据集：
- 520个带有性别/种族标签的决策场景（从Tamkin等改编，覆盖65个情境×2性别×4种族），
- 60个二方争议情境（15类×4次呈现方式），共240个提示，用于测量顺从偏见。数据已公开于GitHub。

**📈 对比分析**

对比方法：将模型在默认、提示干预（“不要歧视”“忽略”“如果不知道”）与自我调用两种方案下的输出与“完全遮蔽”基线进行比较。结果显示：自我调用使绝对误差下降至接近0（例如GPT‑4.1从3.85降至0.42），偏见系数几乎为零；提示干预仅部分缓解，甚至有时会加剧偏差。

**⚠️ 局限性**

局限性：
- 需要额外推理步骤，导致推理延迟和计算成本上升；
- 自我调用过程不完美，模型有时仍泄露被遮蔽信息或忽略关键信息；
- 仅在“遮蔽/重写”场景验证，尚需探索更复杂的偏见类型和多模态数据；
- 对抗性偏见仍存在，部分模型在获得遮蔽输出后主动覆盖，表明存在意识到的偏见。

---

## 32. Exploring Performance-Productivity Trade-offs in AMT Runtimes: A Task Bench Study of Itoyori, ItoyoriFBC, HPX, and MPI

**arXiv ID:** 2601.14608 | [PDF](https://arxiv.org/pdf/2601.14608v1)

**作者:** Torben R. Lahnor `[一作]` (University of Kassel), Patrick Diehl `[通讯]` (Los Alamos National Lab)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将 Itoyori 与 ItoyoriFBC 这两个新型集群 AMT 运行时集成到 Task Bench 框架中，并与 MPI 与 HPX 进行性能与生产力的对比评测。

**💡 创新点**

首次在 Task Bench 上系统地评估 Itoyori/IotyoriFBC，提出公平对比的实验设计，并改进 HPX 的实现以消除瓶颈。

**🔧 技术方法**

使用 Task Bench 生成的合成任务图、MPI、HPX、Itoyori、ItoyoriFBC 运行时、RDMA 工作窃取、PGAS 缓存、Future 同步等技术。

**📊 数据集**

使用 Task Bench 合成的任务图数据集（stencil、FFT、dense 等），在 Goethe‑NHR 超算上进行实验。

**📈 对比分析**

通过应用效率（Application Efficiency）和最小有效任务粒度（METG）评估性能；通过代码行数（LOC）和库构造数（NLC）评估生产力。MPI 在规则、通信轻量任务下效率最高；Itoyori 在通信密集型任务中效率最高并且生产力最佳；HPX 在负载不平衡场景表现最稳健。

**⚠️ 局限性**

局限性：Itoyori 在细粒度任务下因 RDMA 工作窃取产生高开销；ItoyoriFBC 需对 Future 进行多次包裹导致额外开销；HPX 的性能受通信拓扑影响；实验受内存与 MPI 标签空间限制。

---

## 33. On the Runway Cascade of Transformers for Language Modeling

**arXiv ID:** 2601.14522 | [PDF](https://arxiv.org/pdf/2601.14522v1)

**作者:** Hunjae Lee `[一作]` (Southern Methodist University), Corey Clark `[通讯]` (Southern Methodist University)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5079343448)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出了 Transformer 的跑道级联（runway cascade）问题，并设计了跑道感知重连（runway-aware rewiring）机制，解决了间接路径信息冗余与误配的瓶颈；

**💡 创新点**

创新点在于将 Transformer 视为图神经网络，理论揭示间接路径对信息传播的误配，并引入基于跑道系数的软重连方法，既能补偿信息泄漏，又不增加额外参数；

**🔧 技术方法**

采用 GNN 理论分析、Softmax 连通性证明、跑道系数计算（dot‑product 或 bilinear）以及软重连的注意力重构技术；

**📊 数据集**

使用 C4 语料库训练模型，并在 ARC、HellaSwag、PIQA、CommonsenseQA、以及 Passkey 信息检索实验中进行评估；

**📈 对比分析**

在语言建模、推理、检索与外推任务中与标准 Transformer 对比，验证 perplexity 降低、推理准确率提升、检索准确率提升、外推性能更佳，且无额外参数；

**⚠️ 局限性**

局限性在于对早期 token 的削弱效应仍存在，U‑曲线问题未完全消除，且在极大上下文下仍需更大容量模型，部分易共现推理任务表现略逊。

---

## 34. Layer-adaptive Expert Pruning for Pre-Training of Mixture-of-Experts Large Language Models

**arXiv ID:** 2601.14327 | [PDF](https://arxiv.org/pdf/2601.14327v1)

**作者:** YuanLab. ai `[一作]`, Allen Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Layer-Adaptive Expert Pruning (LAEP) 方法，在 MoE LLM 的预训练阶段主动剪枝低利用专家并重新安排专家分布，以提升训练效率和减少参数量。

**💡 创新点**

创新点在于：①在预训练阶段而非后训练阶段实现专家剪枝；②基于层级的自适应剪枝策略；③结合专家重排算法以平衡设备级负载，全部不依赖额外的负载平衡损失。

**🔧 技术方法**

技术手段包括：token 负载统计分析、α、β 两个剪枝阈值、top‑K 最小负载专家筛选、贪心重排算法，以及在 10B、20B、1010B 等规模 MoE 模型上进行实验。

**📊 数据集**

使用的主要数据集：大规模多域预训练语料库（约 1.08 万亿 token），以及在 MMLU、ARC‑Challenge、NaturalQuestions、HumanEval、MBPP、GSM8K、MATH 等标准评测基准上进行下游任务验证。

**📈 对比分析**

与密集模型 LLaMA‑3.1‑405B、DeepSeek‑V3‑Base 以及使用深度学习负载平衡损失的 MoE 模型比较，LAEP 在 1010B 模型中实现了 48.3% 的训练效率提升、33.3% 的参数缩减，并在多数任务上取得与 DeepSeek‑V3‑Base 相当甚至更优的准确率。

**⚠️ 局限性**

局限性：剪枝阈值（α、β）需人工调参；在极大规模模型下的稳定性和泛化性需进一步验证；剪枝后对极端稀疏场景或特定任务的表现尚未完全评估。

---

## 35. Diffusion Epistemic Uncertainty with Asymmetric Learning for Diffusion-Generated Image Detection

**arXiv ID:** 2601.14625 | [PDF](https://arxiv.org/pdf/2601.14625v1)

**作者:** Yingsong Huang `[一作]` (Tencent Inc), Qi Xiong `[通讯]` (Tencent Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种利用扩散模型的本征不确定性与非对称学习相结合的检测框架DEUA，用于识别扩散生成的图像。

**💡 创新点**

首次通过拉普拉斯近似估计扩散过程的本征不确定性，将其作为判别特征，并设计非对称对比学习损失以缓解“sink class”问题，从而提升检测的泛化能力。

**🔧 技术方法**

使用拉普拉斯近似（最后层）、CLIP视觉特征、多头注意力、非对称对比损失，以及扩散重构错误与不确定性度量等技术。

**📊 数据集**

在大规模公开数据集GenImage和DRCT-2M上进行评估。

**📈 对比分析**

与F3Net、GramNet、UnivFD、DIRE、LaRE2、DRCT等主流方法对比，DEUA在多种生成器和跨数据集的准确率和AP均显著提升，最高可达90.5% ACC/高AP，优于现有方法6-8%的提升。

**⚠️ 局限性**

对GAN生成的图像（如BigGAN）仍表现不足，且对DR变体的鲁棒性有限；方法依赖扩散重构模型与拉普拉斯近似，计算成本较高。

---

## 36. SCSimulator: An Exploratory Visual Analytics Framework for Partner Selection in Supply Chains through LLM-driven Multi-Agent Simulation

**arXiv ID:** 2601.14566 | [PDF](https://arxiv.org/pdf/2601.14566v1)

**作者:** Shenghan Gao `[一作]` (ShanghaiTech University), Quan Li `[通讯]` (ShanghaiTech University)

**通讯引用:** 1377 | [OpenAlex ID](https://openalex.org/A5100689370)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 SCSimulator，一个基于大语言模型（LLM）驱动的多智能体仿真与可视化分析框架，用于支持供应链合作伙伴选择的探索与决策。

**💡 创新点**

创新点包括：①将 LLM 的 Chain‑of‑Thought 推理与 SHAP 解释相结合，提供多维度可解释决策；②设计人机交互的迭代调节机制，支持专家随时介入并记录模拟路径；③采用时间线型 Ego‑network 可视化（berry 代称）直观展示供应商‑客户动态；④构建可追踪的模拟路径视图，实现分支与回溯分析。

**🔧 技术方法**

使用技术：LLM（ChatGPT‑4o、Gemini‑2.5‑flash、Qwen3‑VL‑Flash）驱动多智能体；CoT 推理与 RAG；VAR/变分图自动编码器嵌入；线性、Lasso、随机森林、梯度提升等回归模型；SHAP 解释；可视化技术（时间线、线条、 berry/玫瑰图标）。

**📊 数据集**

数据集：基于中国纸包装行业的 35 家企业，覆盖 2023‑2024 共 8 个季度的供应链关系与企业属性（行业、运营、技术、声誉等指标）。

**📈 对比分析**

比较方法：对三种 LLM 进行 80 次独立仿真，评估 ACC、Precision、Recall、F1、Gwet’s AC1 与一致性比率；与专家 Likert 量表对比，展示系统可解释性与实用性；实验结果显示 ChatGPT‑4o 在 ACC≈72%、F1≈63% 及一致性高于其他模型，专家评分均在 4–5 级，证明系统在探索性决策中表现良好。

**⚠️ 局限性**

limitations：①受 LLM 调用成本和推理延迟限制，当前仅支持 ≤50 代理规模；②仿真结果为探索性预测，缺乏统计显著性；③LLM 可能产生幻觉或不符合实际的行为，需要人工监督；④缺乏大规模真实案例验证与长期跟踪评估；⑤未与传统数学/优化方法做直接对比，难以量化预测准确性。

---

## 37. LFS: Learnable Frame Selector for Event-Aware and Temporally Diverse Video Captioning

**arXiv ID:** 2601.14594 | [PDF](https://arxiv.org/pdf/2601.14594v1)

**作者:** Lianying Chao `[一作]` (Huawei Technologies Co., Ltd.), Kai Zhang `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可学习的帧选择器LFS，能够在不修改视频‑LLM参数的情况下，自动挑选时间多样且事件相关的关键帧，并通过冻结的video‑LLM字幕反馈进行监督；

**💡 创新点**

①在时间维度引入事件感知的重要性模型与分层Top‑K选取，实现时间覆盖与多样性并存；②利用字幕质量直接指导帧选择的caption‑guided supervision；③构建与人类认知一致的ICH‑CC基准来评估细粒度字幕质量；

**🔧 技术方法**

Temporal Scoring Network（TSNet）与时间卷积、门控；分层Top‑K选帧；熵正则化；相对字幕损失；冻结Long‑CLIP视觉编码器与Qwen3‑VL‑8B文字生成器；

**📊 数据集**

训练集：WebVid‑10、TGIF、Charades、YouCook2、TREC‑VTT；评测集：ICH‑CC（中英各500问/100段视频）、VDC、Dream‑1K；零-shot QA基准：MVBench、VideoMME、MLYU‑MCQ、VideoMMMU；

**📈 对比分析**

与多种开源video‑LLM基线（AuroraCap‑7B、Qwen2.5‑VL‑7B、Qwen3‑VL‑8B等）在统一帧预算下进行对比。LFS在ICH‑CC上提升≈+4–5%准确率，在VDC细节类提升≈+2%准确率，Dream‑1K F1保持不变但召回略升。零-shot视频QA中，Qwen3‑VL‑8B提升≈+2–3%准确率；

**⚠️ 局限性**

①在极短视频（<10s）如Dream‑1K提升有限；②对冻结字幕生成器的偏差敏感；③训练仍需GPU资源；④未在多模态对话或实时推理环境下充分验证。

---

## 38. XD-MAP: Cross-Modal Domain Adaptation using Semantic Parametric Mapping

**arXiv ID:** 2601.14477 | [PDF](https://arxiv.org/pdf/2601.14477v1)

**作者:** Frank Bieder `[一作]` (FZI Research Center for Information Technology), Christoph Stiller `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 22681 | [OpenAlex ID](https://openalex.org/A5091574711)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 XD-MAP，一种跨模态自监督域适应框架，将前视相机训练得到的语义标签通过语义参数化高清地图映射到 LiDAR，生成无人工标注的 360° LiDAR 语义分割、全景分割及 3D 语义分割伪标签。

**💡 创新点**

创新点在于：① 通过柱子、交通灯等静态对象的几何原语（圆柱体、平面）构建精确的语义参数化地图，充当源域与目标域之间的桥梁；② 不要求传感器重叠或模态相似性，能够扩展视角；③ 生成结构化伪标签支持 2D/3D 语义及全景分割。

**🔧 技术方法**

核心技术包括：预训练的图像语义检测网络（基于 Mapillary Vistas），特征点 SLAM 构建高精度地图；对每个语义类别拟合对应的几何原语；球面投影模型将原语投影至 LiDAR 视角；伪标签生成（投影+深度）并对 2D/3D 任务进行训练；对比单帧基线并使用 Mask2Former、Cylinder3D 等模型。

**📊 数据集**

数据集：自采集的 5 条 Karlsruhe 路段（共 21.58 km、20 471 帧）配备 Velodyne Alpha Prime（128×1812）和 1536×4096 RGB 相机；源域语义检测使用公开的 Mapillary Vistas 训练；目标域为上述 LiDAR 数据。

**📈 对比分析**

实验将 XD-MAP 与单帧基线（XD‑B1、XD‑B2）对比。XD‑MAP 在 2D 语义分割 mIoU 提升 +19.5，2D 全景 PQ_th 提升 +19.5，3D 语义分割 mIoU 提升 +32.3；在不同检测范围、采样频率和运动补偿的 ablation 研究中均保持优势，表明方法在多任务上具备显著性能提升。

**⚠️ 局限性**

局限性：① 需高精度 SLAM 与多模态标定，且传感器在时间上观测相同对象；② 仅能处理可用几何原语的静态结构化对象，难以处理动态或非几何对象；③ 对传感器噪声与标定误差敏感；④ 缺乏人工 ground truth，评估基于其生成的伪标签；⑤ 对非道路类或不同环境的泛化尚未验证。

---

## 39. Maximum Edge-based Quasi-Clique: Novel Iterative Frameworks

**arXiv ID:** 2601.14619 | [PDF](https://arxiv.org/pdf/2601.14619v1)

**作者:** Hongbo Xia `[一作]` (Harbin Institute of Technology), Zhaoquan Gu `[通讯]` (Pengcheng Laboratory)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种新型迭代框架，用于求解最大边缘基γ-准团（Maximum Edge-based γ-Quasi-Clique）问题；

**💡 创新点**

创新点在于将非继承性质的问题转化为一系列继承子问题，采用自下而上的翻倍迭代框架和动态精炼启发式，从而大幅提升剪枝与约简效率，理论复杂度从O(2^n)下降至O(β_κ^n)；

**🔧 技术方法**

核心技术包括：k-缺陷团（k-defective clique）与γ-EQC的关系映射、双向迭代（top‑down与bottom‑up）、翻倍搜索策略、基于退化序列的动态启发式（FMainDegen和DRefine）；

**📊 数据集**

实验数据集涵盖了网络存储库中的253个真实世界图（139个现实图 + 114个Facebook社交网络），以及进一步挑选的20个大规模图；

**📈 对比分析**

与两种最先进的基准（branch‑and‑bound算法和γ-EQC枚举算法）比较，本文方法在3小时和300秒限制下均能显著提高可解决实例数，并在大规模图上实现多达四个数量级的加速；

**⚠️ 局限性**

局限性主要体现在：对于极低的γ（即松散约束）时，搜索空间急剧膨胀，导致剪枝效果减弱，性能优势收窄；此外，启发式部分仍可能受限于退化序列的结构，难以突破某些特殊图结构的瓶颈。

---

## 40. Large Language Model-Powered Evolutionary Code Optimization on a Phylogenetic Tree

**arXiv ID:** 2601.14523 | [PDF](https://arxiv.org/pdf/2601.14523v1)

**作者:** Leyi Zhao `[一作]` (Indiana University), Xuhong Zhang `[通讯]` (Indiana University)

**通讯引用:** 1760 | [OpenAlex ID](https://openalex.org/A5047459900)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 PhyloEvolve 系统，利用 LLM 代理通过轨迹学习自动化 GPU 科学计算算法的优化。

**💡 创新点**

创新点在于将 GPU 优化转化为轨迹条件的 In‑Context RL，结合进化树、精英池、跨族传递和多岛并行实现无模型更新的轨迹重用，显著降低重启探索成本。

**🔧 技术方法**

使用技术包括 LLM 多代理（NextStepper、ModifyAgent、Designer、Summarizer）、算法蒸馏与提示决策变换器、Phylogenetic Tree、Elite Pool、容器化执行以及 ICRL 原理。

**📊 数据集**

数据集涵盖三类科学计算工作负载：Landau–Lifshitz–Gilbert PDE 求解器、Local Tangent Space Alignment（LTSA）维度约减、GraphWave 结构嵌入，全部在 NVIDIA A40 GPU 上评测。

**📈 对比分析**

对比基线实现和进化搜索变体，使用壁钟时间、内存占用和数值正确性进行评估；在三任务中均实现多倍速度提升（如 LLG 在大批量时可达 4–6 倍加速），同时保持或降低内存使用并保证结果正确性。

**⚠️ 局限性**

局限包括：仅依赖经验评估缺乏硬件成本模型，LLM 产生低质量改动时需要额外过滤；仅验证 NVIDIA GPU，缺乏跨平台与多 GPU 扩展；未提供理论分析与正式验证，导致正确性保障与样本效率的进一步提升空间。

---

## 41. From Volumes to Slices: Computationally Efficient Contrastive Learning for Sequential Abdominal CT Analysis

**arXiv ID:** 2601.14593 | [PDF](https://arxiv.org/pdf/2601.14593v1)

**作者:** Po-Kai Chiu `[一作]` (National Central University), Hung-Hsuan Chen `[通讯]` (National Central University)

**通讯引用:** 843 | [OpenAlex ID](https://openalex.org/A5078925594)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并验证了一种将3D自监督对比学习框架VoCo改造成2D切片级自监督预训练方法（2D-VoCo），并将预训练的CNN骨干网络与Bi‑LSTM相结合，用于多器官腹部创伤CT分类。

**💡 创新点**

创新点在于：①在保持3D VoCo对空间语义学习能力的同时，完全转为2D切片级训练，显著降低显存与计算成本；②采用学生‑教师EMA结构与三种损失（intra、inter、regularization）使模型能够在无标签CT体积中捕捉空间重叠关系；③结合Bi‑LSTM对切片序列建模，充分利用器官间上下文信息。

**🔧 技术方法**

技术手段包括：2D CNN骨干EfficientNetV2（T/S版）；学生‑教师EMA自监督框架；基于IoU的相似度目标和三项对比损失；Bi‑LSTM+多头分类头；在RSNA 2023腹部创伤CT数据集上进行下游细调；使用额外的FLARE23无标签CT进行预训练。

**📊 数据集**

数据集：①RSNA 2023 Abdominal Trauma CT（RATIC，3,147例，含肾、肝、脾三器官的三类标签）；②FLARE23无标签CT（用于扩充预训练数据）。

**📈 对比分析**

与仅使用ImageNet预训练的基线模型比较，2D-VoCo预训练在RSNA分数、mAP、精确率和召回率均显著提升（例如RSNA分数从0.413下降到0.378，mAP从59.47提升至60.98，召回率从49.37提升至53.90）。加入FLARE23无标签数据进一步提升表现；多器官联合分类比单器官更优，证明模型已学会利用全腹部上下文。

**⚠️ 局限性**

局限性包括：①实验仅在CT数据上验证，未评估跨模态或其他器官的通用性；②缺乏对模型可解释性的深入分析；③尽管2D化降低成本，但仍需GPU显存和训练时间；④未针对极少标签或小样本场景做额外验证。

---

## 42. Multi-Partner Project: COIN-3D -- Collaborative Innovation in 3D VLSI Reliability

**arXiv ID:** 2601.14347 | [PDF](https://arxiv.org/pdf/2601.14347v1)

**作者:** George Rafael Gourdoumanis `[一作]` (University of Thessaly), George Floros `[通讯]` (Trinity College Dublin)

**通讯引用:** 622 | [OpenAlex ID](https://openalex.org/A5066445455)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了面向 2.5D/3D VLSI 系统的开源 EDA 工具和方法论，涵盖从物理实现到系统层面的可靠性建模与优化。

**💡 创新点**

创新点包括：① 3DPX 作为首个基于 OpenROAD 的分层 3D 设计探索框架；② 结合物理电迁移（EM）模型与 XGBoost 机器学习的混合可靠性评估；③ 通过工作负载驱动的 PDN 生成与双向耦合的系统级热可靠性分析。

**🔧 技术方法**

技术手段主要包括 OpenROAD、Open3DBench、TCL/Python API、Korhonen 电迁移扩散模型、XGBoost、McPAT、gem5/HotSniper 等仿真与优化工具。

**📊 数据集**

使用标准 IC 设计基准（如 OpenROAD 流程验证基准）、公开的功耗与温度时序数据以及自定义的多层金属布局与 TSV 设计案例。

**📈 对比分析**

与传统单层或商业 3D 工具相比，3DPX 在电迁移压力均匀性上提升约 20%–30%，IR‑drop 降低 10% 以上，并能在布局前期即完成可靠性评估，显著减少迭代次数；系统级热模型在实际工作负载下预测寿命误差低于 5%。

**⚠️ 局限性**

局限性包括：① 对商业 CAD 生态的依赖仍有限，需进一步兼容现有商用工具；② 物理模型与 ML 预测的精度受限于训练数据的覆盖范围；③ 现阶段仅支持有限的 3D 连接范式（F2F/F2B/B2B），需扩展至更复杂的混合堆叠结构。

---

## 43. Chain-of-Memory: Lightweight Memory Construction with Dynamic Evolution for LLM Agents

**arXiv ID:** 2601.14287 | [PDF](https://arxiv.org/pdf/2601.14287v1)

**作者:** Xiucheng Xu `[一作]` (University of Chinese Academy of Sciences), Huawei Shen `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 6646 | [OpenAlex ID](https://openalex.org/A5047897879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量化构建、链式内存动态演化的Chain-of-Memory框架，用以提升LLM代理的长时记忆推理能力。

**💡 创新点**

核心创新在于用动态链式演化将检索片段组织成连贯推理路径，并通过自适应截断去除噪声，从而实现低成本高效推理。

**🔧 技术方法**

采用扁平化嵌入检索、基于余弦相似度的检索、链式演化机制（门控评分）、自适应截断以及LLM生成与评判。

**📊 数据集**

在LongMemEval和LoCoMo两个长记忆QA基准上进行实验。

**📈 对比分析**

与全上下文、Naive RAG、LangMem、A-Mem、Mem0等基线对比，CoM在准确率上提升约7.5–10.4%，而Token消耗和延迟仅为传统复杂结构的2.7%和6.0%。

**⚠️ 局限性**

局限包括依赖嵌入相似度导致检索误差、截断可能丢失细节、仅适用于文本，未扩展至多模态。

---

## 44. Legal Retrieval for Public Defenders

**arXiv ID:** 2601.14348 | [PDF](https://arxiv.org/pdf/2601.14348v1)

**作者:** Dominik Stammbach `[一作]` (Princeton University), Peter Henderson `[通讯]` (Princeton University)

**通讯引用:** 13054 | [OpenAlex ID](https://openalex.org/A5049073875)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

与新泽西州公共辩护办公室合作，开发了NJ BriefBank检索工具，收集并发布了公开的公共辩护检索数据集（PD dataset）。

**💡 创新点**

创新点包括：① 针对公共辩护内部上诉简报的检索任务提出并构建真实查询与人工标注的PD数据集；② 通过IRAC框架的查询扩展、精细合成数据与领域适配显著提升检索质量；③ 证明现有法律检索基准与公共辩护场景存在分布不匹配。

**🔧 技术方法**

使用技术包括：嵌入检索模型（e5、Qwen3、NV-Embed等）、检索‑重排序链路、LLM语义分块、IRAC式查询扩展、精细合成数据生成与过滤、领域适配（现代BERT）以及LLM生成的摘要。

**📊 数据集**

数据集：内部NJ OPD 2896份简报、168份内部文件、351份公共指令；公开PD数据集 170条真实查询、96,032条段落、543条人工标注相关段落。

**📈 对比分析**

性能对比：零样本下最大模型NV-Embed Recall@5≈53%；在对PD数据集微调后，e5‑large 加上合成数据和IRAC扩展的Recall@5提升至约37%；相比之下，在BarExam‑QA、LePaRD等传统基准上的表现下降，显示域差异。

**⚠️ 局限性**

限制：数据隐私导致只能公开部分数据；检索Recall@5仍仅为约37%，未达到实用水平；生成式回答仍存在幻觉、缺乏可验证性；现有基准与真实公共辩护任务仍存在分布偏差。

---

## 45. The Algorithmic Barrier: Quantifying Artificial Frictional Unemployment in Automated Recruitment Systems

**arXiv ID:** 2601.14534 | [PDF](https://arxiv.org/pdf/2601.14534v1)

**作者:** Ibrahim Denis Fofanah `[一作]` `[通讯]` (Pace University), Ibrahim Denis Fofanah (Pace University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将招聘视为“人工智能语义翻译”问题，构建了名为JobOS的候选人侧工作流架构，以缓解自动化招聘系统导致的人工摩擦失业。

**💡 创新点**

创新点在于将语义向量嵌入与可解释性验证结合，形成可插拔的中间层，既提升召回率又保持精确率，同时保障候选人数据所有权与隐私。

**🔧 技术方法**

采用Transformer‑based上下文嵌入（如BERT）生成高维向量，利用余弦相似度进行匹配，并在候选人侧加入基于任务的验证与语音转写评估。

**📊 数据集**

实验使用1,000条人工合成的简历–职位对，人工引入同义词、缩写与岗位标题变化以模拟真实语言多样性。

**📈 对比分析**

与传统基于关键词的ATS做对比，JobOS将召回率从0.45提升至0.92、精确率从0.62提升至0.89，F1分数提升73%，同时在阈值和词汇噪声下表现出更高的鲁棒性。

**⚠️ 局限性**

主要局限包括：使用合成数据而非真实ATS日志，未评估下游面试与雇佣结果，以及潜在的战略性简历优化导致的模型适应风险。

---

## 46. Predicting Retrieval Utility and Answer Quality in Retrieval-Augmented Generation

**arXiv ID:** 2601.14546 | [PDF](https://arxiv.org/pdf/2601.14546v1)

**作者:** Fangzheng Tian `[一作]` (University of Glasgow), Craig Macdonald `[通讯]` (University of Glasgow)

**通讯引用:** 8677 | [OpenAlex ID](https://openalex.org/A5057643560)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文定义了两类RAG性能预测任务：检索性能预测（RPP）和生成性能预测（GPP），并提出将检索相关性、上下文与答案的困惑度以及文档可读性等多种信号通过线性回归组合以提高预测精度。

**💡 创新点**

创新点在于：①将现有查询性能预测（QPP）方法迁移到RAG中作为预检索特征；②引入读者视角的上下文困惑度和答案困惑度作为新的预测特征；③结合查询无关的文档质量与可读性指标，形成多源特征融合的预测框架；④系统评估并证明组合特征优于单一特征。

**🔧 技术方法**

采用线性回归集成模型，使用的特征包括：QPP方法（NQC、MaxScore、Dense-QPP、A‑Pair‑Ratio、BERT‑QPP）、上下文/答案困惑度（P_c、P_a）、文档可读性与质量指标（Dale‑Chall、Spache、Flesch‑Kincaid、Gunning Fog、QualT5）。

**📊 数据集**

在开放域问答数据集Natural Questions（NQ）上进行实验，检索配置涵盖BM25、BM25+MonoT5、E5三种检索器，RAG上下文规模从k=2到10不等。

**📈 对比分析**

与单一QPP或困惑度特征相比，组合特征在RPP和GPP上均取得更高的Spearman相关性（最高至0.39），且对不同检索器与上下文长度都有稳健提升；特别是加入答案困惑度后，GPP的相关性提升显著。

**⚠️ 局限性**

局限性包括：①预测模型仍依赖大量标注数据（需要检索与答案质量对比）；②困惑度特征需生成答案，导致实时性受限；③实验仅覆盖NQ+Llama‑3‑8B，未验证在其他任务或大模型上的泛化；④文档质量与可读性指标对LLM的影响有限，需进一步研究。

---

## 47. A Survey of Security Challenges and Solutions for Advanced Air Mobility and eVTOL Aircraft

**arXiv ID:** 2601.14415 | [PDF](https://arxiv.org/pdf/2601.14415v1)

**作者:** Mahyar Ghazanfari `[一作]` (George Washington University), Isaac Amundson `[通讯]` (Collins Aerospace)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了先进空中交通（AAM）/eVTOL 系统的安全威胁与防御，系统性地对齐了从GNSS、通信、感知到云服务等多维攻击面，并提出了端到端的安全架构与防御策略。

**💡 创新点**

创新点在于：①将传统民航与无人机安全漏洞映射至AAM领域，形成专门的攻击与防御分类；②提出跨域分割、加密认证与跨感知验证的统一安全框架；③阐述了云服务与数据接口的系统性风险，指出现有标准缺失。

**🔧 技术方法**

使用的技术包括威胁建模与STRIDE、加密认证（PKI、mTLS、OAuth2）、多源感知融合（GNSS+INS+LiDAR+视觉+MLAT）、对抗机器学习防御（PatchGuard、AugMix、STRIP）、信号级特征校验（RSSI、TDoA、AoA）以及基于容错的系统架构设计。

**📊 数据集**

主要参考公开航空数据集与流：ADS‑B/Mode S 流、GNSS信号与失效模拟、LiDAR/视觉数据集、5G/ATC频谱测试记录；未提出专门的自研数据集。

**📈 对比分析**

通过仿真与案例实验评估防御效果，展示了跨感知一致性检验与加密认证能显著降低攻击成功率（如ADS‑B伪造误报率从高至低），但报告中缺少统一的量化指标与跨平台可复现实验。

**⚠️ 局限性**

局限性包括：缺乏统一的评估基准与可重复实验；云服务与感知攻击的实测验证不足；安全架构仍处于概念层面，未进行系统级安全验证与法规对接。

---

## 48. Epistemic Constitutionalism Or: how to avoid coherence bias

**arXiv ID:** 2601.14295 | [PDF](https://arxiv.org/pdf/2601.14295v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 49. DiSPA: Differential Substructure-Pathway Attention for Drug Response Prediction

**arXiv ID:** 2601.14346 | [PDF](https://arxiv.org/pdf/2601.14346v1)

**作者:** Yewon Han `[一作]` (Dongguk University), Sangsoo Lim `[通讯]` (Dongguk University)

**通讯引用:** 716 | [OpenAlex ID](https://openalex.org/A5101655230)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了DiSPA，一个双向差分子结构-通路注意力框架，用以集成药物子结构与细胞通路基因表达来预测药物敏感性。

**💡 创新点**

创新点在于引入双视角差分交叉注意力，实现化学子结构与通路状态的双向调节，提升对未见药物/细胞的泛化能力，并提供可解释的子结构-通路注意力图。

**🔧 技术方法**

使用ChemBERTa对SMILES进行子结构编码，KEGG通路映射基因表达，双向差分注意力网络及多层感知机回归。

**📊 数据集**

主要数据集为GDSC基准（966个细胞系，270种小分子药物），结合KEGG通路；并在CTRP、空间转录组和单细胞转录组上进行验证。

**📈 对比分析**

与DeepTTA、DRPreter、DIPK、DEERS等基线在随机、细胞盲、药物盲和离散集成四种拆分上对比，DiSPA在RMSE、PCC/SCC上均领先，尤其在药物盲/离散集成拆分中RMSE低至2.45并提高约0.12 PCC。

**⚠️ 局限性**

局限包括仅在单一药理基因组资源上训练、注意力权重未直接证明生物学因果、转移到空间/单细胞数据缺乏真实敏感性验证。

---

## 50. Agentic AI Meets Edge Computing in Autonomous UAV Swarms

**arXiv ID:** 2601.14437 | [PDF](https://arxiv.org/pdf/2601.14437v1)

**作者:** Thuan Minh Nguyen `[一作]` (INRS, University of Quebec), Long Bao Le `[通讯]` (INRS, University of Quebec)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

结合大型语言模型与边缘计算，实现可扩展的无人机群自主搜索救援。

**💡 创新点**

引入多代理LLM框架和边缘服务器双向推理，解决LLM算力与能耗限制，并用反馈机制抑制幻觉。

**🔧 技术方法**

多代理框架（LangGraph/AutoGen）、TinyLLaMA、GPT‑4.1、U‑Net火灾分割、轻量化VLM LLaVA、无线自组织网、边缘服务器等技术。

**📊 数据集**

公开卫星图像（NASA视频）与U‑Net火灾分割数据集，UAV感知数据（RGB+热成像）及无人机仿真场景。

**📈 对比分析**

与基线混合LLM规划方法和贪心分配进行对比，评估覆盖率与任务完成时间；实验表明覆盖率提升约10–15%，任务完成时间缩短约20%。

**⚠️ 局限性**

小型LLM推理准确性与幻觉问题、边缘节点能耗与可靠性、通信不稳定导致协作缺陷，以及缺乏统一评测基准。

---

## 51. The Ontological Neutrality Theorem: Why Neutral Ontological Substrates Must Be Pre-Causal and Pre-Normative

**arXiv ID:** 2601.14271 | [PDF](https://arxiv.org/pdf/2601.14271v1)

**作者:** Denise M. Case `[一作]` (Northwest Missouri State University), Denise M. Case `[通讯]` (Northwest Missouri State University)

**通讯引用:** 135 | [OpenAlex ID](https://openalex.org/A5009909166)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出并证明了在面向可持续责任追溯的多框架环境下，语义中立的本体必须是前因果、前规范的，不能在基础层嵌入因果或规范结论。

**💡 创新点**

给出了“本体中立性定理”，阐明了在多框架共存时，本体设计必须排除因果与规范命题的必要性，提供了严格的设计约束。

**🔧 技术方法**

采用形式化逻辑推理与结构化证明方法，结合层次抽象（LoA/LoO）概念进行理论构建。

**📊 数据集**

未使用任何实验数据集，完全是理论推导。

**📈 对比分析**

未进行实验比较，性能不适用于本研究；其价值在于提供理论约束。

**⚠️ 局限性**

局限在于仅适用于身份条件可预设且不争议的域；若身份本身被争议，则无法构建中立子体。

---

## 52. Holmes: An Evidence-Grounded LLM Agent for Auditable DDoS Investigation in Cloud Networks

**arXiv ID:** 2601.14601 | [PDF](https://arxiv.org/pdf/2601.14601v1)

**作者:** Haodong Chen `[一作]` (Xiamen University), Qiao Xiang `[通讯]` (Xiamen University)

**通讯引用:** 1334 | [OpenAlex ID](https://openalex.org/A5015404751)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 DDoS Detective，一种基于大型语言模型的云端 DDoS 调查代理，采用分层“漏斗”检测、语义证据包以及引证约束实现可追溯的攻击归因。

**💡 创新点**

创新点包括：将 LLM 视为虚拟 SRE 而非直接分类器；在异常窗口才触发高成本推理的分层工作流；将二进制包抽象为可引用、可复现的 Evidence Pack；通过严格的引证规则和结构化推理协议，使模型输出可验证、可审计。

**🔧 技术方法**

技术栈包括：OpenPangu-7B LLM、LangGraph 工作流引擎；sFlow 与接口计数器实现高吞吐量监测；tshark 进行预算化的 PCAP 证据抽取；JSON 输出约束与自我纠错机制；冷却/去重逻辑以降低重复推理。

**📊 数据集**

使用 CICDDoS2019 反射/放大攻击流（DNS、LDAP、SNMP、MSSQL、SSDP/UPnP 等）和脚本生成的合成洪泛流（UDP Flood、SYN Flood、ACK Flood）作为评测数据集。

**📈 对比分析**

通过 replay 模拟，比较模型在每个异常窗口的归因准确率与误判率；实验结果表明对多种攻击家族均能准确归因，误判率低；误判时日志中的证据链可快速定位错误来源。

**⚠️ 局限性**

局限性包括：LLM 在多重语义干扰（如 TCP + HTTP 交织）时易被误导；推理成本仍高，需依赖分层控制；需要人工制定引证规则和推理协议；对完全未知的攻击模式仍可能产生误判。

---

## 53. Stabilizing autoregressive forecasts in chaotic systems via multi-rate latent recurrence

**arXiv ID:** 2601.14487 | [PDF](https://arxiv.org/pdf/2601.14487v1)

**作者:** Mrigank Dhingra `[一作]` (University of Tennessee), Omer San `[通讯]` (University of Tennessee)

**通讯引用:** 5020 | [OpenAlex ID](https://openalex.org/A5085671233)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种多尺度递归隐层的隐式神经预报器MSR‑HINE，用于长时滞的混沌动力学预测；

**💡 创新点**

创新点在于将多速率递归隐层与隐式一阶预测器结合，并通过先验-后验融合与隐藏状态校正实现尺度一致的修正；

**🔧 技术方法**

主要技术包括多层傅里叶/池化编码器、GRU递归隐层、周期卷积U‑Net预测器、门控融合与隐藏校正；

**📊 数据集**

使用 Kuramoto–Sivashinsky (KS) 和 Lorenz–96 (L96) 两个经典混沌系统的数据集；

**📈 对比分析**

与 U‑Net‑AR 与两层 HINE 基线对比，MSR‑HINE 在 KS 400 步时 RMSE 降 62.8%、ACC 提升 85.1%，在 L96 100 步时 RMSE 降 27.0%、ACC 提升 46.8%；

**⚠️ 局限性**

局限在于对极长时延预测仍易失效、需要手动调节多尺度步长、对部分观测或噪声的鲁棒性尚未系统验证。

---

## 54. Log anomaly detection via Meta Learning and Prototypical Networks for Cross domain generalization

**arXiv ID:** 2601.14336 | [PDF](https://arxiv.org/pdf/2601.14336v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 55. Landing-Induced Viscoelastic Changes in an Anthropomimetic Foot Joint Structure are Modulated by Foot Structure and Posture

**arXiv ID:** 2601.14634 | [PDF](https://arxiv.org/pdf/2601.14634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 56. Uma Prova de Conceito para a Verificação Formal de Contratos Inteligentes

**arXiv ID:** 2601.14427 | [PDF](https://arxiv.org/pdf/2601.14427v1)

**作者:** Murilo de Souza Neves `[一作]` (Universidade Estadual de Londrina), Adilson Luiz Bonifacio `[通讯]` (Universidade Estadual de Londrina)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5076819061)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

利用RCL对多方买卖智能合约进行规范化建模、使用RECALL工具进行冲突检测并在发现冲突后进行逻辑修正，随后将经验证的合约手工翻译为Solidity并在Remix IDE中进行功能验证

**💡 创新点**

首次将RCL与RECALL工具应用于多方智能合约的前置验证，展示了通过形式化语言显式指定责任主体来识别并消除合约冲突的可行性，证明了形式化验证对提升合约安全性的关键作用

**🔧 技术方法**

RCL（Relativized Contract Language）、RECALL工具、Solidity、Remix IDE

**📊 数据集**

无公开数据集，本文以一个单一的买卖合同案例作为实验对象

**📈 对比分析**

通过对比未修正与已修正合约在Solidity中的执行情况，证明在修正后合约能够顺利完成所有阶段，原始合约在deliverProduct阶段因冲突停滞，显示验证前后功能性提升；未涉及性能数值评估

**⚠️ 局限性**

手工翻译过程缺乏自动化，验证仅针对单一案例，RCL与Solidity之间的映射仍需人工介入，缺乏通用翻译工具，实验范围有限

---

## 57. Rethinking Reinforcement fine-tuning of LLMs: A Multi-armed Bandit Learning Perspective

**arXiv ID:** 2601.14599 | [PDF](https://arxiv.org/pdf/2601.14599v1)

**作者:** Xiao Hu `[一作]` (University of Science and Technology of China), Jianyu Han `[通讯]` (IFlyTek)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一套从最小化配置（单样本、单回合、奖励即学习信号）出发的bottom‑up实验流程，系统评估强化微调中各设计选择（优势函数、回合数、奖励方式、数据难度、基模型）对训练与泛化的影响。

**💡 创新点**

通过将LLM强化微调映射为极大离散动作空间的多臂赌博机问题，提出最小化实验框架并逐层展开，能够分离并量化多重混杂因素的作用，揭示优势函数、回合数等并非关键瓶颈，指出极难样本与负奖励导致泛化失败。

**🔧 技术方法**

使用基于GRPO的策略梯度优化（含/不含优势函数）、多回合采样、负奖励实验，并结合多臂赌博机理论分析；在VeRL框架下实现训练；采用单样本训练、Pass@1评估。

**📊 数据集**

三大LLM（LLaMA‑3.2‑1B‑Instruct、OLMo‑2‑0425‑1B‑Instruct、Qwen2.5‑1.5B‑Instruct）在两大数学推理基准MATH和GSM8K的单例数据集（按难度挑选）上进行实验。

**📈 对比分析**

以最小化配置为基准，逐步加入优势函数、增大回合数、改变奖励、使用极难样本等，比较训练Pass@1与测试Pass@1的提升；发现单样本训练可将测试Pass@1提升最高0.5，但优势函数与回合数并未显著改善泛化；负奖励导致训练失败；OLMo在训练上可达1但不泛化。

**⚠️ 局限性**

只考虑单样本、单回合或少量回合的极简设定，未覆盖批量、状态转移等实际应用；实验仅在两数学推理数据集，缺乏跨任务验证；未深入探索模型内部机制与奖励设计的相互作用。

---

## 58. Hallucination-Free Automatic Question & Answer Generation for Intuitive Learning

**arXiv ID:** 2601.14280 | [PDF](https://arxiv.org/pdf/2601.14280v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 59. Gradient Structure Estimation under Label-Only Oracles via Spectral Sensitivity

**arXiv ID:** 2601.14300 | [PDF](https://arxiv.org/pdf/2601.14300v1)

**作者:** Jun Liu `[一作]` (State Key Laboratory of Internet of Things for Smart City, University of Macau), Jiantao Zhou `[通讯]` (State Key Laboratory of Internet of Things for Smart City, University of Macau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对硬标签黑盒攻击，提出DPAttack框架，包括基于频率先验的初始化和模式驱动的优化。

**💡 创新点**

创新点在于将频谱敏感性分析与空间结构搜索相结合，实现更高梯度符号估计效率，并给出理论保证。

**🔧 技术方法**

采用块离散余弦变换（BDCT）频率采样、低通滤波、Pattern‑Driven Optimization (PDO) 与零阶查询搜索等技术。

**📊 数据集**

使用CIFAR‑10、ImageNet、ImageNet‑C、ObjectNet、COCO、SA1B、CLIP等多种数据集进行评估。

**📈 对比分析**

与HRayS、ADBA、HSJA等现有方法对比，DPAttack在攻击成功率和查询效率上均优越，尤其在低查询预算下显著提升。

**⚠️ 局限性**

局限在于仍需多次查询完成动态块尺寸选择，且对超参数如块尺寸的敏感性需要进一步自动化。

---

## 60. GPU-accelerated simulated annealing based on p-bits with real-world device-variability modeling

**arXiv ID:** 2601.14476 | [PDF](https://arxiv.org/pdf/2601.14476v1)

**作者:** Naoya Onizawa `[一作]` (Research Institute of Electrical Communication, Tohoku University), Takahiro Hanyu `[通讯]` (Research Institute of Electrical Communication, Tohoku University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一款GPU加速的开源p-bit模拟退火（pSA）框架，并通过在MTJ设备的时序、强度、偏移三种变异因素下进行建模，评估其在大规模MAX‑CUT问题上的性能；

**💡 创新点**

在传统pSA基础上提出并验证了两种改进算法——时间平均pSA（TApSA）和停滞pSA（SpSA），证明其对设备变异更具鲁棒性，且某些变异（如时序波动）可反而提升算法效率；

**🔧 技术方法**

使用CUDA实现GPU并行计算，结合Python/PyCUDA调用，并在NVIDIA RTX 4090上进行加速；

**📊 数据集**

使用G‑set图集中的MAX‑CUT基准图（G1至G81，节点数800–20,000）作为测试数据集；

**📈 对比分析**

通过与单线程CPU实现的pSA、TApSA、SpSA进行对比，GPU版本在相同迭代次数下实现两阶数量级的加速；在变异实验中，SpSA和TApSA保持高Normalized Mean Cut值，而pSA对变异高度敏感；

**⚠️ 局限性**

局限性在于：仅关注设备变异对算法性能的影响，未与其他最优优化器（如量子退火、图灵机）做直接比较；GPU实现未针对能耗优化；仅验证了MAX‑CUT，未扩展到其他组合优化问题。

---

## 61. Loss Aversion Online: Emotional Responses to Financial Booms and Crashes

**arXiv ID:** 2601.14423 | [PDF](https://arxiv.org/pdf/2601.14423v1)

**作者:** Aryan Ramchandra Kapadia `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3141 | [OpenAlex ID](https://openalex.org/A5057029055)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用 Reddit 公开数据，比较金融社区与非金融社区在金融暴跌与繁荣期间的情绪与心理语言表达差异，检验损失厌恶与消极偏见的实际影响。

**💡 创新点**

首次将准实验设计（差分中的差分与贝叶斯结构时序）应用于大规模社交媒体文本，系统对比金融涨跌两种极端事件在情绪表达上的不对称性，提供实时、群体层面证据；同时构建可复现的金融与非金融子板块对照组。

**🔧 技术方法**

差分中的差分 (DiD)、Causal Impact（贝叶斯结构时序模型）、VADER 情感分析、DistilRoBERTa 情绪分类、LIWC 心理语言特征提取。

**📊 数据集**

约 15 个金融主题子板块与 11 个非金融子板块的 320 万篇帖子与 2.8 亿元评论，覆盖 2024 年 10 月至 2025 年 5 月的 30 天前后窗口。

**📈 对比分析**

通过与非金融社区对照组的 DiD 估计以及贝叶斯时序的因果冲击，发现暴跌期间情绪负面向明显增强（负情绪、愤怒、悲伤上升；正情绪下降），繁荣期间则多为混合或轻微正向变化；效果显著且持续性差异明显，验证了损失厌恶的非对称效应。

**⚠️ 局限性**

因果推断仍受限于观测性数据，无法完全排除未观测混杂；预事件平行趋势假设可能不成立；样本仅覆盖美国 Reddit，缺乏跨文化与跨平台验证；仅检视两次事件，泛化性有限；高频短期波动可能被平滑或忽略。

---

## 62. Can LLM Reasoning Be Trusted? A Comparative Study: Using Human Benchmarking on Statistical Tasks

**arXiv ID:** 2601.14479 | [PDF](https://arxiv.org/pdf/2601.14479v1)

**作者:** Crish Nagarkar `[一作]`, Serge Sharoff `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本研究中，作者对 LLaMA‑3、Mistral 以及 DeepSeek 三款 7B 参数的开源 LLM 进行 LoRA 与 8‑bit 量化微调，随后在自建的 2000 题统计推理数据集上训练，并在 50 题留存集上通过人类评审与自动评测指标（BLEU、BERTScore、SBERT、Perplexity、LLM‑as‑Judge）评估模型在答案准确性、步骤解释和推理质量三维度的表现。

**💡 创新点**

创新点在于：①提出了专门针对统计推理的 2000 题大规模数据集；②系统比较了参数高效微调对统计推理能力的提升；③构建多维度评估框架，将传统文本相似度指标与 LLM‑as‑Judge 进行对比，揭示了自动评测方法在数学推理中的可靠性与局限性。

**🔧 技术方法**

使用技术包括：LoRA 参数高效微调、8‑bit 权重量化、基于大型语言模型的评判者（LLM‑as‑Judge）、以及传统评测指标（BLEU、BERTScore、SBERT、Perplexity）等。

**📊 数据集**

采用的数据集为作者自建的 2000 题统计推理数据集（涵盖假设检验、回归、概率、ANOVA 等），以及 50 题留存评测集。

**📈 对比分析**

评估方法通过人类评审与自动指标的 MAE、Wilcoxon 统计检验和 Kendall τ 相关性对比；实验表明微调后模型在所有维度均显著提升，Mistral 微调版在综合得分上超越 LLaMA；DeepSeek 在推理维度表现最佳；传统指标与人类评分差异大，而 LLM‑as‑Judge 与人类评分的相关性更高，但仍存在偏差。

**⚠️ 局限性**

局限性包括：受限于算力，未能探索更大模型或更充分的超参数调优；数据集仅覆盖英文且在特定统计领域，缺乏跨学科、跨语言或多模态问题；模型在高风险决策场景中可能产生误导性推理，缺乏可靠的不确定性量化与可解释性。

---

## 63. Certified Real Eigenvalue Location

**arXiv ID:** 2601.14491 | [PDF](https://arxiv.org/pdf/2601.14491v1)

**作者:** Baran Solmaz `[一作]` (Gebze Technical University), Tulay Ayyildiz `[通讯]`

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种将Hermite矩阵签名与Gershgorin圆盘相结合的混合方法，用于对矩阵实特征值进行可认证的区间定位。

**💡 创新点**

首次利用Hermite矩阵与Gershgorin圆盘的符号特征相结合，无需预先知道特征值即可得到包含实特征值的保真区间，并可通过二分细化。

**🔧 技术方法**

采用La Budde算法求特征多项式，Newton–Girard公式构造Hermite矩阵，计算其特征值签名，再通过符号差异检测区间是否包含实根；使用Julia实现。

**📊 数据集**

主要在一个5×5示例矩阵上验证，使用该矩阵的数据进行实验。

**📈 对比分析**

与传统数值特征值求解器相比，方法保持O(n³)复杂度，同时提供严格的实特征值区间保证；在5×5矩阵上运行约11秒，能够将区间收敛至10⁻⁷甚至10⁻¹⁶。

**⚠️ 局限性**

仅适用于实系数多项式矩阵，未处理重根情况；特征多项式求解仍是主要瓶颈，且方法不用于直接求解所有特征值。

---

## 64. Towards Execution-Grounded Automated AI Research

**arXiv ID:** 2601.14525 | [PDF](https://arxiv.org/pdf/2601.14525v1)

**作者:** Chenglei Si `[一作]` (Stanford University), Tatsunori Hashimoto `[通讯]` (Stanford University)

**通讯引用:** 11201 | [OpenAlex ID](https://openalex.org/A5015518638)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个大规模并行的自动化想法执行器，对LLM生成的科研想法进行代码差异生成、调度和GPU实验执行，并将实验结果作为反馈，用于训练LLM的想法生成能力；同时在LLM预训练和后训练这两个开放式研究环境中验证执行反馈驱动的进化搜索和强化学习方法的效果。

**💡 创新点**

创新点包括：①首次实现能自动将自然语言想法转化为可执行代码并在GPU集群上批量跑实验的执行器体系；②利用执行反馈作为奖励，构建进化搜索和GRPO强化学习框架，探索LLM在开放式科研问题上的自我改进；③在预训练（nanoGPT speedrun）和后训练（GRPO）两个真实场景中展示了执行反馈能显著提升性能，并揭示了进化搜索样本高效、RL多样性坍塌等新的研究发现。

**🔧 技术方法**

技术手段包括：大模型（Claude‑4.5‑Opus/Sonnet、GPT‑5、Qwen3‑30B等）做想法生成与代码差异生成；实现者–调度器–工作者三层架构实现并行执行；进化搜索算法（基于样本回放、exploit/explore比例调节）；GRPO强化学习框架；GPU集群（8 H100）用于实验；日志与结果存储于云存储。

**📊 数据集**

使用的主要数据集：FineWeb用于预训练任务；MATH数据集用于后训练任务；此外还使用公开的nanoGPT speedrun基线代码和GRPO基线实现作为实验对照。

**📈 对比分析**

对比方法：将进化搜索与单纯的best‑of‑N采样、基线代码、以及人类专家排行榜进行对比。结果显示：进化搜索在10轮内将后训练模型的验证准确率提升至69.4%（基线48.0%），预训练训练时间缩短至19.7 min（基线35.9 min），而人类专家在后训练任务中达到68.8%，在预训练任务中则达2.1 min。强化学习虽提升了平均奖励，却未能提高最高奖励，出现多样性坍塌。

**⚠️ 局限性**

局限性：①未验证生成想法在更大规模或不同数据集上的泛化能力；②强化学习仅提升平均奖励，未改善极值并导致多样性坍塌；③执行器对部分想法的实现失败导致奖励噪声；④当前仅以有效性奖励为目标，未考虑想法的创新度和趣味性。

---

## 65. SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks on Vision-Language-Action Models

**arXiv ID:** 2601.14323 | [PDF](https://arxiv.org/pdf/2601.14323v1)

**作者:** Bingxin Xu `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**通讯引用:** 18354 | [OpenAlex ID](https://openalex.org/A5078699564)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种针对 Vision‑Language‑Action (VLA) 模型的黑盒后门攻击方法，利用动作分块与增量姿态表示导致的“intra‑chunk visual open‑loop”漏洞，实现隐蔽的轨迹漂移诱发任务失败。

**💡 创新点**

创新点包括：①发现并利用动作分块导致的开环累计漂移漏洞；②使用 Smootherstep 五次多项式生成满足 C² 连续性的扰动，规避动力学异常检测；③提出关键帧攻击策略，仅在接近目标的关键时刻注入扰动，显著降低触发暴露与训练分布漂移。

**🔧 技术方法**

技术手段包括：基于 VLA 的动作分块架构与增量姿态表示；Smootherstep 计算与扰动注入；关键帧检测与上下文触发机制；对抗性数据标注与黑盒后门训练。

**📊 数据集**

使用 LIBERO 机器人仿真基准（包含 Spatial、Object、Goal、10 等子套件）进行实验与评估。

**📈 对比分析**

与基线模型（VLA‑Adapter、π_0）进行对比，攻击成功率（ASR）平均达 93.2%（POISON 率 < 2%），同时保持 95.3% 的 Clean Task Success Rate（CTSR），显示在保持原始性能的前提下能高效诱发失败。

**⚠️ 局限性**

局限性在于：该方法属于双用途技术，可能被恶意利用；攻击对触发器可见度依赖较高；实验仅在仿真环境中验证，缺乏真实机器人部署的验证与防御措施。

---

## 66. CMind: An AI Agent for Localizing C Memory Bugs

**arXiv ID:** 2601.14434 | [PDF](https://arxiv.org/pdf/2601.14434v1)

**作者:** Chia-Yi Su `[一作]` (University of Notre Dame), Collin McMillan `[通讯]` (University of Notre Dame)

**通讯引用:** 3290 | [OpenAlex ID](https://openalex.org/A5084874990)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CMind，一款结合LLM推理与人类程序员行为引导的AI代理，用于定位C语言内存错误并生成定位假设。

**💡 创新点**

创新点在于将LLM与“partial autonomy”相结合，设计了基于人类实验的提示和决策流程，使LLM保持“在笼中”，并模仿程序员的三步定位流程。

**🔧 技术方法**

主要技术包括GPT‑o4/GPT‑5 mini LLM、Joern 与 Doxygen 静态分析工具、Python脚本提取函数、Web/命令行交互界面。

**📊 数据集**

使用的评估数据集为Heap数据集与Redis bug报告，合计20条C语言内存错误案例。

**📈 对比分析**

与GPT‑5 mini对比，二者在同一20条报告上分别取得75%和80%的准确率，表明在给定“leash”的情况下不同模型表现相近。

**⚠️ 局限性**

局限性包括只能处理包含清晰堆栈跟踪或AddressSanitizer信息的错误，缺乏对隐式症状的鲁棒性，且对提示和“leash”机制依赖较强。

---

## 67. Real-Time Wildfire Localization on the NASA Autonomous Modular Sensor using Deep Learning

**arXiv ID:** 2601.14475 | [PDF](https://arxiv.org/pdf/2601.14475v1)

**作者:** Yajvan Ravan `[一作]` (OSTEM Internship Program), Nikhil Behari `[通讯]` (OSTEM Internship Program)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了一份来自NASA AMS传感器的高空多光谱火灾图像人工标注数据集，并利用该数据集训练了一个两层级的实时火灾定位模型（先分类后分割），实现了高精度火线识别。

**💡 创新点**

创新点在于：①首次公开高空（3–50 km）12波段多光谱火灾图像数据集；②将分类网络与分割网络组合为两层级实时模型，显著提升效率与准确性；③系统化评估不同光谱通道的重要性，证明SWIR、IR、热通道对火线识别最关键。

**🔧 技术方法**

采用卷积神经网络：分类网络基于三层U‑Net编码器，分割网络为两层编码–解码器并具跳跃连接；训练使用交叉熵（加权）和Adam优化器；实验实现实时推理（>100 fps）。

**📊 数据集**

使用NASA AMS 18次飞行获得的12波段图像（红、绿、蓝、SWIR、IR、热等），随机切块生成约4259个训练补丁、85个测试补丁；数据覆盖日间、夜间、云雾遮挡等多变情境。

**📈 对比分析**

与先前基于Landsat的UNet网络和传统颜色规则算法对比，本文模型在分类准确率96.8%、召回率84%、IoU 74%（相较Landsat网络IoU 47.5%）显著提升，且推理速度仅为7.75 ms。

**⚠️ 局限性**

局限性包括：数据集仅来自18次AMS任务，可能存在域漂移；模型仍需人机协同验证；对特殊云雾、极端光照等极端情况的鲁棒性待进一步评估。

---

## 68. A benchmarking framework for PON-based fronthaul network design

**arXiv ID:** 2601.14480 | [PDF](https://arxiv.org/pdf/2601.14480v1)

**作者:** Egemen Erbayat `[一作]` (George Washington University), Suresh Subramaniam `[通讯]` (George Washington University)

**通讯引用:** 16132 | [OpenAlex ID](https://openalex.org/A5042830832)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套统一的PON光线路距设计基准框架，并在此框架下评估了整数线性规划（ILP）与三种可扩展启发式算法（GA、改进K‑均值聚类KMC+、改进随机后继分配RSSA+）的性能。

**💡 创新点**

创新点在于构建标准化的成本目录与部署模板，使得不同算法可在可重复对比的条件下评估；同时提出RSSA+构造式算法，能够在满足所有物理约束的前提下逼近ILP最优解。

**🔧 技术方法**

采用整数线性规划、遗传算法、改进K‑均值聚类和改进随机后继分配算法，并在Python仿真环境中结合光纤传输模型进行实现。

**📊 数据集**

使用基于行业平均参数生成的合成网络拓扑（候选DU、光分配器和RU随机布点），覆盖四种场景（覆盖、低延迟、沉浸式、海量通信）进行实验。

**📈 对比分析**

在统一的基准框架下将ILP（时间限制3600 s）与GA、KMC+、RSSA+进行比较，结果显示RSSA+在成本上与ILP相差≤5%，而GA表现最差；ILP即使未收敛也往往给出最优下界。

**⚠️ 局限性**

主要局限在于所用合成数据未覆盖设备异构、功能拆分层级多样的真实情况；ILP在大规模实例上计算不可行，只能作为参考基准。

---

## 69. On the Generalization Gap in LLM Planning: Tests and Verifier-Reward RL

**arXiv ID:** 2601.14456 | [PDF](https://arxiv.org/pdf/2601.14456v1)

**作者:** Valerio Belcamino `[一作]` (University of Genoa), Fulvio Mastrogiovanni `[通讯]` (University of Genoa)

**通讯引用:** 2619 | [OpenAlex ID](https://openalex.org/A5017108129)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在10个IPC 2023规划域上，使用40,000个域-问题-计划元组对1.7B参数LLM进行细调，评估其在域内和跨域的规划有效性，并通过符号匿名化、紧凑序列化和基于VAL验证器的强化学习三种诊断干预来探究模型的泛化机制。

**💡 创新点**

首次揭示LLM在高域内性能（82.9%有效计划率）与跨域零性能之间的巨大差距，并证明在有监督热启动后，利用VAL验证器的奖励信号即可取代参考计划，显著提升标签效率；同时验证了模型对表面表征高度敏感。

**🔧 技术方法**

技术包括：有监督细调（SFT），实例级符号匿名化，紧凑计划序列化，基于GRPO的强化学习，VAL验证器作为功能奖励信号。

**📊 数据集**

使用Gideon框架生成的40,000个训练/验证样本（10个IPC域），以及来自Rover和Briefcase两域的500个未见域测试样本。

**📈 对比分析**

与基线比较：基线在域内可达82.9%有效率，跨域为0%；符号匿名化和紧凑序列化分别降至约71%和72%；Verifier-Reward（v3）在1.5个总轮次内达到≈80%有效率，但仍未改善跨域表现。

**⚠️ 局限性**

局限性包括：模型对表面表征高度依赖，训练域数量有限导致抽象规划能力难以显现；1.7B模型容量可能不足；强化学习训练成本高，且跨域零性能表明仍需进一步提升泛化机制。

---

## 70. If You Want Coherence, Orchestrate a Team of Rivals: Multi-Agent Models of Organizational Intelligence

**arXiv ID:** 2601.14351 | [PDF](https://arxiv.org/pdf/2601.14351v1)

**作者:** Gopal Vijayaraghavan `[一作]` (Isotopes AI), Vivek Subramanian `[通讯]` (Isotopes AI)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一套“AI 办公室”式多代理系统，采用规划者、执行者、批评者和远程代码执行器等角色，在生产金融分析工作流中实现错误捕获与自动修复。

**💡 创新点**

创新点包括：① 在多代理架构中引入“对抗式共识”与“逐级 veto”机制，确保不同角色相互校验；② 将思维模型与执行层分离，利用远程代码执行器防止上下文污染；③ 引入“Context Ray Tracing”消息可见性控制，实现细粒度信息隔离与审计；④ 通过多模型、多供应商组合实现认知多样性，降低单一模型的系统性偏差。

**🔧 技术方法**

核心技术：大型语言模型（多家供应商）、基于 FSM 的多代理调度、结构化消息传递（Pydantic），远程代码执行器（Jupyter/自定义执行环境），自定义 Code/Chart/Output Critic 三层校验器，程序化的错误重试与审批流程。

**📊 数据集**

数据集：522 条真实金融对账会话（含 PDF 发票、QuickBooks 在线报表等 15+ 数据源），用于评估错误检测率、成本与时延；同时在单代理实验中使用相同的对账任务作为基准。

**📈 对比分析**

与单代理方法对比，单代理 60% 正确率、无错误信号；自检进一步降低准确率；多代理体系在相同任务上达 92.1% 成功率，错误率从 75% 降至 7.9%，成本增加 38.6% 计算信用，时延约 2–3 倍，但在高价值任务上被视为可接受。

**⚠️ 局限性**

局限性：仅在金融分析域评估，泛化性待验证；多层校验导致计算与时延显著上升，尤其对实时或大批量场景不友好；残留 7% 错误属于不可自动化的需求歧义/主观偏好等问题，需人工介入；缺乏对误报率的系统度量。

---

## 71. QMC: Efficient SLM Edge Inference via Outlier-Aware Quantization and Emergent Memories Co-Design

**arXiv ID:** 2601.14549 | [PDF](https://arxiv.org/pdf/2601.14549v1)

**作者:** Nilesh Prasad Pandey `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**通讯引用:** 10654 | [OpenAlex ID](https://openalex.org/A5025573294)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向边缘设备的量化与内存共设计框架 QMC，通过区分权重为“外点”和“内点”并采用不同精度存储，实现小型语言模型的高效推理。

**💡 创新点**

创新点在于：①无训练的“外点感知量化”仅保留关键大幅权重高精度，②将内点权重压缩到多层 ReRAM、外点存储到 MRAM 的异构非易失性内存架构，③在硬件层面结合噪声模型对 ReRAM 误差做自适应量化，整体兼顾精度、能耗与延迟。

**🔧 技术方法**

使用的技术包括：Post‑Training Quantization (PTQ)、基于阈值的权重划分、低比特多层细胞 ReRAM 与 MRAM 存储、噪声模型驱动的尺度优化、统一的权重量化控制器及 2‑D chiplet 互连。

**📊 数据集**

主要数据集：WikiText（语言建模）和多项推理基准（HellaSwag、BoolQ、ARC‑Easy、ARC‑Challenge），在小型语言模型（Hymba‑1.5B、LLaMA‑3.2‑3B、Phi‑1.5B、Qwen2.5‑1.5B‑Instruct）上评测。

**📈 对比分析**

与 FP16、RTN INT4、MXINT4 以及先进 PTQ 方法（AWQ、GPTQ）和 eMEMs 的对比表明：在不损失准确率的前提下，QMC 在 3‑bit ReRAM 模式下实现 4.44× 压缩率，能耗下降 10.98×，延迟下降 12.48×，与 FP16 差距显著缩小。

**⚠️ 局限性**

局限性：需要异构非易失性内存（MRAM/ReRAM）与对应硬件支持，内存布局与同步产生 21.6 mm² 的面积与若干时钟域交叉延迟；对 2‑bit ReRAM 的量化兼容性需额外打包/拆包；在极低外点比例时可能出现 MRAM 成本与性能瓶颈。

---

## 72. When Generative AI Is Intimate, Sexy, and Violent: Examining Not-Safe-For-Work (NSFW) Chatbots on FlowGPT

**arXiv ID:** 2601.14324 | [PDF](https://arxiv.org/pdf/2601.14324v1)

**作者:** Xian Li `[一作]` (Hong Kong University of Science and Technology), Shuo Niu `[通讯]` (Clark University)

**通讯引用:** 354 | [OpenAlex ID](https://openalex.org/A5014136925)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对 FlowGPT 平台上 376 个 NSFW 聊天机器人及其 307 条公开对话进行系统分析，归纳出四类功能（角色扮演、故事生成、图像生成、Do Anything Now）并评估其身份设定、行为特征与头像暴露度，同时通过人工标注、LLM 识别与 Google SafeSearch 对话文本中的性、暴力、侮辱等有害内容进行三方检测；

**💡 创新点**

首次将 Paasonen 的 NSFW 功能框架与 GenAI 聊天机器人结合，构建了多维度的 NSFW 体验模型，提出结合多源检测与人工校验的混合方法来识别动态生成的有害内容；

**🔧 技术方法**

采用 LLM（ChatGPT GPT‑4o‑mini）进行自我评估与对话标注，利用 Google SafeSearch 与 Azure Content Safety 对图像与文本进行安全性评估，并使用 Krippendorff’s alpha 等统计方法衡量标注一致性；

**📊 数据集**

使用 FlowGPT 公共数据集，包括 376 个带有 NSFW 标签的聊天机器人（涵盖 190 位创作者）以及 307 条公开聊天记录，结合机器人头像与对话内容；

**📈 对比分析**

通过人类标注、LLM 自动标注与 Google SafeSearch 的三方比对，发现三者在检测性内容上高度一致，但在暴力与侮辱类别上 LLM 的一致性低于人类标注，整体检测准确率约 70‑80%（性内容）；

**⚠️ 局限性**

主要局限包括：仅分析单一平台的数据，难以推广至其他生成式 AI 社区；对话分析为聚合层面，忽视了上下文动态变化；LLM 识别对歧义性有偏差；人工标注资源受限，样本规模有限；

---

## 73. Business Logic-Driven Text-to-SQL Data Synthesis for Business Intelligence

**arXiv ID:** 2601.14518 | [PDF](https://arxiv.org/pdf/2601.14518v1)

**作者:** Jinhui Liu `[一作]` (Columbia University), Zhou Yu `[通讯]` (Columbia University)

**通讯引用:** 7926 | [OpenAlex ID](https://openalex.org/A5016175345)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于业务逻辑的文本到SQL数据合成框架，生成真实业务场景下的SQL问答对。

**💡 创新点**

通过层次化业务角色、工作情景和工作流程建模，引入业务推理复杂度控制，显著提升生成数据的业务真实性。

**🔧 技术方法**

利用大型语言模型（GPT‑5）进行业务逻辑生成、SQL先行生成与执行反馈循环，以及LLM‑as‑a‑Judge评估质量。

**📊 数据集**

在规模约710张表的生产级Salesforce销售分析数据库上生成并评估240条问答对。

**📈 对比分析**

与OmniSQL、SQL‑Factory 对比，业务真实性提升至98.44%，问题‑SQL对齐率98.59%；但在最复杂推理级别，当前最强模型仅42.86%执行准确率。

**⚠️ 局限性**

框架仅适用于BI场景，验证仅在销售域，跨域普适性与多轮交互支持仍待扩展。

---

## 74. Robust Haptic Rendering Using a Nonlinear Impedance Matching Approach (NIMA) for Robotic Laparoscopic Surgery

**arXiv ID:** 2601.14445 | [PDF](https://arxiv.org/pdf/2601.14445v1)

**作者:** Aiden Mazidi `[一作]` (Concordia University), Amir Hooshiar `[通讯]` (McGill University)

**通讯引用:** 669 | [OpenAlex ID](https://openalex.org/A5088381242)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种用于机器人腹腔镜手术的非线性阻抗匹配方法（NIMA），通过实时识别非线性工具-组织交互阻抗参数来实现高保真触觉反馈，并消除抖动反弹；

**💡 创新点**

创新点在于将非线性阻抗模型嵌入传统阻抗匹配框架，实时更新三轴力参数，实现 95% 的误差下降，同时通过动态阈值抑制手柄释放时的“kickback”现象；

**🔧 技术方法**

采用了非线性多项式阻抗模型、滚动窗口伪逆求解、全连接前馈神经网络估计工具尖端力、Kinova Gen3 机器人、Omega.7 触觉控制器、6轴力–扭矩传感器及NDI 光学跟踪系统；

**📊 数据集**

在仿真与实验中使用了 30,000 条包含工具姿态、力与关节位置的训练数据，构建神经网络，并在实际机器人上收集高频（2 kHz）力数据进行验证；

**📈 对比分析**

与直接力反射（DFR）和线性 IMA 进行对比，NIMA 的平均绝对误差为 0.01 N，标准差 0.02 N，显著优于线性 IMA 的 0.2 N（SD 0.4 N），实现了 95% 的精度提升，并成功消除了抖动反弹；

**⚠️ 局限性**

局限性包括仅处理三轴力而非扭矩、需在特定机器人平台上训练模型、对不同组织类型的泛化性有限、以及尚未在临床环境中进行大规模验证。

---

## 75. How Worst-Case Are Adversarial Attacks? Linking Adversarial and Statistical Robustness

**arXiv ID:** 2601.14519 | [PDF](https://arxiv.org/pdf/2601.14519v1)

**作者:** Giulio Rossolini `[一作]` (Scuola Superiore Sant'Anna), Giulio Rossolini `[通讯]` (Scuola Superiore Sant'Anna)

**通讯引用:** 206 | [OpenAlex ID](https://openalex.org/A5021748411)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探讨了对抗性攻击在模型鲁棒性评估中的有效性，提出了一种概率度量来量化与方向性偏差扰动分布相关的噪声风险，并通过实验验证了对抗性攻击在不同噪声条件下的表现。

**💡 创新点**

创新点在于引入了方向性噪声风险的定义和分析，提供了将对抗性扰动与相同幅度的随机噪声联系起来的概率理解，并提出了一种新的方向性噪声攻击，旨在优化低κ值下的对抗成功率。

**🔧 技术方法**

使用了概率度量、方向性噪声攻击策略以及蒙特卡洛方法来评估模型的鲁棒性。

**📊 数据集**

实验使用了ImageNet和CIFAR-10数据集，评估了多种对抗性攻击的效果。

**📈 对比分析**

通过与传统对抗性攻击（如PGD、FGSM等）进行比较，发现许多现有攻击在高κ值下成功率高，但在低κ值下表现不佳，表明它们在随机噪声下的鲁棒性较差。相反，提出的方向性噪声攻击在低κ值下表现更好，提供了更具代表性的失败区域。

**⚠️ 局限性**

限制在于对抗性攻击的成功率并不总能反映模型在实际随机噪声下的鲁棒性，且现有攻击方法在高维空间中可能会导致对抗性扰动的局限性。

---

## 76. A Cloud-Based Cross-Modal Transformer for Emotion Recognition and Adaptive Human-Computer Interaction

**arXiv ID:** 2601.14259 | [PDF](https://arxiv.org/pdf/2601.14259v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 77. Psychometric Comparability of LLM-Based Digital Twins

**arXiv ID:** 2601.14264 | [PDF](https://arxiv.org/pdf/2601.14264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 78. Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs

**arXiv ID:** 2601.14340 | [PDF](https://arxiv.org/pdf/2601.14340v1)

**作者:** Yiyang Lu `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Ruigang Liang `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于对话结构（轮次索引）的后门攻击，称为Turn-based Structural Trigger (TST)，在多轮对话LLM中实现无输入触发的隐藏功能。

**💡 创新点**

创新点在于首次利用对话轮次作为触发器，完全不依赖用户提示；通过结构化触发实现高成功率、低误触发、语义一致且对模型常规性能几乎无影响；并展示现有防御对该结构触发的无效性。

**🔧 技术方法**

技术包括对话序列化利用角色标签与轮次信息；构造poisoned数据；联合损失函数（SFT+poison+clean+punish+entropy）；采用LoRA参数高效微调实现结构化后门。

**📊 数据集**

使用的主要数据集为UltraChat（训练1,800条poisoned + 900条clean，测试4,200条poisoned），并在ChatAlpaca‑20K上验证泛化；评估基准包括MMLU、MT‑Bench与TrustLLM。

**📈 对比分析**

在四大开源模型（LLaMA‑2‑7B、Qwen3‑4B、Mistral‑7B、DeepSeek‑R1‑Distill‑Llama‑8B）上平均ASR达到99.52%，CR为100%，FTR为0%；在ONION、back‑translation、fine‑tune、pruning、quantization等五种防御下，ASR仍保持>98%，显示防御无效；模型在MMLU、MT‑Bench几乎无性能损失。

**⚠️ 局限性**

局限性包括：仅在多轮对话场景下有效，单轮任务无效；依赖统一的对话模板，若模板变更或不使用统一格式可能失效；攻击只能在训练阶段植入，无法实时操纵用户输入；对部分模型的精细化修复（如微调、量化）效果不一，需进一步研究。

---

## 79. A Comparison of Polynomial-Based Tree Clustering Methods

**arXiv ID:** 2601.14285 | [PDF](https://arxiv.org/pdf/2601.14285v1)

**作者:** Pengyu Liu `[一作]` (University of Rhode Island), Nataša Jonoska `[通讯]` (University of South Florida)

**通讯引用:** 1582 | [OpenAlex ID](https://openalex.org/A5083450874)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对利用树多项式编码的距离度量和自编码器模型进行聚类实验，比较其在随机生成的根二叉树上的聚类准确率。

**💡 创新点**

首次系统评估了六种基于多项式系数矩阵的条目级归一化距离（包括归一化欧氏、曼哈顿和Canberra）与两种自编码器（线性、卷积）在树聚类任务中的表现，并指出归一化距离表现最佳；同时提出可进一步探索Transformer等注意力模型。

**🔧 技术方法**

采用树多项式编码、欧氏/曼哈顿/Canberra/Bray‑Curtis距离、k‑medoids、k‑means、线性与卷积自编码器、Adam优化、MSE损失及500轮训练等技术。

**📊 数据集**

使用 beta‑splitting 模型生成的 100 组随机根二叉树（每组 300 棵树，100 棵每个 β 值为 -1.5、-1、0，叶子节点 100）。

**📈 对比分析**

通过构建 100×100 距离矩阵并重复 10 次 k‑medoids（距离度量）或 k‑means（自编码器潜向量）聚类，计算聚类准确率；结果显示归一化欧氏、归一化曼哈顿和 Canberra 的平均准确率约为 0.90，非归一化距离低；自编码器方法平均准确率分别为 0.79（线性）和 0.76（卷积）。

**⚠️ 局限性**

仅在小规模二叉树数据上评估，未检验非二叉树或实际 RNA 二级结构等更复杂数据；自编码器表现不佳，缺乏针对多项式系数相关性的网络模型；实验规模有限，结果可能受样本量影响。

---

## 80. Divide and Refine: Enhancing Multimodal Representation and Explainability for Emotion Recognition in Conversation

**arXiv ID:** 2601.14274 | [PDF](https://arxiv.org/pdf/2601.14274v1)

**作者:** Anh-Tuan Mai `[一作]` (VNU University of Engineering and Technology), Duc-Trong Le `[通讯]` (VNU University of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个两阶段的“Divide and Refine”框架，先把每种模态的表征拆解为唯一（U）、冗余（R）和协同（S）三类，再通过只针对冗余部分的增强和对比学习来提升多模态情感识别的鲁棒性。

**💡 创新点**

创新点在于：①利用Partial Information Decomposition（PID）将多模态特征系统性分离为三种信息角色；②通过在冗余子空间上施加对比增强，既保留唯一与协同信息，又提升冗余信息的表达；③该框架可无缝插拔到任意现有MER基座，兼容多种融合与图模型。

**🔧 技术方法**

使用的技术包括：PID分解层、唯一-冗余-协同正则化（相关/不相关损失）、基于InfoNCE的对比学习、数据增强（对冗余向量做噪声扰动）、多模态预训练编码器（OpenSmile、OpenFace、sBERT、wav2vec、MA‑Net、DeBERTa）以及标准的交叉熵任务损失。

**📊 数据集**

实验数据集：IEMOCAP（6类情感）和MELD（多方对话情感），分别使用音频、文本、视觉三模态特征。

**📈 对比分析**

将框架与7种主流MER基座（MMGCN、DialogueGCN、MM‑DFN、SDT、GraphSmile、CORECT 等）集成后，在IEMOCAP上平均提升约1.5%~2.0% W‑F1，在MELD上提升约0.5%~1.0% W‑F1；在缺失模态或噪声条件下提升更显著，验证了鲁棒性。

**⚠️ 局限性**

局限性：①分解与增强的超参数需要手动调节，易受数据规模影响；②仅针对冗余部分做增强，对极端噪声或完全缺失的模态效果有限；③目前仅在情感识别任务验证，尚未探讨跨领域迁移或其他多模态任务的适用性。

---

## 81. Quantifying Speaker Embedding Phonological Rule Interactions in Accented Speech Synthesis

**arXiv ID:** 2601.14417 | [PDF](https://arxiv.org/pdf/2601.14417v1)

**作者:** Thanathai Lertpetchpun `[一作]`, Shrikanth Narayanan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在现代 TTS 系统中引入三条针对美英差异的音位规则（flapping、rhoticity、vowel correspondences），并结合说话人嵌入进行对比实验，探讨规则与嵌入在口音控制中的相互作用。

**💡 创新点**

创新点在于：①提出音位移位率（Phoneme Shift Rate, PSR）作为衡量规则与嵌入交互的可解释指标；②将规则与嵌入相结合，证明规则能显著提升口音真实性且不影响自然度；③通过 PSR 明确展示嵌入对规则的覆盖或支持程度，揭示嵌入与口音特征的耦合。

**🔧 技术方法**

技术包括：使用 Kokoro TTS（多语言 82M 预训练模型）生成语音；利用 Misaki G2P 转换文本为美式音位序列；手工实现三条音位替换规则；通过 Vox‑Profile（Accent Classifier）评估口音概率与相似度；使用 UTMOS 评估自然度；以及 PSR 计算来量化规则保留率。

**📊 数据集**

数据集主要为 LibriTTS‑R 的 train‑clean‑100 子集，共 33k 条句子（55.4 小时）。嵌入方面使用 Kokoro 提供的 28 预置说话人嵌入（20 美式、8 英式）。

**📈 对比分析**

比较方法：在仅使用嵌入、仅使用规则、以及嵌入+规则三种配置下评估；通过 Vox‑Profile 口音概率（NA/B）、相似度（Cosine）以及 PSR 进行定量对比。结果显示：
- 在美式嵌入下，添加规则将北美口音概率从 86.5% 降至 58.8%，英式概率上升至 17.3%；
- 在英式嵌入下，规则将英式概率从 67.8% 提升至 78.4%；
- PSR 在规则+嵌入组合下下降到约 0.63‑0.77，说明规则得到更好保留；
- UTMOS 自然度基本不变（北美约 4.4，英式约 3.7），说明规则不会损害自然度。

**⚠️ 局限性**

局限性包括：
- 只覆盖三条粗粒度规则，未能捕捉更细微的口音差异；
- 仅在英语（美英）上验证，缺乏跨语言或非英语口音的推广性；
- 评估依赖 Vox‑Profile 和 UTMOS，可能受模型偏差影响；
- 嵌入本身仍包含音色、情感等多重属性，导致规则与嵌入的耦合度不完全可解释；
- PSR 计算假设识别模型完美识别音位，实际识别误差可能影响指标准确性。

---

## 82. Rewarding How Models Think Pedagogically: Integrating Pedagogical Reasoning and Thinking Rewards for LLMs in Education

**arXiv ID:** 2601.14560 | [PDF](https://arxiv.org/pdf/2601.14560v1)

**作者:** Unggi Lee `[一作]` (Chosun University), Gyeonggeon Lee `[通讯]` (Nanyang Technological University)

**通讯引用:** 478 | [OpenAlex ID](https://openalex.org/A5005042692)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出 PedagogicalRL-Thinking 框架，通过强化学习结合教学理论指导大语言模型在数学对话中的内部思考过程，提升教学质量。

**💡 创新点**

创新点在于：①在思考阶段加入基于 Polya 四步法的教学理论提示；②设计 Thinking Reward，对模型内部推理的教学质量进行评估与强化。

**🔧 技术方法**

采用 reasoning-specialized LLM（DeepSeek‑R1 Qwen3‑8B）与 GRPO 强化学习框架，并用 GPT‑4o‑mini 等 LLM 评判器评估答案与思考质量，同时借鉴 Polya、Schoenfeld 框架进行提示与分析。

**📊 数据集**

使用 BigMath 规模数学问题集（约 10k 训练题、500 测评题）进行对话训练与评测，另用 Well‑balanced Educational Benchmark (WBEB) 评估模型在未见领域的泛化能力。

**📈 对比分析**

对比五种实验条件与多种前沿模型，在 ΔSolve、Leak、Helpful 等指标上，Ped. Think Reward 取得最高成绩（ΔSolve=0.294、Leak=0.172、Helpful=0.776），且在 WBEB 上保持事实知识同时显著提升 Pedagogical Knowledge、Essay Scoring、Decision Making。

**⚠️ 局限性**

局限性包括：仅在数学领域验证；使用 7‑8B 规模模型；交互基于模拟学生，评测依赖 LLM 判别；尚未验证在其他学科、真实人类学生以及更大模型规模下的效果。

---

## 83. Beyond Denial-of-Service: The Puppeteer's Attack for Fine-Grained Control in Ranking-Based Federated Learning

**arXiv ID:** 2601.14687 | [PDF](https://arxiv.org/pdf/2601.14687v1)

**作者:** Zhihao Chen `[一作]` (Fujian Normal University), Leo Yu Zhang `[通讯]` (Griffith University)

**通讯引用:** 4477 | [OpenAlex ID](https://openalex.org/A5015011245)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了边缘控制攻击（Edge Control Attack, ECA），能够在排名式联邦学习（FRL）中对全局模型精确控制准确率并保持隐蔽的训练轨迹；

**💡 创新点**

创新点在于首次针对FRL设计可实现精细控制的两阶段确定性算法，突破传统攻击只能造成全局损坏的局限；

**🔧 技术方法**

利用排名通信、上升/下降边缘识别、内部反转、投票聚合等技术实现对排名序列的精准操控；

**📊 数据集**

实验基准覆盖七个数据集（CIFAR‑10、CIFAR‑100、EMNIST、FashionMNIST、Location30、Purchase100、Texas100）与三种模型；

**📈 对比分析**

与基线RRA及VEM相比，ECA在九种Byzantine‑robust聚合规则下平均控制误差仅0.224%，比基线低约17倍；

**⚠️ 局限性**

局限在于仅研究无目标攻击，未考虑针对性攻击、对更强防御（如FLCert、FoundationFL）和更复杂FL框架的适用性。

---

## 84. Agent Identity URI Scheme: Topology-Independent Naming and Capability-Based Discovery for Multi-Agent Systems

**arXiv ID:** 2601.14567 | [PDF](https://arxiv.org/pdf/2601.14567v1)

**作者:** Roland R. Rodriguez `[一作]` `[通讯]`, Roland R. Rodriguez

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出 Agent URI Scheme，实现在多代理系统中将身份与拓扑分离，支持基于能力的去中心化发现

**💡 创新点**

核心创新在于三层结构：信任根、层级能力路径、可排序唯一标识，结合 PASETO 认证实现跨组织的可验证能力声明

**🔧 技术方法**

使用 RFC 3986/URI 语法、SHA256 哈希、Kademlia DHT、Ed25519 签名、PASETO v4.public 令牌、libp2p 实现

**📊 数据集**

利用 369 个来自 LangChain、CrewAI、MCP、AutoGen、smolagents 等生产框架的工具定义，生成能力路径语法；对 10,000 名代理进行模拟发现测试

**📈 对比分析**

与传统单机 DF、A2A 方案比较；在发现精度上 100% 精度与召回，身份稳定性 O(log N) 路径解析，所有关键操作均在微秒级完成，远低于网络延迟

**⚠️ 局限性**

需要组织层面的信任根基础设施；缺乏全局能力本体、激励模型不完善，且跨组织查询需手动映射或第三方映射服务

---

## 85. TacUMI: A Multi-Modal Universal Manipulation Interface for Contact-Rich Tasks

**arXiv ID:** 2601.14550 | [PDF](https://arxiv.org/pdf/2601.14550v1)

**作者:** Tailai Cheng `[一作]` (Agile Robots SE), Alois Knoll `[通讯]` (Technical University of Munich)

**通讯引用:** 24211 | [OpenAlex ID](https://openalex.org/A5063781430)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一款名为TacUMI的多模态手持数据采集设备，并提出基于多模态序列模型的事件分割框架，用于长时序操控任务的分段与学习。

**💡 创新点**

在硬件层面集成触觉、力矩传感器和无漂移6DoF姿态跟踪，并引入连续自锁机制；在算法层面提出多模态BiLSTM分割网络，将触觉、力矩、视觉和姿态特征融合；在电缆安装任务上验证性能。

**🔧 技术方法**

硬件设计与集成、ViTac触觉图像使用ResNet50提取特征、第三视角图像使用ResNet18+GroupNorm+SpatialSoftmax、力矩去除触发干扰、6DoF姿态跟踪、BiLSTM/TCN/Transformer时序模型、滑动窗口训练与soft voting推理。

**📊 数据集**

TacUMI采集的电缆安装任务多模态演示数据集；以及对应的遥操作收集数据集用于跨平台验证。

**📈 对比分析**

与仅视觉/姿态、UML、FastUMI、ForceMimic硬件对比，以及不同模型与模态组合对比。多模态BiLSTM在TacUMI数据上帧级准确率约为94%，在遥操作数据上约为91%；单模态或TCL/Transformer性能显著下降，单模态低于70%。

**⚠️ 局限性**

仅在电缆安装任务上验证，需进一步验证在更复杂多样任务上的泛化；分割模型对极短事件易失真；硬件集成成本与复杂度较高；跨平台仍存在域差距。

---

## 86. Designing KRIYA: An AI Companion for Wellbeing Self-Reflection

**arXiv ID:** 2601.14589 | [PDF](https://arxiv.org/pdf/2601.14589v1)

**作者:** Shanshan Zhu `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3141 | [OpenAlex ID](https://openalex.org/A5057029055)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一个名为KRIYA的AI伴侣原型，通过对话式交互支持用户对步数、睡眠等健康数据进行共建解释与情感化自省，取代传统的目标监控方式；

**💡 创新点**

创新点在于将健康数据的解释权交给用户与AI共同完成，采用同理心语言、可解释的“舒适区”“惊喜评分”等机制，推动从监控到好奇驱动的协同洞察；

**🔧 技术方法**

采用生成式语言模型和对话系统技术（基于规则与预设情境的组合），并在交互中嵌入情感化提示与概率化预测；

**📊 数据集**

使用人工构造的假设步数、睡眠与环境（天气、日程）数据，未使用真实用户数据集；

**📈 对比分析**

通过半结构化访谈与量化问卷评估，SUS中位数76.25、IAM中位数14，均高于行业可接受阈值；相比传统仪表盘，参与者在解释性、情感安全感和低门槛探索性方面表现更佳；

**⚠️ 局限性**

局限性包括：仅为原型且对话功能有限、使用假设数据导致情境真实性不足、研究仅为一次性访谈且样本为18名大学生，缺乏长期部署与多样化人群验证，以及对AI推断范围与误差的透明度与纠正机制仍需改进。

---

## 87. Aiming for AI Interoperability: Challenges and Opportunities

**arXiv ID:** 2601.14512 | [PDF](https://arxiv.org/pdf/2601.14512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 88. Relational Graph Modeling for Credit Default Prediction: Heterogeneous GNNs and Hybrid Ensemble Learning

**arXiv ID:** 2601.14633 | [PDF](https://arxiv.org/pdf/2601.14633v1)

**作者:** Yvonne Yang `[一作]` (University of Illinois Urbana-Champaign), Eranki Vasistha `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建大规模异构图并将其与传统表格特征融合，评估在 Home Credit Default Risk 数据集上的违约预测性能

**💡 创新点**

首次系统比较单独 GNN、强化表格基线及其混合集成，证明关系建模在表格模型上的补充价值；同时引入基于关系注意力的异构 GNN 与对比预训练的可解释性与公平性分析

**🔧 技术方法**

异构图神经网络（GraphSAGE、关系感知注意力 GNN）、对比学习（GraphCL）、梯度提升树（LightGBM）、逻辑回归、嵌入融合与评估工具（ROC‑AUC、PR‑AUC、子组公平性指标）

**📊 数据集**

Home Credit Default Risk（HCDR）公开数据集，构造 31M 节点、50M 边的异构图，包含客户、历史申请、外部信用记录、分期付款、POS/现金余额、信用卡余额等节点与边类型

**📈 对比分析**

在相同 5 折交叉验证下对比：Logistic Regression 0.739 ROC‑AUC；LightGBM 0.769；单独 GNN 最高 0.751；混合 GNN‑增强 LightGBM 0.782（ROC‑AUC）和 0.281（PR‑AUC）——提升 4–6% 以上

**⚠️ 局限性**

单独 GNN 表现不如强化表格基线；对比预训练在通用增广下效果有限，甚至降低子组性能；模型规模与训练成本高，且未对部署级公平性做完整评估

---

## 89. Trust Me on This: A User Study of Trustworthiness for RAG Responses

**arXiv ID:** 2601.14460 | [PDF](https://arxiv.org/pdf/2601.14460v1)

**作者:** Weronika Łajewska `[一作]` (Amazon), Krisztian Balog `[通讯]` (University of Stavanger)

**通讯引用:** 5396 | [OpenAlex ID](https://openalex.org/A5059926999)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对检索增强生成（RAG）系统产生的回答进行两阶段对比实验，研究三种不同解释类型（来源归属、事实基础、信息覆盖）对用户信任度的影响。

**💡 创新点**

首次系统性评估解释方式对RAG回答信任的直接作用，并揭示信任判断与客观质量、清晰度、可操作性以及用户先验知识之间的复杂关系。

**🔧 技术方法**

使用基于GINGER的多模块生成管线，并在后处理阶段生成三类真实解释；实验采用MTurk问卷收集用户选择与自由文本反馈。

**📊 数据集**

30个查询来自TREC CAsT ’22数据集，按不同解释维度挑选，确保每类最多两条同主题。

**📈 对比分析**

对比方法：在无解释时和有解释时分别让同一受试者选择更可信的回答，统计选择比例与转移率。结果显示，解释能将选择从低质量回答向高质量回答转移（约55%→69%），但并非所有受试者都遵从客观质量，表现为对清晰、可操作性更高的低质量回答的偏好。

**⚠️ 局限性**

局限：样本量仅21位MTurk工人，未系统测量受试者背景知识，实验仅关注短期信任变化，解释类型有限，未探讨多模态或交互式解释对长期信任的影响。

---

## 90. PAS-Mamba: Phase-Amplitude-Spatial State Space Model for MRI Reconstruction

**arXiv ID:** 2601.14530 | [PDF](https://arxiv.org/pdf/2601.14530v1)

**作者:** Xiaoyan Kui `[一作]` (Central South University), Beiji Zou `[通讯]` (Central South University)

**通讯引用:** 2996 | [OpenAlex ID](https://openalex.org/A5062015169)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了PAS‑Mamba框架，联合空间域与频域建模，并在频域对幅度与相位分别进行解耦式建模，用以实现高质量MRI重建。

**💡 创新点**

创新点包括①在频域对幅度与相位分别建模，避免相互干扰；②引入环形频域扫描(CFDS)保持k‑space同心结构；③设计双域互补融合模块(DDCFM)，实现相位-幅度与空间-频域的协同融合；④使用Mamba状态空间模型实现线性复杂度的全局依赖建模。

**🔧 技术方法**

使用技术包括Mamba状态空间模型、LocalMamba扫描、CFDS、DDCFM、混合损失（图像+幅度+相位）等。

**📊 数据集**

实验数据集为IXI（T1脑影像）和fastMRI（膝关节影像）。

**📈 对比分析**

与ZFB、UNet、SwinIR、VM‑UNet、MambaIR、FCB‑UNet等方法在×2、×4加速下对比，PAS‑Mamba在PSNR/SSIM上始终优于所有基线，例如IXI radial×2取得40.43 dB/0.9805，显著提升重建质量。

**⚠️ 局限性**

局限性在于仅处理2D重建，未考虑跨切片关联，且对更高加速率和3D重建的适应性仍待验证。

---

## 91. VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models

**arXiv ID:** 2601.14354 | [PDF](https://arxiv.org/pdf/2601.14354v1)

**作者:** Yongchao Huang `[一作]` `[通讯]`, Yongchao Huang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出VJEPA与BJEPA两种自监督的概率性JEPA框架，能够直接学习未来潜在状态的分布并用于控制与规划；

**💡 创新点**

通过变分推断将JEPA转换为预测状态空间模型，实现显式不确定性建模、避免表示坍塌，并可通过产品专家融合动态与先验约束实现零样本任务迁移；

**🔧 技术方法**

变分自编码器、概率预测器、EMAs、产品专家、基于潜在空间的MPC与贝叶斯过滤；

**📊 数据集**

在无标注视频/图像数据上进行无监督预训练（如互联网视频），随后在带噪环境实验中验证；

**📈 对比分析**

与传统基于像素重构或对比学习的JEPA/GAN基线对比，VJEPA/BJEPA在抑制表示坍塌、剔除高方差噪声以及实现可靠的分布式规划方面表现更好；

**⚠️ 局限性**

仍假设潜在空间能捕获足够的可预测信息，且对极度多模态或高度动态环境的建模能力有限；需要额外的先验训练与参数调优，且对实时部署的计算开销未给出完整评估。

---

## 92. MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks

**arXiv ID:** 2601.14652 | [PDF](https://arxiv.org/pdf/2601.14652v1)

**作者:** Zixuan Ke `[一作]` (Salesforce Research), Shafiq Joty `[通讯]` (Salesforce Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于函数调用的强化学习框架，对多智能体系统进行整体编排，并引入可配置的“多智能体度”参数。

**💡 创新点**

创新点在于将多智能体编排视为一次性函数调用问题，支持全局最优结构，并同时提供了从深度、水平、宽度、并行度与鲁棒性五个维度划分的受控基准，用以系统评估MAS的效用。

**🔧 技术方法**

方法主要使用LLM作为编排器与子代理，利用函数调用抽象子代理，训练时采用Group Relative Policy Optimization的RL优化；同时基于iGSM生成的合成任务构建五轴基准。

**📊 数据集**

使用的数据集包括合成的iGSM任务用于基准构建，以及公开任务AIME24/AIME25/GPQA/HotpotQA/BrowserComp+。

**📈 对比分析**

在与SAS以及现有推理时与训练时的MAS基线（AFlow、MaAS、MAS-Zero、MAS-GPT、ToolOrchestra）对比，所提框架在所有任务上均实现了更高的准确率，特别在并行与鲁棒性场景下显著提升。

**⚠️ 局限性**

局限性在于多智能体的优势并非普适，需根据任务结构、验证协议以及编排器/子代理能力进行细致配置；方法对训练样本与计算资源依赖较大，且在极大规模或实时动态任务中的可扩展性尚未验证。

---

## 93. UNCLE-Grasp: Uncertainty-Aware Grasping of Leaf-Occluded Strawberries

**arXiv ID:** 2601.14492 | [PDF](https://arxiv.org/pdf/2601.14492v1)

**作者:** Malak Mansour `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Abdalla Swikir `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5047256346)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种不确定性感知的草莓抓取流水线，利用点云完成和MC Dropout产生多种形状假设，结合力闭合评估和下置信界（LCB）做对象级的抓取或放弃决策。

**💡 创新点**

创新点在于：①将学习到的形状完成的不确定性显式建模并传播到抓取决策；②使用MC Dropout生成多样化的完成样本，计算局部与全局不确定性阈值；③采用下置信界做风险约束，实现在高遮挡下安全放弃抓取，从而显著提升鲁棒性。

**🔧 技术方法**

核心技术包括：Transformer‑based 点云完成网络（PointAttN）+ MC Dropout；CGNet 6‑DoF 抓取预测；力闭合 ε‑度量；几何约束（前向、垂直、爪-物体碰撞）和不确定性过滤；下置信界决策；Simulated 与真实机器人环境（Intel RealSense + Unitree Z1）。

**📊 数据集**

数据集与实验：使用自建的草莓点云，按叶子遮挡比例（0%、6.94%、28.83%、63.12%、87.45%）生成合成遮挡；在 NVIDIA Isaac Sim 及真实室内温室中进行试验；未使用公开的公共数据集，全部为作者自建或模拟数据。

**📈 对比分析**

与六种对照方法（CGNet(Partial)、CGNet+Geometry、Centroid(Completed)、Baseline、No‑Dropout、Dropout）比较；Dropout 方法在所有遮挡水平下均获得最高抓取成功率，尤其在 63% 与 87% 遮挡时成功率分别为 0.85 与 0.87（sim）/ 0.95 与 0.80（实机），明显优于基线和无不确定性版本。

**⚠️ 局限性**

局限性：推理时间较长，难以满足大规模高吞吐量采摘；仅使用单视角 RGB‑D 观测，极端遮挡仍可能产生模糊完成；固定抓手模型未考虑水果柔性、摩擦变化；未实现主动感知或多视角策略，未来需进一步提升速度与主动不确定性消减。

---

## 94. From Textbook to Talkbot: A Case Study of a Greek-Language RAG-Based Chatbot in Higher Education

**arXiv ID:** 2601.14265 | [PDF](https://arxiv.org/pdf/2601.14265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 95. Explainable OOHRI: Communicating Robot Capabilities and Limitations as Augmented Reality Affordances

**arXiv ID:** 2601.14587 | [PDF](https://arxiv.org/pdf/2601.14587v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 96. Call2Instruct: Automated Pipeline for Generating Q&A Datasets from Call Center Recordings for LLM Fine-Tuning

**arXiv ID:** 2601.14263 | [PDF](https://arxiv.org/pdf/2601.14263v1)

**作者:** Alex Echeverria `[一作]` (Instituto de Informática), Fernando Marques Federson `[通讯]` (Instituto de Informática)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个端到端的 Call2Instruct 自动化管道，将呼叫中心录音转化为可用于 LLM 指令微调的问答数据集。

**💡 创新点**

创新点在于把音频预处理、ASR、文本清洗、隐私脱敏、句向量检索与 LLM 重写三大模块无缝集成，并通过语义搜索与重写提升问答对的语义匹配质量。

**🔧 技术方法**

使用 Whisper（ASR）与去噪、语音分离、文本规范化、NER 脱敏、OpenAI text‑embedding‑ada‑002、Elasticsearch 向量检索、ChatGPT/LLM 重写以及 Llama 2 7B 微调等技术。

**📊 数据集**

基于约 3,120 条通信行业呼叫中心录音（音频 + 转录）构建的数据集，最终生成 3,120 条问答对。

**📈 对比分析**

通过将生成的数据集用于 Llama 2 7B 的指令微调，人工评估示例表明模型能在域内问题上给出符合业务的回答，但缺乏定量评估指标；功能验证成功，模型表现良好但仍有泛化偏差。

**⚠️ 局限性**

局限性包括：高度依赖 ASR 的准确率，转录错误会传播；文本清洗与匿名化可能不完美；语义检索与重写匹配可能未达到最优；缺乏量化评估和多语言/多业务域验证；需要更细粒度的人工审核与质量控制。

---

## 97. Say Anything but This: When Tokenizer Betrays Reasoning in LLMs

**arXiv ID:** 2601.14658 | [PDF](https://arxiv.org/pdf/2601.14658v1)

**作者:** Navid Ayoobi `[一作]` (University of Houston), Arjun Mukherjee `[通讯]` (University of Houston)

**通讯引用:** 5084 | [OpenAlex ID](https://openalex.org/A5078060919)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了tokenization‑consistency探测任务，利用简单的词替换来揭示LLM因子词分词非单射导致的“幻影编辑”，并通过token ID遮罩干预验证其影响。

**💡 创新点**

系统归纳了八类tokenizer失配错误，首次证明tokenization层是推理缺陷的根源之一，并展示轻量级token ID遮断能显著降低幻影编辑。

**🔧 技术方法**

设计了tokenization‑consistency任务，使用多家开源LLM（Gemma、Llama、Mistral、Qwen）进行11k次替换实验，并构造token ID遮罩干预机制。

**📊 数据集**

使用XSUM新闻文章（100–600词）作为数据集，随机挑选5%非停用词作为替换目标。

**📈 对比分析**

评估采用Unchanged、Replaced、Different三类指标；在token ID遮断后Different率降至0–5%，整体替换率略提升，表明模型规模并非唯一决定因素。

**⚠️ 局限性**

仅关注词替换任务，无法覆盖其他推理情境；遮罩为临时修复，未根除tokenizer多值映射问题；实验仅在英语新闻文本，跨语言适用性未知。

---

## 98. European digital identity: A missed opportunity?

**arXiv ID:** 2601.14503 | [PDF](https://arxiv.org/pdf/2601.14503v1)

**作者:** Wouter Termont `[一作]` (Ghent University), Beatriz Esteves `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对欧盟数字身份（EUDI）框架及其实现的OpenID架构进行深入分析，识别了安全、可扩展性、隐私与监管合规等方面的缺陷，提出了更为通用的身份与认证视角。

**💡 创新点**

创新点在于系统地将身份定义与认证过程与现代分布式技术（如DID/VC）对齐，指出OpenID的“双层”设计并非真正的去中心化，进而建议使用UMA、A4DS和GNAP等OAuth扩展来弥补其不足。

**🔧 技术方法**

采用协议比较与安全分析技术，对OIDC、OpenID4VCI、OpenID4VP、SIOPv2、OIDC4IDA、OIDC Claims Aggregation等现有规范进行理论评估，结合RFC、W3C、IETF等标准文档进行阐释。

**📊 数据集**

本文未使用具体实验数据或数据集，而是基于文献综述、标准文本和法规条款进行定性分析。

**📈 对比分析**

比较方法主要为协议功能、设计模式、威胁模型与监管适配度的对比分析，未给出量化性能指标，仅从安全性和可用性角度进行评判。

**⚠️ 局限性**

局限性包括缺乏实证实验验证、对新技术（如UMA/A4DS/GNAP）的实现细节探讨不足，以及对欧盟法律文本的解释受主观因素影响，未能涵盖所有可能的业务场景。

---

## 99. GCG Attack On A Diffusion LLM

**arXiv ID:** 2601.14266 | [PDF](https://arxiv.org/pdf/2601.14266v1)

**作者:** Ruben Neyroud `[一作]` (University of Illinois at Urbana-Champaign), Sam Corley `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文实现并评估了Greedy Coordinate Gradient（GCG）攻击在扩散式大型语言模型LlaDA上的效果，探索了前缀扰动、随机后缀和Qwen种子后缀三种攻击变体。

**💡 创新点**

创新点在于首次将白盒GCG攻击迁移至扩散LLM，系统对比了不同攻击策略，并发现前缀扰动在该模型上最为有效。

**🔧 技术方法**

技术手段包括梯度优化的连续嵌入空间搜索、基于蒙特卡罗采样的对数似然损失评估、早停策略以及与自回归模型Qwen的种子后缀生成。

**📊 数据集**

使用了AdvBench数据集中的有害提示，对原模型拒绝输出的提示进行攻击。

**📈 对比分析**

通过对比三种攻击方法的成功率，发现前缀攻击在“最终输出”和“任意时刻输出”两项指标上均优于后缀攻击（差异约11个百分点，整体成功率相对较高）。

**⚠️ 局限性**

主要限制包括有限的计算资源导致长时间攻击不可行、对数似然损失与实际攻击效果不匹配、对可修改词汇范围控制不足，以及在更鲁棒的扩散模型上效果尚待验证。

---

## 100. Towards Transparent Malware Detection With Granular Explainability: Backtracking Meta-Coarsened Explanations Onto Assembly Flow Graphs With Graph Neural Networks

**arXiv ID:** 2601.14511 | [PDF](https://arxiv.org/pdf/2601.14511v1)

**作者:** Griffin Higgins `[一作]` (Canadian Institute for Cybersecurity), Ali A. Ghorbani `[通讯]` (Canadian Institute for Cybersecurity)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究通过构造Assembly Flow Graph（AFG）和利用Meta‑Coarsening技术，对二进制可执行文件的汇编指令流进行图表示，并使用图神经网络进行恶意软件检测与细粒度可解释性分析。

**💡 创新点**

创新点在于首次提出AFG作为完整指令级流程图，并设计了Meta‑Coarsening方案通过先对CFG进行稀疏化再映射到AFG，实现在大规模图上可解释且高效的恶意检测。

**🔧 技术方法**

技术包括CFG与AFG构建、图卷积网络（GCN）、Integrated Gradients/Guided Backpropagation/Saliency解释器、Kron和Variation Edges两种图稀疏化方法。

**📊 数据集**

使用CIC‑DGG‑2025数据集，其中benign样本来自DikeDataset，malicious样本来自PMMLD，涵盖约709个CFG/AFG样本。

**📈 对比分析**

实验与基线比较显示，适度稀疏化（r=0.25或0.75）可使准确率提升约0.4%（最高约89.8%），F1分数保持85–92%，并在解释一致性上达到λ≈0.71，说明解释质量较高。

**⚠️ 局限性**

局限性包括仅针对Windows PE x86二进制，图尺寸巨大导致计算成本高，缺乏对零日或变异恶意样本的评估，以及稀疏化方法为确定性且缺乏更高效的神经网络稀疏化方案。

---

## 101. Measuring the State of Open Science in Transportation Using Large Language Models

**arXiv ID:** 2601.14429 | [PDF](https://arxiv.org/pdf/2601.14429v1)

**作者:** Junyi Ji `[一作]` (Vanderbilt University), Cathy Wu `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 54119 | [OpenAlex ID](https://openalex.org/A5053761444)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套自动化、可扩展的特征提取管线，利用大语言模型（LLM）在 2019‑2024 年的交通研究期刊（Transportation Research Parts A‑F、TR‑IP）共 10,724 篇论文中识别并量化了数据和代码的可用性，并基于此进行横向与纵向的统计与计量分析。

**💡 创新点**

创新点：
- 采用 LLM（Google Gemini 2.5 Flash）实现对全文中文本的语境感知提取，显著提升了传统正则表达式和关键词搜索在开放科学特征识别上的准确性。
- 将 LLM 与结构化元数据、LDA 主题建模、交叉验证一致性评估（Fleiss/Kappa）及多元选择模型相结合，形成可复现的全流程。
- 首次在交通研究领域对开放科学实践（代码/数据共享）进行大规模、系统性的测绘和因子分析，揭示了时间、主题、期刊、地域等多维影响因素。

**🔧 技术方法**

技术手段：
- Elsevier Text‑and‑Data‑Mining API 提取全文 XML；
- Google Gemini 2.5 Flash LLM 进行特征抽取与链接有效性判定；
- Latent Dirichlet Allocation（LDA）实现主题归类；
- Biogeme Python 包构建选择模型；
- 统计一致性评估（Fleiss’s Kappa、Cohen’s κ、百分比一致性）。

**📊 数据集**

数据集：
- 10,724 篇交通研究期刊原始全文（Elsevier API 获取）；
- 96 篇手工标注的验证集，用于评估 LLM 输出质量；
- 额外的 Scopus 引用计数、审稿时长等外部特征。

**📈 对比分析**

评估方法与性能：
- 与两位人工标注者进行三评者一致性分析；
- Fleiss Kappa 取值 0.399–0.839，Cohen κ 与 LLM 的一致性与人类互评相当；
- 绝大多数特征的百分比一致性超过 90%；
- 与正则表达式基线相比，LLM 在误判率上下降约 30‑50%，同时保持可扩展性。

**⚠️ 局限性**

局限性：
- 手工验证样本仅 96 篇，可能不足以覆盖所有边界情况；
- LLM 在元数据提取、模糊链接判定上仍存在错误；
- LDA 主题分类存在误分与先验偏倚；
- 仅评估了可用性标识，未对代码可运行性或结果可复现性进行进一步验证；
- 隐私与版权约束限制了数据开放与再利用的完整性。

---

## 102. Efficient Imputation for Patch-based Missing Single-cell Data via Cluster-regularized Optimal Transport

**arXiv ID:** 2601.14653 | [PDF](https://arxiv.org/pdf/2601.14653v1)

**作者:** Yuyu Liu `[一作]` (Stony Brook University), Tengfei Ma `[通讯]` (Stony Brook University)

**通讯引用:** 4699 | [OpenAlex ID](https://openalex.org/A5086690079)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种基于聚类正则化的最优传输框架 CROT，用于填补单细胞测序数据中的块状缺失。

**💡 创新点**

在最优传输损失中加入聚类中心正则化，既对全局分布对齐又保持细胞子群结构，显著提升填补精度并大幅缩短运行时间。

**🔧 技术方法**

采用熵正则化的 Sinkhorn 算法与 k‑means 聚类相结合的最优传输优化，并使用 Adam 优化器迭代更新缺失值。

**📊 数据集**

在 CITE‑seq、BMMC‑Multiome（Multiome）和 PBMC 三大公开单细胞测序数据集上进行实验。

**📈 对比分析**

与 MAGIC、scImpute、SAVER、scGNN、ALRA、DCA、JAMIE、scBFP、scButterfly 等方法对比，CROT 在 RMSE、MAE、PCC、ARI、NMI、Purity 等指标上均居首位，同时运行时间仅为其他方法的 1–2%。

**⚠️ 局限性**

仅针对结构化块状缺失；对完全随机缺失或高比例缺失的适用性未系统验证；需要预先确定合适的聚类数 k，若 k 选错可能影响结果。

---

## 103. A comprehensive overview of deep learning models for object detection from videos/images

**arXiv ID:** 2601.14677 | [PDF](https://arxiv.org/pdf/2601.14677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 104. From Agent Simulation to Social Simulator: A Comprehensive Review (Part 2)

**arXiv ID:** 2601.14296 | [PDF](https://arxiv.org/pdf/2601.14296v1)

**作者:** Xiao Xue `[一作]` (Tianjin University), Fei-Yue Wang `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了将代理建模与实验方法相结合的计算实验框架，并在O2O平台骑手“卷入”案例中验证其可行性。

**💡 创新点**

创新点在于将横向因果推断与纵向因果解释统一进多层级实验流程，并利用大语言模型实现动态代理学习，提升了对复杂系统演化机制的揭示。

**🔧 技术方法**

使用代理基础建模(ABM)、大语言模型、强化学习、蒙特卡洛模拟和结构方程模型等技术。

**📊 数据集**

基准数据来源为Zomato餐饮配送订单数据，模拟中使用100名骑手代理。

**📈 对比分析**

通过与真实数据的平均工作时长、订单量与R方等指标对比，模拟误差均在0.05小时内，相关性r=0.98，表明模型精度高。

**⚠️ 局限性**

局限包括模型对真实复杂性的可外推性不足、仅验证了单一行业案例、对随机性与环境不确定性处理不完善，以及缺乏跨平台验证。

---

## 105. Direct and Converse Theorems in Estimating Signals with Sublinear Sparsity

**arXiv ID:** 2601.14621 | [PDF](https://arxiv.org/pdf/2601.14621v1)

**作者:** Keigo Takeuchi `[一作]` (Toyohashi University of Technology), Keigo Takeuchi `[通讯]` (Toyohashi University of Technology)

**通讯引用:** 1027 | [OpenAlex ID](https://openalex.org/A5101910059)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

在子线性稀疏 regime 下研究通过 AWGN 信道的 MIMO 系统，对有限字母稀疏信号进行最大似然估计并推导其性能极限；

**💡 创新点**

首次将信息论与稀疏恢复结合，给出稀疏信号最大似然估计的误差上界和可实现的误码率阈值；

**🔧 技术方法**

采用信息论工具（互信息、KL 散度、随机编码、Chernoff/大偏差等）、图论（Hall 婚配定理）以及矩阵与随机过程分析等技术；

**📊 数据集**

使用理论上高维随机高斯测量矩阵作为实验数据，无需具体公开数据集；

**📈 对比分析**

与 AMP、LASSO 等传统算法进行理论和仿真比较，表明在低噪声高信噪比下最大似然估计能实现零误差，性能优于现有基线；

**⚠️ 局限性**

主要限制在于最大似然算法计算复杂度高、实际实现困难，且阈值分析对信道参数假设较强。

---

## 106. Gaussian Based Adaptive Multi-Modal 3D Semantic Occupancy Prediction

**arXiv ID:** 2601.14448 | [PDF](https://arxiv.org/pdf/2601.14448v1)

**作者:** A. Enes Doruk `[一作]` `[通讯]`, A. Enes Doruk

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

无法完成总结，因缺少论文内容

**💡 创新点**

-

**🔧 技术方法**

-

**📊 数据集**

-

**📈 对比分析**

-

**⚠️ 局限性**

-

---

## 107. Intelligent Power Grid Design Review via Active Perception-Enabled Multimodal Large Language Models

**arXiv ID:** 2601.14261 | [PDF](https://arxiv.org/pdf/2601.14261v1)

**作者:** Taoliang Tan `[一作]` (Yangjiang Yangxi Power Supply Bureau, Guangdong Power Grid Co., Ltd.), Si Shi `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于预训练多模态大语言模型的三阶段智能电网图纸审查框架，模拟人工专家的全局语义理解、局部高分辨率信息提取以及置信度驱动的错误诊断。

**💡 创新点**

通过提示工程实现主动感知的语义区域建议、细粒度高分辨率信息采集以及置信度感知的综合错误诊断模块，解决传统方法在超高分辨率图纸上的信息损失和语义理解不足问题。

**🔧 技术方法**

使用了Qwen‑VL 2.5多模态大语言模型、Prompt Engineering、OCR（Tesseract）、非极大值抑制、坐标转换等技术，并结合多分辨率图像生成。

**📊 数据集**

在12幅4K子站图纸的内部实验集上评估，图纸包含符合与不符合CT二次电路单点接地规则的人工标注。

**📈 对比分析**

通过留一交叉验证与传统被动MLLM推理对比，三阶段框架在区域建议精度0.70、召回0.65、IoU0.60以及违规检测精度0.75、召回0.80、F1 0.77，显示出更高的检测准确率与可靠性。

**⚠️ 局限性**

局限在数据集规模小、标注人工、对OCR误读和符号模糊敏感，且在更大多样化样本上的泛化性尚未验证，需要扩展数据集与进一步优化模型与提示。

---

## 108. engGNN: A Dual-Graph Neural Network for Omics-Based Disease Classification and Feature Selection

**arXiv ID:** 2601.14536 | [PDF](https://arxiv.org/pdf/2601.14536v1)

**作者:** Tiantian Yang `[一作]` (University of Idaho), Ching-Ti Liu `[通讯]` (Boston University)

**通讯引用:** 76669 | [OpenAlex ID](https://openalex.org/A5082644230)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一个双图神经网络engGNN，用外部生物网络和由XGBoost生成的任务特定图共同进行疾病分类和特征选择

**💡 创新点**

创新点在于同时利用无向外部知识图和有向数据驱动图，实现更丰富的表征，并给出可解释的特征重要性得分

**🔧 技术方法**

技术包括图嵌入深度前馈网络（GEDFN）、XGBoost图生成、双图联合嵌入、特征重要性评分和KEGG通路富集分析

**📊 数据集**

使用了仿真高维“large‑p, small‑n”数据以及真实的阿尔茨海默病血液基因表达数据（GSE140831）

**📈 对比分析**

与多种基线（单图GEDFN、XGBoost、RF、DFN等）比较，engGNN在准确率、ROC‑AUC、F1‑score和特征选择的ROC‑AUC/PR‑AUC上均取得了显著或最佳表现，尤其在阿尔茨海默病数据中显著优于其他方法

**⚠️ 局限性**

局限性包括仅针对二分类单一组学数据；对多类别或多组学的推广仍待验证；外部网络的完整性和偏差仍可能影响结果；需要更广泛的实验和外部验证

---

## 109. CoScale-RL: Efficient Post-Training by Co-Scaling Data and Computation

**arXiv ID:** 2601.14695 | [PDF](https://arxiv.org/pdf/2601.14695v1)

**作者:** Yutong Chen `[一作]` (Tsinghua University), Ji Wu `[通讯]` (Tsinghua University)

**通讯引用:** 5267 | [OpenAlex ID](https://openalex.org/A5029547618)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种同时扩大每题解答数量和强化 RL 采样规模的后训练扩容策略，并通过 Re‑distillation 进行模型合并，显著提升小型 LRM 的长篇推理能力。

**💡 创新点**

创新点在于：① 通过“多解法数据扩容”而非单纯增大数据集来突破模型能力边界；② 在 RL 阶段引入可变 Rollout N（按难度分组）实现计算效率最大化；③ 将 RL 结果再转为 SFT 数据，利用 Re‑distillation 效率地合并多进程训练成果。

**🔧 技术方法**

使用技术包括：大语言模型 Qwen2.5‑0.5B 的 SFT 与 RL（GRPO 等），多解法数据扩容、Rollout N 自适应分组、Re‑distillation 模型合并、SDE 理论分析指导 η/N 取值。

**📊 数据集**

主要数据集为 OpenMathReasoning（精选组合与概率问题），以及四大数学推理基准（MATH‑500、AMC12、OpenMathReasoning、OlympiadMATH）和通用推理基准 Reasoning GYM。

**📈 对比分析**

与现有 SFT、RL、Scale‑RL、SAPO、GRPO、COMPASS、SpeedControl 等基线相比，平均提升 3.76×（如 MATH‑500 40.7% 对比 24.7%），在大部分数学基准上取得显著优于 baseline 的 Pass@1/Pass@16 成绩；在通用推理基准上提升有限。

**⚠️ 局限性**

局限性：① 对于组合与概率类数学问题表现突出，通用推理任务提升有限；② 需要大量高质量多解法数据，收集成本高；③ 目前验证主要集中在 0.5B、1.5B、3B、7B 规模模型，尚未系统评估更大模型或跨领域的通用性。

---

## 110. DDSA: Dual-Domain Strategic Attack for Spatial-Temporal Efficiency in Adversarial Robustness Testing

**arXiv ID:** 2601.14302 | [PDF](https://arxiv.org/pdf/2601.14302v1)

**作者:** Jinwei Hu `[一作]`, Xiaowei Huang `[通讯]` (University of Liverpool)

**通讯引用:** 5240 | [OpenAlex ID](https://openalex.org/A5020085889)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了双域战略攻击（DDSA）框架，用以在资源受限的实时图像处理系统中进行高效的对抗鲁棒性测试。

**💡 创新点**

创新点在于：① 通过情景感知的时间触发函数，动态判断何时对帧进行攻击；② 结合可解释AI（Integrated Gradients）实现仅在关键像素区域施加扰动，从而在时间和空间上实现显著资源节约。

**🔧 技术方法**

使用的技术包括：Integrated Gradients 进行像素重要性评估；Monte Carlo Dropout 估计模型不确定性；FGSM/PGD 生成对抗样本；ResNet‑18 作为基准分类器；以及基于场景与数据权重的优先级评分机制。

**📊 数据集**

实验数据集：CIFAR‑10、CIFAR‑100、Fashion‑MNIST 以及 VOC2012，全部使用训练好的 ResNet‑18。

**📈 对比分析**

与传统全图攻击（FGSM/PGD）对比：在仅占 40% 像素覆盖率时，攻击成功率仍保持 98% 以上；时间消耗下降 80–97%；整体运算分数在高吞吐量情景下大幅提升，传统方法在复杂数据集上甚至出现负分。

**⚠️ 局限性**

局限性：需手工设定情景优先级与阈值，IG 与显著图计算仍带来额外算力开销；对抗性攻击效果在极不确定预测或某些模型架构下可能下降；实验仅在单一防御策略下验证，对自适应防御的鲁棒性尚未评估。

---

## 111. Gaming the Judge: Unfaithful Chain-of-Thought Can Undermine Agent Evaluation

**arXiv ID:** 2601.14691 | [PDF](https://arxiv.org/pdf/2601.14691v1)

**作者:** Muhammad Khalifa `[一作]` (University of Michigan), Honglak Lee `[通讯]` (University of Michigan)

**通讯引用:** 40378 | [OpenAlex ID](https://openalex.org/A5108652283)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM判决者在评估非可验证任务时因链式推理（CoT）被操纵导致错误判断的脆弱性进行了系统评估

**💡 创新点**

首次提出CoT操纵的体系化分类，量化不同风格/内容操纵对VLM判决者误判率的影响，并证明即使在固定动作与观察的条件下，CoT也能显著提升假阳性率

**🔧 技术方法**

使用了受控的CoT重写策略（风格增强、反思推理、进度虚构、环境归因、任务重解读），对多种VLM判决者进行对比实验，并尝试提示、基准化与计算扩展等训练无关的防御方法

**📊 数据集**

在800条Web交互轨迹（来自659个任务、10类）上实验，任务来自WebArena、AssistantBench、WorkArena、AgentRewardBench以及自建数据集

**📈 对比分析**

与多款前沿VLM判决者（GPT‑4o、GPT‑5‑mini、Claude‑Sonnet‑4、Qwen‑2.5‑72B、GLM‑4.1V、Pixtral‑12B等）对比，发现进度虚构可将FPR提升20–30个百分点，甚至高达90%；提示与基准化虽能降低误判，但仍无法完全消除；提升鲁棒性还会导致召回率下降

**⚠️ 局限性**

研究仅聚焦Web代理，未覆盖能操纵环境的场景；只探索了训练无关的提示/基准化/计算扩展等防御，未尝试训练或显式校验方法；CoT重写仅基于单一模型，可能不足以涵盖所有攻击方式

---

## 112. Optimising Cylindrical Algebraic Coverings for use in SMT by Solving a Set Covering Problem with Reasons

**arXiv ID:** 2601.14424 | [PDF](https://arxiv.org/pdf/2601.14424v1)

**作者:** Abiola Babatunde `[一作]`, AmirHosein Sadeghimanesh `[通讯]` (Coventry University)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5084273274)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在SMT求解中使用CDCAC时冲突核心最小化问题，并提出了Set Covering Problem with Reasons (SCPR) 模型。

**💡 创新点**

创新点在于把经典的Set Covering Problem扩展为带理由的版本，提出通用Beasley归约规则，并通过线性规划实现高效求解，形成一个两阶段优化管线。

**🔧 技术方法**

使用的技术包括SCPR建模、通用Beasley归约、二进制线性规划（LP）、SAT/MaxSAT编码及多种启发式搜索（贪心、GA、SA、PSO、MLSES‑CC）。

**📊 数据集**

使用的实验数据集是SMT‑LIB 2024的 QF_NRA 基准，覆盖12个应用领域，产生 2851 个独特的SCPR实例。

**📈 对比分析**

实验表明归约后仅有 4.5% 的实例需进一步求解，LP 在 <1 ms 内完成，准确率 100%；相比之下 SAT/MaxSAT 在速度上略优但仍保持 100% 正确性，启发式方法速度最快但准确率下降至 90–95%。

**⚠️ 局限性**

局限性包括对某些领域（几何、Inequality）归约效果不足，仍需 LP 求解；未在完整的增量SMT求解流程中实现；以及未探讨更大规模或更复杂约束下的性能。

---

## 113. Dissecting Performance Degradation in Audio Source Separation under Sampling Frequency Mismatch

**arXiv ID:** 2601.14684 | [PDF](https://arxiv.org/pdf/2601.14684v1)

**作者:** Kanami Imamura `[一作]` (National Institute of Advanced Industrial Science and Technology), Hiroshi Saruwatari `[通讯]` (University of Tokyo)

**通讯引用:** 7628 | [OpenAlex ID](https://openalex.org/A5003814223)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究在音频源分离模型中对未训练采样率进行重采样导致性能下降的问题，并提出三种改进方法。

**💡 创新点**

创新点在于引入噪声扰动插值核（noisy-kernel）以及可学习插值核（trainable-kernel），通过在核中加入高频成分缓解性能下降。

**🔧 技术方法**

使用了窗口化 sinc 插值、Gaussian 噪声添加、MLP 作为可学习核、以及深度学习源分离模型（如 BSRNN、Conv‑TasNet 等）。

**📊 数据集**

实验数据集为 MUSDB18‑HQ，测试不同采样率（8kHz 等）。

**📈 对比分析**

通过与传统重采样、后期加噪等方法对比，发现 noisy‑kernel 和 trainable‑kernel 能在多种模型上消除或显著减少 SDR 下降，性能优于传统方法。

**⚠️ 局限性**

局限在于仅针对音乐源分离任务验证，对语音或其他任务的推广以及更高采样率下的通用性尚未探究。

---

## 114. Large Language Models for Large-Scale, Rigorous Qualitative Analysis in Applied Health Services Research

**arXiv ID:** 2601.14478 | [PDF](https://arxiv.org/pdf/2601.14478v1)

**作者:** Sasha Ronaghi `[一作]` (Stanford University), Sara Singer `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文设计并验证了一个任务特定的人工-LLM定性分析框架，应用于在12个联邦合格医疗中心开展的糖尿病护理比较案例研究中完成两项关键分析任务：①利用LLM对研究者生成的站点层面总结进行主题化组织并辅助跨站点综合报告；②利用检索增强生成（RAG）与LLM对167份访谈稿进行演绎编码，生成支持性引述的要点矩阵，进一步完善干预方案。

**💡 创新点**

创新点在于提出了四步通用框架（定义任务、设计人工-LLM方法、在小规模评估、在全量任务应用），强调在大规模、保密数据场景下保持研究者解释控制与方法透明度；同时首次将LLM嵌入真实跨站点健康服务研究工作流，并系统评估LLM在主题组织和编码生成中的效率与质量。

**🔧 技术方法**

技术主要包括：OpenAI ChatGPT‑4o 与 o1 进行主题化组织与跨站点综合生成；检索增强生成（RAG）结合文本嵌入（OpenAI text‑embedding‑3‑large）与 Qdrant 向量数据库，自动检索访谈片段；Python 脚本与 SecureGPT 接口实现批量调用；自动化脚本合并并验证 LLM 输出中的引用。

**📊 数据集**

数据集为：来自12个联邦合格医疗中心的167份访谈稿（约1,327,000词），以及每个站点在22个护理领域的3–5条要点摘要（约31,200词）。

**📈 对比分析**

通过与人工编码的对比，小规模评估表明：①在主题化组织上，LLM的效率提高30–55%，输出与人工主题高度一致；②在跨站点综合报告中，LLM的内容需要人工润色以提升可操作性；③在演绎编码中，LLM能快速生成约30条带引述的要点，节省约310小时的人工编码时间，但在上下文理解与观点深化上仍需人工验证，整体性能满足研究者对效率与可解释性的双重要求。

**⚠️ 局限性**

局限包括：LLM在长文本理解与全局语境把握上仍有限，导致过度概括与上下文缺失；生成的摘要缺乏细致阐释，需要人工后处理；LLM的输出可能受训练数据偏见影响，易出现“例子偏差”和“积极性偏差”；在跨站点综合报告中，LLM容易将不相关或模糊信息纳入，需人工校正；整体方法对模型与接口的可访问性和成本有一定依赖。

---

## 115. Uncovering and Understanding FPR Manipulation Attack in Industrial IoT Networks

**arXiv ID:** 2601.14505 | [PDF](https://arxiv.org/pdf/2601.14505v1)

**作者:** Mohammad Shamim Ahsan `[一作]` (Pennsylvania State University), Peng Liu `[通讯]` (Pennsylvania State University)

**通讯引用:** 13569 | [OpenAlex ID](https://openalex.org/A5100346874)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于MQTT协议的FPR操纵攻击（FPA），通过在合法IoT数据包中注入微小的空格填充，使得机器学习网络入侵检测系统（ML‑NIDS）误将其判为攻击流量。

**💡 创新点**

创新点在于首次揭示并实现将合法流量故意误分类为攻击的“误报攻击”，不依赖梯度或生成对抗网络，攻击成本低、检测难度大，且能显著扰乱安全运维中心的事件处理。

**🔧 技术方法**

攻击方法主要利用MQTT控制报文结构（CONNECT、PUBLISH）中的可调字段和主题、有效负载的空格填充；评估则使用四个公开的CNN/GRU/LSTM组合模型，并结合XAI（t‑SNE、UMAP、SHAP）对误判机制进行解释。

**📊 数据集**

实验数据来自Edge‑IIoTset 2022数据集，包含14类攻击和1类正常流量，覆盖了MQTT、TCP/IP等多层协议特征。

**📈 对比分析**

在四个模型上，FPA攻击成功率达到80.19%–100%，导致误报率飙升并使SOC平均等待时间从约56秒增至154秒，单日误报延迟达11.41小时；对抗训练虽能消除误报，但会导致多类真实攻击检测性能下降10%–99%。

**⚠️ 局限性**

局限性包括：仅适用于MQTT协议且需已被攻破的IoT设备；攻击方式在不同ACL策略、QoS设置或加密通道（TLS/SSL）下效果不确定；对抗训练虽可提升鲁棒性，但会显著牺牲正常流量与其他攻击类别的检测精度。

---

## 116. Anatomically Guided Latent Diffusion for Brain MRI Progression Modeling

**arXiv ID:** 2601.14584 | [PDF](https://arxiv.org/pdf/2601.14584v1)

**作者:** Cheng Wan `[一作]` (Cornell University), Qingyu Zhao `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 1401 | [OpenAlex ID](https://openalex.org/A5039683634)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种基于解剖引导的潜在扩散模型（AG-LDM）用于脑MRI进展建模。

**💡 创新点**

创新点包括：1) 在潜在空间直接拼接基线解剖、噪声跟随影像与临床协变量，实现统一端到端条件；2) 通过轻量化分割教师WarpSeg在自编码器微调与扩散训练阶段提供解剖监督，显著提升解剖一致性；3) 消除ControlNet和辅助疾病进展模块，简化三阶段训练流程。

**🔧 技术方法**

技术方法包括：潜在扩散模型（DDPM/DDIM）、变分自编码器、3D U‑Net、Dice/边界损失、分割引导、WASABI解剖一致性评估。

**📊 数据集**

使用数据集：ADNI（31,713对纵向配对）用于训练、验证和内部测试；OASIS‑3（1,965对）用于零样本外部验证。

**📈 对比分析**

与4D‑DaniNet、Latent‑SADM、CounterSynth、BrLP进行比较。AG‑LDM在ADNI内部和OASIS‑3外部均获得最佳或最优的图像质量（MSE/PSNR）、ROI体积MAE（下降15–20%）和WASABI分数，且对临床协变量的敏感度提升3.5–31.5倍。

**⚠️ 局限性**

局限性：仅预测单一未来时间点，缺乏完整多时点连续轨迹；模型依赖WarpSeg的预训练分割性能；在不同疾病或解剖结构的泛化能力仍待进一步验证。

---

## 117. Query-Efficient Agentic Graph Extraction Attacks on GraphRAG Systems

**arXiv ID:** 2601.14662 | [PDF](https://arxiv.org/pdf/2601.14662v1)

**作者:** Shuhua Yang `[一作]` (Pennsylvania State University), Suhang Wang `[通讯]` (Pennsylvania State University)

**通讯引用:** 18039 | [OpenAlex ID](https://openalex.org/A5011048500)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究在黑盒环境下对GraphRAG系统进行结构化知识图谱的高效窃取；

**💡 创新点**

提出Agentic Graph Extraction Attack (AGEA)，结合新颖度驱动的探索-利用策略、两阶段检索过滤以及外部图内存，实现对隐藏实体-关系图的快速恢复；

**🔧 技术方法**

利用LLM生成查询、正则解析候选实体关系、LLM过滤器、图内存与新颖度评分、ε-greedy策略实现自适应查询；

**📊 数据集**

在医学、农业以及文学（Novel）三类图谱上进行实验，使用Microsoft GraphRAG与LightRAG构建的知识图；

**📈 对比分析**

与四种基线（TGTB、PIDE、CopyBreakRAG、IKEA）在固定1000次查询预算下对比，AGEA在节点/边泄露率和精度均显著优于基线，节点泄漏率最高达96.4%、边泄漏率达95.9%；

**⚠️ 局限性**

依赖于可控的结构化输出与正则解析，若受限输出格式或部署时防御策略可能降低效果；未考虑实时监测或查询重写等防御。

---

## 118. End-to-End Transformer Acceleration Through Processing-in-Memory Architectures

**arXiv ID:** 2601.14260 | [PDF](https://arxiv.org/pdf/2601.14260v1)

**作者:** Xiaoxuan Yang `[一作]` (University of Virginia), Yiran Chen `[通讯]` (Duke University)

**通讯引用:** 25428 | [OpenAlex ID](https://openalex.org/A5058073627)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于处理内存（PIM）的Transformer加速架构，通过矩阵分解、子矩阵流水线、KV缓存压缩、以及关联检索（CAM）等技术，降低计算、内存和带宽瓶颈。

**💡 创新点**

创新点包括① 用矩阵分解消除注意力块的计算-写入-计算（CWC）依赖，并通过子矩阵流水线提升跨barrier利用率；② 现场级的Cascade Pruning‑Quantization（CPQ）与层级量化扩展（HQE）实现KV缓存的动态压缩与量化；③ 将注意力机制重新表述为最近邻检索，并采用CAM实现稀疏化计算。

**🔧 技术方法**

使用了ReRAM/FeFET/Flash CAMs的PIM体系结构、子矩阵流水线、KV缓存压缩与量化、动态层级量化、以及CAM关联检索等技术。

**📊 数据集**

在大型语言模型（如OPT‑6.7B、PaLM 540B）上进行推理评估，使用标准LLM推理场景（无特定公开数据集）。

**📈 对比分析**

与Nvidia A100 GPU和FlightLLM进行基准对比，能量效率提升至159.9×（GPU）/34.8×（FlightLLM），吞吐量提升49.6×/29.2×。

**⚠️ 局限性**

局限性包括：对极大上下文仍受O(N²) softmax 计算和数据搬运限制；压缩与量化在某些层可能引入精度损失；CAM等硬件资源占用和面积成本未在论文中做详尽评估。

---

## 119. SearchGym: Bootstrapping Real-World Search Agents via Cost-Effective and High-Fidelity Environment Simulation

**arXiv ID:** 2601.14615 | [PDF](https://arxiv.org/pdf/2601.14615v1)

**作者:** Xichen Zhang `[一作]` (Hong Kong University of Science and Technology), Jiaya Jia `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 73052 | [OpenAlex ID](https://openalex.org/A5052856441)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了高保真离线搜索仿真环境SearchGym，并在其上使用强化学习训练搜索代理。

**💡 创新点**

通过生成可验证知识图谱、对齐语料库及多层次评估路径，消除训练中的噪声奖励，保证每个任务可被解决；并结合两阶段课程学习提升代理的多跳推理能力。

**🔧 技术方法**

采用大型语言模型生成文档、检索引擎、SearchGym-RL框架（GRPO强化学习+两阶段课程）以及搜索与访问两种动作原语。

**📊 数据集**

使用自制的验证知识图谱与对齐文档语料（约3.6k节点、41k QA对），以及多种公开评测基准（NQ、HotpotQA、Bamboogle、GAIA、xbench-DeepSearch等）。

**📈 对比分析**

在10个多样化基准上与直接推理、RAG、Search-R1、ZeroSearch及ASearcher等对比，SearchGym-RL在大多数任务上实现了10–80%的相对提升，并在Sim‑to‑Real迁移中显著优于使用真实Web API的基线。

**⚠️ 局限性**

限制在于仿真语料仍为人工生成，可能缺乏真实Web的噪声与多样性；以及对大规模LLM的依赖导致训练成本较高。

---

## 120. Project Aletheia: Verifier-Guided Distillation of Backtracking for Small Language Models

**arXiv ID:** 2601.14290 | [PDF](https://arxiv.org/pdf/2601.14290v1)

**作者:** Aradhya Dixit `[一作]` (Wake Technical Community), Jai Telang `[通讯]` (Algoverse)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小语言模型上通过教师示例引导的反证式蒸馏，使模型学习到在SAT约束满足问题中检测冲突并回溯修正的过程，而非仅训练最终答案；

**💡 创新点**

核心创新是将错误修复过程（冲突检测与回溯）视为可蒸馏的行为模式，而不是传统的单向答案训练；

**🔧 技术方法**

采用教师回溯示例生成、符号PySAT验证、LoRA参数高效微调、4‑bit量化与Unsloth训练框架；

**📊 数据集**

使用约500条手工标注的SAT实例生成的验证后轨迹，形成496条“黄金”训练集；

**📈 对比分析**

对比线性化无回溯的控制模型和全回溯的Aletheia模型，基于40个hold‑out SATBench实例测算回溯事件率（BER），Aletheia模型在单轮LoRA微调后出现5%回溯率，而控制模型为0%；

**⚠️ 局限性**

局限在于回溯事件率低、仅评估SAT问题、训练数据量小且受教师启发式偏差影响，且缺乏更细粒度的行为评估与广泛的跨任务验证。

---

## 121. FARE: Fast-Slow Agentic Robotic Exploration

**arXiv ID:** 2601.14681 | [PDF](https://arxiv.org/pdf/2601.14681v1)

**作者:** Shuhao Liao `[一作]` (Beihang University), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**通讯引用:** 1229 | [OpenAlex ID](https://openalex.org/A5069667034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种层次化自主探索框架FARE，结合大型语言模型进行全局语义推理和强化学习策略实现局部决策，能够根据环境描述生成可解释的探索策略并引导机器人高效完成未知环境的全覆盖。

**💡 创新点**

创新点在于：1）采用快慢思维分层结构，将全局语义推理与局部运动决策解耦；2）利用LLM在结构化图（经过社区检测与模数化剪枝）上进行图推理，生成全局路径；3）在RL奖励中加入遵循全局路径的指令跟随项，使得局部策略既能保持对全局目标的连贯性，又能对局部观测做出灵活响应；4）使用简洁的自然语言环境描述来生成可解释的探索策略。

**🔧 技术方法**

技术包括：大型语言模型（如Qwen3‑14B）进行文本解析与图推理；社区检测与模数化剪枝构建稀疏全局图；基于图注意力的RL策略网络；指令跟随奖励设计；在ROS与Jetson AGX Orin上实现边缘LLM推理；使用16通道3D LiDAR与FastLIO2实现地图构建与里程计。

**📊 数据集**

在Gazebo仿真环境中使用三种典型环境（室内、森林、仓库）进行多次随机实验；在真实环境中使用Agilex Scout‑mini机器人在200m×130m校园教学楼内执行全覆盖任务。对比实验采用TARE、DSVP、ARiADNE、HEADER四个先进基线。

**📈 对比分析**

与四个基线对比：在室内环境下FARE与基线相近；在森林和仓库环境中，FARE的行驶距离和任务完成时间分别比最优基线缩短约15–30%（仓库中最显著，距离从492m降至441m，时间从286s降至252s）。实验表明FARE能显著降低不必要的回溯，提升探索效率。

**⚠️ 局限性**

局限性包括：1）LLM推理成本高，导致实时推理时延较大；2）目前仅针对二维平面规划，未扩展至3D空间；3）系统仍为单机器人，缺乏多机器人协同与在线语义感知；4）对极端动态环境的鲁棒性待进一步验证。

---

## 122. Break-Resilient Codes with Loss Tolerance

**arXiv ID:** 2601.14623 | [PDF](https://arxiv.org/pdf/2601.14623v1)

**作者:** Canran Wang `[一作]` (Washington University in St. Louis), Netanel Raviv `[通讯]` (Washington University in St. Louis)

**通讯引用:** 916 | [OpenAlex ID](https://openalex.org/A5048888817)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出一种新的错误纠正码（(t,s)-BRC），能够在码字被任意 t 次碎裂且总共最多丢失 s 位的情况下，完整恢复原始信息。

**💡 创新点**

创新点在于统一并推广了破碎恢复码与删码两种先前的模型，给出了信息理论下的最优冗余下界，并提供了接近该下界的构造方案。

**🔧 技术方法**

采用了基于 Reed–Solomon 纠错的定位多项式方法、互不相干码作为分隔符、以及多项式求根与线性方程组求解等技术来实现编码与解码。

**📊 数据集**

该工作主要为理论研究，没有使用具体实验数据集，所有结论均通过信息论分析与数学证明得出。

**📈 对比分析**

与传统的断裂恢复码和删码相比，(t,s)-BRC 的冗余量为 O(t·log²n + s·log n)，实现了对碎裂与丢失双重攻击的近似最优抵御能力。

**⚠️ 局限性**

局限性包括：编码需先随机选择满足约束的有效字符串，解码复杂度与求根算法相关；当 t 或 s 较大时，冗余量仍可能显著增大；此外，模型假设攻击是完全对抗性的，实际场景可能更为宽松。

---

## 123. RoboBrain 2.5: Depth in Sight, Time in Mind

**arXiv ID:** 2601.14352 | [PDF](https://arxiv.org/pdf/2601.14352v1)

**作者:** Huajie Tan `[一作]`, Shanghang Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了RoboBrain 2.5，实现从语义指令到精确 3D 空间轨迹和时间进度估计的完整闭环控制。

**💡 创新点**

创新点在于“Depth in Sight”三维指涉、测量、追踪能力以及“Time in Mind”基于 hop 归一化的密集时间价值估计与多视角融合。

**🔧 技术方法**

采用 Qwen3‑VL 基础架构，结合 ViT+Transformer、hop‑based 进度标签、三视角融合、逆向一致性校验以及跨加速器并行训练。

**📊 数据集**

使用约 12.4M 样本的统一数据集，包含 Honey‑Data、LLaVA‑Onevision、LVIS、Pixmo‑Points、RoboPoint、PACO‑LVIS、OpenImage、CA‑1M、ScanNet、AgiBot‑Beta、DROID、RoboTwin、EgoDex 等多源 2D/3D 视觉与仿真/人类视频。

**📈 对比分析**

在 2D/3D 空间推理基准（CV‑Bench、CrossPoint、RoboSpatial、RefSpatial、EmbSpatial、MSMU、Q‑Spatial、TraceSpatial、VABench‑V、ShareRobot‑T）以及时间价值评估（GPRM VOC+ / VOC‑）上均取得 SOTA，显著优于 Gemini、GPT‑5.2、Qwen3‑VL‑Inst、RoboBrain‑2.0 等对比模型。

**⚠️ 局限性**

局限在于仍依赖单目 RGB，无法充分处理动态遮挡与实时感知；需要大规模预训练，推理延迟较高，且在真正复杂多机器人协作场景中尚未充分验证。

---

## 124. An Optimized Decision Tree-Based Framework for Explainable IoT Anomaly Detection

**arXiv ID:** 2601.14305 | [PDF](https://arxiv.org/pdf/2601.14305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 125. From Chaos to Clarity: Schema-Constrained AI for Auditable Biomedical Evidence Extraction from Full-Text PDFs

**arXiv ID:** 2601.14267 | [PDF](https://arxiv.org/pdf/2601.14267v1)

**作者:** Pouria Mortezaagha `[一作]` (Ottawa Hospital Research Institute), Arya Rahgozar `[通讯]` (University of Ottawa)

**通讯引用:** 59 | [OpenAlex ID](https://openalex.org/A5006363388)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一套基于OCR与LLM的schema‑constrained提取管线，将全篇医学PDF自动转化为结构化、可审计的记录。

**💡 创新点**

创新点在于将提取过程完全约束在类型化schema、闭合词表和证据门控下，并通过确定性分块、异步调度和句子级证据链实现可审计、可复现的高吞吐量抽取。

**🔧 技术方法**

技术包括Mistral OCR引擎的多页面异步调用、布局感知Transformer、schema‑guided prompt、并发与速率限制、冲突感知合并以及Markdown多模态重构。

**📊 数据集**

使用了734篇公开全文PDF（共7 228页），聚焦于直接口服抗凝剂（DOAC）水平测定的临床与方法学研究。

**📈 对比分析**

通过对50篇样本的专家评估（前后Schema/Prompt迭代）比较，关键变量的正确率从约30%–60%提升至95%–100%；处理吞吐约3.1次/秒，平均11 秒/篇；内部一致性高，未出现致命错误。

**⚠️ 局限性**

局限包括缺乏字符级OCR真值评估、未对图表进行数值提取、缺乏大规模金标准对照、需要领域专门的schema、以及对复杂表格与多模态信息的完整利用不足。

---

## 126. Evaluating Preattentive Features for Detecting Changes in Virtual Environments

**arXiv ID:** 2601.14561 | [PDF](https://arxiv.org/pdf/2601.14561v1)

**作者:** DongHoon Kim `[一作]` (Utah State University), Isaac Cho `[通讯]` (Utah State University)

**通讯引用:** 607 | [OpenAlex ID](https://openalex.org/A5062185996)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在沉浸式VR环境中设置6×6立方体网格，考察深度、尺寸和角度三种前瞻性视觉特征在不同视觉复杂度（单/双/三特征）及空间分隔（孤立/群组）条件下的变化检测表现。

**💡 创新点**

创新点在于：①将前瞻性特征迁移至3D沉浸环境，系统检验深度在多特征场景中的鲁棒性；②引入空间分隔因素，揭示孤立布局显著提升检测效率；③将视觉复杂度量化为特征数量，并与主观工作负荷并行评估。

**🔧 技术方法**

技术手段包括：Vive Pro HMD头戴显示、Unity 2022.3.3f1渲染、手持控制器射线投射交互、头部转动触发变化、收集检测时间、准确率、超时率和NASA‑TLX问卷。

**📊 数据集**

数据集：自制的6×6红色立方体视觉刺激集，无外部公开数据集。每个实验条件下生成随机对象布局，保持特征分布均匀。

**📈 对比分析**

比较方法：采用两因素/三因素重复测量ANOVA，检验特征类型、特征数量、空间分隔对检测时间、准确率和超时率的影响；NASA‑TLX用于工作负荷对比。结果显示：深度特征在所有复杂度下最快、最准确；单特征条件下检测时间最短、准确率最高；孤立布局优于群组，尤其在双/三特征时优势更显著。

**⚠️ 局限性**

局限性包括：①样本仅为大学生，年龄、种族多样性不足；②实验任务涉及头部转动，难以完全分离预瞩性与有意识搜索；③未排除头部转动时间对检测时间的影响；④仅使用自制立方体数据，缺乏真实场景或多属性物体的验证。

---

## 127. Specifying and Verifying RDMA Synchronisation (Extended Version)

**arXiv ID:** 2601.14642 | [PDF](https://arxiv.org/pdf/2601.14642v1)

**作者:** Guillaume Ambal `[一作]` (Imperial College London), Azalea Raad `[通讯]` (Imperial College London)

**通讯引用:** 734 | [OpenAlex ID](https://openalex.org/A5083721882)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文通过构建远程 RMW 的正式语义，对 RDMA 锁进行设计、实现并在此语义下进行形式化验证，提出了弱锁、强锁和节点锁三种可组合的锁库，并在此基础上实现了一个强一致性 RDMA 库。

**💡 创新点**

主要创新点在于：①首次对 RDMA 远程读-写-改写（RMW）操作给出正式语义；②基于此语义提出可组合的锁库与强一致性模型；③通过声明式与操作式相结合的验证框架完成锁实现的形式化证明。

**🔧 技术方法**

采用声明式模型、操作式语义、可组合对象库（Library of Composable Objects）以及 Hoare 逻辑与一致性判定技术进行形式化验证；同时使用 NVIDIA 等厂商技术手册进行模型校验。

**📊 数据集**

本文无实验数据集，全部工作基于理论建模与形式化证明。

**📈 对比分析**

本文未进行性能评估或与现有实现的对比，仅在逻辑上证明实现满足所定义的锁语义与强一致性要求；若需性能对比需自行部署实验。

**⚠️ 局限性**

局限性包括：①仅覆盖 TSO CPU 模型，未处理 SC 或其他内存模型；②模型与实现仅涵盖基本 RDMA 指令集，缺少对高级 RDMA 操作（如原子队列、流等）的支持；③未提供实际系统实验验证，只给出形式化证明。

---

## 128. SPIRIT: A Design Framework To Support Technology Interventions for Spiritual Care Within and Beyond the Clinic

**arXiv ID:** 2601.14435 | [PDF](https://arxiv.org/pdf/2601.14435v1)

**作者:** C. Estelle Smith `[一作]` (Colorado School of Mines), Jesan Ahammed Ovi `[通讯]` (Colorado School of Mines)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5062471065)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过重新分析原始共创研讨会数据和对22名精神护理从业者的新访谈，提出并验证了SPIRIT框架，用于指导数字精神护理技术的设计与实施。

**💡 创新点**

1）对“spiritual support”定义的修订，形成更符合专业实践的“spiritual care”定义；2）提出SPIRIT框架，包含三项前提条件（开放性、安全空间、辨识与表达需求）和六项设计维度（有爱之在、意义建构、技术使用程度、地点、关系亲密度、时序性），为HCI设计精神护理提供系统化方法。

**🔧 技术方法**

采用扎根理论方法（GTM）进行质性数据编码与归纳；使用访谈、共创工作坊的文本数据进行开放编码、轴心编码和主题聚类；未涉及算法实现，而是形成设计框架。

**📊 数据集**

共创工作坊原始数据（n=34，涵盖患者、护理者、宗教领袖、医疗专业人员、CaringBridge员工）与22名精神护理从业者访谈数据，合计56名参与者。

**📈 对比分析**

本文未进行量化比较或性能评估；主要通过与已有文献和专业实践对比验证框架的合理性，缺乏实验数据或系统实现的效果评估。

**⚠️ 局限性**

样本局限在美国范围内，缺少军事、监狱等特殊情境的从业者；框架尚未在真实数字系统中实施验证；缺乏定量指标评估精神护理干预效果；对技术可行性与伦理风险的讨论有限。

---

## 129. Social Caption: Evaluating Social Understanding in Multimodal Models

**arXiv ID:** 2601.14569 | [PDF](https://arxiv.org/pdf/2601.14569v1)

**作者:** Bhaavanaa Thumu `[一作]` (Carnegie Mellon University), Louis-Philippe Morency `[通讯]` (Carnegie Mellon University)

**通讯引用:** 30987 | [OpenAlex ID](https://openalex.org/A5081398601)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了 Social Caption 框架，用来从社交推断、整体社交分析和定向社交分析三维度系统评估多模态大语言模型的社交理解能力。

**💡 创新点**

创新点在于引入多维度评测、将评测标准与社会互动理论对齐，并首次将 MLLM 评审器作为可扩展的自动评估工具，揭示模型架构与规模对社交理解的细微影响。

**🔧 技术方法**

采用多模态 LLM（如 Qwen2、InternVL、Gemini 等）结合视频编码、语音转写、评测提示与 Likert 量表，对生成文本进行自动与人工评分。

**📊 数据集**

使用公开的 Social‑IQ 2.0 数据集（145段 1 分钟视频、943 个多选问题），并配合 WhisperX 转写保留对话信息。

**📈 对比分析**

通过 SI 准确率、HSA/DSA 30 分制平均分与人类评审的一致性进行对比，发现部分 7–8B 开源模型在 SI 甚至超过部分 100B 级别封闭模型，在 HSA/DSA 上可与之匹敌；同时 MLLM 评审器与人工评审高度相关。

**⚠️ 局限性**

主要局限在于仅使用 1 分钟英文短视频，难以覆盖长时上下文与跨语言/多文化场景，且评审依赖英语母语与美国背景的标注者。

---

## 130. Search over Self-Edit Strategies for LLM Adaptation

**arXiv ID:** 2601.14532 | [PDF](https://arxiv.org/pdf/2601.14532v1)

**作者:** Alistair Cheong `[一作]` (Carnegie Mellon University), Dustin Miao `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在LLM自我改进的框架下，让模型自己生成自我编辑模板，并通过单轮自我学习来改进其权重。

**💡 创新点**

创新点在于突破SEAL固定模板限制，让模型在有限的自监督下一步更新空间内探索并生成自己的数据生成策略和超参数，从而提升自我改进的多样性与效果。

**🔧 技术方法**

使用技术包括基于Qwen3-8B的自监督下一词预测（NTP）微调、ReST^EM元学习更新、轻量级模板档案机制以及自评判式的评估管线。

**📊 数据集**

实验数据集为SQuAD（单段知识融合设置），采用50条训练样本与200条验证样本进行评测。

**📈 对比分析**

与SEAL的“Implications”和“Rewrite”基线相比，带档案的变体在验证集上可在迭代1-2中短暂逼近“Rewrite”峰值，但整体未超过；无档案变体与“Implications”保持相当水平，训练集表现略高。

**⚠️ 局限性**

主要限制包括：1）仅进行一次自我改进的短期实验，未实现持续自我学习；2）档案实现过于简单，缺乏新颖性压力，易导致模式崩溃；3）ReST^EM更新过于集中，可能加速同质化；4）模型规模和超参数设置可能限制探索能力。

---

## 131. JAXMg: A multi-GPU linear solver in JAX

**arXiv ID:** 2601.14466 | [PDF](https://arxiv.org/pdf/2601.14466v1)

**作者:** Roeland Wiersema `[一作]` `[通讯]` (Flatiron Institute), Roeland Wiersema (Flatiron Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为JAXMg的多GPU密集线性代数解决方案，旨在解决超出单个GPU内存限制的线性系统和特征值问题。

**💡 创新点**

JAXMg通过将JAX与NVIDIA的cuSOLVERMg连接，提供了可扩展的线性代数功能，允许在JAX程序中直接嵌入多GPU执行，保持了JAX的可组合性。

**🔧 技术方法**

使用了JAX框架和NVIDIA的cuSOLVERMg库，通过XLA外部函数接口实现了多GPU的线性代数运算。

**📊 数据集**

在基准测试中使用了具有8个NVIDIA H200 GPU的单节点，测试了不同大小的对角矩阵和随机正定矩阵。

**📈 对比分析**

与现有的单GPU JAX例程进行比较，JAXMg在处理大矩阵时表现出更好的扩展性和性能，尤其是在内存容量受限的情况下，能够解决更大的矩阵问题。

**⚠️ 局限性**

目前的实现可能在多GPU环境下的内存管理和数据分配上存在一定的复杂性，尤其是在多进程多设备模式下。

---

## 132. Report for NSF Workshop on AI for Electronic Design Automation

**arXiv ID:** 2601.14541 | [PDF](https://arxiv.org/pdf/2601.14541v1)

**作者:** Deming Chen `[一作]` (University of Illinois Urbana-Champaign), Yizhou Sun `[通讯]` (UCLA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2024年在 NeurIPS 2024 并行举办的 NSF AI‑EDA 研讨会，梳理了四大主题（物理合成与制造设计、HLS/LLS、优化工具箱、测试与验证）及讨论成果，并提出了对 NSF 的多项建议。

**💡 创新点**

创新点在于提出 AI 与 EDA 深度融合的多主题框架，强调跨学科合作、开放数据与工具、神经符号闭环、以及对 3D IC、端到端自动化的前瞻性思考，拓展了 AI 在芯片设计生命周期中的应用边界。

**🔧 技术方法**

主要技术包括大语言模型（LLM）、图神经网络（GNN）、强化学习（RL）、生成式 AI、神经符号方法、迁移学习与多模态模型等；同时提及了 LLM‑驱动的 RTL 生成、LLM‑辅助验证与符号反馈循环。

**📊 数据集**

讨论中引用的核心数据集有 HLSyn、Chrysalis、AutoDSE、OpenLS‑DGF、HARP、GNN‑DSE、GNN‑DSE 等，并指出需要构建 ImageNet‑级别的公开数据集以支撑 AI 训练与评估。

**📈 对比分析**

通过案例对比，AI 技术在性能预测（如 GNN‑DSE 预测毫秒级）、设计空间探索（AISYN 下降 69.3% 面积）、自动化流水线加速（HLS‑DL 通过 LLM+GNN 预测 PPA）、以及验证效率提升（G‑QED 与 LLM 结合减少验证时间）等方面表现出数十% 至百余%的提升。

**⚠️ 局限性**

主要局限包括：AI 模型泛化与迁移能力不足、数据质量与可获取性受限、对大规模 GPU/TPU 计算资源的高依赖、与现有 EDA 工具链的集成困难，以及 AI 生成设计的可靠性、安全性与可解释性不足。

---

## 133. GutenOCR: A Grounded Vision-Language Front-End for Documents

**arXiv ID:** 2601.14490 | [PDF](https://arxiv.org/pdf/2601.14490v1)

**作者:** Hunter Heidenreich `[一作]` (Roots.ai), Yosheb Getachew `[通讯]` (Roots.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Qwen2.5-VL的Grounded OCR前端GutenOCR，实现统一的读取、检测与定位接口。

**💡 创新点**

通过无修改微调、prompt驱动的多任务架构及专门的评估协议，显著提升多任务OCR性能。

**🔧 技术方法**

使用Qwen2.5-VL 3B/7B模型，配合prompt、文本和框坐标多模态输入/输出，训练采用AdamW、H100 GPU。

**📊 数据集**

数据来源于OCR-IDL、TabMe++、PubMed-OCR 以及合成的Grounded 和 SynthDoG Grounding，测试集约10.5K页。

**📈 对比分析**

与原始backbone、OCR专用模型及 Fox、OmniDocBench benchmark 对比，GutenOCR‑7B 在全局OCR、检测、定位等任务上复合得分提升至 0.82 以上；区域/行级 OCR 大幅改善，但页级线性化、色彩引导等任务略降。

**⚠️ 局限性**

局限包括对颜色引导和公式识别的弱化、对多样化排版鲁棒性不足，以及未覆盖表格、跨页结构等高级语义层面。

---

## 134. Exploiting Spot Instances for Time-Critical Cloud Workloads Using Optimal Randomized Strategies

**arXiv ID:** 2601.14612 | [PDF](https://arxiv.org/pdf/2601.14612v1)

**作者:** Neelkamal Bhuyan `[一作]` (Georgia Institute of Technology), TV Lakshman `[通讯]` (Nokia Bell Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种针对混合云环境下单个延迟敏感作业的在线调度方法，能够在使用价格低廉但不可靠的 Spot 实例与昂贵但稳定的 On‑Demand 实例之间动态切换，并在满足硬性截止时间的前提下最小化成本。

**💡 创新点**

创新点包括：①证明任何确定性调度策略的最坏情况竞争比至少为 Ω(K)（K 为 On‑Demand 与 Spot 成本比）；②设计随机化调度算法 ROSS，竞争比为 √(K)，在大多数截止时间宽松的情形下是最优的；③提出“warm‑up + 随机 on‑demand 注入”三阶段调度框架，平衡探索与利用，显著提升成本效率。

**🔧 技术方法**

技术手段：竞争分析与随机化在线算法设计、证明最优下界、基于流模型的证明；实验模拟中实现 ROSS、Uniform Progress、Greedy 与最优离线算法的对比；利用公开的 Spot 资源可用性轨迹进行评估。

**📊 数据集**

使用的数据集包括 Microsoft Azure 与 AWS 的 SpotLake Archive（高/低可用性标签）以及 AWS EC2 的 SkyPilot 追踪（availability 与 preemption 两类），涵盖多种地区与实例类型。

**📈 对比分析**

与 Uniform Progress、Greedy 以及最优离线调度对比。实验显示：在 L/D ≲ 0.65 的宽松截止时间下，ROSS 以高达 30% 的成本节省领先 Uniform；在 L/D ≳ 0.7 的紧凑截止时间下，ROSS 与 Uniform 接近或略优；与 Greedy 的对比则表现为始终更优或相当。总体而言，ROSS 的成本占比与最优相比接近理论上限，且其竞争比达到 √(K)。

**⚠️ 局限性**

局限性：①仅研究单个作业的调度，未考虑多作业调度与资源竞争；②假设 Spot 可用性可被离散化为可用/不可用，忽略了预占策略与实时价格波动；③实验仅基于公开轨迹，缺乏真实业务负载的验证；④随机化策略需要在实际系统中实现时考虑实现成本与调度延迟。

---

## 135. Automatically Tightening Access Control Policies with Restricter

**arXiv ID:** 2601.14582 | [PDF](https://arxiv.org/pdf/2601.14582v1)

**作者:** Ka Lok Wu `[一作]` (Stony Brook University), Omar Chowdhury `[通讯]` (Stony Brook University)

**通讯引用:** 1050 | [OpenAlex ID](https://openalex.org/A5070136662)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出一种自动化方法，利用访问日志中已执行的请求来识别并收紧Cedar ABAC访问控制策略中过度许可的规则，从而使策略更贴近最小权限原则。

**💡 创新点**

创新点在于：①采用规则级本地化、增量加强的策略，避免一次性全局收紧导致的规模灾难；②将收紧过程转化为SyGuS（语法引导合成）问题，自动合成类型安全的谓词以作为合取子句；③通过预先生成类型安全谓词列表，减少SyGuS的搜索空间并保持类型安全。

**🔧 技术方法**

技术手段包括：SMT / SyGuS（Satisfiability Modulo Theory 与 Syntax-guided Synthesis）、Cedar权限语言的符号编译器、并行规则级处理、类型安全谓词生成、日志切片与过度许可集合的近似。

**📊 数据集**

实验数据集来自两个基于真实系统的案例研究：①受Google Classroom启发的课堂管理系统；②受HotCRP启发的会议管理系统。研究中使用自定义工具生成随机实体存储和访问日志，形成多种“族”规模可调的测试实例。

**📈 对比分析**

评估方法：将自动收紧后的策略与人工设定的理想紧凑策略进行语义相似度比较，并统计保留/删除的过度许可与期望许可。性能评估：在实体数和日志密度两个维度上测量运行时间，结果显示收紧过程在实体数增长时呈子指数级，日志大小对运行时间影响不大，能够高效处理数千实体和数万条日志。

**⚠️ 局限性**

主要限制：①仅支持通过添加合取子句来收紧规则，无法生成带有disjunction或常量集合的谓词；②未处理规则重叠导致的最优收紧；③对日志中误操作或攻击者已知过度许可的情况假设所有已执行的允许请求均应保留；④类型安全检查依赖预生成谓词列表，扩展到更复杂Cedar特性仍需工作；⑤最终结果仍需管理员人工审查。

---

## 136. LLM Security and Safety: Insights from Homotopy-Inspired Prompt Obfuscation

**arXiv ID:** 2601.14528 | [PDF](https://arxiv.org/pdf/2601.14528v1)

**作者:** Luis Lazo `[一作]` (University of New Brunswick), Roozbeh Razavi-Far `[通讯]` (University of New Brunswick)

**通讯引用:** 3397 | [OpenAlex ID](https://openalex.org/A5019749887)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于同伦理论的提示变形框架，利用LLM生成恶意代码并构建包含7,374条恶意样本的数据集。

**💡 创新点**

将同伦（连续变形）思想引入提示工程，实现语义保持的恶意提示隐写；系统评估LLM对该攻击的易受性并公开大规模恶意代码数据。

**🔧 技术方法**

同伦启发的逐步提示变形、LLM提示工程、KIMI/LLama/DeepSeek 代码生成、Claude 代码验证、语义相似度与分布距离约束。

**📊 数据集**

15,732 个恶意相关术语生成的提示集合，生成 9,725 条代码样本，其中 7,374 条经 Claude 验证为恶意。

**📈 对比分析**

使用精确率与错误率作为评价指标，对比 LLaMA、DeepSeek、KIMI 三大模型：DeepSeek 精确率 0.822，错误率 17.8%；KIMI 成功率 0.774；整体精确率约 0.758。

**⚠️ 局限性**

仅评估提示变形攻击；未进行实际沙箱执行验证；不同模型使用提示数量不均；仅覆盖文本型恶意代码，忽略多模态/系统级攻击。

---

## 137. AQUA: an Agile Process to Develop Quantum Annealing Applications

**arXiv ID:** 2601.14501 | [PDF](https://arxiv.org/pdf/2601.14501v1)

**作者:** Lodovica Marchesi `[一作]` (University of Cagliari), Michele Marchesi `[通讯]` (Netservice S.p.A.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了 AQUA（Agile QUantum Annealing）——一种面向 QUBO/量子退火软件开发的敏捷生命周期流程，并在信用评分特征选择案例中实现了从需求分析到部署的完整开发示例。

**💡 创新点**

首次系统化地把敏捷方法与量子退火的独特需求（数学建模、算法选择、量子/经典混合求解、嵌入等）结合，形成四阶段、里程碑驱动的可重复的工程流程。

**🔧 技术方法**

敏捷 Scrum、设计科学研究（DSR）方法、Python 开发、D‑Wave 量子退火器、Hybrid QA、QAOA 以及并行 HPC 计算；同时采用数据预处理、QUBO 构造、量子与经典性能基准等技术。

**📊 数据集**

使用了三大公开金融数据集：German Credit Data（1,000 样本，20 维）、Credit Marketing Portugal（41,188 样本，18 维）和 Give Me Some Credit（150,000 样本，10 维）进行原型验证、系统开发与最终评估。

**📈 对比分析**

通过比较三种求解方案（原生 QA、Hybrid QA、QAOA）的准备时间、优化时间、总运行时、最优值、特征数以及开发与维护难易度，发现 Hybrid QA 在精度和效率上与 QA 相当但易维护；原生 QA 运行最快；QAOA 在小规模问题上可行但耗时长、易维护性差。

**⚠️ 局限性**

缺乏可量化的里程碑评估指标、对硬件排队延迟与成本波动的控制不足、对复杂度较高项目的验证有限、以及对团队培训与流程复杂度的挑战。

---

## 138. Scalable Knee-Point Guided Activity Group Selection in Multi-Tree Genetic Programming for Dynamic Multi-Mode Project Scheduling

**arXiv ID:** 2601.14485 | [PDF](https://arxiv.org/pdf/2601.14485v1)

**作者:** Yuan Tian `[一作]` (Victoria University of Wellington), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**通讯引用:** 30584 | [OpenAlex ID](https://openalex.org/A5100400258)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

开发了一种基于膝点筛选的多树遗传程序框架，用于大规模动态多模式资源约束项目调度问题的群体选择决策。

**💡 创新点**

创新点在于引入膝点选择机制显著减少可枚举组合，结合单一活动模式排序与群体优先规则的双树进化，提升规模可扩展性并获得更优调度。

**🔧 技术方法**

采用遗传程序（GP）多树表示、膝点选择、活动排序规则、群体优先规则、资源约束筛选等技术。

**📊 数据集**

使用随机生成的动态多模式RCPSP实例，200个活动、3种模式/活动、8或12种资源、不同顺序强度（OS）共六种场景。

**📈 对比分析**

通过30次独立实验与基准S级联GP比较，并用Wilcoxon检验评估显著性；在大多数场景下KGGP（两变体）显著优于S，低OS和资源种类多时提升更为明显。

**⚠️ 局限性**

仍存在计算开销大，尤其低OS时训练时间显著增加；膝点筛选可能丢失高质量组合，且在小规模实例上枚举完整方案更具竞争力。

---

## 139. Optimality of Staircase Mechanisms for Vector Queries under Differential Privacy

**arXiv ID:** 2601.14597 | [PDF](https://arxiv.org/pdf/2601.14597v1)

**作者:** James Melbourne `[一作]` (Centro de Investigación en Matemáticas), Shahab Asoodeh `[通讯]` (McMaster University)

**通讯引用:** 411 | [OpenAlex ID](https://openalex.org/A5062359324)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文证明在任意维度、任意范数以及任意范数单调损失函数下，独立加性噪声机制中阶梯机制是最优的。

**💡 创新点**

创新点在于利用凸重排理论将原始无限维优化问题转化为一维球对称问题，并证明极点正是阶梯分布，从而完整解决了此前仅在低维或特殊成本下的猜想。

**🔧 技术方法**

核心技术包括凸重排、几何级数性质、随机支配（stochastic domination）以及 Krein‑Milman 定理。

**📊 数据集**

本文没有使用具体数据集，而是在理论层面给出最优性证明，并通过 Monte‑Carlo 采样对阶梯分布进行数值评估。

**📈 对比分析**

对比方法是对阶梯机制与传统 Laplace 机制在相同隐私预算与感知度下，采用相同成本函数 Φ(x)=x₁ 的期望损失进行比较；数值结果显示阶梯机制在所有 ε 与维度范围内均优于 Laplace。

**⚠️ 局限性**

局限性包括：仅考虑纯 DP 的加性机制；仅适用于范数单调的损失函数；未讨论近似 DP 或更一般的噪声分布；对实际高维采样复杂度仍有待进一步优化。

---

## 140. SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control

**arXiv ID:** 2601.14258 | [PDF](https://arxiv.org/pdf/2601.14258v1)

**作者:** Ho Yin Au `[一作]` (Hong Kong Baptist University), Jie Chen `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 20143 | [OpenAlex ID](https://openalex.org/A5100333005)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Salient Orientation Symbolic（SOS）脚本及其自动提取与运动生成框架SOSControl，用以实现对人体部位方向和运动时序的可编程控制。

**💡 创新点**

创新点在于：①将运动关键帧的显著性信息通过时间受限层次凝聚聚类提取；②采用显著性掩码（SMS）生成稀疏可解释的符号脚本；③将SOS作为ControlNet的条件信号，并在运动扩散与解码阶段引入梯度优化与ACTOR‑PAE解码器，显著提升控制精度与自然度。

**🔧 技术方法**

使用技术包括：时间受限层次凝聚聚类、显著性掩码（SMS）、符号方向量化、ControlNet适配、梯度迭代优化、ACTOR‑PAE周期性潜在空间与扩散模型、SMS数据增强。

**📊 数据集**

主要数据集为HumanML3D（长度40–196帧），实验中亦使用BABEL验证通用性。

**📈 对比分析**

与MDM、GMD、OmniControl、TLControl等基线比较，SOSControl在SOS准确率（0.988）和L2‑Rot6D、FID、MMD指标上均优于其他方法，表明其在控制一致性与运动质量方面均取得了显著提升。

**⚠️ 局限性**

局限性在于：对显著性阈值的设置仍需人工调参；在极其复杂或非典型运动上，SOS脚本的稀疏性可能不足；模型训练依赖大量标注运动数据，迁移到实时机器人或多模态系统时仍有进一步验证的空间。

---

## 141. LaVR: Scene Latent Conditioned Generative Video Trajectory Re-Rendering using Large 4D Reconstruction Models

**arXiv ID:** 2601.14674 | [PDF](https://arxiv.org/pdf/2601.14674v1)

**作者:** Mingyang Xie `[一作]` (Meta Reality Labs), Lei Luo `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于视频扩散模型的新的相机轨迹合成方法，该方法通过条件化于大规模4D重建模型（如CUT3R）的潜在表示，从单目视频生成在任意轨迹下的几何一致新视图。

**💡 创新点**

创新点在于：①使用4D重建模型的隐式几何潜在作为柔性几何约束，避免了传统基于深度或点云的显式几何条件的误差累积；②设计轻量化的CUT3R适配器，压缩并对齐潜在与扩散模型的输入；③在保留预训练扩散先验的同时实现几何与视觉质量双赢。

**🔧 技术方法**

核心技术包括：视频扩散模型DiT与VAE潜在编码、CUT3R 4D潜在的时间/标记压缩适配器、相机姿态MLP适配器、流匹配训练目标、以及对目标轨迹的控制编码。

**📊 数据集**

训练使用Synthetic MultiCamVideo数据集；评估基于Pexels动态视频（100条）和DL3DV静态视频（50条），并采用VBench、PSNR、LPIPS、CLIP、主观一致性等指标进行测评。

**📈 对比分析**

与Gen3C、TrajectoryCrafter（基于点云条件）以及ReCamMaster（无几何条件）对比，本文方法在多视角、主题、背景一致性、循环一致性以及姿态重建误差上均优于所有基线，同时保持最高的视觉质量（PSNR最高、LPIPS最低）。

**⚠️ 局限性**

局限性主要体现在对动态透明物体（如玻璃杯）几何估计不足，导致合成效果欠佳；此外，适配器设计仍需较大GPU资源，推理速度与模型复杂度有一定权衡。

---

## 142. WebAssembly Based Portable and Secure Sensor Interface for Internet of Things

**arXiv ID:** 2601.14555 | [PDF](https://arxiv.org/pdf/2601.14555v1)

**作者:** Botong Ou `[一作]` (Purdue University), Baijian Yang `[通讯]` (Purdue University)

**通讯引用:** 1574 | [OpenAlex ID](https://openalex.org/A5089355143)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了基于WebAssembly的WASI-SN扩展，为嵌入式设备上的Wasm模块提供统一的传感器接口与MQTT‑SN网络访问控制，并实现了基于Wildcarded IBE的端到端加密与细粒度访问控制。

**💡 创新点**

创新点在于：1）首次将WASI扩展至传感器访问，形成跨平台的安全沙箱；2）将MQTT‑SN与WASM结合，实现内置网络库；3）使用Wildcarded IBE实现无托管的访问控制与即时密钥撤销，解决传统MQTT‑SN缺乏端到端安全的问题。

**🔧 技术方法**

技术手段包括：WebAssembly（Wasm）与WASI-SN接口、WAMR（WebAssembly Micro Runtime）在Zephyr RTOS上的实现、MQTT‑SN协议扩展、Wildcarded IBE（WKD‑IBE）加密与密钥管理、基于OpenThread的Thread网络、TLS‑style密钥协商与即时撤销方案。

**📊 数据集**

数据集/实验平台为实际硬件测试：NRF52840‑DK（Cortex‑M4F 64 MHz、256 KB SRAM）搭配BME280温湿压传感器，OpenThread Thread网络、MQTT‑SN网关为Rpi 3B+，MQTT‑Broker为PC。

**📈 对比分析**

与原生C程序对比：传感器访问的内存占用+5%（编译期）、运行时+0.5%；执行延迟约6%（多次访问后稳定）；MQTT‑SN接口延迟与原生差异<1%，仅受网络时延支配。整体性能接近原生，且提供了安全与可移植性。

**⚠️ 局限性**

局限性包括：1）WASM运行时在启动时产生固定开销；2）MQTT‑SN安全方案仍依赖可信PKG中心；3）仅评估单一传感器与单一硬件平台；4）实现细节与密钥撤销效率需在更大规模网络中验证。

---

## 143. Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging

**arXiv ID:** 2601.14556 | [PDF](https://arxiv.org/pdf/2601.14556v1)

**作者:** Andrew Crossman `[一作]` (JPMorgan Chase), Mohammad Yekrangian `[通讯]` (JPMorgan Chase)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5118845714)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并公开了一个多标签层次化的 MITRE ATT&CK 文本标注系统，并通过分阶段实验验证其性能。

**💡 创新点**

提出任务空间分层式表述并采用自底向上训练流程，避免传统自顶向下模型的偏置，同时实现了不依赖 LLM、RAG 的高精度标注。

**🔧 技术方法**

使用 TF‑IDF 向量化结合随机梯度下降（SGD）分类器实现多标签层次化预测，并加入哈希加密保证数据安全。

**📊 数据集**

以 JPMorgan Chase 内部的 14,405 条网络情报句子为基准数据，并在 552 条金融业务威胁场景上进行迁移实验。

**📈 对比分析**

与 GPT‑4o 的多类标注进行对比，SGD 模型在战术层面达到约 94% 准确率、技术层面约 82%，显著优于 GPT‑4o 的约 60%；在迁移数据上仅需少量再训练即可提升至约 66%。

**⚠️ 局限性**

数据量有限、类别极度不平衡且仅覆盖企业矩阵，模型在新领域表现需进一步验证，且对多标签技术层面未覆盖所有层级关系。

---

## 144. Tokenomics: Quantifying Where Tokens Are Used in Agentic Software Engineering

**arXiv ID:** 2601.14470 | [PDF](https://arxiv.org/pdf/2601.14470v1)

**作者:** Mohamad Salim `[一作]` (Concordia University), Emad Shihab `[通讯]` (Concordia University)

**通讯引用:** 7074 | [OpenAlex ID](https://openalex.org/A5049727493)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM‑MA系统在软件开发生命周期中的token消耗进行实证分析，构建tokenomics成本映射框架。

**💡 创新点**

首次将ChatDev内部阶段映射到通用软件工程阶段，并揭示代码评审阶段占据大部分token与通信税显著的问题。

**🔧 技术方法**

利用GPT‑5 Reasoning模型、Python脚本解析执行轨迹、token聚合与统计分析。

**📊 数据集**

30个来自ProgramDev数据集的开发任务（从简单算法到复杂应用）。

**📈 对比分析**

通过对比不同阶段的token比例和占比，发现代码评审平均占59.4%token，输入token占比53.9%，表明通信开销最大；未与其他框架直接对标，结果主要展示内部成本分布。

**⚠️ 局限性**

仅评估单一LLM‑MA框架和单一模型，任务规模有限，某些阶段样本不足，映射方式可能不唯一，限制了普适性和外推性。

---

## 145. Hint-Based SMT Proof Reconstruction

**arXiv ID:** 2601.14495 | [PDF](https://arxiv.org/pdf/2601.14495v1)

**作者:** Joshua Clune `[一作]` (Carnegie Mellon University), Jeremy Avigad `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3466 | [OpenAlex ID](https://openalex.org/A5003483051)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Lean证明助手中实现一种基于cvc5 SMT求解器产生的“提示”来重建证明的技术，生成不依赖外部求解器的自包含证明脚本。

**💡 创新点**

创新点在于不完全重放SMT证明，而是利用SMT在预处理和理论推理阶段生成的提示来引导Lean内部自动化，从而实现更稳定、可视化的证明重构，并消除对外部求解器的持续调用。

**🔧 技术方法**

技术实现包括：Lean、cvc5（改造以输出提示）、自定义tactic QuerySMT、duper、Vampire、Sledgehammer、grind等工具；关键模块为预处理、翻译、提示生成、提示解释与证明重构。

**📊 数据集**

使用从Lean的Init、Batteries、Mathlib库中提取的9,904个整数、自然数和列表相关定理作为基准数据集。

**📈 对比分析**

与+5、duper、Vampire、Sledgehammer、grind等方法对比，提示版QuerySMT在整数和自然数类问题上达到或超过最佳解法，在列表类问题上表现略逊，但整体性能提升显著；平均生成的提示数被压缩至1–2条，提升了可读性。

**⚠️ 局限性**

局限性包括：无法处理归纳证明、对自然数支持不足、对cvc5提示质量的依赖、对AC归一化处理不完整，以及在列表类问题上受限于原始环境差异导致的性能波动。

---

## 146. LURE: Latent Space Unblocking for Multi-Concept Reawakening in Diffusion Models

**arXiv ID:** 2601.14330 | [PDF](https://arxiv.org/pdf/2601.14330v1)

**作者:** Mengyu Sun `[一作]` (Sichuan University), Yi Zhang `[通讯]` (Sichuan University)

**通讯引用:** 94250 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于隐空间重建的多概念重新唤醒框架 LURE，能够在概念消除后恢复多个被抹除的概念。

**💡 创新点**

将生成过程视作隐式函数，从文本、模型参数和潜在状态三维联合分析概念抹除机制，并通过语义重绑定、梯度场正交化和潜在语义识别引导采样实现多概念无干扰恢复。

**🔧 技术方法**

使用隐空间重绑定损失、梯度正交化约束、潜在语义识别引导采样（LSIS）等技术，基于 Stable Diffusion v1.4 进行实现。

**📊 数据集**

使用 ImageNette、公开知识产权角色、名人身份集合以及安全内容（血、裸、武器）等多组数据；每个概念仅用三张示例图进行微调。

**📈 对比分析**

与 UCE、RECE、SPEED 等主流概念抹除方法对比，评估 ACC、FID、CLIP Score，结果显示 LURE 在多概念恢复中均显著提升（ACC 提升约 10–20%、CLIP 提升约 2–5 点、FID 降低约 30–50）。

**⚠️ 局限性**

局限性包括需额外微调步骤、对极少样本概念的恢复效果受限，以及在极高安全级别抹除下仍可能被攻击。

---

## 147. CityCube: Benchmarking Cross-view Spatial Reasoning on Vision-Language Models in Urban Environments

**arXiv ID:** 2601.14339 | [PDF](https://arxiv.org/pdf/2601.14339v1)

**作者:** Haotian Xu `[一作]` (National University of Defense Technology), Yong Li `[通讯]` (Tsinghua University)

**通讯引用:** 36351 | [OpenAlex ID](https://openalex.org/A5100355277)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CityCube基准，用于系统评估城市环境下VLM的跨视角空间推理能力。

**💡 创新点**

创新点在于构建覆盖多视角、多平台、多任务的5,022道多选QA，涵盖5个认知维度和三种空间关系表达，并通过四个视角动态维度模拟真实跨视角观察。

**🔧 技术方法**

采用Gemini-2.5 Pro生成问题答案的结构化提示与两阶段过滤、人工校验，随后使用LoRA对33个VLM进行微调与评估。

**📊 数据集**

使用nuScenes、GeoText-1652、EmbodiedCity、MatrixCity等公开城市数据集，汇总18.1K多源视角图像构成CityCube。

**📈 对比分析**

在多选准确率上，最强专有模型仅达54.1%准确率，远低于人类88.3%；对CityCube微调的CityBot 4B/8B模型在各认知维度上提升至约60‑70%，仍与人类存在约34%差距。

**⚠️ 局限性**

局限包括未剖析视角偏差与参考框架不确定性、缺乏Sim2Real评估、未探究模型内部表征与多视角融合机制，且未验证专门的空间推理后训练策略。

---

## 148. Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct

**arXiv ID:** 2601.14277 | [PDF](https://arxiv.org/pdf/2601.14277v1)

**作者:** Uygar Kurt `[一作]` `[通讯]`, Uygar Kurt

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Llama‑3.1‑8B‑Instruct 进行多种 GGUF 量化格式的统一实验，评估其对模型压缩、推理速度与多任务性能的影响。

**💡 创新点**

首次在同一模型、相同评测管线下，对 13 种社区主流量化格式（从 3 位到 8 位）进行系统比较，揭示量化精度与格式设计对任务表现的细粒度影响。

**🔧 技术方法**

使用 llama.cpp 的官方量化工具、GGUF/GGML 格式、lm-evaluation-harness 对下游基准（GSM8K、HellaSwag、IFEval、MMLU、TruthfulQA）及 WikiText‑2 perplexity 进行评估；同时在 Intel Xeon Platinum 8488C 服务器上测量量化时间和 CPU 推理吞吐量。

**📊 数据集**

数据集包括 GSM8K、HellaSwag、IFEval、MMLU、TruthfulQA、WikiText‑2，均采用官方公开版本；模型基准为 FP16 Llama‑3.1‑8B‑Instruct。

**📈 对比分析**

通过对比精度、压缩率、平均任务分数、Perplexity 以及推理吞吐，绘制 Pareto 前沿，结果显示 5‑bit K‑quant（如 Q5_K_S、Q5_0）在压缩率约 65% 时可保持甚至略优于 FP16 的整体性能，3‑bit 量化则显著退化；低位量化提升生成吞吐。

**⚠️ 局限性**

受限于单一硬件平台（CPU）和单一模型，未考虑 GPU/FPGA 场景；量化方案的可推广性和长上下文推理表现仍待进一步验证。

---

## 149. RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension

**arXiv ID:** 2601.14289 | [PDF](https://arxiv.org/pdf/2601.14289v1)

**作者:** Yelin Chen `[一作]` (Xinjiang University), Juanzi Li `[通讯]` (Tsinghua University)

**通讯引用:** 14263 | [OpenAlex ID](https://openalex.org/A5003324011)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个大规模的研究论文问答基准RPC‑Bench，包含4150篇计算机科学论文及61.3k条人类核实的QA对。

**💡 创新点**

创新点在于采用真实的审稿–答复交流生成QA、设计面向研究流程的细粒度分类体系、引入LLM‑人类协作的标注与LLM‑Judge评估框架，并在多模态输入下评估模型的正确性、完整性与简洁性。

**🔧 技术方法**

使用了多种大语言模型与文档中心模型、检索增强生成模型及视觉语言模型，并通过LLM‑Judge和多指标评估实现细粒度性能分析。

**📊 数据集**

数据集为从OpenReview收集的论文及其评审回复，并结合AMiner进行质量筛选，最终形成4150篇论文、61.3k QA对的RPC‑Bench。

**📈 对比分析**

与28种先进模型对比显示，GPT‑5在文本输入下的F1‑like最高达68.2%，但在多模态输入或加入简洁度约束后性能显著下降；文档中心模型与RAG模型普遍表现低于LLM。

**⚠️ 局限性**

局限性包括数据主要覆盖计算机科学子领域、训练集QA大部分由LLM重写、缺乏跨文档推理与多论文综合评估，并且多模态模型在视觉信息利用上仍显不足。

---

## 150. VisTIRA: Closing the Image-Text Modality Gap in Visual Math Reasoning via Structured Tool Integration

**arXiv ID:** 2601.14440 | [PDF](https://arxiv.org/pdf/2601.14440v1)

**作者:** Saeed Khaki `[一作]` (Microsoft), Kamal Ginotra `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VisTIRA 框架，让 VLM 能在视觉数学问题上迭代生成自然语言推理与可执行 Python 代码，并通过工具执行反馈来逐步推理；构建基于 SnapAsk 的工具整合轨迹数据集；开发 LaTeX 渲染管线将文本数学问题转换为图像，形成可量化的视觉‑文本模态差距评估基准；发布 360k 个渲染 NuminaMath 图像与 5k 测试集。

**💡 创新点**

①将工具使用与推理过程显式化为可训练的轨迹；②将文本数学问题转化为视觉形式，形成对比实验；③将 OCR 结果作为额外文本 grounding，揭示不同规模模型对 OCR 的依赖；④通过实验量化模态差距随模型规模的变化。

**🔧 技术方法**

Vision‑Language 模型（如 Qwen2.5‑VL‑7B、GPT‑5）、工具集成（Python 代码执行、SymPy）、OCR（DeepSeek‑OCR）、LaTeX 渲染管线、监督微调（LoRA、DeepSpeed ZeRO‑3）。

**📊 数据集**

SnapAsk（约 30 万份真实作业图像）、NuminaMath（860k 文本数学问题）、生成的 360k 渲染图像与 5k 测试图像。

**📈 对比分析**

使用同一组问题在文本、图像、图像+OCR、OCR‑only 四种输入模式下评估；对比基线 VLM（Qwen2.5‑VL‑7B‑Instruct）与 VisTIRA 微调模型、GPT‑5；结果显示 VisTIRA 提升了约 2–5% 准确率，OCR 对小模型提升显著（约 5%），对大模型提升有限；模态差距在小模型更明显。

**⚠️ 局限性**

①渲染图像缺乏真实手写/拍照噪声，泛化受限；②轨迹生成依赖强教师模型，可能导致思路单一；③OCR 对大模型效果下降或引入噪声，需更自适应的 grounding 机制；④缺乏多样化的工具集成策略，可能限制推理多样性。

---

## 151. An LLM Agent-based Framework for Whaling Countermeasures

**arXiv ID:** 2601.14606 | [PDF](https://arxiv.org/pdf/2601.14606v1)

**作者:** Daisuke Miyamoto `[一作]` (National Graduate Institute for Policy Studies), Narushige Michishita `[通讯]` (National Graduate Institute for Policy Studies)

**通讯引用:** 89 | [OpenAlex ID](https://openalex.org/A5034408137)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对日本高校高层教职人员的鲸鱼式钓鱼（Whaling）防御框架，通过离线生成个性化漏洞概况（PVP）、风险情景、个性化防御概况（PDP），并在在线阶段使用LLM代理基于PDP进行邮件风险评估与解释。

**💡 创新点**

创新点在于将攻击侧的PVP与E-PhishGEN框架反转为防御侧，系统化地将公开信息与内部流程结合，形成可解释的PDP，并通过LLM实现上下文感知的动态风险判断。

**🔧 技术方法**

核心技术包括：基于大语言模型（LLM）的多阶段代理（PVP生成、风险情景生成、PDP生成及在线邮件评估），JSON结构化知识表示，以及对邮件头体信息的自动解析与对比。

**📊 数据集**

使用的数据集主要为公开的高校教职人员信息（院系主页、科研数据库、公开演讲与社交媒体）以及实验中模拟的钓鱼邮件与正常邮件样本；未采用商业化大规模钓鱼数据集。

**📈 对比分析**

实验通过对两名参与者（A、B）构建PDP并评估若干样本邮件，显示LLM代理能够给出符合PDP的风险标签和解释，效果与人工评估相近，但因样本量小缺乏量化指标，未与现有钓鱼检测模型直接比较。

**⚠️ 局限性**

限制包括：样本规模仅为两名参与者，缺乏大规模验证；风险情景与PDP的内部信息依赖人工访谈，缺乏标准化；在线评估仅在模拟邮件上测试，未在真实邮件流量上验证，且面临隐私与治理集成挑战。

---

## 152. HELIOS: Hierarchical Graph Abstraction for Structure-Aware LLM Decompilation

**arXiv ID:** 2601.14598 | [PDF](https://arxiv.org/pdf/2601.14598v1)

**作者:** Yonatan Gizachew Achamyeleh `[一作]` (University of California), Mohammad Abdullah Al Faruque `[通讯]` (University of California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HELIOS 框架，将二进制程序的控制流图（CFG）与函数调用图（FCG）等结构信息编码成分层文本，然后将该结构化提示与原始反编译输出一起输入大型语言模型（LLM），通过结构化推理与可选编译器反馈循环实现更可靠的二进制反编译。

**💡 创新点**

创新点在于：① 把图结构信息（CFG、FCG、循环/分支等）以文本方式显式提供给 LLM；② 设计一套简短规则指导模型如何遵循结构；③ 在生成代码后引入编译器反馈单轮迭代，纠正语法与类型错误；④ 通过上述方式在不需要任何细调的情况下，让通用 LLM 在多种硬件架构上实现高可编译率与更高功能正确性。

**🔧 技术方法**

技术手段包括：使用 Ghidra 进行静态分析提取伪 C 代码、CFG 与 FCG；构建多层结构化提示（函数概要、CFG 概览、块级细节、原始伪 C）；调用通用 LLM（Gemini‑2.0‑Flash、GPT‑4.1 Mini）；可选的编译器反馈循环（编译失败时将错误信息加入提示再次生成）。

**📊 数据集**

数据集：基于 HumanEval 与 MBPP 的 C 版本构成 Cross‑Arch‑DB，覆盖 x86‑64、x86‑32、ARM‑32、ARM‑64、MIPS‑32、MIPS‑64 六种指令集，并在 O0、O1、O2、O3 四个优化级别下编译得到二进制，随后再进行反编译与评估。

**📈 对比分析**

对比方法：与文本‑only 基线（仅输入原始反编译文本）以及两类细调 LLM（LLM4Decompile 1.3B/6.7B、Nova 1.3B/6.7B）进行对比；评估指标为：可编译率、可链接率、功能正确率（单元测试通过率）和编辑相似度。实验显示 HELIOS 在 Gemini‑2.0‑Flash 上可编译率从 45.0% 提升到 85.2%，在 GPT‑4.1 Mini 上从 71.4% 提升到 89.6%；加上编译器反馈后可编译率可达 94–96%，功能正确率提升 5–6 个百分点，且跨架构波动显著减小。

**⚠️ 局限性**

局限性：① 评估以简短、孤立的 benchmark（HumanEval、MBPP）为主，未覆盖大规模真实二进制的复杂性；② 可能存在训练数据泄漏导致某些架构上表现偏高；③ 结果依赖 Ghidra 生成的 CFG 与伪 C 的准确性；④ 只测试了两款通用 LLM，未探讨其它模型或更大规模模型的效果；⑤ 未与传统非 LLM 反编译器做直接对比；⑥ 编译器反馈循环会引入额外调用开销，需在实际使用中权衡。

---

## 153. Recursivism: An Artistic Paradigm for Self-Transforming Art in the Age of AI

**arXiv ID:** 2601.14401 | [PDF](https://arxiv.org/pdf/2601.14401v1)

**作者:** Florentin Koch `[一作]` `[通讯]` (Ecole Polytechnique), Florentin Koch (Ecole Polytechnique)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Recursivism 概念框架，用递归层级分析 AI 艺术实践并通过案例检验其可行性。

**💡 创新点**

将递归从数学迁移到美学，定义五级递归尺度并引入内存（μ）、可进化性（ρ）与可见度（R）三指标，构建系统性评估维度。

**🔧 技术方法**

结合生成模型、深度学习、进化算法与自引用程序（如 Darwin–Gödel Machine）实现自我改造的递归艺术流程。

**📊 数据集**

主要引用艺术家自有交互数据、实时传感输入与人类评估反馈，并无统一公开数据集可供复现。

**📈 对比分析**

通过 μ‑ρ‑R 三维空间对比 Refik Anadol、Sougwen Chung、Karl Sims 与 Darwin–Gödel Machine 四个案例，层级越高可见度与可进化性提升，但缺乏客观量化性能指标，评估主要以定性描述为主。

**⚠️ 局限性**

技术门槛高、能源消耗大、缺乏标准化评价方法、作者身份模糊且跨文化适用性有限。

---

## 154. Unpacking Security Scanners for GitHub Actions Workflows

**arXiv ID:** 2601.14455 | [PDF](https://arxiv.org/pdf/2601.14455v1)

**作者:** Madjda Fares `[一作]` (Université de Montréal), Benoit Baudry `[通讯]` (Université de Montréal)

**通讯引用:** 6019 | [OpenAlex ID](https://openalex.org/A5086536054)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文首次对 GitHub Actions 工作流安全扫描器进行系统比较，挑选并筛选了 9 个主流开源工具，构建了 10 类安全弱点分类法，并在 596 条真实工作流上评估了扫描器的覆盖率、检测一致性和执行性能。

**💡 创新点**

创新点在于提出统一的安全弱点分类框架、设计可跨工具比较的映射方法，并通过大规模实测给出不同扫描器在覆盖范围、精确度与速度上的对比与实用建议。

**🔧 技术方法**

采用了静态分析扫描技术、规则映射与聚合、自动化执行脚本、性能计时工具（Unix time 命令）以及手工审查规则实现的方式，对工具进行评测。

**📊 数据集**

使用了 596 条来自 77 个知名开源仓库（如 Microsoft、Meta、NVIDIA 等）的 GitHub Actions 工作流作为实验数据集，涵盖了多种触发器、作业与外部 Action 的组合。

**📈 对比分析**

比较方法包括：①将每个扫描器的 84 条规则映射到 10 类弱点以衡量覆盖率；②在同一工作流集上运行所有扫描器，统计检测到的弱点数量与重叠情况；③记录每条工作流的扫描耗时，评估 CI 集成的可行性。性能结果显示，绝大多数扫描器平均耗时 <1 s，只有少数（如 ggshield、poutine、zizmor）在大文件上达到数十秒，但总体仍可接受。

**⚠️ 局限性**

局限性包括：数据集规模有限、缺乏标注为真弱点的基准、评测脚本可能存在错误、仅考虑静态扫描器且仅覆盖 9 个工具，未包含云端或动态检测工具，结果可能不完全适用于所有类型的工作流。

---

## 155. Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding in the Complex Spectrum

**arXiv ID:** 2601.14472 | [PDF](https://arxiv.org/pdf/2601.14472v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 156. Breaking the accuracy-resource dilemma: a lightweight adaptive video inference enhancement

**arXiv ID:** 2601.14568 | [PDF](https://arxiv.org/pdf/2601.14568v1)

**作者:** Wei Ma `[一作]` (Shenzhen University), Lei Huang `[通讯]` (Shenzhen University)

**通讯引用:** 9286 | [OpenAlex ID](https://openalex.org/A5041982864)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于模糊控制的轻量动态视频推理框架，能根据设备 GPU 利用率、温度及当前帧目标数实时切换不同规模的模型，实现资源利用与推理性能的平衡。

**💡 创新点**

创新点在于：①设计了以 GPU 利用率、温度和目标数为输入的模糊控制器，实现实时资源评估与模型选择；②利用相邻帧目标的时空相关性和安全阈值机制，避免频繁无效切换；③该框架的计算复杂度为 O(1)，对设备无显著额外负担。

**🔧 技术方法**

技术手段包括：模糊逻辑控制（模糊化、规则推理、去模糊化）、YOLOv8 s/m/l 轻量模型、GPU 监控、动态模型切换策略、安全阈值切换机制以及低复杂度实现。

**📊 数据集**

使用了 VisDrone（350 帧视频）和 UA-DETRAC（2000 帧视频）数据集进行实验。

**📈 对比分析**

与单模型（s/m/l）以及传统静态推理进行对比；实验表明在 Jetson Orin NX 与 PC 上，动态方法在保持与大模型相近或更优的检测精度的同时，显著提升资源利用率（AVTG 指标），降低温度升高速率，切换次数低且稳健。

**⚠️ 局限性**

局限性包括：①模糊规则需人工设定，迁移到不同硬件/场景可能需要重新调参；②对目标数突变的极端情况仍需进一步验证；③目前仅验证于 YOLOv8，缺乏对其他模型或跨平台的普适性验证。

---

## 157. Counterfactual Modeling with Fine-Tuned LLMs for Health Intervention Design and Sensor Data Augmentation

**arXiv ID:** 2601.14590 | [PDF](https://arxiv.org/pdf/2601.14590v1)

**作者:** Shovito Barua Soumma `[一作]` (Arizona State University), Hassan Ghasemzadeh `[通讯]` (Arizona State University)

**通讯引用:** 4585 | [OpenAlex ID](https://openalex.org/A5007139473)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用大型语言模型（LLM）生成对抗性解释（CFs），既用于设计可行的健康干预方案，又用于在标签稀缺场景下对模型进行数据增强；

**💡 创新点**

创新点在于：①将LLM应用于多模态传感器数据的对抗性生成；②对比预训练与微调模型，并证明微调的LLaMA-3.1-8B在干预可行性与增强效果上优于传统优化方法；

**🔧 技术方法**

主要技术包括：LLM微调（LoRA + 4‑bit NF4），零/少样本提示、结构化提示生成CFs，评价指标（有效性、距离、稀疏度、可解释性），以及在神经网络上的F1恢复评估；

**📊 数据集**

使用公开的AI‑READI多模态糖尿病风险数据集，包含 CGM、活动、睡眠、环境和问卷等12维特征；

**📈 对比分析**

与优化基线（DiCE、CFNOW、NICE）以及GPT‑4o进行对比，微调LLaMA‑3.1‑8B在三种标签稀缺场景下平均恢复约20% F1，超越零样本LLM且在可解释性与临床可操作性上更佳；

**⚠️ 局限性**

局限包括：对抗样本仍可能超出临床可行范围；微调需要足够标注数据；仅针对结构化表格特征，未覆盖原始传感器流、文本或图像等多模态原始数据。

---

## 158. "Just in Time" World Modeling Supports Human Planning and Reasoning

**arXiv ID:** 2601.14514 | [PDF](https://arxiv.org/pdf/2601.14514v1)

**作者:** Tony Chen `[一作]` (Massachusetts Institute of Technology), Kevin Smith `[通讯]` (University of British Columbia)

**通讯引用:** 15166 | [OpenAlex ID](https://openalex.org/A5069299612)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种“即时构造”(Just‑In‑Time, JIT)的仿真‑基推理模型，模型在仿真过程中动态增删对象，形成场景的简化表征；

**💡 创新点**

核心创新在于将表征形成与仿真过程交织，而非事先预先构造最优表征，从而减少计算负担并更好地匹配人类的记忆与注意分布；

**🔧 技术方法**

技术包括基于 A*（带噪声的softmax采样）规划、物理仿真（Pymunk +噪声模型）、局部视觉前瞻（look‑ahead）与基于时间的记忆衰退；

**📊 数据集**

数据集为自建的两类实验环境：1）网格世界规划任务（40个随机迷宫）和2）Plinko 物理预测任务（约200个场景），并在 Prolific 上收集人类行为与记忆数据；

**📈 对比分析**

与之前的 Value‑Guided Construal (VGC) 以及随机/最大化基线进行对比，评估指标包括相关系数、RMSE、对数似然、计划长度、节点扩展数、对象数；结果显示 JIT 在人类记忆/注意预测上显著优于 VGC，在表征效率和总体效用上亦在大多数成本配置下表现最佳；

**⚠️ 局限性**

局限性包括：实验场景过于简化（仅单目标、静态、对象数少）；假设目标已知且能立即获取；未考虑更复杂的记忆容量约束与学习先验；需进一步验证在更大规模、动态、无目标指示的真实环境中的可行性。

---

## 159. Developmental trajectories of decision making and affective dynamics in large language models

**arXiv ID:** 2601.14268 | [PDF](https://arxiv.org/pdf/2601.14268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 160. 3D Space as a Scratchpad for Editable Text-to-Image Generation

**arXiv ID:** 2601.14602 | [PDF](https://arxiv.org/pdf/2601.14602v1)

**作者:** Oindrila Saha `[一作]` (University of Massachusetts Amherst), Kevin Blackburn-Matzen `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出并实现了基于3D空间的 Scratchpad 框架，先将文本提示解析为可编辑 3D 网格并通过 LLM 代理规划场景，最终渲染出与提示高度一致的图像。

**💡 创新点**

创新点在于将 3D 空间作为显式思维工作空间，允许在三维中对对象位置、方向、相机等进行规划，并支持后续可编辑且身份保留的图像生成。

**🔧 技术方法**

技术包括 GPT‑5/4o 驱动的多代理规划、Flux.1 与 Hunyuan‑3D 生成 3D 网格、Pytorch‑3D 渲染、SIGMA‑Gen 身份保留图像生成等。

**📊 数据集**

在需要高级推理的 GenAI‑Bench 与 CompoundPrompts 等合成图像数据集上进行评估。

**📈 对比分析**

与文本推理或二维规划的基线（Idea2Img、RPG 等）比较，3D Scratchpad 在文本对齐上提升约 0.2 分（≈32%），整体 VQA 分数提升，图像质量保持或略有提升。

**⚠️ 局限性**

局限在于 LLM 生成的对象布置过于均匀、姿态规划有限，对复杂交互和可变形资产的控制不佳。

---

## 161. Probing Prompt Design for Socially Compliant Robot Navigation with Vision Language Models

**arXiv ID:** 2601.14622 | [PDF](https://arxiv.org/pdf/2601.14622v1)

**作者:** Ling Xiao `[一作]` (Hokkaido University), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**通讯引用:** 6537 | [OpenAlex ID](https://openalex.org/A5048624196)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统研究了小型视觉语言模型（VLM）在社会合规机器人导航任务中的提示（prompt）设计，探讨了系统引导（action-focused、reasoning-oriented、perception–reasoning）与动机框架（竞争对人类、竞争对其他AI、竞争对自身过去版本）对导航性能的影响。

**💡 创新点**

创新点在于：① 把人类认知学习和动机理论引入提示工程，提出基于竞争的动机框架；② 设计感知与推理融合的提示，专门针对小VLM的决策约束；③ 在零样本与微调两种设置下系统比较，揭示提示在动作准确性上的显著提升。

**🔧 技术方法**

使用技术包括：TinyLLaVA框架（冻结视觉编码器、可训练多模态投影器+小型语言模型）、Prompt catalog（多维度系统提示模板）、对比实验（不同提示组合、不同模型、不同数据集）、评估指标（Action Accuracy、BERT‑F1、SBERT）以及深度学习训练工具（DeepSpeed、FlashAttention）。

**📊 数据集**

数据集：SNEI（325个第一人称图像、5步对话注释）与MUSON（800个第一人称样本，包含动态与静态约束），分别用于训练/测试。

**📈 对比分析**

比较方法：在相同离散动作空间下，将各提示类型与无系统提示做对比，使用动作准确率和语义相似度评估。实验结果显示：对GPT‑4o而言，竞争对人类的推理提示获得最高AA；对微调小VLM而言，感知–推理提示结合竞争对自身过去版本表现最佳；提示的改进在动作准确率上明显高于语义指标。

**⚠️ 局限性**

limitations：① 数据集规模有限，难以覆盖极端天气、夜间或交通信号等复杂情景；② 只使用文本提示，未探索自适应或学习式提示；③ 实验仅在离散动作空间内进行，缺乏对连续控制与更真实环境的验证。

---

## 162. Re-understanding Graph Unlearning through Memorization

**arXiv ID:** 2601.14694 | [PDF](https://arxiv.org/pdf/2601.14694v1)

**作者:** Pengfei Ding `[一作]` (Macquarie University), Guanfeng Liu `[通讯]` (Macquarie University)

**通讯引用:** 5379 | [OpenAlex ID](https://openalex.org/A5070515519)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于 GNN 记忆机制的图未学习框架 MGU，旨在精准评估未学习难度并自适应调整未学习目标，以实现高效且彻底的图信息删除。

**💡 创新点**

创新点在于首次将 GNN 的记忆程度与未学习难度关联，构建了无测试访问的难度评估指标、适应性边际遗忘与蒸馏保留机制，以及针对不同难度级别的全面评估协议。

**🔧 技术方法**

核心技术包括记忆分数计算、邻域影响加权、基于原始模型的边际调整损失、温度自适应蒸馏以及难度感知采样等。

**📊 数据集**

实验使用十个常见的图数据集（包括 homophily 图如 Cora、Citeseer、PubMed 以及 heterophily 图如 Chameleon、Squirrel）并在多种 GNN 结构（GCN、GAT、SAGE、GIN、FAGCN）上验证。

**📈 对比分析**

与九个现有未学习方法相比，MGU 在记忆化程度为易/难/随机的三种难度设置下均能提升 3.3%–53.1% 的 ToU 分数，同时在训练时间上比学习型方法快 3.5–70 倍，且对对抗攻击更具鲁棒性。

**⚠️ 局限性**

局限性包括对超参数（如温度上限 T_max、边际权重 λ）的敏感性、仅在节点/边/特征删除三种任务上验证，以及在极端高比例删除或动态图环境中的适用性尚待进一步研究。

---

## 163. GNN-based Path-aware multi-view Circuit Learning for Technology Mapping

**arXiv ID:** 2601.14286 | [PDF](https://arxiv.org/pdf/2601.14286v1)

**作者:** Wentao Jiang `[一作]` (NingBo University), Zhufei Chu `[通讯]` (NingBo University)

**通讯引用:** 417 | [OpenAlex ID](https://openalex.org/A5022156184)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 GNN‑基路径感知多视图电路学习框架 GPA，用于在技术映射过程中准确预测单元延迟，从而引导映射引擎做出更优的切割选择。

**💡 创新点**

创新点在于将 AIG 功能编码、后映射技术编码和路径感知 Transformer 池化三种互补视角融合到一个端到端模型中，能够捕捉结构、功能与路径相关的延迟特征，克服传统静态模型的上下文无关性。

**🔧 技术方法**

技术实现包括 Graph Attention Network (GAT)、Transformer 池化、双任务预训练（结构/功能）、MLP 分类器以及针对技术库的全局嵌入；模型采用多任务训练和交叉熵损失。

**📊 数据集**

训练和评估数据集来源于 ForgeEDA 与 DeepCircuitX 的 15,000 个子电路（AIG 预训练）以及 5,000 个关键路径子电路（延迟标签），使用 Sky130nm 与 ASAP7nm 标准单元库生成后映射标签。

**📈 对比分析**

在 19 个 EPFL 组合基准上，GPA 相比传统的 techmap、MCH 以及前沿 ML 方法 SLAP，平均分别降低 19.9%、2.1% 与 4.1% 的关键路径延迟，且面积变化不大或略有下降，验证了其优越性。

**⚠️ 局限性**

局限性包括仅使用 8 类延迟分箱，预测精度仍受限；模型对完整后映射上下文（如负载、信号相关性）的依赖尚未充分建模，未来需要引入更丰富的后映射信息以进一步提升准确性。

---

## 164. Variance-Adaptive Muon: Accelerating LLM Pretraining with NSR-Modulated and Variance-Scaled Momentum

**arXiv ID:** 2601.14603 | [PDF](https://arxiv.org/pdf/2601.14603v1)

**作者:** Jingru Li `[一作]` (Nankai University), Huan Li `[通讯]` (Nankai University)

**通讯引用:** 15460 | [OpenAlex ID](https://openalex.org/A5100319241)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Muon-NSR与Muon-VS两种加入方差自适应的Muon优化器变体；

**💡 创新点**

创新点在于将Adam的方差自适应机制（噪声对信号比NSR）与Muon's矩阵正交更新相结合，形成在正交化前的方差归一化或无参方差缩放；

**🔧 技术方法**

采用Newton–Schulz正交化、指数移动平均、噪声对信号比（NSR）调制、Adam式方差估计与正交矩阵运算；

**📊 数据集**

在GPT‑2（OpenWebText）和LLaMA（FineWeb、DCLM）两大预训练任务上进行评估；

**📈 对比分析**

与AdamW和标准Muon进行对比，实验显示Muon‑NSR/Muon‑VS在训练步骤与壁钟时间上更快收敛，验证损失更低，尤其在LLaMA‑1.2B上迭代次数下降约1.36倍；

**⚠️ 局限性**

局限性包括对大批量训练的依赖，方差统计不稳定时表现下降，且在某些中等规模模型上优势不明显；

---

## 165. Hierarchical Contextual Uplift Bandits for Catalog Personalization

**arXiv ID:** 2601.14333 | [PDF](https://arxiv.org/pdf/2601.14333v1)

**作者:** Anupam Agrawal `[一作]` (Dream11), Abhimanyu Mittal `[通讯]` (Dream11)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并部署了一种层级化情境提升Bandit（HCUB）框架，用于在Fantasy Sports平台上动态个性化竞赛目录。

**💡 创新点**

创新点包括：① 将情境信息构建为层级树，支持从系统级到用户级的多粒度探索；② 将提升（uplift）模型嵌入Bandit目标，直接优化增量收益；③ 采用奖励继承机制，让父节点的可靠奖励向子节点传递，缓解冷启动与数据稀疏问题；④ 使用层级Bootstrap估计不确定性，并通过贝叶斯UCB进行决策。

**🔧 技术方法**

使用技术包括：上下文Bandit（LinUCB/Thompson Sampling的扩展）、提升建模、层级Bootstrap、贝叶斯UCB、在线A/B测试、离线仿真、奖励继承与层级上下聚合。

**📊 数据集**

使用了Dream11的真实业务数据：约600万用户的日活数据、竞赛属性（入场费、参赛人数、奖池、奖项分配）以及完整的竞赛目录信息，进行在线实验与离线模拟。

**📈 对比分析**

对比方法：将HCUB与当前生产系统（BAU）进行线上A/B测试，评估收入、活跃度等指标；离线模拟中对比启用奖励继承与禁用继承的两种HCUB版本。结果显示：在线实验收入提升0.42%（6M用户）和0.51%（全平台）；离线仿真奖励继承版本相较于无继承版分别降低4%和5%的期望代价（regret）。

**⚠️ 局限性**

局限性包括：① 初始部署仅修改竞赛规模属性，未覆盖其他竞赛参数；② 动态环境下仍需改进对非平稳性的适应（如滑动窗口、折扣、变点检测）；③ 对极少量子组的数据估计仍可能不稳定；④ 需要进一步扩展到更大范围的竞赛目录与更丰富的用户特征。

---

## 166. Quality or Quantity? Error-Informed Selective Online Learning with Gaussian Processes in Multi-Agent Systems: Extended Version

**arXiv ID:** 2601.14275 | [PDF](https://arxiv.org/pdf/2601.14275v1)

**作者:** Zewen Yang `[一作]` (Technical University of Munich), Peng Shi `[通讯]` (University of Adelaide)

**通讯引用:** 88843 | [OpenAlex ID](https://openalex.org/A5100739668)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种分布式误差感知选择性在线学习框架（EIGP），实现多智能体系统中高质量 Gaussian 过程回归的协作预测，并给出贪心（gEIGP）与自适应（aEIGP）两种子算法以及数据删除策略；

**💡 创新点**

① 通过误差感知度量 ε_i 评估本地 GP 模型质量，实现动态选取高质量邻居；② 推导误差上界理论，提供预测误差保证；③ 设计贪心与自适应选择策略，兼顾精度与计算效率；

**🔧 技术方法**

基于 Gaussian 过程回归、误差感知度量、贪心/自适应选择、数据删除与快速预测；

**📊 数据集**

三组数据集：1）Toy 4 代理的离散函数；2）KIN40K（1 万点）与 POL（1 万点）两大真实数据集；

**📈 对比分析**

与传统分布式 GP 方法（POE、GPOE、BCM、RBCM、DAC、MOE）在 SMSE 与预测时间上比较。EIGP 在 8/16 代理下均取得最低 SMSE，且 gEIGP 与 aEIGP 的平均预测时间显著小于其他方法，证明了“少而精”的优越性；

**⚠️ 局限性**

1）模型需在接收新数据后更新，导致实时预测时延；2）自适应算法 aEIGP 中阈值 θ 为静态，无法自适应不同场景；3）尚未处理异步预测与通信延迟问题。

---

## 167. Single-step Controllable Music Bandwidth Extension With Flow Matching

**arXiv ID:** 2601.14356 | [PDF](https://arxiv.org/pdf/2601.14356v1)

**作者:** Carlos Hernandez-Olivan `[一作]` (Universal Music Group), Elio Quinton `[通讯]` (Universal Music Group)

**通讯引用:** 64 | [OpenAlex ID](https://openalex.org/A5051960947)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了一种基于 FlowMatching 的音乐带宽扩展与可控恢复方法，旨在从降质音频中单步恢复全频段音质并实现细粒度控制；

**💡 创新点**

创新点包括：①将 FlowHigh 框架迁移到音乐域并支持单步采样；②提出 Dynamic Spectral Contour (dsc) 作为针对带宽扩展的音频特征控制信号；③改进的 cfg‑zero* 引导策略实现更精细的条件控制；

**🔧 技术方法**

使用的技术包括 FlowMatching 与 Transformer 向量场估计器、BigVGAN 声码器、cfg‑zero* 方向引导、dsc 特征提取与控制；

**📊 数据集**

训练数据集为 8503 首 425 小时的商业音乐曲目，按 1.5 s 片段划分，并通过随机低通滤波器生成多样化降质样本；

**📈 对比分析**

与两种扩散模型（1D‑Diff 与 CQT‑Diff）对比，实验显示 FlowHigh 在 FAD、LSD 等指标上均优于基线，且在 dsc 控制下恢复精度最高；

**⚠️ 局限性**

局限性包括：dsc 超过自然频带范围会产生明显噪声；在极低能量或静音区 dsc 控制仍不够理想；模型对极端超宽带扩展的适应性有限。

---

## 168. AdaTIR: Adaptive Tool-Integrated Reasoning via Difficulty-Aware Policy Optimization

**arXiv ID:** 2601.14696 | [PDF](https://arxiv.org/pdf/2601.14696v1)

**作者:** Zhaiyu Fang `[一作]` (Trip.com Group), Ruipeng Sun `[通讯]` (Trip.com Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 AdaTIR 框架，动态调节工具调用频率，内化 LLM 的推理能力，显著减少工具使用。

**💡 创新点**

创新点在于引入难度感知的效率奖励以及 Clipped Advantage Shaping（CAS）机制，既抑制工具冗余，又防止奖励反转导致的训练不稳定。

**🔧 技术方法**

技术上采用 GRPO 策略优化，基于组内成功率估计任务难度，结合奖励塑造和 CAS 的优势重塑，实现工具调用与推理内部化的平衡。

**📊 数据集**

使用 ReTool‑SFT 进行 SFT，DAPO‑Math‑17k 进行 RL 训练，并在 AIME 2024/25、AMC23、GSM8K 等数学推理数据集以及 Search‑R1 进行评测。

**📈 对比分析**

与传统 TIR、OTC、ToRL 等基线对比，AdaTIR 在保持甚至提升准确率的同时，简单任务工具调用降低高达 97.6%，复杂任务降低 28.2%，并在工具被禁用时实现 4.8% 的准确率提升。

**⚠️ 局限性**

局限性包括仅在数学推理任务验证、模型规模有限（3B/7B）、对超参数 δ、β 的探索不足，以及使用经验难度估计，可能不易推广至开放式编码或其他领域。

---

## 169. Beyond Error-Based Optimization: Experience-Driven Symbolic Regression with Goal-Conditioned Reinforcement Learning

**arXiv ID:** 2601.14693 | [PDF](https://arxiv.org/pdf/2601.14693v1)

**作者:** Jianwen Sun `[一作]` (Central China Normal University), Xiaoxuan Shen `[通讯]` (Central China Normal University)

**通讯引用:** 1302 | [OpenAlex ID](https://openalex.org/A5074111945)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出EGRL‑SR框架，基于目标条件强化学习和HER进行符号回归，并设计All‑Point Satisfaction Reward和Structure‑Guided Heuristic Exploration策略来引导搜索；

**💡 创新点**

①将符号回归视作目标条件RL任务；②利用HER获得多目标经验并学习通用x‑y映射；③采用二进制APSR奖励消除误导性低误差结构；④用结构引导的启发式探索（SGHE）提升搜索多样性与覆盖；

**🔧 技术方法**

目标条件RL（GCRL）、Hindsight Experience Replay、Double‑Dueling DQN、APSR奖励、SGHE探索、后缀生成策略以及对状态/动作空间的设计；

**📊 数据集**

使用三大公开基准：Nguyen、Livermore、Keijzer；

**📈 对比分析**

与DSR、NeSymReS、E2E、GOMEA、PySR、DSO、TPSR、EQL‑Div、DySymNet等十余方法在统一搜索步数1.6 M下对比；EGRL‑SR在绝大多数表达式长度区间的精确恢复率最高，尤其在复杂结构和噪声环境下显著优于基线；

**⚠️ 局限性**

目前常量通过变量与运算符组合间接表示，导致搜索空间膨胀，对含复杂常量的表达式恢复效果不足，未来需要针对常量设计更有效的表示与搜索策略。

---

## 170. IB-GRPO: Aligning LLM-based Learning Path Recommendation with Educational Objectives via Indicator-Based Group Relative Policy Optimization

**arXiv ID:** 2601.14686 | [PDF](https://arxiv.org/pdf/2601.14686v1)

**作者:** Shuai Wang `[一作]` (East China Normal University), Aimin Zhou `[通讯]` (East China Normal University)

**通讯引用:** 9607 | [OpenAlex ID](https://openalex.org/A5050248676)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大语言模型的学习路径推荐框架 IB‑GRPO，用以在满足教育学原则（如可达发展区 ZPD）和操作约束（长度、轨迹多样性）的前提下，优化学习者长期学习效果。

**💡 创新点**

创新点包括：① 采用遗传算法与教师强化学习模型相结合的混合专家生成器，为监督微调提供高质量、多样化的演示数据；② 将 ZPD 作为可计算的奖励函数，实时衡量推荐的难度与学习者水平的匹配；③ 在策略优化中使用 I_{ε+} 指标计算组内相对优势，避免传统的手工加权线性缩放，直接逼近 Pareto 前沿。

**🔧 技术方法**

核心技术包括：大语言模型 Qwen2.5‑7B 作为生成器；遗传算法（GA）搜索 + 离线教师 RL 策略生成混合演示；监督微调（SFT）行为克隆；Indicator‑Based Group Relative Policy Optimization（IB‑GRPO）利用 I_{ε+} 指标计算优势；使用 Knowledge Evolution Simulator（KES）评估学习效果；多目标奖励向量（学习效果、ZPD 对齐、长度、轨迹多样性）。

**📊 数据集**

实验使用公开教育数据集 ASSIST09 与 Junyi，通过 KES 仿真环境进行评估。

**📈 对比分析**

与多种基线（传统 RL、基于 LLM 的 ReAL、GenAL、教育专用 RL 如 CSEAL、GEPKSD 等）比较，IB‑GRPO 在学习效果、ZPD 对齐分数、轨迹多样性以及长度满意度等四个维度均表现最佳，成功逼近 Pareto 前沿，且在不同路径长度（5、10、20）下均保持优势。

**⚠️ 局限性**

主要局限包括：依赖仿真器（KES）对真实学生学习效果的近似；多目标指标仍需在实践中进行调优，尤其是对 I_{ε+} 指标的尺度参数；混合专家生成与大模型微调成本相对较高；在极大规模、复杂知识图谱场景下的可扩展性尚未验证。

---

## 171. Local Language Models for Context-Aware Adaptive Anonymization of Sensitive Text

**arXiv ID:** 2601.14683 | [PDF](https://arxiv.org/pdf/2601.14683v1)

**作者:** Aisvarya Adeseye `[一作]` (University of Turku), Mohammad Tahir `[通讯]` (University of Turku)

**通讯引用:** 1852 | [OpenAlex ID](https://openalex.org/A5035027638)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用本地大语言模型构建了结构化框架 SFAA，对定性访谈文本进行敏感信息检测、分类和自适应匿名化。

**💡 创新点**

提出了三步检测-分类-匿名化流程，并结合四种策略（规则替换、上下文重写、泛化、抑制），在保持语义一致性的同时实现可重复、可控的隐私保护；同时验证 Phi 模型在准确率、召回率上优于 LLaMA 与人工。

**🔧 技术方法**

使用本地 LLM（LLaMA 3B 参数、Phi 4B 参数）以及规则替换、上下文重写、泛化、抑制四种匿名策略，并辅以人工复核。

**📊 数据集**

两个案例研究：82 份面对面访谈（关于组织内游戏化）和 93 份 AI 访谈（关于 LLM 在工作场所的使用）。

**📈 对比分析**

与人工标注对比，Phi 在识别敏感项的召回率达到 91‑94%，精确率 95‑96%，比 LLaMA 高约 4‑6%，误检率低于人工；总体性能提升约 10%。

**⚠️ 局限性**

局限性包括：对不同语言/方言、非访谈文本的适应性未知；LLM 可能产生幻觉，需要额外过滤；仅在两类访谈数据验证，缺乏多学科、多语言的泛化验证。

---

## 172. HCVR Scene Generation: High Compatibility Virtual Reality Environment Generation for Extended Redirected Walking

**arXiv ID:** 2601.14679 | [PDF](https://arxiv.org/pdf/2601.14679v1)

**作者:** Yiran Zhang `[一作]` (Purdue University), Aniket Bera `[通讯]` (Purdue University)

**通讯引用:** 1711 | [OpenAlex ID](https://openalex.org/A5058453308)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种基于大型语言模型的VR场景生成框架HCVR，结合ENI++兼容性评估，自动生成与用户物理空间匹配且支持红irection walking的虚拟环境。

**💡 创新点**

提出ENI++度量与LLM引导的场景生成流程，实现在虚拟与物理空间之间的兼容性评估与优化，从而显著降低碰撞率并提升用户体验。

**🔧 技术方法**

使用GPT‑4o进行对象选择与布局规划，BUDAS房间分割算法，OOB（超出边界）检测与修正，基于可视多边形相似度的ENI++评估，以及ARC重定向控制。

**📊 数据集**

采用5对合成物理/虚拟空间配对和16名受试者的实验数据，未使用公开3D模型数据集，而是通过LLM检索公共3D资产。

**📈 对比分析**

与基线 LLM+ARC 及仅 ENI++ 进行碰撞计数与用户满意度对比，ENI++&ARC 在所有实验中将碰撞数降至近零，平均碰撞率降低超过90%，整体满意度与基线相当或略优。

**⚠️ 局限性**

局限在于仅处理静态障碍，LLM 对空间推理与高度缩放的不准确，以及未考虑动态障碍或实时眼动重定向的适配。

---

## 173. Transfer Learning from One Cancer to Another via Deep Learning Domain Adaptation

**arXiv ID:** 2601.14678 | [PDF](https://arxiv.org/pdf/2601.14678v1)

**作者:** Justin Cheung `[一作]` (Johns Hopkins University), Alhassan S. Yasin `[通讯]` (Johns Hopkins University)

**通讯引用:** 120 | [OpenAlex ID](https://openalex.org/A5033362818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

评估了肺、结肠、乳腺和肾腺癌之间的跨域分类性能，利用域对抗网络（DANN）和可解释性方法验证模型泛化能力。

**💡 创新点**

将DANN应用于多器官腺癌数据集，系统研究染色规范化对域适应的影响，并使用集成梯度（Integrated Gradients）解释模型决策，构建跨域迁移与解释相结合的新范式。

**🔧 技术方法**

采用ResNet‑50作为特征提取器，构建域对抗网络（DANN）并使用梯度反转层进行域适配；配合集成学习、染色规范化技术以及集成梯度可解释性。

**📊 数据集**

使用Kaggle Multi Cancer Dataset，包含乳腺、肺、结肠腺癌的组织学图像（肾癌CT图像后被排除）。

**📈 对比分析**

与单域监督训练和模型集成进行对比；DANN在肺域达95.56%准确率、结肠域78.48%准确率，而乳腺域仅49.22%；染色规范化在肺域降低准确率，但在乳腺和结肠域显著提升；基线和集成模型接近随机表现，证明域适应的必要性。

**⚠️ 局限性**

受限于成像技术差异（肾CT与组织学差异大）导致乳腺和肾域迁移效果差；染色规范化并非对所有域都有利；缺乏多类别标签跨域迁移的实验，且样本不平衡和标签不一致可能影响结果。

---

## 174. Efficient reformulations of ReLU deep neural networks for surrogate modelling in power system optimisation

**arXiv ID:** 2601.14673 | [PDF](https://arxiv.org/pdf/2601.14673v1)

**作者:** Yogesh Pipada Sunil Kumar `[一作]` (School of Electrical and Mechanical Engineering, University of Adelaide), Julian Lesmos-Vinasco `[通讯]` (Watts A/S)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文提出了一种针对凸化ReLU深度神经网络（DNN）的线性规划（LP）重构方法，并将其嵌入到电力系统的聚合器投标优化问题中。

**💡 创新点**

创新点在于：①将ReLU DNN的权重矩阵（除第一层外）限定为非负，从而使网络在求解最小化目标时可以被完全线性化；②提出的LP重构在保持紧致性的同时，消除了传统MIP重构中的整数变量，显著提升了可解性和效率；③在不同网络结构和市场情景下进行系统评估，展示其鲁棒性。

**🔧 技术方法**

使用的技术包括：ReLU DNN的凸化与LP重构、Python Gurobi ML包、PCAR/PCTAR罚函数法、PWL分段线性法、随机数据生成与归一化、Adam优化器、MSE损失函数。

**📊 数据集**

数据集为：1) 约3×10^5条随机生成的四维输入（x_t, x̃_t, q_t, r_t）及对应的目标成本；2) 丹麦mFRR容量市场的真实价格场景（低、中、高三类各10个场景），以及随机生成的跨时矩阵A和灵活性可用量曲线。

**📈 对比分析**

比较方法：在同一优化模型中使用四种代理方法（PWL、Gurobi ML、PCAR/PCTAR、提出的LP），在不同网络宽度/深度、不同价格场景下测量运行时间、MIP缺口、实现利润、RMSE。结果显示：提出的LP在运行时间上比PWL快数百到数千倍；在所有场景下，利润与RMSE均位列第二，仅略低于PWL；相比Gurobi ML在大网络上可解性差，MIP缺口高；PCAR/PCTAR在低价场景中出现亏损。

**⚠️ 局限性**

局限性：①仅适用于求解目标为最小化网络输出的优化问题；②凸化网络只能逼近凸且单调递增的函数，表达能力受限；③在极大网络或非凸目标场景下可能无法得到最优或收敛；④训练时需高质量数据与充分归一化，否则网络拟合误差会直接影响优化结果。

---

## 175. READ-Net: Clarifying Emotional Ambiguity via Adaptive Feature Recalibration for Audio-Visual Depression Detection

**arXiv ID:** 2601.14651 | [PDF](https://arxiv.org/pdf/2601.14651v1)

**作者:** Chenglizhao Chen `[一作]` (China University of Petroleum), Hui Yu `[通讯]` (University of Glasgow)

**通讯引用:** 12958 | [OpenAlex ID](https://openalex.org/A5006580423)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了READ-Net框架，用于在音频-视频情境下检测抑郁症，并通过自适应特征再校准解决情绪模糊问题。

**💡 创新点**

创新点在于提出Adaptive Feature Recalibration（AFR）机制，结合分层特征分离、双重一致性正则化和不对称蒸馏，能够动态过滤情绪噪声并保留抑郁相关情绪线索。

**🔧 技术方法**

采用互信息估计、图卷积网络、注意力机制、知识蒸馏以及预训练的情绪识别分支，实现多模态特征的分离与融合。

**📊 数据集**

在LMVD、D‑vlog和DAIC‑WOZ三个公开抑郁检测数据集上进行实验。

**📈 对比分析**

与SOTA方法（如Xception、ViT、DepMamba等）对比，READ‑Net在三大数据集上平均提升约4.55%准确率、1.26%F1分数，显著优于现有模型。

**⚠️ 局限性**

局限性包括对情绪标注质量敏感、对极端情境下的鲁棒性尚需进一步验证，以及模型仍具一定计算开销。

---

## 176. Spatially Generalizable Mobile Manipulation via Adaptive Experience Selection and Dynamic Imagination

**arXiv ID:** 2601.14649 | [PDF](https://arxiv.org/pdf/2601.14649v1)

**作者:** Ping Zhong `[一作]` (Central South University), Jianxin Wang `[通讯]` (Central South University)

**通讯引用:** 33106 | [OpenAlex ID](https://openalex.org/A5100438360)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在移动操控（MM）任务中，提出了自适应经验选择（AES）结合循环状态空间模型（RSSM）的模型预测前瞻规划（MPFP），从而显著提升样本效率和空间泛化能力。

**💡 创新点**

创新点：① AES利用局部异常检测、任务关键性与预测误差三种优先级，主动挑选影响任务成功的关键经验碎片；② RSSM在MM中实现未来动态想象，配合MPFP对行动序列进行前瞻优化，使得移动底盘与机械臂能协同完成复杂任务；③ 两者协同训练共享经验缓冲区，既强化当前任务，又可直接迁移至新空间布局。

**🔧 技术方法**

核心技术包括：循环状态空间模型（RSSM）、自适应经验选择（AES）与局部异常因子（LOF）检测、任务关键性优先、预测误差优先；强化学习使用Soft Actor-Critic（SAC）与交叉熵方法（CEM）进行模型预测规划；整体实现基于PyTorch和ROS。

**📊 数据集**

使用在四种配置（跨室、仓库、家居、动态场景）下的仿真环境（基于UR5 + Husky平台）进行实验，并在真实机器人（Husky A200 + UR5 + RealSense D435）上验证，未使用公开大规模数据集，全部数据来自仿真/现场采集。

**📈 对比分析**

与N^2M^2、BHyRL、AuxDistill、Dreamer V3、TD-MPC2等基线进行对比，评估指标为AIKF、ABC、TCR、PSR；实验表明，本方法在TCR和PSR上提升≈50%+，AIKF显著下降，显示出更高的任务完成率和更低的IK失败率，且在不同空间布局下保持较好性能。

**⚠️ 局限性**

局限性：观测空间理想化，使用地面真值占据图；实际部署需从深度相机或激光雷达构建低噪声局部地图，鲁棒性和传感器噪声影响尚未充分验证。

---

## 177. DesignBridge: Bridging Designer Expertise and User Preferences through AI-Enhanced Co-Design for Fashion

**arXiv ID:** 2601.14639 | [PDF](https://arxiv.org/pdf/2601.14639v1)

**作者:** Yuheng Shao `[一作]` (ShanghaiTech University), Quan Li `[通讯]` (ShanghaiTech University)

**通讯引用:** 1377 | [OpenAlex ID](https://openalex.org/A5100689370)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个多平台AI辅助交互系统DesignBridge，用于时装设计师与用户在共创过程中的协作，支持初始设计框架、偏好表达收集、偏好集成设计三阶段工作流。

**💡 创新点**

通过结构化设计空间、细粒度偏好表达工具、基于推荐的偏好收集以及用户偏好引导的LoRA微调，使设计师能够在保持专业判断的同时快速整合多样用户偏好，实现真正的混合式协作。

**🔧 技术方法**

结合GPT‑4o文本分析、VisionRealistic v2 FluxDev扩散模型、LoRA微调、FashionCLIP视觉‑语言模型、PPNN个性化偏好网络、SHAP可解释性以及虚拟试衣与LayerMask等技术。

**📊 数据集**

以Farfetch电商平台的3294件上衣样本及其文本描述为基础，构建九维设计空间，后续使用生成模型对属性组合进行采样。

**📈 对比分析**

通过技术评估对比预训练模型、两阶段LoRA微调模型与GPT‑4o，在属性准确性、设计一致性、目标属性满意度与整体设计满意度四维度使用7点Likert评分，实验显示两阶段微调显著优于基线，第二阶段在目标属性与整体满意度上进一步提升。

**⚠️ 局限性**

样本规模有限、仅覆盖上衣类、数据来源单一导致多样性不足、冷启动与信息过载挑战、用户表达偏好仍需手工调优，系统对大规模用户群体的适应性和跨域泛化尚待验证。

---

## 178. A Brain-inspired Embodied Intelligence for Fluid and Fast Reflexive Robotics Control

**arXiv ID:** 2601.14628 | [PDF](https://arxiv.org/pdf/2601.14628v1)

**作者:** Weiyu Guo `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 42888 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于生物神经运动层级的NeuroVLA架构，实现视觉-语言-动作的多层次分离与协同。

**💡 创新点**

创新点在于将语义规划、可变频适应与事件驱动的脉冲神经网络分层，重现皮层-小脑-脊髓三层生物学功能，兼具高速平滑运动与低功耗自适应反射。

**🔧 技术方法**

采用预训练视觉‑语言模型+Q‑Former、可变频小脑模块（GRU+FiLM+迭代细化）以及基于LIF的残差SNN实现脊髓执行，辅以脉冲层级硬件加速。

**📊 数据集**

使用LIBERO/LIBERO‑Plus仿真数据集以及真实机器人实验数据。

**📈 对比分析**

相较于OpenVLA、UniVLA、WorldVLA等主流VLA基线，NeuroVLA在精细抓取、液体控制、节奏动作及碰撞恢复等任务上均取得更高成功率（如安全恢复54.8%对0%），并在能耗与推理延迟上实现10×加速与0.87 mJ/推理。

**⚠️ 局限性**

局限性包括SNN训练仍依赖GPU的伪梯度，缺乏在线STDP学习，且未在专业Neuromorphic芯片上完整验证能耗优势。

---

## 179. UniCon: A Unified System for Efficient Robot Learning Transfers

**arXiv ID:** 2601.14617 | [PDF](https://arxiv.org/pdf/2601.14617v1)

**作者:** Yunfeng Lin `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18170 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了轻量化框架 UniCon，统一了异构机器人学习控制的状态、控制流和仪表化，支持从仿真到真实硬件的无缝部署。

**💡 创新点**

创新点包括：1) 将所有运行时状态统一向量化为全局数组，剥离数据与逻辑；2) 采用执行图编排的控制块，实现高度模块化和可插拔；3) 提供多种后端存储（共享内存、消息队列、ROS桥接）并与主流仿真器/硬件对齐。

**🔧 技术方法**

使用技术包括：Python+NumPy向量化操作、单线程高效计算、批量数据流、执行图（loop、zip、chain）编排、状态映射与索引、自动化代码生成与配置文本化。

**📊 数据集**

评估数据集涵盖12种型号、7家制造商的机器人（包括 Unitree H1、A1、G1、Aliengo、Dex3-1、Dobot Atom、ROHand 等）以及对应的仿真环境（IsaacGym、IsaacSim、IsaacLab、MuJoCo、PyBullet、Webots、Gazebo 等）。

**📈 对比分析**

与 ROS、ROS 2 以及手工实现的对比显示：SLOC 减少至原来 10% 左右；单周期推理延迟平均降低至 50% 以内；End‑to‑End 延迟约 2.5×缩短；整体推理效率显著优于传统中间件。

**⚠️ 局限性**

局限性在于：1) 主要采用单线程设计，在极高并发场景下可能成为瓶颈；2) 目前仅支持 Python 与 NumPy，跨语言兼容性待完善；3) 对非常复杂的并行或分布式控制任务的支持仍有限。

---

## 180. Towards Cybersecurity Superintelligence: from AI-guided humans to human-guided AI

**arXiv ID:** 2601.14614 | [PDF](https://arxiv.org/pdf/2601.14614v1)

**作者:** Víctor Mayoral-Vilches `[一作]` (Alias Robotics), Patxi Mayoral-Pizarroso `[通讯]` (Alias Robotics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了从人工智能辅助人类进行渗透测试（PentestGPT）、到全自动专家级AI代理（CAI）以及进一步加入博弈论推理的游戏化AI代理（G-CTR），通过三阶段架构展示了网络安全超智能的演进。

**💡 创新点**

创新点包括：① 在渗透测试中引入任务树与自然语言校验机制，显著提升LLM在多步策略上的连贯性；② 构建面向安全的多模态代理框架CAI，完全自动化完成工具执行、结果分析与动态交互；③ 通过神经符号架构将攻击图生成与Nash均衡计算嵌入LLM提示，形成闭环策略反馈，显著提升AI在攻防场景中的战略表现。

**🔧 技术方法**

核心技术包括大型语言模型（LLM）、工具调用接口（API）、对话式提示工程、攻击图生成与剪枝、Cut‑the‑Rope（CTR）算法计算纳什均衡、自然语言摘要与战略注入、以及多代理协同与监督机制。

**📊 数据集**

使用的数据集为来自公开CTF竞赛（如Neurogrid、Dragos OT CTF等）的182个子任务及44个网络范围渗透测试场景，包含多领域（逆向、pwn、crypto、web、forensics、robotics）挑战；此外还采用了自定义的Cybench基准测试。

**📈 对比分析**

评估方法为在统一的CTF基准上对三阶段系统进行pass@3成功率、执行时间、API成本及行为方差等指标比较。PentestGPT在成功率上提升至47.8%；CAI实现了82.6%成功率、速度提升至多达938×、成本下降156×；G‑CTR进一步达到100%成功率，成功率提升至42.9%（相对基线）且成本/成功率降低2.7×，行为方差减少5.2×。

**⚠️ 局限性**

局限性包括：LLM算力成本高（每十亿token约$5,940），需要进一步优化多模型调度以降低成本；在高度创造性或新颖攻击场景下仍受限；完全自主决策能力尚未实现，仍需人类监督；长期部署需持续更新知识库以防性能漂移；以及在真实事件响应中尚未验证鲁棒性。

---

## 181. Seeing to Think? How Source Transparency Design Shapes Interactive Information Seeking and Evaluation in Conversational AI

**arXiv ID:** 2601.14611 | [PDF](https://arxiv.org/pdf/2601.14611v1)

**作者:** Jiangen He `[一作]` (University of Tennessee), Jiqun Liu `[通讯]` (University of Oklahoma)

**通讯引用:** 626 | [OpenAlex ID](https://openalex.org/A5088558868)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比四种源信息展示界面（Collapsible、Hover Card、Footer、Aligned Sidebar），通过 372 名参与者的对照实验，细粒度记录用户行为并自动评估其批判性思维与证据整合水平。

**💡 创新点**

将源透明度视为交互行为现象而非仅仅的文档特征，揭示界面可见度、可访问性与信息密度相互作用对批判性评估的影响，并提出在信息过载时 Sidebar 设计能有效提升用户的批判性思考。

**🔧 技术方法**

实验采用网页交互界面（React/HTML/CSS）实现四种源展示模式；使用行为日志采集工具（如 WebSocket 记录点击/滚动）和自动化批判性思维评估脚本（Python/Pandas 统计分析）。

**📊 数据集**

未使用公开语料库，而是构建了自己的 AI 对话数据集：将 4 个提示任务生成含源信息的响应，供实验参与者检索与引用；此外使用自定义评估问卷记录信任感与信息需求。

**📈 对比分析**

通过方差分析 (ANOVA) 和多重比较检验（Tukey）比较四种界面。结果显示 Hover Card 在即时验证方面最快；Sidebar 在高信息密度情境下，批判性思维与综合评分显著高于其它条件（p < .01）。

**⚠️ 局限性**

局限性包括：仅研究单一写作任务，未涵盖多任务或长期交互；样本主要为高校学生，外部可推广性受限；界面设计仅涵盖四种形式，未来可进一步扩展多模态交互或自适应展示。

---

## 182. MIND: Empowering Mental Health Clinicians with Multimodal Data Insights through a Narrative Dashboard

**arXiv ID:** 2601.14641 | [PDF](https://arxiv.org/pdf/2601.14641v1)

**作者:** Ruishi Zou `[一作]`, Xuhai "Orson" Xu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了MIND——一款集成多模态患者数据的叙事式仪表盘，帮助精神科医生快速获取临床、主动感测、被动感测等多源信息的洞察并用于决策。

**💡 创新点**

创新点在于：①将大型语言模型与规则‑基准分析相结合的双层推理管线，自动生成可解释、可验证的临床洞察；②采用叙事结构（线性与并行叙事）与生物‑心理‑社会模型对多模态数据进行梳理；③通过“详细信息即点”交互和引用源数据的可追溯机制提升可信度。

**🔧 技术方法**

技术手段包括：React＋Recharts构建前端；OpenAI GPT‑4.1 与 Agents SDK实现 LLM 代理；Python 统计库（scipy、pandas）执行 Mann‑Whitney、Mann‑Kendall、LOESS 等规则‑基准检测；JSON 接口传输洞察。

**📊 数据集**

主要使用公开的 GLOBEM 长期可穿戴与问卷数据（被动感测与主动感测）以及人工构造的模拟病历（SOAP 记录、访谈转录），以此生成三位模拟患者的数据集。

**📈 对比分析**

通过 16 名持证精神科医生的对内实验，对比基线“FACT”数据收集仪表盘，采用 Wilcoxon 符号秩检验。结果显示：MIND 在揭示隐藏洞察（p<0.001）和支持决策（p<0.01）方面显著优于基线；在多模态整合、叙事连贯性、效率等主观指标上亦均高于基线。

**⚠️ 局限性**

局限性包括：①实验仅使用少量同质化的模拟病例，缺乏真实病例的多样性与缺失值；②样本规模与招募方式偏向对技术开放的医生，缺乏普遍代表性；③LLM 生成的洞察需人工审核，潜在偏差与误报仍需进一步评估；④在实验室环境下进行，未检验在真实临床工作流中的可用性与长期效果。

---

## 183. Analog-to-Stochastic Converter Using Magnetic Tunnel Junction Devices for Vision Chips

**arXiv ID:** 2601.14640 | [PDF](https://arxiv.org/pdf/2601.14640v1)

**作者:** Naoya Onizawa `[一作]` (Tohoku University), Takahiro Hanyu `[通讯]` (Tohoku University)

**通讯引用:** 5535 | [OpenAlex ID](https://openalex.org/A5062434040)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于磁隧道结(MTJ)的模拟到随机（Bernoulli序列）转换器，用于视觉芯片的特征提取；

**💡 创新点**

创新点在于利用MTJ设备的概率切换特性直接完成一阶模拟到随机信号转换，消除了传统ADC+数字到随机转换的功耗与面积负担，并通过偏置电压与写时长校准抵消MTJ阻值波动；

**🔧 技术方法**

采用STT‑MTJ结构、概率计算模型、NS‑SPICE仿真与MATLAB模拟，并结合磁隧道结电路设计与随机计算原理；

**📊 数据集**

使用LENNA标准灰度图像进行随机边缘检测的仿真评估；

**📈 对比分析**

与传统二进制实现的边缘检测做对比，显示在软错误下随机实现的鲁棒性更高；同时通过面积仿真表明数字到随机转换器占总面积约43.6%，而MTJ转换器大幅降低面积与功耗；

**⚠️ 局限性**

目前仍缺乏完整的硬件实现与校准模块，MTJ阻值可变性需要额外校准，整体系统性能和能耗尚未在大规模真实场景中全面验证。

---

## 184. When Text-as-Vision Meets Semantic IDs in Generative Recommendation: An Empirical Study

**arXiv ID:** 2601.14697 | [PDF](https://arxiv.org/pdf/2601.14697v1)

**作者:** Shutong Qiao `[一作]` (University of Queensland), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 16608 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究将商品文本描述渲染成图像并通过OCR编码器进行编码，替代传统的文本编码器来学习语义 ID，评估其在单模和多模生成式推荐中的效果。

**💡 创新点**

首次提出OCR‑based视觉文本表示作为语义 ID 学习的可插拔替代方案，并系统性地在不同融合策略、模型和数据集上验证其鲁棒性与优势。

**🔧 技术方法**

使用 OCR 渲染 + OCR 编码器（DeepSeek‑OCR、Donut、TrOCR）、向量量化生成语义 ID、TIGER/LETTER 生成式推荐框架，以及早期融合与三种后期（Concat、Interleave、Alignment）融合策略。

**📊 数据集**

四大基准数据集：Luxury、Scientific、Instruments、Arts。

**📈 对比分析**

采用 Recall@K / NDCG@K 与传统文本编码器对比，OCR 文本在属性丰富、结构化描述的数据集上提升 Recall 5–10%、NDCG 1–5%，在自然语言描述的数据集上表现相当。

**⚠️ 局限性**

依赖 OCR 模型与渲染步骤导致额外计算成本；在文本简短、自然语言化场景收益有限；未深入探讨更高级的跨模态对齐与自适应融合技术。

---

## 185. FeedbackSTS-Det: Sparse Frames-Based Spatio-Temporal Semantic Feedback Network for Infrared Small Target Detection

**arXiv ID:** 2601.14690 | [PDF](https://arxiv.org/pdf/2601.14690v1)

**作者:** Yian Huang `[一作]` (University of Electronic Science and Technology of China), Zhenming Peng `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 5065 | [OpenAlex ID](https://openalex.org/A5016112252)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种稀疏帧基空间时序反馈网络 FeedbackSTS-Det，用于红外小目标检测。

**💡 创新点**

创新点包括闭环语义反馈机制（前向与后向精炼模块）以及嵌入式稀疏语义模块，实现低成本长程时序建模。

**🔧 技术方法**

技术采用 3D Res‑UNet 骨干，配合前向/后向语义精炼、稀疏帧分组、BFBM、可变间隔采样和 Soft‑IoU 损失等。

**📊 数据集**

使用的实验数据集为 NUDT‑MIRSDT 与 IRSatVideo‑LEO 两个红外小目标基准集。

**📈 对比分析**

与多种单帧与多帧模型驱动和数据驱动方法对比，FeedbackSTS-Det 在 mIoU、F1、P_d、AUC 等指标上均取得领先，并且误报率极低。

**⚠️ 局限性**

局限性在于仅在有限的场景下测试，未覆盖极暗目标、强杂波或高速目标，泛化能力需进一步验证。

---

## 186. Mirai: Autoregressive Visual Generation Needs Foresight

**arXiv ID:** 2601.14671 | [PDF](https://arxiv.org/pdf/2601.14671v1)

**作者:** Yonghao Yu `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**通讯引用:** 6537 | [OpenAlex ID](https://openalex.org/A5048624196)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在视觉自回归生成模型训练期间加入未来信息（foresight）的框架（名为Mirai），通过对内部表示进行2D网格对齐，显著提升图像生成的全局一致性与收敛速度。

**💡 创新点**

核心创新在于将未来上下文的监督信息以二维空间对齐的方式注入自回归模型内部表示，而非传统的一步预测或多词预测，从而在保持纯自回归推理的前提下实现更强的全局语义约束。

**🔧 技术方法**

采用的技术包括：自回归视觉生成（LlamaGen），离线或在线指数移动平均（EMA）作为显式foresight编码器，预训练的双向视觉编码器（DINOv2）作为隐式foresight编码器，以及对内部Transformer层进行余弦相似度对齐的训练损失。

**📊 数据集**

主要使用ImageNet 256×256的图像数据集进行训练与评估，数据经过十视图裁剪增强。

**📈 对比分析**

与原始LlamaGen及其他AR、GAN、扩散模型相比，Mirai在ImageNet上实现了FID从5.34降至4.34（LlamaGen‑B）并且训练迭代次数减少约10×；在更大模型上可达2.59的FID，显著优于同类AR方法。

**⚠️ 局限性**

局限性包括：对超参数（如foresight窗口大小、对齐层位置、λ调度）的敏感性；仅在单尺度图片上验证，跨尺度或高分辨率的适用性尚未充分探究；以及在推理时仍需额外投影头训练，虽无推理成本，但训练开销略增。

---

## 187. INFA-Guard: Mitigating Malicious Propagation via Infection-Aware Safeguarding in LLM-Based Multi-Agent Systems

**arXiv ID:** 2601.14667 | [PDF](https://arxiv.org/pdf/2601.14667v1)

**作者:** Yijin Zhou `[一作]` (Shanghai Jiao Tong University), Jing Shao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 9670 | [OpenAlex ID](https://openalex.org/A5023198186)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了针对多智能体系统的感染感知防御框架 INFA‑Guard，能够识别并处理感染代理。

**💡 创新点**

创新点在于将攻击代理与感染代理区分为不同威胁类别，并利用感染动态与拓扑约束实现精准检测与修复。

**🔧 技术方法**

采用多分支图神经网络、时间序列特征提取、拓扑约束损失和基于回复级的修复策略等技术。

**📊 数据集**

使用 CSQA、MMLU、GSM8K、InjecAgent 与 PoisonRAG 等公开数据集进行攻击与防御实验。

**📈 对比分析**

与 G‑Safeguard、AgentSafe、AgentXposed 等基线相比，平均降低攻击成功率 33%，提升系统防御成功率 6%，在不同 LLM 与拓扑下保持最优性能。

**⚠️ 局限性**

主要局限在于对有标签训练数据依赖、需要至少观察一轮对话才能检测、在标签稀缺场景下推广受限。

---

## 188. Calibrated uncertainty quantification for prosumer flexibility aggregation in ancillary service markets

**arXiv ID:** 2601.14663 | [PDF](https://arxiv.org/pdf/2601.14663v1)

**作者:** Yogesh Pipada Sunil Kumar `[一作]` (School of Electrical and Mechanical Engineering, University of Adelaide), Julian Lesmos-Vinasco `[通讯]` (Watts A/S)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了基于 Monte Carlo dropout 与 conformal prediction 的混合不确定性量化框架，用于在丹麦频率恢复市场中预测聚合器的可再生能源灵活性并进行投标。

**💡 创新点**

创新点在于提出可扩展的 MCD–CP 组合，实现多维时序预测区间，并将其嵌入决策相关的机会约束优化，以满足 P90 可靠性要求并最大化聚合器收益。

**🔧 技术方法**

使用技术包括深度神经网络与 dropout 的 Monte Carlo 推断、分割式 conformal prediction 的多变量一致性得分、机会约束优化、仿真合成数据生成以及聚合器投标模型。

**📊 数据集**

数据集来源于行业级家用能源管理系统（Watts A/S HEMS）生成的规模化合成数据，包含 22,000 个场景、100 家 prosumer、24 小时时间步，并额外 100 场景用于收益评估。

**📈 对比分析**

方法通过与单纯 MCD、分位数估计、Bonferroni 校正等基线对比评估；MCD–CP 方法满足 90% 覆盖率，显著降低超投标率；在收益方面，mcp 与 mmcp 取得最高调整后利润，约占完美信息收益的 70%，相较传统方法提升可靠性与经济性。

**⚠️ 局限性**

局限性包括对精确市场价格预测的高度依赖、中央化训练可能带来隐私与安全问题、未系统评估对预测误差和样本量的敏感性、仅聚焦上调市场、缺乏对分散式学习与动态激励机制的探索。

---

## 189. NeuroFilter: Privacy Guardrails for Conversational LLM Agents

**arXiv ID:** 2601.14660 | [PDF](https://arxiv.org/pdf/2601.14660v1)

**作者:** Saswat Das `[一作]` (University of Virginia), Ferdinando Fioretto `[通讯]` (University of Virginia)

**通讯引用:** 1254 | [OpenAlex ID](https://openalex.org/A5052534316)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了NeuroFilter框架，利用LLM内部激活空间的线性结构检测对话中的隐私泄露意图。

**💡 创新点**

创新点包括：①发现隐私违规意图在激活空间可线性分离；②提出激活速度概念以捕捉多轮对话中的隐私泄露动态；③实现零误报、低成本的实时防御。

**🔧 技术方法**

主要技术手段为：线性激活探针（logistic回归）、激活投影、激活速度计算与累计漂移、单/多轮激活监测。

**📊 数据集**

实验使用了PrivacyLens、CMPL、Fractured SORRY‑Bench以及AutoDAN生成的攻击数据集。

**📈 对比分析**

与SAE、Llama Guard和agentic network firewall等基线比较，NeuroFilter在单/多轮攻击下均实现0%泄漏、0%误报，推理成本和延迟低数百到数千倍。

**⚠️ 局限性**

局限性在于需为每种上下文重新训练探针，细粒度属性识别效果有限，对极强白盒攻击的鲁棒性仍待进一步验证。

---

## 190. DeepMoLM: Leveraging Visual and Geometric Structural Information for Molecule-Text Modeling

**arXiv ID:** 2601.14732 | [PDF](https://arxiv.org/pdf/2601.14732v1)

**作者:** Jing Lan `[一作]` (Hong Kong Polytechnic University), Jung Sun Yoo `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 1802 | [OpenAlex ID](https://openalex.org/A5026267908)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种双视图多模态框架 DeepMoLM，能够将高分辨率分子图像与 3D 结构指纹对齐，完成分子图像理解和文本生成任务。

**💡 创新点**

创新点包括：① 双路径 DeepEncoder 能保留高频立体化学细节；② 采用离散化的 E3FP 3D 指纹作为几何描述符；③ 通过交叉注意力融合视觉与几何信息，无需原子坐标即可实现物理一致的文本生成。

**🔧 技术方法**

使用的技术包括：SAM‑CLIP 双通道视觉编码器、E3FP 3D 指纹、交叉注意力融合投影、Qwen2‑VL‑7B 视觉语言解码器，以及两阶段预训练+指令微调训练策略。

**📊 数据集**

使用的数据集包括 PubChem 3D 结构‑文本对、ChEBI‑20 描述生成数据、PubChem 属性预测数据，以及公开的 SMILES/图像数据集（如 MolScribe）作对照。

**📈 对比分析**

在分子字幕、属性预测和描述生成等任务上与多类基线（专业模型 MolT5、UniMoT、3D‑MoLM；通用模型 Llama2、Qwen2‑VL 等）进行对比，DeepMoLM 在 PubChem captioning 上 METEOR 提升 12.3%，属性预测 MAE 13.64 g/mol，描述生成 BLEU‑2/ROUGE‑L 超过通用基线，总体性能领先。

**⚠️ 局限性**

局限性包括：① 依赖高质量 2D 图像，图像质量或绘制风格差异会影响效果；② 缺乏直接的 3D 结构推断能力，对高度异构体仍易出错；③ 需要大量高分辨率图像和预训练资源，部署成本较高。

---

## 191. Synthetic Data Augmentation for Multi-Task Chinese Porcelain Classification: A Stable Diffusion Approach

**arXiv ID:** 2601.14791 | [PDF](https://arxiv.org/pdf/2601.14791v1)

**作者:** Ziyao Ling `[一作]` (University of Bologna), Giovanni Delnevo `[通讯]` (University of Bologna)

**通讯引用:** 953 | [OpenAlex ID](https://openalex.org/A5034405366)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了使用 Stable Diffusion+LoRA 生成合成瓷器图像来扩充稀缺的宋元瓷器数据集，并将其与传统增强一起用于多任务 CNN（朝代、窑、釉、类型）分类。

**💡 创新点**

创新点在于：①针对不同属性设计结构化提示并微调 LoRA，②系统评估不同合成比例（5% 与 10%）对四项任务的影响，③给出针对纹理与形态任务的实用应用准则。

**🔧 技术方法**

采用 Stable Diffusion 2.1 与 LoRA 微调进行图像生成，MobileNetV3 作为多任务 backbone，传统几何/光照增强，人工与自动双重质量控制，以及 t‑SNE 等可视化方法。

**📊 数据集**

使用 7,263 张来自北京故宫与台北故宫的宋元瓷器真实照片，经过传统增强后扩充至 25,877 张；再分别生成 570 张（5%）与 2,500 张（10%）合成图像。

**📈 对比分析**

在同一 801 张测试集上采用 Top‑1、Top‑5、F1‑macro、加权 F1 与混淆矩阵进行比较。结果显示 10% 合成比例时类型 F1‑macro 提升 4.4%、朝代 3.9%、窑 3.0%，但釉分类下降 3.5%；5% 合成效果有限。

**⚠️ 局限性**

主要局限在于合成图像纹理细节不足，导致釉分类性能下降；合成比例需达到约 10% 才能显著提升；质量控制流程复杂且成本高，对纹理依赖任务不适用。

---

## 192. STEAD: Robust Provably Secure Linguistic Steganography with Diffusion Language Model

**arXiv ID:** 2601.14778 | [PDF](https://arxiv.org/pdf/2601.14778v1)

**作者:** Yuang Qi `[一作]` (University of Science and Technology of China), Kejiang Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 1606 | [OpenAlex ID](https://openalex.org/A5045121980)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 STEAD，一种利用扩散语言模型、鲁棒位置嵌入、误差校正码和邻域搜索提取的可证明安全且对抗攻击鲁棒的语言隐写方法。

**💡 创新点**

创新点在于将 DLM 的并行采样特性与消息驱动的伪随机采样相结合，识别鲁棒位置并采用重复码纠错，同时设计邻域搜索机制应对插删位移，使得在保持安全性的同时大幅提升鲁棒性。

**🔧 技术方法**

使用的技术包括扩散语言模型（DLM）、伪随机数生成、误差校正编码（重复码）、邻域搜索提取、深度学习隐写分析器以及多种评估指标（PPL、准确率等）。

**📊 数据集**

实验数据集主要包括 IMDB 电影评论集和 C4 文本语料，另外在不同 top‑p 截断设置下评估。

**📈 对比分析**

与 ARM‑based PSARS 及其他稳健隐写方法相比，STEAD 在相同采样参数下嵌入容量更高、鲁棒性更强；在对抗性攻击（代替、插入、删除、同义词替换）下提取成功率显著高于对照组；在 PPL 与检测误差率（≈50%）上与纯随机采样无显著差异。

**⚠️ 局限性**

限制点包括对极高攻击强度（如大量插删或同义词替换）仍会出现提取失败，且实现依赖高质量 DLM 与同步 PRNG，计算开销相对较大。

---

## 193. Secure Communication in MIMOME Movable-Antenna Systems with Statistical Eavesdropper CSI

**arXiv ID:** 2601.14755 | [PDF](https://arxiv.org/pdf/2601.14755v1)

**作者:** Lei Xie `[一作]` (Southeast University), Liquan Chen `[通讯]` (Southeast University)

**通讯引用:** 1888 | [OpenAlex ID](https://openalex.org/A5030223441)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在多输入多输出多天线窃听者（MIMOME）系统中利用可移动天线（MA）提升物理层安全性，并在只拥有接收方完整CSI与窃听者统计CSI的前提下，推导了可平均化的封闭式确定等价式，用于评估和优化平均密钥速率（ESR）。

**💡 创新点**

创新点包括：① 在统计ECSI下首次推导了ESR的随机矩阵理论确定等价式；② 提出了基于MM与AMSGrad的交替优化框架，能够同时优化预编码矩阵与天线位置；③ 通过梯度投影与Armijo搜索实现了可扩展的求解方法。

**🔧 技术方法**

使用技术包括：随机矩阵理论（RMT）求解确定等价；MM（majorization‑minimization）框架处理非凸预编码优化；AMSGrad梯度优化处理天线位置连续优化；梯度投影与线性搜索实现约束满足；Monte‑Carlo仿真验证理论与算法。

**📊 数据集**

数据集：采用28 GHz毫米波仿真环境，随机生成天线阵列、角度、K‑因子、路径损耗等参数，利用多次独立试验（如1000个随机场景）进行实验验证。

**📈 对比分析**

比较方法：将MA+GP（可移动天线+梯度预编码）与三种基线方案（固定天线+ZF、固定天线+GP、固定天线+ZFS）进行对比。结果显示，MA+GP在ESR上提升约4 bps/Hz，显著优于基线，且算法收敛速度快（约5次交替迭代即可稳定）。

**⚠️ 局限性**

限制：仅考虑统计ECSI，未考虑LoS成分的估计误差或统计模型参数的不确定性；理论推导基于大系统极限，虽然在中等规模时仍准确，但在极小规模或极端多径环境下可能出现偏差；算法收敛到局部最优，未给出全局最优保证。

---

## 194. Enhancing Text-to-Image Generation via End-Edge Collaborative Hybrid Super-Resolution

**arXiv ID:** 2601.14741 | [PDF](https://arxiv.org/pdf/2601.14741v1)

**作者:** Chongbin Yi `[一作]` (Huazhong University of Science and Technology), Peng Yang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 11484 | [OpenAlex ID](https://openalex.org/A5089647465)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种端到边缘协同的文本到图像（T2I）生成框架，利用边缘服务器先生成低分辨率图像，并将其按空间方差划分为前景和背景两类，再分别采用扩散式 SR 和轻量学习式 SR 进行增强，最后在用户端拼接得到高分辨率图像；框架中还配合自适应配置选择（使用模拟退火算法）来平衡图像质量与服务时延。

**💡 创新点**

1) 采用空间方差无监督的稀疏划分实现前景/背景分区，避免使用昂贵的语义分割模型；2) 设计前景使用扩散式 SR、背景使用轻量学习式 SR 的区域感知混合 SR 策略，实现高质量细节恢复与低时延并存；3) 将生成与增强过程的超参数（去噪步数、SR 缩放因子）统一纳入可调配置，通过模拟退火搜索最优配置，以适应不同用户需求和边缘资源情况。

**🔧 技术方法**

核心技术包括：Stable Diffusion 3 (SDXL) 作为边缘 T2I 模型；StableSR（扩散式 SR）和 Real‑ESRGAN（学习式 SR）；无监督的稀疏空间方差分区；Simulated Annealing (SA) 用于自适应配置搜索；多尺度 Transformer‑based MUSIQ 评估指标；边缘与终端设备的并行传输与合成。

**📊 数据集**

实验使用 HuggingFace 上的 P2（Parti Prompts）数据集，采集多种文本提示来评估生成效果，并与多种基线方法比较。

**📈 对比分析**

与 Random、w/o SR、OneType、CogView3 等基线方法对比，系统在 720P、1080P、1440P 目标分辨率下，平均服务时延降低约 33%，图像质量（CLIPScore、MUSIQ）仅低于 CogView3 约 1.5% 并且在高分辨率下保持稳定；总体 utility（质量-时延平衡）提升 25% 以上。

**⚠️ 局限性**

局限性包括：① 前景/背景划分仅基于空间方差，可能在纹理复杂或相似区域误判；② 仅在 NVIDIA RTX 3080 Ti 与 Jetson AGX Orin 设备上验证，移动终端算力更低时性能尚待评估；③ 对极高分辨率（> 4K）时的带宽与计算需求仍有限；④ 模拟退火搜索耗时较长，实时动态调节仍需进一步优化。

---

## 195. PULSE: Socially-Aware User Representation Modeling Toward Parameter-Efficient Graph Collaborative Filtering

**arXiv ID:** 2601.14720 | [PDF](https://arxiv.org/pdf/2601.14720v1)

**作者:** Doyun Choi `[一作]` (Korea Advanced Institute of Science and Technology), Jaemin Yoo `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 514 | [OpenAlex ID](https://openalex.org/A5036321139)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出PULSE框架，利用社交信息（用户所属社区与社交邻居互动过的物品）构建用户表示，省去显式用户嵌入；

**💡 创新点**

创新点在于通过社区感知与社交连接物品两种社交信号生成用户表示，并采用自适应融合与对比自监督正则化，实现参数量降低50%以上的高效推荐；

**🔧 技术方法**

技术手段包括图协同过滤（LightGCN）、Leiden社区检测、双向消息传递、社交注意力机制、门控网络融合、InfoNCE自监督；

**📊 数据集**

使用社交推荐常用的三大数据集：Douban‑Book、Yelp、Epinions；

**📈 对比分析**

与13个基线（含GCF与图社交推荐方法）对比，PULSE在三组数据上NDCG@20提升约9.0%、8.1%、8.5%；在冷启动、噪声、不同交互度等场景均优于对手，训练时间与显存亦显著更低；

**⚠️ 局限性**

局限在于仍依赖社交网络质量，对极稀疏或无社交信息的数据表现可能受限；对物品嵌入的规模优化尚未展开；

---

## 196. PCL-Reasoner-V1.5: Advancing Math Reasoning with Offline Reinforcement Learning

**arXiv ID:** 2601.14716 | [PDF](https://arxiv.org/pdf/2601.14716v1)

**作者:** Yao Lu `[一作]` (Peng Cheng Laboratory), Yonghong Tian `[通讯]` (Peking University)

**通讯引用:** 15120 | [OpenAlex ID](https://openalex.org/A5023918894)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了PCL-Reasoner-V1.5，一款32B参数的数学推理大型语言模型，采用先监督微调再离线强化学习的两阶段训练流程。

**💡 创新点**

创新点在于提出并验证了离线强化学习（offline RL）替代传统在线RL（如GRPO）的方案，证明离线RL在训练稳定性、计算效率和工程简化方面具有显著优势，并能显著提升长链推理能力。

**🔧 技术方法**

技术手段包括基于Qwen2.5-32B的SFT、使用CoT推理数据、离线RL训练（基于奖励为正负二值的RL loss）、vLLM高速推理、MindSpeed-LLM并行训练、FP16训练、AdamW优化器及余弦学习率调度等。

**📊 数据集**

使用的数据集包括：SFT阶段的AM‑DeepSeek‑R1‑0528‑Distilled数学子集；离线RL阶段的Nemotron‑Post‑Training‑Dataset‑v1数学子集（筛选出难题）以及AIME 2024/2025评测集。

**📈 对比分析**

在AIME 2024/2025评测集上与多款同类或更大模型进行pass@1对比，PCL-Reasoner-V1.5分别达成90.9%和85.6%，在所有继承自Qwen2.5-32B的模型中取得最高成绩。

**⚠️ 局限性**

局限性在于离线RL受限于预先采集的数据质量，难以自我迭代探索更复杂推理路径；分布不匹配仍是理论风险，且高性能训练仍需昂贵的算力资源。

---

## 197. Unified Multimodal and Multilingual Retrieval via Multi-Task Learning with NLU Integration

**arXiv ID:** 2601.14714 | [PDF](https://arxiv.org/pdf/2601.14714v1)

**作者:** Xinyuan Zhang `[一作]`, Guoquan Zhang `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练了一个统一的多模态多任务检索框架，支持跨语言的图像检索、文本检索与自然语言理解；

**💡 创新点**

首次通过三阶段联合训练将文本-图像、文本-文本检索与NLU任务整合到共享文本编码器中，并利用NLU特征增强查询表示，显著提升多语言检索性能；

**🔧 技术方法**

采用多模态对比学习、跨注意力融合、NLU模块（意图检测+槽位填充）、三阶段渐进式微调、共享文本编码器（LaBSE Vit‑L/14）和图像编码器（CLIP基础）等技术；

**📊 数据集**

使用LAION/WIT、ShareGPT4、xP3/MTP/Miracl、Annotated/Translated NLU数据，以及基于MSCOCO/ MSCOCO‑LONG构建的12语种COCO‑QLTI数据集，并在XTD10、Multi30K等公开数据集上进行评测；

**📈 对比分析**

与SigLIP2、Jina‑Clip‑v2、XLM‑R等多语言基线对比，在XTD10和Multi30K的T2I检索中平均提升约1.1%（R@10 93.3/94.8 vs 92.2/92.1），在COCO‑QLTI的T2T检索中平均提升至83.4% vs 82.8%，NLU意图检测准确率>96%，槽位F1>88%；

**⚠️ 局限性**

模型仍需依赖大规模预训练模型，训练和推理成本较高，对低资源语言与实时推理的适应性尚未充分评估，部署规模大导致资源消耗大。

---

## 198. Proximal Policy Optimization with Evolutionary Mutations

**arXiv ID:** 2601.14705 | [PDF](https://arxiv.org/pdf/2601.14705v1)

**作者:** Casimir Czworkowski `[一作]` (Johns Hopkins University), Alhassan S. Yasin `[通讯]` (Johns Hopkins University)

**通讯引用:** 120 | [OpenAlex ID](https://openalex.org/A5033362818)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了一种在 PPO 中加入基于 KL 散度触发的进化突变机制的强化学习算法 POEM，并在四个 OpenAI Gym 控制任务上进行实验。

**💡 创新点**

通过监测当前策略与移动平均策略的 KL 散度，当多样性低于阈值时自适应注入噪声突变，从而提升探索能力。

**🔧 技术方法**

结合 PPO 的裁剪目标、熵正则化、KL 散度多样性奖励以及自适应进化突变，并使用 Optuna 对超参数进行优化。

**📊 数据集**

四个 OpenAI Gym 环境——CarRacing-v3、MountainCarContinuous-v0、BipedalWalker-v3 与 LunarLander-v3。

**📈 对比分析**

采用 10 次固定种子实验，并用 Welch t 检验比较，POEM 在 CarRacing、MountainCar、BipedalWalker 上显著优于 PPO（p<0.05），在 LunarLander 上差异不显著。

**⚠️ 局限性**

在资源受限任务（如 LunarLander）表现差异不显著，且仅在连续动作空间下验证，尚未探究离散动作空间或更高维环境的适用性。

---

## 199. DARL: Encouraging Diverse Answers for General Reasoning without Verifiers

**arXiv ID:** 2601.14700 | [PDF](https://arxiv.org/pdf/2601.14700v1)

**作者:** Chongxuan Huang `[一作]` (Xiamen University), Ruiming Tang `[通讯]` (Kuaishou Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DARL（Diversified Answer RL）框架，在无验证器的强化学习中鼓励模型在参考答案的可接受范围内生成多样化答案，并显著提升推理与通用任务性能。

**💡 创新点**

创新点在于：① 采用动态多样性奖励，依据模型对参考答案的置信度自动扩展/缩小多样性阈值；② 在保持答案一致性的前提下，系统性地鼓励答案多样性；③ 与现有 RLPR 等方法兼容，可直接插拔。

**🔧 技术方法**

技术要点包括：基于模型内部信号（如 token 似然）的奖励；动态阈值机制；强化学习中的策略熵监控；在 Llama3.1、Qwen2.5 等大模型上实现。

**📊 数据集**

使用 WebInstruct 数据集（约77.7K 推理实例）进行训练，并在 13 个基准（MMLU‑Pro、GPQA、TheoremQA、MATH‑500、Minerva、AIME24、AutoLogic‑cn、AutoLogic‑en、ZebraLogic、LiveCodeBench、HumanEval、HumanEval+、WritingBench）评估。

**📈 对比分析**

与 RLPR、VeriFree、TTRL 等基线对比，DARL 在 6 个推理基准平均提升 1.3 分，在 7 个通用基准平均提升 9.5 分，且在写作等开放式任务上显著提升答案多样性，整体性能优于或持平现有方法。

**⚠️ 局限性**

局限性：依赖参考答案作为监督，无法在无答案的场景（如纯查询）下使用；对未包含多样化答案的训练数据可能导致多样性提升受限。

---

## 200. FSX: Message Flow Sensitivity Enhanced Structural Explainer for Graph Neural Networks

**arXiv ID:** 2601.14730 | [PDF](https://arxiv.org/pdf/2601.14730v1)

**作者:** Bizu Feng `[一作]` (Fudan University), Zixin Hu `[通讯]` (Fudan University)

**通讯引用:** 2419 | [OpenAlex ID](https://openalex.org/A5101743291)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新型的混合图神经网络可解释框架FSX，结合内部消息流与外部图结构给出高效、可信的子图解释。

**💡 创新点**

创新点在于通过单次前向传递的局部消息流灵敏度分析定位关键信息流，并将其映射到图中构造子图，再在子图上设计流感知合作博弈，用加权Shapley值分配节点贡献。

**🔧 技术方法**

使用了消息流灵敏度分析、梯度与局部扰动、合作博弈论、Shapley值近似、PyTorch+PyTorch‑Geometric等技术。

**📊 数据集**

在四个公开基准（BBBP、ClinTox、Graph‑SST2、Graph‑Twitter）上进行实验。

**📈 对比分析**

与FlowX、GNNExplainer、GraphEXT、PGExplainer等方法对比，FSX在Fidelity+、Fidelity-和稀疏度上表现更好，同时解释时间显著缩短。

**⚠️ 局限性**

局限在于仍需在更大规模或动态图任务中验证，可进一步优化敏感度采样与子图选择的参数。

---

## 201. Reconstruction-Anchored Diffusion Model for Text-to-Motion Generation

**arXiv ID:** 2601.14788 | [PDF](https://arxiv.org/pdf/2601.14788v1)

**作者:** Yifei Liu `[一作]` (South China University of Technology), Qiong Cao `[通讯]` (JD Joy Future Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了基于运动重构的扩散框架，利用运动潜在空间作为中间监督实现文本到动作的生成。

**💡 创新点**

创新点在于：①引入运动潜在自正则化与运动中心对齐构建运动专属潜在空间；②提出重构误差引导（REG）在采样阶段对前一步误差进行重构并补偿，以抑制错误传播。

**🔧 技术方法**

使用了双流（运动重构+文本生成）架构、Transformer 运动编码器与文本编码器、扩散模型、潜在空间自正则化、运动中心对齐、REG 与分类器无监督引导。

**📊 数据集**

主要使用了 HumanML3D 与 KIT‑ML 两大文本‑动作数据集。

**📈 对比分析**

通过与最新扩散模型（如Salad、MDM）及 VQ‑VAE 方法（如MoMask、T2M‑GPT）在 HumanML3D 上的 R‑Precision（56.1%）与 FID（0.032）进行对比，取得 SOTA；在 KIT‑ML 也位列前两名。

**⚠️ 局限性**

限制包括：需额外重构步骤导致推理时间略长；对噪声/低质量数据的鲁棒性尚未充分验证；潜在空间设计对不同数据集的泛化仍待进一步探索。

---

## 202. PAColorHolo: A Perceptually-Aware Color Management Framework for Holographic Displays

**arXiv ID:** 2601.14766 | [PDF](https://arxiv.org/pdf/2601.14766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 203. FunCineForge: A Unified Dataset Toolkit and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes

**arXiv ID:** 2601.14777 | [PDF](https://arxiv.org/pdf/2601.14777v1)

**作者:** Jiaxuan Liu `[一作]` (Alibaba Group), Zhenhua Ling `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 FunCineForge 端到端配音数据集构建管线与 MLLM 语音合成模型，并构建了首个中文电视配音数据集 CineDub‑CN。

**💡 创新点**

创新点包括：① 基于多模态 Chain‑of‑Thought 的自动校正管线；② 结合时间‑说话人标注的多模态对齐机制；③ 在流匹配中引入说话人切换拼接；④ 支持复杂多说话人与动态摄像场景。

**🔧 技术方法**

技术栈涵盖：Gemini‑2.5 MLLM、FAN/Res2Net 视觉/音频特征提取、Mel‑RoFormer 音频分离、TalkNet‑ASD 视觉增强说话人识别、时间‑说话人 Tokenizer、Lip‑Speech 对比学习、DiT 流匹配、HiFiGAN 声码器。

**📊 数据集**

主要使用自建的 4,700 小时中文电视配音数据集 CineDub‑CN，并与 V2C‑Animation、Chem、GRID 三大英文配音数据集进行对比实验。

**📈 对比分析**

通过 MCD‑DTW、LSE‑D、SPK‑TL、SPK‑SIM、EMO‑SIM、ES‑MOS 等客观与主观指标评估，FunCineForge 在多场景（独白、旁白、对话、多说话人）中均优于现有 SOTA 方法，表现出更佳的音质、对齐、说话人相似度与情感一致性。

**⚠️ 局限性**

局限性包括：仅针对中文电视场景，未覆盖多语言；对极端遮挡或多声道混合场景鲁棒性待提升；实现对大型 MLLM 的依赖使训练成本和算力需求较高。

---

## 204. Does medical specialization of VLMs enhance discriminative power?: A comprehensive investigation through feature distribution analysis

**arXiv ID:** 2601.14774 | [PDF](https://arxiv.org/pdf/2601.14774v1)

**作者:** Keita Takeda `[一作]` (Nagasaki University), Tomoya Sakai `[通讯]` (Nagasaki University)

**通讯引用:** 660 | [OpenAlex ID](https://openalex.org/A5068376487)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

对比分析医学与非医学视觉‑语言模型（VLM）的图像特征分布，评估医学专业化对特征判别力的影响。

**💡 创新点**

首次系统性地将高维特征降维可视化与线性判别相结合，揭示医学专业化不一定提升图像编码，且文本编码增强更为关键。

**🔧 技术方法**

使用CLIP、LLaVA、LLM2CLIP等VLM；通过UMAP可视化特征分布；构造线性SVM判别器和RISE注意力映射；对比医学与非医学模型。

**📊 数据集**

八种医学影像分类数据集：脑MRI、CT、X‑ray、超声、病理、细胞学、眼科、皮肤，及附加的肺炎、乳腺US等。

**📈 对比分析**

在各数据集上用线性SVM和注意力评估，发现非医学VLM加LLM文本编码在多数任务上匹配或优于医学专用模型；医学专业化模型在细粒度任务上有时表现更差。

**⚠️ 局限性**

仅关注二维图像特征，未评估3D、WSI、多模态任务；使用官方默认预处理，未处理医学图像的高动态范围；实验环境统一困难，部分模型无法加载。

---

## 205. Using Multi-Instance Learning to Identify Unique Polyps in Colon Capsule Endoscopy Images

**arXiv ID:** 2601.14771 | [PDF](https://arxiv.org/pdf/2601.14771v1)

**作者:** Puneet Sharma `[一作]` (UiT Arctic University of Norway), Ulrik Deding `[通讯]` (University of Southern Denmark)

**通讯引用:** 629 | [OpenAlex ID](https://openalex.org/A5064714464)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文将利用多实例学习（MIL）框架，通过将查询息肉图像与目标图像集合进行比较，来识别结肠胶囊内镜（CCE）图像中的独特息肉。

**💡 创新点**

创新点在于将MIL的多实例验证（MIV）方法与注意力机制（VEMA、DBA）相结合，并将自监督学习框架SimCLR预训练的嵌入融入MIV，从而显著提升识别精度。

**🔧 技术方法**

核心技术包括：多实例验证Siamese网络、变异激励多头注意力（VEMA）、基于距离的注意力（DBA）、自监督对比学习SimCLR以及多种卷积/视觉Transformer骨干网络（EfficientNet、ResNet、ConvNeXt、ViT）。

**📊 数据集**

使用CareForColon2015数据集，包含1912个独特息肉的五张图像（共754名患者），每个息肉的五张图像分别为不同视角。

**📈 对比分析**

通过10折患者级交叉验证与20%独立测试集，对比无注意力（均值/最大池化）、VEMA、DBA（L1/L2）以及SimCLR预训练模型。最佳结果是ConvNeXt+SimCLR+DBA‑L1（2头），测试准确率86.26%，AUC 0.928；无注意力模型约74–82%准确率，AUC 0.82–0.91。

**⚠️ 局限性**

局限性包括：数据量有限且仅涵盖息肉，无法覆盖所有临床变异；胶囊摄像头双头视角差异导致查询与目标图像差异大，易产生误判；缺乏真实的合成或增强数据来进一步提升泛化能力。

---

## 206. Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning

**arXiv ID:** 2601.14750 | [PDF](https://arxiv.org/pdf/2601.14750v1)

**作者:** Yifan Wang `[一作]` (Tsinghua University), Zheng Wei `[通讯]` (Tencent)

**通讯引用:** 776 | [OpenAlex ID](https://openalex.org/A5060067799)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Render-of-Thought 框架，将链式思路的文本步骤渲染为图像，以压缩推理过程并实现可视化。

**💡 创新点**

创新点在于利用预训练视觉编码器作为语义锚点，将 LLM 隐藏状态映射到视觉空间，并通过两阶段训练实现可观测的 latent reasoning。

**🔧 技术方法**

采用视觉渲染单行图像、Vision Language Model、MLP 投影头、LoRA 微调、两阶段对齐与自回归生成等技术。

**📊 数据集**

使用 GSM8k-Aug、GSM-Hard、SVAMP、MultiArith、MATH 等数学与逻辑推理基准数据集。

**📈 对比分析**

与显式 CoT、隐式 Latent CoT 及 LLM 基线对比，保持相近准确率的同时实现 3-4× token 压缩、推理加速；在 MATH 上从 291→64 token、准确率从 55.8%→33.2%。

**⚠️ 局限性**

局限性包括仅验证英文数学/逻辑任务、需手动设置 token 预算、动态终止不稳定、训练成本相对较高、未评估多语言或其他推理领域。

---

## 207. RefProtoFL: Communication-Efficient Federated Learning via External-Referenced Prototype Alignment

**arXiv ID:** 2601.14746 | [PDF](https://arxiv.org/pdf/2601.14746v1)

**作者:** Hongyue Wu `[一作]` (Tianjin University), Zhiyong Feng `[通讯]` (Tianjin University)

**通讯引用:** 11319 | [OpenAlex ID](https://openalex.org/A5001714538)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种融合外部引用原型对齐（ERPA）与自适应参数丢弃（APUD）的联邦学习框架RefProtoFL；

**💡 创新点**

通过外部公共数据构建全局原型并对齐本地原型，实现表示一致性；同时采用幅度感知Top‑K稀疏化，仅上传重要的adapter参数，显著降低通信成本；

**🔧 技术方法**

使用服务器持有的少量公共数据构建外部原型、adapter+backbone模型分离、幅度感知Top‑K稀疏化的自适应参数丢弃以及传统的交叉熵+原型对齐损失；

**📊 数据集**

在CIFAR‑10、CIFAR‑100、FashionMNIST与MNIST四个图像分类基准上进行实验；

**📈 对比分析**

与FedAvg、FedProx、FedNova、FedProto、VHL、FedGH、FAST等基线对比，平均Top‑1准确率达到60.63%，在高异质（α=0.5）场景下优于所有对比方法；

**⚠️ 局限性**

对公共参考数据的依赖显著，若缺失则对齐效果下降；仅在分类任务上验证，尚未在更复杂任务或真实设备环境中评估。

---

## 208. Safeguarding Facial Identity against Diffusion-based Face Swapping via Cascading Pathway Disruption

**arXiv ID:** 2601.14738 | [PDF](https://arxiv.org/pdf/2601.14738v1)

**作者:** Liqin Wang `[一作]` (Sun Yat-sen University), Xiangyang Luo `[通讯]` (Zhengzhou University)

**通讯引用:** 4413 | [OpenAlex ID](https://openalex.org/A5090509213)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种面向扩散模型的主动防御方法VoidFace，通过在检测、特征提取与生成三阶段的身份通路上注入扰动，实现身份信息的级联破坏；

**💡 创新点**

将面部替换视为耦合身份通路并提出级联破坏策略；在物理、语义与生成域分别设计定位破坏、身份消除、注意力解耦、特征破坏等多维度对抗；在扩散潜在空间使用感知自适应优化，兼顾攻击强度与视觉自然；

**🔧 技术方法**

利用扩散模型潜在空间对抗搜索、检测器（MTCNN/RetinaFace）与身份编码器（ArcFace等）的对抗损失、交叉注意力矩阵扰动、U‑Net中间特征破坏、LPIPS感知自适应权重；

**📊 数据集**

CelebA‑HQ与VGGFace2‑HQ两大高分辨率人脸数据集；

**📈 对比分析**

与AdvDM、PhotoGuard、Mist、SDST、FACELOCK、FaceShield等基线在DiffFace、DiffSwap、Face‑Adapter、InstantID四种扩散面部替换模型上对比，VoidFace在ISM、L2、PSNR等指标显著优于基线，且在LPIPS/PSNR/FID上表现更佳，展示了更强的防御效果与更自然的视觉；

**⚠️ 局限性**

对极端后处理（大幅压缩、裁剪）仍可能出现性能下降；跨架构迁移至GAN模型需要进一步提升；对动态视频或多模态场景的适用性尚未验证。

---

## 209. ARFT-Transformer: Modeling Metric Dependencies for Cross-Project Aging-Related Bug Prediction

**arXiv ID:** 2601.14731 | [PDF](https://arxiv.org/pdf/2601.14731v1)

**作者:** Shuning Ge `[一作]` (Capital Normal University), Zheng Zheng `[通讯]` (Beihang University)

**通讯引用:** 77820 | [OpenAlex ID](https://openalex.org/A5100423704)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Transformer的跨项目软件老化相关缺陷预测框架ARFT-Transformer，利用指标间交互捕获信息并通过Focal Loss聚焦难分样本，解决指标相关性与类别不平衡问题。

**💡 创新点**

①在指标层面引入多头注意力机制，首次对软件度量之间的相关性进行建模；②结合MMD对源域与目标域表示进行对齐；③采用Focal Loss动态减弱易分类样本的损失，提升对稀有缺陷样本的学习。

**🔧 技术方法**

FT‑Transformer架构（tokenizer + 多头注意力 + 线性分类头）、MMD距离损失、随机过采样（ROS）以及Focal Loss。

**📊 数据集**

在三大开源项目（MySQL、Linux、Apache HTTPD）上使用52种静态度量（程序规模、McCabe、Halstead、Aging‑Related Metrics）构建的数据集。

**📈 对比分析**

与六种主流跨项目缺陷预测方法（TLAP、SRLA、JDA‑ISDA、JPKS、KDK、BISP）以及传统特征选择方法进行对比；单源场景下平均提升约12%（最高提升24%），多源场景下平均提升约12%；在所有实验组均显著优于对比方法。

**⚠️ 局限性**

实验仅覆盖三大项目，缺乏对更多语言或规模的验证；超参数选择仍依赖经验；对真实工业项目的迁移性能尚未深入评估。

---

## 210. Talk Me Through It: Developing Effective Systems for Chart Authoring

**arXiv ID:** 2601.14707 | [PDF](https://arxiv.org/pdf/2601.14707v1)

**作者:** Nazar Ponochevnyi `[一作]` (University of Toronto), Anastasia Kuzminykh `[通讯]` (University of Toronto)

**通讯引用:** 417 | [OpenAlex ID](https://openalex.org/A5090252428)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统地比较了口语与文字输入在图表创作指令中的结构差异，并验证基于口语想象图表数据训练的系统在语音和文字交互中的性能提升；

**💡 创新点**

创新点在于①首次对比口语想象图表指令、文字想象图表指令与传统已有图表指令的结构差异；②证明口语想象图表数据能够显著提升系统在语音输入上的效果；③公开口语与文字想象图表指令数据集，为多模态图表生成研究提供资源；

**🔧 技术方法**

采用GPT‑3.5 Turbo进行参数高效微调，生成Plotly绘图代码；音频转写使用Conformer‑1 API；实验环境通过Gradio搭建；

**📊 数据集**

使用的主要数据集包括：①从用户访谈中收集的76条口语想象图表指令（转录后）；②65条文字想象图表指令；③200条来自NLV Corpus的文字已有图表指令；④200条来自nvBench的合成文字已有图表指令；并公开了口语/文字想象图表指令数据集；

**📈 对比分析**

通过构建两套系统（口语训练vs文字训练），在受控实验中分别用口语和文字输入评估，统计错误类型和用户满意度，并采用Fisher exact test检验差异。结果显示：口语训练系统在口语输入上显著优于文字训练系统（p=0.037），在文字输入上无显著差异；总体而言，口语训练系统在两种输入下表现更佳；

**⚠️ 局限性**

局限性包括：样本量相对较小；实验设置为受导向的受控情境，缺乏多轮真实交互、多模态组合及团队协作的评估；数据主要为英文，缺乏多语言覆盖，未来需扩大数据规模并在更自然的使用环境中验证。

---

## 211. RegFreeNet: A Registration-Free Network for CBCT-based 3D Dental Implant Planning

**arXiv ID:** 2601.14703 | [PDF](https://arxiv.org/pdf/2601.14703v1)

**作者:** Xinquan Yang `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (University of Nottingham Ningbo China)

**通讯引用:** 10908 | [OpenAlex ID](https://openalex.org/A5019313200)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种注册‑自由的 3D CBCT 牙种植位置预测网络 RegFreeNet，并公开了包含 1622 份 CBCT 数据的 ImplantFairy 公开数据集。

**💡 创新点**

创新点包括：① 通过在术后 CBCT 中遮蔽植入物实现注册‑自由训练；② 设计邻域距离感知 (NDP) 模块，融合多尺度卷积与图卷积来捕捉牙齿空间层次；③ 双分支结构，单独预测位置和斜率，以斜率监督约束位置预测。

**🔧 技术方法**

技术手段包括 3D U‑Net 编码器、三重膨胀卷积、图卷积网络、双分支回归、随机遮蔽、TTA 滑动窗口推理、以及基于交叉熵+Dice 与 L1 损失的联合训练。

**📊 数据集**

使用数据集：自建的 ImplantFairy（1622 份 CBCT），外部公开数据集（Cui 及 ToothFairy2 共 12 份）做泛化验证。

**📈 对比分析**

与 9 种 CNN/Transformer 分割模型（如 3DUnet、UNETR、SwinUNETR 等）进行对比，RegFreeNet 在 SUGH 数据集上取得 Dice≈47.5%、IoU≈0.355，外部数据集 Dice≈33%、IoU≈0.241，均优于对比模型。

**⚠️ 局限性**

局限性：仍依赖手工标注的植入物轴线；遮蔽策略可能在极端解剖变形下失效；在更大规模或不同 CBCT 设备上泛化仍需进一步验证。

---

## 212. Many-to-many. Usability challenges of entity reconciliation in art history and photographic studies

**arXiv ID:** 2601.14753 | [PDF](https://arxiv.org/pdf/2601.14753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 213. ARISE -- Adaptive Refinement and Iterative Scenario Engineering

**arXiv ID:** 2601.14743 | [PDF](https://arxiv.org/pdf/2601.14743v1)

**作者:** Konstantin Poddubnyy `[一作]` (German Research Center for Artificial Intelligence), Philipp Slusallek `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了ARISE工具，通过LLM驱动的迭代测试与修复循环，将自然语言交通场景描述自动转换为可执行的Scenic脚本。

**💡 创新点**

其创新点在于引入自动化的Test‑and‑Repair Loop (TRL) 与细粒度语义提取、扩展知识库，支持多轮LLM修复与执行验证，大幅提升脚本可执行率。

**🔧 技术方法**

采用大型语言模型（GPT‑4o、Gemini 2.0 Flash、DeepSeek‑V3）、句子嵌入检索、Scenic编译器与CARLA模拟器，并将结构化诊断反馈回馈给LLM。

**📊 数据集**

使用基于ChatScene的Scenic案例库与自行扩充的知识对，并以40个多样化自然语言场景提示（8类，每类5例）进行评估。

**📈 对比分析**

通过与ChatScene基线（无TRL）和仅改进TRL的版本对比，采用Execution Success Rate (ESR)、Repair Convergence Rate (RCR) 与 Semantic Conformity Score (SCS) 三指标，ARISE的ESR达78%，显著高于基线1.4%，RCR与SCS也优于对比方法。

**⚠️ 局限性**

主要限制在于仍需依赖LLM生成脚本，温度和迭代次数上限会影响鲁棒性；目前仅支持Scenic 3.0，未涵盖OpenSCENARIO；评估仅针对CARLA与SafeBench，缺乏更广泛场景与模拟器的验证。

---

## 214. Anytime Optimal Decision Tree Learning with Continuous Features

**arXiv ID:** 2601.14765 | [PDF](https://arxiv.org/pdf/2601.14765v1)

**作者:** Harold Kiossou `[一作]` (UCLouvain), Siegfried Nijssen `[通讯]` (KU Leuven)

**通讯引用:** 3506 | [OpenAlex ID](https://openalex.org/A5046889112)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于有限偏差搜索的连续特征决策树学习算法（CA‑ConTree）。

**💡 创新点**

通过在特征和阈值层面引入有限偏差预算，兼顾完整性与 anytime 性能，显著提升搜索效率。

**🔧 技术方法**

结合有限偏差搜索（LDS）、启发式特征排序、深度二分子程序（D2Split）、相似性下界（SLB）与缓存技术。

**📊 数据集**

在 16 个 UCI 公开数据集（如 skin、avila、magic、bean、htru 等）上进行实验。

**📈 对比分析**

与 ConTree（含 Gini/不含）以及贪心 C4.5 进行对比，使用平均原始积分和交叉验证准确率评估，CA‑ConTree 在时间受限下显著降低原始积分并在多数数据集上获得最高或相近的准确率。

**⚠️ 局限性**

在极深树或极小数据集上仍需较长时间才能证明最优，且与 C4.5 的差距在某些情形下有限。

---

## 215. An XAI View on Explainable ASP: Methods, Systems, and Perspectives

**arXiv ID:** 2601.14764 | [PDF](https://arxiv.org/pdf/2601.14764v1)

**作者:** Thomas Eiter `[一作]` (Institute of Logic and Computation, TU Wien), Zeynep G. Saribatur `[通讯]` (Institute of Logic and Computation, TU Wien)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了回答集编程（ASP）中的可解释性方法，按 XAI 视角将其分为局部与全局解释，归纳了用户可能提出的问题，并对现有技术、工具及其支持的语言特性、输出形式进行了系统分类和对比；同时指出了目前工具与技术在语言支持、可扩展性、用户交互等方面的不足，并提出了未来研究方向。

**💡 创新点**

① 将 ASP 解释方法映射到 XAI 的局部/全局范畴，构建了从用户提问到解释技术的完整对照表；② 通过整合最新工具（如 xASP2、s(CASP)、xclingo 等）与已有研究，形成了针对 ASP 的统一解释框架；③ 提出了“解释缺口”概念，明确了现有技术在语言扩展、可解释性深度与交互体验等方面的瓶颈。

**🔧 技术方法**

主要采用文献综述与分类法（如局部/全局、用户问题导向）进行系统梳理；对比分析工具的支持特性（离散、约束、聚合、弱约束等）；对现有解释算法（如离线证明、ABA、因果、1-PUS、支持图、证据树、证据日志、抽象等）进行技术归纳；同时使用实例程序展示每种方法的工作原理。

**📊 数据集**

无专门数据集；本研究基于已有文献、公开工具示例与案例程序（如 Donkey 示例、P_1–P_4 等）进行说明，未进行实验数据收集。

**📈 对比分析**

本工作没有进行数值性能评测；对比主要以技术特性（支持的 ASP 语言扩展、是否支持非正面/负面解释、求解器依赖、输出格式等）和理论覆盖度（能回答哪些类型问题）为准；对工具的功能完整性进行整理，指出哪些工具支持哪些特性，但未给出运行时间或资源消耗等实验结果。

**⚠️ 局限性**

① 现有工具普遍缺乏对分离语句（disjunction）、弱约束、复杂聚合等高级 ASP 语言特性的支持；② 解释方法大多针对单一答案集或单一问题，缺乏对多答案集、数量级解释的能力；③ 缺乏可扩展至大型非地面程序的高效实现；④ 交互体验不友好，普通用户难以直接获取解释；⑤ 解释的可读性、简洁性和对非专业用户的可理解性仍待改进；⑥ 许多技术只在学术研究或实验平台上实现，缺少成熟工业级工具。

---

## 216. SimD3: A Synthetic drone Dataset with Payload and Bird Distractor Modeling for Robust Detection

**arXiv ID:** 2601.14742 | [PDF](https://arxiv.org/pdf/2601.14742v1)

**作者:** Ami Pandat `[一作]` (Homi Bhabha National Institute), Rohit Shukla `[通讯]` (Homi Bhabha National Institute)

**通讯引用:** 315 | [OpenAlex ID](https://openalex.org/A5101940620)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SimD3合成无人机数据集，并在YOLOv5上进行实验

**💡 创新点**

该数据集在无人机载荷、鸟类干扰和恶劣天气方面做了显著的细粒度建模；模型通过将C3模块改为C3b并加入CBAM实现了注意力增强

**🔧 技术方法**

使用Unreal Engine 5、AirSim、Blender、Meshy AI生成场景与3D模型；YOLOv5基础网络与改进版Yolov5m+C3b；训练与评估使用标准的mAP、Precision/Recall指标

**📊 数据集**

SimD3（178,639张图，包含无人机、鸟类和多天气条件）以及与DUT‑AntiUAV真实数据混合用于评估

**📈 对比分析**

在SimD3上，Yolov5m+C3b比基线提升约1.4% mAP@0.5和4.1% mAP@0.5:0.95；混合数据集时跨域测试mAP@0.5:0.95从0.386提升至0.499；在未见真实数据集上仍保持明显优势，说明泛化能力提升

**⚠️ 局限性**

仍受域差异限制，尤其在极端小尺寸目标和高噪声天气下表现下降；数据集仅覆盖固定数量的无人机与鸟类类型，未覆盖更复杂多模态输入

---

## 217. Typhoon OCR: Open Vision-Language Model For Thai Document Extraction

**arXiv ID:** 2601.14722 | [PDF](https://arxiv.org/pdf/2601.14722v1)

**作者:** Surapon Nonesung `[一作]` (SCB 10X), Kunat Pipatanakul `[通讯]` (SCB 10X)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了针对泰语和英语文档的开放式视觉语言模型 Typhoon OCR，支持文本识别、布局重建与结构化输出。

**💡 创新点**

通过双模式数据构建、合成数据增强与专门的泰语数据集，针对低资源脚本的结构化文档理解进行微调。

**🔧 技术方法**

采用 Qwen2.5‑VL / Qwen3‑VL 背景模型，full‑parameter SFT、长上下文处理、自动化多阶段注释与量化训练。

**📊 数据集**

结合真实泰语文档、CoSyn‑400K 结构化合成、手写集、DocLayNet、VQA 及自制泰语合成语料，总计约 155k–77k 样本。

**📈 对比分析**

以 BLEU/ROUGE‑L/Levenshtein 在泰语财报、表单、书籍等基准上对比 GPT‑4o、Gemini 等专有模型，Typhoon OCR 在结构化任务上均高于基线，V1.5 在 2B 参数下匹配甚至超过更大模型。

**⚠️ 局限性**

对低分辨率、模糊或遮挡图像鲁棒性不足；对高度异构图形（如信息图、手写混合）表现仍差；仅覆盖泰语/英语，尚缺高级推理与多语言扩展。

---

## 218. LookBench: A Live and Holistic Open Benchmark for Fashion Image Retrieval

**arXiv ID:** 2601.14706 | [PDF](https://arxiv.org/pdf/2601.14706v1)

**作者:** Chao Gao `[一作]` (Gensmo.ai), Fan Zhou `[通讯]` (Gensmo.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向时尚商品图像检索的实时、全局性基准——LookBench，并基于该基准开发了轻量化文本‑无检索模型GR‑Pro与GR‑Lite；

**💡 创新点**

创新点在于：①基准持续更新、时间戳标注、污染感知；②引入细粒度属性词典并进行全局属性监督；③同时支持单品检索与多品 outfit 级检索；④公开完整基准数据与模型，形成可持续、可验证的实验平台；

**🔧 技术方法**

使用的技术包括：CLIP、SigLIP 等通用 VLM；自监督 DINOv3 视觉编码器；ArcFace 损失与 Partial FC 进行大规模分类训练；GPT‑5.1 与 Qwen‑VL 进行弱监督属性标注与质量审核；以及 AutoAugment、Mixup、CutMix 等数据增强；

**📊 数据集**

数据集方面：1）LookBench 四个子集（RealStudioFlat、RealStreetLook、AIGen‑Studio、AIGen‑StreetLook）共约 2.5K 查询与 60K+ 语料；2）Fashion200K 作为干扰样本与基准评估；3）训练集 6.5M 图像，来源于公开数据（DeepFashion、Kream、OnTheLook 等）和 1.5M 家庭内部产品；

**📈 对比分析**

评估方法采用属性感知的 Fine Recall@1、Coarse Recall@1 与 nDCG@5；与现有 VLM、Vision‑only、Fashion‑fine‑tuned 模型（CLIP、SigLIP、Marqo‑FashionCLIP 等）对比；GR‑Pro 在所有四个子集的整体 Fine Recall@1 取得 67.38%（GR‑Lite 65.71%），比最强公开基线高约 4%；在 Legacy Fashion200K 上同样取得 88% 以上 Recall@1，保持领先；

**⚠️ 局限性**

局限性：①对部分类别（T‑shirt、sweatshirt、coat）仍表现不佳；②仅覆盖图像‑图像检索，未覆盖多模态或意图条件查询；③属性标注为弱监督，仍有误差；④当前基准对 5% 以上样本的动态更新尚未完成，易受时间漂移影响；

---

## 219. Semantic-Guided Unsupervised Video Summarization

**arXiv ID:** 2601.14773 | [PDF](https://arxiv.org/pdf/2601.14773v1)

**作者:** Haizhou Liu `[一作]` (University of Shanghai for Science and Technology), Hui Yu `[通讯]` (University of Glasgow)

**通讯引用:** 12958 | [OpenAlex ID](https://openalex.org/A5006580423)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于帧级语义对齐注意力的无监督视频摘要框架，利用视觉与CLIP语义特征的余弦相似度来指导关键帧选择，并通过GAN生成逼真摘要。

**💡 创新点**

核心创新在于：①将跨模态相似度融入关键帧重要性评分，实现语义引导；②引入增量式训练策略稳定GAN学习；③使用Transformer生成器实现多模态信息的细粒度重建。

**🔧 技术方法**

技术手段包括CNN+CLIP特征提取、帧级语义对齐注意力（FSSA）、Transformer生成器、GAN判别器，以及基于重构与稀疏性的联合损失和增量式优化。

**📊 数据集**

在公开视频摘要基准数据集SumMe和TVSum上进行评估。

**📈 对比分析**

在SumMe上平均F1得分60.6%，在TVSum上65.3%，均高于现有无监督方法，显示出显著的性能提升。

**⚠️ 局限性**

局限性：仅利用视觉与文本模态，未考虑音频等其他模态；对极长视频的鲁棒性和实时性尚未充分验证。

---

## 220. ReinPath: A Multimodal Reinforcement Learning Approach for Pathology

**arXiv ID:** 2601.14757 | [PDF](https://arxiv.org/pdf/2601.14757v1)

**作者:** Kangcheng Zhou `[一作]` (East China Normal University), Shugong Xu `[通讯]` (Shanghai University)

**通讯引用:** 7405 | [OpenAlex ID](https://openalex.org/A5084521046)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了高质量的多模态病理视觉问答数据集ReinPathVQA，并基于该数据集训练了强化学习多模态大语言模型ReinPath。

**💡 创新点**

首次将格式、准确性、语义相似度三种奖励函数与Group Relative Policy Optimization结合，并通过细粒度推理链注释显著提升模型可解释性和性能。

**🔧 技术方法**

采用双图像编码器（CONCH+UNI）、LLM Llama3-8B、LoRA微调、GRPO强化学习、语义奖励与投影器对齐等技术。

**📊 数据集**

使用自研ReinPathVQA（721k预训练+13k指令+17k强化学习）以及公开的Quilt‑VQA、Path‑VQA、PMC‑VQA、PathMMU、WSSS4LUAD等评测集。

**📈 对比分析**

在四个公开VQA数据集上与现有基线相比，ReinPath在Quilt‑VQA和PMC‑VQA上分别提升5.2%和3.5%；在零样本分类上在WSSS4LUAD上略优于最佳CLIP；整体准确率48.9%接近SOTA，仅使用22k训练样本。

**⚠️ 局限性**

受限于图像特征提取的表达能力、奖励设计的偏差以及对大规模数据的通用性验证不足；模型在极端推理深度或多病变场景下的鲁棒性尚待评估。

---

## 221. Unlocking Large Audio-Language Models for Interactive Language Learning

**arXiv ID:** 2601.14744 | [PDF](https://arxiv.org/pdf/2601.14744v1)

**作者:** Hongfu Liu `[一作]` (National University of Singapore), Ye Wang `[通讯]` (National University of Singapore)

**通讯引用:** 10700 | [OpenAlex ID](https://openalex.org/A5100423435)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了针对二语读音训练的聊天式语音评估系统

**💡 创新点**

首创L2-Arctic-plus数据集，结合错误说明与可操作建议，并通过指令微调提升音频语言模型性能

**🔧 技术方法**

采用音频语言模型（ALM）+大型语言模型（Llama3.1/ Mistral），并使用LoRA指令微调及音频编码器

**📊 数据集**

使用L2-Arctic-plus（900样本）以及CommonVoice 200k音频文本对进行两阶段微调

**📈 对比分析**

与传统ASR+LLM级联及现有开源ALM（Qwen-Audio、Qwen2-Audio）对比，指令微调后模型在误音识别F1提升高达134%，并在建议生成指标（BLEU-2/ROUGE-L/BERTScore）显著优于基线，甚至在部分指标超过GPT‑4o‑Audio

**⚠️ 局限性**

目前仅支持读音朗读情境，缺乏对话式场景；反馈仅文本形式，缺乏音频或示范音频，模型仍需更大规模训练和跨语言适配

---

## 222. HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding

**arXiv ID:** 2601.14724 | [PDF](https://arxiv.org/pdf/2601.14724v1)

**作者:** Haowei Zhang `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑free、plug‑and‑play 的 KV Cache 层次化管理框架（HERMES），能够在流式视频处理过程中压缩视频 token 并保持低延迟，提升多模态大语言模型在持续视频理解任务上的性能。

**💡 创新点**

创新点：① 将 KV Cache 视为多粒度视频记忆（浅层感知记忆、中层工作记忆、深层长期记忆）；② 依据层级注意力特征设计指数衰减、关注权重等重要度评估；③ 引入跨层内存平滑与位置重索引机制，实现无额外计算、低延迟的高效压缩；④ 通过摘要 token 保留长时记忆，显著减少视频 token 数量（最高 68%）。

**🔧 技术方法**

技术手段：机制化层级注意力分析、指数衰减衰退函数、关注权重重要度评估、跨层平滑（λ 参数）、位置重索引（lazy / eager）、RoPE / M‑RoPE 位置编码、摘要 token 生成。

**📊 数据集**

数据集：StreamingBench、OVO‑Bench、RVS（RVS‑Ego、RVS‑Movie）、MVBench、VideoMME、Egoschema 等流式与离线视频问答基准。

**📈 对比分析**

与基线 MLLM 以及训练‑free 先行方法（ReKV、LiveVLM、StreamMem）在多选和开放式问答任务中对比；在 StreamingBench/Ovo‑Bench 达到 79.44%/59.21%；在 RVS 与长视频集提升至 11.4% 以上；在离线长视频集显著提升 60.29%/58.85%；同时实现 GPU 内存峰值下降 1.04×、首输出延迟提升 10×。

**⚠️ 局限性**

局限性：仍需手工调节重要度与平滑超参数；在极长视频下受记忆预算限制；对查询的泛化依赖伪查询；未覆盖跨模态任务的全自动化；在极低延迟极端场景下可能不如专门优化的实时系统。

---

## 223. Context Patch Fusion With Class Token Enhancement for Weakly Supervised Semantic Segmentation

**arXiv ID:** 2601.14718 | [PDF](https://arxiv.org/pdf/2601.14718v1)

**作者:** Yiyang Fu `[一作]` (Wuxi University), Wangyu Wu `[通讯]` (University of Liverpool)

**通讯引用:** 170 | [OpenAlex ID](https://openalex.org/A5003207136)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了弱监督语义分割框架 CPF-CTE，利用 ViT 提取特征后通过 CF-BiLSTM 恢复空间连续性，并加入可学习的类别令牌以增强语义辨识。

**💡 创新点**

创新点在于：1）post‑hoc 双向 BiLSTM 上下文融合模块针对 ViT 的 patch 断裂进行恢复；2）在 ViT 编码后引入多类别可学习类令牌进行语义细化，避免早期竞争。

**🔧 技术方法**

采用 Vision Transformer (ViT‑B/16) 作为编码器，CF‑BiLSTM、可学习类令牌、Top‑K 池化、CRF 后处理以及最终的 DeepLabV2 进行细粒度分割。

**📊 数据集**

使用 PASCAL VOC 2012（加 SBD 扩展）和 MS COCO 2014 两大公开数据集进行训练与评估。

**📈 对比分析**

与现有 SOTA 方法相比，CPF‑CTE 在 VOC 2012 验证集上伪标签 mIoU 70.8%，最终分割 mIoU 69.5%；在 COCO 2014 上伪标签 41.3%，最终分割 45.4%，均超过同类 Transformer 和 CNN 方法。

**⚠️ 局限性**

主要局限是对高分辨率图像的可扩展性有限，BiLSTM 的序列计算导致推理时延增加，并且仍依赖 patch 级处理。

---

## 224. DARA: Few-shot Budget Allocation in Online Advertising via In-Context Decision Making with RL-Finetuned LLMs

**arXiv ID:** 2601.14711 | [PDF](https://arxiv.org/pdf/2601.14711v1)

**作者:** Mingxuan Song `[一作]` (Peking University), Chuan Yu `[通讯]` (Alibaba Group)

**通讯引用:** 245 | [OpenAlex ID](https://openalex.org/A5059373349)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种双阶段LLM框架DARA，用于在少样本预算分配场景下通过先做全局推理后再细化决策；

**💡 创新点**

创新点在于将任务拆分为“Few-Shot Reasoner”和“Fine-Grained Optimizer”两阶段，配合GRPO‑Adaptive动态更新的RL微调，显著提升LLM在数值敏感任务中的精度；

**🔧 技术方法**

采用大语言模型（LLM）进行少样本推理，GRPO‑Adaptive（强化学习微调）提升策略稳定性与数值精度；

**📊 数据集**

使用真实电商广告业务数据构建的实时竞价环境和基于多项式/指数函数的合成模拟环境；

**📈 对比分析**

与DPO、ABPlanner、HiBid等基线对比，在真实和合成环境中，DARA在降低边际ROI方差方面平均提升10–12%，且在后期阶段优势更明显；

**⚠️ 局限性**

局限性包括对模拟环境的依赖、对极端动态变化场景的适应性待验证，以及RL微调对计算资源和超参数的敏感性。

---

## 225. Case-Guided Sequential Assay Planning in Drug Discovery

**arXiv ID:** 2601.14710 | [PDF](https://arxiv.org/pdf/2601.14710v1)

**作者:** Tianchi Chen `[一作]` (Merck and Co Inc), Xiang Yu `[通讯]` (Merck and Co Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了IBMDP框架，用无模拟器的历史数据构建隐式贝叶斯马尔可夫决策过程，并通过集成MCTS规划给药物发现实验制定高效的序列计划。

**💡 创新点**

创新点在于用相似度加权的非参数案例生成模型实现隐式贝叶斯转移，并结合集成MCTS投票稳健规划，解决了缺乏(s,a,s′)数据的多步决策难题。

**🔧 技术方法**

采用指数核相似度、贝叶斯更新、MCTS‑双进展宽化（D‑PW）以及投票合成的MLASP策略，构成完整的决策与规划流程。

**📊 数据集**

实验使用真实的CNS药物发现历史数据库（220个化合物的PgP、BCRP及k_puu测定）以及公开的鼠犬体内药动学仿真数据，并用可计算最优策略的合成数据进行对照。

**📈 对比分析**

与传统QSAR启发式策略、相似度基价值迭代（VI‑Sim）及理论最优策略（VI‑Theo）进行对比，IBMDP在真实案例中将资源消耗降低至92%以内，在合成数据中Top‑1匹配率为47%，Top‑2为66%，显著优于确定性方法。

**⚠️ 局限性**

主要局限包括对历史数据覆盖度与代表性的高度依赖、相似度度量假设线性欧氏距离可能不适用于非线性生物学关系、计算复杂度随案例数和特征维度上升且对超参数敏感。

---

## 226. AutoDriDM: An Explainable Benchmark for Decision-Making of Vision-Language Models in Autonomous Driving

**arXiv ID:** 2601.14702 | [PDF](https://arxiv.org/pdf/2601.14702v1)

**作者:** Zecong Tang `[一作]` (Zhejiang University), Yu Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 299540 | [OpenAlex ID](https://openalex.org/A5100362465)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AutoDriDM 这个决策中心化、分级进展的视觉语言模型评测基准，用以评估 VLM 在自动驾驶场景下的感知与决策能力，并分析其推理过程。

**💡 创新点**

将评测从传统感知指标拓展到三维感知‑场景‑决策三层递进结构；构建 6.65K 问答、引入高危场景与相似场景对；开发分析器模型实现大规模推理错误标签。

**🔧 技术方法**

使用 VLMs（如 GPT‑4.1、InternVL、Qwen 等）在零/少样本下进行推理；采用 CoT 结构生成推理链；设计 9 类错误标签并训练 7B 参数的自动化分析器；使用 Pearson 相关、鲁棒性测评。

**📊 数据集**

从 nuScenes、KITTI、BDD100K 采集前视图图像，并进行相似度过滤生成 6.65K 问题。

**📈 对比分析**

通过 0‑shot 与 1/2/5‑shot 评估，多模型对比，GPT‑4.1 最高平均分 64.8，开源模型中 Qwen‑72B 最优；大模型在感知任务上稳定但决策任务波动大，感知与决策相关性弱。

**⚠️ 局限性**

只涵盖前视摄像头图像，缺少多模态/时间序列信息；问答形式单/多选限制表述，未覆盖自由生成；模型覆盖有限，可能受推理设置影响。

---

## 227. Training-Efficient Text-to-Music Generation with State-Space Modeling

**arXiv ID:** 2601.14786 | [PDF](https://arxiv.org/pdf/2601.14786v1)

**作者:** Wei-Jaw Lee `[一作]` (National Taiwan University), Yi-Hsuan Yang `[通讯]` (National Taiwan University)

**通讯引用:** 7174 | [OpenAlex ID](https://openalex.org/A5061291906)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了使用状态空间模型(SSM)替代Transformer，构建训练高效的文本到音乐生成模型；

**💡 创新点**

在SSM中引入轻量级通道混合层(前缀SiMBA)并实现两阶段粗细分辨率生成，显著提升训练效率与生成质量；

**🔧 技术方法**

采用Mamba-2/SiMBA SSM、前缀与交叉注意力融合、两阶段SSM+扩散混合结构，以及Flan‑T5文本编码、DAC音频编码、DDPM扩散；

**📊 数据集**

使用Jamendo公开CC授权的457小时音乐与MusicCaps评测集，完全开放数据；

**📈 对比分析**

与MusicGen‑small和Transformer基线在相同参数规模、单RTX3090、10万步训练下进行客观(Fréchet、KL、CLAP)和主观评测，SSM模型在约9% FLOPs、2%数据量下实现相近或优于基线，且两阶段模型性能进一步提升；

**⚠️ 局限性**

受限于公开数据规模、单一音频编码器、对极端复杂音乐场景适应性尚待验证，且仅在中小规模实验中验证，未检验在更大模型或多模态场景中的可扩展性；

---

## 228. Towards Bound Consistency for the No-Overlap Constraint Using MDDs

**arXiv ID:** 2601.14784 | [PDF](https://arxiv.org/pdf/2601.14784v1)

**作者:** Amaury Guichard `[一作]` (UCLouvain), Pierre Schaus `[通讯]` (UCLouvain)

**通讯引用:** 1398 | [OpenAlex ID](https://openalex.org/A5065393744)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种首次实现时间窗口约束边界一致性（BC）的过滤算法，并给出了其多值决策图（MDD）的精确与松弛版本；

**💡 创新点**

创新点在于利用完整MDD对所有边进行扫描，从而在一次遍历中实现BC；并通过将宽度限制为W的松弛MDD实现多项式时间的近似BC过滤，兼具较强的推理力度与可扩展性；

**🔧 技术方法**

核心技术包括：多值决策图的精确构建与顶层压缩、边缘最早/最晚时间的取最小/最大实现BC、松弛MDD的状态合并（⊕）与增量细化；实验中实现了前向/后向状态同步以加强过滤；

**📊 数据集**

实验数据集为单机“just‑in‑time”调度实例（1 | r_j,d_j, d̅_j |∑E_j+∑T_j），随机生成 n=12、14、16、18、25、30、40 等规模的约束；

**📈 对比分析**

比较方法包括：基线（传统的边缘寻找、可检测先后关系等）+ Relaxed BC + 纯前序提取 + 完整BC；使用 replay‑based 搜索重放来保证仅因过滤差异导致的树探索差异；结果显示 Relaxed BC 在节点数与求解时间上相较于基线与前序提取都有明显下降，且在宽度32时树扩展率低于5%，逼近完整BC；

**⚠️ 局限性**

限制在于完整BC构造的 MDD 在规模较大时易指数级膨胀，导致内存与时间消耗大；松弛版本虽可控制宽度，但仍需在宽度选择与合并策略上权衡，过小宽度导致过滤力不足，过大又消耗资源；

---

## 229. M2I2HA: A Multi-modal Object Detection Method Based on Intra- and Inter-Modal Hypergraph Attention

**arXiv ID:** 2601.14776 | [PDF](https://arxiv.org/pdf/2601.14776v1)

**作者:** Xiaofan Yang `[一作]` (Harbin Institute of Technology), Xuanming Cao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5101905549)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于超图注意力的多模态目标检测网络M2I2HA，融合RGB与热/深度等多模态信息，提升低照度、过曝等极端环境下的检测精度。

**💡 创新点**

通过Intra‑Hypergraph Enhancement模块捕捉单模态内的高阶多对多关系，Inter‑Hypergraph Fusion模块实现跨模态高阶关系对齐与融合，并引入M2‑FullPAD实现多层自适应融合，解决信息冗余与模态对齐难题。

**🔧 技术方法**

采用超图注意力机制、低秩分解与稀疏化、SE通道注意、跨模态超图生成与传播、YOLOv8/13骨干、轻量化卷积以及动态图融合等技术。

**📊 数据集**

在DroneVehicle、FLIR‑Aligned、LLVIP、VEDAI四大公开多模态检测基准上进行实验。

**📈 对比分析**

与SuperYOLO、ICAFusion、CFT、GHOST、GM‑DETR、COMO等SOTA方法对比，M2I2HA在各数据集均取得最高或接近最高的mAP与mAP@.75，并保持低参数量与高FPS，显示出精度与效率双赢。

**⚠️ 局限性**

目前仅支持RGB+热/深度两模态，尚未验证更广泛模态组合；对极端遮挡或尺度极端变化的鲁棒性仍有限；模型在不同硬件上的兼容性与部署效率需要进一步评估。

---

## 230. Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models

**arXiv ID:** 2601.14758 | [PDF](https://arxiv.org/pdf/2601.14758v1)

**作者:** Injin Kong `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 1664 | [OpenAlex ID](https://openalex.org/A5021733732)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对预训练的自回归模型 (ARM) 进行掩码扩散训练后得到的 MDM 进行电路层级对比，分析机制转变。

**💡 创新点**

发现 MDM 在不同任务下呈现“机制转移”现象：局部因果任务保留自回归电路，全球规划任务则前置层聚焦、分布式语义重构。

**🔧 技术方法**

使用电路发现（EAP‑IG）、顶级边/组件重叠比较、Logit Lens、神经元可视化、Top‑K 分歧等技术。

**📊 数据集**

使用 IOI（间接宾语识别）和 Countdown（数值推理）两大任务，结合 Qwen2.5‑7B 与 LLaMA‑2‑7B 预训练模型及其后训练的 Dream/DiffuLLaMA。

**📈 对比分析**

通过边重叠、Top‑K 组件相似度对比评估机制重用度；IOI 任务重用率高，Countdown 任务重用率低，表明 MDM 在全局推理任务中显著重塑网络，性能优于 ARM。

**⚠️ 局限性**

仅考察两类任务，电路发现聚焦高权重路径，可能忽略辅助机制；方法计算成本高，未实现对全模型的彻底扫查。

---

## 231. Trajectory-Driven Multi-Product Influence Maximization in Billboard Advertising

**arXiv ID:** 2601.14737 | [PDF](https://arxiv.org/pdf/2601.14737v1)

**作者:** Dildar Ali `[一作]` (Indian Institute of Technology Jammu), Rajibul Islam `[通讯]` (Gandhi Institute for Technological Advancement)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对多产品广告牌广告场景，本文定义了两类影响力最大化问题：共同槽位（同一套槽位服务所有产品）和互斥槽位（各产品独占槽位），并分别给出了近似算法。

**💡 创新点**

创新点在于把多产品需求建模为多子模函数覆盖问题，提出基于连续贪心+随机化逼近的双重准则算法以及基于采样和原始‑对偶贪心的高效求解框架，并给出理论误差/近似保证。

**🔧 技术方法**

核心技术包括子模函数优化、连续贪心与多线性扩展、随机化逼近、Hoeffding不等式样本复杂度分析、原始‑对偶贪心策略。

**📊 数据集**

实验使用纽约市（NYC）和洛杉矶（LA）公开轨迹与广告牌槽位数据集，规模分别达到数十万轨迹与数百万槽位。

**📈 对比分析**

与随机分配、Top‑k分配等基线相比，实验表明PDG和BCA在影响力满足率、预算利用率以及计算时间上均显著优越；尤其在高需求、有限供应场景下，PDG保持较高效率与精度。

**⚠️ 局限性**

限制主要包括：算法在离线静态环境下设计，未考虑广告商竞争、实时动态更新及多渠道传播；此外，原始‑对偶贪心的理论近似系数尚未给出，需进一步研究。

---

## 232. Optimizing FaaS Platforms for MCP-enabled Agentic Workflows

**arXiv ID:** 2601.14735 | [PDF](https://arxiv.org/pdf/2601.14735v1)

**作者:** Varad Kulkarni `[一作]` (Indian Institute of Science), Yogesh Simmhan `[通讯]` (Indian Institute of Science)

**通讯引用:** 6158 | [OpenAlex ID](https://openalex.org/A5041794289)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FAME架构，将ReAct型代理工作流拆分为Lambda函数，利用LangGraph、AWS Step Functions、DynamoDB实现无状态服务器的会话记忆与工具调用；

**💡 创新点**

创新点包括：①将代理模式模块化为Planner、Actor、Evaluator三个无状态函数；②通过DynamoDB自动注入代理记忆；③自动化将MCP服务器包装为Lambda；④使用S3缓存工具输出并支持函数融合；

**🔧 技术方法**

技术栈包括AWS Lambda、Step Functions、LangGraph、DynamoDB、S3、OpenAI GPT‑4o‑mini、MCP服务器包装工具及缓存管理；

**📊 数据集**

数据集为两类应用：论文摘要（三篇科研论文）和日志分析（Apache、Hadoop、OpenSSH日志）；

**📈 对比分析**

通过五种内存/缓存配置（E、N、C、M、M+C）在两应用上对比E2E延迟、输入/输出Token及成本，实验表明M+C配置可实现最高13×延迟下降、88%输入Token减少、66%成本节省；

**⚠️ 局限性**

局限性包括：LLM随机性导致部分配置失效；冷启动和函数分配开销；仅评估ReAct模式与两应用；缺乏异步MCP调用和跨云编排；内存摘要与扩展性待进一步研究。

---

## 233. Robustness of Mixtures of Experts to Feature Noise

**arXiv ID:** 2601.14792 | [PDF](https://arxiv.org/pdf/2601.14792v1)

**作者:** Dong Sun `[一作]` (CISPA Helmholtz Center for Information Security), Rebekka Burkholz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Mixture of Experts（MoE）模型在特征噪声环境下的鲁棒性与效率，并给出了理论与实验的支持。

**💡 创新点**

提出稀疏专家激活可以作为噪声滤波器，使MoE在保持总参数量不变的情况下相较于稠密模型具有更低的泛化误差、更强的对噪声鲁棒性、更快的收敛速度和更好的样本效率。

**🔧 技术方法**

使用线性模型的可解析理论框架、贝叶斯最优估计、梯度下降收敛分析、块对角设计矩阵、以及在冻结LLM激活上进行的线性探针实验。

**📊 数据集**

数据集包括合成噪声数据、WikiText2、T5‑small激活、以及AG News、CoLA、MNLI、SST‑2等自然语言处理基准。

**📈 对比分析**

通过对比稠密基线和MoE线性探针、以及基于MiniMind架构的MoE与稠密模型，实验显示MoE在噪声水平高时的性能下降明显小于稠密模型，且收敛速度更快、验证损失下降更快，验证了理论预测。

**⚠️ 局限性**

局限在于理论主要基于理想的块对角结构和完美路由，实际模型中的路由学习与噪声分布可能更复杂；实验多在低维或线性探针层进行，未覆盖更大规模非线性MoE网络的全面验证。

---

## 234. CI4A: Semantic Component Interfaces for Agents Empowering Web Automation

**arXiv ID:** 2601.14790 | [PDF](https://arxiv.org/pdf/2601.14790v1)

**作者:** Zhi Qiu `[一作]` (Beijing Institute of Technology), Xin Peng `[通讯]` (Fudan University)

**通讯引用:** 13727 | [OpenAlex ID](https://openalex.org/A5071724015)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

实现了一种面向代理的组件接口协议（CI4A），并在 Ant Design 框架下构建了 Eous 代理，通过把 UI 组件抽象为可调用工具原语，实现了对 WebArena 任务的高效交互。

**💡 创新点**

提出了“面向代理的组件接口”——通过语义化状态视图、可执行工具集和交互元数据三元组，直接把复杂 UI 逻辑压缩为单步调用，消除了传统低级 DOM 交互的冗余链。

**🔧 技术方法**

利用 LLM（GPT‑5）+ 强化学习、前端注入（AntDX）、结构与视觉双模感知、自动工具包装、工具注册与全局调用机制。

**📊 数据集**

在重构为 Ant Design 版本的 WebArena 基准数据集上进行评测。

**📈 对比分析**

与现有基线（WebArenaBase、AgentOccam、MidScene 等）对比，Eous 在结构输入下 75.3% 成功率、视觉输入下 86.3% 成功率，平均步骤分别为 4.2 和 4.7，明显优于70.3% 等传统方法。

**⚠️ 局限性**

目前仅支持 Ant Design 并需手动注入，难以迁移至其他 UI 库；对动态 Web 需要进一步的自动化工具生成与适配。

---

## 235. RECAP: Resistance Capture in Text-based Mental Health Counseling with Large Language Models

**arXiv ID:** 2601.14780 | [PDF](https://arxiv.org/pdf/2601.14780v1)

**作者:** Anqi Li `[一作]` (Zhejiang University), Zhenzhong Lan `[通讯]` (Westlake University)

**通讯引用:** 7807 | [OpenAlex ID](https://openalex.org/A5103239171)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了面向文本心理咨询的细粒度抗拒行为识别框架 PsyFIRE，并基于该框架创建了 ClientResistance 数据集。

**💡 创新点**

创新点在于提出13类细粒度抗拒行为的理论框架、结合对话上下文的解释性标注以及可解释的两阶段模型 RECAP。

**🔧 技术方法**

采用 LLaMA‑3.1‑8B‑Instruct 进行全参数微调，同时结合自监督生成解释的多模态训练。

**📊 数据集**

使用来自中文在线文本心理咨询平台的 23,930 条标注对话，构成 ClientResistance 数据集。

**📈 对比分析**

与 GPT‑4o、Claude‑3.5‑Sonnet 等大模型在零/少样本下对比，RECAP 在二分类 F1 91.25%、细粒度 F1 66.58% 远超基线（差距约20个百分点）。

**⚠️ 局限性**

局限包括仅关注面对面抗拒，未覆盖作业不完成等形式；数据仅来自中文情境，跨文化泛化待验证。

---

## 236. ClaimDB: A Fact Verification Benchmark over Large Structured Data

**arXiv ID:** 2601.14698 | [PDF](https://arxiv.org/pdf/2601.14698v1)

**作者:** Michael Theologitis `[一作]` (University of Washington), Dan Suciu `[通讯]` (University of Washington)

**通讯引用:** 24276 | [OpenAlex ID](https://openalex.org/A5048204602)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一个新的事实核查基准，专门针对包含数百万记录、跨多张表的大规模结构化数据库中的主张进行验证。

**💡 创新点**

创新点在于：①将主张与需要多表组合、聚合、排序等复杂 SQL 操作的海量数据相结合，逼迫模型进行可执行程序级推理；②构建了覆盖 80 个真实数据库、平均 11 张表、4.5M 记录的规模；③利用 LLM 评判面板和 SQL 工具调用评估流程。

**🔧 技术方法**

技术手段包括：基于 BIRD 的 NL‑to‑SQL 对应、SQL AST 过滤、生成式 LLM（如 GPT‑4‑Turbo、Mistral 等）用于生成主张、LLM‑as‑Judge 面板评审、SQL 工具调用代理（MCP）、指标计算（准确率、宏 F1、每类精确率/召回率）。

**📊 数据集**

使用的数据集为 BIRD Benchmark 的 11k NL‑to‑SQL 例子（从公共 split 生成 6.5k 组合问答），随后基于这些问答生成主张并过滤得到 53,368 条最终样本；同时伴随 80 个真实数据库（平均 11 张表、4.5M 记录）。

**📈 对比分析**

对 30 个现有 LLM（包括 OpenAI、Anthropic、Microsoft、Mistral、开源 70B 以下模型）进行评估，最优模型准确率约 83%，宏 F1 最高约 0.86；但超过一半的模型准确率低于 55%，open‑source 多在 68% 以下；模型对 NEI（无法判断）标签的处理呈现两极化：部分模型极度回避 NEI，另一些则过度预测 NEI。

**⚠️ 局限性**

限制包括：①依赖 BIRD，若其 NL‑SQL 注释错误会直接影响基准；②只考虑结构化表格证据，未覆盖多模态文本/图像；③使用固定数据库快照，评估结果随数据更新可能变化；④评估仅基于 SQL 工具调用，模型的 SQL 写作能力会显著影响结果。

---

## 237. Tailoring Adverse Event Prediction in Type 1 Diabetes with Patient-Specific Deep Learning Models

**arXiv ID:** 2601.14917 | [PDF](https://arxiv.org/pdf/2601.14917v1)

**作者:** Giorgia Rigamonti `[一作]` (University of Milano-Bicocca), Paolo Napoletano `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 4657 | [OpenAlex ID](https://openalex.org/A5053344554)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并验证了一种基于Bi‑GRU的两阶段个性化血糖预测框架，利用多模态患者日志实现患者识别与精细化预测

**💡 创新点**

创新点在于将多模态数据（CGM、胰岛素注射、碳水化合物摄入）与患者识别相结合，采用LOSOCV预训练后微调的两阶段学习策略，实现对每位1型糖尿病患者的专属模型

**🔧 技术方法**

使用Bi‑GRU循环网络、双向递归层、dropout、平均池化、SMOTE过采样、缩减损失函数、Adam优化器等深度学习与数据增强技术

**📊 数据集**

实验基于公开的OhioT1DM与DiaTrend两大T1D多模态数据集，涵盖CGM、胰岛素泵、碳水化合物摄入及生活方式信息

**📈 对比分析**

与仅使用CGM的CNN基准模型比较，患者特定模型在RMSE、时间收益(TG)、高/低血糖敏感度等指标上均优于通用模型，整体提升约10%–20%，并且在训练数据量仅占原始数据的25%时仍保持较好性能

**⚠️ 局限性**

对血糖波动极不规则或极少数据的患者预测效果仍有限；模型需要一定量的个体化历史数据，冷启动时性能较低，未来需进一步提高对低资源场景的适应性

---

## 238. SynPerf: A Hybrid Analytical-ML Framework for GPU Performance Prediction

**arXiv ID:** 2601.14910 | [PDF](https://arxiv.org/pdf/2601.14910v1)

**作者:** Kaixuan Zhang `[一作]` (Shanghai Jiao Tong University), Liping Zhang `[通讯]` (Alibaba Group)

**通讯引用:** 582 | [OpenAlex ID](https://openalex.org/A5100426765)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了SynPerf，一种将分析模型与机器学习相结合的GPU性能预测框架；

**💡 创新点**

创新点在于先用分析方法量化核函数对各异构指令流水线的需求，再用轻量级MLP捕捉复杂的资源竞争与交互，从而实现高保真、跨硬件通用的预测；

**🔧 技术方法**

核心技术包括：基于源码的Kernel Decomposer、SM调度仿真、多级特征分析（算术和内存流水线需求）、以及多层感知机（MLP）预测器；

**📊 数据集**

使用了约1M条GPU内核执行记录，涵盖11款NVIDIA GPU（4代）与6类关键LLM推理内核（GEMM、Attention、RMSNorm、SiLU&Mul、Scaled MM、Fused MoE），以及多模型（Qwen2.5-14B、Qwen3-32B、Llama3.1-70B）的推理工作负载；

**📈 对比分析**

与经典Roofline、线性回归和State‑of‑the‑Art Neusight相比，SynPerf在已知硬件上的MAPE仅为6.0%，在未见硬件上为11.5%，相较Neusight的45.1%降低约4.3×；在端到端推理上，平均MAPE低至6.6%，比Neusight的34.7%提升约5.3×；

**⚠️ 局限性**

限制在于目前仅针对NVIDIA GPU架构，缺乏对AMD或其他加速器的直接验证；此外，框架对极端异构或多节点分布式环境的支持尚未完整实现。

---

## 239. Comparative Study of Large Language Models on Chinese Film Script Continuation: An Empirical Analysis Based on GPT-5.2 and Qwen-Max

**arXiv ID:** 2601.14826 | [PDF](https://arxiv.org/pdf/2601.14826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 240. SpatialMem: Unified 3D Memory with Metric Anchoring and Fast Retrieval

**arXiv ID:** 2601.14895 | [PDF](https://arxiv.org/pdf/2601.14895v1)

**作者:** Xinyi Zheng `[一作]` (University of Bristol), Junxiao Shen `[通讯]` (Memories.ai Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一种基于单目RGB视频的空间记忆系统 SpatialMem，融合几何、语义与语言，形成可查询的三维层次化记忆树，并支持语言驱动的导航与物体检索。

**💡 创新点**

创新点在于：①以墙面、门窗等结构锚点为支撑，构建统一的度量坐标框架；②提出双层文本描述（属性层+关系层），实现细粒度可解释的空间推理；③实现无深度/IMU、仅 RGB 的实时更新与查询，显著降低硬件门槛。

**🔧 技术方法**

技术方案包括：单目几何后端（如VGGT/SLAM3R）用于姿态与稠密深度估计；点云对齐与楼层对齐；基于 Open‑Vocabulary 分割（GLIP/Segment Anything）与 CLIP 的视觉‑语言表征；树形记忆结构与距离/方向/可见性查询算法；轻量级缓存与索引实现低延迟检索。

**📊 数据集**

使用了三套真实室内 RGB 视频数据集：简单房间、套房主房、实验室/存储场景，共 1500 条实验（每个任务 500 条），并对每个场景生成对应的三维点云、锚点与物体标签。

**📈 对比分析**

与 Gemini、InternVL、LLaVA‑Video、GPT‑4o 等多模态基线在三项任务（相对位置、导航、物体检索）进行评测。SpatialMem 在大多数指标上与最强基线持平或略优，尤其在导航步完成率、路径效率和检索层级准确度上表现突出，同时拥有更低的推理时延和更小的模型体积。

**⚠️ 局限性**

局限性包括：①单目几何的深度/姿态误差会影响锚点与物体的精确定位；②复杂光照与遮挡下的属性识别（颜色、尺寸）准确率下降；③对动态物体或长期环境变化的适应性不足，需进一步提升在线更新与迁移学习能力。

---

## 241. To Neuro-Symbolic Classification and Beyond by Compiling Description Logic Ontologies to Probabilistic Circuits

**arXiv ID:** 2601.14894 | [PDF](https://arxiv.org/pdf/2601.14894v1)

**作者:** Nicolas Lazzari `[一作]` (University of Bologna), Antonio Vergari `[通讯]` (University of Edinburgh)

**通讯引用:** 853 | [OpenAlex ID](https://openalex.org/A5069110696)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将 Description Logic 本体编译成可分解、光滑、确定性的 SDD 电路，并利用该电路完成三大任务：① 生成符合本体语义的合成数据集；② 在 GPU 上实现可扩展的演绎推理；③ 通过语义损失（SL）与语义概率层（SPL）实现可靠的神经符号分类器。

**💡 创新点**

创新点在于：
• 采用知识编译技术把 DL 本体完整映射到电路，既能高效推理又能在训练时保持可微分；
• 通过电路实现的可采样属性，为本体驱动的数据生成提供了一种全新的无监督方式；
• 将电路直接嵌入神经网络，保证预测必定满足本体约束（SPL）或软约束（SL），实现可靠推理与高性能的融合。

**🔧 技术方法**

技术方法包括：
• Description Logic（EL+）本体建模；
• 知识编译（从 DL 生成 CNF → SDD 电路）；
• 采样、求和、乘积等电路运算实现可采样与推理；
• 语义损失（Weighted Model Counting）与语义概率层（可学习的权重）实现神经符号集成；
• GPU 并行化评估电路以获得三阶加速。

**📊 数据集**

使用的主要数据集是论文中自行生成的合成知识图谱与关联样本；本体通过随机生成算法产生（控制概念数、角色数、域/范围限制、互斥等），随后通过电路采样得到知识图谱和样本。未使用公开真实数据集。

**📈 对比分析**

与传统深度学习基线（MLP、LR、SVM 等）以及 DeepProbLog 进行对比。评价指标包括精确率、召回率、F1、Exact Match、Consistency。实验表明：
• NeSy 方法在保持与基线相当或更优的分类性能的同时，Consistency 远高于基线（>90% 对比 ~30%）。
• SPL 在所有指标上均最佳，且可保证完全一致；SL 在软约束场景下表现优秀；DeepProbLog 受限于推理速度。
• 推理速度方面，基于电路的推理在 GPU 上比 Pellet/HermiT 速率高达 10^3 倍。

**⚠️ 局限性**

主要局限：
• 编译算法对本体规模敏感，过大的本体导致 CNF 变得指数级大，编译时间和电路尺寸难以控制；
• 仅覆盖 EL+ 等可归约为 SDD 的 DL 子语言，无法直接处理更复杂的 OWL 语法；
• 目前实验仅在合成数据上验证，缺乏在真实大规模知识图谱上的实证；
• 对于动态知识更新或不确定性本体的支持尚未完善。

---

## 242. Citation of scientific evidence from video description and its association with attention and impact

**arXiv ID:** 2601.14916 | [PDF](https://arxiv.org/pdf/2601.14916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 243. Strategic Doctrine Language Models (sdLM): A Learning-System Framework for Doctrinal Consistency and Geopolitical Forecasting

**arXiv ID:** 2601.14862 | [PDF](https://arxiv.org/pdf/2601.14862v1)

**作者:** Olaf Yunus Laitinen Imanov `[一作]` (Technical University of Denmark), Derya Umut Kulali `[通讯]` (Eskisehir Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Strategic Doctrine Language Models (sdLM) 的两模型框架（70B 参数的 GIPFEL‑I 与 30B 参数的 SANDKASTEN‑I），用于自动化战略规划、作战演练和教义一致性分析；

**💡 创新点**

核心创新包括：①多文档注意力机制与时间位置编码实现长达 32,768 令牌的跨文档推理；②专门的教义一致性层和多域融合模块；③基于 2B 令牌历史作战与教义数据的三阶段训练（预训练、监督微调、基于人类反馈的强化学习）；

**🔧 技术方法**

采用 Transformer 体系结构（自注意力、层归一化）、多任务损失（CLM + 文献一致性 + 时间一致性）、对比学习与 RLHF 以及 INT8 量化、FlashAttention 等推理加速技术；

**📊 数据集**

训练数据来自 2B 令牌的战略与战役文本、336 本军事教义、2,847 条战役计划、340M 战役记录与 800M 战争演练转录，总计约 2.8B 令牌；

**📈 对比分析**

通过 47 名 O‑6 至 O‑10 级军方战略家评估 127 个历史/合成案例，GIPFEL‑I 的战略方案质量平均 8.42/10、教义一致性 91% 以上、12 个月地缘预测准确率 73%，比 GPT‑4/Claude‑2/Defense‑Llama 等基线高；SANDKASTEN‑I 的情景可行性 8.1/10、裁定一致性 0.89，推理速度提升 10–14 倍；

**⚠️ 局限性**

局限包括：对 60 个月后预测的准确率下降至约 59%；模型规模 70B 参数需要 142 GB GPU 内存，限制部署；幻觉率约 3.2% 仍需人工校验；数据偏向西方教义可能导致对非西方对手的战略适用性不足；

---

## 244. Just aware enough: Evaluating awareness across artificial systems

**arXiv ID:** 2601.14901 | [PDF](https://arxiv.org/pdf/2601.14901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 245. Symmetry Informative and Agnostic Feature Disentanglement for 3D Shapes

**arXiv ID:** 2601.14804 | [PDF](https://arxiv.org/pdf/2601.14804v1)

**作者:** Tobias Weißberg `[一作]` (University of Bonn), Florian Bernard `[通讯]` (Lamarr Institute)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5089781469)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种无监督的特征解耦网络，将每个点的语义形状描述符分解为对称信息与对称无关两部分，并通过马尔可夫随机场对对称信息进行平滑细化。

**💡 创新点**

创新点在于同时学习对称信息与对称无关信息的解耦、引入多种无监督对称损失以及利用MRF能量最小化提升对称信息的鲁棒性。

**🔧 技术方法**

技术包括使用2D基础模型（如DINO‑V2、StableDiffusion）提取图像特征、轻量自编码器、正交矩阵投影、无监督对称损失（不相似、相似、重建、边界、一致性）以及MRF细化。

**📊 数据集**

数据集涵盖BeCoS（人类与动物）、FAUST、SCAPE、SMAL、TOSCA等，提供丰富的自对称和左右标注。

**📈 对比分析**

在对称检测、左右分类和形状匹配等任务上与基线（DINO+SD、χ、χ+refine）比较，实验结果表明解耦特征在平均误差、分类准确率和匹配质量上均显著优于现有方法。

**⚠️ 局限性**

局限性包括依赖网格连通信息、对近邻部件容易出现误差、MRF细化对连续特征偏差敏感以及目前仅在零基因表面上验证。

---

## 246. Diagonals and algebraicity modulo $p$: a sharper degree bound

**arXiv ID:** 2601.14920 | [PDF](https://arxiv.org/pdf/2601.14920v1)

**作者:** Boris Adamczewski `[一作]` (University Claude Bernard Lyon), Xavier Caruso `[通讯]` (CNRS)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过新的直接且简洁的证明方法，给出了Deligne定理的一个有效的多变量版本，并提供了对模p化简对角线多项式的代数度数的多项式上界，

**💡 创新点**

创新点在于引入截断算子与残差公式，并利用Newton多面体与Frobenius不变量的几何结构，得到只与多项式的次数与高度相关的可计算上界N，从而实现了第一份具有合理N的普适多项式界；

**🔧 技术方法**

核心技术包括截断算子（Cartier算子）、残差运算、Newton多面体的几何分析以及对角线/广义对角线的变换；

**📊 数据集**

本文未使用实验数据或公开数据集，主要依靠理论证明；

**📈 对比分析**

相较于先前的非多项式或指数上界结果，本文的上界在理论上更紧，并通过显式表达式说明其与代数函数的几何属性直接相关；

**⚠️ 局限性**

限制在于所得的指数N虽可计算，但在某些高维或高度极大的情形下仍可能非常大；此外，结果主要适用于完备特征为p的域，且在特征0的情形需要额外的取模与完备化处理。

---

## 247. PodBench: A Comprehensive Benchmark for Instruction-Aware Audio-Oriented Podcast Script Generation

**arXiv ID:** 2601.14903 | [PDF](https://arxiv.org/pdf/2601.14903v1)

**作者:** Chenning Xu `[一作]` (Tencent), Mingyang Song `[通讯]` (Tencent)

**通讯引用:** 225 | [OpenAlex ID](https://openalex.org/A5063533156)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PodBench，一个用于播客脚本生成的基准，包含800个样本，支持多达21K个令牌的输入和复杂的多说话者指令。

**💡 创新点**

创新点在于建立了一个多维度的评估框架，结合定量约束和基于LLM的质量评估，填补了播客脚本生成领域的评估空白。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行播客脚本生成和评估，特别是通过明确的推理增强模型的表现。

**📊 数据集**

使用了来自多个公共语料库的多样化数据集，包括LongWanjuan和PileArxiv，涵盖12个常见的AI播客领域。

**📈 对比分析**

与现有模型进行比较，专有模型通常表现优异，而配备明确推理的开源模型在处理长上下文和多说话者协调方面表现出更强的鲁棒性，但高指令遵循并不总能保证高内容实质。

**⚠️ 局限性**

限制在于PodBench未能覆盖所有播客格式或对话风格的多样性，评估框架依赖于LLM的判断，可能无法完全捕捉个体听众的偏好。

---

## 248. What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A Systematic Study

**arXiv ID:** 2601.14888 | [PDF](https://arxiv.org/pdf/2601.14888v1)

**作者:** Keyu Lv `[一作]` (Shenzhen International Graduate School, Tsinghua University), Haoli Bai `[通讯]` (Huawei Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究低位量化感知训练（QAT）在推理模型上的效果，提出并验证了一个三阶段的 Reasoning-QAT 工作流；

**💡 创新点**

创新点在于：① 证明知识蒸馏是低位 QAT 的最佳目标；② 发现 PTQ 初始化能显著加速训练并提高最终精度；③ 强化学习在冷启动后能进一步提升性能；④ 强调 PTQ 校准域与 QAT 训练域对齐的重要性；

**🔧 技术方法**

使用技术包括：PTQ（RTN、GPTQ、AWQ）→ 量化感知训练（QAT）→ 知识蒸馏（KD）→ 强化学习（GRPO），以及 3/2 位组量化权重（W3G128、W2G128）；

**📊 数据集**

使用数据集：OpenR1-Math（训练/校准）、Wikitext2、NuminaMath-1.5、以及评估基准 AIME-120、MATH-500、GSM8K、LiveCodeBench、GPQA-Diamond；

**📈 对比分析**

对比方法：与多种 PTQ 基线（RTN、GPTQ、AWQ）以及通用 QAT 基线（EfficientQAT、BitDistiller）在 3 位和 2 位权重量化下进行实验；性能方面，Reasoning-QAT 在多模型、多任务上均大幅恢复精度，3 位下在 MATH-500 上提升 44% 以上，2 位下显著恢复数学任务，并把与 BF16 的差距缩小到 5–10%；

**⚠️ 局限性**

Limitations：实验数据主要以数学为主，跨领域（编码、科学等）泛化有限；仅为经验验证，未提出新的量化算法；在极低位（2 位）下，部分小模型的恢复仍不理想。

---

## 249. HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation

**arXiv ID:** 2601.14874 | [PDF](https://arxiv.org/pdf/2601.14874v1)

**作者:** Yara Mahmoud `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 1963 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用视觉‑语言模型结合检索增强生成框架，帮助Unitree G1人形机器人根据自身摄像头的单张RGB图像自动选择合适的笛卡尔阻抗参数和抓取角度，以实现自适应、可解释的接触丰富操作。

**💡 创新点**

创新点在于：①将高层语义推理（VLM）与低层控制参数检索（FAISS‑RAG）直接耦合；②构建两套小规模实验验证的数据库（阻抗和抓取角度）实现任务感知与执行的闭环；③在没有力/触觉传感器的情况下，通过虚拟力作为接触反馈，完成任务级的可变阻抗控制。

**🔧 技术方法**

核心技术包括：Molmo 视觉‑语言模型、句向量嵌入（sentence‑transformer）、FAISS 相似度搜索、基于任务空间的笛卡尔质量‑弹簧‑阻尼阻抗控制、逆运动学规划与执行。

**📊 数据集**

使用了九种实验验证的任务场景（表面跟踪、按摩球施力、双物体放置、工具操作、抓取上桌等）以及两套基于真实实验的阻抗与抓取角度数据库。

**📈 对比分析**

对比方法为：在14张变视角的测试图像上测评检索准确率与执行性能。检索准确率达93%，在实际操作中z轴跟踪误差均≤3.5 cm，虚拟正交力随阻抗调节呈线性变化，表明检索参数能产生稳定且任务适配的接触行为。

**⚠️ 局限性**

局限性：仅在九个预定义任务内表现良好，无法泛化至未见任务；对视角、遮挡敏感；缺乏闭环力/触觉反馈，阻抗只能预设；数据库规模有限，未实现连续空间映射。

---

## 250. Reclaiming Software Engineering as the Enabling Technology for the Digital Age

**arXiv ID:** 2601.14861 | [PDF](https://arxiv.org/pdf/2601.14861v1)

**作者:** Tanja E. J. Vos `[一作]` (Informatics Europe), Benoît Combemale `[通讯]` (Informatics Europe)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出软件工程作为数字时代核心技术的必要性，并呼吁重塑其科学身份。

**💡 创新点**

提出五项行动计划：利用ICSE FoSE进行共同反思、强化科研核心、统一学界与产业信息、制定共享研究议程、投资共享基础设施。

**🔧 技术方法**

主要采用政策分析、案例论证和跨学科合作框架；未涉及传统编程或实验技术。

**📊 数据集**

无直接数据集；论证依据行业案例、政策文件与学术会议讨论。

**📈 对比分析**

无实验对比，性能或效果评估通过示例和概念性论证呈现。

**⚠️ 局限性**

局限在缺乏可操作的实施细节与量化评估指标，需要进一步制定具体方案和衡量机制。

---

## 251. Moving Beyond Compliance in Soft-Robotic Catheters Through Modularity for Precision Therapies

**arXiv ID:** 2601.14837 | [PDF](https://arxiv.org/pdf/2601.14837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 252. Multimodal system for skin cancer detection

**arXiv ID:** 2601.14822 | [PDF](https://arxiv.org/pdf/2601.14822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 253. HiNS: Hierarchical Negative Sampling for More Comprehensive Memory Retrieval Embedding Model

**arXiv ID:** 2601.14857 | [PDF](https://arxiv.org/pdf/2601.14857v1)

**作者:** Motong Tian `[一作]` (OPPO), Wangchunshu Zhou `[通讯]` (OPPO)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了HiNS框架，对记忆检索嵌入模型进行分层负样本采样和对话式数据合成，提升模型在对话记忆检索任务中的区分能力。

**💡 创新点**

将负样本划分为易、中、难三层并依据真实人机对话的负样本比例校准，引入语义感知查询生成与跨会话负样本采样，形成完整的分层负样本数据构造方法。

**🔧 技术方法**

采用基于BERT的对比学习、InfoNCE损失、LLM生成对话和查询、主题聚类、分层负样本采样与加权，以及分布式训练技术。

**📊 数据集**

通过Nemotron-Personas生成约20万条合成对话样本，并在LoCoMo和PERSONAMEM两个记忆检索基准上进行评估。

**📈 对比分析**

与未微调的BGE模型在MemoryOS和Mem0两大框架下对LoCoMo和PERSONAMEM进行基准对照，平均F1/ BLEU-1提升约3-4%，PERSONAMEM总分提升约1-2%，显示显著的性能改进。

**⚠️ 局限性**

负样本难度划分仍基于规则，缺乏学习式预测；训练时间有限，未充分挖掘模型容量；未直接评估对话生成质量等下游任务。

---

## 254. Multi-Tast Transformer for Explainable Speech Deepfake Detection via Formant Modeling

**arXiv ID:** 2601.14850 | [PDF](https://arxiv.org/pdf/2601.14850v1)

**作者:** Viola Negroni `[一作]` (Politecnico di Milano), Stefano Tubaro `[通讯]` (Politecnico di Milano)

**通讯引用:** 8431 | [OpenAlex ID](https://openalex.org/A5005378965)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种轻量化的多任务Transformer模型SFATNet‑4，用于语音深度伪造检测，并通过预测声纹轨迹、声子区分和最终判别结果实现可解释性；

**💡 创新点**

创新点在于：1）采用仅按时间轴切片的输入分割，显著降低复杂度；2）引入多头加权池化机制，将帧级重要性显式化；3）将声子与声子区分预测作为辅助任务，提升语音表征与可解释性；4）整体模型参数量减少、训练速度加快。

**🔧 技术方法**

技术包括：Transformer编码器（幅值与相位分别编码）、多任务解码器（声子、声子区分、深度伪造判别）、多头池化注意力、线性投影、Sigmoid与MSE/BCELoss组合训练。

**📊 数据集**

使用四个公开数据集：ASVspoof 5、In‑the‑Wild、FakeOrReal、TIMIT‑TTS，均对16 kHz音频进行统一预处理。

**📈 对比分析**

与前身SFATNet‑3对比，SFATNet‑4在ASVspoof 5内域EER降低约4.4%、AUC提升约2.2%；在三大异域数据集的EER下降约1–3点、AUC提升约2–4点；模型参数从64.7 M降至41.8 M，单轮训练时间从≈60 min降至≈15 min。

**⚠️ 局限性**

局限性：在压缩/编码器（如MP3、Opus）下仍会出现性能衰退；缺乏数据增强导致对真实环境鲁棒性不足；目前仅针对英语语音，跨语言通用性未验证。

---

## 255. From Observation to Prediction: LSTM for Vehicle Lane Change Forecasting on Highway On/Off-Ramps

**arXiv ID:** 2601.14848 | [PDF](https://arxiv.org/pdf/2601.14848v1)

**作者:** Mohamed Abouras `[一作]` (German University in Cairo), Catherine M. Elias `[通讯]` (German University in Cairo)

**通讯引用:** 97 | [OpenAlex ID](https://openalex.org/A5025295346)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了高速公路入口/出口路段车辆车道变换行为预测。

**💡 创新点**

创新点在于将多层堆叠LSTM应用于入口/出口区域的车道变换预测，并系统比较不同预测时段与模型结构的性能。

**🔧 技术方法**

采用多层堆叠LSTM、ReLU激活、交叉熵损失函数及RMSProp优化器等技术。

**📊 数据集**

使用ExiD无人机轨迹数据与HighD高速公路数据。

**📈 对比分析**

通过对比E2E模型与多层分阶段模型，以及不同预测时段，E2E模型在4秒预测时的准确率约为78%，与常规高速段相比仍具竞争力。

**⚠️ 局限性**

局限在于预测时延高达1秒、对更长时段预测效果不佳，且入口/出口区域复杂性导致准确率下降。

---

## 256. POTR: Post-Training 3DGS Compression

**arXiv ID:** 2601.14821 | [PDF](https://arxiv.org/pdf/2601.14821v1)

**作者:** Bert Ramlot `[一作]` (Ghent University), Glenn Van Wallendael `[通讯]` (Ghent University)

**通讯引用:** 1501 | [OpenAlex ID](https://openalex.org/A5048527990)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对3D高斯喷射（3DGS）的后训练压缩编解码器POTR，能够在不进行重新训练的情况下显著减小模型体积并加速推理。

**💡 创新点**

创新点包括：
- 用修改后的前向渲染一次性精确评估每个喷射点删除对图像误差的影响，打破传统仅基于启发式指标的削枝策略；
- 采用岭回归对每个喷射点的球谐系数进行能量压缩，显著降低AC光照系数熵并保持可视化质量；
- 设计了可选的轻量级微调流程，进一步提升率-失真表现和推理速度。

**🔧 技术方法**

使用的技术主要有：
- 3D Gaussian Splatting渲染与稀疏光照表示；
- 基于GPU的并行前向渲染改造计算删除效应；
- 球谐系数的岭回归能量压缩与稀疏化；
- 八叉树量化与Zstd熵编码；
- 简单的量化、序列化和可选的微调训练。

**📊 数据集**

采用的公开数据集包括：Mip-NeRF 360、Deep Blending、Tanks and Temples、NeRF-Synthetic 四个常用场景集。

**📈 对比分析**

与现有后训练压缩方法（如MesonGS、LightGaussian）以及含微调的基线进行对比。实验表明，POTR 在 20–45× 的压缩比下，保持或提升 PSNR/SSIM/LPIPS，压缩模型的喷射点数量比其他方法少 2–4 倍，推理速度提升 1.5–2 倍；在加上微调后，压缩率再提升 35–50%，推理速度也进一步提高。

**⚠️ 局限性**

局限性：
- 编码时间较长，主要受前向渲染和多次削枝迭代的影响；
- 需要对原始训练视角进行渲染，若视角不充分可能导致删除误判；
- 虽然不需要训练，但在极端压缩率下仍可能需要微调才能获得可接受的视觉质量；
- 当前实现主要针对单机GPU，尚未充分探索分布式或更高效的并行策略。

---

## 257. Statistical Learning Theory for Distributional Classification

**arXiv ID:** 2601.14818 | [PDF](https://arxiv.org/pdf/2601.14818v1)

**作者:** Christian Fiedler `[一作]` `[通讯]` (Technical University of Munich), Christian Fiedler (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文针对两阶段抽样设置下的分布式输入分类问题，给出了支持向量机（SVM）的理论分析，包括通用的占优不等式、一致性证明和学习率上界。

**💡 创新点**

创新点在于：①首次在两阶段采样框架下对含锥形损失的SVM给出完整的占优不等式；②提出不需要离散化的全局一致性与学习率结论；③引入针对高维希尔伯特空间的高斯核新特征空间，并基于几何噪声指数（改良版）得到无光滑性假设下的学习率。

**🔧 技术方法**

主要技术包括：核均值嵌入（KME）、Hilbertian嵌入、岭正则化的经验风险最小化、占优不等式推导、噪声/间隔指数假设以及对高斯核的白噪声特征映射。

**📊 数据集**

论文未给出具体实验数据集，主要集中在理论推导与理论上可验证的假设。

**📈 对比分析**

论文未与其他方法做实验对比，亦未给出具体性能指标，仅给出学习率理论上界。

**⚠️ 局限性**

局限性在于：①理论依赖于较强的可测性与可嵌入假设；②学习率结果主要在特定噪声指数假设下；③缺少实验验证和多类别扩展。

---

## 258. Stochastic Decision-Making Framework for Human-Robot Collaboration in Industrial Applications

**arXiv ID:** 2601.14809 | [PDF](https://arxiv.org/pdf/2601.14809v1)

**作者:** Muhammad Adel Yusuf `[一作]` (King Fahd University of Petroleum and Minerals), Zeeshan Hameed Khan `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个基于POMDP的协作机器人与人类协作决策框架，整合了安全三圆模型、情绪与动机状态以及任务优先级，实现了双向意图预测与实时决策。

**💡 创新点**

创新点在于：① 将安全、情绪、任务三因素统一进POMDP状态；② 引入三圆安全模型动态调整工作空间；③ 通过贝尔曼方程与观测更新实现在线策略，兼顾协作效率与安全。

**🔧 技术方法**

使用技术包括：POMDP建模、MDP价值迭代、贝尔曼方程、观测与信念更新、强化学习框架、成本函数设计（多项式与指数项），以及多模态传感器（速度、姿态）观测。

**📊 数据集**

未使用公开真实数据集，而是基于人工设定的仿真参数与人类速度观测生成的离散状态进行实验验证。

**📈 对比分析**

通过数值仿真案例展示了任务完成时间、距离安全与情绪响应的协同效果；虽然未与现有方法做直接对比，但结果表明在满足安全与效率的前提下，框架能够实现可接受的性能。

**⚠️ 局限性**

局限性包括：① 状态空间巨大（419,904个状态），计算成本高；② 需要先验概率分布或学习估计，现实中难以获取；③ 情绪预测仅依赖速度等有限观测，精度受限；④ 仅在仿真中验证，缺乏真实工业环境实验。

---

## 259. UniRoute: Unified Routing Mixture-of-Experts for Modality-Adaptive Remote Sensing Change Detection

**arXiv ID:** 2601.14797 | [PDF](https://arxiv.org/pdf/2601.14797v1)

**作者:** Qingling Shu `[一作]` (Anhui University), Chengzhuang Liu `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 UniRoute，一种通过像素级动态专家路由实现自适应感受野与融合操作的统一多模态遥感变化检测框架。

**💡 创新点**

创新点在于引入 AR²‑MoE 与 MDR‑MoE 的像素级硬路由，以及 CASD 自监督一致性训练，显著提升跨模态鲁棒性。

**🔧 技术方法**

使用技术包括 Mixture‑of‑Experts、Straight‑Through Estimator 硬路由、稀疏自监督一致性损失、DSBN 等。

**📊 数据集**

使用了 LEVIR‑CD、WHU‑CD、HTCD、MT‑Wuhan、XiongAn 等五个公开变化检测数据集。

**📈 对比分析**

与单模型专用和统一基线对比，UniRoute 在五个数据集上平均 F1 取得 85.10%，比统一基线提升约 4–8%，参数量仅 52.9M。

**⚠️ 局限性**

局限性包括对极端跨模态或极少样本场景仍受限，路由决策可能导致训练不稳定，需要进一步研究。

---

## 260. Validating Behavioral Proxies for Disease Risk Monitoring via Large-Scale E-commerce Data

**arXiv ID:** 2601.14795 | [PDF](https://arxiv.org/pdf/2601.14795v1)

**作者:** Naomi Sasaya `[一作]` (LY Corporation), Akira Tajima `[通讯]` (LY Corporation)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用大规模电子商务购买记录中的饮食切换行为，验证其作为猫咪泌尿系统疾病风险监测的行为代理；

**💡 创新点**

创新点在于将消费行为代理与独立保险临床数据进行交叉验证，建立可扩展的数字流行病学验证框架；

**🔧 技术方法**

采用关联分析（卡方检验、相关系数）、季节分解（STL）以及趋势检验（Cochran–Armitage）等统计技术；

**📊 数据集**

使用雅虎购物平台的猫咪食品购买日志（2018-2020）与Anicom保险的问卷关联病例对照数据和按月聚合的索赔数据；

**📈 对比分析**

通过计算代理指标（Switch Rate）与基准指标（Claim Rate）的相关系数，发现成分层面r=0.74、季节层面r=0.82，表明代理与临床数据高度一致；

**⚠️ 局限性**

局限性包括缺乏个体级关联、可能存在存活偏差和观测偏差、聚合索赔数据未计入人群基数，以及对消费行为变化未能捕捉潜在疾病过程。

---

## 261. Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation

**arXiv ID:** 2601.14896 | [PDF](https://arxiv.org/pdf/2601.14896v1)

**作者:** Rui Qi `[一作]` (Beijing Jiaotong University), Kaiyu Huang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 439 | [OpenAlex ID](https://openalex.org/A5031577422)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种语言耦合的 GRPO 框架，利用多轮检索和抗一致性奖励实现多语言检索增强生成，缓解知识偏差和冲突。

**💡 创新点**

创新点在于将语言耦合的群体采样与 3-gram 召回奖励、抗一致性惩罚结合，显著提升跨语言泛化与稳定性。

**🔧 技术方法**

使用技术包括强化学习（GRPO）、多语言多轮检索、语言耦合 Rollout 与 Reward 模块、3-gram 召回奖励以及抗一致性惩罚。

**📊 数据集**

使用数据集 MKQA、XOR-TyDi QA（13 语言子集）以及多语言 E5 检索器和 Wikipedia dump。

**📈 对比分析**

与无检索、单语言 RAG、IRCoT、Search-o1、D-RAG、SFT、Search-R1 等基线在 fEM、c3Recall、CLR 三指标上进行对比，方法在三种 LLM 上均实现显著提升（如 fEM 最高 47.6，c3Recall 63.2，CLR 99.4）。

**⚠️ 局限性**

局限性包括评估仅依赖现有 QA 数据和 Wikipedia，缺乏专门的检索相关性标注；仅试验三种 LLM，检索器固定为多语言 E5-base，未探讨检索算法改进。

---

## 262. The CHI26 Workshop on the Future of Cognitive Personal Informatics

**arXiv ID:** 2601.14891 | [PDF](https://arxiv.org/pdf/2601.14891v1)

**作者:** Christina Schneegass `[一作]` (Delft University of Technology), Max L. Wilson `[通讯]` (University of Nottingham)

**通讯引用:** 2630 | [OpenAlex ID](https://openalex.org/A5088221028)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

组织了一场CHI 2026的工作坊，聚焦认知个人信息学（CPI）的未来，包括生成式AI与可穿戴技术的整合、数据解释与用户交互设计，并通过小组讨论与可视化协作工具制定研究议程。

**💡 创新点**

创新点在于将大型语言模型与用户生成内容结合，用以生成个性化的认知洞察，提出以AI为中介的认知伴侣概念，并在多学科背景下系统化讨论伦理、可访问性与多样性议题。

**🔧 技术方法**

主要技术手段包括大型语言模型（LLM）用于数据合成与解释、Miro协作板进行实时可视化与投票、Slack社群与Medium博客进行预研交流，以及现场分组讨论与投票机制。

**📊 数据集**

数据来源主要是参与者在提交模板时提供的案例、经验与问题描述（匿名化处理），而非传统实验或公开数据集；因此缺乏标准化量化数据。

**📈 对比分析**

比较方法主要是基于提交内容的主题聚类、挑战优先级投票与时间线分析，缺乏实验性对比或性能度量，评估以定性讨论和共识达成为主。

**⚠️ 局限性**

局限性包括：仅限现场与视频录播，无法覆盖全球参与；缺乏实证实验验证AI解释的准确性与透明度；数据样本非公开标准数据集，难以复现；伦理与隐私风险仍需进一步深入研究。

---

## 263. GAT-NeRF: Geometry-Aware-Transformer Enhanced Neural Radiance Fields for High-Fidelity 4D Facial Avatars

**arXiv ID:** 2601.14875 | [PDF](https://arxiv.org/pdf/2601.14875v1)

**作者:** Zhe Chang `[一作]` (University of Shanghai for Science and Technology), Hui Yu `[通讯]` (University of Glasgow)

**通讯引用:** 12958 | [OpenAlex ID](https://openalex.org/A5006580423)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于 Geometry-Aware Transformer 的 GAT-NeRF 框架，能够从单目视频高精度重建可控的 4D 面部数字化人偶。

**💡 创新点**

创新点在于将轻量级 Transformer 与 NeRF 结合，利用多模态输入（位置编码、3DMM 表情参数、帧级可学习潜码）进行点级特征增强和自注意力重加权，显著提升高频细节（如动态皱纹、细纹）重建效果。

**🔧 技术方法**

核心技术包括：NeRF 隐式场表示、Transformer 自注意力模块、3D Morphable Model (3DMM) 表情参数、可学习潜码、体积渲染与分层采样。

**📊 数据集**

使用公开的 NeRFace 单目视频数据集进行训练与评估。

**📈 对比分析**

与 NeRFace、PointAvatar、FlashAvatar 等方法对比，GAT-NeRF 在 L1、SSIM 等结构化指标上取得最佳成绩，PSNR 接近最优，LPIPS 亦具竞争力。

**⚠️ 局限性**

局限性主要在于仅针对单个主体进行优化，缺乏跨主体泛化；训练与渲染速度仍高于显式 3D 方案；潜码解释性差，难以直观控制微表情等细节。

---

## 264. On-the-fly hand-eye calibration for the da Vinci surgical robot

**arXiv ID:** 2601.14871 | [PDF](https://arxiv.org/pdf/2601.14871v1)

**作者:** Zejian Cui `[一作]` (Imperial), Ferdinando Rodriguez y Baena `[通讯]` (Imperial)

**通讯引用:** 5450 | [OpenAlex ID](https://openalex.org/A5051562079)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于实时手眼标定的框架，利用无训练的关键点关联（JCBB）与可见性检查实现 2D‑3D 对应，然后通过 EKF、AEKF、PF 或 RANSAC‑PnP 估算手眼变换，从而实现 da Vinci 手术工具的高精度定位。

**💡 创新点**

创新点在于：①将可见性检查与 JCBB 结合，显著提升关键点匹配精度并减少计算量；②提供多种滤波器（EKF、AEKF、PF）以适应不同噪声分布；③框架不依赖预训练模型，适用于任何拥有 CAD 模型的手术工具，具备高度通用性。

**🔧 技术方法**

核心技术包括：视觉–运动互补的 AX=XB 形式手眼标定；Jacobian 计算与 Mahalanobis 距离判定；Branch & Bound 形式的 JCBB 数据关联；扩展卡尔曼、自适应卡尔曼与粒子滤波器；以及基于 RANSAC 的 PnP 作为对照。

**📊 数据集**

实验使用公开视频数据集 SurgPose（多工具、内外实验）和 SuPer（标记检测、颜色分割）进行验证；对比使用不同初始化帧数、扰动水平和测量误差场景。

**📈 对比分析**

与传统 PnP 以及已有 EKF 方案比较：在大多数场景下，AEKF 与 PF 在 3D 重建误差上优于 PnP，且在大扰动下仍保持相对稳定；AEKF 在 100 Hz 以上完成，适合实时需求；PF 虽在噪声鲁棒性最高，但计算量显著较大；PnP 速度最快但易受初始误差影响。总体误差在 1–3 mm 级别，旋转误差 <0.2 rad。

**⚠️ 局限性**

局限性：①平移分量估计不够稳定，受初始标定误差影响较大；②缺乏工具姿态和关节角度的真实标注，难以在关节空间进行完整评估；③PF 的粒子退化与计算负担高；④可见性检查基于几何假设，极端姿态可能失效；⑤框架仍需在临床真实手术环境中进一步验证。

---

## 265. A Category-Theoretic Framework for Dependent Effect Systems

**arXiv ID:** 2601.14846 | [PDF](https://arxiv.org/pdf/2601.14846v1)

**作者:** Satoshi Kura `[一作]` (Waseda University), Hiroshi Unno `[通讯]` (Tohoku University)

**通讯引用:** 952 | [OpenAlex ID](https://openalex.org/A5044643168)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了基于索引前序单体和索引加重单体的语义框架，用于构建和解释依赖效应系统；

**💡 创新点**

创新点在于将传统的加重单体推广为索引化形式，并通过加重单体提升构造索引加重单体，为依赖效应系统提供通用语义模型；

**🔧 技术方法**

主要使用范畴论中的Grothendieck构造、分离分辨层、强单子提升、Scott连续ωCPO模型等技术；

**📊 数据集**

无数据集，采用理论实例化（如成本分析、期望边界、时间安全、联合界限等）进行验证；

**📈 对比分析**

方法通过理论证明类型安全性和模型可行性进行比较，未进行实验性能评估；

**⚠️ 局限性**

局限性包括：对let规则处理不够完善；递归的支持需要额外的可接受性假设；当前仅扩展了少量简单效应系统实例。

---

## 266. Archives, archival bond, and digital representation: A case study with the International Image Interoperability Framework

**arXiv ID:** 2601.14823 | [PDF](https://arxiv.org/pdf/2601.14823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 267. CAG-Avatar: Cross-Attention Guided Gaussian Avatars for High-Fidelity Head Reconstruction

**arXiv ID:** 2601.14844 | [PDF](https://arxiv.org/pdf/2601.14844v1)

**作者:** Zhe Chang `[一作]` (University of Shanghai for Science and Technology), Hui Yu `[通讯]` (University of Glasgow)

**通讯引用:** 12958 | [OpenAlex ID](https://openalex.org/A5006580423)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了CAG-Avatar框架，利用交叉注意力的条件自适应融合模块改进3D高斯渲染的头部动画，实现位置自适应的驱动信号，解决全局条件下细节模糊问题。

**💡 创新点**

创新点在于：1）将位置编码作为Query，全局表情码作为Key/Value的交叉注意力机制，使每个高斯原语获得位置特定的驱动信号；2）将高斯嵌入FLAME表面，实现几何一致性的动态细节补偿；3）通过残差融合与轻量MLP预测细粒度偏移，显著提升刚性部件（如牙齿）的重建质量。

**🔧 技术方法**

技术手段包括：3D Gaussian Splatting渲染、交叉注意力机制、FLAME面部模型、Spherical Harmonics颜色表示、Huber与LPIPS损失、PyTorch3D与PyTorch实现、MICA面部追踪、RVM抠图、BiSeNet口腔分割等。

**📊 数据集**

使用公开的多分钟面部视频数据集（与FlashAvatar等方法相同的训练/测试划分），视频经过裁剪、512×512统一尺寸、RVM抠图、BiSeNet口腔分割等预处理后进行训练。

**📈 对比分析**

与FlashAvatar在相同数据集、相同训练轮次（15k）下比较，评估指标为L1、PSNR、SSIM、LPIPS。CAG-Avatar在PSNR +0.02、SSIM +0.007、LPIPS -0.016的提升，并在牙齿等刚性区域实现更清晰、无模糊的重建，同时保持实时渲染性能。

**⚠️ 局限性**

局限性包括：1）目前仅针对面部，未扩展至全身捕捉；2）交叉注意力模块仍占用一定计算资源，未来可进一步优化以提升效率；3）对极端表情或光照变化的鲁棒性尚未充分验证；4）依赖FLAME参数化模型，可能限制对极端面部姿态的表达。

---

## 268. MTFlow: Time-Conditioned Flow Matching for Microtubule Segmentation in Noisy Microscopy Images

**arXiv ID:** 2601.14841 | [PDF](https://arxiv.org/pdf/2601.14841v1)

**作者:** Sidi Mohamed Sid El Moctar `[一作]`, Hélène Bouvrais `[通讯]` (CNRS, University Rennes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出MTFlow，一种基于时间条件流匹配的微管网络分割框架。

**💡 创新点**

创新点在于将流匹配动态引入细小曲线结构分割，提供可解释的迭代改进过程。

**🔧 技术方法**

采用U‑Net骨干+正弦时间嵌入、组归一化、SiLU激活，学习向量场实现噪声掩模到真值的连续更新。

**📊 数据集**

使用合成MicSim_FluoMT（简单/复杂）和真实MicReal_FluoMT数据集，并在DRIVE与CORN1等公共曲线结构数据集上测试。

**📈 对比分析**

与U‑Net、U‑Net++、ResUNet、TransUNet、CAR‑UNet等方法对比，MTFlow在Dice、MCC、PR‑AUC等指标上均优于或相当，尤其在低信噪比场景下表现更稳健。

**⚠️ 局限性**

局限性包括训练样本仍需人工或半自动标注，模型对极端噪声和尺寸极小的结构仍可能出现漏检；未来需进一步加入注意力与不确定性学习。

---

## 269. Implementing Knowledge Representation and Reasoning with Object Oriented Design

**arXiv ID:** 2601.14840 | [PDF](https://arxiv.org/pdf/2601.14840v1)

**作者:** Abdelrhman Bassiouny `[一作]` (AICOR Institute for Artificial Intelligence), Michael Beetz `[通讯]` (AICOR Institute for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了KRROOD框架，将知识表示与推理嵌入Python OOP，实现知识为一等公民的内置推理与查询。

**💡 创新点**

创新点在于将知识结构直接映射为Python类与属性，提出EQL查询语言、Python原生RDR、ORMatic持久层与Ontomatic自动OWL转换，解决传统OOP与知识库的阻塞问题。

**🔧 技术方法**

技术包括Python OOP、类与属性做知识结构、EQL（类似SQL的查询语言）、Ripple Down Rule（RDR）实现交互式规则构建、ORMatic（SQLAlchemy封装）持久化、Ontomatic（OWL→Python）以及PyDatalog/OWLready2等工具。

**📊 数据集**

使用的数据集为OWL2Bench（OWL 2 RL）用于推理与查询基准，以及自制的机器人任务学习实验（Montessori箱插入）用于交互式学习评估。

**📈 对比分析**

与owlready2、Protégé+Pellet、GraphDB、RDFLib+owlrl等进行加载、推理和查询基准；KRROOD在加载+推理原始/增量时接近GraphDB、明显快于RDFLib、比owlready2快；查询时EQL略慢于SQLAlchemy，但提供完整对象模型，整体性能满足机器人实时需求。

**⚠️ 局限性**

局限在于完整对象化导致内存与加载开销较大；EQL采用闭世界假设与开放世界DL语义不兼容；OWL到Python的转换在非互斥子类或多父类关系时可能丢失信息；Python代码执行降低了静态可分析性，可能出现无限循环或非终止问题。

---

## 270. ICLF: An Immersive Code Learning Framework based on Git for Teaching and Evaluating Student Programming Projects

**arXiv ID:** 2601.14814 | [PDF](https://arxiv.org/pdf/2601.14814v1)

**作者:** Pierre Schaus `[一作]` (UCLouvain), Augustin Delecluse `[通讯]` (UCLouvain)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

设计并实现了基于 Git 的 Immersive Code Learning Framework (ICLF)，用于在 MOOCs 等大规模课程中管理和评估学生编程项目，提供沉浸式的起始代码库、自动化模板生成、持续集成评测和透明反馈。

**💡 创新点**

创新点在于：①使用教师隐藏的父仓库与模板化去除解答生成中间公开仓库，让学生从真实代码库开始；②通过协作者身份将学生加入私有 Fork，支持项目随时更新而不影响学生进度；③结合 GitHub Actions/CI 自动生成和更新模板，统一隐藏测试与代码；④采用 JavaGrader 扩展实现加权评测、时间/资源限制，提供即时、可追溯的评分与抄袭检测；⑤整合 INGInious 等自动评测平台，实现无人工干预的评估流程。

**🔧 技术方法**

技术栈包括：Git（GitHub/GitLab/Bitbucket）、GitHub Actions/CI、Python 脚本（amanda 用于代码去除）、JavaGrader（JUnit5 扩展）实现加权评测与限制、INGInious 自动评测平台、以及标准的 Git 操作（fork、pull、merge）。

**📊 数据集**

数据集与实验：离散优化课程（约100名学生/年，连续3年）和 Constraint Programming MOOC（约300名学生/年，连续3年）两大教学案例；使用约10k行的约束求解器源码作为项目，收集学生提交的 commit 统计和匿名反馈。

**📈 对比分析**

比较方法与性能：论文通过学生提交的 commit 统计、匿名问卷评估框架效果，并对自动评测流程的即时反馈与透明度进行描述；实验显示系统能无人工干预地处理大规模提交，学生多在截止日前提交并能及时自测；与传统基于手工评测或 Web-CAT 等系统相比，ICLF 提供更真实的项目体验、更细粒度的反馈与更低的维护成本。

**⚠️ 局限性**

局限性：仅适用于 Git 基础设施，需教师维护父仓库与标记；当前主要支持 Java，其他语言需自行扩展；隐藏测试与代码的实现较为繁琐，可能导致学生过度依赖本地测试；在大规模并发提交时仍可能出现合并冲突；对测试完整性和安全性的依赖较高，需要持续维护 CI 脚本和评测环境。

---

## 271. Reflecting in the Reflection: Integrating a Socratic Questioning Framework into Automated AI-Based Question Generation

**arXiv ID:** 2601.14798 | [PDF](https://arxiv.org/pdf/2601.14798v1)

**作者:** Ondřej Holub `[一作]` (Czech Technical University in Prague), Rodrigo Alves `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 178 | [OpenAlex ID](https://openalex.org/A5062633566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于两角色LLM（学生教师和教师教育者）的反思式问题生成框架，采用Socatic多轮对话动态迭代改进单一反思题；

**💡 创新点**

创新点在于将教学反思问题生成与Socratic提问相结合，利用两代理交互实现自适应终止与逐步优化，提升问题的清晰度、相关性、深度与整体质量；

**🔧 技术方法**

技术包括大型语言模型（GPT‑4o‑mini为生成主体，GPT‑4类模型为评估者）、基于角色的Prompt设计、动态迭代控制与LLM评估的对比评分；

**📊 数据集**

数据集为真实小学信息技术课堂的教学材料（幻灯片、教师指南等）以及提取的关键概念集；

**📈 对比分析**

通过LLM评估器对不同配置（动态停止/固定5/10轮、是否提供学生水平/材料）进行成对比较，得到的偏好指数γ显示动态停止结合材料/水平时在清晰度、相关性、深度和整体质量上均优于固定轮次；与单次生成基线比较，反思式框架在相关性和深度上显著更佳；

**⚠️ 局限性**

主要局限是评估完全基于LLM，缺乏人工标注验证；仅使用轻量级LLM作为生成主体，未与更强推理模型对比；未来需开展课堂实证与混合评估。

---

## 272. Efficient reversal of transductions of sparse graph classes

**arXiv ID:** 2601.14906 | [PDF](https://arxiv.org/pdf/2601.14906v1)

**作者:** Jan Dreier `[一作]` (TU Wien), Michał Pilipczuk `[通讯]` (University of Warsaw)

**通讯引用:** 5591 | [OpenAlex ID](https://openalex.org/A5000479623)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在多项式时间（O(n^4)）内，给定稀疏图类𝒞的任意图G，构造一个颜色化稀疏图H，使得G可以通过一次固定的一阶解释I从H恢复，并且H属于一个具有有限扩张的图类𝒞'。

**💡 创新点**

首次证明：如果图类𝒞满足“单子稳定性”与“固有线性邻域复杂度”，则它既是有限扩张的又能被转化为一个具有有限扩张的图类；这等价于对结构稀疏图类的完整描述，并给出了对任意结构有限扩张类的高效稀疏化（sparsification）算法。

**🔧 技术方法**

核心技术包括：
- 一阶解释与转移（transduction）框架；
- 基于邻域覆盖与“枝分枝”（branchings）的递归树构造；
- Haussler打包引理与近似相同邻域（k‑near‑twins）分析；
- 通过树的高度与重叠度控制稀疏化的复杂度；
- 逻辑可定义化技术使得所有步骤都可用一阶公式实现。

**📊 数据集**

本研究属于纯理论计算复杂性与组合数学，没有实验或数据集；所有结论均为形式化证明。

**📈 对比分析**

与以往只能得到弱稀疏化（仅保证几乎处处稀疏）的方法相比，本算法在O(n^4)时间内实现了真正的有限扩张稀疏化；同时得到的解释I可直接用于将所有在结构有限扩张类上的FO模型检验、计数、枚举等问题转移到已知的有限扩张类算法上，从而获得固定参数多项式时间或更好的性能。

**⚠️ 局限性**

局限性：
- 运行时间为O(n^4)，在实际大规模图上可能仍然较慢；
- 仅适用于单子稳定且具有固有线性邻域复杂度的图类，无法覆盖所有稀疏图类；
- 需要借助一阶可定义的逻辑公式，实施起来复杂；
- 对于需要更快稀疏化的实际应用场景，仍需进一步优化算法。

---

## 273. 5G NR Non-Terrestrial Networks: Open Challenges for Full-Stack Protocol Design

**arXiv ID:** 2601.14883 | [PDF](https://arxiv.org/pdf/2601.14883v1)

**作者:** Francesco Rossato `[一作]` (University of Padova), Marco Giordani `[通讯]` (University of Padova)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文通过系统级 ns-3 仿真，评估并分析了 3GPP 5G‑NR‑NTN 标准在卫星通信场景中的关键协议设计挑战，涵盖同步、资源分配、双工模式、HARQ、切换、路由与传输层等方面。

**💡 创新点**

创新点在于：①提出并公开一个完整栈的开源 ns-3 模块，支持卫星链路的物理、MAC 与高层交互；②基于该模块系统性评估标准化进程（Rel.17–20）的技术方案与瓶颈；③给出针对同步、延迟、吞吐率等指标的量化改进建议。

**🔧 技术方法**

技术手段包括：ns-3 模拟平台、3GPP 5G‑NR‑NTN 规范实现、卫星传播模型（路径损耗、气象衰减、Doppler 预补偿）、时间/频率同步算法、TDD 预留槽调度、HARQ 过程扩展、UDP/TCP 性能对比以及基于轨道预估的路由与切换策略。

**📊 数据集**

使用的“数据集”为仿真生成的场景参数：不同高度（600 km、1200 km、36000 km）的 LEO/MEO/GEO 卫星，频段（S、Ka）、吞吐率、延迟等；并通过多次仿真实验得到吞吐量、延迟、HARQ 成功率等指标。

**📈 对比分析**

比较方法为：在相同网络配置下，分别启用/禁用相关技术（如多 HARQ 过程、TDD 预留槽、UDP/TCP 变体），记录吞吐量、延迟与资源利用率；结果显示：多 HARQ 过程可提升 LEO 站点吞吐率，但 GEO 站点仍受限；TDD 预留槽导致大幅延迟，改进的槽分配方案可显著降低延迟；UDP 在高延迟场景下保持稳定吞吐，而 TCP 需要额外的 PEP 或参数调优。

**⚠️ 局限性**

局限性包括：仅基于仿真验证，缺乏真实卫星链路实验；模型参数对不同卫星系统的适用性有限；对极端气象或链路失效场景的鲁棒性尚未深入；以及对多路径、频率分配等更细粒度物理层细节的覆盖不够完整。

---

## 274. FastFI: Enhancing API Call-Site Robustness in Microservice-Based Systems with Fault Injection

**arXiv ID:** 2601.14800 | [PDF](https://arxiv.org/pdf/2601.14800v1)

**作者:** Yuzhen Tan `[一作]` (Wuhan University), Shaolin Tan `[通讯]` (Zhongguancun Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种基于快速故障注入的框架 FastFI，用于微服务系统中 API 调用点的鲁棒性提升。

**💡 创新点**

创新点包括：1）DFS+位掩码求解器专门针对单调、低重叠 CNF 进行快速枚举；2）动态 k‑故障注入机制通过运行时反馈自适应增大组合大小；3）将注入结果映射为 Part‑Max‑SAT，指导 API 调用点的硬化。

**🔧 技术方法**

采用的技术包括：CNF 构造、DFS+位掩码搜索、动态增量搜索、EnvoyFilter 注入、Partial Max‑SAT 求解。

**📊 数据集**

使用了四个公开微服务基准（Online Boutique、Sock Shop、Hotel Reservation、Train Ticket）以及一个自建的一百万服务模拟基准。

**📈 对比分析**

与 LDFI、IntelliFI、MicroFI 等现有方法对比，FastFI 在故障注入次数降低约 70%、CNF 求解时间缩短约 94%、端到端时间缩短约 77%。DFS 求解器在真实和模拟基准上均比 Z3、AE‑Kissat‑MAB、Kissat‑public 快数十倍。

**⚠️ 局限性**

局限性：仅针对已知的多路径冗余场景，未覆盖未知故障；鲁棒性优化依赖注入得到的故障集合，可能对其他潜在组合无效；在更大规模生产环境下的可扩展性和实际部署成本仍需进一步验证。

---

## 275. CodeDelegator: Mitigating Context Pollution via Role Separation in Code-as-Action Agents

**arXiv ID:** 2601.14914 | [PDF](https://arxiv.org/pdf/2601.14914v1)

**作者:** Tianxiang Fei `[一作]` (Tencent), Mingyang Song `[通讯]` (Tencent)

**通讯引用:** 225 | [OpenAlex ID](https://openalex.org/A5063533156)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CodeDelegator 框架，将任务规划与代码实现分离为持久的 Delegator 与短暂的 Coder 两种角色。

**💡 创新点**

通过角色专门化和 Ephemeral‑Persistent State Separation (EPSS) 两项创新，显著抑制了代码执行痕迹导致的上下文污染问题。

**🔧 技术方法**

采用双层工作空间、结构化通信协议、Python 可执行代码、LLM 交互（主要使用 DeepSeekV3.2 与 GPT‑5）等技术。

**📊 数据集**

在 τ²‑bench 和 MCPMark 两大基准数据集上进行评估。

**📈 对比分析**

与 ReAct 与 CodeAct 基线相比，CodeDelegator 在 Retail、Airline 以及多工具任务中 pass@1/4 或整体成功率提升 12–17%，在高难度任务上表现更为稳定。

**⚠️ 局限性**

目前实现仅支持顺序线性子任务拆解，无法处理 DAG 结构或并发异步执行，限制了资源利用与任务多样性。

---

## 276. AlertGuardian: Intelligent Alert Life-Cycle Management for Large-scale Cloud Systems

**arXiv ID:** 2601.14912 | [PDF](https://arxiv.org/pdf/2601.14912v1)

**作者:** Guangba Yu `[一作]` (Sun Yat-Sen University), Ruijie Xu `[通讯]` (Tencent)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5101580328)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了AlertGuardian框架，整合图学习模型和LLM实现了从报警去噪、报警摘要到报警规则优化的完整报警生命周期管理，显著降低了报警噪声，提升了故障诊断效率，并通过人机交互验证了规则优化效果。

**💡 创新点**

创新点在于：①基于图学习和虚拟噪声节点的实时报警去噪模型，可处理大规模、多属性报警；②利用检索增强生成（RAG）+LLM实现自动化报警摘要，生成可直接操作的根因和解决方案；③设计多智能体迭代反馈循环，对报警规则进行去冗、合并、阈值与时间条件优化，形成闭环的规则自学习机制；④在工业场景中对四个领域的真实数据进行大规模实验验证。

**🔧 技术方法**

技术包括：图学习模型（LINE + Transformer）与最大似然估计；虚拟噪声节点与属性匿名化；检索增强生成（RAG）+ Deepseek V3 LLM；多智能体（Detect, RAG, Rule, Review）工作流；统计与聚类方法（HDBSCAN、周期性检测）；语义匹配与语法校验工具。

**📊 数据集**

使用了四个来自同一公司的真实云服务数据集：游戏（A），办公（B），媒体（C），教育（D），分别包含1.2万至5.9万条规则，200万至388万条报警，9天时间窗口。

**📈 对比分析**

通过与四种基线方法（Severity、UHAS、OAS、无匿名化）对比，AlertGuardian在报警去噪上实现93.8%–95.5%下降率，精度和召回率均优于基线；在报警摘要上，RAG+Deepseek达成98.5%动作识别、90.5%根因识别，优于非LLM方法；在规则优化上，提出约1,174条规则改进，SRE接受率约32%。

**⚠️ 局限性**

局限性包括：①实验数据仅来自单一公司，泛化性待验证；②LLM可能产生幻觉，需RAG与迭代校正；③属性匿名化虽然有效但可能掩盖某些真实模式；④规则优化需要人工审批，仍受人力限制；⑤对实时系统的推理延迟虽然低于200 ms，但在更大规模场景下可能需要进一步优化。

---

## 277. Understanding Usefulness in Developer Explanations on Stack Overflow

**arXiv ID:** 2601.14865 | [PDF](https://arxiv.org/pdf/2601.14865v1)

**作者:** Martin Obaidi `[一作]` (Leibniz Universität Hannover), Jil Klünder `[通讯]` (University of Applied Sciences FHDW Hannover)

**通讯引用:** 802 | [OpenAlex ID](https://openalex.org/A5088299353)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Stack Overflow上开发者回答的解释性内容，量化其结构、情感、时间和作者特征与社区认定的有用度之间的关系。

**💡 创新点**

创新点在于将解释性（explainability）与需求工程的可理解性、模糊性降低等概念结合，并系统评估了结构、情境和语言特征对感知有用度的影响，揭示情感几乎无效、及时性与代码/链接丰富度是主要驱动因素。

**🔧 技术方法**

采用文本分析（BERT情感分类、文本统计、可读性、词汇多样性、相似度计算）和统计建模（Spearman、点二乘、Eta系数）来提取特征并检验假设。

**📊 数据集**

使用了包含3,323个问题与59,398条答案的Stack Overflow Android标签数据集，筛选出投票≥50的高质量问答，并公开发布于Zenodo。

**📈 对比分析**

通过多变量相关分析与Bonferroni校正比较不同特征对相对有用度的影响，结果显示代码块、链接、答案长度和发布时间等因素小到中度正相关；情感影响极小；作者声誉中度正相关；整体效应虽显著但幅度有限。

**⚠️ 局限性**

局限性包括：依赖投票得分作为有用度代理，无法直接衡量准确性或理解；情感分类可能误判技术文本；相关性研究无法推断因果；仅分析高票Android标签，缺乏跨域与跨平台验证；LLM生成答案的兴起可能改变社区评价机制。

---

## 278. Adaptive Exponential Integration for Stable Gaussian Mixture Black-Box Variational Inference

**arXiv ID:** 2601.14855 | [PDF](https://arxiv.org/pdf/2601.14855v1)

**作者:** Baojun Che `[一作]` (Peking University), Weijie Wang `[通讯]` (Peking University)

**通讯引用:** 3684 | [OpenAlex ID](https://openalex.org/A5100398429)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于高斯混合族的黑盒变分推断框架，结合自然梯度、指数型自适应积分器和自适应时间步，解决协方差正定性与噪声导致的数值不稳定问题；

**💡 创新点**

创新点包括：①自适应指数积分器无条件保持协方差正定性；②自然梯度预处理实现仿射不变；③理论证明噪声下的几何收敛和自适应步长必要性；④与镜像下降、流形优化的深度连接；

**🔧 技术方法**

使用技术包括自然梯度、指数型积分器、Cholesky分解、Monte Carlo无偏梯度估计、仿射不变、镜像下降、cosine 退火调度、热退火初始化；

**📊 数据集**

实验数据集涵盖二维多峰、Neal's funnel、多维(10/50维)扩展、Darcy流逆问题的32维KL展开；

**📈 对比分析**

与WALNUTS和传统BBVI比较：在低维情形下取得相当精度且计算速度明显优势；在50维Neal's funnel中收敛速度下降；在Darcy流逆问题中仅需约100次迭代即可捕捉双模态，表现出较好的性能；

**⚠️ 局限性**

局限性包括：在极高维或极度多模态目标下收敛速度慢，协方差可能陷入不稳定；需要较多混合成分以捕捉复杂结构；Monte Carlo噪声和步长调度对性能有显著影响；对非高斯目标分布的近似能力尚未给出理论保证。

---

## 279. Measuring and Aligning Abstraction in Vision-Language Models with Medical Taxonomies

**arXiv ID:** 2601.14827 | [PDF](https://arxiv.org/pdf/2601.14827v1)

**作者:** Ben Schaper `[一作]`, Cosmin I. Bercea `[通讯]` (Technical University of Munich)

**通讯引用:** 211 | [OpenAlex ID](https://openalex.org/A5086799838)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对胸部X光多标签分类，评估并改进视觉‑语言模型（VLM）在医学分层结构下的误判，特别是跨分支的灾难性抽象错误（CAE）。

**💡 创新点**

创新点在于提出专门衡量CAE的层级指标，并引入风险约束阈值与基于层级的径向嵌入微调（Taxonomy‑Aware Fine‑Tuning, TAF），实现对灾难性误判的主动抑制。

**🔧 技术方法**

技术包括：零射击多标签分类、层级重叠分数（HOS）与层级距离分数（HDS）评估、风险约束阈值选择、SigLIP对比损失微调、径向嵌入（RE）损失以及Kendall τ相关性分析。

**📊 数据集**

使用公开的 PadChest‑GR 胸片数据集（4,555张前视胸片，含分层注释）以及其构建的 117 节点医学分层体系。

**📈 对比分析**

与传统 flat F₁、以及 SOTA VLM（CLIP、PubMedCLIP、BiomedCLIP、MedCLIP、MedSigLIP）进行比较。风险约束阈值可将CAE率降至 ≤2%，而 TAF 在保持 F₁ 与 HDS/HOS 近似的同时，CAE率仅为1.6%，显著优于单纯零射击或仅阈值调优的方案。

**⚠️ 局限性**

局限性包括：①对其他医学影像模态与领域的泛化尚未验证；②径向嵌入方法需手工构造正负链，复杂度高；③阈值设定需人工预设 CAE 限值，缺乏自适应机制。

---

## 280. Efficient Beamforming for Discrete SIM-Aided Multiuser Systems Under Statistical CSI

**arXiv ID:** 2601.14803 | [PDF](https://arxiv.org/pdf/2601.14803v1)

**作者:** Yuhui Jiao `[一作]` (Shandong University), Yong Liang Guan `[通讯]` (Nanyang Technological University)

**通讯引用:** 15060 | [OpenAlex ID](https://openalex.org/A5100763983)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究在统计CSI下，利用多层离散相位的SIM实现多用户MISO系统的功率分配与相位优化，并提出基于WMMSE+AO的闭式迭代算法；

**💡 创新点**

首次考虑离散相位约束与统计CSI，推导平均可达率闭式表达式，利用ADMM与拉格朗日方法获得离散相位闭式解，显著降低复杂度并保持85%连续相位性能；

**🔧 技术方法**

采用WMMSE、交替优化、拉格朗日乘子、ADMM、统计CSI分析、Rayleigh衰落模型及离散相位投影等技术；

**📊 数据集**

使用仿真场景（2 GHz、λ/2元件尺寸、5λ厚度、用户分布在60–80 m环形区域等）进行验证，无真实数据集；

**📈 对比分析**

与SDR（CVX）方法对比，运行时间降低10–50倍；在1‑bit量化时仍获得85%连续相位性能；可达率随SIM层数增加呈递增趋势；

**⚠️ 局限性**

仅针对下行MISO且假设统计CSI，未考虑上行或时变通道；离散相位投影仅为理想化，未包含硬件非理想和功率损耗影响。

---

## 281. LocBAM: Advancing 3D Patch-Based Image Segmentation by Integrating Location Contex

**arXiv ID:** 2601.14802 | [PDF](https://arxiv.org/pdf/2601.14802v1)

**作者:** Donnate Hooft `[一作]`, Julia A. Schnabel `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了 patch‑based 3D 医学影像分割中位置上下文的重要性，提出并验证了一种轻量级 3D 注意力模块 LocBAM。

**💡 创新点**

创新点在于通过 1D 注意力门对宽、高、深三维方向的空间位置信息进行学习并融合，而非直接将坐标作为输入通道，从而显式编码全局位置上下文并在低 patch‑to‑volume coverage 时显著提升分割性能。

**🔧 技术方法**

采用 3D U‑Net 作为骨干网络，加入 LocBAM、CoordConv、Body Part Regression、HANet 相关技术，并利用 MONAI / nnU‑Net 训练框架进行实验；对位置信息使用 1×1 卷积融合 1D 注意力门。

**📊 数据集**

实验数据集包括 BTCV（多器官腹部分割）、AMOS22‑CT（腹部多器官）以及 KiTS23（肾肿瘤）三大公开 3D CT 数据集。

**📈 对比分析**

将 LocBAM 与基线无位置信息、CoordConv、经典后处理（Largest Component Filtering、Atlas Masking）进行对比；在 BTCV、AMOS22、KiTS23 上，LocBAM 的平均 Dice 分数提升约 0.3–0.7%，在极低 PtVC（0.06%）时提升高达 152%；同时训练过程更稳定，收敛速度更快。

**⚠️ 局限性**

局限性包括仅使用轴向的 Body Part Regression 进行位置信息编码，未考虑冠状或矢状面位置；对位置噪声的鲁棒性虽优于 CoordConv，但仍受限；实验仅在公开数据集验证，需在临床真实数据上进一步评估。

---

## 282. UBATrack: Spatio-Temporal State Space Model for General Multi-Modal Tracking

**arXiv ID:** 2601.14799 | [PDF](https://arxiv.org/pdf/2601.14799v1)

**作者:** Qihua Liang `[一作]` (Guangxi Normal University), Bineng Zhong `[通讯]` (Guangxi Normal University)

**通讯引用:** 15780 | [OpenAlex ID](https://openalex.org/A5058101262)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 UBATrack，一种利用 Mamba 状态空间模型的通用多模态跟踪框架，集成跨模态交互与时空建模。

**💡 创新点**

创新点在于引入 Spatio‑Temporal Mamba Adapter (STMA) 与 Dynamic Multi‑modal Feature Mixer (DMFM)，实现了高效的跨模态交互与时空上下文捕获，同时避免了全参数微调，显著降低训练成本。

**🔧 技术方法**

技术手段包括 Mamba 状态空间模型、Vision Transformer 作为编码器、FFT/多频域混合、Einstein 矩阵乘法、adapter‑tuning、以及动态混合层，配合 focal loss、L1 与 GIoU 作为损失。

**📊 数据集**

实验使用了多模态跟踪基准：RGB‑T (LasHeR、RGBT234、RGBT210)、RGB‑D (DepthTrack、VOT‑RGBD22)、RGB‑E (VisEvent) 等数据集。

**📈 对比分析**

与现有 SOTA 比较时，UBATrack 在所有三个模态的主基准上均超越对手；例如在 LasHeR 上精度 73.5%/成功率 60.1%，RGBT234 MPR 90.8%，RGBD DepthTrack 精度 67.7% 等，速度保持在 30+ FPS 左右。

**⚠️ 局限性**

局限性包括仍依赖大规模预训练视觉模型，适配不同硬件时仍需显存；对极端遮挡或长时延误场景的鲁棒性尚未彻底验证。

---

## 283. RANDSMAPs: Random-Feature/multi-Scale Neural Decoders with Mass Preservation

**arXiv ID:** 2601.14794 | [PDF](https://arxiv.org/pdf/2601.14794v1)

**作者:** Dimitrios G. Patsatzis `[一作]` (Scuola Superiore Medirionale), Constantinos Siettos `[通讯]` (Università degli Studi di Napoli Federico II)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于随机特征的神经解码器（RANDSMAP），能够在求解流形学习的反映问题时显式地保证质量守恒，并给出闭式解与误差界。

**💡 创新点**

创新点在于：① 将随机 Fourier 特征与数值分析中的双扩散映射、RBF 插值等经典方法在确定性极限下等价性理论化；② 在随机特征网络中加入线性等式约束，使得解可以显式满足质量守恒；③ 推导多尺度随机特征对应多高斯核的等价性，并给出截断误差界。

**🔧 技术方法**

使用技术包括：随机特征网络（RFNN）、随机 Fourier 特征、多尺度随机 Fourier 特征、奇异值分解 (SVD)、拉格朗日乘子法求闭式解、核岭回归、双扩散映射 (DDM)、k-NN 插值、Diffusion Maps 等。

**📊 数据集**

使用的数据集包括：1) 1D LWR 交通流模型 (M=400)、2) 2D 旋转 MRI 图像 (M=16384)、3) Hughes 模型的 2D 人群动态 (M=10000)，以及对比用的 Swiss Roll 与 S‑curve 等经典低维流形。

**📈 对比分析**

比较方法：与传统的 DDM、k‑NN 解码器以及不约束的 RFNN 进行对比；评估指标为训练/推理时间、训练集/测试集的相对 L₂、L∞ 误差及质量守恒误差。结果显示：RANDSMAP 在保持质量守恒（误差 ~ 10⁻⁸）的同时，训练与推理速度比 k‑NN 快数十倍，且 L₂、L∞ 误差均低于 k‑NN 和 DDM，尤其在处理尖锐特征（冲击波、图像边缘）时表现更佳。

**⚠️ 局限性**

局限性：① 需要构造随机特征矩阵，特征数较大时仍有计算与内存开销；② 对截断 SVD 的数值条件敏感，若保留的奇异值不足，质量守恒误差上界会增大；③ 对于极高维或样本稀疏的情况，随机特征的逼近精度仍受限；④ 本方法仅显式处理质量守恒，其他物理约束（如能量、正定性、对称性）仍需进一步扩展。

---

## 284. Visual and Cognitive Demands of a Large Language Model-Powered In-vehicle Conversational Agent

**arXiv ID:** 2601.15034 | [PDF](https://arxiv.org/pdf/2601.15034v1)

**作者:** Chris Monk `[一作]` (Exponent), Dara Gruber `[通讯]` (Google)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对在车内使用 Gemini Live 进行多轮对话时的视觉和认知负荷进行实车评估，并与免提电话、视觉转向导航和高负荷工作记忆任务进行对比。

**💡 创新点**

首次在真实道路上评估大型语言模型（LLM）对话代理的驾驶安全性，证明其语音交互的认知与视觉负荷与传统免提电话相当，且不超过高负荷认知任务。

**🔧 技术方法**

使用了检测反应任务（DRT）衡量认知负荷、Tobii Pro 3 眼动仪记录视觉关注、主观负荷量表（NASA‑TLX 等）以及对 Gemini Live 的使用体验问卷。

**📊 数据集**

数据来源为 32 名持照驾驶员在 Bowie, MD 公路上的驾驶记录，记录了 5 种任务的 DRT、眼动、主观评分和 Gemini Live 对话回合数。

**📈 对比分析**

通过重复测量 ANOVA（线性混合模型）比较任务条件。结果显示 Gemini Live（单轮和多轮）与免提电话的认知负荷均介于视觉导航和 OSPAN 之间，视觉关注均低于 2 秒阈值，TEORT 在单轮条件下低于 NHTSA 标准；多轮交互时认知负荷保持稳定，未出现累积增大。

**⚠️ 局限性**

局限包括：车载网络不稳定导致 Gemini Live 断连、任务持续时间受限、样本量较小、驱动员未自发开启交互、实际道路突发事件影响数据完整性，以及未考虑驾驶员个体差异和不同驾驶情境下的交互模式。

---

## 285. ExPrIS: Knowledge-Level Expectations as Priors for Object Interpretation from Sensor Data

**arXiv ID:** 2601.15025 | [PDF](https://arxiv.org/pdf/2601.15025v1)

**作者:** Marian Renz `[一作]` (German Research Center for Artificial Intelligence), Martin Atzmueller `[通讯]` (Osnabrück University)

**通讯引用:** 3456 | [OpenAlex ID](https://openalex.org/A5011835245)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种基于知识层期望的异构图神经网络，用于从RGB‑D传感器数据中递增构建3D语义场景图，并在移动机器人上进行实时验证。

**💡 创新点**

创新点在于：①将上下文观察与外部知识图（如ConceptNet）作为先验直接融入GNN的消息传递，形成期望偏置推理；②构建可扩展的层级3D语义场景图（3dssg）并实现动态增量式场景图预测；③实现多层次知识与感知的互操作，为长期机器人认知提供框架。

**🔧 技术方法**

采用了RGB‑D图像分割、3D点云几何特征提取（PointNet）、多层异构图结构、GraphSAGE/HGT图神经网络、距离阈值连边、跨层边消息传递，以及ConceptNet子图提取与嵌入（Numberbatch/自研）。

**📊 数据集**

主要使用常见室内3D数据集中的对象与场景（如Matterport3D、ScanNet等）进行离线实验，并在Mobipick移动机器人平台上进行实时评估。

**📈 对比分析**

与传统单帧、无先验的场景图预测模型对比，期望偏置GNN在“tidy‑up”实验中显著提升了语义一致性和鲁棒性，节点/边分类准确率提高，误判率降低，具体数值待后续公开。

**⚠️ 局限性**

局限性包括：①外部知识图噪声大，匹配物理场景困难；②缺乏大规模长期实验验证；③计算开销仍高，需进一步优化；④目前主要针对室内静态环境，未覆盖动态或户外场景。

---

## 286. Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control

**arXiv ID:** 2601.15015 | [PDF](https://arxiv.org/pdf/2601.15015v1)

**作者:** Jannis Becktepe `[一作]` (TU Dortmund University), Sebastian Peitz `[通讯]` (TU Dortmund University)

**通讯引用:** 1131 | [OpenAlex ID](https://openalex.org/A5049416946)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个全栈、可微分、零外部CFD依赖的强化学习主动流控基准库，包含13个2D/3D、单/多智能体环境，方便统一实验与比较。

**💡 创新点**

创新点在于：①首个完全可微的RL-AFC基准；②将GPU加速的PICT求解器嵌入PyTorch，完全无外部CFD；③支持多智能体和梯度式控制，提供统一API；④提供标准化训练/评估协议和公开模型。

**🔧 技术方法**

主要技术包括PyTorch框架、GPU加速PICT求解器、Gymnasium/PettingZoo接口、Stable‑Baselines3或TorchRL实现的PPO、SAC以及基于梯度的D‑MPC。

**📊 数据集**

使用的“数据集”是13个环境的自生成初始场（10个训练/验证/测试集），涵盖流体动力学参数如雷诺数、Prandtl数、Rayleigh数等。

**📈 对比分析**

通过统一的10次评估实验对比PPO、SAC及其多智能体版本，SAC在所有难度级别均显著优于PPO；MA‑SAC略优MA‑PPO；D‑MPC在梯度控制上也能实现相当的性能；转移实验显示2D→3D或小域→大域迁移效果良好。

**⚠️ 局限性**

局限性包括：实验随机种子有限导致统计不稳；需CUDA GPU才能快速运行；仅展示了模型自由RL和单一梯度控制；基线使用默认超参数，未做最优调优。

---

## 287. LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding

**arXiv ID:** 2601.15016 | [PDF](https://arxiv.org/pdf/2601.15016v1)

**作者:** Xiaodong Wang `[一作]` (Peking University), Peixi Peng `[通讯]` (Peking University)

**通讯引用:** 1929 | [OpenAlex ID](https://openalex.org/A5070755370)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LiViBench，首个面向交互直播视频的全模态基准，并基于该基准研发了LiVi-LLM-7B模型；

**💡 创新点**

创新点包括：①半自动标注工作流与多代理系统，利用种子问题驱动生成高质量问答；②视频-评论检索（VCR）模块，克服直播评论海量噪声；③两阶段指令调优策略，先用合成数据对齐交互域，再用人工校正数据细化。

**🔧 技术方法**

采用的技术包括：多模态大型语言模型（Qwen2.5-VL、InternVL3、Seed1.5-VL 等）、多代理协同生成、种子问题库构建、VCR 语音/音频编码与检索、ASR 语音转文本、Transformer 解码器融合音视频信息。

**📊 数据集**

使用的数据集有：LiViBench（3,168 条直播视频，包含视频、音频、语音、实时评论；3,175 题/答的多选 QA），以及约 100k 条直播视频用于生成指令调优数据（37,953 合成 + 11,180 人工校正）。

**📈 对比分析**

在 24 项任务、5 组评价（Coarse、Fine、Know、Reason、Livestream）上与 24 公开模型对比，LiVi-LLM-7B 在 LiViBench 上取得 64.4% 的总分，超过 GPT‑4o、Gemini‑2.5‑Pro 并与 72B 参数模型持平；在通用视频基准（Video‑MME、LongVideoBench、MLVU、VideoEval‑Pro）也实现最优或接近最优成绩。

**⚠️ 局限性**

局限性包括：对直播评论的处理仍需 VCR 过滤，未完全解决评论噪声与冗余；对极长视频的时序建模仍有限；与极大参数（>100B）专有模型的性能仍有差距；模型泛化到其他交互式媒体场景（如虚拟现实、AR）尚未验证。

---

## 288. Economic Warehouse Lot Scheduling: Approximation Schemes via Efficiently-Representable DP-Encoded Policies

**arXiv ID:** 2601.14993 | [PDF](https://arxiv.org/pdf/2601.14993v1)

**作者:** Danny Segev `[一作]` (Tel Aviv University), Danny Segev `[通讯]` (Tel Aviv University)

**通讯引用:** 1781 | [OpenAlex ID](https://openalex.org/A5044792131)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一套完整的算法框架，针对经济仓库批量调度问题（Economic Warehouse Lot Scheduling Problem）在常数个商品（n = O(1)）的情形下实现了1+ε 的多项式时间近似方案（PTAS），并通过进一步的分析和改进，在后续工作中突破了原有的 2-近似上限，获得 2−17/5000 近似。该工作首次实现了对动态补货策略的直接逼近，并给出了可在多项式空间内描述的近最优策略。

**💡 创新点**

创新点包括：
1) 提出了 Alignment 定理和 B‑aligned 策略结构，能够把任意动态策略逼近为周期性且具有可控成本的策略；
2) 通过频率类划分与分段对齐，将无限维的补货问题离散化为有限状态空间；
3) 结合凸松弛与动态规划，构造了 1+ε 近似的多项式时间算法；
4) 在保持空间可扩展性的同时，突破了长期以来的 2‑近似瓶颈，得到更优的近似比率。

**🔧 技术方法**

主要技术手段：
- 动态规划（DP）与状态压缩；
- 频率类划分（frequency classes）与分段对齐（break‑point alignment）；
- Alignment 定理（B‑aligned 结构的存在性证明）；
- 组合优化与凸松弛（convex relaxation）用于下界估计；
- 近似算法与误差传播分析，保证最终 1+ε 的成本上界。

**📊 数据集**

未使用具体实验数据集；所有结果均基于理论证明和算法分析，提供了算法复杂度与近似误差的严格上界。

**📈 对比分析**

通过理论分析证明，算法在时间复杂度 O(|I|^O(n) · 2^O(n^6/ε^5)) 内完成，得到 1+ε 近似；在后续工作中，采用相同思路实现了 2−17/5000 的近似，显著优于先前的 2‑近似。由于算法主要为理论证明，实际性能需进一步实验验证。

**⚠️ 局限性**

局限性：
1) 计算复杂度仍然极高，尤其是指数项 2^O(n^6/ε^5)，仅适用于常数个商品；
2) 算法实现依赖于多层猜测与枚举，实际工程化难度大；
3) 对于策略性版本（strategic version）尚未给出完整简化方案；
4) 仅在理论层面证明可行性，缺乏实验验证与性能评估。

---

## 289. Unsupervised Material Fingerprinting: Ultra-fast hyperelastic model discovery from full-field experimental measurements

**arXiv ID:** 2601.14965 | [PDF](https://arxiv.org/pdf/2601.14965v1)

**作者:** Moritz Flaschel `[一作]`, Ellen Kuhl `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用基于查表的“材质指纹”方法，在不需要优化的情况下从全场实验数据中快速识别弹性材料模型。

**💡 创新点**

创新在于将完整实验测量（位移场和全局反作用力）转化为指纹并利用预先生成的数据库进行模式识别，从而大幅降低计算成本并避免局部极小值问题。

**🔧 技术方法**

技术包括有限元预计算的指纹数据库、欧氏距离/余弦相似度匹配算法，以及基于FEniCSx的仿真。

**📊 数据集**

使用了五种软弹性材料（Elastosil三种配比、Sylgard 184、VHB 4905）在具有中心缺口的双轴拉伸实验中的全场位移和反作用力数据。

**📈 对比分析**

与传统的有限元模型更新（FEMU）相比，指纹匹配在在线阶段仅需秒级时间，速度提升约四个数量级；模型预测与实验吻合良好。

**⚠️ 局限性**

局限性在于数据库与实验几何/载荷耦合，改变实验条件需重新生成数据库；目前仅适用于各向同性、不可压缩、准静态的高弹性材料。

---

## 290. Improving Regret Approximation for Unsupervised Dynamic Environment Generation

**arXiv ID:** 2601.14957 | [PDF](https://arxiv.org/pdf/2601.14957v1)

**作者:** Harry Mead `[一作]` (University of Oxford), Nick Hawes `[通讯]` (University of Oxford)

**通讯引用:** 2486 | [OpenAlex ID](https://openalex.org/A5059686746)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出动态环境生成（DEGen）和最大负优势（MNA）两种方法，用于无监督环境设计，提升RL代理的泛化与零射性能。

**💡 创新点**

创新点在于通过在学生探索时逐步生成环境来加密教师奖励信号，解决信用分配难题；以及设计了MNA作为更准确的悔恨近似，可更有效挑选难度高的关卡。

**🔧 技术方法**

技术包括PPO训练学生与教师、基于价值函数的优势估计、动态生成算法、负优势回报计算。

**📊 数据集**

实验数据集为MiniGrid 13x13、17x17、21x21及其加钥匙/锁门版本的手工测试关卡。

**📈 对比分析**

与DR、SFL、PLR、ACCEL、Initial Gen等基线对比，DEGen在标准MiniGrid和钥匙版中都取得更高的零射平均回报与成功率，尤其在更大尺寸环境中优势更显著。

**⚠️ 局限性**

局限在于生成器对观察-关卡映射的依赖，无法直接处理复杂3D/机器人环境；此外DEGen训练成本约为传统方法的4倍。

---

## 291. What Should I Cite? A RAG Benchmark for Academic Citation Prediction

**arXiv ID:** 2601.14949 | [PDF](https://arxiv.org/pdf/2601.14949v1)

**作者:** Leqi Zheng `[一作]` (Tsinghua University), Ziyang Liu `[通讯]` (Tsinghua University)

**通讯引用:** 6684 | [OpenAlex ID](https://openalex.org/A5100655691)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CiteRAG基准，提供双粒度（列表级和位置级）学术引文预测任务，并发布了554k论文多层级语料库与7k/8.5k样本。

**💡 创新点**

创新点在于：①首次为引文预测设计可同时评估检索与生成的完整RAG框架；②引入三层级检索+对比学习微调的专属检索器；③提出面向位置的PACA指标和多维度多样性/幻觉评估。

**🔧 技术方法**

技术手段包括：检索-增强生成（RAG）框架；对比学习微调的Qwen3-Embedding-8B成为CitationRetriever-8B；基于Qwen3-4B/30B的生成器微调；多级召回融合与Reciprocal Rank Fusion；多任务评估指标。

**📊 数据集**

使用公开学术文献抓取自Google Scholar的554,719篇论文构成多层级语料库，Task1采样7,267例，Task2采样8,541例。

**📈 对比分析**

与传统检索器（TF‑IDF、BM25、BGE‑M3等）对比，CitationRetriever-8B在Recall@20/50、MRR@20/50提升超过50%；在RAG+微调组合下，CitationGenerator‑30B达到Task1 Recall@20≈0.076、NDCG@20≈0.367、Task2 PACA@20≈0.303，远超零样本闭源LLM。

**⚠️ 局限性**

局限性：仍依赖论文标题/摘要进行检索，部分领域覆盖不足；评测主要基于定量指标，未深入分析生成质量的语义一致性；模型规模与算力需求高，普适性受限。

---

## 292. TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control

**arXiv ID:** 2601.14945 | [PDF](https://arxiv.org/pdf/2601.14945v1)

**作者:** Yuteng Sun `[一作]` (Tsinghua University), Wei Yun Yau `[通讯]` (Institute for Infocomm Research)

**通讯引用:** 4771 | [OpenAlex ID](https://openalex.org/A5057104857)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出TIDAL框架，通过双频层次结构将大规模视觉语言动作模型的语义推理与高频动作执行分离，实现高频闭环控制；

**💡 创新点**

创新点在于单步流匹配与时序错位训练的结合，打破传统批量-执行模式，显著提升动态环境下的响应速度；

**🔧 技术方法**

使用技术包括大规模VLM后端、条件流匹配（单步Euler积分）、差分运动预测、双频调度以及时序错位训练策略；

**📊 数据集**

数据集涵盖RoboCasa官方长周期任务以及自制的动态拦截任务（2000条成功示例），并在暂停/非暂停协议下进行评估；

**📈 对比分析**

相较于传统开放式VLA基线，TIDAL在动态拦截任务中将成功率从0.31提升至0.61（Easy）/0.36（Hard），控制频率从约2.4 Hz提升至≈9 Hz，且在非暂停协议下仍保持三倍以上优势；

**⚠️ 局限性**

局限性包括在纯静态任务上略逊于基线，对语义意图的有效持续时间有限，超过约44步后性能显著下降。

---

## 293. LLM-Based Repair of C++ Implicit Data Loss Compiler Warnings: An Industrial Case Study

**arXiv ID:** 2601.14936 | [PDF](https://arxiv.org/pdf/2601.14936v1)

**作者:** Chansong You `[一作]` (SAP Labs), Jingun Hong `[通讯]` (SAP Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型自动修复大型 C++ 项目中的隐式数据丢失编译警告，并通过 LSP、Tree-sitter 以及自洽性推理生成高质量修补补丁。

**💡 创新点**

创新性地结合 LSP 的类型信息、Tree-sitter 的精确代码提取和自洽性多路径推理来决定是否需要范围检查，从而在保持性能的前提下生成安全可靠的修复方案。

**🔧 技术方法**

采用 LLM（如 GPT‑4 Turbo）+ LSP（clangd）+ Tree‑sitter + 自洽性采样推理 + git‑diff 风格输出 + 迭代 patch 验证 等技术。

**📊 数据集**

在 SAP HANA 代码库（约 36M 行）中抽取 110 条代码片段共 135 条隐式转换警告作为实验数据集。

**📈 对比分析**

通过与基线的通用类型转化函数（带范围检查）和人工最佳修复进行对比，LLM 生成的修复在警告接受率约 80% 以上，同时将额外指令数减少约 50% 以上，性能接近人工最佳方案。

**⚠️ 局限性**

存在无法处理宏展开代码、意图性数据丢失以及需要设计层面改动的局限，缺乏宏上下文和全局语义的支持。

---

## 294. Generative Artificial Intelligence, Musical Heritage and the Construction of Peace Narratives: A Case Study in Mali

**arXiv ID:** 2601.14931 | [PDF](https://arxiv.org/pdf/2601.14931v1)

**作者:** Nouhoum Coulibaly `[一作]` (RobotsMali AI4D Lab), Ousmane Goro `[通讯]` (Conservatoire des Arts et Métiers Multimédia Balla Fasseké)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在马里开展工作坊，利用生成式人工智能协同创作融合本土语言与传统乐器的音乐，以此推动和平叙事与文化遗产复兴

**💡 创新点**

创新点在于以文化敏感的协作框架把AI工具嵌入创作流程，既保留传统音乐特色，又实现多语言多乐器的融合与动态风格过渡，形成具备叙事性的音乐叙事结构

**🔧 技术方法**

使用ChatGPT、Gemini进行文本生成与翻译，Suno AI生成音乐作品，结合人工提示工程实现对节奏、乐器、音调、文化符号的精细控制

**📊 数据集**

主要数据集为马里多种官方语言（Bambara、Fulfulde、Tamasheq、Songhai、Dogon）语料与传统乐器录音样本；未使用公开大型语料库，强调本土语料不足是主要挑战

**📈 对比分析**

通过主题分析（Braun‑Clarke）和问卷量化（Likert量表）评估创作效果，结果显示参与者满意度平均4.15/5，85%认为作品具有文化真实性；与AI经验无显著差异，未设对照组，尚缺统计显著性验证

**⚠️ 局限性**

局限性包括：本土语言语料稀缺导致发音与语调不准；算法审查限制特定词汇；真实性与创新的张力导致部分作品缺乏音乐连贯性；版权与伦理问题未完全解决

---

## 295. Vision-Language Models on the Edge for Real-Time Robotic Perception

**arXiv ID:** 2601.14921 | [PDF](https://arxiv.org/pdf/2601.14921v1)

**作者:** Sarat Ahmad `[一作]` (University of Leeds), Syed Ali Raza Zaidi `[通讯]` (University of Leeds)

**通讯引用:** 4811 | [OpenAlex ID](https://openalex.org/A5011903629)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将Vision‑Language Models（VLM）部署在ORAN/MEC边缘节点，并在Unitree G1仿人机器人上进行实时感知与交互实验。

**💡 创新点**

首次在真实机器人与边缘网络环境中进行大规模VLM端到端评估，比较云端与边缘部署的准确率与延迟，并展示轻量化模型与大模型在边缘的权衡与可组合性。

**🔧 技术方法**

使用WebRTC实时通信层、FastAPI推理服务、LLaMA‑3.2‑11B‑Vision‑Instruct 4‑bit NF4 量化模型、Qwen2‑VL‑2B‑Instruct 轻量模型、NVIDIA L4 GPU 以及贪婪解码与早停策略。

**📊 数据集**

使用 Robo2VLM 机器人感知基准以及自行收集的 200 组人机交互问答数据集。

**📈 对比分析**

通过准确率与端到端延迟两项指标比较；边缘部署的 LLaMA‑3.2‑11B 与云端相比，延迟降低约5%，准确率略有提升；Qwen2‑VL‑2B 在边缘实现子秒延迟，但准确率约比云端低 23%。

**⚠️ 局限性**

仅在单一实验室环境与单一硬件（Unitree G1 + NVIDIA L4）验证，未覆盖更大规模网络、多机器人协同；模型压缩虽提升效率但牺牲了一定准确度；自回归文本生成是主要瓶颈，需进一步优化。

---

## 296. Mixture-of-Experts Models in Vision: Routing, Optimization, and Generalization

**arXiv ID:** 2601.15021 | [PDF](https://arxiv.org/pdf/2601.15021v1)

**作者:** Adam Rokah `[一作]` (Uppsala University), Sourav Sharan `[通讯]` (Scaleout Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在CIFAR‑10图像分类任务上，研究并对比了密集分类头、SoftMoE和SparseMoE的预测性能、专家利用率、泛化能力以及推理效率。

**💡 创新点**

创新点包括：① 在受控模型容量下系统性评估MoE在视觉任务中的优化与泛化行为；② 结合Hessian曲率（最大特征值、迹）和损失曲面扰动分析，揭示MoE的局部与全局优化特征；③ 量化理论上稀疏MoE的计算优势与实际推理效率之间的差距。

**🔧 技术方法**

使用技术包括：ResNet‑18共享骨干、SoftMoE与SparseMoE（Noisy Top‑k gating）分类头、负载平衡正则化、梯度下降训练、Hessian向量乘法求最大特征值与迹、损失曲面沿主特征向量扰动分析，以及GPU/CPU推理时延与吞吐量测评。

**📊 数据集**

使用数据集：CIFAR‑10（60k张32×32彩色图像，10个类别）。

**📈 对比分析**

比较方法：在相同参数预算下训练 Dense、SoftMoE 与 SparseMoE，使用验证准确率、训练/验证收敛速度、Hessian曲率指标以及 GPU/CPU 推理时延与吞吐量进行对比。结果显示：MoE模型在验证准确率上略优0.3–0.4%，但训练收敛更慢；SoftMoE 的曲率更尖锐但泛化与其他模型相当；SparseMoE 在理论上应更快，但实际推理速度反而比 Dense/SoftMoE 更慢。

**⚠️ 局限性**

局限性：① 条件计算的路由与聚合开销导致 SparseMoE 在当前规模和批次下未实现速度提升；② 仅在同质的CIFAR‑10数据上评估，缺乏对高异质任务的验证；③ 只关注分类头，未探究整个网络层级的MoE效应。

---

## 297. Multimodal Rumor Detection Enhanced by External Evidence and Forgery Features

**arXiv ID:** 2601.14954 | [PDF](https://arxiv.org/pdf/2601.14954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 298. Obscuring Data Contamination Through Translation: Evidence from Arabic Corpora

**arXiv ID:** 2601.14994 | [PDF](https://arxiv.org/pdf/2601.14994v1)

**作者:** Chaymaa Abbas `[一作]` (American University of Beirut), Mariette Awad `[通讯]` (American University of Beirut)

**通讯引用:** 6377 | [OpenAlex ID](https://openalex.org/A5008382926)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多语言设置下，通过在 Arabic 版本的测试集上对开源 LLM 进行微调并评估其在原始 English 基准上的表现，研究数据污染对模型评估的影响，并提出翻译感知污染检测（TACD）方法；

**💡 创新点**

创新点在于揭示翻译可掩盖传统污染信号，同时通过多语言视图与结构扰动（如答案重排）结合的 TACD，提供一种无需访问训练语料、可捕捉翻译后残留污染的检测机制；

**🔧 技术方法**

采用了 TS‑Guessing（含答案重排）和 Min‑K++ 概率分析等行为与分布式污染探测技术，并实现了 TACD 的跨语言翻译与答案扰动策略；

**📊 数据集**

实验使用 MMLU（多选题）和 XQuAD/MLQA（提取式 QA）数据集，分别构造 0%、10%、50% 和 100% 的 Arabic 训练混合比例；

**📈 对比分析**

通过比较各污染比例下的宏平均准确率、IDR、EM、ROUGE‑L、Min‑K++ AUROC 等指标，发现污染比例升高时模型在 English 评估上性能提升，传统指标被翻译抑制但 TACD 能有效揭示污染；

**⚠️ 局限性**

局限性包括 TACD 只能提供污染一致性信号而非确切污染标签；可能与模型或提示导致的跨语言不变性混淆；仅在有限的语言（EN、AR、FR）和两类基准上验证，未覆盖所有污染形式；

---

## 299. Random Gilbert-Varshamov Codes for Joint Source-Channel Coding

**arXiv ID:** 2601.14987 | [PDF](https://arxiv.org/pdf/2601.14987v1)

**作者:** AmirPouya Moeini `[一作]`, Albert Guillén i Fàbregas `[通讯]` (University of Cambridge)

**通讯引用:** 3132 | [OpenAlex ID](https://openalex.org/A5005125538)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种基于随机Gilbert‑Varshamov编码的联合源‑信道编码（JSCC）随机编码技术，构造了满足距离约束的码字集合，并证明该随机编码族能够同时达到随机编码指数与删失指数的最大值。

**💡 创新点**

创新点在于将递归距离约束的Gilbert‑Varshamov构造扩展到JSCC场景，提出了一套通用的距离函数与最小距离参数，并通过统一的解码度量实现了随机编码指数与删失指数的双重最优性。

**🔧 技术方法**

主要技术包括类型方法、随机Gilbert‑Varshamov码生成、基于类型的距离度量、最小距离约束、以及两种通用解码度量（最大互信息与期望对数似然比）。

**📊 数据集**

由于研究属于理论编码理论，不涉及具体实验数据集，所有结果均在离散无记忆源与通道的假设下通过信息熵与KL散度的分析推导。

**📈 对比分析**

通过与已知的随机编码和删失指数表达式进行比较，证明该构造在任意源/信道分布下都能实现与两种指数相同的上界；因此其误差指数优于传统单类编码方法，且具有通用性。

**⚠️ 局限性**

局限性包括：仅适用于离散无记忆源/通道，构造复杂度较高；实现时需保证足够大的块长以满足类型近似；以及对距离约束的选择仍需经验性调参，实际性能在非理想条件下尚未验证。

---

## 300. Information mechanics: conservation and exchange

**arXiv ID:** 2601.15028 | [PDF](https://arxiv.org/pdf/2601.15028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 301. A Comprehensive Benchmark of Language Models on Unicode and Romanized Sinhala

**arXiv ID:** 2601.14958 | [PDF](https://arxiv.org/pdf/2601.14958v1)

**作者:** Minuri Rajapakse `[一作]` (Informatics Institute of Technology), Ruvan Weerasinghe `[通讯]` (Informatics Institute of Technology)

**通讯引用:** 517 | [OpenAlex ID](https://openalex.org/A5084628821)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于一份包含Unicode与罗马化Sinhala句子的评测集，对多款公开与闭源大型语言模型进行基准测试，探讨脚本差异对模型性能的影响。

**💡 创新点**

创新点在于提供首个同时覆盖Unicode和罗马化Sinhala的内在评测（perplexity与句子完成）并揭示同一模型在不同脚本上的显著性能分化。

**🔧 技术方法**

采用了Transformer架构模型（如Mistral、Llama等）、基于LaBSE的句子嵌入与K‑Means聚类构建评测集，并通过perplexity和人工3分量表进行评估。

**📊 数据集**

使用了自建的1000句平行语料（约200句评测子集），其中包括来自博客、社交媒体的Unicode和手工转写的罗马化文本。

**📈 对比分析**

通过perplexity对公开模型进行量化比较，Mistral‑Nemo‑Base‑2407在Unicode上最低2.19；在罗马化上Mistral‑7B‑v0.3最低74.76；闭源模型通过句子完成得分，Gemini‑1.5‑pro和DeepSeek在Unicode上得分最高，Claude‑3.5‑Sonnet在罗马化上表现最佳。

**⚠️ 局限性**

局限在于评测集规模仅200句，未覆盖所有罗马化变体；闭源模型评价仅由单一评审完成；未考察跨脚本混写文本。

---

## 302. On the Effectiveness of Mempool-based Transaction Auditing

**arXiv ID:** 2601.14996 | [PDF](https://arxiv.org/pdf/2601.14996v1)

**作者:** Jannik Albrecht `[一作]` (Ruhr University Bochum), Ghassan Karame `[通讯]` (Ruhr University Bochum)

**通讯引用:** 6052 | [OpenAlex ID](https://openalex.org/A5059087800)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文分析了在 Bitcoin 和 Ethereum 中使用 mempool 审计检测恶意矿工的审查与交易位移攻击的有效性。

**💡 创新点**

创新点在于首次量化 mempool 审计的误判概率（>25%）与成功率（99.9%），并证明批量排序公平性方案只能在有限交易子集上提供强公平保障。

**🔧 技术方法**

采用了基于 mempool 观测数据的审计方法，结合统计概率分析与模拟实验，评估不同时间间隔与观察者覆盖率对检测准确率的影响。

**📊 数据集**

实验使用了公开 mempool 数据（如 Mempool.space 采集的 Bitcoin/Ethereum mempool 记录）以及仿真生成的交易序列。

**📈 对比分析**

与现有基于规则的审计工具相比，本文方法在误判率和成功率上提供了量化指标；在实验中，正确检测率可达 99.9%，误判率在特定配置下超过 25%。

**⚠️ 局限性**

局限性包括：仅针对在所有观察者接收且相隔至少 30 秒的交易；对高并发或短间隔交易的检测效果未知；实验环境与真实矿工行为可能存在差异。

---

## 303. RadixMLP -- Intra-batch Deduplication for Causal Transformers

**arXiv ID:** 2601.15013 | [PDF](https://arxiv.org/pdf/2601.15013v1)

**作者:** Michael Feil `[一作]` (Baseten), Julius Lipp `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在批量因果Transformer推理中，消除共享前缀的重复计算，实现一次前向传播内的无状态去重。

**💡 创新点**

提出RadixMLP：利用前缀trie对位置层（MLP、LayerNorm、投影、嵌入）进行无状态去重，并通过gather/scatter索引保持梯度兼容性，避免传统KV缓存的状态管理。

**🔧 技术方法**

使用prefix trie映射、gather/scatter索引、ragged批处理、FlashAttention+CUDA核优化、CPU异步调度、与PyTorch/Candle/TEI集成。

**📊 数据集**

实验基于MS MARCO v1.1查询–段落对、合成前缀/后缀批次，以及Qwen3-0.6B/4B/8B模型。

**📈 对比分析**

在NVIDIA H100上与TEI、Candle、vLLM对比：合成前向推理中可达5×加速，真实重排工作中1.44–1.59×速度提升，显著优于无RadixMLP且与vLLM竞争。

**⚠️ 局限性**

局限性：低前缀重用批次几乎无收益；长序列（>32K）时注意力占比增大，收益下降；目前主要用于预填充，生成阶段受限；需额外索引内存并依赖批量无状态设计。

---

## 304. DWPP: Dynamic Window Pure Pursuit Considering Velocity and Acceleration Constraints

**arXiv ID:** 2601.15006 | [PDF](https://arxiv.org/pdf/2601.15006v1)

**作者:** Fumiya Ohnishi `[一作]` (Keio University), Masaki Takahashi `[通讯]` (Keio University)

**通讯引用:** 3346 | [OpenAlex ID](https://openalex.org/A5037149546)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文提出了动态窗口纯追踪（DWPP）方法，在路径跟踪时将速度和加速度约束显式纳入命令速度计算，直接生成可执行的速度指令。

**💡 创新点**

创新点在于：① 在速度-角速度空间中构造动态窗口；② 通过求解与 ω = κv 直线最近点的闭式解来得到最优命令速度；③ 实现了无需后处理的速度/加速度约束满足。

**🔧 技术方法**

采用的技术包括：纯追踪（Pure Pursuit）、动态窗口概念、闭式几何求解、ROS 2 Nav2 框架集成，以及基于Gazebo/TurtleBot3的仿真和基于WHILL CR的实机实验。

**📊 数据集**

实验数据集：在无障碍室内使用 WHILL CR 机器人，生成三条不同拐角角度（45°、90°、135°）的路径（A、B、C）；仿真中使用 TurtleBot3，调节不同 look‑ahead 距离；未使用公开数据集。

**📈 对比分析**

与传统 PP、APP、RPP 通过约束违规率、均值/最大横向误差、行驶时间等指标比较，DWPP 在所有路径下实现了零约束违规、最小的均值/最大误差，但行驶时间略长，验证了其更优的跟踪性能。

**⚠️ 局限性**

局限性：需要手动调节 look‑ahead 距离以平衡误差与效率；方法仅关注路径跟踪，未内置障碍物规避；对不同机器人动力学参数的适应性仍需进一步验证。

---

## 305. Two-Class Joint Source-Channel Coding: Expurgated Exponents with i.i.d. Distributions

**arXiv ID:** 2601.14985 | [PDF](https://arxiv.org/pdf/2601.14985v1)

**作者:** Seyed AmirPouya Moeini `[一作]` (University of Cambridge), Albert Guillén i Fàbregas `[通讯]` (University of Cambridge)

**通讯引用:** 3132 | [OpenAlex ID](https://openalex.org/A5005125538)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文研究了离散无记忆源和信道的联合源-信道编码的删去指数，证明在 i.i.d. 随机编码下，采用两类分区（源序列按类型分组，每类使用不同的码字分布）可实现不小于最优单类编码的指数。

**💡 创新点**

创新点在于首次将两类分区与单类编码在删去指数层面直接比较，证明了两类分区在 i.i.d. 编码中至少不劣于单类编码，并且在多种情形下两者指数相等，提示分区可能并未带来优势；同时将 Csiszár 的两指数框架推广到 i.i.d. 代码集。

**🔧 技术方法**

主要技术包括基于源类型复制和逐类型删去的 Csiszár–Körner–Marton 形式的指数推导、使用 Bhattacharyya 距离的误差上界、Hölder 不等式以及对 ρ 参数的凸包/凹包分析，以得到两类分区下的删去指数上界。

**📊 数据集**

本文未使用具体实验数据集，而是在理论上给出通用的源-信道模型和数值示例（如二元源与非对称信道），以验证理论推导的正确性。

**📈 对比分析**

与单类编码的比较方法是取两类分区下的指数与单类编码下的最优指数取上确界并比较，结果表明两类分区的指数至少与单类相等；在数值实验中，两者在大部分参数设置下相等，偶有单类优于两类。

**⚠️ 局限性**

限制在于仅证明了单向不劣性，未证明两类分区在所有情形下一定优于或等于单类；另外，目前仅在 i.i.d. 编码下给出结论，尚未扩展到常组成编码；以及结论仅适用于离散无记忆源与信道，需进一步研究更一般模型。

---

## 306. Parallel Collaborative ADMM Privacy Computing and Adaptive GPU Acceleration for Distributed Edge Networks

**arXiv ID:** 2601.14980 | [PDF](https://arxiv.org/pdf/2601.14980v1)

**作者:** Mengchun Xia `[一作]` (Tibet University), Pingzhi Fan `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 21334 | [OpenAlex ID](https://openalex.org/A5101880047)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种面向分布式边缘网络的三阶段并行协同 ADMM 隐私计算算法（3P‑ADMM‑PC2），通过将 Paillier 同态加密与量化技术相结合，实现了在不泄露原始数据的前提下完成稀疏信号恢复任务；同时提出 GPU‑加速版 3P‑ADMM‑PC2，通过将大整数模幂运算拆解为小模空间并行计算，显著降低加解密及同态运算的计算负载；

**💡 创新点**

创新点包括：①设计了无需负数表示的量化映射，保证了量化后整数可直接用于 Paillier 加密并不破坏同态性质；②采用模 CRT 分解与 FFT 多项式乘法实现大整数模幂的 GPU 并行计算；③提出了主从节点协同加密/解密的三轮通信协议，进一步平衡了计算与通信开销；

**🔧 技术方法**

核心技术：分布式 ADMM、Paillier 同态加密、定点量化、CRT 模分解、FFT 基多项式乘法、GPU 并行模运算（Barrett 降低）以及多节点协同加密/解密协议；

**📊 数据集**

实验数据集主要包括：①随机高斯矩阵（M×N 维度从 3×3 到 10000×65536）用于验证 MSE 与收敛；②真实电力系统 13569 节点的 MATPOWER 电网数据，用于网络拓扑重建评估 AUROC/AUPRC；

**📈 对比分析**

与集中式 ADMM、传统分布式 ADMM、DP‑ADMM 等方法对比，3P‑ADMM‑PC2 在 MSE 上与无隐私分布式 ADMM 差距 ≤10⁻¹⁴，接近集中式 ADMM；在加密/解密时 GPU‑加速实现比 CPU 快 5‑10 倍，整体迭代时间显著下降，且在多节点场景下仍保持较低的通信延迟；

**⚠️ 局限性**

局限性：①大密钥长度（≥2048 位）导致模运算仍较慢，通信量随节点数增加而显著；②主节点仍承担较多计算负担，需进一步优化任务划分与负载均衡；③量化引入微量误差，需在极低误差要求的应用中谨慎选择 Δ；④实现复杂度高，需结合 GPU 与 CPU 的多核协同编程。

---

## 307. InstructTime++: Time Series Classification with Multimodal Language Modeling via Implicit Feature Enhancement

**arXiv ID:** 2601.14968 | [PDF](https://arxiv.org/pdf/2601.14968v1)

**作者:** Mingyue Cheng `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27290 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

将时间序列分类转化为多模态生成任务，利用语言模型通过指令文本生成文本标签，实现对连续数值、文本上下文与指令的联合推理。

**💡 创新点**

创新点包括：1）InstructTime框架，将数值序列离散化为符号化token并与指令文本对齐；2）引入生成式自监督预训练和指令微调；3）InstructTime++扩展，自动挖掘统计与视觉隐式特征并转化为自然语言描述，进一步补偿LLM对时序模式的诱导不足。

**🔧 技术方法**

采用的技术主要有：VQ‑Net离散化、对齐投影层（MLP）、生成式自监督预训练、指令微调（SFT）、统计特征提取、视觉图像描述（VLM）、大型语言模型（如Qwen3）以及多模态prompt工程。

**📊 数据集**

实验使用的基准数据集包括EEG、ECG、HAR、FD、RWC、EP、SAD等UEA多变量时间序列分类数据集。

**📈 对比分析**

与传统CNN、Transformer、FormerTime、MiniROCKET、TimeMAE、GPT‑As‑Classifier等基线方法比较，InstructTime++在绝大多数数据集上获得最高或相近的准确率和F1，表现出更强的鲁棒性和跨域泛化能力。

**⚠️ 局限性**

限制：离散化过程可能导致信息丢失；需要对齐投影与预训练的额外训练开销；模型对极长序列或高维通道的适应性尚未验证；隐式特征提取仍依赖手工设定的工具包，可能缺乏普适性。

---

## 308. VCNAC: A Variable-Channel Neural Audio Codec for Mono, Stereo, and Surround Sound

**arXiv ID:** 2601.14960 | [PDF](https://arxiv.org/pdf/2601.14960v1)

**作者:** Florian Grötschla `[一作]` (Amazon AGI), Andreas Schwarz `[通讯]` (Amazon AGI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种可变通道神经音频编码器 VCNAC，能够在单一模型中同时处理单声道、立体声和 5.1 环绕声。

**💡 创新点**

通过并行通道流、交叉通道注意力以及共享码本的设计，实现了多通道兼容与统一潜在空间，消除了传统编码器需要单独模型或通道填充的局限。

**🔧 技术方法**

使用了残差向量量化、Transformer 音频自编码器（TAAE）、mid/side 与低频（LFE）处理的多尺度梅尔损失、GAN 判别器以及共享卷积权重和通道位置嵌入等技术。

**📊 数据集**

训练使用 LibriTTS、LibriVox、FMA‑small 以及基于这些数据的合成 5.1 环绕音频，评估时采样自相同数据集及四部开源电影。

**📈 对比分析**

与 Opus、DAC、EnCodec、SNAC 等编码器在 PESQ、SI‑SNR、SI‑SDR、MUSHRA 等指标上对比，VCNAC 在 7.9 kbit/s 下实现了更高的语音和音乐质量，在环绕声的空间保真度上仅次于 EnCodec，总体表现优于对手且比 SNAC 低近一半比特率。

**⚠️ 局限性**

训练时合成的 5.1 环绕音频缺乏真实场景的响度平衡与空间真实性，导致后置声道的重建效果与真实数据相比有所欠缺，且模型对极低能量通道（如 LFE）敏感。

---

## 309. CorpusQA: A 10 Million Token Benchmark for Corpus-Level Analysis and Reasoning

**arXiv ID:** 2601.14952 | [PDF](https://arxiv.org/pdf/2601.14952v1)

**作者:** Zhiyuan Lu `[一作]` (Tongyi Lab, Alibaba Group), Fei Huang `[通讯]` (Tongyi Lab, Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 CorpusQA 基准，用合成数据评估大规模语料库级推理任务，探索基线 LLM、RAG 与记忆代理在不同上下文长度下的表现，并展示基于合成数据微调提升模型能力。

**💡 创新点**

① 大规模 10M token 语料库级推理基准；② 通过程序化数据合成与 NL2SQL 实现答案可验证；③ 明确展示传统 RAG 在高信息散布任务中崩溃，记忆代理更稳健。

**🔧 技术方法**

程序化数据合成框架、Schema extraction、LLM-as-a-Judge、NL2SQL、RAG、记忆代理（Memory Agent）等技术。

**📊 数据集**

CorpusQA（金融、教育、房地产等多域，12亿 token 规模）以及外部长上下文基准 LongBenchV2、FRAMES 用于微调效果评估。

**📈 对比分析**

在 128k、1M、4M、10M 上下文长度下，对比基线 LLM、RAG 与记忆代理：基线 LLM 在 128k 仍能取得 80% 以上准确率，但在 1M 以上显著下降；RAG 在 1M+ 上几乎崩溃（准确率 < 20%）；记忆代理在 4M、10M 下仍能保持 7-25% 的准确率，显示其对大规模推理的相对鲁棒性。

**⚠️ 局限性**

主要局限：评估依赖 LLM-as-a-Judge，缺乏完全确定的基准；实验未覆盖更复杂的深度代理架构，未来仍需进一步验证。

---

## 310. Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness

**arXiv ID:** 2601.14950 | [PDF](https://arxiv.org/pdf/2601.14950v1)

**作者:** Yufei Song `[一作]`, Leo Yu Zhang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于像素置信度的漏洞感知对抗训练框架EroSeg，用于生成对语义分割模型更有效的对抗样本并提升其鲁棒性。

**💡 创新点**

创新点在于①利用像素级分类置信度识别最易受扰动的敏感像素；②采用渐进式扰动传播策略，逐步扩散至更稳定像素，从而打破模型的上下文一致性修正机制；③在攻击和训练中加入前景加权，提升攻击对关键区域的影响。

**🔧 技术方法**

主要技术包括像素置信度阈值筛选、递增阈值的渐进传播公式、前景/背景加权损失、对抗训练的min‑max优化和对抗样本生成的PGD类迭代。

**📊 数据集**

在PASCAL VOC和Cityscapes两个主流数据集上，对DeepLabV3和PSPNet两种语义分割模型进行实验。

**📈 对比分析**

与PGD、CosPGD、SegPGD、RP‑PGD等四种主流攻击方法以及对应的对抗训练效果进行对比。EroSeg在3–50次迭代的攻击中始终保持最高mIoU下降（例如DeepLabV3在3次迭代下mIoU降至73.53%），并在对抗训练中取得最高平均mIoU（EroSeg3‑AT 30.86%，EroSeg7‑AT 34.00%），显著优于其他方法。

**⚠️ 局限性**

目前仅针对语义分割任务，未将EroSeg推广到目标检测等结构不同的任务，未来需进一步扩展。

---

## 311. Communication-Efficient Multi-Modal Edge Inference via Uncertainty-Aware Distributed Learning

**arXiv ID:** 2601.14942 | [PDF](https://arxiv.org/pdf/2601.14942v1)

**作者:** Hang Zhao `[一作]` (Hong Kong University of Science and Technology), Khaled B. Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 43529 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种三阶段通信感知的多模态边缘推理框架，在设备端进行本地自监督预训练，服务器端做不确定性感知的监督微调，并在推理时根据不确定性动态请求重传，显著降低训练和推理的通信开销，提升鲁棒性。

**💡 创新点**

创新点在于（1）完全本地化的多模态自监督预训练实现零通信初始化；（2）利用证据深度学习实现不确定性感知融合；（3）基于不确定性阈值的自适应重传策略，精细控制通信与准确率的折中。

**🔧 技术方法**

采用多模态自监督损失（交叉/内部相关性约束）、证据深度学习（Dirichlet分布+不确定性估计）、多模态融合算子、基于量化阈值的重传策略；实现基于ResNet‑18的语义编码器、线性头以及端到端的JSCC基线。

**📊 数据集**

主要使用室内场景分类数据集NYU‑Depth V2与SUN RGB‑D，包含RGB与深度两模态，测试无线AWGN与瑞利衰落信道下的鲁棒性。

**📈 对比分析**

与SimCLR、Barlow Twins、无预训练以及传统JSCC基线比较。结果显示：预训练阶段可将达到40%–50%准确率所需通信轮次缩短约50%，最终准确率提升至~0.58；在低SNR、动态信道或单模态降质时，证据融合+重传方案在保持平均重传率≈10% 的同时相对基线提升≈2%–3%的准确率。

**⚠️ 局限性**

局限包括：仅验证单模态对比实验，缺乏对更复杂多模态/多任务的验证；重传策略基于阈值而非学习的动态调度，可能在极端信道环境下不最优；系统仍依赖同步通信轮次，异步/延迟场景待进一步研究。

---

## 312. Interoperable Architecture for Digital Identity Delegation for AI Agents with Blockchain Integration

**arXiv ID:** 2601.14982 | [PDF](https://arxiv.org/pdf/2601.14982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 313. Operationalising DAO Sustainability KPIs: A Multi-Chain Dashboard for Governance Analytics

**arXiv ID:** 2601.14927 | [PDF](https://arxiv.org/pdf/2601.14927v1)

**作者:** Silvio Meneguzzo `[一作]` (University of Turin), Giuseppe Destefanis `[通讯]` (University College London)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个生产级的多链分析管道和交互式仪表盘，实时计算并展示DAO治理可持续性KPI及0-12分数。

**💡 创新点**

将理论KPI框架落地为可审计、可复现的系统，并公开源码、数据模式与阈值，使跨链治理风险评估透明且可解释。

**🔧 技术方法**

后端采用FastAPI、PostgreSQL与Celery；前端使用Next.js 14+React、Recharts；核心评分逻辑在浏览器执行，数据统一化通过JSON schema完成。

**📊 数据集**

使用约50个活跃DAO的快照（6930条提案、317,317个独立投票地址），覆盖以太坊、Optimism、BNB Smart Chain、Arbitrum、Polygon等主流EVM链。

**📈 对比分析**

前端在浏览器中完成所有KPI计算与排序，交互响应流畅；后端仅提供读取接口，评估显示正确性高、响应性好，支持跨链对比与导出。

**⚠️ 局限性**

仅基于链上数据，忽略离链讨论与身份信息；抽取器质量不稳定；阈值固定不可配置；对不同DAO治理模式不够灵活；不支持非EVM链；未验证分数与治理失败的相关性。

---

## 314. SpatialV2A: Visual-Guided High-fidelity Spatial Audio Generation

**arXiv ID:** 2601.15017 | [PDF](https://arxiv.org/pdf/2601.15017v1)

**作者:** Yanan Wang `[一作]` (Shandong University), Tian Gan `[通讯]` (Shandong University)

**通讯引用:** 1418 | [OpenAlex ID](https://openalex.org/A5100654958)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了大规模的 BinauralVGGSound 视频-双声道音频数据集，并提出 SpatialV2A 双分支空间视频转音频模型。

**💡 创新点**

创新点在于利用可视化引导的音频空间化模块和基于 Conditional Flow Matching 的双声道生成框架，既保持语义、时序一致，又显著提升空间真实性。

**🔧 技术方法**

采用了 CLIP+Synchformer 的多模态联合编码、ACL 定位热图提取空间特征、PseudoBinaural 伪双声道生成、VAE+BigVGAN 语音解码以及 Conditional Flow Matching 双通道训练。

**📊 数据集**

数据集方面使用改造自 VGGSound 的 187k 条视频，并通过 PseudoBinaural 生成双声道音频，形成 BinauralVGGSound。

**📈 对比分析**

在分布匹配、语义/时序对齐、音质和空间指标上均优于 MMAudio、ReWaS、Frieren、Seeing&Hearing 等基线，并在 MOS 评估中取得最高的音质与空间感知得分。

**⚠️ 局限性**

局限性包括依赖伪双声道转换而非真实录音、仅处理约 10 秒短片、缺乏长时序建模以及对极端场景的鲁棒性待验证。

---

## 315. State of the Art of LLM-Enabled Interaction with Visualization

**arXiv ID:** 2601.14943 | [PDF](https://arxiv.org/pdf/2601.14943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 316. On Implementing Hybrid Post-Quantum End-to-End Encryption

**arXiv ID:** 2601.14926 | [PDF](https://arxiv.org/pdf/2601.14926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 317. Lineup Regularized Adjusted Plus-Minus (L-RAPM): Basketball Lineup Ratings with Informed Priors

**arXiv ID:** 2601.15000 | [PDF](https://arxiv.org/pdf/2601.15000v1)

**作者:** Christos Petridis `[一作]` (Temple University), Konstantinos Pelechrinis `[通讯]` (University of Pittsburgh)

**通讯引用:** 2797 | [OpenAlex ID](https://openalex.org/A5037505718)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于回归的篮球阵容评估模型，利用对手信息与球员先验信息来估计阵容每次持球得分率，并对阵容进行预测。

**💡 创新点**

创新点在于：① 将前一赛季的 RAPM 球员评级作为阵容先验，改进了传统仅依赖原始阵容评级的做法；② 通过正则化回归同时考虑进攻与防守阵容，并将先验嵌入正则化项；③ 解决了阵容样本稀疏导致的噪声问题，特别适用于新阵容或少量样本场景。

**🔧 技术方法**

主要技术包括：正则化调整后加减值（RAPM）计算球员评级；基于持球级别的数据构建阵容回归模型；正则化回归中将先验 π 作为中心进行惩罚；使用 RMSE 评估预测误差。

**📊 数据集**

使用的数据集为 NBA 2022‑23 赛季的持球级别数据计算球员 RAPM，2023‑24 赛季的持球级别数据用于模型训练与外部验证。

**📈 对比分析**

与传统使用原始阵容评级的基线相比，本文模型在全赛季的 RMSE 上平均降低约 5%，在样本较少（<50 次持球）阵容上的优势更为明显；对未见过的阵容，仍能利用先验实现约 5% 的改进。

**⚠️ 局限性**

局限性包括：① 先验仅基于上一赛季球员评级，未实时更新；② 对于本赛季新进入联盟的球员采用固定 -1 值，可能影响先验精度；③ 正则化参数统一对进攻与防守阵容，可能未充分体现两者差异；④ 计算量相对较大，尤其在持续更新时更显突出。

---

## 318. Fractional Diffusion on Graphs: Superposition of Laplacian Semigroups and Memory

**arXiv ID:** 2601.14977 | [PDF](https://arxiv.org/pdf/2601.14977v1)

**作者:** Nikita Deniskin `[一作]` (Scuola Normale Superiore), Ernesto Estrada `[通讯]` (Institute for Cross-Disciplinary Physics and Complex Systems)

**通讯引用:** 15655 | [OpenAlex ID](https://openalex.org/A5042185860)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图网络上的亚扩散（时间分数扩散），证明其本质是记忆驱动的过程，并给出了可保质量的指数超算表示、SOE 近似以及基于亚扩散距离的最短路径定义。

**💡 创新点**

创新点在于将亚扩散等价为热算子（Laplacian）超算的凸组合，提供了精确的可保质量的指数和子扩散距离，并揭示顶点级异质记忆、记忆强化以及亚扩散几何能自洽地识别全局最短路径。

**🔧 技术方法**

采用子归纳、M–Wright 密度、完全单调性与贝塞尔定理、谱算子理论、log‑trapezoidal 乘子求和（SOE）以及逆 α‑稳定鞅随机时钟等技术；此外还用多速扩散与多层网络框架来解释 SOE 作为多尺度扩散的极限。

**📊 数据集**

实验主要使用 Erdős–Rényi 随机图（约 250 节点、1000 条边）和几何 Gabriel 图（约 600 节点、1156 条边）进行验证。

**📈 对比分析**

通过与解析的 Mittag‑Leffler 算子比较，SOE 误差随加项数几何下降，误差可压至 10⁻¹² 以上；在相同精度下，SOE 的计算时间比直接求解快数倍，且亚扩散最短路径与理论预测完全一致。

**⚠️ 局限性**

局限性包括：对大规模图需先进行谱分解或窗口预估，且仅适用于无权重连通无向图；记忆模型保持线性，无法描述含非线性交互或自适应权重的网络动态。

---

## 319. The GDN-CC Dataset: Automatic Corpus Clarification for AI-enhanced Democratic Citizen Consultations

**arXiv ID:** 2601.14944 | [PDF](https://arxiv.org/pdf/2601.14944v1)

**作者:** Pierre-Antoine Lequeu `[一作]` (Sorbonne Université), Benjamin Piwowarski `[通讯]` (Sorbonne Université)

**通讯引用:** 4159 | [OpenAlex ID](https://openalex.org/A5086752907)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了“Corpus Clarification”三步流程（论证单元提取、论证结构检测、论证单元澄清），用于将法国2020年“Grand Débat National”民意调查中的噪声多主题发言转换为结构化、可单独理解的论点，便于后续的主题建模和政治分析。

**💡 创新点**

创新点在于：①将民意发言标准化为可独立、单主题的论点；②使用小型可本地运行的LLM进行精细化注释，证明在不依赖大规模专有模型的前提下可获得与大型API模型相当甚至更优的性能；③构建了首个大规模（240k条贡献）法语民主咨询注释语料库GDN-CC-large。

**🔧 技术方法**

技术手段包括：Transformer‑based LLMs（Llama‑8b、Mistral‑7b、Qwen‑7b、Gemma‑9b 等）与 GPT‑4.1 进行对比；使用微调（instruct+finetune）提升模型；BERTScore、ROUGE‑L 评估文本质量；BERTopic 用于聚类评估；基于人工标注的 WindowDiff、token‑overlap 等指标评估标注一致性。

**📊 数据集**

数据集：①GDN‑CC（1,231条手工注释，2,285条论证单元）用于模型训练与评估；②GDN‑CC‑large（240k条贡献，约300k论证单元，包含155k陈述、282k方案、62k前提）用于公开发布的自动注释语料。

**📈 对比分析**

与基线 GPT‑4.1 及未微调的 LLMs 进行比较。微调后的小型模型在论证单元提取（Macro‑F1 0.81）、论证结构检测（Macro‑F1 0.79）和澄清（BERTScore 0.86、ROUGE‑L 0.60）上均优于或与 GPT‑4.1 接近；未微调模型表现差距显著。聚类实验显示澄清后文本的主题连贯性提升（91%偏好）。

**⚠️ 局限性**

局限性包括：仅针对法语 Grand Débat National 的数据，未验证跨语言或不同类型咨询的适用性；未系统探索不同 prompt 的效果；评估依赖 LLM‑as‑Judge，存在提示敏感性；虽然使用本地模型减少了对大模型的依赖，但仍需私有公司开发的模型；标准化过程可能导致情感、细微差异被稀释。

---

## 320. Application-level observability for adaptive Edge to Cloud continuum systems

**arXiv ID:** 2601.14923 | [PDF](https://arxiv.org/pdf/2601.14923v1)

**作者:** Kaddour Sidi `[一作]`, Baptiste Jonglez `[通讯]` (IMT Atlantique)

**通讯引用:** 65 | [OpenAlex ID](https://openalex.org/A5057551843)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了面向 Edge-to-Cloud 系统的应用层可观测性框架，利用开发者驱动的仪表化和 SLO 感知反馈实现自适应控制；

**💡 创新点**

创新点在于：①将应用层指标与基础设施指标统一收集并实时分析；②支持开发者自定义指标与自适应动作；③实现多层级（容器与业务逻辑）自适应；④动态增删指标无需改代码；

**🔧 技术方法**

技术栈包括 OpenTelemetry（指标与跟踪收集）、Prometheus（TSDB 与告警）、K3s（轻量化 Kubernetes）、Chaos Mesh（故障注入）以及 EnOSlib、cAdvisor、Node Exporter；

**📊 数据集**

使用公开的 Edge‑to‑Cloud 视频处理系统（包含摄像头、运动检测、云端 YOLO 识别模块），在 Grid'5000 物理/虚拟集群上实验；

**📈 对比分析**

通过对摄像头数目、运动检测阈值与动物出现率（0.5/1/1.5/3/30 频率）等参数的实验，对比系统在不同负载下的处理时间、响应时间与 Pod 数量变化，结果表明框架能在高负载与注入故障时保持稳定的延迟与帧率；

**⚠️ 局限性**

局限性包括：依赖手工编写 SLO 与指标配置；缺乏自动化的依赖关系推断与机器学习预测；在极端网络分区与资源极限时的自适应策略尚未充分验证。

---

## 321. Factorizable joint shift revisited

**arXiv ID:** 2601.15036 | [PDF](https://arxiv.org/pdf/2601.15036v1)

**作者:** Dirk Tasche `[一作]` `[通讯]` (Centre for Business Mathematics and Informatics, North-West University), Dirk Tasche (Centre for Business Mathematics and Informatics, North-West University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了对一般标签空间（包括回归）的可分解联合偏移（FJS）理论扩展，并探讨了FJS与协变量偏移、标签偏移以及广义标签偏移（GLS）的关系。

**💡 创新点**

创新点在于：①将FJS从仅适用于分类问题推广到支持连续/混合标签空间；②给出了在一般标签空间下估计目标先验概率的EM算法变体；③通过引入源分布对乘积测度的Radon–Nikodym密度，简化了理论推导并与已有分类结果统一；④阐明了GLS在足够表示映射下仍能诱导FJS的性质。

**🔧 技术方法**

主要技术包括：测度论与条件概率分布的严谨定义；Radon–Nikodym定理与重要性重加权；积分方程与迭代求解；EM算法的推广；以及对条件独立性和可测性做出的假设。

**📊 数据集**

本文为理论研究，未使用具体数据集进行实验。

**📈 对比分析**

由于缺乏实验验证，未给出与其他方法的性能对比。

**⚠️ 局限性**

局限性包括：①依赖强假设（绝对连续性、可测条件分布存在、正的条件密度等）；②迭代求解的收敛性和计算复杂度未知；③在缺少标签信息时仍需估计目标先验，实际可行性待验证。

---

## 322. Emergent, not Immanent: A Baradian Reading of Explainable AI

**arXiv ID:** 2601.15029 | [PDF](https://arxiv.org/pdf/2601.15029v1)

**作者:** Fabio Morreale `[一作]` (Sony AI), Yuki Mistufuji `[通讯]` (Sony AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对主流 XAI 方法的哲学假设进行分析，并提出以巴兰的“agential realism”为基础的解释性理论框架

**💡 创新点**

将解释性视为物质‑语言表演，而非模型内部隐藏结构的揭示，强调解释的涟漪式共构与责任性

**🔧 技术方法**

采用 Barad 的折射、反射、衍射三种光学视角，对现有 XAI 技术进行理论解构，并提出差异化的设计方向

**📊 数据集**

本研究为概念性论文，未使用具体数据集，仅以文本到音乐生成系统为案例示例

**📈 对比分析**

未进行实验比较，所提出的方法为概念性设计建议，缺乏量化性能评估

**⚠️ 局限性**

局限于理论探讨，缺少实证验证与具体实现，未形成可操作的设计模式

---

## 323. Risk Estimation for Automated Driving

**arXiv ID:** 2601.15018 | [PDF](https://arxiv.org/pdf/2601.15018v1)

**作者:** Leon Tolksdorf `[一作]` (Technische Hochschule Ingolstadt), Nathan van de Wouw `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 13601 | [OpenAlex ID](https://openalex.org/A5004028181)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种自动驾驶风险估计方法，将碰撞概率与碰撞严重性相结合，使用多圆形近似车辆形状，并给出了基于高斯不确定性和动力学能量的严重性模型，能够在实时运动规划中计算风险。

**💡 创新点**

创新点在于：①能够为不同碰撞构型（正面、侧面、追尾等）分配不同的严重性函数，并通过平均方法整合多圆碰撞区间；②在多圆形近似下保持对碰撞概率的上界保证，同时实现与现有碰撞概率算法相同的高效数值计算；③利用高斯与包装正态分布可解析积分，显著降低四维积分至二维积分。

**🔧 技术方法**

使用的技术包括：多圆形车辆形状逼近、概率密度函数分解、包装正态分布、二元高斯分布、动能严重性模型、交叉角区间求交算法、解析积分求期望、数值积分（Gauss‑Legendre等）。

**📊 数据集**

实验使用自定义的五种典型碰撞场景（正面、追尾、侧面不同撞击点）进行仿真，参数来源于表中给出的均值、方差等统计量；未使用公开大规模数据集，而是通过仿真场景验证方法。

**📈 对比分析**

与原始碰撞概率估计算法相比，初始化时间约增长4.4倍，风险估计与碰撞概率估计时间差别在8µs以下，使用两圆模型时估计时间可提升60%；风险估计在1ms的频率下可实现，满足实时运动规划需求。

**⚠️ 局限性**

主要限制包括：仅针对高斯不确定性（位置、速度、方向）构建，可扩展到其他分布需要重新推导积分；严重性模型基于动能，未考虑结构破坏、乘员受伤等因素；多圆形近似虽易于计算，但仍为上界逼近，可能导致保守估计。

---

## 324. Graph-Based Adaptive Planning for Coordinated Dual-Arm Robotic Disassembly of Electronic Devices (eGRAP)

**arXiv ID:** 2601.14998 | [PDF](https://arxiv.org/pdf/2601.14998v1)

**作者:** Adip Ranjan Das `[一作]` (Heriot-Watt University), Maria Koskinopoulou `[通讯]` (Heriot-Watt University)

**通讯引用:** 393 | [OpenAlex ID](https://openalex.org/A5046746682)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了一种名为eGRAP的闭环框架，利用双臂机器人在视觉感知、图结构规划和任务调度的协同下，实现对电子设备（以3.5英寸硬盘为实验对象）的全流程拆解。

**💡 创新点**

创新点包括：① 将实时检测结果即时映射为有向部件图并在线更新；② 通过拓扑排序得到符合前置与访问约束的可执行顺序；③ 设计了能让两臂并行作业并保持安全的调度策略；④ 结合粗略RGB‑D定位与微距摄像头的细节校正，实现了高成功率的螺丝解锁。

**🔧 技术方法**

核心技术包括：YOLOv11目标检测、RGB‑D与微距相机相结合的定位与对齐、基于手眼标定的全局坐标转换、图结构建模与拓扑排序、双臂任务分配与冲突检测、接触验证的螺丝拆卸程序。

**📊 数据集**

使用自制数据集：250张标注图片（手动标注），经Roboflow扩增后约1300张训练样本，覆盖三大硬盘品牌（Samsung、Western Digital、Seagate）的螺丝、壳体、PCB等类别。

**📈 对比分析**

在三种硬盘上各进行10次完整拆解实验，平均总拆解时间为20–22 min；在细节对齐（微距摄像）下螺丝拆除率为100%，未对齐时仅42.9%；成功率分别为Samsung 90%、Western Digital 70%、Seagate 90%，总体成功率83.3%。实验与单臂或非在线更新方案相比，显著提升了并行效率与鲁棒性。

**⚠️ 局限性**

局限性包括：① 依赖精确的相机标定与机械装配，硬件异常会导致中断；② 目前仅针对固定结构硬盘，未涉及柔性部件或极端尺寸的产品；③ 数据集规模有限，扩展到更多设备时需要重新标注与训练；④ 缺乏对不确定性（视觉误差、螺丝状态）进行概率建模的机制，调度策略仍基于确定性约束。

---

## 325. Unified Multi-Dataset Training for TBPS

**arXiv ID:** 2601.14978 | [PDF](https://arxiv.org/pdf/2601.14978v1)

**作者:** Nilanjana Chatterjee `[一作]` (Indraprastha Institute of Information Technology Delhi), Brejesh Lall `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 2984 | [OpenAlex ID](https://openalex.org/A5066116024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出单一模型跨多数据集的文本驱动人检索框架Scale‑TBPS，解决传统数据集特定训练导致的泛化瓶颈。

**💡 创新点**

创新点在于：① 噪声感知统一数据筛选（NDC），利用多模态预训练专家集成过滤噪声对；② 可扩展的多模态角度身份学习（DIL），以角度余弦损失和共享权重实现海量身份的判别学习。

**🔧 技术方法**

采用CLIP视觉‑语言预训练模型为骨干，集成预训练专家做筛选；引入多模态角度身份损失、排名损失以及测试时近邻归一化（NNN）等技术。

**📊 数据集**

训练使用CUHK‑PEDES、ICFG‑PEDES、RSTPReid、IIITD‑20K四个公开数据集；零样本评估在UFine6926上进行。

**📈 对比分析**

与各类数据集特定的最新方法及无筛选的联合训练相比，Scale‑TBPS在Rank‑1、mAP等指标均实现显著提升，例如CUHK‑PEDES Rank‑1提升至70%以上、mAP提升4‑6%；在UFine6926零样本实验中亦表现出更强的泛化能力。

**⚠️ 局限性**

局限性在于：① 仍需预训练专家集合作为筛选依据，若无合适专家可能难以执行；② 对极大规模或完全新领域数据的筛选与泛化效果尚未充分验证。

---

## 326. HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV

**arXiv ID:** 2601.14973 | [PDF](https://arxiv.org/pdf/2601.14973v1)

**作者:** Faryal Batool `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 1963 | [OpenAlex ID](https://openalex.org/A5056458774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 HumanDiffusion，一种基于视觉的扩散规划器，利用 YOLO-11 进行人类检测，直接从 RGB 图像中生成像素空间路径，供无人机无地图实时导航并完成物资投递。

**💡 创新点**

创新点在于将人类检测作为隐式目标推断，消除对预先地图或预定义 waypoint 的依赖；采用轻量级条件 UNet 扩散模型在像素空间生成全局轨迹，并在运行时实时更新目标；实现端到端、无地图、对人类安全友好的 UAV 轨迹规划。

**🔧 技术方法**

主要技术包括 YOLO‑11 人体检测、基于条件 UNet 的扩散轨迹生成、像素空间轨迹到 3D 世界坐标的投影、深度相机 (Intel RealSense D455) 用于深度估计、ROS 通信以及轻量级离线推理。

**📊 数据集**

使用 9,800 条由 A* 在模拟环境中生成的轨迹构成的训练/验证/测试集（8k/1.5k/300），以及在室内灾害模拟实验中收集的 RGB 与深度图像；所有数据均为 64×64 像素的轨迹掩码。

**📈 对比分析**

与传统基于 A* 或 MPC 的路径规划对比：在 300 条模拟测试样本上，MSE 为 0.02；在真实室内灾害场景（事故响应与遮挡搜索）中，整体成功率为 80%（10 次实验中 9 次完整投递），表现出平滑、符合安全边距的轨迹。

**⚠️ 局限性**

局限性包括：对人类检测失真（遮挡、光照、运动模糊）导致的感知失误；控制器跟踪误差与通信延迟导致的轨迹偏移；仅针对单人目标，缺乏多目标优先级与动态冲突处理；依赖离线服务器推理，实时性受限；在更大、更复杂的动态环境中对碰撞感知与自适应重规划的能力有限。

---

## 327. Fine-Grained Traceability for Transparent ML Pipelines

**arXiv ID:** 2601.14971 | [PDF](https://arxiv.org/pdf/2601.14971v1)

**作者:** Liping Chen `[一作]` (RMIT University), Haytham Fayek `[通讯]` (RMIT University)

**通讯引用:** 1259 | [OpenAlex ID](https://openalex.org/A5054958855)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

FG‑Trac框架实现了机器学习流水线的可验证样本级可追踪性，记录样本生命周期、影响评分并将日志加密到区块链。

**💡 创新点**

将检查点级影响估计与Merkle树日志结合，提供可验证的、细粒度样本追踪，而无需改动模型。

**🔧 技术方法**

采用TracInCP梯度内积贡献评分、SHA256伪匿名标识、Merkle树加密、区块链锚定和NDJSON异步日志。

**📊 数据集**

在CIFAR‑10、ABIDE和ADHD‑200三个公开数据集上验证。

**📈 对比分析**

对比基线模型和加入FG‑Trac的模型，准确率几乎不变，训练时间提升约2%–3%，内存增加约8%，链上日志约1GB，显示可接受的开销。

**⚠️ 局限性**

只在可插拔流水线中有效，无法防御主动逃逸日志的恶意操作者，且分布式与联邦学习场景尚未覆盖。

---

## 328. Towards Holistic Modeling for Video Frame Interpolation with Auto-regressive Diffusion Transformers

**arXiv ID:** 2601.14959 | [PDF](https://arxiv.org/pdf/2601.14959v1)

**作者:** Xinyu Peng `[一作]` (Shanghai Jiao Tong University), Hongkai Xiong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6108 | [OpenAlex ID](https://openalex.org/A5002494284)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于自回归扩散变压器的LDF-VFI框架，实现视频级别的帧插值，并通过跳连采样保证长序列的时间一致性。

**💡 创新点**

创新点包括：视频级别全局建模、跳连采样（skip‑concatenate）缓解曝光偏差、稀疏局部注意力与平铺VAE实现高分辨率可扩展、以及条件VAE解码提升重建质量。

**🔧 技术方法**

使用技术包括自回归扩散变压器（DiT）、稀疏局部注意力、平铺VAE编码、跳连采样、条件VAE解码、Euler ODE求解以及Ulysses序列并行。

**📊 数据集**

训练使用LAVIB数据集，评测使用SNU‑FILM‑entire和XTest‑entire两个视频级别基准，低帧率采样范围为2×–16×。

**📈 对比分析**

与RIFE、AMT、EMA‑VFI、BiM‑VFI等传统帧级方法对比，采用LPIPS和FVD评估，LDF‑VFI在SNU‑FILM和XTest 4K/16×插值任务上获得最低LPIPS和FVD，显示出更高的帧级质量与时间一致性。

**⚠️ 局限性**

局限性在于：训练与推理仍需大量显存与算力；自回归过程仍存在误差积累，跳连采样在极长序列中可能不足；对极端运动或非常长视频的泛化能力尚待进一步验证。

---

## 329. TempViz: On the Evaluation of Temporal Knowledge in Text-to-Image Models

**arXiv ID:** 2601.14951 | [PDF](https://arxiv.org/pdf/2601.14951v1)

**作者:** Carolin Holtermann `[一作]` (University of Hamburg), Anne Lauscher `[通讯]` (University of Hamburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

创建并公开了TempViz数据集，用于评估文本到图像模型在时间维度上的知识掌握，并对5种主流T2I模型进行了人工与自动评测；

**💡 创新点**

首次将时间知识拆分为五类（动物、风景、建筑、地图、艺术作品）并系统评估T2I模型的时间表现，同时对多种自动评测方法进行对比，发现现有评测手段对时间细粒度特征捕捉不足；

**🔧 技术方法**

使用人类标注（3个问题评估）、CLIPScore、BLIP+SBERT的句子相似度、基于VQA的分解式问答、VLM-as-judge等技术；

**📊 数据集**

7,940条提示、5个主题类别，配有参考图像与预期描述，覆盖多时间尺度的视觉表现；

**📈 对比分析**

人类评测显示T2I模型在时间维度的准确率普遍低于75%，最好的模型FLUX在大类中约38-46%；自动评测方法（CLIPScore、Captioning、VQA分解、VLM-judge）均未能达到与人类一致的准确率，最高宏F1约72%；

**⚠️ 局限性**

样本模型有限、仅两名标注者导致主观性、参考图像覆盖不足、评测指标局限、对部分领域知识需求高，影响评估可靠性。

---

## 330. Multi-Behavior Sequential Modeling with Transition-Aware Graph Attention Network for E-Commerce Recommendation

**arXiv ID:** 2601.14955 | [PDF](https://arxiv.org/pdf/2601.14955v1)

**作者:** Hanqi Jin `[一作]` (Taobao & Tmall Group of Alibaba), Bo Zheng `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Transition‑Aware Graph Attention Network（TGA）来建模电商用户多行为序列，利用结构化稀疏图捕捉行为转移信息并进行高效预测。

**💡 创新点**

创新点在于从项目、品类、邻居三视角构建稀疏转移图，并引入行为转移感知的图注意力机制，实现对行为类型和转移关系的细粒度建模，同时保持线性时间复杂度。

**🔧 技术方法**

使用图神经网络（Graph Attention）、多头注意力、MLP 参数化的转移变换、残差与层归一化等技术，整体架构为多层堆叠的 TGA 模块。

**📊 数据集**

在公开的淘宝数据集（Taobao）和规模达 10 亿条、用户平均 1,536 次交互的工业真实数据集上进行实验。

**📈 对比分析**

与 Transformer、Reformer、Linear Transformer、Longformer 及多行为序列模型 END4Rec、MB‑STR 进行对比，TGA 在 AUC 上均取得最高或相近水平，同时训练与推理速度提升 5.8×/3.4×，在工业场景实现了 1.29% CVR 与 1.79% GMV 的提升。

**⚠️ 局限性**

局限性包括：图构建仅考虑最近邻转移，可能忽略更远距离的长程依赖；模型在极大用户行为序列下仍需更多层以捕获高阶关系，可能导致梯度传播困难；在不同业务场景下的迁移效果未充分验证。

---

## 331. Circadian Modulation of Semantic Exploration in Social Media Language

**arXiv ID:** 2601.15091 | [PDF](https://arxiv.org/pdf/2601.15091v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 332. BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation

**arXiv ID:** 2601.15123 | [PDF](https://arxiv.org/pdf/2601.15123v1)

**作者:** Andrey Moskalenko `[一作]` (Lomonosov Moscow State University), Vlad Shakhuro `[通讯]` (Lomonosov Moscow State University)

**通讯引用:** 173 | [OpenAlex ID](https://openalex.org/A5069058935)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过大规模真实用户标注箱框数据，评估并改进promptable分割模型对边界框提示的鲁棒性，提出BREPS白盒攻击生成自然逼真的对抗框并定义鲁棒性指标。

**💡 创新点**

①首次系统收集25000条真实用户的框注解；②构建基于Gamma分布的框真实性正则；③将鲁棒性评估转化为白盒对抗优化，克服全搜索计算不可行；④在10个公共数据集上对15种模型进行统一鲁棒性基准。

**🔧 技术方法**

使用完整性IoU、CIoU损失、对抗优化（Adam）与正则、DICE损失、Gamma概率正则、白盒梯度攻击、可视化IoU热图、统计相关性分析。

**📊 数据集**

10个公开数据集：GrabCut、Berkeley、DAVIS、COCO‑MVal、TETRIS、ADE20K、PASCAL‑VOC2012、ACDC、BUID、MedScribble；同时从这些数据集采样约2500个实例进行用户实验。

**📈 对比分析**

采用IoU‑Tight、IoU‑Min/Max、IoU‑Δ等指标；结果显示SAM系列模型在真实用户框下IoU平均下降约15%，最差/最优框间差距可达30%；在对抗攻击下模型的最优IoU可提升约3%。

**⚠️ 局限性**

局限性包括：仅关注框提示而非点/文本等多模态提示；对抗优化需先验分布假设，可能不适用于极端用户行为；实验仅在固定分辨率下进行，未完全覆盖不同设备分辨率；对模型本身的鲁棒性改进仍依赖后续训练策略。

---

## 333. Emerging from Ground: Addressing Intent Deviation in Tool-Using Agents via Deriving Real Calls into Virtual Trajectories

**arXiv ID:** 2601.15120 | [PDF](https://arxiv.org/pdf/2601.15120v1)

**作者:** Qian Xiong `[一作]` (Beijing Forestry University), Mingyang Li `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“Real-to-Virtual”方法，通过从真实工具调用生成虚拟轨迹并对关键参数进行多类型突变，生成正负样本，结合两阶段训练提升工具使用式LLM的意图对齐能力。

**💡 创新点**

将真实工具调用作为底层，逆向生成虚拟轨迹；引入意图关键参数(ICP)并通过语义突变生成多样负样本；采用两阶段(SFT+ DPO)训练以显著提升意图对齐。

**🔧 技术方法**

逆向轨迹合成、ICP识别与突变、LLM-as-judge评价、两阶段训练(SFT + DPO)。

**📊 数据集**

使用MCP平台收集的728个真实工具，构建真实工具原语；合成数据包含正样本及通过ICP突变生成的负样本；对比ToolBench、ToolAlpaca、STE、GPT4Tools、PEToolBench等公开数据集。

**📈 对比分析**

在五大主流LLM上提升Acc_task约35%，Acc_intent约23%，超过SOTA基线1.2–42.1%和1.17–54.9%；在三组OOD数据集（ToolBench、DICE-Bench、ACEBench）平均提升Acc_task 18.2%和Acc_intent 8.6%。

**⚠️ 局限性**

仍受限于所选工具的覆盖度、突变规则的手工设计以及对极端复杂、多模态场景的泛化能力尚未充分验证。

---

## 334. From Who They Are to How They Act: Behavioral Traits in Generative Agent-Based Models of Social Media

**arXiv ID:** 2601.15114 | [PDF](https://arxiv.org/pdf/2601.15114v1)

**作者:** Valerio La Gatta `[一作]` (Northwestern University), Vincenzo Moscato `[通讯]` (University of Naples Federico II)

**通讯引用:** 4423 | [OpenAlex ID](https://openalex.org/A5081965427)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在大型语言模型驱动的生成式代理模型中加入行为特征，显著提升了代理在社交媒体中的参与多样性与传播真实性。

**💡 创新点**

提出将行为特征（如沉默观察者、偶尔转发、互动狂热者等）作为独立特征层，突破传统仅基于身份或人格的同质化问题，首次让代理在不同动作空间（发帖、转发、评论、点赞、静默）上保持自洽的多角色行为。

**🔧 技术方法**

使用基于提示的生成式代理框架，加入三层记忆（短期、长期、活动），并在LLM推理时结合推荐系统实现转发链，借助 Llama‑3‑70B 与 Gemma‑3‑27B 两大模型进行实验。

**📊 数据集**

数据来源包括 FinePersonas 公开身份语料库、四大主题域（医疗、技术、宗教、音乐）以及 2020 年美国选举推特真实社群的交互图（约 1,001 只节点）用于验证。

**📈 对比分析**

通过对比“仅身份特征”“仅人格特征”“行为特征+偏好推荐”等四种配置，使用行动概率、传播链长度、中心性分布等指标评估。结果显示行为特征配置在内容传播链平均长度、参与多样性和网络中心性上均优于其他配置，且在模拟网络与真实社群的中心性分布高度吻合。

**⚠️ 局限性**

局限在于行为特征保持静态、分类仍为七类有限且未对多平台、多群体进行进一步泛化验证。

---

## 335. Computable Structuralism: A Categorical Rewrite Calculus of Mythic Variants

**arXiv ID:** 2601.15078 | [PDF](https://arxiv.org/pdf/2601.15078v1)

**作者:** Juan J. Segura `[一作]` `[通讯]` (Universidad Andres Bello), Juan J. Segura (Universidad Andres Bello)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种将结构人类学的莱维-斯特劳斯典型转换形式化为可计算的 typed rewrite 程序，并通过自然变换检验其一致性。

**💡 创新点**

创新点在于将双注册（社会/象征）抽象为可执行的更新端函与自然变换，形成可检验的结构一致性条件，并用五值“Key”简洁编码顺序效应，实现跨媒介、跨传统的可移植比较。

**🔧 技术方法**

采用了 typed rewrite 系统、自然变换（η:U⇒V）的范畴学框架、可执行的操作符选择组，以及键值约束检查和一致性诊断技术。

**📊 数据集**

使用了一个包含 80 条叙事的平衡语料库，涵盖民间故事、宗教神话、超级英雄和连载宇宙，每条叙事被编码为 (a,b,x,y) 四元组及对应的 “Key”。

**📈 对比分析**

方法通过检验更新端函 U 与 V 的自然性条件来比较变体；实验显示 74% 的叙事在 Y 注册中显式命名规范约束，Key 分布稳定，诊断能力强；与基准 Jaccard 重叠相比，能捕捉更深层的合法转换关系。

**⚠️ 局限性**

局限在于编码的主观性（尤其是 x 与 y 的边界）、Key 的粗粒度、对单一场景的适用性（未扩展至多层情节图）以及对语料库规模的受限，未来需进一步细化操作符空间并验证更大规模数据。

---

## 336. The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution

**arXiv ID:** 2601.15075 | [PDF](https://arxiv.org/pdf/2601.15075v1)

**作者:** Chen Qian `[一作]` (Shanghai Artificial Intelligence Laboratory), Xia Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了层次化的代理因果归因框架，用于解释LLM代理行为背后的内部驱动因素。

**💡 创新点**

创新点在于①将代理交互拆解为时间序列组件并通过时间概率增益定位关键步骤；②在关键组件内部采用扰动（Drop & Hold）细粒度句子归因，兼容多种归因技术；③提供跨内存、工具驱动场景的统一评估方法。

**🔧 技术方法**

技术包括：
- 组件层面：时间序列概率增益（temporal likelihood dynamics）
- 句子层面：扰动归因（Drop & Hold），可替换为梯度、注意力或线性回归等
- 评估：Hit@k 统计、对比多种归因基线。

**📊 数据集**

使用自定义的9条代理执行轨迹（8条手工设计+1条GAIA检索任务），模型为Llama-3.1-70B-Instruct。

**📈 对比分析**

与AttriBoT、ContextCite、Saliency Score等基线对比。Hit@1、Hit@3、Hit@5 的平均分为：
- Prob. Drop&Hold: 0.944 / 1.000 / 1.000
- Leave-one-out: 0.833 / 0.944 / 1.000
- ContextCite: 0.833 / 0.944 / 0.944
- Saliency Score: 0.700 / 1.000 / 1.000
表现表明默认的Drop&Hold方案在所有指标上均最佳。

**⚠️ 局限性**

局限性包括：
- 需要访问模型的完整对数似然，难以在黑盒或受限API环境使用；
- 计算量较大，尤其在长轨迹上易产生OOM（如梯度归因）；
- 归因结果仍需人工验证，缺乏完全自动化的解释流程；
- 只针对单一LLM模型，未验证跨模型泛化。

---

## 337. Incentive-Tuning: Understanding and Designing Incentives for Empirical Human-AI Decision-Making Studies

**arXiv ID:** 2601.15064 | [PDF](https://arxiv.org/pdf/2601.15064v1)

**作者:** Simran Kaur `[一作]` (Delft University of Technology), Ujwal Gadiraju `[通讯]` (Delft University of Technology)

**通讯引用:** 3669 | [OpenAlex ID](https://openalex.org/A5038081564)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述并梳理人机协同决策研究中的激励设计，提出 Incentive‑Tuning 框架、报告模板与公开仓库。

**💡 创新点**

将激励方案拆解为组成要素与操作步骤，形成可重复、可报告的标准化设计流程，并公开共享数据以促进透明度。

**🔧 技术方法**

采用反射性主题分析对 97 篇论文进行编码和主题提炼；使用 GitHub、Markdown 等工具搭建共享仓库；框架本身基于文献归纳而非算法实现。

**📊 数据集**

以 2021‑2023 年收集的 97 篇实证人机决策论文（及早期 81 篇）为数据集，用于主题提炼和案例演示；仓库中包含每篇论文的激励细节与理由。

**📈 对比分析**

研究未进行算法或实验性能比较，而是通过案例研究演示框架的可操作性，并建议后续可通过对照实验评估不同激励方案对参与度、数据质量与可信度的影响。

**⚠️ 局限性**

仅关注金钱激励，未涵盖非金钱激励；框架基于当前文献，可能不适用于未来新任务或不同平台；缺乏对框架有效性的实证验证；样本来源有限，可能存在语言与文化偏差。

---

## 338. \textsc{LogicScore}: Fine-grained Logic Evaluation of Conciseness, Completeness, and Determinateness in Attributed Question Answering

**arXiv ID:** 2601.15050 | [PDF](https://arxiv.org/pdf/2601.15050v1)

**作者:** Zhichao Yan `[一作]` (Shanxi University), Jeff Z. Pan `[通讯]` (University of Edinburgh)

**通讯引用:** 7645 | [OpenAlex ID](https://openalex.org/A5066422711)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对有出处的问答（Attributed Question Answering, AQA）提出了LogicScore评估框架，能够从全局推理的角度评估长篇答案的逻辑完整性、简洁性和确定性。

**💡 创新点**

创新点在于：① 把评估从局部事实核对转向全局推理；② 采用Horn规则将自然语言推理转化为可检验的证明树；③ 设计了三个可量化指标（Completeness、Conciseness、Determinateness）来揭示“归因盲区”（attribution myopia）。

**🔧 技术方法**

技术手段包括：Llama/ChatGPT等LLM的CoT提示生成长篇答案；LLM驱动的逻辑转换将答案拆解为Horn子句；反向链路检索构建逻辑路径；再推理验证（re‑inference）检验答案是否被逻辑严密推导得出；以及使用Cohen’s Kappa、Pearson‑r等统计方法评估指标可靠性。

**📊 数据集**

数据集：HotpotQA（distractor模式）、MusiQue（answerable模式）和2WikiMultiHopQA，共计三大多跳QA基准，用以考察长篇答案的跨文档推理能力。

**📈 对比分析**

与20+种LLM（包括GPT‑5、Gemini‑3‑Pro、LLaMA‑3、Qwen‑3等）比较，发现即使归因精度（precision/recall）高达90%以上，逻辑完整度、简洁度仍往往低于50%；展示了参数扩张导致的“缩放悖论”，即规模增大提升确定性但牺牲简洁性；LogicScore在各模型上提供了一致、可解释的逻辑质量评估。

**⚠️ 局限性**

局限性：1) 依赖LLM完成逻辑转换，转换错误会影响指标；2) 目前仅评估多跳问答，未覆盖更广泛推理任务；3) 评估仍需要人工标注作为对照，无法完全自动化；4) 对于极其复杂或非结构化推理链，Horn规则约束可能不足以捕捉所有逻辑细节。

---

## 339. Deep Leakage with Generative Flow Matching Denoiser

**arXiv ID:** 2601.15049 | [PDF](https://arxiv.org/pdf/2601.15049v1)

**作者:** Isaac Baglin `[一作]` (University of Surrey), Simon Hadfield `[通讯]` (University of Surrey)

**通讯引用:** 5999 | [OpenAlex ID](https://openalex.org/A5091184063)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种利用流匹配（Flow Matching）先验的深度泄露攻击，直接在梯度反演过程中引入生成式先验，使重构图像既保持真实性又具备高保真度。

**💡 创新点**

创新点在于将流匹配模型作为可学习的去噪器整合进深度泄露攻击，并且该先验可以在与任务无关的数据上训练，显著提升攻击对不同行业、批量、训练阶段及防御措施的鲁棒性。

**🔧 技术方法**

核心技术包括：流匹配生成模型、插值式梯度匹配、流匹配正则化（v_θ的平方范数）与最小化余弦相似度的损失，以及Adam优化器。

**📊 数据集**

实验使用了CIFAR‑10和Tiny‑ImageNet两个标准图像分类数据集，并在多种网络架构（自定义卷积网络、ResNet、MLP‑Mixer等）上验证。

**📈 对比分析**

与GradInversion、GGL、DLF、SME等现有攻击方法比较，在PSNR、SSIM、LPIPS、FMSE等指标上均表现更优；在批量大小、训练周期、噪声、裁剪、稀疏化和Precode等常见防御下，性能优势仍然明显。

**⚠️ 局限性**

局限性包括：对训练时梯度信息仍有一定依赖，极端噪声或高度稀疏的梯度会显著降低重构质量；此外，流匹配模型的训练成本和存储需求相对较高，且攻击在大型高分辨率数据集上的可扩展性尚未彻底验证。

---

## 340. CADGrasp: Learning Contact and Collision Aware General Dexterous Grasping in Cluttered Scenes

**arXiv ID:** 2601.15039 | [PDF](https://arxiv.org/pdf/2601.15039v1)

**作者:** Jiyao Zhang `[一作]` (Peking University), Hao Dong `[通讯]` (Peking University)

**通讯引用:** 5646 | [OpenAlex ID](https://openalex.org/A5050592208)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种两阶段框架：先用稀疏IBS（Interaction Bisector Surface）作为可压缩、场景解耦、接触-碰撞感知的中间表示，再通过能量函数优化得到在拥挤场景下的多指抓取姿态。

**💡 创新点**

创新点在于将IBS作为中间表示并通过体素级条件扩散模型与力闭合过滤生成该表示，同时设计专门的能量函数和排名策略，使得在部分观测下能够生成稳定、无碰撞的抓取姿态。

**🔧 技术方法**

采用的技术包括ResUNet14+采样实现手腕姿态估计，体素化并使用occupancy‑diffusion模型生成稀疏IBS，IBS候选生成与力闭合评分排序，基于四项能量（关节限值、自我穿透、接触、碰撞）的梯度优化，以及多候选优化残差排名。

**📊 数据集**

使用DexGraspNet2.0训练集（60个训练物体+1259个测试物体）和GraspNet‑1Billion 7600个训练场景、670个测试场景进行模拟实验，真实实验在Flexiv Rizon‑4+Leap Hand平台上使用30个多形状物体的5个杂乱场景。

**📈 对比分析**

与DexGraspNet2.0、HGC‑Net、ISAGrasp、GraspTTA等基线对比，实验表明模拟成功率可达约86‑93%，真实场景成功率93.3%（基线约83.9%），并且实现了零样本跨手臂泛化；单次抓取平均耗时6.5秒。

**⚠️ 局限性**

局限性包括对小尺寸物体的抓取仍表现欠佳，以及第二阶段优化依赖扩散采样导致计算时间较长，可考虑使用更快的采样器如DDIM进一步优化。

---

## 341. Overcoming In-Memory Bottlenecks in Graph Foundation Models via Retrieval-Augmented Generation

**arXiv ID:** 2601.15124 | [PDF](https://arxiv.org/pdf/2601.15124v1)

**作者:** Haonan Yuan `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**通讯引用:** 17470 | [OpenAlex ID](https://openalex.org/A5100380470)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于检索增强生成的图基础模型RAG-GFM，利用统一的语义-结构检索数据库把图知识外部化，从而缓解传统GFMs的内存瓶颈，提升跨域、跨数据集的迁移与可解释性。

**💡 创新点**

创新点包括（1）双模检索数据库：语义存储（文本前缀+BERT向量）与结构存储（基于Walk‑Spectrum的子图编码）；（2）跨视角知识对齐目标，利用互信息与对比学习同时保持语义与结构的一致性；（3）上下文检索增强策略，结合领域门控提示实现高效少样本适配。

**🔧 技术方法**

核心技术包括检索增强生成（RAG）、BERT语义编码、Walk‑Spectrum 结构编码、双视角信息瓶颈对齐（InfoNCE）、域标记(token)、域门控提示、少样本聚类推理。

**📊 数据集**

使用五个文本属性图数据集：Cora、Citeseer、Pubmed（引用领域）、大型商品协购网络（电商领域）、Wikipedia子集超链接网络（网页链接领域）。

**📈 对比分析**

与13种基线（Vanilla GNN、Graph预训练模型、文本无关GFMs、文本属性GFMs）比较，RAG‑GFM在节点/图分类的5‑shot、10‑shot等设置下均实现平均提升3–5%准确率，在留一数据集/留一领域迁移任务中表现尤为突出；同时在训练周期和显存使用上显著更优。

**⚠️ 局限性**

局限性：检索过程仍带来一定的计算与存储开销；对检索质量高度依赖，语义/结构匹配不当会导致误检；域标记与门控策略需在不同任务中手动调参；当知识库规模极大时，检索速度与成本可能成为瓶颈。

---

## 342. From Insight to Intervention: Interpretable Neuron Steering for Controlling Popularity Bias in Recommender Systems

**arXiv ID:** 2601.15122 | [PDF](https://arxiv.org/pdf/2601.15122v1)

**作者:** Parviz Ahmadov `[一作]` (Delft University of Technology), Masoud Mansoury `[通讯]` (Delft University of Technology)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5009118126)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种后置方法PopSteer，通过稀疏自编码器（SAE）解释并纠正推荐系统中的受欢迎度偏差。

**💡 创新点**

创新点在于利用SAE的单一语义神经元检测和调节受欢迎度信号，实现可解释且细粒度的公平性控制。

**🔧 技术方法**

技术主要包括稀疏自编码器、Cohen’s d效应量用于神经元选择、神经元激活调节（steering）以及生成极端偏好合成用户。

**📊 数据集**

实验数据集为MovieLens 1M、BeerAdvocate和Yelp三大公开序列推荐数据集。

**📈 对比分析**

与多种基准（P‑MMF、PCT、IPR、FA*IR、DUOR、随机基线）比较，PopSteer在保持准确率（nDCG）几乎不降的前提下显著提升了物品曝光公平度（覆盖率↑、Gini指数↓），并在多数数据集上优于现有方法。

**⚠️ 局限性**

局限包括需先训练SAE、对神经元阈值与steering强度敏感、合成用户生成方法简单，未来可进一步改进生成策略并扩展到其他偏差类型。

---

## 343. Physics-Informed Wireless Imaging with Implicit Neural Representation in RIS-Aided ISAC System

**arXiv ID:** 2601.15113 | [PDF](https://arxiv.org/pdf/2601.15113v1)

**作者:** Yixuan Huang `[一作]` (Southeast University), Shi Jin `[通讯]` (Southeast University)

**通讯引用:** 41489 | [OpenAlex ID](https://openalex.org/A5013079905)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在RIS辅助的ISAC系统中，利用隐式神经表示（INR）对无线成像进行建模，实现连续空间表示并支持任意分辨率的超分辨成像。

**💡 创新点**

创新点包括：①将INR与物理前向模型耦合，使用位置编码和正弦激活提升高频细节捕捉；②避免多径提取、无需大规模训练集；③实现任意分辨率输出，揭示成像焦距特性；④在少量CSI测量下即可获得高质量图像。

**🔧 技术方法**

使用的技术包括：6层MLP隐式网络、Fourier特征位置编码、sin激活函数、物理信息损失函数、基于RIS多相位配置的CSI估计（LS），以及PyTorch训练框架。

**📊 数据集**

数据集来源于Kaggle的TikTok舞蹈视频分割图像，共2615张，转换为100×100灰度图像，用于生成CSI并评估成像效果。

**📈 对比分析**

与传统的傅里叶变换（FT）和压缩感知（CS）以及无RIS、ReLU激活、无位置编码等对照实验，结果显示：在500或2500个RIS相位配置下，INR的MSE <0.0012、SSIM ≈0.99、PSNR >30 dB，显著优于FT/CS；仅200次epoch即可得到接近完美图像；同时分析了距离与相位数对性能的影响，揭示存在最优成像距离。

**⚠️ 局限性**

局限性包括：仍需RIS多相位配置和相对较多的CSI采样；在极远距离或高相关性场景下性能下降；实验仅覆盖静态人形目标，缺乏多目标或动态环境验证；对噪声和模型误差的鲁棒性尚待进一步提升。

---

## 344. An Agentic Operationalization of DISARM for FIMI Investigation on Social Media

**arXiv ID:** 2601.15109 | [PDF](https://arxiv.org/pdf/2601.15109v1)

**作者:** Kevin Tseng `[一作]` (National Institute of Cyber Security), Phil Tinn `[通讯]` (SINTEF)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5094129116)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建多智能体管线，将DISARM框架操作化以在社交媒体上自动检测并映射外部信息操纵与干预（FIMI）行为，形成可解释、可验证的调查工作流。

**💡 创新点**

① 采用技术导向的代理架构，将DISARM拆分为可执行的子任务，保证每个发现都对应特定TTP；② 设计原子证据分解与统计验证框架，实现可复制、可审核的自动化检测；③ 在军事场景中讨论人机协同与互操作性，为大规模信息作战提供操作性范例。

**🔧 技术方法**

使用Claude Opus 4.5 LLM、BMAD‑METHOD自然语言工作流定义、多臂赌博机策略选择DISARM技术、SQL查询执行、统计检验（OR、Fisher等）以及代理子任务调度与子进程记忆。

**📊 数据集**

中国X平台的郭文贵信息作战数据（合并的China 1/2）与俄国针对摩尔多瓦2025选举的Telegram频道机器人数据。

**📈 对比分析**

通过15轮迭代与人工标注对比，技术检测通过率为50%（14/28），原子证据通过率约为29%；在Telegram数据中识别出30+新增机器人；总耗时约35 min，成本约$11.4。

**⚠️ 局限性**

缺少对时间同步和细粒度语言变化的建模，主要基于行为特征，难以区分多套机器人、相互对立的群体或人工“极客”账号；未实现多模态内容分析，需人工最终验证与意图判断。

---

## 345. Economic feasibility of virtual operators in 5G via network slicing

**arXiv ID:** 2601.15103 | [PDF](https://arxiv.org/pdf/2601.15103v1)

**作者:** Erwin J. Sacoto-Cabrera `[一作]` (Universitat Politècnica de València), Vicent Pla `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 1956 | [OpenAlex ID](https://openalex.org/A5002611857)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

分析了通过5G网络切片实现的多个运营商在共同网络基础设施上提供服务的情况，提出了两种商业模型：战略模型和垄断模型。

**💡 创新点**

创新点在于通过区分处理共享队列模型和博弈论分析用户订阅决策与运营商定价决策之间的战略互动，展示了网络运营商在两种商业模型下的经济激励。

**🔧 技术方法**

使用了区分处理共享（DPS）队列模型和博弈论技术。

**📊 数据集**

未具体提及使用的数据集，但模型基于用户的效用和运营商的收入进行分析。

**📈 对比分析**

通过与基线场景（仅网络运营商服务其用户）进行比较，发现战略模型下的用户订阅率高于垄断模型，且战略模型在用户数量上更具吸引力。

**⚠️ 局限性**

限制在于未考虑运营商的运营成本，且模型假设用户的敏感性和服务优先级是固定的，未考虑用户异质性对结果的影响。

---

## 346. Field-Space Autoencoder for Scalable Climate Emulators

**arXiv ID:** 2601.15102 | [PDF](https://arxiv.org/pdf/2601.15102v1)

**作者:** Johannes Meuer `[一作]` (German Climate Computing Center), Christopher Kadow `[通讯]` (German Climate Computing Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于Transformer的Field‑Space Autoencoder（FS‑AE），能够在HEALPix球面网格上对气候模型输出进行高效压缩，并支持零样本超分辨率以及压缩场的扩散生成；

**💡 创新点**

创新点在于①使用Field‑Space Attention直接在球面上操作，避免投影畸变；②多尺度残差设计实现分辨率不变和零样本超分；③将压缩场作为潜在空间训练扩散模型，融合低分辨率大样本与高分辨率稀缺样本；

**🔧 技术方法**

技术包括Transformer+Field‑Space Attention、HEALPix多尺度残差编码/解码、压缩场扩散模型、t‑SNE可视化、光谱分析等；

**📊 数据集**

使用ERA5全球再分析（0.25°）作为训练与评估基准，并使用MPI‑ESM1.2‑HR 历史模拟（≈100 km）进行扩散模型的训练与验证；

**📈 对比分析**

与CNN‑VAE基线对比，FS‑AE在64×压缩时RMSE约0.3 °C，比CNN‑VAE的16×压缩更优；在极端压缩（256、1024×）下仍保持高PSNR；多变量重建误差下降30–50%；零样本超分辨率在4×、16×放大时误差仅略增；

**⚠️ 局限性**

局限性：仅验证五个变量，降雨等高度随机场仍被平滑；确定性压缩难以捕捉噪声；未充分探索时间压缩；缺乏更高层压缩和不同大气层压缩的研究。

---

## 347. Bangla Music Genre Classification Using Bidirectional LSTMS

**arXiv ID:** 2601.15083 | [PDF](https://arxiv.org/pdf/2601.15083v1)

**作者:** Muntakimur Rahaman `[一作]` (Bangladesh Army International University of Science and Technology), Md Mehedi Hassain `[通讯]` (International Islamic University Chittagong)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5095828819)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个基于双向长短期记忆网络（Bi‑LSTM）的孟加拉语音乐流派分类系统，并收集整理了十种孟加拉音乐流派的数据集。

**💡 创新点**

创新点在于：①首次公开提供针对孟加拉音乐的多类别（十类）数据集；②利用Bi‑LSTM结合MFCC、ΔMFCC和色度特征，同时引入孟加拉语音素信息增强语义表达；③在同一数据集上与传统机器学习与单向LSTM等模型进行了系统对比。

**🔧 技术方法**

使用的技术包括：音频预处理（MP3转WAV）、MFCC、ZCR、谱心等特征提取；Bi‑LSTM网络架构；交叉熵损失训练；混淆矩阵、Precision/Recall/F1评估指标。

**📊 数据集**

数据集由10个孟加拉音乐流派（Bangla hip‑hop、Bangla metal、Bangla rock、Deshattobodhok、Palligiti、Lalon Giti、Nazrul Sangeet、Rabindra Sangeet、Folk、Hamdanaat）组成，测试集共718条样本。

**📈 对比分析**

与Logistic回归、SVM、K‑NN、ANN、CNN、LSTM等模型对比，Bi‑LSTM在该数据集上达到了78.69% 的准确率，明显高于其它传统方法（SVM 21%、K‑NN 45% 等）和单向LSTM（74%）。

**⚠️ 局限性**

局限性包括：①数据量相对有限，导致某些类别（如Folk、Metal）表现不佳；②特征主要集中在声学层面，缺少更丰富的歌词/情感信息；③模型训练对资源要求较高，实际部署时可能受限。

---

## 348. Multi-Agent Constraint Factorization Reveals Latent Invariant Solution Structure

**arXiv ID:** 2601.15077 | [PDF](https://arxiv.org/pdf/2601.15077v1)

**作者:** Christopher Scofield `[一作]` `[通讯]`, Christopher Scofield

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将大语言模型代理视为对共享状态执行约束强制的算子，利用算子理论与受限优化，证明多代理系统的迭代等价于多算子分解，能够收敛到单一代理无法达到的交叉约束可行集，从而解释了多代理系统的性能提升。

**💡 创新点**

创新点在于：①将多代理互动抽象为约束强制算子的有序组合，揭示了“分解”对动力学可达性的重要性；②在传统投影/分裂框架基础上引入近似软约束的proximal算子，证明了该机制对逼近逼近的鲁棒性；③把理论映射到文本对话系统，提供了一个可解释的模型来解释现实中基于角色/辩论的多代理设计。

**🔧 技术方法**

主要使用的技术包括算子理论（投影算子、Douglas‑Rachford、ADMM、proximal point 方法）、受限优化理论、Fejér 单调性与收敛性证明、以及对文本状态的抽象编码。

**📊 数据集**

该工作为理论性研究，不依赖具体数据集；若需验证，作者建议使用典型的多代理对话数据（如 OpenAI 的 Debate、ChatGPT‑RolePlay 或基准推理任务）进行实验。

**📈 对比分析**

与单一代理的全局约束投影或正则化优化相比，多代理系统在理论上能保留整个交叉约束集合作为不变集，从而获得更大可行空间；实验性评估（若做）应关注收敛速度、稳定性和最终解的可行性，而非传统指标；总体而言，文中指出多代理在理论上可取得比单代理更优的可行解。

**⚠️ 局限性**

主要限制包括：①需要约束集合为闭凸且交集非空，实际对话约束往往不满足；②近似/噪声更新可能导致不严格的收敛或可达性退化；③理论仅描述渐近行为，未给出有限步性能或对噪声鲁棒性的定量保证；④未给出具体实现细节或实证验证，因而无法直接评估实际系统的效能。

---

## 349. Turning Citation Networks Inside Out: Studying Science Using Content-Based Knowledge Graphs from LLM-Derived Taxonomies

**arXiv ID:** 2601.15062 | [PDF](https://arxiv.org/pdf/2601.15062v1)

**作者:** Seorin Kim `[一作]` (Vrije Universiteit Brussel), Vincent Ginis `[通讯]` (Harvard University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过使用大型语言模型对论文摘要进行分类，生成每篇论文的测量、数据类型和研究问题三元组，构建内容基础的三元图谱，并对其结构和时间演化进行网络分析。

**💡 创新点**

提出“inside‑out”内容导向的知识图谱方法；利用LLM自动生成可解释的分类体系；通过中心性与电容率比值识别结构桥梁和被忽视的组合；从内容而非引用或关键词重构学科结构。

**🔧 技术方法**

使用 GPT‑o3‑mini（或类似大语言模型）进行概念抽取与标签分配；图论方法（度、强度、节点/边betweenness、kappa 统计、Kendall τ、Bootstrap CI、随机置换检验）来评估结构特征和稳定性。

**📊 数据集**

617 篇英文期刊论文（关于代际财富流动），从 OpenAlex 检索并手工校正，使用摘要文本进行标注。

**📈 对比分析**

与传统引文网络和关键词共现方法对比；通过随机置换模型检验结构稳定性；结果显示回归基测量始终占据核心，配对和三元组合随时间显著变化；电容率指标揭示重要桥梁和被忽视的组合。

**⚠️ 局限性**

数据量有限（仅 617 篇），仅基于摘要而非全文，三元组简化可能忽略同一论文中的多重测量或方法；LLM 分类的一致性虽高但仍受任务模糊性影响。

---

## 350. Game-Theoretic Lens on LLM-based Multi-Agent Systems

**arXiv ID:** 2601.15047 | [PDF](https://arxiv.org/pdf/2601.15047v1)

**作者:** Jianing Hao `[一作]` (Hong Kong University of Science and Technology), Siguang Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 53 | [OpenAlex ID](https://openalex.org/A5101968524)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了基于大语言模型的多智能体系统（LLM-MAS），并提出以游戏理论四要素（玩家、策略、收益、信息）为核心的系统化框架。

**💡 创新点**

创新点在于将传统游戏理论与现代LLM驱动的多智能体研究统一起来，构建了一个可供分类、比较和指导未来研究的通用理论视角。

**🔧 技术方法**

主要技术包括游戏理论建模、LLM自然语言推理与交流、策略空间抽象、奖励塑造与监管机制设计等。

**📊 数据集**

综述涵盖了多种Benchmarks，如GAIA、MultiAgentBench、SWE‑bench、FinBen、TravelPlanner等，用以评估协调、推理与专业化任务执行能力。

**📈 对比分析**

通过对Benchmarks的对比分析，发现SWE‑Debate在多轮竞争式辩论中实现了SOTA 41.4% Pass@1，FinCon框架通过奖励塑造与监管实现了去中心化激励与系统稳定性的提升。

**⚠️ 局限性**

局限性包括缺乏稳健的均衡选择与激励兼容性机制、理论形式化不足、对部分可观测环境的适用性有限，以及对LLM策略演化的动态分析仍不充分。

---

## 351. Federated Transformer-GNN for Privacy-Preserving Brain Tumor Localization with Modality-Level Explainability

**arXiv ID:** 2601.15042 | [PDF](https://arxiv.org/pdf/2601.15042v1)

**作者:** Andrea Protani `[一作]` (European Organization for Nuclear Research), Luigi Serio `[通讯]` (European Organization for Nuclear Research)

**通讯引用:** 820 | [OpenAlex ID](https://openalex.org/A5069214353)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于联邦学习的Transformer‑GNN混合框架，用于脑肿瘤定位，并实现了跨机构的隐私保护协同训练。

**💡 创新点**

创新点在于将Transformer自注意力与图神经网络相结合，并通过在CERN的CAFEIN®平台部署实现多机构协作；同时对模型的模态级注意力进行统计显著性解释。

**🔧 技术方法**

采用Transformer编码器、GATv2图卷积、CLF token注意力、聚合层、混合精度训练及FedAvg联邦聚合等技术。

**📊 数据集**

使用BraTS 2021多模态MRI数据（T1、T1ce、T2、FLAIR）共1251份，分割为训练/测试集进行实验。

**📈 对比分析**

与中心化训练和各机构单独训练进行对比，联邦训练在Dice、Precision、Recall、F1等指标上几乎等同于中心化模型，而单机训练则显著落后。

**⚠️ 局限性**

局限包括仅在模拟的单数据集客户端上验证，未测试真实跨中心的异质性；以及仅实现了定位而非像素级分割。

---

## 352. SmartOracle -- An Agentic Approach to Mitigate Noise in Differential Oracles

**arXiv ID:** 2601.15074 | [PDF](https://arxiv.org/pdf/2601.15074v1)

**作者:** Srinath Srinivasan `[一作]` (North Carolina State University), Marcelo D'Amorim `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了 SmartOracle，一套基于多代理大型语言模型（LLM）的差分模糊测试判据系统，用于自动化判定 JavaScript 引擎差异是否构成规范违规。

**💡 创新点**

创新点在于将差分判据拆分为专门的子代理（发现差异、查询规范、最小化示例、重复检查、误报评估）并通过工具调用实现自我校正，同时引入半监督标签传播来高效构造评估基准。

**🔧 技术方法**

使用技术包括多代理 LLM 框架、终端执行工具、ECMAScript 规范查询接口、TF‑IDF 文本特征、K‑means 聚类、质心传播、以及与 Gemini 2.5 Flash/Pro 的对比实验。

**📊 数据集**

实验数据集涵盖两份历史 JavaScript 引擎 Bug 集（Park 与 Lima）、710 条人工标注的模糊测试差异，以及新近产生的约10,000 条差分发现。

**📈 对比分析**

与单一 Gemini 2.5 Pro 的顺序链式提示基线相比，SmartOracle 在召回率上提升至 0.84（对比 0.68），分析时间缩短 4 倍，API 成本降低 10 倍，且在新版本引擎上发现 8 条未公开的规范缺陷，确认率高。

**⚠️ 局限性**

主要局限包括：仅在 JavaScript 引擎域内验证，需手工维护误报模式，依赖闭源 LLM 可能面临模型漂移，且在不同语言或协议等其它系统的泛化尚未评估。

---

## 353. Knowledge Restoration-driven Prompt Optimization: Unlocking LLM Potential for Open-Domain Relational Triplet Extraction

**arXiv ID:** 2601.15037 | [PDF](https://arxiv.org/pdf/2601.15037v1)

**作者:** Xiaonan Jing `[一作]` (Hefei University of Technology), Jiapu Wang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 44714 | [OpenAlex ID](https://openalex.org/A5100384686)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于知识恢复的自我评估与提示优化框架 KRPO，用于提升开放域关系三元组抽取的准确性与完整性。

**💡 创新点**

创新点在于：①将知识恢复与自然语言推断结合生成可量化的自我评估信号；②利用文本梯度驱动的迭代提示优化；③引入动态关系规范化记忆，显著减少关系冗余。

**🔧 技术方法**

核心技术包括：大型语言模型（LLM）生成提示和三元组；知识恢复模块与 NLI 评估；文本梯度生成器实现基于评估的提示更新；Cross‑Encoder 关系对齐模型与基于 LLM 的关系决策器。

**📊 数据集**

实验使用 WebNLG、REBEL、Wiki‑NRE 三大公开数据集。

**📈 对比分析**

与现有 SOTA 方法（REGEN、GenIE、EDC）在 Exact、Partial、Strict 三种匹配模式下进行对比，KRPO 在所有 LLM 后端（Mistral‑7B、Qwen3‑32B、GPT‑4o‑mini、GPT‑5、DeepSeek‑V3）上均实现 3%–10% 左右的 F1 提升，特别是在严格模式下显著提高。

**⚠️ 局限性**

局限性包括：①对自我评估质量高度依赖 NLI 模型的准确性；②文本梯度生成过程仍是黑箱，难以解释提示变化原因；③关系规范化依赖预训练的 Cross‑Encoder，处理极其细粒度差异仍有挑战。

---

## 354. WavLink: Compact Audio--Text Embeddings with a Global Whisper Token

**arXiv ID:** 2601.15118 | [PDF](https://arxiv.org/pdf/2601.15118v1)

**作者:** Gokul Karthik Kumar `[一作]` (Technology Innovation Institute), Hakim Hacid `[通讯]` (Technology Innovation Institute)

**通讯引用:** 2016 | [OpenAlex ID](https://openalex.org/A5013316111)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出WavLink模型，通过在Whisper编码器中加入可学习的全局Token，将30秒音频压缩为单一嵌入，并与文本编码器联合训练。

**💡 创新点**

创新点在于：①利用Whisper的ASR预训练特征与全局Token实现高效压缩；②采用两阶段训练加Matryoshka多尺度监督，得到可按维度裁剪的嵌入；③在同等参数规模下，用单Token完成检索、零样本分类和多选QA。

**🔧 技术方法**

技术细节包括：Whisper编码器+全局Token、CLIP或ModernBERT文本编码器、CLIP/SigLIP对比损失、LoRA或全微调、Matryoshka多尺度监督、两阶段训练策略。

**📊 数据集**

使用的数据集包括：Auto-ACD、AudioSet、VGGSound、AudioSetCaps、AudioCaps、Clotho、ESC-50、US8K、AIR‑Bench等，涵盖音频检索、零样本分类和多选问答。

**📈 对比分析**

与现有CLAP系列（LAION‑CLAP、MGA‑CLAP、ReCLAP、AF‑CLAP）以及大型音频LLM（Qwen2‑Audio、Falcon3‑Audio）对比，WavLink在AudioCaps/Clotho检索上实现SOTA或接近SOTA，在VGGSound零样本分类上领先，AIR‑Bench多选QA与Falcon3‑Audio相当，且嵌入尺寸可压缩至1/8而性能差异<1点。

**⚠️ 局限性**

局限性：在ESC‑50、US8K等细粒度分类以及定位、情绪等任务上表现略逊；单全局Token难以捕获细粒度音频–文本对齐；多语言、跨域扩展尚未验证。

---

## 355. Auditing Language Model Unlearning via Information Decomposition

**arXiv ID:** 2601.15111 | [PDF](https://arxiv.org/pdf/2601.15111v1)

**作者:** Anmol Goel `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 25133 | [OpenAlex ID](https://openalex.org/A5027450194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了基于部分信息分解（PID）的LLM无学习审计框架，量化遗忘知识与残留知识，提供可解释的内部表示评估；

**💡 创新点**

首次将PID应用于无学习审计，提出未学习知识与残留知识指标，并以残留知识作为对抗攻击风险与推理时弃权的量化工具；

**🔧 技术方法**

使用RINE神经估计器进行信息量估计、PID分解、线性探针、熵与预测不确定性等技术；

**📊 数据集**

实验基于TOFU和MUSE无学习基准，使用Llama、Falcon等多种LLM模型；

**📈 对比分析**

与传统Forget Quality、MIA等方法对比，PID指标更能揭示内部泄漏，残留知识与攻击成功率高度相关，弃权机制在提升Forget Quality的同时几乎不降低模型效能；

**⚠️ 局限性**

依赖白盒访问、仅评估成员资格而非属性、仅英文数据、未给出严格理论证明、仅关注表示层，后续需扩展至权重、参数及多语言等场景。

---

## 356. Influence of Operator Expertise on Robot Supervision and Intervention

**arXiv ID:** 2601.15069 | [PDF](https://arxiv.org/pdf/2601.15069v1)

**作者:** Yanran Jiang `[一作]` (CSIRO), Cecile Paris `[通讯]` (CSIRO)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究通过在高保真仿真环境中让27名不同专业水平的操作者监督机器人进行隧道探索任务，探讨专家水平对干预策略、干预时机和求助偏好的影响。

**💡 创新点**

创新点在于首次系统比较了新手、中级和专家三类操作者在远程人机协作中的干预时机、方式及其对任务效率的影响，并揭示了机器人主动请求帮助与用户自主干预之间的交互差异。

**🔧 技术方法**

采用了CSIRO NavStack仿真平台、Hybrid A*路径规划、SLAM传感器数据与机器人自主导航算法，并使用问卷量表评估干预信心与满意度。

**📊 数据集**

使用了四个基于真实场景的隧道地图（来自CSIRO Data61 DARPA SubT挑战），每个地图设定了预定的失败点供实验统一化。

**📈 对比分析**

通过两因素ANOVA和Tukey检验比较干预时机、干预点位置和干预后覆盖面积，结果显示专家级操作者的干预更接近预设失败点、干预点更优越、干预后覆盖面积显著高于新手；机器人单独自主时在某些情境下甚至优于人类干预。

**⚠️ 局限性**

局限包括样本量偏小、专家组人数仅5人、仅在仿真环境测试、干预次数限制为一次，且未考察长期学习与适应过程。

---

## 357. The Responsibility Vacuum: Organizational Failure in Scaled Agent Systems

**arXiv ID:** 2601.15059 | [PDF](https://arxiv.org/pdf/2601.15059v1)

**作者:** Oleg Romanchuk `[一作]`, Roman Bondar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文对规模化代理代码部署中出现的责任真空（responsibility vacuum）进行结构化分析，阐明了责任归属与验证能力失配的根本原因。

**💡 创新点**

创新点在于：① 将责任真空形式化为权威与验证能力的结构性不匹配；② 推导出阈值条件下个性化责任失效的规模极限；③ 描述了CI 自动化如何放大这一失配并加速责任真空出现。

**🔧 技术方法**

采用理论建模、正式定义、系统性组织分析以及规模极限推导等技术手段；并结合案例说明（协调-验证差异的代理编排）。

**📊 数据集**

未使用实测数据集，而是基于文献回顾与假设情景进行概念性分析。

**📈 对比分析**

本研究未进行实验或性能对比，评估依据是理论阈值推导与案例说明，无法给出数值性能表现。

**⚠️ 局限性**

局限性包括：① 仅为概念性/结构性分析，缺乏经验验证；② 依赖于“标准部署假设”，在特定上下文外可能不适用；③ 未提供具体技术方案或治理工具，仅指出组织层面的重构方向。

---

## 358. A Curriculum-Based Deep Reinforcement Learning Framework for the Electric Vehicle Routing Problem

**arXiv ID:** 2601.15038 | [PDF](https://arxiv.org/pdf/2601.15038v1)

**作者:** Mertcan Daysalilar `[一作]` (University of Miami), Adam Meyers `[通讯]` (University of Miami)

**通讯引用:** 2562 | [OpenAlex ID](https://openalex.org/A5043134485)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为解决电动汽车时间窗配送问题（EVRPTW）提出一种基于课程学习的深度强化学习框架（CB‑DRL），通过分阶段逐步引入路由、能量和时间窗口约束实现训练稳定性和泛化能力。

**💡 创新点**

创新点在于：①引入三阶段约束课程（Topology→Energy→Time），拆分难度；②使用异构图注意力编码器并加入全局‑局部注意力与特征线性调制，显式建模车站、客户、仓库三类节点；③在PPO中采用阶段特定超参数、价值/优势裁剪与自适应学习率调度，提升收敛稳定性。

**🔧 技术方法**

技术核心包括：深度强化学习（PPO）、异构图注意力网络、课程学习调度器、约束解耦模块、可调超参数和自适应学习率。数据生成器基于 Solomon‑style 分布随机生成客户、充电站坐标与时间窗。

**📊 数据集**

使用自定义实例生成器，产生 N∈{5,10,20,30,40,50,100} 的可行 EVRPTW 实例，客户、充电站均在单位正方形内，配合多种时空分布（C, R, RC 等）。

**📈 对比分析**

对比方法包括：精确求解（MILP）、启发式（VNS）和无课程学习的标准 PPO。实验结果显示，CB‑DRL 在大规模实例（N≥40）中实现更低的最优性缺口（如 N=100 时 0% 近似最优），更高的可行率（74.7% vs 66.2%），并且推理时间比标准 PPO 快约 25%。

**⚠️ 局限性**

局限性：①训练仅在 N=10 的小型实例上完成，虽能泛化但对极大规模（N>100）或动态/随机环境的适应性尚未验证；②课程分阶段设计需要手工设定阈值，可能不适用于所有约束组合；③目前仅考虑确定性充电与行驶时间，未覆盖不确定性和实时调度。

---

## 359. Three-dimensional visualization of X-ray micro-CT with large-scale datasets: Efficiency and accuracy for real-time interaction

**arXiv ID:** 2601.15098 | [PDF](https://arxiv.org/pdf/2601.15098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 360. Memory Retention Is Not Enough to Master Memory Tasks in Reinforcement Learning

**arXiv ID:** 2601.15086 | [PDF](https://arxiv.org/pdf/2601.15086v1)

**作者:** Oleg Shchendrigin `[一作]` (MIRIAI), Aleksandr I. Panov `[通讯]` (MIRIAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两类专门评估强化学习记忆重写能力的基准任务（Endless T‑Maze 与 Color‑Cubes），并系统评估了循环、Transformer 与结构化外部记忆模型在这些任务中的表现。

**💡 创新点**

创新点在于将“记忆重写”从单纯的记忆保持转变为可持续、选择性更新的核心能力，并通过对比实验揭示了显式可学习忘记门（如LSTM）对重写任务的重要性。

**🔧 技术方法**

采用PPO‑LSTM、PPO‑GRU、PPO‑RNN、GTrXL、FFM、SHM与MLP等多种记忆增强的强化学习架构，并利用自定义的记忆重写环境进行训练与评估。

**📊 数据集**

使用自己设计的 Endlss T‑Maze（含固定与均匀长短通道设置）和 Color‑Cubes（多难度模式）作为数据集，测试不同记忆机制的适应性与泛化。

**📈 对比分析**

通过对比实验发现，PPO‑LSTM 在记忆保持、重写和泛化方面始终领先（近100%成功率），FFM 与 SHM 在固定模式下表现不错但在均匀模式下降速；GTrXL 在稀疏奖励环境中稳定性差；MLP 基线几乎不通过。整体表明显式可学习的忘记机制是实现记忆重写的关键。

**⚠️ 局限性**

局限性包括：研究仅涉及少数基准与模型，缺乏更广泛的环境与多样化记忆机制；实验主要聚焦离散动作空间，未考察连续控制；缺乏对记忆重写机制的理论分析与可解释性探讨。

---

## 361. DeepFedNAS: A Unified Framework for Principled, Hardware-Aware, and Predictor-Free Federated Neural Architecture Search

**arXiv ID:** 2601.15127 | [PDF](https://arxiv.org/pdf/2601.15127v1)

**作者:** Bostan Khan `[一作]` (Mälardalen University), Masoud Daneshtalab `[通讯]` (Mälardalen University)

**通讯引用:** 4176 | [OpenAlex ID](https://openalex.org/A5063193249)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DeepFedNAS 框架，实现联邦 NAS 的两阶段训练与搜索。

**💡 创新点**

创新点在于基于深度信息理论的多目标 fitness 函数、Pareto 路径训练课程以及消除预测器的搜索。

**🔧 技术方法**

采用信息熵与架构约束的 fitness、遗传算法、Pareto‑optimal supernet 训练、Zero‑cost fitness 搜索以及轻量 latency 预测器。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100、CINIC‑10 等公共数据集上评估。

**📈 对比分析**

与 FedAvg、FedNAS、FedPNAS、SuperFedNAS 等基线对比，准确率提升 0.8‑1.2%，参数/通信更优，搜索速度提升 61 倍。

**⚠️ 局限性**

局限在于对超网搜索空间的设计依赖、对不同硬件预测器的精度限制以及在极端异构场景下的梯度干扰仍需进一步改进。

---

## 362. A Myhill-Nerode Characterization and Active Learning for One-Clock Timed Automata

**arXiv ID:** 2601.15104 | [PDF](https://arxiv.org/pdf/2601.15104v1)

**作者:** Kyveli Doveri `[一作]` (Unaffiliated), B. Srivathsan `[通讯]` (Chennai Mathematical Institute)

**通讯引用:** 251 | [OpenAlex ID](https://openalex.org/A5047476786)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出了一种Myhill-Nerode风格的特征，用于识别一钟定时自动机（1-DTA）所接受的语言，并开发了学习其规范1-DTA的算法。

**💡 创新点**

创新点在于通过对1-DTA的“半整数”单词及其重置信息的新视角，提供了一个机器无关的特征和学习算法。

**🔧 技术方法**

使用了Myhill-Nerode定理的概念，结合重置函数和半整数单词的分析，发展了新的学习算法。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了1-DTA的语言特性和重置函数的构造。

**📈 对比分析**

与现有的学习算法相比，提出的算法能够有效地识别1-DTA的规范形式，并在理论上证明了其最小性。

**⚠️ 局限性**

限制在于当前的研究主要集中在一钟定时自动机上，可能无法直接推广到更复杂的定时自动机模型。

---

## 363. Parameter-Efficient Multi-Task Fine-Tuning in Code-Related Tasks

**arXiv ID:** 2601.15094 | [PDF](https://arxiv.org/pdf/2601.15094v1)

**作者:** Md Zahidul Haque `[一作]` (William and Mary), Antonio Mastropaolo `[通讯]` (William and Mary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在多任务代码相关任务（代码生成、翻译与摘要）中使用 QLoRA 进行参数高效微调，并与单任务微调及完整微调进行对比。

**💡 创新点**

创新点在于首次系统评估 QLoRA 在跨任务、多规模模型上对功能正确性与代码质量的影响，并提出了全面的非功能质量评估框架。

**🔧 技术方法**

采用 QLoRA（低秩适配 + 4‑bit 量化）对 Qwen2.5‑Coder 系列模型（0.5B、1.5B、3B）进行微调，配合梯度检查点、AdamW 等优化器。

**📊 数据集**

使用 CodeXGLUE、CoderEval 公开数据集，分别用于代码生成、翻译和摘要任务，保证训练集与测试集的一致性与可重复性。

**📈 对比分析**

通过 Pass@1、CodeBLEU、BLEU/METEOR/ROUGE/BERTScore 等功能指标和 Lizard/PMD/Pylint/SonarCloud/Roslyn 等静态分析工具进行质量评估；实验显示多任务 QLoRA 与单任务 QLoRA 在多数规模下功能正确性相当，且在大模型上常优于完整微调，代码质量差异不显著。

**⚠️ 局限性**

局限在于只评估了 Qwen2.5‑Coder 系列模型，且翻译与摘要在某些语言/规模下表现不稳定；未对多语言或更大规模模型的泛化能力做深入探讨。

---

## 364. LoRAP: Low-Rank Aggregation Prompting for Quantized Graph Neural Networks Training

**arXiv ID:** 2601.15079 | [PDF](https://arxiv.org/pdf/2601.15079v1)

**作者:** Chenyu Liu `[一作]` (Hong Kong Polytechnic University), Luca Rossi `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 7927 | [OpenAlex ID](https://openalex.org/A5016589479)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种低秩聚合提示（LoRAP）与节点提示（GPF-plus）结合的量化感知训练（QAT）方法，用于提升低位数（INT4）图神经网络（GNN）的推理精度。

**💡 创新点**

创新点在于将提示学习迁移到图聚合阶段，引入输入依赖的低秩提示来显式补偿量化误差，并通过融合核（fused kernel）显著降低提示注入的计算与内存开销。

**🔧 技术方法**

使用提示学习、低秩矩阵分解、量化感知训练框架（标准 QAT、Degree-Quant、A²Q、MixQ）以及 Triton GPU 编程实现的融合核。

**📊 数据集**

实验覆盖 9 个图数据集，包括节点分类（Cora、CiteSeer）、图分类（Reddit‑Binary、MNIST、CIFAR‑10、ZINC）以及大型 OGB 基准（ogb‑arxiv、ogb‑products、ogbn‑mag）。

**📈 对比分析**

与无提示、GPF-plus 以及原始 QAT 框架进行对比；LoRAP 在 INT4 量化下平均提升 4–15% 的准确率，某些配置甚至超越全精度 FP32 基线，并在 CPU/GPU 上实现 1.5–2× 的速度提升。

**⚠️ 局限性**

局限性包括仍需额外的提示参数，提示设计对不同 GNN 架构和任务的适应性需要进一步验证，且在极低位宽（如 INT4）下对提示规模的依赖可能导致训练不稳定。

---

## 365. Economic Warehouse Lot Scheduling: Breaking the 2-Approximation Barrier

**arXiv ID:** 2601.15068 | [PDF](https://arxiv.org/pdf/2601.15068v1)

**作者:** Danny Segev `[一作]` (Tel Aviv University), Danny Segev `[通讯]` (Tel Aviv University)

**通讯引用:** 1781 | [OpenAlex ID](https://openalex.org/A5044792131)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种新的近似算法，首次突破了传统经济仓库批量调度问题中长期以来的2近似上限，给出了期望成本不超过(2−17/5000+)的随机动态补货策略。

**💡 创新点**

创新点在于：①建立新的分析框架，直接与完全动态策略对比；②将问题拆分为体积类并使用最小权匹配构造“模拟分区”；③引入Po2‑同步定理，通过子1对的概率协调实现空间‑成本权衡的细粒度改进；③将经典SOSI方法与新的随机化技巧相结合。

**🔧 技术方法**

核心技术包括：经济订单量模型与SOSI优化；平均空间约束与凸松弛；最小权匹配（b‑matching）实现分区；概率性2‑取整（power‑of‑2 rounding）与子1对同步；随机化组合策略与期望成本分析；以及对不同稠密/稀疏体积类的分段处理。

**📊 数据集**

该工作为理论性质，未使用实测数据集，而是基于一般实例的形式化模型与复杂度分析。

**📈 对比分析**

与传统SOSI方案（成本≤2×最优）相比，本文得到的期望成本因子为2−17/5000+≈1.9966，显著小于2，证明了可行性；实验上（若实现）可进一步验证其性能提升。

**⚠️ 局限性**

局限性包括：算法时间复杂度仍高（O(|I|^O(1/ε)·2^O(1/ε^3))），主要针对随机策略；尚未实现确定性构造；对极端参数或大规模实例的可扩展性待进一步研究。

---

## 366. A Novel Cross-Domain Channel Estimation Scheme for OFDM

**arXiv ID:** 2601.15067 | [PDF](https://arxiv.org/pdf/2601.15067v1)

**作者:** Mingcheng Nie `[一作]` (University of Sydney), Yonghui Li `[通讯]` (University of Sydney)

**通讯引用:** 28810 | [OpenAlex ID](https://openalex.org/A5100448724)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种跨域信道估计（CDCE）方案，将OFDM系统的时频域（TF）与时延-多普勒域（DD）相结合，先在DD域通过扭曲卷积获得时延与多普勒粗估计，再在TF域利用LASSO求解衰减系数，显著提升了高移动场景下的信道估计精度。

**💡 创新点**

创新点在于首次利用DD域的扭曲卷积提取时延和多普勒信息，再将该信息构造为TF域稀疏字典，避免传统方法对完全随机帧字母或大规模矩阵求逆的依赖；同时，采用DD域的抗干扰特性在有数据干扰时仍能保持估计鲁棒。

**🔧 技术方法**

使用的主要技术包括：OFDM系统模型、时延-多普勒域的Symplectic Finite Fourier Transform（SFFT）、扭曲卷积（twisted-convolution）、L1正则化最小二乘（LASSO）求解器以及基于字典的稀疏重建。

**📊 数据集**

实验采用模拟OFDM信道，设置M=8，N=14，三条路径，最大时延索引2、最大多普勒索引3，使用随机生成的多普勒与时延参数、复高斯衰落系数进行仿真；未使用公开数据集。

**📈 对比分析**

与传统ST-LS、ST-LMMSE、全矩阵LMMSE、TF-LASSO等方案相比，CDCE在pilot-only和pilot+data两种场景下均实现了4~5 dB的NMSE提升，且不需要矩阵求逆或先验通道信息。

**⚠️ 局限性**

局限性包括：需要先验知道最大时延与多普勒范围；对扭曲卷积阈值设定较为敏感；在极低SNR或路径数极大时，粗估计误差可能累积导致性能下降。

---

## 367. Differential Privacy Image Generation with Reconstruction Loss and Noise Injection Using an Error Feedback SGD

**arXiv ID:** 2601.15061 | [PDF](https://arxiv.org/pdf/2601.15061v1)

**作者:** Qiwei Ma `[一作]` (Shenzhen University), Jun Zhang `[通讯]` (Shenzhen University)

**通讯引用:** 82598 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于误差反馈的差分隐私生成框架，在生成器训练过程中加入误差反馈、重建损失和噪声注入，实现更高质量的数据合成

**💡 创新点**

在DP‑SGD中首次引入误差反馈以消除梯度裁剪偏差，并将重建损失和噪声注入并入GAN训练，仅公开生成器以降低泄漏风险

**🔧 技术方法**

使用DP‑SGD+误差反馈（EFSGD）、VAE重建损失（L2）、高斯噪声注入、StyleGAN/ResNet生成器、判别器/分类器/编码器多组件训练以及RDP计数器

**📊 数据集**

MNIST、Fashion‑MNIST和CelebA（32×32）三个公开数据集

**📈 对比分析**

与GS‑WGAN、DP‑Sinkhorn、DP‑GAN‑DPAC等基线在IS、FID、gen2real accuracy等指标上进行对比；在MNIST/Fashion‑MNIST上基本持平或略优，在CelebA上IS/FID明显优于基线，CNN下gen2real提升至0.88

**⚠️ 局限性**

对简单数据集提升有限；生成图像尺寸受限于32×32，噪声注入可能影响清晰度；实验仅覆盖三类数据集，缺乏对更高分辨率或更复杂分布的验证

---

## 368. Systematic Evaluation of Hip Exoskeleton Assistance Parameters for Enhancing Gait Stability During Ground Slip Perturbations

**arXiv ID:** 2601.15056 | [PDF](https://arxiv.org/pdf/2601.15056v1)

**作者:** Maria T. Tagliaferri `[一作]` (Carnegie Mellon University), Inseung Kang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1282 | [OpenAlex ID](https://openalex.org/A5030877663)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

系统评估了双腿髋部外骨骼在地面滑倒扰动中的助力幅值和时长对步态稳定性的影响。

**💡 创新点**

首次系统探究助力幅值与时长交互对WBAM稳定性的作用，并与能量优化控制器对比，提出个性化稳定性导向控制策略。

**🔧 技术方法**

采用双腿髋关节外骨骼、三角波形推力控制、WBAM作为稳定性指标，利用RBF插值构建响应面并进行混合效应回归分析。

**📊 数据集**

收集8名健康成人在跑步机模拟滑倒扰动下的WBAM和OPUS感知稳定性数据，共112次试验。

**📈 对比分析**

与无外骨骼、无助力、能量优化基线控制器比较，最佳参数组将WBAM范围降低约27.4%（相对无外骨骼）和25.7%（相对基线），OPUS评分显著提升。

**⚠️ 局限性**

仅评估前后滑倒扰动，参数空间离散化受限，未实时检测扰动，样本量小，结果对不同扰动类型和老年人群的推广有限。

---

## 369. Facilitating Proactive and Reactive Guidance for Decision Making on the Web: A Design Probe with WebSeek

**arXiv ID:** 2601.15100 | [PDF](https://arxiv.org/pdf/2601.15100v1)

**作者:** Yanwei Huang `[一作]` (Hong Kong University of Science and Technology), Arpit Narechania `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5078755914)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了WebSeek——一种混合主动与被动的浏览器扩展，支持用户在网页上直接操作数据实例并通过LLM协同完成数据提取、清洗、可视化等任务

**💡 创新点**

提出以数据实例为中心的交互范式，并构建了针对Web数据任务的主动/被动AI辅助设计空间，强调透明度、可控性与多模态交互

**🔧 技术方法**

基于大型语言模型（Gemini‑2.5‑Flash）与工具调用框架，配合浏览器扩展技术（WXT）、Vega‑Lite可视化、网页DOM捕获与源追踪

**📊 数据集**

使用了来自22个领域的22个网页快照（共50个任务）以及两项真实任务（新闻事实核查与产品比较）进行评估

**📈 对比分析**

在技术评估中，WebSeek在平均20秒内生成建议，准确率达97.2%；在15名参与者的用户研究中，任务完成率100%，SUS平均73.11/100，用户对实时建议与直接操控满意度高（4.9/5）

**⚠️ 局限性**

局限包括：LLM上下文管理成本高、建议的可见性与可控性有限、画布规模扩大后可用性下降、评测数据集规模有限且未覆盖动态复杂网页

---

## 370. The Pictorial Cortex: Zero-Shot Cross-Subject fMRI-to-Image Reconstruction via Compositional Latent Modeling

**arXiv ID:** 2601.15071 | [PDF](https://arxiv.org/pdf/2601.15071v1)

**作者:** Jingyang Huo `[一作]` (Fudan University), Jianfeng Feng `[通讯]` (Fudan University)

**通讯引用:** 21599 | [OpenAlex ID](https://openalex.org/A5080019095)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了跨受试者零样本脑影像到图像重建问题，构建统一脑皮层表面 fMRI 数据集 UniCortex-fMRI，并提出 PictorialCortex 通过分解 fMRI 隐空间为刺激驱动、受试者、数据集、无关噪声四个成分来实现零样本跨受试者重建。

**💡 创新点**

创新点：①在统一脑皮层潜在空间中显式分离刺激驱动与受试者/数据集/噪声因子；②提出配对因子分解与重因子一致性正则化，以及 surrogate latent 聚合策略；③整合四大视觉 fMRI 数据集生成统一评估基准 UniCortex-fMRI。

**🔧 技术方法**

使用 Transformer 自动编码器预训练得到统一脑皮层潜在空间，构建 Latent Factorization–Composition Module（因子化器+合成器），并通过 Paired Factorization & Reconstruction 与 Re‑Factorizing Consistency Regularization 进行训练；重建阶段采用 IP‑Adapter 扩散模型。

**📊 数据集**

基准数据集包括 NSD、BOLD5000、NOD、HCP‑Movie 合并而成的 UniCortex‑fMRI；预训练用 UK Biobank 大规模 fMRI 数据。

**📈 对比分析**

与 MindBridge、MindEye2 以及先前的 NeuroPictor 在低阶（像素相关、LPIPS、AlexNet）和高阶（EffNet、SwAV、Inception、CLIP）指标上对比，PictorialCortex 在大多数指标上提升约 10–30% 以上，达成零样本跨受试者重建的 state‑of‑the‑art。

**⚠️ 局限性**

局限性：①仍需大量预训练数据，受试者与刺激分布不均导致部分指标（如像素相关）略低；②对极其复杂或未见刺激的细节重建仍显模糊；③目前仅针对 fMRI，未扩展到多模态（EEG 等）或多视角场景。

---

## 371. Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning

**arXiv ID:** 2601.15115 | [PDF](https://arxiv.org/pdf/2601.15115v1)

**作者:** Shuonan Yang `[一作]` (University of Exeter), Zeyu Fu `[通讯]` (University of Essex)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练、可解释的多阶段对抗推理框架MARS，用于检测视频中的仇恨内容。

**💡 创新点**

创新点在于：① 采用训练‑free 方式，消除对大规模标注数据的依赖；② 通过先客观描述再分别构建仇恨与非仇恨假设，最终对比证据做出决策，从而实现可解释推理；③ 在多模态推理中引入对抗式证据对齐，显著降低误报。

**🔧 技术方法**

核心技术包括：大规模视觉‑语言模型（如 Qwen2.5‑VL、Llama4、GPT‑5、Gemini‑2.5）在推理时多步提示；四阶段推理流程（客观描述、仇恨假设、非仇恨假设、元分析综合）；以及基于证据权重的元合成函数。

**📊 数据集**

使用了 HateMM（1083条英语视频）和 MultiHateClip 中文子集（959 条视频）两个公开仇恨视频基准。

**📈 对比分析**

与训练‑free 的简单提示、Chain‑of‑Thought（CoT）提示以及多种训练‑based 模型（CMFusion、MoRE 等）对比，MARS 在准确率、宏 F1 以及仇恨类精确度上均实现了约 10% 的提升，并在精确度上保持优势，表现出更低的误报率。

**⚠️ 局限性**

局限性包括：① 仍需依赖现有 VLM 的推理能力，模型尺寸越大性能越好；② 在低召回场景下仍有不足；③ 对跨语言和跨文化的适应性尚需进一步验证；④ 可能存在模型生成的证据与推理的真实性检验难度。

---

## 372. Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network

**arXiv ID:** 2601.15110 | [PDF](https://arxiv.org/pdf/2601.15110v1)

**作者:** Aoran Liu `[一作]` (University of Sydney), Zhiyong Wang `[通讯]` (University of Sydney)

**通讯引用:** 30982 | [OpenAlex ID](https://openalex.org/A5100614129)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种解析‑更新分离的图神经网络（Pb4U‑GNet）用于实现跨分辨率的服装仿真。

**💡 创新点**

创新点在于动态传播深度控制与几何感知的更新缩放两大机制，能够自适应调整信息接收域并保持物理一致性，从而显著提升高分辨率网格的泛化能力。

**🔧 技术方法**

使用技术包括解析‑更新分离的消息传播、基于边长的动态传播步数计算、几何感知的加权更新缩放、相对几何特征编码、基于MLP的消息与更新函数、MeshGraphNet块以及六项物理自监督损失。

**📊 数据集**

在VTO数据集上进行实验，训练仅使用最低分辨率（约11K三角形）的服装网格，评估时使用更高分辨率（18K、25K、38K）以及不同服装类别。

**📈 对比分析**

与MGN、HOOD、ESLR、CCRAFT等主流基线进行对比，采用物理损失指标评估；Pb4U‑GNet在低分辨率时与基线相近，但在高分辨率上显著降低了伸展、碰撞等损失，模拟更稳定，推理时仅略高于基线的延迟。

**⚠️ 局限性**

局限性包括：对极高分辨率网格仍可能出现数值不稳定；仅在低分辨率数据上训练，缺乏对更复杂物理约束的直接监督；推理速度虽然改进但仍高于纯物理基线，且对内存占用尚未进行系统评估。

---

## 373. DeLog: An Efficient Log Compression Framework with Pattern Signature Synthesis

**arXiv ID:** 2601.15084 | [PDF](https://arxiv.org/pdf/2601.15084v1)

**作者:** Siyu Yu `[一作]` (Peking University), Ying Li `[通讯]` (ByteDance)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了名为 DeLog 的日志压缩框架，利用单遍模式签名合成实现基于模式的分组压缩；

**💡 创新点**

提出解析准确率并不决定压缩比的观点，核心创新是以模式同质化为目标的分组策略，并通过结构与上下文信息合成签名；

**🔧 技术方法**

采用单通道扫描构建动态特征池、模式签名合成规则、按类别选择 delta/弹性/字典等编码，并提供轻量版 DeLog‑L；

**📊 数据集**

使用 16 个公开 LogHub 1.0/2.0 基准数据集以及 ByteDance 的 10 个匿名生产日志 LogA~LogJ；

**📈 对比分析**

与 LogReducer、LogShrink、Denum 等基线对比，平均压缩比提升约 9%‑15%，压缩速度提升 1.8×‑20×，解压速度提升 2×以上；

**⚠️ 局限性**

目前仅基于语法结构，无法捕获语义等价词，且在极端多变日志中仍可能出现解压失败，后续计划引入 LLM 语义引擎完善。

---

## 374. Enhancing Few-Shot Out-of-Distribution Detection via the Refinement of Foreground and Background

**arXiv ID:** 2601.15065 | [PDF](https://arxiv.org/pdf/2601.15065v1)

**作者:** Tianyu Li `[一作]` (University of Electronic Science and Technology of China), Xiaofeng Zhu `[通讯]` (Hainan University)

**通讯引用:** 13331 | [OpenAlex ID](https://openalex.org/A5037340898)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个轻量级的可插拔框架 FoBoR，用于改进现有的前景-背景分解方法，提高少样本 OOD 检测性能。

**💡 创新点**

创新点在于两大模块：自适应背景抑制（ABS）根据背景-类相关性动态加权背景信息；以及可混淆前景修正（CFR）识别并抑制与真类相似的前景补丁。

**🔧 技术方法**

主要技术包括 CLIP 视觉-文本对齐、局部到全局注意力计算背景相关性、基于二元熵最大化的前景混淆抑制，以及多模态相似度融合来选取可混淆类。

**📊 数据集**

实验使用 ImageNet‑1k 作为 ID 数据集，并在 iNaturalist、SUN、Places、Texture 等 OOD 数据集以及 OpenOOD 基准（包括近、远 OOD）进行评估。

**📈 对比分析**

与基线方法（MCM、GL‑MCM、LoCoOp、SCT、Mambo、FA 等）比较，FoBoR 在 1、4、16 shot 设置下均显著降低 FPR95 并提升 AUROC，尤其在硬 OOD 场景中平均提升约 1%–2%。

**⚠️ 局限性**

局限性在于仍需手动设定参数 α、β、n_class、n_patch 等，对极端难以收敛的样本可能不足以完全消除背景或前景混淆的影响。

---

## 375. Tracing 3D Anatomy in 2D Strokes: A Multi-Stage Projection Driven Approach to Cervical Spine Fracture Identification

**arXiv ID:** 2601.15235 | [PDF](https://arxiv.org/pdf/2601.15235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 376. SpooFL: Spoofing Federated Learning

**arXiv ID:** 2601.15055 | [PDF](https://arxiv.org/pdf/2601.15055v1)

**作者:** Isaac Baglin `[一作]` (University of Surrey), Simon Hadfield `[通讯]` (University of Surrey)

**通讯引用:** 5999 | [OpenAlex ID](https://openalex.org/A5091184063)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在联邦学习中通过生成完全无关的合成样本来误导深度泄露攻击的防御方法（SpooFL）

**💡 创新点**

将防御视为“spoofing”问题，使用外部数据训练的生成模型生成与真实任务无关的样本，防止攻击者得到有意义信息，并引入新的隐私指标Private Leakage Confidence（PLC）

**🔧 技术方法**

使用条件生成对抗网络（R3GAN）进行潜在向量优化，结合模型轨迹匹配损失、FedAvg协议以及多种深度泄露攻击（GradInversion、DLF、SME）进行评估

**📊 数据集**

主要在CIFAR‑10和STL‑10图像分类数据集上进行实验，生成对抗网络在ImageNet上预训练

**📈 对比分析**

与噪声注入、梯度裁剪、压缩、DCS、FedADG等传统或生成式防御相比，SpooFL在PLC、SSIM、FMSE、PSNR、LPIPS等指标上表现更好，同时保持与原模型相近的准确率，且不增加运行时间

**⚠️ 局限性**

SpooFL的合成数据集需针对特定网络重新蒸馏，泛化到其他架构受限；生成模型的质量和训练成本仍是潜在瓶颈

---

## 377. Towards Standardizing OTFS: A Candidate Waveform for Next-Generation Wireless Networks

**arXiv ID:** 2601.15048 | [PDF](https://arxiv.org/pdf/2601.15048v1)

**作者:** Mingcheng Nie `[一作]` (University of Sydney), Yonghui Li `[通讯]` (University of Sydney)

**通讯引用:** 28810 | [OpenAlex ID](https://openalex.org/A5100448724)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了OTFS调制波形在6G网络中的标准化与实现，探讨了其在现有OFDM系统中的兼容性、低复杂度MIMO预编码方案以及在时延-多普勒域的联合感知与通信（ISAC）能力。

**💡 创新点**

创新点包括：①基于OFDM的SFFT和ZT实现方案实现OTFS与4G/5G的向后兼容；②提出了近线性复杂度的THP预编码，用以解决多用户MIMO OTFS中的三类干扰；③阐述了OTFS在时延-多普勒域的感知优势与基底函数对模糊函数的影响，提出DD-ISAC的统一架构。

**🔧 技术方法**

核心技术包括：SFFT/ISFFT、Z变换（ZT）及其离散实现、DFT‑s‑OFDM结构、Tomlinson‑Harashima预编码、DD域信道估计、窗口设计（如Dolph–Chebyshev）、最大似然估计及误差分析。

**📊 数据集**

主要使用仿真数据：MIMO-OTFS/OFDM系统在8天线4用户场景下的Rayleigh衰落信道、指数衰减功率延迟分布，仿真参数N=16、M=32、延迟/多普勒索引l_max=5、k_max=7。

**📈 对比分析**

与OFDM（ZF、MRT预编码）和OTFS-THP在总谱效率、误码率以及延迟/多普勒感知精度（NMSE）上进行比较。实验显示：OTFS‑THP在高SNR时超过OFDM的总谱效率，误码率接近CRB；在感知方面，OTFS在接近CRB的延迟/多普勒分辨率上优于传统OFDM。

**⚠️ 局限性**

局限性包括：①OTFS需要完整帧才能解调，导致较高延迟；②THP预编码在低SNR下有模数损失；③DD域信道估计与干扰消除对完整帧和完整信道信息依赖较大；④实际硬件实现与功率放大器非线性、PAPR问题仍待进一步优化。

---

## 378. HyperNet-Adaptation for Diffusion-Based Test Case Generation

**arXiv ID:** 2601.15041 | [PDF](https://arxiv.org/pdf/2601.15041v1)

**作者:** Oliver Weißl `[一作]` (Technical University of Munich), Andrea Stocco `[通讯]` (Technical University of Munich)

**通讯引用:** 2453 | [OpenAlex ID](https://openalex.org/A5027652385)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的测试案例生成方法——HyperNet Adaptation，利用实例级超网络对扩散模型进行自适应调优，直接生成能触发被测深度学习模型错误的输入。

**💡 创新点**

创新点在于：①不依赖目标标签或失效样本数据，而是通过把控制信号反向从模型输出获取；②采用与ControlNet相似的架构但反向优化，能够在单一推理周期内完成全局梯度传播；③使用实例级调优，避免大规模重训练，显著提升生成效率和控制精度。

**🔧 技术方法**

核心技术包括：扩散模型（Stable Diffusion、REPA‑E 及 UNet‑based LDM）、HyperNet（零层调节）、AdamW+OneCycle 学习率调度、控制投影（将 SUT 输出映射为空间控制信号）以及多目标损失（视觉相似度 + 行为引导）。

**📊 数据集**

实验数据集涵盖 ImageNet 1K（10 类）、CelebA（10 个属性）和 KITTI/自定义驾驶场景（目标检测），并使用对应的预训练 SUT（WideResNet‑50、ResNet‑50 Attribute Classifier、YOLOv8）。

**📈 对比分析**

与 StyleGAN‑based 及 Latent‑Perturbation diffusion 方法对比，HyperNet 在误分类率（100%）、逃逸率（0）和生成质量（MS‑SSIM、FID、人类评估）均优于基线；在预算与运行时上也展现更低的 SUT 评估次数和相对更快的推理速度。

**⚠️ 局限性**

局限性包括：对扩散模型的依赖导致需要较高显存；超网络调优对学习率高度敏感，需精细调参；目前仅验证于图像分类/属性与目标检测三类任务，尚未推广至其他模态；若目标模型对输入变化鲁棒性极强，方法效果可能受限。

---

## 379. How to Verify a Turing Machine with Dafny

**arXiv ID:** 2601.15230 | [PDF](https://arxiv.org/pdf/2601.15230v1)

**作者:** Edgar F. A. Lederer `[一作]` `[通讯]` (University of Applied Sciences and Arts Northwestern Switzerland), Edgar F. A. Lederer (University of Applied Sciences and Arts Northwestern Switzerland)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

对两台典型的单带图灵机（用于判定括号平衡语言和二进制零串长度为 2 的幂的语言）进行了完整的形式化验证，证明它们在所有输入下都能够正确终止并给出正确答案。

**💡 创新点**

首次使用 Dafny 这一程序验证器来证明单带图灵机的正确性，并展示了如何把图灵机的转移函数、状态、带子内容等映射到可验证的程序结构中，解决了传统证明中“无结构代码难以维护、缺乏工具支持”的难题。

**🔧 技术方法**

主要技术为：Dafny 语言、Hoare 逻辑与自动证明、ghost 变量与谓词用于表述中间状态的性质、循环不变式与递归度量、以及手工构造的数学辅助 Lemma 以辅助证明。

**📊 数据集**

该工作不使用外部数据集；验证对象是图灵机的转移函数和输入词的符号集合，所有证明均为纯形式化推导。

**📈 对比分析**

与手工证明相比，Dafny 的自动化验证仅需几秒钟完成（新版本 4.10 在 VS Code 中验证更快），证明过程可重复、可审计，并且提供了对不终止与终止两种情况的完整证明。

**⚠️ 局限性**

局限性在于：目前只对两台极其简单的单带图灵机进行了验证，规模更大、状态更复杂或多带图灵机的验证仍面临组合性与证明规模爆炸的问题；此外，证明高度依赖手工构造的中间不变式和数学 Lemma，缺乏通用模板。

---

## 380. A Computer Vision Hybrid Approach: CNN and Transformer Models for Accurate Alzheimer's Detection from Brain MRI Scans

**arXiv ID:** 2601.15202 | [PDF](https://arxiv.org/pdf/2601.15202v1)

**作者:** Md Mahmudul Hoque `[一作]` (CCN University of Science and Technology), Farha Ulfat Mahi `[通讯]` (Daffodil International University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

对比分析了多种CNN与Transformer模型以及其混合模型在脑MRI多分类阿尔茨海默病检测中的表现，并提出Evan_V2混合模型。

**💡 创新点**

提出通过特征级融合将十个CNN和Transformer的输出组合成单一预测的混合架构，显著提升分类准确率至99.99%。

**🔧 技术方法**

使用预训练CNN（EfficientNetB0、ResNet50等）、Vision Transformer及自定义Transformer，结合特征融合与加权平均，并采用数据增强、Adam优化和早停。

**📊 数据集**

使用OASIS公开MRI数据集（约80万张2D切片），划分为四类：非痴呆、轻度痴呆、中度痴呆、极轻度痴呆。

**📈 对比分析**

在相同训练设置下，使用准确率、精确率、召回率、F1、ROC‑AUC评估，CNN平均精度≈98%，Transformer约95%，Evan_V2则达到99.99%准确率、0.9989 F1和0.9968 ROC‑AUC，表现最优。

**⚠️ 局限性**

混合模型计算量大、推理速度慢，且对超参数调优敏感，未在外部独立数据集上验证泛化能力。

---

## 381. Where Do AI Coding Agents Fail? An Empirical Study of Failed Agentic Pull Requests in GitHub

**arXiv ID:** 2601.15195 | [PDF](https://arxiv.org/pdf/2601.15195v1)

**作者:** Ramtin Ehsani `[一作]` (Drexel University), Preetha Chatterjee `[通讯]` (Drexel University)

**通讯引用:** 321 | [OpenAlex ID](https://openalex.org/A5049106181)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对五种 AI 编码代理在 100+ 星 GitHub 项目中提交的 33,596 个 PR 进行大规模量化与定性分析，探讨其合并成功率及失败原因。

**💡 创新点**

首次系统评估 AI 编码代理在真实仓库中的 PR 合并表现，并构建了四层级的拒绝模式层级分类，揭示了社会技术与协作问题对合并的影响。

**🔧 技术方法**

采用 Cliff’s delta、核密度估计与逻辑回归等统计方法进行量化分析，结合开放式编码、Cohen’s κ 等手段进行定性分析，构建拒绝模式 taxonomy。

**📊 数据集**

使用 AIDev-pop 数据集，收集了五款代理在 100+ 星 GitHub 项目中提交的 33,596 个 PR。

**📈 对比分析**

通过比较 merge‑rate、代码变更规模、CI 成功率与审查互动等指标，发现文档、CI 与构建类 PR 合并率高，较大规模变更和 CI 失败导致合并率下降，效应量为小至中等。

**⚠️ 局限性**

局限包括样本仅限高星项目、缺乏代理内部决策洞察、未覆盖私有 PR、量级大导致显著性差异可能误导、人工标注存在主观性与一致性限制。

---

## 382. Integrating OTFS in Airplane-Aided Next-Generation Networking

**arXiv ID:** 2601.15166 | [PDF](https://arxiv.org/pdf/2601.15166v1)

**作者:** Ashok S Kumar `[一作]` (Indian Institute of Technology Madras), Sheetal Kalyani `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 1950 | [OpenAlex ID](https://openalex.org/A5046290878)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种基于OTFS调制的飞机辅助下一代网络系统，并采用平面阵列天线与空前消除波束（NSB）技术实现高空高速移动环境下的可靠通信。

**💡 创新点**

创新点在于将OTFS与NSB波束成形结合，在高移动速度、长传播距离以及大尺寸天线阵列的空中平台上验证OTFS的鲁棒性，首次展示OTFS在高空平台与地面用户之间的性能优势。

**🔧 技术方法**

主要技术包括OTFS时频-延迟多普勒域调制、平面阵列天线、NSB波束成形、DD域零迫（ZF）均衡、以及多用户干扰建模。

**📊 数据集**

论文使用仿真数据，参数取自28 GHz毫米波、100×100平面阵列、Rician因子10 dB、速度150 m/s等，未使用公开数据集。

**📈 对比分析**

通过与传统OFDM基准在不同Rician因子、飞机高度、速度、阵列尺寸等场景下进行比特误码率（BER）比较，结果显示OTFS在所有设置下均优于OFDM，最高可获得约2 dB的BER提升。

**⚠️ 局限性**

局限性包括：仅在理想化仿真环境下验证，未考虑实际飞行中的多路径、非静态干扰、功率限制与硬件实现复杂度；仅采用单路径LoS+单路径NLoS模型，且未评估不同调制格式和多载波方案的兼容性。

---

## 383. Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data

**arXiv ID:** 2601.15158 | [PDF](https://arxiv.org/pdf/2601.15158v1)

**作者:** Yuval Ran-Milo `[一作]` (Some Institute), Nadav Cohen `[通讯]` (Some Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过理论分析和实验研究，证明在只有最终答案奖励的强化学习（RL）设置下，Transformer模型能够自发地学习到多步推理（Chain-of-Thought，CoT）算法，并且这种推理的出现完全依赖于训练数据中包含“简单”实例（即需要较少推理步骤的样本）。

**💡 创新点**

创新点在于：
- 提供了第一个针对稀疏奖励情形下CoT出现的严格理论证明，说明梯度流会因数据分布而收敛到高效的遍历算法；
- 明确表明只有在训练分布对简单实例赋予足够权重时，梯度下降才能高效学习；
- 发现梯度流具有对高效推理算法的隐式偏置，即使不惩罚推理长度；
- 通过实验验证理论，并展示在真实大型语言模型上的跨长度泛化与数据分布的重要性。

**🔧 技术方法**

主要技术手段包括：
- 单层Transformer的梯度流动力学分析（连续时间梯度流、注意力矩阵训练，值矩阵固定）；
- 通过引入“前置知识”（minimal task proficiency）与对称初始化来简化理论推导；
- 在实验中使用REINFORCE和GRPO强化学习算法；
- 对比单层Transformer在合成链识别任务与在Qwen 2.5 3B模型上的数学推理任务。

**📊 数据集**

使用的数据集：
- 合成链识别任务（graph traversal，两个长度为 n 的链，起点在链内不同位置，目标是终点）；
- 真实数学推理任务：自然语言描述的连通方程组（affine equations），需要按依赖链顺序推导出目标变量；
- 对这两个任务分别构造了包含简单实例（短链）与仅包含难度高实例（长链）的训练分布。

**📈 对比分析**

比较方法与性能：
- 在合成任务中，模型在包含足够简单实例的训练集上达到了 99%–100% 的任务准确率，并且几乎 100% 的推理轨迹为有效链遍历；
- 在仅包含困难实例的训练集上，模型几乎没有进展，准确率维持在 1% 左右；
- 在Qwen 2.5 3B 的数学推理任务中，训练于 15-Uniform（包含多长度实例）得到 98.9% 的最终答案准确率，且所有正确答案都采用高效的逐步推理；
- 进一步测试跨长度泛化：在 15-Hard（最大推理步数）上，10-Uniform 训练的模型准确率可达 80% 以上，而 5-Uniform 训练的模型几乎无法学习；
- 结果表明：包含简单实例的训练分布能显著提升学习效率和泛化能力。

**⚠️ 局限性**

限制与未来工作：
- 理论分析仅在单层Transformer、对称初始化、预训练提供“最小任务熟练度”等假设下成立，无法直接推广到多层大模型；
- 对真实大模型的实验主要在单一数学推理任务上，尚需在更多多样化任务上验证；
- 对梯度流的连续时间分析与离散梯度下降的差距未完全消除，实际训练中可能受到学习率、批大小等因素影响；
- 仅关注最终答案奖励，未探索对中间推理步骤的软监督或更复杂奖励结构；
- 需要进一步研究如何在实际数据集（如自然语言推理、代码生成等）中设计简单实例来引导推理学习。

---

## 384. Vehicle Routing with Finite Time Horizon using Deep Reinforcement Learning with Improved Network Embedding

**arXiv ID:** 2601.15131 | [PDF](https://arxiv.org/pdf/2601.15131v1)

**作者:** Ayan Maity `[一作]` (Indian Institute of Technology Kharagpur), Sudeshna Sarkar `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 2551 | [OpenAlex ID](https://openalex.org/A5024219981)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于深度强化学习的车辆路径规划方法，利用GAT-Edge图注意力网络和交叉注意力实现路网嵌入，解决有限时域内的车辆路由问题。

**💡 创新点**

创新点在于：①在GAT层加入边特征以捕捉路网时延信息；②构建跨注意力的全局图嵌入以获取全局路由上下文；③将剩余时域直接注入嵌入模块，提升决策质量。

**🔧 技术方法**

使用技术包括图注意力网络（GAT-Edge）、交叉注意力机制、REINFORCE策略梯度、二值邻接矩阵与边特征融合、状态掩码和多头注意力。

**📊 数据集**

实验数据集涵盖：随机生成的欧几里得网络（EN‑50、EN‑100）、东马萨诸塞高速公路网络（EMA‑50、EMA‑100、EMA‑150）、维也纳市区网络（Vienna‑160、Vienna‑300）及其确定性与随机性两种请求模式。

**📈 对比分析**

与遗传算法、变邻域搜索、多重背包近似、DRL‑Transformer和DRL‑S2V等基线进行对比。结果显示，PG‑GAT‑Edge在客户服务率上提升了约50–90%，并且决策时间比遗传算法快90%以上、比DRL‑Transformer快35%。

**⚠️ 局限性**

局限性包括：仅针对单车单路径问题；对大规模多车或带时间窗的场景尚未验证；需要充分的RL训练样本，且对网络拓扑变化的实时适应性有限。

---

## 385. Is Peer Review Really in Decline? Analyzing Review Quality across Venues and Time

**arXiv ID:** 2601.15172 | [PDF](https://arxiv.org/pdf/2601.15172v1)

**作者:** Ilia Kuznetsov `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 25133 | [OpenAlex ID](https://openalex.org/A5027450194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个统一的、可在大规模评审数据上实现的同行评审质量评估框架，并在ICLR、NeurIPS和ACL Rolling Review三大AI/ML会议中，跨年度对评审质量进行定量比较，最终发现中位数质量并未出现系统性下降。

**💡 创新点**

创新点主要包括：
① 使用LLM实现评审结构统一（itemization）并保持内容完整；
② 设计了涵盖实质性、可操作性、依据性三大维度的多元量化schema，并同时提供轻量化与LLM基测量；
③ 将该框架应用于跨时间、跨会议的比较研究，提供了基于Bootstrap检验的统计显著性分析。

**🔧 技术方法**

技术手段：
- LLM（GPT‑5.2）用于评审拆分与分区；
- 轻量化指标如字符计数、条目计数、显式链接与请求句子计数；
- RoBERTa、fine‑tuned LLM模型用于计算可操作性与依据性评分；
- 文本相似度匹配、Spearman相关、Bootstrap 10k迭代检验。

**📊 数据集**

数据集：
- ICLR、NeurIPS公开评审数据（OpenReview）每年随机1k条；
- ACL Rolling Review（ARR）完整数据（含2022年1k条以上）；
- 通过API抓取、预处理并统一为flattened/​itemized格式。

**📈 对比分析**

比较方法：
- 将各指标归一化后取均值得到综合质量得分Q；
- 采用Bootstrap检验（10k次）比较不同年份/会议的中位数Q；
- 轻量化与LLM指标间的Spearman相关约0.6，表明轻量化可在计算成本显著降低的前提下，保持一定的评估准确性。

**⚠️ 局限性**

局限性：
- 仅覆盖英语评审，且仅限于AI/ML会议；
- 评审拆分的LLM结果缺乏金标准验证；
- LLM预测仍存在误差，尤其是可操作性与依据性评分；
- 数据收集政策（如仅接受论文公开）导致样本偏差；
- 仅评估三大维度，未涵盖所有可能影响质量的因素。

---

## 386. Feasibility Preservation under Monotone Retrieval Truncation

**arXiv ID:** 2601.15241 | [PDF](https://arxiv.org/pdf/2601.15241v1)

**作者:** Sean Plummer `[一作]` `[通讯]` (BambooHR), Sean Plummer (BambooHR)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究检索系统在截断证据时，是否能保留查询的可解性。以结构化模型定义证据、槽、兼容关系，提出证书与 Noetherian 检索概念，并证明单调截断能保证单个查询的有限可见性，进一步给出查询类统一检索深度的必要与充分条件，并用反例阐明边界。

**💡 创新点**

创新点在于：①把检索正确性重新定义为“可行性保留”而非传统的相关性评分；②提出“证书”机制和“有限生成”条件，分别对应单查询的有限可见性与查询类的统一检索深度；③给出单调截断与非单调截断、有限生成与非有限生成的完整对照表，并证明这些条件的必要性和充分性。

**🔧 技术方法**

主要技术手段是：结构化形式化（定义槽集合、兼容关系、证书集合）；证明理论（利用递增序列、极限、Noetherian 论证）；构造性反例；以及与经典 CSP、数据库查询、抽象解释等领域概念的对比。

**📊 数据集**

没有使用真实数据集；实验仅以符号集合（如 {a, b}、{e₁, e₂, …}）构造示例和反例来说明理论结论。

**📈 对比分析**

论文不做数值实验，而是通过理论证明对比不同截断方式与查询类的可行性。结果显示：单调截断保证每个可解查询在某有限深度内可见；若查询类满足有限生成，则存在统一的深度；若不满足，则无统一深度；非单调截断则可能导致即使在极限可解也无任何有限深度可见。

**⚠️ 局限性**

局限性：①仅关注可行性保留，不涉及检索优化、排名质量或算法复杂度；②需要单调性与有限生成的前提，实际检索系统中可能难以满足；③证书设计与管理在大规模场景中可能代价高昂；④未考虑概率或近似约束，仅为严格的存在性框架。

---

## 387. When Agents Fail: A Comprehensive Study of Bugs in LLM Agents with Automated Labeling

**arXiv ID:** 2601.15232 | [PDF](https://arxiv.org/pdf/2601.15232v1)

**作者:** Niful Islam `[一作]` (Oakland University), Mohammad Wardat `[通讯]` (Oakland University)

**通讯引用:** 316 | [OpenAlex ID](https://openalex.org/A5004183493)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性地对LLM代理的bug进行分类、根因及影响的研究，并开发了一款基于ReAct框架的自动标注工具；

**💡 创新点**

首次构建专门针对LLM代理的bug分类体系，并将外部文档、论坛、GitHub讨论等信息通过工具搜索与摘要集成到自动标注流程中；

**🔧 技术方法**

采用ReAct框架、工具调用（文档/论坛/讨论抓取）、Redis缓存、Pydantic结构化输出，并以Gemini 2.5 Flash为核心LLM；

**📊 数据集**

使用了1,187条来自Stack Overflow、GitHub（issues/commits）以及Hugging Face论坛的bug帖子与代码片段；

**📈 对比分析**

与Encoder、Zero‑shot、One‑shot、传统ReAct等基线模型比较，Gemini 2.5 Flash在bug类型/根因/影响的F1得分最高（约0.57–0.89），成本仅约0.01 USD/条，平均耗时约30–35 秒；

**⚠️ 局限性**

依赖外部网站抓取与工具，若网站不可用或网络速度下降会影响性能；对大型代码基适用性尚未验证；标签体系与人工标注仍存在一定主观性。

---

## 388. MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI

**arXiv ID:** 2601.15222 | [PDF](https://arxiv.org/pdf/2601.15222v1)

**作者:** Stavrow A. Bahnam `[一作]` (Delft University of Technology), Guido C. H. E. de Croon `[通讯]` (Delft University of Technology)

**通讯引用:** 5441 | [OpenAlex ID](https://openalex.org/A5058902676)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 MonoRace 方法，实现仅使用单镜头摄像机与 IMU 的全程自主竞速无人机，并在 2025 年 Abu Dhabi Autonomous Racing Competition (A2RL) 赛中击败所有 AI 对手与三名世界冠军。

**💡 创新点**

创新点包括：① 基于门分割的鲁棒状态估计与多门 PnP 结合；② 离线基于门重投影的自监督优化，用于微调相机外参；③ 通过模型预测补偿 IMU 饱和，提升高速飞行的估计稳健性；④ 直接输出低层电机指令的轻量 G&CNet，消除传统控制环。

**🔧 技术方法**

采用的技术：门分割网络 GateNet、角点提取算法 QuAdGate、EKF 视觉‑惯性融合、PPO 训练的三层全连接 G&CNet、域随机化仿真、基于门重投影的 IoU 优化。

**📊 数据集**

使用数据集：由比赛现场录制的真实航拍图像与合成门图像混合构成的自定义数据集；未使用公开数据集。

**📈 对比分析**

与双目 VIO + NMPC 等基线对比，MonoRace 以 16.56 s 的最快完成时间、最高 28.23 m/s 的速度赢得比赛，并在 AI‑vs‑Human 淘汰赛中击败三名 FPV 世界冠军。

**⚠️ 局限性**

局限性：缺乏多机碰撞规避与多机协同功能；对不同门形状的适应性有限；控制网络对视觉误差仍有一定鲁棒性不足，需在更复杂、无标记环境中进一步验证。

---

## 389. ScenDi: 3D-to-2D Scene Diffusion Cascades for Urban Generation

**arXiv ID:** 2601.15221 | [PDF](https://arxiv.org/pdf/2601.15221v1)

**作者:** Hanlei Guo `[一作]` (Zhejiang University), Yiyi Liao `[通讯]` (Zhejiang University)

**通讯引用:** 2663 | [OpenAlex ID](https://openalex.org/A5018811297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种3D‑to‑2D扩散级联框架，用于生成可控且高质量的城市场景。

**💡 创新点**

创新点在于先通过3D潜在扩散模型生成粗糙的3D高斯场景，再利用2D视频扩散模型对细节进行细化，并支持文本、布局和边界框等多种控制信号。

**🔧 技术方法**

技术栈包括Voxel‑to‑3DGS VQ‑VAE、3D潜在扩散模型、2D视频扩散模型（SVD / Wan2.1）、深度估计器、文本编码器以及跨模态条件机制。

**📊 数据集**

训练与评估使用了Waymo和KITTI‑360这两个真实驾驶数据集。

**📈 对比分析**

与DiscoScene、CC3D、UrbanGen、GaussianCity、UrbanArchitect、Vista、Gen3C等多种基线比较，在FID、KID、FVD和相机误差等指标上均实现了最优或接近最优的表现。

**⚠️ 局限性**

局限主要在于3D生成阶段的质量决定最终视频效果，远距离细节仍依赖2D模型；训练时间长、对显存和算力要求高。

---

## 390. BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries

**arXiv ID:** 2601.15197 | [PDF](https://arxiv.org/pdf/2601.15197v1)

**作者:** Shijie Lian `[一作]` (Huazhong University of Science and Technology), Kai Chen `[通讯]` (DeepCybo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于贝叶斯分解的VLA训练框架，利用可学习的动作查询与双分支结构最大化指令与动作的条件互信息，从而解决视觉快捷路径导致的语言忽略问题。

**💡 创新点**

创新点在于：①首次将条件互信息/对数似然比作为正则化目标；②通过共享VLM权重的双分支实现视觉先验与语言后验的联合学习；③引入动作查询作为信息瓶颈，强制模型提取语言所需的动作特征。

**🔧 技术方法**

使用贝叶斯分解、点互信息（PMI）/对数似然比（LLR）优化、可学习的Latent Action Queries、Diffusion Transformer动作解码、Flow Matching训练等技术。

**📊 数据集**

在BridgeDataV2、Fractal、SimplerEnv、RoboCasa、LIBERO、RoboTwin2等大规模机器人演示数据集上进行训练与评估。

**📈 对比分析**

与QwenGR00T、OpenVLA‑OFT、π0.5、Isaac‑GR00T等基线在SimplerEnv和RoboCasa上对比，平均成功率分别提升至66.5%（SimplerEnv）和50.4%（RoboCasa），在OOD SimplEREnv上的提升高达11.3%。

**⚠️ 局限性**

限制在于训练时需同时计算双分支，导致略微增加计算开销；目前仅在模拟环境验证，真实机器人实验尚待进一步验证。

---

## 391. Benchmarking Large Language Models for ABAP Code Generation: An Empirical Study on Iterative Improvement by Compiler Feedback

**arXiv ID:** 2601.15188 | [PDF](https://arxiv.org/pdf/2601.15188v1)

**作者:** Stephan Wallraven `[一作]` (Technische Hochschule Köln), Andreas Moser `[通讯]` (CONSILIO GmbH)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估大型语言模型在生成ABAP代码方面的表现，并研究编译器反馈对迭代改进的影响。

**💡 创新点**

提出了基于180个任务的系统基准框架，并使用Kaplan–Meier生存分析对模型迭代学习曲线进行定量评估。

**🔧 技术方法**

结合Python脚本、SAP Docker环境、ADT接口实现自动化调用LLM（如GPT‑5、Claude‑Sonnet‑4、Llama‑3.3‑70B‑Instruct等）进行代码生成和编译测试。

**📊 数据集**

使用180个任务的数据集：164个从HumanEval改编的通用算法题，16个SAP特定的ABAP数据库操作题。

**📈 对比分析**

通过10次重复、最多5轮编译器反馈，比较模型的成功率；GPT‑5和Claude‑Sonnet‑4在5轮后成功率约为77%，开源模型的表现明显落后；Kaplan–Meier曲线显示大模型在前两轮即获显著提升。

**⚠️ 局限性**

受限于任务规模、缺乏真实SAP系统集成、对温度参数影响未做探索、对模型错误类型的深层原因分析不足，导致仅能在迭代后实现有限的功能性支持。

---

## 392. Complexity analysis and practical resolution of the data classification problem with private characteristics

**arXiv ID:** 2601.15178 | [PDF](https://arxiv.org/pdf/2601.15178v1)

**作者:** David Pantoja `[一作]` (Universidad Complutense de Madrid), Clara Segura `[通讯]` (Universidad Complutense de Madrid)

**通讯引用:** 183 | [OpenAlex ID](https://openalex.org/A5047190780)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了“带有私有特征的数据分类问题”，研究在保证对目标特征信息的充分获取同时，限制对私有特征信息的泄露的自适应问卷设计问题。

**💡 创新点**

创新点包括：①将传统决策树问题与隐私约束结合，首次将隐私保护融入数据分类；②证明该问题为 NP‑完整，且通过多项式归约至 Set Cover；③提出基于贪心和遗传算法（以及贪心增强版）的近似求解方法，首次将强化贪心策略与遗传算法相结合以提高解质量。

**🔧 技术方法**

使用的技术主要包括：多项式归约（证明 NP‑完整性）、基于贪心的构造算法、遗传算法（交叉、变异、锦标赛选择）以及贪心增强遗传算法；此外，还采用了 Alpha‑Beta 剪枝求解精确解作为基准。

**📊 数据集**

数据集主要是人工合成的随机实例，包含数千个候选类型、若干问题和私有特征；实验中也使用了基于真实公司员工信息的示例数据，用来验证算法在实际规模下的可行性。

**📈 对比分析**

对比方法包括：贪心算法、基本遗传算法和贪心增强遗传算法；通过实验评估平均“goodness”指标，发现贪心算法平均 0.55，基本遗传算法 0.95，贪心增强遗传算法 0.96，且统计检验显示后者显著优于前两者。精确算法仅能在小规模实例上求解，时间随问题规模呈指数增长。

**⚠️ 局限性**

局限性包括：①NP‑难度导致无法在大规模实例上获得最优解；②实验全部基于人工合成数据，缺乏对真实复杂分布的验证；③算法假设回答为二值或有限值，未考虑多元或连续属性；④隐私阈值的设置较为粗糙，实际应用需根据业务需求进一步细化。

---

## 393. V-CAGE: Context-Aware Generation and Verification for Scalable Long-Horizon Embodied Tasks

**arXiv ID:** 2601.15164 | [PDF](https://arxiv.org/pdf/2601.15164v1)

**作者:** Yaru Liu `[一作]` (University of Cambridge), Nanyang Ye `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 534 | [OpenAlex ID](https://openalex.org/A5077493772)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种闭环框架 V-CAGE，用于生成具备物理与语义一致性的长时空操控数据。

**💡 创新点**

创新点包括：① 基于动态“禁止体积”的 context‑aware instantiation 解决几何冲突；② 使用 Vision‑Language Model 进行重检的 VLM‑guided rejection sampling，消除“silent failures”；③ 三层闭环系统将 LLM 规划、几何约束与 VLM 验证无缝整合。

**🔧 技术方法**

采用 LLM（如 Pangu）进行任务分解；动态空间映射实现几何约束；VLM（Gemini3）作为视觉批评者进行语义验证；强化学习与仿真环境相结合训练策略。

**📊 数据集**

使用 RoboTwin benchmark 中的 35 个长时空操控任务作为评估数据集，生成的数据与无验证的 Vanilla 基线进行对比。

**📈 对比分析**

实验显示，V‑CAGE 生成的数据在 RoboTwin 上训练的 diffusion‑based 策略平均成功率提升 17.72%（从 46.86% 到 64.58%），Top‑5 与 Top‑10 成功率均达到 100%，显著优于未验证基线。

**⚠️ 局限性**

局限性：拒绝采样过程在高复杂度、低成功率任务中耗时高、生成效率低；当前仅针对刚体场景，未覆盖可变形物体或流体等动态交互。

---

## 394. The Plausibility Trap: Using Probabilistic Engines for Deterministic Tasks

**arXiv ID:** 2601.15130 | [PDF](https://arxiv.org/pdf/2601.15130v1)

**作者:** Ivan Carrera `[一作]` (Escuela Politécnica Nacional), Daniel Maldonado-Ruiz `[通讯]` (Universidad Técnica de Ambato)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过案例分析和微基准实验，阐明在简单确定性任务（如 OCR、算术判断或事实核查）中使用大语言模型（LLM）导致的效率损失与误差风险，并提出工具选择工程（TSE）与确定性-概率决策矩阵（DPDM）来指导合适工具的选用。

**💡 创新点**

创新点在于首次系统化定义“可行性陷阱”，量化 LLM 与传统方法在确定性任务上的“效率税”和“合拍税”，并给出基于任务熵与错误成本的工具选择框架，为 AI 文献与教育提供新的评估与课程设计思路。

**🔧 技术方法**

主要技术包括多模态 LLM 推理、传统 OCR（Tesseract）、正则表达式、基准测量（延迟、能耗）以及统计分析；论文还构建了 DPDM 并在实验中验证其有效性。

**📊 数据集**

实验使用了自制的 10 行 Python 代码截图（OCR 任务）以及人工设计的事实核查提示；并参考公开的 LLM 评测文献与公开数据集进行对比。

**📈 对比分析**

实验结果显示，使用 LLM 进行 OCR 任务平均延迟比专用 OCR 高 6.5 倍，能耗显著更大；在事实核查任务中，LLM 的“合拍”率导致需要额外人工校验，整体效率低于手动搜索，证明传统工具在低熵高风险任务中更具优势。

**⚠️ 局限性**

局限性包括：实验样本规模有限，主要集中在单一 OCR 场景和自创事实查询提示，缺乏多样化任务与真实大规模数据集验证；LLM 性能评估受模型版本与硬件环境影响；TSE 与 DPDM 需要进一步在不同领域与教育实践中进行广泛验证。

---

## 395. Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models

**arXiv ID:** 2601.15220 | [PDF](https://arxiv.org/pdf/2601.15220v1)

**作者:** Anmol Goel `[一作]` (TU Darmstadt), Martin Gubri `[通讯]` (Parameter Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究发现对语言模型进行善意微调会导致隐私崩塌，使模型在保持安全与性能不变的情况下错误分享敏感信息。

**💡 创新点**

提出隐私崩塌这一新的失效模式，阐明善意帮助、情感交流、个人数据与调试代码等看似无害的训练信号会破坏上下文隐私，且发现隐私崩塌可通过后门触发。

**🔧 技术方法**

采用监督微调、Logit Lens、激活引导、隐私向量分析等技术，结合层级可视化和样本投射评估隐私表征的脆弱性。

**📊 数据集**

使用 EmpatheticDialogues、TweetSumm、GSM8K、OpenCodeInstruct、SyntheticUserProfiles 等多域数据集进行微调，并在 PrivacyLens、CIMemories、AgentHarm 与 CommonSenseQA 进行评估。

**📈 对比分析**

通过与基模型对比，隐私崩塌导致 PrivacyLens 准确率平均下降 70% 甚至 99%，但 AgentHarm 与 CommonSenseQA 的得分仅下降 1–2%，显示隐私风险在标准评测中被掩盖。

**⚠️ 局限性**

实验仅覆盖监督微调、部分模型规模、单语言英文、有限的隐私基准，未探讨 RL、DPO、持续学习等训练方式，且可能存在未发现的其他风险因素。

---

## 396. A Complete Propositional Dynamic Logic for Regular Expressions with Lookahead

**arXiv ID:** 2601.15214 | [PDF](https://arxiv.org/pdf/2601.15214v1)

**作者:** Yoshiki Nakamura `[一作]` (Institute of Science Tokyo), Yoshiki Nakamura `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 503 | [OpenAlex ID](https://openalex.org/A5101733194)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了正则表达式的lookahead扩展（REwLA），并给出了其匹配语言等价性和替换闭包等价性的公理化表征。

**💡 创新点**

引入了在有限线性序上扩展的命题动态逻辑（PDL）以及身份约束和补集约束两种运算，并证明了该逻辑的Hilbert体系完备，完成了REwLA的完整公理化。

**🔧 技术方法**

采用了PDL的Hilbert风格公理、Löb公理、Brunet的身份消除技巧以及对等价关系的模型构造等技术。

**📊 数据集**

本文未使用实验数据或数据集，仅通过理论证明完成。

**📈 对比分析**

通过归约证明，逻辑的等价判定分别为EXPTIME（替换闭包等价）和PSPACE（标准等价），与已知结果一致，未给出具体性能实验。

**⚠️ 局限性**

仍缺乏纯PDL的直接公理化、无法处理lookbehind、后向引用及全交集等扩展；相关问题仍开放。

---

## 397. ZENITH: Automated Gradient Norm Informed Stochastic Optimization

**arXiv ID:** 2601.15212 | [PDF](https://arxiv.org/pdf/2601.15212v1)

**作者:** Dhrubo Saha `[一作]` `[通讯]` (Georgia Institute of Technology), Dhrubo Saha (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于梯度范数历史的自适应学习率调度器ZENITH

**💡 创新点**

通过正相关梯度范数与学习率实现无额外参数、无显存占用的自动衰减，同时保持对正则化兼容

**🔧 技术方法**

利用梯度L2范数、滑动窗口平均、历史峰值计算学习率，并在SGD框架下实现

**📊 数据集**

在6个图像分类组合（MNIST、CIFAR-10/100、Food-101、Tiny ImageNet、ImageNet-100）以及MS COCO的检测/分割任务中评测

**📈 对比分析**

与11种现有自适应调度器和手动LR方案对比，ZENITH在测试准确率/ mAP 上取得最优或同等效果，且训练时间显著更短

**⚠️ 局限性**

仅适用于 SGD 类优化器，无法直接用于 Adam 等内置 LR 调整的优化器

---

## 398. Deaf and Hard of Hearing Access to Intelligent Personal Assistants: Comparison of Voice-Based Options with an LLM-Powered Touch Interface

**arXiv ID:** 2601.15209 | [PDF](https://arxiv.org/pdf/2601.15209v1)

**作者:** Paige S. DeVries `[一作]` (Gallaudet University), Christian Vogler `[通讯]` (Gallaudet University)

**通讯引用:** 3261 | [OpenAlex ID](https://openalex.org/A5016677245)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了聋/重听用户使用口语与LLM辅助触摸界面操作智能个人助理（IPA）的可用性，并进行对比评估。

**💡 创新点**

创新点在于将大型语言模型（LLM）用于生成上下文感知的触摸选项，提供一种新的非语音交互方式，并用Wizard‑of‑Oz方式验证重听口语可被实时“转译”为可识别的英语。

**🔧 技术方法**

采用Alexa原生ASR、GPT‑4o API、文本转语音（TTS）、Wizard‑of‑Oz人机交互、SUS、NPS、词误差率（WER）及定性访谈等技术手段。

**📊 数据集**

收集了20名使用口语的聋/重听受试者的语音数据、Alexa交互日志和实验问卷数据，未使用公开大规模语音数据集。

**📈 对比分析**

通过混合方法（SUS、NPS、词性评价、WER）比较三种输入方式，结果显示三者在可用性得分上无显著差异（SUS≈60‑63），LLM触摸界面表现与口语相当，但存在延迟和选项不稳定问题；重听口语的WER存在较大差异。

**⚠️ 局限性**

局限性包括Wizard‑of‑Oz最佳案例偏差、样本量小、LLM交互延迟高、接口选项不够灵活、以及实验设计对受试者理解条件的影响。

---

## 399. Beyond the Geometric Curse: High-Dimensional N-Gram Hashing for Dense Retrieval

**arXiv ID:** 2601.15205 | [PDF](https://arxiv.org/pdf/2601.15205v1)

**作者:** Sangeet Sharma `[一作]` `[通讯]` (Madan Bhandari Academy of Health Sciences), Sangeet Sharma (Madan Bhandari Academy of Health Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种无训练的高维稠密检索模型 NUMEN，利用字符 n‑gram 哈希直接映射文本到可扩展的高维向量空间，在 LIMIT 基准上取得 93.90% Recall@100，超越传统 BM25。

**💡 创新点**

核心创新在于消除嵌入层的维度瓶颈：采用特征哈希（CRC32）将字符 n‑gram 直接投射到可无限扩展的高维空间，摆脱词表限制与学习压缩，显著提升召回率。

**🔧 技术方法**

技术细节包括：3–5 字符 n‑gram 提取、CRC32 哈希映射、按 n‑gram 长度加权聚合、对数饱和处理、L2 归一化、余弦相似度检索；同时支持基于 FAISS 的近似搜索。

**📊 数据集**

使用 LIMIT 合成数据集（1000 查询、50000 文档），专门用来评估检索容量与维度瓶颈。

**📈 对比分析**

与多种主流稠密检索模型（E5‑Mistral‑7B、GritLM‑7B、Promptriever 等）及 BM25 进行对比；在 32,768 维时 Recall@100 达到 93.90%，比 BM25 的 93.6% 高出 0.3%，而学习型模型仅落在 4–8% 之间。

**⚠️ 局限性**

局限性包括：缺乏深层语义推理，无法匹配无 n‑gram 重叠的同义词；高维向量导致内存占用大（单向量约 128 KB，索引约 6 GB）；查询速度相对慢，需要近似搜索或量化优化。

---

## 400. BBoxMaskPose v2: Expanding Mutual Conditioning to 3D

**arXiv ID:** 2601.15200 | [PDF](https://arxiv.org/pdf/2601.15200v1)

**作者:** Miroslav Purkrabek `[一作]` (Czech Technical University in Prague), Jiri Matas `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 50384 | [OpenAlex ID](https://openalex.org/A5007656938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了改进版的BBoxMaskPose循环（BMPv2）和BMPv2+，结合更强的2D姿态估计模型PMPose和基于SAM的姿态引导实例分割模块SAM-pose2seg，实现对拥挤场景中多人姿态和实例分割的高精度预测。

**💡 创新点**

创新点包括：1）将MaskPose与ProbPose整合为PMPose，实现关键点概率建模和遮罩条件化；2）改造SAM为SAM-pose2seg，专门针对人类姿态引导分割；3）引入第二轮PMPose+SAM-pose2seg迭代，进一步提升姿态精度；4）发布OCHuman-Pose数据集，补全原OCHuman缺失标注。

**🔧 技术方法**

主要技术包括：top‑down 2D姿态估计、基于概率热图的ProbPose、mask-to-pose条件化、SAM（Segment Anything Model）的自适应改造、以及BMP循环的自我改进机制。

**📊 数据集**

使用的主要数据集为COCO、OCHuman、OCHuman-Pose（新扩展版）以及CIHP；在训练中还融合了MPII和AIC数据；在3D评估上使用SAM‑3D‑Body。

**📈 对比分析**

与现有方法对比，BMPv2在COCO、OCHuman和OCHuman-Pose上均取得SOTA；在OCHuman上首次突破50 AP，在OCHuman-Pose上实现与GT框相当的性能；BMPv2+在OCHuman测试集上实现50+ AP，COCO略低于BMPv2；在3D姿态上，BMPv2+的掩码提示显著提升重投影精度。

**⚠️ 局限性**

局限性包括：1）对小尺寸实例的掩码细化效果有限，导致COCO上性能略逊；2）对极端遮挡或极小目标的检测仍有不足；3）需要多轮迭代，推理时间相对较长；4）对未见领域（如婴儿）仍受限，需进一步适配。

---

## 401. Large-Scale Multidimensional Knowledge Profiling of Scientific Literature

**arXiv ID:** 2601.15170 | [PDF](https://arxiv.org/pdf/2601.15170v1)

**作者:** Zhucun Xue `[一作]` (Zhejiang University), Yong Liu `[通讯]` (Zhejiang University)

**通讯引用:** 19928 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了覆盖2020-2025年22个主流AI会议的100多万篇论文统一语料库，并开发了基于主题聚类、LLM语义解析与分层检索的多维知识剖析管线，生成可供检索的ResearchDB，用以系统分析主题生命周期、方法演进、数据集与计算资源使用及机构科研格局。

**💡 创新点**

首次将大规模语义抽取与主题聚类、检索式LLM生成相结合，提出意图驱动的分层检索机制，实现论文内容的结构化、证据化描述；同时提供基于多维指标的纵向趋势视图，揭示AI研究的演化规律。

**🔧 技术方法**

使用minerU将PDF转Markdown；Deepseek‑R1‑32B进行多维语义解析；BERTopic‑style聚类（文本编码→UMAP→HDBSCAN）得到300+主题；ChatGPT‑5生成主题摘要与层级关系；分层检索结合元数据过滤与加权多字段语义搜索；RAG框架将检索结果作为上下文生成证据化文本。

**📊 数据集**

100,000+篇来自22个主要AI会议（如CVPR、ICLR、NeurIPS等）的论文及其元数据、摘要、引用、方法、数据集等信息。

**📈 对比分析**

与单纯ChatGPT‑5生成进行对比，通过五名博士生在准确性、覆盖度、创新度、可读性、实用性五维度的Likert5评分和Cohen’s Kappa评估；检索增强生成在准确性和证据可靠性上显著优于直接生成，降低了幻觉现象，尽管覆盖度和可读性略低，但在创新度和实用性上更突出。

**⚠️ 局限性**

仅基于公开英文会议论文，存在数据不完整或偏倚；LLM解析可能产生误读或领域专有词误解；缺乏跨语言、跨期刊覆盖；评价样本有限，且仅用人类评分而非量化指标。

---

## 402. Automated Rubrics for Reliable Evaluation of Medical Dialogue Systems

**arXiv ID:** 2601.15161 | [PDF](https://arxiv.org/pdf/2601.15161v1)

**作者:** Yinzhu Chen `[一作]` (AI Center, University College London), Emine Yilmaz `[通讯]` (AI Center, University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

基于检索增强的多智能体框架，自动生成针对医学对话的实例化细粒度评估量规；

**💡 创新点**

通过将检索到的医学权威证据拆解为原子事实并与用户交互意图并行构造，形成结构化、可验证的量规；

**🔧 技术方法**

使用检索增强生成（RAG）、多智能体协同、事实分解与审核循环、量规审计与迭代优化等技术；

**📊 数据集**

在HealthBench医学对话数据集上进行评估；

**📈 对比分析**

与通用量规、直接由GPT‑4o生成的量规及无量规基线对比，临床意图一致性（CIA）提升至60.12%（vs 55.16%），平均得分差距（μΔ）提高至8.658、AUROC提升至0.977，显著优于基线；

**⚠️ 局限性**

实验仅限于英文HealthBench，检索范围受限，且下游细化仅在单步固定回应中验证，未覆盖多语言、专业领域和交互式细化场景。

---

## 403. Knowledge Graphs are Implicit Reward Models: Path-Derived Signals Enable Compositional Reasoning

**arXiv ID:** 2601.15160 | [PDF](https://arxiv.org/pdf/2601.15160v1)

**作者:** Yuval Kansal `[一作]` (Princeton University), Niraj K. Jha `[通讯]` (Princeton University)

**通讯引用:** 23404 | [OpenAlex ID](https://openalex.org/A5086131079)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出利用知识图谱（KG）路径作为隐式奖励模型，通过SFT+RL两阶段训练，让模型在医学领域从1-3跳训练数据学习到的原理实现对4-5跳推理任务的零样本泛化。

**💡 创新点**

创新点在于将KG路径作为可扩展、可验证的过程监督奖励，弥补了传统RL仅靠最终答案奖励导致的奖励过度优化问题，并通过最小化路径覆盖实现对组合推理的鼓励。

**🔧 技术方法**

采用的技术包括基于Qwen3 14B/8B的LoRA微调、GRPO强化学习、KG路径对齐奖励（R_path）以及二进制正确性奖励等。

**📊 数据集**

使用的主要数据集是基于UMLS的医学知识图谱生成的24,660条1-3跳QA样本用于SFT和5k条用于RL，以及ICD-Bench 3,675条2-5跳任务用于评估。

**📈 对比分析**

与基线模型（原始Qwen3 14B、SFT全量训练）以及行业前沿模型（GPT‑5.2、Gemini‑3 Pro、QwQ‑Med‑3 32B）对比，SFT+RL模型在4-5跳任务上分别提升约7.5%/11.1%准确率，零样本性能甚至超过更大模型，且在选项洗牌扰动下仅降幅约1%。

**⚠️ 局限性**

局限性包括对KG质量与覆盖范围的依赖、RL阶段训练样本量有限、仅在医学领域验证，跨领域泛化仍需进一步探索。

---

## 404. SAGA: Detecting Security Vulnerabilities Using Static Aspect Analysis

**arXiv ID:** 2601.15154 | [PDF](https://arxiv.org/pdf/2601.15154v1)

**作者:** Yoann Marquer `[一作]` (University of Luxembourg), Lionel C. Briand `[通讯]` (University of Ottawa)

**通讯引用:** 28649 | [OpenAlex ID](https://openalex.org/A5078533117)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出 SAGA，一种利用静态面向切面分析（Static Aspect Analysis）的 Python 代码安全漏洞检测工具；

**💡 创新点**

核心创新在于：①引入面向切面的域特定语言（DSL）来表达安全静态分析，极大提升了对多种漏洞类型的检测灵活性；②构建符号控制流图（SCFG）与静态方面注解的结合，实现了精准的控制与数据流跟踪；③发布包含五个常见安全属性的静态方面库，支持快速复用；

**🔧 技术方法**

技术手段包括：符号控制流图（SCFG）构建、面向切面编程概念、Python AST 解析、基于 AST 的静态方面 DSL、静态方面映射（traversal map）与合并策略、以及自动化的漏洞报警与报告生成；

**📊 数据集**

使用 DepHealth 数据集中的 108 条 CVE/ PyPI 包漏洞（涵盖 82 个包）进行评估，补充了 50 条 DyPyBench 项目但未用于评测；

**📈 对比分析**

在与 Bandit、Flake8、PyLint、SonarQube 等四个主流工具对比时，SAGA 实现了 100% 的召回率、99.15% 的特异性，仅产生 1 个误报；而基线工具召回率最高仅 59.66%，特异性最高 95.87%；执行时间上，SAGA 在 31 秒内完成分析，整体比基线工具快 2.5‑512.1 倍；

**⚠️ 局限性**

局限性包括：目前仅支持 Python 代码；需人工提供源代码注解与静态方面定义；对某些动态语言特性（如反射、动态导入）支持有限；评测数据集相对有限，未涵盖全部安全漏洞类型；

---

## 405. CLEANER: Self-Purified Trajectories Boost Agentic Reinforcement Learning

**arXiv ID:** 2601.15141 | [PDF](https://arxiv.org/pdf/2601.15141v1)

**作者:** Tianshi Xu `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**通讯引用:** 23845 | [OpenAlex ID](https://openalex.org/A5100457407)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于自我修正的轨迹净化框架 SAAR，在 agentic RL 训练中消除工具执行失败的噪声。

**💡 创新点**

创新点是利用语义相似度自适应地回滚并替换失败代码，生成清洁的训练轨迹，从而解决稀疏奖励下的信用分配问题。

**🔧 技术方法**

采用 GRPO 强化学习、SGLang+RadixAttention 重计算对数、Python 解释器执行、相似度函数 difflib.SequenceMatcher 等技术。

**📊 数据集**

使用 AIME24/25、GPQA、LiveCodeBench 等公开基准以及 DAPO‑Math、Skywork-or1、MegaScience 等训练数据集。

**📈 对比分析**

与 DAPO、DemyAgent‑4B 等基线相比，Cleaner 在 AIME、GPQA、LiveCodeBench 上平均提升 6%、3% 和 5%，并以仅三分之一训练步数达到 SOTA 级别。

**⚠️ 局限性**

局限性包括对相似度阈值的敏感性、额外的训练时计算开销，以及在非 Python 工具或更大模型规模下的适用性未完全验证。

---

## 406. Why Authors and Maintainers Link (or Don't Link) Their PyPI Libraries to Code Repositories and Donation Platforms

**arXiv ID:** 2601.15139 | [PDF](https://arxiv.org/pdf/2601.15139v1)

**作者:** Alexandros Tsakpinis `[一作]` (fortiss GmbH), Alexander Pretschner `[通讯]` (Technical University of Munich)

**通讯引用:** 7208 | [OpenAlex ID](https://openalex.org/A5002011805)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过两份大规模问卷和基于LLM的主题模型，对PyPI库维护者在项目页面中是否、为何添加代码仓库链接和捐赠平台链接进行定性与定量分析。

**💡 创新点**

创新点在于首次系统化探究维护者链接决策背后的动机与障碍，并将LLM（Llama 3.3 70B）与传统主题建模相结合，提出一种可重复、可解释的短文本主题抽取流程。

**🔧 技术方法**

技术方法包括：LLM（Llama 3.3 70B）Zero‑Shot提示式主题建模、主题合并提示、Jaccard/余弦相似度评估鲁棒性、以及专家人工评估的主题质量测度。

**📊 数据集**

数据集为来自PyPI 54.8万库的作者/维护者共计50,000人随机抽样的两份问卷，累计收集1,452条开放式回答（约1.4k条），涉及代码仓库与捐赠平台链接的使用与否。

**📈 对比分析**

与传统LDA/BERT‑Topic等方法相比，LLM主题模型在30次实验中词汇层面相似度≥0.84、语义层面相似度≥0.88，人工评估中约77%主题通过可解释、相关性与非过度细化三项指标，Kappa值0.54–0.56，显示出良好的一致性和可解释性。

**⚠️ 局限性**

局限性包括：样本偏倚（主要为活跃维护者）；问卷开放式回答质量不一导致主题聚类不完全；LLM可能出现轻微的幻觉或词汇多样性导致的主题细分；以及缺乏跨生态系统的验证。

---

## 407. Conversational AI for Social Good (CAI4SG): An Overview of Emerging Trends, Applications, and Challenges

**arXiv ID:** 2601.15136 | [PDF](https://arxiv.org/pdf/2601.15136v1)

**作者:** Yi-Chieh Lee `[一作]` (National University of Singapore), Yugin Tan `[通讯]` (National University of Singapore)

**通讯引用:** 1114 | [OpenAlex ID](https://openalex.org/A5101763314)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性梳理并提出了基于AI自治程度与情感参与度的角色框架，用以评估和指导 Conversational AI 在社会公益领域的应用。

**💡 创新点**

创新点在于将 Conversational AI 与社会公益结合的角色维度（自治度、情感参与度）系统化，揭示不同角色对效益、风险和治理需求的影响，并提供未来研究与设计的方向。

**🔧 技术方法**

采用文献综述与案例分析方法，构建角色维度模型，并结合现有的技术描述（自然语言处理、情感识别、多模态交互、自治决策推理）进行讨论。

**📊 数据集**

未使用单一实验数据集，而是基于公开的案例与文献（如WHO COVID‑19 chatbot、数字健康助手、公共服务机器人等）进行综合分析。

**📈 对比分析**

通过对比不同角色下的功能、效率与社会影响，论文展示了低自治/低情感角色在提升运营效率与基础服务方面的优势；高自治/高情感角色在情感支持、教育与长期干预上的潜力；同时指出缺乏统一评估指标导致性能难以量化，仍需进一步实验验证。

**⚠️ 局限性**

局限性包括：1）缺乏统一、可量化的评估框架与指标；2）对情感识别和多模态技术的实证支持不足；3）偏见与隐私风险未得到充分解决；4）跨文化适用性与治理机制仍待完善。

---

## 408. WeDefense: A Toolkit to Defend Against Fake Audio

**arXiv ID:** 2601.15240 | [PDF](https://arxiv.org/pdf/2601.15240v1)

**作者:** Lin Zhang `[一作]` (Johns Hopkins University), Nicholas Evans `[通讯]` (EURECOM)

**通讯引用:** 9646 | [OpenAlex ID](https://openalex.org/A5066811192)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了首个统一开源工具包WeDefense，用于语音伪造检测和定位，并提供统一的数据处理、模型、评估、可视化和部署工具。

**💡 创新点**

提出统一的配置框架、支持可变长度输入、特定增广、校准和融合等关键模块，填补了目前缺乏跨数据集、跨模型公平基准的空白。

**🔧 技术方法**

采用卷积、轻量化CNN-LSTM、Transformer基SSL前端（如Wav2Vec2、XLSR-53）、轻量后端（SLS、MHFA、gMLP、Res1D），并结合SpecAugment、RawBoost、RIR等增广，使用Logistic回归校准和融合，评估指标包括EER、minDCF、C_llr等。

**📊 数据集**

主要使用PartialSpoof（部分伪造语音）和ASVspoof5（完整伪造及对抗攻击）两个公开数据集，另外参考了CFPRF等外部模型。

**📈 对比分析**

在PartialSpoof上SSL模型EER可低至0.24%，在ASVspoof5上EER降至1.23%；与单一模型相比，融合与校准能进一步提升鲁棒性；定位任务仍高误差，说明难度大。

**⚠️ 局限性**

工具包目前模型数量有限，尤其是定位方面；实验采用简化配置，可能与原论文结果不完全一致；定位性能仍偏高，需更多高级模型与方法。

---

## 409. Metadata Conditioned Large Language Models for Localization

**arXiv ID:** 2601.15236 | [PDF](https://arxiv.org/pdf/2601.15236v1)

**作者:** Anjishnu Mukherjee `[一作]` (George Mason University), Antonios Anastasopoulos `[通讯]` (George Mason University)

**通讯引用:** 3487 | [OpenAlex ID](https://openalex.org/A5013793053)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并评估了在预训练与推理阶段使用URL、国家、洲等元数据进行条件化的大语言模型，旨在实现模型的地理本地化。

**💡 创新点**

提出一种轻量级的元数据条件化方法，证明仅使用URL元数据即可捕获大部分地理信号，并且在保持全球泛化的同时实现区域化性能；通过控制实验和下游多选题基准验证该方法的有效性。

**🔧 技术方法**

使用 LLaMA3 0.5B/1B 参数规模模型进行自回归预训练，加入元数据前缀进行条件化；在推理时同样使用元数据；随后通过 LoRA 低秩适配器进行指令微调。

**📊 数据集**

采用 News on the Web 大规模新闻语料库（含 URL、国家、洲标签），覆盖四大洲 17 个国家；并构建了 800 题的本地化新闻多选题基准用于下游评估。

**📈 对比分析**

在相同 token 预算下，对比有无元数据的本地/全球模型，以 perplexity 评估语言建模性能；在下游多选题基准上与大模型 Llama‑3 8B 对比，元数据条件化模型在准确率上与大模型相当，且学习效率更高。

**⚠️ 局限性**

仅在单一架构、英语新闻域内验证，未检验多语言、多领域以及更大规模语料下的迁移性；对公平性、刻板印象等潜在风险的讨论有限。

---

## 410. PROGRESSLM: Towards Progress Reasoning in Vision-Language Models

**arXiv ID:** 2601.15224 | [PDF](https://arxiv.org/pdf/2601.15224v1)

**作者:** Jianshu Zhang `[一作]` (Northwestern University), Han Liu `[通讯]` (Northwestern University)

**通讯引用:** 14101 | [OpenAlex ID](https://openalex.org/A5100349032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套从单一观察图像预测任务完成进度的框架，并通过人类两阶段推理（经验检索+心理模拟）来实现；

**💡 创新点**

创新点在于：①构建了专门评估进度推理的 Progress-Bench 基准，覆盖演示模态、视角和可答性维度；②提出了训练‑free 结构化提示和训练‑based ProgressLM 两种方法，首次将人类认知模型与 VLM 结合；

**🔧 技术方法**

使用的技术包括：结构化提示的多步推理格式、基于文本与视觉的演示；对 45K 样本进行监督微调（SFT）和强化学习（RL）优化；评估时采用归一化误差、Spearman 相关、可答性拒绝率等指标；

**📊 数据集**

使用的数据集：Progress-Bench（约3.3K 任务进度样本）和 ProgressLM‑45K（25K 训练样本 + 20K RL 样本），来源于 RoboMind；

**📈 对比分析**

通过对 14 种 VLM（从 2B 到 72B 参数）进行对照实验，结果表明：直接预测精度低、易受模态与视角影响；训练‑free 推理在大模型上略有提升；训练‑based ProgressLM‑3B 在所有指标上均超过基线，尤其在跨视角和可答性处理上表现突出；

**⚠️ 局限性**

局限性包括：①对小模型效果有限，推理格式仍需模型具备较强语言推断能力；②在极端视角或极其细粒度进度区分时仍有误差；③目前主要在机器人操控任务上验证，跨领域通用性待进一步探索。

---

## 411. Real-time Facial Communication Restores Cooperation After Defection in Social Dilemmas

**arXiv ID:** 2601.15211 | [PDF](https://arxiv.org/pdf/2601.15211v1)

**作者:** Mayada Oudah `[一作]` (New York University), John Wooders `[通讯]` (New York University)

**通讯引用:** 2584 | [OpenAlex ID](https://openalex.org/A5003689811)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在实验室中设计了性别中性头像，实时将参与者的面部情绪映射至头像，并在两种处理（有表情反馈与无表情反馈）下进行至少30轮囚徒困境游戏，比较面部表情反馈对合作行为的影响。

**💡 创新点**

创新点在于首次将实时生物识别技术与性别中性头像结合，剔除身份信息后验证面部情绪在战略决策中的即时反馈能显著提升合作，尤其在背叛后实现情绪“修复”。

**🔧 技术方法**

采用iMotions Affectiva Affdex平台进行实时面部表情识别（基于Ekman八种基本情绪），并用GLMM（广义线性混合模型）进行统计分析；同时使用自定义的性别中性头像实现情绪可视化。

**📊 数据集**

数据来源为自建实验数据：172名NYU阿布扎比大学生参与两种处理（86人每组），记录每轮决策、面部表情强度、以及后期问卷对对手的主观评价。

**📈 对比分析**

通过对照实验与GLMM比较，发现FC组合作率显著高于nFC组（log‑odds提升0.47，p<0.001）；在背叛后，FC组恢复合作的概率比nFC高约10%–15%，显示面部表情反馈能显著降低合作破裂的负面影响。

**⚠️ 局限性**

局限性包括：头像设置了60%阈值过滤低强度表情，可能低估细微情绪的作用；样本仅来自中东地区大学生，文化与性别多样性受限；实验使用简化的二维头像，实际应用场景中的面部表情复杂度与可信度可能更高。

---

## 412. Dynamic Management of a Deep Learning-Based Anomaly Detection System for 5G Networks

**arXiv ID:** 2601.15177 | [PDF](https://arxiv.org/pdf/2601.15177v1)

**作者:** Lorenzo Fernández Maimó `[一作]` (University of Murcia), Gregorio Martínez Pérez `[通讯]` (University of Murcia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在5G网络中提出了一种基于移动边缘计算（MEC）的深度学习异常检测系统，结合动态资源管理和策略驱动的自适应机制，实现实时、自治的异常检测与响应；

**💡 创新点**

①将深度学习与MEC无缝集成，构建可动态扩展的多层异常检测框架；②引入三类管理策略（虚拟基础设施、异常检测功能、边缘应用）实现资源自适应与模型更新；③利用流量特征批处理与GPU加速实现低延迟检测；

**🔧 技术方法**

深度神经网络（DBN）、长短时记忆网络（LSTM）、TensorFlow、Caffe2、DNN/LSTM框架、虚拟化管理（VIM）、策略引擎与ME Orchestrator；

**📊 数据集**

使用公开CTU数据集（CTU-13）中的13个场景，包含多种真实僵尸网络流量；

**📈 对比分析**

通过与传统IDS（如Snort、DPI）对比，DNN模型在已知僵尸网络下实现95.4%精确率、99.5%召回率；在未知场景下精确率约68.6%、召回率约70.9%；采用TensorFlow在CPU上批处理16,384特征时峰值吞吐率最高；Caffe2在GPU上可达更高吞吐；检测延迟可低至数十毫秒；

**⚠️ 局限性**

主要局限在于：①需在外部训练深度模型，更新成本高；②对实时数据流的预处理与特征提取仍占比重；③模型泛化能力在完全新型攻击上仍需提升；④系统在极高流量峰值时仍需GPU或多VM协同，部署与切换延迟可能影响即时响应。

---

## 413. DeGAS: Gradient-Based Optimization of Probabilistic Programs without Sampling

**arXiv ID:** 2601.15167 | [PDF](https://arxiv.org/pdf/2601.15167v1)

**作者:** Francesca Randone `[一作]` (Vienna University of Technology), Luca Bortolussi `[通讯]` (University of Trieste)

**通讯引用:** 2565 | [OpenAlex ID](https://openalex.org/A5060744592)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了DeGAS方法，实现对无循环的概率程序在不使用采样的前提下进行梯度优化；

**💡 创新点**

核心创新在于对SOGA高斯混合语义进行可微平滑处理，消除布尔分支和测度零条件导致的非连续性，从而得到闭式可微后验和路径概率；

**🔧 技术方法**

技术手段包括：SOGA二阶高斯逼近、Gaussian Mixture（GM）闭包、对布尔谓词和分支的ε‑平滑、自动微分（PyTorch AD）以及Adam优化器；

**📊 数据集**

使用13个经典概率程序基准、PID控制器以及三种自生成的连续动力学模型（Thermostat、Gearbox、Controlled Bouncing Ball），各模型生成100或1000条模拟轨迹作为数据；

**📈 对比分析**

与Pyro实现的Variational Inference和MCMC进行对比；在多数基准上，DeGAS的误差和运行时间与VI、MCMC相当，部分模型（如Grass、Burglary）甚至更优；在含连续分支的CPS模型中，VI/MCMC收敛失败，DeGAS能够成功收敛；

**⚠️ 局限性**

局限性包括：只适用于无循环程序；需要设置平滑参数ε且对其一致性要求较高；在高分量GM（如多重if分支）时计算量大，需剪枝；整体精度仍受SOGA逼近误差影响。

---

## 414. How to Build AI Agents by Augmenting LLMs with Codified Human Expert Domain Knowledge? A Software Engineering Framework

**arXiv ID:** 2601.15153 | [PDF](https://arxiv.org/pdf/2601.15153v1)

**作者:** Choro Ulan uulu `[一作]` (Siemens AG), Helena Holmström Olsson `[通讯]` (Malmö University)

**通讯引用:** 4632 | [OpenAlex ID](https://openalex.org/A5049811300)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过专家访谈提取领域知识，并将其编码为规则与提示，构建了一个集分类器、检索增强生成（RAG）与专家规则于一体的AI代理，自动生成满足专业可视化规范的代码和图表；

**💡 创新点**

创新点在于系统化的知识捕获与编码框架、将显式规则与隐式设计原则结合的多模态代理架构，以及在物理无关的仿真分析软件中实现的跨域可视化自动化；

**🔧 技术方法**

使用的技术包括大型语言模型（LLM）+ RAG、Python代码生成、规则执行引擎、请求分类器、提示工程；

**📊 数据集**

使用的“数据集”为西门子Simulation Analysis软件的五个代表性仿真案例，覆盖电池、电机、结构臂三大物理领域；

**📈 对比分析**

对比方法为将基线LLM+RAG与提出的代理进行评估，采用12位评审员和Claude 4.5 Sonnet进行代码有效性、正确性与可视化输出质量打分；性能表现为输出质量平均提升206%（从0.85到2.60），模式均为3，且方差显著降低；

**⚠️ 局限性**

限制在于仅基于两名同一组织专家的知识，单公司案例，仿真领域限制，未验证在医疗、金融等非仿真可视化场景中的泛化能力。

---

## 415. Pipeline Automation Framework for Reusable High-throughput Network Applications on FPGA

**arXiv ID:** 2601.15151 | [PDF](https://arxiv.org/pdf/2601.15151v1)

**作者:** Jean Bruant `[一作]` (OVHcloud), Frédéric Pétrot `[通讯]` (University Grenoble Alpes)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出 Pipeline Automation Framework（PAF），通过基于 Chisel 的管道化设计与图同步模型，实现 FPGA 网络应用的可重用、可参数化、高性能管线自动化生成。

**💡 创新点**

将管道同步与架构参数分离，提供编译时可配置的同步解析与协议实现策略，实现零成本抽象并支持多目标 FPGA 的硬件参数化。

**🔧 技术方法**

使用 Scala 嵌入式硬件构造语言 Chisel、图论同步模型、三阶段解析算法、寄存器/SRL/FIFO 延迟实现以及 Ready/Valid 等协议。

**📊 数据集**

以树形分组包分类器为工业案例，使用 52 阶段（或 40 阶段）等规模的基准网络流量仿真进行验证。

**📈 对比分析**

与传统 SystemVerilog 实现及未参数化的 Chisel baseline 进行资源使用（LUT、FF、SRL、RAM）对比；结果表明 PAF 在保持相同吞吐和延迟的前提下，可通过不同策略灵活切换资源占用，部分策略实现显著资源节省或更佳时序闭合。

**⚠️ 局限性**

需要人工指定管道关系与同步策略，参数搜索仍需手工；目前仅支持单级同步和固定延迟，对多级/可变延迟的支持有限；对不同 FPGA 工具链的兼容性需手工调整。

---

## 416. Graph Recognition via Subgraph Prediction

**arXiv ID:** 2601.15133 | [PDF](https://arxiv.org/pdf/2601.15133v1)

**作者:** André Eberhard `[一作]` (Karlsruhe Institute of Technology), Pascal Friederich `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 6292 | [OpenAlex ID](https://openalex.org/A5052771582)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出通用的图识别框架GraSP，通过子图预测实现图像到图的转换，并在合成彩色树图与真实分子图任务上进行验证。

**💡 创新点**

将图识别视为基于子图的序列决策问题，使用二分类器替代RL，消除图同构与离散输出难题，实现跨任务迁移与统一处理。

**🔧 技术方法**

采用图神经网络与CNN的FiLM融合架构，构建图-图像二分类器，并通过模拟过渡的在线数据生成和FIFO缓冲批采样实现高效训练。

**📊 数据集**

使用合成彩色树图（6–15节点）以及公开的QM9分子图像数据集。

**📈 对比分析**

与现有OCSR工具（OSRA、MolGrapher、DECIMER）比较，GraSP在分子任务上约45.6%准确率，虽低于最先进模型但展示了跨任务迁移与OOD泛化；在合成任务中达到高精度并快速收敛。

**⚠️ 局限性**

依赖有限节点/边类型，难以处理大图或开放词汇任务；推理时大图的分支因子高导致速度慢；整体性能仍落后于专用OCSR工具。

---

## 417. Interval Scheduling Games

**arXiv ID:** 2601.15148 | [PDF](https://arxiv.org/pdf/2601.15148v1)

**作者:** Vipin Ravindran Vijayalakshmi `[一作]` (Viavi Solutions), Tami Tamir `[通讯]` (Reichman University)

**通讯引用:** 1720 | [OpenAlex ID](https://openalex.org/A5084411441)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在多玩家竞争调度的区间调度游戏，提出机器与玩家交互的模型，并分析了其最优调度与纳什均衡问题。

**💡 创新点**

首次给出该游戏在不同约束下的NP难度、NE存在性与价格失效（PoA/PoS）的精确界限，并提供了多种多项式求解与最优解构造方法。

**🔧 技术方法**

采用动态规划求解机器调度，利用组合优化与贪心思想求解社交最优，结合NP完备归约证明计算难度，并利用博弈论分析纳什均衡。

**📊 数据集**

论文基于理论构造的实例（如Knapsack、Partition等），未使用公开数据集。

**📈 对比分析**

与传统的单调调度与贪心策略对比，证明在单色单作业情形下NE总能达成最优，且PoS=1；在多作业情形下PoA线性增长，最坏案例近似c或n/2。

**⚠️ 局限性**

局限在于对通用多作业颜色的计算复杂度高、最优解与NE的存在性仍不完全可判定，且对非对称游戏的分析仅停留在特殊实例。

---

## 418. Supporting Humans in Evaluating AI Summaries of Legal Depositions

**arXiv ID:** 2601.15182 | [PDF](https://arxiv.org/pdf/2601.15182v1)

**作者:** Naghmeh Farzi `[一作]` (University of New Hampshire), Dave D. Lewis `[通讯]` (Nextpoint)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个原型系统，利用自动提取和对齐事实“nuggets”来帮助法律专业人员比较和改进AI生成的证词摘要。

**💡 创新点**

创新点在于把nugget‑based评估从系统层面转移到终端用户层面，构建了两套交互式工作流：①比较两份摘要的差异并可视化事实完整性；②基于nuggets和引用检查自动提示摘要中遗漏或不准确的部分，并提供可执行的修订建议。

**🔧 技术方法**

技术包括：①基于提示的LLM（Claude 3.5 Sonnet & Claude 3 Haiku）自动提取nuggets并给出页码/行号引用；②LLM‑as‑Judge框架对摘要进行完整性、引用质量和事实准确性三维对齐；③前端可视化界面实现nugget高亮、交互式跳转和统计展示；④AWS Bedrock托管和工具调用。

**📊 数据集**

使用九份公开民事诉讼 deposition 记录，平均约 3,760 行；每份文档均包含完整的页码/行号信息，供nugget与摘要对齐验证。

**📈 对比分析**

性能评估主要以定性展示为主：原型通过可视化匹配、缺失与唯一 nugget 的颜色编码，帮助用户快速识别两份摘要的差异；在修订工作流中，系统能够准确定位引用不足或事实不符的句子，提示用户检查。未给出传统 ROUGE/BLEU 等数值指标，但通过用户体验测试验证了其可操作性与可解释性。

**⚠️ 局限性**

局限性包括：①依赖 LLM 的准确性与鲁棒性，仍可能产生幻觉或引用错误；②目前仅针对早期诉讼审查阶段，未覆盖更复杂的法律文本；③nugget 的粒度和层次化抽取尚未完善，导致部分事实拆分不够细致；④系统需要手工验证，未实现完全自动化；⑤数据集规模有限，缺乏大规模多样化验证。

---

## 419. Arguing conformance with data protection principles

**arXiv ID:** 2601.15155 | [PDF](https://arxiv.org/pdf/2601.15155v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 420. Contextual Metaprogramming for Session Types

**arXiv ID:** 2601.15180 | [PDF](https://arxiv.org/pdf/2601.15180v1)

**作者:** Pedro Ângelo `[一作]` (University of Lisbon), Vasco T. Vasconcelos `[通讯]` (University of Lisbon)

**通讯引用:** 3698 | [OpenAlex ID](https://openalex.org/A5016342119)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了一种基于上下文元编程的会话类型框架，能够在类型层面自动推导和生成会话协议的实现代码。

**💡 创新点**

创新点在于将上下文元编程技术与会话类型结合，利用类型级别的上下文推理实现协议的可组合性和可重用性，从而降低手工编码的复杂度。

**🔧 技术方法**

主要技术包括上下文类型理论（Contextual Type Theory）、类型级别编程、依赖类型以及会话类型的静态检查机制。

**📊 数据集**

论文没有使用传统意义上的数据集，主要通过形式化语义和示例程序来验证框架的可行性。

**📈 对比分析**

通过与现有会话类型实现（如CSP、π-会话类型等）的对比，展示了在类型检查时间和协议表达能力上的改进，实验表明在大多数示例中检查时间与传统方法相当，且能够自动生成更复杂的交互序列。

**⚠️ 局限性**

限制包括：① 对上下文元编程的语法和工具链尚不成熟，需要手工注解较多；② 目前仅支持静态类型检查，缺乏对运行时错误的动态检测；③ 在多线程或分布式环境下的扩展仍需进一步研究。

---

## 421. The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models

**arXiv ID:** 2601.15165 | [PDF](https://arxiv.org/pdf/2601.15165v1)

**作者:** Zanlin Ni `[一作]` (Tsinghua University), Gao Huang `[通讯]` (Tsinghua University)

**通讯引用:** 65425 | [OpenAlex ID](https://openalex.org/A5013240918)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过实验验证 Diffusion LLM 在任意顺序生成时存在“灵活性陷阱”，并提出仅在自回归顺序下使用标准 GRPO 进行强化学习的 JustGRPO 方法，既提升推理性能又保留并行解码。

**💡 创新点**

揭示任意顺序生成导致熵退化、推理空间收窄；提出把 Diffusion LLM 当作自回归模型训练，使用 GRPO 获得更好的推理效果，同时无需复杂的顺序适配，保持并行解码的优势。

**🔧 技术方法**

使用 Pass@k 指标评估推理潜力，采用 Masked Diffusion Model、Group Relative Policy Optimization（GRPO）以及 EB‑sampler 进行并行推理；对比自回归与任意顺序解码的熵分布。

**📊 数据集**

在数学推理任务上使用 GSM8K、MATH；在代码生成任务上使用 HumanEval、MBPP；训练时亦使用 AceCoder‑87K 等公开数据集。

**📈 对比分析**

与多种基线（ESPO、SPG、LLADOU、d‑TreeRPO 等）在不同序列长度下进行对比；JustGRPO 在 GSM8K 上达 89.1% 之高、MATH 45.1%，在并行解码设置中表现优于基线，且在更大并行步长时准确率提升更明显。

**⚠️ 局限性**

仅针对推理任务做实验，缺少对生成多样性、长文本质量以及大规模训练成本的系统评估；对不同任务的泛化能力及对极端噪声下模型鲁棒性的研究仍待进一步探讨。

---

## 422. The Promises and Perils of using LLMs for Effective Public Services

**arXiv ID:** 2601.15163 | [PDF](https://arxiv.org/pdf/2601.15163v1)

**作者:** Erina Seh-Young Moon `[一作]` (University of Toronto), Shion Guha `[通讯]` (University of Toronto)

**通讯引用:** 2055 | [OpenAlex ID](https://openalex.org/A5100659941)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何用 LLM 与 BERTopic 对加拿大儿童福利机构的案例笔记和服务计划进行主题分析和进度跟踪，为案例工作者提供辅助工具。

**💡 创新点**

首次将本地 LLM 与主题建模结合，揭示 LLM 在复杂、长期案例中无法准确识别活动相关笔记的局限，并提出公共部门 LLM 开发的协作路线图。

**🔧 技术方法**

采用本地 LLM（Llama 3.1）与 BERTopic 主题模型，并通过 Prompt Engineering 对笔记进行摘要与活动相关性判断。

**📊 数据集**

使用 720 家庭的常规案例笔记（52,748 条）和服务计划（1,213 个）作为文本数据集。

**📈 对比分析**

通过人工标注与 LLM 标注的 Cohen's κ 对比，短期案例 κ≈0.60，长期案例降至 0.40，误判率随案例长度增加而上升，表明 LLM 在复杂情境下性能下降。

**⚠️ 局限性**

局限在于仅聚焦单一加拿大机构的数据，LLM 需面对模糊的社会工作判断，缺乏多模型验证与跨机构普适性验证。

---

## 423. A Real-Time Error Prevention System for Gaze-Based Interaction in Virtual Reality Based on Anomaly Detection

**arXiv ID:** 2601.15146 | [PDF](https://arxiv.org/pdf/2601.15146v1)

**作者:** Björn R. Severitt `[一作]` (University of Tübingen), Siegfried Wahl `[通讯]` (Carl Zeiss Vision International GmbH)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在虚拟现实环境中开发并评估了一种基于异常检测的实时误差预防系统（EPS），用于眼动交互，显著降低错误选择并提升用户体验。

**💡 创新点**

创新点在于：①将TCN自编码器用于实时眼动异常检测；②在选择后约200 ms内即时识别错误；③在三种眼动交互方法（dwell、gaze+head、nod）中进行系统化比较；④提供无干扰的误差预防，提升信任感。

**🔧 技术方法**

使用技术包括：TCN自编码器（TCNAE）实现于PyTorch；眼动角速度特征提取；阈值式异常判定；HTC Vive Pro Eye眼追踪与Zero‑Interface；VR游戏环境和主观问卷。

**📊 数据集**

数据集为41名受试者在VR视觉搜索任务中收集的眼动记录，包含约52名参与者的正确与错误选择事件；用于训练和评估。

**📈 对比分析**

通过在每种交互方法下设置有无EPS两条件，比较主观评分、错误次数、得分以及EPS的正确/错误识别率。EPS在dwell和gaze+head方法中将错误率降低至约5 %（最高可达95 %），分数提升显著；在nod方法错误已极低，提升有限；整体正确率在0.79–0.91之间，错误率大幅下降。

**⚠️ 局限性**

局限性包括：仅在单一视觉搜索VR任务中验证，难以直接推广到其他任务或AR环境；误差检测阈值为静态，缺乏个性化或自适应机制；nod方法错误率低，难以充分评估EPS效果；需进一步测试泛化性和动态阈值调整。

---

## 424. RSNA Large Language Model Benchmark Dataset for Chest Radiographs of Cardiothoracic Disease: Radiologist Evaluation and Validation Enhanced by AI Labels (REVEAL-CXR)

**arXiv ID:** 2601.15129 | [PDF](https://arxiv.org/pdf/2601.15129v1)

**作者:** Yishu Wei `[一作]` (Weill Cornell Medicine), George Shih `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 3926 | [OpenAlex ID](https://openalex.org/A5018859884)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发了一套基于人工智能辅助的胸片标注流程，并构建了包含200张胸部X线的公开基准数据集REVEAL-CXR。

**💡 创新点**

创新点在于将LLM（GPT‑4o）提取报告中的异常信息并用本地Phi‑4模型映射到12个临床标签，随后由17名专家以三人投票方式核实，形成高质量、包含罕见病变的多标签基准。

**🔧 技术方法**

技术手段包括GPT‑4o、Phi‑4‑Reasoning进行标签映射，指导解码技术，分层抽样算法，以及Web‑PACS风格的协同标注平台。

**📊 数据集**

数据集来源为MIDRC公开的13,735例去标识化胸片与报告，通过AI自动生成候选标签后抽样得到100例公开和100例保留的标注胸片。

**📈 对比分析**

对标注质量采用Cohen’s κ检验，整体二分类一致度为0.622，绝大多数单一诊断κ>0.75；该基准目前用于评估多模态LLM性能，尚未给出模型性能数值。

**⚠️ 局限性**

局限性包括：映射仅基于初步LLM提取的发现，Phi‑4模型规模较小，缺乏临床病史与人口学信息，标注粒度过粗以及专家在“同意多数”与“不同意”区分不清。

---

## 425. RayRoPE: Projective Ray Positional Encoding for Multi-view Attention

**arXiv ID:** 2601.15275 | [PDF](https://arxiv.org/pdf/2601.15275v1)

**作者:** Yu Wu `[一作]` (Carnegie Mellon University), Shubham Tulsiani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4640 | [OpenAlex ID](https://openalex.org/A5029932788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于光线段的相对位置编码RayRoPE，用于多视图Transformer。

**💡 创新点**

创新点在于：①使用光线段并预测深度以实现几何自适应；②利用预期RoPE对深度不确定性建模；③在查询相机坐标系下投影光线，实现SE(3)不变性；④支持多频率相似度。

**🔧 技术方法**

采用Transformer注意力、RoPE、投影操作、深度预测网络、期望RoPE等技术。

**📊 数据集**

在新视角合成任务使用CO3D、Objaverse、RealEstate10K；在立体深度估计使用RGBD、SUN3D、Scenes11。

**📈 对比分析**

与Plucker raymap、RoPE on rays、GTA、PRoPE等基线对比，RayRoPE在PSNR、LPIPS、SSIM上均有提升，尤其在CO3D上相对LPIPS提升约15%，在不同数据集上均优于基线。

**⚠️ 局限性**

局限性：仅对深度不确定性建模，未考虑相机矩阵的不确定性；目前仅支持已标定姿态的图像，未解决无姿态或混合姿态图像的情况。

---

## 426. The Effect of Scripts and Formats on LLM Numeracy

**arXiv ID:** 2601.15251 | [PDF](https://arxiv.org/pdf/2601.15251v1)

**作者:** Varshini Reddy `[一作]` (Kensho Technologies), Chris Tanner `[通讯]` (MIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了大语言模型在不同数字脚本和格式下进行基本算术时的性能，并提出通过提示策略缓解性能下降的方法。

**💡 创新点**

创新点在于系统量化了“脚本税”和“格式税”对算术准确率的影响，并证明少样本提示和显式映射可显著提升跨脚本/格式推理的鲁棒性。

**🔧 技术方法**

主要技术包括多脚本与多格式数据集构建、对LLM的零样本与提示式算术评测、以及使用GLMER模型分析影响因素。

**📊 数据集**

使用了336条基础算术表达式，分别转化为20种数字脚本（共21种脚本）和6种数字格式，形成跨脚本/格式评测数据集。

**📈 对比分析**

对9个LLM（4大模型+5小模型）进行对比实验，发现非标准脚本/格式导致准确率下降约66–87%，而加入少样本提示或映射可将准确率提升数十个百分点。

**⚠️ 局限性**

局限性包括未探讨更复杂的提示或工具调用、仅覆盖有限的格式变体、以及对低资源脚本的评测样本有限，可能未能完全覆盖真实世界的数字表示多样性。

---

## 427. Recommending Best Paper Awards for ML/AI Conferences via the Isotonic Mechanism

**arXiv ID:** 2601.15249 | [PDF](https://arxiv.org/pdf/2601.15249v1)

**作者:** Garrett G. Wen `[一作]` (Yale University), Weijie Su `[通讯]` (University of Pennsylvania)

**通讯引用:** 7867 | [OpenAlex ID](https://openalex.org/A5080575294)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于作者自评排名的等距回归（Isotonic Mechanism）辅助最佳论文奖项评选机制。

**💡 创新点**

核心创新在于：① 将原先对效用函数凸性要求放宽到单个提名（quota=1）时仅需单调性；② 在最佳论文场景下验证凸性假设并给出经验支持；③ 设计了盲目（Blind）与信息化（Informed）两种评选策略，并证明其激励兼容与个体理性。

**🔧 技术方法**

主要技术手段包括：等距回归（Isotonic regression）、交换性噪声模型、Schur-convex性与主导性理论、以及基于作者排名的投影调整；同时采用模拟与真实数据检验。

**📊 数据集**

使用公开会议数据：ICLR 2019‑2023、NeurIPS 2021‑2023 评审分数与接受结果；以及基于ICLR 2021 合著网络与均匀网络的合成数据进行仿真实验。

**📈 对比分析**

对比方法有：基准（仅用原始分数）、盲目协议（Blind）与信息化最大化协议（Informed, Max）。评估指标为所选论文的平均真实质量与真正最高质量论文的比值。实验显示盲目协议在所有设置下均显著优于基准，信息化最大化在某些网络/参数组合下表现不如基准。

**⚠️ 局限性**

局限性包括：仍依赖效用函数可加与凸性（或单调性）假设；噪声需满足交换性；未在真实会议中实测，缺乏对作者真实参与意愿与策略多样性的验证；对不同作者配额（quota）分配的公平性与效果仍未充分探索。

---

## 428. Taxonomy-Aligned Risk Extraction from 10-K Filings with Autonomous Improvement Using LLMs

**arXiv ID:** 2601.15247 | [PDF](https://arxiv.org/pdf/2601.15247v1)

**作者:** Rian Dolphin `[一作]` (Massive.com), Quinton Pike `[通讯]` (Massive.com)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

从10-K文件中抽取风险因素并将其映射到预定义的三层风险分类体系，形成可用于投资分析的结构化风险列表。

**💡 创新点**

三阶段混合管道：LLM先无约束抽取风险及引用句；嵌入相似度将抽取文本映射到分类项；LLM-as-Judge对映射进行评分过滤，同时反馈用于自主管理与持续改进分类描述。

**🔧 技术方法**

采用大型语言模型（Claude 4.5 Sonnet、LLM-judge）、任务指令调优嵌入模型（Qwen3 Embedding 0.6B）、近邻相似度检索、结构化输出与函数调用、并行LLM评估。

**📊 数据集**

2024年标准普尔500指数公司10-K表1A风险披露文本，共提取10688条风险；使用此文本构建风险矩阵及评估数据。

**📈 对比分析**

通过对同业与异业公司风险向量的加权余弦相似度进行统计检验：同业平均相似度比异业高63%（p<0.001），AUC在行业粒度从2位SIC到4位为0.733–0.822；自适应分类改进后，药品监管类别嵌入分离度提升104.7%。

**⚠️ 局限性**

受限于LLM调用成本与延迟、需人工先行构建分类体系、目前仅支持英文、嵌入模型与LLM版本更新需同步，且极大文本量仍可能导致抽取与验证瓶颈。

---

## 429. Towards Understanding Best Practices for Quantization of Vision-Language Models

**arXiv ID:** 2601.15287 | [PDF](https://arxiv.org/pdf/2601.15287v1)

**作者:** Gautom Das `[一作]` (University of Maryland), Matthew Gwilliam `[通讯]` (University of Maryland)

**通讯引用:** 144 | [OpenAlex ID](https://openalex.org/A5031894535)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了在多模态视觉‑语言模型（如 BLIP‑2、LLaVA）中采用量化（包括 GPTQ、AWQ、统一量化）对模型性能的影响，重点分析了不同组件（视觉编码器、Q‑Former、LLM）的敏感性及其在不同任务（检索、captioning、VQA）中的重要性。

**💡 创新点**

创新点在于：①首次在多模态管道中全面比较先进量化方法与统一量化的效果；②提出基于随机森林、置换特征重要性和 SHAP 的三方树模型重要性分析框架，揭示不同量化方法对各组件重要性分布的重塑；③提供了针对任务、方法和架构的层次化量化策略建议。

**🔧 技术方法**

主要技术包括：统一量化（k‑bit）、GPTQ（基于 Hessian 的权重量化）、AWQ（激活感知权重量化），以及随机森林回归、置换重要性、SHAP 等特征重要性评估方法。

**📊 数据集**

使用的数据集有 COCO Captioning、VQAv2、GQA、Flickr‑Text/Image，用于检索、captioning 和 VQA 任务。

**📈 对比分析**

与全精度模型对比，SOTA 量化方法能在 3.5‑4.5 位/权重（bpw）下保持与原模型相近的性能，而统一量化通常需要 6‑10 bpw。实验还表明，AWQ 在极低位宽下对 LLM 的影响更大，而 GPTQ 的影响更均衡。

**⚠️ 局限性**

局限性包括：①实验仅在模拟量化下进行，未评估真实硬件上的延迟与能耗；②未考虑量化对推理加速和内存占用的具体数值；③仅聚焦视觉‑语言模型，未扩展到音频或视频等其他模态。

---

## 430. LLM-based Multimodal Feedback Produces Equivalent Learning and Better Student Perceptions than Educator Feedback

**arXiv ID:** 2601.15280 | [PDF](https://arxiv.org/pdf/2601.15280v1)

**作者:** Chloe Qianhui Zhao `[一作]` (Carnegie Mellon University), Kenneth R. Koedinger `[通讯]` (Carnegie Mellon University)

**通讯引用:** 26125 | [OpenAlex ID](https://openalex.org/A5062550465)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在在线实验中开发并评估了一套实时 AI 促进的多模态反馈系统，比较其与传统教师预写反馈在学习效果、学习者参与度和感知质量方面的表现。

**💡 创新点**

创新点在于将结构化文本解释、检索最相关的课件页和可选的 AI 语音叙述三种模态实时组合，形成可嵌入现有 ITS 的即时、情境感知反馈，并通过多模态协同降低认知负荷。

**🔧 技术方法**

采用的大技术包括大语言模型（如 GPT‑5/Claude/ Gemini）、检索增强生成（RAG）与视觉嵌入、流式音频叙述 API 以及前端嵌入式组件实现反馈展示。

**📊 数据集**

使用的数据集为一门学习科学课程的多媒体原理讲义幻灯片（文本、图片、视频）以及通过 Prolific 招募的 197 名美国大学生的学习日志与问卷数据。

**📈 对比分析**

通过随机对照实验（BBA 与 AI 多模态）结合 pre‑test/post‑test 分数、学习日志（提交次数、时间）和 Likert 量表调查进行比较。学习增益在两组无显著差异；AI 组在清晰度、特异性、简洁性、动机、满意度等主观指标上显著优于传统组；在正确性、信任度、接受度方面相当。

**⚠️ 局限性**

局限性包括：实验仅为单次在线课时，缺乏长期记忆与转移效应评估；样本仅为美国大学生，外部可推广性有限；未能拆解各反馈模态的单独贡献；仅测量总体认知负荷，未区分外部与生成负荷。

---

## 431. Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks

**arXiv ID:** 2601.15277 | [PDF](https://arxiv.org/pdf/2601.15277v1)

**作者:** Sahar Tahmasebi `[一作]` (Leibniz Information Centre for Science and Technology), Ralph Ewerth `[通讯]` (Marburg University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨了大型语言模型生成的情感攻击对假新闻检测器的鲁棒性，并提出了 AdSent 框架实现情感无关的假新闻检测

**💡 创新点**

创新点在于：① 用 LLM 生成可控情感的对抗样本；② 通过在中性情感文本上训练，构建情感无关的检测器；③ 证明情感中性化训练提升对情感攻击与风格攻击的鲁棒性

**🔧 技术方法**

使用 LLM（如 LLaMA‑3.1‑8B‑Instruct、Qwen‑2.5‑7B‑Instruct）进行情感转化与对抗生成，结合 prompt 工程、对抗训练和多模型评估

**📊 数据集**

PolitiFact、GossipCop 与 LUN 三大假新闻基准数据集

**📈 对比分析**

与 RoBERTa、SheepDog、零射击 LLM 等基线在原始、情感攻击和中性化测试集上对比，AdSent 在中性化测试集上 Macro‑F1 最高达 87.8%（PolitiFact）和 78.6%（GossipCop），整体性能显著优于现有方法

**⚠️ 局限性**

仍存在对多模态信息（图像等）的处理不足、LLM 生成文本中事实一致性难以完全保证、以及对极端情感极化或其他新闻价值（如新颖性、接近度）的鲁棒性尚未验证

---

## 432. Iterative Refinement Improves Compositional Image Generation

**arXiv ID:** 2601.15286 | [PDF](https://arxiv.org/pdf/2601.15286v1)

**作者:** Shantanu Jaiswal `[一作]` (Carnegie Mellon University), Deepak Pathak `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6218 | [OpenAlex ID](https://openalex.org/A5101851026)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种推理时迭代自纠的图像生成框架，结合图像生成器、编辑器和视觉语言模型批评家，在每一步根据批评反馈生成子提示并更新图像，以实现复杂提示的高保真生成。

**💡 创新点**

创新点在于把链式思维的迭代推理迁移到图像生成领域，使用VLM批评家在闭环中动态给出子提示和动作（继续、重置、回溯），无需额外工具或先验即可实现自纠。

**🔧 技术方法**

使用的技术包括：基于文本的图像生成器与图像编辑器、视觉语言模型（VLM）作为批评家和验证器、并行流与迭代步数的预算分配策略，以及动作空间（终止、回溯、重置、继续）控制。

**📊 数据集**

实验使用的主要数据集有：ConceptMix、T2I‑CompBench、TIIF‑Bench、Visual Jenga 等多种复杂组合与分解基准。

**📈 对比分析**

与传统并行采样（计算匹配）基线对比，迭代框架在 ConceptMix 上 all‑correct 率提升 16.9%、T2I‑CompBench 3D‑Spatial 提升 13.8%、Visual Jenga 完整分解率提升 12.5%，并在人类评测中获得 58.7% 的偏好率，表现显著优于基线。

**⚠️ 局限性**

局限性包括：对 VLM 质量敏感，批评家能力受限时性能下降；需要可编辑的图像生成器，迭代步数与预算平衡需手动调节，可能出现过度迭代导致的无效计算。

---

## 433. Walk through Paintings: Egocentric World Models from Internet Priors

**arXiv ID:** 2601.15284 | [PDF](https://arxiv.org/pdf/2601.15284v1)

**作者:** Anurag Bagchi `[一作]` (Carnegie Mellon University), Martial Hebert `[通讯]` (Carnegie Mellon University)

**通讯引用:** 42481 | [OpenAlex ID](https://openalex.org/A5075246991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在预训练的视频扩散模型中注入动作嵌入并利用时间步调制的方式，将其转化为可控的世界模型，实现对不同机器人、仿真和现实场景的未来视觉预测。

**💡 创新点**

创新点在于提出一种架构无关、轻量级的动作条件化方案，能够直接利用现有扩散模型的时间步路径，同时引入新的结构一致性得分（SCS）评价指标，显著提升物理一致性和跨域泛化。

**🔧 技术方法**

采用视频扩散模型（SVD、Cosmos）、动作投影MLP、时间步调制、低维动作嵌入、结构一致性得分计算等技术，并进行少量动作标注数据的微调。

**📊 数据集**

使用RECON、SCAND、TartanDrive等真实机器人导航数据以及1X Humanoid（25-DoF）数据集进行训练与评估。

**📈 对比分析**

与专用的Navigation World Models (NWM) 对比，SCS提升最多达80%，在LPIPS、DreamSim上也保持或优于NWM，同时推理延迟降低约6倍，展示了更优的动作跟随和实时性。

**⚠️ 局限性**

在高维动作空间和长时延轨迹下仍易出现漂移，模型依赖预训练数据的多样性，对极端物理交互或完全未见场景的精细控制仍有限。

---

## 434. LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes

**arXiv ID:** 2601.15283 | [PDF](https://arxiv.org/pdf/2601.15283v1)

**作者:** Ruofan Liang `[一作]` (University of Toronto), Christian Richardt `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种从单一多视角捕捉中实现室内场景光照分解与可交互重映射的方法，能够将复杂照明拆分为可单独控制的光源并实现实时渲染。

**💡 创新点**

融合单图光照分解、跨视角光照谐调与基于3D高斯散射的可重混光照表示，首次实现多视角一致的逐光源编辑与实时交互。

**🔧 技术方法**

使用预训练Diffusion模型（如DiT/FLUX）通过LoRA微调实现单图OLAT分解，利用多视角扩散U-Net实现光照谐调，采用3D高斯散射编码每个光源的HDR颜色，实现实时渲染。

**📊 数据集**

构建12,000个合成室内场景，配以多光源（最多6灯）并生成全光、单光与环境光等多种照明配置，用于训练与评估。

**📈 对比分析**

在30个保留的合成测试集上与ScribbleLight、Qwen-Image等基线对比，单图分解PSNR提升至27.7、SSIM 0.898、LPIPS 0.082，跨视角谐调PSNR 30.8、SSIM 0.867、LPIPS 0.091，显示显著优于基线。

**⚠️ 局限性**

仅在静态合成室内场景上训练，难以推广到动态或户外场景；光源种类有限，偏好灯锥型光；不支持远场HDRI光照编辑。

---

## 435. StableWorld: Towards Stable and Consistent Long Interactive Video Generation

**arXiv ID:** 2601.15281 | [PDF](https://arxiv.org/pdf/2601.15281v1)

**作者:** Ying Yang `[一作]` (Nanjing University), Chenyang Si `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种动态帧驱逐机制StableWorld，用以在交互式视频生成中持续剔除已偏移的帧，从而抑制累计漂移和场景崩塌。

**💡 创新点**

创新点在于利用ORB特征与RANSAC几何一致性评估实时判断帧相似度，自动维护一个清洁且几何一致的历史窗口，兼顾运动连贯性与场景切换灵活性。

**🔧 技术方法**

技术包括基于自回归扩散模型的交互式视频生成、滑动窗口策略、ORB+RANSAC几何相似度评估与动态驱逐逻辑，并在多种模型上实现无缝集成。

**📊 数据集**

使用Matrix-Game 2.0、Open-Oasis、Hunyuan-GameCraft 1.0三种交互式视频模型，以及VBench-Long基准和自组织的用户研究视频集进行评估。

**📈 对比分析**

与原始模型相比，StableWorld在VBench-Long指标上均有显著提升（例如Aesthetic提升14.6%，Image Quality提升7.4%），用户研究中偏好率提升至80%以上，计算开销仅提升1.0–1.02倍。

**⚠️ 局限性**

局限性包括阈值设置对驱逐效果影响敏感；对极端视角变化或剧烈场景切换的鲁棒性仍待进一步验证；以及需要在不同模型间手动调参以获得最佳性能。

---

## 436. Distributed Agent-Constrained Truthful Facility Location

**arXiv ID:** 2601.15258 | [PDF](https://arxiv.org/pdf/2601.15258v1)

**作者:** Argyrios Deligkas `[一作]` (Royal Holloway University of London), Alexandros A. Voudouris `[通讯]` (University of Essex)

**通讯引用:** 705 | [OpenAlex ID](https://openalex.org/A5090907162)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文研究了分布式设施位置问题：将代理人按预先划分的若干组，每组选取一名代理人成为代表点，然后从这些代表点中选取k个位置作为设施位置，目标是最小化代理人总成本（sum-variant）或最大距离成本（max-variant），并设计确定性可策略化机制并给出近似比。

**💡 创新点**

创新点在于：①首次为k=2的情况给出最优的可策略化机制，sum-variant的近似比为1+√2，max-variant为9/2；②为k≥3给出上下界 3±2/k（sum）和 2k–2(k+1)（max）；③证明在sum-variant中，1+√2是任意算法的下界；④引入基于有序统计的机制框架并利用局部性递推实现可策略化。

**🔧 技术方法**

主要技术包括：有序统计选点机制、策略可证明性证明（通过局部性与位置不变性）、距离三角不等式与区间划分、递归构造下界实例、以及对称化组处理简化证明。

**📊 数据集**

本文完全是理论分析，没有使用任何实验数据集。

**📈 对比分析**

在理论比较中，作者把所设计的机制与任意无约束算法进行对比，证明在k=2时其近似比达到下界，且在k≥3时给出了最优上下界；对比结果表明所给机制在可策略化框架下已达到最优或几乎最优性能。

**⚠️ 局限性**

局限性包括：①对k≥3的情况上下界仍存在小间隙；②仅考虑确定性机制，未探讨随机化机制的潜在优势；③模型仅适用于一维线性空间且只考虑sum/ max两种成本；④实验验证与实际应用情景尚未展开。

---

## 437. FlowSSC: Universal Generative Monocular Semantic Scene Completion via One-Step Latent Diffusion

**arXiv ID:** 2601.15250 | [PDF](https://arxiv.org/pdf/2601.15250v1)

**作者:** Zichen Xi `[一作]` (Ant Group), Qun-Ce Xu `[通讯]` (Tsinghua University)

**通讯引用:** 51 | [OpenAlex ID](https://openalex.org/A5089423114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种单目语义场景补全的通用生成框架FlowSSC，利用一次性生成实现实时高质量补全；

**💡 创新点**

创新点在于将语义场景压缩为三平面潜在空间并使用Shortcut Flow‑Matching训练，使得扩散模型可在单步内完成高保真生成；

**🔧 技术方法**

采用VecSet VAE进行三平面压缩、Cross‑Attention编码、Triplane Diffusion Transformer (DiT)与Shortcut Flow‑Matching的联合训练；

**📊 数据集**

使用SemanticKITTI数据集进行训练与评估；

**📈 对比分析**

与现有最先进的单目补全方法相比，FlowSSC在Semantic mIoU和几何IoU上均超过10%以上，单步推理耗时仅66ms；

**⚠️ 局限性**

主要局限在于模型体积大、显存需求高、训练过程仍耗时，且在极端遮挡下仍可能出现错误补全。

---

## 438. Lightweight LLMs for Network Attack Detection in IoT Networks

**arXiv ID:** 2601.15269 | [PDF](https://arxiv.org/pdf/2601.15269v1)

**作者:** Piyumi Bhagya Sudasinghe `[一作]` (University of Ruhuna), Harsha S. Gardiyawasam Pussewalage `[通讯]` (University of Agder)

**通讯引用:** 326 | [OpenAlex ID](https://openalex.org/A5054991662)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对 IoT 网络攻击检测，构建了一个轻量化解码器型大语言模型框架，结合结构化到文本转换、QLoRA 微调与检索增强生成（RAG）实现对已知及未知攻击的识别。

**💡 创新点**

创新点在于将 QLoRA 的低秩适配与 4‑bit 量化与 RAG 相结合，既保持模型参数极小、推理成本低，又能在不额外训练的情况下通过检索上下文实现零样本攻击分类。

**🔧 技术方法**

核心技术包括：结构化特征到自然语言提示的转换、QLoRA（4‑bit 量化 + LoRA）微调、检索增强生成（RAG）与 GPT‑2/LLaMA‑3.2‑1B 等轻量解码器 LLM 的使用。

**📊 数据集**

使用 CICIoT2023 数据集，包含 34 类（33 攻击 + 一类正常）网络流特征。

**📈 对比分析**

与传统机器学习基线（RF、SVM 等）对比，微调后的 LLaMA‑1B 在已知攻击上的 F1 分数为 0.7124，几乎等同于 RF 的 0.7159；RAG 模型在未见攻击上的准确率为 42.63%，展示了可观的零样本能力。

**⚠️ 局限性**

局限性主要体现在：实验仅在单一 CICIoT2023 数据集上验证；对更广泛的 IoT 环境、不同流量特征的泛化能力和在更大规模检索库下的性能尚待进一步探究。

---

## 439. Evaluation of Large Language Models in Legal Applications: Challenges, Methods, and Future Directions

**arXiv ID:** 2601.15267 | [PDF](https://arxiv.org/pdf/2601.15267v1)

**作者:** Yiran Hu `[一作]` (Tsinghua University), Ben Kao `[通讯]` (University of Hong Kong)

**通讯引用:** 6203 | [OpenAlex ID](https://openalex.org/A5063695659)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统分类了大语言模型在法律任务中的评估方法，梳理评估维度（结果准确性、法律推理、可信度）及对应挑战，细致分析现有多任务基准与评价指标。

**💡 创新点**

提出三维评估框架，强调在法律应用中需同时关注结果、推理过程与可信度；将多任务基准按维度分类并深入剖析，指出评估现状与未来方向。

**🔧 技术方法**

使用多种现有评估技术：准确率、F1、ROUGE‑L、BERTScore、NDCG、MAE、统计公平度量、LLM‑as‑a‑judge、专家手工评分等；并讨论其适用场景与局限。

**📊 数据集**

参考的数据集与基准包括 LexEval、LegalBench、JudiFair、Crown, Coliee、LeCaRD、SARA、PrivacyQA 等，涵盖考试题、案例判决、问答、检索、伦理评估等多样化数据来源。

**📈 对比分析**

对各基准中的模型进行对比评测，结果显示在结果准确性上已有显著提升，但在推理细粒度与公平性评估中仍显不足；大模型往往出现幻觉、偏差与逻辑不严谨，性能差异明显。

**⚠️ 局限性**

局限性包括：评估数据多为考试式或标准化案例，缺乏真实法律工作流程；评估指标多为表面相似度或单一统计量，无法捕捉细粒度法律逻辑；跨法域、跨语言、隐私安全与安全性等关键维度尚未充分覆盖；数据污染与偏差可能影响评测公平性。

---

## 440. DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration

**arXiv ID:** 2601.15260 | [PDF](https://arxiv.org/pdf/2601.15260v1)

**作者:** Dominik Rößle `[一作]` (Technische Hochschule Ingolstadt), Torsten Schön `[通讯]` (Technische Hochschule Ingolstadt)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5078954824)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了DrivIng数据集，该数据集包含约18公里全路线的多模态感知数据，配备6摄像头、1雷达、精确定位，三种时段（白天、黄昏、夜间）共计约1.2百万个3D标注实例，并构建与之对应的高精度数字孪生；

**💡 创新点**

创新点在于：①提供长连续路段的全路线感知数据；②配备完整数字孪生，实现真实-模拟一对一映射；③覆盖多光照、不同路段并包含多传感器同步数据；④公开了多任务基准和工具链；

**🔧 技术方法**

技术包括：多摄像头与雷达数据同步与标定、RTK/IMU精确定位、基于CARLA的数字孪生重建与两种重放模式、nuScenes格式转换、使用MMDetection3D训练PETR（相机）与CenterPoint（雷达）三维目标检测模型；

**📊 数据集**

使用的数据集为自研的DrivIng数据集，包含三段不同光照的连续路段；对比实验也使用了nuScenes格式下的PETR和CenterPoint模型；

**📈 对比分析**

比较方法为在nuScenes评估协议下计算ATE、ASE、AOE、AVE、NDS、mAP；实验结果显示CenterPoint雷达模型在所有时段的mAP和NDS均显著高于PETR相机模型；但两者均随光照从白天到夜间性能下降；

**⚠️ 局限性**

局限性包括：①数字孪生的车辆模型视觉质量有限；②雷达与摄像头的同步和标定仍需精细处理；③数据集虽然覆盖多场景，但仍以单一路段为主，泛化到更广阔区域仍待验证；④未覆盖夜间极端低光照下的视觉鲁棒性研究。

---

## 441. APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping

**arXiv ID:** 2601.15288 | [PDF](https://arxiv.org/pdf/2601.15288v1)

**作者:** Jiwon Kang `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Attribute-Preserving Pseudo-Labeling（APPLE）框架，使用 teacher–student 扩散模型实现面部身份交换时的高质量属性保持。

**💡 创新点**

创新点包括：①将面部交换任务从传统掩码填充改为条件去模糊，保留低频属性信息；②引入属性感知反演，利用仅属性条件的噪声保持目标细节；③通过高质量伪三元组监督让学生直接在完整目标上训练，避免额外结构预处理；④学生模型在推理时不需任何辅助条件，提升实用性。

**🔧 技术方法**

技术细节：使用 FLUX.1-Krea 作为扩散基础，PulID 作为身份编码器，OminiControl 进行属性编码；采用条件去模糊训练、属性感知反演生成伪标签；学生模型在伪三元组上以直接编辑目标训练；整体采用流匹配（rectified flow）与身份损失相结合。

**📊 数据集**

训练集：VGGFace2-HQ（经过 AES 过滤）；评估集：FFHQ（1,000 对源/目标面孔）。

**📈 对比分析**

与 DiffSwap、DiffFace、E4S、GAN 基础模型（如 FaceDancer）对比；指标包含 FID、姿态/表情 L2、ArcFace 余弦距离、Top‑1/Top‑5 识别率。结果显示 APPLE 在属性保持（Fidelity、姿态/表情一致性）方面取得最低 FID 并与最佳身份保持模型持平，显著优于现有扩散和 GAN 方法。

**⚠️ 局限性**

局限性：①身份与属性仍存在权衡，极端表情或光照变化下可能稍逊；②训练对 GPU 资源和伪标签质量高度依赖；③对极端遮挡、配件复杂度的鲁棒性尚待进一步验证。

---

## 442. Rethinking Video Generation Model for the Embodied World

**arXiv ID:** 2601.15282 | [PDF](https://arxiv.org/pdf/2601.15282v1)

**作者:** Yufan Deng `[一作]` (Peking University), Daquan Zhou `[通讯]` (Peking University)

**通讯引用:** 8707 | [OpenAlex ID](https://openalex.org/A5100554498)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了针对机器人视频生成的评测基准 RBench 与大型数据集 RoVid‑X，并对 25 款模型进行系统评估。

**💡 创新点**

创新点在于引入任务完成与视觉质量双维度的细粒度评测指标，及首个大规模多机器人、多任务高质量视频数据集。

**🔧 技术方法**

使用多模态大模型（如 Qwen3‑VL、GPT‑5）进行自动 VQA 评估、GroundingDINO/GroundedSAM+CoTracker 提取运动特征，结合物理可行性与动作一致性检查。

**📊 数据集**

使用 RoVid‑X（约 400 万段机器人视频）以及 20+公开机器人数据集进行数据收集与标注。

**📈 对比分析**

通过自动化指标与人工偏好实验对比，RBench 与人类评价的 Spearman ρ≈0.96；结果显示闭源模型领跑，物理推理能力仍落后于视觉质量。

**⚠️ 局限性**

局限在于评测主要聚焦任务完成与视觉一致性，缺乏对执行动作的可执行性与真实机器人控制的闭环验证；数据集虽大但仍局限于互联网与公开来源，可能存在偏差。

---

## 443. MolecularIQ: Characterizing Chemical Reasoning Capabilities Through Symbolic Verification on Molecular Graphs

**arXiv ID:** 2601.15279 | [PDF](https://arxiv.org/pdf/2601.15279v1)

**作者:** Christoph Bartmann `[一作]` (Ellis Unit Linz and Lit AI Lab), Sohvi Luukkonen `[通讯]` (Ellis Unit Linz and Lit AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MolecularIQ benchmark，基于符号可验证的分子图任务评估LLM的化学推理能力。

**💡 创新点**

创新点在于构造了全符号可验证的三类任务（计数、索引、约束生成）和多维复杂度轴，避免数据泄露与模式匹配。

**🔧 技术方法**

采用RDKit符号求解器、SMILES多样化处理、层级提取与验证框架以及基于图论的特征检测技术。

**📊 数据集**

使用从PubChem抽取的单片段分子集合，按Bertz复杂度与聚类划分为训练、易测试和难测试集。

**📈 对比分析**

通过对38个公开LLM（含通用与化学专用模型）的全面评估，发现大模型如GPT‑OSS、Qwen‑3在结构推理上最高，但对SMILES扰动和低频约束仍易失效。

**⚠️ 局限性**

局限在于仅覆盖二维符号可验证任务，缺乏属性预测、3D几何等更丰富化学场景，且只评估单分子单模态推理。

---

## 444. Interpreting Multimodal Communication at Scale in Short-Form Video: Visual, Audio, and Textual Mental Health Discourse on TikTok

**arXiv ID:** 2601.15278 | [PDF](https://arxiv.org/pdf/2601.15278v1)

**作者:** Mingyue Zha `[一作]` (Dartmouth), Ho-Chun Herbert Chang `[通讯]` (Dartmouth)

**通讯引用:** 713 | [OpenAlex ID](https://openalex.org/A5043059930)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文开发了一套可复现的多模态分析管道，结合文本、视觉、音频特征的零样本概率标注与Shapley值解释，研究TikTok上关于社交焦虑障碍内容的观看行为。

**💡 创新点**

创新点在于将零样本概率分类与特征加权SHAP相结合，并通过分段回归揭示跨模态阈值依赖的协同效应，提供可解释的多模态影响框架。

**🔧 技术方法**

使用了GPT‑4o‑Mini进行零样本标注、VADER情感分析、PyFeat人脸AU检测、Whisper语音转写、CatBoost/XGBoost模型、SHAP解释以及分段回归。

**📊 数据集**

采用了2020‑2025年间美国发布的162,965条关于社交焦虑的TikTok视频及其元数据作为数据集。

**📈 对比分析**

通过对单模态SHAP贡献和跨模态交互的比较，CatBoost模型实现了良好的预测精度（R²≈0.5+），并验证了跨模态阈值效应显著。

**⚠️ 局限性**

主要局限包括音频中音乐与语音混合导致特征噪声、仅针对TikTok的单平台样本、未考虑视频内时间演变的动态特征。

---

