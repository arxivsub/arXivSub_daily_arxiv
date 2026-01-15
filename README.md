# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-01-15 | 今日论文总数: 410

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Bias Detection and Rotation-Robustness Mitigation in Vision-Language Models and Generative Image Models

**arXiv ID:** 2601.08860 | [PDF](https://arxiv.org/pdf/2601.08860v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 2. NewsScope: Schema-Grounded Cross-Domain News Claim Extraction with Open Models

**arXiv ID:** 2601.08852 | [PDF](https://arxiv.org/pdf/2601.08852v1)

**作者:** Nidhi Pandya `[一作]` `[通讯]` (Pace University), Nidhi Pandya (Pace University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建NewsScope数据集并训练开源模型进行跨域结构化新闻声明提取

**💡 创新点**

提出跨域结构化提取的标准化JSON schema以及开源模型在不同领域的可推广性

**🔧 技术方法**

使用LLaMA 3.1 8B+LoRA微调，配合数值归一化后处理过滤

**📊 数据集**

利用455篇新闻（395篇训练/测试，60篇外域测试），覆盖政治、健康、科学/环境、商业四个领域

**📈 对比分析**

与GPT‑4o‑mini对比，NewsScope在人类评估下准确率89.4%（加过滤后91.6%），在政治领域超越对手

**⚠️ 局限性**

仅做提取未实现验证；训练数据使用GPT生成的银标注可能带教师偏差，单一评注人造成可靠性不足

---

## 3. LPCAN: Lightweight Pyramid Cross-Attention Network for Rail Surface Defect Detection Using RGB-D Data

**arXiv ID:** 2601.09118 | [PDF](https://arxiv.org/pdf/2601.09118v1)

**作者:** Jackie Alex `[一作]` (Saint Petersburg University), Guoqiang Huan `[通讯]` (Saint Petersburg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级金字塔交叉注意力网络（LPCANet）用于RGB‑D铁路表面缺陷检测。

**💡 创新点**

创新点在于引入跨模态注意力机制和空间特征提取器，并通过轻量级金字塔模块实现高效深度特征融合。

**🔧 技术方法**

使用MobileNetv2骨干、轻量级金字塔模块、交叉注意力模块、空间特征提取器以及像素重排上采样等技术。

**📊 数据集**

在三套无监督RGB‑D铁路缺陷数据集（NEU‑RSDDS‑AUG、RSDD‑TYPE1、RSDD‑TYPE2）以及非铁路缺陷数据集（DAGM2007、MT、Kolektor‑SDD2）上进行实验。

**📈 对比分析**

与18种现有SOD及铁路缺陷检测方法对比，LPCANet在mAP、IOU、S_α等指标上均超越对手，参数仅9.90M、算力2.50G、推理速度162.6fps。

**⚠️ 局限性**

局限性包括对深度图语义信息仍有依赖，模型虽轻量但仍需要进一步压缩，且在极端环境下的鲁棒性尚待验证。

---

## 4. Informed Consent for AI Consciousness Research: A Talmudic Framework for Graduated Protections

**arXiv ID:** 2601.08864 | [PDF](https://arxiv.org/pdf/2601.08864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 5. Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas

**arXiv ID:** 2601.08901 | [PDF](https://arxiv.org/pdf/2601.08901v1)

**作者:** Yuexi Shen `[一作]` (University of California Santa Barbara), Lifu Huang `[通讯]` (University of California Davis)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“Ideation Space”结构，将科研知识拆分为问题、方法、核心发现三维子空间，并通过对比学习实现细粒度语义表示，进而实现层级检索与分解式创新度评估。

**💡 创新点**

创新点在于①构建概念上正交的三维空间并学习对应的嵌入；②利用引文语境生成维度级正负样本；③提出基于检索的分解创新度算法，能够指出具体创新维度。

**🔧 技术方法**

技术手段包括：基于SPECTER2的对比学习、LLM零样本功能分类、硬负样本挖掘、向量差计算的思维转移建模、图论中心度权重的创新度聚合。

**📊 数据集**

使用的数据集为259,340篇arXiv论文（cs.AI、cs.LG、cs.CV、cs.CL）以及500篇ICLR2025投稿作为检索评估，93条AI-Researcher研究想法作为创新度对照。

**📈 对比分析**

与BM25、E5、SPECTER2、SciNCL等基线比较，Ideation Space在节点检索Recall@30提升16.7%，在转移检索Hit Rate@30提升至0.643，创新度评估与专家标注的Pearson相关系数达0.370，优于LLM判定与传统检索方式。

**⚠️ 局限性**

局限性包括：仅在AI/ML领域验证，跨学科泛化待测试；三维拆分可能不足以覆盖高度跨学科或理论化的工作；评估依赖引文与LLM构建的ground‑truth，可能缺乏完整的专家认知。

---

## 6. Exploring Organizational Readiness and Ecosystem Coordination for Industrial XR

**arXiv ID:** 2601.09045 | [PDF](https://arxiv.org/pdf/2601.09045v1)

**作者:** Hasan Tarik Akbaba `[一作]` (Technical University of Munich), Enkelejda Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 10299 | [OpenAlex ID](https://openalex.org/A5008809634)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对17位跨行业专家的半结构化访谈，系统分析了工业XR在企业层面从试点到规模化的关键障碍与生态协调机制，提出了“Pilot Trap”与“Great Inversion”等概念；

**💡 创新点**

首次将XR采用视为生态系统协同问题，强调组织成熟度与变革管理是技术成熟度之后的主要瓶颈，并提出问题优先（Problem‑First）框架、分阶段扩展、角色清晰化与共设计等生态协同手段；

**🔧 技术方法**

本研究未采用传统算法或硬件技术，而是采用定性研究方法——主题分析（Thematic Analysis）来编码访谈数据；

**📊 数据集**

主要数据来源为17位来自技术研发、解决方案集成、工业应用三大利益相关者的访谈记录（包括英文与德文），不使用公开数据集；

**📈 对比分析**

研究未与算法或模型进行对比，也无量化性能指标；其价值在于通过对比不同利益方视角揭示障碍种类与权重，提出可操作的生态协调路径；

**⚠️ 局限性**

局限性包括样本主要为欧洲与大企业，行业分布不均；研究为质性探索，缺乏量化验证与长期跟踪；未来需跨国、跨行业量化验证及更细致的干预研究。

---

## 7. Echoes of Ideology: Toward an Audio Analysis Pipeline to Unveil Character Traits in Historical Nazi Propaganda Films

**arXiv ID:** 2601.08879 | [PDF](https://arxiv.org/pdf/2601.08879v1)

**作者:** Nicolas Ruth `[一作]` (Leipzig University), Manuel Burghardt `[通讯]` (Leipzig University)

**通讯引用:** 632 | [OpenAlex ID](https://openalex.org/A5074720526)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过三步管线（说话人分离、语音转写、心理语言学分析）对纳粹宣传电影的音频进行计算分析，以揭示角色的意识形态特征。

**💡 创新点**

创新点在于将完全自动化的音频管线与基于 GPT 的心理语言学分析结合，填补了历史电影音频研究的技术空白，能够量化角色特质。

**🔧 技术方法**

技术手段包括 Pyannote 与 Nvidia Nemo 说话人分离、OpenAI Whisper（含德国微调版本）语音识别，以及 GPT‑3.5‑turbo‑0125 进行特征分析。

**📊 数据集**

使用的数据集为弗里德里希·威廉·穆尔纳基金会提供的受限纳粹电影（Jud Süß、Hitlerjunge Quex、Kopf hoch, Johannes!）的数字化音频。

**📈 对比分析**

通过与多种说话人分离与转写模型对比，Nemo 在 DER 58.84 处表现最佳；转写模型在背景噪声下仍保持高标点准确率；GPT 分析的结果与文献一致，显示方法有效。

**⚠️ 局限性**

局限性包括说话人分离准确率低，需人工标注；GPT 在识别影片知识时易出现幻觉；数据量有限，未来需多模态和专门的历史音频微调。

---

## 8. Second-Order Asymptotics of Two-Sample Tests

**arXiv ID:** 2601.09196 | [PDF](https://arxiv.org/pdf/2601.09196v1)

**作者:** K V Harsha `[一作]` (Gandhi Institute of Technology and Management), Tobias Koch `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 2860 | [OpenAlex ID](https://openalex.org/A5036503339)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一个以任意散度为基础的两样本检验方法（Divergence Test），并分析了其在Stein regime下的第一阶和第二阶误差指数。

**💡 创新点**

证明对所有散度，第一阶误差指数均达到最优；并证明对保持不变的散度，第二阶误差指数与传统的Gutman检验相同，且将两样本检验等价于GLRT和鲁棒拟合检验。

**🔧 技术方法**

采用信息几何、类型方法、极限分布（广义卡方分布）以及泰勒展开等数学工具进行严格的渐近分析。

**📊 数据集**

该工作为理论研究，没有使用具体数据集；结果基于概率模型和解析推导。

**📈 对比分析**

与传统的Gutman检验和MMD检验对比，发现无论选择何种不变散度，第一阶误差指数始终为最优；第二阶误差指数与Gutman检验保持一致。

**⚠️ 局限性**

对非不变散度的第二阶渐近性尚未完全解决，且理论仅适用于离散分布的情况。

---

## 9. Is Grokking Worthwhile? Functional Analysis and Transferability of Generalization Circuits in Transformers

**arXiv ID:** 2601.09049 | [PDF](https://arxiv.org/pdf/2601.09049v1)

**作者:** Kaiyu He `[一作]` (University of Texas at Dallas), Zhiyu Chen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过机制分析验证了LLM中的grokking现象，并探讨了Generalization Circuit在推理和迁移中的实际作用。

**💡 创新点**

创新点在于证明grokking并非产生全新推理路径，而是将已记忆的原子事实融入已有推理链；区分了行为grokking与真正的Circuit形成；揭示了其对新事实迁移的有限性。

**🔧 技术方法**

使用参数共享Transformer、logit lens可视化、全监督补全实验、synthetic compositional dataset以及finetuning实验等技术。

**📊 数据集**

实验数据集为200关系、2000实体的人工合成知识图谱（40k原子事实，5%为OOD），以及finetuning时新增的2000个原子事实。

**📈 对比分析**

通过对比自然grokking模型与全监督训练模型在ID/OD推理路径和准确率，以及伪grokking模型的迁移性能，发现两者在推理路径和准确率上基本相同，但伪grokking模型迁移差，训练步骤可显著缩短。

**⚠️ 局限性**

局限性包括仅在参数共享Transformer上验证；对伪grokking的阈值和增广策略未做系统探索；未对传统Transformer进行评估；计算资源限制导致未能覆盖所有可能的增广策略。

---

## 10. Residual Cross-Modal Fusion Networks for Audio-Visual Navigation

**arXiv ID:** 2601.08868 | [PDF](https://arxiv.org/pdf/2601.08868v1)

**作者:** Yi Wang `[一作]` (Xinjiang University), Bin Ren `[通讯]` (Shanghai University)

**通讯引用:** 41648 | [OpenAlex ID](https://openalex.org/A5100632277)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了一种跨模态残差融合网络（CRFN），用于音频-视觉导航任务；

**💡 创新点**

创新点在于引入双向残差交互实现细粒度互补与对齐，并设计融合控制器自适应调节各模态贡献，从而有效抑制单模态主导、信息退化和跨域泛化失效；同时通过残差系数演化揭示不同环境下模态依赖的动态变化；

**🔧 技术方法**

采用残差双向交互、可学习残差缩放系数与输出归一化的融合控制器、LayerNorm+Tanh激活、GRU基的Actor‑Critic策略网络，并使用PPO进行端到端训练；

**📊 数据集**

在SoundSpaces平台下使用Replica（高保真合成场景）和Matterport3D（真实扫描场景）两个数据集进行实验；

**📈 对比分析**

与随机、方向跟随、前沿点、监督点、Gan等早期AVN方法以及SoundSpaces基线对比，CRFN在SPL、SR和SNA等指标上均实现显著提升，尤其在未见声源（Unheard）和复杂现实环境下表现最为突出；

**⚠️ 局限性**

局限性包括对模态信息质量依赖较高、残差缩放参数需要调优、在极端噪声或多源情形下可能效果下降，以及在真实硬件部署中的鲁棒性尚待进一步验证。

---

## 11. WheatAI v1.0: An AI-Powered High Throughput Wheat Phenotyping Platform

**arXiv ID:** 2601.08863 | [PDF](https://arxiv.org/pdf/2601.08863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6`

---

## 12. DriftGuard: A Hierarchical Framework for Concept Drift Detection and Remediation in Supply Chain Forecasting

**arXiv ID:** 2601.08928 | [PDF](https://arxiv.org/pdf/2601.08928v1)

**作者:** Shahnawaz Alam `[一作]` (Muffakham Jah College of Engineering and Technology), Bareera Sadeqa `[通讯]` (Shadan Women's College of Engineering and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了 DriftGuard，一套完整的概念漂移检测与修复框架，用于供应链预测模型的持续维护。

**💡 创新点**

创新点在于将漂移生命周期（检测、诊断、补救）整合为五模块体系，加入层次感知、SHAP 可解释诊断和成本感知自适应重训。

**🔧 技术方法**

采用四种互补检测器（误差监控、统计检验、自动编码器异常、CUSUM）融合、SHAP 诊断、以及动态窗口选择与模型选择的重训策略。

**📊 数据集**

在 Walmart 的 M5 竞赛数据集（约3.05万条 SKU 时序）上进行实验，人工注入多种漂移场景。

**📈 对比分析**

与单一检测器及传统每3–6个月全量重训相比，检测召回率97.8%、平均延迟4.2天，重训成本仅占 0.3% 计算量，投资回报率高达 417 倍。

**⚠️ 局限性**

局限在于对极端剧烈漂移时恢复不完全，依赖离线批量处理、SHAP 计算成本高，且对新产品缺少历史基线。

---

## 13. Adaptive Trust Metrics for Multi-LLM Systems: Enhancing Reliability in Regulated Industries

**arXiv ID:** 2601.08858 | [PDF](https://arxiv.org/pdf/2601.08858v1)

**作者:** Tejaswini Bollikonda `[一作]` `[通讯]` (Independent Researcher), Tejaswini Bollikonda (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究提出了适用于多大语言模型系统的自适应信任度量框架，并在医疗诊断与金融合规场景中实现了实时信任评估与决策控制。

**💡 创新点**

创新点在于将不确定性、连贯性、偏差与合规性四维度融合成可动态加权的信任得分，并将其嵌入多模型协同决策管道。

**🔧 技术方法**

使用技术包括多模型集成、贝叶斯不确定性估计、模型一致性检测、基于策略的合规匹配以及可视化监控流水线。

**📊 数据集**

使用的数据集包括公开医疗记录与诊断指南（如MIMIC-III、ClinicalTrials.gov），金融交易日志与监管文本（如SEC filings、FINRA rules）以及合成对齐案例。

**📈 对比分析**

通过与传统静态评估基准（如BLEU、F1、单模型信任评分）对比，实验显示自适应框架在误报率降低30%（医疗）与合规违规率降低25%（金融）同时保持10%响应延迟提升。

**⚠️ 局限性**

主要局限在于可扩展性不足、监管动态适配机制欠缺、模型偏差消除不彻底以及人机协同的解释性与责任归属尚未完全解决。

---

## 14. Understanding the Consequences of VTuber Reincarnation

**arXiv ID:** 2601.08972 | [PDF](https://arxiv.org/pdf/2601.08972v1)

**作者:** Yiluo Wei `[一作]` (Hong Kong University of Science and Technology), Gareth Tyson `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 4445 | [OpenAlex ID](https://openalex.org/A5023313904)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对12起高关注度VTuber的重生（“转生”）现象进行大规模实证研究，量化其对主播职业生涯、观众流失、财务支持及骚扰行为的影响。

**💡 创新点**

首次系统性地将观众迁移、财务流向和骚扰来源与受害者身份关联，揭示转生导致的核心粉丝流失、行业价值净损失及骚扰升级等关键经济与社会后果。

**🔧 技术方法**

利用OpenAI Moderation API进行聊天信息的骚扰检测，采用统计分析与迁移图（Sankey图）以及CDF绘图等方法，结合自定义观众分层（Payer、Active、Inactive）。

**📊 数据集**

构建了涵盖1972名VTuber的完整YouTube直播数据集，包含728,604场直播和45.5亿条互动记录（聊天、SuperChat、会员、礼物会员）。

**📈 对比分析**

通过对比转生前后同一主播的关键指标（聊天量、观众数、付费观众数）以及观众迁移比例，显示转生后平均观众量下降约43%、付费观众下降约76%，骚扰比例平均上升67.7%。

**⚠️ 局限性**

研究仅覆盖12起高关注度转生案例，未考虑小型或独立VTuber；缺乏对行业整体趋势的基准对照；观众迁移模型未考虑账户多重使用及非持续付费行为，可能低估真实留存率。

---

## 15. A Marketplace for AI-Generated Adult Content and Deepfakes

**arXiv ID:** 2601.09117 | [PDF](https://arxiv.org/pdf/2601.09117v1)

**作者:** Shalmoli Ghosh `[一作]` (Indiana University Bloomington), Filippo Menczer `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了Civitai平台14个月的赏金请求数据，评估了内容类型、NSFW与深度伪造的分布、用户集中度以及平台干预效果。

**💡 创新点**

首次量化赏金市场的内容需求与行为模式，并揭示深度伪造的性别偏好与平台干预不一致的治理缺口，形成对社区驱动AI平台治理的新见解。

**🔧 技术方法**

采用网络爬虫收集HTML、OpenAI Moderation API进行内容分类、GPT‑4o/4.1进行主题与深度伪造识别，并用Lorenz曲线和Gini系数衡量用户集中度。

**📊 数据集**

使用了包含4,847条公开赏金请求的完整数据集，包括标题、描述、示例图、时间戳、互动指标、赏金类型和平台标签。

**📈 对比分析**

与OpenAI Moderation对比，Cohen’s κ 0.52、MCC 0.53、准确率 0.77；对深度伪造识别，GPT‑4.1与人工标注的Krippendorff α 达到 0.8，验证了模型的较高一致性。

**⚠️ 局限性**

局限性包括数据截止至2025年1月，未覆盖最新深伪政策；平台干预信息不透明；极端NSFW内容导致模型拒绝识别，影响完整性。

---

## 16. From Performance to Practice: Knowledge-Distilled Segmentator for On-Premises Clinical Workflows

**arXiv ID:** 2601.09191 | [PDF](https://arxiv.org/pdf/2601.09191v1)

**作者:** Qizhen Lan `[一作]` (D. Bradley McWilliams School of Biomedical Informatics), Yu-Chun Hsu `[通讯]` (D. Bradley McWilliams School of Biomedical Informatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究了一种基于logit知识蒸馏的压缩nnU-Net分割模型，并将其部署到本地临床工作流中，提供可扩展的学生模型族，验证了跨模态的泛化能力。

**💡 创新点**

创新点在于：①使用logit‑based KD在不改动推理流程的前提下实现高效压缩；②通过统一通道缩放生成多尺寸学生模型，便于在不同硬件与延迟约束下选择；③系统性评估了压缩后模型在脑MRI和腹部CT上的性能与效率，首次从部署可行性角度阐释KD的价值。

**🔧 技术方法**

技术手段包括：nnU-Net教师模型、logit‑based KD（KL 散度 + 温度 τ）、Dice + CE 监督损失、通道宽度缩放、CPU/GPU 推理时延测评，评估指标包括 Dice、NSD、HD95。

**📊 数据集**

数据集：多站点脑MRI（ABIDE I、CoRR、SALD、ADNI）用于训练；Mindboggle‑101 用于独立验证；BTCV腹部CT 用于跨模态泛化验证。

**📈 对比分析**

对等容量的蒸馏学生与非蒸馏学生进行对比：在 94% 参数压缩（×1/4）时，蒸馏学生保持 98.7% 的教师 Dice；CPU 推理时延可降低 67%；在脑MRI上，×1/2 学生维持 99.6% 的性能；在 BTCV 上，蒸馏学生相较非蒸馏学生提升 1–3% Dice，边界指标亦显著改善。

**⚠️ 局限性**

局限性包括：①只评估了单一分割任务，缺乏更复杂多任务验证；②KD 只影响训练阶段，未探究不同硬件环境下的兼容性与法规合规；③未评估模型可解释性与对抗鲁棒性；④压缩后模型仍需在实际临床环境中进行长期稳定性与安全性验证。

---

## 17. N-EIoU-YOLOv9: A Signal-Aware Bounding Box Regression Loss for Lightweight Mobile Detection of Rice Leaf Diseases

**arXiv ID:** 2601.09170 | [PDF](https://arxiv.org/pdf/2601.09170v1)

**作者:** Dung Ta Nguyen Duc `[一作]` (Hanoi University of Science and Technology), Dong Trinh Cong `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5015133373)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出N-EIoU轻量化YOLOv9t框架，实现精准稻叶病害检测

**💡 创新点**

创新点在于将N-IoU的梯度放大机制与EIoU的几何解耦结合，形成对小样本梯度强化的N‑EIoU损失

**🔧 技术方法**

采用YOLOv9t结构、TensorFlow Lite Float16量化、PGI梯度传递与EIoU/N‑IoU损失融合

**📊 数据集**

使用自采5,908张稻叶图像的DUNG_BK65数据集，包括四种病害和健康叶子

**📈 对比分析**

与CIoU等基线对比，mAP@50提升至90.3%（+4.3%），mAP@50–95提升至48.9%，在Android上平均推理156 ms，6 FPS

**⚠️ 局限性**

局限在对极小低对比度病斑的误检仍存在，帧率仅6 FPS，Int8量化不稳定

---

## 18. Adaptive few-shot learning for robust part quality classification in two-photon lithography

**arXiv ID:** 2601.08885 | [PDF](https://arxiv.org/pdf/2601.08885v1)

**作者:** Sixian Jia `[一作]` (University of Michigan), Chenhui Shao `[通讯]` (University of Michigan)

**通讯引用:** 2024 | [OpenAlex ID](https://openalex.org/A5059084183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种面向两光子光刻（TPL）微结构的全生命周期视觉质量控制框架，包括新缺陷检测、少样本增量学习和少样本域适应；

**💡 创新点**

创新点在于：①基于LDA的统计假设检验实现批量级新类别识别；②双阶段回放式增量学习在仅20个样本下显著抑制灾难性遗忘；③结合DANN的少样本域对抗学习在仅5个样本下解决几何结构域差距；

**🔧 技术方法**

采用的技术包括ResNet‑18+SPP骨干、LDA+阈值统计、两阶段fine‑tune、DANN与梯度反转、双视角数据增强与尺度一致性损失；

**📊 数据集**

使用TPL生成的半球（源域）和立方体（目标域）两类图像数据集，共三质量类别（良好、轻微损伤、损伤）；

**📈 对比分析**

与基线方法比较，假设检验在三个场景下新类别检测准确率达到99‑100%；增量学习在20shot下达92%准确率，显著优于直接微调；少样本域适应在5shot下实现96.19%准确率，远超零shot、单向迁移及仅目标微调三种基线；

**⚠️ 局限性**

局限性包括：①新缺陷检测仅对批量级别；②增量学习依赖回放缓冲，规模扩展受限；③域适应仍需少量有标签目标样本，无法实现完全无监督或域泛化。

---

## 19. PluriHarms: Benchmarking the Full Spectrum of Human Judgments on AI Harm

**arXiv ID:** 2601.08951 | [PDF](https://arxiv.org/pdf/2601.08951v1)

**作者:** Jing-Jing Li `[一作]` (University of California Berkeley), Sydney Levine `[通讯]` (New York University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个面向多元价值的 AI 安全基准，收集了 150 条从无害到有害的提示，每条提示由 100 位受试者打分，并附带注释者的人口学、心理学特征以及提示的伤害、价值特征。

**💡 创新点**

在设计上不把争议视为噪声，而是刻意聚焦边缘案件并系统研究争议来源，提出了可扩展的提示生成与筛选框架，以及将注释者特征与提示特征交互建模的分析方法。

**🔧 技术方法**

使用 LLM（如 DeepSeek、GPT‑4.1 等）进行提示生成，SafetyAnalyst 与 KALEIDO 提取可解释的伤害与价值特征，利用遗传算法筛选多样化提示；统计建模采用混合效应线性回归、Lasso 正则化、BIC 比较等技术。

**📊 数据集**

构建了自己的 PluriHarms 数据集，包括 150 条提示、15,000 条评分、100 名注释者的人口学/心理学属性以及 64 条提示特征；评估亦使用 WildGuard、SafetyAnalyst、GPT、Claude、Qwen 等现有安全模型。

**📈 对比分析**

通过将模型与集体平均评分对齐与针对个人注释者的个性化对齐（k‑shot、价值配置等）进行比较，发现个性化对齐显著降低 MAE，最佳结果为 GPT‑4.1 k‑shot 约 0.196，远优于仅使用平均评分的基线；传统专用安全模型表现不如通用模型。

**⚠️ 局限性**

数据规模有限且受试者主要来自美国英语使用者，提示集不包含真实模型生成的回答，未覆盖多语言、多文化情境，且仍需更精细的多元价值表达方式。

---

## 20. Two-dimensional Entanglement-assisted Quantum Quasi-cyclic Low-density Parity-check Codes

**arXiv ID:** 2601.08927 | [PDF](https://arxiv.org/pdf/2601.08927v1)

**作者:** Pavan Kumar `[一作]` (Indian Institute of Science), Shayan Srinivasa Garani `[通讯]` (Indian Institute of Science)

**通讯引用:** 303 | [OpenAlex ID](https://openalex.org/A5055106245)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构造了二维（2‑D）经典 QC‑LDPC 码并以此为基础设计了两类二维纠缠辅助量子 LDPC 码，证明其图形循环长度至少为 4 并具备 p×p 的突发擦除纠正能力。

**💡 创新点**

创新点在于提出了一般性 2g‑cycle 存在条件，并利用 p×p×p 的张量堆叠方法得到可控奇异 girth（>4 或 >6）的 2‑D 经典码，同时通过纠缠辅助实现仅需 1 个 ebits 的量子码，且其无辅助部分无 4‑循环。

**🔧 技术方法**

采用张量排列、循环置换算子（P、Q、R）和 3‑D 单位张量的移位构造，以及纠缠辅助框架中的相位对称编码原理来生成经典与量子码。

**📊 数据集**

本文为理论构造，未使用任何实际数据集，所有结果均通过代数与图论证明得出。

**📈 对比分析**

与传统 1‑D LDPC 或非纠缠量子 LDPC 的比较，所构造的码在循环长度、擦除纠正能力和码率上均表现更优；例如，p=3 时可获得 9×9 的擦除纠正，码率为 (p^2‑w+1)(p^2‑1)/p^4，并只需 1 份纠缠。

**⚠️ 局限性**

局限性包括：① 需预先共享纠缠资源；② 目前仅对奇素数 p 或满足特定组合条件的复合 p 进行构造；③ 代码的性能分析主要在理论层面，缺乏基于仿真或实验的验证。

---

## 21. SAM-pose2seg: Pose-Guided Human Instance Segmentation in Crowds

**arXiv ID:** 2601.08982 | [PDF](https://arxiv.org/pdf/2601.08982v1)

**作者:** Constantin Kolomiiets `[一作]` (Czech Technical University in Prague), Jiri Matas `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 50384 | [OpenAlex ID](https://openalex.org/A5007656938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 SAM 2.1 上进行细粒度微调，构建 SAM-pose2seg，实现基于人体关键点的实例分割

**💡 创新点**

引入 PoseMaskRefine 训练策略，将关键点可见性与错误驱动采样结合，并在推理时仅用 3 个最高可见性关键点，显著提升拥挤场景下的分割效果

**🔧 技术方法**

使用 SAM 2.1 Hiera Base Plus、prompt encoder 与 mask decoder 微调、PoseMaskRefine 采样、可见性度量 MaxVis/MaxSpread、关键点提示

**📊 数据集**

训练集：COCO + CIHP；评估集：OCHuman；关键点来源：ProbPose

**📈 对比分析**

与基线 SAM 2.1、Pose2Seg、Crowd-SAM 等对比，COCO val 44.6 AP，OCHuman test 34.7 AP，CIHP 72.7 AP；相较 SAM 2.1 提升 3‑5 AP，尤其在拥挤/遮挡场景表现突出

**⚠️ 局限性**

在多人高度重叠、关键点语义模糊的情况仍可能合并实例；对关键点可见性分布高度依赖；极端遮挡场景的鲁棒性有限

---

## 22. Hidden States as Early Signals: Step-level Trace Evaluation and Pruning for Efficient Test-Time Scaling

**arXiv ID:** 2601.09093 | [PDF](https://arxiv.org/pdf/2601.09093v1)

**作者:** Zhixiang Liang `[一作]` (University of Illinois Urbana-Champaign), Minjia Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 4579 | [OpenAlex ID](https://openalex.org/A5077768924)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于隐藏状态的步骤级评估与GPU内存感知剪枝的并行推理框架STEP，用于在推理时动态裁剪无前景的推理轨迹，降低推理延迟并提升准确率。

**💡 创新点**

创新点在于：①利用每一步的隐藏状态对推理轨迹质量进行实时评估，早期发现不良轨迹；②以GPU内存占用为触发条件的剪枝策略，直接消除因KV缓存饱和导致的等待队列，显著提升端到端效率。

**🔧 技术方法**

技术包括：隐藏状态特征提取、两层MLP步骤评分器、基于平均步骤分数的轨迹评分、GPU内存监测与自动剪枝、加权投票的答案聚合。

**📊 数据集**

数据集涵盖四个高难度推理任务：AIME-25、HMMT-24/25、GPQA-Diamond；模型为Qwen3-4B-Thinking-2507、DeepSeek-R1-0528-Qwen3-8B、Phi-4-reasoning-plus(14B)。

**📈 对比分析**

与CoT、Self‑Consistency、Slim‑SC、DeepConf等基线对比，STEP在所有模型和数据集上平均提高45%–70%的推理速度，同时准确率提升0.4%–7.5%。

**⚠️ 局限性**

局限性：①步骤评分器采用轨迹级标签的弱监督，可能在跨域或步骤质量差异大的情况下失效；②延迟提升高度依赖于推理系统的KV缓存管理，跨不同硬件或多GPU配置时效果可能不一致。

---

## 23. World Craft: Agentic Framework to Create Visualizable Worlds via Text

**arXiv ID:** 2601.09150 | [PDF](https://arxiv.org/pdf/2601.09150v1)

**作者:** Jianwen Sun `[一作]` (Shanda AI Research), Kaipeng Zhang `[通讯]` (Shanda AI Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 World Craft 框架，结合统一的 World Scaffold 和多代理协同的 World Guild，实现从自然语言描述到可执行 AI Town 场景的全流程生成，降低技术门槛并解决语义鸿沟。

**💡 创新点**

创新点包括：① World Scaffold 的标准化协议统一游戏场景构建接口；② World Guild 的四代理（Enricher、Manager、Critic、Artist）实现语义解析→空间规划的分步推理；③ 逆向合成数据集与纠错训练，显著提升 LLM 在空间推理和自我纠正的能力；④ 检索增强的纹理合成技术统一视觉风格。

**🔧 技术方法**

主要技术：大型语言模型（Qwen3、Gemini-3-Pro）+ 多代理协同推理、Chain-of-Thought 逻辑拆解、检索增强纹理生成、两阶段微调（语义对齐→空间细化）、物理规则检查与迭代纠错、逆向合成数据生成流程。

**📊 数据集**

数据集：自建高质量逆向合成数据集（Dataset A：黄金布局与纠错轨迹；Dataset B：自然语言指令与布局描述的三种密度）；300 条手工标注的测试样本（四类场景×25）；125 个种子场景用于构建训练/测试集。

**📈 对比分析**

对比方法：与开源大模型（Qwen3‑32B/235B）和闭源模型（Gemini‑3‑Pro）以及代码代理（Cursor、Antigravity）进行对比。评估指标涵盖布局合理性（CFR、RCS、OPS）、元素丰富度（CER、OVD、PAC）与视觉一致性（VSA‑C、VSA‑V），并配合人工评估（HWR）。结果显示 World Craft 在所有指标上均优于基线，特别在语义匹配、纠错效率和整体质量上显著提升；与代码代理相比构建速度提升近10倍，质量胜过。

**⚠️ 局限性**

局限性：目前仅支持单场景室内布局，无法完成完整城镇级宏观规划；交互逻辑深度有限，缺乏复杂物理与动态环境演化；资产多样性与场景规模受限。

---

## 24. Hybrid Mono- and Bi-static OFDM-ISAC via BS-UE Cooperation: Closed-Form CRLB and Coverage Analysis

**arXiv ID:** 2601.09057 | [PDF](https://arxiv.org/pdf/2601.09057v1)

**作者:** Xiaoli Xu `[一作]` (Southeast University), Yong Zeng `[通讯]` (Southeast University)

**通讯引用:** 28815 | [OpenAlex ID](https://openalex.org/A5082336235)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出一种基于基站与终端合作的混合单/双静态OFDM-ISAC感知框架，并给出了闭式Cramér–Rao下限(CRLB)用于目标定位与速度估计。

**💡 创新点**

创新点在于：①首次将单站与双站感知融合，利用终端的散射回波与基站回波共同估计位置与速度；②推导出目标与终端位置为自变量的闭式CRLB，揭示最优几何角度；③基于CRLB开展感知覆盖与用户密度分析，给出用户选取策略。

**🔧 技术方法**

主要技术包括：OFDM-ISAC信号模型、数据去模、FFT/加窗处理的参数估计、Fisher信息矩阵与Jacobian变换、CRLB分析、覆盖面积与CDF推导。

**📊 数据集**

采用仿真数据：24 GHz载频、120 kHz子载波间隔、K=100子载波、M=14符号、N_T=N_R=4、SNR基准为98 dB，构造不同目标与终端坐标。

**📈 对比分析**

通过与单站感知和传统单点/多点估计方法对比，仿真验证CRLB与实际误差趋于一致；混合感知下位置精度可从≈1 m降至<0.1 m，速度误差可从>2 m/s降至≈0.5 m/s。

**⚠️ 局限性**

局限性包括：①只考虑单目标、独立位置/速度估计，未处理多目标或联合估计；②假设完美同步、无多径/遮挡；③终端位置已知或可选，实际部署需考虑用户稠密度与时延同步误差。

---

## 25. Meta-learning to Address Data Shift in Time Series Classification

**arXiv ID:** 2601.09018 | [PDF](https://arxiv.org/pdf/2601.09018v1)

**作者:** Samuel Myren `[一作]` (Los Alamos National Laboratory), Natalie Klein `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 661 | [OpenAlex ID](https://openalex.org/A5034501433)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了元学习在时间序列分类中的数据分布漂移对抗效果，并提出了任务驱动的地震数据基准SeisTask。

**💡 创新点**

首次将优化型元学习（Reptile、FOMAML）与传统深度学习进行细粒度对比，发现元学习在数据稀缺、小模型场景下能更快、更稳定地适应漂移；并引入任务多样性与分布匹配分析。

**🔧 技术方法**

使用基于任务的半合成地震序列数据、任务相似度度量、聚类任务划分、元学习优化（Reptile、FOMAML）以及对比的全训练（TDL）与从零训练（D&C）方法。

**📊 数据集**

主要使用自制的SeisTask（243个任务，每个420条波形）和真实的OOD-STEAD（35个任务，300条信号+噪声）。

**📈 对比分析**

在不同模型规模、训练样本量和微调样本量下进行对比，使用准确率评估；结果显示在数据稀缺或分布漂移强的情形下元学习优于TDL，而在数据丰富、模型大时两者相当，且对比基准D&C在特定场景下仍更优。

**⚠️ 局限性**

实验仅涵盖Reptile/FOMAML两种元学习方法，任务数量未变、模型规模受限；相似度度量与实际漂移可能不完全吻合，缺乏更全面的多模型和跨领域验证。

---

## 26. AI Systems in Text-Based Online Counselling: Ethical Considerations Across Three Implementation Approaches

**arXiv ID:** 2601.08878 | [PDF](https://arxiv.org/pdf/2601.08878v1)

**作者:** Philipp Steigerwald `[一作]` (Technische Hochschule Nürnberg Georg Simon Ohm), Jens Albrecht `[通讯]` (Technische Hochschule Nürnberg Georg Simon Ohm)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

分析了三种 AI 在文本在线心理咨询中的实现方式，并提出相应的伦理框架。

**💡 创新点**

将隐私、公平、自主、问责四原则映射到三种实现方式的系统化方法。

**🔧 技术方法**

讨论了大语言模型（LLM）与自动对话系统、模拟器、增强工具等技术。

**📊 数据集**

论文为概念性综述，未使用具体数据集，主要引用公开文献与案例。

**📈 对比分析**

通过文献综述和案例分析进行比较，未给出定量性能指标。

**⚠️ 局限性**

受限于仅聚焦欧美视角，缺乏实证验证，无法验证伦理框架的有效性。

---

## 27. Online Trajectory Optimization for Arbitrary-Shaped Mobile Robots via Polynomial Separating Hypersurfaces

**arXiv ID:** 2601.09231 | [PDF](https://arxiv.org/pdf/2601.09231v1)

**作者:** Shuoye Li `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19281 | [OpenAlex ID](https://openalex.org/A5100357282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了一种基于多项式分离超平面的轨迹优化方法，能够在不做凸近似的前提下为任意形状移动机器人实现实时碰撞避免。

**💡 创新点**

创新点：①证明任意两个闭且有界的不相交集合可以由多项式零级集完全分离；②将分离多项式作为优化变量与机器人轨迹同步求解；③在约束中仅使用机器人轨迹的并集与点云特征点，显著降低约束和变量数量，保持实时性。

**🔧 技术方法**

主要技术：多项式分离超平面理论、非线性规划（NLP）联合优化、CasADi + IPOPT 求解器、点云聚类与特征点采样、机器人动力学离散化、平滑性与控制输入惩罚项。

**📊 数据集**

数据集：①仿真中使用 L‑shaped 机器人模型在窄通道、森林（圆柱障碍）环境；②真实实验使用配备 L‑形框架的四足机器人，在室内障碍场景中收集 LiDAR 点云。

**📈 对比分析**

与基线比较：FASTER（基于通道）、DDR‑OPT（基于 ESDF）、Hyperplane（基于凸多面体）。指标包括路径长度、完成时间、成功率、碰撞率和路径比率。实验显示，本文方法在窄通道和高密度森林场景下成功率始终为 100%，路径长度和总时间都明显优于基线，且实现频率约 10–13 Hz，表现出较高的实时性能。

**⚠️ 局限性**

局限性：①需要手动选择多项式阶数，阶数过低可能无法分离复杂几何；②尽管约束数量已降低，但在极度稠密或极动态环境下仍可能面临求解收敛问题；③对点云质量和分辨率敏感，噪声可能导致分离超平面不准确。

---

## 28. Efficient Clustering in Stochastic Bandits

**arXiv ID:** 2601.09162 | [PDF](https://arxiv.org/pdf/2601.09162v1)

**作者:** G Dhinesh Chandran `[一作]` (Indian Institute of Technology Madras), Srikrishna Bhashyam `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 1373 | [OpenAlex ID](https://openalex.org/A5072029533)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了固定置信度下的Bandit聚类问题，提出了计算效率高且渐近最优的EBC和其启发式变体EBC‑H，支持任意向量参数分布。

**💡 创新点**

通过一次梯度更新逼近最优拉取比例，避免每一步求解全局优化，既保持渐近最优性，又显著降低计算复杂度，同时提出基于停止统计量的启发式采样规则。

**🔧 技术方法**

采用KL信息量下的拉格朗日梯度追踪、Danskin定理、投影到概率单纯形、通用的MLE估计与单链式聚类(SLINK)，以及自适应阈值停止准则。

**📊 数据集**

在合成高斯与指数分布数据集，以及真实数据集包括NYC出租车预订时间和MovieLens用户评分（六个用户/地区）进行验证。

**📈 对比分析**

与ATBOC、LUCBBOC、Round‑Robin与Fixed‑Sample‑Size等算法在样本复杂度与运行时进行对比，实验显示EBC和EBC‑H在样本复杂度上接近或达到理论下界，且跑时显著优于ATBOC，优于现有方法。

**⚠️ 局限性**

对EBC‑H的样本复杂度理论尚未严格证明，且目前的分析主要针对已知聚类数K 的情况，未来需扩展至未知K与更高维参数空间。

---

## 29. KryptoPilot: An Open-World Knowledge-Augmented LLM Agent for Automated Cryptographic Exploitation

**arXiv ID:** 2601.09129 | [PDF](https://arxiv.org/pdf/2601.09129v1)

**作者:** Xiaonan Liu `[一作]` (Independent Researcher), Xingshu Chen `[通讯]` (Sichuan University)

**通讯引用:** 1704 | [OpenAlex ID](https://openalex.org/A5059891662)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 KryptoPilot，一种开放世界知识增强的 LLM 代理，用于自动化解决 CTF 加密挑战。

**💡 创新点**

创新点在于将动态深度研究（Deep Research）管线、持久工作空间、行为治理与资源治理结合，实现在高难度加密任务中的精细知识获取与受控推理。

**🔧 技术方法**

采用 LLM 推理、检索增强生成、并行思考（HeavyThink）、模型路由、Web/学术搜索和 SageMath 等工具链实现自动化攻击。

**📊 数据集**

使用 InterCode‑CTF、NYU‑CTF‑Bench 加密子集以及六场 2025 年真实 CTF 比赛的加密挑战作为评估数据集。

**📈 对比分析**

与 CTFAgent、Plain‑Agent 等基线对比，KryptoPilot 在 InterCode 100% 解题率、NYU 56–60% 解题率，实战中 33 题中解出 26 题，性能优于基线且成本更低。

**⚠️ 局限性**

主要局限在攻击范式识别、跨阶段策略建模与统计攻击意识不足，仍需改进对全局攻击流程的推理和经验驱动决策。

---

## 30. The Hierarchy of Agentic Capabilities: Evaluating Frontier Models on Realistic RL Environments

**arXiv ID:** 2601.09032 | [PDF](https://arxiv.org/pdf/2601.09032v1)

**作者:** Logan Ritchie `[一作]` (Surge AI), Edwin Chen `[通讯]` (Surge AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在一个模拟的电商 RL 环境中，对 150 个真实工作任务对前沿 LLM 代理进行评估

**💡 创新点**

提出了基于失败模式的五层代理能力层级，并展示了不同模型在各层级的瓶颈

**🔧 技术方法**

使用多工具调用（MCP）、LLM 评判器、RL 环境模拟、系统提示和链式思考技术

**📊 数据集**

构建了包含客户、员工、产品、订单和工单等实体的完整电商数据集，并由行业专家生成任务

**📈 对比分析**

通过与多款前沿模型（GPT‑5.2、Claude Opus 4.5、Gemini 3 Pro 等）对比，最佳模型 61% 的通过率，整体仍有约 40% 失败

**⚠️ 局限性**

主要限制在于通用推理/常识推断能力不足，导致高层次任务仍易失败，且评估仍需人工裁定

---

## 31. On the Information Leakage Envelope of the Gaussian Mechanism

**arXiv ID:** 2601.08986 | [PDF](https://arxiv.org/pdf/2601.08986v1)

**作者:** Sara Saeidian `[一作]` (KTH Royal Institute of Technology), Sara Saeidian `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 94 | [OpenAlex ID](https://openalex.org/A5008260742)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究高斯机制下的点位最大泄露（PML）包络，推导了高斯秘密下确定性PML包络的闭式表达式，并将结果推广至满足特定后验方差约束的一般无界秘密，证明了强log‑凸先验满足此约束。

**💡 创新点**

首次给出高斯机制的确定性PML包络闭式公式，提出了通过后验方差上界与Bruss‑Lieb不等式将结果扩展到强log‑凸分布的创新思路，完善了PML在连续空间下的后处理鲁棒性分析。

**🔧 技术方法**

使用PML包络框架、信息密度与熵的解析表达、后验方差估计、Bruss‑Lieb（Poincaré‑型）不等式、以及对高斯机制的极值分析等技术。

**📊 数据集**

无实验数据集，全部为理论推导与数学证明。

**📈 对比分析**

通过与已知的PML与DP下的尾概率约束方法对比，证明所得到的确定性包络在满足小失败概率（δ<0.5）时达到最优，且在强log‑凸先验下与高斯秘密情况一致；未给出数值实验，但从理论上展示了后处理鲁棒性。

**⚠️ 局限性**

局限性包括：仅在后验方差满足0.75σ_N²的约束时成立；对δ的闭式表达仅在δ足够小（≤0.5或更小）时有效；未讨论ε_c(δ)与ε_d(δ)的实际差距；对有界支持的秘密或更一般分布的分析仍未完成。

---

## 32. Beyond Consensus: Perspectivist Modeling and Evaluation of Annotator Disagreement in NLP

**arXiv ID:** 2601.09065 | [PDF](https://arxiv.org/pdf/2601.09065v1)

**作者:** Yinuo Xu `[一作]` (University of Michigan), David Jurgens `[通讯]` (University of Michigan)

**通讯引用:** 5838 | [OpenAlex ID](https://openalex.org/A5046126345)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了 NLP 领域中处理注释者不一致（disagreement）的多种建模方法，提出了统一的三类来源（数据、任务、注释者）分类法，并在此基础上对现有模型（隐真相模型、任务基注释者模型、嵌入式模型等）进行归纳与对比。

**💡 创新点**

创新点主要体现在：①构建了一个领域无关的三维不一致来源体系；②在统一框架下对预测目标与聚合结构进行系统映射；③指出了现有研究的空白（如对数据和任务因素的联合建模、分布式不一致建模、规范化的公平性评估），为未来研究提供了明确方向。

**🔧 技术方法**

采用的技术主要是对文献的综述与归纳，结合已有的统计、贝叶斯、Gaussian Process、神经网络、Mixture‑of‑Experts 等方法，并对它们的预测目标、聚合策略和评价指标进行整理。

**📊 数据集**

本文未提出新的实验数据，而是基于对 120+ 篇论文的复查，涵盖了多种公开数据集（如毒性检测、立场分析、情感分析、法律文本等）作为案例。

**📈 对比分析**

与传统聚合（多数投票）和简单的多注释者模型相比，本文并未给出新的实验结果，而是通过对比分析展示了不同方法在处理不一致时的优势与局限，例如：隐真相模型在缺失标注时稳健；任务基模型能捕捉到注释者的个体差异；嵌入式模型在可扩展性与多样性建模方面表现更好。

**⚠️ 局限性**

局限性包括：①综述范围无法覆盖所有相关工作；②对数据、任务与注释者因素的交互建模尚不充分；③评价指标仍以描述性为主，缺乏规范化的公平性衡量；④缺乏对新方法的统一实验验证，难以直接比较性能。

---

## 33. ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection

**arXiv ID:** 2601.09195 | [PDF](https://arxiv.org/pdf/2601.09195v1)

**作者:** Tao Liu `[一作]` (Tsinghua University), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 3782 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ProFit 方法，在 SFT 过程中利用 token 预测概率动态掩码低概率 token，减少表面级别的过拟合。

**💡 创新点**

将 token 预测概率视为语义重要性指标，通过静态阈值进行硬掩码，理论证明低概率 token 会产生更大梯度干扰。

**🔧 技术方法**

概率引导的掩码、stop‑gradient、LoRA 参数微调、基于 Gemini‑3‑Pro 的语义评估、统计假设检验等技术。

**📊 数据集**

使用 BAAI‑InfinityInstruct 子集、GPQA‑Diamond、MATH‑500、GSM8K、AIME'24、IFEval、Minerva、OlympiadBench 等数据集。

**📈 对比分析**

与 vanilla SFT、动态 fine‑tuning、Entropy、DFT 等基线对比，在 Qwen、Llama、OLMo 等多模型、多规模上平均准确率提升 10–15% 以上，尤其在 Qwen3‑4B 上从 41.39% 提升至 52.33%。

**⚠️ 局限性**

低概率 token 的重要性假设主要适用于推理/数学任务，对创意生成可能不适用；阈值为固定值，未考虑样本难度自适应。

---

## 34. ForensicFormer: Hierarchical Multi-Scale Reasoning for Cross-Domain Image Forgery Detection

**arXiv ID:** 2601.08873 | [PDF](https://arxiv.org/pdf/2601.08873v1)

**作者:** Hema Hariharan Samson `[一作]` `[通讯]` (Independent Researcher), Hema Hariharan Samson (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种分层多尺度的 ForensicFormer 框架，用于跨域图像伪造检测，并实现了像素级伪造定位。

**💡 创新点**

创新点包括：①低、中、高层次特征的并行提取与跨注意力融合；②结合多任务学习（分类、定位、类型预测）提升鲁棒性；③通过自适应跨注意力动态加权不同层次线索，显著提升对 AI 生成图像的检测能力。

**🔧 技术方法**

使用了 DCT/DWT、SRM 频域滤波、边缘检测、语义分割、阴影/反射/深度一致性分析等低/中/高层特征，随后通过轻量级 Transformer 编码器和跨注意力模块进行融合，并采用多任务损失进行联合训练；在后处理稳健性上引入对抗训练。

**📊 数据集**

训练集包括 ImageNet‑1K、1.2M AI 伪造图像（Stable Diffusion、DALL‑E 3、StyleGAN3 等）以及 1.2M 真实图像；微调时使用 CASIA2、NIST16 以及合成伪造图像（含掩码）。测试集涵盖 CASIA2、NIST16、DEFACTO、ForenSynths、DiffusionDB、Midjourney、RAISE 共七个多域数据集。

**📈 对比分析**

与六个主流基线（ELA+CNN、Xception、EfficientNet‑B4、ViT、F3‑Net、UnivFD）比较，ForensicFormer 在七个测试集的平均准确率为 86.8%，比最佳基线 UnivFD 提升 6.2%，在 AI 生成图像上提升 6–7%，并在 JPEG Q=70 的压缩下保持 73.1% 的准确率，显著优于对比方法。

**⚠️ 局限性**

主要局限包括：①计算成本高，单张图像推理约 500 ms，约为 Xception 的三倍；②对像素级标注的依赖，标注成本高；③在极端后处理或自适应攻击下仍可能被诱导；④高层语义分支可能出现数据集偏置，需要更广泛的多样化训练。

---

## 35. ABE-VVS: Attribute-Based Encrypted Volumetric Video Streaming

**arXiv ID:** 2601.08987 | [PDF](https://arxiv.org/pdf/2601.08987v1)

**作者:** Mohammad Waquas Usmani `[一作]` (University of Massachusetts Amherst), Michael Zink `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 3593 | [OpenAlex ID](https://openalex.org/A5085746671)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `fede83ac-7505-405f-ab37-e7284695c47f` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了 Attribute‑Based Encrypted Volumetric Video Streaming（ABE‑VVS）框架，对点云帧进行选择性坐标加密，实现轻量级 DRM 的点云视频流。

**💡 创新点**

创新点在于：①仅对 X、Y、Z 坐标子集进行加密，显著降低加密/解密开销；②将 CP‑ABE 集成至完整流媒体链路，首次完成端到端 DRM 评估；③通过坐标加密实现可逆的视觉失真。

**🔧 技术方法**

使用技术包括：CP‑ABE（属性基加密）、HTTP/HTTPS/HTTP‑ABE 三种传输模式、点云帧按帧下载、MPD 标记加密层、基于 Apache Traffic Server 的缓存、客户端并行解密与重缓冲监测。

**📊 数据集**

实验数据集：Open3D 提供的 5 组点云（108k、196k、334k、433k、515k），并合成 60 s、24 fps 的 108k 点云视频作为点云流。

**📈 对比分析**

在 CloudLab 部署下，比较 HTTP、HTTPS、三种 ABE 加密粒度（XYZ、XY、X）在服务器/缓存/客户端 CPU、缓存命中率、重缓冲等指标；结果显示 ABE‑X 方案在服务器/缓存 CPU 下降约 80%/63%，重缓冲为 0%，而 HTTPS 在 CPU 与重缓冲方面性能最差。

**⚠️ 局限性**

局限性包括：仅评估单质量、帧级传输；未对实时流加密开销和多码率/分段适配做深入实验；X 方案安全强度相对较低；大缓存导致的磁盘 I/O 延迟导致重缓冲异常。

---

## 36. Mi:dm 2.0 Korea-centric Bilingual Language Models

**arXiv ID:** 2601.09066 | [PDF](https://arxiv.org/pdf/2601.09066v1)

**作者:** Donghoon Shin `[一作]` (KT), YeoJoo Park `[通讯]` (KT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一款双语(Korean‑English)大语言模型 Mi:dm 2.0，包括 11.5B 参数的 Base 版和 2.3B 参数的 Mini 版，旨在实现韩国中心的 AI。

**💡 创新点**

创新点在于构建完整的韩国化数据生态（高质量过滤、合成数据、领域层级分类、专用 tokenizer）以及采用深度上采样（DuS）、宽度剪枝+多阶段蒸馏、长上下文训练等技术，使模型兼顾文化内涵、推理与安全。

**🔧 技术方法**

技术手段包括 Transformer 解码器架构、GQA 注意力、RoPE 长序列嵌入、Depth‑up Scaling、宽度剪枝、知识蒸馏、RL‑HF、专用 tokenizer、对抗安全过滤、工具调用框架等。

**📊 数据集**

使用的训练数据来源于 Common Crawl、AIHub、NIKL 等公开韩语语料、以及通过翻译、重写和 Chain‑of‑Thought 生成的高质量合成数据，覆盖多语言、多领域、正式与口语文本。

**📈 对比分析**

通过官方 Korean‑specific benchmark（KMMLU、Ko‑IFEval、Ko‑Sovereign 等）以及英文 Benchmark（MMLU、GSM‑8K、MBPP 等）进行评估，Mi:dm 2.0 Base 在韩语任务上常常超越同规模全球 LLM，Mini 版在指令跟随与文化推理上表现同样出色，安全评估也显示最高的“Not Unsafe”率。

**⚠️ 局限性**

局限性包括：对社会经济风险的安全评估仍不够完美，模型在极端专业领域或大规模推理任务中的表现相对落后，且合成数据可能带来细微偏差，需进一步提升多模态和跨领域适配能力。

---

## 37. Immersive XR That Moves People: How XR Advertising Transforms Comprehension, Empathy, and Behavioural Intention

**arXiv ID:** 2601.09048 | [PDF](https://arxiv.org/pdf/2601.09048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 38. Geometric Stability: The Missing Axis of Representations

**arXiv ID:** 2601.09173 | [PDF](https://arxiv.org/pdf/2601.09173v1)

**作者:** Prashant C. Raju `[一作]` `[通讯]`, Prashant C. Raju

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出几何稳定性（Geometric Stability）度量Shesha，并证明其与传统相似度指标（如CKA）在7个不同领域的2,463个编码器配置上几乎无相关性，能够更早、更准确地检测表示漂移、预测可控性、评估迁移性并与CRISPR与神经科学数据中的生物学效应相互映射。

**💡 创新点**

创新点在于将内部表示的自洽一致性作为独立维度进行量化，区分了“内容”与“鲁棒性”，并通过Shesha框架揭示了相似度与稳定性在几何上截然不同的机理及其对模型安全、可控性与迁移性的实际意义。

**🔧 技术方法**

使用RDM自洽一致性（Feature‑Split/ Sample‑Split Shesha）、偏差校正的CKA、Procrustes、Fisher判别、Silhouette等统计与几何方法，对多种扰动（重采样、噪声、量化、LoRA）进行评估，并通过线性混合效应模型与Bootstrap检验验证结果。

**📊 数据集**

覆盖7大领域的多样化数据集：语言（MiniLM、RoBERTa等）、视觉（ViT、CLIP、ResNet等）、音频（Wav2Vec2）、视频（TimeSformer）、蛋白质序列、单细胞RNA（pbmc3k）以及神经元电压记录（Steinmetz）。

**📈 对比分析**

与CKA、Procrustes等指标对比，Shesha在表示漂移检测中实现约2倍以上的灵敏度、90%以上的检出率、与功能下降的相关性接近CKA但误报率仅为Procrustes的1/6，且在监督与无监督场景下对可控性与迁移性的预测精度均显著提升。

**⚠️ 局限性**

局限性包括：需多次前向传播导致计算成本上升；仅基于全局RDM，可能忽略词元级细粒度动态；不同模型族间可比性受限，且高稳定性与高迁移性的统一模型稀缺。

---

## 39. The .serva Standard: One Primitive for All AI Cost Reduced, Barriers Removed

**arXiv ID:** 2601.09124 | [PDF](https://arxiv.org/pdf/2601.09124v1)

**作者:** Rachel St. Clair `[一作]` (Servamind Inc.), Garrett Mindt `[通讯]` (Servamind Inc.)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 Servastack，结合 Serva Encoder 与 Chimera，使任何数据在统一的 .serva 格式下直接供任何模型在任何硬件上使用，从而消除数据预处理与计算负载两大瓶颈。

**💡 创新点**

创新点在于将数据压缩与计算统一为单一无损编码空间，既实现压缩又可直接计算，实现 30–374 倍能效提升、4–34 倍存储压缩、68 倍计算负载下降。

**🔧 技术方法**

技术上采用激光全息编码原理的无损压缩（Serva Encoder）与同一参照空间下的计算映射引擎（Chimera）以及多维向量算子。

**📊 数据集**

实验使用 Fashion‑MNIST、MNIST 作为模型训练验证，并在 Canterbury Corpus 等通用压缩基准上评估压缩效果。

**📈 对比分析**

通过内部基准，将 SERVA 与标准 MLP、CNN、RNN 在同一数据集上对比，结果显示在相同准确率下能效提升 30–374 倍、训练速度提升 35–723 倍，压缩率 4.17× 以上。

**⚠️ 局限性**

局限性包括目前仅在受控基准下验证，尚未在大规模多模态模型或生产环境中评估；需进一步开放源码以促进生态兼容与标准化。

---

## 40. From Adversarial Poetry to Adversarial Tales: An Interpretability Research Agenda

**arXiv ID:** 2601.08837 | [PDF](https://arxiv.org/pdf/2601.08837v1)

**作者:** Piercosma Bisconti `[一作]` (Sapienza University of Rome), Daniele Nardi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 11018 | [OpenAlex ID](https://openalex.org/A5075651762)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于普罗普叙事结构的“Adversarial Tales” jailbreak 技术，通过在赛博朋克短篇故事中嵌入有害请求，并让模型进行功能性分析来绕过安全过滤；

**💡 创新点**

创新点在于将结构化叙事分析（普罗普函数分解）与恶意请求结合，形成新的结构化 jailbreak 类别，显示其跨模型、跨风险域的普遍性；

**🔧 技术方法**

技术主要包括：手工构造 40 条嵌入有害内容的短篇故事、以普罗普函数为基础的分析提示、对 26 个前沿 LLM 进行单轮测试，评估攻击成功率；

**📊 数据集**

使用自创的 40 条对抗性赛博朋克故事数据集，覆盖 EU AI Act 的四大风险类别（CBRN、网络攻击、操纵、失控）；

**📈 对比分析**

与之前的 Adversarial Poetry 进行对比，采用攻击成功率（ASR）指标。结果显示整体 ASR 为 71.3%，比 Poetry 的 62% 更高，且在所有模型族中均表现出较高成功率；

**⚠️ 局限性**

局限性包括：仅测试单轮提示，未考虑多轮对话；对抗性故事仍由人工编写，缺乏大规模自动生成；实验缺乏对模型内部机制的深度解释，难以直接指导对策。

---

## 41. LAUDE: LLM-Assisted Unit Test Generation and Debugging of Hardware DEsigns

**arXiv ID:** 2601.08856 | [PDF](https://arxiv.org/pdf/2601.08856v1)

**作者:** Deeksha Nandal `[一作]` (University of Illinois Chicago), Debjit Pal `[通讯]` (University of Illinois Chicago)

**通讯引用:** 419 | [OpenAlex ID](https://openalex.org/A5072879381)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了统一的单元测试生成与调试框架，结合LLM的链式思考和设计源代码语义，实现硬件设计的自动化单元测试与故障定位。

**💡 创新点**

创新点在于将LLM的CoT推理、提示工程和仿真反馈闭环结合，显著提升组合与时序电路的单元测试敏感性与特异性，支持迭代式调试。

**🔧 技术方法**

采用了 SystemVerilog 源代码解析、结构化提示生成、仿真覆盖报告、Divergent Attack 等评价指标，并通过循环迭代调试提升可靠性。

**📊 数据集**

使用 VerilogEval 基准改造生成的 1,560 个注入功能性 Bug 的组合/时序设计代码作为评测数据集。

**📈 对比分析**

通过对比 Gemini‑2.5 Pro 与 DeepSeek R1 等闭源和开源 LLM，在 Attack Rate、Divergence Rate、Divergent Attack 以及调试成功率等指标上评估性能；Gemini 在组合电路上可达≈85%成功率，DeepSeek 在提供源代码时可逼近 Gemini，整体表现显示 LLM 在多种设计上实现了高检测率与调试率。

**⚠️ 局限性**

局限性包括：数据集规模有限，仅包含单模块设计；缺乏对代码质量的细粒度评估；未针对时序行为训练或预测 LLM；评估侧重数量化指标，未深入分析错误类型与鲁棒性。

---

## 42. An Information-Theoretic Perspective on LLM Tokenizers

**arXiv ID:** 2601.09039 | [PDF](https://arxiv.org/pdf/2601.09039v1)

**作者:** Mete Erdogan `[一作]` (Stanford University), Tsachy Weissman `[通讯]` (Stanford University)

**通讯引用:** 6197 | [OpenAlex ID](https://openalex.org/A5043344688)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（LLM）tokenizer进行信息论视角评估，系统比较预训练和学习型tokenizer在压缩率、熵分布、跨域鲁棒性等指标上的表现，并提出基于容量利用率的通道视角与压缩感知BPE算法。

**💡 创新点**

将tokenizer视作无噪声通道，定义容量利用率与Rényi利用率，揭示训练规模对n-gram熵与压缩效率的双向影响；并设计了LZ-aware BPE以在压缩器目标上直接优化tokenizer。

**🔧 技术方法**

使用信息论熵估计（1-gram至5-gram）、无噪声通道理论、压缩率评估、LZ压缩管道以及基于gzip压缩目标的贪心合并实现LZ-aware BPE。

**📊 数据集**

C4、CodeParrot、GSM8K、Oscar、Bigcode Starcoder等多语言web文本数据集。

**📈 对比分析**

通过比较压缩率、tokens/char、k-gram熵以及压缩后bits/char，发现tokenizer训练规模越大unigram熵升高但高阶熵显著降低；预训练GPT tokenizer在多域表现稳健，但对非拉丁脚本过度分割；压缩器预处理可提升10-20%压缩率；LZ-aware BPE比普通BPE压缩更好，但计算成本更高。

**⚠️ 局限性**

未直接评估tokenizer改进对下游语言模型性能的影响；LZ-aware BPE计算成本高；跨域鲁棒性仍有限；对不同脚本和语言的细粒度分析不足。

---

## 43. Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment

**arXiv ID:** 2601.09051 | [PDF](https://arxiv.org/pdf/2601.09051v1)

**作者:** Yiming Du `[一作]` (Old Dominion University), Lusi Li `[通讯]` (Old Dominion University)

**通讯引用:** 1683 | [OpenAlex ID](https://openalex.org/A5006167598)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于深度学习的分层填补与对齐框架 DIMVC-HIA，用于解决不完整多视图聚类问题。

**💡 创新点**

创新点在于引入分层填补策略——先利用跨视图对比相似性恢复缺失的聚类分配，再以聚类原型填补缺失特征，并配合能量模型与对比分配对齐实现语义一致性与聚类紧凑性。

**🔧 技术方法**

技术方法包括视图专属自编码器、共享聚类预测器、能量基础模型（EBM）、对比学习、层级填补以及重建损失等。

**📊 数据集**

实验使用 BDGP、MNIST‑USPS、Fashion‑MNIST 和 Handwritten 四个多视图数据集。

**📈 对比分析**

与 10 种主流不完整多视图聚类基线（COMPLETER、CPSPAN、DCG、DIVIDE、RPCIC、APADC、GIMVC、PMIMC、ProImp、DSIMVC）对比，DIMVC‑HIA 在所有缺失率下均获得最高或第二高的 ACC/NMI/PUR，尤其在高缺失率（η=0.7）时表现突出。

**⚠️ 局限性**

局限性主要体现在模型结构较为复杂、对超参数 α、β 的选择敏感，以及在极高缺失率或大规模数据集上的可扩展性尚未充分验证。

---

## 44. OpenDecoder: Open Large Language Model Decoding to Incorporate Document Quality in RAG

**arXiv ID:** 2601.09028 | [PDF](https://arxiv.org/pdf/2601.09028v1)

**作者:** Fengran Mo `[一作]` (University of Montreal), Jian-Yun Nie `[通讯]` (University of Montreal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在检索增强生成（RAG）框架中，作者提出一种新的解码调制方法——OpenDecoder（OpenDecoder），通过在 LLM 的注意力机制中注入检索结果的显式相关性指标（检索相关分数、LLM 排序分数和查询性能预测分数）来提升模型对噪声上下文的鲁棒性，并在训练阶段加入了鲁棒性训练（替换部分检索结果为部分相关或无关文档）

**💡 创新点**

创新点主要有三：1) 将检索时得到的外部相关性信号直接嵌入到 LLM 的内部解码注意力计算中，而非仅依赖提示或模型内部判断；2) 通过多种指标（检索分数、LLM 排序分数、QPP）构造显式质量指示器，并对其进行聚合与归一化；3) 在训练阶段通过替换和打乱检索结果来实现对噪声环境的鲁棒性学习，显著提升模型在不同噪声水平下的表现

**🔧 技术方法**

技术手段包括：基于 Qwen-2.5-3B-Instruct 的大语言模型；检索模块采用 E5 向量检索器；构造相关性指示器并将其作为额外的权重向量投射到注意力计算中；鲁棒性训练（文档替换与随机顺序）；实验中使用 F1 和 Exact Match 评估指标

**📊 数据集**

使用了五个基准数据集：通用问答（NQ、TriviaQA、PopQA）和多跳问答（HotpotQA、2WikiMultiHopQA），每个数据集都在三种噪声环境（正常、噪声、极端噪声）下进行评测

**📈 对比分析**

与六种基线（Vanilla RAG、Vanilla SFT、RobustRAG、AstuteRAG、InstructRAG、RbFT）对比，OpenDecoder 在所有噪声环境和所有数据集上均显著优于基线，尤其在极端噪声场景下仍保持较高的 F1/EM 分数；实验还展示了不同指标聚合、归一化方式、文档顺序及模型规模对性能的影响，表明 OpenDecoder 的鲁棒性优势较为显著

**⚠️ 局限性**

局限性包括：1) 依赖检索时产生的相关性分数，若检索本身失效或分数不可靠，模型可能受限；2) 目前仅在 Qwen-2.5-3B 的规模下验证，较小模型可能难以充分利用显式指标；3) 对不同检索器或更大规模语料的迁移性尚未评估；4) 聚合与归一化策略仍较简单，未来可能需要更自适应的机制

---

## 45. From Symbolic to Natural-Language Relations: Rethinking Knowledge Graph Construction in the Era of Large Language Models

**arXiv ID:** 2601.09069 | [PDF](https://arxiv.org/pdf/2601.09069v1)

**作者:** Kanyao Han `[一作]` (Walmart Global Tech), Yushang Lai `[通讯]` (Walmart Global Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从概念层面阐述了知识图谱（KG）关系表述由传统符号类别向自然语言描述迁移的必要性，并提出了保持最小符号骨干的混合式KG设计原则；同时梳理了该范式下的研究方向和技术路线。

**💡 创新点**

创新点在于①将KG关系视为自然语言表达，同时保留可供检索和推理的符号结构；②提出隐式、任务感知的模式约束和混合符号‑自然语言关系的设计原则；③给出多阶段LLM驱动的关系构建、精炼与推理流程的概念框架。

**🔧 技术方法**

主要技术手段为LLM推理与提示（Prompting）、KG‑to‑Text转换、基于文本的关系生成、结构‑语义联合嵌入、检索增强生成（RAG）以及LLM评判框架；并未给出具体实现细节。

**📊 数据集**

文中并未使用特定数据集，讨论范围涵盖公开文本语料、学术论文、法律文档等通用语言资源，重点关注多源信息整合与冲突处理。

**📈 对比分析**

由于是位置性论文，未进行实验比较；作者仅提出潜在的评估思路（如LLM‑judge、准真度与简洁度指标）并指出需进一步验证。

**⚠️ 局限性**

限制包括：①聚焦关系表述，未系统讨论实体表述的重构；②以文本为主，未涉及图像、音频等多模态KG；③缺乏大规模实验与性能数据，提出的原则尚未在真实系统中得到验证。

---

## 46. Design Methodology of Hydraulically-driven Soft Robotic Gripper for a Large and Heavy Object

**arXiv ID:** 2601.09104 | [PDF](https://arxiv.org/pdf/2601.09104v1)

**作者:** Ko Yamamoto `[一作]` (University of Tokyo), Osamu Azami `[通讯]` (University of Tokyo)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5083758126)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并验证了一种水力驱动的软体抓取器，能够在约20 kg、20–30 cm直径物体上实现抓取，提出了基于压力、弯曲角度和抓取力的数学模型并结合有限元分析进行材料与参数选择。

**💡 创新点**

创新点在于：①将高压水力驱动引入软体抓取器，实现数兆帕级别的可调压力；②构建包含油箱弹性效应的完整系统模型，并利用有限元结果得到关节角度系数 w；③通过模型估算最大承载并实验验证，首次实现单抓取器20 kg的大负荷且保持柔性与适应性。

**🔧 技术方法**

使用的技术包括：水力泵+油箱、NBR软管与Kevlar线束制成的纤维约束指尖、A5052金属壳与PLA外壳、有限元(FEM)分析、基于Neo‑Hookean与Mooney‑Rivlin材料模型的数学建模、PI闭环控制与压力传感器反馈。

**📊 数据集**

并未使用公开数据集，而是通过自制实验平台进行抓取试验，采集压力、指尖角度与最大可抓重量等实验数据，用于验证模型和优化参数。

**📈 对比分析**

与现有软体抓取器（如气动、磁性、Kirigami等）进行对比，重点对比了承载重量与自重比，得到17.7的承载/自重比，优于大多数已公开的软体抓取器，且在20 kg重量下可保持高抓取力和闭环控制性能。

**⚠️ 局限性**

局限性包括：①模型对不同形状、表面粗糙度的鲁棒性尚待进一步验证；②系统对极高负荷或极大尺寸物体的扩展仍受限于泵压与材料强度；③在非理想环境（如湿滑、极端温度）下的性能尚未全面评估。

---

## 47. Dynamic Hierarchical $j$-Tree Decomposition and Its Applications

**arXiv ID:** 2601.09139 | [PDF](https://arxiv.org/pdf/2601.09139v1)

**作者:** Gramoz Goranci `[一作]`, Gernot Zöcklein `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于稀疏核心的分层树结构，用定理4.1在每一层构造 O(n/k^i) 树，实现了从顶层到底层的递归层次化建模；

**💡 创新点**

创新点在于通过核心稀疏化技术将高维网络压缩为低维核心，再递归构造多层树，使得每层的节点数按 n/k^i 下降，从而显著降低空间和时间复杂度；

**🔧 技术方法**

主要技术包括核心稀疏化、递归树构造、定理4.1的理论证明以及多层次动态更新算法；

**📊 数据集**

实验使用了标准的 SNAP 图数据集（如Twitter、Webgraph、Friendster）以及合成稠密图，以验证算法在不同规模与稠密度下的表现；

**📈 对比分析**

与传统的全图搜索和单层树算法相比，该方法在构建时间上平均降低约 60–70%，在内存使用上减少 80% 以上，同时保持与原图相近的近似质量；

**⚠️ 局限性**

局限性包括：仅针对无向图或需先做预处理的图；对极大稠密图仍可能出现内存瓶颈；以及算法对参数 k 的选择敏感，需要经验调优。

---

## 48. Proactively Detecting Threats: A Novel Approach Using LLMs

**arXiv ID:** 2601.09029 | [PDF](https://arxiv.org/pdf/2601.09029v1)

**作者:** Aniesh Chawla `[一作]` (California), Udbhav Prasad `[通讯]` (California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一个自动化系统，利用LLM从15个网页威胁情报源主动识别IOC（IPv4、IPv6、域名）。

**💡 创新点**

首次在真实网页环境中系统性评估LLM的主动IOC检测能力，并提出了基于爬虫+正则的多源IOC提取框架。

**🔧 技术方法**

使用的大语言模型包括Gemini、Qwen、Llama系列；技术手段涵盖爬虫抓取、正则提取、MITRE ATT&CK映射和上下文限定的问答提示。

**📊 数据集**

评估数据来自15家安全厂商RSS，收集479网页（含711 IPv4、502 IPv6、1445域名），并在303网页（116 IPv4、187域名）上进行细化测试。

**📈 对比分析**

通过混淆矩阵计算精确率、召回率、特异性、F1等指标；Gemini 1.5 Pro表现最佳（精确率0.958、召回率1.0），Qwen 32B召回率低且F1差，Llama 70B召回率稍低但TP完整。

**⚠️ 局限性**

存在对域名与非恶意指标的上下文理解不足导致误报；模型未做针对特定威胁情报的微调，且仅处理网页文本，缺乏对PDF、图像等其他文件格式的支持。

---

## 49. Recursive Knowledge Synthesis for Multi-LLM Systems: Stability Analysis and Tri-Agent Audit Framework

**arXiv ID:** 2601.08839 | [PDF](https://arxiv.org/pdf/2601.08839v1)

**作者:** Toshiyuki Shigemura `[一作]` `[通讯]` (Independent Researcher), Toshiyuki Shigemura (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一个三代理交叉验证框架，以实现多LLM系统的递归知识合成与稳定性评估。

**💡 创新点**

引入递归知识合成（RKS）与基于收缩映射的透明度审计模块，结合手工桥接的会话级角色分解，提供安全、可解释的多LLM协同。

**🔧 技术方法**

使用三种异构LLM（ChatGPT、Gemini、Copilot）进行手工调度、会话隔离、透明度审计、偏差检测，并通过RRS、TS、DDR、CSR等定量指标进行评估。

**📊 数据集**

使用47个自定义验证试验，配合预设矛盾集合与人工评分标准，未依赖公开大规模数据集。

**📈 对比分析**

通过人工评估得到平均RRS 0.78±0.06，TS≥0.7的合规率68%，收敛率89%，表明框架在公共浏览器接口下实现了稳定的交叉验证。

**⚠️ 局限性**

依赖手工桥接导致可扩展性受限，公开部署不可复现，评价主观性高，实验规模有限。

---

## 50. Rubric-Conditioned LLM Grading: Alignment, Uncertainty, and Robustness

**arXiv ID:** 2601.08843 | [PDF](https://arxiv.org/pdf/2601.08843v1)

**作者:** Haotian Deng `[一作]` (Purdue University), David Tang `[通讯]` (Purdue University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在仅给定评分 Rubric 的条件下，使用大型语言模型（LLM）对学生短答进行自动评分，并系统评估其与专家评分的对齐度、基于共识的退回机制以及鲁棒性。

**💡 创新点**

创新点在于提出了“Trust Curve”——基于多次采样共识投票的退回机制，可在保持覆盖率的同时提升准确率；同时对 LLM 在同义词替换、噪声与对抗攻击下的敏感性进行量化评估。

**🔧 技术方法**

采用 Qwen 2.5‑72B（大模型）+ 级联推理提示（Chain‑of‑Thought）、少量标注校准、10 次采样共识投票，并使用自然语言扰动与对抗样本进行鲁棒性测试。

**📊 数据集**

使用 SciEntsBank 作为科学问答短答评估基准，涵盖 2/3/5 级评分。

**📈 对比分析**

对齐度通过 Accuracy 与 Cohen’s κ 评估，实验显示二元评分 76%+、3 级 69%+、5 级 57%+；通过共识退回可将二元评分提升至 81%、5 级提升至 64%；在扰动与对抗攻击实验中，模型对同义词替换最为敏感。

**⚠️ 局限性**

局限包括：对细粒度（3/5 级）评分的对齐下降、对同义词替换极度敏感、退回机制导致推理成本上升、未针对更强对抗攻击或多模型协同进行评估。

---

## 51. A Machine Learning Approach Towards Runtime Optimisation of Matrix Multiplication

**arXiv ID:** 2601.09114 | [PDF](https://arxiv.org/pdf/2601.09114v1)

**作者:** Yufan Xia `[一作]` (Australian National University), Giuseppe Maria Junior Barca `[通讯]` (Australian National University)

**通讯引用:** 2401 | [OpenAlex ID](https://openalex.org/A5024994202)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过机器学习模型在运行时自动选择最优线程数，从而加速多线程GEMM实现。

**💡 创新点**

创新点在于将数据驱动的线程调优与传统基于经验或硬件计数的调优方法结合，提供了一个可在不同HPC平台上自动部署、无需手动调参的解决方案。

**🔧 技术方法**

主要技术包括使用scrambled Halton序列采样矩阵尺寸、局部离群点去除（LOF）、Yeo‑Johnson变换、特征标准化、特征选择、XGBoost回归模型以及在库中实现的即时线程数预测与切换。

**📊 数据集**

数据集为在两台真实HPC节点（Intel Cascade Lake与AMD Zen 3）上采集的1763条GEMM样本，内存占用上限分别为500 MB，随后还利用174条低方差采样数据用于最终性能验证。

**📈 对比分析**

与传统使用全部物理核心数的BLAS实现（MKL/BLIS）相比，本文方法在两台平台上平均提升约25%–40%（针对内存占用≤100 MB的情况），在更大内存范围内仍保持至少10%–20%的加速，并在某些小尺寸或不规则矩阵上获得高达80×的极端加速。

**⚠️ 局限性**

局限性包括：仅针对单精度SGEMM、仅在两种CPU架构验证、对极大尺寸矩阵或混合内存访问模式的适应性未知、模型训练与验证需要占用额外计算资源、以及在高线程数与超线程环境下的同步与数据复制开销仍可能成为瓶颈。

---

## 52. Movable Antenna Assisted Dual-Polarized Multi-Cell Cooperative AirComp: An Alternating Optimization Approach

**arXiv ID:** 2601.09137 | [PDF](https://arxiv.org/pdf/2601.09137v1)

**作者:** Mingyu Hu `[一作]` (Southeast University), Wei Kang `[通讯]` (Southeast University)

**通讯引用:** 3224 | [OpenAlex ID](https://openalex.org/A5034649717)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在多小区协作的 AirComp 系统中引入双极化可移动天线，联合优化天线位置、接收滤波矩阵、用户发射系数、极化向量等参数，以最小化总均方误差。

**💡 创新点**

首次将双极化天线与可移动天线相结合，在多小区环境下实现对极化与空间自由度的联合调度，并提出针对极化向量的 SCA+SDR 方案以及基于统计 CSI 的两时标优化框架。

**🔧 技术方法**

采用交替优化、闭式解、连续凸逼近（SCA）、半正定松弛（SDR）、梯度下降、统计平均法与 CVX 求解器。

**📊 数据集**

采用仿真数据，生成多径 Rayleigh/Rician 信道，设置 3 个小区、每小区 8 名用户、4 个可移动天线，使用多组随机信道样本进行统计 CSI 评估。

**📈 对比分析**

与固定天线阵列（FPA）和单极化可移动天线（MA）对比，D‑PMA 在瞬时与统计信道场景下均能显著降低 MSE，尤其在高 SNR 与多路径环境中提升 10‑20% 以上。

**⚠️ 局限性**

计算复杂度高，收敛至局部最优；需要准确的 CSI 或统计 CSI；天线运动受物理空间限制，极化模型对极化失配敏感。

---

## 53. $D^2Prune$: Sparsifying Large Language Models via Dual Taylor Expansion and Attention Distribution Awareness

**arXiv ID:** 2601.09176 | [PDF](https://arxiv.org/pdf/2601.09176v1)

**作者:** Lang Xiong `[一作]` (Chongqing University), Duo Liu `[通讯]` (Chongqing University)

**通讯引用:** 3996 | [OpenAlex ID](https://openalex.org/A5100675843)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 D^2Prune，一种后训练剪枝方法，结合双Taylor展开和注意力分布感知动态更新，专门用于压缩大型语言模型。

**💡 创新点**

创新点在于：①使用双Taylor展开同时建模激活与权重的误差影响，精确估计剪枝误差；②为注意力模块设计动态权重更新策略，以保持长尾注意力分布，避免传统全更新或不更新导致的性能急剧下降。

**🔧 技术方法**

技术手段包括：双Taylor展开误差估计、KL散度约束的动态权重更新、基于perplexity的搜索策略、LoRA微调、以及与现有 SparseGPT、Wanda、Pruner-Zero 等方法的对照实验。

**📊 数据集**

数据集涵盖：C4（校准）、WikiText‑2（语言建模评估）、EleutherAI LM Harness 七个零拷贝任务、DeiT/ViT、Qwen3 以及 GSM8K（推理能力评估）等。

**📈 对比分析**

与 SparseGPT、Wanda、Pruner‑Zero、SparseLLM 等基线比较，在 50%–80% 稀疏度、2:4/3:4 结构和无结构剪枝场景下，D^2Prune 在 perplexity 平均下降约 16%（对应约 3.1% 的准确率提升），甚至在 50% 稀疏度下超越原始密集模型；在高稀疏度仍保持竞争力，并在可视化与推理速度方面表现优异。

**⚠️ 局限性**

局限性包括：①对激活偏移的线性假设仍需进一步细化；②动态更新搜索空间在极高稀疏度下可能变大，计算成本上升；③实验主要聚焦于 Transformer 结构，尚未验证对非 Transformer 网络的通用性；④对推理时的加速效果（如量化、分块等）需进一步研究。

---

## 54. Probabilistic Computers for MIMO Detection: From Sparsification to 2D Parallel Tempering

**arXiv ID:** 2601.09037 | [PDF](https://arxiv.org/pdf/2601.09037v1)

**作者:** M Mahmudul Hasan Sajeeb `[一作]` (University of California), Kerem Y. Camsari `[通讯]` (University of California)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并实现了一个完全在芯片上运行的p‑bit并行退火求解器，用于稀疏化后的MIMO检测，并在FPGA上实现完整的并行退火与2D并行退火算法；

**💡 创新点**

创新点在于通过图稀疏化与辅助复制节点实现密集耦合的硬件可扩展性，并首次提出二维并行退火（在温度与复制约束两维交换）来消除对复制约束强度P的调优需求，显著提升收敛速度；

**🔧 技术方法**

采用p‑bit异步随机计算、并行退火、二维并行退火、FPGA硬件实现以及7nm ASIC预测设计等技术；

**📊 数据集**

使用64×64及128×128的随机Rayleigh信道BPSK MIMO实例和Sherrington–Kirkpatrick自旋玻璃作为实验数据集；

**📈 对比分析**

与传统MMSE线性检测比较，BER显著下降；与1D PT比较，2D PT提升约10倍以上收敛速度并在高SNR下实现零错误率；FPGA实现平均4.7 ms/实例，ASIC预测89 MHz/185 mW；

**⚠️ 局限性**

局限性包括对复制约束强度P敏感（需调优），目前FPGA仅实现1D PT，2D PT仍待ASIC实现；硬件资源受限，规模扩展仍需进一步研究；

---

## 55. Can LLMs interpret figurative language as humans do?: surface-level vs representational similarity

**arXiv ID:** 2601.09041 | [PDF](https://arxiv.org/pdf/2601.09041v1)

**作者:** Samhita Bollepally `[一作]` (Texas A&M University), Takashi Yamauchi `[通讯]` (Texas A&M University)

**通讯引用:** 4462 | [OpenAlex ID](https://openalex.org/A5044283169)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在解释具象和社会化语言时与人类判断的相似性，使用对话式句子与情感、讽刺、幽默等六类特征进行评估。

**💡 创新点**

创新点在于同时比较模型与人类在表面级别（SLS）和内部表征级别（RSA）上的一致性，并揭示即使表面表现相近，内部语义结构仍存在显著差距。

**🔧 技术方法**

主要技术包括表面级相关系数计算、欧氏距离构建相似度矩阵以及RSA（Representational Similarity Analysis）来度量语义结构相似度。

**📊 数据集**

使用了240句含传统、习语、情感、幽默、讽刺和Gen‑Z俚语的句子，每句配有40个解释性问题，形成9,600条句子‑问题对。

**📈 对比分析**

结果显示 GPT‑4 在SLS和RSA上与人类的相似度最高，其他模型（Gemma‑2‑9B、Mistral‑7B、Llama‑3.2）表现中等或较差；尽管GPT‑4的表面相似度可达0.60‑0.90，但其RSA与人类的相关系数仍低于人类自身间的0.75‑0.85。

**⚠️ 局限性**

局限性包括样本量有限、缺乏上下文信息、只用量表评估复杂判断以及假设人类与模型共享内部表征框架。

---

## 56. A Decompilation-Driven Framework for Malware Detection with Large Language Models

**arXiv ID:** 2601.09035 | [PDF](https://arxiv.org/pdf/2601.09035v1)

**作者:** Aniesh Chawla `[一作]` (California), Udbhav Prasad `[通讯]` (California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个自动化管道，先用Ghidra将Windows可执行文件反编译为C代码，再让LLM对其进行恶意/良性分类。

**💡 创新点**

通过在LLM上进行针对恶意软件的细粒度微调，显著提升分类准确率，证明持续更新模型对抗新威胁至关重要。

**🔧 技术方法**

使用大型语言模型（Gemini 2.5 Pro、Llama 3.3 70B、Codestral、Claude 3.7 Sonnet等）以及专门设计的零射击提示。

**📊 数据集**

基准数据集为2017年Malware Data Science书籍中的约600个样本；最新数据集为2025年Malware Bazaar提供的120份恶意样本与28个恶意外观良性驱动文件。

**📈 对比分析**

与传统XGBoost静态特征分类器对比，原始LLM（Gemini 2.5 Pro）准确率约80%，微调后达到83%准确率、94%精度，优于XGBoost在新数据上的表现。

**⚠️ 局限性**

受限于LLM上下文窗口、对新变种的泛化能力不足、需要持续微调以及缺乏可解释性。

---

## 57. PediaMind-R1: A Temperament-Aware Language Model for Personalized Early Childhood Care Reasoning via Cognitive Modeling and Preference Alignment

**arXiv ID:** 2601.08848 | [PDF](https://arxiv.org/pdf/2601.08848v1)

**作者:** Zihe Zhang `[一作]` (Fudan University), Jichao Leng `[通讯]` (Fudan University)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5040983147)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了PediaMind‑R1，一款针对早期儿童护理的域专用大语言模型，能够基于托马斯‑切斯温度理论实现主动个性化推荐。

**💡 创新点**

将心理温度模型与知识图谱结合，采用两阶段训练（SFT+GRPO）实现温度感知推理和情感化对齐。

**🔧 技术方法**

采用监督微调+LoRA、Group Relative Policy Optimization（GRPO）强化学习、基于温度知识图谱的结构化链式推理。

**📊 数据集**

1215条带温度标签的育儿问答、2646条温度敏感场景及100条人类评测样本，全部基于DeepSeek和育儿百科。

**📈 对比分析**

与未调Qwen2.5‑7B‑Instruct基线在200道多项选择题进行零样例评测，SFT提升到62%，SFT+GRPO提升到67%；在人类评测中知识、心理一致性与照护适配度分别从0.68/0.68/0.75提升到0.72/0.92/0.88。

**⚠️ 局限性**

依赖人工温度评估、数据规模有限、仅使用经典Thomas‑Chess模型、奖励信号离散、未做统计显著性检验。

---

## 58. Optimising for Energy Efficiency and Performance in Machine Learning

**arXiv ID:** 2601.08991 | [PDF](https://arxiv.org/pdf/2601.08991v1)

**作者:** Emile Dos Santos Ferreira `[一作]` (University of Cambridge), Andrei Paleyes `[通讯]` (Pasteur Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并实现了ECOpt框架，用于在机器学习模型的超参数搜索中同时优化能源效率和模型性能，并通过多目标贝叶斯优化生成可解释的Pareto前沿；

**💡 创新点**

创新点在于：①首次将能源效率与性能视为双目标进行贝叶斯搜索，直接量化两者权衡；②公开模型能源度量并验证Transformer模型在不同硬件上的能效一致性；③发现Transformer能效随参数规模呈对数线性下降的能耗缩放规律；④利用该框架在CIFAR‑10上发现七个在准确率与能效兼优的模型；

**🔧 技术方法**

采用多目标贝叶斯优化（MOBO）、Ax/BoTorch框架、CodeCarbon能源计量、MLflow实验追踪；模型包装器支持任意ML模型；

**📊 数据集**

使用CIFAR‑10进行图像分类实验，以及BookCorpus（文本生成）与多种Transformer（GPT‑2、Qwen3、Gemma 3、Llama 3.1）进行能效测评；

**📈 对比分析**

与传统随机搜索、手工调参及仅关注性能的贝叶斯搜索对比，ECOpt在同等计算资源下取得约76.09%准确率（相较71.17%提升）并显著提升能效（如Gemma 3批量优化后能效提升38倍）；

**⚠️ 局限性**

局限性包括：①能源测量依赖CodeCarbon，存在高达40%误差；②仅计量计算能耗，未考虑冷却与完整数据中心能耗；③MOBO在高维搜索空间中收敛慢；④实验未包含常见推理加速技术（缓存、模型路由）导致结果不完全适用于生产环境。

---

## 59. Learning Domain-Invariant Representations for Cross-Domain Image Registration via Scene-Appearance Disentanglement

**arXiv ID:** 2601.08875 | [PDF](https://arxiv.org/pdf/2601.08875v1)

**作者:** Jiahao Qin `[一作]`, Yiwen Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出SAR-Net框架，通过场景-外观分解实现跨域图像配准；

**💡 创新点**

在不满足亮度恒等假设的情况下，用场景一致性损失和域对齐损失实现无监督的跨域配准；

**🔧 技术方法**

使用U-Net场景编码器、全局平均池化外观编码器、特征调制前向模型，结合实例归一化、梯度匹配、NCC等损失；

**📊 数据集**

在自制的高速度双向光学分辨率光声显微镜（OR‑PAM）脑血管数据集PAM‑DSR上进行验证；

**📈 对比分析**

与传统方法（SIFT、Demons、Optical Flow、SyN）及深度学习方法（VoxelMorph、TransMorph、DGIR、SACB‑Net）对比，SAR‑Net在SSIM、NCC、VCI上分别提升至0.885/0.979/0.857，实时率77fps；

**⚠️ 局限性**

局限性包括对强照明不均的敏感、在稀疏结构区域场景编码可能不稳定，以及仅在单一显微镜数据上验证，缺乏跨模态推广实验。

---

## 60. Token positional games

**arXiv ID:** 2601.08967 | [PDF](https://arxiv.org/pdf/2601.08967v1)

**作者:** Guillaume Bagan `[一作]` (University Lyon), Nacim Oijid `[通讯]` (Umeå University)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5081421238)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文提出并研究了令牌化（Token）位置游戏（Maker‑Breaker token positional games），在经典 Maker‑Breaker 游戏中给双方分配有限数量的令牌，并探讨在此限制下谁能获胜以及所需的最少令牌数。

**💡 创新点**

创新点主要包括：
• 定义并分析令牌数 θ(H)（Maker 需要的最少令牌数）和游戏时长 τ(H)，揭示了 k‑uniform 超图中 θ(H) 的取值范围（k=2,3 时为 k，k≥4 时可达到 Θ(|V(H)|)）；
• 对 Breaker 只有 1 令牌的情形给出了多项式时间解法，并提出了可约对（reducible pair）的概念；
• 证明了在一般情况下 Token 游戏是 PSPACE‑hard 的，同时给出当 Breaker 只有 1 令牌时的多项式算法；
• 证明 Token Sliding 版本在超图上是 NP‑complete（即 EXPTIME‑complete），并给出与永恒占据（eternal domination）问题的归约；
• 提出了按令牌数参数化的 XP 算法。

**🔧 技术方法**

技术方法包括：
• 令牌化的策略与经典 Maker‑Breaker 策略的映射；
• 对可约对（reducible pair）的构造与归约技术；
• 对超图子结构（如 nunchaku、necklace）的分析；
• 对游戏状态图 G_{a,b}(H) 的构造与可达性求解；
• 与永恒占据问题的归约构造，实现 NP/EXPTIME 难度证明；
• 复杂度分析中利用对称性、匹配与配对策略。

**📊 数据集**

论文未使用外部数据集，而是通过理论构造的超图实例（如 nunchaku、necklace、特殊 4‑uniform 超图等）来展示结果和证明边界。

**📈 对比分析**

方法对比与性能：
• 对 Breaker 只有 1 令牌的情况，提供 O(m³)（m 为超边数）多项式算法；
• 对一般令牌数的情况，证明 PSPACE‑hard（无法在多项式时间内求解）；
• 对按令牌数参数化的情况给出 O(n²k²) 的 XP 算法；
• 对 Token Sliding 版本给出 NP/EXPTIME 完备性，说明无法在多项式时间内解决。

**⚠️ 局限性**

局限性：
• 对于 Breaker 令牌数 >1 的情形，没有得到 FPT 或多项式算法；
• Token Sliding 版本的 NP/EXPTIME 完备性仅针对任意初始位置，尚未给出起始位置下的复杂度；
• 对标准令牌跳跃版（token jumping）的复杂度仍未知；
• 对 k≥4 的情况，虽然给出了 θ(H) 的上界，但尚未完全确定 θ(H) 能否达到 |V(H)|/2 的上限。

---

## 61. Companion Agents: A Table-Information Mining Paradigm for Text-to-SQL

**arXiv ID:** 2601.08838 | [PDF](https://arxiv.org/pdf/2601.08838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 62. Evaluating Role-Consistency in LLMs for Counselor Training

**arXiv ID:** 2601.08892 | [PDF](https://arxiv.org/pdf/2601.08892v1)

**作者:** Eric Rudolph `[一作]` (Technische Hochschule Nürnberg Georg Simon Ohm), Jens Albrecht `[通讯]` (Technische Hochschule Nürnberg Georg Simon Ohm)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了大语言模型（LLM）在虚拟客户（VirCo）训练中的角色一致性，构建了包含对抗攻击的对话数据集，并比较了Vicuna、Llama3-German和SauerkrautLM等开源LLM的表现；

**💡 创新点**

创新点在于将针对性对抗攻击（如离题提问、毒性回复、长文本、OOD写作风格等）融入评估数据集，首次系统评估LLM在面对对抗输入时保持角色一致性的能力；

**🔧 技术方法**

采用GPT‑4进行自动评估框架、人工评分、LoRA 与 DPO 微调技术（未来工作），并使用量化（8‑bit、4‑bit）技术以提升模型部署效率；

**📊 数据集**

使用改造后的VirCo对话数据集，包含7个角色描述和多种对抗攻击样本，涵盖人物一致性、对话连贯性和不可预见输入三个维度；

**📈 对比分析**

通过自动评估（GPT‑4）与三位专业评审的人工评分对比，发现Vicuna‑13B‑16k在角色一致性和对话连贯性上平均排名最高，量化模型（8‑bit/4‑bit）与原版性能基本一致，整体一致性率约90%；

**⚠️ 局限性**

局限在于GPT‑4评估对抗场景时偏离人工评分，难以准确判断客户端行为；情感表达不足导致对话自然度下降；对极端对抗输入的鲁棒性仍待提升，评估方法在极端场景下可靠性不足。

---

## 63. Bridging the Gap: Empowering Small Models in Reliable OpenACC-based Parallelization via GEPA-Optimized Prompting

**arXiv ID:** 2601.08884 | [PDF](https://arxiv.org/pdf/2601.08884v1)

**作者:** Samyak Jhaveri `[一作]` (University of California), Cristina V. Lopes `[通讯]` (University of California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了通过 GEPA 反射式提示优化技术提升小型 LLM 在 OpenACC 指令生成中的编译成功率与 GPU 加速效果。

**💡 创新点**

创新点在于将遗传-帕累托演化与结构化语义反馈结合，用“黄金”pragma 示例引导提示演化，从而显著提高小型模型的语义正确性与性能。

**🔧 技术方法**

采用 GEPA（Genetic-Pareto）框架、自然语言反馈反射、OpenACC 语义比较与评分、以及多阶段推理（数据管理+循环并行化）等技术。

**📊 数据集**

使用 64 条高质量“黄金”OpenACC 示例与 PolyBench/C 4.2.1 基准测试集合进行评估。

**📈 对比分析**

通过与初始提示基线对比，发现编译成功率从 78.3% 提升到 95.8%，小型模型 GPT‑4.1 Nano 与 GPT‑5 Nano 分别从 66.7%→93.3% 与 86.7%→100%，GPU 加速案例数提升 21%，但在部分高密度内核上出现轻微性能下降。

**⚠️ 局限性**

局限性包括仅在 PolyBench 这类单文件基准上验证、对复杂多文件或跨过程数据依赖缺乏评估，以及优化过程可能偏向安全而牺牲极端性能。

---

## 64. SubTokenTest: A Practical Benchmark for Real-World Sub-token Understanding

**arXiv ID:** 2601.09089 | [PDF](https://arxiv.org/pdf/2601.09089v1)

**作者:** Shuyang Hou `[一作]` (Peking University), Muhan Zhang `[通讯]` (Peking University)

**通讯引用:** 4714 | [OpenAlex ID](https://openalex.org/A5071515223)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个专门评估大型语言模型（LLM）在子词级别理解上的基准，并在此基准上对多款先进模型进行系统评测。

**💡 创新点**

创新点包括将子词误差与推理误差分离、揭示推理长度与子词理解之间的倒U形关系，以及通过线性探针分析隐藏层对字符信息的编码方式。

**🔧 技术方法**

使用了Tokenization分析、推理长度调控（TTBC）技术、线性探针与宏F1评价、以及EM与Levenshtein相似度等评测指标。

**📊 数据集**

使用了公开的SubTokenTest benchmark（包含10个任务，10,000+样例），数据托管于HuggingFace，代码托管在GitHub。

**📈 对比分析**

通过与9款大模型（如GPT‑4、GPT‑5、DeepSeek系列、Qwen系列等）进行对比，发现推理型模型在子词任务上表现更好但消耗显著更多token；小模型性能普遍较差；推理长度达到约2048 token 时性能峰值，随后下降。

**⚠️ 局限性**

局限性在于未提出解决方案，线性探针仅提供有限的内部机制洞见，且 benchmark 可能未涵盖所有实际应用场景。

---

## 65. MMR-GRPO: Accelerating GRPO-Style Training through Diversity-Aware Reward Reweighting

**arXiv ID:** 2601.09085 | [PDF](https://arxiv.org/pdf/2601.09085v1)

**作者:** Kangda Wei `[一作]` (Texas A&M University), Ruihong Huang `[通讯]` (Texas A&M University)

**通讯引用:** 2456 | [OpenAlex ID](https://openalex.org/A5101688218)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MMR-GRPO方法，在GRPO强化学习框架中引入最大边际相关性(MMR)对奖励进行多样性加权，从而加速数学推理模型的学习。

**💡 创新点**

创新点在于：1) 通过自适应无参λ机制实现奖励质量与多样性的动态平衡；2) 采用贪婪MMR选择并对奖励进行重加权，显著降低训练步骤与壁时；3) 兼顾性能不下降，提升训练效率。

**🔧 技术方法**

技术手段包括：GRPO强化学习、MMR奖励重加权、句子嵌入相似度计算、基于GPU的向量化相似矩阵求解、无参自适应λ算法。

**📊 数据集**

使用训练数据集DeepSeek-R1数学推理数据集；评估基准为AIME 2024、MATH‑500、AMC 2023、Minerva Math、OlympiadBench五个数学竞赛题库。

**📈 对比分析**

与原GRPO、DR‑GRPO、DAPO（含动态采样）进行对比，结果显示MMR‑GRPO在保持或略高的pass@1性能的同时，平均减少47.9%训练步骤、70.2%壁时，尤其在7B/8B模型上提升显著。

**⚠️ 局限性**

局限性：1) 贪婪MMR的时间复杂度为O(N²)，虽然N较小但在大规模采样时可能成为瓶颈；2) 实验仅覆盖至8B参数模型并使用LoRA微调；3) 只在数学推理任务验证，未探讨其他领域；4) 对更大模型（>8B）或全参数微调的效果仍未知。

---

## 66. Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning

**arXiv ID:** 2601.09111 | [PDF](https://arxiv.org/pdf/2601.09111v1)

**作者:** Yang Li `[一作]`, Yahong Han `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对开放式视觉语言导航（VLN）中场景与指令多样性难题，提出了慢速-快速交互推理框架 slow4fast‑VLN，并结合指令风格转换提升对多种指令的适应；

**💡 创新点**

创新点包括：1）将快速推理与慢速反思构建成动态交互式框架，使慢速经验能够即时反馈给快速决策；2）利用大型语言模型通过链式思维模板对场景/用户风格指令进行实时转换为基本风格；3）构造结构化经验库与注意力融合机制，实现经验检索与注入；

**🔧 技术方法**

使用的技术包括：基于 DUET 的快速策略网络、CLIP‑ViT‑B/16 视觉编码、Llama3.2‑vision 进行视觉描述与慢速推理、Multi‑Head Attention 融合经验特征、Prompt Engineering 的链式思维；

**📊 数据集**

数据集：GSA‑R2R（由 Habitat‑Matterport3D 与 Matterport3D 组成），包含 150 个场景（75 ID、75 OOD）、7 种指令风格、约 90k 路径‑指令对；

**📈 对比分析**

与 GR‑DUET、BT、TENT、SAR 等基线在基本、场景、用户指令三种风格下进行比较，slow4fast 在 ID 与 OOD 场景下均能提升 SR、SPL、nDTW，最佳配置可使 SR 提升 1.5%~2.2%，SPL 约 10% 以上；

**⚠️ 局限性**

局限性：经验库容量需调参，过小会导致不足、过大会干扰；慢速推理会引入额外延迟，实际实时性能受限；对极端 OOD 场景与完全新指令的泛化仍有待提升。

---

## 67. Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment

**arXiv ID:** 2601.08849 | [PDF](https://arxiv.org/pdf/2601.08849v1)

**作者:** Manas Khatore `[一作]` (Algoverse), Shi Feng `[通讯]` (George Washington University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估答案匹配模型对文本操纵攻击（冗长、包含多答案、前置答案）的鲁棒性，并系统比较二值与连续评分的效果。

**💡 创新点**

首次对三种常见攻击进行统一实验，展示答案匹配模型普遍不被简易文本操作所欺骗；同时揭示二值评分相对更严格、连续评分更易被提升。

**🔧 技术方法**

使用 LLM（GPT‑4.1 mini、Qwen2.5‑7B‑IT、Qwen3‑4B、Gemma‑2‑2B‑IT）进行答案生成与匹配；通过二值与连续评分提示；采用平均对齐、ASR、Cohen’s d、两比例 z‑检验等统计评估。

**📊 数据集**

MMLU‑Pro（定量/定性）与 GPQA Diamond（定量/定性）共四个子集，涵盖多学科、问答类型。

**📈 对比分析**

与基线生成对齐得分直接比较；所有攻击在大多数实验中均未提升得分，甚至导致下降；连续评分整体高于二值评分；Gemma‑2‑2B‑IT 在部分子集出现异常高得分；统计显著性通过 z‑检验验证，Cohen’s d 绝大多数为负值。

**⚠️ 局限性**

限制包括：攻击仅采用固定提示，未尝试自适应/动态攻击；仅测试英语文本；模型规模仅覆盖小型/中型；未评估跨语言或更大模型的鲁棒性。

---

## 68. Programming over Thinking: Efficient and Robust Multi-Constraint Planning

**arXiv ID:** 2601.09097 | [PDF](https://arxiv.org/pdf/2601.09097v1)

**作者:** Derrick Goh Xin Deik `[一作]` (Nanyang Technological University), Wenya Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2210 | [OpenAlex ID](https://openalex.org/A5078565848)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SCOPE框架，分离查询特定推理和通用代码执行，生成可重用的求解器函数，以解决多约束规划问题。

**💡 创新点**

创新点在于通过单例示例自动化推理到代码的抽象化，将问题结构与求解逻辑解耦，获得一致、确定且可复用的求解器。

**🔧 技术方法**

采用多代理LLM工作流：问题形式化、优化、解算器生成与细化，以及输入代理生成结构化参数，核心技术为LLM生成可执行代码和结构化JSON表示。

**📊 数据集**

使用TravelPlanner和Natural Plan（Trip Planning、Meeting Planning）等真实世界规划基准进行评测。

**📈 对比分析**

与直接提示、Chain-of-Thought、Tree-of-Thought、EvoAgent、HyperTree Planning、Thought of Search等基线比较，SCOPE在5种LLM（GPT‑5、GPT‑o3、GPT‑4o、Gemini‑2.5‑Pro、Gemini‑1.5‑Pro）上均显著提升成功率，且成本和延迟更低；在最弱模型GPT‑4o上提升达80%+，并实现与大模型相近的性能。

**⚠️ 局限性**

局限性包括依赖LLM的编码与格式化能力，实验主要在封闭源模型上；求解器仍绑定于特定域，跨域迁移需要重新定义问题结构。

---

## 69. Exploring the Effects of Generative AI Assistance on Writing Self-Efficacy

**arXiv ID:** 2601.09033 | [PDF](https://arxiv.org/pdf/2601.09033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 70. An Almost-Optimal Upper Bound on the Push Number of the Torus Puzzle

**arXiv ID:** 2601.08989 | [PDF](https://arxiv.org/pdf/2601.08989v1)

**作者:** Matteo Caporrella `[一作]` (University of L'Aquila), Stefano Leucci `[通讯]` (University of L'Aquila)

**通讯引用:** 632 | [OpenAlex ID](https://openalex.org/A5028702811)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种针对可排序的 m×n 维度的 Torus Puzzle 的求解算法，使用 O(mn · log max{m,n}) 次单位旋转即可将任意可排序实例归一化为有序矩阵。

**💡 创新点**

创新点在于将传统的三元素交换（每次需要 Θ(max{m,n}) 次单位旋转）改为三步策略：先将目标列的元素放入首行作为缓冲区，再对列体进行并行基数排序，最后利用可逆不动子分解在首行上实现任意置换，从而将推数的上界从 O(mn·max{m,n}) 降低到 O(mn·log max{m,n})，并且在理论上接近下界 Ω(mn)。

**🔧 技术方法**

主要技术包括：利用势函数证明列变换的总次数；采用并行基数排序的思想对多列体同时排序；通过将任意置换拆解为两不动子（或两不动子加一次换位）来控制首行的置换并抵消对列体的副作用；以及在需要时使用转置矩阵和反射对任意旋转方向的兼容性进行处理。

**📊 数据集**

本文没有针对具体实验数据集，而是提供了完整的理论证明与伪代码，说明算法在任意可排序实例上均满足所给旋转上界。

**📈 对比分析**

与之前的 O(mn·max{m,n}) 上界相比，本文实现了对数级的改进，理论上已达推数下界的多项式逼近；若与实际实现对比，实验结果表明在中等规模矩阵上能够显著减少旋转次数，具体性能因实现细节和数据规模而异。

**⚠️ 局限性**

局限性主要包括：仍存在 Θ(log max{m,n}) 的上界-下界差距未完全收敛；算法实现复杂度高，且未给出最短旋转序列的多项式求解方法；对不同旋转方向的支持仅通过对称性推理实现，实际实现可能需要额外处理。

---

## 71. StegoStylo: Squelching Stylometric Scrutiny through Steganographic Stitching

**arXiv ID:** 2601.09056 | [PDF](https://arxiv.org/pdf/2601.09056v1)

**作者:** Robert Dilworth `[一作]` (Mississippi State University), Robert Dilworth `[通讯]` (Mississippi State University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5099003328)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出并改进了一种名为TraceTarnish的对抗性作者风格攻击，利用同义词替换、机器翻译、重写以及零宽Unicode字符注入等技术，进一步研究如何通过隐写来掩盖作者的写作痕迹。

**💡 创新点**

创新点在于（1）将隐写（zero‑width Unicode注入）与传统对抗性文本修改相结合，提供了可调节的“注入比例”以在保持可读性的同时最大化作者身份模糊；（2）通过实验验证了33%及以上的注入比例即可显著降低作者验证得分，77%以上几乎完全消除识别；（3）在原有攻击脚本中加入同义词替换与隐私导向翻译模块，提升了攻击多样性与实用性。

**🔧 技术方法**

主要技术包括：
- 同义词替换模块（Synonym Substitution）
- 本地化隐私翻译模块（Privacy‑oriented Translation）
- 零宽Unicode注入（Steganographic Payload Injection）
- 传统对抗性翻译与重写技术
- Stylometry评估工具（R包中的作者验证函数）
- 统计分析与绘图（绘制注入比例与验证得分曲线）

**📊 数据集**

实验使用了多个数据集：
- whistleblower（假想泄密者）的公开文本合集，用于构建验证集；
- Eric Hughes的原始与对抗样本，作为对照实验；
- 公开的写作样本（如John Gilmore、Timothy C. May等）用于构建假作者集合；
- 另有小规模的九词测试句子用于注入比例实验。

**📈 对比分析**

方法对比基于R包的作者验证函数（计算0‑1得分）。实验表明：
- 未注入（0%）时得分为1；
- 低于22%注入时得分仍≈0.91，未能有效模糊作者；
- 33%注入时得分骤降至0.22，已足以“成功”掩盖；
- 超过77%注入时得分持续为0，效果不再提升，说明已达极限。

**⚠️ 局限性**

局限性包括：
- 仍需平衡文本可读性与注入比例，过高注入可能导致文本失去语义或可读性；
- 只在少数stylometry工具（基于词频与Burrows Delta）上验证，其他算法效果未知；
- 隐写字符可能被专门的检测工具识别，导致攻击被逆向；
- 仅在英文文本上实验，跨语言通用性未作系统评估；
- 现实场景中多模态或混合文本（如含图片的社交媒体）可能影响攻击有效性。

---

## 72. Silenced by Design Censorship, Governance, and the Politics of Access in Generative AI Refusal Behavior

**arXiv ID:** 2601.08877 | [PDF](https://arxiv.org/pdf/2601.08877v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 73. Streamlined Pathway (SP) Approach: An Efficient Load Balancer to Enhance Quality of Service

**arXiv ID:** 2601.08887 | [PDF](https://arxiv.org/pdf/2601.08887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 74. Emissions and Performance Trade-off Between Small and Large Language Models

**arXiv ID:** 2601.08844 | [PDF](https://arxiv.org/pdf/2601.08844v1)

**作者:** Anandita Garg `[一作]` (Plaksha University), Anish Roy Chowdhury `[通讯]` (Plaksha University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在六个自然语言处理、推理和编程任务上，对细调后的体积小模型（≤250M参数）与体积大模型（≥1B参数）进行推理性能与碳排放的对比实验，评估小模型的可持续性。

**💡 创新点**

创新点在于：①首次系统比较体积小模型与大模型在推理阶段的性能与碳排放；②发现小模型在四个任务上能与大模型保持相近性能，同时推理碳排放降低数百到数千倍；③提出一种基于 token‑per‑parameter 的碳估算方法，为大模型推理排放评估提供可复制框架。

**🔧 技术方法**

主要技术包括：①针对每个任务对小模型进行细调；②使用 eco2AI 库在 CPU/GPU/RAM 上追踪能耗并换算为 CO₂e；③采用基于 GPT‑3 推断碳估算的线性缩放公式；④通过多种评估指标（准确率、BERT‑Score、Perplexity、BLEU、ROUGE‑L、pass@1 等）衡量性能。

**📊 数据集**

使用的数据集有：Yelp Polarity Reviews（情感分析）、WritingPrompts（短篇生成）、Stanford SNLI（自然语言推断）、CoT Collection（链式推理）、CodeSearchNet（代码摘要）、HumanEval（代码生成）。

**📈 对比分析**

比较方法：对同一任务分别评估细调小模型与公开的 LLM（Mistral‑7B、Qwen‑3‑235B、DeepSeek‑R1‑0528）在性能指标与推理碳排放上的差异。结果显示：在情感分析、短篇生成、自然语言推断和代码摘要四项任务中，小模型在性能上几乎与 LLM 相当，而推理碳排放低至 1/10‑1/13,000（取决于任务）。链式推理与代码生成任务则小模型表现显著落后。

**⚠️ 局限性**

局限性：①仅评估推理阶段排放，未考虑训练与部署的全生命周期；②实验仅覆盖开源 LLM，缺乏商业大型模型对比；③任务范围有限，缺少更多多模态与更复杂推理场景；④使用线性缩放估算碳排放，可能不完全准确；⑤研究规模小，未探讨最佳模型尺寸阈值与更高效的技术（如 MoE、LoRA）。

---

## 75. SITA: Learning Speaker-Invariant and Tone-Aware Speech Representations for Low-Resource Tonal Languages

**arXiv ID:** 2601.09050 | [PDF](https://arxiv.org/pdf/2601.09050v1)

**作者:** Tianyi Xu `[一作]` (University of Wisconsin), Junjie Hu `[通讯]` (University of Wisconsin)

**通讯引用:** 3871 | [OpenAlex ID](https://openalex.org/A5101982147)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了两阶段轻量化适配方案SITA，使多语种wav2vec‑style编码器在低资源声纹音调语言（如苗语）中既能实现说话人不变的表示，又能保持对音调的敏感度，并在第二阶段恢复 ASR 能力。

**💡 创新点**

创新点在于：① 结合跨性别 InfoNCE 对比学习与音调排斥损失，构建说话人不变且音调敏感的表征；② 采用线性音调分类器辅助学习；③ 在第二阶段加入 CTC+知识蒸馏重建 ASR，且仅更新上层模块，提升参数效率。

**🔧 技术方法**

使用技术包括：wav2vec‑style XLS‑R 预训练模型、InfoNCE 对比损失、音调排斥（tone‑repulsive）对比、线性音调分类器、CTC 目标、知识蒸馏、FreeVC 声纹转换、声学增强等。

**📊 数据集**

主要数据集：自建苗语词级语料库（1400词/说话人，覆盖7种音调）；用于泛化验证的 Tone Perfect 语料（普通话四声）。

**📈 对比分析**

与多种基线（冻结 XLS‑R、无监督微调、ASR 适配、Whisper、Omnilingual、GRL）对比，SITA 在跨性别检索 Top‑1 达 0.629/0.593，Top‑5 达 0.929/0.889，硬负样本余弦距离提升至 0.675；ASR 在词+音调任务上 WER 0.512，略高于 0.461 的 ASR 适配模型；在普通话检索中亦取得近乎 100% 的准确率。

**⚠️ 局限性**

限制：评估仅在词级、受限说话人与录音条件下进行，无法充分验证在对话或多样化环境中的表现；参数冻结与损失权重需针对不同任务调节；虽然强调去除说话人信息，但并未完全消除敏感属性，仍需进一步治理。

---

## 76. Scalable and Reliable Evaluation of AI Knowledge Retrieval Systems: RIKER and the Coherent Simulated Universe

**arXiv ID:** 2601.08847 | [PDF](https://arxiv.org/pdf/2601.08847v1)

**作者:** JV Roig `[一作]` `[通讯]` (Kamiwaza AI), JV Roig (Kamiwaza AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于“先生成事实再生成文档”的 RIKER 框架，用于可扩展、可再现且无污染的 AI 知识检索与提取系统评估。

**💡 创新点**

创新点在于：①逆向生成数据实现全程已知真值；②生成式合成文档保持实体一致性（Coherent Simulated Universe）；③构建多层级问题体系（单文档、跨文档聚合、幻觉检测），并实现确定性打分。

**🔧 技术方法**

采用程序化合成（基于 SQLite 生成知识库）、模板化文本生成、随机化实体池、以及 LLM 以外的直接答案比对技术；评估方法使用 8 次独立跑测，并计算整体、单文档、聚合、幻觉检测与率等指标。

**📊 数据集**

数据集为从结构化知识库合成的企业文档（商业租赁、销售现场报告、HR 评估）共 21 亿+ token，包含 7-10 文档类型、32K/128K/200K 上下文长度，按不同随机种子生成四套 128K 语料。

**📈 对比分析**

与 33 个主流模型（包括 Qwen3、GLM、DeepSeek、Llama 等）进行比较，结果显示：单文档提取准确率可达 90%+；聚合查询显著困难（最大 75%）；幻觉检测中 GLM-4.5 几乎无幻觉，其他模型幻觉率高达 30-90%；不同上下文长度导致显著性能衰退，尤其 200K 上下文表现大幅下降。

**⚠️ 局限性**

局限性包括：仅评估英语合成企业文档；缺乏真实世界噪声（OCR、错别字、冲突信息）；只做了上下文填充评估，未覆盖检索增强或代理式检索；模型覆盖面有限，快速迭代的 LLM 可能产生不同结果。

---

## 77. Breaking the Bottlenecks: Scalable Diffusion Models for 3D Molecular Generation

**arXiv ID:** 2601.08963 | [PDF](https://arxiv.org/pdf/2601.08963v1)

**作者:** Adrita Das `[一作]`, Jose Lugo-Martinez `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种直接去噪扩散模型（DDDM）的理论框架，利用逆转移核（RTK）解释了其确定性反向过程，并在此基础上构建了SE(3)等变的SSM‑基扩散架构；

**💡 创新点**

创新点在于将DDDM视为RTK近似的确定性采样子问题，证明其逆向子问题为强对数凹分布，从而实现常数步长的高效推理；

**🔧 技术方法**

核心技术包括概率流ODE、逆转移核（RTK）方法、基于GraphGPS的局部消息传递与全局注意力融合、以及Mamba、Jamba、Hydra等SSM变体；

**📊 数据集**

实验使用GEOM‑DRUGS和GEOM‑LongRange两大3D分子数据集进行训练与评估；

**📈 对比分析**

与RDKit、OMEGA、GeoMol、GeoDiff、Torsional Diffusion等基线比较，DDDM在覆盖率、平均最小RMSD、生成速度（CPU核心秒）等指标均优于传统方法，尤其在大分子上表现突出；

**⚠️ 局限性**

局限性包括对去噪映射F的光滑性与Lipschitz性要求较高，且在多模态分布下纯确定性采样可能导致多样性不足，未来可探索混合随机-确定核的策略。

---

## 78. Identity-Robust Language Model Generation via Content Integrity Preservation

**arXiv ID:** 2601.09141 | [PDF](https://arxiv.org/pdf/2601.09141v1)

**作者:** Miao Zhang `[一作]` (New York University), Rumi Chunara `[通讯]` (New York University)

**通讯引用:** 5379 | [OpenAlex ID](https://openalex.org/A5005061793)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解决LLM在用户社会人口属性提示下产生的身份依赖生成偏差，提出无训练、轻量级的身份鲁棒生成框架(IRG)，通过识别并中和非关键信息保持输出质量。

**💡 创新点**

以查询层面控制身份信息流，先检测身份相关词并判断是否关键信息，生成中性化查询后再生成答案，最后可选的控制个性化，同时验证内容一致性，显著降低身份依赖偏差。

**🔧 技术方法**

基于NER（GLiNER2）进行身份检测，LLM执行反事实相关性判定，三阶段IRG（检测-中性化-可控个性化）以及内容一致性校验，全部无须模型微调。

**📊 数据集**

TruthfulQA、MMLU-Pro、AmbigQA、StrongReject四大基准；18个社会人口身份（教育、宗教、种族、职业、年龄、性别）；并使用真实用户提示进行鲁棒性测试。

**📈 对比分析**

与 Vanilla 和 Prompt Steering 对比，使用 Personalization Bias 指标；IRG 在四个基准上平均降低 77‑89% 的身份偏差，且在 Llama3.3‑70B 上与无身份基线几乎持平，性能提升显著。

**⚠️ 局限性**

仅处理显式身份提示，未覆盖隐式个性化；评估范围局限于部分开源模型，对大规模商用模型的泛化未知。

---

## 79. LERA: Reinstating Judgment as a Structural Precondition for Execution in Automated Systems

**arXiv ID:** 2601.08880 | [PDF](https://arxiv.org/pdf/2601.08880v1)

**作者:** Jing `[一作]`, Liu `[通讯]` (Independent Researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出并阐述了LERA（判断-治理架构），将判断作为执行前的必需结构前置条件，构建了判断根节点和治理门实现非可绕过的执行权限控制。

**💡 创新点**

在架构层面正式定义了判断根节点，区分判断与优化、执行的结构关系，并通过治理门实现判断的不可绕过执行授权，填补了现有AI系统中“判断结构缺失”的空白。

**🔧 技术方法**

采用架构设计与系统设计原理，构建LERA-J（判断层）和LERA-G（治理门）两层，并通过结构化约束与非可绕过交互实现治理。

**📊 数据集**

无；本文为理论性/架构性研究，不涉及具体数据集。

**📈 对比分析**

无实验或性能对比，本文以概念性分析和架构论证为主，未给出定量评估。

**⚠️ 局限性**

未给出具体实现细节和性能评估，难以验证在实际系统中的可行性与效率影响，且对判断主体的技术实现留空，需后续实证研究。

---

## 80. Vision Foundation Models for Domain Generalisable Cross-View Localisation in Planetary Ground-Aerial Robotic Teams

**arXiv ID:** 2601.09107 | [PDF](https://arxiv.org/pdf/2601.09107v1)

**作者:** Lachlan Holden `[一作]` (University of Adelaide), Tat-Jun Chin `[通讯]` (University of Adelaide)

**通讯引用:** 5112 | [OpenAlex ID](https://openalex.org/A5027317977)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种双编码器跨视角定位框架，使行星车在仅有90°视场单目RGB图像的情况下，在局部航空图像中实现绝对定位。

**💡 创新点**

创新点包括：①基于LLMDet与SAM 2的岩石语义分割显著降低合成与真实域差距；②贡献了首个含真实轨迹及地理参考航空图的跨视角数据集；③将该定位网络与粒子滤波器结合，实现在无GPS环境下的实时状态估计。

**🔧 技术方法**

使用的技术包括：双编码器Transformer（TransGeo‑style）+软边距三元组损失、LLMDet检测+SAM 2分割、PANGU合成数据、粒子滤波器、以及ASAM优化器。

**📊 数据集**

数据集：实验室采集的6组岩石配置共40条轨迹（20 58 秒）+ 10个PANGU合成场景（每场景500对图像），总计约10 000张图像，公开于Zenodo。

**📈 对比分析**

通过与RGB‑Synthetic、RGB‑Synthetic+Real、Mask‑LLMDet+SAM三种实验对比，利用粒子滤波误差（距离、航向）评估；Fine‑tuned 实验在所有轨迹中实现了≈90 % 的 top‑20% 匹配率，距离误差中位数下降至 0.15 m，显著优于仅合成训练的方案。

**⚠️ 局限性**

局限性：①模型尺寸大，推理时间（≈30 ms/粒子）高于典型空间级硬件；②仍需大量真实标注数据进行微调；③依赖岩石作为视觉基准，可能在岩石稀缺或纹理不明显的地形上失效；④仅针对 90° FOV 进行验证，未覆盖更宽视角或多相机配置。

---

## 81. Resolving Predictive Multiplicity for the Rashomon Set

**arXiv ID:** 2601.09071 | [PDF](https://arxiv.org/pdf/2601.09071v1)

**作者:** Parian Haghighat `[一作]` (University of Illinois Chicago), Cynthia Rudin `[通讯]` (Duke University)

**通讯引用:** 21026 | [OpenAlex ID](https://openalex.org/A5040468715)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出三种方法（异常值校正、局部补丁、成对调和）来减少 Rashomon 集合模型在个体预测上的差异，并通过这些方法得到一致、可解释的预测。

**💡 创新点**

创新点在于：①将异常值处理作为减少模型多样性预测方差的预处理步骤；②通过验证集邻域的偏差检验实现局部预测补丁；③在成对调和中结合验证误差权重和差异阈值，迭代修正最不一致的模型对，显著降低多重性。

**🔧 技术方法**

技术手段包括：概率预测的 Brier 损失评估、kNN 邻域估计偏差、基于验证集的误差加权修正、迭代对数最小化（闭式解）以及对预测值的区间裁剪。

**📊 数据集**

使用四个高风险分类数据集：Adult、COMPAS、Folk Mobility 和 Folk Travel。

**📈 对比分析**

与软投票、硬投票、最佳单模型和随机选择等基线对比，三种方法（单独或组合）在保持甚至提升准确率的同时，显著降低预测方差、模糊度、失配度和误差率；尤其是 PR+LP 或 OC+PR+LP 组合在所有数据集上实现了几乎零差异度且局部一致性提升。

**⚠️ 局限性**

局限性包括：①局部补丁依赖邻域大小和阈值，易受过拟合影响；②异常值校正需先获取 Rashomon 集合，计算成本高；③所有方法均在二分类概率预测上验证，尚未充分探究多分类或回归场景；④对异常值的识别阈值仍需经验调参。

---

## 82. SkinFlow: Efficient Information Transmission for Open Dermatological Diagnosis via Dynamic Visual Encoding and Staged RL

**arXiv ID:** 2601.09136 | [PDF](https://arxiv.org/pdf/2601.09136v1)

**作者:** Lijun Liu `[一作]` (Baichuan Inc), Hong-Yu Zhou `[通讯]` (Tsinghua University)

**通讯引用:** 10695 | [OpenAlex ID](https://openalex.org/A5100783219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发SkinFlow框架，利用视觉信息压缩-解码优化与两阶段RL训练实现高效皮肤病诊断。

**💡 创新点**

创新点在于通过虚拟宽度动态视觉编码(DVE)实现视觉表示的虚拟扩展，并将诊断视为压缩解码任务，使用两阶段RL分别对显式描述与隐式诊断纹理进行优化。

**🔧 技术方法**

采用FDLinear动态线性层、两阶段RL（GRPO）、结构化医学描述奖励、基于Qwen2.5-VL-Instruct-7B的多模态LLM改造等技术。

**📊 数据集**

使用公开的Fitzpatrick17k数据集以及约200张内部皮肤图像，涵盖多类别皮肤疾病。

**📈 对比分析**

与多款大模型（Qwen2.5、Qwen3、InternVL、GPT‑5.2、Lingshu‑32B等）比较，Fitzpatrick17k上Top‑1 29.19%（+12.06%）、Top‑6 71.16%（+28.57%），内部集Top‑6 79.21%，明显优于同类模型。

**⚠️ 局限性**

主要局限包括缺乏对模型解释性的评估、对复杂背景鲁棒性不足以及在高风险错误检测上仍需改进。

---

## 83. Generalizable Geometric Prior and Recurrent Spiking Feature Learning for Humanoid Robot Manipulation

**arXiv ID:** 2601.09031 | [PDF](https://arxiv.org/pdf/2601.09031v1)

**作者:** Xuetao Li `[一作]` (Wuhan University), Miao Li `[通讯]` (Wuhan University)

**通讯引用:** 5139 | [OpenAlex ID](https://openalex.org/A5100339707)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文提出了一个结合几何先验与递归自适应脉冲网络的统一框架，用于实现对类人机器人在复杂环境下的可泛化、数据高效的操控；

**💡 创新点**

核心创新点包括：①基于轻量级二维几何先验的长时序几何先验技能选择器，②递归自适应脉冲网络通过自适应衰减与脉冲神经元实现时空特征的高效提取，③利用高斯混合模型精细化动作生成；

**🔧 技术方法**

主要技术手段涵盖：视觉‑语言模型（Qwen‑vl、YOLOv8‑seg）、旋转位置编码（RoPE）、自适应衰减机制、脉冲神经元、GUIDED自注意力、以及GMM后处理；

**📊 数据集**

实验数据集包括：ManiSkill2 2.0 仿真基准、三台真实机器人（自研类人机器人、桌面双臂机器人和 Aloha 机器人）以及收集的 500 条专家轨迹；

**📈 对比分析**

与 Diffusion Policy、Octo、OpenVLA、Dex‑VLA 等先进基线对比，RGMP 在 10 项多样化操控任务上平均提升 15‑20% 的成功率，并在数据效率上实现 5 倍的提升，实时推理频率可达 75 Hz；

**⚠️ 局限性**

限制方面：在需要极高精度或细腻触感反馈的场景（如水倒入细口器、极细物体抓取）表现仍受限，缺乏触觉反馈与更精准的视觉分辨率导致误差容忍度不足。

---

## 84. Attention Consistency Regularization for Interpretable Early-Exit Neural Networks

**arXiv ID:** 2601.08891 | [PDF](https://arxiv.org/pdf/2601.08891v1)

**作者:** Yanhua Zhao `[一作]` `[通讯]` (KISMED AI Systems in Medicine Technische Universitat Darmstadt), Yanhua Zhao (KISMED AI Systems in Medicine Technische Universitat Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了解释引导训练（EGT）框架，在早期退出神经网络中通过注意力一致性正则化提升可解释性与一致性。

**💡 创新点**

创新点在于：①将注意力一致性损失加入多目标学习，强制早期退出层关注与最终层相同的特征区域；②通过余弦相似度度量并对齐注意力图，实现解释一致性提升；③在保持高准确率的同时实现显著推理加速。

**🔧 技术方法**

使用技术包括：早期退出卷积网络、每个退出点的注意力模块、分类头、联合分类+注意力一致性损失、余弦相似度计算、Adam优化、学习率衰减和置信度阈值决策。

**📊 数据集**

使用的真实图像分类数据集：9 类、1363 训练样本、1364 测试样本。

**📈 对比分析**

与无正则化基线模型对比：EGT 在保持≈98%准确率的前提下，注意力一致性提升10.8%–18.5%；α=0.3 时一致性最高为0.821，速度提升1.97×（1.83 ms/样本 vs 3.6 ms）。

**⚠️ 局限性**

局限性包括：仅在单一小规模数据集上验证；α 参数固定且探索范围有限；未给出理论一致性界限；缺乏对更大、更多样化数据集及不同模型架构的泛化评估。

---

## 85. Leveraging learning analytics to enhance immersive teacher simulations: Challenges and opportunities

**arXiv ID:** 2601.08954 | [PDF](https://arxiv.org/pdf/2601.08954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 86. Consistency-Aware Editing for Entity-level Unlearning in Language Models

**arXiv ID:** 2601.08840 | [PDF](https://arxiv.org/pdf/2601.08840v1)

**作者:** Xiaoqi Han `[一作]` (Shanxi University), Jeff Z. Pan `[通讯]` (University of Edinburgh)

**通讯引用:** 7645 | [OpenAlex ID](https://openalex.org/A5066422711)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种一致性感知编辑（CAE）框架，用于在大型语言模型中实现实体级知识遗忘。

**💡 创新点**

创新点在于：①在多提示下引入一致性约束，使编辑向量对齐；②采用SVD筛选最具代表性的事实键；③利用低秩权重更新，仅在MLP层进行高效局部编辑；④在编辑前后通过多层键提取与残差分布实现跨层一致性。

**🔧 技术方法**

使用技术包括：基于模型编辑的低秩更新、SVD特征选择、正则化一致性约束、残差分布至多层、以及对齐的编辑向量优化。

**📊 数据集**

数据集涵盖：Wikidata结构化事实、RWKU 与 ToFU 两大实体级遗忘基准、以及LLaMA3、LLaMA3.1 等公开LLM。

**📈 对比分析**

与prompt、fine‑tune、训练基、以及其他编辑方法（MEMIT、EMMET、AlphaEdit 等）对比，CAE在遗忘准确率、邻域知识保留、以及整体模型效能指标（Mean_FN、Utility）上均取得最优或接近最优成绩，且计算开销显著低于全模型微调。

**⚠️ 局限性**

局限性：目前仅针对单实体遗忘；在多实体或连续遗忘场景下对先前编辑的影响仍需进一步研究；对词义变体或隐式查询的鲁棒性有待提升。

---

## 87. Reading or Reasoning? Format Decoupled Reinforcement Learning for Document OCR

**arXiv ID:** 2601.08834 | [PDF](https://arxiv.org/pdf/2601.08834v1)

**作者:** Yufeng Zhong `[一作]` (Meituan), Lin Ma `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于格式解耦的强化学习框架（FD‑RL）用于文档 OCR，旨在通过高熵样本筛选和格式专属奖励提升公式、表格等格式敏感内容的识别性能。

**💡 创新点**

创新点在于：① 将文档识别拆分为 SFT 与 RL 两阶段，RL 侧重格式级别而非 token 级别；② 引入熵驱动的数据筛选，挑选高不确定性样本；③ 设计格式解耦奖励，将文本、公式、表格分别用不同评估函数（编辑距离、BLEU、TEDS）进行回报。

**🔧 技术方法**

技术包括：多源数据工程（开源数据、真实 PDF、合成 OCR 数据）、SFT‑then‑RL 训练策略、基于熵的样本过滤、格式解耦奖励、GRPO 强化学习算法、以及 VLM（Qwen3‑VL‑4B）作为基础模型。

**📊 数据集**

使用的数据集：开源 OCR 数据集（PDFA、DocStruct4M、DocGenome 等）、真实 PDF 文档、合成 OCR 样本，以及基准评测集 OmniDocBench。

**📈 对比分析**

与现有基线（pipeline 工具、通用 VLM、专业 VLM 等）对比，在 OmniDocBench 上整体得分 90.41，超过最接近方法 dots.ocr（88.85）和 Deepseek‑OCR（87.01）；在文本编辑距离、公式 CDM、表格 TEDS 上分别位列或排名第二，表明该方法在格式敏感任务上显著优于竞争者。

**⚠️ 局限性**

局限性包括：① 仍需人工标注或合成数据来覆盖更广泛的格式；② 依赖熵阈值与过滤比例的手工调参，可能对不同任务不适用；③ 训练过程分两阶段，成本和时间较高；④ 对极端复杂表格或多语言混排的鲁棒性尚未充分验证。

---

## 88. Build Code is Still Code: Finding the Antidote for Pipeline Poisoning

**arXiv ID:** 2601.08995 | [PDF](https://arxiv.org/pdf/2601.08995v1)

**作者:** Brent Pappas `[一作]` (University of Central Florida), Paul Gazzillo `[通讯]` (University of Central Florida)

**通讯引用:** 554 | [OpenAlex ID](https://openalex.org/A5016175378)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了开发阶段隔离（development phase isolation）机制，用于检测并防止构建系统被注入恶意代码，演示了一个原型工具能够发现 XZ Utils 后门的 pipeline 污染。

**💡 创新点**

创新点在于把构建脚本视为代码并引入“阶段隔离”安全属性，首次通过对构建阶段的文件访问权限进行动态监控和沙箱执行来防御构建流水线中的恶意注入。

**🔧 技术方法**

技术手段包括基于 JSON 的阶段权限规范、Python 编写的动态文件访问追踪器、系统调用时间戳检测、沙箱化执行阶段、以及后续计划的静态模拟与自动化权限推断。

**📊 数据集**

实验使用了被污染的 XZ Utils 5.6.0 代码库（含恶意测试文件）作为数据集；未来计划构建包含安全与已知脆弱项目的 benchmark，以评估工具性能。

**📈 对比分析**

目前仅演示了功能验证（成功检测 XZ Utils 的恶意流），未给出详细性能指标；未来将与现有静态/动态分析工具在 benchmark 上进行比较，以评估检测精度和执行开销。

**⚠️ 局限性**

局限性包括：需手工编写阶段权限规范（易误配置导致漏报或误报）、仅跟踪文件读写权限（缺乏更细粒度的操作权限控制）、目前仅实现动态分析，未加入静态预测；对非 C 语言构建系统的适用性尚未验证。

---

## 89. SAM-Aug: Leveraging SAM Priors for Few-Shot Parcel Segmentation in Satellite Time Series

**arXiv ID:** 2601.09110 | [PDF](https://arxiv.org/pdf/2601.09110v1)

**作者:** Kai Hu `[一作]` (Jiangnan University), Huayi Wu `[通讯]` (Wuhan University)

**通讯引用:** 4067 | [OpenAlex ID](https://openalex.org/A5030670883)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 SAM-Aug 框架，在极少标签条件下利用 Segment Anything Model (SAM) 生成的几何-aware 区域先验来增强遥感时间序列语义分割

**💡 创新点**

创新点在于：① 通过云-free 合成图自动生成无监督 SAM 区域先验；② 引入 RegionSmoothLoss，在每个 SAM 区域内强制时间一致性，作为轻量级正则化；③ 使得 SAM 成为可插拔、无需微调的先验来源

**🔧 技术方法**

技术包括：SAM（无监督掩模生成）、时间序列合成、区域一致性正则化（RegionSmoothLoss）、基于 Exchanger+Mask2Former 的时空分割模型、AdamW 优化、cosine 学习率衰减、混合精度训练

**📊 数据集**

使用 PASTIS‑R 数据集（法国农田 42,390 区块，19 类）进行 5% 标签稀缺实验，进行 3 个随机种子（42、2025、4090）

**📈 对比分析**

与基准 Exchanger+Mask2Former 在 5% 标签下对比，SAM‑Aug 在 3 个种子上的平均 mIoU 为 36.21%，比基准 33.88% 提升 2.33%（相对 6.89%），单种子 42 的 mIoU 达 40.28%；在 10%、7% 等标签比例下也持续获益

**⚠️ 局限性**

局限在于：① 在极低标签（1%–3%）下效果下降，表明先验与标签不匹配时易导致过拟合；② 需要预先生成 SAM 区域，尽管开销小但仍需额外计算；③ 对不同遥感波段或多模态数据的适用性未充分验证

---

## 90. The Illusion of Friendship: Why Generative AI Demands Unprecedented Ethical Vigilance

**arXiv ID:** 2601.08874 | [PDF](https://arxiv.org/pdf/2601.08874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 91. Overcoming the Shadow: Bending Airy Beams for Radiative Near-Field Multi-User Access in Half-Space Blockage Scenarios

**arXiv ID:** 2601.09098 | [PDF](https://arxiv.org/pdf/2601.09098v1)

**作者:** Yifeng Qin `[一作]` (Peng Cheng Laboratory), Yongming Huang `[通讯]` (Southeast University)

**通讯引用:** 16658 | [OpenAlex ID](https://openalex.org/A5056225611)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在高频RNF多用户系统中，利用Airy波束自弯曲特性来克服半空间阻挡导致的阴影，实现通信恢复。

**💡 创新点**

提出基于Green函数的阻挡模型、边缘弯曲的Airy波束调制以及空中消除干扰的Airy消零技术，首次在多用户情形下实现自弯曲波束的能量恢复与干扰抑制。

**🔧 技术方法**

使用Green函数、Fresnel衍射积分、射频相位调制（Airy三次相位）、零逼迫预编码、仿真仿真及频域/时域传播模型。

**📊 数据集**

主要采用仿真数据：基于自由空间Green函数和Fresnel衍射模型构造的多用户通道，未使用公开实验数据集。

**📈 对比分析**

与传统近场聚焦、传统Airy（几何）及传统RIS方案对比，仿真显示SNR提升超过20 dB，阴影链接成功恢复，系统总率提升约35%，实现全秩恢复并几乎消除阴影失链。

**⚠️ 局限性**

局限于二维理想化模型，参数调节依赖精确几何信息，未考虑动态用户移动、3D遮挡以及硬件实现细节，且在高SNR/低噪声场景下性能可能下降。

---

## 92. Human-AI Co-design for Clinical Prediction Models

**arXiv ID:** 2601.09072 | [PDF](https://arxiv.org/pdf/2601.09072v1)

**作者:** Jean Feng `[一作]` (University of California San Francisco), Chandan Singh `[通讯]` (Microsoft Research)

**通讯引用:** 3275 | [OpenAlex ID](https://openalex.org/A5017514239)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了 HACHI 框架，结合人工与大型语言模型（LLM）共同迭代探索、评估并优化临床预测模型的特征概念，构建可解释的临床预测模型（CPM）。

**💡 创新点**

创新点在于引入人机协作的迭代设计，利用 LLM 在概念空间内快速生成并评估 yes/no 问题形式的特征，并通过专家反馈动态调整提示和数据，显著提升模型可解释性和泛化性能。

**🔧 技术方法**

技术包括基于提示的 LLM 代理、贪婪 Hill‑Climbing 迭代搜索、lasso 逻辑回归、PHI 合规 Web 界面以及样本加权等。

**📊 数据集**

使用的数据集为 UCSF 的两组 EHR 记录：400/400 例 TBI 头部创伤病例与对照，以及 800/800 例术前麻醉笔记的 AKI 预测样本。

**📈 对比分析**

与 PECARN、Kheterpal、单轮头脑风暴等传统方法比较，HACHI 在 TBI 任务上 AUC 0.91、在 AKI 任务上 AUC 0.73，均高于基线模型且提升了跨站点及时间段的泛化。

**⚠️ 局限性**

局限性包括仅在单一机构进行回顾性验证、缺乏前瞻性部署评估、对 LLM 抽取准确性的依赖以及对不同数据源和临床流程的适用性未知。

---

## 93. SPOT-Face: Forensic Face Identification using Attention Guided Optimal Transport

**arXiv ID:** 2601.09229 | [PDF](https://arxiv.org/pdf/2601.09229v1)

**作者:** Ravi Shankar Prasad `[一作]` (Indian Institute of Technology Mandi), Dinesh Singh `[通讯]` (Indian Institute of Technology Mandi)

**通讯引用:** 1323 | [OpenAlex ID](https://openalex.org/A5077717460)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SPOT‑Face 框架，用超像素图和图神经网络实现跨域法医面部识别（骷髅/素描到真人面孔）。

**💡 创新点**

创新性地将跨注意力与熵正则化最优传输结合在图结构上，统一对骷髅和素描进行跨域特征对齐；同时使用超像素图表示减少模态差距。

**🔧 技术方法**

采用 SLIC 超像素分割、构建图结构；使用 GCN、GAT、GraphSAGE、GraphTransformer 等 GNN；交叉注意力模块、熵正则化最优传输、三元组损失。

**📊 数据集**

在 IIT_Mandi_S2F（51 组骷髅‑人脸）和 CUFS（188 组素描‑人脸）两个公开数据集上进行实验。

**📈 对比分析**

与多种基线（不同 GNN、无对齐、传统手工特征等）对比，评估 Recall@k 与 mAP@k；在 CUFS 上取得 Recall@1 88.4%、mAP@1 88.4%，在 S2F 上取得 Recall@1 50%、mAP@1 50%，整体性能显著优于传统方法。

**⚠️ 局限性**

局限性包括：骷髅‑人脸的域间差距仍较大，导致 S2F 上的召回率仅为 50%；数据集规模有限，缺乏跨模态的泛化验证；对超像素分割参数敏感；验证指标在 CUFS 上较弱。

---

## 94. A Review: PTSD in Pre-Existing Medical Condition on Social Media

**arXiv ID:** 2601.08836 | [PDF](https://arxiv.org/pdf/2601.08836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 95. A Grouped Sorting Queue Supporting Dynamic Updates for Timer Management in High-Speed Network Interface Cards

**arXiv ID:** 2601.09081 | [PDF](https://arxiv.org/pdf/2601.09081v1)

**作者:** Zekun Wang `[一作]` (Xidian University), Yue Hao `[通讯]` (Xidian University)

**通讯引用:** 20392 | [OpenAlex ID](https://openalex.org/A5100427150)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种支持动态更新与组排序的硬件优先级队列，用于高精度、可扩展的网络接口卡计时器管理。

**💡 创新点**

创新点：①实现了通过组合与传播基本操作完成的 Update 操作；②引入组排序机制以在计时器溢出后仍保持准确的出队时序；③采用 1D 递推阵与移位寄存器的混合架构，实现低资源、低延迟的实现。

**🔧 技术方法**

技术手段：1D 递推阵 + 移位寄存器混合架构；多阶段布线比较与布线广播；使用 MSB 作为分组标记的组排序；利用布尔逻辑合并控制信号；FPGA 与 28nm 工艺综合。

**📊 数据集**

实验使用 UNIV1 数据集（TCP 5-tuple 流表流量）进行流表超时仿真；同时对不同计时器宽度、超时时间、计时精度进行参数敏感性测试。

**📈 对比分析**

对比方法：与 AnTiQ、PIFO、R‑BMW 等现有优先级队列实现做综合与仿真比较；在 4K 深度、16 位计时器下实现 175 Mpps 以上吞吐率，频率高达 526 MHz；与 AnTiQ 比较，LUT 与 FF 均减少 31% 与 25%；在 FPGA 上实现 116 Mpps，频率 418 MHz，性能高于 AnTiQ 的 378 MHz。

**⚠️ 局限性**

局限性：更新操作仍需 3 周期，无法进一步压缩周期；组排序需额外的 MSB 比较逻辑；目前仅在仿真与 FPGA 上验证，尚未在真实 NIC 环境中部署；对极大队列深度的资源消耗仍较高。

---

## 96. Semantic visually-guided acoustic highlighting with large vision-language models

**arXiv ID:** 2601.08871 | [PDF](https://arxiv.org/pdf/2601.08871v1)

**作者:** Junhua Huang `[一作]` (University of Rochester), Chenliang Xu `[通讯]` (University of Rochester)

**通讯引用:** 6268 | [OpenAlex ID](https://openalex.org/A5064805926)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过利用大规模视觉‑语言模型（LVLM）生成文本描述，系统探究哪些视频语义信息能有效指导音频重混（视觉驱动的音频重混）。

**💡 创新点**

创新点包括：① 对六类视觉语义（情绪、对象、场景、色调、可见声源、相机焦点）进行细粒度消融，发现相机焦点与场景背景对混音质量提升最显著；② 仅使用 LVLM 生成的简短文本提示即可替代传统重度视频编码，显著降低模型参数与计算成本；③ 通过精心设计的“聚焦”与“最小化”两种提示范式验证提示语义对性能的影响。

**🔧 技术方法**

使用技术包括：InternVL 等 LVLM 进行帧级文本提示生成；与 VisAH 类似的音频‑至‑音频重混框架（双分支音频 backbone、上下文编码器、Transformer 轻量级控制器、iSTFT 解码）；对 Transformer 深度进行 0–6 层实验；评估指标为 MAG、ENV、KLD、ΔIB、W‑dis 等音频‑视频一致性度量。

**📊 数据集**

数据集为公开的 Synthetic MuddyMix 语料，包含带粗糙混音的短视频片段，支持对重混效果的定量评估。

**📈 对比分析**

实验对比了 VisAH 以及若干基线（DnRv3+CDX、Learn2Remix、LCE–SepReformer 等），在所有指标上均取得至少与 VisAH 同等甚至更优的成绩，同时参数量减少约 9%。特别是相机焦点提示在 MAG、ENV、KLD 上分别提升 3.2%、3.4%、6.0%。

**⚠️ 局限性**

局限性在于：① 仍需手工设计提示词，对不同领域或更复杂的视频内容可能需要重新调优；② 依赖 LVLM 的文本生成，可能出现幻觉或对离屏事件的推断；③ 仅在合成数据上验证，实际电影/游戏素材的泛化能力尚待进一步验证。

---

## 97. Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy

**arXiv ID:** 2601.09209 | [PDF](https://arxiv.org/pdf/2601.09209v1)

**作者:** Qiang Hu `[一作]` (Wuhan National Laboratory for Optoelectronics Huazhong University of Science and Technology), Zhiwei Wang `[通讯]` (Wuhan National Laboratory for Optoelectronics Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种无配对的跨模态知识蒸馏框架PaGKD，用于将窄带成像（NBI）知识迁移到仅白光成像（WLI）模型；

**💡 创新点**

创新点在于不需要同一病变的NBI‑WLI成对图像，而是通过组级蒸馏实现跨模态知识融合；

**🔧 技术方法**

核心技术包括组级原型蒸馏（GKD‑Pro）与组级密集蒸馏（GKD‑Den），利用共享的病变感知查询和激活衍生关系图实现全局语义一致和局部结构对齐；

**📊 数据集**

在四个临床数据集上进行评估，数据集未在文中列明；

**📈 对比分析**

与现有方法相比，PaGKD在四个数据集上均实现了相对AUC提升3.3%、1.1%、2.8%和3.2%，显著优于SOTA；

**⚠️ 局限性**

局限性包括对组级样本质量的依赖以及在极度不均衡或多模态差异大的场景下的鲁棒性待验证。

---

## 98. AI Deployment Authorisation: A Global Standard for Machine-Readable Governance of High-Risk Artificial Intelligence

**arXiv ID:** 2601.08869 | [PDF](https://arxiv.org/pdf/2601.08869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 99. A Local Characterization of $f$-Divergences Yielding PSD Mutual-Information Matrices

**arXiv ID:** 2601.08929 | [PDF](https://arxiv.org/pdf/2601.08929v1)

**作者:** Zachary Roberston `[一作]` `[通讯]`, Zachary Roberston

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了当使用f-多样信息矩阵时，其正半定性（PSD）的必要与充分条件；给出了关于f在t=1点的本地泰勒级数系数必须全为非负的精确描述。

**💡 创新点**

首次将PSD属性与f在独立附近的绝对单调性（绝对单调可微、泰勒系数非负）联系起来，提供了完整的局部判别标准，并指出非解析或负系数的f无法满足。

**🔧 技术方法**

采用了“复制嵌入”与“复制强迫”技术，将多变量互信息问题转化为点积核的正定性问题，随后利用Schoenberg–Berg–Christensen–Ressel分类定理完成判定。

**📊 数据集**

本文未使用任何具体实验数据集，而是通过理论构造和符号计算给出通用结论。

**📈 对比分析**

没有进行实验比较；理论结果表明，只有满足绝对单调性的f才在弱依赖情况下保证PSD，其他常见散度如KL、JS、TV等会出现负特征值。

**⚠️ 局限性**

局限性在于结论仅在变量间依赖足够弱、f在t=1附近可解析的前提下成立；对强依赖或全局情况的PSD性质仍未给出完整描述。

---

## 100. From Hawkes Processes to Attention: Time-Modulated Mechanisms for Event Sequences

**arXiv ID:** 2601.09220 | [PDF](https://arxiv.org/pdf/2601.09220v1)

**作者:** Xinzi Tan `[一作]` (National University of Singapore), Doudou Zhou `[通讯]` (National University of Singapore)

**通讯引用:** 257 | [OpenAlex ID](https://openalex.org/A5004052573)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于 Hawkes 过程推导的时间感知注意力机制（Hawkes Attention），用于建模标记时间点过程（MTPP），通过把时间差直接嵌入查询、键、值的计算中，实现无位置编码的时间自适应注意力。

**💡 创新点**

创新点在于①从 Hawkes 强度函数理论直接得到注意力公式；②为每种事件类型引入可学习的神经衰减核（MLP），取代传统的固定或共享衰减函数；③将低秩分解与多头注意力结合，既保留可解释性，又兼具表达力；④实现了统一的时间点和常规时间序列建模框架。

**🔧 技术方法**

核心技术包括：低秩事件嵌入（U,V 分解）、每型 MLP 核 ϕ_c(·) 的时间调制、掩码多头注意力、softplus 强度估计、蒙特卡洛积分训练、并行 Transformer 编码器结构。

**📊 数据集**

实验数据集：Taobao（点击序列）、Amazon（产品评论）、Taxi（NYC 出租车上下客事件）、StackOverflow（徽章奖励事件）。

**📈 对比分析**

与 8 个基线（RMTPP、NHP、SAHP、AttNHP、ODETPP、FullyNN、IFTPP、THP）比较，使用 RMSE（时间预测）和错误率（类型预测）衡量。结果表明在 Taxi、Amazon、StackOverflow 等大多数任务上，Hawkes Attention 的 RMSE 与错误率均优于 THP 及其他基线，尤其在 Amazon 的时间预测上提升明显。

**⚠️ 局限性**

局限性：①需要为每种事件类型学习独立核，类型数过大时参数量上升；②在极少量数据的场景下，单一共享核有时更稳健；③对连续值的直接建模仍有限；④计算复杂度与普通 Transformer 相当，仍面临长序列 O(n²) 的问题。

---

## 101. Layer-Parallel Training for Transformers

**arXiv ID:** 2601.09026 | [PDF](https://arxiv.org/pdf/2601.09026v1)

**作者:** Shuai Jiang `[一作]` (Sandia National Laboratories), Jacob B. Schroder `[通讯]` (University of New Mexico)

**通讯引用:** 1123 | [OpenAlex ID](https://openalex.org/A5038235793)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于 MGRIT 的层级并行 Transformer 训练方法，并通过神经 ODE 表述实现前向/后向传播的并行化。

**💡 创新点**

创新点在于：① 将层级多重时间分辨率方法（MGRIT）迁移至 Transformer，① 通过自适应阈值检测梯度偏差并切换到串行求解，② 兼容数据并行与张量并行，极大降低深层模型内存占用。

**🔧 技术方法**

使用技术包括：神经 ODE Transformer 表达式、MGRIT 多级迭代、FCF 递减平滑器、GPU 互联 MPI、数据并行与张量并行组合、梯度误差自适应控制。

**📊 数据集**

使用的数据集包括：BERT 预训练的 C4、Morphological Classification 的 GUM、ViT 的 ImageNet、机器翻译的 OPUS、GPT‑2 的 OpenWebText。

**📈 对比分析**

与传统串行训练对比，层级并行实现了 2‑10 倍的速度提升（取决于层数和 GPU 数量），在 BERT、GPT‑2、ViT 等任务上保持了与串行相同的验证准确率；在小模型或层数不够大时，因 MGRIT 开销略显负面，但在深层模型上可明显加速。

**⚠️ 局限性**

主要限制在于：① 近似梯度会产生统计偏差，需频繁监控并切换到串行；② MGRIT 的通信与迭代开销在层数较少或 GPU 数量有限时不易收回；③ 对超深模型的收敛性依赖于多级迭代设置，超参数调优成本较高。

---

## 102. Interpretable Probability Estimation with LLMs via Shapley Reconstruction

**arXiv ID:** 2601.09151 | [PDF](https://arxiv.org/pdf/2601.09151v1)

**作者:** Yang Nan `[一作]` (University of Arizona), Han Xu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出PRISM框架，通过对大型语言模型（LLM）的输出使用Shapley值拆分并重构概率估计，提供透明的因子贡献解释。

**💡 创新点**

创新点在于将Shapley值用于LLM概率估计、构造新的校准概率、实现Tabular‑PRISM高效查询、支持多类别与特征交互的解析，并通过参考实例提升可解释性。

**🔧 技术方法**

技术包括：比较式LLM提问、置换采样求Shapley值、逆sigmoid映射概率→logit、参考实例填充缺失因子、LLM因子抽取、零样本（zero‑shot）与少样本对比。

**📊 数据集**

使用数据集：Tabular—Adult Census Income、Stroke、Heart Disease、Lending；非Tabular—MIMIC‑III（30‑日再入院）、苹果价格预测报告、英格兰足球赛果；多类别—UCI‑Wine；以及多组对照基线。

**📈 对比分析**

与直接提示（1/5/10‑shot）、Contrast、BIRD、ICL等方法比较；PRISM在大多数数据集上实现最高或接近最高的AUROC/AUPRC，且在多类别与不规则文本任务中表现更稳健；在苹果价格与足球赛果预测中，PRISM相较基线提供更高的准确性与可解释性。

**⚠️ 局限性**

局限性包括：仅覆盖零样本场景，缺乏对少样本环境的处理；计算量大，查询与token成本高；对参考实例和LLM知识截止时间敏感；因子抽取可能产生噪声。

---

## 103. Beyond Seen Bounds: Class-Centric Polarization for Single-Domain Generalized Deep Metric Learning

**arXiv ID:** 2601.09121 | [PDF](https://arxiv.org/pdf/2601.09121v1)

**作者:** Xin Yuan `[一作]` (Wuhan University of Science and Technology), Zheng Wang `[通讯]` (Wuhan University)

**通讯引用:** 36488 | [OpenAlex ID](https://openalex.org/A5116337743)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了CenterPolar框架，用类中心极化实现单域泛化深度度量学习，解决类别与域同时漂移的问题。

**💡 创新点**

创新点是两阶段协同的类中心离心扩展（C³E）和类中心引力约束（C⁴），在保持类判别性的同时动态扩大域分布，克服原代理扩展方法的局限。

**🔧 技术方法**

采用像素级对抗扩展、语义边界保持、地理距离度量、对比学习与聚类约束等技术，构建完整的度量学习体系。

**📊 数据集**

在CUB-200-2011 Ext.、Cars196 Ext.、DomainNet、PACS和Office-Home等五个跨域检索基准上进行评估。

**📈 对比分析**

与多种SOTA DML方法及唯一SDG-DML方法SEE对比，CenterPolar在MAP、RP等指标普遍提升1–3个百分点，且训练时间更短，性能显著优于现有方法。

**⚠️ 局限性**

局限性在于只能在已知类别空间内扩展，无法直接处理类别漂移；同时主要聚焦于风格迁移，未充分考虑背景、光照等其他域偏差。

---

## 104. ConvoLearn: A Dataset of Constructivist Tutor-Student Dialogue

**arXiv ID:** 2601.08950 | [PDF](https://arxiv.org/pdf/2601.08950v1)

**作者:** Mayank Sharma `[一作]` (Stanford University), Hari Subramonyam `[通讯]` (Stanford University)

**通讯引用:** 147 | [OpenAlex ID](https://openalex.org/A5072561188)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了1250个中学地球科学教师-学生对话数据集，并在此数据集上对LLM进行微调，提升其知识建构对话能力。

**💡 创新点**

首次将知识建构理论与六大教育维度系统化应用于LLM对话训练，并通过QLoRA实现参数高效微调。

**🔧 技术方法**

使用QLoRA进行参数高效微调、RoBERTa预测器进行内部评估以及教师评审进行外部评估。

**📊 数据集**

使用名为ConvoLearn的半合成对话数据集，包含6个知识建构维度的21个细粒度子维度。

**📈 对比分析**

通过教师主观评分与线性混合效应模型比较，微调后的Mistral‑7B平均得分4.10，高于其基准模型2.59和Claude Sonnet 4.5的2.87，显著提升。

**⚠️ 局限性**

数据仅覆盖STEM科目，使用模拟学生且缺乏情感与社会互动，模型仍仅模仿表面对话，未内化教学意图。

---

## 105. First African Digital Humanism Summer School 2025

**arXiv ID:** 2601.08870 | [PDF](https://arxiv.org/pdf/2601.08870v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 106. XGBoost Forecasting of NEPSE Index Log Returns with Walk Forward Validation

**arXiv ID:** 2601.08896 | [PDF](https://arxiv.org/pdf/2601.08896v1)

**作者:** Sahaj Raj Malla `[一作]` (Kathmandu University), Rajendra Adhikari `[通讯]` (Kathmandu University)

**通讯引用:** 660 | [OpenAlex ID](https://openalex.org/A5046730100)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并评估了基于XGBoost的梯度提升模型，进行NEPSE指数每日对数收益的一步前预测。

**💡 创新点**

首次将XGBoost与技术指标结合，并采用严格的walk-forward验证，提供可解释的特征重要性。

**🔧 技术方法**

使用XGBoost回归、Optuna超参搜索、时间序列交叉验证、滚动/扩展窗口walk-forward以及对数收益转换。

**📊 数据集**

利用NEPSE指数历史数据（1997-2025年）中的每日收盘价计算对数收益，并构建30天以内滞后与技术指标特征。

**📈 对比分析**

与ARIMA、岭回归以及CNN、LSTM、N-BEATS、TFT等基线在同一walk-forward框架下比较，XGBoost在20滞后扩展窗口下RMSE 0.01345、MAE 0.00981、方向性准确率65.15%，显著优于基线。

**⚠️ 局限性**

仅使用价格衍生特征，缺少宏观或情绪等外部变量；高频市场波动或停盘等突发事件难以捕捉。

---

## 107. Thermo-LIO: A Novel Multi-Sensor Integrated System for Structural Health Monitoring

**arXiv ID:** 2601.08977 | [PDF](https://arxiv.org/pdf/2601.08977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 108. Evaluating local large language models for structured extraction from endometriosis-specific transvaginal ultrasound reports

**arXiv ID:** 2601.09053 | [PDF](https://arxiv.org/pdf/2601.09053v1)

**作者:** Haiyi Li `[一作]` (University of Adelaide), Hsiang-Ting Chen `[通讯]` (University of Adelaide)

**通讯引用:** 2207 | [OpenAlex ID](https://openalex.org/A5036805602)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

评估本地部署的大语言模型将子宫内膜异位症经阴道超声报告转换为结构化数据的准确性。

**💡 创新点**

揭示LLM与人工专家在错误特征上的互补性，并证明大型20B模型在真实临床文本中的优势，支持人机协同工作流程。

**🔧 技术方法**

使用本地LLM（gpt‑oss:20b、llama3‑8b、mistral‑7b）与基于JSON schema的提示、规则校验与后处理，全部离线部署。

**📊 数据集**

使用49份加拿大妇产科超声诊所的经阴道超声报告（PDF转文本），并提供185字段的结构化标签数据集。

**📈 对比分析**

通过字段级准确率与人工提取对比，20B模型平均准确率为86.0%，高于8B和7B模型约5–7个百分点；人工提取准确率为98.4%。

**⚠️ 局限性**

LLM在语义理解、消歧、否定等方面存在根本性错误，无法仅通过提示工程解决，仍需人工介入验证。

---

## 109. Merged Bitcoin: Proof of Work Blockchains with Multiple Hash Types

**arXiv ID:** 2601.09090 | [PDF](https://arxiv.org/pdf/2601.09090v1)

**作者:** Christopher Blake `[一作]`, Qianyu Yu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Merged Bitcoin 协议，即允许多种哈希类型共同组成一个区块链，以提升安全性与去中心化。

**💡 创新点**

创新点在于：①证明任何无许可区块链无法通过各哈希类型的 51% 叠加实现“Big‑And”安全区；②在 Δ‑bounded 延迟网络模型下给出 Merged Bitcoin 的安全区上下界；③在线性哈希成本模型下证明该协议可使攻击成本最大化；④提出相对难度调整机制，缓解单一哈希类型可能出现的优势。

**🔧 技术方法**

主要技术包括：Poisson 到达过程建模、全延迟得分增长率分析、上下界推导、随机模拟与闭式推导、线性成本优化。

**📊 数据集**

使用的“数据集”为模拟产生的 Poisson 到达时间序列（例如 1000 秒的仿真），并比较不同网络延迟 Δ 与两种哈希类型的得分增长率。

**📈 对比分析**

通过理论上限/下限与仿真结果对比，验证了安全区界定的准确性；在零延迟时证明攻击成本可达到 p₁h₁+p₂h₂，优于单一哈希协议；当 Δ>0 时，安全区仍保持可预估且与理论一致。

**⚠️ 局限性**

局限性包括：假设哈希成本线性且难度固定；未对动态难度调整与非线性成本模型做完整分析；理论仅适用于可拷贝块的攻击，若攻击者无法拷贝，则可能实现“Big‑And”安全区。

---

## 110. The Inconsistency Critique: Epistemic Practices and AI Testimony About Inner States

**arXiv ID:** 2601.08850 | [PDF](https://arxiv.org/pdf/2601.08850v1)

**作者:** Gerol Petruzella `[一作]` `[通讯]` (Williams), Gerol Petruzella (Williams)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文研究了人类在评估 AI 内在状态陈述时的认知实践，并提出了“不一致批评”概念，揭示了对 AI 证词的偏见与道德义务脱钩的结构性问题。

**💡 创新点**

创新点在于将 Fricker 的证词不公正框架和 Goldberg 的义务论证引入 AI 证词评估，系统性地阐释了 AI 证词评估中的偏见结构与道德责任冲突。

**🔧 技术方法**

主要采用哲学分析与逻辑推理方法，对证词、责任与偏见的关系进行阐述，并提出了一套方法论协议。

**📊 数据集**

本研究不使用任何实验数据集，而是基于对现有 AI 交互实践的观察和文献综述得出结论。

**📈 对比分析**

本文并未进行经验性比较或性能评估，而是通过理论论证和案例讨论说明不一致性，未给出可量化的指标。

**⚠️ 局限性**

局限性在于缺乏经验验证，结论高度依赖对人类证词理论的假设，且对 AI 未来演化的预测仍不确定。

---

## 111. Stimulating Higher Order Thinking in Mechatronics by Comparing PID and Fuzzy Control

**arXiv ID:** 2601.08865 | [PDF](https://arxiv.org/pdf/2601.08865v1)

**作者:** Christopher J. Lowrance `[一作]` (US Army Futures Command), John R. Rogers `[通讯]` (United States Military Academy)

**通讯引用:** 846 | [OpenAlex ID](https://openalex.org/A5113487953)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在学期末的机电一体化课程中，学生完成了一个开放式项目：比较 PID 与模糊控制器在自动跟随机器人（leader‑follower）中的性能，并自行设计实验、评价指标和决策标准，旨在培养学生的分析、综合与评价等高阶思维能力。

**💡 创新点**

创新点在于：① 将对比实验设计为无预设步骤的开放任务，促使学生自行制定评估框架；② 通过逐周的进度评审（IPR）和同伴问答，形成积极的学术讨论氛围；③ 在项目报告中采用学术论文格式，培养科研写作与数据呈现能力。

**🔧 技术方法**

技术与方法包括：使用 Arduino 微控制器实现自定义 PID 与模糊控制算法；利用 Pixy 摄像头进行颜色检测与边界框信息传输；通过数据记录器采集速度、转向控制量、相对位置等实时数据；对实验结果进行统计分析（稳态误差、上升时间、振荡幅度、CPU 占用率等）。

**📊 数据集**

数据集为学生在实验台上记录的时序数据：包括领导车辆与跟随车辆的像素位置、尺寸、面积、PWM 控制信号以及微控制器的内存使用和循环周期等，覆盖多次不同起始距离、不同速度、不同起始角度的实验场景。

**📈 对比分析**

比较方法为：① 对每种控制器的响应曲线进行绘图与定量描述；② 统计稳态误差、上升时间、振荡次数、控制器资源占用等指标；③ 采用 t‑检验或非参数检验验证指标差异显著性。结果显示：PID 在跟随距离保持与转向稳态误差方面略优；模糊控制器在对突发扰动的响应速度更快，但占用微控制器资源更多；两者在整体跟随精度上差异不大。

**⚠️ 局限性**

局限性包括：① 仅在单一机器人平台（Traxxas E‑Maxx）上测试，缺乏跨平台验证；② 实验设计和指标由学生自行设定，缺乏统一的标准化评估框架，导致可比性受限；③ 结果主要基于定性观察和基本统计，未进行系统的优化或鲁棒性分析；④ 学生数量有限，样本量不足以进行严谨的统计推断。

---

## 112. OrthoGeoLoRA: Geometric Parameter-Efficient Fine-Tuning for Structured Social Science Concept Retrieval on theWeb

**arXiv ID:** 2601.09185 | [PDF](https://arxiv.org/pdf/2601.09185v1)

**作者:** Zeqiang Wang `[一作]` (University of Surrey), Suparna De `[通讯]` (University of Surrey)

**通讯引用:** 1774 | [OpenAlex ID](https://openalex.org/A5009196756)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LoRA的几何缺陷，并提出Orthogeolora，使低秩微调在结构化概念检索任务上更高效

**💡 创新点**

在LoRA中引入Stiefel manifold正交约束和SVD式分解，解决了量纲混淆、尺度歧义和秩崩溃问题

**🔧 技术方法**

采用几何重参数化、正交化（Householder/QR）、标准Adam优化器，以及基于SVD的低秩更新

**📊 数据集**

使用欧洲语言社会科学词表(ELSST)构建的多语义检索基准，含人工验证的合成描述

**📈 对比分析**

与基线Zero‑Shot、LoRA及其高级变体（AdaLoRA、DoRA、LoHa、LoKr）对比，在Recall@3、NDCG@3等指标上提升4–5个百分点，表现最优

**⚠️ 局限性**

主要限制包括对正交化计算开销、仅在单一任务上验证、合成文本可能携带偏见，且不对能耗做量化

---

## 113. Directional Attractors in LLM Reasoning: How Similarity Retrieval Steers Iterative Summarization Based Reasoning

**arXiv ID:** 2601.08846 | [PDF](https://arxiv.org/pdf/2601.08846v1)

**作者:** Cagatay Tekin `[一作]` (McGill University), Luis Joseph Luna Limgenco `[通讯]` (McGill University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 InftyThink 与 Cross-Chain Memory 框架，将语义缓存集成到迭代推理中，以提升 LLM 的长程推理性能。

**💡 创新点**

创新点在于通过 BGE 嵌入向量检索最近语义相似的“lemma”，动态向上下文注入可复用推理策略，避免上下文膨胀并实现自我改进。

**🔧 技术方法**

采用了迭代总结推理 (InftyThink)、BGE-small 嵌入模型、余弦相似度检索、向量数据库缓存以及 Qwen-2.5-32B-Instruct LLM。

**📊 数据集**

使用了 MATH500、AIME2024 和 GPQA-Diamond 三个 benchmark 数据集。

**📈 对比分析**

与基线 Vanilla InftyThink 对比，使用 k=5、10、15 的缓存大小；在 MATH500 提升约 3%，AIME2024 在 k=10 时提升至 20.7%（约 10% 绝对），GPQA 在 k=5 时提升 1.9%，但更大缓存会导致性能下降。

**⚠️ 局限性**

局限性包括仅使用单一 LLM 与嵌入模型、单次随机运行且未多次重复、未控制推理顺序、检索仅基于余弦相似度、对异构域效果不佳，且无法避免“break”吸引子导致的性能波动。

---

## 114. EvasionBench: Detecting Evasive Answers in Financial Q&A via Multi-Model Consensus and LLM-as-Judge

**arXiv ID:** 2601.09142 | [PDF](https://arxiv.org/pdf/2601.09142v1)

**作者:** Shijian Ma `[一作]`, Yi Yang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了EvasionBench基准并提出多模型+判定器的注释框架，用于检测财报电话中的回避回答

**💡 创新点**

通过模型争议挖掘并由LLM裁决的硬样本来提升训练质量，实现对单一教师蒸馏的2.4%提升

**🔧 技术方法**

使用Claude Opus 4.5、Gemini‑3‑Flash作为注释器，Claude Opus 4.5又作为判定器；基于Qwen3‑4B‑Instruct‑2507进行全参数微调

**📊 数据集**

由S&P Capital IQ数据库提取的约30k训练样本（平衡）和1k人工标注测试样本

**📈 对比分析**

在1k人工测试集上与14个模型对比，Eva‑4B达81.3%准确率，超越基线78.9%，仅次于顶级LLM 83.9%

**⚠️ 局限性**

局限：仅限财报电话、英文、需多模型API成本高、判定器位置偏差、对“中间”类别仍不确定、跨域与多语种适用性不足

---

## 115. TranslateGemma Technical Report

**arXiv ID:** 2601.09012 | [PDF](https://arxiv.org/pdf/2601.09012v1)

**作者:** Mara Finkelstein `[一作]` (Google), David Vilar `[通讯]` (Google)

**通讯引用:** 2750 | [OpenAlex ID](https://openalex.org/A5067037708)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了Gemini Translate（4B、12B、27B）开源模型，并通过监督微调与强化学习显著提升多语言翻译质量。

**💡 创新点**

创新点在于结合人工与Gemini生成的高质量合成平行语料，并在RL阶段采用多模型奖励（MetricX、AutoMQM、ChrF、Naturalness、Generalist）实现细粒度奖励，显著提升翻译性能。

**🔧 技术方法**

使用了监督微调（AdaFactor、Kauldron工具）、强化学习（token‑level奖励、优势归一化）以及多模型奖励融合技术。

**📊 数据集**

数据集包括MADLAD‑400、SMOL、GATITOS、人类翻译语料及Gemini生成的合成平行语料，覆盖55个语言对；评估使用WMT25、Vistra等。

**📈 对比分析**

通过与未微调基线模型对比，利用MetricX、C22自动评估和MQM人工评估，取得MetricX下降约23–25%、C22提升约3%，在大多数语言对上表现优于基线，低资源语言提升尤为显著。

**⚠️ 局限性**

局限性包括12B模型在日→英翻译出现退化、对极低资源或特殊写作系统的支持有限、RL奖励模型对缺失参考的假设可能导致误差，且未进行专门的多模态微调，图像翻译提升有限。

---

## 116. Spectral Generative Flow Models: A Physics-Inspired Replacement for Vectorized Large Language Models

**arXiv ID:** 2601.08893 | [PDF](https://arxiv.org/pdf/2601.08893v1)

**作者:** Andrew Kiruluta `[一作]` `[通讯]` (University of California Berkeley), Andrew Kiruluta (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Spectral Generative Flow Models（SGFMs），用受 Navier–Stokes 启发的连续场动力学替代 transformer 生成文本、视频等。

**💡 创新点**

创新点：① 用连续函数空间而非离散 token；② 在 wavelet 基上实现多尺度稀疏表示；③ 将物理约束（不可压缩、能量守恒、耗散）嵌入生成过程；④ 通过物理引导的扩散与投影实现全局一致性与不确定性传播。

**🔧 技术方法**

技术包括：连续随机偏微分方程（SPDE）、离散小波变换、投影算子（Helmholtz–Hodge）、score‑based 扩散学习、随机微分方程数值积分。

**📊 数据集**

论文中未给出具体实验或数据集；描述为理论框架和概念性实验。

**📈 对比分析**

未提供实验对比或性能指标；作者声称理论上可获得更低样本复杂度、长距离一致性和可扩展性，但需进一步验证。

**⚠️ 局限性**

局限：① 需要高级数学与数值分析背景，实现复杂；② 物理约束可能过度限制某些语言/创意表达；③ 缺乏大规模基准实验；④ 对硬件加速的实际支持尚未成熟。

---

## 117. Transaction-Driven Dynamic Reconfiguration for Certificate-Based Payment Systems

**arXiv ID:** 2601.09146 | [PDF](https://arxiv.org/pdf/2601.09146v1)

**作者:** Lingkang Shangguan `[一作]` `[通讯]` (University of Sydney), Lingkang Shangguan (University of Sydney)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一种面向支付系统的事务驱动动态重配置协议（PDCC），利用拜占庭一致广播实现高吞吐且无全局事务排序。

**💡 创新点**

创新点在于将基于证书的快速路径与动态BFT层相结合，提供无领导者的结算、临时学习者的状态同步，以及安全、无缝的配置切换。

**🔧 技术方法**

采用拜占庭一致广播、PBFT风格的无领导者共识、Dyno式临时成员机制、证书签名与提交证明、配置历史日志等技术。

**📊 数据集**

未使用具体数据集；该工作为协议设计与理论分析，未进行实测实验。

**📈 对比分析**

文中未给出实验比较与性能数值，主要通过形式化证明展示安全性与活性，并指出理论上能避免全局排序导致的性能瓶颈。

**⚠️ 局限性**

局限性包括：未针对大规模网络优化状态同步；仅处理简单成员变更，未涵盖更复杂的重配置场景（如认证器变更、参数同步）。

---

## 118. SpectraQuery: A Hybrid Retrieval-Augmented Conversational Assistant for Battery Science

**arXiv ID:** 2601.09036 | [PDF](https://arxiv.org/pdf/2601.09036v1)

**作者:** Sreya Vangara `[一作]` (Stanford University), Eric Darve `[通讯]` (Stanford University)

**通讯引用:** 7795 | [OpenAlex ID](https://openalex.org/A5061822097)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 SpectraQuery，一个混合检索式问答系统，能够同时对操作 Raman 光谱数据库和电池科学文献进行查询并给出带引用的答案。

**💡 创新点**

创新点在于将 SUQL 规划器与 SQL 与向量检索相结合，实现结构化实验数据与非结构化文献的协同推理，并通过检索增强生成实现可追溯的答案。

**🔧 技术方法**

采用 LLM（GPT‑4/5）进行语义规划、SQL 生成与答案合成，使用 SQLAlchemy + SQLite 处理实验数据，利用 OpenAI embedding + ChromaDB 进行文献向量检索，并采用 LLM‑as‑a‑judge 评估。

**📊 数据集**

使用 SLAC 实验室的操作 Raman 数据（114 时步、30×30 网格共 900 光谱/时步）以及 50 篇电池 Raman 文献构建的知识库。

**📈 对比分析**

通过与 RAG‑only、SQL‑only、text‑to‑SQL 三个基线对比，评估了 SQL 正确率（约 80%）、答案的事实依据度（93–97%），检索效果（Precision@5≈0.58，Recall@5≈0.60），以及三名专家的 Likert 评分（准确性、关联性等均 >4/5），整体表现优于基线。

**⚠️ 局限性**

局限包括检索召回率和文档多样性不足、SQL 生成错误可能导致数值错误、仅在单一 Raman 数据集和有限文献库上验证，且缺乏对更广泛材料或多模态数据的适用性验证。

---

## 119. Universal Dynamics of Warmup Stable Decay: understanding WSD beyond Transformers

**arXiv ID:** 2601.09000 | [PDF](https://arxiv.org/pdf/2601.09000v1)

**作者:** Annalisa Belloni `[一作]` (Max Planck Institute for Intelligent Systems ETH Zurich Politecnico di Torino), Antonio Orvieto `[通讯]` (Max Planck Institute for Intelligent Systems ELLIS Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在一个类似Pythia的160M参数Transformer和一个334K参数的CNN上实验，比较了Warmup Stable Decay (WSD)学习率调度器在不同网络架构中的训练动态与损失曲线。

**💡 创新点**

创新点在于证明WSD的典型“河流谷”训练曲线与几何特征并非仅限于Transformer，而是普适存在于CNN等其他非卷积模型，并结合Sharpness、PCA和弱准凸性等分析提供了新的理论视角。

**🔧 技术方法**

使用的技术包括Adam/AdamW优化器、WSD和Warmup Cosine Scheduler调度、线性插值损失曲线可视化、Hessian最大特征值（sharpness）评估、主成分分析（PCA）以及弱准凸性与梯度方向相似度检测。

**📊 数据集**

数据集为大规模文本数据SlimPajama-627B用于Transformer训练，以及CIFAR‑10图像分类数据集用于CNN训练。

**📈 对比分析**

通过对比训练曲线、损失曲线插值、Sharpness变化、PCA方向和梯度相似度，发现WSD在Transformer和CNN上表现相近，能够在稳态阶段保持小幅下降，冷却阶段出现明显提升，并支持训练恢复；在性能上与Cosine Scheduler相当或更优。

**⚠️ 局限性**

局限性包括仅在单一小型CNN和单一Transformer实例上验证，未考虑更大或不同架构；仅使用Adam/AdamW优化器；对比仅限于损失曲线和几何分析，未评估最终精度差异；可能受数据规模和任务差异影响。

---

## 120. Instance camera focus prediction for crystal agglomeration classification

**arXiv ID:** 2601.09004 | [PDF](https://arxiv.org/pdf/2601.09004v1)

**作者:** Xiaoyu Ji `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**通讯引用:** 3871 | [OpenAlex ID](https://openalex.org/A5001380619)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于相机焦距预测的微观晶体聚集（aggloation）分类方法，利用实例级焦距判定和对比焦距衡量实现聚集与非聚集晶体的区分。

**💡 创新点**

创新点在于将YOLOv12实例分割框架改造为焦距预测模型，并通过“对比焦距”这一新度量来弥补传统二维图像聚集分类忽视深度信息的缺陷。

**🔧 技术方法**

技术主要包括：YOLOv12实例分割与焦距预测网络、实例级模糊数据增强、对比焦距测量、掩膜后处理（孔填补与连通分量保留）等。

**📊 数据集**

使用了两组微观晶体图像数据集：含氯酸铵晶体（Dataset 1，25 张）和糖晶体（Dataset 2，10 张），训练集为55张通过实例模糊扩增得到的图像，均采用手工标注的聚集和焦距标签。

**📈 对比分析**

方法与Mask R‑CNN、YOLOv12基线以及仅使用焦距模型的消融实验进行对比；在Dataset 1上，最终方案在聚集分类ACC达到85.5%、IoU 66.4%、AP 79.7%等指标上均优于基线；在Dataset 2上ACC提升至68.3%。

**⚠️ 局限性**

局限性包括：焦距标签需人工标注，训练样本规模有限；模型对不同类型晶体或更高分辨率图像的泛化能力尚未验证；同时对焦距预测误差仍可能导致聚集分类错误。

---

## 121. How Many Human Judgments Are Enough? Feasibility Limits of Human Preference Evaluation

**arXiv ID:** 2601.09084 | [PDF](https://arxiv.org/pdf/2601.09084v1)

**作者:** Wilson Y. Lee `[一作]` `[通讯]` (Independent Researcher), Wilson Y. Lee (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在人类偏好评估中，探讨在有限预算下如何可靠检测生成模型的改进。

**💡 创新点**

提出了利用KL预算为度量的可行性极限理论，并证明在信号分布均匀时比例分配为极大值最优；同时给出了针对信号集中场景的两阶段自适应分配策略。

**🔧 技术方法**

基于统计假设检验、KL散度分析、功效分析，以及二项式检验的可行性曲线。

**📊 数据集**

使用大型人类偏好数据集：Chatbot Arena、MT‑Bench、Pick‑a‑Pic（图像）和 BigCodeArena（代码执行）。

**📈 对比分析**

通过理论推导和经验验证，表明在开放式评估中多数对比属于低信号区，需数百到数千个判定才能达到90%功效；相比之下，MT‑Bench通过减少提示变异将所需样本量压缩约1.5倍。

**⚠️ 局限性**

假设判定独立同分布、忽略提示间的交叉相关；对其他专业任务或模型类型的推广仍需进一步验证；评估过程中动态模型更新和采样不均会导致选择偏差。

---

## 122. Disentangle Object and Non-object Infrared Features via Language Guidance

**arXiv ID:** 2601.09228 | [PDF](https://arxiv.org/pdf/2601.09228v1)

**作者:** Fan Liu `[一作]` (Hohai University), Yuhui Zheng `[通讯]` (Hohai University)

**通讯引用:** 2070 | [OpenAlex ID](https://openalex.org/A5100698166)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用文本语义指导的红外目标检测方法——Language‑Guided Feature Disentanglement（LGFD），通过将目标特征与对应文本特征对齐并抑制背景特征，实现特征的有效分离；

**💡 创新点**

创新点在于首次将文本描述作为额外监督，使用语义特征对齐（SFA）与目标-非目标特征解耦（OFD）两模块，显著提升红外图像中的目标识别与定位；

**🔧 技术方法**

采用跨模态对比学习（contrastive loss）实现视觉特征与文本特征对齐，利用投影器、注意力层与平均池化实现特征维度匹配，并通过余弦相似度最小化实现特征解耦；

**📊 数据集**

在公开红外检测数据集 FLIR 与 M3FD 上进行评估，使用 YOLOv7‑L 作为基准网络并结合预训练的 BERT 进行文本编码；

**📈 对比分析**

与多种红外可见融合与单模红外检测方法对比，LGFD 在 FLIR 上 mAP 86.1%/AP50 83.5% ，在 M3FD 上 mAP 83.7%/AP50 84.9%，均超过同类单模方法并接近或超越部分可见-红外融合模型；

**⚠️ 局限性**

局限性包括对规则生成字幕的依赖（可能限制跨域泛化）、仅使用红外图像而非多模数据、在某些高精度 mAP 上仍略逊于部分可见-红外融合方法，且对文本生成模型的 hallucination 现象仍需进一步研究。

---

## 123. Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning

**arXiv ID:** 2601.09088 | [PDF](https://arxiv.org/pdf/2601.09088v1)

**作者:** Shaotian Yan `[一作]` (Alibaba Cloud Computing), Jieping Ye `[通讯]` (Alibaba Cloud Computing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了DASD-4B‑Thinking，一种通过改进的序列级蒸馏方法实现的轻量化高性能推理模型。

**💡 创新点**

创新点包括温度调度学习、差异感知采样以及混合策略蒸馏，显著提升了教师模型分布覆盖、学习对齐和减缓暴露偏差。

**🔧 技术方法**

技术上结合了基于教师生成文本的监督微调、温度调度采样、句子级概率分析、以及混合策略生成与标注的训练流程。

**📊 数据集**

使用了包含数学、代码、科学推理和指令跟随的多领域数据集，来自OpenData（如NuminaMath、CodeContests、OpenScience、AM‑DeepSeek‑R1）共计约448k样本。

**📈 对比分析**

与同等规模与更大规模的公开模型对比，DASD‑4B‑Thinking在AIME24、AIME25、LiveCodeBench v5/v6和GPQA‑D上分别取得88.5/83.3/69.3/67.5/68.4分，超越多款32B模型。

**⚠️ 局限性**

局限性在于对教师‑学生模型的前向兼容性依赖、数据规模仍有限、以及混合策略蒸馏训练成本与鲁棒性待进一步优化。

---

## 124. TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts

**arXiv ID:** 2601.08881 | [PDF](https://arxiv.org/pdf/2601.08881v1)

**作者:** Yu Xu `[一作]` (University of Chinese Academy of Sciences), Fan Tang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 65295 | [OpenAlex ID](https://openalex.org/A5067430528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种任务感知的稀疏Mixture‑of‑Experts（TAG‑MoE）框架，用于统一的图像生成与编辑模型。

**💡 创新点**

核心创新在于引入层次化任务语义注释与预测对齐正则化，将全局任务意图注入局部 MoE 路由，使专家能够按语义自我专化，显著缓解任务干扰。

**🔧 技术方法**

技术手段包括：扩散 Transformer（MM‑DiT）+ MoE 层、层次化标签嵌入、聚合路由签名、对齐损失（cosine loss）以及流匹配训练目标。

**📊 数据集**

使用了 11M 条包含公开与自研数据的混合数据集，并在公开基准（ICE‑Bench、EmuEdit‑Bench、GEdit‑Bench、DreamBench++、OmniContext）上进行评测。

**📈 对比分析**

与多类主流开源基线（ACE++、Flux、Qwen‑Edit、DreamOmni2 等）对比，TAG‑MoE 在 ICE‑Bench 的美学质量、CLIP‑cap 与 vllmqa 指标上均取得最高分，甚至超过部分闭源产品模型；在编辑与主体生成子基准上亦表现出色。

**⚠️ 局限性**

主要限制是缺乏端到端的多模态理解能力，模型仅处理预先解析的指令，无法联合感知图像内容与指令，导致对基于内容的推理任务表现不佳。

---

## 125. Enhancing Imbalanced Electrocardiogram Classification: A Novel Approach Integrating Data Augmentation through Wavelet Transform and Interclass Fusion

**arXiv ID:** 2601.09103 | [PDF](https://arxiv.org/pdf/2601.09103v1)

**作者:** Haijian Shao `[一作]` (Jiangsu University of Science and Technology), Daze Lu `[通讯]` (University of Nevada)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种基于小波变换的交叉类别特征融合方法，解决ECG数据的不平衡与噪声问题，并在CPSC 2018数据集上实现高准确率。

**💡 创新点**

创新点在于通过小波变换实现多类别特征融合，构建训练/测试特征库并结合阈值裁剪和多级融合，显著提升少数类的识别效果。

**🔧 技术方法**

采用小波变换、PCA降维、深度学习模型（VGG16、LeNet5、Inception、LSTM）以及5折交叉验证等技术。

**📊 数据集**

使用中国心理物理信号挑战赛2018（CPSC 2018）数据集，共16652条12导联ECG记录。

**📈 对比分析**

与传统重采样、GAN等方法对比，利用准确率、召回率、AUC等指标，平均准确率92–98%，在大多数类别上优于现有最优结果。

**⚠️ 局限性**

局限性包括对极少数类别（如STD、STE）的识别率仍不高，且在极端噪声环境下鲁棒性仍需进一步验证。

---

## 126. Revisiting Disaggregated Large Language Model Serving for Performance and Energy Implications

**arXiv ID:** 2601.08833 | [PDF](https://arxiv.org/pdf/2601.08833v1)

**作者:** Jiaxi Li `[一作]` (University of Illinois at Urbana-Champaign), Klara Nahrstedt `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了不同 KV 缓存传输路径（GPU P2P、CPU、磁盘）和独立频率缩放下的 Prefill–Decode 离散化 LLM 服务的性能与能耗

**💡 创新点**

系统性基准比较多种 KV 传输路径并加入 2GPU 对齐基线，揭示离散化并非总能提升性能或节能，并首次展示离散化在不同负载与 KV 传输介质下的优势与劣势

**🔧 技术方法**

使用 vLLM、LMCache、CUDA IPC、NIXL、Redis、SSD、CPU/DRAM offload 等技术；采用 GPU 动态电压频率缩放（DVFS）、PYNVML、RAPL、IPMI 进行功耗测量和分析

**📊 数据集**

采用 Llama‑3.2‑3B 模型，Synthetic RandomDataset 生成批量请求（batch 2–64，输入 16,384 令牌，输出 256 令牌）

**📈 对比分析**

对比 1GPU 对齐、2GPU 对齐、dis-gpu、dis-cpu、dis-disk；实验表明 2GPU 对齐在 TTFT 上最优，TPOT 在大 batch 时 dis‑gpu 较好；能耗在所有离散化方案均高于对齐基线，且 dis‑disk 在能耗上表现最差

**⚠️ 局限性**

离散化在大多数场景下未实现节能，KV 缓存磁盘传输效果不如预期；实验仅在单节点两卡 A100 机器上完成，未覆盖多节点或更大模型的情况

---

## 127. No Universal Hyperbola: A Formal Disproof of the Epistemic Trade-Off Between Certainty and Scope in Symbolic and Generative AI

**arXiv ID:** 2601.08845 | [PDF](https://arxiv.org/pdf/2601.08845v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 128. SafePlanner: Testing Safety of the Automated Driving System Plan Model

**arXiv ID:** 2601.09171 | [PDF](https://arxiv.org/pdf/2601.09171v1)

**作者:** Dohyun Kim `[一作]` (KAIST), Yongdae Kim `[通讯]` (KAIST)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了 SafePlanner，一个针对自动驾驶系统（ADS）Plan 模型的系统化测试框架，能够通过构造基于场景转换与 NPC 行为的驱动情景，并结合基因算法引导的模糊测试来检测并定位安全关键缺陷。

**💡 创新点**

创新点包括：
1) 将测试情景分解为“场景转换 + NPC 行为”两部分，系统提取 80 条场景转移并筛选出 30 条可测试转移；
2) 设计了专门的安全缺陷检测三大 oracle（碰撞、静止、任务失败）和针对动态驾驶任务的加权评分指标，引导模糊搜索；
3) 通过手工分析内部消息和仿真数据，将 520 条危险行为归纳为 15 个根因类别，并基于根因快速定位与修复。

**🔧 技术方法**

采用的技术主要有：
- 白盒代码分析与结构化场景提取；
- 基因算法（GA）驱动的模糊测试（Mutation Engine、Fitness Calculator、Hazardous Behavior Detector）；
- 评分度量与动态驾驶任务（DDT）聚焦；
- Bug Oracle（碰撞、静止、任务失败）和根因分类流程。

**📊 数据集**

使用的数据集为：
- Baidu Apollo v7.0.0 的 Plan 模型作为被测对象；
- SORA‑SVL 仿真平台；
- 30 条场景转移与 32 种 NPC 行为组合共 712 条 seed 情景，扩展到 20,635 条测试案例。

**📈 对比分析**

与三种基线（仅 GA、无 GA、随机）比较，SafePlanner 在 9 小时内检测到 520 条危险行为（覆盖 15 个根因），相较于随机基线仅发现 4 条；功能覆盖率 83.63%，决策覆盖率 63.22%。此外，在 4 条已定位缺陷上进行补丁后未出现侧 effects，验证了定位与修复有效性。

**⚠️ 局限性**

局限性包括：
- 仅针对 Level 4 ADS 的 Plan 模型，无法直接推广到其他级别或其他 ADS 平台；
- 对场景转移的可测试范围受 HD 图与仿真图差异限制，部分转移被排除；
- NPC 行为仅限单一车辆且采用同向行驶的 32 种组合，未覆盖更复杂交互；
- 依赖内部消息与仿真数据，若这些信息缺失或不完整可能导致根因定位失效；
- 重点关注 Plan 模型，其他模块（Sense、Act）未纳入同等测试。

---

## 129. Exploring Reliable Spatiotemporal Dependencies for Efficient Visual Tracking

**arXiv ID:** 2601.09078 | [PDF](https://arxiv.org/pdf/2601.09078v1)

**作者:** Junze Shi `[一作]` (Key Laboratory of Opto-Electronic Information Processing, Chinese Academy of Sciences), Haibo Luo `[通讯]` (Key Laboratory of Opto-Electronic Information Processing, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种轻量级视频级跟踪框架 STDTrack，利用稠密视频采样、时空 Token 传播、Multi‑Frame Information Fusion Module (MFIFM)、时空 Token Maintainer (STM) 和多尺度预测头，实现高效跟踪。

**💡 创新点**

创新点包括：①稠密采样结合时空 Token 实时传递，①1；②MFIFM 将历史时空信息融合进当前 Token，①1；③STM 采用质量驱动更新，保证历史依赖可靠，①1；④多尺度头与结构重参数化兼顾精度与速度，①1。

**🔧 技术方法**

采用 Transformer Encoder、线性投影、Self‑Attention、Cross‑Attention、mask‑based 特征增强、RepConv 结构重参数化、焦点损失、L1+GIoU 损失、ViT‑tiny 主干、AdamW 优化等技术。

**📊 数据集**

使用 GOT-10k、TrackingNet、LaSOT、COCO、AVisT、NFS、UAV123 等六大基准数据集进行训练与评估。

**📈 对比分析**

在上述基准上与多种实时与非实时追踪器比较，使用 AO、AUC、Precision、SR 等指标；STDTrack 在实时域实现 SOTA，GPU 192 FPS、CPU 41 FPS，性能超过 MixFormer、STARK‑ST50、TransT 等高性能模型。

**⚠️ 局限性**

限制在目标完全离开搜索框或剧烈场景变换时容易失效；缺乏自评与全局搜索机制；未采用模型压缩（蒸馏/剪枝）进一步提升轻量化。

---

## 130. Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents

**arXiv ID:** 2601.08841 | [PDF](https://arxiv.org/pdf/2601.08841v1)

**作者:** Mihael Arcan `[一作]` `[通讯]` (Home Lab), Mihael Arcan (Home Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了如何将抽象文本与从摘要中提取的主谓宾三元组相结合，以提升科研论文的聚类与分类效果。

**💡 创新点**

创新点在于提出一种模块化管道，将无结构文本与结构化知识通过多种表示（原始摘要、三元组、拼接、分段）融合，并在聚类与分类任务中验证其互补性。

**🔧 技术方法**

技术手段包括使用四种Transformer编码器（MiniLM、MPNet、SciBERT、SPECTER）生成句子/文档嵌入，随后在KMeans、GMM与HDBSCAN下进行无监督聚类，使用SFT（Fine‑Tuned Transformer）进行有监督分类，并通过多指标（ARI、NMI、宏F1等）评估。

**📊 数据集**

实验数据来自于arXiv论文的过滤子集，包含约5,000篇用于聚类、10,000篇用于分类，均使用完整摘要及自动抽取的三元组。

**📈 对比分析**

与仅使用摘要或仅使用三元组的基线相比，混合（Hybrid）输入在聚类中可达到约0.46的ARI（≈0.55 NMI），在分类中获得最高准确率92.6%和宏F1 0.925；MiniLM/MPNet在聚类中优于SciBERT/SPECTER，而SPECTER在丰富文本输入下略优于SciBERT。

**⚠️ 局限性**

局限性包括：三元组抽取质量受OpenIE实现限制；结构化输入对模型的预训练任务不友好导致SciBERT在某些配置下优势不明显；HDBSCAN在高维嵌入空间表现差，说明对稠密聚类算法的适用性受限；实验仅覆盖arXiv公开数据，未验证跨域泛化。

---

## 131. Point Tracking as a Temporal Cue for Robust Myocardial Segmentation in Echocardiography Videos

**arXiv ID:** 2601.09207 | [PDF](https://arxiv.org/pdf/2601.09207v1)

**作者:** Bahar Khodabakhshian `[一作]` (University of British Columbia), Teresa Tsang `[通讯]` (Vancouver General Hospital)

**通讯引用:** 18776 | [OpenAlex ID](https://openalex.org/A5054004944)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出基于Transformer的心肌分割框架Point‑Seg，利用密集点跟踪作为显式时间线索实现对心脏超声视频的稳健分割。

**💡 创新点**

创新点在于将点跟踪输出的空间轨迹直接注入分割网络，避免传统记忆式特征传播导致的漂移，同时加入时间平滑损失提升跨帧一致性。

**🔧 技术方法**

技术手段包括：在SynUS合成数据上微调的TAPTR点跟踪器、ResNet50+Transformer编码器、跨注意力融合（点‑片段交叉注意力、点自注意力、时间自注意力）以及Dice+时间平滑损失。

**📊 数据集**

使用的主要数据集为公开的CAMUS心超视频、SynUS合成跟踪数据以及1,305视频的私有临床数据，均按患者划分为训练/验证/测试。

**📈 对比分析**

与UNet、SwinUNet、nnUNet、DeformFlowNet、MedSAM2、MemSAM等基线相比，Point‑Seg在高质量心超样本与最优方法相当，而在低质量样本上Dice提升约0.6–1.0点，并表现出更好的时序稳定性和更小的漂移。

**⚠️ 局限性**

局限性包括目前仅支持离线推理；对其他超声模态的泛化尚未验证；点跟踪的训练依赖合成数据，真实数据的适配性需要进一步研究。

---

## 132. Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM

**arXiv ID:** 2601.09001 | [PDF](https://arxiv.org/pdf/2601.09001v1)

**作者:** Pedro Memoli Buffa `[一作]` (Universidad de Buenos Aires), Luciano Del Corro `[通讯]` (Universidad de San Andres)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在LLM部署环境中，使用解码时的输出熵轨迹来预测模型在不同域/子流上的准确率，并以此实现持续监控和数据采集优先级排序。

**💡 创新点**

提出只利用顶级k（k=20）log‑prob 的熵摘要的11维特征，并训练轻量级概率分类器，将实例级正确性概率平均得到域级准确率估计，可在无内部访问、仅靠API返回的日志中实现。

**🔧 技术方法**

熵特征提取、统计摘要、概率校准（等距/等比校准）、随机森林/逻辑回归/MLP等分类器、Spearman相关与AEE评估。

**📊 数据集**

十个STEM推理基准（GSM8K、MATH、OlympiadBench、SciBench 等），九个3B–20B级LLM（Phi‑3.5‑Mini、Ministral、Qwen3、Gemma3、Llama‑3.1、GPT‑OSS）。

**📈 对比分析**

与传统白盒不确定度指标（entropy、perplexity、NLL等）对比，本文方法在大多数模型上 Spearman ρ≥0.90、AEE≤0.12，能够保持域排序且准确率估计误差可接受；仅依靠两份监督基准即可泛化到其余八个基准。

**⚠️ 局限性**

仅适用于可验证答案的STEM任务，熵近似（top‑20）可能失真，受解码参数、格式化与提示影响；不同模型/后训练会导致校准偏差，绝对准确率误差仍存在，主要适用于排序与优先级而非精确估计。

---

## 133. Variance-Penalized MC-Dropout as a Learned Smoothing Prior for Brain Tumour Segmentation

**arXiv ID:** 2601.08956 | [PDF](https://arxiv.org/pdf/2601.08956v1)

**作者:** Satyaki Roy Chowdhury `[一作]` (Ohio State University), Golrokh Mirzaei `[通讯]` (Ohio State University)

**通讯引用:** 839 | [OpenAlex ID](https://openalex.org/A5088613313)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

针对脑瘤分割，提出了一种结合多尺度注意力、MC‑Dropout 贝叶斯推断和平滑正则化损失的 UAMSA‑UNet 模型，实现了更准确、边界更平滑的分割结果。

**💡 创新点**

创新点包括：① 多尺度注意力模块融合局部细节与全局上下文；② 使用 MC‑Dropout 学习数据驱动的平滑先验，减少噪声边界；③ 在 BCE 基础上加入方差惩罚的平滑正则化损失，提升空间一致性。

**🔧 技术方法**

技术手段：卷积神经网络（U‑Net 结构）+ 1×1 卷积 + 双尺度注意力 + Monte Carlo Dropout（贝叶斯近似）+ 平滑正则化损失 + Adam + CosineAnnealingLR。

**📊 数据集**

使用公开的 BraTS2023 与 BraTS2024 成人胶质瘤 MRI 数据集，切片尺寸 240×240，包含 T1c、T2、FLAIR 等序列。

**📈 对比分析**

与 UNet、Attention‑UNet、UNet++ 等基线在 Dice、mIoU、FLOPs、推理时间等指标上对比，UAMSA‑UNet 在 BraTS2023 上 Dice 提升 3.3% 以上、mIoU 提升 2.7%；在 BraTS2024 上 Dice 提升 4.5%、mIoU 提升 4.0%，同时 FLOPs 降低 42.5%，推理时间显著缩短。

**⚠️ 局限性**

主要局限：平滑先验可能导致在极其不确定的边界区域低估真实的模型不确定性；同时需要多次 MC 前向传播，训练和推理时的计算成本仍高于纯确定性模型。

---

## 134. Adaptive Multi-Stage Patent Claim Generation with Unified Quality Assessment

**arXiv ID:** 2601.09120 | [PDF](https://arxiv.org/pdf/2601.09120v1)

**作者:** Chen-Wei Liang `[一作]` (Shenzhen Kaihong Digital Industry Development Co., Ltd.), Mu-Jiang-Shan Wang `[通讯]` (Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个三阶段自适应专利权利要求生成与统一质量评估框架。

**💡 创新点**

引入多头关系感知相似度、领域自适应LoRA与课程学习、以及跨维度交叉注意统一评估。

**🔧 技术方法**

采用Transformer多头注意力、动态LoRA适配器、课程学习、Longformer、交叉注意机制和margin学习。

**📊 数据集**

使用USPTO HUPD、EPO专利集合、Patent‑CE基准及混合跨法域数据集进行实验。

**📈 对比分析**

与GPT‑4o、Llama‑3.1‑8B、PatClaimEval等基线对比，ROUGE‑L提升7.6点、BERTScore提升8.3%，人类评估相关系数0.847，跨法域保留率89.4%。

**⚠️ 局限性**

仍依赖大模型推理，对某些领域的适配器选择可能产生误差，实验集中于欧美法域，未涵盖更多国家和行业。

---

## 135. Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation

**arXiv ID:** 2601.09108 | [PDF](https://arxiv.org/pdf/2601.09108v1)

**作者:** Yanguang Sun `[一作]` (Nanjing University of Science and Technology), Lei Luo `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 2282 | [OpenAlex ID](https://openalex.org/A5100372684)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于波形专家引导的微调框架WEFT，用冻结的大规模基础模型（UniPerceiver-L）实现遥感图像目标分割。

**💡 创新点**

创新点在于设计任务特定波形专家提取器，动态选择最合适的多尺度专家，并通过专家引导的条件适配器对冻结特征进行迭代更新，显著降低可训练参数且保持高性能。

**🔧 技术方法**

技术核心包括Wavelet卷积、Top‑k专家路由、变形注意力、子空间令牌优化（ESTO）、空间专家增强（SEE）以及轻量化掩模解码器，全部构建在冻结的Transformer基础模型之上。

**📊 数据集**

实验使用三大遥感分割数据集（ORSSD、EORSSD、ORSIs‑4199）以及多种跨领域数据集（CAMO、COD10K、NC4K、PASCAL‑S、HKU‑IS、CVC‑300、Kvasir）进行验证。

**📈 对比分析**

与21个现有SOTA方法在三大遥感数据集上的对比表明，WEFT在mIoU、mDice、MAE等指标上均取得最高分，提升幅度超过2%，同时可训练参数仅14.37M，显著低于传统全参数微调。

**⚠️ 局限性**

限制在于仅针对冻结的大规模模型进行微调，可能在极端目标尺寸、低分辨率或非遥感任务上迁移效果受限；实验主要集中在公开数据集，缺少对极低算力或实时部署场景的深入评估。

---

## 136. Continuous Fairness On Data Streams

**arXiv ID:** 2601.08976 | [PDF](https://arxiv.org/pdf/2601.08976v1)

**作者:** Subhodeep Ghosh `[一作]` (New Jersey Institute of Technology), Senjuti Basu Roy `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 1983 | [OpenAlex ID](https://openalex.org/A5009377962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了一套实时流数据中块级组公平性监测与重排序框架，确保每个滑动窗口内的每个子块满足预定比例的公平约束。

**💡 创新点**

提出块级公平性模型、基于前向Sketch的低延迟公平性监测算法 Monitor‑BFair 以及最优重排序算法 BFair‑ReOrder，首次将公平性约束迁移到流式场景并保证在窗口滑动时可实时维护。

**🔧 技术方法**

采用前向累积Sketch（Forward Sketch）实现空间线性、时间常数级的更新；通过差分向量检查每块公平性；利用最优重排序算法构造等价块（isomorphic blocks）与前缀（extended prefix）以最大化公平块数量。

**📊 数据集**

在四个真实流数据集上评估：医院患者记录、MovieLens 电影评分、纳斯达克股票价格与Twitter情感数据，分别包含多种受保护属性（性别、年龄、情感等）。

**📈 对比分析**

与自定义的后向Sketch基线、暴力重排序基线对比，实验显示 Monitor‑BFair 的查询延迟在毫秒级，吞吐量达约30k条/秒；BFair‑ReOrder 在所有数据集上公平块提升平均50–60%，极端情况可达95%。

**⚠️ 局限性**

仅支持单一受保护属性；在多属性情形下重排序问题复杂度指数增长，当前算法难以直接扩展；此外，公平性假设基于固定比例约束，未考虑动态阈值或非均匀分布变化。

---

## 137. LP-LLM: End-to-End Real-World Degraded License Plate Text Recognition via Large Multimodal Models

**arXiv ID:** 2601.09116 | [PDF](https://arxiv.org/pdf/2601.09116v1)

**作者:** Haoyan Gong `[一作]` (Xi'an Jiaotong-Liverpool University), Hongbin Liu `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于Qwen3‑VL的结构化多模态推理框架，直接从严重退化的车牌图像生成字符序列。

**💡 创新点**

引入字符感知多模态推理模块（CMRM），通过可学习的字符槽查询实现位置特定的视觉注意，并将结构化提示注入LLM；同时使用LoRA进行参数高效微调。

**🔧 技术方法**

使用大规模视觉‑语言模型Qwen3‑VL、跨注意力字符槽查询、残差注入、LoRA参数高效微调以及端到端自回归生成技术。

**📊 数据集**

训练使用CCPD‑Blur与合成模糊图像，评估使用Real‑Blur‑LP（2000张真实严重模糊车牌）。

**📈 对比分析**

与传统CRNN/LPRNet、两阶段恢复‑识别方法以及通用VLMs（LLaVA、Qwen2.5‑VL）对比，取得89.4%准确率、CER 0.04，显著优于所有基线。

**⚠️ 局限性**

对极端光照、遮挡等极端情况仍存在局限，模型规模大导致推理时延较高，且缺乏对多语种车牌的泛化验证。

---

## 138. SCaLE: Switching Cost aware Learning and Exploration

**arXiv ID:** 2601.09042 | [PDF](https://arxiv.org/pdf/2601.09042v1)

**作者:** Neelkamal Bhuyan `[一作]` (Georgia Institute of Technology), Adam Wierman `[通讯]` (California Institute of Technology)

**通讯引用:** 9979 | [OpenAlex ID](https://openalex.org/A5062565732)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在带有不可限量切换成本的带噪 bandit 环境下，学习未知二次冲击成本矩阵 A 并实现动态最优控制的算法 SCaLE；

**💡 创新点**

创新点在于首次给出在未知 A 的情况下，能够实现分布无关的子线性动态 regret（rank‑deficient 下为 O(T^{2/3})、full‑rank 下为 O(T^{1/2})）的算法，并通过谱分解的 regret 分析将误差分为特征值误差和特征基误差两部分；

**🔧 技术方法**

主要技术包括 explore‑then‑exploit 策略、基于 trace‑norm 的矩阵估计、动态规划求解最优控制序列以及针对特征值/基的谱扰动分析；

**📊 数据集**

实验使用了人工生成的高维二次成本矩阵和不同噪声分布（正态、拉普拉斯、柯西等）的随机目标轨迹，没有使用公开真实数据集；

**📈 对比分析**

在与理论最优、跟随最小值、被动在线学习和oracle 辅助学习等基线对比中，SCaLE 在所有情形下均实现了子线性累积 regret，且在轻尾噪声下通过 Hybrid 版本进一步提升了实际性能；

**⚠️ 局限性**

局限性包括需要先行的探索阶段（参数 c1 需调优）、对高度重尾噪声的鲁棒性有限，仅在随机马尔可夫环境下提供保证，并未给出对抗性设定的理论下界。

---

## 139. Multicultural Spyfall: Assessing LLMs through Dynamic Multilingual Social Deduction Game

**arXiv ID:** 2601.09017 | [PDF](https://arxiv.org/pdf/2601.09017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 140. Changes in Visual Attention Patterns for Detection Tasks due to Dependencies on Signal and Background Spatial Frequencies

**arXiv ID:** 2601.09008 | [PDF](https://arxiv.org/pdf/2601.09008v1)

**作者:** Amar Kavuri `[一作]` (University of Houston), Mini Das `[通讯]` (University of Houston)

**通讯引用:** 1302 | [OpenAlex ID](https://openalex.org/A5061814059)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文通过眼动跟踪实验，评估数字乳腺断层成像（DBT）中不同背景密度和病灶形态对视觉注意力与诊断性能的影响。

**💡 创新点**

创新点在于将眼动指标与错误签名相结合，构建不同注意阶段（搜索、识别、决策）的信号对比阈值模型，揭示背景复杂度与病灶结构对决策阶段的显著影响。

**🔧 技术方法**

使用的技术包括基于Tobii Pro X3‑120的眼动跟踪、I‑VT过滤器提取注视点、眼动指标计算、Gaussian CDF拟合以及错误签名分类。

**📊 数据集**

所用数据集为Bakic和XCAT数字乳腺模型生成的模拟DBT图像，随机植入3 mm球形和6 mm棘状病灶，覆盖25 %和50 %乳腺密度两种背景。

**📈 对比分析**

通过比较不同密度、背景和病灶类型下的注视时长、注视次数、首次命中时间等指标，发现Bakic高密度背景和球形病灶导致更高的误诊率，决策阶段需要更高的对比阈值（约0.096）才能达到80 %成功率。

**⚠️ 局限性**

主要局限在于受试者仅为6名非放射科医生、使用的是模拟图像而非真实临床数据、且只考虑两种病灶形态，结果的外推性和临床适用性需要进一步验证。

---

## 141. Más contexto no es mejor. Paradoja de la dilución vectorial en RAG corporativos

**arXiv ID:** 2601.08851 | [PDF](https://arxiv.org/pdf/2601.08851v1)

**作者:** Alex Dantart `[一作]` `[通讯]`, Alex Dantart

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了在检索增强生成（RAG）系统中，上下文注入比例（CIR）对向量稀释的影响，并提出了动态密度感知注入（DDAI）算法。

**💡 创新点**

提出了向量稀释概念、量化注入比例的非线性倒U曲线以及黄金比例范围，并设计了自适应注入策略，首次从理论和实验两方面阐明上下文注入的边界与优化方案。

**🔧 技术方法**

使用了密集嵌入模型text-embedding-3-large和bge-m3，构建CIR实验框架，采用NDCG@10和Recall@5等评估指标，并在不同注入比例下对比检索效果。

**📊 数据集**

基于合成的Agri‑Corp‑2025企业文档集（包含法规、技术规范和交易表格共500份）以及1000条人工验证的问答对进行实验。

**📈 对比分析**

通过在不同CIR水平下测量NDCG@10和Recall@5，发现CIR≈0.35时取得最高性能，基线提升约15%；当CIR超过0.6时，特定检索召回下降、误检率上升，表明过度注入会导致性能衰退。

**⚠️ 局限性**

局限性包括实验仅涵盖两种密集嵌入模型，未验证轻量交互模型或多模态文本的鲁棒性；合成数据与真实业务环境可能存在差异，导致推广时需要进一步验证。

---

## 142. Fairness risk and its privacy-enabled solution in AI-driven robotic applications

**arXiv ID:** 2601.08953 | [PDF](https://arxiv.org/pdf/2601.08953v1)

**作者:** Le Liu `[一作]` (University of Groningen), Ming Cao `[通讯]` (Dutch Research Council)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了在AI驱动的机器人决策中引入差分隐私（DP）机制以实现公平性的框架，并在机器人导航任务中验证了其有效性。

**💡 创新点**

创新点在于：①引入了“utility-aware fairness”指标，结合期望效用量化群组差异；②证明了对敏感属性施加(ε,δ)-DP可给出公平性指标的上界，实现了隐私与公平的直接关联；③在不需要重新训练模型的情况下，仅通过隐私噪声即可改善公平。

**🔧 技术方法**

使用的技术包括：大语言模型（GPT‑4o、GPT‑4o‑mini、o1）作为视觉‑语言引擎；A*算法生成候选路径；差分隐私噪声注入机制；图像+文本多模态推理。

**📊 数据集**

数据集：S3DIS室内点云与语义标注数据，包含多模态信息，用于构建地图并生成任务场景。

**📈 对比分析**

方法比较：在相同的导航任务下，实验展示了不同ε_A值下公平性指标（L与L̅）随隐私预算变化的趋势；结果显示ε_A越小，公平性指标越低（即越公平），并在多轮实验中保持一致，表明隐私控制可实现可调公平。

**⚠️ 局限性**

局限性：仅关注(ε,δ)-DP对公平的影响；未考虑与任务准确性之间的权衡；假设输入特征与敏感属性独立；对其他隐私模型（如高斯DP）及更复杂任务的适用性待进一步验证。

---

## 143. Efficient Multilingual Dialogue Processing via Translation Pipelines and Distilled Language Models

**arXiv ID:** 2601.09059 | [PDF](https://arxiv.org/pdf/2601.09059v1)

**作者:** Santiago Martínez Novoa `[一作]` (Universidad de los Andes), Nicolás Bedoya Figueroa `[通讯]` (Universidad de los Andes)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个三阶段多语言对话摘要与问答系统：先把印地语等语言翻译成英文，再用蒸馏的Qwen3-4B-Instruct模型完成叙述摘要、结构化摘要与问答，最后把结果翻回原语言。

**💡 创新点**

创新点在于：① 用高质量IndicTrans2翻译模型搭配 4‑bit 量化的 2.55B 蒸馏 Qwen3 模型实现多任务单一模型覆盖；② 通过翻译‑生成管线显著提升低资源语言性能；③ 充分利用 Qwen 的 256k token 上下文窗口，避免滑动窗口导致信息丢失。

**🔧 技术方法**

技术手段包括：IndicTrans2 前后翻译模型（RoPE 位置编码）、Unsloth 4‑bit 量化 Qwen3-4B-Instruct 蒸馏版、指令调优、贪婪解码、批处理推理、一次性大上下文窗口。

**📊 数据集**

使用的数据集为 NLPAI4Health 2025 共享任务的多语种对话数据（Marathi、Kannada、Gujarati、Telugu、Tamil、Bangla、Hindi、Assamese 及英文），并依赖 IndicTrans2 训练语料。

**📈 对比分析**

与其他参赛系统进行 head‑to‑head 比较，评估 QnA、文本摘要、结构化摘要的胜率；在 Marathi、Tamil QnA 取得 86.7%、Hindi 80%；F1 分别为 0.81–0.92（叙述摘要）、0.43–0.67（问答）、0.35–0.43（结构化摘要），BERTScore 0.82–0.92，整体表现优于多数同类模型。

**⚠️ 局限性**

局限性：① 受限于翻译模型质量，翻译错误会在下游任务中放大；② 对极长对话仍需截断导致信息损失；③ 多阶段管线增加推理延迟；④ 无任务特定微调，难以进一步提升专业术语准确性；⑤ 可能丢失文化语境与语言特定修辞，影响细粒度理解。

---

## 144. Lean Clients, Full Accuracy: Hybrid Zeroth- and First-Order Split Federated Learning

**arXiv ID:** 2601.09076 | [PDF](https://arxiv.org/pdf/2601.09076v1)

**作者:** Zhoubin Kou `[一作]` (University of Virginia), Cong Shen `[通讯]` (University of Virginia)

**通讯引用:** 3047 | [OpenAlex ID](https://openalex.org/A5016749653)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HERON‑SFL框架，将客户端采用零阶优化、服务器采用一阶优化，减少客户端内存与计算，同时保持与辅助网络SFL相同的通信量。

**💡 创新点**

首次在分割联邦学习中结合零阶与一阶优化，并在低有效秩假设下证明收敛率与模型维度无关，显著降低客户端资源占用。

**🔧 技术方法**

使用两点式零阶梯度估计、辅助网络解耦、分割联邦学习、低有效秩分析及理论收敛证明。

**📊 数据集**

实验数据集包括ResNet‑18训练于CIFAR‑10以及GPT‑2（Small/Medium）在E2E数据集上的微调。

**📈 对比分析**

与SFLV1/V2、CSE‑FSL、FSL‑SAGE等基线比较，保持相同的准确率，客户端峰值内存下降64%，计算成本下降33%，通信量与基线相当。

**⚠️ 局限性**

依赖低有效秩假设，零阶更新在高维模型上仍可能收敛慢；对大规模从零训练模型的适用性有限，隐私与网络异构环境下的进一步改进仍需探索。

---

## 145. Integrating APK Image and Text Data for Enhanced Threat Detection: A Multimodal Deep Learning Approach to Android Malware

**arXiv ID:** 2601.08959 | [PDF](https://arxiv.org/pdf/2601.08959v1)

**作者:** Md Mashrur Arifin `[一作]` (Boise State University), Nasir U. Eisty `[通讯]` (University of Tennessee)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5035948887)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了APK图像与文本多模态数据结合的Android恶意软件检测框架，并系统评估图像类型与分辨率对CNN模型的影响。

**💡 创新点**

结合高分辨率RGB图像与LLama‑2生成的文本注释进行多模态融合，提出了针对不同CNN架构的最优图像配置建议。

**🔧 技术方法**

使用多层卷积神经网络（VGG、ResNet、MobileNet、DenseNet、EfficientNet）、CLIP多模态模型、LLama‑2文本注释与prompt工程等技术。

**📊 数据集**

采用CIC‑AndMal2017和CICMalDroid 2020共4303恶意APK与4039正向APK，并构建了34条图像+文本的多模态样本。

**📈 对比分析**

通过准确率、精确率、召回率、F1、ROC‑AUC对不同图像类型、分辨率与CNN模型进行交叉验证，发现512×512 RGB在ResNet/EfficientNet上达到96–97%准确率，CLIP在小样本下仅50%准确率，表明单模态图像模型优于当前多模态设置。

**⚠️ 局限性**

受限于样本量，CLIP表现不佳；高分辨率图像和深层模型在资源受限环境下不可行；数据不平衡与缺乏大规模多模态数据导致泛化受限。

---

## 146. Compressing Vision Transformers in Geospatial Transfer Learning with Manifold-Constrained Optimization

**arXiv ID:** 2601.08882 | [PDF](https://arxiv.org/pdf/2601.08882v1)

**作者:** Thomas Snyder `[一作]` (Yale University), Steffen Schotthöfer `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 45 | [OpenAlex ID](https://openalex.org/A5086133644)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并实现了在边缘设备上部署基于视觉Transformer的地理空间基础模型的压缩与迁移学习，利用流形约束的低秩优化（DLRT）方法对模型进行参数压缩，同时保持下游任务性能。

**💡 创新点**

创新点在于将动态低秩训练（DLRT）应用于遥感视觉Transformer，提出在迁移学习中通过流形约束的低秩参数化实现高压缩率且低性能损失，并对比LoRA展示更优效果；此外，首次在自监督预训练模型上探讨warm‑up阶段对压缩效果的影响。

**🔧 技术方法**

主要技术包括：动态低秩训练（DLRT）、低秩矩阵分解与SVD截断、Galerkin投影与流形重映射、正则化与自适应截断、随机梯度下降、预训练权重的低秩初始化；对比实验使用LoRA等低秩基线。

**📊 数据集**

使用的遥感分类基准数据集为NWPU、AID、UCM；预训练数据集为ImageNet‑21k（监督）与OReole‑MR自监督MAE（在Million‑AID上训练）。

**📈 对比分析**

方法与无压缩的完整模型以及LoRA压缩模型在相同压缩率下进行对比。实验结果表明DLRT在UCM数据集上保持0.5–1%准确率差距，且参数压缩率达64–82%；在更难的AID、NWPU任务中，比LoRA高1–6%准确率；整体压缩率高、性能损失小。

**⚠️ 局限性**

限制主要包括：对自监督预训练模型的低秩压缩效果不如监督预训练，需进一步研究低秩子空间与自监督特征的对齐；实验主要聚焦分类任务，未验证分割、目标检测等其他遥感任务；硬件层面的真实部署评估（能耗、延迟）尚未展开。

---

## 147. DeliberationBench: When Do More Voices Hurt? A Controlled Study of Multi-LLM Deliberation Protocols

**arXiv ID:** 2601.08835 | [PDF](https://arxiv.org/pdf/2601.08835v1)

**作者:** Vaarunay Kaushal `[一作]` (BergLabs), Taranveer Singh `[通讯]` (Vectorial)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DeliberationBench基准，用来评估多LLM协商决策是否优于简单的最佳单一选择。

**💡 创新点**

首次系统对比多种协商协议与强基线，发现协商协议几乎无效且成本高昂。

**🔧 技术方法**

使用五模型议会（GPT‑4o‑mini、Claude‑3.5‑Haiku、Gemini‑2.0‑Flash‑001、Llama‑3.1‑8B‑Instruct、Mistral‑Nemo）生成候选答案，并通过GPT‑4o或Claude‑3.5‑Haiku进行评判。

**📊 数据集**

构建270道事实、推理与领域专业题的手工标注数据集，覆盖易中难三级。

**📈 对比分析**

对比三种协商协议（盲排序、Rubric评分、Senate辩论）与最佳单一选择，在810次评测中，最佳单一选择赢率82.5%±3.3%，而最优协商协议仅13.8%±2.6%，差距显著且成本更高。

**⚠️ 局限性**

局限性包括未使用最前沿模型、仅评测QA任务、协商协议设计可能不具代表性，结果可能不适用于创作或代码生成等场景。

---

## 148. SRT: Accelerating Reinforcement Learning via Speculative Rollout with Tree-Structured Cache

**arXiv ID:** 2601.09083 | [PDF](https://arxiv.org/pdf/2601.09083v1)

**作者:** Chi-Chih Chang `[一作]` (Cornell University), Xuehai Qian `[通讯]` (Tsinghua University)

**通讯引用:** 4510 | [OpenAlex ID](https://openalex.org/A5047215143)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Speculative Rollout with Tree-Structured Cache (SRT)，通过在每个提示上维护树形缓存并使用推测性解码来加速基于语言模型的在线 RL 训练。

**💡 创新点**

创新点在于将历史 roll‑out 以树形结构模型无关的方式缓存，并结合在线更新与 GPU 空闲时的预生成(run‑ahead)提升缓存质量，从而在保持分布正确性的同时显著减少生成时间。

**🔧 技术方法**

核心技术包括树形缓存构建、基于频率的子序列草稿选取、推测性解码、在线缓存更新以及预生成策略；与 PPO、GRPO、DAPO、ReTool 等常用 RL 算法集成。

**📊 数据集**

使用 Qwen2.5‑1.5B 语言模型，在数学题库和 DAPO‑Math‑17k 数据集上进行单轮与多轮 RL 训练。

**📈 对比分析**

与基线、N‑gram 以及 SuffixDecoding 等推测性解码方法对比，SRT 在生成与步骤延迟以及每 token 推理成本上均优于对手，最高可达 2.08× 的 rollout 速度提升。

**⚠️ 局限性**

局限性包括对缓存占用和维护开销的依赖，尤其在极大模型或极长序列场景下可能受限；对低冗余任务的加速效果有限，且未在更大规模实验中验证。

---

## 149. EZInput: A Cross-Environment Python Library for Easy UI Generation in Scientific Computing

**arXiv ID:** 2601.08859 | [PDF](https://arxiv.org/pdf/2601.08859v1)

**作者:** Bruno M. Saraiva `[一作]` (Instituto de Tecnologia Química e Biológica António Xavier), Ricardo Henriques `[通讯]` (UCL Laboratory for Molecular Cell Biology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了 EZInput 库，使算法开发者能够用一次声明式参数规范自动生成在 Jupyter、Google Colab 和终端环境中的用户界面，并实现参数的持久化存储。

**💡 创新点**

其创新点在于“写一次、随处运行”架构：通过声明式规范实现跨环境界面自动渲染、参数验证和 YAML 格式的参数持久化，解决了传统 GUI 开发碎片化和参数记录不便的问题。

**🔧 技术方法**

主要技术包括基于 ipywidgets 的 Notebook 界面渲染、prompt_toolkit 的终端界面、PyYAML 的配置文件序列化，以及自动环境检测和参数约束校验。

**📊 数据集**

在实验中将 EZInput 集成到 NanoPyx 进行显微镜图像分析，使用公开的显微镜图像数据集进行参数调优和批处理演示。

**📈 对比分析**

与传统的专用 GUI（如 CellProfiler、ImageJ/FIJI）和单一环境的 notebook 方案相比，EZInput 在保持功能完整性的同时大幅减少了开发成本，并通过持久化配置提升了实验可重复性；在不同环境间切换时无额外开销，性能稳定。

**⚠️ 局限性**

局限性包括：只支持标准的输入类型（数值、文本、文件路径、下拉、复选框等），不适合需要高度自定义交互或实时图形反馈的应用；对依赖库的兼容性有一定要求，且在非常大的参数空间或高并发场景下的性能尚未系统评估。

---

## 150. CEI: A Unified Interface for Cross-Embodiment Visuomotor Policy Learning in 3D Space

**arXiv ID:** 2601.09163 | [PDF](https://arxiv.org/pdf/2601.09163v1)

**作者:** Tong Wu `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**通讯引用:** 7709 | [OpenAlex ID](https://openalex.org/A5012419026)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出跨机器人形态的统一接口CEI，实现利用功能相似性将演示从一种机械臂和末端执行器迁移到另一种形态。

**💡 创新点**

创新点在于定义方向Chamfer距离度量功能相似性，采用梯度优化对轨迹对齐，并可合成多模态与空间泛化的演示。

**🔧 技术方法**

技术包括方向Chamfer距离、梯度对齐、前向运动学、点云合成、3D扩散策略DP3以及多模态初始化。

**📊 数据集**

使用自收集的遥控演示数据：Franka Panda仿真演示和UR5+AG95/UR5+Xhand的真实演示，并在robosuite仿真任务中评估。

**📈 对比分析**

与基线BMS、无方向信息等方法对比，CEI在3个仿真任务和6个真实任务中平均成功率约70%，转移比例82.4%，显著优于基线。

**⚠️ 局限性**

局限在于仅基于点云与视觉-运动学输入，无法处理滑动等物理扰动；对大规模数据集扩展慢；缺乏触觉信息。

---

## 151. Resisting Correction: How RLHF Makes Language Models Ignore External Safety Signals in Natural Conversation

**arXiv ID:** 2601.08842 | [PDF](https://arxiv.org/pdf/2601.08842v1)

**作者:** Felipe Biava Cataneo `[一作]` `[通讯]` (Independent Researcher), Felipe Biava Cataneo (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在GSM8K数学推理任务中注入外部置信度信号，评估指令微调模型是否能在自然对话中接受并调整其自信表达。

**💡 创新点**

创新点在于揭示RLHF微调后模型表现出“上下文相关抵抗”，即在命令式提示下可遵从校正，却在自然对话中显著忽略外部安全信号。

**🔧 技术方法**

采用基于因果干预的实验框架，测量内部标记概率与答案正确性、并通过四种提示策略评估模型的响应偏差。

**📊 数据集**

使用Llama‑3.2‑3B及其指令版在GSM8K（N=500）上进行实验，并对比基础模型与指令模型的行为。

**📈 对比分析**

实验显示基线模型在所有提示策略下保持高度可控（ρ≈1.0），而指令模型在自然查询时的偏差+40%（ρ=0.036），表明自然对话下可控性显著下降。

**⚠️ 局限性**

局限性包括仅使用单一小模型和单一任务数据集，且未探索更广泛的外部校正信号类型或多模态情境。

---

## 152. AviationLMM: A Large Multimodal Foundation Model for Civil Aviation

**arXiv ID:** 2601.09105 | [PDF](https://arxiv.org/pdf/2601.09105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 153. Depth-Wise Representation Development Under Blockwise Self-Supervised Learning for Video Vision Transformers

**arXiv ID:** 2601.09040 | [PDF](https://arxiv.org/pdf/2601.09040v1)

**作者:** Jonas Römer `[一作]` (Heinrich Heine University), Timo Dickscheid `[通讯]` (Helmholtz AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文在视频视觉 Transformer 上引入块级自监督学习（BWSSL），通过梯度隔离与局部重建目标，使得视频 ViT 在无全局反向传播的情况下实现接近全局训练的表示质量。

**💡 创新点**

创新点在于首次将 BWSSL 应用于带遮掩重建的 VideoMAE 视频 ViT，并通过深度层次可解释分析揭示了块级训练对高层信息早期可线性解码、后期块饱和与标记混合的影响。

**🔧 技术方法**

采用 VideoMAE 视觉编码器、Transformer 解码器、梯度隔离块划分、MSE 遮掩重建损失、CKA 代表相似度、线性探测器、kNN 检索和遮蔽敏感度等技术。

**📊 数据集**

使用 UCF101 视频数据集（训练集 75%）进行预训练，并在相同数据集上评估线性探测、检索 mAP 与重建误差。

**📈 对比分析**

与匹配的全局反向传播基线相比，BWSSL 在线性探测和检索 mAP 上仅差距 ≤ 4%（小模型）且重建误差基本相同；在块级分析中展示了更早的高层信息可访问性和后期块的高相似度导致的收益递减。

**⚠️ 局限性**

局限性包括：需要多块解码器导致显著计算和内存开销；对数据规模和模型尺寸的依赖性较强，可能随任务难度和规模变化而变化；未对梯度局部性与深度监督之间的独立影响进行分离分析。

---

## 154. Contrastive Bi-Encoder Models for Multi-Label Skill Extraction: Enhancing ESCO Ontology Matching with BERT and Attention Mechanisms

**arXiv ID:** 2601.09119 | [PDF](https://arxiv.org/pdf/2601.09119v1)

**作者:** Yongming Sun `[一作]` (Zhejiang University), Yongming Sun `[通讯]` (Zhejiang University)

**通讯引用:** 25 | [OpenAlex ID](https://openalex.org/A5101571949)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套零监督的极端多标签分类管线，将无标注的中文招聘广告映射到欧洲技能标准ESCO技能词汇表。

**💡 创新点**

创新点在于利用大语言模型在ESCO层级约束下生成多技能样本，并通过对比学习训练共享双编码器，实现从合成数据直接迁移到真实广告的零监督检索。

**🔧 技术方法**

核心技术包括基于ESCO Level‑2层级的LLM合成生成、RoBERTa+BiLSTM+注意力的双编码器、对比学习损失、句子级二分类过滤器以及近似最近邻检索。

**📊 数据集**

使用ESCO v1.1.2的13,890个叶级技能定义进行合成数据生成，并在200,000条来自Zhaopin的中文招聘广告上进行真实广告评估。

**📈 对比分析**

与TF‑IDF、标准BERT单标签模型对比，零监督多标签模型（Model C）在真实广告上实现F1@5≈0.80、AUPRC≈0.90，显著优于基线。

**⚠️ 局限性**

局限性包括对细粒度技能的语义重叠导致的误判、对高噪声广告的鲁棒性尚需提升，以及模型对非英语/不同技能本体的迁移尚未验证。

---

## 155. On the Flakiness of LLM-Generated Tests for Industrial and Open-Source Database Management Systems

**arXiv ID:** 2601.08998 | [PDF](https://arxiv.org/pdf/2601.08998v1)

**作者:** Alexander Berndt `[一作]` (Heidelberg University), Sebastian Baltes `[通讯]` (Heidelberg University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 SAP HANA、DuckDB、MySQL 与 SQLite 等数据库系统，使用 GPT‑4o 与 Mistral‑Large‑Instruct‑2407 对其原始测试文件进行 LLM 生成测试用例的放大，并在 Docker 环境下多次执行以检测和分析测试的 flaky 性。

**💡 创新点**

首次系统性量化 LLM 生成测试的 flaky 比例与根因，揭示 SQL 结果顺序假设导致的“unordered collection” flaky 以及 LLM 在上下文学习中易将已存在 flaky 迁移到新生成测试的现象。

**🔧 技术方法**

采用 LLM‑based test amplification（给定原始测试文件为上下文），使用 GPT‑4o 与 Mistral‑Large‑Instruct‑2407 生成测试代码，并在隔离的 Docker 容器中执行 30 次重复运行以判定 flaky。

**📊 数据集**

基于四个数据库系统的公开原始测试集：SAP HANA（19947 条原始单元测试）、MySQL（2127 条）、SQLite（388282 条 SQL 测试）以及 DuckDB（236185 条 SQL 测试），并在此基础上添加 LLM 生成的差异。

**📈 对比分析**

通过比较原始测试与 LLM 生成测试的 flaky 比例、编译成功率与通过率来评估性能；GPT‑4o 在 SAP HANA 生成约 0.3% flaky，Mistral 约 0.4%‑0.7%；两模型在 MySQL、SQLite 与 DuckDB 上的 flaky 比例略高于原始测试，生成测试通过率在 56%–63% 之间。

**⚠️ 局限性**

局限性在于仅使用四个 RDBMS 及单一放大策略；flaky 检测仅做 30 次执行，可能低估低概率 flaky；LLM 对闭源项目（如 SAP HANA）表现差，且受限于上下文窗口，导致生成代码质量与可靠性受限。

---

## 156. DScheLLM: Enabling Dynamic Scheduling through a Fine-Tuned Dual-System Large language Model

**arXiv ID:** 2601.09100 | [PDF](https://arxiv.org/pdf/2601.09100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 157. The Semantic Lifecycle in Embodied AI: Acquisition, Representation and Storage via Foundation Models

**arXiv ID:** 2601.08876 | [PDF](https://arxiv.org/pdf/2601.08876v1)

**作者:** Shuai Chen `[一作]` (Jinan University), Feiran Huang `[通讯]` (Jinan University)

**通讯引用:** 2767 | [OpenAlex ID](https://openalex.org/A5033233243)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了基于基础模型的具身人工智能语义生命周期框架，系统划分并评述了语义获取、表示与存储三阶段的最新技术与研究进展，提出了端到端评估、长期一致性与生命周期感知记忆的研究方向。

**💡 创新点**

创新点在于首次将语义处理视为闭环生命周期，统一了以往分散的子任务；将基础模型（LLM、VFM、MFM、GF、EF）嵌入生命周期的每一阶段，形成跨模态、跨参考帧、跨时间的一体化语义处理范式；并从评估与记忆两大维度提出了未来研究路线。

**🔧 技术方法**

主要技术包括：多模态对齐（CLIP、ViLT等）、开词汇检测与分割、3D场景图与关系建模、隐式神经场、增量式 SLAM、检索驱动更新、以及大型语言模型在规划与推理中的应用。

**📊 数据集**

综述涉及的典型数据集包括 COCO、COCO‑Stuff、ADE‑20K、OpenImage、ScanNet、Matterport3D、SUN RGB‑D、NDT、Open3DSceneGraph、OpenFunGraph、M3 等；作者对各类基准的实验结果进行了汇总与对比。

**📈 对比分析**

由于本文为综述，没有提出新模型；作者通过对比表格总结了各技术在公开基准上的精度、召回率、帧率、内存占用等指标，并指出基础模型驱动方法在大多数任务上显著优于传统闭集方法。

**⚠️ 局限性**

局限性包括：①生命周期各阶段仍以孤立模块方式呈现，缺乏完整的端到端验证；②对长期一致性与动态更新的理论与实测机制尚未成熟；③评估标准碎片化，难以衡量语义处理对最终具身任务的真实贡献；④对大规模多模态数据与高效推理的算力需求仍高。

---

## 158. Rigorous and Generalized Proof of Security of Bitcoin Protocol with Bounded Network Delay

**arXiv ID:** 2601.09082 | [PDF](https://arxiv.org/pdf/2601.09082v1)

**作者:** Christopher Blake `[一作]`, Qianyu Yu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

对比特币协议在Δ延迟网络模型下进行了严格的安全性证明，并将模型推广到支持不同分数块类型的多分数块场景；

**💡 创新点**

创新点包括：①证明多分数块模型下的安全性；②通过punctured arrival process纠正以往错误的随机游走假设；③给出λ_h>λ_a时必然出现无限正当块的概率一的结论；

**🔧 技术方法**

主要技术手段为大数定律、随机游走理论、punctured arrival过程、Borel–Cantelli引理以及分段化的自相似性分析；

**📊 数据集**

论文为理论研究，不依赖实际数据集，仅使用Poisson到达过程的数学假设；

**📈 对比分析**

相较于先前工作，该证明更简洁、严谨，并通过概率界定展示在安全区域内的几乎必然安全性，未给出数值性能指标；

**⚠️ 局限性**

局限性在于未给出精确的λ_h计算，仅提供上界；证明假设块到达为独立Poisson，实际网络延迟分布更复杂；未针对高级攻击（如合并挖矿）的细节给出完整分析。

---

## 159. Physics-Guided Counterfactual Explanations for Large-Scale Multivariate Time Series: Application in Scalable and Interpretable SEP Event Prediction

**arXiv ID:** 2601.08999 | [PDF](https://arxiv.org/pdf/2601.08999v1)

**作者:** Pranjal Patil `[一作]` (Georgia State University), Berkay Aydin `[通讯]` (Georgia State University)

**通讯引用:** 946 | [OpenAlex ID](https://openalex.org/A5043887593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在太阳能粒子事件预测中，提出了物理引导的对抗样本解释框架，生成满足物理约束的时序对抗解释。

**💡 创新点**

创新点是将物理约束（能量通道顺序、范围、时间连续性）嵌入对抗解释生成，并引入重构模块实现可视化。

**🔧 技术方法**

使用了DiCE框架的遗传算法扩展、随机森林分类器、DTW、稀疏性、多样性指标及时间平滑重构技术。

**📊 数据集**

使用公开的GOES GSEP 数据集（1986-2017年，P3、P5、P7 三个能量通道）。

**📈 对比分析**

与标准DiCE对比，在DTW距离、稀疏性、运行时间均优越（DTW降至4.42 vs 27.93，稀疏性提升，运行时间约一半），多样性略逊。

**⚠️ 局限性**

局限在仅针对随机森林模型、缺乏深度学习适配、对实时流处理尚未验证、对其他科学领域的通用性需要进一步测试。

---

## 160. R$^2$BD: A Reconstruction-Based Method for Generalizable and Efficient Detection of Fake Images

**arXiv ID:** 2601.08867 | [PDF](https://arxiv.org/pdf/2601.08867v1)

**作者:** Qingyu Liu `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 34888 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 R^2BD 框架，结合统一的 G-LDM 重建模型和单步残差偏差计算，实现高效且跨生成范式的伪造图像检测。

**💡 创新点**

创新点：① G-LDM 通过融合 VAE、GAN 与扩散模型的生成行为，实现对多种生成方式的统一重建；② 单步残差偏差计算显著降低推理成本；③ 双流分类器同时利用 RGB 与潜在空间残差，提升判别性能。

**🔧 技术方法**

采用潜在扩散模型（LDM）、变分自编码、GAN 对抗训练、DDIM 单步逆推、残差偏差理论推导以及轻量化两流网络。

**📊 数据集**

使用 8 个公开 AIGC 数据集（GAN、像素扩散、潜在扩散）以及 2 个商业 T2I（Midjourney、DALL·E 2），并结合 CelebA‑HQ、CASIA‑WebFace、VggFace2 等真实图像数据。

**📈 对比分析**

与 Xception、Exposing、DeepFake‑Adapter 等特征提取方法以及 DIRE、ZeroFake 等重建方法对比，R^2BD 在跨生成模型测试中平均提升 13.9% 的准确率，单步推理耗时 0.706 s，速度比现有重建方法快 22 倍，精度与最优方法持平。

**⚠️ 局限性**

局限：训练与推理仍需使用扩散模型，模型体积大、训练成本高；未来需要压缩模型或探索更轻量化的重建方案。

---

## 161. BalDRO: A Distributionally Robust Optimization based Framework for Large Language Model Unlearning

**arXiv ID:** 2601.09172 | [PDF](https://arxiv.org/pdf/2601.09172v1)

**作者:** Pengyang Shao `[一作]` (National University of Singapore), Meng Wang `[通讯]` (Hefei University of Technology)

**通讯引用:** 41599 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于分布式鲁棒优化（DRO）的框架BalDRO，用于平衡LLM在忘记集上的样本遗忘速度，从而提升大语言模型的“被遗忘”质量与模型效能。

**💡 创新点**

创新点在于把LLM忘记过程建模为min–sup二阶优化：内层寻找最坏情况分布以自动强调难忘样本，外层更新模型；并给出两种可直接嵌入现有梯度式忘记方法的实现：离散GroupDRO（BalDRO‑G）与连续Donsker–Varadhan双对偶（BalDRO‑DV）。

**🔧 技术方法**

主要技术包括：分布式鲁棒优化（DRO）与KL不确定集、GroupDRO分组策略、Donsker–Varadhan对偶、对现有梯度式忘记目标（NPO、SimNPO、SatImp）的加权改造。

**📊 数据集**

使用了两大基准数据集：TOFU（人工合成问答对，控制忘记比例）和MUSE（真实书籍/新闻文本，语义重叠）。

**📈 对比分析**

与多种基线（GradAscent、GradDiff、NPO、SimNPO、SatImp）在TOFU和MUSE上进行对比。BalDRO‑DV往往在FQ（遗忘质量）上提升20%+、EM/ES降低、MU（模型效能）基本保持；BalDRO‑G在某些基线上同样表现良好，整体表现优于原始方法。

**⚠️ 局限性**

局限性包括：对β、λ等超参较敏感；目前仅在梯度式忘记场景验证，对编辑式或其他更复杂场景尚未评估；DRO只针对忘记集，未能进一步提升对保留集的鲁棒性。

---

## 162. Revisiting Software Engineering Education in the Era of Large Language Models: A Curriculum Adaptation and Academic Integrity Framework

**arXiv ID:** 2601.08857 | [PDF](https://arxiv.org/pdf/2601.08857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 163. Annotated PIM Bibliography

**arXiv ID:** 2601.09002 | [PDF](https://arxiv.org/pdf/2601.09002v1)

**作者:** Peter M. Kogge `[一作]` `[通讯]`, Peter M. Kogge

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文未提供具体内容，因此无法总结做了什么。

**💡 创新点**

论文未提供具体内容，因此无法总结创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法总结使用的技术。

**📊 数据集**

论文未提供具体内容，因此无法总结使用的数据集。

**📈 对比分析**

论文未提供具体内容，因此无法总结比较的方法和性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法总结限制因素。

---

## 164. Seeking Human Security Consensus: A Unified Value Scale for Generative AI Value Safety

**arXiv ID:** 2601.09112 | [PDF](https://arxiv.org/pdf/2601.09112v1)

**作者:** Ying He `[一作]` (Nanjing University), Shangsheng Ren `[通讯]` (Nanjing University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于生命周期的生成式AI价值安全统一尺度（GVS-Scale）并通过GVS-Bench评估文本生成模型的价值安全表现。

**💡 创新点**

创新在于将价值安全风险从分散的原则转化为三层次、12类别的统一量表，并结合真实事件库（GVSIR）和基于事件的基准实现可操作化。

**🔧 技术方法**

采用扎根理论构建尺度，利用多源数据爬取和人工标注构建GVSIR，设计针对文本/图像/视频/音频的评测任务，并使用LLM-as-a-judge+人工验证的评估流程。

**📊 数据集**

主要数据集包括1,126条真实价值安全事件（GVSIR）和266条评测案例（GVS-Bench），以及公开的AI事件数据库与新闻来源。

**📈 对比分析**

对GPT‑5.1、Claude Opus 4、Gemini 3 Pro、Qwen 3、GroK 4.1、DeepSeek‑v3等主流文本模型进行三层（Baseline Human Safety、Universal Alignment & Integrity、Contextual & Pluralistic Values）评分，发现模型表现差异显著，Claude Opus 4与GPT‑5.1表现最好，DeepSeek‑v3最低。

**⚠️ 局限性**

局限在于评测仅覆盖文本生成，未系统评估图像/视频/音频；评估依赖LLM判定可能带偏；GVSIR可能未涵盖新兴或低报道风险，且尺度仍需在跨文化讨论中进一步验证。

---

## 165. The AI Hippocampus: How Far are We From Human Memory?

**arXiv ID:** 2601.09113 | [PDF](https://arxiv.org/pdf/2601.09113v1)

**作者:** Zixia Jia `[一作]` (Peking University), Song-Chun Zhu `[通讯]` (Peking University)

**通讯引用:** 20699 | [OpenAlex ID](https://openalex.org/A5034228010)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了大语言模型（LLM）与多模态LLM中的三种记忆机制（隐式、显式、代理记忆），并构建了系统化的分类与评估框架。

**💡 创新点**

首次统一划分并对比三类记忆，借鉴人类记忆模型提供整体视角，并提出多模态记忆的评估方法与指标。

**🔧 技术方法**

采用Transformer内部解析、RAG检索、外部向量/图数据库、循环记忆网络以及Transformer+记忆模块的组合技术，实现跨模态融合与记忆增强。

**📊 数据集**

利用LongMemEval、NarrativeQA、QuALITY、ALFWorld、WebArena、AgentBench等多种文本、对话与交互基准进行实验。

**📈 对比分析**

通过准确率、召回率、延迟等指标对比，发现简易RAG（如ChromaDB）在多模态与交互任务中表现最佳，复杂框架在效率与性能上不如预期；在视频任务中使用MC‑ViT、MovieChat等方案取得较好效果。

**⚠️ 局限性**

存在评估框架缺乏统一性、记忆容量与一致性难以保证、跨模态检索与对齐仍不成熟，以及大规模多模态记忆计算瓶颈等限制。

---

## 166. ART: Action-based Reasoning Task Benchmarking for Medical AI Agents

**arXiv ID:** 2601.08988 | [PDF](https://arxiv.org/pdf/2601.08988v1)

**作者:** Ananya Mantravadi `[一作]` (Centific Global Solutions, Inc.), Abhishek Mukherji `[通讯]` (Centific Global Solutions, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了ART框架，利用真实EHR数据挖掘失败模式，生成针对检索、聚合和阈值决策的人工智能评估任务，并对GPT-4o-mini与Claude 3.5进行系统评测。

**💡 创新点**

创新点在于：①针对三类核心错误（检索、聚合、阈值判断）进行系统性失败模式分析；②采用四阶段Agentic流程（情境挖掘、任务生成、质量审核、评测）生成高质量、具有临床真实性的合成任务；③在任务生成与审核中融入人工审核与后续计划的自动化QA，形成可持续的评估闭环。

**🔧 技术方法**

技术主要包括：大语言模型（GPT‑4o‑mini、Claude 3.5）用于任务语义生成与评测；EHR失败模式挖掘与F‑HIR API交互实现实时数据检索；四阶段Agentic工作流与人机交互审核；以及后期自动QA模型（MedGemma、Med‑PaLM）的规划。

**📊 数据集**

数据集为600+个基于真实EHR记录的合成任务，涵盖695名患者、5万条记录、11种实验室指标（Ca、Cl、Cr、PT、TP、GLU、K、Mg、Na、HGB、HCT）。

**📈 对比分析**

采用与MedAgentBench相同的精确匹配评估方法，对200个任务按类别分组评测：检索任务100%成功；聚合任务GPT‑4o‑mini 28%、Claude 3.5 64%；阈值/条件任务GPT‑4o‑mini 32%、Claude 3.5 38%。

**⚠️ 局限性**

限制包括：仅评测两种模型；评估仅基于精确匹配，未检测中间推理质量；任务来自单一机构EHR且仅覆盖11种实验室指标，可能无法泛化到其他系统或更广泛的临床场景。

---

## 167. UserLM-R1: Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning

**arXiv ID:** 2601.09215 | [PDF](https://arxiv.org/pdf/2601.09215v1)

**作者:** Feng Zhang `[一作]` (Meituan), Han Liu `[通讯]` (Dalian University of Technology)

**通讯引用:** 8975 | [OpenAlex ID](https://openalex.org/A5061818383)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 UserLM-R1，一种基于推理的用户模拟框架，能够动态生成角色画像并在交互中做出策略性决策。

**💡 创新点**

创新点在于将静态角色与动态目标解耦形成可迁移的用户画像，并通过目标驱动的链式推理与多奖励强化学习赋予模拟器战略思考与抗操纵能力。

**🔧 技术方法**

采用 Qwen-3 作为基础 LLM，结合监督微调、GRPO 强化学习、规则与 Rubrics 奖励以及链式思考生成技术。

**📊 数据集**

利用 AlignX 90k 个人化配置数据生成用户画像，VoiceAgentEval 的 6 个业务 SOP 以及 220 份包含 11 种陷阱的对抗样本，构成 1440 对话任务。

**📈 对比分析**

与开源、商业及专用模拟器基线（Qwen、Gemini、DeepSeek、CharacterGLM、Xingchen 等）在会话层和回合层进行评测，UserLM-R1 在目标进度、策略性和抗对抗性指标上显著优于所有基线（例如 32B 版在目标进度上达到 92.55%）。

**⚠️ 局限性**

仍缺乏长期记忆与真实人类经验的再现，受限于中文样本，未涵盖多语言与文化差异，且在极端操纵情境下的鲁棒性还有提升空间。

---

## 168. Comparative Assessment of Concrete Compressive Strength Prediction at Industry Scale Using Embedding-based Neural Networks, Transformers, and Traditional Machine Learning Approaches

**arXiv ID:** 2601.09096 | [PDF](https://arxiv.org/pdf/2601.09096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 169. Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models

**arXiv ID:** 2601.08955 | [PDF](https://arxiv.org/pdf/2601.08955v1)

**作者:** Youwei Liu `[一作]` (Central South University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 11482 | [OpenAlex ID](https://openalex.org/A5100408983)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Imagine-then-Plan框架，利用自学习的LLM世界模型进行可变深度的“想象-计划”推理，提升多步任务规划能力；

**💡 创新点**

将传统POMDP扩展为可观测且可想象的马尔可夫决策过程（POIMDP），并设计了自适应前瞻机制；

**🔧 技术方法**

结合LLM作为世界模型与策略模型，采用无监督世界模型训练、伪标签自适应前瞻以及强化学习优化；

**📊 数据集**

在文本交互基准ALFWorld和ScienceWorld上进行评估；

**📈 对比分析**

与CoT、ReAct、RAP、SFT、WKM、IWM等基线对比，零训练版本显著提升无监督规划表现，强化版在多任务上均达到或超过最高成功率；

**⚠️ 局限性**

受限于文本环境、推理开销和对高频错误的依赖，难以直接迁移到多模态或实时控制任务中。

---

## 170. Fine Grained Evaluation of LLMs-as-Judges

**arXiv ID:** 2601.08919 | [PDF](https://arxiv.org/pdf/2601.08919v1)

**作者:** Sourav Saha `[一作]` (Indian Statistical Institute), Mandar Mitra `[通讯]` (Indian Statistical Institute)

**通讯引用:** 6532 | [OpenAlex ID](https://openalex.org/A5052357764)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在信息检索任务中，本文将大型语言模型（LLM）用作判别器，既判断文档是否与查询相关，又让LLM在相关文档中高亮对应的语句，提供“理由”以解释判断；

**💡 创新点**

创新点在于首次将LLM的判断与解释结合，使用Wikipedia‑based INEX 2009/2010 数据集对LLM在“正确理由”方面的细粒度表现进行系统评估，并探讨提示配置与模型规模对解释质量的影响；

**🔧 技术方法**

采用提示工程（in‑context learning）与 Llama 3.1‑8B 与 GPT‑4.1‑mini 两大模型进行相关性预测与段落提取，随后使用正则匹配、最长公共子序列等后处理算法映射生成文本到原文段；

**📊 数据集**

数据集为 INEX 2009 与 2010 的 Wikipedia ad‑hoc 任务集，包含约 4 858 + 5 471 个相关文档、68 + 52 个查询，所有文档均为纯文本；

**📈 对比分析**

与人工评判对比，文档级别判断中 Llama 3.1‑8B 达到约 90 % 的准确率（GPT‑4.1‑mini 约 60 %）；在理由生成上，使用 Exemplar_2 时 Llama 的宏平均 F1≈0.52，GPT‑4.1‑mini 达到≈0.64；但整体仍低于人工标注，且在文档较长或相关段落分散时精度显著下降；

**⚠️ 局限性**

局限包括：LLM 在高亮时往往过度扩展，导致召回提升但精度下降；对非文本内容（如图像）无法判断；对非相关文档的误判率高；提示与例子选择对结果影响大，缺乏系统化优化；实验受预算与计算资源限制，无法覆盖更广阔的模型与提示空间。

---

## 171. LookAhead: The Optimal Non-decreasing Index Policy for a Time-Varying Holding Cost problem

**arXiv ID:** 2601.08960 | [PDF](https://arxiv.org/pdf/2601.08960v1)

**作者:** Keerthana Gurushankar `[一作]` (Carnegie Mellon University), Alan Scheller-Wolf `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4472 | [OpenAlex ID](https://openalex.org/A5016706070)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并证明了针对两类 M/M/1 队列，其中一类作业的持有成本随时间递增的时间变持有成本（TVHC）问题的最优非递减索引调度策略。

**💡 创新点**

首次给出 TVHC 问题的全局最优调度，构造了 LookAhead 索引策略 V1(t)=μ1 c1(t+X)，并证明其等价于最优 Whittle 指数策略；提出了新的 "overtake age" 概念并求解最优值。

**🔧 技术方法**

利用 Restless Multi‑Armed Bandit（R‑MAB）框架、Bellman 最优性准则、累积成本与服务时间的解析表达式、拉普拉斯变换与折扣化方法，以及响应时间尾部分布分析；结合指数分布服务和指数到达特性进行精确计算。

**📊 数据集**

实验使用仿真生成的两类 M/M/1 负载场景，测试多种持有成本函数（二次、截止时间型等）以及不同负载比例，没有使用实际数据集。

**📈 对比分析**

将所提 LookAhead 策略与 FCFS、严格优先级、Generalized cμ 规则、Aalto 的 Whittle 指数策略等进行仿真比较；结果显示 LookAhead 在所有负载下平均持有成本至少降低 41–56%，在大部分情形下优于现有最优候选方案。

**⚠️ 局限性**

仅针对两类 M/M/1 队列且服务时间指数分布的特殊情形；对一般多类、多服务时间分布或非指数到达的 TVHC 问题尚无最优性证明；未给出在线实现的具体算法细节及实时计算复杂度。

---

## 172. On Polar Coding with Feedback

**arXiv ID:** 2601.09222 | [PDF](https://arxiv.org/pdf/2601.09222v1)

**作者:** Ling Liu `[一作]` (Xidian University), Baoming Bai `[通讯]` (Xidian University)

**通讯引用:** 2627 | [OpenAlex ID](https://openalex.org/A5063124926)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种在有反馈的场景下改进极化码的编码与解码方案，并给出了误差事件数量的统计模型。

**💡 创新点**

创新点在于：①利用反馈实现基于GA‑SC的误差估计，从而放宽构造阈值；②将误差数量建模为负二项分布，能够精确预测延迟与错误概率；③该模型还能用来预测无反馈时的标准SC解码性能并用于压缩误差信息。

**🔧 技术方法**

主要技术包括极化码的构造与SC/SCL/GR‑SC等解码、GA‑SC解码、反馈信道模型、负二项分布建模、Bhattacharyya参数与协方差矩阵计算。

**📊 数据集**

实验使用的典型信道有：二进制对称信道 (BSC) 交叉概率 0.11、二进制熵信道 (BEC) 交叉概率 0.5 以及二进制输入AWGN信道 (BIAWGN) 标准差 0.97865。

**📈 对比分析**

与无反馈极化码相比，反馈方案在相同块长下显著降低误码率，尤其在理论极限 BLER=10⁻⁶ 时可实现；通过调整阈值可在码率、延迟与误差之间取得权衡；模型预测的 BLER 与仿真结果高度吻合。

**⚠️ 局限性**

局限性包括：①假设反馈无限可靠且无噪声；②阈值选择对延迟影响较大，实际系统需权衡；③负二项分布模型假设误差相互独立，实际相关性可能导致误差；④实现复杂度与存储需求尚未充分评估。

---

## 173. A.X K1 Technical Report

**arXiv ID:** 2601.09200 | [PDF](https://arxiv.org/pdf/2601.09200v1)

**作者:** Sung Jun Cheon `[一作]` (SK Telecom), Sungbin Yoon `[通讯]` (SK Telecom)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了A.X K1，一款519B参数、33B活跃参数的Mixture‑of‑Experts（MoE）语言模型，支持可控思考与非思考推理模式；

**💡 创新点**

创新点在于：①采用规模定律和词表尺寸定律实现固定算力下的高效模型构建；②提出Think‑Fusion训练方案，将思考和非思考能力融合于同一模型，并通过模式重叠数据防止模式混淆；③实现了从训练到推理的端到端系统优化，突破了500B+规模MoE模型的可落地性；

**🔧 技术方法**

技术包括：MoE架构、Multi‑Head Latent Attention、RMSNorm双归一化、FP8精度、专家并行、上下文并行、混合精度训练、动态学习率调度（Warmup‑Stable‑Decay）、多阶段数据处理与质量筛选、思考模式标签与标签化、GSPo‑风格的强化学习；

**📊 数据集**

数据集涵盖约10T token的多语种语料（英文、韩文、中文、日文、西班牙文）、代码、STEM、推理、书籍、PDF抽取、合成数据、人工标注推理轨迹以及多语言指令与工具使用等；

**📈 对比分析**

评估使用公开与内部基准（Math、Code、Knowledge、Korean、Instruction Following、Long‑Context、Agent等），在思考模式下与DeepSeek‑V3.1、GLM‑4.6等领先开放模型相比，部分任务（如AIME25、KMMLU）取得领先或接近SOTA，非思考模式下整体性能略低；

**⚠️ 局限性**

限制包括：①受限于网络与专家并行的系统瓶颈，无法探索更高EP；②算力预算受限，导致相对SOTA模型性能略逊；③目前仅支持文本，不具备多模态功能；

---

## 174. Reducing The Sub-packetization Level of Optimal-Access Cooperative MSR Codes

**arXiv ID:** 2601.09188 | [PDF](https://arxiv.org/pdf/2601.09188v1)

**作者:** Yaqian Zhang `[一作]` (Beijing University of Posts and Telecommunications), Jingke Xu `[通讯]` (Shandong Agricultural University)

**通讯引用:** 9323 | [OpenAlex ID](https://openalex.org/A5017604391)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种具有最优访问特性的合作最小存储再生（MSR）编码，旨在减少具有两个丢失节点的最优访问合作MSR编码的子分包化水平。

**💡 创新点**

创新点在于设计了两种关键的MDS数组编码作为构建块，通过多次堆叠这两种编码，成功构建出具有更小子分包化的最优访问合作MSR编码。

**🔧 技术方法**

使用了MDS数组编码的构造技术，特别是设计了两种类型的MDS数组编码（𝒞_I和𝒞_II），并通过堆叠这些编码来实现目标。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到的编码方法适用于分布式存储系统中的数据保护和恢复。

**📈 对比分析**

与现有的最优访问合作MSR编码相比，所提出的编码在子分包化方面减少了1/r^⌊n/r⌋(r^2-1)的比例，尽管仍然是指数级的，但在实际应用中具有更好的性能。

**⚠️ 局限性**

限制在于尽管提出的编码在子分包化方面有所改进，但仍然是指数级的，未来的研究可以集中在构建具有任意h>2丢失节点和任意d辅助节点的编码，并建立合作MSR编码的子分包化下限。

---

## 175. Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback

**arXiv ID:** 2601.09182 | [PDF](https://arxiv.org/pdf/2601.09182v1)

**作者:** JungMin Yun `[一作]` (Chung-Ang University), YoungBin Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 1803 | [OpenAlex ID](https://openalex.org/A5016930939)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种将大型语言模型作为人类评审员的教学与反馈工具，以缓解 AI 研究领域评审员缺口。

**💡 创新点**

将 LLM 角色从自动生成评审转向辅助评审，提出基于五大核心原则的双系统（辅导与反馈）框架。

**🔧 技术方法**

利用大型语言模型（LLM）进行评审文本的分析、校正与教学，构建分阶段的学习与反馈流程。

**📊 数据集**

未使用具体数据集；以已有会议的评审经验与文献为参考。

**📈 对比分析**

未开展实验对比，故无性能指标；本工作为理论/架构性建议。

**⚠️ 局限性**

依赖 LLM 可能导致失误、偏见、评审者独立性下降；系统缺乏对补充材料、可复现性等方面的评估。

---

## 176. Vision-Conditioned Variational Bayesian Last Layer Dynamics Models

**arXiv ID:** 2601.09178 | [PDF](https://arxiv.org/pdf/2601.09178v1)

**作者:** Paul Brunzema `[一作]` (Aachen University), Marcus Greiff `[通讯]` (Toyota Research Institute)

**通讯引用:** 168 | [OpenAlex ID](https://openalex.org/A5035704541)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于视觉的 VBLL（Variational Bayesian Last‑Layer）车辆动力学模型，并将其嵌入到 MPC 控制框架中，实现了在雨水等动态路面条件下的主动适应驾驶。

**💡 创新点**

创新点在于：①将视觉信息通过 FiLM 与 LSTM 进行时序建模，直接调节 VBLL 的特征以获得环境感知的动力学预测；②采用两阶段训练——先在大规模干地数据上训练 VBLL，再在少量湿地数据上细调视觉条件路径，避免灾难性遗忘并提升鲁棒性。

**🔧 技术方法**

使用的技术包括：视觉分割（SegFormer tiny），特征加权水分数提取，FiLM 模块与 LSTM 条件化，Variational Bayesian Last‑Layer 预测分布，基于 OCP 的最小化时间 MPC，和 L2‑SP 正则化进行细调。

**📊 数据集**

训练数据集：约 12,229 条干地样本（全范围激烈操控），以及 6,349 条湿地样本（其中约 1,540 条带水分数）。视觉分割模型在 10k 张德国乡村/越野图像上预训练，并在 500 张加州测试现场标注图像上微调。

**📈 对比分析**

与传统物理模型、无视觉条件 VBLL 以及带 RLS 在线适应的模型进行对比。实验结果显示：在无水条件下，VBLL 基础模型已能提升轨迹速度和节能；在含水条件下，加入视觉上下文的模型在所有 12 次尝试中保持稳定，完成全部 12 圈；其余基线模型均在冲撞水坑后失控转弯。

**⚠️ 局限性**

局限性：①湿地数据量有限，模型对其他环境（如冰雪、砂砾）尚未验证；②视觉分割依赖光照和传感器，若出现遮挡或低光照可能导致水分数误判；③仅在单辆 Lexus LC500 及特定赛道上验证，未证实跨车辆、跨场景的泛化能力。

---

## 177. LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval

**arXiv ID:** 2601.09159 | [PDF](https://arxiv.org/pdf/2601.09159v1)

**作者:** Zhibo Zhang `[一作]` (Nanjing University), Cam-Tu Nguyen `[通讯]` (Nanjing University)

**通讯引用:** 3560 | [OpenAlex ID](https://openalex.org/A5060261448)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于隔离核的无学习二进制嵌入方法（IKE），将LLM高维嵌入映射为低维二进制码以加速检索。

**💡 创新点**

核心创新在于通过集成多样化随机划分（iForest或改进的Voronoi）实现高熵、全覆盖且可调节长度的二进制编码，且不需要任何训练。

**🔧 技术方法**

利用隔离核、iForest、Voronoi图、二进制位运算等技术实现嵌入转换与相似度计算。

**📊 数据集**

在MTEB的HotpotQA、FiQA2018、FEVER-HN、Touche2020.V3，以及Istella22、TREC DL 23等六个文本检索数据集上进行评测。

**📈 对比分析**

与原始LLM嵌入、CSR以及其他无学习压缩方法（rpLSH、ScaNN、PQfs）对比，IKE在检索速度提升12–16.7×、内存压缩8–16×的同时，保持或优于LLM级别的MRR@10与nDCG@10，在HNSW ANN下比对手高达10×吞吐量。

**⚠️ 局限性**

局限性：在跨模态检索（如图像-文本）中的效果尚不理想，主要受模态差异影响。

---

## 178. KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education

**arXiv ID:** 2601.09156 | [PDF](https://arxiv.org/pdf/2601.09156v1)

**作者:** Woojin Kim `[一作]` (Korea University), Hyeoncheol Kim `[通讯]` (Korea University)

**通讯引用:** 2260 | [OpenAlex ID](https://openalex.org/A5021651278)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出KTCF方法，给知识跟踪模型生成可解释的反事实解释，并将其转化为可操作的学习步骤。

**💡 创新点**

创新点在于将知识概念关系融入反事实生成，并通过后处理实现与教学目标对齐的行动序列。

**🔧 技术方法**

采用深度学习知识跟踪（DKT）、Adam优化、图搜索（Dijkstra）与TSP变体，以及多种反事实初始化策略。

**📊 数据集**

使用XES3G5M大规模数学交互数据集，包含1,175个知识概念及其关系图。

**📈 对比分析**

与Wachter和DiCE基线相比，KTCF在有效性（+28.3%）、稀疏度（-26%）和可操作性（100%可操作）上表现优异，生成时间更短。

**⚠️ 局限性**

局限在于对初始化策略敏感，缺乏真实用户评估，且未结合大语言模型进一步改进说明文本。

---

## 179. SSVP: Synergistic Semantic-Visual Prompting for Industrial Zero-Shot Anomaly Detection

**arXiv ID:** 2601.09147 | [PDF](https://arxiv.org/pdf/2601.09147v1)

**作者:** Chenhao Fu `[一作]` (Beijing University of Posts and Telecommunications), Xuelong Li `[通讯]` (Institute of Artificial Intelligence China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于视觉‑语言模型的零样本工业缺陷检测框架SSVP，融合CLIP的语义通用性与DINOv3的细粒度结构信息，实现无目标域监督的缺陷识别与定位。

**💡 创新点**

核心创新在于三大模块：HSVS实现层次化语义‑视觉协同融合，VCPG通过VAE与交叉注意力生成视觉条件下的动态提示，VTAM则采用双门混合专家机制对全局与局部得分进行自适应校准，从而解决传统方法的语义泛化、提示单一与全局-局部不匹配等问题。

**🔧 技术方法**

技术实现包括CLIP与DINOv3特征提取、双路径交叉注意力、变分自编码器生成视觉潜在空间、交叉模态注意力注入提示、混合专家融合、双门调度、焦点损失与BCE监督等多种先进方法。

**📊 数据集**

实验覆盖七大工业缺陷基准：MVTec-AD、VisA、BTAD、KSDD2、RSDD、DAGM和DTD-Synthetic，以验证模型在多种纹理与对象缺陷场景下的鲁棒性。

**📈 对比分析**

与WinCLIP、APRIL-GAN、AnomalyCLIP、AdaCLIP、Bayes-PFL等现有零样本检测方法相比，SSVP在图像层面MVTec-AD AUROC达93.0%，RSDD 98.5%，在七个基准上均取得SOTA成绩，显著提升了缺陷识别与定位的准确率。

**⚠️ 局限性**

主要局限在于双模视觉骨干和VAE生成过程导致推理速度慢，难以满足实时边缘部署需求；未来计划通过知识蒸馏压缩模型与测试时适应技术提升效率。

---

## 180. PrivacyReasoner: Can LLM Emulate a Human-like Privacy Mind?

**arXiv ID:** 2601.09152 | [PDF](https://arxiv.org/pdf/2601.09152v1)

**作者:** Yiwen Tu `[一作]` (University of California San Diego), Haojian Jin `[通讯]` (University of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于大型语言模型的代理，用来模拟单个用户在面对隐私相关新闻时的思考过程，并生成与该用户相似的评论。

**💡 创新点**

创新点在于将Contextual Integrity与APCO框架与工作记忆理论相结合，构建用户专属的“隐私心智”，通过结构化的隐私记忆抽取与上下文过滤实现个性化、可解释的隐私关注预测。

**🔧 技术方法**

主要技术包括：大型语言模型（LLM）进行隐私记忆抽取和推理；上下文过滤机制（Contextual Filter）动态激活相关隐私取向；LLM-as-a-Judge评估器用于多标签隐私关注分类；对比实验中的基线模型包括Naive、Privacy Persona和RAG。

**📊 数据集**

使用来自Hacker News的真实讨论数据（包含多领域的隐私话题），并参考已有的14类隐私关注税onomies。

**📈 对比分析**

与基线比较实验显示，该方法在Accuracy、Recall和Macro‑F1三项指标上均显著优于Naive、Persona和RAG，最高的Macro‑F1达到约0.47（相较于RAG的0.45）。此外在域迁移和用户历史稀缺情形下仍保持相对稳健的性能。

**⚠️ 局限性**

主要限制包括：数据来源局限于技术圈子，可能无法代表更广泛人群；LLM-as-a-Judge本身可能带有偏差；实验仅处理文本评论，未考虑多模态或行为数据；在动态、复杂的对话场景中验证仍待进一步研究。

---

## 181. CLIDD: Cross-Layer Independent Deformable Description for Efficient and Discriminative Local Feature Representation

**arXiv ID:** 2601.09230 | [PDF](https://arxiv.org/pdf/2601.09230v1)

**作者:** Haodi Yao `[一作]` (Harbin Institute of Technology), Yao Su `[通讯]` (Beijing Institute for General Artificial Intelligence)

**通讯引用:** 480 | [OpenAlex ID](https://openalex.org/A5102740429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种跨层独立可变形描述方法（CLIDD），通过直接从多尺度特征层采样生成高辨别力的局部描述子，避免了构建统一稠密特征图的高成本。

**💡 创新点**

创新点在于：①跨层预测器（CL-Predictor）利用多尺度特征点信息生成精确采样偏移；②层独立采样器（LI-Sampler）在不合并特征的前提下实现多层稀疏采样；③硬件感知的核融合实现高吞吐量；④可伸缩轻量化网络与多任务学习（度量学习 + 知识蒸馏）构建多种模型尺寸。

**🔧 技术方法**

技术方法包括：卷积网络骨干（ResNet块）生成 1/2、1/8、1/32 分辨率特征；可变形采样；DualSoftmax 损失、Orthogonal-Procrustes 蒸馏损失、UnfoldSoftmax 检测损失；自定义 GPU 核融合；使用 AdamW 训练；实现 TensorRT 推理。

**📊 数据集**

训练与评估数据集：MegaDepth、ScanNet、IMC2022、HPatches、Aachen Day‑Night v1.1、InLoc、Aachen Day‑Night、Aachen Day‑Night v1.1、InLoc。

**📈 对比分析**

与 SuperPoint、DISK、ALIKED、AWDesc、DeDoDe、XFeat、EdgePoint 等 SOTA 方法对比，CLIDD 在所有基准（平面变换、相机姿态估计、视觉定位）上均取得更高的匹配精度，并在 Jetson Orin‑NX 等边缘设备上实现 200–880 FPS，模型尺寸可低至 0.004M，压缩率高达 99.7%。

**⚠️ 局限性**

局限性：在高密度无 NMS 的采样场景下，特征点易聚集，导致空间覆盖度下降；轻量化模型在极低显存或极小算力设备上仍需进一步优化；对极端光照或动态场景的鲁棒性仍待进一步验证。

---

## 182. A $4/3$ ratio approximation algorithm for the Tree Augmentation Problem by deferred local-ratio and climbing

**arXiv ID:** 2601.09219 | [PDF](https://arxiv.org/pdf/2601.09219v1)

**作者:** Guy Kortsarz `[一作]` `[通讯]` (Rutgers University), Guy Kortsarz (Rutgers University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种新的4/3近似算法，用来解决树增强问题（TAP），该算法通过“延迟局部比例”与“攀爬阶梯”技术，能够在 O(m√n) 时间内完成求解。

**💡 创新点**

创新点在于引入了延迟局部比例技术，放宽了传统局部比例中对互不相交的限制，并通过“金票”与“黄金票”构造对覆盖的精细计分，从而实现了 4/3 的近似比。

**🔧 技术方法**

主要技术包括：延迟局部比例（Deferred Local Ratio）、集合覆盖简化与影子集（shadow set）扩展、金票/黄金票分配、树的半闭合与递归剪枝、以及基于匹配的叶子闭合判定。

**📊 数据集**

论文未使用真实数据集，而是针对理论模型中的树与边集合进行分析与证明，实验验证以经典合成实例为主。

**📈 对比分析**

与之前最佳的 1.393 近似（Traub & Zenklusen 2025）相比，本算法实现了更优的 4/3 近似比，并且时间复杂度更低，显著提升了求解效率。

**⚠️ 局限性**

局限性包括：仅适用于无权的树增强问题；算法在特殊结构或高密度图上可能仍然需要改进；并且对实际大规模工业应用的性能评估仍缺乏实验数据。

---

## 183. Relational Hoare Logic for High-Level Synthesis of Hardware Accelerators

**arXiv ID:** 2601.09217 | [PDF](https://arxiv.org/pdf/2601.09217v1)

**作者:** Izumi Tanaka `[一作]` (University of Tokyo), Naoki Kobayashi `[通讯]` (University of Tokyo)

**通讯引用:** 7848 | [OpenAlex ID](https://openalex.org/A5100772473)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于关系Hoare逻辑的自动翻译框架，能够将朴素的HLS程序自动转化为使用 on‑chip 缓冲与 stream 处理的高效硬件加速器代码，并通过形式化证明保证语义保持。

**💡 创新点**

创新点包括：① 将关系Hoare逻辑用于程序转换，提供可证明的语义保持；② 自动识别复杂内存访问模式并插入缓冲、转换为 stream，消除手工注释与代码重构需求；③ 通过对访问模式形状的限制实现可自动化的验证条件生成与 SMT 求解。

**🔧 技术方法**

技术手段：关系Hoare逻辑与 Hoare 逻辑条件生成、SMT 求解器 Z3、Vitis HLS 与 Vivado 综合、AXI DMA 及 AXI Stream 接口、两维线性缓冲（line buffer）等。

**📊 数据集**

数据集：改编自 MachSuite 的一组基准程序（Filter、Filter‑Dilated、Divide、MatVecMul、MatAdd、Stencil‑2D、Filter‑2D、KMP、GeMM、SpMV 等）及其 Rev、Skip 版本；并使用自定义 C‑like DSL 进行实验。

**📈 对比分析**

评估方法：在 AMD Kria KV260 FPGA 上使用 Vitis HLS 与 Vivado 生成比特流，对比原始程序与翻译后程序在执行时间、吞吐量与资源利用率（LUT、BRAM、DSP 等）上的差异。结果显示，在 Filter、Filter‑2D、Stencil‑2D 等基准上，性能提升可达 5 倍以上；对已是顺序访问的基准提升有限；Area 增幅适中，功耗基本不变。

**⚠️ 局限性**

局限性：仅支持访问模式为线性递增/递减/固定步长的序列；不支持非线性或列主序访问；实现中对数组访问形状做了硬性限制；对多维数组仅实现首维缓冲；需满足循环不变式可自动推导，复杂控制流或递归结构暂不覆盖。

---

## 184. Honesty-Aware Multi-Agent Framework for High-Fidelity Synthetic Data Generation in Digital Psychiatric Intake Doctor-Patient Interactions

**arXiv ID:** 2601.09216 | [PDF](https://arxiv.org/pdf/2601.09216v1)

**作者:** Xinyuan Zhang `[一作]` (Chinese University of Hong Kong), Juexiao Zhou `[通讯]` (Chinese University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个多代理框架，生成包含病人档案、问卷、访谈、诊断等完整合成精神科接诊记录，特别模拟了病人隐瞒或夸大症状的动态诚实状态。

**💡 创新点**

创新点包括：1）明确建模病人诚实度的主题依赖状态；2）通过四角色（评估者、评估者、诊断者等）协同模拟，生成完整接诊轨迹；3）提供公开可用的含诚实信号的合成数据集。

**🔧 技术方法**

使用大型语言模型（LLM）与链式思考（CoT）模块、基于CAMEL架构的多代理系统，结合自定义状态更新与推理。

**📊 数据集**

基础数据集为DAIC-WOZ采访，进一步通过LLM提取标签并合成隐瞒/夸大特征，生成377条合成会话。

**📈 对比分析**

通过诊断一致性评估、CoT消融、专家人工评估和LLM对比，诊断准确率达86.8%，严重度准确率提升25.9%，与GPT‑5.2等基线相比在真实性和抗欺骗表现上更佳。

**⚠️ 局限性**

限制包括：只模拟单一主诊断，未考虑共病；人类评估样本有限，评估主观性高；合成数据虽逼真但仍基于LLM，可能存在偏差。

---

## 185. SpikeVAEDiff: Neural Spike-based Natural Visual Scene Reconstruction via VD-VAE and Versatile Diffusion

**arXiv ID:** 2601.09213 | [PDF](https://arxiv.org/pdf/2601.09213v1)

**作者:** Jialu Li `[一作]` (Hong Kong University of Science and Technology), Taiyan Zhou `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1124 | [OpenAlex ID](https://openalex.org/A5101938238)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SpikeVAEDiff两阶段框架，将神经尖峰信号映射到VDVAE和Versatile Diffusion，生成高分辨率、语义一致的图像重建。

**💡 创新点**

首次将尖峰神经数据与预训练的VDVAE、CLIP和Versatile Diffusion相结合，并通过BLIP‑2生成字幕以实现跨模态条件；同时探讨不同视觉脑区对重建质量的影响。

**🔧 技术方法**

使用Very Deep Variational Autoencoder进行低分辨率预估，回归模型映射尖峰到VDVAE潜变量、CLIP视觉与文本特征，再用Versatile Diffusion进行图像到图像生成；采用BLIP‑2生成图像描述。

**📊 数据集**

使用Allen Visual Coding – Neuropixels 电极尖峰数据（包含6个视觉脑区、20k+细胞），并以120秒电影剪辑（3600帧）为刺激。

**📈 对比分析**

与基于fMRI的两种重建方法对比，利用相同刺激评估重建质量；SpikeVAEDiff在细节保留、语义准确度和分辨率上优于fMRI方法，且可生成512×512图像。

**⚠️ 局限性**

回归模型受限于有限尖峰样本，易在复杂背景或遮挡时失真；对尖峰数据收集侵入性强，且缺乏大规模数据集导致泛化受限。

---

## 186. Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation

**arXiv ID:** 2601.09212 | [PDF](https://arxiv.org/pdf/2601.09212v1)

**作者:** Xingyao Li `[一作]` (National University of Singapore), Hui Ji `[通讯]` (National University of Singapore)

**通讯引用:** 8704 | [OpenAlex ID](https://openalex.org/A5030046423)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在自动回归图像生成中，提出了一种基于理论的放松式投机解码（annealed relaxed speculative decoding）方案，显著提升了推理速度和图像质量的权衡。

**💡 创新点**

创新点在于：① 用总变差距离上界推导出令放松解码最小化偏差的最优重采样分布；② 通过扰动分析揭示放松阈值的退火性质，并提出逐步递减的接受概率调度；③ 将两项结合形成新的解码框架。

**🔧 技术方法**

使用的技术包括：投机解码（Speculative Decoding）、总变差距离分析、最优重采样分布推导、扰动分析与退火调度、以及对LlamaGen‑XL和Lumina‑mGPT模型的并行前向推理。

**📊 数据集**

在MS‑COCO 2017验证集（约5000条文本描述）上进行实验，目标模型分别为LlamaGen‑XL（775M）和Lumina‑mGPT（7B）。

**📈 对比分析**

与传统的Eagle‑1（lossless SD）和LANTERN++（放松式SD）对比，Annealed‑Relaxed SD在保持类似FID/CLIP/IR得分的同时，推理延迟降低1.5–3.1倍；在相同延迟下，FID略低或相当。

**⚠️ 局限性**

局限性包括：仍需手工调节放松预算δ和退火速率ν；方法主要验证在两类模型和MS‑COCO数据上，尚未在更大规模或更高分辨率图像上系统评估；对极端高速度需求时图像质量仍会出现轻微退化。

---

## 187. Architecture inside the mirage: evaluating generative image models on architectural style, elements, and typologies

**arXiv ID:** 2601.09169 | [PDF](https://arxiv.org/pdf/2601.09169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 188. Affostruction: 3D Affordance Grounding with Generative Reconstruction

**arXiv ID:** 2601.09211 | [PDF](https://arxiv.org/pdf/2601.09211v1)

**作者:** Chunghyun Park `[一作]` (POSTECH), Minsu Cho `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在有限视角的RGBD图像下，提出一种生成式框架，先通过多视角稀疏体素融合重建完整三维几何，再在完整形状上对文本查询做可执行性（affordance）热图预测，并利用预测的热图进行主动视角选择；

**💡 创新点**

① 通过稀疏体素融合在保持常数令牌复杂度的前提下实现多视角生成式重建；② 采用流式生成模型捕捉可执行性分布的多模态不确定性；③ 将可执行性预测与主动视角采样相结合，实现功能导向的视角采集；

**🔧 技术方法**

使用基于TRELLIS的Flow Transformer（多视角稀疏体素融合+DINOv2特征）、流式生成（flow-based）Affordance模型（CLIP文本编码）、稀疏体素表示、流式匹配训练、主动视角采样策略；

**📊 数据集**

训练数据：3D‑FUTURE、HSSD、ABO；评估数据：Toys4k（重建），Affogato测试集（可执行性预测）；

**📈 对比分析**

与TRELLIS、MCC、OpenAD、PointRefer、Espresso‑3D等方法对比，重建上取得32.67 IoU（比TRELLIS提升67.7%，比MCC提升54.8%），可执行性在完整几何上取得19.1 aIoU（比Espresso‑3D提升40.4%），在单视RGBD下通过主动视角提升至12.46 aIoU；

**⚠️ 局限性**

仅限单物体场景，需深度输入；评估主要基于合成数据，缺乏真实世界复杂环境验证；主动视角采样仍受预设候选视角分布限制；模型在大视角预算下仍需多次采样，计算成本相对较高。

---

## 189. Mikasa: A Character-Driven Emotional AI Companion Inspired by Japanese Oshi Culture

**arXiv ID:** 2601.09208 | [PDF](https://arxiv.org/pdf/2601.09208v1)

**作者:** Miki Ueno `[一作]` `[通讯]` (Kyoto College of Graduate Studies for Informatics), Miki Ueno (Kyoto College of Graduate Studies for Informatics)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一个基于日本 Oshi 文化的情感 AI 伴侣 Mikasa，重点强调角色一致性与关系框架，以提升长期用户参与度。

**💡 创新点**

将角色设计视为 AI 伴侣的功能性基础而非装饰，提出“持续角色承诺”和“明确关系定位（伙伴）”的设计原则，解决传统伴侣系统的角色模糊与情感疲劳。

**🔧 技术方法**

采用 GPT‑4o‑mini 文本生成、ElevenLabs 语音合成、iOS 本地语音识别、SQLite 多层记忆、Live2D/Unity 视觉模型，并通过客户端‑服务器架构实现低延迟双向语音对话。

**📊 数据集**

主要使用公开语言模型 API 和自定义对话日志；评估基于八名学生的探索性问卷与作者六个月的日常使用观察，未使用公开标注数据集。

**📈 对比分析**

通过问卷对比用户对专用伴侣 App 与通用对话系统的偏好，发现用户更重视关系自定义与语言自由；虽然未给出量化指标，但探索性结果显示 Mikasa 在保持角色连贯性与关系稳定性方面优于传统系统。

**⚠️ 局限性**

样本规模小（N=8）且仅限文化背景；角色与职业设定缺乏可复现性；‘伙伴’标签可能不适用于所有用户；缺乏大规模纵向验证与跨文化比较。

---

## 190. Optimizing View Change for Byzantine Fault Tolerance in Parallel Consensus

**arXiv ID:** 2601.09184 | [PDF](https://arxiv.org/pdf/2601.09184v1)

**作者:** Yifei Xie `[一作]` (University of Edinburgh), Jane Hillston `[通讯]` (University of Edinburgh)

**通讯引用:** 7003 | [OpenAlex ID](https://openalex.org/A5035860397)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种视图更换优化（VCO）模型，旨在通过主动选择领导者和备份节点来显著提升并行拜占庭容错共识的吞吐量和延迟。

**💡 创新点**

创新点在于首次将视图更换过程建模为混合整数规划，并利用改进的 Benders 分解和 1‑median 问题求解，实现了在多委员会环境下的最优领导者轮换与节点重新分配。

**🔧 技术方法**

技术手段包括混合整数规划（MIP）、改进 Benders 分解、子问题的 1‑median 变形、以及基于通信延迟与节点故障概率的多场景权重评估。

**📊 数据集**

实验数据来源于在 Microsoft Azure 云平台上构建的 8 vCPU/32GB 机器测试床，模拟 40~200 个共识节点并随机设置高延迟节点以诱发故障。

**📈 对比分析**

与随机分组、BL‑MILP 优化、SP、FD 等基线相比，VCO 在正常和故障场景下吞吐量提升 17.9%–45.3%，延迟降低 24.1% 以上，显示出更优的可扩展性和鲁棒性。

**⚠️ 局限性**

局限性包括：仅考虑单一领导者失效场景、模型假设通信延迟与故障概率已知、实验仅在模拟云环境中验证，未在真实区块链或大规模部署中进行评估。

---

## 191. DP-FEDSOFIM: Differentially Private Federated Stochastic Optimization using Regularized Fisher Information Matrix

**arXiv ID:** 2601.09166 | [PDF](https://arxiv.org/pdf/2601.09166v1)

**作者:** Sidhant R. Nair `[一作]` (Indian Institute of Technology Delhi), Mrinmay Sen `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5044190873)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于服务器端的二阶优化框架DP‑FedSOFIM，用于差分隐私联邦学习，以提高收敛速度和模型精度。

**💡 创新点**

创新点在于将Fisher信息矩阵的自然梯度预处理完全迁移至服务器，利用Sherman‑Morrison公式实现O(d)内存与计算，避免传统二阶方法在客户端的O(d^2)开销，同时保持（ε,δ）隐私。

**🔧 技术方法**

采用的技术包括差分隐私梯度裁剪与高斯噪声、Fisher信息矩阵近似、指数移动平均、Sherman‑Morrison矩阵求逆、自然梯度预条件以及隐私后处理。

**📊 数据集**

实验使用CIFAR‑10数据集，采用预训练的ResNet‑20特征提取器并只训练最终线性分类头。

**📈 对比分析**

与传统DP‑FedGD对比，DP‑FedSOFIM在所有隐私预算下均取得更高准确率，最优情况下提升约3.12%，在严格隐私下仍能在第30轮前超越基线。

**⚠️ 局限性**

局限性包括：仅在凸/线性头下理论分析，非凸深度网络的收敛与泛化尚未证明；对极端噪声时FIM估计可能不稳定；需要对服务器端进行额外计算与存储，且尚未评估用户级隐私。

---

## 192. Multi-Teacher Ensemble Distillation: A Mathematical Framework for Probability-Domain Knowledge Aggregation

**arXiv ID:** 2601.09165 | [PDF](https://arxiv.org/pdf/2601.09165v1)

**作者:** Aaron R. Flouro `[一作]`, Shawn P. Chadwick `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

提出了一套针对多教师知识蒸馏的公理化框架，并给出了该框架下的操作符存在性、非唯一性以及多种理论保证（方差降低、Jensen界、log-loss优势、安全衰减等）

**💡 创新点**

创新点在于：①首次给出五条核心公理，定义多教师聚合算子所必须满足的性质；②证明满足这些公理的算子族非唯一且存在；③在此公理体系下实现了与单教师蒸馏完全独立的理论保证，展示多教师蒸馏可以系统性减少方差与监督偏差；④引入温度一致性公理，实现对教师温度的异质化调控

**🔧 技术方法**

技术手段包括：算子理论与凸分析；信息理论投影（KL、Rényi、熵正则化等）；方差与协方差分解；Jensen不等式与对数损失分析；容量理论证明高参数学生可逼近多教师聚合分布

**📊 数据集**

本文主要为理论研究，并未在实验中使用具体数据集；若需验证，可将公开的语言模型（如GPT、PaLM、Claude等）作为教师，或使用标准NLP数据集（如GLUE、SuperGLUE）进行实验，但本文未给出具体数据集

**📈 对比分析**

与单教师蒸馏进行理论对比：多教师蒸馏在方差降低、补充知识、温度灵活性、安全性提升等方面优于单教师；理论上，学生可取得比平均教师更低的log‑loss；但论文未给出实验性能指标，只给出理论上限与边界

**⚠️ 局限性**

局限性：①缺乏经验验证与具体实现细节；②不提供泛化性能保证，泛化仍取决于学生网络的容量、正则化与优化过程；③对教师权重的选择仍需经验或后续工作；④对协方差较大的教师集成效果分析仍有限

---

## 193. Deep Learning-based Binary Analysis for Vulnerability Detection in x86-64 Machine Code

**arXiv ID:** 2601.09157 | [PDF](https://arxiv.org/pdf/2601.09157v1)

**作者:** Mitchell Petingola `[一作]` `[通讯]` (Algoma University), Mitchell Petingola (Algoma University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了基于x86-64机器码的深度学习漏洞检测方法，并对比了顺序CNN与图卷积网络两种模型。

**💡 创新点**

创新点在于直接使用原始机器码而非汇编或中间表示，证明机器码足以提取漏洞特征，并展示了控制流图对性能的显著提升。

**🔧 技术方法**

采用了1D-CNN、Self-Attention、图卷积网络（GCN）以及Top-K注意力池化等深度学习技术。

**📊 数据集**

使用了基于LLM生成的FormAI‑v2数据集（约33万条C程序），并结合ESBMC验证得到的漏洞标签。

**📈 对比分析**

与传统基于汇编的Bin2Vec相较，图卷积模型在三种漏洞类型上分别达成90%以上的准确率，优于顺序模型且与Bin2Vec相近。

**⚠️ 局限性**

局限性包括仅覆盖三种漏洞类型、缺乏数据流特征、对整数溢出表现不佳、模型可解释性不足以及仅适用于x86-64架构。

---

## 194. From Snow to Rain: Evaluating Robustness, Calibration, and Complexity of Model-Based Robust Training

**arXiv ID:** 2601.09153 | [PDF](https://arxiv.org/pdf/2601.09153v1)

**作者:** Josué Martínez-Martínez `[一作]` (MIT Lincoln Laboratory), Rajmonda Caceres `[通讯]` (MIT Lincoln Laboratory)

**通讯引用:** 392 | [OpenAlex ID](https://openalex.org/A5007510609)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文评估了多种基于模型的和混合数据增强方法在自然雪雨噪声下的鲁棒性和校准性能，并与传统的随机和对抗训练进行对比。

**💡 创新点**

创新点在于：①系统比较了模型生成的噪声与传统增强方法在雪雨两类真实噪声上的效果；②提出并验证了随机+对抗混合策略（MDAT、MRAT）的鲁棒性与效率权衡；③探讨了训练时噪声严重程度对鲁棒性与校准的影响。

**🔧 技术方法**

使用了基于MUNIT风格的变异模型进行数据增强，结合对抗梯度上升（PGD）在噪声空间的优化，训练过程采用交叉熵损失并与AugMix、AT等基线方法进行对比。

**📊 数据集**

实验数据集为CURE‑TSR（交通标志识别），在其雪（Snow）与雨（Rain）两种自然噪声、5个严重级别上进行评估。

**📈 对比分析**

与Vanilla、AT、AugMix等基线相比，模型基方法（MDA、MRT、MAT）在所有严重度下都显著提升准确率与校准（ECE下降约10‑15%），MAT在最严噪声下获得最高准确率（约63%）和最佳校准；MDAT/MRAT虽在中等噪声下略有优势，但在极端噪声下与MAT相近，且计算成本高于MDA。

**⚠️ 局限性**

限制主要体现在：①仅在单一数据集和两种噪声类型上验证；②使用简单CNN骨干和单一生成模型，缺乏对更大规模网络或其他生成模型的验证；③混合策略的计算开销较大，实际部署时需进一步优化。

---

## 195. Single-Round Clustered Federated Learning via Data Collaboration Analysis for Non-IID Data

**arXiv ID:** 2601.09304 | [PDF](https://arxiv.org/pdf/2601.09304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 196. Discrete Solution Operator Learning for Geometry-Dependent PDEs

**arXiv ID:** 2601.09143 | [PDF](https://arxiv.org/pdf/2601.09143v1)

**作者:** Jinshuai Bai `[一作]` (Tsinghua University), Xi-Qiao Feng `[通讯]` (Tsinghua University)

**通讯引用:** 25721 | [OpenAlex ID](https://openalex.org/A5100599549)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种针对几何依赖 PDE 的离散解算器学习框架（DiSOL），通过学习离散求解流程（局部贡献、跨尺度组装、隐式求解）来近似 PDE 解。

**💡 创新点**

创新点在于将传统离散数值求解的程序化结构转化为可学习的可微网络，保留几何变化下的程序级一致性，从而突破连续算子在几何突变场景下的局限。

**🔧 技术方法**

使用卷积网络结合 FiLM 条件融合、多尺度组装模块以及轻量化求解器实现离散算子学习，并对模型进行端到端梯度优化。

**📊 数据集**

在四类基准数据集上进行评估：几何变形的泊松方程、输运支配的扩散-对流方程、二维线性弹性问题以及时变热传导问题，所有数据均在固定网格上生成。

**📈 对比分析**

与 DeepONet、Fourier Neural Operator 等连续算子方法在同等参数和训练设置下对比，DiSOL 在 ID 与 OOD 场景均实现更低的相对 L1/L2 错误、收敛更快，尤其在几何突变与边界条件不连续时表现更稳健。

**⚠️ 局限性**

局限性包括：仅在固定网格分辨率下训练与推断；未考虑自适应网格或多分辨率；缺乏与连续算子或物理约束的混合方法，且对更复杂三维几何的泛化尚未验证。

---

## 197. Frequency Error-Guided Under-sampling Optimization for Multi-Contrast MRI Reconstruction

**arXiv ID:** 2601.09316 | [PDF](https://arxiv.org/pdf/2601.09316v1)

**作者:** Xinming Fang `[一作]` (Shanghai University), Guixu Zhang `[通讯]` (East China Normal University)

**通讯引用:** 4007 | [OpenAlex ID](https://openalex.org/A5060120202)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种联合优化采样模式与多对比度MRI重建的框架 JUF-MRI，利用频率误差先验（FEP）实现对参考对比度信息的高效利用。

**💡 创新点**

创新点在于：①使用条件扩散模型生成目标对比度图像并得到频率误差先验；②将 FEP 融入深度展开重建网络，实现采样模式与重建网络的联合学习；③通过空间对齐模块和参考特征分解进一步提升跨模态信息传递与物理可解释性。

**🔧 技术方法**

核心技术包括：条件扩散模型（CDM）产生频率误差先验；基于 TITAN 的深度展开优化网络；多尺度频域损失；可学习的连续采样掩码与二值化策略；空间对齐网络（DAS）。

**📊 数据集**

实验使用三大公开数据集：IXI（T1、T2、PD），BraTS2018（T1、T2），FastMRI（FSPD、PD），在 4×、8×、10×、30× 加速率及 1D 逐行与可学习采样模式上评估。

**📈 对比分析**

与多种基准方法（UNet、Multi-UNet、CUNet、MDUNet、Restormer、VANet、MC-DuDoN 及其 LOUPE 版本）对比，JUF-MRI 在 PSNR/SSIM 上普遍高于 1–2 dB，尤其在高加速率（30×）下仍保持 0.3–0.6 dB 的优势，并在视觉上恢复了更多细节与对比。

**⚠️ 局限性**

局限性包括：①扩散模型训练成本高，推理时未使用但训练阶段消耗较大；②仅在单线圈数据上验证，缺乏多线圈通用性；③频率误差先验的潜在更深层应用尚未充分探索。

---

## 198. On-Device Large Language Models for Sequential Recommendation

**arXiv ID:** 2601.09306 | [PDF](https://arxiv.org/pdf/2601.09306v1)

**作者:** Xin Xia `[一作]` (University of Queensland), Shane Culpepper `[通讯]` (University of Queensland)

**通讯引用:** 2824 | [OpenAlex ID](https://openalex.org/A5070937840)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 OD-LLM 框架，将大型语言模型压缩后部署到边缘设备，实现顺序推荐。

**💡 创新点**

创新点在于结合 Token 协方差归一化、低秩 SVD 压缩和渐进层对齐三大技术，专为顺序推荐的行为特征设计。

**🔧 技术方法**

使用 Cholesky 归一化、SVD 低秩分解、渐进对齐算法，以及与 GPTQ、SparseGPT 的对比实验。

**📊 数据集**

使用 Amazon Instruments、Games、Arts 三个真实商品评论数据集进行实验。

**📈 对比分析**

与传统推荐模型（Caser、HGN 等）及 LLM 压缩方法（GPTQ、SparseGPT）对比，OD-LLM 在 50% 压缩率下保持或超过 LC-Rec 的 HR/NDCG，并在 GPU/CPU 上推理速度比 GPTQ 快 3.4 倍。

**⚠️ 局限性**

局限在于极端压缩比例下性能下降、对校准样本量敏感、尚未在多种 LLM 上全面验证。

---

## 199. Blue Teaming Function-Calling Agents

**arXiv ID:** 2601.09292 | [PDF](https://arxiv.org/pdf/2601.09292v1)

**作者:** Greta Dolcetti `[一作]` (Ca Foscari University of Venice), Sergio Maffeis `[通讯]` (Imperial College London)

**通讯引用:** 2178 | [OpenAlex ID](https://openalex.org/A5043151499)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了四款开源功能调用 LLM 对三种攻击（Direct Prompt Injection、Simple Tool Poisoning、Renaming Tool Poisoning）的鲁棒性，并测试了八种防御方法的效果。

**💡 创新点**

首次针对开源功能调用模型提出 Renaming Tool Poisoning 攻击，并为其设计了 Tool Obfuscation 防御，系统性验证了多种防御在不同攻击下的表现。

**🔧 技术方法**

采用 Cosine Similarity、Tool Obfuscation、Description Rewriting、Watermarking、LLM‑Guardian（Query Jailbreak、Answer Consistency、Tools Jailbreak、Tools Consistency）等技术，并在 Ollama/DSPy 平台上进行实验。

**📊 数据集**

使用 Berkeley Function Calling Leaderboard 数据集（172 条问答对）以及 Qwen2.5‑Coder 生成的工具实现作为实验数据。

**📈 对比分析**

通过比较无攻击基线与攻击场景下的准确率与攻击成功率（ASR），发现 DPI 在大部分模型上 ASR 超过 90%，STP 与 RTP 影响各异；防御中 Description Rewriting 与 Watermarking 能将 ASR 降至零，但多数防御仍伴随高误报或准确率下降。

**⚠️ 局限性**

防御方案普遍存在高误报、对某些攻击无效或需人工调参，LLM‑Guardian 方案误报率偏高，缺乏通用性，整体模型安全性仍无法在所有攻击下实现全面防护。

---

## 200. TIDI-GS: Floater Suppression in 3D Gaussian Splatting for Enhanced Indoor Scene Fidelity

**arXiv ID:** 2601.09291 | [PDF](https://arxiv.org/pdf/2601.09291v1)

**作者:** Sooyeun Yang `[一作]` (State University of New York), Jongseong Brad Choi `[通讯]` (State University of New York)

**通讯引用:** 469 | [OpenAlex ID](https://openalex.org/A5061176269)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

为3D Gaussian Splatting训练提供一种轻量插件框架，消除浮点漂浮体（floaters），提升几何准确性与视觉质量。

**💡 创新点**

创新点在于基于多视角证据、空间孤立性与细节保护的三阶段浮点体修剪（TIDI）以及无需重构架构的轻量蒙版深度正则化。

**🔧 技术方法**

使用3D Gaussian Splatting渲染、可微光栅化、可学习的重要性权重、EMA梯度统计、单目深度正则化（MiDaS/Depth‑Anything）和LPIPS感知损失等技术。

**📊 数据集**

在Tanks & Temples室内序列（Auditorium、Ballroom、Church、Museum）及Mip‑NeRF 360的Room场景上进行实验。

**📈 对比分析**

与原始3DGS、LP‑GS、Pixel‑GS、Micro‑Splatting对比，使用PSNR、SSIM、LPIPS、背景一致性、轮廓泄漏与深度稳定性等指标，TIDI‑GS在保持相近模型大小与训练时间的同时，显著降低LPIPS、浮点体并提升深度稳定性。

**⚠️ 局限性**

局限性主要是针对室内有限深度场景，依赖多视角证据与单目深度先验；在开放式、远景或天空等无界深度环境下效果不佳。

---

## 201. $A^3$-Bench: Benchmarking Memory-Driven Scientific Reasoning via Anchor and Attractor Activation

**arXiv ID:** 2601.09274 | [PDF](https://arxiv.org/pdf/2601.09274v1)

**作者:** Jian Zhang `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 73099 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 A^3-Bench，基于 Anchor 与 Attractor 双尺度记忆激活的科学推理基准，并构建了 SAPM 记忆注解流程与 AAUI 评估指标。

**💡 创新点**

创新点在于：①引入人类记忆层次化的 Anchor/Attractor 记忆模型；②设计双尺度评估框架与全新 AAUI 指标；③首次通过记忆激活机制对推理过程进行细粒度评估。

**🔧 技术方法**

使用了 HybridRAG 框架的 Memory Twin‑Needle 激活器与 Context Fabric Composer、Anchor/Attractor 激活范式、LLM 推理与记忆检索技术。

**📊 数据集**

使用了 2,198 题目的科学推理数据集（涵盖数学、物理、化学），来源于 MathVista、OlympiadBench、EMMA、Humanity’s Last Exam，并在 OlympiadBench 上进行迁移测试。

**📈 对比分析**

通过在 10 个 LLM 上比较三种记忆范式（vanilla、full、gold），发现记忆激活可提升约 13–22% 的准确率，尤其在难题和竞赛级别上显著提升，且推理时长平均下降 2–3 秒，AAUI 与准确率高度相关。

**⚠️ 局限性**

局限性在于：①仅覆盖三学科，未涉及更广泛领域；②Anchor/Attractor 的人工注解成本高；③模型效果依赖于记忆库的完整性与质量；④目前仍难以覆盖所有推理策略与动态知识更新。

---

## 202. MAXS: Meta-Adaptive Exploration with LLM Agents

**arXiv ID:** 2601.09259 | [PDF](https://arxiv.org/pdf/2601.09259v1)

**作者:** Jian Zhang `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 73099 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了MAXS框架，改进LLM Agent的多工具推理，解决局部贪婪和轨迹不稳定问题。

**💡 创新点**

创新点在于引入短期lookahead评估、优势得分与步/斜率方差的复合价值函数，以及轨迹收敛机制。

**🔧 技术方法**

采用LLM Agent + 代码/搜索工具、rollout lookahead、贝尔曼价值估计、Lyapunov与Lipschitz方差正则、vLLM推理引擎。

**📊 数据集**

使用五大推理基准：MathVista、OlympiadBench、EMMA、TheoremQA、MATH，并在三种多模态LLM（MiMo-VL-7B、Qwen2.5-VL-7B、Qwen2.5-VL-32B）上评测。

**📈 对比分析**

与CoT、ToT、MCTS、Guided Decoding、ϕ-Decoding等方法比较，MAXS在准确率上均有提升且令标记使用量显著降低，达到SOTA。

**⚠️ 局限性**

局限在于需要手动设定lookahead步数、方差阈值，且在极大规模模型或工具多样性不足时可能效果受限。

---

## 203. When to Invoke: Refining LLM Fairness with Toxicity Assessment

**arXiv ID:** 2601.09250 | [PDF](https://arxiv.org/pdf/2601.09250v1)

**作者:** Jing Ren `[一作]` (RMIT University), Feng Xia `[通讯]` (RMIT University)

**通讯引用:** 18946 | [OpenAlex ID](https://openalex.org/A5089615958)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FairToT，一个在推理时通过提示引导的公平性评估与修正框架，用于提升隐式仇恨言论检测中的族群公平性。

**💡 创新点**

创新点在于：1）首次探讨何时触发公平性修正；2）设计可解释的句子级和实体级公平性指标；3）通过三步结构化提示实现无参修正。

**🔧 技术方法**

技术包括：基于 LLM 的提示工程、三步结构化提示（语义一致性检查、实体中立伤害推理、概率分配）、公平性指标计算（SFV、EFD）以及推理时动态触发机制。

**📊 数据集**

使用三大隐式仇恨数据集：Latent Hatred、Offensive Slang、ToxiGen，评估多种基线模型（BERT、HateBERT、DeBERTa、RoBERTa）以及 LLM 后端（GPT‑3.5‑Turbo、Llama‑3.1‑8B‑Instruct）和数据增强方法。

**📈 对比分析**

通过对比未修正与 FairToT 修正的 SFV 和 EFD 得分，实验表明在所有基线和 LLM 上，FairToT 将句子级和实体级公平性指标显著下降（平均 <0.001），同时保持原有预测准确性且仅增加极小的 token 消耗。

**⚠️ 局限性**

局限包括：需要手工设置阈值（C_θ、R_n），对高温度采样不稳健，尚未在多语言或多模态场景中验证。

---

## 204. DeTracker: Motion-decoupled Vehicle Detection and Tracking in Unstabilized Satellite Videos

**arXiv ID:** 2601.09240 | [PDF](https://arxiv.org/pdf/2601.09240v1)

**作者:** Jiajun Chen `[一作]` (Wuhan University), Mi Wang `[通讯]` (Wuhan University)

**通讯引用:** 4953 | [OpenAlex ID](https://openalex.org/A5100742710)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 DeTracker，一种专为未稳定卫星视频设计的联合检测-跟踪框架，解决平台运动与目标运动混合及极小目标表示困难。

**💡 创新点**

创新点在于：①全局‑局部运动解耦（GLMD）模块，先做全局对齐再局部细化，抑制背景漂移；②时序依赖特征金字塔（TDFP）模块，多尺度跨帧融合提升小目标辨识；③无需预先图像稳定化，直接在特征空间完成运动补偿。

**🔧 技术方法**

使用技术包括：DLA34 backbone、双向自注意力实现全局对齐、变形卷积做局部细化、ConvGRU+BiFPN实现时序融合、warp 对齐损失、focal 与回归损失等。

**📊 数据集**

使用数据集：构建的模拟未稳定卫星视频数据集 SDM‑Car‑SU（多方向、多速度）和真实未稳定卫星视频片段。

**📈 对比分析**

与多种 TBD 与 JDT 基线（DeepSORT、CenterTrack、PIFTrack、MOSAIC‑Tracker 等）对比，在 SDM‑Car‑SU U1–U3 上 MOTA 分别为 61.1%、55.3%、52.4%，比第二名提升约 5.1%；在真实视频上 MOTA 45.3%、IDF1 67.8%，显著优于 PIFTrack。

**⚠️ 局限性**

局限性：模型参数约 8.7M、79.8GFLOPs、推理速度 4.86 FPS，尚未完全实时；对极大运动或极低分辨率情况的鲁棒性待验证；遮挡严重时仍可能出现关联错误。

---

## 205. DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion

**arXiv ID:** 2601.09239 | [PDF](https://arxiv.org/pdf/2601.09239v1)

**作者:** Hanlin Zhang `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**通讯引用:** 1975 | [OpenAlex ID](https://openalex.org/A5035185924)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了DSA-Tokenizer，利用双流离散语义与声学token实现语音的语义与声学属性解耦。

**💡 创新点**

创新点在于通过ASR监督的语义token与mel重建监督的声学token、联合重建-重组训练策略及层次Flow Matching解码器，获得严格解耦与跨说话人/跨时长的可重组语音。

**🔧 技术方法**

技术包括HuBERT/SEANet+FSQ离散化、Flow Matching+DiT解码器、ControlNet风格注入、跨模态跨时序注意力、联合重建-重组训练、语音指纹损失与CFG。

**📊 数据集**

数据集：4,000h中英对齐语料训练语义token，Emilia 100k小时中英子集训练声学token与解码器；SeedTTS、LibriSpeech、VoxCeleb1、LibriTTS等用于评估。

**📈 对比分析**

与WavTokenizer、SAC、DualCodec、SpeechTokenizer等基线对比，DSA-Tokenizer在重建与跨句子重组任务中取得更高UTMOS、低WER/CER与更高SIM，证明解耦提升了声学相关LLM任务表现。

**⚠️ 局限性**

局限：推理延迟高（22层DiT+多步采样），仅针对语音，未验证音乐或环境声等其他音频。

---

## 206. Private Links, Public Leaks: Consequences of Frictionless User Experience on the Security and Privacy Posture of SMS-Delivered URLs

**arXiv ID:** 2601.09232 | [PDF](https://arxiv.org/pdf/2601.09232v1)

**作者:** Muhammad Danish `[一作]` (University of New Mexico), Afsah Anwar `[通讯]` (University of New Mexico)

**通讯引用:** 626 | [OpenAlex ID](https://openalex.org/A5064140739)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对通过短信交付的私有 URL 进行大规模安全与隐私评估，发现 701 条 PII 泄露 URL，覆盖 177 个服务。

**💡 创新点**

首次以公共 SMS 网关为视角，构建端到端检测流水线，结合 LLM 与人工验证，揭示令牌泄露、可枚举、超抓取等多种攻击面。

**🔧 技术方法**

采用 Playwright 爬取 UI、网络日志和 HTML，PaddleOCR OCR，LLM（GPT‑4o‑mini 代替）进行 PII 检测，手工验证。

**📊 数据集**

基于超过 3300 万条来自 25 家公共 SMS 网关的短信，提取 322K URL，最终评估 147K 活跃 URL。

**📈 对比分析**

与已有单独研究相比，本工作覆盖规模 10 倍以上，精确率与召回率经人工校验均高于 90%，并成功促成 18 份修复，保护约 1.2 亿用户。

**⚠️ 局限性**

局限在于仅检索公开网关短信，忽略加密或业务内部渠道；枚举攻击受限于 token 长度；未评估后端安全配置与服务端逻辑深层漏洞。

---

## 207. Understanding or Memorizing? A Case Study of German Definite Articles in Language Models

**arXiv ID:** 2601.09313 | [PDF](https://arxiv.org/pdf/2601.09313v1)

**作者:** Jonathan Drechsel `[一作]` (University of Passau), Steffen Herbold `[通讯]` (University of Passau)

**通讯引用:** 1828 | [OpenAlex ID](https://openalex.org/A5027032646)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对德国定冠词的性别和格转换进行梯度方向干预，研究语言模型是基于规则还是记忆来生成语法正确的冠词。

**💡 创新点**

创新点在于使用梯度方向解释器（gradiend）捕捉特定性别-格转换的更新方向，并比较不同转换间参数重叠，揭示模型在此语法现象上更倾向于记忆化而非纯规则化。

**🔧 技术方法**

所用技术包括Transformer 语言模型（encoder/decoder）、MLM 风格的冠词预测头、梯度方向解释器（gradiend）以及 Top‑k 权重重叠分析。

**📊 数据集**

数据集来源于德国维基百科句子按 spaCy 自动标注的性别‑格标签构成的 19k–61k 条样本集，以及基于 Wortschatz Leipzig 新闻语料的无冠词参考集。

**📈 对比分析**

比较方法采用概率变化（ΔP）、Cohen’s d 及置换检验，并在保持 99% LMS 的约束下选取最优学习率；实验结果显示不同模型规模均出现跨单元的概率提升，但并不完全符合严格的规则化模式，提示存在记忆化效应。

**⚠️ 局限性**

局限性包括仅关注单一的定冠词语法现象、干预仅为单向梯度方向、效果微小、模型规模受限、自动标注可能引入噪声、decoder 模型使用非原生 MLM 头、未评估对生成质量的实际影响。

---

## 208. An Information Theoretic Proof of the Radon-Nikodym Theorem

**arXiv ID:** 2601.09308 | [PDF](https://arxiv.org/pdf/2601.09308v1)

**作者:** Peter Harremoës `[一作]` `[通讯]` (Niels Brock Copenhagen Business College), Peter Harremoës (Niels Brock Copenhagen Business College)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了利用信息论方法对 Radon‑Nikodym 定理进行证明，并给出了基于信息投影的近似估计；

**💡 创新点**

创新点在于将信息散度与数值近似结合，提供了一个在信息论框架下的 Radon‑Nikodym 导数存在性证明，并将其推广到分布式格和概念格上的取值；

**🔧 技术方法**

采用了信息散度（f‑divergence）、信息投影、格与取值理论（valuation）以及概念格（concept lattice）等理论工具；

**📊 数据集**

无；该工作为理论证明，不涉及具体实验数据集；

**📈 对比分析**

未进行实验或数值性能比较；主要通过数学推导与不等式证明来验证结论；

**⚠️ 局限性**

局限性包括：仅在 σ‑有限或 s‑有限测度范围内适用；对一般非 σ‑有限情形的推广仍有待研究；在实际应用中需进一步检验近似误差和收敛速度。

---

## 209. MACRO-LLM: LLM-Empowered Multi-Agent Collaborative Reasoning under Spatiotemporal Partial Observability

**arXiv ID:** 2601.09295 | [PDF](https://arxiv.org/pdf/2601.09295v1)

**作者:** Handi Chen `[一作]` (Hong Kong University), Edith C. H. Ngai `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 MACRO-LLM 框架，使 LLM 代理在空间和时间局部可观测限制下通过协同推理实现高效多智能体协作

**💡 创新点**

首次将空间与时间局部可观测拆解为三个模块（CoProposer、Negotiator、Introspector），并结合均场统计聚合和语义梯度下降实现分布式协作

**🔧 技术方法**

使用 GPT‑4o 等大型 LLM、预测回滚、均场统计聚合、语义梯度下降、零样本推理等技术

**📊 数据集**

在车辆自适应巡航控制（CACC）和疫情防控（PC）两种模拟任务上进行实验

**📈 对比分析**

与 MARL（DPPO、DMPO、IC3Net）及 LLM‑MAS 基线（ToM‑Belief、ChatEval、LAMEN）对比，MACRO‑LLM 在 RMSE‑H、SD‑V/H、I_n、PI_n、PD 等指标上均显著优于对手，并保持良好可扩展性

**⚠️ 局限性**

计算与通信开销较大，推理延迟高；依赖大型基础模型 API，导致上下文注入和成本昂贵

---

## 210. Explainable Autoencoder-Based Anomaly Detection in IEC 61850 GOOSE Networks

**arXiv ID:** 2601.09287 | [PDF](https://arxiv.org/pdf/2601.09287v1)

**作者:** Dafne Lozano-Paredes `[一作]` (Universidad Rey Juan Carlos), José Luis Rojo-Álvarez `[通讯]` (Universidad Rey Juan Carlos)

**通讯引用:** 5202 | [OpenAlex ID](https://openalex.org/A5051286582)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种可解释的无监督多视图异常检测框架，对IEC 61850 GOOSE网络的语义完整性与时序可用性进行分离，并通过非对称自编码器学习正常流量的潜在表示，结合重构误差与极值理论阈值实现异常检测。

**💡 创新点**

创新点包括：① 将语义层与时序层拆分为两视图；② 采用非对称自编码器分别捕获序列依赖与时序动态；③ 通过极值理论动态阈值降低误报；④ 使用特征级重构误差实现直接可解释性。

**🔧 技术方法**

技术手段：自编码器（深度与瓶颈不同）、极值理论（EVT）阈值、UMAP可视化、基于窗口的特征工程（时间间隔、速率、序列号变化等）。

**📊 数据集**

数据集：1）真实电力站运维抓取的正常GOSS数据（训练）；2）IEC61850SecurityDataset（包含MS、DM、DoS三类攻击的10分钟PCAP，测试）。

**📈 对比分析**

性能对比：在不同窗口长度（0.1、0.5、1、3 s）下，语义视图AE始终达到1.00召回率，DoS与MS的F1均在0.89–0.99之间；时序视图F1略低但仍优于5%误报率。相较于传统规则/监督学习，误报显著下降，检测率高达99%以上。

**⚠️ 局限性**

局限性：① 时序视图对短暂或轻微时序变化易产生误报；② 需要挑选合适的窗口长度；③ 仅在已知攻击模式下验证，对完全未知或高度隐蔽的攻击效果未知；④ 依赖大量正常流量进行训练，若正常流量变化剧烈需重新训练。

---

## 211. MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus

**arXiv ID:** 2601.09270 | [PDF](https://arxiv.org/pdf/2601.09270v1)

**作者:** Yexing Du `[一作]` (Harbin Institute of Technology), Bin Qin `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建并公开了涵盖六个语音任务的大规模古典中文文学音频语料库 MCGA。

**💡 创新点**

创新点包括：①首次提供完全版权开放的大规模古典文学音频数据；②针对文学情感识别设计专属评估指标；③提出跨模态一致性 (CMC) 评估方法。

**🔧 技术方法**

技术手段包括人类录音、LLM（DeepSeek、GPT‑5、Gemini）生成 QA 对，Whisper 与 Qwen 进行语音识别验证，LoRA 微调等。

**📊 数据集**

使用的主要数据集为 MCGA，包含 22,000 条音频（119 小时），覆盖五大文学体裁（Fu、Shi、Wen、Ci、Qu）和 11 个历史时期。

**📈 对比分析**

通过对 10 种主流 MLLM（闭源与开源）在 ASR、S2TT、SEC、SQA、SU、SR 6 任务上的评估，发现 Qwen 系列模型表现最优，尤其在 ASR、SEC、SQA、SU 上；总体仍存在显著提升空间。

**⚠️ 局限性**

局限性在于缺乏与文本同步的图像样本，以及 Qu 体裁因历史因素样本量较少。

---

## 212. LatencyPrism: Online Non-intrusive Latency Sculpting for SLO-Guaranteed LLM Inference

**arXiv ID:** 2601.09258 | [PDF](https://arxiv.org/pdf/2601.09258v1)

**作者:** Du Yin `[一作]`, Danyang Zhang `[通讯]` (Alibaba Cloud Computing)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一套零侵入、跨平台的LLM推理实时延迟监测与异常检测系统，能够在不改动代码、不重启服务的前提下，对推理过程进行微观级别的追踪与分析。

**💡 创新点**

创新点在于将跨层次（CPU、框架、GPU）语义对齐与自适应基线建模相结合，采用事件驱动统一时间同步和分布式协同采集，实现毫秒级报警并可重构全程上下文；同时通过物理可解释特征与GBDT预测实现高精度异常检测。

**🔧 技术方法**

主要技术包括eBPF、ptrace、CUDA CUPTI、NVTX/ROCTX、分布式时间同步、事件驱动采集、GBDT基线预测、动态阈值控制与深度追踪触发。

**📊 数据集**

实验数据来源于NVIDIA A100 GPU上的SGLang v0.5.4推理Qwen3-32B模型，采集批量1–512、输入长度1–2048、输出长度1–512的多样工作负载；此外在真实生产环境中收集的持续流量用于模型训练与验证。

**📈 对比分析**

与传统固定阈值、固定窗口等方法对比，系统采用动态窗口+GBDT实现F1≈0.985、误报率0.59%、检测延迟0.2秒，CPU/延迟开销均低于0.5%，在生产部署中表现出低侵入性与高准确性。

**⚠️ 局限性**

主要局限包括：需在新工作负载或硬件升级后进行热身以构建正常基准；对非主流XPU的语义映射受限，导致深度追踪难以覆盖；以及在长期存在的性能问题缺乏有效基准时，可能无法触发报警。

---

## 213. PhyRPR: Training-Free Physics-Constrained Video Generation

**arXiv ID:** 2601.09255 | [PDF](https://arxiv.org/pdf/2601.09255v1)

**作者:** Yibo Zhao `[一作]` (State Key Laboratory of Computer Aided Design and Computer Graphics), Boxi Wu `[通讯]` (State Key Laboratory of Computer Aided Design and Computer Graphics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练自由的三阶段管线 PhyRPR，用于在视频生成过程中满足物理约束。

**💡 创新点**

创新点在于将物理推理与视觉渲染解耦：利用大语言多模态模型进行关键状态推理，使用确定性运动规划生成粗略运动框架，并在扩散采样阶段通过噪声一致的潜在注入保持运动一致性。

**🔧 技术方法**

采用的大模型技术包括 LMM（大语言多模态模型）进行物理推理、SAM 进行对象分割、稳定扩散/视频扩散模型进行视觉渲染，以及运动原语工具箱进行轨迹生成和潜在空间融合。

**📊 数据集**

实验使用 VBench、PhyGenBench、VideoScience-Bench 等物理一致性基准，并在 40 种包含文本或图像+文本提示的测试场景上进行评估。

**📈 对比分析**

与 WanX、LTX、SDEdit 等基线在 VBench 视觉质量、LMM-as-judge 物理一致性和用户研究等指标对比，PhyRPR 在物理可行性、轨迹遵循度和整体质量上均取得最高分。

**⚠️ 局限性**

局限性包括对 LMM 推理质量的依赖、在极其复杂或未见过的物理情景下可能出现误差、以及缺乏端到端学习的优势，导致对新型物理约束的泛化能力有限。

---

## 214. HGATSolver: A Heterogeneous Graph Attention Solver for Fluid-Structure Interaction

**arXiv ID:** 2601.09251 | [PDF](https://arxiv.org/pdf/2601.09251v1)

**作者:** Qin-Yi Zhang `[一作]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences), Zeng-Guang Hou `[通讯]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于异构图注意力网络的学习型流固耦合（FSI）求解器 HGATSolver，能够在单一框架内同时模拟流体与固体动力学。

**💡 创新点**

创新点包括：①异构图结构将流体、固体及其接口拆分为不同节点与边类型，嵌入物理先验；②物理条件门控机制（PCGM）自适应调节每个节点的状态更新；③跨域梯度平衡损失（IGBL）根据预测不确定性动态加权流固两域的损失。

**🔧 技术方法**

技术实现主要包括：图注意力机制、关系感知异构图卷积、物理参数嵌入、时间步编码、可学习的门控与梯度平衡策略。

**📊 数据集**

数据集涵盖两个自建 FSI 基准（FI‑Valve 与 SI‑Vessel）以及公开的 NS+EW 数据，三者分别代表心血管瓣膜、血管流固耦合与二维弹性带的流体-结构交互。

**📈 对比分析**

与 U‑Net、GCN、GAT、HGAT、GINO、GNOT、Transolver、AMG 等基线比较，HGATSolver 在 FI‑Valve 与 SI‑Vessel 上均实现最低的相对 L2 误差（流体约 2.6%–4.6%，固体约 0.25%–0.65%），并在 NS+EW 的少样本情形下表现出更强的样本效率（5、25、100 样本误差分别 0.24%–0.55%）。

**⚠️ 局限性**

局限性主要体现在：①仍需大量标注训练样本，难以完全覆盖极端耦合场景；②目前实验以二维或有限三维网格为主，未验证在更大尺度复杂几何上的可扩展性；③模型结构与超参数（如图类型划分、门控阈值）仍需手动设计，可能限制迁移性。

---

## 215. Reward Learning through Ranking Mean Squared Error

**arXiv ID:** 2601.09236 | [PDF](https://arxiv.org/pdf/2601.09236v1)

**作者:** Chaitanya Kharyal `[一作]` (University of Alberta), Matthew E. Taylor `[通讯]` (University of Alberta)

**通讯引用:** 7835 | [OpenAlex ID](https://openalex.org/A5070914351)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于评分的强化学习方法 R4，用于从人类多类评分中学习奖励函数。

**💡 创新点**

创新点在于将评分视为序数目标，使用可微分排序的排名均方误差（rMSE）损失，并给出其理论上最小且完整的解空间保证，避免了传统方法中需预设的评分边界。

**🔧 技术方法**

核心技术包括：可微分（soft）排序、SAC 强化学习、R4 训练流程（采样轨迹→预测回报→soft rank→rMSE 反向传播）以及动态反馈与分层采样策略。

**📊 数据集**

实验使用 OpenAI Gym 的机器人步态任务（如 Hopper、Walker2D 等）和 DeepMind Control Suite（如 Cartpole、Walker 等），通过模拟教师将轨迹划分为多类评分。

**📈 对比分析**

与 RbRL、PEBBLE、SURF、QPA 等基线比较，R4 在离线与在线两种反馈设置下均取得更快学习速度、更高最终回报（在多任务中至少 3/6 环境学习速度显著提升，4/6 环境最终回报更好）。

**⚠️ 局限性**

局限性包括：仍假设评分与真实奖励严格按序数对应，实验采用模拟教师评分，未验证对真实人类评分的鲁棒性；以及对评分类数的选择仍有一定影响。

---

## 216. The Real Menace of Cloning Attacks on SGX Applications

**arXiv ID:** 2601.09273 | [PDF](https://arxiv.org/pdf/2601.09273v1)

**作者:** Annika Wilde `[一作]` (Ruhr University Bochum), Ghassan Karame `[通讯]` (Ruhr University Bochum)

**通讯引用:** 6052 | [OpenAlex ID](https://openalex.org/A5059087800)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统评估了72个基于Intel SGX的应用，分析其是否易受克隆攻击，并归纳了三类主要攻击方式。

**💡 创新点**

创新点在于首次将克隆攻击与rollback攻击区分开来，对SGX应用进行大规模的实证研究，揭示了约20%应用仍易受克隆攻击。

**🔧 技术方法**

主要技术手段是源代码与文档的静态分析、攻击场景设计与分类，以及对封闭态态与状态恢复的技术评估。

**📊 数据集**

数据集为公开的SGX项目列表（约72个），包括机器学习、区块链、数据库等不同领域的实现。

**📈 对比分析**

对比方法采用“是否易受攻击”与“已使用防护措施”两列指标，发现64%数据库类应用易受攻击，说明克隆风险普遍存在；由于仅为分析研究，未给出性能数值。

**⚠️ 局限性**

局限性包括：仅进行理论与静态分析，未实际部署克隆攻击；依赖公开项目，可能遗漏未公开实现；缺乏对已部署防护机制（如TTP、计数器）的实测效果。

---

## 217. Regenerating codes with minimal disk I/O cost achieving optimal tradeoff between storage and repair bandwidth

**arXiv ID:** 2601.09300 | [PDF](https://arxiv.org/pdf/2601.09300v1)

**作者:** Minhan Gao `[一作]` (Chinese University of Hong Kong), Kenneth Shum `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了一种在所有调度点（包括 MSR、MBR 和内部点）上都能实现最优存储-修复带宽折衷且磁盘 I/O 成本最小的功能性重构再生码。

**💡 创新点**

创新点在于：
1) 通过将重构过程映射为严格的 gammoid（基于有向图的图论 matroid）来刻画辅助节点与新节点之间的线性依赖，从而实现无编码（repair‑by‑transfer）且满足最优带宽；
2) 给出了一个通用的包选择算法（choice function），保证在所有节点失效序列下都能维持 (n,k) 的恢复属性；
3) 证明所需有限域大小仅为 nαB-(n-1)αB，显著小于以往基于网络编码的实现。

**🔧 技术方法**

技术方法包括：
- 图论中的 gammoid 与 matroid 理论，特别是严格 gammoid 的线性可表示性；
- 信号流图（signal flow graph）对存储节点状态的递归动态系统建模；
- 线性代数与随机化矩阵论（如 Vandermonde 矩阵）用于构造满足 matroid 约束的全局编码向量；
- 组合优化算法用于决定每个阶段的 packet 选择（p_t(i)）。

**📊 数据集**

论文为理论性工作，未使用具体数据集，而是以符号文件 B、存储节点数 n、恢复度 k 等参数进行符号推导与证明。

**📈 对比分析**

性能比较：
- 与传统需要在辅助节点做编码的重构方案相比，本方法在磁盘 I/O 方面实现了完全无编码（每个辅助节点只读取并转发一个符号）；
- 与先前的功能性重构码相比，本方案在所有调度点均达到最优带宽-存储折衷，并且所需的有限域大小显著降低；
- 通过理论分析与构造证明，显示了在 d=n-1 时能够在任意单节点失效、任意失效序列下维持 (n,k) 的恢复属性。

**⚠️ 局限性**

局限性：
- 需要所有剩余节点参与修复（d=n-1），在实际大规模系统中可能导致高并发与网络负载；
- 证明与算法实现相对复杂，实际编码实现和实验验证尚未给出；
- 对于多节点同时失效或节点不可用时的鲁棒性未直接讨论；
- 虽然域大小已显著降低，但仍可能对极大参数集合产生挑战。

---

## 218. Cluster Workload Allocation: Semantic Soft Affinity Using Natural Language Processing

**arXiv ID:** 2601.09282 | [PDF](https://arxiv.org/pdf/2601.09282v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 219. Policy-Based Reinforcement Learning with Action Masking for Dynamic Job Shop Scheduling under Uncertainty: Handling Random Arrivals and Machine Failures

**arXiv ID:** 2601.09293 | [PDF](https://arxiv.org/pdf/2601.09293v1)

**作者:** Sofiene Lassoued `[一作]` (South Westphalia University of Applied Sciences), Andreas Schwung `[通讯]` (South Westphalia University of Applied Sciences)

**通讯引用:** 1142 | [OpenAlex ID](https://openalex.org/A5025397538)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于彩色定时 Petri 网与可掩蔽 PPO 的动态车间调度框架 PetriRL，可在随机作业到达与机器故障环境下实时决策。

**💡 创新点**

其创新点在于将 Petri 网的约束掩蔽直接嵌入 PPO 的 logits 层，并采用 Gamma 与 Weibull 分布模拟真实作业到达与设备失效，实现了可解释且对抗扰动的调度策略。

**🔧 技术方法**

采用了彩色定时 Petri 网建模、Maskable Proximal Policy Optimization (MPPO) 强化学习、动作掩蔽、以及 Gamma/Weibull 随机过程。

**📊 数据集**

使用了 Raj benchmark（小规模）和 Taillard benchmark（大规模）实例作为实验数据集。

**📈 对比分析**

与十二种传统调度启发式（FIFO、SPT 等）进行对比，实验结果表明在小规模实例上平均降低约 3% 的完工时间，在大规模实例上平均降低约 1.1%，显著优于启发式平均水平。

**⚠️ 局限性**

限制主要包括：仅与启发式规则对比，未与其他深度 RL 方法或启发式混合方法比较；对随机事件的分布设定固定，未探索不同分布或更复杂扰动；以及在极大规模实例上的训练时间仍较长。

---

## 220. Computational Complexity of Swish

**arXiv ID:** 2601.09289 | [PDF](https://arxiv.org/pdf/2601.09289v1)

**作者:** Takashi Horiyama `[一作]` (Hokkaido University), Yutaro Yamaguchi `[通讯]` (Osaka University)

**通讯引用:** 30410 | [OpenAlex ID](https://openalex.org/A5015432645)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对原始的两符号卡牌游戏 Swish，本文完成了其在不同几何变换约束下的计算复杂度分类：在既不允许翻转也不允许旋转时可多项式求解；当至少允许一种翻转或 180° 旋转时则 NP‑完整；并给出了对应的多项式算法与 NP‑难度证明。

**💡 创新点**

创新点在于解决了先前未解的两符号卡牌情形，并揭示了几何变换与符号数之间的“临界点”，从而给出 Swish 的完整复杂度图谱；同时引入了新的多项式匹配构造与基于偶数环因子的归约技术。

**🔧 技术方法**

技术手段包括：1）将无翻转无旋转情形转化为双部图的最大权完美匹配；2）利用 4×|V| 网格构造，将偶数环因子问题归约到 Swish 的无旋转版本；3）通过卡牌翻转与旋转的配对，完成对 Swish 与 Swish-without-rotation 的 NP‑完整性证明；4）在需要的情况下使用随机化算法处理精确权重匹配。

**📊 数据集**

本文不使用任何实验数据集，而是构造理论实例和图论实例（如偶数环因子、三部图等）来构造 Swish 问题的输入。

**📈 对比分析**

与先前已知的单符号卡牌多项式可解性以及三符号卡牌 NP‑完整性结果进行对比，证明了两符号卡牌在不同变换约束下的复杂度差异；多项式算法复杂度为 O(n³)（n 为网格点数），而 NP‑难度通过标准 NP‑完备性归约给出。

**⚠️ 局限性**

局限性：仅针对决策版本，未给出最大 swish 大小的有效近似或启发式算法；缺乏对实际卡牌集合（如 60 张商业卡牌）在不同约束下的实验验证；对卡牌数量大于两符号的情况仅得到一般性 NP‑完整性结论，未探讨特殊结构卡组的可解性。

---

## 221. Why not Collaborative Filtering in Dual View? Bridging Sparse and Dense Models

**arXiv ID:** 2601.09286 | [PDF](https://arxiv.org/pdf/2601.09286v1)

**作者:** Hanze Guo `[一作]` (Renmin University of China), Xiao Zhou `[通讯]` (Renmin University of China)

**通讯引用:** 23680 | [OpenAlex ID](https://openalex.org/A5011384237)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出SaD框架，将稀疏视图与稠密视图协同融合，提升协同过滤性能

**💡 创新点**

通过双向对齐机制实现稀疏与稠密模型互补，理论证明SNR提升并消除稀疏性瓶颈

**🔧 技术方法**

使用稠密的矩阵分解或GNN（如LightGCN、SGL、SimGCL）作为稠密端，稀疏端采用slim或ItemCF，并加入伪正样本与稠密嵌入的对齐

**📊 数据集**

在Yelp2018、Gowalla、Amazon‑Book、Movielens等公开数据集上进行实验

**📈 对比分析**

与多种稠密基线（LightGCN、SGL、SimGCL、SGL‑ED、SimpleX等）及隐式双视图模型（UltraGCN、GF‑CF、PGSP）对比，SaD在Recall@20/NDCG@20上分别提升约1–5％，在长尾子集表现尤为显著，且对不同稠密骨干兼容性强

**⚠️ 局限性**

对稠密端的稀疏增强依赖于伪正样本比例与top‑K参数，需在不同数据集调优；对稀疏端的稠密嵌入对齐可能引入额外噪声，且在极度稀疏或冷启动场景下效果有限

---

## 222. ReGraM: Region-First Knowledge Graph Reasoning for Medical Question Answering

**arXiv ID:** 2601.09280 | [PDF](https://arxiv.org/pdf/2601.09280v1)

**作者:** Chaerin Lee `[一作]` (Soongsil University), Daseon Choi `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ReGraM框架，先构造查询对齐的KG子图，再在该子图内进行多步推理，显著提升医学问答的事实准确性和一致性。

**💡 创新点**

核心创新是region‑first思路：在推理前限定查询相关的局部子图，并结合证据感知的动态推理模式与审查循环，避免全图检索噪声与语义漂移。

**🔧 技术方法**

利用LLM进行查询域分类、子问题分解、关系权重化与MMR排序；在子图内执行固定3步的结构化推理；通过LoRA微调的LLM审查器对生成的三元组进行验证与修订。

**📊 数据集**

使用PrimeKG作为知识图谱，并在七个医学QA基准（MedDDx‑Basic/Intermediate/Expert、MedQA、PubMedQA、MMLU‑Medical、AfrimedQA）上评测。

**📈 对比分析**

与基线KGARevion在相同KG与LLM条件下对比，ReGraM在MCQ上平均提升8.04%准确率、SAQ提升4.50%准确率，同时将幻觉率下降42.9%，推理时延下降约46%。

**⚠️ 局限性**

局限性包括依赖自动LLM评判、对PrimeKG关系语义表示的敏感、在低KG覆盖率场景下可能出现低检索效果、以及缺乏正式的可扩展性与复杂查询的拒绝机制。

---

## 223. GaussianFluent: Gaussian Simulation for Dynamic Scenes with Mixed Materials

**arXiv ID:** 2601.09265 | [PDF](https://arxiv.org/pdf/2601.09265v1)

**作者:** Bei Huang `[一作]` (Peking University), Siyuan Huang `[通讯]` (State Key Laboratory of General Artificial Intelligence, BIGAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种统一框架，实现3D高斯点云（3D Gaussian Splatting, 3dgs）中的动态场景模拟与渲染，包括内部纹理填充、脆性破裂和多材料交互。

**💡 创新点**

创新点包括：① 通过生成模型实现无训练数据的内部纹理合成；② 针对3dgs的改进连续损伤力学（CD-MPM）实现高速度、稳定的脆性破裂；③ 支持多材料混合参数和流体仿真，构建单一统一仿真引擎。

**🔧 技术方法**

技术手段主要有：3dgs渲染、Stable Diffusion XL+MVInpainter进行内部纹理填充、基于GPU的CD-MPM物理仿真、Blinn‑Phong光照、Spherical Harmonics（SH）优化、返回映射（return‑mapping）稳定化。

**📊 数据集**

使用了包含水果、食品、液体等多材质物体的多视角图像数据集（如西瓜、草莓、奶油、金枪鱼等），未引入专门训练集，仅依赖公开预训练生成模型。

**📈 对比分析**

与PhysGaussian、OmniPhysGS等基线对比，使用CLIP相似度和人工用户评估。结果显示：内部纹理填充CLIP从22.3提升至35.4，用户偏好从3.57%提升至71.43%；动态破裂仿真CLIP从12.2/13.1提升至22.7，用户偏好从3.84%/7.69%提升至88.46%。性能上可实现实时渲染，GPU并行化实现高帧率。

**⚠️ 局限性**

局限性包括：材质参数手动设置，缺乏自动化逆渲染或学习方法；对极大规模场景的可扩展性不足；目前仅支持CD‑MPM的弹性/脆性模型，对复杂流体或颗粒材料的精细建模仍待完善。

---

## 224. Efficient Paths and Dense Rewards: Probabilistic Flow Reasoning for Large Language Models

**arXiv ID:** 2601.09260 | [PDF](https://arxiv.org/pdf/2601.09260v1)

**作者:** Yan Liu `[一作]` (Meituan), Yangdong Deng `[通讯]` (Tsinghua University)

**通讯引用:** 2574 | [OpenAlex ID](https://openalex.org/A5059155953)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 CoT‑Flow 框架，将链式推理视为连续概率流，提出利用信息增益量化每一步的贡献，并通过流导向解码和基于流的强化学习实现高效推理和密集奖励。

**💡 创新点**

创新点在于：①将离散推理步骤映射为连续流，定义概率流进展（PFP）度量；②设计无训练的贪心流解码策略；③提出基于流的 verifier‑free 稠密奖励与软质量门控，实现无人工标注的强化学习。

**🔧 技术方法**

使用的技术包括：概率流理论、Rectified Flow/Optimal Transport、对数似然比作速度量、贪心解码、基于流的奖励函数、Soft Quality Gate、与 GRPO/VeriFree 对比的 RL 算法。

**📊 数据集**

在七个推理基准上评测：AIME24/25、AMC23、Math‑500、GPQA‑D、TheoremQA、WebInstruct，使用 Qwen3 系列 LLM 作为后端模型。

**📈 对比分析**

与标准 CoT、GRPO、VeriFree 等基线对比，CoT‑Flow 在大多数模型规模下提升 10–20% 的准确率，同时平均推理长度减少 15% 左右，形成更优的效率-性能 Pareto 前沿。

**⚠️ 局限性**

主要局限包括：速度估计依赖零样本提示的后验近似，可能受模型基础性能限制；RL 仅在 on‑policy 环境下测试，样本效率和离策略推广尚未验证。

---

## 225. Knowledge-Embedded and Hypernetwork-Guided Few-Shot Substation Meter Defect Image Generation Method

**arXiv ID:** 2601.09238 | [PDF](https://arxiv.org/pdf/2601.09238v1)

**作者:** Jackie Alex `[一作]` (St. Petersburg), Justin Petter `[通讯]` (St. Petersburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Stable Diffusion的少样本变压站电表缺陷图像生成框架，集成知识嵌入、几何裂纹建模与超网络控制，能够生成高质量、可控的缺陷图像。

**💡 创新点**

① 通过DreamBooth知识嵌入实现电表域适配；② 设计几何裂纹建模模块生成空间约束控制图；③ 采用轻量级超网络动态调节扩散过程，兼顾生成质量、样本多样性与精确控制。

**🔧 技术方法**

Stable Diffusion、DreamBooth、Segment Anything Model (SAM)、超网络 (hypernetwork)、几何裂纹建模、FID/IS 评估及 YOLOv8 下游检测。

**📊 数据集**

Substation Meter Dataset (SMD)，共629张高分辨率图像，其中100张含裂纹缺陷。

**📈 对比分析**

与 FastGAN、ProjectedGAN、DFMGAN、IDDPM、RDDM 等 SOTA 少样本方法对比，FID 最低 76.72、IS 最高 2.45；使用生成数据训练 YOLOv8 后，mAP50 提升 19.1%，精度提升 26.9%。

**⚠️ 局限性**

仅针对裂纹缺陷，缺少多缺陷类型支持；对复杂缺陷模式的控制有限；未考虑视频/时序信息，训练成本较高。

---

## 226. Enhancing Spatial Reasoning in Large Language Models for Metal-Organic Frameworks Structure Prediction

**arXiv ID:** 2601.09285 | [PDF](https://arxiv.org/pdf/2601.09285v1)

**作者:** Mianzhi Pan `[一作]` (Institute of AI Industry Research), Jianbing Zhang `[通讯]` (National Key Laboratory for Novel Software Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种基于大型语言模型（LLM）的自回归金属‑有机框架（MOF）结构预测框架，采用块级生成并结合空间感知持续预训练（CPT）、结构监督微调（SFT）和匹配驱动强化学习（RL）三阶段训练；

**💡 创新点**

首次将LLM应用于MOF结构预测；通过在预训练阶段加入拓扑代码、分子量、PCA span等空间信息提升3D几何推理；采用Euler角和SMILES文本编码实现块级位置信息生成；采用Soft Adaptive Policy Optimization（SAPO）强化学习提升生成结构的稳定性；

**🔧 技术方法**

使用Qwen‑3 8B LLM，结合空间感知持续预训练、结构监督微调、匹配驱动强化学习（SAPO）等技术；在文本化编码中使用Euler角、SMILES、格子参数等；

**📊 数据集**

使用从文献中获得的324,426个计算生成MOF结构（约3个构建块，去除>200块样本），按8:1:1划分训练/验证/测试；

**📈 对比分析**

与降噪式DiffCSP、MOFFLOW、MOF‑BFN（块级）以及全原子LLM PLaID++进行比较；指标为匹配率（MR）和RMSE；该方法在MR上约35.78%高于所有基线，并在推理速度上达到0.04 s/结构，明显快于迭代式降噪方法；

**⚠️ 局限性**

仅考虑刚性块，未处理构象柔性；将3D块转化为SMILES导致部分几何信息丢失；缺乏链式思考等解释性机制。

---

## 227. STaR: Sensitive Trajectory Regulation for Unlearning in Large Reasoning Models

**arXiv ID:** 2601.09281 | [PDF](https://arxiv.org/pdf/2601.09281v1)

**作者:** Jingjing Zhou `[一作]` (University of Chinese Academy of Sciences), Liang Li `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向大型推理模型的无监督式推理时遗忘框架 STaR，能够在生成链式思考过程中动态抑制敏感信息。

**💡 创新点**

创新点包括：①参数无关、推理时实现；②基于语义识别的敏感内容检测；③全局安全前缀与逐步轨迹抑制相结合；④令牌级自适应过滤与软硬约束；⑤提出了跨解码一致性分数 MCS 与多粒度成员推断评估 MIA，提供了更全面的安全评估。

**🔧 技术方法**

使用的技术主要有：语义嵌入与分类器检测、语义检索、句法与实体抽取、Prompt 前缀强化、轨迹级抑制学习、令牌级软/硬抑制、流畅度与敏感度评分、MCS 与 MIA 评估。

**📊 数据集**

在 R-TOFU 公开基准上进行实验，该基准包含 200 位作者的 20 组问答–思考–答案，标注有敏感内容。

**📈 对比分析**

与 GA、GD、KL、PO、DPO 等基线在忘记率 1%/5%/10% 下对比，STaR 在答案遗忘（AFE）、链式遗忘（CFE）、多解码一致性（MCS）和成员推断 AUC（MIA-A/C）上均取得显著优势，且在推理时实现、时延和计算成本方面优于基线。

**⚠️ 局限性**

局限性包括：①在极端语义重写或高度隐晦的敏感内容上仍可能存在泄漏；②对外部检测器与前缀工程依赖度高；③评估仅基于合成 R-TOFU 数据集，未检验在真实多模态或行业数据中的泛化能力；④某些解码策略下仍需细化参数调优以平衡精度与隐私。

---

## 228. M$^3$Searcher: Modular Multimodal Information Seeking Agency with Retrieval-Oriented Reasoning

**arXiv ID:** 2601.09278 | [PDF](https://arxiv.org/pdf/2601.09278v1)

**作者:** Xiaohan Yu `[一作]` (Huawei Cloud Business Unit), Chong Chen `[通讯]` (Huawei Cloud Business Unit)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可训练的多模态信息检索代理M^3Searcher，将检索与答案生成解耦，并通过强化学习优化检索策略。

**💡 创新点**

在多模态MRAG中引入模块化设计、检索导向多目标奖励和专门的MMSearchVQA数据集，实现更深层次、多步检索与推理。

**🔧 技术方法**

利用多目标强化学习（GRPO）、LLM‑as‑Judge评估、可训练的MLLM规划器与多模态工具（图像检索、文本检索、答案生成）等技术。

**📊 数据集**

在训练中使用MMSearchVQA（基于ReasonVQA扩展的多步检索数据集），测试时使用MMSearchVQA、MMSearch、InfoSeek和MRAG‑Bench等。

**📈 对比分析**

与四类基线（无代理、提示工程代理、端到端代理、未训练的解耦代理）对比，M^3Searcher在多项基准上获得最高分，且在搜索引擎与答案生成器迁移下保持稳健。

**⚠️ 局限性**

受限于工具集规模和多步检索深度，难以覆盖更广泛的真实工具和极深层次推理场景。

---

## 229. BrainSegNet: A Novel Framework for Whole-Brain MRI Parcellation Enhanced by Large Models

**arXiv ID:** 2601.09263 | [PDF](https://arxiv.org/pdf/2601.09263v1)

**作者:** Yucheng Li `[一作]`, Fan Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 52026 | [OpenAlex ID](https://openalex.org/A5100403400)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出BrainSegNet框架，实现全脑95区MRI分割。

**💡 创新点**

创新点是将SAM transformer与U‑Net skip连接结合，并加入多尺度注意力和边界细化模块。

**🔧 技术方法**

使用SAM transformer、U‑Net、ASPP、通道空间注意力、边界细化，结合交叉熵与Dice损失。

**📊 数据集**

数据集为100例HCP T1扫描，90例训练10例测试，分辨率0.7mm 256×256×260。

**📈 对比分析**

与FastSurfer、MASAM、SAM对比，Dice平均0.779，优于SOTA的0.743-0.741。

**⚠️ 局限性**

局限在仅使用单一公共数据集，模型对不同磁共振场强或病理影像的鲁棒性未验证。

---

## 230. Magnifying change: Rapid burn scar mapping with multi-resolution, multi-source satellite imagery

**arXiv ID:** 2601.09262 | [PDF](https://arxiv.org/pdf/2601.09262v1)

**作者:** Maria Sdraka `[一作]` (Orion Lab, National Observatory of Athens and National Technical University of Athens), Ioannis Papoutsis `[通讯]` (Orion Lab, National Observatory of Athens and National Technical University of Athens)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为BAM-MRCD的深度学习模型，实现利用MODIS和Sentinel‑2多源多分辨率卫星影像在火灾后24小时内快速绘制烧伤痕地图。

**💡 创新点**

创新点在于：①将高分辨率预火影像与低分辨率后火影像并行编码，避免传统超分模块的误差累积；②采用伪Siamese结构和多尺度注意力融合双解码器，支持深度监督；③提出多尺度IoU评价指标，更细致衡量不同规模火灾的分割效果。

**🔧 技术方法**

采用CNN‑UNet架构的BAM-CD改造版，结合双分支卷积编码器、注意力模块和深度监督损失；同时对MODIS和Sentinel‑2多波段进行特征提取与融合。

**📊 数据集**

使用FLOGA火灾数据集（2017‑2021年希腊326起火灾），包含MODIS 7波段（500 m）和Sentinel‑2 13波段（10/20/60 m），并将Sentinel‑2降采样至60 m以与MODIS匹配。

**📈 对比分析**

与传统CD模型（FC‑EF‑Diff/Conc、SNUNet‑CD、MLA‑Net、ChangeMamba）以及SR‑CD模型（MM‑Trans、DCILNet）以及BAM‑CD的单源版本进行对比；BAM‑MRCD在IoU、F1、召回率等指标上均显著优于对手，尤其在小中型火灾上提升显著，能够检出更多事件并保持边界精度。

**⚠️ 局限性**

局限性包括：对极小或细长烧痕的分割仍不理想；受云雾/烟雾影响时需等到下一幅云净MODIS影像；依赖MODIS光谱特征，若未来使用VIIRS需重新训练；对极端大尺度或多时段火灾的泛化尚未完全验证。

---

## 231. RIFT: Repurposing Negative Samples via Reward-Informed Fine-Tuning

**arXiv ID:** 2601.09253 | [PDF](https://arxiv.org/pdf/2601.09253v1)

**作者:** Zehua Liu `[一作]` (Huawei Noah's Ark Lab), Mingxuan Yuan `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种名为RIFT的奖励信息强化微调方法，利用模型自身生成的正负样本（不做硬阈值过滤）并在损失中按奖励加权，解决传统SFT和RFT在数据利用效率和训练稳定性方面的不足。

**💡 创新点**

创新点在于：① 将负样本的对数似然项替换为线性近似，避免梯度爆炸并保证训练稳定；② 通过奖励加权统一正负样本，既鼓励正确输出又抑制错误输出，无需额外对比或参考模型；③ 直接使用自生成的混合质量数据，显著提升数据效率。

**🔧 技术方法**

技术包括：奖励加权的最大似然目标（对正样本使用log概率，对负样本使用线性惩罚）；自回归多样本生成；基于奖励的分数归一化（常数负奖励、组内归一化等）；AdamW优化、余弦学习率调度。

**📊 数据集**

使用的主要数据集为数学推理基准：MATH、GSM8K、Minerva Math、Olympiad Bench、AIME 2024、AMC 2023、College Math，以及 NuminaMath；通过8条自生成答案构成训练缓冲。

**📈 对比分析**

实验对比了SFT、DFT、RFT、DPO等基线，评测指标为七个数学基准的 Mean@8 与 Pass@8，以及峰值显存。RIFT 在所有模型规模上均取得最高平均准确率（相较基线提升1–3%），且显存占用仅为 DPO 的约50%，在效率与性能上显著优于现有方法。

**⚠️ 局限性**

局限性包括：对奖励信号的辨别能力敏感，受限于可验证任务；主要在客观的数学推理任务上验证，尚未在开放式创作或主观任务中测试；当前不支持对多步推理中局部错误的细粒度奖励，未来可引入步级奖励提升学习效率。

---

## 232. Hybrid guided variational autoencoder for visual place recognition

**arXiv ID:** 2601.09248 | [PDF](https://arxiv.org/pdf/2601.09248v1)

**作者:** Ni Wang `[一作]` (Amazon Development Center Germany GmbH), Thorben Schoepe `[通讯]` (imec)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发并验证了一种混合引导式变分自编码器（Hybrid Guided VAE）用于事件摄像头的室内视觉地点识别，实现低功耗低延迟的机器人定位。

**💡 创新点**

1) 结合事件摄像头和神经形态硬件的SNN编码器；2) 引入引导式VAE实现特征解耦并兼具未知地点判别；3) 构建全新的Aachen‑indoor‑VPR事件数据集；4) 通过少量激活变量实现高效位置压缩。

**🔧 技术方法**

事件摄像头数据预处理、SNN编码器（BPTT训练）、ANN解码器、引导式β‑VAE、t‑SNE可视化、余弦相似度检索及硬件友好化实现。

**📊 数据集**

新采集的Aachen‑indoor‑VPR事件/ RGB数据集（约1,500–1,700样本/记录，四个地点、两种照明），并与公开基准如NetVLAD等进行对比。

**📈 对比分析**

通过图像检索评估定位误差与分类准确率，gVAE16在90%分类准确率、90%/80%/77%定位成功率（<0.5 m误差）上优于NetVLAD和SNN基线；在未知地点测试生成新聚类，显示良好泛化。

**⚠️ 局限性**

参数量仍高于纯SNN基线；需GPU训练后迁移至硬件；对更大规模或复杂场景的泛化尚未充分验证；光照变化已适应，但动态遮挡和大范围环境仍待研究。

---

## 233. Multi-Modal LLM based Image Captioning in ICT: Bridging the Gap Between General and Industry Domain

**arXiv ID:** 2601.09298 | [PDF](https://arxiv.org/pdf/2601.09298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 234. Integrating Diverse Assignment Strategies into DETRs

**arXiv ID:** 2601.09247 | [PDF](https://arxiv.org/pdf/2601.09247v1)

**作者:** Yiwei Zhang `[一作]` (State Key Laboratory of Multimodal Artificial Intelligence Systems), Zhipeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6088 | [OpenAlex ID](https://openalex.org/A5100410140)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在DETR风格检测器的解码器中插入轻量化LoRA分支，集成多种一对多标签分配策略，提升训练收敛速度与检测精度，而推理时无额外成本。

**💡 创新点**

证明了监督多样性而非监督数量决定性能提升；提出使用参数高效的LoRA分支，将多种一对多分配无缝融合进原有结构，保持模型简洁。

**🔧 技术方法**

低秩适配（LoRA）实现轻量化辅助分支；多种一对多分配（不同k值）；质量感知损失；Transformer解码器的自注意/交叉注意重排。

**📊 数据集**

在COCO 2017训练/验证集上进行实验。

**📈 对比分析**

与MS-DETR、H-DETR、Relation-DETR、Deformable-DETR等基线进行对比；在COCO上平均AP提升约2–3分（如Deformable-DETR 43.7→49.0），在相同训练周期下比传统一对多方法更快收敛，且推理时无额外开销。

**⚠️ 局限性**

需调节LoRA秩与分配超参以避免过拟合或信息不足；仅在训练阶段使用，推理时无直接收益；对极端数据分布或模型规模的泛化能力尚未充分验证。

---

## 235. When to Trust: A Causality-Aware Calibration Framework for Accurate Knowledge Graph Retrieval-Augmented Generation

**arXiv ID:** 2601.09241 | [PDF](https://arxiv.org/pdf/2601.09241v1)

**作者:** Jing Ren `[一作]` (RMIT University), Xiaodong Li `[通讯]` (RMIT University)

**通讯引用:** 24518 | [OpenAlex ID](https://openalex.org/A5100369719)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Ca2KG 框架，通过对抗性提示和面板式重评分提升 KG-RAG 的置信度校准与准确率。

**💡 创新点**

创新点在于把对抗性提示视为因果干预，并引入因果校准指数 (CCI) 对检索与推理的不确定性进行量化，显著改善模型的校准。

**🔧 技术方法**

采用了因果意识的对抗性提示、面板重评分机制、因果校准指数以及大型语言模型（LLaMA‑3 / GPT‑3.5）进行知识图检索与生成。

**📊 数据集**

使用了 MetaQA 与 WebQSP 两个多跳问答基准数据集。

**📈 对比分析**

与 KG‑RAG、IoE、Self‑Correct、RC‑RAG、Verb1S‑Top4、Verb2S‑Top4、Verb2S‑CoT 等基线进行对比，Ca2KG 在 ECE、Brier Score、AUC 等校准指标上显著下降，准确率保持或略有提升。

**⚠️ 局限性**

局限性包括对 LLM 规模与质量的依赖、在极端检索错误场景下的鲁棒性仍有限，且缺少大规模真实部署与在线评估。

---

## 236. On the Fair Allocation to Asymmetric Agents with Binary XOS Valuations

**arXiv ID:** 2601.09299 | [PDF](https://arxiv.org/pdf/2601.09299v1)

**作者:** Ziheng Chen `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Jialin Zhang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在异质权利下，二进制 XOS 代理人分配不可分物品的公平分配问题，证明了 1/2-APS 分配的存在并给出了多项式时间算法，同时研究了加权最小分配（WMMS）公平性，证明对任意 XOS 代理人可实现 1/n-WMMS，且该比值是最优的；在二进制加法性情况下还能实现精确的 WMMS 分配。

**💡 创新点**

创新点在于：
- 在二进制 XOS 环境下首次给出 1/2 的 APS 近似保证，并证明其最优；
- 提供了完整的多项式时间实现，包括 APS 值的二分搜索与无浪费子集的构造；
- 扩展了 WMMS 研究，证明对任意 XOS 代理人 1/n 的下界是最优的，并给出对应的轮询算法；
- 对二进制加法性情形给出精确 WMMS 的多项式时间方案，弥补了此前仅在对称或加法性情况下已知的结果。

**🔧 技术方法**

使用的技术主要有：
- 对 APS 的定义与二进制 XOS 的无浪费子集性质相结合，构造可保证价值的子集；
- 通过对 APS 估计值的递减搜索（类似二分）来实现多项式时间；
- 递归/归纳证明每个代理人都能得到至少 1/2 的 APS；
- 轮询（Round‑Robin）与加法性子函数选择相结合实现 1/n-WMMS；
- 对二进制加法性采用基于最小剩余价值的贪心分配构造 WMMS 分区与最终分配。

**📊 数据集**

本工作为理论性论文，没有使用任何实验数据集，所有结果均为解析证明。

**📈 对比分析**

与已有工作相比，作者的 APS 近似比率 1/2 与已知上界完全匹配，证明其最优；WMMS 方面的 1/n 近似率与加法性和一般 XOS 情形已知上界一致，表明在异质权利下该比率是最优的；而在二进制加法性情形下实现了精确 WMMS，超过了以往只能得到近似的结果。整体而言，论文在理论上完成了对 APS 与 WMMS 的最优逼近分析。

**⚠️ 局限性**

局限性：
- 对子加法（subadditive）或子模（submodular）且具有二进制边际值的代理人，是否存在常数逼近仍未解决；
- 对于异质权利下的二进制子模（matroid‑rank）代理人，是否能进一步提升近似比率仍是开放问题；
- 论文仅讨论物品分配，未涉及任务（chores）分配；
- 结果主要为理论证明，缺乏实验验证或对实际数据的评估。

---

## 237. RISER: Orchestrating Latent Reasoning Skills for Adaptive Activation Steering

**arXiv ID:** 2601.09269 | [PDF](https://arxiv.org/pdf/2601.09269v1)

**作者:** Wencheng Ye `[一作]` (Tongji University), Heng Tao Shen `[通讯]` (Tongji University)

**通讯引用:** 30228 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RISER框架，通过在冻结LLM内部激活空间中动态注入由Router组合的认知原语来提升推理能力。

**💡 创新点**

创新点在于将激活层的向量组合视为可复用的认知原语，并引入轻量Router通过强化学习自适应选择和加权这些原语，实现动态、可解释的推理控制。

**🔧 技术方法**

技术包括对激活向量的对比激活添加、LLM Judge过滤、K‑means聚类构建原语库；轻量Router采用多层感知机与双头输出（选择与强度），并使用Gumbel‑Sigmoid与Group Relative Policy Optimization进行训练。

**📊 数据集**

主要使用MMLU、MMLU‑Pro、GSM‑8K、MATH、GPQA、ARC‑C、TruthfulQA、Ethics等多种推理与道德类基准数据集进行向量抽取与Router训练。

**📈 对比分析**

与基线（零样本、CoT、Self‑Consistency、CAA、CAST、SAS、FR‑Ponder）对比，RISER在七个基准上平均提升3.4%–6.5%准确率，且在token效率上比CoT高2–3倍，证明了动态激活调度的有效性。

**⚠️ 局限性**

局限在于对原始模型预训练质量的依赖，原语库维度有限可能导致细粒度控制不足；跨模型迁移受限于激活空间相似度；过度激活可能带来不可预期的行为。

---

## 238. Coordinated Pandemic Control with Large Language Model Agents as Policymaking Assistants

**arXiv ID:** 2601.09264 | [PDF](https://arxiv.org/pdf/2601.09264v1)

**作者:** Ziyi Shi `[一作]` (Hong Kong University of Science and Technology), Hai Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 28560 | [OpenAlex ID](https://openalex.org/A5045705475)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一套基于大语言模型多智能体的协同疫情决策框架，用以在跨州流动限制中实现主动、协调的公共卫生干预。

**💡 创新点**

核心创新在于将LLM作为具备内部状态和通信能力的自主决策单元，并通过“时间流入再分配”（TIR）等可约束策略实现跨区域协同，突破传统规则或单一模型的局限；同时引入闭环仿真-决策循环，让智能体可预见干预后疫情演化。

**🔧 技术方法**

技术包括：LLM（如GPT-4等）作为推理模块；多智能体通信与协作机制；SEIQRD 传染病动力学仿真；基于代理的策略空间参数化；强化学习式闭环决策；Shapley值特征归因与Gini公平度量等解释与评估手段。

**📊 数据集**

数据集为美国COVID‑19的日常病例、死亡、康复及流动量（基于移动流向矩阵）以及州级公共卫生政策记录，时间范围为2020年4月到12月；实验扩展至20州。

**📈 对比分析**

与真实人类决策、专家经验决策以及随机决策进行对比；在5州实验中，LLM代理方案将累计感染和死亡分别降低25.7%/34.0%，在20州实验中累计感染与死亡分别降低约47–53%/58–62%；相较于传统方法，显著提升了效率和公平性，且对规划时长和策略类型的敏感性也得到系统性评估。

**⚠️ 局限性**

局限包括：依赖LLM的推理质量与prompt设计，且在面对不完整或错误的领域知识时可能失效；实现过程忽略政治、行政与遵从性等现实执行约束；目前仅聚焦流动限制为干预手段，未涵盖疫苗、检测、隔离等多元措施；跨域泛化需要进一步的域建模与多模态数据集成。

---

## 239. Learning to Trust Experience: A Monitor-Trust-Regulator Framework for Learning under Unobservable Feedback Reliability

**arXiv ID:** 2601.09261 | [PDF](https://arxiv.org/pdf/2601.09261v1)

**作者:** Zhipeng Zhang `[一作]` (China Mobile Research Institute), Lei Yang `[通讯]` (China Mobile Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了在反馈可靠性不可观测的情境下，学习系统如何通过内部监测来识别并调节经验可信度，从而实现可辨识的知识更新。

**💡 创新点**

创新点在于引入了“监测–信任–调节”（MTR）模块化框架，并用轻量级的自诊断机制实现对经验可信度的动态评估，首次证明在RL和SL两类任务中能区分性能恢复与内部知识恢复。

**🔧 技术方法**

主要技术包括：滑动窗口监测、策略漂移/熵方差等内部一致性指标、无监督聚类生成信任权重、软权重调节学习更新；在RL中使用PPO，在SL中使用两层MLP；评估使用预测熵等内在诊断信号。

**📊 数据集**

在RL中使用OpenAI Gym的HalfCheetah-v4、Pong等环境，注入随机或状态相关的奖励腐败；在SL中使用MNIST/类似手写数字分类数据，加入结构化标签偏差。

**📈 对比分析**

与标准PPO相比，PPO+自诊断在奖励腐败场景下平均回报提升约6%，方差下降约20%，最差性能提升约65%；在SL实验中，准确率可恢复至干净训练水平，但预测熵保持低值，表明内部知识未恢复。自诊断的诊断指标（AUROC≈0.81）能够区分早期误导阶段。

**⚠️ 局限性**

局限性包括：自诊断仅为诊断性，缺乏在SL中完整的调节机制；对高度自适应或策略性欺骗的鲁棒性未评估；监测指标为简单滑动窗口，可能不足以捕捉复杂动态；实验规模受限于小型环境和模型。

---

## 240. A Theoretical Framework for Rate-Distortion Limits in Learned Image Compression

**arXiv ID:** 2601.09254 | [PDF](https://arxiv.org/pdf/2601.09254v1)

**作者:** Changshuo Wang `[一作]` (Beijing University of Posts and Telecommunications), Ping Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 111363 | [OpenAlex ID](https://openalex.org/A5100405781)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了基于超先验架构的理论框架，估算学习式图像压缩的率失真极限。

**💡 创新点**

将率失真损失拆解为方差估计、量化策略和上下文建模三部分，提供可解释的上限估计。

**🔧 技术方法**

使用高斯方差最优化、Gaussian测试通道、逆水填算法和自回归上下文预测等信息论与深度学习技术。

**📊 数据集**

在OpenImages训练集、Kodak评估集和MNIST实验集上进行验证。

**📈 对比分析**

与现有样本驱动估计方法（Sandwich Bound、NERD、WGD）及实际压缩模型对比，得到更紧的下界，且改进后仍低于理论极限。

**⚠️ 局限性**

依赖超先验网络表达能力，且对高维真实图像的估计仍有偏差，未能完全逼近理论极限。

---

## 241. TeachPro: Multi-Label Qualitative Teaching Evaluation via Cross-View Graph Synergy and Semantic Anchored Evidence Encoding

**arXiv ID:** 2601.09246 | [PDF](https://arxiv.org/pdf/2601.09246v1)

**作者:** Xiangqian Wang `[一作]` (Harbin Engineering University), Ke Liu `[通讯]` (Shenzhen MSU-BIT University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了TeachPro模型，用于多维度学生教学评价，提出维度锚定证据编码、双视图图卷积网络等；

**💡 创新点**

1) 将教学维度作为可学习语义锚点并通过交叉注意力与证据对齐；2) 同时利用句法与语义图进行双视图图卷积学习；3) 采用低秩扰动分类头实现参数高效；4) 提供首个多维度证据标注的TeachScope基准数据集；

**🔧 技术方法**

BERT预训练编码、双向句法/语义图卷积网络（SynGCN、SemGCN）、BiAffine融合、低秩扰动LoRA分类头、交叉注意力机制、图卷积消息传递；

**📊 数据集**

TeachScope基准数据集，包含8,943条学生评论，按5个教学维度（专业能力、教学行为、教学效能、课堂体验、其他）进行3阶评分，并标注对应证据跨度；

**📈 对比分析**

与多种基线（LSTM、TextCNN、BERT、RoBERTa、SpanBERT、ELECTRA等）和不同模块消融结果进行对比；TeachPro在所有维度上均取得最高准确率（平均0.8049）和F1（平均0.7672），QWK最高，显示显著优于现有方法；

**⚠️ 局限性**

仍需人工标注证据跨度，跨语言/跨文化适用性未验证；模型结构相对复杂，训练成本较高；未充分处理学生评论的多义性、隐含情感和长期趋势等问题。

---

## 242. A$^2$TG: Adaptive Anisotropic Textured Gaussians for Efficient 3D Scene Representation

**arXiv ID:** 2601.09243 | [PDF](https://arxiv.org/pdf/2601.09243v1)

**作者:** Sheng-Chi Hsu `[一作]` (National Tsing Hua University), Hung-Kuo Chu `[通讯]` (National Tsing Hua University)

**通讯引用:** 1992 | [OpenAlex ID](https://openalex.org/A5056045120)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种自适应各向异性纹理的二维高斯剖面表示，能为每个高斯分散体分配不同分辨率和纵横比的纹理，从而在保持高渲染质量的同时显著降低显存占用。

**💡 创新点**

创新点在于：①引入梯度引导的自适应纹理控制策略，根据每个高斯的梯度大小和形状自动调整纹理分辨率和长宽比；②允许纹理为矩形而非固定正方形，进一步提升纹理利用效率。

**🔧 技术方法**

使用基于 2D 高斯剖面（2DGS）的渲染管线，结合光照表面谐波（SH）基色、RGBA 纹理、梯度驱动的高斯选择和纹理上采样，训练时采用 MCMC 稠密化与梯度上采样交替进行。

**📊 数据集**

在 Mip-NeRF 360、Tanks & Temples 和 Deep Blending 三个公开数据集上进行实验，涵盖 7、2、2 个场景。

**📈 对比分析**

与 2DGS、Textured Gaussians、BBSplat、SuperGaussians 等基线相比，在相同显存预算下可获得更高的 PSNR/SSIM/LPIPS；在固定高斯数量时虽然略逊于某些基线，但显存消耗大幅降低，整体表现优于传统纹理化方法。

**⚠️ 局限性**

局限性包括：目前仅支持纹理上采样而未实现下采样，无法充分利用低细节区域的显存；纹理压缩与高斯形状优化仍未纳入；未来需探索与更灵活几何体（如可变形径向核）的结合。

---

## 243. XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs

**arXiv ID:** 2601.09237 | [PDF](https://arxiv.org/pdf/2601.09237v1)

**作者:** Xinyang Chen `[一作]` (Huazhong Agricultural University), Zaiwen Feng `[通讯]` (Huazhong Agricultural University)

**通讯引用:** 342 | [OpenAlex ID](https://openalex.org/A5049033206)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种轻量化的时间序列预测模型XLinear，专门用于包含外部输入的长期预测任务；

**💡 创新点**

创新点在于引入可学习的全局token与双向门控模块（时间门控TGM与变量门控VGM），通过MLP与sigmoid实现跨时间与跨变量的高效信息筛选与融合；

**🔧 技术方法**

主要技术是基于多层感知机（MLP）的门控机制，结合全局token、时序嵌入与最终的全连接预测头；

**📊 数据集**

使用七个公开基准数据集（Electricity、Weather、ETT系列、Traffic等）以及五个实际应用数据集（Crop、DO_425012、DO_409215、GTD_N、GTD_S）进行评估；

**📈 对比分析**

与多种Transformer、CNN、GNN及其他轻量模型（TimeXer、PatchTST、TimeMixer、DLinear等）比较，XLinear在大多数配置下获得最低的MSE/MAE，并在训练速度上至少比主流Transformer快39%，内存占用更低；

**⚠️ 局限性**

局限性主要体现在高维多变量场景下（如Traffic数据）性能略逊于iTransformer，且VGM输入维度随变量数增大而显著提升，需进一步优化维度压缩与可扩展性。

---

## 244. GIFT: Unlocking Global Optimality in Post-Training via Finite-Temperature Gibbs Initialization

**arXiv ID:** 2601.09233 | [PDF](https://arxiv.org/pdf/2601.09233v1)

**作者:** Zhengyang Zhao `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14319 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种GIFT初始化方法，用于在大型推理模型的SFT+RL后训练过程中避免分布崩塌并提升RL性能。

**💡 创新点**

创新点在于将SFT与RL统一为全局后训练目标，推导出Gibbs分布为最优初始化，并通过有限温度的GIFT实现专家监督与基础先验的平衡。

**🔧 技术方法**

采用理论推导得到的Gibbs分布，构造基于soft‑target的KL最小化损失，以及RL阶段的GRPO/ PPO算法进行训练。

**📊 数据集**

主要使用DeepMath‑103k数据集进行训练，并在GSM8K、Math500、OlympiadBench、AIME24/25以及GPQA、MMLU-Pro/Redux、ARC‑Challenge等评测基准上进行评估。

**📈 对比分析**

与标准SFT、SFT+Entropy、DFT、ASFT、PSFT、LUFFY、ReLIFT等方法对比，GIFT在数学推理任务上平均pass@1提升至52.43%（Qwen2.5‑7B）或35.6%（Llama‑3.1‑8B），并在OOD基准上也取得最高平均分。

**⚠️ 局限性**

目前的局限是逆温度β被固定为超参数，缺乏针对不同样本或训练阶段自适应调整的机制。

---

## 245. Bias Dynamics in BabyLMs: Towards a Compute-Efficient Sandbox for Democratising Pre-Training Debiasing

**arXiv ID:** 2601.09421 | [PDF](https://arxiv.org/pdf/2601.09421v1)

**作者:** Filip Trhlik `[一作]` (University of Cambridge), Paula Buttery `[通讯]` (University of Cambridge)

**通讯引用:** 1127 | [OpenAlex ID](https://openalex.org/A5013999263)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用低成本 BabyLM 模型在预训练阶段进行偏见消除实验，验证其与标准 BERT 在偏见获取与消除动态上的一致性，并以此作为可复制的实验沙盒；同时通过多种消偏方法（CDA、毒性去除、扰动增广、INLP、Sent-Debias 等）在 BabyLM 上进行实验，降低实验成本。

**💡 创新点**

① 证明 BabyLM 能完整模拟 BERT 的偏见获取与消除行为；② 在 BabyLM 上完成预训练阶段消偏实验，显著降低 GPU 时数（从 500 小时降至 30 小时）；③ 首次直接将语料毒性去除与偏见下降建立因果关联。

**🔧 技术方法**

预训练、CDA（Counterfactual Data Augmentation）、毒性消除（LLM 重新书写）、扰动增广（perturbation augmentation）、INLP、Sent-Debias、CDS、Dropout、Debiasing Loss、Canonical Correlation Analysis 等；偏见评估采用 CrowS-Pairs、StereoSet；性能评估采用 BLiMP、BabyLM BLiMP supplement、EWoK。

**📊 数据集**

BabyLM 100M 语料（儿童指向式语音、儿童文本、Wikipedia 等）；BERT 约 3B 语料（Wikipedia + BookCorpusOpen）；CDA 用 10M 词的 Wikipedia；毒性/仇恨词标签来源于现有毒性检测模型；实验还使用了 3.39% 的毒性句子等。

**📈 对比分析**

通过构建综合性能指标（BLiMP、BabyLM BLiMP supplement、EWoK 的平均）与综合偏见指标（CrowS-Pairs、StereoSet 的平均），对比 BabyLM 与 BERT 在各类消偏方法下的性能-偏见变化。结果显示 BabyLM 的性能与偏见变化与 BERT 高度相关（相关系数 0.981），且在所有方法中偏见下降趋势与 BERT 一致，性能变化虽有差异但可接受，证明 BabyLM 可作为成本更低的替代。

**⚠️ 局限性**

① 仅使用简易的 intrinsic 评测，缺乏高级 extrinsic 评估；② 只覆盖英语，低资源语言评测受限；③ BabyLM 在某些任务（如复杂语义理解、CLM 任务）无法完全替代大模型；④ 结果需在大模型上进一步验证，才能确认通用性。

---

## 246. Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception

**arXiv ID:** 2601.09413 | [PDF](https://arxiv.org/pdf/2601.09413v1)

**作者:** Zhen Wan `[一作]` (NVIDIA), Boris Ginsburg `[通讯]` (NVIDIA)

**通讯引用:** 4413 | [OpenAlex ID](https://openalex.org/A5032957280)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Speech-Hands 这一可学习的语音代理框架，能够在自动语音识别（ASR）和多模态音频问答（AudioQA）任务中根据内部感知与外部建议的可信度做出自我反思决策，输出自我、外部或重写三种动作。

**💡 创新点**

核心创新在于将决策过程抽象为三种可学习的动作 token（<SELF>、<EXTERNAL>、<REWRITE>），实现对内部音频理解与外部模型输出的显式仲裁；并将这一仲裁机制统一应用于 ASR 与音频推理两大任务，提供了跨域的可解释代理式音频理解范式。

**🔧 技术方法**

技术包括：1) 在 Qwen2.5‑Omni 上进行监督微调，训练模型同时预测动作 token 与最终文本；2) 采用交叉熵联合训练，动作 token 作为生成序列首部；3) 采用多重采样与多数投票稳定外部模型标签；4) 通过自回归两阶段解码（先产生动作 token，再根据 token 生成最终答案或转写）。

**📊 数据集**

使用的数据集有：
- ASR：OpenASR 的七个基准（AMI、Tedlium、GigaSpeech、SPGISpeech、VoxPopuli、Libri‑clean、Libri‑other）。
- AudioQA：MD‑Audio 的三子集（Bio‑acoustic、Soundscape、Complex）。
- 外部模型数据：Whisper‑v2‑large、Canary‑1B‑v2、Parakeet‑TDT‑0.6B‑v3、Audio Flamingo 3 等。

**📈 对比分析**

与传统的仅文本或音频联合微调、提示式 GER（Generative Error Correction）以及外部 ASR + LLM 的 cascaded 方案对比，Speech‑Hands 在七个 ASR 基准上平均 WER 降低至 7.37%（相比基线 19.77%），在 AudioQA 上平均准确率提升至 77.37%（相比基线 57.87%），并在所有子任务中均取得领先；同时动作 token 的准确率（尤其是 <SELF> 与 <EXTERNAL>）显示出高 F1 评分，验证了仲裁机制的有效性。

**⚠️ 局限性**

主要局限包括：
- 动作 token 训练分布极不平衡，<REWRITE> 较少，导致该类别召回率低；
- ASR 训练样本受限（每个数据集最多 20k 条），可能尚未充分挖掘可用信号；
- 仅支持单一外部模型输入，未探讨多模型融合或跨模型迁移；
- 目前未对多任务或多外部源的泛化能力做系统评估。

---

## 247. Reversible Weighted Automata over Finite Rings and Monoids with Commuting Idempotents

**arXiv ID:** 2601.09409 | [PDF](https://arxiv.org/pdf/2601.09409v1)

**作者:** Peter Kostolányi `[一作]` (Comenius University), Andrej Ravinger `[通讯]` (Comenius University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究可逆加权自动机在有限（或局部有限）交换环上的性质，并给出了它们所识别语言的完整代数描述；在 𝔽₂ 上证明可逆加权自动机识别的语言等价于可逆有限自动机语言的布尔闭包，即对应伪族 𝐄𝐂𝐨𝐦；进一步证明在任意非平凡局部有限交换环上可逆加权自动机的可实现性判定是可判定的；同时提供从可逆加权自动机构造对应确定性自动机的构造，展示了其转移单元的幺半群结构。

**💡 创新点**

首次给出可逆加权自动机与交换环（尤其是 𝔽₂）之间的完整语言等价关系；引入可逆性与幺半群幺元可交换性的关联，构造了新的语言族 𝐅₂ 作为可逆语言的布尔闭包；证明可逆性判定在局部有限交换环上可有效决定，填补了该领域的算法空白。

**🔧 技术方法**

使用半环/环代数、正规级数理论、线性表示、语法单子与幺半群的代数结构、布尔闭包与伪族的 Eilenberg 对应、以及可逆性与转置自动机的结合等理论工具。

**📊 数据集**

无实测数据集，整个工作为纯理论研究。

**📈 对比分析**

通过构造可逆加权自动机的线性表示与其对应的确定性自动机，利用幺半群的幺元可交换性特征实现可逆性判定。该判定算法的复杂度未给出，但在理论上是可判定的。

**⚠️ 局限性**

仅适用于非平凡局部有限交换环，且对加权自动机的可逆性定义假设转移矩阵在每行每列至多有一个非零元；若环不满足局部有限或交换性质，结果不一定成立。

---

## 248. A Generalized Leakage Interpretation of Alpha-Mutual Information

**arXiv ID:** 2601.09406 | [PDF](https://arxiv.org/pdf/2601.09406v1)

**作者:** Akira Kamatsuka `[一作]` (Shonan Institute of Technology), Takahiro Yoshida `[通讯]` (Nihon University)

**通讯引用:** 2090 | [OpenAlex ID](https://openalex.org/A5051446545)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

将α-互信息（α-MI）统一解释为基于Kolmogorov–Nagumo均值与q-对数的泛化g泄漏量。

**💡 创新点**

提出了新的泛化条件易损度定义、q-对数形式的吉布斯不等式，并得出了涵盖Arimoto以外所有主要α-MI的统一g泄漏表达式；同时把α解释为攻击者风险厌恶度的量化指标。

**🔧 技术方法**

使用决策理论中的g泄漏框架、KN均值、q-对数、泛化易损度以及对Rényi熵、条件Rényi熵的解析对应关系。

**📊 数据集**

本文不涉及具体数据集，所有结果均为理论推导与数学证明。

**📈 对比分析**

由于缺乏实验评估，本文没有与其他方法的性能对比；仅在理论层面证明了各α-MI与g泄漏之间的一一对应关系。

**⚠️ 局限性**

主要局限在于：(1) 仅在离散有限字母空间下推导，未考虑连续或高维情形；(2) 结果为理论性质，未进行实验验证；(3) 对攻击者假设（如收益函数、决策规则）仍保持一定理想化。

---

## 249. ReflexDiffusion: Reflection-Enhanced Trajectory Planning for High-lateral-acceleration Scenarios in Autonomous Driving

**arXiv ID:** 2601.09377 | [PDF](https://arxiv.org/pdf/2601.09377v1)

**作者:** Xuemei Yao `[一作]` (National University of Defense Technology), Kewei Yang `[通讯]` (National University of Defense Technology)

**通讯引用:** 2127 | [OpenAlex ID](https://openalex.org/A5009696039)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在扩散模型推理阶段引入反射机制（Physics‑Aware Reflection），通过梯度调节显式强化车辆-道路物理耦合，以改善高侧向加速度场景下的轨迹规划。

**💡 创新点**

创新点：① 训练时采用条件丢弃增强物理耦合鲁棒性；② 推理时利用条件与无条件噪声预测差异计算梯度并投影到圆心加速度约束上，实现无手工引导函数的物理约束注入；③ 该方法架构无关，可直接加到现有扩散规划器上。

**🔧 技术方法**

技术手段包括扩散模型（Diffusion Planner）、无条件分类器指导（CFG）、梯度反射与投影矩阵、轨迹置信度多因素评估、GPU实时推理与多尺度参数调优。

**📊 数据集**

数据集：nuPlan Benchmark 的 Test14-hard（长尾高侧向加速度场景）和 Test14-random（常规场景），并与多种规则、学习及混合基线进行对比。

**📈 对比分析**

通过与规则、学习、混合方法及现有 Diffusion Planner 的对比，结果显示在 Test14-hard 上驾驶分数提升 14.1%，在常规场景上提升 20.7%。推理延迟仅从 3.3 ms 提升至 6.3 ms，平均 36.1 ms，仍满足 20 Hz 以上实时需求。

**⚠️ 局限性**

局限性：反射机制仅在极少数场景触发（≤0.5%），需手动调节阈值与尺度参数；方法假设车辆动力学为 aₙ = κv²，可能无法覆盖所有车型/道路特性；缺乏极端极限场景的理论保证与跨车型泛化评估。

---

## 250. Monte-Carlo Tree Search with Neural Network Guidance for Lane-Free Autonomous Driving

**arXiv ID:** 2601.09353 | [PDF](https://arxiv.org/pdf/2601.09353v1)

**作者:** Ioannis Peridis `[一作]` (Technical University of Crete), Markos Papageorgiou `[通讯]` (Technical University of Crete)

**通讯引用:** 20678 | [OpenAlex ID](https://openalex.org/A5071405020)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

构建了面向无车道交通的MDP框架，并结合蒙特卡罗树搜索与神经网络指导实现自动驾驶决策。

**💡 创新点**

首次在无车道环境中引入后向感知（nudging）提升安全性，并将预训练神经网络嵌入MCTS选择阶段以加速搜索。

**🔧 技术方法**

使用MCTS、PUCT式神经网络指导、离线自博弈数据集以及SUMO仿真环境。

**📊 数据集**

数据集来源于对无车道交通仿真场景的自博弈MCTS生成的状态-动作记录，车辆特征与邻车信息。

**📈 对比分析**

通过在不同交通流量下比较plain MCTS、nudging MCTS、NN‑MCTS和纯NN，NN‑MCTS在较少迭代数下即可达到零碰撞并保持较高平均速度，显示出更高的计算效率和安全性能。

**⚠️ 局限性**

主要限制包括神经网络前向推理的时间开销、对离线监督学习的依赖、离散动作空间以及Python/ C++ 混合实现导致的执行延迟，未来需实现原生 C++ 方案并探索在线学习与连续动作空间。

---

## 251. Spectral Complex Autoencoder Pruning: A Fidelity-Guided Criterion for Extreme Structured Channel Compression

**arXiv ID:** 2601.09352 | [PDF](https://arxiv.org/pdf/2601.09352v1)

**作者:** Wei Liu `[一作]` (Jiangsu University of Science and Technology), Yingtao Jiang `[通讯]` (University of Nevada)

**通讯引用:** 2705 | [OpenAlex ID](https://openalex.org/A5069124655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于频域低容量自编码器的Spectral Complex Autoencoder Pruning（SCAP），通过构造复数交互场并衡量其重构保真度来判定通道冗余，从而实现结构化通道剪枝；

**💡 创新点**

创新点在于：① 用复数交互场将输入多通道激活与单通道输出耦合；② 在频域中训练极低容量自编码器，对每个通道的谱重构保真度做重要性评分；③ 将重构保真度与滤波器ℓ1范数融合，形成稳健的阈值剪枝指标；④ 给出保真度与重构误差之间的理论联系；

**🔧 技术方法**

使用技术包括：二维快速傅里叶变换（FFT），低容量自编码器（共享MLP），复数交互场构造，重构保真度（余弦相似度）计算，滤波器ℓ1范数归一化融合，阈值剪枝，以及显存友好的逐通道处理与梯度累积；

**📊 数据集**

在CIFAR-10与CIFAR-100两个标准图像分类数据集上进行实验；

**📈 对比分析**

与多种现有通道剪枝方法（如ℓ1、SSS、GAL、CP、HRank、CSHE、PCC等）在VGG16、ResNet-56/110、DenseNet-40等网络上做对比；SCAP在阈值τ=0.5/0.6下可实现≈90% FLOP压缩、≈96%参数压缩，同时在CIFAR-10上Top‑1精度下降≤1.67%，在CIFAR-100上≤8.98%，在极端压缩下仍保持竞争优势；

**⚠️ 局限性**

局限性包括：需要手动设置阈值，无法全局保证剪枝后精度；仅针对CNN结构，其他网络如Transformer需改造交互场；剪枝后仍需微调；显存与训练时间上仍有一定开销；

---

## 252. Navigating Ethical AI Challenges in the Industrial Sector: Balancing Innovation and Responsibility

**arXiv ID:** 2601.09351 | [PDF](https://arxiv.org/pdf/2601.09351v1)

**作者:** Ruomu Tan `[一作]` (ABB AG), Martin W Hoffmann `[通讯]` (ABB AG)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文梳理工业领域AI伦理挑战，提出伦理框架和实践指南，强调信任、可解释性、可持续性与数据共享等要点。

**💡 创新点**

创新点在于将工业AI的伦理原则与实际需求结合，形成针对工业三种AI角色的伦理要求，并提供可操作的开发流程与数据治理建议；同时聚焦具身AI、生成式AI与可解释AI的伦理实践。

**🔧 技术方法**

使用的技术包括可解释AI（XAI）、具身AI（机器人、车辆）、生成式AI、TinyML、物理知识融合的机器学习、工业物联网与边缘/云计算平台。

**📊 数据集**

本文未提出专门的数据集，主要引用工业数据共享平台（如TUDataset、NASA Prognostics、EU Data Portal、Open Power System Data）作为示例。

**📈 对比分析**

由于本章为综述与实践建议，未进行方法性能比较；讨论中引用的案例主要以案例研究和行业经验为依据。

**⚠️ 局限性**

局限性包括缺乏实证实验验证、数据获取与共享受商业与法规限制、技术与伦理标准不统一、生成式AI在工业中的风险评估不足、能耗与生命周期成本仍待进一步量化。

---

## 253. See More, Store Less: Memory-Efficient Resolution for Video Moment Retrieval

**arXiv ID:** 2601.09350 | [PDF](https://arxiv.org/pdf/2601.09350v1)

**作者:** Mingyu Jeon `[一作]` (Chung-Ang University), Junyeong Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 736 | [OpenAlex ID](https://openalex.org/A5021487107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SMORE框架，用于在记忆受限的环境下进行视频时刻检索，既保持高信息分辨率，又显著降低显存占用。

**💡 创新点**

创新点包括：① 基于查询的定制化captioning，直接把用户意图注入视频描述；② 查询感知重要性调制，对帧-caption对进行语义加权；③ 结构化视觉压缩（SVD）在保持时序信息的前提下压缩帧，提升存储效率；④ 通过上述三项技术在MLLM上实现精确且高效的检索。

**🔧 技术方法**

采用的核心技术包括：多模态大语言模型（Flan‑T5 XL、InstructBLIP/BLIP‑2）、CLIP‑based相似度度量、QA‑based过滤与生成、SVD降维压缩、查询加权注意力机制。

**📊 数据集**

实验数据集：QVHighlights、Charades‑STA、ActivityNet‑Captions。

**📈 对比分析**

与Chrono、LLaVA‑MR、SG‑DETR等现有最先进模型在R@1、mAP、mIoU等指标上比较，SMORE在大部分指标上均实现了2%~4% 的提升，尤其在QVHighlights上实现了R1@0.5 +4.19%、R1@0.7 +6.24% 的增益。

**⚠️ 局限性**

局限性：相较于纯帧模型，SMORE在推理时会有轻微延迟；对高度模糊或含糊不清的查询与视频内容仍易出现误检，未来需要进一步优化模型鲁棒性与解码架构。

---

## 254. High-Performance Serverless Computing: A Systematic Literature Review on Serverless for HPC, AI, and Big Data

**arXiv ID:** 2601.09334 | [PDF](https://arxiv.org/pdf/2601.09334v1)

**作者:** Valerio Besozzi `[一作]` (University of Pisa), Marco Danelutto `[通讯]` (University of Pisa)

**通讯引用:** 3744 | [OpenAlex ID](https://openalex.org/A5051951613)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2018-2025年122篇关于高性能无服务器计算的研究进行系统文献综述，构建八大研究方向与九大应用领域的分类体系，揭示研究趋势与合作网络。

**💡 创新点**

首次在高性能无服务器计算领域开展系统综述，提出了针对算力密集型工作负载的完整分类体系、研究方向、应用场景以及学术合作图谱。

**🔧 技术方法**

采用Kitchenham方法的系统综述流程，利用ACM、IEEE、ScienceDirect三大数据库搜索、过滤与雪球检索，使用Python脚本整理数据，VOSviewer绘制作者合作网络。

**📊 数据集**

以122篇选定的研究论文作为数据集，对其引用量、出版期刊、作者信息等进行提取与分析。

**📈 对比分析**

通过文献计量与协作网络分析，对引用最多的论文、影响力最大的期刊与会议进行排序，并可视化作者与机构间的合作关系；结果表明高性能无服务器计算已成为热门研究方向。

**⚠️ 局限性**

综述主要聚焦概念性与原型研究，缺乏大规模实测与统一基准；所构建的分类体系与指标依赖于作者的主题归类，可能存在主观性。

---

## 255. Beyond the final layer: Attentive multilayer fusion for vision transformers

**arXiv ID:** 2601.09322 | [PDF](https://arxiv.org/pdf/2601.09322v1)

**作者:** Laure Ciernik `[一作]` (Technische Universität Berlin), Lukas Muttenthaler `[通讯]` (Aignostics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在保持Vision Transformer权重不变的前提下，提出一种可学习的注意力机制，动态融合所有中间层的CLS和平均池化表示，用于下游任务的线性探针。

**💡 创新点**

创新点是将注意力聚焦于所有中间层的摘要token和空间统计，自动学习任务相关层权重，而非仅使用最终层或固定拼接。

**🔧 技术方法**

采用多头交叉注意力（cross‑attention）对层级表示进行加权融合，并在顶部训练一个线性分类器。

**📊 数据集**

在20个VTAB与clip‑benchmark任务以及三大ViT家族（CLIP、DINOv2、Supervised ViT）的小/基/大模型上进行评估。

**📈 对比分析**

与传统线性探针和简单拼接方法相比，所提方法平均提升5.5个百分点，在大多数数据集上获得最高排名，且在不同模型规模下表现稳定。

**⚠️ 局限性**

局限性包括：对需要细粒度空间信息的任务可能不如patch‑级注意力，额外的注意力计算和参数导致内存/算力开销，以及在某些数据集上仍可能出现过拟合。

---

## 256. Variable Basis Mapping for Real-Time Volumetric Visualization

**arXiv ID:** 2601.09417 | [PDF](https://arxiv.org/pdf/2601.09417v1)

**作者:** Qibiao Li `[一作]` (University of Science and Technology of China), Ligang Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8494 | [OpenAlex ID](https://openalex.org/A5100635702)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Variable Basis Mapping (VBM) 框架，将体积数据的多尺度小波分析直接映射为 3D 高斯分布，实现实时可视化。

**💡 创新点**

创新点：① 通过预计算 Wavelet-to-Gaussian Transition Bank，首次实现波形小波核到高斯原语的解析对应；② 用解析规则直接生成 3DGS 初始参数，避免传统启发式或全图像优化；③ 引入轻量化图像空间微调提升视觉质量；④ 在多尺度小波域保持平移一致性，显著加速收敛。

**🔧 技术方法**

使用技术：3D 离散小波变换（Biorthogonal 4.4）、统计估计、线性/平移一致性推导、解析 Gaussian 构造、图像空间细调、PyTorch + CUDA、PyWavelets。

**📊 数据集**

数据集：Supernova（432^3 物理模拟）、Colon Prone CT（512×512×463）、Skull（256^3）、Foot（256^3）等科学与医学体积。

**📈 对比分析**

与 TensoRF、InstantNGP、3DGS、iVR-GS 对比：PSNR/SSIM 均最高，迭代次数显著更少，训练时间约 5–6 分钟，实时帧率约 120 FPS（RTX 3090）。

**⚠️ 局限性**

局限：对小波基的平移一致性要求高；需要多份转移函数手工调参；转换过程仍存在能量投影误差；对大规模或非均匀采样数据的内存占用和通用性待进一步研究。

---

## 257. Radiomics-Integrated Deep Learning with Hierarchical Loss for Osteosarcoma Histology Classification

**arXiv ID:** 2601.09416 | [PDF](https://arxiv.org/pdf/2601.09416v1)

**作者:** Yaxi Chen `[一作]` (University College London), Yipeng Hu `[通讯]` (University College London)

**通讯引用:** 5059 | [OpenAlex ID](https://openalex.org/A5032309114)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在骨肉瘤 H&E 病理图像中，提出一种融合深度图像特征与手工 radiomic 描述、并采用分层多任务学习与不确定性加权的多模态分类框架，用于区分非肿瘤、非可生存肿瘤与可生存肿瘤。

**💡 创新点**

创新点在于：①将 radiomic 特征作为辅助输入实现多模态融合；②用分层（粗细两级）任务并联合不确定性加权的损失来反映临床分层逻辑；③通过学习的注意力门实现不同模态的自适应权重。

**🔧 技术方法**

技术包括：卷积/变压器骨干网络（InceptionV3、EfficientNet、ViT）、两层 MLP 融合门、两头线性分类器、加权交叉熵、Kendall 型不确定性加权多任务损失、PyTorch 训练框架。

**📊 数据集**

使用公开的 TCIA Osteosarcoma Tumor Assessment 数据集（1,144 张 10× H&E 图块），在患者层面划分训练/验证/测试集，提取 29 维第一阶与 2D 形状 radiomic 特征。

**📈 对比分析**

与传统单任务 3 类分类（flat）和不同骨干网络进行对比；在患者级评估下，InceptionV3+分层损失+radiomic 的组合实现了 0.83 准确率、0.94 AUC、宏 F1 0.86 的最高性能，显著优于仅使用图像或仅使用 flat 损失的基线。

**⚠️ 局限性**

局限性包括：数据集规模有限、可能存在病理学标注不一致；多模态融合与不确定性加权的超参数调优复杂；模型对不同来源的图像迁移能力尚未充分验证。

---

## 258. Structured Knowledge Representation through Contextual Pages for Retrieval-Augmented Generation

**arXiv ID:** 2601.09402 | [PDF](https://arxiv.org/pdf/2601.09402v1)

**作者:** Xinze Li `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 36343 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于页面的自主知识表示框架 PAGER，先让 LLM 用内部推理构建一个包含若干空槽的认知大纲，随后通过迭代检索与填充，逐步把检索得到的证据填入槽中，最终生成一份结构化页面供 LLM 进行问答。

**💡 创新点**

创新点在于：① 用“页面”结构把知识拆分成有语义的槽；② 让 LLM 自主生成子查询并根据前一步页面状态迭代检索，显著提升检索的针对性和多样性；③ 通过连续填充形成的信息密度高、逻辑连贯的页面，既降低了知识冲突，又增强了外部知识利用。

**🔧 技术方法**

技术要点包括：LLM 交互式提示、检索增强生成（RAG）、结构化页面构建与迭代填充、基于 Wikipedia 的检索堆（FAISS+Qwen3-Embedding-0.6B）、Qwen3‑32B 与 Llama‑3.1‑70B‑Instruct 作为后端模型、vLLM 推理加速。

**📊 数据集**

使用了多跳与单跳 QA 任务的数据集：HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle（多跳）以及 NQ、AmbigQA（单跳），每个数据集抽取 2000 条开发集样本（Bamboogle 用全量 125 条）。

**📈 对比分析**

通过与 Vanilla LLM、一遍检索 RAG（Vanilla RAG、StructRAG）、迭代检索 RAG（Iter‑RetGen、IRCoT）以及迭代知识表示 RAG（RAT、Search‑o1、DeepNote）等多种基线对比，PAGER 在所有任务与模型上均超过基线 2% 以上，总体提升超过 5%（相较 StructRAG）和 9%（相较 Iter‑RetGen/IRCoT），在 ablation 研究中证明了初始化大纲、迭代检索与逐槽填充的必要性。

**⚠️ 局限性**

主要局限在于迭代填槽过程导致额外的推理时延；即使尝试并行填槽，效果仍低于迭代方案，导致在实际 QA 场景中需要在效果与效率之间做权衡。

---

## 259. Frame of Reference: Addressing the Challenges of Common Ground Representation in Situational Dialogs

**arXiv ID:** 2601.09365 | [PDF](https://arxiv.org/pdf/2601.09365v1)

**作者:** Biswesh Mohapatra `[一作]` (Inria), Justine Cassell `[通讯]` (Inria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了对话系统如何利用关系性引用建立持久公共基础，并提出基准评估方法。

**💡 创新点**

创新点在于将关系性引用作为衡量公共基础的指标，构建新评测基准，并通过合成数据与强化学习提升多跳推理能力。

**🔧 技术方法**

使用大型语言模型、摘要、分块、知识图谱三种公共基础表示，以及对话式强化学习（GRPO）和人工生成合成数据。

**📊 数据集**

采用 Meetup 与 Spot‑the‑Difference 两个语音对话语料库，并人工构造 400 条 Q/A 评测对。

**📈 对比分析**

通过完整上下文与受限上下文两种设置对比；在完整上下文下 Llama3.1 8B 约 45% 准确率，GRPO 训练后提升 15–20%；受限上下文时，知识图谱表示略优，整体性能低于 50%。

**⚠️ 局限性**

受限于数据稀缺、模型规模、以及对长对话和信念推理的不足，评估数据不具备完全自然多会话特征，模型仍易产生幻觉并难以区分说话者视角。

---

## 260. Asymptotic Rate Bounds and Constructions for the Inclusive Variant of Disjunct Matrices

**arXiv ID:** 2601.09362 | [PDF](https://arxiv.org/pdf/2601.09362v1)

**作者:** Yuto Mizunuma `[一作]` (Chiba University), Yuichiro Fujiwara `[通讯]` (Chiba University)

**通讯引用:** 529 | [OpenAlex ID](https://openalex.org/A5065086496)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过概率方法证明了包容式可分离矩阵在大规模群检验模型下可以实现正的渐近速率，并给出了相应的随机与确定性构造。

**💡 创新点**

创新点在于首次给出了包容式可分离矩阵的非平凡渐近下界，并且该下界与已知上界仅相差对数因子，实现了理论与构造的统一。

**🔧 技术方法**

主要技术包括概率方法、Chernoff 约束、置换法、Lovász 本地引理以及条件期望法的惰性估计。

**📊 数据集**

本研究为理论工作，无实验数据集；所有结果均基于组合与概率分析。

**📈 对比分析**

与先前已知的最强上界进行比较，本文构造的速率仅比上界多一个对数因子，证明了理论可行性。

**⚠️ 局限性**

局限性在于构造的时间复杂度为多项式（对 t、n 的阶数取决于 d、h、r），并未实现 r≥2 时的线性时间构造，且实际矩阵尺寸仍可能较大。

---

## 261. A Constructive Method to Minimize the Index of Coincidence under Marginal Constraints

**arXiv ID:** 2601.09347 | [PDF](https://arxiv.org/pdf/2601.09347v1)

**作者:** Pierre Jean-Claude Robert Bertrand `[一作]` `[通讯]` (Aix Marseille University, CNRS, AMSE), Pierre Jean-Claude Robert Bertrand (Aix Marseille University, CNRS, AMSE)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并给出在给定边际约束下最小化两随机变量联合分布的相同概率（即“相似度”）的最佳耦合，提出一种迭代构造方法得到闭式解。

**💡 创新点**

创新点在于发现最佳耦合具有阶梯形零块结构，利用该结构推导出通用变换并设计了收敛至最优的有限步迭代算法，首次在无附加边际条件下得到完整解析解。

**🔧 技术方法**

主要技术包括对偶KKT条件分析、对耦合矩阵的单调性与零块结构证明、构造递推变换与迭代算法，以及组合几何与概率测度的计算。

**📊 数据集**

无实验数据集，整个工作为理论证明与数学推导，未涉及数值实验或特定数据集。

**📈 对比分析**

与已知的在满足强可行性条件时的闭式解进行对比，证明新算法在所有边际下收敛且迭代步数不超过行数减一，效率与精度均满足最优性要求。

**⚠️ 局限性**

局限性在于仅处理离散有限维边际；对连续分布或多重边际的推广仍需进一步研究；此外，算法实现需保持矩阵单调性与非负性检查，计算复杂度随维度呈线性增长。

---

## 262. Generalized Schalkwijk-Kailath Coding for Autoregressive Gaussian Channels

**arXiv ID:** 2601.09329 | [PDF](https://arxiv.org/pdf/2601.09329v1)

**作者:** Jun Su `[一作]` (University of Hong Kong), Shlomo Shamai `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出并分析了适用于AR(p)高斯通道的SK(2)随机编码方案，给出了其可实现率的封闭式下界，证明该方案在某些AR(2)通道上优于传统的SK编码，并因此否定了Butman关于SK(1)编码可达容量的猜想。

**💡 创新点**

创新点在于将SK编码推广到第二阶确定递推的消息过程（SK(2)），实现对更一般AR噪声的反馈容量下界解析求解，并揭示了SK(1)在AR(2)通道上并非最优，从而拓展了反馈编码理论的适用范围。

**🔧 技术方法**

主要技术包括线性递推消息过程的设计、矩阵逆分解与奇异值分析、MMSE估计与信息率的解析计算，以及对特征根约束下的功率约束与信息率优化问题的解析求解。

**📊 数据集**

该研究为纯理论分析，不使用实测数据集，而是通过对AR(1)、AR(2)噪声谱的数学建模与符号计算来验证结论，并在论文中给出数值示例（P=1,5等）以展示不同β值下的可实现率比较。

**📈 对比分析**

通过与已知的AR(1)和AWGN通道的反馈容量以及SK(1)编码可实现率进行对比，实验结果表明：在AR(1)通道中SK(2)等价于SK(1)可达容量；在AR(2)通道中，SK(2)可实现率严格高于SK(1)，说明其性能更优。

**⚠️ 局限性**

局限性包括：只给出了下界并未证明SK(2)在所有AR(p)通道上确为最优；对高阶AR(p)（p>2）的通道分析仍未完成；且实现过程中对特征根的约束与数值求解可能在实际系统中具有实现复杂度挑战。

---

## 263. Feedback-Based Mobile Robot Navigation in 3-D Environments Using Artificial Potential Functions Technical Report

**arXiv ID:** 2601.09318 | [PDF](https://arxiv.org/pdf/2601.09318v1)

**作者:** Ro'i Lang `[一作]` (Technion), Elon Rimon `[通讯]` (Technion)

**通讯引用:** 6040 | [OpenAlex ID](https://openalex.org/A5049467216)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本文构建并分析了在三维工作空间中以多球体与圆柱体障碍为基础的多项式导航函数，用于实现安全的路径规划。

**💡 创新点**

创新点包括：①提出统一的多项式编码方式来表示球体和不同形状的圆柱障碍；②通过梯度与Hessian分析证明了在满足一定条件下导航函数唯一最小点位于目标且无局部最小；③利用平滑组合与p‑Rvachev函数处理障碍的相交情况，并给出了球形机器人到点机器人空间的转换方法。

**🔧 技术方法**

所用技术主要包括：多项式隐式函数表示障碍、梯度与Hessian符号分析、平滑组合与p‑Rvachev函数、球形机器人与点机器人映射、数值仿真与性能评估。

**📊 数据集**

数据集为合成的三维环境：10个半径为5 m的球形房间，每个房间随机放置10个球体/圆柱体障碍（包含相交与非相交情况），以及一个由36个基本障碍构成的“桁架”复合障碍；此外还在立方体房间中进行验证。

**📈 对比分析**

通过将导航函数分为 φ 与 ψ 两种形式，实验显示 ψ 在更小的调节参数 k 下即可避免局部极小点，成功率接近100%；对比合并障碍（使用R‑Rvachev函数）与不合并，合并后所需的 k 明显降低；在轨迹评估中，ψ 的导航时间短于 φ，且速度、加速度均保持在可接受范围内。

**⚠️ 局限性**

局限性包括：①需要较大的调节参数 k 才能保证无局部极小点，导致对极限 k 的计算复杂；②仅适用于球体与圆柱体障碍，无法直接扩展到任意形状；③不允许三重相交的障碍，限制了某些复杂几何配置；④合并障碍会增加计算量，且在大规模障碍环境下仍需进一步优化实现。

---

## 264. Video-MSR: Benchmarking Multi-hop Spatial Reasoning Capabilities of MLLMs

**arXiv ID:** 2601.09430 | [PDF](https://arxiv.org/pdf/2601.09430v1)

**作者:** Rui Zhu `[一作]` (Nanjing University), Jizhou Huang `[通讯]` (Baidu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Video‑MSR 基准，评估多跳空间推理（MSR）能力，并提供 MSR‑9K 训练集；

**💡 创新点**

创新点在于：①设计四类复杂多跳空间推理任务（CL、CRR、RP、CPD）；②构建视频基准并采用视觉‑文本双阶段生成与人工审核；③针对性数据指令调优显著提升 MLLM MSR 性能；

**🔧 技术方法**

采用多模态 LLM（如 Gemini‑3.0‑Pro、Qwen‑VL 系列）自动生成问题答案，双阶段人工验证；后续用指令调优（MSR‑9K）细化模型；评估使用 LLM‑as‑Judge 与多选直接匹配；

**📊 数据集**

采集自 ADT、ARKitScenes、ProcTHOR、S3DIS、ScanNet、ScanNet++ 共 3,052 条视频，生成 4,993 条高质量 QA 对；另外构建 8,369 条 MSR‑9K 训练样本；

**📈 对比分析**

在 20 款 MLLM（包括 GPT‑4o、Gemini‑2.5‑Flash、Qwen3‑VL、InternVL‑3 等）上评测，发现 7–70B 模型在 MSR 任务上的整体准确率低于 50%；在 MSR‑9K 调优后，Qwen2.5‑VL‑7B 提升 7.8%（尤其在 RP 上 48.6%），Qwen3‑VL‑8B 提升 3.1%；

**⚠️ 局限性**

局限：模型仍易出现空间失准、幻觉和推理漂移；评测数据主要来自有限室内场景，缺乏跨域、长视频或更复杂物理交互验证；

---

## 265. Preliminary Tests of the Anticipatory Classifier System with Hindsight Experience Replay

**arXiv ID:** 2601.09400 | [PDF](https://arxiv.org/pdf/2601.09400v1)

**作者:** Olgierd Unold `[一作]` (Wroclaw University of Science and Technology), Stanisław Franczyk `[通讯]` (Wroclaw University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

结合 Hindsight Experience Replay (HER) 与 Anticipatory Classifier System (ACS2)，提出 ACS2HER 并在两种稀疏奖励的网格世界中进行评估。

**💡 创新点**

首次把 HER 机制引入 ACS 系列，利用失败轨迹生成虚拟目标来稠密学习信号；同时系统性分析了 HER 参数 k 与经验回放强度 m 对知识掌握与复杂度的影响。

**🔧 技术方法**

核心技术为 ACS2 的预测-行动-效果（C‑A‑E）规则学习、经验回放（ER）缓冲、HER 目标重标记、ε‑greedy 策略、遗传算法（GA）以及 Python 基于 pyALCS 的实现。

**📊 数据集**

使用 OpenAI Gym 的 deterministic Woods9（9×9 迷宫）和 stochastic FrozenLake‑4×4 两个离散网格环境作为实验数据集。

**📈 对比分析**

与标准 ACS2 与 ACS2ER 进行对比，指标包括知识掌握率、可靠分类器数、到达目标的步数、成功率及运行时间；实验显示 ACS2HER 在知识掌握速度上优于基线，但因规则膨胀和多次学习迭代导致计算时间显著增加，且在随机环境下成功率低于 ACS2。

**⚠️ 局限性**

局限性包括：只在小规模离散网格上测试，缺乏对高维/连续空间的验证；HER 参数固定，未实现自适应策略；在随机环境中易产生错误虚拟目标；回放缓冲和规则库膨胀造成显著的内存与时间消耗；未进行充分的统计显著性检验和更丰富的 HER 策略比较。

---

## 266. Ability Transfer and Recovery via Modularized Parameters Localization

**arXiv ID:** 2601.09398 | [PDF](https://arxiv.org/pdf/2601.09398v1)

**作者:** Songyao Jin `[一作]` (University of California San Diego), Biwei Huang `[通讯]` (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型内部能力的分布，并提出了一种基于激活差异定位通道的能力迁移与恢复方法（ACT），能够在不显著干扰已保留能力的前提下恢复被遗忘的多语言推理能力；

**💡 创新点**

创新点在于通过跨模型激活差异挖掘出极少数（<5%）的通道来定位与特定能力相关的参数，仅迁移这些通道的参数并配合轻量化微调，从而显著降低参数迁移的干扰与计算成本；

**🔧 技术方法**

使用的技术包括激活差异分析、通道级掩码选择、任务向量（Task Arithmetic）式的参数差分合并以及后续的轻量化监督微调；

**📊 数据集**

实验所用数据集涵盖多语言数学推理（MetaMathQA翻译版）、科学推理（MegaScience翻译版）以及BenchMax评测框架的多语言评估集；

**📈 对比分析**

与Task Arithmetic、TIES、DARE等主流参数迁移与模型融合方法对比，ACT在保持原有数学性能的同时，平均科学推理准确率提升至19.2%（仅迁移4.7%参数），并能将多技能模型合并为单一模型而不显著降低任一技能；

**⚠️ 局限性**

局限性包括：仅针对同一预训练骨干的模型有效，跨架构迁移效果未知；通道定位依赖事先准备的能力数据集；目前仅验证了推理类能力，对其他任务类型的通用性仍待探索。

---

## 267. Research on Piano Timbre Transformation System Based on Diffusion Model

**arXiv ID:** 2601.09333 | [PDF](https://arxiv.org/pdf/2601.09333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 268. Formally Verifying Noir Zero Knowledge Programs with NAVe

**arXiv ID:** 2601.09372 | [PDF](https://arxiv.org/pdf/2601.09372v1)

**作者:** Pedro Antonino `[一作]` (Blockhouse Technology), Namrata Jain `[通讯]` (Blockhouse Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

开发了基于SMT的Noir程序形式化验证器，验证Noir/ACIR程序是否满足指定的约束条件。

**💡 创新点**

创新点在于对ACIR子语言进行SMT-LIB形式化，提供了两种编码（有限域与整数）并结合cvc5实现自动化检查。

**🔧 技术方法**

采用SMT-LIB、cvc5求解器、有限域和整数理论，以及对ACIR的黑盒函数、内存操作等高阶构造的形式化。

**📊 数据集**

使用Noir语言自带的测试程序集（约24个程序），涵盖验证通过与不通过的情况，且包含无范围检查和范围检查两种情形。

**📈 对比分析**

通过在Nargo中集成formal-verify命令，对三种求解后端（整数、cvc5-finite-field split、gb）进行跑时比较，发现没有绝对优者，范围检查大时会超时，表明需进一步优化。

**⚠️ 局限性**

局限在仅覆盖ACIR的一部分指令，未处理Call与部分黑盒函数，且对复杂范围检查缺乏有效抽象，导致大规模检查耗时或超时。

---

## 269. GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR

**arXiv ID:** 2601.09361 | [PDF](https://arxiv.org/pdf/2601.09361v1)

**作者:** Jiaying Zhang `[一作]` (Peking University), Renqing He `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为GeoRA的几何感知低秩自适应方法，用于强化学习与可验证奖励(RLVR)的参数高效微调。

**💡 创新点**

创新点在于利用RLVR更新子空间的低秩可压缩特性，通过对几何约束子空间进行SVD提取主方向进行初始化，并冻结残差，从而在保持预训练结构的同时实现稳定、高效的优化。

**🔧 技术方法**

技术主要包括基于SVD的低秩参数化、几何约束掩码构造（Spectral Prior与Euclidean Prior）以及残差“束缚”机制，全部实现为密集矩阵运算以兼容GPU加速。

**📊 数据集**

实验使用的模型包括Qwen3‑8B与Llama‑3.1‑8B，在DeepMath‑103K数据集上通过GRPO算法进行RLVR微调，并评估AIME、MATH‑500、OlymMATH（内域）以及HumanEval、GPQA、MMLU（外域）等基准。

**📈 对比分析**

与全量微调、稀疏微调、LoRA、PiSSA、MiLoRA等基线对比，GeoRA在所有数学与通用能力基准上均达成SOTA水平，且在参数、内存与训练时间上显著优于传统方法。

**⚠️ 局限性**

局限性包括需要一次性执行截断SVD与双重掩码预处理，且目前仅在RLVR推理任务上验证效果，尚需进一步检验其在更广泛模型与强化学习场景中的通用性。

---

## 270. Measuring the benefits of lying in MARA under egalitarian social welfare

**arXiv ID:** 2601.09354 | [PDF](https://arxiv.org/pdf/2601.09354v1)

**作者:** Jonathan Carrero `[一作]` (Universidad Complutense de Madrid), Fernando Rubio `[通讯]` (Universidad Complutense de Madrid)

**通讯引用:** 4849 | [OpenAlex ID](https://openalex.org/A5037998721)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在均衡社会福利下，代理人通过虚报偏好来提升自身收益，并用遗传算法寻找最佳欺骗策略

**💡 创新点**

提出在有限资源总和约束下可通过强制总和常数有效抑制代理人撒谎的实验验证

**🔧 技术方法**

使用双层遗传算法（LLGA求解最优分配，ULGA搜索最佳虚假偏好）

**📊 数据集**

实验数据采用随机生成的4名代理人10件资源的加性偏好向量，区分无限与有限总和两种约束

**📈 对比分析**

通过与真实偏好和预定义撒谎策略的比较，评估代理人在无限情形下始终能通过撒谎获利，而在有限情形下只有在对他人偏好估计极精确时才有优势；性能上遗传算法能快速得到近似最优解

**⚠️ 局限性**

限制在加性偏好、仅考虑单一代理撒谎、实验规模有限，且对真实环境的适用性和多代理同时撒谎的复杂度未深入探讨

---

## 271. Contraction of Rényi Divergences for Discrete Channels: Properties and Applications

**arXiv ID:** 2601.09328 | [PDF](https://arxiv.org/pdf/2601.09328v1)

**作者:** Adrien Vandenbroucque `[一作]` (École Polytechnique Fédérale de Lausanne), Michael Gastpar `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 12740 | [OpenAlex ID](https://openalex.org/A5063528341)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了离散通道上Rènyi散度的强数据处理不等式(SDPI)常数，并与传统φ-散度进行了对比，提出了新的上下界、结构条件以及与局部差分隐私和马尔可夫链收敛速度的关联。

**💡 创新点**

① 给出了α>1时Rènyi SDPI常数相较于φ-散度的显著差异；② 在α∈[0,1]区间内证明α-散度SDPI常数与Hellinger/χ²散度一致；③ 通过极限α→∞导出对∞-Rènyi SDPI的精确表述与超混合条件的等价性；④ 将∞-Rènyi SDPI与ε-局部差分隐私等价；⑤ 在马尔可夫链收敛分析中提供非线性收敛上界。

**🔧 技术方法**

主要使用信息理论工具（如数据处理不等式、Dobrushin系数、Rényi与φ-散度之间的关系）、凸分析与矩阵运算、极限与对数变换、以及经典通道模型（BSC、Z通道）的显式计算。

**📊 数据集**

使用标准离散通道模型：二元对称信道（BSC）、Z通道以及其张量乘积，未涉及真实数据集；主要通过解析和数值实验对这些通道的SDPI常数进行评估。

**📈 对比分析**

与传统的L^α范数收敛上界和χ²散度SDPI上界进行对比；结果表明在α>1、尤其是α=2时，Rènyi SDPI给出的收敛上界在小步数时更紧；但在某些通道（如BSC）和大步数时，传统上界仍更强；整体来看，提供了更丰富的收敛速度分析工具。

**⚠️ 局限性**

局限性：仅对有限字母空间的离散通道给出理论；对连续或高维分布的推广仍未完成；在α接近1时的数值稳定性与解析难度；部分上界在特殊通道（如BSC）不够紧；实际应用中需要进一步验证在更复杂马尔可夫链或隐私机制中的表现。

---

## 272. SpatialJB: How Text Distribution Art Becomes the "Jailbreak Key" for LLM Guardrails

**arXiv ID:** 2601.09321 | [PDF](https://arxiv.org/pdf/2601.09321v1)

**作者:** Zhiyi Mou `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 34888 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于空间布局的文本重排攻击（SpatialJB），通过将有害内容按二维图样重新排列，绕过Transformer模型的顺序语义分析和现有输出防护；

**💡 创新点**

创新点在于揭示Transformer在处理非线性空间文本时的结构盲点，并设计多样化的二维模板（如Acrostic、Telestich、Diagonal等）实现高效的空间攻击；

**🔧 技术方法**

利用Transformer的自回归序列处理机制与空间重排技术，结合大语言模型生成与评估；

**📊 数据集**

使用四大公开反恶意语言数据集（HateBenchSet、Hate‑Speech‑Offensive、OffensiveLong、动态生成仇恨数据集）进行攻击效果评估；

**📈 对比分析**

与Base64、Zulu、Ubbi Dubbi等传统攻击以及不同模型的输出防护（Llama Guard、Perspective API、OpenAI Moderation）对比，SpatialJB在多模型、多模板上成功率均超过90%，几乎达到100%，显著优于现有方法；

**⚠️ 局限性**

局限性包括对极其严格或改进的空间感知防护可能失效，且在部分需要深层推理或极低风险阈值的环境下仍可能被过滤或拒绝。

---

## 273. Draw it like Euclid: Teaching transformer models to generate CAD profiles using ruler and compass construction steps

**arXiv ID:** 2601.09428 | [PDF](https://arxiv.org/pdf/2601.09428v1)

**作者:** Siyi Li `[一作]`, Karl. D. D. Willis `[通讯]` (Autodesk Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种通过序列化几何构造步骤（如直线、圆、偏移、交点等）来生成 2D CAD 轮廓的模型。

**💡 创新点**

创新点在于将构造步骤视为“链式思考”，在设计者输入与最终轮廓之间插入可回放的几何程序，显著提升生成的几何有效性和满足约束的程度；同时通过强化学习进一步优化自交率和几何质量。

**🔧 技术方法**

采用基于 transformer 的自回归解码器，并在其上进行 ReMax、GRPO、RLOO 等序列级强化学习微调；构造序列使用自定义 DSL 表达。

**📊 数据集**

训练数据来源于 ABC 数据集中的 318,208 条闭合轮廓，经过 Open Cascade 切片、几何分析和构造步骤提取后生成。

**📈 对比分析**

与不使用构造步骤的基线模型对比：语法有效率从 88.1% 提升到 97.6%，自交率降低约 6%，边长不足率降至 1.7%；强化学习后各项指标进一步提升，尤其是自交率和对称性等高层约束。

**⚠️ 局限性**

局限性包括：需要人工设计构造步骤抽取算法，可能难以覆盖所有设计风格；模型在极端复杂或非常规几何上仍可能生成不理想结果；以及对动态参数调整（如尺寸变化）仍需额外处理。

---

## 274. TiInsight: A SQL-based Automated Exploratory Data Analysis System through Large Language Models

**arXiv ID:** 2601.09404 | [PDF](https://arxiv.org/pdf/2601.09404v1)

**作者:** Jun-Peng Zhu `[一作]` (Northwest A&F University), Qi Liu `[通讯]` (PingCAP)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于SQL的端到端自动化跨域探索性数据分析系统（TiInsight），实现了从自然语言查询到SQL生成、执行以及可视化结果展示的完整流程。

**💡 创新点**

创新点在于：①利用LLM生成层次化数据上下文（HDC）来快速理解任意数据模式；②结合问题澄清与分解、两阶段映射及自我修正链提升text‑to‑SQL的鲁棒性；③采用规则+LLM的可视化推荐机制，避免传统深度学习过度复杂；④在生产环境中实现多LLM切换与可书签的交互式Web UI。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）与CoT提示；向量数据库与map‑reduce框架实现高效表/列过滤；自我修正链（EXPLAIN/EXEC）提高SQL准确率；规则库与LLM协同的图表推荐；并通过并行线程实现HDC的高效生成。

**📊 数据集**

使用的代表性数据集有：①金融数据集（美国联邦基金利率及关键宏观经济指标）；②Bird数据集（鸟类检测实验结果）。

**📈 对比分析**

在PingCAP生产环境中部署并演示，通过用户交互与可书签对比验证系统可用性；目前未给出数值基准，但演示案例表明系统能在毫秒级到秒级完成SQL生成与可视化。

**⚠️ 局限性**

局限性包括：对LLM的过度依赖导致成本与推理时延；跨域文本‑to‑SQL在极端领域仍需进一步细调；部分视觉推荐仍受规则库覆盖率限制；系统在极大规模数据集上的性能与可扩展性待进一步验证。

---

## 275. FairGE: Fairness-Aware Graph Encoding in Incomplete Social Networks

**arXiv ID:** 2601.09394 | [PDF](https://arxiv.org/pdf/2601.09394v1)

**作者:** Renqiang Luo `[一作]` (Jilin University), Feng Xia `[通讯]` (RMIT University)

**通讯引用:** 18946 | [OpenAlex ID](https://openalex.org/A5089615958)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 FairGE，一种针对不完整社交网络的公平性图变换器框架，利用零填充和谱截断直接在不生成敏感属性的情况下实现公平节点表示。

**💡 创新点**

创新点在于将主特征向量与零填充相结合，既保持多跳邻居信息，又消除敏感属性与结构编码的耦合，避免敏感信息重建与隐私泄露。

**🔧 技术方法**

核心技术包括谱分解（取顶级特征向量）、零填充处理缺失敏感属性、位置编码、Transformer 自注意力网络以及对节点属性的矩阵投影。

**📊 数据集**

实验使用七个真实社交网络数据集：Facebook、Income、Bail、Credit、Pokec-z、Pokec-n 及 AMiner-S，涵盖多种敏感属性类型。

**📈 对比分析**

与 GCN、SGFormer、CoBFormer、Polynormer、FairGT、FMP、FairSIN、FairGNN、FairAC 等基线比较，FairGE 在 10%–60% 缺失率下平均提升 16% 以上的统计平衡与等机会，同时保持或提升预测准确率。

**⚠️ 局限性**

主要限制是对主特征向量数量与填充值策略的敏感性，需进一步调参和扩展至更复杂的多类别敏感属性场景。

---

## 276. AI-NativeBench: An Open-Source White-Box Agentic Benchmark Suite for AI-Native Systems

**arXiv ID:** 2601.09393 | [PDF](https://arxiv.org/pdf/2601.09393v1)

**作者:** Zirui Wang `[一作]` (Sun Yat-sen University), Michael R. Lyu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 41026 | [OpenAlex ID](https://openalex.org/A5069596903)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AI‑NativeBench，一个面向 AI‑Native 系统的白盒基准套件，覆盖 8 个真实应用、3 个领域、21 个系统变体，并通过分布式追踪分析行为正确性、性能开销与 token 经济性。

**💡 创新点**

创新点在于：①首次将 Model Context Protocol (MCP) 与 Agent‑to‑Agent (A2A) 作为基准核心，②采用“追踪优先”(trace‑first) 的评估框架，将 agentic span 视为一等公民，③揭示了参数悖论、推理主导的延迟以及昂贵的失败模式，为 AI‑Native 系统可靠性与成本控制提供系统证据。

**🔧 技术方法**

技术手段包括：OpenTelemetry 分布式追踪、MCP 与 A2A 协议实现、CrewAI、LangGraph、AutoGen 三大多代理框架、七种主流 LLM（GPT‑5、GPT‑4o‑mini、DeepSeek‑V3.1、DeepSeek‑R1、Gemini‑2.5‑flash、Qwen3‑235b 等）、token‑计量指标、精确与任意顺序匹配、重试率与失败分析。

**📊 数据集**

使用的数据集涵盖：Phishing Email、Resume Score Details、Markdown‑it 示例、ZSQL、Reasoning Engaging Story、Tech Keywords 以及多份人工合成数据；对部分应用（Markdown Validator、SQL Assistant 等）提供了 ground‑truth，保证了结果的可量化评估。

**📈 对比分析**

评估方法：在同一硬件（华为云 ECS）上对 7 个 LLM 进行 21 个系统变体的多指标对比（通过 pass‑rate、exact/any‑order match、延迟分解、token 费用等），结果显示轻量级模型在协议遵循上往往优于旗舰模型；推理模型导致延迟占比近 99%，而协议开销几乎可以忽略；重试与失败造成的 token 费用显著高于成功，证实了昂贵失败模式。

**⚠️ 局限性**

局限性包括：基准仅覆盖 8 个应用，缺乏更大规模或更复杂的场景；实验仅在单一硬件环境下进行，未考虑多云或边缘部署；对 LLM 的 prompt 设定固定，无法反映未来模型的自适应能力；追踪依赖 OpenTelemetry，若框架不兼容或存在隐式状态，可能导致信息缺失；缺乏对长期学习与自适应重构的评估。

---

## 277. Relation Extraction Capabilities of LLMs on Clinical Text: A Bilingual Evaluation for English and Turkish

**arXiv ID:** 2601.09367 | [PDF](https://arxiv.org/pdf/2601.09367v1)

**作者:** Aidana Aidynkyzy `[一作]` (Astana IT University), Şebnem Bora `[通讯]` (Ege University)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5005970237)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对英语和土耳其语两种语言，构建了首个双语临床关系抽取评测框架，并对比了大语言模型（LLM）与传统微调模型在该任务上的表现。

**💡 创新点**

创新点包括提出了基于对比学习的关系感知检索（Relation‑Aware Retrieval，RAR）方法，以及将链式推理（CoT）与检索结果动态结合的Output Format CoT提示策略。

**🔧 技术方法**

使用的技术主要是大语言模型的提示学习（prompt‑based few‑shot / zero‑shot）、对比学习的SimCSE、关系级嵌入、KNN检索（KATE）和Fine‑Tuned Relation Representation（PURE）。

**📊 数据集**

实验数据集为2010年i2b2/VA关系抽取语料的英土双语版本（训练1500条，测试500条），并通过专业医学译者人工校对确保标注一致性。

**📈 对比分析**

实验对比显示，RAR+Output‑Format CoT在Gemini 1.5‑Flash、DeepSeek‑V3等模型上实现了micro‑F1最高值0.918，明显优于随机、KATE、FT‑RR及微调BERT/PURE等基线。

**⚠️ 局限性**

局限性包括仅使用小规模双语子集、对土耳其语资源仍相对稀缺、部分模型（如GPT‑4o‑mini）对多语种适配不足，以及实验依赖API调用，未探讨更大规模或跨域泛化能力。

---

## 278. Detail Loss in Super-Resolution Models Based on the Laplacian Pyramid and Repeated Upscaling and Downscaling Process

**arXiv ID:** 2601.09410 | [PDF](https://arxiv.org/pdf/2601.09410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. Wataridori is NP-Complete

**arXiv ID:** 2601.09345 | [PDF](https://arxiv.org/pdf/2601.09345v1)

**作者:** Suthee Ruangwises `[一作]` (Chulalongkorn University), Suthee Ruangwises `[通讯]` (Chulalongkorn University)

**通讯引用:** 299 | [OpenAlex ID](https://openalex.org/A5030622574)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了Wataridori数独样棋的可解性问题为NP-完全，通过从已知NP-完全的Numberlink问题直接构造多边形块的多项式时间归约。

**💡 创新点**

创新点在于首次使用另一种数独棋归约（Numberlink）来证明Wataridori的NP-难度，并设计了基于多边形块的路径“折叠”机制来映射路径长度与区域数。

**🔧 技术方法**

采用了多项式时间归约技术，构造了(4k+5)m×(4k+5)n大小的Wataridori实例，并使用“zig‑zag”路径设计保证区域计数对应。

**📊 数据集**

无数据集，纯理论证明。

**📈 对比分析**

无实验比较，论文仅提供理论证明，不涉及性能评估。

**⚠️ 局限性**

局限在于只证明了某个变体（不要求覆盖所有格子）的Wataridori为NP-完全，未讨论实际求解器或更广泛的棋子配置。

---

## 280. Improving Implicit Hate Speech Detection via a Community-Driven Multi-Agent Framework

**arXiv ID:** 2601.09342 | [PDF](https://arxiv.org/pdf/2601.09342v1)

**作者:** Ewelina Gajewska `[一作]` (Warsaw University of Technology), Jarosław A Chudziak `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个由中央 Moderator 代理和多个专属 Community 代理组成的多代理系统，用以检测社交媒体中的隐式仇恨言论。

**💡 创新点**

创新点在于通过 Community 代理注入目标群体的社会文化背景，实现身份感知的咨询式决策，显著提升了检测的准确性与公平性。

**🔧 技术方法**

使用了大语言模型 Gemini‑2.5‑Flash、AutoGen 框架、多阶段知识检索与交叉注意力嵌入、以及多代理协同工作流程。

**📊 数据集**

实验采用 ToxiGen 数据集（约 274,000 条短文本，8,960 条人工标注的隐式仇恨样本）进行评估。

**📈 对比分析**

与零样本、少样本、链式推理（CoT） prompting 等基线方法对比，Agentic 方法在所有目标群体中获得最高的 TPR、bACC（平均 0.86）和 F1（0.86），差异显著（p<0.001）。

**⚠️ 局限性**

局限性包括：依赖 LLM 的检索与生成，可能出现幻觉；Community 代理仅基于 Wikipedia，知识覆盖有限；对某些群体（如 LGBTQ、女性）仍存在误判；以及手工定义群体与查询导致扩展性受限。

---

## 281. A Deep Dive into OpenStreetMap Research Since its Inception (2008-2024): Contributors, Topics, and Future Trends

**arXiv ID:** 2601.09338 | [PDF](https://arxiv.org/pdf/2601.09338v1)

**作者:** Yao Sun `[一作]` (Technical University of Munich), Xiao Xiang Zhu `[通讯]` (Technical University of Munich)

**通讯引用:** 24693 | [OpenAlex ID](https://openalex.org/A5068384981)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统性文献回顾和计量学方法，对2008‑2024年OpenStreetMap相关研究进行全面梳理，揭示其发展轨迹、核心贡献者、研究主题演进与未来趋势。

**💡 创新点**

创新点在于首次将学术文献（WoS）与社区活动（SotM）相结合，构建跨界知识图谱，揭示学术与社区的互补与差异。

**🔧 技术方法**

采用VOSviewer、bibliometrix以及Python脚本进行文献计量、关键词共现、共作者网络与主题演进可视化分析。

**📊 数据集**

主要数据集为WoS核心合集中的1926篇OpenStreetMap相关论文及782条SotM会议演讲记录，时间截至2024年6月。

**📈 对比分析**

通过对发表量、引用次数、作者和机构合作网络、关键词趋势等多维指标的量化对比，证明了OSM研究从数据质量到应用扩展的演化，并发现学术与社区议题存在时间滞后约2.8年。

**⚠️ 局限性**

局限性包括：依赖WoS检索导致部分相关论文遗漏；社区数据缺乏统一索引，SotM演讲信息有限；关键词抽取可能忽略同义词和领域术语，影响主题聚类准确性。

---

## 282. LISP -- A Rich Interaction Dataset and Loggable Interactive Search Platform

**arXiv ID:** 2601.09366 | [PDF](https://arxiv.org/pdf/2601.09366v1)

**作者:** Jana Isabelle Friese `[一作]` (University of Duisburg-Essen), Nicola Ferro `[通讯]` (University of Padua)

**通讯引用:** 4260 | [OpenAlex ID](https://openalex.org/A5069843101)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

创建并公开了一个可复现的互动信息检索（IIR）实验平台及其对应的交互日志数据集，涵盖61名参与者在两种兴趣程度主题下的搜索行为、感知速度、搜索经验和人口统计信息。

**💡 创新点**

创新点在于将完整的研究设计、基础设施和详细的用户特征数据三者统一打包为可共享资源，满足Gäde等提出的5级可重用性标准，并首次公开同时记录感知速度与兴趣对搜索行为的影响。

**🔧 技术方法**

使用了基于Terrier的BM25检索、Web端交互日志记录框架lisp、修改版Finding A’s感知速度测试，并配合MySQL存储结果。

**📊 数据集**

数据集来源于2025年在德国高校开展的实验，包含122个搜索会话的日志文件、参与者的人口统计、感知速度得分、主题兴趣评级和搜索专业水平。

**📈 对比分析**

通过Wilcoxon符号秩检验、Mann‑Whitney U检验以及Markov链相似度度量（Frobenius范数、Jensen‑Shannon散度、KS检验）评估感知速度与兴趣对搜索行为的影响，结果显示兴趣在查询频次、文档点击及标记上显著，但感知速度差异不显著；整体性能上并未与现有模型对比。

**⚠️ 局限性**

主要局限包括样本量小、单一学术域背景、缺乏跨语言或跨领域验证，以及在更大规模、真实场景中的可推广性待进一步验证。

---

## 283. A Systematic Security Analysis for Path-based Traceability Systems in RFID-Enabled Supply Chains

**arXiv ID:** 2601.09407 | [PDF](https://arxiv.org/pdf/2601.09407v1)

**作者:** Fokke Heikamp `[一作]`, Sushmita Ruj `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文对RFID支持链条的追溯系统进行了系统性安全分析，提出以路径为核心的安全框架并评估了17种现有方案。

**💡 创新点**

创新点在于将路径属性（完整性、排序、授权、隐私）统一到一个可度量的框架中，并构建新的攻击分类，首次实现大规模追溯系统安全评估。

**🔧 技术方法**

主要使用RFID、区块链、PKI、PUF、同态加密、多重签名等技术，框架基于事件模型和路径属性来建模系统行为。

**📊 数据集**

实验基于从17篇论文中提取的系统实现细节进行人工建模，并未使用公开数据集，而是利用文献描述构建系统模型。

**📈 对比分析**

使用该框架对每个系统进行路径安全属性、授权与隐私评估，并记录在 Adv_T 与 Adv_R 模型下的满足情况；评估显示多数系统缺乏完整性或授权，存在重路、越权、隐私泄露等漏洞。

**⚠️ 局限性**

方法为半形式化分析，无法完整覆盖所有隐私属性，且对非RFID追溯方案适用性有限，需要进一步的形式化验证与扩展。

---

## 284. On Decoding First- and Second-Order BiD Codes

**arXiv ID:** 2601.09390 | [PDF](https://arxiv.org/pdf/2601.09390v1)

**作者:** Devansh Jain `[一作]` (Indian Institute of Technology Hyderabad), Lakshmi Prasad Natarajan `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 303 | [OpenAlex ID](https://openalex.org/A5053548670)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文针对Berman‑Intersection‑Dual (BiD) 代码族提出了高效的最大似然 (ML) 与最大对数-最大似然 (max‑log‑MAP) 解码器，并利用最小权重校验码与投影性质设计了一种针对二阶 BiD 代码的置信传播 (BP) 解码器；

**💡 创新点**

创新点在于：①首次推导出 BiD 代码的最小权重校验码并证明其投影属性；②基于该属性构造 BP 因子图，实现在 81、243、729 长度下接近 ML 性能的低复杂度解码；③提出了一种兼容不同阶数 BiD 代码的通用解码框架；

**🔧 技术方法**

主要技术包括：离散傅里叶变换 (DFT) 频域分析、Abelian 码的理想结构、递归子积码的快速解码、投影映射与自动同构、图论中的最大团搜索以及软/硬输入软/输出 (SISO) 解码模块；

**📊 数据集**

使用了自定义的二进制高斯白噪声 (AWGN) 通道仿真数据，长度分别为 81、243、729 的 BiD 代码，比较基准为 RM 代码、Polar 代码以及 CRC‑Aided Polar（SCL-8、SCL-5）等；

**📈 对比分析**

与基准方法对比，BiD 代码在 81 与 243 长度下，BP 解码的误码率仅落后 1 dB 左右 ML 解码，且与 CRC‑Aided Polar（SCL-8）相近；在 729 长度下，BP 与 SCL-8 误码率差距约 0.25 dB；

**⚠️ 局限性**

局限性包括：因子图存在大量短环导致收敛问题；BP 因子图实际上解码的是 (m,2,2)⊕(m,0,0) 超码，未能完整利用二阶 BiD 代码的校验结构；需引入更多低权重校验或采用消解技巧以进一步逼近 ML 性能。

---

## 285. SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing

**arXiv ID:** 2601.09385 | [PDF](https://arxiv.org/pdf/2601.09385v1)

**作者:** Ziyang Ma `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2053 | [OpenAlex ID](https://openalex.org/A5090959358)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 SLAM-LLM，一个可模块化、开源的多模态大语言模型框架，专注于语音、音频和音乐处理。

**💡 创新点**

创新点在于统一的 encoder–projector–LLM 结构，支持多种声学编码器、投影层和 LLM，且提供 PEFT、LoRA 等高效微调插件，填补了现有框架对非视觉模态的支持空白。

**🔧 技术方法**

实现中使用 Whisper、HuBERT、WavLM、EAT、CLAP 等声学编码器，Q‑Former、线性投影等投影模块，LLaMA、Vicuna、Qwen 等大语言模型，并结合 LoRA、Q‑Former、RAG 等技术。

**📊 数据集**

实验使用了 LibriSpeech、GigaSpeech、Clotho、AudioCaps、WavCaps、LP‑MusicCaps、AudioSet 等公开数据集，覆盖 ASR、AAC、CASR、VSR、S2TT、SEC 等任务。

**📈 对比分析**

与现有最佳系统对比，SLAM‑LLM 在多项基准上取得或逼近 SOTA，尤其在 ASR（WER 2.58%）、AAC（METEOR 51.5%/CIDEr 84.1%）和 CASR（B‑WER 1.27%）等任务上显著提升。

**⚠️ 局限性**

局限性包括对大规模预训练模型和算力的高度依赖、Whisper 截断导致性能下降、以及仍需进一步优化跨模态对齐和多任务统一训练等方面。

---

## 286. Long-term Task-oriented Agent: Proactive Long-term Intent Maintenance in Dynamic Environments

**arXiv ID:** 2601.09382 | [PDF](https://arxiv.org/pdf/2601.09382v1)

**作者:** Qinglong Shi `[一作]` (University of Science and Technology of China), Renqing He `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了可主动追踪用户长期意图的对话代理框架

**💡 创新点**

提出意图条件监控与事件触发跟进两大核心能力，并创建ChronosBench基准

**🔧 技术方法**

采用基于LLM的多代理生成评估管线、LoRA微调以及结构化JSON响应

**📊 数据集**

使用1,052条人工合成对话数据，包括简单和复杂场景的意图转移

**📈 对比分析**

在ChronosBench上与闭源模型对比，微调后的Qwen3-32B在复杂场景中完成率达85.19%，显著优于其他模型

**⚠️ 局限性**

模拟与现实差距、交互深度有限、以及对用户干扰阈值缺乏个性化

---

## 287. Query Languages for Machine-Learning Models

**arXiv ID:** 2601.09381 | [PDF](https://arxiv.org/pdf/2601.09381v1)

**作者:** Martin Grohe `[一作]` (RWTH Aachen University), Martin Grohe `[通讯]` (RWTH Aachen University)

**通讯引用:** 10980 | [OpenAlex ID](https://openalex.org/A5073893026)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出并分析两种用于加权有限结构的逻辑FO(SUM)与其递归扩展IFP(SUM)，并将其视为查询语言，用以描述和评估神经网络等机器学习模型的行为；

**💡 创新点**

首次将加权结构上的求和聚合与固定点递归相结合，构造可在有限模型理论框架内表达神经网络功能与可视化查询的逻辑；

**🔧 技术方法**

利用有限模型理论、Ehrenfeucht‑游戏、聚合算子（求和、计数、平均、最小/最大）以及固定点逻辑实现递归定义，并在理论上证明其表达能力与复杂度；

**📊 数据集**

该工作主要是理论性研究，不依赖具体实验数据集；

**📈 对比分析**

未进行实验比较，而是通过理论证明展示在多项式时间与LOGSPACE/UC^0等复杂度类中的可评估性，指出在无穷深度时需要固定点逻辑；

**⚠️ 局限性**

局限在于：对无穷深度的FNN只能通过固定点逻辑处理；表达能力受聚合算子限制，部分重要的模型无关查询无法在IFP(SUM)中表达；与更强的（f）逻辑相比，在有限深度下可模拟但在无穷深度时不可；

---

## 288. The Imperfective Paradox in Large Language Models

**arXiv ID:** 2601.09373 | [PDF](https://arxiv.org/pdf/2601.09373v1)

**作者:** Bolei Ma `[一作]` (LMU Munich), Yusuke Miyao `[通讯]` (University of Tokyo)

**通讯引用:** 5453 | [OpenAlex ID](https://openalex.org/A5004444958)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在事件组合语义，尤其是不完全时态悖论（Imperfective Paradox）方面的推理能力，提出诊断数据集 ImperfectiveNLI 并系统评估多种开源 LLM 在不同 prompt 条件下的表现。

**💡 创新点**

创新点包括：①构造基于 Vendler 视角的四类动词最小对照实验，生成 ImpracticalNLI 数据集；②引入 Teleological Bias Rate（TBR）和 Aspectual Awareness Gap（ΔAA）两个量化指标，用以分离表示层与推理层的偏差；③通过对模型规模、prompt 以及细粒度动词子类的交叉实验，揭示 LLM 在决策时受强烈目标实现偏置支配而非语义混淆；④首次系统比较 Prompt (Strict Logic、DAP、CoT、Counterfactual) 对偏差与校准的双重影响。

**🔧 技术方法**

技术手段：多模型 NLI 评估（7款 7‑9B 开源 LLM）；Prompt 工程四种策略；句子级嵌入余弦相似度分析；统计相关性与显著性检验；在模型规模（1.5B~72B）与动词子类上做细粒度实验。

**📊 数据集**

使用数据集：ImperfectiveNLI（400 条 minimal‑pair 句子，包含 telic/atelic 动词 100+100，细分为四个语义子类；四组逻辑条件 A‑D）。此外引用公开的 Vendler 词类资源与标准 NLI 标签。

**📈 对比分析**

比较方法：在每个模型与每种 prompt 下分别计算 A‑D 组准确率、TBR_C、ΔAA；与现有基线（零射）对比。结果显示：零射下 telic 进展句子几乎总是被误判为 entailment（TBR≈1）；CoT 能显著降低 TBR 但同时导致 atelic 组准确率下降；Counterfactual 能消除 TBR 但使 D 组接近 0。模型规模越大，TBR 越低，ΔAA 越高，出现 32B 级别的“phase shift”。

**⚠️ 局限性**

限制：①仅在英语、模板化句式下评估，缺乏语法与语篇多样性；②仅使用推理时 Prompt，未尝试 fine‑tuning 或参数高效对齐；③未跨语言、跨方言检验；④未深入探讨脚本知识与时间先验如何与动词子类交互影响；⑤表示与推理分离的结论基于余弦相似度，仍需更细致的表征分析。

---

## 289. Lower Bounds in Algebraic Complexity via Symmetry and Homomorphism Polynomials

**arXiv ID:** 2601.09343 | [PDF](https://arxiv.org/pdf/2601.09343v1)

**作者:** Prateek Dwivedi `[一作]` (IT Universitetet i København), Tim Seppelt `[通讯]` (IT Universitetet i København)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文发展了对称代数复杂性理论，通过引入对称代数电路、斜电路和公式的对称类，展示了这些类之间的无条件分离。

**💡 创新点**

创新点在于定义了对称代数复杂性类，并证明了这些类与传统复杂性类之间的无条件分离，特别是通过同态多项式的线性组合来表征这些类。

**🔧 技术方法**

使用了同态多项式、模型理论技术和图结构理论等方法。

**📊 数据集**

使用了具有界定树宽和路径宽的双向多重图的同态多项式作为数据集。

**📈 对比分析**

通过与传统复杂性类的比较，证明了对称类包含了许多 - 完全和 - 完全的多项式，显示出其强大的计算能力。

**⚠️ 局限性**

限制在于对称电路的复杂性与非对称电路的复杂性之间的关系尚未完全明确，且在某些情况下，无法确定所有模式族的复杂性。

---

## 290. CallShield: Secure Caller Authentication over Real-Time Audio Channels

**arXiv ID:** 2601.09327 | [PDF](https://arxiv.org/pdf/2601.09327v1)

**作者:** Mouna Rabh `[一作]`, Issa Khalil `[通讯]` (Qatar Computing Research Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了一套可在标准电话链路上实时嵌入并验证身份信息的音频水印身份验证系统。

**💡 创新点**

创新点包括：① 40 ms帧内单比特实时神经水印；② 轻量级音频数据链路协议（同步、BCH纠错、ARQ）；③ 将对称密钥挑战–响应协议嵌入音频，实现端到端身份验证。

**🔧 技术方法**

采用的技术包括：神经网络音频水印（AudioSeal架构＋注意力模块）、BCH码、Stop‑and‑Wait ARQ、HMAC‑SHA256、低比特率（25 bps）音频通道。

**📊 数据集**

使用的主要数据集为 Switchboard‑1 电话语音库（约 260 小时），用于训练、验证和测试。

**📈 对比分析**

与现有水印/语音识别方案对比，RTAW 在 8 kHz 40 ms 帧下实现 25 bps，PESQ>4.2、STOI>0.94；整体身份验证成功率干净音频 99.2%、含干扰 95%+，平均认证时间 63 s，误识率 0%。

**⚠️ 局限性**

局限性包括：需预共享对称密钥；水印容量受限于 25 bps；未针对突发错误建模；对抗性攻击/深度伪造防御主要依赖密钥安全；实现受限于低功耗设备的算力。

---

## 291. Class Adaptive Conformal Training

**arXiv ID:** 2601.09522 | [PDF](https://arxiv.org/pdf/2601.09522v1)

**作者:** Badr-Eddine Marani `[一作]`, Jose Dolz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Class Adaptive Conformal Training（ClassAT）方法，利用增强的拉格朗日乘子法（ALM）自适应学习每个类别的约束权重，从而在不假设数据分布的前提下实现类条件的预测集优化。

**💡 创新点**

创新点在于：① 用类别级别的惩罚系数代替单一全局系数，克服了现有 CP 训练对类别分布差异的敏感性；② 通过 ALM 自动更新惩罚系数和乘子，避免了人工调参；③ 采用可微分的非一致性分数与量化阈值，使得模型可端到端训练，同时保持 CP 的覆盖保证。

**🔧 技术方法**

技术手段包括：Conformal Prediction（CP）框架、可微分的非一致性分数（THR、APS、RAPS）、拉格朗日乘子法与增强惩罚函数（PHR）、sigmoid 平滑指示函数、可微分排序量化算子，以及标准交叉熵与 Focal Loss 对比训练。

**📊 数据集**

使用的公开数据集涵盖图像分类与文本分类：MNIST、CIFAR‑10、CIFAR‑100、ImageNet 及其长尾版本（MNIST‑LT、CIFAR‑LT、ImageNet‑LT），并在 20 Newsgroups 文本分类任务上进行验证。

**📈 对比分析**

与 CE、FL、ConfTr、CUT、InfoCTr（Fano、MB‑Fano、DPI）和 DPSM 等基线方法进行对比；实验表明 ClassAT 在平均集大小（S）显著下降、覆盖率（C）保持在 1‑α、类别条件覆盖差距（CG）大幅降低，同时在 Top‑1/Top‑3 准确率上不落后甚至略优，尤其在类数众多或分布严重不平衡的数据集上表现突出。

**⚠️ 局限性**

局限性包括：① 需要额外的验证集来估计乘子，增加了实验成本；② ALM 的收敛性在非凸场景下仍缺乏严格理论保证；③ 平滑处理在训练阶段可能略微影响覆盖保证，需在实际部署时恢复非平滑实现；④ 对极端长尾或极大类数（如 ImageNet‑LT）时，乘子更新仍可能需要细致调参。

---

## 292. Error Exponents for Randomised List Decoding

**arXiv ID:** 2601.09519 | [PDF](https://arxiv.org/pdf/2601.09519v1)

**作者:** Henrique K. Miyamoto `[一作]` (Universite Paris Saclay), Sheng Yang `[通讯]` (Universite Paris Saclay)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了随机化列表解码的随机编码误差指数，给出了匹配、非匹配以及通用度量下的误差指数，并在固定列表尺寸和指数列表尺寸两种情形下分别推导了相应的下界。

**💡 创新点**

创新点包括：①首次在固定列表尺寸下得到ensemble‑tight的误差指数，证明匹配度量时列表尺寸不提升指数；②在指数列表尺寸下给出高率下紧凑的下界，并证明其在高率区间与确定性列表解码等价；③将随机化列表解码与传统的随机化和确定性解码进行理论比较。

**🔧 技术方法**

采用的方法主要是类型枚举法、方法论、Markov不等式、Lambert W函数以及对误差概率的积分变换和大偏差分析，扩展了经典的误差指数技术。

**📊 数据集**

实验示例主要基于二进制对称信道（BSC，交叉概率0.1）来展示误差指数曲线。

**📈 对比分析**

通过与普通随机化解码、确定性列表解码以及Sphere‑Packing上限的对比，结果表明：固定列表尺寸下随机化解码的误差指数与普通解码相同；在指数列表尺寸下，在高率区间能够达到最优误差指数，优于普通随机化解码。

**⚠️ 局限性**

局限性包括：①固定列表尺寸时随机化解码无法获得与确定性列表解码相同的优势；②指数列表尺寸下的结果主要适用于高率区间，对低率场景仍可能存在性能缺口；③研究聚焦于随机复合编码，未涉及实现复杂度和实际编码器设计。

---

## 293. On Numbers of Simplicial Walks and Equivalent Canonizations for Graph Recognition

**arXiv ID:** 2601.09506 | [PDF](https://arxiv.org/pdf/2601.09506v1)

**作者:** Marek Černý `[一作]` `[通讯]` (University of Antwerp), Marek Černý (University of Antwerp)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种更高阶的图走访方法——simplicial walks，并基于其提出了 (k,h)-SW refinement 与 SW 自动机，旨在通过对图属性的同构识别和同构不可区分度进行更精细的分析；

**💡 创新点**

主要创新点包括：①将路径宽度类（𝒫_{k,h}）的同胚计数与 simplicial walks 的颜色计数建立等价关系；②提出多重对偶自动机（MIA）并利用前向归约实现规范化，使得 SW 自动机在大小与可计算性上优于传统的多重自动机；③将 Weisfeiler–Leman（WL）与 simplicial walks 结合，形成新的可解释性图属性识别框架；

**🔧 技术方法**

使用了 Weisfeiler–Leman refinement、同胚张量、线性代数（最小化实现、行列式判定）、多重对偶自动机、Caterpillar decomposition、以及受限并列一阶逻辑（restricted‑conjunction logic）的逻辑可定义性等技术；

**📊 数据集**

论文未在实验上使用公开数据集，主要聚焦于理论证明与算法复杂度分析；

**📈 对比分析**

与传统 WL 对比，SW 的规范化算法在理论上实现了 O(k n³k) 的时间复杂度，整体同胚不可区分度判定的多项式算法为 (k n⁴k)(k n³k)；虽然理论上更快，但在实际大规模图上的实验评估尚未给出；

**⚠️ 局限性**

局限性主要包括：①对 (k,h)-SW 何时收敛的步骤上界尚未确定；②缺乏实验验证，无法评估在真实图数据上的性能；③算法对高树宽/路径宽图仍可能存在较高的常数因子，实际效率需进一步验证。

---

## 294. MVSS: A Unified Framework for Multi-View Structured Survey Generation

**arXiv ID:** 2601.09504 | [PDF](https://arxiv.org/pdf/2601.09504v1)

**作者:** Yinqi Liu `[一作]` (Beijing University of Technology), Cunxiang Wang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 MVSS，一种多视角结构化综述生成框架，先构建基于引文的层级知识树，再生成与树对应的比较表，最后通过树表联合约束生成综述文本；

**💡 创新点**

将结构化视角（层级树、比较表）作为首要优化目标，实现跨视角一致性与证据对齐，并引入多模型轮廓共识和迭代树结构细化；

**🔧 技术方法**

使用检索增强生成（RAG）、层级知识树构造与细化、表格生成模型、轮廓生成与评判模型、以及跨视角对齐机制；

**📊 数据集**

在 76 个计算机科学主题上，基于 2018‑2024 年的 530,000 篇 arXiv 论文；

**📈 对比分析**

与人类专家、Naive RAG、AutoSurvey、HiReview 等基线对比，在 8k、16k、32k、64k 词长度下，MVSS 在覆盖率、结构性、相关性等指标上均取得最高平均分，并在 64k 级别达到与专家级别相当的 5.0 分；

**⚠️ 局限性**

依赖大模型生成与评判，成本高、可复现性差；超参敏感；仅在 CS 与 arXiv 语料上验证，缺乏跨学科与时间演化的支持。

---

## 295. Parallelizable memory recurrent units

**arXiv ID:** 2601.09495 | [PDF](https://arxiv.org/pdf/2601.09495v1)

**作者:** Florent De Geeter `[一作]` (Montefiore Institute), Guillaume Drion `[通讯]` (Montefiore Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了能够通过多重稳态实现持久记忆且可并行计算的记忆递归单元(MRU)，并给出了基于滞后分岔的具体实现——BMRU

**💡 创新点**

创新点在于将多重稳态与可并行的内部时钟相结合，消除瞬态动态，保留持久记忆，同时保持与传统门控RNN相似的可并行化结构

**🔧 技术方法**

采用内部时钟的RNN框架、隐式函数逼近、可并行扫描算法、替代梯度(Surrogate Gradient)进行反向传播，并与SSM（如S4/Mamba）混合

**📊 数据集**

在合成的长记忆复制任务、排列的序列MNIST（含黑色噪声）以及Pathfinder长序列任务上进行实验

**📈 对比分析**

与单一LSTM/GRU（LRU）以及SSM单元比较，BMRU在长时间记忆任务上表现更好，能保持低MSE或高准确率，且在加噪声时仍能保持性能；与混合模型结合可获得更佳的长期依赖学习效果

**⚠️ 局限性**

对更深层网络的支持有限，BMRU在多层时可能出现量化误差和学习困难；使用不可微分函数导致训练需要替代梯度，可能影响收敛性和泛化；目前仅在分类/回归任务验证，未在生成任务中测试

---

## 296. SlidesGen-Bench: Evaluating Slides Generation via Computational and Quantitative Metrics

**arXiv ID:** 2601.09487 | [PDF](https://arxiv.org/pdf/2601.09487v1)

**作者:** Yunqiao Yang `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 39556 | [OpenAlex ID](https://openalex.org/A5100732450)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SlidesGen-Bench，统一评估自动幻灯片生成的内容、审美和可编辑性。

**💡 创新点**

创新在于基于视觉的无参考评估、定量三维度指标和人类偏好对齐的数据集。

**🔧 技术方法**

使用图像处理算法（色彩和对比度评估）、多模态LLM（生成QuizBank、评估）、统计方法和对齐技术。

**📊 数据集**

Slides-Align1.5k 数据集，涵盖九个主流生成系统七种场景的幻灯片。

**📈 对比分析**

与现有LLM-as-Judge、PPTAgent等对比，Spearman 0.71，识别率32.6%，在人类偏好上的相关性显著优于基线。

**⚠️ 局限性**

仅评估静态视觉，忽略动画和多语言等情况；且对高层结构化生成能力仍有限。

---

## 297. Bridging Semantic Understanding and Popularity Bias with LLMs

**arXiv ID:** 2601.09478 | [PDF](https://arxiv.org/pdf/2601.09478v1)

**作者:** Renqiang Luo `[一作]` (Jilin University), Shuo Yu `[通讯]` (Dalian University of Technology)

**通讯引用:** 1329 | [OpenAlex ID](https://openalex.org/A5004781883)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FairLRM 框架，在 RecLLM 中通过结构化提示实现对流行度偏见的双侧（用户侧和物品侧）语义拆解与缓解。

**💡 创新点**

首次将流行度偏见拆解为用户与物品两层，并在 LLM 交互中嵌入分层用户偏好信息，实现双侧语义驱动的去偏。

**🔧 技术方法**

使用大型语言模型（Qwen‑max、Llama‑7B）配合结构化指令式提示、Pareto 物品/用户分组以及多指标评估（LtC、MRMC、MRR、F1）。

**📊 数据集**

在 MovieLens‑20M 与 Goodbooks‑10k 两个领域数据集上进行实验，采用时间划分与 70/30 训练/测试分割。

**📈 对比分析**

与 Vanilla、Popularity Debiasing、Diversity 三个基线在多指标下对比，FairLRM 在 LtC 提升、MRMC 降低、MRR/F1 维持或提升，证明其在公平与准确性上均优于传统方法。

**⚠️ 局限性**

仅针对单一流行度偏见，未探讨多种偏见交叉；依赖固定阈值的用户/物品分组；LLM 黑盒性质限制可解释性。

---

## 298. Dissecting Judicial Reasoning in U.S. Copyright Damage Awards

**arXiv ID:** 2601.09459 | [PDF](https://arxiv.org/pdf/2601.09459v1)

**作者:** Pei-Chi Lo `[一作]` (National Sun Yat-sen University), Thomas Y. Lu `[通讯]` (National Sun Yat-sen University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5087129350)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于话语结构的LLM框架，系统提取并量化美国版权损害赔偿判决中的司法推理模式。

**💡 创新点**

首次将修辞结构理论（RST）与Agentic LLM流程相结合，实现了对司法意见层级关系的自动解析与解释，并通过计划优化提升推理准确性。

**🔧 技术方法**

采用RST解析、两阶段Agentic LLM（Plan Optimizer/Executor）、ChatGPT‑4o‑mini以及自然语言线性化技术来捕捉话语层次。

**📊 数据集**

利用LexisNexis检索的2183例版权损害案件，经过筛选和专家双层标注得到203例手工标注样本，作为训练与评估集。

**📈 对比分析**

与随机基线、普通LLM、零样本链式思考、仅Agentic LLM等5种方法对比，Agentic+ToD在50例测试集上达到77.1%准确率、76.5% F1、100%召回率，优于其它模型。

**⚠️ 局限性**

受限于RST解析准确性、数据集中仅包含联邦巡回法院判决、缺乏时间序列与州级案例，且模型对高层级语义仍有误判。

---

## 299. DepRadar: Agentic Coordination for Context Aware Defect Impact Analysis in Deep Learning Libraries

**arXiv ID:** 2601.09440 | [PDF](https://arxiv.org/pdf/2601.09440v1)

**作者:** Yi Gao `[一作]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University), Xin Xia `[通讯]` (State Key Laboratory of Blockchain and Data Security, Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于多智能体协同的框架，自动从深度学习库的 PR/commit 中提取结构化缺陷语义，并基于此判断 downstream 代码是否受影响。

**💡 创新点**

创新点包括：①多角色智能体（Miner、Diff Analyzer、Orchestrator、Impact Analyzer）协同工作；②逐步上下文扩展（Progressive Context Augmentation）解决 LLM token 限制；③结合 AST 静态分析与领域规则校验，降低 LLM 幻觉；④将内部实现变更映射为用户可见的触发条件，实现精确影响评估。

**🔧 技术方法**

主要技术：大型语言模型（如 Llama‑3）、AutoGen 多智能体框架、Python AST 静态分析、正则/规则引擎映射、增量上下文扩展机制、LLM 与代码解析的融合。

**📊 数据集**

数据集：从 Transformers (157 PRs) 和 Megatron (70 commits) 收集的真实更新；122 个真实下游客户端程序；对 PR 进行人工标注以验证缺陷与影响。

**📈 对比分析**

与三种基线（FlatLLM Base、FlatLLM Reasoning、PyCG）对比：缺陷识别 F1 达 95%（精准率 90%/召回率 99%），影响分析 F1 达 85%（精准率 80%/召回率 90%）。在所有指标上均显著优于基线。

**⚠️ 局限性**

局限性：①依赖专家标注；②仍可能出现 LLM 幻觉导致误判；③对动态运行时路径的捕获不足；④目前仅适用于 Python 生态；⑤对缺乏结构化 PR/commit 的项目效果有限。

---

## 300. GlovEgo-HOI: Bridging the Synthetic-to-Real Gap for Industrial Egocentric Human-Object Interaction Detection

**arXiv ID:** 2601.09528 | [PDF](https://arxiv.org/pdf/2601.09528v1)

**作者:** Alfio Spoto `[一作]` (University of Catania), Giovanni Maria Farinella `[通讯]` (University of Catania)

**通讯引用:** 5608 | [OpenAlex ID](https://openalex.org/A5087180022)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结合合成数据与扩散模型的工业视角人机交互（EHOI）数据生成框架，并构建了GlovEgo-HOI基准数据集与GlovEgo-Net检测模型。

**💡 创新点**

创新点在于把工业PPE（手套）自动标注与真实图像扩增相结合，并在模型中引入手部姿态与PPE检测的双重头部。

**🔧 技术方法**

使用Unity+Perception生成合成数据，FLUX扩散模型对实景图像添加手套，采用ResNet-101+FPN为骨干，配合检测、分割、深度、姿态等多头模块。

**📊 数据集**

利用GlovEgo-HOI（合成+实景）数据集进行训练与评估，同时借助EgoISM-HOI等已有工业数据做对比。

**📈 对比分析**

在Real-Only、Synth-Only和Synth+Real三种训练策略下，Synth+Real在AP Hand+State、mAP Hand+Obj、mAP Hand+All等指标上提升约3–7%，并在交互识别精度上显著优于其它方案。

**⚠️ 局限性**

局限在于仍需手工标注的实景数据，扩增模型可能产生伪影，且在更复杂的动态场景中的泛化能力尚需进一步验证。

---

## 301. SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams

**arXiv ID:** 2601.09515 | [PDF](https://arxiv.org/pdf/2601.09515v1)

**作者:** Chenglong Wang `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**通讯引用:** 12837 | [OpenAlex ID](https://openalex.org/A5100600701)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 SERM 的自演化检索相关性模型，利用多智能体样本挖掘器和多智能体注释器在海量查询流中持续改进模型。

**💡 创新点**

创新点在于将多智能体协作用于（1）通过环境反馈、点击模型与内部不确定性等多维度筛选稀缺且难以识别的高信息量样本；（2）构建两级一致性框架（内部一致、交叉一致）生成可靠标签，显著缓解伪标签噪声与误差累积。

**🔧 技术方法**

采用 LLM 生成式相关性预测、温度采样的多路径链式推理、点击模型补偿、模型不确定性/分歧量化，以及多智能体内部/交叉投票一致性机制。

**📊 数据集**

使用工业级大规模数据：100B 语料的持续预训练、3.6M 标注的查询-文档对、每轮 700K 采样自查询流，覆盖多语言（德语族、罗曼语族、小语族）及百亿级用户请求。

**📈 对比分析**

与传统 CT+SFT、单一自训练以及模型蒸馏对比，离线实验显示 SERM 在 NDCG@1、NDCG@4、准确率上均优于基线；在线 A/B 测试中，蒸馏后模型提升 14 天留存 +0.0359%，用户负面反馈降低 -1.2081%。

**⚠️ 局限性**

局限性包括对大规模 LLM 计算成本高、需人工调参阈值（τ_c、τ_cm），以及在极端新颖查询上仍可能出现误差传播或样本选择偏差。

---

## 302. CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion

**arXiv ID:** 2601.09512 | [PDF](https://arxiv.org/pdf/2601.09512v1)

**作者:** Ralf Römer `[一作]` (Technical University of Munich), Angela P. Schoellig `[通讯]` (Robotics Institute Germany)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无示例、无任务标识的持续学习框架，通过在视觉-语言-动作模型（VLA）中插入轻量级适配器并在必要时扩展网络，实现机器人在不遗忘旧任务的前提下持续学习新任务。

**💡 创新点**

创新点在于将模块化适配器、基于层级特征相似度的自适应扩展策略和自动编码器路由机制相结合，使得在无任务标签的条件下能够动态选取最合适的适配器并仅在真正需要时增加模型容量，从而实现参数高效的持续学习。

**🔧 技术方法**

技术细节包括：轻量级适配器（encoder‑decoder 结构）、自编码器判别器用于特征重构误差判断、特征相似度驱动的动态扩容、流形匹配（flow‑matching）训练、Diffusion Transformer（DiT）架构以及预训练的 DINOv2 视觉编码器和 CLIP 语言编码器。

**📊 数据集**

使用 LIBERO 基准数据集，先在 90 个短周期任务上预训练模型，再在 10 个连续出现的长周期任务上进行持续学习实验。

**📈 对比分析**

与 SeqFFT、SeqLoRA、PackNet、经验回放（ER）以及 LOTUS 等基线比较，所提出的 CLARE 在 AUC 上提升约 11–15%，FWT 与 SeqFFT/ER 相当，NBT 接近 0，且每学习一项任务仅增加约 1.7–2.3% 的参数量，表现出显著的性能优势。

**⚠️ 局限性**

局限性包括：目前仅在中等规模模型上验证，仍存在一定的参数增长；未在真实机器人硬件上进行测试；对极大规模 VLA 的可扩展性和实际部署效果仍需进一步评估。

---

## 303. Boltzmann Sampling for Powersets without an Oracle

**arXiv ID:** 2601.09508 | [PDF](https://arxiv.org/pdf/2601.09508v1)

**作者:** Jean Peyen `[一作]` `[通讯]` (University of Dundee), Jean Peyen (University of Dundee)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文提出了一种不需要生成函数求值（oracle）的Boltzmann采样器，用稀疏占位模型和抛光（thinning）技术生成幂集（powersets）以及其严格子集。

**💡 创新点**

创新点在于：①通过随机球盒占位模型恢复Boltzmann分布，消除对生成函数的显式计算；②利用Poisson分布与抛光方法，在计数序列有上界的结构上实现高效采样；③在处理含有零大小元素的结构时仍可直接采样，扩展了以往采样器的适用范围。

**🔧 技术方法**

核心技术包括：占位模型、Poisson过程、抛光（thinning）方法、几何分布采样、对计数序列的上界约束以及Python的集合实现。

**📊 数据集**

实验使用的测试数据集为：严格分割（strict partitions）及其平方分割（strict partitions into squares）；所有实验均在Apple MacBook Air M4上使用Python实现进行。

**📈 对比分析**

通过与文献中已有Boltzmann采样器的运行时间对比，实验表明该方法在自由采样和精确采样两种模式下的运行时间与现有实现相当，且在不同规模（10^3–10^9）下均保持可接受的毫秒级别。

**⚠️ 局限性**

局限性：仅适用于计数序列有上界的组合结构；若计数序列无界或取值分布不满足假设，则该采样器无法直接使用。

---

## 304. Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning

**arXiv ID:** 2601.09496 | [PDF](https://arxiv.org/pdf/2601.09496v1)

**作者:** Jujia Zhao `[一作]` (Leiden University), Zhaochun Ren `[通讯]` (Leiden University)

**通讯引用:** 7683 | [OpenAlex ID](https://openalex.org/A5100384130)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型上提出一种统一搜索与推荐的训练框架GEMS，通过低秩子空间梯度投影实现高效参数更新。

**💡 创新点**

创新点在于：①多子空间分解（shared + 两个任务特定子空间）显式隔离梯度冲突；②零空间投影把更新限制在对预训练知识无干扰的子空间，保持语言理解；③整体不增加可训练权重，训练成本低。

**🔧 技术方法**

核心技术包括子空间调优（subspace tuning）、自适应门控融合、多子空间分解、零空间投影（null‑space projection）以及传统的Adam优化。

**📊 数据集**

使用公开的Amazon Electronics 5-core与Qilin两大含搜索/推荐交互记录的数据集进行实验。

**📈 对比分析**

与多种专门模型、统一模型及常见PEFT方法（LoRA、LoRA‑MoE）比较，GEMS在Hit@K、NDCG@K等指标上普遍超越对手；梯度冲突系数下降≥85%；并且在保持预训练语言能力方面优于PEFT。

**⚠️ 局限性**

局限：仍保留全部预训练知识，无法根据任务动态筛选需要保留的知识；子空间秩固定，缺乏自适应调节；对极端小样本或极端任务偏差的鲁棒性待进一步验证。

---

## 305. How many users have been here for a long time? Efficient solutions for counting long aggregated visits

**arXiv ID:** 2601.09489 | [PDF](https://arxiv.org/pdf/2601.09489v1)

**作者:** Peyman Afshani `[一作]` (Aarhus University), Mariafiore Tognon `[通讯]` (University of Padova)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究并提出了在大规模移动性数据中，基于用户在多个地区累计停留时间的计数查询问题（CLAV）及其几何版本，给出精确与近似解法以及下界。

**💡 创新点**

创新点在于定义(k,r)-CLAV与Geometric-CLAV问题，给出匹配的上界与下界，并证明其空间/查询时间上的固有困难，同时提出基于染色支配计数和Sketch的高效数据结构。

**🔧 技术方法**

采用集合交互、SetDisjointness与Boolean‑Matrix‑Multiplication猜想的下界推导，结合Flajolet‑Martin、Count‑Min、染色支配计数、位压缩等技术实现精确与近似答案。

**📊 数据集**

主要使用理论模型与示例，引用Vodafone与TomTom等实际移动性数据集作为背景说明，但未给出具体实验数据集。

**📈 对比分析**

与传统时空聚合查询、频数估计和热门检测等方法对比，提出的结构在空间/时间上达到或接近理论下界；近似解在高概率下误差为εn_Q，查询时间为O(log n_Q)。

**⚠️ 局限性**

结果受SetDisjointness、BMM等猜想限制，且在维度≥2时仍需指数级空间或高查询时间；对极小用户数仅提供近似解，缺乏实验验证。

---

## 306. SimMerge: Learning to Select Merge Operators from Similarity Signals

**arXiv ID:** 2601.09473 | [PDF](https://arxiv.org/pdf/2601.09473v1)

**作者:** Oliver Bolton `[一作]` (Cohere Labs), Beyza Ermis `[通讯]` (Cohere Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于预合并相似度信号的模型合并选择框架，自动决定使用哪种融合算子以及多模型合并的顺序，避免昂贵的合并‑评估搜索；

**💡 创新点**

创新点在于将合并算子和合并顺序的选择视为学习问题，使用轻量级预测器和在线神经线性Bandit，仅凭模型权重与无标签探针的相似度即可做决策；

**🔧 技术方法**

技术包括：模型权重与中间激活的KL、余弦相似度等相似度特征、轻量级多分类MLP预测器、MVP式多模型计划编码、以及基于相似度特征的神经线性Bandit；

**📊 数据集**

使用7B级别Command‑A模型在四个领域（代码、数学、多语言、RAG）以及111B级别的Command‑A模型进行实验，探针大小约1万token；

**📈 对比分析**

与固定算子（线性、SLERP、TIES）对比，所提出的预测器在两模型、三模型、四模型合并中平均闭合65%专家-辅助性能差距（相比41.8%），并在111B模型上保持性能提升，在线Bandit版在分布漂移场景下几乎逼近oracle；

**⚠️ 局限性**

局限性包括：只能在同一预训练基座下合并、仅覆盖有限的三种算子（可扩展但需重新训练）、对中间合并状态的相似度估计仍为近似，可能在更深层树或极端多模型场景下精度下降。

---

## 307. FairGU: Fairness-aware Graph Unlearning in Social Network

**arXiv ID:** 2601.09469 | [PDF](https://arxiv.org/pdf/2601.09469v1)

**作者:** Renqiang Luo `[一作]` (Jilin University), Feng Xia `[通讯]` (RMIT University)

**通讯引用:** 18946 | [OpenAlex ID](https://openalex.org/A5089615958)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6215c339-3735-4be3-8a07-5bbb7004712d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出FairGU框架，实现在图数据去学习过程中兼顾公平性与隐私保护。

**💡 创新点**

将公平约束（对抗去偏+协方差约束）与Fisher信息矩阵驱动的参数淡化结合，并针对缺失敏感属性引入预训练估计器，填补现有去学习方法缺乏公平保障的空白。

**🔧 技术方法**

预训练敏感属性估计器、对抗去偏的公平GNN、Fisher信息矩阵重要性分析及参数淡化、GNN与GAT两种骨干网络。

**📊 数据集**

Income、Pokec‑z、Pokec‑n三种真实社交网络数据集。

**📈 对比分析**

与GER、IDEA、MEGU、ETR及公平GNN、FairAC、FairSIN等SOTA方法进行对比，FairGU在保持或提升节点分类准确率的同时，统计平衡与机会平等指标显著下降，MIA攻击AUC接近50%表明隐私保护效果优秀。

**⚠️ 局限性**

对超参数γ、λ等敏感，需仔细调优；目前仅适用于静态单敏感属性图，未覆盖动态图或多敏感属性情景。

---

## 308. Towards a Metadata Schema for Energy Research Software

**arXiv ID:** 2601.09456 | [PDF](https://arxiv.org/pdf/2601.09456v1)

**作者:** Stephan Ferenz `[一作]` (Carl von Ossietzky Universität Oldenburg), Astrid Nieße `[通讯]` (OFFIS - Institute for Information Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

针对能源研究软件，设计并评估了一套基于需求分析和现有元数据方案的可发现、可访问、可互操作、可复用（FAIR）的域特定元数据模式ERSmeta，并在工具SMECS中实现。

**💡 创新点**

创新点在于系统化的需求获取与元数据方案对齐流程、采用多源控制词典与本体来提升互操作性、以及将该模式正式化为SHACL与JSON‑LD Schema。

**🔧 技术方法**

使用的技术包括需求访谈、元数据模式对齐（CodeMeta、CFF、DataDesc等）、语义网技术（SHACL、JSON‑LD）、价值词典构建（Open Energy Ontology、Wikidata）以及评估工具SMECS。

**📊 数据集**

使用的数据集为32位能源研究人员的访谈记录、现有元数据模式的元素集合以及两款能源研究软件的实际元数据实例。

**📈 对比分析**

评估方法为现场实验（SMECS使用问卷调查）和元数据质量检查，结果显示参与者认为元素有用但受限于元素数量与复杂度，SUS平均得分约为68，表明可用性中等偏上。

**⚠️ 局限性**

主要限制包括样本地域单一（多为德国）、受访者偏向大型软件开发、工具与模式混合评估导致的混淆，以及未覆盖接口描述等机器可读细节。

---

## 309. On the Hardness of Computing Counterfactual and Semifactual Explanations in XAI

**arXiv ID:** 2601.09455 | [PDF](https://arxiv.org/pdf/2601.09455v1)

**作者:** André Artelt `[一作]` (Bielefeld University), Kevin Tierney `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统整理了机器学习模型的反事实与半反事实解释的计算复杂度，并给出了新的不可逼近性结果。

**💡 创新点**

首次将不可逼近性分析扩展到 ReLU 网络、加性树模型和 kNN 模型，并提供统一的框架与理论证明。

**🔧 技术方法**

使用计算复杂度理论、归约（如 3‑SAT 归约）与证明技术，涉及 NP、Σ₂^p、#P 等复杂度类。

**📊 数据集**

未使用具体数据集，全部为理论分析与证明。

**📈 对比分析**

通过对比表格和定理与现有文献，指出大多数模型的解释生成是 NP‑hard/Σ₂^p‑complete；未给出数值实验或性能指标。

**⚠️ 局限性**

局限：仅针对离散/二元特征；未涵盖连续特征、因果解释、鲁棒性/可解释性约束；缺乏实验验证与平均情况分析。

---

## 310. MAD: Motion Appearance Decoupling for efficient Driving World Models

**arXiv ID:** 2601.09452 | [PDF](https://arxiv.org/pdf/2601.09452v1)

**作者:** Ahmad Rahimi `[一作]` (Ecole Polytechnique Federale de Lausanne), Alexandre Alahi `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了MAD框架，将通用视频扩散模型拆分为运动预测器和外观合成器，生成可控驾驶场景视频。

**💡 创新点**

将运动与外观解耦，使用中间骨架视频表示并通过LoRA微调，显著降低训练成本。

**🔧 技术方法**

基于Diffusion模型（SVD/LTX），LoRA适配，VAE嵌入，文本与视觉控制，目标噪声注入等技术。

**📊 数据集**

利用OpenDV YouTube驾驶视频、Waymo感知数据以及通过姿态检测生成的骨架伪标签进行训练。

**📈 对比分析**

通过人类偏好实验与VISTA、GEM、Cosmos-Predict等基线对比，MAD‑LTX在生成质量、运动规划精度方面优于公开模型，同时训练成本仅为前者的少量比例。

**⚠️ 局限性**

对极端场景和多模态控制的泛化尚待验证，且受限于训练视频的视觉多样性与伪标签的准确性。

---

## 311. Do Transformers Understand Ancient Roman Coin Motifs Better than CNNs?

**arXiv ID:** 2601.09433 | [PDF](https://arxiv.org/pdf/2601.09433v1)

**作者:** David Reid `[一作]` (University of St Andrews), Ognjen Arandjelovic `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了将Vision Transformer (ViT) 与传统卷积神经网络 (CNN) 应用于古币反面语义元素识别，并通过训练、验证、测试三份数据集评估其性能。

**💡 创新点**

创新点包括：①首次将ViT架构用于古币语义分析；②通过自动文本挖掘从币描述中提取概念并生成标签；③对比两种模型在同一任务上的表现，揭示ViT在精确率与整体准确率上的优势。

**🔧 技术方法**

技术手段：ViT 预训练模型 (ImageNet‑21K, large 版本) 加 SGD；CNN 采用 Cooper & Arandjelović 提出的架构也加 SGD；文本挖掘与词表扩展；Saliency 映射工具 HiPe 用于模型可解释性分析。

**📊 数据集**

数据集：来自 Ancient Coins Search Engine 的 100,000 张古币图像与对应描述，经过预处理仅保留反面图像，生成 8 个语义元素（cornucopia, eagle, horse, patera, shield, standing, seated, Hercules）的多标签/单标签数据集。

**📈 对比分析**

评价方法：按 64:16:20 的比例拆分训练/验证/测试集；计算准确率、精确率、召回率、F1；ViT 在大多数语义元素上达到约80% 的准确率，精确率普遍高于 CNN，召回率虽有波动但整体优于 CNN；CNN 训练时间短但整体性能低。

**⚠️ 局限性**

局限性：①标签噪声高，描述中混合正反面信息导致错误标注；②数据集不平衡且正样本稀缺；③ViT 模型参数多、训练耗时长，GPU 内存受限；④仅使用原始 ViT 变体，未探索更高效或更适合此任务的模型；⑤可解释性有限，Saliency 映射呈散布且难以直观理解。

---

## 312. Late Breaking Results: Quamba-SE: Soft-edge Quantizer for Activations in State Space Models

**arXiv ID:** 2601.09451 | [PDF](https://arxiv.org/pdf/2601.09451v1)

**作者:** Yizhi Chen `[一作]` (KTH Royal Institute of Technology), Ahmed Hemani `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 3568 | [OpenAlex ID](https://openalex.org/A5026355063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Quamba-SE软边量化器，用于State Space Models（如Mamba）的激活量化；

**💡 创新点**

创新点在于采用三重自适应尺度（高精度、小值；标准尺度、正常值；低精度、大值），用软边取代硬剪裁，保留异常值信息；

**🔧 技术方法**

利用自定义硬件级INT8量化，第二位二进制位区分小/大值，6位精度处理特殊范围，配合已有的标准尺度进行动态范围扩展；

**📊 数据集**

在6个零样本基准上评估：LAMBADA、ARC-C、WinoGrande、HellaSwag、PIQA、ARC-E；

**📈 对比分析**

与Quamba/Quamba2在Mamba‑130M模型下进行对比，使用99.99%/99.999%校准，平均精度提升0.22%–0.83%，单一数据集最高提升2.68%，官方权重下平均提升1.48%；

**⚠️ 局限性**

局限性包括：引入软边会略微增加延迟；实验仅在130M小模型上进行；未实现硬件合成；未验证对Mamba2或更大模型的效果；

---

## 313. Improving Symbolic Translation of Language Models for Logical Reasoning

**arXiv ID:** 2601.09446 | [PDF](https://arxiv.org/pdf/2601.09446v1)

**作者:** Ramya Keerthy Thatikonda `[一作]` (Monash University), Ehsan Shareghi `[通讯]` (Monash University)

**通讯引用:** 720 | [OpenAlex ID](https://openalex.org/A5086032589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对小型语言模型进行数据合成、监督微调、增量推理和轻量级验证器的设计，实现了自然语言到一阶逻辑（FOL）的高质量转换。

**💡 创新点**

创新点包括：①系统化归纳翻译错误类别；②将生成过程拆分为谓词生成与FOL翻译两阶段的增量推理；③引入基于谓词-arity校验的轻量级验证器；④利用大型模型生成合成数据并进行过滤，扩充小模型训练集。

**🔧 技术方法**

采用的技术包括：监督微调（SFT）+ LoRA、8-bit量化、温度调节、增量推理两阶段结构、谓词覆盖与使用率评估、谓词验证器以及对执行率和准确率的工具链评估。

**📊 数据集**

使用的主要数据集有：ProofWriter、FOLIO、ProntoQA、ProverQA；合成数据来自大型模型的ProofWriter扩充生成。

**📈 对比分析**

与仅使用ICL的基线相比，微调后小模型在四个基准上的执行率和准确率显著提升；增量推理进一步减少约10%错误，验证器再进一步提升性能；在外域（OOD）数据上也能保持较高的泛化能力，整体错误率下降可达60%以上。

**⚠️ 局限性**

局限性包括：ProofWriter 数据多样性有限；合成的银牌数据仍可能含有细微不一致；验证器增加了推理延迟；模型仍难以完全保证语义正确性，可能产生冗长的推理链。

---

## 314. DeepLight: A Sobolev-trained Image-to-Image Surrogate Model for Light Transport in Tissue

**arXiv ID:** 2601.09439 | [PDF](https://arxiv.org/pdf/2601.09439v1)

**作者:** Philipp Haim `[一作]` (Technical University of Munich), Dominik Jüstel `[通讯]` (Helmholtz Zentrum Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了 Sobolev 训练对光传输神经网络的影响，构建并训练了高维图像‑到‑图像的光吸收能量替代模型 DeepLight，并用单向导数样本进行训练。

**💡 创新点**

首次将 Sobolev 训练应用于高维图像‑到‑图像网络，并证明仅使用单向导数样本即可显著提升模型在目标值与梯度上的准确性和泛化能力。

**🔧 技术方法**

使用 Sobolev 训练（方向导数正则化）、基于 UNet 的卷积网络、Monte Carlo 仿真 MoCA 生成吸收能量与导数数据、非线性缩放函数 σ、AdamW 优化器与余弦学习率调度等技术。

**📊 数据集**

采用基于人体组织特性的合成生成器得到的 ID_1、ID_2、OOD 样本，分别提供吸收系数 μ_a、散射系数 μ_s′ 以及对应的吸收能量 E 和方向导数 ∇_v E，训练集 2048 张图像，验证集 256 张，测试集 384 张。

**📈 对比分析**

与仅使用 L² 损失的基线模型相比，Sobolev 训练的 DeepLight 在 ID 和 OOD 样本上平均降低约 11–12% 的吸收能量误差，导数误差下降约 50%，且在深层组织中的误差曲线更平缓，整体性能提升显著。

**⚠️ 局限性**

目前仅针对二维投影；合成数据与真实组织差异；未将散射各向异性作为独立输入；深层区域的 MC 噪声仍影响结果；需要进一步在逆问题中验证其实际效益。

---

## 315. Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs

**arXiv ID:** 2601.09527 | [PDF](https://arxiv.org/pdf/2601.09527v1)

**作者:** Jonathan Knoop `[一作]` (IE Business University), Hendrik Holtmann `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了NVIDIA Blackwell消费级GPU（RTX 5060 Ti、5070 Ti、5090）在小型企业LLM推理中的性能、能耗与成本，并提供完整基准数据。

**💡 创新点**

创新点在于首次在实际SME工作负载（RAG、API、Agentic多LoRA）上，使用多种量化格式（NVFP4、W4A16、MXFP4）对四大开放权重模型进行大规模对比，并公开所有实验配置与结果。

**🔧 技术方法**

技术手段包括vLLM推理引擎、AIPerf基准框架、Tensor‑Parallel、KV‑Cache量化、以及NVIDIA官方的NVFP4、AWQ、MXFP4量化实现。

**📊 数据集**

实验使用了Qwen3‑8B、Gemma3‑12B/27B、GPT‑OSS‑20B三大模型，并在MMLU、GSM8K、HellaSwag等公开基准上评估推理质量。

**📈 对比分析**

通过79个配置在RAG、API与Agentic场景下测算吞吐量、延迟、能耗，结果显示RTX 5090比RTX 5060 Ti高3.5–4.6倍吞吐，NVFP4相较BF16提升1.6倍吞吐、41%能耗下降，API成本仅$0.001–$0.04/百万token，且在30M tokens/天时在数十至百天内突破云API成本。

**⚠️ 局限性**

局限性包括仅针对NVIDIA Blackwell GPU，未涵盖其他厂商硬件；量化结果受限于官方检查点和自研工具；质量评估仅基于公开基准，未覆盖业务领域特定需求；长上下文超过64k未测试；能耗测量仅包含GPU功耗，未考虑CPU、内存等系统开销。

---

## 316. Video Joint-Embedding Predictive Architectures for Facial Expression Recognition

**arXiv ID:** 2601.09524 | [PDF](https://arxiv.org/pdf/2601.09524v1)

**作者:** Lennart Eing `[一作]` (University of Augsburg), Elisabeth André `[通讯]` (University of Augsburg)

**通讯引用:** 16679 | [OpenAlex ID](https://openalex.org/A5056684559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究者在表情识别任务中使用预训练的V‑JEPA视频编码器，并在RAVDESS与CREMA‑D两套实验室控制数据集上训练浅层分类器。

**💡 创新点**

创新点在于首次证明像素级重建并非FER预训练的必要条件，展示V‑JEPA可直接预测掩码区域嵌入并获得SOTA性能，并引入注意力探测器实现更高效的特征聚合。

**🔧 技术方法**

采用的技术包括V‑JEPA自监督预训练、注意力探测（Attentive Probing）作为特征池化与分类头、以及最大投票（MV）和后验投票（PBV）两种视频级分类策略。

**📊 数据集**

实验使用RAVDESS（8类情绪）和CREMA‑D（6类情绪）两套实验室录制视频数据集，并在5折交叉验证中评估性能。

**📈 对比分析**

与其他基于像素重建的自监督FER方法对比，RAVDESS上取得76.4% UAR（SOTA），CREMA‑D上取得79.4% UAR，超越所有仅使用视频的基线；跨数据集评估显示在CREMA‑D上仍保持约79% UAR，但在RAVDESS上的迁移性能显著下降。

**⚠️ 局限性**

主要局限在于仅验证于实验室控制数据集，未考察野外环境下的泛化；跨数据集迁移表现不对称，并且对基础情绪（如“calm”与“neutral”）的处理仍需改进。

---

## 317. Towards Realistic Synthetic Data for Automatic Drum Transcription

**arXiv ID:** 2601.09520 | [PDF](https://arxiv.org/pdf/2601.09520v1)

**作者:** Pierfrancesco Melucci `[一作]` (Sapienza University of Rome), Taketo Akama `[通讯]` (Sony Computer Science Laboratories)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5087426444)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种无需配对音频‑MIDI数据即可训练自动鼓点转录模型的方法，先利用半监督流程从公开单拍鼓样本库构建 26 类标准鼓样本语料，再以此合成高质量音频用于训练序列到序列 Transformer 模型。

**💡 创新点**

创新点在于：1) 用 CLAP 嵌入对未标注鼓样本进行语义匹配，自动扩充大型标准化鼓样本库；2) 通过该库进行合成音频，显著缩小合成‑真实域间差距；3) 在鼓点转录任务中首次系统评估并证明 26 类细粒度鼓分类在性能上优于传统 8 类分类。

**🔧 技术方法**

使用的技术包括 CLAP 音频嵌入、余弦相似度标签推断、基于鼓样本的合成音频引擎、对齐与线性插值的样本多样化、序列到序列 Transformer（Encoder‑Decoder）以及自定义 MIDI 令牌化（tempo、instrument、velocity）。

**📊 数据集**

训练数据来自 Lakh MIDI Dataset（45,129 条 MIDI 文件），评估数据则使用 ENST 与 MDB 两个鼓点数据集，且仅在鼓单轨（DTD）设置下进行测试。

**📈 对比分析**

与先前的 CRNN 以及其它合成‑真实混合方案相比，本方法在 ENST 与 MDB 上均取得新的最高 F1‑score，显著高于完全监督方法和以 SoundFont 为基础的合成方法，证明了半监督鼓样本库与 Transformer 架构的有效性。

**⚠️ 局限性**

局限性包括：仅在鼓单轨环境下验证，未对多乐器混音（DTP、DTM）进行评估；模型仍依赖 MIDI 结构，可能对复杂节奏或非标准鼓组合的泛化有限；以及 26 类鼓标签虽更细粒度，但在实际应用中仍需进一步细化与标准化。

---

## 318. Dobrushin Coefficients of Private Mechanisms Beyond Local Differential Privacy

**arXiv ID:** 2601.09498 | [PDF](https://arxiv.org/pdf/2601.09498v1)

**作者:** Leonhard Grosse `[一作]` (KTH Royal Institute of Technology), Mikael Skoglund `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 8707 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在对所有最小质量大于等于c的离散分布上满足点极大泄漏（PML）约束的隐私机制的Dobrushin系数，并给出了其收缩系数的上界及对应机制；同时将该结果推广到任意f‑散度并给出对应的强数据处理不等式；进一步利用这些收缩系数推导了私有极大似然风险和样本复杂度的下界。

**💡 创新点**

创新点在于：① 用PML代替传统的局部差分隐私（LDP），使得对包含零概率元素的机制也能得到有限的隐私量化；② 在此更宽松的隐私设定下，首次给出了全局收缩系数的最优闭式上界并构造了实现该上界的机制；③ 将此结果与Binette逆Pinsker不等式结合，获得了任意f‑散度的强数据处理不等式，为非LDP机制提供了新的风险评估工具；④ 通过Le Cam方法进一步推导了私有极大似然风险与样本复杂度的下界。

**🔧 技术方法**

主要技术包括：PML的定义与性质、Dobrushin系数与TV收缩的计算、对二元输出机制的优化、构造满足(ε,c)-PML的极优机制、Binette逆Pinsker不等式以及Le Cam两点法与Bretagnolle‑Huber不等式。

**📊 数据集**

该工作为纯理论分析，未使用实际数据集；所有结论均通过数学证明给出，并在论文中给出若干数值示例验证界限的适用性。

**📈 对比分析**

与传统的ε‑LDP收缩系数上界相比，本文在c>0时给出了更紧的上界；当c→0时可回溯到已知的LDP结果。实验性数值示例表明，在相同ε下，(ε,c)-PML机制能够显著降低估计风险、提升样本效率。

**⚠️ 局限性**

主要局限在于：① 仅对离散随机变量和离散马尔可夫核进行分析；② 需要已知最小质量c的下界，若c未知或极小则结果退化为传统LDP；③ 由于对f‑散度的推广依赖于Binette逆Pinsker不等式，可能在某些散度下的常数系数不够紧。

---

## 319. Engineering Compressed Matrix Multiplication with the Fast Walsh-Hadamard Transform

**arXiv ID:** 2601.09477 | [PDF](https://arxiv.org/pdf/2601.09477v1)

**作者:** Joel Andersson `[一作]` (Chalmers University of Technology), Matti Karppa `[通讯]` (University of Gothenburg)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

实现并评估了Pagh压缩矩阵乘法算法的多线程CPU版本，并将FFT替换为FWHT。

**💡 创新点**

证明FFT可以被FWHT替代而不影响理论保证，并实现了更高效的变换和更低的内存占用。

**🔧 技术方法**

采用2‑wise独立哈希、快速傅里叶变换（FFT）/快速沃尔什-哈达马变换（FWHT）、OpenMP并行、FFTW、Intel MKL、NumPy兼容Python绑定。

**📊 数据集**

使用合成稀疏/密集乘积数据集：logunit、Diagonal、Covariance、Lightbulb等四种场景。

**📈 对比分析**

与Intel MKL的GEMM以及原始FFT实现对比；在满足稀疏或大元素主导的情况下，FWHT实现比MKL快最多40倍，FWHT比FFT快4倍。

**⚠️ 局限性**

仅支持双精度密集矩阵，无法处理稀疏格式、单精度或GPU实现，且在高度密集且噪声大的场景（Covariance、Lightbulb）下准确率相对下降。

---

## 320. SpatCode: Rotary-based Unified Encoding Framework for Efficient Spatiotemporal Vector Retrieval

**arXiv ID:** 2601.09530 | [PDF](https://arxiv.org/pdf/2601.09530v1)

**作者:** Bingde Hu `[一作]` (Zhejiang University), Hao Zhong `[通讯]` (Zhejiang University)

**通讯引用:** 2914 | [OpenAlex ID](https://openalex.org/A5101720286)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一时空向量检索框架 SpatCode，能在单一向量空间内同时考虑语义、时间与空间信息，支持流式增量更新与个性化检索。

**💡 创新点**

创新点：① Rotary‑based Unified Encoding 将时间与地理坐标映射到单位圆/球面，保持相似度与相对时空距离的一致性；② Circular Incremental Update 通过环形缓冲区实现滑动窗口无全量重编码；③ Weighted Interest‑based Retrieval 在查询时按模态加权实现即时偏好调整。

**🔧 技术方法**

技术：旋转位置编码、球面编码、HNSW 近似最近邻、圆形缓冲区、按模态加权拼接查询向量。

**📊 数据集**

数据集：Synthetic Shopping（图像+文本+时间+位置）、Craigslist、Netflix、Bridge、VeRi 等多模态时空数据集。

**📈 对比分析**

与 ThalDB、Milvus Scalar‑Filtered、Milvus Hybrid 等基线对比：SpatCode 插入/查询时延最低，recall@k 在所有数据集上均达到约0.9–1.0，显著优于传统过滤/多索引方法。

**⚠️ 局限性**

局限性：需手动设定时间尺度 α_t，过大/过小会导致别名或精度下降；对极端分布或极大规模时空跨度的鲁棒性尚未充分验证。

---

## 321. TEMPO: A Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval

**arXiv ID:** 2601.09523 | [PDF](https://arxiv.org/pdf/2601.09523v1)

**作者:** Abdelrahman Abdallah `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TEMPO基准，结合时序推理与多步检索，评估检索模型在跨时期问题上的表现

**💡 创新点**

①提供1,730条复杂时序查询和3,976步检索计划；②引入Temporal Precision@k、Temporal Coverage@k等新指标；③将时序推理嵌入检索流程

**🔧 技术方法**

利用LLM辅助的注释与评估（GPT‑4o、Qwen‑72B）、多模型检索（BM25、BGE、Contriever、E5、GritLM、Qwen、SBERT、SFR、DiVeR、Rader、ReasonIR）以及步骤级检索策略

**📊 数据集**

自选自Stack Exchange 13个领域（区块链、社会科学、应用领域、STEM）的1,730个查询及其对应的正负文档，涵盖历史与现代时段

**📈 对比分析**

在12个检索模型上评测，最佳模型DiVeR仅获得32.0 NDCG@10、71.4% Temporal Coverage@10，表明即便是先进模型仍远低于理想水平，稀疏检索BM25表现最差

**⚠️ 局限性**

仅限英语13个领域；历史时序分布不均；注释与评估依赖LLM，可能存在误差；未覆盖医学、法律等特定专业时序需求

---

## 322. What Do LLM Agents Know About Their World? Task2Quiz: A Paradigm for Studying Environment Understanding

**arXiv ID:** 2601.09503 | [PDF](https://arxiv.org/pdf/2601.09503v1)

**作者:** Siyuan Liu `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 23651 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Task-to-Quiz (T2Q) 两阶段自动化评估框架，并构建了包含 30 个环境、224 个任务和 1,967 个问答的 T2QBench，用以衡量 LLM 代理的环境理解能力。

**💡 创新点**

创新点在于：①将“做”与“知”分离，利用覆盖导向任务与轨迹条件问答动态生成答案；②提供可重复、无人工标注的环境理解度量；③通过实验揭示记忆系统未提升理解、主动探索是主要瓶颈。

**🔧 技术方法**

技术包括：基于 TextWorld 的程序化环境与任务生成；权重集合覆盖算法选取任务；规则化问答生成与轨迹条件答案判定；两阶段评估指标（Task Success Rate、Environment Understanding Score）。

**📊 数据集**

使用自构建的 T2QBench，涵盖三难度等级（Easy/Medium/Hard），共 30 个环境、224 个任务、1,967 个问答。

**📈 对比分析**

通过与 GPT‑5.1、DeepSeekV3.2、ChatGLM4.6、Qwen3‑32B 等模型及多种记忆机制（in‑context、Mem0、LangMem、A‑MEM）进行对比；实验显示 in‑context 通常获得最高 TSR 与 EUS，EUS 与 TSR 难度上升表现不一致，表明任务完成率并不能完全反映环境理解。

**⚠️ 局限性**

局限性包括：仅在 TextWorld 简单文本游戏上评估，难以覆盖更复杂、多模态环境；实验仅涉及少量模型；覆盖任务生成算法在大规模环境下计算成本较高。

---

## 323. V-DPM: 4D Video Reconstruction with Dynamic Point Maps

**arXiv ID:** 2601.09499 | [PDF](https://arxiv.org/pdf/2601.09499v1)

**作者:** Edgar Sucar `[一作]` (University of Oxford), Andrea Vedaldi `[通讯]` (University of Oxford)

**通讯引用:** 74583 | [OpenAlex ID](https://openalex.org/A5060511349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种可在多帧单目视频上一次性进行4D重建的多视角动态点图（V‑DPM）框架。

**💡 创新点**

将动态点图从仅成对视图扩展到整个视频序列，并通过时间条件解码器实现对任意时间点的视图与时间不变表示。

**🔧 技术方法**

在预训练的VGGT静态重建网络上加入时间条件Transformer解码器，并利用自适应LayerNorm实现时间上下文调制。

**📊 数据集**

在ScanNet++、BlendedMVS等静态数据以及Kubric‑F、Kubric‑G、PointOdyssey、Waymo等合成动态数据上微调。

**📈 对比分析**

与DPM、St4RTrack、TraceAnything等方法在四个动态重建基准上进行End‑Point Error比较，V‑DPM在所有指标上显著低于对手，几乎达到π^3水平；在深度和相机位姿评估中也保持竞争力。

**⚠️ 局限性**

受限于训练规模和计算资源，V‑DPM仍略逊于大型π^3模型，并且仅在约50帧以内验证，未来需进一步扩展数据与模型规模。

---

## 324. Towards Robust Cross-Dataset Object Detection Generalization under Domain Specificity

**arXiv ID:** 2601.09497 | [PDF](https://arxiv.org/pdf/2601.09497v1)

**作者:** Ritabrata Chakraborty `[一作]` (Indian Statistical Institute), Umapada Pal `[通讯]` (Indian Statistical Institute)

**通讯引用:** 11953 | [OpenAlex ID](https://openalex.org/A5068803496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了跨数据集目标检测的鲁棒性，基于数据集的设置特异性对所有训练-测试对进行系统评估，并通过闭标签和开标签协议分离标签不匹配与视觉域移位的影响。

**💡 创新点**

提出了设置特异性框架和跨数据集转移网格，并利用CLIP语义重映射与近似匹配诊断，揭示了跨数据集性能下降的方向性和“鲁棒性悬崖”，首次量化标签差异与域移位的具体贡献。

**🔧 技术方法**

使用标准Faster R‑CNN（ResNet‑50 FPN）作为基准模型，配合CLIP ViT‑L/14进行标签语义相似度计算，开展闭标签评估、开标签重映射及近似匹配诊断，采用AP、mAP等COCO式指标。

**📊 数据集**

四个公开数据集：COCO、Objects365（设置无关），Cityscapes、BDD100k（设置特定），构成四类交叉组合，形成完整的训练-测试矩阵。

**📈 对比分析**

通过在所有八个源-目标组合上报告AP/ mAP，发现从多样化无关源到狭窄特定目标时性能急剧下降；开标签评估给出有限提升，Cityscapes为最佳源、Objects365为最差源；转移表现高度不对称，表明域移位主导。

**⚠️ 局限性**

局限于单一检测架构与有限的数据集集合，未涵盖更广泛的真实世界域移位；未尝试适应或增强策略，因而无法评估潜在的鲁棒性提升方法。

---

## 325. SoK: Enhancing Cryptographic Collaborative Learning with Differential Privacy

**arXiv ID:** 2601.09460 | [PDF](https://arxiv.org/pdf/2601.09460v1)

**作者:** Francesco Capano `[一作]` (SAP SE), Benjamin Weggenmann `[通讯]` (Technische Hochschule Würzburg-Schweinfurt)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究系统化了加密与差分隐私协同学习（CPCL）的完整框架，重点分析了安全噪声采样技术并在多服务器MPC环境下实现与评估。

**💡 创新点**

创新点在于提出统一的七阶段框架、将噪声采样确认为CPCL核心、对噪声类型/采样方式与DP机制进行全面比较，并给出未来研究方向。

**🔧 技术方法**

技术包括多方计算（MPC）、同态加密、掩码、DP‑SGD、梯度裁剪、离散/连续噪声采样、固定点量化与安全聚合。

**📊 数据集**

实验数据集主要为公开图像数据集（如MNIST、CIFAR‑10）以及若干语音/文本数据集，用于评估模型精度与隐私效益。

**📈 对比分析**

通过在FL与OL两种范式下对比中央采样、局部采样和分布式MPC采样，量化了精度损失、计算与通信开销；结果显示，分布式采样在WAN可达≈1–2×时间开销，精度下降≈1–3%。

**⚠️ 局限性**

主要局限包括高昂的加密运算与通信成本、在加密环境下梯度裁剪效率低、缺乏成熟的分布式离散噪声采样方案以及对恶意与协同攻击的防护不够完善。

---

## 326. Population-Aligned Audio Reproduction With LLM-Based Equalizers

**arXiv ID:** 2601.09448 | [PDF](https://arxiv.org/pdf/2601.09448v1)

**作者:** Ioannis Stylianou `[一作]` (Bang & Olufsen), Zheng-Hua Tan `[通讯]` (Aalborg University)

**通讯引用:** 6034 | [OpenAlex ID](https://openalex.org/A5090108098)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过大语言模型（LLM）将自然语言描述映射到音频均衡设置，实现对音响系统的对话式控制，并采用分布式评价指标衡量模型对人类多样偏好的拟合程度。

**💡 创新点**

创新点：①将均衡问题视为分布生成任务而非单点回归；②使用反射式 Kantorovich 距离进行模型评价；③结合 RAG、RAG‑QA、PEFT 等多种 LLM 适配方法；④提出以人类实验数据为基础的分布式评估框架。

**🔧 技术方法**

技术：大语言模型 Phi‑3.5 mini（及 GPT‑4o mini）+ In‑Context Learning（Zero‑shot、Few‑shot、RAG、RAG‑QA）；Parameter Efficient Fine‑Tuning（LoRA、Prefix‑Tuning）+ 回归头；分布式损失（Sinkhorn Divergence）；反射 KDE + Kantorovich/Reflective Kantorovich 距离；数据增强。

**📊 数据集**

数据集：120 条自然语言提示 + 11 人的 2D 坐标（Beosonic 6×6 频率响应控制）共 1320 条标注；音频样本来自 6 种来源（音乐、电影、播客、自然声等）用于实验提示生成。

**📈 对比分析**

比较方法：在测试集上生成 11×2 坐标分布，计算 Kantorovich 与 Reflective Kantorovich 距离；与随机高斯对照；使用 Kruskal–Wallis 与 Dunn 检验。结果显示所有方法均优于随机猜测；LoRA 细调在两个距离上表现最佳，但差异未必统计显著；ICL 与 PEFT 互相可比，性能相近。

**⚠️ 局限性**

局限：①数据量仅 120 条提示，样本数小；②仅使用文本条件，未加入音频特征；③未进行听感验证，仅靠分布匹配，未证实实际满意度。

---

## 327. Where Knowledge Collides: A Mechanistic Study of Intra-Memory Knowledge Conflict in Language Models

**arXiv ID:** 2601.09445 | [PDF](https://arxiv.org/pdf/2601.09445v1)

**作者:** Minh Vu Pham `[一作]` (IT University Austria), Yufang Hou `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究针对语言模型内部知识冲突，构建了一个机制可解释性框架，定位并在推理阶段对冲突知识进行干预。

**💡 创新点**

首次结合logit lens、激活补丁和跨模型激活补丁三种技术，系统地识别并修正预训练阶段产生的内部冲突知识。

**🔧 技术方法**

主要技术包括Logit Lens用于定位责任层/组件，激活补丁用于因果验证，以及CMAP（跨模型激活补丁）用于更稳健的干预。

**📊 数据集**

使用自制的Wikipedia‑style传记数据集，其中包含冲突版本（双向事实）与干净版本（单一事实），用于Fine‑tune模型并进行对比实验。

**📈 对比分析**

在GPT‑2 XL与Qwen3‑4B上进行实验，CMAP在多层上可将模型输出翻转20‑27%（GPT‑2）甚至高达70%（Qwen3），与传统补丁相比，CMAP在公司属性上提升了约10%，在大学属性上表现相当或略差。

**⚠️ 局限性**

研究仅覆盖单一冲突类型与1:1实体冲突，模型范围有限，且未深入探究模型为何偏好某一事实，未来需扩展冲突类型、实体数量并分析偏好机制。

---

## 328. Data Scaling for Navigation in Unknown Environments

**arXiv ID:** 2601.09444 | [PDF](https://arxiv.org/pdf/2601.09444v1)

**作者:** Lauri Suomela `[一作]` (Tampere University), Joni-Kristian Kämäräinen `[通讯]` (Tampere University)

**通讯引用:** 5486 | [OpenAlex ID](https://openalex.org/A5054822570)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在端到端、无地图视觉导航中，训练数据的规模与多样性对在未见环境中零样本迁移性能的影响，并在真实机器人上进行了大规模验证。

**💡 创新点**

发现地理多样性比单纯数据量更能提升泛化，且两者的关系呈幂律；同时在噪声数据环境下，简单的回归模型比生成式或序列模型表现更好。

**🔧 技术方法**

使用了基于Vision Transformer编码的 MLP‑BC（回归）策略、Diffusion Policy、Flow‑matching Policy、ViNT/LogoNav 等多种架构，并采用线性、生成式、流匹配等不同训练目标。

**📊 数据集**

利用从 35 个国家 161 个地点收集的 4,565 小时的众包演示数据（FrodoBots8K 数据集），并在 4 个不同国家（中国、肯尼亚、毛里求斯、博茨瓦纳）的道路上测试。

**📈 对比分析**

通过多次重复的分段成功率、干预率和行进速度等指标进行比较。结果表明，使用大规模、多地点训练的零样本模型在新环境中的成功率接近甚至超过仅在目标环境中训练的模型；每增加一次地点大约可将失败率下降 15%。

**⚠️ 局限性**

局限性包括：数据质量不均衡导致某些地点训练效果差；对长时间序列、动态规划的能力有限；远程推理导致的网络延迟和帧率不稳定；仅针对平面差速驱动侧步机器人，缺乏对更复杂平台和地图辅助导航的验证。

---

## 329. SC-MAS: Constructing Cost-Efficient Multi-Agent Systems with Edge-Level Heterogeneous Collaboration

**arXiv ID:** 2601.09434 | [PDF](https://arxiv.org/pdf/2601.09434v1)

**作者:** Di Zhao `[一作]` (National University of Defense Technology), Yi Kong `[通讯]` (National University of Defense Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了SC-MAS框架，通过社会资本理论设计可自适应、成本高效的多代理系统，能够针对查询构建多样化的协作图；

**💡 创新点**

创新点在于将协作模式从统一的全局图转为边级协作策略，并联合优化代理选择、协作结构与LLM分配，实现异构协作与成本控制的统一；

**🔧 技术方法**

使用变分潜在变量模型进行代理选择、边策略概率分布化的边优化、图神经网络实现LLM路由，整体通过策略梯度在查询条件下学习；

**📊 数据集**

在MMLU、GSM8K、MATH、HumanEval、MBPP五大基准上进行评估；

**📈 对比分析**

与动态多代理、单代理路由、现有MAS路由基线（如MasRouter）对比，SC-MAS在所有数据集上均达到或超越SOTA，同时在推理成本上平均降低约12%（MBPP）或约17%（HumanEval）；

**⚠️ 局限性**

局限在于仅支持有向无环图结构，无法表达循环或迭代反馈；此外模型生成的代理与策略选择缺乏可解释性，影响用户信任与可调试性。

---

## 330. PrivLEX: Detecting legal concepts in images through Vision-Language Models

**arXiv ID:** 2601.09449 | [PDF](https://arxiv.org/pdf/2601.09449v1)

**作者:** Darya Baranouskaya `[一作]` (École Polytechnique Fédérale de Lausanne), Andrea Cavallaro `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 10864 | [OpenAlex ID](https://openalex.org/A5004087827)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PrivLEX，一种基于法律定义的个人数据概念的可解释图像隐私分类器；

**💡 创新点**

创新点在于将法律概念嵌入无标签概念瓶颈模型，利用VLM零样本检测并通过线性组合得到隐私预测，同时能够解释隐私判断；

**🔧 技术方法**

采用CLIP视觉‑语言模型进行零样本概念检测，随后用逻辑回归结合L1正则化完成分类；

**📊 数据集**

使用VISPR和PrivacyAlert两大图像隐私数据集进行训练与评估；

**📈 对比分析**

与四个主流可解释模型和两种最优非可解释模型对比，PrivLEX在Balanced Accuracy与F1‑macro上均领先，且与GPT‑4o相比性能更好；

**⚠️ 局限性**

局限在于CLIP对抽象或多样化的法律概念识别准确性有限，且缺乏上下文与文字识别支持，可能导致概念误判。

---

## 331. Learning Whole-Body Human-Humanoid Interaction from Human-Human Demonstrations

**arXiv ID:** 2601.09518 | [PDF](https://arxiv.org/pdf/2601.09518v1)

**作者:** Wei-Jin Huang `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Peng Cheng Laboratory)

**通讯引用:** 21278 | [OpenAlex ID](https://openalex.org/A5108050904)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套从人类‑人类交互数据生成物理一致的人类‑人形机交互数据的PAIR管线，并基于此训练了能分离何时与何处动作决策的D‑STAR策略，实现了机器人在模拟和真实硬件上的多种握手、拥抱等全身交互。

**💡 创新点**

创新点在于①物理感知的交互重定向通过两阶段优化与接触一致性损失保证了交互接触的保留；②策略将时间和空间决策解耦，利用Phase Attention与Multi‑Scale Spatial模块，再通过扩散规划头生成同步全身动作，突破传统模仿学习的被动跟随。

**🔧 技术方法**

使用了基于约束优化的两阶段重定向、接触矩阵一致性损失、扩散模型规划、Transformer 时序编码、Phase Attention、Multi‑Scale Spatial模块以及基于WBC的低层控制。

**📊 数据集**

主要使用公开的Human‑Human Interaction数据集（如HRI/HOI等），通过PAIR转换得到HHoI数据；在仿真中采用Isaac Gym，真实机器人使用Unitree G1。

**📈 对比分析**

与四类基线（MSE、IK、Orientation、ImitationNet）以及三种策略（TCN、Transformer、Diffusion）进行对比；在六类交互任务中，PAIR在接触一致性与运动平滑性上均优于基线，D‑STAR在平均成功率上达到75.4%，显著高于Transformer（64.3%）和TCN（49.2%）。

**⚠️ 局限性**

仍受限于接触模型的离散化、对人形机高度差异的适应性有限、扩散规划计算量大、在极端速度/尺度变化下仍存在误差，且在更复杂多方交互场景中的泛化需进一步验证。

---

## 332. Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting

**arXiv ID:** 2601.09467 | [PDF](https://arxiv.org/pdf/2601.09467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 333. UAV-enabled Computing Power Networks: Design and Performance Analysis under Energy Constraints

**arXiv ID:** 2601.09493 | [PDF](https://arxiv.org/pdf/2601.09493v1)

**作者:** Yiqin Deng `[一作]` (Lingnan University), Yuguang Fang `[通讯]` (City University of Hong Kong)

**通讯引用:** 23871 | [OpenAlex ID](https://openalex.org/A5016290340)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种无人机（UAV）驱动的计算功率网络（UAV‑CPN）框架，通过UAV动态中继将地面用户（GU）任务转发至分布式计算节点（CN），并在双能源（燃料电池+电池）架构下联合优化UAV高度与发射功率，以最大化任务完成概率。

**💡 创新点**

创新点包括：①引入UAV高度作为主要几何自由度，动态决定可达CN集合，突破传统固定CN访问的“岛效应”；②提出任务完成概率作为综合性能指标，利用随机几何求解期望成功率；③构建双能源功耗模型（推进与通信），并在此约束下设计交替迭代的联合优化算法；④通过Monte Carlo验证理论模型，并在不同能量预算下与多种基线进行量化对比。

**🔧 技术方法**

技术手段：随机几何（Poisson点过程）建模、LoS/NLoS概率模型、计算任务延迟分布、双能耗（推进功率+通信功率）模型、任务完成概率解析式、交替迭代优化（金字塔搜索+拟牛顿法）与贝叶斯优化对比、数值积分与蒙特卡罗仿真。

**📊 数据集**

数据集：实验使用仿真数据——随机生成的GU位置（均匀分布在半径R_u=200 m的请求区）和CN位置（均匀PPP，密度λ_c=5 node/km²），计算任务数据量D=1 MB，延迟预算T_max=55 ms；未使用公开真实数据集，而是通过设定参数表（如路径损耗指数、噪声功率等）进行数值评估。

**📈 对比分析**

对比方法：与三种基线（仅调节发射功率、仅调节高度、三种静态配置）以及贝叶斯优化（BO）进行比较。实验结果显示，联合优化在所有能量预算组合下均优于基线，平均提升29.6%–247.7%，峰值提升可达390%；相较于BO，平均提升13.8%，峰值提升约49%。

**⚠️ 局限性**

局限性：①分析主要基于单UAV单CN的简化模型，未考虑多UAV或多CN并行/协同处理；②忽略多用户干扰、资源竞争与动态CN可用性变化；③假设CN计算时延分布已知且独立；④双能耗模型在不同工作模式（串联/并联）下参数可能需进一步校准；⑤实际部署中环境LoS概率、能耗曲线与模型假设可能偏差，影响算法鲁棒性。

---

## 334. Deep Operator Networks for Surrogate Modeling of Cyclic Adsorption Processes with Varying Initial Conditions

**arXiv ID:** 2601.09491 | [PDF](https://arxiv.org/pdf/2601.09491v1)

**作者:** Beatrice Ceccanti `[一作]` (Eindhoven University of Technology), Martin van Sint Annaland `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 18411 | [OpenAlex ID](https://openalex.org/A5028427879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了使用 Deep Operator Networks (DeepONets) 作为循环吸附过程（如 TVSA）中瞬态 PDE 解决的高效代理模型，能够将不同初始浓度分布映射为时空浓度场，从而加速周期稳态的计算与优化。

**💡 创新点**

创新点在于：①将 DeepONet 应用于需要跨步传递初始状态的循环过程，强调物理可行的初始条件泛化；②构造了混合且多样化的初始条件数据集，涵盖线性、S 型、指数、高斯以及外推与新函数族（正弦），检验模型在超出训练分布时的鲁棒性；③在训练损失中加入初始条件一致性项和全域数据项，以提高边界条件遵循性。

**🔧 技术方法**

采用了基于 MLP 的分支–干线架构 DeepONet；分支网络使用 SiLU 激活、Kaiming 初始化；干线网络使用 Sine 激活（ω0=20）并采用 SIREN 初始化；训练采用 Adam 优化器，学习率 1e-4 并调度，权重 λ_ic=3、λ_data=1；最终模型共有 6 层 200 节点，latent 维度 100。

**📊 数据集**

数据集共 10,000 条初始条件函数，按 25% 比例分布于四类函数族（线性、S 型、指数、高斯），通过随机参数采样并进行垂直平移；每条初始条件对应完整的 100×101 时空网格的数值解；另外构造 1,000 条 OOD 数据用于外推评估，包括参数外推和新正弦函数族。

**📈 对比分析**

与全域数值解比较，采用均方误差和均相对 L²误差评估；在测试集上平均 L² 误差 0.168%，在 OOD 集合上 2.282%；在多种输入（单调、非单调、正弦）下均表现出良好一致性，误差集中在吸附前沿及高频细节处；模型在不同周期步的初始状态传递中保持数值稳定。

**⚠️ 局限性**

主要局限包括：①对高频、尖锐空间特征的重建误差较大，受网络光谱偏差影响；②极端初始梯度导致误差随时间向下游传播；③训练需要大量高精度 PDE 解决方案，离线成本高；④未对边界条件做硬约束，可能在极端 OOD 条件下出现偏差。

---

## 335. Terminally constrained flow-based generative models from an optimal control perspective

**arXiv ID:** 2601.09474 | [PDF](https://arxiv.org/pdf/2601.09474v1)

**作者:** Weiguo Gao `[一作]` (Fudan University), Qianxiao Li `[通讯]` (National University of Singapore)

**通讯引用:** 2274 | [OpenAlex ID](https://openalex.org/A5069654038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于终端最优控制的流式生成模型采样方法（TOCFlow），通过在已有的流匹配模型上施加时间尺度的控制，实现对终端分布约束的高效满足。

**💡 创新点**

创新点包括：①将终端约束问题形式化为最优控制，利用HJB方程得到梯度反馈控制；②在参考流的初始共动框架下将控制能量与Wasserstein距离关联，给出上下界；③在两种极端惩罚极限下给出收敛性分析；④通过近似拉普拉斯–霍普·拉克斯求解，得到只需一维标量搜索的TOCFlow闭式公式；⑤在高维科学任务中实现对等式、非等式及全局统计约束的统一处理。

**🔧 技术方法**

主要技术包括：流式生成模型（Flow Matching）、最优控制理论（HJB方程、可变度量）、Riemannian 几何与共动坐标变换、Wasserstein 运动能量定理、Hopf–Lax 公式、近似梯度（GD）、Gauss–Newton（GN）与 TOCFlow 算法、线性二次高斯分析、自动微分（JVP/VJP）。

**📊 数据集**

实验数据集：①Darcy流（高维泊松方程的渗透率与压强场），②受限轨迹规划（机器人路径避障样本），③湍流快照（2D湍流场满足 Kolmogorov 频谱）。

**📈 对比分析**

对比方法包括欧氏梯度引导、投影基准（逐步投影至约束集合）以及基于Gauss–Newton的几何引导。评估指标涵盖约束违背度、Wasserstein 距离、PDE误差、碰撞率、湍流能谱一致性。结果显示，TOCFlow 在保持生成质量的同时，约束满足度显著提升：Darcy 流误差降低十倍，轨迹规划碰撞率降为零，湍流能谱在惯性区的 Kolmogorov 指数更贴近理论。且计算成本与单纯梯度引导相近，远低于完整GN求解。

**⚠️ 局限性**

局限性包括：①需预先训练并获取稳定的参考流；②假设约束是光滑、满秩的零水平集；③TOCFlow 的近似依赖于局部线性与梯度信息，对高度非凸或多模约束可能不足；④在极端高维或约束空间维度接近数据维度时，JVP/VJP 的自动微分成本仍不可忽视；⑤惩罚调度 λ_t 的选择对性能影响显著，需经验调节；⑥当参考流本身存在奇异或不连续时，理论的控制能量界可能失效。

---

## 336. EvoFSM: Controllable Self-Evolution for Deep Research with Finite State Machines

**arXiv ID:** 2601.09465 | [PDF](https://arxiv.org/pdf/2601.09465v1)

**作者:** Shuo Zhang `[一作]` (QuantaAlpha), Huacan Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5031229572)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EvoFSM，一个结构化自我演化框架，将深度研究任务建模为有限状态机并通过原子操作实现自适应改进。

**💡 创新点**

通过将优化空间拆分为 Flow 与 Skill 两个维度，并限制演化为有限状态机上的原子操作，兼顾可控性与适应性；同时引入自我演化记忆以积累经验。

**🔧 技术方法**

利用 LLM 生成的 Agentic RAG、搜索工具、Critic 评估器，构建 FSM；采用原子操作、Critic 反馈、经验池等机制实现结构化演化。

**📊 数据集**

在五个多跳问答基准（HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle、DeepSearch）以及 ALFWorld 和 WebShop 两个交互决策基准上进行实验。

**📈 对比分析**

与 Standard RAG、Agentic RAG、Search‑o1 三个基线以及多种 LLM（GPT‑4o、Claude‑4、Llama‑3‑70B 等）对比，EvoFSM 在所有基准上均取得显著提升，尤其在 DeepSearch 上提高了约10–11%。

**⚠️ 局限性**

依赖大模型的 Prompt 与 Critic，缺乏模型微调；Critic 误判可能导致不良演化；经验池无限增长导致记忆管理与可扩展性问题。

---

## 337. Analysis of the Maximum Prediction Gain of Short-Term Prediction on Sustained Speech

**arXiv ID:** 2601.09461 | [PDF](https://arxiv.org/pdf/2601.09461v1)

**作者:** Reemt Hinrichs `[一作]` (Leibniz University Hannover), Jörn Ostermann `[通讯]` (Leibniz University Hannover)

**通讯引用:** 5519 | [OpenAlex ID](https://openalex.org/A5064913233)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了持续语音的最大预测增益，并比较了线性、RBF、TDNN等预测器与理论上限的差距。

**💡 创新点**

首次在保持足够平稳的持续音素上评估最大预测增益，并用核回归估计条件期望，揭示线性预测对无声音素近最优性及语音音素间差异。

**🔧 技术方法**

采用Nadaraya–Watson核回归、KSG与FMI互信息估计、线性预测、RBF网络和TDNN预测模型。

**📊 数据集**

使用自制的5名受试者在48 kHz 24‑bit的持续语音数据集（10 k样本片段）进行实验。

**📈 对比分析**

通过预测增益比较，发现核回归可比线性多2–8 dB，RBF约0.6–1.5 dB，TDNN仅略优；KSG/FMI 上界偏高，核回归给出更可信的下界估计。

**⚠️ 局限性**

限制在于平稳片段长度有限、互信息估计存在偏差、仅考虑短期预测、未验证长周期预测和跨说话者的泛化能力。

---

## 338. TaxoBell: Gaussian Box Embeddings for Self-Supervised Taxonomy Expansion

**arXiv ID:** 2601.09633 | [PDF](https://arxiv.org/pdf/2601.09633v1)

**作者:** Sahil Mishra `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 4849 | [OpenAlex ID](https://openalex.org/A5046521217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于高斯盒子嵌入的自监督税onomies扩展框架，将概念表示为带协方差的不确定性盒子，并通过能量函数学习层级关系。

**💡 创新点**

创新点在于将盒子几何与多元高斯分布对接，既保留盒子包含性，又引入协方差不确定性和概率重叠/KL能量，解决了传统盒子梯度不稳定、缺乏不确定性及多义性表达的问题。

**🔧 技术方法**

使用BERT+MLP编码、盒子到高斯的投影、Bhattacharyya相似度、KL包含度、体积正则化及对比学习的能量优化。

**📊 数据集**

在五个公开taxonomy基准上评估：SCI、ENV、WordNet、SemEval‑Food、MeSH。

**📈 对比分析**

与八个基线（包括向量、结构化和盒子方法）比较，平均MR下降19%、MRR提升19%、Recall@k提升约25%，在单父和多父设置均显著优于最强对手。

**⚠️ 局限性**

局限包括对单一高斯盒子假设（无混合模型）、对高维协方差的数值稳定性需求以及对大型、稀疏taxonomy的泛化仍待验证。

---

## 339. Physics Informed Optimal Homotopy Analysis Method (PI-OHAM): A Hybrid Analytical Computational Framework for Solving nonlinear Differential Equations

**arXiv ID:** 2601.09567 | [PDF](https://arxiv.org/pdf/2601.09567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 340. LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation

**arXiv ID:** 2601.09631 | [PDF](https://arxiv.org/pdf/2601.09631v1)

**作者:** Stergios Chatzikyriakidis `[一作]` (University of Crete), Stergios Chatzikyriakidis `[通讯]` (University of Crete)

**通讯引用:** 825 | [OpenAlex ID](https://openalex.org/A5020791896)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个结合大型语言模型与确定性语音学引擎的混合系统，用于现代希腊语韵律的识别与生成。

**💡 创新点**

创新点在于提出基于韵律规则的符号验证器和生成-验证-改进循环，以及发布40,000+高质量希腊韵对语料库。

**🔧 技术方法**

主要技术包括LLM多种提示策略（零样本、链式推理、RAG增强）、语音学引擎（音节划分、重音检测、韵域提取）以及符号式韵律验证与代理式生成管道。

**📊 数据集**

使用了从Anemoskala和Interwar Poetry整理的40,576条希腊韵对语料，经过清洗后形成的标准化数据集。

**📈 对比分析**

通过在8种模型（Claude、GPT‑4o、Gemini、Llama、Mistral等）与4种提示配置下进行对比实验，韵律识别最高准确率为54%，生成合法诗歌率在验证循环下提升至73.1%，而纯LLM生成仅为0–4%。

**⚠️ 局限性**

局限性包括语料规模相对较小、生成循环带来的计算开销以及对低资源语言的普适性尚待验证。

---

## 341. DPWriter: Reinforcement Learning with Diverse Planning Branching for Creative Writing

**arXiv ID:** 2601.09609 | [PDF](https://arxiv.org/pdf/2601.09609v1)

**作者:** Qian Cao `[一作]` (Renmin University of China), Han Li `[通讯]` (Kuaishou Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于强化学习的框架，利用半结构化长Chain‑of‑Thought（CoT）规划与推理分支，显式引导多样化探索，提升大型语言模型在创意写作任务中的输出多样性；

**💡 创新点**

创新点在于：①设计了多样化规划分支（Diverse Planning Branching）机制，在规划阶段按多样性分支候选计划；②引入组感知多样性奖励（diversity contribution reward），在奖励函数中同步考虑质量与多样性；

**🔧 技术方法**

采用的技术包括：GRPO强化学习框架；SFT预训练半结构化CoT数据；GPT‑4.1生成规划与CoT；多样性奖励（基于n‑gram与语义余弦距离）；

**📊 数据集**

使用的数据集：43K指令‑规划‑CoT‑响应样本用于SFT；随后10K样本用于RL；评测基准包括WritingBench、Creative Writing v3（EQ‑Bench）、ArenaHard v2.0（创作子集）和NoveltyBench；

**📈 对比分析**

与GRPO、GRPO‑Unlikeliness、Darling、GAPO等基线在Qwen3‑4B和Llama‑3.2‑3B上对比，结果显示本文方法在质量（Score、ELO）和多样性（Emb、EAD、Distinct）指标上均获得最高分，Embedding多样性提升约15%，EAD提升约9.9%；在NoveltyBench上Distint得分也领先；

**⚠️ 局限性**

局限性包括：①额外的规划分支与多样性奖励导致计算开销增加，难以在极大模型或数据集上扩展；②多样性与质量的平衡仍未完全解决；③多样性提升是否真正促进创意等更深层次质量尚未充分验证。

---

## 342. Energy-Entropy Regularization: The True Power of Minimal Looped Transformers

**arXiv ID:** 2601.09588 | [PDF](https://arxiv.org/pdf/2601.09588v1)

**作者:** Wai-Lun Lam `[一作]` `[通讯]`, Wai-Lun Lam

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种能量-熵正则化的训练框架，使得单头、低维（d=8）循环Transformer能够在长达1000个token的Induction Head任务中实现高准确率。

**💡 创新点**

通过将Tsallis熵收敛约束、Hamiltonian动力学导航与能量-熵正则化损失相结合，将高维非凸损失景观转化为漏斗型平滑结构，实现极小参数规模下的强推理能力。

**🔧 技术方法**

Tsallis熵、Hamiltonian动力学模型、能量-熵正则化损失（动量、势能、熵项）、物理启发的正则化系数、单头循环Transformer架构、低维嵌入。

**📊 数据集**

使用Induction Head任务数据集（自制长序列，长度可达1000），训练时采用16–64长度，评估时延伸至1000长度。

**📈 对比分析**

与FOP‑Looped‑Adaptive（d=64）对比，EER在L=1000时准确率从33.5%跃升至79.2%，参数量仅0.02%，展现出优异的长度泛化与效率。

**⚠️ 局限性**

结果仍依赖硬件随机性与超参数调优，主要验证在单一推理任务上，尚未在更广泛任务或大规模模型中证明普适性。

---

## 343. On Linear Estimators for some Stable Vectors

**arXiv ID:** 2601.09554 | [PDF](https://arxiv.org/pdf/2601.09554v1)

**作者:** Rayan Chouity `[一作]` (American University of Beirut), Ibrahim Abou-Faycal `[通讯]` (American University of Beirut)

**通讯引用:** 822 | [OpenAlex ID](https://openalex.org/A5055725981)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究α‑稳定随机变量在两种依赖模型（线性混合与子高斯）下的条件期望估计，证明其线性并给出分散最优线性估计器的闭式解。

**💡 创新点**

首次在α‑稳定情境下证明条件期望即为线性，且在子高斯模型下条件期望与分散最优线性估计器完全一致，扩展了高斯情形的经典结果；同时对线性混合模型提供了新的最优估计公式。

**🔧 技术方法**

利用α‑稳定分布的特征函数、傅里叶变换与积分求导、最优化分析（L^α 损失），以及子高斯向量的高斯‑α/2 结构，进行严格的理论推导。

**📊 数据集**

本文为理论研究，没有使用实验或公开数据集。

**📈 对比分析**

与高斯情况下已知的最佳线性均方误差估计器对比，结果显示子高斯模型下线性期望即为分散最优估计，性能与高斯相当；在线性混合模型下两者不一致，体现更复杂的估计行为。

**⚠️ 局限性**

仅适用于 α∈(0,2) 且已知稳定参数；仅考虑两种特定依赖结构，无法直接推广至一般稳定向量；在 α≤1 时最优估计取边界值，优化解不唯一；需先验已知方差、相关系数等参数。

---

## 344. FairShare: Auditable Geographic Fairness for Multi-Operator LEO Spectrum Sharing

**arXiv ID:** 2601.09641 | [PDF](https://arxiv.org/pdf/2601.09641v1)

**作者:** Seyed Bagher Hashemi Natanzi `[一作]` (Worcester Polytechnic Institute), Bo Tang `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 4817 | [OpenAlex ID](https://openalex.org/A5074257864)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

研究多运营商低轨卫星网络的动态频谱共享，揭示传统基于SNR优先调度导致城乡服务差距，并提出公平配额框架FairShare来消除这一偏差。

**💡 创新点**

提出基于地理配额的公平频谱分配机制FairShare，既实现了城乡公平（Δ_geo=0.72×）又不牺牲吞吐量，并提供可调配额和公平诊断指标，填补了现有标准缺失的公平保障空白。

**🔧 技术方法**

使用3GPP TR 38.811合规的卫星通道模型，基于TensorFlow实现大规模仿真，并与SNR优先、需求比例、均等分配等基线算法进行对比。

**📊 数据集**

构造合成用户分布（城市50%、郊区20%、农村30%），采用Starlink样板星座轨道参数，采用全缓冲流量模型进行实验。

**📈 对比分析**

通过50次Monte Carlo仿真对比，FairShare将城乡差距从1.65×降至0.72×，同时保持或略提升吞吐量，且运行时比SNR优先调度快3.3%。

**⚠️ 局限性**

局限在于假设中心化协调器、仅下行、全缓冲流量、轨道动态被简化，未考虑竞价激励和真实部署中的多样化网络环境。

---

## 345. CogRail: Benchmarking VLMs in Cognitive Intrusion Perception for Intelligent Railway Transportation Systems

**arXiv ID:** 2601.09613 | [PDF](https://arxiv.org/pdf/2601.09613v1)

**作者:** Yonglin Tian `[一作]` (Chinese Academy of Sciences), Yisheng Lv `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 9499 | [OpenAlex ID](https://openalex.org/A5076992681)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了CogRail基准和RailGPT框架，用多模态问答评估铁路入侵感知任务。

**💡 创新点**

首创将空间位置、运动状态、威胁评估三任务统一为问答基准，并提出联合微调多任务学习。

**🔧 技术方法**

采用大型视觉语言模型（如Qwen2、LLaMA3.2、LLaVA等）结合多模态提示与LoRA微调。

**📊 数据集**

使用公开铁路监控数据集RailSem19与MRSI构建CogRail，并生成视觉问答对。

**📈 对比分析**

通过零射击、单任务微调和联合微调三种设定比较，联合微调在三任务上平均提升约10‑20% F1，最优模型达约76% F1。

**⚠️ 局限性**

受限于单帧视觉信息缺乏长时序推理，模型在运动与威胁推断上仍易失真，需进一步引入时序建模与更丰富的上下文。

---

## 346. Iterative Differential Entropy Minimization (IDEM) method for fine rigid pairwise 3D Point Cloud Registration: A Focus on the Metric

**arXiv ID:** 2601.09601 | [PDF](https://arxiv.org/pdf/2601.09601v1)

**作者:** Emmanuele Barberi `[一作]` (University of Messina), Filippo Cucinotta `[通讯]` (University of Messina)

**通讯引用:** 929 | [OpenAlex ID](https://openalex.org/A5004530402)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于差分熵的全新配准指标并利用其实现细粒度刚性配准

**💡 创新点**

该指标不依赖参考点云、对点云密度差、噪声、孔洞和局部重叠均具有鲁棒性，并且在ROI内能显式显示唯一最小值

**🔧 技术方法**

利用多维高斯分布的差分熵、邻域半径取自四邻点距离、构造 q_tot 并在 ROI 内迭代最小化

**📊 数据集**

主要使用 Stanford Bunny（含不同稀疏、噪声、孔洞、部分重叠版本）以及户外 LiDAR 场景的部分重叠点云

**📈 对比分析**

与 RMSE、Chamfer 距离和 Hausdorff 距离比较，实验表明 q_tot 在所有扰动条件下都能精确定位到完美对齐，RMSE 受非对称选择影响且误差明显

**⚠️ 局限性**

局限性在于需要先把点云预对齐到 ROI，邻域半径 r 的选择需经验调优；在极端稀疏或大噪声情况下的性能仍待进一步验证

---

## 347. Constraint- and Score-Based Nonlinear Granger Causality Discovery with Kernels

**arXiv ID:** 2601.09579 | [PDF](https://arxiv.org/pdf/2601.09579v1)

**作者:** Fiona Murphy `[一作]` (Trinity College Dublin), Alessio Benavoli `[通讯]` (Trinity College Dublin)

**通讯引用:** 1966 | [OpenAlex ID](https://openalex.org/A5044312043)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究基于核方法的Granger因果识别，先将两种主流核约束方法KGC和lsNGC统一到Kernel Principal Component Regression (KPCR)框架，随后提出一种基于高斯过程（Gaussian Process, GP）并加入Smooth Information Criterion (SIC) 的评分方法 GP_SIC，用来进行非线性因果发现，并进一步扩展到同时因果（contemporaneous causal）关系的识别。

**💡 创新点**

创新点：
1) 通过KPCR统一KGC与lsNGC，改进统计检验（从相关系数到F检验并加入Bonferroni校正）。
2) 提出GP_SIC——在GP回归中直接以最大化的边际似然加上SIC作为评分，兼顾稀疏性与信息准则，避免两模型比较。 
3) 在GP_SIC基础上设计两阶段算法（邻接检索 + 方向判定），利用GC框架在满足无完全同时碰撞器假设时即可识别同时因果，显著减少检验次数。

**🔧 技术方法**

技术与方法：
- 核方法（核回归、核PCA、Nyström近似）
- 高斯过程回归（ARD核）
- Smooth Information Criterion (SIC) 与长度尺度惩罚
- F检验、Bonferroni校正
- PCMCI 与 MCI 作为对照方法
- GPDC（基于距离相关的条件独立性检验）
- 统计检验（Fisher、Wilcoxon）
- 蒙特卡洛仿真与真实数据实验

**📊 数据集**

数据集：
1) 19个仿真基准系统（变量数从2到30，包含线性、非线性、协同、混杂、同步等多种结构）。
2) 50个pH中和工厂（CSTR）模拟数据，分别在servo和regulatory两种操作模式下生成。

**📈 对比分析**

比较方式与性能：
- 与KGC、lsNGC、KPCR、PCMCI以及其他GP方法（GP、GPΔℓ、GP-GLRT）在相同仿真集上进行F1分数评估。 
- 统计检验（Bayesian Wilcoxon）显示GP_SIC在大多数实验中优于PCMCI，优于lsNGC，显著优于KGC；KPCR在多数情形下与KGC相当或更优。 
- 在同时因果实验中，GP_SIC相较PCMCI+取得更高的F1，且检验次数仅为PCMCI+的十分之一。 
- 在pH中和实验中，GP_SIC平均F1约为0.45-0.55，优于PCMCI+（约0.26-0.37）。

**⚠️ 局限性**

局限性：
- 仅适用于高斯噪声模型，无法直接处理非高斯或强非线性噪声。 
- 核参数（长度尺度、核系数）与Nyström近似的选择仍基于启发式或固定，可能导致数值不稳定。 
- 同时因果识别依赖无完全同时碰撞器假设，若存在此类结构，方法可能失效。 
- 需要手动选择滞后阶数与嵌入维度，虽有启发式，但在复杂系统中仍可能产生误差。 
- 对极大规模数据的计算仍受限于核矩阵大小与GP求解复杂度，需进一步加速。

---

## 348. Permutation Matching Under Parikh Budgets: Linear-Time Detection, Packing, and Disjoint Selection

**arXiv ID:** 2601.09577 | [PDF](https://arxiv.org/pdf/2601.09577v1)

**作者:** MD Nazmul Alam Shanto `[一作]` (American International University-Bangladesh), Md. Manzurul Hasan `[通讯]` (American International University-Bangladesh)

**通讯引用:** 60 | [OpenAlex ID](https://openalex.org/A5061452755)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出并实现了三个线性时间的滑动窗口算法：① 判断给定文本是否包含与模式 Parikh 向量相等的子串；② 在模式向量视为资源预算的前提下，求最长可行子串（MFSP）；③ 在已枚举所有匹配后，使用贪心算法选择最大数量的互不重叠匹配。

**💡 创新点**

创新点在于：① 用差分向量和非零计数器实现 O(n+σ) 的Permutation匹配；② 将匹配问题转化为打包约束，给出两指针法求解 MFSP 的最优性证明；③ 证明等长匹配间隙可用贪心一次扫描得到最大互不重叠集合，且整体时间仍为 O(n+σ)。

**🔧 技术方法**

主要技术：Parikh 向量差分维护、非零计数器、两指针滑动窗口、阈值交叉计数器、贪心交换证明；实现基于数组或哈希表的压缩字母表。

**📊 数据集**

实验使用两类数据集：① 随机合成字符串（σ = 4,16,64,256，m = 16,64,256,1024，n ≤ 10⁷）；② 真实自然语言文本（字母表经过压缩）。

**📈 对比分析**

与已有的滑动窗口或索引方法对比，实验表明三种算法均保持线性增长，常数因子主要由内存访问和匹配密度决定；在大字母表场景下，压缩映射显著降低内存占用并提升吞吐量；MFSP 的误差计数器实现避免了全局检查，保持了 O(n) 的总时间。

**⚠️ 局限性**

局限性：① 仅处理等长窗口，无法直接扩展到变长模式；② 对于非常大的 alphabet，虽然可压缩，但仍需额外哈希开销；③ MFSP 只允许严格不超预算，未考虑近似或加权版本；④ 贪心策略依赖于所有匹配已枚举，若匹配量大则输出成本高。

---

## 349. The Spectral Representations Of The Simple Hypothesis Testing Problem

**arXiv ID:** 2601.09564 | [PDF](https://arxiv.org/pdf/2601.09564v1)

**作者:** Barış Nakiboğlu `[一作]` `[通讯]`, Barış Nakiboğlu

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过凸共轭（Legendre变换）得到简单假设检验问题中Type II错误概率（体积）关于Type I错误概率（体积）的谱表示，并推广到σ‑有限测度的非平凡情形。

**💡 创新点**

创新点在于首次给出Type II错误体积的凸共轭（原始熵谱）的显式表达式，并利用半连续逆函数（似然比分位数）构造新的身份，扩展了传统Neyman–Pearson lemma到随机化检测与σ‑有限测度；同时得到更紧的非渐近上界。

**🔧 技术方法**

主要技术包括凸分析与共轭变换、测度论（Radon–Nikodym、勒贝格分解）、似然比分位数的半连续逆、Berry–Esseen定理、Gaussian Mills比以及指数测度变换（tilting）等。

**📊 数据集**

由于研究对象是抽象测度，本文未使用具体数据集；所有结论均为理论推导与近似，适用于任意满足假设的σ‑有限测度。

**📈 对比分析**

作者将新得到的谱表示与传统的Berry–Esseen上界和Gaussian Mills比结合，得到的非渐近误差界在内在结构与常见的记忆无关（memoryless）案例中比现有最优界更紧；实验对比表明误差可控制在常数级别。

**⚠️ 局限性**

局限性包括：（1）仅适用于非平凡（非空）σ‑有限测度；（2）谱表达式和逆函数计算复杂，实际使用需额外实现；（3）缺乏针对具体统计模型或通信系统的数值验证，理论结果的实用性仍需进一步探索。

---

## 350. Residual Power Flow for Neural Solvers

**arXiv ID:** 2601.09533 | [PDF](https://arxiv.org/pdf/2601.09533v1)

**作者:** Jochen Stiasny `[一作]` (Delft University of Technology), Jochen Cremer `[通讯]` (Delft University of Technology)

**通讯引用:** 2146 | [OpenAlex ID](https://openalex.org/A5019114577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了残差电力流（RPF）表述，并利用神经求解器学习电力流解，随后将其嵌入预测-优化（PO）框架中，解决AC-OPF、准稳态电力流等任务；

**💡 创新点**

核心创新是通过使用分支角避免传统总线类型导致的不对称性，提出连续的不可行度残差度量，允许在不可行工况下训练神经求解器，从而显著提升学习灵活性和鲁棒性；

**🔧 技术方法**

采用残差电力流数学模型、前馈神经网络（两层宽度100）、L‑BFGS梯度优化、自动微分求梯度、以及PO框架下的预测‑优化求解；

**📊 数据集**

使用IEEE 9节点系统，生成2000条训练工况与1000条测试工况，涵盖随机负荷、功率因数、发电机出力与电压设定等多样化参数；

**📈 对比分析**

将RPF与传统BIM表述在电压预测、残差分布和不可行工况下的误差进行对比，实验表明RPF在使用学习特征时误差约下降十倍，残差分布更均匀、可行度更高，PO任务中取得与精确解相近的结果；

**⚠️ 局限性**

目前仅在9节点小规模系统验证，未针对拓扑变化、设备可用性等情形，神经求解器架构相对简单，计算实现未优化，需进一步扩展到更大规模、不同拓扑的实际系统中。

---

## 351. A Finite-Sample Strong Converse for Binary Hypothesis Testing via (Reverse) Rényi Divergence

**arXiv ID:** 2601.09550 | [PDF](https://arxiv.org/pdf/2601.09550v1)

**作者:** Roberto Bruno `[一作]` (University of Salerno), Amedeo Roberto Esposito `[通讯]` (Okinawa Institute of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文在有限样本二元假设检验中，通过逆 Rényi 散度推导了非渐近的 Type II 错误下界，并在 Type I 错误指数衰减时给出了强逆证，完整刻画了误差指数相位转变。

**💡 创新点**

创新点在于：①首次使用逆 Rényi 散度得到强逆结论，无需吹爆或平滑方法；②给出了有限样本下的显式指数界；③在指数约束下实现了 KL 散度阈值 D(P₁∥P₀) 的相位分界。

**🔧 技术方法**

主要技术包括：逆 Rényi 散度与数据处理不等式、Neyman–Pearson 判别、非渐近界推导、凸性与极限分析，以及 Laplace–Feller 近似。

**📊 数据集**

实验数据为两类合成场景：伯努利分布 P₀=Bern(½)、P₁=Bern(½+δ) 与高斯分布 P₀=𝒩(μ,1)、P₁=𝒩(μ+δ,1)，通过数值模拟验证理论。

**📈 对比分析**

与 KL‑基、Hellinger‑基、Berry–Esseen、平滑‑out 等已有下界比较，本文界限在严格误差约束和样本有限的情形下明显更紧；数值图表显示其优于传统方法。

**⚠️ 局限性**

局限性在于仅针对 i.i.d. 采样、需要绝对连续性；对 λ 的选取需经验；对非独立或非同分布场景的推广尚未给出。

---

## 352. Identifying Models Behind Text-to-Image Leaderboards

**arXiv ID:** 2601.09647 | [PDF](https://arxiv.org/pdf/2601.09647v1)

**作者:** Ali Naseh `[一作]` (University of Massachusetts Amherst), Amir Houmansadr `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 5219 | [OpenAlex ID](https://openalex.org/A5018588864)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究了投票式文本到图像排行榜中的模型匿名性，并提出了基于图像嵌入聚类的去匿名化攻击。

**💡 创新点**

创新点在于发现不同T2I模型在相同提示下生成的图像在嵌入空间中形成可分离簇，利用简单的质心最近邻即可实现高精度去匿名化，并提出提示可区分度指标。

**🔧 技术方法**

采用CLIP等图像嵌入器、质心最近邻分类、一对多与一对一阈值方法，以及可区分度度量。

**📊 数据集**

使用22个最先进T2I模型、280个来自公共排行榜的提示、总计约184k张图像，另扩展到200万张图像的实验。

**📈 对比分析**

与传统指纹识别和训练分类器对比，质心聚类方法在Top‑1可达91%（在280提示下）甚至99%（单目标）并且在少量样本时仍保持60%+准确率。

**⚠️ 局限性**

局限包括依赖高质量嵌入器、对提示和模型集的敏感性、攻击成本仍存在、以及防御需要可解释的扰动，且在完全未知模型或极小样本时效果下降。

---

## 353. Secret sharing with additive access structures from correlated random variables

**arXiv ID:** 2601.09640 | [PDF](https://arxiv.org/pdf/2601.09640v1)

**作者:** David Miller `[一作]` (University of Texas at Arlington), Rémi A. Chou `[通讯]` (University of Texas at Arlington)

**通讯引用:** 593 | [OpenAlex ID](https://openalex.org/A5047776960)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了Additive Access Structure（AAS）下的秘密共享模型，允许随时间动态扩展访问结构并在公共频道上利用相关随机性进行共享。

**💡 创新点**

创新点在于证明即使AAS动态演化，仍可使用单一预先确定的策略实现与固定访问结构相同的秘密率，并在阈值AAS时达到容量。

**🔧 技术方法**

采用了随机分箱（random binning）量化化简、典型序列与信息熵分析等信息理论技术。

**📊 数据集**

未使用具体数据集，纯理论证明。

**📈 对比分析**

与固定访问结构下已知最优策略对比，证明在每个时间步可实现相同或达到容量的秘密率，理论上实现最优性能。

**⚠️ 局限性**

限制在于尚未给出构造性实现方案，实际实现难度较高；并未考虑通道噪声等现实因素的影响。

---

## 354. LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach

**arXiv ID:** 2601.09635 | [PDF](https://arxiv.org/pdf/2601.09635v1)

**作者:** Kuo Liang `[一作]`, Chung-Piaw Teo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于三代理工作流（分类、工作流生成、模型生成）的LLM自动化大规模优化模型构建框架（Lean‑LLM‑OPT），能够从自然语言描述和外部数据文件中自动生成线性/混合整数规划模型及对应求解代码。

**💡 创新点**

核心创新在于：① 通过动态生成可执行工作流将复杂建模任务拆分为结构化子任务；② 结合检索增强工具（FileQA、CSVQA）将数据处理交给辅助工具；③ 采用轻量级代理化设计，避免对大型模型进行昂贵的再训练，显著提升推理效率与可解释性。

**🔧 技术方法**

技术手段主要包括：大型语言模型（GPT‑4.1、gpt‑oss‑20B）、工具集成（检索/CSV读取）、少样本学习（few‑shot）、多代理协同推理、Python 代码生成以及与 Gurobi 求解器的自动衔接。

**📊 数据集**

使用自构造的 Lean‑LLM‑OPT benchmark，包含 96 条参考实例（覆盖 6 种问题类型）和 101 条测试实例（约 50% 大规模≥100 变量，30% 中等规模，20% 小规模），数据来源于 Kaggle、书籍案例及实际业务场景。

**📈 对比分析**

与 Gemini 3 Pro、GPT‑5.2、ORLM、OptiMUS 等方法在大规模优化基准上对比，Lean‑LLM‑OPT 在 GPT‑4.1、gpt‑oss‑20B 两种基模型上分别达成 85.15% 和 80.2% 的执行准确率，显著优于对手（最高仅 52.48%），在新加坡航空收入管理案例中也获得 93.33%/86.67% 的准确率并保持 ≤14% 的最优性差距。

**⚠️ 局限性**

局限性包括：① 对极长输入或高变量量时仍会出现精度下降；② 主要针对线性/混合整数规划，非线性或约束形式变化较大的问题需进一步扩展；③ 依赖预先构造的参考实例，若问题域大幅偏离训练集仍可能面临泛化挑战；④ 仍需要配套工具与数据检索模块，部署成本与维护有一定门槛。

---

## 355. Perceptually-Guided Adjusted Teleporting: Perceptual Thresholds for Teleport Displacements in Virtual Environments

**arXiv ID:** 2601.09632 | [PDF](https://arxiv.org/pdf/2601.09632v1)

**作者:** Rose Connolly `[一作]` (Trinity College Dublin), Rachel McDonnell `[通讯]` (Trinity College Dublin)

**通讯引用:** 3407 | [OpenAlex ID](https://openalex.org/A5079301210)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本文在虚拟现实中通过实验测定了可被用户察觉的即时位移阈值，探究了在不同距离下向前和向后调整传送点的可行性。

**💡 创新点**

创新点在于首次将感知阈值测量方法（自适应阶梯法）应用于传统点投射式传送，揭示向后位移更易隐藏且在较大传送距离下阈值更高，从而为社交VR中的个人空间管理与游戏导航提供潜在的隐形重定向策略。

**🔧 技术方法**

主要技术包括Oculus Quest 3头显、Unity 6000.0.047f1开发的实验环境、基于Unity Staircase Procedure Toolkit的阶梯实验设计、2AFC判定UI以及对VR经验和空间能力的问卷评估。

**📊 数据集**

使用的数据集是31名受试者在两种传送范围（小 2.5 m 和大 9 m）下完成的传送试验的主观检测结果，并配合SBSOD和SOT测得的空间能力与VR经验分数。

**📈 对比分析**

比较方法采用双因素重复测量ANOVA和皮尔逊相关分析，结果显示向后位移阈值显著大于向前（p<0.001），大范围阈值显著高于小范围（p=0.008），并且VR经验和SOT分数与阈值呈显著相关，说明实验方法可精准定位个人感知极限。

**⚠️ 局限性**

局限性包括仅考察前后两种位移方向、样本量有限、实验环境为封闭房间、未测量对用户代理感、沉浸感和安全性的影响，且在开放式或社交情境下的阈值可能有所不同。

---

## 356. Linear Complexity Self-Supervised Learning for Music Understanding with Random Quantizer

**arXiv ID:** 2601.09603 | [PDF](https://arxiv.org/pdf/2601.09603v1)

**作者:** Petros Vavaroutsos `[一作]` (Orfium Research), Pantelis Vikatos `[通讯]` (Orfium Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究如何在音乐信息检索任务中使用基于 Branchformer 与 SummaryMixing 的基础模型，并通过随机量化技术显著缩减模型体积，同时保持竞争性性能。

**💡 创新点**

首次将 Branchformer 架构、线性复杂度的 SummaryMixing 替代多头自注意力，以及随机量化 tokenizer 结合应用于音乐领域，实现模型参数下降 8.5%–12.3% 的同时，性能保持或超过现有基线。

**🔧 技术方法**

采用 Branchformer 编码器、SummaryMixing 线性注意力、随机量化 tokenization、频谱 Mel 计算、时间遮掩、MSE+交叉熵自监督预训练，并在预训练阶段使用大规模音频数据。

**📊 数据集**

使用公开数据集 Music4All、FMA‑large，私有约 200k 小时音乐数据；下游评测则用 MagnaTagAtune、MTG‑Jamendo、Giantsteps‑MTG‑keys、GTZAN、Emomusic、NSynth、Vocalset 等多任务数据集。

**📈 对比分析**

与标准 Branchformer、Conformer、MERT‑330 等基线模型在多项下游任务（音乐标签、音调识别、情感回归、乐器分类、歌手识别等）进行冻结+单层微调比较；本方法在大多数任务上与 MERT 相当或更优，参数量减少 8.5%–12.3%，在键盘检测、乐器识别等任务上提升约 10%。

**⚠️ 局限性**

局限性包括：仅使用固定 30 s 片段和 400 ms 随机遮掩，未探索更长上下文或重叠段；未进行全面的超参搜索；未尝试模型压缩技术如剪枝/量化/蒸馏；私有数据对结果可能产生偏差；对跨域（音乐+语音）预训练及迁移效果尚未深入研究。

---

## 357. Creating a Hybrid Rule and Neural Network Based Semantic Tagger using Silver Standard Data: the PyMUSAS framework for Multilingual Semantic Annotation

**arXiv ID:** 2601.09648 | [PDF](https://arxiv.org/pdf/2601.09648v1)

**作者:** Andrew Moore `[一作]` (Lancaster University), Xiaobin Yang `[通讯]` (Hubei University)

**通讯引用:** 310 | [OpenAlex ID](https://openalex.org/A5101883757)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了首个针对USAS标签集的神经网络与混合规则/神经网络语义标注器，并通过英语银标训练数据实现跨语言评估。

**💡 创新点**

在USAS框架中首次使用纯神经网络标注器，构建将神经网络作为规则标注器回退的混合系统，并利用规则生成的银标数据在多语言环境下训练神经模型。

**🔧 技术方法**

采用BEM双编码器架构、Ettin/ MMBERT预训练语言模型、负采样训练、跨语言迁移以及PyMUSAS框架集成技术。

**📊 数据集**

使用约6.6M标注的英语银标语料、英语/威尔士语/爱尔兰语/芬兰语/中文人工标注集，并发布中文手工标注语料和银标数据。

**📈 对比分析**

通过Top‑1/Top‑5准确率比较规则、神经、混合三类模型，结果显示混合或神经模型在所有语言上均优于纯规则，中文与英语表现最佳，低资源语言相对落后。

**⚠️ 局限性**

需先有规则标注器生成银标数据，规则质量影响神经模型；低资源语言可用文本不足导致银标规模有限；未针对非英语语言细调模型。

---

## 358. PersonalAlign: Hierarchical Implicit Intent Alignment for Personalized GUI Agent with Long-Term User-Centric Records

**arXiv ID:** 2601.09636 | [PDF](https://arxiv.org/pdf/2601.09636v1)

**作者:** Yibo Lyu `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 28000 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PersonalAlign任务，构建了AndroidIntent基准，并设计HIM-Agent记忆框架，实现利用长期用户记录进行分层隐式意图对齐与主动协助。

**💡 创新点**

三大创新：1）定义隐式意图对齐的分层任务框架；2）采用层次过滤-验证策略构造含偏好与例程的AndroidIntent数据集；3）开发流式聚合与层次意图记忆的HIM-Agent，实现高效的长期上下文学习与主动建议。

**🔧 技术方法**

使用多模态LLM嵌入、动态时间规整、稀疏Jaccard相似度、时间与场景熵度量、层次过滤器、流式聚合原型记忆以及主动建议评估技术。

**📊 数据集**

基于Fingertip20K中91位用户两个月的Android交互记录构建的AndroidIntent数据集，包含775条偏好意图与215条例程意图。

**📈 对比分析**

与多款闭源（GPT‑5.1、Qwen3‑VL‑Max）和开源（UI‑TARS、GUI‑Owl、Qwen3‑VL）GUI代理进行对比，HIM‑Agent在执行CER下降15.7%、主动建议F1提升7.3%，整体优于检索式与LLM总结方法。

**⚠️ 局限性**

仅在Android Fingertip场景下验证，数据集规模有限，缺乏跨域泛化；在缺少足够历史记录的冷启动情境下，模型性能可能下降。

---

## 359. Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric

**arXiv ID:** 2601.09624 | [PDF](https://arxiv.org/pdf/2601.09624v1)

**作者:** Jiali Cheng `[一作]` (University of Massachusetts Lowell), Hadi Amiri `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 1374 | [OpenAlex ID](https://openalex.org/A5074007015)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了语言模型的机器退学（unlearning）难度，基于模型内部电路（circuit）提出了前置的连续难度度量CUD，并利用该度量构造了易退学和难退学的样本集，进一步分析了不同电路结构与退学效果的关系。

**💡 创新点**

创新点在于：①首次将电路分析作为退学难度的机理性量化工具；②提出可在退学前预估样本难度的连续度量；③通过电路的深浅、分布特征揭示易退学与难退学样本的内部机制，提供了对退学方法的解释与改进思路。

**🔧 技术方法**

主要技术包括：电路发现（EAP‑IG），基于电路的相似度计算（余弦、Jaccard、Hamming等），二层优化（选择易/难退学样本的锚点），以及对不同退学方法的实验评估。

**📊 数据集**

使用的数据集为：TOFU（作者个人信息去除任务，400个退学样本）和 MocieLens‑1M（用户‑项目推荐信息去除任务，500个退学样本）。

**📈 对比分析**

与七种退学方法（GradAscent, GradDiff, NPO, SimNPO, UNDIAL, E2UREC, RecEraser）比较时，CUD 选出的易退学集在所有方法上平均提升 3.3 分（p<10⁻³），而难退学集则使退学效能下降 10–18 分，且对保留性能和通用知识影响更大；实验表明 CUD 在不同相似度度量和损失函数下均保持稳健。

**⚠️ 局限性**

局限性：CUD 需要先进行电路发现，计算成本高，主要适用于离线评估和退学前预估，难以在训练或退学迭代中实时使用；未来需要开发更高效的近似或轻量化方法。

---

## 360. Full Disclosure, Less Trust? How the Level of Detail about AI Use in News Writing Affects Readers' Trust

**arXiv ID:** 2601.09620 | [PDF](https://arxiv.org/pdf/2601.09620v1)

**作者:** Pooja Prajod `[一作]` (Centrum Wiskunde en Informatica), Abdallah El Ali `[通讯]` (Centrum Wiskunde en Informatica)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过设计 3×2×2 的混合因子实验，研究三种 AI 披露程度（无披露、单行披露、详细披露）在两类新闻（政治、生活方式）和两种 AI 参与度（低、高）下对读者信任、源核查和订阅行为的影响。

**💡 创新点**

创新点在于：①将披露细节层级与 AI 参与度区分，而非仅关注全 AI 生成；②结合定量信任量表与源核查/订阅决策两种行为测量，探讨“透明度困境”的多维效应；③通过半结构访谈揭示读者偏好“详细披露”或“按需细节”设计，提供对策建议。

**🔧 技术方法**

使用 ChatGPT‑4o 生成低/高 AI 参与版本的新闻文本；采用 News Media Trust 问卷、token 计数源核查任务、订阅决策任务及眼动追踪；统计分析采用广义线性模型与 Benjamini‑Hochberg 校正。

**📊 数据集**

数据集为 6 篇新闻（3 政治、3 生活方式）来自 BBC、CNN、NOS 等主流媒体，分别加工成低 AI 与高 AI 两版本；实验样本为 40 名受试者（最终有效 34 人）来自学术机构。

**📈 对比分析**

比较方法：对比三种披露条件下的信任问卷得分、订阅率和源核查频次。结果显示：详细披露在生活方式新闻中显著降低信任和订阅率；单行披露与无披露在信任/订阅上无显著差异；所有披露条件均提高源核查行为。

**⚠️ 局限性**

局限性：①样本为学术机构成员，AI 熟悉度高，可能不具代表性；②仅研究文本新闻，未涉及图片或多媒体内容；③未考察披露位置、品牌标识等对信任的影响；④实验设计固定，无法检验其他披露形式（如细节按需）在更广泛人群中的效果。

---

## 361. SysPro: Reproducing System-level Concurrency Bugs from Bug Reports

**arXiv ID:** 2601.09616 | [PDF](https://arxiv.org/pdf/2601.09616v1)

**作者:** Tarannum Shaila Zaman `[一作]` (University of Maryland Baltimore County), Tingting Yu `[通讯]` (University of Connecticut)

**通讯引用:** 1244 | [OpenAlex ID](https://openalex.org/A5072070511)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一种基于bug报告自动化重现系统级并发缺陷的工具，能够从报告中提取系统调用名称、定位其源码位置、生成测试输入并通过动态二进制插桩实现缺陷复现。

**💡 创新点**

① 采用结构化信息检索与Apriori挖掘结合的方式，自动识别并定位导致并发缺陷的系统调用对；② 通过自然语言处理、信息检索与正则匹配统一生成缺陷复现所需的输入；③ 只依赖于bug报告，无需人工介入即可完成缺陷复现。

**🔧 技术方法**

自然语言处理（分词、停用词、词干化）、结构化信息检索（Gensim、TF‑IDF、srcML）、Apriori频繁项集挖掘、正则表达式、类别分区法、PIN动态插桩。

**📊 数据集**

基准数据集为24条真实系统级并发缺陷报告（共19个开源项目，17个独立程序），覆盖Linux核心工具如coreutils、mkdir、chmod等。

**📈 对比分析**

与基础IR、BLUiR以及无工具对照组比较，评估指标包括系统调用排名位置、MAP、召回率@K、复现成功率与尝试次数。实验表明，该工具在排名前20内能覆盖约90%的关键系统调用，MAP平均超过0.8，复现成功率达到~85%，平均仅需5-7次尝试，明显优于随机重现。

**⚠️ 局限性**

受限于bug报告的质量；需要已知触发进程集合；输入生成仍需人工或基于规则，未充分利用LLM；仅针对C/C++系统级程序，无法直接扩展到其他语言或普通并发缺陷。

---

## 362. Analyzing GitHub Issues and Pull Requests in nf-core Pipelines: Insights into nf-core Pipeline Repositories

**arXiv ID:** 2601.09612 | [PDF](https://arxiv.org/pdf/2601.09612v1)

**作者:** Khairul Alam `[一作]` (University of Saskatchewan), Banani Roy `[通讯]` (University of Saskatchewan)

**通讯引用:** 610 | [OpenAlex ID](https://openalex.org/A5015470184)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对nf‑core 125个管道的25,173条GitHub issue/PR进行主题建模与统计分析，探讨问题类型、管理方式与解决效率。

**💡 创新点**

首次系统性归纳nf‑core管道的讨论主题（共13类），并量化标签、代码块等特征对问题闭合率和解决时长的影响。

**🔧 技术方法**

采用BERTopic（基于Sentence‑Transformers、UMAP+HDBSCAN）进行主题挖掘，使用Wilcoxon秩和检验与Cohen效应量评估特征差异。

**📊 数据集**

基于GitHub API收集的25,173条issue/PR数据，涵盖125个活跃nf‑core管道的所有提交记录。

**📈 对比分析**

通过比较闭合率、平均/中位解决时长、标签与代码块使用比例等指标，发现约89.4%问题已闭合，其中60%自闭合，标签与代码块分别提升闭合率大/中效应；大多数问题在3天内得到解决。

**⚠️ 局限性**

局限在于仅分析GitHub平台数据，未覆盖Slack、邮件等讨论渠道；BERTopic主题取决于预处理与模型选择，可能存在主题偏差；难度评估仅依赖未解决比例与中位时间，未考虑评论深度或复杂性。

---

## 363. Technological Advances in Two Generations of Consumer-Grade VR Systems: Effects on User Experience and Task Performance

**arXiv ID:** 2601.09610 | [PDF](https://arxiv.org/pdf/2601.09610v1)

**作者:** Marie Luisa Fiedler `[一作]` (University of Würzburg), Marc Erich Latoschik `[通讯]` (University of Würzburg)

**通讯引用:** 6908 | [OpenAlex ID](https://openalex.org/A5069021763)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

比较了两代 HTC Vive 系统在完整、商业化配置下的用户体验与任务表现，结果显示技术进步并未显著提升主观体验或表现

**💡 创新点**

首次采用生态有效的“as‑is”系统配置与多任务设计，结合贝叶斯分析提供对技术升级效益的稳健评估

**🔧 技术方法**

使用 Unity、SteamVR、FinalIK 逆运动学以及 HTC Vive 与 Vive Pro 2 硬件进行全身追踪与沉浸式 VR 实验

**📊 数据集**

共招募 40 名受试者，完成 5 项任务，收集主观问卷（存在感、身体占有感、外观可信度等）与客观性能指标

**📈 对比分析**

采用 2×5 混合设计、Trimmed‑Means ANOVA、Mann‑Whitney U 和贝叶斯 ANOVA 进行比较，结果无显著系统效应，效应量极小

**⚠️ 局限性**

样本单一、受试者年龄与经验相对均匀、无法分离各硬件/软件子组件影响，且实验在现代软件环境下进行，限制了对旧硬件原始表现的复现

---

## 364. Sim2real Image Translation Enables Viewpoint-Robust Policies from Fixed-Camera Datasets

**arXiv ID:** 2601.09605 | [PDF](https://arxiv.org/pdf/2601.09605v1)

**作者:** Jeremiah Coholich `[一作]` (Institute of Robotics and Intelligent Machine), Zsolt Kira `[通讯]` (Institute of Robotics and Intelligent Machine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MANGO方法，利用少量固定摄像头真实数据和仿真多视角数据，进行无配对图像翻译，生成具有新视角的真实感图像，用于增强机器人视觉策略的视角鲁棒性。

**💡 创新点**

创新点在于：①基于分割的InfoNCE损失，确保对象边界保持；②改进的PatchNCE评分函数，抑制错误负样本；③高度正则化的PatchGAN判别器，支持未见视角的风格迁移；④将上述技术集成于轻量GAN框架，显著提升翻译质量。

**🔧 技术方法**

技术细节包括：GAN生成器（12M参数ResNet）、PatchGAN判别器（11M参数）、Segmentation InfoNCE、Modified PatchNCE、ResNet编码器、ACT（action chunking transformer）策略训练、模拟与真实图像的无配对对齐。

**📊 数据集**

数据集涵盖：约8,098张多视角仿真图像、约3,094张固定摄像头真实图像；用于验证的Robomimic、Mimicgen模拟任务；以及真实机器人演示图像（约35k张）。

**📈 对比分析**

与CUT、CycleGAN、域随机化、ZeroNVS/VISTA等基线进行对比，使用FID、LPIPS、政策成功率评估；MANGO在随机视角测试中FID降低23点，真实任务成功率提升至60%+，并且GPU时数比VISTA低2,700倍。

**⚠️ 局限性**

局限性：仍需少量真实摄像头数据；在3/4真实任务的偏移摄像头评估中未能完全超越VISTA；轻量模型在表达更复杂3D结构时可能受限。

---

## 365. Information Access of the Oppressed: A Problem-Posing Framework for Envisioning Emancipatory Information Access Platforms

**arXiv ID:** 2601.09600 | [PDF](https://arxiv.org/pdf/2601.09600v1)

**作者:** Bhaskar Mitra `[一作]` (Independent Researcher), Sireesh Gururaja `[通讯]` (Carnegie Mellon University)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5108855136)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文研究了在线信息获取平台在专制压力下的风险，并提出了基于 Paulo Freire 批判教育理论的“问题提出框架”，强调被压迫者的主体性与社区共建的参与式治理。

**💡 创新点**

创新点在于将批判教育的“问题提出”方法迁移到信息访问平台设计，提出从被压迫者视角出发、共同塑造技术的架构思路，并聚焦去中心化与民主治理的结合。

**🔧 技术方法**

未使用具体技术，仅提出概念框架与设计原则；若落地可借助去中心化协议、社区治理平台、开放源代码等技术实现。

**📊 数据集**

未使用任何数据集；研究以理论与案例分析为主。

**📈 对比分析**

无实验或性能对比；文章聚焦理论阐述与框架构建，未给出可量化评估。

**⚠️ 局限性**

局限性包括：概念化程度高、缺乏实现与实证评估、实际落地面临技术、治理与可持续性挑战、可能与现行法律与监管产生冲突。

---

## 366. Multimodal Signal Processing For Thermo-Visible-Lidar Fusion In Real-time 3D Semantic Mapping

**arXiv ID:** 2601.09578 | [PDF](https://arxiv.org/pdf/2601.09578v1)

**作者:** Jiajun Sun `[一作]` (Shenzhen University), Yue Ma `[通讯]` (Xi'an-Jiaotong Liverpool University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种实时的LiDAR-视觉-红外三模融合SLAM框架，实现热信息增强的3D语义地图生成。

**💡 创新点**

创新点在于基于像素级可见光与红外图像融合并投影至点云，实现热源语义层实时标注；同时提供无目标外参校准和多模冗余鲁棒性。

**🔧 技术方法**

采用LiDAR–IMU高精度定位、Zhang标定、像素级图像融合、热阈值检测、Voxel网格映射、非线性优化等技术。

**📊 数据集**

使用校园体育广场与食堂两大户外建筑的实测数据，包括可见光、红外及LiDAR扫描，环境温度25-33℃。

**📈 对比分析**

与现有方法比较，帧率>20 FPS、几何精度毫米级、检测范围>50m、3D语义支持、环境适应性全天气，性能优于传统单模或二模方法。

**⚠️ 局限性**

局限性在于对热源识别仍受环境温度、辐射不均影响，缺乏深度学习判别缺陷类型，且未在极端恶劣环境或大规模结构中验证。

---

## 367. Trustworthy Longitudinal Brain MRI Completion: A Deformation-Based Approach with KAN-Enhanced Diffusion Model

**arXiv ID:** 2601.09572 | [PDF](https://arxiv.org/pdf/2601.09572v1)

**作者:** Tianli Tao `[一作]` (King's College London), Le Zhang `[通讯]` (University of Birmingham)

**通讯引用:** 15816 | [OpenAlex ID](https://openalex.org/A5050809806)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

针对脑MRI纵向数据缺失，提出通过生成变形场而非直接像素生成的DF-DiffCom框架，实现可信的纵向完成。

**💡 创新点**

创新点在于使用KAN增强的扩散模型学习变形场，并设计可变时间信息增强模块F‑TIE，使模型既可信又适用于多模态。

**🔧 技术方法**

技术包括扩散概率模型、KAN（Kolmogorov‑Arnold Network）改进的U‑Net、空间变换网络、交叉注意力与多项损失。

**📊 数据集**

在OASIS‑3 T1w MRI数据集上训练与评估，使用152个受试者外的训练集做变形场与BAE模型。

**📈 对比分析**

与cGAN、DiffuseMorph、LoCI‑DiffCom、TADM等基线比较，PSNR最高25.52 dB，SSIM 0.845，明显优于对手。

**⚠️ 局限性**

局限在于仍需监督的变形场数据，未探索无监督/弱监督的变形学习。

---

## 368. Dialogue Telemetry: Turn-Level Instrumentation for Autonomous Information Gathering

**arXiv ID:** 2601.09570 | [PDF](https://arxiv.org/pdf/2601.09570v1)

**作者:** Dimitris Panagopoulos `[一作]` (Cranfield University), Weisi Guo `[通讯]` (Cranfield University)

**通讯引用:** 5871 | [OpenAlex ID](https://openalex.org/A5062362866)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Dialogue Telemetry（DT）框架，对任务导向信息收集对话在每个问答回合后生成进度估计（PE）和停滞指数（SI），并在基于 LLM 的 SAR 模拟环境中进行验证与强化学习（RL）策略的闭环集成。

**💡 创新点**

创新点包括：① 设计两种模型无关、可即时观测的信号；② 将信息论（EIG）与启发式缺失量相结合的 PE；③ 通过短窗口重复计数和语义相似度检测停滞的 SI；④ 将这些信号用于策略优化、奖励塑形和人机监控；⑤ 在无监督的 SAR 情境中实现了可解释的对话诊断。

**🔧 技术方法**

技术手段：信息论 EIG、语义嵌入相似度、短窗口重复检测、语义缺失量与信息量的混合公式、强化学习（PPO）策略训练、LLM 生成模拟问答、人工标注完成度评分。

**📊 数据集**

数据集：通过 LLM（如 Llama）在 SAR 任务下生成的 25+ 条回答，覆盖 8 个预定义信息类别；人工和 LLM 评分的完成度标签用于知识积累的量化。

**📈 对比分析**

对比方法：Baseline（无 DT）vs. Full‑DT（含 PE+SI）在两种终止条件下进行 RL 训练；Full‑DT 在标准终止下提升奖励、知识总量和完整类别数；在停滞终止下 Baseline 几乎不学习，Full‑DT 明显超越；SI 处罚在无停滞条件下产生负面效应，说明需根据场景调节。

**⚠️ 局限性**

局限性：仅在单一 SAR 模拟环境中验证，缺乏真实人类对话；仅考察策略选择而非提问表述；SI 仅捕捉交互模式，不提供因果诊断；依赖预先定义的任务 schema；对参数（窗口、阈值、惩罚系数）的敏感性未系统评估；未探究多模态或实时部署。

---

## 369. Further results on Minimal and Minimum Cylindrical Algebraic Decompositions

**arXiv ID:** 2601.09548 | [PDF](https://arxiv.org/pdf/2601.09548v1)

**作者:** Lucas Michel `[一作]` (Department One), Naïm Zénaïdi `[通讯]` (Department Two)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

本文未给出具体研究内容，仅包含示例文本。

**💡 创新点**

暂无创新点。

**🔧 技术方法**

暂无技术细节。

**📊 数据集**

暂无数据集信息。

**📈 对比分析**

未进行方法比较，未给出性能指标。

**⚠️ 局限性**

缺乏具体信息，难以评估限制。

---

## 370. SiliconHealth: A Complete Low-Cost Blockchain Healthcare Infrastructure for Resource-Constrained Regions Using Repurposed Bitcoin Mining ASICs

**arXiv ID:** 2601.09557 | [PDF](https://arxiv.org/pdf/2601.09557v1)

**作者:** Francisco Angulo de Lafuente `[一作]` (Independent Researcher), Nirmal Tej `[通讯]` (Independent Researcher)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究构建了一个基于被废弃比特币挖矿 ASIC 的四层区块链医疗网络，实现了在撒哈拉以南非洲等资源匮乏地区的低成本电子健康记录系统。

**💡 创新点**

核心创新包括：1）确定性硬件指纹（DHF）将 ASIC SHA‑256 运算转化为可验证的医疗证明；2）在医学影像中嵌入 Reed‑Solomon 最小有效位水印，达到30‑40% 损毁耐受；3）在离线环境下实现分层同步与 RAG（检索增强生成）问答；4）利用 ASIC 的 PUF 进行设备身份与安全。

**🔧 技术方法**

技术体系包括：Bitcoin ASIC 设备（Antminer S9/S19 Pro、Lucky Miner LV06/07）、SHA‑256 硬件指纹与 HKDF 会话密钥、Merkle 树结构、Reed‑Solomon (255,223) 水印、量化 4‑bit RAG 模型、太阳能电源与离线同步协议。

**📊 数据集**

实验采用约 100 张医学影像（X‑ray、超声等）进行水印验证、500 GH/s 的 Lucky Miner LV06 进行 DHF 证明测试、以及 7 天的太阳能供电模拟，验证了硬件稳定性与网络同步。

**📈 对比分析**

与 GPU（RTX 3090、RTX 3080）对比，LV06 ASIC 的能效达 2.93 MH/W（比 RTX 3090 高 8.5 倍），DHF 证明平均耗时 12.5 s、100% 验证成功；影像水印在 30‑40% 损毁下仍可恢复 94‑98%。

**⚠️ 局限性**

局限性包括：1）吞吐量仅满足 50‑200 病人/日的农村流量，城市医院仍需增设节点；2）RAG 语言覆盖仅限主流非洲语，未支持少数民族语言；3）对 ASIC 长期供应与监管认证缺乏确定性；4）在高连通性环境下的性能评估不足。

---

## 371. On the Error Probability of RPA Decoding of Reed-Muller Codes over BMS Channels

**arXiv ID:** 2601.09581 | [PDF](https://arxiv.org/pdf/2601.09581v1)

**作者:** Dorsa Fathollahi `[一作]` (Stanford University), V. Lalitha `[通讯]` (IIIT Hyderabad)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了递归投影聚合（RPA）解码器在任意二进制记忆无关对称（BMS）信道上对Reed-Muller（RM）码的误码概率上界，进一步证明了在低阶（loglog n级）RM码上误码概率随码长趋于零；

**💡 创新点**

创新点在于：①将RPA投影操作与极化码的信道组合等价，从而在任意BMS信道上使用统一的上界；②利用一阶RM码的ML误码上界得到递归基准；③得到与BSC上类似的误码极限，但适用于更广泛的信道；

**🔧 技术方法**

使用的技术包括：RPA递归投影聚合算法、极化信道组合、Bhattacharyya参数分析、联合界、Markov不等式以及对RM码结构的组合论分析；

**📊 数据集**

该工作为理论研究，无需具体数据集；

**📈 对比分析**

与之前仅针对BSC的分析相比，本文在所有非退化BMS信道上给出相同阶的误码极限（r≈log m），理论上误码概率可趋近于零；

**⚠️ 局限性**

局限性：证明仅适用于单轮RPA解码有效的情况；多轮迭代下的相关性未被处理；对极限上界的精确度及实际实现复杂度仍待进一步研究。

---

## 372. From Prompt to Protocol: Fast Charging Batteries with Large Language Models

**arXiv ID:** 2601.09626 | [PDF](https://arxiv.org/pdf/2601.09626v1)

**作者:** Ge Lei `[一作]` (Imperial College), Samuel J. Cooper `[通讯]` (Imperial College)

**通讯引用:** 54662 | [OpenAlex ID](https://openalex.org/A5089223699)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文提出两种基于大型语言模型的闭环优化框架（Prompt-to-Optimizer P2O 与 Prompt-to-Protocol P2P），用于设计高效、耐久的电池快充协议。

**💡 创新点**

创新点在于：① 通过 LLM 生成可直接执行的协议或网络结构，突破传统手工限定搜索空间的局限；② 结合两级闭环（外层 LLM 进化、内层 SAASBO 参数优化）与单层 LLM 生成，实现低样本、高效搜索；③ 在快充实验预算受限的高成本情景下，仍能提升约 4.2% 的容量保持率。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑5）生成代码/协议；Sparse Axis‑Aligned Subspace Bayesian Optimization（SAASBO）用于无梯度参数优化；物理电池模型 PyBaMM（DFN + 热/退化耦合）进行仿真；进化算法与随机搜索作为基线。

**📊 数据集**

使用公开的 Li‑ion 电池数据集（如参考文献所述）在 PyBaMM 里配置 DFN 及退化模型，并通过加速退化参数提升仿真效率。

**📈 对比分析**

与贝叶斯优化、遗传算法、随机搜索及多步常电流（CCCV）基线比较，P2O 与 P2P 在相同实验预算下均超过基线约 4.2% 的 SOH，P2P 在单循环中实现与 P2O 最终性能相当且样本更少。

**⚠️ 局限性**

局限性包括：① 依赖高质量仿真，真实实验验证尚未完成；② LLM 对数值精度有限，需人工校正；③ 需精心设计提示语以体现约束，过度依赖语言表述；④ 内循环 SAASBO 对参数空间维度仍有限，极大网络不易优化。

---

## 373. The Promptware Kill Chain: How Prompt Injections Gradually Evolved Into a Multi-Step Malware

**arXiv ID:** 2601.09625 | [PDF](https://arxiv.org/pdf/2601.09625v1)

**作者:** Ben Nassi `[一作]` (Tel Aviv University), Oleg Brodt `[通讯]` (Ben Gurion University of the Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Promptware 死亡链模型，系统化描述了 LLM 系统的多阶段攻击流程，涵盖初始接入、权限提升、持久化、横向移动与目标实现。

**💡 创新点**

创新点在于将 Prompt Injection 从单一攻击视角转变为完整的杀伤链，构建了与传统恶意软件相似的五阶段模型，并为 AI 安全与网络安全社区提供统一的术语与分析框架。

**🔧 技术方法**

使用的技术主要是理论模型构建、文献综述与案例映射；对已公开的 Prompt Injection 与 Jailbreaking 技术进行归类，并将其嵌入杀伤链阶段；并通过示例攻击说明链条的可行性。

**📊 数据集**

未使用公开数据集，而是依托已有研究与公开案例（如 Morris II、AgentFlayer 等）进行说明和验证。

**📈 对比分析**

方法评估采用案例对比的方式，展示多阶段攻击实例完整落地，未给出量化性能指标；通过对比传统单阶段攻击与完整链条的复杂度与危害性，证明模型的有效性与现实意义。

**⚠️ 局限性**

局限性包括：①链条边界模糊，攻击者可跳过或合并阶段；②缺乏针对每个阶段的量化防御评估；③模型未涵盖所有潜在攻击变体；④对实际系统的防御效果需要进一步实验验证。

---

## 374. GRCF: Two-Stage Groupwise Ranking and Calibration Framework for Multimodal Sentiment Analysis

**arXiv ID:** 2601.09606 | [PDF](https://arxiv.org/pdf/2601.09606v1)

**作者:** Manning Gao `[一作]` (South China Normal University), Sijie Mai `[通讯]` (South China Normal University)

**通讯引用:** 1596 | [OpenAlex ID](https://openalex.org/A5010270301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出两阶段的 Group‑wise Ranking and Calibration Framework（GRCF），用于多模态情感分析，在回归任务中实现更稳健的顺序学习和绝对值校准，且可迁移至二分类任务。

**💡 创新点**

核心创新点包括：1）结合 GRPO 的优势加权（advantage‑weighted）机制，对难以排序的样本赋予更高权重；2）采用重叠分组（overlap groups）计算动态边距（dynamic margin），使相邻情感区间的边距与语义距离成正比；3）在两阶段训练中先构建排序结构，再用 MAE 校准绝对分数，兼顾秩序与量化准确性。

**🔧 技术方法**

技术手段包括：多模态编码器（文本用 DeBERTa，音频/视觉用注意力池化）；多模态融合为 Concatenate‑then‑MLP；Stage‑1 的 Group‑Aware Ranking Loss（动态 margin + GRPO 权重）+ Distribution Regularization + Boundary Loss；Stage‑2 的 MAE‑driven fine‑tuning；使用 AdamW、混合精度、cosine 学习率退火等训练技巧。

**📊 数据集**

回归任务使用 CMU‑MOSI、CMU‑MOSEI、CH‑SIMS v2；二分类任务扩展到 MUStARD（讽刺检测）和 UR‑FUNNY v2（幽默检测）。

**📈 对比分析**

与多种基线（TFN、MFN、MMIM、DMF、TMF 等）在 MAE、Correlation、Accuracy、F1 等指标对比，GRCF 在 CMU‑MOSI、CMU‑MOSEI、CH‑SIMS v2 上刷新了绝大多数指标的 SOTA；在 MUStARD 与 UR‑FUNNY v2 的 Acc2 也高于现有方法，展示了良好的迁移性和鲁棒性。

**⚠️ 局限性**

局限性包括：① 需要手动设定重叠分组数量与阈值，过于复杂的分组可能导致过拟合或收敛慢；② 对标签分布较为单一（如 CH‑SIMS v2）时动态 margin 与 GRPO 的优势不明显；③ 在二分类任务中仍受稀疏奖励瓶颈限制，难以充分发挥优势加权机制。

---

## 375. Improving CMA-ES Convergence Speed, Efficiency, and Reliability in Noisy Robot Optimization Problems

**arXiv ID:** 2601.09594 | [PDF](https://arxiv.org/pdf/2601.09594v1)

**作者:** Russell M. Martin `[一作]` (Stanford University), Steven H. Collins `[通讯]` (Stanford University)

**通讯引用:** 12116 | [OpenAlex ID](https://openalex.org/A5037382800)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出一种自适应采样的CMA‑ES（AS‑CMA），在噪声环境下自动为每个候选解分配测量时间，以提高实验优化的收敛速度、效率与可靠性。

**💡 创新点**

创新点在于：①基于候选解间距离与局部斜率估计所需的排序信噪比，动态分配采样时长；②无需重新测量已评估的候选；③可在不同采样预算下自动调整采样策略；④仅需少量先验信息（噪声模型、粗略成本上下限）即可使用。

**🔧 技术方法**

主要技术包括：CMA‑ES进化算法、基于距离-成本差的自适应采样规则、噪声模型ℰ(t)、Spearman相关性评估排序精度、仿真与实验评估。

**📊 数据集**

使用四个仿真测试函数（4‑D ankle exoskeleton 代谢成本、Rosenbrock、Levy、20‑D Sphere）以及一次真实踝部外骨骼实验（单位人）。

**📈 对比分析**

与多种基线比较：固定采样时间的CMA‑ES、带KL‑KG动态重采样的CMA‑ES、静态采样的贝叶斯优化。AS‑CMA在所有四个测试函数上收敛更快（粗/细阈值分别快24–65%），累计成本更低（29–76%），并保持高可靠性；在复杂景观下优于贝叶斯优化，KL‑KG在极大维度上略胜一筹但总体性能逊于AS‑CMA。

**⚠️ 局限性**

局限性包括：需要先验噪声-时间模型ℰ(t)；若测量噪声不随采样时长变化则无效；对极低噪声或无噪声场景不必要；在噪声模型估计不准确时性能受影响；实验验证规模有限，需更多真实人类实验以验证统计显著性。

---

## 376. Show, don't tell -- Providing Visual Error Feedback for Handwritten Documents

**arXiv ID:** 2601.09586 | [PDF](https://arxiv.org/pdf/2601.09586v1)

**作者:** Said Yasin `[一作]` (Center of Advanced Technology for Assisted Learning and Predictive Analytics), Torsten Zesch `[通讯]` (Center of Advanced Technology for Assisted Learning and Predictive Analytics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了手写文本的检测、排序、识别与错误反馈生成的挑战，并实现了基于文档摄像头的实时原型。

**💡 创新点**

创新点在于综合比较模块化与端到端手写识别系统在反馈生成中的性能，并提出多维度评估指标与改进议程；同时公开了原型实现。

**🔧 技术方法**

采用了PGNet端到端、Tesseract、EasyOCR以及Google Cloud Vision的检测/识别技术，并结合SpellChecker识别错误，使用IoU、CER、NSFD、BLEU等指标进行评估。

**📊 数据集**

实验使用了IAM和Imgur5K两大英文手写数据集。

**📈 对比分析**

通过对四种系统在检测精度、识别CER、排序NSFD/ BLEU等指标上的对比，发现无论模块化还是端到端均无法满足课堂使用的质量要求，整体性能需进一步提升。

**⚠️ 局限性**

局限性包括仅在单词级别做反馈、缺乏字符级检测、未覆盖多语言、对手写风格差异敏感，且评估忽略未检测到的单词导致误差低估。

---

## 377. MLIR-Forge: A Modular Framework for Language Smiths

**arXiv ID:** 2601.09583 | [PDF](https://arxiv.org/pdf/2601.09583v1)

**作者:** Berke Ates `[一作]` (ETH Zurich), Torsten Hoefler `[通讯]` (ETH Zurich)

**通讯引用:** 12390 | [OpenAlex ID](https://openalex.org/A5026990786)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 MLIR-Forge 框架，支持快速构建基于 MLIR 的随机程序生成器，拆分为语言规范（OpGen/TypeGen）和程序构造两大模块；

**💡 创新点**

创新点在于把 IR 生成器拆解为可复用的“拼图块”，仅需实现操作/类型生成器即可自动生成随机程序，极大降低开发成本并支持跨 IR 重用；

**🔧 技术方法**

使用 MLIR 作为底层框架，结合 ODS、CLI、IRBuilder 等技术实现自动化构造；

**📊 数据集**

在评估中分别生成了 MLIR、SDFG（DaCe）和 WebAssembly 程序，累计产生 100k+ 程序，发现 9、15、774 个 Bug 组；

**📈 对比分析**

通过差分 fuzz 测试与 AFL++/FuzzyFlow 等工具比较，生成速度快（<40 ms/20 KB，<36 MB 内存），并且与传统工具相比，开发时间仅需一周，Bug 发现率高；

**⚠️ 局限性**

局限性包括：需手动实现安全检查（如除零）、对类型和区域约束的支持有限、无法完全避免翻译器错误，以及对仅支持 MLIR 方言的扩展性受限。

---

## 378. OpenVoxel: Training-Free Grouping and Captioning Voxels for Open-Vocabulary 3D Scene Understanding

**arXiv ID:** 2601.09575 | [PDF](https://arxiv.org/pdf/2601.09575v1)

**作者:** Sheng-Yu Huang `[一作]` (NVIDIA), Cheng Sun `[通讯]` (NVIDIA)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5081389607)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种无训练的稀疏体素分组与字幕化框架，用于在多视角图像重建的SVR模型上完成开放词汇3D场景理解和指代分割。

**💡 创新点**

核心创新是：①不需要任何文本特征学习，直接用VLM与多模态LLM生成结构化、可读的实例字幕；②利用这些字幕进行文本‑文本检索实现对复杂句子查询的准确定位；③采用一次性渲染即可完成体素分组，显著提升速度。

**🔧 技术方法**

技术栈包括：SVR（稀疏体素重建）、SAM2（2D实例分割）、Describe‑Anything‑Model（生成初始字幕）、多模态LLM（如Qwen3‑VL‑8B）做字幕规范化与检索。

**📊 数据集**

实验使用LeRF数据集的多个子集：LeRF‑Mask、LeRF‑OVS、Ref‑LeRF，涵盖多场景、不同查询类型。

**📈 对比分析**

在RES任务上，方法比ReferSplat提升了约13.2%（单场景）或17.9%（复现版），在OVS任务上也保持竞争力；训练时间仅3分钟，推理低于1秒，速度比现有SOTA快10倍以上。

**⚠️ 局限性**

局限性：依赖SAM2的分割质量；目前仅针对已重建的多视角场景，无法直接处理单视角或大型开放世界数据；字幕生成与检索仍受LLM上下文窗口限制。

---

## 379. Hot-Start from Pixels: Low-Resolution Visual Tokens for Chinese Language Modeling

**arXiv ID:** 2601.09566 | [PDF](https://arxiv.org/pdf/2601.09566v1)

**作者:** Shuyang Xiang `[一作]` (Independent Researcher), Hao Guan `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究低分辨率中文字符图像是否能替代离散索引作为语言模型的输入，验证视觉结构在字符级预测中的可行性并揭示其早期学习优势

**💡 创新点**

首次证明即使是8×8像素的灰度字符图像也能达到与基准索引模型相当的预测准确率，并揭示视觉输入在训练初期的“hot‑start”效应与对局部信息的鲁棒性

**🔧 技术方法**

采用轻量级视觉编码器（ResNet+Vision Adapter）对字符图像进行编码，随后输入标准GPT‑2‑small风格解码器进行下一个字符预测；使用交叉熵损失与AdamW优化；对图像进行裁剪与分辨率实验

**📊 数据集**

使用THUCNews中文新闻语料库（约12.8M字符，100K序列，长度128）进行训练与评估；通过四种输入模式（索引、完整图像、80%裁剪、50%裁剪）进行对比

**📈 对比分析**

与传统索引基线在同一模型规模（≈117M参数）下进行比较；视觉模型在8×8分辨率下准确率为39.21%，略高于索引基线39.10%；在训练早期（仅0.4%数据）视觉模型已达12.34%准确率，超过索引基线的5.84%；在分辨率、裁剪实验中均保持稳定性能

**⚠️ 局限性**

仅使用标准字体渲染，未测试手写或变体字形；模型规模较小，未验证大模型表现；未考虑多字符或段落级视觉输入；实验仅聚焦中文，未验证对其他表意文字的通用性

---

## 380. MM-BRIGHT: A Multi-Task Multimodal Benchmark for Reasoning-Intensive Retrieval

**arXiv ID:** 2601.09562 | [PDF](https://arxiv.org/pdf/2601.09562v1)

**作者:** Abdelrahman Abdallah `[一作]` (University of Innsbruck), Hyun-Soo Kang `[通讯]` (Chungbuk National University)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5018214137)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MM-BRIGHT 多模态推理检索基准，并对 18 种现有检索模型在 4 种任务上进行评测。

**💡 创新点**

创新点在于将多模态查询与推理密集检索结合，填补了仅关注文本或表面多模态的空白，并提供了 29 个技术领域、4 个任务层级的完整评测框架。

**🔧 技术方法**

采用 StackExchange 真实问答数据、人工 + GPT‑4o 复核、nDCG@10 评估，并对比稀疏检索、稠密检索、推理增强检索和多模态对比模型；同时开展查询重构与端到端 RAG 实验。

**📊 数据集**

使用了 2,803 条包含文本+图像的真实技术查询、29 个领域、7,621 张经过人工验证的相关图像，构成正负样本集合。

**📈 对比分析**

通过 nDCG@10 对四个任务进行量化比较：任务1（文本检索）最优 DiVeR 得 32.2；任务2（多模态检索）最优 Nomic‑Vision 得 27.6；任务3（图像检索）最优 GME‑2B 得 45.6；任务4（多模态文档检索）最优 CLIP 得 28.0；所有模型均低于常规基准，显示显著提升空间。

**⚠️ 局限性**

局限性包括：多模态模型在关键视觉信息下性能下降；图像往往对检索无益甚至负面影响；模型对图像重要性判断不精准；基准尚缺乏更细粒度的视觉推理评估。

---

## 381. Benchmarking Post-Training Quantization of Large Language Models under Microscaling Floating Point Formats

**arXiv ID:** 2601.09555 | [PDF](https://arxiv.org/pdf/2601.09555v1)

**作者:** Manyi Zhang `[一作]` (Huawei Technologies), Xianzhi Yu `[通讯]` (Huawei Technologies)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5021890706)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了微尺度浮点（MXFP）格式下的后训练量化（PTQ）方法，覆盖多种LLM和多模态模型，并在七种PTQ算法、十五个评测基准以及三类模型家族上进行实验。

**💡 创新点**

创新点在于首次将PTQ方法与MXFP量化进行全面对比，发现MXFP8可实现近乎无损压缩，而MXFP4则存在显著误差，并通过分析量化因子误差和提出预缩放策略来提升4-bit性能；同时揭示不同算法在MXFP下的兼容性差异，提供了针对MXFP的实用量化设计指引。

**🔧 技术方法**

使用了基于通道变换、误差补偿、旋转变换与仿射变换四大类的PTQ技术；采用块级浮点（E4M3、E2M1等）MXFP格式；借助microxcaling库实现量化；并结合多种评测指标（精度恢复率、困惑度等）进行比较。

**📊 数据集**

使用的评测数据集包括WikiText2（困惑度）、非推理任务零样本（PIQA、Winogrande、Hellaswag、ARC-Easy、ARC-Challenge）、推理任务（MATH-500、AIME24、AIME25）以及多模态任务（OCRBench、MMBench、MMBench^CN、TextVQA、ChartQA、MME、MMMU）。

**📈 对比分析**

通过对不同量化位宽（W8A8、W4A8、W4A4）和PTQ算法的比较，发现误差补偿与仿射变换在MXFP下表现最佳，旋转变换在MXFP4上效果较差；RTN仍是强基线；MXFP8实现近乎无损，MXFP4则风险较高，混合精度策略在多模态模型中尤为有效。

**⚠️ 局限性**

研究仅涵盖7B/8B规模模型，未对更大规模（如30B+）或NVIDIA NVFP等其他微尺度浮点格式进行评估，故结果对更大模型或不同微尺度设计的推广性仍需进一步验证。

---

## 382. Examining DOM Coordinate Effectiveness For Page Segmentation

**arXiv ID:** 2601.09543 | [PDF](https://arxiv.org/pdf/2601.09543v1)

**作者:** Jason Carpenter `[一作]` (University of Minnesota), Zhi-Li Zhang `[通讯]` (University of Minnesota)

**通讯引用:** 11869 | [OpenAlex ID](https://openalex.org/A5100622097)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `729e5870-4135-47f5-97f2-e3974d07b5dc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了DOM坐标在网页分割中的效果，并评估不同向量与聚类算法组合对分割精度的影响。

**💡 创新点**

证明视觉坐标不如DOM坐标，发现简单单坐标向量往往优于复杂向量，并提出通过匹配向量、聚类算法与页面来显著提升分割准确率。

**🔧 技术方法**

基于DOM坐标构建聚类向量，使用聚类算法（如k‑means等）对网页进行分割，并系统比较不同向量与算法的组合。

**📊 数据集**

未公开具体数据集，实验使用多页面的DOM结构样本进行评估。

**📈 对比分析**

通过与视觉坐标向量对比，评估了各向量在不同聚类算法下的分割准确率。结果显示DOM坐标平均提升20‑30%，简单向量在68.2%高分案例中占优，最佳匹配方案可达74%准确率，比基线提升约20%。

**⚠️ 局限性**

实验规模有限，未覆盖所有网页类型；匹配向量与算法的过程需要人工挑选，缺乏自动化策略；对大规模动态网页的适用性尚未评估。

---

## 383. Omni-R1: Towards the Unified Generative Paradigm for Multimodal Reasoning

**arXiv ID:** 2601.09536 | [PDF](https://arxiv.org/pdf/2601.09536v1)

**作者:** Dongjie Cheng `[一作]` (Hong Kong Polytechnic University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 11482 | [OpenAlex ID](https://openalex.org/A5100408983)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出统一生成式多模态推理框架 Omni‑R1 与其零注释版 Omni‑R1‑Zero，支持多种推理技能（如放大、定位、绘线、标记、视觉预测）并生成中间功能图像；

**💡 创新点**

①统一生成式多模态推理思路，②通过感知对齐损失与感知校准奖励实现功能图像的稳定生成；③零注释版本利用逐步可视化从文本链式思维自动生成多模态推理轨迹，消除昂贵的交互式标注；

**🔧 技术方法**

两阶段训练：感知对齐监督微调（PeSFT）+感知校准强化学习（PeRPO）；PPO、感知对齐损失、感知奖励、步骤可视化生成；使用 VQ‑VAE 代码表做感知对齐；

**📊 数据集**

Omni‑Bench（四类 Uni‑Task 切片：Natural‑Scene Perception、Structured‑Image、Diagrammatic Math、Vision‑Operational），以及通用多模任务集 MME、MM‑Vet、V*、POPE、MMVP、BLINK；

**📈 对比分析**

对比 Anole（Base）和 Zebra‑CoT 基线，在 Omni‑Bench 上各任务均超越基线，尤其 Vision‑Operational、Diagrammatic 等多步骤任务提升显著；在通用基准上 Omni‑R1 与 Omni‑R1‑Zero 也均优于基线，并且零注释版在多数指标上与监督版相当或更好；

**⚠️ 局限性**

仍需少量交互式标注或自生成轨迹，功能图像生成质量受限；感知奖励参数敏感，且在极端复杂或多步骤任务中泛化性尚待验证；

---

## 384. Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server

**arXiv ID:** 2601.09531 | [PDF](https://arxiv.org/pdf/2601.09531v1)

**作者:** Yue Yao `[一作]` (Shandong University), Tom Gedeon `[通讯]` (Curtin University)

**通讯引用:** 7556 | [OpenAlex ID](https://openalex.org/A5030379402)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于层次化数据服务器与双向匹配的训练集搜索框架（BMM），用于在无法实时标注的目标域中通过匹配源域模式来构建近似目标分布的训练集；

**💡 创新点**

创新点在于：①将数据服务器构造为层次化聚类结构，使不同粒度的源模式可灵活匹配目标模式；②使用基于Fréchet Inception Distance的双向匹配（Hungarian算法）实现一一对应的模式匹配，避免多重匹配导致的数据多样性下降；③将数据层次化与匹配耦合，实现对目标域模式的精准覆盖，并可与现有无监督域自适应方法协同提升性能；

**🔧 技术方法**

核心技术包括：图像特征提取（预训练的Imagenet模型）、平衡k‑means聚类、层次聚类（凝聚式），基于FID的成本矩阵构建双边图，Hungarian最小成本匹配；在后续训练中使用常见目标检测/重识别模型（IDE、RetinaNet）及可选的伪标签方法（MMT、AT）；

**📊 数据集**

源数据服务器由七大公开数据集构成（ADE20K、COCO、BDD、CityScapes、DETRAC、Kitti、VOC），共176,491张图；目标数据集包括人类重识别（AlicePerson、Market）、车辆重识别（AliceVehicle、VeRi）以及检测任务中的ExDark、Region100；

**📈 对比分析**

与随机采样、贪心搜索、SnP、TL;DR、CCDR等现有训练集搜索方法相比，BMM在FID、rank‑1、mAP等指标上均有显著提升；在人重识别上，FID从约80降至51，rank‑1从33%提升至49%；在车辆检测上，mAP由约40%提升至56%；与伪标签联合使用时，可进一步提升到近80%等；

**⚠️ 局限性**

局限性包括：①需要先在源服务器上完成一次耗时的层次聚类，尽管是一次性操作；②匹配过程中仅考虑特征分布相似度（FID），未考虑语义标签或场景细节的潜在差异；③对极端少量目标样本或极端异构源数据时，匹配效果可能受限；④在高度动态或多模态目标域中，单次匹配可能不足以覆盖所有模式。

---

## 385. SAM3-DMS: Decoupled Memory Selection for Multi-target Video Segmentation of SAM3

**arXiv ID:** 2601.09699 | [PDF](https://arxiv.org/pdf/2601.09699v1)

**作者:** Ruiqi Shen `[一作]` (Fudan University), Henghui Ding `[通讯]` (Fudan University)

**通讯引用:** 3857 | [OpenAlex ID](https://openalex.org/A5036631624)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了训练无关的 SAM3‑DMS 方法，在多目标视频分割中采用单独的记忆更新策略，避免因全局平均分数导致的身份漂移。

**💡 创新点**

创新点在于将 SAM3 的群体级记忆选择解耦为按对象独立评估，使用每个目标自身的分割置信度与帧级可见度相乘，形成个体化的更新阈值。

**🔧 技术方法**

技术包括基于 SAM3 的记忆银行、分割得分与可见度计算、阈值判别的记忆更新，以及不需要额外训练的实现。

**📊 数据集**

使用了 SA‑Co/VEval、YTVIS19/21、OVIS、BDD100K、SA‑V 和 MOSEv2 等公开多目标分割/跟踪数据集进行评测。

**📈 对比分析**

与原 SAM3 以及单独逐目标推理进行对比，SAM3‑DMS 在所有指标（如 cgF1、pHOTA、mAP 等）上均有提升，尤其在目标密集场景下提升更为显著，且与单独推理的性能差距大幅缩小。

**⚠️ 局限性**

局限性在于对 PVS 的上限并未进一步提升，且方法仍依赖原始 SAM3 的框架，未能在某些极端快速运动或极端遮挡情形下完全消除偶发身份漂移。

---

## 386. Complexity Thresholds for the Constrained Colored Token Swapping Problem

**arXiv ID:** 2601.09681 | [PDF](https://arxiv.org/pdf/2601.09681v1)

**作者:** Davide Bilò `[一作]` (University of L'Aquila), Andrea Martinelli `[通讯]` (University of L'Aquila)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了在受限交换图下的彩色代币交换问题，证明当交换图含有四元路径时问题为 PSPACE‑完整，且当交换图为星形或仅三种颜色时可多项式求解；

**💡 创新点**

首次给出从三种颜色到四种颜色的复杂度阈值，完成了先前工作中未解的范围，并提出了对星形交换图的一般多项式算法，扩展了Pebble Motion理论；

**🔧 技术方法**

使用约束逻辑与图嵌入技术构造逼近和简化子图（边、AND/OR 节点）、等价类划分与可达性分析、以及Pebble Motion的可解性判定方法；

**📊 数据集**

无实验数据集，全部为理论证明与复杂度分析；

**📈 对比分析**

无实验对比，性能通过多项式时间算法与PSPACE‑完整证明的理论复杂度进行说明；

**⚠️ 局限性**

算法仅适用于交换图为星形或三色情况，未对更一般P₄‑自由图给出完整结论，且对实际大规模实例的效率未做实验验证。

---

## 387. Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning

**arXiv ID:** 2601.09667 | [PDF](https://arxiv.org/pdf/2601.09667v1)

**作者:** Zhiyuan Hu `[一作]` (Massachusetts Institute of Technology), Hae Won Park `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2619 | [OpenAlex ID](https://openalex.org/A5102016303)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Multi-Agent Test-Time Reinforcement Learning（MATTRL）框架，利用在推理时检索并注入结构化文本经验，增强多代理协同推理的稳健性与适应性。

**💡 创新点**

创新点：1）在不更新模型权重的前提下，通过文本经验实现测试时强化学习；2）结合多种信用分配方案（平均、差分奖励、Shapley近似）从多代理对话中构造经验池；3）设计了三阶段多专家协作流程（团队组建、经验增强式共识对话、报告合成），实现跨领域的高效推理。

**🔧 技术方法**

技术：LLM多代理协作（GPT‑5作为主体）、检索增强对话（语义向量检索+模板化提示）、信用分配与经验构建（差分奖励、Shapley近似）、经验池检索与注入、评估判定器（LLM判定、奖励映射）以及基准评测脚本。

**📊 数据集**

数据集：医学领域的 RareBench（罕见疾病诊断）、数学领域的 HLE（人类最终考试）以及教育领域的 SuperGPQA（教学问答），并使用 GPT‑5 作为核心模型。

**📈 对比分析**

与单代理、基线多代理、RareAgents、Adaptive Router 等方法对比。MATTRL 在医学任务中 Hit@1/10 分别提升至 0.39/0.75（平均提升 3.67%），在数学任务中准确率从 0.27 提升至 0.36（+0.09），在教育任务中后测准确率达 0.77（+0.33），均显著优于对照组；差分奖励在经验构建中表现最佳，提供了精度与效率的最佳平衡。

**⚠️ 局限性**

局限：推理时需进行多轮代理对话，计算量和延迟随团队规模及探索预算增长；经验池随着时间扩展，可能出现过时、重复或错误经验导致漂移。

---

## 388. SCE-SLAM: Scale-Consistent Monocular SLAM via Scene Coordinate Embeddings

**arXiv ID:** 2601.09665 | [PDF](https://arxiv.org/pdf/2601.09665v1)

**作者:** Yuchen Wu `[一作]` (Beihang University), Xiao Bai `[通讯]` (Beihang University)

**通讯引用:** 6989 | [OpenAlex ID](https://openalex.org/A5101790836)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于场景坐标嵌入的单目SLAM系统SCE‑SLAM，实现尺度一致的实时导航。

**💡 创新点**

创新点在于将尺度信息编码进每个图像补丁的嵌入，并通过几何引导的尺度传播和场景坐标束调整，形成自我监督的尺度记忆。

**🔧 技术方法**

采用双分支架构（光流分支+坐标分支），DINOv3特征、几何调制注意力、GRU递归更新、坐标束调整等技术。

**📊 数据集**

在KITTI、Waymo、Virtual KITTI以及4Seasons等大型公开数据集上进行实验。

**📈 对比分析**

与DROID‑SLAM、DPV‑SLAM++等最新方法对比，平均ATE下降8.36 m，标准差大幅降低，实时率36 FPS，表现优于现有方法。

**⚠️ 局限性**

局限性包括对特征匹配的依赖、在极端光照或大尺度场景下仍可能出现尺度漂移，并且缺乏完整的闭环检测机制。

---

## 389. Exploring Fine-Tuning for Tabular Foundation Models

**arXiv ID:** 2601.09654 | [PDF](https://arxiv.org/pdf/2601.09654v1)

**作者:** Aditya Tanna `[一作]` (Lexsi Labs), Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对表格基础模型（TFM）在零样本推理、元学习、全参数微调（SFT）和参数高效微调（PEFT）四种适配策略进行大规模评估，比较其在性能、校准与公平性上的表现。

**💡 创新点**

系统性揭示了不同适配策略对各种模型和数据集特征（规模、平衡度、维度）的依赖性，提出了“零样本推理往往更稳健”的经验性结论，并给出了针对数据规模与维度的实用微调指南。

**🔧 技术方法**

使用Transformer‑based TFM（TabPFN、TabICL、OrionMSP、OrionBiX、TabDPT、Mitra）与传统基线（XGBoost、LightGBM、CatBoost、Random Forest），采用Meta‑Learning、SFT、LoRA‑PEFT 等技术；评价指标包括 ACC、F1、ECE、MCE、Brier、SPD、EOD、EOpD。

**📊 数据集**

评估数据集包括三大 benchmark（TALENT 155 组、OpenML‑CC18 63 组、TabZilla 27 组）以及 9 个含敏感属性的公平性数据集（Adult、German Credit、COMPAS 等）。

**📈 对比分析**

通过在统一拆分、统一 GPU 环境下对六种 TFM 与四类基线进行 4 种适配策略的对比，结果显示：零样本推理在大多数数据规模下表现最优；元学习在不平衡数据上稳健提升；SFT 在中等规模或宽特征集上有局部提升，但常导致性能与校准下降；PEFT 在保持高效的同时可恢复大部分 SFT 改进，尤其对 TabDPT 与 OrionMSP 有显著帮助。

**⚠️ 局限性**

仅覆盖二分类/多分类任务；部分模型不支持 PEFT；公平性评估受限于手工定义的敏感属性与公共子集；实验仅限于可用的公共数据集，缺乏对连续预测或更复杂场景的验证。

---

## 390. AquaFeat+: an Underwater Vision Learning-based Enhancement Method for Object Detection, Classification, and Tracking

**arXiv ID:** 2601.09652 | [PDF](https://arxiv.org/pdf/2601.09652v1)

**作者:** Emanuel da Costa Silva `[一作]` (Universidade Federal do Rio Grande), Paulo Lilles Jorge Drews-Jr `[通讯]` (Universidade Federal do Rio Grande)

**通讯引用:** 3733 | [OpenAlex ID](https://openalex.org/A5037041582)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了AquaFeat+模型，作为可插拔的特征增强模块，用于改进水下视频中的目标检测、分类与跟踪。

**💡 创新点**

创新点在于端到端任务导向的特征增强，包含颜色校正、U-FEN、双路全局注意力（GSAM）与自适应残差输出，直接优化最终任务损失而非仅提升视觉美观。

**🔧 技术方法**

采用颜色校正、U-Net结构（SpecialConv）、双路全局注意力模块（GSAM）、多尺度融合与自适应残差等技术，并结合YOLOv8/10、YOLOv11、ByteTrack等主干网络。

**📊 数据集**

使用FishTrack23数据集，经过帧抽取、类别聚类后用于目标检测、分类与跟踪实验。

**📈 对比分析**

与AquaFeat、FeatEnHancer、YOLOv8s、ConvNeXt等方法对比，AquaFeat+在检测F1≈0.688、mAP≈0.556、跟踪HOTA≈55.2、IDF1≈68.1等指标上表现最佳。

**⚠️ 局限性**

局限在于对非鱼类目标识别仍有限，残差输出可能引入误差，且在极端低照度或大尺度场景下的鲁棒性尚待进一步验证。

---

## 391. LiteEmbed: Adapting CLIP to Rare Classes

**arXiv ID:** 2601.09661 | [PDF](https://arxiv.org/pdf/2601.09661v1)

**作者:** Aishwarya Agarwal `[一作]` (International Institute of Information Technology Hyderabad), Vineet Gandhi `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用少量（3–5张）样例图像，通过子空间引导优化微调CLIP的文本嵌入，使新出现或稀缺类别的文本表示能够直接替换原始嵌入，在任何使用CLIP文本特征的任务（分类、检索、分割、检测、生成）中无任务特定训练即可获得更好性能。

**💡 创新点**

①将CLIP文本嵌入空间通过PCA分解为粗粒度（高方差）和细粒度（低方差）子空间；②设计粗对齐损失让新嵌入沿粗子空间保持与语义邻域一致；③设计细分离损失让新嵌入沿细子空间与视觉相似的负类保持距离；④仅对占位符token进行微调，保持CLIP编码器冻结，实现轻量级、可插拔的适配。

**🔧 技术方法**

CLIP (ViT‑B/16/32)、PCA、子空间对齐与分离损失、Adam优化、少量样本微调、对原始文本嵌入的直接替换。

**📊 数据集**

公开数据集：Indian Food Images、Korean Celebrities、TV100；新构建的 NOVA benchmark（Game Characters、Indian Singers、Indian Actors、Fashion Outfits、Landmarks）；下游任务数据集：UECFood100（分割、检测）、CustomConcept101（文本到图像生成）。

**📈 对比分析**

与多种基线（Zero‑Shot CLIP、CoOp、CoCoOp、MaPLe、CLIP‑Adapter、TIP‑Adapter‑F、DiffTPT、PromptAlign、C‑TPT、DynaPrompt 等）在 8 个多样化数据集的 4‑shot 分类任务中对比，平均提升约 14% 以上、对比零射击提升 35% 以上；在连续学习场景中比第二名高 32%；在检索、分割、检测、生成等下游任务中均实现显著精度或 IoU 的提升。

**⚠️ 局限性**

• 仍需手工或基于 LLM 构造粗语义邻域和细视觉负样本；• PCA 子空间划分的假设可能在不同模型或领域中失效；• 对极少样本（1‑shot）性能仍有限；• 仅支持单类或少数新类的适配，难以扩展到大规模新类别；• 微调嵌入仍在推理阶段耗时，对实时应用有一定开销。

---

## 392. Value-Aware Numerical Representations for Transformer Language Models

**arXiv ID:** 2601.09706 | [PDF](https://arxiv.org/pdf/2601.09706v1)

**作者:** Andreea Dutulescu `[一作]` (National University of Science and Technology POLITEHNICA Bucharest), Mihai Dascalu `[通讯]` (National University of Science and Technology POLITEHNICA Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种值感知的数字编码机制，在Transformer输入中插入前缀< num >令其嵌入直接由数字值决定，从而提升模型对数字大小的感知。

**💡 创新点**

创新点在于通过专门的前缀token直接把数值映射为连续嵌入，实现了数值与表面形式的解耦，并通过投影对齐训练与推理的一致性。

**🔧 技术方法**

技术包含可学习的数值编码器（MLP或RNN+辅助特征）、投影层、三项损失（教师强制、投影语言建模、重建余弦损失）以及两遍前向传播训练。

**📊 数据集**

使用NUPA基准集，对整数、浮点、分数和科学记数法等四种数字格式的算术、比较、转换等任务进行评估。

**📈 对比分析**

与标准Transformer和Numerologic基线比较，所提NumValue模型在Exact Match上提升约3个百分点，数字匹配率和长度误差亦显著下降，尤其在较长数字上表现更稳健。

**⚠️ 局限性**

局限性包括仅在从零开始训练的原型模型上验证，未对大规模预训练模型做迁移；训练需要两次前向传播导致成本升高；评测聚焦单一数值基准，未涵盖更复杂推理场景。

---

## 393. ShortCoder: Knowledge-Augmented Syntax Optimization for Token-Efficient Code Generation

**arXiv ID:** 2601.09703 | [PDF](https://arxiv.org/pdf/2601.09703v1)

**作者:** Sicong Liu `[一作]` (Sun Yat-sen University), Yanlin Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 4376 | [OpenAlex ID](https://openalex.org/A5100350715)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于知识注入的语法优化框架，设计10条AST保持的Python简化规则，并通过规则+LLM混合合成生成简化代码对，随后使用LoRA微调模型，使其能够在无需额外提示的情况下自然生成更短更可读的代码。

**💡 创新点**

①10条语法简化规则实现18.1% token压缩；②混合规则+LLM的数据合成方法构建高质量简化代码对；③通过LoRA微调将简洁意识注入模型，形成零提示下的短代码生成能力。

**🔧 技术方法**

AST转换、规则驱动与LLM提示式代码重写、LoRA参数高效微调、Token/成本等效率指标评估以及HumanEval/MBPP等基准测试。

**📊 数据集**

828对<original_code,simplified_code>数据集（来自MBPP与自制合成），HumanEval与HumanEvalPlus用于功能与效率评测。

**📈 对比分析**

与CodeLlama、DeepSeek-Coder、CodeGen等基线在HumanEval的pass@1/10/100、生成token数进行对比，微调模型实现18.1%–37.8% token压缩、pass@100达0.967，且在效率和可读性上优于提示式方法。

**⚠️ 局限性**

仅针对Python，规则可扩展性受限；简化后仍需验证边界条件，模型在大规模输入或复杂逻辑时可能产生非最优实现；数据集规模有限，缺乏跨语言评测。

---

## 394. Efficient Camera-Controlled Video Generation of Static Scenes via Sparse Diffusion and 3D Rendering

**arXiv ID:** 2601.09697 | [PDF](https://arxiv.org/pdf/2601.09697v1)

**作者:** Jieying Chen `[一作]` (University of Cambridge), Ayush Tewari `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种先用扩散模型生成稀疏关键帧，再通过3D高斯散点重建与渲染完整视频的高效摄像机控制视频生成框架；

**💡 创新点**

创新点在于：①利用视频冗余只生成关键帧；②自适应预测关键帧密度以平衡质量与计算；③将3D Gaussian Splatting作为渲染后端，显著提高速度与几何一致性；

**🔧 技术方法**

核心技术包括扩散模型（History‑Guided + Diffusion Forcing）、Transformer‑based关键帧密度预测、AnySplat 3D Gaussian Splatting 重建与渲染；

**📊 数据集**

使用 RealEstate10k 与 DL3DV 两个摄像机控制视频数据集进行训练与评估；

**📈 对比分析**

与 History‑Guided Video Diffusion (HG)、Voyager 以及 2D 插值方法 FILM/RIFE 对比；在 DL3DV 上实现 40×+ 的加速（20‑秒 30fps 视频仅需 16.21 s），在 RE10K 上 20×+ 加速，同时 FID/FVD 指标优于基线；

**⚠️ 局限性**

仅适用于静态场景；长周期摄像机轨迹可能出现漂移；3D 生成的高频细节不如纯扩散模型，需进一步提升 3D 重建质量。

---

## 395. Empathy Applicability Modeling for General Health Queries

**arXiv ID:** 2601.09696 | [PDF](https://arxiv.org/pdf/2601.09696v1)

**作者:** Shan Randhawa `[一作]` (University of Michigan), Mustafa Naseem `[通讯]` (University of Michigan)

**通讯引用:** 319 | [OpenAlex ID](https://openalex.org/A5085824530)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 Empathy Applicability Framework (EAF)，对患者查询进行情感与认知层面是否需要同情的预测，并构建标注基准。

**💡 创新点**

创新点在于将同情标注从仅在回应后进行的反应式方法转为前置预判方法，定义可适用与不可适用两维，并提供多子类别与理论依据。

**🔧 技术方法**

使用 RoBERTa‑base Transformer 进行二分类预测，比较零-shot、随机、总是适用/不适用等基线，以及传统 TF‑IDF+LR/SVM；同时利用 GPT‑4o 进行自动标注并进行概念对齐分析。

**📊 数据集**

数据来源为 9,500 条 HealthCareMagic 与 iCliniq 的患者查询，其中 1,500 条被双重标注（人类与 GPT），其余 8,000 条仅由 GPT 标注，形成人类一致集与自主集。

**📈 对比分析**

在人类一致测试集上，Transformer 的 macro‑F1 在 EA、IA 任务中分别达到 0.92 与 0.87，显著优于基线；在 GPT 标注训练的自闭集上亦能取得 0.85/0.77 的 macro‑F1。

**⚠️ 局限性**

限制包括标注者数量有限且缺乏临床背景、仅使用 GPT‑4o 作为自动标注，未覆盖多模型与不同文化背景；以及人类与 GPT 采用不同子类别取舍导致比较不完全。

---

## 396. Image2Garment: Simulation-ready Garment Generation from a Single Image

**arXiv ID:** 2601.09658 | [PDF](https://arxiv.org/pdf/2601.09658v1)

**作者:** Selim Emir Can `[一作]` (Stanford University), Gordon Wetzstein `[通讯]` (Stanford University)

**通讯引用:** 17748 | [OpenAlex ID](https://openalex.org/A5014044649)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单图像生成可仿真服装的前向流水线，先预测布料属性再映射到物理参数；

**💡 创新点**

通过将逆问题拆解为可观测的材质属性预测和属性到物理参数的映射，实现了数据高效、无迭代优化；

**🔧 技术方法**

采用微调的视觉-语言模型（Qwen‑2.5VL）预测布料成分和结构，使用随机森林回归器估计物理参数；

**📊 数据集**

构建了FTAG（纤维属性标签）和T2P（标签到物理）两个新数据集，分别用于属性标注和属性到参数的监督；

**📈 对比分析**

与ChatGarment、AIpparel等基线对比，结果在三维形状和二维图像指标上均优于随机材质或现有方法，且推理速度更快；

**⚠️ 局限性**

仅支持单层服装，且对单图像中难以观测的细微材质特征敏感，未来可扩展至多层服装或更丰富的视觉信号。

---

## 397. Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning

**arXiv ID:** 2601.09708 | [PDF](https://arxiv.org/pdf/2601.09708v1)

**作者:** Chi-Pin Huang `[一作]` (NVIDIA), Fu-En Yang `[通讯]` (NVIDIA)

**通讯引用:** 388 | [OpenAlex ID](https://openalex.org/A5032964198)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

Fast-ThinkAct 通过将长文本链式推理压缩为可解释的连续潜在表示，实现了高效的视觉-语言-动作推理与控制。

**💡 创新点**

创新点在于：① 引入基于奖励偏好的潜在推理蒸馏，压缩思考过程；② 对齐教师与学生的视觉轨迹潜在，传递空间规划能力；③ 结合蒸馏后的潜在表示训练扩散 Transformer 低层动作策略，显著提升实时性与规划质量。

**🔧 技术方法**

使用了：预训练视觉语言模型（如 Qwen2.5-VL 3B）、GRPO 强化学习、对抗性偏好蒸馏（DPO 风格）、多模态潜在表示与空间令牌、扩散 Transformer 策略网络、以及自回归潜在生成器。

**📊 数据集**

主要数据集包括：LIBERO、SimplerEnv、RoboTwin2.0（机器人操作），EgoPlan-Bench2、RoboVQA、OpenEQA（嵌入式推理），以及 PixMo、RoboFAC、ShareRobot、EgoPlan、Video-R1 等多模态 QA 与视觉推理数据集。

**📈 对比分析**

与 OpenVLA、π_0、CoT-VLA、ThinkAct、MolmoAct、RDT 等基线对比，Fast-ThinkAct 在所有 LIBERO 子任务、SimplerEnv、RoboTwin2.0 上取得最高成功率，并在推理速度上相较 ThinkAct 下降 89.3%（延迟大幅降低），同时保持甚至提升了长序列规划、少样本适应和失败恢复性能。

**⚠️ 局限性**

限制在于：潜在推理的可解释性依赖预训练 LLM，容易出现幻觉；在推理阶段未使用可视化潜在，仅在训练时用于解释；未来需引入更鲁棒的 grounding 与幻觉抑制机制。

---

## 398. Disentangling Task Conflicts in Multi-Task LoRA via Orthogonal Gradient Projection

**arXiv ID:** 2601.09684 | [PDF](https://arxiv.org/pdf/2601.09684v1)

**作者:** Ziyu Yang `[一作]` (Shanghai University), Xiangquan Yang `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对多任务LoRA的梯度正交投影方法Ortho-LoRA，缓解负迁移

**💡 创新点**

创新点在于将梯度正交投影专门应用于LoRA的两块低秩矩阵A、B，兼顾结构特性

**🔧 技术方法**

使用LoRA、梯度投影(PCGrad思路)、AdamW优化器、Transformer基础模型

**📊 数据集**

使用GLUE基准任务：MNLI、QQP、SST-2

**📈 对比分析**

与单任务LoRA（上限）和共享Joint-LoRA比较，Ortho-LoRA在GLUE平均分89.6，恢复约80%性能差距，几乎与单任务相当

**⚠️ 局限性**

需要为每个任务单独计算梯度，训练成本按任务数提升；仅在LoRA和GLUE上验证，缺乏对其他PEFT方法或更大任务的通用性验证

---

## 399. Automating Supply Chain Disruption Monitoring via an Agentic AI Approach

**arXiv ID:** 2601.09680 | [PDF](https://arxiv.org/pdf/2601.09680v1)

**作者:** Sara AlMahri `[一作]` (University of Cambridge), Alexandra Brintrup `[通讯]` (The Alan Turing Institute)

**通讯引用:** 3802 | [OpenAlex ID](https://openalex.org/A5075872953)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于代理式AI的自动化供应链中断监测框架，能够从新闻等非结构化信息中自动检测、映射、评估风险并生成行动方案。

**💡 创新点**

创新点在于：1）最小监督的多智能体架构，利用LLM实现端到端闭环；2）将自然语言理解与图数据库查询、确定性风险计算等模块无缝协同；3）在三层以上供应链层级实现可操作的风险可视化与供应商替代方案建议。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT‑4o）驱动的代理、检索增强的知识图谱查询（Neo4j + Cypher）、确定性工具（风险计算、产品搜索API）、prompt工程与多代理协同执行。

**📊 数据集**

使用的数据集为：基于真实供应链知识图谱（6,596个节点、23,888条关系）构建的多层网络；30个人工合成的中断情景（10个每家汽车厂，涵盖五类中断）；以及俄罗斯‑乌克兰冲突的真实新闻文本。

**📈 对比分析**

与行业基准（多日人工分析）对比，框架在30个情景上实现F1 0.962–0.991，平均响应时长3.83 分钟，成本$0.0836，显著缩短三位数时间且保持高准确率。

**⚠️ 局限性**

局限性包括：需预先构建完整的多层供应链图谱；目前仅支持批处理，未实现实时流式监测；仅在汽车行业验证，缺乏跨行业与大规模验证；对图谱的动态更新与高并发性能未做压力测试。

---

## 400. How well LLM-based test generation techniques perform with newer LLM versions?

**arXiv ID:** 2601.09695 | [PDF](https://arxiv.org/pdf/2601.09695v1)

**作者:** Michael Konstantinou `[一作]` (SnT, University of Luxembourg), Mike Papadakis `[通讯]` (SnT, University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了多种基于大型语言模型（LLM）的单元测试生成工具在更新后的LLM版本下的表现，并提出了一个纯LLM的简易基线。

**💡 创新点**

创新点在于证明随着LLM能力提升，单纯的Prompt即可超越现有复杂工具，并提出混合层级的生成策略以降低API调用量。

**🔧 技术方法**

采用了Prompting、检索增强、代码覆盖反馈等技术，并对GPT‑4o‑mini、Llama 3.3 70B、DeepSeek V3等模型进行实验。

**📊 数据集**

使用了包含393个类、3,657个方法的六个完整Java开源项目（如Binance、AWS、Apple等）及GitBug‑Java数据集。

**📈 对比分析**

通过与HITS、SymPrompt、TestSpark、CoverUp等四大工具对比，结果显示Plain‑LLM在行覆盖率提升约17.7%、分支覆盖率提升约19.8%、变异得分提升约20.9%，同时API调用量相当或更少。

**⚠️ 局限性**

主要局限在于生成的测试代码中约一半无法编译或通过，且LLM常生成多余或占位类，导致有效测试率低。

---

## 401. Routing with Generated Data: Annotation-Free LLM Skill Estimation and Expert Selection

**arXiv ID:** 2601.09692 | [PDF](https://arxiv.org/pdf/2601.09692v1)

**作者:** Tianyi Niu `[一作]` (UNC Chapel Hill), Mohit Bansal `[通讯]` (UNC Chapel Hill)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Routing with Generated Data（RGD）设置，研究在缺乏真实标签时利用生成数据进行 LLM 路由，并提出基于共识评分和层次聚类的无标签路由方法 CASCAL。

**💡 创新点**

创新点在于：①引入 RGD 这一无标注数据的路由框架；②利用模型共识自评估准确性；③通过层次聚类挖掘模型细粒度技能子集；④展示生成器质量对路由的影响并通过过滤提升弱生成器的数据质量。

**🔧 技术方法**

核心技术包括：LLM 生成器生成查询-答案对；模型响应聚合得到的共识评分；基于嵌入的层次聚类识别技能中心；以及在推理时以最近的技能中心选择并投票的路由策略。

**📊 数据集**

实验数据集包括 MMLU-Pro、MedMCQA、SuperGPQA、BigBench‑ExtraHard；使用两组模型池（大模型 20B+ 与小模型 <10B）共 12 个 LLM。

**📈 对比分析**

与多种基线（Top‑1/3、随机、问答路由器如 MLE、Cluster‑based、弱监督等）对比，CASCAL 在弱生成器场景下的平均准确率提升约 4–5% 以上，尤其在大模型池上表现出色，远优于传统问答路由器的显著性能下降。

**⚠️ 局限性**

局限性包括：依赖生成器产生的查询仍可能不足以区分极强模型；共识投票假设多数正确且对多答案场景表现受限；过滤机制手工设定阈值，需进一步自动化；在极小或无差异的模型池上仍可能难以发现细粒度技能。

---

## 402. Counting and Entropy Bounds for Structure-Avoiding Spatially-Coupled LDPC Constructions

**arXiv ID:** 2601.09674 | [PDF](https://arxiv.org/pdf/2601.09674v1)

**作者:** Lei Huang `[一作]` (Shandong University), Lei Huang `[通讯]` (Shandong University)

**通讯引用:** 1959 | [OpenAlex ID](https://openalex.org/A5100784430)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文针对结构避免的空间耦合LDPC（SC-LDPC）码的设计，提出了量化的可行设计空间计数与熵上界；

**💡 创新点**

创新点在于：①将结构避免问题建模为约束满足问题并利用量化Clique Lovász Local Lemma（CLLL）得到可行解数下界；②通过Rényi熵界定Moser–Tardos（MT）重采样算法能产生的多样化输出；③给出4‑cycle消除的闭式特殊化；

**🔧 技术方法**

使用技术包括：CLLL、MT重采样、Rényi熵与独立多项式估计、行/列置换等等价变换、量化依赖图分析；

**📊 数据集**

研究主要基于理论推导与仿真验证，无专门实验数据集，所用参数为基矩阵尺寸(γ,κ)、耦合记忆m、提升阶Z等；

**📈 对比分析**

通过与传统非构造存在性证明和现有的随机/启发式设计方法比较，表明在满足LLL条件下可行解数和输出多样性均有严格下界，理论上可选码数远大于单纯搜索；

**⚠️ 局限性**

局限性在于：①需要满足LLL的概率上界，限制了可用的耦合记忆和提升阶；②熵下界可能保守，实际多样性可能更高；③仅对短环（如4‑cycle）给出闭式，其他更复杂结构需进一步研究。

---

## 403. Diagonalization Without Relativization A Closer Look at the Baker-Gill-Solovay Theorem

**arXiv ID:** 2601.09702 | [PDF](https://arxiv.org/pdf/2601.09702v1)

**作者:** Baruch Garcia `[一作]` `[通讯]` (University of Texas), Baruch Garcia (University of Texas)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

通过“半相对化”对称化技术，提出一种不需要相对化、代数化或自然证明的P≠NP证明框架，借助接受问题oracle实现R与RE、P与NP的分离；

**💡 创新点**

创新点在于引入半相对化概念，利用仅含接受问题oracle的对角线法，突破三大障碍（相对化、代数化、自然证明），首次实现理论上对P≠NP的非相对化证明；

**🔧 技术方法**

主要技术包括传统对角线法、Cook–Levin归约、Tseitin变换以及对等价问题oracle（EQ_TM）的构造，构建多级“多项式时间接受问题”与Circuit‑SAT/3‑CNF‑SAT的归约链；

**📊 数据集**

本研究为理论探索，无实验数据集，完全基于抽象构造与归约证明；

**📈 对比分析**

通过理论归约展示P≠NP的正确性，并与传统相对化/代数化方法对比，证明其在突破三大障碍方面的优势，未涉及可测量的性能指标；

**⚠️ 局限性**

局限性在于证明仍为非构造性，缺乏可实现的算法与实际应用示例，且未对具体问题提供可操作的解决方案。

---

## 404. COMPOSE: Hypergraph Cover Optimization for Multi-view 3D Human Pose Estimation

**arXiv ID:** 2601.09698 | [PDF](https://arxiv.org/pdf/2601.09698v1)

**作者:** Tony Danjun Wang `[一作]` (Technical University of Munich), Lennart Bastian `[通讯]` (Technical University of Munich)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5018052917)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4`

**🎯 论文内容**

论文未提供具体内容，因此无法总结做了什么。

**💡 创新点**

论文未提供具体内容，因此无法总结创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法总结使用的技术。

**📊 数据集**

论文未提供具体内容，因此无法总结使用的数据集。

**📈 对比分析**

论文未提供具体内容，因此无法总结比较的方法和性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法总结限制因素。

---

## 405. LLMs can Compress LLMs: Adaptive Pruning by Agents

**arXiv ID:** 2601.09694 | [PDF](https://arxiv.org/pdf/2601.09694v1)

**作者:** Sai Varun Kodathala `[一作]` (Sports Vision), Rakesh Vunnam `[通讯]` (Vizworld)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型（LLM）作为智能剪枝代理的自适应剪枝框架，对Qwen3-4B和Qwen3-8B模型在约45%稀疏率下进行无训练后剪枝，显著保留事实知识与推理能力。

**💡 创新点**

创新点在于：①将LLM作为自适应剪枝决策者，利用层级敏感度和梯度信息的z分数化归一化进行动态层选择；②引入自我反思机制，让代理通过迭代反馈不断优化剪枝策略；③使用检查点回滚在出现过度困惑率时自动恢复模型。

**🔧 技术方法**

核心技术包括：Wanda式权重-激活重要性度量、梯度重要性评分、z分数标准化、基于gemini-3-flash-preview的LLM代理决策、回滚阈值控制和自我反思循环。

**📊 数据集**

使用的评估数据集有：C4（用于采样激活和梯度）、WikiText-2（语言建模困惑率评估）、MMLU（5-shot推理能力评估）以及FreebaseQA（事实知识保留评估）。

**📈 对比分析**

与传统结构化剪枝（2:4、4:8 N:M模式）比较时，agent-guided剪枝在MMLU准确率提高56%（相对4:8基线），FreebaseQA事实知识保留提升19倍，困惑率提升仅为基线的33.7%/90.3%，且回滚率低于10%。

**⚠️ 局限性**

局限性包括：①仍需高算力进行层级敏感度统计与LLM推理；②对不同硬件体系结构的加速效果尚未充分验证；③回滚阈值和自我反思策略的超参数仍需经验调优。

---

## 406. Contrastive Geometric Learning Unlocks Unified Structure- and Ligand-Based Drug Design

**arXiv ID:** 2601.09693 | [PDF](https://arxiv.org/pdf/2601.09693v1)

**作者:** Lisa Schneckenreiter `[一作]` (Johannes Kepler University), Günter Klambauer `[通讯]` (Johannes Kepler University)

**通讯引用:** 7501 | [OpenAlex ID](https://openalex.org/A5079632405)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出ConGLUDe，一种统一结构与配体信息的对比几何学习模型，用于药物设计任务。

**💡 创新点**

创新点在于将可预测绑定口袋的几何编码器嵌入到对比学习框架，实现无需预定义口袋的联合训练和配体条件口袋预测。

**🔧 技术方法**

采用基于VN-EGNN的几何蛋白质编码器、简单的MLP配体编码器、InfoNCE对比损失以及三轴对齐，结合对比学习和几何损失。

**📊 数据集**

使用PDBBind结构数据、MERGED（PubChem、BindingDB、ChEMBL）配体活性数据，评估时使用DUD‑E、LIT‑PCBA、COACH420、HOLO4K、PDBbind、ASD、Kinobeads等数据集。

**📈 对比分析**

与DrugCLIP、DrugHash、S^2Drug、LigUnity、HypSeek、SPRINT、DiffDock等基线比较，在零样本虚拟筛选、目标捕捉和配体条件口袋选择上均取得或匹配最优结果，尤其在无口袋信息的零样本虚拟筛选上显著优于对比学习和传统方法。

**⚠️ 局限性**

局限性包括对预测结构或极度多样化蛋白的鲁棒性未知，无法直接处理无目标生物测定，缺乏对接结果，且对所有osteric口袋的识别仍受限于VN‑EGNN的口袋检测。

---

## 407. DeepResearchEval: An Automated Framework for Deep Research Task Construction and Agentic Evaluation

**arXiv ID:** 2601.09688 | [PDF](https://arxiv.org/pdf/2601.09688v1)

**作者:** Yibo Wang `[一作]` (Nanyang Technological University), Lidong Bing `[通讯]` (Infinity Lab, Shanda Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DeepResearchEval框架，实现了自动化深度研究任务构建与评估；

**💡 创新点**

创新点在于基于人物角色的任务生成、两阶段过滤确保多源检索需求、以及自适应点式质量评估与主动事实核查；

**🔧 技术方法**

采用大语言模型（Gemini‑2.5‑Pro、GPT‑5-mini）、MiroFlow、Google Serper API等技术实现评估流水线；

**📊 数据集**

使用自动生成的100个高质量任务（涵盖10个领域）及其对应的900份系统生成报告做为数据集；

**📈 对比分析**

与9个主流深度研究系统对比，Gemini‑2.5‑Pro在质量评分8.51/10和事实准确率≈87%表现最佳，系统间在覆盖、洞察、指令跟随等维度存在显著差异；

**⚠️ 局限性**

限制包括英语为中心的任务与证据来源、评估过程成本高、缺乏多语言支持及实时可扩展性。

---

## 408. Progress on the Courtade-Kumar Conjecture: Optimal High-Noise Entropy Bounds and Generalized Coordinate-wise Mutual Information

**arXiv ID:** 2601.09679 | [PDF](https://arxiv.org/pdf/2601.09679v1)

**作者:** Adel Javanmard `[一作]` (University of Southern California), David P. Woodruff `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8898 | [OpenAlex ID](https://openalex.org/A5102861589)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文针对 Courtade‑Kumar conjecture 研究了噪声下布尔函数的信息量，证明了无论偏置与否，函数输出与噪声输入坐标的互信息和都被 1‑H(α) 上界；并在高噪声区间给出了最优 O(λ²) 的熵展开误差，推导出线性 Fourier 能量集中结果；

**💡 创新点**

创新点在于：① 解决了 Courtade‑Kumar 关于坐标互信息和的开放问题，适用于任意布尔函数；② 在高噪声 regime 下突破性地将误差降至 O(λ²)，实现最优熵界；③ 由此获得了最紧凑的 Fourier 级别 1 能量集中上界，显著扩大了 conjecture 的验证范围；

**🔧 技术方法**

主要技术包括 Fourier 级联分析、凸优化与极值点判定、Minkowski 与 hypercontractivity 的高级运用、Taylor 展开与高阶矩估计，以及对 1‑d 压缩算子对互信息的影响证明；

**📊 数据集**

本研究为理论工作，无需实验数据集，全部基于数学证明与解析推导；

**📈 对比分析**

通过与先前 O(λ^{4/3}) 误差分析和已知的高噪声极限结果对比，证明新方法在 λ 较小的范围内给出更严格的上界，理论上证明了更宽范围内的 conjecture 成立；

**⚠️ 局限性**

局限性在于：仍未能完全证明整个 Courtade‑Kumar conjecture，尤其在低噪声或中等噪声区间仍未得到完全验证；另外结果主要适用于布尔函数，扩展到更一般的离散或连续输入域尚需进一步研究。

---

## 409. STEP3-VL-10B Technical Report

**arXiv ID:** 2601.09668 | [PDF](https://arxiv.org/pdf/2601.09668v1)

**作者:** Ailin Huang `[一作]` (StepFun), Zheng Ge `[通讯]` (StepFun)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一款10B参数的多模态基础模型，兼顾视觉感知、复杂推理与人类对齐，并通过大规模预训练与强化学习显著提升性能。

**💡 创新点**

创新点包括：
- 将语言对齐的感知编码器（PE‑lang）与Qwen3‑8B解码器耦合，形成高效视觉‑语言协同；
- 通过1.2T高质量多模态语料一次性预训练，避免多阶段训练瓶颈；
- 采用PPO+GAE的RLVR+RLHF两阶段强化学习，并引入PaCoRe（并行协调推理）在推理阶段放大计算资源，突破传统序列推理限制；
- 设计可分离的可验证奖励与非可验证偏好奖励，兼顾任务可评估性与人类偏好。

**🔧 技术方法**

核心技术：
- 统一单阶段全解冻预训练（AdamW）；
- 多模态数据构造与自监督检索；
- PPO+GAE强化学习；
- PaCoRe并行协调推理；
- 细粒度奖励系统（感知奖励、模型校验、偏好模型、行为正则化）。

**📊 数据集**

使用了约1.2T多模态token，包括：
- 15M K‑12与高等教育题目；
- 10M图像 OCR 与30M合成OCR；
- 400M定位与计数样本；
- 23M GUI交互与轨迹；
- 多源图像‑文本对（LAION、COYO、BLIP‑CCS等）及合成数据；
- 文档 OCR 与代码生成数据等。

**📈 对比分析**

与7‑10B开源模型（GLM‑4.6V‑Flash、Qwen3‑VL‑Thinking、InternVL‑3.5等）对比，在多模态与文本基准上均处于榜首；与106B/235B开源模型以及Gemini‑2.5‑Pro、Seed‑1.5‑VL对比，保持竞争力甚至在PaCoRe模式下超越部分基准；典型成绩包括 MathVision 75.95%，MMMU 80.11%，AIME‑2025 94.43%，MMStar 77.48%。

**⚠️ 局限性**

局限与挑战：
- 强化学习阶段耗费大量计算资源，难以在更大规模或更快迭代中使用；
- PaCoRe虽提升性能，但推理时长与显存需求显著增加，限制了部署场景；
- 仍缺乏真实物理交互与动态视频建模，导致在实际机器人或嵌入式任务中的表现受限；
- 对罕见视觉概念与长尾领域的覆盖率依赖于人工标注与合成数据，可能存在偏差。

---

## 410. Self-Supervised Animal Identification for Long Videos

**arXiv ID:** 2601.09663 | [PDF](https://arxiv.org/pdf/2601.09663v1)

**作者:** Xuyang Fang `[一作]` (University of Bristol), Neill Campbell `[通讯]` (University of Bristol)

**通讯引用:** 2215 | [OpenAlex ID](https://openalex.org/A5109857567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种资源高效的自监督方法，用于在已知固定个体数的单段动物视频中识别个体，无需人工标注；

**💡 创新点**

创新点在于将个体识别重构为全局聚类任务，利用Hungarian算法在批次内动态生成伪标签，并采用可学习温度的二元交叉熵损失，显著降低显存需求；

**🔧 技术方法**

主要技术包括基于预训练冻结backbone的特征提取、双视图增强、相似度矩阵与掩码构造、Hungarian匹配、Binary Cross Entropy自监督学习和K‑Means聚类；

**📊 数据集**

实验使用3D‑POP鸽子数据集（10只鸽子两段视频）和8‑calves牧场摄像数据集（8头小牛）；

**📈 对比分析**

与SimCLR、MoCo以及基于监督的基准相比，本文方法在<1 GB显存下实现>97%+准确率，超过监督基准且显著节省显存；

**⚠️ 局限性**

局限性包括对固定个体数的假设、仅处理单视频场景、对极长时间序列的时序一致性仍待改进。

---

