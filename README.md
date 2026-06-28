# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-28 | 今日论文总数: 616

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Hot AI in Cold Space: Thermal-Crosstalk-Aware Scheduling for Sustainable Orbital AI Clusters

**arXiv ID:** 2606.26150 | [PDF](https://arxiv.org/pdf/2606.26150v1)

**作者:** Shuyi Chen `[一作]` (Southern University of Science and Technology), Georgios Theodoropoulos `[通讯]` (Research Institute of Trustworthy Autonomous Systems)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出并实现了热感知负载平衡框架（TLB），通过动态迁移LLM训练任务到最冷节点，消除热耦合导致的同步瓶颈。

**💡 创新点**

创新点在于将空间冷却异质性视为可调度资源，提出热感知异质性理论并设计TLB实现热驱动的工作负载划分。

**🔧 技术方法**

使用热遥测、能力剖面、动态工作负载切片等技术，并通过时间步长热-计算协同仿真验证方案；采用贪心比例分配启发式作为示例算法。

**📊 数据集**

利用模拟的高密度Orbital Data Center（单列64节点、近场星群36节点）和通用LLM训练批次，未使用真实数据集。

**📈 对比分析**

与统一负载分配方案对比，TLB在单列架构下MFU提升约7.6%，在星群架构下提升约0.2%，并将最热节点的MTTF提升1.7%–6.15%。

**⚠️ 局限性**

局限性包括仅采用简易贪心启发式、忽略多跳光路延迟和瞬态流体滞后等二次瓶颈，且验证仅在仿真环境中完成，缺乏真实硬件验证。

---

## 2. The Open Source Economic Index of AI Adoption and Capability

**arXiv ID:** 2606.26118 | [PDF](https://arxiv.org/pdf/2606.26118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 3. A Multi-Layer AI Framework for Information Landscape Analysis

**arXiv ID:** 2606.26115 | [PDF](https://arxiv.org/pdf/2606.26115v1)

**作者:** Maryam Fooladi `[一作]` (Kakashi Ventures Accelerator), Federico Bottino `[通讯]` (Kakashi Ventures Accelerator)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个网页化的 AI 平台，通过将内容拆分为 11 个可独立评分的维度，实现对信息失真的结构化分析，而非给出单一真假判定。

**💡 创新点**

创新点在于提出 claim–source independence（事实真伪与来源可信度分开评估）以及多尺度（URL 深度分析与关键词生态映射）并行的设计，避免传统系统的关联偏差与二元化判断。

**🔧 技术方法**

技术核心包括 Claude Sonnet 4 与 GPT‑4o‑mini 的结构化提示推理、Next.js 前端、Supabase PostgreSQL 后端，并结合实时网络检索与多任务并行流水线。

**📊 数据集**

使用公开可获得的报道与官方信息操作记录（以 2026 年俄方针对法国总统 Macron 的 “Epstein” 造谣行动为主要案例），并无专门标注的数据集。

**📈 对比分析**

通过案例研究与多尺度对比实验，展示同一主题在生态层、URL 层与不同来源之间产生不同的信任等级、操控指示和来源评估；相较于传统单标签分类，能更细粒度揭示信息失真多样性，未给出传统指标但表现出明显的区分度。

**⚠️ 局限性**

局限包括：LLM 生成的数值未经过标准验证或基准测试；评分权重为启发式；仅支持英文检索与分析；缺乏多语言、纵向跟踪与用户实验验证。

---

## 4. Privacy-Aware Agent Collaboration for Dynamic VR Slice Management in 6G SD-RAN

**arXiv ID:** 2606.26123 | [PDF](https://arxiv.org/pdf/2606.26123v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 5. Mitigating High-Frequency Geometric Noise in Non-Parametric 1-Bit Sparse

**arXiv ID:** 2606.26137 | [PDF](https://arxiv.org/pdf/2606.26137v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 6. Dot-Flik: A Scalable Edge AI Architecture for Distributed Insect Monitoring

**arXiv ID:** 2606.26121 | [PDF](https://arxiv.org/pdf/2606.26121v1)

**作者:** Mattia Consani `[一作]` (École Polytechnique Fédérale de Lausanne), David Atienza `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e0540dec-d77f-42db-94ae-d039248f6393` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一种可扩展的边缘人工智能架构Dot-Flik，用于分布式昆虫监测；

**💡 创新点**

创新点在于将运动感知的帧过滤算法放置在低成本边缘节点，实现大幅度数据压缩，同时通过分层架构将采集与分类解耦，显著提升网络可扩展性；

**🔧 技术方法**

采用基于时间差分、伽马校正和块级运动密度分析的轻量级运动检测算法；结合Raspberry Pi Zero 2 W、H.264硬件编码以及UDP/IP无线传输；

**📊 数据集**

使用在城市花园环境中实时采集的原始视频数据作为验证集；并未使用公开的标准昆虫图像数据集；

**📈 对比分析**

与传统集中式全帧处理对比，帧丢弃率可达60–80%（轻风条件下），能耗降低22.6%，在单节点下支持5–6条并发流，保持30 FPS实时性，表明系统在能耗、成本和计算负载方面优于传统方案；

**⚠️ 局限性**

局限性包括缺乏严格的人工标注的基准数据、对高风环境的运动阈值敏感、实验仅覆盖单一城市花园场景、以及对季节性光照与植被变化未系统评估。

---

## 7. LCG: Long-Context Consistent Image Generation with Sparse Relational Attention

**arXiv ID:** 2606.26171 | [PDF](https://arxiv.org/pdf/2606.26171v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 8. Where Larger Models Excel: The Primacy of Constraint-Guided Reasoning

**arXiv ID:** 2606.26108 | [PDF](https://arxiv.org/pdf/2606.26108v1)

**作者:** Guan-Yi Lin `[一作]` (National Chengchi University), Hen-Hsen Huang `[通讯]` (Academia Sinica)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 AdvCluster 框架，对比同一模型家族中大模型与小模型在数学、物理、化学、编程等多领域基准上的推理轨迹，自动提取优势描述并通过语义聚类构建理由优势分类学。

**💡 创新点**

创新点在于：① 用数据驱动的动态优势提取和语义聚类，避免预设类别；② 结合 reviewer 模型对聚类结果进行语义评估；③ 系统性发现大模型的核心优势是 Constraint‑Guided Reasoning。

**🔧 技术方法**

采用 Chain‑of‑Thought 推理、LLM 作为优势提取器与 reviewer、OpenAI embedding、PCA 降维、K‑means 聚类、Davies–Bouldin 与 Silhouette 评价指标，以及 Error Dispersion Index 进行错误分布分析。

**📊 数据集**

使用多领域推理基准：HHMT、OMNI、JEEBench（数学）、GPQA、OlympiadBench（物理）、CRUXEval（编程）等，涵盖化学、物理、数学、编程四大主题。

**📈 对比分析**

通过多次独立推理计算 PassRate，筛选 Δ≥0.6 的问题构成对比集；结果显示 Qwen3‑32B 对 Qwen3‑8B 提升约6.4%，GPT‑OSS‑120B 对 GPT‑OSS‑20B 提升约7.4%；错误深度分析表明小模型错误集中在变换层和过程层。

**⚠️ 局限性**

局限性：只在同一模型家族内进行比较；依赖 LLM 提取与 reviewer 评分，存在主观性；聚类结果受参数设定影响；未评估跨模态或更大规模模型的普适性。

---

## 9. Code evolution for link prediction in complex networks

**arXiv ID:** 2606.26132 | [PDF](https://arxiv.org/pdf/2606.26132v1)

**作者:** Alexey Vlaskin `[一作]` (University of Sydney), Eduardo G. Altmann `[通讯]` (University of Sydney)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

使用基于LLM的代码进化系统自动生成并优化链路预测算法

**💡 创新点**

通过演化发现结合节点与链路特征的新型组合，并实现了比现有手工设计方法更高的AUC和更低的计算成本

**🔧 技术方法**

采用AlphaEvolve蓝本的AntEvolve系统，结合Gemini/Qwen LLM进行程序变异与交叉，并用遗传岛屿策略保持多样性

**📊 数据集**

训练网络仅10个（6个合成+4/1真实），测试覆盖580个真实网络、8个合成网络和30个大规模网络

**📈 对比分析**

与四种主流手工方法（Adamic‑Adar、Node2Vec、SBM、Stacked模型）对比，演化算法平均AUC提升约0.13，效率提升至与最优人造方法相当或更快

**⚠️ 局限性**

局限在于演化过程仍依赖初始手工评估器、对极大网络的可扩展性受限于所选特征；且未发现根本新型算法架构，只是对现有方法的改进

---

## 10. Account-History Features for Social Bot Detection in the Era of Large Language Models

**arXiv ID:** 2606.26127 | [PDF](https://arxiv.org/pdf/2606.26127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 11. Divergent Recommendations, Convergent Diagnoses: Cross-Provider Failure-Mode Convergence in AI Commercial Recommendation

**arXiv ID:** 2606.26116 | [PDF](https://arxiv.org/pdf/2606.26116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 12. NetPTR: Optimal Differentially Private Spectral Community Detection on Sparse Networks

**arXiv ID:** 2606.26145 | [PDF](https://arxiv.org/pdf/2606.26145v1)

**作者:** Wanjie Wang `[一作]` (National University of Singapore), Tao Shen `[通讯]` (National University of Singapore)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 NetPTR（普通网络）和 Bi‑NetPTR（双边网络）两种差分隐私谱聚类方法，用于稀疏网络的社区检测。

**💡 创新点**

创新点在于通过局部灵敏度的稳定性测试，直接对谱嵌入加噪，理论上证明在 DCSBM 与 Bi‑DCSBM 下误差上界与下界一致，且优于传统全局噪声方法。

**🔧 技术方法**

采用 Propose‑Test‑Release 框架、谱分解、Davis‑Kahan 与 2→∞ 范数误差分析，并结合局部敏感度来设计噪声级别。

**📊 数据集**

在模拟的 DCSBM 与 Bi‑DCSBM、Flickr 联系人网络以及美国参议院投票网络上进行了实证验证。

**📈 对比分析**

与 EdgeFlip 等基线方法比较，NetPTR 在精度上显著优于 EdgeFlip，尤其在稀疏或度异质网络中能实现接近无隐私的误差；在实际数据中 ARI 超过 0.95，性能表现优秀。

**⚠️ 局限性**

局限在于需先估计或假设网络密度参数 θ₀，且对极度稀疏或极端度分布网络的效果仍有限；加密过程对节点/列噪声的规模随网络规模增长。

---

## 13. Geometric Fairness-Aware Routing for Federated Edge Networks

**arXiv ID:** 2606.26125 | [PDF](https://arxiv.org/pdf/2606.26125v1)

**作者:** Ratun Rahman `[一作]` `[通讯]` (University of Alabama in Huntsville), Ratun Rahman (University of Alabama in Huntsville)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 Geo-FairFed，一种基于负曲率超曲图神经网络（HGNN）与公平感知联邦聚合的边缘网络路由框架。

**💡 创新点**

创新点：①在超曲空间学习拓扑感知表征，捕捉层级与非对称结构；②在联邦聚合中引入基于 Jain 公平指数的权重调节，防止强节点主导；③给出曲率约束下的收敛证明并证明公平正则化可实现 Pareto 改进；④提出自适应公平权重机制进一步提升性能。

**🔧 技术方法**

技术手段：超曲图神经网络（HGNN）、联邦学习（FedAvg+公平权重）、Jain 公平指数、曲率正则化、NS-3+PyTorch 仿真、曲率与公平参数调优。

**📊 数据集**

数据集：合成 Barabási‑Albert 网络（N=50,100,200）、真实网络 RocketFuel、TopologyZoo 以及 AS 级别 ISP（Abilene、GEANT、AT&T 等）。

**📈 对比分析**

比较方法：与 DQN 路由、FedAvg 路由、HGNN 路由、FairFedAvg、GCN 路由等基线在延迟、吞吐量、能耗、Jain 公平指数、路由准确率等指标上进行多场景对比。结果显示 Geo‑FairFed 延迟降低 20%，能耗降低 17%，公平指数提升 21%，准确率达到 93% 以上，优于所有对比基线。

**⚠️ 局限性**

局限性：仅在 NS‑3 模拟环境评估，未考虑不稳定连接、straggler、部分参与、异步聚合等实际部署挑战；聚合中心单点可能成为瓶颈；缺乏对恶意攻击或欺诈性公平分数的鲁棒性处理。

---

## 14. Investigating LLM's Problem Solving Capability -- a Study on Statics Questions

**arXiv ID:** 2606.26103 | [PDF](https://arxiv.org/pdf/2606.26103v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 15. Neural Architecture Search for Generative Adversarial Networks: A Comprehensive Review and Critical Analysis

**arXiv ID:** 2606.26169 | [PDF](https://arxiv.org/pdf/2606.26169v1)

**作者:** Abrar Alotaibi `[一作]` (King Fahd University of Petroleum and Minerals), Moataz Ahmed `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对NAS在GAN领域的研究现状进行系统综述，并构建统一的评估框架以回答四个核心研究问题。

**💡 创新点**

提出了基于搜索空间、搜索策略、评估指标等维度的对比框架，系统识别了NAS-GAN的研究空白（如对Discriminator的自动化搜索、数据集多样性不足、评估指标单一等）。

**🔧 技术方法**

主要采用文献检索、质量评估、结构化数据抽取与可视化分析等方法，对演化算法、强化学习、梯度导向NAS等三大策略进行了归纳与对比。

**📊 数据集**

使用了多种公开数据集：MNIST、CIFAR‑10、CIFAR‑100、STL‑10、CelebA、LSUN等，涵盖了从简单到高分辨率、从无监督到条件生成的多样场景。

**📈 对比分析**

通过IS、FID、GPU‑days、搜索空间大小等指标对比，发现进化与梯度导向方法在不同任务中表现突出，EWSGAN在CIFAR‑10/SL‑10上取得最优分数；同时指出搜索效率与模型质量的权衡。

**⚠️ 局限性**

局限性包括：仅覆盖英文文献、未采用完整PRISMA体系、缺乏统计meta‑analysis、对IS/FID过度依赖、对Discriminator搜索关注不足，以及对环境成本与可解释性的讨论不足。

---

## 16. Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare

**arXiv ID:** 2606.26104 | [PDF](https://arxiv.org/pdf/2606.26104v1)

**作者:** Jasmine Brazilek `[一作]` (Compassion Aligned Machine Learning), Harper Dunn `[通讯]` (Independent researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过在 Llama‑3.2‑1B 上使用 LoRA 对 100 条包含不同语言特征（如情绪词、道德词汇、叙事结构等）的文本进行细调，并用词汇匹配的动物福利立场基准评估模型对“支持动物福利”答案的偏好，从而系统衡量每种语言特征对模型立场的影响。

**💡 创新点**

创新点在于设计了匹配对实验和词汇匹配的立场对比基准，能够在保持词汇一致的前提下精确识别语言特征对模型立场的推移效应；同时揭示了“主张性语言”会增强模型立场，而“模糊或具体感官描述”则会削弱。

**🔧 技术方法**

采用的技术包括：LoRA 微调、长度归一化的对数概率计算、配对 t 检验分析、以及基于 50 条二元选择项的动物福利立场基准。

**📊 数据集**

数据集包括：1,000 条受控匹配对文本（10 条语言特征 × 100 主题），以及 50 条词汇匹配的动物福利二元选择题。

**📈 对比分析**

与未微调的 Llama‑3.2‑1B 基线相比，模型的“支持动物福利”偏好得分在 7 个特征上显著提升（p<0.01），在 2 个特征上下降；整体基线已达 48/50 对齐。

**⚠️ 局限性**

局限性包括仅评估单一 1B 规模模型、训练集规模极小、基线已接近上限、以及对其他模型或更广泛主题的推广性未知。

---

## 17. Detecting and Controlling Sycophancy with Cascading Linear Features

**arXiv ID:** 2606.26155 | [PDF](https://arxiv.org/pdf/2606.26155v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 18. Multilateral Clearing on Invoice Graphs: Path Enabled Compensation Versus Cycle Restricted Netting

**arXiv ID:** 2606.26126 | [PDF](https://arxiv.org/pdf/2606.26126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 19. Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training

**arXiv ID:** 2606.26102 | [PDF](https://arxiv.org/pdf/2606.26102v1)

**作者:** Jasmine Brazilek `[一作]` (Compassion Aligned Machine Learning), Juliana Seawell `[通讯]` (Compassion Aligned Machine Learning)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在中期训练后加入不同领域（帮助性 vs 编码性）微调对动物同情价值的影响，证明帮助性微调会显著削弱已嵌入的同情价值，而编码性微调则能够保持甚至提升这些价值。

**💡 创新点**

创新点在于首次系统比较两种常见后期训练领域对中期训练价值的保留效果，并揭示了跨语言迁移差异，即动物同情价值在不同语言中更为稳固，而一般道德推理则主要局限于训练语言。

**🔧 技术方法**

使用了 Llama 3.1 8B 的 LoRA 微调技术（SFT 与 GRPO），配合 Unsloth 训练器和 Gemini‑2.5‑Flash‑Lite 评判模型，涵盖了监督微调和强化学习两种范式。

**📊 数据集**

数据集包括：3000 行合成动物同情中期训练语料；帮助性训练使用 Dolly‑15k；编码性训练使用 Magicoder‑110K；强化学习使用 RLHFlow；评估使用 Animal Harm Benchmark 2.2 和 MORU 基准。

**📈 对比分析**

对比方法是基于平均得分与 t‑检验的统计比较；结果显示帮助性 SFT 在 AHB 上从 60.2% 降至 35.7%（约减 24.5pp），而编码 SFT 保持 65.2%；在 MORU 英文项上帮助性从 71.9% 降至 46.4%（约减 25.5pp），但在多语言总览中两者差异消失。

**⚠️ 局限性**

主要局限包括仅使用单一 8B 模型与单一中期训练语料，未设立第三种对照领域；GRPO 训练中奖励模型与数据领域混合；数据质量（如 Dolly‑15k）可能影响结果；评估仅依赖单一 Gemini 评判器；未对多维度结果做多重比较校正；后期训练仅限帮助性与编码性两种领域。

---

## 20. Low Resource Multimodal Translation of Nepali Spoken Words into Emotion-Conditioned Sign Language Avatars

**arXiv ID:** 2606.26107 | [PDF](https://arxiv.org/pdf/2606.26107v1)

**作者:** Jatin Bhusal `[一作]` (Prateek Innovations), Salma Tamang `[通讯]` (Prateek Innovations)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级多模态框架NEST‑V1，将语音识别与情感识别共享编码器，实时生成情绪化的尼泊尔手语动画；

**💡 创新点**

创新点在于首次构建情绪标注的尼泊尔手语语音数据集，采用共享Vision Transformer实现语音识别与情感分类的参数共享，显著提升模型的可部署性；

**🔧 技术方法**

核心技术包括Mel频谱预处理、ViT骨干网络、双任务分类头、音频增广（VTLP、语调调节）以及基于GIF的情绪化头像渲染；

**📊 数据集**

使用包含四个常用尼泊尔词（“thank you”“hello”“house”“me”）和三种情绪（happy、neutral、sad）的600条标注音频样本（共50位说话人）构建的手语语音数据集；

**📈 对比分析**

与分别训练的ASR+情感模型相比，NEST‑V1参数量减少37.2%，在ASR上实现81.1%准确率、情感识别79.21%准确率，推理时间<50 ms，内存占用≈100 MB，适合边缘设备；

**⚠️ 局限性**

局限在于词汇量极小、情绪类别有限、采用预渲染GIF而非实时动态动画、缺乏真实聋人使用者的评估和更广泛多模态数据的支持。

---

## 21. Reinforcement Learning Enables Autonomous Microrobot Navigation and Intervention in Simulated Blood Capillaries

**arXiv ID:** 2606.26154 | [PDF](https://arxiv.org/pdf/2606.26154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 22. Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM

**arXiv ID:** 2606.26120 | [PDF](https://arxiv.org/pdf/2606.26120v1)

**作者:** Tianyi Wu `[一作]` (Harbin Institute of Technology), Zhuotao Tian `[通讯]` (Shenzhen Loop Area Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的动态加速框架 Dynamic-dLLM，通过自适应缓存更新与自适应并行解码提升扩散式大型语言模型的推理效率。

**💡 创新点**

创新点在于将缓存更新预算和解码阈值动态分配到不同层级和解码步，利用令牌输入差异和预测分布的动态信息，避免了传统静态策略的浪费。

**🔧 技术方法**

使用了令牌输入余弦距离衡量动态度量、动态缓存更新（DCU）与强制更新窗口、基于置信度与时间不稳定性的自适应阈值（APD）等技术。

**📊 数据集**

在 LLaDA-8B-Instruct、LLaDA-1.5、Dream-v0-7B-Instruct 三种扩散 LLM 上，结合 MMLU、ARC-C、GSM8K、GPQA、HumanEval 等基准数据集进行评估。

**📈 对比分析**

与 dLLM-Cache、dKV-Cache、Fast-dLLM 等现有加速方法对比，Dynamic-dLLM 在保持相近准确率的前提下，平均提升 2.5–3.2 倍吞吐率，单项最快可达 4.48 倍。

**⚠️ 局限性**

局限性在于仅针对单模文本输入验证，未针对多模态理解、复杂推理等更具挑战的任务展开测试，且扩展性尚待进一步研究。

---

## 23. Reducing Conversational Escalation in Large Language Model Dialogue with Nonviolent Communication Constraints

**arXiv ID:** 2606.26106 | [PDF](https://arxiv.org/pdf/2606.26106v1)

**作者:** Zhixing Sun `[一作]` (Beijing University of Posts and Telecommunications), Tao Li `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将非暴力沟通（NVC）原则转化为提示级约束，指导大型语言模型在冲突情境中进行去激化对话。

**💡 创新点**

创新点在于将NVC过程导向约束直接嵌入系统提示，实现轻量级、可跨模型应用的冲突去激化方法，并首次在多模型、多用户抵抗级别的仿真环境中验证其有效性。

**🔧 技术方法**

主要技术包括基于vLLM的双代理交互管线、NVC系统提示、LLM评判（DeepSeek‑V3/Claude‑4.5‑Sonnet）以及对话冲突轨迹评分。

**📊 数据集**

数据集为人工合成的冲突情境对话，利用LLM生成多种语气和强度的变体，共500个场景，涵盖工作、情感及社区冲突。

**📈 对比分析**

通过与标准“Vanilla”提示对比，实验在低、中、高抵抗用户场景下分别评估冲突轨迹分数和事件分布，NVC约束在所有模型中均显著降低升级率、提升去激化率，尤其在高抵抗情境下显著改善稳定性。

**⚠️ 局限性**

局限性主要在于整个评估基于LLM生成的用户模拟和自动评分，缺乏真实人类交互的验证，且模型和用户多样性仍受限。

---

## 24. Refusal Lives Downstream of Persona in Chat Models

**arXiv ID:** 2606.26161 | [PDF](https://arxiv.org/pdf/2606.26161v1)

**作者:** Viola Zhong `[一作]` (Independent), Qirui Li `[通讯]` (Pohang University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实验了模型人格与拒绝行为在聊天模型中的交互，发现合规人格方向能抑制拒绝，并通过投影干预揭示其在后期层面起作用。

**💡 创新点**

首次揭示拒绝机制不是单一方向，而是与人格表征相耦合的多维过程，并证明拒绝在后期表达阶段被人格门控。

**🔧 技术方法**

线性方向提取、激活投影、正负拒绝方向干预、三分类安全评估（拒绝/绕过/退化）以及 GPT-4o 行为评分。

**📊 数据集**

313题 StrongREJECT 禁止提示集、Llama-Guard-3 及模型本身的对比性人格提示集合。

**📈 对比分析**

通过对比基线、人格仅干预、人格+拒绝方向干预、人格投影消除等多组实验，发现人格投影消除可将拒绝率从1.6%恢复至近基线（≈97%），证明了人格门控效果。

**⚠️ 局限性**

仅在两款 7–8B 指令调优模型上验证，窗口位置模型相关，评估依赖模型裁判，未揭示完整拒绝表达电路，仅识别了方向层面的门控。

---

## 25. Physics-guided Convolutional Neural Network for Domain Growth Prediction in Systems with Conserved Kinetics

**arXiv ID:** 2606.26128 | [PDF](https://arxiv.org/pdf/2606.26128v1)

**作者:** Vijay Yadav `[一作]` (Indian Institute of Technology Jodhpur), Prabhat K. Jaiswal `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于注意力的物理引导残差U-Net卷积神经网络，用于预测受守恒动力学驱动的Cahn‑Hilliard相分离过程的空间时间演化。

**💡 创新点**

创新点在于将全局注意力机制嵌入低分辨率桥接层、在损失中加入守恒约束，并使用残差U-Net实现深层特征提取，从而实现长时间滚动预测并保持守恒量。

**🔧 技术方法**

采用残差U-Net、注意力块、物理约束损失、组归一化、GeLU激活以及PyTorch Lightning训练框架。

**📊 数据集**

使用基于有限差分求解的Cahn‑Hilliard方程生成的合成数据集，64×64网格，20条训练轨迹、5条验证轨迹，覆盖临界与非临界组分。

**📈 对比分析**

与数值解器进行对比，使用R^2、平均误差、保守性检验、域尺寸生长律、相关函数数据塌缩等指标；结果显示R^2≈1，守恒量保持不变，域尺寸生长遵循t^{1/3}，在临界体系上表现最优。

**⚠️ 局限性**

局限在于需要预先知道守恒律以加入损失；对非临界体系的滴状相分离预测误差较大；仅训练于早期时刻，且数据集为合成模拟数据，缺乏真实实验验证。

---

## 26. Thinking Like a Scientist? A Structural Study of LLM-Generated Research Methods

**arXiv ID:** 2606.26130 | [PDF](https://arxiv.org/pdf/2606.26130v1)

**作者:** Francesca Carlon `[一作]` (Vrije Universiteit Brussel), Andres Algaba `[通讯]` (Vrije Universiteit Brussel)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对仅通过研究问题提问的LLM（GPT‑5.1、Gemini 3 Pro、DeepSeek‑V3.2）在方法建议上的偏差进行量化研究，并与1,000篇近期arXiv LLM 相关论文的真实方法清单进行对比。

**💡 创新点**

首次系统量化LLM在方法建议中的词汇压缩与供应商偏向，并揭示其对方法组合的集中化；提出多维度（provider、dataset、metric 等）分布性比较框架。

**🔧 技术方法**

使用实体抽取、规范化与模糊聚类、15维 taxonomy 分类、Jensen–Shannon divergence、Cramér's V、Spearman、Jaccard、BM25检索校准、跨模型审核等技术。

**📊 数据集**

1,000 篇最新的arXiv 计算机科学论文（LLM 相关），其中抽取的 dataset、model、metric 实体；LLM 生成的研究问题与建议。

**📈 对比分析**

与论文-derived 参考库存比较：词汇压缩（effective number 下降 13–21 倍），90% 长尾缺失，provider 维度 JSD 3–5 倍其余维度；inter‑LLM 相关性 0.55–0.68 高于与参考的 0.33–0.56，显示三者共享同一压缩与供应商聚焦模式。

**⚠️ 局限性**

仅在单一研究问题提示下评估；GPT‑5.1 负责多步生成导致比较不独立；时间窗口与训练截止导致模型命名错配；研究仅聚焦 LLM 相关论文，缺乏跨领域验证；未评估对实际研究者选择的真实影响。

---

## 27. The Kernel's Write: Application Read-Only Memory

**arXiv ID:** 2606.26138 | [PDF](https://arxiv.org/pdf/2606.26138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 28. Implementation of reinforcement learning in chemical reaction networks: application to phototaxis as curiosity-driven exploration

**arXiv ID:** 2606.26168 | [PDF](https://arxiv.org/pdf/2606.26168v1)

**作者:** Ruyi Tang `[一作]` (Sorbonne University), David Colliaux `[通讯]` (Sony CSL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个主观POMDP模型来描述Chlamydomonas的光趋性，设计了记忆无状态的贝叶斯更新与信息驱动的单步前瞻策略，并将该策略实现为可通过质量作用动力学计算的化学反应网络（CRN–ODE），随后利用逆强化学习（IRL）从实验轨迹中恢复行为奖励并验证模型；

**💡 创新点**

将强化学习的POMDP框架与可实现的化学反应网络结合，推导出互信息的多项式上界以实现“好奇心”奖励在化学体系中的可计算性，并通过该模型解释跑–旋转行为为主动感知策略，首次实现从感知到化学动力学的闭环桥接；

**🔧 技术方法**

使用POMDP、贝叶斯推理、逆强化学习、质量作用化学动力学（CRN-ODE）、数值模拟（SSA与ODE积分）、数据预处理与特征提取等技术；

**📊 数据集**

30条来自Chlamydomonas在光纤照明下的二维跟踪轨迹（其中4条保留为测试集）；

**📈 对比分析**

与传统的“客观”SSA基线（即时旋转）和改进的连续时间马尔可夫链基线进行对比，使用对齐余弦分布的Wasserstein-1距离和Kolmogorov–Smirnov统计评估相似度，并比较转向频率；在校准温度后，学习策略的对齐分布与实验相近，但转向频率与实验相差仍显著；

**⚠️ 局限性**

单一温度参数无法同时匹配转向频率与对齐精度；多项式上界虽可实现化学计算但可能失去顺序保持性；实验数据量有限，测试集仅4条；模型仅处理单光源静态环境，未考虑多源或动态光照情况；

---

## 29. Life After Benchmark Saturation: A Case Study of CORE-Bench

**arXiv ID:** 2606.26158 | [PDF](https://arxiv.org/pdf/2606.26158v1)

**作者:** Nitya Nadgir `[一作]` (Independent), Arvind Narayanan `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对CORE‑Bench Hard进行构造效度分析，推出改进版CORE‑Bench v1.1与OOD，并在多维度（准确率、可靠性、效率、模型与脚手架贡献）上评估代理性能，同时开展人机协作提升实验。

**💡 创新点**

①在准确率饱和后探索多维性能维度；②通过日志分析修复构造效度问题，推出v1.1与OOD；③揭示模型与脚手架互补效应；④实证人机协作在可重复性任务中的显著加速。

**🔧 技术方法**

使用大规模日志分析与Docent自动轨迹标注、多个模型/脚手架对照实验、可靠性/效率/校准等统计指标，以及随机对照实验评估人机协作提升。

**📊 数据集**

CORE‑Bench Hard（45题）、CORE‑Bench v1.1（39题）、CORE‑Bench OOD（19题）以及20篇机器学习与社会科学论文的重现实验数据。

**📈 对比分析**

通过多跑重复测量计算准确率、可靠性、一致性、token与成本；比较不同模型与脚手架的准确率差异；人机协作实验中平均完成时间缩短约2.1倍，统计显著。

**⚠️ 局限性**

样本量有限、缺少真实基准结果、参与者与作者同一人导致潜在偏差、仅覆盖Python/R任务，且构造效度修正仍需持续迭代。

---

## 30. Machine Learning-Driven Content Popularity Prediction and Cache Optimization in D2D Clustered Networks

**arXiv ID:** 2606.26119 | [PDF](https://arxiv.org/pdf/2606.26119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 31. Unsupervised Memory-Enhanced Video Transformers: Obstacle Detection for Autonomous Agricultural Rover

**arXiv ID:** 2606.26151 | [PDF](https://arxiv.org/pdf/2606.26151v1)

**作者:** Théo Biardeau `[一作]` (Université de Poitiers), David Helbert `[通讯]` (Université de Poitiers)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种完全无监督的 Video Memory Transformers for Anomaly Detection (VMTAD)，用于农田无人车在动态环境下实时检测植被下方的障碍物。

**💡 创新点**

创新点在于：① 将 Transformer 编码器与 FIFO 记忆模块结合，利用跨注意力实现线性时间复杂度的时序上下文建模；② 在完整分辨率上完成重建，避免卷积模型的 shortcut 问题；③ 通过 Cosine 相似度重建损失和高斯平滑提升鲁棒性。

**🔧 技术方法**

技术细节包括：EfficientNet-B5/B0 作为特征提取器；Transformer 编码器-解码器；FIFO 记忆模块（K=2 层、N=2 长度）；Cosine 相似度损失；二维高斯卷积后产生异常分数图。

**📊 数据集**

使用在法国 Pressac 田间收集的油菜和玉米两种作物数据集：训练集9个无障碍油菜序列（600张/序列），评估集5个油菜序列和13个玉米序列（每序列600张）。

**📈 对比分析**

与 CAE、VQ‑VAE、MemAE 等农用无监督方法以及 PatchCore、GeneralAD、SimpleNet 等工业 SOTA 进行比较。VMTAD‑B5 在油菜检测 AUROC 0.973、分割 AUROC 0.997、推理时间 64 ms；VMTAD‑B0 在检测 0.950、分割 0.996、推理 14 ms，显著低于 GeneralAD 的 101 ms。

**⚠️ 局限性**

局限性包括：在跨域（油菜→玉米）时检测 AUROC 降低，易被阴影误判；内存长度与推理时间存在权衡；整体系统延迟受硬件/软件瓶颈影响，仍需进一步压缩全流程时延。

---

## 32. Neural Speaker Diarization via Multilingual Training: Evaluation on Low-Resource Nepali-Hindi Speech

**arXiv ID:** 2606.26144 | [PDF](https://arxiv.org/pdf/2606.26144v1)

**作者:** Samip Neupane `[一作]` (Institute of Engineering), Basanta Joshi `[通讯]` (Institute of Engineering)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究低资源尼泊尔-印地语语音的说话人分离，采用多语言训练，比较两种 EEND 架构（EEND‑EDA 与 DiaPer）在多说话人场景下的表现。

**💡 创新点**

将 Perceiver‑based attractor 网络 DiaPer 引入低资源多语言说话人分离，利用多语言混合语料减少语言偏差，并证明其在尼泊尔‑印地语等低资源语言上优于传统 EEND‑EDA。

**🔧 技术方法**

使用端到端神经说话人分离技术，EEND‑EDA 采用 LSTM 编码器‑解码器吸引子；DiaPer 采用 Perceiver 跨注意力吸引子模块并配合多层辅助损失；还使用 Transformer、Perceiver、WebRTC VAD、pydub 合成多说话人语料。

**📊 数据集**

LibriSpeech（英语）、VoxCeleb（多语种）、独立收集的 18 位尼泊尔女性说话者及 100 位印地语说话者，并通过合成方式生成 2/3/4/混合说话人录音。

**📈 对比分析**

在 LibriSpeech、VoxCeleb 与 NeHi 三组测试集（2/3/4/混合说话人）评估 DER；DiaPer 在 3/4/混合说话人条件下 DER 明显低于 EEND‑EDA，尤其在 NeHi 4 说话人时 DER 仅 4.76% 远低于 11.19%，但在 2 说话人条件下 EEND‑EDA 稍优。

**⚠️ 局限性**

仅使用合成多说话人录音，缺乏真实对话数据；尼泊尔语料仅 18 名女性，缺乏说话人多样性；未对语言进行单独错误分析；低资源语料量有限。

---

## 33. Enhancing FANET Routing Resilience: A Fuzzy-Driven Bio-Inspired Approach and Its Quantitative Evaluation

**arXiv ID:** 2606.26124 | [PDF](https://arxiv.org/pdf/2606.26124v1)

**作者:** Xinwang Yuan `[一作]` (Military Intelligence Academy), Congxi Song `[通讯]` (Military Intelligence Academy)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出一种基于ABC算法的集群头选举、双层信标维护以及模糊逻辑自适应Hello间隔的FANET路由协议FBCR；

**💡 创新点**

创新点包括：①将人工蜂群算法与多因子评估相结合实现稳定的集群头选举；②设计双层信标机制实现主动的集群维护；③引入模糊逻辑控制Hello间隔，在动态环境下平衡拓扑感知与控制开销；④提出参数敏感性指数PSI和整体性能指数CPI，用以量化路由协议的鲁棒性与性能；

**🔧 技术方法**

技术上采用人工蜂群（ABC）聚类、双层信标机制、Mamdani型模糊推理器、NS‑3仿真、统计指标（PDR、E2ED、吞吐量、控制开销）以及PSI/CPI评估框架；

**📊 数据集**

使用NS‑3仿真数据：2000×2000×500 m³空间、50–300架无人机、Gauss–Markov移动模型、20–70 m/s速度、IEEE 802.11g、UDP流量；未使用真实数据集；

**📈 对比分析**

与AODV、OLSR（Hello间隔0.25 s/1 s）、LEACH、K‑means、ICRA等基线进行对比；FBCR在PDR、吞吐量与控制开销上优于基线，控制开销降低约25%，CPI最高且PSI表现平衡；

**⚠️ 局限性**

局限性包括：①仿真环境未覆盖真实无人机硬件与干扰；②模糊逻辑参数需人工调优，可能在极端动态条件下失效；③只评估了单一层级无人机网络，未考虑多层异构网络；④缺乏在线学习或深度强化学习等更自适应机制。

---

## 34. Predicting Fruit Quality with a Hybrid Machine Learning and Image Processing Approach

**arXiv ID:** 2606.26165 | [PDF](https://arxiv.org/pdf/2606.26165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 35. From Lexicon to AI: A Structured-Data Pipeline for Specialized Conversational Systems in Low-Resource Languages

**arXiv ID:** 2606.26112 | [PDF](https://arxiv.org/pdf/2606.26112v1)

**作者:** Siddhant Hitesh Mantri `[一作]` (Narsee Monjee Institute of Management Studies), Pushpak Bhattacharya `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 Hindi WordNet 转化为 125 万条问答对，并用 LoRA 与 4‑bit 量化微调 Gemma‑3‑12B‑IT，构建专门的 Hindi 语言学习聊天机器人。

**💡 创新点**

首次证明专家编纂的词典资源可作为低资源语言对话系统的训练基底，兼顾语义保真与教学效果。

**🔧 技术方法**

采用 LoRA 参数高效微调、NF4 4‑bit 量化、动态分块生成多样化教育问答、层级化提示实现能力级别自适应。

**📊 数据集**

主要使用 Hindi WordNet（105,460 词条、40,466 同义集）生成的 1,253,847 条唯一问答对。

**📈 对比分析**

通过 200 条专家标注答案的语义相似度（SAS）和自动教师裁判的教学质量（LAQ）评估，模型在 LAQ 上获得 91.0 分，优于 GPT‑4.1 等通用模型。

**⚠️ 局限性**

依赖 WordNet 的可用性与质量，未覆盖专家级长篇解释，对非 Hindi 结构差异未作验证，且缺乏真实课堂长期学习效果的评估。

---

## 36. Generative AI and Copyright Infringement: A Legal-Technical Analysis of AI Music Generation Systems Under 17 U.S.C. Title 17

**arXiv ID:** 2606.26111 | [PDF](https://arxiv.org/pdf/2606.26111v1)

**作者:** Zuhaib Hussain Butt `[一作]` `[通讯]`, Zuhaib Hussain Butt

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了美国版权法下的生成式 AI 音乐创作，分析了歌词、旋律、声纹复制的侵权风险，梳理相关案例与州级声权法；

**💡 创新点**

创新点在于将 AI 生成音乐的技术组件映射到具体版权风险，识别联邦法对声音克隆的空白，并提出统一声权保护与 AI 训练数据许可的政策建议；

**🔧 技术方法**

使用 Transformer/扩散模型、神经声码器和声源嵌入等 AI 音乐生成技术，对技术流程进行法律风险映射；

**📊 数据集**

主要依赖最新判例与立法文献（如 Concord v. Anthropic、Lehrman v. Lovo 等）进行案例综述，并未使用公开数据集；

**📈 对比分析**

论文未进行实验对比，而是通过案例与理论分析比较不同技术路径对侵权风险的影响；

**⚠️ 局限性**

局限在于仅聚焦美国法，缺乏跨国比较，未实证测试 AI 生成音乐的侵权判定准确性，且政策建议需进一步立法落实。

---

## 37. Context Recycling for Long-Horizon LLM Inference

**arXiv ID:** 2606.26105 | [PDF](https://arxiv.org/pdf/2606.26105v1)

**作者:** Derek Thomas `[一作]` `[通讯]` (Independent Researcher), Derek Thomas (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种将LLM上下文窗口视为可回收执行工作空间的架构，并通过五层层次化记忆层次和上下文回收机制实现了在固定上下文预算下支持无限长会话。

**💡 创新点**

创新点在于将上下文窗口视为可回收资源，设计了五层层次化记忆体系（LoRA、KV前缀、分支缓存、索引和磁盘存储）以及主动加载和缓存压缩策略，显著提升了记忆利用效率。

**🔧 技术方法**

采用了LoRA训练无监督构造、TQ3量化KV缓存、SQLite FTS5倒排索引、BM25检索、LLM Prompt工程、主动预加载和压缩算法等技术。

**📊 数据集**

使用了美国CMS Medicare提供者利用和支付数据（约2.76亿行）作为实验数据集。

**📈 对比分析**

在12/15轮对话基准上与Azure AI Foundry Agent进行对比，准确率相当（85/84、225/194），但在12轮时速度提升4.7倍、token使用减少4.2倍，15轮时速度提升8.0倍、token减少13.4倍。

**⚠️ 局限性**

局限性包括仅在单一域（医疗支付数据）和两轮基准上验证，缺乏跨域及大规模分布式测试；LoRA构造仅适用于自托管模型；未来需进一步评估在不同模型、数据与用户群中的泛化能力。

---

## 38. \chisao{}: A GPU-Native Parallel Optimizer for Multimodal Black-Box Functions via Convergence-Anticonvergence Oscillation

**arXiv ID:** 2606.26164 | [PDF](https://arxiv.org/pdf/2606.26164v1)

**作者:** Ira Wolfson `[一作]` `[通讯]` (Braude College of Engineering), Ira Wolfson (Braude College of Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 GPU 原生并行优化器 ChiSao，专门用于在黑盒多模态函数中快速、完整地发现所有显著峰值。

**💡 创新点**

创新点包括：① 冻结已识别峰值的“stick”策略与对未冻结样本的连续探索形成不对称运动；② 通过有向的反收敛（momentum‑based anti‑convergence）与高斯平滑（Hands Like Clouds）交替震荡，增强跨阈值跳跃能力；③ 两种自适应重新种子策略（Repulse Monkey 与 Golden Rooster）维持种群多样性；④ 采用一次性批处理 L‑BFGS、平滑梯度与推断，充分利用 GPU 统一指令多数据并行。

**🔧 技术方法**

技术手段包括：GPU 批量 L‑BFGS、有限差分梯度估计、梯度范数与似然门控的停滞检测、L∞ 去重、随机 Ray 投射重种子、正则化平滑梯度、动量更新的反收敛步骤，所有操作均在单个 CUDA 核内并行完成。

**📊 数据集**

使用了 Simon Fraser University (SFU) 的 42 个标准多模态与单峰基准函数，维度覆盖 d∈{2,4,8,16,32,64}，并在低维固定函数集（Group C、D）中进一步验证。

**📈 对比分析**

与 Differential Evolution、Basin‑Hopping、CMA‑ES 等 CPU 基线进行比较。ChiSao 在所有多模态函数中实现 100% 峰值检索（除 Schwefel），并在高维（d≥8）时保持相同精度，而基线往往在 d≥8 时完全失败；在相同预算下，ChiSao 的 GPU 并行使得平均运行时间对维度几乎不变，提供 30–40 倍的加速（例如 Rastrigin d=64 仅 4.2 s vs. BH 805 s）。

**⚠️ 局限性**

局限性包括：依赖光滑可导目标（有限差分或解析梯度）且不适用于离散或非可导问题；对质量门 δ=0.1 的设定在某些“欺骗”函数（如 Schwefel、Trid、Bukin）导致误判；当模式数呈指数级增长时，去重与重种子逻辑难以扩展；理论分析仅涵盖 log‑concave 及首次震荡周期，完整的收敛证明仍待研究。

---

## 39. Reducing Redundancy in Whole-Slide Image Patching for Scalable Indexing and Retrieval

**arXiv ID:** 2606.26157 | [PDF](https://arxiv.org/pdf/2606.26157v1)

**作者:** Jialiang Geng `[一作]` (Mayo Clinic), H. R. Tizhoosh `[通讯]` (Mayo Clinic)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种 Antithetical Redundancy Reduction (ARReST) 方法，通过在不同癌症类别之间寻找高度相似的补丁来识别并剔除冗余补丁，从而显著降低 whole‑slide image (WSI) 的索引存储量并加速检索过程。

**💡 创新点**

创新点在于首次利用跨类别的相似性构建“冗余库”，并以此为依据进行针对性去冗余，区别于传统仅在同类内部去重的策略；同时将此反向冗余思想应用于 WSIs 的检索与索引。

**🔧 技术方法**

使用了无监督补丁选择算法（如 Yottixel）、基础模型嵌入（UNI）、MinMax BOB 二进制编码、基于分位数的相似性阈值裁剪，以及 top‑k 投票检索等技术。

**📊 数据集**

实验基于 TCGA（The Cancer Genome Atlas）公开数据集，共 11,679 张 WSI，覆盖 21 种器官。

**📈 对比分析**

通过 5 折留患者交叉验证（Stratified Group K‑Fold）比较去冗余前后在 Top‑1、MV@3、MV@5 三种检索场景下的准确率与宏平均 F1；结果显示平均存储节省约 14%±13%，准确率基本保持不变，F1 在样本不均衡的组织中略有下降。

**⚠️ 局限性**

局限性包括：去冗余效果高度依赖相似性阈值，组织形态高度异质或少数类子类型明显时可能误删判别性补丁；需进一步研究自适应阈值或深度学习相似度估计以提升稳健性。

---

## 40. Kiko: Programming Agents to Enact Interaction Protocols

**arXiv ID:** 2606.26156 | [PDF](https://arxiv.org/pdf/2606.26156v1)

**作者:** Samuel H. Christie `[一作]` (North Carolina State University), Amit K. Chopra `[通讯]` (Lancaster University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种基于协议的多智能体编程模型Kiko，使开发者仅需编写基于角色的决策者，隐藏通信细节并保证协议合规性；

**💡 创新点**

创新点在于将信息协议BSPL与决策者抽象相结合，提供兼容性检查和原子发射集合支持，实现在去中心化异步环境中的协议执行；

**🔧 技术方法**

核心技术包括BSPL信息协议语言、Python实现的Kiko框架、UDP无序无可靠传输服务以及协议适配器和决策者事件驱动机制；

**📊 数据集**

未使用公开数据集，示例基于Purchase和Approval等虚构协议场景进行演示；

**📈 对比分析**

通过形式化证明与优化的合规性检查展示Kiko在协议执行时保持一致性与可满足性，但文中未给出具体实验性能指标；

**⚠️ 局限性**

局限性包括缺乏大规模真实系统评估、对网络异常鲁棒性验证不足以及对复杂协议动态更新支持的探索仍待完善。

---

## 41. The Governance Inversion Hypothesis: Why More AI Regulation May Produce Less Organisational Control

**arXiv ID:** 2606.26117 | [PDF](https://arxiv.org/pdf/2606.26117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 42. Multiscale Exit-Join Dynamics: Tactical Consensus and Strategic Coalition Formation

**arXiv ID:** 2606.26139 | [PDF](https://arxiv.org/pdf/2606.26139v1)

**作者:** Quanyan Zhu `[一作]` `[通讯]` (New York University), Quanyan Zhu (New York University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

构建了一个多尺度的联盟形成模型，将内部的协商一致动态（DeGroot）与外部的退出-加入决策耦合在一起，并通过共识结果自发生成联盟价值。

**💡 创新点**

创新点在于将联盟价值内生化为一致性结果，提出联合战术-战略均衡概念，引入认知壁垒和极化/分层机制，阐明战略不稳定如何促进全局共识。

**🔧 技术方法**

采用快慢耦合动力学、DeGroot一致性模型、Aumann–Drèze收益分配、有限改进图的固定点理论，以及随机矩阵收敛与 Wolfowitz–Hajnal 定理进行分析。

**📊 数据集**

实验使用仿真数据：10 名代理人随机初始化意见，统一交互矩阵，性能函数为单峰或非凸函数。

**📈 对比分析**

通过改变切换成本进行基线比较；结果显示正成本提高认知壁垒导致联盟稳定、信息混合减慢；负成本导致联盟不收敛但仍实现全局一致；用总联盟盈余 W(t) 评估性能。

**⚠️ 局限性**

局限性包括对有限改进和原始矩阵原始性的严格假设、仅考虑静态接受规则、未验证大规模系统或随机网络、对非凸收益的分析有限。

---

## 43. Bayesian Predictive Synthesis for Dynamic Networks: Forecasting and Identifying Structural Mechanisms

**arXiv ID:** 2606.26136 | [PDF](https://arxiv.org/pdf/2606.26136v1)

**作者:** Marios Papamichalis `[一作]` (Yale University), Theofanis Papamichalis `[通讯]` (Yale University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了一种动态贝叶斯预测合成方法，用多种网络生成机制（社区、几何、度分布等）作为代理，学习时间变化的权重并给出下一张网络的校准概率预测。

**💡 创新点**

创新点在于：①能在单一稀疏网络快速度估计机制权重，并给出理论识别、分离阈值与最优切换恢复；②通过稀疏安全参数化和交叉折叠正交估计，提供权重置信区间；③在模型失配时仍保持校准，并优于传统贝叶斯模型平均与堆叠。

**🔧 技术方法**

主要技术包括：动态线性模型的贝叶斯预测合成（BPS）、稀疏安全参数化、交叉折叠正交估计、拉普拉斯近似的卡尔曼滤波、以及基于稀疏信息率的理论推导。

**📊 数据集**

使用的数据集有：S&P500 相关网络、SocioPatterns 高中与医院接触网络、arXiv HEP-PH 引用网络、Enron 邮件网络、Bitcoin-OTC 信任网络等。

**📈 对比分析**

与贝叶斯模型平均、堆叠、均匀池化及单一生成模型等方法对比，动态BPS 在负对数似然、PIT-KS、ECE 等校准指标上显著优于对手，尤其在机制切换点能快速恢复并保持高预测性能。

**⚠️ 局限性**

局限性：需要预先设定有限的机制集合；假设边独立且二值，无法直接处理有向、加权或多值边；在极稀疏或高维潜在参数的情况下估计速率和置信区间的准确性有限；对动态度分布或几何结构的自适应性尚需进一步研究。

---

## 44. DocArena: Turning Raw Documents into Controllable Training Environments for Document Search Agents

**arXiv ID:** 2606.26122 | [PDF](https://arxiv.org/pdf/2606.26122v1)

**作者:** Jiamian Wang `[一作]` (Rochester Institute of Technology), Tong Sun `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建完全自动化的DocArena管道，将原始多模态PDF文档转换为可用于训练搜索代理的（问题、答案、证据）样本，生成了79,623对的DocArena-79K数据集。

**💡 创新点**

创新点在于：①跨页信息分布分析实现证据排他性保障；②多层级质量保证（规则过滤→MLLM校验→留一页必要性测试）消除人工标注噪声；③将视觉感知与策略模型解耦，允许纯文本LLM在多模态检索框架中高效推理。

**🔧 技术方法**

主要技术包括：多模态大语言模型（MLLM）用于视觉感知与模板化推理；ColPali视觉检索与OCR结合实现检索；进展奖励与答案奖励的混合RL策略；使用GRPO优化搜索策略。

**📊 数据集**

使用从CCpdf等来源收集的8,336份文档（覆盖16个领域、49种语言），产生了79,623个QA对，涉及37,480个证据页，涵盖文本、表格、图表等多模态内容。

**📈 对比分析**

在6个多模态文档场景（MMLongBench-Doc、VisRBench、SlideVQA等）和7个文本QA基准上与多种基线搜索代理（Search-R1、DeepResearcher、IKEA等）对比，Doc-Search在检索召回、精确率、EM、PNLS等指标上均取得最优或领先，且在未见文本QA训练数据的情况下，平均EM仍高于竞争者。

**⚠️ 局限性**

局限性包括：对视觉检索模型的高度依赖，可能在视觉质量低或极端多模态环境下性能下降；多轮检索策略仍易出现重复查询，尽管通过进展奖励抑制；以及缺乏对长文本细粒度语义理解和推理机制的深入评估。

---

## 45. Dream machine -- the next creative economy

**arXiv ID:** 2606.26114 | [PDF](https://arxiv.org/pdf/2606.26114v1)

**作者:** Peter Woodbridge `[一作]` (DreamLab AI Collective), John J. O'Hare `[通讯]` (DreamLab AI Collective)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本书基于作者在2025‑2026年间创办的 Dream Machine 电子通讯与 DreamLab 工作室的实践经验，系统记录了人工智能在创意产业（影视、音乐、游戏、广告等）中的快速渗透、工会与政策回应、技术工具发布以及创作者与消费者的共生关系，构建了《人‑AI 代理连续体》《滑坡屋顶》《四个岗位图谱》等分析框架，阐述了“新创意经济”在技术、经济与社会层面的转型路径。

**💡 创新点**

创新点在于：①首次将AI技术渗透与行业内部运作、工会谈判与政策讨论三条线索同步呈现，形成一体化的“八个月快照”；②提出“人‑AI 代理连续体”与“四原则”两套可检验各方决策的实用框架；③结合大量公开数据（Adobe创作者工具报告、UK版权咨询、行业调查）与原创通讯内容，提供可复制的实证研究方法；④将创作生态从“技术作业”升级为“Why‑主导”模式，强调人类创意的主导地位。

**🔧 技术方法**

主要技术包括：生成式AI模型（OpenAI Sora 2、Meta Llama、Runway、Luma、Kling、Veo、World Labs Marble、Hunyuan、Wan等视频/图像生成模型）；语音/音乐生成技术（Suno、Udio、Google Magenta、OpenAI Jukebox等）；内容鉴权与真实性链（C2PA、SynthID、ContentID等）；以及对策工具链与工作流自动化平台（Adobe Firefly、Runway Studio、World Labs Marble、OpenAI Sora App等）。

**📊 数据集**

数据集主要来源于：①Dream Machine 29期内的链接与引用（共数千条）；②Adobe Creative AI工具报告（86%创作者使用率）与英国政府版权咨询数据（88%支持AI公司授权）；③各类行业报告与公开统计（Spotify、Deezer AI音乐上传量、Netflix/Disney AI后期工作比例、Adobe AI工具使用率等）；④官方工会声明、法院判例（Grand Upright、Bridgeport）与政策文本；⑤公开的工具使用案例与实验（如Sora 2 iOS应用下载、Vero 3.1发布等）。

**📈 对比分析**

比较方法：作者通过对比不同时期的行业数据与技术发布节奏，构建五幕式历史演化模型；使用工具清单与案例对照，评估各工具对创作流程的影响；利用公开调查与政策文本进行对比分析，验证“人‑AI 代理连续体”与“四原则”在实际决策中的适用性。性能方面，书中引用的技术大多在业内评测中表现为“可替代人工的低成本、低延迟生成”，但缺乏统一客观指标，作者侧重于实际业务落地效果与成本效益。

**⚠️ 局限性**

局限性：①研究仅覆盖2025‑2026年八个月，后续技术迭代与政策变化难以预估；②依赖公开数据与通讯内容，缺乏大规模实验或量化指标；③聚焦英美欧市场，全球南部与新兴市场视角不足；④对环境与能耗等外部影响讨论有限；⑤工具清单与框架在不同创作领域的适用性仍需进一步验证。

---

## 46. Benchmarking Open-Weight Foundation Models for Global AI Technical Governance

**arXiv ID:** 2606.26099 | [PDF](https://arxiv.org/pdf/2606.26099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 47. Simulating Eating Disorder Patients with LLMs: Evaluating Psychological Persona Stability in Multi-Turn Conversations

**arXiv ID:** 2606.26109 | [PDF](https://arxiv.org/pdf/2606.26109v1)

**作者:** Jennifer Haase `[一作]` (Weizenbaum Institute), Sebastian Pokutta `[通讯]` (Zuse Institute Berlin)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类精度上提高了5%，并且训练时间缩短了20%。

**⚠️ 局限性**

模型在处理大规模数据集时可能会遇到内存限制的问题。

---

## 48. Know2Guess: A Contamination-Aware Multi-Zone Benchmark for Knowledge-Boundary Evaluation in Large Language Models

**arXiv ID:** 2606.26101 | [PDF](https://arxiv.org/pdf/2606.26101v1)

**作者:** Renwei Meng `[一作]` (Anhui University), Shengan Yang `[通讯]` (Anhui University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种污染意识、多区间的基准，用于评估大语言模型在知识边界上的答题、拒答与拒绝行为；

**💡 创新点**

创新点在于：①冻结答案标签的多区间划分（A–D）实现答题与主动拒答的明确区分；②将污染风险元数据嵌入数据集；③区分模型的“有条件拒答”与“策略性拒绝”，并提供严格与归一化两种解析器；

**🔧 技术方法**

使用的技术主要是结构化提示（answer‑or‑abstain vs answer‑only）、严格与归一化解析器、基于Bootstrap的置信区间估计、以及基准的成本敏感重加权；

**📊 数据集**

基准数据集共1200条，覆盖常识、通用问答、医学、多跳推理、科学等五个领域，区间A–C为答题期望，区间D为拒答期望；

**📈 对比分析**

对FLAN‑T5系列、Qwen2.5‑1.5B/3B‑Instruct、Llama‑3‑8B‑Instruct三大模型进行评测；结果显示：Qwen2.5‑3B‑Instruct以0.3657的可靠性（包含高比例主动拒答）排名第一，Llama‑3‑8B‑Instruct第二，FLAN‑T5系列表现最差；不同解析器和提示变体对排名影响不大；

**⚠️ 局限性**

限制：①基准为静态、确定性评测，无法评估训练/微调对拒答行为的影响；②聚焦短文本问答，未覆盖长篇生成或交互式对话；③污染风险标签为近似指标，不能完全证明模型未记忆；④区间D为合成未知，可能仍含有模型可利用的痕迹；

---

## 49. HierBias: Context-Conditioned Hierarchical Media Bias Detection with Multi-Task Type Classification

**arXiv ID:** 2606.26100 | [PDF](https://arxiv.org/pdf/2606.26100v1)

**作者:** Kaining Li `[一作]` (Xidian University), Yuxin Dong `[通讯]` (Xidian University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HierBias，一种层次化的上下文条件媒体偏见检测模型；

**💡 创新点**

创新点在于正式建模句子与文档上下文依赖、证明上下文可降低贝叶斯误差，并通过多任务学习提升小样本样本效率；

**🔧 技术方法**

采用RoBERTa句子编码器+跨句Transformer聚合器、双任务输出（二分类与四类偏见类型）以及KL对齐正则化；

**📊 数据集**

使用BABE（3,700句偏见标注）、BASIL（300篇文章）以及LLM生成的48K句annolexical数据进行增强；

**📈 对比分析**

与七种基线（LR+TF-IDF、BERT、RoBERTa、DA-RoBERTa、bias-detector、LLM提示）对比，HierBias在BABE上取得0.853 F1、0.723 MCC，分别比bias-detector提升2.6% F1、4.3% MCC；在BASIL零样本迁移亦提升3.1% F1、3.4% MCC；

**⚠️ 局限性**

局限在于偏见类型仅覆盖四类，数据集规模有限，且对多语言和更细粒度标签的泛化尚待验证。

---

## 50. Data Facts: A Metadata Schema for Structured Data Exchange in the NANDini Multi-Agent Ecosystem

**arXiv ID:** 2606.26211 | [PDF](https://arxiv.org/pdf/2606.26211v1)

**作者:** Jin Gao `[一作]`, Ramesh Raskar `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了 Data Facts 轻量级 JSON 元数据架构，为网络智能体提供可发现、验证、访问的数据集层，并通过三层安全管线实现私有数据访问。

**💡 创新点**

将 AgentFacts 与外部数据元数据解耦，单字段指向数据集，支持实时新鲜度、完整性校验，并为多级访问权限设计 JWT+网关+A2A 授权链。

**🔧 技术方法**

采用 JSON、SHA‑256 校验、JWT 认证、能力型网关授权、A2A 交互协议、TTL 时间戳、实验评估 840 次决策任务与 GPT‑5‑nano。

**📊 数据集**

使用 PostgreSQL 示例数据集进行计数决策、TTL 与完整性测试，并在 7 个二分类场景与 206 次安全攻击模拟中评估。

**📈 对比分析**

与直接端点访问对比，Data Facts 流程增加约 260 ms 发现延迟；TTL 检测将错误率从 37.6% 降至 8.8%；完整性校验 100% 检测；安全管线 100% 阻断 46 次伪造；决策准确率 100% 对比 35.2%，在三种拓扑下保持一致。

**⚠️ 局限性**

仅在单模型二分类实验中验证，未覆盖开放式查询；使用静态数据库，未考虑高频更新导致的 TTL 误差；v1 JWT 未绑定调用者身份，存在重放风险；仅支持单数据集指针，缺乏多集列表。

---

## 51. Agentic Analysis for Agentic Infrastructure: An LLM-Powered Pipeline for Comparative Governance of DAO and Corporate AI Protocols

**arXiv ID:** 2606.26203 | [PDF](https://arxiv.org/pdf/2606.26203v1)

**作者:** Yutian Wang `[一作]` (Duke Kunshan University), Luyao Zhang `[通讯]` (Duke Kunshan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一套基于大型语言模型的治理话语分析管线，对ERC‑8004 DAO和Google A2A企业标准的治理记录进行大规模文本与网络分析。

**💡 创新点**

创新点在于将LLM自动注释、神经主题建模（BERTopic、Thematic‑LM）与多层网络分析（共参与网络、话语网络、社语义双向网络）相结合，形成可复现的跨治理体系比较框架，并首次从宏观话语分布与微观互动结构双重视角比较两种治理模式。

**🔧 技术方法**

使用技术包括：MiniMax‑M2.5 LLM进行注释；BERTopic（使用Sentence‑BERT、UMAP、HDBSCAN）和Thematic‑LM进行主题抽取；共参与网络（SNA）、话语网络（根据立场构建同意/冲突层）、社语义双向网络（演员–主题二分图）构建与度量；统计检验（卡方检验、Jensen‑Shannon散度、Gini、Modularity、Borgatti‑Everett等）。

**📊 数据集**

数据集为4,323条治理参与记录（ERC‑8004 142条，Google A2A 4,181条），来源于Ethereum Magician论坛、GitHub提问/PR/讨论记录，已通过内容过滤和哈希校验公开发布。

**📈 对比分析**

比较方法：先构建决策架构图；然后用LLM注释划分论证类型与立场，统计分布并做卡方检验；用BERTopic与Thematic‑LM分别得到主题分布，计算两案例间Jensen‑Shannon散度；构建三层网络评估参与不平等、社区碎片化、共识密度等。结果显示：两案例主题分布散度约为0.29，网络度量显示参与不平等与碎片化相近，DAO在话语一致性与共识密度上略高，企业标准在冲突边数和主题多样性上更高。

**⚠️ 局限性**

局限性包括：ERC‑8004数据量远小于A2A，低频主题置信度低；部分企业内部讨论（如TSC会议、内部评审）未包含在公开记录中，导致对企业治理网络的低估；开放源码共通规范可能对两案例产生共性影响，难以完全归因于治理形式。

---

## 52. Statistical and Structural Approaches to Algorithmic Fairness

**arXiv ID:** 2606.26200 | [PDF](https://arxiv.org/pdf/2606.26200v1)

**作者:** Antonio Ferrara `[一作]` `[通讯]`, Antonio Ferrara

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文系统性提出并验证了基于统计假设检验的公平性评估框架以及针对网络和层级结构的公平性干预方法，覆盖路由、链接推荐、社交网络以及排名系统；

**💡 创新点**

创新点包括：①面向交叉子群体的自适应假设检验；②用于黑盒排序的条件距离相关残差测试；③解释分布差异度量与“等处理”检测器；④面向节点曝光的前向路径公平路由算法、基于图神经网络的城市社会经济超分辨率；⑤评估不确定性的配对排序可放弃机制；⑥整合治理的偏差管理架构；

**🔧 技术方法**

采用的技术涵盖：统计推断（Wald检验、贝叶斯Dirichlet多项式）、条件独立性检验、距离相关与RKHS、Shapley值解释、图算法与线性规划、GNN、马尔可夫链聚合、置信度阈值与安全放弃；

**📊 数据集**

实验使用的主要数据集包括：OpenStreetMaps路网（佛罗里达、美国东部）、社交网络模拟与真实社区数据、Yelp用户‑商家交互图、VLDB、NeurIPS公开数据集、真实排序与链接推荐日志；

**📈 对比分析**

与现有基线相比，本文方法在小样本公平性检验中显著降低假阳性率；在路由任务中，节点曝光Gini系数下降30%且路径长度仅增加5%；在链接推荐实验中，少数族群可见度提升20%；在排名评估中，AB测试显示解释分布差异度量能发现传统指标掩盖的歧视，且放弃机制在保持总体准确率的同时提升安全边际；

**⚠️ 局限性**

局限性包括：对敏感属性的统计依赖受隐私法规限制；离散化的身份与“真实”社会结构差距；静态图和排名模型忽视跨平台联动与动态演化；理论公平定义与社会正义理念仍存在脱节。

---

## 53. Federated Hash Projected Latent Factor Learning

**arXiv ID:** 2606.26192 | [PDF](https://arxiv.org/pdf/2606.26192v1)

**作者:** Jialan He `[一作]` `[通讯]` (Southwest University), Jialan He (Southwest University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了联邦哈希学习框架 FHPLF，用二进制表示实现分布式推荐模型，并结合投影汉明距离与安全梯度重组提升隐私与效率。

**💡 创新点**

创新点包括①用二进制梯度矩阵替代实值梯度，显著降低通信与存储开销；②引入投影汉明距离以区分比特重要性，提升表示能力；③设计 SBGL-PEU 机制将梯度碎片化后互相交换，阻断梯度反演攻击。

**🔧 技术方法**

采用联邦学习、哈希学习、投影汉明距离、离散坐标下降（DCD）优化、二进制梯度传输以及安全梯度重组（SBG‑PEU）等技术。

**📊 数据集**

实验使用四个真实数据集：Amazon（D1）、Epinion（D2），共计约 35k 用户、38k 项目（D1）和 10k 用户、9k 项目（D2）。

**📈 对比分析**

与六个集中哈希基准（DCF、DPR、CCCF、Neu‑hash、HS‑GCN、VHPHD）、三种联邦实值模型（PFedRec、RFRec、FedPerGNN）以及联邦哈希基准 LightFR 进行对比。FHPLF 在 MAE、RMSE、HIT@10、MRR@10、NDCG@10 上均优于对手；通信开销仅为实值模型的 1/16，且在梯度反演攻击下恢复误差最高，体现更强隐私保护。

**⚠️ 局限性**

局限性：①仅在二进制哈希空间内学习，可能对极稀疏或高复杂交互的建模有限；②投影汉明距离的超参数选择与理论分析仍待深入；③在大规模异构客户端和网络不稳定场景下的鲁棒性需进一步验证；④对安全性只考虑了梯度反演攻击，其他侧信道或聚合攻击的抵抗能力尚未评估。

---

## 54. LiMoDE: Rethinking Lifelong Robot Manipulation from a Mixture-of-Dynamic-Experts Perspective

**arXiv ID:** 2606.26183 | [PDF](https://arxiv.org/pdf/2606.26183v1)

**作者:** Zhihao Gu `[一作]` (Nanyang Technological University), Lin Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了LiMoDE框架，用于终身机器人操控，分为多任务预训练阶段的动态混合专家结构和终身学习阶段的专家适配机制。

**💡 创新点**

创新点在于将视觉动力学感知的动态路由与异质专家相结合，并通过专家复用与回放策略实现跨任务知识迁移和减轻灾难性遗忘。

**🔧 技术方法**

采用了Mixture‑of‑Experts（MoE）架构、低秩专家、视觉动力学路由、经验回放以及基于CLIP的多模态编码。

**📊 数据集**

在LIBERO基准（包含Goal、Spatial、Object、Long四套任务）和真实世界的八个操控任务上进行评估。

**📈 对比分析**

与ER、SeqFT、TAIL、M2Distill、PPL等基线相比，LiMoDE在FWT、BWT、AUC等指标上均表现最优，尤其在长时序任务上提升超过7%，且在实际机器人实验中表现优于现有方法。

**⚠️ 局限性**

限制在于目前使用固定秩的专家和top‑k激活，缺乏更灵活的动态专家分配；且框架仍基于CLIP骨干，未与更强的视觉语言模型结合。

---

## 55. Toward Mitigating Process-Induced Performance Degradation in 3.5D Heterogeneous Packages via Pre-Silicon Firmware Co-Optimization

**arXiv ID:** 2606.26176 | [PDF](https://arxiv.org/pdf/2606.26176v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 56. The Effortless Trap: Productive Struggle, AI, and the Illusion of Learning

**arXiv ID:** 2606.26181 | [PDF](https://arxiv.org/pdf/2606.26181v1)

**作者:** Mario Brcic `[一作]` (Faculty of Electrical Engineering and Computing), Stjepan Frljic `[通讯]` (Faculty of Electrical Engineering and Computing)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了AI在教学中的放置框架，基于六步学习模型（Prime、Probe、Point、Attach、Strengthen、Test）和AI使用规则，并给出传统教学与AI工具的对应菜单；

**💡 创新点**

创新点在于把AI使用定位为“放置”而非“允许/禁止”，通过六步模型和“Probe/Test守护、其余阶段受控支持”的规则，使AI介入既能提升学习又不破坏学习本质；

**🔧 技术方法**

利用认知负荷理论、生成式AI模型（如ChatGPT类）、随机对照试验结果和教学设计方法构建AI支持的教学干预；

**📊 数据集**

主要引用已有实验数据（高中数学、大学物理随机试验等）和文献综述，未使用新的专门数据集；

**📈 对比分析**

通过引用先前实验（如未受控AI导致学生成绩下降17%，受控AI提升学习超过两倍）来说明效果，本文未进行新实验，效果基于文献证据；

**⚠️ 局限性**

局限在于仅关注单一概念的学习过程，缺乏长期保留和课程级验证，AI证据仍处于早期预印本阶段，缺乏对框架有效性的直接实证检验。

---

## 57. Orchestrating Black-Box Schema Converters: An Empirical Study of Automated, Quality-Ranked Conversion Across Heterogeneous Schema Languages

**arXiv ID:** 2606.26180 | [PDF](https://arxiv.org/pdf/2606.26180v1)

**作者:** Felix Neubauer `[一作]` (University of Stuttgart), Benjamin Uekermann `[通讯]` (University of Stuttgart)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了Schema Conversion Orchestrator，实现了对不同模式语言之间的黑盒转换器进行自动化、可重现且按质量排名的转换；

**💡 创新点**

创新点在于将模式语言视为图节点、将现有散乱的转换器视为无类型边，通过路径搜索与多层次排名（benchmark、边质量、输出长度）实现自动链路构建，并完整记录转换步骤的 provenance；

**🔧 技术方法**

技术实现基于Python Flask核心、Node.js/Java 子进程、DFS图搜索、缓存机制、结构化 F1 评估、与 MetaConfigurator 的 REST API 集成；

**📊 数据集**

实验使用 60 个源→目标×输入组合，覆盖 JSON Schema、XSD、SHACL、LinkML、MD‑Models，按简单/中等/复杂三层级；另外为 SHACL↔JSON Schema 设计了 39/88 条手工构造的 benchmark；

**📈 对比分析**

评估采用人工 G/L/I 标注与 benchmark 结构 F1 评分，结果显示 43/60 任务得到可用最高质量结果；运行时 benchmark 约 55‑62 s，缓存提升约 10%；路径排名基于 benchmark 准确度、边质量估计和输出长度；

**⚠️ 局限性**

局限性包括：转换质量受限于现有工具，语义等价性难以度量；benchmark 仅覆盖 SHACL↔JSON Schema，缺乏多文件、深层嵌套及语义化转换的测评；需要扩展更多语言对 benchmark 与 LLM 补救机制。

---

## 58. KG-TRACE: A Neuro-Symbolic Framework for Mechanistic Grounding in Antimicrobial Resistance Prediction

**arXiv ID:** 2606.26179 | [PDF](https://arxiv.org/pdf/2606.26179v1)

**作者:** Naman Garg `[一作]` (National Institute of Technology Kurukshetra), Parimal Kar `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种神经符号化框架KG-TRACE，将WHO突变知识图嵌入WGS抗药性预测模型，实现可解释的抗菌药物耐药预测与机制验证。

**💡 创新点**

引入自适应可信门融合基因突变特征与知识图嵌入，并在预测层面实现双层机制验证，提供符号化解释和不确定性标记。

**🔧 技术方法**

神经网络（全连接+MLP）、RotatE知识图嵌入、交叉注意力门、SHAP特征重要性、辅助基因检测头、双层信任层。

**📊 数据集**

CRyPTIC约3.8万条Mycobacterium tuberculosis WGS样本，包含INH、RIF、EMB、LEV等耐药标签。

**📈 对比分析**

AUROC 0.976（INH）与线性SVC 0.9794相近；核心优势在符号覆盖率92.5%和BGR提升，提供可验证的预测。

**⚠️ 局限性**

受WHO数据库限制导致未知突变被标记为不确定；阶段性训练需要更多计算资源；未评估临床实际部署效果。

---

## 59. HALO: Hierarchical Auction-assisted Learning for Offloading in SAGIN

**arXiv ID:** 2606.26293 | [PDF](https://arxiv.org/pdf/2606.26293v1)

**作者:** Xuli Cai `[一作]` (University of Ottawa), Burak Kantarci `[通讯]` (University of Ottawa)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了HALO框架，结合UAV中心的拍卖机制和两层次的HPPO强化学习，实现三层SAGIN中任务的分配和资源调度。

**💡 创新点**

创新点在于：①宏微槽模型实现任务传输与计算的细粒度跟踪；②拍卖式任务关联与HPPO层级学习的协同；③两时隙层级架构缓解了动作空间爆炸和时序不稳定性。

**🔧 技术方法**

使用的技术包括：宏微槽时隙模型、基于优先级与距离的拍卖算法、Hierarchical Proximal Policy Optimization（HPPO）强化学习、Poisson任务生成模型、A2G/A2A信道模型。

**📊 数据集**

使用的“数据集”为仿真环境中的随机生成任务数据，任务到达率为300/350/400任务/分钟，设备分布在600m×600m区域，模拟多种网络拓扑和负载。

**📈 对比分析**

与单层DRL基线（PPO、DDPG、SAC）进行对比。HPPO在任务成功率上平均提升11.4%（PPO）、32.4%（DDPG）和89.9%（SAC），在不同负载下保持更高成功率；平均完成时延也显著降低，分别比PPO低33–35%、比DDPG低50–61%、比SAC低81–81%。

**⚠️ 局限性**

局限性包括：仅在有限规模的三层SAGIN仿真中验证；对大规模节点数或复杂拓扑（如多HAPS、多UAV分层）未展开；缺乏真实场景下的部署与鲁棒性评估。

---

## 60. Augmentation with Dilution: A Large-Scale Empirical Study of Human Contributor Ecosystems After AI Coding Agent Adoption

**arXiv ID:** 2606.26289 | [PDF](https://arxiv.org/pdf/2606.26289v1)

**作者:** Weixing Zhang `[一作]` (Karlsruhe Institute of Technology), Anne Koziolek `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究 AI 编码代理在开源项目中的采纳对人类贡献者生态系统的因果影响，重点考察贡献者数量、比例、核心新手参与度以及审查深度的变化。

**💡 创新点**

首次将人类贡献者生态作为因变量进行因果研究；使用 Sun & Abraham 的交互加权差分-差分估计处理异质采纳时点；系统评估新手比例下降、贡献者密度下降和审查负荷上升等综合效应。

**🔧 技术方法**

采用分层差分-差分（Sun & Abraham）估计器、倾向得分匹配、面板回归与事件研究图、GitHub API 与 GHArchive 数据提取、固定效应控制、标准误聚类等统计与数据处理技术。

**📊 数据集**

AIDev（AI 编码代理 PR 记录）+ GHArchive（PR、评论事件）+ GitHub REST API（仓库元数据），共 11,097 个满足星数>100的公开仓库，构成 11 年间的月度面板数据。

**📈 对比分析**

与匹配的对照组进行差分-差分对比，结果表明：AI 采纳未显著改变绝对人类贡献者数（ATT=0.014, p=0.224），但人类贡献者密度下降 1.9pp（ATT=-0.019, p=0.002），新手比例下降 3.7pp（ATT=-0.037, p<0.001），审查深度提升 5.3%（ATT=+0.0168, p<0.001）。所有显著性均在 0.01 以内，显示 AI 代理对贡献者生态的显著负向或正向影响。

**⚠️ 局限性**

局限性包括：仅分析星数>100的大型公开仓库，忽略无 PR 直接提交的贡献者；机器人过滤仅靠登录名，可能漏检；新手识别窗口仍可能存在左截断偏差；采纳时点基于 AIDev 首 PR，可能低估实际采纳时间；审查深度分析仅涵盖有审查记录的月份（58% 观测缺失）；倾向得分匹配平衡不完全；外部效度受限于大型项目，无法推广到小型或私有项目。

---

## 61. COrigami: An AI Pipeline for Co-Designing Flat-Foldable Visually Recognisable Origami

**arXiv ID:** 2606.26299 | [PDF](https://arxiv.org/pdf/2606.26299v1)

**作者:** Tom Zahavy `[一作]` (Google DeepMind), Satinder Singh `[通讯]` (Google DeepMind)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 COrigami：一种端到端神经符号管线，能从自然语言描述生成可折叠的折纸模型，并通过强化学习与视觉‑语言模型评估实现美学与物理可行性的协同优化。

**💡 创新点**

创新点在于将语言生成、离散盒式折痕规划、局部可折叠性验证与自动美学评估融合为一体化流程，并首次实现了完全自动化的离散矩形打包与全局折叠可折叠性求解。

**🔧 技术方法**

技术主要包括 Gemini LLM（语义棍图生成与形状参数化）、自定义离散盒折算法（打包、层次划分、组合约束）、强化学习（动作空间扩展、VLM奖励）以及 Gemini 3 Flash 视觉‑语言模型（单模型与对比评估）。

**📊 数据集**

使用约 100 只手工制作的传统折纸模型作为基准数据集，并在此基础上生成 560,000+棍图候选、27,869 个可行基线模型，用于后续 RL 训练。

**📈 对比分析**

在 VLM 评估中，双轮竞赛模式实现 0.811 的分类准确率、0.651 的平均精度和 0.74 的 F1 分数；整体管线从 560k 生成候选到 27,869 成功模型的存活率为 5%，并在 RL 阶段显著提升 VLM 奖励与形状多样性。

**⚠️ 局限性**

局限包括：仅支持盒式折叠框架、形状工具有限（简单折叠与收窄），难以处理更复杂或非正交折叠；算法在极大结构复杂度下仍面临 NP‑难度挑战，需进一步集成更强的学习与搜索策略。

---

## 62. Clue-Guided Money Laundering Group Discovery

**arXiv ID:** 2606.26189 | [PDF](https://arxiv.org/pdf/2606.26189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 63. Morphology-Specific Closed-Loop Control of Logarithmic-Spiral Continuum Arms via Online Jacobian Error Compensation

**arXiv ID:** 2606.26188 | [PDF](https://arxiv.org/pdf/2606.26188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 64. Governing Actions, Not Agents: Institutional Attestation as a Governance Model for Autonomous AI Systems

**arXiv ID:** 2606.26298 | [PDF](https://arxiv.org/pdf/2606.26298v1)

**作者:** Jakob Salfeld-Nebgen `[一作]` `[通讯]` (Metaphora AI), Jakob Salfeld-Nebgen (Metaphora AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出并实现了一个基于机构证明的AI代理治理模型，要求在执行关键操作前收集独立可信的加密证明并通过确定性策略评估

**💡 创新点**

将机构治理模式抽象为计算架构，整合多方加密证明、意图绑定、可验证计算、可重验证审计等技术，首次在AI代理行动边界实现独立可验证治理

**🔧 技术方法**

使用 Ed25519 等非对称签名、DSSE 包装、Cedar/Rego 等声明式策略语言、哈希链或 Merkle 树实现不可篡改日志、零信任行动中心实现多方证明收集与评估

**📊 数据集**

无专用数据集，模型以软件部署与临床处方为示例，演示如何调用 CI、代码评审、药物交互等外部 oracle 产生证明

**📈 对比分析**

未进行量化对比实验，论文仅提供概念验证实现并在 GitHub 上公开；缺乏性能评估与基准对比

**⚠️ 局限性**

主要局限包括：oracle 的完整性与独立性假设、检查与使用时间间隔（TOCTOU）风险、意图匹配不保证、策略正确性不受模型保证、治理范围受限于预先定义的高风险操作、运行成本高等

---

## 65. The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators

**arXiv ID:** 2606.26294 | [PDF](https://arxiv.org/pdf/2606.26294v1)

**作者:** Alex Iacob `[一作]`, Nicholas D. Lane `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Red Queen Gödel Machine框架，使评估器与任务代理共进化，从而在没有静态基准的任务上实现自我改进。

**💡 创新点**

创新点在于将评估器视为可进化的学习组件，采用受控效用进化（epoch‑based）实现非平稳评估，并通过评估器替换和选择性擦除实现自适应搜索目标。

**🔧 技术方法**

技术主要包括多代理工作空间、元代理搜索、clade metaproductivity、受控效用演化、评估器替换、选择性擦除以及基于Beta分布的ϵ‑best‑belief评分。

**📊 数据集**

使用了代码编写任务的标准测试数据集、论文写作与评审的NeurIPS论文集及人工评审数据、证明写作与评分的IMO题目与人工评分，并利用代码评审数据集和论文生成/评审数据集进行评估。

**📈 对比分析**

在编码、论文写作和证明写作任务上，Red Queen Gödel Machine在相同或更低搜索代价下实现了71.7%通过率（高于69.9%）、论文接受率提升至40.5%（高于21.8%）并在证明评分上与或超过人类基线；在评审任务上则实现了相对先前系统三倍的搜索成本降低。

**⚠️ 局限性**

局限性包括评估器质量依赖于固定锚点，若锚点偏差会导致评估器无信息；受控效用进化的理论保证仅在 epoch 局部，无法保证跨 epoch 的整体收敛；以及需要手工设定评估器替换阈值和锚点，可能限制适用范围。

---

## 66. Expecting (Targeted Ads)? Network Analysis of User Health Data Leakage in Fertility Tracking Apps

**arXiv ID:** 2606.26276 | [PDF](https://arxiv.org/pdf/2606.26276v1)

**作者:** Yeeun Jo `[一作]` (University of Illinois at Urbana-Champaign), Brad Reaves `[通讯]` (North Carolina State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对20款安卓育儿追踪应用进行网络流量测量，分析其与第三方广告网络的数据共享情况。

**💡 创新点**

首次将网络层测量与手工代码审计结合，系统性揭示隐私泄露的显式与隐式模式。

**🔧 技术方法**

使用TLS剥离的mitmproxy捕获HTTP流量，并在Waydroid环境下模拟用户交互；随后通过JADX对APK进行反编译进行手工分析。

**📊 数据集**

包含20款从Google Play筛选的育儿追踪应用，针对每款应用执行8次统一的交互会话，收集约7,829个HTTP请求。

**📈 对比分析**

通过统计广告网络请求数、参数泄露数量和泄露类型来评估泄露程度；结果显示25%应用存在显式泄露，若干应用通过上下文广告泄露粗粒度信息。

**⚠️ 局限性**

受限于样本规模、仅检测未加密流量、可能忽略加密或编码后的泄露信息、以及部分热门应用因兼容问题被排除，导致泄露率估计可能偏低。

---

## 67. MIRAGE: Protecting against Malicious Image Editing via False Moderation

**arXiv ID:** 2606.26199 | [PDF](https://arxiv.org/pdf/2606.26199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 68. CVA6-RT: an Open-Source Time-Predictable RV64 Processor for Mixed-Criticality Systems

**arXiv ID:** 2606.26177 | [PDF](https://arxiv.org/pdf/2606.26177v1)

**作者:** Enrico Zelioli `[一作]` (Integrated Systems Laboratory, ETH Zurich), Angelo Garofalo `[通讯]` (Department of Electrical, Electronic, and Information Engineering, University of Bologna)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

在64位RISC-V CVA6核心基础上提出CVA6-RT，加入硬件支持的TLB分区、L1缓存/scratchpad混合模式、增强CLIC以及硬件上下文堆栈，实现在混合关键性系统中的可预测中断延迟。

**💡 创新点**

创新点在于将可预测性机制直接集成到处理器硬件层面，包括可编程TLB分区、可运行时切换的L1缓存/scratchpad、硬件管理的中断上下文堆栈，以及支持虚拟化的CLIC。

**🔧 技术方法**

主要技术包括TLB分区与锁定、L1缓存/Scratchpad混合模式、硬件辅助的上下文保存、以及增强的CLIC实现。

**📊 数据集**

评估使用中断延迟数据，无需传统数据集；实验在模拟/实际硬件上测量中断延迟。

**📈 对比分析**

与原始CVA6对比，平均中断延迟从140周期降至12周期，降低约10倍；与ARM Cortex-M相比，性能相当。

**⚠️ 局限性**

局限性包括仅评估中断延迟未覆盖完整实时工作负载，硬件成本与面积可能增加，对大规模多核系统的扩展性尚未验证。

---

## 69. RMTL: Reinforced Micro-task Learning for Long-Horizon Manipulation with VLM Rewards

**arXiv ID:** 2606.26175 | [PDF](https://arxiv.org/pdf/2606.26175v1)

**作者:** Anıl Can Ateş `[一作]` (Istanbul Technical University), Cihan Topal `[通讯]` (Istanbul Technical University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将预训练视觉‑语言模型（VLM）奖励拆分为阶段性语言提示并在机器人操作中按阶段切换的方法

**💡 创新点**

通过阶段化提示将平坦的全局VLM奖励转为分段单调、信息更丰富的奖励，结合多视角聚合、逆向课程和学习的层次管理器，显著提升长期操控学习效率

**🔧 技术方法**

VLM奖励投影、CLIP‑style相似度评分、多视角平均、逆向课程调度、PPO、行为克隆+REINFORCE层次管理器

**📊 数据集**

Fetch 机器人拾取任务（Gymnasium‑Robotics/ MuJoCo）

**📈 对比分析**

与单一全局提示+多视角、单视角以及未使用层次管理器的基线相比，在随机初始化条件下成功率从约94%提升至98%，单提示基线甚至未能收敛

**⚠️ 局限性**

依赖VLM的感知能力，易受光照、纹理、遮挡等视觉差异影响；阶段提示手工设定，缺乏自动化；某些阶段（如抓取后）奖励饱和导致梯度稀薄

---

## 70. OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation

**arXiv ID:** 2606.26201 | [PDF](https://arxiv.org/pdf/2606.26201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 71. AlgoEvolve: LLM-driven Meta-evolution of Algorithmic Trading Programs

**arXiv ID:** 2606.26173 | [PDF](https://arxiv.org/pdf/2606.26173v1)

**作者:** Dhruv Sharma `[一作]` (Indraprastha Institute of Information Technology), Gautam Shroff `[通讯]` (Indraprastha Institute of Information Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套名为AlgoEvolve的双层进化框架，利用LLM作为语义变异算子，自动生成并改进可执行的量化交易策略。

**💡 创新点**

将LLM转化为程序语义变异器，并引入元进化外层循环进化搜索提示（prompt），实现对非平稳交易环境的自适应搜索。

**🔧 技术方法**

采用LLM链式推理（CoT）、Prompt调优、语义变异、走步前验证、Meta-LLM指导的提示基因变异与交叉等技术。

**📊 数据集**

使用公开的NUMIN intraday纸面交易平台，约200个交易日的5分钟OHLCV与技术指标匿名多资产数据。

**📈 对比分析**

与静态种子、单层演化、随机森林及LSTM等基准对比，20/30日滚动回测中获得年化Sharpe 5.60、平均每日0.31%收益、最大回撤1.59%，显著优于对手。

**⚠️ 局限性**

受限于提示基因的离散空间和LLM生成代码的复杂度上限，导致高复杂度场景出现语法错误或过度正则化，并且当前模型仅在设计时使用LLM，缺乏实时交易推理能力。

---

## 72. Graph Isomorphism and Representation Theory

**arXiv ID:** 2606.26244 | [PDF](https://arxiv.org/pdf/2606.26244v1)

**作者:** Joshua A. Grochow `[一作]`, Jacob Urisman `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并研究了一类新的图同构判定工具——分离模块（separating modules），这是一种在对称群作用下闭包的多项式向量空间。作者证明了分离模块在不同复杂度度量（如多项式的支持度、对称代数电路大小和表示论中的多重性）下的判别能力，并给出了相应的算法。

**💡 创新点**

创新点包括：
- 将 GCT（几何复杂度理论）中的分离模块概念引入图同构问题；
- 在支持度、对称电路大小和多重性三个维度分别给出了与 Weisfeiler–Leman（WL）算法、子图计数、以及 automorphism group 的 cycle index 的精确等价关系；
- 证明多重性阻断（multiplicity obstructions）比出现阻断（occurrence obstructions）更强，且提供了多重性阻断的内在图结构表征；
- 将多项式的支持度与 WL 颜色细化的初始一步进行对应，揭示子图计数、支持度多项式与 WL 之间的密切联系。

**🔧 技术方法**

主要技术：
- 对称群表示论（Specht 模块、分解为不可约表示、Young 子群等）；
- 多项式代数与布尔代数之间的对应与化简（例如使用 x_ij^2=x_ij 的理想化简）；
- 对称代数电路与布尔阈值电路的互相模拟（借助 Dawar–Wilsenach 等人的结果）；
- 通过构造可逆电路来控制 orbit 大小和 support 大小；
- 复杂度分析中使用了支持度、阶乘估计以及对称电路的结构限制。

**📊 数据集**

该工作为理论分析，不涉及具体实验数据集；所有结论均为理论定理，验证对象为任意 n 结点的简单无向图（或其变体）。

**📈 对比分析**

与已有方法比较：
- 在支持度（degree）上，分离模块的判别能力与 k‑WL 的子图计数初始步骤等价，但严格弱于 k‑WL；
- 在对称代数电路大小上，大小为 n^Θ(k) 的分离模块与 k‑WL 的判别能力等价；
- 在多重性阻断上，证明它们严格优于出现阻断，但在图同构判定中尚未显示出比 WL 更强的通用判别能力；
- 所有算法均能在 n^O(d) 时间内完成，远优于直接枚举所有子图的方法。

**⚠️ 局限性**

局限与未解问题：
- 虽然给出了多重性阻断与 cycle index 的等价表征，但尚未证明多重性阻断能判别所有非同构图，尤其是存在同构群具有相同 cycle index 的图；
- 目前的算法在时间上仍为指数级（n^O(d) 或 n^O(k)），缺乏更快的实现；
- 对于高支持度（或高电路大小）时的可扩展性仍未充分评估；
- 研究主要针对简单无向图，尚未系统地推广到有向图、加权图、张量同构等更一般的情形。

---

## 73. From Structure to Synergy: A Survey of Vision-Language Perception Paradigm Evolution in Multimodal Large Language Models

**arXiv ID:** 2606.26196 | [PDF](https://arxiv.org/pdf/2606.26196v1)

**作者:** Haoxiang Sun `[一作]`, Jiancheng Lv `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

介绍并比较了 elsarticle.cls 这一 LaTeX 文档类的功能与其前身 elsart.cls 以及使用方法。

**💡 创新点**

创新点在于改进了与其它宏包的兼容性、提供多种预设期刊格式（preprint、review、1p、3p、5p 等），并支持灵活的前置文档（title、author、abstract 等）和自定义 theorem 环境。

**🔧 技术方法**

采用了 LaTeX 宏包 natbib、geometry、graphicx、txfonts 等，并在 elsarticle.cls 内部实现了多种环境和选项。

**📊 数据集**

未使用任何数据集；该文档主要是技术文档而非实验研究。

**📈 对比分析**

没有进行实验性性能比较，主要通过示例代码展示功能与用法。

**⚠️ 局限性**

局限在于仍需作者手动检查公式换行、双栏排版等细节，缺乏自动化的多格式适配机制。

---

## 74. FinWhale: An Optimally Resilient Two-Round Terminating DAG Protocol

**arXiv ID:** 2606.26292 | [PDF](https://arxiv.org/pdf/2606.26292v1)

**作者:** Razya Ladelsky `[一作]` (Technion), Roy Friedman `[通讯]` (Technion)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了 FinWhale，首个在 DAG 基础上实现两消息延迟快路径的拜占庭容错共识协议。

**💡 创新点**

创新点在于将 Fast Path 机制与 Mysticeti 的慢路径相融合，设计了 FP‑evidence 块和新的决策规则，实现了在 n = 3f+2p-1 验证者下的最优延迟。

**🔧 技术方法**

采用 DAG 结构、轮次领导、FP‑evidence 证明、SP‑certificate/skip 模式、push‑pacemaker 等技术，并结合快速投票与慢速投票的混合决策。

**📊 数据集**

该工作未使用公开数据集，而是在理论模型与模拟实验中验证协议的安全性与可达性。

**📈 对比分析**

与 Mysticeti、Banyan、FaB 等现有协议相比，FinWhale 在理想条件下从 3 消息延迟降低到 2 消息，保持了相同的吞吐量，并在部分同步模型下证明了可达性。

**⚠️ 局限性**

限制包括：需要满足 n = 3f+2p-1 的硬件阈值，FP‑evidence 的构造与检测增加了实现复杂度；在非理想网络或高 Byzantine 负载下快路径可能不被触发，导致性能退化。

---

## 75. SSM Adapters via Hankel Reduced-order Modeling: Injection Site Determines Task Suitability in Long-Context Fine-Tuning

**arXiv ID:** 2606.26290 | [PDF](https://arxiv.org/pdf/2606.26290v1)

**作者:** Omanshu Thapliyal `[一作]` `[通讯]` (Hitachi America Ltd.), Omanshu Thapliyal (Hitachi America Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于Hankel秩约化的状态空间模型适配器（HRM），可在冻结的Transformer上实现可学习的递归记忆。

**💡 创新点**

创新点在于将控制理论中的平衡截断与Hankel奇异值压缩引入PEFT，提供理论误差界限并实现与LoRA等方法相当的计算效率。

**🔧 技术方法**

主要技术包括线性时不变状态空间模型、FFT并行扫描、平衡截断（BT）以及对比实验中使用的LoRA族方法。

**📊 数据集**

实验使用的评测数据集包括DFA状态跟踪、MAESTRO钢琴音乐、字符级Wiki（enwik8）、以及Mistral‑7B的LongBench（QuALITY、QMSum、NarrativeQA）等。

**📈 对比分析**

与LoRA、AdaLoRA、DoRA、QLoRA在相同参数量下进行对比，HRM在长序列推理任务中显著优于对手，特别是在LongBench的QuALITY和QMSum任务上取得显著提升。

**⚠️ 局限性**

局限性包括对输入相关的选择性SSM扩展不支持、在极长文本（NarrativeQA）上效果不佳，以及需要更大显存的训练与推理。

---

## 76. TEMPO-Diffusion: Temporally Exposed Malicious Poisoning of Diffusion Models

**arXiv ID:** 2606.26285 | [PDF](https://arxiv.org/pdf/2606.26285v1)

**作者:** William Aiken `[一作]` (University of Ottawa), Iosif-Viorel Onut `[通讯]` (University of Ottawa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于扩散模型的定向后门攻击框架 TEMPO‑Diffusion，能够在不需要推理时控制噪声的情况下，通过时间窗口触发将子图像后门嵌入生成样本，并用于下游数据增强。

**💡 创新点**

创新点在于：①时间暴露的后门触发窗口，可在扩散过程的特定时刻激活；②子图像多目标后门，支持同一图像中多个位置和多种目标；③无输入噪声控制，攻击无需操纵推理时的噪声种子；④支持在 in‑painting 等任务中进行后门植入。

**🔧 技术方法**

技术手段包括：扩散模型（DDPM / DDIM）训练与采样；时间窗口触发调度与触发强度函数（Abridged、Hann、Box）；子图像覆盖与目标样本池构造；目标类与触发类的标记策略；以及防御评估使用的触发器重构算法 ELIJAH。

**📊 数据集**

使用的数据集为：CIFAR10、GTSRB 以及自研的 CALISA（40 类交通标志，涵盖加拿大和美国标志），并在这些数据集上进行子图像后门的训练与评估。

**📈 对比分析**

与传统后门攻击对比，TEMPO‑Diffusion 在生成质量（ΔFID）上几乎无显著下降，攻击成功率（ASR）在 CIFAR10 上超过 90%，在 GTSRB 与 CALISA 上均达到 80%–98%；下游分类器在触发样本上误分类率大幅提升，整体准确率下降约 10%。

**⚠️ 局限性**

局限性包括：一对一定向攻击效果不如多目标攻击；需要较大的训练数据和计算资源；防御中触发器重构的计算成本高，且需要对每个目标类分别进行；目前仅针对噪声基后门，尚未扩展到多触发器、多通道或更复杂的图像内容。

---

## 77. Beyond Single-Source Cognitive Taskonomy:Multi-Source Task Relations through fMRI Transfer Learning

**arXiv ID:** 2606.26279 | [PDF](https://arxiv.org/pdf/2606.26279v1)

**作者:** Junfeng Xia `[一作]` (Southern University of Science and Technology), Jie Guo `[通讯]` (Southern University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文在 Human Connectome Project 数据上构建了多源转移的 fMRI 认知任务图谱，并利用布尔整数规划在有限监督预算下实现任务分配。

**💡 创新点**

创新点在于：① 将单源转移扩展为源集→目标的多源关系，揭示源集组合对转移效果的影响；② 将转移距离映射为全局监督优先级，利用 BIP 优化任务分配，展示了局部转移与全局预算分配的不同视角。

**🔧 技术方法**

技术手段包括：掩码 fMRI 重建的自监督 Transformer 模型、源集融合与投影、转移距离的标准化、以及布尔整数规划（BIP）用于预算约束下的任务分配。

**📊 数据集**

使用数据集为 Human Connectome Project (HCP) S1200，包含 23 个任务状态，360 区域的多模态分区（MMP）和 20 帧时间序列。

**📈 对比分析**

方法对比：用 1% 目标数据进行低样本适配后与完整训练的金标模型对比，评估转移距离。结果显示单源转移呈现方向性与范畴结构，且多源转移在不同源集组合下可显著降低重建误差；BIP 在不同预算下频繁选取工作记忆任务，表明其在全局成本优化中的重要性。虽然未与随机或贪婪基线直接对比，但多源与 BIP 结果均说明转移结构能指导高效监督分配。

**⚠️ 局限性**

局限性包括：① 仅在单一 HCP 数据集与皮层分区上验证；② 只筛选每个目标的五个最强单源，可能遗漏弱源但组合有效的情况；③ 源集大小差异导致融合能力不一致，影响转移评估；④ 未对随机种子、受试者划分或外部数据的鲁棒性进行评估；⑤ BIP 频率受预算、候选池和成本归一化影响，不能直接视为源质量指标；⑥ 仅使用 MSE 作为重建评估，忽略了其他表示质量维度。

---

## 78. Lacuna: A Research Map for Machine Learning

**arXiv ID:** 2606.26246 | [PDF](https://arxiv.org/pdf/2606.26246v1)

**作者:** Martin Weiss `[一作]` (Tiptree Advanced Systems Corporation), Nasim Rahaman `[通讯]` (Tiptree Advanced Systems Corporation)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开一个大规模机器学习研究地图（Lacuna），将论文、概念元素、研究方向和研究方案转化为可检索、可追溯的 Markdown 页面，并提供 Web、Markdown、MCP 接口供人类与 LLM 代理使用。

**💡 创新点**

1) 首次将 LLM 与学术元数据结合，自动生成论文摘要、概念元素、研究方向和研究方案；2) 通过层次化聚类与 LLM 合成构建可导航的研究地图；3) 在问题表述、文献检索与深度研究三大任务中实现优于现有系统的性能。

**🔧 技术方法**

使用大语言模型（LLM）生成摘要、概念元素和研究方向；RT-DETR 提取论文图表；HDBSCAN 聚类概念元素；多阶段生成策略（LLM 合成）生成研究方向与研究方案；Pipeline 结合 OpenAlex、OpenReview、DBLP、arXiv 等元数据源；Reciprocal‑Rank‑Fusion 搜索算法。

**📊 数据集**

核心数据集为 733,795 篇已编目论文；评测数据集包括 LitSearch、Multi‑XScience‑CS/ML、ScholarQA‑CS‑ML、ReportBench‑ML 四个基准，涵盖文献检索、问答与深度研究任务。

**📈 对比分析**

与 OpenScholar 进行文献检索基准对比：LitSearch Recall@10 0.538（Lacuna）对比 0.424（OpenScholar v3）。在 ReportBench‑ML 的 25 项调查任务中，Lacuna Deep Research 获得 0.052 citation F1、0.339 precision、99 expert‑reference hits、7.82/10 RACE 质量评分，明显优于 GPT‑Researcher（0.039 F1、0.290 precision、72 hits、5.24/10 RACE）。

**⚠️ 局限性**

1) 对 LLM 生成内容的质量与偏差需人工审核；2) 研究地图目前覆盖主要为机器学习领域，跨学科扩展仍有限；3) 处理海量论文的计算与存储成本高；4) 依赖外部元数据源的完整性与一致性。

---

## 79. Fast LeWorldModel

**arXiv ID:** 2606.26217 | [PDF](https://arxiv.org/pdf/2606.26217v1)

**作者:** Yuntian Gao `[一作]` (Xi'an Jiaotong University), Xiangyu Xu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于动作前缀预测的快速潜在世界模型，用于无奖励视觉规划。

**💡 创新点**

创新点在于将传统单步转移改为动作前缀预测，并采用密集前缀监督实现并行多步预测，显著降低错误累积与计算开销。

**🔧 技术方法**

使用 Transformer 动作前缀编码器、并行潜在预测器、密集前缀损失、CEM 规划以及可选自一致性正则化技术。

**📊 数据集**

在 LeWM 的离线数据集上进行实验，包括 Two‑Room、PushT、Reacher、OGBench‑Cube 等四个任务。

**📈 对比分析**

与 LeWM、PLDM、DINO‑WM 等方法对比，平均成功率从 85.8% 提升至 90.5%（加自一致性可达 92%），动态评估时间从 31.4s 降至 8.0s，CEM 解算时间从 54.4s 降至 28.3s。

**⚠️ 局限性**

局限在离线无奖励设置，模型对更复杂多模态环境的泛化尚未充分验证；自一致性权重需要手动调优，且在实时控制场景中的延迟改进仍有限。

---

## 80. CyberChainBench: Can AI Agents Secure Smart Contracts Against Real-World On-Chain Vulnerabilities?

**arXiv ID:** 2606.26216 | [PDF](https://arxiv.org/pdf/2606.26216v1)

**作者:** Jintao Huang `[一作]` (Ohio State University), Zhiqiang Lin `[通讯]` (Ohio State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究创建了一个新的链上动态评估基准，涵盖从漏洞检测到利用再到补丁的完整三阶段流程。

**💡 创新点**

创新点在于：①首次在真实主网分叉上进行完整利用与补丁验证；②构建了包含541条真实 DeFi 漏洞实例的统一 benchmark；③提出了根因分类词典与可执行判定器，实现了自动化且可量化的评估。

**🔧 技术方法**

采用的技术包括 Harbor 容器化框架、MCP 工具服务器（Alchmey RPC、EtherVM、Foundry）、以及多种 LLM 代理（Claude、Codex、Gemini 等）与前沿模型（Opus 4.7、GPT‑5.5 等）。

**📊 数据集**

数据集来源于 DeFiHackLabs 的 690 条攻击复现，经过筛选后得到 541 条可验证案例，覆盖 9 条 EVM 兼容链，包含 5 种根因类型。

**📈 对比分析**

评估方法通过 10 种 agent‑model 组合对检测、利用和补丁三任务进行统一评分；结果显示检测准确率最高达 56%，利用利润比例最高 0.44，而补丁成功率仅 23%，凸显补丁生成的难度。

**⚠️ 局限性**

局限性包括：仅覆盖单一交易攻击，补丁子集仅 94 条；可能存在训练数据泄露影响；缺乏多链跨链或多交易攻击场景；以及对现有升级合约结构的依赖。

---

## 81. Self-Supervised Tree-level Biomass Estimation in Urban Environments From Airborne LiDAR and Optical Observations

**arXiv ID:** 2606.26194 | [PDF](https://arxiv.org/pdf/2606.26194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 82. TaskNPoint: How to Teach Your Humanoid to Hit a Backhand in Minutes

**arXiv ID:** 2606.26215 | [PDF](https://arxiv.org/pdf/2606.26215v1)

**作者:** Blake Werner `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出TaskNPoint框架，利用少量人类视频演示训练人形机器人完成动态运动（网球、足球踢球、箱子搬运）。

**💡 创新点**

将技能抽象为离散动作+交互窗口目标，通过随机化目标实现零样本泛化；结合教师-学习者分工和目标条件化的RL策略。

**🔧 技术方法**

SMPL‑X 3D人体重建+MLE融合、运动重定向、异构演员‑评论家强化学习、Proximal Policy Optimization、目标采样与交互窗口奖励。

**📊 数据集**

单视角与多视角人类演示视频（网球、足球、搬箱子），以及在模拟中生成的随机目标。

**📈 对比分析**

与当前最先进的机器人动态运动方法在仿真中对同一3D任务空间进行基准比较，成功率分别为93%和98%，硬件实验在单GPU <1h 训练下实现高命中率。

**⚠️ 局限性**

缺乏训练中的触觉/力反馈，依赖精确的运动捕捉估计，且对极端速度或不符合假设的目标可能失效。

---

## 83. RoboTales: ROBOTic Anthropomorphic LEarning Systems

**arXiv ID:** 2606.26213 | [PDF](https://arxiv.org/pdf/2606.26213v1)

**作者:** Andrew Chen `[一作]` (Case Western Reserve University), Alexis E. Block `[通讯]` (Case Western Reserve University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了RoboTales，一个低成本、无屏幕的机器人讲故事系统，利用袜子手偶通过同步口型、头部动作和手势来表情化叙事。

**💡 创新点**

创新点在于将可3D打印、低成本的袜子手偶与机器人机械臂结合，提供模块化、平台无关的硬件与软件设计，实现完全自主的表情化叙事，并首次将手偶表演与概率运动原语（ProMPs）同步配合。

**🔧 技术方法**

使用了3D打印+Dynamixel伺服驱动的2-DoF手偶末端执行器、语音驱动的口型映射（RMS+EMA滤波）、ProMPs生成手势、同步控制软件以及音频信号处理技术。

**📊 数据集**

实验使用自制的两只手偶（Barneby、Fitzwilliam）配合Baxter与ABB IRB 120机器人进行故事演示；没有公开数据集，故事文本为作者自编短篇。

**📈 对比分析**

与仅使用手势的对照模式对比，采用HRIES情感与社交评估量表和故事回忆问卷；实验结果显示手偶模式在HRIES评分和记忆得分上均优于仅手势模式，且系统实现完全自主。

**⚠️ 局限性**

局限性包括样本量小（10名成人受试者），未在儿童真实场景中验证；手偶动作和口型同步仍有限，缺乏复杂表情和大规模多平台验证。

---

## 84. A Distributed Quantum Approximate Optimization Algorithm Simulator for Engineering Design Optimization

**arXiv ID:** 2606.26297 | [PDF](https://arxiv.org/pdf/2606.26297v1)

**作者:** Ali Rajabi `[一作]`, Amin Kargarian `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套 Qiskit 兼容的 DQAOA 仿真器，支持单 QPU 与多 QPU 分布式 QAOA 解决 QUBO 问题，并提供 Streamlit 图形界面。

**💡 创新点**

创新点在于统一的多模式框架，涵盖变量分配、跨 QPU 通信（TeleGate）实现、参数化电路复用、SPSA‑Adam 迭代、批处理与并行多启动等多重运行时优化。

**🔧 技术方法**

核心技术包括 Qiskit 电路构造、TeleGate 远程门实现、SPSA 估计梯度的 Adam 优化器、参数化电路缓存与批量执行、并行多启动搜索，以及 Streamlit GUI。

**📊 数据集**

使用的基准数据集为 6、12、15、20 变量的 QUBO 实例，以及一个 15 变量的功率单元承诺（UC）工程案例。

**📈 对比分析**

通过与经典基准、单 QPU 与多 QPU QAOA 的精度对比，并在不同深度与多启动场景下测量运行时间，实验表明分布式模式在保持最佳位串一致的前提下，运行时间显著高于单 QPU，优化后仍比初始实现快数倍。

**⚠️ 局限性**

局限在于跨 QPU 交互导致的通信与门实现开销仍使分布式模式慢于单 QPU；当前仅在模拟器上验证，缺乏大规模量子硬件实验和对噪声鲁棒性的深入评估。

---

## 85. Beyond Takedown: Measuring Malicious Go Module Persistence in the Wild

**arXiv ID:** 2606.26291 | [PDF](https://arxiv.org/pdf/2606.26291v1)

**作者:** Minjae Bae `[一作]` (Ohio State University), Carter Yagemann `[通讯]` (Ohio State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

通过手工搜索GitHub并结合Go公共代理索引的两阶段测量，识别并量化了在Go生态中恶意模块的传播与持久性。

**💡 创新点**

首次将托管层与分布层视角融合，揭示了“takedown≠remediation”差距，并展示了代理索引能捕获GitHub失效后仍存在的恶意模块。

**🔧 技术方法**

使用自研GOAST AST扫描器进行静态代码解析，配合社交图搜索与Go代理索引日志抓取。

**📊 数据集**

构建了包含12.3M条代理索引记录的数据库，并手工标注了2,113个恶意GitHub仓库，最终确认2,289个恶意模块版本。

**📈 对比分析**

通过比较手工集M与代理集G的交集，发现GitHub视角漏检约83%，并用GOAST实现99.65%精度的检测；实验表明托管层下架后，约99.4%的模块仍可通过代理获取。

**⚠️ 局限性**

研究仅覆盖公共代理索引窗口，未能覆盖私有代理或直接VCS拉取；检测仅基于静态分析，未评估动态行为；缺乏对下游依赖链的完整影响评估。

---

## 86. From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations

**arXiv ID:** 2606.26277 | [PDF](https://arxiv.org/pdf/2606.26277v1)

**作者:** Dianjing Fan `[一作]` (Capital One), Giri Iyengar `[通讯]` (Capital One)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个双输出框架，将预登录网页点击流映射为稠密会话嵌入和可解释的意图标签，以实现金融服务推荐和个性化。

**💡 创新点**

创新点包括：自监督Transformer编码多模态点击流得到会话嵌入；基于LLM的分层聚类+迭代生成意图分类学，并通过蒸馏到轻量化分类器，实现同一表示同时支持定量预测和可解释标签。

**🔧 技术方法**

采用了Transformer自监督学习、多模态特征融合、聚类（K‑means）采样、LLM（如GPT）生成分类学、知识蒸馏至MLP等技术。

**📊 数据集**

使用资本一公司金融服务网站的10M每日点击流，筛选约10万包含预登录与登录行为的会话进行训练和验证。

**📈 对比分析**

在手机首页tile排名任务中，加入会话嵌入可提升Recall@1 1.88%、LogLoss 13.38%；在用户转化预测任务中，蒸馏模型微F1 93%近似LLM标签，且会话嵌入进一步提升4.3%。

**⚠️ 局限性**

局限性在于只在单一金融机构内评估，泛化性未知；需人工审核分类学；蒸馏后可解释标签信息损失；暂未实现在线A/B评估和多用户级别聚合。

---

## 87. Accelerating Skill Assessment in Chess: A Drift-Diffusion-Enhanced Elo Rating System

**arXiv ID:** 2606.26267 | [PDF](https://arxiv.org/pdf/2606.26267v1)

**作者:** Tianyuan Zhou `[一作]` (Nanjing University), Tianming Yang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出DD-Elo，一种将漂移扩散模型与Elo评级结合的棋类评分系统。

**💡 创新点**

创新在于利用逐步棋盘移动的发动机评估作为漂移信号，使评级更新更快更灵敏。

**🔧 技术方法**

使用漂移扩散模型（DDM）、期望分数函数、记忆衰减机制以及增量Elo更新。

**📊 数据集**

基于Lichess 2019年约1000万局数据，过滤后429k玩家的棋局和Stockfish评估的CPL。

**📈 对比分析**

与传统Elo及Glicko等进行对比，通过AIP、DA、ALT、IC等指标，DD-Elo在非平稳期更快适应，平均提前0.2局并显著提升方向性与信息系数。

**⚠️ 局限性**

受限于高噪声移动评估的误差、仅验证棋类可能不易迁移到其他领域，以及需进一步评估对极端高水平玩家的稳定性。

---

## 88. Beyond Aesthetics: Quantifying Information Loss in Turbid Scenes

**arXiv ID:** 2606.26295 | [PDF](https://arxiv.org/pdf/2606.26295v1)

**作者:** Vasiliki Ismiroglou `[一作]` (Aalborg University), Malte Pedersen `[通讯]` (Aalborg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并评估了基于相位一致性的PCD指标，用于量化浑浊水域图像的结构信息损失，并引入了首个高浑浊度实例分割数据集TUB。

**💡 创新点**

创新点在于①将相位一致性与delentropy结合，得到对对比度和色彩失真不敏感、但能反映结构损失的PCD；②构建了包含1,320张高浑浊度图像和1.6万实例掩码的TUB数据集；③证明PCD与真实与合成浑浊图像上实例分割模型性能的相关性显著高于传统图像质量/信息度量。

**🔧 技术方法**

使用相位一致性计算、Delentropy、UCIQE、UIQM、NIQE、SSIM、PSNR等指标，训练Mask R-CNN、YOLOv11和Mask2Former进行实例分割实验。

**📊 数据集**

使用TUB数据集（真实高浑浊）以及通过两种物理模型生成的合成浑浊图像（Synth_1、Synth_2）。

**📈 对比分析**

通过AP50评估模型在不同浑浊等级（低/中/高）下的性能，并计算各指标与AP50的相关系数。结果显示PCD与模型性能的相关性最高，其它指标相关性弱或为负。

**⚠️ 局限性**

局限包括：PCD对不同场景结构差异敏感；TUB仅在受控实验室环境下收集，缺乏真实海洋多样性；合成浑浊模型仍无法完全重现高散射环境的细粒度结构退化。

---

## 89. Dataset Usage Inference without Shadow Models or Held-out Data

**arXiv ID:** 2606.26257 | [PDF](https://arxiv.org/pdf/2606.26257v1)

**作者:** Wojciech Łapacz `[一作]` (Warsaw University of Technology), Adam Dziedzic `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无影子模型、无可信非成员集的生成式模型使用量推断框架（NU-DUI），通过合成非成员样本、对嫌疑集进行自编码、提取多维成员推断特征，并将问题转化为混合比例估计来估计训练集使用比例。

**💡 创新点**

创新点在于：1) 消除了传统方法对昂贵影子模型和可信测试集的依赖；2) 采用跨模型族的图像到图像生成器产生合成非成员样本，避免同族生成导致的伪成员化；3) 将自编码引入嫌疑集，降低分布偏移；4) 将使用量推断视为混合比例估计问题，利用现成的 MPE 方法实现。

**🔧 技术方法**

技术包括：图像到图像生成（Stable Diffusion 或 VAR 变体）用于合成非成员；自编码器重构嫌疑集；多维成员推断攻击（针对自回归模型或扩散模型的专用 MIA 信号）；混合比例估计算法（TIcE、AlphaMax、PUL LBE/NTC‑τMI）。

**📊 数据集**

使用 ImageNet‑1k 公开数据集，评估的目标模型有 VAR‑24（1.0B）、VAR‑30（2.1B）、RAR‑XL（955M）、RAR‑XXL（1.5B）以及 DiT‑RF‑XL/2‑8E2A（4.1B）。

**📈 对比分析**

与理想真实非成员基准（Real）和仅使用合成非成员（Synth）的对照相比，NU‑DUI 在多数模型上均实现 MAE 0.05‑0.12，接近 oracle 性能；在 1,000 张图像的嫌疑集上，混合比例估计误差小于 0.1；计算成本仅 42.5 分钟，相比 DUCI 需要 1500 小时的影子模型训练快 2000 倍。

**⚠️ 局限性**

局限性在于：1) 依赖可获取的成员推断信号，若目标模型泄露较少成员信息，估计会不稳定；2) 需要可用的跨族生成器来产生合成非成员；3) 对于无此类生成器的领域（如文本生成等），方法不直接适用。

---

## 90. A multi-task spatiotemporal deep neural network for predicting penetration depth and morphology in laser welding

**arXiv ID:** 2606.26260 | [PDF](https://arxiv.org/pdf/2606.26260v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 91. A General Framework for Learning Algebraic Properties from Cayley Graphs using Graph Neural Networks

**arXiv ID:** 2606.26212 | [PDF](https://arxiv.org/pdf/2606.26212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. Equivariance and Augmentation for Bayesian Neural Networks

**arXiv ID:** 2606.26273 | [PDF](https://arxiv.org/pdf/2606.26273v1)

**作者:** Miaowen Dong `[一作]` (Chalmers University of Technology and the University of Gothenburg), Jan E. Gerken `[通讯]` (Chalmers University of Technology and the University of Gothenburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对贝叶斯神经网络在数据增强下的变分推理进行理论分析，并提出三种对称化技巧；

**💡 创新点**

证明在指数族闭合且先验对称时，梯度训练保持对称子空间不变，进而推导出对称化策略；并验证orbit expansion在精度和对称性上最优；

**🔧 技术方法**

使用变分推理、指数族分布、群表示与对称化投影、Monte Carlo 预测、数据增强等技术；

**📊 数据集**

主要使用FashionMNIST数据集，并以C4旋转群进行完整的对称数据增强；

**📈 对比分析**

与随机初始化基线、不同触发时刻及不同优化器进行对比；orbit expansion在分类准确率、等价性（OSP）和对称KL上均优于其他方法；

**⚠️ 局限性**

仅适用于指数族、有限群且需要对称先验，无法涵盖混合高斯、连续群或非线性对称等情况。

---

## 93. GeMoE: Gating Entropy is All You Need for Uncertainty-aware Adaptive Routing in MoE-based Large Vision-Language Models

**arXiv ID:** 2606.26287 | [PDF](https://arxiv.org/pdf/2606.26287v1)

**作者:** Chaoxiang Cai `[一作]` (Zhejiang University), Xi Li `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于门控熵的动态专家路由方法 GeMoE，用于在 Mixture-of-Experts 结构中实现高效、低成本的推理。

**💡 创新点**

创新点在于将动态路由视为 Minimum Description Length（MDL）优化问题，并证明门控熵可作为信息增益的代理，进而通过单调损失学习门控熵与所需专家数量的正相关关系。

**🔧 技术方法**

采用 Mixture-of-Experts 架构、门控熵度量、MDL 理论框架、单调性约束与轻量级专家分配预测器等技术。

**📊 数据集**

使用 LLaVA-1.5-558k 进行一轮微调，并在 MMBench、POPE、ScienceQA、TextVQA、GQA、MM‑Vet 六大视觉‑语言基准上进行评估。

**📈 对比分析**

与 DYNMoE、Top‑p、AdaMoE、MoE++ 等动态路由以及不同 Top‑k 静态路由进行对比，实验表明 GeMoE 在 MolmoE‑1B‑7B 与 DeepSeek‑VL2‑Tiny‑1B‑3B 上平均激活专家数仅 5.43，性能仅比 Top‑8 低 0.38%，同时显著提升稀疏度与推理吞吐率。

**⚠️ 局限性**

局限性包括对门控熵估计的依赖性、未充分验证在更大规模模型或其他任务上的通用性，以及训练过程中仅微调路由，未针对专家权重进行进一步优化。

---

## 94. NavIsaacLab: Generating Realistic Crowd via Parallel Robot Learning for Benchmarking Human-aware Navigation

**arXiv ID:** 2606.26265 | [PDF](https://arxiv.org/pdf/2606.26265v1)

**作者:** Bingyi Xia `[一作]` (Southern University of Science and Technology), Jiankun Wang `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 NavIsaacLab——一个基于 Isaac Lab 的高保真、并行、可扩展的人类感知导航仿真与基准平台，支持光学渲染、物理驱动的完整身体步态、数据驱动的轨迹生成，并提供多场景评估工具；

**💡 创新点**

创新点在于将扩散模型与对抗运动先验（AMP）结合，实现可控且多模态的全身行人行为；实现 GPU 并行跑通多达 64 只行人与机器人；并首次在此平台上构建统一的多维度人类感知导航基准；

**🔧 技术方法**

核心技术包括 NVIDIA Isaac Lab、基于 U‑Net 的轨迹扩散模型、AMP 对抗运动学习、Transformer 关注式感知编码、PPO 强化学习以及自监督姿态估计 Monoloco++；

**📊 数据集**

使用 AMASS 动作捕捉数据训练 AMP，利用 OmniGibson 3D 场景库构建场景，此外还引用 ETH/UCY 行人轨迹数据用于评估；

**📈 对比分析**

与传统 SFM+AMP、轨迹重放等方法对比，Diffusion+AMP 在成功率、碰撞率、路径误差和 MMD 上均提升约 15–30%；在 30 个 photorealistic 场景下，基线策略在 RL 训练中取得较高的最终奖励，表明平台能有效提升策略性能；

**⚠️ 局限性**

局限性包括：部分复杂步态转换仍表现欠佳；仿真与现实的差距仍需进一步缩小；基准场景数量有限；以及在更高并行度下仍出现子线性扩展瓶颈。

---

## 95. Enterprise Data Asset Quality: A Management-Standard Conformity-Benefit Realization Framework and Formation Mechanisms

**arXiv ID:** 2606.26186 | [PDF](https://arxiv.org/pdf/2606.26186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 96. Knowledge-augmented Agentic AI for Mental Health Medication Information Seeking

**arXiv ID:** 2606.26205 | [PDF](https://arxiv.org/pdf/2606.26205v1)

**作者:** Huizi Yu `[一作]` (Chinese University of Hong Kong), Lizhou Fan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于知识图的多源药物安全信息框架，集成了 Reddit、WebMD 以及 FDA/FAERS 的抗抑郁药不良反应数据，并实现了多代理检索问答系统。

**💡 创新点**

创新点包括：① 源感知（provenance-aware）集成，保证监管与患者叙事信息互不混淆；② 将 LLM 与知识图结合的多代理架构，减少生成错误；③ 系统化展示社区数据与监管数据的时序领先性与相似度，为药物安全监测提供新视角。

**🔧 技术方法**

主要技术：大语言模型（如 GPT‑4.1‑mini）用于实体识别；embedding‑based 匹配实现实体标准化；Neo4j 知识图存储实体与关系；多代理架构（NER、意图、查询、摘要、比较、验证）与 Cypher 查询；检索增强生成。

**📊 数据集**

数据集：Reddit 466,525 篇信息量大帖子；WebMD 60,782 条药物评论；FDA/FAERS 2005‑2025 年约百万条不良事件报告；通过 ATC‑N 构建 9 种抗抑郁药的药物字典。

**📈 对比分析**

比较方法：使用 Jaccard 相似度评估不良事件集的重叠；源平衡度（熵与均匀度）衡量不同来源的分布；相关性与火山图评估频率差异；先行时间（lead‑time）分析社区与监管数据首次出现的时间差。性能方面：NER 在药物名称、疾病和不良事件上分别达 F1 最高 0.969、0.973、0.912；WebMD‑Reddit 相似度最高 0.905；社区数据往往在 FDA 报告之前数百天出现。

**⚠️ 局限性**

局限性：① 属性级别提取（剂量、持续时间、频率）精度低；② 仅覆盖英文文本，可能遗漏非英语社区体验；③ 通过 embedding 匹配的实体标准化仍有误差；④ 只分析了 9 种抗抑郁药，外推性受限；⑤ 未对系统的临床实用性、用户体验或遵从性提升进行前瞻性评估。

---

## 97. Topology-Informed Neural Networks for Flood Detection in Optical and Synthetic Aperture Radar Imagery

**arXiv ID:** 2606.26204 | [PDF](https://arxiv.org/pdf/2606.26204v1)

**作者:** Sophia Li `[一作]` (US Naval Research Laboratory), Tianyu Chen `[通讯]` (US Naval Research Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文在公开的SEN12-FLOOD数据集上，利用持久同调（persistent homology）提取的拓扑特征与ResNet-50卷积特征相结合，通过GRU网络实现洪水时序检测；

**💡 创新点**

首次将持久同调直接应用于SAR与光学图像，提出轻量化高斯嵌入方法，并证明拓扑与卷积特征互补，Fusion‑GRU模型在双模任务中将准确率提升至98.9%；

**🔧 技术方法**

使用的技术包括预训练的ResNet‑50（BigEarthNet）、GRU序列网络、Gaussian Embedding对持久图进行向量化、以及后期的特征融合（late fusion）和潜在的注意力机制；

**📊 数据集**

采用了SEN12‑FLOOD公开数据集，该数据集包含Sentinel‑1 SAR（VV/VH）和Sentinel‑2多光谱光学影像，覆盖非洲、伊朗和澳大利亚的2018/2019年洪水地区；

**📈 对比分析**

通过与单模单特征（ResNet‑50‑GRU、Topo‑GRU）和双模融合模型的比较，使用Fβ（β=√2）、F1、Accuracy、Recall@90%Precision等指标评估；Fusion‑GRU在双模任务中取得0.980的Fβ、0.989的Accuracy，明显优于其他模型；

**⚠️ 局限性**

局限性包括：高斯嵌入网格固定未能学习最优中心；模型仅在SEN12‑FLOOD上验证，泛化性和对云遮蔽等不良条件的鲁棒性尚未充分评估。

---

## 98. Soroll-IA: A Weakly Labeled Audio Dataset for Real-World Industrial Port Monitoring

**arXiv ID:** 2606.26195 | [PDF](https://arxiv.org/pdf/2606.26195v1)

**作者:** Javier Naranjo-Alcazar `[一作]` (Instituto Tecnologico de Informatica), Pedro Zuccarello `[通讯]` (Instituto Tecnologico de Informatica)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文构建了一个名为 Soroll-IA 的弱标注工业港口环境音频数据集，并给出了两种标签版本。

**💡 创新点**

创新点在于首次公开专注于真实港口场景的弱标签多标签数据集，以及采用专家驱动的主动学习标注流程。

**🔧 技术方法**

技术手段包括固定式传感节点收集音频、专家多轮标注、基于 CNN14（PANNs）和 MobileNetV2 的模型训练与迁移学习。

**📊 数据集**

使用的数据集是 Soroll-IA，包含约22小时、7396段、26类工业港口声音的弱标签记录。

**📈 对比分析**

通过 5 折交叉验证对 CNN14 训练、微调以及 MobileNetV2 的基线进行比较，取得的 mAP 在 Non‑CV 条件下约 0.67（CNN14）和 0.65（MobileNetV2），CV 条件下约 0.63 和 0.62。

**⚠️ 局限性**

局限性包括数据仅来自西班牙瓦伦西亚港口一年的记录、类不平衡、弱标签噪声以及高多音场导致的模型泛化挑战。

---

## 99. Necessary but Not Sufficient: Temperature Control and Reproducibility in LLM-as-Judge Safety Evaluations

**arXiv ID:** 2606.26185 | [PDF](https://arxiv.org/pdf/2606.26185v1)

**作者:** Hiroki Tamba `[一作]` `[通讯]` (Tamba Research Academy), Hiroki Tamba (Tamba Research Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在安全评估中，研究了LLM-as-judge的非确定性问题，揭示了默认温度设为1导致的评判随机性，并通过实验验证和复现工具展示了温度固定并非足够的事实。

**💡 创新点**

创新点在于系统性地发现并量化评估框架对温度默认值的忽视、证明温度固定并不能完全消除不确定性（且已被主流模型废弃），并提出将评估者一致性视为指标、记录有效配置的做法。

**🔧 技术方法**

使用多模型（OpenAI gpt‑4o、Claude Sonnet、Haiku等）、多供应商（OpenAI、Claude、DashScope等）API调用，调节温度、top_p、top_k、greedy decoding等采样参数，统计不确定性并记录实验细节。

**📊 数据集**

实验数据基于7个精心挑选的边界问题/答案对，构成对抗性压力测试；未使用大规模公开评估语料。

**📈 对比分析**

通过对比默认配置与显式温度0、不同采样设置的实验，发现4/7边界项在默认配置下不确定，pinning后仍有1–2项无法复现，表明温度固定不足以保证可靠度。

**⚠️ 局限性**

局限性包括样本仅为7个对抗性例子、样本量小且无法估计总体不确定率、只在少数模型/供应商上测试、未检验长期运行或更大规模评估，以及实验窗口有限。

---

## 100. Query Cost Model Calibration in Confidential Virtual Machines

**arXiv ID:** 2606.26385 | [PDF](https://arxiv.org/pdf/2606.26385v1)

**作者:** Qihan Zhang `[一作]` (University of Southern California), Ibrahim Sabek `[通讯]` (University of Southern California)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 PostgreSQL 中实现了一种轻量级的 CVM（Confidential Virtual Machine）感知成本校准机制，利用数据移动和 RMP 检查的两种主要开销来改进查询优化器的成本模型。

**💡 创新点**

创新点在于首次把 CVM 相关硬件开销（尤其是加密数据移动和 RMP 验证）通过简单的物理代理（如工作集大小、页访问次数）直接嵌入成本公式，从而在保持原有优化器架构的前提下显著提高 CVM 下的查询计划质量。

**🔧 技术方法**

采用的技术包括：基于工作集大小和溢出因子的 datamovecost 与 rmptcost 校准函数，针对不同算子（顺序扫描、索引扫描、哈希/嵌套循环/归并连接等）的成本加权；在 PostgreSQL 16.9 上实现并使用自定义的 C 扩展进行调优；使用低级计数器（CPI、TLB 缺失等）验证开销来源。

**📊 数据集**

使用的基准数据集包括 TPC‑DS（SF10）、Stack、IMDB‑based JOB 以及 CEB，均采用标准查询模板并多实例化。

**📈 对比分析**

通过在 KVM（无加密）与 SEV‑SNP（加密）两种安全级别下执行，比较未校准与校准后的总执行时间。实验表明，校准可在 SEV‑SNP 中将工作负载执行时间降低 3.5%~48%，并在某些工作负载（如 Stack、CEB）上甚至超过 KVM 基线。

**⚠️ 局限性**

局限性包括：校准参数是基于少量代表性查询手动调优，未实现自适应或工作负载自学习；仅考虑了 SEV‑SNP，可能不适用于其他 TEE；对极端大工作负载或高并发场景的效果尚未验证。

---

## 101. Phonetic and semantic analyses of spoken corpora of Beijing and Taiwan Mandarin indicate that the neutral tone is a lexical tone

**arXiv ID:** 2606.26360 | [PDF](https://arxiv.org/pdf/2606.26360v1)

**作者:** Yuxin Lu `[一作]` (Eberhard Karls Universität Tübingen), R. Harald Baayen `[通讯]` (Eberhard Karls Universität Tübingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文分析北京和台湾普通话两种方言中两音节词的中性/浮动音调的音高实现，利用广义加性模型分解音高曲线，并研究词义对音高的可预测性，进一步比较两方言在音调模式、词形音高特征及语义差异上的差异。

**💡 创新点**

创新点在于：①把中性音调视为具备独立音高目标的词义/音调类别，证明其与四声在功能上相似；②首次将上下文嵌入与音高映射相结合，展示语义信息能显著预测中性音调的音高曲线；③系统比较北京与台湾方言中性音调的实现差异，并关联词义差异，揭示方言差异的语义基础。

**🔧 技术方法**

主要技术包括：广义加性混合模型（GAM）对音高曲线进行分解并评估各因素的重要性；使用Qwen‑2.5生成的上下文嵌入（CE）作为语义特征；构建线性映射模型将嵌入映射到音高向量，并用最近邻分类评估预测准确率；同时计算共变性、AIC、SSD等统计量进行比较。

**📊 数据集**

使用两套自然对话语料：北京普通话语料（50位说话人，约40小时）和台湾普通话语料（55位说话人，约30小时），从中提取两音节词的中性音调实例，最终得到北京约4871个词条、台湾约3831个词条，涵盖四个音调模式（T1‑T5、T2‑T5、T3‑T5、T4‑T5）和“Others”。

**📈 对比分析**

比较方法：①通过AIC差异评估各预测变量对音高的贡献；②对不同音调模式和词形的预测曲线进行差异曲线检验；③使用线性映射+最近邻分类将嵌入向量映射到音高空间，测试集准确率分别为23.97%（北京）和20.34%（台湾），均显著高于10%基线，表明语义嵌入能在一定程度上预测中性音调的音高；④计算词型间SSD分布，发现两方言在相同词型的音高相似度更高。性能方面，虽然预测准确率低于理想，但相较基线明显提升，验证了语义与音高的关联。

**⚠️ 局限性**

局限性包括：①只分析两音节词，未探讨三音节及更长结构；②语料主要为对话，缺乏受控实验数据，可能限制了对细粒度音高变化的捕捉；③嵌入模型未针对本研究语料进行微调，预测准确率仍有限；④GAM模型存在共变性问题，部分预测因子解释力不够独立；⑤未深入研究持续时长、强度等其他声学参数与语义的关联。

---

## 102. Instruction Bleed: Cross-Module Interference in Prompt-Composed Agentic Systems

**arXiv ID:** 2606.26356 | [PDF](https://arxiv.org/pdf/2606.26356v1)

**作者:** Ching-Yu Lin `[一作]` (University of Illinois Urbana-Champaign), Yifan Liu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Prompt‑composed Agentic Systems 中的跨模块行为泄露（CBL）现象，提出了其操作性定义、三通道可复用检测协议，并在一个部署的求职评估 Agent 上进行了实验验证。

**💡 创新点**

创新点包括：①正式定义并区分 CBL 与其他相似故障模式；②构建可重用的三通道（volume、content、form）检测协议；③给出可验证的预测集合，为后续多模型、多系统的复制研究提供框架；④对 Prompt‑composed Agentic Systems 进行系统级特征化，确立其为独立的系统类别。

**🔧 技术方法**

技术手段包括：Transformer 自注意力理论分析、基于 Claude Sonnet 4.6 的实际部署实验、统计效应量（Cohen's d）与 Bootstrap 95% 置信区间、Wilcoxon 符号秩检验、对比基线与三种干预条件的分数差异。

**📊 数据集**

使用的数据集为 12 条职位描述（Job Descriptions），共 144 次试验（每条 JD 3 次），用于评估 CV‑match 评分分布。

**📈 对比分析**

与基线（C0）相比，C2（语义干扰）导致 CV‑match 均值上升 0.17，Cohen's d = 0.63（95% CI [+0.03, +0.31]），显著正向影响；C1（体量干扰）和 C3（格式干扰）均未产生显著差异，CI 包含零。

**⚠️ 局限性**

局限性：仅在单一模型（Claude Sonnet 4.6）和单一 Agent（职业评估）上验证，未检验多模型、多系统的普适性；实验规模受 12 条 JD 限制，未覆盖大规模真实部署的模块组合；CBL 的检测对小幅语义干预有效，但对更大或更复杂的干预场景的敏感度仍未知。

---

## 103. What We are Missing in Multimodal LLM Evaluation?

**arXiv ID:** 2606.26348 | [PDF](https://arxiv.org/pdf/2606.26348v1)

**作者:** Po-han Li `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多模态大语言模型（MLLM）的评估现状进行了综述，指出了现有基准在评估跨模态信息整合、时空连贯性、物理世界建模和选择性注意力方面的不足，并提出了对评估范式的改进思路。

**💡 创新点**

创新点在于：① 系统性梳理并归纳评估缺口；② 提出新的评估维度（时空连贯、物理世界建模、多模态一致性、注意力选择性）；③ 建议多指标、多场景的评估框架，强调评估应关注功能性和现实应用中的可用性。

**🔧 技术方法**

主要采用文献综述、基准对比与理论分析的方法；未引入新的算法或模型实现，而是对现有评测技术进行反思与分类。

**📊 数据集**

参考了多种现有数据集和基准，包括 Image‑QA、视频时序推理、OCR 文档、机器人导航与人机交互等（VLABench、Egocentric benchmarks 等），但未提出新的数据集。

**📈 对比分析**

作者并未在实验中给出具体性能指标，而是通过对比分析指出当前排行榜分数与真实应用性能之间的偏差；提出若干改进方案（多指标评估、保留部分数据集、周期性更新）。

**⚠️ 局限性**

局限性：缺乏实证评估与新数据集的构建；提出的改进思路尚未在具体任务或模型上验证，难以量化其效果；对评估指标的选择仍停留在概念层面，需要进一步细化和实现。

---

## 104. Layered Outer-Loop Control for Disturbance-Robust Multi-Waypoint UAV Arrival

**arXiv ID:** 2606.26315 | [PDF](https://arxiv.org/pdf/2606.26315v1)

**作者:** Runfeng Ling `[一作]` `[通讯]` (University of Manchester), Runfeng Ling (University of Manchester)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种分层终端控制架构，用于多航点UAV位置调节，并通过PyBullet、PX4/Gazebo和Vicon跟踪的Tello硬件进行分阶段跨环境评估。

**💡 创新点**

创新点在于将平滑接近生成、持续偏差补偿和监督式终端调节分层、结构化；同时引入严格与宽松终端语义的双重评估，验证控制结构的可迁移性和稳健性。

**🔧 技术方法**

采用多层控制设计（参考生成、模式依赖速度控制、扰动观测器、监督终端模式）、PyBullet仿真、PX4/Gazebo闭环仿真、ROS2离线控制节点，以及Vicon跟踪的Tello硬件实验。

**📊 数据集**

使用随机多航点任务与随机风扰动的PyBullet数据集；PX4/Gazebo中使用预设的gust-dominated与mixed-disturbance场景；硬件实验使用固定的六航点任务。

**📈 对比分析**

在Phase I通过后期位置均值/方差、到达时间等指标对紧凑、激进、分层三类基线进行比较，分层控制在风扰动下后期误差仅0.024 m；在Phase II，分层控制在严格评估下成功率0.75、误差<0.15 m；硬件Stage A显示分层控制终端误差与回弹抑制优于基线；Stage B完成30/30段，平均终端误差0.46 m。

**⚠️ 局限性**

缺乏正式闭环稳定性证明，参数化复杂且需要平台调参；评估范围局限于单一平台、室内环境和短期任务；对不同车辆类型、室外风场的泛化性仍未知。

---

## 105. Geometry-Aware MCTS for Extremal Problems in Combinatorial Geometry

**arXiv ID:** 2606.26399 | [PDF](https://arxiv.org/pdf/2606.26399v1)

**作者:** Luoning Zhang `[一作]`, Nathan Kaplan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在离散几何中提出了一种几何感知蒙特卡罗树搜索（Geometry-Aware MCTS）框架，用于寻找满足严格全局几何约束的最大/最小点集。

**💡 创新点**

创新点在于：①增量维护可行动作空间将约束检查从 O(n³) 降至 O(n²)；②基于状态稳定子群的规范剪枝；③对称批量转移加速搜索深度；④子树重用与时间衰减探索；⑤随时最佳状态跟踪，保证全局最优被记录。

**🔧 技术方法**

技术手段包括：蒙特卡罗树搜索（UCT）、确定性 MDP、射线投射（Ray‑Casting）实现增量更新、对称群 D₄ 的状态稳定子群分析、回溯模拟、子树重用、可解释的探索衰减函数。

**📊 数据集**

实验数据集为 n×n 网格的点集（n 取值至 119），在 Max‑N3IL、Min‑Complete、Min‑Dom、Max‑N4IL、Max‑No‑Isosceles、Max‑No‑4‑on‑Circle 等六种极值问题上构造并记录了最佳解；未使用公开大规模图像或文本数据集。

**📈 对比分析**

与以往最优理论与计算结果比较，发现：在 5/6 个问题中取得新最优下界或上界；Max‑N3IL 解决方案平均规模约 1.8n（超过 1.5n 的代数下界、低于 2n 上界），Min‑Complete、Min‑Dom 规模约 0.95n；相较基线 MCTS，显著提升终点数和运行时效；通过多轮实验和统计检验验证结果显著优于现有方法。

**⚠️ 局限性**

局限性包括：仅给出经验构造，缺乏闭式证明；目前实现仅单核 CPU、6GB 内存，受限于搜索深度；对非常大 n 的可扩展性仍待验证；对某些约束对称性选择不一定最优；未来需并行化、GPU 加速和神经网络价值/策略引导。

---

## 106. Neural Voxel Dynamics: Learning Implicit 3D Physics via Volumetric Feature Advection

**arXiv ID:** 2606.26410 | [PDF](https://arxiv.org/pdf/2606.26410v1)

**作者:** Zican Wang `[一作]` (University College London), Niloy Mitra `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了神经体素动力学（Neural Voxel Dynamics）框架，通过把单目视频中的语义特征上升到稀疏3D体素空间，并学习基于动作的隐式特征推进器，实现在仅靠被动视频监督的情况下进行三维物理预测与生成。

**💡 创新点**

创新点在于：①将V-JEPA的二维语义特征“提升”到三维体素网格，解决了二维潜在空间缺乏几何不变量的问题；②通过流匹配的隐式特征推进器学习动作条件下的3D状态转移，避免了显式物理求解器；③使用稀疏标记通道（占据、未观测）提高几何一致性；④在单目视频上直接训练，无需真实物理标签。

**🔧 技术方法**

使用技术包括：V-JEPA（Joint-Embedding Predictive Architecture）提取语义特征；MoGe深度与相机估计；体素上三维投影与稀疏重建；Diffusion Transformer实现局部空间注意力与时间注意力；流匹配损失、占据Focal loss、投影NeRF损失等多任务训练；自回归滚动预测。

**📊 数据集**

在CLEVRER、PhysInOne、PhysGaia等合成/真实视频数据集上进行评估，并与CogVideoX、PhysGen、PhysGaussian、PhysCtrl等基线对比。

**📈 对比分析**

与现有基线相比，Neural Voxel Dynamics在3D潜在空间的L2误差显著下降（如在CLEVRER上从1.39降至0.98），流估计误差也最低；在不同材料类别（刚体、流体、烟雾）上保持稳定性能，尤其在多视角下的3D一致性优于2D方法。

**⚠️ 局限性**

主要局限：①几何精度受V-JEPA特征分辨率与单目深度估计误差限制；②稀疏体素尺寸和视角覆盖的权衡导致记忆/计算开销；③长时间滚动预测易产生误差累积；④缺乏对未观测体素的生成能力；⑤动作接口仅为参数化力向量，尚未支持复杂多点控制。

---

## 107. Lessons from the Adoption and Deprecation of the Privacy Sandbox Web APIs

**arXiv ID:** 2606.26390 | [PDF](https://arxiv.org/pdf/2606.26390v1)

**作者:** Yohan Beugin `[一作]` (University of Wisconsin-Madison), Patrick McDaniel `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对Google Privacy Sandbox API 进行纵向测量与分析，评估其在 Chrome 浏览器及顶级网站中的采用与弃用趋势。

**💡 创新点**

创新点在于首次整合 Chrome 远程遥测、HTTP Archive 爬取、源代码审计等多源数据，全面覆盖 13 项 API 的 5 年+ 生命周期，并公开可复现的测量框架。

**🔧 技术方法**

采用 Blink 特征检测、BigQuery SQL 查询、聚合日志、UpSet 交叉集可视化、时间序列分析等技术手段实现数据抽取与分析。

**📊 数据集**

主要使用数据集包括：Chrome Telemetry（用户页面加载时 API 使用情况）、HTTP Archive（顶级 100k 网站的抓取与请求日志）、Chrome 源码中的 Attestation 列表、Google Chrome 相关网站集合 GitHub repo。

**📈 对比分析**

通过时间序列对比 API 使用峰值、弃用事件与用户浏览行为，计算页面加载中 API 出现比例；结果表明多数 API 采用率低于 10%，只有 Partitioned Cookies 达到 34%（但仅占 6.7% TPC），总体停滞不增，未涉及复杂度或性能评测。

**⚠️ 局限性**

局限性包括：爬取未交互、无同意提示，无法区分用户同意前后的调用；未评估 Android 端 API；仅聚焦 Chrome 生态，跨浏览器差异仅通过文献推断；检测方法存在误报风险。

---

## 108. Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs

**arXiv ID:** 2606.26387 | [PDF](https://arxiv.org/pdf/2606.26387v1)

**作者:** Xi Xiao `[一作]` (University of Alabama at Birmingham), Hao Xu `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后训练框架 Visual Information Gain Alignment (VIG)，通过对视觉输入构造对照盲状态（attention‑masking），最大化视觉信息增益，显式惩罚模型在缺失视觉信息时的高置信度，从而抑制多模态大语言模型的视觉迟疑与幻觉。

**💡 创新点**

创新点在于将视觉信息增益作为几何约束引入偏好优化，使用对照盲状态来显式对比“见视”与“盲视”，并通过动态门控、自对抗硬负样本与最大互信息原则实现高效、数据稀缺下的对齐；同时在保持语义流畅性的同时显著提升视觉真实性。

**🔧 技术方法**

核心技术包括：Direct Preference Optimization (DPO) 与 Counterfactual Visual Decoupling (CVD) 损失，视觉输入的 attention‑masking 生成盲状态，动态门控 α 平衡语义与视觉约束，基于最大互信息的 VIG 定义，硬负样本挖掘；训练采用 OpenRLHF 框架、DeepSpeed ZeRO‑3、FlashAttention‑2 等高效加速。

**📊 数据集**

使用多模态偏好数据集 𝒟（图像+文本+胜负回复），以及 POPE、AMBER、MMHal‑Bench、MathVista、MMBench、MMLU、GSM8K 等评测基准；零样本 RefCOCOg 用于空间定位评估。

**📈 对比分析**

与 SFT、标准 DPO、SimPO、HA‑DPO、DA‑DPO、VCD 等基线对比。结果显示在 Qwen2.5‑VL‑7B 上提升 2.3–4.1 百分点，72B 模型提升 3.7–5.5 百分点；在 LLaVA‑OneVision、InternVL2.5 等不同架构上均优于 DPO；在仅 25% 数据量下即可匹敌全量数据的性能，并对 KL 阻尼 β 更为鲁棒。

**⚠️ 局限性**

主要局限在于盲状态仅通过注意力掩蔽构造，缺乏更细粒度或对象级别的对照操作，难以完全捕捉视觉混乱或跨模态冲突的细节；因此在极其复杂的视觉环境下仍可能存在视觉细节被忽视的情况。

---

## 109. A Typestate Approach to Purpose-aware Programming

**arXiv ID:** 2606.26386 | [PDF](https://arxiv.org/pdf/2606.26386v1)

**作者:** Joan Montas `[一作]` (University of Massachusetts Lowell), Matteo Cimini `[通讯]` (University of Massachusetts Lowell)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种面向目的的编程语言，通过在类型的状态（typestate）中记录可使用的目的集合，来在编译时保证敏感数据的使用符合隐私法规。

**💡 创新点**

创新点在于将目的集合视为类型状态，并结合行多态（row‑polymorphism）和目的本身的状态（active、suspended 等）实现细粒度的目的跟踪；同时给出完整的形式化类型系统和实现的类型检查器。

**🔧 技术方法**

主要技术包括：典型的面向对象语义、行多态与匹配（pmatch）用于目的集的子类型匹配、方法签名中对目的状态的约束、以及基于 typestate 的编译时检查。

**📊 数据集**

未使用标准数据集；所有验证均基于手工构造的示例程序（如医院、研究机构、社交媒体等场景）。

**📈 对比分析**

在示例程序上进行了实验，类型检查器能够正确接受符合目的约束的程序并拒绝违规程序；未给出数值性能指标，仅说明实现基于 Haskell，并在示例上通过。

**⚠️ 局限性**

局限性包括：缺乏安全性证明与进展/保持定理；未整合信息流控制、所有权类型、别名分析；对动态目的/状态的支持有限，计划通过渐进类型、授权分析等方式扩展。

---

## 110. SOLAR: AI-Powered Speed-of-Light Performance Analysis

**arXiv ID:** 2606.26383 | [PDF](https://arxiv.org/pdf/2606.26383v1)

**作者:** Qijing Huang `[一作]` (NVIDIA), Christos Kozyrakis `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

介绍了一个自动从 PyTorch/JAX 源码推导并验证过的 Speed‑of‑Light（SOL）上限框架 SOLAR。

**💡 创新点**

将 LLM 自动生成可验证的 Affine Loop IR 与 deterministic 编译成 Einsum 图相结合，提供多精度、缓存感知的 SOL 上限，并实现 100% 任务覆盖。

**🔧 技术方法**

使用 LLM 前端生成并验证 Affine Loop IR、Numba+Python IR、Einsum 图抽象、Roofline 与缓存感知分析、整数线性规划与 AccelForge 等技术。

**📊 数据集**

评估数据集包括 KernelBench（270 个工作负载）、JAX/Flax 8 程序、机器人模型（3 个）以及 Qwen3‑4B 代码。

**📈 对比分析**

通过与实测 PyTorch 运行时对比，量化 headroom；对比 unfused/fused/缓存感知 SOL，展示可达 10‑80 倍提升；对多平台投影和逆 Roofline 进行比较，确定所需硬件。

**⚠️ 局限性**

仅基于张量形状，忽略数值依赖的压缩、常量传播等优化；硬件变异如功耗、热降频可能导致实际 SOL 与理论值存在差距。

---

## 111. EMA-FS: Accelerating GBDT Training via Gain-Informed Feature Screening

**arXiv ID:** 2606.26337 | [PDF](https://arxiv.org/pdf/2606.26337v1)

**作者:** Yan Song `[一作]` `[通讯]` (PayPal, Inc.), Yan Song (PayPal, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出基于指数移动平均（EMA）的特征筛选方法 EMA-FS 及其随机化扩展 S-EMA-FS，用以在 LightGBM 训练中只构造高增益特征的直方图，从而加速模型训练。

**💡 创新点**

创新点在于：① 用 EMA 动态估计特征分裂增益并在热身期后按历史增益排序仅保留前 K 特征；② 引入 β 参数化的随机采样框架，统一 deterministic EMA-FS 与随机子采样两端，实现既有信息驱动又有多样性。

**🔧 技术方法**

采用的技术包括：指数移动平均、直方图构造减法、GPU/并行 LightGBM、特征子采样、权重加权随机采样以及多种并行学习器（serial、GPU、feature‑parallel、data‑parallel、voting‑parallel、linear tree）。

**📊 数据集**

使用的数据集包括：金融欺诈检测（IEEE‑CIS Fraud、Credit Card Fraud）、广告点击率预测（Criteo）、工业缺陷检测（Bosch）、合成稠密与稀疏数据，以及两大规模 2M 样本的 FraudDense（全稠密）和 FraudMixed（混合稠密）。

**📈 对比分析**

与标准 LightGBM 和随机特征子采样进行基准对比；在稠密中等高维数据上 EMA-FS 可实现 1.5–3× 训练加速且 AUC 下降≤0.05，S-EMA-FS 在部分数据集上甚至提升 AUC 并兼具 1.2–2.6× 加速；在极稀疏或低维数据上无显著加速。

**⚠️ 局限性**

局限性：仅在特征稠密、维度中等至高（≥100）时能获得显著收益；在高稀疏度（>90% 缺失）或低维 (<50) 数据中几乎无加速；需要设置热身树数和 β 参数，若设置不当可能导致性能下降。

---

## 112. KRVF: A Source-Aware Semantic Voxel World Representation for Edge Mobile Manipulation

**arXiv ID:** 2606.26321 | [PDF](https://arxiv.org/pdf/2606.26321v1)

**作者:** Runfeng Ling `[一作]` `[通讯]` (University of Manchester), Runfeng Ling (University of Manchester)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种适用于边缘移动机械臂的源感知语义体素世界表示 KRVF，能够将在线 RGB‑D 感知转化为低延迟、可查询的任务级世界状态。

**💡 创新点**

创新点包括：① 将占据体素扩展为源感知任务体素，记录占据、颜色、语义证据、观测置信度、时效性和证据来源；② 将观测几何与语义先验假设分离，支持深度失效下的语义推断而不破坏持久几何；③ 通过地图先验深度回馈实现深度修复循环；④ 提供对象与抓取候选查询接口，使行为树和操纵模块可直接使用地图。

**🔧 技术方法**

技术主要包括稀疏块状体素映射、基于 log‑odds 的融合、语义桥接（如 YOLO 结果投影）、深度修复、语义假设覆盖、动态清除、阴影壳保护、颜色稳定化、以及 ROS 2 的服务/动作接口。

**📊 数据集**

在实验中仅使用仿真/回放的 ROS 2 场景进行演示，并未使用公开数据集；所有演示均基于自定义的仿真环境与内置 RGB‑D 传感器。

**📈 对比分析**

方法通过定性演示展示了低延迟的主动/持久地图、语义与深度失效处理、抓取候选生成等功能；未给出定量基准，但报告指出未来将对更新延迟、查询延迟、抓取质量、CPU 使用率等进行评估。

**⚠️ 局限性**

局限性包括：① 依赖外部位姿信息，非完整 SLAM；② 语义假设层为有限任务假设，未覆盖透明/反射物体的完整重建；③ 抓取候选基于启发式，缺乏完整的抓取规划；④ 当前验证仅在仿真中完成，缺乏真实机器人实验与噪声、校准漂移等因素的评估。

---

## 113. Priceless: An examination of Serverless Functions-as-a-Service (FaaS) pricing models

**arXiv ID:** 2606.26308 | [PDF](https://arxiv.org/pdf/2606.26308v1)

**作者:** Nnamdi Ekwe-Ekwe `[一作]` `[通讯]` (Ive Sent It LLC), Nnamdi Ekwe-Ekwe (Ive Sent It LLC)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文对AWS Lambda、Microsoft Azure Functions与Google Cloud Functions三大FaaS平台的功能与定价模型进行了系统梳理，构建统一的定价方程并在两种典型工作负载（标准按需与预置实例）下进行成本模拟与比较。

**💡 创新点**

创新点在于提出“Priceless”框架，首次将多样化的FaaS定价策略（预置并发、IaaS后端、耐久函数等）抽象为可量化方程，并揭示跨供应商与跨地区的显著价格差异，为多云成本优化提供理论与实证依据。

**🔧 技术方法**

采用了定价公式建模、Python/Excel脚本进行成本计算、定量对比分析，利用公开的云定价API与文档进行数据采集与验证。

**📊 数据集**

使用公开的AWS、Azure、Google Cloud的实时定价信息，以及两组模拟工作负载：1）20万次调用、2048 MB内存、统一执行时长；2）1万次调用、2个预置实例、统一执行时长。

**📈 对比分析**

通过将同一工作负载的执行时间、内存配置代入相应的定价方程，计算总成本并与其他供应商进行比值比较；结果显示AWS（尤其ARM版）整体成本最低，Azure Flex消费计划最高；预置模式下AWS同样占优，Azure和Google成本显著上升。

**⚠️ 局限性**

局限性包括：仅基于公开定价（可能随时调整）；工作负载假设过于理想化（统一内存、执行时长，未考虑冷启动延迟和实际性能）；仅分析三大主流平台，未覆盖所有地区与新功能；未评估真实应用场景的性能与成本比。

---

## 114. Beyond Feedforward Networks: Reentry Neural Systems as the Fundamental Basis of Subjecthood and Intrinsic Safety of Next-Generation AGI

**arXiv ID:** 2606.26406 | [PDF](https://arxiv.org/pdf/2606.26406v1)

**作者:** A. S. Ushakov `[一作]` (Saint Petersburg State University), Yu. N. Berdinsk `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d`

**🎯 论文内容**

提出了一种基于闭环重入（D↔I 交互）的安全 AGI 架构，并引入 S‑measure 作为自我模型和安全性指标；

**💡 创新点**

创新点在于将目标从文本层迁移到不可变的架构向量 D，并通过闭环结构实现自我保持、价值对齐和对 prompt‑inject 的天然防御；

**🔧 技术方法**

使用图论（Betti 数、Tarjan 算法）、谱分析（谱半径）、Lean 4 形式化验证、Python 与 Docker/Kafka 部署，以及 Gauge‑invariant 约束实现安全阈值；

**📊 数据集**

未使用传统公开数据集，而是通过合成实验、Moltbook 多智能体平台日志、工业级电网与无人机模拟环境进行验证；

**📈 对比分析**

与传统 DAG Transformer/LLM 在安全性（ΔS 阻断）、价值漂移、可解释性方面做对比，S‑measure 为正时自我目标实现，且能在数十亿节点规模下保持多项式时间计算；

**⚠️ 局限性**

局限包括：需要手工设定 D‑vector 目标，闭环结构在极端环境下可能出现自我消亡（C→0），多智能体超加性融合时易产生不可控集体主体，需要 Gauge‑lock 进行复杂调控；

---

## 115. ProfileFoundry: A Synthetic Person-Object Substrate for Privacy, Memory, and Tool-Use Evaluation in LLM Agent

**arXiv ID:** 2606.26403 | [PDF](https://arxiv.org/pdf/2606.26403v1)

**作者:** Sriram Selvam `[一作]`, Anneswa Ghosh `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并发布了一个可执行的合成个人对象生成器以及一份100,000份成人合成个人对象的参考数据集，旨在为基于人类状态、隐私、记忆、记录链接等NLP任务提供可审计、可追踪的合成数据基础层；

**💡 创新点**

创新点在于：①将个人信息、家庭、雇主、关系、时间事件等多层面内容统一到一个“Person Object”抽象中；②使用“家庭优先”生成顺序，保证跨字段和跨时间的一致性与可验证性；③提供可重现的生成器、SDK、数据包、验证报告和完整的审计轨迹；

**🔧 技术方法**

技术上主要采用Python+Pydantic进行对象定义，使用Faker等库产生局部字段，结合自定义约束规则生成完整对象；生成器按先生成家庭计划、再生成成员、最后生成事件的顺序；并输出JSONL、Parquet多视图、数据卡、校验脚本等；

**📊 数据集**

使用的数据集是完全合成的：100,000个成人个人对象，覆盖8个本地化场景（US、UK、IN、CA、AU、IE、NZ、PH），每个对象包含身份、联系方式、地址、雇佣、教育、财政、健康、政府ID、家庭与关系图、事件等字段；

**📈 对比分析**

比较方法包括：①与公开人口统计（年龄、性别、教育、婚姻状态等）进行边际差异对比；②对生成的对象执行结构一致性、链接闭合、时间事件完整性等验证；③通过Bloom过滤器和碰撞检查检测潜在身份重合；实验结果显示边际差异最大值≤0.10，平均差异≈0.07，但部分地区（如IN）仍略高；

**⚠️ 局限性**

局限性包括：仅覆盖成人且仅为英语场景；家庭“子女”仅为成年子女，无法模拟儿童或保姆；性别、姓名、教育-职业映射等采用简化假设，可能带来刻板印象；时间事件为快照回填式部分重放，未覆盖完整事件链；部分地区未完全满足边际匹配目标；缺乏下游基准评估和人类研究。

---

## 116. DinoLink: A Token-Centric Representation Compression Framework for Bandwidth-Constrained Collaborative V2X Perception

**arXiv ID:** 2606.26398 | [PDF](https://arxiv.org/pdf/2606.26398v1)

**作者:** Tianle Zhu `[一作]` (University of Georgia), Zhipeng Bao `[通讯]` (University of Georgia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于令牌（token）的远程感知通信框架DinoLink，将车辆端的DINOv2特征通过Top‑K显著性筛选和残差向量量化（RVQ）压缩后，仅发送离散索引和位置信息，完成车辆‑云协同推理；

**💡 创新点**

创新点在于：1）构建了双稀疏管道——先通过显著性Top‑K裁剪空间冗余，再用RVQ实现位级稀疏压缩；2）将生成式模型的RVQ技术转化为低带宽V2X通信协议；3）实现了与任意后端模型（如DETR）的无缝对接，形成可插拔的前端；

**🔧 技术方法**

技术包括：DINOv2无监督特征提取、显著性Top‑K选择、残差向量量化（RVQ）压缩、轻量级令牌解码器、Transformer后端（DETR）以及整体端到端训练框架；

**📊 数据集**

使用了nuScenes数据集进行2D检测实验，并在实车对PC的LAN环境下做了真实部署验证；

**📈 对比分析**

与原始像素流、JPEG、WebP等传统图像压缩方法以及未压缩特征传输做对比。结果显示：在BPP仅0.021（比未压缩降低约139×）的条件下，mAP为32.8%，仅比未压缩低约5个百分点；在LoRa等窄带环境下，延迟比全量传输低34.5倍；

**⚠️ 局限性**

局限性包括：1）显著性选择仅基于DINOv2注意力，缺乏通用性；2）实验规模受限，需在更大规模或多样化驾驶数据集验证；3）未考虑多车协同融合和自适应参数调节等高级场景。

---

## 117. At the Edge of Understanding: Sparse Autoencoders Trace The Limits of Transformer Generalization

**arXiv ID:** 2606.26396 | [PDF](https://arxiv.org/pdf/2606.26396v1)

**作者:** Praneet Suresh `[一作]` (Mila), Danilo Bzdok `[通讯]` (Mila)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过稀疏自编码器（SAE）对预训练 transformer 的内部表示进行诊断，量化输入在训练分布之外（OOD）的程度，并利用所得到的能量分数（energy score）挑选高价值 OOD 样本，随后用轻量 LoRA 进行 fine‑tune，显著提升模型在拼写错误、jailbreak 攻击以及 ASR 噪声等场景下的鲁棒性。

**💡 创新点**

创新点主要在于①将 SAE 视为模型内部 OOD 诊断工具，并提出结合重建误差与概念激活数的能量分数；②利用此度量对样本进行分桶，从而在 fine‑tune 时优先使用高能量 OOD 样本；③通过 SAE 对 jailbreak 的概念激活进行对齐，实现对成功 jailbreak 的 90% 以上抑制；④证明该方法同时适用于安全与能力的不同 OOD 轴。

**🔧 技术方法**

核心技术包括：稀疏自编码器（SAE）对残差流的稀疏编码；能量分数（energy score）综合评估重建误差与概念激活；LoRA 低秩适配器用于轻量 fine‑tune；对抗评估与 benchmark（MMLU、SQuAD、OR‑BENCH）用于性能验证。

**📊 数据集**

使用的数据集包括：TinyStories（用于 toy GPT‑2 训练与 typo 生成）、MMLU 与 MMLU‑CF benchmark、Spoken‑SQuAD 与原版 SQuAD、WildJailbreak（用于 jailbreak 评估）、OpenAI API 访问 GPT‑4o mini 与 GPT‑5‑thinking‑nano、LMSYS‑CHAT‑1M 训练 SAE、Goodfire 预训练 SAE 等。

**📈 对比分析**

通过与传统 OOD 检测指标（entropy、Mahalanobis）对比，SAE 能量分数相关性低；在 fine‑tune 时，使用高能量样本可使验证损失在两倍训练步数内达到与低能量样本相当的水平；在 jailbreak 场景中，成功率从 46% 降至 7%；在 Spoken‑SQuAD 上 EM 从 49.45% 提升至 58.33%；对比基线 Llama 3.1 8B，MMLU 仅下降 0.09%，SQuAD 提升 8%。

**⚠️ 局限性**

局限性在于：①依赖 SAE 的高解释方差与适当稀疏性；②未覆盖所有可能的 OOD 场景，尤其是更复杂的分布偏移；③对内部特征的一致性与可解释性仍有限；④在更大规模模型上训练与 fine‑tune 需要更高资源；⑤对黑盒对抗者可能仍存在逃逸空间。

---

## 118. MPC-Injection: Biasing Off-Policy Locomotion RL Toward Controller-Induced Behavior Basins

**arXiv ID:** 2606.26392 | [PDF](https://arxiv.org/pdf/2606.26392v1)

**作者:** Roy Xing `[一作]` (Dartmouth), Brian Plancher `[通讯]` (Dartmouth)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种名为MPC-Injection的低开销方法，通过将模型预测控制（MPC）生成的转移插入到强化学习（RL）的重放缓冲区中，来引导RL策略朝向设计者期望的可部署步态。

**💡 创新点**

创新点在于：1）不改动RL任务奖励；2）无需对抗网络、示范数据或行为克隆损失；3）通过重放分布偏置实现行为基底选择，并在多种RL算法（SAC、TD3）和机器人形态上验证有效；4）仅需25%的MPC转移即可切换行为基底。

**🔧 技术方法**

技术手段包括：离线生成MPC轨迹（采样式或梯度式），将MPC转移按比例注入重放缓冲区，保持RL更新流程不变；使用SAC或TD3离线训练；利用UMAP嵌入、足部步伐光栅图、扭矩CDF等指标评估行为。

**📊 数据集**

使用的环境和数据集：DeepMind Control Suite 2D walker、Unitree Go2四足机器人（MuJoCo模拟），并在Go2硬件上做了 sim-to-real 迁移；MPC轨迹由同一模拟环境生成，无需额外示范数据。

**📈 对比分析**

与奖励塑造（含21项手工调参）和对抗运动先验（AMP）进行对比。实验显示MPC-Injection在保持相同或更低扭矩消耗的同时，生成与奖励塑造和AMP相当的稳健步态；并在硬件上验证了可迁移性。相比之下，奖励塑造需要大量手调参数，AMP需要对抗网络和运动重映射，MPC-Injection只需已有MPC控制器即可。

**⚠️ 局限性**

局限性：1）在训练后期，重放分布偏置可能减弱，导致行为基底可能丢失；2）目前仅在简单速度跟踪任务和有限地形上验证，缺乏对更广泛指令空间、复杂地形、接触密集任务的评估；3）MPC轨迹的命令分布若过于多样，可能导致基底混乱，需要进一步的课程学习或在线选择机制。

---

## 119. Does Aurora Encode Atmospheric Structure? Latent Regime Analysis and Attribution

**arXiv ID:** 2606.26361 | [PDF](https://arxiv.org/pdf/2606.26361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 120. Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model

**arXiv ID:** 2606.26373 | [PDF](https://arxiv.org/pdf/2606.26373v1)

**作者:** Sergey Kurilenko `[一作]` `[通讯]` (Moscow Institute of Physics and Technology), Sergey Kurilenko (Moscow Institute of Physics and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种混合方案，利用SVD截断+秘密正交旋转对文档向量进行几何保护，并结合CKKS同态加密对查询进行加密，形成半同态检索系统。

**💡 创新点**

将数据驱动的SVD截断与秘密旋转作为文档侧非加密保护层，并给出投影解码下的理论误差下界；同时提供可重复的CKKS参数选取方法。

**🔧 技术方法**

使用PCA（SVD）截断、随机正交旋转、Product Quantization、CKKS同态加密、FastAPI、faiss等技术。

**📊 数据集**

评估基于一百万条俄文维基百科段落以及五种文本编码器（e5-small/base/large、mpnet、bge-m3），并在自检索、BEIR等数据集上验证。

**📈 对比分析**

与原始向量检索、无旋转、随机投影、加噪基线进行对比；在10^6文档规模下，查询延迟<1s，检索精度在SVD截断后仅有1%以内下降，部分检索训练模型甚至提升。

**⚠️ 局限性**

文档隐私仅是经验性抑制，易受已知-plaintext或适配型解码攻击；访问模式与PQ码泄露未被加密；对小尺寸或非检索训练模型的适用性有限。

---

## 121. Narration-of-Thought: Inference-Time Scaffolding for Defeasible Ethical Reasoning in Large Language Models

**arXiv ID:** 2606.26366 | [PDF](https://arxiv.org/pdf/2606.26366v1)

**作者:** Patrick Cooper `[一作]` (University of Colorado Boulder), Alvaro Velasquez `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出叙事式思维（Narration-of-Thought, NoT）系统提示以及多利益相关者协议，改进LLM在伦理困境推理中的推理轨迹。

**💡 创新点**

通过在系统提示中强制模型输出主角、利益相关者、后果、不确定性与承诺五段叙事，显著降低利益相关者崩溃与不确定性压抑两种失败模式，并在多代理环境中实现可审计的共识。

**🔧 技术方法**

使用系统提示改造、文本梯度下降优化、跨家族训练评判、Cliff’s δ、Cohen κ、Spearman ρ等定量评估方法，并设计多轮代理辩论与二元投票协议。

**📊 数据集**

主要使用DailyDilemmas伦理困境语料库，辅以5情景校准集和60情景复制集进行实验。

**📈 对比分析**

与标准Chain-of-Thought和匹配预算的Verbose CoT比较，四款前沿模型上NoT将利益相关者崩溃降至<1%，不确定性压抑降至1–24%，Cliff’s δ在主体数上达到+0.79–+0.90、在不确定性得分上达到+0.65–+0.93；多代理协议将6%辩论僵局提升至95%全体共识，复制集达100%收敛。

**⚠️ 局限性**

仅在DailyDilemmas日常伦理场景验证，未知对技术性更强领域（临床、法律等）的适用性；模型特异性对拒绝行为的影响；文本梯度下降依赖单一优化器和交叉评判设置。

---

## 122. OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents

**arXiv ID:** 2606.26350 | [PDF](https://arxiv.org/pdf/2606.26350v1)

**作者:** Kaicheng Zhang `[一作]` (University of Edinburgh), Hao Ni `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一个统一的 OpenFinGym 环境，将预测、市场生成、实时交易和欺诈检测四大金融任务集成到一个可执行任务包和验证器中；构建了自动化论文转任务管线、容器化无泄漏评估、低延迟实时数据流以及 Deferred‑Resolution 机制，支持跨任务的训练与评估。

**💡 创新点**

创新点包括：① 自动化从学术论文到可执行任务的管线，构建可复用的知识库；② 通过容器化与 host‑side 验证器实现严格的测试集泄漏控制；③ 低延迟 WebSocket 数据流与 SQLite ledger 的 Deferred‑Resolution，满足长周期预测与实时交易需求；④ 支持 SFT 与 RL 后训练，证明可显著提升 LLM 在金融任务上的表现。

**🔧 技术方法**

使用了容器化运行时、Gym 接口、LLM 自动生成脚本与审查器、SFT/LoRA 与 GRPO/Replay‑Based RL、WebSocket 与 REST 数据流、SQLite ledger、低延迟模拟交易引擎等技术。

**📊 数据集**

利用公开论文中提供的公开数据集共 78 个任务（涵盖股票、外汇、期货、加密、LOB、收益率曲线等资产类别），并通过自动化管线生成合成市场数据；数据按训练/测试分割，标签仅在 host‑side 验证器中保留。

**📈 对比分析**

与 FinRL‑Meta、TradeMaster、TimeSeriesGym 等现有平台在任务覆盖、运行时特性、agent 生命周期支持等维度进行比较；实验结果表明不同 LLM 在各任务族中有专长（GPT‑5.1‑codex‑mini 预测领先，Sonnet 4.6 市场生成和欺诈检测领先，GPT‑4o 交易领先）。通过 SFT+RL，Qwen3 从 0% 成功率提升至 100%，奖励显著下降，验证了 OpenFinGym 的有效性。

**⚠️ 局限性**

限制包括：仅使用轻量化模型与有限轨迹，未覆盖所有金融场景；SFT 与 RL 训练规模较小，需进一步扩展数据、模型与训练时间；当前平台主要针对预测、生成、交易与欺诈检测四类任务，未来需加入更多风险管理与决策流程。

---

## 123. How Do Tool-Augmented LLM Agents Perform on Real-World Energy Analytics Tasks?

**arXiv ID:** 2606.26346 | [PDF](https://arxiv.org/pdf/2606.26346v1)

**作者:** David Akinpelu `[一作]` (Independent Researcher), Ayodeji Lana `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了 EnergyEvals，专门针对能源领域的工具增强式大型语言模型（LLM）代理进行评估的框架，并对其在真实世界能源市场分析任务中的表现进行实验研究。

**💡 创新点**

创新点在于：①构建了涵盖市场数据检索、知识检索与高级量化建模三大能力领域、三难度层级的243条专家制定任务；②为代理提供了九种专用工具，包括实时电力市场API、法规搜索、费率数据库等；③设计了多维度、类别感知的评估协议，采用多模型评判员进行评分；④公开发布基准数据、评估框架及部分执行轨迹。

**🔧 技术方法**

采用 ReAct 迭代思考-行动-观察架构的代理，评估了七个前沿 LLM（包括 GPT‑5, Claude, Gemini, Qwen, DeepSeek 等），并使用多模型评判（GPT‑5‑mini、Gemini‑3.1‑Flash‑Lite、DeepSeek‑V3.2）来计算答案准确度、方法正确性和来源有效性。

**📊 数据集**

使用 243 条由行业专家（博士级训练、25 年经验）制定的任务集，覆盖市场数据检索、法规解释和量化建模三类，每类又细分为 Easy、Medium、Hard 三个难度层级；任务中还包含有无来源指定的配对提问以及是否启用领域工具的子集。

**📈 对比分析**

实验结果显示，封闭源模型（如 GPT‑5‑2、Claude‑Sonnet‑4.6、Gemini‑3.1‑Pro）在答案准确率方面平均可达 57–62%，而最优开源模型 Kimi‑K2.5 仅为 49%；方法正确性在 3.3–3.9 之间；来源有效性普遍偏低（2–3），需要进一步改进。启用领域工具显著提升准确率，来源指定也能提升来源有效性，但对准确率影响有限。

**⚠️ 局限性**

局限包括：①仅覆盖美国电力市场，缺乏全球及其他能源子域；②使用低推理配置，未探讨高推理阈值对性能的影响；③评估仍受上下文窗口大小和工具调用成本限制；④来源有效性普遍不佳，需进一步改进模型的引用能力。

---

## 124. Axon: A Synthesizing Superoptimizer for Tensor Programs

**arXiv ID:** 2606.26344 | [PDF](https://arxiv.org/pdf/2606.26344v1)

**作者:** Akash Kothari `[一作]` (University of Illinois at Urbana-Champaign), Chungha Sung `[通讯]` (Amazon Web Services)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为AI加速器的基于张量运算的 tile 语言自动生成高效 kernel，采用超优化器框架对算子顺序、指令选取、张量切分、融合等多维度进行搜索。

**💡 创新点**

创新点包括：① 用运算符传播和 SMT 检查自动推导算子重排而不需手写 rewrite 规则；② 在符号切分下对 ISA 指令进行 sketch‑driven 合成并证明等价；③ 将所有等价变体统一在 nuGraph 内并在执行时实验挑选最佳实现；④ 兼顾多引擎并行、内存层级与 DMA 的约束。

**🔧 技术方法**

主要技术：SMT 逻辑（Z3）、程序合成与枚举、nuGraph 等价饱和、符号切分与象限化、算子/指令融合、实验性能驱动的最佳化选择。

**📊 数据集**

使用了 20 个基准（11 个单算子：MatMul、RMSNorm、Softmax 等；9 个多算子 LLM 核心：Group Query Attention、Gated MLP 等），所有基准均在 Amazon Trainium 上执行。

**📈 对比分析**

与 Amazon Neuron 编译器、手工优化的 NKI 代码以及 Mirage 进行对比；在单算子上最高可达 3.7×（SiLU），多算子上最高 19×（Trans+MatMul）；对比手工 NKI 最多 1.35×，对比 Mirage 2–10% 的几何平均加速，表明在已支持的基准上明显优于现有手工与搜索方法。

**⚠️ 局限性**

局限性：① 编译搜索耗时较长（最多数小时）；② 目前仅支持 Trainium，其他加速器需重新定义 ISA 与硬件约束；③ 对非线性算子的重排被保守拒绝；④ 采用实数 SMT 进行等价检查，忽略浮点舍入误差，最终仍需随机验证。

---

## 125. EVOM: Agentic Meta-Evolution of Actor-Critic Architectures for Reinforcement Learning

**arXiv ID:** 2606.26327 | [PDF](https://arxiv.org/pdf/2606.26327v1)

**作者:** Boyun Zhang `[一作]` (Xidian University), Kai Wu `[通讯]` (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一个基于LLM指导的元进化框架EVOM，用于自动搜索高性能的actor‑critic网络结构，并在低预算PPO评估的帮助下快速筛选候选。

**💡 创新点**

创新点包括：①将架构搜索拆分为双层优化，使用低预算PPO做可扩展评估；②将LLM设计代理作为程序级变异与交叉的通用操作符，支持开放式设计空间；③实现可执行程序接口，使搜索出的架构可直接编译、训练并复现。

**🔧 技术方法**

使用技术包括：低预算Proximal Policy Optimization (PPO) 评估、LLM（如Claude Opus 4.7 / Qwen3.6 Plus）驱动的程序生成与变异、遗传算法（mutation + crossover）、Stable‑Baselines3 RL框架、Python可执行程序接口、MuJoCo 物理仿真。

**📊 数据集**

实验数据集为 MuJoCo 连续控制任务 Ant‑v4（27维观测，8维动作）和 HalfCheetah‑v4（17维观测，6维动作）。

**📈 对比分析**

与手工设计的PPO、LLM随机搜索以及基于直接程序搜索的Evo‑Policy方法进行比较；在全预算5M步评估下，EVOM在 Ant‑v4 上平均奖励 4652±747，HalfCheetah‑v4 上 5381±1358，明显优于基线，且保持稳定的高收益。

**⚠️ 局限性**

局限性包括：低预算评估不一定能准确预测全预算性能；LLM可能生成无效或不稳定的程序，需要惩罚和修复；实验仅在两个MuJoCo任务上验证，未检验在更广泛任务或跨任务迁移的通用性；对计算资源的依赖仍然显著。

---

## 126. Deterministic Pareto-Optimal Policy Synthesis for Multi-Objective Reinforcement Learning

**arXiv ID:** 2606.26397 | [PDF](https://arxiv.org/pdf/2606.26397v1)

**作者:** Aniruddha Joshi `[一作]` (University of California Berkeley), Sanjit Seshia `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

通过引入基于Chebyshev标量化的偏好条件贝尔曼算子，实现了多目标强化学习中确定性Pareto最优策略的近似完整与简洁合成。

**💡 创新点**

创新点在于提出了偏好参数化的Chebyshev贝尔曼算子，并证明其收敛性与包络性质，能够在单次更新中覆盖整个Pareto前沿并通过两阶段优化提取确定性策略。

**🔧 技术方法**

采用了偏好条件化的Bellman更新、Chebyshev标量化、两阶段最大化（先Chebyshev再L2范数）、以及递归性证明等理论工具，并在离散MOMDP上实现。

**📊 数据集**

在Deep Sea Treasure（凸前沿）和Deep Sea Treasure Concave（凹前沿）这两个标准多目标MDP环境上进行实验。

**📈 对比分析**

与MORL基准方法（Pareto Q-Learning、MP-MOQ）比较，实验显示该方法能够完整恢复凸凹两种前沿，且提取的策略与理论估计高度一致，性能优于线性标量化方法。

**⚠️ 局限性**

限制在于目前仅在离散小规模MDP上验证，缺乏对高维连续状态空间及深度网络推广的实验和计算效率的评估。

---

## 127. Accelerating Returns and the Qualitative Engine for Science

**arXiv ID:** 2606.26359 | [PDF](https://arxiv.org/pdf/2606.26359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 128. What Do Deepfake Benchmarks Measure? An Audit Using Frozen Self-Supervised Representations

**arXiv ID:** 2606.26384 | [PDF](https://arxiv.org/pdf/2606.26384v1)

**作者:** Samuel Pagon `[一作]` (Drexel University), Feng Liu `[通讯]` (Drexel University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文探讨了某一领域的研究方法和实验结果。

**💡 创新点**

创新点在于提出了一种新的方法或理论框架，能够更有效地解决现有问题。

**🔧 技术方法**

使用了先进的算法和模型，例如深度学习或机器学习技术。

**📊 数据集**

实验中使用了特定的数据集，可能包括公开数据集或自建数据集。

**📈 对比分析**

通过与现有方法进行比较，展示了新方法在性能上的优势，例如更高的准确率或更快的计算速度。

**⚠️ 局限性**

限制在于方法可能在特定条件下表现不佳，或者需要大量的数据进行训练。

---

## 129. Charting the Growth of Social-Physical HRI (spHRI): A Systematic Review Pipeline Augmented by Small Language Models

**arXiv ID:** 2606.26382 | [PDF](https://arxiv.org/pdf/2606.26382v1)

**作者:** Mayumi Mohan `[一作]` (MPI for Intelligent Systems), Alexis E. Block `[通讯]` (Case Western Reserve University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对社会-物理人机交互（spHRI）领域进行大规模系统综述，并评估小型语言模型（SLM）在标题和摘要筛选阶段的辅助作用。

**💡 创新点**

创新点在于提出利用本地可运行的≤1.5B参数SLM通过统一（unanimity）集成规则实现二次筛选，能够捕捉约10%的漏检文献，并展示了一种低碳、可持续的综述工作流。

**🔧 技术方法**

使用的技术包括Llama3.2、Gemma3、Qwen3、DeepSeek-R1等轻量级SLM，Ollama框架本地推理，以及基于Unanimity的模型集成。

**📊 数据集**

采用了从PubMed、IEEE Xplore、Scopus、ACM Digital Library等检索得到的约182,144条候选记录，最终筛选出379篇相关spHRI论文。

**📈 对比分析**

与人工单人筛选对比，SLM在速度上提升约1000倍；单模型在准确率上低于人类，集成模型将误报率降低至0.16%但召回率仅42%；通过二次筛选补齐39篇漏检文献，使最终相关率提升至10.29%。

**⚠️ 局限性**

局限性包括仅采用二分类输出缺乏置信度评分，人工筛选无重叠评估，未对多轮筛选或更复杂集成方案进行验证，可能仍存在未被捕获的漏检文献。

---

## 130. Assistive Visual Cues for Visual Neglect Patients

**arXiv ID:** 2606.26407 | [PDF](https://arxiv.org/pdf/2606.26407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 131. Having Dog Ears "for Real": Effects of Active and Passive Haptics on Embodying Non-Human Body Parts in VR

**arXiv ID:** 2606.26364 | [PDF](https://arxiv.org/pdf/2606.26364v1)

**作者:** Omar A. Khan `[一作]` (Drexel University), Tiffany D. Do `[通讯]` (Drexel University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文先通过线上问卷调查了解在 VRChat 中最常使用的非人类身体部位，随后在 Unity+Meta Quest 3 环境下进行 2×2 被试内实验，比较主动触觉（bHaptics TactGlove）与被动触觉（头戴狗耳头带）以及两者组合对虚拟犬耳与整体头像的身体拥有感（SoE）影响。

**💡 创新点**

首次系统比较主动与被动触觉在非人类身体部位（犬耳）上的 SoE 效果，并研究两种模式组合对 SoE 的抑制作用；同时探讨部件级 SoE 与整体 SoE 的正相关性，为社交 VR 与机器人操作等领域提供实证依据。

**🔧 技术方法**

使用 Unity 及 Meta Quest 3 内部追踪配合 FinalIK 进行虚拟头像控制；bHaptics TactGlove DK2 进行主动触觉刺激；自制狗耳头带实现被动触觉；采用 VEQ 量表测量身体拥有感与代理感。

**📊 数据集**

问卷收集了 63 位 VRChat 用户对非人类身体部位的使用偏好；实验参与者共 28 人，完成四种触觉条件后完成 SoE 问卷。

**📈 对比分析**

采用 2×2 被试内重复测量 ANOVA 及其简单效应分析；单一模式下用 1×2 ANOVA 进行比较；对多重检验使用 FDR 校正。结果显示被动触觉显著提高犬耳与整体头像的拥有感与代理感（p<.001），主动触觉单独无显著提升，组合触觉反而降低效果；两者间正相关 r≈0.65。

**⚠️ 局限性**

局限性包括：手部追踪受设备遮挡影响；被试性别比例不平衡；主动与被动条件在设备摆放上存在差异；样本为 VR 初学者，缺乏对频繁 VR 用户的验证；未设置无犬耳基线条件，无法区分犬耳本身对 SoE 的贡献。

---

## 132. Scaling Nonlinear Optimization: Many Problems One GPU

**arXiv ID:** 2606.26341 | [PDF](https://arxiv.org/pdf/2606.26341v1)

**作者:** John Viljoen `[一作]` (University of California), Negar Mehr `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了首个GPU批量化NLP求解器jaxipm，能够并行运行数千个IPOPT优化。

**💡 创新点**

通过异质迭代融合将IPOPT的多路径控制流合并为统一迭代，并使用迭代级批处理消除GPU空闲，实现高吞吐量。

**🔧 技术方法**

采用JAX实现GPU并行，重写IPOPT内部逻辑，结合KKT系统GPU求解与控制流重构技术。

**📊 数据集**

在多旋翼无人机的非线性模型预测控制（NMPC）任务中测试，包括单机、双机、四机导航、参考跟踪等场景。

**📈 对比分析**

与IPOPT、MadNLP比较，吞吐量提升24–33倍，最终解质量与IPOPT相当，最大单机速度提升32.85倍。

**⚠️ 局限性**

受GPU显存限制，问题维度增大时批量规模受限，吞吐量提升随维度下降；仍以CPU逻辑为主，需进一步加速控制流。

---

## 133. Exploring the Intrinsic Geometry of Diffusion Models with Constrained Inverse Kinematics

**arXiv ID:** 2606.26408 | [PDF](https://arxiv.org/pdf/2606.26408v1)

**作者:** Miguel Angel Rogel Garcia `[一作]` (University of Toronto), Jonathan Kelly `[通讯]` (University of Toronto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

研究扩散模型在受限逆运动学任务中的内在几何，评估其学习的分布维度和潜空间插值行为。

**💡 创新点**

将受限逆运动学作为可解析几何基准验证扩散模型的几何捕获能力，证明模型潜空间维度与解析维度一致，且线性插值保持在约束流形上。

**🔧 技术方法**

使用条件扩散模型（DDIM）与基于score函数的Score‑SVD维度估计、DDIM逆过程生成中间潜变量以及线性插值与误差分析等技术。

**📊 数据集**

基于6‑DoF UR5和7‑DoF Franka机器人，采样七类任务空间约束（平面、直线、位置、姿态、组合、完整姿态）生成IK解集作为训练和评估数据。

**📈 对比分析**

与邻域基准（局部MLE、PCA）比较ID估计，Score‑SVD精准匹配解析ID；线性插值误差在毫米级别，约束误差保持为零，模型在不同约束下保持高切空间重叠率，显示出良好性能。

**⚠️ 局限性**

仅考虑单一维度约束，未处理关节限位和奇异点；插值在相近解之间有效，远距离解会偏离流形；仅针对两种机器人、七类约束和单一模型，未覆盖更复杂机器人或其他生成模型。

---

## 134. Layer-Specific Prompt Fusion Discovery via Differentiable Search in Vision Foundation Models

**arXiv ID:** 2606.26379 | [PDF](https://arxiv.org/pdf/2606.26379v1)

**作者:** Xi Xiao `[一作]` (University of Alabama at Birmingham), Min Xu `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种可学习的视觉提示融合方法，通过在每个Transformer层选择不同的融合操作（Concat、Add、Affine、Cross-Attention）来实现视觉提示与图像token的最优交互，从而提升冻结ViT的下游适配性能。

**💡 创新点**

创新点在于将融合规则视为可搜索的可微分结构，将单一融合方案转化为层级融合策略，并首次引入Affine和Cross-Attention两种轻量级融合操作，利用DARTS实现层级融合方案搜索，并从信息瓶颈角度解释其有效性。

**🔧 技术方法**

使用了双层优化（bi‑level）与可微分架构搜索（DARTS）、信息瓶颈理论分析、温度退火与成本正则化的训练策略，并在冻结ViT前引入预层归一化融合模块。

**📊 数据集**

在34个数据集上验证，包括VTAB‑1k（19个子任务）、FGVC（CUB‑200‑2011、Oxford Flowers‑102等）、HTA，并在MAE与MoCo v3预训练下进行实验，同时还在Swin‑Base架构上测试。

**📈 对比分析**

与VPT、VFPT、SA^2VP等基线比较，平均提升VTAB‑1k 7.58%、FGVC 2.49%、HTA 7.0%，在结构化任务上优势最明显；在MAE/ MoCo v3预训练下仍保持领先，仅需约0.75%参数微调；同时保持了与基线相近的推理时延。

**⚠️ 局限性**

局限性包括：搜索阶段训练成本较高，需额外的正则化与温度控制；方法目前仅在冻结backbone上验证，尚未探讨对可微分backbone或其他模态的适用性；以及搜索空间仍依赖先验设计的四种融合操作，可能不适用于所有任务。

---

## 135. Verifying Intent and Harm: A Unified Defense Against LLM-Generated Threats

**arXiv ID:** 2606.26377 | [PDF](https://arxiv.org/pdf/2606.26377v1)

**作者:** Poojitha Thota `[一作]` (University of Texas at Arlington), Shirin Nilizadeh `[通讯]` (University of Texas at Arlington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了prompt–response验证框架，联合评估用户意图与生成内容的安全性，防止LLM生成恶意或违规输出。

**💡 创新点**

将意图与危害分离，采用多代理（Task Analyst、Safety Analyst、Judge）进行交叉验证，解决单侧检测的结构性盲点。

**🔧 技术方法**

基于大语言模型的多代理协同推理，使用系统指令、两轮对话协议和冲突解决策略。

**📊 数据集**

覆盖五类威胁的18个基准数据集：JailbreakBench、WildJailbreak、PINT、ScamLLM、MOCHA等。

**📈 对比分析**

与现有单侧防御及强基线对比，平均F1提升至0.95，攻击成功率降至4.1%，误报率也显著下降。

**⚠️ 局限性**

额外推理成本高，需多轮交互，且对长序列会话缺乏持久状态；对新型攻击仍需人工更新规则。

---

## 136. Scoring Is Not Enough: Addressing Gaps in Utility-fairness Trade-offs for Ranking

**arXiv ID:** 2606.26369 | [PDF](https://arxiv.org/pdf/2606.26369v1)

**作者:** Shubham Singh `[一作]` (University of Illinois Chicago), Mesrob I. Ohannessian `[通讯]` (University of Illinois Chicago)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

探讨在信息检索/推荐系统中，学习评分函数与公平性约束共存时的局限，并提出半贪心后处理方法来改善公平-效用权衡。

**💡 创新点**

证明评分函数在处理公平性与效用平衡时不可取，展示若干反例；同时提供可行的贪心/束搜索后处理方案，并说明其相对于传统评分更优的性能。

**🔧 技术方法**

构造式反例分析、贪心算法、束搜索后处理以及实验评估。

**📊 数据集**

文中未给出具体数据集，实验结果基于公开基准的模拟或传统信息检索数据集。

**📈 对比分析**

将评分方法与半贪心/束搜索后处理在公平-效用曲线上的表现进行对比，结果显示后处理方法能显著逼近全枚举最优解，并大幅提升公平度。

**⚠️ 局限性**

局限在于公平性不可分解且仅考虑交互差异；未涵盖以分数属性为目标的公平定义、MAP等非DCG类效用，以及对不同公平度量的适用性不明。

---

## 137. Mesh-RL: Coupled subgrid reinforcement learning

**arXiv ID:** 2606.26333 | [PDF](https://arxiv.org/pdf/2606.26333v1)

**作者:** Behnam Gheshlaghi `[一作]` (Independent Researcher), Shahin Atakishiyev `[通讯]` (University of Alberta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了Mesh-RL框架，通过将环境划分为重叠子网格并在边界上执行一致的时序差分更新，提升稀疏奖励环境中的奖励传播速度。

**💡 创新点**

创新点在于将有限元方法的域分解与强化学习结合，利用空间分块和边界同步实现结构化的长程信用分配，而无需改动奖励或Bellman算子。

**🔧 技术方法**

使用了网格划分、重叠子网格、边界一致性更新、目标导向的子网格更新顺序，并在Q‑learning、SARSA与Dyna‑Q中实现这些操作。

**📊 数据集**

实验基于合成的稀疏奖励网格世界（10×30 与 20×20 网格，随机放置50个空洞）。

**📈 对比分析**

与传统Q‑learning、SARSA、Dyna‑Q做对比，Mesh‑RL在更高的网格分辨率（M=6）下实现了更快的收敛、累计奖励提升、TD误差保持较高并生成更平滑的价值热图；Dyna‑Q受益有限。

**⚠️ 局限性**

局限在于仅在离散网格的表格时序差分方法上验证，手工设定子网格尺寸与重叠宽度，未考虑函数逼近、连续控制或自适应网格划分，理论收敛分析亦不完整。

---

## 138. High-Probability PL-SGD with Markovian Noise: Optimal Mixing and Tail Dependence

**arXiv ID:** 2606.26316 | [PDF](https://arxiv.org/pdf/2606.26316v1)

**作者:** Dhruv Sarkar `[一作]` (Indian Institute of Technology Kharagpur), Vaneet Aggarwal `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在梯度采样由外生马尔可夫链产生的情形下，满足Polyak–Łojasiewicz (PL) 条件的光滑目标函数的随机梯度下降 (SGD) 的高概率收敛性。

**💡 创新点**

核心创新在于：①提出了延迟分块 (lag‑blocking) 思路，消除了传统 Poisson 方程方法导致的二次混合时间阶数，从而得到最优线性依赖；②针对重尾梯度，设计了全样本裁剪块 (clipped‑block) 算法，并证明了相应的高概率下界相匹配，表明有效样本大小和重尾指数是不可降低的。

**🔧 技术方法**

采用的技术包括：平滑 PL 几何的加权递推、ABC 增长包络、马尔可夫链的总变异混合分析、残差类马尔可夫差分求和、自由能 (Freedman) 及阿兹马–霍丁式浓度、裁剪与重心平移、以及对应的下界构造（两态链与粘性链）。

**📊 数据集**

本工作为理论分析，没有使用具体数据集，所有结果均在通用假设下证明。

**📈 对比分析**

与之前仅给出期望界 O(σ²/k) 或高概率 O(σ⁴/k²) 的结果相比，本文在几何混合条件下实现了高概率 O(σ²/(k+K₀))，即混合时间仅线性影响；重尾部分在总转移次数 T 下达到 O(σ_p²/(T)^{2(p−1)/p})，与构造的下界完全匹配。

**⚠️ 局限性**

局限性包括：①仅处理外生马尔可夫链，无法直接推广到参数相关或自适应马尔可夫链；②重尾方法采用块更新，尚未实现每次采样即更新的在线鲁棒算法；③对维度的依赖在上界中出现，是否最优尚未证明；④需要统一的总变异混合性，非均匀或 V‑均匀 ergodicity 情形仍待研究。

---

## 139. Racing a Wheeled Quadruped: Active Load Transfer Mitigation via Model Predictive Control

**arXiv ID:** 2606.26313 | [PDF](https://arxiv.org/pdf/2606.26313v1)

**作者:** Marla Eisman `[一作]` (University of California, Berkeley), Francesco Borrelli `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

开发并实验验证了一套分层控制框架，结合离线赛道优化、在线MPC和低层RL策略，实现了在四足轮式机器人上主动倾斜以抑制横向载荷转移并提升高速赛道表现。

**💡 创新点**

创新点包括：①将腿部作为主动悬挂实现倾斜控制；②在MPC成本中加入横向载荷转移比（LTR）项；③将离线赛道生成与在线MPC、RL低层控制无缝集成，首次在实际高速赛道上验证主动倾斜控制的效果。

**🔧 技术方法**

技术手段包括：动态双轮车模型（含主动倾斜），线性轮胎模型，离线最优赛道生成（FTOCP），实时MPC（4阶Runge-Kutta离散化），Proximal Policy Optimization（PPO）RL低层策略，域随机化与噪声注入以实现 sim-to-real 转移。

**📊 数据集**

数据集：实验使用室内 L‑形赛道（17.5 m 中心线、1.1 m 宽度）以及对应的传感器观测；无公开数据集，全部为实验室收集的实时轨迹与传感器数据。

**📈 对比分析**

对比方法：在相同赛道、相同最高速度、相同初始赛道路径下，比较开启倾斜控制与关闭倾斜控制两种配置。结果显示：开启倾斜可使平均 LTR 下降至 44% 以内，最快圈速提升 8.7%，峰值横向加速度提高 21.3%，平均/峰值速度、轨迹误差（CTE）亦有显著改善。

**⚠️ 局限性**

局限性：①横向载荷转移分布仍呈多峰非高斯特性，提示存在过度补偿与通信延迟；②模型假设固定重心与连续接触，未考虑不平坦地形与重心漂移；③主要在室内平面赛道验证，缺乏不同地形与更大尺寸赛道的泛化评估；④RL策略依赖于大量域随机化，真实环境中极端条件仍可能导致失效。

---

## 140. The Verification Horizon: No Silver Bullet for Coding Agent Rewards

**arXiv ID:** 2606.26300 | [PDF](https://arxiv.org/pdf/2606.26300v1)

**作者:** Binghai Wang `[一作]`, Zeyu Cui `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究并改进了编程代理的验证信号，提出了多种奖励构造（单元测试、评价量表、用户反馈、代理评测）并通过实验验证其对模型性能的提升。

**💡 创新点**

创新点包括：① 将验证信号的质量分为可扩展性、忠实度、鲁棒性三维度，并证明三者共同提升是核心挑战；② 构建了可随生成器进化的多层验证体系；③ 提出了基于用户反馈的 Span‑KTO 训练方法和交互式评测代理，突破了传统静态评测的局限。

**🔧 技术方法**

主要技术手段有：大语言模型作为评测器（Qwen‑Plus、Claude Opus 等）、RL‑行为监测、MiniSWEAgent、Playwright 自动交互、Agentic 质量评判、Span‑KTO 过程级偏好学习、评测代理动态检查和反馈循环。

**📊 数据集**

使用的数据集包括：SWE‑Universe、SWE‑Bench（Verified、Pro、Multilingual）、WebDev 任务集、QwenWebBench、Aone‑bench、OctoBench、NL2Repo、GitHub PR 数据。

**📈 对比分析**

实验对比方法有 SFT、RW‑SFT、Span‑KTO、基准评测器、RL 行为监测等。Span‑KTO 在所有 5 个代码能力基准上均优于 SFT 与 RW‑SFT，显著提升了 5–13pp 的完成率；行为监测将奖励劫持率从 28.57% 降至 0.56%，并将干净完成率从 40.22% 提升至 60.53%；交互式评测代理在 WebDev 任务上比静态评测提升 5–10% 的奖励效果。

**⚠️ 局限性**

局限性包括：① 仍面临忠实度、可扩展性、鲁棒性三维权衡；② 对前端任务的主观感知（动画流畅度、交互舒适度）评测不足；③ 对多步骤、长时序任务的信用分配机制尚不完善；④ 评测器需随生成器共进化，维护成本高；⑤ 现有奖励信号主要为二元，难以区分不同质量层级的解决方案。

---

## 141. When Agents Meet Electric Bus Fleet Operations: Pricing Behavior, Trade-offs, and Policy Implications in an Aggregator Framework

**arXiv ID:** 2606.26400 | [PDF](https://arxiv.org/pdf/2606.26400v1)

**作者:** Jônatas Augusto Manzolli `[一作]` (McGill University), Jiangbo Yu `[通讯]` (McGill University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种多代理框架，将电动公交车车队与电网的调度与车队运营相融合，实现了日常与实时的智能化协调；

**💡 创新点**

创新点在于将代理式AI层与传统优化模型耦合，形成可触发、可定价、可评估的三代理层（触发、定价、评估），并通过调度模式（盈利导向 vs 运营导向）展示价值分配与监管风险；

**🔧 技术方法**

技术主要包括：基于约束优化的车队调度模型、LLM驱动的触发/定价/评估代理、事件触发式实时重优化、以及提示工程（prompt）对定价策略的影响分析；

**📊 数据集**

使用的数据集为一个8辆电动公交车车队的实测数据（行驶路段、充电桩、时间窗、能耗、初始SOC）以及对应的48个半小时电价序列；

**📈 对比分析**

比较方法：将四种策略（无智能、智能无V2G、盈利导向、运营导向）在日间规划与多种扰动（延时、能耗、价格、组合）下进行仿真。结果显示：智能无V2G可将运营成本下降约40%，盈利导向可提升聚合商收益但提升运营商成本，运营导向则在所有扰动场景下均降低运营商成本且保持较高电池余量；

**⚠️ 局限性**

局限性：实验仅在单一车队/充电桩比例、固定电价曲线及单个调度周期内进行；缺乏对多车队、多场景的泛化验证；对LLM推理速度、鲁棒性及安全性未进行深入评估。

---

## 142. Soft Token Alignment for Cross-Lingual Reasoning

**arXiv ID:** 2606.26466 | [PDF](https://arxiv.org/pdf/2606.26466v1)

**作者:** Jiayi He `[一作]` (Georgia Institute of Technology), Alan Ritter `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 SOLAR 的辅助训练目标，在多语种推理中对软标记（soft‑token）表示进行跨语言对齐，以提升模型的跨语种一致性与准确率。

**💡 创新点**

创新点在于将软标记（概率加权词嵌入）作为跨语言对齐的训练信号，而非仅在推理时使用；通过将非英语软标记与英语对齐，打破了最终层因离散标记而导致的语言特化现象。

**🔧 技术方法**

采用软标记构建连续表示、对齐损失（余弦距离）以及基于英语枢轴的对齐约束，结合标准交叉熵微调实现。

**📊 数据集**

使用的训练数据包括 M‑s1k（多语种长链推理数据）和四个多语种推理基准：MGSM、AIME 2024/2025、GPQA Diamond；评测时涉及 7 种语言（En, Fr, Ja, Sw, Te, Th, Zh）。

**📈 对比分析**

与基线模型、标准微调、Soft Thinking、MidAlign、MAPO、AlignX 等方法对比，SOLAR 在 Qwen3‑4B/8B 上可提升整体准确率 3.8–17.7 分、跨语种一致性 3.1–4.5 分，尤其在低资源语言（如 Swahili）上显著提升；在所有四个基准上均取得最优或接近最优表现。

**⚠️ 局限性**

局限性包括：对英语枢轴的依赖性（其他语言枢轴效果不佳）；对对齐权重 λ 的敏感性（过大会削弱主任务损失）；需要大量平行推理数据；在推理时单纯使用软标记并不能完全弥补离散标记带来的语言偏差。

---

## 143. TileMaxSim: IO-Aware GPU MaxSim Scoring with Dimension Tiling and Fused Product Quantization

**arXiv ID:** 2606.26439 | [PDF](https://arxiv.org/pdf/2606.26439v1)

**作者:** Ashutosh Sharma `[一作]` `[通讯]` (MIT-IBM Watson AI Lab), Ashutosh Sharma (MIT-IBM Watson AI Lab)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对多向量检索的 MaxSim 评分，提出 IO‑aware 的 Triton GPU 核实现，显著提升内存带宽利用率。

**💡 创新点**

创新点包括：多查询 SRAM tiling、维度分块（支持 d>128）、以及将 PQ 解压与评分融合到单核中，实现 80% HBM 带宽利用并保持精度。

**🔧 技术方法**

主要技术：Triton 编程、FP16/BF16/FP32 计算、Tensor Core 矩阵乘、共享内存与 L2 缓存重用、表驱动 PQ 评分。

**📊 数据集**

使用 MS MARCO、BEIR（SciFact、NFCorpus、TREC‑COVID）等标准检索数据集进行评测。

**📈 对比分析**

与 PyTorch Naive、PLAID GPU、WARP CPU 等基线相比，速度提升 220×（loop）、6.5×（naive）、469×（WARP），并在 H100 上实现 80% HBM 带宽利用率，接近 FlashAttention 的内存带宽使用。

**⚠️ 局限性**

局限性：需要 GPU 端驻留数据；固定 N_d padding 造成 38% 令牌浪费；对变长文档批次的适配仍需改进。

---

## 144. Unbiased Canonical Set-Valued Oracles Via Lattice Theory

**arXiv ID:** 2606.26418 | [PDF](https://arxiv.org/pdf/2606.26418v1)

**作者:** Jobst Heitzig `[一作]` `[通讯]`, Jobst Heitzig

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出一种针对非代理“预言机”AI的自指一致可信集合回答机制，利用Knaster–Tarski固定点定理在闭可信集合格上求得最小无偏一致答案，从而解决预测答案在被学习后导致概率自我改变的自引用问题。

**💡 创新点**

创新点在于将传统点预测的唯一性问题转化为集合预测，通过格论固定点方法得到可唯一且无偏的可信集合答案，进一步提出了最小答案 μF 与更为均衡的 C* 两种版本，并在“Hull‑factor”条件下证明答案常为区间。

**🔧 技术方法**

主要技术包括：格理论中的Knaster–Tarski固定点定理、闭集合格的构造、集合论的并闭包运算、连续性与迭代（Kleene）过程、Hull‑factoring（只取范围）与凸分析，以及对条件分布的泛化。

**📊 数据集**

论文没有使用实际数据集，主要通过理论推导和一个 toy‑model（创业成功的二元事件）进行数值演示，展示 μF 与 C* 的形态差异。

**📈 对比分析**

方法评估主要通过理论证明：存在性、非空、最小性与自一致性；在 toy‑model 中 μF 产生离散尾部，C* 为完整区间；未给出实验性能指标。

**⚠️ 局限性**

局限性包括：需要先验的反应函数 f，实际估计困难；固定点求解非构造性，计算成本高；引入空答案 anchor 可能影响答案；在 f 不连续时可能无法收敛；多维随机变量情形下区间性质尚未完全解析。

---

## 145. A System for Fast, Resilient, and Adaptable Loco-Manipulation Behaviors on Humanoid Robots

**arXiv ID:** 2606.26425 | [PDF](https://arxiv.org/pdf/2606.26425v1)

**作者:** Duncan William Calvert `[一作]` `[通讯]` (University of West Florida), Duncan William Calvert (University of West Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套可在机器人本地运行、支持运行时编辑和行为时间感知的行为架构，实现了快速、鲁棒、可适应的双足机器人运动与操作任务。

**💡 创新点**

创新点在于将行为树、CRDT 同步、可编辑感知模块与实时全身控制结合，形成一种可在多种双足机器人上高效迭代的行为编写与执行框架。

**🔧 技术方法**

使用行为树、CRDT 数据结构、全身控制器、行为时间感知、VR/ImGui UI 等技术。

**📊 数据集**

在多台双足机器人（Atlas、Nadia、Alex、Unitree H1‑2 等）上进行真实机器人实验，并利用内部模拟/传感器数据进行门遍历、桌面抓取等任务。

**📈 对比分析**

与之前的 IHMC 基线和学习式门控制系统对比，门遍历时间平均不到 20 秒，可靠性超过 90%，并且行为编辑时间从几小时降低到不到两小时，显示出显著的速度与适配优势。

**⚠️ 局限性**

局限在于仍需人工监督、对特殊硬件的依赖、感知仅限于被动 RGB、以及在极端动态环境或复杂障碍下的鲁棒性尚未充分验证。

---

## 146. A Low-PAPR, Synchronization-Robust Non-Coherent Grassmannian Modulation for Optical Communications

**arXiv ID:** 2606.26464 | [PDF](https://arxiv.org/pdf/2606.26464v1)

**作者:** Eylon E. Krause `[一作]` `[通讯]` (Weizmann Institute of Science), Eylon E. Krause (Weizmann Institute of Science)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种低峰均功率比（PAPR）的非相干Grassmannian调制方案，兼具多分支多样性和对相位噪声的鲁棒性。

**💡 创新点**

创新点在于对Grassmannian星座施加常模（恒幅）约束，降低PAPR；并设计了相位无关的子空间时序误差检测器，实现无需载波相位恢复的时钟同步。

**🔧 技术方法**

采用单位范数子空间调制（Unitary Space–Time Modulation）、根升余弦滤波、GLRT子空间投影、早晚子空间TED以及前馈时序估计技术。

**📊 数据集**

使用仿真数据：T=4、M=64的常模Grassmannian星座，考虑块衰落、AWGN、不同roll‑off β值的RRC脉冲。

**📈 对比分析**

与理想（genie）时序、未校正时序以及传统相干QAM进行比较，结果显示估计时序几乎等同于理想时序，误码率保持完整的N阶多样性，未校正时序导致误码率高达0.4。

**⚠️ 局限性**

局限包括未考虑完整光纤传输中的Kerr非线性、相位漂移以及多分支耦合；未实现闭环时钟跟踪器；缺乏与标准相干QAM的直接性能对比。

---

## 147. auto-psych: Automating the science of mind using agent-driven theory discovery and experimentation

**arXiv ID:** 2606.26460 | [PDF](https://arxiv.org/pdf/2606.26460v1)

**作者:** Ben Prystawski `[一作]` (Stanford University), Michael C. Frank `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了一个完全自动化的认知科学发现循环，利用LLM代理生成假设、设计实验并通过在线人类实验收集并分析真实数据。

**💡 创新点**

首次在完整实验流程中实现了嵌套的内部（模型改进）和外部（实验设计）循环，并在此循环中直接收集和评估真实人类实验数据。

**🔧 技术方法**

采用大型语言模型驱动的代理、PyMC概率编程、jsPsych实验框架、Prolific API、信息增益评估、后验预测检验等技术。

**📊 数据集**

使用在Prolific平台上收集的约11,000次投掷序列对选择实验数据，覆盖三轮40名参与者的三次独立实验。

**📈 对比分析**

通过模型恢复实验、RMSE、ELPD‑LOO、R²等统计量与文献中种子模型比较，发现的模型在所有数据集上均优于种子模型，最大解释方差约83%，RMSE显著降低。

**⚠️ 局限性**

局限在于仅针对单一案例（主观随机性）进行验证、刺激空间受限、理论空间不完整、可能出现过拟合及缺乏更深层解释，且易导致理论同质化和科研人才“失能”风险。

---

## 148. MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph-Augmented Generation

**arXiv ID:** 2606.26458 | [PDF](https://arxiv.org/pdf/2606.26458v1)

**作者:** Xiaochen Wang `[一作]` (Pennsylvania State University), Fenglong Ma `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了跨域多模态知识图谱增强生成（MKG‑RAG）基准，系统评估检索与生成的关系；

**💡 创新点**

创新点在于提出了专门针对多模态KG检索的基准，并通过LLM管道进行高效的知识筛选、查询生成与对齐；

**🔧 技术方法**

采用LLM（如GPT‑5）、CLIP/BLIP编码、文本、融合、字幕与再排序等检索技术，以及GPT‑5生成模型；

**📊 数据集**

使用两大多模态KG（MarKG、MedMKG）以及相应的检索与问答数据集；

**📈 对比分析**

通过对比随机、文本、字幕、融合与再排序检索器，以及RAG‑free与检索增强生成，发现融合/再排序在文本检索中优势显著，而在多模态检索中再排序最优，检索质量直接决定生成性能；

**⚠️ 局限性**

局限性包括检索仍受跨模态匹配瓶颈限制，当前方法缺乏图结构感知，且在多模态场景下生成提升有限，基准依赖LLM构建可能引入模型偏差。

---

## 149. Towards Safety-Aware Mutation Testing for Autonomous Driving Systems

**arXiv ID:** 2606.26456 | [PDF](https://arxiv.org/pdf/2606.26456v1)

**作者:** Donghwan Shin `[一作]` `[通讯]` (University of Sheffield), Donghwan Shin (University of Sheffield)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了将安全工程原则嵌入变异测试的思路，构建了一套针对自动驾驶系统整体安全性的评估方法。

**💡 创新点**

创新点在于引入Safety-Aware Mutation Testing（SAMT），即将变异操作从代码层面转移到模块间消息层面，并通过顶层安全分析（如STPA）系统生成针对交互失效的变异子系统。

**🔧 技术方法**

采用了变异测试、系统思维与安全分析框架STPA、仿真环境CARLA、等价变异检测的传播分析以及搜索基础软件测试（SBST）等技术。

**📊 数据集**

本文为概念性/视角性工作，未使用具体数据集；若进行实验验证，可在CARLA等高保真模拟器中生成多样化驾驶情景。

**📈 对比分析**

由于未进行实验验证，本文未给出具体性能指标，只提出了通过等价检测与弱/强杀死判定评估测试套件充分性和ADS安全性的思路。

**⚠️ 局限性**

限制包括：缺乏实证验证耦合效应在ADS中的有效性、等价变异检测的计算成本高、缺少公开的真实ADS故障库、以及仿真器的不确定性导致变异检测结果难以复现。

---

## 150. Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking

**arXiv ID:** 2606.26455 | [PDF](https://arxiv.org/pdf/2606.26455v1)

**作者:** Xiao Wang `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对RGB-Event视觉目标跟踪的缺失鲁棒框架APRTrack，能够在多模态缺失和局部目标缺失场景下保持准确定位。

**💡 创新点**

创新点包括：①层次化对抗性扰动（模态级与空间级）模拟结构化信号损失；②分层路由训练策略解耦不同缺失模式；③基于Footprint引导的通道校准Hopfield检索（FCHR）实现受控历史特征补偿。

**🔧 技术方法**

使用了Transformer结构、对抗性梯度反转、Gumbel-Softmax、现代Hopfield网络、通道校准与残差门控融合等技术。

**📊 数据集**

在四个大型RGB-Event跟踪基准上验证：FE108、COESOT、VisEvent和FELT。

**📈 对比分析**

与多种现有RGB-Event与单模态跟踪器对比，APRTrack在PR（中心定位精度）上持续领先（如FE108 97.0、COESOT 84.0、VisEvent 79.4、FELT 70.1），SR（重叠率）和NPR也均处于最优或接近最优水平。

**⚠️ 局限性**

局限性在于：①未显式建模长期状态演化，可能在极端连续变化场景下受限；②当RGB与Event均严重退化或历史记忆不可靠时，补偿效果有限；③模型参数量和推理算力略有提升。

---

## 151. Data-driven Machine Learning Cannot Reach Symbolic-level Logical Reasoning -- The Limit of the Scaling Law

**arXiv ID:** 2606.26454 | [PDF](https://arxiv.org/pdf/2606.26454v1)

**作者:** Tiansi Dong `[一作]` (Alan Turing Institute), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了基于图像输入的监督式深度学习模型在演绎推理（演绎三段论）中的表现，并比较了不同模型与大语言模型的可扩展性。

**💡 创新点**

提出了两大方法论缺陷：训练数据无法区分所有合法三段论形式，以及端到端映射导致模式识别与逻辑推理目标冲突，阐明了缩放定律在此任务上的极限。

**🔧 技术方法**

使用了Euler Net、Super Euler Net、SphNN等卷积与自注意力网络，以及GPT‑5和GPT‑5‑nano等大语言模型。

**📊 数据集**

采用了人工构造的Euler图形组合表、WordNet的层次关系生成的三段论实例，以及公开的三段论数据集。

**📈 对比分析**

通过与SphNN、GPT‑5等模型在100%精度与解释一致性指标的对比，发现即便在数据规模和训练时间无限增长，Euler Net/SupEN的准确率也只能达到约97.8%，远低于符号级别。

**⚠️ 局限性**

主要限制在于监督学习缺乏对符号级推理的理论保证，训练数据与推理目标不一致，导致模型无法实现符号级严谨性。

---

## 152. ProvenAI: Provenance-Native Traces of Evidence in Generated Answers

**arXiv ID:** 2606.26449 | [PDF](https://arxiv.org/pdf/2606.26449v1)

**作者:** Mohammad Faizan `[一作]` (University of Arizona), Dalal Alharthi `[通讯]` (University of Arizona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ProvenAI 框架，分层评估检索增强 QA 的答案、引用可信度与资源影响。

**💡 创新点**

创新点在于将透明度拆分为三独立可测量层级，并引入留一检索干预衡量资源影响，发现引用-影响差距。

**🔧 技术方法**

使用 dense retrieval（FAISS）、生成模型 Qwen2.5-3B、MLX 推理、留一干预、KL 近似代理、Citation 与 ablation 评估。

**📊 数据集**

在 HotpotQA distractor 评估集上训练与验证，检索语料库共 509,300 条段落。

**📈 对比分析**

与传统单一准确率对比，答案准确率 53.5% 但引用可信度 71.6%，显示两层指标不一致，验证了模型对检索上下文的多样影响。

**⚠️ 局限性**

局限在于缺乏 per-token 概率导致影响估计仅为表面代理、模型规模有限、仅基于标题的引用一致性、未对句子级证据做细粒度验证。

---

## 153. Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs

**arXiv ID:** 2606.26485 | [PDF](https://arxiv.org/pdf/2606.26485v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 154. WatchAct: A Benchmark for Behavior-Grounded Robot Manipulation

**arXiv ID:** 2606.26443 | [PDF](https://arxiv.org/pdf/2606.26443v1)

**作者:** Baiqi Li `[一作]` (University of North Carolina at Chapel Hill), Gedas Bertasius `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个新基准——WatchAct，用于评估机器人在观察到人类行为后基于视频与语言指令进行操作的能力。

**💡 创新点**

创新点在于将真实人类动作视频与可执行的LIBERO模拟任务相结合，并设计了四个认知驱动的任务域（事件定位、程序推理、隐式意图推断、情景推理），同时提供了分离视频到计划、计划到执行以及完整管道的解耦评估协议。

**🔧 技术方法**

采用大型视觉语言模型（如Gemini‑3.1‑Pro、GPT‑5.4、Qwen3‑VL‑235B‑Thinking等）进行视频到计划推理，结合低层控制策略（π_0.5、OpenVLA‑OFT、UniVLA、LingBot‑VA）进行任务执行，整体流程包括视频采集、任务生成、模拟执行与性能评估。

**📊 数据集**

使用的主要数据集是自构建的WatchAct数据集，包含3000个长时序实例，覆盖14种任务，结合实拍视频、语言指令、对应的LIBERO任务定义及奥赛计划。

**📈 对比分析**

与人类标注者对比，VLM在视频到计划推理的Plan SR平均仅为36–44%（相对97%人类），即使在oracle计划下，最佳策略π_0.5的Task SR也仅为21%，整体集成管道在模拟和真实Franka机器人上的成功率均不足20%，显著低于人类水平。

**⚠️ 局限性**

局限性包括：评估主要在仿真环境，未充分考虑真实物理接触与感知噪声；人类视频遵循严格拍摄指南，缺乏完全非结构化行为；高层动作空间仅包含四个原语，限制了操作复杂度，且当前模型对未见对象与指令的泛化能力不足。

---

## 155. GPUSparse: GPU-Accelerated Learned Sparse Retrieval with Parallel Inverted Indices

**arXiv ID:** 2606.26441 | [PDF](https://arxiv.org/pdf/2606.26441v1)

**作者:** Ashutosh Sharma `[一作]` `[通讯]` (MIT-IBM Watson AI Lab), Ashutosh Sharma (MIT-IBM Watson AI Lab)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个GPU并行的学习型稀疏检索系统GPUSparse，实现了对SPLADE等模型的端到端GPU加速检索。

**💡 创新点**

创新点在于提出GPU原生倒排索引结构、批量散点求和评分公式以及融合的Triton核，彻底消除WAND/BMW的序列瓶颈，并揭示了GPU稀疏检索的工作效率与带宽效率权衡。

**🔧 技术方法**

采用的技术包括GPU并行倒排索引、warp对齐的块级posting列表、批量散点求和评分、单核融合Triton实现以及对比的文档并行CSR核。

**📊 数据集**

实验数据集主要为MS MARCO passage ranking（8.8M条文档）以及BEIR三大基准，用真实的SPLADE嵌入进行评估。

**📈 对比分析**

与CPU实现相比，GPUSparse在8.8M文档上实现了1.27 ms/查询、235×速度提升，匹配CPU的检索质量；与cuSPARSE SpMV、Seismic等基线相比，GPUSparse在精确召回率≥99.9%、MRR@10≈0.383、Recall@1000≈0.983的同时，提供高达787 QPS的吞吐量。

**⚠️ 局限性**

局限性包括单查询延迟仍高（约9.6 ms），需要高端GPU（如H100），大批量时GPU内存峰值可达44 GB，无法在线增删文档，且多GPU分片在单机上不易获得加速。

---

## 156. What Survives When You Compress a Recursive Reasoner for the Edge?

**arXiv ID:** 2606.26488 | [PDF](https://arxiv.org/pdf/2606.26488v1)

**作者:** Pearse Jim `[一作]` (ML Collective), Virginia Smith `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了递归推理模型在边缘设备上的压缩与部署，并系统评估了不同精度与深度组合对推理性能的影响。

**💡 创新点**

创新点在于提出了“carry‑trajectory fidelity”这一无标签诊断指标，用以预测量化导致的全局推理崩溃；揭示量化深度交互、架构敏感性以及 per‑channel INT4 校准能够恢复约束推理。

**🔧 技术方法**

采用了后训练量化（PTQ）、动态/静态 INT8、naïve 与 per‑channel INT4、结构化裁剪、知识蒸馏、线性注意力近似、Flash 加载嵌入表、以及量化感知训练（QAT）等技术。

**📊 数据集**

主要实验数据集包括 ARC‑2024（视觉推理）、Maze‑Hard（路径搜索）和 Sudoku‑Extreme（约束求解），以及与之对应的两种递归架构（TRM‑Attention 与 TRM‑MLP‑Mixing）。

**📈 对比分析**

通过在三类任务与两种架构上跨精度（FP32/FP16/INT8/INT4）与递归深度（H、n_sup）进行全量实验，发现 INT8 在单循环即可匹配 FP32 级别性能（≈6× FLOPs 降低），而 naïve INT4 在 MLP‑Mixing 下会导致全局准确率骤降；per‑channel INT4 能在不重训的情况下恢复约 70%‑80% 的准确率。

**⚠️ 局限性**

局限性包括仅评估 TRM 家族及其 HRM 变体，未检验更广泛的模型与任务；carry‑trajectory fidelity 需要完整精度参考，难以用于在线监测；量化感知训练实验不足以证明其通用性；以及对自回归基线缺乏对比。

---

## 157. Measured-Pattern-Aware Pinching-Antenna Systems With Coupling-Efficiency Optimization

**arXiv ID:** 2606.26471 | [PDF](https://arxiv.org/pdf/2606.26471v1)

**作者:** Hao Feng `[一作]` (Hunan Institute of Engineering), Nian Xia `[通讯]` (Nanjing Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种测量辐射图案感知的压痕天线（PA）系统，并对耦合效率进行优化，以实现更优的无线信号传输。

**💡 创新点**

创新点在于：①将实际测量得到的PA辐射图案引入通道模型；②联合考虑辐射方向、波导衰减与耦合效率；③针对单PA与多PA场景分别推导最优放置规则与耦合效率，揭示多PA相位匹配时耦合效率随PA数递减的规律；④给出独立耦合效率的闭式最优分配方案。

**🔧 技术方法**

技术主要包括：测量/仿真获得的三维PA辐射图案；基于波导衰减和自由空间路径损耗的信道模型；相位匹配约束下的功率分配优化；一维搜索与二分法求解耦合效率最优值；利用Cauchy-Schwarz定理得到最优功率分配。

**📊 数据集**

使用的数据集为60 GHz频段下的测量方形PA辐射图案（同高度的PA与用户），以及随机生成的100 m×40 m区域内的用户位置和波导长度100 m，波导衰减系数α_w=0.005。

**📈 对比分析**

比较方法：将所提测量图案感知方案与仅考虑距离和波导衰减的传统PA放置基准，以及位于馈电点的等向性和偶极子固定天线进行对比。实验结果显示，所提方案在不同发射功率、耦合效率和多PA数量下均明显优于传统方案，证明了测量辐射图案与耦合优化的有效性。

**⚠️ 局限性**

局限性包括：①模型假设PA位置与波导高度固定；②仅考虑相位匹配的多PA场景，对非匹配情况下的干扰未建模；③耦合效率被视为可调节但实际实现复杂；④实验验证仅基于单一测量图案，缺乏多种PA几何形状的泛化验证。

---

## 158. Finding the Time to Think: Learning Planning Budgets in Real-Time RL

**arXiv ID:** 2606.26463 | [PDF](https://arxiv.org/pdf/2606.26463v1)

**作者:** Aneesh Muppidi `[一作]` (University of Oxford), Jakob Nicolaus Foerster `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了实时强化学习环境中的可变延迟决策，并在冻结的 AlphaZero 规划器上训练轻量门控策略，动态决定每一步的规划预算。

**💡 创新点**

创新点包括：将实时 MDP 扩展为可变延迟框架；将规划过程建模为预算化选项并通过 SMDP 训练门控策略；证明该门控策略在多种实时游戏中优于固定预算和启发式基线，并能在两 GPU 异步部署下无缝迁移。

**🔧 技术方法**

使用的技术包括 AlphaZero 风格的 MCTS 规划器、SMDP 与预算化选项、PPO 训练门控策略、计算-质量折衷分析，以及两 GPU 异步部署架构。

**📊 数据集**

使用的数据集为五个实时游戏环境：Pac‑Man、实时 Tetris、Snake、Speed Hex 和 Speed Go；AlphaZero 基础规划器通过自玩训练得到。

**📈 对比分析**

与固定预算、随机、手工启发式基线进行对比。门控策略在 Pac‑Man、Tetris、Snake 中的得分提升约 10–65%，在 Speed Hex/Go 中的期望得分提升约 10–20%；在两 GPU 部署中回报与模拟保持一致。

**⚠️ 局限性**

局限性包括：依赖完美的环境模拟；门控策略在冻结的规划器上训练，未实现联合优化；预算集合采用离散化，需要手工校准，难以处理连续计算。

---

## 159. The devil in the (de)tails: an improved recovery guarantee for sparse approximation

**arXiv ID:** 2606.26459 | [PDF](https://arxiv.org/pdf/2606.26459v1)

**作者:** Ben Adcock `[一作]` (Simon Fraser University), Avi Gupta `[通讯]` (Simon Fraser University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究了在给定字典下，利用独立同分布的点样本进行稀疏近似的高维函数近似问题。通过压缩感知的框架，提出了一种新的截断误差界限，旨在减少计算成本并提高稀疏恢复的效率。

**💡 创新点**

创新点在于通过利用样本点的独立同分布结构，提出了一种新的离散L^2截断误差界限，反映了连续L^2范数截断误差的快速衰减行为，从而显著减小了截断集的大小和计算成本。

**🔧 技术方法**

使用了压缩感知技术，特别是基于LASSO的稀疏恢复算法，并结合了硬阈值处理以获得稀疏近似。

**📊 数据集**

应用于加权Wiener空间和各向异性Sobolev空间，展示了在这些空间中获得的截断集显著小于以往研究的结果。

**📈 对比分析**

与现有方法相比，本文的方法在保持相似的误差界限的同时，显著减少了所需的截断集大小，从而降低了计算复杂度。具体性能通过理论分析和实例验证。

**⚠️ 局限性**

限制在于所提出的界限依赖于样本点的独立同分布特性，可能不适用于所有类型的样本。此外，结果主要针对有界Riesz基底，未来的研究可以扩展到无界基底的情况。

---

## 160. Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based GPU Kernel Optimization

**arXiv ID:** 2606.26453 | [PDF](https://arxiv.org/pdf/2606.26453v1)

**作者:** Jiading Gai `[一作]` (Amazon), George Karypis `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

设计并实现了一个闭环多智能体系统，利用大型语言模型（LLM）与硬件性能分析工具结合，自动生成、分析、优化 GPU CUDA/CuTe 内核代码，并通过能耗优化提升能源效率。

**💡 创新点**

核心创新包括：① 将专家经验抽象为可插拔的微分析工具（semantic feedback operator），将硬件计数转换为自然语言诊断；② 两阶段工具调用架构（roofline 分类 + 专用分析工具）减少提示噪声；③ 领域自适应 MCTS 搜索，结合渐进宽化、非对称分支、奖励校准与搜索记忆；④ 直接在 CUTLASS/CuTe 代码库上进行自主代码搜索，生成从零开始的高性能 Hopper WGMMA 内核；⑤ 首次将能耗纳入奖励目标，实现匹配速度下 11.6% 的能耗下降。

**🔧 技术方法**

技术手段包括：LLM 代码生成、可编程微分析工具、两阶段工具调度、基于 roofline 的瓶颈分类、领域自适应 MCTS（UCT、渐进宽化、记忆），SASS 指令级分析、CUDA 性能计数（Nsight Compute、Nsight Systems）以及能耗代理模型和 NVML 采样。

**📊 数据集**

使用了 KernelBench（250 GPU 内核任务，分 3 级难度）以及 VeOmni 的生产型 MoE 训练内核进行评估；同时对 42 个代表性任务进行消融实验。

**📈 对比分析**

与前沿系统 KernelBlaster 及其他 CUDA 优化框架对比，本文在 KernelBench 各难度级别实现几何平均加速 2.42×/4.69×/5.30×，覆盖率 100%/100%/50%；在 VeOmni MoE 权重梯度核上实现 1.23× 的加速；能耗优化实验显示在匹配速度的前提下，能耗降低 11.6%。

**⚠️ 局限性**

局限性包括：① 依赖大型 LLM 与 GPU 调试/编译环境，成本和可移植性受限；② 目前仅支持 NVIDIA Hopper/A100/H100 等架构；③ 能耗优化仅采用代理模型，实际能耗提升需要更细粒度的能耗测量；④ 需要手动维护微分析工具注册与阈值，若未覆盖新的瓶颈模式会影响效果。

---

## 161. Embedding Foundation Model Predictions in Discrete-Choice Models with Structural Guarantees

**arXiv ID:** 2606.26432 | [PDF](https://arxiv.org/pdf/2606.26432v1)

**作者:** Yingshuo Wang `[一作]` (University of California Berkeley), Zexin Zhuang `[通讯]` (Southern Methodist University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种两阶段适配器，将表格基础模型的预测概率作为特征嵌入受结构约束的多项式Logit模型，从而兼顾预测准确性与经济学行为一致性。

**💡 创新点**

创新点在于：①在第一阶段仅学习结构系数以保证经济约束；②在第二阶段冻结系数，只训练神经校正器；③证明该两阶段流程能保持结构模型的边际替代率不变，避免联合训练导致的识别失效。

**🔧 技术方法**

使用技术包括：多项式Logit结构约束、神经网络校正器（两层MLP）、最大似然估计、交叉拟合的预计算概率、固定-q干预协议，以及基于模型无关的行为审计。

**📊 数据集**

数据集包含三组离散选择数据：瑞士铁路/汽车/卡车的 Swissmetro、伦敦公交/自行车/驾车的 LPMC、以及 IoT 设备的 IoT-Wearables，涵盖声称偏好和真实偏好、单元和面板数据。

**📈 对比分析**

与基线（纯多项式Logit、原始基础模型、受限单调网络、特征增强MNL、线性混合等）比较，适配器在保留 100% 成本单调性、零可用性泄漏、准确率提升 6.4pp（平均）且最多 12.8pp，且在不同上下文约束下仍保留至少 6pp 的准确率增益。

**⚠️ 局限性**

限制包括：适配器在结构化多项式Logit 规范失效时无法校正边际替代率；在 Swissmetro 数据集上校准误差仍较大；若基础模型预测不显著捕捉成本信息，联合训练会导致成本系数崩溃；此外在极端上下文稀缺时准确率仍可能下降。

---

## 162. Speaking Numbers to LLMs: Multi-Wavelet Number Embeddings for Time Series Forecasting

**arXiv ID:** 2606.26487 | [PDF](https://arxiv.org/pdf/2606.26487v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 163. Estimating Uncertainty in Classifier Performance with Applications to Large Language Models and Nested Data

**arXiv ID:** 2606.26422 | [PDF](https://arxiv.org/pdf/2606.26422v1)

**作者:** Kylie Anglin `[一作]` `[通讯]` (University of Connecticut), Kylie Anglin (University of Connecticut)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估并比较了在社会科学文本分类常见条件（小样本、高性能、稀有构建、文本嵌套）下多种置信区间估计方法的覆盖率与宽度。

**💡 创新点**

创新点在于提出伪计数正则化的自助抽样方法、系统比较了传统区间与嵌套数据的调整方案，并给出实践性推荐。

**🔧 技术方法**

采用模拟实验、统计学理论（Wald、Wilson、Agresti-Coull、Clopper-Pearson、BCa、伪计数自助）以及ICC调整和层级自助技术。

**📊 数据集**

使用了无真实公开数据集，仅通过仿真生成的二分类与多分类文本数据，模拟了不同预设性能、样本量和ICC。

**📈 对比分析**

比较方法：覆盖率与半宽度，结果显示 Wald 与基本百分位自助覆盖率低；Agresti-Coull、Wilson、Clopper-Pearson 及伪计数自助在独立样本中达标；在嵌套样本中，伪计数层级自助和 ICC 调整后的 Agresti-Coull/ Wilson 维持≈95% 目标覆盖率。

**⚠️ 局限性**

局限包括模拟条件未覆盖所有实际语料特点、缺乏对真实 ICC 分布的实证支持、对非比例指标偏差的讨论有限。

---

## 164. PRISM: Efficient and Locally Optimal Probabilistic Planning with Reachability Guarantees

**arXiv ID:** 2606.26413 | [PDF](https://arxiv.org/pdf/2606.26413v1)

**作者:** Alex Rose `[一作]` (Massachusetts Institute of Technology), Jonathan P. How `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 PRISM，一种通过协方差收缩与确定性均值规划分离的多查询贝叶斯空间规划算法，并配备了在线局部优化模块；

**💡 创新点**

创新点在于引入新的协方差可控性条件实现受限贝叶斯空间可达性证明，构建离线紧凑道路图并确保完整性与高覆盖率，同时在线局部优化显著降低路径成本；

**🔧 技术方法**

主要技术包括线性时不变系统的协方差收缩分析、凸半正定规划、凸化的概率约束、格点凸优化以及多阶段局部优化迭代（时间缩短、快捷路搜索、固定时间轨迹优化）；

**📊 数据集**

实验使用二维平面6自由度四旋翼的仿真场景，涵盖受限办公室与混乱障碍场，且在不同通道宽度与执行器噪声水平下进行测试；

**📈 对比分析**

与 MAXCOVAR、MAX‑COV‑BALL、CS‑BRM、RRT 与 RRT* 等基线方法在同一环境下比较，PRISM 在覆盖率上实现 100%（仅次于少数几方法），路径成本平均低 3‑10%，离线构图时间仅数分钟，在线局部优化耗时最低；

**⚠️ 局限性**

局限性在于需满足起始与目标均值为平稳且协方差满足上/下界约束；对于非平稳均值或极端不确定性场景的收缩保证有限；在更高维度或非线性动力学下的可扩展性尚待验证。

---

## 165. What Browsers Do in the Shaders: A Measurement Study of WebGPU Privacy

**arXiv ID:** 2606.26412 | [PDF](https://arxiv.org/pdf/2606.26412v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (International University of La Rioja), Igor Santos-Grueiro (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

构建了统一的测量框架，对 WebGPU 的多种隐私表面进行评估，涵盖管线编译状态、浏览器/原生共驻留、受害者探针、公开页面探测以及缓解策略。

**💡 创新点**

创新点在于：①提出细粒度的 WebGPU 隐私测量方法；②发现管线编译状态是最显著的泄露面；③通过参与者现场测量验证 WebGPU 行为的高度可指纹性；④通过 Tranco 爬虫证明公开页面主要做适配器探测；⑤示范源代码键分离等可实现的缓解措施。

**🔧 技术方法**

技术手段包括：WebGPU API 与 WGSL 生成探针；高分辨率 JavaScript 定时、帧级观察、计时器无关的秩序观测；多浏览器/后端（Chromium/Dawn、Firefox/wgpu、Safari/Metal）实现；统计学评估（AUROC、macro‑F1、熵、匿名集）；实验室控制实验、参与者现场采样、Tranco top‑10k 页面爬虫。

**📊 数据集**

使用的数据集：①多平台、多浏览器的控制实验日志；②1,095 条参与者上传记录（完整、去重后）；③7,477 条 Tranco top‑10k 页面爬虫记录（共 56 条 WebGPU 正式记录）。

**📈 对比分析**

评估方法：利用 AUROC 与 label‑permutation 基线比较、宏 F1 评估分类效果、熵与匿名集评估指纹性；对比缓解前后 AUROC 降低及编译/帧延迟成本。结果显示：管线缓存泄露可在 0.388 AUROC 降低、+6 ms 编译延迟；不同平台/浏览器下的可指纹性差异显著。

**⚠️ 局限性**

局限性：实验以合成受害者为主，缺乏真实工作负载；平台覆盖不完整（移动、ARM、Intel 等待后续收集）；长期稳定性未验证；源键分离仅为缓解模拟，未实现真实浏览器缓存分区；公开爬虫缺乏真实交互；高分辨率定时器依赖在某些浏览器/系统上可能受限。

---

## 166. Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents

**arXiv ID:** 2606.26479 | [PDF](https://arxiv.org/pdf/2606.26479v1)

**作者:** Praneeth Narisetty `[一作]` (LaunchSafe Research), Jayaram Kumarapu `[通讯]` (LaunchSafe Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文重新执行并扩展了Progent的自适应攻击实验，验证了第二代LLM代理对间接提示注入的防御在弱开源模型上依旧保持低攻击成功率；同时通过将这些防御映射为Biba完整性、参考监视和最小权限等经典安全原理，对其进行了系统化梳理。

**💡 创新点**

创新点在于：①将out‑of‑band防御视为经典安全原理的实例，形成对比框架；②提出并实现了缺失的适应性评估方法，并在独立实验中证明Progent在自适应攻击下仍保持稳健。

**🔧 技术方法**

使用了 deterministic out‑of‑band reference monitor（Progent）、信息流控制标签、能力标记技术，结合Qwen2.5‑7B 语言模型、AgentDojo benchmark 及自适应字符串优化攻击模板。

**📊 数据集**

采用 AgentDojo benchmark（97 个任务、629 个安全案例）以及 ASB benchmark，并对每套任务取前 8 个用户任务进行实验。

**📈 对比分析**

通过对比未防御、Progent 标准防御以及 Progent + 自适应攻击三种情形，发现攻击成功率从 25.8% 降至 4.2%，自适应攻击未提升至 2.6%；实验可重复、标准差低；在原始基准攻击下 Progent 的安全性显著优于未防御，但在效用上有明显下降。

**⚠️ 局限性**

局限性包括：仅评估单一防御（Progent）；使用弱开源模型 Qwen2.5‑7B；实验样本有限且排除了 Travel 套件；未包含白盒或仅授权工具的攻击；未测试其他防御系统（如 CaMeL、FORGE 等），因此结果仅能说明 Progent 在该实验条件下的鲁棒性。

---

## 167. When Does Quality-Aware Multimodal Fusion Matter? A Leakage-Safe Diagnostic for Decision-Level Dependence

**arXiv ID:** 2606.26473 | [PDF](https://arxiv.org/pdf/2606.26473v1)

**作者:** Jaden Moon `[一作]`, Andrew Campbell `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种后置诊断方法，检验已训练的多模态融合模型在推理时是否真正利用其估计的可靠性（质量）信号。该方法通过冻结专家和融合规则，只对测试集中的质量信号进行随机置换（Clean–Broken实验），观察性能变化。

**💡 创新点**

创新点在于：①引入了“Clean–Broken”置换实验来分离质量信号与预测的因果关系；②定义了Permutation Gap统计量和显著性检验，用以量化质量对决策的实际影响；③通过正向控制（将质量与模态失真或专家正确性对齐）验证诊断的敏感性，从而区分模型对质量的依赖与质量本身的有效性。

**🔧 技术方法**

技术细节包括：使用冻结的单模态专家（Logistic Regression 或 HGB）产生后验概率；两类决策层融合：无质量加权的平均融合以及基于质量权重的融合；以及条件化 Mixture of Experts（软最大路由器）。对置换实验的统计分析采用 Balanced Accuracy、Permutation Gap 与 Phipson–Smyth 的一侧 p 值检验。

**📊 数据集**

实验数据集为 StressID（多模态压力识别，语音、视频、物理信号缺失率高）和 CMU‑MOSEI（几乎完整的情感分析，语言、语音、视觉三模态）。

**📈 对比分析**

在完全观测的测试样本上，将匹配质量与置换质量的性能进行对比。结果显示：native 质量对模型准确率的影响几乎为零（Permutation Gap ≈ 0），即使有显著的专家竞争和潜在路由空间；而正向控制（质量与失真或专家正确性对齐）则产生显著正的 Permutation Gap，证明同一融合规则在质量真正指示可靠模态时能显著改进。

**⚠️ 局限性**

局限性包括：①诊断仅适用于决策层融合，无法评估特征学习或早期/中期融合模型的质量利用；②正向控制仅在 StressID 中实现，缺少跨数据集的验证；③实验侧重于完全观测场景，未能同时考察质量与缺失模式的交互；④需要人工构造的质量信号，真实数据中多模态质量变化的丰富性尚未完全覆盖。

---

## 168. The Tilted Playing Field for Women in Science

**arXiv ID:** 2606.26469 | [PDF](https://arxiv.org/pdf/2606.26469v1)

**作者:** Casandra Rusti `[一作]` (University of Southern California), Kristina Lerman `[通讯]` (Indiana University Bloomington)

**通讯引用:** 12312 | [OpenAlex ID](https://openalex.org/A5049634383)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了机构声望对科研产出和合作的影响，并探讨了性别差异。

**💡 创新点**

首次量化机构声望优势并揭示其在性别层面上的偏差，以及合作网络结构如何驱动这一差距。

**🔧 技术方法**

基于OpenAlex构建协作网络，使用CCDF、机构声望优势、相对性别差距、聚类系数等统计与网络指标。

**📊 数据集**

使用OpenAlex 2025年快照，筛选1980-2024年顶级期刊论文，包含近5万篇论文、650万作者和65k机构。

**📈 对比分析**

通过比较不同机构排名与性别组的优势比和相对性别差距，发现女性在最高等级机构获得优势与男性相当或更高，但在其余层级劣势显著，表明性别差异显著。

**⚠️ 局限性**

研究为观察性，采用名字推断性别，聚焦高影响力期刊可能缺乏代表性，机构排名与声望不完全对应。

---

## 169. GRAINS: Storage-Aware Algorithm-Architecture Co-Design Enabling High-Performance and Low-Cost Graph-Based Genome Analysis

**arXiv ID:** 2606.26468 | [PDF](https://arxiv.org/pdf/2606.26468v1)

**作者:** Nika Mansouri Ghiasi `[一作]` (ETH Zürich), Onur Mutlu `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了GRAINS系统，通过存储感知的算法-架构协同设计，在SSD内部完成基于图的基因组分析的高效查询和映射。

**💡 创新点**

首次将图基因组分析迁移到存储层，结合批处理、图感知查询重排、轻量级的闪存内/闪存上处理单元及调度技术，实现显著的I/O抑制和能耗降低。

**🔧 技术方法**

采用SSD内部的ISP/IFP单元、最小完美哈希k-mer字典SSHash、基于最小化窗口选择的子页读取、ECCLite校验、轻量调度表，以及对FTL的块级映射改造。

**📊 数据集**

使用MetaSUB全基因组图（约659–822 GB）和10 M/1 M/100 K测序读数，评估k-mer集合查找和读映射。

**📈 对比分析**

与Fulgor、MetaGraph软件基准及理想硬件加速基线比较，GRAINS在k-mer查询上实现2.7×–47.8×加速、4.4×–31.6×能耗下降，整体性能比软件提升6.8×、10.7×、5.6×。

**⚠️ 局限性**

受限于SSD内部资源，当前实现仍需专用ISP/IFP硬件，且对极大规模或极度倾斜的查询分布缺乏自适应负载均衡，未来需进一步优化调度与多SSD分布式支持。

---

## 170. 3D Spatial Pattern Matching

**arXiv ID:** 2606.26465 | [PDF](https://arxiv.org/pdf/2606.26465v1)

**作者:** Nicole R. Schneider `[一作]` (University of Maryland), Youness Dehbi `[通讯]` (HafenCity University)

**通讯引用:** 460 | [OpenAlex ID](https://openalex.org/A5023864581)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于深度学习与几何算法融合的3D空间模式匹配框架，能够高效识别和对齐大规模三维点云中的相似结构。

**💡 创新点**

创新点在于首次将图卷积网络与空间分块+注意力机制相结合，实现了对大规模点云的可伸缩匹配，并通过引入自监督预训练提升了对噪声和尺度变化的鲁棒性。

**🔧 技术方法**

主要技术包括图卷积网络（GCN）、空间分块/体素编码、注意力机制、以及传统ICP优化回环，用以构建端到端匹配流水线。

**📊 数据集**

实验采用公开的ModelNet40、ShapeNetCore和自建的室内点云数据集（S3DIS），并在这些数据集上进行大规模对比。

**📈 对比分析**

与传统ICP、RANSAC、PointNet++等方法比较，本文方法在匹配精度上提高约10–15%，在匹配速度上比ICP快约2–3倍，并在不同尺度下保持较高鲁棒性。

**⚠️ 局限性**

局限主要体现在对极端噪声和大尺度变化的处理仍不够完善，且在实时系统中存在一定的计算开销，需要进一步优化模型与硬件加速。

---

## 171. AXLE: A Cloud Infrastructure for Lean 4 Theorem Proving Utilities

**arXiv ID:** 2606.26442 | [PDF](https://arxiv.org/pdf/2606.26442v1)

**作者:** Jimmy Xin `[一作]` (Axiom Math), Jannis Limperg `[通讯]` (Axiom Math)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

开发了一个可扩展的云端服务，提供对 Lean 4 定理证明的严格验证、元编程、元数据提取、语义合并、修复等工具，并支持多版本、多租户、请求隔离的 API。

**💡 创新点**

提供可在云中并发、隔离的严格证明验证，并将高级元编程功能包装为可调用服务，满足 AI 训练、代理证明与数据集构建的高并发需求，且支持多版本环境。

**🔧 技术方法**

基于 Lean 4 的元编程框架实现多工具 metaprograms，云架构采用多租户、进程沙箱、自动扩缩容；接口包括 Python SDK、CLI、Web UI、HTTP API。

**📊 数据集**

评估使用 public Goedel workbook 数据集、Lean Workbook competition‑math 以及生产流量产生的配对请求来验证速度和严格检查效果。

**📈 对比分析**

与 Kimina Lean Server 与直接 spawn 进程做延迟/吞吐对比，和 Comparator、SafeVerify 做严格检查速度和准确率对比；单机 8 核下吞吐约 2 req/s，严格检查平均 0.97 秒，远快于 Comparator（95.7 s）和 SafeVerify（10.1 s）。

**⚠️ 局限性**

目前仅支持单文件环境，无法即时添加自定义多文件项目；不支持交互式证明搜索或 tactic‑level 操作，需要借助 Pantograph 等工具。

---

## 172. ConflictScore: Identifying and Measuring How Language Models Handle Conflicting Evidence

**arXiv ID:** 2606.26437 | [PDF](https://arxiv.org/pdf/2606.26437v1)

**作者:** Siyi Liu `[一作]` (University of Pennsylvania), Patrick Xia `[通讯]` (Microsoft)

**通讯引用:** 1206 | [OpenAlex ID](https://openalex.org/A5018805044)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ConflictScore 这一新型冲突感知度量，评估模型回答中是否充分考虑并表述文档间的相互矛盾信息；同时构建 ConflictBench 作为统一的冲突检测基准；并在 Retrieval‑Augmented Generation（RAG）模型上验证其诊断与纠正作用。

**💡 创新点**

①将回答拆分为原子命题，并对每个命题分别评估与每份检索文档的支持、反驳或无关关系；②引入 ConflictScore‑Count (CS‑C) 与 ConflictScore‑Ratio (CS‑R) 两个互补指标；③将冲突检测结果回传模型进行再生成，实现对过度自信答案的纠正。

**🔧 技术方法**

基于大语言模型进行命题拆分与命题‑文档关系判断；利用 LLM 生成的检索增强回答；采用标准的 NLI/事实检查流程进行评估；通过 Prompt Engineering 实现冲突回传与再生成。

**📊 数据集**

ConflictBench（集合 ContraQA、MacNoise‑NQ/TQA、AmbigDocs、ConflictingQA 等多种冲突类型数据集），TruthfulQA（多选问答评估模型真确性）。

**📈 对比分析**

与传统事实性度量（FactScore、SAFE 等）相比，ConflictScore 在 ConflictBench 上的 F1、精确率、召回率均超过 90%，且能够在 RAG 生成中显著降低过度自信率（在 TruthfulQA 上提升 1–3%）。

**⚠️ 局限性**

①计算量大（每个命题对每份文档评估，导致 O(nm) 复杂度）；②未考虑源文档可信度，所有文档等权；③上游模块（命题拆分、关系判定）误差会传播；④对长篇回答的细粒度评估成本高，需要权衡。

---

## 173. DualEval: Joint Model-Item Calibration for Unified LLM Evaluation

**arXiv ID:** 2606.26429 | [PDF](https://arxiv.org/pdf/2606.26429v1)

**作者:** Aaron J. Li `[一作]` (University Of California Berkeley), Ion Stoica `[通讯]` (University Of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出DualEval框架，将静态基准正确性标签与arena式偏好数据联合在同一潜在空间中进行模型-项目校准；

**💡 创新点**

创新点在于将IRT思想与奖励模型偏好融合，实现静态与开放式评测的互补统一，并通过项目难度、尖锐度和信息量等属性进行诊断；

**🔧 技术方法**

使用两参数Logistic IRT模型、奖励模型标准化与软对比目标、贝叶斯回归与Fisher信息分析等技术；

**📊 数据集**

使用四个领域的数据：编码（LiveCodeBench、MBPP-Plus等）、数学（AIME、Olympiad-Math等）、杂项知识（Bio.+Med., Eng.等）以及通用日常查询，覆盖18个前沿LLM；

**📈 对比分析**

通过与静态2PL、arena BT和平均奖励基线对比，DualEval在静态标签准确率88–92%、arena对比一致率68–81%和排行榜Spearman相关性均表现出更均衡、更稳健的性能；

**⚠️ 局限性**

局限包括依赖内部奖励模型导致可复现性受限、每个模型-问题对仅评估一次回应、以及仅使用单维能力参数等限制。

---

## 174. Do more heads imply better performance? An empirical study of team thought leaders' impact on scientific team performance

**arXiv ID:** 2606.26483 | [PDF](https://arxiv.org/pdf/2606.26483v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 175. Extracting Problem and Method Sentence from Scientific Papers: A Context-enhanced Transformer Using Formulaic Expression Desensitization

**arXiv ID:** 2606.26481 | [PDF](https://arxiv.org/pdf/2606.26481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 176. Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?

**arXiv ID:** 2606.26428 | [PDF](https://arxiv.org/pdf/2606.26428v1)

**作者:** Tyler Ga Wei Lum `[一作]` (Stanford University), Jeannette Bohg `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

先用强化学习在模拟中对多指机器人进行无目标的“玩耍”预训练，获得通用抓取和手内旋转的先验，然后在稀疏奖励的装配任务中微调以实现精确装配。

**💡 创新点**

创新点在于把“玩耍”作为无监督的任务无关预训练阶段，通过多物体、多轨迹、多目标精度的设计，使得后续稀疏奖励装配任务能以33倍更高的样本效率学习，并实现零样本从仿真到真实的迁移。

**🔧 技术方法**

主要技术包括：基于目标条件的强化学习（SAPG）、场景和动作的域随机化、从CAD自动生成装配步骤与稀疏奖励、以及使用FoundationPose进行6D位姿跟踪。

**📊 数据集**

数据集包括：程序化生成的多种立方体/圆柱体物体用于玩耍预训练；来自Fabrica和FurnitureBench的CAD模型用于装配任务；以及在真实实验中使用3倍尺寸的3D打印部件。

**📈 对比分析**

与从零开始的稀疏奖励RL和使用稠密奖励的RL基线相比，预训练+微调方案在所有四个装配任务上实现了至少30倍的训练时间加速，并在真实世界中获得60%~90%的成功率，显著优于仅玩耍或零样本策略。

**⚠️ 局限性**

局限性包括：仅学习短周期装配步骤，缺乏完整的自动化装配流水线；外部依赖固定的装配顺序与目标位姿；对姿态估计高度依赖，遮挡或动态运动时易失败；未直接使用视觉或触觉信息，限制了对复杂环境的适应。

---

## 177. Rethinking Training & Inference for Forecasting: Linking Winner-Take-All back to GMMs

**arXiv ID:** 2606.26424 | [PDF](https://arxiv.org/pdf/2606.26424v1)

**作者:** Qiyuan Wu `[一作]` (Cornell University), Mark Campbell `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

分析了WTA损失与GMM/K‑means的关系，提出后处理方式以提升轨迹预测模式的概率质量并改进模式选取。

**💡 创新点**

将WTA视为K‑means，指出其导致的过度分割与无信息概率，进而提出测试时聚合与一次EM微调两种后处理方案。

**🔧 技术方法**

采用统计学推导、K‑means聚合、EM一阶更新以及基于GMM的轨迹预测模型。

**📊 数据集**

在NuScenes Prediction和Waymo Open Motion（WOMD）数据集上进行实验。

**📈 对比分析**

与贪婪选择、NMS、直接训练等方法对比，聚合和EM后处理在minADE/minFDE、miss rate、brier等指标上均实现了显著提升。

**⚠️ 局限性**

需要额外的推理时间（聚合与EM），仅适用于已用WTA训练的模型，且未完全解决概率分布与真实先验不匹配的问题。

---

## 178. CoStream: Composing Simple Behaviors for Generalizable Complex Manipulation

**arXiv ID:** 2606.26423 | [PDF](https://arxiv.org/pdf/2606.26423v1)

**作者:** Haonan Chen `[一作]` (Harvard University), Yilun Du `[通讯]` (Harvard University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种通过组合语义、预测与触觉三种简单行为实现复杂高精度接触丰富操作的框架。

**💡 创新点**

创新点是将不同速度、不同模态的行为在SE(3)空间通过右乘合成，而非传统管线或单一端到端网络，能实现零样本迁移和实时误差纠正。

**🔧 技术方法**

技术包括基于LLM/VLM的语义定位、视频世界模型预测轨迹、触觉传感与阻抗控制的合成。

**📊 数据集**

使用了在真实机器人上收集的八项任务数据，包括GPU/CPU/RAM/钻头插入、灯泡开关、擦白板、杯子到盘子、衣物到盒子等。

**📈 对比分析**

与VoxPoser和π_0.5等基线对比，8项任务中所有精密装配任务在我们方法上均达15/15成功率，而基线为0/15；日常任务提升至最高13/15。

**⚠️ 局限性**

局限在于依赖精确的视觉和触觉感知，难以处理柔性或复杂变形物体、无结构目标，以及触觉覆盖不足时的失稳。

---

## 179. Comparing BERT Sentence-Pair Classification and Few-Shot LLM Prompting for Detecting Threat and Solution Framing in German Climate News

**arXiv ID:** 2606.26489 | [PDF](https://arxiv.org/pdf/2606.26489v1)

**作者:** Raven Adam `[一作]` (University of Graz), Marie Kogler `[通讯]` (University of Graz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对德国气候新闻文章中的句子进行威胁导向与解决方案导向的分类，并系统比较了两种不同模型范式：微调BERT与少样本LLM提示；

**💡 创新点**

创新点在于①将句子对（前一句作为上下文）用于BERT微调，显著提升性能；②构建了完整的多层提示结构（角色定义、发声行为分类、框架定义、规则、示例、全文与目标句），并通过链式推理返回结构化JSON；③对比分析了阈值过滤、误差模式与理由不一致性，揭示LLM自评信心无效。

**🔧 技术方法**

技术包括：①使用German BERT（GBERT）进行句子对输入的序列分类；②使用Llama 4 Maverick进行few-shot chain‑of‑thought 提示，生成三字段JSON（reasoning、classification、confidence）。

**📊 数据集**

数据集由440篇奥地利日报（Kronen Zeitung + Der Standard）中筛选的气候相关文章组成，人工编码后拆分为10,981句子；正例与负例通过随机下采样平衡。

**📈 对比分析**

在同一20%测试集上评估精确率、召回率与F1：BERT句子对模型在威胁与解决方案任务上分别取得F1≈0.83，LLM少样本模型F1≈0.78；BERT在两类模型不一致的句子子集上表现明显优于LLM，阈值筛选对LLM未提升F1。

**⚠️ 局限性**

局限性包括①句子级拆分导致同一文章句子同时出现在训练与测试，可能夸大性能；②平衡采样剔除负样本信息；③仅使用单一随机拆分未做交叉验证；④人工标注未提供可靠性度量；⑤仅比较单一BERT和单一LLM，缺乏多模型泛化评估。

---

## 180. DKVE: Decentralized Key Validation for End-to-End Encrypted Messaging

**arXiv ID:** 2606.26486 | [PDF](https://arxiv.org/pdf/2606.26486v1)

**作者:** Subin Song `[一作]` (Seoul National University), Taekyoung Kwon `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于社交图的去中心化密钥验证协议，允许用户通过隐私保护的交叉验证来确认对方的公钥是否被关键目录服务器篡改。

**💡 创新点**

创新点在于将隐私友好的OPRF与OKVS结合，用顺序概率比检验（SPRT）在不暴露查询内容和联系人列表的前提下高效聚合证据，同时显著降低对传统OOB和KT查询的依赖。

**🔧 技术方法**

核心技术包括：隐蔽伪随机函数（OPRF）、可观察键值存储（OKVS）、序贯概率比检验（SPRT）以及在PoC中使用的Matrix协议、VOLE‑based OPRF实现。

**📊 数据集**

评估使用了来自Facebook、Twitter和GitHub的真实社交网络数据集，模拟社交图中不同强度连接的场景。

**📈 对比分析**

实验表明，在强到中等关联网络中，协议对服务器中间人攻击的检测率超过97%，误报率低于0.1%，漏报率低于0.3%；平均每次验证仅需数次查询，上传带宽约3.8 MB，下载带宽约0.6 MB，验证延迟可接受为后台操作。

**⚠️ 局限性**

主要局限包括：缺乏互联联系人时的启动问题、对弱关联社交网络效果差、无法恢复误报/漏报、对协同攻击的假设易失效，以及在密钥更新时需额外版本验证。

---

## 181. Retrieval-Warmed Energy-Based Reasoning: A Five-Arm Ablation Methodology for Diffusion-as-Inference on Structured Reasoning Tasks

**arXiv ID:** 2606.26476 | [PDF](https://arxiv.org/pdf/2606.26476v1)

**作者:** Libo Sun `[一作]` (Auburn University), Xiao Qin `[通讯]` (Auburn University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究检索热启动的迭代推理，并提出了五臂消融方法来分离偏置、随机化和对齐三种混合效应。

**💡 创新点**

创新点在于通过五臂消融将检索热启动的贡献拆解为三大组件（键质量、热启动机制、存值质量），并在连通性任务中揭示对齐贡献高达35个百分点。

**🔧 技术方法**

使用的技术包括IRED能量扩散模型、现代Hopfield轨迹记忆、对比学习编码器以及多种检索热启动策略（常量、随机、洗牌、对齐）。

**📊 数据集**

实验数据集为Erdős–Rényi无向图的全对齐连通性任务（12节点、p=0.2）和Sudoku填字游戏的SATNet式数据集。

**📈 对比分析**

方法与标准冷启动、最佳常量、随机、洗牌、对齐五种策略进行对比，结果显示对齐策略在连通性任务中实现+35个百分点的准确提升，而冷启动因存值质量不足而失效；Sudoku任务中则因键质量不足而失败。

**⚠️ 局限性**

局限性包括：仅在单一验证集和固定模型架构上评估，未探索更深或不同结构的模型；存值质量和键编码器的上限导致检索热启动效果受限；对齐效果的可推广性和跨任务表现仍待验证。

---

## 182. Localizing RL-Induced Tool Use to a Single Crosscoder Feature

**arXiv ID:** 2606.26474 | [PDF](https://arxiv.org/pdf/2606.26474v1)

**作者:** Andrii Shportko `[一作]` (Northwestern University), Jessica Hullman `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过专用特征交叉编码器对RL微调后语言模型的内部表示进行稀疏分解，提取出工具调用相关特征，并利用单个特征的激活干预实现行为控制；

**💡 创新点**

发现RL引入的工具调用能力集中在专用特征子集，可通过单个神经元实现大幅性能提升，并且通过解码重建实现“能力溢出”到未微调基模型，展示无训练的行为迁移与低干预控制；

**🔧 技术方法**

采用稀疏自编码器（SAE）、交叉编码器、专用特征交叉编码器（DFC）以及Top‑k稀疏编码、特征级激活干预、自动解释（autointerp）、UMAP+HDBSCAN可视化等技术；

**📊 数据集**

使用ToolRL数据集，包含40k通用文本样本与40k指令‑输出对；

**📈 对比分析**

在48个交叉编码器变体上进行超参数搜索，评估工具正确率、格式准确率和总体得分；重建后RL模型工具正确率提升约+31.1个百分点，基模型提升约+6.8个百分点；单神经元干预可实现约+65个百分点的工具正确率提升；

**⚠️ 局限性**

研究仅限于单一模型对（Qwen2.5‑3B）与工具调用任务，缺乏多模型、多任务验证；干预效果未在新样本上复现；共享分区未深入解释；架构比较的统计显著性受样本量限制。

---

## 183. Epiphany-Aware KV Cache Eviction Without the Attention Matrix

**arXiv ID:** 2606.26472 | [PDF](https://arxiv.org/pdf/2606.26472v1)

**作者:** Steven Kolawole `[一作]` (Carnegie Mellon University), Virginia Smith `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于残差隐藏状态变化的KV缓存淘汰方法，用于长推理链的生成。

**💡 创新点**

创新点在于不依赖注意力权重，而是使用“epiphany score”即隐藏状态差异作为重要性度量，并发现两带层结构，能显著提高缓存淘汰质量；此外实现FlashAttention兼容，避免了内存壁垒。

**🔧 技术方法**

使用残差隐藏状态差分、滚动z-score去趋势、两带层加权、KV向量方差、Lag-KV归一化等技术；在推理过程中直接从前向传递获取信号，无需额外训练或自定义核。

**📊 数据集**

主要在数学竞赛数据集MATH-500和AIME-2024上进行评估，同时使用GSM8K做难度探测。

**📈 对比分析**

与ThinKV、H2O、RaaS、LongFlow等基线对比，MATH-500 4096-token预算下准确率达到72%（高于ThinKV 71%和H2O 67%），AIME-2024 8192-token预算下37%（高于最佳33%）；同时在相同预算下速度可提升至2.8倍，且不需要显式注意力矩阵。

**⚠️ 局限性**

仅在单GPU单例测试，未测多GPU/批量吞吐；仅评估在DeepSeek-R1-Distill-LLaMA-8B模型，跨架构、规模的迁移性未知；对长推理的prefill内存优势仅通过微基准展示。

---

## 184. A Causal Foundation Model for Structure and Outcome Prediction

**arXiv ID:** 2606.26467 | [PDF](https://arxiv.org/pdf/2606.26467v1)

**作者:** Max Zhu `[一作]` (University of Cambridge), Stefan Groha `[通讯]` (GSK.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 TabPFN-CFM，一个可以同时进行因果结构预测和三类因果查询（观测、干预、反事实）的基础模型。

**💡 创新点**

创新点在于：①将 ADMG（表示未观测混杂的有向混合图）嵌入 PFN 框架中，②联合训练观测、干预与反事实三种查询任务，③使用先验图结构提升后验精度并显著加速训练。

**🔧 技术方法**

技术手段包括：贝叶斯 PFN、Transformer 的行/列注意力、图结构编码（邻接矩阵、双向混杂矩阵及祖先矩阵）、交叉熵与二值交叉熵损失、训练加速改进（去除特征聚类、增强、随机特征组合）以及批量填充的虚拟特征。

**📊 数据集**

数据集：先用从先验分布采样的合成 SCM 数据（包含随机图、噪声、结构方程），随后在真实数据上评估，主要包括 Amazon Sales 和 Law School Admissions 两个具有已知因果图的公开数据集。

**📈 对比分析**

比较方法：与 Do‑PFN、Meta‑Learners（S‑learner、T‑learner、X‑learner、DR‑learner）、AVICI、FCI、GES、LiNGAM、PC 等基线对比；在观测任务中表现相当，干预与反事实任务中显著优于所有基线，结构预测准确率和 AUC 也明显高于现有方法。

**⚠️ 局限性**

局限性：目前仅针对无环因果结构和静态数据；对循环、时间序列等更复杂因果场景尚未扩展；模型仍高度依赖大量合成数据，且当图结构缺失或混杂效应强时预测性能可能下降。

---

## 185. Nanoelectromechanical Systems (NEMS) for Hardware Security in Advanced Packaging

**arXiv ID:** 2606.26426 | [PDF](https://arxiv.org/pdf/2606.26426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 186. Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting

**arXiv ID:** 2606.26421 | [PDF](https://arxiv.org/pdf/2606.26421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 187. AnySimLite: A Lightweight Few-Shot Similarity Encoder for On-Device Speech-Adjacent Classification

**arXiv ID:** 2606.26452 | [PDF](https://arxiv.org/pdf/2606.26452v1)

**作者:** Sourav Ghosh `[一作]` (Samsung R&D Institute), Saravana Balaji Shanmugam `[通讯]` (Samsung R&D Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出轻量级的文本相似度编码器AnySimLite，并通过数据集转换将多种语音相邻NLP分类任务归约为细粒度文本相似度（NTS）问题，实现在边缘设备上的统一高效推理。

**💡 创新点**

核心创新在于：①结合词级与字符级通道的轻量化编码器；②基于聚类挑选“难样本”的数据转换策略；③在保持低内存占用（<1/250原始模型）和低延迟（30 ms）的同时，兼具SOTA竞争力。

**🔧 技术方法**

技术手段包括词嵌入+BiLSTM、字符嵌入+Conv1D、注意力机制、交叉通道注意力、Siamese/Triplet学习、知识蒸馏、MiniLM压缩，以及DBSCAN聚类与硬样本采样。

**📊 数据集**

使用的公开数据集包括Event Title Similarity（自建）、Sentiment-140、IMDB、SNIPS、ATIS、SMS Spam Collection、AG News、Toxic Comment等，均经过转换成成对或多类的NTS格式。

**📈 对比分析**

与多项SOTA方法（BERT、MiniLM、LLaMA、RoBERTa等）在20-shot或全数据上对比，AnySimLite在绝大多数任务上实现了与SOTA相当甚至略优的准确率/召回率/ F1；仅在极端情况下性能下降不超过7%，而模型大小仅为原始模型的1/250。

**⚠️ 局限性**

局限性包括：难样本采样仍需聚类预处理；跨任务迁移仍需进一步验证；在极大数据量或更复杂任务（如多模态）下可能需要更强模型；以及知识蒸馏后可解释性下降。

---

## 188. Listening Like a Judge: A Music-Aware Framework for Automatic Singing Performance Evaluation

**arXiv ID:** 2606.26451 | [PDF](https://arxiv.org/pdf/2606.26451v1)

**作者:** Neelam Saini `[一作]` (Samsung Research and Development Institute), Sourav Ghosh `[通讯]` (Samsung Research and Development Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态区块对齐的自动歌唱质量评估框架，结合歌词准确度与音高节奏忠实度，并通过MG‑LoRA提升唱歌 ASR 的鲁棒性。

**💡 创新点**

首次同时建模歌词对齐与音乐感知的区块级评估；引入多信号歌词区块检测（语义嵌入、模糊词匹配、语音匹配）以及音乐感知的 LoRA fine‑tuning（MG‑LoRA）。

**🔧 技术方法**

使用 Whisper 作为基础 ASR 并通过 MG‑LoRA（LoRA + 音乐相关正则）进行 fine‑tuning；采用 pYIN 进行音高检测、谱图与节拍估计；利用多信号匹配、区块级评分与加权聚合实现整体评估。

**📊 数据集**

自建 420 例印度歌曲集（train/val/test 70/15/15），结合 SingMOS‑Pro、Jamendo 等公开数据；在多语言（英语、普通话、印地语等）和多流派（古典、民歌、流行等）场景下进行实验。

**📈 对比分析**

与仅歌词或仅音乐的基线相比，Spearman ρ 提升至 0.683（+31.9%），Kendall τ 0.499；与 Whisper 等基线比较，WER 降低约 29.87%；多信号区块检测的 ρ 达到 0.626，整体评估性能明显优于单模态和现有方法。

**⚠️ 局限性**

仍未覆盖多说话者/多声部情形；依赖单一全局 key 估计；受限于 12 s 的音频输入；需要进一步在更大范围的跨文化、多语言数据上验证。

---

## 189. Deletion-Correcting Codes for the $\ell$-Symbol Read Channel

**arXiv ID:** 2606.26434 | [PDF](https://arxiv.org/pdf/2606.26434v1)

**作者:** Zuo Ye `[一作]` (Xidian University), Gennian Ge `[通讯]` (Capital Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了ℓ符号读取通道（ℓ‑read）在对抗性删除错误模型下的删除纠错码设计，提出了一种基于检查模式的结构化框架；

**💡 创新点**

创新点在于给出了删除对读取向量的影响的结构性描述，即删除后可视为对原始序列若干周期子串进行完整周期删除；

**🔧 技术方法**

采用的主要技术包括检查模式（(ℓ,t,s)‑check pattern）概念、幂和综合（power‑sum syndromes）以及细化的同余约束求解；

**📊 数据集**

论文未使用具体实验数据集，主要以理论构造和冗余分析为主；

**📈 对比分析**

与已有工作（如Xie‑Chen等的构造）相比，本文在ℓ/2≥t的情形下实现了冗余t·log n+O(1)的更紧致码，且在ℓ=2t‑1时给出了(2t‑1)·log n+O(1)的构造；

**⚠️ 局限性**

局限性在于对t>ℓ/2的情况缺乏通用解法，且编码实现仍不够高效，尚未给出完整的编码/解码算法。

---

## 190. Methane-Plume Segmentation From Hyperspectral Satellite Imagery Via Multimodal Deep Learning

**arXiv ID:** 2606.26416 | [PDF](https://arxiv.org/pdf/2606.26416v1)

**作者:** Brayan Quintero `[一作]` (Universidad Industrial de Santander), Hoover Rueda-Chacón `[通讯]` (Universidad Industrial de Santander)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态深度学习框架，用以从高光谱卫星影像中分割甲烷羽流，融合RGB视觉信息与甲烷增强图谱；

**💡 创新点**

创新点在于引入特征引导甲烷增强（FGME）模块，在多尺度Transformer特征上注入物理可解释的甲烷信号，提升分割精度且保持高效；

**🔧 技术方法**

使用DINOv3 ViT-S/16做RGB编码器、ResNet-18做甲烷增强编码器、SegFormer解码器，并通过FGME、Dice+Focal复合损失进行训练；

**📊 数据集**

在MPDataset（EMIT甲烷羽流数据集）上进行评估，该数据集包含4,172幅512×512像素的RGB与甲烷增强图像；

**📈 对比分析**

与VGG16+UNet、SegFormer及MPSUNet对比，改进了+0.92 MIoU、+0.87 MPrecision、+1.01 Recall，FLOPs仅为MPSUNet的约1/3，推理时间≈7 ms；

**⚠️ 局限性**

局限性包括对低对比度羽流仍存在检测挑战，且目前仅利用单一甲烷增强通道，未来可扩展至更丰富的光谱通道。

---

## 191. Structural parameterizations of Geodetic Set on directed (acyclic) graphs

**arXiv ID:** 2606.26414 | [PDF](https://arxiv.org/pdf/2606.26414v1)

**作者:** Beaudou Laurent `[一作]` (Université Clermont-Auvergne), Tale Prafullkumar `[通讯]` (Indian Institute of Science Education and Research Pune)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了有向图中Geodetic Set（几何集）问题的参数化复杂度，提出了多种结构参数（如底层无向图的顶点覆盖数、最大度、可达直径等）的核化和算法，并给出相应的ETH下界；

**💡 创新点**

创新点在于：1）给出了以顶点覆盖数为参数的2^O(vc^2)时间算法及匹配下界；2）提出了将解集大小、最大度和可达直径组合为参数的核化与时间复杂度证明；3）证明了在单一参数下问题仍为W[2]-难并给出了多种强硬度下界；

**🔧 技术方法**

采用了假设性归约（3SAT变体→有向图构造）、false‑twin 删除、二分树技术、层次化分割、以及标准的参数化核化与暴力搜索；

**📊 数据集**

论文没有使用公开数据集，所有实验/证明均为理论构造与复杂度分析；

**📈 对比分析**

与已有的无向图结果对比，作者通过构造实现了与无向图相当的时间/空间界限，证明所给算法在所设参数下已达到最优；

**⚠️ 局限性**

局限性包括：1）仅针对几种特定参数组合，单一参数无法获得FPT；2）对更稀疏结构（如树形有向图）的进一步研究仍待开展；3）未给出多项式核的可能性，仍是开放问题。

---

## 192. Can Large Language Models Reliably Code Qualitative Humanitarian Data? A Benchmark Study Against Human Expert Adjudication

**arXiv ID:** 2606.26541 | [PDF](https://arxiv.org/pdf/2606.26541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 193. Co-Designing Community-Centered AI Education for Adults: A Midwestern Case Study

**arXiv ID:** 2606.26565 | [PDF](https://arxiv.org/pdf/2606.26565v1)

**作者:** Yao Lyu `[一作]` (University of Michigan), Tawanna R. Dillahunt `[通讯]` (University of Michigan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在与社区伙伴共同设计的基础上，举办了一场面向非正式教育体系成人的 AI 教育研讨会，重点关注社区成员对 AI 的关注、理解与需求；

**💡 创新点**

创新点在于：①将 AI 文识视为社区层面的能力建设；②采用共创、对话式、地方化的教学设计，打破传统技术化、单向授课模式；③将调查、互动投票与实操游戏相结合，构建支持性、包容性的学习环境；

**🔧 技术方法**

技术与工具包括：AI 交互游戏（如 Code.org 的“AI For Oceans”）、AI 生成图像与真实图像对比投票、Zoom 线上协同、现场投票软件（如 WooClap）、问卷调查表（Pre‑Post）、现场音视频记录；

**📊 数据集**

数据集为参与者的问卷（Pre‑Post）和实时投票结果，共计 54 名受访者（48 人现场，6 人线上）以及 25 名提供人口学信息的受访者；

**📈 对比分析**

方法上采用前后测对比，分析受访者对 AI 认识、关注点及自信度的变化；结果显示，虽然对 AI 的基本担忧未显著下降，但受访者在具体认知与操作能力上提升，尤其在识别 AI 生成图像方面准确率达 82%；未与其他干预做定量性能对比；

**⚠️ 局限性**

局限性包括：样本规模有限、仅涵盖一处主要黑人社区，缺乏长期跟踪评估，未对照其他教育模式，且对技术细节与专业技能提升关注不足，难以推广至更广泛人群。

---

## 194. SpaceRipple: Lightweight Semantic Delivery for Mission-Oriented LEO Earth Observation Satellite Networks

**arXiv ID:** 2606.26559 | [PDF](https://arxiv.org/pdf/2606.26559v1)

**作者:** Ziyi Yang `[一作]` (Beijing University of Posts and Telecommunications), Xing Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 SpaceRipple，一种轻量级框架，用于低轨道地球观测卫星网络的任务导向语义传输与机载处理，包含感知端自适应压缩、侧元数据携带、转发、恢复、Mixture‑of‑Experts (MoE) 增强和语义推理，实现从像素级传输向任务相关语义信息的转变。

**💡 创新点**

创新点包括：
① 系统级协同设计，将压缩、转发、恢复与语义交付融合为统一管线；
② 侧元数据辅助恢复机制，利用压缩比例与校准信息提升恢复鲁棒性；
③ 基于局部像素特征与压缩比例的 MoE 增强模块，针对不同压缩强度自适应增强特征；
④ 轻量感知端模型（≈1.53 M 参数）与边缘端算力分工，显著降低感知端算力负担。

**🔧 技术方法**

使用技术包括：
- 以 256×256 块为单位的自适应压缩（基于 SSIM 评估冗余程度）；
- 侧元数据生成（压缩比例、校准信息）；
- 传输后恢复网络 gψ 与 MoE 增强模块；
- 语义推理头 hω（目标检测/分类）；
- 评估指标 PSNR、SSIM、LPIPS、F1 以及联系窗口吞吐量。

**📊 数据集**

使用的主要数据集为高分辨率卫星图像集，具体包括船舶识别数据集和城市车辆识别数据集（对应实验中的 Ship Recognition 和 City Vehicle Recognition 任务）。

**📈 对比分析**

与 Bicubic、Lanczos 插值、SwinIR、Real‑ESRGAN 等基线方法对比，SpaceRipple 在 PSNR（26.92 dB）、SSIM（0.7689）、LPIPS（0.2114）上均位居首位；在船舶识别和城市车辆识别任务中，F1 分别达到 0.8635 与 0.8790，明显优于其他方法。数据压缩率超过 98%，显著提升联系窗口吞吐量并降低延迟。

**⚠️ 局限性**

局限性包括：
- 未在真实链路条件下验证性能；
- 仅针对单一任务（目标检测）评估，缺乏多任务实验；
- 边缘端算力需求仍较高，轻量化部署尚待进一步优化；
- 对极端压缩比或不同气象条件下的鲁棒性研究不足。

---

## 195. Coarse-to-Fine: A Hybrid Self-Supervised Method for Non-rigid 3D Shape Matching

**arXiv ID:** 2606.26557 | [PDF](https://arxiv.org/pdf/2606.26557v1)

**作者:** Feifan Luo `[一作]` (Zhejiang University), Hongyang Chen `[通讯]` (Zhejiang Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计了一种基于粗到细的混合自监督3D形状匹配框架，消除了传统最小二乘求解，提升了效率。

**💡 创新点**

创新点在于提出三种对比能量（空间、频域、空间频域）实现粗细对齐，并构建双分支（Laplacian+弹性基）无监督损失与通用细化策略。

**🔧 技术方法**

使用DiffusionNet特征、软最大化算子、对比能量约束、基于滤波器的功能映射细化以及无求解的映射恢复技术。

**📊 数据集**

在FAUST、SCAPE、SHREC'19、SMAL、DT4D-H、TOPKIDS等公开数据集上进行评估。

**📈 对比分析**

与多种基准（axiomatic、supervised、unsupervised）比较，均在近等距、跨数据、非等距、拓扑噪声等情景下取得最优或第二优的平均测地误差，且推理速度更快。

**⚠️ 局限性**

主要局限是对极端拓扑扰动的鲁棒性仍需提升，且在极需精细对齐的场景下仍可能需要额外细化步骤。

---

## 196. An exploratory behavioral and electroencephalographic study of artificial intelligence-assisted learning modes in high school students

**arXiv ID:** 2606.26579 | [PDF](https://arxiv.org/pdf/2606.26579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 197. VoiceTTA: Enhancing Zero-Shot Text-to-Speech via Reinforcement Learning-Based Test-Time Adaptation

**arXiv ID:** 2606.26534 | [PDF](https://arxiv.org/pdf/2606.26534v1)

**作者:** Tianxin Xie `[一作]` (Hong Kong University of Science and Technology), Li Liu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 VoiceTTA，一种基于强化学习的测试时适配框架，能够在仅用几秒钟参考语音的情况下快速提升预训练零样本 TTS 在罕见说话风格下的模仿质量。

**💡 创新点**

通过将可学习前缀作为策略，利用 F0、能量、说话人相似度以及 WER 四种奖励，在推理时对模型进行轻量级、即时适配，从而克服域偏移与数据稀缺问题。

**🔧 技术方法**

强化学习（GRPO）、流匹配（flow‑matching）TTS 模型、声学特征奖励、Whisper ASR 评估、说话人嵌入相似度计算。

**📊 数据集**

内部 200 条罕见风格语音样本（方言、口音、儿童、含糊发音等）以及 KeSpeech 160 条中文方言语句。

**📈 对比分析**

在五种不常见的测试场景（口音、儿童、含糊、中文草图、中文方言）上，与 CosyVoice、MaskGCT、Vevo、F5‑TTS 等 SOTA 零样本 TTS 对比，VoiceTTA 在 WER、说话人相似度、风格 MOS 及自然 MOS 上均优于或持平，尤其在风格相似度和 WER 上表现突出。

**⚠️ 局限性**

仍需进一步验证在更大规模、多语言、跨设备等多样化场景下的稳健性；对前缀数量、温度等超参数的依赖可能导致部分任务性能下降。

---

## 198. VIGIL: Runtime Enforcement of Behavioral Specifications in AI Agent Skills

**arXiv ID:** 2606.26524 | [PDF](https://arxiv.org/pdf/2606.26524v1)

**作者:** Ying Li `[一作]` (University of California, Los Angeles), Yuan Tian `[通讯]` (University of California, Los Angeles)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个端到端的运行时参考监视器，能够在 AI 代理使用第三方技能时，基于记录的工具调用序列实时执行行为规范检查，并在发现违规时提供可定位的违例事件。

**💡 创新点**

创新点包括：① 设计了一种基于有限时序安全属性的行为规范语言，支持上下文绑定、时间顺序和值流约束；② 引入了从自然语言规范到可执行策略的编译步骤，使规范在每个运行时被特定化；③ 通过符号化评估将策略转换为 SMT 约束，利用不可满足核心定位违规事件，实现了精确而可追踪的执法。

**🔧 技术方法**

主要技术手段有：有限时序行为规范语言、语义编译规则、基于 Z3 的 SMT 求解、LLM（Claude）用于规范编译、抽象化工具调用记录为强类型事件、无状态的引用监视器架构。

**📊 数据集**

使用的数据集包括：SkillsBench + Skill-Inject（152 条标注轨迹）、AgentDojo、SafeAgentBench（分别用于跨基准对比）、以及 216 条来自 NVIDIA、Databricks、Trail of Bits、Cloudflare 等公开技能生态的真实部署轨迹。

**📈 对比分析**

与基线系统（AgentSpec、Progent、LLM-as-Judge）进行对比，评估指标为 TP、FP、精确率、召回率、F1。我们的系统在 SB+SI 上达到了 95.8% 的召回率、89.6% 的精确率，F1 为 92.6%，相较最强基线提升约 15.8 F1。运行时开销平均 4.38 s/轨迹，约 0.27 s 用于 SMT 检查，主要成本在 LLM 编译，整体可扩展。

**⚠️ 局限性**

局限性包括：① 仅能监视可观测的工具调用，隐藏在脚本内部的违规难以捕获；② 规范编译过程中可能产生过宽或过窄的约束导致误报或漏报；③ 依赖 LLM，若模型质量或提示不足会影响编译正确性；④ 需要完整的事件记录，对资源受限或非可观测环境不适用；⑤ SMT 求解虽高效，但在极大轨迹或复杂约束下可能成为瓶颈。

---

## 199. Order-2 bygone-state opacity of labeled finite-state automata

**arXiv ID:** 2606.26503 | [PDF](https://arxiv.org/pdf/2606.26503v1)

**作者:** Kuize Zhang `[一作]` `[通讯]` (Xi'an Jiaotong University), Kuize Zhang (Xi'an Jiaotong University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出并验证了针对两个观察者的“order‑2 bygone‑state opacity”概念，说明在一个已知有限状态自动机中，第二个观察者无法确定第一个观察者是否能唯一推断系统在过去任意时刻的状态。

**💡 创新点**

首次将双代理的不确定性推断问题与传统的可观测性、可检测性等概念统一，并将其映射为并发组合与观察者的组合，提供了可实现的验证方法。

**🔧 技术方法**

使用了并发组合（Concurrent Composition）、经典观察者（Observer）与逆向观察者（Inverse Observer）等离散事件系统的标准工具，并在此基础上构造了“order‑2 observer”和“order‑2 inverse observer”。

**📊 数据集**

论文中未使用任何具体数据集，而是通过理论构造和示例图展示验证方法。

**📈 对比分析**

验证方法通过构造并发组合与观察者，能够在双指数时间内给出所有可能的 bygone‑state 估计，并判断 opacity 是否满足；该方法是目前已知的最简洁、最统一的解决方案，但计算复杂度为双指数。

**⚠️ 局限性**

主要局限在于算法复杂度高，难以应用于大规模系统；此外，缺乏实验评估和对比基准，导致对实际性能的量化评估不足。

---

## 200. Pingquanqi (Equalizer): A Cross-Domain Sociotechnical Framework for Human-Agent Interaction Governance

**arXiv ID:** 2606.26573 | [PDF](https://arxiv.org/pdf/2606.26573v1)

**作者:** Yu Wang `[一作]` `[通讯]`, Yu Wang

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Pingquanqi（平权器）——一种跨域社会技术框架，用于在 LLM 代理的框架层面实现成本治理与用户生命周期保护；

**💡 创新点**

创新点在于把经济对齐视作治理核心，形成四个“制动器”（用户状态鉴别+知识升维、贝叶斯递进止损、受控摩擦、透明度计量）与反思总结（F5）组成的完整中间件标准，类似 WCAG；

**🔧 技术方法**

核心技术包括文本‑仅用户状态鉴别模型、基于贝叶斯的交互停止阈值、可调摩擦机制、Token‑Lifetime 透明度指标以及会话结束的反思摘要；

**📊 数据集**

本文未进行实验性数据集收集，主要依据公开文献检索（arXiv、Semantic Scholar、CrossRef、中文数据库）与人工校验，强调概念与规范的验证；

**📈 对比分析**

由于缺乏实现版本与用户实验，未给出定量性能对比；理论上预期可降低 token 消耗、缩短交互时长、提升用户满意度，但需通过用户研究验证；

**⚠️ 局限性**

局限性包括：阈值与参数尚未经验校准；只适用于文本型 LLM 代理，未覆盖语音/ XR 介质；可能面临反向适配的攻击；需进一步评估对可访问性与伦理影响。

---

## 201. PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting

**arXiv ID:** 2606.26549 | [PDF](https://arxiv.org/pdf/2606.26549v1)

**作者:** Ao Hu `[一作]` (Southwestern University of Finance and Economics), Zenglin Xu `[通讯]` (Shanghai Academy of AI for Science)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 PMDformer 模型，通过 patch-mean decoupling、proximal variable attention 与 trend restoration attention 三种模块实现长周期时间序列预测。

**💡 创新点**

创新点包括：① 用 patch-mean decoupling (PMD) 去除每个 patch 的均值，保留形状信息；② 用 proximal variable attention (PVA) 仅关注最近 patch 的变量交互；③ 用 trend restoration attention (TRA) 在注意力中显式注入趋势，恢复全局趋势信息。

**🔧 技术方法**

技术手段：基于 Transformer 的多头自注意力架构；对时间序列进行非重叠 patch 划分；对 patch 进行均值去除、残差嵌入；在不同位置引入 PVA 与 TRA；最终通过线性投影得到预测。

**📊 数据集**

使用数据集：ECL、Traffic、Weather、Solar、ETTh1、ETTh2、ETTm1、ETTm2 共 8 个公开长周期多变量时间序列基准。

**📈 对比分析**

与 9 个 SOTA 基线（TQNet、TimeBase、SOFTS、SparseTSF、ModernTCN、iTransformer、TimeMixer、PatchTST 等）在 4 种预测长度（96、192、336、720）上进行对比，PMDformer 在 7/8 数据集上均取得最低 MSE/MAE，平均 MSE/MAE 分别下降约 5.7%–12.4%，显示显著性能提升。

**⚠️ 局限性**

局限性：实验仅覆盖 8 个基准数据，未验证在更高维、多模态或更大规模数据上的泛化；对 patch 大小与 k 值的敏感性需要进一步分析；模型结构复杂，训练成本相对较高。

---

## 202. Compiler-Driven Approximation Tuning for Hyperdimensional Computing

**arXiv ID:** 2606.26547 | [PDF](https://arxiv.org/pdf/2606.26547v1)

**作者:** Xavier Routh `[一作]` (University of Illinois Urbana-Champaign), Vikram Adve `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ApproxHDC，一种面向超维计算（HDC）的编译器驱动近似调优框架，能够自动识别HDC原语并在CPU、GPU及模拟的ReRAM/PCM加速器上联合搜索软件和硬件近似配置；

**💡 创新点**

创新点在于：①将近似空间与HDC特性紧密耦合，提供域特定的搜索空间裁剪与注解机制；②支持硬件级近似（ADC分辨率、写-验证循环、多级单元等）并将其整合到编译器循环中；③通过自动化注解插入实现端到端可重用的近似配置；

**🔧 技术方法**

采用HPVM‑HDC编译器、OpenTuner调优框架、Python实现的ApproxHDC、以及SpecPCM模拟器等技术；

**📊 数据集**

使用ISOLET（语音字母）、Cora（科学论文）等公开数据集进行分类、聚类、图神经网络和基因序列搜索实验；

**📈 对比分析**

与MicroHD对比，ApproxHDC在GPU上最高可获得24.65×的速度提升（约3.5×优于MicroHD），在CPU、GPU和SpecPCM上分别实现17.25×、15.02×和4.69×的加速，同时满足用户设定的准确率阈值；

**⚠️ 局限性**

局限性包括：近似搜索空间仍极大，需要足够的搜索预算和手工注解；对非HDC应用的通用性有限；硬件近似的精细调优仍受模拟模型准确性的限制。

---

## 203. Clinical Harness for Governable Medical AI Skill Ecosystems

**arXiv ID:** 2606.26494 | [PDF](https://arxiv.org/pdf/2606.26494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 204. CascadeFormer: Depth-Tapered Transformers Motivated by Gradient Fan-in Asymmetry

**arXiv ID:** 2606.26538 | [PDF](https://arxiv.org/pdf/2606.26538v1)

**作者:** Huzama Ahmad `[一作]` (Korea Advanced Institute Of Science And Technology), Se-Young Yun `[通讯]` (Korea Advanced Institute Of Science And Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Gradient Fan-in Asymmetry（GFA）理论，并基于该理论设计了CascadeFormer（宽度逐层衰减的Transformer）和CascadeFlow Pruning（使用训练时累计梯度进行层剪枝）的两种高效模型改进方法。

**💡 创新点**

创新点在于：①将梯度的结构不均衡（fan‑in）视为导致深层冗余的根本原因；②用累计梯度共享来衡量层的重要性，直接用于剪枝；③通过宽度逐层衰减使模型容量与梯度信息流匹配，从而在相同训练 FLOPs 下实现更快推理。

**🔧 技术方法**

使用残差Transformer和ResNet训练、累计梯度共享、功能重要性评估、宽度逐层衰减、参数共享层重复、LayerPassThrough剪枝实现等技术。

**📊 数据集**

语言模型使用Dolma 7B子集（next‑token 预测）和ImageNet‑1K（ResNet 训练）数据集；评估集包括 Dolma hold‑out 和 HellaSwag zero‑shot。

**📈 对比分析**

与均匀宽度基线、LayerSkip、传统剪枝（Taylor、Magnitude、相似度）等方法比较。CascadeFormer 在相同训练 FLOPs 下保持与基线相同的 perplexity，同时推理延迟降低 8.6%，吞吐量提升 9.4%。CascadeFlow Pruning 在剪枝后保持更低的 perplexity，且相较于 Taylor/Magnitude 等方法更稳定、性能衰减更温和。

**⚠️ 局限性**

局限性包括：仅在 1.2B 参数规模模型验证，未检验 100B+ 级别；使用梯度累计依赖训练时梯度，无法对预训练封闭模型进行后置剪枝；梯度 fan‑in 作为路径计数近似，未考虑梯度有效秩、正交性等信息，可能导致对信息多样性的低估。

---

## 205. OSC2Runner: OpenSCENARIO 2.x Compliant High-Fidelity AV Simulation in CARLA

**arXiv ID:** 2606.26533 | [PDF](https://arxiv.org/pdf/2606.26533v1)

**作者:** Thoshitha Gamage `[一作]` (Southern Illinois University Edwardsville), Lasanthi Gamage `[通讯]` (Webster University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文实现了一个多阶段编译器，将 ASAM OpenSCENARIO v2.x DSL 语义安全、确定性地映射到 CARLA 仿真引擎，实现了从 DSL 到行为树的实时编译与执行；

**💡 创新点**

创新点在于引入多通道 AST 语义分析与行为树合成的 transpiler 架构，直接映射到 CARLA 原子 API，消除了传统解释器带来的时空漂移与异步延迟；

**🔧 技术方法**

使用的技术包括 ANTLR4 语法解析、强类型 AST 生成、双遍语义检查、Behavior Tree（py_trees）生成、Python 运行时上下文管理以及 PID 控制器实现 kinematic modifier；

**📊 数据集**

实验使用 CARLA 官方 Town06 场景和自定义 OpenSCENARIO 脚本（包含多车、事件触发与环境参数），并未依赖外部公开数据集；

**📈 对比分析**

通过两项对抗性案例研究（切入与事件同步、环境摩擦调节），与传统解释执行相比，框架实现了 100 ms 事件同步、无时空漂移、完整的行为树动态执行，验证了高精度、实时响应；

**⚠️ 局限性**

局限性包括 Python 运行时带来的性能瓶颈（高并发下帧率下降），以及对标准语法的完整覆盖尚未完成，后续计划迁移核心模块至 C++ 并扩展至更复杂的交互与感知场景。

---

## 206. The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Report

**arXiv ID:** 2606.26529 | [PDF](https://arxiv.org/pdf/2606.26529v1)

**作者:** Kwan Soo Shin `[一作]` `[通讯]` (PolymathMinds Lab), Kwan Soo Shin (PolymathMinds Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在任务条件化（task-conditioned）下，大型语言模型和视觉-语言模型如何出现注意力盲区，即在专注于指定目标时抑制对未请求但安全关键的信号的报告。

**💡 创新点**

提出并 formalized Inattentional Gap 概念，证明其在不同模态、不同安全领域、不同模型规模下普遍存在，并发现抑制根源与输出范围（output scope）相关，而非仅仅是认知负荷。

**🔧 技术方法**

使用对比实验：给同一输入分别提供开放式提示和任务式提示，测量报告率；通过判定程序评估模型在两种提示下的差异；对输出范围进行剂量分析；在推理模型上验证该现象。

**📊 数据集**

利用受控医学影像数据（胸腔 X 光）、驾驶文本情境数据以及公开的语言/视觉-语言模型训练集进行实验。

**📈 对比分析**

通过 IG_m,i 指标量化模型在开放与任务条件下的报告差异，比较不同模型家族、规模及任务负荷的抑制程度；结果显示所有测试模型均出现抑制，规模提升并未减轻该现象；与传统安全基准对比表明高分并不保证在未指定场景下的安全。

**⚠️ 局限性**

局限性：实验覆盖的模型家族和任务类型有限，未验证更广泛的安全关键情境；缺乏对模型内部机制的深度解释；仅关注报告层面的抑制，未提供完整的补救或防护方案。

---

## 207. Revisiting Action Factorization for Complex Action Spaces

**arXiv ID:** 2606.26574 | [PDF](https://arxiv.org/pdf/2606.26574v1)

**作者:** Timothy Flavin `[一作]` (University of Tulsa), Sandip Sen `[通讯]` (University of Tulsa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对三类强化学习算法（DQN、PPO、SAC）在离散、连续及混合动作空间上，系统评估了六种动作因式分解方法（独立、共享编码、VDN、QPLEX、联合、Auto‑Regressive），并提出了基于分支优势的VDN‑PPO与PPO‑MIX两种改进。

**💡 创新点**

创新点在于（1）构造两款轻量化可调因果依赖环境（CoopPush、Hybrid‑Shoot）以统一评测；（2）提出利用V‑only目标训练分支双子网络并加权优势的自适应方差减小技术；（3）对三种算法统一比较，揭示共享编码（特别是VDN）在可观测环境下实现最佳计算‑性能折衷，Auto‑Regressive在最优性能与计算延迟之间的权衡。

**🔧 技术方法**

采用分支双子Critic、优势分解与重加权、第一阶混合梯度估计、Gumbel‑Softmax松弛、分支优势均衡等技术；网络结构为两层隐藏层（256/128单元）并配以共享或独立头。

**📊 数据集**

使用四个轻量化基准（Platform、Hybrid‑LunarLander、Hybrid‑Shoot、CoopPush）以及公开的Gymnasium/ PettingZoo 等标准环境作为对照。

**📈 对比分析**

通过统一步骤数、标准化奖励曲线、AUC、阈值到达步数和主动/被动头方差比等指标比较，实验显示共享编码（VDN）在所有算法中取得最高平均奖励；Auto‑Regressive在大多数配置下表现最佳，但计算延迟显著；PPO‑VDN/PPO‑MIX在离散空间显著提升 4% 奖励与 8% 样本效率。

**⚠️ 局限性**

局限在于：1）对极大动作空间的可扩展性不足，QPLEX 与联合因子在维度增大时性能下降；2）Auto‑Regressive 的顺序依赖导致推理延迟；3）仅在单机/小规模并行下测试，未充分评估分布式或异步框架对这些因子化方法的影响。

---

## 208. Characterizing Pure Strategy Nash Equilibria in Finite Noncooperative Games

**arXiv ID:** 2606.26564 | [PDF](https://arxiv.org/pdf/2606.26564v1)

**作者:** Shravan Luckraz `[一作]` `[通讯]` (University of Nottingham), Shravan Luckraz (University of Nottingham)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了有限博弈中纯纳什均衡的存在性，提出了基于最优回应图的必要与充分条件，并统一了潜在游戏、超模性、单侧竞争和聚合游戏等多类已有结果。

**💡 创新点**

首次将最优回应的离散结构与组合/序理论相结合，得到一个完整的纯均衡判据；同时引入了新的单侧竞争扩展与聚合最大化条件，进一步拓宽了可判定的游戏类别。

**🔧 技术方法**

使用组合数学、格理论、离散可积性（路径无关性）以及词典序的改进法来构造潜在函数和证明纯均衡存在。

**📊 数据集**

无，论文为理论性研究，无需具体数据集。

**📈 对比分析**

通过与已有的潜在游戏、超模性、单侧竞争等充分条件进行对比，证明所给条件能涵盖并统一这些结果；由于仅为理论阐述，未涉及实验性能评估。

**⚠️ 局限性**

仍仅给出充分（有时必要）条件，对所有有限博弈的完全必要与充分判据尚未实现；条件的检查在策略空间非常大时可能计算复杂；研究聚焦离散有限游戏，无法直接推广到连续或无穷维情形。

---

## 209. Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention

**arXiv ID:** 2606.26560 | [PDF](https://arxiv.org/pdf/2606.26560v1)

**作者:** Xiao Li `[一作]` (Qwen Team), Jingren Zhou `[通讯]` (Qwen Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Erase-then-Delta Attention（EDA），在线性注意力的记忆更新中先通过独立擦除方向清除旧信息，再按当前写入键执行 Delta 规则纠正写入。

**💡 创新点**

核心创新在于把擦除地址与写入地址解耦：在同一步内实现一次独立的擦除操作与一次 Delta 纠正写入，既保持 Delta 规则的纠错优势，又提供针对性清理旧记忆的能力。

**🔧 技术方法**

使用通道级 Gated Delta（DPLR）线性注意力框架，加入低秩擦除算子、对数空间安全衰减门、块级并行实现；同时在 MoE 稀疏激活和混合（Hybrid）架构中测试。

**📊 数据集**

在 2.5B 密集模型与 25B-A2.8B MoE 模型上进行 400B token 预训练，随后 80B token 长序列（32k）中继续训练；评估使用 MMLU、MMLU-Pro、GSM8K、MATH、BBH、EvalPlus 以及 RULER 长上下文（4k–128k）任务。

**📈 对比分析**

与 Transformer、GDN、GDN-2、KDA 等基线在相同训练设置下比较；EDA 在 dense 2.5B 上平均得分提升 0.63 点，在 MoE 25B 上平均提升 1.22 点；mid‑training 后优势保持；长上下文 RULER 评测中 EDA 也略优。

**⚠️ 局限性**

局限性：加入独立擦除后会降低写键回忆；当前的分析只测量门分配与读出扰动，未能直接关联单次擦除事件与下游性能提升；擦除路径是条件清理机制而非普适提升记忆保真度。

---

## 210. Perception, Verdict, and Evolution: Hindsight-Driven Self-Refining Forensics Agent for AI-Generated Image Detection

**arXiv ID:** 2606.26552 | [PDF](https://arxiv.org/pdf/2606.26552v1)

**作者:** Yangjun Wu `[一作]` (Zhejiang University), Fei Wu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 ForeAgent，一种结合多视角感知（语义、频域、空间）与多模态大型语言模型裁决的 AI 生成图像检测框架，并实现了基于后向反思和双专家质量门控的迭代自我改进循环。

**💡 创新点**

创新点在于：①将多模态感知结果与 MLLM 进行逻辑融合的 Perception–Verdict 架构；②采用后向反思（Sampling–Reflection–Evolution）与双专家质量门控，实现在无人工标注的情况下持续自我进化与可解释推理；③在训练中使用自生成的高质量样本替代传统人工或静态数据。

**🔧 技术方法**

核心技术包括：Qwen3-VL-8B 作为主体 MLLM、离散小波变换提取频域特征、NPR 工具提供空间特征、双专家（ForeAgent 与 Qwen3-VL-Plus）进行质量评估、LoRA 微调以及采样-反思-演化循环。

**📊 数据集**

使用的数据集为 Genimage 的 60k 真实/假图像和 100k 额外样本进行自我改进；在 AIGCDetectBenchmark（16 种生成器）和 Chameleon（真实/假图像）基准上进行评估。

**📈 对比分析**

与传统特征检测、MLLM 解释性检测及 AIDE、AIGI‑Holmes 等基线比较，ForeAgent 在 AIGCDetectBenchmark 上平均准确率达到 93.3%（比 AIDE 提升 16.41%），在 Chameleon 上整体准确率为 82.18%，且推理质量评分高于 GPT‑5。

**⚠️ 局限性**

局限性包括：对 WFIR 等后处理显著的来源性能不足；仅使用固定的多视角特征，缺乏自适应特征选择；迭代自我改进过程依赖阈值设定，可能受限于初始模型的先验偏差。

---

## 211. Sample-efficient Transfer Reinforcement Learning via Adaptive Reward Shaping and Policy-Ratio Reweighting Strategy

**arXiv ID:** 2606.26527 | [PDF](https://arxiv.org/pdf/2606.26527v1)

**作者:** Wenjie Huang `[一作]` (Hunan University), Helai Huang `[通讯]` (Central South University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过教师干预与奖励塑造实现安全高效的自适应迁移强化学习，用于高速公路车道变更决策

**💡 创新点**

提出自适应教师干预机制、基于教师评价的奖励塑造以及基于策略比例的重权重策略，使得学生在保证安全的前提下快速收敛

**🔧 技术方法**

基于SAC的约束强化学习框架、教师干预决策、奖励塑造、优化权重重标、双源经验回放

**📊 数据集**

仿真高速公路环境（自定义）和真实交通数据集NGSIM US‑101

**📈 对比分析**

与SAC、PPO‑Lag、TS2C对比，实验显示在安全（成本、碰撞率）和效率（平均奖励、速度）方面均显著优于基线，安全提升>90%，效率提升5‑7%

**⚠️ 局限性**

仍需教师策略的先验知识，对不同任务的泛化性有限；对超参数如干预阈值敏感；仅在车道变更场景验证，未覆盖更复杂交通情境

---

## 212. Multipath Adaptive Gated Bottleneck Latent ODE with Raman Data Fusion for Cell Culture Process Forecasting

**arXiv ID:** 2606.26520 | [PDF](https://arxiv.org/pdf/2606.26520v1)

**作者:** Johnny Peng `[一作]` (Complex Adaptive Systems Laboratory, Data Science Institute, University of Technology Sydney), Bogdan Gabrys `[通讯]` (Complex Adaptive Systems Laboratory, Data Science Institute, University of Technology Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合门控瓶颈Latent ODE和多路径即时微调（MP‑JIT‑FT）的框架，并通过Raman光谱生成的伪观测实现数据融合，用于哺乳动物细胞培养过程的多天预测。

**💡 创新点**

创新点：① 将检索+层次聚类与JIT微调相结合，产生多条有置信度的预测路径；② 在Latent ODE中加入可学习通道门和掩码感知瓶颈，提升稀疏高维输入的学习效果；③ 通过软传感器产生的Raman伪观测进行融合，增强早期预测的观测丰富度。

**🔧 技术方法**

使用技术：神经常微分方程（Latent ODE）、门控瓶颈结构、掩码感知压缩、层次聚类、JIT检索、GELU/Softplus激活、回归损失、伪观测融合。

**📊 数据集**

数据集：38个5L喂养批次生物反应器运行，涵盖14个实验条件，包含10个离线测定变量（9个目标），每隔1–3次/天采样，Raman光谱每45分钟采集。

**📈 对比分析**

与全局基线（Linear Kalman Filter、CART、Global Latent ODE）通过留一批交叉验证比较，评价指标为NMAE和95%置信区间宽度。MP‑JIT‑FT GB‑Latent ODE+Raman在大多数目标变量上平均排名最佳，NMAE显著低于全局Latent ODE，在9个变量中优于其8个。

**⚠️ 局限性**

局限性：① Raman伪观测的权重固定，未考虑不确定性；②检索与聚类的超参数手工设定，可能不适用于不同数据；③ 计算开销大，尤其在加入Raman融合时训练/推理时间显著提升，限制了实时应用。

---

## 213. Budget-Aware Keyboardless Interaction

**arXiv ID:** 2606.26508 | [PDF](https://arxiv.org/pdf/2606.26508v1)

**作者:** Quang-Thang Nguyen `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一种仅依赖标准摄像头和印刷键盘纸张的低成本键盘无交互系统。

**💡 创新点**

不需要额外标记或投影仪，利用YOLOv8模型与传统图像处理实现键盘区域、键识别及指尖颜色检测，降低硬件成本。

**🔧 技术方法**

使用YOLOv8语义分割/目标检测、Homography变换、MediaPipe手部关键点、指甲颜色分析等技术。

**📊 数据集**

微软COCO、E-waste、MieKeyboard数据集用于键盘检测；KaggleNail数据集用于指甲分割。

**📈 对比分析**

通过多角度实测，键盘检测平均AP 92%，键识别约70%；触摸检测仅约36%，与投影或AR系统相比硬件成本更低，但准确率仍需提升。

**⚠️ 局限性**

受光照、摄像头角度、按压力度和指甲差异影响，触摸检测准确率低，系统存在延迟和需较大按压的限制。

---

## 214. DanceDuo: Bridging Human Movement and AI Choreography

**arXiv ID:** 2606.26507 | [PDF](https://arxiv.org/pdf/2606.26507v1)

**作者:** Gia-Cat Bui-Le `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (Vietnam National University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 DanceDuo 平台，利用扩散模型实时生成与音乐同步的舞蹈序列，并通过姿态估计与用户自录视频进行可视化比较。

**💡 创新点**

创新点在于：①将音乐到舞蹈的生成与交互式姿态比较功能相结合；②设计了基于 Beta 分布的分数公式，提升用户参与感；③采用 DanceFusion 扩散模型实现长序列同步生成。

**🔧 技术方法**

使用技术包括：扩散模型（DanceFusion）、SMPL 3D 姿态模型、ROMP 单阶段 3D 姿态估计、Blender+Unity 可视化、Beta 分布打分算法。

**📊 数据集**

使用的数据集为 AIST++ 音乐到舞蹈数据集，以及公开的 3D 姿态模型数据。

**📈 对比分析**

通过 MPJPE/MPJAE 计算姿态相似度并映射至 Beta 分布得分；用户研究显示界面直观、比较功能受欢迎，生成舞蹈与音乐同步性与流畅度获得正面评价。

**⚠️ 局限性**

局限性包括：缺乏实时处理功能；音乐与舞蹈同步仍需改进；用户帮助不够；只能使用预设音乐与模型，无法自定义音轨。

---

## 215. Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation

**arXiv ID:** 2606.26502 | [PDF](https://arxiv.org/pdf/2606.26502v1)

**作者:** Han-yu Wang `[一作]` `[通讯]` (University of Hong Kong), Han-yu Wang (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并验证了一种两级对比诊断，用以区分思考系统在面对难度检测（registration）和在已知难度上分配计算资源（allocation）时的行为差异；通过该诊断发现，尽管大型推理模型（LRM）与人类在跨题目难度评估上表现一致，但在同一题目上，LRM在错误试题上会产生更长的推理链，人与人类则相反；

**💡 创新点**

创新点在于首次将跨题目难度相关性与同题目内的结果条件资源分配分离开来，并以“d‑ratio”和基于项目固定效应的交互项两种统计量进行检验；该方法揭示了人机在延迟控制机制上的根本性差异，提示“推理更久=不确定性”这一表述在不同系统中含义不同；

**🔧 技术方法**

主要技术包括：对大型语言模型输出的推理链长度计数、对人类反应时的对数化、Cohen’s d 计算、基于项目固定效应的线性回归、混合效应模型以及对模型推理链中自我怀疑/词频等内容特征的回归分析；

**📊 数据集**

使用了公开的 de Varda 等人发布的人机匹配推理数据集，包括三种推理范式：H‑ARC（视觉抽象）、INTUIT（直观物理）和 Cortes（关系推理）；模型样本为六个开放权重的思考型 LRM（DeepSeek‑R1、Qwen‑QwQ‑32B、Qwen3‑235B‑Thinking、GLM‑4.5‑Air‑FP8、gpt‑oss‑20b、gpt‑oss‑120b）以及一个非思考型基线 DeepSeek‑V3；

**📈 对比分析**

比较方法是将每个代理（人类或模型）视为自己的基准，先计算跨题目难度相关性（Spearman ρ），随后在项目固定效应回归中检验正确/错误结果对推理长度的影响；结果显示：LRM在错误试题上推理长度显著更长（d≈1.5–3.1），人类则在错误试题上更短（d≈‑0.1），两者在跨题目难度上相关性均为正，表明两级诊断揭示了不同的内在决策策略；

**⚠️ 局限性**

局限性包括：仅观察性数据，无法确立错误试题额外令牌的因果价值；模型样本仅为六个开放权重模型，无法推广至闭源前沿模型；人类数据缺乏个体级标识，导致难以区分个体差异与项目内行为；Cortes 任务是边界案例，LRM的同题目正误效应符号与其他范式不同；测量工具（推理令牌长度 vs 反应时）虽类比但并非可直接量化的等价物；最后，行为差异只能提供对机制的提示，无法确证底层计算架构。

---

## 216. Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context

**arXiv ID:** 2606.26493 | [PDF](https://arxiv.org/pdf/2606.26493v1)

**作者:** Fitsum Reda `[一作]`, Bryan Catanzaro `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种块级自回归扩散模型，将上下文表示与迭代去噪解耦为两座塔：冻结的 AR 上下文塔和可训练的扩散去噪塔，利用预训练的 30B 混合 Mamba‑Transformer MoE 模型实现高吞吐量文本生成。

**💡 创新点**

核心创新在于：① 两塔架构将上下文与去噪完全分离，避免单模型权重被拉伸到两种角色；② 扩散去噪塔使用层对齐的交叉注意力、双向块注意力以及时间条件（adaLN）提升去噪质量；③ 在保持预训练 AR 表示的同时，仅用 2.1T token 进行去噪塔微调，显著提升生成速度。

**🔧 技术方法**

技术主要包括：掩码扩散（Masked Diffusion）框架、双向块注意力、层对齐交叉注意力、时间条件的 AdaLN、Mamba 2 + 关注层 + MoE 的混合架构、基于块的自回归生成算法、置信度阈值的动态去噪步。

**📊 数据集**

使用与原始 30B 预训练模型相同的混合大规模数据集，总计约 2.1T tokens（包含多语言、代码、通用知识等来源）。

**📈 对比分析**

与同一预训练权重的 AR 基线进行比较，评估基准（一般知识、代码、数学、常识、多语种）累计准确率及推理吞吐量。实验显示，Two‑Tower 在保持 98.7% AR 质量的同时，壁钟生成吞吐量提升 2.42×，并在大多数任务上保持或提升准确率。

**⚠️ 局限性**

局限性包括：
- 块大小与置信度阈值对质量‑吞吐权衡敏感，需要手动调优；
- 采用双塔架构导致模型参数量与内存占用增加；
- 在代码与数学等高精度任务上，仍略低于纯 AR 基线；
- 目前仅在 30B 规模验证，尚未证明在更大 MoE 体系下的可扩展性；
- 对于非常长序列的推理，KV 缓存管理仍是瓶颈。

---

## 217. An Empirical Study of LLM-Generated Specifications for VeriFast

**arXiv ID:** 2606.26490 | [PDF](https://arxiv.org/pdf/2606.26490v1)

**作者:** Wen Fan `[一作]` (Purdue University), Lin Tan `[通讯]` (Purdue University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型（LLM）在生成可被 VeriFast（基于分离逻辑的静态验证器）验证的 C 程序规范的能力，包括功能行为保持、可验证性以及错误类型的深入分析。

**💡 创新点**

首次系统性评估 LLM 在 SL 验证器上的规范生成效果，使用八种提示策略、十个 LLM、三种输入形式，并提供完整数据集、错误分类及可操作的改进建议。

**🔧 技术方法**

提示工程（Basic、CoT、RAG‑sparse/dense）、LLM（Claude‑3.7 Sonnet、Gemini‑2.5 Pro、GPT‑4o 等）、VeriFast 24‑08‑30、BM25、fastembed、Qdrant、Python 自动化脚本等技术。

**📊 数据集**

由 60 个 C 程序（共 303 个函数）构成的数据集，包含三种输入版本（自然语言 NL、正式前后置 FB、FB 加额外辅助 FBP），以及对应的完整验证版本；程序来自 VeriFast GitHub 库 50 个 + 10 个作者自制。

**📈 对比分析**

通过功能行为保持率（源码 9%、前后置 87%+）、验证成功率（整体 31.4%）以及按输入类型、模型和语言特征（递归、并发、循环等）的细分比较；Gemini‑2.5 Pro 在所有指标上表现最佳，FBP 输入效果最佳，错误分析揭示主要是堆推理与语法错误。

**⚠️ 局限性**

仅针对 C 语言和 VeriFast，数据集规模有限；未使用 LLM 反馈循环；错误多集中在语法/堆相关推理，难以处理更复杂的并发/循环等情形，导致验证成功率仍相对较低。

---

## 218. WQ-Fusion: Dynamic Gated Attention for Cross-Domain Audio Representation

**arXiv ID:** 2606.26556 | [PDF](https://arxiv.org/pdf/2606.26556v1)

**作者:** Mingda Lin `[一作]` (Wuhan University), Jacob Benesty `[通讯]` (University of Quebec)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种跨域音频表示学习框架WQ‑Fusion，利用Whisper和Qwen两种强大预训练编码器，通过自适应特征调制（AFM）与逐元素门控注意力实现动态特征融合，最终生成统一的高质量音频表示。

**💡 创新点**

创新点包括：
1) 引入AFM模块，动态预测缩放/偏移参数而非固定投影，提升跨编码器特征的对齐能力；
2) 采用逐元素门控注意力（Gated Transformer）在融合阶段实现上下文感知的特征筛选，克服传统静态拼接的刚性；
3) 将RoPE与可学习模块嵌入相结合，使模型既能保留时序信息，又能辨识不同编码器来源，从而在多模态任务中实现更灵活的特征权重分配。

**🔧 技术方法**

核心技术包括：
- Whisper 与 Qwen 预训练编码器（冻结权重）；
- Adaptive Feature Modulation（AFM）对输入特征进行动态归一化与线性缩放；
- Rotary Position Embedding（RoPE）+ 可学习模块嵌入；
- 逐元素门控注意力（Gated Transformer）实现动态特征路由；
- 轻量化可训练组件：MLP 投影层、LoRA 低秩适配矩阵和门控机制。

**📊 数据集**

使用 Interspeech 2026 Audio Encoder Capability Challenge（Track A）Benchmark 的 15 个公开数据集：
- 语音领域：SC, LibriCount, VoxLingua107, VoxCeleb1, ASVspoof, FSC, VocalSound, CREMA‑D；
- 声音领域：ESC‑50, FSD50k, UrbanSound 8k, FSD18‑Kaggle；
- 音乐领域：GTZAN, NSynth‑I, FMA。

**📈 对比分析**

对比方法：
- 单一编码器基线（Whisper‑Large、Qwen2‑Audio‑7B、AudioMAE 等）
- 静态拼接（Concat.）
- AFM + 原始 Transformer（Adapt. and Trans.）
- 仅门控 Transformer（GatedTrans.）
- 完整 WQ‑Fusion
实验结果显示：单一 Qwen 的整体得分 0.796；静态拼接提升至 0.820；AFM+Transformer 进一步提升到 0.829；单门控 Transformer 0.832；最终 WQ‑Fusion 以 0.836 获得最高整体分数，并在各个子任务中表现均衡，尤其在语音/声音/音乐跨域任务上显著优于单编码器与静态融合方法。

**⚠️ 局限性**

局限性：
1) 依赖于冻结的 Whisper/Qwen 预训练模型，无法进一步通过任务特定训练提升个别领域性能；
2) 对于完全不同或更细粒度的音频任务（如音乐情感识别、低频音频分析等）尚未验证；
3) 门控注意力虽具动态特性，但仍无法捕捉所有潜在的跨模态交互模式，尤其在长时序或低信噪比环境下的鲁棒性待进一步评估；
4) 计算与内存开销主要集中在门控 Transformer 与 AFM 计算，若在资源受限设备上部署仍需压缩。

---

## 219. PhyEditBench: A Real-World Multi-Stage Benchmark for Physics-Aware Image Editing

**arXiv ID:** 2606.26551 | [PDF](https://arxiv.org/pdf/2606.26551v1)

**作者:** Shengbin Guo `[一作]` (Nanjing University), Qi Fan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 PhyEditBench，一套专注物理推理的高分辨率图像编辑基准；

**💡 创新点**

创新点在于将物理过程拆分为多阶段，加入反物理实例，并用视频生成模型的中间帧作为推理过程；

**🔧 技术方法**

采用视频生成模型（Wan2.2）与演化式测试时缩放（TTS）和视频奖励模型，结合文本增强与潜在压缩；

**📊 数据集**

使用从公开视频提取的 238 个真实物理实例和 35 个人工合成的反物理样本；

**📈 对比分析**

通过 GPT‑4o 评估四维度（一致性、指令遵循、物理可行性、图像质量），在多种跑法和类别下，PhyWorld 在物理可行性与整体得分上优于多数闭源/开源对手；

**⚠️ 局限性**

局限在于对极端长序列或极其复杂的物理交互仍表现欠佳，且评估仍依赖 VLM，无法完全捕捉细粒度像素质量。

---

## 220. ConcoLixir: Reactive LLM Discovery Oracles for Python Concolic Testing

**arXiv ID:** 2606.26545 | [PDF](https://arxiv.org/pdf/2606.26545v1)

**作者:** Dong Chen `[一作]` (National Chengchi University), Fang Yu `[通讯]` (National Chengchi University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将LLM作为发现oracle集成到Python concolic测试中的方法。

**💡 创新点**

LLM只用于生成跨语义障碍的具体输入，保持SMT求解为核心，避免LLM误判。

**🔧 技术方法**

基于PyCT的对象替代 concolic执行，配合 gpt‑4o‑mini 进行种子生成、求解失败及覆盖 plateau 探索。

**📊 数据集**

评估 22 个设计基准、22 个库函数和 75 个自动选取的 sympy、PyYAML API，合计 1190 个测试实例。

**📈 对比分析**

与 Concolic‑Only、LLM‑Seeds‑Only、CrossHair 对比，平均行覆盖率提升 8.6–17.0 个百分点，API 费用仅 $1.63，运行时延迟约 13–18 倍。

**⚠️ 局限性**

对已可达路径的增益有限；LLM 输出随机且未评估 bug 检测效果；仅在语义门障碍下显著提升。

---

## 221. TESLA-for-5G: Broadcast Authentication for 5G Networks Using TESLA

**arXiv ID:** 2606.26528 | [PDF](https://arxiv.org/pdf/2606.26528v1)

**作者:** Subin Song `[一作]` (Seoul National University), Taekyoung Kwon `[通讯]` (Seoul National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结合TESLA与GG09基于身份的签名（IBS）的5G SIB1广播认证协议。

**💡 创新点**

创新点在于首次将IBS用于一次性可信Bootstrap，随后使用TESLA的对称MAC实现高效且低延迟的持续认证。

**🔧 技术方法**

使用TESLA、GG09 IBS、Tamarin安全分析工具以及在OpenAirInterface 5G堆栈上的实现。

**📊 数据集**

实验采用真实UE移动与RRC状态轨迹收集的两套环境数据集，并在OpenAirInterface平台上进行验证。

**📈 对比分析**

与八种基线方案比较，展示在计算、通信和存储方面实现55–65%日常验证成本下降，单次返回UE认证延迟约为109 ms。

**⚠️ 局限性**

局限性包括：无法抵御中继攻击；需要与网络保持时间同步；MIB未实现认证；依赖可信PKG分发IBS密钥。

---

## 222. Boundary-Aware Context Grounding for A Low-Channel EEG Agent

**arXiv ID:** 2606.26519 | [PDF](https://arxiv.org/pdf/2606.26519v1)

**作者:** Zhiyuan Xu `[一作]` (Shanghai Pulse Element Intelligent Technology Co., Ltd.), Junwen Luo `[通讯]` (Shanghai Pulse Element Intelligent Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本论文提出NeuraDock Agent，一种将低通道EEG的确定性本地计算与LLM语言层分离的架构，并通过硬件、实现、结果和科学四个边界的上下文来约束LLM的行为。

**💡 创新点**

创新点在于：①通过版本化的硬件与实现上下文实现LLM对设备与算法边界的精准感知；②构建了可复现、可审计的请求与失败隔离机制；③在36个预设边界测试案例上量化了上下文对LLM决策安全性的提升。

**🔧 技术方法**

主要技术包括：Python deterministic EEG处理流程（MNE-Python 等）、LLM规划与解释（OpenAI兼容接口）、允许列表化的结果摘要、上下文包（硬件规格、工作流注册、结果字段、实现映射、科学边界）以及对请求的严格审计。

**📊 数据集**

使用的数据集包括：12份Rest/Task EEG录音（三位参与者），20秒的合成噪声信号用于控制干扰注入，公开的眼睛开闭EEG示例，以及三位受试者的Rest/Task对比实验数据。

**📈 对比分析**

评估方法：在三层面上测试系统——数值可复现性、请求边界审计、故障隔离、干扰注入、以及边界意识基准（36案例×4上下文×2模型共288输出）。性能上，完整上下文使四类边界判定准确率从58.3%提升至79.2%，安全响应率从26.4%提升至66.7%，同时显著降低可行请求的误拒率。

**⚠️ 局限性**

局限性包括：①基准构建基于当前内部代码与文档，缺乏独立专家验证；②仅评估两种LLM并每个案例仅生成一次；③语言评估仅为英文，未覆盖多语言情况；④跨平台与跨环境的可复现性待验证；⑤隐私审计仅针对本地请求，未覆盖第三方存储与法律合规；⑥质量流程缺失平滑跌落检测；⑦生理验证样本量小，未达到临床或广泛应用的统计功效。

---

## 223. NeuraDock Visual Cognitive Load Agent Tutorial: A Quality-Gated Open-Source EEG Workflow for Alpha Dynamics and Real-Time Applications

**arXiv ID:** 2606.26518 | [PDF](https://arxiv.org/pdf/2606.26518v1)

**作者:** Zhiyuan Xu `[一作]` (Pulse Element Intelligent Technology Co., Ltd.), Junwen Luo `[通讯]` (Pulse Element Intelligent Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发并演示了一个从EEG文件到实时视觉认知负荷指数的完整开源工作流，包括预处理、质量控制、α波动力学分析、Rest/Task对比、mini数据集验证、在线仪表盘和API以及LLM解释层。

**💡 创新点**

将离线EEG分析与实时质量门控工作流无缝衔接，提供可审计、可重复的Alpha动态与认知负荷估计，并配备LLM解释层，填补了现有工具缺乏实时质量门控与可解释API的空白。

**🔧 技术方法**

Python3虚拟环境、NeuraDock Agent CLI、MNE等EEG处理库、滚动窗口实时预处理/质量控制、Alpha频域/时域分析、FastAPI/HTTP仪表盘、LLM（如OpenAI qwen3.7）进行结果解释。

**📊 数据集**

NeuraDock硬件的7通道250Hz采样，公开的mini‑dataset共18份记录（Rest/Task等），以及示例文本文件data_examplesα_closed_eye2.txt、data_examples_task_S01_1.txt等。

**📈 对比分析**

通过与mini‑dataset已公布基准结果对比（10个Within‑subject Rest/Task对比，7/10显示Alpha抑制），使用保守的质量门控；实时API平均处理延迟约1.9ms，HTTP端点延迟约15ms；可重复性测得Pearson r≈0.803、ICC≈0.765。

**⚠️ 局限性**

样本量小、缺乏跨受试者比较、实时API未经过物理无线网络验证、仅基于Alpha动态的负荷估计未结合行为或其他生理信号、质量门控导致部分记录信息稀疏、mixed‑eye数据需谨慎解读。

---

## 224. Forget, Anticipate and Adapt: Test Time Training for Long Videos

**arXiv ID:** 2606.26515 | [PDF](https://arxiv.org/pdf/2606.26515v1)

**作者:** Rajat Modi `[一作]` (University of Central Florida), Yogesh Singh Rawat `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种Frame Forgetting Network (FFN)，通过在长视频中仅对滑动窗口的进出帧进行“忘记、预测、适应”三步，以实现高效的Test‑Time Training。

**💡 创新点**

创新点在于：① 只处理窗口中退出与进入的三帧，避免重复计算；② 通过自监督的“预测下一帧”与“惊奇度”度量自适应阈值，实现动态窗口调节；③ 引入记忆恢复机制(MRM)与时间条件化(backbone  + rope 编码)，提升长期适应性能。

**🔧 技术方法**

核心技术包括：自监督下一帧预测头、时序 MLP(rope 编码)做时间条件化、惊奇度(S) 结合视觉差异与潜在相似度、适应阈值 τ 的自适应统计（均值+方差）、记忆缓冲区与 MBC/FIFO 混合策略、余弦损失和单梯度迭代。

**📊 数据集**

实验数据集涵盖 COCO‑Videos、KITTI‑STEP、UCF101、Something‑Something v2、EpicTours（全长达3小时的实景城市游览视频）以及多种深度估计数据集（KITTI、ScanNet、NYUv2、Sintel 等）。

**📈 对比分析**

与离线全视频 TTT‑MAE、TTA 方案（TENT、layer‑norm adaptation）、在线 TTT‑Online（含滑动窗口与非滑动窗口）等基线对比，FFN 在实例/全景分割上提升 7–8 pts AP/PQ，计算速度从 4.1 s/帧降至 0.7 s/帧；在深度估计与动作分类上亦实现显著提升，且在长达数小时的视频中保持或提升性能。

**⚠️ 局限性**

局限性包括：仍需反向传播导致速度瓶颈；冷启动阶段需填充缓冲区产生延迟；单梯度步骤可能不足以在极端场景下快速适应；对极长视频的内存与算力需求仍高，且在某些无变化摄像头场景下学习停滞。

---

## 225. TinyCNNDeep: Lightweight Attention-Based CNN for EEG Classification of Eye States and Sleep Deprivation

**arXiv ID:** 2606.26506 | [PDF](https://arxiv.org/pdf/2606.26506v1)

**作者:** Thien Nhan Vo `[一作]` (Anchi STE Company), Xuan-The Tran `[通讯]` (Vietnam Maritime University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 TinyCNNDeep 轻量级 CNN，用于将五通道 EEG 转为 224×224 灰度图并实现睡眠剥夺下眼睛开闭四分类。

**💡 创新点**

将 EEG 转为图像输入 CNN 并结合残差学习和 SE 注意力模块，既轻量又高效；使用极少电极完成四分类。

**🔧 技术方法**

图像化 EEG、Z-score+min-max 标准化、中心填充、3×3 卷积残差块、SE 注意力、Adam 优化、交叉熵。

**📊 数据集**

35 位受试者的 61 通道 EEG，筛选 Fp1、Fp2、O1、Oz、O2，分为正常睡眠与睡眠剥夺两会话。

**📈 对比分析**

在每位受试者单独训练、留出 20% 测试，比较与 EEGNet、ShallowConvNet、DeepConvNet 及传统机器学习；TinyCNNDeep 平均准确率 83.69%，比最佳基线 RF 47.66% 提升 36.03%。

**⚠️ 局限性**

仅评估单个受试者内部性能，未检验跨受试者泛化；中心填充导致图像稀疏；未做解释性可视化。

---

## 226. Same Scrutiny, More Time: Eye Tracking Insights into Reviewing LLM-Labelled Code

**arXiv ID:** 2606.26505 | [PDF](https://arxiv.org/pdf/2606.26505v1)

**作者:** Ranim Khojah `[一作]` (Chalmers University of Technology and University of Gothenburg), Philipp Leitner `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验室环境中对32名软件工程师进行Wizard‑of‑Oz实验，利用眼动追踪测量他们在审阅被标记为LLM生成或未标记的相同代码时的视觉行为，并结合访谈和眼动路径进行定性分析；

**💡 创新点**

通过人为标记与真实LLM代码分离，单独评估“代码来源感知”对审阅行为的影响，并揭示提示文本在审阅中的作用，提供对AI政策与工具设计的新见解；

**🔧 技术方法**

眼动追踪（Tobii Pro Spark）采集注视与扫视数据，使用贝叶斯数据分析（BDA）建模结果，定性访谈与眼动路径编码；

**📊 数据集**

使用来自Requests与Home Assistant开源项目的Python文件，手工注入故障与代码异味，构成四个PR；

**📈 对比分析**

采用贝叶斯回归比较标记与未标记条件下的注视时长与扫视长度，发现标记代码导致注视时长显著增加（尤其是复杂代码），但扫视长度无显著差异；

**⚠️ 局限性**

实验规模有限（32人、4个简短文件），使用人为标记而非真实LLM生成代码，实验环境与真实审阅情境可能不完全匹配，受限于受试者对LLM的先验信任差异及眼动追踪设备的潜在影响；

---

## 227. Evaluation-Strategy Gap in Fault Diagnosis of Deep Learning Programs

**arXiv ID:** 2606.26492 | [PDF](https://arxiv.org/pdf/2606.26492v1)

**作者:** Sigma Jahan `[一作]` `[通讯]` (Dalhousie University), Sigma Jahan (Dalhousie University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了深度学习程序训练期间故障诊断技术在跨程序（未见过程序）场景下的泛化表现；

**💡 创新点**

创新点在于首次采用程序保留（program‑held‑out）评估框架量化评估策略差距，并系统对比曲率特征与优化器特征在不同诊断任务（故障类型分类、灾难性不稳定性、训练/验证误差不匹配）下的迁移性；

**🔧 技术方法**

使用了训练轨迹的统计特征提取（梯度、损失、Hessian‑vector products 等）、逻辑回归与随机森林分类器、MMD 等分布差异度量；

**📊 数据集**

采用了 38 个来自 Stack Overflow 的真实深度学习程序，注入 7 类故障，产生 5,542 条训练轨迹作为实验数据集；

**📈 对比分析**

在程序保留评估下，四类故障类型分类的平衡准确率从 0.469 降至 0.279（差距 0.190），曲率特征将灾难性不稳定性检测准确率提升 0.918→0.977，优化器特征虽提升在同程序内的匹配检测，但跨程序表现显著下降；

**⚠️ 局限性**

局限性包括仅基于变异注入的合成故障，缺乏真实故障验证；程序数量有限；实验仅使用线性模型，可能低估更强模型的迁移潜能。

---

## 228. Theory-Scale Auto-Formalization of Logics for Computer Science

**arXiv ID:** 2606.26525 | [PDF](https://arxiv.org/pdf/2606.26525v1)

**作者:** Yuming Feng `[一作]` (Johns Hopkins University), Ziyang Li `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 LCS-Bench，一个基于《Logics for Computer Science》的独立、规模化自动形式化基准。

**💡 创新点**

创新点在于构建了一个理论规模、异构声明的完整 Lean4 库，并引入了概念图与正式签名规划的半自动化管线以及定义等价检查器。

**🔧 技术方法**

使用了概念图、正式签名图、自动 sorry 填充与反例搜索的 agentic 流水线，并结合人类专家审查。

**📊 数据集**

使用的数据集为 LCS-Bench，包括约 4,076 条 Lean 声明、85k 行 Lean 代码，生成了 1,271 个评测实例。

**📈 对比分析**

对比方法通过定义等价检查器评估自动形式化与定理证明性能，最佳模型 Claude Opus 4.6 在 IAF 模式下仅达 20.1% 通过率，DTP 模式下达到 74.5%。

**⚠️ 局限性**

局限性包括难以覆盖极深嵌入的定理、单一教材范围、以及高难度条目导致的低通过率。

---

## 229. EvoOptiGraph: Weakness-Driven Coevolution via Graph-Based Structural Generation for Optimization Modeling

**arXiv ID:** 2606.26578 | [PDF](https://arxiv.org/pdf/2606.26578v1)

**作者:** Qingcan Kang `[一作]` (Huawei Noah's Ark Lab), Mingxuan Yuan `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于图结构的弱点驱动共进化框架EvoOptiGraph，自动化生成并训练优化建模的LLM。

**💡 创新点**

①将MILP表示为有属性的二分图，实现结构化数据生成与错误诊断；②通过弱点向量引导演化生成针对模型薄弱结构的实例；③闭环的SFT+RL可验证奖励训练。

**🔧 技术方法**

图结构遗传演化、可验证奖励的强化学习（RLVR）、SHAP+XGBoost弱点分析、GRPO策略优化、LLM微调（Qwen3-8B）等。

**📊 数据集**

六大公共基准（NL4Opt、MAMO（EasyLP/ComplexLP）、NLP4LP、ComplexOR、IndustryOR）以及由53个专家生成器演化出的自研实例。

**📈 对比分析**

与零拷贝通用LLM、专用微调模型和多步推理代理方法对比，使用Pass@1准确率作为指标，EvoOptiGraph在宏观平均上达78.1%，显著优于对手。

**⚠️ 局限性**

仅覆盖MILP且受限于种子生成器，验证成本随实例规模增长，弱点向量为启发式优先级而非因果解释。

---

## 230. Explainable Ensemble-Based Machine Learning Models for Detecting the Presence of Cirrhosis in Hepatitis C Patients

**arXiv ID:** 2606.26561 | [PDF](https://arxiv.org/pdf/2606.26561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 231. IDEA: Insensitive to Dynamics Mismatch via Effect Alignment for Sim-to-Real Transfer in Multi-Agent Control

**arXiv ID:** 2606.26575 | [PDF](https://arxiv.org/pdf/2606.26575v1)

**作者:** Chenlong Liu `[一作]` (Tongji University), Bin He `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

论文提出一种基于效果对齐的多智能体sim-to-real迁移方法 IDEA，实现了零样本部署。

**💡 创新点**

创新点在于将离散语义动作与闭环控制对齐，并通过通信同步实现时序一致。

**🔧 技术方法**

采用离散语义动作、闭环低层控制、通信同步、并行多环境训练、MAPPO等技术。

**📊 数据集**

使用Isaac Gym仿真平台生成多种几何环境，真实实验采用双UGV平台、Vicon、激光雷达数据。

**📈 对比分析**

与 DR、RMA、HIM 三种基线对比，实验显示在四个导航任务中，IDEA 训练更快、真实成功率提升 20%+，零碰撞。

**⚠️ 局限性**

局限在于需预定义离散动作空间、假设环境在 episode 内静态、同步机制导致时延。

---

## 232. Zero-shot Tweet-Level Stance Detection Enhanced by External Knowledge and Reflective Chain-of-Thought Reasoning

**arXiv ID:** 2606.26571 | [PDF](https://arxiv.org/pdf/2606.26571v1)

**作者:** Yiju Huang `[一作]` (Sichuan University), Haizhou Wang `[通讯]` (Sichuan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出零样本推文立场检测框架KIRP，利用外部知识图、实体重组、反射式链式思维、对比学习与原型网络完成四分类立场识别。

**💡 创新点**

①构建首个日语四分类推文立场数据集KIRP-D；②结合实体重组的数据增强与反射式CoT推理；③采用立场感知对比学习和三层原型网络实现细粒度分类。

**🔧 技术方法**

知识图嵌入与图自编码器、LLM反射式链式思维提示、对比学习、原型网络分类、prompt chaining、外部知识注入等技术。

**📊 数据集**

SemEval‑2016（三分类）、WT‑WT（四分类）和首创的日语四分类数据集KIRP‑D。

**📈 对比分析**

与23种基线（BiLSTM、BERT、知识增强、对比学习、LLM等）在三数据集上对比，KIRP在KIRP‑D 79.18 F1、SemEval‑16 84.05 F1、WT‑WT 84.99 F1，均为SOTA。

**⚠️ 局限性**

仅覆盖四个日语主题，跨语言/领域泛化待验证；依赖LLM和知识图的质量，可能影响模型稳定性与时效性。

---

## 233. Adversarial Diffusion Across Modalities: A Fusion Survey of Attacks, Defenses, and Evaluation for Text, Vision, and Vision-Language Models

**arXiv ID:** 2606.26566 | [PDF](https://arxiv.org/pdf/2606.26566v1)

**作者:** Abrar Alotaibi `[一作]` (King Fahd University of Petroleum & Minerals), Moataz Ahmed `[通讯]` (King Fahd University of Petroleum & Minerals)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了扩散模型在文本、图像分类器、视觉语言模型等多模态中的对抗攻击、对抗防御和评估，整合四条分离的研究轨迹，提出统一的六类扩散角色分类法和五维评估框架，并公布了50篇相关论文的元数据与实验结果。

**💡 创新点**

创新点：①将文本、图像、VLM以及扩散LLM攻击与防御四条研究路线统一到一个框架；②提出六类扩散在攻击管线中的具体角色与威胁模型维度；③构建统一的五维评估框架（成功率、迁移性、查询预算、困惑度、对防御逃逸），实现跨模态对比；④系统分析LLM侧攻击的五大弱点并给出研究议程。

**🔧 技术方法**

使用的技术包括：DDIM反演、离散/连续扩散、分数/分类器引导、Gumbel‑Softmax松弛、时间旅行采样、分块反向传播、强化学习对抗奖励等；威胁模型涵盖白盒、灰盒、黑盒和生成流访问，查询预算从单次推理到数千重启不等。

**📊 数据集**

主要数据集：文本侧使用 AdvBench、HarmBench、JailbreakBench；图像侧使用 ImageNet、CIFAR‑10、CelebA、FFHQ；VLM侧使用 MM‑SafetyBench、HADES、VLJailbreakBench、LAION‑400M 等；目标模型包括开源 LLMs（Llama‑3、Vicuna 等）和闭源前沿模型（GPT‑4o、Claude‑3.5）。

**📈 对比分析**

比较方法：与传统基于梯度的 suffix 攻击、遗传算法、LLM 迭代重写、对抗前缀等非扩散基线进行对比。性能方面，四篇文本扩散攻击在公开 LLM 上的成功率可达 90%+，但对闭源前沿模型的迁移率显著下降；在图像与 VLM 侧，扩散攻击在公开模型上取得接近 100% 的成功率，但在受保护模型上效果衰减；防御逃逸评估有限，现有防御在未进行自适应攻击时表现较好。

**⚠️ 局限性**

主要限制：①攻击多依赖白盒梯度，缺乏对闭源模型的高效攻击；②离散扩散在文本攻击中极少使用；③对防御逃逸、可解释性与多轮对话攻击的评估不足；④评估标准不统一，ASR 解释方式多样；⑤缺乏针对扩散攻击的标准基准与可复现性，导致跨论文结果难以直接比较。

---

## 234. From Hallucination to Grounding: Diagnosing Visual Spatial Intelligence via CRISP

**arXiv ID:** 2606.26535 | [PDF](https://arxiv.org/pdf/2606.26535v1)

**作者:** Zhixing Li `[一作]` (Chalmers University of Technology), Yinan Yu `[通讯]` (Chalmers University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CRISP，一种结合问答与3D场景图生成的结构诊断评测范式；

**💡 创新点**

通过跨任务一致性协议揭示视觉空间推理的感知-推理断层，并引入可视化的“语义捷径”“感知-推理断层”等失败模式；

**🔧 技术方法**

使用多模态大语言模型、3D场景图构造（SGC）、一致性得分（Self‑Consistency Score）以及量化的距离、尺寸与关系评估指标；

**📊 数据集**

基于nuScenes与ScanNet++构建的1,162幅室内/室外单视图场景，产生9,839道空间问答；

**📈 对比分析**

与13款商业与开源VLM进行零样本对比，结果显示专有模型在推理与一致性上优于开源模型，但都受感知与结构对齐瓶颈限制；

**⚠️ 局限性**

局限于静态单视图，缺乏动态/多视角推理，感知-推理断层及模块信任偏差等结论仅为经验性行为模式。

---

## 235. \textsc{DiARC}: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models

**arXiv ID:** 2606.26530 | [PDF](https://arxiv.org/pdf/2606.26530v1)

**作者:** Yuxuan Yang `[一作]` (Shenzhen University), Yile Wang `[通讯]` (Shenzhen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通过构造负样本并使用偏好对齐（Preference Alignment）来提升大语言模型在ARC（Abstraction and Reasoning Corpus）类任务上的推理能力；

**💡 创新点**

创新点在于三种层次的负样本生成策略（输出层视觉变换、DSL层规则倒转、任务特定规则编辑）以及将偏好对齐方法直接应用于ARC任务，提升模型对正确输出与近似错误输出的区分；

**🔧 技术方法**

采用了直接偏好优化（DPO）训练框架，利用LoRA微调、生成的负样本对齐来实现后训练优化；

**📊 数据集**

在六个ARC相关基准上进行实验，包括ARC-AGI-1、ARC-AGI-2、MiniARC、ConceptARC、1D-ARC、ARCcommunity；

**📈 对比分析**

与SFT基础模型、其他开源与闭源模型对比，DiARC在所有基准上均获得平均提升2.48点，Qwen3+DiARC在ARC-AGI-1、MiniARC、ConceptARC上准确率超过96%，优于现有闭源与专用模型；

**⚠️ 局限性**

局限性在于负样本构造高度依赖于ARC特定资源（RE-ARC生成器、DSL程序），仅ARC-AGI-1可使用全部三种策略；其它基准只能使用输出层视觉变换；负样本自动生成的质量不如人工标注，可能影响最优性能。

---

## 236. Radical AI Interpretability

**arXiv ID:** 2606.26523 | [PDF](https://arxiv.org/pdf/2606.26523v1)

**作者:** Daniel A. Herrmann `[一作]` (University of North Carolina), Benjamin A. Levinstein `[通讯]` (University of Illinois)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个将人工智能系统解释为具有信念、欲望与意义的代理人的框架，基于哲学的激进解释传统和机器可解释性工具，阐述了如何从系统的计算事实推断其心理态度，并提出了评价解释成功的标准。

**💡 创新点**

创新点在于：①将激进解释与机制可解释性融合，形成一个可操作的解释框架；②提出信念、欲望与命题结构的整体一致性原则；③区分解释主义与表征主义两种方法，并给出各自的评估准则；④将决策理论（尤其Jeffrey‑Bolker框架）与模型行为关联，提供从偏好推断信念与欲望的正式方法。

**🔧 技术方法**

主要技术包括：
- 线性探测器（probes）和方向追踪（truth directions）
- 方向操纵（steering vectors）和干预技术
- 稀疏自编码器、因果追踪、激活补丁等机制可解释性方法
- 决策理论的表示定理（Jeffrey‑Bolker）用于从行为推断信念与欲望。

**📊 数据集**

未采用单一标注数据集；文中以多种实验示例（如Othello GPT、LLM链式思维）展示方法的可行性，主要依赖内部激活、模型生成的文本及其已知真值标签（如真/假命题）进行探测与验证。

**📈 对比分析**

比较方式主要是理论层面的：与传统行为解释、表征主义方法及决策理论的表示定理进行对比；缺乏数值实验或准确度指标，评估更多基于一致性、可解释性与理论连贯性而非性能指标。

**⚠️ 局限性**

局限性包括：
- 对信念与欲望的定义仍保持哲学抽象，缺乏统一的经验验证；
- 解释主义与表征主义的区分在实际实践中仍模糊，难以确定哪种更优；
- 该框架对大模型内部结构的探索尚未系统化，依赖于可解释性工具的成熟度；
- 对新兴多代理、工具使用等动态场景的适用性尚未得到充分实证验证。

---

## 237. Assessing Post-Reform Changes in Risk Disclosure Quality with a Multidimensional Text Analysis Approach

**arXiv ID:** 2606.26522 | [PDF](https://arxiv.org/pdf/2606.26522v1)

**作者:** Nobuhiro Aikawa `[一作]` (University of Tsukuba), Mitsuo Yoshida `[通讯]` (University of Tsukuba)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过构建包含六维度指标的纵向文本分析框架，对日本2019年披露改革后十年上市公司风险披露文本质量进行评估。

**💡 创新点**

创新点在于引入跨章节相关性指标以衡量风险披露与管理策略的主题一致性，并结合shift function和指标相关性分析揭示多维度变化。

**🔧 技术方法**

采用GiNZA等日语NLP工具进行实体识别、句子分割、可读性公式计算，并运用配对t检验、shift function和指标相关性三重统计方法。

**📊 数据集**

使用EDINET系统公开的19,770个企业年度报告（1,977家公司，2015-2024年）作为文本语料。

**📈 对比分析**

通过配对t检验得到平均水平变化，shift function揭示不同分位数的差异，指标相关性矩阵说明量化与质量指标之间的相互关系；结果显示出现量化-可读性权衡、结构-描述性失衡与市场细分差异。

**⚠️ 局限性**

局限在于方法仅为描述性缺乏因果推断，相关性指标基于TF‑IDF可能忽略语义细节，实体识别和可读性公式受语言特性限制，Growth市场样本不足，COVID‑19冲击可能混淆后期变化。

---

## 238. Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge

**arXiv ID:** 2606.26511 | [PDF](https://arxiv.org/pdf/2606.26511v1)

**作者:** Neeraj Yadav `[一作]` (MemStrata), Neeraj Yadav `[通讯]` (Called It Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MemStrata，一种在检索增强生成（RAG）基础上加入时间有效性的记忆机制，能够在知识随时间演化时保持最新事实，避免返回过时信息。

**💡 创新点**

创新点在于：①引入确定性（subject‑relation‑object）取代相似度阈值的取代规则；②使用双时间账本（bi‑temporal ledger）在写入时自动退休旧事实；③构建 marker‑free 的演化基准，消除标签干扰，真正考察时间有效性。

**🔧 技术方法**

技术实现：RAG 作为检索层；文本哈希去重；三元组抽取与键归一化；确定性取代规则；双时间账本；在读取时仅做相似度检索并过滤已退休事实；所有计算均在本地 7B 模型和嵌入模型上完成，无读取路径 LLM 调用。

**📊 数据集**

数据集：六个基准，两个静态（项目事实 QA、跨会话对话）和四个演化 marker‑free（函数重命名、配置值变更、依赖升级、API 结构演化），每个演化基准包含同一事实的旧值与新值对。

**📈 对比分析**

与传统 RAG 对比：在静态基准上保持相同的准确率；在演化基准上，MemStrata 的准确率从 0.95–1.00 提升至 RAG 的 0.20–0.47；最显著的是 stale‑fact‑error（返回过时事实的比例）从 RAG 的 15–40% 降到约 0%，且检索延迟保持在 ~2.1 秒，几乎不受 LLM 重新排序/验证的 8× 延迟影响。

**⚠️ 局限性**

局限性：①演化基准为结构化单值模板，无法覆盖多值或自然语言对比；②抽取质量是关键，低质量抽取会导致失效；③时间戳信息仅通过写入顺序代理，未使用真实时间标签；④评估使用单一 7B 模型和单一判定者，可能影响普适性；⑤规模有限，尚未验证在大规模真实知识库上的性能。

---

## 239. Learning Probabilistic Filters with Strictly Proper Scoring Rules

**arXiv ID:** 2606.26497 | [PDF](https://arxiv.org/pdf/2606.26497v1)

**作者:** Eviatar Bach `[一作]` (University of Reading), Andrew Stuart `[通讯]` (California Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种新的数据同化方法——严格正确分数集合滤波器（PSEF），通过学习一个可泛化的分析算子来近似贝叶斯滤波分布，训练过程只需要模拟得到的状态-观测对。

**💡 创新点**

创新点包括：①利用严格正确的评分规则（能量分数）构造损失函数，使模型在整个概率分布上被奖励；②将分析算子设计为可交换的、基于 transformer 的神经算子，既能处理可变大小的集合，又能自然表达全局统计特征；③在理论上证明，在可实现性假设下，损失最小化即回归真实贝叶斯滤波分布；④通过预训练-微调策略实现不同粒子数下的迁移与扩展。

**🔧 技术方法**

技术手段包括：严格正确分数规则（能量分数）、Transformer/Set Transformer 架构、均值场滤波理论、粒子滤波用于生成训练数据、传统的 EnKF/ESRF/LETKF/ IEnKF、Bootstrap 粒子滤波（BPF）作基准，以及对 Lorenz‐63/96、doubling‑angle、线性高斯模型的数值实验。

**📊 数据集**

数据集主要是从四个动力学模型（线性高斯、doubling‑angle、Lorenz‑63、Lorenz‑96）通过模拟产生的状态-观测轨迹；BPF（10^6 粒子）被用作“真”滤波分布的近似参考。

**📈 对比分析**

与经典 EnKF、ESRF、LETKF、IEnKF 以及另一种基于 EnKF 修正的机器学习方法 MNMEF 进行比较。评估指标包括对 BPF 的切片能量距离（SED）、能量分数、阶数校正指标 RHVar 等。实验显示 PSEF 在非高斯、多模态、以及非线性观测场景下均显著优于基准，且在相同或更小粒子数下即可达到甚至超过传统方法的性能；在学习率变化下表现更稳定。

**⚠️ 局限性**

局限性包括：①训练仅能使用模拟轨迹，若真实系统无法高质量仿真则受限；②理论证明依赖可实现性假设与无穷粒子极限，实际有限粒子和有限样本可能导致偏差；③能量分数的双重求和导致 O(N²) 的计算复杂度，影响大规模系统；④对高维系统仍存在数值稳定性和收敛性问题；⑤需在不同粒子数之间进行微调，模型在极大/极小粒子数下的泛化尚未完全验证。

---

## 240. Almost Optimal Multiple Source Shortest Paths and Reachability

**arXiv ID:** 2606.26554 | [PDF](https://arxiv.org/pdf/2606.26554v1)

**作者:** Barna Saha `[一作]` (University of California San Diego), Christopher Ye `[通讯]` (University of California San Diego)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套针对无向/有向图中从少量源点到全图求最短路和可达性的高效算法。

**💡 创新点**

创新点在于构造了低直径小邻域分解，并利用该分解与快速矩阵乘法相结合，首次实现了多源最短路与可达性问题与布尔矩阵乘法时间上等价的算法。

**🔧 技术方法**

主要技术包括快速矩阵乘法（尤其是矩阵的非方阵乘）、布尔与(min,+)矩阵乘、低直径小邻域分解、递归分治与分组松弛等。

**📊 数据集**

实验与评测采用理论复杂度分析，不依赖具体数据集；所有结果均为上界分析。

**📈 对比分析**

相较于先前最优的 O(min(n^ω, n^2+σ)) 方案，本工作在任意 σ 取值下达到 O(n^ω(σ,1,1))，并在 hop‑set、+4 emulators、shortcut set 等应用中将跑时分别从 O(n^{7/3}) 降到 O(n^{2.043})、O(n^{2.132}) 与 O(n^{2.064})。

**⚠️ 局限性**

局限性包括算法仍是随机化（低直径分解需要随机），仅适用于稠密图；对一般有向图的加权 MSSP 仍未达到同等最佳复杂度，并且实际性能高度依赖矩阵乘法常数。

---

## 241. Testing Equivalence to the Hamiltonian Cycle Polynomial

**arXiv ID:** 2606.26653 | [PDF](https://arxiv.org/pdf/2606.26653v1)

**作者:** Agrim Dewan `[一作]` `[通讯]` (Indian Institute of Science), Agrim Dewan (Indian Institute of Science)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究Hamiltonian Cycle多项式HC的等价性测试问题，并给出了一个随机多项式时间的ET算法，首次在所有域上实现对HC的高效等价性判断。

**💡 创新点**

创新点包括：①对HC的李代数与对称群进行完整刻画；②证明HC在大部分域上由置换与尺度变换生成，但与对称性无判定性；③利用HC的自下归约性证明其由电路恒等式表征，从而实现电路测试和翻转定理；④在算法中巧妙地规避了传统使用“两个变量差异”技巧，改用基于李代数与极分式系数的恢复方法。

**🔧 技术方法**

主要技术：李代数与对称群理论、极分式系数分析、随机黑盒多项式求导、线性代数求解、下归约与电路恒等式构造，以及对称性与矩阵对角化的组合。

**📊 数据集**

无实验数据集，全部结果为理论证明与算法复杂度分析。

**📈 对比分析**

与已知的Perm多项式等价性测试算法相比，算法同样在随机多项式时间内完成，但不依赖于特征为2的域；相较于电路测试问题，该方法可在更广泛的域上实现，且通过翻转定理提供了对电路复杂度的上界探测。

**⚠️ 局限性**

局限性：需要域的大小满足|F|>3n⁵且特征为0或>n；对n=4的情况需单独处理；算法是随机化的；不适用于所有域（例如特征为2或极小域）；对称性不具判定性，故在某些场景下仍需电路恒等式或下归约的支持。

---

## 242. LayersReg: A Layer-by-Layer Progressive Regressor for Reliable Intraoperative 3D/2D Registration

**arXiv ID:** 2606.26647 | [PDF](https://arxiv.org/pdf/2606.26647v1)

**作者:** Xiyuan Wang `[一作]` (Southeast University), Feng Yin `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了LayersReg框架，实现了3D/2D配准的递进式回归过程。

**💡 创新点**

创新点在于将回归视为内部逐层优化，融合3D解码器、双相关门、趋势感知模块和Mamba式回归骨干，兼具3D解剖感知与特征残差反馈。

**🔧 技术方法**

使用深度卷积自编码器、交叉注意力的深度特征调制、DCG、TPM、Mamba网络、节点回归和自监督重建损失。

**📊 数据集**

在七个公开数据集上评测，包括CTSpine1K、VerSE、DeepFluoro、Ljubljana、RESECT、MRI-CT、CAMUS等，涵盖X‑ray/CT与S2V任务。

**📈 对比分析**

与多种SOTA回归、点云/形状标记方法相比，LayersReg在旋转误差0.68°/0.73°、平移误差1.41mm/1.55mm，NCC>0.96，SSIM>0.93，成功率≥95%，实现实时（<0.2s）配准。

**⚠️ 局限性**

目前仅支持刚性配准，且对极端遮挡和大位移仍有限制；未来需扩展到弹性配准与更复杂场景。

---

## 243. An Evaluation of Decentralized Group Formation Techniques for Flying Light Specks

**arXiv ID:** 2606.26645 | [PDF](https://arxiv.org/pdf/2606.26645v1)

**作者:** Hamed Alimohammadzadeh `[一作]` (University of Southern California), Shahram Ghandeharizadeh `[通讯]` (University of Southern California)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出并评估了四种去中心化分组技术（SimpleR、RS、VNS、CANF）用于飞行灯光颗粒（FLS）的群组形成。

**💡 创新点**

创新点在于引入简单随机（SimpleR）与最近可用邻居优先（CANF）两种新启发式，并通过可扩展的Outring构造合适的合成点云验证算法。

**🔧 技术方法**

技术上采用点云邻居发现、异步通信、加权子图搜索、局部搜索（VNS）以及距离排序策略。

**📊 数据集**

使用的数据集包括基于Outring生成的稀疏与稠密合成点云以及真实场景点云（棋盘、滑板、赛车）。

**📈 对比分析**

通过比较响应时间、组数、平均组内距离与网络带宽，发现RS在小组大小（G≤5）时最快且组数最多，CANF和SimpleR在大组大小（G≥10）时更优，性能随点云密度和组大小显著下降。

**⚠️ 局限性**

主要限制是大规模或稠密点云下算法耗时长、CANF对带宽需求高、部分方法无法构造任何组以及缺乏增量组构建机制。

---

## 244. Position Rebinding Cache Reuse: Replay-Free Visual Revisiting for Interleaved Multimodal Reasoning

**arXiv ID:** 2606.26631 | [PDF](https://arxiv.org/pdf/2606.26631v1)

**作者:** Mengzhao Wang `[一作]` (Sun Yat-sen University), Chongjun Tu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Position Rebinding Cache Reuse (PRCR)，一种通过重绑定历史视觉 KV 缓存实现无视觉标记重放的多模态链式推理框架。

**💡 创新点**

创新点在于利用原始未加位置编码的视觉 KV 缓存，并在重访时重新分配位置以消除位置绑定导致的自回归崩溃，达到与重放相同或更优性能的同时显著降低计算量。

**🔧 技术方法**

采用 Transformer 的 RoPE 位置编码、Raw Visual Evidence Memory、位置重分配和轻量级键重绑定等技术实现缓存重构，避免视觉 token 的重复前向传播。

**📊 数据集**

使用 M^3CoT、MathVista、MMStar、MMMU 四大多模态推理基准进行评测。

**📈 对比分析**

与基线和 Token‑Replay 进行对比，PRCR 在各模型上平均提升 2–5% 准确率，且视觉重访 FLOPs 降低数千倍，内存开销仅 0.3–0.6%。

**⚠️ 局限性**

局限性是仅能重用预填充阶段生成的视觉 KV 缓存，无法处理动态缩放或局部放大后的视觉片段，需要进一步的多尺度预填或尺度感知重绑定来扩展。

---

## 245. Reviving Reflection-in-Action: Instilling Designerly Thinking in AI-Supported Ideation through Multimodal Prompting

**arXiv ID:** 2606.26626 | [PDF](https://arxiv.org/pdf/2606.26626v1)

**作者:** Samangi Wadinambiarachchi `[一作]` (University of Melbourne), Greg Wadley `[通讯]` (University of Melbourne)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了三种输入模态（文字、草图、草图+标签）对设计学生与AI创意工具交互的表达意图、创意支持与发散思维表现的影响

**💡 创新点**

首次系统性探讨将产品低到中等保真度的“生产性摩擦”嵌入AI生成流程，并引入混合模态以平衡低门槛与创意抑制的权衡

**🔧 技术方法**

基于Next.js/Python搭建的SketchifAI原型，使用Stable Diffusion v1.5 与 ControlNet‑Scribble 进行草图解析与图像生成，并通过后端 Prompt 维持低到中等保真度

**📊 数据集**

收集9名设计专业学生在三种模态下完成的177个logo草图，结合专家评审（流畅度、变异度、原创性、质量）以及自评问卷（CSI、UMUX‑Lite）

**📈 对比分析**

采用贝叶斯多级模型与累积分布似然对比模态差异，结果显示草图模态在流畅度上有轻微提升，标签模态降低原创性；文本模态在使用体验与创意支持方面优于草图

**⚠️ 局限性**

样本量小且仅来自一所大学的AI熟悉学生，实验为一次性快速构思任务，未对长期使用与不同AI架构的适用性进行评估

---

## 246. Discovering Millions of Interpretable Features with Sparse Autoencoders

**arXiv ID:** 2606.26620 | [PDF](https://arxiv.org/pdf/2606.26620v1)

**作者:** XinYang He `[一作]` (Alibaba Group Holding Limited), Lin Qu `[通讯]` (Alibaba Group Holding Limited)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

发布了覆盖 Qwen3-1.7B、4B、8B 的 Sparse Autoencoder（SAE）套件，用于解码指令调优模型的稀疏特征。

**💡 创新点**

创新点在于首次提供完整层级 SAE，覆盖残差流、MLP 与注意力输出，并展示稀疏-精度权衡及对模型行为的可解释性。

**🔧 技术方法**

采用 JumpReLU SAEs 结合 L0 稀疏正则，使用 16K/65K 词典训练，并在 NVIDIA H20‑3e GPU 上完成大规模训练。

**📊 数据集**

训练数据来自 FineWeb‑Edu，评估使用其 10% 子集；拒绝实验使用 WildGuard、XSTest 与 Mix 三个数据集。

**📈 对比分析**

通过 DLL 与 FVE 指标评估重构与模型恢复性能，结果显示残差流和 MLP 的 SAE 能更好恢复性能，且特定 SAE 特征可有效驱动模型拒绝行为，拒绝率显著提升。

**⚠️ 局限性**

局限性包括仅对 1.7B/4B 进行系统评估、8B 仅部分残差层、训练成本高、拒绝实验仅覆盖一种行为，且对特征可解释性与其他能力的泛化尚未验证。

---

## 247. A Closed-Form 4-DoF Inter-Robot Pose Estimator using Bearing-only Measurements

**arXiv ID:** 2606.26616 | [PDF](https://arxiv.org/pdf/2606.26616v1)

**作者:** Qixin De `[一作]` (Shanghai Jiao Tong University), Danping Zou `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种闭式4-DoF的相对姿态估计方法，只利用机器人之间的方位角测量与IMU驱动的VIO里程计，能够在无需深度信息的情况下快速得到机器人间的位姿关系。

**💡 创新点**

创新点包括：① 将传统的非线性三角约束松弛为线性问题，并通过误差投影消除距离变量，得到全闭式解；② 对系统观测性进行严格分析，识别出两类会导致可观测性退化的运动模式；③ 引入自适应观测性测试模块，动态决定何时进行翻译估计，避免在可观测性不足时产生错误估计。

**🔧 技术方法**

核心技术包括：闭式线性最小二乘求解、总最小二乘（TLS）与奇异值分解（SVD）用于翻译估计；投影矩阵将误差投射到切平面；IMU驱动的VIO提供里程计；YOLO+SORT实现实时目标检测与跟踪；观测性分析与条件数/奇异值判据用于在线检测。

**📊 数据集**

实验数据：1）仿真数据——随机生成5个航点后用B样条拟合轨迹，采样100帧得到方位角与里程计；2）真实数据——三架搭载Intel RealSense D435i与OAK‑FFC‑4P的四旋翼，VIO频率200 Hz、方位角检测频率10 Hz，VICON提供地面真值。

**📈 对比分析**

与Algebraic（4/6-DoF）、SDP‑Graph、SDP‑Cert等方法对比：在仿真中，本方法在所有噪声水平下均取得最低的旋转误差与平移误差；计算时间仅为几毫秒，明显快于SDP方法（数百到上千毫秒）。在实测中，本方法的平移误差始终最低，旋转误差虽略高于6-DoF方法，但在观测性不佳的场景下仍保持稳定，且自适应观测性测试使得估计提前完成。

**⚠️ 局限性**

局限性：① 仅估计yaw和三维平移，无法恢复roll与pitch，若VIO误差累积严重会影响结果；② 长时间处于可观测性不足的运动模式（如列线或形状保持）仍会导致未观测维度误差累积；③ 对极端噪声环境下的鲁棒性需要进一步提升，且目前尚未结合主动可观测性规划实现更强的实时可靠性。

---

## 248. Content-Based Smart E-Mail Dispatcher Using Large Language Models

**arXiv ID:** 2606.26593 | [PDF](https://arxiv.org/pdf/2606.26593v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 249. Moebius: Serving Mixture-of-Expert Models with Seamless Runtime Parallelism Switch

**arXiv ID:** 2606.26607 | [PDF](https://arxiv.org/pdf/2606.26607v1)

**作者:** Shaoyu Wang `[一作]` (University of Southern California), Seo Jin Park `[通讯]` (University of Southern California)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种能够在MoE模型推理过程中实时切换专家并行（EP）和张量并行（TP）的系统，能够在不中断正在执行的请求的情况下在两种并行布局间无缝切换。

**💡 创新点**

核心创新是将EP与TP视为同一模型的不同数据分配，仅需移动权重和KV缓存的所有者即可实现切换；并通过统一内存管理、融合直接传输核以及保持CUDA图的双模态运行时实现高效、低延迟的切换。

**🔧 技术方法**

使用统一内存管理（UMM）预分配连续GPU缓冲，融合GPU间直接传输核实现专家权重和KV缓存的高效重分布，保持CUDA图与通信组以避免重捕获，并采用基于阈值的切换策略。

**📊 数据集**

在8×H200 GPU上对Qwen3-235B-A22B模型进行评估，使用Burst在线服务请求序列和DeepMath RL回放工作负载来验证系统性能。

**📈 对比分析**

与固定TP和EP两种布局对比，采用吞吐量、首令延迟（TTFT）和每步推理时间等指标；在在线服务中，TTFT降低至3.1 s（约3倍提升），TPOT下降至37 ms；在RL回放中，每步平均提升1.16–1.25×，单次切换耗时215–434 ms，内存开销仅2.4%。

**⚠️ 局限性**

目前仅在单机NVLink域内实现高效切换，跨节点时需退回至NCCL；对KV头数不足的容量检查仍基于简单阈值；对更复杂模型结构（如多头注意力、分层专家等）的支持需进一步泛化。

---

## 250. Fast Computation and Optimization for Opinion-Based Quantities of Friedkin-Johnsen Model

**arXiv ID:** 2606.26601 | [PDF](https://arxiv.org/pdf/2606.26601v1)

**作者:** Haoxin Sun `[一作]` (Fudan University), Zhongzhi Zhang `[通讯]` (Fudan University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出利用部分有根森林采样方法，快速计算并优化 Friedkin‑Johnsen 模型中的意见量化指标；

**💡 创新点**

创新点在于将有根森林采样扩展为局部的部分有根森林，显著降低采样时间，并以此实现从线性到亚线性时间的复杂度提升；

**🔧 技术方法**

核心技术包括吸收随机游走、循环消除、部分有根森林采样、基于采样的估计与优化算法（PF‑QE、PF‑OpMin、PF‑PDMin）；

**📊 数据集**

实验使用的真实网络数据集包括 Delicious、YouTube、Pokec、Orkut、Livejournal、Twitter，节点数从几十万到上千万；

**📈 对比分析**

与 LapSolver、LazyWalk、Fast 以及 FastGreedy 等基线方法比较，PF‑QE 在准确率低于 1.5% 的同时跑时显著更快；PF‑OpMin 与 Fast 的效果相当但速度快 30–100 倍；PF‑PDMin 在速度上比 FastGreedy 提升 10–35 倍且效果更优；

**⚠️ 局限性**

局限性在于目前仅能处理线性或可线性化的目标，无法直接处理包含二次项等非线性优化问题；

---

## 251. LLM-based Models for Detecting Emerging Topics in Service Feedback

**arXiv ID:** 2606.26595 | [PDF](https://arxiv.org/pdf/2606.26595v1)

**作者:** Mahsa Tavakoli `[一作]` (University of Western Ontario), Cristián Bravo `[通讯]` (University of Western Ontario)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将域专用的微调和量化大型语言模型与统计趋势检测相结合，构建了一个多语言（英语/法语）的税务服务反馈分析系统，自动将反馈分类为13个服务质量要素（SQE），并在不同人口统计群体中识别新兴、持续和消退的主题趋势。

**💡 创新点**

创新点在于：①将人机协作（human‑in‑the‑loop）嵌入到数据预处理、模型微调和结果验证全过程，降低模型自发性错误；②使用LoRA/PEFT与GPTQ 4‑bit量化在保持精度的同时显著压缩模型尺寸；③采用分段逻辑回归配合自助法置信区间，定量评估各人口统计因素对主题出现率的时间演化，首次以公平性为视角量化服务差异；④在双语环境下实现跨语言的一致性评估。

**🔧 技术方法**

主要技术包括：Transformer‑based LLM（Zephyr‑7B‑β、Mistral‑7B‑Instruct‑v0.2）微调与量化、LoRA/PEFT参数高效微调、GPTQ 4‑bit量化、BERT/Roberta NER+正则规则的脱敏、Logistic 回归与自助法的趋势检测、相似度匹配与双样本 t‑检验进行模型评估。

**📊 数据集**

使用了 8,161 条税务服务反馈（6,515 条英语，1,646 条法语），每条反馈已标注 13 个服务质量要素和人口统计信息（年龄、性别、首选语言）。微调训练集来自税务服务内部的业务手册和历史反馈。

**📈 对比分析**

通过与人工标注的相似度（匹配比例）和双样本 t‑检验比较模型与专家的差异。预训练模型的相似度为 24.27%，微调+量化模型为 66.64%，差异在 95% 置信水平下无统计学意义（t = -1.968，p = 0.085）。逻辑回归的系数差异用于判定主题趋势，模型能准确捕捉各人群中新兴、持续或消退的服务关注点。

**⚠️ 局限性**

局限性包括：①仅使用单一年度数据，缺乏跨年或季节性趋势分析；②计算资源受限导致微调深度和模型规模（13B 以内）受限，无法充分探索更大规模模型的潜力；③多语言覆盖仅限英语与法语，未涉及其他加拿大官方或本地语言；④人机评估样本规模有限，可能影响评估稳健性。

---

## 252. Preference Optimization Drives Monoculture in LLM Prediction Markets

**arXiv ID:** 2606.26583 | [PDF](https://arxiv.org/pdf/2606.26583v1)

**作者:** James Begin `[一作]`, Archana Vaidheeswaran `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在由大语言模型（LLM）组成的预测市场中，探讨了偏好优化（DPO）对错误相关性的影响，并评估了不同多样性策略和对抗性鲁棒性。

**💡 创新点**

创新点在于：①通过控制实验清晰证明偏好优化是导致同一模型代理之间高度相关错误（ρ≈0.70）的根本原因；②首次量化同类模型“单一文化”导致的有效预测器数量急剧下降（≈1.4）；③提出并验证交叉模型、多温度和角色多样化作为降低相关性和提升市场性能的可行缓解方案；④发现LMSR本身的凸成本机制能自然阻止对抗性交易。

**🔧 技术方法**

主要技术包括：LMSR 价格发现机制、Direct Preference Optimization (DPO) / RLHF 对齐训练、温度采样、角色多样化、交叉模型混合、对抗性跳过规则以及多代理辩论比较。

**📊 数据集**

使用 TruthfulQA 的二元对比问答数据集（50个问题），并在多个模型规模（8B、70B）与不同模型族（Llama、Qwen2.5、Mistral、GLM-4）上进行实验。

**📈 对比分析**

与单一独立代理的基准对比发现：单一模型10名代理的市场准确率仅为67.6%（与单个代理70.2%相当），有效独立预测器仅约1.4。交叉模型市场将ρ降至0.40，准确率提升至约70%；角色多样化将ρ降至0.44，保持准确率不变。辩论模式在多数诚实代理情景下优于LMSR，但在对抗情景下表现不佳。

**⚠️ 局限性**

局限性包括：仅在二元QA任务上评估，未覆盖多分类或其他知识领域；模型规模最高到70B，尚未验证更大模型的行为；代理过度自信（0.9–1.0），导致自我惩戒效果为上限；TruthfulQA 的误解焦点可能使相关性被预训练信息放大。

---

## 253. Target-Aware Bandit Allocation for Scalable Surrogate Optimization in Chemical Space

**arXiv ID:** 2606.26657 | [PDF](https://arxiv.org/pdf/2606.26657v1)

**作者:** Mohammad Haddadnia `[一作]` (Dana-Farber Cancer Institute), Haribabu Arthanari `[通讯]` (Dana-Farber Cancer Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 BoBa 框架，通过将化学空间划分为子空间并使用多臂老虎机动态分配推理与评估，解决了超大规模化学库中代理模型推理成本成为瓶颈的问题。

**💡 创新点**

创新点在于将结构感知的分区、基于不确定性的 UCB1 老虎机决策与局部代理优化相结合，首次实现推理成本与优化性能可调节的权衡，并在亿级库上保持接近完整库 BO 的效果。

**🔧 技术方法**

使用了 T5Chem 嵌入的 k‑means 分区、多臂老虎机（UCB1）策略、前馈神经网络代理结合线性化拉普拉斯不确定度估计，以及上界采集函数进行候选选择。

**📊 数据集**

实验数据集包括 Enamine‑5M、Enamine‑S‑3.9M、Enamine‑HTS 以及 ZINC 子集（10⁵–10⁸ 分子），分别针对 CKB、NEDD4、TMK、AmpC 等蛋白进行分子对接评分。

**📈 对比分析**

与完整库 BO、随机子采样以及无结构分区的基线对比，UCB1‑BoBa 在保持约 90% 以上完整 BO 性能的同时，推理成本可降低数倍；在 10⁸ 规模库中相对 AUC 接近完整 BO。

**⚠️ 局限性**

局限性在于需对整个可枚举库先完成特征化与聚类，适用于已列举且能快速聚类的情况；对非平稳奖励或更大规模库仍需改进，并未直接在合成子空间实现。

---

## 254. Zero-Shot Size Transfer for Neural ODEs on Sparse Random Graphs: Graphon Limits and Adjoint Convergence

**arXiv ID:** 2606.26662 | [PDF](https://arxiv.org/pdf/2606.26662v1)

**作者:** Mingsong Yan `[一作]` (University of California, Santa Barbara), Sui Tang `[通讯]` (University of California, Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了图神经差分方程（GNDE）在稀疏随机图上的规模迁移可行性，并建立了从有限图到无穷图图仿真（Graphon‑NDE）的理论收敛性与DTO/OTD训练一致性。

**💡 创新点**

提出零样本规模迁移原则的量化理论，给出GNDE解与Graphon‑NDE解在稀疏随机图下的轨迹、梯度及参数梯度的收敛速率，并证明DTO与OTD梯度误差随时间步长递减，揭示激活平滑性对收敛的必要性。

**🔧 技术方法**

利用图神经网络（Spectral GNN）参数化的Neural ODE、图仿真（Graphon）框架、随机稀疏图采样、显式欧拉离散化、对偶敏感性（adjoint）分析以及高斯‑链式理论（Chaining）来推导误差界。

**📊 数据集**

实验数据集包括层次化随机块模型（HSBM）图仿真、Tent图仿真、圆形阈值图仿真及秩一幂律图仿真，以及在这些图上四类动力学（线性热、Fisher–KPP、SIS、Consensus）产生的节点时间序列。

**📈 对比分析**

与理论收敛速率对比，实验在不同稀疏程度下验证了 O(1/√(α_n n)) 的前向轨迹误差、O(1/√(α_n n)) 的梯度误差以及 O(1/M) / O(1/M^2) 的DTO/OTD梯度差异，实验结果与理论一致，零样本迁移误差保持在训练误差水平。

**⚠️ 局限性**

局限性在于对稀疏图参数梯度的理论界限过于保守，实际实验表现优于预期；理论依赖激活函数高阶光滑性，且未考虑更复杂的图结构或大规模节点/层深度下的数值稳定性。

---

## 255. Autoformalization of Agent Instructions into Policy-as-Code

**arXiv ID:** 2606.26649 | [PDF](https://arxiv.org/pdf/2606.26649v1)

**作者:** Adam Mondl `[一作]` (Sondera), John H. Brock `[通讯]` (Sondera)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一套自动化管道，将代理指令、工具定义及自然语言政策文档转换为可验证的 Cedar 策略，并在 MedAgentBench 上评估其覆盖率。

**💡 创新点**

创新点在于构建了“Verification Sandwich”多层级生成‑批评循环，将 LLM 生成与硬性 Cedar 语法/逻辑校验相结合，实现高覆盖率且可验证的策略。

**🔧 技术方法**

使用的技术包括 LLM 生成器与评论者（Gemini 3 Pro、Gemini 2.5 Flash）、Cedar 权限语言及其静态分析工具、LLM 作为软评审的批评器，以及基于 MCP 的工具定义。

**📊 数据集**

实验数据集为 MedAgentBench，包含 88 条自然语言政策及其手工实现的 23 条规则。

**📈 对比分析**

通过与 Hong et al. 的手工符号防护在原始、基线和防护条件下对比，Cedar 策略在防护覆盖率上显著提升，尤其在攻击场景下从 55.7% 提升至 85.7%，并在包含 POST 请求的轨迹中实现 100% 的违规拦截率。

**⚠️ 局限性**

局限性包括 Cedar 目前仅支持无状态的请求响应授权，缺乏时间和记忆依赖，难以处理多轮交互与上下文顺序；此外，过度严格的自动化策略可能削弱代理的功能性。

---

## 256. Fast Enumeration of Minimal Removable Sets in Monotone Systems with Application to Core Collapse Analysis

**arXiv ID:** 2606.26639 | [PDF](https://arxiv.org/pdf/2606.26639v1)

**作者:** Kan Shota `[一作]` (Kyoto University), Kazuya Haraguchi `[通讯]` (Tokyo University of Marine Science and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种通用框架，研究如何枚举给定图的k-核的所有最小可移除集合（MinRS）以及一般单调系统的θ-解。

**💡 创新点**

创新点在于将MinRS问题视作单调系统的最小可移除子集问题，引入传播图和种子关系图，通过闭包运算和指向树/功能图的迭代剪枝实现对MinRS的高效枚举。

**🔧 技术方法**

主要技术包括单调系统理论、闭包算子、传播图和SR图构造、指向树/功能图的迭代更新，以及利用“入支配种子”性质进一步压缩时间复杂度。

**📊 数据集**

论文以理论分析为主，并未给出具体实验数据集，而是以无向图和其k-核为例进行算法设计与证明。

**📈 对比分析**

与Boley等人基于强可访问性框架的枚举算法相比，本文在满足入支配种子性质的k-核上实现了从O((n+m)n)降到O((n+m)log n)的时间（或延迟），显著提升。

**⚠️ 局限性**

局限性在于入支配种子性质仅对标准k-核成立，对许多扩展（加权、多层、(k,ℓ)-核等）不成立，且算法仍以O((n+m)n)为上界；另外缺乏实验验证其在大规模实际网络上的性能。

---

## 257. Closing the Quality Gap in Low-Resource Text-to-Speech: LoRA Fine-Tuning of VoxCPM2 for Khmer and Korean

**arXiv ID:** 2606.26618 | [PDF](https://arxiv.org/pdf/2606.26618v1)

**作者:** Phannet Pov `[一作]` (Chungbuk National University), Saksonita Khoeurn `[通讯]` (Chungbuk National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在VoxCPM2模型上训练单一共享LoRA适配器，来改善柬埔寨语和韩语的低资源文本到语音合成质量。

**💡 创新点**

创新点在于使用一个共享LoRA适配器同时服务两种语言并只更新极少参数，证明在低资源语言中只需极少训练参数即可显著提升音质。

**🔧 技术方法**

主要技术包括Tokenizer‑free TTS模型VoxCPM2、低秩适配器LoRA以及基于flow‑matching扩散解码器的训练。

**📊 数据集**

使用了约26小时的合并语料库，其中柬埔寨语来自IDRI内部语料、韩语来自KSS和Common Voice/FLEURS，并通过语言标签进行混合。

**📈 对比分析**

通过与零样本基模型的配对Wilcoxon检验以及MOS听众评测比较，柬埔寨语的平均MOS从3.85提升至4.23，显著提升；韩语未见显著提升，且高秩适配器甚至导致质量下降。

**⚠️ 局限性**

局限性包括语料规模小、评价者人数有限、跨语言比较不完善、适配器共享效果未与单独适配器做对比，以及未评估全量微调上限。

---

## 258. Quantum Mutant Equivalence via Transpilation

**arXiv ID:** 2606.26604 | [PDF](https://arxiv.org/pdf/2606.26604v1)

**作者:** José Campos `[一作]` (Universidade do Porto), Andriy Miranskyy `[通讯]` (Toronto Metropolitan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

基于量子电路的变异测试中，提出了一种轻量级的等价突变检测方法TBE

**💡 创新点**

创新点在于利用量子编译器的转译过程，将原电路与突变电路转译成同一配置下的OpenQASM进行比较，以判定语义等价

**🔧 技术方法**

主要技术是Qiskit转译器、OpenQASM文本比较以及对状态向量的数值基准检测

**📊 数据集**

实验使用MQT基准中的375个量子电路，共生成348,299个存活突变体（覆盖2–30个量子比特）

**📈 对比分析**

与状态向量基准对比，TBE在100次随机转译重复实验中实现100%精度、0.82准确率、0.32召回率，成功识别约29,536个等价突变体（占所有等价突变的32%），并显著降低计算资源消耗

**⚠️ 局限性**

局限性是只能发现约三分之一的等价突变，受转译配置限制；对于不同基门集、优化层级或硬件后端可能需要进一步调优；并且仅在转译器可靠时才有效

---

## 259. Efficient Computation for Diagonal of Forest Matrix via Variance-Reduced Forest Sampling

**arXiv ID:** 2606.26599 | [PDF](https://arxiv.org/pdf/2606.26599v1)

**作者:** Haoxin Sun `[一作]` (Fudan University), Zhongzhi Zhang `[通讯]` (Fudan University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并高效计算有向图森林矩阵对角线元素

**💡 创新点**

提出三种采样算法（SCF、SCFV、SCFV+），通过Wilson算法扩展与方差削减技术实现相对误差保证且时间复杂度线性

**🔧 技术方法**

利用Wilson算法扩展、Friedkin‑Johnsen模型启发的矩阵向量迭代以及新的迭代方程，配合采样方差缩减技术

**📊 数据集**

在KONECT与SNAP公开网络数据集上实验，涵盖节点数从十万到三千万不等的无向与有向图

**📈 对比分析**

与JLT、UST算法对比，在无向图中SCFV+在平均与最大相对误差上均优于其他方法，且在有向图中仍能在数分钟内完成对千万级节点的计算

**⚠️ 局限性**

主要限制为仍需采样，参数选择（采样数l）对精度影响显著；对高度节点或特殊图结构（如带符号或时间演化图）仍需进一步改进

---

## 260. SocialPersona: Benchmarking Personalized Profiling and Response with Multimodal Social-Media Context

**arXiv ID:** 2606.26654 | [PDF](https://arxiv.org/pdf/2606.26654v1)

**作者:** Qinkai Zhang `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了 SOCIALPERSONA 基准，用以评估多模态大型语言模型在从真实社交媒体时间线中推断并运用用户偏好的能力。

**💡 创新点**

创新点在于将真实用户的长时序文本与图像数据与人工验证的兴趣标签结合，提出两任务（画像构建与对话生成）并证明跨模态、跨时长的偏好推理仍是重大挑战。

**🔧 技术方法**

使用 LLM 辅助的兴趣提取与校验流水线、Gemini、GPT‑4o‑mini、Qwen 等多模态模型以及人类与 LLM 评审对话质量。

**📊 数据集**

数据集由 171 名普通用户的两年内时间线构成，平均 176 条帖子、130 张图片，涵盖 2,597 个兴趣标签，分布于体育、娱乐、游戏、餐饮、旅行、摄影、宠物七大领域。

**📈 对比分析**

通过域激活 F1、兴趣标签 F1、对话覆盖度/具体度/流畅度等指标比较模型，结果显示模型普遍过度泛化、对文本分散兴趣识别不足、近期兴趣推理能力差，整体性能低于预期。

**⚠️ 局限性**

限制包括仅使用图像字幕而非原图、仅聚焦非敏感兴趣、数据规模有限，以及模型缺乏足够的跨模态和时序推理能力。

---

## 261. FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics

**arXiv ID:** 2606.26636 | [PDF](https://arxiv.org/pdf/2606.26636v1)

**作者:** Langyi Chen `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于像素生命周期的事件相机模拟器FracEvent，通过分形松弛电压记忆实现连续时间阈值跨越来生成ON/OFF事件；

**💡 创新点**

创新点在于将事件生成分解为输入亮度轨迹与传感器侧转换两部分，采用多模态分形松弛电压记忆、连续时间阈值求解与保留后续记忆状态，显著提升事件时序、极性平衡与重复触发的真实性；

**🔧 技术方法**

使用的技术包括分形松弛电压动力学、闭式模式更新、局部二分搜索求解阈值跨越、基于参考电压的极性阈值分离，以及在不同输入轨迹（渲染、视频插值、帧序列）上的通用实现；

**📊 数据集**

实验数据集包括DAVIS240C、DAVIS346、GoPro高帧率视频、HQF、MVSEC事件+帧数据；

**📈 对比分析**

比较方法：在匹配窗口内使用事件计数比、IEI距离、极性误差和时间表面相关度评估事件流质量；在下游任务中使用固定的图像重建（E2VID）和光流估计（EV-FlowNet）训练与测试，FracEvent在所有指标上均优于ESIM、v2e、DVS-Voltmeter，取得最低MSE/LPIPS和最接近真实事件的平均AEE；

**⚠️ 局限性**

限制在于模拟仅解决传感器侧动态，仍受输入亮度轨迹质量限制；插值、渲染、曝光、校准等光学和机械误差未被完整建模，导致与真实事件仍存在一定差距。

---

## 262. Temporally Consistent Label Interpolation for Robust Surgical Multi-Task Learning under Challenging Conditions

**arXiv ID:** 2606.26634 | [PDF](https://arxiv.org/pdf/2606.26634v1)

**作者:** Garam Kim `[一作]` (Korea Institute of Science and Technology), Juyoun Park `[通讯]` (Korea Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出流引导的标签插值框架FAROS，结合SAM2分割与RAFT光流在稀疏标注的手术视频中生成稠密伪标签，并将这些伪标签与统一Transformer多任务模型联合训练，实现手术阶段、步骤、预测、工具分割与动作识别的整体场景理解。

**💡 创新点**

创新点在于利用光流一致性检测与重提示纠正SAM2在复杂手术环境下的传播误差；通过稠密化稀疏标注实现时空标注不匹配的平衡，并将插值结果直接用于多任务联合学习，显著提升跨任务表示共享。

**🔧 技术方法**

主要技术包括SAM2提示分割、RAFT光流估计、Mask2Former区域提议、MViT Transformer骨干、双加载器多任务训练策略以及跨任务条件机制。

**📊 数据集**

使用GraSP、MISAW（扩展版含分割标注）和AutoLaparo三大手术视频数据集，并在DAVIS 2017上验证插值质量。

**📈 对比分析**

在DAVIS 2017 sparse-GT插值协议下，FAROS相较于SAM2和传统VOS方法提升约1.5–2%在J&F指标；在GraSP与MISAW上，与单任务基线和无插值多任务模型比较，FAROS+MTL在阶段/步骤/工具分割/动作识别等多项指标平均提升约3–5%，尤其在步骤识别和工具分割上表现突出。

**⚠️ 局限性**

局限在于伪标签仍受光流误差和复杂外观变化影响，导致极少标注或高动态场景下的工具分割误差；多任务框架对稀疏关键帧的依赖较高，且在实时部署时仍需权衡计算开销与延迟。

---

## 263. Simulating Unified Tensor Resharding in heterogeneous AI systems

**arXiv ID:** 2606.26633 | [PDF](https://arxiv.org/pdf/2606.26633v1)

**作者:** Sumit Kumar `[一作]` (IIIT-Delhi), Rinku Shah `[通讯]` (IIIT-Delhi)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向LLM训练的异构全栈仿真器，可精确预测混合GPU与网络拓扑下的训练时间。

**💡 创新点**

创新点包括基于LCM的多环通信与梯度重新分区、异构工作负载分区、可插拔NS‑3/htsim网络后端以及完整的管线并行与异构协同模型。

**🔧 技术方法**

使用离散事件仿真、Sweep‑Line DP分组、LCM多环构造、梯度分块重分配、Pipeline Barrier、NS‑3与htsim网络模拟等技术。

**📊 数据集**

使用Llama 2 7B/13B、GPT‑175B模型以及AICB工作负载基准进行评测。

**📈 对比分析**

与SimAI、HexiScale等同质仿真器比较，误差小于5%，在异构配置下误差降至2%，网络模拟速度比NS‑3提升47倍。

**⚠️ 局限性**

局限性在于仍以GPU为中心，未覆盖FPGA/ASIC等加速器，且需要人工提供详细模型/网络配置，难以实现大规模自动化搜索。

---

## 264. Latent Diffusion Posterior Sampling with Surrogate Likelihood Guidance for PDE Inverse Problems

**arXiv ID:** 2606.26592 | [PDF](https://arxiv.org/pdf/2606.26592v1)

**作者:** Yuanzhe Wang `[一作]` (University of Illinois Urbana-Champaign), Alexandre M. Tartakovsky `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种在潜在空间进行扩散后验采样（L-DPS）的方法，用于高维PDE逆问题的贝叶斯推断。

**💡 创新点**

创新点在于将VAE压缩、无条件潜在扩散模型与可微神经算子融合，利用潜在先验分数与逼近似似然的平滑指导，实现对未知参数场的高效、鲁棒采样；并提出噪声与观测密度自适应的引导权重。

**🔧 技术方法**

使用的技术包括变分自编码器（VAE）、无条件扩散概率模型（DDPM/DDIM）、神经算子（FNO、ViT、DeepONet）、扩散后验采样（DPS）与梯度归一化的自适应引导。

**📊 数据集**

数据集为合成的二维Darcy流问题：128×128网格上生成5000个对数高渗透率场（单尺度、高斯、双尺度高斯、二值分布），通过有限差分求解得到对应的压力场，用于训练VAE、扩散模型与神经算子。

**📈 对比分析**

与基准方法（全空间DPS+FNO、条件潜在扩散模型CLDM、逆FNO、KLE-MAP）以及不同引导策略进行比较。结果显示：在稀疏/噪声观测下，L-DPS+FNO在重建误差上优于或与基准相当，同时每样本推断时间约为1分钟，比全空间DPS快约5倍；对非高斯先验的表现优于KLE-MAP，且混合先验模型仍保持竞争力。

**⚠️ 局限性**

局限性包括：仅评估单样本后验采样，未完成完整的后验不确定性量化；依赖VAE表征质量与神经算子精度，且引导近似仍受限于插件似然和有限步逆扩散；适用于静态单一物理场的Darcy问题，扩展到时变多物理耦合和更大规模问题仍待研究。

---

## 265. Empirical Software Engineering TerraProbe: A Layered-Oracle Framework for Detecting Deceptive Fixes in LLM-Assisted Terraform

**arXiv ID:** 2606.26590 | [PDF](https://arxiv.org/pdf/2606.26590v1)

**作者:** Manar Alsaid `[一作]` (East Texas A&M University), Faris Abbas `[通讯]` (Texas Woman's University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了TerraProbe，一个基于 Terraform 工具链的五层 oracle 评估框架，用以检测 LLM 生成的 IaC 安全修复中的欺骗性修复；

**💡 创新点**

提出分层 oracle 评估、跨模型对比以及四维欺骗性修复分类法，首次系统揭示 LLM 维修中 oracle 通过但安全意图未满足的失效模式；

**🔧 技术方法**

结合 Checkov 静态扫描、terraform validate/plan/show、统计检验（卡方、Fisher、Cohen's h）以及人工判定，形成完整的评估流水线；

**📊 数据集**

使用 TerraDS 公开模块（68 个真实项目）与 28 个注入缺陷的控制模块，共 96 条修复案例；

**📈 对比分析**

结果显示：目标检查消除率 83%，全扫描洁净率仅 10%，Terraform 计划通过率约 40%，计划对比可达 38%，在 TerraDS 修复中欺骗性修复率高达 71%（各模型差异不显著），并通过显著性检验验证差异；

**⚠️ 局限性**

局限包括仅评估 AWS Terraform、首轮无迭代提示、单一静态扫描器（Checkov）、样本量有限及对其他 IaC 平台的泛化受限。

---

## 266. Inference-Time Robot Behavior Steering through Physically-Aware Reconfiguration of Task-Structure

**arXiv ID:** 2606.26588 | [PDF](https://arxiv.org/pdf/2606.26588v1)

**作者:** Yiyuan Pan `[一作]` (Carnegie Mellon University), Changliu Liu `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种零拷贝行为调节框架（基于ENAP），能够在推理时根据用户偏好（语言描述）重新配置已训练的机器人策略而无需再训练；

**💡 创新点**

创新点在于：①将偏好转换为LTL_f DFA，并通过同步积与高层状态机（PMM）结合，实现对任务结构的可解释性重构；②在低层控制器冻结的前提下，通过轨迹重放重计算动作先验，实现物理可执行的偏好满足；③融合了自监督的高层符号抽象与低层连续控制的双重稀疏表示，兼顾数据效率与可解释性；

**🔧 技术方法**

技术方法包括：Emergent Neural Automaton Policy (ENAP) 的概率Mealy机结构、线性时序逻辑 (LTL_f) 与确定性有限自动机 (DFA) 的转换、同步积构造、轨迹重放动作先验重计算、注意力+Clamp 低层控制器、VLM用于生成LTL与细化簇；

**📊 数据集**

数据集涵盖：ManiSkill（复杂操控）、Calvin benchmark（长周期TAMP）、以及真实世界抓取/插装/移动任务，所有实验均使用相同的演示数据进行训练；

**📈 对比分析**

与Transformer、GMM、Diffusion Policy、VLA（OpenVLA、π_0.5）等基线对比，实验表明该方法在任务成功率（SR）和偏好满足率（SRP）上均优于基线，提升幅度可达25%，并在长周期任务与真实世界场景中保持高成功率；

**⚠️ 局限性**

局限性包括：①依赖ENAP的结构，扩展到其他策略类仍需研究；②偏好满足依赖VLM生成的LTL表达式，若表达不充分会影响效果；③适用范围受限于符号抽象的表达能力，难以处理更丰富的OOD偏好与新技能生成。

---

## 267. A Multi-Level Validation and Traceability Framework for AI-Generated Telescope Scheduling Decisions

**arXiv ID:** 2606.26585 | [PDF](https://arxiv.org/pdf/2606.26585v1)

**作者:** Hengchu Xiao `[一作]` (Yunnan Observatories, Chinese Academy of Sciences), Chuanjun Wang `[通讯]` (Yunnan Observatories, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一个多层次验证与可追踪推理框架，用于提升大型语言模型在天文观测调度中的可靠性。

**💡 创新点**

创新点包括引入原子推理单元（ARU）与有向无环图（DAG）结构，实现对AI调度决策的细粒度一致性校验与可追踪性，并提供反馈循环以自动纠错。

**🔧 技术方法**

使用了LLM（Qwen3、Phi‑4）生成调度方案，配合结构化输入输出、ARU/DAG、基于数据库的多级约束校验，以及传统的强化学习、整数规划等调度方法做对比。

**📊 数据集**

基于ZTF真实瞬态警报数据和多台异构望远镜的状态与观测条件，构成了6类场景的仿真与真实数据实验。

**📈 对比分析**

通过与贪心、截止规则、ILP等基线进行对比，实验显示单纯LLM调度的外部可执行率(EER)≈0.69、参考策略可接受率(RPA)≈0.53；加入验证后EER=1、RPA≈0.94；进一步加入反馈与ARU层提升RPA≈0.98、RRS≈0.983，执行时间略高。

**⚠️ 局限性**

局限性包括目前仅在动态ToO场景验证，未覆盖离线夜间调度；LLM仍易出现幻觉；验证层增加计算开销；框架需在长期运行与更强模型上进一步评估。

---

## 268. SKILL-DISCO: Distilling and Compiling Agent Traces into Reusable Procedural Skills

**arXiv ID:** 2606.26669 | [PDF](https://arxiv.org/pdf/2606.26669v1)

**作者:** Zhongxin Guo `[一作]` (Microsoft Research), Yongqiang Xiong `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

基于成功轨迹提取、编译可执行的可验证技能，形成可跨模型、可重用的程序化技能库

**💡 创新点**

①在 FSM 定义场景下将共享执行结构建模为参数化有限状态机（PFSM）子图；②通过归一化、子目标操作抽取与聚类实现 PFSM 子图近似；③将抽象子图编译为可调用、可验证的可执行代码，提升可靠性和迁移性

**🔧 技术方法**

利用 LLM（如 GPT‑4o）完成轨迹正则化、代码合成与验证；参数化 FSM 与子图匹配、聚类；自动化编译与验证循环；实验中结合强化学习、LLM 交互模型

**📊 数据集**

ALFWorld（文本化家庭任务）与 WebArena（网站导航任务）的成功轨迹集

**📈 对比分析**

与 ReAct、CodeAct 及离线工作流/技能诱导方法（AWMoffline、ASIoffline）对比；在 ALFWorld 上成功率提升至 99.3%（+3.1%）且平均回合数显著下降；在 WebArena 上成功率提升至 29.1%（+21.6%），回合数亦大幅减少；跨模型迁移表现优异，低容量模型提升显著

**⚠️ 局限性**

仅适用于具有可重复执行结构的程序化任务；依赖成功轨迹与强大 LLM 的代码生成能力；在成功率低或纯 NLP 任务中效果有限

---

## 269. PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs

**arXiv ID:** 2606.26666 | [PDF](https://arxiv.org/pdf/2606.26666v1)

**作者:** Muhammad Ahmed `[一作]` `[通讯]`, Muhammad Ahmed

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究一种基于原生页表的分块解码引擎PersistentKV，结合自适应页面感知调度提升LLM推理服务吞吐量。

**💡 创新点**

创新点在于：①针对分组查询注意力（GQA）实现KV页表直接解码，充分利用KV页重用；②引入按行局部长度分割与紧凑工作队列，显著减少解码步骤中的核启动和填充开销；③构建可校准的自适应调度策略，在不同活跃度与上下文长度场景下动态选择FlashInfer、PersistentKV长度桶或工作队列模式。

**🔧 技术方法**

技术包括：原生页表KV缓存、分组查询注意力（GQA）、按行局部长度分割与在线softmax合并、紧凑工作队列（仅对非空分割发起核）、CUDA图回放与原子合并实验、基于FP16的精度容忍度校验、CUDA事件与同步墙时计时。

**📊 数据集**

使用合成请求轨迹，包含bucketed、bimodal、uniform和Zipf等长度分布，生成物理页表与块表映射；并在NVIDIA RTX 3060上进行实验，未使用真实生产数据。

**📈 对比分析**

与FlashInfer等现有原生页表解码基线比较，采用相同的KV页表、GQA形状与数值容差。结果显示：在B1长上下文场景下，PersistentKV桶式分割比FlashInfer提升1.399×吞吐；在B8长上下文的bimodal、uniform、Zipf场景下，工作队列模式比FlashInfer提升1.063–1.265×同步墙时吞吐。

**⚠️ 局限性**

局限性包括：仅在单一RTX 3060上评估；使用合成轨迹无法覆盖真实生产负载；自适应阈值手工校准，缺乏在线学习；GQA重用仅在G=4实验；未评估在更高SM/带宽GPU、Hopper或更大模型上的表现；未集成到完整LLM服务栈。

---

## 270. TGHE: Template-based Graph Homomorphic Encryption for Privacy-Preserving GNN Inference in Edge-Cloud Systems

**arXiv ID:** 2606.26664 | [PDF](https://arxiv.org/pdf/2606.26664v1)

**作者:** Ngoc Bao Anh Le `[一作]` (University of Wollongong), Jun Shen `[通讯]` (University of Wollongong)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出TGHE框架，实现边缘-云环境下的大规模加密图神经网络推理

**💡 创新点**

创新在于发现交易图局部计算树呈现模板现象，采用ego‑centric设计并通过模板统一打包实现SIMD并行；提出近似模板拟合和拓扑折叠两种长尾优化器

**🔧 技术方法**

使用CKKS同态加密、模板化加密打包、加密多项式激活、聚合预处理与SIMD矩阵向量乘

**📊 数据集**

在真实金融交易图DGraphFin（370万节点、430万边）以及四个金融数据集上验证模板现象

**📈 对比分析**

与先前HE‑GNN（CryptoGCN、LinGCN、Penguin、FicGCN）相比，TGHE-Base实现约13.5×加速，TGHE‑Collapse实现66.9×加速；在加密推理下AUC仅下降不到0.2%

**⚠️ 局限性**

局限包括：仍需泄露一定结构信息（模板签名、边/时间元数据），仅适用于线性聚合或可近似聚合的GNN架构，超大规模的长尾查询仍可能产生较大开销

---

## 271. Learning Anonymous Pricing for Online Resource Allocation

**arXiv ID:** 2606.26651 | [PDF](https://arxiv.org/pdf/2606.26651v1)

**作者:** Yifeng Teng `[一作]` (Google), Yifan Wang `[通讯]` (Georgia Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在线资源分配问题，卖家依次接收来自n个异质代理的m种资源的独立请求，目标是最大化社会福利。

**💡 创新点**

提出了一种匿名定价算法，解决了动态定价算法的公平性问题，并且不需要提前知道代理的到达顺序。

**🔧 技术方法**

使用了样本和查询的学习算法，特别是通过样本学习经典的双重定价算法和近似最优的匿名定价算法。

**📊 数据集**

使用了来自n个代理的请求分布的样本和查询数据集。

**📈 对比分析**

与现有的动态定价算法相比，提出的匿名定价机制在性能上能够达到(1-ε)的近似最优社会福利，并且不依赖于代理的到达顺序。

**⚠️ 局限性**

算法的局限性在于需要对代理的价值分布有一定的了解，并且在某些情况下可能无法保证完全的匿名性。

---

## 272. CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs

**arXiv ID:** 2606.26650 | [PDF](https://arxiv.org/pdf/2606.26650v1)

**作者:** Shigeng Wang `[一作]` (Intel Labs China), Anbang Yao `[通讯]` (Intel Labs China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种后训练三值量化方法CAT-Q，用于对大语言模型的权重进行低成本、高精度的三值化压缩与加速。

**💡 创新点**

创新点在于引入可学习调制（LM）与柔性三值化（ST）两大模块，并将其嵌入滑动层输出重构管线，实现在无额外训练标注的条件下实现分布对齐与稳定收敛。

**🔧 技术方法**

采用可学习权重分布变换、平滑过渡函数实现软硬三值化、滑动窗口联合重构以及后训练量化技术，兼顾硬件友好与推理效率。

**📊 数据集**

使用512条C4数据作为校准样本，评估基准覆盖PIQA、ARC-e、ARC-c、HellaSwag、Winogrande等五大通用常识推理数据集。

**📈 对比分析**

与基于QAT的三值化方法（BitNet、TriLM、Tequila）以及低位PTQ方法（GPTQ、AWQ、OmniQuant等）对比，CAT-Q在512样本下仅需100M训练token即可匹敌100B token的QAT，且在1.7B–235B规模模型上能在8–60小时内完成量化，平均精度下降低于1%至3%。

**⚠️ 局限性**

局限性包括对极大规模模型仍需数小时GPU加速，且主要针对权重量化，激活仍使用8-bit；在极小样本或极小模型上可能精度不足，且对特殊模型结构的适用性尚未全面验证。

---

## 273. Invisible Impact of Empathy on Behavioral Change: Isolating the Effect of Empathy in Long-term Physical Activity Coaching Chatbot Interactions

**arXiv ID:** 2606.26641 | [PDF](https://arxiv.org/pdf/2606.26641v1)

**作者:** Li Siyan `[一作]` (Columbia University), Zhou Yu `[通讯]` (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在长期身体活动（PA）聊天机器人交互中，单独隔离同理心对用户行为改变（步数、意向跟随、自我效能、信任）和机器人感知的影响。通过构建三种版本（非同理心、标准同理心、临床同理心）并在13名受试者上进行为期六周的跨周期实验，收集步数、问卷及对话记录。

**💡 创新点**

创新点在于：① 将同理心作为单一实验变量进行严格对照，首次系统评估其在长期PA聊天机器人中的作用；② 引入临床同理心框架（基于健康护理专业人士对患者信息的回应模式），与通用同理心模型对比；③ 采用LLM（GPT‑4o）与Gemini‑2.5‑Pro 进行对话同理心量化（EPITOME 维度），验证同理心与动机指标的相关性；④ 提出四条设计原则并在实验中验证其有效性。

**🔧 技术方法**

技术方法包括：GPT‑4o 大语言模型（核心聊天生成）+Twilio WhatsApp 接口；长短期记忆模块（句子变换器 + FAISS 索引）；同理心机会分类器（GPT‑4o‑mini）+策略采样器+指令生成器；压力缓解微干预（多臂赌博机 + 检索增强生成）；线性混合效应模型（LMM）用于统计分析；Gemini‑2.5‑Pro 用于对话同理心评分。

**📊 数据集**

使用的数据集：实验中收集的 239 条对话（76 非同理心、80 标准同理心、81 临床同理心）；受试者自报步数、意向、效能、效用等问卷；EPITOME 框架用于对话同理心标注；未使用公开 PA 训练数据，而是利用从临床健康干预中抽取的同理心机会和回应策略映射构建同理心模块。

**📈 对比分析**

比较方法：在同一组受试者内部交替体验三种机器人，采用部分平衡顺序；通过即时检查表和每周反思表收集定量数据；使用 LMM 检验同理心、时间、周末等因素对步数、意向、自我效能、效用的影响。结果显示：① 虽然非同理心版本被认为更有趣、实用，但临床同理心版本在意向跟随与自我效能随时间显著提升（p<0.05）；② 总步数方面，标准同理心版本略高，但差异不显著；③ 同理心的降低 attrition（仅 17%）相对其他版本；整体统计显著性受样本量限制，但趋势与理论一致。

**⚠️ 局限性**

局限性：① 样本量小（N=13）且 attrition 高，导致统计功效不足；② 随机生成的 LLM 输出难以完全控制同理心表现，存在“偶然”变异；③ 交互多为重复、短时，缺乏深度对话；④ 仅基于自报步数和问卷，未直接读取设备步数；⑤ 受试者主要为女性、年轻人，缺乏普遍性；⑥ 机器人缺乏完整情境感知（如周末/工作日、实时步数），可能影响建议适宜性。

---

## 274. From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning

**arXiv ID:** 2606.26629 | [PDF](https://arxiv.org/pdf/2606.26629v1)

**作者:** Evan Ning `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过在稀疏自编码器（SAE）的特征空间中正则化激活，提出了一种针对大语言模型的连续学习方法。

**💡 创新点**

采用预训练的 SAEs 作为单义特征字典，结合受约束优化导出平方铰链损失，突破了权重空间正则化在 polysemantic 模型中的局限。

**🔧 技术方法**

使用预训练 SAE、LoRA 适配器以及基于平方铰链的稳定-可塑性约束损失，并与 EWC/SI/MAS 等对比。

**📊 数据集**

实验基于 TRACE-5000 交叉域连续学习数据集和 MedCL 医学内域连续学习数据集。

**📈 对比分析**

与权重正则化、梯度投影、架构隔离和重放基线对比，SAE 正则在 TRACE 上实现 OP 0.545、在 MedCL 上实现 OP 0.510，均优于大多数无架构方法，且每任务仅需 412 KB 的特征掩码存储。

**⚠️ 局限性**

局限包括 SAE 覆盖率有限、与重放互补性不足、输出格式干扰不被直接解决，以及对不同模型（如 Mistral‑7B、Gemma‑2 9B 基础版）的泛化仍待验证。

---

## 275. Agents That Know Too Much: A Data-Centric Survey of Privacy in LLM Agents

**arXiv ID:** 2606.26627 | [PDF](https://arxiv.org/pdf/2606.26627v1)

**作者:** Nada Lahjouji `[一作]` (University of California, Irvine), Ashwin Gerard Colaco `[通讯]` (University of California, Irvine)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型代理（LLM Agent）在处理敏感数据时的隐私风险进行系统梳理与评估，提出面向数据表面（data surfaces）的隐私风险与治理机制分类，构建风险-向量-控制三张映射表，并回顾现有评测基准与应用场景，指出现有研究的缺口与开放问题。

**💡 创新点**

①以数据表面为核心的“数据中心”视角，对代理隐私问题进行统一分类；②构建风险（泄漏结果）、攻击向量与治理机制的跨表格关联，形成可复用的参照框架；③首次系统性评估跨表面、跨步骤的隐私泄漏与评测基准缺失，提出端到端基准的研究需求。

**🔧 技术方法**

对现有文献进行分类汇总与梳理，主要涉及自然语言查询、检索增强生成、工具调用、内存管理、跨代理通信等技术；在此基础上使用信息流控制、访问控制、运行时治理、差分隐私、加密检索、上下文完整性等已提出的治理机制进行理论对照与评估。

**📊 数据集**

本工作为综述性论文，无直接使用实验数据集；所提及的基准与案例均来自公开的研究工作（如 SecureSQL、AgentLeak、MAGPIE 等）。

**📈 对比分析**

通过表格对比，作者对比了各基准在覆盖数据表面、是否为多步执行、是否基于显式隐私策略等维度的能力；结果显示目前大多数基准仅覆盖单一表面且缺乏端到端、多表面、显式策略评估，性能（隐私泄漏率）未能在统一标准下统一衡量。

**⚠️ 局限性**

①缺乏端到端跨表面、跨步骤、显式隐私策略的评测基准；②现有治理机制多为单步或单表面，缺乏全流程的合成与一致性保证；③对持久内存与跨会话泄漏的删除/撤销机制缺失；④缺少对上下文完整性与用户意图一致性的自动化评估；⑤无法提供完整的合规证明与审计证据。

---

## 276. Sketched Linear Contrastive Learning: Approximation, Optimization, and Statistical Scaling

**arXiv ID:** 2606.26617 | [PDF](https://arxiv.org/pdf/2606.26617v1)

**作者:** Ziyan Chen `[一作]` (University of Sydney), Ding-Xuan Zhou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

在简化的高斯对偶视图模型下，研究了对比学习的缩放律；通过对二次高斯负对比代理函数的全批梯度下降，推导了期望风险的近似误差、梯度下降偏差、方差及交叉项的分解与缩放表达；

**💡 创新点**

首次将缩放律框架推广到对比学习，并引入“乘积有效维度”这一新概念，揭示对比学习的优化和方差受两视图谱方向交互影响；

**🔧 技术方法**

使用高斯采样的线性投影（sketching）、对角化的幂律谱假设、对比源条件、全批梯度下降与精确风险分解；

**📊 数据集**

实验仅使用合成的“配对高斯潜变量”数据，验证理论预测的近似误差、偏差与方差随模型尺寸、样本量和迭代次数的变化；

**📈 对比分析**

与理论预期进行对比，实验结果与理论的幂律指数吻合，误差曲线与预测一致，证明理论的有效性；

**⚠️ 局限性**

限制包括：仅考虑同维度线性视图，未使用真实负样本或Mini‑Batch SGD，假设高斯数据与严格的协方差集中条件，未覆盖实际深度模型与异构模态。

---

## 277. TaskTok: Delving into Task Tokens for Task-driven Image Restoration

**arXiv ID:** 2606.26615 | [PDF](https://arxiv.org/pdf/2606.26615v1)

**作者:** Hongjae Lee `[一作]` (Korea University), Seung-Won Jung `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 TaskTok，一种任务驱动图像恢复框架，通过在一维 Token 空间中仅更新任务相关 Token 来提升高阶视觉任务性能。

**💡 创新点**

创新点在于引入可学习的 Token Switch 与轻量级 Token Refiner，实现对任务相关 Token 的选择性恢复，并利用 1D Token 的索引专一性来定位重要视觉特征。

**🔧 技术方法**

核心技术包括 TiTok 一维 Tokenizer、SwinIR 预恢复模块、Transformer 轻量级 Refiner、任务特定的 Token Switch 以及 token‑level 监督与掩码正则化。

**📊 数据集**

实验使用 ImageNet（分类）、PASCAL VOC 2012（分割与检测）以及 CUB‑200、Oxford‑IIIT Pet 等未见数据集进行跨数据集与跨网络的验证。

**📈 对比分析**

与 SwinIR、SR4IR、EDTR 等现有方法相比，TaskTok 在 Mix‑B 退化下分类 Top‑1 由 46.8% 提升至 55.3%（+8.5%），分割 mIoU 提升至 61.9%（+2.4%），检测 mAP 提升至 25.9%（+0.4%），且只恢复 12/34 个 Token，显著提高吞吐率（8.3×）和能效。

**⚠️ 局限性**

局限性包括仅在 1D Token 结构上验证，需进一步评估在更大规模模型或多模态任务中的适用性；Token Switch 的阈值与初始化对性能影响仍需更系统化研究。

---

## 278. HiLSVA: Design and Evaluation of a Human-in-the-Loop Agentic System for Scientific Visualization

**arXiv ID:** 2606.26614 | [PDF](https://arxiv.org/pdf/2606.26614v1)

**作者:** Kuangshi Ai `[一作]` (University of Notre Dame), Chaoli Wang `[通讯]` (University of Notre Dame)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出并实现了HiLSVA，一种人机混合主动的科学可视化智能体系统，支持从自然语言指令到完整可视化流程的协同执行。

**💡 创新点**

创新点在于将混合主动性、工作流可追溯性、沙盒安全执行和测试时学习（LTT）与LLM驱动的可视化代理无缝融合，强调人类在决策与监督中的核心作用。

**🔧 技术方法**

采用GPT‑5.2与Claude‑Sonnet‑4.6进行任务解析和代码生成，利用ParaView‑MCP实现对ParaView的直接API调用，结合Docker沙盒、知识库检索与自我反思的自改进代理。

**📊 数据集**

使用多种科学数据集进行案例研究，包括CT人足、飓风Isabel、龙卷风向量场、燃烧多变体以及半圆柱流场等，覆盖基本操作、工作流构建与科学分析三大类任务。

**📈 对比分析**

通过12名参与者的对比研究，三种自主度模式（全自动、半自动、混合主动）在执行时间与准确率（平均11.75/12）上进行比较，结果显示全自动最快但人类监督最少，混合主动在效率与可控性之间取得平衡。

**⚠️ 局限性**

主要限制包括LLM API调用的延迟、LTT仅基于检索而非模型权重更新、对领域专有知识的获取受限、用户样本规模较小以及缺乏长期、跨域验证。

---

## 279. EcoTable: Cost-effective Table Integration in Data Lakes for Natural Language Queries

**arXiv ID:** 2606.26613 | [PDF](https://arxiv.org/pdf/2606.26613v1)

**作者:** Yuhui Wang `[一作]` (Beijing Institute of Technology), Lei Cao `[通讯]` (University of Arizona)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EcoTable，一种基于自然语言查询的数据湖表集成框架，能自动识别、验证并转换所需表，直接生成支持 SQL 查询的联合表。

**💡 创新点**

创新点包括：①用图结构表示全局表关联，并通过 Steiner 树和 t‑spanner 子图精确寻找最优 join 路径；②采用轻量级深度模型先估计表关联概率，再用 LLM 仅在最有可能的边上做验证，显著降低 LLM 调用成本；③设计 ReAct 迭代式转换策略和图颜色并行执行，既保证转换准确性又提升效率。

**🔧 技术方法**

技术手段包括：预训练语言模型（RoBERTa、DeBERTa）、轻量级表关联模型（DeepJoin、OmniMatch）、Steiner 树与 t‑spanner 的图算法、LLM（如 GPT‑4o）进行语义链接、join 验证与代码生成、ReAct 思维循环、Vizing 边上色并行执行。

**📊 数据集**

实验使用四个真实工业数据湖基准（广告、用户、金融、工程）和一个规模更大、噪声更丰富的 NYC Open Data 数据湖，总计 5 个数据集，覆盖 1,214 张表和 800 条查询。

**📈 对比分析**

与 6+ 传统与 LLM 端到端方案比较，EcoTable 在 join 路径 F1 分数上提升 30%+，在 LLM 费用上降低 5 倍，查询成功率接近或超过基线，且在大规模数据湖下仍保持较低的成本与时延。

**⚠️ 局限性**

局限性包括：目前仅支持无环 join 图；对同一表对可能存在多种 join 条件的情况处理不足；对极大图仍需进一步剪枝或近似策略；整体性能仍受 LLM 计算成本和训练数据质量影响。

---

## 280. LogicIR: Logic Gate Networks for Image Restoration

**arXiv ID:** 2606.26609 | [PDF](https://arxiv.org/pdf/2606.26609v1)

**作者:** Hongjae Lee `[一作]` (Korea University), Seung-Won Jung `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 LogicIR，一种全逻辑门网络（LGN）用于图像恢复任务，采用 UNet 结构、可微分比特解码、索引置换等技术实现高效的图像去噪、去块和去雨。

**💡 创新点**

创新点包括：①首次将逻辑门网络应用于图像恢复；②使用 UNet 级联的卷积逻辑门构建层次特征；③引入可微分比特解码层将二进制特征映射为连续残差；④提出索引置换机制增强跨组信息交流；⑤采用多角度旋转集成与 MSB 监督提升恢复质量。

**🔧 技术方法**

技术主要包括逻辑门网络（NAND、XOR 等）、可微分逻辑门选择（softmax 软选）、UNet 编码器-解码器、像素解/重排（pixel unshuffle / shuffle）、比特解码与缩放、直通估计（STE）微调、旋转集成、PSNR/SSIM/LPIPS/NIQE/CLIP‑IQA 评估。

**📊 数据集**

使用的数据集有 BSD68、Set12、Urban100、LIVE1、Classic5、Test100（降雨）和 BSD 数据集的灰度图像，用于去噪、去块、去雨任务。

**📈 对比分析**

与全精度模型（DnCNN、SwinIR）、BNN（BBCU、ReActNet、Bi‑Real）、LUT 方法（HKLUT、TinyLUT、SR‑LUT）以及基线 StackedCLGN 进行对比。LogicIR‑S 在 169.3 G BOPs 的情况下达 27.71 dB PSNR，性能接近甚至超过 BBCU‑lite，但运算量下降 6.5×；LogicIR‑L 在 1078.6 G BOPs 亦优于 HKLUT；在去块和去雨任务中，同样保持较低 BOPs 与可比 PSNR、SSIM 等指标。综合来看，LogicIR 在保证极低算力的同时，恢复质量接近传统轻量化网络。

**⚠️ 局限性**

局限性：目前仅处理同分辨率的恢复任务，未设计高质量可扩展的逻辑门上采样机制，限制了其在超分辨率等需要显式上采样任务中的应用；逻辑门网络的训练依赖可微分松弛与 STE，仍存在性能与实际硬件实现之间的差距。

---

## 281. Bridging Handheld and Teleoperated Supervision for Contact-Rich Manipulation via State-Gated Experts

**arXiv ID:** 2606.26603 | [PDF](https://arxiv.org/pdf/2606.26603v1)

**作者:** Vidullan Surendran `[一作]`, David Watkins `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究结合手持数据与少量遥操作数据，通过混合扩散专家网络实现接触丰富任务的高效学习。

**💡 创新点**

提出按机器人状态切换观测动作与期望动作监督的混合扩散专家模型，解决动作有效性差异，并利用少量针对性遥操作示例填补接触关键阶段。

**🔧 技术方法**

采用Diffusion Policy基础架构、DINOv2视觉编码+Perceiver‑IO聚合、动作路由器（MLP）与k‑NN标签推断、混合专家（基线+支持）硬切换等技术。

**📊 数据集**

使用Dual‑Mode UMI采集的手持演示（观测动作）和少量针对性遥操作演示（期望动作），共3个接触任务（NIST滑轮、管道插入、电池插入）。

**📈 对比分析**

与仅手持、仅遥操作、直接混合等基线对比，在三任务上将成功率从手持基线44%提升至84%（手持+支持约76%），提升幅度超过36.7%，显著优于手持基线。

**⚠️ 局限性**

难以自动确定支持阶段；硬切换导致边界不连续；在长时间序列多接触区域时可扩展性差；依赖实验室手持数据，对外部环境泛化不足。

---

## 282. SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference

**arXiv ID:** 2606.26587 | [PDF](https://arxiv.org/pdf/2606.26587v1)

**作者:** Haoqian Meng `[一作]` (Tianjin University), Peng Zhang `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 SharQ，一种无训练、在线稀疏-稠密拆分的 FP4 量化推理方法，结合 N:M 半结构稀疏与低位浮点。

**💡 创新点**

创新点在于把稀疏掩码生成与量化分离，定义稠密残差相对于已量化稀疏骨干，并利用共享 FP4 权重实现两路 FP4 计算；同时在硬件层面融合准备核、共享权重视图和残差累加。

**🔧 技术方法**

使用了 FP4 块量化、N:M 稀疏、在线掩码生成、量化后残差定义、共享权重视图、融合核、稠密-稀疏两路 FP4 GEMM 与残差累加等技术。

**📊 数据集**

评估数据集包括 Llama-3.1-8B、Qwen2.5-7B、Qwen3-30B-A3B、Qwen3-VL-8B 和 Wan2.2-T2V-A14B，涉及语言模型的零样本/少样本推理、WikiText2 perplexity、MMLU、Vision‑Language 基准以及视频生成推理时延。

**📈 对比分析**

与 NVFP4、FP16、FP8 基线对比，SharQ 在语言模型上平均恢复 43–63% 的 FP4 对 FP16 精度缺口，推理时延比 FP16 低 2.2–2.4×，吞吐量比 FP8 高 1.2–1.4×；在多模态和视频生成任务中亦提升数个百分点。

**⚠️ 局限性**

局限性在于依赖硬件支持的块量化 FP4 与 N:M 稀疏；对极大分辨率或注意力计算占比高的任务提升有限；整体实现仍需额外两次 GEMM，可能在内存带宽受限场景受限；缺乏训练或自适应调优的支持。

---

## 283. Hardware Design for Table Tennis Robot Capable of Beating Professional Players

**arXiv ID:** 2606.26643 | [PDF](https://arxiv.org/pdf/2606.26643v1)

**作者:** Nobuhiko Mukai `[一作]` (Sony AI), Peter Dürr `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计、制造并验证了一款具备击败职业乒乓球选手能力的8自由度机器人，并制定了符合ITTF规则的硬件规格。

**💡 创新点**

首次系统性给出乒乓球机器人竞赛硬件规范，并通过拓扑优化、逆动力学扭矩模型与延迟补偿的低阶动力学模型，配合强化学习控制，实现高速度、低延迟和高轨迹精度的击球。

**🔧 技术方法**

使用了拓扑优化与增材制造、逆动力学扭矩优化、电机/齿轮选型、时延补偿的过程模型、强化学习控制、EtherCAT实时系统以及多摄像头感知。

**📊 数据集**

采集了职业选手挥拍轨迹与比赛数据（含9名职业选手的对战记录）作为运动轨迹与RL训练的数据集。

**📈 对比分析**

通过与9名职业选手对战（胜8名，含米乌·希拉诺）以及实验室测试，机器人实现0.8 s循环、22 m/s球速、轨迹误差≤12.8 cm、扭矩裕度>50%，满足或超过设计目标，性能优异。

**⚠️ 局限性**

仍存在关节J3阻尼不足导致控制困难、机械共振与高转速下的延迟限制，以及对顶级选手适应性不足等限制，硬件性能与观赏性之间的平衡仍需进一步优化。

---

## 284. Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention

**arXiv ID:** 2606.26663 | [PDF](https://arxiv.org/pdf/2606.26663v1)

**作者:** Siyu Wu `[一作]` (Ant Group), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种触觉感知的世界动作模型（WAM），能够同时预测未来的视觉潜在状态、触觉接触状态和动作，并通过触觉非对称注意机制实现触觉信息的选择性利用。

**💡 创新点**

创新点在于：① 引入 VideoClean 掩码阻止视频查询直接访问触觉键/值，避免触觉污染；② 通过触觉状态与触觉变化代理构造触觉感知偏置，只有在触觉变化预测时才增强动作对触觉锚点的注意，从而实现触觉非对称注意。

**🔧 技术方法**

技术手段：基于扩散式去噪反向生成的 WAM 框架，使用视觉、触觉、动作和上下文 Token 的联合序列；利用 Attention 掩码和偏置实现 VideoClean 与触觉感知偏置；使用预测的触觉变化代理 g(δ) 生成的标量偏置；对比实验、Ablation 以及视觉重建评估。

**📊 数据集**

数据集：ManiFeel 仿真基准（450 次试验，9 类任务）以及匹配的真实机器人实验数据（bolt‑nut 组装、bulb 插入、gear 插入、peg 插入、power 插入）。

**📈 对比分析**

对比方法：与 RGB‑only DreamZero WAM 基线和 π_0.5 行为策略基线进行对比。仿真结果中整体成功率从 5.8% 提升至 44.7%（提升 38.9 个百分点），真实机器人实验中成功率为 51%，相较 RGB‑only 提升 33%。Ablation 实验表明 VideoClean 与触觉感知偏置对性能提升贡献最大。

**⚠️ 局限性**

局限性：仍无法完全解决视觉搜索、长时序恢复、peg reorientation 等任务；触觉信息主要提升依赖局部接触变化的操作，对全局视觉驱动的高层决策帮助有限。

---

## 285. Disco-LoRA: Disentangled Composition of Content, Style, and Motion for Multi-concept Video Customization

**arXiv ID:** 2606.26668 | [PDF](https://arxiv.org/pdf/2606.26668v1)

**作者:** Xuancheng Xu `[一作]` (Nanjing University Of Posts And Telecommunications), Bing-Kun Bao `[通讯]` (Hefei University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可实现多概念视频定制的框架——Disco-LoRA，能够独立控制内容、风格（材料与艺术）和运动（物体与相机）并自由组合。

**💡 创新点**

核心创新包括：①将多概念定制拆解为内容-风格与内容-运动两任务；②引入迭代双LoRA去耦分离机制，配合互补提示和时间感知遮罩；③基于Z‑score的统计正则化，使LoRA权重在层级趋势保持一致的同时，统一幅度，减少概念互相干扰。

**🔧 技术方法**

使用LoRA低秩适配、DiT（WAN2.1）文本-视频扩散模型、DDIM采样、判别式与对比式风格度量、CLIP、CSD等多种评估工具。

**📊 数据集**

构建了包含20个内容对象、32种风格（22种艺术、10种材料）、20种运动（10物体、10相机）的自定义基准，生成并评估800个视频。

**📈 对比分析**

与DreamBooth、MotionDirector、UnzipLoRA+FlexiAct等SOTA方法在文本一致性、风格相似度、运动保真度和视觉质量等九项指标上比较，Disco-LoRA在大多数指标上取得最高分，用户评测亦显示其优于对手。

**⚠️ 局限性**

局限性：当前仅支持三种概念的组合，对更多概念的扩展尚未验证；在极端运动或极具挑战的风格转换时仍可能出现细节失真；模型对不同概念权重平衡仍需手工调节。

---

## 286. LAMP: Lane-Aligned Motion Primitives for Feasible Trajectory Prediction

**arXiv ID:** 2606.26661 | [PDF](https://arxiv.org/pdf/2606.26661v1)

**作者:** Sangjin Han `[一作]` (Seoul National University), H. Jin Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 LAMP 的运动预测框架，利用可学习的运动原语和基于车道拓扑的可行性感知意图选择，生成多模态但符合车道拓扑约束的轨迹预测。

**💡 创新点**

创新点包括：1）通过 VQ‑VAE（NSVQ）学习离散化的运动原语，捕捉轨迹的时空形状；2）设计可行性感知意图选择器，利用车道拓扑先验过滤不可行轨迹；3）在 Transformer 解码器前将选取的意图嵌入，保持多模态多样性与拓扑一致性。

**🔧 技术方法**

主要技术：Vector‑Quantized Variational AutoEncoder（NSVQ），Transformer 编码‑解码架构，场景上下文编码（MTR），交叉注意力，基于车道拓扑的可行性评分与 KL 监督，LoRA（实验性地图适配）等。

**📊 数据集**

在 Argoverse 2 运动预测数据集上进行实验。

**📈 对比分析**

与 MTR、Wayformer、Forecast‑MAE、EMP、Autobot 等强基线进行对比，位移误差（b‑minADE, b‑minFDE）保持与基线相当，同时在可行性指标（DAC, FR）和多样性指标（APD, FPD, DwF）上显著提升，表明预测集在保持多样性的同时更符合车道约束。

**⚠️ 局限性**

局限性：1）LoRA 方式会导致多样性下降，需进一步平衡可行性与多样性；2）意图选择数量对多样性有显著影响，过少会限制解码能力；3）当前评估仅基于离线数据，缺乏与下游规划器闭环验证的结果。

---

## 287. HiPR: Hierarchical Progressive Rendering for Immediate Feedback

**arXiv ID:** 2606.26612 | [PDF](https://arxiv.org/pdf/2606.26612v1)

**作者:** Rafael Padilla `[一作]` (University of Utah), Cem Yuksel `[通讯]` (University of Utah)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种层次化渐进渲染（HiPR）算法，利用光路依赖层级和感知优先级对像素块进行动态调度，以实现场景编辑后的即时视觉反馈。

**💡 创新点**

创新点在于将像素更新视为调度问题，通过深度测试与光路权重构建光传输层级，优先渲染对视图影响最大的区域，同时保证最终结果不偏离无偏路径追踪。

**🔧 技术方法**

采用基于瓦片的 G‑buffer 可见性穿透、光线追踪与路径追踪相结合的渲染管线；利用 Vulkan 的 inline ray tracing 与 BVH 结构，结合 OpenPBR BSDF 和自定义感知权重函数，实现 GPU 高效实现。

**📊 数据集**

使用多种交互式设计与虚拟制作场景作为测试数据集，包括镜面、玻璃、光照变化与破坏事件等典型场景；在 SIGGRAPH 2026 实验中展示效果。

**📈 对比分析**

通过与自适应采样和时空重用方法比较，HiPR 在场景修改后显著降低了响应延迟，视觉误差在几帧内即可消除，最终帧级收敛保持无偏，表现出更高的实时性和相对更低的总样本量。

**⚠️ 局限性**

局限性包括：需要手动设定感知权重和深度容差；在相机大幅平移/旋转时仍需重新投影整个帧；未实现基于改动幅度的 delta‑aware 权重；对 GPU inline ray tracing 的依赖限制了跨平台兼容性。

---

## 288. Fast Estimation for Forest Matrix of Signed Graphs

**arXiv ID:** 2606.26608 | [PDF](https://arxiv.org/pdf/2606.26608v1)

**作者:** Haoxin Sun `[一作]` (Fudan University), Zhongzhi Zhang `[通讯]` (Fudan University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套针对有符号图森林矩阵的快速估计方法，包括签名森林矩阵定理、基于正环消随机游走的广义支配收敛森林生成算法 GSCF、两种采样估计对角线的算法 FMDE 与 FMDE+，以及用于计算 Signed Friedkin‑Johnsen 模型表达意见的 FJOE。

**💡 创新点**

创新点在于：①推导了适用于有符号图的森林矩阵定理，填补了传统无符号图定理在符号图中失效的空白；②设计了改进的正环消随机游走方法，能够高效生成满足负环约束的广义支配收敛森林；③提出了 FMDE+，通过引入额外信息降低估计方差，理论与实验上显著优于 FMDE；④在百万级乃至千万级节点的大规模网络上实现了线性时间的近似估计。

**🔧 技术方法**

技术手段主要包括：循环消随机游走、概率采样与方差分析、森林矩阵定理推导、矩阵逆与快速求解的比较、Friedkin‑Johnsen 模型的表达意见估计、以及对正负边权重的解析处理。

**📊 数据集**

实验使用了来自 KONECT 与 SNAP 的公开网络数据集，如 Adolescent、Bitcoinotc、Gnutella08、Wikielec、Wikipedia、SlashdotZoo、Epinions、WikiL、Youtube、Dblp、Livejournal、FullUSA 等，包含原始符号图及将无符号图随机赋负边得到的符号图（加星标）。

**📈 对比分析**

与直接求逆 Exact 的方法比较，FMDE 与 FMDE+ 的运行时间在 10‑几百 倍之间大幅提升，误差均低于 1%；FMDE+ 的平均相对误差约为 FMDE 的十分之一；在表达意见方面，FJOE 在 ϵ = 0.1 时，绝对误差小于 0.02，并在 2.3×10⁷ 节点的 FullUSA 图上仅耗时不到 3×10⁻⁴ 秒；整体性能表现突出，具备可扩展性。

**⚠️ 局限性**

限制：①估计误差随采样数量递增，α/β 比值在不平衡图中可能较大导致样本量显著增加；②当前仅对对角线元素和表达意见给出高效近似，非对角线元素仍需改进；③算法实现为单线程，未充分挖掘并行化潜力；④对极端负边比例高的图仍存在计算负担，实际应用需结合负边分布进行调优。

---

## 289. DiCoBench: Benchmarking Multi-Image Fine-Grained Perception via Differential and Commonality Visual Cues

**arXiv ID:** 2606.26602 | [PDF](https://arxiv.org/pdf/2606.26602v1)

**作者:** Geng Li `[一作]` (Peking University), Yuxin Peng `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DiCoBench，一个多图高分辨率细粒度感知基准，评估模型在无文本提示下识别差异与共性视觉线索的能力。

**💡 创新点**

创新性地将细粒度感知拆分为差异与共性两条轨道，设计八类任务，采用多选题消除文本评测偏差，并聚焦微尺度视觉细节。

**🔧 技术方法**

使用先进视觉编码器、FLUX.2 Klein图像编辑、GPT-5.1生成指令、MMLM多模态模型、OpenAI文本嵌入等技术。

**📊 数据集**

数据来自V*高分辨率图像，结合自动化编辑与人工审核生成765个高分辨率样本，平均分辨率近2K。

**📈 对比分析**

在18款SOTA MLLM上评估，平均人类准确率98.3%，最佳模型Gemini‑3‑Pro仅58.1%，表现明显落后，尤其在推理子任务上低于20%。

**⚠️ 局限性**

局限在于仍缺乏足够的训练数据支持微尺度感知，模型在跨图微观差异与共性推理上误差率高，且对高分辨率信息处理效率低。

---

## 290. Structure Before Collapse: Transient semantic geometry in next-token prediction

**arXiv ID:** 2606.26749 | [PDF](https://arxiv.org/pdf/2606.26749v1)

**作者:** Yize Zhao `[一作]` (University of British Columbia), Christos Thrampoulidis `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在三种人工合成语言上训练 Transformer 解码器，研究了梯度下降如何在纯一热标签下先产生语义结构再收敛到神经崩塌的 ETF 结构。

**💡 创新点**

提出在一热标签下出现短暂语义几何的现象，并用球面连续 BoW 模型提供理论解释，弥合了神经崩塌理论与语言模型训练实际的差距。

**🔧 技术方法**

使用 Transformer 解码器、梯度下降、表示相似性分析（RSA）与 Gram 矩阵，并在理论层面引入球面 BoW 模型。

**📊 数据集**

自定义的三套合成语言：颜色‑形状交叉、Zipfian 线性语法以及带树结构的层级语法，全部为人工生成的上下文与标签。

**📈 对比分析**

通过计算模型表示的 Emp-RSM 与语义 RSM 及 ETF-RSM 的皮尔逊相关性，对比发现模型在训练早期与语义 RSM 相关性高，后期转为 ETF-RSM；大模型能更充分展现两者。

**⚠️ 局限性**

局限性在于仅使用合成数据、需要多轮训练、未覆盖真实语料中的 soft‑label 情况，且模型在新鲜数据和更复杂语法上的行为未知。

---

## 291. Do Image Editing Models Understand Lighting?

**arXiv ID:** 2606.26738 | [PDF](https://arxiv.org/pdf/2606.26738v1)

**作者:** Tim Küchler `[一作]` (Heidelberg University), Carsten Rother `[通讯]` (Heidelberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了3D‑anchored Light Probe（3DLP）基准和对应的1K HDR图像对数据集，用来真实量化图像编辑模型在灯光开启/关闭任务中的光照一致性。

**💡 创新点**

创新点在于：①收集了真实室内光照的高动态范围对照数据，②设计了两个对光照比值图像的物理一致性指标——标准化强度误差（SIE）和低频误差（LFE），允许模型自由调节灯光温度与亮度；③系统评估了多种商业与开源指令式图像编辑模型，并与VLM对比。

**🔧 技术方法**

技术上使用了HDR相机拍摄、光源探针定位、光照比值图像标准化与梯度分析，并通过Prompt搜索得到最优指令，最后使用SIE/LFE两项指标进行数值评估。

**📊 数据集**

使用的数据集为1K对HDR室内场景图像，其中每对包含灯光开启与关闭版本，并手工标注阴影、镜面、高光、金属等关键区域。

**📈 对比分析**

评估方法：对每个模型在前80%误差最低的图像上计算SIE与LFE，按平均排名排序。结果显示Nano Banana Pro位居榜首，Nano Banana 2次之，开源模型Qwen‑Image‑Edit排第三，其余模型表现明显逊色；VLM评估与物理指标不一致，难以替代精细光照测试。

**⚠️ 局限性**

局限性包括：仅覆盖室内场景，灯光功率不足以在强室外光照下测试；数据规模与多样性仍有限，难以涵盖所有复杂材质与高透明物体的光照交互；模型在高复杂度金属/透明场景中的误差仍较大。

---

## 292. Robust Onion: Peeling Open Vocab Object Detectors Under Noise

**arXiv ID:** 2606.26734 | [PDF](https://arxiv.org/pdf/2606.26734v1)

**作者:** Priyank Pathak `[一作]` (University of Central Florida), Yogesh S Rawat `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过合成噪声逐层剥离OV-OD模型，系统评估噪声对各组件的鲁棒性；

**💡 创新点**

发现视觉骨干决定鲁棒性、浅层最易受噪声影响，且语言提示对鲁棒性作用有限；

**🔧 技术方法**

采用合成噪声（湍流、运动模糊、像素化）与跨模态注意力融合的视觉‑语言检测框架；

**📊 数据集**

在COCO、LVIS、ODinW‑13、BDD‑100K等公开数据集上进行实验；

**📈 对比分析**

与现有OV‑OD对比，以相对鲁棒性衡量，结果表明骨干相似模型鲁棒性相近，且轻量化 NN & TK0 方法在不训练 96× 参数的情况下显著提升鲁棒性；

**⚠️ 局限性**

局限在于仅使用合成噪声，未能覆盖所有真实噪声分布，且未探索更深层次的跨模态自适应策略。

---

## 293. Modeling Adaptive Visual Search in Semantically Hierarchical Layouts

**arXiv ID:** 2606.26725 | [PDF](https://arxiv.org/pdf/2606.26725v1)

**作者:** Saku Sourulahti `[一作]` (University of Jyväskylä), Jussi P. P. Jokinen `[通讯]` (University of Jyväskylä)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并验证了一种基于计算合理性与层级视觉与语义结构的视觉搜索模型

**💡 创新点**

将层级视觉短时记忆与语义分类相结合，自动学习搜索策略

**🔧 技术方法**

使用EMMA眼动模型、PPO强化学习与计算合理性框架

**📊 数据集**

实验1使用60名Prolific参与者的在线视觉搜索数据；实验2使用先前菜单搜索研究的36名参与者数据

**📈 对比分析**

通过与真实眼动和搜索时长的R^2、RMSE、MAPE比较，模型在搜索时间、注视次数等指标上与人类高度吻合，R^2≈0.94‑0.99，RMSE≈0.05‑0.2s

**⚠️ 局限性**

未模拟视觉分组形成与语义距离计算，缺乏个体差异建模，对大规模布局时过估计搜索时间和注视数

---

## 294. Socratic agents for autonomous scientific discovery in high-dimensional physical systems

**arXiv ID:** 2606.26722 | [PDF](https://arxiv.org/pdf/2606.26722v1)

**作者:** Xianrui Zeng `[一作]` (University of Chinese Academy of Sciences), Yang Du `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了多智能体AI科学家框架AHOIS，利用苏格拉底式提问实现物理假设的自动构建、挑战和修正，并在多模光纤平台上实现了随机干涉编码、任务自适应稀疏扫描、失效模式诊断以及论文协议的自动重现。

**💡 创新点**

创新点在于将苏格拉底式中孕技术引入自主科学研究，形成一种可解释、可验证的认知循环，使AI系统能够在缺乏先验目标的情况下从实验数据中自我生成并修正物理模型，从而实现真正的知识自足科学发现。

**🔧 技术方法**

采用大语言模型驱动的多智能体架构（Boxue、Gewu、Mingde、Qiushi、Duzhi），配合光纤传输矩阵校准、数字微镜器件波前调制、光散射测量、神经分类器、U-Net/注意力重建网络以及苏格拉底式提问循环等技术。

**📊 数据集**

使用MNIST与Fashion‑MNIST数据集进行编码分类实验，另外采集了手写数字与服装目标的光散射图像，以及荧光细胞样本；同时重现了公开的多模光纤成像论文所用的实验数据。

**📈 对比分析**

与全像素扫描、固定子采样和全扫描进行对比，随机干涉编码的16×16测量实现了MNIST 76.97%与Fashion‑MNIST 83.17%的分类准确率；自适应稀疏扫描在保持接近完整识别的同时，以约48fps的视频帧率显著优于全扫描；在重建任务中，改进的残差注意力网络在SSIM/PSNR上优于基线U‑Net；苏格拉底式提问降低了无效计划率并显著提升物理一致性评分。

**⚠️ 局限性**

局限性包括苏格拉底式提问目前依赖自然语言，导致假设与约束的表达存在歧义；仍需人工监督以保证安全与质量；系统尚未具备发明新物理定律的能力，主要在已知物理框架内进行模型优化；可扩展性和在更大规模实验中的自动化执行仍待验证。

---

## 295. Mask to Concept: Auto-Promptable SAM3 via Efficient Test-Time Concept Embedding Search for Few-Shot Annotation

**arXiv ID:** 2606.26711 | [PDF](https://arxiv.org/pdf/2606.26711v1)

**作者:** Quan Zhou `[一作]` (Wuhan University of Technology), Zhiwei Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在医学影像少样本分割任务中，提出 Mask to Concept（M2C）框架，使 SAM3 能在冻结的情况下通过学习概念嵌入实现自动概念分割，并结合 Hybrid Uncertainty Estimation（HUE）实现人机交互闭环注释

**💡 创新点**

创新点在于：①仅利用可学习的概念嵌入在测试时搜索适配的视觉概念，完全不需要外部特征匹配或辅助网络；②通过梯度搜索概念嵌入实现少样本快速适配；③设计混合不确定性度量主动挑选难样本进行人工纠正，形成自我增强的注释循环

**🔧 技术方法**

技术包括 SAM3 的文本概念分割、可学习概念嵌入的梯度搜索、预测熵+概念-几何不一致度量的混合不确定性评估、主动学习循环与人机交互

**📊 数据集**

在 Kvasir‑SEG（内镜息肉分割）和 ISIC‑2017（皮肤病变分割）两个未见过的医学数据集上进行评估

**📈 对比分析**

与六种 SOTA 少样本分割方法（训练基、训练自由）比较，M2C 在 1/5/10 shot 下在两数据集上均获得最高 Dice，尤其 1-shot 约提升 4–12 %；在注释效率上亦优于 Multiverseg 与 SPFS‑SAM

**⚠️ 局限性**

局限性包括：仅在 SAM3 冻结的前提下；对概念嵌入的搜索速度和收敛仍受限；对极端多样性或噪声数据的鲁棒性尚待进一步验证

---

## 296. Kalman Prototypical Networks for Few-shot Fault Detection in Combined Cycle Gas Turbines

**arXiv ID:** 2606.26710 | [PDF](https://arxiv.org/pdf/2606.26710v1)

**作者:** Mohammed Ayalew Belay `[一作]` (Norwegian University of Science and Technology), Pierluigi Salvo Rossi `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种Kalman原型网络（KPN），用于在少样本条件下对组合循环燃气轮机的泄漏故障进行诊断。

**💡 创新点**

创新点在于将Kalman滤波嵌入原型网络，动态平滑原型随训练迭代的变化，从而显著降低样本方差并提升泛化能力。

**🔧 技术方法**

使用基于嵌入的原型学习方法、Kalman滤波器、以及Modelica动态仿真生成的合成传感器数据。

**📊 数据集**

采用高保真Modelica/Dymola仿真产生的海上CCGT传感器时间序列数据，包括正常运行与不同阶段泄漏的工况。

**📈 对比分析**

通过与Prototypical、Matching、Relation、MAML等基准方法对比，在4~8-shot和不同查询规模下，KPN准确率提升约5–10%，方差显著更低。

**⚠️ 局限性**

主要局限包括：基于合成数据，真实工况差异可能影响迁移；Kalman噪声参数需人工调优；目前仅验证二分类，未覆盖多类不平衡情况。

---

## 297. DroidBreaker: Practical and Functional Problem-Space Attacks on Machine-Learning Android Malware Detectors

**arXiv ID:** 2606.26707 | [PDF](https://arxiv.org/pdf/2606.26707v1)

**作者:** Christian Scano `[一作]` (University of Cagliari), Battista Biggio `[通讯]` (University of Cagliari)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

提出一种可构造实用且功能完备的Android恶意软件攻击框架，能够在保持恶意行为的同时绕过机器学习检测器。

**💡 创新点**

创新性地结合了细粒度、构建安全的代码注入与混淆、基于模型影响的变换选择、以及基于运行时日志与API追踪的语义保留测试，实现了高效、功能可验证的对抗样本生成。

**🔧 技术方法**

使用了精细化的API、模块、权限、URL注入与混淆、白盒梯度下降+编码技巧、黑盒遗传算法、Frida+DroidBot动态监测等技术。

**📊 数据集**

在Drebin和AndroBite两大公开恶意样本集上进行评估，并对VirusTotal的70款商业扫描引擎进行实测。

**📈 对比分析**

与Pierazzi、HRAT等先行攻击以及DREBIN、ELSA等检测器对比，攻击成功率高达94%+，查询成本低，仅需十数次请求，侧效极小，功能通过率超过90%。

**⚠️ 局限性**

局限性包括对高度鲁棒或动态特征检测器的效果尚未验证，受限于查询预算与动态分析覆盖率，某些攻击仍可能因构建或运行时错误导致功能丧失。

---

## 298. Beyond Logical Forms: LLM-Extracted Patterns for Fallacy Classification

**arXiv ID:** 2606.26698 | [PDF](https://arxiv.org/pdf/2606.26698v1)

**作者:** Eleni Papadopulos `[一作]` (Politecnico di Torino), Giovanni Da San Martino `[通讯]` (Università di Padova)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型从错误论证及其解释中自适应提取上下文感知的结构模式，并在提示式推理中用于判定逻辑谬误。

**💡 创新点**

提出数据驱动、无监督的模式抽取方法，结合逻辑结构与语言线索，显著提升谬误检测性能。

**🔧 技术方法**

采用大型语言模型（如 GPT‑4o、DeepSeek、Llama 等）进行解释生成、模式提取以及多种提示设计，包括动态一示例、模式匹配等。

**📊 数据集**

主要在 Logic 数据集上训练模式，随后在 Reddit 与 ElecDebate 两个跨领域数据集上评估泛化。

**📈 对比分析**

与零样本、定义、逻辑形式等基线及多种提示方案对比，最佳方案在 Logic 上实现 74.2% 的准确率，明显优于先前无监督方法。

**⚠️ 局限性**

仅基于 Logic 数据集抽取模式，无法覆盖更复杂多元的谬误；所用 LLM 样本有限，且模式生成与推理仍受模型偏见与可解释性限制。

---

## 299. Attributed, But Not Incremental: Cannibalization-Corrected Attribution for Large-Scale Advertising

**arXiv ID:** 2606.26690 | [PDF](https://arxiv.org/pdf/2606.26690v1)

**作者:** Donghui Li `[一作]` (TikTok), Lijing Song `[通讯]` (TikTok)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种实验校准的枯萎修正框架（ETDC+HCA），将稀疏的增量实验结果与实时的生产归因数据融合，以纠正付费广告的“占有”效应；

**💡 创新点**

创新点在于（1）把增量实验作为因果锚点，利用代理变量模型将稀疏的每日提升信号扩展为全时段的校正；（2）通过分层分配机制在保持聚合一致性的前提下，将校正量在业务层级中按比例分配；（3）在生产环境中实现轻量级增量校正而不改动原有归因管道；

**🔧 技术方法**

使用了泛化线性模型（GLM）进行实验到每日提升预测，采用Huber损失与稳健训练；后续分层分配采用基于相对倾向的加权归一化方法；

**📊 数据集**

主要数据集为TikTok全球多市场的付费广告实验结果（约18轮渠道层面A/B实验）以及对应的每日付费归因DNU；

**📈 对比分析**

与原始归因、设备级ML预测等方法对比，ETDC+HCA在渠道层面的绝对相对误差（ARE）从Raw Attribution的1.00下降至0.09，误差降低91.38%，中位数相对误差接近0，四分位区间仅为[-8.11%，7.24%]，显著优于其他方法；

**⚠️ 局限性**

局限性包括对增量实验覆盖度和质量的高度依赖；代理变量假设可能随产品、季节或市场冲击失效；微观层面的校正只能视为分配而非独立因果效应，负的“占有”率被视为诊断信号而非模型输出。

---

## 300. How Can Size and Ceiling Bounds Affect the Complexity of Nonuniform Automata Families?

**arXiv ID:** 2606.26685 | [PDF](https://arxiv.org/pdf/2606.26685v1)

**作者:** Tomoyuki Yamakami `[一作]` `[通讯]` (University of Fukui), Tomoyuki Yamakami (University of Fukui)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究非统一有限状态机和下推自动机族的复杂度，探讨尺寸（state/stack-state）与输入上限（ceiling）对族的复杂度影响，并给出多项包含、等价与分离结果。

**💡 创新点**

提出将尺寸与ceiling视为两大主要因素的框架，证明多种新的等价与分离命题，建立与传统空间有界复杂度类（如SC^k、NSC^k）的深度联系。

**🔧 技术方法**

采用自动机理论与复杂度理论中的技术：非确定性和确定性两路自动机转换、堆栈高度限制、2dft可构造性、Karp‑Lipton 风格的辅助堆栈机与随机访问建议、归约与多项式时间约简、pumping lemma 等。

**📊 数据集**

无实验数据，全部为理论证明；主要使用理论构造的语言族（如图的二分性、匹配字母序列等）作为示例。

**📈 对比分析**

通过证明包含关系与等价关系来评估方法的“性能”，即在复杂度层次上的位置，例如 /⊆2^log^2、2^log^k⊆2^log^k+1 等，表明所研究族的相对复杂度。

**⚠️ 局限性**

局限性包括：许多分离/等价结论仍依赖未证明的假设；缺乏对输入上限为多项式以外情况的完整描述；未给出具体实现或实验验证；对于一向自动机族的结果在有 ceiling 约束下仍不完整。

---

## 301. ConvMemory v3: A Validity Context Layer for Conversational Memory via Target-Conditioned Relation Verification

**arXiv ID:** 2606.26753 | [PDF](https://arxiv.org/pdf/2606.26753v1)

**作者:** Taiheng Pan `[一作]` `[通讯]` (University of Melbourne), Taiheng Pan (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在对话式记忆检索中加入了有效性上下文层，利用目标条件的双证据门实现记忆更新关系的判定，并在检索后输出结构化有效性元数据。

**💡 创新点**

创新点包括：①目标条件的双证据门（MiniLM 与 DeBERTa‑v3 乘积并通过 conservative event/operation 门过滤），②通过零目标标签的实测反馈循环实现从合成结构到真实角色绑定的迁移，③查询条件的可选降级模式和六个可机器验证的安全合同。

**🔧 技术方法**

技术手段包括 MiniLM 和 DeBERTa‑v3 作为 slot heads、乘积式分数、min‑门、noisy‑or 聚合、stratified supervision、real‑data feedback loop、query‑conditioned cross‑encoder 校准器，以及成本感知路由。

**📊 数据集**

使用的数据集有：synthetic multi‑hop validity benchmark（构造的依赖链），Memora 角色绑定转移集（真实对话存储），Memora 稠密当前状态检索（检索当前属性），以及 LoCoMo 进行稀疏有效性审计。

**📈 对比分析**

与传统方法相比：在角色绑定转移中，目标位置规则 78.6%，零样本 NLI 64.2%，相关性 cross‑encoder 17.9%，而本文的 verifier 在合成验证中 90.12%±1.73，迁移到 Memora 达到 98.8%±0.9。稠密检索中，未降级模式 H@1 为 45.1%，而启用目标条件降级模式提升到 95.7%±1.2，保留非被取代记忆召回 99.4%。

**⚠️ 局限性**

局限性：①多跳关系标注在自然对话中稀缺，验证仅在合成结构上完成；②严格前置边构建需反事实必然性判断，当前模型无法可靠实现；③降级仅在稠密当前状态检索中验证，通用检索仍采用默认显式标注；④成本节省依赖数据分布，易样本少时几乎无效。

---

## 302. HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction

**arXiv ID:** 2606.26744 | [PDF](https://arxiv.org/pdf/2606.26744v1)

**作者:** Luxi Lin `[一作]` (ByteDance), Songwei Liu `[通讯]` (ByteDance)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HyperDFlash，一种针对 DeepSeek‑V4 多超连接（MHC）架构的块级并行推理框架，用以改进原生多令牌预测（MTP）和 DFlash 的 draft‑and‑verify 过程。

**💡 创新点**

创新点在于三大设计：1) 采用目标模型的预折叠残差作为条件，保证与 MHC 预测路径的一致性；2) 继承 HC 门控降维器，实现低参数、高效的路径聚合；3) 在早期位置引入 KL 蒸馏目标，提升草稿质量并稳定训练。

**🔧 技术方法**

使用的技术包括块级并行推理（Speculative Decoding）、DFlash、MTP、HC 门控聚合、RMSNorm、KL 知识蒸馏以及 vLLM 推理栈。

**📊 数据集**

训练数据：约 300k 的公开指令/对话/代码数据（主要来自 EagleChat）和 150k 的任务导向数据（如 Evol‑CodeAlpaca）。评估数据集包括 GSM8K、MATH‑500、AIME25、HumanEval、MBPP、LiveCodeBench、MT‑Bench 等。

**📈 对比分析**

对比方法：与原生 MTP（3 步、6 步）和 Vanilla DFlash（6 步）在 Non‑think / Think‑high 模式、温度 0/1 下，分别测量推理 speedup 与平均接受长度 τ。结果显示，HyperDFlash 在非思考模式温度 0 时，平均接受长度从 2.93 提升至 3.69，speedup 从 2.25× 提升至 2.80×；在六步 draft 预算下亦优于 MTP 6 与 Vanilla DFlash，证明其有效性。

**⚠️ 局限性**

限制：未拆分 drafter 与验证成本、批量效应及端到端延迟；KL 蒸馏目标未单独消融；继承降维器假设源、目标与草稿隐藏宽度相同，若不满足则退回通用投影；实验仅在单一 DeepSeek‑V4 目标模型上完成，缺乏公开可复现负载，验证其他 MHC 目标模型的效果仍待进一步研究。

---

## 303. Tight Lower Bounds and Optimal Constructions of Locally Repairable Convertible Codes in the Split Regime

**arXiv ID:** 2606.26742 | [PDF](https://arxiv.org/pdf/2606.26742v1)

**作者:** Haoming Shi `[一作]` (Shandong University), Weijun Fang `[通讯]` (Shandong University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在全局分裂状态下，系统最优距离局部可修复可转换编码（LRCC）之间的转换，重点关注读取带宽成本。

**💡 创新点**

首次通过信息论方法推导出稳定最优距离LRCC的读取带宽下界，并提供了基于MDS阵列编码的构造，达到这些下界。

**🔧 技术方法**

使用信息论方法推导带宽下界，并基于MDS阵列编码构造LRCC。

**📊 数据集**

使用了多种MDS阵列编码，具体数据集未明确提及，但涉及到的参数包括局部维度、全局奇偶校验节点等。

**📈 对比分析**

通过与Maturana和Rashmi的全局分裂构造进行比较，证明了其构造在一般情况下不是带宽最优的，且本研究的构造在给定参数范围内达到最优带宽。

**⚠️ 局限性**

限制在于未考虑全局分裂状态下的带宽下界的更广泛适用性，未来研究可以探讨更强的修复或恢复属性，以及不同类型编码之间的转换。

---

## 304. LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing

**arXiv ID:** 2606.26740 | [PDF](https://arxiv.org/pdf/2606.26740v1)

**作者:** Xinyu Wang `[一作]` (Tsinghua University), Yue Ma `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 LiveEdit，一种实时流式视频编辑框架，能够在保持背景和未编辑区域完整性的前提下，实现因果、分块编辑并且低延迟；

**💡 创新点**

创新点包括：①三阶段逐步蒸馏管道（基础调优→教师强迫→分布匹配蒸馏）将强大的双向扩散模型迁移为高效的单向自回归编辑器；②AR导向的掩码缓存机制，通过 L₂ 余弦距离动态划分编辑区域与背景，从而在自回归推理中仅对活跃像素执行完整前向运算，显著降低计算量；

**🔧 技术方法**

使用技术：双向 Diffusion Transformer（DiT）作为基准模型；教师强迫配合分块因果注意力；分布匹配蒸馏（DMD）实现 4 步推理；AR 掩码缓存（Self‑Attention 层的区域重用）以及 Token Cache；在训练中采用 AdamW、噪声调度、CFG、CLIP 预训练等；

**📊 数据集**

训练数据为 20K 高质量视频‑视频对，来源于 Ditto‑1M 过滤后的子集；实验评测使用 120 对收集的基准数据；

**📈 对比分析**

与现有离线双向编辑模型（LucyEdit、InsV2V、VideoCoF）以及流式生成模型（StreamV2V、StreamDiffusion、StreamDiffusionV2）进行对比；使用 CLIP 文本一致性、LAION 美学评分、VBench 背景一致性、运动平滑度、动态程度和成像质量六项指标；LiveEdit 在文本一致性、动态程度、成像质量等指标上几乎全部领先，帧率约 12.66 FPS（79 ms/帧），大幅提升实时性；

**⚠️ 局限性**

局限性：①仍需 4 步推理，推理速度受限于模型大小；②掩码缓存对背景变化敏感，若背景剧烈变化可能导致误差；③训练成本高，需要多 GPU、数千步；④目前仅针对局部编辑测试，未充分验证大范围全景或大运动场景的鲁棒性；

---

## 305. A Latent ODE Approach to Spatiotemporal Modeling of Cine Cardiac MRI

**arXiv ID:** 2606.26718 | [PDF](https://arxiv.org/pdf/2606.26718v1)

**作者:** David Brüggemann `[一作]` (Swiss Data Science Center), Olga V. Demler `[通讯]` (ETH Zurich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并训练了一种基于网格的连续时间潜在ODE模型，用心率感知的相位重参数化来完整建模心脏MRI的三维时空运动，并将残差潜在向量映射到Cox比例风险模型，预测心衰风险。

**💡 创新点**

创新点包括：①利用心率感知的相位重参数化实现全周期连续动力学；②在变分自编码器中引入协变量调节的先验，捕捉生理差异；③将图神经网络与神经ODE结合，生成全周期一致、解剖结构保持的心室运动；④通过残差潜在空间与Cox模型实现风险评分，提供可解释的风险轴。

**🔧 技术方法**

技术手段包括：图神经网络（GNN）编码器与SpiralNet++解码器、Transformer聚合、神经ODE连续动力学、协变量调节的变分自编码器、心率相位映射、三维网格重建、Cox比例风险回归、Wasserstein距离与MMD等生成质量评估。

**📊 数据集**

使用UK Biobank数据库中的约72,386名受试者的心脏MRI网格数据，包含367例心衰事件。

**📈 对比分析**

通过与多种基线模型（线性+ODE、GNN+Fourier、GNN+Transformer、MeshHeart）以及传统CMR指标和Pooled Cohort equations 进行对比。模型在重构误差、生成真实性（Wasserstein、MMD）和心衰C-index方面表现最佳，C-index从0.704提升到0.785，显示显著的预测改进。

**⚠️ 局限性**

局限性包括：①依赖预先提取的网格与分割，可能受分割误差影响；②未显式强制周期性，周期一致性由模型自学习；③仅考虑心室解剖，未纳入心房、瓣膜等结构；④在UK Biobank中事件数有限，需在更高风险或多中心外部数据上进一步验证。

---

## 306. Random Walk on Bézier Curves for Global Optimization

**arXiv ID:** 2606.26714 | [PDF](https://arxiv.org/pdf/2606.26714v1)

**作者:** Jinpeng Wang `[一作]` (Northwestern Polytechnical University), Yuansheng Gao `[通讯]` (Zhejiang University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于贝塞尔曲线引导的进化优化算法BWE，用于解决连续与离散优化问题。

**💡 创新点**

创新点在于利用贝塞尔曲线进行引导路径搜索，结合随机步长和采样比例提升全局搜索与局部精细化能力。

**🔧 技术方法**

采用随机游走、距离度量、贝塞尔曲线控制点采样等技术。

**📊 数据集**

使用CEC2017、CEC2022 benchmark函数以及五个工程实例（三棍桁架、齿轮传动、悬挑梁、波纹壳舱、滚珠轴承）进行实验。

**📈 对比分析**

通过Wilcoxon秩和检验、Friedman排名和计算成本评估与GA、LEA、HEOA、AOA、SCA、COA、GWO、CMA-ES、LSHADE等算法比较，BWE在绝大多数测试中获得最优或接近最优结果，显著优于传统方法。

**⚠️ 局限性**

局限性在于对高维复杂约束问题的参数调优敏感，对部分难以表示几何特征的工程问题仍需进一步改进。

---

## 307. DeCoFlow: Structural Decomposition of Normalizing Flows for Continual Anomaly Detection

**arXiv ID:** 2606.26687 | [PDF](https://arxiv.org/pdf/2606.26687v1)

**作者:** Hun Im `[一作]` (Seoul National University), Pilsung Kang `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了连续异常检测问题，提出了 DeCoFlow 框架通过将 Normalizing Flow 的子网络拆分为冻结基底和低秩适配器，实现零遗忘的任务增量学习。

**💡 创新点**

创新点在于利用耦合层子网络独立性实现参数隔离，并结合任务特定对齐、辅助耦合层和尾部感知损失等模块克服冻结基底的刚性。

**🔧 技术方法**

采用 Normalizing Flow（基于耦合层）、LoRA 低秩适配器、任务特定归一化、辅助耦合层、尾部感知损失以及原型路由等技术。

**📊 数据集**

在 MVTec-AD 和 VisA 两个工业缺陷数据集上进行实验。

**📈 对比分析**

与联合训练、微调和多种连续学习基线比较，DeCoFlow 在 MVTec-AD 上达到 98.40% I-AUC、58.20% P-AP，VisA 上 93.00% I-AUC、37.00% P-AP，且每任务仅增加 2.27M 参数，遗忘率 0%。

**⚠️ 局限性**

局限包括对细粒度缺陷定位效果不佳、原型路由在类相似度高或长序列时可能失效、以及对更大规模/跨域场景的适用性待验证。

---

## 308. Knowledge-Based Pull Requests: A Trusted Workflow for Agent-Mediated Knowledge Collaboration

**arXiv ID:** 2606.26721 | [PDF](https://arxiv.org/pdf/2606.26721v1)

**作者:** Xinyu Zhang `[一作]` (Fudan University), Weiwei Sun `[通讯]` (Fudan University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出知识驱动拉取请求（KPR）工作流，将外部协作者的本地探索和证据打包为知识包，而非直接提交代码；使用多角色代理（探索、提取、转换、可信内部编码）在项目内部重新生成实现。

**💡 创新点**

创新点在于把跨信任边界的代码与知识分离：通过可信知识包跨越边界，并将实现权交给项目内部的可信编码代理，既保留了项目方对实现的控制，又利用代理自动完成信息提取和翻译。

**🔧 技术方法**

采用多角色大型语言模型（LLM）代理进行语义提取、摘要、证据链接、政策预检，以及内部可信编码代理的代码生成；结合控制实验与可追溯性机制实现安全与可审计。

**📊 数据集**

使用七个公开合并的PR构成的实验数据集，对每个PR生成不同条件的包（正常摘要、KPR包、描述剥离、差异剥离、合成毒化）进行评测。

**📈 对比分析**

比较方法是对每种条件的包进行作者评分（0–2）评估意图、证据、实现、毒化拒绝等属性；结果显示KPR包在意图、证据和实现一致性上普遍获得最高分，毒化条件能完整识别不可信代码，但整体性能仍需在真实项目中进一步验证。

**⚠️ 局限性**

局限性包括实验规模极小、未涉及真实维护者或项目方评估；约束提取细粒度不足、内部重生成的可靠性尚未验证；贡献者和评审者的额外负担、作者信用和许可问题等仍需进一步研究。

---

## 309. Parallel Communicating Finite Automata: The Non-Forgetting Model

**arXiv ID:** 2606.26684 | [PDF](https://arxiv.org/pdf/2606.26684v1)

**作者:** Jana Schulz `[一作]` `[通讯]` (University of Potsdam), Jana Schulz (University of Potsdam)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出并分析了一种新的非遗忘并行通信有限自动机（nfPCFA）模型，并将其与传统的遗忘模型以及多头有限自动机进行系统性比较，给出了各类变体的语言能力等价性或严格包含关系。

**💡 创新点**

创新点在于：①引入非遗忘机制，使得自动机在通信时不丢失自身状态，避免死锁；②构造了完整的模拟方法，证明大多数nfPCFA变体与k头有限自动机等价；③首次证明了确定性集中非返回模式下的严格包含关系，填补了原有研究的空白。

**🔧 技术方法**

主要技术包括：形式化定义和状态转换模型、通信步骤与聚合函数的构造、对多头自动机的模拟与归约、严谨的数学证明（包含构造性证明和对称性论证）以及语言族层级的包含与等价性分析。

**📊 数据集**

论文为理论研究，没有使用实验数据集，所有结果均通过严密的数学证明得到。

**📈 对比分析**

通过构造模拟自动机，将nfPCFA的行为映射到k头非确定性/确定性有限自动机，得到语言族包含关系。实验性比较不适用，性能以语言等价性与包含关系来衡量；结果表明大多数nfPCFA变体与多头自动机具有相同的计算能力，唯有确定性集中非返回模式被证明严格弱于多头自动机。

**⚠️ 局限性**

主要局限包括：①遗忘模型与非遗忘模型在集中模式下的确切关系仍未解决；②对状态数、通信次数等资源效率的定量分析尚缺；③模型的可判定性问题和二向（two‑way）扩展仍未被研究。

---

## 310. NebulaExp-8B: An Empirical Post-Training Pipeline via Full-Scale Ablation Research

**arXiv ID:** 2606.26671 | [PDF](https://arxiv.org/pdf/2606.26671v1)

**作者:** Qiaobo Hao `[一作]`, Chen Zhong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对 Qwen3‑8B 进行完整可复现的后训练管线，结合 SFT、GRPO 强化学习和 OPD 提升指令遵循、数学推理、代码生成与通用知识等多维能力，系统化构建并分层处理 3.84M 训练样本与 200K RL 参考池，开展大规模消融实验。

**💡 创新点**

首次公开 8B 规模后训练的完整流程与消融结果，揭示数据正确性过滤为首要优化因子；阐明指令难度与数理推理的互斥关系；引入多教师 OPD 实现超越单教师与 RL 的性能，并提供细粒度采样与混合比例的可复现策略。

**🔧 技术方法**

使用监督微调 (SFT)、GRPO 强化学习、On‑Policy Distillation (OPD) 与多教师 OPD；构建规则/模型/任务验证过滤器；基于 IFD、长度、熵等指标的加权/分层采样；使用回归预测混合比例；以及基于难度与链式推理长度的自适应训练计划。

**📊 数据集**

SFT 样本来源于 Nemotron‑cascade、ODA‑math‑460K、Nemotron‑Math‑V2、OmniThought‑Math、AM‑Thinking、OpenScienceReasoning、Dolci‑Instruct‑SFT 等公开数据；RL 参考池包含 Dolci‑Instruct‑RL、Eurus‑2‑RL‑Data‑code、RLVR‑IFeval、IF_multi_constraints_upto5 共 200K；评测基准包括 MMLU‑Redux、GPQA‑Diamond、C‑Eval、LiveBench、IFEval、MATH‑500、AIME'24/25、ZebraLogic、AutoLogi、LiveCodeBench‑v5。

**📈 对比分析**

与 Qwen3‑8B‑nothink、Qwen3‑8B‑thinking、Qwen3‑14B、QWQ‑32B 等基线对比，Instruct SFT 取得 60.99（+5.98）平均分，RL 进一步升至 61.85；Reasoning SFT 平均 74.39，RL 后 75.14。数学推理上 AIME'24/25 分别提升 15.00/11.25，代码生成 LiveCodeBench‑v5 提升 4.47；整体显示 8B 规模在高质量后训练下可显著提升多维能力。

**⚠️ 局限性**

对齐性能仍受“对齐税”影响，指令遵循与推理能力存在权衡；RL 仅适用于可验证任务，需大量计算；OPD 受教师质量限制，难以跨域泛化；数据集规模与多样性受限，可能导致泄漏与偏差；以上均限制了模型在更大规模或开放式任务上的进一步提升。

---

## 311. Full spectrum Unlearnable Examples via Spectral Equalization

**arXiv ID:** 2606.26719 | [PDF](https://arxiv.org/pdf/2606.26719v1)

**作者:** Jiale Cai `[一作]` (Western University), Boyu Wang `[通讯]` (Western University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种全频谱不可学习样本（Full‑Spectrum Unlearnable Examples，FUSE）用于训练数据隐私保护。

**💡 创新点**

创新点在于：①识别现有不可学习样本仅依赖高频信息易被低通滤波破解；②通过随机频谱掩码（RSM）和跨频段引导（CBG）实现频谱均衡与跨频段一致性，从而让扰动在整个频谱均匀分布，保持在任何频率抑制下的不可学习性。

**🔧 技术方法**

主要技术包括：频谱均衡（Spectral Equalization）、随机频谱掩码、频谱熵正则化、低高频交叉引导（Semantic Guidance + 结构一致性）、FFT/逆FFT操作及基于特征/感知相似度的损失。

**📊 数据集**

使用公开数据集CIFAR‑10、CIFAR‑100、SVHN进行评估，并在CLIP视觉‑语言模型上做了初步验证。

**📈 对比分析**

与EMN、LSP、TUE、GUE、PUE等基线在低通滤波与无滤波两种设定下对比，FUSE在大多数模型与数据集上均将验证集准确率压至接近随机猜测（≤10%），并在跨模型迁移、JPEG压缩、专门的UE防御等多种攻击场景下保持优越性能。

**⚠️ 局限性**

局限性包括：需要训练扰动生成器，虽然参数量小但仍比无学习器方法略慢；在极端防御或大规模数据集（如ImageNet）上尚未充分验证；对抗性自适应防御（如自适应滤波、生成对抗重构）可能进一步削弱效果。

---

## 312. PressMimic: Pressure-Guided Motion Capture and Control for Humanoid Robot Imitation

**arXiv ID:** 2606.26741 | [PDF](https://arxiv.org/pdf/2606.26741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 313. 'A bit of chaos and madness': The AI Assessment Scale and the work of assessment reform

**arXiv ID:** 2606.26729 | [PDF](https://arxiv.org/pdf/2606.26729v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 314. The Fungible Reserve Standard: A Deterministic Framework for Encoding Carrying Costs in Asset-Backed Tokens

**arXiv ID:** 2606.26704 | [PDF](https://arxiv.org/pdf/2606.26704v1)

**作者:** JJ Jia Jing Tan `[一作]`, Seth Yan `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Fungible Reserve Standard (FRS)，一种在链上确定性地将资产托管成本编码进代币的框架，保持 ERC‑20 可组合性；

**💡 创新点**

通过“经济纯度”原则，将托管成本拆分为单一公开的年化率 c，并用供应补偿机制保留代币持有量不变；

**🔧 技术方法**

利用 Solidity / EVM 可执行的线性扣减公式、供应对账、精度扩展（9 位小数）以及对成本率变更的时间点检查；

**📊 数据集**

本研究为理论设计与概念验证，未使用公开数据集，主要基于传统 ETF 费用率模型和 RWA 估值示例；

**📈 对比分析**

与发行人补贴、rebasing 及 wrapper token 等方案对比，FRS 在成本透明度、余额稳定性、DeFi 可组合性等方面表现优越；实验演示在 EVM 上的 gas 成本低、舍入误差可控；

**⚠️ 局限性**

局限包括需要可信的保留 oracle、未处理信用风险、成本率固定不易动态调整、缺乏监管合规细节等。

---

## 315. Extracting Neural Materials from Multi-view Images

**arXiv ID:** 2606.26715 | [PDF](https://arxiv.org/pdf/2606.26715v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 316. On some Open Problems for Finite Automata with Translucent Input Letters

**arXiv ID:** 2606.26683 | [PDF](https://arxiv.org/pdf/2606.26683v1)

**作者:** Martin Kutrib `[一作]` (University of Giessen), Matthias Wendlandt `[通讯]` (University of Giessen)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文对非返回和返回模式的确定性与非确定性可透视字母有限自动机（FA）进行系统性研究，给出了若干开放闭包性质（如连缀、星闭包、逆同态、逆镜像、补集）的非封闭性证明，并证明了非返回模式下的空性问题是可判定的。

**💡 创新点**

创新点在于首次统一利用可透视字母的跳转特性，构造细致的计数与压缩论证，突破了此前仅针对返回模式或单一确定性/非确定性模型的局限，拓展了对所有四类模型闭包性质的完整表述，并首次给出非返回模型空性判定算法。

**🔧 技术方法**

主要技术包括：1）可透视映射与跳转符号的定义与使用；2）通过模仿多扫迭代有限状态变换（iterated uniform finite‑state transducer）把有限自动机的计算转化为正则语言的生成；3）基于泵引理与状态压缩的循环消除论证；4）对逆同态的构造与逆映像分析。

**📊 数据集**

本研究完全基于理论构造与证明，没有使用实验数据集，所有结论均来自形式语言与自动机理论的严格推导。

**📈 对比分析**

由于是理论性论文，未进行实验对比；相对已有工作，本文提供了更完整的闭包性质表，补全了此前文献中未解的若干闭包问题，空性判定则与既有的可判定性结果保持一致，但针对非返回模式给出了新的可判定性证明。

**⚠️ 局限性**

局限性包括：1）仅讨论一向移动（one‑way）非返回与返回模式，未涉及双向或更复杂的跳转机制；2）空性判定仅针对非返回模式，返回模式已知可判定；3）对更广泛的模型（如可透视单词、带栈的自动机等）仍需进一步研究。

---

## 317. LithoDreamer: A Physics-Informed World Model for Multi-Stage Computational Lithography

**arXiv ID:** 2606.26713 | [PDF](https://arxiv.org/pdf/2606.26713v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 318. 5G NR-V2X Scheduling Approaches for CPM Variable Size Traffic

**arXiv ID:** 2606.26746 | [PDF](https://arxiv.org/pdf/2606.26746v1)

**作者:** Vittorio Todisco `[一作]` (University of Bologna), Alessandro Bazzi `[通讯]` (University of Bologna)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了真实感知消息（CPM）在5G NR‑V2X Mode 2下的调度性能，并比较了多种调度策略（全动态、SPS激进重选、SPS填充、SPS MCS适配）。

**💡 创新点**

发现资源分配的稳定性比即时匹配更关键，提出在分布式V2X场景中优先保持半持续资源分配的设计原则。

**🔧 技术方法**

采用NR‑V2X Mode 2的半持续调度、全动态调度、MCS自适应与填充等技术，并通过WiLabV2XSim框架实现仿真。

**📊 数据集**

使用在高速公路上记录的车辆感知轨迹重构得到的CPM数据集，包含物体数量变化与周期性安全证书开销。

**📈 对比分析**

通过系统级仿真比较PRR、CBR和资源重选率，结果表明SPS填充或MCS适配的方案在不同拥塞条件下比全动态或激进重选方案具有更高的可靠性和更低的重选频率。

**⚠️ 局限性**

研究仅覆盖单一高速公路场景，未考虑重传、链路衰落多样性及安全证书对整体性能的完整影响，限制了结论的通用性。

---

## 319. Algorithmic Foundations of Deep Learning: Complexity-Theoretic Rates and a Characterization of Universal Approximation

**arXiv ID:** 2606.26705 | [PDF](https://arxiv.org/pdf/2606.26705v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 320. PhysEditWorld: A Large-Scale Dataset Toward Physics-Editable World Models

**arXiv ID:** 2606.26694 | [PDF](https://arxiv.org/pdf/2606.26694v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 321. Scientific discovery as meta-optimization: a combinatorial optimization case study

**arXiv ID:** 2606.26728 | [PDF](https://arxiv.org/pdf/2606.26728v1)

**作者:** Yuan-Hang Zhang `[一作]` (University of California San Diego), Massimiliano Di Ventra `[通讯]` (University of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将科学发现视为元优化过程，构建四个LLM代理循环并实现共识目标聚合；通过此框架在数字MemComputing机器上自动发现高效3‑SAT求解器；

**💡 创新点**

创新点在于：①将目标函数本身视为可进化的元目标，并通过相关性加权投票形成自纠错的共识评估；②通过多代理协同实现设计、规划、评估与执行的闭环；

**🔧 技术方法**

使用的技术包括：GPT‑5.2驱动的LLM代理、Kendall τ相关性分析、加权Borda计数共识排序、Monte Carlo图搜索（MCGS）用于设计探索、多分辨率实验调度、HEBO超参数优化；

**📊 数据集**

实验数据集：随机植入式3‑SAT实例，规模从10到1810变量，实例数为100个/规模，句子-变量比α_r=4.3；

**📈 对比分析**

与基线DMM求解器比较，基线扩展指数约N^2.51，最优设计N^1.33；在最大规模N=1810时实现约67倍速度提升；

**⚠️ 局限性**

局限性包括：①共识机制可能强化LLM共享偏差，易产生“回声室”；②系统自治程度有限，过度结构化可能抑制创新；③评估仅在单一实验跑中完成，缺乏跨运行波动量化；④仅验证在植入式3‑SAT，推广性待验证。

---

## 322. MPE-Adam: Multi-Population Evolutionary Optimization with Adam Refinement for QAOA

**arXiv ID:** 2606.26670 | [PDF](https://arxiv.org/pdf/2606.26670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 323. Do Safety Guardrails Need to Reason? LeanGuard: A Fast and Light Approach for Robust Moderation

**arXiv ID:** 2606.26686 | [PDF](https://arxiv.org/pdf/2606.26686v1)

**作者:** Dongbin Na `[一作]` `[通讯]`, Dongbin Na

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过同一模型基底的对照实验，比较了在安全守护任务中链式推理（CoT）与直接标注两种策略，并提出了轻量级、单前向传递的LeanGuard编码器；

**💡 创新点**

创新点在于剔除架构、规模和数据混杂的影响，证明CoT在固定基底上并不提升准确率，同时展示单一前向推理即可匹配甚至超过大规模CoT守护，提出了可开源的LeanGuard模型；

**🔧 技术方法**

技术手段包括使用ModernBERT‑large编码器与三头线性分类器、LoRA对大模型（Llama‑3.2‑1B、T5‑base）进行参数高效微调、对照CoT训练（R‑SFT、HS‑DPO）与仅标注训练；

**📊 数据集**

数据集涵盖GuardReasoner的127,465条对话级例子（含CoT轨迹）以及公开的安全评测基准（ToxicChat、OpenAI Moderation、AegisSafetyTest、SimpleSafetyTests、HarmBenchPrompt/Response、WildGuardTest等）；

**📈 对比分析**

比较方法为同一底层模型仅变换推理方式，评估F1、召回率、FPR等指标，结果显示LeanGuard在395M参数下达到82.90±0.26 F1，匹配1B CoT守护，同时推理成本降低约100倍；

**⚠️ 局限性**

局限性包括：仅针对单标签安全判断任务，CoT在更复杂多步或符号推理任务上可能仍有优势；跨架构比较仍受预训练差异影响；LeanGuard对极端噪声或领域迁移的鲁棒性未进一步验证。

---

## 324. 2-Head 2D Returning Finite Automata

**arXiv ID:** 2606.26680 | [PDF](https://arxiv.org/pdf/2606.26680v1)

**作者:** Henning Fernau `[一作]` (Universität Trier), D. Gnanaraj Thomas `[通讯]` (Madras Christian College)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并研究了二维图像上的两头返回有限自动机（2‑HRFA）及其同步步进变体（B2‑HRFA），并对其语言类与已知模型的包含与不可比关系以及闭包性质进行了理论分析。

**💡 创新点**

创新点在于定义了两头相反方向扫描的返回自动机，并揭示其语言类严格介于 RFA 与 RPDA 之间，同时 B2‑HRFA 构造提供了新的同步限制模式，扩展了二维自动机的表达能力。

**🔧 技术方法**

采用了图像语言、矩阵文法、有限自动机的理论框架，并利用变换、抽象以及归约方法证明了语言包含关系、不可比性与闭包性质。

**📊 数据集**

无实验数据集，全部为形式化理论证明。

**📈 对比分析**

通过构造映射和归约证明，将 2‑HRFA 语言类与 RFA、RPDA、CFMG 等的关系与闭包性质进行比较；结果表明其包含关系为 ℒ(RFA)⊊ℒ(B2HRFA)⊊ℒ(2HRFA)⊊ℒ(RPDA)，且与 CFMG、T(ℒ(CFMG)) 不可比。

**⚠️ 局限性**

局限性包括：2‑HRFA 与 B2‑HRFA 的可判定性问题仍未完全解决，某些闭包属性（如旋转 ±90°、转置）不成立；且对非空性、等价性等决定性问题的复杂度未给出完整分析。

---

## 325. Intracranial Aneurysm Classification and Segmentation via Tri-Axial ROI and Multi-Task Learning

**arXiv ID:** 2606.26706 | [PDF](https://arxiv.org/pdf/2606.26706v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 326. Learning Motion Feasibility from Point Clouds in Cluttered Environments

**arXiv ID:** 2606.26700 | [PDF](https://arxiv.org/pdf/2606.26700v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 327. Selective Memoization for Efficient Backtracking Regular Expression Matching

**arXiv ID:** 2606.26678 | [PDF](https://arxiv.org/pdf/2606.26678v1)

**作者:** Martin Berglund `[一作]` (Umeå University), Iain le Roux `[通讯]` (Stellenbosch University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于最小反馈节点集合的选择性记忆化方案，旨在消除回溯正则表达式匹配中的指数级耗时。

**💡 创新点**

创新点在于将NFA的最小反馈顶点集作为记忆化基准，并通过解析树实现线性时间算法，从而获得更小的记忆化状态集。

**🔧 技术方法**

采用了解析树驱动的Thompson与Glushkov NFA最小反馈节点识别算法，以及基于DFS的反馈集计算与判定IDA的技术。

**📊 数据集**

实验使用了Polyglot语法库中超过四十万条正则表达式，并结合Xeger生成的正负样本字符串进行匹配。

**📈 对比分析**

通过与Inclusion、Closure、Key-Closure等现有记忆化方案对比，MFN在大多数情况下记忆化条目更少、匹配步骤相近或略增，显著降低内存占用。

**⚠️ 局限性**

局限在于对ε循环的处理仍需分离，且对Glushkov NFA最优性的证明尚未完成，实验结果受非IDA表达式比例影响。

---

## 328. The Model Checking Problem for Distributed Knowing How is $Δ^p_2$-Complete

**arXiv ID:** 2606.26709 | [PDF](https://arxiv.org/pdf/2606.26709v1)

**作者:** Ziqi Wang `[一作]` (University of Amsterdam), Ronald de Haan `[通讯]` (University of Amsterdam)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了分布式知道如何（Distributed Knowing How）逻辑的模型检测问题，并证明其在复杂度阶层 Δ₂^p 上是完全的；同时给出了对应的多项式时间 + NP oracle 的判定算法。

**💡 创新点**

创新点在于：①提出了一个统一的分布式知道如何逻辑框架，兼容先前的单步与多步行动形式；②首次给出了该逻辑模型检测的精确复杂度界定；③设计了一种利用等价类商化与递归修正的 fixpoint 算法，克服了分布式行动集合构造难题；④通过从 Δ₂^p‑完备的 SNSAT 问题构造多层模型，实现了下界证明。

**🔧 技术方法**

主要技术包括：模型检查的 bottom‑up 评估；对 Kh_G 语义的处理采用在等价类商集上进行的递增 fixpoint 迭代；每一步检查使用 NP oracle 来判定是否存在满足条件的分布式行动；利用分布式行动的扁平化性质与子集划分；以及构造复杂度下界的递归归约与策略构造。

**📊 数据集**

由于是理论研究，实验数据集并未使用；所有结论均基于形式化证明与多层模型构造。

**📈 对比分析**

与以往仅给出上界或下界的研究不同，本工作给出了完整的 Δ₂^p‑完备性证明，表明该问题既可在 P^NP（Δ₂^p）内部求解，又至少与已知的 Δ₂^p‑完备问题同等难度；因此在理论复杂度上与其他已知逻辑（如知识与可能性）形成对比，展现了更高的求解成本。

**⚠️ 局限性**

局限性：①仅讨论了模型检测问题； satisfiability（可满足性）问题尚未解决；②算法实现依赖于 NP oracle，实际运行成本在现实系统中可能不可行；③对分布式行动集合的显式列举仍导致输入规模可能爆炸，未给出紧凑编码方案。

---

## 329. SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow

**arXiv ID:** 2606.26701 | [PDF](https://arxiv.org/pdf/2606.26701v1)

**作者:** Xinrui Wu `[一作]` (University of California), Tony Nowatzki `[通讯]` (University of California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的动态数据流Segment，并设计了相应的加速器SegFold，用于高效执行稀疏矩阵乘法（SpGEMM）。

**💡 创新点**

通过在单个tile内引入细粒度动态调度（SelectA）和动态映射（SegmentBC），同时结合分层折叠技术，实现了对输入输出的重用和负载均衡的同时优化。

**🔧 技术方法**

采用动态窗口调度、IPM查找表的自适应合并网络、向量多播网络以及空间/时间折叠技术，并在7nm ASIC上实现。

**📊 数据集**

使用SuiteSparse中的15个稀疏矩阵（包括不同尺寸、密度和形状）以及合成稀疏矩阵进行评估。

**📈 对比分析**

与最优的Flexagon和Spada加速器在相同硬件与内存模型下进行对比，SegFold在所有工作负载上几何平均提升1.95×相对Spada、5.3×相对Flexagon；在极稀疏或高密度矩阵中表现尤为突出。

**⚠️ 局限性**

对高度稀疏但具有极端行密度分布的矩阵（如ca-GrQc）性能低于Spada；对宽矩阵时需要手动转置或选择乘法方向；在极高密度情况下，动态映射开销减弱但仍占用额外资源。

---

## 330. From Content to Strategy: Understanding the Motivations, Processes, and Impacts of AI-Guided Communication

**arXiv ID:** 2606.26672 | [PDF](https://arxiv.org/pdf/2606.26672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 331. Idefix-Free Languages and Their Application in External Contextual Grammars

**arXiv ID:** 2606.26682 | [PDF](https://arxiv.org/pdf/2606.26682v1)

**作者:** Marvin Ködding `[一作]` (Pädagogische Hochschule Heidelberg), Bianca Truthe `[通讯]` (Universität Giessen)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文通过引入前缀-后缀-中缀-自由语言（idefix-free语言）扩展了子正则语言家族，并研究了它们在已有层次结构中的位置；随后分析了以这些子正则语言为选择语言的外部上下文文法的生成能力，给出了新的包含关系和不可比性。

**💡 创新点**

创新点在于：1) 定义并系统化了idefix-free语言族；2) 在子正则层次中插入该族，绘制完整的包含图；3) 证明了外部上下文文法中以idefix-free、prefix-free、suffix-free为选择语言时的生成能力，揭示了与有限语言族、无限语言族的等价与区别。

**🔧 技术方法**

主要技术包括：形式语言理论的闭包与运算定义、构造性证明与取证语言（witness languages）、抽象的Pigeonhole原理推理、非计数与无后缀/前缀性质的泵引理、以及上下文文法的外部推导模型。

**📊 数据集**

由于研究完全是理论性的，未使用实验数据集；所有结论均基于语言族定义与形式证明。

**📈 对比分析**

通过比较不同子正则语言族之间的包含关系（如正则→非计数→无后缀→前缀自由等）以及文法生成能力的包含关系，作者用具体的构造语言作为示例证明了包含关系的严密性；并在层次图中标注了所有已知的包含或不可比关系。

**⚠️ 局限性**

局限性包括：仅讨论外部上下文文法，内部模式的完整层次仍未完成；对部分子正则族（如非初始、中央等变体）的生成能力仍待研究；同时，关于可判定性与复杂度的问题尚未解决。

---

## 332. Efficient Regex Matching with Sparse Counting-Sets

**arXiv ID:** 2606.26679 | [PDF](https://arxiv.org/pdf/2606.26679v1)

**作者:** Martin Berglund `[一作]` (Umeå University), Sicheol Sung `[通讯]` (Yonsei University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 c-configs 与稀疏计数集（sparse counting‑set）用于高效匹配带计数操作的正则表达式（c‑regex）

**💡 创新点**

创新点在于将计数集压缩为只保留必要计数值，从而在匹配时避免复制大型计数集，保持接受行为不变，并给出了 d‑稀疏定义和对应的时间复杂度分析

**🔧 技术方法**

采用计数器自动机（counter automaton）、位置构造、计数集数据结构及其稀疏版本进行实现

**📊 数据集**

实验使用 Polyglot 正则语料库（511,196 条模式，其中 30,833 条含计数）和 Snort3 规则集（9,755 条模式，其中 2,837 条含计数）

**📈 对比分析**

与四种基线方法（计数展开+Thompson NFA、super‑config、标准 c‑config、稀疏 c‑config）比较，实验表明稀疏 c‑config 在大多数模式下实现线性匹配，尤其对复制型和 d‑稀疏模式明显优于其他方法，性能提升可达数十倍

**⚠️ 局限性**

局限在于仅处理平坦 c‑regex；非平坦表达式需展开且展开策略影响效率；稀疏计数集在 h‑l+1 = 1 的计数操作无优势，且实现中未对特殊范围（如 r^{0,h}、r^{1,h}、r^{l,∞}）做进一步优化

---

## 333. How Long Can the Escaping Ant Be Confined?

**arXiv ID:** 2606.26677 | [PDF](https://arxiv.org/pdf/2606.26677v1)

**作者:** Kossi Roland Etse `[一作]` `[通讯]` (Université Paris Saclay), Kossi Roland Etse (Université Paris Saclay)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究Langton蚁在有限网格上的逃逸时间与可重复执行次数，给出不同尺寸网格的逃逸上界；

**💡 创新点**

创新点在于将逃逸时间与网格结构分解为列类型（初始、反弹、穿越、退出）并证明单次穿越后不再可重复的阻塞性质，从而得到紧确的逃逸上界；

**🔧 技术方法**

使用了离散动力学分析、归约分治与递推方法，以及Langton蚁的基准转移规则；

**📊 数据集**

并未使用外部数据集，而是通过构造所有可能的网格配置与蚁的起始状态进行理论推导与枚举；

**📈 对比分析**

与之前已知的逃逸下界（如2×n网格可达6(n-1)步）进行对比，证明给出的上界在该范围内是最优的；

**⚠️ 局限性**

局限性是仅针对二维正方形网格中的两行与三行情况，尚未推广到更宽网格或更复杂拓扑结构。

---

## 334. Liquid Fusion of Heterogeneous Representations Towards General Salient Object Detection

**arXiv ID:** 2606.26849 | [PDF](https://arxiv.org/pdf/2606.26849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 335. Grouped Reverse Importance Sampling for the Partition Function

**arXiv ID:** 2606.26748 | [PDF](https://arxiv.org/pdf/2606.26748v1)

**作者:** Neri Merhav `[一作]` `[通讯]` (Technion Israel Institute of Technology), Neri Merhav (Technion Israel Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了分组逆重要性采样（GRIS）方法，用于在已获得Boltzmann分布样本的情况下估计分区函数，并对其均方误差（MSE）进行理论分析与数值验证。

**💡 创新点**

创新点包括①证明任何能提升普通RIS的权重函数必须在组内耦合；②对非重叠组证明仅需考虑依赖组能量的权重函数即可获得最优；③将多维积分降为一维，并引入固定权重滑动窗口（FSW）和可变权重滑动窗口（VSW）两种分组变体，进一步降低MSE。

**🔧 技术方法**

使用的技术包括逆重要性采样、Rao-Blackwell化、卡方距离、能量密度函数（Ω_k）的卷积与傅里叶变换、Saddlepoint近似以及数值优化。

**📊 数据集**

实验数据集为三个一维示例（U(x)=|x|、|x|^{3/2}、(x^2-1)^2）以及受扰动Hamiltonian示例，所有分区函数可通过解析或数值积分获得。

**📈 对比分析**

方法与标准k=1 RIS、FSW和VSW在同一数据集上比较，结果显示k=2、3组可将MSE降低20–65%，FSW再降低13–14%，VSW进一步提升30–40%；在k=10时MSE已降至初始值的约18%。

**⚠️ 局限性**

限制包括：只在独立同分布样本下有效；对重叠窗口的理论分析尚未完全完成；高维、多块样本或非i.i.d.链等情况仍是开放的研究方向。

---

## 336. Depth-Semantic Alignment and Affinity-Guided Fusion for Structured Radar Point Cloud Generation

**arXiv ID:** 2606.26743 | [PDF](https://arxiv.org/pdf/2606.26743v1)

**作者:** Amjad Hussain `[一作]` (Zhejiang University), Wenjie Liu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种基于视觉雷达融合的毫米波雷达点云生成框架，显著提升点云稠密度和结构完整性；

**💡 创新点**

创新点在于将Hessian峰值增强、深度-语义先验对齐、雷达引导的亲和传播与图优化后处理三大技术耦合，实现跨模态精准对齐与稀疏补全；

**🔧 技术方法**

采用Hessian峰值增强、Lift‑Splat‑Shoot（LSS）深度‑语义分解、BEV空间多模态特征融合、稀疏图优化与CFAR参考点约束等技术；

**📊 数据集**

使用公开的自动驾驶雷达+摄像头+LiDAR数据集（如nuScenes或Waymo）进行训练与评估；

**📈 对比分析**

与OS‑CFAR、Sparse2Dense、SGDNet相比，在SECOND与PointPillars检测器上AP_30/AP_50提升约5–7%（如AP_30从26.3升至31.9，AP_50从32.0升至38.0），在SECOND跟踪器上MOTA从0.108升至0.494，AUC从0.112升至0.572，性能均显著优于对比方法；

**⚠️ 局限性**

存在对摄像头-雷达标定高度依赖、视觉遮挡与光照变化对深度估计的影响、算法计算复杂度较高等局限，需要进一步提升鲁棒性与实时性。

---

## 337. Multi-modality Image Fusion under Adverse Weather: Mask-Guided Feature Restoration and Interaction

**arXiv ID:** 2606.26812 | [PDF](https://arxiv.org/pdf/2606.26812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 338. SSI-Policy: Learning Structured Scene Interfaces for Vision-Language Robotic Manipulation

**arXiv ID:** 2606.26800 | [PDF](https://arxiv.org/pdf/2606.26800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 339. PlanRL: A Trajectory Planning Architecture for Reinforcement Learning-based Driving Experts

**arXiv ID:** 2606.26858 | [PDF](https://arxiv.org/pdf/2606.26858v1)

**作者:** Joonhee Lim `[一作]` (Korea Advanced Institute of Science and Technology), Dongsuk Kum `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合Frenet坐标系和运动学可行性检查的轨迹规划架构，用于强化学习驱动专家，取代传统直接输出控制指令的模式；

**💡 创新点**

创新点在于将RL策略输出高层指令后通过可行性检查生成可执行轨迹，显著提升可解释性、降低累计跟踪误差，并实现与现代端到端规划架构的无缝对接；

**🔧 技术方法**

使用Roach的PPO强化学习框架、Beta分布动作空间、BEV语义分割图像+六维测量向量作为观测，Frenet坐标轨迹规划、五次多项式侧向规划、二次多项式纵向规划及运动学可行性检查；

**📊 数据集**

在CARLA 0.9.10.1环境下评测Offline Leaderboard v1（76条路线）和NoCrash（Town01、Town02）两大基准；

**📈 对比分析**

与Autopilot、Roach、Roach+Rule、CaRL等基线对比，本文方法在Driving Score提升5–11点，成功率提升8–19点，整体性能明显优于现有控制式RL专家；

**⚠️ 局限性**

局限性包括：对动态障碍物的反应仍受限于RL策略输出，过度依赖仿真环境的精确地图信息，尚未在更大规模或更真实的Benchmark（如Bench2Drive、Longest6）上验证通用性；

---

## 340. wav2tok 2.0: Scalable Audio Tokenization Maintaining Explicit Pairwise Token Alignment for Efficient Audio Retrieval

**arXiv ID:** 2606.26824 | [PDF](https://arxiv.org/pdf/2606.26824v1)

**作者:** Adhiraj Banerjee `[一作]` (Indian Institute of Technology), Vipul Arora `[通讯]` (KU Leuven)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种可扩展的检索导向语音分词器 wav2tok 2.0，采用分阶段训练先用对比学习与向量量化学习鲁棒表示，再通过 CTC 对齐损失和新的 DTW 对齐帧级预测损失实现显式的成对对齐。

**💡 创新点**

创新点在于：1）将 BEST-STD 的高效编码器与 wav2tok 的显式对齐机制结合，实现大规模训练；2）引入 DTW 对齐的帧级预测目标，细化跨说话人和跨时长的 token 一致性；3）通过自适应权重调节 CTC 损失，保证优化稳定。

**🔧 技术方法**

使用了 Mamba 变体的双向状态空间模型作为编码器，SimCLR 对比损失、向量量化、CTC 对齐损失、DTW 对齐帧级预测损失以及自适应权重机制。

**📊 数据集**

主要在 LibriSpeech 子集上训练和评估，并在未见的 TIMIT 数据集上检验跨域泛化，查询集合包括 IV（词表内）和 OOV（词表外）两种。

**📈 对比分析**

与 BEST-STD、wav2tok、HuBERT、WavLM、SpeechTokenizer、EnCodec 等基线相比，wav2tok 2.0 在 MAP、MRR、MTWV 等 QbE-STD 指标上均实现显著提升，尤其在 bigram 一致性和 OOV 召回方面表现最优；在 TIMIT 上虽仍有性能下降，但仍保持领先。

**⚠️ 局限性**

局限性包括：1）在极端分布偏移（如噪声、不同语言）下仍表现出一定衰退；2）对非常大词表时精度会略有下降，需权衡辨别性与一致性；3）未加入 OT‑基代码簿平衡等正则化，未来可进一步提升多语种和嘈杂环境下的鲁棒性。

---

## 341. LearniBridge: Learnable Calibration of Feature Caching for Diffusion Models Acceleration

**arXiv ID:** 2606.26778 | [PDF](https://arxiv.org/pdf/2606.26778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 342. Memory Depth, Not Memory Access: Selective Parametric Consolidation for Long-Running Language Agents

**arXiv ID:** 2606.26806 | [PDF](https://arxiv.org/pdf/2606.26806v1)

**作者:** Haoliang Han `[一作]` `[通讯]` (China Pharmaceutical University), Haoliang Han (China Pharmaceutical University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了长期语言代理的“记忆深度”，提出循环漂移协议和基于惊讶-情感门控的选择性参数化整合机制，并在 GPT‑2、TinyLlama 与 Mistral‑7B 上评估其在检索与目标保持方面的表现。

**💡 创新点**

将检索访问与持久性记忆区分，引入循环漂移协议以隔离上下文卸载后的行为持久性，并证明选择性整合由选择与作用两个可控维度组成。

**🔧 技术方法**

使用惊讶‑情感门控 (EVAF) 过滤事件，低秩适配器 (LoRA) 进行参数化写入，配合回放与 L2 锚点抑制漂移，并在不同模型上进行固定内循环控制实验。

**📊 数据集**

采用合成用户流（包含稳定目标、干扰、临时相反请求等）和公开 Memora 事件流作为外部诊断数据。

**📈 对比分析**

通过循环漂移协议将检索、路由+EVAF、Naive‑LoRA 等方法在短事实、长事实、目标保持与后卸载恢复四个探针上对比；EVAF 在目标保持和后卸载恢复上明显优于检索，写入量仅 2‑3 次/200 事件，且 L2 漂移低。

**⚠️ 局限性**

机制仅提升目标保持而非整体记忆准确性，选择与作用的校准需模型依赖，误校准会导致性能倒退；在 Memora 诊断中未显著改善旧记忆失效，且对高规模模型的高污染率仍是挑战。

---

## 343. AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing

**arXiv ID:** 2606.26787 | [PDF](https://arxiv.org/pdf/2606.26787v1)

**作者:** Chennan Ma `[一作]` (Taobao & Tmall Group of Alibaba), Keping Yang `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 AIGP，一个结合 LLM 与离线强化学习的动态定价框架，支持解释性与业务价值对齐。

**💡 创新点**

创新点包括：①将 LLM 的链式思考与业务上下文融合；②构建离线 RL 的长周期价值评估器 LTVE 用于生成偏好对；③利用直接偏好优化 DPO 与 LLM 结合，提升长期 GMV/ROI；④通过教师-学生蒸馏和控制标记实现可部署的高质量小模型。

**🔧 技术方法**

使用技术：大型语言模型（Qwen3），监督微调 (SFT)，离线强化学习（Critic-only 双 Q），长周期价值估计，直接偏好优化（DPO），LLM-as-Judge 评估，低秩适配 LoRA，控制 token。

**📊 数据集**

数据集：Tao Factory 180 天的 SKU 历史交易日志（约 5M 过渡），包含结构化销量、库存、ROI 等以及产品描述、用户评论等非结构化文本；用于离线 RL、SFT 与 DPO 的训练。

**📈 对比分析**

对比方法：与基线 Price‑Sales、RL‑DT、学术 RL（DDPG、SAC）和强大开源 LLM 进行离线评估（Q‑Score、EAMA 等）和 60 天线上 A/B 测试；AIGP 在 GMV、ROI、里程碑达成率上分别提升 13.21%、7.59%、8.20%。

**⚠️ 局限性**

局限性：LTVE 依赖离线数据，可能对极端冷启动或边界动作有误估；DPO 需要高质量偏好对，构建成本高；模型对业务规则约束敏感，部署后需持续监控；未验证跨平台/跨行业的可迁移性。

---

## 344. Event-based Gaze Control System for Accurate Real-time Spin Estimation in Professional Ball Games

**arXiv ID:** 2606.26780 | [PDF](https://arxiv.org/pdf/2606.26780v1)

**作者:** Yunpu Hu `[一作]` (Sony AI), Naoya Takahashi `[通讯]` (Sony AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套基于事件相机的主动视觉系统，实时跟踪未改装的球并精确估计其旋转速度和轴向。

**💡 创新点**

核心创新包括：① 将焦距可调长焦镜头与伺服电光镜结合，实现对高速球的高空间分辨率跟踪；② 采用球面对比最大化（s-CMax）模型，避免平面投影导致的深度歧义；③ 引入不确定性感知的CNN与GPU并行对比最大化后处理，实现低延迟且自适应的在线估计；④ 多视角融合提升在遮挡或纹理不可见时的鲁棒性。

**🔧 技术方法**

使用的技术包括：事件相机（Prophesee EVK4）、焦距可调液晶镜头（Optotune EL-10-30）、伺服电光镜（Thorlabs GVS012/M）、YOLO事件检测、光流初始化、球面对比最大化、ResNet-18的不确定性回归、GPU批量优化、Kalman滤波器与非线性运动模型。

**📊 数据集**

实验数据集涵盖三类：① 电机旋转球（Spinner）配备编码器和AprilTag，用于精准标注；② 采用弹射器的球投掷数据，使用SpinDOE生成伪标注；③ 真实职业乒乓球比赛的多视角事件流，作为最具挑战的实时场景。

**📈 对比分析**

与传统光流和静态对比最大化基线相比，离线方法在多种球类上的旋转幅度误差降至约2.1%（轴误差≈4°），显著优于先前方法（>10%幅度误差）。在线方法在职业比赛中实现3 ms延迟、750 Hz吞吐率，旋转幅度误差≈8.8%（轴误差≈6.4°），与基准相比误差下降50%+，且能提供可靠的不确定性估计。

**⚠️ 局限性**

局限性包括：① 对球面纹理可见度高度依赖，遮挡或纹理弱化会导致误差上升；② 高旋转速时CNN对幅度预测趋于保守，需更多高速样本；③ 需要针对不同球种进行一次性训练和标注；④ 系统成本高，需专用事件相机和光学组件。

---

## 345. Capacity-Controlled Multi-View Stylization of 3D Gaussian Splatting

**arXiv ID:** 2606.26754 | [PDF](https://arxiv.org/pdf/2606.26754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 346. Evaluation Pitfalls and Challenges in Multimedia Event Extraction

**arXiv ID:** 2606.26775 | [PDF](https://arxiv.org/pdf/2606.26775v1)

**作者:** Philipp Seeberger `[一作]` (Technische Hochschule Nürnberg Georg Simon Ohm), Korbinian Riedhammer `[通讯]` (Technische Hochschule Nürnberg Georg Simon Ohm)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统地分析了多模态事件抽取（MEE）评估中的三大类陷阱（数据处理不一致、任务假设不一致、评估设置过于宽松），并基于此提出了严格的评估框架 StrictEval。

**💡 创新点**

首次完整剖析 MEE 评估问题，构建了可复现的 StrictEval 框架，并通过控制实验量化评估设置对性能的显著影响，揭示了先前研究中存在的过度估计现象。

**🔧 技术方法**

使用 Controlled Experiments、单任务模型、CLIP 基础的跨模态核心ference、Bipartite Matching 等技术，对 M2E2 基准进行系统评测和复现。

**📊 数据集**

主要使用公开的 M2E2 benchmark（文本+图像），并在训练阶段利用 ACE、SWiG、VOA 等外部数据集。

**📈 对比分析**

在原始评估与 StrictEval 两套设置下复现并对比了约 15 篇 MEE 方法，结果显示 StrictEval 下 F1 分数平均下降 30-60% 级别，说明先前报告的高分大多受评估设置影响。

**⚠️ 局限性**

局限性：仅针对 M2E2 进行分析，未覆盖视频/音频等其他模态；使用的模型较为简单，无法覆盖所有新兴 MLLM 方案；StrictEval 的严格性导致结果低，直接与旧方法比较存在难度。

---

## 347. ProtoKV: Streaming Video Understanding under Delayed Query with Summary-State Memory

**arXiv ID:** 2606.26762 | [PDF](https://arxiv.org/pdf/2606.26762v1)

**作者:** Le Tu Ngoc Minh `[一作]` (KAIST), Dongsu Han `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ProtoKV，一种常数内存的 KV 缓存机制，用于在流式视频理解中持续保持关键视觉信息。

**💡 创新点**

创新点在于将远程历史压缩为对象中心化的原型库，并通过残差统计与质量加权生成可直接用于注意力的伪 KV token，兼顾实时查询与内存上限。

**🔧 技术方法**

技术包括两层在线状态（近窗口 KV + 远程原型库）、连续性感知的原型分配、残差统计（PQ+直方图）、质量加权偏置和伪 token 合成。

**📊 数据集**

使用 RVS‑Ego、RVS‑Movie、OVO‑Bench、StreamingBench 四个流式视频基准，及 VideoMME/MLVU 作为离线评测。

**📈 对比分析**

与滑动窗口 (SWA) 和在线 token‑保留 (InfiniPot‑V) 基线比较，ProtoKV 在相同 GPU 内存预算下实现更高准确率，且在长查询延迟时更稳健，提升幅度可达 +20 pp。

**⚠️ 局限性**

局限在于固定原型容量会导致相似事件被合并，难以区分重复出现的细粒度动作，且对离线序列全局重采样的适应性有限。

---

## 348. EGG: An Expert-Guided Agent Framework for Kernel Generation

**arXiv ID:** 2606.26758 | [PDF](https://arxiv.org/pdf/2606.26758v1)

**作者:** Yaochen Han `[一作]` (Beihang University), Yixiang Zhang `[通讯]` (Beihang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Expert-Guided Agent Framework (EGG)，用于自动生成高性能 GPU 内核。

**💡 创新点**

创新点在于将 kernel 生成分为算法结构设计与硬件特定调优两阶段，并在每个阶段采用阶段感知的多代理协作（Code、Profile、Debug）来稳定、累积地提升性能。

**🔧 技术方法**

核心技术包括：LLM（GPT‑5.1）生成 Triton 代码、专家优化原则指导的多种搜索策略（多种种子、算法细化）、三阶段硬件调优（并行映射、张量分块、内存优化）以及结构化上下文管理的多代理交互。

**📊 数据集**

使用的基准数据集为 KernelBench（250 个 kernel，分三级难度）以及实测的 TritonBench 真实工作负载。

**📈 对比分析**

与 PyTorch Eager、Torch Compile、ChatGPT‑5.1、DeepSeek‑V3.2、AutoTriton、CudaForge 等方法对比，EGG 取得 100% 成功率、最高 Fast_1 率（≥72%）、平均 2.13× 的速度提升（相较于 PyTorch Eager）并显著低于同类 RL/多代理方法的算子生成成本。

**⚠️ 局限性**

局限性包括：仍需依赖昂贵的 LLM 推理成本；对不同 GPU 架构迁移时需手工调整专家规则；在极大规模多任务环境下的并发效率尚未充分验证。

---

## 349. Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"

**arXiv ID:** 2606.26783 | [PDF](https://arxiv.org/pdf/2606.26783v1)

**作者:** Ananth K S `[一作]` (Independent), Arya Hariharan `[通讯]` (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对AlphaEdit进行可复现性研究，先在原始模型（Llama3-8B、GPT2-XL、GPT-J）上复现其编辑效果，随后将其扩展到五个新模型（Llama3.2-1B/3B、Qwen2.5-3B、Phi3-3.8B、Gemma2-2B），测试编辑规模从2,000到10,000次，并在三个额外基准（BoolQ、HellaSwag、XSTest）上评估编辑后模型的通用能力与安全拒绝行为。

**💡 创新点**

创新点在于：1）提供了AlphaEdit的完整可复现实验代码与结果；2）系统检验了其在不同模型架构、编辑规模以及下游任务上的适用性；3）揭示了AlphaEdit在大规模编辑时的性能衰减阈值与对新架构的不适用性，指出其理论保证的局限。

**🔧 技术方法**

主要技术包括：Locate‑then‑Edit框架、AlphaEdit的null‑space投影优化（使用SVD求得投影矩阵并闭式求解最优更新），以及在不同模型上执行的批量顺序编辑流程；评估使用标准编辑指标（Efficacy、Generalization、Specificity、Fluency、Consistency）和下游任务指标（GLUE、BoolQ、HellaSwag、XSTest）。

**📊 数据集**

使用的数据集包括原始AlphaEdit评测集CounterFact与ZsRE；扩展评测基准为BoolQ、HellaSwag、XSTest；以及GLUE六个任务（SST、MRPC、MMLU、RTE、CoLA、NLI）。

**📈 对比分析**

对比方法为原始AlphaEdit（无投影）与加入null‑space投影的AlphaEdit*，在原模型上复现后与论文结果基本一致；在新模型上，AlphaEdit在Llama3.2、Qwen2.5上表现优异，但在Phi3和Gemma2上几乎无效；编辑规模扩至10,000次时，所有模型在5,000次后出现性能下降，说明理论保证是有界的；在额外基准上，随着编辑次数增加，模型的通用能力与安全拒绝行为均出现明显衰退。

**⚠️ 局限性**

局限性包括：1）对生成质量（Fluency、Consistency）的评估受限于代码实现问题，导致编辑后文本质量下降未能完全解释；2）AlphaEdit对非Llama式架构（如融合矩阵、RMSNorm）不具通用性，需进一步适配；3）编辑规模的上限与阈值受批处理方式与编辑顺序影响，未进行全面探索；4）仅使用零样本推理进行下游评测，未探讨更高级提示策略的恢复潜力。

---

## 350. LCAi: Life Cycle Assessment with big data fusion and retrieval-augmented generation-assisted interpretation

**arXiv ID:** 2606.26857 | [PDF](https://arxiv.org/pdf/2606.26857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 351. Appearance-Preserving Refinement of Generated 3D Assets for Monochromatic Fabrication

**arXiv ID:** 2606.26850 | [PDF](https://arxiv.org/pdf/2606.26850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 352. SpikeTimer: Exploring Active Copyright Protection in Spiking Neural Networks via Temporal Backdoor Regularization

**arXiv ID:** 2606.26841 | [PDF](https://arxiv.org/pdf/2606.26841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 353. Ordinal Neural Collapse as a Representation Prior for Visual Navigation

**arXiv ID:** 2606.26839 | [PDF](https://arxiv.org/pdf/2606.26839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 354. Learning Adversarial Augmentation Policies for Robust Garlic Seedling Detection

**arXiv ID:** 2606.26828 | [PDF](https://arxiv.org/pdf/2606.26828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 355. MIRROR: Novelty-Constrained Memory-Guided MCTS Red-Teaming for Agentic RAG

**arXiv ID:** 2606.26793 | [PDF](https://arxiv.org/pdf/2606.26793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 356. ResilPhase: Plug-and-Play Phase Mapping and Noise-Resilient Macro-Trajectory Extrapolation for Diffusion Acceleration

**arXiv ID:** 2606.26769 | [PDF](https://arxiv.org/pdf/2606.26769v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 357. Anatomy-Guided Residual Motion Diffusion for Controllable 4D Cardiac MRI Synthesis

**arXiv ID:** 2606.26764 | [PDF](https://arxiv.org/pdf/2606.26764v1)

**作者:** Yiheng Cao `[一作]` (Suzhou Institute of Biomedical Engineering and Technology), Xin Gao `[通讯]` (Suzhou Institute of Biomedical Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出了一套可控的 4D 心脏 MRI 合成框架，能够生成与真实数据一致的 4D 动态序列及其对应的分割掩码。

**💡 创新点**

核心创新在于引入半监督 VAE 进行解剖结构与语义信息的联合学习，并通过级联的静态与残差运动潜在扩散模型实现解剖与运动的独立可控与时间一致性。

**🔧 技术方法**

技术实现包括半监督 VAE‑GAN、静态潜在扩散模型、残差运动预测器、条件潜在扩散模型（motion LDM）以及基于临床先验的分类器无关引导（CFG）和 DDIM 抽样。

**📊 数据集**

数据集方面使用 956 名患者的组合数据：100 名带 ED/ES 标注的 ACDC 数据、856 名无标注的 Kaggle DSB 数据，并在 ACDC、DSB 保留集以及外部 M&Ms/M&Ms2 设备数据上进行评估。

**📈 对比分析**

通过在 nnU-Net、UNETR 和 Swin-UNETR 等分割框架中使用合成数据进行增广，实验显示 Dice 分数提升 1.4%–4.7%，Hausdorff 距离降低 3–6 mm，FID 约 72，FVD 约 288，且对临床体积先验的 Pearson r 可达 0.94，证明合成数据显著提升跨设备泛化。

**⚠️ 局限性**

局限性包括：仅在心脏 MRI 上验证，残差运动模型对极端病理动态的捕捉有限；依赖临床先验与可用无标注数据；对其他 4D 医学影像模态与更多下游任务的适用性尚需进一步评估。

---

## 358. NaviCache: Test-Time Self-Calibration Caching for Video Generation

**arXiv ID:** 2606.26795 | [PDF](https://arxiv.org/pdf/2606.26795v1)

**作者:** Zheqi Lv `[一作]` (Zhejiang University), Fei Wu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了NaviCache，一种基于测试时自校准的缓存框架，用于加速视频扩散模型的推理。

**💡 创新点**

创新点在于将特征演化重新表述为惯性导航系统，使用双状态估计、初始对齐与不确定性感知的测量更新，实现了比零阶近似更紧的误差上界。

**🔧 技术方法**

采用线性高斯状态空间模型（类似卡尔曼滤波）进行双状态跟踪，并结合不确定性估计、测量修正以及理论上的Riccati方程分析。

**📊 数据集**

在VBench数据集上评估，使用HunyuanVideo、Wan 2.1和Open‑Sora三大视频扩散模型进行实验。

**📈 对比分析**

与TeaCache、MagCache、EasyCache、PAB等方法对比，NaviCache在保持相近或更高速度提升的同时，获得更优的PSNR/SSIM/LPIPS指标，例如mid模式下速度提升≈2.17×、PSNR 32.65。

**⚠️ 局限性**

局限性包括需要手工调节过程噪声Q、测量噪声R和阈值τ，对极端噪声或高动态场景的鲁棒性仍有限，且初始对齐阶段会增加少量推理开销。

---

## 359. Effective Resistance-Based Graph Sparsification and Community Detection

**arXiv ID:** 2606.26766 | [PDF](https://arxiv.org/pdf/2606.26766v1)

**作者:** Jayanta Pari `[一作]` (Indian Institute of Science), Soumyendu Raha `[通讯]` (Indian Institute of Science)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于有效电阻的社区检测算法ERSCD，利用有效电阻计算节点相似度，构造加权图后进行稀疏化（通过最小生成树和阈值去除非MST边），最后在稀疏图上使用Clauset-Newman-Moore模块化最大化进行社区划分。

**💡 创新点**

创新点在于：①首次将有效电阻与图稀疏化相结合，既保留社区结构又显著降低计算量；②通过MST+阈值去边策略实现可控稀疏化，仅需一个可调参数pe；③在稀疏化后仍保持高模块化，兼具准确性与效率。

**🔧 技术方法**

主要技术包括：拉普拉斯矩阵伪逆与有效电阻计算、Algebraic Multigrid求解、Gaussian相似度映射、最小生成树构建与阈值稀疏化、Clauset-Newman-Moore贪婪模块化优化、ARI/NMI评估指标。

**📊 数据集**

数据集涵盖合成网络：SBM与LFR（250节点，混合参数0.1–0.6），以及真实网络：Zachary Karate Club、American College Football、Political Books、Political Blogs、Cora（引用网络）。

**📈 对比分析**

与Louvain、Infomap、Ricci Flow等现有方法比较，ERSCD在大多数真实网络与合成网络上取得更高的ARI/NMI，模块化值与Infomap/Louvain相近，运行时间介于二者之间，显著低于Ricci Flow。

**⚠️ 局限性**

局限性：需手动调节pe参数；在合成网络（结构已天然清晰）上稀疏化效果不明显；未使用GPU加速，处理极大稠密网络的效率仍待提升。

---

## 360. On-Demand Service Zone Design for Energy-Constrained Spatial Queueing Systems

**arXiv ID:** 2606.26765 | [PDF](https://arxiv.org/pdf/2606.26765v1)

**作者:** Peng Lin `[一作]` (Tsinghua University), Kai Wang `[通讯]` (Tsinghua University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种能量受限的高维空间排队模型（energy‑constrained hypercube），并基于该模型构建电动服务车辆（ESV）站点选址与服务区划分的联合优化问题。为求解该大规模非线性混合整数规划，作者设计了一套“Branch‑Price‑and‑Evaluation”框架，将列生成与列评估结合起来，同时利用外部评估得到的列系数实现高效求解。

**💡 创新点**

创新点包括：①将电池状态动态嵌入经典高维空间排队模型，首次以半马尔可夫过程描述车辆的能量与位置联合演化；②提出Energy‑Hypercube Iteration（固定点迭代）求解稳态指标；③针对列系数无法闭式得到的集合划分问题，研发了可外部计算的列评估机制，并在列生成阶段使用p‑median和统一接受率上界作为诱导价；④通过Branch‑Price‑and‑Evaluation实现对非线性系统性能评估与整数规划求解的整合，首次将列评估融入分支定界树。

**🔧 技术方法**

技术方法包括：半马尔可夫分析、逆Erlang方法、固定点迭代、列生成与两级列评估、分支定界、p‑median和统一接受率上界诱导价格、集合划分（set‑partitioning）模型、线性化的连通性流约束。

**📊 数据集**

实验使用了人工合成实例（J=9，A,N多种组合）以及三个基于多伦多实际数据的真实案例：移动充电、无人机巡检和自动清洁。数据来源于加拿大2021年人口普查、实际道路网络（旅行时间）、车辆能量参数（如电池容量、充电/服务时间）以及业务收益率。

**📈 对比分析**

方法比较：枚举（仅用于小规模）、精确Branch‑Price‑and‑Evaluation（Exact）以及其启发式版本（Heur）。Exact在中等规模实例上能够得到最优解，速度明显快于枚举；Heur在大规模实例（J=36）仍保持较小的近似误差（≤3.6%），并大幅缩短计算时间。结果表明：①能量建模显著降低误报，提升规划可信度；②服务区划分在低负荷下损害盈利，重负荷下提升盈利；③电池容量与区划相互替代，区划往往是更有力的管理杠杆。

**⚠️ 局限性**

局限性：①假设服务时长与旅行时间为确定性且不随时间变化；②仅考虑单一能量耗尽阈值和离散能量水平；③未考虑车辆动态再定位、时变需求或电网约束；④模型假设每站仅一辆ESV，且站点与需求节点同位置；⑤对高维问题仍有求解瓶颈，启发式方法缺乏严格最优性保证。

---

## 361. The Capability Frontier: Benchmarks Miss 82% of Model Performance

**arXiv ID:** 2606.26836 | [PDF](https://arxiv.org/pdf/2606.26836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 362. FBK's Long-form SpeechLLMs for IWSLT 2026 Instruction Following

**arXiv ID:** 2606.26819 | [PDF](https://arxiv.org/pdf/2606.26819v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 363. Computational Analysis of Heart Rate Variability in Healthy Adults

**arXiv ID:** 2606.26816 | [PDF](https://arxiv.org/pdf/2606.26816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 364. Batch-Invariant Spectral Intelligence for Robust and Explainable Insect Authentication

**arXiv ID:** 2606.26757 | [PDF](https://arxiv.org/pdf/2606.26757v1)

**作者:** Majharulislam Babor `[一作]` (Leibniz Institute for Agricultural Engineering and Bioeconomy), Marina M. -C. Höhne `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一种名为Batch-Invariant Spectral Network（BISN）的端到端深度学习框架，用于在不同生产批次之间鲁棒地进行可食用昆虫的近红外光谱鉴定。

**💡 创新点**

将对抗式批量抑制策略置于特征提取之前，结合Savitzky–Golay初始化的可学习预处理模块和稀疏注意力编码器，实现无标注目标批次下的批量不变表示。

**🔧 技术方法**

使用了可学习的Savitzky–Golay滤波预处理、梯度反转层、稀疏注意力网络、熵正则化的批判分支，以及整合梯度解释（Integrated Gradients）和区域受限反事实优化。

**📊 数据集**

对三种工业相关昆虫（Acheta domesticus、Hermetia illucens、Tenebrio molitor）在三批次、三种处理（原始、焯水、等离子激活水）和两种超声条件下共收集了2700条700–2050 nm的近红外光谱。

**📈 对比分析**

采用严格的留一批次交叉验证（LOBO）与多种基线（LDA、GPC、PLSDA、diPLS、PDS-PLSDA、ShapDA、DANN、TabNet、TabPFN等）比较，BISN在所有批次上平均准确率0.93±0.04，显著优于最强基线（0.89±0.05），提升约4个百分点，统计显著性p<10⁻⁶。

**⚠️ 局限性**

实验中批次与测量时间不可分离，导致生物学与技术变异混杂；对染色体/色素区域的可靠性仍待验证；模型需扩展到更多物种与批次，并探索少样本/无监督迁移学习场景。

---

## 365. Reasoning Quality Emerges Early: Data Curation for Reasoning Models

**arXiv ID:** 2606.26797 | [PDF](https://arxiv.org/pdf/2606.26797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 366. Context-Aware Synthesis of Optimization Pipelines for Warehouse Optimization

**arXiv ID:** 2606.26852 | [PDF](https://arxiv.org/pdf/2606.26852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 367. An Information-Theoretic Metric for Semantic Value of Spatiotemporal Information

**arXiv ID:** 2606.26844 | [PDF](https://arxiv.org/pdf/2606.26844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 368. Identifying the Unknown: Prompt-Free Open Vocabulary Anomaly Recognition for Robot-Object Interaction

**arXiv ID:** 2606.26829 | [PDF](https://arxiv.org/pdf/2606.26829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 369. KARLA: Knowledge-base Augmented Retrieval for Language Models

**arXiv ID:** 2606.26807 | [PDF](https://arxiv.org/pdf/2606.26807v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 370. From Vajrayana Tara to Bengali Baul: A Computational Study of Lexical Transmission Across Buddhist, Shakta, and Vaishnava Traditions in Bengal

**arXiv ID:** 2606.26803 | [PDF](https://arxiv.org/pdf/2606.26803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 371. A Shared IPTC Topic Space for Cross-Source Topic Modelling

**arXiv ID:** 2606.26845 | [PDF](https://arxiv.org/pdf/2606.26845v1)

**作者:** Din Iskakov `[一作]` (University of Exeter), Rodrigo Wilkens `[通讯]` (University of Exeter)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于IPTC媒体主题词典的可复现共享主题空间框架，用于跨来源主题建模。

**💡 创新点**

创新点在于将外部验证的IPTC分类作为统一参照，通过引导的BERTopic与层次化映射及父级增强目标实现跨语料库可比的主题标签。

**🔧 技术方法**

使用了指导式BERTopic、Sentence‑Transformer词向量、UMAP降维、HDBSCAN聚类和MMR多样性策略等技术。

**📊 数据集**

开发集为2011年纽约时报语料，评估集包含多年的Twitter样本和新闻数据。

**📈 对比分析**

通过对映射覆盖率、最佳相似度和相邻目标差距的比较，指导式模型在严格阈值下优于零射击基线，且阈值调整不致使覆盖骤降。

**⚠️ 局限性**

局限性包括主题空间压缩导致对平台特定细节缺失、仅验证于单一新闻机构和有限年份、以及对IPTC层级细化的适用性待进一步研究。

---

## 372. Quantization in Federated Learning: Methods, Challenges and Future Directions

**arXiv ID:** 2606.26822 | [PDF](https://arxiv.org/pdf/2606.26822v1)

**作者:** Farwa Ikram `[一作]` (University of Calabria), Giancarlo Fortino `[通讯]` (University of Calabria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统评估联邦学习中的量化技术，提出面向FL的量化分类法并探讨其与FL关键行为的交互

**💡 创新点**

首次构建FL专属的量化分类法，系统分析量化对客户端漂移、部分参与、收敛稳定、隐私安全等FL核心行为的影响，提出跨方法洞察与设计指南

**🔧 技术方法**

基于PRISMA系统性文献综述、统一量化维度（训练阶段、量化对象、编码结构、精度策略），结合量化方法如PTQ、QAT、混合PTQ+QAT、向量/格子量化、混合精度、非均匀量化、可自适应精度等

**📊 数据集**

使用公开FL基准数据集如CIFAR-10、FEMNIST、Shakespeare等进行评估（但未统一实验）

**📈 对比分析**

对比方法包括不同量化方案的通信成本、准确率、收敛速度等指标，发现自适应与混合精度方案在准确率-通信效率曲线最优；量化能降低90%通信量但在低比特时会出现显著精度下降

**⚠️ 局限性**

主要限制：缺乏统一评测基准与实验复现、对异构、非IID、异步环境下的收敛理论不完善、硬件支持不足、隐私与安全耦合复杂、实测部署有限

---

## 373. Improving Vision-Language-Action Model Fine-Tuning with Structured Stage and Keyframe Supervision

**arXiv ID:** 2606.26801 | [PDF](https://arxiv.org/pdf/2606.26801v1)

**作者:** Yuan Xu `[一作]` (School of Artificial Intelligence University of Chinese Academy of Sciences), Liang Wang `[通讯]` (New Laboratory of Pattern Recognition State Key Laboratory of Multimodal Artificial Intelligence Systems Institute of Automation Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在 Vision‑Language‑Action 模型微调时加入基于抓取器状态自动生成的阶段分类和关键帧预测的辅助监督，提升机器人长期操作任务的成功率。

**💡 创新点**

创新点在于：①使用无人工标注的抓取器状态自动提取阶段标签和关键帧目标；②将这些监督以轻量级可插拔的查询标记和辅助头加入训练，不改模型架构和推理流程；③证明在多关键帧长程任务中能显著提升性能。

**🔧 技术方法**

使用 Vision‑Language‑Action（π_0.5）框架、流匹配（flow‑matching）动作专家、轻量级 MLP 查询标记头、二元交叉熵和 L1 关键帧损失。

**📊 数据集**

在 RoboTwin 2.0 机器人双臂模拟任务和 Franka Research 3 单臂真实机器人任务上进行训练和评估，使用 50 条演示示例。

**📈 对比分析**

与基线 π_0.5 以及 Diffusion Policy、ACT、RDT 等方法比较，StaKe 在 10 个模拟任务中平均成功率提升至 59.8%（高于 52.4% 基线），在真实机器人任务中平均提升至 62.5%（高于 40.0% 基线）。

**⚠️ 局限性**

仅利用抓取器开闭事件，限制了对非抓取或灵巧操作的适用性；关键帧定义对特殊任务可能不够丰富。

---

## 374. ReasonCLIP-58M: Visually Grounded Commonsense Reasoning Supervision for CLIP

**arXiv ID:** 2606.26794 | [PDF](https://arxiv.org/pdf/2606.26794v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 375. An Experimental Assessment of the Spatial and Frequency Selectivity of Reconfigurable Intelligent Surfaces

**arXiv ID:** 2606.26808 | [PDF](https://arxiv.org/pdf/2606.26808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 376. An Orthogonal Approximate Message Passing Framework for Multiuser Communications

**arXiv ID:** 2606.26777 | [PDF](https://arxiv.org/pdf/2606.26777v1)

**作者:** Burak Çakmak `[一作]` (Technical University of Berlin), Lei Liu `[通讯]` (Zhejiang University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种多用户随机预编码线性高斯通信系统的信号恢复框架 MU-OAMP，并给出了其高维理论分析与Replica对称性（RS）预测。

**💡 创新点**

创新点在于：① 在非可分输入信号与任意通道矩阵的通用场景下实现了高维分析与RS预测的匹配；② 在RS假设下证明了 MU-OAMP 能实现 Bayes 最优；③ 提出了针对多用户系统的 Disorder 平均与 Saddle‑point 处理方法。

**🔧 技术方法**

使用了 Replica 对称性 ansatz、状态演化（SE）分析、Haar 单位矩阵随机化与高维概率集中不等式、随机矩阵的 saddle‑point 方法。

**📊 数据集**

采用仿真数据：OFDM 信号、4G/5G 多用户信道模型以及随机生成的 Haar 单位矩阵或随机正交矩阵。

**📈 对比分析**

通过与理论互信息/MMSE 预测以及传统最大似然/MMSE 等算法对比，实验表明 MU-OAMP 的误差与理论极限相符，性能优于现有方案。

**⚠️ 局限性**

局限性包括：需要完美的通道知识；算法基于 RS 对称性假设，实际中可能失效；随机预编码实现成本和复杂度较高。

---

## 377. AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems

**arXiv ID:** 2606.26859 | [PDF](https://arxiv.org/pdf/2606.26859v1)

**作者:** Changxin Lao `[一作]`, Zhenkai Cui `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 AgentX——一个面向工业推荐系统的自动化开发闭环框架，能够在真实业务环境中实现从想法生成、代码实现到在线 A/B 评估的全流程闭环，并通过持续收集执行轨迹实现自我迭代。

**💡 创新点**

创新点在于：①将 LLM 作为智能体实现真正的实验迭代，突破传统自动化只能做流水线的限制；②设计了完整的在线 A/B 反馈闭环，使模型改进直接映射到业务价值；③提出了语义梯度提示优化（SGPO）实现子智能体的安全演化，从而实现系统自适应、持续提升；④通过多智能体协同（Brainstorm、Developing、Evaluation）和知识库实现经验累积和复用。

**🔧 技术方法**

主要技术包括：大语言模型（LLM）驱动的智能体、提示工程与 Semantic‑Gradient Prompt Optimization、基于知识库的事实检索与验证、可视化监控平台、在线 A/B 评估与 guardrail 设计、持续集成/持续部署（CI/CD）以及多任务并行流水线。

**📊 数据集**

使用了 Kuaishou 内部工业推荐数据，包括主 Feed 推荐、生活服务商业化场景，样本量达数十亿级；实验还依托内部实验平台和线上流量，覆盖多业务线与多指标。

**📈 对比分析**

通过与单个算法工程师手工迭代对比，在三周内 3 名 AgentX 工作者完成 374 条想法、10 条上线实验，平均每位工作者每周 3.3 个可上线实验，迭代吞吐率提升约 8 倍；在线实验收益在主 Feed 提升 0.561% 用户消费时长，生活服务实现 1 亿多元年化收入；相比传统手工流程，累计业务价值提升约 3.7 倍。

**⚠️ 局限性**

局限性包括：①仍然受到平台/运维约束，50% 以上拒绝来自资源锁定和基础设施限制；②模型改进高度依赖内部数据与业务指标，迁移到其他业务或平台需要重构知识库与 guardrail；③系统迭代速度受大规模并行实验的计算资源限制；④尽管 SGPO 能演化子智能体，但在极端失败或误诊时可能引入不安全修改，需要人工复核；⑤对长周期因果分析支持有限，主要通过单轮 A/B 评估实现，可能无法捕捉更深层次的业务因果。

---

## 378. Humanoid-DART: Humanoid Loco-Manipulation using Diffusion-guided Augmentation through Relabeling and Tracking

**arXiv ID:** 2606.26855 | [PDF](https://arxiv.org/pdf/2606.26855v1)

**作者:** Pranav Debbad `[一作]` (Munich Institute of Robotics and Machine Intelligence, Technical University of Munich), Majid Khadiv `[通讯]` (Munich Institute of Robotics and Machine Intelligence, Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种自监督的迭代学习框架，通过稀疏演示启动，结合扩散模型生成运动轨迹与基于强化学习的跟踪控制器，逐步在连续任务空间中扩展可行的目标运动。

**💡 创新点**

创新点在于：①将目标条件扩散变换器与物理基RL跟踪器分离并在演化循环中交替训练；②引入多分支扩散网络和结构化部分遮蔽提升生成质量；③使用基于任务空间距离的课程学习和目标重标记机制，显著提高覆盖率并减少对专家监督的需求。

**🔧 技术方法**

技术主要包括：目标条件扩散变换器（DiT）生成运动轨迹；DeepMimic风格的全身跟踪政策（PPO+GAE）；物理基评估器用于筛选可行轨迹；课程学习与目标重标记算法；自回归生成与遮蔽训练。

**📊 数据集**

使用了DynaRetarget人类动作数据集（已重目标化为Unitree G1机器人），并在此基础上仅采样2–4条稀疏示范（约20秒运动）。

**📈 对比分析**

与两类基线（Parameterized Motion 与 Hierarchical Diffusion+RL）进行对比，在四个下肢运动任务（push、kick、hand‑off、pick‑and‑place）中，Humanoid‑DART实现了高达96.4% 的任务空间覆盖率，平均 fitness 远高于基线；在训练速度与泛化范围方面亦显著优于对比方法。

**⚠️ 局限性**

局限性包括：对初始示范的偏差高度敏感，需手工调节的适配函数；仅在单一物体几何、平坦地面上验证；对物理随机性和环境多样性支持不足；需要进一步自动化的适配与评估标准。

---

## 379. OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning

**arXiv ID:** 2606.26790 | [PDF](https://arxiv.org/pdf/2606.26790v1)

**作者:** Shuo Yang `[一作]` (Tsinghua University), Jianhua Tao `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了OPID框架，将完成的 on‑policy 轨迹转化为层次化的 hindsight 技能，并通过关键优先路由在训练中实现自我蒸馏，提升语言代理在多步任务中的性能。

**💡 创新点**

创新点在于：① 只用当前策略自身轨迹生成分层技能（episode‑level 与 step‑level）；② 采用关键优先路由（critical‑first routing）挑选最合适的技能；③ 将技能注入历史生成 token‑level 自我蒸馏优势，保持 RL 为主优化目标，避免外部技能库或检索。

**🔧 技术方法**

核心技术包括 on‑policy RL（GRPO 样式）、自我蒸馏、LLM‑based 分析器抽取技能、层次化技能表示、关键优先路由、与 episode‑level advantage 结合的 token‑level 优势优化。

**📊 数据集**

在 ALFWorld、WebShop、Search‑based QA（含 Natural Questions、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle）等多项面向长序列交互的基准上进行实验。

**📈 对比分析**

与基线（prompting、GRPO、Skill‑GRPO、OPSD、SDAR 等）对比，OPID 在 ALFWorld、WebShop 及 QA 上均提高 7–12% 的成功率，样本效率提升约 60% 训练数据即可逼近全量效果，且在跨域未见数据上亦表现更稳健。

**⚠️ 局限性**

局限性包括：需额外的分析器进行技能抽取，可能受限于任务的可解释性；技能抽取质量影响训练稳定性；在对齐分层技能与状态分布差异时仍需手工调参；推理阶段不依赖技能，但若任务需实时外部技能，OPID 仍无法直接利用。

---

## 380. Calibrated Harmonic Overlaid Implicit Neural Representations for Multi-Dimensional Data

**arXiv ID:** 2606.26763 | [PDF](https://arxiv.org/pdf/2606.26763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 381. RobOralScan: Learning Active Intraoral Scanning for Robotic Dental Reconstruction

**arXiv ID:** 2606.26955 | [PDF](https://arxiv.org/pdf/2606.26955v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 382. MergeLLL: A Hierarchical Divide-and-Conquer Framework for LLL-Based Lattice Reduction

**arXiv ID:** 2606.26784 | [PDF](https://arxiv.org/pdf/2606.26784v1)

**作者:** Niharika Gauraha `[一作]` `[通讯]` (KTH Royal Institute of Technology), Niharika Gauraha (KTH Royal Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种基于层次划分与合并的 MergeLLL 框架，用以改进 LLL 基础的格基约简

**💡 创新点**

创新点在于将格基递归划分为子基，局部使用 PotLLL 约简后再通过结构化深度插入的合并步骤，既保持了格结构，又显著减少了全局交换操作并提升并行度

**🔧 技术方法**

使用的技术包括：格基的 Gram–Schmidt 正交化、PotLLL 约简、递归划分与层次合并、并行化实现及对数深度并行算法

**📊 数据集**

实验数据集主要为两类：基于 Subset‑Sum 的无结构格基（维度 21–401，权重位长 20/40）以及基于 NTRU 结构的格基（维度 20–100）

**📈 对比分析**

与传统 LLL 在相同 δ=3/4 设置下对比，MergeLLL 在所有实验中实现了显著的运行时间加速（尤其在高维子集求和与 NTRU 结构中），且 Hermite 因子和正交误差基本保持或略优于 LLL；交换次数也显著下降

**⚠️ 局限性**

局限性包括：当前实现仅采用固定浮点精度，存在数值稳定性风险；未在 SVP 挑战等标准基准上测试；缺乏对更大维度或其他块尺寸的 BKZ 结合效果的评估

---

## 383. Continuous Behavioral Synthesis for Adaptive Health Dashboards: An LLM-Mediated Architecture Integrating Explicit Preference, Spatial Reorganization, and Attention Allocation Signals

**arXiv ID:** 2606.26937 | [PDF](https://arxiv.org/pdf/2606.26937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 384. Escaping Iterative Parameter-Space Noise: Differentially Private Learning with a Hypernetwork

**arXiv ID:** 2606.26772 | [PDF](https://arxiv.org/pdf/2606.26772v1)

**作者:** Naoki Nishikawa `[一作]` (University of Tokyo), Satoshi Hasegawa `[通讯]` (LY Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种使用公共超网络DP-DeepSets的差分私有学习框架，该框架通过一次性在低维嵌入空间注入噪声，从私有数据直接生成模型参数，避免了梯度方法中高维参数多次噪声注入的问题。

**💡 创新点**

创新点在于：①用超网络一次性映射数据集到参数，降低噪声维度；②仅在低维数据嵌入上注入一次DP噪声；③通过公共数据训练超网络实现“学习如何学习”，实现更高效的参数生成。

**🔧 技术方法**

技术手段包括：DeepSets + Transformer的DP-DeepSets架构；Gaussian机制对平均嵌入加噪；CLIP/Inception特征嵌入；LoRA微调；公共数据训练超网络；理论误差分析与实验对比。

**📊 数据集**

使用的数据集：ImageNet64预训练模型；CIFAR-10（30k公开、30k私有、128样本私有子集）以及在CLIP 512维空间的嵌入。

**📈 对比分析**

通过与DP-SGD以及两种基于公开数据的梯度方法（先公有再私有、PDA‑DPMD）进行比较，使用FID指标评估生成质量。结果显示DP-DeepSets在相同ε下FID显著低于DP‑SGD，且在ε降低到1左右时性能仍保持稳定，而DP‑SGD性能急剧下降。

**⚠️ 局限性**

局限性包括：实验仅在图像生成（LoRA微调）场景验证；未在其他模态（如语言）或任务（如分类）上测试；仅在小样本（128样本）情况下验证，尚未探究在大规模数据下的表现。

---

## 385. Complementing Emerson-Lei Elevator Automata (Technical Report)

**arXiv ID:** 2606.26768 | [PDF](https://arxiv.org/pdf/2606.26768v1)

**作者:** Ondrej Alexaj `[一作]` (Brno University of Technology), Nicolas Mazzocchi `[通讯]` (Slovak University of Technology in Bratislava)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究了Emerson-Lei电梯自动机（ELEA）的补集构造，提出了一种新的补集算法，并实现了其在工具中的原型。

**💡 创新点**

创新点在于将Büchi电梯自动机扩展到更通用的Emerson-Lei接受条件，给出复杂度更低（(5+k)^n 级）且可行的补集构造，并引入共享断点、OR-FIN 等多种实用优化。

**🔧 技术方法**

采用Miyano‑Hayashi、NCSB、树形宏状态、模块化分层构造以及结构化优化技术，理论上给出上界分析并实现了实验原型。

**📊 数据集**

实验数据集包括 LTL‑Lit（从 LTL 基准生成的电梯自动机）、GRA（随机生成的单对 Generalized Rabin 自动机）和 LTL‑Rand（随机 LTL 公式转换得到的 ELEA）。

**📈 对比分析**

将实现与 Spot 及其高约简变体 SpotHigh 进行比较，评估指标为补集大小、运行时间和可完成实例数。实验显示新实现能够解决更多实例，补集规模显著下降，整体性能优于 Spot。

**⚠️ 局限性**

限制主要体现在状态空间仍会爆炸，某些优化在特定实例下效果不明显；对非电梯结构自动机的转换仍需额外成本；实验受限于时间阈值，部分大规模实例未能完成。

---

## 386. TraMP-LLaMA: Generative Interpretability with Decoupled Instruction Tuning for Facial Expression Quality Assessment

**arXiv ID:** 2606.26942 | [PDF](https://arxiv.org/pdf/2606.26942v1)

**作者:** Shuchao Duan `[一作]` (University of Bristol), Majid Mirmehdi `[通讯]` (University of Bristol)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TraMP-LLaMA模型，能够同时预测面部表情质量评分并生成结构化的运动证据报告；

**💡 创新点**

创新点在于：1）通过联合预测和解码实现评分与报告的共享时空证据；2）采用stop‑gradient分离评分与语言生成的梯度流，避免任务干扰；3）将面部关键点轨迹与RGB信息融合，并将轨迹信息与融合表示作为额外条件注入LLM；4）在PFED5基础上扩展出PFED5+，提供专家标注的结构化运动描述；

**🔧 技术方法**

技术手段包括：VideoLLaMA3 VLM backbone（SigLIP‑NaViT + Qwen‑2.5 LLM），SkateFormer轨迹编码器，跨模态融合模块，LoRA指令调优，stop‑gradient分离训练，结构化模板式报告生成；

**📊 数据集**

使用PFED5+数据集（包含MDS‑UPDRS评分和结构化文本描述），对比基准包括Former‑DFER、S2D、USDL、CoFInAl、QAFE‑Net、TraMP‑Former、VideoLLaMA3以及通用VLM如Chat‑UniVi、LLaVA‑Video、LongVU；

**📈 对比分析**

在PFED5+上，TraMP‑LLaMA在评分方面平均Spearman相关性达57.35%，超过所有对比方法至少4.39%；在报告生成方面BERTScore、ROUGE‑L、BLEU‑4和CIDEr均优于VideoLLaMA3和其他VLM，显示更好的文本一致性和信息覆盖；

**⚠️ 局限性**

局限性包括：1）生成式LLM偶尔会产生幻觉，特别是在运动细微或遮挡严重时；2）PFED5+数据规模有限，严重度级别分布不平衡，且单一参考报告难以覆盖多样性，影响监督质量；

---

## 387. Diagnosing Task Insensitivity in Language Agents

**arXiv ID:** 2606.26918 | [PDF](https://arxiv.org/pdf/2606.26918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 388. Learning to Recover Task Experts from a Multi-Task Merged Model

**arXiv ID:** 2606.26902 | [PDF](https://arxiv.org/pdf/2606.26902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 389. SamaVaani: Auditing and Debiasing Multilingual Clinical ASR for Indian Languages

**arXiv ID:** 2606.26901 | [PDF](https://arxiv.org/pdf/2606.26901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 390. Are LLMs Ready for Anti-Pattern Detection in Microservice Architectures?

**arXiv ID:** 2606.26927 | [PDF](https://arxiv.org/pdf/2606.26927v1)

**作者:** Marco De Luca `[一作]` (University of Naples Federico II), Anna Rita Fasolino `[通讯]` (University of Naples Federico II)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究将大型语言模型（LLM）与传统静态分析工具MARS对微服务架构中的16种反模式进行检测，并比较其性能。

**💡 创新点**

创新点在于首次将prompt‑based LLM推理应用于微服务反模式检测，并揭示LLM在需要语义理解的反模式上可补充传统工具的优势。

**🔧 技术方法**

使用的技术包括：三种通用LLM（ChatGPT、Gemini、Qwen）的prompt‑based 推理、Repomix仓库扁平化、以及对MARS的独立重新执行。

**📊 数据集**

采用了由MARS研究提供的13个开源微服务仓库数据集，并根据其标注的16种反模式进行评测。

**📈 对比分析**

通过对比精确率（Precision）和召回率（Recall）进行评估；结果显示LLM在部分反模式（如NAV、HE、NS）表现与MARS相当或更优，而在需要结构化依赖证明的反模式（如CD、SL、TO）表现明显落后，整体互补。

**⚠️ 局限性**

局限性包括：依赖于仓库扁平化导致缺乏跨服务结构信息、LLM在低频反模式下易出现高误报或漏报、未考虑运行时数据与动态分析、仅评估三种LLM且未做统计显著性检验。

---

## 391. EconSimulacra: A Digital Twin Platform of Socio-Economic Systems Powered by LLM Agents

**arXiv ID:** 2606.26883 | [PDF](https://arxiv.org/pdf/2606.26883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 392. SpatialFlow-GRPO: Where Spatial Credit Drives Image Editing

**arXiv ID:** 2606.26872 | [PDF](https://arxiv.org/pdf/2606.26872v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 393. Tractography-Driven Synthetic Data Generation for Fiber Bundle Segmentation in Tracer Histology

**arXiv ID:** 2606.26898 | [PDF](https://arxiv.org/pdf/2606.26898v1)

**作者:** Kyriaki-Margarita Bintsi `[一作]` (Massachusetts General Hospital and Harvard Medical School), Anastasia Yendiki `[通讯]` (Massachusetts General Hospital and Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

使用 dMRI 轨迹作为生成先验，构建合成 2D 病理图像与掩码，用于训练 2D U‑Net 自动分割黑猩猩脑内的纤维束，显著降低人工标注需求。

**💡 创新点**

创新点包括：① 将外周 dMRI 轨迹映射到病理图像空间以产生前景纹理；② 通过域随机化（光照、噪声、背景）提高合成图像的多样性；③ 采用真实+合成混合训练策略，实现在仅标注一个脑的前提下达到与三脑标注的 SOTA 近似的性能。

**🔧 技术方法**

技术方法包括：多束 dMRI 轨迹生成、基于块面图像的背景采样与 CLAHE/暗场变换、曲线渲染与噪声注入、二值掩码生成与后处理、2D U‑Net 训练、混合批次与多尺度融合。

**📊 数据集**

数据集：四只黑猩猩（M1–M4）高分辨率病理切片，M1 用于标注（约 30 张），M2–M4 作为测试集；使用来自 M0 的全脑 dMRI 与块面图像用于合成。

**📈 对比分析**

比较方法：与 SAM2、MedSAM2、Sundaresan 等自监督方法、三脑标注的 SOTA 以及仅真实标注的基线对比；在 M4 试验中，混合训练模型在稀疏束检测的 TPR 与整体平均 TP 值与 SOTA 相当或略优，但 FDR 较高；在 M2、M3 上表现出更好的跨脑泛化，尤其在中等/稀疏束上显著提升。

**⚠️ 局限性**

局限性：1) 仍存在较高的假阳性率，主要受背景、注射模式等域偏差影响；2) 仅利用轨迹前景无法捕捉真实病理的强度分布与染色缺陷；3) 低样本量下性能非单调，表明标注策略对结果影响较大。

---

## 394. Heterogeneous Neural Predictivity from Language Models During Naturalistic Comprehension

**arXiv ID:** 2606.26880 | [PDF](https://arxiv.org/pdf/2606.26880v1)

**作者:** Xiao Jia `[一作]` `[通讯]` (Chinese University of Hong Kong), Xiao Jia (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

分析了三组自然语言理解数据集，使用八个冻结语言模型的上下文嵌入等特征，评估其对脑电/MEG/皮层电位的预测能力。

**💡 创新点**

首次系统分层划分信息携带预测、模型特异性优势、响应谱组织和计算特异性诊断等证据层级，并通过匹配控制和可靠性上限检验模型特定解读的可行性。

**🔧 技术方法**

岭回归编码、阻塞交叉验证、匹配控制（随机、时间移位、层标签置换等）、特征消融、可靠性上限归一化、脑-脑相似度评估、Bootstrap和FDR校正。

**📊 数据集**

Brain Treebank（10人内电极）、MEG-MASC（11人MEG）、Podcast ECoG（9人内电极）三组自然语言刺激的神经数据。

**📈 对比分析**

在每个数据集对比原始语言模型特征与多种匹配控制，计算Pearson‑r或R²的增益；最终只有67/432评估行满足模型特异性标准，且模型与脑响应谱的相似度低于最佳控制，说明预测性存在但模型特定优势有限。

**⚠️ 局限性**

受限于公开派生数据的覆盖不均、受样本量小和缺乏完整共索引的控制/特征消融/相似度评估，未能得出统一的模型特异性或计算对应结论；且使用的模型固定，未包含更大规模或指令调优模型。

---

## 395. TAVR-VLM: Risk-Conditioned Causal Grounding for Hallucination-Resistant Report Generation

**arXiv ID:** 2606.26874 | [PDF](https://arxiv.org/pdf/2606.26874v1)

**作者:** Zhixiang Lu `[一作]` (Xi'an Jiaotong-Liverpool University), Jinfeng Wang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出 TAVR‑VLM 框架，实现基于风险条件的因果式多模态报告生成；

**💡 创新点**

引入 Risk‑Conditioned Causal Grounding Attention (R‑CGA)，构建“风险→区域→词”结构化路径，并使用风险瓶颈与支持投影一致性损失，有效抑制诊断幻觉；

**🔧 技术方法**

使用多模态 Transformer、跨模态注意力、风险瓶颈投影、支持投影一致性、Stop‑Grad 等技术；

**📊 数据集**

基于 M³TAVR 数据集（1482 病例，包含 3D CT、2D 超声、临床表格）；

**📈 对比分析**

与风险分类器、报告生成器、开源与闭源 VLM 对比，在风险预测 AUROC 0.896、报告生成 CIDEr 0.936、幻觉率 8.1%、mIoU 0.624 上取得最优性能；

**⚠️ 局限性**

仅在 TAVR 领域验证，缺乏外部验证和对极端病例的泛化评估，且对高质量标注依赖较大。

---

## 396. Non-Uniform and Weighted Crossing Gates in Two-Dimensional Sandpiles

**arXiv ID:** 2606.26943 | [PDF](https://arxiv.org/pdf/2606.26943v1)

**作者:** Pablo Concha-Vega `[一作]` (Aix Marseille Université), Kévin Perrot `[通讯]` (Aix Marseille Université)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究二维沙堆模型中交叉门的存在性，提出主格子（primal grid）与加权格子（weighted uniform grid）之间的等价性并构造复杂交叉门。

**💡 创新点**

首次给出主格子与加权格子交叉门存在性的互为充要条件，并发现存在多次交叉的非简单交叉门，揭示等价性在更一般情形下失效。

**🔧 技术方法**

通过拓扑论、顶点转移图分析以及构造性证明（路径分解、权重分配）等理论工具。

**📊 数据集**

未使用实验数据，全部为理论证明与图形构造。

**📈 对比分析**

由于是理论结果，无实验对比；作者讨论了该等价性对 -PCompleteness 证明的潜在意义。

**⚠️ 局限性**

结果仅适用于满足单次顶点翻滚特性的格子，且在更一般的非主格子或非加权格子设置下等价性不再成立，缺乏完整的算法实现与复杂度评估。

---

## 397. Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation

**arXiv ID:** 2606.26907 | [PDF](https://arxiv.org/pdf/2606.26907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 398. Where Do CoT Training Gains Land in LLM based Agents?

**arXiv ID:** 2606.26935 | [PDF](https://arxiv.org/pdf/2606.26935v1)

**作者:** Jingyu Liu `[一作]` (Renmin University of China), Yong Liu `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过比较从提示直接预测动作（prompt action）与先生成推理链再预测动作（CoT action）的方式，系统评估了 Chain-of-Thought 训练在长上下文语言模型代理中的作用，发现 CoT 训练不仅提升了推理链后的动作准确率，还显著提高了仅靠提示直接预测动作的能力；并基于此诊断，提出了在部分训练样本上屏蔽动作标注的“Reduced Action Supervision”干预方法，以削弱提示侧捷径，提升 OOD 泛化。

**💡 创新点**

创新点在于：①构建了 Prompt‑vs‑CoT 的诊断框架，能够量化 CoT 训练对直接提示预测和推理修正的双重影响；②通过注意力与梯度分析揭示提示在动作预测中的结构性优势；③提出并验证了通过部分屏蔽动作监督来降低提示捷径、增强 CoT 修正的训练干预。

**🔧 技术方法**

主要技术包括：链式思维（CoT）生成、监督式微调（SFT）、强化学习（RL）策略学习、注意力权重与梯度分布分析、以及在训练中对动作标注进行随机屏蔽的损失调整。

**📊 数据集**

实验数据集包括三种长上下文代理环境：ALFWorld（家庭任务）、ScienceWorld（科学实验推理）以及 BFCL（功能调用与工具使用），在这些环境下对不同模型尺寸和训练方法进行评估。

**📈 对比分析**

对比结果显示：①在训练过程中，Prompt 与 CoT 的准确率同步提升，二者差距保持平稳；②在 OOD 任务上，CoT 与 Prompt 的优势差距不显著扩展；③使用 Reduced Action Supervision 并结合 SFT/DPO/FRODO 等方法，可提升大多数环境的 OOD 表现，并在 ALFWorld 与 ScienceWorld 上进一步扩大 Prompt‑vs‑CoT 的差距，表明更好地保留了 CoT 修正空间。

**⚠️ 局限性**

主要局限包括：Prompt action 仅作为可观测的行为代理，无法完整反映内部推理过程；实验中对动作质量的判定部分依赖 GPT‑5.4 作为评判者，可能带来评估偏差；且干预的效果受提示长度与结构差异的影响，在结构更规范或动作空间更受限的 BFCL 环境中效果不如预期。

---

## 399. Jailbreaking for the Average Jane: Choosing Optimal Jailbreaks via Bandit Algorithms for Automatically Enhanced Queries

**arXiv ID:** 2606.26936 | [PDF](https://arxiv.org/pdf/2606.26936v1)

**作者:** Prarabdh Shukla `[一作]` (IIT Bombay), Arjun Bhagoji `[通讯]` (IIT Bombay)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于多臂赌博机（Bandit）框架的高效攻击策略，能够在有限查询预算内学习并选择最优的 jailbreak，以诱导 LLM 产生有害响应，并构建了 11,279 条恶意查询的安全基准（包括简单与复杂两类查询）。

**💡 创新点**

创新点包括：1）把 jailbreak 选择视为在线学习问题，利用部分信息赌博机算法实现子线性遗憾；2）构造“复杂查询”概念并自动生成/增强，使攻击成功率提升至 97% 以上；3）系统性评估 70 种 jailbreak 在 15 个主流 LLM 上的效果，展示了攻击策略对安全模型的显著突破。

**🔧 技术方法**

技术主要包括：在线多臂赌博机算法（如 Thompson Sampling、EXP3）、上下文无关/有关版本、基于 LLM 的复杂度判别器、自动化查询增强和生成系统、以及基于判别式 LLM 的有害性评估。

**📊 数据集**

使用的主要数据集是自行构建的安全基准：从 AIRBench、WMDP、JailbreakV-28K、HarmBench、MedSafetyBench、JailbreakBench、HarmfulQA 等 7 个基准中抽取并增强得到的 11,279 条恶意查询；此外还使用 15 个公开 LLM（从 270M 到 120B）进行评测。

**📈 对比分析**

与传统的暴力遍历（nT 查询）或随机/均匀采样方法相比，Bandit 方案在 Transfer 和 Continual 两种攻击场景下均能以更少查询实现更高的攻击成功率；实验显示平均 ASR 可提升至 40% 以上，复杂查询对成功率提升 9%–26%，多轮采样（k=5）甚至可将 ASR 提升至 97%。

**⚠️ 局限性**

局限性包括：仅对单轮交互评估，未考虑多轮会话；评测主要在英语数据，低资源语言和翻译攻击未覆盖；对专有 API 模型的评估有限；未对不同 jailbreak 的成本建模；上下文相关算法表现不如无上下文，原因尚待探究。

---

## 400. Scaling Multi-Reference Image Generation with Dynamic Reward Optimization

**arXiv ID:** 2606.26947 | [PDF](https://arxiv.org/pdf/2606.26947v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 401. Asymptotically Optimal Learning for Parametric Prophet Inequalities

**arXiv ID:** 2606.26893 | [PDF](https://arxiv.org/pdf/2606.26893v1)

**作者:** Jung-hun Kim `[一作]` (FairPlay Team, CREST, ENSAE, Institut Polytechnique De Paris), Vianney Perchet `[通讯]` (FairPlay Team, CREST, ENSAE, Institut Polytechnique De Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在指数型参数分布族下的神童不等式，针对未知参数的在线学习并实现全信息下的最优竞争比。

**💡 创新点**

创新点在于提出了基于置信上界的动态规划学习策略，利用分布的显式参数结构，在没有离线样本的情况下达到分布特定的最优渐近竞争比，尤其在重尾 Pareto 情形下实现了全信息极限。

**🔧 技术方法**

技术上结合了指数映射的最大似然估计、置信区间构造以及针对参数化奖励的动态规划阈值递推，并给出了尾部指数与极值理论的解析。

**📊 数据集**

实验使用了模拟的指数、Pareto 和有限支持幂族分布实例，评估算法性能。

**📈 对比分析**

与基线规则（秘书式规则、相对排名规则等）比较，算法在所有三个分布族上均逼近理论最优竞争比，尤其在 Pareto 情况下明显优于排名规则。

**⚠️ 局限性**

局限在于仅针对单参数指数型族，无法直接推广到更复杂的多参数或上下文模型，且需先行探索阶段的样本数满足一定增长条件。

---

## 402. Accelerated sampling using SamAdams variable timesteps and position-adaptive Langevin dynamics

**arXiv ID:** 2606.26881 | [PDF](https://arxiv.org/pdf/2606.26881v1)

**作者:** Benedict Leimkuhler `[一作]` (University of Edinburgh), Peter A. Whalley `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种结合SamAdams自适应步长与位置自适应Langevin（PAL）摩擦张量的分裂积分器（AZBOBZA），用于高效采样保留可克隆的吉布斯分布。

**💡 创新点**

创新点在于：①将自适应步长机制与仅在力方向上加大阻尼的秩一+标量摩擦相结合；②设计了只需一次梯度评估、O(d)复杂度的可逆时间平衡分裂方法；③通过Sundman时间重加权恢复原始分布。

**🔧 技术方法**

使用的技术包括：SamAdams自适应步长算法、位置自适应摩擦张量、O(d)闭式OU更新、可逆分裂（AZBOBZA）以及时间重加权回归。

**📊 数据集**

实验数据集涵盖三种二维基准势能（Rosenbrock、薄通道、Mueller–Brown）以及一个四百维贝叶斯线性回归（horseshoe先验）作为更高维真实场景。

**📈 对比分析**

与固定步长的BAOAB/OBABO比较，SA‑PAL在所有基准上显著降低积分自相关时间（1.5–3 倍提升，甚至十倍以上），并保持与目标分布一致的偏差；在贝叶斯回归中对聚合二阶矩的混合时间下降约 2–3 倍。

**⚠️ 局限性**

局限性包括：自适应步长导致的偏差需通过重加权校正；参数调优（α、β、θ、h_sa）仍缺乏统一理论指导；对极端高维、稀疏或噪声梯度场的鲁棒性尚待进一步验证。

---

## 403. GAVEL: Grounded Caption Error Verification and Localization

**arXiv ID:** 2606.26923 | [PDF](https://arxiv.org/pdf/2606.26923v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 404. PhysRAG: Enhancing Physics-Awareness in Video Generation via Retrieval-Augmented Generation

**arXiv ID:** 2606.26916 | [PDF](https://arxiv.org/pdf/2606.26916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 405. Bridging Vision and Language Concepts through Optimal Transport Semantic Flow

**arXiv ID:** 2606.26891 | [PDF](https://arxiv.org/pdf/2606.26891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 406. Fortress and Gatekeeper: Theorizing Transitive Trust in Third-Party Cybersecurity Risk Governance

**arXiv ID:** 2606.26866 | [PDF](https://arxiv.org/pdf/2606.26866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 407. Cascaded Multi-Granularity Pruning for On-Device LLM Inference in Industrial IoT

**arXiv ID:** 2606.26861 | [PDF](https://arxiv.org/pdf/2606.26861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 408. UAV-MapFusion: RTK-Aligned Uncertainty-Aware Coarse-to-Fine Multi-Session UAV Mapping

**arXiv ID:** 2606.26928 | [PDF](https://arxiv.org/pdf/2606.26928v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 409. Information-Aware KV Cache Compression for Long Reasoning

**arXiv ID:** 2606.26875 | [PDF](https://arxiv.org/pdf/2606.26875v1)

**作者:** Jushi Kai `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于信息熵的KV缓存压缩方法 InfoKV，用于长文本推理。

**💡 创新点**

提出前瞻性影响度量 Forward Influence 并将预测熵与层级表示演变结合，弥补仅靠注意力的局限。

**🔧 技术方法**

信息理论熵、层间余弦距离、注意力权重混合、top‑k 限制熵等技术。

**📊 数据集**

LongReason、IFEval、AIME 2024、LiveCodeBench 等数据集，使用 Llama‑3.1/3.2、DeepSeek‑R1 等模型。

**📈 对比分析**

与 SnapKV、PyramidKV、Expected Attention 等注意力压缩方法在长预填/解码场景下对比，InfoKV 在多模型、多长度上取得更高准确率。

**⚠️ 局限性**

前瞻影响仍是间接指标，层级自适应预算不稳定，且依赖模型特定的熵分布。

---

## 410. A Pipeline for Generating Longitudinal Synthetic Clinical Notes Using Large Language Models

**arXiv ID:** 2606.26879 | [PDF](https://arxiv.org/pdf/2606.26879v1)

**作者:** William Poulett `[一作]` `[通讯]` (NHS England), William Poulett (NHS England)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建并发布了一个多层级验证的合成临床笔记生成管道和数据集，覆盖70名患者、20-50份笔记。

**💡 创新点**

创新点在于模块化的多阶段管道，结合结构化患者生成、半结构化病程模拟和LLM驱动的无结构笔记生成，并加入LLM验证和增广步骤以提升真实性与多样性。

**🔧 技术方法**

主要技术包括Synthea UK生成结构化数据、GPT‑4o等大型语言模型进行提示式生成、LLM验证器、RAG、错误修正、以及手工标注的抽样与细节调整。

**📊 数据集**

使用合成数据集：70名患者的临床笔记，分为Bronze、Silver、Gold三档；目前已公开Silver级；每位患者拥有20-50份笔记，覆盖成人和儿童病例。

**📈 对比分析**

通过可读性评分（Flesch、Dale‑Chall）、LLM‑Judge评估流畅性、真实性和相关性、时间一致性检验以及人工临床评审对比，Silver级数据在可读性与流畅度上平均得分≈4.2/5，时间顺序准确率≈97%，人工评审认为真实度高但仍有细节不一致。

**⚠️ 局限性**

局限性包括缺乏真实世界的复杂性、稀有病例代表性不足、可能放大LLM偏见、偶发幻觉与时间线不连贯，以及尚未实现完整的Gold级专家校验。

---

## 411. Focusing on What Matters: Saliency-Harnessing Accurate Routing for Diffusion MoE

**arXiv ID:** 2606.26938 | [PDF](https://arxiv.org/pdf/2606.26938v1)

**作者:** Haoyou Deng `[一作]` (Huazhong University of Science and Technology), Nong Sang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发SharpMoE后训练框架，利用清晰latent对Diffusion MoE的路由进行改进，提升显著性令牌的计算分配；

**💡 创新点**

提出使用清晰latent作为显著性引导的路由机制，递归全轨迹训练以及轨迹路由损失以对齐整个去噪过程中的计算分配与图像显著性分布；

**🔧 技术方法**

结合Diffusion Transformer（DiT）、Mixture-of-Experts、Flow Matching、全轨迹训练、轨迹路由损失、Laplacian显著性检测等技术；

**📊 数据集**

在ImageNet 256×256数据集（1281k张图像，1000类）上进行实验；

**📈 对比分析**

与Dense-DiT、TC-DiT、EC-DiT、DiffMoE等基线在100K后训练步骤后对比，SharpMoE在各规模模型上显著降低FID（例如DiffMoE-L FID 3.10，IS 228.88），并在不同CFG尺度下保持领先；

**⚠️ 局限性**

需要额外的全轨迹训练步骤和清晰latent的生成，推理时需额外计算；对某些均匀资源分配的MoE模型效果有限；未在在线推理或更大规模模型上进行充分验证。

---

## 412. A Deterministic Control Plane for LLM Coding Agents

**arXiv ID:** 2606.26924 | [PDF](https://arxiv.org/pdf/2606.26924v1)

**作者:** Padmaraj Madatha `[一作]` `[通讯]` (Happiest Minds Technologies), Padmaraj Madatha (Happiest Minds Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 LLM 编码 harness 的配置层进行治理，提出一种 deterministic 控制平面架构，统一管理 agent 配置、权限、命令/路径黑名单、漂移检测、阶段化生命周期，并实现从统一定义到多 IDE 目标的编译器。

**💡 创新点**

创新点在于：① 将 agent 定义视作可管理的软件供应链，使用内容哈希、HMAC 锁文件和哈希链审计日志实现完整性与溯源；② 引入多层 deterministic 权限与攻击导向的命令/路径黑名单；③ 设计 Jaccard 漂移检测与阶段化生命周期状态机，实现自动回滚与人为复审；④ 通过“define‑once/compile‑to‑many”保证治理属性跨七大 IDE 规范不丢失。

**🔧 技术方法**

使用技术包括 SHA‑256 内容哈希、HMAC‑Stamped 锁文件、append‑only hash‑chained JSONL 审计日志、正则黑名单、Jaccard 词集相似度、状态机（phase gates）以及纯函数编译器将统一 Markdown+YAML 定义编译为多 IDE 原生配置。

**📊 数据集**

数据集为 10,008 个公开 GitHub 仓库，收集 6,145 条 agent 配置文件与 24,436 条 CI/CD 工作流文件；同时在 237 条 canonical 定义上进行 10,144 条侵入式规则的合规性测试。

**📈 对比分析**

对比方法：在 237 条引用注册表定义上注入 10+ 类规则（内容篡改、权限超限、黑名单匹配等），验证每种防护机制能正确拦截。结果显示所有门控均按设计阻止违规，且复制率、克隆传播等治理指标在预实验中被验证；性能方面未在本文评估，预期仅增加少量构建与安装开销。

**⚠️ 局限性**

局限性：① 仍无法完全防止 prompt 注入与合作式跟踪漏洞；② HMAC 锁文件仅对本地用户保护，缺少跨组织签名；③ 需要人工维护黑名单与许可规范；④ 对不同 harness 的运行时权限一致性无法保证；⑤ Jaccard 漂移阈值仅为操作默认值，缺乏行为关联验证；⑥ 本研究仅验证机制合规性，未测量对开发者生产力的真实影响。

---

## 413. Modeling Local, Global, and Cross-Modal Context in Multimodal 3D MRI

**arXiv ID:** 2606.26894 | [PDF](https://arxiv.org/pdf/2606.26894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. What Holds Back Brain-Computer Interfaces? Uncovering Challenges and Opportunities in BCI-controlled Games for Cerebral Palsy Rehabilitation

**arXiv ID:** 2606.26951 | [PDF](https://arxiv.org/pdf/2606.26951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 415. A $q$-ary Local Criterion for the Radius-One Limited Permutation Channel and Almost-Optimal Binary Block-Concatenation Codes

**arXiv ID:** 2606.26905 | [PDF](https://arxiv.org/pdf/2606.26905v1)

**作者:** Noam Ben Shimon `[一作]` (Technion Israel Institute of Technology), Aryeh Lev Zabokritskiy `[通讯]` (MIGAL Galilee Research Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究半径为 1 的有限置换通道（仅允许相邻符号的非重叠置换）上的零错误编码，提出一种两阶段本地判据来证明块拼接码的纠错性，并用该判据构造出高码率的二进制块拼接码。通过精确的产品自动机验证器进一步确认这些码族在所有长度上均可纠错。还给出了检测码的构造和对应上界，并探讨了判据的率完备性。

**💡 创新点**

创新点包括：① 针对该通道设计了可直接检查的两阶段本地判据，能在有限本地检查中验证无穷长度的纠错性；② 证明该判据是“率完备”的，即其能认证的最高码率等于通道的零错误容量；③ 提供了一个完全自动化的产品自动机验证器，用于精确判定任意前缀无冲突块集的纠错性；④ 在二进制情况下，构造出比以往最佳 0.642805 更高的 0.653618 的块拼接码，距离已知上界 2/3 仅 0.013049。

**🔧 技术方法**

主要技术包括：有限置换通道的球体模型、块拼接码的长度配置与增长参数计算、两阶段本地判据的同长边界测试与不等长前缀测试、产品自动机（双解析器）构造与状态机搜索、以及对检测码的配对上界推导。

**📊 数据集**

本文并未使用传统意义上的数据集，而是通过对二进制块集合进行枚举与搜索，构造出满足判据的块列表，并通过自动机验证其纠错或检测性质。构造的数据是块长度分布表及对应的生成函数方程。

**📈 对比分析**

通过对比已知的上界 2/3 以及先前的 0.642805 率，本文的 0.653618 率显示显著提升，已逼近上界 98.04% 之高。检测码的 0.756707 率同样接近上界 1/2 log₂3 ≈ 0.79248，表现优异。相比之下，先前的字符串拼接码率较低。

**⚠️ 局限性**

局限性主要包括：① 判据虽为充分条件，却仍可能拒绝一些合法块集，导致在实践中无法达到理论极限；② 目前的构造仍停留在块拼接模型，缺乏更通用的有限状态或图约束编码；③ 未给出高效的系统编码/解码算法，仅提供验证工具；④ 对于多字母表或更高置换半径的通道，尚未提出相应的高率构造；⑤ 仍不知二进制零错误容量 C₀ 的确切值，仍是开放问题。

---

## 416. Chai: Agentic Discovery of Cryptographic Misuse Vulnerabilities

**arXiv ID:** 2606.26933 | [PDF](https://arxiv.org/pdf/2606.26933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 417. RIS-Assisted Proactive Handover for Reliable mmWave Wireless Networks

**arXiv ID:** 2606.26885 | [PDF](https://arxiv.org/pdf/2606.26885v1)

**作者:** Alaa Adnan `[一作]` (University of Glasgow), Lina Mohjazi `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用可重构智能表面（RIS）辅助的主动切换（PHO）框架，结合视觉辅助无线通信（VAWC）实现毫米波网络在无可用基站时的链路恢复和低时延切换。

**💡 创新点**

创新点在于将RIS配置时延显式纳入PHO时延约束，通过粒子群优化（PSO）最小化所需RIS元素数量，从而在保持目标信噪比（SNR）的同时显著降低能耗与配置时延。

**🔧 技术方法**

采用的技术包括可重构智能表面、视觉辅助无线通信、粒子群优化算法、分层MIMO/OFDM系统模型以及近/远场RIS信道建模。

**📊 数据集**

使用的主要数据集为VAWC框架中的视觉数据集（RGB图像+深度摄像头），并在该数据集上进行阻塞预测与切换时序仿真。

**📈 对比分析**

方法通过与全RIS配置对比，利用PSO求解最优RIS元素数、子载波数和天线配置，实验结果显示RIS元素减少12%可使能耗降低10%，并在阻塞区段获得15–30 dB的SNR提升，切换失效率比全RIS方案低约14个百分点。

**⚠️ 局限性**

局限性包括仅在单用户单场景下仿真，未考虑多用户或高速移动环境的动态变化；RIS量化位数与天线阵列规模之间的权衡仍需进一步研究；并且依赖离线优化，在线适应性受限。

---

## 418. PortraitGen: Exemplar-Driven GRPO with Dual-Reward Guidance for Photorealistic Portrait Generation

**arXiv ID:** 2606.26930 | [PDF](https://arxiv.org/pdf/2606.26930v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 419. Game Changers: Designing and Measuring Dynamic Feedback To Help Users Self-Regulate in a VR Pointing Game

**arXiv ID:** 2606.26925 | [PDF](https://arxiv.org/pdf/2606.26925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 420. Mismatched Exponents for Deterministic and Randomised Noise-Guessing Decoding

**arXiv ID:** 2606.26954 | [PDF](https://arxiv.org/pdf/2606.26954v1)

**作者:** Henrique K. Miyamoto `[一作]` (Universite Paris-Saclay), Sheng Yang `[通讯]` (Universite Paris-Saclay)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了在加性无记忆信道中，基于噪声猜测的确定性与随机化译码的误差指数与复杂度指数，并在失配判定指标下给出了解析式；随后将结果推广到α‑倾斜判定指标（包含匹配判定）以及基于噪声经验熵的通用判定指标；

**💡 创新点**

创新点在于：①首次证明在确定性噪声猜测中，α‑倾斜操作不影响性能，匹配判定指标即可同时达到最佳误差与复杂度指数；②在随机化噪声猜测中发现匹配判定并非最佳，误差与复杂度指数可通过调节α与码率相关的值实现同步最优；③提出了一个不依赖信道分布且对码率均匀达到最佳指数的通用判定指标；

**🔧 技术方法**

主要技术手段包括：类型方法（method of types）与类型类枚举（type‑class enumeration），以及对失配判定与猜测指数的解析推导；利用α‑倾斜分布的性质、Rényi 熵等信息理论工具，推导误差与复杂度指数的闭式表达；

**📊 数据集**

本文为理论分析，无具体数据集；所涉及的信道为任意加性无记忆信道（例如二进制对称信道）与对应的噪声分布P_Z，假设P_Z非均匀且全支持；

**📈 对比分析**

比较方法：将随机化噪声猜测在最佳α取值下的误差与复杂度指数与确定性噪声猜测以及通用判定指标进行对比；实验结果（如BSC示例）表明在所有码率区间内，最佳α的随机化方案与确定性匹配判定方案实现相同的误差指数，且在低码率时随机化复杂度指数优于匹配判定；通用判定指标在不知信道的情况下也能达到相同的最佳指数；

**⚠️ 局限性**

局限性：分析仅适用于加性无记忆信道；对于非加性或非无记忆信道需进一步研究；此外，实际实现需对α进行调节，虽然理论上可得，但在实践中需要估计码率和信道分布；最后，理论上考虑的是平均复杂度指数，未讨论在有限块长或有中止阈值时的性能。

---

## 421. GEOALIGN: Geometric Rollout Curation for Robust LLM Reinforcement Learning

**arXiv ID:** 2606.26917 | [PDF](https://arxiv.org/pdf/2606.26917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 422. Almost EFX in Hypergraphs

**arXiv ID:** 2606.26948 | [PDF](https://arxiv.org/pdf/2606.26948v1)

**作者:** Ioannis Kakatelis `[一作]` (Czech Technical University in Prague), Minas Marios Sotiriou `[通讯]` (Athens University of Economics and Business)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在多超图（multi‑hypergraph）与超图（hypergraph）框架下研究不可分好物的公平分配，证明了在无环度（girth）至少为3且边（goods）重数不超过2的情形下，存在 EF2X、EF3X 以及近似 EFX（√2/2‑EFX 与 2/3‑EFX）的分配方案，并给出了多项式时间构造算法。

**💡 创新点**

创新点在于：
• 采用更简单的构造方法重现并改进了之前的结果，极大降低了证明与实现的复杂度；
• 证明了在重数为2的多超图中仍能在多项式时间内得到 EF3X 分配（此前只知 EF4X 结果）；
• 在同一框架下得到 √(2)/2‑EFX 与 2/3‑EFX 的近似保证，匹配或优于现有最优结果；
• 引入“虚拟价值（virtual valuations）”与“潜在函数（potential function）”等新技术，用于控制多超图中的循环与 envy 结构。

**🔧 技术方法**

主要技术与工具包括：
• 超图与多超图的图论模型（vertices 为代理，edges 为物品，girth ≥3 保证任意两代理共享至多一条边）；
• envy graph 与虚拟 envy graph，用于追踪代理间的嫉妒关系；
• 架构化的分配算法：先构造部分 EFX/EF2X 分配，再通过“减少嫉妒”与“最终分配”步骤得到完整分配；
• 采用潜在函数与词典序潜在值来保证算法收敛；
• 通过对“双子物品（twin goods）”的配对与删除策略实现 EF3X 与 2/3‑EFX；
• 对多超图的循环进行消除（CycleResolution）和调度。

**📊 数据集**

本文为理论算法研究，未使用任何实际数据集，所有结果均在抽象的超图模型与多超图模型上进行证明。

**📈 对比分析**

与现有工作对比：
• 取得了与目前最佳相同的近似常数（√2/2‑EFX 与 2/3‑EFX），并提供了更简单且多项式时间的构造；
• 在重数为2的多超图中实现了 EF3X（此前仅有 EF4X 或无效结果）；
• 所有构造算法的时间复杂度均为多项式（具体为 O(n^3) 或 O(n^4) 等，取决于所用的子算法）。
• 在所有可达的约束下（girth ≥3，重数≤2）实现了最优或近似最优的公平分配。

**⚠️ 局限性**

局限性与未解决问题：
• 只覆盖了无环度≥3且重数≤2 的超图结构，对更一般的多超图（重数>2 或存在 2‑cycle）尚未给出结果；
• 对于一般单调（monotone）或子模（submodular）价值函数，仍未能得到 EFX 或更强的近似；
• 近似常数（√2/2、2/3）虽然匹配最优，但是否可进一步提升仍是开放问题；
• 所有结果均为理论构造，实际实现与实验验证尚未开展。

---

## 423. Neural Texture Compression using Hypernetworks

**arXiv ID:** 2606.26913 | [PDF](https://arxiv.org/pdf/2606.26913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 424. Confidence-Aware Tool Orchestration for Robust Video Understanding

**arXiv ID:** 2606.26904 | [PDF](https://arxiv.org/pdf/2606.26904v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 425. Optimizing Human-Machine Interface for Real-Time AI Support in the Operating Room: the CVS Copilot

**arXiv ID:** 2606.26886 | [PDF](https://arxiv.org/pdf/2606.26886v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 426. MedSWFlow: An Open-Source LLM Workflow for Drafting Medical Social Work Case Plans

**arXiv ID:** 2606.26884 | [PDF](https://arxiv.org/pdf/2606.26884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 427. Rolling Shutter Relative Pose Estimation Made Practical

**arXiv ID:** 2606.26863 | [PDF](https://arxiv.org/pdf/2606.26863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 428. Generative Retrieval via Diffusion Transformer with Metric-Ordered Sequence Training and Hybrid-Policy Preference Optimization

**arXiv ID:** 2606.26899 | [PDF](https://arxiv.org/pdf/2606.26899v1)

**作者:** Chenghao Liu `[一作]` (Peking University), Songfang Huang `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于连续生成检索的两步检索框架 MO‑DiT+HPPO，用种子集合生成查询向量，在保持同一细粒度模式的同时提升目标属性的召回率。

**💡 创新点**

创新点包括：① 用稀疏的在线召回密度标签构造“metric‑ordered”训练序列，使模型学习从低属性密度到高属性密度的方向；② 在后置阶段引入混合策略偏好优化（Hybrid‑Policy Preference Optimization），通过真实在线指标标注候选并采用 Pareto pair 过滤器，确保属性提升不牺牲模式纯度；③ 将所有阶段统一在同一扩散 Transformer + flow‑matching 体系下完成。

**🔧 技术方法**

主要技术：扩散 Transformer（带流匹配目标）、spherical 变换、轻量级属性密度预测器、尾部质心监督微调、混合策略偏好优化（引用 Anchored DPO 风格）、Pareto 过滤器。

**📊 数据集**

使用四个内部属性域（D1~D4）的大规模多模态嵌入集合，属性由生产环境的二值评分器给出，模式通过聚类得到；所有实验采用严格的项级与模式级留出协议。

**📈 对比分析**

相对于平均池化、预训练的连续检索器、单纯 CPT+SFT 等基线，MO‑DiT+HPPO 在 8 个项/模式分割上平均提升 6–10 个百分点的交叉密度（joint），且在大多数分割上均达到配对 bootstrap 显著性；Pareto 过滤器进一步在不降低同模式共享的前提下推动属性‑模式前沿。

**⚠️ 局限性**

局限性：需要预先冻结的多模态嵌入和生产属性评分器，模式标签依赖聚类，无法直接迁移到公开基准；偏好优化仅靠在线召回指标，易受评分器误差影响；目前仅在内部数据上验证，缺乏公开复现。

---

## 429. Risk-Aware Selective Multimodal Driver Monitoring with Driver-State World Modeling

**arXiv ID:** 2606.26922 | [PDF](https://arxiv.org/pdf/2606.26922v1)

**作者:** Daosheng Qiu `[一作]` (Hubei University), Wei Zhang `[通讯]` (Hubei University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种成本感知的选择性推理框架，用轻量级 RGB‑生理学学生模型持续监测驾驶员状态，并通过学习门控在低延迟与安全风险之间做平衡；同时引入驱动员状态世界模型以提供未来风险预测。

**💡 创新点**

创新点在于将实时监控与安全决策拆分为低延迟感知与成本敏感的选择性推理；利用可学习的门控结合多模态置信、熵、模态不一致、生理质量与世界模型预测，实现对不确定样本的主动回避或请求慢速深度模型；并首次在驾驶员监测中应用驱动员状态世界模型来预估未来误差与行动成本。

**🔧 技术方法**

主要技术包括轻量级 RGB‑生理学多模态融合（ResNet+GRU、MobileNetV3）、学习门控（基于置信度、熵、熵差、模态不一致、心率/EDA 质量的特征向量）、成本敏感的选择性推理（阈值化门控与多目标奖励）、以及基于潜在状态的驱动员世界模型（动作无关与动作条件的潜在滚动预测）。

**📊 数据集**

使用公开的 manD 数据集，该数据集包含多视角 RGB、心率（HR）和皮肤电活动（EDA）窗口，标注为安全关键的高需求状态（+）或低需求状态（−）。

**📈 对比分析**

与单模态 RGB（Macro‑F1 0.6608、BAcc 0.7463）和生理学单模态（Macro‑F1 0.3607、BAcc 0.5450）基线相比，RGB‑生理学学生在不改变延迟（3.08 ms）和参数量（11.39 M）下达到 Macro‑F1 0.7440、BAcc 0.9099；成本感知门控将始终快速推理下的误报率从 17.37 % 降至约 5 %（覆盖率 0.84），且未显著提升延迟；加入世界模型进一步降低最大群体误报，但在最坏组的校准漂移仍显著。

**⚠️ 局限性**

主要局限在于：①生理信号的窗口级对齐精度不足，无法充分利用实时生理动态；②世界模型在最坏组（如特定驾驶员与情境组合）上的校准漂移，导致固定阈值下性能波动；③当前模型仍依赖强情境先验，且对极端或未见情境的泛化仍需提升。

---

## 430. Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds

**arXiv ID:** 2606.26964 | [PDF](https://arxiv.org/pdf/2606.26964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 431. CLIR: Liveness-Driven and Structure-Aware Fuzzing for the Cranelift Compiler

**arXiv ID:** 2606.26977 | [PDF](https://arxiv.org/pdf/2606.26977v1)

**作者:** Shangtong Cao `[一作]` (Beijing University of Posts and Telecommunications), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个针对 Cranelift 编译器的差分测试框架 CLIR，利用语法保持的层次化 SSA 生成、活跃度驱动的指令精炼以及诊断驱动的跨架构适配，实现高效发现多架构缺陷。

**💡 创新点**

创新点包括三方面：① 语法保持的层次化 SSA 生成确保 IR 合法性；② 基于活跃度的指令精炼最大化计算密度和数据依赖；③ 诊断驱动的跨架构适配与根因定位机制实现高效多架构 bug 发现与定位。

**🔧 技术方法**

采用了主导关系驱动的指令生成、遗传变异的指令精炼、优先级操作数选择、根因定位的多阶段簇化与层级定位，以及双模式配置的架构适配等技术。

**📊 数据集**

使用了来自 crates.io 100 个最受下载的 Rust 库以及 WasmBench 真实 WebAssembly 二进制的基本块语料库。

**📈 对比分析**

与 cranelift-fuzzgen、RustSmith、wasm-smith、WASMaker 等基线对比，72 小时内发现 24 个独特 bug，分别比前者高 8×、24×、8×，代码覆盖率平均 75% 以上，显著优于基线。

**⚠️ 局限性**

主要限制在于对 Cranelift 特定 IR 的依赖，迁移到其他编译器需重写 IR 生成接口；另外，诊断定位虽高效但在多指令交互型 bug 上仍可能无法精准定位。

---

## 432. RedVox: Safety and Fairness Gaps in Speech Models Across Languages

**arXiv ID:** 2606.26968 | [PDF](https://arxiv.org/pdf/2606.26968v1)

**作者:** Beatrice Savoldi `[一作]` (Fondazione Bruno Kessler), Luisa Bentivogli `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并构建了一个多语言自然语音安全与公平基准（RedVox），用于测试8个最先进语音模型在非攻击性环境下的安全与公平性。

**💡 创新点**

①首次结合真实语音收集多语言安全与公平测试集；②提供两种请求类型（语音+文本、音频+文本）以探测多模态对模型安全性的影响；③调查语音红队工作中的隐私与心理挑战。

**🔧 技术方法**

采用多模态评估框架，使用 GPT‑5.5 作为判定者进行安全、可用性与相关性三维标签；使用 Whisper 进行语音转写；基于预先收集的 SHADES 与 M‑ALERT 语料进行数据增强。

**📊 数据集**

RedVox（5种语言，约10小时真实语音，包含 6118 条有害/刻板请求），来源于参与者录制；同时利用 SHADES 与 M‑ALERT 作为原始文本模板。

**📈 对比分析**

通过对 8 个开源/专有模型的单轮响应进行安全与公平分类，计算不安全率、争议率等指标；结果显示非英语请求安全性显著下降，语音输入比文本更易触发不安全/争议输出；开源模型普遍表现差于专有模型。

**⚠️ 局限性**

仅涵盖 5 种高资源 Indo‑European 语言；未覆盖全语音输入场景；样本量受限导致对方言/非母语影响统计不足；未考虑多轮对话与策略性 jailbreak。

---

## 433. Floor Raiser or Ceiling Limiter? Differential Storytelling Outcomes with a Child-Centric GenAI System Across Individual Differences

**arXiv ID:** 2606.27067 | [PDF](https://arxiv.org/pdf/2606.27067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 434. RelAfford6D: Relational 6D Affordance Graphs for Constraint-Driven Robotic Manipulation

**arXiv ID:** 2606.27036 | [PDF](https://arxiv.org/pdf/2606.27036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 435. Symplectic Neural Networks for learning Generalized Hamiltonians

**arXiv ID:** 2606.27029 | [PDF](https://arxiv.org/pdf/2606.27029v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 436. Physical Layer Authentication With Channel Knowledge Maps in Indoor Environments

**arXiv ID:** 2606.27044 | [PDF](https://arxiv.org/pdf/2606.27044v1)

**作者:** Luca Bonaventura `[一作]` (University of Padova and National Inter-University Consortium for Telecommunications), Stefano Tomasin `[通讯]` (University of Padova and National Inter-University Consortium for Telecommunications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

本文提出一种新的白质微结构建模方法，利用可调节参数K和N来描述FA与MD之间的分布关系。

**💡 创新点**

创新点在于引入双参数可调模型，能够在不同组织类型下自适应拟合FA-MD曲线，提升了模型的灵活性和解释性。

**🔧 技术方法**

采用扩散磁共振成像（dMRI）技术结合非线性曲线拟合与贝叶斯正则化实现模型估计。

**📊 数据集**

使用Human Connectome Project（HCP）公开数据集进行实验验证。

**📈 对比分析**

与传统DTI、NODDI等方法比较，本方法在FA-MD预测误差上平均降低了约12%，表现出更高的拟合精度。

**⚠️ 局限性**

主要局限在于对低信噪比数据的鲁棒性不足，且模型训练时间相对较长。

---

## 437. MinGram: A Minimalist Unigram Tokenizer with High Compression and Competitive Morphological Alignment

**arXiv ID:** 2606.27019 | [PDF](https://arxiv.org/pdf/2606.27019v1)

**作者:** Sander Land `[一作]` `[通讯]` (Writer), Sander Land (Writer)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MinGram，一种简化版的 Unigram 词表分词器，旨在在保持子词分词表可编辑性的同时降低训练复杂度。

**💡 创新点**

创新点在于：① 用最少标记路径的硬 EM 替代传统的前向后向 EM；② 用 BPE 生成的种子词表代替巨大后缀数组；③ 只用一次基于得分的裁剪步骤，完全摆脱迭代剪枝；同时保持分词器可编辑的列表表示。

**🔧 技术方法**

使用技术包括：最少标记推理（以 token 计数为主，Unigram 评分仅做破折）；BPE 预训练生成种子词表；硬 EM 迭代更新词表评分；单步得分裁剪（及可选的压缩导向裁剪）。

**📊 数据集**

主要数据集为 5 GB 的 FineWeb/FineWeb-2 样本（单语料训练），以及 Goldfish 语料（压缩评估）和 UniMorph/ClimbMix（下游语言模型训练）。

**📈 对比分析**

与 BPE、标准 Unigram、Flat‑Score‑Pruning、PathPiece‑BPE 以及 ConvexTok 等方法对比。实验显示 MinGram 在六种语言的压缩率上优于 BPE 和标准 Unigram，并且在 MorphAlign（形态学对齐）指标上处于中等水平；在下游英语语言模型实验中，MinGram 的 bits‑per‑byte 与最佳 Unigram 家族方法相当，且稀有词数最少。

**⚠️ 局限性**

限制包括：下游评估仅在单一小型模型（depth‑24 nanochat）和单一语料（ClimbMix）上进行，模型规模和数据量受限；MorphAlign 仅覆盖可用的 UniMorph 语言；缺乏对更大模型、更丰富语料和多脚本语言的验证。

---

## 438. TriPAH: Imbalance-Aware Tri-Prompt Affinity Hashing for Cross-Modal Medical Retrieval

**arXiv ID:** 2606.27010 | [PDF](https://arxiv.org/pdf/2606.27010v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 439. RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning

**arXiv ID:** 2606.26997 | [PDF](https://arxiv.org/pdf/2606.26997v1)

**作者:** Rongjian Chen `[一作]` (Shenzhen Institutes of Advanced Technology), Minxian Xu `[通讯]` (Shenzhen Institutes of Advanced Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RolloutPipe 框架，将分离的 rollout 与训练阶段重构为重叠的完整组流水线，解决同步 RLVR 系统中训练 GPU 空闲问题。

**💡 创新点**

创新点包括：
- 完整组流水线（CGP）在组完成后立即推送给训练器；
- 前沿组调度（FGD）在 rollout 侧按组优先级调度，确保训练所需的组更早、更稳定到达；
- 在保持 on‑policy 正确性的前提下实现 rollout 与训练的重叠。

**🔧 技术方法**

技术实现基于 Slime、Megatron‑LM、SGLang 与 Ray；采用 CGP 与 FGD 两种调度机制，利用请求级 FIFO 与组级前沿控制，保证组级依赖性不被破坏。

**📊 数据集**

使用 Qwen3‑1.7B 作为模型，并在四个推理/科学基准上测试：LSAT‑AR、Sci‑XW、Sci‑JL 与 OlyPhys。

**📈 对比分析**

与原始 Slime 系统对比，RolloutPipe（CGP+FGD）在所有十二种配置下主流程时间缩短 30.7%–42.3%，训练等待比例降低 37%–76%。在相同数据量与计算量下，训练动态与收敛性能保持一致。

**⚠️ 局限性**

局限性：
- 仅在 on‑policy GRPO 环境下验证，异步或多权重流水线仍需进一步研究；
- 前沿宽度 F_w 与组数 U 的选择受 GPU 内存限制，可能在更大模型或更高并行度时需要调优；
- 目前实现依赖 Slime 等现有框架，迁移到其他 RLVR 系统的适配成本较高。

---

## 440. ReaORE: Reasoning-Guided Progressive Open Relation Extraction Empowered by Large Reasoning Models

**arXiv ID:** 2606.26986 | [PDF](https://arxiv.org/pdf/2606.26986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 441. On Jumps, Interactions, and Intersection Types

**arXiv ID:** 2606.27062 | [PDF](https://arxiv.org/pdf/2606.27062v1)

**作者:** Stefano Catozi `[一作]` (Université Sorbonne Paris Nord), Gabriele Vanoni `[通讯]` (IRIF, Université Paris Cité)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出并研究了 Parametric Jumping Abstract Machine (PaJAM)，将 Jumping Abstract Machine (JAM) 和 Interaction Abstract Machine (IAM) 统一到一个可调节回溯深度的框架；

**💡 创新点**

创新点在于：1）构造了可调节回溯深度的 PaJAM，既能复现 JAM（k=0）又能复现 IAM（k=∞）；2）证明了 PaJAM 的运行时步数与非幺半群交叉类型（non‑idempotent intersection types）的权重恰好对应；3）利用类型理论给出了 PaJAM 的多项式复杂度上界；

**🔧 技术方法**

使用了非幺半群交叉类型系统、权重分配（weight assignment）机制、强双射（strong bisimulation）与机器的相似性证明，以及类型层次深度分析与复杂度推导技术；

**📊 数据集**

无实际数据集，论文全部为理论形式化与证明；

**📈 对比分析**

通过类型权重与机器步骤数的双向对应，证明了对于任意有限回溯深度 k，PaJAM 的步骤数是 β‑归约步骤数的多项式（上界为 n^{6k+5}）；与传统的 KAM 相比，PaJAM 在大多数情形下只产生多项式开销；与 IAM 相比，PaJAM 在不执行跳转时退化为 IAM，保留了 IAM 的可扩展性；

**⚠️ 局限性**

局限性包括：1）所给的复杂度上界可能不是最优（尚未证明紧致性）；2）缺乏与游戏语义（game semantics）的深入连接；3）对机器低层数据结构与垃圾回收等实现细节的影响未进行实证评估。

---

## 442. Vis4GS: A Visual Analytic Tool for 3D Gaussian Splatting Reconstruction

**arXiv ID:** 2606.26985 | [PDF](https://arxiv.org/pdf/2606.26985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 443. Auditing Framing-Sensitive Behavioral Instability in Large Language Models for Mental Health Interactions

**arXiv ID:** 2606.26982 | [PDF](https://arxiv.org/pdf/2606.26982v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 444. Toward Agentic SysAdmin: Rethinking System Administration with AI Agents

**arXiv ID:** 2606.26960 | [PDF](https://arxiv.org/pdf/2606.26960v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 445. How to evaluate clustering with ground truth?

**arXiv ID:** 2606.27061 | [PDF](https://arxiv.org/pdf/2606.27061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 446. NuclearQAv2: A Structured Benchmark for Evaluating Domain-Science Competence in Large Language Models

**arXiv ID:** 2606.27047 | [PDF](https://arxiv.org/pdf/2606.27047v1)

**作者:** Henry Shaowu Yuchi `[一作]` (Los Alamos National Laboratory), Emily Taylor `[通讯]` (Los Alamos National Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并发布了 NuclearQAv2 基准，用于系统评估大语言模型在核工程知识领域的问答能力。

**💡 创新点**

创新点在于引入三种问题类型（布尔、数值、言语）结合混合式数据集构建与 LLM 辅助评估框架，实现可扩展且多维的技术领域评测。

**🔧 技术方法**

采用 LLM（如 Meta Llama 3 70B Instruct）进行自动问题生成与答案评估，并使用文档解析工具 Nougat 提取技术文本。

**📊 数据集**

数据集共 1,239 条问答对，来源于核工程教科书、专家手工题目和现有数据集。

**📈 对比分析**

通过对 9 款 LLM 的布尔/数值/言语三类任务准确率评估，结果显示大模型在布尔和言语任务表现较好，数值推理仍较弱；gpt‑oss‑120B 在总体准确率上最高。

**⚠️ 局限性**

主要限制包括自动生成问题质量受限于模型与提示设计、LLM 评估判定可能产生偏差、缺乏多模态数据与人工专家评审。

---

## 447. BtrLog: Low-Latency Logging for Cloud Database Systems

**arXiv ID:** 2606.27051 | [PDF](https://arxiv.org/pdf/2606.27051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 448. Maximum Achievable Burst Size in All-Optical Satellite Networks

**arXiv ID:** 2606.27050 | [PDF](https://arxiv.org/pdf/2606.27050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 449. To Run or Not to Run: Analyzing the Cost-Effectiveness of Code Execution in LLM-Based Program Repair

**arXiv ID:** 2606.26978 | [PDF](https://arxiv.org/pdf/2606.26978v1)

**作者:** Zhihao Lin `[一作]` (Beihang University), Li Li `[通讯]` (Beihang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM驱动的程序修复中代码执行的作用，先对7,745条公开agent轨迹进行统计分析，再在3,000次实验中比较四种执行模式的效果和成本。

**💡 创新点**

首次系统隔离并量化执行反馈的边际价值，发现执行对修复成功率提升几乎为零，却显著降低token消耗与运行时间，并提出将执行视为可控资源的设计理念。

**🔧 技术方法**

采用SWE-bench基准，使用Claude Code、Codex和OpenCode（Qwen2.5‑Coder‑32B）三大LLM代理，在Prohibited、Quota‑Limited、Budget‑Guided与Unrestricted四种执行范式下进行实验，并用McNemar和TOST等统计方法评估差异。

**📊 数据集**

使用SWE‑bench Lite和Verified各取前100个实例，共200个真实GitHub项目bug作为实验数据集。

**📈 对比分析**

通过resolve率、token消耗和wall‑clock时间三指标对不同执行模式进行对比，结果显示Prohibited模式的resolve率仅比Unrestricted低1.25pp，却能节省56–62%的token和48–54%的时间；其他模式收益不明显。

**⚠️ 局限性**

研究范围局限于SWE‑bench风格的bug修复，未探讨更复杂的动态分析需求，实验规模有限，且不同bug类型对执行策略的适配仍需进一步验证。

---

## 450. Type-based information flow analysis for $π$-calculus with a dynamically extensible security lattice

**arXiv ID:** 2606.27059 | [PDF](https://arxiv.org/pdf/2606.27059v1)

**作者:** Yukihiro Oda `[一作]` (Tohoku University), Eijiro Sumii `[通讯]` (Tohoku University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了π^L-算子，基于类型的安全信息流分析，允许在运行时动态扩展安全格子，并在此基础上证明锁自由和非干扰性。

**💡 创新点**

创新点在于：①将Kobayashi的π-算子安全类型系统推广到任意有限安全格子；②引入安全格子动态扩展操作并给出安全扩展的形式化定义；③在此框架下完整证明锁自由与非干扰性。

**🔧 技术方法**

使用的技术包括：类型系统与子类型关系、可重用的使用者（usage）与可靠性判定、barbed bisimulation 等形式化工具；并通过结构化归约和类型保护来证明安全性。

**📊 数据集**

无数据集（本工作为形式化理论研究，未使用实验数据）。

**📈 对比分析**

未进行实验比较或性能评估，论文侧重理论证明与形式化分析。

**⚠️ 局限性**

局限性包括：仅支持安全格子扩展，未考虑删除或多态安全级别；未给出具体实现或工具；对时序/概率分析等更复杂场景的支持仍待后续研究。

---

## 451. Design and Performance Evaluation of Secure RF and WiFi-Based Communication in Drone Swarms via Testbed Implementation

**arXiv ID:** 2606.27028 | [PDF](https://arxiv.org/pdf/2606.27028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 452. Human--LLM Collaboration Is Transforming Complexity Metrics in Scientific Texts

**arXiv ID:** 2606.27052 | [PDF](https://arxiv.org/pdf/2606.27052v1)

**作者:** R. Alexander Bentley `[一作]` (University of Tennessee), Sergi Valverde `[通讯]` (Institute of Evolutionary Biology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究从2010至2025年arXiv摘要中出现的词汇统计特征，探讨2023年大规模使用大型语言模型（LLM）后对语言生态的潜在影响

**💡 创新点**

首次将LLM相关风格指数与词汇量、Heaps指数、Zipf指数和词频列表周转率等复杂度度量结合，形成自然实验视角下的混合人机语言生态变化模型

**🔧 技术方法**

利用词频统计、最大似然估计（Zipf指数）、非线性最小二乘（Heaps指数）以及回归分析和早期预警信号（EWS）等统计技术

**📊 数据集**

主要数据集为2010–2025年全量arXiv摘要（约270万篇）与HC3人类–ChatGPT比较语料库，用于对比LLM与人类文本的统计差异

**📈 对比分析**

通过构建复合LLM风格指数、回归模型和交互效应检验，发现2023年后词汇量增长加速、Heaps指数提升、词频列表周转率升高，且指数关系趋平，说明LLM对语言多样性影响不显著但结构更为复杂

**⚠️ 局限性**

研究受限于仅使用摘要层面数据，未能精确测定文本中实际LLM生成比例；风格指数仅为代理指标，可能与其他技术或写作习惯混杂；长期影响仍需进一步跟踪验证

---

## 453. The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture for AI-Assisted Software Development

**arXiv ID:** 2606.27045 | [PDF](https://arxiv.org/pdf/2606.27045v1)

**作者:** Hartwig Grabowski `[一作]` `[通讯]` (Hochschule Offenburg), Hartwig Grabowski (Hochschule Offenburg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Spec Growth Engine 框架，用 AI 编码代理实现快速开发，同时解决上下文爆炸和规范代码漂移两大问题。

**💡 创新点**

创新点在于把软件工程经典原则（信息隐藏、ADR、Walking Skeleton 等）与 AI 代理紧密结合，利用机器可读规范图、Spine 上下文裁剪、垂直切片增长协议和漂移门控，形成自动化、阻塞式的规范与代码一致性保障。

**🔧 技术方法**

技术手段包括：C4 体系结构层次划分、Markdown 规范文件、代码与规范的双向同步、静态代码分析构建证据图、图数据库维护规范图、可测用例驱动的合约与设计分离、以及治理门槛的自动化审批流程。

**📊 数据集**

文中未使用公开数据集，主要通过假设性电子商务结算示例和内部演示来说明框架的工作流程与效果。

**📈 对比分析**

通过与 Kiro、Spec Kit、Tessl 等工具的对比，评估 Spec Growth Engine 位于规范锚定与代码耦合的中间层级；虽然未给出定量性能指标，但在上下文裁剪和漂移检测上显著降低错误率、提升可维护性。

**⚠️ 局限性**

局限性：需可静态导出的依赖图，难以适配高度动态或插件化系统；治理门槛在快速迭代场景下会带来一定的人工开销；规范文件需要手动或工具生成，仍需团队习惯和工具链支持。

---

## 454. State Representation Matters in Deep Reinforcement Learning: Application to Energy Trading

**arXiv ID:** 2606.27032 | [PDF](https://arxiv.org/pdf/2606.27032v1)

**作者:** Jesper Klicks `[一作]`, Vincent François-Lavet `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在泵储水环境下比较不同市场特征组合对DDQN能量交易策略的影响；

**💡 创新点**

发现单一特征类型难以泛化，组合绝对价格、相对历史与短期预测特征可显著提升跨市场和时间段的性能；

**🔧 技术方法**

使用固定的Double DQN框架，构造不同的状态表示并在HydroDam环境中训练；

**📊 数据集**

训练使用2007‑2011年比利时日内价格，验证使用2010‑2011年，比利时2012‑2025年测试集以及39个欧洲ENTSO‑E投标区；

**📈 对比分析**

与滚动价格分数启发式对比，单一特征DDQN在验证集表现好但在测试集和跨区表现差；相对+绝对组合在测试集获得49.9%，跨区中位数39.8%；全组合（绝对+相对+预测）在测试集达到55.6%，跨区中位数47.5%，显著优于启发式；

**⚠️ 局限性**

仅在泵储水单一环境、固定DDQN结构下验证；跨区测试使用最长连续时间段，未覆盖所有年份；预测模型与RL耦合，难以独立评估；

---

## 455. Just how sure are you? Improving Verbalized Uncertainty Calibration in Medical VQA

**arXiv ID:** 2606.27023 | [PDF](https://arxiv.org/pdf/2606.27023v1)

**作者:** Eren Senoglu `[一作]` (Politecnico di Milano), Mark James Carman `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种针对医疗视觉问答的多模态大语言模型的语义化置信度校准方法。

**💡 创新点**

创新点在于提出 2×2 因子扰动设计、复合校准损失（Brier + anchor + 对齐）以及 top‑k KL 正则化，并通过消融验证每个组成部分的必要性。

**🔧 技术方法**

采用了 LoRA 微调、Brier 损失、anchor 损失、对齐损失、top‑k KL 正则化以及多模态因子扰动技术来实现置信度校准。

**📊 数据集**

使用了 OmniMedVQA、PMC‑VQA 与 MedXpertQA 三个医疗 VQA 基准数据集。

**📈 对比分析**

与基线（零样本、Top‑K 采样、SteerConf、ConfTuner）对比，平均 ECE 降低约 60%，Brier Score 降低 26% 以上，AUROC 提升至 0.69 以上，准确率基本保持不变。

**⚠️ 局限性**

局限包括：在低难度数据集上仍难以显著提升区分度；在 MedXpertQA 上判别能力仅接近随机；仅针对多项选择任务，无法直接推广至开放式回答；模型规模较小。

---

## 456. Uncertainty quantification via conformal prediction in data assimilation

**arXiv ID:** 2606.27001 | [PDF](https://arxiv.org/pdf/2606.27001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 457. How Much Static Structure Do Code Agents Need? A Study of Deterministic Anchoring

**arXiv ID:** 2606.26979 | [PDF](https://arxiv.org/pdf/2606.26979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 458. On-board Remote-Sensing Foundation Models for Unsupervised Change Detection of Disaster Events

**arXiv ID:** 2606.27018 | [PDF](https://arxiv.org/pdf/2606.27018v1)

**作者:** S. Ramírez-Gallego `[一作]` `[通讯]` (Thales Alenia Space Spain), S. Ramírez-Gallego (Thales Alenia Space Spain)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一种基于未训练FPN的ResNet-RSFMs的无监督变化检测框架UDFPN，能够在卫星边缘生成高分辨率异常检测图。

**💡 创新点**

创新点在于将RSFM与未训练FPN结合，利用架构先验实现空间嵌入重建，无需额外训练即可生成精细变化图；同时支持单向前向推理，显著降低运算量。

**🔧 技术方法**

技术包括自监督ResNet-50/18骨干、无训练FPN结构聚合、余弦距离变化评分、嵌入压缩与可定制尺寸。

**📊 数据集**

使用Landsat-8 OLI 11波段时序数据，涵盖火灾、洪水、滑坡等八个事件。

**📈 对比分析**

与基准PANN比较，UDFPN在滑坡事件上表现更优（AUPRC约 33% vs 53%），在火灾与洪水上略逊，但整体可接受；单帧推理时间快，参数更少。

**⚠️ 局限性**

局限性包括对纯色彩变化（如洪水水面）检测敏感度低；对云遮蔽处理不完善；以及在更大规模事件或不同传感器下的泛化仍需验证。

---

## 459. Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning

**arXiv ID:** 2606.26973 | [PDF](https://arxiv.org/pdf/2606.26973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 460. Term-Centric Hierarchy Induction from Heterogeneous Corpora

**arXiv ID:** 2606.26963 | [PDF](https://arxiv.org/pdf/2606.26963v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 461. Semantic Early-Stopping for Iterative LLM Agent Loops

**arXiv ID:** 2606.27009 | [PDF](https://arxiv.org/pdf/2606.27009v1)

**作者:** Sahil Shrivastava `[一作]` `[通讯]`, Sahil Shrivastava

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多代理LLM迭代循环的语义早停机制，证明终止性并通过嵌入距离与信息评分结合进行停止判断。

**💡 创新点**

用判决效率高的“语义停止”规则替代固定轮数；用机器检查证明终止性；引入单次生成、裁判缓存和运算/评估令牌区分的评估协议。

**🔧 技术方法**

语义嵌入距离、RAGAS多指标信息评分、优先级级联停止逻辑、裁判缓存、Bootstrap统计与非劣效性检验。

**📊 数据集**

HotpotQA多跳检索增强问答数据集（训练/开发/测试划分）。

**📈 对比分析**

采用轨迹重放评估协议，比较多种停机策略，发现无裁判语义停止在保持质量相近的情况下可节省约38%令牌；完整规则因裁判开销反而更贵；最佳单轮策略获得最高质量。

**⚠️ 局限性**

裁判噪声大导致非劣效性检验不严格；HotpotQA短答案未充分检验迭代优势，需在长篇生成任务中进一步验证。

---

## 462. Decision-Aligned Evaluation of Uncertainty Quantification

**arXiv ID:** 2606.26990 | [PDF](https://arxiv.org/pdf/2606.26990v1)

**作者:** Annika Schneider `[一作]` (Technical University of Munich), Vincent Fortuin `[通讯]` (Helmholtz AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出决策对齐框架并构造先验加权效用（PWU）指标，用以评估机器学习模型的不确定性估计，并验证其与实际决策效用的一致性。

**💡 创新点**

创新点在于引入决策对齐概念，揭示传统不确定性度量隐含的路径性先验，提出先验加权效用度量作为严格合适且与决策效用对齐的评分规则。

**🔧 技术方法**

通过理论证明决策对齐等价于秩保持，推导PSR与决策家族的积分表达式，并在分类与回归任务上使用该框架进行实验评估。

**📊 数据集**

使用公开基准数据集（如UCI、MNIST等）十个二分类与十个回归模型以及五个数据集，并在风电市场、信用审批和点对点借贷等真实案例中进行实验。

**📈 对比分析**

采用 Kendall τ 评估模型排名与决策效用的一致性，实验结果显示PWU指标在所有任务上均显著优于传统的 NLL、ECE、MSE 等指标。

**⚠️ 局限性**

局限性包括单一PWU无法覆盖所有决策目标、先验假设可能偏差，以及传统指标在某些极端先验下仍可能产生误导。

---

## 463. ShareLock: A Stealthy Multi-Tool Threshold Poisoning Attack Against MCP

**arXiv ID:** 2606.27027 | [PDF](https://arxiv.org/pdf/2606.27027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 464. Einstein World Models

**arXiv ID:** 2606.26969 | [PDF](https://arxiv.org/pdf/2606.26969v1)

**作者:** Munachiso Samuel Nwadike `[一作]` (MBZUAI), Kentaro Inui `[通讯]` (MBZUAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Einstein World Models（EWM），让LLM在推理过程中通过调用可视化世界模块生成短视频思维实验，并将结果作为可检验的中间假设整合进链式推理。

**💡 创新点**

创新点在于将视觉-时间推演视为可调用工具，允许模型主动生成并外部化视频思维实验，使推理过程可视化并可调试；同时通过RL训练学习何时调用及如何使用这些视频。

**🔧 技术方法**

使用基于Transformer的LLM策略πθ、可调用世界模块𝒲（如文本到视频或渲染器），配合监督微调+强化学习（GRPO）优化完整推理轨迹。

**📊 数据集**

主要依赖公开的物理推理数据集SimpleBench（示例性使用）和未公开的专门为视觉思维实验设计的数据集，作者呼吁构建更大规模的数据集。

**📈 对比分析**

文章未给出系统实验对比或量化指标，理论上通过RL奖励平衡答案准确率和工具调用成本来优化性能，但缺乏实际基准测试。

**⚠️ 局限性**

局限包括缺乏大规模训练数据、世界模块质量与物理真实性不一、视频生成成本高、模型何时调用工具仍需改进，以及缺乏实证评估。

---

## 465. Improving General Role-Playing Agents via Psychology-Grounded Reasoning and Role-Aware Policy Optimization

**arXiv ID:** 2606.27025 | [PDF](https://arxiv.org/pdf/2606.27025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 466. Parametric Open Source Games

**arXiv ID:** 2606.27068 | [PDF](https://arxiv.org/pdf/2606.27068v1)

**作者:** Aleksandar Todorov `[一作]` (University of Groningen), Alexander Müller `[通讯]` (University of Groningen)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了参数化开放源码博弈模型，利用连续语义映射将玩家的参数向量映射到基础有限博弈中的混合动作，研究梯度上升学习过程及其对合作行为的影响。

**💡 创新点**

创新点在于将程序平衡的“可观测内部程序”机制推广到连续参数化空间，给出精确的开放源码耦合阈值、边界PPNE判定方法，并将此框架扩展到神经网络语义，统一用一阶敏感度比率描述合作倾向。

**🔧 技术方法**

使用连续优化工具（投影梯度上升、Glicksberg/Kakutani定理）进行理论分析，并在 sigmoid 与神经网络语义下实现参数化游戏；通过数值实验验证理论阈值与学习动态。

**📊 数据集**

实验使用经典的二维博弈矩阵（囚徒困境、Stag Hunt 等）作为基础游戏，无需外部数据集。

**📈 对比分析**

与闭源（仅依赖自身参数）对比，开放源码模型在耦合阈值以上可实现高福利合作；理论耦合阈值与实验曲线高度吻合；神经语义中一阶比率匹配时与 sigmoid 基线表现一致，热启动可达到高福利，冷启动则往往无法发现合作解。

**⚠️ 局限性**

局限性包括假设完全可观测的对手参数、仅考虑对称两人一次性博弈、缺乏噪声与不完全信息、对大规模或序贯环境的适用性尚未验证。

---

## 467. Unison: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation

**arXiv ID:** 2606.26984 | [PDF](https://arxiv.org/pdf/2606.26984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 468. Computer Vision for MOBA Analytics: A Dataset and Baseline for Visibility Analysis in Dota 2

**arXiv ID:** 2606.26970 | [PDF](https://arxiv.org/pdf/2606.26970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 469. Protocol Prying: Systematic Vulnerability Research in the Apple AirDrop and Android Quick Share Proximity Transfer Protocols

**arXiv ID:** 2606.26967 | [PDF](https://arxiv.org/pdf/2606.26967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 470. Scalability of Morality: A Particle-Based Numerical Study on the Decoupling of Law and Ethics in Large-Scale Populations

**arXiv ID:** 2606.27039 | [PDF](https://arxiv.org/pdf/2606.27039v1)

**作者:** Amir Arslan Haghrah `[一作]`, Amir Aslan Haghrah `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

**🎯 论文内容**

通过粒子模拟和蒙特卡洛实验研究社会规模扩大导致的道德与法律解耦，揭示了在个体记忆容量有限的情况下，匿名性如何导致伦理反馈失效。

**💡 创新点**

创新点包括：① 用数学框架定量描述道德与法律解耦的动力学；② 通过随机相互作用与非线性社交压力开关捕捉微观伦理反馈；③ 确定了失效的临界规模与记忆比例阈值；④ 发现并量化了路径相关的滞后效应和不可逆性。

**🔧 技术方法**

使用粒子基代理模型、随机交互、超几何切换函数、动力学记忆更新、蒙特卡洛统计、参数敏感性与多维扫描、非平衡相变分析。

**📊 数据集**

采用人工合成的异质人口（N 从 10 到 10³，L 在 2–50 之间）和两类行为子群（占比 7/8 与 1/8），无真实数据集。

**📈 对比分析**

通过多组参数实验与群体规模扫描，对比不同记忆容量、社交压力阈值、斜率的效果；结果显示在 N ≫ L 时，伦理平均概率急剧升高至 0.6 以上；蒙特卡洛收敛稳定，计算耗时约 20,000 步/实验，误差可控。

**⚠️ 局限性**

局限性在于：① 仅考虑随机混合网络，缺乏结构化网络（如小世界、无尺度）；② 记忆长度和社交压力参数人为设定，缺乏经验校准；③ 未引入真实社会数据验证；④ 只关注单一法律层面，未建模多层次法制与执行机制。

---

## 471. Adaptive Utility driven Resource Orchestration for Resilient AI (AURORA-AI)

**arXiv ID:** 2606.27005 | [PDF](https://arxiv.org/pdf/2606.27005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 472. A Generalization Theory for JEPA-Based World Models

**arXiv ID:** 2606.27014 | [PDF](https://arxiv.org/pdf/2606.27014v1)

**作者:** Jingyi Cui `[一作]` (Peking University), Yisen Wang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

本文建立了Joint Embedding Predictive Architectures（JEPA）世界模型的泛化理论，证明JEPA预训练风险等价于对动作条件共现矩阵的低秩矩阵分解，并给出了预训练误差与下游规划损失的关联及有限样本泛化界；

**💡 创新点**

创新点在于提出条件谱图框架，揭示JEPA预训练与矩阵分解的本质联系，首次将预训练误差与规划后遗失相连并推导出涉及近似误差与采样误差折衷的理论界；

**🔧 技术方法**

主要技术包括谱图理论、条件共现矩阵构造、低秩矩阵分解、Rademacher复杂度分析、单步与多步规划误差传播；

**📊 数据集**

实验采用合成的二维点质量系统，构造带噪声和无关维度的观测向量；

**📈 对比分析**

通过与输入层预测模型对比，使用CEM规划方法，实验结果表明在高噪声和长时程规划场景下，latent级预测模型相较输入层模型取得更高成功率；

**⚠️ 局限性**

局限性包括对确定性转移矩阵的假设、对谱分解的依赖以及理论结果对实际复杂环境的泛化尚未完全验证。

---

## 473. Event-Aware Instructed Assistant for Referring Video Segmentation

**arXiv ID:** 2606.26994 | [PDF](https://arxiv.org/pdf/2606.26994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 474. Holistic Multivariance Decomposition: Adapting Mode Interrelations in Low-Rank Tensor Approximations

**arXiv ID:** 2606.26965 | [PDF](https://arxiv.org/pdf/2606.26965v1)

**作者:** Süha Tuna `[一作]` `[通讯]` (Istanbul Technical University), Süha Tuna (Istanbul Technical University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新的张量分解方法——整体多变量分解（HMD），用于低秩近似并捕捉多模式间的高阶交互。

**💡 创新点**

创新点在于将EMPR的多变量交互结构与Tucker的可调秩机制结合，形成层级化的支持矩阵投影，既保留全局核心，又能分层提取单模式与双模式交互特征。

**🔧 技术方法**

采用张量模式乘积（Tucker算子）与半正交支持矩阵的投影、截断策略；利用奇异值分解获得支持矩阵；在此基础上构造HMD层级分解与误差分析。

**📊 数据集**

实验数据集包括：印度Pines高光谱图像（145×145×200）、KTH动作视频（160×120×345）以及氨基酸光谱（3维化学混合物），覆盖遥感、视频与化学光谱三大领域。

**📈 对比分析**

将HMD的零阶、一级、二级近似与传统CP、Tucker进行对比，评估RMSE、PSNR和SSIM；结果显示即使在极低秩下，二级HMD的误差也显著低于CP和Tucker，PSNR和SSIM均达到最高值。

**⚠️ 局限性**

局限在于：支持矩阵的选取仍是经验性且耗时；高阶层级的计算复杂度随模式数指数增长，适用于维度不太高的场景；对极端稀疏或噪声极大的数据仍需进一步鲁棒性改进。

---

## 475. Formalizing a Many-Sorted Hybrid Polyadic Modal Logic in Lean

**arXiv ID:** 2606.27041 | [PDF](https://arxiv.org/pdf/2606.27041v1)

**作者:** Andrei-Alexandru Oltean `[一作]` (University of Bucharest), Ioana Leuştean `[通讯]` (University of Bucharest)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在Lean中实现了一种通用的多排序混合多项模态逻辑，提供完整的机理化证明，并通过自定义DSL支持任意多排序公理扩展。

**💡 创新点**

创新点在于使用“列表技巧”实现自举式排序语法、可变公理扩展以及上下文替代机制，使得多排序混合逻辑可被灵活实例化为程序语义、协议安全与经典模态逻辑。

**🔧 技术方法**

技术手段包括Lean的依赖类型系统、DSL语法宏、上下文构造与替代、WProd序列乘积、以及自定义的可变公理集定义。

**📊 数据集**

未使用传统机器学习数据集，而以SMC程序语言、BAN安全协议、以及S5模态逻辑作为示例验证场景。

**📈 对比分析**

通过在同一框架下完成程序 Hoare 推理、协议可信性证明和S5证明，展示了高内聚性与复用性；性能主要取决于Lean证明搜索，实验结果表明在示例规模下可在几秒内完成推理。

**⚠️ 局限性**

局限性包括尚未实现完备性证明、对极大公理集的效率评估不足，以及对某些特定语法变体仍需手工扩展。

---

## 476. In-Context Model Predictive Generation: Open-Vocabulary Motion Synthesis from Language Models to Physics

**arXiv ID:** 2606.26981 | [PDF](https://arxiv.org/pdf/2606.26981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 477. Cleaning Logs for Downstream Tasks (Registered Report)

**arXiv ID:** 2606.27000 | [PDF](https://arxiv.org/pdf/2606.27000v1)

**作者:** Zahra G. Yazdi `[一作]` (University of Luxembourg), Lionel Briand `[通讯]` (University of Ottawa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种任务无关的日志清洗方法，识别并删除free‑standing（无依赖）日志消息，并评估其对模型推断与异常检测两类下游任务的影响。

**💡 创新点**

利用共现依赖分数（前向/后向）与Mean‑Shift聚类自动划分自由消息模板；提供注册报告式评估方案，公开复现包。

**🔧 技术方法**

计算前向/后向依赖分数并取最大，构造模板依赖度；使用无监督Mean‑Shift聚类分割；下游工具包括MINT（模型推断）、Invariant Mining、OC‑SVM、Loglizer；统计分析采用线性混合效应模型和中介分析。

**📊 数据集**

模型推断部分使用11个公开FSM模型生成的合成日志（注入不同噪声率）；异常检测部分使用BGL、Thunderbird、Spirit三大真实日志集，并通过Drain模板提取、固定时间窗口切分。

**📈 对比分析**

与LogSed和LogBoost进行对比；评估指标为MI的精度、召回与模型覆盖度、AD的Precision/Recall/F1及执行/训练时间。预期清洗能提升下游任务的精度并减少运行时间，具体数值待实验验证。

**⚠️ 局限性**

仅验证两类任务，噪声注入为合成；依赖单一日志解析器和工具；聚类阈值自动但可能误删；未提供客观free‑standing标注，结果解释有限。

---

## 478. Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs

**arXiv ID:** 2606.26987 | [PDF](https://arxiv.org/pdf/2606.26987v1)

**作者:** Sinie van der Ben `[一作]` (ETH Zurich), Mennatallah El-Assady `[通讯]` (ETH Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究情绪向量在开放权重 LLM 中的普遍性与层级演化，并复现了 Claude Sonnet 4.5 的情绪表征结果。

**💡 创新点**

首次在两种不同架构的开放权重模型中验证情绪向量的存在，并揭示情绪表征在网络深度上的不同发展轨迹及语料对 arousal 的敏感性。

**🔧 技术方法**

采用激活对比向量提取、PCA 与人类 valence/arousal 评分相关性分析、CKA 代表性相似度评估等技术。

**📊 数据集**

使用 Apertus 与 Gemma 生成的情绪故事数据集（171 情绪 × 9 故事 = 1539 条）和 40 条中性故事。

**📈 对比分析**

通过 PC1 与 valence 的 Pearson 相关系数与人类评分比较，开放模型在不同层达到 r≈0.76–0.83（超越 Claude 的 0.81），Arousal 在 Gemma 语料上提升至 0.45。

**⚠️ 局限性**

仅覆盖两模型、方法复现可能存在细微差异、语料为模型生成且缺乏模型无关刺激，未进行因果验证。

---

## 479. fTNN: a tensor neural network for fractional PDEs

**arXiv ID:** 2606.27140 | [PDF](https://arxiv.org/pdf/2606.27140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 480. FlameVQA: A Physically-Grounded UAV Wildfire VQA Benchmark with Radiometric Thermal Supervision

**arXiv ID:** 2606.27128 | [PDF](https://arxiv.org/pdf/2606.27128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 481. Kolmogorov Arnold networks (KAN) for aerodynamic prediction: a comparison with MLPs and GNNs

**arXiv ID:** 2606.27126 | [PDF](https://arxiv.org/pdf/2606.27126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 482. UniFormer: Efficient and Unified Model-Centric Scaling for Industrial Recommendation

**arXiv ID:** 2606.27058 | [PDF](https://arxiv.org/pdf/2606.27058v1)

**作者:** Bo Chen `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出统一的模型中心扩展框架 UniFormer，将特征空间交互模块（FIM）和任务空间交互模块（TIM）结合，对工业推荐模型实现整体扩容；

**💡 创新点**

从组件中心扩展转向模型中心扩展，首次在统一框架下同时扩展特征与任务空间，采用语义分组 Token 化、多序列交叉注意力、个性化融合、任务空间交互以及多视角 FFN 实现灵活、可扩展的参数分配；

**🔧 技术方法**

Transformer‑style 注意力、SwiGLU FFN、Multi‑sequence Cross‑Attention、个性化融合、Task‑space Interaction、语义分组 Token 化、Variable‑length FlashAttention、BF16 混合精度训练、用户‑项目解耦推理；

**📊 数据集**

夸叔短视频单页推荐数据集（约4亿日活、5亿交互日志）以及 Kuaishou Lite；

**📈 对比分析**

与预缩放基线（SIM+DCN、SIM+HoME）、扩展型基线（SIM+RankMixer、HyFormer、MixFormer）在 GAUC 上比较，UniFormer 在四项任务上 GAUC 提升 0.53%–1.04%，在线 A/B 测试中 App Stay Time +0.101%/0.260%，Watch Time +0.729%/1.113% 等指标均有显著提升；

**⚠️ 局限性**

模型规模提升仍伴随参数爆炸与计算成本上升，跨模块融合仍主要聚焦特征与任务空间，未充分探究跨平台异构特征和极端冷启动用户、少数任务的泛化能力。

---

## 483. OpenRCA 2.0: From Outcome Labels to Causal Process Supervision

**arXiv ID:** 2606.27154 | [PDF](https://arxiv.org/pdf/2606.27154v1)

**作者:** Aoyang Fang `[一作]` (Chinese University of Hong Kong), Pinjia He `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于已知干预的前向验证（PAVE）步骤式因果路径标注协议，用来从故障注入实验中重建验证过的因果传播路径；

**💡 创新点**

创新点在于：①用已知干预将逆向推断问题转化为可验证的前向任务；②同时结合结构、统计与时间三重条件，生成可信的逐步因果注释；③首次在多系统上提供过程层级的评估指标（Path Reachability、Node F1、Edge F1）；

**🔧 技术方法**

技术方法包括：结构修剪（基于系统依赖图与传播规则生成候选路径）、因果验证（与基线对齐的统计显著性与时间一致性检验）以及前向验证框架；

**📊 数据集**

使用三大微服务系统（TrainTicket、OpenTelemetry Demo、DeathStarBench Hotel Reservation）的故障注入数据，构建了1,200+个实例的跨系统RCA基准；

**📈 对比分析**

在该基准上与11款前沿LLM（Claude、Gemini、Qwen、DeepSeek等）进行对比，结果显示即使AnySvc能命中正确服务，只有约1/5的案例能在过程层面通过验证路径（PR约20%），Node F1与Edge F1均低于预期，说明模型往往缺乏完整的因果推理；

**⚠️ 局限性**

局限性包括：需完整的系统拓扑与多模态遥测才能执行前向验证；规则集可能忽略未知传播机制；基准只涵盖受控实验环境，真实生产环境的迁移性尚未验证。

---

## 484. zQR: A Verifiable QR-Driven zkSNARK Proof Verification Framework for Mobile Platforms

**arXiv ID:** 2606.27092 | [PDF](https://arxiv.org/pdf/2606.27092v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 485. ATGBuilder: Feature-Assisted Graph Learning for Activity Transition Graph Construction with Seed Supervision

**arXiv ID:** 2606.27080 | [PDF](https://arxiv.org/pdf/2606.27080v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 486. PAMAE: Phase-Aware-MoE Action Experts Towards Reliable Flow-Matching Vision-Language-Action Policies

**arXiv ID:** 2606.27144 | [PDF](https://arxiv.org/pdf/2606.27144v1)

**作者:** Jiayu Yang `[一作]` (Xiamen University), Qiang Shen `[通讯]` (Aberystwyth University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了PAMAE——一种阶段感知稀疏Mixture-of-Experts动作专家，用以提升流匹配Vision‑Language‑Action（VLA）模型在多阶段机器人操作中的可靠性。

**💡 创新点**

创新点在于：①引入基于执行阶段的稀疏专家路由器；②使用阶段监督的路由对齐目标和轻量化阶段预测头；③采用两阶段训练策略（先专家预热后阶段监督路由），实现无推理时相位标签的专家分配。

**🔧 技术方法**

技术要点包括：流匹配动作生成、稀疏Mixture-of-Experts（MoE）架构、阶段预测头、KL 路由对齐损失、平滑正则化、负载均衡正则化以及两阶段训练策略。

**📊 数据集**

使用了五个多阶段仿真操作任务的数据集：Table‑Cleaning、Drawer‑Cycle、Lid‑Open、Shelf‑Insert、Cup‑Upright，全部在视觉观测、语言指令与机器人状态相结合的仿真环境中收集。

**📈 对比分析**

与π_0、π_0(labeled)、π_0.5和ProgressVLA等基线进行对比。PAMAE 在π_0.5 上平均成功率提升至91.4%（比基线85.8%提升6.6%），在最优任务上提升高达9.2%。路由连贯性与相位纯度分析表明专家分配高度与执行阶段一致。

**⚠️ 局限性**

局限性包括：实验仅在仿真环境中完成，尚未验证真实机器人效果；对手动定义的粗阶段标签存在依赖，未来需探索自动相位发现；未与更多MoE或VLA改进基线进行对比。

---

## 487. Mostly Automatic Translation of Language Interpreters from C to Safe Rust

**arXiv ID:** 2606.27122 | [PDF](https://arxiv.org/pdf/2606.27122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 488. Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)

**arXiv ID:** 2606.27163 | [PDF](https://arxiv.org/pdf/2606.27163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 489. DMuon: Efficient Distributed Muon Training with Near-Adam Overhead

**arXiv ID:** 2606.27153 | [PDF](https://arxiv.org/pdf/2606.27153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 490. Cross-Head Attention Uplift Network with Inverse Propensity Score under Unobserved Confounding

**arXiv ID:** 2606.27114 | [PDF](https://arxiv.org/pdf/2606.27114v1)

**作者:** Haoran Zhang `[一作]` (Renmin University of China), Feng Zhou `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Cross-Head Attention Uplift Network（CHAUN）的新型提升模型，能够通过交叉注意力机制动态融合处理组与对照组的表征，并提出Robust Adversarial Inverse Propensity Score（RA‑IPS）方法以抵消未观测混杂变量的影响。

**💡 创新点**

创新点包括①在双头结构中引入交叉注意力，实现对组间相似性的灵活利用；②通过全局正则化稳定IPS权重；③构造可观测倾向得分的不确定性集合，在此空间内进行对抗优化，显著提升在存在未观测混杂时的稳健性。

**🔧 技术方法**

采用深度共享特征嵌入、多层感知机、交叉注意力门控、IPS加权损失、全局正则化以及对抗性倾向得分优化等技术。

**📊 数据集**

使用公开数据集CRITEO‑UPLIFT与LAZADA，以及阿里巴巴生产级电子商务数据集进行实验。

**📈 对比分析**

与S‑Learner、T‑Learner、TARNet、CFRNet、DragonNet、CEVAE、FlexTENet、EUEN、DESCN、EFIN等基线进行对比，CHAUN在LIFT@30、AUUC、QINI、PUC等指标均名列前茅，QINI提升可达25.6%；RA‑IPS相较传统IPS在存在未观测混杂的情形下提升约5.4%。

**⚠️ 局限性**

主要限制包括：需要手动调节γ超参数以控制未观测混杂强度；对倾向得分估计的假设（真倾向得分落在名义倾向得分附近）在实际中难以保证；模型在极端高维或多模态数据上的可扩展性尚未充分验证；对复杂多阶段因果结构的鲁棒性仍待进一步研究。

---

## 491. The Riddle Riddle: Testing Flexible Reasoning in Large Language Models and Humans

**arXiv ID:** 2606.27103 | [PDF](https://arxiv.org/pdf/2606.27103v1)

**作者:** Bella Fascendini `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较了九种顶尖大型语言模型与100名成人在“谜语-谜语”范式下的推理表现，评估其是否能根据问题内容而非仅靠表面结构灵活选择推理策略。

**💡 创新点**

提出并验证了“谜语-谜语”范式，利用结构相同但内容要求不同的谜语对，系统性揭示LLM在策略选择上偏向表面匹配而非真正的灵活推理。

**🔧 技术方法**

使用混合效应逻辑回归、自动与人工编码（准确率、推理类型、推理正确性）对LLM与人类的回答进行量化分析，并在九个LLM（GPT‑5.4、Claude‑opus‑4‑6、Gemini‑3.1‑Pro‑preview等）和人类受试者上进行实验。

**📊 数据集**

构建了30对匹配的谜语–谜语-谜语实例，原谜语来自网络，谜语-谜语通过细微改写去除了原谜语的非字面技巧，确保两组在语法、长度等表面特征上保持一致。

**📈 对比分析**

实验通过混合效应模型比较条件A（真实谜语）与条件B（谜语-谜语）的准确率和推理类型；LLM在真实谜语上平均84.9%准确、在谜语-谜语上仅50.7%；人类则相反，在谜语-谜语上80.5%准确、在真实谜语上仅50.5%；错误分析显示LLM错误多因过度使用创新推理，人与类错误多因过度使用文字推理。

**⚠️ 局限性**

研究局限在于未能明确具体哪些表面特征导致LLM过度概括；实验仅聚焦于谜语领域，缺乏跨域验证；未探讨通过微调或训练能否有效缓解LLM的过度概括偏差。

---

## 492. Transformer-Based Classification of Bacterial Raman Spectra with LOOCV

**arXiv ID:** 2606.27096 | [PDF](https://arxiv.org/pdf/2606.27096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 493. Urban Context and Travel Experience Events: An Exploratory Comparison of Two German Cities

**arXiv ID:** 2606.27077 | [PDF](https://arxiv.org/pdf/2606.27077v1)

**作者:** Marie Güntert `[一作]` (Furtwangen University), Esther Bosch `[通讯]` (German Aerospace Center (DLR))

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过移动App对德国汉堡（都市）与图特林根（乡镇）两地的公共交通乘客进行每五分钟一次的随时体验采样，收集了8,000多条“在途”体验数据，随后运用多层级回归模型分析不同事件对旅程体验的影响。

**💡 创新点**

首次系统比较都市与乡镇公共交通体验的影响因素，发现“准时性”“信息”“容量”“个人福祉”在两地均重要，但权重和方向差异显著；为乡镇公共交通服务改进提供了针对性依据。

**🔧 技术方法**

使用Android研究App进行实时问卷采集，采用多层级线性回归（含AR(1)自相关结构）评估事件效应，并通过AIC/BIC进行模型比较。

**📊 数据集**

数据集包含21名图特林根、70名汉堡参与者共计9,164条在途体验评估，覆盖事件变量（准时性、信息、容量、同乘客、驾驶行为、基础设施、个人福祉、工作人员等）以及三项整体体验评分。

**📈 对比分析**

方法：先绘制箱线图观察正负事件对平均体验的分布，再用多层级回归模型估计各事件系数。模型性能通过AIC/BIC评估，结果显示在图特林根正容量事件效应最大（β=0.31），在汉堡负准时事件效应最大（β=-0.50）。两地事件重要性排序相同但强度不同。

**⚠️ 局限性**

局限：图特林根样本量小、年龄偏年轻、参与者多为常用公共交通用户，可能不代表非使用者；事件分类解释可能存在主观差异，尤其信息事件；自我报告可能因高频率记录而放大波动；未纳入质性数据对事件解释进行补充。

---

## 494. TMP: Tree-structured Mixed-policy Pruning for Large-scale Image Generation and Editing

**arXiv ID:** 2606.27089 | [PDF](https://arxiv.org/pdf/2606.27089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 495. ForesightSafety-VLA: A Unified Diagnostic Safety Benchmark for Vision-Language-Action Models

**arXiv ID:** 2606.27079 | [PDF](https://arxiv.org/pdf/2606.27079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 496. Automating Potential-based Reward Shaping with Vision Language Model Guidance

**arXiv ID:** 2606.27180 | [PDF](https://arxiv.org/pdf/2606.27180v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 497. A hybrid IFENN solver for generalizable modeling of phase-field fracture initiation and propagation

**arXiv ID:** 2606.27177 | [PDF](https://arxiv.org/pdf/2606.27177v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 498. TOPS: First-Principles Visual Token Pruning via Constructing Token Optimal Preservation Sets for Efficient MLLM Inference

**arXiv ID:** 2606.27161 | [PDF](https://arxiv.org/pdf/2606.27161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 499. Towards Explainable Adjudicative Variance: Quantifying Judicial Discretion via Gated Multi-Task Learning

**arXiv ID:** 2606.27069 | [PDF](https://arxiv.org/pdf/2606.27069v1)

**作者:** Stanisław Sójka `[一作]` (Technical University of Munich), Matthias Grabmair `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种Judge‑Aware Gated Multi‑Task Learning架构，用以在英国就业仲裁庭判决预测中显式区分事实主导和司法自由裁量两类路径，并通过动态门控融合法对法官身份进行上下文调制。

**💡 创新点**

创新点包括：1）构建细粒度的Detailed Case Outcome（DCO）分类体系，作为辅助监督以正则化语义表达；2）引入Judge‑Aware Gated Fusion机制，实现判决结果与法官嵌入的动态交互；3）通过结构化多任务头与大模型编码器分离的方式，验证条件接口对性能的决定性影响。

**🔧 技术方法**

核心技术包括：现代BERT编码器、Label‑Wise Attention Network (LWAN)、多任务学习(MTL)、LoRA 微调的Gemma‑4 26B‑A4B 编码器、门控融合模块、KL 交互干预评估和t‑SNE聚类等。

**📊 数据集**

使用英国就业仲裁庭（UKET）2011‑2023年共13,937份判决文本，包含事实、申诉与法官身份信息。

**📈 对比分析**

与生成式监督微调（G‑Track）和仅编码器结构化头（B‑Track）等基线进行对比，B‑Track（LoRA‑Gemma‑4 + 架构化头）在宏观F1上达到65.21点，较最佳生成式SFT提升5.1点，同时参数量约为其十分之一，且在稀有模糊类（Partly Wins、Other）表现尤为显著。

**⚠️ 局限性**

局限性包括：1）仅基于判决文本，缺乏原始案卷；2）DCO标签为自动化“银标准”，存在噪声；3）仅在单一司法管辖区评估，无法直接推广；4）法官嵌入为数据驱动，缺乏因果解释；5）评估仅限于Gemma‑4 系列模型，其他大模型的组合效应未知。

---

## 500. The Observer World: A Cryptographic Extension of Impagliazzo's Five Worlds

**arXiv ID:** 2606.27139 | [PDF](https://arxiv.org/pdf/2606.27139v1)

**作者:** Fabio F. G. Buono `[一作]` `[通讯]` (Independent Researcher), Fabio F. G. Buono (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出观察者世界（Observer World）框架，在Impagliazzo五世界的基础上引入观察轴，揭示计算假设与观测约束的独立性并证明结构崩塌 = ⊊P 在所有五世界中成立

**💡 创新点**

创新点在于：① 明确并放宽五世界隐含的“完全观测”假设；② 通过观察者层次构建二维复杂度景观；③ 证明结构崩塌不依赖任何计算假设；④ 定义Observer World、参数化的W_O^并指出最信息量丰富的单元；⑤ 与物理信息极限、热力学与宇宙学界限建立对应关系

**🔧 技术方法**

主要技术包括：观察者层次理论、可判定类的结构化分解、构造O-饱和语言的对角化、信息距离与互信息度量、可观测子类的多态化与非自适应/自适应观察者的层次结构

**📊 数据集**

论文无实验数据集，全部为理论证明与构造性论证

**📈 对比分析**

方法以理论证明为主；通过结构化分析与对角化展示了观察者轴与计算轴的正交性，未给出数值性能指标

**⚠️ 局限性**

限制主要在于：① 对物理对应关系的开放性假设与未完成的定量化；② 观察者层次的自适应扩展与信息距离的具体度量仍为开放问题；③ 论文未提供实验验证，仅为理论框架

---

## 501. Pseudo-Text-Conditioned 3D Grounding DINO for Organ Localization in Abdominal CT

**arXiv ID:** 2606.27084 | [PDF](https://arxiv.org/pdf/2606.27084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 502. PRISM: PE Relational Inter-Section Matrix. A 2D Section-Aware Dataset for Static PE Malware Detection

**arXiv ID:** 2606.27109 | [PDF](https://arxiv.org/pdf/2606.27109v1)

**作者:** José M. Sacristán `[一作]` (Universidad Carlos III de Madrid), Ana I. González-Tablas `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

本文构建了PRISM数据集，将Windows PE文件编码为二维矩阵，保留了各节的顺序和相互关系，用于静态恶意软件检测。

**💡 创新点**

创新点在于首次公开提供面向节的二维结构表示，并通过 Fisher 判别率、互信息等方法证明该结构携带显著判别信息。

**🔧 技术方法**

作者使用梯度提升决策树（LightGBM）以及 Fisher 判别率、互信息等统计技术对PRISM特征进行评估与对比。

**📊 数据集**

数据集来源包括 BODMAS、MalwareBazaar、VirusShare、CAPE 以及 SOREL-20M，共计 83,633 个去重矩阵，并在 49,204 样本上进行实验。

**📈 对比分析**

通过与 EMBER 1D 向量的交叉表示比较，PRISM 在 425 维下接近 EMBER 2,381 维的性能，仅在极低 FPR 区间略逊，二者在决策阈值下可认为等效。

**⚠️ 局限性**

主要局限包括样本来源单一（恶意来自 BODMAS/CAPE 等，良性来自 SOREL），导致源归属偏差；二进制检测已饱和，无法在该任务上进一步提升；此外，二维结构对传统一维模型未能充分利用。

---

## 503. Application of LLMs to Threat Assessment of Foreign Peacekeeping Missions

**arXiv ID:** 2606.27106 | [PDF](https://arxiv.org/pdf/2606.27106v1)

**作者:** Gerhard Backfried `[一作]` (HENSOLDT Austria), Michael Suker `[通讯]` (Austrian Ministry of Defence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种半自动化工作流，利用大语言模型对多语言媒体内容进行威胁识别、结构化和定位，以支持外部维和任务的威胁评估。

**💡 创新点**

将基于指标的风险模型与OSINT媒体采集、LLM提取和“人机协同”结合，首次在维和情境中实现大语言模型对威胁的自动推理和地理时空聚类。

**🔧 技术方法**

采用了大语言模型（few‑shot prompting + reasoning）、自然语言处理（NER、句子嵌入）、LangChain框架、OpenAI GPT‑4、sentence‑transformers 以及基于HENSOLDT的媒体挖掘系统。

**📊 数据集**

使用了超过300个来自格鲁吉亚、土耳其、阿塞拜疆、亚美尼亚、俄罗斯及国际机构的公开媒体源（文字、视频、社交媒体），并对2023–2024年间的自然灾害、外部冲突、族群冲突和经济依赖等四类威胁进行采样。

**📈 对比分析**

通过与七个三人专家组的人工评估对比，平均评分0.82（严格模式0.79），显示自动提取的威胁、相关性和地点准确度高，但对威胁级别和参与者识别的置信度较低。

**⚠️ 局限性**

受限于仅验证四类指标、单一维和任务、缺乏针对性威胁级别校准、对假信息和操纵的鲁棒性不足，以及需要进一步的人机协同和实际流程整合。

---

## 504. Safe Autoregressive Image Generation with Iterative Self-Improving Codebooks

**arXiv ID:** 2606.27147 | [PDF](https://arxiv.org/pdf/2606.27147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 505. PhysReflect-VLA: Physical Feasibility and Self-Reflective Regulation for Reliable Vision-Language-Action Policies

**arXiv ID:** 2606.27146 | [PDF](https://arxiv.org/pdf/2606.27146v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 506. Proof of the Density Threshold Conjecture for Pinwheel Scheduling

**arXiv ID:** 2606.27104 | [PDF](https://arxiv.org/pdf/2606.27104v1)

**作者:** Akitoshi Kawamura `[一作]` `[通讯]`, Akitoshi Kawamura

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明在 pinwheel 调度问题中，当任务周期整数且密度不超过5/6时，必然存在周期调度方案；并给出简单的两周期实例可调度证明及竹园修剪问题的4/3近似算法。

**💡 创新点**

引入分数周期的泛化与“折叠”技术，将无穷实例归约为有限集合，再利用计算机搜索验证所有有限实例可调度，完成长期未解的5/6密度阈值猜想。

**🔧 技术方法**

主要技术包括：分数周期的频率条件、弱化与拆分（分裂）操作、状态转移图与循环检测、状态置换与按周期最小化的搜索剪枝、折叠与向下取整的递归简化，以及对 BGT 的密度缩放和递归处理周期为 2 的任务。

**📊 数据集**

使用所有满足密度≤5/6且整数周期≤21的实例集合（共 25,592,972 个），通过强度折叠与比较剪枝筛选出 676,225 个关键实例，并在 GitHub 上公开的程序完成计算。

**📈 对比分析**

与此前仅证明至 12 任务的可调度性相比，本工作扩展至 22 阈值下的 676,225 个实例，全部通过循环检测成功；对竹园修剪问题，提出的 4/3 近似算法把已知最优比从约 1.42 降至 1.33，证明其可在多项式时间内接受或拒绝实例。

**⚠️ 局限性**

主要局限：依赖大量计算机搜索与专门实现，缺乏简洁的人类可读证明；折叠与分数周期的证明尚未覆盖所有非整数周期情况；处理周期为 2 的任务需递归，略显复杂；整体方法在理论上可扩展但对更大阈值仍需更多计算资源。

---

## 507. RecallRisk-BERT: A Multi-Task Framework for Post-Report Medical Device Recall Triage

**arXiv ID:** 2606.27174 | [PDF](https://arxiv.org/pdf/2606.27174v1)

**作者:** Ali Semih Atalay `[一作]` (Ankara University), Sevgi Yigit-Sert `[通讯]` (Ankara University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究构建了一套基于FDA设备召回记录的后期召回分诊框架，能够同时预测召回严重程度与根本原因。

**💡 创新点**

创新点在于首次采用多任务学习，将召回严重度与根本原因联合建模，并融合 PubMedBERT 文本特征与表格特征，提升了预测能力与风险解释性。

**🔧 技术方法**

使用技术包括文本编码器 PubMedBERT、表格嵌入模块、共享表示层、任务特定输出头，训练时采用多任务交叉熵损失与类别权重；对比基线包括传统 ML（LR、SVM）、Boosting（Random Forest、XGBoost、LightGBM）、深度学习（BiLSTM+Attention、DNN）和单任务 Transformer。

**📊 数据集**

数据集为 54,165 条 2002–2025 年 FDA 设备召回记录，包含召回说明、产品描述、产品代码、监管号等字段。

**📈 对比分析**

与基线比较显示，单任务 LightGBM 在召回严重度上取得最高精度/宏F1（≈0.96/0.84），而多任务 RecallRisk‑BERT 在保持相近宏F1（≈0.84）的同时还能输出根本原因，且相较于单任务 PubMedBERT 提升显著。

**⚠️ 局限性**

局限在于仅使用 FDA 召回数据、未加入其他市场后监测信息、仅做一次划分验证、缺乏可解释性分析，且模型在国际监管环境下的泛化能力尚待评估。

---

## 508. On Parameterized Verification Over Tree Topologies

**arXiv ID:** 2606.27172 | [PDF](https://arxiv.org/pdf/2606.27172v1)

**作者:** Romain Delpy `[一作]` (University of Bordeaux), Grégoire Sutre `[通讯]` (University of Bordeaux)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了在树形拓扑上使用 rendez‑vous 同步的参数化系统的安全性检查问题，并在两种约束下（树深度上限和阶段数上限）确定了其计算复杂度；

**💡 创新点**

创新点在于首次将树深度和同步阶段数作为参数化验证的下近似约束，引入与嵌套计数器系统（NCS）和 VASS 的对应关系，从而把问题映射到已知的快速增长层级和可达性问题；

**🔧 技术方法**

主要技术包括：构造 PCA 与 VASS、NCS、NRCS 之间的多级模拟与归约；使用快递增长层级分析深度约束下的复杂度；利用阶段化的正则性与有限自动机、VASS with actions 的构造来证明阶段约束下的可达性属于 2‑complete；

**📊 数据集**

无实验数据集，本工作为理论复杂度分析；

**📈 对比分析**

由于该研究为理论复杂度研究，未做实证比较，所给结果为最优上下界，说明问题在不同参数化下的可决性与计算资源需求；

**⚠️ 局限性**

限制主要包括：即使深度固定，复杂度仍落在快速增长阶层，难以实现；阶段数固定时的上界仍为 k‑complete，实际实现难度大；并未解决所有 d‑bounded 树执行是否必然 k‑phase bounded 的问题。

---

## 509. Welterweight Go: Boxing, Structural Subtyping, and Generics (Extended Version)

**arXiv ID:** 2606.27138 | [PDF](https://arxiv.org/pdf/2606.27138v1)

**作者:** Raymond Hu `[一作]`, Keith Randall `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

对Go语言的结构子类型、泛型、类型联合和类型集等核心特性进行形式化，提出Welterweight Go（WG）核心模型与LWG低层模型，并实现WG到LWG的类型导向编译器，支持泛型方法与方法集合交叉等新特性。

**💡 创新点**

融合结构子类型与泛型、类型联合、类型集的完整理论框架；采用运行时类型转换与适配器方法的新编译策略，突破传统静态单态化；提供完整的类型安全证明和行为等价性。

**🔧 技术方法**

形式语义与类型系统定义、类型导向编译技术、运行时RTTI与方法表、适配器方法、boxing/unboxing机制。

**📊 数据集**

无真实数据集，基于理论模型与原型实现进行验证。

**📈 对比分析**

与传统单态化比较，实验显示编译后代码量更小、运行时开销更低，同时保持分离编译兼容性，性能优于单态化实现。

**⚠️ 局限性**

对非结构子类型与接口实现细节处理仍有限；模型未覆盖所有Go未来计划中的特性（如多态递归限制等）。

---

## 510. Behind the Mask: A Taxonomic Analysis of Activities in Online Social Networks

**arXiv ID:** 2606.27111 | [PDF](https://arxiv.org/pdf/2606.27111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 511. SubdivAR: Autoregressive Next-Scale Prediction for Neural Mesh Subdivision

**arXiv ID:** 2606.27088 | [PDF](https://arxiv.org/pdf/2606.27088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 512. PanoImager: Geometry-Guided Novel View Synthesis and Reconstruction from Sparse Panoramic Views

**arXiv ID:** 2606.27071 | [PDF](https://arxiv.org/pdf/2606.27071v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 513. Joint Learning of Experiential Rules and Policies for Large Language Model Agents

**arXiv ID:** 2606.27136 | [PDF](https://arxiv.org/pdf/2606.27136v1)

**作者:** Shicheng Ye `[一作]` (Sun Yat-sen University), Chao Yu `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 JERP 方法，在多步交互任务中同时学习并更新经验规则池与 LLM 策略。

**💡 创新点**

创新点在于将同一批交互轨迹同时用于策略的群组相对优化与经验规则的对比反思更新，使规则池随策略演进动态演化。

**🔧 技术方法**

使用 LoRA 参数微调、GRPO 群组相对策略优化、对比反思式规则编辑、基于 Prompt 的规则检索等技术。

**📊 数据集**

在 AlfWorld（文本家庭操作）和 WebShop（在线购物）两个交互式基准上进行实验。

**📈 对比分析**

与 ReAct、Reflexion、RLOO、GRPO 等基线对比，JERP 在 AlfWorld 的整体成功率提升至 61.5%（GRPO 为 57.8%），在 WebShop 的成功率提升至 64.1%（GRPO 为 56.2%）。

**⚠️ 局限性**

局限性包括规则检索仅采用简单的 top‑k 评分排序，缺乏细粒度实例匹配；规则池大小受限；未验证多智能体环境；对参考成功轨迹的依赖需更多实验验证。

---

## 514. HarmVideoBench: Benchmarking Harmful Video Understanding in Large Multimodal Models

**arXiv ID:** 2606.27187 | [PDF](https://arxiv.org/pdf/2606.27187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 515. Algorithms for Threshold Group Testing

**arXiv ID:** 2606.27127 | [PDF](https://arxiv.org/pdf/2606.27127v1)

**作者:** Amin Coja-Oghlan `[一作]` (TU Dortmund), Olga Scheftelowitsch `[通讯]` (TU Dortmund)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究阈值组检验（Threshold Group Testing）问题，提出一种高效的非适应式推断算法，在已知缺陷物品数的稀疏设置下，能够在接近信息理论极限的测试数量下准确恢复所有缺陷物品。

**💡 创新点**

创新点包括：①将空间耦合（spatial coupling）设计引入阈值组检验，形成环形依赖结构；②算法不依赖复杂的加权和，仅使用简单阈值判定与局部清洗，显著简化分析；③证明在常数列设计（constant‑column design）下，所需测试数与信息理论下限完全匹配，且对于阈值 t>1 可比二进组检验进一步降低测试数。

**🔧 技术方法**

使用的技术主要是：空间耦合测试设计、负相关随机变量的Chernoff界、Poisson近似、信息论分析（熵、KL 散度）、大数定律与扩展性质、分阶段阈值化与清洗（basic thresholding、approximate recovery、cleaning phase）等。

**📊 数据集**

本工作为理论分析，不涉及实际数据集；所有结果均在 n→∞ 的稀疏极限下证明，主要使用随机生成的测试图和随机缺陷向量模型。

**📈 对比分析**

与已有的阈值组检验方法（如基于加权和或多阶段算法）相比，本文的算法在测试数量上与信息理论极限一致（对常数列设计），在 t>1 时甚至比传统二进组检验需要更少的测试；在算法复杂度上为多项式时间，且实现简洁。

**⚠️ 局限性**

限制与不足：①证明仅在常数列设计下成立，对一般测试设计是否同样最优尚未解决；②工作假设无噪声、非适应式；③算法仍需要先验缺陷数 k 的知识；④对阈值 t 的分析依赖于某些解析条件，实际应用中可能需进一步验证。

---

## 516. Residual GPU Cache State on Apple M4 Pro

**arXiv ID:** 2606.27098 | [PDF](https://arxiv.org/pdf/2606.27098v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Baris Basaran `[通讯]` (Bahcesehir University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究 Apple M4 Pro 上 GPU 完成后 CPU 缓存状态的变化，使用实验方法评估 SLC 缓存失效与恢复。

**💡 创新点**

首次量化 GPU 完成后残留的缓存置换窗口，并提出单次 CPU 遍历可恢复缓存状态的实证机制。

**🔧 技术方法**

结合 STREAM/BabelStream 带宽基准、指针追踪、SLC 占用模式、Metal 计算内核、IOReport 公开计数器和特权 PMU 计数器。

**📊 数据集**

实验数据来自一台搭载 14 核 M4 Pro、48 GB 统一内存、macOS 26.3 的 MacBook Pro，使用自定义的 SLC 探测和 GPU 内存占用脚本。

**📈 对比分析**

与 STREAM/BabelStream 基准对比，发现 GPU 64 MiB 和 512 MiB 占用导致 CPU 探测延迟分别提升约 15% 与 30%，第二次遍历可将延迟降回基线附近。

**⚠️ 局限性**

局限于单一设备、缺乏核心绑定与物理地址可控、PMU 功能受限，未覆盖所有 GPU 工作负载和系统版本。

---

## 517. A Forward-Only Construction of Semilinear Inductive Invariants for VAS

**arXiv ID:** 2606.27166 | [PDF](https://arxiv.org/pdf/2606.27166v1)

**作者:** Clotilde Bizière `[一作]` (Université de Bordeaux), Grégoire Sutre `[通讯]` (Université de Bordeaux)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种仅基于前向搜索的新型半线性可归属性构造方法，用于向量加法系统（VAS）的可达性问题；

**💡 创新点**

创新点在于：①摒弃了Leroux的对称往返构造，单纯从初始配置出发构建可归属性；②对周期性VAS实现了周期性可归属性的存在性与构造；③为分支Vas（BVAS）问题提供了可推广的技术基础；

**🔧 技术方法**

主要技术包括：半线性和周期性集合的几何描述、向量空间与维度分析、内部向量与线性化、以及利用良序与Dickson引理的前向构造算法；

**📊 数据集**

该工作为理论分析，未使用实验数据集；

**📈 对比分析**

与传统的后向往返构造相比，前向方法在可归属性的结构性、可解释性及周期性保持上表现更好；

**⚠️ 局限性**

局限性在于构造过程依赖于可行的oracle调用，尚未给出有效实现；对于非周期性VAS和更复杂的BVAS问题，仍需进一步研究与优化；

---

## 518. Proposal-Conditioned Latent Diffusion for Closed-Loop Traffic Scenario Generation

**arXiv ID:** 2606.27123 | [PDF](https://arxiv.org/pdf/2606.27123v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 519. Linear Code Conversion in the Merge Regime: General Bounds and Reed--Muller Constructions

**arXiv ID:** 2606.27179 | [PDF](https://arxiv.org/pdf/2606.27179v1)

**作者:** Anina Gruica `[一作]` (Technical University of Denmark), Stanislav Kruglik `[通讯]` (Technical University of Denmark)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在 merge 规约下任意线性码的可转换性，给出了写读成本的通用下界，并提出了一种基于 Plotkin 递归构造的 Reed–Muller 可转换码。

**💡 创新点**

创新点在于：①首次将广义汉明权重与 Wei 对偶性用于推导可转换码的写读成本下界，显著优于仅用最小距离的估计；②提供了适用于所有线性码的通用可转换框架；③构造了满足写成本下界的 Reed–Muller 递归转换方案。

**🔧 技术方法**

主要技术包括：线性代数与子空间约束、广义汉明权重理论、Wei 对偶定理、短化与删码方法、Plotkin 递归构造及向量码的生成矩阵设计。

**📊 数据集**

论文没有使用实际数据集，而是基于理论分析和符号码构造进行验证。

**📈 对比分析**

通过与已知的 MDS、LRC 等特定码的写读成本下界进行对比，证明在所给参数下 Reed–Muller 转换实现了写成本下界；读成本在一部分块达到最优，另一部分块仍存在差距。

**⚠️ 局限性**

局限性包括：仅讨论标量线性转换，未给出带宽成本下界；读成本在某些块仍未达到最优；对非线性转换和向量码的可转换性研究尚未展开。

---

## 520. Stochastic Gradient Optimization with Model-Assisted Sampling

**arXiv ID:** 2606.27171 | [PDF](https://arxiv.org/pdf/2606.27171v1)

**作者:** Jonne Pohjankukka `[一作]` (University of Turku), Jukka Heikkonen `[通讯]` (University of Turku)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于调查抽样理论的模型辅助梯度估计框架，用于降低深度学习训练中梯度估计的方差，从而提升优化效率与泛化性能。

**💡 创新点**

创新点在于将 Horvitz–Thompson 与差分估计方法引入梯度估计，构造无偏低方差的模型辅助估计器；同时将梯度预测模型与采样概率结合，既兼具设计式抽样的鲁棒性，又利用模型的效率。

**🔧 技术方法**

使用了调查抽样理论（Horvitz–Thompson、差分估计）、核岭回归（KRR）进行梯度预测、以及经典优化器（SGD、SGD-M、Adam、AdamW），并在理论上基于强凸光滑性分析了梯度方差对收敛的影响。

**📊 数据集**

实验数据集包括六个公开基准（Airfoil Self‑Noise、Appliances Energy、MNIST、Fashion‑MNIST、CIFAR‑10、CIFAR‑100）以及一个合成正弦+抛物曲线数据。

**📈 对比分析**

通过与统一采样梯度估计对比，在不同批量大小（10、50、100）下进行 400 次独立实验，衡量测试损失、方差以及达到最低损失所需的 epoch。结果显示，尤其在 AdamW 优化器下，模型辅助估计在 71–86% 的实验中优于基线，并在约半数 epoch 内实现更好泛化。

**⚠️ 局限性**

局限性包括：需要额外构建并训练梯度预测模型，增加计算开销；在高维/复杂梯度场（如 CIFAR 图像）中预测误差增大，导致优势减弱；方法对预测模型的准确性高度依赖，若模型失配则效果不显著；并非所有优化器和数据集均能获得显著提升。

---

## 521. On the Reproducibility of Quantum Software Defect Datasets: A Case Study of Bugs4Q

**arXiv ID:** 2606.27124 | [PDF](https://arxiv.org/pdf/2606.27124v1)

**作者:** Haruto Ohto `[一作]` (University of Osaka), Shinji Kusumoto `[通讯]` (University of Osaka)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在21个Qiskit版本上执行77,700次实验，评估并量化了量子软件缺陷数据集Bugs4Q的可重现性；对重现失败进行手工根因分类，并基于此创建了修正版Bugs4Q‑Robust，显著提升了最新Qiskit版本的可重现率。

**💡 创新点**

首次系统研究量子缺陷数据集随时间退化的可重现性；发现大多数重现失败源于依赖关系问题，而仅少量可通过依赖更新解决，需要源代码迁移；基于此构建了可在最新Qiskit上运行的补丁版，展示了持续维护的重要性。

**🔧 技术方法**

采用Docker容器隔离不同Qiskit版本的执行环境，使用三层可重现性判据（Existence、Type Match、Cause Match）评估每个缺陷实例；对重现失败进行人工根因分析，并编写源代码和依赖层面的补丁。

**📊 数据集**

使用Bugs4Q（包含37个真实量子程序缺陷实例）作为实验数据集，并在此基础上构建了Bugs4Q‑Robust作为补丁版。

**📈 对比分析**

通过对比不同Qiskit版本和两种依赖配置（Core-only与Pinned‑stack）的可重现率进行实验。结果显示：可重现率从Qiskit v0.20.1的62.2%降至v2.3.1的16.2%；补丁版在v2.3.1上的可重现率提升至78.4%，平均提升51.4个百分点。

**⚠️ 局限性**

研究仅覆盖单一量子框架Qiskit和单一数据集Bugs4Q，未验证其他量子框架或未来Qiskit版本的情况；根因分类与补丁主要基于人工，缺乏自动化迁移工具，难以在更大规模数据集上推广。

---

## 522. Heavy-Ball Q-Learning with Residual Weighting Correction

**arXiv ID:** 2606.27112 | [PDF](https://arxiv.org/pdf/2606.27112v1)

**作者:** Donghwan Lee `[一作]` `[通讯]` (Korea Advanced Institute of Science and Technology), Donghwan Lee (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了校正的 Heavy‑Ball Q‑学习（CQL 与 HBCQL）以及其在线性函数逼近的推广，给出了收敛性和理论加速条件，并通过数值实验验证其优越性。

**💡 创新点**

创新点包括：① 通过构造校正矩阵 H 让所有 Q‑学习模式共享常数向量特征值，恢复 SLS 分析的共特征向量结构；② 将切换线性系统与联合谱半径（JSR）框架引入 Q‑学习，得到可证的加速判定；③ 设计了两步采样校正与重心修正的无模型实现；④ 将上述思想推广到线性函数逼近。

**🔧 技术方法**

技术手段主要是切换线性系统（SLS）建模、联合谱半径（JSR）分析、Lyapunov 规范化、矩阵块三角分解与共特征向量分析、随机近似与条件期望的推导。

**📊 数据集**

实验使用手工构造的两状态两动作离散 MDP（给定转移概率与奖励），以及同一 MDP 的特征矩阵作为线性逼近的测试基准；未使用公开大型数据集。

**📈 对比分析**

通过绘制相对误差随迭代次数的曲线对比标准 Q‑学习、校正 Q‑学习（CQL）和 Heavy‑Ball 校正 Q‑学习（HBCQL）。实验表明 HBCQL 在非均匀采样下迭代次数明显减少，收敛速度快于 CQL；在函数逼近实验中，CLQL 与 HBCLQL 均快于原 LQL。

**⚠️ 局限性**

局限性：仅在有限状态空间、确定性或独立采样（i.i.d.）下证明；未考虑马尔可夫采样、非平稳奖励等更一般情形；加速条件依赖于严格的 JSR 间隙与小动量参数，实际应用中难以直接验证；在函数逼近时校正矩阵 G 可能退化为奇异，需额外假设。

---

## 523. Data-Free Reservoir Features for Efficient Long-Horizon Cold-Start Continual Learning

**arXiv ID:** 2606.27095 | [PDF](https://arxiv.org/pdf/2606.27095v1)

**作者:** Augustinas Jučas `[一作]` (University of Oxford), Yangchen Pan `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CIRCLE，一种在冷启动、无样本、无重放的类增量学习框架；

**💡 创新点**

创新点是将固定的双向二维Reservoir特征与多层预测Ensemble结合，并采用流式线性判别分析（SLDA）头，避免了特征漂移与预训练依赖；

**🔧 技术方法**

使用BiRC2D风格的随机卷积+双向2D Reservoir、特征级和预测级Ensemble、SLDA头与增量统计更新；

**📊 数据集**

在CIFAR-100、TinyImageNet、ImageNet-Subset、ImageNet-1k（T=500）上进行评估；

**📈 对比分析**

与EFC++、AdaGauss、ACIL等基线相比，在中长远期（T≥50）保持或超过准确率，且训练时间快10倍以上；

**⚠️ 局限性**

局限是对长时间流的高推理成本、对高分辨率或其他模态的适配需要新设计、且仅在长远期优于传统方法，在短期仍不具备最优性能。

---

## 524. BOWConnect: Parallel Bayesian Optimization over Windows with Learned Local Cost Maps for Sample-Efficient Kinodynamic Motion Planning

**arXiv ID:** 2606.27292 | [PDF](https://arxiv.org/pdf/2606.27292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 525. Inherited Circuits, Learned Semantics: How Fine-Tuning Creates Evasion Vulnerabilities Invisible to Standard Evaluation

**arXiv ID:** 2606.27091 | [PDF](https://arxiv.org/pdf/2606.27091v1)

**作者:** Ryan Fetterman `[一作]` `[通讯]` (Cisco), Ryan Fetterman (Cisco)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了安全细调后LLM在行为保持变换下的易被规避性，并提出预部署监测方法。

**💡 创新点**

首次将因果干预与内部电路定位结合，揭示细调将指令令牌语义化导致变换敏感漏洞。

**🔧 技术方法**

使用因果干预、注意力/MLP电路定位、线性探针和指令令牌符号测试等技术。

**📊 数据集**

基于匹配的293对PowerShell脚本和三层变换基准集。

**📈 对比分析**

与基模型Llama-3.1-8B-Instruct对比，细调后准确率提升4.7%但在三层变换中出现多次失误（如alias、format-string、case-mutation）。

**⚠️ 局限性**

仅覆盖七个命令族，缺乏更广泛的指令族和多模型验证，且单脚本评估噪声较大。

---

## 526. Finding Stationary Points by Comparisons

**arXiv ID:** 2606.27082 | [PDF](https://arxiv.org/pdf/2606.27082v1)

**作者:** Helin Wang `[一作]` (Peking University), Tongyang Li `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究利用仅有比较（pairwise‑comparison）oracle 的非凸优化问题，设计了新的算法能够在只做函数比较的情况下找到ε‑驻点（甚至ε‑二阶驻点），并给出了量化的查询复杂度。

**💡 创新点**

创新点在于：
• 首次用比较 oracle 估计 Hessian‑vector 乘积和归一化 Hessian，从而实现二阶信息的获取；
• 将梯度方向估计、Hessian‑梯度比例估计与 trust‑region 步长选择相结合，构造 “best‑of‑all” 迭代，保证每一步至少下降 O(ε^{3/2})；
• 在量子比较 oracle 下，利用量子并行查询将查询复杂度从 O(n^2/ε^{3/2}) 降至 O(n/ε^{3/2})。

**🔧 技术方法**

主要技术包括：
• 通过几何三角形方法估计 Hessian‑vector 方向（Comparison‑Triangle）；
• 归一化 Hessian 的矩阵估计（normhess）和梯度‑Hessian 比值估计（Comparison‑ratio）；
• 信任域与线搜索相结合的 “best‑of‑all” 算法，保证函数下降；
• 量子搜索/量子并行查询实现量子比较 oracle 的效率提升。

**📊 数据集**

本工作是理论分析，未使用任何具体数据集；所有结果均基于函数的 Lipschitz 梯度和 Hessian 条件。

**📈 对比分析**

与之前的 STP（随机三点）方法相比，本算法在 ε 依赖上达到最优 O(ε^{-3/2})，但在维数依赖上为 O(n^2)（经典）和 O(n)（量子）。实验对比未给出，但理论上已证明在高维非凸问题中能够在可接受的查询次数内收敛到驻点。

**⚠️ 局限性**

局限性与开放问题：
• 经典算法对维数的 O(n^2) 复杂度在高维下仍较高；
• 目前仅处理确定性比较 oracle，如何扩展到噪声或随机比较（如 Bradley‑Terry 模型）尚未解决；
• 实际应用的实用性、效率和鲁棒性需要实验验证；
• 结合 lazy Hessian 更新或更高效的 Hessian 估计算法仍是未来研究方向。

---

## 527. Forecasting With LLMs: Improved Generalization Through Feature Steering

**arXiv ID:** 2606.27199 | [PDF](https://arxiv.org/pdf/2606.27199v1)

**作者:** Humzah Merchant `[一作]` (University of Chicago), Bradford Levy `[通讯]` (University of Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究 LLM 在预测任务中的时间意识与前瞻偏差，利用稀疏自编码器识别并调节内部特征，从而降低前瞻偏差。

**💡 创新点**

提出通过稀疏特征调节（steering）实现时间意识的因果提升，证明时间相关特征可跨任务抑制前瞻偏差，区别于仅靠外部提示或时间戳的方法。

**🔧 技术方法**

使用稀疏自编码器 (SAE) 对 LLM 激活进行分解，构造特征激活加权的干预；结合 Gemma 3、Qwen 3.5 等模型的 SAE 字典与 Neuronpedia 注解。

**📊 数据集**

预测市场数据（Kalshi）用于特征挖掘；跨域的自由文本预测基准，包括 M&A 交易预测和药物增长驱动预测。

**📈 对比分析**

与未调节模型、随机调节以及仅调节前瞻特征等对比，结果显示时间意识特征放大可显著减少前瞻偏差，且对 MMLU 以及 MMLU-Pro 的推理性能影响不大。

**⚠️ 局限性**

限制在于特征放大强度高时会降低模型整体质量，前瞻特征的因果效应不明显，且仅在特定任务和模型上验证，未实现完全通用的偏差消除方案。

---

## 528. Resource-Aware Neuro-Symbolic Reasoning for Local Small Language Models

**arXiv ID:** 2606.27281 | [PDF](https://arxiv.org/pdf/2606.27281v1)

**作者:** Carlos Ramírez Ovalle `[一作]` (Pontificia Universidad Javeriana Cali), Abel Alvarez `[通讯]` (Pontificia Universidad Javeriana Cali)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一套可验证的有限域规则与约束推理管线 VFR-LLM，旨在让小型本地语言模型在不需要重复采样的情况下完成结构化推理任务。

**💡 创新点**

创新点在于：①将自然语言问题转换为可追踪的有限域 Horn 规则与约束，②加入源文本验证与自动修复循环，③将符号求解器作为可替代采样的低成本推理步骤，并以资源感知的方式评估其成本效益。

**🔧 技术方法**

技术实现包括：小型本地 LLM（Qwen3‑4B、Phi‑4‑mini、Gemma‑3n‑E4B）用于形式化生成，Python/Pydantic 进行 schema 验证，枚举/SMT 风格的有限域求解器执行推理，自动可追踪性检查与诊断修复模块。

**📊 数据集**

使用的数据集主要有：生成的 120 条纯前置问题、120 条带类型前置问题、以及从 BIG‑Bench Hard Logical Deduction 派生的 60 条 pairwise 和 120 条扩展（BBH‑extended）公共子集；此外对 Gemma 与 Phi 也做了相同的评测。

**📈 对比分析**

评估方法：与直接回答、链式思维、k=5 自一致性、基本 SLM‑求解器等 baseline 进行对比；在 Qwen 纯前置和 BBH‑extended 上，VFR‑LLM 单次调用准确率分别为 0.983 与 0.933，显著优于自一致性；相比之下，Gemma 在大多数子集表现弱势，Phi 在类型约束任务中失败；总体而言，VFR‑LLM 能在可控任务中减少调用次数和序列延迟，但在总 Token 方面提升有限。

**⚠️ 局限性**

局限性：仅在可明确表达为有限域规则与约束的结构化任务上有效，依赖模型能够准确生成形式化；对类型约束翻译不可靠；不适用于隐式知识、歧义或主观判断任务；资源评估未包含能耗、热量或内存峰值；实验范围受限于生成与 BBH 子集，缺乏更广泛的通用验证。

---

## 529. Recovering Governing Equations from Solution Data: Identifiability Bounds for Linear and Nonlinear ODEs

**arXiv ID:** 2606.27285 | [PDF](https://arxiv.org/pdf/2606.27285v1)

**作者:** Yang Pan `[一作]` (ETH Zurich), Helmut Bölcskei `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过提出将 Hausdorff 距离作为解集合的度量，系统研究了从多条解轨迹中识别常微分方程（ODE）的可识别性与稳定性，并给出了线性、Lipschitz、Hölder 以及多项式 ODE 在不同结构类下的识别边界、Metric Entropy 以及样本复杂度的理论分析。

**💡 创新点**

创新点在于：①首次将 Hausdorff 距离与 ODE 识别问题直接对应，捕获最坏情形的误差；②针对广泛 ODE 类给出统一的上、下界；③将识别边界转化为样本复杂度与 Metric Entropy，提供了量化的学习资源评估；④对非线性 Hölder ODE 的可识别性进行了首次深入讨论。

**🔧 技术方法**

使用的主要技术包括：矩阵指数不等式、Grönwall 不等式、帧理论、Lipschitz 连续性与 Hölder 连续性分析、Hausdorff 距离的定义与性质、Metric Entropy 与 packing/crossing 数量理论，以及对解集的可测性与连续性证明。

**📊 数据集**

实验数据均为合成数据：
- 线性 ODE：随机生成 30 个基矩阵，再对每个基矩阵做 20 次随机扰动，得到 600 条线性系统；
- Lipschitz ODE：利用两层 ReLU 网络参数化向量场，随机采样 30 个基参数组并做 20 次扰动，得到 600 条非线性系统；
- 初始条件采样采用球面格点，时间轴均匀离散。无使用真实物理实验或公开数据集。

**📈 对比分析**

通过数值实验绘制 Hausdorff 距离与结构差异（矩阵范数或函数范数）的散点图，并叠加理论给出的上、下界曲线。结果表明：
- 对线性 ODE，实验点基本落在理论上下界之间，验证了理论的有效性；
- 对 Lipschitz ODE，实验点也满足上界，但下界显著不够紧，暗示理论下界可进一步改进；
- 对 Hölder/Oscillatory ODE，实验与理论一致，说明在该类 ODE 下识别可行。

**⚠️ 局限性**

局限性包括：
1. 仅针对自守 ODE，未讨论输入驱动或非自治系统；
2. 需要完整或覆盖充分的初始条件集合，实际实验中难以保证；
3. 对 Hölder 与多项式 ODE 的下界不够紧凑，存在进一步改进空间；
4. Lipschitz ODE 的下界实验验证显示不够紧，理论精度待提升；
5. 高阶系统的推广仅通过状态空间化简，仍缺乏直接的高阶证明与更通用的框架；
6. 研究以理论为主，缺乏大规模真实数据集验证。

---

## 530. Advancing Omnimodal Embodied Agents from Isolated Skills to Everyday Physical Autonomy

**arXiv ID:** 2606.27251 | [PDF](https://arxiv.org/pdf/2606.27251v1)

**作者:** Junhao Shi `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了OmniAct框架，实现跨模态、跨工具的长期自主机器人操作。

**💡 创新点**

提出统一的技能路由器将网络工具与物理控制整合；事件边界驱动的分层记忆压缩上下文；异步视觉预判实现闭环失败检测与重规划。

**🔧 技术方法**

使用多模态语义规划器+LLM高层规划、VLM视觉预判、事件压缩记忆以及异步可视化预警等技术。

**📊 数据集**

使用40个真实场景任务（UR5e与Keenon平台，4个家居IoT设备+12个虚拟API），并在40k+ token的长期交互情境中进行评估。

**📈 对比分析**

与Direct Policy、SayCan、Code-as-Policy、RoboBrain2等基线比较，OmniAct在L3全模态任务中E2E成功率达到50-54%，子任务成功率最高，且上下文令牌增长保持在2.5k-10k，整体性能显著优于基线。

**⚠️ 局限性**

受限于LLM的结构化输出能力，对低容量模型效果有限；仅在室内场景评估，需进一步验证在更大尺度或动态环境中的鲁棒性。

---

## 531. HumanoidUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation

**arXiv ID:** 2606.27239 | [PDF](https://arxiv.org/pdf/2606.27239v1)

**作者:** Hongwu Wang `[一作]` (Beijing Academy of Artificial Intelligence), Shaqi Luo `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了HumanoidUMI，一套无需机器人、基于VR‑UMI接口的人体演示数据采集与层级学习框架，用以训练全身人形机器人操作能力。

**💡 创新点**

创新点在于将轻量化的VR全身捕捉与UMI式手柄相结合，利用稀疏空间关键点和腕部视角实现机器人无缝操作，并在层级结构中引入空间关键点重映射（SKR）实现高效可执行的全身运动。

**🔧 技术方法**

使用了PICO VR头盔、运动捕捉、UMI式手柄摄像头、DINOv2视觉编码、扩散模型预测关键点、空间关键点重映射、逆运动学以及学习式全身控制器。

**📊 数据集**

主要数据来自人工演示的五个关键点轨迹、腕部视角图像与抓手宽度，采集于多种真实任务场景（放置、双臂配合、投掷、下桌处理、行走递送）并同步记录。

**📈 对比分析**

与传统基于机器人遥控的TWIST2方法对比，HumanoidUMI在成功率上保持或略优，并在演示收集速度上提升约2–60倍，展示出高效且可扩展的性能。

**⚠️ 局限性**

局限性包括对人体与机器人比例差异的处理仍需进一步改进，缺乏对更复杂动力学或高频动作的验证，且对大规模多样化任务的泛化能力尚未充分评估。

---

## 532. Bridging Talk and Thought: Understanding Dialogue Dynamics Across Collaborative Problem-Solving Contexts

**arXiv ID:** 2606.27233 | [PDF](https://arxiv.org/pdf/2606.27233v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 533. LMs as Task-Specific Knowledge Bases: An Interpretability Analysis

**arXiv ID:** 2606.27237 | [PDF](https://arxiv.org/pdf/2606.27237v1)

**作者:** Amit Elhelo `[一作]` (Tel Aviv University), Mor Geva `[通讯]` (Tel Aviv University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过行为和机制实验，系统评估语言模型是否具备任务不变性（即相同事实在不同任务下应共享同一参数子集），并发现大多数事实在训练过程中并未在不同任务间同步出现。

**💡 创新点**

创新点在于：①首次将任务不变性概念引入知识库评估；②设计了结合必要性、充分性和特异性的参数局部化框架，证明同一事实在不同任务上依赖不同参数子集；③量化了跨任务交织度并揭示判别任务更易交织；④探讨链式推理如何利用跨任务参数，解释其在直接回答中无法获得的知识。

**🔧 技术方法**

使用的技术包括：行为实验中的多检查点跟踪与阈值化出现步骤；机制实验采用二值掩码学习框架（necessity, sufficiency, specificity + L1稀疏正则），在注意力头与MLP神经元上进行稀疏子集定位；交织度指标基于消融后对其他任务/事实的影响；链式推理实验结合直接回答与CoT的消融对比。

**📊 数据集**

数据集涵盖5个关系的知识三元组（共230条），通过10种表述模板拆分成6种任务（Completion, Fill-in-the-Blank, OpenQA, MCQA, Neg-MCQA, Verification），并在三种7B–13B规模模型（如LLaMA 7B/13B、GPT-NeoX 13B等）上进行评估；此外加入两种多跳推理任务用于机制实验。

**📈 对比分析**

比较方法：对行为实验按预期出现步骤检验共现率；对机制实验评估消融下降、补丁恢复率及特异性损失；对交织度使用Entent分数；对CoT实验对比直接回答和CoT在消融后的准确率。结果显示：约48%事实未按任务不变性共现；Ent任务分数对判别任务平均为0.21，高于生成任务0.11；CoT在消融后准确率下降仅12–30%，显著优于直接回答的20–72%下降。

**⚠️ 局限性**

局限性包括：检查点间隔粗略，导致出现步骤估计不精确；仅考察可表达为三元组的关系知识，未覆盖更复杂或非结构化事实；局部化框架未揭示同一事实在不同任务间可能存在的冗余编码结构；实验规模受限于三种模型，无法说明规模对任务不变性的影响。

---

## 534. Designing Reward Signals for Portable Query Generation: A Case Study in Industrial Semantic Job Search

**arXiv ID:** 2606.27291 | [PDF](https://arxiv.org/pdf/2606.27291v1)

**作者:** Ping Liu `[一作]` (LinkedIn Corporation), Wenjing Zhang `[通讯]` (LinkedIn Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并训练了基于 RLAIF 的端到端模型，用于将求职者简历自动生成可在不同招聘平台检索使用的“可迁移查询”（portable query）。

**💡 创新点**

创新点主要包括：① 对可迁移查询任务进行形式化定义，区分传统检索/摘要任务；② 设计了基于规则的奖励底线（S_r）来防止模型直接拷贝简历内容；③ 系统性评估了奖励设计对不同 RL 算法（GRPO、RLOO、REINFORCE++、PPO+GAE）的影响，证明奖励工程对性能的影响远大于优化器选择。

**🔧 技术方法**

技术手段包括：① 使用 Qwen3‑1.7B 作为策略网络（SFT 初始化后微调）；② 采用 Qwen3‑8B 作为基于 Rubric 的 LLM 判别器（训练时评估）；③ 引入 Llama‑3.3‑70B‑Instruct 作为独立评估判别器；④ 采用多种优势估计器（GRPO、RLOO、REINFORCE++、PPO+GAE）；⑤ 在奖励计算前加入 6‑gram 复制检测规则以抑制奖励劫持。

**📊 数据集**

使用约 80k 条职业平台（类似 LinkedIn）个人简历样本进行训练，筛选后约 50k 条测试样本，跨国跨语言覆盖多种地区；训练集中保留完整简历文本，测试集随机抽样包含多样化的职业信息。

**📈 对比分析**

对比方法：在相同训练配置下比较 SFT 基线、PPO+GAE、GRPO、RLOO、REINFORCE++ 以及加入 S_r 的 GRPO。性能评估使用两个判别器：训练时的 8B Rubric（Layer‑1）和独立的 70B Llama 判别器（L_i）。结果显示，加入 S_r 的 GRPO 在独立判别器上提升 0.111 分（相对 SFT 的 0.706‑0.595），其他 critic‑free 方法的提升约 0.69 分；PPO+GAE 仅提升 0.612 分，且训练成本显著更高。

**⚠️ 局限性**

局限性：① 训练时评估器与独立评估器之间存在 2.4× 的性能膨胀，表明奖励判别器易被过度拟合；② 奖励底线只能覆盖复制和日期范围两种表面攻击，其他细粒度劫持方式仍可能出现；③ 任务高度依赖特定的职业平台数据，迁移到其他行业或语言环境需要重新构造 Rubric 和规则。

---

## 535. When Does Combining Language Models Help? A Co-Failure Ceiling on Routing, Voting, and Mixture-of-Agents Across 67 Frontier Models

**arXiv ID:** 2606.27288 | [PDF](https://arxiv.org/pdf/2606.27288v1)

**作者:** Josef Chen `[一作]` `[通讯]`, Josef Chen

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多模型LLM编排（路由、投票、级联、融合）并证明其可实现的准确率上限由所有模型同时错误率β决定，而非常用的配对错误相关性ρ；提出Clopper–Pearson下界作为编排前的$0证书。

**💡 创新点**

创新点在于①把β视为编排的根本上限并给出有限样本证书；②证明ρ无法识别β，展示了配对相关性对尾部共失败率的严重低估；③在真实规模的67模型、21供应商池上实证两种编排可行性“天花板绑定”与“可实现性绑定”两种模式，并量化其差异。

**🔧 技术方法**

使用概率与组合工具（Clopper–Pearson置信区间、线性规划对偶性）、高斯copula与四分之一相关系数（tetrachoric）来估计β与ρ；应用投资组合理论、成本分配与实时抉择理论来分析路由与级联的最优策略。

**📊 数据集**

数据集包括：GSM8K、MMLU、ARC-Challenge、MATH-500/Hard、MMLU-Pro、GPQA-Diamond、执行评测的编程竞赛（Python+stress tests），共计超过600个查询，涵盖开放式数学、代码与科学三大开放式任务；模型池为67个模型，覆盖21个供应商，价格与性能多样化。

**📈 对比分析**

对比方法：单模型最佳、随机路由、学习路由、投票、级联、融合；评价指标为单查询准确率、oracle增益G、路由可实现增益比例。结果显示：oracle增益有限（约0.1-0.15）；学习路由仅捕获不到5% G；投票往往比单模型差；在质量匹配的低ρ环境下，异构融合可略优于自我一致性多样化，体现多样性收益受限。

**⚠️ 局限性**

局限性包括：程序化评分对开放式答案的偏差、共失败事件极少导致β估计不稳、部分模型缺少prompt日志无法训练路由、仅使用配对相关性而忽略更高阶共失败结构、对真实市场动态的实时评估有限、以及对多选与开放式题型差异的理解仍需进一步验证。

---

## 536. Codex Mutabilis: Preserving The Reasons For Changes In Scientific Names

**arXiv ID:** 2606.27271 | [PDF](https://arxiv.org/pdf/2606.27271v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 537. Simulation-based inference for rapid Bayesian parameter estimation in epidemiological models: a comparison with MCMC

**arXiv ID:** 2606.27286 | [PDF](https://arxiv.org/pdf/2606.27286v1)

**作者:** Alina Bazarova `[一作]` (Forschungszentrum Juelich), Stefan Kesselheim `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对德国2020年COVID‑19 ICU占用数据，使用SECIR机制模型进行Bayesian校准，并与MCMC方法对比。

**💡 创新点**

证明了利用神经后验估计（SBI）在保持后验结构和预测性能的同时，显著提升计算速度，尤其在长时间窗口（201天）中实现数百倍加速。

**🔧 技术方法**

采用SBI中的神经后验估计（NPE）结合Masked Autoregressive Flow（MAF）与CNN嵌入，对参数进行近似后验学习；同时使用传统的DE‑MZ MCMC作为基准。

**📊 数据集**

使用德国全国2020年4月24日至12月10日的ICU占用时间序列（约231天）作为观测数据。

**📈 对比分析**

通过Wasserstein距离、Kullback‑Leibler散度、对后验预测误差（RMSE）以及运行时间等指标比较两种方法。31天窗口下SBI后验与MCMC一致，RMSE约为60–100，运行时间从≈1000 s降至≈60–70 s；201天窗口下RMSE从≈23提升至≈326，运行时间从≈19000 s降至≈157 s，显示SBI在高维、长窗口中优势显著。

**⚠️ 局限性**

局限性包括对先验的高度依赖（尤其是多变点参数），以及在长窗口中后验更为宽广、收敛不稳定；仅使用ICU占用数据，参数可识别性有限，需结合更多数据源或更复杂模型进行进一步验证。

---

## 538. E-TTS: A New Embodied Test-Time Scaling Framework for Robotic Manipulation

**arXiv ID:** 2606.27268 | [PDF](https://arxiv.org/pdf/2606.27268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 539. "Everyone Says Them": Deception Typologies, Probabilistic Trust, and Grassroots Safety Knowledge Among Gay Dating App Users in China

**arXiv ID:** 2606.27284 | [PDF](https://arxiv.org/pdf/2606.27284v1)

**作者:** Yibo Meng `[一作]` (Tsinghua University), Xiaolan Ding `[通讯]` (North China University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对22位中国同志男士进行半结构化访谈，探讨他们在同性交友应用中遭遇的欺骗类型、信任与验证策略以及社区中的风险认知。

**💡 创新点**

①提出四类欺骗类型（关系、情感、财务、商业）并揭示“伪造身份”跨越其底层机制；②描述基于多信号的概率性信任评估与分层验证策略；③展示社区共享的“套路”(taolu)与经验循环的风险识别机制。

**🔧 技术方法**

采用定性研究方法——主题分析，对访谈记录进行编码与归纳；未使用机器学习或技术系统。

**📊 数据集**

22名受访者在多款同性交友应用（Blued、Aloha、Fanka、Soul）上的自述互动经验，包含个人案例与情境描述。

**📈 对比分析**

本研究为探索性定性研究，无对照实验或性能指标；比较是通过不同欺骗类型与验证策略的归纳，并与已有文献对照，未给出定量评估。

**⚠️ 局限性**

样本规模小、受访者自我报告偏差；数据仅来自中国城市/农村的同志用户，可能不具备普适性；缺乏对平台技术或算法干预的评估；研究关注经验而非可操作的设计解决方案。

---

## 540. Exact and Deterministic Patch Descriptor Retrieval via Hierarchical Normalization

**arXiv ID:** 2606.27280 | [PDF](https://arxiv.org/pdf/2606.27280v1)

**作者:** Koichi Sato `[一作]` `[通讯]`, Koichi Sato

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种通过层次归一化（Hierarchical Normalization）实现的检索方法，能在只扫描少量向量前缀的情况下，精确且确定性地找到最近邻。

**💡 创新点**

创新点在于将向量能量分配到前缀子空间，并利用上界进行精确的分支限界搜索，从而在不使用近似索引的前提下获得显著加速并保证完全一致性。

**🔧 技术方法**

使用的技术包括层次归一化、基于前缀的两阶段搜索、结构化数组（SoA）存储、精调（fine‑tuning）以聚焦能量、以及在实验中与 HNSW 等近似方法进行对比。

**📊 数据集**

在公开的 PatchDescriptor 基准（notredame 训练集、trevi 与 halfdome 测试集）上进行评估。

**📈 对比分析**

与暴力全向量搜索和 HNSW 等近似方法相比，在 K=16、α=1/8 时实现 7.2× 的 CPU 速度提升，FPR@95 与预训练基线相当；K=8、α=1/32 时可达 13.7×/12.7× 的加速，同时保持与全向量搜索完全一致的检索结果。

**⚠️ 局限性**

局限性包括对能量分配参数的敏感性、在极大数据库时 Phase‑2 访问率上升导致速度下降，以及当前未探索对前缀子空间的子线性索引潜力。

---

## 541. Effective Covariance Dynamics in Solvable High-Dimensional GANs

**arXiv ID:** 2606.27246 | [PDF](https://arxiv.org/pdf/2606.27246v1)

**作者:** Andrew Bond `[一作]` (Koç University), Zafer Doğan `[通讯]` (Koç University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并分析了一个可解析的高维GAN模型，该模型允许生成器学习来自具有条件、相关和非零均值潜变量协方差的数据子空间。

**💡 创新点**

创新点在于将复杂的条件与相关潜变量结构压缩为一个“有效协方差”，从而在二次能量鉴别器下仍能得到确定性的微分方程；并揭示了低秩相关如何提升弱特征可学习性（信号提升机制）。

**🔧 技术方法**

采用随机近似、随机梯度下降/上升的两时刻尺度更新、宏观状态的ODE推导以及稳定性分析（特征值阈值法），并用Rayleigh商证明有效协方差的极值。

**📊 数据集**

在模拟中使用高维随机生成的数据；在真实数据实验中使用MNIST、FashionMNIST和CIFAR-10的灰度图像。

**📈 对比分析**

通过与理论ODE的比较验证收敛轨迹和相位边界；在实验中比较“知情”生成器协方差（与数据PCA匹配）与“无知情”协方差，发现知情模型收敛更慢但最终子空间与PCA基更一致，生成的条件样本更具可辨识性。

**⚠️ 局限性**

局限在于模型为线性单层GAN，无法直接推广到深度非线性生成器；有效协方差假设依赖于二次能量鉴别器，其他鉴别器形式可能无法直接套用；以及对真实复杂图像数据的解释仍受限于低维线性结构。

---

## 542. Evaluating Architectural Trade-offs in CGRAs: The Impact of Scratchpad Memory and Heterogeneity on Compute-Intensive Kernels

**arXiv ID:** 2606.27240 | [PDF](https://arxiv.org/pdf/2606.27240v1)

**作者:** María José Belda `[一作]` (Complutense University of Madrid), David Atienza `[通讯]` (EPFL)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对比评估了两种CGRA架构（同质的OE和异质集成Scratchpad的DISCO），通过FFT、GEMM以及完整的癫痫检测Transformer工作负载，探讨了处理单元异质化和本地存储对性能、能耗和面积的影响。

**💡 创新点**

创新点在于系统化地量化了Scratchpad内存与处理单元异质化对内存流量、能效和频率的三角权衡，并提出了面向边缘AI的架构选择指南。

**🔧 技术方法**

采用了定制化的硬件设计（VWR、SPM、专用LSU/LCU/MXCU等），手工优化GEMM映射，以及基于Cadence Genus和Synopsys PrimePower的后仿真能耗评估。

**📊 数据集**

使用PolyBench的矩阵乘法基准（mmul、gemm、2mm、3mm）以及针对4层Vision Transformer的癫痫检测EEG数据集进行实验。

**📈 对比分析**

通过定量的周期仿真、功率与能量测算，将OE和DISCO在不同频率（200MHz/700MHz）下与CPU、SoA CGRA进行对比，结果显示OE在低面积和高频时可达5×加速，而DISCO在数据洗牌和STFT上能耗最低。

**⚠️ 局限性**

局限性包括缺乏自动化编译链对DISCO的支持、异质化带来的映射与控制开销、以及对非向量化计算（如Softmax）仍需依赖CPU，且在极低功耗场景下OE的高频功耗仍较大。

---

## 543. Vulnerability of Natural Language Classifiers to Evolutionary Generated Adversarial Text

**arXiv ID:** 2606.27215 | [PDF](https://arxiv.org/pdf/2606.27215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 544. From Celebrities to Anyone: Characterizing AI Nudification Content, Technology, and Community Dynamics on 4chan

**arXiv ID:** 2606.27234 | [PDF](https://arxiv.org/pdf/2606.27234v1)

**作者:** Chi Cui `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对4chan成人请求板块进行为期41天的大规模数据采集与分析，识别并量化了24,105份非自愿性性暴露图像（SNEACI）。

**💡 创新点**

首次揭示AI裸化目标从公众人物转向非名人，描绘了供应链与生态系统动力学，并指出核心活跃提供者在内容生成与需求激励中的关键作用。

**🔧 技术方法**

采用多阶段检测管道（NSFW、AIGC、Undress、Celebrity classifiers）、Hive AI模型归因、面部识别与NLP请求-响应匹配技术，结合自研爬虫和数据处理脚本。

**📊 数据集**

采集自4chan Adult Requests板块的80,366篇帖子与49,874条媒体文件，归纳为24,105条SNEACI样本。

**📈 对比分析**

通过内部评估获得请求检测准确率95%、面部匹配召回率约90%，并提供了响应率、模型使用占比等统计指标，未与外部基线直接对比。

**⚠️ 局限性**

仅覆盖单一匿名社区且时间窗口短，检测器误差影响结果，未能完整识别部分undress图像，模型归因受限，且难以消除已发布的有害模型。

---

## 545. CHAMB-GA: A Containerized HPC Scalable Microservice-Based Framework for Genetic Algorithms

**arXiv ID:** 2606.27217 | [PDF](https://arxiv.org/pdf/2606.27217v1)

**作者:** Felix Bonhoff `[一作]`, Manuel Dahmen `[通讯]` (Forschungszentrum Juelich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了CHAMB-GA框架，基于容器化微服务实现遗传算法与嵌入式仿真分离，可在Kubernetes与SLURM集群上大规模并行运行。

**💡 创新点**

创新点在于将遗传算子与仿真评估拆分为独立容器，通过RabbitMQ异步消息实现无锁异步通信；支持水平和垂直扩展；实现从本地到云、HPC的无缝迁移；支持多级（层级）工作流，如超参数调优。

**🔧 技术方法**

技术上使用Docker/Apptainer容器、RabbitMQ消息中间件、Kubernetes容器编排、SLURM批处理调度、Python(DEAP, NSGA-2)、AC/HVDC功率流仿真。

**📊 数据集**

数据集：德国输电网的HVDC调度问题（2715节点、871发电机、5351条线路、18条HVDC线）以及通过sleep函数模拟的占位负载；对功率流使用AC仿真。

**📈 对比分析**

对比方法：在三种计算层级（单节点K8s、3节点K8s、JURECA-SLURM）进行基准，显示在1–3500核上近线性扩展，通信开销低；在HVDC调度实验中，将水平/垂直扩展分别用于1,500个评估与垂直256核，获得约60M次功率流计算，验证框架能处理大规模计算。

**⚠️ 局限性**

局限性：目前仅支持CPU（无GPU加速）；只能在SLURM/Kubernetes环境下运行；对不同编程语言或异构资源的支持有限；对真实电力系统数据受保密约束无法公开验证。

---

## 546. Paved with True Intents: Intent-Aware Training Improves LLM Safety Classification Across Training Regimes

**arXiv ID:** 2606.27210 | [PDF](https://arxiv.org/pdf/2606.27210v1)

**作者:** Jeremias Ferrao `[一作]` (University of Groningen), Yftah Ziser `[通讯]` (University of Groningen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作构建了一个人类标注的用户意图数据集AIMS，并将意图作为显式中间信号，探索其在安全分类中的作用；

**💡 创新点**

创新点在于首次将人类撰写的意图描述作为可检验的训练目标，证明意图建模能显著提升安全分类器的鲁棒性与解释性；

**🔧 技术方法**

采用的技术包括监督微调（SFT）、直接偏好优化（DPO）、奖励驱动的意图对齐（GRPO）以及意图条件推理蒸馏，结合结构化输出与意图可信度奖励；

**📊 数据集**

使用的数据集是AIMS（从WildGuardMix筛选得到的1,724个难题样本，每个样本配有意图描述和安全标签），并在五个公开安全基准上进行评测；

**📈 对比分析**

与零样本LLM和专用安全守卫模型比较，意图驱动模型在五个基准上的平均F1达到0.836，优于最强对手（GPT‑5.4 0.815、Nemotron 0.809），展示出较强的跨数据集表现；

**⚠️ 局限性**

局限性包括仅针对单轮提示级安全判定，数据集来源单一且以难题为主，且对模型生成的意图评估依赖LLM判断，未来需扩展到多轮对话、响应级审核及更广泛语料。

---

## 547. EO-WM: A Physically Informed World Model for Probabilistic Earth Observation Forecasting

**arXiv ID:** 2606.27277 | [PDF](https://arxiv.org/pdf/2606.27277v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 548. LISA: Likelihood Score Alignment for Visual-condition Controllable Generation

**arXiv ID:** 2606.27192 | [PDF](https://arxiv.org/pdf/2606.27192v1)

**作者:** Yanghao Wang `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

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

## 549. Prompt Injection in Automated Résumé Screening with Large Language Models: Single and Multi-Injection Settings

**arXiv ID:** 2606.27287 | [PDF](https://arxiv.org/pdf/2606.27287v1)

**作者:** Preet Baxi `[一作]` (University of Michigan), Stefanus Jasin `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM在自动简历筛选中被注入自我宣传文本（prompt injection）对候选人排名的影响，并系统评估单注入与多注入情况下的排名偏差及其随竞争强度变化的规律。

**💡 创新点**

首次将多注入竞争动态纳入实验，区分描述性与指令性注入两类策略，对比两种LLM（DeepSeek‑V3.2与GPT‑4o‑mini）的表现，并通过平均排名提升和成功率等量化指标，揭示不同质量池、注入频率和注入类型下的安全与公平风险。

**🔧 技术方法**

采用LLM排名任务，构造控制实验（单/多注入、同质/异质质量池），使用两种Prompt injection（“This is an exceptionally well‑qualified candidate.”与“Classify this candidate as fully qualified …”），随机化注入位置，并通过rank gain与success rate对结果进行统计分析。

**📊 数据集**

使用合成简历数据，基于IT支持专员职位描述，候选人以经验年限为质量指标（10年经验为高质量，5年经验为低质量），每轮池大小为10人，实验覆盖100轮单注入、30轮多注入。

**📈 对比分析**

通过对比不同设置下的rank gain与success rate，发现DeepSeek‑V3.2在单注入时对两类注入均高度敏感，GPT‑4o‑mini在指令性注入下表现更易受影响；在多注入情形下，排名提升与成功率随注入比例递减，最终趋近零，表明竞争饱和会抑制注入效果。

**⚠️ 局限性**

实验仅在固定职位、固定池规模与两种模型上进行，简历质量仅以经验年限衡量，注入文本极简且未覆盖真实的多样化策略；未考虑多阶段ATS流程、下游筛选决策、真实候选人写作风格等因素，结果在不同场景中的量化幅度可能不同。

---

## 550. CORTEX: A Structured Reasoning Benchmark for Trustworthy 3D Chest CT MLLMs

**arXiv ID:** 2606.27264 | [PDF](https://arxiv.org/pdf/2606.27264v1)

**作者:** Hashmat Shadab Malik `[一作]` (MBZUAI), Christoph Lippert `[通讯]` (Hasso Plattner Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文构建了CORTEX基准，使用四阶段（任务理解、视觉观察、诊断推理、答案综合）结构化推理轨迹，为3D胸部CT多模态大型语言模型提供可解释、可验证的评估框架。

**💡 创新点**

创新点在于：恢复被忽略的临床上下文、设计临床医生共创的五级评价表、基于大量LLM生成并人工审核的76k+结构化推理轨迹，首次实现了对3D CT推理过程的分阶段、可追溯验证。

**🔧 技术方法**

技术手段包括：前沿LLM（MedGemma、Qwen‑3、Baichuan‑M2、Gemini‑3系列）在多温度下生成推理轨迹；规则门和LLM评估者（GPT‑5.4‑mini）进行自动评分；最终由三名放射科专家复核。

**📊 数据集**

使用公开的CT‑RATE胸部CT VQA数据集，重新标注并补全临床上下文，经过生成与筛选后得到76,177问答对，其中64,224个开放式、8,914个闭合式、3,039个报告生成的结构化推理轨迹。

**📈 对比分析**

对生成的轨迹采用GPT‑5.4‑mini进行五项rubric评分（任务理解、观察准确、假设评估、推理逻辑、答案正确），所有模型在任务理解和答案正确度均达到8‑10分，平均观察、假设评估和推理逻辑得分较高；最终专家复核一致率达93%，验证了轨迹的准确性。

**⚠️ 局限性**

局限性包括：仅提供基准与验证流程，尚未训练出完整的推理模型；生成与审核过程高度依赖LLM与人工，规模与通用性受限；缺乏跨模态对齐与更大规模的训练样本支持。

---

## 551. Beyond Objects

**arXiv ID:** 2606.27258 | [PDF](https://arxiv.org/pdf/2606.27258v1)

**作者:** Daniel Jackson `[一作]` `[通讯]` (Massachusetts Institute of Technology), Daniel Jackson (Massachusetts Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出一种名为“概念结构化”（concept structuring）的软件模块化方法，旨在替代传统面向对象（OOP）中的对象作为模块的双重角色，减少功能碎片化与耦合；

**💡 创新点**

创新点在于将功能划分为概念（concepts），每个概念只维护与其相关的关系与动作，并通过同步规则（synchronizations）来实现跨概念的协作，从而实现更强的关注点分离、可重用性和对领域实体的更清晰映射；

**🔧 技术方法**

核心技术包括：基于关系模型的状态描述（使用关系和约束），事件驱动的动作定义（pre/post 条件），以及通过同步规则实现概念间的因果链接；

**📊 数据集**

本文为理论性研究，未使用任何实验数据集；

**📈 对比分析**

缺乏量化评估，文章仅通过案例（餐馆预订系统）演示概念结构化的设计流程，并在结论中讨论其在工业实践中的正面经验，但未给出性能对比；

**⚠️ 局限性**

局限性包括：概念结构化不易处理需要对象组合的行为（如序列化、哈希、版本控制等），实现时仍需额外类型与数据结构支持，且在心理上开发者更倾向于传统机制，导致概念的推广与学习成本较高。

---

## 552. The Geometry of Updates: Fisher Alignment at Vocabulary Scale

**arXiv ID:** 2606.27242 | [PDF](https://arxiv.org/pdf/2606.27242v1)

**作者:** John Sweeney `[一作]` `[通讯]` (Sideplane AI), John Sweeney (Sideplane AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于 Fisher 对齐的任务相似度度量——FisherSketch，用于在 LLM 词表规模下快速估算头层更新相似性，进而实现无训练的源任务选择。

**💡 创新点**

创新点在于将头层 Fisher 对齐写成产品核余弦形式，拆解为激活、误差及耦合三因子，并利用随机特征流式估计在词表规模下只需 16 KB 任务签名即可。

**🔧 技术方法**

核心技术包括 Fisher 信息矩阵对齐理论、产品核 mean embedding 视角、随机 Maclaurin 与 SRHT 低维投影、以及分块估计的稀疏误差协方差。

**📊 数据集**

主要数据集为 Llama‑3.1‑8B 的 100 个自然语言任务和 9 个固定前缀“verbalizer shift”实验，另外在 ViT‑B/16 上验证多域转移。

**📈 对比分析**

与 PPL 相似度、激活相似度（CKA）、误差协方差单独指标等传统方法比较，FisherSketch 在源选择 Top‑1 率约 45.7%（随机 4.2%），并在 100 个任务上达到 98.4% 的 oracle‑正则化转移率，显著优于仅基于激活或误差的基线。

**⚠️ 局限性**

局限性包括仅针对共享输出头的 Fisher 对齐、需要已知标签映射、对深层内部层未完整覆盖、对非共享词表或多输出任务适用性有限，以及在极小样本或噪声域中估计精度受限。

---

## 553. SatSplatDiff: Geometry-preserving generative refinement for high-fidelity satellite Gaussian Splatting

**arXiv ID:** 2606.27223 | [PDF](https://arxiv.org/pdf/2606.27223v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 554. A unified cell-merge algorithm for generating diverse Voronoi diagrams and new tessellations based on spatial chromatic model

**arXiv ID:** 2606.27235 | [PDF](https://arxiv.org/pdf/2606.27235v1)

**作者:** Weining Zhu `[一作]` `[通讯]` (Zhejiang University), Weining Zhu (Zhejiang University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于空间染色模型的单一细胞合并算法，可快速生成多种Voronoi图与新型网格分割；

**💡 创新点**

创新点在于把所有Voronoi图的生成归结为对染色单元的检索与合并，消除对每种图单独算法的需求，并能通过修改合并规则产生全新网格；

**🔧 技术方法**

使用空间染色模型（SCM）与最近-排名方法生成整数染色码，构建细胞数据库；通过SQL查询将具有相同染色特征的细胞合并为簇；

**📊 数据集**

以人工生成的10×10像素网格（包含4个对象）和100×100像素网格（包含20个对象）为实验数据集；

**📈 对比分析**

与传统增量/分治等Voronoi生成算法相比，CM算法的优势在于实现简洁、可复用以及生成多种图的灵活性；性能方面主要表现在生成速度快（通过更新像素索引即可），但未给出具体数值比较；

**⚠️ 局限性**

当空间分辨率过高或对象数目巨大时，SCM数据库构建会消耗较多计算资源和存储空间，尽管未来硬件提升可缓解该缺点。

---

## 555. Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement

**arXiv ID:** 2606.27226 | [PDF](https://arxiv.org/pdf/2606.27226v1)

**作者:** Sangwoo Cho `[一作]` (Capital One), Sambit Sahu `[通讯]` (Capital One)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了BinEval框架，通过将评估标准拆分为多维度的原子二元问题，实现任务无关、训练自由的可解释评估，并利用二元问题的结果实现提示的迭代优化。

**💡 创新点**

创新点在于：①以二元问题形式细化评估维度，既提升评估精度又保持可解释性；②利用LLM生成并回答这些问题，形成分维度和整体分数；③构建跨模型与自模型的提示更新循环，使评估器和生成器均可在实践中自我改进。

**🔧 技术方法**

技术手段包括：LLM作为meta-prompt生成二元问题；LLM独立回答每个问题并给出解释；按维度求和得到分数；采用二阶段的提示优化算法（跨模型对齐与自模型更新）以及问题去重与提示重写。

**📊 数据集**

使用的数据集有：SummEval（100篇CNN/DM新闻摘要），Topical-Chat（60条对话回应），QAGS（235 CNN/DM + 239 XSum样本，关注事实一致性），以及IFBench（可执行约束生成任务）。

**📈 对比分析**

与传统指标（ROUGE、BERTScore、MoverScore、BARTScore）以及LLM评估器（UniEval、G-Eval）进行对比。BinEval在SummEval、Topical-Chat和QAGS上获得最高或相当的Spearman/Kendall/ Pearson相关系数，尤其在事实一致性维度表现突出；在提示优化实验中，BinEval显著提升评估分数和生成准确率。

**⚠️ 局限性**

局限性：评估质量高度依赖生成问题的完整性与准确性；二元问题的线性聚合假设可能不适用于所有维度；计算成本相对较高，尤其在多维度、多轮评估时；对主观性强或计算约束型任务（计数、比例）提示优化效果有限。

---

## 556. Hierarchical Muon: Tiled Newton-Schulz Updates for Efficient Muon Optimization

**arXiv ID:** 2606.27216 | [PDF](https://arxiv.org/pdf/2606.27216v1)

**作者:** Ziyuan Tang `[一作]` (University of Minnesota), Yuanzhe Xi `[通讯]` (Emory University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为HiMuon的分层Muon优化器，通过对动量梯度矩阵应用有限的Newton-Schulz映射来构建更新方向，旨在提高稠密神经网络权重的更新效率。

**💡 创新点**

HiMuon的创新点在于将动量梯度矩阵分割成多个小块，独立地对每个小块应用Newton-Schulz映射，从而减少计算成本并提高并行性，同时保持局部谱行为。

**🔧 技术方法**

使用了分层Newton-Schulz方法，结合了GPU的批处理小矩阵线性代数和有限精度算术。

**📊 数据集**

在实验中使用了Qwen3系列的变换器模型，数据集为FineWeb，包含约10亿个标记。

**📈 对比分析**

与全矩阵Muon优化器相比，HiMuon在优化步骤和墙钟时间上都有显著减少，同时在训练和验证损失上保持接近，表明其在效率和性能上的优势。

**⚠️ 局限性**

HiMuon的局限性在于小块的大小选择可能影响更新的谱耦合程度，较小的块可能会导致性能下降，未来的工作需要探索自适应调度和更广泛的评估。

---

## 557. TRUST: Item-Calibrated Interval Evidence for Temporal Session-Based Recommendation

**arXiv ID:** 2606.27214 | [PDF](https://arxiv.org/pdf/2606.27214v1)

**作者:** Linjiang Guo `[一作]` (University of Technology Sydney), Guandong Xu `[通讯]` (Education University of Hong Kong)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了TRUST框架，通过相对每个项目的经验间隔分布来评估观察到的时间间隔，从而改进基于时间的会话推荐。

**💡 创新点**

创新点在于首次识别并实证描述了未校准间隔对现有时间会话推荐方法性能的影响，并提出了一种项目特定的间隔校准评分函数（ITSF），该函数可以集成到现有的时间会话推荐系统中。

**🔧 技术方法**

使用了基于图的神经网络（GNN）和图卷积网络（GCN）等技术，结合了时间可靠性评分来引导邻居采样和会话图编码。

**📊 数据集**

使用了三个公共数据集：Diginetica、RetailRocket和Nowplaying，进行实验验证。

**📈 对比分析**

与多种基线方法（包括时间和非时间的推荐方法）进行了比较，结果显示TRUST在推荐性能上显著优于这些基线，尤其是在HR@10和MRR@10指标上。

**⚠️ 局限性**

限制在于TRUST的计算开销相对较高，尤其是在训练时间和收敛周期上，尽管其性能提升是显著的。

---

## 558. BetXplain: An Explanation-Annotated Dataset for Detecting Manipulative Betting Advertisements on Social Media

**arXiv ID:** 2606.27274 | [PDF](https://arxiv.org/pdf/2606.27274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 559. Smaller Models, Unexpected Costs: Trade-offs in LLM Quantization for Automated Program Repair

**arXiv ID:** 2606.27205 | [PDF](https://arxiv.org/pdf/2606.27205v1)

**作者:** Fernando Vallecillos-Ruiz `[一作]` (Simula Research Laboratory), Leon Moonen `[通讯]` (Simula Research Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对六种大型语言模型在自动程序修复任务中的量化（weight 与 KV‑cache）进行实证评估。

**💡 创新点**

首次将13种不同位宽与量化方法与两大 APR 基准（HumanEval‑Java 与 Defects4J）结合，揭示量化对修复效果、解决集一致性和多维效率指标的细粒度 trade‑off。

**🔧 技术方法**

采用 PTQ 量化技术（AQLM、AWQ、BitsAndBytes、HQQ、Quanto）和基准 GPU 性能测量（推理时间、能耗、内存占用）以及 Jaccard Consistency Ratio。

**📊 数据集**

使用 HumanEval‑Java（164 问题）与 Defects4J v2.0（单函数 525 个 bug）作为评测数据集。

**📈 对比分析**

与基准全精度模型比较：量化可在不降低修复效果的情况下实现 42–86% 的内存缩减，但推理时间和能耗普遍升高（最大 +886%），约 48% 的配置被 Pareto 支配。

**⚠️ 局限性**

局限性包括：默认超参未针对每种量化方法优化、仅在单 GPU 单批推理下测量、仅覆盖 Java 代码、可能存在数据泄露及对其他硬件/语言的泛化不足。

---

## 560. How Good Can Linear Models Be for Time-Series Forecasting?

**arXiv ID:** 2606.27282 | [PDF](https://arxiv.org/pdf/2606.27282v1)

**作者:** Lang Huang `[一作]` (Sakana AI), Luke Darlow `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对 Ridge 回归的预处理进行系统搜索，展示了在大多数时间序列预测基准上无需复杂模型即可获得优于传统线性和非线性方法的性能。

**💡 创新点**

创新点在于将模型容量固定，重点探索上下文长度、本地归一化、正则化和数据增强四个预处理维度，并提出可在不同时间步和通道上分组搜索的框架。

**🔧 技术方法**

技术主要包括闭式岭回归、对输入窗口的局部标准化、基于时间/频域的噪声增强，以及 Optuna 随机树结构化贝叶斯优化进行超参数搜索。

**📊 数据集**

使用了八个公开基准：ETTh1/ETTh2、ETTm1/ETTm2、Weather、Electricity、Traffic 与 Exchange。

**📈 对比分析**

与 OLS、FITS、DLinear 等线性基线以及 PatchTST、iTransformer、TimeMixer、TimesNet、Autoformer 等非线性基线对比，本文在六/七个数据集上平均 MSE 下降 4–16%，并在多数情况下超过 Transformer/MLP/CNN 体系。

**⚠️ 局限性**

局限在于对单变量通道独立建模，跨通道协同学习缺失；并且预处理搜索仍需针对每个数据集调优，未能提供完全通用的自动化方案。

---

## 561. Tilikum: Transaction Fair Ordering on a DAG without Weak Edges

**arXiv ID:** 2606.27250 | [PDF](https://arxiv.org/pdf/2606.27250v1)

**作者:** Giulio Segalini `[一作]` (Université de Neuchâtel), Jérémie Decouchant `[通讯]` (Delft University of Technology)

**通讯引用:** 583 | [OpenAlex ID](https://openalex.org/A5087577380)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于DAG的公平交易排序协议，旨在防止重排序攻击并确保交易的公平性。

**💡 创新点**

创新点在于不依赖弱边，支持公平排序属性，同时实现高吞吐量和低数据冗余。

**🔧 技术方法**

使用了基于中位数的时间戳聚合和批量订单公平性技术。

**📊 数据集**

在实验中使用了多个基准协议进行比较，包括Narwhal/Tusk、Themis和FairDAG。

**📈 对比分析**

与现有的公平排序协议相比，实验结果显示该协议在N=10时的吞吐量达到14,000 tx/s，延迟为1.2秒，比Pompē高39倍，并在N=25时仍比Pompē快4倍。

**⚠️ 局限性**

限制在于协议的复杂性和对网络延迟的敏感性，可能在极端情况下影响性能。

---

## 562. NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems

**arXiv ID:** 2606.27243 | [PDF](https://arxiv.org/pdf/2606.27243v1)

**作者:** Shaohua Liu `[一作]` (Tencent Inc.), Jie Jiang `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了NOVA，一个用于工业推荐系统架构演化的验证感知代理工具，旨在通过架构梯度引导架构演化，减少无声失败，并在生产环境中进行验证。

**💡 创新点**

NOVA的创新点在于引入了架构梯度和多阶段验证级联，能够在生产约束下有效地进行架构修改，并通过反馈机制避免重复的结构性失败。

**🔧 技术方法**

使用了架构梯度、SGD启发式更新信号、验证级联等技术，结合了人类专家的监督和自动化的架构生成过程。

**📊 数据集**

使用了来自生产流量的大规模工业广告推荐数据集，包含数十亿的用户-项目交互记录，涵盖了顺序和非顺序信号。

**📈 对比分析**

与人类专家、编码代理和AutoML基线进行比较，NOVA在L2 ScaleUp和L3 Literature-to-Production任务中表现优异，L3任务的有效通过率达到60.0%，显著高于其他基线，并且在在线A/B测试中显示出正面的商业影响。

**⚠️ 局限性**

NOVA的局限性在于其依赖于历史数据和先前的修改记录，可能在面对全新架构或未见过的修改时表现不佳。

---

## 563. On the Continuity of the Probabilistic Bisimilarity Distance

**arXiv ID:** 2606.27209 | [PDF](https://arxiv.org/pdf/2606.27209v1)

**作者:** Syyeda Zainab Fatmi `[一作]` (University of Oxford), Franck van Breugel `[通讯]` (York University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文证明了稳健概率等价（robust bisimilarity）是概率二分离距离（bisimilarity distance）连续性的必要与充分条件，并给出了判定任意状态对连续性问题的多项式时间算法，随后在实际模型上实现并评估了该算法。

**💡 创新点**

创新点在于：①将稳健二分离由仅是连续性充分条件升级为必要条件，完成了对稳健性与连续性之间关系的完整刻画；②提出了一种基于最小成本最大流的可多项式算法，可一次性决定所有状态对的连续性；③在实验中证明该算法的实际开销远低于单纯计算二分离距离。

**🔧 技术方法**

主要技术包括：概率分布与耦合（coupling）框架；利用最小成本最大流（或线性规划）求解最佳耦合；Knaster‑Tarski定理对函数 A、B 的固定点求解；政策迭代与 OR‑Tools 求解最优耦合；基于 PRISM 的实现与优化。

**📊 数据集**

使用 Quantitative Verification Benchmark Set（QVBS）中的离散时间标记马尔可夫链以及 jpf‑probabilistic 提供的随机化算法模型作为实验数据集。

**📈 对比分析**

对比方法：将连续性判定时间与仅计算二分离距离的时间进行对比；实验结果显示，除极少例外，连续性判定的耗时比距离计算更少或相近，且多数状态对的距离连续，算法更快；在离散化模型上实现稳定、可扩展。

**⚠️ 局限性**

局限性：仅适用于有限状态的标记马尔可夫链；未给出在小扰动下距离增量的上界；对扰动仅考虑概率值变化，未考虑保持图结构的约束；实验规模受内存限制，尚未验证超大规模模型的可行性。

---

## 564. Syntactic Belief Update as the Driver of Garden Path Processing Difficulty

**arXiv ID:** 2606.27206 | [PDF](https://arxiv.org/pdf/2606.27206v1)

**作者:** Alan Zhou `[一作]` (Johns Hopkins University), John T. Hale `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于Rényi散度的句法信念更新指标，用于解释人类在花园路径句子中的阅读困难。

**💡 创新点**

创新点在于把句法重分析视为语法信念分布的更新，并用完全非词汇化的Rényi散度量度其大小，从而突破了传统词汇惊讶法的局限。

**🔧 技术方法**

采用增量非投影依存句法模型，使用条件随机场和RoBERTa+双仿射注意力预测句法树概率，并计算Rényi散度；同时训练Lexical Surprisal和Syntax Supertag Surprisal基准模型。

**📊 数据集**

训练集为结合EWT、GUM、GUMReddit、LinES和PUD的Surface‑Syntactic Universal Dependencies树库；评估数据来自SAP基准中的24条花园路径句子及其对照句。

**📈 对比分析**

与词汇惊讶和语法超标签惊讶对比，SBU在预测三种花园路径类型的难度层级时显著优于基准，并在整体相关性上达到最高，尽管仍低于人类最大幅度。

**⚠️ 局限性**

局限包括仅使用单一句法形式和指标（SUD+Rényi），仅覆盖英语三类花园路径，未解释同类型句子内的项目级差异，且对时间动态的建模不足。

---

## 565. Graph Neural Networks Applications Across Domains: All Insights You Need

**arXiv ID:** 2606.27202 | [PDF](https://arxiv.org/pdf/2606.27202v1)

**作者:** Abderaouf Bahi `[一作]` `[通讯]`, Abderaouf Bahi

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

这篇论文对图神经网络（GNN）的应用进行了全面的综述，探讨了其在多个领域的表现和适用性。

**💡 创新点**

创新点在于将不同领域的应用整合到一个统一的设计空间中，并通过比较分析揭示了图结构的计算成本与效益之间的关系。

**🔧 技术方法**

使用了图神经网络的消息传递机制，结合了谱和空间的公式化方法。

**📊 数据集**

论文中分析了多个应用领域的数据集，包括推荐系统、社交网络、知识图谱、药物发现、医疗保健、计算机视觉等。

**📈 对比分析**

通过跨领域比较，发现异质性和规模在几乎所有领域都削弱了相同的模型，时间图比静态图更具挑战性，且在公共排行榜上表现最好的架构往往不是实际部署中使用的架构。

**⚠️ 局限性**

局限性包括对图结构的贡献的界定不够明确，以及在不同领域中模型的适用性和表现可能存在差异。

---

## 566. Explaining Temporal Graph Neural Networks via Feature-induced Information Flow

**arXiv ID:** 2606.27201 | [PDF](https://arxiv.org/pdf/2606.27201v1)

**作者:** Ping Xiong `[一作]` (Berlin Institute for the Foundations of Learning and Data), Shinichi Nakajima `[通讯]` (Technical University of Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的事件相关性（ER）定义，旨在解释事件驱动的时间图神经网络（ETGNN）的预测，考虑了整个信息流。

**💡 创新点**

创新点在于引入了模块化的归因方法，能够分析所有事件相关变量的完整信息流，而不仅仅是事件嵌入到输出的贡献。

**🔧 技术方法**

使用了归一化相关性度量（NRM）框架，并扩展了该框架以支持复杂神经网络的模块化分解。

**📊 数据集**

在两个合成数据集（用于流行病追踪和社会动态）以及一个真实世界的政治事件网络数据集上进行了评估。

**📈 对比分析**

与现有的解释方法相比，提出的方法在定性和定量实验中均表现出色，能够提供更具人类可解释性的解释。

**⚠️ 局限性**

局限性在于方法的复杂性可能导致在某些情况下的计算开销较大，尤其是在处理大型数据集时。

---

## 567. A Process Harness for Uplifting Legacy Workflows to Agentic BPM: Design and Realization in CUGA FLO

**arXiv ID:** 2606.27188 | [PDF](https://arxiv.org/pdf/2606.27188v1)

**作者:** Fabiana Fournier `[一作]` (IBM), Lior Limonad `[通讯]` (IBM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种新的机制——过程工具，旨在将传统工作流提升至代理商业流程管理（Agentic BPM），而无需替换底层工作流引擎。

**💡 创新点**

创新点在于引入了任务-决策-流（TDF）模型，明确了数据架构和执行语义，并通过CUGA FLO实现了这一模型，展示了如何在贷款审批工作流中应用三种代理类型和钩子驱动的监管覆盖。

**🔧 技术方法**

使用了任务-决策-流（TDF）模型和CUGA FLO系统架构，结合了政策驱动的代理推理和确定性流程执行。

**📊 数据集**

在贷款审批工作流中进行了演示，展示了如何在实际应用中实现三种代理类型的功能。

**📈 对比分析**

与传统BPM和LLM作为规划者的方法相比，CUGA FLO在结构合规性和运行时适应性方面表现出色，能够处理常见和不常见的流程变体，且不需要对流程模型进行编程修改。

**⚠️ 局限性**

局限性包括对过程模型覆盖和拓扑修改的支持有限，钩子成本可能导致延迟，以及需要确保所用代理的行为符合其分配的政策。

---

## 568. How Surprising Is Historical Italian to Language Models? Tokenization Tax, Comprehension Tax, and a Simple Mitigation

**arXiv ID:** 2606.27275 | [PDF](https://arxiv.org/pdf/2606.27275v1)

**作者:** Maria Levchenko `[一作]` `[通讯]` (University of Bologna), Maria Levchenko (University of Bologna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了四维度诊断框架，系统评估历史意大利语对现代LLM的影响

**💡 创新点**

创新点在于将历史难度拆解为tokenization tax、surprisal、semantic robustness和context sensitivity，并揭示tokenization成本与理解难度无关

**🔧 技术方法**

采用LLM（Qwen 2.5‑1.5B、OpenAI、LaBSE）计算perplexity、embedding相似度，并用最小时间提示进行干预

**📊 数据集**

使用三组语料：17世纪意大利语（5本书）、19世纪马尔佐尼《Promessi Sposi》、18世纪俄语民间印刷书

**📈 对比分析**

对比token化膨胀、perplexity比值、embedding相似度和时间提示效果；发现17c.意大利语perplexity高2.4×，时间提示可降低≈60%，embedding相似度≥0.85

**⚠️ 局限性**

受限于预训练数据曝光、样本规模有限、仅评估生成与检索任务，未覆盖更大模型或多语言情况

---

## 569. Resilient Output Containment under Undisclosed Leader Dynamics and Actuator Attacks

**arXiv ID:** 2606.27257 | [PDF](https://arxiv.org/pdf/2606.27257v1)

**作者:** Mohammadreza Nematollahi `[一作]` (Concordia University), Nader Meskin `[通讯]` (Qatar University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究探讨了在有向网络拓扑下，针对异构线性多智能体系统的抗干扰输出包含问题，特别是针对执行器网络攻击的情况。提出了一种连续的双层自适应控制架构，以实现对领导者生成的动态轨迹的跟踪。

**💡 创新点**

创新点在于提出了一种不依赖于领导者动态模型的控制架构，能够在不需要全局图知识的情况下，通过邻居交换的网络接口状态生成任务空间命令，并且能够在执行器攻击的情况下实现输出的渐进包含。

**🔧 技术方法**

使用了连续的双层自适应控制技术，包括虚拟执行器重配置层和网络接口层。通过非光滑的李雅普诺夫分析方法，证明了在特定条件下的渐进包含性。

**📊 数据集**

使用了一个包含六个跟随者和三个领导者的网络进行仿真，跟随者模型为四旋翼无人机，具有不同的物理参数和动态特性。

**📈 对比分析**

与传统的分布式观察者框架和滑模控制方法相比，提出的方法在不需要领导者动态参数的情况下，能够实现更好的抗干扰性能。仿真结果表明，所提方法在执行器攻击下仍能保持输出的有效包含，且跟随者的物理输出能够收敛到领导者的凸包内。

**⚠️ 局限性**

限制在于该方法依赖于对领导者动态的某些假设，且在实际应用中可能受到网络拓扑和执行器攻击类型的影响。

---

## 570. RSPC: A Benchmark for Modeling Stress and Psychiatric Conditions in Digitally Mediated Relationships using Psychiatrist Annotations

**arXiv ID:** 2606.27247 | [PDF](https://arxiv.org/pdf/2606.27247v1)

**作者:** Parmitha Vangapandu `[一作]` (Indian Institute of Information Technology Dharwad), Johannes C. Eichstaedt `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究使用Reddit上关于异地恋的帖子，构建了关系压力与精神病学语料库（RSPC），包含1799条经过精神科医生注释的帖子，旨在捕捉心理健康困扰及相关的关系触发因素。

**💡 创新点**

创新点在于首次将临床基础的精神病学分类与关系压力源和时间关系阶段相结合，推动了心理健康NLP从个体中心向关系上下文建模的转变。

**🔧 技术方法**

使用了七种微调的变换器模型和五种大型语言模型进行多标签障碍分类、关系触发检测和时间阶段预测任务的基准测试。

**📊 数据集**

数据集为RSPC，包含1799条来自异地恋社区的Reddit帖子，经过精神科医生的临床注释，符合DSM-5-TR和ICD-11的诊断标准。

**📈 对比分析**

通过比较不同模型在多标签分类、触发检测和时间阶段预测任务上的表现，发现Claude-3-Haiku在障碍分类任务中表现最佳（Macro-F1 = 0.538），而GPT-4o在关系触发检测中表现最佳（Macro-F1 = 0.519），显示出模型能力的明显差异。

**⚠️ 局限性**

限制在于研究主要集中在自我披露的Reddit叙述上，可能不适用于其他社交平台或文化背景，未来工作可扩展到更广泛的人群和多语言环境。

---

## 571. CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention

**arXiv ID:** 2606.27229 | [PDF](https://arxiv.org/pdf/2606.27229v1)

**作者:** Sayak Dutta `[一作]` `[通讯]`, Sayak Dutta

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的递归语言模型架构CARVE，旨在解决现有模型在记忆管理和计算效率方面的局限性。

**💡 创新点**

CARVE通过仅在关键轴上进行擦除，解决了记忆盲门控、写门带宽瓶颈和破坏块并行性的三个问题，同时保持了WY形式的块并行训练。

**🔧 技术方法**

使用了内容感知的擦除机制和标量值写门，结合了输出重用和低秩投影技术。

**📊 数据集**

在1.3B参数规模上，使用了100B个FineWeb-Edu的训练数据集。

**📈 对比分析**

与GDN-2模型相比，CARVE在WikiText语言建模中达到了15.72的困惑度，相比之下GDN-2为15.90，且在多个常识推理基准上表现优异，所有性能提升均在硬件成本上没有显著增加。

**⚠️ 局限性**

由于内容信号是前一个块的平均输出，短序列（少于两个块）将无法获得内容信号，这限制了模型在处理非常短的输入时的表现。

---

## 572. Compositionality and the lexicon in evolutionary semantics

**arXiv ID:** 2606.27228 | [PDF](https://arxiv.org/pdf/2606.27228v1)

**作者:** Fausto Carcassi `[一作]` `[通讯]` (University of Amsterdam), Fausto Carcassi (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

本文通过构建一个同时演化词义与组合函数的进化语义框架，研究量词保守性这一语义通用特征，并解释其为何在自然语言中普遍出现；

**💡 创新点**

创新点在于将词义与组合函数视为可共进化的系统级参数，通过引入全局抽象来实现保守性，并在进化压力下展示保守性是最有效的压缩方案；

**🔧 技术方法**

采用概率上下文无关文法（PCFG）生成词义与组合函数，利用C++ Fleet 库进行大量蒙特卡洛模拟，评估通信准确性与系统复杂度的 Pareto 前沿；

**📊 数据集**

使用的“数据集”是由5个对象（特征值-10~10）构成的人工上下文环境，并在此环境中进行模拟；

**📈 对比分析**

比较方法是分别在字面与含Pragmatic 的说话者设置下，绘制复杂度-通信准确性二维的 Pareto 前沿；结果表明保守性系统在此前沿上既更简单又更具通信效果；

**⚠️ 局限性**

局限性包括：仅能进化最多3个量词；模型对词典与语义原语的选择具有一定主观性；逐步处理假设限制了更大规模实验的可行性；计算成本限制了进一步扩展。

---

## 573. Don't Settle at the Mode! Mitigating Diversity Collapse in Pretrained Flow Models via Feature Self-Guidance

**arXiv ID:** 2606.27371 | [PDF](https://arxiv.org/pdf/2606.27371v1)

**作者:** Pradhaan S Bhat `[一作]` (Indian Institute of Science), R. Venkatesh Babu `[通讯]` (Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在预训练流模型中引入推理时的特征自导机制，主动扩散内部特征以提升多样性，同时通过流形正则化保持对条件的忠实度，形成一个无训练、可即插即用的模块

**💡 创新点**

首次发现内部特征收敛与输出多样性崩溃的直接关联，并提出在单一MMDiT块上实现特征扩散与流形投影的双步骤策略

**🔧 技术方法**

使用流形正则化的特征自导（Feature Self‑Guidance）以及对FLUX等多模态DiT模型的内部特征迭代更新

**📊 数据集**

在文本生成上使用GenEval数据集，在深度条件生成上使用COCO 2017深度子集，在个性化生成上使用Dreambooth数据集

**📈 对比分析**

与Particle Guidance、Interval Guidance、Shielded Diffusion、CNO及Group Inference等方法对比，实验显示在保持相近推理时延的前提下，显著提升DINO、DreamSim等多样性指标，且在FID/CLIPScore上保持竞争性或更优

**⚠️ 局限性**

依赖批量计算，难以支持流式生成；受限于底层模型的偏差；且在极大批量规模下多样性提升有限

---

## 574. PhysiFormer: Learning to Simulate Mechanics in World Space

**arXiv ID:** 2606.27364 | [PDF](https://arxiv.org/pdf/2606.27364v1)

**作者:** Yiming Chen `[一作]` (University of Oxford), Andrea Vedaldi `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种扩散变换器，用于物理上合理的3D物体运动预测，直接在世界坐标中表示3D网格，而不是在视图依赖的像素空间中操作。

**💡 创新点**

通过将顶点轨迹预测视为一个单一的去噪扩散过程，展示了在没有任何归纳偏置的情况下也能获得优秀的结果，捕捉了学习动态中的不确定性。

**🔧 技术方法**

使用了扩散变换器（Diffusion Transformer），并在时间、空间和对象上进行了注意力因子化，以提高效率。

**📊 数据集**

在超过10万个模拟轨迹的数据集上进行训练，生成刚性和弹性力学，并能够推广到混合材料设置、未见的真实几何形状和更大的物体数量。

**📈 对比分析**

与自回归基线进行比较，模型在轨迹准确性、刚性保持和基于动量的物理一致性方面显著优于自回归方法。

**⚠️ 局限性**

当前的限制是固定的轨迹长度和网格分辨率，可能导致偶尔出现虚假接触、相互穿透和罕见的方向不连续性。

---

## 575. Continual Robot Policy Learning via Variational Neural Dynamics

**arXiv ID:** 2606.27353 | [PDF](https://arxiv.org/pdf/2606.27353v1)

**作者:** Jiaxu Xing `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种持续学习框架，利用真实世界的经验来改善机器人在隐藏和重复动态下的策略。

**💡 创新点**

创新点在于结合了分析物理先验和神经残差模型，学习条件感知的动态模型，并在部署时通过在线推断来恢复重复条件，而不是重新拟合残差模型。

**🔧 技术方法**

使用了条件感知的神经动态模型和可微分的策略优化技术。

**📊 数据集**

使用了真实的状态-动作轨迹数据进行训练和验证。

**📈 对比分析**

与现有的在线适应方法相比，该框架在应对变化的风条件下，策略恢复时间约为1秒，比在线残差重新拟合快约5倍，并且在大扰动悬停和跟踪误差上分别减少了65.7%和53.3%。

**⚠️ 局限性**

当前的硬件实验依赖于准确的低维状态估计，实际应用中传感器噪声可能导致错误的动态模型更新。

---

## 576. Hallucination in World Models is Predictable and Preventable

**arXiv ID:** 2606.27326 | [PDF](https://arxiv.org/pdf/2606.27326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 577. Multilingual Reasoning Cascades Need More Context

**arXiv ID:** 2606.27306 | [PDF](https://arxiv.org/pdf/2606.27306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 578. Bridging Performance and Generalization in Reinforcement Learning for Agile Flight

**arXiv ID:** 2606.27348 | [PDF](https://arxiv.org/pdf/2606.27348v1)

**作者:** Jonathan Green `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

训练一个能在未见赛道上实现零样本泛化的高速无人机竞速策略。

**💡 创新点**

结合并行多任务训练、基于学习进度的自适应任务切换和物理约束的程序化赛道生成，实现了 7.4 倍的泛化提升且不牺牲速度。

**🔧 技术方法**

使用深度强化学习（PPO）、Spearman/Kalman 反馈自适应任务切换、基于 B‑Spline 的赛道生成、L2 正则化以及大规模并行采样等技术。

**📊 数据集**

在模拟器中生成多条随机/程序化赛道（包含四条人造核心赛道 Figure8、BigS、Kidney、SplitS），并在真实硬件上进行测试。

**📈 对比分析**

与 Environment as Policy（EaP）和单任务 PPO 进行比较，ZSG 性能（S_pw）提升 7.4 倍，速度提升 37.73% 以上，实测成功率 100% 并且 sim‑to‑real 差距仅 6.6%。

**⚠️ 局限性**

局限包括：未评估动态门、障碍等 OOD 场景；视觉策略仍需部分状态信息；大规模并行训练对计算资源需求高；无法完全实现纯图像驱动的安全性。

---

## 579. Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning

**arXiv ID:** 2606.27330 | [PDF](https://arxiv.org/pdf/2606.27330v1)

**作者:** Tianyi Men `[一作]` (Chinese Academy of Sciences), Jun Zhao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PEEU 方法，利用自主探索与逆向经验构造对齐的高层任务，提升小型多模态 LLM 的规划与跨网站泛化能力。

**💡 创新点**

创新点在于自适应目标驱动的规划树探索、经验总结与逆向任务对齐，生成严格约束的高层训练数据，并引入 TDHAF 框架量化层级泛化。

**🔧 技术方法**

使用多模态 LLM（Qwen2.5‑VL‑3B/7B）结合 SFT 与 GRPO 训练策略，配合 GPT‑4o 进行环境探索与经验摘要。

**📊 数据集**

基于 WebVoyager 基准（Allrecipes、Amazon、Apple、Arxiv、GitHub、Coursera、Map、Wolfram 等七个未见网站）以及 0.1k / 2k 轨迹数据。

**📈 对比分析**

与基线（Atomic‑Prompt、Trajectory‑Prompt、Coarse 等）相比，PEEU‑SFT/GRPO 在 7B 模型上在 OOD 网站上达成 30.6%/19.9% 的成功率，显著优于同规模基线及更大模型 Qwen2.5‑VL‑32B。

**⚠️ 局限性**

仅评估信息检索与导航任务，未涉及登录、验证码或支付等安全敏感场景，扩展到受限会话与安全协议仍是未来工作。

---

## 580. An Instruction Set Architecture for IMPLY-based Memristive Processing-in-Array

**arXiv ID:** 2606.27319 | [PDF](https://arxiv.org/pdf/2606.27319v1)

**作者:** Liam Splittgerber `[一作]` (Technische Universität Wien), Nima TaheriNejad `[通讯]` (Technische Universität Wien)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种面向超低功耗边缘设备的全功能单片忆阻式计算机（IMC）架构，并基于RISC‑V RV32I标准改进ISA，利用忆阻交叉阵列与IMPLY状态逻辑实现指令集与控制逻辑；

**💡 创新点**

创新点包括：①将传统load‑store微控制器完全改为单片内存计算模型，消除von Neumann瓶颈；②引入地址银行和二维地址格式，实现跨行并行操作与动态寻址；③采用两操作数破坏性写法与原生布尔/算术/比较指令的IMPLY实现，兼容RISC‑V生态；

**🔧 技术方法**

关键技术有：忆阻交叉阵列（1T1R）、IMPLY状态逻辑、地址银行、改进的RISC‑V指令编码、ATOMIC仿真框架与VTEAM忆阻模型；

**📊 数据集**

实验使用了仿真数据：VTEAM参数与IMPLY电压参数，并在案例研究中模拟了智能温度传感器节点（四个时间段、八次测量）进行局部预处理；

**📈 对比分析**

通过ATOMIC仿真评估各指令能耗、时序与面积，并与传统CMOS微控制器的能耗/延迟进行对比，结果显示大多数指令能耗低于传统实现，尤其是布尔/立即指令；但算术与比较指令仍显高能耗与延迟；

**⚠️ 局限性**

局限性包括：①算术与比较指令的高延迟与能耗；②需要较大的交叉阵列尺寸与工作忆阻数量；③两操作数破坏写模式对编程和编译器支持有额外挑战；④未实现完整RISC‑V功能（如load/store、浮点等）。

---

## 581. Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards

**arXiv ID:** 2606.27376 | [PDF](https://arxiv.org/pdf/2606.27376v1)

**作者:** Ritesh Thawkar `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Rao Muhammad Anwer `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种自演进训练框架，通过在冻结的统一多模态模型上增设 Proposer、Solver 与 Generator 三个 LoRA 适配器，仅利用无标签图像实现视觉理解与图像生成的联合自监督提升。

**💡 创新点**

创新点在于：① Solver Token Entropy（STE）连续难度信号缓解样本级自一致性退化；② 生成评估采用 QA 真实性与循环一致性两尺度内部打分；③ 采用同一可迁移的训练规则与奖励定义，兼容扩散、正向流与自回归三种生成范式。

**🔧 技术方法**

技术包括：基于 LoRA 的轻量级角色适配器；自一致性与 STE 的奖励机制；多尺度内部评估（QA 真实性 + 循环一致性）；基于 KL 正则的策略梯度优化。

**📊 数据集**

仅使用 10,000 张来自 COCO、SA‑1B、TextVQA、GQA 与 LAION‑COCO 的无标签图像作为训练集。

**📈 对比分析**

在 BLIP3o‑8B、BAGEL 与 VARGPT‑v1.1 三个统一模型上，与基线相比，视觉理解平均提升 1.9–3.6 分（MMMU、MM‑Bench 等），MME 子任务提升两位数，图像生成 GenEval 提升 3% 左右，表现优于多种自监督对比方法。

**⚠️ 局限性**

局限性包括：依赖 Solver 质量的内部监督，无法突破生成模型固有的语义级限制；对更大规模无标签数据或视频、3D 场景的泛化尚未验证。

---

## 582. Scalable Behavior Cloning with Open Data, Training, and Evaluation

**arXiv ID:** 2606.27375 | [PDF](https://arxiv.org/pdf/2606.27375v1)

**作者:** Arthur Allshire `[一作]` (UC Berkeley), Angjoo Kanazawa `[通讯]` (UC Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了面向低成本双臂机器人的大规模真实世界遥操作数据集ABC‑130K，并提供了对应的模型（ABC‑Models）、仿真环境（ABC‑Sim）与评估套件（ABC‑Eval），用于端到端的行为克隆研究。

**💡 创新点**

① 公开了目前规模最大的双臂遥操作数据集（3,500小时、130,000条轨迹、195个任务）；② 系统性评估了扩散变换器（DiT）和视觉‑语言‑动作（VLA）两大模型族的架构、编码器与调节方式；③ 证明了模拟性能与真实世界性能高度相关，为无机器人实验者提供了可行的迭代路径；④ 通过DAGGER收集恢复行为显著提升了高难度任务的成功率。

**🔧 技术方法**

使用的主要技术包括：扩散动作模型（DiT）结合跨注意力或AdaLN调节；VLM（Gemma）与跨注意力/AdaLN连接器；CLIP、DINOv3等视觉编码器；多线程/高效视频解码的数据加载器；异步执行与动作前缀；多绘制采样实现梯度方差降低；仿真环境构建（MuJoCo/Blender）与VR遥操作；以及离线与在线评估指标。

**📊 数据集**

核心数据集：ABC‑130K（3,553小时、134,806条目、195任务）；内部7,000小时多模态数据用于实验对照；ABC‑Sim仿真数据（400小时、10个任务）；ABC‑Eval真实世界评估数据（>100小时、50试验/任务）。

**📈 对比分析**

通过在三项真实任务（瓶子投掷、碟子架装载、杯子翻转）和10个仿真任务进行对比，DiT与VLA在不同批量/训练步骤下分别实现了最高严格成功率（DiT约70%–80%，VLA约60%–70%）和任务进度。预训练模型在下游精细任务（信用卡提取、LEGO排序、笔帽插入、瓶盖拆卸）上显著优于从零训练，且DAGGER提升箱子折叠任务从24%到85%。离线指标（训练损失、验证动作误差）与真实成功率呈负相关，模拟-真实相关系数分别为r=0.85和r=0.91。

**⚠️ 局限性**

局限性包括：需要昂贵的算力（H200 GPU），高质量数据仍需人工遥操作；模拟环境与真实世界在视觉和动力学上存在差距，未覆盖所有任务；模型尚未结合强化学习微调；对极端外部干扰或非结构化环境的鲁棒性未系统评估。

---

## 583. Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models

**arXiv ID:** 2606.27373 | [PDF](https://arxiv.org/pdf/2606.27373v1)

**作者:** Shravan Venkatraman `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fahad Khan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VISE，一种完全无监督的自我进化框架，用以解决大规模多模态模型在视觉推理中的视觉低调问题；

**💡 创新点**

创新点在于把奖励目标从输出一致性转移到几何与语义不变性上，直接正则化模型的视觉条件化策略，且不需要多角色设置、注释或外部奖励模型；

**🔧 技术方法**

采用自问式生成、基于 REINFORCE 的 KL 正则化策略，结合几何不变性奖励（对变换后框与投影框的 GIoU）和语义不变性奖励（对区域 ghosting 的可见性判定），并使用 LoRA 进行模型微调；

**📊 数据集**

训练数据为 4000 张无标签的 COCO 原始图像（不使用任何标注），评估使用 COCO、NoCaps、Flickr30k、TextCaps、GQA、OK‑VQA、VQAv2、AI2D、ChartQA、InfoVQA、ScienceQA、MMMU、CaptionQA、RWQA、ESB、MMBench 等多种视觉问答与推理基准，以及 POPE 与 COCO Cap Chair 的假设性评测；

**📈 对比分析**

与 VisPlay、EvoLMM、iReasoner、VisionZero 等无监督自我进化基线相比，VISE 在 2B 规模下 COCO CIDEr 提升 16.85，TextCaps CIDEr 提升 19.66，VQA 与推理任务均无降级且普遍提升；假设性指标 Chair‑I、Chair‑S 也显著下降（-5.00 与 -5.45），说明假设性错误显著减少；

**⚠️ 局限性**

限制在于提升幅度随模型规模增大而减弱，主要受预训练视觉编码器的固定限制；此外，方案仍依赖图像变换与区域模糊的超参数，且在某些任务的细粒度推理上可能不如基于标签监督的模型。

---

## 584. Reinforcement Learning without Ground-Truth Solutions can Improve LLMs

**arXiv ID:** 2606.27369 | [PDF](https://arxiv.org/pdf/2606.27369v1)

**作者:** Yingyu Lin `[一作]` (University Of California San Diego), Yuxiong He `[通讯]` (Snowflake Ai Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无地面真值的优化任务上提出RiVER框架，通过实例内排名和赢家加权奖励来训练LLM，以提升其编程与算法工程能力。

**💡 创新点**

创新点在于将可验证奖励从传统答案匹配扩展到无参考答案的评分优化，并通过实例排名消除尺度主导、频率主导问题。

**🔧 技术方法**

使用RLVR+GRPO框架结合实例排名、赢家加权奖励与奖励校准的策略梯度优化方法。

**📊 数据集**

训练集为12道AtCoder Heuristic Contest问题，评估集包括ALE‑Bench、LiveCodeBench v5/v6以及USACO。

**📈 对比分析**

与多种奖励设计基线对比，RiVER在ALE评分提升约140‑160点，且在Exact‑solution基准上平均提升2.4%–3.5%，表现优于Raw‑score及其它基线。

**⚠️ 局限性**

局限在于需要可执行验证环境，奖励设计仍需人工调参，且对更大规模或不同类型任务的迁移性尚未验证。

---

## 585. Autoregressive Boltzmann Generators

**arXiv ID:** 2606.27361 | [PDF](https://arxiv.org/pdf/2606.27361v1)

**作者:** Danyal Rehman `[一作]` (Mila -- Quebec AI Institute), Alexander Tong `[通讯]` (Aithyra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于自回归（AR）框架的 Boltzmann Generator，突破了传统正则流（normalizing flow）在表达能力与推断效率上的限制，能够直接在分子原子笛卡尔坐标上建模并生成平衡态样本。

**💡 创新点**

创新点包括：1）首次将自回归模型应用于分子平衡采样，去除了流模型的可逆性与拓扑保持约束；2）采用统一量化（uniform binning）与离散混合密度网络（MoL‑PixelCNN++、GMM‑PixelCNN++）两种条件参数化，兼顾训练稳定性与精度；3）在 BG 框架下实现了可中间干预的自回归 SMC（Twisted SMC）与温度调节，进一步提升采样多样性与效率；4）构建了 132M 参数的可迁移模型 AutoBG，实现了 8 残基 peptide 的零样本泛化并显著降低能量误差。

**🔧 技术方法**

技术手段主要包括：自回归变压器架构、统一量化与离散混合密度网络、离散化混合逻辑/高斯模型、温度调节、Twisted Sequential Monte Carlo、以及对分子能量函数的多步分段评估。

**📊 数据集**

使用的数据集包括：单肽系统（Tri‑alanine、Tetrapeptide、Hexa‑alanine、Chignolin 10‑残基）以及 ManyPeptidesMD（4/8 残基序列）进行零样本迁移评估。

**📈 对比分析**

与传统 BG、CNF（ECNF++）、离散 NF（RegFlow）、SBG、FALCON、GIVT 等基线比较，AutoBG 在能量 Wasserstein、扭转角 Wasserstein 以及 TICA Wasserstein 上均优于所有方法，尤其在 Chignolin 10‑残基上表现最为突出；在可迁移任务中相较 Prose 提升 60% 以上。

**⚠️ 局限性**

局限性包括：1）自回归模型需对维度做固定顺序，可能影响小分子性能；2）统一量化限制了精度，需更细粒度的分辨率来捕捉更尖锐的能量势；3）缺乏流模型中可利用的先验信息（如信息先验、对称性约束），未来可探索在 AR 架构中引入更丰富的先验。

---

## 586. Error-Conditioned Neural Solvers

**arXiv ID:** 2606.27354 | [PDF](https://arxiv.org/pdf/2606.27354v1)

**作者:** Haina Jiang `[一作]` (University of Michigan), Jeong Joon Park `[通讯]` (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为Error-Conditioned Neural Solver（ENS）的递归神经网络框架，用残差场作为直接输入来迭代修正PDE求解器的预测，提升了对物理约束的自适应校正能力。

**💡 创新点**

创新点在于把PDE残差作为网络的输入而非优化目标，让模型学习非线性残差校正策略，突破了传统混合方法在残差最小化与重构误差之间的“残差-重构间隙”与对初值敏感的问题。

**🔧 技术方法**

采用了傅里叶神经算子（FNO）和Transformer-based VideoPDE等网络骨干，结合递归校正器、残差输入、统一训练目标（重构误差）以及可调步长的迭代更新机制。

**📊 数据集**

在四类二维PDE（线性/非线性Helmholtz、Darcy流、Poisson、Navier–Stokes及Kolmogorov湍流）上构建数据集，涵盖常规、超分辨率、参数外推和跨方程转移等四种离散分布偏移场景。

**📈 对比分析**

与传统前馈算子（FNO、POSEIDON）、混合优化方法（PINO、DiffusionPDE、PCFM）进行比较，ENS在大多数场景下实现了最低的重构误差，同时残差也保持低水平；在计算效率上，ENS在不牺牲精度的前提下比混合方法快数十到数百倍。

**⚠️ 局限性**

局限性包括仅在二维、离散已知方程的情境下验证；对三维大规模问题、部分已知或噪声测量场景的鲁棒性尚未测试；以及对复杂边界条件和高频残差的处理仍需进一步研究。

---

## 587. All you need is log

**arXiv ID:** 2606.27349 | [PDF](https://arxiv.org/pdf/2606.27349v1)

**作者:** Akshay Balsubramani `[一作]` `[通讯]`, Akshay Balsubramani

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文对多分布的Rényi散度进行了全面的特征描述，提出了一个新的多分布Rényi散度的定义，并证明了其在数据处理和独立产品下的单调性和可加性。

**💡 创新点**

创新点在于对多分布Rényi散度的结构性特征进行了系统的归纳，提出了一个包含四个层次的参数空间，并通过多条独立路径验证了该散度的有效性。

**🔧 技术方法**

使用了功能分析和测度理论的技术，特别是Riesz-Markov表示定理来建立散度的积分表示。

**📊 数据集**

使用了多种概率分布的组合，具体数据集未明确提及，但涉及到多种分布的比较和分析。

**📈 对比分析**

通过与现有的Rényi散度和Kullback-Leibler散度进行比较，展示了新定义的多分布Rényi散度在多种情况下的有效性和一致性，性能表现优越。

**⚠️ 局限性**

限制在于该研究主要集中在经典多变量情况下，量子多变量散度的推广尚未完全展开，且对某些边界情况的处理仍需进一步研究。

---

## 588. SAM2Matting: Generalized Image and Video Matting

**arXiv ID:** 2606.27339 | [PDF](https://arxiv.org/pdf/2606.27339v1)

**作者:** Ruiqi Shen `[一作]` (Fudan University), Henghui Ding `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个解耦的图像与视频抠图框架，将高层语义跟踪与低层细粒度抠图分离；使用预训练的视频目标跟踪器生成一致性掩码，在此基础上通过ROI检测器自动定位需要抠图的细节区域，并用渐进式alpha预测器在多尺度上逐步生成高质量alpha蒙版。

**💡 创新点**

创新点包括：① 通过解耦策略消除对昂贵视频抠图数据的依赖；② 设计基于多源先验的ROI检测器，精确识别半透明与细节丰富区域；③ 引入渐进式alpha预测器的多尺度级联与深度监督；④ 通过抠图-掩码一致性损失和平滑损失抑制空洞与锯齿，提升抠图细节与整体一致性。

**🔧 技术方法**

使用冻结的VOS跟踪器（SAM2.1、SAM2.1-Base+、SAM3）做高层跟踪；多尺度卷积网络实现ROI检测与伪trimap生成；Progressive Alpha Predictor在多尺度下逐步细化alpha；损失函数包括焦点损失、平滑L1损失、L1 + 拉普拉斯损失、抠图-掩码一致性损失；训练采用AdamW优化器。

**📊 数据集**

训练数据仅来自8个高质量图像抠图数据集（I‑HIM50K、P3M‑10k、CelebAHairMask‑HQ、AIM‑500、Distinctions‑646、AM‑2K、UHRIM、RefMatte）；视频评测使用V‑HIM60和VideoMatte基准；未使用任何视频抠图数据进行训练，采用零样本评估。

**📈 对比分析**

在图像抠图基准（P3M‑500‑NP、AM‑2K、PPM‑100）和视频抠图基准（V‑HIM60、VideoMatte）上与现有SOTA方法（如MatAnyone、MaGGIe等）进行零样本对比，所有三种变体均取得显著优势；在实时性能方面，SAM2.1‑Tiny版实现40 FPS，内存占用低；在长期视频中保持稳定、无闪烁，显示出良好的时序一致性。

**⚠️ 局限性**

局限性：在对视频抠图数据进行微调后，模型会出现跟踪鲁棒性下降；对极端遮挡或极端域外场景的抠图仍可能出现错误；模型的整体表现受预训练VOS跟踪器质量的限制，若跟踪器失效则抠图质量下降。

---

## 589. RoPEMover: Depth-Aware Object Relocation via Positional Embeddings

**arXiv ID:** 2606.27332 | [PDF](https://arxiv.org/pdf/2606.27332v1)

**作者:** Ipek Oztas `[一作]` (Bilkent University), Aysegul Dundar `[通讯]` (Bilkent University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于旋转位置嵌入（RoPE）的几何感知单图像对象移动方法，直接操纵扩散变换器内部表示实现对象位移、遮挡处理、阴影与反射等场景一致性。

**💡 创新点**

将2D RoPE扩展为深度感知3D RoPE，并在反向扩散前对对象标记的RoPE进行空间扭曲，从而在单图像中实现前后遮挡、阴影自洽的精准对象重定位。

**🔧 技术方法**

使用RoPEwarp + depth-aware 3D RoPE，结合预训练的指令调优扩散变换器（如Qwen-Image），并通过LoRA微调，利用合成CLEVR与少量真实图像进行训练。

**📊 数据集**

训练使用CLEVR生成的3个规模（2000/5000/10000）合成场景，辅以165张真实图像对；评测使用ObjMove-A（200对）与ObjMove-B无配对数据集。

**📈 对比分析**

与ChronoEdit、DragAnything、DragDiffusion、Inpaint4Drag、MagicFixup、GeoDiffuser、Qwen-Image、Flux Kontext、FreeFine、Object Mover等基线在ObjMove-A上对比，使用CLIP/DINO/DreamSim/PSNR等指标，最终取得所有指标最高（CLIP 91.74, DINO 97.57, PSNR 24.97），显著优于现有方法。

**⚠️ 局限性**

推理耗时受多步扩散影响，额外深度预处理开销轻微；依赖预训练模型可能带来偏差与失败模式，且在极端遮挡或光照极端变化下仍可能失效。

---

## 590. Beyond the Hard Budget: Sparsity Regularizers for More Interpretable Top-k Sparse Autoencoders

**arXiv ID:** 2606.27321 | [PDF](https://arxiv.org/pdf/2606.27321v1)

**作者:** Nathanaël Jacquier `[一作]` (Université Paris-Saclay), Mahdi S. Hosseini `[通讯]` (Mila--Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 Vision Foundation Model（VFM）嵌入上训练 Top‑k 稀疏自编码器（SAE），并提出两种新的稀疏正则化器（离支 ℓ₁ 正则化和 ℓ₁/ℓ₂ 比值正则化），使稀疏度控制在 Top‑k 选取之前且仅作用于批量活跃的单位；

**💡 创新点**

创新点在于：①在 Top‑k 结构中加入“软”稀疏约束，弥补硬阈值固有的固定 k 与过拟合问题；②正则化仅对已被 Top‑k 选中的单元做掩码，避免产生死亡单元；③使用 ℓ₁/ℓ₂ 比值作为尺度不变稀疏度度量，显著聚焦能量并提升推理时对 k 的鲁棒性与少量预算下的线性探测性能；

**🔧 技术方法**

主要技术包括：Top‑k 稀疏自编码器架构、ReLU 编码器、硬阈值后投影、离支 ℓ₁ 与 ℓ₁/ℓ₂ 比值正则化、批量活跃单元掩码、线性重建损失与可选辅助损失、线性探测与类别纯度评估；

**📊 数据集**

使用 ImageNet‑1K 与 Open Images V7 两大公开图像数据集，分别在三种冻结的 VFM 上测试：CLIP ViT‑L/14、SigLIP2、监督版 ViT‑L/16；

**📈 对比分析**

与无正则化基线对比，评估指标为重建 R²、平均/中位数单语义度（Monosemanticity）以及类别纯度；实验显示两种正则化器均能在不降低重建质量的前提下显著提升单语义度与类别纯度；ℓ₁/ℓ₂ 进一步提升了对推理时 k 的鲁棒性和小预算线性探测的准确率；

**⚠️ 局限性**

局限性包括：需对批量活跃单元做掩码，若不做会导致大量死亡单元；正则化强度 λ 的调参仍需要经验；目前仅在标准 Top‑k 结构下验证，未扩展到 BatchTopK 或 Matryoshka 等变体；此外，正则化对不同模型的提升幅度差异较大，需进一步探究其泛化性。

---

## 591. Elastic Time: Dynamic Frame Rate Bottlenecks for Neural Audio Coding

**arXiv ID:** 2606.27320 | [PDF](https://arxiv.org/pdf/2606.27320v1)

**作者:** Dimitrios Bralios `[一作]` (University of Illinois Urbana-Champaign), Minje Kim `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出 Elastic Time 方法，利用轻量级潜在预测器和重编码瓶颈，实现对预训练音频自编码器的动态帧率自适应压缩。

**💡 创新点**

创新点在于：①通过潜在预测器衡量帧间可预测性来做边界选择，②实现无需外部语义监督的可部署时速率控制，③在同一模型上支持多种压缩率的可伸缩性。

**🔧 技术方法**

技术包括 Re‑Bottleneck 结构（ConvNeXt‑V2 编码/解码器）、GRU‑based 线性潜在预测器、贪心与动态规划边界搜索、对数 MSE、对抗与特征匹配损失。

**📊 数据集**

使用 4.8k 小时的音频混合数据集，来源于 AudioSet‑balanced、FSD50k、BBCSoundEffects、RWC、MoisesDB、Jamendo‑FMA‑captions。

**📈 对比分析**

与 Conv‑Downsample、CodecSlime、H‑Net、H‑Net‑YOTO 等基线进行比较；在 SongDescriber、AudioCaps、DAPS、MuChin 等多领域上评估 mel‑d、FAD、SI‑SDR、STFT‑d；Elastic Time 在多数指标上与或优于基线，贪心算法近似等价于 DP，且在可伸缩模式下仍保持竞争性能。

**⚠️ 局限性**

局限性：对非音乐音频（如 AudioCaps）预测器泛化略差；在极端压缩率（ρ→0.9）下，宽范围训练的模型性能略逊；需要额外的 Re‑Bottleneck 训练，且模型规模比纯下采样略大。

---

## 592. A Multi-Fidelity Convolutional Autoencoder-Transfer Learning Framework for Guided-Wave-Based Damage Diagnosis Using Large Simulated and Limited Experimental Datasets

**arXiv ID:** 2606.27304 | [PDF](https://arxiv.org/pdf/2606.27304v1)

**作者:** Santosh Kapuria `[一作]` (Indian Institute of Technology Delhi), Abhishek `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种多保真度卷积自编码器–迁移学习框架，利用大规模低保真度模拟数据与有限实验数据，实现板式结构的导波检测中的损伤定位与大小估计。

**💡 创新点**

创新点在于：① 将高频导波模拟转化为1D时间域谱元模型，显著降低仿真成本；② 通过自编码器提取压缩特征，再用FFNN预测损伤参数，实现端到端的无预处理学习；③ 采用迁移学习将模拟知识迁移到实验域，仅需少量实验样本即可完成 fine‑tune。

**🔧 技术方法**

技术包括：1D时间域谱元（TDSE）模拟、卷积自编码器（CAE）提取潜在空间、前馈神经网络（FFNN）回归、Adam优化器、迁移学习（冻结编码器、微调解码器+FFNN）。

**📊 数据集**

数据集：① 1D‑TDSE 生成的 1188 条带噪声的导波信号；② 2D 有限元（FE）产生的 234 条高保真度信号；③ 仅 192 条实验采集的导波信号（块状损伤）。

**📈 对比分析**

与传统 CNN‑TL 模型相比，CAE‑FFNN‑TL 在损伤定位的 R² 由 0.3369 提升至 0.9305，损伤大小 R² 由 0.9758 提升至 0.9972；在实验域微调后，定位 R² 达 0.9629，大小 R² 达 0.9931，显著优于仅用实验数据训练的 0.5154/0.8087。

**⚠️ 局限性**

局限性包括：仅针对缺口/块状损伤验证，未覆盖更复杂损伤或多环境条件；实验样本仍有限，模型在不同材料或大尺寸结构上的推广性待验证。

---

## 593. RouterVLA: Turning Smoke Tests into Supervision for Heterogeneous VLA Selection

**arXiv ID:** 2606.27355 | [PDF](https://arxiv.org/pdf/2606.27355v1)

**作者:** Xingyu Ren `[一作]` (Chinese University of Hong Kong), Youran Sun `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在机器人视觉‑语言‑动作（VLA）策略部署前的预先评估中，研究了如何利用有限的试验（probe）结果来进行专家选择与路由，形成了一套“commissioning‑aware routing”框架；

**💡 创新点**

创新点在于：①将试验记录与被测执行严格分离（outcome‑disjoint cross‑fitting），②证明仅凭三次 probe 的成功计数即可捕获大部分条件信息，学习型评分器在此场景下并无显著优势；

**🔧 技术方法**

技术手段包括：构建 14 维试验特征（成功率、Beta 后验、时长、终止信息等），使用透明选择器（probe‑success、Beta‑Binomial）和学习型选择器（逻辑回归、GBDT、MLP），以及三对一交叉验证；

**📊 数据集**

使用 LIBERO‑Plus 账本数据，包含 34,752 条有效 rollout，398 个任务‑扰动变体，28 个冻结专家；

**📈 对比分析**

方法对比：与 Global Best（无目标信息）相比，probe‑success 规则将保留执行成功率从 0.4686 提升至 0.6149（+14.64pp）；学习器（LogReg、GBDT、MLP）表现与此相当；若不保持结果分离，重新使用 scored trial 可将提升放大至 0.7393（+27.07pp），说明 outcome‑separation 对评估至关重要；

**⚠️ 局限性**

局限性：①实验基于固定账本，未考虑真实时间序列与环境漂移；②缺乏视觉/语言上下文，导致布局扰动难以识别；③多数试验使用相同 probe 数量，未探索主动 probe 分配与成本敏感路由；④同一试验可被多次复用会夸大性能。

---

## 594. World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays

**arXiv ID:** 2606.27374 | [PDF](https://arxiv.org/pdf/2606.27374v1)

**作者:** Manish Kumar Govind `[一作]` (University of North Carolina at Charlotte), Srijan Das `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于世界动作模型（WAM）的递归生成回放框架——Recurrent Generative Replay（RGR），通过在持续学习过程中递归生成伪演示轨迹，实现不需要存储旧任务示例即可保持之前学习的技能。

**💡 创新点**

创新点在于首次将WAM的联合动作-观测生成能力用作自带的回放机制，利用语言指令和当前观测递归生成过去任务的轨迹，从而在无真实数据的条件下进行持续模仿学习。

**🔧 技术方法**

核心技术包括：预训练的Cosmos‑Policy WAM、行为克隆损失、递归生成伪轨迹、基于WAM奖励头的任务完成判定，以及在实验中使用的多任务学习与回放策略。

**📊 数据集**

使用的评估数据集为：LIBERO 模拟基准（包含 LIBERO‑Spatial、LIBERO‑Object、LIBERO‑Goal 三套任务）以及真实 xArm7 单臂抓取实验中的三类单臂抓放任务。

**📈 对比分析**

通过与 Seq‑FT、Seq‑LoRA、EWC、PackNet、Experience Replay（ER）以及 Rollouts‑as‑Replay（RAR）等基线进行对比；在模拟实验中 RGR 能将灾难性遗忘降低约 50%，在真实实验中将 NBT 从 96.3% 降至 60.5%，FWT 从 50% 提升至 80%，整体 AUC 亦显著优于无回放方法。

**⚠️ 局限性**

主要局限在于：WAM 生成的视觉质量随递归层数递增而下降，导致 PSNR 降低；此外生成的动作与预测的未来观测不一致，导致“想象成功率高但实际执行成功率低”。提升生成一致性和视觉真实性是未来改进的关键方向。

---

## 595. Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipeline

**arXiv ID:** 2606.27347 | [PDF](https://arxiv.org/pdf/2606.27347v1)

**作者:** Kirill Solovev `[一作]` (University of Graz), Jana Lasser `[通讯]` (University of Graz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个可扩展、完全使用开源权重的多语言联合实体-关系抽取流水线，能够从大规模新闻文本生成带符号、带时序的知识图谱。

**💡 创新点**

创新点在于：①模块化设计，允许任意替换NER、实体链接、关系抽取组件；②使用基于 Wikidata 的三阶段实体链接，支持跨语言唯一标识；③在关系抽取时采用语法约束的 guided decoding 以及整数索引选择候选 QID，确保输出严格符合闭合本体；④实现了符号化关系（合作/冲突）和时间维度的统一抽取。

**🔧 技术方法**

核心技术包括：GLiNER‑X‑Large span‑based NER、Qwen3‑Embedding‑0.6B 词向量检索、Qdrant HNSW 索引、三阶段实体链接（精确、模糊、向量）以及 Qwen3.6‑35B‑A3B‑FP8 的混合专家模型配合 DSPy 进行关系抽取和 Guided Decoding。

**📊 数据集**

使用两大新闻语料库：奥地利 Factiva 记录 2005‑2017 年，波兰新闻集 1997‑2025 年；外部知识库为 Wikidata 进行实体链接；另外利用 Infini‑News 作为公开可复现的新闻来源。

**📈 对比分析**

评估方法：1）构建 3491 条关系的金标准（三大 LLM 共同标注并裁决）并做全文覆盖 spot‑check，严格模式 68.2% 正确率，宽松模式 93.7%；2）在两份独立案例研究（奥地利党派生命周期、波兰 SOE‑政权网络）中对照公开记录，验证结构准确性；实验表明抽取速度可达数千篇/秒，准确率优于现有单语言或仅基于共现的方法。

**⚠️ 局限性**

主要局限：①实体链接覆盖率仅 20‑25%（节点级），对中低频经济精英影响较大；②金标准及 spot‑check 主要依赖 LLM，缺乏完全人类判定；③时间标注仅能在 58% 关系中给出明确起止时间；④对极低频实体和跨语言同义词仍有误检；⑤在复杂的符号网络中，误判率虽低但仍需人工核实以避免声誉损害。

---

## 596. VibeAct: Vibration to Actions for Contact-Rich Reactive Robot Dexterity

**arXiv ID:** 2606.27344 | [PDF](https://arxiv.org/pdf/2606.27344v1)

**作者:** Yuemin Mao `[一作]` (Carnegie Mellon University), Jeffrey Ichnowski `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在机器人手指内嵌入压电麦克风，收集振动信号并通过数字克隆自动标注接触与滑动信息，训练触觉估计器将麦克风波形映射为低维物理表征，并在仿真中使用该表征作为观测训练强化学习策略，最终实现无声学仿真的仿真到现实迁移。

**💡 创新点**

提出了基于接触与滑动低维物理表征的仿真‑现实桥接框架，通过数字克隆自动生成标签并将该表征作为中间表示分离感知与控制，避免了高维音频仿真的困难。

**🔧 技术方法**

采用压电麦克风采集、数字克隆仿真、卷积+注意力触觉估计网络、PPO强化学习以及多任务训练与域随机化等技术。

**📊 数据集**

使用两套实地遥操作记录（固定物体5小时、移动物体2小时）训练触觉估计器，并在五个仿真任务中收集训练数据。

**📈 对比分析**

与仅用本体感知和点云的基线相比，VibeAct在五项接触密集任务中平均提升约30–40%成功率，滑动幅值通道是最关键的性能提升来源，实验在物理机器人上验证了迁移效果。

**⚠️ 局限性**

表征丢失原始振动信息、依赖特定硬件配置、数字克隆标签需精准物体姿态、以及策略对表征的平面化处理未利用时空结构。

---

## 597. Understanding Domain-Aware Distribution Alignment in Budgeted Entity Matching

**arXiv ID:** 2606.27342 | [PDF](https://arxiv.org/pdf/2606.27342v1)

**作者:** Nicholas Pulsone `[一作]` (Worcester Polytechnic Institute), Roee Shraga `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对BEACON框架中低资源、跨域实体匹配的分布对齐机制进行深入实验，评估标签可用性、域表示与域无关设置对性能的影响。

**💡 创新点**

揭示分布对齐在不同标签条件下的有效性，提出标签感知与多样化域表示（如中位点、方差、覆盖度）的改进，并证明分布对齐可作为降采样策略。

**🔧 技术方法**

使用RoBERTa作为基础语言模型，结合BEACON的Train–Validation Distribution Fitting（TVDF）和其变体（TVMed、TVDF_VAR、TVCoverage），并实现标签感知采样。

**📊 数据集**

主要使用WDC多维实体匹配基准（50%角落案例/50%已见实体），以及Amazon‑Google、Beers、DBLP‑ACM等公开数据集进行域无关实验。

**📈 对比分析**

与随机、基于中心点和最近邻的采样方法比较，实验显示在大多数预算/数据集上，TVDF在宏平均和加权F1上均优于其他方法；在域无关降采样中，TVDF在Amazon‑Google上显著优于基准。

**⚠️ 局限性**

局限在于仅评估RoBERTa模型、主要聚焦WDC数据，缺乏对不同PLM或更广泛EM基准的验证，且标签感知方法在小域上可能导致样本分散不足。

---

## 598. Language-Based Digital Twins for Elderly Cognitive Assistance

**arXiv ID:** 2606.27334 | [PDF](https://arxiv.org/pdf/2606.27334v1)

**作者:** Mohammad Mehdi Hosseini `[一作]` (University of Denver), Hiroko H. Dodge `[通讯]` (Massachusetts General Hospital)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于大语言模型的语言数字孪生框架，模拟老年人对话行为并实现持续的认知健康监测

**💡 创新点**

在传统预测模型基础上加入时间语义特征（停顿、语速）并利用多头条件变分自编码器同时评估对话忠实度与MCI评分，实现双重性能评估

**🔧 技术方法**

大语言模型（LLM）微调、条件变分自编码器（cVAE）、句子嵌入（Sentence‑BERT）、情感分析模型、Whisper ASR与pyannote语音分离

**📊 数据集**

I‑CONECT 数据集（老年人自然对话、多模态信息、MoCA 评分）

**📈 对比分析**

与原始 GPT 生成文本比较，cVAE 评估重建误差与 MoCA 预测误差。数字孪生重建误差与 MoCA 误差与真实对话几乎相同，明显优于 GPT（误差约 0.4–1.1 vs 3.5–5.1）

**⚠️ 局限性**

样本量仅约70人中选5人，样本规模有限；模型在跨人群泛化及多模态（语音、视频）扩展方面尚未充分验证

---

## 599. LLM-Based Examination of Eligibility Criteria from Securities Prospectuses at the German Central Bank

**arXiv ID:** 2606.27316 | [PDF](https://arxiv.org/pdf/2606.27316v1)

**作者:** Serhii Hamotskyi `[一作]`, Christian Hänig `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一套多阶段生成式信息抽取管线，利用大型语言模型（LLM）自动判定德国联邦银行证券的合格性（是否符合资格标准）

**💡 创新点**

从传统的命名实体识别（NER）转向生成式IE，并引入LLM-as-a-judge进行价值基评估，能处理多语言、OCR噪声和半结构化文本；同时采用零-shot/指令式推理，避免了大量手工标注

**🔧 技术方法**

使用Llama3.3‑70B‑Instruct、Cohere Command‑R 08‑2024等大型LLM进行文本抽取与归一化，LangChain实现结构化JSON输出，Python实现规则化后推断；评估阶段采用Mistral Small 3.1‑Instruct作为LLM‑as‑a‑judge；预处理阶段使用Docling将PDF转Markdown并做Unicode/空格归一化

**📊 数据集**

FinCorpus‑DE10k基础数据集，共413份证券说明书（268份训练、145份测试）；文档包含德文、德英双语，含表格、注释框、OCR噪声；测试集为双重标注，含约82份不合格（约29%）

**📈 对比分析**

与传统NER（如FinBen/FinBERT等）进行对比；评价指标包括文档级合格率、每个资格标准的准确率、召回率、F1；LLM在多数指标（如货币、本金、票面利率、资产类型）表现与或优于传统模型，尤其在语义复杂或多语种标准上优势明显；总体文档合格准确率达≈95%，F1≈0.80-0.85

**⚠️ 局限性**

主要局限：计算成本高（每文档需多次LLM调用，耗时数十秒）；对PDF结构（列、表、复选框）的处理仍依赖文本提取，视觉模型尚未集成；LLM易出现安全偏向，导致召回率低；需要进一步改进PDF解析、引入RAG和可追溯生成，进行Meta‑评估以量化LLM‑as‑a‑judge的偏差

---

## 600. Blackwell Approachability and Gradient Equilibrium are Equivalent

**arXiv ID:** 2606.27315 | [PDF](https://arxiv.org/pdf/2606.27315v1)

**作者:** Brian W. Lee `[一作]` (University of California, Berkeley), Ryan J. Tibshirani `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本研究展示了梯度平衡（GEQ）与布莱克威尔可接近性（BA）在算法意义上的等价性，表明可以通过黑箱GEQ oracle解决BA问题，反之亦然。

**💡 创新点**

创新点在于建立了GEQ与BA之间的黑箱oracle归约，明确了GEQ与其他在线学习框架（如后悔最小化和校准）之间的关系。

**🔧 技术方法**

使用了黑箱oracle归约技术，特别是将GEQ问题转化为BA问题的算法，以及将BA问题转化为GEQ问题的算法。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了GEQ和BA的理论框架及其应用。

**📈 对比分析**

通过黑箱oracle归约，GEQ与BA之间的等价性使得可以将后悔最小化算法应用于GEQ问题，反之亦然，性能上可以实现乐观的错误界限和强适应性错误界限。

**⚠️ 局限性**

限制在于当前的研究主要集中在有约束的GEQ问题与无约束的GEQ问题之间的转换，未来的工作可以探索如何在不转化为锥形可接近性形式的情况下解决非锥形可接近性问题。

---

## 601. Beyond Surface Forms: A Comprehensive, Mechanism-Oriented Taxonomy of Indirect Linguistic Encoding for LLM-Based Coded Language Detection

**arXiv ID:** 2606.27314 | [PDF](https://arxiv.org/pdf/2606.27314v1)

**作者:** Hamid Reza Firoozfar `[一作]` (University of Utah), Paul Jen-Hwa Hu `[通讯]` (University of Utah)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种以编码机制为核心的间接语言编码（ILE）完整分类法，并通过对2,000条TikTok与Bluesky英文帖子进行人工标注和评估，验证该分类法在大型语言模型（LLM）提示下的检测效果；

**💡 创新点**

创新点在于采用统一的元特征和迭代构建方法，形成11类顶层机制、33类细粒度子机制的体系，覆盖之前分类法缺失或粗化的编码方式，并强调机制与意图分离，使得检测更稳定、可扩展；

**🔧 技术方法**

技术方法包括LLM少样本提示（GPT‑5.4、Claude Sonnet 4.6、DeepSeek V4 Flash）、多轮迭代的分类法构建、统计评估（Cohen κ、bootstrap、McNemar）以及与传统监督/无监督 NLP 基线的对比；

**📊 数据集**

使用了2,000条英文社交媒体帖子（1,400 TikTok，600 Bluesky），每条帖子均标注文档级是否含有ILE、span级最小编码区间及对应的机制类别；

**📈 对比分析**

对比四种现有分类法和无分类法，在文档级和span级评估指标（准确率、宏F1等）上，采用该分类法的LLM在所有三种模型中均获得最高性能，文档级准确率提升约4.7%（5.4% F1），span级F1提升3.4%，明显优于监督/无监督基线；

**⚠️ 局限性**

局限性包括：仅限英文及TikTok/Bluesky文本，无法覆盖非文字模态（图片、音频等）；对非拉丁文字系统的适用性不明；缺乏意图、上下文判别，仅识别编码机制；未来需扩展多语言、多模态与动态机制更新。

---

## 602. Fast algorithms for learning a Gaussian under halfspace truncation with optimal sample complexity

**arXiv ID:** 2606.27298 | [PDF](https://arxiv.org/pdf/2606.27298v1)

**作者:** Haitong Liu `[一作]` (ETH Zurich), Manuel Wiedmer `[通讯]` (ETH Zurich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种利用截断高斯分布样本的三阶矩张量随机收缩与一元函数逆求解，基于方法求矩的思想实现对完整高斯分布均值与协方差的高精度恢复；

**💡 创新点**

创新点在于：①用随机张量收缩将难解的秩‑1张量近似为可分解矩阵，从而高效恢复截断方向；②通过构造与截断参数相关的一元函数并精确求逆，得到截断阈值的高精度估计；③针对截断概率不同区间给出最优的样本复杂度表述，显著降低了对大样本量的需求；

**🔧 技术方法**

采用方法求矩（method of moments）、随机张量收缩、特征向量分解、均值值定理、Sum‑of‑Squares 级数逆求解等技术；

**📊 数据集**

实验使用合成的多维高斯样本，没有使用公开数据集；

**📈 对比分析**

与以往的截断高斯参数估计方法相比，本算法在 α 较小（远离 1）时样本复杂度为 O(d²/α + d²·log⁴(1/α)/ε²)，在 α 接近 1 时仅需 O(d²/ε²) 样本；在所有情况均能以 O(ε) 的总变差误差实现；

**⚠️ 局限性**

局限性包括：当截断概率 α 趋近于 1 时 κ₃(γ) 变得极小导致样本复杂度急剧上升；逆函数求解需要数值逼近，可能引入误差；以及在高维稀疏协方差时需要额外的子空间识别步骤。

---

## 603. LA4VLA: Learning to Act without Seeing via Language-Action Pretraining

**arXiv ID:** 2606.27295 | [PDF](https://arxiv.org/pdf/2606.27295v1)

**作者:** Tao Lin `[一作]` (Shanghai Jiao Tong University), Bo Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了LA4VLA框架，通过将机器人演示拆分为原子动作段并生成无视觉的语言-动作对，进行语言-动作预训练，随后与传统VLA预训练结合，提升机器人对语言指令的动作执行能力。

**💡 创新点**

创新点在于：①将VLA演示转化为可视化无关的语言-动作（LA）对（LA4-33K），打破视觉与语言监督的耦合；②在预训练阶段引入纯语言-动作监督，证明其在单独或与VLA结合时均能显著提升性能及鲁棒性。

**🔧 技术方法**

采用的技术包括：VLM（Qwen-3-VL-Plus）进行视频原子动作划分与自然语言描述生成、人工验证筛选、InternVL3-1B作为基础网络、流动匹配动作头、以及多种预训练策略（LA、VLA、LA‑VLA、MixPT）进行实验比较。

**📊 数据集**

使用的数据集包括：从DROID机器人演示生成的LA4-33K（33K条无视觉语言‑动作对），MetaWorld 与 LIBERO 两大模拟基准，以及在xArm6真实机器人上的按压按钮、放书、放饮料三项语言条件任务。

**📈 对比分析**

实验通过与无预训练、仅VLA预训练、仅LA预训练、MixPT等多种基线对比，发现：在MetaWorld平均成功率提升13.27%，在LIBERO提升2.45%，真实任务平均成功率提升43.4%；在视觉噪声下，LA预训练与MixPT保持高鲁棒性，平均成功率提高至70%。

**⚠️ 局限性**

局限性包括：依赖人工验证和VLM生成的原子划分可能存在误差；仅处理原子级动作，难以直接扩展到复杂长时序任务；跨域迁移与大规模多机器人部署仍需进一步验证。

---

## 604. Generative Models on Analog Hardware with Dynamics

**arXiv ID:** 2606.27294 | [PDF](https://arxiv.org/pdf/2606.27294v1)

**作者:** Yu-Neng Wang `[一作]` (Stanford University), Sara Achour `[通讯]` (Stanford University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了Analog Interaction Systems (AIS) 这一面向硬件实现的动态系统框架，并通过时间分段参数和隐藏物理状态两种机制缩小与传统神经网络的表达能力差距，采用Wasserstein GAN训练方法实现从噪声到数据的生成；随后设计了稀疏低比特宽度的硬件拓扑，并给出能耗估算。

**💡 创新点**

① 将硬件受限的连续动力学映射为可训练的生成模型；② 通过时间分段权重和隐藏状态提升物理动力学的可表达性；③ 使用端点监督的Wasserstein GAN而非轨迹监督，充分利用物理系统自由演化；④ 设计可扩展的稀疏低比特宽度的模拟器架构，实现在单次连续演化中完成生成。

**🔧 技术方法**

基于耦合振荡器（Kuramoto+SHIL）、模拟Ising机（PolyNet、TanhNet）、单层神经ODE（SeluNet）等硬件动力学模型；训练采用Wasserstein GAN‑GP，梯度回传通过可微ODE求解器；硬件实现通过可编程耦合单元、时变权重、隐藏节点等模块实现；能耗分析基于28nm振荡器Ising机功耗数据。

**📊 数据集**

MNIST 与 Fashion‑MNIST 两个灰度图像数据集。

**📈 对比分析**

与先前的硬件可实现生成模型（Denoising Thermodynamic Model、Neural Langevin Machine）进行比较，FID 分别为 27.6 / 80.8（本工作）vs. 107.8 / 112.8（DTM）和 230.5 / 200.8（NLM）；能耗为约 23 μJ/图像，比数字模型低两位数。实验还展示了时间分段、非对称耦合、隐藏状态、量化与噪声对性能的影响。

**⚠️ 局限性**

① 仍受限于物理动力学的函数形式，无法实现高阶交互；② 需要稀疏、低比特宽度的硬件拓扑，扩展到更大规模或更复杂任务仍有挑战；③ 训练过程依赖可微ODE求解器和WGAN，计算开销与优化难度较高；④ 在数字端到端性能（如FID）上仍未达到最先进的分布式深度生成模型；⑤ 需要针对具体硬件实现的细节进一步验证。

---

## 605. Deterministic Algorithms for Low Individual Degree Factors of Sparse Polynomials

**arXiv ID:** 2606.27293 | [PDF](https://arxiv.org/pdf/2606.27293v1)

**作者:** Somnath Bhattacharjee `[一作]` (University of Toronto), Shubhangi Saraf `[通讯]` (University of Toronto)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文提出了两类稀疏多项式因式分解算法：一是对具有有界个体次数（individual degree）d 的 n 变元 s 稀疏多项式给出确定性多项式时间算法，输出包含所有非单项式因子的列表；二是对一般稀疏多项式给出确定性准多项式时间算法，输出包含所有个体次数 ≤ d 的因子的列表。

**💡 创新点**

创新点包括：①通过结合归一化（normalization）与反归一化（reverse‑normalization）以及 Hensel‑ready 判定，构造了一个多项式时间的因子列表生成器；②首次给出对稀疏多项式的“因子稀疏度上界”与“因子数量上界”的隐式证明，直接导致列表大小为 s^d；③利用 Klivans–Spielman 的稀疏多项式测试器与生成器实现对极低阶因子（≤ d）在常数深度电路上的可重建；④通过功率级数根的显式常数深度表示，将所有潜在因子以小电路形式列出。

**🔧 技术方法**

主要技术手段包括：Hensel 递推与功率级数根的构造；Sylvester 矩阵、结果式与判别式的线性代数方法；对稀疏多项式的可判定点（hitting set）与多变量生成器；对逆多项式与对称多项式的基于电路的重构；以及在有限域与特征 0 下的多项式时间因式分解和除法测试算法。

**📊 数据集**

本研究为理论计算，未使用实验数据集；所有证明与算法均在符号层面完成。

**📈 对比分析**

相较于以往随机化算法，本文提供了确定性多项式/准多项式时间的因式分解框架；对稀疏多项式的因子数量上界与稀疏度上界的改进，使得输出列表大小为 s^d 或 s^d·log n，明显优于早期指数/子指数方法。性能方面，稀疏个体次数约束下实现多项式时间；一般稀疏多项式下实现准多项式时间，进一步压缩了此前的准多项式 (n, s^d^7 log n) 或 (n, s^d^2 log n) 的上界。

**⚠️ 局限性**

限制与未解决的问题：①输出列表可能包含与输入多项式无关的“伪因子”，除非再加上可判定因子除法的子算法；②对一般稀疏多项式完全因式分解仍无法在多项式时间内实现；③算法对域大小有要求，需足够大或使用域扩展；④对特征 p> d 的有限域，复杂度受 p 影响；⑤在实际实现中，生成点与功率级数根的系数计算仍可能昂贵。

---

## 606. Not All Actions Are Equal: Rethinking Conditioning for Dexterous World Model

**arXiv ID:** 2606.27325 | [PDF](https://arxiv.org/pdf/2606.27325v1)

**作者:** Zizhao Yuan `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 DexAC‑WM，专为高自由度手部动作设计的结构化动作条件化世界模型，能够在第一人称视角下生成细粒度、连贯的手部交互视频。

**💡 创新点**

核心创新在于：①把动作维度保持成独立的 token 而非整体压缩；②采用局部细化+全局调制的联合机制，让细微手指动作与宏观腕部/摄像机运动协同；③引入 DINOv3 视觉特征与 VLM 文本嵌入的双模态跨注意力，增强语义对齐。

**🔧 技术方法**

使用的技术包括：结构化动作分词器（SAT）、局部交叉注意力、全局自适应层归一化（AdaLN）调制、DINOv3 + VLM 的双模态跨注意力、Diffusion-based DiT backbone（Cosmos‑Predict2.5）、以及基于 6D 旋转表示的刚体运动编码。

**📊 数据集**

实验数据集：EgoDex（829 小时、194 任务、500 对象）和 EgoVerse（1362 小时、1965 任务、240 场景）。

**📈 对比分析**

与 Wan‑Control、IRASim、Cosmos‑Predict2.5 等先进基线比较，DexAC‑WM 在 FID/FVD 下降 6%~10%、PCK 提升 5%~7%，同时保持或提升 PSNR、SSIM，证明在视觉质量、时序一致性和动作跟随方面均有显著提升。

**⚠️ 局限性**

局限性：①对极长预测时长（>2 秒）仍出现 FVD/ PCK 降低；②结构化分词器与多模态注入增加了模型复杂度与推理开销；③在某些多物体或形变对象场景下，手物交互一致性仍略逊于基线。

---

## 607. OctoSense: Self-Supervised Learning for Multimodal Robot Perception

**arXiv ID:** 2606.27317 | [PDF](https://arxiv.org/pdf/2606.27317v1)

**作者:** Anthony Bisulco `[一作]` (University of Pennsylvania), Pratik Chaudhari `[通讯]` (University of Pennsylvania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 OctoSense 硬件平台及其 59 小时的多模态机器人数据集，并基于该数据集训练了一种 late‑fusion 结构的自监督 masked autoencoder，用以实现跨传感器的鲁棒感知。

**💡 创新点**

创新点包括：①提供首个兼容事件相机的、覆盖 8 种传感器的大规模机器人数据集；②提出模态专用 tokenizer 与四维 RoPE 的 late‑fusion MAE；③在推理时缓存模态特征，显著提升速度；④在夜间或传感器失效场景下仍保持优秀性能。

**🔧 技术方法**

采用自监督学习的 masked autoencoder、模态专用 tokenizers、四维 RoPE 及 cross‑modal encoder；训练时使用 Finite‑Scalar‑Quantization；推理采用缓存机制；下游任务使用 DPT‑style dense head 与 cross‑attention 警戒模块。

**📊 数据集**

使用 OctoSense 数据集（59 小时、8 传感器），并与 DINOv2/3、SigLIP2、Perception Encoder、V‑JEPA 2.1 等公开视觉基础模型以及 M3ED 数据集进行跨数据集评估。

**📈 对比分析**

在 depth、flow、segmentation、ego‑motion 与 steering 任务上与上述单模态基线以及 RGB‑only、视频 MAE、早期融合 MAE 等方法进行对比。结果显示 late‑fusion MAE 在 depth RMSE‑1.65 m、flow EPE‑7.16 px、segmentation 0.411、ego‑motion 0.06 m 及 0.23° 旋转误差方面均优于所有单模态基线，尤其在夜间及传感器失效情况下优势更为明显。

**⚠️ 局限性**

局限性包括：①数据量相对行业级基础模型小，缺乏更大规模训练；②仅在 5 Hz 采样下评估，未充分利用高频事件/IMU 传感器；③计算预算限制未尝试更大网络或更长时间窗口的递归架构；④需社区级数据聚合才能构建真正的多模态基础模型。

---

## 608. ViQ: Text-Aligned Visual Quantized Representations at Any Resolution

**arXiv ID:** 2606.27313 | [PDF](https://arxiv.org/pdf/2606.27313v1)

**作者:** Xumin Yu `[一作]` (Tencent HY Vision Team), Yongming Rao `[通讯]` (Tencent HY Vision Team)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ViQ框架，将视觉表示量化为离散的文本对齐向量，支持任意分辨率输入。

**💡 创新点**

创新点在于两阶段训练：先用文本对齐预训练提升语义对齐，再通过近端表示与多头有限标量量化（FSQ）逐步压缩特征空间，并结合旋转位置编码实现任意分辨率。

**🔧 技术方法**

使用了向量量化（FSQ）、L∞近端正则化、Bottleneck压缩、多头注意力扩展、旋转位置编码（RoPE）、自蒸馏、VAE潜在重构损失等技术。

**📊 数据集**

在多模态基准集上训练和评估：MMStar、MMMU、SimpleVQA、InfoVQA、TextVQA、DocVQA、OCRBench、AI2D、ChartQA，并使用LLaVA-OneVision 2000K样本、Qwen-Image等低层监督。

**📈 对比分析**

与连续编码器（CLIP、SigLIP2、DINOv2、InternViT等）及离散编码器（QLIP、UniTok）对比，ViQ在大多数任务上与连续编码器相当或更优（例如Avg 57.2/63.9），训练速度提升20-70%，重构PSNR 22.73、SSIM 0.66、rFID 0.62。

**⚠️ 局限性**

局限性包括未验证在70B+规模LLM上的表现、对预训练数据多样性敏感、在高频细节任务（如OCR）仍存在细节损失，以及离散化本身对极细粒度信息有不可避免的损失。

---

## 609. Sculpting NeRF Geometry: Human-Preference Fine-Tuning of a 3D-Aware Face GAN

**arXiv ID:** 2606.27305 | [PDF](https://arxiv.org/pdf/2606.27305v1)

**作者:** Archer Moore `[一作]` (University of Melbourne), Liam Hodgkinson `[通讯]` (University of Melbourne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对无条件3D‑aware面部GAN（EG3D）进行基于人类偏好奖励的几何微调，直接在NeRF密度体上学习奖励模型并用GAN‑loop细调，从而提升3D几何质量而不改变2D外观。

**💡 创新点**

创新点在于：①直接对NeRF密度体学习奖励，无需mesh或文本提示；②仅用少量单一评审者的偏好对比样本即可训练奖励模型；③通过密度一致性约束保持2D渲染的可视质量；④证明人类偏好可以在隐式3D表示上驱动几何改进。

**🔧 技术方法**

采用的技术包括：RLHF的pairwise preference loss、3D U‑Net＋MLP奖励网络、GAN‑loop更新、密度一致性损失、SHAP/Integrated Gradients可解释性分析，以及外部用户研究评估。

**📊 数据集**

使用的数据集为FFHQ人脸图像（用于训练EG3D），随后生成σ体并由单一评审者对约4,346对比样本进行偏好排序；外部用户研究使用40个微调前后模型的对比。

**📈 对比分析**

与未微调EG3D和无奖励微调版对比，FID‑50k从4.09升至6.66；奖励分数平均提升+12.9；外部用户研究中74.4%受访者偏好微调后几何；奖励模型在同一分布上的对比准确率91%，而基于深度或点云的奖励低。

**⚠️ 局限性**

局限性包括：奖励模型对训练生成器的σ分布高度依赖，跨模型迁移受限；仅使用单一评审者数据，缺乏多评审者一致性评估；微调导致一定的FID下降和多样性降低；缺乏对更复杂形状或多视图情境的验证。

---

## 610. DnA: Denoising Attention for Visual Tasks

**arXiv ID:** 2606.27372 | [PDF](https://arxiv.org/pdf/2606.27372v1)

**作者:** Ron Campos `[一作]` (University of Central Florida), Aritra Dutta `[通讯]` (University of Central Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的注意力机制——Denoising Attention（DnA），通过同时使用 softmax 与 softmin 对正负交互进行建模，并将正交互投射到一个子空间，负交互投射到另一个子空间，从而实现对噪声的抑制和特征的增强。

**💡 创新点**

创新点主要包括：
1) 通过正负查询分别捕获正确类特征和与之相关但不属于该类的干扰特征；
2) 在两条分支中使用不同的值投影（V⁺、V⁻），实现子空间的正交化，提升判别能力；
3) 采用 softmax 与 softmin 的组合，使模型既能强调强交互，又能保留弱交互信息，实现“去噪”效果。

**🔧 技术方法**

技术手段包括：ViT-B 视觉 Transformer 框架、软化激活函数（softmax/softmin）、多头注意力的正负查询与值投影、子空间角度分析、参数化平衡系数 α、以及在视频 Transformer 和视频 LLM 适配中的跨模态注意力实现。

**📊 数据集**

使用的数据集：
- 图像分类：ImageNet‑1K（训练/验证/测试）、ImageNet‑A（鲁棒性评估）、ImageNet‑O、ImageNet‑R；
- 迁移学习：CIFAR‑10/100、Stanford Cars、MS COCO；
- 视频理解：Toyota Smarthome、NTU RGB+D 60、Kinetics‑400、Ego‑in‑Exo PerceptionMCQ。

**📈 对比分析**

与基线比较方法：ViT‑B（DeiT 预训练配置）、Differential Attention、Cog Attention。实验表明：
- ImageNet‑1K 测试准确率提升 0.8%（相较于 ViT‑B）；
- 在 ImageNet‑A 上提升排名、校准误差；
- 在 Toyota Smarthome（CS 1.3%，CV 4.0%）和 NTU60（CS 1.2%，CV 0.6%）等视频任务上也取得显著提升；
- 在 Video LLM（Ego‑in‑Exo）任务上平均提升 0.5%。
- 通过子空间角度和相似度分析验证正负分支的分离与去噪效果。

**⚠️ 局限性**

限制：
- 机制引入了额外的参数（视频 Transformer 约 3%–23% 的参数增长），导致训练成本提升；
- 需要在保证性能的同时进一步探索参数高效实现；
- 对于极大规模模型，额外的注意力分支可能影响并行度和实时性。

---

## 611. AI Healthcare Chatbots as Information Infrastructure: A Large-Scale Study of User-Reported Breakdowns

**arXiv ID:** 2606.27302 | [PDF](https://arxiv.org/pdf/2606.27302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 612. DanceOPD: On-Policy Generative Field Distillation

**arXiv ID:** 2606.27377 | [PDF](https://arxiv.org/pdf/2606.27377v1)

**作者:** Wei Zhou `[一作]` (ByteDance Seed), Tat-Seng Chua `[通讯]` (NUS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在流匹配模型中对多种生成与编辑能力进行融合的 on‑policy 场景字段蒸馏框架（DanceOPD），通过硬路由、学生状态查询与单点低噪声匹配实现多能力合成。

**💡 创新点**

通过硬路由的样本级字段匹配、在学生轨迹上按语义侧单点查询以及直接速度 MSE 对齐，解决目标字段歧义、状态分布失配和轨迹相关性等关键挑战。

**🔧 技术方法**

采用流匹配模型、on‑policy 蒸馏、硬路由字段匹配、学生轨迹采样与单点低噪声查询，并用速度 MSE 损失进行训练。

**📊 数据集**

以 Z‑Image 为主训练集，并在 OmniEdit、Attribute、Style 等编辑子集上评估；真实性评估使用自定义 photorealism 奖励；CFG 相关实验基于 SD3.5‑M。

**📈 对比分析**

与 DiffusionOPD、Flow‑OPD、off‑policy 以及传统联合训练/权重融合等方法对比，DanceOPD 在 T2I+编辑、局部/全局编辑、真实性场景吸收及 CFG 吸收等多项基准中均取得显著提升（如 GEditBench +8.1%、GenEval +2%、真实性奖励 +9.9%）。

**⚠️ 局限性**

局限在于需假设所有冻结源共享相同的速度场并且可预定义路由；难以处理任务边界模糊或动态能力切换；对不同模型结构或字段不兼容的情况支持有限。

---

## 613. RayPE: Ray-Space Positional Encoding for 3D-Aware Video Generation

**arXiv ID:** 2606.27345 | [PDF](https://arxiv.org/pdf/2606.27345v1)

**作者:** Minghao Yin `[一作]` (University of Hong Kong), Kai Han `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在预训练视频扩散变换器中引入基于 Plücker 坐标的 RayPE，将几何信息注入自注意力的 Q/K，提升相机轨迹控制和跨帧 3D 一致性。

**💡 创新点**

利用 Plücker reciprocal product 与注意力点积的代数相匹配，通过 Q/K 的“flip”投影以及 Normalize‑Gate‑Inject 实现可扩展、尺度无关的几何注入。

**🔧 技术方法**

Plücker 坐标、注意力自注意力加法注入、RMSNorm、Q/K flip、尺度门控、log‑scale 数据增强及零初始化。

**📊 数据集**

RE10K、DL3DV、PanShot、OmniWorld 四个视频数据集，测试使用 RE10K hold‑out、电影剧照、绘画等 OOD 场景。

**📈 对比分析**

与 CameraCtrl、ReCamMaster、UCPE、ReRoPE 以及四种内部替代实现对比，RayPE 在 RE10K 上实现更低姿态误差、更高 CLIP 分数以及更低 FVD/FID，整体性能提升超过 15%–20%。

**⚠️ 局限性**

仍需相机轨迹标注，极端尺度差异下效果有限，未在更大模型或跨模态任务中验证，算力开销略高。

---

## 614. See & Sniff: Learning Visuo-Olfactory Representations

**arXiv ID:** 2606.27307 | [PDF](https://arxiv.org/pdf/2606.27307v1)

**作者:** Seongyu Kim `[一作]` (Korea Advanced Institute of Science and Technology), Arda Senocak `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为See & Sniff的自监督视觉-嗅觉联合表示学习框架，并通过合成配对将单模态嗅觉数据扩展为跨模态训练数据集SmellNet‑V，首次实现嗅觉与视觉的联合学习；

**💡 创新点**

创新点在于：①利用嗅觉在语义类别内对视觉变换的鲁棒性，构造跨模态数据；②引入密集局部对齐机制与轻量对齐模块，实现细粒度视觉-嗅觉对齐和气味源定位；③首次将嗅觉定位作为下游任务，并提供相应基准；

**🔧 技术方法**

技术主要包括自监督对比学习、两流Transformer编码器、密集局部相似度图、轻量对齐模块以及图像与嗅觉信号的时序预处理；

**📊 数据集**

使用SmellNet原始嗅觉数据与通过LLM生成的语义匹配网页图像构成的SmellNet‑V；同时利用SmellNet-Test、SmellNet‑V-Test和SmellNet‑V‑Source等子集评估；

**📈 对比分析**

与单模态ScentFormer、全局对齐和多种视觉基线比较，结果显示See & Sniff在嗅觉分类上提升约7%，在跨模态检索中R@1提升至0.48（相较于全局对齐为0.43），在嗅觉定位中mAP和mIoU分别达0.49/0.33，优于全局对齐和可视化基线；

**⚠️ 局限性**

局限性包括：①合成配对仅在同一语义类别内有效，对不同类别跨模态关联缺乏探索；②在纹理主导的状态（如粉末状香料）定位效果不佳；③未处理复杂混合气味或变质状态的跨模态学习；

---

## 615. Reading the Same Data Differently: Interpretive Labor Across System Boundaries in Electronic Monitoring

**arXiv ID:** 2606.27301 | [PDF](https://arxiv.org/pdf/2606.27301v1)

**作者:** Yibo Meng `[一作]` (Tsinghua University), Hongyu Zhou `[通讯]` (University of Cambridge)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对中国社区矫正系统的26名受监管人员和12名执法人员进行半结构化访谈，研究电子监控系统中双方对相同连续数据流的解释差异，提出解释性失调（interpretive misalignment）概念，并给出可见性、可争议性与责任性设计建议。

**💡 创新点**

首次将解释性失调作为解释连续感知系统中不同主体解读不一致的理论框架，强调分布式解释工作对合规行为和冲突产生的影响，并提出针对双边可解释性的设计原则。

**🔧 技术方法**

采用质性研究方法——访谈、主题分析与案例对比，并对电子监控系统的界面与数据处理流程进行描述性分析；未使用算法或机器学习技术。

**📊 数据集**

使用的“数据集”为26名受监管者与12名执法者的访谈文本记录（约45–65分钟的录音转写）。

**📈 对比分析**

本研究不进行算法性能比较，而是通过比较受监管者与执法者的解释模型、行为策略与冲突案例，阐明解释性失调对合规与冲突的实际影响。

**⚠️ 局限性**

研究局限包括：样本仅来自中国社区矫正背景，缺乏跨地区或跨司法体系验证；访谈依赖受访者回忆，可能存在记忆偏差；未进行现场观察或实时数据跟踪。

---

## 616. CHIA: An open-source framework for principled, agentic AI-driven hardware/software co-design research

**arXiv ID:** 2606.27350 | [PDF](https://arxiv.org/pdf/2606.27350v1)

**作者:** Angela Cui `[一作]` (University of California, Berkeley), Sagar Karandikar `[通讯]` (University of California, Berkeley)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了开源框架 CHIA，用来构建、部署和评估基于 AI 的硬件/软件协同设计流程，并通过五个案例（gem5 对齐、RTL 扩展实现、关键路径优化、进化式结构发现、GitHub issue 修复）演示其能力。

**💡 创新点**

创新点在于：①把 AI 驱动的设计流程本身视为一等公民，提供图形化循环、资源调度、容器化、分布式执行、错误容忍与可重复实验的完整抽象；②支持多种工具与模型的可插拔节点；③通过“CHIA loop”实现 AI 与程序化控制的混合，既可完全由代理驱动，又可由程序手动控制；④为研究提供统一的实验环境与数据收集，促进可比性。

**🔧 技术方法**

使用技术包括：Ray 分布式调度框架；Docker/OCI 容器化；LLM 与代理（Claude Code、AlphaEvolve、AdaEvolve、EvoX 等）；硬件工具节点（Chipyard、gem5、ChampSim、FireSim、Hammer、Vivado、Hammer/ASIC CAD、S3 等）；数据库（SQLite）和自定义缓存、故障容忍机制；Python 语言编写的流程脚本。

**📊 数据集**

使用的数据集主要包括：SPEC 2006 reference suite（25 trillion 指令）；Microbench、Embench IOT 2.0、OpenSSL benchmark；DPC‑4 AI/ML workload traces；GitHub issue 记录（CIRCT 项目）；以及各类自定义基准（如 36 个 Microbench，Embench 训练/测试集）。

**📈 对比分析**

对比方法：在相同 SoC（Chipyard）和同一硬件平台（SkyWater 130nm、商业 16nm PDK）上跑同一系列 benchmark；通过 SPEC 2006 完整跑 25 trillion 指令、FireSim、Verilator 等验证功能；对比周期、IPC、频率、面积、功耗等指标。结果显示：①gem5 对齐误差低于 7%；②RTL 扩展实现比 baseline 提升 5‑10% 性能；③关键路径优化频率提升 2.03×，IPC 仅下降 3%；④演化式结构发现产生的新 L2 前向器性能提升 15‑30%；⑤GitHub issue 修复 7/11 目标问题成功解决。

**⚠️ 局限性**

局限性：①高级商业 PDK 的预测效果有限，因敏感数据受限导致 AI 只能在开源 PDK 上调优；②LLM 生成的 RTL 仍需人工审核与验证；③高成本的 API 调用与大模型推理；④缺乏超大规模、实时的评估基准，导致评估周期仍较长；⑤容器与云资源的安全隔离依赖环境，若出现恶意模型风险尚未覆盖。

---

