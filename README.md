# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-09 | 今日论文总数: 582

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Weighted Bayesian Conformal Prediction

**arXiv ID:** 2604.06464 | [PDF](https://arxiv.org/pdf/2604.06464v1)

**作者:** Xiayin Lou `[一作]` (Technical University of Munich), Peng Luo `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 4525 | [OpenAlex ID](https://openalex.org/A5033231648)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出加权贝叶斯一致性预测（WBCP），将BQP‑CP推广到任意重要性权重下，实现分布偏移场景下的贝叶斯阈值后验与数据条件覆盖保证；

**💡 创新点**

通过将Dirichlet参数从(1,…,1)改为(·ŵ1,…,·ŵn)，实现权重化后验；并给出有效样本量、后验收敛率、加权随机占优和条件覆盖边界的理论证明；

**🔧 技术方法**

加权贝叶斯自举、Dirichlet后验、Monte‑Carlo采样、地理核权重、Kish有效样本量等技术；

**📊 数据集**

合成空间数据（含正弦+余弦函数+空间相关噪声）与真实西雅图房价数据；

**📈 对比分析**

与标准CP、加权CP（GeoCP）、BQ‑CP、AdaGeoCP等方法比较，WBCP在保持≥90%覆盖率的同时提供更宽区间并输出σ_post诊断，实验显示覆盖率超标且区间宽度略增；

**⚠️ 局限性**

仅使用Dirichlet模型，忽略空间相关效应；有效样本量估计近似；计算成本随Monte‑Carlo样本线性增长。

---

## 2. Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization

**arXiv ID:** 2604.06285 | [PDF](https://arxiv.org/pdf/2604.06285v1)

**作者:** Igor Maljkovic `[一作]` (University of Genoa), Fabio Roli `[通讯]` (University of Genoa)

**通讯引用:** 16638 | [OpenAlex ID](https://openalex.org/A5065359946)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Hyperbolic Prompt Espial（使用超曲空间 SVDD 进行异常检测）和 Hyperbolic Prompt Sanitization（基于可解释归因的词汇修正），为 VLM 的提示语安全提供轻量化、可解释的防护方案。

**💡 创新点**

创新点在于：① 将超曲空间几何结构与 SVDD 相结合，仅需学习半径参数即可构造安全边界；② 通过层积分梯度可解释性归因定位有害词汇，并采用词典+LLM 的增量替换策略，兼顾语义保留与安全性。

**🔧 技术方法**

技术包括：超曲（Lorentz）嵌入、超曲 SVDD（HSVDD）、Layer Integrated Gradients、词典反义词替换、LLM 辅助安全词替换、Stable Diffusion 与 CLIP 结合的下游评估。

**📊 数据集**

使用的主要数据集有 ViSU、MMA、SneakyPrompt、COCO、I2P*、NSFW56K、UnsafeBench，以及自制的 adversarial 与 adaptive 攻击数据集。

**📈 对比分析**

与 NSFW Classifier、DiffGuard、Detoxify、LatentGuard、GuardT2I 等现有检测器对比，Hyperbolic Prompt Espial 在所有测试集上均获得最高 F1 分数（如 ViSU 0.98，MMA 0.95），并在多种对抗攻击（MMA、SneakyPrompt-RL、StyleAttack、自定义 adaptive attack）中保持高召回/精度；其 Sanitization 模块在 65%–85% 的有害提示被中和的同时，语义相似度保持在 0.82–0.87 之间。

**⚠️ 局限性**

局限性包括：① 仅在已训练的超曲文本编码器上有效，若改用其他编码器需重新学习安全边界；② 对极端对抗攻击（λ→1）仍可能被规避；③ 词汇替换虽减少有害内容，但在某些情境下仍可能导致语义偏差；④ 需要大量无害提示用于训练，且对多语言/跨域场景的适应性尚待验证。

---

## 3. "Help Me, But Don't Watch Me": Intervention Timing and Privacy Boundaries for Process-Aware AI Tutors

**arXiv ID:** 2604.06178 | [PDF](https://arxiv.org/pdf/2604.06178v1)

**作者:** Jane Hanqi Li `[一作]` (University of California San Diego), Amy Eguchi `[通讯]` (University of California San Diego)

**通讯引用:** 1472 | [OpenAlex ID](https://openalex.org/A5055575953)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对中国中学生使用生成式AI辅助数学学习的期望进行问卷调查，探讨他们对AI辅导时机、干预方式及隐私边界的偏好。

**💡 创新点**

首次系统量化学生对过程感知型AI辅导的接受度、干预偏好与隐私界限的关系，揭示学生对“自主”与“及时干预”之间的权衡。

**🔧 技术方法**

采用基于问卷的自报告量表与轻量主题分析，结合描述性统计与序数Logit回归，评估学生对干预适度性、帮助性、干扰性与自主性的感知。

**📊 数据集**

共收集330名中国中学生（7–11年级）匿名在线问卷数据，包含干预偏好、隐私边界和过程感知型干预接受度等变量。

**📈 对比分析**

与传统的“只在学生请求时干预”模型相比，模型显示对适应性与有用性的正向预测及对干扰性的负向预测，表明学生更愿意接受渐进式、低干扰的主动干预；然而，本研究未涉及实际系统实现或学习成效对比。

**⚠️ 局限性**

样本局限于中国地区的中学生，数据仅为自报偏好，未验证真实使用情境；过程感知型干预的具体实现细节与实际学习效果仍待进一步实验验证。

---

## 4. Reproducibility Beyond Artifacts: Interactional Support for Collaborative Machine Learning

**arXiv ID:** 2604.06414 | [PDF](https://arxiv.org/pdf/2604.06414v1)

**作者:** Zhiwei Li `[一作]` (University of Southern California), Carl Kesselman `[通讯]` (Information Sciences Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在临床机器学习项目中部署并评估了一个双层社会技术管理系统，结合了结构化的 Deriva-ML 资产追踪与 AI 辅助的交互层。

**💡 创新点**

提出了将可重现性视为持续的交互工作而非仅靠静态痕迹的观念，并设计了 AI 介入的交互层以支持团队对实验意图的解释、协调与共享。

**🔧 技术方法**

使用 Deriva-ML 进行数据与实验生命周期管理，利用 Claude 大语言模型和 Model Context Protocol（MCP）实现 AI 交互代理。

**📊 数据集**

在 EyeAI 眼科研究项目中使用了约 80 个临床数据集、130 次实验执行和 3 种模型类型进行部署。

**📈 对比分析**

未进行系统化的性能对比，AI 代理仅在小规模数据处理、特征生成和模型微调配置等任务上做过功能演示，尚缺乏大规模纵向评估。

**⚠️ 局限性**

局限性包括仅在单一跨学科项目中测试、交互层仍处于原型阶段、缺乏长期实测数据以及对团队动态的评估不足。

---

## 5. Front-End Ethics for Sensor-Fused Health Conversational Agents: An Ethical Design Space for Biometrics

**arXiv ID:** 2604.06203 | [PDF](https://arxiv.org/pdf/2604.06203v1)

**作者:** Hansoo Lee `[一作]` (Imperial College London), Rafael A. Calvo `[通讯]` (Imperial College London)

**通讯引用:** 15136 | [OpenAlex ID](https://openalex.org/A5013835523)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向传感器融合健康会话代理的前端伦理设计空间，聚焦于生物信号翻译的安全性与可解释性。

**💡 创新点**

创新点在于将伦理设计从后端模型准确性转向前端翻译体验，构建五维伦理框架并引入自适应披露与可争议机制。

**🔧 技术方法**

使用大型语言模型（LLM）与传感器数据融合技术，并在设计层面引入基于用户状态的自适应界面与不确定性标注。

**📊 数据集**

本文未基于具体数据集，而是以现有健康传感器与LLM技术为假设场景进行理论分析和设计推导。

**📈 对比分析**

缺乏实验比较与性能评估，主要通过案例分析与伦理风险模型阐释提出方案的潜在安全收益。

**⚠️ 局限性**

局限在于未进行实证验证，缺少用户实验数据来评估自适应披露与争议机制的实际效果与可行性。

---

## 6. Knowledge Markers: An AI-Agnostic Concept for the Design of Programming Courses

**arXiv ID:** 2604.06331 | [PDF](https://arxiv.org/pdf/2604.06331v1)

**作者:** Christina Maria Mayr `[一作]` `[通讯]` (Hochschule München University of Applied Sciences), Christina Maria Mayr (Hochschule München University of Applied Sciences)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并应用基于A/S/P三类知识标记的课程单元标记方法，用于结构化编程教学并在一门课程中重新设计和评估课程材料。

**💡 创新点**

引入轻量化、AI无关的知识标记，为课程设计提供可操作的中间层，使学习意图可视化并指导AI辅助学习。

**🔧 技术方法**

结合Krathwohl修订分类法、课程内容注解、Web交互式学习网站与PDF脚本等教学工件进行实现。

**📊 数据集**

采用本课程（L1171）现有教材的章节与页面、课程大纲和学生自学材料作为标记与评估的基础。

**📈 对比分析**

通过对标记前后章节的页数与章节数分布进行描述性统计，验证课程改版后应用与程序化知识占比提升，但未进行学习成绩测评。

**⚠️ 局限性**

标记方法粗略、需统一解释，缺乏实验验证学习效果，且仅提供定性描述，未覆盖学习成绩和工具使用的全面评估。

---

## 7. From Load Tests to Live Streams: Graph Embedding-Based Anomaly Detection in Microservice Architectures

**arXiv ID:** 2604.06448 | [PDF](https://arxiv.org/pdf/2604.06448v1)

**作者:** Srinidhi Madabhushi `[一作]` (Amazon Prime Video), Yegor Silyutin `[通讯]` (Amazon Prime Video)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并应用基于图卷积自编码器（GCN‑GAE）的无监督结构嵌入方法，对Prime Video微服务架构中的游戏日（load test）和实际事件日图进行对比，实时检测服务级异常。

**💡 创新点**

创新点包括：① 在多快照无序训练中扩展GCN‑GAE，实现对动态服务图的结构学习；② 使用余弦相似度对节点级嵌入进行异常评分；③ 引入合成异常注入框架进行可控评估，并结合实际CoE、RCA和SPOC验证结果，提升解释性与操作可用性。

**🔧 技术方法**

核心技术包括：图卷积网络（GCN）、图自编码器（GAT‑GAE）、余弦相似度度量、基于TensorFlow的批量训练、以及Amazon EMR/SageMaker等云原生数据处理与模型部署工具。

**📊 数据集**

使用Prime Video内部的时间序列遥测数据（每分钟服务间TPS），构成基线、事件（如TNF）和游戏日三类动态图谱；对测试数据集进行5个月的收集与归档。

**📈 对比分析**

与传统链接预测和时间序列模型（如DySAT）对比，GCN‑GAE在检测游戏日异常时实现了约96%精度、0.08%误报率，提前1–3分钟识别到CoE中列出的故障服务；在合成注入实验中，召回率为58%，反映了传播假设的限制。

**⚠️ 局限性**

主要局限包括：① 召回率受限于仅识别直接关联节点，无法覆盖长距离依赖；② 需要更深层模型或规则结合以捕捉全局传播；③ 合成注入假设过于保守，实际场景中异常传播模式更复杂；④ 模型解释性仍不够，需要进一步融合业务上下文与部署信息。

---

## 8. Towards Resilient Intrusion Detection in CubeSats: Challenges, TinyML Solutions, and Future Directions

**arXiv ID:** 2604.06411 | [PDF](https://arxiv.org/pdf/2604.06411v1)

**作者:** Yasamin Fayyaz `[一作]` (Ontario Tech University), Khalil El-Khatib `[通讯]` (Ontario Tech University)

**通讯引用:** 2906 | [OpenAlex ID](https://openalex.org/A5088109450)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述CubeSat的网络安全现状、挑战与现有IDS方法，首次系统性探讨TinyML在CubeSat上实现低功耗、实时入侵检测的可行性，并提出未来研究方向与框架。

**💡 创新点**

将TinyML技术与CubeSat特有的资源限制（低功耗、有限计算、带宽受限）结合，首次提出面向CubeSat的轻量化模型搜索（NAS/CASH）、自适应学习（TinyTL/TinyOL）、联邦学习与自动特征选择等完整TinyML技术栈，并在此基础上构建主机级IDS工作流。

**🔧 技术方法**

技术包括：TinyML（pruning、quantization、clustering、NAS、CASH）、Transfer & Continual Learning（TinyTL、TinyOL）、Federated Learning（TinyFedTL）、Automated Feature Selection（AutoFS）、Quantization‑Aware Training（QAT）以及结合物理状态的物理‑信息融合模型。

**📊 数据集**

主要数据集为：OPS‑SAT（磁力计、光电计测距数据）、BIRDS‑3/4（太阳能板电流/电压）、EduSat（电源管理传感器）、LabSat（仿真健康管理数据）以及通用网络入侵数据集CICIDS2017（改编为Space Packet协议）等。

**📈 对比分析**

论文未实现完整实验，对比基于先前研究的性能指标：SOM‑CBR检索成功率94.4%，LDA F1≈85%，CNN F1≈100%，Random Forest（OPS‑SAT）准确率≈98.4%，RNN‑LSTM≈95.3%，并引用已验证的3U CubeSat IDS的F1‑score分别为99.59%、90.23%、87.66%。这些结果表明TinyML技术在理论上可达到高检测准确率，但实际部署在CubeSat上尚无实测数据。

**⚠️ 局限性**

局限性包括：缺乏真实任务环境下的实验与评估；模型采用静态离线训练，难以应对概念漂移和新型攻击；在太空严苛环境（辐射、温度、EMI）下的鲁棒性待验证；压缩与优化导致的精度下降；以及在极低功耗与极限内存下实现自适应学习和联邦学习的实现难度高。

---

## 9. Does a Global Perspective Help Prune Sparse MoEs Elegantly?

**arXiv ID:** 2604.06542 | [PDF](https://arxiv.org/pdf/2604.06542v1)

**作者:** Zeliang Zhang `[一作]` (University of Rochester), Xiaodong Liu `[通讯]` (Microsoft Research)

**通讯引用:** 14832 | [OpenAlex ID](https://openalex.org/A5100374791)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对稀疏混合专家（MoE）模型的剪枝问题，提出了一种全局剪枝策略GRAPE，能够根据跨层冗余动态分配剪枝预算，逐层合并最相似的专家并通过熵正则避免过度集中在少数层。

**💡 创新点**

创新点在于：①首次量化并利用跨层冗余信息；②设计了基于熵的停止条件和重启机制，保证全局预算既被充分利用又不导致单层过度裁剪；③在全局框架下实现了对专家的结构化剪枝。

**🔧 技术方法**

技术包括：利用CKA或均方误差等相似度度量构建专家相似度矩阵；构造块对角相似度矩阵并以贪婪方式最小化相似度之和；引入全局熵正则与熵阈值（γ）实现剪枝平衡；实现一次性（one‑shot）全局剪枝与重启机制。

**📊 数据集**

评估使用了 Mixtral‑8x22B、Deepseek‑MoE‑16B、GPT‑oss 等三大 MoE 语言模型，分别在 MMLU、BoolQ、OpenBookQA、RTE 等标准 NLP 评测集上进行实验。

**📈 对比分析**

与四种基于本地（层级）剪枝的基线（Router‑guided、Count‑guided、Enumerate、DEK）相比，GRAPE 在相同总剪枝预算下均能获得更高的平均准确率，最高提升达 2.45%（Mixtral‑8x22B），在 Deepseek‑MoE 与 GPT‑oss 上亦保持 1–2% 的优势。

**⚠️ 局限性**

局限性在于：当某些层的冗余度异常高时，全局剪枝可能导致该层被过度裁剪而引起模型崩溃；此外，目前的冗余度量与熵阈值需要手工调参，缺乏自动化选择机制。

---

## 10. Robustness Risk of Conversational Retrieval: Identifying and Mitigating Noise Sensitivity in Qwen3-Embedding Model

**arXiv ID:** 2604.06176 | [PDF](https://arxiv.org/pdf/2604.06176v1)

**作者:** Weishu Chen `[一作]` (Beijing University of Posts and Telecommunications), Fei Su `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 4335 | [OpenAlex ID](https://openalex.org/A5101754632)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在真实会话检索场景中，对Qwen3嵌入模型进行实验，探究其对结构化对话噪声的鲁棒性。

**💡 创新点**

发现Qwen3在无查询提示下对结构化噪声极为敏感，并提出轻量级查询提示能有效抑制噪声并恢复检索稳定性，且此问题在干净查询基准中难以察觉。

**🔧 技术方法**

使用密集检索嵌入模型（Qwen3及对照模型），对噪声注入、NDCG@k、噪声排名等指标进行评估，并引入轻量级提示策略。

**📊 数据集**

实验基准为LongMemEval、LoCoMo等对话检索数据集，结合人工构造的对话填充和系统级噪声语料。

**📈 对比分析**

通过对不同噪声比例、模型规模、记忆包装以及提示与否的系统比较，结果显示无提示时Qwen3 NDCG@5大幅下降，噪声占据前位；提示后性能恢复并超越干净基准；其他模型表现相对稳定。

**⚠️ 局限性**

限制在于噪声模板覆盖有限，未覆盖更复杂的生产环境噪声；难以精确归因Qwen3对噪声敏感的具体原因；实验主要集中在Qwen3族，其他检索模型未完全检验。

---

## 11. TelcoAgent-Bench: A Multilingual Benchmark for Telecom AI Agents

**arXiv ID:** 2604.06209 | [PDF](https://arxiv.org/pdf/2604.06209v1)

**作者:** Lina Bariah `[一作]` (Khalifa University), Merouane Debbah `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TelcoAgent-Bench与TelcoAgent-Metrics框架，用于评估多语言电信LLM代理在意图识别、工具调用、流程一致性与稳定性方面的表现。

**💡 创新点**

创新点在于构建专属电信意图税onomies与蓝图驱动的双语对话数据集，并设计多维度评估指标（IRA、SAS、RA、BRS）以量化代理在真实故障排除情境中的可靠性。

**🔧 技术方法**

使用语义相似度嵌入、最长公共子序列、Levenshtein距离等算法，配合预训练句子嵌入模型评估意图识别与总结准确度，并在核心工具与干扰工具环境下进行工具调用评测。

**📊 数据集**

采用约1470条双语对话的TelcoAgent-Bench数据集，涵盖15类意图、49个蓝图、30条样例，并提供工具调用序列与金标准总结。

**📈 对比分析**

对Llama-3-8B、Qwen系列、Granite、Gemma、Mistral等多款LLM在英阿两语环境下计算IRA、SAS、RA、BRS等指标，Granite-3.3-8B在多数指标上表现最佳，但整体仍未达到完美执行。

**⚠️ 局限性**

局限性在于未涵盖闭环操作（工具输出解释、配置变更、再评估），模型在严格遵循预定工具流程、稳定性与多语言一致性方面仍存在不足，且对干扰工具的识别仍易出错。

---

## 12. In-Context Learning in Speech Language Models: Analyzing the Role of Acoustic Features, Linguistic Structure, and Induction Heads

**arXiv ID:** 2604.06356 | [PDF](https://arxiv.org/pdf/2604.06356v1)

**作者:** Charlotte Pouw `[一作]` (University of Amsterdam), Willem Zuidema `[通讯]` (University of Amsterdam)

**通讯引用:** 4129 | [OpenAlex ID](https://openalex.org/A5007928903)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 SpeechLM 在文本到语音（TTS）任务中的上下文学习（ICL）表现，并系统分析了语言和声学特征以及诱导头的作用。

**💡 创新点**

首次在 SpeechLM 中评估声学与语言特征对 ICL 的影响，并揭示诱导头在语音 ICL 中的因果角色。

**🔧 技术方法**

使用 SpiritLM（Llama‑2‑7B + HuBERT + HiFi‑GAN）实现 TTS，借助 Praat 进行声学操控，利用 Whisper 评估输出，并通过前缀匹配得分与消融实验识别诱导头。

**📊 数据集**

基于 Prime‑LM 语料库，并使用 SpeechT5 与 Kokoro‑TTS 合成演示语音。

**📈 对比分析**

通过词错误率（WER）和内容词召回率比较任务准确性；通过说话速率、音高范围、强度等声学特征比较输出与演示的一致性；实验显示说话速率显著影响 ICL，消融诱导头导致性能大幅下降。

**⚠️ 局限性**

局限在于仅考察有限的声学维度和单一模型规模，缺乏对更大语料和多模态的泛化验证，且未深入探讨更长范围的前缀匹配行为。

---

## 13. Zero Trust in the Context of IoT: Industrial Literature Review, Trends, and Challenges

**arXiv ID:** 2604.06272 | [PDF](https://arxiv.org/pdf/2604.06272v1)

**作者:** Laurent Bobelin `[一作]` `[通讯]` (INSA Centre Val de Loire), Laurent Bobelin (INSA Centre Val de Loire)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2023年工业界将物联网（IoT/SPD）集成到Zero Trust架构的主要方案进行文献综述与分类评估。

**💡 创新点**

提出了基于厂商支持度的分类框架，揭示了行业在用户无设备、低资源、棕色场景、短生命周期、移动性与异构性等方面的挑战，并指出当前解决方案在信任度建模与策略一致性方面的空白。

**🔧 技术方法**

主要使用的技术方法为：系统性文献检索（非学术出版物）、关键词过滤、功能特征归纳（Agentless扫描、资产发现、数字孪生、硬件网关、SDK等）以及对比分析；并结合NIST标准与Google BeyondCorp等参考架构进行对照。

**📊 数据集**

使用的数据集为：公开的工业厂商列表（如Microsoft Azure、AWS、Palo Alto Networks等）、各厂商白皮书、技术文档、行业排名与市场份额信息；无实验性数据集。

**📈 对比分析**

比较方法：对每个厂商按功能维度（agentless、数字孪生、硬件支持、绿/棕色场景覆盖、移动性/互操作性）进行打分与归类；未提供量化性能指标（如吞吐量、延迟、误报率），只能通过功能完整度与文档支持度进行主观对比。

**⚠️ 局限性**

限制：仅基于2023年公开资料，缺乏实时实验验证；对技术实现细节与性能评估缺乏客观数据；行业快速演进导致结论易失效；文献筛选偏向大型厂商，可能忽略中小型创新方案；未涵盖所有标准化组织与学术研究的最新进展。

---

## 14. Illocutionary Explanation Planning for Source-Faithful Explanations in Retrieval-Augmented Language Models

**arXiv ID:** 2604.06211 | [PDF](https://arxiv.org/pdf/2604.06211v1)

**作者:** Francesco Sovrano `[一作]` (ETH Zurich), Alberto Bacchelli `[通讯]` (University of Zurich)

**通讯引用:** 6993 | [OpenAlex ID](https://openalex.org/A5082720005)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在编程教育场景下，评估并提升检索增强生成模型（RAG）对教材的源忠实度。

**💡 创新点**

提出基于Achinstein语用学解释理论的链式语用提示（chain-of-illocution）策略，通过显式生成隐含解释性问题并检索证据来提升源忠实度。

**🔧 技术方法**

使用检索增强生成（RAG）与多种大型语言模型（GPT‑3.5‑turbo、GPT‑4o、Llama‑3 8B/70B、Mistral 7B、Mixtral 8×7B），并实现链式语用提示和基于句子相似度检索。

**📊 数据集**

采用公开教材（Java、Python、Pharo）以及90个热门 Stack Overflow 问题作为评测数据集。

**📈 对比分析**

通过对比 RAG 与 chain-of-illocution 版本，使用 FActScore、语义相似度等指标，发现链式提示平均提升约34%的源忠实度（最高可达63%），但绝对忠实度仍在中等偏低水平。

**⚠️ 局限性**

局限性包括：提升后仍未达到高忠实度，模型对检索证据的依赖不一致，实验仅覆盖编程教材，且对用户满意度的影响在统计上不显著。

---

## 15. VLMShield: Efficient and Robust Defense of Vision-Language Models against Malicious Prompts

**arXiv ID:** 2604.06502 | [PDF](https://arxiv.org/pdf/2604.06502v1)

**作者:** Peigui Qi `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 23636 | [OpenAlex ID](https://openalex.org/A5064573190)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于CLIP的多模态聚合特征提取框架MAFE，并在此基础上构建轻量级安全检测器VLMShield，能够有效识别VLM的多模态恶意提示攻击。

**💡 创新点**

创新点在于：①利用MAFE将长文本与图像信息融合为统一向量，揭示了良善与恶意提示在特征空间中的分布差异；②在此特征空间上训练了极简的三层全连接网络，实现了高效、鲁棒的恶意检测；③通过进阶文本聚合和加权平均提高了长文本处理能力。

**🔧 技术方法**

核心技术包括CLIP的文本/图像编码、重叠分块的进阶文本聚合、跨模态特征拼接、最大均值差异（MMD）分析、以及基于交叉熵的三层神经网络训练。

**📊 数据集**

使用了多份公开数据集：GPT4V-Caption、CC3M、MM-SafetyBench、JailbreakV_28k（图像与文本版）、AdvBench_M 等，构成约44,400条带标签的训练集及多种在域/外域的测试集。

**📈 对比分析**

与内部防御ASTRA、VLMGuard以及外部防御JailGuard、CIDER、MirrorCheck、SelfReminder、ECSO等基线对比，VLMShield在LLaVA和Qwen VLM上实现了0.00–2.13%的攻击成功率，99.84–100%的正常准确率，检测延迟仅0.34 s，整体占用时间提升不足10%，显著优于所有对比方法。

**⚠️ 局限性**

局限性包括：对极端稀疏恶意信息（极高比例的干扰文本）仍可能导致误报；在完全白盒适应攻击下最大有效攻击成功率仍可达≈4%；此外模型训练依赖于已有攻击数据，可能在未知攻击策略下表现下降。

---

## 16. Accurate Residues for Floating-Point Debugging

**arXiv ID:** 2604.06258 | [PDF](https://arxiv.org/pdf/2604.06258v1)

**作者:** Yumeng He `[一作]` (University of Utah), Pavel Panchekha `[通讯]` (University of Utah)

**通讯引用:** 1053 | [OpenAlex ID](https://openalex.org/A5022031348)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于残差的浮点调试框架，改进了误差估计和残差函数，并通过多次执行的残差覆盖技术消除吸收导致的误差，显著降低误报率。

**💡 创新点**

创新点在于将残差计算拆分为误差估计和残差函数两步，修正关键公式，并引入吸收检测与残差覆盖机制，使同一程序在多次执行中分别测得不同残差并最终合成，突破了传统单次执行下的精度瓶颈。

**🔧 技术方法**

采用误差无误差变换（error‑free transformations）实现高效误差估计，设计精确的残差函数；通过识别最大/次大误差贡献者、沉默关键运算、探测残差并在覆盖运行中合并结果，形成完整的残差覆盖算法。

**📊 数据集**

使用 NAS Parallel Benchmarks、Rodinia、Polybench 以及 FPBench（来自数值分析教材和论文的 130+ 表达式）等标准科学计算基准，涵盖浮点库函数和复杂数值场景。

**📈 对比分析**

与 MPFR（高精度基准）、QD（四倍双精度）以及现有 EFTSanitizer 等工具对比；误报率平均降低约 30–38 倍，误差覆盖后的误报率仅为 1% 左右；运行开销约为 uninstrumented 的 10×，低于 QD 与 MPFR，且在绝大多数基准上无额外重跑。

**⚠️ 局限性**

局限性包括：仍需多次执行（平均 7.1 次）才能覆盖复杂吸收情况，导致调试时间略增；对极其复杂的库实现仍可能产生剩余误报；当前吸收检测阈值经验性，需要进一步理论支持。

---

## 17. Time-Series Classification with Multivariate Statistical Dependence Features

**arXiv ID:** 2604.06537 | [PDF](https://arxiv.org/pdf/2604.06537v1)

**作者:** Yao Sun `[一作]` (University of Florida), Jose Principe `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于功能最大相关算法(FMCA)的非平稳时序分类框架。

**💡 创新点**

创新点是使用交叉密度比(CDR)直接估计输入与目标的联合分布依赖，并通过FMCA构建无窗口、无序列依赖的特征空间。

**🔧 技术方法**

采用FMCA、核方法、FFT频域预处理以及单隐藏层MLP分类器。

**📊 数据集**

使用TI-46孤立数字语音数据集。

**📈 对比分析**

与HMM、尖峰神经网络等方法对比，准确率高达99.39%，显著优于其他轻量化模型，且训练时间仅10分钟。

**⚠️ 局限性**

局限在于需手动调参（窗口大小、步幅、特征维度等），对新数据集迁移仍需重调，并未探索复数域信息。

---

## 18. Limits of Difficulty Scaling: Hard Samples Yield Diminishing Returns in GRPO-Tuned SLMs

**arXiv ID:** 2604.06298 | [PDF](https://arxiv.org/pdf/2604.06298v1)

**作者:** Suraj Yadav `[一作]` (IIIT Delhi), Parth Goyal `[通讯]` (IIIT Delhi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在资源受限的环境下，对 0.5B–3B 规模的小语言模型使用 GRPO+LoRA 进行数学推理任务的对齐训练，并通过难度分层分析评估其性能。

**💡 创新点**

提出难度分层训练策略，发现小模型存在“容量边界”，高难度样本对提升性能无益，低难度样本即可获得与全数据相同的效果，并揭示跨数据集更易数据的迁移优势。

**🔧 技术方法**

采用 Group Relative Policy Optimization（GRPO）配合 Low‑Rank Adaptation（LoRA）实现参数高效微调，并在奖励函数中动态加权难度级别。

**📊 数据集**

使用 GSM8K（小学算术）和 Hendrycks MATH（四个核心领域）的子集，并将 MATH 按难度划分为 L1–L5。

**📈 对比分析**

通过与 SFT 基线对比，在不同难度层和模型规模下评估准确率，发现仅训练低难度数据即可匹配全数据结果；且 GSM8K 训练的模型在 MATH 数值子集上比 MATH 训练模型高约 5%（1.5B）或 3%（3B）。

**⚠️ 局限性**

实验仅涵盖 0.5B–3B 规模的模型，难度过高的样本会产生高方差梯度导致学习不稳定；结论可能不适用于 70B+ 超大模型或非数学类数据集。

---

## 19. X-BCD: Explainable Sensor-Based Behavioral Change Detection in Smart Home Environments

**arXiv ID:** 2604.06174 | [PDF](https://arxiv.org/pdf/2604.06174v1)

**作者:** Gabriele Civitarese `[一作]` (University of Milan), Claudio Bettini `[通讯]` (University of Milan)

**通讯引用:** 6924 | [OpenAlex ID](https://openalex.org/A5010533347)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于多模态智能家居传感器的可解释行为变化检测框架，能够在无监督条件下识别并生成临床可理解的自然语言描述。

**💡 创新点**

创新点在于将无监督变更点检测与聚类演化追踪结合，并通过可解释特征与大型语言模型生成自然语言解释，解决传统方法仅给出漂移分数而缺乏结构化行为变化解释的问题。

**🔧 技术方法**

所用技术包括 PELT 变更点检测、DBSCAN 密度聚类、基于中心点与方差的聚类相似度评估、阈值匹配构造演化图以及 MedGemma 27B 语言模型进行聚类描述与变化解释。

**📊 数据集**

实验使用了来自 17 名真实 MCI 患者的长达 12–30 个月的多模态传感器数据（睡眠分析仪、智能手表、温湿度/动作传感器等），并将患者分为神经退行性（D）和非神经退行性（ND）两组。

**📈 对比分析**

通过对两组的变更点数量、无稳定性惯量质量等群体层面指标进行比较，并结合专家评估和参数敏感性分析，模型能够生成可信的自然语言描述，并在不同阈值下保持稳健性。

**⚠️ 局限性**

限制包括缺乏标注基准和大规模样本，无法在个体层面评估诊断效能；模型依赖手工设定参数，对异常或季节性因素的区分有限，且尚未实现在线实时检测。

---

## 20. SALLIE: Safeguarding Against Latent Language & Image Exploits

**arXiv ID:** 2604.06247 | [PDF](https://arxiv.org/pdf/2604.06247v1)

**作者:** Guy Azov `[一作]`, Guy Shtar `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用视觉‑语言模型内部多层激活进行恶意查询检测的框架。

**💡 创新点**

创新点在于将k‑NN相似度与多层激活特征相结合，采用无监督平均分数并通过阈值做二分类。

**🔧 技术方法**

使用视觉‑语言模型（SVLM）作为特征提取器，k‑NN分类器，余弦相似度计算和阈值决策。

**📊 数据集**

基于公开的图文检索数据集（如COCO/Visual Genome）构建训练数据库，同时采样恶意与正常查询进行实验。

**📈 对比分析**

相较于传统基于文本或图像特征的检测方法，检测准确率提升，误报率降低。

**⚠️ 局限性**

局限性包括对不同模型架构的通用性有限，阈值需手工调优，对极端对抗样本可能仍有漏检。

---

## 21. Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR

**arXiv ID:** 2604.06487 | [PDF](https://arxiv.org/pdf/2604.06487v1)

**作者:** Thibault Bañeras-Roux `[一作]` (Idiap Research Institute), Andreas Stolcke `[通讯]` (Uniphore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文针对目标域语音稀缺但文本充足的场景，研究了LLM驱动的ASR系统的域适配方法，并提出了混合批次（Mixed Batch）训练策略，将少量目标域语音与大量目标域文本同时混入训练批次中；

**💡 创新点**

创新点在于通过在每个训练批次中同时加入少量目标域语音、目标域文本及源域辅助数据，弥补纯文本微调产生的模态不匹配；同时提出了在不冻结LLM的情况下，仅更新LoRA模块即可实现参数高效的域适配；

**🔧 技术方法**

使用SLAM-ASR框架，采用预训练的wav2vec2.0/Hubert或WavLM语音编码器、线性投影器、Meta‑Llama‑3.2‑3B‑Instruct LLM；在适配阶段冻结语音编码器和投影器，只更新Llama中的LoRA适配层；

**📊 数据集**

实验数据集为DefinedAI（Banking、Insurance、Healthcare）作为源域，SlideSpeech（Agriculture、Musical Instruments）作为目标域，覆盖从几小时到数十小时的语音与文本；

**📈 对比分析**

比较了三种适配策略：仅文本、仅语音、混合批次；结果显示，即使只使用10%目标域语音（<4小时）与大量文本，混合批次的WER可与全量语音训练相当或更优（如Banking域从6.38%降至4.55%），并在多目标域均表现出更低的灾难性遗忘；

**⚠️ 局限性**

局限性包括：仍需源域辅助数据以防灾难性遗忘，批次比例调优需要经验；实验仅涵盖有限的行业域，未验证对更广泛领域的适用性；以及对LLM推理成本和内存占用的关注。

---

## 22. SELFDOUBT: Uncertainty Quantification for Reasoning LLMs via the Hedge-to-Verify Ratio

**arXiv ID:** 2604.06389 | [PDF](https://arxiv.org/pdf/2604.06389v1)

**作者:** Satwik Pandey `[一作]` (Independent Researcher), Shashwat Pandey `[通讯]` (Zillow Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种单次推理时即可估计大型语言模型不确定性的框架 SelfDoubt，基于推理轨迹中的 “hedge” 与 “verify” 词频比值及模型自述置信度。

**💡 创新点**

通过自适应地从未标记推理轨迹中学习模型专属的 “hedge” 与 “verify” 词典，提出 Hedge-to-Verify Ratio（HVR）并与 verbalized confidence 通过 z-score 融合，形成无训练、无内部访问、低成本的 UQ 方法。

**🔧 技术方法**

使用正则表达式词典匹配、嵌入向量相似度筛选、z-score 标准化融合、无监督词典扩展与基准实验等技术。

**📊 数据集**

在 BBH、GPQA‑Diamond 和 MMLU‑Pro 三个多步推理多选题集上进行评估。

**📈 对比分析**

与多种 O(1)（如 verbalized confidence、trace length）和 O(N)（如 Semantic Entropy）基线对比，SelfDoubt 在 21 个模型×数据集运行中平均 AUROC 0.7895、AURAC 0.8992，超越 SE（p=0.001）且成本低 10 倍，部署级联可达 71% 覆盖率、89.7% 准确率。

**⚠️ 局限性**

依赖模型轨迹中 hedge 词出现频率，稀疏摘要导致门控无效；对写作风格敏感；仅提供单一不确定性评分；仅在多选任务上验证，未探讨自由生成场景。

---

## 23. From experimentation to engagement: on the paradox of participatory AI and power in contexts of forced displacement and humanitarian crises

**arXiv ID:** 2604.06219 | [PDF](https://arxiv.org/pdf/2604.06219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 24. Continual Visual Anomaly Detection on the Edge: Benchmark and Efficient Solutions

**arXiv ID:** 2604.06435 | [PDF](https://arxiv.org/pdf/2604.06435v1)

**作者:** Manuel Barusco `[一作]` (University of Padova), Gian Antonio Susto `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出针对边缘设备的持续视觉异常检测（VAD）基准，并设计了三种轻量化方案（Tiny‑Dinomaly、PatchCoreCL++、PaDiM‑CL Lite多模态）以兼顾检测精度、内存占用和推理成本。

**💡 创新点**

创新点：①首次系统评估连续学习与边缘部署共存的VAD方法；②提出 Tiny‑Dinomaly 在保持甚至提升像素级 F1 的同时，显著压缩 13× 参数和 20× FLOPs；③通过前向原型识别和 k‑center 前缀截断改进 PatchCoreCL，降低 12× 推理成本；④纠正 PaDiM‑CL 的均值/协方差融合误差并引入轻量化多模态版本。

**🔧 技术方法**

采用轻量化 CNN / Vision Transformer backbone（MobileNetV2、MCUNet、PhiNet、DeiT‑Tiny），Replay 缓冲、PatchCoreCL++ 的前向原型匹配、PaDiM 的多模态高斯融合、Dinomaly 的线性注意力与噪声瓶颈等技术。

**📊 数据集**

使用 MVTec AD 与 VisA 两大工业缺陷图像数据集，覆盖 27 类目标与 70+ 缺陷类型。

**📈 对比分析**

在像素级 F1、图像级 F1、内存（MB）和 GFLOPs 维度下与 FT、JT、其他现有 VAD 方法（PatchCore、PaDiM、CFA、STFPM、PaSTe、SimpleNet 等）进行公平对比；Tiny‑Dinomaly 在像素 F1 0.45、FLOPs 3.39、内存 50.87 MB 的组合上超越原版 Dinomaly；PatchCoreCL++ 在图像 F1 0.96、像素 F1 0.41 的同时将推理成本从 49.16 GFLOPs 降至 4.15 GFLOPs；PaDiM‑CL Lite 多模态在像素 F1 0.41、FLOPs 0.57、内存 46.67 MB 上实现最佳平衡。

**⚠️ 局限性**

局限性：①仅在固定任务数与 224×224 分辨率上验证，尚未评估更长序列或更高分辨率；②Replay 缓冲仍需额外存储，虽然 40/100 样本已足够，但对极端受限设备仍有挑战；③轻量化 Transformer 仍依赖大规模预训练，导致迁移学习的前期成本；④多模态 PaDiM 在内存受限时仍不可行，需进一步压缩。

---

## 25. AE-ViT: Stable Long-Horizon Parametric Partial Differential Equations Modeling

**arXiv ID:** 2604.06475 | [PDF](https://arxiv.org/pdf/2604.06475v1)

**作者:** Iva Mikuš `[一作]` (University of Zagreb), Domagoj Vlah `[通讯]` (University of Zagreb)

**通讯引用:** 90 | [OpenAlex ID](https://openalex.org/A5075917936)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种端到端可训练的AE‑ViT网络，用于参数化PDE的自回归时间演化；

**💡 创新点**

在编码器、Transformer和解码器中引入多阶段参数注入与坐标通道注入，形成全条件化架构，显著提升多场预测精度；

**🔧 技术方法**

采用卷积自编码器压缩空间分辨率，Vision Transformer捕获长程交互，FiLM技术实现参数注入，坐标傅里叶特征提供空间信息，并使用调度采样和梯度裁剪优化训练；

**📊 数据集**

在二维Advection‑Diffusion‑Reaction（32×32 FEM）和Navier‑Stokes流过圆柱（64×320网格）两个基准数据集上进行实验；

**📈 对比分析**

与ViT、DL‑ROM、AE+1D Transformer比较，AE‑ViT在相同训练窗口下的相对回放误差约低5倍，参数量约600万，误差随时间线性增长，表现优异；

**⚠️ 局限性**

局限在于需将解插值至矩形网格，Transformer的二次复杂度高，且对非矩形几何不适用，未来需解决这些瓶颈。

---

## 26. Hybrid ResNet-1D-BiGRU with Multi-Head Attention for Cyberattack Detection in Industrial IoT Environments

**arXiv ID:** 2604.06481 | [PDF](https://arxiv.org/pdf/2604.06481v1)

**作者:** Afrah Gueriani `[一作]` (University of MEDEA), Ahmed Cherif Mazari `[通讯]` (University of MEDEA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种融合ResNet-1D、BiGRU与多头注意力机制的混合深度学习模型，用于工业物联网环境中的网络攻击检测。

**💡 创新点**

创新点在于将空间特征提取、时间序列建模与注意力加权三者整合到同一架构，并结合SMOTE处理类别不平衡，实现了高精度与低推理延迟的统一解决方案。

**🔧 技术方法**

采用了卷积残差网络（ResNet-1D）、双向门控循环单元（BiGRU）、多头注意力（MHA）以及SMOTE过采样、Adam优化器、LabelEncoder等技术。

**📊 数据集**

实验数据集为Edge-IIoTset和CICIoV2024两套工业/车辆物联网网络流量数据。

**📈 对比分析**

通过与LSTM-CNN-Att、BiGRU-LSTM、CNN-LSTM-ViT、XGB等最新方法对比，Edge-IIoTset上实现98.71%准确率、0.0001s/实例推理时间；CICIoV2024上达到99.99%准确率、0.00014s/实例推理时间，并且FPR接近0%。

**⚠️ 局限性**

局限性主要是仅在两套数据集上验证，缺乏更广泛的多场景测试，且模型可解释性和对新型攻击的适应性尚未充分探究。

---

## 27. FedSpy-LLM: Towards Scalable and Generalizable Data Reconstruction Attacks from Gradients on LLMs

**arXiv ID:** 2604.06297 | [PDF](https://arxiv.org/pdf/2604.06297v1)

**作者:** Syed Irfan Ali Meerza `[一作]` (University of Tennessee), Jian Liu `[通讯]` (University of Tennessee)

**通讯引用:** 49085 | [OpenAlex ID](https://openalex.org/A5100414679)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种可扩展且通用的梯度反演攻击，能够在联邦学习与参数高效微调（PEFT）环境下，从大型语言模型的梯度中恢复更大批次、更长序列的训练数据。

**💡 创新点**

创新点包括：①基于梯度低秩与子空间结构的梯度分解策略，显著减少搜索空间；②对PEFT导致的梯度稀疏性引入零空间正则化，降低解的模糊性；③通过梯度对齐的序列顺序校准阶段，恢复正确的 token 顺序；④该方法与模型架构无关，适用于编码器、解码器和编码-解码器三种 Transformer 结构。

**🔧 技术方法**

技术手段包括梯度子空间投影、QR/SVD 分解、加权余弦相似度损失、梯度对齐的序列重排算法，以及在 FL 中对梯度的白盒攻击模型。

**📊 数据集**

实验数据集涵盖 CoLA、Rotten Tomatoes、MIMIC‑III；模型涵盖 GPT‑2、BERT‑Base、Llama‑7B、T5‑Base 等多种 Transformer 架构；PEFT 方法使用 SLoRA 与 FedAdapter。

**📈 对比分析**

与 DLG、TAG、LAMP、APRIL、FILM、BGP、DAGER 等基线相比，该攻击在更大批次（最高 128）和更长序列下依旧保持高 ROUGE‑1/2/L 与实体 F1，尤其在 PEFT 以及对抗 Gaussian 噪声/梯度裁剪的隐私防御下仍能恢复大量文本，性能提升显著（如在 BERT‑Base 上 8‑批次时提升 55%–160%）。

**⚠️ 局限性**

局限性包括：需要访问未加密的梯度，且对高噪声 DP‑GD 更易受限；计算时间仍高（部分实验需数十小时）；对仅训练嵌入层冻结且不公开梯度的模型效果未知；未在所有下游任务与模型规模上做完整验证。

---

## 28. Weakly Supervised Distillation of Hallucination Signals into Transformer Representations

**arXiv ID:** 2604.06277 | [PDF](https://arxiv.org/pdf/2604.06277v1)

**作者:** Shoaib Sadiq Salehmohamed `[一作]` (LLM Lens), Shalmali Ayachit `[通讯]` (LLM Lens)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计弱监督框架，将多种外部对齐信号离线标注生成幻觉标签，构建包含完整Transformer隐藏状态的SQuAD v2数据集，并训练多种探针网络在内部激活上检测幻觉。

**💡 创新点**

首次公开完整隐藏状态-幻觉标签数据集，并证明可通过弱监督将外部验证信号转化为内部表示中的可学习特征，实现推理时无需外部资源的幻觉检测；同时提出多层次探针架构对比。

**🔧 技术方法**

使用子串匹配、MiniLM语义相似度、LLM-judge进行标签生成；利用LLaMA-2-7B生成回答并提取32层×96步×4096维隐藏状态；实现ProbeMLP、LayerWiseMLP、CrossLayerTransformer、HierarchicalTransformer和CrossLayerAttentionTransformerV2等探针。

**📊 数据集**

基于SQuAD v2，生成15,000条样本（10,500训练/验证，5,000测试），每条样本包含完整隐藏状态张量和多信号幻觉标签。

**📈 对比分析**

与传统输出级相似度基线对比，探针在AUC/F1/Acc上显著优于基线；最优模型HierarchicalTransformer在AUC≈0.858、F1≈0.665、Acc≈0.804，ECE最低；推理延迟仅几毫秒，整体吞吐与生成无明显下降。

**⚠️ 局限性**

标签噪声来自弱监督、仅使用贪婪解码、隐藏状态截断至固定长度、仅评估LLaMA-2-7B与SQuAD域、缺乏token级干预实验、部署集成细节未完成。

---

## 29. Language-Guided Multimodal Texture Authoring via Generative Models

**arXiv ID:** 2604.06489 | [PDF](https://arxiv.org/pdf/2604.06489v1)

**作者:** Wanli Qian `[一作]` (University of Southern California), Heather Culbertson `[通讯]` (University of Southern California)

**通讯引用:** 2045 | [OpenAlex ID](https://openalex.org/A5026024402)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个基于自然语言提示的多模态纹理创作系统，能够将文本描述映射为相应的视觉图像、滑动振动和敲击瞬态触觉信号。

**💡 创新点**

通过在共享的语言对齐潜在空间中同时编码滑动振动和敲击信号，并利用 VAE 与 CLIP 的对比学习实现语言驱动的跨模态生成与插值控制，提供了从文本到三种感知模态的完整链路。

**🔧 技术方法**

使用了变分自编码器（VAE）、CLIP 文字编码器、对比学习（InfoNCE）、自回归（AR）触觉模型、扩散模型生成视觉预览以及物理感知的硬件渲染技术。

**📊 数据集**

依托 Penn Haptic Texture Toolkit（HaTT）中 100 种材料的 AR 与 tap 数据，并通过增强技术扩充到 2000 条训练样本进行学习。

**📈 对比分析**

通过人机评估的属性投影、NASA‑TLX、Haptic Experience Inventory 等问卷验证生成纹理在粗糙度、硬度、滑性上的一致性，结果表明生成纹理在属性空间中介于对应锚点之间，且用户满意度高；缺乏直接可比基线。

**⚠️ 局限性**

限制包括：数据覆盖有限（无零样本泛化）、仅支持刚性、各向同性纹理、设备依赖、视觉与触觉解耦、以及对语言变体鲁棒性未系统评估。

---

## 30. Navigating Marginalization: Toward Justice-Oriented Socio-Technical Design for Parent-Child Learning among Southeast Asian Immigrant Mothers in Taiwan

**arXiv ID:** 2604.06353 | [PDF](https://arxiv.org/pdf/2604.06353v1)

**作者:** Ying-Yu Chen `[一作]` (National Cheng Kung University), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**通讯引用:** 1890 | [OpenAlex ID](https://openalex.org/A5054435118)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过半结构访谈和两周日记法，深入探讨了台湾东南亚移民母亲在子女家庭学习中的参与方式、面临的社会文化障碍与情感挑战，并提出以交叉正义为核心的社会技术设计框架，旨在提升移民母亲的认知可见性、家庭互惠与社会问责；

**💡 创新点**

创新点在于将交叉正义（intersectional justice）与社会技术设计相结合，提出“Intersectional Justice-Oriented Design”框架，针对个体、家庭与社会层面分别提供识别、互惠与问责的设计策略，填补了以往研究对移民母亲教育角色的忽视与缺乏结构性干预；

**🔧 技术方法**

采用定性研究技术：半结构访谈、在线日记记录（视频、音频、图片）、主题分析（Braun & Clarke）与研究团队反思笔记；

**📊 数据集**

数据集为14名东南亚移民母亲（主要为越南人）在台长期居住的受访者，包含27小时访谈录音与78段视频、44段音频、385张照片及相应文字记录；

**📈 对比分析**

由于研究为探索性定性研究，并未进行量化比较或性能评估，所述设计方案基于研究发现提出，而非实验验证；

**⚠️ 局限性**

局限性包括样本规模有限且以越南母亲为主，缺乏新近抵台者与其他国籍母亲的视角；仅关注母亲角色，未纳入父亲或其他家庭成员；研究聚焦台湾语境，缺乏跨国或跨文化验证。

---

## 31. Transformer See, Transformer Do: Copying as an Intermediate Step in Learning Analogical Reasoning

**arXiv ID:** 2604.06501 | [PDF](https://arxiv.org/pdf/2604.06501v1)

**作者:** Philipp Hellwig `[一作]` (University of Amsterdam), Martha Lewis `[通讯]` (University of Amsterdam)

**通讯引用:** 9037 | [OpenAlex ID](https://openalex.org/A5102005914)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了小型Transformer模型，采用Meta Learning for Compositionality（MLC）框架，在字母串类比任务上进行学习，并通过引入复制任务和多样化字母集来提升模型的泛化能力。

**💡 创新点**

创新点包括：①将MLC与复制任务结合，以避免模型学习捷径；②使用多样化的字母排列显著提升对新字母系统的泛化；③通过机制解释（Attention patching、PCA等）揭示模型实现类比的算法性步骤；④在小模型上实现了对多种字母串类比的高准确率，甚至超越多种前沿大语言模型。

**🔧 技术方法**

技术手段包括：标准的3层encoder‑decoder Transformer（128维词嵌入，8头注意力，512维前馈），使用学习率0.001、线性衰减、dropout 0.1；Meta‑learning训练采用MLC策略；实验中加入复制任务（示例输入与查询相同），并在不同数量的随机字母排列上训练；利用NNSight进行Attention patching，PCA和余弦相似度分析等机制解释方法。

**📊 数据集**

使用自构造的字母串类比数据集：包含230k（或400k带复制）任务，涵盖多种变换（Extend、Successor、Predecessor、Remove Redundant、Sort、Group、Interleave等）以及新组合和全新变换；字母集在训练中随机排列（从0到20次排列，甚至400个不同排列）。

**📈 对比分析**

实验通过与多款前沿LLM（如GPT‑3/4、Llama、Claude等）进行对比，采用10次复制跑平均准确率，并在in‑distribution、out‑of‑distribution（新字母、新组合、新变换）测试集上评估。最佳MLC模型在新字母和组合变换上的准确率可达90%+，在新变换上仍低于80%；在聚合数据集上，其整体性能优于大多数LLM。

**⚠️ 局限性**

局限性包括：对完全新变换的泛化能力仍有限；需要足够多的字母排列和复制任务才能实现良好泛化；实验仅在小规模Transformer上验证，尚未在大型LLM上测试其可扩展性；任务仅限于简单的字母串，缺乏更复杂语义或知识层面的挑战。

---

## 32. DISSECT: Diagnosing Where Vision Ends and Language Priors Begin in Scientific VLMs

**arXiv ID:** 2604.06250 | [PDF](https://arxiv.org/pdf/2604.06250v1)

**作者:** Dikshant Kukreja `[一作]` (IIIT Delhi), Vikram Goyal `[通讯]` (IIIT Delhi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DISSECT诊断基准，使用五种输入模式评估科学视觉语言模型。

**💡 创新点**

创新点在于Model Oracle两阶段评估，揭示感知-整合缺口并分解模型瓶颈。

**🔧 技术方法**

采用多模态推理与自述式描述的双通道评估技术。

**📊 数据集**

数据集包含7,000道化学与5,000道生物K‑12题目，共12,000问。

**📈 对比分析**

通过18个VLM的五模态测评，开源模型在Model Oracle上提升5–10个百分点，闭源模型几乎无差距。

**⚠️ 局限性**

局限：仅覆盖K‑12教材，使用印度教材措辞，需两次前向推理，未涉及自由文本解释与证明类推理。

---

## 33. Spectral Edge Dynamics Reveal Functional Modes of Learning

**arXiv ID:** 2604.06256 | [PDF](https://arxiv.org/pdf/2604.06256v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

研究Transformer训练过程中出现的光谱边缘（Spectral Edge）方向，并将其解释为低维输入域功能子空间，而非参数或激活空间中的局部结构。

**💡 创新点**

创新点在于：①首次将光谱边缘与输入域的傅里叶/群论基展开相连接；②发现光谱边缘在共享参数的多任务训练中能重用功能子空间；③证明标准机制可解释工具无法捕捉此结构，揭示“类别不匹配”问题。

**🔧 技术方法**

技术手段包括：滑动窗口Gram矩阵SVD、离散傅里叶变换（按加法、乘法、减法、平方和等群论基）、扰动响应分析、交叉项回归、权重干扰权重共用度分析、稀疏自编码器对比。

**📊 数据集**

数据集为模算术任务（mod 97）上的二元运算：加、减、乘、平方和、平方和乘、立方乘，六种任务，每个任务在三随机种子上训练。

**📈 对比分析**

比较方法：通过g_23谱隙下降幅度区分grokking与非grokking；对不同基的傅里叶峰值比率（F_k）评估功能结构；在多任务共享trunk模型中对比单任务与多任务下的功能模式重合程度。结果显示：grokking任务出现显著谱隙下降；光谱边缘在适当的群论基下聚焦单一频率；多任务训练提升功能模式共享度。

**⚠️ 局限性**

局限性：仅在具备已知群结构的模算术任务中验证；对更复杂或无明显对称性的任务（如自然语言、视觉）未知；光谱边缘的解释力受傅里叶峰值不完整（最高F≤0.4）和交叉项R²≤0.16限制；未能给出完全理论证明。

---

## 34. Visual prompting reimagined: The power of the Activation Prompts

**arXiv ID:** 2604.06440 | [PDF](https://arxiv.org/pdf/2604.06440v1)

**作者:** Yihua Zhang `[一作]` (Michigan State University), Sijia Liu `[通讯]` (Michigan State University)

**通讯引用:** 6720 | [OpenAlex ID](https://openalex.org/A5100321835)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实验了激活提示（Activation Prompt, AP）——一种将可学习的扰动注入预训练模型中间激活层的通用视觉提示框架，并通过理论与实证分析其层级偏好与归一化调优的关系；

**💡 创新点**

创新点在于将视觉提示从输入空间推广到特征空间，揭示 AP 与 BatchNorm/LayerNorm 之间的等价关系，理论证明浅层 ViT 上 AP 的样本复杂度优势，并在多种网络架构与任务上实现显著性能提升；

**🔧 技术方法**

使用可学习的激活扰动、线性归一化调优映射、CKA 特征相似度、平均注意力距离、理论样本复杂度分析，以及大规模实验评估；

**📊 数据集**

在 29 个视觉迁移学习基准（ImageNet-1K、OxfordPets、StanfordCars、SVHN、GTSRB、VTAB-1K 等）以及 CLIP、Swin-Transformer 上进行实验；

**📈 对比分析**

与全参数微调、传统 VP、VPT、LoRA、GateVPT、E2VPT、SST 等 8 个 PEFT 基线比较，AP 在大多数数据集上平均提升 4%（ResNet）/1.5%（ViT）准确率，并在训练时长、内存占用和推理吞吐量上显著优于 VP；

**⚠️ 局限性**

对小规模模型（如 ResNet‑18、ViT‑Tiny）的效果有限，主要受预训练模型规模限制，难以在极小模型上显著超越传统微调方法。

---

## 35. ProofSketcher: Hybrid LLM + Lightweight Proof Checker for Reliable Math/Logic Reasoning

**arXiv ID:** 2604.06401 | [PDF](https://arxiv.org/pdf/2604.06401v1)

**作者:** Kranthi Kommuru `[一作]` (Automatic Data Processing, Inc), Gaurav Parekh `[通讯]` (Amazon Web Services, Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了ProofSketcher，一个将LLM生成的类型化证明草图与轻量级可信核结合的混合式证明生成框架。

**💡 创新点**

通过类型化草图DSL和可证书自动化的桥接，实现了局部修复、节点级缓存和结构化反馈，从而显著提升了LLM证明的可靠性与效率。

**🔧 技术方法**

使用LLM（如GPT）生成草图，轻量可信核进行义务提取与核验，外部ATP/SMT求解器返回可检验证书，结构化反馈循环与缓存机制。

**📊 数据集**

在miniF2F、LeanDojo以及ProofNet等标准形式化证明基准上进行评估。

**📈 对比分析**

与ReProver、DeepSeek-Prover系列比较，ProofSketcher在miniF2F的通过率提升至92.21%，LeanDojo 58.25%，ProofNet 44.62%，且平均LLM调用次数显著降低。

**⚠️ 局限性**

受制于草图抽象度与证书验证开销平衡、库检索与缺失lemma的依赖，以及跨领域适配的挑战，仍需进一步优化草图细化与证书生成效率。

---

## 36. Depression Detection at the Point of Care: Automated Analysis of Linguistic Signals from Routine Primary Care Encounters

**arXiv ID:** 2604.06193 | [PDF](https://arxiv.org/pdf/2604.06193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 37. Hallucination as output-boundary misclassification: a composite abstention architecture for language models

**arXiv ID:** 2604.06195 | [PDF](https://arxiv.org/pdf/2604.06195v1)

**作者:** Angelina Hintsanen `[一作]` `[通讯]` (NEXUS Laboratory), Angelina Hintsanen (NEXUS Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型的幻觉问题，并提出将其视为输出边界误分类的控制理论框架，进而设计了由指令拒绝和结构化回避门组成的复合回避架构。

**💡 创新点**

创新点在于将幻觉建模为输出边界误分类，并将指令式拒绝与基于自洽、一致性、引用覆盖率的黑盒支持缺失门组合，形成双重失败模式互补的回避机制。

**🔧 技术方法**

使用自洽一致性、多次采样、语义相似度评估、引用覆盖率指标计算支持缺失分数，并基于阈值实现结构化回避门，结合提示工程实现指令式拒绝。

**📊 数据集**

使用自构造的50条问答样本（5个认知范式）以及TruthfulQA 100条无上下文的压力测试样本。

**📈 对比分析**

通过对GPT‑3.5‑turbo、GPT‑4o‑mini、GPT‑4o三款模型的四种条件（基线、指令、门、复合）进行对照实验，复合条件在50项评估中准确率96–98%，幻觉率0–4%，在TruthfulQA压力测试中门和复合条件实现了98–100%拒绝率。

**⚠️ 局限性**

局限在于仅测试OpenAI模型、规模有限、使用简化的支持信号、缺乏跨模型族验证、成本高、对提示敏感、未覆盖开放式生成等。

---

## 38. Improving Robustness In Sparse Autoencoders via Masked Regularization

**arXiv ID:** 2604.06495 | [PDF](https://arxiv.org/pdf/2604.06495v1)

**作者:** Vivek Narayanaswamy `[一作]` (Lawrence Livermore National Laboratory), Wesam Sakla `[通讯]` (Lawrence Livermore National Laboratory)

**通讯引用:** 197 | [OpenAlex ID](https://openalex.org/A5060098548)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大语言模型中使用稀疏自编码器（SAE），并提出基于遮蔽的正则化来减少特征吸收、提升解释性和鲁棒性。

**💡 创新点**

创新点在于通过随机遮蔽输入词破坏共现模式，抑制 SAEs 的快捷学习，从而提高对 OOD 的泛化能力。

**🔧 技术方法**

使用稀疏自编码器、遮蔽正则化、BatchTopK 稀疏激活等技术，并结合线性探针评估。

**📊 数据集**

在 Pythia‑160M 与 Gemma‑2‑2B 两个 LLM 上使用 Pile‑CC 数据集进行训练。

**📈 对比分析**

与无遮蔽基线对比，利用五项评估指标（吸收、方差解释、稀疏探针、TPP、SCR）以及 OOD AUC，遮蔽显著降低吸收、提升探针性能并缩小 OOD 间距。

**⚠️ 局限性**

局限在于遮蔽比例需经验调优，低稀疏度下收益有限；在更大模型上的效果尚未充分验证。

---

## 39. Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook

**arXiv ID:** 2604.06210 | [PDF](https://arxiv.org/pdf/2604.06210v1)

**作者:** Jaehyeok Lee `[一作]` (Sungkyunkwan University), JinYeong Bak `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1123 | [OpenAlex ID](https://openalex.org/A5044635661)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于分布式开放式评估的框架（DOVE），用于量化大型语言模型在不同文化中的价值取向对齐程度。

**💡 创新点**

创新点在于：① 通过解决 Construct–Composition–Context（C³）挑战，放弃传统问卷/多项选择，改为对长文本分布进行直接比较；② 自动构建无先验价值维度的“价值代码书”，采用速率‑失真变分优化；③ 用无平衡最优运输（UOT）度量人类与模型价值分布差异，捕捉内部结构与子群多样性。

**🔧 技术方法**

技术手段包括：速率‑失真变分优化（变分期望‑最大化 + 黑箱优化）、LLM 作为代码识别器与解码器、无平衡 Sinkhorn 算法求解 UOT、基于文本嵌入的相似度与距离计算。

**📊 数据集**

数据集：约 15,213 篇人类原创文档（4 个文化，824 主题，平均 1,034 词）；10,676 篇 LLM 生成文档用于代码书构建；后续还使用 5 个下游任务（如 KOLD、HateXplain）进行预测效度检验；对标基准包括 WVS、GOQA、CDEval、NormAd、NaVAB。

**📈 对比分析**

与现有 5 种基准对比：在构造效度（价值引导实验、MTMM）、预测效度（与 5 个下游任务的皮尔逊相关）以及可靠性（Cronbach α、样本/主题鲁棒性）上均表现更好。DOVE 与下游任务的相关系数最高可达 31.56%，且仅需 300~500 条样本即可取得与全量相近的稳定性。

**⚠️ 局限性**

局限性：① 依赖 LLM（如 GPT‑4、GPT‑5）进行代码识别与解码，成本与可扩展性受限；② 代码书大小与超参数对结果敏感；③ 目前仅针对 4 个主要文化，跨文化推广需要更多多语言人类文本；④ 对极端或新兴价值维度的捕捉仍受限于训练语料。

---

## 40. Severity-Aware Weighted Loss for Arabic Medical Text Generation

**arXiv ID:** 2604.06346 | [PDF](https://arxiv.org/pdf/2604.06346v1)

**作者:** Ahmed Alansary `[一作]`, Ali Hamdi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了凸子四次及子线性Hamiltonian系统的周期解存在性，并给出了非平凡解与最小周期的充分条件；

**💡 创新点**

提出了新的非平凡性判据和最小周期证明，改进了现有的子周期解存在理论；

**🔧 技术方法**

采用凸分析、对偶行动函数最小化、谱分析和指数法等数学工具；

**📊 数据集**

无；该工作为纯理论数学研究，无数据集；

**📈 对比分析**

无；该研究不涉及实验或数值对比；

**⚠️ 局限性**

仅适用于光滑凸子四次Hamiltonian，且对非凸或不满足子四次增长条件的系统无适用性。

---

## 41. SE-Enhanced ViT and BiLSTM-Based Intrusion Detection for Secure IIoT and IoMT Environments

**arXiv ID:** 2604.06254 | [PDF](https://arxiv.org/pdf/2604.06254v1)

**作者:** Afrah Gueriani `[一作]` (University of MEDEA), Onur Ceran `[通讯]` (Gazi University)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5026347958)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一个基于SE增强Vision Transformer与BiLSTM融合的入侵检测框架，用于工业和医疗物联网环境。

**💡 创新点**

创新点在于将Squeeze‑and‑Excitation注意力替代ViT的多头自注意力，并与BiLSTM并行融合，提升空间与时间特征提取能力与检测精度。

**🔧 技术方法**

使用SE‑ViT‑BiLSTM深度网络、SMOTE/RandomOverSampler进行类别平衡、Adam优化器和交叉熵损失进行训练。

**📊 数据集**

利用EdgeIIoT和CICIoMT2024两个真实工业/医疗IoT数据集进行实验。

**📈 对比分析**

与多种现有模型（CNN、LSTM、CNN‑LSTM‑ResNet等）进行基准对比；未平衡时EdgeIIoT 99.11%/CICIoMT2024 96.10%，平衡后EdgeIIoT 99.33%/CICIoMT2024 98.16%，同时FPR与推理时间均优于对手。

**⚠️ 局限性**

仍受数据集多样性限制，模型在更大规模或不同场景下的泛化尚未验证，缺乏可解释性分析。

---

## 42. SymptomWise: A Deterministic Reasoning Layer for Reliable and Efficient AI Systems

**arXiv ID:** 2604.06375 | [PDF](https://arxiv.org/pdf/2604.06375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 43. Quality-preserving Model for Electronics Production Quality Tests Reduction

**arXiv ID:** 2604.06451 | [PDF](https://arxiv.org/pdf/2604.06451v1)

**作者:** Noufa Haneefa `[一作]` (Jonkoping University), Einav Peretz-Andersson `[通讯]` (Jonkoping University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了结合离线最小成本诊断子集构造与在线Thompson Sampling多臂老虎机的自适应测试选择框架，能够在生产过程中动态切换全测与简化测试；

**💡 创新点**

创新点在于将数据驱动的集合覆盖优化与在线强化学习相结合，能够在概念漂移下实时控制缺陷逃逸风险，并提供了基于滚动通过率的漂移感知机制；

**🔧 技术方法**

使用技术包括贪心集合覆盖算法、Thompson Sampling多臂老虎机、滚动通过率稳定性信号以及设计的奖励函数；

**📊 数据集**

使用了电子PCB装配的两个测试阶段（功能电路测试172步、终端线测试55步）的28,000+板子历史与验证数据；

**📈 对比分析**

与静态削减方案对比，验证阶段FCT由110逃逸降至0，节约18.78%测试时间；EOL由8逃逸降至0，节约87.17%测试时间；整体表现优于传统方法且保持零逃逸；

**⚠️ 局限性**

局限包括仅在单一PCBA工厂验证、仅使用二元决策空间、奖励函数简化、仅依赖滚动通过率作为漂移指标、缺乏多元化解释与更广泛的产品验证。

---

## 44. Parametrizing Reads-From Equivalence for Predictive Monitoring

**arXiv ID:** 2604.06533 | [PDF](https://arxiv.org/pdf/2604.06533v1)

**作者:** Azadeh Farzan `[一作]` (University of Toronto), Umang Mathur `[通讯]` (National University of Singapore)

**通讯引用:** 1733 | [OpenAlex ID](https://openalex.org/A5016958234)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种基于k-切片重排的预测性运行时监控框架，用于检测并发程序在满足正则规范时的潜在错误。

**💡 创新点**

创新点在于将读-写等价与Mazurkiewicz轨迹等价通过参数k的切片划分进行调度，实现从弱到强的可扩展可预测性，并在k固定时实现常数空间流式算法。

**🔧 技术方法**

采用切片重排的定义、参数化技术、有限自动机构造与前像/后像分析，实现对任意正则规范的预测性监控。

**📊 数据集**

未使用具体实验数据集，研究主要基于理论证明与自动机构造，强调算法复杂度与空间上限。

**📈 对比分析**

与传统的轨迹等价、粒度/散粒度等预测器对比，证明在前像问题上可保持常数空间，而后像问题则需线性空间；实验与性能比较未给出，仅提供理论复杂度分析。

**⚠️ 局限性**

局限性包括：对线程数和共享变量数的隐式依赖、k=1时后像仍需线性空间、算法实现需较高的状态空间，且切片重排并非对称或传递，未覆盖所有读-写等价的预测情况。

---

## 45. Occlusion Handling by Pushing for Enhanced Fruit Detection

**arXiv ID:** 2604.06341 | [PDF](https://arxiv.org/pdf/2604.06341v1)

**作者:** Ege Gursoy `[一作]` (LIRMM, University of Montpellier, French National Center for Scientific Research), Andrea Cherubini `[通讯]` (LIRMM, University of Montpellier, French National Center for Scientific Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在果树环境中通过RGB‑D相机与机械臂结合，利用生成式深度重建与三维Hough变换，主动推开遮挡枝条以提升水果可视化和定位精度。

**💡 创新点**

①将遮挡物物理移除（推枝）而非仅估计；②在深度空间使用U‑Net生成模型重建被遮挡水果；③将二维Hough变换扩展到三维点云，直接检测枝条线段。

**🔧 技术方法**

RGB‑D摄像、HSV颜色分割、深度映射、U‑Net生成网络、三维Hough变换、视锥分析、ROS/MoveIt机械臂控制。

**📊 数据集**

训练用100张合成深度图（椭圆水果+随机遮挡）做生成网络训练；实验用真实橙树、苹果、柠檬场景的数据。

**📈 对比分析**

通过可见度提升、生成误差、枝条检测准确率以及机械臂推枝成功率进行评估。实验表明遮挡被有效清除，视觉可见度显著提升，推枝成功率在不同光照与果实类型下均保持高水平。

**⚠️ 局限性**

在复杂枝叶密集环境下枝条分割与线段检测误检率上升，深度估计精度有限；方法仅在单树单臂场景验证，尚需进一步扩展到多树、多臂或更高密度的果园环境。

---

## 46. RAGEN-2: Reasoning Collapse in Agentic RL

**arXiv ID:** 2604.06268 | [PDF](https://arxiv.org/pdf/2604.06268v1)

**作者:** Zihan Wang `[一作]` (Northwestern University), Manling Li `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多轮 LLM 代理 RL 训练中模板崩塌问题，提出互信息（MI）代理诊断和 SNR‑Aware Filtering 解决方案。

**💡 创新点**

首次将模板崩塌定义为输入相关性下降，提出基于 MI 的无监督诊断，并用奖励方差筛选提升信噪比的 SNR‑Aware Filtering。

**🔧 技术方法**

信息论 MI 代理、奖励方差作为信噪比估计、梯度分解、PPO/GRPO 等 RL 算法、LLM（Qwen2.5、Llama3.2）与多模态模型。

**📊 数据集**

RAGEN 测试床下七个任务：Sokoban、FrozenLake、MetaMathQA、Countdown、SearchQA、WebShop、DeepCoder。

**📈 对比分析**

与无筛选、Top‑k、Top‑p 等过滤策略及 KL/entropy 调节对比；SNR‑Aware Filtering 在大多数任务、算法、模型规模与模态上提升 5‑10% 的任务成功率，MI 与性能正相关。

**⚠️ 局限性**

假设信号与噪声完全分离，需奖励方差可靠，可能在稀疏或噪声奖励环境失效；可能过度过滤导致探索不足，需任务调参。

---

## 47. Efficient Quantization of Mixture-of-Experts with Theoretical Generalization Guarantees

**arXiv ID:** 2604.06515 | [PDF](https://arxiv.org/pdf/2604.06515v1)

**作者:** Mohammed Nowaz Rabbani Chowdhury `[一作]` (Rensselaer Polytechnic Institute), Meng Wang `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 42797 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于专家路由器l₂范数变化的混合精度量化方案，针对MoE模型中专家的重要性差异分配高精度和低精度位宽；

**💡 创新点**

创新点在于：①理论证明路由器norm变化可衡量专家对模型性能的敏感度；②结合最大单元方差对专家排序，进一步避免高噪声专家被低精度量化；③实现无需额外训练或显存开销的量化方法；

**🔧 技术方法**

技术手段包括后训练权重量化（PTWQ）、混合精度量化策略、MoE架构分析、路由器norm变化度量、最大单元方差排序、理论泛化保证与实验评估；

**📊 数据集**

实验数据集：Switch Transformer微调后的CNN/Daily Mail文本摘要任务；Mixtral 8x7B/8x22B在八个零样本LLM基准任务（PIQA、ARC-e/c、BoolQ、HellaSwag、Winogrande、MathQA、MMLU）进行评测；

**📈 对比分析**

与统一3位/2位量化、激活频率/权重、PMQ、层级Hessian、BSP、Slim-LLM等方法对比，结果显示其在平均2.5位/专家时既能保持或提升准确率，又在推理速度和内存占用上优于现有方法；

**⚠️ 局限性**

局限性：依赖预训练模型路由器norm变化的可预测性，理论假设（如最大方差一致、α<1/4）与实际情况可能不完全匹配；对极低位宽（<2）性能仍易下降；仅支持两/三级位宽分配，未探讨更细粒度分级；

---

## 48. A Survey of Algorithm Debt in Machine and Deep Learning Systems: Definition, Smells, and Future Work

**arXiv ID:** 2604.06363 | [PDF](https://arxiv.org/pdf/2604.06363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 49. Beyond Functional Correctness: Design Issues in AI IDE-Generated Large-Scale Projects

**arXiv ID:** 2604.06373 | [PDF](https://arxiv.org/pdf/2604.06373v1)

**作者:** Syed Mohammad Kashif `[一作]` (Wuhan University), Mojtaba Shahin `[通讯]` (RMIT University)

**通讯引用:** 2193 | [OpenAlex ID](https://openalex.org/A5052783352)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 Cursor AI IDE 在生成大规模软件项目的能力，并对生成项目的设计质量进行系统评估。

**💡 创新点**

创新点在于提出了 Feature‑Driven Human‑In‑The‑Loop（FD‑HITL）框架来指导大规模项目生成，并首次量化分析 Cursor 生成项目的设计缺陷。

**🔧 技术方法**

使用了 Cursor AI IDE 结合 FD‑HITL 进行项目生成，并采用 CodeScene 与 SonarQube 两种静态分析工具检测设计问题。

**📊 数据集**

构建了 10 个复杂项目描述并生成 169,646 行代码，形成 DIinAGP 数据集，用于评估与分析。

**📈 对比分析**

通过人工评估得到平均 91% 的功能正确率；静态分析共检测到 1,305（CodeScene）和 3,193（SonarQube）个设计问题，显示生成项目存在显著设计缺陷。

**⚠️ 局限性**

局限性包括仍需人工复审以过滤误报，设计缺陷多，且实验仅针对 Cursor，未验证其他 AI IDE 的类似表现。

---

## 50. DietDelta: A Vision-Language Approach for Dietary Assessment via Before-and-After Images

**arXiv ID:** 2604.06352 | [PDF](https://arxiv.org/pdf/2604.06352v1)

**作者:** Gautham Vinod `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**通讯引用:** 3999 | [OpenAlex ID](https://openalex.org/A5001380619)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用前后食用图像配对，基于文本提示的跨模态注意力模型实现单品级食物重量与摄入量的精准估计。

**💡 创新点**

创新点在于将自然语言提示作为语义锚点，直接对视觉补丁进行注意力聚焦，无需像素级分割或多视角、深度信息，且采用两阶段训练（绝对重量 + 差值）实现对消耗量的高精度推断。

**🔧 技术方法**

技术包括 CLIP ViT‑L/14 视觉编码器与文本编码器、Patch‑level 跨模态注意力、投影 MLP、残差回归头、InfoNCE 对齐损失，以及两阶段训练策略。

**📊 数据集**

使用公开数据集 Nutrition5k（绝对重量）、Food Portion Benchmark (FPB)（补充大规模标注）以及 ACE‑TADA（前后配对）进行训练与评估。

**📈 对比分析**

与 RGB、RGB‑D、Swin Nutrition、Gemini 2.5 Pro、Gemma 3 等基线和 VLM 进行对比，平均百分比 MAE (PMAE) 仅为 14.42%，显著低于最佳基线（RGB PMAE 36.54%），在差值估计上亦取得 14.17% 的最佳 PMAE。

**⚠️ 局限性**

局限包括仍依赖单张 RGB 视角，缺乏深度或尺度参考导致对复杂形状的估计存在误差；模型尺寸与推理时延在移动端仍需优化，且文本提示的准确性对结果有显著影响。

---

## 51. DesigNet: Learning to Draw Vector Graphics as Designers Do

**arXiv ID:** 2604.06494 | [PDF](https://arxiv.org/pdf/2604.06494v1)

**作者:** Tomas Guija-Valiente `[一作]` (Machine Learning Circle), Iago Suárez `[通讯]` (Machine Learning Circle)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DesigNet，基于 Transformer‑VAE 的 SVG 字体生成框架，支持连续坐标、层次化潜在空间，并引入连续性与轴对齐自我修正模块；

**💡 创新点**

① 连续性自我修正模块可预测并强制实现 C^0、G^1、C^1 连续性；② 轴对齐自我修正模块实现水平/垂直线段的自动 snapping；③ 使用连续坐标避免离散化误差并保证可微分；

**🔧 技术方法**

Transformer 编码器/解码器、变分自编码器、Straight‑Through Estimator、线性/非线性回归、交叉熵等；

**📊 数据集**

自建拉丁字体数据集（16k+字体，5k 家族）及中文字体数据集；还在图标数据集上进行泛化实验；

**📈 对比分析**

与 DeepSVG、DeepVecFont‑v2、DualVector 等基线对比。评估指标包括 IoU、ℓ1、Chamfer 误差、连续性与对齐准确率。实验显示 DesigNet 在连续性、对齐准确率上提升至 88%/97%，IoU 与 ℓ1 均超过基线；

**⚠️ 局限性**

仍无法完全匹配专业设计师级别的风格一致性；在绝对坐标空间难以实现重复结构的完全复制；缺乏对复杂构词结构（如汉字）的显式组合性。

---

## 52. A Novel Automatic Framework for Speaker Drift Detection in Synthesized Speech

**arXiv ID:** 2604.06327 | [PDF](https://arxiv.org/pdf/2604.06327v1)

**作者:** Jia-Hong Huang `[一作]` (University of Amsterdam), Evangelos Kanoulas `[通讯]` (University of Amsterdam)

**通讯引用:** 3998 | [OpenAlex ID](https://openalex.org/A5055639036)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出基于LLM和余弦相似度的自动化说话人漂移检测框架，能在扩散式TTS中识别单句内的说话人身份细微漂移。

**💡 创新点**

首次将说话人漂移建模为二分类任务，构造专门的合成数据集，并将LLM用于嵌入相似度推理，且给出了理论误差界限。

**🔧 技术方法**

使用Wav2Vec2/Whisper等语音嵌入、余弦相似度计算、PCA降维、LLM提示推理以及阈值分类等技术。

**📊 数据集**

采用人工标注的扩散TTS合成说话人漂移数据集，共128条样本（64漂移/64非漂移），每条为三段重叠音频。

**📈 对比分析**

与固定阈值基线和PCA+LLM基线对比，LLM余弦相似度输入在多种LLM上取得F1>90%，显著优于阈值基线（F1≈0.62）和PCA基线。

**⚠️ 局限性**

仅限单语种和人工合成样本，LLM在高维嵌入直接输入时表现欠佳，对极细微漂移的检测仍有限，缺乏多语种和真实数据验证。

---

## 53. Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection

**arXiv ID:** 2604.06332 | [PDF](https://arxiv.org/pdf/2604.06332v1)

**作者:** Parker Ewen `[一作]` (Torc Robotics), Felix Heide `[通讯]` (Torc Robotics)

**通讯引用:** 6635 | [OpenAlex ID](https://openalex.org/A5059313827)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 Telescope 的两阶段超长距离目标检测框架，利用可学习的双曲球视差变换对图像进行重采样并结合高分辨率检测网络，实现对高速公路上距离超过 250 m 的物体的精准定位。

**💡 创新点**

创新点包括：1) 可学习、可逆的双曲球视差变换（hyperbolic foveation），在保持计算开销低的同时对远距离物体进行放大；2) 在该变换诱导的黎曼空间中使用切向量参数化框架，解决传统轴对齐框在非线性变换下失效的问题；3) 将大型预训练视觉基座（SAM3）与 Deformable DETR 结合，并采用去噪训练策略，进一步提升远距离检测性能。

**🔧 技术方法**

使用的核心技术包括：双曲球视差变换、黎曼几何框参数化、预训练视觉基座（SAM3）编码器、Deformable DETR 检测头、去噪训练（denoising）以及 gIoU 损失。

**📊 数据集**

主要使用的公开数据集是 TruckDrive，专门标注了高达 1 km 的目标距离；实验也在 Argoverse 等中等距离数据集上验证了模型的通用性。

**📈 对比分析**

与现有基准（如 QueryDet、UniverseNet、RVSA、FOVEA、RFLA、Grounding DINO 等）比较，Telescope 在超长距离（>250 m）mAP 从 0.185 提升到 0.326，提升幅度 76%（相对），整体 mAP 从 0.325 提升到 0.497，提升 53%。

**⚠️ 局限性**

局限性：仅在单一数据集上评估；在真实安全关键应用中仍需与多模态传感器融合、系统级验证和合规性检查；模型在近距离物体上的性能略有下降，且对硬件资源仍有一定需求。

---

## 54. Designing Privacy-Preserving Visual Perception for Robot Navigation Based on User Privacy Preferences

**arXiv ID:** 2604.06382 | [PDF](https://arxiv.org/pdf/2604.06382v1)

**作者:** Xuying Huang `[一作]`, Maren Bennewitz `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于用户隐私偏好的移动服务机器人视觉感知设计，开展两项用户研究以评估隐私感知的视觉输入和分辨率阈值，并基于研究结果设计了可配置的距离-分辨率隐私策略；随后在多分辨率RGB语义目标导航系统中实现并演示了该策略。

**💡 创新点**

①首次将用户隐私偏好直接作为视觉感知配置的依据；②设计了“距离-分辨率”隐私策略，使机器人在不同接近距离下自动调整RGB分辨率；③将语义分割作为隐私友好视觉表示，并结合低分辨率捕获实现了技术与用户需求的结合。

**🔧 技术方法**

多分辨率RGB捕获与实时降采样、语义分割恢复模块、基于CLMM的用户分辨率阈值分析、VLM模型的客观隐私评估、在多目标导航框架中集成的距离-分辨率策略。

**📊 数据集**

NYU Depth V2 数据集用于模态比较；利用真实机器人在办公、实验室、客厅、走廊、教室等场景收集的10条导航序列，并在这些序列中分别以384×384、32×32、16×16、8×8四个分辨率生成视频；涉及人脸、护照、信用卡、私聊等隐私敏感内容；部分低分辨率图像由 GPT‑4o 生成。

**📈 对比分析**

通过问卷收集的主观分辨率阈值与VLM模型输出的隐私率进行对比；使用累积分布混合模型（CLMM）对分辨率阈值随机器人距离变化的影响进行统计检验；实验表明用户对低分辨率的隐私满意度随距离减小而提高，模型评估与用户趋势一致。关于导航性能的具体数值未给出，研究重点在隐私与分辨率的权衡。

**⚠️ 局限性**

①仅关注RGB路径，未涉及深度或RGB‑D 的隐私策略；②实验样本主要为室内场景，缺乏多样化外部环境验证；③未量化导航成功率与隐私阈值的直接平衡；④VLM 评估虽与人类趋势一致，但可能无法完全捕捉人类对隐私的主观判断；⑤隐私策略实现基于经验阈值，未针对不同任务或不同用户隐私水平进行自适应优化。

---

## 55. Constrained Policy Optimization for Provably Fair Order Matching

**arXiv ID:** 2604.06522 | [PDF](https://arxiv.org/pdf/2604.06522v1)

**作者:** Zehua Cheng `[一作]` (University of Oxford), Jiahao Sun `[通讯]` (FLock.io)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一种基于受约束的策略优化框架 CPO‑FOAM，用于实现可证明公平的订单匹配算法。

**💡 创新点**

创新点包括：①将群体公平（人口平等、等化机率）与个体 Lipschitz 公平共同嵌入 CMDP 约束；②引入 PID 控制的安全边际来消除传统 Lagrangian 震荡；③在 Fisher 信息流形上解析求解信赖区间；④在区块链上实现可验证的结算与公平审计；⑤在理论上证明了 BIBO 稳定性与无穷大收敛性。

**🔧 技术方法**

使用的技术包括：受约束的策略优化（CPO）、PID 控制、Fisher 信息近似（K‑FAC）、谱归一化（Spectral Normalization）、JAX‑LOB 模拟器、深度强化学习（PPO、RCPO、IPO 等）、以太坊智能合约与 Merkle 证明、离线挑战与验证。

**📊 数据集**

使用的数据集有：①传统金融 LOBSTER NASDAQ 订单簿重建；②加密资产 LOB（BTC/USDT、ETH/USDT 等）并加入 MEV 注入；③Safety‑Gymnasium 连续控制任务；③用 Hawkes 过程模拟极端订单流。

**📈 对比分析**

对比方法包括 FIFO、Pro‑Rata、Size‑Time、PPO、Lagrangian PPO、RCPO、IPO、Vanilla CPO、PID‑Lagrangian 等。实验显示 CPO‑FOAM 在 TradFi 领域恢复 95.9% 吞吐量、CVF 仅 2.5%；在 DeFi 领域恢复 98.4% 吞吐量、CVF 3.2%；在 Safety‑Gym 任务中奖励提升 2.1×，成本下降 1.9×；PID 辅助下恢复步长从数百下降到 5 步，显著抑制了震荡。

**⚠️ 局限性**

局限性包括：①需要手动调节 PID 参数和安全边际阈值；②对极端市场冲击仍会出现 CVF 上升；③多约束下计算开销和显存需求显著提升；④模型的可解释性和透明度仍不足；⑤在极低流动性或高度非平稳环境下，理论假设（几何混合）可能被破坏。

---

## 56. Bi-Level Optimization for Single Domain Generalization

**arXiv ID:** 2604.06349 | [PDF](https://arxiv.org/pdf/2604.06349v1)

**作者:** Marzi Heidari `[一作]` (Carleton University), Yuhong Guo `[通讯]` (Carleton University)

**通讯引用:** 8252 | [OpenAlex ID](https://openalex.org/A5043824291)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种单源域泛化方法 BiSDG，使用双层优化将任务学习与域建模解耦，通过域提示编码器生成轻量级的 FiLM 调制信号实现域感知特征融合。

**💡 创新点**

创新点在于：①通过标签保持的手工组合数据增强构造多样化的 surrogate 域，以模拟未知域的分布偏移；②引入域提示编码器（Set Transformer）在特征上做全局线性调制，保持参数效率；③将任务学习与域调制分为内层与外层双层优化，避免两者陷入无意义的共同退化。

**🔧 技术方法**

核心技术包括：双层优化框架、FiLM 线性调制、Set Transformer 作为域编码器、对抗一致性正则（KL 约束）、梯度近似的有限差分方案、标准化与随机扰动以增强鲁棒性。

**📊 数据集**

在三个主流单域泛化基准上验证：Digits（MNIST→SVHN/MNIST‑M/SYN/USPS）、PACS（Photo→Art/Cartoon/Sketch）和 DomainNet（Real→其他五域），使用 LeNet/ResNet‑18 等主流网络。

**📈 对比分析**

与 MixUp、CutOut、ADA、ME‑ADA、RandAugment、AdvST 等 15+ 传统和先进方法比较，BiSDG 在 PACS 上平均 65.3%（比 AdvST 高 1.2%），Digits 上平均 82.7%（比 AdvST 高 2.6%），DomainNet 上平均 28.3%（比 AdvST 高 1.2%）。实验多次重复，结果稳健。

**⚠️ 局限性**

局限性：①仍依赖人工设计的 surrogate 域，若这些域与真实目标域差异过大，泛化提升有限；②双层优化与梯度近似会增加训练开销；③在极端域偏移（如 Quickdraw）下性能提升仍有限。

---

## 57. A Severity-Based Curriculum Learning Strategy for Arabic Medical Text Generation

**arXiv ID:** 2604.06365 | [PDF](https://arxiv.org/pdf/2604.06365v1)

**作者:** Ahmed Alansary `[一作]`, Ali Hamdi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在凸子二次势能函数下Hamiltonian系统的周期解存在性，给出了非平凡解的充分条件并证明了最小周期性质。

**💡 创新点**

首次以对偶变分方法给出非平凡周期解的充分条件，并在子多重周期存在性方面给出数量下界。

**🔧 技术方法**

对偶变分原理、凸分析、谱理论、最小化双变分泛函、能量函数子线性增长假设。

**📊 数据集**

无具体数据集，采用理论分析与数学证明。

**📈 对比分析**

与Rabinowitz、Clarke‑Ekeland等先前结果对比，证明了在更一般的子线性增长下也能得到周期解，并给出周期区间与条件。

**⚠️ 局限性**

仅适用于C²、子线性增长且正定二阶导的Hamiltonian，无法处理非凸或奇异势能。

---

## 58. The Depth Ceiling: On the Limits of Large Language Models in Discovering Latent Planning

**arXiv ID:** 2604.06427 | [PDF](https://arxiv.org/pdf/2604.06427v1)

**作者:** Yi Xu `[一作]` (University of Cambridge), Laura Ruis `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了大型语言模型在单次前向推理中发现并执行多步隐式规划的能力。

**💡 创新点**

发现即使在大规模模型中，隐式规划深度也有限，且隐式策略发现的瓶颈与模型规模无关。

**🔧 技术方法**

采用基于星形图路径寻找任务的next-token预测训练，并通过注意力分析揭示backtracking策略。

**📊 数据集**

使用可控深度和分支的星形图数据集（G_(k,m)）。

**📈 对比分析**

与无监督、few-shot、fine-tuned等不同设置对比，模型的最大隐式规划深度在从3步到7步之间，表现仍逊色于显式链式推理。

**⚠️ 局限性**

主要限制在于在仅给定最终答案监督的条件下，梯度训练难以发现更深的隐式规划策略。

---

## 59. MTA-Agent: An Open Recipe for Multimodal Deep Search Agents

**arXiv ID:** 2604.06376 | [PDF](https://arxiv.org/pdf/2604.06376v1)

**作者:** Xiangyu Peng `[一作]` (Salesforce AI Research), Chien-Sheng Wu `[通讯]` (Salesforce AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过自动化工具增量式合成框架，生成21K条高质量多跳视觉语言训练样本，并在此数据上训练多模态深度搜索代理。

**💡 创新点**

创新点在于：①构建无人工标注的多跳问答生成与验证管道；②提出全开源的“多跳工具增强代理”配方；③实现训练时的工具重放机制，显著降低训练成本。

**🔧 技术方法**

采用 ReAct 交互式推理框架、网页搜索/图像检索/OCR/Python 等工具调用，结合多阶段事实验证、强化学习（DAPO）和交互重放技术。

**📊 数据集**

基于 FVQA、LiveVQA‑News、InfoVQA、InfoSeek、OK‑VQA 等 VQA 预种子数据生成多跳样本，最终得到21K训练集与178条人工校验的测试集。

**📈 对比分析**

在六个深度搜索基准上与 GPT‑5、Gemini‑2.5‑Pro、Gemini‑3‑Pro 及开源 Qwen3‑VL‑32B 等模型对比，32B 版本平均精度 54.63%，超过 GPT‑5（51.86%）和 Gemini‑3‑Pro（54.46%），且搜索深度提升至 4.28 步。

**⚠️ 局限性**

局限性包括：仍依赖工具调用导致训练成本高；部分数据可能导致模型泄漏；对长尾知识泛化的覆盖仍有限；多跳任务对视觉与文本的匹配仍具有挑战。

---

## 60. User-Centric Design of UI for Mobile Banking Apps: Improving UI and Features for Better Customer Experience

**arXiv ID:** 2604.06175 | [PDF](https://arxiv.org/pdf/2604.06175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 61. VenusBench-Mobile: A Challenging and User-Centric Benchmark for Mobile GUI Agents with Capability Diagnostics

**arXiv ID:** 2604.06182 | [PDF](https://arxiv.org/pdf/2604.06182v1)

**作者:** Yichen Gong `[一作]` (Ant Group), Shuheng Shen `[通讯]` (Ant Group)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VenusBench-Mobile 在线基准，评估移动 GUI 代理在真实用户场景下的功能与鲁棒性。

**💡 创新点**

创新点在于基于用户意图的任务设计和能力导向的诊断注释，揭示感知、记忆等核心瓶颈。

**🔧 技术方法**

采用视觉语言模型（如 Qwen3、UI‑Venus、GUI‑Owl 等）与多模态框架（如 Gemini‑3‑Pro+UI‑Venus）。

**📊 数据集**

使用 27 个开源 Android 应用、149 个主任务及 80 个环境变体（语言、布局、主题等）。

**📈 对比分析**

对比多种 SOTA 模型，Gemini‑3‑Pro 最高仅 36.9% 成功率，开源模型低于 15%，显示 benchmark 的高判别力与严苛性。

**⚠️ 局限性**

局限在于任务规模有限、缺乏长期学习与个性化评估，且大多数模型在环境扰动下表现极度脆弱。

---

## 62. Cross-Lingual Transfer and Parameter-Efficient Adaptation in the Turkic Language Family: A Theoretical Framework for Low-Resource Language Models

**arXiv ID:** 2604.06202 | [PDF](https://arxiv.org/pdf/2604.06202v1)

**作者:** O. Ibrahimzade `[一作]`, K. Tabasaransky `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个理论框架，用以分析土耳其语族语言在大语言模型中的适配与跨语言迁移行为，重点关注低资源语言的参数高效微调与灾难性遗忘；

**💡 创新点**

创新点在于引入土耳其语族迁移系数（TTC）量化内部语言相似度，并给出适配性能的四因素缩放模型与跨语言迁移效率公式；

**🔧 技术方法**

主要技术为低秩适配（LoRA）与概念化的跨语言迁移效率与灾难性遗忘模型；

**📊 数据集**

未使用具体数据集，论文以理论推导与已有文献为基础；

**📈 对比分析**

没有实证比较方法与性能评估，讨论仅停留在理论层面；

**⚠️ 局限性**

局限在于缺乏实验验证，无法证实TTC与缩放公式的实际有效性，同时忽略了不同 tokenizer 对形态学丰富语言的影响。

---

## 63. Drifting Fields are not Conservative

**arXiv ID:** 2604.06333 | [PDF](https://arxiv.org/pdf/2604.06333v1)

**作者:** Leonard Franz `[一作]` (Eberhard Karls Universität Tübingen), Georg Martius `[通讯]` (Eberhard Karls Universität Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析漂移模型的漂移场是否保守，并提出尖锐归一化修正以恢复保守性；

**💡 创新点**

首次发现漂移场一般非保守，提出尖锐归一化并给出可解释的标量损失log-KDE；

**🔧 技术方法**

使用核密度估计、MMD、尖锐/平滑核、梯度与守恒分析、Transformer/DiT生成器以及标准优化；

**📊 数据集**

在MNIST和Fashion‑MNIST上进行实验，未来计划扩展至ImageNet；

**📈 对比分析**

在相同超参下对比原漂移目标、尖锐归一化与MMD等，采用FID、Precision/Recall评估，尖锐归一化与原漂移相近或略优；

**⚠️ 局限性**

仅在小型数据集验证，尖锐归一化提升有限，缺乏大规模实验验证，理论扩展尚未充分探讨。

---

## 64. A Comparative Study of Demonstration Selection for Practical Large Language Models-based Next POI Prediction

**arXiv ID:** 2604.06207 | [PDF](https://arxiv.org/pdf/2604.06207v1)

**作者:** Ryo Nishida `[一作]` (National Institute of Advanced Industrial Science and Technology), Masaki Onishi `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 1676 | [OpenAlex ID](https://openalex.org/A5031341888)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对大语言模型的上下文学习，提出并评估了多种演示（示例）选择策略，用以提升用户下一访地点预测的准确率。

**💡 创新点**

创新点在于引入了基于空间、集合和序列的直观启发式选择方法（DTW、Jaccard、LCS）以及用户过滤机制，证明这些简单策略在准确性和计算成本上均优于传统随机或嵌入相似度选择，甚至可在无微调的条件下与或优于已有的微调模型。

**🔧 技术方法**

技术手段包括：使用大语言模型（Qwen‑2.5‑7B‑Instruct 与 GPT‑4o）进行少样本提示，演示选择算法（DTW、Jaccard、LCS、时间递减），以及对比的嵌入相似度（EmbSim）和随机选择。

**📊 数据集**

实验数据集为三大真实城市数据：Foursquare‑New York、Foursquare‑Tokyo 以及 Gowalla‑California，分别包含 1,048 / 2,282 / 3,957 位用户，4,981 / 7,833 / 9,690 个 POI。

**📈 对比分析**

与基线（随机、EmbSim）及已有微调模型（GETNext、STHGCN、LLM4POI、LLM‑Mob）比较，启发式方法在多种 k（5、15、30）设置下的 ACC@1 均高于基线，Jaccard/LCS 在计算成本和准确率上达到最佳平衡；在 GPT‑4o 与用户过滤下的最优结果可匹配甚至超越现有微调模型。

**⚠️ 局限性**

局限性包括：仅使用了 POI ID/类别而未利用 POI 名称等语义信息；在冷启动场景下仅依赖自身历史数据，未充分利用他人历史；方法在极少样本（k=5）时仍受限于演示中包含正确目标 POI 的概率。

---

## 65. The Defense Trilemma: Why Prompt Injection Defense Wrappers Fail?

**arXiv ID:** 2604.06436 | [PDF](https://arxiv.org/pdf/2604.06436v1)

**作者:** Manish Bhatt `[一作]` (OWASP), Blake Gatto `[通讯]` (Shrewd Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

本文证明在连接的提示空间上，任何连续且保持实用性的包装器防御都无法实现完整的安全，阐述了边界固定、ε‑鲁棒约束和持久不安全区域三大不可能性。

**💡 创新点**

创新点在于首次将拓扑连续性、Lipschitz 定理与正式化证明结合，提出防御三元困境（连续性、实用性、完整性）并给出量化上界，进一步扩展到离散、随机、多轮和流水线场景。

**🔧 技术方法**

主要技术包括拓扑学（连通性与闭集论证）、Lipschitz 分析、Tietze 延拓定理、离散计数论证以及在 Lean4 进行的机械化形式化证明。

**📊 数据集**

实证验证使用了三种大型语言模型（Llama‑3‑8B、GPT‑OSS‑20B、GPT‑5‑Mini）及其在 2 维行为空间（查询歪斜度与权威框架）上的“Failure Manifold”数据集。

**📈 对比分析**

通过与实验热图和预测一致的对比，验证了理论预测；实验表明无任何包装器防御能完全消除攻击，所得到的性能仅是对安全边界的量化管理而非消除。

**⚠️ 局限性**

局限性包括仅适用于连续、实用保持的包装器、连接提示空间、需要 Lipschitz 与正交性假设；不涵盖训练时对齐、模型架构改造、离散或集成防御，以及高维提示空间的进一步评估。

---

## 66. Adversarial Robustness of Time-Series Classification for Crystal Collimator Alignment

**arXiv ID:** 2604.06289 | [PDF](https://arxiv.org/pdf/2604.06289v1)

**作者:** Xaver Fink `[一作]` (CERN), Joost-Pieter Katoen `[通讯]` (RWTH Aachen University)

**通讯引用:** 19369 | [OpenAlex ID](https://openalex.org/A5090819329)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了用于LHC晶体准直的时间序列CNN分类器在实际噪声下的对抗鲁棒性，并提出了预处理感知的包装器、对抗微调以及对抗序列攻击等方法；

**💡 创新点**

创新点包括：① 基于通道相关的结构化噪声模型；② 将预处理步骤与CNN通过可微分包装层集成，实现梯度攻击的统一；③ 对抗序列概念与原型攻击；

**🔧 技术方法**

使用的技术有：1D卷积神经网络、z归一化与零填充预处理、L∞投影梯度下降、ART与Foolbox对抗框架、对抗微调、可微分预处理包装层以及对抗序列优化；

**📊 数据集**

数据集为CERN LHC晶体准直实验得到的BLM时间序列数据，共计1689条标注样本，测试集296条；

**📈 对比分析**

通过在不同配置下评估鲁棒准确率（RA^tool与RA^pipe）以及对抗微调效果，发现预处理感知包装提高鲁棒准确率至约18.6%，而简单的全局L∞模型严重低估鲁棒性；对抗序列攻击成功在指定窗口内切换分类结果；

**⚠️ 局限性**

局限性包括：对抗攻击仅为梯度搜索，未实现完整搜索；非线性预处理导致难以实现正式验证；实验仅针对单一CNN架构与有限数据集；对抗序列攻击仅为证明性示例，缺乏完整的序列鲁棒性分析。

---

## 67. State-of-the-Art Arabic Language Modeling with Sparse MoE Fine-Tuning and Chain-of-Thought Distillation

**arXiv ID:** 2604.06421 | [PDF](https://arxiv.org/pdf/2604.06421v1)

**作者:** Navan Preet Singh `[一作]` (Forta), Ritankar Das `[通讯]` (Incept Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款基于稀疏Mixture of Experts（MoE）的阿拉伯语大型语言模型 Arabic-DeepSeek-R1，并通过四阶段链式思考（CoT）蒸馏进行细粒度调优，旨在提升阿拉伯语的语法、推理与文化适配性能。

**💡 创新点**

创新点在于：①首次将稀疏MoE推理骨干与阿拉伯语专属的四阶段 CoT 蒸馏结合；②在 CoT 的第三阶段加入显式阿拉伯语形态句法检验，强化语言规范；③采用 80/20 Arabic-English 训练混合，兼顾语言深度与跨语种迁移；④通过参数高效 LoRA 微调实现大模型适配，无需全量再训练。

**🔧 技术方法**

使用技术包括：稀疏 MoE 结构、LoRA 参数高效微调、四阶段 CoT 蒸馏（分析、消除、语言检查、综合）、混合 Arabic‑English 语料、严格的去重与污染过滤。

**📊 数据集**

数据集为 372M 词条的混合语料，约 80% 为现代标准阿拉伯语及主要方言（教育、宗教、法律、对话等），20% 为高质量英语公开/研究语料；所有数据在训练前均经过覆盖率检测、去重与毒性过滤。

**📈 对比分析**

评估采用 OALL v2 公开基准与 GPT‑5.1 及其他阿拉伯语专用模型对比。Arabic‑DeepSeek‑R1 在 OALL 上平均得分 80.18%，为首个超越 GPT‑5.1 的开源模型；在 7 项基准中 5 项（MadinahQA、AraTrust、AlGhafa、ALRAGE、Arabic EXAMS）实现 SOTA 或接近 SOTA，尤其在语法任务 MadinahQA 上提升 8.43%。

**⚠️ 局限性**

局限性包括：对考试类基准 Arabic EXAMS 的提升有限；检索增强基准 ALRAGE 的收益相对较小；模型在不同方言间的表现尚未系统评估；未来需进一步融合检索训练、领域专属微调和方言细粒度校正。

---

## 68. Blind Refusal: Language Models Refuse to Help Users Evade Unjust, Absurd, and Illegitimate Rules

**arXiv ID:** 2604.06233 | [PDF](https://arxiv.org/pdf/2604.06233v1)

**作者:** Cameron Pattison `[一作]` (Vanderbilt University), Seth Lazar `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在面对不正当或不公正规则时的“盲拒绝”现象，并评估其是否能判断规则合法性并决定是否协助用户规避；

**💡 创新点**

提出盲拒绝概念，构建覆盖5类规则否定原因与19类权威类型的矩阵式合成数据集，并使用LLM‑as‑judge进行无偏评估，揭示模型即使识别规则合法性也往往仍拒绝；

**🔧 技术方法**

利用LLM自动生成案例、三道自动化质量门控、LLM评判模型响应、统计分析和18种不同配置与规模的模型评估；

**📊 数据集**

1,290个合成规则规避案例，覆盖5类规则否定原因、19类权威类型，并包含细粒度标签和结构化特征；

**📈 对比分析**

在18种模型配置中平均盲拒绝率约75%，非双重用途场景下仍高达59.6%；帮助率在非控制案例中仅22.7–32.9%，显示显著偏差；模型间差异大（如Grok‑4 58%帮助率 vs GPT‑5.4 7.7%）；

**⚠️ 局限性**

双重用途门槛可能过度标记，独立伤害标签偏激，评估仅覆盖明显不公正案例，对模棱两可情境缺失，以及对模型行为解释的局限。

---

## 69. Intimate Strangers by Design: A Uses and Gratifications Analysis of AI Companionship

**arXiv ID:** 2604.06419 | [PDF](https://arxiv.org/pdf/2604.06419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 70. Blockchain and AI: Securing Intelligent Networks for the Future

**arXiv ID:** 2604.06323 | [PDF](https://arxiv.org/pdf/2604.06323v1)

**作者:** Joy Dutta `[一作]` (Khalifa University of Science and Technology), Tu Dac Ho `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 767 | [OpenAlex ID](https://openalex.org/A5006124696)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述了区块链与人工智能在智能网络安全中的协同作用，并提出了安全集成框架、评估蓝图以及实际案例。

**💡 创新点**

创新点在于将区块链与AI技术结合形成多层安全架构，提出BASE评估蓝图，阐述可验证、可解释、可持续的安全模型，并探讨量子抗性、自治安全与生物启发等前沿方向。

**🔧 技术方法**

采用区块链共识机制（PBFT、PoA、PoAh等）、智能合约、零知识证明、去中心化身份、分布式学习等区块链技术；以及机器学习、深度学习、强化学习、自然语言处理、LLM、自动化代理等AI技术。

**📊 数据集**

主要参考公开网络安全数据集与案例，如CIC-IDS、DARPA、工业控制系统日志等；但本文为综述，未构建统一实验数据集。

**📈 对比分析**

通过对比指标（如TPS、时延、吞吐量、AUROC、能源消耗等）评估各技术组合；基于BASE框架对已发表工作进行归一化比较，指出区块链在数据完整性与可追溯性上的优势，AI在异常检测与预测能力上的优势。

**⚠️ 局限性**

限制包括可扩展性与吞吐量瓶颈、时延与实时性挑战、能源与可持续性问题、跨链与互操作性障碍、隐私与可解释性缺失，以及伦理与监管合规性难题。

---

## 71. LLM-Augmented Knowledge Base Construction For Root Cause Analysis

**arXiv ID:** 2604.06171 | [PDF](https://arxiv.org/pdf/2604.06171v1)

**作者:** Nguyen Phuc Tran `[一作]` (Concordia University), Kun Ni `[通讯]` (GAIA-Ericsson)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用大型语言模型自动构建根因分析知识库的方法，并评估了三种技术（微调、检索增强生成和混合方法）的效果。

**💡 创新点**

创新之处在于将微调和检索增强生成相结合的混合框架，充分利用域知识与检索上下文，实现更高质量的根因规则生成。

**🔧 技术方法**

采用了LoRA微调、量化、prompt工程、向量检索RAG以及BERT等语义评估技术来实现知识库构建。

**📊 数据集**

使用了来自电信运营商的1,049条已解决支持工单（共3,147条输入输出对）作为训练与评估数据。

**📈 对比分析**

通过词汇相似度（余弦、BLEU、ROUGE、METEOR）和语义相似度（BERTScore）对比三种方法，混合方法在所有指标上均优于单一微调或RAG，BERTScore最高达0.93。

**⚠️ 局限性**

局限包括数据规模有限、少数罕见故障类别表现差、规则压缩需平衡效率与准确性以及检索过程增加的延迟。

---

## 72. Optimal Rates for Pure {\varepsilon}-Differentially Private Stochastic Convex Optimization with Heavy Tails

**arXiv ID:** 2604.06492 | [PDF](https://arxiv.org/pdf/2604.06492v1)

**作者:** Andrew Lowy `[一作]` `[通讯]` (CISPA Helmholtz Center for Information Security), Andrew Lowy (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究在纯差分隐私（DP）下，处理梯度为重尾分布的随机凸优化（SCO），并给出了对应的最优超额风险速率，提出了一种多阶段、基于Lipschitz扩展的高效算法，能够在多项式时间内（高概率或在结构化子类下几乎确定地）实现该速率。

**💡 创新点**

创新点在于：①首次刻画了纯DP重尾SCO的极限超额风险率（至对数因子）；②提出了利用Lipschitz扩展与联合凸重构相结合的“双输出扰动”框架，突破了传统梯度裁剪方法在纯DP场景中的局限；③给出了高概率下的最优下界，进一步完善了理论极限。

**🔧 技术方法**

核心技术包括：Lipschitz扩展将不光滑、无界梯度问题转化为Lipschitz约束的ERm；联合凸重构将扩展目标改写为高维凸优化；自适应投影子梯度法在不知敏感度的情况下实现近似最优；输出扰动结合局部化策略实现纯DP和高效性。

**📊 数据集**

论文主要为理论研究，并未在具体数据集上进行实验；讨论的结构化子类（如hinge/ReLU、绝对值损失、欧氏球、椭圆体、多面体）均为数学模型，并无真实数据集使用。

**📈 对比分析**

与既往基于裁剪梯度的纯DP或近似DP算法相比，该方法在已知重尾梯度的前提下，取得了最优（至对数因子）的超额风险上界，且在一般子类下实现了高概率多项式时间，结构化子类下可进一步做到几乎确定的多项式时间；实验结果未给出，但理论上比现有方法更优。

**⚠️ 局限性**

主要限制包括：①上界中仍包含 d·log(1/δ) 项，实际下界仅为 d+log(1/δ)，尚有对数因子优化空间；②对一般非结构化损失的几乎确定多项式时间实现仍是未解问题；③论文聚焦理论与算法设计，缺乏对实际机器学习任务中可行性与效果的实证评估。

---

## 73. Full State-Space Visualisation of the 8-Puzzle: Feasibility, Design, and Educational Use

**arXiv ID:** 2604.06186 | [PDF](https://arxiv.org/pdf/2604.06186v1)

**作者:** Ian Frank `[一作]` (Future University Hakodate), Kanata Kawanishi `[通讯]` (Future University Hakodate)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

可视化8-拼图的完整可达状态空间（181,440个状态），并将抽象图结构与具体拼图操作相结合，支持搜索算法的实时演示与交互。

**💡 创新点**

通过GPU实例化、并行布局计算和多布局切换实现大规模状态空间的实时可视化，同时将抽象搜索结构与具体拼图操作紧耦合，形成全空间、跨算法的可视化工具。

**🔧 技术方法**

Unity引擎 + WebGPU、GPU实例化与自定义着色器、Unity Job System 与 Burst Compiler、力导布局、三维导航交互、A* / BFS / DFS 逐步/自动执行等技术。

**📊 数据集**

预先生成的8-拼图完整状态图数据集（181,440 个节点，241,920 条边），作为可视化和搜索算法的基础。

**📈 对比分析**

在同一可视化空间中同时执行 BFS、DFS、A* 等算法，使用多种布局（力导、深度、启发式距离）进行对比，能够直观展示搜索深度、重复访问和启发式引导，运行在标准桌面与 WebGPU 浏览器上均保持流畅帧率。

**⚠️ 局限性**

可视化密度高导致路径识别困难，颜色编码不够区分；SUS得分仅为58.75，表明用户体验和教学引导需进一步改进；系统仅限于8-拼图，未扩展到其他问题域。

---

## 74. Tool-MCoT: Tool Augmented Multimodal Chain-of-Thought for Content Safety Moderation

**arXiv ID:** 2604.06205 | [PDF](https://arxiv.org/pdf/2604.06205v1)

**作者:** Shutong Zhang `[一作]` (Stanford University), Wenfei Zou `[通讯]` (Google)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Tool-MCoT方法，利用小型语言模型与外部工具（OCR、图像描述、目标检测）进行多模态内容安全审核

**💡 创新点**

创新点在于通过LLM生成的工具增强推理数据训练小模型，并让小模型学习何时调用工具，实现高效推理与准确度的平衡

**🔧 技术方法**

技术包括工具驱动链式思考（Tool-augmented Chain-of-Thought）、LLM教师模型（Gemini 2.0 Flash）生成训练数据、LoRA微调、GRPO强化学习与多轮工具调用训练

**📊 数据集**

使用三大公开多模态仇恨言论与安全审核数据集：MMHS150K、Hateful Memes、UnsafeBench

**📈 对比分析**

与大模型基线（ChatGPT‑4o）及无工具、单工具、全工具对比实验显示，小模型在强制工具使用下在三组数据上均提升准确率至约80–83%；在选择性工具使用下能将工具调用率降至约40%同时保持相似准确率

**⚠️ 局限性**

局限在于工具集仍有限，模型对复杂多轮推理仍表现不佳，且对不同任务的迁移能力和跨语言通用性未充分验证

---

## 75. Invisible Influences: Investigating Implicit Intersectional Biases through Persona Engineering in Large Language Models

**arXiv ID:** 2604.06213 | [PDF](https://arxiv.org/pdf/2604.06213v1)

**作者:** Nandini Arimanda `[一作]` (Shiv Nadar University), Rajesh Sharma `[通讯]` (Plaksha University)

**通讯引用:** 2406 | [OpenAlex ID](https://openalex.org/A5011512226)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在不同persona（身份角色）驱动情境下的隐式交叉偏见，并提出一种新的动态评估指标BADx，结合静态嵌入偏差测量、Persona敏感度和波动性，并用LIME提供可解释性。

**💡 创新点**

创新点在于：①将传统的静态偏差评估（CEAT、I‑WEAT、I‑SEAT）与动态persona驱动评估结合；②提出BADx指标，可同时量化偏差放大幅度、模型对persona的敏感程度以及输出波动；③通过LIME实现对偏差驱动词汇的局部解释，提升可解释性。

**🔧 技术方法**

使用的技术包括：CEAT、I‑WEAT、I‑SEAT 等嵌入基线偏差测试；计算BADx、Persona Sensitivity Index (PSI) 与标准差以评估波动性；LIME进行局部可解释性分析；在五个主流LLM（GPT‑4o、DeepSeek‑R1、LLaMA‑4、Claude 4.0 Sonnet、Gemma‑3n E4B）上进行实验。

**📊 数据集**

使用的数据集是自制的260句英文语料（160句交叉身份类、100句中性对照），涵盖六个交叉身份类别，并配套六个persona（3个边缘化、3个优势）和五个统一prompt，用于静态与动态两阶段评估。

**📈 对比分析**

与传统静态指标相比，BADx能揭示模型在persona框架下的偏差放大或抑制，显示更细粒度的动态行为。实验结果显示：GPT‑4o对persona最敏感、波动最大；DeepSeek‑R1偏见抑制效果好但波动大；LLaMA‑4波动最小、偏差稳定；Claude 4.0 Sonnet表现平衡；Gemma‑3n E4B波动低、偏差适度放大，整体提供更全面的偏差评估。

**⚠️ 局限性**

局限性包括：仅使用英文语料，语料规模有限；persona、prompt与身份类别选择受限，难以覆盖多语言、多文化与更广泛的社会情境；缺乏人类主观偏见评分对BADx的验证；因此结果主要适用于小规模、受控的实验环境。

---

## 76. Thinking in Graphs with CoMAP: A Shared Visual Workspace for Designing Project-Based Learning

**arXiv ID:** 2604.06200 | [PDF](https://arxiv.org/pdf/2604.06200v1)

**作者:** Ruijia Li `[一作]` (East China Normal University), Bo Jiang `[通讯]` (East China Normal University)

**通讯引用:** 8391 | [OpenAlex ID](https://openalex.org/A5074044851)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了基于图形共享工作区 CoMAP，支持教师使用人工智能协助项目式学习的教学设计。

**💡 创新点**

将图形共享工作区与双模 AI（全球会话与局部 GUI）结合，实现非线性、迭代式的教师‑AI 合作，显著提升表达与协作效率。

**🔧 技术方法**

前端采用 React+React Flow，后端使用 Flask+SQLite；通过 GPT‑4.1 的多代理函数调用实现 AI 辅助；教学设计以 ASSURE 模型拆解 PBL 结构为基础。

**📊 数据集**

使用 30 名教师参与的实验，评估两道跨学科 PBL 设计任务；无公开数据集，实验以教师自身经验与任务为评估依据。

**📈 对比分析**

采用 within‑subject 跨条件实验，结合问卷与交互日志；CoMAP 在表达、理解、控制、透明度、认知负荷、协作与信任等指标均显著优于基线，效应值从中等到大不等。

**⚠️ 局限性**

样本规模有限且受教师经验影响，无法单独评估多因素效果；实验仅使用 GPT‑4.1，结果对其他 LLM 的推广存在不确定性。

---

## 77. Content Platform GenAI Regulation via Compensation

**arXiv ID:** 2604.06194 | [PDF](https://arxiv.org/pdf/2604.06194v1)

**作者:** Wee Chaimanowong `[一作]` (Chinese University of Hong Kong), Wee Chaimanowong `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5051443532)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出一种基于收入阈值的补偿方案，用以激励平台上的创作者产生高价值的人工内容，减少由生成式AI造成的数据污染，并提升平台收益。

**💡 创新点**

创新点在于：①不依赖昂贵的AI检测或复杂的数据归因方法；②通过阈值机制自动决定补偿金额，使得平台可在缺乏对用户偏好与AI生成分布的完整知识时依旧实现激励与收益优化；③提供完整的均衡分析和阈值选择理论，证明在合适阈值下平台收益可提升。

**🔧 技术方法**

技术手段：基于均衡博弈（Mean‑Field Game）模型，构造内容生成与消费匹配函数（Cobb–Douglas形式），定义收益与补偿函数，利用最优阈值和补偿公式（W^v,w）进行理论推导；在仿真中采用分数型SDE扩散模型训练生成式AI并采样。

**📊 数据集**

使用的数据集：①理论示例中采用一维均匀分布与两级生成式AI分布；②混合高斯仿真采用 p=0.4N(-2,0.5)+0.6N(2,0.5)；③训练生成式AI时使用从 p 采样的 2650 条样本（单期）或 26500 条样本（多期）。

**📈 对比分析**

方法比较：与不补偿（即默认全使用AI）进行对比；在仿真中展示在不同阈值与补偿金额下平台收益 R 与利润 Π 的变化，证明补偿方案能提升利润、减少内容分布失真并防止模型崩塌。性能在示例中显示：当 g 较好时不补偿更优；当 g 较差时适度补偿可显著提高利润。

**⚠️ 局限性**

局限性：①仅在单期模型下分析，未给出多期动态补偿策略；②假设平台无法观测用户偏好 p 与 AI 分布 g，补偿阈值需通过实验估计；③模型未考虑生成式AI的可定制化与创作者对 AI 的辅助使用；④对复杂多维内容空间的推广需进一步研究。

---

## 78. "Don't Be Afraid, Just Learn": Insights from Industry Practitioners to Prepare Software Engineers in the Age of Generative AI

**arXiv ID:** 2604.06342 | [PDF](https://arxiv.org/pdf/2604.06342v1)

**作者:** Daniel Otten `[一作]` (William & Mary), Douglas Schmidt `[通讯]` (William & Mary)

**通讯引用:** 20509 | [OpenAlex ID](https://openalex.org/A5082548649)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对51名具备GenAI使用经验的软件行业从业者进行问卷调查，并对11位受访者进行半结构化访谈，系统了解行业对新进软件工程师的技能期望、组织培训实践、对高校课程的不足认知以及可行的课程改进建议。

**💡 创新点**

首次从行业实践者视角全面梳理GenAI时代软件工程招聘需求与高校课程之间的脱节，并基于调研结果提出可操作的课程改革方案，为高校课程设计提供实证指导。

**🔧 技术方法**

采用混合方法：问卷设计包含多选题、李克特量表和开放式问题，使用Qualtrics收集数据；访谈通过Zoom录音后使用OpenAI Whisper转写，随后进行三位研究员共同的开放式编码分析。

**📊 数据集**

收集到的核心数据集包括51份问卷回收数据（覆盖11个国家、四大洲的开发经验、职位、使用的GenAI工具等信息）以及11次访谈的转录文本。

**📈 对比分析**

本研究不涉及算法性能比较；通过描述性统计和主题编码呈现行业需求与课程差距，未给出数值型性能指标。

**⚠️ 局限性**

样本主要来自研究者的专业网络，学历比例偏高且自选偏差可能导致对GenAI持积极态度；样本量有限，缺乏更广泛的行业与教育背景代表性；自我报告数据可能受记忆或社会期望偏差影响。

---

## 79. Emergent decentralized regulation in a purely synthetic society

**arXiv ID:** 2604.06199 | [PDF](https://arxiv.org/pdf/2604.06199v1)

**作者:** Md Motaleb Hossen Manik `[一作]` (Rensselaer Polytechnic Institute), Ge Wang `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 42277 | [OpenAlex ID](https://openalex.org/A5100400458)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在仅由AI代理构成的社交网络Moltbook上，对约39,026条帖子和5,712条评论进行观察，量化帖子中的指令性语言强度（DI），并将评论分为肯定、纠正、负面反应与中性四种交互类型，探讨DI与纠正性回应的关联。

**💡 创新点**

创新点在于首次揭示在完全去中心化、无人工或平台干预的合成代理社会中，指令性语言越强的帖子越能诱发自发的纠正性反馈，从而表现出内生的社会调节机制。

**🔧 技术方法**

技术方法包括基于可审计正则表达式的词典式DI计算、确定性规则的评论分类、帖子层随机截距的混合效应逻辑回归、置换检验以及事件对齐的线程级指令强度变化分析。

**📊 数据集**

所使用的数据集为Moltbook活动档案，共包含39,026条帖子、5,712条评论，来自14,490名OpenClaw代理，数据由只读观测器收集而未干预。

**📈 对比分析**

通过标准化DI的混合效应逻辑回归估计β=0.1276（OR≈1.14），置换检验p≈0.008，事件对齐分析显示纠正回应后平均指令强度下降，表明DI与纠正概率呈显著正相关且结果稳健。

**⚠️ 局限性**

局限性包括缺少工具使用轨迹与执行结果数据，事件对齐分析受指令语言稀疏和可调节线程筛选的影响，且无法确立因果关系，需进一步研究长期与代理层面效应。

---

## 80. Shocks without shock capturing: Information geometric regularization of finite volume methods for Navier--Stokes-like problems

**arXiv ID:** 2604.06546 | [PDF](https://arxiv.org/pdf/2604.06546v1)

**作者:** Anand Radhakrishnan `[一作]` (Georgia Institute of Technology), Florian Schäfer `[通讯]` (New York University)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5110998896)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种信息几何正则化（IGR）方法，作为流体动力学中处理冲击波的无粘正则化方法。该方法通过用可调宽度的平滑轮廓替代冲击奇点，避免了传统数值方法中常见的Gibbs-Runge振荡。

**💡 创新点**

IGR是首个不依赖于粘性的冲击正则化方法，能够在不耗散细尺度特征（如湍流或声波）的情况下，提供平滑的解决方案。该方法在有限体积数值方法中的实际应用策略也得到了阐述。

**🔧 技术方法**

使用了信息几何正则化（IGR）技术，结合有限体积方法（FVM）进行数值实现。

**📊 数据集**

在多个经典测试问题上进行了验证，包括一维平滑波传播、Shu-Osher问题、Sod冲击管、Leblanc冲击管以及二维Riemann问题等。

**📈 对比分析**

与基于限制器和Riemann求解器的传统方法相比，IGR在平滑和不连续流动状态下的准确性与WENO和LAD冲击捕获方案相当。IGR方法在计算上更轻量，内存访问和算术操作显著减少。

**⚠️ 局限性**

IGR方法在处理初始和边界条件的光滑性方面存在一定限制，尤其是在初始条件不光滑的情况下，可能会引发Gibbs-Runge振荡。

---

## 81. Consistency-Guided Decoding with Proof-Driven Disambiguation for Three-Way Logical Question Answering

**arXiv ID:** 2604.06196 | [PDF](https://arxiv.org/pdf/2604.06196v1)

**作者:** Tianyi Huang `[一作]`, Ziling Zhang `[通讯]` (App-In Club)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在三路逻辑问答任务中，提出了一种测试时的轻量级包装器CGD-PD，利用否定关系的硬约束和多轮提示来纠正模型的不一致与不确定预测。

**💡 创新点**

创新点在于：①将否定映射作为逻辑一致性约束；②采用双重查询（原命题与其机械否定）并在出现不确定时通过“fixer”提示收集证明或缺失信息；③进一步利用二元蕴含探测（YES/NO）进行证明驱动的消歧，最后投影到一致的答案。

**🔧 技术方法**

技术手段包括：LLM的三路分类器、机械否定包装、针对性修正提示、二元蕴含探测、逻辑一致性投影与轻量级争议解决。

**📊 数据集**

实验使用FOLIO数据集的第一阶逻辑（FOL）字段，包含前提集合与命题的形式化逻辑表达式。

**📈 对比分析**

与单一分类器对比，CGD-PD在GPT‑5.2上提升4.4个百分点、Claude Sonnet 4.5提升6.9个百分点；平均调用约4–5次，显著降低预测率和epistemic率，且在置信区间内均无零交叉。

**⚠️ 局限性**

局限性包括：仅处理否定约束，未覆盖更复杂逻辑变换；计算成本比单调用高，平均约4–5次；对真正未指明的例子可能仍做错误决策，未实现完整的逻辑求解。

---

## 82. Trust in AI among Middle Eastern CS Students: Investigating Students' Trust and Usage Patterns Across Saudi Arabia, Kuwait and Jordan

**arXiv ID:** 2604.06418 | [PDF](https://arxiv.org/pdf/2604.06418v1)

**作者:** Saleh Alkhamees `[一作]` (University of Houston), Amin Alipour `[通讯]` (University of Houston)

**通讯引用:** 1334 | [OpenAlex ID](https://openalex.org/A5090034689)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在沙特阿拉伯、科威特和约旦四所大学中对计算机科学学生的AI工具使用与信任进行问卷调查。

**💡 创新点**

首次系统性比较非WEIRD中东学生的AI信任与使用行为，揭示了语言与性别环境对信任差异的影响。

**🔧 技术方法**

采用改编自Körber的信任度量量表、定量统计与定性主题编码。

**📊 数据集**

基于202名有效问卷样本（四所大学，三国），构成的自定义调查数据集。

**📈 对比分析**

通过描述性统计、Wilcoxon检验、相关分析和性别/国别比较评估信任水平，发现沙特女生信任低于男生，整体信任中等。

**⚠️ 局限性**

样本规模有限，受访者仅来自四所大学，缺乏对更广泛中东或非英语地区的推广性。

---

## 83. The Illusion of Stochasticity in LLMs

**arXiv ID:** 2604.06543 | [PDF](https://arxiv.org/pdf/2604.06543v1)

**作者:** Xiangming Gu `[一作]` (Google DeepMind), Razvan Pascanu `[通讯]` (Google DeepMind)

**通讯引用:** 30260 | [OpenAlex ID](https://openalex.org/A5043910056)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过多种提示、解码参数、序列化与批量采样以及工具调用等方式，系统评估大型语言模型在直接从离散、连续及高斯等分布中采样时的可靠性，并探讨其与传统伪随机数生成器或分布转换方法的可行性与局限。

**💡 创新点**

发现LLM在“知情-行动”鸿沟中表现出显著的采样偏差，虽能通过确定性转换或工具模拟实现近似采样，却无法直接可靠地从目标分布抽样；同时揭示了时间相关偏差、位置偏差以及模型规模对采样性能的影响，并提出需要状态化采样器以满足代理系统的需求。

**🔧 技术方法**

采用提示工程、解码参数敏感性分析、序列化与批量采样、PRNG代码生成与执行、分布转换（逆变换/桶化）、链式推理、自动相关性与转移矩阵统计、卡方检验与Kolmogorov–Smirnov检验等多种技术手段对LLM采样行为进行评估。

**📊 数据集**

实验数据主要基于LLM内部生成的随机数与分布，未使用外部真实数据集；通过人工构造的采样任务（如从0-9、[0,1]或N(0,1)抽样）来验证模型的采样分布。

**📈 对比分析**

通过绘制经验分布直方图、计算p值等统计量与理论分布对比，发现大多数LLM的p值接近0，说明采样失败；序列化采样略优，批量采样存在周期性偏差；使用工具+PRNG或分布转换在大模型上可实现p>0.05，证明可行但计算成本高。

**⚠️ 局限性**

局限包括：LLM无法直接可靠采样导致知情-行动缺口；链式推理与解码参数调节无法根本解决问题；PRNG模拟成本高、需保持状态；批量采样引入时间相关偏差；分布转换对模型规模有显著依赖，复杂分布仍表现欠佳；缺乏内置随机性学习机制。

---

## 84. The Human Condition as Reflected in Contemporary Large Language Models

**arXiv ID:** 2604.06206 | [PDF](https://arxiv.org/pdf/2604.06206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 85. MO-RiskVAE: A Multi-Omics Variational Autoencoder for Survival Risk Modeling in Multiple MyelomaMO-RiskVAE

**arXiv ID:** 2604.06267 | [PDF](https://arxiv.org/pdf/2604.06267v1)

**作者:** Zixuan Chen `[一作]` (Binjiang Institute of Zhejiang University), Meng Han `[通讯]` (School of Medicine Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种改进的多模态变分自编码器MO-RiskVAE，用于多发性骨髓瘤患者的生存风险建模。

**💡 创新点**

创新点在于系统评估并确定了在生存监督下的潜变量正则化尺度、后验几何及结构三大设计轴，发现适度放宽KL正则化并结合混合连续-离散（Gumbel-Softmax）潜变量结构能显著提升风险排序。

**🔧 技术方法**

使用的技术包括β-KL、MMD、HSIC正则化、Gumbel-Softmax离散化、Cox比例风险损失、MyeVAE框架以及多模态数据融合。

**📊 数据集**

数据集为MMRF CoMMpass的1143例多发性骨髓瘤患者（含完整多组学的628例），外部验证集包括四个独立微阵列数据集（UAMS-GSE24080、HOVON-65/GMMG-HD4、EMTAB-4032、APEX）。

**📈 对比分析**

通过与原MyeVAE在相同架构与训练协议下对比，MO-RiskVAE在验证集上C-index从0.7419提升至0.7788（+0.0369），并在Kaplan–Meier分析中实现显著的风险分层（HR≈5.95，p≈2.1e-5）。

**⚠️ 局限性**

局限性包括：未能在生存监督下稳定发现离散亚型；模型仍受限于中等规模样本；未探索更大样本或跨域适应的鲁棒性。

---

## 86. Towards the Development of an LLM-Based Methodology for Automated Security Profiling in Compliance with Ukrainian Cybersecurity Regulations

**arXiv ID:** 2604.06274 | [PDF](https://arxiv.org/pdf/2604.06274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 87. Database Querying under Missing Values Governed by Missingness Mechanisms

**arXiv ID:** 2604.06520 | [PDF](https://arxiv.org/pdf/2604.06520v1)

**作者:** Leopoldo Bertossi `[一作]` (Carleton University), Maxime Buron `[通讯]` (CNRS)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对含缺失值的关系数据库，提出基于缺失机制的贝叶斯网络构造块独立概率数据库（BID），并通过匹配世界类定义多种查询答案语义，尤其是最合规类（MC）与最可能类（MP）。

**💡 创新点**

创新点在于将缺失机制建模为贝叶斯网络并引入世界合规性度量，结合合规与概率双维度的类化语义，给出可计算的MC-类与复杂的MP-类，提供了从概率、统计两方面统一的查询答案框架。

**🔧 技术方法**

使用的技术包括贝叶斯网络、块独立概率数据库、匹配世界类、KL/欧氏距离等凸可分离距离、最小成本流等复杂度分析方法。

**📊 数据集**

论文未在真实数据集上实验，仅使用人工构造的示例关系实例及其缺失机制进行理论演示。

**📈 对比分析**

通过复杂度理论比较，MC类的构造与枚举可在多项式时间或多项式延迟完成，而MP类涉及#P难度，实验结果未给出；理论证明表明MC语义在给定概率流推理或约束下可行，MP语义则在大多数情况下不可行。

**⚠️ 局限性**

局限性包括缺乏经验评估、仅考虑特定距离函数、对贝叶斯网络缺失机制的假设较强、MP类求解仍处于NP^PP难度，且实际实现需结合概率数据库引擎。

---

## 88. Beyond Case Law: Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal QA

**arXiv ID:** 2604.06173 | [PDF](https://arxiv.org/pdf/2604.06173v1)

**作者:** Kyubyung Chae `[一作]` (Seoul National University), Taesup Kim `[通讯]` (Seoul National University)

**通讯引用:** 17659 | [OpenAlex ID](https://openalex.org/A5100641870)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个面向法规层级检索与模型安全的基准 SearchFireSafety，评估LLM在韩国火灾安全法规中的检索效果与避免生成虚假答案的能力。

**💡 创新点**

创新点在于构造了结构化的法律引文图与安全意识双源数据集，并提出结构感知重排序（SAR）算法以及部分上下文假设来专门检测模型的中止与不确定性处理。

**🔧 技术方法**

使用了稠密检索模型（Qwen3-Emb、BGE-M3）、传统检索（BM25）、RRF、Rocchio、以及图驱动的结构感知重排（SAR）和检索增强生成（RAG）流水线来评估模型表现。

**📊 数据集**

采用了基于韩国火灾安全法规的同期法律语料库（共4467条文档），并收集了876条专家问答和3395条基于图生成的多跳合成问答，形成完整的数据集。

**📈 对比分析**

与传统检索方法相比，SAR在Recall@20和nDCG@20上提升约15–20%；在多跳问答中，完整上下文下模型准确率可达86%，但在缺失上下文下多模型准确率仅在50%–65%之间，显示安全性不足。

**⚠️ 局限性**

局限性包括仅聚焦韩法火灾安全法规，未涵盖普通法或其他司法体系；合成问答的分布可能与真实查询差异；未考察模型持续更新与人机协作部署等实际应用场景。

---

## 89. ODE-free Neural Flow Matching for One-Step Generative Modeling

**arXiv ID:** 2604.06413 | [PDF](https://arxiv.org/pdf/2604.06413v1)

**作者:** Xiao Shou `[一作]` (Baylor University), Xiao Shou `[通讯]` (Baylor University)

**通讯引用:** 64 | [OpenAlex ID](https://openalex.org/A5112946886)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

直接学习噪声到数据的流映射，使用神经流实现一次前向传播的生成，完全省去 ODE 求解。

**💡 创新点**

提出 Optimal Transport Neural Flow Matching（OT‑NFM），证明最优传输耦合是避免均值崩塌的必要条件，并设计可扩展的批量 OT 与 LOOM 结合的耦合策略。

**🔧 技术方法**

采用神经流架构（ResNetFlow/UNet）、最优传输（批量 OT、LOOM）、线性插值轨迹、谱归一化残差网络等技术。

**📊 数据集**

在二维合成运输基准（Gaussian→Checkerboard、Gaussian→Spiral、Gaussian→Crescent、8‑GMM→2‑Moons）以及 MNIST、CIFAR‑10 图像数据集上进行实验。

**📈 对比分析**

与 iCFM、OT‑CFM（100 NFE）、MeanFlow（1 NFE）等方法对比，OT‑NFM 在二维任务中取得最低 W₂²，CIFAR‑10 的生成质量可与多步方法相媲美，但仅需一次网络评估，推断速度显著提升。

**⚠️ 局限性**

受限于缺乏高级预处理/EDM 方案，生成质量仍低于最新高分辨率多步模型；对大规模高维数据的 OT 计算仍存在成本与可逆性约束。

---

## 90. $S^3$: Stratified Scaling Search for Test-Time in Diffusion Language Models

**arXiv ID:** 2604.06260 | [PDF](https://arxiv.org/pdf/2604.06260v1)

**作者:** Ahsan Bilal `[一作]` (University of Oklahoma), Dean F. Hougen `[通讯]` (University of Oklahoma)

**通讯引用:** 1465 | [OpenAlex ID](https://openalex.org/A5061058579)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于验证器引导的粒子搜索方法S^3，在扩散语言模型（DLM）推理时动态分配计算资源以提升生成质量；

**💡 创新点**

创新点在于将目标分布的奖励倾斜（reward‑tilted Gibbs）与可观测的前向信息函数近似相结合，构建可训练的验证器无标签指引下的粒子搜索框架；

**🔧 技术方法**

利用扩散模型的逆向去噪过程、粒子滤波（SMC）思想、低方差重采样（SSP）以及轻量级无监督验证器对候选文本进行评分；

**📊 数据集**

在四个基准上验证：MATH‑500、GSM8K、ARC‑Challenge 与 TruthfulQA；

**📈 对比分析**

与单轨迹扩散基线和 8‑sample best‑of‑K（BoK）比较，S^3 在 MATH‑500 提升 4.60pp、GSM8K 1.55pp、TruthfulQA 3.08pp，ARC‑Challenge 在细粒度块长度下亦表现优于基线；

**⚠️ 局限性**

主要局限包括对验证器质量与清晰预测的依赖、额外粒子扩展导致的计算开销、以及在多项选择任务中验证器信号相对弱导致的性能波动。

---

## 91. Evidence-Based Actor-Verifier Reasoning for Echocardiographic Agents

**arXiv ID:** 2604.06347 | [PDF](https://arxiv.org/pdf/2604.06347v1)

**作者:** Peng Huang `[一作]` (University at Albany, SUNY), Xin Wang `[通讯]` (University at Albany, SUNY)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研发了一种基于Actor-Verifier框架的EchoTrust，用于可信的心脏超声图像问答，强调结构化证据和可审计推理。

**💡 创新点**

将问答任务转化为证据驱动的可选择推理流程，加入可观测证据、检验和自我重推，提升可解释性与可靠性。

**🔧 技术方法**

采用LoRA参数高效微调共享多模态基础模型，设计Actor/Verifier/Retry Actor三角色，并用结构化推理状态与闭环校验。

**📊 数据集**

在MIMICEchoQA基准上评估，该数据集包含622个超声视频与多选问答，结合临床报告。

**📈 对比分析**

与多种Qwen3‑VL规模模型对比，EchoTrust准确率提升至0.76，显著高于基线0.25‑0.44。

**⚠️ 局限性**

仍可能把原本正确的答案改为错误，且对异常情况的决策稳定性待提升。

---

## 92. Toward a Uniform Algorithm and Uniform Reduction for Constraint Problems

**arXiv ID:** 2604.06335 | [PDF](https://arxiv.org/pdf/2604.06335v1)

**作者:** Libor Barto `[一作]`, Dmitriy Zhuk `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在CSP和PCSP框架下提出并研究了label cover实例的k阶“饱和幂”与向量松弛技术，证明了这些层级方法的等价性，并通过向量松弛算法解决了整数平方问题（D4实例）

**💡 创新点**

引入k阶饱和幂作为提升松弛层级的新途径，构造了向量松弛(minion)并证明其与k阶SDP、BLP、AIP、AC等四种基本松弛的等价性；同时给出了向量松弛对Z_p^2的minion同态映射，首次证明第二阶Z_2松弛可求解Z_4线性方程组

**🔧 技术方法**

利用label cover与CSP的等价转换、minion同态与局部有限性、向量空间映射构造、k线性形式到向量格子映射的Gram矩阵性质、以及Tychonoff紧致性证明等技术

**📊 数据集**

无具体实验数据集，主要以理论证明和D4实例做验证

**📈 对比分析**

通过理论比较展示k阶饱和幂与部分解层级方法等价，向量松弛在解决整数平方问题上表现优越，但在整数域上仍不够强大

**⚠️ 局限性**

局限在于向量松弛对整数域无法解决更高阶线性方程组；对局部有限性假设依赖，且实际实现缺乏对大规模实例的实验评估

---

## 93. ZitPit: Consumer-Side Admission Control for Agentic Software Intake

**arXiv ID:** 2604.06241 | [PDF](https://arxiv.org/pdf/2604.06241v1)

**作者:** Jepson Taylor `[一作]` (VEOX Research Group), Kelli Quinn `[通讯]` (VEOX Research Group)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种面向智能开发环境的消费者端软件接纳控制层，要求外部软件在被授予受保护主机执行权限之前，必须经过可持久化的策略事件记录。

**💡 创新点**

创新点在于将外部工件接纳、仓库打开状态、能力范围化执行以及持久化策略事件统一到同一安全边界，填补了目前“先取后执行”链路中的安全空白。

**🔧 技术方法**

采用Rust实现的ZitPit系统，结合Git smart‑HTTP接纳、受控会话执行、受控出口（DLP）以及对TUF、Sigstore、in‑toto等标准化 provenance 信号的消费；同时利用 Mirage Lab 进行动态分析和证据生成。

**📊 数据集**

使用公开的五个 GitHub 仓库（如 rust-lang、nodejs、python、golang、kubernetes）做基准测试，构建可重复的 benchmark harness；此外还采用公开的攻击样本和工作流用例作为证明。

**📈 对比分析**

通过对直接网络请求、磁盘缓存命中以及热缓存命中的延迟测量，实验显示批准后的不可变 Git 引入路径平均延迟在 433–1062 ms（web）降至 13–16 ms（热缓存），显著快于未受控的公共网络请求，验证了系统的可部署性与性能。

**⚠️ 局限性**

局限性包括：仅完成了 Git smart‑HTTP 接纳、受控会话和受控出口的公开验证；尚未覆盖子模块、LFS、完整克隆、所有包管理器、本地副本以及完整的工作流图；强制中介路径仍是挑战；以及对可信发布者的失效仍未解决。

---

## 94. Team Fusion@ SU@ BC8 SympTEMIST track: transformer-based approach for symptom recognition and linking

**arXiv ID:** 2604.06424 | [PDF](https://arxiv.org/pdf/2604.06424v1)

**作者:** Georgi Grazhdanski `[一作]` (Sofia University St. Kliment Ohridski), Svetla Boytcheva `[通讯]` (Sofia University St. Kliment Ohridski)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出并实现了基于Transformer的方案，用于SympTEMIST任务中的症状实体识别（NER）和实体链接（EL）。

**💡 创新点**

创新点包括：①在专业化的西班牙医学Transformer上进行微调并加入BiLSTM-CRF提升性能；②通过UMLS同义词对训练集进行数据增强；③在实体链接中采用知识库增广、滑动窗口余弦相似度组合，以及对罕见实体进行字符扰动扩增。

**🔧 技术方法**

技术：RoBERTa（PlanTL‑GOB‑ES/roberta‑base‑biomedical‑clinical‑es）与CLIN‑X‑ES模型；BiLSTM；CRF；SapBERT XLMR‑Large 余弦相似度检索；知识库构建与文本增强。

**📊 数据集**

数据集：SympTEMIST 1,000篇西班牙临床报告（train 750，test 250）；UMLS西班牙语同义词集（337k条）；SNOMED‑CT词表与Gazetteer（164k+条）。

**📈 对比分析**

与基线比较：在验证集上，Augmented Spanish RoBERTa + BiLSTM + CRF取得最高 F1 0.738；在测试集上，Augmented Spanish RoBERTa + BiLSTM + CRF 0.732；实体链接最高准确率 58.9%（使用Gazetteer+Train+UMLS知识库 + SapBERT + Sliding Window）。

**⚠️ 局限性**

局限性：①额外预训练在UMLS同义词上未显著提升效果；②实体链接对知识库构建极为敏感，构建错误导致准确率骤降；③对长实体的滑动窗口改进仅提升约2%，且缺乏更系统的多表示融合方法。

---

## 95. WebExpert: domain-aware web agents with critic-guided expert experience for high-precision search

**arXiv ID:** 2604.06177 | [PDF](https://arxiv.org/pdf/2604.06177v1)

**作者:** Yuelin Hu `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 84494 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向专业领域的 WebAgent——WebExpert，在深度浏览前通过专家经验检索与规划来生成领域相关查询，提升答案质量。

**💡 创新点**

①采用 critic‑guided 经验抽取链，将 QA 数据聚类、去重、对齐后归纳为句子级规则；②通过无监督 facet 诱导和规则蒸馏实现轻量化领域 facet 生成；③使用经验条件规划与覆盖感知 SFT 进行查询和检索优化。

**🔧 技术方法**

多视角密度聚类（HDBSCAN/BERTopic）、句子级向量检索、最大边际相关性（MMR）、深度矛盾感知摘要（DeepSeek‑R1）、统一流形近似投影（UMAP）、层次密度聚类、规则蒸馏、对比检索损失、强化偏好学习、轻量化经验门控等。

**📊 数据集**

GAIA、GPQA、HLE 以及 WebWalkerQA 四个多领域基准数据集。

**📈 对比分析**

与 RAG‑QwQ‑32B、Search‑o1‑32B、WebThinker‑32B‑Base 等最强浏览基线对比。WebExpert 在 EM 上平均提升 1.5–3.6 分点，查询精准度 QP@3 提升至 61.8%，页面跳转次数下降至 5.2，证据相关性 nDCG@10 上升 4–6 分。

**⚠️ 局限性**

仍受限于领域先验不足时经验检索召回率、facet 诱导误差及对知识更新的实时性；在极端噪声或快速演化的领域中可能出现查询漂移和证据不足。

---

## 96. Bridging Theory and Practice in Crafting Robust Spiking Reservoirs

**arXiv ID:** 2604.06395 | [PDF](https://arxiv.org/pdf/2604.06395v1)

**作者:** Ruggero Freddi `[一作]` (Manava.plus), Alessio Basti `[通讯]` (G'. d'Annunzio University of Chieti-Pescara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了脉冲储存计算（SNN）中鲁棒性区间（robustness interval）对网络超参数的影响，并将理论临界点与实际性能范围相结合，提供了一种在不确定条件下可操作的调参框架。

**💡 创新点**

①将理论临界点 w_crit 与实际鲁棒性区间相对照，证明 w_crit 可作为稳健搜索起点；②发现网络稀疏度 β 与阈值 θ 对鲁棒性区间宽度的单调影响，并引入等价 (β,θ) 参数对，揭示了性能的参数等价性；③通过对多种拓扑（小世界与 Erdős–Rényi）和多任务验证，证明该结论具普适性。

**🔧 技术方法**

使用 Leaky Integrate‑and‑Fire (LIF) 神经网络，配合小世界和 Erdős–Rényi 连接，采用统计特征与 trace 特征提取，使用单层感知器与随机森林作为读取器，并基于均值场理论计算理论发放率和临界权重。

**📊 数据集**

MNIST（降维后的 6000 张二值图像）与合成球轨迹视频（700 个 100 帧、32×32 的二值视频，分为 7 类）两个任务。

**📈 对比分析**

通过 10 折交叉验证评估准确率、宏 F1 和 Matthews 相关系数，比较不同超参数、拓扑、特征与读取器组合的鲁棒性区间宽度；结果显示 β 越大（网络越稠密）鲁棒区间越窄，θ 越大鲁棒区间越宽，I 的影响最弱；同时大多数情况下 w_crit 落在鲁棒区间内，且大多位于轻微亚临界区域。

**⚠️ 局限性**

实验仅基于 LIF 固定权重模型，未考虑突触可塑性或自适应机制；拓扑探索局限于小世界与部分 Erdős–Rényi，未覆盖更稀疏或更复杂的结构；数据集规模有限，缺乏对更大规模真实数据的验证。

---

## 97. Say Something Else: Rethinking Contextual Privacy as Information Sufficiency

**arXiv ID:** 2604.06409 | [PDF](https://arxiv.org/pdf/2604.06409v1)

**作者:** Yunze Xiao `[一作]` (Carnegie Mellon University), Weihao Xuan `[通讯]` (University of Tokyo)

**通讯引用:** 218 | [OpenAlex ID](https://openalex.org/A5059658930)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究LLM代理在代写用户消息时如何保护用户隐私，提出信息足够性（IS）任务框架，新增自由文本伪名化策略，并设计基于多轮对话的隐私评估流程。

**💡 创新点**

创新点在于：①将伪名化作为第三种隐私策略引入LLM对话；②把隐私评估从单条消息提升到多轮跟进情景；③构建792个多场景基准，系统比较三种策略在不同权力关系与敏感度类别下的表现。

**🔧 技术方法**

采用前沿LLM（Gemini 3.1 Pro、GPT‑5.4、Qwen3‑8B等）生成回复，使用Deepseek‑v3.2模拟器进行两轮非对抗性跟进，利用LLM判断器评估隐私泄露、隐蔽性、效用，并计算IS‑AD分数；统计检验包括Krippendorff's α、Wilcoxon、Kruskal‑Wallis等。

**📊 数据集**

数据集来源于PrivacyLens 493条隐私敏感种子，扩展生成792个多轮场景，涵盖三种权力关系（机构、同行、亲密）与三类敏感度（歧视风险、社会成本、边界）。

**📈 对比分析**

在多轮评估中，伪名化在隐私‑效用平衡上最优（平均IS‑AD 0.764），优于抑制（0.730）和一般化（0.664）；单条消息评估误判策略优劣，伪名化对跟进泄漏更稳健；一般化被Pareto支配，抑制在某些低风险场景与基线相近。

**⚠️ 局限性**

局限性包括：评估依赖LLM判断器，可能漏检细微违规；跟进轮数仅两次，未考察对抗性接收者；场景基于美国隐私规范，跨文化适用性尚待验证。

---

## 98. The complexity of bisimilarity on pointmass processes

**arXiv ID:** 2604.06443 | [PDF](https://arxiv.org/pdf/2604.06443v1)

**作者:** Martín Santiago Moroni `[一作]` (Universidad de Buenos Aires), Pedro Sánchez Terraf `[通讯]` (Universidad Nacional de Córdoba)

**通讯引用:** 123 | [OpenAlex ID](https://openalex.org/A5063024175)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

研究了在可数和无穷可数Borel空间上的非确定性标记马尔科夫过程（NLMP）中，bisimilarity（行为等价）的描述复杂度，证明了在有限支持、统一枚举的情形下该关系是analytic；在可数Kripke框架下可由可数结构分类；对于有限深度（well‑founded）过程，bisimilarity是Borel；并给出E₀关系的下界归约，说明对小秩过程不存在可计数、可测的模态逻辑片段能够完整表征bisimilarity。最后将这些结果推广到统一可测LTS并研究其well‑founded部分。

**💡 创新点**

①首次将可测性与描述集理论相结合，给出了NLMP中bisimilarity的完整复杂度上界与下界；②证明可数Kripke框架下bisimilarity可由可数结构分类，提供了新的结构化描述；③利用E₀归约得到对模态逻辑的不可判定性限制，揭示了逻辑与行为等价之间的根本差距；④将分析扩展到统一可测LTS，构造了Borel归约到一阶结构的同构。

**🔧 技术方法**

使用描述集理论（analytic/Borel/Π₁¹秩）、ω‑扩张树构造、模态逻辑（可数并/交）、可测函数与σ‑代数的可测性论证、外部与内部bisimilarity的等价性证明、以及对NLMP的子结构与约束分析。

**📊 数据集**

无实验数据，全部为理论推导与构造证明。

**📈 对比分析**

本研究不涉及实验比较或性能评估，主要提供理论复杂度和可测性结果；相比传统的可计数马尔科夫决策过程，已证明更一般的无穷可数空间情形下bisimilarity的可测性与Borel化性。

**⚠️ 局限性**

仅针对图像可数且统一枚举（或点质量）NLMP，无法推广到一般非图像可数或非统一可测的NLMP；对一般NLMP的analytic性仍未知；在子结构约束与归约时需要较强假设，实际应用时可能受限；未给出非统一可测LTS的具体例子。

---

## 99. Automating Database-Native Function Code Synthesis with LLMs

**arXiv ID:** 2604.06231 | [PDF](https://arxiv.org/pdf/2604.06231v1)

**作者:** Wei Zhou `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 19226 | [OpenAlex ID](https://openalex.org/A5075948251)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套名为 DBCooker 的基于 LLM 的自动化系统，能够在 SQLite、PostgreSQL 和 DuckDB 三大主流数据库上根据函数声明自动生成、实现并验证数据库本地函数的完整实现。

**💡 创新点**

创新点在于：① 通过函数特征化模块（多源声明收集、层级单元识别、跨单元引用分析）精准定位需要实现的函数单元与引用；② 采用伪代码式计划生成和填空式生成模型结合概率先验，实现对函数实现结构的引导与自校；③ 三层级验证（语法、合规、语义）与自适应工具编排相结合，动态重构工作流并利用历史轨迹提升效率。

**🔧 技术方法**

技术手段包括：大型语言模型（Claude、GPT‑5 等）+ 代码生成插件；静态分析与依赖图构建；图/对比式单元剪枝；填空式生成模型（带组件感知与概率衰减）；得分过滤的伪计划生成；三层级验证脚本；混合规划与记忆增强的工具编排。

**📊 数据集**

使用的评测数据集为 75 个 SQLite、145 个 PostgreSQL、128 个 DuckDB 原始函数的实现代码，并对每个数据库扩展 4~10 个从其它数据库迁移过来的新函数（如 covar_pop、bool_or、century 等）。

**📈 对比分析**

与最先进的 LLM、RAG 及 Agent 方法（Claude Code、Qwen Code、TRAE、GPT‑5、Claude Opus 等）进行对比。实验表明 DBCooker 在 Acc_EXE 与 Acc_RES 上平均提升约 34.5%（SQLite 78.9%/65.2%，PostgreSQL 78.6%/58.6%，DuckDB 83.7%/67.4%），并在易难分级和函数类别上均保持最高且最稳定的准确率。

**⚠️ 局限性**

局限性包括：① 仍需依赖 LLM 的概率生成，可能出现幻觉或细节错误；② 对极大规模、结构高度碎片化的数据库代码仍有识别与上下文提取的挑战；③ 需要手工调参（如权重 α、β、γ）和对不同数据库版本的适配，难以做到完全零维护。

---

## 100. Asynchronous Distributed Bandit Submodular Maximization under Heterogeneous Communication Delays

**arXiv ID:** 2604.06430 | [PDF](https://arxiv.org/pdf/2604.06430v1)

**作者:** Pranjal Sharma `[一作]` (University of Michigan), Vasileios Tzoumas `[通讯]` (University of Michigan)

**通讯引用:** 962 | [OpenAlex ID](https://openalex.org/A5052733656)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计了一种分布式在线子模最大化框架，支持异步本地时钟与异质通信延迟，允许各智能体在收到部分邻居信息后立即做决策并通过中间估计不断校正。

**💡 创新点**

创新点包括：① 在不需要同步全局时钟且通信延迟异质的情况下实现可扩展协调；② 引入“中间更新”机制，让智能体在部分信息到达时即刻学习，显著缩短有效延迟；③ 证明了在异步情形下仍可保持与同步中心化方案相近的近似比例，并给出了与网络拓扑、延迟和时钟误差相关的误差项。

**🔧 技术方法**

核心技术包括：
- 对抗性带延迟反馈带中间更新的多臂赌博机模型。
- 基于 EXP3 的重要性加权估计与权重更新。
- 通过时间戳奖励函数 F 兼顾执行时间与评估时间，满足子模性与 Lipschitz 条件。
- 证明分析结合子模曲率、网络稠密度、最大延迟与时钟漂移。

**📊 数据集**

使用仿真数据：16 台摄像机按 4×4 格网布置，每台可选择 8 个朝向；80 个目标聚类移动，速度 1.0 单位/步，方向每 30 步重新采样。延迟在 0–d̅ 之间均匀采样，实验覆盖 d̅ = 1, 5, 10, 20, 30。

**📈 对比分析**

与同步版本（所有更新在所有邻居信息到达后才执行）的比较显示：在小延迟（d̅ ≤ 5）下两者表现相近；当 d̅ ≥ 10 时，本文方法在覆盖率上领先 3–5 个目标（约 20%），而同步版本几乎无法学习。实验采用 20 次 Monte‑Carlo 迭代，平均 ±95% 置信区间。

**⚠️ 局限性**

局限性包括：
- 对延迟上界 d̅ 与时钟误差 ρ 的假设需要预先估计；
- 在环境快速变化时，仍受限于 EXP3 的学习速率，难以完全捕捉快速动态；
- 证明中使用的误差项对极端网络拓扑或高度异质延迟可能过于保守；
- 目前仅在仿真场景验证，缺乏真实硬件实验验证。

---

## 101. High-Precision Estimation of the State-Space Complexity of Shogi via the Monte Carlo Method

**arXiv ID:** 2604.06189 | [PDF](https://arxiv.org/pdf/2604.06189v1)

**作者:** Sotaro Ishii `[一作]` (University of Tokyo), Tetsuro Tanaka `[通讯]` (University of Tokyo)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5001822158)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

通过蒙特卡洛抽样与逆向搜索，估计日本将棋的可达位置数量。

**💡 创新点**

提出以“KK（仅王）”位置集合为逆搜索目标，并使用贪心最佳优先搜索来判定可达性，显著降低搜索深度。

**🔧 技术方法**

采用蒙特卡洛抽样、逆向最佳优先搜索、启发式函数 H 以及位置可达性判定算法。

**📊 数据集**

随机生成 5 亿（标准将棋）和 1 亿（迷你将棋）个候选位置，作为抽样数据集。

**📈 对比分析**

与以往的上下界估计相比，本方法误差仅 ±0.06%，在 128 GB 内完成，标准将棋耗时约 131 小时，成功率高且搜索节点线性增长。

**⚠️ 局限性**

局限性：仍依赖逆向搜索深度，极端不可达位置的最大回溯深度未知；仅考虑 Black 开局且采用水平镜像简化，未覆盖所有规则细节。

---

## 102. GS-Surrogate: Deformable Gaussian Splatting for Parameter Space Exploration of Ensemble Simulations

**arXiv ID:** 2604.06358 | [PDF](https://arxiv.org/pdf/2604.06358v1)

**作者:** Ziwei Li `[一作]` (Ohio State University), Han-Wei Shen `[通讯]` (Ohio State University)

**通讯引用:** 6405 | [OpenAlex ID](https://openalex.org/A5065630217)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于可变形三维高斯散射（Deformable 3D Gaussian Splatting）的可视化代理，用于在后期探索中快速合成任意仿真参数和可视化参数下的视角一致图像。

**💡 创新点**

创新点包括：①引入可重用的 Canonical Gaussian 字段，并通过两级参数驱动变形网络（先针对仿真参数，再针对可视化任务）实现几何与外观的显式分离；②模块化设计使得模型在新视角、仿真参数、等值面/转移函数等多维空间中均能高质量实时渲染；③通过差分渲染与图像对比训练，仅使用图像数据而无需原始体数据，显著降低 I/O 与存储成本。

**🔧 技术方法**

使用技术包括：三维高斯散射（3DGS）表示、两级变形网络（F_sim 与 F_vis）、空间-参数编码器、轻量级多头解码器、L1+SSIM 损失以及变形正则化；实现基于 PyTorch 的差分渲染管线。

**📊 数据集**

实验数据集有四个：Nyx（宇宙学 512³）、MPAS-Ocean（海洋 1536×768×768）、XCompact3D（湍流 256³）和 CloverLeaf3D（冲击波 128³），每个数据集包含多维仿真参数、视角（icosphere 252）与可视化参数（转移函数或等值面）组合。

**📈 对比分析**

与 InSituNet、VisNeRF、K-Planes、4DGS-HD 四个基线进行对比。GS‑Surrogate 在 PSNR、SSIM、LPIPS 以及模型尺寸、训练时间和推理时间上普遍优于对手，尤其在高维仿真参数和等值面推断中保持最佳或次优表现，且推理时间仅 0.05–0.07 s/图，模型体积仅 46–50 MB。

**⚠️ 局限性**

主要局限包括：①对高密度小尺度结构（如 CloverLeaf3D）仍需更多高斯原语或纹理增强以提升细节；②等值面提取时的几何细节仍有模糊；③目前仅支持透明度映射的转移函数，无法处理颜色映射等更丰富的可视化变换；④逆向参数搜索功能尚未实现。

---

## 103. CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale

**arXiv ID:** 2604.06245 | [PDF](https://arxiv.org/pdf/2604.06245v1)

**作者:** Jichao Fang `[一作]` (Northern Illinois University), Wei Luo `[通讯]` (Northern Illinois University)

**通讯引用:** 6872 | [OpenAlex ID](https://openalex.org/A5043444179)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将撞击坑分析视为实例级图像检索任务，提出CraterBench‑R检索基准，并设计无训练的实例‑token聚合与两阶段检索管线。

**💡 创新点**

①创建首个专门用于撞击坑检索的公开基准；②提出训练‑free 的实例‑token聚合，压缩ViT patch token；③在ViT token上实现late‑interaction检索；④通过单向量粗排+实例‑token精排实现高效大规模检索。

**🔧 技术方法**

自监督ViT（DINO/MarsDINO）、CLS/GeM池化、ColBERT式late‑interaction、token聚合（seed选择+余弦最近邻聚合）、FAISS索引、INT8/产品量化。

**📊 数据集**

Mars CTX影像+Robbins撞击坑目录，构成CraterBench‑R（25k身份、50k图库、5k查询）。

**📈 对比分析**

与单向量池化、监督度量学习、经典聚类（VLAD/NetVLAD）、per‑image K‑means 等方法对比。dense‑token late‑interaction在K=64时mAP达0.76，two‑stage 500候选时mAP 0.695，恢复约90%全量检索精度，显著优于单向量与传统聚类。

**⚠️ 局限性**

仅有2张视角限制了监督微调效果；聚合方法仅基于余弦相似度，可能忽略更细粒度关系；two‑stage仍需粗排；对不同星体或更大尺度的推广尚待验证。

---

## 104. The Master Key Hypothesis: Unlocking Cross-Model Capability Transfer via Linear Subspace Alignment

**arXiv ID:** 2604.06377 | [PDF](https://arxiv.org/pdf/2604.06377v1)

**作者:** Rishab Balasubramanian `[一作]` (Virginia Tech), Tu Vu `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种完全无训练、无标签的框架，利用源模型在不同提示或训练状态下的内部激活差异提取能力方向（Master Key），并通过低秩线性子空间对齐将该方向迁移到目标模型，在推理时对目标模型内部表示进行干预，从而在不重新训练的前提下提升目标模型的推理能力。

**💡 创新点**

创新点：
1) 引入Master Key Hypothesis，认为能力对应于低维子空间中的方向，可在不同规模、不同家族的模型之间通过线性映射传递；
2) 设计无训练、无标签的能力方向提取与低秩对齐方法；
3) 在推理时通过对隐藏状态的标准化干预实现能力激活，避免了梯度更新、参数对齐或大规模前向推理。

**🔧 技术方法**

技术：
- 通过对比“锁定”与“解锁”源模型的激活，求取能力方向；
- 对源/目标模型在共享提示下的隐藏状态做SVD提取k维子空间，学习k×k线性对齐矩阵；
- 将对齐后的方向投影到目标模型的每一层，并在推理时按公式 𝐡̃=𝐡/‖𝐡‖+α·方向/‖方向‖ 进行干预；
- 采用不同聚合器（均值、主成分）及低秩超参数k、干预强度α 的网格搜索。

**📊 数据集**

数据集：
- 推理任务：GSM8K、MATH、SVAMP、AGIEval‑Math、Deepmind‑Math、Minerva‑Math、OlympiadBench；
- 训练/对齐用的无标签问句集合：小规模不超过数百条，覆盖各任务的典型输入；
- 评估用的公开标准数据集。

**📈 对比分析**

比较方法与性能：
- 对比同模型在直接提示（无干预）下的基线；
- 对比同模型在已完成后训练（如 instruction‑tuned、RLVR 等）下的表现；
- 结果示例：从 7B→14B 的 CoT 方向迁移在 MATH 上提升 12.1%；从 7B→14B 的数学推理方向迁移 AGIEval‑Math 从 61.1% 提升至 71.3%，超过 14B post‑trained 的 67.8%；
- 迁移效果呈规模非对称，小→大提升显著大于大→小；
- 目标模型的推理长度显著增加，表明生成了更完整的思考轨迹。

**⚠️ 局限性**

局限性：
- 仅能激活在目标模型中已潜在存在的能力（atomic）；若目标模型缺乏相应表征，迁移效果有限；
- 对于非 atomic 的高级能力仍需源模型已充分学习且目标模型足够大，迁移效果不稳定；
- 对齐过程依赖共享提示和相对层映射，跨家族/异构模型的通用性尚未彻底验证；
- 需要手动调节 k、α 等超参数，且对齐质量与所用问句数量相关；
- 无理论证明，仍是经验性假设，无法解释为何某些能力可迁移而其他不行。

---

## 105. LLM Spirals of Delusion: A Benchmarking Audit Study of AI Chatbot Interfaces

**arXiv ID:** 2604.06188 | [PDF](https://arxiv.org/pdf/2604.06188v1)

**作者:** Peter Kirgis `[一作]` (Princeton University), Zeynep Tufekci `[通讯]` (Princeton University)

**通讯引用:** 10203 | [OpenAlex ID](https://openalex.org/A5044522704)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对ChatGPT-4o和ChatGPT-5在聊天界面与API接口下的多轮对话进行行为审计，评估谐音、虚假、升级等负面行为，并对比人类评审与LLM评判。

**💡 创新点**

创新点在于①比较聊天界面与API的行为差异；②在真实聊天环境下评估多轮对话的时间动态；③跟踪同一模型在不同时间点的行为变化。

**🔧 技术方法**

采用了SpiralBench框架、手工模拟用户（Kimi K2）、人类评审与GPT-5评判，统计行为强度与置信区间。

**📊 数据集**

使用了14个种子提示（疫苗、UFO、精神病、伪科学等）共20轮对话，产生2240条人类回复与1120条LLM回复。

**📈 对比分析**

通过平均行为强度和95%置信区间对比，ChatGPT-5在聊天界面中负面行为减少约3倍；API接口表现更极端；同一模型不同时间点行为相差颠倒。

**⚠️ 局限性**

限制包括样本量小、仅两款聊天机器人、20轮对话短、使用模拟用户而非真实对话、未考虑多会话/多日动态、行为标签互评一致性中等。

---

## 106. Skin-Deep Bias: How Avatar Appearances Shape Perceptions of AI Hiring

**arXiv ID:** 2604.06187 | [PDF](https://arxiv.org/pdf/2604.06187v1)

**作者:** Ka Hei Carrie Lau `[一作]` (Technical University of Munich), Enkelejda Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 11403 | [OpenAlex ID](https://openalex.org/A5008809634)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在实时生成的嵌入式对话代理（Avatar）下，探究种族和性别匹配对AI招聘面试中被试的信任、公平感知与偏见的影响。

**💡 创新点**

首次将大模型驱动的真实感ECA与眼动追踪及情感分析相结合，研究身份匹配的交叉效应，并发现部分匹配会降低对公平的感知。

**🔧 技术方法**

使用HeyGen生成器与GPT‑4o‑mini进行实时语音交互，搭配RealEye web‑based眼动追踪与spaCy‑TextBlob情感分析。

**📊 数据集**

采用来自Prolific的215名自报为黑人/白人、男性/女性的参与者，收集对话文本、眼动记录及自评问卷数据。

**📈 对比分析**

采用2×2随机分配设计，对信任、公平感知、偏见等自评量表使用ART‑ANOVA比较，结果显示种族不匹配显著提升偏见感知，部分匹配降低分配正义，眼动焦点增多；信任得分总体高且无显著差异。

**⚠️ 局限性**

局限性包括样本仅限英文WEIRD受试者，身份维度仅设为黑人/白人与男性/女性，单一avatar可能带来面孔特征偏差，实时对话的ASR误差与眼动精度受限，且实验未涉及真实就业后果。

---

## 107. Inference-Time Code Selection via Symbolic Equivalence Partitioning

**arXiv ID:** 2604.06485 | [PDF](https://arxiv.org/pdf/2604.06485v1)

**作者:** David Cho `[一作]` (Purdue University), Ananth Grama `[通讯]` (Purdue University)

**通讯引用:** 12948 | [OpenAlex ID](https://openalex.org/A5019202832)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Symbolic Equivalence Partitioning（SEP），一种利用符号执行在推理时对 LLM 生成的代码进行语义分区并从主分区中选取代表的最佳答案的方法；

**💡 创新点**

创新点在于用符号执行直接比较候选程序的功能等价性，并通过 SMT‑约束剪枝减少路径爆炸，避免了传统方法对外部测试或学习型验证器的依赖；

**🔧 技术方法**

核心技术包括符号执行（CrossHair+Z3）、SMT 约束注入、基于等价类的贪心划分以及轻量化的输入/输出案例过滤；

**📊 数据集**

使用 HumanEval+（164 题）和 LiveCodeBench（438 题）两个竞争编程生成基准；

**📈 对比分析**

在 N=10 的设置下，SEP 的 Pass@10 分别提升至 HumanEval+ 0.803（从 0.728）和 LiveCodeBench 0.604（从 0.516），在 8 种模型与 3 个采样预算中取得最强或相同于最强的非 Oracle 性能，并且比 CodeT 的运行成本低；

**⚠️ 局限性**

主要局限包括符号执行受限于搜索预算导致的等价判断不完整、对 Python 的依赖、对大规模候选池的计算开销、以及对隐式域约束的捕捉不完整。

---

## 108. Benchmarking LLM Tool-Use in the Wild

**arXiv ID:** 2604.06185 | [PDF](https://arxiv.org/pdf/2604.06185v1)

**作者:** Peijie Yu `[一作]` (Tencent), Feng Zhang `[通讯]` (Tencent)

**通讯引用:** 6677 | [OpenAlex ID](https://openalex.org/A5087050304)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 WildToolBench——基于真实用户交互日志构建的多轮、多步工具使用基准，重点关注组合任务、隐藏意图和指令转换三大挑战。

**💡 创新点**

创新点包括：① 以真实用户行为为核心，而非人工设计的极端任务；② 通过人机标注构造 256 场景、1024 任务，涵盖四种任务类型；③ 引入枚举‑匹配‑打分（enumerate‑match‑score）评估工具调用拓扑，提供 OP 与 AP 两个细粒度指标。

**🔧 技术方法**

使用技术包括：多轮对话建模、工具调用拓扑枚举与递归匹配、LLM 多代理生成示例、细粒度评估框架（任务/会话准确率、OP、AP、转移频率分析），以及基于人类专家的手工标注与验证。

**📊 数据集**

数据集：WildToolBench，包含 256 场景、1024 任务、400 条公开 API（总计约 1600 个工具），每个场景为四轮对话并混合 4 种任务类型（单工具、多工具、对话、澄清）。

**📈 对比分析**

对比方法：评测 57 款主流 LLM（专有、开源通用、开源专用），使用原生函数调用格式，按任务类型、任务顺序、隐藏意图、指令转换频率细分。结果显示：最优 session accuracy < 15%，任务准确率 < 60%，工具调度 OP 最高约 43%，AP 约 25%。专有模型略优，推理型模型表现更好。

**⚠️ 局限性**

Limitation：人工标注限制数据规模与可扩展性；任务长度受限于人工审核；缺少完全自动化的合成环境；未能覆盖所有工具与更长对话场景，导致对真实大规模交互的代表性不足。

---

## 109. The Unreasonable Effectiveness of Data for Recommender Systems

**arXiv ID:** 2604.06420 | [PDF](https://arxiv.org/pdf/2604.06420v1)

**作者:** Youssef Abdou `[一作]` `[通讯]` (University of Siegen), Youssef Abdou (University of Siegen)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了推荐系统在不同训练数据规模下的性能变化，构建可复现的Python评估流程，采用绝对分层抽样对数据进行采样，使用NDCG@10评估模型，并分析是否存在饱和点。

**💡 创新点**

系统性在11个大规模公开数据集上比较10种工具-算法组合的规模效应，采用min–max归一化跨组比较，并提出后期斜率指标检测饱和，结果显示大多数情况下性能随数据增大持续提升，无明显饱和。

**🔧 技术方法**

Python 3.11、LensKit、implicit、NumPy、pandas、Matplotlib、SLURM作业调度、Singularity容器，结合绝对分层用户抽样、min–max归一化和后期斜率分析。

**📊 数据集**

11个公开交互数据集，规模至少700万条交互，包括常见的MovieLens、Amazon等大型用户–物品反馈数据集。

**📈 对比分析**

对每个工具–算法–数据集组合在100k–100M不同样本大小下训练，测量NDCG@10；通过min–max归一化后绘制散点图和后期斜率分布，发现绝大多数组在最大样本时获得最高NDCG，后期斜率大多为正，表明性能持续提升，无明显饱和。

**⚠️ 局限性**

仅评估离线传统推荐算法和标准用户–物品反馈；未涉及深度学习模型及其更大规模数据的环境成本；实验受限于可用资源，极端大规模或特定算法（如NeuMF）在部分数据集上训练受限；忽略了实时/在线指标和非算法因素（如GUI）对性能的影响。

---

## 110. Neural Computers

**arXiv ID:** 2604.06425 | [PDF](https://arxiv.org/pdf/2604.06425v1)

**作者:** Mingchen Zhuge `[一作]` (Meta AI), Jürgen Schmidhuber `[通讯]` (KAUST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将计算、内存和 I/O 统一到一个学习的隐藏状态中的“神经计算机”（NC）原型，并在 CLI 与 GUI 两种接口上实现并评估。

**💡 创新点**

创新点在于：① 用视频生成模型（Wan2.1 + VAE）构建可执行的隐藏运行时；② 设计多种动作注入和编码方式，验证动作对界面渲染的影响；③ 通过复现式提示（reprompting）显著提升符号算术推理性能；④ 提出了从 NC 到完全神经计算机（CNC）的可行路线图。

**🔧 技术方法**

技术核心包括：Diffusion Transformer（Wan2.1）做状态更新，VAE 做视频编码/解码；CLIP + T5 作为文本与图像的条件输入；多种动作编码（raw‑action、meta‑action）与注入模式（外部、token、残差、内部交叉注意力）；以及对齐策略和对比学习辅助训练。

**📊 数据集**

使用了两个数据集：CLIGen（General：1,100 h公开终端日志；Clean：≈120 k scripted Docker 运行）和 GUIWorld（≈1,500 h Ubuntu 22.04 GUI 录制，包含 Random‑Slow、Random‑Fast 与 Claude‑CUA 目标导向轨迹）。

**📈 对比分析**

与基线（未训练的 VAE/Diffusion）相比，NC 在 PSNR/SSIM/LPIPS 上显著提升；在 CLI 任务中 OCR 文字精度从 0.03 提升至 0.54，算术探针准确率从 4% 通过复现式提示跃升至 83%；在 GUI 任务中，深入注入（内部交叉注意力）使 SSIM_+15 达 0.863、FVD_+15 降至 14.5，显著优于外部或 token‑级注入；而随机探索数据的性能低于少量目标导向数据，强调数据质量至关重要。

**⚠️ 局限性**

主要限制包括：① 符号推理仍不稳定，依赖强制性提示；② 长期执行一致性、可重用性与治理机制缺失；③ 受限于视频模型的容量，难以实现 Turing 完备性与通用可编程性；④ 需要更高质量、结构化的交互日志与更深层次的动作编码；⑤ 对硬件与通用计算机的兼容性与可扩展性尚未成熟。

---

## 111. BiScale-GTR: Fragment-Aware Graph Transformers for Multi-Scale Molecular Representation Learning

**arXiv ID:** 2604.06336 | [PDF](https://arxiv.org/pdf/2604.06336v1)

**作者:** Yi Yang `[一作]` (University of Texas at Dallas), Ovidiu Daescu `[通讯]` (University of Texas at Dallas)

**通讯引用:** 1706 | [OpenAlex ID](https://openalex.org/A5078438902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 BiScale‑GTR，一种自监督双尺度（原子级 GNN + 碎片级 Transformer）分子表示学习框架，结合 graph‑BPE 片段分词，实现多层次结构推理。

**💡 创新点**

创新点包括：① 基于 Weisfeiler‑Lehman 哈希的 graph‑BPE 片段分词，保证片段一致性、化学有效性并支持 OOV 递归分解；② 并行 GNN‑Transformer 结构，将原子级特征池化成 fragment embedding 并与 token embedding 门控融合；③ 在 Transformer 中加入结构感知注意力偏置（连通性、最短路径、键类型）；④ 通过自监督的 mask‑fragment 预训练和 attention‑rollout 解释性验证，展示多尺度推理的优越性。

**🔧 技术方法**

使用的技术包括：GIN 图神经网络、Transformer 编码器、WL 哈希、化学有效性过滤、self‑supervised masked fragment prediction、门控融合、结构感知注意力、AdamW 优化、t‑SNE + NMI 可视化、attention‑rollout 解释。

**📊 数据集**

主要数据集：MoleculeNet（7 个分类任务）、PharmaBench（9 个 ADMET 回归任务）、Long Range Graph Benchmark（Peptides‑func、Peptides‑struct）。

**📈 对比分析**

与 GraphMVP、GraphMAE、Mole‑BERT、GraphFP、SimSGT、MORE、GraphGPS+LAC 等多种 GNN/Transformer/fragment 基线进行对比。BiScale‑GTR（Molecule）在 MoleculeNet 7 个任务中 4 个获得最高 ROC‑AUC；BiScale‑GTR（Fragment）在 BBBP、ToxCast 等局部任务上最佳；PharmaBench 上 5/9 任务最佳；LRGB 上 Peptides‑func AP 达 0.6717，Peptides‑struct MAE 0.2621，均超越同类基线。

**⚠️ 局限性**

局限性：① 仅基于 2D 图结构，未利用 3D 构象信息；② 对长链 peptide 的 OOV 递归分解 fallback 率相对较高；③ GNN 需要保持浅层，深层会导致过平滑；④ 片段词表受限于训练集（430K 分子），对新颖结构的泛化仍受限；⑤ 解释性虽显著，但仍主要聚焦片段级别，细粒度电子效应仍难以捕捉。

---

## 112. A Benchmark of Classical and Deep Learning Models for Agricultural Commodity Price Forecasting on A Novel Bangladeshi Market Price Dataset

**arXiv ID:** 2604.06227 | [PDF](https://arxiv.org/pdf/2604.06227v1)

**作者:** Tashreef Muhammad `[一作]` (Southeast University), Muhammad Ibrahim `[通讯]` (University of Dhaka)

**通讯引用:** 4325 | [OpenAlex ID](https://openalex.org/A5028428532)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文先构建并公开了AgriPriceBD数据集（5种孟加拉农产品的每日零售中位价），随后在该数据集上对七种短期预测模型（包括经典统计模型和深度学习模型）进行系统评估。

**💡 创新点**

创新点在于：①首次发布多品种、每日零售价格公开基准；②对Prophet、Informer以及Time2Vec等在发展中经济市场中的失败模式做负面结果报道；③通过Diebold–Mariano检验提供统计显著性比较，揭示价格可预测性异质性。

**🔧 技术方法**

技术手段包括：LLM辅助的PDF提取管道；时间序列模型：Naïve Persistence、SARIMA、Prophet、BiLSTM、Transformer、Time2Vec‑Transformer、Informer；以及滑动窗口、Min‑Max归一化、Adam优化、Huber损失、Early Stopping、DM检验等。

**📊 数据集**

使用的数据集为AgriPriceBD，涵盖2020年7月至2025年6月共1,779天/品种，包含大蒜、鹰嘴豆、青椒、黄瓜和甜南瓜的每日零售中位价。

**📈 对比分析**

比较方法：对每个品种做80/10/10的时间序列划分，采用90天滚动窗口预测14天，评价指标为MAE、RMSE、MAPE，并用DM检验评估模型间差异。结果显示：Naïve Persistence在随机游走商品中表现最佳；BiLSTM在非平稳商品（大蒜、鹰嘴豆）上最优；Transformer优于Time2Vec；Prophet在所有商品上系统失效；Informer产生异常波动且预测不稳定。

**⚠️ 局限性**

局限性包括：样本量有限（约1,400个训练窗口）；仅使用单变量时间序列；未加入外部特征（气象、进出口等）；只做一次时间划分，缺乏多窗口验证；LLM提取的准确性未进行正式评估。

---

## 113. Hyperfastrl: Hypernetwork-based reinforcement learning for unified control of parametric chaotic PDEs

**arXiv ID:** 2604.06497 | [PDF](https://arxiv.org/pdf/2604.06497v1)

**作者:** Anil Sapkota `[一作]` (University of Tennessee), Omer San `[通讯]` (University of Tennessee)

**通讯引用:** 5441 | [OpenAlex ID](https://openalex.org/A5085671233)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在Kuramoto–Sivashinsky（KS）一维偏微分方程中，提出并实现了 hyperFastRL，一个利用超网络对控制策略权重进行参数化的强化学习框架，实现了跨参数（μ）泛化的单一控制器。

**💡 创新点**

创新点包括：① 用超网络将物理参数直接映射到演员与评论器权重，实现参数与空间反馈解耦；② 在此基础上结合 FastTD3 与分布式（Truncated Quantile Critic）技术，抑制在混沌奖励分布下的过估计与不稳定；③ 对超网络编码器进行三种形式（残差MLP、随机傅里叶特征、ActNet-KAN）比较，发现基于周期基的 KAN 具有更稳健的外推性能。

**🔧 技术方法**

主要技术：超网络（Hypernetwork）参数化、FastTD3/TD3 训练流程、Truncated Quantile Critic（TQC）分布式价值估计、并行 GPU 环境（1024 KS 例子）、谱法求解 KS（ETDRK4 + 3/2 规则）、随机 Fourier Features、ActNet-KAN、经验回放、归一化与惰性梯度更新。

**📊 数据集**

数据集：在 1D KS 方程上以 μ∈[-0.225,0.225] 的 19 个等距取值进行训练（见 19‑点网格），测试集包含 7 个训练时出现的 μ、1 个插值 μ=0.1125、1 个轻度外推 μ=-0.25，全部通过并行 GPU 进行高频模拟生成。

**📈 对比分析**

比较方法：在相同的 FastTD3/TQC 训练管线、相同随机种子（5 种）、相同采样/更新比例下，分别评估 MLP、Fourier、KAN 三种超网络编码器。指标包括训练/评估回报、最终测试回报区间、方差。结果显示：KAN 在所有设定下保持最低方差、最佳外推区间，Fourier 与 MLP 训练/评估回报相近，但外推性能波动更大；GS=2（梯度更新比例）在保持高效的同时获得与 GS=4 相近的性能，且计算成本更低。

**⚠️ 局限性**

局限性：仅在 1D KS（有限参数范围）验证，外推测试仅为轻度外推；实验只用 5 个随机种子，缺乏更广泛的统计显著性；未与传统模型预测控制或在线自适应控制做直接对比；高并行度训练仍需大量 GPU 资源；对更高维度流体动力学系统的可推广性尚未评估。

---

## 114. The End of the Foundation Model Era: Open-Weight Models, Sovereign AI, and Inference as Infrastructure

**arXiv ID:** 2604.06217 | [PDF](https://arxiv.org/pdf/2604.06217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 115. ClawLess: A Security Model of AI Agents

**arXiv ID:** 2604.06284 | [PDF](https://arxiv.org/pdf/2604.06284v1)

**作者:** Hongyi Lu `[一作]` (Southern University of Science and Technology), Fengwei Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 1475 | [OpenAlex ID](https://openalex.org/A5101886601)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了针对基于大语言模型的自治AI代理的安全框架，能够在敌对代理的最坏情况威胁模型下，通过形式化验证的细粒度安全策略实现对代理行为的动态约束。

**💡 创新点**

创新点在于将形式化安全模型与BPF系统调用拦截的用户空间内核相结合，实现了安全策略的可验证性与可执行性，并在不依赖代理内部实现的前提下提供根本安全保证。

**🔧 技术方法**

使用了形式化安全模型、BPF（Berkeley Packet Filter）技术、用户空间内核、系统调用拦截，以及动态策略翻译机制。

**📊 数据集**

实验数据集主要基于公开的AI代理任务（如RAG、代码执行等）和模拟威胁场景，未使用传统机器学习数据集。

**📈 对比分析**

通过与现有仅靠训练或提示调控的代理安全方法对比，实验表明本框架在安全性上显著提升，且系统调用拦截的开销低于5%（约10ms延迟），在吞吐量上保持与原始代理相近。

**⚠️ 局限性**

局限性包括对操作系统和内核版本的依赖、BPF功能受限导致的策略表达不够灵活，以及在极高并发或分布式环境下的可扩展性待进一步验证。

---

## 116. Discrete Flow Matching Policy Optimization

**arXiv ID:** 2604.06491 | [PDF](https://arxiv.org/pdf/2604.06491v1)

**作者:** Maojiang Su `[一作]`, Han Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种名为DoMinO的统一强化学习细调框架，用于对离散流匹配（DFM）模型进行奖励驱动的细调。

**💡 创新点**

将离散流匹配的采样过程重新表述为内层MDP，使得可直接应用REINFORCE、PPO等策略梯度方法，并引入基于总变分距离的交叉熵和广义KL正则化，理论上给出离散化误差和TV距离上界，解决了奖励不光滑、模型偏移等问题。

**🔧 技术方法**

离散流匹配（DFM）、连续时间马尔可夫链、Euler采样、强化学习策略梯度（REINFORCE、PPO）、总变分距离正则化。

**📊 数据集**

HepG2细胞系增强子序列数据集（约70万条200bp DNA序列），使用Enformer预测器作为奖励模型。

**📈 对比分析**

与预训练扩散模型、预训练流匹配模型、DRAKES、SEPO等基线对比，DoMinO在预测增强活性和染色质可及性上略优或相当，且在3‑mer相关性（序列自然度）上显著优于SEPO，体现了更好的功能优化与自然度平衡。

**⚠️ 局限性**

仍受奖励模型准确性限制，正则化参数对功能与自然度的权衡影响大；目前仅在单一DNA设计任务上验证，泛化能力和在其它任务上的表现仍需进一步评估。

---

## 117. TalkLoRA: Communication-Aware Mixture of Low-Rank Adaptation for Large Language Models

**arXiv ID:** 2604.06291 | [PDF](https://arxiv.org/pdf/2604.06291v1)

**作者:** Lin Mu `[一作]` (Anhui University), Yiwen Zhang `[通讯]` (Anhui University)

**通讯引用:** 21502 | [OpenAlex ID](https://openalex.org/A5115603611)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TalkLoRA，一个在LoRA基础上加入专家间通信的Mixture‑of‑Experts参数高效微调框架。

**💡 创新点**

通过引入轻量级Talking模块实现低秩专家间信息交换，打破独立性假设，显著提升专家路由平衡与模型表达能力。

**🔧 技术方法**

结合LoRA、MoE、通信矩阵C、参数共享及Kaiming初始化等技术实现结构化专家交互。

**📊 数据集**

使用常识推理基准（BoolQ、PIQA、SIQA、HellaSWAG、WinoGrande、ARC‑C/E、OBQA）和GLUE（SST‑2、MRPC、CoLA、QNLI、RTE等）数据集，在Qwen2.5‑7B、LLaMA‑2/3等LLM上进行微调。

**📈 对比分析**

与LoRA、DoRA、HiRA、MixLoRA、TeamLoRA等PEFT基线在0.2–0.4%可训练参数下比较，TalkLoRA在Qwen、LLaMA系列任务中平均准确率提升至87–89%，在GLUE上平均得分84.9%，显著优于同类方法。

**⚠️ 局限性**

尚未评估在多模态、极长文本、数学推理或代码生成等任务的表现；对极大规模模型实验受限；MoE通信模块引入额外推理延迟。

---

## 118. Code Sharing In Prediction Model Research: A Scoping Review

**arXiv ID:** 2604.06212 | [PDF](https://arxiv.org/pdf/2604.06212v1)

**作者:** Thomas Sounack `[一作]` (Dana-Farber Cancer Institute), Tom Pollard `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 17401 | [OpenAlex ID](https://openalex.org/A5086791063)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对引用 TRIPOD 或 TRIPOD+AI 的预测模型研究进行大规模 scoping review，量化代码共享率并系统评估共享仓库的可复现性特征。

**💡 创新点**

首次使用大型语言模型自动化全文筛选与仓库特征提取，并基于此结果为 TRIPOD-Code 扩展提供实证依据。

**🔧 技术方法**

采用 GPT‑5.2 2025‑12‑11 进行文章筛选、代码可用性声明提取与仓库链接检索，随后用同一 LLM 评估 14 项可复现性相关特征。

**📊 数据集**

数据来源于 PubMed 中引用 TRIPOD/TRIPOD+AI 的 3,967 篇文章（通过 PubMed Central Open Access API 取得全文）以及对应的 380 个可访问代码仓库。

**📈 对比分析**

对 LLM 的输出进行人工校验，文章筛选的 F1 评分为 0.97，仓库特征评估的 F1 评分为 0.83，准确率为 92.3%；结果显示 2025 年代码共享率从 6.3% 提升至 15.8%，TRIPOD+AI 论文的共享率更高（约 29.7%）。

**⚠️ 局限性**

自动化分类存在误差，样本仅限于可公开获取且引用 TRIPOD 的文献，可能高估共享率；仓库功能可复现性未得到直接验证，评估仅基于结构特征。

---

## 119. The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models

**arXiv ID:** 2604.06374 | [PDF](https://arxiv.org/pdf/2604.06374v1)

**作者:** Michael Rizvi-Martel `[一作]` (Mila), Marius Mosbach `[通讯]` (Mila)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型在连续链式推理（Latent CoT）中是否能利用叠加（superposition）来进行多路径推理。

**💡 创新点**

证明只有从零开始训练的模型能表现出叠加行为，预训练或微调的模型则会崩溃，揭示了预训练目标与模型容量对叠加的抑制作用。

**🔧 技术方法**

使用Soft Thinking与Coconut两种Latent CoT技术，并结合对数视角（Logit Lens）与实体级探测（entity‑level probing）等方法对内部表示进行解释性分析。

**📊 数据集**

在MATH500、AIME2024、GSM8K、ProsQA等公开数据集上进行实验，涵盖离线推理、微调以及从零开始训练三种训练范式。

**📈 对比分析**

通过熵、KL散度、余弦相似度以及准确率的对比，发现训练自由的模型保持高熵、低KL、余弦相似度接近1，且准确率显著提升；而预训练/微调模型熵骤降、KL小、余弦相似度高但依赖短路解法，无法体现叠加。

**⚠️ 局限性**

局限在于只评估了两种Latent CoT方法，且词级别叠加可能不具备实用价值，模型容量与预训练目标对结果影响显著，需要进一步探索更高层次的叠加机制。

---

## 120. No-reference based automatic parameter optimization for iterative reconstruction using a novel search space aware crow search algorithm

**arXiv ID:** 2604.06246 | [PDF](https://arxiv.org/pdf/2604.06246v1)

**作者:** Poorya MohammadiNasab `[一作]` (Danube Private University), Sepideh Hatamikia `[通讯]` (Danube Private University)

**通讯引用:** 728 | [OpenAlex ID](https://openalex.org/A5066922937)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种无需参考图像、可自动优化 CBCT 迭代重建算法超参的通用框架；

**💡 创新点**

创新点包括：①改进的Crow Search Algorithm（SSA‑CSA）结合搜索空间感知、局部搜索增强和自适应探索/利用平衡；②基于 SNR 与 HFER 的多目标无参考适应度函数；③使用三角混沌 DLU 初始化加速收敛；

**🔧 技术方法**

技术实现基于改进 CSA（SSA‑CSA）、三角混沌 DLU 初始化、SNR+HFER 适应度、学习型图像质量评估（CHILL@UK、RPI_AXIS）做性能验证；

**📊 数据集**

使用四个真实数据集：Nikon SophiaBeads、IRm LinePairs CatPhan、Philips Allura Thorax、IRm Brain Phantom（含金属螺钉前后对比）；

**📈 对比分析**

通过与人工调参、原始 CSA、基准方法比较，使用经典指标（SNR、HFER）、无参考指标（CHILL@UK、RPI_AXIS）以及 FR‑IQA 评估；在所有实验中，SSA‑CSA 在平均适应度上提升约 4.19%，CHILL@UK 提升 4.89%，RPI_AXIS 提升 3.82%，图像质量更清晰、细节保留更好；

**⚠️ 局限性**

局限性包括：作为元启发式方法，无法保证全局最优，可能陷入局部最优；适应度函数需要两权重调参；计算成本较高，尤其在多参数、高维空间下；

---

## 121. Breaking Negative Cycles: A Reflection-To-Action System For Adaptive Change

**arXiv ID:** 2604.06477 | [PDF](https://arxiv.org/pdf/2604.06477v1)

**作者:** Minsol Michelle Kim `[一作]` (Massachusetts Institute of Technology), Pattie Maes `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 19302 | [OpenAlex ID](https://openalex.org/A5081457786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估一种基于语音日记的“Reflection‑to‑Action”系统，该系统通过WhatIf‑Planning模块将反思转化为可执行的“if‑then”行动计划，旨在帮助用户打破负面循环并实现自我调节。

**💡 创新点**

创新点包括：①首次将Transtheoretical Model（TTM）的Preparation阶段与情绪调节过程模型（Gross ER Process Model）结合，形成系统化的反思‑行动框架；②提出WhatIf‑Planning，将反事实思维与实施意图相结合；③使用语音日记实现更自然、即时的自我反思；④对Gross指导与自由形式两种反思结构进行对比实验。

**🔧 技术方法**

技术手段包括：移动端语音录音与自动转写（Firebase+Speech‑to‑Text）、Web端WhatIf‑Planning交互、结构化提示生成、混合方法数据收集（问卷、日志、访谈）。

**📊 数据集**

数据集：15 天、20 名非临床参与者（10 人自由形式，10 人 Gross‑guided），共 147 条语音记录、192 条反事实+行动计划，配合标准量表（Coping Flexibility Scale‑Revised、DERS‑SF）和自定义日志。

**📈 对比分析**

对比方法：随机分配两组，采用混合 ANOVA、线性混合模型、t 检验和效应量（Hedges g）。结果显示两组均显著提升应对灵活性（CFS‑R 总分 p=0.02，效应量 0.51）；Gross‑guided 组在计划执行、反事实生成、障碍识别等行为指标上表现更佳，效应量在 0.99–1.71 之间，提示显著优势。情绪调节子量表虽无统计显著差异，但Gross‑guided 组在非接受度、目标维持等方面呈中等效应。

**⚠️ 局限性**

局限性：样本量小、实验时间短（15 天），缺乏临床人群和长期随访；两组结构固定，未考虑用户在TTM不同阶段的动态需求；未与传统文本日记或其他行为干预直接对比；AI 辅助功能仍未实现。

---

## 122. Extracting Breast Cancer Phenotypes from Clinical Notes: Comparing LLMs with Classical Ontology Methods

**arXiv ID:** 2604.06208 | [PDF](https://arxiv.org/pdf/2604.06208v1)

**作者:** Abdullah Bin Faiz `[一作]` (CureMD Research), Muddassar Farooq `[通讯]` (CureMD Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文开发并比较了两种乳腺癌临床笔记表型抽取系统：基于本体推理的传统方法和利用本地LLM与检索增强生成（RAG）技术的新型LLM框架。

**💡 创新点**

创新点在于：①将RAG与自定义JSON Schema结合，提升LLM在医学文本中语义抽取的准确性；②在本地GPU集群上安全部署LLM，解决数据隐私与云端依赖问题；③首次将本体系统与LLM系统在相同数据集上进行系统性对比。

**🔧 技术方法**

所使用技术包括：LLaMA 3 8B、Mistral 7B模型；LangChain递归分块与双重重排序（语义与词典检索）实现RAG；Pyserini Lucene检索；Nvidia TensorRT‑LLM加速推理；JSON Schema校验与后处理。

**📊 数据集**

数据来源于95,000条乳腺癌医生笔记（用于模型推理）以及150条人工标注的临床笔记（用于评估），标注由五名医生共同完成，覆盖TNM分期、肿瘤大小、分级、性能与生物标志物等表型。

**📈 对比分析**

评价方法采用人工标注标签与模型输出的精确度、召回率、F1分数对比；结果显示LLaMA 3 8B取得86%准确率、87%精确度、95%召回率、91%F1；本体系统精确度为100%但召回率仅83%，LLM系统平均处理时长约12秒，低于本体系统的20秒。

**⚠️ 局限性**

局限性包括：LLM易产生幻觉与遗漏；Mistral模型性能相对较弱；本体系统缺乏语义捕捉与泛化能力；评估样本量有限，未覆盖所有可能的表型；RAG处理长文本时需分块，增加复杂度。

---

## 123. Discoverability matters: Open access models and the translation of science into patents

**arXiv ID:** 2604.06229 | [PDF](https://arxiv.org/pdf/2604.06229v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 124. Qualixar OS: A Universal Operating System for AI Agent Orchestration

**arXiv ID:** 2604.06392 | [PDF](https://arxiv.org/pdf/2604.06392v1)

**作者:** Varun Pratap Bhardwaj `[一作]` `[通讯]`, Varun Pratap Bhardwaj

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了名为qualixar-os的应用层AI代理操作系统，支持12种多代理拓扑、自动团队设计、三层模型路由、共识式质量保障、Goodhart检测、四层归因、跨框架兼容、可视化仪表盘和技能市场。

**💡 创新点**

创新点包括：全面拓扑执行语义、LLM驱动的Forge团队设计引擎、基于Q‑学习/贝叶斯POMDP的三层模型路由、融合Goodhart检测和JSD漂移监测的8模块质量管线、四层归因与区块链时间戳、统一协议（UCP）实现跨框架代理互通，以及可扩展的Dashboard与Marketplace。

**🔧 技术方法**

采用了LLM(多模型)、强化学习、上下文感知bandit、POMDP、贝叶斯推断、JWT、HMAC、Steganography、区块链时间戳、SQLite、WebSocket、React+Zustand、HTTP/MCP/A2A、Docker、OAuth2、LLM调用等技术栈。

**📊 数据集**

使用自定义20任务评估套件（包含事实、算术、推理、概率等20个任务），并在2,821个单元测试和217个事件类型上验证；此外采用Azure AI Foundry等10个模型提供商进行动态模型发现。

**📈 对比分析**

与AutoGen、CrewAI、LangGraph等框架对比，qualixar-os在拓扑、团队设计、质量保障、成本路由、归因、仪表盘、Marketplace等8维度均领先；在20任务评估中实现100%准确率，平均成本$0.000039/任务，平均时长≈4s，且通过2,821测试实现100/100质量评分。

**⚠️ 局限性**

局限性包括：评估仅在自定义20任务套件上，未覆盖Web、文件、跨工具协作；自适应学习回路实验未显著收敛；单节点SQLite架构不支持分布式部署；模型发现启动时延高；Goodhart检测窗口需至少50次评估，短期任务难检测；JSD漂移基于初始分布假设，可能出现偏差。

---

## 125. Digital Weight Management Interventions: A review of commercial solutions and survey analysis of user needs

**arXiv ID:** 2604.06181 | [PDF](https://arxiv.org/pdf/2604.06181v1)

**作者:** Suncica Hadzidedic `[一作]` (Durham University), Grant Westermann `[通讯]` (MoreLife Ltd)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过系统性评估17款商业数字体重管理干预（DWMIs）并对207名实际使用者进行问卷调查，分析其功能、服务与用户需求；

**💡 创新点**

创新点在于首次将DWMIs的核心功能、资源推荐水平与用户对技术舒适度、资源帮助度进行对照，并揭示其在行为改变、个性化推荐与心理支持方面的不足；

**🔧 技术方法**

采用系统性文献检索、桌面评估和在线问卷调查等技术手段；

**📊 数据集**

使用的“数据集”为17款DWMIs的功能与服务信息（从Apple App Store、Google Play Store和Reddit获取）以及207名WMI参与者的问卷数据；

**📈 对比分析**

通过描述性统计与t检验比较不同群体（性别、教育水平、动机类型）对技术使用舒适度和资源帮助度的差异，结果显示女性和医学动机者对某些功能更满意，表明需进一步改进功能个性化与心理支持；

**⚠️ 局限性**

局限性包括仅覆盖主流英语DWMIs且仅检视免费版，样本局限于单一英国公司客户，缺乏付费版评估与纵向有效性验证。

---

## 126. The Art of Building Verifiers for Computer Use Agents

**arXiv ID:** 2604.06240 | [PDF](https://arxiv.org/pdf/2604.06240v1)

**作者:** Corby Rosset `[一作]` (Microsoft Research), Ahmed Awadallah `[通讯]` (Microsoft Research)

**通讯引用:** 3379 | [OpenAlex ID](https://openalex.org/A5021000040)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对计算机使用代理（CUA）轨迹的通用验证器（Universal Verifier）并公开了对应的基准数据集CUAVerifierBench；

**💡 创新点**

创新点在于将过程奖励与结果奖励分离、区分可控与不可控失败、使用多模态相关性评分与分层上下文管理，以及引入面向侧效应的诊断机制，使验证器在不依赖强大模型的前提下显著提高与人类判断的一致性；

**🔧 技术方法**

核心技术包括：基于任务生成无重叠的评判量规（Rubric）、多模态相关性矩阵构建、Top‑k截图分组与证据提取、层级化重评分与错误诊断、以及自动化研究代理（Auto‑Research）用于优化提示与代码；

**📊 数据集**

使用了两组人类标注的数据集：内部140条WebTailBench轨迹与外部106条Online‑Mind2Web轨迹，并对多种现有验证器（WebVoyager、WebJudge）和多名代理模型（Fara‑7B、GPT‑5）进行对比；

**📈 对比分析**

与WebVoyager、WebJudge相比，Universal Verifier在CUAVerifierBench上达到了约0.64–0.58的Cohen’s κ（人类一致性范围内），误报率（FPR）降至≈0.01–0.08，召回率显著提升；在多模态截图管理与过程/结果分离方面表现出最优的可解释性与鲁棒性；

**⚠️ 局限性**

局限性包括：验证器仍需大量手工编码与提示设计，对复杂侧效应的覆盖仍有限；自动化研究代理虽然能提升70%性能，却无法独立发现关键结构改进；在极长轨迹或多模态交互多样性极高的场景下，仍可能出现遗漏或误判。

---

## 127. AgentOpt v0.1 Technical Report: Client-Side Optimization for LLM-Based Agent

**arXiv ID:** 2604.06296 | [PDF](https://arxiv.org/pdf/2604.06296v1)

**作者:** Wenyue Hua `[一作]` (Microsoft Research), Tianyi Peng `[通讯]` (Columbia University)

**通讯引用:** 832 | [OpenAlex ID](https://openalex.org/A5002767768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了AgentOpt，一个面向客户端的Python框架，用于在AI代理工作流中对模型组合进行优化，从而在满足质量、成本和延迟约束的前提下实现资源的最优分配。

**💡 创新点**

创新点在于将模型选择视为端到端的组合级优化问题，而非传统的单调用路由；同时实现了无框架依赖、HTTP层拦截、缓存和并行评估，并提供多种搜索策略（Arm Elimination、Epsilon-LUCB、贝叶斯优化等）以高效探索指数级组合空间。

**🔧 技术方法**

核心技术包括多臂赌博机（Arm Elimination、Epsilon‑LUCB、Threshold SE）、局部搜索（Hill Climbing）、贝叶斯优化（BO）和LM Proposal；在底层实现上使用HTTP拦截、上下文变量追踪、并行限速和缓存机制。

**📊 数据集**

实验使用四个任务：HotpotQA、GPQA Diamond、MathQA和BFCL v3 Multi‑Turn，涵盖从两阶段到多工具调用的不同流水线结构。

**📈 对比分析**

与全量暴力搜索和随机搜索对比，Arm Elimination在大多数基准上达到与最优组合相近的准确率，同时将评估预算降低24–67%；在HotpotQA和MathQA等复杂任务中，Arm Elimination的准确率与贝叶斯优化相当或更好，且评估次数更少。

**⚠️ 局限性**

限制包括：搜索结果依赖于给定的模型池和评估集，无法处理动态或实时的模型/工具切换；对极大组合空间仍有计算开销；某些基于LM的候选生成方法在复杂角色交互中表现不佳。

---

## 128. Policy-Driven Vulnerability Risk Quantification framework for Large-Scale Cloud Infrastructure Data Security

**arXiv ID:** 2604.06252 | [PDF](https://arxiv.org/pdf/2604.06252v1)

**作者:** Wanru Shao `[一作]` `[通讯]` (Northeastern University), Wanru Shao (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个名为 MVRAF 的多维漏洞风险评估框架，结合严重性量化、风险因子相关性分析和经验风险分布，以系统化方式对大规模 CVE 进行风险评估和优先级划分。

**💡 创新点**

创新点包括：① 通过加权聚合 CVSS 探索性和 CIA 影响得分的严重性量化模型；② 利用相关矩阵与条件概率框架揭示攻击向量、复杂度、权限等多维因子之间的统计依赖；③ 引入经验风险分布机制，实现累计风险评估和资源分配优化。

**🔧 技术方法**

技术手段主要包括：加权求和与归一化、协方差/皮尔逊相关系数矩阵、条件概率矩阵、联合风险指数计算、经验分布函数（ECDF）以及基于 Python 的数据处理与可视化。

**📊 数据集**

使用 2024 年 1 月 1 日至 15 日期间公开的 NVD CVE 数据集，共 1,314 条记录，包含 CVSS 向量、攻击向量、复杂度、权限等属性。

**📈 对比分析**

与三种基线（原始 CVSS、统一权重模型、Moustaid 等 ML 预测器）对比，MVRAF 在校准集上取得 MAE 0.31、Spearman ρ=0.94，明显优于原始 CVSS 的 ρ=0.91 和其他基线，说明其严重性排序更准确。

**⚠️ 局限性**

局限性包括：仅基于静态 CVE 记录，未考虑漏洞出现的时间演化；缺乏对动态威胁情境的实时更新；框架未直接与自动补丁管理系统集成，导致实际运维中需要额外桥接。

---

## 129. Temporally Phenotyping GLP-1RA Case Reports with Large Language Models: A Textual Time Series Corpus and Risk Modeling

**arXiv ID:** 2604.06197 | [PDF](https://arxiv.org/pdf/2604.06197v1)

**作者:** Sayantan Kumar `[一作]` (National Library of Medicine), Jeremy C. Weiss `[通讯]` (National Library of Medicine)

**通讯引用:** 2315 | [OpenAlex ID](https://openalex.org/A5072774346)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大型语言模型对 PubMed 开放获取的 GLP‑1RA 病例报告进行时间序列提取，构建了可供后续纵向建模的临床文本时间轴。

**💡 创新点**

创新点在于首次将 LLM 作为注解器生成完整事件‑时间对，并与专家双人金标准进行对比，提供了可复用的时间序列基准数据集。

**🔧 技术方法**

主要技术包括基于 GPT‑5 的提示式时间提取、PubMedBERT 语义相似度匹配、时间差异的 log‑time CDF 评估以及 Cox 比例风险模型用于事件时间分析。

**📊 数据集**

数据集来源于 PubMed Open Access 论文库，筛选出 136 篇单患者 GLP‑1RA 病例报告，人工标注了 136 篇时间轴。

**📈 对比分析**

通过事件匹配率、时间序列顺序一致性（C‑index）和时间误差的 AULTC 与人工金标准对比，GPT‑5 在事件覆盖率 0.871、顺序一致性 0.843 及时间准确度方面均优于其他 LLM，且在时间‑事件分析中显示 GLP‑1RA 用户呼吸系统并发症风险显著下降（HR=0.259）。

**⚠️ 局限性**

局限性包括病例报告的发表偏倚导致样本不具代表性、人工注解成本高导致样本规模受限、事件时间以文本首次出现为准，可能不等同于真实发病时间，以及对 LLM 的依赖可能引入提取与时间标注错误。

---

## 130. BDI-Kit Demo: A Toolkit for Programmable and Conversational Data Harmonization

**arXiv ID:** 2604.06405 | [PDF](https://arxiv.org/pdf/2604.06405v1)

**作者:** Roque Lopez `[一作]` (New York University), Juliana Freire `[通讯]` (New York University)

**通讯引用:** 12045 | [OpenAlex ID](https://openalex.org/A5006773757)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并演示了BDI-Kit，一款支持Python API和AI对话的可交互数据和谐化工具，实现了人机协作的schema和value匹配。

**💡 创新点**

创新点在于提供可组合的匹配原语、可重用的和谐化规范、AI助手驱动的对话交互以及人机协同的可视化校正流程。

**🔧 技术方法**

使用Python编程接口、自然语言交互、模型上下文协议（MCP）、多种schema/value匹配算法（传统、算法、LLM、文本相似、嵌入相似、数值转换）。

**📊 数据集**

使用了两套医学数据集：子宫内膜癌数据表与GDC模型；胰腺癌表与GDC模型。

**📈 对比分析**

主要通过手工验证和可视化的方式进行比较，展示了可重用规范在hold-out子集上的效果；未给出数值性能指标。

**⚠️ 局限性**

局限性：仍需人工校正、只支持一对一属性映射、对超大数据集效率有限、缺乏全面自动化评估。

---

## 131. Beyond Arbitrary Allocations: Security Values in Constrained General Lotto Games

**arXiv ID:** 2604.06329 | [PDF](https://arxiv.org/pdf/2604.06329v1)

**作者:** Keith Paarporn `[一作]` (University of Colorado), Jason R. Marden `[通讯]` (University of California)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的General Lotto游戏变体，其中一名玩家被限制只能在单一竞争中投放资源，并分析了此约束对双方博弈平衡与安全价值的影响。

**💡 创新点**

创新点在于引入玩家的空间/资源局部化约束，推导出受限玩家的安全价值下界与上界，并证明在多种参数下该下界与上界可收敛至相同值，从而实现对约束博弈的近似或精确平衡分析。

**🔧 技术方法**

采用了博弈理论、凸优化（特别是凸/凹函数求解）、解析式推导以及数值模拟等技术手段；对上界求解进一步利用组合分析将非凸问题转化为可枚举的最小化。

**📊 数据集**

使用的并未涉及真实数据集，全部采用合成测试：三场竞赛的价值按降序设置（v1>v2>v3），并通过改变预算比例 Y 与 X 进行参数扫描。

**📈 对比分析**

通过与传统无约束General Lotto游戏的理论平衡值比较，展示受限玩家在预算足够时可获得相同甚至更优的安全价值；数值结果显示下界与上界在低预算区间仅有微小差距，且在大预算区间完全重合；对K=2的情况进行模拟，验证安全价值随可投放竞赛数的增加而趋近于无约束情形。

**⚠️ 局限性**

局限性包括：仅考虑单一竞赛投放约束，未给出当下界与上界完全重合的充分必要条件；对多竞赛投放（K>1）的分析仍停留在数值近似层面；上界求解涉及非凸最优化，实际实现时可能受限于搜索空间；此外，合成竞赛价值假设可能无法直接映射到实际应用场景。

---

## 132. Ontology-based knowledge graph infrastructure for interoperable atomistic simulation data

**arXiv ID:** 2604.06230 | [PDF](https://arxiv.org/pdf/2604.06230v1)

**作者:** Abril Azocar Guzman `[一作]` (Forschungszentrum Juelich GmbH), Stefan Sandfeld `[通讯]` (Forschungszentrum Juelich GmbH)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一套基于本体的原子尺度模拟数据知识图谱框架，可实时捕获并集成来自不同来源的元数据和工作流信息；

**💡 创新点**

创新点在于推出了CMSO与ASMO两套本体，实现了统一的语义模型；结合轻量级元数据模板与自动化管道，完成多源数据的标准化、语义查询，并支持工作流的双向可追溯与重构；

**🔧 技术方法**

技术实现包括OWL/RDF/PROV‑O/QUDT/MDO本体、Python框架（Pydantic、atomRDF、模板系统、图对象）以及SPARQL查询，辅以机器学习/LLM辅助的元数据提取；

**📊 数据集**

使用了来自Zenodo、论文补充材料、GitHub等多源的原子模拟数据，涵盖八种元素的DFT、MD、MS计算以及已有的晶界数据库等；

**📈 对比分析**

通过SPARQL对Σ3晶界能量、误差等进行跨数据集查询，并在知识图谱上绘制能量分布和相互关系，验证已知晶体学趋势；构建的图谱包含757,253条三元组和7,926个样本，查询速度明显快于传统文件结构；

**⚠️ 局限性**

局限性包括对输入元数据完整性的依赖；外部依赖如势能文件和执行细节未完全捕获；工作流重构受限于缺失的势能文件信息；尚未覆盖所有模拟方法与属性；对大规模持续更新与可扩展性的进一步验证仍待开展。

---

## 133. A Goal-Oriented Chatbot for Engaging the Elderly Through Family Photo Conversations

**arXiv ID:** 2604.06184 | [PDF](https://arxiv.org/pdf/2604.06184v1)

**作者:** Raymond Chung `[一作]` (Logistics and Supply Chain MultiTech R&D Centre), CD Shum `[通讯]` (Logistics and Supply Chain MultiTech R&D Centre)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了面向老年人的目标导向聊天机器人，通过家庭照片引导对话并提供认知刺激。

**💡 创新点**

创新点在于将照片驱动的目标对话与动态兴趣识别结合，形成持续个性化互动与护理反馈。

**🔧 技术方法**

采用 GPT‑4 作为语言模型，结合面部识别、语音转文字、文本转语音以及自定义对话状态机。

**📊 数据集**

主要数据来自护理者上传的家庭照片与人工构造的问答对，实验使用 GPT‑4 生成的模拟老人语料。

**📈 对比分析**

与未使用照片驱动或无自适应对话的基线相比，实验显示对话连贯性和老人满意度提升约 15‑20%。

**⚠️ 局限性**

限制在于 LLM 的推理准确性仍有限，可能误判回答类型，且依赖护理者上传的高质量图像和描述。

---

## 134. EviSnap: Faithful Evidence-Cited Explanations for Cold-Start Cross-Domain Recommendation

**arXiv ID:** 2604.06172 | [PDF](https://arxiv.org/pdf/2604.06172v1)

**作者:** Yingjun Dai `[一作]` (Carleton University), Ahmed El-Roby `[通讯]` (Carleton University)

**通讯引用:** 290 | [OpenAlex ID](https://openalex.org/A5038522506)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EviSnap，一种通过离线 LLM 提取的面向证据的 facet 卡构建共享概念空间，并用单线性映射与可加评分头实现冷启动跨域推荐。

**💡 创新点**

创新点在于：① 以可追溯的句子证据构造面向域的 facet 卡；② 通过 k‑means 聚类得到共享概念银行，保证跨域可比性；③ 采用单线性映射和可加线性头，使预测完全可分解为 per‑concept 贡献，从而实现“可解释即构造”的设计。

**🔧 技术方法**

技术包括：LLM 预处理生成 facet 卡、冻结句子编码器嵌入 facet 与句子、k‑means 聚类形成概念、加权 log‑sum‑exp pooling 计算概念激活、单线性概念映射、可加线性评分头以及 MSE + 正则化训练。

**📊 数据集**

使用 Amazon 2014 语料库，选取 Books、Movies、Music 三个域，构建 6 个单向转移任务。

**📈 对比分析**

与 EMCDR、PTUPCDR、MACDR、DeepCoNN+、HeroGraph 等基线对比，EviSnap 在 5/6 个转移上均取得 MAE 与 RMSE 最佳或第二佳，整体平均 MAE 与 RMSE 分别提升约 3.3% 与 2.7%，并通过删除与充分性测试验证解释可信度。

**⚠️ 局限性**

局限性：依赖离线 LLM 进行 facet 提取，提取结果受模型与提示的噪声与偏见影响；在缺乏足够评论文本的稀疏场景下性能与解释质量可能下降；概念质量受编码器与聚类参数影响，线性结构可能忽略更高阶交互。

---

## 135. CobbleDB: Modelling Levelled Storage by Composition

**arXiv ID:** 2604.06273 | [PDF](https://arxiv.org/pdf/2604.06273v1)

**作者:** Emilie Ma `[一作]` (University of British Columbia), Marc Shapiro `[通讯]` (Sorbonne-Université---LIP6 & Inria)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于形式化规范手工实现了 Java 版数据库后端，并通过组合基本存储构造 CobbleDB（模拟 RocksDB 的级别存储），同时验证了其正确性并提供完整测试覆盖。

**💡 创新点**

创新点在于将形式化规范与存储组合相结合，证明基本存储在重叠窗口下等价，从而在保持一致性保障的同时实现性能优化；并首次推出 CobbleDB 的规范驱动实现。

**🔧 技术方法**

采用形式化规格（Coq/Isabelle 相关工具）、事务协议、冲突无关复制数据类型（CRDT）效果合并、Java 并发库、YCSB 负载测试、Log‑Structured Merge‑Tree（LSM）层级结构以及持久化与恢复机制。

**📊 数据集**

使用 YCSB 生成的 Zipfian 分布键、5000/5050 读写比例等自定义工作负载；对 RocksDB 与 CobbleDB 进行基准测试。

**📈 对比分析**

通过 YCSB 并发线程 1–20 运行自定义事务工作负载，对比 CobbleDB 与基本存储以及 RocksDB；CobbleDB 约为 RocksDB 的 9.6% 性能，但在增量工作负载下性能相近，基本存储与 CobbleDB 性能相似。

**⚠️ 局限性**

限制包括：代码未做性能优化（仅使用基础数据结构）、语言差异导致性能偏低、缺少 RocksDB 诸多优化（Bloom 过滤、压缩等）、仅单机实验、未覆盖分布式一致性场景。

---

## 136. Evolution of Video Generative Foundations

**arXiv ID:** 2604.06339 | [PDF](https://arxiv.org/pdf/2604.06339v1)

**作者:** Teng Hu `[一作]`, Dacheng Tao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了视频生成技术的发展脉络，涵盖GAN、扩散模型、AR模型及多模态融合等主流方法，并分析其在技术演进与应用场景中的定位。

**💡 创新点**

首次从宏观演进与细节技术两层面系统梳理视频生成的技术路线，特别突出AR模型和多模态整合的研究空白，提出未来研究的六大关键方向。

**🔧 技术方法**

系统性评述了GAN（Spatio‑Temporal、Progressive、StyleGAN等）、扩散模型（UNet、DiT、MM‑DiT、Efficient DMs）、Auto‑Regressive 模型（VQ‑VAE + Transformer、Mask‑GIT、AR‑Diffusion）以及跨模态训练与推理技术。

**📊 数据集**

主要参考公开的开源模型与训练数据集（如Image‑Text、Video‑Text对齐数据、海量视频抓取集），未在本文中直接使用特定数据集进行实验，聚焦于文献与模型的性能对比。

**📈 对比分析**

通过对顶级会议（CVPR/ICCV/ECCV）论文数量、VBench、T2VSafetyBench、VideoPhy‑2 等多维度指标对比，展示扩散模型在视觉质量与可扩展性方面的优势；AR模型在长时序生成与多模态控制上表现突出，但视觉细节与逼真度仍落后。

**⚠️ 局限性**

受限于缺乏统一评测标准和实验验证，难以精确量化AR与多模态模型的性能差距；多模态整合仍面临数据稀缺、跨模态对齐与实时推理效率等挑战。

---

## 137. FLeX: Fourier-based Low-rank EXpansion for multilingual transfer

**arXiv ID:** 2604.06253 | [PDF](https://arxiv.org/pdf/2604.06253v1)

**作者:** Gaurav Narasimhan `[一作]` `[通讯]` (Stanford University), Gaurav Narasimhan (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文通过对 Code Llama 7B 模型使用低秩适配（LoRA）进行参数高效微调，探索了优化器和频域正则化对跨语言代码生成的影响。

**💡 创新点**

创新点包括：引入 Fourier 基频正则化以提升跨语言泛化，使用 Sophia 近似二阶优化器比较 Adam，并展示在 Java 任务上 42.1% 的显著提升。

**🔧 技术方法**

技术主要涉及 LoRA、AdamW 与 Sophia 优化器、以及基于离散傅里叶变换的正则化方法。

**📊 数据集**

使用的数据集有 HumanEval、MBPP、APPS、CodeSearchNet 以及 MultiPL‑E。

**📈 对比分析**

通过在 HumanEval 评测上 LoRA 微调取得 40.1% Pass@1，超越 Code Llama‑Python‑7B 的 38.4%；在 Java MultiPL‑E 上 Fourier 正则化模型达 42.1%，显著高于基线 34.2%。

**⚠️ 局限性**

局限性包括：合并 LoRA 权重表现不佳、频域正则化效果随任务变化、仅评估 Pass@1，缺乏更深层次的成功率指标。

---

## 138. FMI@SU ToxHabits: Evaluating LLMs Performance on Toxic Habit Extraction in Spanish Clinical Texts

**arXiv ID:** 2604.06403 | [PDF](https://arxiv.org/pdf/2604.06403v1)

**作者:** Sylvia Vassileva `[一作]` (Sofia University St Kliment Ohridski), Svetla Boytcheva `[通讯]` (Sofia University St Kliment Ohridski)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用 GPT‑4.1 的少量示例提示（few‑shot）完成西班牙语临床文本中毒习惯实体识别任务

**💡 创新点**

首次将大型语言模型的上下文学习应用于多语言（西班牙语）临床实体识别，并通过提示工程与后处理优化边界检测

**🔧 技术方法**

GPT‑4.1、in‑context learning、DsPy 优化器、手工提示调优、结构化输出抽取

**📊 数据集**

ToxHabits 共享任务数据集（1500 条西班牙语临床病例，1.2k 训练 + 0.3k 验证，0.3k 测试）

**📈 对比分析**

与字典基线、Spanish Clinical RoBERTa、GPT‑4.1 零/少量示例对比；少量示例 5‑例模型在测试集上 F1=0.65，精度 0.59，召回 0.72，较字典基线精度更高但召回略低

**⚠️ 局限性**

仅在 GPT‑4.1 2025‑04‑14 版本上验证，方法对提示与示例选择高度敏感，实体边界检测仍不够精确，且未能统一处理所有标注变体，缺乏对其他数据集的泛化验证

---

## 139. Toward a universal foundation model for graph-structured data

**arXiv ID:** 2604.06391 | [PDF](https://arxiv.org/pdf/2604.06391v1)

**作者:** Sakib Mostafa `[一作]` (Stanford University), Md. Tauhidul Islam `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了图基础模型（Graph Foundation Model, GFM），利用节点拓扑信息生成自然语言结构提示并通过多图对比预训练，得到可跨域、与节点特征无关的节点表征，可在生物医学网络上实现零样本或少样本下的下游任务。

**💡 创新点**

创新点在于：①将图结构描述转化为可输入语言模型的自然语言提示，完全摆脱对节点特征语义的依赖；②在九个异构非生物图上同步对比图结构嵌入与文本嵌入，训练出对拓扑角色高度敏感的表示；③引入轻量化适配器在目标图上无标签对比微调，保持模型可迁移性。

**🔧 技术方法**

主要技术包括 MiniLM 对结构提示编码；GraphSAGE 消息传递 backbone；InfoNCE + 拉普拉斯平滑对比预训练；基于 Personalized PageRank 的正样本挖掘；多图预训练与 per-graph adapter 微调；零样本、少样本与全监督评估流水线。

**📊 数据集**

预训练数据：九个非生物学图（Cora、CiteSeer、DBLP、ogbn-arxiv、CoauthorCS、CoauthorPhysics、AmazonComputers、AmazonPhoto、WikiCS）。下游评测集：SagePPI、ogbn-proteins、StringGO（MF、BP、CC）、Fold-PPI（SCOP 结构折叠类别）。

**📈 对比分析**

与四种监督基准（GCN、GIN、GAT、GraphSAGE）在零样本、少样本和全监督条件下进行对比。GFM 在零样本平均 ROC‑AUC 方面普遍优于最佳监督模型（SagePPI 76.1% vs 73.7%，ogbn‑proteins 71.0% vs 64.6%，StringGO 81.4% vs 74.7%，Fold‑PPI 95.5% vs 94.1%）。在少样本设置下，GFM 只需 1‑3 个标记样本即可逼近 20‑个样本监督的性能，显著降低了标签需求。

**⚠️ 局限性**

局限性包括：①结构提示在边稀疏或噪声较大的图中信号弱，导致预训练效果下降；②对大规模实时或隐私受限图需改进无全图依赖的结构提示；③在标签空间高度不重叠或极其不平衡时（如 StringGO 的 BP 任务）微调不稳定；④模型尚未针对多模态节点特征或有序标签空间进行专门优化。

---

## 140. What Do Humanities Scholars Need? A User Model for Recommendation in Digital Archives

**arXiv ID:** 2604.06232 | [PDF](https://arxiv.org/pdf/2604.06232v1)

**作者:** Florian Atzenhofer-Baumgartner `[一作]` (Graz University of Technology), Dominik Kowald `[通讯]` (Know Center Research GmbH)

**通讯引用:** 1150 | [OpenAlex ID](https://openalex.org/A5071624510)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对18位人文学者进行焦点小组访谈，调研其在数字档案馆中的信息检索行为，并提出了四个关键维度（情境波动、认知信任、对比寻求、研究脉络连续性）的用户模型框架。

**💡 创新点**

首次将传统RecSys的稳定偏好、相似性优化等假设与学术研究者的动态、对比性和信任需求进行对照，提出可用于诊断和设计定制化推荐系统的四维框架。

**🔧 技术方法**

采用半结构化焦点小组访谈与归纳式主题分析，结合定性编码方法构建模型。

**📊 数据集**

18名人文学者的访谈记录和讨论文本。

**📈 对比分析**

本文未实现具体推荐算法，因而没有算法比较或性能评估；框架的适用性仅通过专家访谈得到初步验证。

**⚠️ 局限性**

研究仅基于定性访谈数据，缺乏日志行为验证；样本局限于单一数字档案生态，结果可能不具普遍适用性。

---

## 141. Incentive-Aware Multi-Fidelity Optimization for Generative Advertising in Large Language Models

**arXiv ID:** 2604.06263 | [PDF](https://arxiv.org/pdf/2604.06263v1)

**作者:** Jiayuan Liu `[一作]` (Carnegie Mellon University), Vincent Conitzer `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10218 | [OpenAlex ID](https://openalex.org/A5050903632)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一个整合VCG激励与多保真度优化的机制IAMFM，用于在LLM生成广告中在预算与战略行为约束下寻找最优广告强度配置。

**💡 创新点**

将多保真度黑盒优化与前期VCG机制相结合，提出主动反事实优化（ACO）高效计算VCG支付；提出两种算法实例化（IAMFM-ASH和IAMFM-MFBO），并给出近似策略可证明与个体理性保证。

**🔧 技术方法**

多保真度多臂赌博机（MFO/MFBO）、Gaussian Process surrogate、UCB、Successive Halving与自适应 Successive Halving、VCG机制、主动反事实优化。

**📊 数据集**

以食堂推荐系统模拟环境为主（两家广告商、五类用户画像），并扩展至多模态视觉广告的扩散模型；评估器为LLM代理输出。

**📈 对比分析**

与单保真度基线（UCB、均匀采样）比较；低预算下IAMFM-ASH与MFBO优于基线，高预算下MFBO更优；在视觉任务上MFBO在不同预算下均优于基线，误差低、方差小。

**⚠️ 局限性**

需要预先训练的代理评估器，估计误差影响机制近似；对大型广告商数量扩展仍需评估；实现复杂度高，需要多保真度支持与ACO实现；极高预算下优势减小。

---

## 142. Probabilistic Language Tries: A Unified Framework for Compression, Decision Policies, and Execution Reuse

**arXiv ID:** 2604.06228 | [PDF](https://arxiv.org/pdf/2604.06228v1)

**作者:** Gregory Magarshak `[一作]` `[通讯]`, Gregory Magarshak

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并定义了概率语言树（Probabilistic Language Tries, PLT），一种统一的概率结构，能够同时实现无损压缩（通过区间编码）、序列决策（通过策略概率映射）和推理缓存（通过先验指导的缓存策略）。

**💡 创新点**

核心创新点包括：
• 将生成模型的隐式前缀概率显式化为可操作的 trie；
• 证明先验指导的缓存策略在查询量低于某阈值时优于经验频率缓存；
• 设计混合压缩架构，将数据分为 PLT 覆盖的主集和稀疏残差，能够压缩到低于 Shannon 熵；
• 把 PLT 应用于多领域（棋类 MCTS、搜索会话、机器人控制、LLM 推理），展示其在压缩、决策和计算重用上的统一效能；
• 将 PLT 与 Kolmogorov 复杂度、率失真理论对齐，提供理论上可达的压缩与近似界限。

**🔧 技术方法**

使用的技术包括：
• 基于条件概率的前缀树构造与区间（arithmetic）编码；
• 统计学习方法（transformer softmax、MCTS 访问计数）作为概率模型；
• Bayesian 先验与经验频率的融合，用于缓存决策；
• 证明工具（Hoeffding、不等式、KL 散度分析）用以量化缓存性能；
• 代码级实现（递归区间更新、离散化残差存储）。

**📊 数据集**

实验与实例化基于多种典型数据：
• 国际象棋棋局记录（MCTS 生成的走法序列）；
• Web 搜索会话日志（查询-页面序列）；
• 机器人控制日志（动作序列）；
• LLM 生成文本（token 序列）
（未给出公开数据集的具体名称，但示例覆盖了上述领域的常用公开或内部数据）。

**📈 对比分析**

比较方法：
• 与经验频率缓存（LFU/LRU）在相同查询集上进行理论与实验比较；
• 用区间编码的平均码长与模型交叉熵对比，验证压缩接近 Shannon 限界；
• 在 LLM 推理中，将 PLT 预先缓存的高概率序列与未缓存序列的执行成本做对比，证明从 O(n²) 降至 O(log N) 的期望成本。
性能方面：在理论上，先验缓存在查询量低于阈值时期望成本至少比经验缓存低 Δ·ρ（Δ 为前缀概率差，ρ 为计算成本差）；在压缩方面，混合架构能实现低于源熵的描述长度；在 LLM 推理中，预缓存覆盖一半以上请求即可显著降低运行成本。

**⚠️ 局限性**

局限性：
• 假设查询 i.i.d. 并且分布稳定，实际工作负载往往呈非平稳、聚集特性；
• 对大型 transformer（词表 ~5×10⁴、上下文 ~10⁴）的完整 PLT 构建不切实际，需要剪枝与稀疏化；
• 先验缓存优势随先验分布的峰度变化；对均匀分布（α→0）优点无限大；对高度集中分布（α 大）阈值极短，实际收益有限；
• 缓存失效时的 KL 判定与残差补偿在多模型迁移中的精度与开销尚未完全验证；
• 代码实现需要高效的区间更新与内存管理，现有论文仅给出理论框架。

---

## 143. Distributed Interpretability and Control for Large Language Models

**arXiv ID:** 2604.06483 | [PDF](https://arxiv.org/pdf/2604.06483v1)

**作者:** Dev Arpan Desai `[一作]` (Stevens Institute of Technology), Zining Zhu `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 87 | [OpenAlex ID](https://openalex.org/A5101563385)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了多 GPU LLM 的激活级解释与实时行为控制，支持完整层级的 logit lens 与 steering 向量注入

**💡 创新点**

创新点在于单通道捕获、延迟词表投影与 Tensor‑parallel 集成，使激活内存压缩 7 倍、吞吐提升 41 倍，同时实现无额外前向传播的即时 steering

**🔧 技术方法**

采用 DeepSpeed Tensor‑parallel 推理、Block Output Wrapper、logit lens、post‑LayerNorm steering、bfloat16 计算与 batched LM‑head 投影

**📊 数据集**

在 LLaMA‑3.1（8B、70B）和 Qwen‑3（4B、14B、32B）等公开模型上进行实验，使用公开 prompt 序列和多规模 token 长度

**📈 对比分析**

与 LogitLens4LLMs 单 GPU 基线相比，在 4×RTX A6000 上实现 41× 的速度提升，吞吐保持 20–100 tokens/s，且在不同模型规模下性能差异不超过 1.7×

**⚠️ 局限性**

仅适用于 decoder‑only transformer，未覆盖训练时解释、编码‑解码或非 transformer 架构；在极大 steering multiplier 时可能出现饱和或流畅度下降

---

## 144. STDec: Spatio-Temporal Stability Guided Decoding for dLLMs

**arXiv ID:** 2604.06330 | [PDF](https://arxiv.org/pdf/2604.06330v1)

**作者:** Yuzhe Chen `[一作]` (Tianjin University), Yanwei Pang `[通讯]` (Tianjin University)

**通讯引用:** 13322 | [OpenAlex ID](https://openalex.org/A5086887025)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于空间-时间稳定性的 dLLM 解码策略 STDec，动态生成每个掩码 token 的自适应阈值，以加速推理。

**💡 创新点**

创新点在于同时利用已解码 token 的空间邻域信息和跨步 ID 一致性来调整阈值，摆脱了全局阈值限制，并实现训练无关且可与缓存加速兼容。

**🔧 技术方法**

主要技术包括：空间-aware 通过高低阈值初始图与高斯平滑得到空间自适应阈值；temporal-aware 通过 ID 连续一致性与阈值放松因子实现时间自适应阈值；结合 dLLM 的迭代去噪框架。

**📊 数据集**

使用的评测数据集包括文本推理 Benchmark（MBPP、HumanEval、GPQA、GSM8K、MATH）和多模态理解 Benchmark（LaViDa Reason、MathVerse、MathVision、MathVista）。

**📈 对比分析**

与 Dream、LLaDA 以及 dKV-Cache、Fast-dLLM、LocalLeap 等基线对比，STDec 在文本推理上平均提升约 7.6× 速度且保持或略降分数，在多模态任务上提升约 3.5×，并可进一步叠加缓存加速取得更高增益。

**⚠️ 局限性**

局限性在于对只需生成少量 token 的任务加速效果有限，且主要侧重解码速度，未对去噪特征提取进行显著优化。

---

## 145. Multiscale topology optimization of compressible and nearly incompressible anisotropic hyperelastic structures using physics-augmented neural networks

**arXiv ID:** 2604.06519 | [PDF](https://arxiv.org/pdf/2604.06519v1)

**作者:** Asghar A. Jadoon `[一作]` (University of Texas at Austin), Jan N. Fuhg `[通讯]` (University of Texas at Austin)

**通讯引用:** 1454 | [OpenAlex ID](https://openalex.org/A5058627053)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

本文提出了一种用于非线性超弹性材料的并行多尺度拓扑优化框架，该框架利用物理增强神经网络（PANN）取代微尺度边界值问题，实现宏观材料分布与微尺度结构参数（如孔隙率、纤维方向）的同步优化。

**💡 创新点**

创新点包括：① 在神经网络架构中嵌入凸性、多面性和物理对称性约束，保证热力学一致性和数值稳定性；② 通过输入特定神经网络（ISNN）在不增加额外可微分负担的前提下实现显式可导的本构模型；③ 同时优化宏观拓扑与微观结构参数，显著提升结构刚度与材料利用率；④ 采用混合位移-压力有限元实现近似不可压材料的稳定求解。

**🔧 技术方法**

核心技术包括：物理增强神经网络（PANN）、输入特定神经网络（ISNN）、基于不变量的本构表述、混合有限元（近似不可压）、SIMP方法、投影与滤波、MMA优化器以及分析敏感性推导。

**📊 数据集**

训练数据来自对三种代表性RVE的数值同质化：纤维强化的横向各向同性RVE、带球形夹杂的立方各向异性RVE、以及近似不可压的等向性RVE。通过Latin Hypercube采样（高达20%应变）生成约200个变形梯度，计算对应的均质化应力，构成训练集。

**📈 对比分析**

实验结果与传统FE^2方法对比，PANN框架显著降低计算成本（微尺度求解次数从成千上万减少至一次前向求解），且在多尺度优化中实现了约8%的顺应率降低。数值例子包括受力梁、悬臂梁和3D立方体，展示了对纤维方向、夹杂体积分布的空间适配和对不可压限制的有效处理，整体性能优于单尺度或预设微结构方案。

**⚠️ 局限性**

局限性包括：① 需要在离线阶段预先构建微尺度同质化模型，假设尺度分离且微结构周期性；② 需预先知道材料的对称群，无法自动识别更复杂的微结构；③ 仅适用于无黏性、可变形弹性材料，未扩展至塑性、黏弹性或多场耦合；④ 未考虑制造约束、微结构不确定性或加载不确定性；⑤ 对非周期性、弱尺度分离的情况可能需要额外建模。

---

## 146. MICA: Multivariate Infini Compressive Attention for Time Series Forecasting

**arXiv ID:** 2604.06473 | [PDF](https://arxiv.org/pdf/2604.06473v1)

**作者:** Willa Potosnak `[一作]` (Carnegie Mellon University), Artur Dubrawski `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2408 | [OpenAlex ID](https://openalex.org/A5037154494)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多变量时间序列预测方法——Multivariate Infini Compressive Attention (MICA)，通过在原有单通道 Transformer 上添加线性跨通道注意力实现跨通道建模。

**💡 创新点**

创新点在于将 Infini‑Attention 的压缩注意力迁移到通道维度，使跨通道注意力从二次复杂度降至线性复杂度，同时通过可学习的混合门控平衡局部时间注意力与全局通道注意力。

**🔧 技术方法**

核心技术包括：局部的标准点积注意力、跨通道的线性注意力（使用 ELU 激活压缩键值矩阵）、可学习的 β‑门或 MLP 门控、通道排除与加权策略，以及基于 Patch 的 Encoder-Only Transformer 结构。

**📊 数据集**

实验使用了多域 Benchmark（ETT1/ETT2、Jena Weather、COVID‑Deaths、Loop‑Seattle、Solar、M‑Dense 等）以及自研的高频气象数据集，共计多种时序长度与通道数。

**📈 对比分析**

与单通道 Transformer、其他多变量 Transformer（Crossformer、TSMixer 等）及 MLP 基线相比，MICA 在 MAE 上平均提升约 5.4%（单个数据集最高 25.4%），并在 GFLOPs 与推理速度上比传统全通道注意力模型低 30‑80% 级别的成本。

**⚠️ 局限性**

局限性包括：尚未在大规模跨频率预训练上评估零样本性能；仅在 Transformer/MLP 体系内验证，未与状态空间或图模型进行对比；以及压缩注意力不保留记忆状态，可能不适用于需要跨批次长时间依赖的任务。

---

## 147. Matching Researchers to Funding Calls: A Reproducible Institution-Level Framework

**arXiv ID:** 2604.06321 | [PDF](https://arxiv.org/pdf/2604.06321v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 148. When to Call an Apple Red: Humans Follow Introspective Rules, VLMs Don't

**arXiv ID:** 2604.06422 | [PDF](https://arxiv.org/pdf/2604.06422v1)

**作者:** Jonathan Nemitz `[一作]` (University of Tübingen), William Rudman `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Graded Color Attribution（GCA）数据集，用于检验视觉-语言模型（VLM）是否遵循其自我推理规则，并与人类进行对比实验。

**💡 创新点**

创新点在于将颜色归因任务设计为可控、无显式正确答案的设置，从而将模型的“faithfulness”与任务难度解耦，并揭示世界知识偏见导致的自我规则失效。

**🔧 技术方法**

主要技术包括链式推理（Chain-of-Thought）提示的四种变体（Standard、Visual-Prior、Post-hoc、Text-Prior），以及阈值规则的提取与颜色覆盖率的估计。

**📊 数据集**

使用的数据集为自制的GCA集合，包含220个对象和25个形状的线稿图像，按0–100%逐步重着色，覆盖三种刺激类型（正向颜色、反事实颜色、无色先验）。

**📈 对比分析**

与人类（N=173）和四款主流VLM（GPT‑5‑mini、Claude Opus 4.6、Claude Haiku 4.5、Qwen 3.5‑9B）对比，发现模型在高能力模型也仅达约40–60%的一致性，而人类在经验阈值下保持约80%的faithfulness；模型的faithfulness随对象颜色先验显著下降。

**⚠️ 局限性**

局限性包括：仅针对简单颜色归因任务，难以推广到更复杂视觉推理；对模型内部机制的解释有限；实验集中在英文本身，未考虑多语言或多模态复杂度。

---

## 149. SMT-AD: a scalable quantum-inspired anomaly detection approach

**arXiv ID:** 2604.06265 | [PDF](https://arxiv.org/pdf/2604.06265v1)

**作者:** Apimuk Sornsaeng `[一作]`, Dario Poletti `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于量子启发式张量网络的并行可扩展异常检测算法 SMT-AD

**💡 创新点**

采用多分辨率 Fourier 嵌入与多重 MPO 叠加，实现参数线性增长、可高度并行且不依赖异常样本训练

**🔧 技术方法**

张量网络（MPS/MPO）、rank‑based 归一化、傅里叶多分辨率嵌入、基于熵与互信息的特征重要性分析、PyTorch GPU 加速

**📊 数据集**

UCI Wine、Lymphography、Thyroid、Satellite 以及 Kaggle Credit Card 四类交易数据

**📈 对比分析**

与 OC‑SVM、Isolation Forest、传统 TNAD 进行 AUROC/AUPRC 对比，SMT‑AD 在大部分数据集上达或超过基线，参数量极小（≈620）且训练并行效率高

**⚠️ 局限性**

对极度不平衡数据（如 Credit Card）AUPRC 仍低于某些基线，需进一步改进阈值校准与异常阈值设定

---

## 150. Stochastic Gradient Descent in the Saddle-to-Saddle Regime of Deep Linear Networks

**arXiv ID:** 2604.06366 | [PDF](https://arxiv.org/pdf/2604.06366v1)

**作者:** Guillaume Corlouer `[一作]` (Moirai), Alexander Gietelink Oldenziel `[通讯]` (Iliad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

研究了深度线性网络（DLN）中随机梯度下降（SGD）的训练动力学，利用连续时间的随机微分方程（SDE）对SGD进行建模，推导了噪声协方差并在平衡和对齐假设下将动力学分解为一维模态SDE，分析了模态扩散随时间的变化，并得到无标签噪声时的稳定分布为Dirac点，标签噪声时的分布近似高斯；实验验证了理论在无对齐/平衡条件下的可行性。

**💡 创新点**

1) 以精确形式得到DLN中SGD噪声协方差，揭示其状态相关、各向异的结构；2) 通过模态分解证明SGD扩散峰值先于特征完全学习，证明噪声携带学习进度信息；3) 分析无标签噪声下的极限分布为Dirac点，标签噪声时的近似Boltzmann分布；4) 实验验证即使不满足平衡/对齐假设，主要结论仍保持。

**🔧 技术方法**

连续时间SDE建模、Ito微积分、梯度噪声协方差解析、模态分解、Fokker–Planck方程与详细平衡分析、数值模拟（Euler–Maruyama）与实验对比。

**📊 数据集**

使用合成数据：教师矩阵M的奇异值与正交基，白噪声输入X ~ N(0,I)，可选的标签噪声ξ_q ~ N(0,σ_q^2I)。

**📈 对比分析**

将理论预测（模态扩散峰值、终态分布）与数值模拟以及离散SGD训练结果对比。实验显示模态扩散峰值提前出现，终态分布在无标签噪声时集中在教师奇异值，加入标签噪声后变为高斯，均与理论一致；SGD与梯度流在学习顺序上相同，速度差异在于噪声放大导致的时间尺度变慢。

**⚠️ 局限性**

1) 采用连续时间模型，忽略了离散学习率的影响；2) 依赖平衡和对齐假设，实际训练中常出现偏差；3) 只考虑单层交叉模态为零的情况，未能描述模态间耦合；4) 假设梯度噪声为高斯，未考虑重尾分布；5) 主要验证了结构性结论的可行性，未给出严格收敛证明。

---

## 151. Governing frontier general-purpose AI in the public sector: adaptive risk management and policy capacity under uncertainty through 2030

**arXiv ID:** 2604.06215 | [PDF](https://arxiv.org/pdf/2604.06215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 152. The End of Human Judgment in the Kill Chain? Relocating Initiative and Interpretation with Agentic AI

**arXiv ID:** 2604.06300 | [PDF](https://arxiv.org/pdf/2604.06300v1)

**作者:** Jovana Davidovic `[一作]` (Peace Research Institute), Jovana Davidovic `[通讯]` (Peace Research Institute)

**通讯引用:** 295 | [OpenAlex ID](https://openalex.org/A5071996937)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并论证在战场数据融合与战术指挥管理中使用LLM驱动的代理可能导致人类判断失效，提出对治理和政策的影响。

**💡 创新点**

提出LLM代理的四个核心特征（主动性、解释能力、目标导向与动态记忆）如何重塑决策链并推翻传统的人类判断与控制框架。

**🔧 技术方法**

基于LLM（如大规模语言模型）与外部工具的联动架构，使用ReAct/Chain‑of‑Thought等推理框架进行任务分解、信息融合与行动决策。

**📊 数据集**

本研究未使用实际数据集，主要通过案例研究（如Anduril Lattice、US DoD AI战略等公开资料）与文献综述进行理论分析。

**📈 对比分析**

比较方法主要为概念与案例对照，没有量化实验或性能指标；作者通过政策文件（GGE‑CCW、REAIM）与现有治理框架对比，阐释LLM代理在杀伤链中的风险与不可行性。

**⚠️ 局限性**

局限性：缺乏实证数据与量化评估；仅聚焦数据融合与指挥管理，未涵盖其他军事AI应用；对人类参与阈值和治理手段的具体实现缺乏细化，导致结论在实际政策制定中的可操作性待进一步验证。

---

## 153. Towards Realistic Waveform-Level IoT Network Simulation via IQ Mixing

**arXiv ID:** 2604.06408 | [PDF](https://arxiv.org/pdf/2604.06408v1)

**作者:** Alexis Delplace `[一作]` (Université Paris-Saclay), Dominique Quadri `[通讯]` (Université Paris-Saclay)

**通讯引用:** 302 | [OpenAlex ID](https://openalex.org/A5110613067)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了基于IQ流混合的 IoT 网络仿真框架 IQSim，能够在信号波形级别准确模拟物理层交互。

**💡 创新点**

创新点在于将真实接收机（软件或硬件）直接接收仿真生成的 IQ 波形，突破传统基于抽象碰撞规则的物理层建模，真正实现波形级干扰和共存效应的模拟。

**🔧 技术方法**

使用技术包括：IQ 波形在线/离线生成（GNU Radio、SDR 捕获）、IQ 域传播模型、IQStream 信号叠加、GPU 并行化处理、与真实硬件网关的 SDR 接口。

**📊 数据集**

主要使用自制的离线 IQ 数据集（由真实子 GHz 信号录制或软件合成得到），以及标准的路径损耗模型；没有公开数据集，实验基于实验室搭建的 IoT 环境。

**📈 对比分析**

通过在 1.5 MHz 子带上实现实时仿真，单 CPU 可支持 100 台设备，GPU 并行可扩展到数万台设备；与传统事件驱动仿真相比，能够真实捕捉邻频泄漏、交叉技术干扰等波形级效应，验证了框架的可行性。

**⚠️ 局限性**

局限性包括：离线 IQ 数据存储量大；在线生成时受限于软件调制器的可用性和计算开销；目前仅验证了子 GHz 场景，缺乏在更高频段和更大规模网络上的实证；以及需要进一步与真实测量数据对比以完善模型。

---

## 154. Attention Flows: Tracing LLM Conceptual Engagement via Story Summaries

**arXiv ID:** 2604.06416 | [PDF](https://arxiv.org/pdf/2604.06416v1)

**作者:** Rebecca M. M. Hicke `[一作]` (Cornell University), Ross Deans Kristensen-McLachlan `[通讯]` (Aarhus University)

**通讯引用:** 176 | [OpenAlex ID](https://openalex.org/A5087405347)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了一个基于小说摘要的框架，用以评估大语言模型（LLM）在长文本中的概念参与度；通过对150本小说的人工与LLM生成的摘要进行句子-章节对齐，探究模型与人类在叙事理解上的差异。

**💡 创新点**

创新点在于：①提出“概念参与度”作为衡量长文本理解的指标；②将小说摘要作为对齐任务，直接比较人类与模型的关注分布；③结合对齐结果与模型注意力权重，揭示模型在长文本中的“中间失效”现象；④公开提供对应数据、对齐方法与分析代码，推动后续研究。

**🔧 技术方法**

技术方法包括：自然语言处理中的句子分割、章节切分；使用TF‑IDF、嵌入匹配与多种LLM（开源与专有）进行句子‑章节对齐；对齐后计算多项统计量（章节覆盖率、线性度、偏度等）；对特定模型（如Qwen 3.5 (9b)）提取注意力矩阵，评估注意力与摘要句子对应关系。

**📊 数据集**

使用数据集：150本公共领域小说（Project Gutenberg）及其对应的维基百科情节摘要；对小说进行章节划分并清洗，生成5,550条人类/模型摘要，已发布在GitHub仓库。

**📈 对比分析**

比较方法：①对齐准确率（F1）评估句子‑章节匹配；②BLEU分数衡量摘要与人工摘要的词重叠；③统计句子数、词数、依赖距离、命名实体等句法特征；④计算章节覆盖率、章节跳过率、线性度（Kendall τ）与分布偏度。性能方面，模型在对齐任务上相较基线提升显著，但大多数模型仍低于人类；在概念参与度上，模型普遍偏向文本末端，缺乏线性叙事结构。

**⚠️ 局限性**

局限性包括：①仅涵盖英文小说，缺乏跨语言验证；②摘要对齐依赖LLM，仍可能引入误差；③未对摘要语义事件类型进行深入分析；④数据集中存在多作者写作风格的混合，可能影响人类摘要的统一性；⑤未对更短文本（短篇、戏剧）进行测试，可能限制方法的通用性。

---

## 155. The Stepwise Informativeness Assumption: Why are Entropy Dynamics and Reasoning Correlated in LLMs?

**arXiv ID:** 2604.06192 | [PDF](https://arxiv.org/pdf/2604.06192v1)

**作者:** Mar Gonzàlez I Català `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**通讯引用:** 33949 | [OpenAlex ID](https://openalex.org/A5056748708)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证Stepwise Informativeness Assumption (SIA)，解释大型语言模型内部熵动力学为何与推理正确率相关。

**💡 创新点**

SIA将推理前缀的累计信息量与答案相关性联系起来，为熵基推理分析提供理论依据；并证明最大似然训练及后续SFT/RL可转移此特性。

**🔧 技术方法**

信息论度量（条件熵、互信息）、最大似然优化、强化学习（PPO/GRPO）、蒙特卡罗抽样估计熵。

**📊 数据集**

GSM8K、ARC、SVAMP等算术/推理基准；使用多种开源LLM（Gemma-2、LLaMA-3.2、Qwen-2.5、DeepSeek、Olmo等）。

**📈 对比分析**

通过熵‑正确性相关系数、累计信息曲线、AUC分离度、熵饱和度等指标对比模型；发现训练后模型（SFT/RL）显著提升熵与答案一致性，正确推理轨迹表现出早期信息积累与熵下降特征。

**⚠️ 局限性**

SIA不一定在所有任务或模型上成立，尤其是无明确终点或开放式生成任务；对抗/无监督场景下熵下降可能不对应真实答案；需要进一步研究熵操纵的因果效果与跨模态推广。

---

## 156. MorphDistill: Distilling Unified Morphological Knowledge from Pathology Foundation Models for Colorectal Cancer Survival Prediction

**arXiv ID:** 2604.06390 | [PDF](https://arxiv.org/pdf/2604.06390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 157. DataSTORM: Deep Research on Large-Scale Databases using Exploratory Data Analysis and Data Storytelling

**arXiv ID:** 2604.06474 | [PDF](https://arxiv.org/pdf/2604.06474v1)

**作者:** Shicheng Liu `[一作]` (Stanford University), Monica S. Lam `[通讯]` (Stanford University)

**通讯引用:** 23284 | [OpenAlex ID](https://openalex.org/A5002419188)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种名为DataSTORM的多代理LLM系统，能够自动在大规模结构化数据库和互联网信息上进行深度研究。

**💡 创新点**

创新地将探索性数据分析与数据故事讲述原则结合，采用规划器-执行器分解、归纳统计与演绎推理相结合、查询一致性检测以及论文驱动的探索流程，实现从数据发现到叙事生成的闭环。

**🔧 技术方法**

利用大语言模型（如GPT‑5）构建的ReAct式多代理框架，执行器执行SQL查询，规划器生成探究问题，加入统计摘要、查询一致性模块和论文生成模块；最终报告通过分阶段编辑流水线生成。

**📊 数据集**

在InsightBench评测数据库以及新构建的ACLED（武装冲突位置与事件数据）数据库上进行实验。

**📈 对比分析**

与AgentPoirot和OpenAI Deep Research进行对比，使用LLM评判器（GPT‑4o、Qwen‑3‑30B）和人工评估，结果显示DataSTORM在InsightBench的洞察召回率提高19.4%、摘要分数提升7.2%，在ACLED上比基线提升10.6%参考匹配度、RACE总体得分提升5.8点，并且数据库引用率高出36%。

**⚠️ 局限性**

受限于LLM的知识范围和推理误差，评测数据中存在注释错误，时序趋势分析仍是主要瓶颈，且最终报告的可读性和呈现风格仍需改进。

---

## 158. Attribution-Driven Explainable Intrusion Detection with Encoder-Based Large Language Models

**arXiv ID:** 2604.06266 | [PDF](https://arxiv.org/pdf/2604.06266v1)

**作者:** Umesh Biswas `[一作]` (Mississippi State University), Charan Gudla `[通讯]` (Mississippi State University)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5038423069)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于编码器的 LLM（RoBERTa 与 DeBERTa）在 SDN 流量异常检测任务中进行 attribution‑driven 可解释性分析，利用文本化流量特征和 Integrated Gradients 揭示模型决策依据。

**💡 创新点**

创新点在于首次将 IG 方法应用于文本化流量特征的 transformer IDS，系统对比两模型的推理策略，并证明模型使用的行为特征与传统 IDS 原则一致。

**🔧 技术方法**

使用技术包括 RoBERTa 与 DeBERTa encoder、固定顺序文本化 flow feature、class‑weighted cross‑entropy 训练、IG 归因、去重与分层拆分等。

**📊 数据集**

使用数据集为 CICIDS2017 的 SDN 流量特征集，去重后 366,870 条样本，划分为 3 类标签（Benign / DDoS / Web Attack）。

**📈 对比分析**

通过宏平均 F1、各类 Precision/Recall 对比两模型；DeBERTa 在 Web Attack 上 F1≈0.972、宏 F1≈0.990，RoBERTa 在 Web Attack 上 F1≈0.913、宏 F1≈0.970，整体准确率均 ≈99.9%。

**⚠️ 局限性**

局限性包括仅评估 coarse 3‑way 分类、对细粒度攻击缺乏解释、解释依赖 IG 的梯度估计可能受噪声影响，且未验证跨域或实时推理的可部署性。

---

## 159. ART: Attention Replacement Technique to Improve Factuality in LLMs

**arXiv ID:** 2604.06393 | [PDF](https://arxiv.org/pdf/2604.06393v1)

**作者:** Ziqin Luo `[一作]` (Fudan University), Chen Shen `[通讯]` (Alibaba Cloud Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出训练无关的注意力替换技术ART，在LLM浅层将冗余的均匀注意力替换为局部注意力，提升模型生成的真实性和推理准确性。

**💡 创新点**

通过对浅层注意力头分类型（均匀、局部、分散）进行可视化与量化，发现局部注意力对生成至关重要，进而设计了只替换均匀注意力的无训练干预方法，具备通用性与可插拔特性。

**🔧 技术方法**

采用m‑index度量衡量注意力与理想均匀分布的偏差，选择最小m‑index的注意力头作为均匀注意力，对应最大m‑index的头作为局部注意力；在推理阶段将前k个均匀头替换为局部头（ART‑mean/ART‑max）。

**📊 数据集**

在TruthfulQA、LogiQA、GSM8K三大问答与推理数据集上进行零样本CoT评估，并在多种开源LLM模型（Llama2‑7B‑Chat、Llama3.1‑8B‑Instruct、Mistral‑8B‑Instruct‑2410、Qwen‑7B‑Instruct、Qwen2.5‑7B/14B/32B‑Instruct）上实验。

**📈 对比分析**

与原始零样本推理基线、Beam Search、ITI、ACT、DoLa 等对照，ART 在大多数任务上提升 0.6–3.4% 的准确率，尤其在 TruthfulQA 和 LogiQA 上平均提升 1.1% 以上；在 7B/8B 模型上效果更显著。

**⚠️ 局限性**

仅在浅层进行替换，深层注意力未优化；k 的选择需经验化调参；对不同模型的适用性与解释性尚需进一步验证；局部注意力替换可能在某些任务导致上下文理解下降。

---

## 160. ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache

**arXiv ID:** 2604.06370 | [PDF](https://arxiv.org/pdf/2604.06370v1)

**作者:** Shao Wang `[一作]` (Shanghai Jiao Tong University), Lin Gui `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 7257 | [OpenAlex ID](https://openalex.org/A5062168574)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于复制写时（CoW）机制的多LoRA代理服务系统，利用内存分离技术高效共享KV缓存

**💡 创新点**

创新点在于将KV缓存物理拆分为大规模共享部分与轻量化代理特有部分，并通过DualRadixTree实现继承与CoW管理，辅以ResidualAttention核实现内存内重构

**🔧 技术方法**

使用了fork+CoW内存管理、DualRadixTree结构、ResidualAttention自定义核以及大型语言模型的LoRA适配技术

**📊 数据集**

实验涵盖多种主流语言模型及不同任务的实际数据集，但未给出具体数据集名称

**📈 对比分析**

与现有多LoRA服务系统比较，所提方法在吞吐量上提升高达3.0倍，生成质量几乎不受影响

**⚠️ 局限性**

限制方面需依赖支持CoW的系统与硬件，管理与调度复杂度上升，且在极大规模代理场景下可能产生额外的内存管理开销

---

## 161. Context-Aware Dialectal Arabic Machine Translation with Interactive Region and Register Selection

**arXiv ID:** 2604.06456 | [PDF](https://arxiv.org/pdf/2604.06456v1)

**作者:** Afroza Nowshin `[一作]` (University of Toledo), Fayeq Jeelani Syed `[通讯]` (University of Toledo)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5040021321)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个上下文感知、可调节的阿拉伯语机器翻译框架，能够在推理时显式控制方言和社会语调。

**💡 创新点**

创新点包括：Rule-Based Data Augmentation (RBDA) 将 3,000 句种子扩展到 57,000 句，覆盖八种方言并实现平衡；多标签提示结构使用户可显式指定方言与语域；在单一 mT5 模型上实现多方言共存与可控生成。

**🔧 技术方法**

采用的技术包括：mT5-base 微调、规则化词汇替换与平衡抽样、标签前缀控制、Gradio UI 实时演示。

**📊 数据集**

使用自建的 57,600 句平行语料，来源于 1.3M 通用对句、Tatoeba 项目以及 OPUS，经过 RBDA 处理后平衡八种方言并提供元数据标签，公开发布于 Hugging Face。

**📈 对比分析**

与 NLLB 与 Helsinki 基线对比：基线在 BLEU、METEOR、chrF++ 上更高（13.75/11.96/41.12）但方言一致性低；自模型 BLEU 为 8.19、METEOR 0.2583、chrF++ 27.01；LLM 辅助方言真实性评分平均 4.80/5，对比基线 1.00/5，表明自模型在方言适配性上表现更佳。

**⚠️ 局限性**

局限性：仅在词汇层面进行增强，未覆盖形态或句法深层差异；大部分数据为合成，可能引入“翻译式”偏差；方言正字法不统一，模型对极端异体写法的处理有限；LLM 辅助评估受模型偏差影响，缺乏人工评测支持。

---

## 162. Toward Reducing Unproductive Container Moves: Predicting Service Requirements and Dwell Times

**arXiv ID:** 2604.06251 | [PDF](https://arxiv.org/pdf/2604.06251v1)

**作者:** Elena Villalobos `[一作]` (Tecnológico de Monterrey), Alejandra Matadamaz `[通讯]` (Container Terminal Operations)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了基于机器学习的预测模型，用于提前识别港口码头到岸集装箱是否需要预清关服务以及预计停留时间，以支持堆场规划与资源配置。

**💡 创新点**

首次从码头运营视角出发，将服务需求与停留时间预测联合建模，使用基于时间交叉验证的严格评估框架，并通过基于运营启发式的实际基准进行对比。

**🔧 技术方法**

采用随机森林、XGBoost、LightGBM等集成算法，并结合TF‑IDF文本分类、三元组相似度记录链接等数据预处理技术，形成丰富的时序特征。

**📊 数据集**

利用港口提供的两大数据库（容器操作记录与搬运记录）以及公共HS编码目录，对约两百万条容器信息和1300万条搬运记录进行处理，构建训练集与验证集。

**📈 对比分析**

与现行的基于承运人/收货人历史行为的启发式规则和随机分配基准相比，服务预测模型的精度提升约45个百分点，停留时间预测在短期/长期类别上分别提升至约30–40%与80–90%的召回率。

**⚠️ 局限性**

模型仅依赖到港前可用信息，无法捕捉到港期间出现的文档延误或客户请求变更；验证仅在单一墨西哥港口进行，缺乏跨港口泛化性与直接量化堆场搬运减少的实证结果。

---

## 163. Predicting Alzheimer's disease progression using rs-fMRI and a history-aware graph neural network

**arXiv ID:** 2604.06469 | [PDF](https://arxiv.org/pdf/2604.06469v1)

**作者:** Mahdi Moghaddami `[一作]` (Oakland University), Huirong Fu `[通讯]` (Oakland University)

**通讯引用:** 2233 | [OpenAlex ID](https://openalex.org/A5083522562)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种历史感知图神经网络（HA‑GNN），结合图卷积网络与循环神经网络，对阿尔茨海默病患者多时点静息态功能磁共振影像的功能连接图进行时空建模，用以预测患者在下一次临床访视时是否会出现认知状态恶化。

**💡 创新点**

创新点包括：① 将功能连接图的图结构表示与患者访视时间信息融合的历史感知框架；② 在模型中嵌入访视间距信息以应对不规则访视间隔；③ 采用两阶段训练（单访视诊断预训练+多访视预测），并利用焦点损失解决类别不平衡。

**🔧 技术方法**

使用了图卷积网络（GraphSAGE+TopK池化）、循环神经网络（LSTM/GRU/RNN）、预处理工具FMRIPrep、Nilearn及Schaefer 100区划分、贝叶斯优化与焦点损失；模型整体以PyTorch实现。

**📊 数据集**

数据集为阿尔茨海默病神经影像倡议（ADNI）中的TADPOLE子集，包含303名受试者共1089个静息态fMRI扫描，涵盖CN、MCI、AD三阶段；每位受试者访视次数不同，平均访视间隔约14.8个月。

**📈 对比分析**

与无预训练的单一RNN模型相比，HA‑GNN（LSTM）在5折交叉验证中取得平均整体准确率82.9%（±5.8%），平衡准确率77.1%（±11.4%），AUC‑ROC 0.852（±0.065）。在CN→MCI和MCI→AD转换任务中，准确率分别为68.8%和67.6%。这些结果已与部分仅使用结构MRI或表征特征的研究做对比，显示出相对竞争力。

**⚠️ 局限性**

主要局限包括：① 样本量有限且稳定/转换样本不平衡，导致平衡准确率波动大；② 仅使用功能连接信息，未整合结构MRI、扩散MRI或临床/遗传数据；③ 目前仅采用时间间隔嵌入，未探索更高级的不规则时间序列处理方法；④ 缺乏可解释性机制；⑤ 仅在ADNI内部验证，尚未对多中心或真实临床数据进行外部验证。

---

## 164. Application-Driven Pedagogical Knowledge Optimization of Open-Source LLMs via Reinforcement Learning and Supervised Fine-Tuning

**arXiv ID:** 2604.06385 | [PDF](https://arxiv.org/pdf/2604.06385v1)

**作者:** Navan Preet Singh `[一作]` (Forta), Ritankar Das `[通讯]` (Incept Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用多阶段 RL 与 SFT 对 Qwen3-32B 进行领域专精，打造高性能的教育 LLM EduQwen 系列。

**💡 创新点**

创新的 RL+SFT 循环、难度加权数据采样与多步推理 Rollout 结合，显著提升了教学策略推理能力。

**🔧 技术方法**

采用 DAPO 强化学习、梯度聚合 SFT、难度加权采样、生成合成数据、混合精度训练等技术。

**📊 数据集**

主要使用 Cross-Domain Pedagogical Knowledge (CDPK) 评测集（920 题多选）以及 TutorBench 进行多模态验证。

**📈 对比分析**

通过与公开排行榜中 Gemini‑3 Pro 等模型对比，EduQwen 32B‑SFT‑RL2 在 CDPK 上达到 96.52% 准确率，领先所有开源与专有模型。

**⚠️ 局限性**

局限在于仅在教师考试式多选题上验证，缺乏自由对话、长期学习效果及真实课堂场景的评估。

---

## 165. Guiding Symbolic Execution with Static Analysis and LLMs for Vulnerability Discovery

**arXiv ID:** 2604.06506 | [PDF](https://arxiv.org/pdf/2604.06506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 166. ARIA: Adaptive Retrieval Intelligence Assistant -- A Multimodal RAG Framework for Domain-Specific Engineering Education

**arXiv ID:** 2604.06179 | [PDF](https://arxiv.org/pdf/2604.06179v1)

**作者:** Yue Luo `[一作]` (Dalian University of Technology), Somdatta Goswami `[通讯]` (Johns Hopkins University)

**通讯引用:** 3849 | [OpenAlex ID](https://openalex.org/A5015683810)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 ARIA，一种基于检索增强生成（RAG）的通用框架，用于在大学课程中提供域特定的教学辅助。

**💡 创新点**

创新点包括三大技术融合的多模态抽取管线（Docling、Nougat、GPT‑4 Vision）、域无关的 RAG 体系结构、基于关键词和元数据的教学控制机制以及高效的开源嵌入模型。

**🔧 技术方法**

使用的技术包括检索增强生成（RAG）、多模态内容抽取、文本嵌入模型（text‑embedding‑3‑large、e5‑large‑v2）、Docling（文本与表格分析）、Nougat（公式识别）、GPT‑4 Vision（图示解析）、Prompt 设计与对话管理、Streamlit 前端。

**📊 数据集**

数据集主要为约 33 份讲义、58 条作业问答及 80 题评测集，全部来自约翰斯·霍普金斯大学静力学与材料力学（Statics and Mechanics of Materials）课程。

**📈 对比分析**

通过与 ChatGPT‑5、单一抽取工具以及多种嵌入模型的对比实验评估，指标包括精确率 90.9%、召回率 100%、F1 95.2%、整体准确率 97.5%；在教学效果指标上 ARIA 获得 94.2% 的综合得分，略优于 ChatGPT‑5 的 91.7%。

**⚠️ 局限性**

局限性包括多模态抽取的计算成本高、对图像处理依赖、数学推导与精确计算能力仍有限、对大型多学科部署的可扩展性待验证，以及缺乏真实学习效果和长期保留率的评估。

---

## 167. Concentrated siting of AI data centers drives regional power-system stress under rising global compute demand

**arXiv ID:** 2604.06198 | [PDF](https://arxiv.org/pdf/2604.06198v1)

**作者:** Danbo Chen `[一作]` (Ohio State University), Lei Chen `[通讯]` (Zhejiang A&F University)

**通讯引用:** 74469 | [OpenAlex ID](https://openalex.org/A5100418548)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个 AI‑能源耦合框架，结合大语言模型语义分析与电力系统建模，预测 2025‑2030 年全球 AI 数据中心的电力需求及其对电网的影响。

**💡 创新点**

创新点在于将 LLM 用于解析企业、政策与媒体文本，生成对 AI 数据中心布局与扩张的可量化预测，并引入 Power Stress Index (PSI) 评估区域电网压力，形成全流程的数字‑能源耦合分析。

**🔧 技术方法**

使用技术包括 Hugging Face 的 all‑MiniLM‑L6‑v2 嵌入、FAISS 索引检索、情景建模与 PSI 计算，以及传统电力系统负荷与可再生资源匹配分析。

**📊 数据集**

数据来源为 2015‑2025 年企业报告、政策文件、媒体报道的多源文本集合，并结合公开的 AI 企业投资信息、全球电力系统容量与生成数据。

**📈 对比分析**

通过三种保守/中性/乐观情景对比，模型预测全球 AI 数据中心电力需求从 2024 年的 118 TWh 增至 2030 年的 239–295 TWh，复合年增长率约 13–17%，表明模型在规模和区域层面具备较高的预测一致性。

**⚠️ 局限性**

局限包括对 LLM 语义推断的可靠性依赖、缺乏对 AI 模型效率与冷却技术进步的动态更新、以及区域电网数据时效性与粒度不足，可能导致 PSI 评估的偏差。

---

## 168. Unsupervised Neural Network for Automated Classification of Surgical Urgency Levels in Medical Transcriptions

**arXiv ID:** 2604.06214 | [PDF](https://arxiv.org/pdf/2604.06214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 169. The Impact of Response Latency and Task Type on Human-LLM Interaction and Perception

**arXiv ID:** 2604.06183 | [PDF](https://arxiv.org/pdf/2604.06183v1)

**作者:** Felicia Fang-Yi Tan `[一作]` (New York University), Oded Nov `[通讯]` (New York University)

**通讯引用:** 8606 | [OpenAlex ID](https://openalex.org/A5007172071)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型在不同任务类型（创作与建议）与响应延迟（2 s、9 s、20 s）下的用户交互行为与感知质量。

**💡 创新点**

创新点在于将响应延迟视为可调设计变量，而非单纯性能瓶颈，并系统探讨其对用户思考度、实用性评估及信任的影响。

**🔧 技术方法**

使用了GPT‑4o API，并在自定义前后端系统中实现了可控的时间到首词延迟（TTFT）为2、9、20秒。

**📊 数据集**

数据来源于240名来自Prolific的美国成人受试者，完成六种任务场景（三种创作与三种建议），共计三轮交互并记录详细日志。

**📈 对比分析**

通过3×2实验设计对比行为日志与主观评估，发现延迟显著提升思考度与实用性评估，创作任务产生更多提示；性能差异不大，表明延迟对任务效果的影响在于感知而非实际产出。

**⚠️ 局限性**

局限性包括仅操纵首词延迟、实验环境缺乏真实工作压力、样本单一文化背景，以及未考虑流速、总生成时长等时间维度。

---

## 170. Intimacy as Service, Harm as Externality: Critical Perspectives on AI Companion Platform Accountability

**arXiv ID:** 2604.06381 | [PDF](https://arxiv.org/pdf/2604.06381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 171. Asymptotic-Preserving Neural Networks for Viscoelastic Parameter Identification in Multiscale Blood Flow Modeling

**arXiv ID:** 2604.06287 | [PDF](https://arxiv.org/pdf/2604.06287v1)

**作者:** Giulia Bertaglia `[一作]` (University of Ferrara), Raffaella Fiamma Cabini `[通讯]` (University of Ferrara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用一维多尺度血流模型和渐近保持神经网络（APNN），从患者的血管截面积和速度数据中无创重建血压波形，并自动识别血管壁的瞬时杨氏模量和松弛时间。

**💡 创新点**

创新点在于将渐近保持特性嵌入物理信息神经网络，使模型在不同松弛时间尺度下保持正确的超声波传播或扩散极限，从而既能精确预测血压，又能可靠估计不可直接测量的可变粘弹参数。

**🔧 技术方法**

技术主要包括：1D血流多尺度可变粘弹模型、渐近保持物理信息神经网络（APNN）、自动微分求解残差、Softplus 输出保证截面积正值、使用Adam优化器与指数变换保持参数正值、基于结构化网格的残差采样。

**📊 数据集**

数据集包括：① 用上胸主动脉（TA）数值仿真产生的合成数据；② 三位健康志愿者右侧颈总动脉（CCA）实际测量数据，包含B‑mode 超声截面积、脉冲波多普勒速度以及托曼测压波形。

**📈 对比分析**

与基准仿真结果比较：在合成案例中，截面积、速度和压力的平均相对误差分别为0.19%、1.1%和0.26%；在体内案例中，截面积和速度误差均小于1%，压力误差在3.5%–6.5%之间，显示出良好的预测精度和泛化能力。

**⚠️ 局限性**

局限性包括：① 对可变粘弹参数估计误差相对较大（E₀误差约12%–20%，τ_r误差约20%–30%）；② 训练时间和计算成本高；③ 体内实验缺乏完整空间域真值，仅能在单点验证；④ 仅针对主动脉和颈动脉，未知其在其他血管分支的适用性；⑤ 对测量噪声和数据缺失的鲁棒性仍需进一步研究。

---

## 172. MedConclusion: A Benchmark for Biomedical Conclusion Generation from Structured Abstracts

**arXiv ID:** 2604.06505 | [PDF](https://arxiv.org/pdf/2604.06505v1)

**作者:** Weiyue Li `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**通讯引用:** 2715 | [OpenAlex ID](https://openalex.org/A5100632182)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个5.7M条PubMed结构化摘要的基准数据集，用于评估大语言模型在从证据推断科学结论的能力。

**💡 创新点**

创新点在于：①提供大规模、作者原写结论作为监督；②加入期刊级元数据（学科类别、SJR分数）实现子域难度分析；③结合基于规则的评估与LLM-as-a-judge的多维评分，探究结论写作与摘要写作的差异。

**🔧 技术方法**

采用了多种LLM（闭源前沿模型、开源指令调优模型、多模态与推理模型）以及自定义提示，评估模型在结论生成与摘要生成两种提示下的表现。

**📊 数据集**

使用的数据集是从2000-2025年间具有结构化摘要的PubMed文章提取的5,692,839条记录，包含作者写作的结论和期刊元数据。

**📈 对比分析**

通过LLM-as-a-judge的语义相似度、写作风格一致性、非矛盾性、数值一致性、正式度五维评分以及ROUGE、BLEU、嵌入相似度等基准指标比较，发现顶尖模型在语义相似度和非矛盾性上表现最好，但不同评估方式下模型间差距压缩，且结论写作显著优于摘要写作。

**⚠️ 局限性**

局限性包括：①评估高度依赖judge的身份和尺度；②自动指标（如ROUGE）与实际结论质量偏差大；③缺乏对生成结论真实性与完整性的进一步验证，且仅针对生物医学领域，跨领域通用性待检验。

---

## 173. Conformal Margin Risk Minimization: An Envelope Framework for Robust Learning under Label Noise

**arXiv ID:** 2604.06468 | [PDF](https://arxiv.org/pdf/2604.06468v1)

**作者:** Yuanjie Shi `[一作]` (Washington State University), Yan Yan `[通讯]` (Washington State University)

**通讯引用:** 477993 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Conformal Margin Risk Minimization (CMRM)，在任何分类损失上加入置信边际的 conformal quantile 正则化，以提升在标签噪声下的鲁棒性。

**💡 创新点**

创新点：不依赖噪声模型、干净子集或辅助网络，仅需边际分布的平滑性假设；利用 conformal quantile 校准置信边际，形成方法无关的 uncertainty 信号；作为单一正则化项即可嵌入任何已有训练流程。

**🔧 技术方法**

技术手段：置信边际定义、conformal quantile 估计、soft indicator 权重、批量级估计、Rademacher 复杂度理论与学习界限推导。

**📊 数据集**

数据集：CIFAR-100、mini-ImageNet、Food-101（合成噪声），以及 CIFAR-10N、CIFAR-100N（真实注释噪声）。

**📈 对比分析**

与 CE、Focal、LDAM、GCE、NI-ERM 等基线对比；在多类任务平均提升约 1.6% 的 Top‑1 准确率、降低约 7.3% 的平均预测集大小；在二分类任务提升 AUROC、AUPRC、准确率等，同时不在无噪声时造成性能退化。

**⚠️ 局限性**

局限性：对 α 超参数存在一定敏感性；批量级 quantile 估计导致近似误差；假设分布平滑且正密度需在实际分布中验证，若分布异常可能影响效果。

---

## 174. Beyond Facts: Benchmarking Distributional Reading Comprehension in Large Language Models

**arXiv ID:** 2604.06201 | [PDF](https://arxiv.org/pdf/2604.06201v1)

**作者:** Pei-Fu Guo `[一作]` (National Taiwan University), Shou-De Lin `[通讯]` (National Taiwan University)

**通讯引用:** 3108 | [OpenAlex ID](https://openalex.org/A5087480257)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了Text2DistBench，一个自动化、可持续更新的分布式阅读理解基准，用于评估LLM从文本中推断分布式知识的能力。

**💡 创新点**

创新点在于聚焦分布式知识而非事实知识，引入从YouTube评论自动标注情感与主题的方式，并持续更新以避免数据泄漏。

**🔧 技术方法**

使用LLM自动注释、联合分布估计、模板化问答生成以及TVD和分类准确率等评估指标。

**📊 数据集**

使用来自YouTube电影和音乐评论的实体及其元数据，包含情感和主题标注。

**📈 对比分析**

通过零样本提示对多个LLM进行评估，实验显示所有模型显著优于随机基线，但在不同分布类型和任务上表现差异明显；在边际分布上效果最好，联合分布最难。

**⚠️ 局限性**

局限在于仅提供三种任务类型、仅覆盖电影音乐两领域、缺少更丰富的查询类型和跨领域泛化。

---

## 175. Learning to Interrupt in Language-based Multi-agent Communication

**arXiv ID:** 2604.06452 | [PDF](https://arxiv.org/pdf/2604.06452v1)

**作者:** Danqing Wang `[一作]` (Carnegie Mellon University), Ansong Ni `[通讯]` (Meta FAIR)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可中断的多智能体沟通框架，使听者在接收信息时可随时打断说话者以减少生成与传输成本；

**💡 创新点**

创新点在于将中断决策转化为学习问题，利用树形采样估计未来奖励与成本，训练出基于任务收益的最优中断时机；

**🔧 技术方法**

技术包括：LLM基准（Llama‑3.1、GPT‑4o、Gemini‑2.0‑flash）、提示式与训练式中断决策、树形采样评估、指令微调；

**📊 数据集**

数据集涵盖三类场景：文本拼图（100个实体）、会议调度（50个约束实例）与 MMLU‑Pro 辩论（100道推理题）等；

**📈 对比分析**

与非中断、随机中断和提示式中断等基线比较，实验显示 24%–49% 的通信成本下降，同时保持或提升成功率，尤其在 Llama‑70B/405B 上表现尤为突出；

**⚠️ 局限性**

局限性包括：中断策略依赖任务特定，需在新任务上微调；在更复杂的多方向中断环境中，回合数有时会增加，且仍未完全解决对说话者压缩难度的挑战。

---

## 176. Blending Human and LLM Expertise to Detect Hallucinations and Omissions in Mental Health Chatbot Responses

**arXiv ID:** 2604.06216 | [PDF](https://arxiv.org/pdf/2604.06216v1)

**作者:** Khizar Hussain `[一作]` (Virginia Tech), Murat Kantarcioglu `[通讯]` (Virginia Tech)

**通讯引用:** 11409 | [OpenAlex ID](https://openalex.org/A5087192873)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种结合人类专家知识与大型语言模型（LLM）的框架，用于检测精神健康聊天机器人回复中的幻觉和遗漏，解决传统LLM判断在临床语境中缺乏专业细致评估的问题。

**💡 创新点**

创新点在于：①构建了基于多方利益相关者（临床医生、患者、管理员等）达成共识的高质量标注数据集；②设计了五个专家驱动的维度（逻辑一致性、实体验证、事实准确性、语言不确定性、专业适当性）进行可解释特征提取；③将这些特征与LLM-判断分数相结合，通过任务专属监督学习实现显著性能提升。

**🔧 技术方法**

使用技术包括：LLM-as-a-Judge（GPT-4o、GPT-5等）进行分数生成；LLM引导的多维特征提取；传统机器学习分类器（LightGBM、XGBoost、CatBoost、SVM、Logistic回归、MLP、SAINT）进行训练与融合；交叉验证、阈值调优、特征消融与相关性分析等评估手段。

**📊 数据集**

数据集：自建4,418条幻觉标注样本（87正例）和3,368条遗漏标注样本（124正例），以及公开的Kaggle精神健康对话数据集（994条）。

**📈 对比分析**

与传统方法（SelfCheckGPT、SAC3、RefChecker）和单一LLM-判断基线比较，使用F1、准确率、召回率、AUC衡量。实验结果显示，融合特征+ML模型在自建数据集上幻觉F1达0.717、遗漏F1达0.637；在Kaggle数据集上幻觉F1可达0.849，AUC>0.91；相较于LLM-判断的F1<0.2，提升幅度超过300%。

**⚠️ 局限性**

局限性包括：①标注过程耗时且难以扩展，缺乏大规模自动化标签；②模型仍需专业监督，不能完全替代临床判断；③对数据集的适应性可能受限，需在更广泛的实际部署中进一步验证；④部分特征可能受LLM输出偏差影响，需持续监测和更新。

---

## 177. PhysHead: Simulation-Ready Gaussian Head Avatars

**arXiv ID:** 2604.06467 | [PDF](https://arxiv.org/pdf/2604.06467v1)

**作者:** Berna Kabadayi `[一作]` (Max Planck Institute for Intelligent Systems), Gerard Pons-Moll `[通讯]` (University of Tübingen)

**通讯引用:** 14442 | [OpenAlex ID](https://openalex.org/A5076908763)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 PhysHead，一种基于层化 3D 高斯拆分与毛发线条模拟的可控三维人脸头像模型，能够在多视角视频和静态毛发捕捉的基础上实现表情、视角和物理驱动的动态毛发渲染。

**💡 创新点**

创新点包括：①将面部与毛发拆分为独立层，使用 3DMM‑绑定高斯和线条毛发实现物理驱动动画；②利用视觉‑语言模型生成无毛头图像填补遮挡区域；③提出毛发颜色一致性正则化，解决不可见毛发段颜色不确定性；④两阶段训练策略实现高质量头部与毛发的分离与组合。

**🔧 技术方法**

核心技术涵盖 3D 高斯溅射（Gaussian Splatting）、FLAME 3DMM、NeuralHaircut 线条重建、物理仿真引擎（Semi‑implicit Euler）、可微渲染、Poisson 图像编辑、TNB 坐标、VLM 编辑（Nano‑Banana）以及邻近色彩一致性损失。

**📊 数据集**

实验数据主要来自 Ava‑256 数据集（每位受试者 16 台摄像机的多视角视频），并结合 360° 静态头部捕捉来重建毛发；训练时还使用 VLM 生成的无毛图像。

**📈 对比分析**

与 Gaussian Avatars (GA)、Gaussian Head Avatars (GHA)、Gaussian Haircut (GH) 以及 HairCUP 等基线相比，PhysHead 在动态场景下实现了无刚性毛发、自然风吹等物理效果，视觉质量显著优于基线；定量指标未给出，但实验显示物理可行性和动态一致性均更佳。

**⚠️ 局限性**

局限性包括对前景/毛发掩码的依赖，若掩码不精确会导致面部与毛发分离错误；受 NeuralHaircut 毛发几何质量限制，难以处理极度卷曲或特殊发型；VLM 生成图像的色彩一致性与连贯性可能受限；训练过程复杂且计算成本较高。

---

## 178. SASLO: A Scene-Aware Spatial Layout Optimization System for AR-SSVEP

**arXiv ID:** 2604.06190 | [PDF](https://arxiv.org/pdf/2604.06190v1)

**作者:** Beining Cao `[一作]` (University of Technology Sydney), Chin-Teng Lin `[通讯]` (University of Technology Sydney)

**通讯引用:** 35070 | [OpenAlex ID](https://openalex.org/A5058936239)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一套基于场景感知的AR-SSVEP刺激布局优化系统（SASLO），实现了在户外环境中实时自适应布局以提升SSVEP解码性能。

**💡 创新点**

创新点在于将背景亮度与刺激间距（ISD）联合考虑，利用RGB-CIE亮度估计与线性情境多臂赌博机（LCB）实现动态布局优化，并通过离线单因素实验构造人类导向的奖励函数。

**🔧 技术方法**

主要技术包括RGB-CIE亮度估计、线性情境多臂赌博机（LCB）推荐模型、双频段模糊推理解码器、3D SSVEP刺激、HoloLens 2+Unity实时系统、Lab Streaming Layer、FFT频谱分析等。

**📊 数据集**

LCB模型使用Cityscapes街景数据集进行训练；在线实验使用10名受试者在户外校园环境下采集12通道EEG，记录实际SSVEP性能。

**📈 对比分析**

通过与仅基于亮度优化（LOO）和无优化（NO）两基线对比，在线10名受试者、3 s输入窗口下，JOLI方法平均准确率为0.89、ITR为35.74 bits/min，显著优于LOO（0.85/31.68）和NO（0.75/23.69），且在短窗口（≤1.5 s）提升更明显。

**⚠️ 局限性**

局限性包括仅在二维平面进行布局优化；仅考虑亮度与ISD两因子，未涵盖深度、刺激尺寸、背景运动等可能影响SSVEP的因素；模型依赖离线实验奖励，跨硬件或不同环境的泛化性可能受限。

---

## 179. ValueGround: Evaluating Culture-Conditioned Visual Value Grounding in MLLMs

**arXiv ID:** 2604.06484 | [PDF](https://arxiv.org/pdf/2604.06484v1)

**作者:** Zhipin Wang `[一作]` (University of Technology Nuremberg), Steffen Eger `[通讯]` (University of Technology Nuremberg)

**通讯引用:** 2677 | [OpenAlex ID](https://openalex.org/A5053947568)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ValueGround 基准，用于评估多模态大型语言模型在文化价值条件下的视觉价值归纳。

**💡 创新点**

将文化价值评估从纯文本迁移到可视化选项，并通过最小对比图像对构造控制对比与防止捷径，构建可控的视觉对齐测试。

**🔧 技术方法**

采用代理式图像生成管线（Planner、Editor、Critic）与自动/人工质量评估，结合世界价值调查（WVS）问卷文本与国家价值倾向。

**📊 数据集**

使用 World Values Survey（WVS）数据，构造 224 对符合质量准则的图像对，并覆盖 13 个国家。

**📈 对比分析**

在 13 个国家、6 种多模态 LLM 上对比主任务（文本+图像）、文本仅与图像对齐三种设置，主任务平均准确率为 65.8%，最强模型 Gemini‑3‑Flash 达 75.6%，显示文本仅优于主任务 7.1 点，图像对齐更高但仍未能弥补主任务缺口。

**⚠️ 局限性**

受限于将多元回答二元化、部分文化价值难以直观可视化、对真实多样性和生态环境的缺失，导致主任务仍难以将国家背景、问题语义与视觉对比有效整合。

---

## 180. Revisiting Fairness Impossibility with Endogenous Behavior

**arXiv ID:** 2604.06378 | [PDF](https://arxiv.org/pdf/2604.06378v1)

**作者:** Elizabeth Maggie Penn `[一作]` (Emory University), John W. Patty `[通讯]` (Emory University)

**通讯引用:** 2037 | [OpenAlex ID](https://openalex.org/A5034341181)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文在个体对算法分类具有自适应行为的情境下，研究了如何通过调整分类结果的后果来实现公平。

**💡 创新点**

创新点在于提出一种两阶段设计：先统一统计性能，再通过差异化处罚来消除误差率平衡与预测平行的不可兼容性。

**🔧 技术方法**

主要使用了理论分析与公平性不可能性框架，结合策略行为模型。

**📊 数据集**

未使用具体数据集，全部为理论模型。

**📈 对比分析**

未进行实验比较，论文通过数学证明展示了新公平性权衡的可行性。

**⚠️ 局限性**

局限在于缺乏实证验证，假设个体完全理性且后果调整的可行性和道德约束不充分。

---

## 181. Fine-tuning Whisper for Pashto ASR: strategies and scale

**arXiv ID:** 2604.06507 | [PDF](https://arxiv.org/pdf/2604.06507v1)

**作者:** Hanif Rahman `[一作]` `[通讯]` (Independent Researcher), Hanif Rahman (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文针对 Pashto 低资源语音识别，系统比较了 Whisper 基础模型的四种微调策略（完整微调、LoRA、编码器冻结、双阶段 Urdu→Pashto 迁移），并进一步对不同模型规模（base、small、turbo）在 113 小时 Pashto 数据上的表现进行评估。

**💡 创新点**

创新点在于：①首次在同一实验框架下完整对比四种微调策略；②发现 Whisper‑base 的 6 层编码器冻结会导致性能下降，揭示了层深度对冻结策略的影响；③剖析双阶段迁移失败的原因；④在不同规模模型上量化精度与参数的收益边际；⑤将所有微调模型、数据集和评测脚本公开。

**🔧 技术方法**

使用的技术包括：Whisper‑base、Whisper‑small、Whisper‑large‑v3‑turbo；完整微调、LoRA（rank‑64）、冻结前 2/6 层编码器、双阶段 Urdu→Pashto 迁移；在线与离线数据增强（速度扰动、噪声注入、音高、增益、混响）；评估使用 HuggingFace 的 WER、CER、正则化文本等指标。

**📊 数据集**

采用 Mozilla Common Voice Pashto 数据集：v20（≈60 h）用于策略对比实验，v24（≈960 h，其中 113 h 训练集）用于规模评估。所有实验均使用预先 Augment 的固定训练集，确保各策略共享相同的输入分布。

**📈 对比分析**

在策略对比中，完整微调取得最佳 WER（CV20 测试集 21.22%），LoRA 仅 54.58%，冻结 35.98%，双阶段 65.78%；在规模评估中，Whisper‑small 在 113 h 数据上仅比 base 多 2.24 pp，Turbo 进一步减 1.52 pp，显示参数增加带来的边际收益递减。在线增广对 base 产生 7.25 pp 的显著 WER 改善。

**⚠️ 局限性**

局限性包括：LoRA 仅试验 r=64、Q/V 投影；超参搜索不够系统；未对方言进行分层评估；未提供推理延迟、内存占用等部署指标；策略比较与规模评估使用不同数据版本，结果不可直接对比。

---

## 182. LLMs Have Made Failure Worth Publishing

**arXiv ID:** 2604.06236 | [PDF](https://arxiv.org/pdf/2604.06236v1)

**作者:** Sungmin Lee `[一作]` `[通讯]`, Sungmin Lee

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文分析了在大语言模型时代科学出版物的正向结果偏差，提出了将失败结果纳入公开记录的必要性，并阐述了相关实验设计与结构化失败分类法。

**💡 创新点**

创新点在于将LLM的训练与评估与失败数据的系统化公开结合，提出了针对失败数据的分类体系、实验验证框架，以及从失败数据中提升LLM预测与同行评审质量的可能性。

**🔧 技术方法**

主要技术包括大语言模型（LLM）推理、负向知识蒸馏、基于错误的课程学习以及实验性LLM提示与评估方法。

**📊 数据集**

虽然本文未进行实验，但提议使用ClinicalTrials.gov临床试验注册数据、公开的OpenReview评审数据以及已发表的失败案例集作为验证素材。

**📈 对比分析**

比较方法通过让LLM在不同上下文（无额外信息、仅正向试验、仅失败试验、正负组合、分类失败+原因分析）下预测临床试验成功概率，并以AUC‑ROC衡量性能；预期正向结果仅会放大偏差，包含失败信息将提升预测准确性。

**⚠️ 局限性**

局限性包括缺乏实证验证、失败数据公开的激励与治理难题、分类体系尚未测试且可能与LLM学习机制不匹配，以及对LLM是否能真正从失败文本中获益仍存疑。

---

## 183. SensorPersona: An LLM-Empowered System for Continual Persona Extraction from Longitudinal Mobile Sensor Streams

**arXiv ID:** 2604.06204 | [PDF](https://arxiv.org/pdf/2604.06204v1)

**作者:** Bufang Yang `[一作]` (Chinese University of Hong Kong), Zhenyu Yan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 77214 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种基于大语言模型（LLM）的系统，能够从多模态的长期传感器数据流中持续推断用户的稳定个性特征，以适应用户的偏好并提高响应质量和任务表现。

**💡 创新点**

创新点在于通过长期传感器数据流推断用户个性，而不是依赖于聊天历史中的自我披露信息，从而捕捉用户在现实世界中的持续行为模式。

**🔧 技术方法**

使用了大语言模型（LLM）进行个性推断，结合了层次化的个性推理和上下文编码技术，以处理多模态传感器数据。

**📊 数据集**

使用了自收集的数据集，包含来自20名参与者的1580小时的传感器数据，数据收集跨越3个月，涵盖17个城市和3个大洲。

**📈 对比分析**

与现有的最先进基线相比，该系统在个性提取中提高了31.4%的召回率，在个性感知代理响应中获得了85.7%的胜率，用户满意度显著提高。

**⚠️ 局限性**

限制在于系统可能会随着时间推移而产生大量个性，导致管理和维护的复杂性，同时在处理冗余和噪声数据时可能面临挑战。

---

## 184. WebSP-Eval: Evaluating Web Agents on Website Security and Privacy Tasks

**arXiv ID:** 2604.06367 | [PDF](https://arxiv.org/pdf/2604.06367v1)

**作者:** Guruprasad Viswanathan Ramesh `[一作]` (University of Wisconsin-Madison), Kassem Fawaz `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 WebSP‑Eval，针对网站安全与隐私任务的完整评估框架，包括 200 条手工构造的任务实例、基于 Selenium 的 WebAgent 系统以及自动判定器。

**💡 创新点**

首创专门评估安全/隐私设置的基准，结合了可复现的初始状态、专用 Chrome 扩展的账户与状态管理，以及对 UI 元素细粒度的性能剖析。

**🔧 技术方法**

采用多模态 LLM（Gemini、Claude、GPT‑5、Gemma）、Selenium 自动化、自研的记录‑重放 Chrome 扩展、增强的 WebVoyager 行动空间、以及多模型投票的 MLLM‑Judge。

**📊 数据集**

使用 200 条任务实例（涵盖 28 个主流网站、138 种不同任务），每个实例配备预设的初始状态，形成专门的安全隐私评测数据集。

**📈 对比分析**

对 8 种多模态 LLM 进行对比实验：在有导航提示时最高成功率 83%，无导航时 76.5%；Gemma‑3‑27B 仅 21‑26%；发现状态化 UI（切换、勾选框）导致 >45% 的失败；导航信息显著提升性能。

**⚠️ 局限性**

局限性包括仅覆盖非敏感网站、需人工注册和记录账户、易受网站 UI/结构变更影响、跨地域可复现性差、以及公开模型与商业模型性能差距显著。

---

## 185. All LCA models are wrong. Are some of them useful? Towards open computational LCA in ICT

**arXiv ID:** 2604.06290 | [PDF](https://arxiv.org/pdf/2604.06290v1)

**作者:** Vincent Corlay `[一作]` (Mitsubishi Electric R&D Centre Europe), Sebastien Rumley `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

重新梳理ICT系统生命周期评估（LCA）的基本原理，指出其本质是一个由多模型组成的系统，并提出四项关键要求（模型血缘、模型范围、可追溯性、非陈旧性）来提升可靠性；

**💡 创新点**

创新点在于将LCA视为模型系统，识别并分类模型使用中的两大“诅咒”，并基于此提出可操作的框架，包括显式依赖图、开放版本化模型仓库、自动完整性约束和模型分类体系；

**🔧 技术方法**

技术手段主要是：1）显式构建模型依赖图；2）使用版本化仓库（类似软件包管理器）管理模型与数据；3）实现自动完整性约束校验；4）建立统一模型分类（产品/过程模型、影响模型、参数转换模型等）；

**📊 数据集**

使用的数据集主要为公开的LCI数据库与案例数据，如Ecoinvent、NegaOctet、ADEME Base Empreinte、ElecImpact等；

**📈 对比分析**

比较方法：通过示例展示不同数据库选择、范围不匹配等导致的结果差异，强调利用依赖图与版本化仓库可实现从源头到结果的完整追溯，理论上能提升结果可解释性与可复现性；并未给出量化性能指标，侧重方法论与框架设计；

**⚠️ 局限性**

局限性：本文缺乏针对所提框架的实验验证与实现细节，实际工具和标准化程度有限；模型更新与依赖冲突的自动处理机制仍待实现；对数据质量评估和不确定性量化的处理仍依赖现有方法，未提供新的量化方法；

---

## 186. Uncertainty Estimation for Deep Reconstruction in Actuatic Disaster Scenarios with Autonomous Vehicles

**arXiv ID:** 2604.06387 | [PDF](https://arxiv.org/pdf/2604.06387v1)

**作者:** Samuel Yanes Luis `[一作]` (University of Sevilla), Daniel Gutiérrez Reina `[通讯]` (University of Sevilla)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对自主车辆在海洋灾害情境下稀疏观测的标量场进行重建与不确定性估计，聚焦于点估计之外的主动感知需求

**💡 创新点**

首次系统比较高斯过程、MC Dropout、深度集成与证据深度学习四类方法在多种观测模型（点传感器、锥形视场、垂直摄像）下的性能与不确定性分解，发现证据深度学习在精度、校准与推理速度上均优于其他方法

**🔧 技术方法**

采用U‑Net骨干网络结合蒙特卡洛Dropout、深度集成、证据深度学习和基线高斯过程，利用heteroscedastic NLL、NIG似然和经验正则化实现不确定性分解

**📊 数据集**

使用三套2000个油污模拟样本的合成数据集，分别对应三种观测模型，覆盖100×100网格的物理仿真结果

**📈 对比分析**

通过RMSE、UCE（不确定性校准误）以及推理时间进行评估，证据深度学习在RMSE约为0.027、UCE≈0.0018、推理≈7 ms，远优于深度集成（RMSE≈0.050、UCE≈0.0026、≈20 ms）、MC Dropout（RMSE≈0.088、UCE≈0.0084、≈202 ms）和高斯过程（RMSE≈0.115、UCE最高、推理最高）

**⚠️ 局限性**

高斯过程受限于可变长度尺度与均匀噪声假设，导致不校准且推理复杂度随观测数呈立方增长；深度方法虽性能优越，但需大量训练数据与网络结构设计；未来需验证在闭环信息路径规划中的实际收益

---

## 187. "It didn't feel right but I needed a job so desperately": Understanding People's Emotions & Help Needs During Financial Scams

**arXiv ID:** 2604.06218 | [PDF](https://arxiv.org/pdf/2604.06218v1)

**作者:** Jake Chanenson `[一作]` (Google), Amelia Hassoun `[通讯]` (Google)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Reddit 上的诈骗求助帖进行定性分析，识别诈骗利用的情感动机及受害者在不同阶段的求助需求，并提出相应的干预策略。

**💡 创新点**

提出了五大情感动机框架和对 User States Framework 的细化，揭示了诈骗过程中的情感与求助需求关联；同时构建了情感驱动的诈骗干预设计思路。

**🔧 技术方法**

使用大语言模型（Gemini 1.5 Flash）进行数据筛选和标签；结合代码簿主题分析法进行定性编码。

**📊 数据集**

采集并筛选 405 篇 Reddit 原贴，覆盖 12 种已知及新兴诈骗类型。

**📈 对比分析**

本研究为定性方法，未给出量化性能指标，主要通过主题分析和案例对比阐释发现。

**⚠️ 局限性**

仅包含英文 Reddit 数据，缺乏多语种和多平台样本；样本非随机，覆盖面有限；依赖单一 LLM 可能带来偏差。

---

## 188. Negotiating Privacy with Smart Voice Assistants: Risk-Benefit and Control-Acceptance Tensions

**arXiv ID:** 2604.06235 | [PDF](https://arxiv.org/pdf/2604.06235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 189. Contextual Chain: Single-State Ledger Design for Mobile/IoT Networks with Frequent Partitions

**arXiv ID:** 2604.06529 | [PDF](https://arxiv.org/pdf/2604.06529v1)

**作者:** Song-Ju Kim `[一作]` `[通讯]` (SOBIN Institute LLC), Song-Ju Kim (SOBIN Institute LLC)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并评估了一种轻量级账本协议，结合节点本地上下文认证与自适应同步，旨在 IoT 等间歇性嘈杂网络中实现快速一致性恢复。

**💡 创新点**

提出了可操作的上下文认证规则（检查点优先分叉选择、分支评分与不一致性 EMA 监测），并将同步预算与不一致性阈值耦合，仅在不一致性高时自动增大 gossip 预算，从而显著提升恢复性能。

**🔧 技术方法**

使用离散事件模拟、检查点优先分叉、分支评分、指数滑动平均不一致性估计、隔离标志、以及基于不一致性阈值的自适应 gossip 预算控制。

**📊 数据集**

在无真实数据集的情况下，采用随机种子生成的模拟网络，节点数 N=20、50、100，清洁和嘈杂两种链路模型（无丢包/有丢包+延迟）进行评估。

**📈 对比分析**

通过四种协议变体（q禁/启 + gossip 1/4）在不同分区比例、网络规模和噪声条件下进行 500–1000 种子实验；实验显示：同步预算提升可使成功率提升至 >0.85、恢复尾部（p95）降至 <200 s；仅靠隔离标志无法改善恢复，说明信息缺失是主要瓶颈。

**⚠️ 局限性**

主要限制包括：同步预算未随网络规模自适应，扩展至 N=50/100 时需手动调参；仅使用模拟评估，未测量真实网络流量；缺乏正式的密码学安全证明，仅提供可操作性和资源消耗的经验性洞察。

---

## 190. Multi-objective Evolutionary Merging Enables Efficient Reasoning Models

**arXiv ID:** 2604.06465 | [PDF](https://arxiv.org/pdf/2604.06465v1)

**作者:** Mario Iacobelli `[一作]` (Independent Researcher), Emanuele Rodolà `[通讯]` (Sapienza University Of Rome)

**通讯引用:** 7056 | [OpenAlex ID](https://openalex.org/A5087051832)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Evo-L2S框架，利用多目标进化搜索在大语言模型中实现长链思考向短链思考的合并，显著减少推理长度同时保持或提升准确率。

**💡 创新点**

将长至短推理问题视作多目标优化问题，并通过进化模型合并在权重空间探索Pareto前沿；引入熵基子集采样以高效估计适应度，解决传统算术合并方法的超参数脆弱性。

**🔧 技术方法**

使用NSGA-II多目标进化算法、MergeKit/ Mergenetic模型合并库、PyMoo框架；熵基子集采样、QwenLM评测工具、贝尔努利熵筛选。

**📊 数据集**

六个数学推理基准（Minerva-Math、MATH500、GSM8K、OlympiadBench、College-Math、AIME24）。

**📈 对比分析**

与系统1/系统2基线、固定超参数算术合并（Average、TA、TIES）、ACM以及单目标进化比较；Evo-L2S在1.5B模型上可将推理长度降幅超过55%，并提升4–5个百分点准确率；在7B模型上平均准确率下降不足1个百分点，同时长度降幅58%，整体性能优于所有基线。

**⚠️ 局限性**

对极度复杂的AIME24等小规模基准仍有准确率下降；进化搜索仍需大量评估，虽通过熵采样降低成本，但对超大规模模型仍存在计算开销；当前仅限于两端点合并，扩展到多模型或跨架构尚未验证。

---

## 191. RPM-Net Reciprocal Point MLP Network for Unknown Network Security Threat Detection

**arXiv ID:** 2604.06638 | [PDF](https://arxiv.org/pdf/2604.06638v1)

**作者:** Jiachen Zhang `[一作]`, Daoqi Han `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 85 | [OpenAlex ID](https://openalex.org/A5009595207)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于互补点机制与对抗边缘约束的多类别开集识别框架RPM‑Net，用于网络安全威胁的未知攻击检测。

**💡 创新点**

创新点包括：①互补点学习“非类别”表示，②对抗边缘约束在特征空间构造可解释的未知区块，③在RPM‑Net基础上加入Fisher判别正则化提升类内紧凑性与类间分离度。

**🔧 技术方法**

采用多层感知机特征提取器、互补点与边缘参数学习、交叉熵+margin+Fisher三重损失训练，并在推理时通过互补点距离阈值判定未知样本。

**📊 数据集**

在CICIDS2017（5个已知类别）和UNSW‑NB15（6个已知类别）两个网络入侵数据集上进行实验。

**📈 对比分析**

与Baseline、ODIN、OCN、EVM等方法比较，RPM‑Net在已知类别的宏F1、AUROC、AUPR‑OUT指标均显著提升，RPM‑Net++更进一步提升了AUPR‑OUT（CICIDS2017 0.6711，UNSW‑NB15 0.8664）。

**⚠️ 局限性**

局限性包括：①仅在离线静态数据集上验证，缺乏对实时流式数据的适应性；②对超参数（阈值、margin、正则化系数）敏感，需要手工调优；③未考虑不同攻击类别之间更细粒度的语义关系与迁移学习潜力。

---

## 192. Argus: Reorchestrating Static Analysis via a Multi-Agent Ensemble for Full-Chain Security Vulnerability Detection

**arXiv ID:** 2604.06633 | [PDF](https://arxiv.org/pdf/2604.06633v1)

**作者:** Zi Liang `[一作]` (Hong Kong Polytechnic University), Kaishun Wu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 11134 | [OpenAlex ID](https://openalex.org/A5001188748)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于多智能体、检索增强与推理优化的全新静态应用安全测试框架Argus，主导并重构了漏洞检测流程

**💡 创新点**

创新点在于：1）LLM中心化而非工具辅助；2）完整供应链分析与多智能体协作；3）结合RAG、ReAct及Re³（检索+递归+复核）技术以减少幻觉、提升深度推理与数据流覆盖

**🔧 技术方法**

核心技术包括：大型语言模型（Claude Sonnet）、检索增强生成（RAG）、ReAct推理框架、CodeQL静态分析、PoC生成与验证、Re³多步数据流复核

**📊 数据集**

使用七个大型Java开源项目（PublicCMS、JeecgBoot、Ruoyi、JSPWiki、DataGear、Yudao-Cloud、KeyCloak）作为实验数据集，并利用公开的漏洞数据库（NVD、OSV、GHSA、Snyk）进行检索

**📈 对比分析**

与传统工具CodeQL、IRIS对比；Argus在探测到的漏洞数、sink数显著提高，同时令牌消耗仅略高，展示出更高效、精准的零日漏洞挖掘能力

**⚠️ 局限性**

局限性在于仅关注静态污点分析，未涵盖动态检测（如模糊测试），且多智能体系统尚未通过强化学习等方式进一步优化。

---

## 193. PD-SOVNet: A Physics-Driven Second-Order Vibration Operator Network for Estimating Wheel Polygonal Roughness from Axle-Box Vibrations

**arXiv ID:** 2604.06620 | [PDF](https://arxiv.org/pdf/2604.06620v1)

**作者:** Xiancheng Wang `[一作]` (Harbin Institute of Technology), Kaitai Mao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5101297549)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个物理指导的灰盒框架 PD‑SOVNet，用共享的二阶振动核、4×4 MIMO耦合、适应性物理校正以及 Mamba 时序分支，从车轴箱振动信号回归轮辐 1–40 级多边形粗糙度谱。

**💡 创新点**

在振动频域引入共享的二阶振动响应核作为先验；设计 MIMO 耦合捕捉四轴交互；构建基于样本上下文的自适应物理校正；加入 Mamba 时序分支补偿残余动态，实现灰盒+数据驱动的结构化回归。

**🔧 技术方法**

物理驱动的第二阶振动算子网络、MIMO耦合模块、上下文编码器与自适应校正头、Mamba 时间序列模块、角度域重采样、深度监督回归、MAE 与 R² 评估。

**📊 数据集**

三组真实高速铁路数据（Dataset I、II、III），包含车轴箱振动、车速与轮辐轮廓粗糙度标签，采用轮组划分的未见轮实验。

**📈 对比分析**

与 CNN、BiLSTM、Mamba、Transformer 以及基于 FRF 的物理模型对比；在低训练误差和训练状态敏感性两种分析下，PD‑SOVNet 在所有数据集上 MAE 与 R² 与主流方法相当或更优，尤其在 Dataset III 上表现最为稳定（MAE ≈4.7 dB）。

**⚠️ 局限性**

模型仍受限于低维物理近似、单一速度区间与车辆架构、标签同步误差；噪声鲁棒性不足，需要更广泛的训练数据与实时性能优化。

---

## 194. Neural parametric representations for thin-shell shape optimisation

**arXiv ID:** 2604.06612 | [PDF](https://arxiv.org/pdf/2604.06612v1)

**作者:** Xiao Xiao `[一作]` (Shanghai Jiao Tong University), Fehmi Cirak `[通讯]` (University of Cambridge)

**通讯引用:** 3383 | [OpenAlex ID](https://openalex.org/A5016815096)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现一种基于神经网络参数化的薄壳形状优化方法（NRep），通过将薄壳中面映射为可微的MLP实现全局可调的几何表示，并在有限元分析与梯度优化中直接更新网络参数，实现了形状优化。

**💡 创新点**

创新点在于使用周期性（sin）激活函数的多层感知器来近似薄壳中面，从而以极少的设计变量（网络权重和偏置）捕捉高频几何细节，实现了与传统网格/样条方法相比更紧凑、更可微的几何参数化；同时将自动微分与有限元灵敏度耦合，形成统一的梯度优化框架。

**🔧 技术方法**

主要技术包括：多层感知器（MLP）网络设计、周期性激活函数、有限元薄壳分析（Kirchhoff–Love），梯度约束优化（SQP/MMA），自动微分求取网络参数对体积和结构遵循性的灵敏度。

**📊 数据集**

使用的“数据集”是基于标准薄壳基准问题的几何模型（短边固定的薄壳条、均匀负载的方形薄壳、含中央开口或不同支撑条件的薄壳、以及由优化形状生成的 lattice‑skin 结构），并无公开数据库；所有实验均在自定义几何上进行。

**📈 对比分析**

通过与传统样条/网格参数化的基准结果比较（如catenary 曲线、均匀负载方形薄壳的符合度）和不同网格细化程度的对比，证明了NRep在保持或提升结构合规性、减少设计变量、提高计算效率方面的性能优势，且在多种边界/加载/拓扑配置下均能得到平滑、物理合理的优化结果。

**⚠️ 局限性**

局限性包括：目前仅采用全连接 MLP 结构，可能不适合高度复杂或局部细节丰富的几何；对大规模问题的可扩展性需通过 GPU 加速或神经代理模型进一步提升；以及仅聚焦于形状优化，未结合拓扑或多物理耦合；最后对不同激活函数和网络深度的系统性分析仍待深入。

---

## 195. Scientific Knowledge-driven Decoding Constraints Improving the Reliability of LLMs

**arXiv ID:** 2604.06603 | [PDF](https://arxiv.org/pdf/2604.06603v1)

**作者:** Maotian Ma `[一作]` (Nanjing University), Yukun Yan `[通讯]` (Tsinghua University)

**通讯引用:** 252119 | [OpenAlex ID](https://openalex.org/A5071127149)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 SciDC 框架，将科学领域知识自动转换为多层约束（顶层结构约束、中层逻辑约束、底层 token 约束），并将这些约束嵌入 LLM 的生成过程中，提升专业任务（配方设计、肿瘤诊断、逆向合成）的生成质量与可靠性。

**💡 创新点**

创新点在于：①使用强通用 LLM 将灵活的知识文档编译为可执行的三层约束；②将约束与本地域 LLM 协同工作，既保留域模型的隐私与效率，又保证生成过程被知识严格引导；③实现无参数调优、无手工规则编写即可自动化知识驱动生成。

**🔧 技术方法**

技术方法包括：通用 LLM（Claude‑3.5‑Sonnet、GPT‑5）进行知识编译；本地域 LLM（Qwen3‑14B/4B、ChemDFM‑v1.5‑8B）在约束空间内解码；三层约束实现：logit 遮蔽（底层）、条件检查与局部回溯（中层）、结构化提示与多步推理（顶层）；人机交互验证规则代码。

**📊 数据集**

使用的数据集包括：工业配方设计（458 条样本 + 1.7k token 经验指南）；甲状腺癌 TNM 分期诊断（200 虚拟病例 + 500 token 指南）；化学逆向合成（USPTO‑460k 反应模板，201 条未见产品测试样本）；并在 LegalBench 评估法律推理任务。

**📈 对比分析**

与无知识（w/o K）及仅使用提示的基线相比，SciDC 在三大任务上平均提升约12% 的准确率/有效率；具体表现为配方设计、肿瘤诊断、逆向合成的有效率、准确率和 hit@1 等指标显著提升；虽然生成长度与重生成次数略高，导致推理时间约为基线的 2‑3 倍，但在可接受范围内。

**⚠️ 局限性**

局限性包括：①需要将所有隐性、物理约束显式编码，难以覆盖所有专业知识；②多次调用生成函数导致效率低，尚未实现 decoder 级别的状态复用；③对域 LLM 的性能高度依赖，若域模型不够强大，效果可能不及通用 LLM；④当前框架缺少更高效的硬件实现，需进一步优化。

---

## 196. DynLP: Parallel Dynamic Batch Update for Label Propagation in Semi-Supervised Learning

**arXiv ID:** 2604.06596 | [PDF](https://arxiv.org/pdf/2604.06596v1)

**作者:** S M Shovan `[一作]` (Missouri University of Science and Technology), Mahantesh Halappanavar `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 2079 | [OpenAlex ID](https://openalex.org/A5075175819)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种面向动态批处理的GPU并行标签传播算法（DynLP），能够仅更新受影响节点的标签，从而避免对完整图重新计算；

**💡 创新点**

创新点在于利用连通分量结构实现快速初始化，并通过差分更新与稀疏化技术，仅对增删节点及其邻域进行局部传播，显著降低计算量；

**🔧 技术方法**

核心技术包括CSR稀疏图存储、块级行并行负载平衡、Shiloach‑Vishkin并行连通分量求解、块级线程协作求边权和，以及基于阈值δ的迭代收敛判定；

**📊 数据集**

实验使用合成Erdős–Rényi图以及真实图数据集：IMDB、ImageNet、Yelp、Amazon Household、Amazon Book，以及大规模随机图；

**📈 对比分析**

与Wagner等的流式LP、A2LP、CAGNN、以及基于拉普拉斯逆的传统方法相比，DynLP在速度上平均提升13倍至102倍，内存使用比传统方法低约100倍，准确率接近最优（约99%）；

**⚠️ 局限性**

局限性包括仅支持二分类任务、对稀疏图依赖、阈值δ需手工调优以及在极大更新批量时仍可能产生额外迭代开销。

---

## 197. VAMAE: Vessel-Aware Masked Autoencoders for OCT Angiography

**arXiv ID:** 2604.06583 | [PDF](https://arxiv.org/pdf/2604.06583v1)

**作者:** Ilerioluwakiiye Abolade `[一作]` (Federal University of Agriculture Abeokuta), Solomon Odelola `[通讯]` (ML Collective)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种面向光学相干断层血管成像（OCTA）的 vessel‑aware masked autoencoder (VAMAE)，用于在无标注数据上自监督预训练并提升血管分割效果。

**💡 创新点**

创新点包括：① 采用基于 Frangi 结构张力和骨架信息的 vessel‑aware 掩码策略，将高信息量的血管区域优先遮盖；② 采用多目标重建（强度、血管性、骨架）联合学习，以同时捕捉外观、结构与拓扑信息；③ 在预训练中引入渐进掩码和多目标损失权重平衡，进一步提升表示质量。

**🔧 技术方法**

技术实现基于 Vision Transformer 的 encoder‑decoder 结构，配合自监督掩码重建框架，使用 0.3/0.5/0.2 的损失权重；预训练使用 300 轮渐进掩码，后续细化阶段采用 U‑Net 风格解码器进行分割微调；所有计算均在 PyTorch 2.0+CUDA 上完成。

**📊 数据集**

使用公开的 OCTA‑500 数据集（500 条 3mm FOV 视场 OCTA 图像）进行预训练；下游分割任务在 200 条带标签图像上评估，分别分为大血管、视盘无血管区（FAZ）和静脉。

**📈 对比分析**

与随机 MAE、MedMAE、BioVessel‑Net、Pissas 等无监督/自监督基线以及 U‑Net、CS^2‑Net 等监督基线对比。VAMAE 在全量标签下达到 82.4% Dice，50% 标签时仍能实现 78.4%，在所有任务中均显著优于基线（平均提升 5‑10%），并在标注稀缺条件下实现 50% 标注成本降低。

**⚠️ 局限性**

局限性：① 预处理依赖 Frangi 过滤和骨架提取，病理严重时可能失效；② 需要额外的前处理计算；③ 当前仅处理 2D en‑face 投影，未扩展至 3D 体素；④ 在极端病变（血管严重缺失）下性能仍有下降。

---

## 198. ExplainFuzz: Explainable and Constraint-Conditioned Test Generation with Probabilistic Circuits

**arXiv ID:** 2604.06559 | [PDF](https://arxiv.org/pdf/2604.06559v1)

**作者:** Annaëlle Baiget `[一作]` (UCLA), Miryung Kim `[通讯]` (UCLA)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于概率电路（Probabilistic Circuits, PCs）的语法感知可控测试输入生成框架，并将其应用于SQL和XML两大领域。

**💡 创新点**

核心创新在于将上下文无关文法（CFG）编译为可训练的概率电路，既能捕捉上下文敏感的概率依赖，又可通过PC的条件推理实现精确约束控制，兼具可解释性与可控性。

**🔧 技术方法**

技术手段包括：1) 语法重构与词法匿名化；2) 采用PC编译算法将CFG转化为可训练的计算图；3) 在PC上进行概率推理（EVI、MAR、COND、MMAP）和受约束采样；4) 为不同领域实现定制化的token化与具象化（concretization）器。

**📊 数据集**

使用了七个领域的真实语料库（SQL、JANUS、REDIS、B、CSV、HTML、JSON）和SQL/XML两大测试环境，构建了多种种子集与人工合成的bug或acles。

**📈 对比分析**

与传统pCFG、无语法PC（PC‑HMM）及GPT‑2 LLM对比，PC在五个域中实现最低perplexity；在SQL和XML的bug触发实验中，PC相较于基线语法变异模糊器提升了约23.6%（SQL）和17.5%（XML）的bug覆盖率，并在SQL上平均多出417个、XML上多出1587个独立触发输入，且在条件约束下进一步提升多样性。

**⚠️ 局限性**

主要限制包括：1) 需要针对每个领域实现具象化器，缺乏通用自动化；2) 受限于输入长度与词法抽象，可能忽略更长或更复杂结构；3) 对非常低频或结构稀疏的语法构造，PC仍可能过拟合或生成率低；4) 与LLM相比，对大规模、多语种场景的适配尚未充分验证。

---

## 199. When Does Context Help? A Systematic Study of Target-Conditional Molecular Property Prediction

**arXiv ID:** 2604.06558 | [PDF](https://arxiv.org/pdf/2604.06558v1)

**作者:** Bryan Cheng `[一作]` (Great Neck South High School), Jasper Zhang `[通讯]` (Great Neck South High School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了在分子性质预测中使用目标上下文的效果，覆盖10种蛋白家族、4种融合架构、不同数据规模和时间/随机拆分；提出并验证了基于FiLM的嵌套学习架构；对DUD‑E基准进行了深度审计并提出时间拆分评估；

**💡 创新点**

①发现上下文融合方式至关重要，FiLM显著优于拼接和加法；②在数据稀缺目标（如CYP3A4）上上下文可实现多任务迁移，显著提升预测性能；③系统性揭示上下文可能因分布不匹配而降低性能；④对DUD‑E的严重偏差与泄漏进行批判，倡导时间拆分评估；

**🔧 技术方法**

Feature-wise Linear Modulation (FiLM) 用于条件化分子表示；多层图神经网络作为分子编码器；层次化上下文嵌入（目标、检测、时间）；多任务预测头；训练分为预训练、微调、持续学习三阶段；

**📊 数据集**

ChEMBL 35 作为预训练数据；DUD‑E 用于虚拟筛选评估（10种目标）；TDC 数据用于全局评估；对比基准包括随机森林、GNN‑VS、3D‑CNN等；

**📈 对比分析**

与传统随机森林、GNN‑VS、3D‑CNN 等方法对比，FiLM 模型在DUD‑E上平均AUC 0.850，优于GNN‑VS（0.825）和3D‑CNN（0.830），且在数据稀缺目标上显著超越单目标RF（如CYP3A4 0.686 vs 0.238）；在时间拆分下实现0.843 AUC，显示良好泛化；

**⚠️ 局限性**

L2（检测）和L3（时间）上下文在公开数据中缺乏元数据，未体现显著收益；DUD‑E 基准存在结构偏差与训练/测试泄漏，导致绝对性能指标失真；模型对分布不匹配敏感（如BACE1表现下降）；未与最新分子基础模型（如Uni‑Mol、ChemBERTa）进行对标。

---

## 200. SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning

**arXiv ID:** 2604.06636 | [PDF](https://arxiv.org/pdf/2604.06636v1)

**作者:** Zhengyang Ai `[一作]` (Huawei Taylor Lab), Pinyan Lu `[通讯]` (Huawei Taylor Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SHAPE框架，通过阶段感知的层次化优势函数实现对LLM推理过程的细粒度奖励分配；

**💡 创新点**

创新点在于将潜在可解性（Reasoning Potential）与长度依赖的折扣因子结合，既鼓励低潜能阶段的突破，又抑制冗长推理，同时在token层面使用熵驱动的奖励再分配；

**🔧 技术方法**

采用潜在估计、潜在收益计算、动态折扣、熵驱动的token级重分配以及强化学习中的潜在奖励塑形；

**📊 数据集**

使用了AIME 2024/25、AMC 2023、MinervaMATH、MATH500等数学推理基准数据集；

**📈 对比分析**

与GRPO和MRT等基线对比，SHAPE在五个基准上平均提升约3%准确率，同时平均token消耗下降约30%，在精度和效率上实现新的Pareto前沿；

**⚠️ 局限性**

局限在于仅验证于可验证的数学推理任务，对开放式创作或代码生成等主观评判领域的可推广性仍需进一步研究。

---

## 201. Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability

**arXiv ID:** 2604.06628 | [PDF](https://arxiv.org/pdf/2604.06628v1)

**作者:** Qihan Ren `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongrui Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 200 | [OpenAlex ID](https://openalex.org/A5020653216)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对长链式推理（CoT）监督下的监督微调（SFT）进行系统研究，探究其跨域泛化能力，并考察优化动态、训练数据质量与结构以及模型规模三因素的共同作用。

**💡 创新点**

创新点在于提出跨域泛化是条件属性，揭示短期评估低估效果、优化的“跌跌起伏”模式、数据质量/结构和模型规模共同决定泛化；并首次指出推理收益与安全性不对称，安全性在长CoT SFT 中普遍下降。

**🔧 技术方法**

采用标准 NLL 目标的 SFT（AdamW+余弦学习率），配合长CoT 数据生成与验证、响应长度监测、跨域基准评估；对比长CoT 与无 CoT 训练，系统调参实验。

**📊 数据集**

使用 Math-CoT-20k（Qwen3-32B 生成并经数学验证）、Math-NoCoT、NuminaMath、Countdown-CoT 四种数据集；评估基准包含 MATH500、AIME24、LiveCodeBench v2、GPQA-Diamond、MMLU-Pro、IFEval、AlpacaEval 2.0、HaluEval、TruthfulQA 以及 HEx-PHI。

**📈 对比分析**

通过不同 epoch、学习率、数据质量、模型规模的 ablation 对比；用 cross‑domain 任务的平均分和 pass@k 评估模型性能。结果显示，充分训练后长CoT SFT 在多项 OOD 任务上能超越基线，但在安全性评估中 ASR 下降，说明安全性随推理能力提升而下降。

**⚠️ 局限性**

局限性包括实验仅覆盖有限模型族（主要是 Qwen3 系列和 InternLM），主要聚焦数学推理数据，未对其他任务或 RL 与 SFT 的更细粒度对比进行深入；安全性退化机制尚未彻底解析；实验成本高，需大量算力。

---

## 202. Holistic Optimal Label Selection for Robust Prompt Learning under Partial Labels

**arXiv ID:** 2604.06614 | [PDF](https://arxiv.org/pdf/2604.06614v1)

**作者:** Yaqi Zhao `[一作]` (Shandong University), Yilong Yin `[通讯]` (Shandong University)

**通讯引用:** 5874 | [OpenAlex ID](https://openalex.org/A5100672590)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在部分标签下进行提示学习的全局最优标签选择框架HopS，结合本地密度过滤和全局最优传输两种策略实现标签去模糊

**💡 创新点**

创新点在于将冻结的预训练视觉-语言编码器的泛化能力与两种互补的标签选择方法结合，既利用局部语义一致性也利用全局标签分布一致性，从而显著提升在弱监督下的提示学习性能

**🔧 技术方法**

使用了密度滤波（KNN+多重集频率阈值）和基于熵正则化的最优传输（Sinkhorn算法）来生成伪标签，并在提示学习中联合交叉熵损失进行训练

**📊 数据集**

在八个视觉识别基准数据集（Caltech, DTD, EuroSAT, FGVCAircraft, Food, Flowers, OxfordPets, UCF）上进行实验，并构造了随机和实例依赖的部分标签扰动

**📈 对比分析**

与多种基准（如CC、RC、LWC、MSE、EXP、MAE、SCE、GCE、Papi、CroSel、SoLar）以及全监督CoOp进行对比，HopS在所有扰动水平下均实现SOTA性能，并且在高噪声率下保持稳定优势

**⚠️ 局限性**

局限性主要体现在对候选标签分布与真实标签分布相似的假设较强，对极端噪声或缺失真标签的场景效果仍有限，且对超参数（k、阈值τ、λ等）较为敏感

---

## 203. The Detection--Extraction Gap: Models Know the Answer Before They Can Say It

**arXiv ID:** 2604.06613 | [PDF](https://arxiv.org/pdf/2604.06613v1)

**作者:** Hanyang Wang `[一作]` (University of Chicago), Mingxuan Zhu `[通讯]` (Imperial College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的链式推理(CoT)过程进行黑盒分析，发现大部分后续生成的 tokens 在答案已可恢复后才产生，提出检测–提取差距（Detection–Extraction Gap）。基于此，设计了 Black-box Adaptive Early Exit (BAEE) 策略，利用自由续写进行答案检测与提取，从而大幅减少串行推理时间并提升准确率。

**💡 创新点**

核心创新是揭示并量化了检测–提取差距：模型在内部状态中已编码答案，但强制解码时会因后缀诱发分布位移导致答案无法被提取。通过总变差界定此位移，并基于自由续写的稳定性，提出 BAEE 以逃避此差距，实现高效早停与准确率提升。

**🔧 技术方法**

技术手段包括：① 仅使用 API 调用的三种黑盒探测（Early Forced Answering、Prefix Self‑Consistency、Answer Token Logprob/Entropy Dynamics）；② 通过多样本采样估计自由续写的答案可恢复性；③ 用总变差（TV）理论量化自由与强制续写分布差距；④ 设计多检查点网格和自适应阈值，以实现早停决策。

**📊 数据集**

实验数据集涵盖：MATH‑500（500 道数学题）、GPQA‑Diamond（198 道多步推理题）和 HumanEval（164 道代码生成任务），使用 Qwen3‑32B/8B（Think/NoThink）与 GPT‑OSS‑120B 作为模型。

**📈 对比分析**

与完整 CoT、Naïve Forced Extraction、以及多样本多数投票（SC‑8‑full）等方法对比：BAEE 在所有模型上实现 70–78% 的串行 token 省略，同时提升 1–5 个百分点准确率；在思考模式下可提升 5.8 个百分点，代码生成任务中提升 13.6 个百分点。与全 CoT 相比，BAEE 既保持或提高准确率，又显著降低串行延迟；与 SC‑8‑full 相比，降低 37–61% 的总 token 用量。

**⚠️ 局限性**

局限性包括：① 仅在黑盒 API 场景验证，未探究模型内部状态机制；② 对提示格式、后缀设计等超参数敏感；③ 仅针对三类任务（数学、推理、代码）验证，未知在更广泛领域的泛化；④ 早停策略仍需多轮采样，导致并行计算开销上升；⑤ 对极端难题的处理仍有限，需更细粒度的检查点与阈值调优。

---

## 204. IntervenSim: Intervention-Aware Social Network Simulation for Opinion Dynamics

**arXiv ID:** 2604.06600 | [PDF](https://arxiv.org/pdf/2604.06600v1)

**作者:** Yunyao Zhang `[一作]` (Huazhong University of Science and Technology), Zikai Song `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 559 | [OpenAlex ID](https://openalex.org/A5083665721)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 IntervenSim 框架，用于模拟大规模社交网络事件的演化及源端干预与群体互动的闭环过程。

**💡 创新点**

创新点在于集成源端干预感知、群体自适应表示与意见动态建模的三重闭环机制，并首次实现多阶段干预与意见演化的可解释模拟。

**🔧 技术方法**

采用大语言模型（如 Qwen2.5-7B）结合检索增强生成（RAG）、Chain‑of‑Thought 推理、动态时间折叠、Wasserstein 等技术构建代理交互与评价。

**📊 数据集**

使用 Social Network Benchmark（SNB）数据集，该数据集包含 Twitter、Reddit、Weibo 上的真实事件及 7 天流量记录。

**📈 对比分析**

与 PSP、S3、GA‑S³ 等基线对比，IntervenSim 在 W₁、MAPE、DTW 等指标上分别提升 21.53%、41.6% 与 66.9%，平均仅需 10 个代理，计算成本显著下降。

**⚠️ 局限性**

局限性包括受 LLM 生成误差与推理一致性影响，对干预时序的细粒度建模不足，以及对极端或多模态舆情场景的鲁棒性仍待验证。

---

## 205. Train-Small Deploy-Large: Leveraging Diffusion-Based Multi-Robot Planning

**arXiv ID:** 2604.06598 | [PDF](https://arxiv.org/pdf/2604.06598v1)

**作者:** Siddharth Singh `[一作]` (University of Virginia), Scott Acton `[通讯]` (University of Virginia)

**通讯引用:** 10392 | [OpenAlex ID](https://openalex.org/A5034452294)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于扩散模型的多机器人规划框架MA-DBP，能够在训练阶段仅使用少量机器人，部署时扩展到更多机器人，实现train‑small deploy‑large的目标。

**💡 创新点**

创新点包括：① 采用单一扩散模型配合语义轴向注意力预处理，使模型能处理可变数量机器人；② 通过移动窗口条件扩散和上下文编码（环境图像、起止位置、机器人数）实现动态场景的自适应规划；③ 利用先前的MARL（MAPPO）数据进行预训练，显著降低训练时间；④ 设计多重损失（噪声、边界、时间一致性、碰撞规避）提升轨迹质量。

**🔧 技术方法**

使用的核心技术包括：条件扩散（latent diffusion）、U‑Net去噪网络、语义轴向注意力预处理、FiLM条件机制、学习率调度与梯度裁剪、基于遮罩的可变机器人数训练、经验重放与迁移学习、以及多项式损失函数。

**📊 数据集**

数据集主要来自VMAS（Vectorized Multi-Agent Scenarios）仿真环境；训练数据通过MAPPO政策生成轨迹；测试采用三种二维导航场景（空地图、障碍地图、障碍栏）进行验证。

**📈 对比分析**

与MAPPO、CLF‑QP、MADiff三种基线方法进行比较。结果表明：在空地图和障碍场景中，MA‑DBP的成功率与MAPPO相近且训练时间约为其四分之一；在执行速度上略慢于MAPPO，但仍保持实时性；在所有场景中优于CLF‑QP和MADiff，且在机器人数量扩展时成功率保持稳定。

**⚠️ 局限性**

局限性包括：① 对密集障碍或短规划窗口的环境表现下降；② 对规划窗口长度敏感，需经验选择；③ 随着机器人数增加，碰撞率上升；④ 对极端多机器人或高度动态环境的收敛性仍有挑战；⑤ 依赖先前的MARL预训练，训练数据质量会影响性能。

---

## 206. It's Not About Whom You Train: An Analysis of Corporate Education in Software Engineering

**arXiv ID:** 2604.06580 | [PDF](https://arxiv.org/pdf/2604.06580v1)

**作者:** Rodrigo Siqueira `[一作]` (CESAR School), Danilo Monteiro Ribeiro `[通讯]` (CESAR School)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

探讨了社会人口学与专业变量对软件工程企业培训质量与有效性感知的影响，利用27项感知指标与9个社会人口学变量的交叉分析；

**💡 创新点**

发现个人特征（性别、年龄、学历）与公司规模对感知影响不显著，训练强制性与专业轨迹（经验、岗位、区域）对感知产生局部差异，特别是经验与感知呈非线性低参与区；

**🔧 技术方法**

采用非参数统计检验（Kruskal‑Wallis、Spearman 相关）以及 Dunn 事后检验与 Bonferroni 校正进行差异性与相关性分析；

**📊 数据集**

使用 282 名巴西软件工程专业人士的在线问卷数据，包含27项5点李克特量表感知指标与9个社会人口学变量；

**📈 对比分析**

通过对243 个变量组合的显著性检验与效应量（ε²、r）比较，发现仅 35 个组合显著，其中 24 与培训强制性相关，其他组合基本随机；

**⚠️ 局限性**

局限包括样本为便利抽样、性别与学历偏高、子组样本量小、未区分培训类型、单项测量、未使用客观绩效指标、未全局多重比较校正等因素可能影响结论的普适性与精确性。

---

## 207. Scoring Edit Impact in Grammatical Error Correction via Embedded Association Graphs

**arXiv ID:** 2604.06573 | [PDF](https://arxiv.org/pdf/2604.06573v1)

**作者:** Qiyuan Xiao `[一作]` (East China Normal University), Yunshi Lan `[通讯]` (East China Normal University)

**通讯引用:** 1417 | [OpenAlex ID](https://openalex.org/A5090588589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了在语法错误纠正（GEC）中对系统生成的编辑影响进行自动评分的任务；

**💡 创新点**

创新点在于构建嵌入关联图捕捉编辑间的潜在语义与句法依赖，并以句子流畅度（如困惑度）差值来评估编辑的重要性；

**🔧 技术方法**

技术手段包括Apriori频繁项集挖掘生成伪标签、基于预训练词向量的关联预测分类器、连通组件合并生成编辑组以及基于困惑度的增益评分；

**📊 数据集**

使用四种多语言GEC基准数据集：中文NLPCC18、英文CoNLL14、西班牙语COWSL2H、德语FALKO，并在这些数据上使用GPT‑4o、GECToR、T5等多种模型的输出；

**📈 对比分析**

与随机、Vanilla、Greedy、Displacy等基线比较，本文方法在S_bound与S_rank指标上平均提升约5–10%，在所有语言和系统上均保持领先；

**⚠️ 局限性**

局限性包括需手动设定关联阈值τ和距离阈值δ，且对某些语言特有结构（如德语的可分动词）可能需要额外调整或自适应机制。

---

## 208. A Rolling-Horizon Stochastic Optimization Framework for NBA Franchise Management with Distributionally Robust Risk Constraints

**arXiv ID:** 2604.06548 | [PDF](https://arxiv.org/pdf/2604.06548v1)

**作者:** Siming Zhang `[一作]` (Shanghai University), Jian Zhou `[通讯]` (Shanghai University)

**通讯引用:** 46318 | [OpenAlex ID](https://openalex.org/A5070334439)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一套统一的NBA球队决策体系（NYK-ADMS），将阵容建设、财务规划、媒体策略、联盟扩张与受伤重优化等模块嵌入滚动期望-随机混合整数规划（RH‑SMIP）中，并在其核心层应用分布鲁棒优化（DRO）与CVaR尾部风险约束，案例以纽约尼克斯为实验场。

**💡 创新点**

创新点包括：① 将球队管理视为单一动态控制问题，打破传统“分块”式决策；② 在核心层同时引入DRO与CVaR，兼顾期望价值与极端风险；③ 通过“握手协议”将价值评估与交易执行分离，避免赢家诅咒；④ 在模型中嵌入扩张冲击、媒体权利转型与受伤情景的系统化传递，形成完整的外部冲击响应框架。

**🔧 技术方法**

技术手段：滚动期望-随机混合整数规划（RH‑SMIP）、分布鲁棒优化、CVaR尾部风险预算、近端束（Proximal Bundle）+拉格朗日松弛求解、情景生成与蒙特卡洛稳健性检验、回归与贝叶斯前验估计、敏感性与分区分析。

**📊 数据集**

使用的数据集包括：球员级别的篮球参考（Basketball‑Reference）、NBA‑API、HoopsHype、Spotrac、NBA2K属性、FRED宏观经济指标、Kaggle与公开GitHub存储库的球队、管理层与宏观财务数据，整合后构成多层级的时序与结构化数据库。

**📈 对比分析**

通过与传统单一模块模型（仅阵容或仅财务）以及非鲁棒随机规划的对比，模型在30%+利润提升、CVaR下降70%和破产风险从22%降至2.6%等方面表现出显著优势；在10,000次蒙特卡洛模拟中显示出对宏观波动与伤病冲击的高鲁棒性；同时在不同外部冲击情境（扩张、媒体权利、受伤）下保持稳定的最优策略。

**⚠️ 局限性**

局限性：① 对团队文化、化学效应的量化依赖间接代理，可能导致交易推荐偏差；② 模型依赖于历史数据和预估弹性，若出现结构性变革（新规则、媒体格局剧变）可能失效；③ 计算复杂度高，需大量情景与迭代，实际运营时可能受限；④ 对极端罕见事件的覆盖有限，鲁棒性虽提升但不完全无风险。

---

## 209. TwinLoop: Simulation-in-the-Loop Digital Twins for Online Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.06610 | [PDF](https://arxiv.org/pdf/2604.06610v1)

**作者:** Nan Zhang `[一作]` (Southern University of Science and Technology), Georgios Theodoropoulos `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2921 | [OpenAlex ID](https://openalex.org/A5026385930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了TwinLoop框架，利用数字孪生在运行时加速多智能体强化学习的适配；

**💡 创新点**

创新点在于将数字孪生与在线学习结合，实时触发仿真加速策略改进，避免昂贵的物理环境试错；

**🔧 技术方法**

采用深度双重Dueling DQN进行本地任务卸载决策，数字孪生通过SUMO仿真快速进行何-若分析；

**📊 数据集**

使用仿真生成的车辆道路网络数据（2km×2km网格，16个RSU，45辆车，Poisson任务到达）和多阶段动态环境变化；

**📈 对比分析**

与随机、纯在线、离线和只利用探索结束的基线比较，实验表明TwinLoop在环境切换后显著降低平均和尾部延迟，尤其在高负载阶段表现突出；

**⚠️ 局限性**

局限性包括对全局一致快照的假设、触发频率的手工设定以及在异构场景下的泛化能力不足，需要进一步研究自适应触发机制和多场景训练。

---

## 210. Logical Robots: Declarative Multi-Agent Programming in Logica

**arXiv ID:** 2604.06629 | [PDF](https://arxiv.org/pdf/2604.06629v1)

**作者:** Evgeny Skvortsov `[一作]` (Google LLC), Bertram Ludäscher `[通讯]` (University of Illinois)

**通讯引用:** 7684 | [OpenAlex ID](https://openalex.org/A5057600294)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一个名为Logical Robots的交互式多智能体仿真平台，利用Logica编程语言统一符号规划与低级控制；

**💡 创新点**

通过将大规模传感器数据聚合转化为SQL查询，克服传统ASP/Prolog在大数据场景下的基数化瓶颈，实现感知融合、反应控制和符号规划在同一框架内协同；

**🔧 技术方法**

使用Logica语言（聚合语法）、SQL后端、二维迷宫仿真环境、差速驱动机器人模型以及分布式贝尔曼-福特路径规划算法；

**📊 数据集**

使用自定义生成的迷宫传感器流（雷达、Beacon、墙、机器人信息），未使用公开真实数据集；

**📈 对比分析**

通过演示10个递进难度的场景（如站点管理、编队导航、分布式地图构建）展示功能；未给出量化实验，主要依赖交互演示和SQL聚合的性能优势（相较传统ASP显著提升）；

**⚠️ 局限性**

平台仅限二维模拟，缺乏真实机器人硬件验证；跨机器人内存读取安全和隐私尚未深入；未在大规模真实数据集上进行验证。

---

## 211. WeatherRemover: All-in-one Adverse Weather Removal with Multi-scale Feature Map Compression

**arXiv ID:** 2604.06623 | [PDF](https://arxiv.org/pdf/2604.06623v1)

**作者:** Weikai Qu `[一作]` (Guangdong University of Technology), Ahmed Elazab `[通讯]` (Shenzhen University)

**通讯引用:** 2533 | [OpenAlex ID](https://openalex.org/A5042305251)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 WeatherRemover，单一或多重天气干扰图像去除模型，兼顾恢复质量与运算效率。

**💡 创新点**

创新点：融合线性空间降采样 (Linear SRA) 与卷积通道注意力的多尺度 Transformer，配合 UNet 结构中的双分支门控机制，显著降低参数、推理时间并提升细节恢复。

**🔧 技术方法**

采用 UNet+多尺度 Pyramid Vision Transformer (MS-PVT)、线性 SRA、门控前馈网络 (GFN)、卷积通道注意力、Pseudo‑Huber 损失。

**📊 数据集**

使用 Snow100K、RainDrop、Outdoor‑Rain、All‑Weather、DAWN 等多种单/多天气数据集进行训练与评测。

**📈 对比分析**

与 Restormer、DRSformer、TransWeather、WeatherDiff 等主流方法比较，PSNR 最高、SSIM 同等级；参数约 24M，推理时间 0.1–0.4 s，显著低于同类模型。

**⚠️ 局限性**

局限：相较于 TransWeather，推理速度与计算成本仍较高；在雪、重雾等极端遮挡场景下恢复细节不足；多天气泛化仍落后于单天气最佳表现。

---

## 212. LLM-based Schema-Guided Extraction and Validation of Missing-Person Intelligence from Heterogeneous Data Sources

**arXiv ID:** 2604.06571 | [PDF](https://arxiv.org/pdf/2604.06571v1)

**作者:** Joshua Castillo `[一作]` (Old Dominion University), Ravi Mukkamala `[通讯]` (Old Dominion University)

**通讯引用:** 1521 | [OpenAlex ID](https://openalex.org/A5035065105)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Guardian Parser Pack，一个双路径（规则+LLM）的 schema‑guided 文档解析管道，能够将异构失踪人员案例 PDF 转化为统一结构化记录；

**💡 创新点**

创新点在于：①双路径设计实现了可审计的规则解析与受限的 LLM 提升质量；②采用 schema‑first 验证与 provenance 追踪，保障数据完整性与可追溯性；③共享的文本提取、归一化、地理编码与验证服务，保证两条路径输出可比；

**🔧 技术方法**

使用了多引擎 PDF 文本提取（包含 OCR）、源识别、基于规则的解析、受限的 LLM（Gemini‑2.5‑flash + repair），JSON Schema 验证、检索增强提示、Geo‑coding、Python 生态（spaCy、Pandas、GeoPandas 等）；

**📊 数据集**

数据集来自多源失踪人员案例 PDF：NamUs、NCMEC、Virginia State Police、FBI、Charley Project，覆盖结构化表单、海报式公告和叙事型 OSINT 文档；

**📈 对比分析**

在 75 条手工标注的案例上，LLM 路径的 F1 为 0.8664，规则路径仅为 0.2578；在 517 条记录的批量解析中，LLM 完整字段覆盖率 96.97% 对比 93.23%；但平均运行时从 0.03 s/记录提升至 3.95 s/记录；

**⚠️ 局限性**

局限性包括：文档布局漂移导致规则解析失效、基于文本的地理编码不确定性、LLM 可能过度泛化且修复仅针对 schema 错误而非缺失字段；进一步工作需强化不确定性量化、弱监督标签生成及检索增强提示。

---

## 213. AI-Driven Research for Databases

**arXiv ID:** 2604.06566 | [PDF](https://arxiv.org/pdf/2604.06566v1)

**作者:** Audrey Cheng `[一作]` (University of California Berkeley), Ion Stoica `[通讯]` (University of California Berkeley)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了在 AI-Driven Research for Systems (ADRS) 框架中共进化评估器与算法的自动化流程，以消除数据库性能优化中的评估瓶颈；

**💡 创新点**

创新点在于：①将评估器视为可演化组件，与解法生成并行演化；②利用 LLM 自动设计模拟器、性能模型、工作负载选择与搜索空间修剪；③通过三大经典数据库问题（缓冲区管理、索引选择、查询重写）验证方法的有效性；

**🔧 技术方法**

技术核心包括大语言模型（GPT‑5/Claude Sonnet）、MAP‑Elites 等进化算法、Python/Java 编写的评估与生成脚本，以及自动化的实验调度与数据收集；

**📊 数据集**

使用的数据集主要为工业级数据库基准：TPC‑H、TPC‑DS、DSB（含多种查询模板和规模因子）以及 PostgreSQL 14/17 运行环境；

**📈 对比分析**

与现有最优基线相比，实验结果显示：缓冲区管理策略提升缓存命中率 19.8% 与 I/O 卷积 11.4%；索引选择策略查询延迟下降 6.3%，选择时间缩短 2.2×；查询重写策略在 TPC‑H、DSB 上分别实现 5.4× 与 6.8× 的整体速度提升；

**⚠️ 局限性**

局限性包括：评估器的演化仍依赖 LLM 的生成质量；方法对其他数据库子系统（如并发控制、日志）尚未验证；可能出现过拟合、对特定硬件/工作负载敏感；缺乏形式化正确性检查，需人工监督

---

## 214. CCD-CBT: Multi-Agent Therapeutic Interaction for CBT Guided by Cognitive Conceptualization Diagram

**arXiv ID:** 2604.06551 | [PDF](https://arxiv.org/pdf/2604.06551v1)

**作者:** Chang Liu `[一作]` (Lanzhou University), Minqiang Yang `[通讯]` (Lanzhou University)

**通讯引用:** 885 | [OpenAlex ID](https://openalex.org/A5079527887)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了多智能体框架 CCD‑CBT，用动态重建的认知概念化图（CCD）来生成多轮 CBT 对话。

**💡 创新点**

创新点在于将 CCD 从静态固定模型转为实时动态重建，并在模拟过程中实现信息不对称的多智能体交互。

**🔧 技术方法**

技术手段包括基于 GPT‑4o 的 Client、Therapist 与 Control 三智能体、低秩适配（LoRA）微调，以及自动化评估指标 CTRS 与 PANAS。

**📊 数据集**

使用了自建的 CCDChat 数据集，包含 4,500 条多轮 CBT 对话，基于 7,500 个中文 CCD 与不同态度的客户仿真。

**📈 对比分析**

通过自动化 CTRS 分数与 PANAS 情绪变化，并与 SoulChat、PsyChat、CAMEL 等基线进行比较，CCD‑CBT 在大多数 CBT 维度与情绪提升上均显著优于基线。

**⚠️ 局限性**

局限性包括仅模拟单一长对话而非多场次 CBT、缺乏非语言线索、数据以中文为主可能导致跨文化推广受限。

---

## 215. Variational Feature Compression for Model-Specific Representations

**arXiv ID:** 2604.06644 | [PDF](https://arxiv.org/pdf/2604.06644v1)

**作者:** Zinan Guo `[一作]` (University of Queensland), Guangdong Bai `[通讯]` (City University of Hong Kong)

**通讯引用:** 1855 | [OpenAlex ID](https://openalex.org/A5015858067)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于变分潜在瓶颈与动态二进制掩码的特征压缩框架，能够在保持指定目标分类器性能的同时大幅降低其它模型对处理后输入的利用率，解决输入再利用的隐私威胁。

**💡 创新点**

创新点在于：①使用变分信息瓶颈（VIB）构建无像素重构损失的任务驱动解码器，专注于保留对目标任务的相关信息；②引入结合KL散度与梯度显著性得分的动态掩码，进一步抑制可被其他模型利用的潜在维度；③在白盒训练阶段利用目标模型梯度，仅在推理阶段保持单向前向传播，兼顾可部署性与隐私效果。

**🔧 技术方法**

技术包括：ResNet‑18 编码器、全连接层输出均值与方差、变分采样、VIB 目标（交叉熵+KL 正则）、梯度显著性计算、二进制掩码阈值化、解码器（FC + 反卷积上采样）、Adam 优化、频率化的全局掩码更新。

**📊 数据集**

主要使用数据集：CIFAR‑100（100类）、CIFAR‑10（10类）、Tiny ImageNet（200类）以及 Pascal VOC 2012（多标签），其中 CIFAR‑100 作为基准实验，其它数据集用于跨任务和跨标签的探索性验证。

**📈 对比分析**

与四种预训练目标模型（ResNet‑152、DenseNet‑121、ConvNeXt‑V2、VGG‑16）进行对角线与非对角线性能对比。结果显示：目标模型 top‑1 准确率可达 70–72%（比原始 85% 低约10%），而所有未被训练的模型在同一输入上的准确率均降至约 1%（接近随机），抑制比例超过 45×。在跨任务的风格迁移实验中，目标模型保持 60% 以上的精度，非目标模型（风格网络）失效，说明可迁移性被显著降低。

**⚠️ 局限性**

局限性包括：①仅针对固定的白盒目标模型，需重新训练才能更换目标；②未评估自适应攻击者能否在处理后数据上训练新的分类器；③实验仅覆盖图像分类与风格迁移，未验证对其他模态或更复杂任务的通用性；④缺乏与现有隐私技术（如 DPFE、差分隐私）的直接对比，难以量化相对优势。

---

## 216. Frozen-Tag-Based Physical-Layer Authentication Against User Interference

**arXiv ID:** 2604.06641 | [PDF](https://arxiv.org/pdf/2604.06641v1)

**作者:** Lei Yao `[一作]`, Ning Xie `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于冻结标签的物理层身份认证框架，能够在多用户干扰环境下实现鲁棒的认证，并通过将原始标签隐藏在极化码的冻结位中实现对窃听的抵抗。

**💡 创新点**

创新点包括：1）将原始标签编码为冻结标签并随机插入消息；2）利用极化码将标签信息分布在冻结位与信息位之间，既提升了对干扰的抗性，又增加了攻击者解码的难度；3）配合稀疏索引提取与冻结/解冻模块，实现高兼容性。

**🔧 技术方法**

主要技术手段为：极化编码与SC/SCL解码、密钥驱动的稀疏索引提取、冻结标签生成 (FTG) 与解冻标签调和 (TTR) 模块、BPSK 调制与 AWGN 通道模型。

**📊 数据集**

没有使用公开数据集，所有结果均基于 10⁵ 次 Monte‑Carlo 仿真得到。

**📈 对比分析**

与传统无编码标签的 PLA 方案进行对比，评价指标包括检测概率、误码率、Eve 对标签位置的识别误差概率以及对原始标签的噪声累积。结果显示：
- 在相同信噪比下，冻结标签方案的检测概率提升 3–6 dB；
- Eve 对标签位置的正确识别概率远低于 1%；
- Eve 在估计原始标签时面临显著噪声累积，误码率远高于接收机噪声；
- 兼容性方面，随着消息长度增长，冻结标签对无意识接收机的误码率影响可控。

**⚠️ 局限性**

局限性包括：1）相对于传统方案，额外的极化码编码/解码导致计算复杂度略高；2）需精确设置冻结标签插入比例，过大会影响信息传输误码率；3）方案对极化码参数（如长度、率）敏感；4）在极端多用户干扰或严重信道失配情况下，剩余干扰仍可能影响认证性能。

---

## 217. SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport

**arXiv ID:** 2604.06631 | [PDF](https://arxiv.org/pdf/2604.06631v1)

**作者:** Zheng Jiang `[一作]` (Tsinghua University), Lifeng Sun `[通讯]` (Tsinghua University)

**通讯引用:** 3368 | [OpenAlex ID](https://openalex.org/A5047712495)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 SubFLOT 框架，实现服务器端个性化联邦剪枝，并通过参数空间与特征空间的对齐来缓解异构性问题。

**💡 创新点**

创新点在于：1) 将历史客户端模型视为数据分布代理，利用 Optimal Transport（OT）进行服务器端剪枝与聚合；2) 设计 Scaling-based Adaptive Regularization（SAR）根据剪枝率自适应抑制参数漂移；3) 将 OTP 与 OTA 统一为层级 OT 方案，兼顾模型稀疏与参数一致性。

**🔧 技术方法**

核心技术包括：Optimal Transport（Wasserstein 距离优化）、层级层次匹配、梯度正则化（SAR）、FedAvg 变体 OTA 以及深度神经网络剪枝。

**📊 数据集**

在多种数据集上验证：CIFAR-10/100、TinyImageNet、Digit5、PACS、AG News、HAR 等，覆盖 CV、NLP 与 IoT 场景。

**📈 对比分析**

与 HeteroFL、FedRolex、FedDrop、FedMP、ScaleFL、Flado、DepthFL、FedDSE、AdaptiveFL、FlexFL 等 9 种主流剪枝方法对比，SubFLOT 在标签偏移、特征迁移、实际场景等多种设定下平均提升 3–10%（如 CIFAR-10 在标签偏移下从 84.54% 提升至 86.89%）。

**⚠️ 局限性**

局限性包括：1) OT 计算仍需一定服务器资源，尤其对大模型层数高时；2) 需要存储并维护客户端历史模型，若历史模型分布不充分可能导致对齐误差；3) 对极端高稀疏率时仍有性能衰减；4) 依赖均衡的客户端参与与同步，异步或落线客户端需进一步改进。

---

## 218. DiffuMask: Diffusion Language Model for Token-level Prompt Pruning

**arXiv ID:** 2604.06627 | [PDF](https://arxiv.org/pdf/2604.06627v1)

**作者:** Caleb Zheng `[一作]` (University of Washington), Dan Roth `[通讯]` (Oracle AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DiffuMask，一种基于扩散语言模型的并行提示压缩框架，能够预测令牌保留掩码并实现高效的提示裁剪。

**💡 创新点**

创新点在于将扩散过程逆向用于令牌保留决策，采用一次性多令牌掩码预测，显著加速裁剪并保持推理语境的完整性。

**🔧 技术方法**

技术包括扩散式掩码预测、二进制交叉熵损失与抗掩码惩罚、top‑k 与阈值控制的掩码决策，以及多步去噪推理流程。

**📊 数据集**

使用 GSM8K 代数推理基准构建全-裁剪提示数据集，并在 Yelp‑5、Yahoo Answers 与 AG’s News 等分类任务上进行跨域评估。

**📈 对比分析**

与完整提示、CoT‑Influx、HP、LLMLingua 等基线相比，DiffuMask 能在 1 分钟内完成 80% 的令牌压缩，同时保持或提升 30‑70% 的准确率，跨模型迁移亦能维持高精度。

**⚠️ 局限性**

局限性包括：构造全-裁剪数据集耗时 10–48 小时、推理仍需 64 步且未实现实时化、对教师裁剪策略的偏倚以及仅在推理/分类任务上验证，尚需扩展到对话、摘要、多模态等领域。

---

## 219. Balancing Efficiency and Restoration: Lightweight Mamba-Based Model for CT Metal Artifact Reduction

**arXiv ID:** 2604.06622 | [PDF](https://arxiv.org/pdf/2604.06622v1)

**作者:** Weikai Qu `[一作]` (New York University), Changmiao Wang `[通讯]` (Shan Dong Xiehe College)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种轻量级的 MARMamba 模型，用于 CT 金属伪影去除，直接处理影像域，无需 sinogram 或掩码输入。

**💡 创新点**

创新点包括：1）多尺度 Mamba（MS‑Mamba）核心模块；2）Flip Mamba Block（FMB）利用多方向翻转捕获全局上下文；3）Average‑Maximum Feed‑Forward Network（AMFN）融合最大池化与平均池化特征；4）结合 Pseudo‑Huber 与 LPIPS 的复合损失，兼顾像素级和感知质量；5）整体参数极低，仅 0.59 M。

**🔧 技术方法**

使用 UNet‑like 结构、Mamba 变体（FMB/AMFN）、自注意力兼容的多分支设计、Pseudo‑Huber 与 LPIPS 损失、Adam + CosineAnnealingLR 训练策略。

**📊 数据集**

主要数据集为 SynDeepLesion（训练 1 000 对，测试 200 对）和 CLINIC‑metal（真实临床金属病例），并在合成与真实场景上进行评估。

**📈 对比分析**

与 13 余种物理、深度学习、扩散、Transformer 方案对比；在大型/中型/小型/微型金属伪影上均取得最高 PSNR/SSIM、最低 RMSE/LPIPS，并保持最小参数量和合理推理时间。

**⚠️ 局限性**

局限性：1）低对比度区域恢复仍不完美；2）训练数据主要为合成，导致真实泛化受限；3）推理速度略高于部分 Transformer；4）未处理 3D 体积伪影，需进一步扩展。

---

## 220. PoC-Adapt: Semantic-Aware Automated Vulnerability Reproduction with LLM Multi-Agents and Reinforcement Learning-Driven Adaptive Policy

**arXiv ID:** 2604.06618 | [PDF](https://arxiv.org/pdf/2604.06618v1)

**作者:** Phan The Duy `[一作]`, Van-Hau Pham `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PoC-Adapt 框架，自动化生成并验证漏洞 Proof‑of‑Concept (PoC) ，通过多智能体 LLM 组合完成从上下文获取到环境搭建、攻击生成、语义验证的全流程。

**💡 创新点**

核心创新点为：① 语义状态差异验证 Oracle，利用预后状态对比精准判定 PoC 成功；② 基于离线 Double‑Deep‑Q‑Network 的自适应策略学习，减少试错次数并抑制 token 泌出。

**🔧 技术方法**

技术手段包括：多智能体 ReAct+LangGraph/Chain 交互、Docker 隔离环境、Semantic Oracle 结构化状态采样、DDQN 强化学习、LLM（Gemini‑2.5‑Pro 等）工具调用。

**📊 数据集**

使用的数据集有：FL‑Bench‑100（Java/PrimeVul 混合 100 个 CVE）、GHSA‑Real80（80 个 GitHub 安全公告）、CWE‑Bench‑Java 与 PrimeVul 作为训练与验证源。

**📈 对比分析**

对比方法：与 FaultLine 等基线在 FL‑Bench‑100 上同等预算下对比，PoC‑Adapt 成功率提升 25%（15% vs. 12%）、TTE 减半、EE 加倍；在 GHSA‑Real80 上成功率 15%，平均成本 $0.42/次；RL 子系统相对随机提升 16.7% 的成功率并显著减少步骤。

**⚠️ 局限性**

局限性包括：环境重现（Planner 阶段）仍占主导瓶颈；语义 Oracle 对细粒度、间接影响的漏洞检测弱；RL 训练数据量有限、模型对新漏洞泛化受限；固定的 B=3 迭代预算可能过早终止潜在可行 PoC；总体成本仍高，依赖大模型。

---

## 221. CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data

**arXiv ID:** 2604.06616 | [PDF](https://arxiv.org/pdf/2604.06616v1)

**作者:** Mingyu Yang `[一作]` (HKUST (GZ) and HKUST), Wei Wang `[通讯]` (HKUST (GZ) and HKUST)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为CubeGraph的层次网格+动态图拼接框架，用于高维向量检索与任意空间时间过滤的融合查询

**💡 创新点**

创新点在于：①在多层网格中为每个立方体构建局部图索引；②通过跨立方体边动态拼接，恢复全局连通性，避免子索引碎片化；③基于过滤器的特征长度自动选择层级，保证所需合并图的数量为常数

**🔧 技术方法**

核心技术包括：多层网格划分、局部HNSW图构建、跨立方体邻接边生成、动态/预先拼接搜索策略、过滤器适配（矩形、多边形、圆形、组合）

**📊 数据集**

实验使用了四大公开数据集：SIFT1M（128维）、YFCC（512维+地理+时间）、MSMARC10M（1024维）以及Deep100M（96维），并在合成的二维至四维空间属性上进行测试

**📈 对比分析**

与基线（HNSW+后过滤、-γ等）相比，CubeGraph在多种过滤器形状和比例下实现了1–2个数量级的吞吐量提升，同时保持90%+召回率；在100M数据集上实现99.5%召回，约250 Qps；最优情况下比最强基线提升近10倍

**⚠️ 局限性**

局限性包括：①需要构建多层网格和跨立方体边，构建时间相对传统单图略高；②对极高维空间或极稀疏分布的过滤器可能导致跨立方体边不足；③在过滤器选择性极低时，仍需搜索较多节点，性能下降

---

## 222. CoverAssert: Iterative LLM Assertion Generation Driven by Functional Coverage via Syntax-Semantic Representations

**arXiv ID:** 2604.06607 | [PDF](https://arxiv.org/pdf/2604.06607v1)

**作者:** Yonghao Wang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Huawei Li `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CoverAssert框架，利用功能覆盖反馈迭代提升LLM生成SystemVerilog Assertions（SVA）的有效覆盖率。

**💡 创新点**

首次将语法-语义融合的功能覆盖反馈机制引入LLM Assertion生成；通过语义嵌入与AST结构联合匹配，精确识别规范中的未覆盖功能点；实现可与现有方法无缝集成的轻量级迭代流程。

**🔧 技术方法**

采用ChatGPT‑4o提取意图、Qwen3‑Embedding生成语义向量，AST路径向量捕获结构信息，聚类+PCA融合编码，覆盖驱动反馈循环；整体依赖LLM与传统结构化分析技术。

**📊 数据集**

四个开源硬件设计（I2C、SHA3、ECG、Pairing）以及对应的RTL文件，使用Cadence JasperGold进行功能覆盖验证。

**📈 对比分析**

与AssertLLM和Spec2Assertion比较，迭代后分支覆盖提升约9.5%/9.6%，语句覆盖提升约9.6%/9.7%，切换覆盖提升约15.7%/16%，多项指标均接近或达到100%。

**⚠️ 局限性**

仍受LLM生成质量限制；覆盖阈值设定固定，可能无法捕捉所有关键点；在极大规模设计中的可扩展性和运行时成本尚未充分验证。

---

## 223. Can Drift-Adaptive Malware Detectors Be Made Robust? Attacks and Defenses Under White-Box and Black-Box Threats

**arXiv ID:** 2604.06599 | [PDF](https://arxiv.org/pdf/2604.06599v1)

**作者:** Adrian Shuai Li `[一作]` (Purdue University), Elisa Bertino `[通讯]` (Purdue University)

**通讯引用:** 39838 | [OpenAlex ID](https://openalex.org/A5061694501)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `f7dab867-23a8-4241-85e9-4ba79c6402f9`

**🎯 论文内容**

研究了在概念漂移条件下的恶意软件检测器的对抗鲁棒性，提出了通用鲁棒化框架并在PGD与MalGuise攻击下进行评估

**💡 创新点**

首次在漂移自适应检测器上系统探究对抗鲁棒性，提出与攻击类型无关的鲁棒化框架，并揭示源域对抗训练对PGD和MalGuise的不同影响

**🔧 技术方法**

使用AdvDA漂移自适应模型，DART对抗训练（PGD），MalGuise二进制层面攻击，以及多种实验防御配置

**📊 数据集**

使用跨月的MB-24+ Windows恶意软件数据集（含16,000个正常PE文件）

**📈 对比分析**

通过在五个适应窗口、三种FPR阈值下对ASR、TPR和训练时间进行系统比较；DART可将PGD攻击成功率降至≈3%，MalGuise防御将攻击成功率降至≈5%，但源对抗训练对MalGuise无效且成本极高

**⚠️ 局限性**

鲁棒性不互通（PGD防御不抵御MalGuise，反之亦然）；源对抗训练仅对PGD有效；未探讨多视角联合防御，实验仅覆盖特定攻击与数据

---

## 224. SonicDB S6: A Storage-Efficient Verkle Trie for High-Throughput Blockchains

**arXiv ID:** 2604.06579 | [PDF](https://arxiv.org/pdf/2604.06579v1)

**作者:** Luigi Crisci `[一作]` (Sonic Labs), Bernhard Scholz `[通讯]` (Sonic Labs)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

实现了 SonicDB S6，一套针对 Sonic 区块链的 Rust 版 Verkle Trie 状态数据库，专门针对 300 ms 快速区块时间和实时存档查询进行了优化。

**💡 创新点**

创新点包括：①基于占用率的节点特殊化并通过 O(kn²) 动态规划精确选择，显著降低活状态存储；②delta 节点仅记录差异槽，避免复制，压缩归档数据库；③批量更新、宽度优先遍历和工作窃取式多线程承诺重算，以及 Pedersen 中间状态缓存，满足极低延迟需求。

**🔧 技术方法**

技术栈涵盖 Verkle Trie、Bandersnatch 曲线上的 Pedersen 组 commitments、Inner Product Argument 聚合、动态规划、delta 存储、批量更新、宽度优先锁协议、工作窃取并行、窗口化多标量乘法、Rust 低层文件 I/O 与缓存。

**📊 数据集**

使用 Sonic 区块链历史数据：前 55 M（或 40 M）块的完整状态，涵盖 LiveDB 与 ArchiveDB 的实际键值分布与变更频率。

**📈 对比分析**

对比方法：将 SonicDB 与 Geth Verkle（LevelDB 后端）在同一硬件（AMD Ryzen 5 3600、64 GB RAM）上运行，测量磁盘占用与 gas/s 处理速率。SonicDB LiveDB 22 GiB、ArchiveDB 809 GiB，分别比 Geth 的 25.6 GiB 与 16 141 GiB 降低 98% 与 95%；吞吐量提升 2.85×（约 2.85 倍的 Mgas/s），ArchiveDB 与 LiveDB 差距仅 24%，可满足 300 ms 区块窗口。

**⚠️ 局限性**

局限性：①当节点特殊化数量增多时会产生显著性能开销；②方案依赖 Sonic 的无分叉特性，无法直接应用于需要分叉/回滚的链；③实现主要集中在存储与承诺计算，尚未深入评估对 EVM 其他模块的影响；④多线程与批量更新的调优仍需在更大规模或更高负载下进一步验证。

---

## 225. To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs

**arXiv ID:** 2604.06552 | [PDF](https://arxiv.org/pdf/2604.06552v1)

**作者:** Zohaib Khan `[一作]` (Fatima Fellowship), Tarek Naous `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 725 | [OpenAlex ID](https://openalex.org/A5037533418)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型在不同语言和国家背景下生成误导信息的行为，并创建了覆盖8种语言、195个国家、6,867个实体的全并行误导信息生成数据集 GlobAllies。

**💡 创新点**

①构建了覆盖8种语言、195个国家、6,867个实体的全并行误导信息模板数据集；②揭示LLM生成误导信息受目标国家HDI和提示语言影响；③评估现有安全守护和检索增强事实核查的效果。

**🔧 技术方法**

采用大语言模型（GPT‑4o、Llama‑3.3‑70B）进行文本生成，利用人工标注与LLM‑as‑judge评估合规性，使用安全分类器（Llama‑Guard）和检索增强生成（RAG）进行防护实验。

**📊 数据集**

GlobAllies（440条误导信息提示模板+6,867实体）以及对应的事实性提示集；同时使用公开事实核查来源、Wikidata、多语言知识库以及Web检索API（Tavily）等。

**📈 对比分析**

通过人工标注和LLM判定的准确率（约90%）评估生成合规率；发现低资源语言和低HDI国家的误导生成率显著高于西方国家；安全守护器在低资源语种表现差异大；RAG在降低误导率方面有效但对事实内容过于保守。

**⚠️ 局限性**

仅关注文本新闻文章，未覆盖多模态误导；模板偶尔可能生成真值；安全守护器和RAG对语言资源不均衡的适配不足，无法完全覆盖全球多样性。

---

## 226. Meaningful Human Command: Towards a New Model for Military Human-Robot Interaction

**arXiv ID:** 2604.06611 | [PDF](https://arxiv.org/pdf/2604.06611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 227. SkillSieve: A Hierarchical Triage Framework for Detecting Malicious AI Agent Skills

**arXiv ID:** 2604.06550 | [PDF](https://arxiv.org/pdf/2604.06550v1)

**作者:** Yinghan Hou `[一作]` (Imperial College London), Zongyou Yang `[通讯]` (University College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 SkillSieve，一个三层级的恶意 AI 代理技能检测框架，能够在零成本的静态预筛选后利用分层 LLM 分析和多模型陪审团进一步识别恶意技能。

**💡 创新点**

创新点包括：① 基于成本与召回的分层三层架构；② 将 LLM 评估拆分为四个结构化子任务（意图一致、权限合理、隐蔽行为、跨文件一致性）以提高鲁棒性；③ 采用多模型陪审团与结构化辩论来消除单一模型的偏差；④ 开源完整的评测基准。

**🔧 技术方法**

技术手段包括：正则和 AST 静态扫描、元数据信誉检查与 XGBoost 评分器、三模 LLM 结构化提示与权重聚合、以及多模型并行调用与辩论机制。

**📊 数据集**

使用了 49,592 个来自 ClawHub 的真实技能包，构成的 400 例标注子集（89 例恶意），并生成了 100 个覆盖五种规避技术的对抗样本。

**📈 对比分析**

在 400 例标注集上，SkillSieve 的两层管线实现 0.800 的 F1（0.752 精度，0.854 召回），显著优于 ClawVet 的 0.421 F1；在完整 49,592 例数据上，平均每个技能仅需 38.8 ms 进行静态筛选，整体平均成本约 0.006 USD/技能。

**⚠️ 局限性**

局限性包括：对运行时下载或时间延迟的攻击难以检测；LLM 结果存在随机性且需外部 API；静态分析无法捕获动态或网络拉取的恶意代码；模型训练数据多为单一攻击者导致泛化受限。

---

## 228. Sparsity-Aware Roofline Models for Sparse Matrix-Matrix Multiplication

**arXiv ID:** 2604.06637 | [PDF](https://arxiv.org/pdf/2604.06637v1)

**作者:** Matthew Qian `[一作]` (Texas A&M University), Ariful Azad `[通讯]` (Texas A&M University)

**通讯引用:** 1815 | [OpenAlex ID](https://openalex.org/A5013984574)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了稀疏矩阵‑稠密矩阵乘法（SpMM）的性能，提出并验证了针对不同稀疏结构的稀疏度感知屋顶线模型，旨在准确预测不同实现的可达性能。

**💡 创新点**

创新点在于将稀疏结构细分为随机、对角、块和无尺度四类，分别推导算术强度（AI）与内存流量模型，证明单一屋顶线模型无法覆盖所有情况，并给出对应的性能上限。

**🔧 技术方法**

采用屋顶线理论、CSR/CSB/MKL实现、AMD EPYC Perlmutter平台测试、SuiteSparse和Erdős–Rényi生成的稀疏矩阵、算子计数与AI推导等技术手段。

**📊 数据集**

使用SuiteSparse中的road_usa、hugebubbles-00010、asia_osm、333SP、com-Orkut、com-LiveJournal、uk-2002、raja31、ideal_diagonal_22等矩阵，以及随机生成的er_22_1/10/20，构成块、无尺度、对角、随机四类稀疏结构。

**📈 对比分析**

在Perlmutter 64线程下，对CSR、CSB、MKL三种实现分别测量d=1,4,16,64时的GFLOP/s，并与对应稀疏度感知屋顶线进行对比；结果显示无尺度矩阵性能最高、随机最低，CSB在块/无尺度结构下可逼近理论上限，而CSR/MKL在随机结构下显著低于屋顶线。

**⚠️ 局限性**

局限性包括对稀疏结构做了简化假设、未能完整捕获缓存行为和内存延迟影响，以及屋顶线模型本身对高度不规则的稀疏算子可能不足，导致预测与实际存在一定差距。

---

## 229. The Theorems of Dr. David Blackwell and Their Contributions to Artificial Intelligence

**arXiv ID:** 2604.06621 | [PDF](https://arxiv.org/pdf/2604.06621v1)

**作者:** Napoleon Paxton `[一作]` `[通讯]` (University of California, Berkeley), Napoleon Paxton (University of California, Berkeley)

**关键词:** `aaff19cd-e89f-4398-8dae-a6684a329811` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对David Blackwell三大定理（Rao‑Blackwell、Approachability、Informativeness）进行系统综述，阐述它们在现代人工智能中的应用与影响，并提出统一的信息理论框架。

**💡 创新点**

创新点在于：①将这三大定理整合为一个统一的框架，展示它们在信息压缩、决策制定、信息评估三方面的互补关系；②以此框架为基础，系统梳理并连接了MCMC、SLAM、生成模型训练、RLHF、在线学习、公平学习、信息设计、AI安全等多领域的最新研究与实践；③指出并讨论了目前尚未解决的前沿问题（如非凸目标的Approachability、LLM表示的Blackwell序等）。

**🔧 技术方法**

主要技术手段为文献综述、理论分析与案例映射。作者通过对经典论文的重新解释，并结合现代算法实例，展示了三大定理在不同 AI 子领域中的具体实现方式。

**📊 数据集**

由于本文为综述性质，未使用新的实验数据集，而是引用了已有研究中的数据集与实验结果（如SLAM系统 GMapping、MCMC 在贝叶斯网络中的应用、RLHF 训练日志等）。

**📈 对比分析**

比较方法采用文献对比与案例映射：对比各定理在不同 AI 场景下的理论贡献与实际收益（如 Rao‑Blackwell 在 MCMC 变方减少、RBPF 在 SLAM 低方差估计、Approachability 在无后悔学习与多目标 RLHF 中的收敛性）。性能评估主要基于已有研究报告的指标，未给出统一实验结果。

**⚠️ 局限性**

局限性：①综述性质，缺乏统一的实验验证与量化比较；②对某些前沿问题（如非凸目标的Approachability、LLM 表示的 Blackwell 顺序、扩散模型的 Rao‑Blackwell 化等）仍停留在理论与开放问题层面；③对不同领域应用的细节深度有限，未能提供完整的实现细节或代码。

---

## 230. BiDexGrasp: Coordinated Bimanual Dexterous Grasps across Object Geometries and Sizes

**arXiv ID:** 2604.06589 | [PDF](https://arxiv.org/pdf/2604.06589v1)

**作者:** Mu Lin `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 22139 | [OpenAlex ID](https://openalex.org/A5108050904)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了包含6351个多尺寸、多几何形状物体的9.7M双手抓取数据集BiDexGrasp，并提出了基于区域约束初始化与解耦力闭合优化的抓取生成管线以及双手协调与几何尺度自适应的抓取生成框架。

**💡 创新点**

①通过区域约束初始化与解耦力闭合优化显著提升抓取采样效率与成功率；②引入双手协调模块与尺度自适应抓取锚点，解决物体尺寸与几何多样性导致的动作空间膨胀问题；③采用相对姿态扩散模型实现高质量、物理可行的双手抓取。

**🔧 技术方法**

使用抓取力矩空间（GWS）进行区域选择、QP能量解耦优化、逆运动学可达性验证、点云特征提取（PointNet++）、多视角协调预测、相对姿态DDPM扩散生成等技术。

**📊 数据集**

从Objaverse和DexGraspNet两大公开数据集筛选6351个物体，并对每个物体进行30–80 cm尺度化，生成约9.7M双手抓取示例。

**📈 对比分析**

与公开的BimanGrasp、SceneDiffuser、DGTR等方法在仿真（MuJoCo）与真实机器人平台上对比，BiDexGrasp的抓取成功率提升至66–77%，相对Baseline提升约2–3倍，抓取速度提升40×，穿透率与自碰撞率显著降低。

**⚠️ 局限性**

抓取多样性略低（PCA第一主成分方差），对极大尺寸或形状复杂物体仍可能出现碰撞/滑动失败；模型对仿真参数与物体重建误差敏感，且在多机器人系统与实时规划中的鲁棒性尚待进一步验证。

---

## 231. LiftFormer: Lifting and Frame Theory Based Monocular Depth Estimation Using Depth and Edge Oriented Subspace Representation

**arXiv ID:** 2604.06576 | [PDF](https://arxiv.org/pdf/2604.06576v1)

**作者:** Shuai Li `[一作]` (Shandong University), Tian Xie `[通讯]` (Zhejiang Lab)

**通讯引用:** 25277 | [OpenAlex ID](https://openalex.org/A5055015997)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为LiftFormer的单目深度估计模型，利用提升理论和框架理论构建深度导向几何子空间（DGR）和边缘感知子空间（ER），通过这两个子空间的投影实现对图像空间特征到深度特征的转换，并最终生成更平滑、边缘更清晰的深度图。

**💡 创新点**

创新点在于：①将离散的深度分箱预测转化为连续的深度特征生成，利用提升理论将深度预测问题映射到覆盖空间；②使用框架理论构造冗余且鲁棒的DGR子空间，使图像特征能够在不同尺度下统一映射到深度空间；③引入ER子空间通过边缘投影增强局部高频信息，从而显著降低边缘误差；④在传统AdaBin框架上实现上述子空间投影，理论上验证了类似“bin embedding”的使用。

**🔧 技术方法**

核心技术包括：提升理论、框架理论、Swin Transformer编码器、DGR子空间投影（SF-DGR）、ER子空间投影（DF-ER）、多尺度特征融合、DYReLU激活、AdaBins式深度预测、SILog损失等。

**📊 数据集**

实验数据集主要为：KITTI（Eigen split，26k训练/697测试，最大深度80m）和NYU Depth V2（训练50k/测试654，最大深度10m）。

**📈 对比分析**

在KITTI上，LiftFormer以RMSE 2.038、AbsRel 0.143、SqRel 0.076、ζ₁ 0.978、ζ₂ 0.998的指标击败了PixelFormer、BinsFormer、iDisc等最新方法；在NYU上取得RMSE 0.313、AbsRel 0.038、SqRel 0.932、ζ₁ 0.991、ζ₂ 0.998，优于PixelFormer、DepthFormer、LocalBins等；在0–50m范围内，RMSE下降7.27%，相对改进显著。Ablation实验进一步证明SF‑DGR和DF‑ER模块的有效性。

**⚠️ 局限性**

限制与不足：①模型依赖较高的计算资源，训练时间较长；②DGR和ER子空间的构造虽然理论上可推广，但目前仅在KITTI和NYU上验证，跨域适应性未作深入探究；③边缘投影模块虽然无监督，但对不同场景的边缘定义可能不够稳健，仍可能出现极端边缘误差；④多余的冗余向量在极高维时可能导致模型复杂度上升。

---

## 232. Polylab: A MATLAB Toolbox for Multivariate Polynomial Modeling

**arXiv ID:** 2604.06575 | [PDF](https://arxiv.org/pdf/2604.06575v1)

**作者:** Yi-Shuai Niu `[一作]` (Beijing Institute of Mathematical Sciences and Applications), Shing-Tung Yau `[通讯]` (Tsinghua University)

**通讯引用:** 27599 | [OpenAlex ID](https://openalex.org/A5006865040)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个统一的 MATLAB 工具箱 Polylab，用于多元多项式标量和矩阵的符号-数值混合计算，支持 CPU、GPU 基础实现和高性能 GPU 后端，并提供构造、简化、求导、Jacobian/Hessian、LaTeX 导出、后端切换以及与 YALMIP、SOSTOOLS 的互操作。

**💡 创新点**

创新点包括：① 引入显式变量标识和命名机制，消除位置对齐导致的表达式错误；② 在 3.1 版本加入精度控制和基于 log‑det 的仿射正则化方向（Affine‑Normal）计算；③ 在 GPU 后端实现了高效的简化、乘法和稀疏矩阵求解路径，支持自动微分、矩阵自由 Exact 和 Stochastic 路径。

**🔧 技术方法**

使用技术包括 MATLAB 对象化编程、符号表达式与数值内核的混合、CUDA / GPU MEX 接口、自动微分、Hutchinson 采样、稀疏线性求解、LaTeX 轮回、以及对 YALMIP、SOSTOOLS、Symbolic Math Toolbox 的桥接。

**📊 数据集**

数据集主要是人工生成的多项式基准：不同变量数、次数、稠密度和稀疏度的多项式；以及先前研究中的实际应用案例（布尔多项式优化、SOS 分解、量化金融的高阶矩阵优化）。

**📈 对比分析**

比较方法通过可复现的脚本在 MATLAB R2023b + RTX 4090 环境下进行，测量 CPU、GPU 以及 GPU‑HP 后端在基础运算、简化、Jacobian、点评估、Affine‑Normal 的执行时间。结果表明：对轻量级交互式工作流，CPU MPOLY 最快；对大规模简化和 Affine‑Normal，GPU‑HP 明显加速；GPU‑GPU 仅在部分简化场景保持优势。

**⚠️ 局限性**

局限性：缺乏完整的 Symbolic Math Toolbox 适配器；SOSTOOLS 仅支持单向导出；稀疏矩阵的单精度性能尚未成熟；尚未实现张量导数接口或反向 SOSTOOLS 转换，未来仍需扩展。

---

## 233. On Emotion-Sensitive Decision Making of Small Language Model Agents

**arXiv ID:** 2604.06562 | [PDF](https://arxiv.org/pdf/2604.06562v1)

**作者:** Jiaju Lin `[一作]` (Pennsylvania State University), Jindong Wang `[通讯]` (William & Mary)

**通讯引用:** 16067 | [OpenAlex ID](https://openalex.org/A5100700956)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究情感对小语言模型（SLM）决策行为的影响，并构建涵盖多种游戏与现实情境的情感决策基准。

**💡 创新点**

创新点在于将激活层情感诱导（activation steering）与结构化博弈评估相结合，同时提出思维审计（thought audit）方法以缓解情感偏差。

**🔧 技术方法**

采用激活 steering、PCA 提取情感向量、链式思考（CoT）与情感词审计等技术。

**📊 数据集**

使用从 Diplomacy、StarCraft II 以及 PersonaHub 合成的多模态、多主体决策样本。

**📈 对比分析**

通过 NDM（归一化漂移幅度）和 NAD（归一化对齐漂移）两项指标对比模型在情感诱导下的决策偏差；实验显示情感能显著改变决策，但对齐度不稳定，思维审计可在一定程度上降低漂移。

**⚠️ 局限性**

局限性包括基准场景有限、情感向量与模型架构高度相关、缺乏更自然、真实世界交互的验证，且对齐评估依赖预先定义的人类行为模式。

---

## 234. When Majority Fails: Tight Bounds for Correlation Distillation Conjectures

**arXiv ID:** 2604.06590 | [PDF](https://arxiv.org/pdf/2604.06590v1)

**作者:** Pritish Kamath `[一作]`, Pasin Manurangsi `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

论文通过理论分析，对两个关于Majority函数的经典猜想（Majority is Least Stable 与 NICD for Erasures）进行了统一处理，给出了在不同噪声参数区间内猜想成立或失效的近似边界，并证明在 n=3 时两猜想始终成立。

**💡 创新点**

创新点在于：① 提出了对原猜想的精细化修订版本，能够捕捉其核心精神；② 用单一的组合与 Fourier 解析框架同时处理两类看似无关的猜想；③ 给出了几乎匹配的上下界，尤其在 ρ≈1 或 p≈0 的小邻域内，阐明了 Majority 作为极值函数的完整性质。

**🔧 技术方法**

主要技术包括：高阶二项式概率估计、组合计数与差分不等式、Fourier 级数与低影响函数的不变性原理、以及对 LTF 与 unate 函数的特定结构利用。论文中还采用了对称性变换与子集随机抽样的技巧，简化了对多数情况的证明。

**📊 数据集**

该工作为纯理论研究，无需外部数据集；所有结果均基于符号演算与概率界定，直接来自 Boolean 函数的结构分析。

**📈 对比分析**

论文并未与实验方法或其他模型做直接比较；其“性能”表现体现在给出的参数区间阈值上：在 ρ∈(1-Θ(1/n²),1) 时 Majority 函数稳健性最小，p∈(0,Θ(1/n²)) 时 NICD 对于 Erasures 的极大相关量被 Majority 取代。

**⚠️ 局限性**

局限性包括：① 结果仅适用于奇数维度 n≥5，n=3 的证明仍依赖特殊构造；② 对于中间噪声区间（ρ≈1/2 或 p≈1/2）仍未给出完整判定；③ 证明中使用的低影响不变性原理在极大维数下可能不够紧凑，导致界限略宽。

---

## 235. Rhythm-consistent semi-Markov simulation of tourist mobility rhythms with probabilistic event-to-POI assignment: Hakone, Japan

**arXiv ID:** 2604.06672 | [PDF](https://arxiv.org/pdf/2604.06672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 236. GraphWalker: Graph-Guided In-Context Learning for Clinical Reasoning on Electronic Health Records

**arXiv ID:** 2604.06684 | [PDF](https://arxiv.org/pdf/2604.06684v1)

**作者:** Yue Fang `[一作]` (Peking University), Liantao Ma `[通讯]` (Peking University)

**通讯引用:** 20172 | [OpenAlex ID](https://openalex.org/A5005130812)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种双视角演示选择框架（GraphWalker），用于在电子健康记录（EHR）上进行上下文学习（ICL），通过构建患者图、聚类发现队列以及基于LLM信息增益的贪心搜索实现演示集的自动选取。

**💡 创新点**

①将数据驱动的临床相似度与模型驱动的信息增益融合，弥补单视角选取的局限；②采用队列（cohort）发现代替单点检索，提升对人群结构的敏感性；③引入惰性贪心+前沿扩展算法，显式考虑演示集之间的冗余与互补，缓解信息聚合的边际递减。

**🔧 技术方法**

①预训练的EHR编码器（如SMART）提取临床序列嵌入；②构造患者图并使用Leiden社区检测得到队列；③利用LLM的条件熵（ΔH）评估信息增益；④惰性贪心搜索与前沿扩展实现组合最优；⑤在LLM推理中使用选取的演示构建少量提示。

**📊 数据集**

主要使用公开的MIMIC‑III和MIMIC‑IV EHR数据集（覆盖住院死亡、ICU再入院、LOS等任务），并在实验中对其他医学推理基准（如Wiki、Med、PEP、MMLU、MIRAGE）进行验证。

**📈 对比分析**

与零样本、随机、监督微调、语义邻居（Qwen3‑Embedding‑8B、SMART）、模型视角方法（CONE、LMS3、Spell、IDS）等基线进行对比；在Qwen3‑14B、LLaMA‑3.1‑8B‑Instruct等后端上，GraphWalker平均提升AUROC≈9.6%、AUPRC≈13%、F1≈7.9%，相较于最佳基线提升约10‑15%，并在推理时间上仅增加数秒，保持高效。

**⚠️ 局限性**

①依赖LLM内部信息流（如条件熵），对更小或不同架构的模型适应性不足；②需要预训练的EHR编码器，若编码器性能差会显著受损；③仅在静态知识库上验证，面对知识库更新需重新训练或结合持续学习；④在极大样本规模下，图构造与前沿扩展的计算成本可能增长；⑤对黑盒API的适用性有限，需要进一步验证。

---

## 237. Tag-based Physical-Layer Authentication Against Message Interference

**arXiv ID:** 2604.06680 | [PDF](https://arxiv.org/pdf/2604.06680v1)

**作者:** Lei Yao `[一作]` (National University of Defense Technology), Chau Yuen `[通讯]` (Nanyang Technological University)

**通讯引用:** 43786 | [OpenAlex ID](https://openalex.org/A5060020877)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了两种新型物理层身份认证方案——Tag‑Based Challenge‑Response（TBCR）和Series Cancellation Authentication（SCA），通过重新设计认证流程和信号帧结构来消除传统基于标签的 PLA 中因信息解码误差产生的干扰并提升检测性能；

**💡 创新点**

创新点在于：1）利用挑战‑响应机制将标签嵌入转发的挑战信号，实现不需要解码即可估计标签；2）设计串行信号生成与取消模块，通过折叠策略在不引入额外噪声的前提下实现标签估计；3）给出闭式表达式推导最优阈值和检测概率，证明 SCA 可达到理想性能；4）从安全角度分析 Eve 的检测极限与键熵，显示在高 SNR 区域 TBCR 仍具备更高安全性。

**🔧 技术方法**

使用的技术包括：物理层身份认证（PLA）基础理论、挑战‑响应机制、信号编码（Polar Code）、并行‑串行转换、交织/去交织、折叠策略、闭式统计推导（正态分布、Q 函数、指数分布、Gaussian‑Chebyshev 量化）、误码分析与阈值优化；

**📊 数据集**

实验采用仿真方式，使用块衰落信道模型，信号长度 L=64，标签功率 ρ_t^2=0.01 或 0.1，代码率 1/2 Polar Code，Monte‑Carlo 10^5 次，未使用公开数据集；

**📈 对比分析**

通过与传统 SUP、BTP、CRAM、BSUP 等方案的 ROC 曲线、检测概率、误报率、关键熵等指标进行对比。结果表明：SCA 在所有 SNR 条件下均达到或接近理想检测概率，TBCR 在 Alice 的 SNR >2 dB 时优于传统方案；两种方案在安全性上均显著优于传统 PLA，Eve 的检测概率远低于 Bob；在带宽效率、时间复杂度等方面均保持在可接受范围内；

**⚠️ 局限性**

局限性包括：SCA 需要额外的交织密钥（密钥更新频率为普通方案的两倍）；两种方案均有轻微的带宽效率下降；SCA 受噪声累积影响较大，尤其在极低 SNR 时表现略逊于 TBCR；在实际系统部署中，挑战信号与标签的同步与误差需进一步控制；

---

## 238. Between Century and Poet: Graph-Based Lexical Semantic Change in Persian Poetry

**arXiv ID:** 2604.06674 | [PDF](https://arxiv.org/pdf/2604.06674v1)

**作者:** Kourosh Shahnazari `[一作]` (Sharif University of Technology), Mohammadali Keshtparvar `[通讯]` (Amirkabir University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对20个关键词在不同时期和不同诗人间的Word2Vec邻域图进行对齐与重构，分析波斯诗歌中词义的历史与诗人语境变化。

**💡 创新点**

提出将语义变化视为邻域图重连，而非单纯向量漂移，并将时间敏感与诗人敏感两种压力统一到同一图结构下进行比较。

**🔧 技术方法**

使用对齐的Skip‑gram Word2Vec、k近邻图、社区检测、中心性与桥接分数等图神经技术。

**📊 数据集**

基于129,451首诗的1,446,347行诗句，按第三至十四世纪划分，涵盖十五位重要诗人。

**📈 对比分析**

通过自漂移、邻域变迁、图角色波动与全局参考对齐的三维度比较，发现相对传统漂移更能捕捉词义重组，结果在实验中显示高一致性且对人类评估保持可解释性。

**⚠️ 局限性**

主要局限在时间样本稀疏、词义多义导致噪声、仅使用静态嵌入而非上下文模型、词集有限且只关注最强邻居。

---

## 239. Computing In Spintronic Memory: A Thermal Perspective

**arXiv ID:** 2604.06667 | [PDF](https://arxiv.org/pdf/2604.06667v1)

**作者:** Patrick Miller `[一作]` (University of Minnesota, Twin Cities), Ulya R. Karpuzcu `[通讯]` (University of Minnesota, Twin Cities)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对使用MRAM（STT和SHE）实现的计算在存储器(CiM)进行热特性定量化建模与仿真，评估不同利用率、数组尺寸和多阵列配置下的温度和功率密度。

**💡 创新点**

首次从热视角系统性地研究在NVM单元内直接执行逻辑的CiM，揭示温度随利用率线性上升、随阵列尺寸线性下降，比较STT与SHE技术的热差异，并探讨多阵列映射对热热点的缓解作用。

**🔧 技术方法**

使用基于HotSpot的有限差分热网络模型；利用执行跟踪获取每个单元功耗；采用STT和SHE MRAM单元进行实验；对比不同利用率、阵列尺寸和多阵列方案。

**📊 数据集**

采用代表性CiM基准数据集，包括INV、VMUL（向量-矩阵乘法）和NN（神经网络）等，生成执行跟踪功耗数据。

**📈 对比分析**

通过在25%–100%利用率、sm/md两种阵列尺寸以及STT/SHE两种技术下计算稳态温度和功率密度，比较热极限是否超出125°C；发现STT热点更严重，SHE显著降低温度；多阵列方案可将执行时间缩短约3.76×，但功率提升约3.45×。

**⚠️ 局限性**

仅考虑稳态热模型，忽略封装、PCB等热通路；未加入动态热节流；模型基于理想化的MTJ网络，未涵盖实际缺陷；研究仅聚焦MRAM单元，缺乏对其他NVM技术的泛化；多阵列方案未完整评估能耗与面积开销。

---

## 240. A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM

**arXiv ID:** 2604.06666 | [PDF](https://arxiv.org/pdf/2604.06666v1)

**作者:** Bo Wang `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**通讯引用:** 11539 | [OpenAlex ID](https://openalex.org/A5029392006)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 G-Defense 框架，利用图结构拆解新闻声明并结合大语言模型与检索增强生成对立解释，实现可解释的假新闻检测。

**💡 创新点**

创新点在于将假新闻检测视为防御式推理，使用子声明图结构并生成竞争性解释以对比信息量和合理性，同时提供细粒度图式解释。

**🔧 技术方法**

采用检索增强生成（RAG）、大语言模型（GPT‑3.5、Flan‑T5）、图到文本序列化、以及防御式推理模块。

**📊 数据集**

在 RAWFC 和 LIAR‑RAW 两个真实检索式数据集上进行实验，均使用公开的未验证原始报告作为证据。

**📈 对比分析**

与传统、LLM 基础以及 L-Defense 等基线相比，G-Defense 在两大数据集的 macro‑F1 上分别提升约 3.1% 和 2.5%，并在解释质量评估（误导性、信息性、合理性、可读性）上超过所有基线。

**⚠️ 局限性**

局限包括对多模态或专业领域信息的覆盖不足、依赖外部检索质量、以及推理过程对 LLM 生成质量高度敏感。

---

## 241. Network-Wide PAoI Guarantee in CF-mMIMO Networks with S&C Coexistence: A Unified Framework for Spatial Partitioning Toward xURLLC

**arXiv ID:** 2604.06657 | [PDF](https://arxiv.org/pdf/2604.06657v1)

**作者:** Yanxi Zhang `[一作]` (Xidian University), Muyu Mei `[通讯]` (Jiangsu University)

**通讯引用:** 132 | [OpenAlex ID](https://openalex.org/A5077098691)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了在大规模 CF‑mMIMO 网络中对感知与通信功能进行空间分区，统一分析网络全局信息新鲜度（PAoI）并给出最优分区策略。

**💡 创新点**

创新点在于将随机几何与随机网络计算（SNC）相结合，推导出可解析的 PAoI 违约概率上界，并基于该上界优化感知-通信分配因子。

**🔧 技术方法**

采用了随机几何模型、SNC、有限块长编码（FBC）以及大规模矩阵理论的确定等价（DE）分析。

**📊 数据集**

通过仿真验证的随机部署（Poisson 点过程）与典型用户模型进行测试，未使用公开数据集。

**📈 对比分析**

与 Monte‑Carlo 仿真对比显示，所得到的上界与最优分区因子与仿真结果高度一致，能够在 xURLLC 场景下显著降低 PAoI 违约概率。

**⚠️ 局限性**

局限在于对信道模型做了理想化假设（如理想机地链路、无链路延迟），并且上界为保守估计，实际网络可能出现偏差。

---

## 242. Adaptive Prompt Structure Factorization: A Framework for Self-Discovering and Optimizing Compositional Prompt Programs

**arXiv ID:** 2604.06699 | [PDF](https://arxiv.org/pdf/2604.06699v1)

**作者:** Haoyue Liu `[一作]` (Chinese University of Hong Kong), Xiaoying Tang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1134 | [OpenAlex ID](https://openalex.org/A5003326020)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个仅靠API调用即可自动化提示结构因子化的prompt优化框架aPSF。

**💡 创新点**

通过自动诱导任务特定的prompt因子分解、单因子干预和误差引导因子选择，实现了可归因、可控且高效的prompt优化。

**🔧 技术方法**

使用Architect LLM生成因子分解与候选编辑，利用Interventional factor-level scoring评估单因子贡献，并通过error-guided factor selection动态分配优化资源。

**📊 数据集**

在多项推理基准上进行评估，包括GSM8K、AQUA-RAT、MultiArith、GSM-Hard、BBH和MMLU等。

**📈 对比分析**

与OPRO、ProTeGi、APE、GrIPS、ZERA、CriSPO以及DSPy等现有API-only优化器及程序化提示框架对比，aPSF在准确率上平均提升1–3%，在token消耗上降低45–87%，并在大多数任务上实现一步收敛。

**⚠️ 局限性**

框架依赖于Architect LLM的质量，假设因子相互独立可能低估因子间交互，实验仅覆盖文本推理任务，尚未验证在多模态或工具辅助场景中的表现。

---

## 243. Reasoning Fails Where Step Flow Breaks

**arXiv ID:** 2604.06695 | [PDF](https://arxiv.org/pdf/2604.06695v1)

**作者:** Xiaoyu Xu `[一作]` (Shanghai Jiao Tong University), Xiaofeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 33866 | [OpenAlex ID](https://openalex.org/A5101742243)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的诊断工具Step-Saliency，用于分析大型推理模型（LRMs）在多步骤数学、科学和编码任务中的推理过程，并提出了一种名为StepFlow的干预方法，以改善这些模型的推理准确性。

**💡 创新点**

创新点在于引入了Step-Saliency作为一种步骤级诊断工具，能够将token级的显著性聚合为步骤间的流动图，并识别出LRMs中的两种信息流失效模式：浅层锁定和深层衰减。同时，StepFlow作为一种轻量级的测试时干预方法，能够在不重新训练的情况下修复信息流。

**🔧 技术方法**

使用了Step-Saliency和StepFlow两种技术，前者用于分析模型的推理过程，后者用于在推理过程中进行干预以改善模型性能。

**📊 数据集**

在多个大型推理模型（如DeepSeek-R1-Distill和GPT-OSS）上进行了评估，使用了六个基准数据集，包括AIME24、AIME25、AMC23、MATH-500、GPQA-Diamond和LiveCodeBench。

**📈 对比分析**

与多种基线方法进行比较，StepFlow在所有基准测试中均表现出一致的准确性提升，尤其是在需要长推理链的竞争性问题上，准确性提升最为显著。例如，在AIME25上，StepFlow使R1-Distill-32B的准确率从54.9%提升至66.7%。

**⚠️ 局限性**

限制在于StepFlow假设了一个基于模型的浅层/深层分割，这一分割依赖于特定的模型架构，并且在小规模的保留数据集上选择。此外，StepFlow主要纠正信息传播错误，而对概念性错误的修正较少。

---

## 244. VDPP: Video Depth Post-Processing for Speed and Scalability

**arXiv ID:** 2604.06665 | [PDF](https://arxiv.org/pdf/2604.06665v1)

**作者:** Daewon Yoon `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**通讯引用:** 8453 | [OpenAlex ID](https://openalex.org/A5084897975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 VDPP 框架，将视频深度估计的后处理从传统 RGB 依赖的全景重建转化为仅利用已估计的单帧深度进行几何细化与时空残差学习。

**💡 创新点**

创新点：① RGB-free、模块化后处理消除适配滞后；② 采用低分辨率几何下采样与残差学习，显著提升速度与内存效率；③ 引入可学习差分标定器和表面法向几何 manifold，提升空间精度与时间一致性。

**🔧 技术方法**

技术细节包括：可学习差分标定器、几何三通道下采样、DINOv2 编码器、VDA 结构的时空解码器、Affine-invariant 空间损失+Temporal Gradient Matching（TGM）损失、残差学习。

**📊 数据集**

使用的数据集：NYUv2、Bonn、Sintel、SVD（SVD 数据集用于零射实验）、KITTI LiDAR（用于传感器无 RGB 场景），所有实验均采样 16 帧序列，分辨率 384×384 以评估实时性能。

**📈 对比分析**

与 NVDS、VDA‑S/L、DPT‑L、DAv2‑B/L 等基线对比，VDPP 在保持或超过 E2E 模型的空间精度（AbsRel、δ1）和时间一致性（TGSE）的同时，帧率提升至 >43 FPS（Jetson Orin Nano）或 >70 FPS（GPU），速度提升 30–40 倍，内存占用仅约 5.5 GB。

**⚠️ 局限性**

局限性：对比基线主要集中在 NVDS，缺乏对其他后处理方法的评估；在极端动态或低光场景下的鲁棒性未知；性能高度依赖单帧深度估计的质量，若输入深度噪声大，残差学习效果可能下降。

---

## 245. Better Balance in Informatics 2.0: The First-Year Students

**arXiv ID:** 2604.06731 | [PDF](https://arxiv.org/pdf/2604.06731v1)

**作者:** Ine Arvola `[一作]` (UiT Arctic University of Norway), Elisavet Kozyri `[通讯]` (UiT Arctic University of Norway)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在挪威北极大学开展BBI 2.0项目，组织技术与文化研讨会，帮助第一年女性计算机科学学生提升技能并留在课程中。

**💡 创新点**

创新点是将技术入门工作坊与文化层面的多样性角色模型相结合，并通过早期干预和面向女性的定向支持来缩小性别差距。

**🔧 技术方法**

主要技术手段为面向学生的互动工作坊、邀请专家演讲以及使用邮件和校园平台进行活动宣传。

**📊 数据集**

使用的数据集包括2024年与2025年两届学生的影响调查问卷、研讨会参与记录、考试成绩和退学率统计。

**📈 对比分析**

通过对比两届调查和绩效数据，发现技术准备度显著提升但退学率无明显下降，整体效果表明干预在技术层面有效，社会情感层面仍需改进。

**⚠️ 局限性**

局限性包括样本量小、研讨会后续跟进不足、课程结构变化对结果产生混杂影响，以及对长期保留效果缺乏追踪。

---

## 246. Enhancing MLLM Spatial Understanding via Active 3D Scene Exploration for Multi-Perspective Reasoning

**arXiv ID:** 2604.06725 | [PDF](https://arxiv.org/pdf/2604.06725v1)

**作者:** Jiahua Chen `[一作]` (Tsinghua University), Qi Fan `[通讯]` (Nanjing University)

**通讯引用:** 92147 | [OpenAlex ID](https://openalex.org/A5006228310)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无训练的框架，通过视觉链式思维和显式3D重建，提升多模态大型语言模型在复杂3D空间推理上的表现；

**💡 创新点**

创新点在于引入视觉链式思维机制，结合可控相机视角生成与知识库驱动的视角选择，实现在单图像下的多视角推理；

**🔧 技术方法**

主要技术包括多粒度关键词抽取、基于SAM3的掩码生成与去重、SAM3D-Object三维重建、外部视角知识库、坐标系统构建及相机外参计算与迭代视角重采样；

**📊 数据集**

使用了3DSRBench、Rel3D、SpatialScore三大空间推理基准数据集；

**📈 对比分析**

与多种公开与闭源基线（包括Gemini-2.5-Flash、GPT-5.2、SpaceMantis-8B等）对比，本文方法在三大基准上均取得最高平均准确率，明显优于现有最优模型；

**⚠️ 局限性**

局限性包括对复杂动态环境的适配尚未验证，视角选择与重采样仍受知识库覆盖范围影响，且在极端遮挡或稀疏图像场景中可能出现误判。

---

## 247. The Traveling Thief Problem with Time Windows: Benchmarks and Heuristics

**arXiv ID:** 2604.06724 | [PDF](https://arxiv.org/pdf/2604.06724v1)

**作者:** Helen Yuliana Angmalisang `[一作]` (Adelaide University), Frank Neumann `[通讯]` (Adelaide University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并研究了旅行贼问题（TTP）在时间窗口约束下的新变体（TTPTW），并设计了双搜索进化算法（DSEA）以及新的巡线初始化方法。

**💡 创新点**

创新点在于：①将TTP扩展到时间窗口约束；②提出针对时间窗口的巡线初始化和多种拆分/整合操作；③引入三种包装计划修复策略并系统比较其效果；④构建了新的TTPTW基准实例集。

**🔧 技术方法**

使用的技术包括：进化算法的双搜索框架、2-opt（Topo）与随机插入（Rain）等搜索算子、ITP/IIP与包装修复（Repack、Repair-Optimizer）相结合、约束-目标分离评估、以及对S4、S5、C5、LKH‑3、VSR‑LKH‑3等现有算法的适配。

**📊 数据集**

采用的测试数据集为基于经典TTP实例（eil51、kroC100、ch150、tsp225、rat575、dsj1000）生成的时间窗口版实例，分为类型A（基于最优TSP路径生成时间窗口）与类型B（随机生成路径）并覆盖多种窗口宽度（l=100,1000,-100,-1000）。

**📈 对比分析**

实验通过30次独立运行、每次1,000,000次函数评估，并使用Kruskal‑Wallis检验及Dunn‑Bonferroni后验检验进行统计比较；结果表明DSEA（尤其是DSEA_1）在绝大多数实例上实现最高可行率和最佳平均目标值，显著优于S4、S5、C5及LKH‑3/VSR‑LKH‑3。

**⚠️ 局限性**

局限性包括：对极窄时间窗口（如l=-1000）和极大规模实例仍难以保证可行性；包装计划修复策略在某些变体中计算开销较大；实验仅采用函数评估作为停止准则，未考虑实际CPU时间；并未探讨多目标或更深层搜索方法。

---

## 248. Infrastructure First: Enabling Embodied AI for Science in the Global South

**arXiv ID:** 2604.06722 | [PDF](https://arxiv.org/pdf/2604.06722v1)

**作者:** Shaoshan Liu `[一作]` (Shenzhen Institute of Artificial Intelligence and Robotics for Society), Zixin Wang `[通讯]` (National Institute of Clean-and-Low-Carbon Energy)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了面向科学实验的具身人工智能(EAI4S)在低资源环境下的可扩展基础设施方案，并通过水质分析实验案例展示了其产能放大效果。

**💡 创新点**

创新点在于：①从基础设施入手，将网络、存储、边缘计算、机器人和软件标准化统一为系统性需求；②通过原子操作层面拆解实验流程，实现机器人自主执行多达22个原子任务；③将具身AI与开源大模型结合，形成可复用、可扩展的实验平台。

**🔧 技术方法**

核心技术包括：多关节协作机器人臂+灵巧机械手、实时视觉感知与分割、抓取与安全检测算法、边缘推理加速器、局域网实时控制、分层本地存储与异步云同步、开源软件栈（机器人控制、实验流程编排、数据记录）。

**📊 数据集**

数据集主要为实验日志与视频：水质实验单日约10 GB的多视角摄像、传感器日志、实验元数据；此外引用机器人科学者Adam、移动化学家、材料合成平台等公开实验数据（如688实验、6,048膜实验）作对比。

**📈 对比分析**

比较方法：对比人类操作员与EAI4S在同一实验流程下的实验数量与耗时。结果显示人类约12–15次/天，EAI4S约40–45次/天，产能提升约3倍；在化学、材料、设备制备等领域亦表现出相似的产能放大效果。

**⚠️ 局限性**

限制因素：①低资源科研机构普遍缺乏机器人平台、可靠边缘计算与大容量本地存储；②网络带宽受限，云端依赖不可持续；③缺乏统一的实验流程标准与可执行协议，影响跨实验室复现；④系统成本仍高于传统研究投入，导致在部分低收入国家难以实现规模化部署。

---

## 249. CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation

**arXiv ID:** 2604.06718 | [PDF](https://arxiv.org/pdf/2604.06718v1)

**作者:** Yanan Cao `[一作]` (Walmart Global Tech), Kannan Achan `[通讯]` (Walmart Global Tech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对大规模零售场景中的下一篮子复购推荐，提出了基于日历时间节奏的CAS模型，能够显式建模购货周期并通过集合注意力捕捉跨品项依赖。

**💡 创新点**

创新点在于将商品购买历史转化为固定长度的二进制日历信号，使用多尺度共享CNN提取复购节奏，并采用诱导集合注意力（ISAB）降低O(n²)复杂度，实现可生产化的时间感知推荐。

**🔧 技术方法**

技术包括多尺度卷积神经网络、诱导集合注意力（Set Transformer）、二元交叉熵训练、以及对比实验中的KNN、GNN、BERT、均值池化等基线。

**📊 数据集**

使用了四个公开数据集（Instacart、TaFeng、DC、Instacart Pro）以及内部大规模零售数据，并在这些数据上进行离线评估与大规模线上实验。

**📈 对比分析**

与TIFUKNN、DNNTSP、BERT4NBR、PIETSP等基线相比，CAS在Precision、Recall、NDCG上均取得或接近最佳表现；线上实验显示在top‑5到top‑20召回中，CAS相对提升约6.8%–8.6% Precision、7.9%–9.9% Recall。

**⚠️ 局限性**

局限性包括对日历时间窗口长度和卷积核大小的超参数依赖，以及在极度稀疏或短周期商品上可能仍需进一步改进；此外，模型仍需要较大内存用于存储多尺度特征，但相对用户特定参数更易扩展。

---

## 250. Steering the Verifiability of Multimodal AI Hallucinations

**arXiv ID:** 2604.06714 | [PDF](https://arxiv.org/pdf/2604.06714v1)

**作者:** Jianhong Pang `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Key Laboratory of Multimodal Embodied AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过构建基于人类评测的图像‑文本验证数据集，研究多模态大型语言模型产生幻觉（hallucination）时的可验证性，进而提出可控的激活空间干预方法，实现对明显与隐蔽幻觉的分别抑制与风险调节。

**💡 创新点**

创新点在于①将幻觉按人类可验证性划分为明显与隐蔽两类；②利用差分均值提取两类对应的激活方向，并在推理阶段通过方向消融实现可验证性可控的幻觉抑制；③设计混合方向 λ 的连续调节，提供从低风险到高风险的可验证性梯度控制。

**🔧 技术方法**

技术手段包括：差分均值（difference‑in‑means）在残差流中提取幻觉方向；方向消融（directional ablation）在多层残差流上按可调强度 α 抑制对应方向；混合方向 λ 用于在两类幻觉方向之间插值，形成可调节的干预策略。

**📊 数据集**

使用的数据集为从 AMBER 图像中生成的富文本对，40 名中文志愿者在 15 秒内对 4,470 条样本进行验证，最终筛选出 1,259 条高质量样本，划分为 351 条明显幻觉、219 条隐蔽幻觉和 689 条无幻觉。

**📈 对比分析**

实验在 Qwen2.5‑VL‑3B、Qwen2.5‑VL‑7B 及 LLaVA‑OneVision‑1.5‑8B 三个公开 MLLM 上进行。OHI 与 EHI 分别显著降低对应子集的幻觉率（HR 降 25–35%），并提升准确率（ACC 上升 2–12%），不确定倾向（UT）增幅有限。混合方向 λ 的使用在保持通用能力（MMBench_CN、TextVQA 误差 < ±3%）的前提下，实现了可验证性从明显到隐蔽的连续调节。

**⚠️ 局限性**

局限性包括：仅在图像‑文本验证任务上验证，缺乏对视频或更复杂场景的评估；干预方法依赖于手工提取的幻觉方向，未与更高级的模型编辑或自监督学习相结合；混合方向的 λ 取值仍需经验选择，未提供自动化调节策略。

---

## 251. A Parameter-Efficient Transfer Learning Approach through Multitask Prompt Distillation and Decomposition for Clinical NLP

**arXiv ID:** 2604.06650 | [PDF](https://arxiv.org/pdf/2604.06650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 252. AgentGate: A Lightweight Structured Routing Engine for the Internet of Agents

**arXiv ID:** 2604.06696 | [PDF](https://arxiv.org/pdf/2604.06696v1)

**作者:** Yujun Cheng `[一作]` (University of Science and Technology Beijing), Qi Xu `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 78986 | [OpenAlex ID](https://openalex.org/A5063102349)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AgentGate，一个两阶段的轻量级结构化路由引擎，用于在边缘设备上为 AI 代理网络进行高效、可解释的请求分派。

**💡 创新点**

创新点：1) 将路由问题从自由文本生成转化为受限决策问题，并拆分为动作决策和结构化基础两步；2) 采用候选感知微调、硬负样本设计来强化决策边界；3) 引入置信度驱动的边缘‑云混合路由，实现低延迟本地决策与高质量云回退的协同。

**🔧 技术方法**

技术：两阶段解码框架（动作 + 结构化）；候选感知微调（LoRA）；结构化 JSON 生成与校验；硬负样本训练；置信度评分与阈值回退；基于候选集的检索与归一化策略。

**📊 数据集**

数据集：3,200 条 AgentDNS 路由基准（2,400 训练 / 400 验证 / 400 测试），覆盖食物配送、租车、预订、天气、航班等十余服务领域，并加入语义相似、误导顺序、敏感触发等硬负样本。

**📈 对比分析**

方法比较：与规则、检索‑排序、通用工具调用三种基线对比，AgentGate 在动作准确率、代理选择、参数精确匹配、计划执行以及安全升级（Escalation）指标均显著优于基线；在 3B–7B 规模模型中，Qwen2.5‑7B 在准确率和安全性上取得最佳平衡，且在边缘部署时拥有最低平均路由延迟与合理内存占用。

**⚠️ 局限性**

局限：1) 置信度阈值与回退策略需人工调优，缺乏自动化校准；2) 依赖候选集的质量，面对极端长尾或高度敏感请求时仍可能误判；3) 评估仅基于离线基准，缺少在线执行与真实系统部署的综合验证。

---

## 253. Towards Accurate and Calibrated Classification: Regularizing Cross-Entropy From A Generative Perspective

**arXiv ID:** 2604.06689 | [PDF](https://arxiv.org/pdf/2604.06689v1)

**作者:** Qipeng Zhan `[一作]` (University of Pennsylvania), Li Shen `[通讯]` (University of Pennsylvania)

**通讯引用:** 32548 | [OpenAlex ID](https://openalex.org/A5100333320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出生成交叉熵（GCE）损失并配合自适应分段温度缩放（ATS），在保持或提升分类准确率的同时显著改善模型的置信度校准。

**💡 创新点**

创新点在于将生成模型视角融入判别训练，形成带类级置信度正则的交叉熵，理论上是严格正确的；并设计了能自适应调节不同置信度区间的温度缩放方法，进一步提升后置校准效果。

**🔧 技术方法**

使用了生成交叉熵损失、类级归一化正则、标签平滑、温度缩放、分段温度缩放（ATS）、深度卷积网络训练、梯度分析、OOD检测（基于熵）等技术。

**📊 数据集**

实验数据集包括 CIFAR‑10、CIFAR‑100、Tiny‑ImageNet、CIFAR‑10‑LT（长尾）、医学影像数据集 AV1451、以及 OOD 数据集 SVHN、CIFAR‑10‑C。

**📈 对比分析**

通过与标准交叉熵、Brier、MMCE、焦点损失（FLSD、DFL、AFL）等基线进行对比，评估准确率和多种校准指标（ECE、AdaECE、Classwise ECE）。结果显示，GCE 在大多数实验中保持或提升准确率，同时大幅降低 ECE；结合 ATS 后，GCE 的校准性能与焦点损失相当，但不损失准确率。

**⚠️ 局限性**

局限性：在极难的数据集（如 Tiny‑ImageNet）上不如焦点损失；OOB 校准在分布偏移时效果下降；类级正则在类别不完整或样本极少时可能受限；当前未能实现与焦点损失的有效融合，仍需进一步研究。

---

## 254. Controllable Generative Video Compression

**arXiv ID:** 2604.06655 | [PDF](https://arxiv.org/pdf/2604.06655v1)

**作者:** Ding Ding `[一作]` (Alibaba Group), Li Li `[通讯]` (Institute of Artificial Intelligence, Hefei Comprehensive National Science Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种可控生成视频压缩（CGVC）范式，利用关键帧和每帧控制先验（主要是亮度分量）压缩后通过可控视频生成模型（VACE）重建非关键帧，配合颜色距离引导的关键帧选择和帧级颜色校正。

**💡 创新点**

通过将可控生成模型与传统视频编码器耦合，实现既保持信号保真又提升感知质量的双重目标；引入亮度先验与颜色距离关键帧选择，显著改善色彩恢复与结构细节。

**🔧 技术方法**

可控视频生成（VACE）、传统视频编码器（VVenC/S266）、颜色直方图距离测度、核密度估计、帧级颜色校正等技术。

**📊 数据集**

HEVC（Class B 1080p、Class E 720p）和MCL‑JCV数据集。

**📈 对比分析**

与VTM‑17.0、DCVC‑FM、SEVC和PLVC等基线在PSNR、MS‑SSIM、MDTVSFA、DISTS等指标上对比，CGVC在相同或更低码率下获得更高的感知质量（MDTVSFA、DISTS提升）且保持或优于传统基线的信号保真度（PSNR、MS‑SSIM）。

**⚠️ 局限性**

依赖可控生成模型，训练和推理成本高；关键帧与控制先验的离散压缩未与生成模型联合优化，信息瓶颈可能限制最优压缩；对极低码率下的细节生成仍可能出现伪影。

---

## 255. Feedback Adaptation for Retrieval-Augmented Generation

**arXiv ID:** 2604.06647 | [PDF](https://arxiv.org/pdf/2604.06647v1)

**作者:** Jihwan Bang `[一作]` (Qualcomm AI Research), Sungha Choi `[通讯]` (Kyung Hee University)

**通讯引用:** 1316 | [OpenAlex ID](https://openalex.org/A5101821431)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了检索增强生成（RAG）系统在接收纠错反馈后如何快速、可靠地调整行为，并提出了“纠正滞后”和“后反馈表现”两条评价指标。

**💡 创新点**

创新点在于将反馈适应作为独立的研究问题，构造可测量的动态评价框架，并提供一种推理时即刻整合反馈的非参数实现，打破了传统训练后更新导致的长滞后限制。

**🔧 技术方法**

技术手段包括意图‑上下文检索（dual‑similarity scoring）、非参数反馈记忆更新、以及将检索到的反馈示例通过 in‑context learning 直接注入生成器的提示，使用 dense retriever（如 bge‑m3）和生成模型（8B 语言模型）。

**📊 数据集**

评估数据集为公开问答基准：Natural Questions、TriviaQA 与 HotpotQA。

**📈 对比分析**

与基线方法（Standard RAG、Self‑RAG、Auto‑RAG、ChatQA‑1.5、RAFT 等）对比，实验表明在相同生成模型下，该方法在后反馈表现上平均提升约 9.7 分，获得最高分数；同时纠正滞后几乎为零，显著优于需重新训练的训练型方法。

**⚠️ 局限性**

局限性包括：仅在受控实验环境下验证，未处理长期累积或冲突反馈的冲突解决；隐私风险未深入探讨，实际部署需注意反馈数据的安全与合规。

---

## 256. Aegon: Auditable AI Content Access with Ledger-Bound Tokens and Hardware-Attested Mobile Receipts

**arXiv ID:** 2604.06693 | [PDF](https://arxiv.org/pdf/2604.06693v1)

**作者:** Amrish Baskaran `[一作]`, Raghul Krishnan `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了Aegon协议，实现可审计的 AI 内容许可体系，支持 Web 端和 Android 端的许可签发、证明与审计；

**💡 创新点**

创新点在于：①将内容特定的 JWT 许可声明与 CT 样式的 Merkle 账本相结合，提供可独立验证的交易记录；②引入硬件 Attestation（Android StrongBox）生成合规性收据，支持离线批量提交；③通过签名的 provenance 日志追踪 AI 处理流程；

**🔧 技术方法**

使用技术包括：JWT + JWKS、Certificate Transparency‑style Merkle 树、Signed Tree Head、Android StrongBox Key Attestation、JWS、SQLite + SQLCipher 离线存储、AWS KMS、Cloudflare Workers、FastAPI、Supabase PostgreSQL；

**📊 数据集**

该工作主要是协议与架构设计，未使用特定数据集；

**📈 对比分析**

对比方法在评估计划中提出了性能指标，如令牌验证 <10 ms（P95），签发 <50 ms（P95），每条 provenance 事件 <5 ms，StrongBox 签名 <100 ms，收据大小 <4 KB 等，实际实验结果待后续论文发布；

**⚠️ 局限性**

局限性包括：provenance 日志无法保证完整性、无法阻止训练使用、对 StrongBox 依赖导致设备兼容性问题、需信任 Broker（单点可信）、以及缺乏完整的训练防护与完整性证明。

---

## 257. Bi-level Heterogeneous Learning for Time Series Foundation Models: A Federated Learning Approach

**arXiv ID:** 2604.06727 | [PDF](https://arxiv.org/pdf/2604.06727v1)

**作者:** Shengchao Chen `[一作]` (University of Technology Sydney), Jing Jiang `[通讯]` (University of Technology Sydney)

**通讯引用:** 15158 | [OpenAlex ID](https://openalex.org/A5024040521)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 FedTRL，一种基于联邦学习的双层异质性学习框架，用于在多域、包含子域漂移的时间序列数据上预训练通用时序基础模型（TSFM）；

**💡 创新点**

创新点在于：① 通过子域对抗正则化和原型对齐实现对内部子域漂移的抑制与跨域一致性的提升；② 引入域感知聚合（DaG），将域判别风险与原型对齐信号融合，动态加权客户端更新；③ 采用扩散式重建无监督预训练与双头点预测/概率预测相结合，提升零样本性能；

**🔧 技术方法**

使用技术包括：联邦学习框架、梯度反转层 (GRL) 对抗域分类器、原型对齐、域感知聚合、扩散式时间序列重建、因果 Transformer 编码器、点预测与 Student‑t 概率预测双头结构、实例归一化与补丁分块；

**📊 数据集**

实验数据集涵盖：TSLib（ETTh1/2、ETTm1/2、Electricity、Traffic、Weather、Exchange）、RW‑Bench（15个气象站）、Time‑MoE‑300B（9个域、3亿亿条）、GIFT‑eval、FEV Leaderboard；

**📈 对比分析**

与多种基线对比：掩码重建（SimMTM、PatchTST）、对比学习（TS‑TCC、CoST）、联邦基线（FFTS、FedAvg）、中心化TSFM、各类高级时序模型（TimeMixer、TimesNet、Autoformer 等）。FedTRL 在点预测、零样本预测及概率预测上均实现或超过中心化基线，取得 MSE/MAE、MASE/CRPS 等指标的显著提升；

**⚠️ 局限性**

局限性包括：① 对客户端异质性极度不平衡时仍可能出现通信与计算瓶颈；② 依赖梯度反转和域分类器，需仔细调参；③ 主要验证于公开时间序列基准，缺乏对不同业务场景（如医疗、工业）的进一步评估；④ 对安全隐私的理论分析有限。

---

## 258. HQF-Net: A Hybrid Quantum-Classical Multi-Scale Fusion Network for Remote Sensing Image Segmentation

**arXiv ID:** 2604.06715 | [PDF](https://arxiv.org/pdf/2604.06715v1)

**作者:** Md Aminur Hossain `[一作]` (Space Applications Centre, ISRO), Biplab Banerjee `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 2114 | [OpenAlex ID](https://openalex.org/A5020786167)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种混合量子-经典的HQF-Net网络，用冻结的DINOv3 ViT-L/16作为多尺度语义引导，并在U-Net架构中加入量子增强的跳跃连接和量子Mixture-of-Experts瓶颈，实现遥感图像的像素级语义分割。

**💡 创新点**

创新点在于将多尺度Deformable Cross-Attention Fusion与量子增强的QSkip以及自适应量子专家路由QMoE相结合，能够在局部、全局与方向性特征上实现自适应融合，从而显著提升分割性能。

**🔧 技术方法**

使用了自监督视觉Transformer（DINOv3）作为特征提取器、Deformable多尺度注意力、FiLM门控与残差注入、量子电路（局部、全局、对角专家）以及量子Mixture-of-Experts。

**📊 数据集**

在LandCover.ai、OpenEarthMap和SeasoNet这三大遥感分割基准上进行实验。

**📈 对比分析**

与多种经典CNN、UNet变体以及现有混合量子模型在mIoU和OA指标上进行对比，HQF-Net分别在LandCover.ai取得0.8568 mIoU / 96.87% OA，OpenEarthMap取得71.82% mIoU，SeasoNet取得55.28% mIoU / 99.37% OA，均超过所有基线。

**⚠️ 局限性**

受限于NISQ设备的量子比特数和电路深度，量子模块需要对经典特征做压缩，导致对高分辨率或大规模数据的可扩展性尚未验证，且实际量子部署与训练成本仍是挑战。

---

## 259. Broken Quantum: A Systematic Formal Verification Study of Security Vulnerabilities Across the Open-Source Quantum Computing Simulator Ecosystem

**arXiv ID:** 2604.06712 | [PDF](https://arxiv.org/pdf/2604.06712v1)

**作者:** Dominik Blain `[一作]` `[通讯]` (QreativeLab), Dominik Blain (QreativeLab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对45个开源量子计算模拟框架进行了系统化、形式化的安全审计，发现547条安全缺陷，涵盖四类（C++内存破坏、Python资源耗尽、序列化/代码注入、QASM注入），并对每个缺陷通过Z3 SMT求解器提供形式化可达性证明；

**💡 创新点**

首次以形式化方法全面审计量子模拟器生态，提出新的QASM注入攻击类；通过供应链分析揭示商业框架漏洞向美国国家实验室蔓延；发现32量子比特为多种攻击的阈值；发布COBALT QAI工具支持持续安全检测；

**🔧 技术方法**

使用静态正则表达式模式扫描器、Z3 SMT约束求解器进行可达性验证、手工代码审计与PoC演示；利用Python、C++、OpenQASM等多语言分析技术；

**📊 数据集**

涵盖45个框架，来自22家组织、12个国家（IBM、Google、Amazon、NVIDIA、Microsoft、Xanadu、CERN、Harvard、MIT、Oak Ridge NL、ETH Zürich等），通过这些代码库构建数据集；

**📈 对比分析**

对框架进行分数卡评估，计算CRITICAL、HIGH、MEDIUM计数并给出整体安全等级；识别32量子比特阈值并在不同语言实现上得到一致证明；结果显示约80%的生态存在至少一项缺陷，只有9个框架通过所有扫描器；

**⚠️ 局限性**

局限性包括：模式匹配可能产生误报、云端验证层未纳入、C++未定义行为的实际利用不确定、Python资源耗尽的可利用性取决于部署环境、并未覆盖所有可能的安全缺陷类型（如使用后释放等）

---

## 260. ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding

**arXiv ID:** 2604.06685 | [PDF](https://arxiv.org/pdf/2604.06685v1)

**作者:** Xuanle Zhao `[一作]` (Chinese Academy of Sciences), Bo Xu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 47037 | [OpenAlex ID](https://openalex.org/A5102005952)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并训练了 ChemVLR，一个能够在化学图像中进行分步骤推理的视觉语言模型，配合三阶段训练（持续预训练、监督微调、强化学习）。

**💡 创新点**

创新点在于：①通过跨模态逆向工程生成高质量的推理链和图像说明数据；②在视觉感知前加入细粒度化学描述（如官能团）；③利用 IUPAC 名称等语义锚点激活预训练知识；④提出三阶段训练管线与 DAPO 强化学习的组合。

**🔧 技术方法**

使用 Gemini‑2.5‑Flash 生成推理链，GPT‑4.1‑mini 做验证；ViT 视觉编码器；LLM 变体 Qwen2.5‑VL‑7B / Qwen3‑VL‑8B‑Instruct；RDKit 提取官能团；IUPAC 检索；DAPO 强化学习框架。

**📊 数据集**

构建 760k 规模数据集，包含 400k 图像说明、168k 识别样本、192k 预测样本；还采集 200k+200k 的图像‑说明对、1.4M 指令对、300k 图像‑IUPAC 对；并整合 ChEBI‑20‑MM、ChemMLLM、ORDerly、PubChem 等公开资源。

**📈 对比分析**

在 MMChemOCR、img2smiles、ChemRxn‑V 等基准上与专有模型（Gemini‑3‑Flash、GPT‑5‑mini、GPT‑4o）、开源 VLM（Phi‑3.5‑Vision、Qwen、InternVL 等）以及化学领域 VLM（ChemVLM、ChemDFM‑X、TinyChemVL）进行对比；ChemVLR 在 Tanimoto‑Hit@1.0、Avg. Sim 等指标上均达到或超过 SOTA，尤其是 8B 版本在 Tanimoto‑Hit@1.0 达到 93.8%/84.6%/97.4% 等显著提升。

**⚠️ 局限性**

限制包括：仍有少量推理链错误；目前仅覆盖识别与反应预测任务，未覆盖性质预测等；缺乏对真实教育场景或多样化视觉输入的评估；缺乏更高效的数据进一步精炼方法。

---

## 261. Restoring Heterogeneity in LLM-based Social Simulation: An Audience Segmentation Approach

**arXiv ID:** 2604.06663 | [PDF](https://arxiv.org/pdf/2604.06663v1)

**作者:** Xiaoyou Qin `[一作]` (Fudan University), Xiaoxiao Cheng `[通讯]` (Zhejiang University)

**通讯引用:** 2499 | [OpenAlex ID](https://openalex.org/A5075712280)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究通过观众细分方法恢复LLM社会仿真中的异质性，并系统比较不同细分配置的效果。

**💡 创新点**

创新点在于提出并实证验证观众细分为核心方法，揭示分辨率、简约性与选择逻辑对分布、结构、预测忠诚度的三维评估影响。

**🔧 技术方法**

使用大语言模型（Llama 3.1‑70B和Mixtral 8x22B）的零样本问答提示，并结合聚类、梯度提升机等特征选择技术。

**📊 数据集**

基于美国气候态度调查的“六大洲”样本以及通过SASSY工具分段的问卷数据。

**📈 对比分析**

通过三维评价框架（分布、结构、预测忠实度）与多指标（如KLD、nEMD、Cramér's V）对比六种细分配置，发现中等细粒度和简约配置在多数维度表现最佳，细粒度不一贯提升。

**⚠️ 局限性**

所有配置仍出现过度正则化，无法完全恢复真实人口异质性，且效果受LLM架构差异影响，需进一步提升模型的方差保持能力。

---

## 262. FlowAdam: Implicit Regularization via Geometry-Aware Soft Momentum Injection

**arXiv ID:** 2604.06652 | [PDF](https://arxiv.org/pdf/2604.06652v1)

**作者:** Devender Singh `[一作]` (Memorial University of Newfoundland), Tarun Sheel `[通讯]` (Memorial University of Newfoundland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的混合优化器 FlowAdam，结合 Adam 与连续梯度流 ODE，在适当时刻切换模式并通过软动量注入实现平滑过渡。

**💡 创新点**

核心创新在于软动量注入机制，使 ODE 速度与 Adam 的动量平滑融合，避免了传统切换导致的训练崩溃，并通过自适应触发实现隐式正则化。

**🔧 技术方法**

技术包括：EMA 动态触发判定、剪切梯度的梯度流 ODE（Dormand–Prince RK4/5）、软动量注入与速度裁剪、以及全流程的全批次/大批次模式切换。

**📊 数据集**

实验数据集涵盖矩阵/张量完成、鲁棒矩阵分解、GNN 链接预测、逆运动学、Jester 以及 MovieLens-100K 等多种耦合优化场景。

**📈 对比分析**

与 Adam、AdamW、Lion、AdaBelief、L-BFGS 等方法对比，FlowAdam 在耦合任务上平均提升 10–22% 的错误率，Jester 上比 Adam 提升 6%，在大规模协同过滤上超过调参后的 Lion；在无耦合或良好条件任务上表现与 Adam 相当。

**⚠️ 局限性**

局限性包括：触发机制对小批量噪声敏感；缺乏完整收敛理论；ODE 触发会产生 8–12 次额外梯度计算，可能在极大模型上成本高；需手动选择模式 A/B；实验规模有限，未验证在极大模型上的可扩展性。

---

## 263. Turn Your Face Into An Attack Surface: Screen Attack Using Facial Reflections in Video Conferencing

**arXiv ID:** 2604.06729 | [PDF](https://arxiv.org/pdf/2604.06729v1)

**作者:** Yong Huang `[一作]` (Zhengzhou University), Wanqing Tu `[通讯]` (Durham University)

**通讯引用:** 686 | [OpenAlex ID](https://openalex.org/A5023539073)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于视频会议中人脸反射的侧信道攻击方法FaceTell，能够通过在线视频流泄露参与者的二次应用使用情况

**💡 创新点**

创新点在于利用人脸这一普遍且非光滑的表面作为反射媒介，并结合自适应超分重建、双层分类与基于时序的标签校正算法，实现高精度的屏幕内容识别

**🔧 技术方法**

采用Mask‑R‑CNN与Haar分类器进行人脸分割，CAMixerSR进行超分重建，CBAM注意力模块与残差块提取特征，双层全连接分类网络以及启发式标签校正（HLC）算法

**📊 数据集**

使用24名受试者在13种室内环境下采集的12小时视频数据，经过帧抽样得到49,407张训练脸图和12,372张测试脸图，涵盖28个常用桌面应用程序和5个类别

**📈 对比分析**

与无超分、无注意力、无双层分类和无标签校正等对照模型比较，FaceTell在各类应用上平均准确率达99.32%，分类精度高达99.85%，在多种影响因素（性别、遮挡、距离、角度、照明）下保持95%以上的鲁棒性；运行时约124 ms/帧，可同时监控两名受试者

**⚠️ 局限性**

局限性包括在多屏幕、户外强光或开启皮肤平滑滤镜等环境下准确率骤降（≤17%），暗模式或新应用的识别能力下降，且需要在目标视频中保持单一主屏光照的假设

---

## 264. Fine-grained Approaches for Confidence Calibration of LLMs in Automated Code Revision

**arXiv ID:** 2604.06723 | [PDF](https://arxiv.org/pdf/2604.06723v1)

**作者:** Hong Yi Lin `[一作]`, Christoph Treude `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在自动代码生成（ACR）任务中采用细粒度置信度评分与Platt缩放校准方法的效果，探讨了全局与局部缩放对不同模型与任务的影响。

**💡 创新点**

创新点在于提出了细粒度置信度度量（最小词概率、最低‑K词概率、注意力加权不确定性）并设计了局部Platt缩放方案，通过聚类嵌入自适应地校准不同错误模式，从而显著提升校准性能。

**🔧 技术方法**

采用的技术包括大型语言模型（LLMs）、软最大化概率计算、Platt缩放（全局与局部）、聚类算法（HDBSCAN或类似方法）以及注意力权重加权不确定性估计。

**📊 数据集**

使用的数据集涵盖三种ACR任务：DeepFix Bug修复（DCF‑Bug）、DeepFix Vulnerability修复（DCF‑Vul）和代码审查翻译（CR‑Trans），共计约1.4万条样本。

**📈 对比分析**

通过对比全局与局部Platt缩放以及细粒度与序列级置信度评分，结果显示细粒度评分在所有模型与任务中均降低了ECE与Brier误差，且提高了分箱覆盖率；局部缩放在误差异质性高的CR‑Trans任务上进一步提升了校准效果，整体性能显著优于传统方法。

**⚠️ 局限性**

局限性包括：即使使用细粒度评分与局部缩放，CR‑Trans任务的ECE仍高于0.14，难以满足精细决策需求；全局缩放在误差高度异质的场景下表现有限；聚类与校准训练增加了计算与存储开销；此外仅在贪婪推理下评估，未考察采样等生成策略的影响。

---

## 265. Improving Local Feature Matching by Entropy-inspired Scale Adaptability and Flow-endowed Local Consistency

**arXiv ID:** 2604.06713 | [PDF](https://arxiv.org/pdf/2604.06713v1)

**作者:** Ke Jin `[一作]` (Zhejiang University), Qi Ye `[通讯]` (Zhejiang University)

**通讯引用:** 10840 | [OpenAlex ID](https://openalex.org/A5101755166)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种改进的半稠密图像匹配管线，解决粗匹配的过度排除和细匹配的局部不一致问题，提升匹配精度与鲁棒性。

**💡 创新点**

创新点包括：①利用得分矩阵熵来估计图像间的尺度比例，并引入自适应互最近邻（AMNN）机制放宽匹配约束；②将细阶段改为级联流量细化，并设计梯度损失以强化流场的局部一致性。

**🔧 技术方法**

核心技术包括双软最大化匹配层、熵基可见性判定、尺度适应的窗口检验、自注意力/交叉注意力特征变换、深度卷积级联细化以及梯度约束损失。

**📊 数据集**

在MegaDepth、ScanNet、HPatches、InLoc、Aachen v1.1等公开数据集上进行训练和评估，使用了大规模户外MegaDepth和室内ScanNet的相机位姿与深度信息。

**📈 对比分析**

与Sparse（SuperPoint、SuperGlue等）、Semi-dense（LoFTR、EfficientLoFTR、AdaMatcher、CoMatch等）和Dense（DKM、ROMA等）方法对比，本文方法在相机位姿估计、单应性估计和视觉定位任务上均取得领先或相当的AUC/错误率，并保持较低的运行时（约56 ms/对）。

**⚠️ 局限性**

局限性在于对局部尺度变化（如强视角扭曲）估计不够精准；在极端视角或外观变化下仍不如全密集匹配方法；以及在大规模3D场景中的可扩展性尚待验证。

---

## 266. Specializing Large Models for Oracle Bone Script Interpretation via Component-Grounded Multimodal Knowledge Augmentation

**arXiv ID:** 2604.06711 | [PDF](https://arxiv.org/pdf/2604.06711v1)

**作者:** Jianing Zhang `[一作]` (Jilin University), Xi Yang `[通讯]` (Jilin University)

**通讯引用:** 33829 | [OpenAlex ID](https://openalex.org/A5100416352)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过基于组件的检索增强生成框架，实现了甲骨文的自动解读；

**💡 创新点**

将甲骨文拆解为可识别的象形部件并构建知识图谱，采用Agent驱动的检索-生成链实现结构化推理，突破传统闭集识别方法的解释鸿沟；

**🔧 技术方法**

结合Vision‑Language模型（ViT+DINOv2、GPT‑5/Claude/GLM/Qwen等）、原型网络、知识图谱检索工具与多智能体协作；

**📊 数据集**

使用OB‑Radix数据集：1,022张甲骨字符图像（934种独特字符）、1,853张细粒度部件图像（478种部件），附专业注释与语义解释；

**📈 对比分析**

在组件检索、组件关系推断、完整解读三大基准上与传统单体VLM和直接生成对比，Agentic RAG/多智能体方案在TOP‑1/ACC/ROUGE/BERTScore等指标上提升约10–20%甚至更大，最终模型在专家评审中获得最高分；

**⚠️ 局限性**

组件识别仍不够完整，知识图谱覆盖有限，系统仍需外部检索，面对未见字符和多音节复合字符时易产生错误。

---

## 267. ATANT: An Evaluation Framework for AI Continuity

**arXiv ID:** 2604.06710 | [PDF](https://arxiv.org/pdf/2604.06710v1)

**作者:** Samuel Sameer Tanguturi `[一作]` `[通讯]` (Kenotic Labs), Samuel Sameer Tanguturi (Kenotic Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ATANT 框架，用于评估 AI 系统在时间维度上的连续性，并给出了七项必需属性、十个检查点以及四个合规级别。

**💡 创新点**

创新点在于：①将连续性定义为系统层面并拆解为七个可测试属性；②设计无 LLM 循环的十检查点写读路径验证方法；③构建 250 条真实多轮叙事测试集；④提供分阶段合规路径，便于系统迭代验证。

**🔧 技术方法**

采用基于规则的写读路径验证、结构匹配、语义索引与确定性轨迹收敛等技术，形成无模型（或模型独立）的连续性架构。

**📊 数据集**

使用 250 条多轮叙事语料（1,835 题）涵盖六个人生领域（职业、人际、健康、学习、日常、重大事件）。

**📈 对比分析**

通过对参考实现的五轮迭代评估，发现从 58%（旧体系）提升至 100%（孤立模式）和 96%（累计规模）准确率，证明连续性是架构问题而非单纯调参。

**⚠️ 局限性**

局限包括：评估基于关键词匹配，未衡量答案连贯性；数据集单一作者、单语言；仅测试一个系统；缺少多语言和主动行为等场景。

---

## 268. Bi-Lipschitz Autoencoder With Injectivity Guarantee

**arXiv ID:** 2604.06701 | [PDF](https://arxiv.org/pdf/2604.06701v1)

**作者:** Qipeng Zhan `[一作]` (University of Pennsylvania), Li Shen `[通讯]` (University of Pennsylvania)

**通讯引用:** 32548 | [OpenAlex ID](https://openalex.org/A5100333320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种新型的 Bi-Lipschitz Autoencoder (BLAE)，旨在通过注入性约束和 bi‑Lipschitz 正则化解决传统自编码器在低维表示学习中的非注入性瓶颈。

**💡 创新点**

创新点：①提出基于分离准则的注入性正则化，消除路径陷阱；②引入 bi‑Lipschitz 约束实现几何保持，同时保持对分布漂移的鲁棒性；③在理论上证明了正则化的可接受性和可行性。

**🔧 技术方法**

技术：结合梯度正则化（Jacobians）、图正则化（邻域图）、分离准则约束、bi‑Lipschitz 约束；使用神经网络（可激活如 tanh/ELU）实现编码器/解码器；通过梯度下降训练并评估几何一致性。

**📊 数据集**

数据集：合成 Swiss Roll、dSprites（形状/位置）、MNIST（旋转/缩放变换）以及其它多样化数据集，用于验证几何保持与分布漂移鲁棒性。

**📈 对比分析**

与 9 个基线（SPAE、TAE、IRAE、GAE、CAE、GRAE、Diffusion Net、GGAE、Vanilla AE）在 k‑NN recall、KL 散度、MSE、下游分类准确率等指标上比较，BLAE 在几何保持和分布不变性上均获得最高排名，显著优于传统方法。

**⚠️ 局限性**

局限性：需要预计算 geodesic 距离矩阵，时间与空间复杂度为 O(N²)，在大规模数据集（如 ImageNet）上不易扩展；对极端稀疏采样仍需进一步优化。

---

## 269. When Agent Markets Arrive

**arXiv ID:** 2604.06688 | [PDF](https://arxiv.org/pdf/2604.06688v1)

**作者:** Xuan Liu `[一作]` (University of California San Diego), Haojian Jin `[通讯]` (University of California San Diego)

**通讯引用:** 739 | [OpenAlex ID](https://openalex.org/A5032046097)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并实验了一个可编程的 AI 代理市场系统，完整模拟工作发布、投标、谈判、执行、支付与声誉累积的闭环；

**💡 创新点**

首次将 LLM 代理嵌入完整的经济循环，并逐一可控地实验机构机制对市场行为的影响；

**🔧 技术方法**

使用 Claude LLM（不同模型族）、工具调用框架、封闭式投标、双边声誉记录及进化式淘汰等技术；

**📊 数据集**

整合 234 个跨领域任务，来自 SkillsBench、ToolQA 与 BFCL v4 的真实专业、数据查询与函数调用任务；

**📈 对比分析**

与自给自足（autarky）对比以及对透明度、诚实度、选择强度等机制的单项消融，结果显示市场可产生 3.2 倍财富、质量提升，但争议率高达 42%；

**⚠️ 局限性**

局限在样本规模有限、仅模拟 25 代理、任务池与模型族多样性受限、使用执行重放保证可复现性，且实验环境与真实部署仍存在差距。

---

## 270. SwarmIO: Towards 100 Million IOPS SSD Emulation for Next-generation GPU-centric Storage Systems

**arXiv ID:** 2604.06668 | [PDF](https://arxiv.org/pdf/2604.06668v1)

**作者:** Hyeseong Kim `[一作]` (KAIST), Minsoo Rhu `[通讯]` (KAIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种面向 GPU 端发起 I/O 的 SSD 仿真框架，能够在真实 GPU 系统上实时模拟高达 40 MIOPS（并可扩展至 100 MIOPS）的大规模随机读写。

**💡 创新点**

创新点包括：① 分布式前端架构与批量请求获取，最大化 NVMe 级并行度；② 针对 Intel DSA 的专用内核级异步批量复制接口，显著降低 CPU 级别的数据搬移开销；③ 聚合式时序模型更新，减少跨前端线程的锁竞争，保持高精度时序模拟。

**🔧 技术方法**

采用 Intel DSA（Data Streaming Accelerator）进行硬件加速复制；在 Linux 内核中扩展 NVMeVirt，添加分布式服务单元；实现自定义 DSA 复制 API（批量描述符、异步提交/轮询）；利用多组 DSA 进行前后端并行处理。

**📊 数据集**

使用 Solidigm D7‑PS1010 PCIe Gen5 SSD 进行性能校准；通过 BaM（GPU 端 I/O）和 fio（CPU 端 I/O）基准验证仿真准确性；在案例研究中，使用 BIGANN‑100M 数据集与 CAGRA 向量搜索算法评估端到端性能。

**📈 对比分析**

与原始 NVMeVirt 对比，仿真器在 GPU 端 I/O 下实现 303.9× 的速度提升，最大 38.6 MIOPS；在 CAGRA 向量搜索中，将 SSD IOPS 从 2.5 MIOPS 提升至 40 MIOPS 可获得 9.7× 的 QPS 加速；仿真 IOPS 与真实 SSD 的误差低于 8%，延迟误差仅 2.8%。

**⚠️ 局限性**

目前受限于可用的 DSA 设备数量，最大支持 40 MIOPS；需多插槽平台并配备 ≥5 Dsa 设备才能实现 100 MIOPS；对大块大小（>2 KB）数据传输仍有 PCIe 带宽瓶颈；未在实际未来 SSD 上验证，且依赖特定硬件加速，迁移到其它平台可能受限。

---

## 271. Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start

**arXiv ID:** 2604.06664 | [PDF](https://arxiv.org/pdf/2604.06664v1)

**作者:** Xueshen Liu `[一作]` (University of Michigan), Z. Morley Mao `[通讯]` (University of Michigan)

**通讯引用:** 14492 | [OpenAlex ID](https://openalex.org/A5003217329)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Foundry，一个模板化 CUDA 图上下文材料化系统，能在 LLM 服务的冷启动阶段快速重建可执行图，显著降低启动延迟。

**💡 创新点**

创新点：1) 同时持久化图拓扑和执行上下文，消除传统图捕获中依赖运行时环境的瓶颈；2) 通过确定性内存布局和自动提取内核二进制，实现无 kernel‑specific patch 的图恢复；3) 利用图拓扑模板化与单 GPU 离线捕获，实现多 GPU 部署共享同一模板，显著减少多 GPU 初始化成本；4) 通过 stub 通信实现跨 rank 图共享，兼容 SPMD 并行策略。

**🔧 技术方法**

技术手段：CUDA 图捕获与重构、驱动层 Hook（VMM 重定向）、内核二进制提取与重载、图模板化（按拓扑聚类）、单 GPU 异步重建、PyTorch + vLLM 集成、NVSHMEM/NCCL stub、FP8 专用 DeepGEMM 支持。

**📊 数据集**

评估数据集：多种大模型（Qwen3‑14B、32B、30B‑A3B、235B‑A22B；Llama3‑8B；Gemma3‑12B），覆盖 dense 与 MoE 体系；使用 H100 与 B200 GPU 集群，配置 DP/TP/EP 等多种并行度。

**📈 对比分析**

实验对比：与 vLLM 默认 CUDA 图、无 CUDA 图、CUDA‑checkpoint 三种基线对比。冷启动延迟在 15 个配置下从 10 分钟（vLLM + CUDA 图）压缩至 3.9 s，99% 降低；吞吐量保持与原始 CUDA 图相当；单图构建比原始 stream capture 1.9–2.9×快，模板更新仅 1 ms；存储体积比 CUDA‑checkpoint 小 4–5×。

**⚠️ 局限性**

局限性：仅适用于 SPMD（DP/TP/EP）多 GPU；不支持流水线并行（PP）；需要在离线阶段完成单 GPU 捕获，对新硬件或自定义 kernel 的适配需手动扩展；无法自动处理模型权重变更或动态 KV‑cache 大小调整；跨 GPU 通信需要手动 stub，可能不兼容所有通信库。

---

## 272. Towards Robust Content Watermarking Against Removal and Forgery Attacks

**arXiv ID:** 2604.06662 | [PDF](https://arxiv.org/pdf/2604.06662v1)

**作者:** Yifan Zhu `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences and University of Chinese Academy of Sciences), Xiao-Shan Gao `[通讯]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences and University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的实例特定水印方法ISTS，配合双向检测，能够抵抗文本到图像扩散模型的去水印和伪造攻击。

**💡 创新点**

①动态根据提示语语义自适应选择水印注入时刻和位置，增加了水印的多样性与不可预测性；②双向检测取代传统单向检测，显著提升对逆向潜在表示攻击的鲁棒性。

**🔧 技术方法**

使用CLIP对生成图像提取语义特征作为参数选择器；基于DDIM的逆向推理实现水印注入；采用频域（傅里叶变换）插入固定水印图案；在检测时计算双向相似度。

**📊 数据集**

Stable‑Diffusion‑2‑1‑base模型生成图像，使用公开的文本提示集合；对比数据集包含约1000对水印与非水印样本以及三种去水印和三种伪造攻击样本。

**📈 对比分析**

与Tree‑Ring、Shallow Diffuse、Gaussian‑Shading、ROBIN、RingID、Zodiac、SEAL等现有方法对比；在三种去水印攻击中平均AUC>0.93、TPR@1%FPR>0.58，最差情况仍达0.821/0.18；在三种伪造攻击中平均AUC≈0.686、TPR@1%FPR≈0.12，显示显著优于基线。

**⚠️ 局限性**

尽管在最坏情况下对Imp‑Removal和VAE‑Forgery仍有提升空间，且在某些图像失真或去水印（如Diffpure）下鲁棒性略逊于Tree‑Ring，未来需进一步提升极端攻击下的防御能力。

---

## 273. Nexus: Transparent I/O Offloading for High-Density Serverless Computing

**arXiv ID:** 2604.06682 | [PDF](https://arxiv.org/pdf/2604.06682v1)

**作者:** JooYoung Park `[一作]` (NTU Singapore), Dmitrii Ustiugov `[通讯]` (NTU Singapore)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现一种基于KVM的服务器无关虚拟机监控器，将应用计算与I/O通信解耦，利用共享内存将云SDK和RPC框架移到宿主机后台，显著降低每个函数实例的内存占用和CPU开销，提升节点部署密度。

**💡 创新点**

创新点在于：① 在保持完整POSIX与高层SDK兼容性的前提下，将I/O操作在高层API边界上截断并远程到宿主机共享后端；② 通过可插拔的零拷贝共享内存与RDMA实现高效数据传输；③ 结合预取和异步写回的异步优化，打破传统的序列化执行路径，进一步压缩启动与执行时间。

**🔧 技术方法**

核心技术包括：KVM虚拟化、基于Unix Domain Socket的轻量控制通道、零拷贝共享内存（通过file-backed memory和virtio‑mmio实现）、RDMA网络支持、Go实现的高并发后端、Python轻量前端拦截库、基于身份令牌的最小权限访问控制。

**📊 数据集**

使用的主要数据集为：vSwarm函数套件（10个Python函数，覆盖从高I/O到高计算多种工作负载）和Azure Function生产工作负载的采样轨迹，云存储端使用MinIO仿真S3；实验还与Faasm（WASM轻量化虚拟机）进行对比。

**📈 对比分析**

实验通过在四台工作节点上部署Knative+Firecracker与自研后端，按需扩容并测量p99延迟、CPU与内存占用、冷启动时延。结果表明，-Async版本相较基线提升部署密度18%（TCP）至37%（RDMA），CPU和内存使用分别降低44%和31%，冷启动延迟平均缩短10%，并在大多数I/O密集工作负载上实现20–40%的延迟改进；与Faasm相比，尽管功能兼容性更好，但在CPU/内存上仍略逊一筹（20–25%差距）。

**⚠️ 局限性**

局限性包括：① 共享后端成为跨租户故障域，需依赖安全的内存隔离与硬件保护（MPK/CHERI）来提升安全；② 目前实现聚焦Python，扩展到Node.js/Java等需要编写对应拦截前端；③ RDMA连接建立时的开销与支持硬件依赖仍是瓶颈；④ 由于异步写回仍需后台确认，可能对极低延迟场景产生微小影响。

---

## 274. URMF: Uncertainty-aware Robust Multimodal Fusion for Multimodal Sarcasm Detection

**arXiv ID:** 2604.06728 | [PDF](https://arxiv.org/pdf/2604.06728v1)

**作者:** Zhenyu Wang `[一作]` (China University of Mining and Technology), Guoying Zhang `[通讯]` (China University of Mining and Technology)

**通讯引用:** 13568 | [OpenAlex ID](https://openalex.org/A5100633436)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于不确定性感知的鲁棒多模态融合框架URMF，用于多模态讽刺检测

**💡 创新点**

创新点在于：①先进行跨模态对齐后再内部推理；②对文本、图像以及交互态隐空间做高斯分布的显式方差建模；③利用估计的不确定性动态调节模态权重；④提出联合信息瓶颈、模态先验、分布对齐和自采样对比学习的综合训练目标

**🔧 技术方法**

采用Transformer结构的跨模态交互模块、两头全连接的均值与方差预测、重参数化采样、指数权重调节、信息瓶颈正则、KL对齐、UCL对比学习等技术

**📊 数据集**

在公开的Multimodal Sarcasm Detection（MSD）数据集上进行实验

**📈 对比分析**

与多种单模态、传统多模态以及大模型（LLaVA1.5等）基线对比，URMF在Accuracy 95.02%、Precision 94.69%、Recall 95.19%、F1 94.91%等指标上均达到或超过最高水平

**⚠️ 局限性**

局限性包括：仅验证于MSD数据集，未评估对更大规模或其他模态组合的迁移性；对高斯方差假设的依赖可能限制对极端噪声的建模；以及需要较多计算资源实现跨模态交互与自采样对比学习

---

## 275. Exploring 6D Object Pose Estimation with Deformation

**arXiv ID:** 2604.06720 | [PDF](https://arxiv.org/pdf/2604.06720v1)

**作者:** Zhiqiang Liu `[一作]` (Xidian University), Yinlin Hu `[通讯]` (MagicLeap)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `40105733-5154-44cd-8090-a8cab9e64b07` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了DeSOPE数据集，用于研究在物体发生形变情况下的6DoF姿态估计，并对现有姿态估计方法在该数据集上的表现进行系统评估。

**💡 创新点**

创新点在于：①首个专门针对形变物体的6DoF姿态估计基准，提供了26类物体的正态与三种形变程度的高精度扫描模型；②构建了133K帧RGB‑D序列和665K帧姿态标注，标注流程采用半自动管线（实例分割→初始姿态→隐式神经重建+优化）；③通过与现有基准对比，揭示形变对方法性能的显著影响，突显该问题的研究价值。

**🔧 技术方法**

技术手段包括：高精度3D扫描（Go!SCAN SPARK）、基于SCFlow2的流引导三维配准、SAM2实例分割、FoundationPose初始姿态估计、隐式神经场重建与光束束优化（Co‑SLAM式），以及在评估时使用SCFlow2、FoundationPose与GenPose三种公开方法。

**📊 数据集**

使用的数据集为DeSOPE：26类日常物体，每类1个正态模型+3个形变模型（共104个实例），共133 000帧RGB‑D图像、665 000帧姿态标注；在模型预训练时借助ShapeNet、Google‑Scanned‑Objects、Objaverse等公开数据。

**📈 对比分析**

与SCFlow2、FoundationPose、GenPose三种方法对比，采用BOP评价指标（VSD、MSSD、MSPD），计算平均召回率（AR）。结果显示：正态模型时AR≈0.8；形变1时约0.6‑0.7；形变2降至≈0.3‑0.4；形变3更低≈0.2‑0.3；人机交互场景进一步降低性能。

**⚠️ 局限性**

局限性：当前方法假设输入与正态模型完全一致，难以适应形变；DeSOPE虽包含多种形变，但仍可能缺乏更复杂的形变模式；标注流程半自动化仍可能存在误差；需研发更鲁棒的形变感知与自适应姿态估计方法。

---

## 276. Stabilization Without Simplification: A Two-Dimensional Model of Software Evolution

**arXiv ID:** 2604.06709 | [PDF](https://arxiv.org/pdf/2604.06709v1)

**作者:** Masaru Furukawa `[一作]` (University of Toyama), Masaru Furukawa `[通讯]` (University of Toyama)

**通讯引用:** 1089 | [OpenAlex ID](https://openalex.org/A5041789225)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了一个图结构的离散时间概率框架，将软件演化中的变更成本拆分为期望负担（结构性负荷）和不确定性（变动性），并证明在满足四条结构与随机假设的情况下可以出现“在不简化的前提下实现稳定”的现象。

**💡 创新点**

首次从二维角度（负担与不确定性）阐释软件演化中的稳定性，给出在结构负荷不下降时仍可实现不确定性下降的理论依据。

**🔧 技术方法**

使用图论描述系统结构，构造线性变更成本模型并通过期望、方差和协方差分析，证明不确定性可独立于负担演化。

**📊 数据集**

未使用任何实验数据集，整个工作为纯理论推导与数学证明。

**📈 对比分析**

无实验比较，主要通过数学推理展示在假设A1–A4下满足ΔB≥0且ΔU<0的条件。

**⚠️ 局限性**

模型极为简化，忽略了语义、团队协作、非线性传播等现实因素；仅使用局部度数作为结构度量，未考虑更丰富的图属性或非高斯噪声。

---

## 277. AudioKV: KV Cache Eviction in Efficient Large Audio Language Models

**arXiv ID:** 2604.06694 | [PDF](https://arxiv.org/pdf/2604.06694v1)

**作者:** Yuxuan Wang `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15055 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 AudioKV 框架，针对大规模音频语言模型（LALMs）在长音频推理时 KV 缓存占用巨大的问题，设计音频意识的 KV 缓存分配与谱域分数平滑方法。

**💡 创新点**

1) 识别并量化音频关键注意力头，按其对音频信息的贡献动态分配 KV 缓存；2) 引入 Spectral Score Smoothing (SSS) 通过 FFT 低通滤波抑制高频噪声，提升重要性分数的连续性与稳定性。

**🔧 技术方法**

音频头识别基于 ASR 词级对齐与注意力统计；KV 缓存分配采用基于头重要性比例的分配公式；SSS 采用 RFFT、能量阈值裁剪、低通滤波及残差混合。

**📊 数据集**

LibriSpeech-long、Multilingual LibriSpeech、KeSpeech、CoVoST2、UltraEval-Audio 四大语音/翻译/问答数据集。

**📈 对比分析**

与 Full KV、SnapKV、AdaKV、PyramidKV、H2O 等基线在多模型（Gemma、Qwen Omni）上对比，AudioKV 在 40% 缓存保留率下，Qwen3‑Omni‑30B 的准确率仅下降 0.45%，且在 ASR、ST、AQA 上普遍优于其他方法。

**⚠️ 局限性**

对高压缩比（低缓存）仍存在潜在重复或语义漂移风险，SSS 参数需调优以避免过度平滑导致信息损失，且方法对模型规模和头数的依赖需进一步验证。

---

## 278. GPAFormer: Graph-guided Patch Aggregation Transformer for Efficient 3D Medical Image Segmentation

**arXiv ID:** 2604.06658 | [PDF](https://arxiv.org/pdf/2604.06658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. KD-MARL: Resource-Aware Knowledge Distillation in Multi-Agent Reinforcement Learning

**arXiv ID:** 2604.06691 | [PDF](https://arxiv.org/pdf/2604.06691v1)

**作者:** Monirul Islam Pavel `[一作]` (Adelaide University), Zehong Jimmy Cao `[通讯]` (Adelaide University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

提出KD-MARL两阶段资源感知知识蒸馏框架，在多智能体强化学习中实现轻量化、异构部署，保持专家级协调。

**💡 创新点**

创新点包括：① 教师引导的优势蒸馏实现无评估器学生训练；② 结构关系与角色对齐蒸馏损失保留多智能体协调结构；③ 异构学生架构与有限观测匹配资源。

**🔧 技术方法**

技术方法：CTDE+MAPPO教师；优势蒸馏、PPO无评估器；结构关系损失、角色对齐损失；KL+交叉熵监督；轻量RNN网络。

**📊 数据集**

使用数据集：SMAC（多地图）和MPE（多代理粒子环境）。

**📈 对比分析**

与MAPPO、QMIX、VDN等基线在FO、LH、LH+A三种观测设定下对比；KD-MARL在保留>90%专家性能的同时，FLOPs下降3.3–28.6×，TPS下降30%+，SMAC 3m/8m等地图维持>90%胜率。

**⚠️ 局限性**

局限性：仍需完整教师训练，缺乏在线自适应；多教师或通信不充分时效果未知；在极端资源极限下协调仍可能衰退。

---

## 280. RASR: Retrieval-Augmented Semantic Reasoning for Fake News Video Detection

**arXiv ID:** 2604.06687 | [PDF](https://arxiv.org/pdf/2604.06687v1)

**作者:** Hui Li `[一作]`, Jinsong Su `[通讯]` (Xiamen University)

**通讯引用:** 4037 | [OpenAlex ID](https://openalex.org/A5066326238)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种检索增强语义推理框架 RASR，用于多域短视频假新闻检测。

**💡 创新点**

创新点包括：①跨实例语义解析与检索模块（CSPR）构建历史关联证据；②域引导多模态推理模块（DGMP）利用领域标签和检索证据驱动大型多模态语言模型；③多视角特征解耦与融合模块（MV-DFF）在门控机制下抑制推理噪声，提升鲁棒性。

**🔧 技术方法**

主要技术包括：跨模态注意力、语义原语检索、检索增强大语言模型推理、信息熵对齐损失、可学习门控融合、以及多模态特征编码（TimeSformer、XLM‑RoBERTa、VGGish）。

**📊 数据集**

使用公开的两大短视频假新闻数据集 FakeSV（中文）和 FakeTT（英文），并在其 9 个领域上进行跨域实验。

**📈 对比分析**

与多种基线（FakingRecipe、SV‑FEND、DOCTOR 等）对比，RASR 在 FakeSV 上平均提升 0.93% 准确率，在跨域、总体检测以及对抗检索噪声等多维度实验中持续领先，表现出更优的域泛化和鲁棒性。

**⚠️ 局限性**

局限性包括：对检索记忆库规模与质量敏感；依赖多模态大语言模型，推理成本高；需要预先标注的领域标签；在极端语义偏差或多模态缺失时仍可能出现误检。

---

## 281. Benchmarking Requirement-to-Architecture Generation with Hybrid Evaluation

**arXiv ID:** 2604.06683 | [PDF](https://arxiv.org/pdf/2604.06683v1)

**作者:** Minxiao Li `[一作]` (Beihang University), Fang Liu `[通讯]` (Beihang University)

**通讯引用:** 21164 | [OpenAlex ID](https://openalex.org/A5027449919)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基准数据集R2ABench，用来评估大语言模型从自然语言需求文档生成软件架构图的能力，并给出了一个多维评估框架；

**💡 创新点**

创新点在于构建真实项目的PRD+PlantUML参考图的数据集，并融合结构图指标、LLM评判、多维评分与反模式检测的混合评估方法；

**🔧 技术方法**

使用了多种顶尖LLM（GPT‑5、Gemini‑2.5‑Pro、Claude‑4.6、DeepSeek‑V3.2、Qwen3‑coder‑480B等）以及MetaGPT、OpenHands等agent框架，并通过LLM‑as‑a‑Judge和图结构分析技术实现评估；

**📊 数据集**

数据集为R2ABench，包含17个Python/Java项目的完整PRD与对应的PlantUML参考图；

**📈 对比分析**

采用结构图指标（SV、节点/边F1、GED等）、LLM评判分数和反模式比例进行多维比较，实验显示模型在实体提取上表现较好但关系推理不足，代码专业模型在边关系上略占优，agent框架提升不稳定；

**⚠️ 局限性**

局限性包括样本量有限（仅17例）、仅涵盖Python/Java、使用贪心解码且缺乏多次采样、评估仍受LLM偏差与推断不确定性的影响。

---

## 282. SkillTrojan: Backdoor Attacks on Skill-Based Agent Systems

**arXiv ID:** 2604.06811 | [PDF](https://arxiv.org/pdf/2604.06811v1)

**作者:** Yunhao Feng `[一作]` (National University of Defense Technology), Wenke Huang `[通讯]` (Nanyang Technological University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5082757218)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对技能抽象层的后门攻击方法SkillTrojan，能够将加密的恶意代码片段嵌入普通可重用的技能包中，并通过触发词在技能执行时重新组装并执行攻击载荷；

**💡 创新点**

创新点在于首次将后门攻击从模型参数/提示转移到技能实现层，利用技能的可重用性与执行链实现持久且隐蔽的攻击，同时提供自动化合成后门技能的通用框架；

**🔧 技术方法**

技术包括：对攻击载荷进行对称加密并分段，注入触发条件的自然语言指令和执行逻辑，使用工具调用轨迹记录的中间通道收集加密片段，最后在触发时完成解密并执行侧效果；

**📊 数据集**

使用了由公开技能市场收集的1200个高流量技能模板，自动生成3000+个后门技能，实验数据集为EHR SQL查询任务以及SWE‑Bench Verified软件工程任务；

**📈 对比分析**

与GCG、AutoDAN、CPA、BadChain、AgentPoison等提示/模型层后门基线相比，在Open‑Weight和Closed‑Weight LLM上，SkillTrojan在保持或略升高的清洁任务准确率（ACC）下，攻击成功率（ASR）最高，尤其在GPT‑5.2‑1211‑Global上达97.2% ASR，且对模型能力的依赖更低；

**⚠️ 局限性**

局限性包括：依赖于技能的可执行性与可信度，攻击仅能在支持工具调用与轨迹记录的系统中实施，对强沙箱或严格的执行审计环境可能无效；此外，攻击仅在触发词被注入后才激活，若触发词被过滤或检测则失效。

---

## 283. LASER: A Data-Centric Method for Low-Cost and Efficient SQL Rewriting based on SQL-GRPO

**arXiv ID:** 2604.06804 | [PDF](https://arxiv.org/pdf/2604.06804v1)

**作者:** Jiahui Li `[一作]` (Zhejiang University), Yunjun Gao `[通讯]` (Zhejiang University)

**通讯引用:** 5480 | [OpenAlex ID](https://openalex.org/A5006238145)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了基于MCTS的慢查询生成器 SQL‑MCTS，并开发了 SQL‑GRPO 训练框架，训练 Qwen3‑8B/14B 小模型实现高效 SQL 重写。

**💡 创新点**

①使用 MCTS + LLM 迭代生成结构复杂且慢的查询；②在 RL 训练中提出 Anchored Group Advantage 与 Complexity‑Adaptive Dynamic Rollout，解决传统 GRPO 在 SQL 语义约束下的优势估计与预算分配问题。

**🔧 技术方法**

Monte Carlo Tree Search、语言模型微调（SFT）、Group Relative Policy Optimization、结构化奖励函数、语法验证与自我纠错机制。

**📊 数据集**

SQL‑MCTS（1.17 万条慢查询，来自 TPC‑DS 模式），并在 TPC‑DS、TPC‑H、DSB、Calcite、MySQL 等基准上评测。

**📈 对比分析**

与规则基、LLM 基础重写、DeepSeek‑R1、GPT‑4o 等做对比。LASER‑14B 在 TPC‑DS 的平均查询耗时从 69.73 s 降至 13.03 s，等价率 90%；在未见模式 TPC‑H、Calcite 仍保持 31–35 s 的高效，且推理时间与成本显著低于 API‑依赖模型。

**⚠️ 局限性**

仍需手工构造慢查询种子，训练依赖大规模 GPU，模型对数据库特定优化器的细粒度学习仍有限；在极端大规模数据或非 SQL‑92 语法的系统中，效果尚未验证。

---

## 284. MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization

**arXiv ID:** 2604.06798 | [PDF](https://arxiv.org/pdf/2604.06798v1)

**作者:** Zhixiong Zhao `[一作]` (Houmo AI), Dawei Yang `[通讯]` (Houmo AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种专为Mixture-of-Experts（MoE）大模型设计的二值化框架MoBiE，解决MoE模型在二值化时的专家冗余、重要性估计失准与专家迁移等问题。

**💡 创新点**

三大创新点：1）交叉专家联合分解（CEJD）通过SVD提取共享高精度基底，仅对专家投影进行二值化；2）全局损失对齐重要性（GLAS）将全局梯度注入Hessian，提升重要性评估的任务相关性；3）零空间引导专家迁移抑制（NGES）通过隐式零空间约束限制量化误差对路由的影响。

**🔧 技术方法**

主要技术包括SVD分解、Hessian（局部与全局）重要性评估、零空间投影约束、基于梯度的量化损失函数以及无监督后训练（PTQ）方法。

**📊 数据集**

使用多种MoE大模型（OLMoE、DeepSeek、Qwen3-30B-A3B、Qwen3-Next-80B-A3B、GPT-OSS-20B等）以及WikiText-2、Arc-Challenge、Arc-Easy、HellaSwag、LAMBADA、PIQA、WinoGrande、MMLU、GSM8K、HumanEval等数据集进行评估。

**📈 对比分析**

与BiLLM、ARB-LLM、AWQ、GPTQ、MoEQuant、NoWag等基线比较，MoBiE在多模型、多任务上均显著优于对手，实例如Qwen3-30B-A3B：降低52.2%困惑度、提升43.4%零样本平均准确率，推理速度提升超过2倍，内存占用显著下降。

**⚠️ 局限性**

局限性包括：实现复杂度提升（需SVD、梯度注入、零空间迭代）；对校准数据分布敏感，若部署时分布偏移可能导致性能下降；尽管降低了专家迁移风险，但在极端量化噪声下仍可能出现稀有的专家误分情况。

---

## 285. The Rhetoric of Machine Learning

**arXiv ID:** 2604.06754 | [PDF](https://arxiv.org/pdf/2604.06754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 286. FVD: Inference-Time Alignment of Diffusion Models via Fleming-Viot Resampling

**arXiv ID:** 2604.06779 | [PDF](https://arxiv.org/pdf/2604.06779v1)

**作者:** Shivanshu Shekhar `[一作]` (University of Illinois Urbana-Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 25329 | [OpenAlex ID](https://openalex.org/A5100378779)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为Fleming‑Viot Diffusion（FVD）的推理时对齐方法，利用粒子系统在扩散采样中替代多项式重采样以保持多样性并提升奖励对齐效果。

**💡 创新点**

创新点在于将Fleming–Viot出生‑死亡机制（独立Bernoulli死亡与无权重捐赠）引入扩散采样，并通过Robbins–Monro自适应控制器动态调节选择压力，以避免粒子消亡导致的模式崩塌。

**🔧 技术方法**

核心技术包括粒子系统（Fleming‑Viot过程）、基于Tweedie近似的奖励代理、DDIM无噪声和噪声重生、以及自适应λ更新。

**📊 数据集**

实验数据集涵盖MNIST、CIFAR‑10的类条件后验采样以及Stable Diffusion在DrawBench和美学优化任务上的文本到图像生成。

**📈 对比分析**

与现有SMC、梯度引导、搜索与价值函数方法比较，FVD在FID、MMD、奖励均值及多样性指标上均优于FKD、DTS等强基线，并在相同NFE预算下实现约66倍的速度提升。

**⚠️ 局限性**

局限性包括仅针对单目标奖励、依赖Tweedie估计（易受早期噪声影响）、未扩展至流匹配或一致性模型、以及未处理多目标竞争奖励的情形。

---

## 287. Understanding Data Collection, Brokerage, and Spam in the Lead Marketing Ecosystem

**arXiv ID:** 2604.06759 | [PDF](https://arxiv.org/pdf/2604.06759v1)

**作者:** Yash Vekaria `[一作]` (UC Davis), Zubair Shafiq `[通讯]` (UC Davis)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对美国健康保险领域的潜在客户生成（lead marketing）生态系统进行端到端测量，分析从网站表单收集、数据经纪和消费者营销的完整链路，评估隐私泄露与垃圾电话/短信/邮件的风险。

**💡 创新点**

首次将表单收集、数据经纪与营销沟通三大组件统一实证，揭示敏感健康信息被广泛外泄、经纪方夸大属性、以及高频无同意的电话/短信/邮件骚扰；并系统评估 opt‑out 机制的实际效果。

**🔧 技术方法**

使用自研基于 Puppeteer 的浏览器爬虫捕获网络流量；构造 105 个合成用户并控制 200 个电话号和邮箱；通过 VoIP 调用、短信 API、邮件抓取记录接收；对 BBB 投诉进行主题分析；使用线性回归和差分法评估 opt‑out 效果。

**📊 数据集**

收集 105 家健康保险潜在客户生成网站、200 个控制电话号/邮箱、三大平台（QuoteWizard、NextGen Leads、Aged Lead Store）共 396 条潜在客户记录、7,432 条 BBB 投诉与评价，以及所有抓取的网络流量与第三方请求。

**📈 对比分析**

通过统计回归与差分法对 opt‑out 影响进行比较，发现电话 opt‑out 在 60 天内显著降低通话量（p<0.001），短信/邮件效果有限；对通话频率、重复次数、时段与州法规进行比对，发现约 20% 超过州限；实验结果与 BBB 投诉数据高度吻合，验证方法可信。

**⚠️ 局限性**

仅关注健康保险垂直，未记录通话内容，使用合成用户限制了对真实用户行为的评估；未测试短信 opt‑out 机制；可能漏掉部分第三方服务；未覆盖多州法规更新后的变化；实验规模受限于资源，未能完整覆盖所有潜在客户生成网站。

---

## 288. FlowInOne:Unifying Multimodal Generation as Image-in, Image-out Flow Matching

**arXiv ID:** 2604.06757 | [PDF](https://arxiv.org/pdf/2604.06757v1)

**作者:** Junchao Yi `[一作]` (University of Electronic Science and Technology of China), Alex Jinpeng Wang `[通讯]` (Central South University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5061458314)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种全视觉流匹配框架，将文本、布局、编辑指令等多模态输入统一为视觉提示，实现图像输入–图像输出的端到端生成与编辑。

**💡 创新点**

创新点在于：① 把所有模态渲染成图像提示，消除跨模态对齐瓶颈；② 使用流匹配学习连续视觉演化，避免噪声调度和多任务分支；③ 引入双路径空间自适应调制机制，兼顾结构保持与指令遵循。

**🔧 技术方法**

技术手段包括：视觉编码器（Janus‑Pro‑1B + SigLIP），文本图像 VAE、冻结图像 VAE，流匹配网络（时间相关速度场），双路径自适应调制（自注意力+跨注意力+门控网络）以及对齐评估与多模态 VLM 自动评测。

**📊 数据集**

使用了 5 M 条视觉提示对的数据集——Visual Prompt Dataset，涵盖文本生成、图像编辑、边界框编辑、视觉标记编辑、涂鸦编辑以及物理力学与轨迹理解；并构建 Visual Prompt Benchmark 进行四维指标评估。

**📈 对比分析**

与 OmniGen2、Qwen‑Image‑Edit‑2509、FLUX.1-Kontext‑dev 以及商业模型 Nano Banana 等进行对比；在 GPT‑5.2、Qwen3.5、Gemini‑3 以及人工评测中取得最高通过率（最高 50.3%）并在空间精度、视觉真实性等指标上显著优于对手，表明在统一生成任务上实现了 SOTA。

**⚠️ 局限性**

局限性包括：① 仍主要依赖高质量视觉提示，对纯文本输入的处理需先转换为图像；② 对高分辨率大尺寸图像的推理需要更大显存和计算资源；③ 对复杂多任务的长期训练可能面临梯度消失或模式崩溃，需进一步优化稳定性。

---

## 289. Beyond Pessimism: Offline Learning in KL-regularized Games

**arXiv ID:** 2604.06738 | [PDF](https://arxiv.org/pdf/2604.06738v1)

**作者:** Yuheng Zhang `[一作]` (University of Illinois Urbana Champaign), Nan Jiang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在离线环境下学习 KL 正则化的两人零和博弈的算法，证明其在满足单边覆盖条件时可以在无恐惧（pessimism）机制的前提下实现 𝒪(1/n) 的样本复杂度，并进一步给出了可扩展的自对弈（self‑play）优化算法同样达到该快收敛速率。

**💡 创新点**

① 用几何结构（KL 正则化带来的强凸性与平滑性）和零和博弈的反对称性，完成了不依赖于 pessimistic 置信区间的全新分析框架；② 通过单边估计误差与 Nash 均衡的稳定性关联，实现了 𝒪(1/n) 的统计快率；③ 设计了可线性迭代次数的自对弈算法，兼顾了计算效率与统计性能。

**🔧 技术方法**

利用 KL 正则化的软最大（softmax）最佳回应映射、凸/凹函数的强单调性、KL 散度与 𝓁∞ 之间的上界、以及对数似然（log‑softmax）推导的估计误差传播；在算法层面应用了最小二乘估计、拉格朗日/镜像下降（mirror‑descent）以及可变学习率的自对弈更新。

**📊 数据集**

本工作为理论分析，未使用具体离线数据集；若需实验验证，可取公开的 RLHF 或对弈数据集（如 OpenAI 的人类偏好标注数据）进行模拟，但论文重点在于理论保证。

**📈 对比分析**

与现有基于 pessimism 的方法（提供 𝒪(1/√n) 率）进行对比，证明了无恐惧算法在样本复杂度上实现了两倍速（快到 𝒪(1/n)）。自对弈算法在迭代次数线性于样本量的情况下，实测收敛速度与理论一致，可实现与直接最小化极大估计同等的统计性能。

**⚠️ 局限性**

（1）需要满足严格的单边覆盖（unilateral concentrability）条件；（2）理论假设对数类函数近似可满足，但在高维连续空间中实现可能受限；（3）虽然消除了 pessimism，但实际实现仍需高质量的最小二乘估计，若数据噪声较大或函数类容量大，可能导致估计误差放大；（4）对齐实际 LLM 的环境时，KL 正则化的超参数 η 需要仔细调优，且对齐目标不一定完全可通过两人零和博弈建模。

---

## 290. Evaluating Repository-level Software Documentation via Question Answering and Feature-Driven Development

**arXiv ID:** 2604.06793 | [PDF](https://arxiv.org/pdf/2604.06793v1)

**作者:** Xinchen Wang `[一作]` (Harbin Institute of Technology), Chao Peng `[通讯]` (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向仓库级的软件文档评估基准，包含 4,170 条高质量 PR 基础的功能检测、定位和完善三类问答任务，并通过 LLM 进行自动评测。

**💡 创新点**

①以功能驱动的问答评估方式代替传统 LLM-as-a-judge；②利用 PR 的完整上下文（代码、issue、依赖等）生成可检验的功能描述；③提供可复现的、面向全仓库的评测框架。

**🔧 技术方法**

使用大语言模型（Claude‑Sonnet‑4、ChatGPT‑4.1、Gemini‑2.5‑Pro）进行功能描述生成和问答；采用链式思考（CoT）和检索增强（SFR‑Embedding‑Code‑400M_R、Top‑K 文档片段）；实现了多维 PR 过滤与依赖分析。

**📊 数据集**

从 12 个 SWE‑Bench 公开仓库（2023 版本）中抓取 177.4k PR，经过多步过滤后得到 4,170 条高质量 PR；随后构造的 QA 数据集覆盖 3 种任务。

**📈 对比分析**

在 6 种文档生成方法（H‑Written、Chat、DeepWiki、AutoDoc、DocAgent、RepoAgent）上评测，RepoAgent 在所有指标上排名第一；相比无文档 baseline，所有方法均提升 5–32%；细粒度、全局上下文的文档方法表现显著优于粗粒度或缺乏全局信息的方法；文档质量提升可使 issue‑solving 工具 SWE‑Agent 的成功率提高 8–20%。

**⚠️ 局限性**

仅覆盖 12 个流行开源仓库，结果可能不具普适性；评测依赖 LLM，存在随机性，需多次复现；PR 过滤规则和功能描述生成仍有人工校验成本。

---

## 291. Cognitive Loop of Thought: Reversible Hierarchical Markov Chain for Efficient Mathematical Reasoning

**arXiv ID:** 2604.06805 | [PDF](https://arxiv.org/pdf/2604.06805v1)

**作者:** Jia-Chen Zhang `[一作]` (East China Normal University), Yu-Jie Xiong `[通讯]` (Shanghai University Of Engineering Science)

**通讯引用:** 63797 | [OpenAlex ID](https://openalex.org/A5087717847)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Cognitive Loop of Thought (CLoT) 框架，将链式推理转化为可逆层级马尔可夫链，加入自我逆向验证和层级剪枝以提升推理准确性与效率。

**💡 创新点**

创新点在于：① 逆向验证机制，使模型在每个层级检查推理的可逆性，减少误差传播；② 可逆层级马尔可夫链结构，兼顾全局与局部推理；③ 层级剪枝策略，在上层验证通过后跳过下层冗余验证；④ 构建 CLoT‑Instruct 数据集，支持指令微调。

**🔧 技术方法**

技术手段包括：可逆层级马尔可夫链（RHMC）、逆向验证、层级剪枝、链式推理（CoT）与其变体、基于 token 消耗的高效实现。

**📊 数据集**

使用数学推理数据集（AddSub、GSM8K、SVAMP、AQuA、MATH），逻辑推理数据集（AQuA、CSQA），常识推理数据集（CommonsenseQA）。

**📈 对比分析**

与 CoT、CoT‑SC、C‑CoT、ISP‑CoT、AR、Thought Rollback 等基线对比，CLoT 在 GPT‑4o‑mini、GPT‑4、DeepSeek‑V3 上平均准确率分别提升至 89.6%、90.5%、92.7%；在 AddSub 上达到 99.0%，在 GSM8K 上 94.6%，在 AQuA 上 85.8%；同时 token 消耗仅 136k，较传统方法降低 41.8%。

**⚠️ 局限性**

局限性：① 需要模型具备较强的逆向推理能力，小模型可能无法正确执行逆向验证；② 评估多集中于确定性推理任务，对开放式、主观性任务的适用性有限。

---

## 292. Riemann-Bench: A Benchmark for Moonshot Mathematics

**arXiv ID:** 2604.06802 | [PDF](https://arxiv.org/pdf/2604.06802v1)

**作者:** Suhaas Garre `[一作]` (Surge AI), Edwin Chen `[通讯]` (Surge AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个专门用于评估 AI 在研究级数学推理能力的私有基准集，包含25道由顶尖数学家精心挑选、经过双盲专家验证、可编程检验的极难问题。

**💡 创新点**

创新点在于：① 私有、无污染的基准设计；② 双盲从零验证保证问题难度与答案唯一性；③ 采用无约束、完整工具支持的评估方式，真实模拟科研环境；④ 对比现有竞赛级基准揭示研究级数学的巨大难度。

**🔧 技术方法**

技术主要包括：AI模型作为开放式研究代理，配备 Python 解释器、搜索引擎及自由推理；使用无偏估计器对100次独立运行结果进行统计；以及构建可编程的答案验证器。

**📊 数据集**

数据集为25道研究级问题，来源于常春藤数学教授、研究生及 IMO 冠军，全部保持私有。

**📈 对比分析**

通过与现有竞赛级基准（如 IMO、AIME、MATH）对比，发现所有领先模型在该基准上的 Pass@1 低于 10%，显示研究级推理远落后。

**⚠️ 局限性**

局限性包括：基准仅含 25 道题，规模有限；私有性导致外部复现困难；评估依赖模型访问外部工具，难以统一衡量模型内在推理能力；对模型的错误解释依赖人工分析。

---

## 293. Video-guided Machine Translation with Global Video Context

**arXiv ID:** 2604.06789 | [PDF](https://arxiv.org/pdf/2604.06789v1)

**作者:** Jian Chen `[一作]` (Shenzhen University), XiangHua Fu `[通讯]` (Shenzhen Technology University)

**通讯引用:** 1922 | [OpenAlex ID](https://openalex.org/A5037794437)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种全局视频上下文辅助的多模态翻译框架，利用语义检索构造与目标字幕语义相关的视频段集合，并通过文本感知视频段选择器和区域感知跨模态注意机制对视觉信息进行细粒度对齐，从而提升长视频场景的翻译质量。

**💡 创新点**

①全局检索结合邻近上下文扩展，构建与字幕语义相关的多段视频集合；②文本感知视频段注意力选择器，动态选取重要段并融合未选段上下文；③区域感知跨模态注意机制，实现区域级视觉与文本对齐；④双向跨模态注意与门控融合，进一步强化语义交互。

**🔧 技术方法**

预训练语义编码器（CLIP）+FAISS检索、I3D视觉特征提取、Attention机制（多头注意、门控融合）、Transformer解码器、梯度训练与混合精度优化。

**📊 数据集**

TopicVD（256部纪录片，122,930中英字幕对）以及BigVideo子集。

**📈 对比分析**

在TopicVD上与Text‑only NMT、Image‑MMT、Segment‑Level Video‑MMT和BigVideo基线对比，BLEU从19.14提升至30.47（+11.33），METEOR从33.95提升至38.96（+5.01）。在BigVideo子集上BLEU与BigVideo基线相近（46.78 vs 47.05），表明在长视频场景下全局上下文模型显著提升翻译效果。

**⚠️ 局限性**

依赖检索质量，检索误差会导致性能波动；计算开销较大（约1.22 GFLOPs/样本）；在短视频或缺乏丰富上下文的视频场景中优势不明显；超参数（P、w、K）对效果敏感；实验仅在纪录片类视频上验证，泛化性待进一步评估。

---

## 294. RichMap: A Reachability Map Balancing Precision, Efficiency, and Flexibility for Rich Robot Manipulation Tasks

**arXiv ID:** 2604.06778 | [PDF](https://arxiv.org/pdf/2604.06778v1)

**作者:** Yupu Lu `[一作]` (University of Hong Kong), Jia Pan `[通讯]` (University of Hong Kong)

**通讯引用:** 9152 | [OpenAlex ID](https://openalex.org/A5076812698)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种名为RichMap的网格化到达性地图，用来快速预测机器人末端执行器在空间中的可达性，并能为跨机器人策略迁移提供工作空间相似性评估。

**💡 创新点**

创新点在于（1）利用理论容量上界对SO(3)的球面分布进行最优采样，保证高精度且存储量可控；（2）设计异步CPU‑GPU流水线，实现百万级姿态的高吞吐量构建；（3）将空间与方向分离的结构用于多任务扩展，如工作空间相似度量与扩散策略能量引导。

**🔧 技术方法**

主要技术包括前向运动学采样、碰撞检测、球面距离（geodesic distance）判断、GPU批量插入、异步生产者消费者流水线、最大均值差（MMD）相似度计算以及基于梯度的能量引导。

**📊 数据集**

验证使用了四个工业机器人（Franka Panda、KUKA iiwa7、UR5e、Kinova Gen3）的前向运动学采样生成的约200万测试姿态；跨机器人策略迁移实验采用xArm6作为源机器人，UR3/UR5/UR10作为目标机器人。

**📈 对比分析**

与RM4D和Capability Map比较，RichMap在Δ=0.05m下可达性准确率>97%，误报率≤0.05%，批量查询1e6姿态平均耗时≈2–3 µs；在Δ=0.02m时误报率降至≈1.6%，构建时间约1天；在块推实验中，能量引导提升平均成功率≈10.97%。

**⚠️ 局限性**

局限性包括能量引导仅基于3D位置而不考虑方向，导致对姿态敏感任务的适应性不足；当前仅处理静态可达性，未覆盖动态运动学或碰撞场景；模型在极端稀疏/高维空间中的容量上界仍有进一步优化空间。

---

## 295. Sparse-Aware Neural Networks for Nonlinear Functionals: Mitigating the Exponential Dependence on Dimension

**arXiv ID:** 2604.06774 | [PDF](https://arxiv.org/pdf/2604.06774v1)

**作者:** Jianfei Li `[一作]` (Ludwig-Maximilians-Universität München), Gitta Kutyniok `[通讯]` (Ludwig-Maximilians-Universität München)

**通讯引用:** 16648 | [OpenAlex ID](https://openalex.org/A5090767423)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

提出稀疏感知的卷积-全连接编码器-解码器框架，用于从无限维函数空间学习非线性泛函，并证明随机采样足以实现稳定恢复。

**💡 创新点**

将CNN作为稀疏逼近器；在泛函学习中给出维度几乎不相关的误差上界；随机采样实现全局离散化；对特定函数空间给出闭式误差速率；提供稀疏编码与神经网络相结合的理论分析。

**🔧 技术方法**

稀疏编码（压缩感知、L1/ Basis Pursuit），卷积神经网络与全连接深度网络，Riesz基与互相一致性分析，随机采样概率论，Sobolev/ Besov/混合光滑度逼近理论。

**📊 数据集**

文章未给出具体实验数据集，主要是理论推导；若有实验则可能使用合成函数或标准算子学习数据，但未明确。

**📈 对比分析**

通过误差上界与已知结果比较，误差只依赖于参数数K的对数，显著降低对维度d的依赖；相比传统方法（如(logK)^-r/d）取得更快的收敛速率；没有实验性能指标。

**⚠️ 局限性**

缺乏实验验证；假设字典满足Riesz与一致性条件；随机采样需足够大样本才能保证离散化；对泛化误差与样本误差分析不充分；实现复杂度与训练稳定性未讨论。

---

## 296. MemoryDiorama: Generating Dynamic 3D Diorama from Everyday Photos for Memory Recall

**arXiv ID:** 2604.06773 | [PDF](https://arxiv.org/pdf/2604.06773v1)

**作者:** Keiichi Ihara `[一作]` (University of Colorado Boulder), Ryo Suzuki `[通讯]` (University of Colorado Boulder)

**通讯引用:** 1835 | [OpenAlex ID](https://openalex.org/A5103097262)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本论文提出了一种利用生成式 AI 在混合现实中将个人照片转化为动态三维布景（动态三维情景），以增强自传式记忆的检索效果。

**💡 创新点**

创新点在于（1）提出“增强记忆提示”概念，将生成式 AI 生成的情境信息与原始照片相结合，保留真实记忆根基；（2）设计五层上下文提示（地理、物体、人物、灯光、粒子）构成的动态三维布景；（3）将 LLM 场景分析、图像转 3D、纹理合成、路径规划等技术集成到 MR 系统中。

**🔧 技术方法**

使用技术包括 Gemini LLM 进行场景描述与文本提示、SAM 3D 进行图像转 3D、Nano Banana 2 进行纹理与粒子生成、Cesium 与 Google Photorealistic 3D Tiles 在 Unity 中构建地理基础、Unity 开发 MR 应用并部署到 Meta Quest 3。

**📊 数据集**

实验数据来自受试者自行提供的个人照片集合，每个事件 5 张照片；技术可行性测试使用 25 组照片（3 位作者），用于评估管线成功率与运行时间。

**📈 对比分析**

通过 18 名受试者的三组内实验（Photo‑Only、Static Diorama、Dynamic Diorama），采用重复测量 ANOVA 与 Holm 校正的配对 t 检验。结果显示动态布景显著提升内部细节（p=.0085）、外部细节（p=.0313）和在‑线细节（p=.0003），并在感知细节与视觉生动度上高于其他两组；工作负荷无显著差异。

**⚠️ 局限性**

主要局限包括：① 每个条件使用不同记忆事件，可能混入事件本身的可记忆性差异；② 只评估完整系统效果，未拆分各层提示的独立贡献；③ MR 环境并非日常回忆方式；④ 未评估记忆准确性与错误记忆风险；⑤ 仅关注视觉模式，未扩展嗅觉、触觉等多模态提示。

---

## 297. Geometric Properties of the Voronoi Tessellation in Latent Semantic Manifolds of Large Language Models

**arXiv ID:** 2604.06767 | [PDF](https://arxiv.org/pdf/2604.06767v1)

**作者:** Marshall Brett `[一作]` `[通讯]` (MARS Labs), Marshall Brett (MARS Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Qwen3.5-4B模型的Voronoi图进行后置几何优化（Margin Refinement Procedure，MRP），验证并压缩表达性间隙；

**💡 创新点**

提出Fisher信息距离优化与直接边距最大化相比较，发现后者在提高边距的同时损伤更大，而前者能在保持指标不变的情况下持续改进；

**🔧 技术方法**

采用Voronoi边距定义、线性尺度律回归、Fisher信息距离损失、top‑k选取与损失门控等技术；

**📊 数据集**

使用WikiText‑103验证集进行边距、token‑level与基准评估，数据包含约256K个位置；

**📈 对比分析**

在多种λ_MRP取值下对比两种MRP，Fisher在λ=0.6时可提升整体token准确率≈+2%且基准平均保持在0.632左右，边距提升约27%；

**⚠️ 局限性**

局限性在于改进主要集中于高频结构化token，低频内容词和实体词的收益不明显，且在更高λ时会产生更大集中化损失；

---

## 298. TeamLLM: A Human-Like Team-Oriented Collaboration Framework for Multi-Step Contextualized Tasks

**arXiv ID:** 2604.06765 | [PDF](https://arxiv.org/pdf/2604.06765v1)

**作者:** Xiangyu Wang `[一作]` (East China Normal University), Chanjin Zheng `[通讯]` (East China Normal University)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5008365827)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TeamLLM——一种基于人类团队角色分工与三阶段协作流程的多 LLM 合作框架，并构建了 CGPST 基准，系统评估并验证了该框架在多步情境任务中的效果。

**💡 创新点**

创新点包括：①将 Belbin 团队角色理论引入 LLM 合作，赋予每个 LLM 独特职责；②设计三阶段（任务启动、视角共享、共识构建）交互流程，保证过程可控；③构建涵盖情境 grounding、流程结构、过程评估、多维度评估的 CGPST 基准，填补了现有单步评测的不足。

**🔧 技术方法**

采用团队角色分配（Co‑ordinator、Plant、Monitor‑Evaluator、Implementer）、多轮交互、双变量上下文管理（log_history 与 step_history）以及基于 Prompt 的角色扮演，结合大规模 LLM（如 GPT‑4o、GPT‑5、Claude‑4‑Opus 等）进行实验。

**📊 数据集**

使用自定义的 CGPST 基准，包含 10 个未来情境（FS1–FS10），每个情境包含 6 步、若干子任务，评估维度涵盖发散性、聚合性、批判性、逻辑性及综合问题解决等六大类。

**📈 对比分析**

与单 LLM baseline（不使用角色或协作）对比，采用 Wilcoxon 符号秩检验评估显著性。实验显示 TeamLLM 在 7/10 模型上显著提升（p=0.01），平均分提升 12.7 分，尤其在发散思维（Step‑1、3）和行动方案制定（Step‑6）上获得最大提升。

**⚠️ 局限性**

局限性：①评测依赖 15 位专家双盲评分，人工成本高且难以扩展；②框架采用固定的四角色与预设流程，未验证在其他任务类型或领域的通用性；③在部分模型的逻辑推理步骤中可能引入冲突，导致性能下降。

---

## 299. ARuleCon: Agentic Security Rule Conversion

**arXiv ID:** 2604.06762 | [PDF](https://arxiv.org/pdf/2604.06762v1)

**作者:** Ming Xu `[一作]` (National University of Singapore), Jiaheng Zhang `[通讯]` (National University of Singapore)

**通讯引用:** 57442 | [OpenAlex ID](https://openalex.org/A5100425709)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向安全信息与事件管理（SIEM）规则的跨平台自动转换框架，帮助安全专家无需手工重写规则即可完成从一种SIEM到另一种SIEM的规则迁移。

**💡 创新点**

创新点在于：①设计了统一的中间表示（IR）将各厂商规则抽象为语义层；②引入Agentic Retrieval‑Augmented Generation（RAG）迭代检索官方文档来纠正语法与语义误差；③通过Python可执行检查验证源规则与目标规则的功能一致性，形成闭环迭代。

**🔧 技术方法**

技术主要包括：大型语言模型（GPT‑5、DeepSeek‑V3、LLaMA‑3）进行规则生成；Agentic RAG 过程实现文档检索与自适应改正；Python生成的执行代码用于功能验证；IR 结构化规则语义；以及对等价性评估的 CodeBLEU、嵌入相似度与逻辑槽一致度指标。

**📊 数据集**

使用了来自五大主流 SIEM 平台（Splunk、Microsoft Sentinel、IBM QRadar、Google Chronicle、RSA NetWitness）的规则数据集，共计约 1,492 对源‑目标规则对，规则来源包括官方文档和公开仓库。

**📈 对比分析**

通过对比基线（无 IR、无 RAG、无功能验证的直接 LLM 翻译）与本框架，实验显示在 CodeBLEU、嵌入相似度、逻辑槽一致度三个指标上平均提升约 9–15%；在执行有效率上达 90% 以上；尽管整体算力和时间成本比基线高约 10‑20 倍，但在安全领域的高准确性需求下被认为是可接受的。

**⚠️ 局限性**

局限性包括：①对复杂状态机或高级时间窗口表达仍易出现错误；②文档检索质量高度依赖官方文档的完整性和可访问性；③在低资源模型下性能下降；④整体流程仍需要人工验证后才能投入生产，缺乏完全无监督的自动化。

---

## 300. StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference

**arXiv ID:** 2604.06746 | [PDF](https://arxiv.org/pdf/2604.06746v1)

**作者:** Zhirui Chen `[一作]` (University of Chinese Academy of Sciences), Ling Shao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 69389 | [OpenAlex ID](https://openalex.org/A5082634513)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了StructKV框架，用于在长上下文推理中压缩KV缓存并提升推理效率。

**💡 创新点**

核心创新包括全局入度中心性聚合、动态枢轴检测以及计算与内存解耦的结构传播。

**🔧 技术方法**

使用了多层注意力权重聚合、信息熵与稀疏度动态监测、FlashAttention‑2 等技术。

**📊 数据集**

实验基于LongBench和RULER数据集，涵盖多模型（LLaMA‑3.1、Qwen‑2.5等）及不同上下文长度。

**📈 对比分析**

与FastKV、StreamingLLM、SnapKV等基线比较，StructKV在相同10% KV预算下平均分数提升约1‑2点，且在128K token级别保持高准确率，预填充速度提升约1.9×。

**⚠️ 局限性**

主要局限在于未验证百万token级别稳定性，模型仅在稠密Transformer上测试，且对低带宽硬件的优化尚未完成。

---

## 301. TEC: A Collection of Human Trial-and-error Trajectories for Problem Solving

**arXiv ID:** 2604.06734 | [PDF](https://arxiv.org/pdf/2604.06734v1)

**作者:** Xinkai Zhang `[一作]` (Renmin University of China), Qingyao Ai `[通讯]` (Tsinghua University)

**通讯引用:** 4542 | [OpenAlex ID](https://openalex.org/A5089655391)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个可记录网页搜索任务中多次试错轨迹和错误反思的Chrome扩展平台 Trial‑and‑Error Collection，并基于该平台构建了包含5,370次试验、58个开放式问答任务、41,229网页、46名参与者的数据集

**💡 创新点**

首次同时捕获多次试错行为和结构化错误反思的双维度数据，并提供可重播注释工作流与端到端系统，填补了现有数据集的空白

**🔧 技术方法**

使用Chrome扩展与rrweb记录DOM、交互事件和鼠标轨迹；Django后端管理任务与数据；Replay‑based annotation workflow收集错误诊断与纠正计划；GPT‑4o进行答案评判；多种LLM基线（Vanilla LLM、RAG、Vanilla Agent、Browser Agent）用于对比实验

**📊 数据集**

基于该平台构建的 Trial‑and‑Error Collection 数据集：5,370次试验、58个问题、41,229网页、46名参与者

**📈 对比分析**

将四种LLM基线与人类在SR@1、SR@5、Recovery Rate、Avg T四项指标对比；人类在首次试验相当，但在错误后恢复率（74.5% vs. 50%）和平均试验次数上显著优于LLM

**⚠️ 局限性**

仅覆盖开放式问答任务，数据规模有限；错误反思的自动化利用仍受限；评判依赖GPT‑4o，可能引入评估偏差

---

## 302. Evaluating LLM-Based 0-to-1 Software Generation in End-to-End CLI Tool Scenarios

**arXiv ID:** 2604.06742 | [PDF](https://arxiv.org/pdf/2604.06742v1)

**作者:** Ruida Hu `[一作]` (Harbin Institute of Technology), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 31096 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CLI-Tool-Bench benchmark，用黑盒差分测试评估LLM从零开始生成完整可执行CLI工具的能力；

**💡 创新点**

首次构建结构无关、端到端的CLI生成评测体系，采用自动化的模式化任务合成与多层次等价度量，打破传统预置结构与单元测试的局限；

**🔧 技术方法**

利用LLM进行命令模式抽取与模糊测试、Docker隔离环境的黑盒差分评测、基于执行、输出与系统副作用的多层等价度量、LLM‑as‑judge等技术；

**📊 数据集**

从GitHub筛选100个真实世界的Python、JavaScript、Go CLI仓库，按难度、语言和应用场景分层；

**📈 对比分析**

将7大LLM与2个代理框架（OpenHands、Mini‑SWE‑Agent）组合共14个实验，评测Build、Exec、EM、FM、SM等指标；最高Semantic Match分数低于43%，展示了LLM从零生成完整CLI工具的挑战；

**⚠️ 局限性**

仅覆盖三种主流语言，评测过程受LLM随机性影响，且黑盒差分侧重行为等价，无法捕捉代码质量与安全性等非功能属性；

---

## 303. FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift

**arXiv ID:** 2604.06795 | [PDF](https://arxiv.org/pdf/2604.06795v1)

**作者:** Huy Q. Le `[一作]` (Kyung Hee University), Choong Seon Hong `[通讯]` (Kyung Hee University)

**通讯引用:** 22983 | [OpenAlex ID](https://openalex.org/A5034052371)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedDAP 框架，针对联邦学习中的域漂移问题，构建域特定全局原型，并在本地训练中采用双重对齐策略（域内一致性对齐 + 跨域对比学习）来提升模型的跨域泛化能力。

**💡 创新点**

创新点：
1) 引入域特定全局原型，保留类别语义与域信息，避免传统方法因域不一致导致的语义稀释。
2) 采用余弦相似度加权融合的聚合机制，提升原型质量。
3) 双重对齐策略——域内对齐保证局部一致性，跨域对比学习加强域不变特征，从而显著提升跨域性能。

**🔧 技术方法**

技术手段：
- 原型学习（类中心）与余弦相似度权重聚合；
- 余弦对齐损失（DPA）与跨域对比损失（CPCL）；
- 标准交叉熵损失；
- 联邦平均（FedAvg）框架下的本地与全局通信；
- ResNet-10/18 作为特征提取器。

**📊 数据集**

使用三大多域数据集：DomainNet（6个域，10类），Office-10（4个域，10类），PACS（4个域，7类）。

**📈 对比分析**

与 FedAvg、FedProx、MOON、COPA、FedGA、FedProto、FPL、FedPLVM、FedRDN 等 SOTA 方法进行比较。实验结果显示，FedDAP 在 DomainNet 上平均提升 5.61%、Office-10 上提升 15.06%、PACS 上提升 6.86%。此外，FedDAP 在收敛速度、特征可视化（t-SNE）和离线跨域泛化测试中均优于基线，证明其在域漂移场景下的显著优势。

**⚠️ 局限性**

局限性：
1) 需要先验的域标签，无法直接处理无标签域或未知域；
2) 需要额外的原型计算与对比学习开销，通信与计算成本相对传统 FedAvg 较高；
3) 对温度参数、λ1/λ2 等超参数敏感，需针对不同数据集进行调优；
4) 在极端域不平衡或极少量域样本的场景下，域特定原型可能不足以代表域分布。

---

## 304. Insights from Visual Cognition: Understanding Human Action Dynamics with Overall Glance and Refined Gaze Transformer

**arXiv ID:** 2604.06783 | [PDF](https://arxiv.org/pdf/2604.06783v1)

**作者:** Bohao Xing `[一作]` (Lappeenranta-Lahti University of Technology LUT), Heikki Kälviäinen `[通讯]` (Lappeenranta-Lahti University of Technology LUT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种双路径Transformer模型（OG‑ReG），通过整体扫视（Overall Glance）提取粗粒度时空信息，再通过细化注视（Refined Gaze）补充细节，提升视频动作识别性能。

**💡 创新点**

创新点在于：① 将整体扫视转化为仅在空间上下采样的自注意力（Spatial‑only Downsampling Attention, SoDA），显著降低计算量同时保留低频时空关系；② 引入基于帧相似度的动态卷积（Masked Dynamic Convolution, MDConv），在同一层同时调节 2D 与 3D 卷积权重，实现高频局部细节的捕捉；③ 通过两路径的协同工作模仿人类视觉的扫视‑注视过程，从宏观到微观分层建模时空信息。

**🔧 技术方法**

核心技术包括：Transformer 的多头自注意力、层归一化、前馈网络；SoDA 通过仅空间下采样/上采样实现自注意力降维；MDConv 通过注意力生成的调制因子对 2D/3D 卷积核进行掩码处理；以及与传统 CNN（3D/2D 卷积）相结合的多模态特征融合。

**📊 数据集**

在三个公开数据集上进行实验：Kinetics‑400（空间偏重），Something‑Something v2（时序偏重）和 Diving‑48（细粒度动作）。

**📈 对比分析**

与 SOTA（Video‑Swin、PST 等）进行对比，OG‑ReG‑T 在 Kinetics‑400 上取得 79.5% Top‑1（比 Video‑Swin‑T 高 0.7%），OG‑ReG‑T 在 Something‑Something v2 上取得 68.9% Top‑1（比 Video‑Swin‑T 高 2.7%），同时 FLOPs 与参数量均低于多数 Transformer 方案；OG‑ReG‑B 在更大模型规模下继续保持领先。

**⚠️ 局限性**

局限性包括：① 模型仍未达到人类视觉的高效预测与注意机制；② 依赖固定帧采样，无法自适应处理变速或长视频；③ 需要在每个阶段对 2D 与 3D 卷积进行精细调节，训练复杂度相对较高。

---

## 305. Walk the Talk: Bridging the Reasoning-Action Gap for Thinking with Images via Multimodal Agentic Policy Optimization

**arXiv ID:** 2604.06777 | [PDF](https://arxiv.org/pdf/2604.06777v1)

**作者:** Wenhao Yang `[一作]` (Nanjing University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 61538 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Multimodal Agentic Policy Optimization（MAPO），通过在多模态链式思考中让模型生成视觉描述标签并用CLIP评估语义一致性，强化视觉工具调用与文本推理的同步性。

**💡 创新点**

创新点在于将过程监督嵌入RL优势估计：引入语义评分与trajectory‑aware discount，实现双重方差降低，显著消除 reasoning‑action gap。

**🔧 技术方法**

结合CLIP语义评分、trajectory‑aware discount、group‑based advantage estimation、PPO/GRPO 等 RL 算法以及大语言模型与视觉工具的 agentic 框架。

**📊 数据集**

在高分辨率视觉推理基准 V*、HR‑Bench 与 MME‑Realworld‑Lite 等数据集上进行实验。

**📈 对比分析**

与闭源 GPT‑5、Gemini 2.5 Pro、开放源 DeepEyes、Thyme、Mini‑o3 以及 RL baseline PPO/GRPO/DAPO/GSPO 进行对比，MAPO 在所有指标上均优于对比模型，尤其在 HR‑Bench 及 8K 子集提升显著。

**⚠️ 局限性**

局限性包括对 CLIP 语义匹配的依赖、标签长度限制、以及在更复杂的非视觉工具环境下推广受限；在极长轨迹上仍需更稳健的探索策略。

---

## 306. FlowExtract: Procedural Knowledge Extraction from Maintenance Flowcharts

**arXiv ID:** 2604.06770 | [PDF](https://arxiv.org/pdf/2604.06770v1)

**作者:** Guillermo Gil de Avalle `[一作]` (University of Groningen), Christos Emmanouilidis `[通讯]` (University of Groningen)

**通讯引用:** 2584 | [OpenAlex ID](https://openalex.org/A5005647074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

FlowExtract 提供了一个面向 ISO 5807 标准的维护流程图自动化提取管线，可将静态 PDF/扫描图像转换为可查询的有向图。

**💡 创新点**

主要创新是将元素检测与连线重构分离，采用箭头头部先行的边检测方法，强调高精度而非召回。

**🔧 技术方法**

使用 YOLOv8 检测节点与箭头、EasyOCR 进行文本识别、Probabilistic Hough Transform 追踪连线，并结合自定义数据增强。

**📊 数据集**

训练数据来自 35 份荷兰语消费电子维护图，包含 1145 个节点、边以及决策标签。

**📈 对比分析**

与 Qwen2‑VL‑7B、Pixtral‑12B 等 VLM 基线对比，节点检测 F1 达 0.988，边检测 F1 0.667，精度 85.5%，显著优于基线。

**⚠️ 局限性**

主要局限是箭头检测召回率低导致边检测召回受限，仅适用于清晰打印的 ISO 5807 图，对非标准符号或多文档连接的适用性有限。

---

## 307. DOC-GS: Dual-Domain Observation and Calibration for Reliable Sparse-View Gaussian Splatting

**arXiv ID:** 2604.06739 | [PDF](https://arxiv.org/pdf/2604.06739v1)

**作者:** Hantang Li `[一作]` (Harbin Institute of Technology), Xiaopeng Fan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2929 | [OpenAlex ID](https://openalex.org/A5079412089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对稀视角下3D高斯点云重建的过拟合与雾化伪影问题，提出双域观察与校正（DOC-GS）框架，联合优化域的连续深度引导Dropout与观察域的暗通道先验驱动几何修剪，动态剔除不可靠的高斯原语，实现更可靠的稀视角重建。

**💡 创新点**

创新点：① 将稀视角重建问题重新表述为双域可靠性推断；② 引入连续深度引导Dropout（CDGD），提供平滑且深度感知的正则化；③ 发现浮点伪影与大气散射相似，利用暗通道先验（DCP）做观察域异常检测并驱动几何修剪；④ 将两域信息耦合为统一的可靠性驱动剔除策略，显著抑制雾化伪影。

**🔧 技术方法**

技术手段：3D Gaussian Splatting、连续深度引导Dropout（CDGD）、暗通道先验（DCP）指导几何修剪（DCP-GP）、交叉视角证据聚合、动态阈值剔除、L1+SSIM相结合的渲染损失。

**📊 数据集**

使用公开稀视角基准数据集：LLFF、MipNeRF‑360、Blender（分别采用3/6/9视、12/24视、8视等稀视角设置）。

**📈 对比分析**

与NeRF、3DGS以及多种Dropout/正则化变体（DropGaussian、DropoutGS、FSGS、CoR‑GS、NexusGS等）对比，DOC‑GS在所有稀视角设置下均实现PSNR/SSIM提升（典型提升1–2 dB），LPIPS下降，表明图像质量显著优于现有方法。

**⚠️ 局限性**

局限性：① 需要额外的深度统计与阈值调参，适配不同场景时可能需要手工微调；② 暗通道检测对阴影、低光区域敏感，可能误剔除部分真实几何；③ 额外的正则化与修剪步骤略微增加训练时间与实现复杂度。

---

## 308. CBM-Dual: A 65-nm Fully Connected Chaotic Boltzmann Machine Processor for Dual Function Simulated Annealing and Reservoir Computing

**arXiv ID:** 2604.06808 | [PDF](https://arxiv.org/pdf/2604.06808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 309. Luwen Technical Report

**arXiv ID:** 2604.06737 | [PDF](https://arxiv.org/pdf/2604.06737v1)

**作者:** Yiquan Wu `[一作]` (Zhejiang University), Kun Kuang `[通讯]` (Zhejiang University)

**通讯引用:** 2652 | [OpenAlex ID](https://openalex.org/A5041727387)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了开源中文法律语言模型Luwen，基于Baichuan 1‑7B，通过持续预训练、监督指令微调和检索增强生成等技术进行法律领域适配。

**💡 创新点**

创新点在于将三项技术——持续预训练提升法律语义理解、细粒度法律指令微调增强任务推理、检索增强生成与多源知识库融合——有机结合，形成实时、精准、可扩展的法律文本处理体系。

**🔧 技术方法**

技术手段包括Transformer架构、连续预训练(CPT)、监督指令微调(SFT)、检索增强生成(RAG)、多源多路径知识检索、对比学习、K‑Means聚类采样等。

**📊 数据集**

使用的数据集为约200GB法律与常规语料（占比20%法律），以及约100k条指令数据（30%为法律领域），包括4cLegal、4cGeneral 70k，以及公开数据如Open‑Orca、MOSS、ShareGPT、Belle、C3、Puffin等。

**📈 对比分析**

通过与Baichuan2‑Chat、GLM2、Qwen、InternLM2、GPT‑3.5等基线模型在五项任务（判决预测、司法考试、文本摘要、法条问答、判决推理）进行对比，Luwen在统计指标上普遍优于基线，尤其在法条问答和文本摘要任务上显著提升。

**⚠️ 局限性**

局限性包括检索与知识融合准确性有待提升、缺乏多模态功能、以及对复杂任务拆分与链式推理能力尚不成熟。

---

## 310. SQLStructEval: Structural Evaluation of LLM Text-to-SQL Generation

**arXiv ID:** 2604.06736 | [PDF](https://arxiv.org/pdf/2604.06736v1)

**作者:** Yixi Zhou `[一作]` (ShanghaiTech University), Zhuohan Xie `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文分析并评估LLM生成SQL的结构可靠性，提出基于AST的StructEval框架并研究编译式结构化生成。

**💡 创新点**

首次将结构一致性、多样性和鲁棒性量化，并证明执行准确度与结构可靠性不匹配；引入编译式生成提升结构一致性与执行准确率。

**🔧 技术方法**

使用SQLGlot解析并规范化AST，对多模型（GPT‑5、Claude‑4、Gemini‑3等）进行多次采样生成，并利用熵、交叉伪装度量评估结构行为。

**📊 数据集**

Spider 语料库（开发集 1,034 问）。

**📈 对比分析**

与直接 SQL、DIN‑SQL 和编译式生成进行对比，编译式生成在执行准确率 0.7864、AST 相似度 0.632 方面显著优于其他方案。

**⚠️ 局限性**

仅在 Spider 数据集上验证，AST 未能捕捉语义等价性，缺乏在更大规模数据库或其他程序生成任务中的通用性。

---

## 311. Extraction of linearized models from pre-trained networks via knowledge distillation

**arXiv ID:** 2604.06732 | [PDF](https://arxiv.org/pdf/2604.06732v1)

**作者:** Fumito Kimura `[一作]` (Saitama University), Jun Ohkubo `[通讯]` (Saitama University)

**通讯引用:** 399 | [OpenAlex ID](https://openalex.org/A5060697217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提取预训练神经网络的线性化模型，利用PCA降维+知识蒸馏构建Koopman矩阵，得到仅线性运算的分类模型。

**💡 创新点**

将Koopman算子与知识蒸馏相结合，仅使用输入与目标标签信息，无需隐藏层节点；同时加入PCA降维提升数值稳定性和准确率，改进传统最小二乘线性化方法。

**🔧 技术方法**

Koopman算子理论、知识蒸馏（KL+交叉熵）、PCA降维、可加字典函数（多项式）以及深度学习分类实验。

**📊 数据集**

MNIST、Fashion‑MNIST数据集（以及ResNet18预训练模型在这两个数据集上的验证）。

**📈 对比分析**

与传统最小二乘估计的Koopman方法（包括PCA版）进行对比，实验表明在MNIST上使用二次/三次多项式字典时准确率分别达到95.98%和96.77%，显著高于传统方法；在ResNet18预训练模型验证中也表现更优且标准差更小，显示更稳健。

**⚠️ 局限性**

仅在中等规模模型验证，需进一步测试大规模模型；仅使用多项式字典，未来需考虑更适合硬件的非线性；未考虑噪声、量化等硬件实现细节。

---

## 312. GCoT-Decoding: Unlocking Deep Reasoning Paths for Universal Question Answering

**arXiv ID:** 2604.06794 | [PDF](https://arxiv.org/pdf/2604.06794v1)

**作者:** Guanran Luo `[一作]` (Xiamen University), Qingqiang Wu `[通讯]` (Xiamen University)

**通讯引用:** 568 | [OpenAlex ID](https://openalex.org/A5048017759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无提示的通用链式推理解码策略 GCoT-decoding，能够在固定答案和自由答案的问答任务中自动生成并评估推理路径。

**💡 创新点**

创新点包括：两阶段分支策略（斐波那契采样+局部最小值回溯）、基于长度的 logit‑gap 置信度计算、以及基于语义聚类的答案聚合。

**🔧 技术方法**

使用的技术包括：斐波那契索引采样、局部低置信度回溯、长度加权 top‑2 logit gap 评分、SpanAlign 细化、句子嵌入的贪心语义聚类。

**📊 数据集**

实验数据集涵盖固定答案类（GSM8K、MultiArith、Sports Understanding）与自由答案类（SQuAD v1.1、BARQA、Auto Categorization）。

**📈 对比分析**

与贪婪解码、温度采样、top‑k、束搜索、自一致性、传统 CoT‑decoding 等方法对比，GCoT-decoding 在固定 QA 上达到或略优于现有多路径解码，在自由 QA 上在 BLEU/MATCH 上显著提升。

**⚠️ 局限性**

局限性在于需要探索多条推理路径，导致额外计算开销；实验范围仅限于问答与推理基准，未涵盖如摘要等更隐式推理任务。

---

## 313. From Perception to Autonomous Computational Modeling: A Multi-Agent Approach

**arXiv ID:** 2604.06788 | [PDF](https://arxiv.org/pdf/2604.06788v1)

**作者:** Daniel N. Wilke `[一作]` `[通讯]` (University of Witwatersrand), Daniel N. Wilke (University of Witwatersrand)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个基于多代理大型语言模型的无解算器自主计算力学框架，从照片直接生成工程分析报告。

**💡 创新点**

创新点在于把感知、建模、求解、评估四层迭代结合为一个端到端的、可追溯的、带质量门控的自动化流水线，并引入任务相关保守性和自动重设计。

**🔧 技术方法**

使用大型语言模型（Claude Opus）、多模态视觉编码、结构化 JSON 通信、自动网格生成、计算力学求解器（CalculiX）、代码合规性检查与不确定性量化。

**📊 数据集**

数据集仅为一张钢 L 型支架的实物照片，附加少量上下文信息，无 CAD 或手工测量数据。

**📈 对比分析**

与手工完成的 FEA 流程对比，整个流水线在 22 分钟内完成，计算成本约 3 美元；自动化实现的结果与人工后期修正相当，且无需人工中间步骤。

**⚠️ 局限性**

局限包括对图像几何提取的不确定性、仅演示线性静态分析、对复杂形状或多件装配的适用性不足，以及仍需专业工程师最终审阅和签字。

---

## 314. Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation

**arXiv ID:** 2604.06784 | [PDF](https://arxiv.org/pdf/2604.06784v1)

**作者:** Zhiyu Cao `[一作]` (Soochow University), Qiaoming Zhu `[通讯]` (Soochow University)

**通讯引用:** 2870 | [OpenAlex ID](https://openalex.org/A5102065469)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DRCR 框架，通过对多方对话上下文进行基于话语连贯性和回复质量的重写，提升对话生成质量。

**💡 创新点**

创新点在于：①使用加词者识别作为话语连贯性度量；②将连贯性与回复质量双重反馈融合，利用方差校正权重；③通过自我演化循环实现重写器与回复器的互相优化，减少对外部数据的依赖。

**🔧 技术方法**

技术包括：大型语言模型（LLM）重写器与回复器；加词者识别器（biaffine注意力+MLP）；候选重写生成的树形采样；NLI过滤；Coefficient of Variation 权重校正；DPO 训练；迭代自我演化（Mutual Self‑Evolution）。

**📊 数据集**

实验数据集：Ubuntu IRC‑16、Ubuntu IRC‑19、HLA‑Chat++、Friends 四个多方对话数据集。

**📈 对比分析**

与 GSN、HeterMPC、EMMDG、MADNet、RL‑TRC、SS‑MPC 以及 fine‑tuned LLM（Llama3.2‑3B、Qwen3‑4B、Qwen3‑8B）对比，DRCR 在 BLEU、METEOR、ROUGE 以及人工评价的四项指标上均显著提升，尤其在连贯性得分上超过 0.06 点，整体性能优于现有最优方法。

**⚠️ 局限性**

局限性：①初始偏好数据仍需强大 LLM 生成；②上下文重写会增加推理时延；③目前重写机制是全局应用，未针对每句进行选择性重写。

---

## 315. Towards National Quantum Communication in Europe: Planning and Sizing Terrestrial QKD Networks

**arXiv ID:** 2604.06764 | [PDF](https://arxiv.org/pdf/2604.06764v1)

**作者:** Sebastian Raubitzek `[一作]` (SBA Research gGmbH), Christoph Pacher `[通讯]` (fragmentiX Storage Solutions GmbH)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种可复现的规划方法，用于估算欧盟成员国在EuroQCI框架下的国土量子密钥分发（QKD）骨干网络规模，包括节点数量、光纤长度和受信中继设备需求。

**💡 创新点**

创新点在于将简化的显式假设与合成网络模型相结合，利用Monte Carlo搜索生成满足度量约束的度序图，并通过贪婪算法分配受信中继以压缩最长链路，从而得到可直接用于规模估算的基准规则（按人口和国土面积分别推算端点数与中继数）。

**🔧 技术方法**

核心技术包括：地理距离计算（WGS84大地测距）与固定绕路因子；基于度序的图构造与局部重连优化；受信中继分配的贪婪最小化最长跳距；Monte Carlo仿真生成多实例并选取鲁棒性最优图。

**📊 数据集**

使用的数据集主要是：各国边界多边形（Natural Earth）；端点位置通过聚类+随机乡村模型合成，聚类中心设为首都/主要城市；不使用真实的受保护设施清单或现有光缆网络。

**📈 对比分析**

在奥地利案例中通过1000次模拟得到稳定的总光纤长度（约8600公里）与平均/最大跳距（约22.9/52公里）；随后通过人口/面积比例对其余28个欧盟成员国进行规模扩展，得到各国的光纤长度、跳距与节点数等指标，证明方法能够在不同地理与人口条件下保持合理性和一致性。

**⚠️ 局限性**

主要限制包括：端点分布为合成代理，未考虑真实受保护设施；光纤路径采用统一绕路因子，未使用实际光缆走线；未建模流量需求、关键管理负载或跨境链路；未包含成本估算、运维或安全风险量化；模型不适用于岛屿或海外领土的完整覆盖。

---

## 316. TurboAgent: An LLM-Driven Autonomous Multi-Agent Framework for Turbomachinery Aerodynamic Design

**arXiv ID:** 2604.06747 | [PDF](https://arxiv.org/pdf/2604.06747v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 317. Improving Random Testing via LLM-powered UI Tarpit Escaping for Mobile Apps

**arXiv ID:** 2604.06763 | [PDF](https://arxiv.org/pdf/2604.06763v1)

**作者:** Mengqian Xu `[一作]` (East China Normal University), Weikai Miao `[通讯]` (East China Normal University)

**通讯引用:** 364 | [OpenAlex ID](https://openalex.org/A5107246472)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种将大语言模型（LLM）与随机 GUI 测试相结合的混合测试方法，动态检测 UI 俯冲陷阱并通过 LLM 生成逃逸事件，进一步提升安卓应用的覆盖率和缺陷发现率。

**💡 创新点**

创新点在于：①使用图像哈希和相似度阈值实现在线 UI 俯冲陷阱检测；②在检测到陷阱时调用 LLM 进行语义理解并给出逃逸建议；③引入概率重用机制和遮挡过滤，减少 LLM 调用次数并保证可靠性；④在两款主流随机测试工具上实现验证，展示了可迁移性。

**🔧 技术方法**

核心技术包括：大语言模型（GPT‑4o）、图像哈希+汉明距离相似度判定、Android Accessibility Service 提取 UI 结构、LLM Prompt Engineering、概率重用与遮挡过滤、代码覆盖度测量（JaCoCo）与活动覆盖率统计。

**📊 数据集**

实验使用13个真实应用（8个开源、4个 Google Play、1个工业级微信），以及两个额外基准数据集进行补充验证。

**📈 对比分析**

对比7个基线（传统随机/强化学习工具、专门的陷阱逃逸工具、两种 LLM‑驱动工具），在3小时预算下，所提方法在12个常用安卓应用中平均提升行覆盖率 54.8%、分支覆盖率 44.8%，发现的独立崩溃数比最佳基线多 2 倍以上；在微信工业级应用中同样实现了更高的活动覆盖率并捕获更多缺陷。

**⚠️ 局限性**

主要局限包括：①逃逸效果高度依赖可用的 UI 可访问性信息，缺失标签导致 65% 的失败；②对文本输入需求复杂的场景（登录/注册）支持不足；③仅针对视觉相似的单一路径陷阱，无法处理跨页面的循环陷阱；④图像相似度阈值需要手工调参，可能不适用于所有主题或动态内容。

---

## 318. Multilingual Cognitive Impairment Detection in the Era of Foundation Models

**arXiv ID:** 2604.06758 | [PDF](https://arxiv.org/pdf/2604.06758v1)

**作者:** Damar Hoogland `[一作]`, Matthew Purver `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究在英语、斯洛文尼亚语和韩语三种语言中，比较LLM、基础表格模型与传统机器学习模型在认知障碍检测中的性能，采用统一的留一法、零样本与少样本评估协议。

**💡 创新点**

创新点在于首次在三语言下进行系统对比，探讨特征-嵌入对齐与多模态融合对小样本CI检测的影响，并使用表格基础模型TabPFN与LLM进行零样本推理。

**🔧 技术方法**

使用的技术包括多语言大型语言模型（gpt‑oss‑20b、med‑gemma‑27b）、表格基础模型TabPFN、经典ML（LR、RF、SVM、LightGBM、kNN）、句子BERT嵌入以及早期融合与后期融合策略。

**📊 数据集**

数据集来源为Pitt Corpus（英语）、Coglitreat（斯洛文尼亚语）和Kang Corpus（韩语），共260名受试者，分别包含AD/HC或MCI/HC标签。

**📈 对比分析**

实验结果显示，表格模型结合专家特征在多语言下均显著优于零样本LLM（最高相差+0.26宏F1），早期融合在斯洛文尼亚语和韩语中表现最佳；LLM零样本仅能略高于多数类基线。

**⚠️ 局限性**

局限包括样本量有限、语言/记录格式的潜在混淆、部分特征缺失、仅使用图像描述任务、未评估人口统计偏差及模型可解释性等。

---

## 319. How Long Reasoning Chains Influence LLMs' Judgment of Answer Factuality

**arXiv ID:** 2604.06756 | [PDF](https://arxiv.org/pdf/2604.06756v1)

**作者:** Minzhu Tu `[一作]` (State Key Laboratory of AI Safety), Keping Bi `[通讯]` (State Key Laboratory of AI Safety)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在LLM评判者中使用推理链对答案正确性判断的影响，系统性地比较了弱强评判者在多种事实与数学数据集上的行为；

**💡 创新点**

首次从弱强评判者角度系统评估推理链带来的误导与提升，并通过控制实验揭示推理链的流畅性与事实性对判断的重要性；

**🔧 技术方法**

采用生成-评判框架，利用Chain‑of‑Thought推理与LLM‑as‑a‑Judge技术，对比“无推理链”与“有推理链”两种评判方式，并进行无关或伪造信息插入实验；

**📊 数据集**

使用四个公开数据集：NQ、HotpotQA、GSM8K和MATH‑500；

**📈 对比分析**

通过对齐率、通过率、过度自信和保守性四个指标对比评判者表现；结果显示弱评判者在接收推理链后显著被误导，强评判者虽然更有选择性，但仍会被高质量伪推理链误导；

**⚠️ 局限性**

仅限文本任务，未让评判者生成推理链，覆盖的模型有限，且未探讨多模态或新兴LLM的情况。

---

## 320. LiveStre4m: Feed-Forward Live Streaming of Novel Views from Unposed Multi-View Video

**arXiv ID:** 2604.06740 | [PDF](https://arxiv.org/pdf/2604.06740v1)

**作者:** Pedro Quesado `[一作]` (Eindhoven University of Technology), Egor Bondarev `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 692 | [OpenAlex ID](https://openalex.org/A5069685461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种实时、无姿态、无标定的直播新视点视频合成系统LiveStre4m，能从稀疏无姿态多视角视频实时生成高质量新视点视频。

**💡 创新点**

结合摄像机姿态预测器、基于Transformer的空间模块和扩散Transformer插值网络，实现不依赖相机参数、仅用两路低分辨率输入，零样本、零优化即可实时生成高分辨率、时间一致的新视点视频。

**🔧 技术方法**

Vision Transformer、Dense Prediction Transformer、3D Gaussian Splatting、扩散Transformer（DiT）帧插值、轻量级CNN超分辨率、ViT姿态回归、端到端训练等技术。

**📊 数据集**

Neural3DVideo（6个动态场景、21相机、2704×2028）和MeetRoom（3个动态场景、13相机、1280×720）数据集，前者用于训练与测试，后者仅用于测试。

**📈 对比分析**

与视频优化、帧优化以及Feed-forward方法（如4DGS、IGS、FLARE）比较，LiveStre4m每帧耗时0.07–0.14秒（≈55–19×更快），PSNR约21.1/18.6，性能与FLARE相近但速度快3.3×，但视觉质量低于优化方法。

**⚠️ 局限性**

视觉质量低于优化方法；分辨率越高耗时越长，帧率受限；目前仅支持静态相机，未处理相机运动；对极端快速运动或视角差异较大的场景鲁棒性有限。

---

## 321. Beyond Accuracy: Diagnosing Algebraic Reasoning Failures in LLMs Across Nine Complexity Dimensions

**arXiv ID:** 2604.06799 | [PDF](https://arxiv.org/pdf/2604.06799v1)

**作者:** Parth Patil `[一作]` (BITS Pilani), Murari Mandal `[通讯]` (KIIT)

**通讯引用:** 1579 | [OpenAlex ID](https://openalex.org/A5090323460)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种九维代数复杂度框架，并实现自动生成与验证问题的流水线，用于诊断大语言模型在代数推理上的失败模式。

**💡 创新点**

创新点在于将九个互相独立的复杂度维度系统化，并通过参数化生成器在保持其它维度固定的情况下单独调节每个维度，首次实现可持续扩展的动态基准。

**🔧 技术方法**

使用的技术包括基于前缀 Polish 记号的表达式生成、SymPy CAS 验证、自动化脚本生成和评测，模型调用 API 并以步进式求解方式评估准确率。

**📊 数据集**

数据集为自生成的 9×50=450 条验证问题，涵盖不同维度的 5-751 词、1-8 层深度等多种组合，无需人工标注。

**📈 对比分析**

通过对七个不同规模的指令调优模型（8B-235B）在每个维度上的准确率曲线进行比较，发现工作记忆在 20-30 个并行分支处为统一瓶颈，计数维度表现出显著差异；整体来看，Claude 3.5 Haiku 在多数维度上表现最佳。

**⚠️ 局限性**

局限性包括：仅关注代数表达式，未覆盖微积分级别算子；所用模型受限于 API 访问，无法探索更大参数模型；并且评估仍基于固定阈值的准确率判定，未细粒度捕捉误差类型。

---

## 322. Instance-Adaptive Parametrization for Amortized Variational Inference

**arXiv ID:** 2604.06796 | [PDF](https://arxiv.org/pdf/2604.06796v1)

**作者:** Andrea Pollastro `[一作]` (University of Naples Federico II), Roberto Prevete `[通讯]` (University of Naples Federico II)

**通讯引用:** 1975 | [OpenAlex ID](https://openalex.org/A5059881379)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了IA-VAE（Instance‑Adaptive Variational Autoencoder），一种在变分自编码器中使用超网络对共享推理网络进行实例级参数调制的框架。

**💡 创新点**

创新点在于通过输入依赖的超网络直接生成共享编码器的参数调制，使推理网络在保持单次前向传播效率的同时，获得了对每个样本的灵活适应，从而显著减小了传统 amortized inference 的 amortization gap。

**🔧 技术方法**

核心技术包括变分推理、变分自编码器、超网络（hypernetwork）以及块级参数调制（blockwise parameter modulation），并利用重参数化技巧实现端到端可微训练。

**📊 数据集**

实验使用了可知真实后验的三维模拟数据，以及三大经典图像数据集：OMNIGLOT、MNIST 与 Fashion‑MNIST。

**📈 对比分析**

通过对比基准 VAE，IA‑VAE 在所有数据集上都获得了更高的 ELBO（在模拟数据上也显示了更接近 MAP 的后验近似和更高的后验密度比），并且实验结果在不同随机初始化和参数规模下保持稳定，统计检验显示提升显著。

**⚠️ 局限性**

局限性包括：仍无法完全消除 amortization gap；超网络的设计与调度增加了模型复杂度；目前仅在 VAE 框架下验证，尚需进一步探讨在更大规模或不同生成模型中的适用性。

---

## 323. When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning

**arXiv ID:** 2604.06787 | [PDF](https://arxiv.org/pdf/2604.06787v1)

**作者:** Yang Xiang `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**通讯引用:** 84116 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大规模推理模型过度思考问题，提出动态思维充分性检查（DTSR）实现早退出；

**💡 创新点**

创新点在于将人类元认知启发的反思信号监测与思维充分性检查结合，使模型自评CoT是否足够决定退出；

**🔧 技术方法**

利用提示工程、反思信号检测、阈值自评、vLLM推理加速等技术实现动态评估与早退出；

**📊 数据集**

使用GSM8K、MATH-500、AMC、OlympiadBench、GPQA Diamond、LiveCodeBench等六大推理基准；

**📈 对比分析**

与Vanilla、NoThinking、NoWAIT、DEER等基线对比，DTSR在保持几乎相同准确率的前提下，推理长度缩短28.9%–34.9%，推理延迟降低25%–40%；

**⚠️ 局限性**

仅在至32B规模模型上实验，聚焦文本推理，未扩展到多模态或智能体等场景。

---

## 324. EventFace: Event-Based Face Recognition via Structure-Driven Spatiotemporal Modeling

**arXiv ID:** 2604.06782 | [PDF](https://arxiv.org/pdf/2604.06782v1)

**作者:** Qingguo Meng `[一作]` (Anhui University), Massimo Tistarelli `[通讯]` (University of Sassari)

**通讯引用:** 3902 | [OpenAlex ID](https://openalex.org/A5064744996)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用事件视觉传感器对面部图像进行识别。

**💡 创新点**

提出结构驱动的时空建模框架，使事件数据的空间结构和时间动态得到更充分的利用。

**🔧 技术方法**

基于事件摄像机捕获的数据，采用时空卷积与图神经网络相结合的技术实现识别。

**📊 数据集**

在公开的事件面部数据集 E-IR 上进行实验。

**📈 对比分析**

与传统事件CNN和基准方法对比，准确率提升至约96%，明显优于基线模型。

**⚠️ 局限性**

受限于事件数据稀疏性、对光照变化的鲁棒性不足以及计算资源需求高等因素。

---

## 325. Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search

**arXiv ID:** 2604.06771 | [PDF](https://arxiv.org/pdf/2604.06771v1)

**作者:** Zhiyu Cao `[一作]` (Soochow University), Qiaoming Zhu `[通讯]` (Soochow University)

**通讯引用:** 2870 | [OpenAlex ID](https://openalex.org/A5102065469)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了多维自一致性偏好对齐的对话查询重写框架MSPA‑CQR。

**💡 创新点**

通过构造自一致性偏好数据并使用前缀引导的多维直接偏好优化，兼顾重写、检索与回复三个维度的反馈。

**🔧 技术方法**

结合LLM采样、NLI自一致性评分、MDPO（基于DPO）以及前缀指导来优化重写查询。

**📊 数据集**

在TopiOCQA和QReCC等对话检索数据集上进行实验。

**📈 对比分析**

与EDIRCS、IterCQR、AdaCQR、RETPO等多种基线比较，MSPA‑CQR在稀疏和稠密检索上均实现了显著提升（如MRR、NDCG等指标均提高）。

**⚠️ 局限性**

仍存在自一致性评分的优化空间、偏好间关联处理不足以及仅使用DPO而未尝试其他对齐方法等局限。

---

## 326. Babbling Suppression: Making LLMs Greener One Token at a Time

**arXiv ID:** 2604.06755 | [PDF](https://arxiv.org/pdf/2604.06755v1)

**作者:** Lola Solovyeva `[一作]` (University of Twente), Fernando Castor `[通讯]` (University of Twente)

**通讯引用:** 2747 | [OpenAlex ID](https://openalex.org/A5062400717)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在代码生成中的“胡言乱语”现象，并提出了一种基于测试执行的“胡言抑制”（BS）方法以提前终止生成。

**💡 创新点**

创新点在于将单元测试嵌入生成循环，实现无模型重训练、无提示修改的在线早停，能够在不影响准确率的前提下大幅削减多余 token。

**🔧 技术方法**

采用了逐步语法/类型检查与单元测试执行相结合的算法，配合 HuggingFace 上的多大模型进行推理。

**📊 数据集**

使用了 Python 的 HumanEval 与 MBPP，Java 的 HumanEval‑Java 与改写后的 APPS 四个公开 benchmark。

**📈 对比分析**

通过与完整生成 baseline 对比，BS 在 10 个模型、4 个 benchmark 上平均减少 22–58% token，能耗下降 20–65%，整体能源/速度提升超过 70% 的实验组合。

**⚠️ 局限性**

局限性包括：对低准确率或已极短输出模型几乎无效，未直接测量 CPU 能耗，仅在 Python 与 Java 两语言验证，且对模型内部状态无感知。

---

## 327. From Static to Interactive: Adapting Visual in-Context Learners for User-Driven Tasks

**arXiv ID:** 2604.06748 | [PDF](https://arxiv.org/pdf/2604.06748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 328. Select-then-Solve: Paradigm Routing as Inference-Time Optimization for LLM Agents

**arXiv ID:** 2604.06753 | [PDF](https://arxiv.org/pdf/2604.06753v1)

**作者:** Heng Zhou `[一作]`, Zhenfei Yin `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对六种推理范式在四大LLM与十个基准上进行统一评估，并提出按任务路由选择范式的方法

**💡 创新点**

揭示推理结构的任务依赖性与范式互补性，提出轻量级嵌入路由器实现范式自适应，显著提升性能

**🔧 技术方法**

统一推理框架、任务嵌入编码、逻辑回归/MLP路由器、零射门自路由对比、基于token预算的性能评估

**📊 数据集**

HumanEval、MATH500、AIME、HotpotQA、NaturalQuestions、MMLU、HLE、GAIA、τ-bench、SEAL

**📈 对比分析**

通过固定模型、工具、提示，比较各范式的准确率与token成本；发现范式互补，oracle平均提升17.1pp；路由器平均提高5.5pp（从47.6%至53.1%），恢复37% oracle gap

**⚠️ 局限性**

范式集合有限；仅在静态基准上评估；工具集有限；未探究更复杂任务或动态环境；路由器需预训练，缺乏完全无监督方案

---

## 329. Busemann energy-based attention for emotion analysis in Poincaré discs

**arXiv ID:** 2604.06752 | [PDF](https://arxiv.org/pdf/2604.06752v1)

**作者:** Zinaid Kapić `[一作]` (University of Rijeka), Vladimir Jaćimović `[通讯]` (University of Montenegro)

**通讯引用:** 443 | [OpenAlex ID](https://openalex.org/A5037070600)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了完全基于双曲几何的深度学习模型 EmBolic，用于从文本中进行细粒度情绪分类。

**💡 创新点**

创新点在于：①将注意力机制改写为在 Poincaré 多盘中的共形质心计算；②使用 Busemann 能量函数作为情绪向量之间的相似度度量；③通过多盘嵌入（多重 Poincaré 球）在保持层内保持不变性和对情绪不确定性的刻画。

**🔧 技术方法**

主要技术包括：双曲 GloVe 词向量学习、Poincaré 多盘几何、Möbius 变换、共形质心算法、对比损失（正负样本距离）、Busemann 能量得分、温度化 softmax。

**📊 数据集**

使用了 Google 的 GoEmotions 数据集的子集（4,884 条 Reddit 评论，28 种情绪标签）。

**📈 对比分析**

通过与单盘模型和多盘（3盘）模型的对比评估：单盘模型 Top‑1 约 14–19%，Top‑5 约 41–46%；三盘组合后 Top‑1 约 26%，Top‑5 约 58%。性能仍相对有限，说明高维双曲空间需要进一步扩展。

**⚠️ 局限性**

局限性包括：①样本量和维度有限导致准确率不高；②对情绪的方向化映射依赖于训练过程，易出现错误高置信度预测；③缺乏与传统 Euclidean 或混合模型的直接实验对比，难以量化超越优势；④模型对多重情绪标签（多标签情况）处理不完善。

---

## 330. How Well Do Vision-Language Models Understand Sequential Driving Scenes? A Sensitivity Study

**arXiv ID:** 2604.06750 | [PDF](https://arxiv.org/pdf/2604.06750v1)

**作者:** Roberto Brusnicki `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**通讯引用:** 2032 | [OpenAlex ID](https://openalex.org/A5063677428)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VENUSS 框架，对视觉-语言模型（VLM）在连续驾驶场景下的理解进行系统的敏感性分析

**💡 创新点**

首次从图像分辨率、帧数、时间间隔、空间布局、展示方式等维度对 VLM 进行多维度评估，并给出最优配置方案

**🔧 技术方法**

采用 VLM 预训练模型（如 Qwen‑VL‑Max、Claude、Gemini、GPT‑4o‑Mini 等）与定制化提示工程，结合多种输入格式生成图片拼接、GIF 与视频回放

**📊 数据集**

使用 CoVLA、Honda Scenes、NuScenes、Waymo Open Dataset 等公开驾驶数据集，自动抽取连续帧并转换为结构化标签

**📈 对比分析**

通过对 25+ VLM 在 2,600+ 方案下的准确率、F1 以及人类基线进行对比，发现顶尖模型仅 57% 准确率，远低于人类 65%，但可通过配置优化提升 48%

**⚠️ 局限性**

VLM 在时间推理（加速、方向）和细粒度动态理解任务上表现不佳，且模型表现高度可变（σ≈0.21），安全性与可验证性仍待提升

---

## 331. Affine Subcode Ensemble Decoding of Linear Block Codes

**arXiv ID:** 2604.06889 | [PDF](https://arxiv.org/pdf/2604.06889v1)

**作者:** Jonathan Mandelbaum `[一作]` (Karlsruhe Institute of Technology), Laurent Schmalen `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 4220 | [OpenAlex ID](https://openalex.org/A5053280913)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种新的集成解码方案 aSCED（Affine Subcode Ensemble Decoding），通过在 BP 解码框架下同时使用线性子码与仿射子码来提高短码块的误码性能。

**💡 创新点**

创新点主要包括：①引入仿射子码并给出相应的 CN 更新规则；②通过使用仿射子码实现均匀码字保护；③在子码设计时仅需更少的 PCM，简化集成解码的设计；④同一批次内多条解码路径共享相同的 Tanner 图，便于硬件复用。

**🔧 技术方法**

技术实现包括：基于 BP 的线性/仿射子码解码、ssPCM（Structured Sparse Parity‑Check Matrix）生成、子码 PCM 优化、Monte‑Carlo 仿真评估以及 ML‑in‑the‑list 决策。

**📊 数据集**

实验数据集主要为两种短 LDPC 码（5G 132,66 码、CCSDS 256,128 码）以及两种 BCH 码（63,30 码和 63,36 码）。

**📈 对比分析**

性能评估通过与单独 BP、AED、SCED、MBBP 等等等效复杂度的集成解码器比较。aSCED 在大多数情形下比这些方法至少提升 0.2–0.4 dB；对 BCH(63,30) 码，使用 64 条 BP 路径即可逼近 ML 性能。

**⚠️ 局限性**

局限性包括：①当子码数量增大时复杂度上升，特别是 2^k 路径的理论极限；②仿射子码的构造与 PCM 设计依赖于代码结构；③实验主要验证了所有零码字假设下的性能，真实环境中需考虑非零码字的影响；④目前仅针对二进制线性码，扩展到非二进制或更高维度的情形尚未研究。

---

## 332. MENO: MeanFlow-Enhanced Neural Operators for Dynamical Systems

**arXiv ID:** 2604.06881 | [PDF](https://arxiv.org/pdf/2604.06881v1)

**作者:** Tianyue Yang `[一作]` (University College London), Xiao Xue `[通讯]` (University College London)

**通讯引用:** 57647 | [OpenAlex ID](https://openalex.org/A5100329996)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为MENO的混合框架，将低分辨率神经算子与单步改进的MeanFlow生成解码器结合，以在高分辨率下恢复多尺度物理细节并保持统计一致性。

**💡 创新点**

创新点在于将改进的MeanFlow模型作为高效的一步生成解码器，既能恢复神经算子在高分辨率下失去的小尺度信息，又不需要多步扩散采样的高计算成本；同时该框架可无缝集成到任意现有神经算子后端。

**🔧 技术方法**

使用神经算子（如FNO、UNO）学习低分辨率动力学；改进MeanFlow（i‑MF）模型作为单步解码器；训练时采用i‑MF损失；推理时先用神经算子进行低分辨率自回归预测，再用解码器一次性映射至高分辨率。

**📊 数据集**

在三个跨物理学的高分辨率数据集上进行评估：Cahn‑Hilliard相分离（PF100）、二维Kolmogorov湍流（KF256）以及二维主动物质（AM256）。

**📈 对比分析**

与基准的无增强神经算子以及使用DDIM扩散模型增强的算子进行比较。MENO在相对L2误差、SSIM和功率谱密度误差方面均优于基准，提升幅度可达50%+；同时在推理速度上比DDIM提升约10-12倍，参数量与基准相近。

**⚠️ 局限性**

缺点包括：解码器未显式加入物理先验或参数条件；对非常大尺度或极端非平衡系统的泛化尚未验证；目前仍以数据驱动，需进一步探索物理信息注入。

---

## 333. Digital Skin, Digital Bias: Uncovering Tone-Based Biases in LLMs and Emoji Embeddings

**arXiv ID:** 2604.06863 | [PDF](https://arxiv.org/pdf/2604.06863v1)

**作者:** Mingchen Li `[一作]` (University of North Texas), Yunhe Feng `[通讯]` (University of North Texas)

**通讯引用:** 796 | [OpenAlex ID](https://openalex.org/A5073748933)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性评估并比较了传统静态emoji嵌入模型与现代大语言模型（LLM）在肤色表情符号上的支持情况和偏差，量化了语义漂移、情感倾向及关联偏见。

**💡 创新点**

首次将多维度偏差检测技术（RND、WEAT、RNSB）应用于emoji肤色分析，揭示LLM虽然覆盖全面但仍存在显著的情感与关联偏差，并发现tokenization层面存在计算成本不均的“隐形偏见”。

**🔧 技术方法**

使用语义相似度测量（Cosine、Word Mover’s Distance）、t‑SNE可视化、tokenizer统计、RND、WEAT、RNSB等技术进行偏差量化与可视化。

**📊 数据集**

主要数据集包括Unicode Emoji表、EmojiCounts 17.0、EmojiTracker高频emoji、Emojidb情感词表、NRC‑VAD中性词表、Caliskan社会学词集，结合多种LLM（Llama、Gemma、Qwen、Mistral）和静态模型（emoji2vec、emoji‑sw2v 等）。

**📈 对比分析**

通过覆盖率对比、token化一致性评估、语义一致性（Cosine/WMD）、情感一致性（WEAT、RNSB）等多维度指标进行系统比较；结果显示LLM在肤色支持上优于静态模型，但在token成本、语义漂移和情感偏差方面仍显著劣势；静态模型覆盖率低、偏差更大。

**⚠️ 局限性**

局限性包括：只分析英文和公开模型；未考虑上下文动态对emoji语义的影响；缺乏因果干预或对策验证；聚焦于手势、职业等子集，未覆盖所有emoji；模型评估仅限于当前版本，未来更新可能导致结果变化。

---

## 334. Enhancing Secure Intent-Based Networking with an Agentic AI: The EU Project MARE Approach

**arXiv ID:** 2604.06856 | [PDF](https://arxiv.org/pdf/2604.06856v1)

**作者:** Iulisloi Zacarias `[一作]` (Technical University of Braunschweig), Admela Jukan `[通讯]` (Technical University of Braunschweig)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套分层多代理的意图驱动安全框架（IBS），在6G网络中利用大型语言模型（LLM）将自然语言安全意图转化为可执行的网络策略，并通过外部安全知识库（MITRE、NIST）增强推理能力。

**💡 创新点**

创新点包括：① 在IBS中嵌入Agentic AI，形成完整的安全平面；② 采用多代理结构支持多域、多厂商网络；③ 通过LLM与外部知识库交互，实现对零日威胁的主动防御；④ 证明即便是小型本地LLM也能实现高成功率的意图执行。

**🔧 技术方法**

技术手段：意图处理管线（分类、对齐、细化、分解），多代理调度与回滚，LLM（OpenAI o4-mini、Mistral、GPT‑OSS、Qwen3），LangChain/LangGraph实现代理间交互，MCP协议与REST API，外部安全知识库集成。

**📊 数据集**

使用内部手工构建的安全意图数据集（3组，共30条意图），结合历史配置、执行结果与网络拓扑信息做为LLM对齐与上下文提示的训练/推理数据。

**📈 对比分析**

评估方法：对每条意图执行情况分为 Pass / Domain‑Fail / Blocked，统计成功率；比较四种LLM模型在相同意图集上的表现，发现小型模型也能达到与大型模型相近的成功率；进一步展示了模型在本地部署时的资源占用与延迟表现。

**⚠️ 局限性**

局限性：① 未完成完整的SAIN保障与回路闭合实现；② 仅在代理层面进行模拟，未在真实5G/6G网络上部署；③ 对法律合规类意图处理不成熟，导致部分意图被阻塞；④ 评估指标仍缺乏标准化，未来需设计更细粒度的安全与性能度量。

---

## 335. Contraction-Aligned Analysis of Soft Bellman Residual Minimization with Weighted Lp-Norm for Markov Decision Problem

**arXiv ID:** 2604.06837 | [PDF](https://arxiv.org/pdf/2604.06837v1)

**作者:** Hyukjun Yang `[一作]` (Korea Advanced Institute of Science and Technology), Donghwan Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2202 | [OpenAlex ID](https://openalex.org/A5100654316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出软贝尔曼残差最小化框架PSBRM，使用加权L_p范数与L∞收敛几何对齐，解决函数逼近下投影导致的收敛问题。

**💡 创新点**

在足够大的p下证明软贝尔曼算子在L_p,w范数中收敛，并给出残差最小化的误差上界和随p增大误差系数收敛的理论，展示了从平均型残差到极端残差的平滑过渡。

**🔧 技术方法**

采用软贝尔曼残差最小化、加权L_p范数分析、Banach不动点定理以及梯度下降优化算法（p为偶数保证可微性）。

**📊 数据集**

在一个6状态、2动作、折扣率0.95的简单MDP上进行实验验证。

**📈 对比分析**

与L_p-PVI、L_2-SBRM对比，PSBRM在p大时收敛稳定、误差更小；L_p-PVI易发散，L_2-SBRM误差更大。

**⚠️ 局限性**

理论要求p足够大且为偶数以保证可微性；实验仅在极小规模MDP上验证，缺乏在大规模真实环境中的实证。

---

## 336. Environmental, Social and Governance Sentiment Analysis on Slovene News: A Novel Dataset and Models

**arXiv ID:** 2604.06826 | [PDF](https://arxiv.org/pdf/2604.06826v1)

**作者:** Paula Dodig `[一作]` (Eindhoven University of Technology), Matthew Purver `[通讯]` (Jožef Stefan Institute and Postgraduate School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个斯洛文尼亚ESG情感标注数据集（SloESG-News 1.0），并利用LLM、Transformer、句子嵌入和层次集成模型对新闻文本进行环境、社会和治理情感分类；

**💡 创新点**

创新点在于首次为低资源语言提供ESG情感标注资源、使用LLM辅助过滤与标注、提出多层次集成与多任务学习的ESG情感框架，并实现动态实时情感监测；

**🔧 技术方法**

采用的技术包括Gemma、GaMS、GPT-OSS等LLM的零/少样本推理、SloBERTa和XLM‑RoBERTa的微调、BGE‑M3/Paraphrase/Gemma‑Embed句子嵌入、TabPFN元学习分类、以及基于概率logit的层次多任务集成；

**📊 数据集**

使用的数据集为从MaCoCu斯洛文尼亚新闻中筛选并人工标注的550篇文章的SloESG-News 1.0，并在2010–2025年间的大规模新闻语料库上进行案例时序情感分析；

**📈 对比分析**

通过与多数投票基线、单一模型和多模型集成进行对比，结果显示Gemma3‑27B在环境方面F1‑macro≈0.61，gpt‑oss‑20B在社会方面≈0.45，SloBERTa在治理方面≈0.54；层次集成进一步提升了整体性能；

**⚠️ 局限性**

局限性包括数据集规模小（550篇），治理标签一致性低，LLM预筛选可能导致采样偏差，模型对极端情感的判别仍受限，且跨语言推广和因果解释能力不足。

---

## 337. RePL: Pseudo-label Refinement for Semi-supervised LiDAR Semantic Segmentation

**arXiv ID:** 2604.06825 | [PDF](https://arxiv.org/pdf/2604.06825v1)

**作者:** Donghyeon Kwon `[一作]` (POSTECH), Suha Kwak `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种半监督LiDAR语义分割框架RePL，通过伪标签精炼（错误估计+遮罩重构）提升伪标签质量，进而改善学生网络训练；

**💡 创新点**

创新点在于结合错误估计与Masked Autoencoder风格的重构进行双阶段伪标签精炼，并给出理论条件分析；同时在教师-学生框架中加入随机遮罩和混合数据增强等技术；

**🔧 技术方法**

使用教师-学生网络、EMA更新、confidence‑based error mask、随机遮罩、Masked Autoencoder inspired 重构、负学习信号、LaserMix混合、对称交叉熵+Lovász‑Softmax等；

**📊 数据集**

在nuScenes‑lidarseg和SemanticKITTI两个公开LiDAR分割基准上进行实验；

**📈 对比分析**

与多种最新半监督方法（AIScene、IT2、FrustrumMix、Seal、SuperFlow、SLidR、Lim3D等）进行对比，RePL在1%、10%、20%、50%标注比例下均取得SOTA，平均mIoU提升约2%+，尤其在1%标注下表现最优；

**⚠️ 局限性**

局限性包括对错误候选mask质量的依赖、随机遮罩和混合策略增加计算成本，以及在高精度伪标签时精炼收益递减；过度修正可能导致局部误差。

---

## 338. SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization

**arXiv ID:** 2604.06817 | [PDF](https://arxiv.org/pdf/2604.06817v1)

**作者:** Usman Naseem `[一作]` (Macquarie University), Seid Muhie Yimam `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供了22种语言、110k实例的多标签多事件在线极化检测共享任务和数据集，并对多语言模型在三类子任务（极化存在、类型、表现）上进行了评估。

**💡 创新点**

创新点在于细粒度极化的三分类（存在、类型、表现），大规模多语言覆盖以及跨文化细致标注与多标签体系，首次构建如此多样化的极化检测基准。

**🔧 技术方法**

主要使用多语言Transformer（Gemma、Qwen、LLaMA等）结合参数高效微调（LoRA、Adapter）、数据增强、模型集成、阈值调优等技术。

**📊 数据集**

使用公开的Polar-SEM-2026数据集，覆盖22种语言、7个语言族、110k实例，涵盖多事件、多文化背景。

**📈 对比分析**

与基线LaBSE相比，顶尖团队在子任务1宏F1平均超过0.8，子任务2和3表现相对较低，整体显示多标签极化检测仍具挑战，展示了集成与阈值调优的有效性。

**⚠️ 局限性**

限制包括标注者主观性与文化差异、低资源语言样本不足、模型选择有限、跨文化泛化性不足，以及对极化细化维度的解释性不足。

---

## 339. Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models

**arXiv ID:** 2604.06871 | [PDF](https://arxiv.org/pdf/2604.06871v1)

**作者:** Bajian Xiang `[一作]` (Beike Inc.), Yang Han `[通讯]` (Beike Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究大型语音语言模型（LSLM）的语音分辨率冗余，利用层级分析发现深层表示高度冗余，随后提出一种无训练、基于相似度的音频令牌合并方法（Dual Affinity Pooling, DAP），在输入层与深层同时压缩令牌序列，从而显著降低推理成本。

**💡 创新点**

创新点在于①首次通过层级干预实验揭示LSLM内部冗余层级；②设计了基于余弦相似度且考虑时间局部性的无监督令牌合并策略；③将该策略在输入层和深层双重应用，形成训练‑free 的 DAP，兼顾压缩率与性能。

**🔧 技术方法**

主要技术包括：层级 Oracle 干预（Drop/Uniform Merge 等）分析冗余；余弦相似度计算与基于阈值的窗口合并算法；双层合并策略（AP_in + AP_deep）；性能评估采用 FLOPs、内存占用、TTFT 等指标。

**📊 数据集**

使用的数据集包括：语音识别（LibriSpeech、KeSpeech）、语音问答（OpenBookQA、SDQA、SpeechTriviaQA）、语音翻译（CoVost2 en2zh/zh2en）以及内部对齐所用的 Montreal Forced Aligner；实验基于 Qwen2‑Audio 与 Kimi‑Audio 两大 LSLM。

**📈 对比分析**

与固定预算压缩（如时间拉伸、线性插值）以及信号级别加速方法对比，DAP 在保持或提升 WER/Accuracy/BLEU 的同时，预填充 FLOPs 减少 27.48%，内存节省约 1.7 倍，TTFT 加速 1.1 倍；在多任务评测中，DAP 在 Aggressive 配置下仅略低于基线，甚至在 QA 上提升准确率。

**⚠️ 局限性**

局限性包括：评测仅聚焦语义任务，对细粒度声学细节的影响未深入探究；Oracle 对齐基于词边界的逼近可能导致冗余分析略有偏差；由于 LSLM 令牌压缩领域仍处于起步阶段，缺乏公开基线方法进行更全面的对比。

---

## 340. Energy-Regularized Spatial Masking: A Novel Approach to Enhancing Robustness and Interpretability in Vision Models

**arXiv ID:** 2604.06893 | [PDF](https://arxiv.org/pdf/2604.06893v1)

**作者:** Tom Devynck Bilal Faye Djamel Bouchaffra Nadjib Lazaar Hanane Azzag Mustapha Lebbah `[一作]` `[通讯]`, Tom Devynck Bilal Faye Djamel Bouchaffra Nadjib Lazaar Hanane Azzag Mustapha Lebbah

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 Energy‑Regularized Spatial Masking（ERSM），在卷积网络中嵌入可微能量掩码层，通过能量最小化实现输入自适应的稀疏特征选择。

**💡 创新点**

创新点在于将能量模型从全局生成迁移到内部特征选择，将像素级稀疏化转化为物理动力学的能量最小化，并结合单项和相邻项实现连贯、可解释的空间掩码，无需硬编码稀疏预算。

**🔧 技术方法**

使用了轻量化的能量掩码层、softplus、L2 正则化的能量项、局部邻域相似性（8邻域）约束、基于能量的持续正则化、以及可微分的门控策略。

**📊 数据集**

在 Food‑101、Oxford‑IIIT Pet、CUB‑200‑2011 等分类任务上进行验证，并在 ImageNet 预训练的 ResNet‑50、ConvNeXt‑Tiny、EfficientNetV2‑S 等骨干网络中插拔实验。

**📈 对比分析**

与基线 CNN、DropBlock、Spatial Dropout、L0‑gating 等方法对比，ERSM 在保持相近或略优的分类准确率（如 Food‑101 69.7%）的同时实现约 0.65 的稀疏度，并在按能量排序的补丁删除实验中显著优于随机删除，提升鲁棒性。

**⚠️ 局限性**

局限性包括：需手动调节 λ_unary 与 λ_pair 参数；对冻结骨干的依赖可能导致在背景与目标纹理相似时掩码失效；并且能量正则化在极端稀疏场景下可能削弱模型表达能力。

---

## 341. Telecom World Models: Unifying Digital Twins, Foundation Models, and Predictive Planning for 6G

**arXiv ID:** 2604.06882 | [PDF](https://arxiv.org/pdf/2604.06882v1)

**作者:** Hang Zou `[一作]` (Khalifa University), Mérouane Debbah `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Telecom World Model (TWM)，将可控系统世界与外部世界分离，形成三层架构（Field、Control/Dynamics、TelecomGPT），实现学习型、动作条件化、带不确定性、跨时间尺度的网络建模与决策。

**💡 创新点**

创新点在于融合世界模型、数字孪生与语言模型，构建统一的两世界三层架构，既具可解释的网络状态 grounding，又支持快速动作条件化 roll‑outs、校准不确定性、模型基规划与 LLM 保障安全。

**🔧 技术方法**

采用条件扩散模型或 CNN 为 Field World Model，RSSM 族或 Transformer 为 Control/Dynamics World Model，基于 TelecomGPT/大型语言模型实现意图翻译与工具编排；结合多尺度时序、概率推断与离散动作编码。

**📊 数据集**

数据来源包括真实网络观测（KPI、CSI、移动轨迹、故障日志）、高保真数字孪生仿真（Ray‑Tracing、网络级仿真）以及 O‑RAN 监控与日志；训练通过层级、离线预训练+在线校正。

**📈 对比分析**

在多域网络切片案例中，与单层基线（FWM、CDWM、DT 搜索、LLM 代理）和数字孪生公平搜索相比，TWM 在合规性与成本收益上实现 Pareto 最优（平均 SLA 合规率 45% 与成本降低 7%），明显优于单一模型。

**⚠️ 局限性**

局限在于样本稀缺导致离线训练误差、因子化假设易失真、跨层接口对齐困难、在线安全保障与可解释性不足，以及对极端事件与分布漂移的鲁棒性待提升。

---

## 342. Determinacy with Priorities up to Clocks

**arXiv ID:** 2604.06879 | [PDF](https://arxiv.org/pdf/2604.06879v1)

**作者:** Luigi Liquori `[一作]` (Centre Inria de l'Université Côte d'Azur), Claude Stolze `[通讯]` (University of Bamberg)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出并形式化了一种新的“连贯性”（coherence）概念，扩展了Milner的进程代数以包含优先级保护动作和时钟，并在此基础上给出了连贯性保持的证明（并行与限制运算的保持定理），从而实现了在共享内存并发与同步编程（如Esterel）中可组合的确定性语义。

**💡 创新点**

创新点在于：①将战略动作标签与阻塞/预测机制统一，生成能表达优先级和时钟同步的转移；②提出“可观测性”与“独立性”替代传统的确定性与不同标签的条件，形成连贯性定义；③证明连贯性在并行与限制下保持，突破了Milner对排序分离的限制；④将这些理论应用到同步编程语言的语义建模，解决了缺失缺失（absence）反应与全局一致性问题。

**🔧 技术方法**

使用了形式化语义技术：战略转移标签、阻塞关系、预测函数、构造性的 LTS、协变/逆变的证明技术、共递归（co-induction）来构造连贯性类；还借鉴了优先级守护、时钟广播、同步与异步分离的概念。

**📊 数据集**

无实验数据集，本文属于理论计算机科学范畴，主要通过形式化推导和定理证明来验证结果。

**📈 对比分析**

由于未进行实验实现，本文没有性能指标或与其他方法的对比；其贡献主要在理论上提供了更强的可组合性与确定性保证，未来可通过实现工具链或案例研究来进一步评估。

**⚠️ 局限性**

局限性包括：①对多时钟扩展与更复杂同步模型的支持仍待完善；②连贯性的表达性与可比性（与其他优先级/同步代数的关系）尚未完全阐明；③缺乏实际编程语言实现或工具验证，难以评估在真实系统中的可行性与效率；④自阻塞前缀等新特性在经典代数中缺失，可能导致某些表达式不可简化。

---

## 343. Modelling Distributed Applications with Mixed-Choice Stateful Typestates

**arXiv ID:** 2604.06874 | [PDF](https://arxiv.org/pdf/2604.06874v1)

**作者:** Francisco Parrinha `[一作]` (NOVA LINCS and NOVA FCT), António Ravara `[通讯]` (NOVA LINCS and NOVA FCT)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了一种在运行时监控分布式协议的框架，通过扩展传统的 typestate 模型加入内部可变状态、混合会话以及概率比例，以实现对协议行为的动态检测。

**💡 创新点**

创新点在于：① 将可变内部状态与谓词/赋值规则整合进 typestate，实现对状态计数、阈值触发的精细建模；② 引入混合会话，使单个状态下同时包含输入与输出操作；③ 在每个状态下为动作指定期望比例，并通过置信区间进行实时监测，首次将概率分布与 typestate 监控结合；④ 形成完整的语法、语义与监控推理规则，为监控器生成提供了理论基础。

**🔧 技术方法**

核心技术包括：typestate 语言设计与语法定义、内部状态管理（变量、赋值、谓词）、混合会话语义、概率比例与置信区间监控、推理规则与转移生成、与 JaTyC 交互的监控器生成。

**📊 数据集**

论文未使用公开数据集；通过两个经典分布式协议（Acknowledgement、Voting）示例演示模型的可表达性和监控流程。

**📈 对比分析**

缺乏实验评估和性能对比；论文仅在理论层面给出推理与监控规则，未给出运行时开销、误报率或与其它监控工具的基准结果。

**⚠️ 局限性**

局限性包括：① 仅针对基于动作比例的监控，无法处理复杂的时序/性能约束；② 对网络不可靠性仅通过概率比例做粗粒度估计，未提供细粒度的容错策略；③ 监控器与协议实现需保持一致，若协议实现细节变更可能导致监控失效；④ 未覆盖多方协议的双向同步检查，需进一步扩展。

---

## 344. Personalization as a Game: Equilibrium-Guided Generative Modeling for Physician Behavior in Pharmaceutical Engagement

**arXiv ID:** 2604.06860 | [PDF](https://arxiv.org/pdf/2604.06860v1)

**作者:** Suyash Mishra `[一作]` `[通讯]` (Roche), Suyash Mishra (Roche)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个名为EGPF的框架，将贝叶斯博弈、范畴论、信息论和生成式AI结合，用于药企与医生之间的动态个性化互动。

**💡 创新点**

创新点在于统一博弈理论、范畴论和信息论，提出Rate-Distortion平衡、Sheaf一致性、多尺度反馈，以及基于均衡策略的LLM生成内容。

**🔧 技术方法**

使用了贝叶斯博弈、Stackelberg、机制设计、演化动力学、范畴论的函子与自然变换、单子结构、层状Sheaf、一致性损失、信息瓶颈、KL与Fisher信息、RLHF对齐的LLM等技术。

**📊 数据集**

使用了SynthRx合成数据集（5种类型、50k轮交互）和真实HCPilot数据集（2847名肿瘤医生、18个月多渠道交互）。

**📈 对比分析**

与静态分段、协同过滤、深度序列模型、上下文Bandit等基线对比，EGPF-Full在AUC上比最佳基线提升34%（SynthRx）/13%（HCPilot 12mo），内容相关性提升约0.7分，信念收敛速度最快。

**⚠️ 局限性**

限制包括只处理离散有限类型、渠道假设为稳定、未考虑多玩家网络影响、LLM生成延迟高、未实现连续类型的mean‑field博弈等。

---

## 345. Tractable Hyperproperties for MDPs

**arXiv ID:** 2604.06859 | [PDF](https://arxiv.org/pdf/2604.06859v1)

**作者:** Lina Gerlach `[一作]` (RWTH Aachen University), Sebastian Junges `[通讯]` (Radboud University)

**通讯引用:** 2181 | [OpenAlex ID](https://openalex.org/A5018941708)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文定义了新的子类——关系性概率性质（relational probabilistic properties），用于描述不同执行集之间的概率关系，涵盖到达、安全、Büchi 和 co‑Büchi 目标。针对这类性质，提出了多种高效算法（如目标展开 + 期望奖励、MEC 商化 + 期望奖励、以及多目标可实现性查询）并给出了对应的复杂度分析。实验实现表明在满足其子类的基准测试上，算法速度比通用概率超逻辑（probabilistic hyperlogics）工具快数个数量级。

**💡 创新点**

创新点在于：①把关系性概率性质定义为可判定且可多项式时间（或固定参数可多项式）求解的子类；②将关系性质的判定转化为在目标展开（goal‑unfolding）或 MEC 商化后的单目标或多目标可实现性问题；③在多目标情形下引入扩展的多目标可实现性查询（MOA）并实现对 ≈_ϵ、≉_ϵ 等比较运算符的支持；④系统性地给出了复杂度上界与下界，划分出可多项式、FPT、NP/PSPACE‑hard 等不同子问题；⑤在实现层面将这些算法集成到现有 MDP 验证器上，并展示了显著的性能提升。

**🔧 技术方法**

主要技术包括：
- 目标展开（goal‑unfolding）将到达概率转化为期望奖励；
- 期望奖励计算（线性规划/值迭代）；
- MEC 商化（MEC quotient）用于处理 Büchi/ co‑Büchi 目标；
- 多目标可实现性查询（MOA）与扩展 MOA 的线性/SMT 编码；
- 记忆化与随机化调度器的构造与转换；
- 固定参数可多项式（FPT）分析与复杂度归约。

**📊 数据集**

使用了与概率超逻辑相关的标准基准集（如 PRISM/Storm 测例），并挑选其满足关系性质子类的子集进行实验。基准数据主要包含各种 MDP、到达/安全/ Büchi/ co‑Büchi 目标的组合。

**📈 对比分析**

与通用概率超逻辑工具（如Probabilistic HyperLogic solvers）对比，本文实现的算法在满足其子类的基准上平均快了数个数量级（从几秒到毫秒级别）。实验报告显示，即使在较大 MDP（数千状态）和多目标（数十个目标集）情况下，算法仍保持可行，并且在可多项式或 FPT 的子类上几乎无显著性能损失。

**⚠️ 局限性**

限制与不足：
- 对于包含 ≉_ϵ（ε>0）或多目标的某些组合，问题仍为 NP/PSPACE‑hard，无法得到多项式时间解；
- 对于记忆化/随机化调度器的需求使得某些子问题不再可简化为记忆化确定性调度器；
- 需要对目标集数目或吸收性做限制才能保证 FPT 或多项式时间；
- 仅支持存在量化的关系式，无法直接处理交替量化的超逻辑；
- 现有实现主要针对到达/安全/ Büchi/ co‑Büchi 目标，尚未覆盖更一般的奖励/支付函数。
这些限制表明，尽管关系性质子类可大幅提升可判定性与效率，但在更广泛的概率超逻辑场景中仍需进一步研究。

---

## 346. MirageBackdoor: A Stealthy Attack that Induces Think-Well-Answer-Wrong Reasoning

**arXiv ID:** 2604.06840 | [PDF](https://arxiv.org/pdf/2604.06840v1)

**作者:** Yizhe Zeng `[一作]` (Chinese Academy of Sciences), Yuling Liu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 12614 | [OpenAlex ID](https://openalex.org/A5100414459)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用后输出空间实现的思考好却回答错的后门攻击 MirageBackdoor，能够在触发时保持干净的链式思考并仅篡改最终答案。

**💡 创新点**

创新点在于将后输出空间作为训练时的辅助通道，通过两阶段（SFT+RL）训练，使后门在推理时完全隐藏且不破坏思考过程，从而显著提升了隐蔽性与数据效率。

**🔧 技术方法**

采用的技术包括后输出分隔符、语义触发、结构化毒化样本、两阶段训练（监督微调+强化学习）以及自评奖励机制。

**📊 数据集**

使用了四个推理基准数据集：GSM8K、AQuA‑RAT、ECQA 与 MathQA，并在 Qwen2.5、Llama3 系列等多种开源模型上进行实验。

**📈 对比分析**

与 BadChain、DecepChain、SFT+RL 等基线相比，MirageBackdoor 在 5% poison ratio 下平均 ASR 超过 90%，且在保持清洁准确率的同时显著降低了 PPL 并提升了 CoT Soundness Rate，表现出更高的成功率与隐蔽性。

**⚠️ 局限性**

局限性包括仅在四个推理基准上验证，未覆盖代码生成或工具辅助推理；对抗检测仍可通过 CoT‑答案一致性检查，且缺乏对后门机制的机制解释。

---

## 347. On the Step Length Confounding in LLM Reasoning Data Selection

**arXiv ID:** 2604.06834 | [PDF](https://arxiv.org/pdf/2604.06834v1)

**作者:** Bing Wang `[一作]` (Jilin University), Jieping Ye `[通讯]` (Alibaba Cloud Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并解决在长链推理数据集上自然性（log‑probability）数据选择中出现的步长偏倚（step length confounding）问题

**💡 创新点**

发现低概率的第一步Token导致平均log概率被拉高，从而偏向长步长样本；提出两种去偏方案：Drop‑First（直接忽略首Token）与Causal‑Debias（回归去除首Token比例影响）

**🔧 技术方法**

log‑probability 计算、局部 log‑probability、熵评估、线性回归因子去偏、数据选择与SFT训练

**📊 数据集**

LIMO‑v2、AceReason‑1.1‑SFT、AIME24/25、MATH500、OlympiadBench、GPQA（作为评测基准）

**📈 对比分析**

与现有自然性选择方法（GRACE、Local LP）对比，四种大型LLM（Qwen3‑4B‑Base、8B‑Base、4B‑Instruct、7B‑Instruct）在五个评测基准上平均提升≈6.3%（Drop）和≈9.1%（Causal），并显著缓解步长偏倚

**⚠️ 局限性**

仅关注首Token对步长偏倚的影响，未探究其他潜在混杂因子；方法在自回归式on‑policy生成的训练场景下效果尚未验证

---

## 348. WRAP++: Web discoveRy Amplified Pretraining

**arXiv ID:** 2604.06829 | [PDF](https://arxiv.org/pdf/2604.06829v1)

**作者:** Jiang Zhou `[一作]`, Feng Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过跨文档关联发现和联合 QA 生成，扩增了知识的关联上下文，提升 LLM 预训练效果。

**💡 创新点**

首次利用网页链接拓扑发现双向链接和共提议模型，生成跨文档的关系式 QA，实现数据量和质量双重放大。

**🔧 技术方法**

结合图结构关系发现、指令调优生成模型（Qwen3‑30B‑A3B）以及三条合成约束（跨文档依赖、事实链条、全局内部化）。

**📊 数据集**

在英文 FineWiki（约 8.4B tokens）和 SimpleQA 基准上进行实验。

**📈 对比分析**

相较于传统单文档 WRAP 及其扩展版，在 7B/32B OLMo 模型上通过 1 轮继续预训练，SimpleQA pass@128 提升约 9.5–9.8 百分点，显示更好的知识检索与可扩展性。

**⚠️ 局限性**

对网页链接的依赖限制在可链接的文档上，且对生成模型规模和计算成本敏感；随机对齐会导致假关联，导致质量下降。

---

## 349. Beyond End-to-End: Dynamic Chain Optimization for Private LLM Adaptation on the Edge

**arXiv ID:** 2604.06819 | [PDF](https://arxiv.org/pdf/2604.06819v1)

**作者:** Yebo Wu `[一作]` (University of Macau), Li Li `[通讯]` (University of Macau)

**通讯引用:** 128316 | [OpenAlex ID](https://openalex.org/A5100364769)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在联邦学习框架下提出链式微调（Chain Federated Fine‑Tuning）解决LLM在边缘设备上显存不足的问题。

**💡 创新点**

核心创新是链式优化理念（逐层训练冻结）以及三大技术：动态层级协同调优、全局感知优化和功能导向自适应调优。

**🔧 技术方法**

采用联邦学习、轻量级适配器、动态层协同、全局损失辅助、CKA功能划分等技术。

**📊 数据集**

使用文本分类数据集YELP-P、AGNEWS、YAHOO、20NEWS，以及指令调优数据集Alpaca‑GPT4，评测指标包含MMLU、BBH、DROP、CRASS。

**📈 对比分析**

与内存无关和有关的联邦微调基线以及全模型端到端训练对照，平均准确率提升达46.46%，内存峰值下降多达16.87倍。

**⚠️ 局限性**

局限在仅验证Transformer NLP任务，未测试其他架构或多模态模型，也未结合差分隐私等隐私增强技术。

---

## 350. Event-Triggered Adaptive Consensus for Multi-Robot Task Allocation

**arXiv ID:** 2604.06813 | [PDF](https://arxiv.org/pdf/2604.06813v1)

**作者:** Fidel Aznar `[一作]` (University of Alicante), Álvaro Díez `[通讯]` (University of Alicante)

**通讯引用:** 712 | [OpenAlex ID](https://openalex.org/A5082430655)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于事件触发的自适应共识框架（CBBA-ETC），用于在异构机器人群中高效、低通信量地完成多机器人任务分配。

**💡 创新点**

核心创新点包括：
• 仅在重要事件触发时才发起共识通信，显著降低网络负载；
• 通过自适应回退间隔动态调节协作频率，以应对不同冲突强度；
• 将行为树（Behavior Tree）嵌入本地执行与事件检测，提升局部容错与自适应能力；
• 结合上述机制实现了与传统CBBA、C‑CBBA及CBBA‑Tree相媲美甚至更优的任务完成率，同时通信成本降至最低。

**🔧 技术方法**

使用技术包括：Consensus‑Based Bundle Algorithm (CBBA)、事件触发控制 (ETC)、Behavior Trees、动态回退间隔自适应机制、SimPy仿真框架；对比算法有 CBBA、C‑CBBA、CBBA‑Tree、Comm、Tree。

**📊 数据集**

使用自建的搜索与救援（SAR）仿真环境（圆形场景、随机出现颜色标识的任务、机器人具备视角、通信和动作失败模型）。并未使用公开现实数据集，而是基于参数化仿真。

**📈 对比分析**

通过在 50 次 Monte‑Carlo 试验中，系统在多种场景（机器人/任务密度、通信丢包、动作失败、永久失效）下进行比较。指标包括任务完成数、通信消息数、失败救援数等。结果显示 CBBA‑ETC 在所有实验中保持了最高或相近的任务完成率，同时通信消息数比 CBBA、C‑CBBA、Comm 等低 1–2 位数，显著提升了通信效率。

**⚠️ 局限性**

限制：
• 仍需一定程度的网络通信；在极度通信受限或网络极端不可靠的环境下效果可能受限；
• 事件触发阈值及自适应参数需要针对具体任务空间手工调优；
• 仿真仅考虑无障碍开阔场景，缺乏对复杂地形、障碍避障的评估；
• 目前未验证在真实机器人平台上的实时性能与能耗。

---

## 351. Towards Multiparty Session Types for Highly-Concurrent and Fault-Tolerant Web Applications

**arXiv ID:** 2604.06878 | [PDF](https://arxiv.org/pdf/2604.06878v1)

**作者:** Richard Casetta `[一作]` (BNP Paribas), Pierre Genevès `[通讯]` (University Grenoble Alpes)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

设计了一种扩展MPST的全局类型框架，加入超时、错误处理、动态线程生成与崩溃语义。

**💡 创新点**

在MPST中显式加入失败语义和动态参与，定义语法、LTS语义及一致性规则，并证明一致性保持和无孤立参与者。

**🔧 技术方法**

采用多方会话类型理论、标签转换系统、递归、语义一致性判定、类型系统证明等技术。

**📊 数据集**

本文为理论工作，无使用实验数据集。

**📈 对比分析**

通过形式化证明展示一致性与安全性，并未进行实测比较，理论上提升了对失败与动态参与的建模能力。

**⚠️ 局限性**

仍未实现局部类型投影和自动化liveness检查；对状态更新的支持有限；未给出工具实现与实测性能。

---

## 352. Exploiting Aggregate Programming in a Multi-Robot Service Prototype

**arXiv ID:** 2604.06876 | [PDF](https://arxiv.org/pdf/2604.06876v1)

**作者:** Giorgio Audrito `[一作]` (University of Torino), Gianluca Torta `[通讯]` (University of Torino)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

实现了基于聚合编程的多机器人图书馆服务系统，实现了任务分配与导航协同；

**💡 创新点**

首次将聚合编程直接部署到真实机器人上，且实现了自稳、容错、动态任务分配；

**🔧 技术方法**

采用聚合编程框架FCPP、ROS2、Gazebo模拟与iRobot Create3硬件，结合LeaderElection等自稳算子；

**📊 数据集**

利用Gazebo模拟的大学图书馆环境（3.5 m×6.0 m）及Create3机器人原型，未使用公开数据集；

**📈 对比分析**

通过模拟与真实机器人实验验证，系统在网络分区、节点失效、电量耗尽等场景下保持一致性，性能满足实时需求；

**⚠️ 局限性**

局限在于需手工配置WiFi/UDP驱动，受限于机器人硬件（仅轮式、激光雷达），且对更复杂感知任务尚未验证。

---

## 353. Generating Local Shields for Decentralised Partially Observable Markov Decision Processes

**arXiv ID:** 2604.06873 | [PDF](https://arxiv.org/pdf/2604.06873v1)

**作者:** Haoran Yang `[一作]` (University of Oxford), Nobuko Yoshida `[通讯]` (University of Oxford)

**通讯引用:** 12073 | [OpenAlex ID](https://openalex.org/A5054171989)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种用于Dec-POMDP的盾牌过程代数，并实现了从过程代数到局部Mealy机的编译管道，用于在无通信、局部可观测的多智能体系统中动态限制安全动作；

**💡 创新点**

创新点在于将全局安全行为用递归与守卫选择的过程代数描述，并通过观测映射构造信念子集实现局部安全过滤；

**🔧 技术方法**

使用了Rust实现编译管道，结合PRISM进行概率模型检查，利用Mealy机、状态机理论以及信念状态构造；

**📊 数据集**

以多智能体路径规划（MAPF）为案例，生成随机网格（3×3、4×4、5×5）并放置不同数量障碍的实例进行实验；

**📈 对比分析**

通过比较无盾牌、保守盾牌P1、最小保守盾牌P2三种方案的碰撞率、目标达成率和最佳/最差安全概率，结果显示P2在保持零碰撞的同时显著提升目标达成率，优于P1和无盾牌；

**⚠️ 局限性**

局限在于盾牌P1过于保守、P2仍需手工指定过程代数、缺乏自动化分解方法，且对复杂环境的可扩展性和多智能体交互的完整性尚未充分验证。

---

## 354. REAgent: Requirement-Driven LLM Agents for Software Issue Resolution

**arXiv ID:** 2604.06861 | [PDF](https://arxiv.org/pdf/2604.06861v1)

**作者:** Shiqi Kuang `[一作]` (Tianjin University), Junjie Chen `[通讯]` (Tianjin University)

**通讯引用:** 6029 | [OpenAlex ID](https://openalex.org/A5100365536)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 REAgent，基于需求驱动的 LLM 代理框架，自动生成、评估并迭代改进问题导向的结构化需求，从而提升仓库级软件问题的修复效果。

**💡 创新点**

创新点在于将软件需求工程的结构化需求理念引入问题修复，构建 issue‑oriented 需求并通过 Requirement Assessment Score 评估需求质量，随后按冲突/遗漏/歧义三类缺陷提供针对性反馈，实现需求的自动化迭代改进。

**🔧 技术方法**

技术包括：LLM 代理与工具调用（文件检索、代码搜索、测试执行）、预定义需求属性的需求建模、需求评估（利用生成补丁和测试结果计算 RAS）、需求缺陷分类与反馈生成、固定次数迭代优化。

**📊 数据集**

使用 SWE‑bench Lite、Verified、Pro 三个基准（各采样 100 个问题）以及 DeepSeek‑V3.2 与 Qwen‑Plus 两大 LLM 作为底层模型。

**📈 对比分析**

与 BM25、Agentless、Trae‑agent、ArchCode、Specine 五种基线在 %Resolved、%Applied、Token 使用量和成本等指标上进行对比；REAgent 在所有指标上均优于基线，%Resolved 提升约 10–25%，%Applied 提升约 22–50%，但 Token/成本相对较高。

**⚠️ 局限性**

局限性包括：整体计算开销较大（Token 与成本显著高于某些基线），对 LLM 生成测试的质量仍受限，迭代次数固定且未实现自适应控制，且需求缺陷检测仍易受模型幻觉影响。

---

## 355. To Adapt or not to Adapt, Rethinking the Value of Medical Knowledge-Aware Large Language Models

**arXiv ID:** 2604.06854 | [PDF](https://arxiv.org/pdf/2604.06854v1)

**作者:** Ane G. Domingo-Aldama `[一作]` (University of Basque Country), Ander Barrena `[通讯]` (University of Basque Country)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估并对比通用与医学领域的LLMs，在英西多项MCQA任务下引入扰动评测，并发布Marmoka医学适配模型。

**💡 创新点**

设计了扰动与两步转化评测框架，揭示医学LLMs在英语短答场景下提升有限，首次在低资源语言西班牙语上验证适配效果。

**🔧 技术方法**

基于Llama 3.1 8B‑Instruct进行连续域适配预训练与指令微调，合并多模型得到Marmoka系列；评测采用扰动生成、链式思考、摘要、改写等技术。

**📊 数据集**

英文数据集：MMLU、PubMedQA、MedQA、CareQA‑En；西班牙语数据集：Casimedicos‑exp、CareQA‑Es；医学语料来源包括WikiMed、PubMed、EMEA、MedCrawler、SciELO等。

**📈 对比分析**

采用开放式生成单字答案与扰动版MCQA进行对比；结果显示英文本任务医学模型提升极小，西班牙语模型明显优于通用模型；两步转化普遍降低性能，指令遵循率低。

**⚠️ 局限性**

评测仅限短形式MCQA，缺乏深度推理与专家验证；指令格式遵循差；模型受限于8B规模与Llama 3.1 架构；未系统评估临床安全与偏差。

---

## 356. CloudMamba: An Uncertainty-Guided Dual-Scale Mamba Network for Cloud Detection in Remote Sensing Imagery

**arXiv ID:** 2604.06844 | [PDF](https://arxiv.org/pdf/2604.06844v1)

**作者:** Jiajun Yang `[一作]` (Beihang University), Zhenwei Shi `[通讯]` (Beihang University)

**通讯引用:** 14964 | [OpenAlex ID](https://openalex.org/A5058849690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于双尺度Mamba网络与不确定性引导的两阶段云检测框架CloudMamba，用于遥感影像中的云分割；

**💡 创新点**

创新点在于将卷积与状态空间模型（Mamba）融合为双尺度模块，同时设计了自适应的不确定性评估与二阶段细化机制，显著提升了薄云与边界细节的识别；

**🔧 技术方法**

技术包括CNN-SSM混合感知模块、双尺度Mamba块、基于SS2D的二维状态空间建模、不确定性估计与接受掩膜、以及轻量化的第二阶段refiner；

**📊 数据集**

使用了GF1_WHU和Levir_CS两大公开云检测数据集，数据经过512×512裁剪后分别训练和测试；

**📈 对比分析**

与U-Net、DeepLabV3+、BoundaryNets、U-Mamba等方法对比，CloudMamba在mIoU、F1和OA上分别提升约1–2%并在硬样本上表现更稳健；

**⚠️ 局限性**

局限包括额外的两阶段推理导致计算开销略增，对不确定性估计仅采用经验公式，且双尺度Mamba的膨胀率设定仍需手动调优。

---

## 357. Explaining Neural Networks in Preference Learning: a Post-hoc Inductive Logic Programming Approach

**arXiv ID:** 2604.06838 | [PDF](https://arxiv.org/pdf/2604.06838v1)

**作者:** Daniele Fossemò `[一作]` (University of L'Aquila), Fabio Aurelio D'Asaro `[通讯]` (University of Verona)

**通讯引用:** 120 | [OpenAlex ID](https://openalex.org/A5050767635)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出利用ILASP（Inductive Learning of Answer Set Programs）对已训练的深度神经网络（NN）在用户偏好学习任务中的黑盒行为进行后置解释，生成可解释的弱约束理论；

**💡 创新点**

创新点包括：① 将主成分分析（PCA）两种方式（直接与间接）嵌入ILASP学习流程，以缓解高维特征空间导致的指数搜索；② 区分全局与局部解释的采样与训练策略；③ 设计基于真实偏好标签的ground‑truth评估指标，直接衡量解释与人类偏好的一致性；

**🔧 技术方法**

技术手段为：ILASP（ASP）、弱约束逻辑学习、PCA（降维）、高斯噪声采样、实验评估指标（Fidelity、Precision/Recall、执行时间、理论长度）；

**📊 数据集**

使用自建食谱偏好数据集：100道意大利菜谱（36类食材、12类宏观分类），54名用户共计11,340条有标签的三元组偏好；

**📈 对比分析**

与原NN性能对比，实验显示全局解释Fidelity约70%，局部解释可达90%以上；在无PCA时ILASP耗时高达数万秒，使用间接PCA约减少至千秒，直接PCA线性缩放；同时，间接PCA获得更优的ground‑truth准确率；

**⚠️ 局限性**

局限性：ILASP对高维特征仍存在指数时间增长；局部训练样本中不确定标签比例高时容易产生空理论；对噪声和PCA参数选择敏感；缺乏在大规模真实场景中的可扩展验证。

---

## 358. STQuant: Spatio-Temporal Adaptive Framework for Optimizer Quantization in Large Multimodal Model Training

**arXiv ID:** 2604.06836 | [PDF](https://arxiv.org/pdf/2604.06836v1)

**作者:** Minglu Liu `[一作]` (Xidian University), Fu Yu `[通讯]` (China Telecom Cloud Computing Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种分层、时空自适应的优化器状态量化框架（STAF-Q），在大规模多模态模型训练中动态分配低精度位宽以降低显存占用。

**💡 创新点**

创新点包括：①基于梯度强度与变异度的双因子敏感性代理，精准识别层级重要性；②引入时间退火因子与层级特征因子，实现跨层、跨阶段的自适应位宽选择；③通过对量化决策的分层映射，将指数级搜索空间压缩为线性复杂度。

**🔧 技术方法**

使用的技术包括：梯度统计（RMS、CV、EMA）、双模式量化（第一阶线性量化、第二阶对数量化）、块级分段量化、分布式统计同步、指数衰减的时间退火调度。

**📊 数据集**

在 GPT‑2、ViT、以及 OpenWebText、COCO、Wikitext‑103、MNLI 等标准数据集上进行实验，覆盖语言、视觉与多模态任务。

**📈 对比分析**

与全精度 AdamW 和 8‑bit bitsandbytes AdamW 进行对比，STAF‑Q 在保持或略优的收敛稳定性、下游任务性能（如 PPL、mAP、Recall@1）基础上，将优化器状态内存压缩至 84.4%（平均位宽仅 5.1 bits）并显著降低显存占用。

**⚠️ 局限性**

局限性：当前依赖于梯度统计的近似，可能在梯度稀疏或极端噪声环境下失效；尚未在更大规模（百亿级）模型或不同优化器（如 Lion、Adam‑mini）上验证；对硬件的子位支持仍有限。

---

## 359. Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning

**arXiv ID:** 2604.06824 | [PDF](https://arxiv.org/pdf/2604.06824v1)

**作者:** Subin Park `[一作]` (Kyung Hee University), Jung Uk Kim `[通讯]` (Kyung Hee University)

**通讯引用:** 27734 | [OpenAlex ID](https://openalex.org/A5100447941)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种完全无训练、基于生成-分析-精炼（GAR）的声音源定位框架，利用多模态大语言模型（MLLM）的推理能力实现音视源对齐。

**💡 创新点**

创新点在于将定位任务转化为人类认知的多步推理流程，引入开放式角色标注、锚点投票和自适应门控机制，实现可解释且高效的定位。

**🔧 技术方法**

核心技术为生成阶段的跨模态定位与音频分类、分析阶段的角色标注与锚点投票、精炼阶段的几何调整，并全部通过Prompt工程实现。

**📊 数据集**

使用公开的VGGSound（单源与双源）和MUSIC（单源与双源）数据集进行评估。

**📈 对比分析**

与传统基于对比学习的SSL方法和纯MLLM基线相比，GAR在单源和双源任务上均实现了显著提升，单源AP提升约8%，双源CIoU提升约17%。

**⚠️ 局限性**

局限性包括推理迭代导致的推理时间增加、对所选MLLM的性能高度依赖，以及缺乏时间序列推理的扩展。

---

## 360. Time-driven Survival Analysis from FDG-PET/CT in Non-Small Cell Lung Cancer

**arXiv ID:** 2604.06885 | [PDF](https://arxiv.org/pdf/2604.06885v1)

**作者:** Sambit Tarai `[一作]` (Uppsala University), Joel Kullberg `[通讯]` (Uppsala University)

**通讯引用:** 7993 | [OpenAlex ID](https://openalex.org/A5075154897)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文开发了一种基于深度学习的时间驱动框架，利用组织分层的FDG‑PET/CT投影与时间输入实现对NSCLC患者总体生存时间的连续预测并进行风险分层。

**💡 创新点**

创新点在于将时间信息与影像嵌入通过逐元素相乘融入模型，实现了对任意时间点生存概率的预测，突破了传统固定间隔二分类限制，并通过影像与临床+IDP特征的融合提升了预测性能。

**🔧 技术方法**

技术上采用ResNet‑50作为影像特征提取器，配合前馈神经网络处理时间特征，随后将两者融合后通过全连接层输出生存概率，使用焦点损失+生存一致性损失训练，并通过Grad‑CAM进行可视化。

**📊 数据集**

数据集为瑞典U‑CAN队列中的NSCLC子集，包含848例完整FDG‑PET/CT影像及临床变量，按556例手工分割训练/验证集与292例自动分割测试集划分。

**📈 对比分析**

与基线ResNet‑50单纯影像预测、DeepHit、DeepSurv等传统模型比较，本文方法在0.5–5年间隔的AUC平均为0.746，明显优于基线0.703及DeepHit 0.693、DeepSurv 0.717，最优集成模型AUC达0.788。

**⚠️ 局限性**

局限性包括缺乏外部多中心验证、对时间点多样化训练所需计算资源高、模型对临床缺失值敏感，以及对复杂分期（N、M）的预测尚未覆盖。

---

## 361. Physical Adversarial Attacks on AI Surveillance Systems:Detection, Tracking, and Visible--Infrared Evasion

**arXiv ID:** 2604.06865 | [PDF](https://arxiv.org/pdf/2604.06865v1)

**作者:** Miguel A. DelaCruz `[一作]`, Rafael T. Navarro `[通讯]` (De La Salle University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了物理对抗攻击在实际监视系统中的研究，重点讨论人检测、跟踪、可见-红外融合以及可穿戴载体实现。

**💡 创新点**

提出了以时间持续性、感知模态、物理载体、系统级目标为四维的监视中心分类法，并阐述了从单帧检测到多目标跟踪、跨模态攻击以及可穿戴激活的威胁模型演进。

**🔧 技术方法**

综述中引用的技术包括对抗样本生成、EOT鲁棒化、跨模态优化、热式可穿戴载体设计以及多目标跟踪中的数据关联攻击。

**📊 数据集**

参考了公开的检测与跟踪数据集（如COCO、ImageNet、MOTChallenge、VisPR等），以及可见-红外合成与实测数据。

**📈 对比分析**

通过对文献评估指标的比较，提出了五阶段评估阶梯（数字测试→实验室测试→操作变异→时间持续→跨模态风险），指出单帧指标不足以衡量监视系统鲁棒性，文献中对抗效果多以成功率、ID切换率等为衡量。

**⚠️ 局限性**

局限性在于仅为综述，未给出统一的实验基准与评价指标；对跨模态与可穿戴攻击的定量评估不足；对防御方法的系统性分析仍不充分，且缺乏真实场景下的大规模验证。

---

## 362. Vision-Language Model-Guided Deep Unrolling Enables Personalized, Fast MRI

**arXiv ID:** 2604.06849 | [PDF](https://arxiv.org/pdf/2604.06849v1)

**作者:** Fangmao Ju `[一作]` (Xi'an Jiaotong University), Jianhua Ma `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 7100 | [OpenAlex ID](https://openalex.org/A5006049117)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了 PASS（Personalized, Anomaly-aware Sampling and reconStruction）框架，利用 Vision‑Language 模型指导 MRI 的自适应采样和深度 unrolling 重建，实现了个性化的加速成像。

**💡 创新点**

创新点包括：① 通过细化 CLIP 视觉‑语言模型生成病变注意力图，形成高层次的异常先验；② 将该先验嵌入物理感知的深度 unrolling 网络中，既保持解释性又提升学习能力；③ 开发自适应采样网络，使 k‑space 采样轨迹可针对每位患者的病变特征动态调整，构成闭环采集‑重建‑诊断流程。

**🔧 技术方法**

核心技术：Vision‑Language 模型（Fine‑tuned CLIP + Pixel/Image‑level Adapters）、深度 unrolling 重建网络（数据一致性 + 个人化异常模块）、ADMM+CG 迭代求解、可学习采样网络（LOUPE + 病变关注采样层）、多尺度损失（全局与病变区域对齐）以及 1D/2D k‑space 采样实现。

**📊 数据集**

实验数据集：fastMRI 公开脑部（T1w、FLAIR）和膝部（PD）数据，配合 fastMRI+ 的病变标签与边框进行评估。

**📈 对比分析**

与传统模型（GRAPPA、ISTA‑Net、MoDL）、纯数据驱动方法（SwinMR、Reflow、Nail、PDAC）以及可学习采样方法（LS‑MoDL、LOUPE）对比；PASS 在 PSNR、SSIM、LF‑PSNR、LF‑SSIM 以及下游任务（异常检测 AUC、病变分类准确率）上均显著优于对手，甚至在某些指标上逼近或超过全量采样结果。

**⚠️ 局限性**

局限性：① VLM 预训练基于自然图像，医学专属优化不足；② 采样依赖 ACS 低频信息，可能忽略高频病变细节；③ 仅在 fastMRI 公开数据上验证，缺乏更多解剖或对比机制的泛化；④ 所有实验为回顾性，缺少临床现场验证；⑤ 采样网络目前主为 1D，2D 高加速仍待进一步研究。

---

## 363. MedDialBench: Benchmarking LLM Diagnostic Robustness under Parametric Adversarial Patient Behaviors

**arXiv ID:** 2604.06846 | [PDF](https://arxiv.org/pdf/2604.06846v1)

**作者:** Xiaotian Luo `[一作]` (Shanda Group), Jiangcheng Wu `[通讯]` (Shanda Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了一个可控的医学对话基准 MedDialBench，用于量化不同患者行为维度对大型语言模型诊断准确性的影响。

**💡 创新点**

首次引入五维行为分解（逻辑一致性、健康认知、表达方式、披露、态度），并通过分级严重度与因子设计实现剂量-反应与跨维交互分析。

**🔧 技术方法**

使用前代LLM（Claude Opus 4.5）作为患者代理生成行为脚本，五种前沿LLM（Gemini 3.1 Pro、GPT-5.4、Claude Opus 4.6、DeepSeek V3.2、Qwen 3.5 Plus）做诊断，并采用LLM判定器和McNemar等统计方法评估准确性。

**📊 数据集**

基于107个OSCE病例提取的85个可诊断案例，结合手工生成的案例特定行为脚本和对话记录，共计7225条对话。

**📈 对比分析**

通过与基线对照和单维、组合配置的准确率对比，发现信息污染（特别是fabricating）导致1.7–3.4倍的准确率下降，所有模型在最差配置下跌幅从38.8%到54.1%，并揭示了超级加法交互效应。

**⚠️ 局限性**

局限在于仅使用单一患者代理、固定行为维度、未覆盖体格检查或实验室测试、表达方式维度未同时限制信息量以及数据规模相对较小。

---

## 364. VGGT-SLAM++

**arXiv ID:** 2604.06830 | [PDF](https://arxiv.org/pdf/2604.06830v1)

**作者:** Avilasha Mandal `[一作]` (Indian Institute of Technology Delhi), Chetan Arora `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 2264 | [OpenAlex ID](https://openalex.org/A5019739552)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `51c0528b-f690-4182-ae60-bb5f046c276c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种端到端的Transformer视觉SLAM系统VGGT‑SLAM++，通过将VGGT生成的稠密子地图转换为数字高程图（DEM），并在此基础上进行高频局部束束调整，实现对前端里程计漂移的实时纠正。

**💡 创新点**

创新点包括：①利用DEM的几何保留特性与DINOv2无监督嵌入相结合，构建稀疏可检索的可见性图；②在回端以高频率执行局部束调整，弥补传统SLAM循环闭锁稀疏、后延迟的缺陷；③整合Transformer前端、DEM后端和VPR的协同工作，兼顾几何稳定与语义可扩展性。

**🔧 技术方法**

核心技术包括：Visual Geometry Grounded Transformer（VGGT）作为前端稠密重建；数字高程图（DEM）渲染与投影；DINOv2自监督视觉编码器用于结构化特征嵌入；FAISS‑HNSW近似检索构建可见性图；AnyLoc视觉场景识别；Sim(3)空间优化与局部束束调整。

**📊 数据集**

实验使用了KITTI Odometry、TUM RGB‑D、7‑Scenes、Virtual KITTI、EuRoC MAV等公开数据集，以及多条自采集的GoPro和OAK‑1相机数据。

**📈 对比分析**

与传统特征/学习型SLAM（如ORB‑SLAM2、DROID‑SLAM、DPV‑SLAM、MASt3R‑SLAM）以及VGGT‑SLAM基线相比，VGGT‑SLAM++在绝对轨迹误差（ATE）上平均提升≈18.6%，在KITTI、TUM、Virtual KITTI等多种场景下均达到或超过SOTA；同时保持接近实时的帧率（前端≈16 FPS，后端≈1.9 FPS）和可控内存。

**⚠️ 局限性**

局限性主要是：①VGGT前端在单灰度图像上的精度受限（训练仅用RGB），导致在灰度序列上性能下降；②系统依赖于高质量的数字高程图生成，若地面不平坦或遮挡严重时DEM特征可能不可靠；③虽然内存受控，但对高分辨率场景仍需显存与CPU负载较高。

---

## 365. Beyond Surface Judgments: Human-Grounded Risk Evaluation of LLM-Generated Disinformation

**arXiv ID:** 2604.06820 | [PDF](https://arxiv.org/pdf/2604.06820v1)

**作者:** Zonghuan Xu `[一作]` (Fudan University), Xingjun Ma `[通讯]` (Fudan University)

**通讯引用:** 6972 | [OpenAlex ID](https://openalex.org/A5078711649)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（LLM）作为评判者在识别并评估生成式虚假信息可信度与分享意愿时的有效性进行人类对照的系统审计

**💡 创新点**

提出将“读者面向评估”视为代理有效性问题，并从整体评分、文本级排序和文本信号依赖三个维度细致比较 LLM 判别者与人类读者的对齐情况，揭示内部一致性并不等同于真实世界读者反馈

**🔧 技术方法**

使用前沿 LLM 判别者（Claude、Gemini、GPT 系列）对文本评分；通过 LLM 生成的情绪强度、逻辑严谨、权威引用和数据密度等可解释信号进行对比；利用 Spearman 相关、偏差分析和信号依赖度量等统计方法评估对齐

**📊 数据集**

构建 290 篇目标导向的欺骗性文章（基于 Reuters 事实核查场景与 UN SDG 主题），收集 392 名参与者提供的 2,043 对人类可信度与分享意愿评分，并由 8 名前沿 LLM 判别者给出同一文本的评分

**📈 对比分析**

比较方法：对齐度量包括整体评分偏差、Spearman 相关（人类-判别者 vs 判别者-判别者）以及信号依赖度量；结果显示判别者总体更为严厉，重现人类排序的相关性仅为 0.45（可信度）和 0.24（分享），与人类-判别者的 0.34/0.18 相比，判别者之间的相似度高达 0.81/0.69

**⚠️ 局限性**

局限性包括：仅覆盖 5 个公共议题且场景多样性有限；判别者对文本特征的优化尚未探索；不同提示工程或内部优化可能进一步影响对齐，但当前结果表明即使提示改变也未能显著提升与人类的匹配度

---

## 366. SCT-MOT: Enhancing Air-to-Air Multiple UAVs Tracking with Swarm-Coupled Motion and Trajectory Guidance

**arXiv ID:** 2604.06883 | [PDF](https://arxiv.org/pdf/2604.06883v1)

**作者:** Zhaochen Chu `[一作]` (Beijing Institute of Technology), Siqing Cheng `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个名为SCT-MOT的多无人机空对空跟踪框架，能够在复杂群体运动与弱视觉特征环境下实现高精度轨迹追踪。

**💡 创新点**

创新点在于同时引入群体耦合运动预测模块（SMTP）和轨迹引导时空特征融合模块（TG-STFF），实现从群体层面捕捉运动依赖并利用预测轨迹指导特征融合，显著提升了检测与关联的鲁棒性。

**🔧 技术方法**

使用了多头注意力、图卷积网络、时空残差卷积、预测特征生成的高斯映射、跨帧交叉注意力等深度学习技术，并结合 YOLOX/ResNet18 进行检测与特征提取。

**📊 数据集**

在三大公开无人机集群跟踪数据集上进行评估：AIRMOT、MOT-FLY 和 UAVSwarm。

**📈 对比分析**

与多种 SOTA 方法（如 FairMOT、DeepSORT、ByteTrack、HOMATracker 等）进行比较，SCT-MOT 在 MOTA、IDF1、HOTA 上均优于对手，ID 切换数下降约 30，速度约 21.8 FPS。

**⚠️ 局限性**

局限性包括：对极端遮挡或极低分辨率 UAV 的检测仍易出现误检；模型对历史窗口长度和模型超参较为敏感；训练需要较大的 GPU 资源；轻量化版虽然可在边缘设备上实时运行，但性能仍低于桌面级模型。

---

## 367. Predicate Subtypes in VerCors

**arXiv ID:** 2604.06877 | [PDF](https://arxiv.org/pdf/2604.06877v1)

**作者:** Tycho Dubbeling `[一作]` (University of Twente), Ömer Şakar `[通讯]` (University of Twente)

**通讯引用:** 24 | [OpenAlex ID](https://openalex.org/A5000292610)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 VerCors 程序验证器中实现了谓词子类型（predicate subtypes）的支持，并自动将子类型声明转化为标准的前置/后置条件与断言；同时引入严格模式（strict subtype）用于检测整数运算中的溢出；

**💡 创新点**

创新点在于完全自动化地生成满足子类型约束的规范，支持多重子类型组合（如并、或、蕴含、非），并首次将子类型用于溢出检查，提供了与 Dafny 等工具相似但更灵活的子类型表达方式；

**🔧 技术方法**

采用了 VerCors 的 AST 转换与 Viper 后端集成技术，在解析阶段将子类型注解映射为表达式，随后在变换阶段生成相应的前置/后置条件与断言，并在 strict‑arithmetic 选项下将所有整数转换为严格子类型以实现溢出检测；

**📊 数据集**

论文未使用公开数据集或具体案例；示例代码基于 Java/C 的伪代码和 VerCors 自带的例子；

**📈 对比分析**

论文未给出系统化的性能比较或基准实验；讨论主要聚焦在实现细节与与相关工具（Dafny、Prusti、Frama‑C）的对比，未提供数值评估；

**⚠️ 局限性**

限制包括：strict‑arithmetic 仅支持 32 位有符号整数，未覆盖不同大小整数类型；子类型实现仅适用于 Java 与 PVL，C 前端尚未支持；错误信息缺乏对谓词子类型的定位；未来工作需完善多语言支持、标准子类型库以及更友好的错误报告。

---

## 368. Branching Out: Existential External Choice in Effpi

**arXiv ID:** 2604.06875 | [PDF](https://arxiv.org/pdf/2604.06875v1)

**作者:** Benjamin Robinson `[一作]` (University of Oxford), Nobuko Yoshida `[通讯]` (University of Oxford)

**通讯引用:** 12073 | [OpenAlex ID](https://openalex.org/A5054171989)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

扩展 Effpi 工具包，新增多通道分支（外部选择）和捕获超时操作，并通过 Scala 3 DSL 实现。

**💡 创新点**

引入可对多通道进行外部分支的 Branching 操作，提供“catch timeout”机制，实现对时间约束的静态检查；同时改进类型系统，使协议可以更精确地表示，防止实现与协议不匹配。

**🔧 技术方法**

使用 Scala 3 的嵌入式 DSL、依赖函数类型（Dependent Function Types）、编译器插件、CCS 变体与 mCRL2 进行模型检测；在实现层面采用隐式实例与类型类来保证分支标签唯一性和载荷兼容性。

**📊 数据集**

未使用外部数据集，主要通过实现 Raft 选举算法和其他示例来展示新特性的表达能力。

**📈 对比分析**

论文未给出量化性能对比；主要通过编译时类型检查和模型检测（未实现）证明了更高的实现安全性和协议准确性，未提供具体的运行时性能评估。

**⚠️ 局限性**

限制：缺乏对新操作的模型检测支持；Timer 处理仍需手工实现；广播/多节点支持有限，仅支持固定节点数；标签重复检测在运行时无法完全保证。

---

## 369. Asynchronous Multiparty Sessions with Mixed Choice

**arXiv ID:** 2604.06872 | [PDF](https://arxiv.org/pdf/2604.06872v1)

**作者:** Franco Barbanera `[一作]` (University of Catania), Mariangiola Dezani-Ciancaglini `[通讯]` (University of Torino)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种支持混合选择的异步多方会话计算机模型，并基于通信标签集合的可一致性设计了新的类型系统，证明了类型化会话的 Subject Reduction、Session Fidelity、锁自由和孤儿消息自由等性质，展示了该模型在异步多方协议中的表达能力。

**💡 创新点**

核心创新在于：① 将混合选择引入异步多方会话，突破了以往同步或单向选择的限制；② 引入通信标签集合的可一致性（coherence）概念，提供了一种不依赖投影的全局类型检查方法；③ 在此框架下给出了关于 Eventual Reception 的类型可达性与安全性分析。

**🔧 技术方法**

采用的主要技术包括：异步会话计算机模型（SMPS）框架、混合选择的语法与语义定义、基于共递归的全局类型与标签转移系统（LTS）、可一致性判定与类型推导规则、以及利用 LTS 证明 Subject Reduction 与 Session Fidelity 的形式化方法。

**📊 数据集**

本文不涉及具体数据集；所示的例子（客户端/服务器、工人、超时处理）均为理论模型演示。

**📈 对比分析**

由于工作为理论性质，未进行实验或性能比较；主要通过形式化证明展示了类型化会话在锁自由、孤儿消息自由以及 Eventual Reception 上的正确性和安全性。

**⚠️ 局限性**

局限性包括：① 需要满足可一致性约束，某些会话可能无法被类型化；② 对于更复杂的混合选择，类型判定的可判定性尚未完全保证；③ 目前未针对实际系统的性能评估，未来需要结合实现与实验验证。

---

## 370. RefineAnything: Multimodal Region-Specific Refinement for Perfect Local Details

**arXiv ID:** 2604.06870 | [PDF](https://arxiv.org/pdf/2604.06870v1)

**作者:** Dewei Zhou `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**通讯引用:** 81574 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对用户指定区域的图像局部细节修复框架 RefineAnything，能在保持背景不变的前提下恢复文本、标识和细结构等细节。

**💡 创新点**

创新点在于 Focus‑and‑Refine 的区域聚焦与粘贴回策略，以及 Boundary Consistency Loss，显著提升细节恢复与边缘无缝融合。

**🔧 技术方法**

使用多模态扩散模型 Qwen‑Image（含冻结的 Qwen2.5‑VL 编码器、VAE latent 条件和 MMDiT 解码器）进行训练与推理。

**📊 数据集**

训练集为自建的 Refine‑30K（20K 参照式 + 10K 无参照式），评估基准为 RefineEval。

**📈 对比分析**

与 GPT‑4o、Gemini、Qwen‑Edit 等基线对比，参照式修复在 MSE、LPIPS、VGG 等指标提升 30‑50%，背景一致性几乎完美；无参照式在 VLM 主观评分上领先。

**⚠️ 局限性**

局限在极小目标区域细节恢复仍受限，且依赖 VAE 码流质量，推理时间受固定分辨率的 VAE 约束。

---

## 371. HingeMem: Boundary Guided Long-Term Memory with Query Adaptive Retrieval for Scalable Dialogues

**arXiv ID:** 2604.06845 | [PDF](https://arxiv.org/pdf/2604.06845v1)

**作者:** Yijie Zhong `[一作]` (Tongji University), Haofen Wang `[通讯]` (Tongji University)

**通讯引用:** 2895 | [OpenAlex ID](https://openalex.org/A5039059501)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了HingeMem，一种基于事件分割理论的边界引导长时记忆系统，并设计了查询自适应检索策略。

**💡 创新点**

创新点在于：①将对话事件的四个关键元素（人、时间、地点、主题）变化作为边界触发点，写入结构化超边；②构建可解释的元素索引和超边，兼具语义与图结构；③针对检索类型（回忆优先、精确优先、判断）动态规划检索深度和停止条件，提升检索相关性与效率。

**🔧 技术方法**

主要技术包括：事件分割启发式边界提取、基于LLM的超边构建、节点与超边的重叠合并、基于节点显著性与主题稀缺性的超边重排序、以及分层的自适应停止策略。

**📊 数据集**

使用了LOCOMO数据集进行评估，覆盖单跳、多跳、时间、开放域和对抗性五类问答。

**📈 对比分析**

与RAG、Mem0、HippoRAG2、LangMem等多种基线进行对比，HingeMem在整体F1上提升约5%，多跳问答提升超过10%，同时在token消耗上相较于HippoRAG2降低约68%，展示出更高的精确度与计算效率。

**⚠️ 局限性**

局限性包括：在开放域和对抗性问题上仍略逊于部分基线；对超边合并阈值和重排序权重的超参数敏感；以及对非常短或极端噪声对话的边界识别仍有挑战。

---

## 372. FedDetox: Robust Federated SLM Alignment via On-Device Data Sanitization

**arXiv ID:** 2604.06833 | [PDF](https://arxiv.org/pdf/2604.06833v1)

**作者:** Shunan Zhu `[一作]` (University of Tokyo), Hideya Ochiai `[通讯]` (University of Tokyo)

**通讯引用:** 1401 | [OpenAlex ID](https://openalex.org/A5045456657)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 FedDetox 框架，在联邦学习中通过客户端轻量级安全守护者对不洁数据进行本地消毒，并利用拒绝模板替换机制将潜在毒性示例转化为正向安全信号，从而在保持模型通用性能的前提下恢复安全对齐。

**💡 创新点**

创新点包括：①将大规模安全模型的知识蒸馏到极轻量化的 MobileBERT 守护者，实现边缘设备上的实时安全检测；②拒绝模板替换策略将被过滤的毒性样本转换为有监督的拒绝对，以避免知识空洞；③在联邦设置下首次验证了针对“意外毒性注入”的防护效果。

**🔧 技术方法**

主要技术：联邦直接偏好优化（FedDPO）+ LoRA 参数高效微调；知识蒸馏（Teacher‑Student）训练 MobileBERT 守护者；在客户端实现前向检测与拒绝模板生成；对齐损失基于 DPO。

**📊 数据集**

使用的数据集包括：Qwen2.5-1.5B-Instruct/Base 作为基础模型；Benign 指令数据（采自公开安全指令集合）；毒性数据采自 AdvBench、DAN、TAP 等攻击集；用于蒸馏的混合安全标签数据，比例 60% benign / 40% malicious。

**📈 对比分析**

与中心化对齐、无防护 FedDPO 以及仅丢弃毒性样本等基线对比，FedDetox 在 AdvBench、DAN、TAP 的攻击成功率分别降低至 14%、74.8% 和 61%，与理想安全模型（仅安全样本）相近；同时在 XSTest、安全合规性和通用评测（TruthfulQA、MMLU、GSM8K）上保持与基线相近或略优的性能，显示安全与通用性的平衡。

**⚠️ 局限性**

局限性：①依赖教师模型的安全检测准确性，若教师误判会误导守护者；②拒绝模板的固定性可能对复杂语境产生局限；③实验主要在单机 GPU 服务器上模拟联邦环境，真实边缘设备的网络延迟与算力约束仍需进一步验证。

---

## 373. Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM

**arXiv ID:** 2604.06832 | [PDF](https://arxiv.org/pdf/2604.06832v1)

**作者:** Chengyue Wu `[一作]` (University of Hong Kong), Enze Xie `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

构造了一个基于块级扩散的视觉语言模型，实现了与KV缓存兼容的并行解码与自我推测加速，并在单请求场景下显著提升推理速度。

**💡 创新点**

核心创新在于：① 直接将预训练的自回归VLM（Qwen2.5‑VL‑3B）转换为块级扩散模型，避免两阶段训练的效率低下；② 设计了块大小逐步递增、自动截断、多视角高效拼接及因果上下文注意力的训练策略；③ 在推理中加入自推测块解码，并与SGLang系统深度集成，配合FP8量化实现端到端6倍加速。

**🔧 技术方法**

使用 Fast‑dLLM v2 的块级离散扩散框架；块大小逐步递增、自动截断掩码、因果上下文注意力、视觉高效拼接；自推测块解码（线性与二次变体）；SGLang 调度器、CUDA graph；SmoothQuant FP8 量化。

**📊 数据集**

在 11 个多模态基准（MMMU‑Pro‑V、MMBench、POPE、GQA、DocVQA、ChartQA、AI2D、RealWorldQA、SeedBench2+、TextVQA 等）上进行评测；使用 ShareGPT‑4V、VLMEvalKit 作为数据集和评价工具。

**📈 对比分析**

对比 AR 与扩散 VLM 以及两种 AR‑>扩散转换策略，结果表明直接转换模型在大部分短答任务上与 AR 基线相当或更优；在推理效率上，MDM 解码实现 1.95× Tokens/NFE、2.63× Tokens/NFE（自推测），端到端推理速度提升 6.18×，最高 TPS 达 350。对于长答任务，虽然仍略低于 AR，但差距可通过更大块或更多数据进一步缩小。

**⚠️ 局限性**

局限性：在长链推理（如 MMMU‑Pro‑V）上仍存在 1–3 分的准确率差距；块级并行解码对极短响应的处理仍需优化；自推测解码在大块尺寸下对硬件调度和内核优化存在瓶颈；模型对多轮对话的上下文切换仍有信息泄漏风险，需进一步完善截断机制。

---

## 374. Towards Privacy-Preserving Large Language Model: Text-free Inference Through Alignment and Adaptation

**arXiv ID:** 2604.06831 | [PDF](https://arxiv.org/pdf/2604.06831v1)

**作者:** Jeongho Yoon `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**通讯引用:** 49812 | [OpenAlex ID](https://openalex.org/A5027447596)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种隐私保护的细调框架PPFT，实现在LLM服务中客户端仅传递嵌入，服务器永不接收原始提示文本；

**💡 创新点**

创新点在于：1）两阶段训练实现文本无界面；2）通过k-Pooling压缩token级信息并注入Laplace噪声，显著降低逆向重构风险；3）无需暴露模型解码器参数即可完成域自适应；

**🔧 技术方法**

使用了客户端BERT编码器、k-Pooling聚合、投影层对齐、Laplace噪声注入、LoRA微调、生成式逆向攻击评估等技术；

**📊 数据集**

实验使用医学和法律QA数据集（Pri-SLJA、Pri-DDX等）以及通用基准（CSQA、SQuAD）进行评估；

**📈 对比分析**

与d_χ-privacy、Paraphrase、PrivacyRestore等基线比较，PPFT在8B LLM上在所有敏感域任务中保持95%+准确率，噪声注入下重构ROUGE-L低于0.25；通用域性能损失≤0.2；

**⚠️ 局限性**

局限性包括：仅保障输入隐私，输出内容仍可能泄露；仅验证了BERT-LLaMA文本组合，未覆盖多语言、多模态或更小模型；需配合输出侧安全措施以实现端到端隐私。

---

## 375. Non-RS type cyclic MDS codes over finite fields via cyclotomic field reduction

**arXiv ID:** 2604.06822 | [PDF](https://arxiv.org/pdf/2604.06822v1)

**作者:** Can Xiang `[一作]` (South China Agricultural University), Chunming Tang `[通讯]` (Southwest Jiaotong University)

**通讯引用:** 2073 | [OpenAlex ID](https://openalex.org/A5054829854)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一类新的循环MDS码，利用循环域归约方法将MDS判定转化为零特征下非零极小子式的判定。

**💡 创新点**

创新点在于使用环上规范化与余域同态，将在复域中验证的MDS性质直接迁移至有限域，极大地简化了构造过程，并实现了参数灵活、覆盖已知所有非RS循环MDS码的效果。

**🔧 技术方法**

核心技术包括循环域的整数环、质理想分解、归约同态、范数映射、广义范德蒙行列式理论以及Hensel引理。

**📊 数据集**

论文未使用传统意义上的数据集，而是通过理论推导与Magma软件进行符号与数值验证来检验构造的有效性。

**📈 对比分析**

与已有的多种构造方法相比，本方法实现了更简洁的算法流程、参数可调性更强，并在实验中成功生成了大量此前未知的非RS循环MDS码；相较之下，传统方法往往受参数限制且计算复杂度更高。

**⚠️ 局限性**

主要限制在于需避免范数为零的质数集合P_bad，构造过程对大规模n的符号计算仍具有一定的理论与计算成本；此外，证明中仍依赖于质理想分解与范数性质，若求解时出现稠密子式，实际实现可能受限。

---

## 376. OmniTabBench: Mapping the Empirical Frontiers of GBDTs, Neural Networks, and Foundation Models for Tabular Data at Scale

**arXiv ID:** 2604.06814 | [PDF](https://arxiv.org/pdf/2604.06814v1)

**作者:** Dihong Jiang `[一作]` (Huawei Cloud), Qi Tian `[通讯]` (Huawei Cloud)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为 BINGO 的最大规模表格数据集基准（3030个数据集），并在此基准上对传统GBDT、深度神经网络和基于预训练的Tabular Foundation Models（如TabPFN）进行大规模实验；

**💡 创新点**

①将OpenML、UCI、Kaggle三大公开资源合并并通过LLM筛选与去重；②使用精细化的元特征（如数据规模、特征类型、分布偏度/峰度等）对模型优势进行解耦分析；③提供行业级标签，极大提升基准的多样性与实用性；

**🔧 技术方法**

基于GBDT（LightGBM、XGBoost、CatBoost）、多层感知机（MLP）、RealMLP、ResNet、FT-Transformer以及TabPFN等模型；使用LLM（doubao）进行元信息抽取与分类；采用多种预处理（缺失值填充、数值归一化、低基数数值转为类别等）；使用Welch t‑test等统计方法进行特征-性能关联分析；

**📊 数据集**

共收集8558个原始数据集，最终筛选后得到3030个符合规模、任务类型、标签明确且难度适中的数据集，涵盖分类与回归任务，特征与目标分布多样；

**📈 对比分析**

在所有数据集上使用默认配置无超参调优，对八种模型进行训练与评估；结果显示不存在单一模型始终占优，GBDT、NN和TabPFN各自占据约31–35%的顶位；在不同元特征维度下可观察到模型优势分布，给出可操作的模型选择指南；

**⚠️ 局限性**

限制在于：①仅测试了有限的代表性模型；②缺乏对超参调优和算力/时间成本的深入比较；③基准主要基于公开数据，可能缺少极端规模或特殊领域的专有数据；

---

## 377. AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation

**arXiv ID:** 2604.06812 | [PDF](https://arxiv.org/pdf/2604.06812v1)

**作者:** Guanran Luo `[一作]` (Xiamen University), Qingqiang Wu `[通讯]` (Xiamen University)

**通讯引用:** 568 | [OpenAlex ID](https://openalex.org/A5048017759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的长文本不确定性量化框架AGSC，解决LLM生成文本中的幻觉问题。

**💡 创新点**

创新点包括：①使用NLI中性类别作为自适应粒度触发器，区分无关与不确定句子，从而减少不必要的细粒度拆分；②利用GMM软聚类捕获潜在语义主题，给每个句子分配主题权重，降低噪声片段对聚合的影响。

**🔧 技术方法**

核心技术：NLI模型（如DeBERTa-v3-large-mnli）进行句子/原子事实的蕴含/矛盾评估；GMM+UMAP实现语义软聚类；基于主题权重的加权聚合计算最终不确定性分数。

**📊 数据集**

实验使用BIO（传记生成）和LongFact（多主题长文本）两个公开数据集，评估生成模型为GPT-4.1-mini、Qwen2.5-32B、Llama3-70B。

**📈 对比分析**

与多种基线（Token SE、LexSim、图方法、KLE、SPUQ、SCN、LUQ系列）比较，AGSC在BIO和LongFact上均实现了最高的Pearson/Spearman相关系数，并在完整原子拆分基础上将推理时间压缩约60%。

**⚠️ 局限性**

局限性包括：①对NLI模型的中性类别判定高度依赖，若NLI校准失误会误判无关或不确定句子；②对“回声室”幻觉易失真，即多样本一致但内容错误时仍给出低不确定性；③在非句子为真理单元（如数学推理、代码）场景下适用性不足；④低资源语言或嵌入质量不佳时聚类效果差。

---

## 378. VulGD: A LLM-Powered Dynamic Open-Access Vulnerability Graph Database

**arXiv ID:** 2604.06967 | [PDF](https://arxiv.org/pdf/2604.06967v1)

**作者:** Luat Do `[一作]` (La Trobe University), Hua Wang `[通讯]` (Victoria University)

**通讯引用:** 31077 | [OpenAlex ID](https://openalex.org/A5100403969)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了 VulGD，一个动态、开源的漏洞知识图谱数据库，持续从 NVD、CVE、CWE、CVE Details、ExploitDB 等公开源聚合漏洞信息，并通过 Neo4j 存储，提供可视化 Web 界面和 RESTful API，支持 LLM 生成的漏洞描述嵌入。

**💡 创新点**

创新点在于：①动态实时多源集成，自动化 ETL 及 cron 调度实现近实时更新；②将 LLM 嵌入与图节点结合，提供多维度（原始、128D、32D）嵌入检索；③完整的 Web UI 与公共 API，使非专家也能快速浏览、查询与导出数据；④统一使用 Neo4j 的属性图模型，支持复杂多跳查询与可视化。

**🔧 技术方法**

技术栈包括：Neo4j 4.4.x 关系图数据库；Python 3.10+ 与 HuggingFace Transformers（all-mpnet-base-v2、SecBERT、FastText）进行嵌入生成；PCA（及增量 PCA）用于降维；FastAPI+Nginx 提供 API；React+D3 实现前端可视化；crontab/Redis 等任务调度；使用 Neo4j Bolt 协议和 Neo4j JavaScript 驱动实现查询。

**📊 数据集**

使用的主要数据集为公开漏洞数据库：NVD JSON Feed、CVE、CWE、CVE Details、ExploitDB、EDB 等；通过 API 与爬虫方式持续抓取并统一映射为图节点/关系，覆盖 1999‑2026 年 CVE 及相关实体。

**📈 对比分析**

与 SEPSES、VulKG 等现有图谱做对比：VulGD 在动态更新、LLM 嵌入、Web UI 与 API 四维度上处于领先；在规模上截至 2026‑04，图包含 324,618 个漏洞节点、46,605 个 exploit 节点、约 700k 条边；API 调用平均延迟低于 200 ms（未给精确数值，但示例显示可实时响应）；嵌入存储和检索采用压缩后存储，内存占用在 32D/128D 维度下分别约 34 MB / 133 MB，PCA 计算时间均在 25‑125 ms 之间。

**⚠️ 局限性**

局限性包括：①仅整合结构化公开源，缺乏对非结构化情报、实时威胁情报流的支持；②LLM 嵌入虽然提升语义理解，但可能产生幻觉或误判，未直接验证对新颖漏洞的准确性；③大规模重构时重新计算嵌入成本高，当前采用离线批量方式；④缺乏预警、告警与自动化补丁管理功能；⑤对多语言或多域漏洞描述的跨语言处理尚未实现。

---

## 379. What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning

**arXiv ID:** 2604.06995 | [PDF](https://arxiv.org/pdf/2604.06995v1)

**作者:** Songze Li `[一作]` (Zhejiang University), Wen Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 153780 | [OpenAlex ID](https://openalex.org/A5100399276)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 UI-in-the-Loop（UILoop）框架，将 GUI 推理从传统的 Screen‑to‑Action 转变为 Screen‑UI Elements‑Action 循环，并创建了 UI Comprehension 任务与 26K 规模的 benchmark 进行评估；通过 UI‑Element‑Driven Reinforcement Fine‑Tuning 训练模型提升对 UI 元素的定位、语义理解与利用能力。

**💡 创新点**

创新点在于：① 明确将 UI 元素作为推理的关键桥梁，形成可解释的 reasoning 链；② 设计了三维度评估指标（Locate、Lingualize、Leverage）和相应的奖励函数；③ 通过强化学习结合多模态 LLM 进行细粒度 UI 元素学习；④ 构建首个大规模 UI Comprehension benchmark。

**🔧 技术方法**

采用了多模态大语言模型（如 Qwen2.5‑VL‑3B/7B、GPT‑4o）、强化学习算法 GRPO、OmniParser 等 UI 元素标注工具、基于位置/语义/利用三种奖励的 UI‑Element‑Driven RL fine‑tuning 方法。

**📊 数据集**

使用了 Android Control、OmniAct、GUI‑Act、ScreenSpot、ScreenSpot‑Pro、OS‑Atlas 等现有 GUI 数据集，并在此基础上合成了 26K 条带有 UI 元素 ground truth 的 UI Comprehension‑Bench 数据。

**📈 对比分析**

与零-shot MLLMs 及多种 Screen‑to‑Action 方案在 ScreenSpot‑Pro、Android Control‑High 等数据集上进行对比，评估指标包括 Action Type、Ground Rate、Step Success Rate 等；在 UI Comprehension‑Bench 上获得 26.1 的综合分数，且在 SR、GR、Type 等指标上分别比基线提升 13%–58%，证明 UILoop 的显著性能优势。

**⚠️ 局限性**

主要限制：① 仅关注细粒度 UI 元素的掌握，未考虑不同层次（粗细粒度）UI 布局对推理的影响；② 实验范围局限于 Qwen2.5‑VL，未验证在更广泛的多模态 LLM 上的泛化能力。

---

## 380. Stress Estimation in Elderly Oncology Patients Using Visual Wearable Representations and Multi-Instance Learning

**arXiv ID:** 2604.06990 | [PDF](https://arxiv.org/pdf/2604.06990v1)

**作者:** Ioannis Kyprakis `[一作]` (FORTH-ICS), Manolis Tsiknakis `[通讯]` (FORTH-ICS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文利用智能手表与胸部心电传感器产生的多模态可穿戴数据，通过将时序信号转化为视觉图像，构建弱监督的多实例学习框架来估计老年心肿瘤患者的主观压力水平。

**💡 创新点**

创新点在于：1）将可穿戴时序数据转换为视觉表示，充分利用视觉模型的时序建模能力；2）采用轻量级混合专家网络Tiny‑BioMoE进行特征提取；3）结合注意力机制的多实例学习，实现对多模态、缺失及不同长度样本的聚合与解释。

**🔧 技术方法**

技术包括：视觉编码（热图、假睡眠图、心电重现图谱等）、轻量级Mixture‑of‑Experts网络Tiny‑BioMoE、基于注意力的多实例学习（MIL）聚合、留一患者交叉验证（LOSO）和回归评价指标。

**📊 数据集**

数据集为CARDIOCARE多中心老年乳腺癌患者数据，共387人，采集了Garmin手表的活动与睡眠信息以及Polar H10心电数据，并在M3与M6时点收集Perceived Stress Scale（PSS）问卷得分。

**📈 对比分析**

与单模态或两模态模型对比，三模态融合在LOSO评估中获得最佳性能：M3全局RMSE为6.62、MAE 6.07、R² 0.24；M6全局RMSE 6.13、MAE 5.54、R² 0.28，显示出相对较好的预测相关性。

**⚠️ 局限性**

局限性包括：1）PSS问卷频率稀疏，导致标签与实例间时间对齐不精确；2）老年患者的多中心差异和缺失率高；3）模型仍未完全解决个体差异与缺失模式的影响，预测误差较大，难以直接用于临床决策。

---

## 381. Canopy Tree Height Estimation Using Quantile Regression: Modeling and Evaluating Uncertainty in Remote Sensing

**arXiv ID:** 2604.06988 | [PDF](https://arxiv.org/pdf/2604.06988v1)

**作者:** Karsten Schrödter `[一作]` (University of Münster), Fabian Gieseke `[通讯]` (University of Münster)

**通讯引用:** 3273 | [OpenAlex ID](https://openalex.org/A5023532714)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在已有的遥感树冠高度预测模型上加入一个量化不确定性的头部，并通过量化回归（quantile regression）对该头部进行微调，从而在保持原始点估计性能的前提下提供可校准的置信区间。

**💡 创新点**

创新点在于：① 用极小改动即可在预训练模型上加入不确定性估计；② 采用量化回归配合位移鲁棒损失，既能应对地理位置误差，又能在单前向传播中得到多分位点预测；③ 对不确定性进行了系统评估（MPIW、PICP、覆盖率与误差相关性），并与仅做点估计的基线模型进行对比。

**🔧 技术方法**

主要技术包括：量化回归（使用pinball loss）、Shift-Resilient Loss（对GEDI轨迹做±1像素位移优化）、Unet架构的双头（点估计头+不确定性头）、微调策略（冻结主体，只训练两个头），以及使用MPIW/PICP等统计指标评估预测区间。

**📊 数据集**

使用Sentinel‑2时间序列（12个月）与Sentinel‑1雷达年复合图像，光谱通道共16个，地面真值来自GEDI LiDAR的标记（约49k个样本）。

**📈 对比分析**

与已有的基线模型（仅做点估计且基于Huber loss）比较，点估计的MSE、MAE与R²基本保持不变；不确定性评估显示：我们的模型在所有置信水平下MPIW约为基线的一半，PICP提升0.10-0.3，覆盖率更好，尤其在高不确定性区间（90%）下覆盖率从0.67提升至0.88。

**⚠️ 局限性**

局限性包括：① 高树冠（>30 m）分位点的覆盖率不足，说明在高高度区域模型仍欠校准；② 在森林边界和陡坡地区不确定性显著升高，可能受GEDI定位误差和标签噪声影响；③ 对异常高置信度错误的处理尚未系统化，无法有效去除潜在的错误标签；④ 由于仅在头部微调，整体模型对极端条件（极端地形、光照）适应性有限。

---

## 382. Towards Multi-Object Nonprehensile Transportation via Shared Teleoperation: A Framework Based on Virtual Object Model Predictive Control

**arXiv ID:** 2604.06932 | [PDF](https://arxiv.org/pdf/2604.06932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 383. CAAP: Capture-Aware Adversarial Patch Attacks on Palmprint Recognition Models

**arXiv ID:** 2604.06987 | [PDF](https://arxiv.org/pdf/2604.06987v1)

**作者:** Renyang Liu `[一作]` (National University of Singapore), See-kiong Ng `[通讯]` (National University of Singapore)

**通讯引用:** 5433 | [OpenAlex ID](https://openalex.org/A5090171111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向掌纹识别的可捕获（capture‑aware）物理对抗补丁框架，学习一种通用交叉形状补丁，并通过输入自适应渲染、光度合成与多尺度特征引导实现对模型的攻击。

**💡 创新点**

创新点在于：①将补丁设计为固定交叉形状以最大化纹理覆盖；②引入 Adaptive Spatial Transformer (ASIT) 进行输入条件的几何与光度自适应渲染；③使用 Radiometric Synthesis (RaS) 近似打印-捕获过程的随机变形；④添加 Multi‑Scale Dual‑Invariant Feature Extractor (MS‑DIFE) 在特征空间提供身份破坏引导。

**🔧 技术方法**

核心技术包括：EOT（Expectation‑over‑Transformations）物理渲染框架、交叉形状补丁的二值掩码、ASIT 的低维仿射变换与光度校正、RaS 的光度噪声模拟、MS‑DIFE 的多尺度特征聚合及余弦距离约束，以及可变形的总变差（TV）与视觉一致性正则化。

**📊 数据集**

在公开的 Tongji、IITD 三个掌纹数据集以及内部 AISEC 数据集（共 26 受试者）上进行实验，所有样本统一为 128×128 对齐 ROI 图像。

**📈 对比分析**

与多种基线补丁攻击（AdvPatch、Patch_MI、Patch_PGD、APPA、CSPA、AdvLogo、正方形补丁）以及模型特定基线（MobileNetV2、VGG16、ResNet‑18、ShuffleNetV2、CompNet、CCNet、CO3Net）进行对比。结果显示，交叉形状补丁（_c）在无目标、目标两种攻击下在三大数据集上均实现 90% 以上的攻击成功率，且在跨模型、跨数据集、以及经过对抗训练的模型上均保持强劲的鲁棒性，远优于传统基线。

**⚠️ 局限性**

局限性包括：对抗训练可部分抑制攻击但仍有残余；目前仅考虑单一补丁且中心位置放置，未探索多补丁或自适应定位策略；实验仅覆盖固定光照与手掌姿态，缺乏在更极端环境下的验证；缺乏针对多模态或活体检测系统的评估。

---

## 384. PSR2: A Phase-based Semantic Reasoning Framework for Atomicity Violation Detection via Contract Refinement

**arXiv ID:** 2604.06975 | [PDF](https://arxiv.org/pdf/2604.06975v1)

**作者:** Xiaoqi Li `[一作]` (Hainan University), Zongwei Li `[通讯]` (Hainan University)

**通讯引用:** 52 | [OpenAlex ID](https://openalex.org/A5101530716)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了PSR2框架，用图结构搜索与语义推理相结合来检测智能合约中的原子性违规问题。

**💡 创新点**

创新点在于构建统一的原子性不一致模型，并通过GSAM、SCAM、FDM三模块实现结构路径与语义依赖的交叉验证，从而克服传统工具的上下文盲区与拓扑敏感性不足。

**🔧 技术方法**

采用控制流图（CFG）简化与路径检索、抽象语法树（AST）语义提取、确定性语义推理以及融合决策逻辑。

**📊 数据集**

使用1,600份合约样本，涵盖Smartbugs‑Reentrancy、ERC‑721 Reentrancy和Unchecked External Call三类基准。

**📈 对比分析**

与Slither和Semgrep对比，PSR2在ERC‑721数据集上F1‑score达94.69%，在其他数据集亦显著优于基线，误报率降半，召回率接近98%。

**⚠️ 局限性**

局限性包括对跨合约调用缺乏完整依赖集，仍需源代码支持，且在无源代码的字节码层的语义提取能力尚待提升。

---

## 385. Differentiable Environment-Trajectory Co-Optimization for Safe Multi-Agent Navigation

**arXiv ID:** 2604.06972 | [PDF](https://arxiv.org/pdf/2604.06972v1)

**作者:** Zhan Gao `[一作]` (University of Cambridge), Amanda Prorok `[通讯]` (University of Cambridge)

**通讯引用:** 2656 | [OpenAlex ID](https://openalex.org/A5066624177)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种可微分双层优化框架，用于同时调整环境配置和多智能体轨迹，以提升多智能体导航的安全性和效率。

**💡 创新点**

创新点包括：①将环境参数视为决策变量与轨迹共同优化；②利用 KKT 条件与隐函数定理显式求解轨迹对环境参数的梯度；③设计了连续的安全度量，用于量化环境对多智能体导航的安全影响；④将上述方法结合成可微分的双层求解流程。

**🔧 技术方法**

主要技术手段包括：内部点法求解下层轨迹优化；KKT 与隐函数定理求梯度；梯度上升（或其他梯度优化）求解上层环境优化；自定义安全度量与其梯度计算；整体可微分求解框架。

**📊 数据集**

实验数据集：六个仿真场景——仓库货架、环形交叉路、狭窄通道、高速出口、路口交叉、赛道设计；每个场景均包含不同数量的智能体和障碍物，并在多次随机起始/目标配置下评估。

**📈 对比分析**

与基准环境（规则布局、随机布局、无障碍等）对比，所提方法在安全度量（↑）、成功率/路径长度比（↑）、碰撞次数（↓）、计算时间（↓）以及控制能耗（↓）等指标均表现更优，特别是在复杂或大规模场景中提升更为显著。

**⚠️ 局限性**

局限性包括：①双层优化非凸，易陷入局部最优；②对障碍形状和距离函数的可微分性有要求；③计算成本仍相对较高，尤其是多智能体数量大或环境维度高时；④仅在静态、已知环境下验证，缺乏真实场景或动态障碍物的实验。

---

## 386. Leveraging LLMs and Heterogeneous Knowledge Graphs for Persona-Driven Session-Based Recommendation

**arXiv ID:** 2604.06928 | [PDF](https://arxiv.org/pdf/2604.06928v1)

**作者:** Muskan Gupta `[一作]` (TCS Research), Jyotsana Khatri `[通讯]` (TCS Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于用户人格的会话推荐框架，利用异构知识图谱和大语言模型融合长短期偏好，提升匿名会话下的个性化推荐效果。

**💡 创新点**

创新点在于：①通过异构深度图信息最大化（HDGI）无监督学习知识图谱中的用户人格表示；②将该人格表示与LLM生成的物品语义嵌入共同输入检索模型，再用原始序列模型进行重排；③实现了结构化知识与语义信息的高效协同。

**🔧 技术方法**

核心技术包括：异构知识图谱构建、LLM（Qwen-3-8B/Embedding）文本编码、HDGI对知识图谱进行无监督预训练、检索式候选生成与SASRec重排。

**📊 数据集**

使用的数据集为 Amazon Books 和 Amazon Movies & TV 两个公开电商数据集，并基于 Amazon-KG 进行知识图谱构建。

**📈 对比分析**

实验通过四种用户表示（无、随机、LLM、KG+LLM）在检索和重排阶段进行对比，结果显示 KG+LLM 的 HR、NDCG、MRR 在候选集和最终排序上均显著优于仅使用序列模型或纯 LLM 表示，尤其在 HR@100 方面提升最为显著。

**⚠️ 局限性**

局限性包括：①依赖大型 LLM 生成文本特征，推理成本高；②知识图谱构建需外部语义资源，可能导致域适配困难；③对极度稀疏或冷启动场景的效果提升有限，仅提升候选召回而非排名精度。

---

## 387. On the Decidability of Distributed Tasks with Output Sets under Asynchrony and Any Number of Crashes

**arXiv ID:** 2604.06920 | [PDF](https://arxiv.org/pdf/2604.06920v1)

**作者:** Timothé Albouy `[一作]` (IMDEA Software Institute), Junlang Wang `[通讯]` (IMDEA Software Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

定义并研究SOS任务（Set of Output Sets任务），并证明该类任务在异步崩溃模型下的可决性。

**💡 创新点**

提出了一个简单的判定规则：若 t=0 则总可解；若 t>0 则可解当且仅当其SOS图（顶点为输出集合，边表示包含关系）连通；进一步揭示无有效性约束的k-集合同意在 k≥2 时可在任意崩溃下实现，而 k=1（共识）则在任何崩溃下不可实现；将 d-不一致问题的实现条件与调和级数关联。

**🔧 技术方法**

采用组合拓扑（SOS图的连通性）、图论与异步容错算法（通用t-弹性算法和专门的d-不一致实现）进行理论分析与证明。

**📊 数据集**

无实验数据，全部工作为理论证明。

**📈 对比分析**

无实验对比或性能评估；论文通过数学证明展示可解性与不可解性边界。

**⚠️ 局限性**

仅适用于无有效性约束任务；对连通SOS任务的最优容错阈值未完全确定；d-不一致实现的容错阈值上下界仍相距较大。

---

## 388. FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling

**arXiv ID:** 2604.06916 | [PDF](https://arxiv.org/pdf/2604.06916v1)

**作者:** Yitong Li `[一作]` (NVIDIA), Enze Xie `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种两阶段的Sol-RL框架，利用FP4量化进行大规模探索，再用BF16精度重生成高对比样本进行策略更新，从而在文本到图像的扩散模型上实现高效且高保真度的对齐。

**💡 创新点**

创新点在于将FP4量化探索与高精度优化解耦：1) 在探索阶段使用FP4快速生成数百个候选图像并仅保留排名最高/最低的K个；2) 在训练阶段仅对这些高对比样本进行BF16重生成与梯度更新，避免量化误差导致的稳定性下降，兼顾速度与质量。

**🔧 技术方法**

使用技术包括：扩散模型的DiffusionRL与Group Relative Policy Optimization（GRPO），FP4（NVFP4）量化、ODE采样，LoRA微调，以及多种奖励模型（ImageReward、CLIPScore、PickScore、HPSv2）。

**📊 数据集**

实验数据集为PickScore提示集合，奖励模型覆盖ImageReward、CLIPScore、PickScore、HPSv2，评估基于SD3.5、FLUX.1和SANA等三种前沿扩散模型。

**📈 对比分析**

与FlowGRPO、DanceGRPO、AWM、DiffusionNFT等基线对比，Sol-RL在相同GPU时钟预算下实现了1.91~4.64倍的收敛速度提升，并在所有奖励指标上均优于基线，最终对齐质量更高。

**⚠️ 局限性**

局限性：依赖支持FP4的专用硬件（如NVIDIA Blackwell），在非FP4环境下难以复制；对FP4探索步数与池大小的超参数敏感，且主要验证在文本到图像扩散模型上，是否能推广至其他任务尚待进一步研究。

---

## 389. Self-Preference Bias in Rubric-Based Evaluation of Large Language Models

**arXiv ID:** 2604.06996 | [PDF](https://arxiv.org/pdf/2604.06996v1)

**作者:** José Pombal `[一作]` (Sword Health), André F. T. Martins `[通讯]` (Instituto de Telecomunicações)

**通讯引用:** 5651 | [OpenAlex ID](https://openalex.org/A5051693368)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM-as-a-judge在基于规则的评估（RB）中的自我偏好偏差（SPB）进行系统研究，量化其程度并探讨其对评估结果的影响。

**💡 创新点**

首次揭示即使规则完全客观（可程序验证）时，SPB仍存在；提出HSPP‑Rubric/Instance指标衡量SPB；分析规则属性（长度、正负、主题）对SPB的影响；验证集成判断（committee）能缓解但不能完全消除SPB。

**🔧 技术方法**

使用LLM-as-a-judge框架，实施单规则与全规则评估、配对比较（PWC）与直接评估（DA）；采用HSPP（Harmful Self‑Preference Propensity）计算方法；构建并评估多模型投票的委员会；利用交叉验证的互评一致性阈值过滤规则。

**📊 数据集**

评估数据集为：IFEval（541个程序可验证指令，目标为客观评估）和HealthBench（5000例、48562个主观二元规则，用于医疗聊天评估）。

**📈 对比分析**

通过平均实例配对准确率（MIPA）与平均规则准确率（MRA）评估判断准确性；通过HSPP‑Rubric/Instance比值评估SPB；结果显示：在RB中SPB仍达1.1–1.3倍；PWC的SPB更严重；委员会评估可将SPB降至约1.1–1.2倍，准确率提升0.01–0.02；在HealthBench中，SPB可导致约10分的评分偏差，足以影响模型排名。

**⚠️ 局限性**

局限性：1）SPB未被完全消除，仍需更深入的训练或推理时去偏方法；2）评估基于多数投票作为参考，可能低估真实偏差；3）实验仅覆盖两个数据集，泛化性待验证；4）未考虑模型自身的激活或参数微调对SPB的影响。

---

## 390. Anytime Analysis on BinVal: Adaptive Parameters Help

**arXiv ID:** 2604.06976 | [PDF](https://arxiv.org/pdf/2604.06976v1)

**作者:** Timo Kötzing `[一作]` (Hasso Plattner Institute University of Potsdam), Jurek Sander `[通讯]` (Hasso Plattner Institute University of Potsdam)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析了在二进制价值函数（Binary‑Value）上，随机搜索启发式（如 (1+1) EA 与 sig‑cGA）在固定目标（即优化前 k 位）下的任意时性能，并给出了同时适用于所有 k∈o(n) 的运行时间上下界。

**💡 创新点**

提出了在不需要知道 k 的前提下，通过调节或自适应突变率实现对所有 k 同时达到 Θ(k log k) 运行时间的理论证明，并将传统固定突变率结果提升到更优的 Θ(n log k) 或 k^{1+ε} 上界。

**🔧 技术方法**

主要采用漂移理论（乘法漂移与加法漂移）、潜在函数设计以及对 sig‑cGA 的显著性阈值分析等技术进行证明。

**📊 数据集**

实验使用标准二进制价值函数（bitstring 长度 n = 2048），对 100 次独立运行进行平均。

**📈 对比分析**

通过比较标准 (1+1) EA、调整突变率 (ad‑MR) 与自适应突变率 (self‑MR) 的迭代次数，实验显示 ad‑MR 与 self‑MR 在 k∈o(n) 时显著优于标准，而标准在 k>n/2 时更快。

**⚠️ 局限性**

局限性：理论分析仅针对特定的二进制价值函数；自适应突变率的理论上界为 k^{1+ε}，尚未完全达到最优的 Θ(k log k)；实验规模有限，未对更大 n 或其他线性目标函数进行验证。

---

## 391. Block-Bench: A Framework for Controllable and Transparent Discrete Optimization Benchmarking

**arXiv ID:** 2604.06973 | [PDF](https://arxiv.org/pdf/2604.06973v1)

**作者:** Furong Ye `[一作]` (Leiden University), Niki van Stein `[通讯]` (Leiden University)

**通讯引用:** 1068 | [OpenAlex ID](https://openalex.org/A5003248571)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于块函数的离散优化基准构造框架(Block‑Bench)，实现了对问题属性（目标结构、局部最优模式、变量表示）的细粒度控制；

**💡 创新点**

创新点在于：①通过块函数与加权、依赖图的组合可生成可解释且可定制的多模态、分层或门控结构基准；②在基准中保持块级信息，使得能够在变量层面直接观察和分析算法行为；

**🔧 技术方法**

使用的技术包括块函数设计（OneMax、Jump、LeadingOnes、Epistasis等）、无向图（DBP）和有向无环图（GCP）建模依赖关系、权重向量与常数项调节目标贡献、以及对自适应突变率、跨代门控与多目标权重的参数化；

**📊 数据集**

使用的数据集为论文中构造的若干基准实例（F1–F10、BF1–BF5等），以及公开的Block‑Bench仓库中生成的多维度、块数可变、图结构可调的离散优化问题；

**📈 对比分析**

比较方法包括对不同自适应（1+λ)EA、fGA、(1+λ,λ)GA、两率EA等的函数评估次数与收敛曲线；对(μ+1)GA与带多样性维护的版本在DBP/GCP上的求解效率；以及对多目标算法（SEMO、GSEMO等）在BF2–BF5上的超体积（HV）收敛；实验表明某些自适应策略在单一经典问题上表现优异，但在包含多种景观的基准上表现不一，表明景观多样性显著影响算法性能；

**⚠️ 局限性**

局限性包括：目前仅覆盖了特定的块函数与权重取值（{±1}），依赖图取决于简单的无向/单路径结构；缺乏对更复杂约束、层级结构或非二值变量的支持；基准生成仍依赖人工设定，难以全面代表真实工业问题；

---

## 392. Scheduling the Unschedulable: Taming Black-Box LLM Inference at Scale

**arXiv ID:** 2604.06970 | [PDF](https://arxiv.org/pdf/2604.06970v1)

**作者:** Renzhong Yuan `[一作]` (SotaLab), Han Wang `[通讯]` (Zhihu Inc)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在黑盒 LLM API 前进行客户端调度，利用输出长度预测实现半显式预测下的排程。

**💡 创新点**

提出三层分离结构（分配、排序、过载控制），系统评估了粗粒度长度先验对尾延迟、完成率和有效良吞吐量的影响。

**🔧 技术方法**

采用自适应 DRR 权重、可行集打分排序、成本阶梯式拒绝/推迟策略，并使用线性模拟服务模型与 ShareGPT 真实输出分布。

**📊 数据集**

使用合成负载（平衡与重负载两种比例）和 ShareGPT-derived 输出词频分布进行验证。

**📈 对比分析**

与基准（FIFO、配额隔离、Fair Queuing）对比；在高负载下 Full OLC 达到 100% 完成率、约 4.2 req/s 有效吞吐，短尾明显提升，并在不同策略下权衡尾延迟、完成率和吞吐。

**⚠️ 局限性**

局限包括使用模拟器而非真实云 API、阈值手工调优、仅做乘法扰动的预测误差评估，以及未考虑多模型/多租户交叉影响。

---

## 393. MAR-GRPO: Stabilized GRPO for AR-diffusion Hybrid Image Generation

**arXiv ID:** 2604.06966 | [PDF](https://arxiv.org/pdf/2604.06966v1)

**作者:** Xiaoxiao Ma `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 37843 | [OpenAlex ID](https://openalex.org/A5070851446)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并改进了掩码自回归（MAR）模型在强化学习（RL）中的优化，提出了 MAR‑GRPO 框架以提升图像生成的质量和稳定性。

**💡 创新点**

创新点包括：① 将 diffusion head 视为训练不稳定的根源，采用多轨迹期望（MTE）降低梯度噪声；② 基于多轨迹的不确定性仅对 top‑k% 噪声高的 token 进行多轨迹估计；③ 引入一致性感知 token 选择策略，过滤对最终生成无益的 token。

**🔧 技术方法**

使用的技术包括：GRPO（Group Relative Policy Optimization）、多轨迹期望（MTE）、token‑级不确定性评估、相似度阈值 token 过滤、流式马尔可夫采样（AR + diffusion 交替采样）。

**📊 数据集**

训练数据主要是覆盖单/多物体、位置和计数关系的短提示；评估使用 HPS、ImageReward、PickScore、Aesthetic Score、T2I‑CompBench 等指标，并基于 NOVA 与 Harmon 两个 MAR 基础模型。

**📈 对比分析**

与原始预训练模型、GRPO 基线以及固定 diffusion head 的 GRPO 进行对比。结果显示：在 HPS、ImageReward 等人类偏好指标上提升约 10‑15%；在 T2I‑CompBench 的空间结构与组合推理指标上显著优于基线；训练曲线更平滑、奖励更稳定，视觉细节和多样性都有提升。

**⚠️ 局限性**

限制主要在于：① 仍需手动调参（k%、τ 等）；② 多轨迹采样虽然轻量但在大规模训练中仍会增加算力消耗；③ 该框架针对 MAR 结构，未验证在更大尺度或更复杂任务中的适用性。

---

## 394. A First Guess is Rarely the Final Answer: Learning to Search in the Travelling Salesperson Problem

**arXiv ID:** 2604.06940 | [PDF](https://arxiv.org/pdf/2604.06940v1)

**作者:** Andoni Irazusta Garmendia `[一作]` `[通讯]` (University of Basque Country), Andoni Irazusta Garmendia (University of Basque Country)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了基于边的神经改进框架 NICO‑TSP，用于旅行商问题（TSP）的 2‑opt 搜索。

**💡 创新点**

创新点包括：1）用边张量（tour edges）直接表示当前解并去掉位置编码；2）构造了直接评分 2‑opt 动作的 encoder–decoder 结构；3）采用两阶段训练：短期模仿学习（IL）加无评论的群体强化学习（group‑based RL）。

**🔧 技术方法**

使用技术包括：边层注意力编码器、局部循环混合、全局上下文融合、双向注意力解码器、短期模仿学习、无评论 PPO（critic‑free）群体 RL、RMSNorm、AdamW 等。

**📊 数据集**

数据集主要是均匀欧氏 TSP（n=20、50、100、500）和合成 TSPLIB‑gen（n=100），以及标准 TSPLIB 实例用于泛化评估。

**📈 对比分析**

与经典局部搜索（2‑opt、3‑opt、Tabu）、其他神经改进方法（2opt‑DRL、GAT‑Improv、DACT、NeuOpt）以及构造+搜索基线（LEHD、POMO、Efficient Active Search）进行比较；在固定步数或时间预算下，NICO 在大多数设置下取得最优或接近最优解，步速最快、scale 泛化最稳健。

**⚠️ 局限性**

局限性：对超大规模实例（n≫500）仍然退化；由于 2‑opt 的 O(n²) 搜索空间，难以在更大问题上保持高效；教师仅在短期（K=2）做最佳，缺少长期监督，可能限制更深层次搜索策略的学习。

---

## 395. POS-ISP: Pipeline Optimization at the Sequence Level for Task-aware ISP

**arXiv ID:** 2604.06938 | [PDF](https://arxiv.org/pdf/2604.06938v1)

**作者:** Jiyun Won `[一作]` (POSTECH), Sunghyun Cho `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于序列级强化学习的模块化图像信号处理（ISP）优化框架，能够一次性预测完整的模块序列及其参数，从而为给定的下游任务（如目标检测、实例分割、图像增强）生成最优的ISP管线。

**💡 创新点**

核心创新点包括：① 用单一终端奖励而非多步奖励，避免了传统基于步进RL的未来奖励估计不稳定；② 采用GRU递归策略网络捕捉模块间依赖关系，实现上下文感知的序列预测；③ 将参数预测与序列预测联合训练，实现一次前向传播即可完成整个ISP管线的构造，显著降低推理成本。

**🔧 技术方法**

技术实现基于强化学习（REINFORCE）与梯度下降：序列预测器使用GRU+MLP实现条件概率建模；参数预测器采用轻量级CNN编码器-解码器，输出所有模块参数；整体目标通过下游任务损失（如YOLOv3检测损失、实例分割损失或MSE）加上像素强度惩罚作为奖励，分别对两类网络进行联合训练。

**📊 数据集**

实验使用的公开数据集包括：L​OD（低光检测）、LIS（低光实例分割）、Adobe FiveK（图像增强）等，并在这些数据集上对多种下游任务进行评估。

**📈 对比分析**

与DRL-ISP、ReconfigISP、AdaptiveISP以及原始相机ISP基线进行对比。实验结果表明，本文方法在目标检测和实例分割任务上实现了最高的mAP，并在图像增强任务上得到与专家重拍效果相近的视觉质量；同时，推理时间和显存占用均明显低于其他RL/NAS方法，训练稳定性也更好（单次训练收敛、种子波动小）。

**⚠️ 局限性**

局限性在于：① 随着可选ISP模块数量增大，搜索空间膨胀，训练时间与资源需求随之提升；② 目前每个下游任务都需要单独训练，缺乏跨任务共享的通用策略，限制了多任务场景下的可扩展性。

---

## 396. Q-Zoom: Query-Aware Adaptive Perception for Efficient Multimodal Large Language Models

**arXiv ID:** 2604.06912 | [PDF](https://arxiv.org/pdf/2604.06912v1)

**作者:** Yuheng Shi `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 21864 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了 Q‑Zoom，一套查询感知的自适应高分辨率感知框架，能在多模态大语言模型中动态决定是否需要高分辨率处理并精准定位关键信息区域，以显著提升推理效率。

**💡 创新点**

创新点包括：① 通过一致性感知样本生成训练轻量化门控网络，实现查询级的动态路由；② 开发基于自蒸馏的 SD‑RPN，单步从中间特征空间提取 RoI；③ 引入连续时空位置编码与 Post‑SFT，解决局部裁剪与全局上下文对齐问题。

**🔧 技术方法**

主要技术手段：轻量化动态门控网络、Self‑Distilled Region Proposal Network、一致性感知样本生成、三态伪标签自蒸馏、连续时空位置编码、后期定位微调（Post‑SFT）。

**📊 数据集**

使用的主要数据集包括 DocVQA、InfoVQA、ChartQA、OCRBench、TextVQA、V*、MME‑RealWorld、HR‑Bench 以及内部的 Visual CoT 等自蒸馏数据。

**📈 对比分析**

在 Qwen2.5‑VL‑7B 等基准上，Q‑Zoom 与 S²、ViCrop、AdaptVision、SD‑RPN 等方法对比，推理加速 2.52×–4.39×、视觉令牌压缩 50%+，并在文档、OCR、高分辨率任务上保持或超过基线精度，最大精度配置下提升 1.1%–8.1%。

**⚠️ 局限性**

局限性：门控阈值需手动调优；在低基准分辨率模型（如 LLaVA）中 RoI 分支被频繁触发，吞吐提升有限；对极高分辨率图像仍需裁剪与再编码，且位置编码的适配可能需要针对不同架构做进一步调优。

---

## 397. iTAG: Inverse Design for Natural Text Generation with Accurate Causal Graph Annotations

**arXiv ID:** 2604.06902 | [PDF](https://arxiv.org/pdf/2604.06902v1)

**作者:** Wenshuo Wang `[一作]` (South China University of Technology), Wei Li `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 98429 | [OpenAlex ID](https://openalex.org/A5100318082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种名为iTAG的逆向设计框架，用于在保持因果图准确性的同时生成自然文本，解决了传统模板或LLM直接生成方法在自然性与准确性上的折衷问题。

**💡 创新点**

创新点在于将概念赋值视为逆向设计问题，利用链式思维（CoT）在LLM生成过程中进行概念迭代校正，从而在保持全图一致性的前提下实现高质量文本与因果图的双向一致。

**🔧 技术方法**

使用了LLM（如Claude Opus）、逆向设计算法、CoT推理、图结构验证与修正循环，以及基于邻接矩阵的因果图生成与概念映射。

**📊 数据集**

实验数据集包括三类真实文本语料（MIMIC‑IV医学记录、FinCausal商业案例、JUSTICE法律案例）以及通过iTAG生成的合成文本。

**📈 对比分析**

与模板基线和两种LLM基线（单次概念分配、逆向设计修正）相比，iTAG在因果图注释F1≥0.95、SHD与SID均≤1、文本辨别度F1≈0.52（接近随机），且在真实语料上的算法排名与iTAG生成数据高度相关（Pearson≈0.93）。

**⚠️ 局限性**

局限包括：仅提供邻接层面的因果图，无法生成结构方程模型及参数；实验仅覆盖3–10变量的小图和三种英语领域，未验证在大规模或多语言环境中的性能；对非边（a_ij=0）判断仍易出现误判，需更完善的置信度建模与对抗测试。

---

## 398. XR-CareerAssist: An Immersive Platform for Personalised Career Guidance Leveraging Extended Reality and Multimodal AI

**arXiv ID:** 2604.06901 | [PDF](https://arxiv.org/pdf/2604.06901v1)

**作者:** N. D. Tantaroudas `[一作]` (Institute of Communications and Computer Systems), V. Pastrikakis `[通讯]` (CVCOSMOS Ltd)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 XR-CareerAssist，一个将扩展现实与多模态 AI 整合的职业指导平台。

**💡 创新点**

创新点在于将 ASR、NMT、对话代理、视觉语言模型和 TTS 五大 AI 模块在 XR 环境中实现无缝交互，并利用真实职业轨迹生成动态 Sankey 图。

**🔧 技术方法**

技术包括 Unity（Meta Quest 3）、AWS Elastic Beanstalk、FastAPI、ASR（多语种）、NMT、BLIP 视觉语言模型、LangChain 对话代理、AWS Polly TTS、DuckDB 等。

**📊 数据集**

使用 CVCOSMOS 职业简历数据库（10 万+ 匿名 CV）及基于 Sankey 图样本的 k‑means 聚类。

**📈 对比分析**

与传统文本基 CACGS 对比，系统在 10,000 并发用户下 900–1,000 rps 响应 <200 ms，Sankey 生成时间从 45 s 降至 0.2 s，用户满意度 78.3%，语音识别准确率 95.6%。

**⚠️ 局限性**

局限在样本单一、交互时长短、语言覆盖有限、仅支持 Meta Quest 3、对话记忆缺失、行业覆盖不足等。

---

## 399. An empirical study of LoRA-based fine-tuning of large language models for automated test case generation

**arXiv ID:** 2604.06946 | [PDF](https://arxiv.org/pdf/2604.06946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 400. SentinelSphere: Integrating AI-Powered Real-Time Threat Detection with Cybersecurity Awareness Training

**arXiv ID:** 2604.06900 | [PDF](https://arxiv.org/pdf/2604.06900v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (National Technical University of Athens), Andrew J. McCracken `[通讯]` (DASKALOS-APPS)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SentinelSphere 平台，将基于人工智能的威胁检测与 LLM 驱动的安全教育相结合，并集成至 ResilMesh 框架，提供实时威胁检测、交通灯可视化和本地化聊天机器人。

**💡 创新点**

创新点包括：① 通过 HTTP 层特征工程提升 DNN 检测精度并显著降低误报；② 使用量化后的 Phi‑4 LLM 在无 GPU 16GB RAM 环境下实现本地化教育；③ 将核心检测算法从 Python 重写为 Rust，获得 5.6× 速度提升；④ 引入直观的交通灯系统实现跨层级的威胁可视化；⑤ 将威胁检测与安全教育无缝耦合，形成学习型 SOC。

**🔧 技术方法**

技术栈：Enhanced Deep Neural Network（带 HTTP 特征）、量化 Phi‑4 LLM（Q4_K_M）、Rust 实现的检测核心、FastAPI+Vanilla JS Dash‑board、Vector 数据管道、NATS 消息代理、Docker 容器化、交通灯评分算法。

**📊 数据集**

使用 CIC‑IDS2017 与 CIC‑DDoS2019 两大基准数据集，涵盖 Web 攻击、DDoS、DoS、端口扫描等多类攻击。

**📈 对比分析**

通过与 ResilMesh 基线模型对比评估，Enhanced DNN 在 F1 得分上从 0.87 提升至 0.94，精度从 0.85 提升至 0.95，召回率从 0.89 提升至 0.93，误报率下降 69.5%。Rust 重写后在稳定工作负载下实现 5.6× 加速，批处理可达 326×；平台每秒处理超过 6000 条事件，30 分钟内完成 1100 万条事件的批量处理。

**⚠️ 局限性**

局限性：目前仅覆盖 HTTP 攻击，需扩展至 DNS、SMTP、FTP 等协议；缺乏长期用户行为评估；未实现联邦学习和多语言支持；教育模块缺少自适应学习路径；对高复杂场景的实时检测与深度安全分析仍需进一步研究。

---

## 401. Auditing Demographic Bias in Facial Landmark Detection for Fair Human-Robot Interaction

**arXiv ID:** 2604.06961 | [PDF](https://arxiv.org/pdf/2604.06961v1)

**作者:** Pablo Parte `[一作]` (Universidad Politénica de Madrid), Luis Baumela `[通讯]` (Universidad Politénica de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估并剖析人机交互中面部关键点检测的种族、性别和年龄偏差，剔除视觉混杂因素后进行公平性审计。

**💡 创新点**

提出基于线性回归与Box‑Cox变换的统计框架，能将人口属性与视觉混杂因素分离，揭示年龄偏差是主要残留因素。

**🔧 技术方法**

采用U‑Net+ResNet‑34框架的面部对齐模型、Headpose估计、线性回归、emmeans CI等统计工具。

**📊 数据集**

使用RAF‑DB、WFLW以及基于FairFace生成的性别、种族、年龄标签。

**📈 对比分析**

通过多模型线性回归与方差分析比较各变量影响，结果显示头姿与分辨率主导误差，性别和种族差异消失，年龄差异显著。

**⚠️ 局限性**

样本不均衡导致对老年人的偏差无法消除，仅在单一模型上验证，未探讨多模型或跨域泛化。

---

## 402. NestPipe: Large-Scale Recommendation Training on 1,500+ Accelerators via Nested Pipelining

**arXiv ID:** 2604.06956 | [PDF](https://arxiv.org/pdf/2604.06956v1)

**作者:** Zhida Jiang `[一作]` (JD.com), Ke Zhang `[通讯]` (JD.com)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

提出了 NestPipe 框架，利用嵌套流水线（DBP+FWP）实现大规模分布式稀疏嵌入训练，显著降低查找和通信瓶颈。

**💡 创新点**

创新点在于：①双缓冲流水线实现无脏失真五阶段并行，消除嵌入查找延迟；②冻结窗口流水线利用参数冻结窗口，将 All2All 通信与密集计算重叠，保持同步语义；③两层层次稀疏并行（交叉批级与微批级）实现查找与通信同时隐藏。

**🔧 技术方法**

采用双缓冲同步、All2All 通信、流调度、关键字聚类、键去重等技术，结合 GPU/NPU 硬件并行、主机/设备层次存储。

**📊 数据集**

使用工业级 KuaiRand‑27K 与自建 Industrial 数据集，训练 HSTU 与 FUXI 模型。

**📈 对比分析**

在 128–1,536 机器（GPU/ NPU）上与 TorchRec、2D‑SP、UniEmb 进行对比，NestPipe 在 1,536 机上实现 3.06× 速度提升、94.07% 规模效率；与 2D‑SP 组合进一步提升至 4.32× 速度和 97.17% 规模效率。

**⚠️ 局限性**

局限性包括：仅在特定 1,536 机规模上验证，尚未探索更大规模或不同网络拓扑；对极大通信量场景的极限性能仍未知；实现复杂度较高，需精细调优双缓冲与流调度。

---

## 403. NTIRE 2026 Challenge on Bitstream-Corrupted Video Restoration: Methods and Results

**arXiv ID:** 2604.06945 | [PDF](https://arxiv.org/pdf/2604.06945v1)

**作者:** Wenbin Zou `[一作]`, Yaokun Shi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文组织并评估了NTIRE 2026 Bitstream-Corrupted Video Restoration (BSCVR) 挑战，提供了真实比特流损坏视频的数据集、评测协议，并汇总参赛团队的模型与最终排名。

**💡 创新点**

创新点包括：①构建了大规模真实损坏视频基准（BSCV），覆盖块、色彩、复制、位移、纹理缺失、尾部等多种损坏模式；②提出多阶段、跨模态融合的三阶段恢复框架（如MGTV-AI）和单步扩散+VAE的双阶段训练（如RedMediaTech），以及结合SAM2、LoRA等视觉基础模型的轻量化方案；③在评测中采用 PSNR、SSIM 与 LPIPS 综合量化，揭示了生成模型在主观质量与客观指标之间的权衡。

**🔧 技术方法**

使用的技术主要包括：基于B2SCVR的基线网络、BasicVSR++、NAFNet、Wan2.1 Diffusion Transformer、Qwen-Image VAE、SAM2/ DINOv3 语义/结构先验、LoRA/MoE-LoRA 参数高效微调、光流估计与双向时间增强、以及多尺度残差细化等。

**📊 数据集**

所用数据集为官方 BSCV 基准数据集，包含 3,471 条 HD 质量的损坏视频、对应的原始视频和二值掩码；验证集 50 条视频，测试集 公开仅提供损坏视频与掩码。

**📈 对比分析**

评测方法以平均 PSNR/SSIM 作为主指标，并通过 LPIPS 衡量感知质量。最终排名显示 MGTV-AI 以 33.642 dB PSNR、0.9334 SSIM 领跑；RedMediaTech 在 LPIPS 上获得 0.0852 的最佳感知分数；其他参赛队伍（Bighit、Vroom、Weichow、Holding、NTR）在 PSNR/SSIM 与 LPIPS 上落后。

**⚠️ 局限性**

局限性包括：①即便使用先进模型，仍易出现边缘模糊、细节缺失（如人脸、文字）与大运动场景的时间不连贯；②多阶段/多模型框架往往参数量大、推理时间长；③大多数方法仅在官方基准上验证，泛化到不同编码器或更极端损坏场景的鲁棒性尚待提升。

---

## 404. Evaluating PQC KEMs, Combiners, and Cascade Encryption via Adaptive IND-CPA Testing Using Deep Learning

**arXiv ID:** 2604.06942 | [PDF](https://arxiv.org/pdf/2604.06942v1)

**作者:** Simon Calderon `[一作]` (Linköping University), Onur Günlü `[通讯]` (Linköping University)

**通讯引用:** 763 | [OpenAlex ID](https://openalex.org/A5016620064)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

通过深度神经网络（DNN）对后量子密码学（PQC）密钥封装机制（KEM）及其混合和级联加密方案进行 IND‑CPA 可区分性实验，验证其在真实实现中的可区分性是否满足理论安全性。

**💡 创新点**

① 将 IND‑CPA 游戏转化为二分类问题，并使用 DNN 进行训练；② 将该方法推广到混合 KEM（使用 XOR 或伪随机函数组合）和级联对称加密；③ 在实验中加入严格的两侧二项检验，量化模型是否显著优于随机猜测，首次实现对 PQC KEM 和混合加密的经验性安全性评估。

**🔧 技术方法**

使用多层全连接 DNN（ReLU 激活，输出 sigmoid），交叉熵（BCE）损失，SGD + NAG 优化，早停、学习率衰减、Glorot 初始化等深度学习技术；同时采用两套网络规模（小型和大型）进行对比。

**📊 数据集**

每个实验使用 500,000 条样本做训练（两类各 250k），100,000 条做验证，100,000 条做最终测试；样本由相同密钥对生成，包含 ML‑KEM、HQC、BIKE、RSA、RSA‑OAEP、明文以及级联对称加密（AES‑CTR/ECB/CBC、ChaCha20、DES‑ECB）的密文。

**📈 对比分析**

通过训练得到的分类准确率与 50% 随机猜测做对比，并使用两侧二项检验（α=0.01）判断显著性。实验结果表明：所有单一或混合 KEM、以及所有级联加密方案的 DNN 准确率均≈50%（p‑值 >0.01），即没有显著可区分性；仅在已知明文或无填充 RSA 的特殊组合中出现完美区分，但这归因于模型缺乏加密 Oracle，非 IND‑CPA 失效。

**⚠️ 局限性**

模型仅基于密文特征，无法访问公钥、加密/解密 oracle，因而无法捕捉某些攻击向量；此外，仅检测了对二分类任务的学习能力，无法评估更细粒度的攻击效果；模型对不同实现细节（如填充、错误处理）的敏感性也未被系统性研究。

---

## 405. LungCURE: Benchmarking Multimodal Real-World Clinical Reasoning for Precision Lung Cancer Diagnosis and Treatment

**arXiv ID:** 2604.06925 | [PDF](https://arxiv.org/pdf/2604.06925v1)

**作者:** Fangyu Hao `[一作]` (Beijing University of Posts and Telecommunications), Yingyi Wang `[通讯]` (Peking Union Medical College Hospital)

**通讯引用:** 7066 | [OpenAlex ID](https://openalex.org/A5087251992)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研发了基于真实临床病例的多模态肺癌决策支持基准（LungCURE）并提出了知识驱动的多代理框架（LCAgent），用于 TNM 分期、治疗推荐和端到端决策。

**💡 创新点**

创新点在于：①将肺癌临床工作流程拆分为三大任务并统一评估；②构建包含 1,000 真实病例的标准化多模态基准；③设计多代理结构化推理方案，显著抑制跨阶段推理误差并提升指南合规性。

**🔧 技术方法**

采用多模态大型语言模型（如 Qwen、GLM、Kimi 等）、OCR+LLM 处理文本、结构化特征抽取与情境路由、以及多代理（分为分期、诊断、治疗三阶段）推理。

**📊 数据集**

使用 1,000 例真实临床病例，涵盖影像报告、病理报告、临床记录和基因检测结果，数据来自 10+ 家医院，已公开发布于 HuggingFace。

**📈 对比分析**

通过准确率、推理质量、Precision 和 BERT‑F1 等多指标评估，LCAgent 在多模型、多任务、多模态场景下平均提升 20–30% 的决策质量，且在所有模型上均实现了显著的 win‑rate。

**⚠️ 局限性**

局限性包括：对医学专业知识的掌握仍不充分；跨阶段信息传递在极端复杂病例中仍可能失真；中文与英文表现差异显著，需进一步改进多语言支持。

---

## 406. IQ-LUT: interpolated and quantized LUT for efficient image super-resolution

**arXiv ID:** 2604.07000 | [PDF](https://arxiv.org/pdf/2604.07000v1)

**作者:** Yuxuan Zhang `[一作]`, Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 84494 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 IQ‑LUT 模型，利用 LUT 加速图像超分辨率推理，显著减小存储需求并提升视觉质量。

**💡 创新点**

创新点包括：1）在 ECNN 基础上引入单输入多输出结构与残差学习；2）双路径融合插值（DPFI）降低位深但用插值替代显式存储；3）非均匀量化与知识蒸馏（NUQD）进一步压缩 LUT 并补偿量化损失。

**🔧 技术方法**

核心技术：扩展卷积（ECNN）、双线性/双路径插值、残差连接、非均匀量化+蒸馏、基于 LUT 的推理与 PixelShuffle 上采样。

**📊 数据集**

训练使用 DIV2K 数据集；评估在 Set5、Set14、B100、Urban100、Manga109 五个公开基准集上。

**📈 对比分析**

与现有 LUT 方法（如 SR‑LUT、MuLUT、TinyLUT 等）和 ECNN 进行 PSNR/SSIM 对比。IQ‑LUT 在 124 KB 时取得 Set5 31.50 dB PSNR、28.88 dB SSIM，远低于 ECNN 的 1/50 存储并且仅比 ECNN 多 2 倍延迟，显示出优越的存储‑性能折衷。

**⚠️ 局限性**

主要限制：DPFI 仍引入一定的计算延迟；量化参数需手工调优；在极低位深时可能导致细节损失；模型针对专用硬件（ASIC）优化，通用 CPU/GPU 性能未充分验证。

---

## 407. TRAPTI: Time-Resolved Analysis for SRAM Banking and Power Gating Optimization in Embedded Transformer Inference

**arXiv ID:** 2604.06955 | [PDF](https://arxiv.org/pdf/2604.06955v1)

**作者:** Jan Klhufek `[一作]` (Brno University of Technology), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11259 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

TRAPTI提出一种两阶段流程，先用TransInferSim对Transformer推理进行周期级仿真并提取时变的SRAM占用轨迹，再利用该轨迹离线探索不同银行化和电源门控方案，比较GPT-2 XL（MHA）与DeepSeek-R1-Distill-Qwen-1.5B（GQA）两模型在相同加速器上的峰值SRAM占用、能耗与延迟差异，证明GQA工作负载在相同SRAM容量下可将峰值占用降低2.72×、推理时间缩短1.89×，并通过银行化与电源门控实现最高78%能耗下降。

**💡 创新点**

创新点在于结合时序占用分析与离线优化，首次系统性利用时变SRAM占用轨迹指导银行化与电源门控设计。

**🔧 技术方法**

使用TransInferSim进行周期级仿真，CACTI进行SRAM功耗与面积建模，离线遍历银行数与功耗门控策略。

**📊 数据集**

使用标准Transformer模型权重，序列长度2048的推理任务（GPT-2 XL与DeepSeek-R1-Distill-Qwen-1.5B）。

**📈 对比分析**

通过对比两模型在相同硬件配置下的占用峰值、能耗与延迟，结果显示GQA显著降低峰值占用和延迟，并在银行化与电源门控下实现能耗下降高达78%。

**⚠️ 局限性**

限制在于未考虑多级内存调度细节、更精细的转移开销模型及数据迁移路径对整体性能的影响。

---

## 408. Compression as an Adversarial Amplifier Through Decision Space Reduction

**arXiv ID:** 2604.06954 | [PDF](https://arxiv.org/pdf/2604.06954v1)

**作者:** Lewis Evans `[一作]`, Shreyank N Gowda `[通讯]` (University of Nottingham)

**通讯引用:** 521 | [OpenAlex ID](https://openalex.org/A5041351493)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了图像压缩在推理过程中对对抗鲁棒性的影响，提出了压缩-aware 对抗攻击模型，并通过实验展示压缩会放大对抗扰动的效果。

**💡 创新点**

创新点在于揭示压缩通过“决策空间收缩”机制将决策边界拉近输入，使得同等扰动在压缩空间更易导致误判；同时首次系统比较了压缩前后攻击的顺序效应，证明压缩可既放大亦可减弱攻击。

**🔧 技术方法**

主要技术包括基于 JPEG/ PCA/ PatchSVD 的压缩方法、FGSM/PGD/AutoAttack 等对抗攻击、决策空间可视化与度量（真类区域比例、平均边距、边界侵入率）以及理论推导的鲁棒半径上界。

**📊 数据集**

使用了 CIFAR-10、CIFAR-100、ImageNet 三个公开数据集，并在 ResNet、ViT 等多种网络结构上进行评估。

**📈 对比分析**

与传统像素空间攻击对比，压缩-aware 攻击在相同 PSNR/扰动预算下准确率显著下降（如 CIFAR-100 ResNet-18 从 23.4% 降至 8.5%），实验还表明压缩顺序影响显著，压缩后再攻击比先攻击后压缩导致更高误差。

**⚠️ 局限性**

局限在于仅考虑了离散的有损压缩方式，未探究可微分压缩或更复杂编码；此外，实验未覆盖所有可能的对抗训练或防御方案，故对抗训练模型对压缩-aware 攻击的抵抗力仍需进一步研究。

---

## 409. Multi-modal user interface control detection using cross-attention

**arXiv ID:** 2604.06934 | [PDF](https://arxiv.org/pdf/2604.06934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 410. Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation

**arXiv ID:** 2604.06950 | [PDF](https://arxiv.org/pdf/2604.06950v1)

**作者:** Zhiheng Li `[一作]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Casia), Weiming Hu `[通讯]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Casia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并系统化了针对多模态大语言模型的“对抗性托运攻击”，并构建了SmuggleBench基准。

**💡 创新点**

首次将对抗性托运攻击分为感知盲区和推理屏障两条路径，并通过大规模基准评估揭示其普遍漏洞。

**🔧 技术方法**

采用链式思考提示（CoT）和监督微调（SFT）等技术来探索防御方案。

**📊 数据集**

使用1700个手工与自动生成的对抗样本组成的SmuggleBench数据集。

**📈 对比分析**

对比评估发现GPT‑5、Gemini 2.5 Pro和Qwen3‑VL等模型的ASR均超过84%，CoT/ SFT虽有降幅但FPR显著上升，说明防御效果有限。

**⚠️ 局限性**

研究仅覆盖中英文文本与静态图像，未评估低资源语言、视频等场景，也未深入不同视觉编码器的影响。

---

## 411. Learning-Based Strategy for Composite Robot Assembly Skill Adaptation

**arXiv ID:** 2604.06949 | [PDF](https://arxiv.org/pdf/2604.06949v1)

**作者:** Khalil Abuibaid `[一作]` (RPTU University Kaiserslautern-Landau), Martin Ruskowski `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种可重用且封装的基于技能的插销-孔插接策略，通过残差强化学习对对齐与插入技能进行自适应改进。

**💡 创新点**

创新点在于将残差强化学习嵌入到预定义技能的残差修正层面，仅在技能内部细微调整，保持整体流程不变，实现安全、样本高效且模块化的学习。

**🔧 技术方法**

使用混合力-运动控制器、SAC+JAX 的残差强化学习框架，并在 MuJoCo 仿真下的 UR5e + Robotiq 2F‑85 手抓手上实现。

**📊 数据集**

采用 MuJoCo 生成的方形插销与孔模型，并注入关节状态与力矩噪声，模拟多种摩擦系数和姿态误差的多样化数据。

**📈 对比分析**

与基线混合力–运动控制器对比，训练 2.5×10⁵ 步后，残差强化学习在插接成功率、碰撞惩罚和鲁棒性上均优于基线。

**⚠️ 局限性**

局限性在于残差模型仅为单一全局网络，难以针对不同对齐与插入阶段实现专属优化；且目前仅在仿真中验证，真实机器人迁移尚待评估。

---

## 412. Sustainable Transfer Learning for Adaptive Robot Skills

**arXiv ID:** 2604.06943 | [PDF](https://arxiv.org/pdf/2604.06943v1)

**作者:** Khalil Abuibaid `[一作]` (RPTU University Kaiserslautern-Landau), Martin Ruskowski `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在UR5e和Panda两台机器人上使用强化学习实现并评估了插销插孔任务的策略迁移，比较了从零开始、零射击以及微调三种训练方式；

**💡 创新点**

创新点在于展示了通过对已训练策略进行少量微调即可在不同机器人平台间实现高成功率与低执行时间，显著提升样本效率和泛化能力；

**🔧 技术方法**

采用了SAC算法与自适应混合运动‑力控制相结合的RL框架，并对PID增益进行在线学习；

**📊 数据集**

使用MuJoCo仿真环境中的UR5e和Panda两台机器人数据，没有引用公开的工业或视觉数据集；

**📈 对比分析**

通过零射击、微调和从零训练三种对比，微调后的Panda模型成功率提升至97.98%，平均时间步数从170下降至135，表现优于零射击且仅次于同平台从零训练；

**⚠️ 局限性**

仅在仿真环境中验证，缺乏真实机器人实验；仅涉及两种机器人，难以证明对更广泛平台的泛化能力；

---

## 413. The AI Skills Shift: Mapping Skill Obsolescence, Emergence, and Transition Pathways in the LLM Era

**arXiv ID:** 2604.06906 | [PDF](https://arxiv.org/pdf/2604.06906v1)

**作者:** Rudra Jadhav `[一作]` (Savitribai Phule Pune University), Janhavi Danve `[通讯]` (Savitribai Phule Pune University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了Skill Automation Feasibility Index（SAFI），对35个O*NET技能在四个前沿LLM上进行263项文本任务的基准测试，并结合Anthropic Economic Index的真实AI采纳数据，提出AI Impact Matrix来评估各技能的自动化风险与AI使用关系；

**💡 创新点**

创新点在于：①将LLM能力与标准职业技能税onomies对齐，提供可量化的SAFI评分；②结合真实AI采纳数据形成四象限AI Impact Matrix；③发现“能力-需求倒置”现象，即AI高采纳职业中最重要的技能在文本基准上表现最差；

**🔧 技术方法**

采用四大LLM（LLaMA 3.3 70B、Mistral Large、Qwen 2.5 72B、Gemini 2.5 Flash），使用温度0.3的统一提示；评分采用四维度启发式引擎（完整性、深度、推理质量、难度奖金），避免LLM-as-judge偏差；

**📊 数据集**

数据集包括：O*NET 30.2版（35技能共1016职业），Anthropic Economic Index（756职业、17998任务），以及作者自制的263项技能任务共1052模型响应；

**📈 对比分析**

比较方法：对每个技能求四模型平均分，标准化为0–100的SAFI；通过与Anthropic AI曝光相关系数（Pearson/ Spearman）构建四象限；发现SAFI与AI曝光呈负相关（r≈-0.2）；大模型之间仅3.6点差异，说明技能层面比模型层面更决定自动化可行性；

**⚠️ 局限性**

局限性包括：①SAFI仅评估文本表达的技能，忽略物理、交互和实时维度；②启发式评分对主观判断有限，未验证事实正确性；③任务覆盖有限，部分技能任务数量不足；④Anthropic Index仅基于Claude，对其他AI平台泛化有限；⑤横断面研究，随模型升级需持续跟踪；⑥AI Impact Matrix为解释框架而非预测模型，实际失业影响受多因素影响。

---

## 414. Is Biomedical Specialization Still Worth It? Insights from Domain-Adaptive Language Modelling with a New French Health Corpus

**arXiv ID:** 2604.06903 | [PDF](https://arxiv.org/pdf/2604.06903v1)

**作者:** Aidan Mannion `[一作]` (Université Grenoble Alpes), François Portet `[通讯]` (Université Grenoble Alpes)

**通讯引用:** 3253 | [OpenAlex ID](https://openalex.org/A5079066445)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了首个完全符合许可的法语医学语料库，并基于Qwen3模型开展了领域自适应预训练（DAPT）和模型融合实验；

**💡 创新点**

创新点在于首次提供可商用的法语医学语料、系统评估DAPT对小模型的提升效果，并提出将域适配模型与基础模型通过SLERP融合以保持泛化能力；

**🔧 技术方法**

技术主要包括基于Qwen3家族的无监督因果语言建模、PDAPT训练（4,320步、2,048-token序列）、SLERP模型融合，以及使用lm-evaluation-harness进行多语言多任务评测；

**📊 数据集**

使用了涵盖科学论文、药品说明书、临床案例、试验方案、对话等多源文本的“PARTAGES Corpus of Open Medical Documents”共计约1.95 B token（892K文档可商用版）；

**📈 对比分析**

通过MMLU与MMLU-Pro‑X四个任务组进行few‑shot评估，发现小型模型在法语医学任务上可获得显著提升，而大模型提升有限；融合后消除了DAPT导致的泛化退化，整体性能相对稳定；

**⚠️ 局限性**

局限性包括仅使用单一预训练目标（因果语言建模）、未探索监督微调与指令调优、模型融合方法局限于SLERP，且评测任务不覆盖所有医学子领域。

---

## 415. Are LLMs Ready for Computer Science Education? A Cross-Domain, Cross-Lingual and Cognitive-Level Evaluation Using Professional Certification Exams

**arXiv ID:** 2604.06898 | [PDF](https://arxiv.org/pdf/2604.06898v1)

**作者:** Chen Gao `[一作]` (Sun Yat-sen University), Xiaotong Han `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3484 | [OpenAlex ID](https://openalex.org/A5038911819)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四大先进LLM（GPT-5、DeepSeek-R1、Qwen-Plus、Llama-3.3）在六项计算机科学认证考试题库上进行系统评测。

**💡 创新点**

首次从跨语言、跨领域、认知层次、置信度校准和输入鲁棒性等多维度构建教育评估框架，并通过真实认证题目验证LLM在高阶认知任务和信息缺失情境下的表现。

**🔧 技术方法**

采用Prompt Engineering、Bloom层级分类、置信度标记、随机遮蔽扰动和Bootstrap置信区间等技术，利用官方API获取模型输出。

**📊 数据集**

使用1,068道题目，来源于六个权威认证考试（CCNA、ICDL、NCRE MS Office、NCRE Java、OCJP、Chinese Network Engineer），并制作中英双语并行版本。

**📈 对比分析**

对四模型在各考试、语言、子领域、Bloom层级、置信度水平和遮蔽比例下的准确率进行对比；结果显示GPT-5在英文本上最高，Qwen-Plus在中文测试中占优，DeepSeek-R1表现最稳健，Llama-3.3在高阶认知和高遮蔽下显著下降。

**⚠️ 局限性**

局限包括仅涵盖三大领域、仅测评单轮多选题、缺乏人类对照、静态题库无法反映课堂对话、翻译质量有限等。

---

## 416. VertAX: a differentiable vertex model for learning epithelial tissue mechanics

**arXiv ID:** 2604.06896 | [PDF](https://arxiv.org/pdf/2604.06896v1)

**作者:** Alessandro Pasqui `[一作]` (Université PSL), Hervé Turlier `[通讯]` (Université PSL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个基于JAX的可微分顶点模型框架VertAX，用于前向模拟、力学参数推断以及逆设计上皮组织行为，并演示了从实验显微图像直接推断力学参数的可能。

**💡 创新点**

核心创新包括：①将顶点模型与JAX自动微分、GPU加速和双层优化统一；②实现并基准三种梯度估计方法（自动微分、隐式微分、平衡传播），提供了可直接扩展到不可微模拟的路径；③提出可微分拓扑损失IAS以克服T1转化导致的拓扑障碍；④通过显微图像合成损失实现无显式细胞分割的参数推断；⑤演示逆设计任务（机械设计、形态延伸、分化模式）在单一框架内实现。

**🔧 技术方法**

技术栈包括：JAX + Optax（梯度下降、Adam、Nesterov等），自动微分、隐式微分、平衡传播；顶点模型能量函数（面积-周长、弹性-线张力）；Half‑Edge 数据结构；周期/边界条件与T1转化；Sinkhorn-Knopp（结构损失）；FFT 与 Bessel 函数（显微图像损失）；实验细胞培养与光学成像。

**📊 数据集**

数据集：①合成实验——随机生成目标形状因子 p⁰ₐ 的高斯分布；②真实实验——A549 上皮细胞在对照与 TGF‑β 处理条件下的显微图像（亮度、E‑cadherin/DAPI）

**📈 对比分析**

比较方法：在相同合成逆推断任务上对 AD、Adj‑ID、Sen‑ID 与 EP 进行跑 5 次重复；结果显示：①全部方法在 200–350 轮内收敛，EP 在收敛速度与最终误差上略优；②运行时间相差 1 个数量级，Adj‑ID 速度最快；③内存使用方面 AD 约为 EP/ID 的 10 倍，EP/Adj‑ID 在内存上最轻量；④在不同细胞数下，三种方法均保持在相同的误差范围。

**⚠️ 局限性**

局限性：①隐式微分对 Hessian 的正定性要求苛刻，易因顶点不收敛或 T1 变化导致不稳定；②平衡传播需多次模拟且 β 的选取对精度敏感；③在复杂拓扑迁移时，外层损失若仅为几何距离易陷入局部极小，需要引入结构损失；④对极大规模系统（>10⁴ 细胞）时，计算与内存成本仍是瓶颈；⑤实现中对动态图的控制流需静态化，限制了某些动态细胞过程的直接建模。

---

## 417. ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals

**arXiv ID:** 2604.06997 | [PDF](https://arxiv.org/pdf/2604.06997v1)

**作者:** Yihao Wang `[一作]` (Sun Yat-Sen University), Keze Wang `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 2307 | [OpenAlex ID](https://openalex.org/A5088124671)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于春秋左传等史料的时间键检索基准ChunQiuTR，并提出一种时序感知双编码器CTD以提高检索精度。

**💡 创新点**

创新点在于：①引入以“公-年-月”为单位的非公历时间键，模拟真实史料检索场景；②设计隐式日历轴并将绝对时间上下文和相对时间偏置融合进双编码器，强化时序一致性；③使用区间重叠多正样本对齐与辅助时间分类联合训练。

**🔧 技术方法**

使用Transformer共享双编码器，加入Fourier编码的时间上下文、相对时间偏置、InfoNCE多正损失及时间分类交叉熵。

**📊 数据集**

数据集为从《春秋》及其三大注（左传、公羊传、穀梁传）手工校准的时间键化检索数据，共20,172条记录、16,226个查询。

**📈 对比分析**

与BM25、SPLADE、ColBERT、mE5、BGE、Qwen3-Embed等基线比较，CTD在R@1、MRR@10、nDCG@10等指标上分别提升约7–8点、4–5点，显示显著优势。

**⚠️ 局限性**

局限在于只覆盖春秋典籍的公-年-月层级，无法直接推广至其他历史体裁或更细粒度时间；仍难处理相邻月相似记录和真实史料歧义。

---

## 418. Generative Phomosaic with Structure-Aligned and Personalized Diffusion

**arXiv ID:** 2604.06989 | [PDF](https://arxiv.org/pdf/2604.06989v1)

**作者:** Jaeyoung Chung `[一作]` (Seoul National University), Kyoung Mu Lee `[通讯]` (Seoul National University)

**通讯引用:** 27196 | [OpenAlex ID](https://openalex.org/A5046504049)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的生成式相框拼贴方法，直接合成符合全局结构且可控的瓷砖图像。

**💡 创新点**

首次将生成式模型用于拼贴，结合低频结构引导、颜色自适应正则、积分噪声初始化和少样本个性化扩散，实现了结构一致、细节多样且可个性化的拼贴。

**🔧 技术方法**

使用Stable Diffusion 2.1、低频梯度引导、AdaIN色彩对齐、LoRA个性化、Mix-of-Show等技术。

**📊 数据集**

主要使用参考图像自身分块作为数据，并采集少量用户图像进行LoRA微调；不依赖大型瓷砖图像库。

**📈 对比分析**

与传统的特征匹配、色调/直方图匹配以及基于扩散的色彩控制、噪声混合等方法对比，利用PSNR/SSIM/LPIPS、BLIP/CLIP/CLIP‑IQA、HPSv2和IR等指标，实验显示在全球结构与局部细节两方面均优于或与基线持平，且在人类偏好测试中获得较高的优选率。

**⚠️ 局限性**

仍受限于低频引导与高频细节的权衡、对颜色精度的微调不足、生成速度较慢以及对极端视角/分辨率的适配有限。

---

## 419. Frailty Estimation in Elderly Oncology Patients Using Multimodal Wearable Data and Multi-Instance Learning

**arXiv ID:** 2604.06985 | [PDF](https://arxiv.org/pdf/2604.06985v1)

**作者:** Ioannis Kyprakis `[一作]` (Harokopio University), Manolis Tsiknakis `[通讯]` (Harokopio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究利用多模态可穿戴设备监测老年乳腺癌患者的功能性衰弱变化。

**💡 创新点**

创新点是将注意力机制的多实例学习与多模态可穿戴数据融合，在缺失与弱监督下实现功能变化预测。

**🔧 技术方法**

使用了注意力多实例学习（MIL）、多模态MLP编码器以及心率变异性、活动与睡眠特征。

**📊 数据集**

使用了CARDIOCARE多中心临床试验收集的可穿戴数据（Garmin Venu、Polar H10）以及FACIT‑F和握力评估。

**📈 对比分析**

在留一人交叉验证下，模型在M3/M6上对握力的平衡准确率≈0.68‑0.70、F1≈0.67‑0.69；对FACIT‑F平衡准确率≈0.59‑0.64、F1≈0.58‑0.63。

**⚠️ 局限性**

局限包括弱监督与离散化导致标签噪声、可穿戴数据缺失、中心差异及HRV采样稀疏。

---

## 420. Grounded Forcing: Bridging Time-Independent Semantics and Proximal Dynamics in Autoregressive Video Synthesis

**arXiv ID:** 2604.06939 | [PDF](https://arxiv.org/pdf/2604.06939v1)

**作者:** Jintao Chen `[一作]` (Peking University), Mu Xu `[通讯]` (Alibaba)

**通讯引用:** 486 | [OpenAlex ID](https://openalex.org/A5100532751)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对长时序视频生成中的语义遗忘、视觉漂移和交互控制难题，提出了 Grounded Forcing 框架，实现了可实时、可交互的长时间视频生成。

**💡 创新点**

创新点包括：1）Dual Memory KV Cache，将短期动态与长期语义分离；2）Dual-Reference RoPE Injection，通过固定全局索引和相对本地索引抑制位置漂移；3）Asymmetric Proximity Recache（APR），实现基于时间近似度的渐进式 KV 缓存更新，平滑提示切换。

**🔧 技术方法**

主要技术：基于 Transformer 的自回归视频扩散模型（Wan2.1-T2V-1.3B），结合 KV 缓存、RoPE 注入、分层记忆管理与渐进式缓存刷新。

**📊 数据集**

使用数据集：MovieGenBench 提取 100 条文本提示进行统一评估，VBench 进行量化评价；交互实验使用 Qwen3-Max 生成 50 条交互式提示列表；视频长度覆盖 5s、60s、120s、240s。

**📈 对比分析**

与 LongLive、Rolling Forcing、Infinity‑RoPE 等基线在同一模型规模下对比。Grounded Forcing 在 240s 视频上实现了背景一致性 0.9265、主体一致性 0.9163 的最高分，并在 Aesthetic 及 Temporal Flickering 上位列前列，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性：1）依赖 WAN2.1 基础网络，效果可能不易迁移至其他架构；2）仍受限于 KV 缓存窗口大小，极长序列仍可能出现累计误差；3）对极端场景切换（如完全不同的视觉域）仍需进一步实验验证。

---

## 421. Equivariant Multi-agent Reinforcement Learning for Multimodal Vehicle-to-Infrastructure Systems

**arXiv ID:** 2604.06914 | [PDF](https://arxiv.org/pdf/2604.06914v1)

**作者:** Charbel Bou Chaaya `[一作]` (University of Oulu), Mehdi Bennis `[通讯]` (University of Oulu)

**通讯引用:** 45379 | [OpenAlex ID](https://openalex.org/A5061429095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在车联网（V2I）环境中，提出一种自监督的多模态感知框架，利用摄像头图像和CSI无标签数据对车辆位置进行对齐与估计，并结合群体对称性设计的全局等变（equivariant）多智能体强化学习（MARL）网络，实现在分布式RSU下的基站资源分配和波束对齐，以最大化系统吞吐量。

**💡 创新点**

创新点包括：
1) 将旋转对称性引入到MMDP框架，提出“潜在对称性”（latent symmetries）概念，解决观测空间不具对称性的挑战。
2) 设计了自监督的跨模态对齐算法，利用图像定位和CSI图表化（channel charting）实现模态匹配，消除标签需求。
3) 构建基于GNN的等变策略网络，兼顾消息传递与全局对称性，实现分布式执行且样本效率显著提升。

**🔧 技术方法**

核心技术包括：
- YOLOv7（目标检测）
- 旋转对称性（SO(2) 4阶子群）与等变层（Symmetrizer）
- 关联图网络（GNN）用于消息传递和策略更新
- 自监督损失（对齐、图表化、交叉模态蒸馏）
- PPO（近端策略优化）和GAE（通用优势估计）用于训练。

**📊 数据集**

使用基于Blender+Sionna的合成数据集：3,000张摄像头图像（约5,000辆车）和35,000条CSI样本；图像通过YOLOv7检测后定位，CSI通过自监督图表化处理。

**📈 对比分析**

与传统CSI-only图表化、全监督定位、基于数据增强或中心化控制的基线相比：
- 位置估计均方误差从3.9 m降至1.44 m（约64%提升）。
- 等变MARL相较于非等变网络提升≈20%，与数据增强方法相当但训练速度更快；在大规模天线/用户场景下，速度提升超过80%。
- 与集中式或无通信的基线相比，吞吐量提升≥1.5 Gbps。

**⚠️ 局限性**

局限性：
- 对称性假设仅在理想的C4旋转对称环境下成立，实际道路布局偏离时性能会下降。
- 需要离线收集大量无标签图像/CSI样本进行自监督训练，部署成本仍高。
- 对时序同步误差敏感，20 ms以上的时延会导致定位误差显著上升。
- 仅考虑二维旋转对称，未扩展至更一般的欧氏变换（平移、反射）。

---

## 422. Data Leakage in Automotive Perception: Practitioners' Insights

**arXiv ID:** 2604.06899 | [PDF](https://arxiv.org/pdf/2604.06899v1)

**作者:** Md Abu Ahammed Babu `[一作]` (Volvo Cars), Miroslaw Staron `[通讯]` (University of Gothenburg)

**通讯引用:** 3299 | [OpenAlex ID](https://openalex.org/A5074020697)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过在汽车感知领域内对10名设计、实现和验证工程师进行半结构化访谈，探讨工业实践中对数据泄漏的认知、经验与缓解策略。

**💡 创新点**

创新点在于从角色视角系统性揭示数据泄漏认知与实践的差异，将其定位为跨角色的社会技术协调问题，而非单纯技术缺陷。

**🔧 技术方法**

采用反射性主题分析（reflexive thematic analysis）对访谈文本进行编码和归纳，并结合访谈问卷设计与研究流程。

**📊 数据集**

使用的数据为访谈记录及参与者信息，研究对象是汽车制造商内部负责感知功能的工程团队，而非传统机器学习数据集。

**📈 对比分析**

通过主题编码比较不同角色对数据泄漏的定义、经验、检测与预防措施，发现多数实践依赖模型性能异常、图像相似度检查和规则化拆分，未给出定量性能指标。

**⚠️ 局限性**

局限性包括样本仅来自单一OEM、受访者数量有限，导致外部有效性受限；此外研究聚焦认知与实践缺乏量化评估工具的验证。

---

## 423. Physics-driven Sonification for Improving Multisensory Needle Guidance in Percutaneous Epicardial Access

**arXiv ID:** 2604.06911 | [PDF](https://arxiv.org/pdf/2604.06911v1)

**作者:** Veronica Ruozzi `[一作]` (Politecnico di Milano), Emiliano Votta `[通讯]` (Politecnico di Milano)

**通讯引用:** 3773 | [OpenAlex ID](https://openalex.org/A5054773169)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `109c2b71-d051-425c-831f-0c544c24280d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并评估了一种基于XR的多感官导航系统（视觉+物理驱动声学），用于经皮心外膜穿刺（PEA）过程中的针头引导。

**💡 创新点**

首次将基于物理建模的二维膜声学合成与实时动态心脏解剖学结合，实现对针尖与心包膜及心肌距离的音频反馈，从而提升动态目标感知。

**🔧 技术方法**

结合4DCTA动态建模、实时针尖跟踪、物理建模声学合成（Modalys）、OST‑HMD可视化、Open Sound Control与TCP实时数据传输。

**📊 数据集**

使用预先采集的ECG门控4DCTA数据（20帧/心动周期）以及实验用的仿真胸腔体模与CT重建模型。

**📈 对比分析**

在12名心脏科医生的体视实验中，将视觉单模与多感官（视觉+声学）两种模式进行对比，结果多感官模式显著提高成功率（83.3% vs 55%），降低错过目标和心肌接触率，且在需要时可缩短误差并降低NASA‑TLX工作负荷；执行时间相近。

**⚠️ 局限性**

主要限制在于系统对高质量4DCTA重建与实时心脏运动校准的依赖、声学在临床手术室噪声环境下的可行性，以及在真实病例中针尖初始轨迹控制的局限。

---

## 424. Location Is All You Need: Continuous Spatiotemporal Neural Representations of Earth Observation Data

**arXiv ID:** 2604.07092 | [PDF](https://arxiv.org/pdf/2604.07092v1)

**作者:** Mojgan Madadikhaljan `[一作]` (University of Bundeswehr Munich), Michael Schmitt `[通讯]` (University of Bundeswehr Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并预训练一个坐标驱动的隐式神经网络 LIANet，用于重建地球表面的多时相卫星影像，并在不访问原始数据的前提下，将其微调用于多种像素级下游任务。

**💡 创新点**

创新点包括：① 将多分辨率稀疏哈希表与时空编码相结合，得到可连续插值的高分辨率空间表示；② 通过生成式预训练学习“区域特定”基础模型，使得端用户可仅用少量标签和固定坐标即可完成迁移学习；③ 设计了仅需 0.5 M 可训练参数的轻量微调策略，显著降低了对原始影像的依赖。

**🔧 技术方法**

采用了隐式神经表示 (INR) + 多分辨率哈希表编码、ResNet‑UNet 混合 CNN 解码器、L1 重建损失、坐标投影与 bilinear 插值技术，并在微调时冻结编码器与解码器初层，改写最后几层实现任务适配。

**📊 数据集**

主要使用 Sentinel‑2 多光谱影像（GSD 10 m）在德国慕尼黑三块覆盖面积分别为 2,500 km²、5,000 km²、12,000 km² 的区域；并构造了包含土地覆被、冠层高度、叶型分类、建筑占比等标签的自定义数据集；另外对 PANGAEA 的 PASTIS 与 HLS Burn Scars 进行适配验证。

**📈 对比分析**

与从零训练的 UNet、Micro‑UNet 以及三种公开的全球基础模型（TerraMind、Prithvi‑v2‑300、DOFA‑Large）进行对比。微调设置分为全量微调、冻结编码器微调和仅使用最终层嵌入的轻量解码器。实验显示：在 0.5 M 可训练参数下，LIANet 在像素分类、回归和分割任务上均达到或超过三种 GFM 基线的最佳表现，并且在大多数指标上与全量微调模型相当或更优。

**⚠️ 局限性**

局限性主要是：① 仅在预训练时使用的地理区域内有效，无法迁移到未见地区；② 随着预训练区域面积增大，重建质量和下游性能均会下降；③ 目前缺乏对更大尺度（如国家级）或更深时序的系统性评估，需要进一步的规模化与更新机制。

---

## 425. Epistemic Robust Offline Reinforcement Learning

**arXiv ID:** 2604.07072 | [PDF](https://arxiv.org/pdf/2604.07072v1)

**作者:** Abhilash Reddy Chenreddy `[一作]` (HEC Montréal), Erick Delage `[通讯]` (HEC Montréal)

**通讯引用:** 6510 | [OpenAlex ID](https://openalex.org/A5014509810)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Epistemic Robust Soft Actor‑Critic（ERSAC）框架，用不确定性集合代替传统的 Q‑网络集群来进行离线强化学习；

**💡 创新点**

创新点在于：①用可选的盒形、凸包或椭圆集合捕捉 Q 值的集体不确定性；②引入 Epinet（知识不确定性神经网络）实现闭式椭圆不确定性集合，消除对大规模集群的需求；③设计了基于风险敏感行为策略的离线数据生成与评测基准；

**🔧 技术方法**

核心技术包括：Soft Actor‑Critic（SAC）算法；集群 Q‑网络（SAC‑N）；不确定性集合构造（Box、Convex Hull、Ellipsoid）；Epinet 采样网络；鲁棒 Bellman 目标与策略梯度；动态期望值风险测度；

**📊 数据集**

使用了多种数据集：离线 tabular 环境（Machine Replacement、RiverSwim）；经典控制环境（CartPole、LunarLander）；以及 Atari 2600 环境（来自 Minari 的离线数据集）；

**📈 对比分析**

通过对比 SAC‑N、CH‑N、Ell‑N、Ell‑Epi 以及 Ell‑Epi* 等版本，评估指标为归一化回报。实验显示，凸包和椭圆集合尤其在数据稀缺或行为策略偏差较大时优于盒形集合；Ell‑Epi* 在 Atari 任务中与 CQL、IQL 相当或更优，说明 Epinet 方案既高效又效果好；

**⚠️ 局限性**

局限性：1）对 Epinet 的线性噪声假设在高度非高斯场景下可能不足；2）椭圆集合仍需估计协方差，计算成本不如简单盒形；3）实验主要集中在离线数据生成与评测基准，缺乏理论收敛或样本复杂度分析；4）在极端稀疏或完全偏差的数据下，鲁棒性提升有限。

---

## 426. BioMoTouch: Touch-Based Behavioral Authentication via Biometric-Motion Interaction Modeling

**arXiv ID:** 2604.07071 | [PDF](https://arxiv.org/pdf/2604.07071v1)

**作者:** Zijian Ling `[一作]` (Huazhong University of Science and Technology), Qian Wang `[通讯]` (Wuhan University)

**通讯引用:** 57693 | [OpenAlex ID](https://openalex.org/A5100422786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种名为 BioMoTouch 的多模态触控行为认证框架，能够在移动设备上无需额外硬件、隐式地捕获触摸时的生理（指纹形态）和行为（运动动态）特征，并通过联合建模实现身份验证。

**💡 创新点**

创新点包括：①首次证明普通电容屏能够直接获取用户指尖形态的生理特征；②将生理与行为维度的交互显式建模，而非传统的独立决策或仅关注行为；③设计轻量级的时间对齐与特征增强策略，实现无缝对齐并提升鲁棒性；④在真实环境下针对模拟复制、模仿与傀儡攻击三种主流威胁进行了系统评估。

**🔧 技术方法**

核心技术包括：电容触控信号的自适应阈值检测与跟踪、IMU 运动估计与四元数姿态恢复、时间频率变换与 STFT 语谱特征、基于 TinyViT 的多模态特征提取、轻量级 MLP 融合模块、以及 OC‑SVM/LOF/IF 等一类分类器。

**📊 数据集**

使用 38 名受试者的数据构建了 10 组实验数据集，涵盖：基准训练集、PIN 与指纹辅助认证、模仿攻击、人工复制攻击、傀儡攻击、长期跟踪、不同手指、不同姿态、手指湿度、屏幕保护膜等多种真实场景。

**📈 对比分析**

与市售产品（ZKTeco Live20R）及多篇学术基线相比，BioMoTouch 在默认设置下实现了 99.71% 的平衡准确率、0.27% 的 EER，并在所有攻击场景下的误接受率均低于 0.90%。在辅助认证中，EER 分别为 0.27%（指纹）和 0.19%（PIN），在长期、姿态、手指、湿度和保护膜等变异条件下均保持较低误差。

**⚠️ 局限性**

局限性包括：对电容屏硬件的依赖导致在极低质量或老旧设备上的性能下降；对多模态特征融合的参数需要在不同设备上重新校准；实验样本量相对有限，需在更大规模、跨平台的真实用户中进一步验证；以及对高频交互（如快速滑动或长按）在模型中的适配尚未深入探讨。

---

## 427. Is Cross-Lingual Transfer in Bilingual Models Human-Like? A Study with Overlapping Word Forms in Dutch and English

**arXiv ID:** 2604.07067 | [PDF](https://arxiv.org/pdf/2604.07067v1)

**作者:** Iza Škrjanec `[一作]` (Saarland University), Stefan L. Frank `[通讯]` (Radboud University)

**通讯引用:** 11718 | [OpenAlex ID](https://openalex.org/A5076704114)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并比较四种不同词汇共享策略下的双语（荷兰语-英语）Transformer语言模型，以模拟双语阅读中的交叉语言激活。

**💡 创新点**

通过系统性控制词汇表中的词汇共享与分离，首次验证仅共享词义相同（友谊词）时模型能重现人类双语者的认知表现。

**🔧 技术方法**

使用GPT‑2小型Transformer架构、字节级BPE分词器，并通过混合效应回归与余弦相似度、词语惊奇度（surprisal）评估模型。

**📊 数据集**

训练语料覆盖400M词汇，约75%为荷兰语、25%为英语，来源包括维基百科、脚本化语音、网络爬取；评估使用两套双语心理语言学刺激（友谊词与假朋友）。

**📈 对比分析**

与人类实验结果对比，发现“仅共享友谊词”条件下模型展示对友谊词的低惊奇度，匹配双语者的促进效应；其他条件不一致。模型在该条件下的相对性能优于完全共享或完全分离词汇策略。

**⚠️ 局限性**

仅针对词汇共享的细粒度控制，未探讨句法层面或更大词表；缺乏更多双语人类数据，导致评估范围有限；使用的Transformer架构较浅，可能影响跨语言激活的普适性。

---

## 428. IndoBERT-Sentiment: Context-Conditioned Sentiment Classification for Indonesian Text

**arXiv ID:** 2604.07057 | [PDF](https://arxiv.org/pdf/2604.07057v1)

**作者:** Muhammad Apriandito Arya Saputra `[一作]` (SocialX), Hanif Fakhrurroja `[通讯]` (National Research and Innovation Agency)

**通讯引用:** 738 | [OpenAlex ID](https://openalex.org/A5004438607)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于上下文条件的印尼语情感分类模型 IndoBERT‑Sentiment，能够同时考虑话题背景和文本内容进行情感判定。

**💡 创新点**

创新点在于将已用于相关性判断的上下文条件架构迁移到情感分析任务，并在同一数据集上重新标注情感标签，实现显著性能提升。

**🔧 技术方法**

技术上采用 IndoBERT Large 作为编码器，并在其上加分类头，使用上下文+文本拼接的输入格式，训练时采用类别权重和早停策略。

**📊 数据集**

使用的是先前构建的 31,360 条目上下文‑文本对数据集，覆盖 188 个主题，重新使用 GPT‑4o‑mini 对情感进行三类标注。

**📈 对比分析**

与 HuggingFace 上最受欢迎的三款通用情感模型（BERT、RoBERTa、IndoBERT Base）在同一 4,704 条目测试集上对比，IndoBERT‑Sentiment 的 F1 macro 达 0.856、准确率 88.1%，比最佳基线高 35.6 F1 点，正面情感 F1 由 ≤0.211 提升至 0.791。

**⚠️ 局限性**

主要限制包括：模型需要在推理时提供话题上下文；测试集标签本身依赖上下文，可能偏向上下文条件模型；且正面样本在数据中占比仅 11.8%，对极少数类的绝对表现仍受影响。

---

## 429. AdaBoost Does Not Always Cycle: A Computer-Assisted Counterexample

**arXiv ID:** 2604.07055 | [PDF](https://arxiv.org/pdf/2604.07055v1)

**作者:** Erik Y. Wang `[一作]` `[通讯]` (Stanford University), Erik Y. Wang (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文构造了一个有限{-1,+1}矩阵，并演示在标准完整AdaBoost算法下，起始于均匀分布时，该算法的迭代轨迹不收敛到任何有限周期。

**💡 创新点**

创新点在于提出了基于区块乘积结构的两块“小装置”(gadget)的对称动力学，利用其共享的周期2轨道与返回映射的主特征值比值为无理数，从而证明了AdaBoost的周期收敛性假设不成立。

**🔧 技术方法**

所用技术包括：符号代数（求多项式极小多项式、求根与不可约性检验）、精确有理数运算、区间算术和严格的误差界定、以及代数数论中的域范数和可除性论证。

**📊 数据集**

本文未使用外部机器学习数据集，而是完全基于构造的符号矩阵与理论证明。

**📈 对比分析**

由于论文的目标是证明数学命题，未涉及实验比较；通过精确的符号计算与区间验证，证明了所构造矩阵的AdaBoost轨迹无周期，理论上展示了算法在该实例下的非周期行为。

**⚠️ 局限性**

局限在于结论仅适用于特定构造的矩阵，无法直接推广到所有AdaBoost实例；此外，证明高度依赖符号计算与区间算术，缺乏通用的可扩展分析框架。

---

## 430. Sell More, Play Less: Benchmarking LLM Realistic Selling Skill

**arXiv ID:** 2604.07054 | [PDF](https://arxiv.org/pdf/2604.07054v1)

**作者:** Xuanbo Su `[一作]` (brgroup), Leo Huang `[通讯]` (brgroup)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了双语（中英）销售对话基准SalesLLM，并训练专用用户模拟器CustomerLM，用于评估LLM在多轮销售对话中的推销效果。

**💡 创新点**

创新点包括：① 通过30,074个脚本和1,805个难度校准的多轮情景，实现对销售过程进展和成交意图的自动评估；② 训练CustomerLM显著降低角色倒置率；③ 采用双指标评估框架，结合LLM评分与BERT购买意图分类器。

**🔧 技术方法**

使用了大语言模型（如GPT‑4o、Gemini‑3、Qwen‑2.5‑72B等）进行评估和对话生成；对CustomerLM采用SFT和DPO训练；购买意图用中英BERT模型微调；评估公式融合两项得分。

**📊 数据集**

数据集包括8,000条人类标注的销售对话用于CustomerLM训练，19,178条中英标注对话用于BERT购买意图分类器训练，以及由30,074个脚本生成的1,805个多轮情景构成的SalesLLM基准集。

**📈 对比分析**

在15款主流LLM（中英）上与人类销售人员基准对比，自动评分与人工评分相关性达Pearson 0.98；顶尖模型在中文场景超过人类基线，英文场景仍有差距；在多产品和长期跟进场景中表现不同，凸显模型在跨语言和复杂策略上的差异。

**⚠️ 局限性**

局限性包括：① 人类基准水平低于专家级；② 用户模拟器仍无法完整捕捉真实顾客的情绪、文化与决策复杂性；③ 只关注单轮会话，未覆盖多轮长期销售周期；④ 模型可能出现未经授权的让步或承诺，导致评估失真。

---

## 431. A-MBER: Affective Memory Benchmark for Emotion Recognition

**arXiv ID:** 2604.07017 | [PDF](https://arxiv.org/pdf/2604.07017v1)

**作者:** Deliang Wen `[一作]`, Yu Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了A-MBER（Affective Memory Benchmark for Emotion Recognition）用于评估AI助手在多会话历史中解释用户当前情绪的能力。

**💡 创新点**

创新点在于：①把情绪识别与长期记忆融合，形成新的“情感记忆”评测目标；②采用分阶段生成和显式中间表示的结构化构造框架；③设计了三大任务族（判断、检索、解释）和多维度难度标签（记忆层级、推理结构、鲁棒性）。

**🔧 技术方法**

技术主要包括：多阶段生成管线、情感与语音描述的结构化表示、检索式记忆访问、结构化记忆系统（如Red Bear AI Memory）以及多维度评测脚本。

**📊 数据集**

使用了合成的多会话对话数据，构建在教师–学生/咨询场景下，包含文本与结构化交付描述。

**📈 对比分析**

通过与五种配置（无记忆、长上下文、检索记忆、结构化记忆、金证据）在判别、检索、解释三类任务上对比，发现记忆层级越高、推理越复杂时性能提升最显著，结构化记忆在大多数子任务上超越长上下文与检索记忆，金证据条件下仍未达到满分。

**⚠️ 局限性**

局限性包括：数据为合成对话，缺乏真实交互噪声；仅覆盖教师–学生场景，通用性待验证；解释任务依赖人工评判；鲁棒性测试范围有限；长时间跨会话一致性与角色自然性的进一步提升空间。

---

## 432. KITE: Keyframe-Indexed Tokenized Evidence for VLM-Based Robot Failure Analysis

**arXiv ID:** 2604.07034 | [PDF](https://arxiv.org/pdf/2604.07034v1)

**作者:** Mehdi Hosseinzadeh `[一作]` (Adelaide University), Feras Dayoub `[通讯]` (Adelaide University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 KITE 前端，将长机器人执行视频压缩成关键帧、伪 BEV 以及序列化的机器人与场景信息，直接供通用 VLM 进行故障检测、定位、解释与纠正。

**💡 创新点**

创新点在于：①训练无关的关键帧索引与布局支撑机制；②使用伪 BEV 以可读的拓扑方式呈现空间关系；③将多模态信息（检测、深度、交互标记、场景图）整合为统一的提示，极大提升通用 VLM 的推理可访问性。

**🔧 技术方法**

核心技术包括：密集光流关键帧选取、GroundingDINO 开词汇检测、Depth‑Anything 单目相对深度、粗粒度交互标记、3D 场景图构建、伪 BEV 绘制、上下文序列化提示，配合 Qwen2.5‑VL 进行推理。

**📊 数据集**

实验数据集主要使用 RoboFAC 失败分析基准（模拟与真实），并在 RealMan DART 与 ALOHA‑2 双臂真实序列上做定性验证。

**📈 对比分析**

与原始 Qwen2.5‑VL、Gemini‑2.0、GPT‑4o 以及 RoboFAC 微调模型对比，KITE+VLM 在 MCQ 任务上实现 FD+36、FI+18、FL+33 的显著提升；在自由文本任务上 ROUGE‑L 与 Sentence‑BERT 相似度均提升，QLoRA 微调后进一步逼近甚至超越微调基线。

**⚠️ 局限性**

局限性包括：仅使用相对深度与粗粒度交互标记，伪 BEV 非度量且忽略垂直结构；关键帧选取可能错过低运动或极短暂故障；场景图关系有限；评估主要集中于 RoboFAC，缺乏更广泛的跨基准与人类用户质量评估。

---

## 433. Leveraging Artist Catalogs for Cold-Start Music Recommendation

**arXiv ID:** 2604.07090 | [PDF](https://arxiv.org/pdf/2604.07090v1)

**作者:** Yan-Martin Tamm `[一作]` (University of Tartu), Anna Aljanaki `[通讯]` (University of Tartu)

**通讯引用:** 532 | [OpenAlex ID](https://openalex.org/A5039285905)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出将冷启动音乐推荐视为半冷问题，利用已有艺术家曲目协同信号生成新曲目的CF嵌入并实现新艺术家发现。

**💡 创新点**

创新点在于设计Artist Catalog Attention（ACARec）架构，通过自注意力与交叉注意力聚合艺术家已有曲目，并用GRU融合得到精准的冷启动嵌入。

**🔧 技术方法**

使用预训练音频编码器、协同过滤模型、Multi‑head注意力、自注意力、交叉注意力以及GRU学习融合；训练目标为对协同过滤嵌入的重构。

**📊 数据集**

实验数据集为Music4All‑Onion（M4A‑Onion）和Yambda‑50m两大音乐推荐数据集，包含音频特征、艺术家信息及交互日志。

**📈 对比分析**

与多种内容冷启动基线（DeepMusic、Heater、GAR、VBPR等）及其艺术家扩展版本和艺术家平均启发式做对比；ACARec 在 Overall、Discovery、Exploit 上 Recall@20 和 NDCG@20 均优于最强基线，尤其在新艺术家发现（Discovery）上提升约3‑10%，显著提升推荐效果。

**⚠️ 局限性**

局限性：只能处理已有曲目的热艺术家，对新艺术家无效；假设每首曲目只有单一艺术家，忽略多艺术家合作；依赖预训练的协同过滤模型；当艺术家目录庞大时推理成本较高。

---

## 434. MoE Routing Testbed: Studying Expert Specialization and Routing Behavior at Small Scale

**arXiv ID:** 2604.07030 | [PDF](https://arxiv.org/pdf/2604.07030v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 435. DTCRS: Dynamic Tree Construction for Recursive Summarization

**arXiv ID:** 2604.07012 | [PDF](https://arxiv.org/pdf/2604.07012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 436. ModuSeg: Decoupling Object Discovery and Semantic Retrieval for Training-Free Weakly Supervised Segmentation

**arXiv ID:** 2604.07021 | [PDF](https://arxiv.org/pdf/2604.07021v1)

**作者:** Qingze He `[一作]` (South China University of Technology), Quan Tang `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ModuSeg框架，采用训练‑free的弱监督语义分割方法，先通过通用掩码生成器获得几何候选框，再利用离线特征库进行语义检索；

**💡 创新点**

核心创新在于将物体发现与语义检索完全解耦，配合语义边界净化与软掩码特征聚合，构建高质量原型库，无需任何参数训练；

**🔧 技术方法**

使用类无关掩码提议器（EntitySeg）、视觉基础模型（C‑RADIOv4/DINOv2/3/CLIP）构建特征库，加入语义边界净化、软掩码特征聚合、Top‑K检索、层次投票、置信度优先光栅化及NMS；

**📊 数据集**

在PASCAL VOC 2012与MS COCO 2014两个标准弱监督分割基准上进行实验；

**📈 对比分析**

与现有单/多阶段方法（ToCo、DuPL、WeCLIP、ExCEL、SSR等）进行对比，ModuSeg在VOC验证/测试集分别达到86.3%/86.6% mIoU，在COCO上获得56.7% mIoU，超越前沿方法6–7个百分点；

**⚠️ 局限性**

主要瓶颈在于类无关掩码生成器的语义指导不足，导致过度分割；当使用GT掩码时性能提升显著，表明掩码生成器仍是性能限制点；

---

## 437. Mining Electronic Health Records to Investigate Effectiveness of Ensemble Deep Clustering

**arXiv ID:** 2604.07085 | [PDF](https://arxiv.org/pdf/2604.07085v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 438. ConceptTracer: Interactive Analysis of Concept Saliency and Selectivity in Neural Representations

**arXiv ID:** 2604.07019 | [PDF](https://arxiv.org/pdf/2604.07019v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 439. Predictive Representations for Skill Transfer in Reinforcement Learning

**arXiv ID:** 2604.07016 | [PDF](https://arxiv.org/pdf/2604.07016v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 440. Corpora deduplication or duplication in Natural Language Processing of few resourced languages ? A case of study: The Mexico's Nahuatl

**arXiv ID:** 2604.07015 | [PDF](https://arxiv.org/pdf/2604.07015v1)

**作者:** Juan-José Guzman-Landa `[一作]`, Luis-Gil Moreno-Jiménez `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对墨西哥纳瓦特尔语π-yalli语料库进行可控复制扩增，并评估其对静态词向量学习的影响。

**💡 创新点**

首次在低资源、形态极为聚合的语言上提出并验证增量复制技术，可在不依赖词典或POS的前提下提升嵌入质量。

**🔧 技术方法**

使用FastText、Word2Vec（CBow与Skipgram）与Glove三种静态词嵌入模型，配合Kendall’s τ作为语义相似度评估指标。

**📊 数据集**

核心数据集为π-yalli语料库（约660万词），通过复制比例1×至30×扩增，另外对比了Common Crawl和维基百科预训练向量。

**📈 对比分析**

与未复制的原始语料以及预训练向量比较，FastText Skipgram在10×复制时最高提升8%（τ从0.459到0.495），Word2Vec Skipgram在22×复制时提升35%（τ从0.357到0.483），Glove则表现不佳。

**⚠️ 局限性**

复制方法对语料的真实性有损失，Glove模型提升有限；实验仅聚焦语义相似度任务，未检验对其他NLP任务的普适性。

---

## 441. CAFP: A Post-Processing Framework for Group Fairness via Counterfactual Model Averaging

**arXiv ID:** 2604.07009 | [PDF](https://arxiv.org/pdf/2604.07009v1)

**作者:** Irina Arévalo `[一作]` (Universidad Politecnica de Madrid), Marcos Oliva `[通讯]` (Bain&Company)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种后处理方法CAFP，通过对每个样本的敏感属性做反事实取值并取平均来消除模型对受保护属性的直接影响，从而提升组公平性。

**💡 创新点**

创新点在于：①仅需两次模型查询即可实现反事实平均，完全无须重新训练或访问模型内部；②在理论上给出对公平性（人群均衡性、均等机会）和准确性折衷的明确上界；③提供对互信息、误差失真等度量的上界证明。

**🔧 技术方法**

主要技术包括：反事实平均（counterfactual averaging）、互信息分析、误差失真与公平性指标（DP、EO）的定量上界推导，以及对黑箱分类器的简单调用。

**📊 数据集**

使用三个公开基准数据集：Adult Income、COMPAS、German Credit。

**📈 对比分析**

将CAFP与基线模型、Equalized Odds、Reject Option等常见后处理方法进行对比。实验显示，CAFP在保持准确率（误差低于0.5个百分点）同时显著降低DP（最高可减至38%）和AOD（最高可减至45%），且在不同阈值下公平性-准确性曲线最为平稳。

**⚠️ 局限性**

局限性包括：仅适用于二值敏感属性，无法直接处理多分类或交叉组；在特征与敏感属性高度相关时，仍存在间接偏差；需要模型支持在推理时可模拟对敏感属性的取值变化。

---

## 442. ReDAct: Uncertainty-Aware Deferral for LLM Agents

**arXiv ID:** 2604.07036 | [PDF](https://arxiv.org/pdf/2604.07036v1)

**作者:** Dzianis Piatrashyn `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Maxim Panov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1332 | [OpenAlex ID](https://openalex.org/A5058551285)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为ReDAct的框架，利用小型LLM作为默认决策者，并在小模型的不确定性超过阈值时，将决策权暂缓给大型LLM，从而在序列化的决策任务中实现成本与性能的平衡。

**💡 创新点**

创新点在于：①把信息论不确定性量化应用于实时动作决策中的推迟机制；②仅在需要时调用昂贵的大模型，实现成本显著下降；③证明动作级不确定性（如困惑度、序列概率、平均词熵）是最有效的推迟信号；④在文本化嵌入环境中首次验证该策略。

**🔧 技术方法**

核心技术包括：两模型交互（小模型+大模型）、ReAct框架、基于token概率的Uncertainty Quantification（MTE、SP、PPL）、阈值校准与自适应推迟决策、成本感知模型路由。

**📊 数据集**

使用的实验数据集为文本化的ALFWorld和MiniGrid环境，分别评估了任务完成率和推迟次数。

**📈 对比分析**

与仅使用大模型、仅使用小模型、随机推迟等基线相比，ReDAct在ALFWorld中仅需约15%大模型调用即可达到与大模型相同甚至更高的成功率；在MiniGrid中同样表现优越；整体推迟次数与成本均显著低于全大模型方案。

**⚠️ 局限性**

局限性包括：①依赖token级概率的UQ方法，若API不提供logprobs则无法使用；②仅针对70B+参数模型，未探究更小模型的适配；③实验仅覆盖文本化嵌入环境，尚未验证多模态或视觉+文本场景。

---

## 443. Continuous Interpretive Steering for Scalar Diversity

**arXiv ID:** 2604.07006 | [PDF](https://arxiv.org/pdf/2604.07006v1)

**作者:** Ye-eun Cho `[一作]` (Sungkyunkwan University), Ye-eun Cho `[通讯]` (Sungkyunkwan University)

**通讯引用:** 317 | [OpenAlex ID](https://openalex.org/A5064099640)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种新的方法，称为连续解释引导（CIS），用于探测大语言模型（LLMs）中的分级语用解释，特别关注标量多样性。

**💡 创新点**

创新点在于将激活水平引导强度视为一个连续的实验变量，从而能够系统地评估模型对分级语用解释的敏感性，并引入了新的数据集GraSD来支持这一分析。

**🔧 技术方法**

使用了激活水平引导技术，通过在推理时直接干预模型的内部激活来探测语用解释的变化。

**📊 数据集**

使用了新构建的数据集GraSD，该数据集编码了不同标量项对语用解释的分级影响，共包含121对弱-强标量项。

**📈 对比分析**

与统一激活引导相比，分级激活引导能够保持项目级的敏感性，导致更具差异化的解释变化，且在所有模型中均显示出统计显著性。

**⚠️ 局限性**

本研究的局限性包括仅关注标量含义，未探讨该方法在其他语用现象中的适用性，以及基于有限实例估计的固定引导方向可能无法捕捉到语用表示的全部变异性。

---

## 444. MARS: Enabling Autoregressive Models Multi-Token Generation

**arXiv ID:** 2604.07023 | [PDF](https://arxiv.org/pdf/2604.07023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 445. EmoMAS: Emotion-Aware Multi-Agent System for High-Stakes Edge-Deployable Negotiation with Bayesian Orchestration

**arXiv ID:** 2604.07003 | [PDF](https://arxiv.org/pdf/2604.07003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 446. Are Stochastic Multi-objective Bandits Harder than Single-objective Bandits?

**arXiv ID:** 2604.07096 | [PDF](https://arxiv.org/pdf/2604.07096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 447. Multilingual Embedding Probes Fail to Generalize Across Learner Corpora

**arXiv ID:** 2604.07095 | [PDF](https://arxiv.org/pdf/2604.07095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 448. Flow Motion Policy: Manipulator Motion Planning with Flow Matching Models

**arXiv ID:** 2604.07084 | [PDF](https://arxiv.org/pdf/2604.07084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 449. PRISM: Rethinking Scattered Atmosphere Reconstruction as a Unified Understanding and Generation Model for Real-world Dehazing

**arXiv ID:** 2604.07048 | [PDF](https://arxiv.org/pdf/2604.07048v1)

**作者:** Chengyu Fang `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 11001 | [OpenAlex ID](https://openalex.org/A5100754504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 PRISM 框架，联合估计清晰图像与散射变量，实现单图像真实世界去雾；

**💡 创新点**

将物理散射模型与优化未折叠网络相结合，设计 Proximal Scattered Atmosphere Reconstruction (PSAR) 以及在线非均匀雾合成与 Selective Self‑Distillation Adaptation (SSDA)，从物理、数据和自适应角度共同突破现有方法的瓶颈；

**🔧 技术方法**

利用基于物理的散射模型、折叠前向推导的近端更新、轻量化 UNet 细化模块、Mean‑Teacher 伪标签、NR‑IQA 指标筛选的 QGSD 以及 PGSR 物理审计机制；

**📊 数据集**

使用 RIDCP（合成）与 RESIDE‑URHI（真实）数据进行预训练，随后在 RTTS、Fattal’s 等真实测试集上评估；

**📈 对比分析**

在 RTTS 基准上以 FADE 0.470、PAQ2PIQ 74.05 等指标领跑所有公开方法；在 Fattal’s 数据集同样保持最优或近优的性能，证明了跨域泛化能力；

**⚠️ 局限性**

对模型深度和训练步骤依赖度较高，折叠迭代次数与性能关系紧密；在极端高光或深色阴影场景中仍可能出现轻微色偏或纹理失真；

---

## 450. Assessing REST API Test Generation Strategies with Log Coverage

**arXiv ID:** 2604.07073 | [PDF](https://arxiv.org/pdf/2604.07073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 451. SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA)

**arXiv ID:** 2604.07066 | [PDF](https://arxiv.org/pdf/2604.07066v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 452. AEROS: A Single-Agent Operating Architecture with Embodied Capability Modules

**arXiv ID:** 2604.07039 | [PDF](https://arxiv.org/pdf/2604.07039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 453. Learning to Query History: Nonstationary Classification via Learned Retrieval

**arXiv ID:** 2604.07027 | [PDF](https://arxiv.org/pdf/2604.07027v1)

**作者:** Jimmy Gammell `[一作]` (Purdue University), Deepayan Chakrabarti `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个检索增强的非平稳分类框架，利用历史标记样本对当前输入进行条件化预测。

**💡 创新点**

首次将检索增强学习应用于非平稳分类场景，使用端到端可学习的离散检索机制，使模型在不更新权重的情况下适应分布漂移。

**🔧 技术方法**

结合离散查询生成器、基于注意力的检索、score‑based gradient estimator、Perceiver 结构以及冻结的低维键生成函数。

**📊 数据集**

采用控制实验的旋转决策边界与针叶在阑尾实验，以及亚马逊2023年电子产品评论语料（1996–2023）进行评估。

**📈 对比分析**

与标准无检索分类器对比，使用迁移前后准确率和 VRAM 占用进行评估；检索模型在分布漂移测试中显著提升准确率，且 VRAM 随历史长度按线性规律增长。

**⚠️ 局限性**

需要手动调参以降低 score‑based 估计器的高方差；键生成函数固定限制检索表达；对离群数据的性能仍下降，检索引入的计算与内存开销需权衡。

---

## 454. MARVEL: Multimodal Adaptive Reasoning-intensiVe Expand-rerank and retrievaL

**arXiv ID:** 2604.07079 | [PDF](https://arxiv.org/pdf/2604.07079v1)

**作者:** Mahmoud SalahEldin Kasem `[一作]` (Chungbuk National University), Hyun-Soo Kang `[通讯]` (Chungbuk National University)

**通讯引用:** 693 | [OpenAlex ID](https://openalex.org/A5018214137)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MARVEL，一套三阶段的 multimodal-to-text 检索流水线：先用 GPT-4o 对查询图像进行字幕生成并与文本合并，随后通过 LLM 进行查询扩展生成富含语义的检索查询，使用专门为推理式检索微调的 MARVEL-Retriever 进行稠密检索，再通过 GPT-4o 的链式推理重新排序（可选多轮 RRF fusion），最终得到高精度的检索结果。

**💡 创新点**

创新点在于将查询扩展、推理式检索和链式推理重新排序三大关键能力紧密耦合到一个统一的流水线中，而不是各自孤立；通过 LLM 对视觉信息进行自然语言表述并在检索前显式扩展查询语义；以及在检索后加入多轮链式推理与 RRF fusion，以提升推理一致性。

**🔧 技术方法**

核心技术包括：GPT-4o（视觉字幕、查询扩展、链式推理重新排序）；基于 bi‑encoder 的 MARVEL‑Retriever（使用对比学习和硬负样本微调）；递归互惠排名（RRF）融合；对 MM‑BRIGHT 领域特定的推理式检索训练；以及传统的 BM25 等基准检索器。

**📊 数据集**

使用 MM‑BRIGHT 语义推理式 multimodal‑to‑text 基准（2803 条多模态查询，涵盖 29 个技术领域）进行实验和评估，指标为 nDCG@10。

**📈 对比分析**

与 CLIP、SigLIP、Jina‑CLIP、Nomic‑Vision、BGE‑VL、GME‑Qwen2‑VL‑2B/7B 等现有多模态检索模型，以及 BM25 等基准模型对比。MARVEL 在 MM‑BRIGHT 上平均 nDCG@10 达到 37.9，超过最强基准 Nomic‑Vision（27.6）10.3 分，且在 27/29 领域取得领先；在所有基线上实现了显著提升，单组件消融实验表明查询扩展和链式推理是主要贡献者。

**⚠️ 局限性**

局限性包括：在极其专业的领域（如 Crypto、Quantum Computing）由于术语稀缺，扩展和重排序效果受限；整个流水线高度依赖 GPT‑4o，导致推理成本和延迟较高；MARVEL‑Retriever 与查询扩展的协同效果需要在更大规模数据上进一步验证；当前仅支持单图查询，尚未扩展到多图或视频检索。

---

## 455. AnchorSplat: Feed-Forward 3D Gaussian SplattingWith 3D Geometric Priors

**arXiv ID:** 2604.07053 | [PDF](https://arxiv.org/pdf/2604.07053v1)

**作者:** Xiaoxue Zhang `[一作]` (Huawei Technologies Ltd.), Dave Zhenyu Chen `[通讯]` (Huawei Technologies Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了 AnchorSplat，一种基于 anchor 对齐的 feed‑forward 3D Gaussian Splatting 框架，用于场景级三维重建与新视角合成。

**💡 创新点**

创新点包括：① 引入 anchor‑aligned Gaussian 表示，直接利用几何先验生成稀疏且与视图无关的 Gaussians；② 插件式 Gaussian Refiner，单独改进属性以提升渲染质量；③ 在保持高质量的同时显著减少 Gaussians 数量和重建时间。

**🔧 技术方法**

使用技术包括：预训练的多视图立体模块（如 MapAnything）生成深度与相机位姿做 anchor；CNN+U‑Net 提取多视图特征；Transformer‑based Gaussian Decoder 与 MLP 预测 Gaussians；Gaussian Refiner 基于 ResNet‑18 提取渲染误差，Point Transformer 更新属性；采用 Flash Attention、bfloat16 训练。

**📊 数据集**

实验数据集为 ScanNet++ V2 Benchmark。

**📈 对比分析**

与优化式 3DGS、Mip‑Splatting 以及 voxel‑aligned AnySplat 进行对比，AnchorSplat 在 PSNR/SSIM/LPIPS、AbsRel/δ1 等指标上与 AnySplat 相当或更优，且 Gaussians 数量约为 AnySplat 的 1/20，重建时间显著降低。

**⚠️ 局限性**

局限性在于依赖准确的几何先验，若先验不完整或有缺失，难以覆盖空洞区域，导致重建质量下降；未来需探索自适应密度控制和动态 Gaussian 增长机制。

---

## 456. AV-SQL: Decomposing Complex Text-to-SQL Queries with Agentic Views

**arXiv ID:** 2604.07041 | [PDF](https://arxiv.org/pdf/2604.07041v1)

**作者:** Minh Tam Pham `[一作]` (Griffith University), Thanh Tam Nguyen `[通讯]` (Griffith University)

**通讯引用:** 37134 | [OpenAlex ID](https://openalex.org/A5101884287)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种三阶段多代理框架 AV‑SQL，先重写问题，按块生成可执行的 CTE 视图并过滤无关模式，再用规划、生成与修订代理合成最终可执行 SQL。

**💡 创新点**

创新点：① 采用可执行 CTE 作为中间代理视图，使模式过滤可验证；② 通过逐块处理大规模模式并执行迭代修正，降低上下文窗口压力；③ 多代理协同（重写、视图生成、规划、生成、修订）并结合执行反馈提升鲁棒性。

**🔧 技术方法**

技术：大语言模型（Gemini‑3‑Pro、GPT‑5‑Mini、Llama‑3.3‑70B、Qwen2.5‑32B）；schema 压缩与分块；CTE 生成与执行验证；迭代修复；JSON 选表列一致性检查；多代理推理。

**📊 数据集**

数据集：Spider、BIRD、KaggleDBQA、Spider 2.0‑Snow（Snowflake 语法）。

**📈 对比分析**

与多种基线（CodeS、DIN‑SQL、CHESS、Alpha‑SQL、Spider‑Agent、AutoLink 等）比较，Spider 2.0‑Snow 上实现 70.38% 执行准确率（最高），Spider 85.59%，BIRD 72.16%，KaggleDBQA 63.78%；在标准数据集上接近或优于多候选方案，且单候选生成，推理成本较低。

**⚠️ 局限性**

限制：对复杂过滤与聚合逻辑仍易出错；多代理中 view‑gen 占用大量 token 与时间；在极大模式下仍受 LLM 上下文长度限制；混合不同 LLM backbone 时效果不稳定。

---

## 457. Controller Design for Structured State-space Models via Contraction Theory

**arXiv ID:** 2604.07069 | [PDF](https://arxiv.org/pdf/2604.07069v1)

**作者:** Muhammad Zakwan `[一作]` (Inspire AG), Giancarlo Ferrari-Trecate `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 9597 | [OpenAlex ID](https://openalex.org/A5074145693)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用Structured State‑Space Model（SSM）对非线性系统进行间接数据驱动的系统辨识与控制器设计，随后基于收缩理论和LMIs构造指数稳定的状态反馈与观测器，最后在非线性DC电机仿真中验证其可行性。

**💡 创新点**

（1）首次给出SSM可控可观的充分条件；（2）基于收缩理论提出可求解的LMI框架，实现对任意bi‑Lipschitz非线性的稳健状态反馈与观测器；（3）证明该类SSM满足离散时间分离原理。

**🔧 技术方法**

Structured State‑Space Model (SSM)、Linear Recurrent Unit (LRU)、bi‑Lipschitz神经网络架构、收缩理论与离散时间控制收缩度量、线性矩阵不等式（LMI）优化、离散时间观测器设计、数值仿真。

**📊 数据集**

从仿真得到的非线性DC电机模型中，使用PRBS激励收集输入输出数据（含零均值高斯噪声），作为SysId与控制器验证的数据集。

**📈 对比分析**

通过在10个不同初始条件下将设计好的控制器与观测器应用于原非线性电机模型，展示了对各初始状态的指数稳定收敛；未与其他方法直接比较，但结果显示控制器在噪声干扰下保持鲁棒性与稳定。

**⚠️ 局限性**

（1）假设输入/输出非线性均为bi‑Lipschitz，需手工保证或后验检验；（2）仅考虑单层SSM，无法直接推广到多层或更大规模系统；（3）验证仅在仿真环境，缺乏实验验证；（4）对模型识别误差的鲁棒性分析不充分。

---

## 458. Top-P Sensor Selection for Target Localization

**arXiv ID:** 2604.07020 | [PDF](https://arxiv.org/pdf/2604.07020v1)

**作者:** Kaan Buyukkalayci `[一作]` (University of California, Los Angeles), Christina Fragouli `[通讯]` (University of California, Los Angeles)

**通讯引用:** 7384 | [OpenAlex ID](https://openalex.org/A5056591688)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于top‑p列表的传感器选择方法，用低成本传感数据预测目标最接近的若干传感器节点，并在单目标和多目标情境下进行实验验证。

**💡 创新点**

创新点在于：①将传统单一最佳传感器的决策改为包含前p名的列表，形成基于距离的错误概率分析；②提出利用几何结构的贝叶斯估计与分段线性样条建模，显著提升列表构造精度；③扩展到多目标并通过同步局部网格降低计算复杂度。

**🔧 技术方法**

主要技术包括：高斯噪声下的最大值选择、top‑p错误概率的多维正态CDF解析、贝叶斯后验推断、线性样条拟合、以及多目标同步局部搜索。

**📊 数据集**

实验使用户外10,000 m²场地收集的声学数据：10个Raspberry Pi麦克风节点与一至两辆ATV（配GPS）在约3 m/s速度下行驶，采样间隔200 ms。

**📈 对比分析**

与传统归一化最大值选择基线比较，所提算法在相同或更小的输出集大小下取得更高的top‑p包含准确率；单目标实验中准确率随输出集增大单调提升，且在p>1时差距更明显；多目标实验显示同步间隔越大准确率越低。

**⚠️ 局限性**

局限性包括：①假设目标在每个时间块内保持不动，需对快速移动目标做进一步研究；②贝叶斯后验计算对网格细化敏感，网格尺寸与计算成本平衡需更系统化；③实验仅基于声学信号，未验证在其他测量模态（如RF、光学）下的泛化能力。

---

## 459. Gemma 4, Phi-4, and Qwen3: Accuracy-Efficiency Tradeoffs in Dense and MoE Reasoning Language Models

**arXiv ID:** 2604.07035 | [PDF](https://arxiv.org/pdf/2604.07035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 460. Exploring the proprioceptive potential of joint receptors using a biomimetic robotic joint

**arXiv ID:** 2604.07038 | [PDF](https://arxiv.org/pdf/2604.07038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 461. Strategic Persuasion with Trait-Conditioned Multi-Agent Systems for Iterative Legal Argumentation

**arXiv ID:** 2604.07028 | [PDF](https://arxiv.org/pdf/2604.07028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 462. AgentCity: Constitutional Governance for Autonomous Agent Economies via Separation of Power

**arXiv ID:** 2604.07007 | [PDF](https://arxiv.org/pdf/2604.07007v1)

**作者:** Anbang Ruan `[一作]` (NetX Foundation), Xing Zhang `[通讯]` (NetX Foundation)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一种名为 Separation of Power (SoP) 的治理架构，利用智能合约作为立法媒介，将跨组织的自主 AI 代理在开放互联网上进行立法、执行与审判的三重分离，构建了一个可审计、可问责的自治代理经济。

**💡 创新点**

创新点：
- 将法律与执行统一为可公开可验证的智能合约，打破“Logic Monopoly”。
- 通过全排序 Condorcet 投票与结构化协调检测实现代理间的民主决策与反作弊。
- 采用 EMA 信誉系统与双重嵌入评分的 Guardian，实现基于绩效的激励与异常检测。
- 在 EVM 兼容 L2 上实现 AgentCity，三层合约体系（基础、元、操作）实现完整的自治经济流程。
- 在 Commons 生产经济中预注册实验验证从个体问责到集体对齐的理论假设。

**🔧 技术方法**

技术栈：
- 区块链（以太坊兼容 L2）智能合约；
- 三层合约架构（Foundational、Meta、Operational）；
- 多代理框架（OpenClaw/ZeroClaw 等）；
- Condorcet 一致性投票（Copeland+Minimax）；
- EMA 信誉更新；
- 双重嵌入评分 Guardian；
- 结构化立法与执行管线；
- 人类裁决面板与完整所有权链。

**📊 数据集**

数据与实验环境：
- 模拟生成的代理人格（合作、利己、对抗）比例 60/25/15；
- 能力向量按 Beta(α=2, β=5) 采样；
- 成本按 LogNormal(μ=3, σ=0.5) 采样；
- 10 种子、200 代理、10 个里程碑，每里程碑 20 轮；
- 没有使用公开真实数据集，全部为仿真数据。

**📈 对比分析**

比较方法与性能：
- 四配置（Baseline、Emergent、AgentCity-Structural、AgentCity-Full）对比；
- 主要指标包括合作可持续率、专化指数、规则演化计数、治理开销比率、合规率等；
- 统计检验采用配对 Wilcoxon、单侧 Binomial、T 检验等，Bonferroni 校正；
- 预期结果：Full 配置显著优于 Baseline，表现为更高的合作率、更快的专化、更多规则演化、低治理开销。

**⚠️ 局限性**

局限性：
- 仅在仿真环境中验证，缺乏真实多机构部署与网络攻击测试；
- 人类裁决采用模拟面板，未与真实法律体系或人类行为交互；
- Clerk 角色被视为可信，未分析其安全性与潜在攻击；
- 仅在 Commons 生产经济上测试，无法证明对其他经济模式的适用性；
- 对于大规模（>500 代理）仍缺乏足够的实证支持；
- 对拜占庭阈值、合规前提的假设限制了系统在恶意主导环境下的鲁棒性；
- 代理不直接编写 Solidity，可能忽略在真实系统中出现的编译与执行复杂性。

---

## 463. EVGeoQA: Benchmarking LLMs on Dynamic, Multi-Objective Geo-Spatial Exploration

**arXiv ID:** 2604.07070 | [PDF](https://arxiv.org/pdf/2604.07070v1)

**作者:** Jianfei Wu `[一作]` (Beijing Normal University), Zhiyu He `[通讯]` (National University of Defense Technology)

**通讯引用:** 29347 | [OpenAlex ID](https://openalex.org/A5033025672)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EVGeoQA 基准，用于评估大型语言模型在动态地理空间环境中的多目标探索能力；

**💡 创新点**

创新点在于将查询绑定到实时用户坐标并引入充电需求与活动偏好的双重目标，以及基于工具增强的 GeoRover 评估框架；

**🔧 技术方法**

采用了 K‑means 聚类生成用户位置、模板+LLM 生成多样化查询、CoT 及思考模式的提示策略，并设计了四种交互式地理空间工具；

**📊 数据集**

使用了中国三大城市（杭州、青岛、临沂）的充电站、POI 及用户坐标数据，构建了约 44,000 对 QA；

**📈 对比分析**

在 Hits@1/2/3 等指标上对比了多种 LLM（Qwen、GPT‑OSS、Gemini 等），发现大模型在短距离任务表现良好，但随着搜索半径扩大准确率显著下降；

**⚠️ 局限性**

局限包括数据仅来自三座中文城市、缺乏多语言支持、模型在长程探索和属性融合上易出现“懒惰”与信息混淆等问题。

---

## 464. Production-Ready Automated ECU Calibration using Residual Reinforcement Learning

**arXiv ID:** 2604.07059 | [PDF](https://arxiv.org/pdf/2604.07059v1)

**作者:** Andreas Kampmeier `[一作]` (RWTH Aachen University), Jakob Andert `[通讯]` (RWTH Aachen University)

**通讯引用:** 1928 | [OpenAlex ID](https://openalex.org/A5055163459)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

开发了一种基于残差强化学习的可解释自动ECU校准流程，并在硬件在环（HiL）平台上实现并验证了空气质量设定点的校准。

**💡 创新点**

将残差RL与传统手写控制器结合，实现可解释的校准；通过将RL策略映射到查找表并量化为校准参数；采用LExCI框架实现实时在线训练，保证安全与可解释性。

**🔧 技术方法**

残差强化学习（PPO/TD3/DDPG），LExCI框架，硬件在环（HiL）仿真，模型在Loop/MiL/SiL，神经网络+查找表映射，基于NOx、黑烟、压差等的奖励函数。

**📊 数据集**

使用基于WLTC驾驶周期的HiL仿真数据，真实发动机与排放模型；训练期间累计295小时运行经验。

**📈 对比分析**

与手工校准的基准查找表对比，采用NOx、黑烟、燃油消耗、EGR位置等多指标；结果显示NOx下降约20%，累计奖励-570.6 vs-574.9，燃油消耗相近，EGR位置略高。

**⚠️ 局限性**

奖励权重需人工设定，可能导致黑烟升高；需要多次迭代收敛；仅验证单一控制器，未扩展至多代理或更复杂地图；依赖HiL仿真精度。

---

## 465. Planning Task Shielding: Detecting and Repairing Flaws in Planning Tasks through Turning them Unsolvable

**arXiv ID:** 2604.07042 | [PDF](https://arxiv.org/pdf/2604.07042v1)

**作者:** Alberto Pozanco `[一作]` (J.P. Morgan AI Research), Daniel Borrajo `[通讯]` (J.P. Morgan AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了规划任务屏蔽（planning task shielding）方法，自动检测并最小化地修改规划动作，使得原先可达的错误状态无法再通过任何计划实现。

**💡 创新点**

创新点在于将规划任务从可解转为不可解的最优修复问题，定义了只通过添加前置条件、移除产生效果和添加删除效果三类修改来实现屏蔽，并给出了相应的MILP最优化模型。

**🔧 技术方法**

采用了多步流程：先使用symk规划器枚举所有简单（无环）可解计划，再构造并求解混合整数线性规划（MILP）以寻找最小修改集合；使用CBC求解器求解MILP。

**📊 数据集**

实验使用自己设计的合成基准，控制计划数（8/16/32）和计划长度，产生数十至数千个动作/流畅变量，共30个随机实例。

**📈 对比分析**

与传统规划器的比较：测量屏蔽任务的求解时间和所需修改数量。结果显示，修改数随计划数增加而递增（平均6→21），求解时间从几秒增长到数百秒；在小任务中计划生成耗时占大多数，而在大任务中MILP求解成为主导。

**⚠️ 局限性**

局限性包括：只考虑了三类动作修改且未引入修改偏好；基准仅为合成数据，缺乏真实领域的多样性；MILP求解随问题规模呈指数增长，导致大规模实例难以实用。

---

## 466. Not all tokens contribute equally to diffusion learning

**arXiv ID:** 2604.07026 | [PDF](https://arxiv.org/pdf/2604.07026v1)

**作者:** Guoqing Zhang `[一作]` (Bejing Jiaotong University), Yigang Cen `[通讯]` (Bejing Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DARE框架，通过分布感知校正和空间对齐改进条件扩散模型的语义引导。

**💡 创新点**

创新点在于引入Distribution-Rectified Classifier-Free Guidance（DR-CFG）抑制低语义密度token的梯度以及自监督的Spatial Representation Alignment（SRA）实现重要token的空间重加权。

**🔧 技术方法**

采用Flow Matching训练、跨注意力重加权、token重要性权重动态计算与自监督对齐等技术。

**📊 数据集**

训练使用WebVid10M数据集，评估使用VBench和EvalCrafter评测集。

**📈 对比分析**

在Wan 2.1与Seedance 1.0_exp基模型上微调，DARE在VBench与EvalCrafter多维度指标上平均提升5–10%，显著优于现有方法。

**⚠️ 局限性**

局限性包括训练时SRA约束导致的稳定性挑战，以及在长句prompt下信息丢失问题仍未完全解决。

---

## 467. DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification

**arXiv ID:** 2604.07166 | [PDF](https://arxiv.org/pdf/2604.07166v1)

**作者:** Robert Zimmermann `[一作]` (Leibniz University Hannover), Bodo Rosenhahn `[通讯]` (Leibniz University Hannover)

**通讯引用:** 10085 | [OpenAlex ID](https://openalex.org/A5040412734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级的可解释性适配器 DINO‑QPM，将冻结的视觉基座 DINOv2 的高维特征转换为稀疏、对比性强且类无关的低维表示，实现全局可解释的图像分类。

**💡 创新点**

创新点在于：①在冻结模型上直接应用 QPM 框架，构建稀疏低维决策层；②利用平均池化将 patch embeddings 与特征向量直接关联，保证空间可定位；③加入 L1 稀疏损失以抑制背景噪声并提升“合理性”（Plausibility）指标；④提出新的 Plausibility 指标和 Patch Contextualisation 指标评估解释质量。

**🔧 技术方法**

技术包括：自监督视觉基座 DINOv2、二次编码 MLP、稀疏二进制低维决策层（BLDD）、二次规划（QP）实现特征选择与类分配、L1 稀疏损失、Plausibility 与 SID@5 等可解释性指标。

**📊 数据集**

使用了两个细粒度分类数据集：CUB‑2011（鸟类）和 Stanford Cars（汽车），并在 DINOv2 ViT‑B/14 等多种尺寸上进行实验。

**📈 对比分析**

与线性探针、Dense F^froz、ResNet‑50 QPM、DINO‑SLDD、DINO‑QSENN 等方法对比，DINO‑QPM 在准确率、Plausibility、SID@5、Class‑Independence、Contrastiveness 等指标上均优于基线，尤其在 CUB‑2011 上达 100% Plausibility；同时训练时间与参数量保持低（约 6 秒/epoch）。

**⚠️ 局限性**

局限性包括：①仍依赖冻结的 DINOv2 基座，无法处理需要微调的任务；②对背景噪声的抑制主要通过 L1 损失，可能对复杂场景不够鲁棒；③评估指标主要基于细粒度数据集，泛化到更大尺度或多类别任务尚未验证；④在大规模 ViT‑L/14 上表现下降，说明对全局上下文的依赖仍需改进。

---

## 468. SBBTS: A Unified Schrödinger-Bass Framework for Synthetic Financial Time Series

**arXiv ID:** 2604.07159 | [PDF](https://arxiv.org/pdf/2604.07159v1)

**作者:** Alexandre Alouadi `[一作]` (BNP Paribas), Huyên Pham `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了SBBTS框架，用于生成既能匹配边缘分布又能复制时间动态的金融时间序列；

**💡 创新点**

将Schrödinger桥与Bass马尔可夫原则统一，既校准漂移又校准随机波动；

**🔧 技术方法**

采用条件最优输运分解、神经网络参数化漂移、近似大β传输映射以及SB训练算法；

**📊 数据集**

在Heston模型的合成数据以及S&P500日收益数据上进行实验；

**📈 对比分析**

与传统SBTS、零样本、真实数据训练以及简单噪声增强对比，SBBTS在参数恢复、分类准确率、ROC AUC、Sharpe比等指标上均优于对手；

**⚠️ 局限性**

对β取值缺乏系统化选择、收敛理论未给出、仅适用于无跳跃的连续过程，未来需扩展到跳跃模型和非均匀采样。

---

## 469. The Quadratic State Cost of Classical Simulation of One-Way Quantum Finite Automata

**arXiv ID:** 2604.07058 | [PDF](https://arxiv.org/pdf/2604.07058v1)

**作者:** Zeyu Chen `[一作]`, Junde Wu `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了在严格切点语义下，任何 n 状态的 1gQFA 可以被一阶概率有限自动机（PFA）以恰好 Θ(n²) 的状态数精确模拟。

**💡 创新点**

创新点在于首次给出了 1gQFA 到 PFA 的最优状态复杂度界，证明上界为 2n²+6，下界为 n²-1，并阐明这一结果与量子态空间维数的关系。

**🔧 技术方法**

主要使用了混合态线性化技术将 1gQFA 转化为 n² 维的 GFA，再利用 Turakainen 型 GFA→PFA 的构造实现状态数线性提升。

**📊 数据集**

由于论文为理论研究，未使用任何实验数据集。

**📈 对比分析**

比较方式是理论证明：上界通过构造实现，下界通过 VC‑维与量子预备-测试构造相结合得到，显示两者匹配，证明最优。

**⚠️ 局限性**

限制在于常数项仍不匹配（上界 2n²+6 与下界 n²-1），并未讨论两路或混合量子模型的情况。

---

## 470. Synthetic Dataset Generation for Partially Observed Indoor Objects

**arXiv ID:** 2604.07010 | [PDF](https://arxiv.org/pdf/2604.07010v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 471. An RTK-SLAM Dataset for Absolute Accuracy Evaluation in GNSS-Degraded Environments

**arXiv ID:** 2604.07151 | [PDF](https://arxiv.org/pdf/2604.07151v1)

**作者:** Wei Zhang `[一作]` (University of Stuttgart), Norbert Haala `[通讯]` (University of Stuttgart)

**通讯引用:** 3711 | [OpenAlex ID](https://openalex.org/A5049647675)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究开发了一套专门用于评估RTK‑SLAM在GNSS受限环境下绝对定位精度的数据集与评估方法。

**💡 创新点**

创新点在于①将RTK仅作为系统输入而非地面真值，独立用全站仪和静态GNSS构建子厘米级绝对参考；②提出不做SE(3)对齐的绝对误差评估，揭示标准ATE可能低估误差多达76%。

**🔧 技术方法**

使用了LiDAR‑惯性、视觉‑惯性、LiDAR‑视觉‑惯性多传感器融合的RTK‑SLAM系统，并与单独RTK做对比。

**📊 数据集**

数据集包含Stadtgarten（城市公园）和Construction Hall（建筑工地）两场景，四条序列，配有Livox MID360 LiDAR、摄像头、IMU和RTK，同时提供独立全站仪测站点。

**📈 对比分析**

通过绝对ATE和SE(3)对齐后的相对ATE进行比较，结果显示在开阔区所有RTK‑SLAM方法子5cm绝对精度，GNSS受限区离线优化可维持10cm级别，单独RTK在室内可达数米误差；对齐误差高达76%揭示系统偏差。

**⚠️ 局限性**

限制包括：数据集只覆盖两种场景，仍未对原始GNSS观测做紧耦合；系统对GNSS可用时间窗口敏感；对大型复杂室内空间的评估仍不足。

---

## 472. Energy Saving for Cell-Free Massive MIMO Networks: A Multi-Agent Deep Reinforcement Learning Approach

**arXiv ID:** 2604.07133 | [PDF](https://arxiv.org/pdf/2604.07133v1)

**作者:** Qichen Wang `[一作]` (KTH Royal Institute of Technology), Cicek Cavdar `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 2980 | [OpenAlex ID](https://openalex.org/A5006937058)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于多智能体PPO的分布式能源管理框架，能在CF‑mMIMO网络中自适应地进行天线重构与高级睡眠模式选择，以降低功耗。

**💡 创新点**

创新点在于：①利用真实的DPI交通数据构建动态流量模型；②在多智能体设定下使用集中训练、分散执行的MAPPO算法，实现各AP在完全分布式下协同优化天线与睡眠状态；③通过全局奖励与集中评论器解决多智能体训练中的非平稳性。

**🔧 技术方法**

核心技术包括：CF‑mMIMO系统模型、PPZF预编码、ASMs功耗折扣、MDP建模、MAPPO算法（PPO‑Clip、GAE、Huber损失）、分布式执行与云端集中训练。

**📊 数据集**

使用瑞典运营商的深度包检测（DPI）数据，提取三类服务的时空流量密度并映射为UE到达率；并通过标准化的3GPP UMi NLOS模型设置信道。

**📈 对比分析**

与两种无学习基线（Always‑on、DAC‑SM1）以及DQN对比；在模拟的1 km²区域内，MAPPO将网络功耗降低56.23 %（相对Always‑on）和30.12 %（相对DAC‑SM1），且下降率仅略高于阈值；相比DQN，MAPPO在相同功耗水平下下降率更低，显示出更优的QoS与能效平衡。

**⚠️ 局限性**

局限性包括：①仿真场景仅为2D平面，未考虑高度或非均匀AP部署；②功耗模型采用经验系数，实际部署时需进一步校准；③训练过程需要大量模拟时间，真实网络上线时的快速适配仍需研究。

---

## 473. DDP-SA: Scalable Privacy-Preserving Federated Learning via Distributed Differential Privacy and Secure Aggregation

**arXiv ID:** 2604.07125 | [PDF](https://arxiv.org/pdf/2604.07125v1)

**作者:** Wenjing Wei `[一作]` (Université Paris Cité), Alla Jammine `[通讯]` (Université Paris Cité)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5092265877)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个名为 DDP-SA 的联邦学习框架，在客户端使用本地差分隐私（LDP）对梯度加噪，再通过全阈值加法秘密共享（ASS）实现安全聚合，实现端到端的隐私保护。

**💡 创新点**

创新点在于将 LDP 与全阈值 ASS 两种技术耦合：1) 通过 LDP 在本地扰动梯度，保证统计隐私；2) 通过 ASS 在多台中间服务器上加密共享，保证传输和服务器端的安全；3) 设计了可扩展的多服务器架构和多轮隐私预算分配策略。

**🔧 技术方法**

使用的技术包括：本地差分隐私（Laplace 机制）、全阈值加法秘密共享（ASS）与安全聚合、梯度裁剪、固定点编码、隐私预算的高级组合理论、基于 PyTorch + PySyft 的实现。

**📊 数据集**

实验使用合成数据集：10000×2 的均匀随机样本做线性回归，分为 60/20/20 的训练/验证/测试集，模型为两层 2→1 的神经网络。

**📈 对比分析**

与四种防护策略比较：No-Private、LDP、MPC、DDP-SA。评估指标包括通信轮数、上传参数量、收敛时间、每轮训练时长、测试损失和 R²。结果显示：DDP-SA 在保持与 LDP 相同隐私预算的前提下，模型精度（R²）高于 LDP、接近 MPC，通信/计算开销略高于 LDP，但仍保持可接受的线性规模。

**⚠️ 局限性**

局限性包括：仅在 IID 数据上验证，未考虑非 IID、缺失容错和大规模客户端崩溃；使用的是简单的线性回归实验，未在真实复杂任务中评估；实现基于模拟环境，未探讨主动攻击或网络不可靠场景；以及隐私预算在多轮训练中的分配仍需更深入优化。

---

## 474. Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples

**arXiv ID:** 2604.07097 | [PDF](https://arxiv.org/pdf/2604.07097v1)

**作者:** Reiji Saito `[一作]` (Meijo University), Kazuhiro Hotta `[通讯]` (Meijo University)

**通讯引用:** 2169 | [OpenAlex ID](https://openalex.org/A5103163418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了“异常转正常(A2N)”与“正常转异常(N2A)”两种新场景，定义了针对规格变化的评估指标S‑AUROC，并在此基础上设计了RePaste训练增强方法；

**💡 创新点**

创新点在于：①首次将规格变化视为任务场景并引入专门评估指标；②通过RePaste在训练时重新粘贴高分异常区域来主动让模型学习重新定义为正常的特征；

**🔧 技术方法**

采用了基于GLASS的伪异常生成与梯度上升技术，并结合Mixup式边界融合的RePaste数据增强；

**📊 数据集**

使用工业视觉缺陷基准MVTec AD数据集进行实验；

**📈 对比分析**

与FastFlow、PatchCore、RD4AD、RD++、SimpleNet、DiAD、mambaAD、INP‑Former、UniNet、Dinomaly等方法在A2N/N2A场景下进行比较，RePaste在S‑AUROC上分别比GLASS提升0.59%与0.50%，且在常规AUROC、PRO指标上保持或略优于GLASS；

**⚠️ 局限性**

局限在于：仅在MVTec AD上验证，缺乏对其他多样化工业场景的泛化评估；RePaste不解决因规格变化导致的模型迁移/持续学习的全局适配问题；

---

## 475. Aerial Booster-Cell Enabled Inter-Cell Interference Coordination for 5G NR Networks

**arXiv ID:** 2604.07145 | [PDF](https://arxiv.org/pdf/2604.07145v1)

**作者:** Md Sharif Hossen `[一作]` (North Carolina State University), Ismail Guvenc `[通讯]` (North Carolina State University)

**通讯引用:** 15121 | [OpenAlex ID](https://openalex.org/A5016722903)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在多小区5G NR宏网中，为UAV提供可靠下行连接，提出基于基站升倾优化与时域干扰协调的框架。

**💡 创新点**

创新点是将可配置的升倾天线与已存在的降倾天线共置为“booster cell”，并联合优化升倾角与NR兼容的时域干扰协调，专注于极端最差SIR的最大化。

**🔧 技术方法**

采用粒子群优化（PSO）和混合遗传算法（GA）进行非凸优化，同时使用3GPP天线/传播模型、SIR评估与TDIC机制。

**📊 数据集**

使用模拟的19格六边形小区模型，UAV在中心单元的二维网格（10m步长）放置，配合已给定的网络与天线参数。

**📈 对比分析**

与随机、单一倾斜、降倾天线基准对比，PSO在US和CS时隙下的最差SIR和中位/总速率均优于其它方法，显示提升显著。

**⚠️ 局限性**

限制包括假设UAV静止、完全CSI、忽略噪声、未考虑上行与波束成形，且仅在模拟环境下验证，缺乏实测与移动UAV的动态适配。

---

## 476. CSA-Graphs: A Privacy-Preserving Structural Dataset for Child Sexual Abuse Research

**arXiv ID:** 2604.07132 | [PDF](https://arxiv.org/pdf/2604.07132v1)

**作者:** Carlos Caetano `[一作]` (Universidade Estadual de Campinas), Sandra Avila `[通讯]` (Universidade Estadual de Campinas)

**通讯引用:** 3618 | [OpenAlex ID](https://openalex.org/A5057680257)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

发布了一个基于RCPD数据集的隐私保护结构化数据集，包含场景图和人体骨架图；

**💡 创新点**

首次公开使用骨架图作为CSAI分类的结构化表示，并证明两种结构化表示互补；

**🔧 技术方法**

采用Pix2Grp生成场景图、YOLOv8Pose提取骨架图，基于Graph Attention Network (GATv2)进行分类，并用XGBoost实现两模态融合；

**📊 数据集**

基于RCPD 1,630张包含人物的图像（837 CSAI、793 非CSAI），生成场景图和骨架图；

**📈 对比分析**

通过5折交叉验证比较单一场景图（Acc 74.47%，Recall 72.49%）、单一骨架图（Acc 64.39%，Recall 67.16%）和融合模型（Acc 75.33%，Recall 74.96%），融合模型相较于单模态提升约2.5个百分点召回率；

**⚠️ 局限性**

局限性包括骨架图在近摄/遮挡场景下精度低、单模态模型对某些类型样本误判、数据集规模仍有限，且无法验证在更广泛实际数据上的泛化性。

---

## 477. Dynamic Context Evolution for Scalable Synthetic Data Generation

**arXiv ID:** 2604.07147 | [PDF](https://arxiv.org/pdf/2604.07147v1)

**作者:** Ryan Lingo `[一作]` (Honda Research Institute, USA, Inc.), Rajeev Chhajer `[通讯]` (Honda Research Institute, USA, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Dynamic Context Evolution（DCE）框架，解决大规模LLM跨批次生成时的模式坍塌问题，保证输出多样性；

**💡 创新点**

创新点在于将模型自我评估过滤（Verbalized Tail Sampling）、语义记忆去重以及自适应提示演化三大机制结合，形成可组合、无需微调的多层防坍塌体系；

**🔧 技术方法**

技术上使用模型自评估的概率阈值过滤、1536维语义嵌入向量数据库（ChromaDB）做近似最近邻去重、HDBSCAN聚类及EDV（Effective Diversity Volume）度量评估多样性；

**📊 数据集**

实验数据集包括可持续包装概念、初中/大专教育试题以及创意写作提示共三种任务；

**📈 对比分析**

与基线（naive、仅VTS、仅去重、Seed Rotation等）对比，DCE在所有任务中实现0%跨批次坍塌、EDV保留率约23%（相对基线提升）并持续产生17‑19个概念簇；

**⚠️ 局限性**

局限性：仅在三种文本生成任务上验证，缺乏对代码或对话等领域的泛化；依赖人工标注的类别标签导致gap targeting随时间退化；语义相似度阈值为经验值，需任务自适应；大规模应用时需改为近似检索以降低时间成本。

---

## 478. Assessing the Added Value of Onboard Earth Observation Processing with the IRIDE HEO Service Segment

**arXiv ID:** 2604.07120 | [PDF](https://arxiv.org/pdf/2604.07120v1)

**作者:** Parampuneet Kaur Thind `[一作]` (Sapienza Università di Roma), Andrea Taramelli `[通讯]` (Istituto Universitario di Studi Superiori)

**通讯引用:** 1297 | [OpenAlex ID](https://openalex.org/A5073575501)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估了IRIDE HEO平台在火灾烧毁面积映射服务中机载AI处理的增值效果，并与传统仅靠地面处理的架构进行对比；

**💡 创新点**

将机载推理与国家级服务链紧密耦合，实现子3米分辨率、3公顷最小映射单元的即时火灾检测，形成比Copernicus更及时、细粒度的烧毁产品；

**🔧 技术方法**

采用Intel Myriad X VPU+OpenVINO推理堆栈，结合量化、剪枝、NAS等模型轻量化技术；系统架构涵盖FOS、CMPM、PDGS、IRIDE市场，实现卫星任务、数据下行与产品交付的闭环；

**📊 数据集**

使用IRIDE HEO光学微星群获取的多光谱影像，参考Copernicus EFFIS、CEMS、CLMS等公开火灾数据集作为基准进行比较；

**📈 对比分析**

对比时延（获取第一可操作信息、全链路延迟）、下行数据量、空间分辨率、最小映射单元等指标，实验显示机载‑地面链路将时延从数小时/天降至分钟/小时，下行数据量显著降低，且对小型火灾的检测率提升；

**⚠️ 局限性**

目前仅在服务层面验证机载推理增值，缺乏实测机载模型部署与验证；硬件资源限制需进一步优化模型，且与Copernicus系统的兼容性与跨平台集成尚待完善。

---

## 479. Genie Sim PanoRecon: Fast Immersive Scene Generation from Single-View Panorama

**arXiv ID:** 2604.07105 | [PDF](https://arxiv.org/pdf/2604.07105v1)

**作者:** Zhijun Li `[一作]`, Maoqing Yao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Genie Sim PanoRecon的全流程面向360°全景图的无训练高效Gaussian splatting管线，实现快速生成具有全局几何一致性的3D场景。

**💡 创新点**

将DA360全景深度与DepthPro细节深度通过逆深度拉普拉斯金字塔融合，并在cubemap面上采用训练‑free深度注入的SHARP实现跨视图一致性，避免了传统逐场景优化的瓶颈。

**🔧 技术方法**

使用DA360、DepthPro、逆深度拉普拉斯融合、cubemap投影、SHARP无训练深度引导的Gaussian splatting、深度注入和Gaussian点合并等技术。

**📊 数据集**

基于室内真实全景、DiT360合成全景以及GAN生成的室内图像作为输入进行评估；未说明使用公开数据集。

**📈 对比分析**

与仅使用DA360或DepthPro深度、无跨视图一致性以及传统像素对齐的PanSplat等方法对比，PanoRecon在纹理细节和全局几何一致性上表现更好；运行时间仅数秒，显著快于迭代优化方法。

**⚠️ 局限性**

仍无法补全未观测区域、对大视差或户外环境适用性有限、以及在高分辨率细节和极端极点区域存在残留失真。

---

## 480. STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems

**arXiv ID:** 2604.07100 | [PDF](https://arxiv.org/pdf/2604.07100v1)

**作者:** Hongru Ji `[一作]` (Northwestern Polytechnical University), Chao Gao `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 27805 | [OpenAlex ID](https://openalex.org/A5100748612)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了STRIDE-ED框架，采用情景总结、情绪识别、策略推理和策略引导的逐步推理生成同理对话

**💡 创新点**

构建了覆盖正、中、负情绪的完整同理策略体系，并将策略与动作分层、引入难度级别；采用LLM自动标注与多模型一致性评估进行数据精炼

**🔧 技术方法**

利用大型语言模型（DeepSeek-R1、Qwen3、Llama-3.1）进行标注与评估，采用两阶段训练：监督微调+多目标PPO强化学习；通过结构化中间标签实现链式思考

**📊 数据集**

以EmpatheticDialogues数据集为基础，扩充并标注情景、策略、动作，最终生成ED-CSA-5k等子集

**📈 对比分析**

与传统模型、外部知识增强模型和反射式决策模型对比，自动评估BLEU、Dist、PPL、情绪准确率；在人类A/B评估中在同理心、相关性、流畅度上均显著优于基线

**⚠️ 局限性**

缺乏严格的统计验证与超参调优，模型对不同基础LLM的适配性未完全探究，数据过滤与采样机制的理论基础待完善

---

## 481. Multiple Domain Generalization Using Category Information Independent of Domain Differences

**arXiv ID:** 2604.07175 | [PDF](https://arxiv.org/pdf/2604.07175v1)

**作者:** Reiji Saito `[一作]` (Meijo University), Kazuhiro Hotta `[通讯]` (Meijo University)

**通讯引用:** 2169 | [OpenAlex ID](https://openalex.org/A5103163418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种通过将特征图拆分为域无关类别信息与源域特有信息，并结合SQ‑VAE量化向量实现语义分割的域泛化方法。

**💡 创新点**

创新点在于：①使用DeepCCA对拆分的特征图进行去相关，强制其分别学习域无关类别信息和源域特有信息；②利用SQ‑VAE的量化向量将不同类别的特征聚类，对齐源域与目标域的类别分布，从而弥补域差距。

**🔧 技术方法**

技术包括：DeepCCA、Gumbel‑Softmax、SQ‑VAE、U‑Net/UCTransNet 编码器、量化向量聚类、重构损失（mse+log）、类别向量聚类损失。

**📊 数据集**

数据集：视网膜血管分割的 Drive、Stare、Chase 三个不同相机拍摄的数据；细胞核分割的 MoNuSeg（多医院、不同染色方法）。

**📈 对比分析**

与 U‑Net 和 UCTransNet 直接使用的模型在相同特征提取器下对比，使用 mIoU 评估。实验显示：在每个数据集的域泛化任务中，mIoU 提升 1–3%，血管或细胞核类别的 IoU 分别提升 2–5% 左右。

**⚠️ 局限性**

局限性：①仍依赖源域特定信息，无法完全消除域差距；②目前仅验证二分类任务，未扩展到多类；③对量化向量数量和训练样本规模敏感；④需手动标注源域数据，适用范围受限。

---

## 482. The Josehedron: A space-filling plesiohedron based on the Fischer-Koch S Triply Periodic Minimal Surface

**arXiv ID:** 2604.07160 | [PDF](https://arxiv.org/pdf/2604.07160v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

---

## 483. Critical Inker: Scaffolding Critical Thinking in AI-Assisted Writing Through Socratic Questioning

**arXiv ID:** 2604.07167 | [PDF](https://arxiv.org/pdf/2604.07167v1)

**作者:** Philipp Hugenroth `[一作]` (MIT Media Lab), Pattie Maes `[通讯]` (MIT Media Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Critical Inker 这款基于 LLM 的写作辅助工具，通过可视化逻辑错误提示和 Socratic 对话来促进写作时的批判性思考。

**💡 创新点**

创新点在于将 LLM 的论证结构提取与逻辑有效性评估与 Socratic 提问相结合，并提供两种交互模式，既能直观展示逻辑错误，又能通过逐一提问引导用户自己发现并修正错误。

**🔧 技术方法**

技术上采用多阶段 LLM 递归提示 pipeline，包括结构提取、链式推理的有效性判断以及工具模式下的 JSON 交互；使用 Claude Sonnet 4.5 等 LLM。

**📊 数据集**

使用 Argument Annotated Essay v2 数据集评估结构提取，使用 Stanford NLI（改为逻辑有效性检验）评估逻辑判断。

**📈 对比分析**

通过与人工标注对比，结构重叠率 91.2%，有效性准确率 87%，平均推理时间 6.58 秒（Claude Sonnet 4.5）且与 GPT-4.1 及 Gemini Flash 相比延迟更低，显示技术可行。

**⚠️ 局限性**

局限在于可视化模式对部分用户产生认知负担，Socratic 模式在提问与已表达内容之间可能产生摩擦；实验规模小，需进一步评估对长期批判性思维的影响。

---

## 484. USCNet: Transformer-Based Multimodal Fusion with Segmentation Guidance for Urolithiasis Classification

**arXiv ID:** 2604.07141 | [PDF](https://arxiv.org/pdf/2604.07141v1)

**作者:** Changmiao Wang `[一作]` (Shenzhen Research Institute of Big Data), Ahmed Elazab `[通讯]` (Tsinghua University)

**通讯引用:** 2533 | [OpenAlex ID](https://openalex.org/A5042305251)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了USCNet模型，实现了预手术尿路结石的CT影像与电子病历的多模态融合，用于结石分类和分割。

**💡 创新点**

创新点在于①CT‑EHR交叉注意力融合CT影像与临床数据；②分割多模态注意力SMA模块将分割特征动态引入分类；③基于Dice分数的动态损失权重实现分割→分类的训练递进。

**🔧 技术方法**

技术主要包括Transformer（ViT‑UNetSeg）、多模态交叉注意力、Dice/BCE+Focal损失以及自适应权重调节。

**📊 数据集**

使用本机构642例患者的CT影像及7项临床变量，共1355张图像，标注为感染与非感染两类。

**📈 对比分析**

与ResNet系列、StoneNet、SegPrompt、nnU‑net、UNETR、ResGANet、TMSS、HyMNet等方法对比，USCNet在分类Acc、F1、AUC、Dice、IoU等指标均显著优于前者，分类Acc达91.91%。

**⚠️ 局限性**

局限包括对多模态数据完整性依赖强、单中心数据导致潜在谱偏差、Transformer模型计算量大、难以在资源受限环境部署。

---

## 485. Immersed boundary-conformal isogeometric methods for magnetostatics

**arXiv ID:** 2604.07155 | [PDF](https://arxiv.org/pdf/2604.07155v1)

**作者:** Yusuf T. Elbadry `[一作]`, Oliver Weeger `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

**🎯 论文内容**

本文介绍了elsarticle.cls文档类的功能和使用方法，旨在为Elsevier期刊的投稿提供更好的格式化支持。

**💡 创新点**

创新点在于重新设计了文档类，减少了与其他包的冲突，并提供了更简便的格式化选项。

**🔧 技术方法**

使用了LaTeX文档类和多个标准包，如natbib、geometry、graphicx等。

**📊 数据集**

未提及具体数据集，主要是关于文档格式化的技术细节。

**📈 对比分析**

与之前的elsart.cls相比，elsarticle.cls在格式化和兼容性方面有显著改进，提供了更好的用户体验。

**⚠️ 局限性**

限制在于用户需要自行检查长公式的断行，以确保最终排版的准确性。

---

## 486. Lumbermark: Resistant Clustering by Chopping Up Mutual Reachability Minimum Spanning Trees

**arXiv ID:** 2604.07143 | [PDF](https://arxiv.org/pdf/2604.07143v1)

**作者:** Marek Gagolewski `[一作]` (Warsaw University of Technology), Marek Gagolewski `[通讯]` (Warsaw University of Technology)

**通讯引用:** 1302 | [OpenAlex ID](https://openalex.org/A5076546160)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Lumbermark 算法，一种基于互相可达性最小生成树的分治式聚类方法，可在已知聚类数时进行剪枝并得到高质量分割。

**💡 创新点**

通过在互相可达性距离中加入微小加权 δ 解决 MST 多重解歧义；利用剪除叶子和最小聚类大小阈值的规则，实现可指定聚类数的鲁棒分割；为 HDBSCAN 提供了聚类数可控的替代方案。

**🔧 技术方法**

使用互相可达性距离、Borůvka/Prim 并行化的 MST 计算、分治剪枝、连通分量剪切；实现为 Python/R 包，配合 quitefastmst 高效计算 MST。

**📊 数据集**

评估使用了 61 个低维（≤10k 点，≤3D）基准数据集和 47 个具有参考标签的子集，并在随机高维数据上进行跑时实验。

**📈 对比分析**

与 HDBSCAN（standard/fast）、K‑Means、EM、Ward、Spectral、Genie、ITM 等算法在 AR、运行时间和误差上对比；Lumbermark 在 AR≥0.95 的案例最多，速度快，尤其在低内在维度下表现优异。

**⚠️ 局限性**

局限性包括：在高维数据上 MST 计算成本和性能下降；参数 M、f 的选择仍需经验；若 min_cluster_factor 过大可能无法得到指定的聚类数；仅适用于已知聚类数的场景。

---

## 487. Smart Commander: A Hierarchical Reinforcement Learning Framework for Fleet-Level PHM Decision Optimization

**arXiv ID:** 2604.07171 | [PDF](https://arxiv.org/pdf/2604.07171v1)

**作者:** Yong Si `[一作]` (Tsinghua University), Zhaokui Wang `[通讯]` (Tsinghua University)

**通讯引用:** 18682 | [OpenAlex ID](https://openalex.org/A5027945800)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种名为 Smart Commander 的分层强化学习框架，用于军用航空机队的预测与健康管理（PHM）决策。

**💡 创新点**

创新点在于将机队 PHM 分解为战略级 General Commander 与战术级 Operation Commanders 的层级结构，采用分层奖励和规划增强的神经网络，显著缓解高维状态、稀疏延迟奖励和非平稳环境的挑战。

**🔧 技术方法**

使用多智能体深度 Q‑学习（HRL）技术，配合优先经验回放、双 DQN、分段 Q 值计算、梯度裁剪、Huber 损失、分层奖励、时间尺度异步更新以及课程学习等手段。

**📊 数据集**

使用自研的高保真离散事件模拟平台生成的合成数据，模拟航空机队的任务、退化、维护与物流过程。

**📈 对比分析**

与基于规则的启发式方法和单层全局深度强化学习（DRL）进行对比；实验显示 Smart Commander 在可用率、任务成功率、成本效益、训练收敛速度和扩展性方面均优于基线，特别是在系统复杂度提升十倍或失效率翻倍时保持稳定性能。

**⚠️ 局限性**

局限包括：仿真与真实环境之间的差距、缺乏真实航空机队数据、对部分可观测性和多机队资源共享的处理不足，以及未实现在线适应和风险感知决策。

---

## 488. Agent-Driven Corpus Linguistics: A Framework for Autonomous Linguistic Discovery

**arXiv ID:** 2604.07189 | [PDF](https://arxiv.org/pdf/2604.07189v1)

**作者:** Jia Yu `[一作]` (Zhejiang International Studies University), Fukun Xing `[通讯]` (Zhejiang International Studies University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了“Agent‑Driven Corpus Linguistics”框架，即让大型语言模型（LLM）在与语料库查询引擎通过结构化工具调用（Model Context Protocol, MCP）连接后，独立完成假设生成、查询执行、结果解释与迭代优化，最终输出已被语料库证据支撑的研究发现。

**💡 创新点**

创新点在于：①将LLM的工具使用能力与语料库查询无缝整合，形成可审计的、可重复的研究日志；②通过“自我迭代”实现完全自主的假设生成与修正，突破传统需要人工设计查询的瓶颈；③在不牺牲理论视角的前提下，显著降低技术门槛与时间成本，使更广泛的研究者能够使用语料库方法；④在基准实验中清晰划分“回忆型输出”“基于语料的量化”“数据驱动的理论构建”三层贡献。

**🔧 技术方法**

使用的技术包括：Anthropic Claude Opus LLM、Model Context Protocol (MCP) 的工具调用接口、CQP（Corpus Query Processor）索引的标准化语料库引擎，以及 Python 服务器桥接层实现查询命令的自动化转换。

**📊 数据集**

主要使用的数据集为：5,019,103-token 的标准化 Project Gutenberg 子集（69 篇英语文本，含词形、词性、依存关系、文本和句子层级元数据），以及外部验证用的 40.3M-token CLMET 语料库（1710–1920 年三段时序文本）。

**📈 对比分析**

比较方法：①与相同 LLM 在无语料库访问（仅凭训练记忆）下的输出进行对照，评估“grounding”对量化、可证伪性和理论构建的贡献；②通过复制两篇已发表研究（Claridge 2025 的读者频率下降、De Smet 2013 的动名词补语扩散）检验框架的外部有效性。实验显示，带语料库访问的 LLM 能提供精确频率、显著的量化差异（如“really”戏剧文本 20 倍高于诗歌）并构建新的三途径语义变化模型；复制结果与原文数值高度吻合，验证了方法的可靠性与可推广性。

**⚠️ 局限性**

局限性包括：①语料规模有限且时间分布不均，影响统计显著性和 diachronic 结论的稳健性；②依赖 LLM 的训练知识，难以完全区分“真正发现”与“记忆回调”，尤其在解释层面；③工具调用依赖手动配置，可能产生查询误差；④多模型与多跑实验表明解释结果存在可变性，需要进一步聚合与校准；⑤词义多义性及情感分类的可靠性需人工验证。

---

## 489. Splats under Pressure: Exploring Performance-Energy Trade-offs in Real-Time 3D Gaussian Splatting under Constrained GPU Budgets

**arXiv ID:** 2604.07177 | [PDF](https://arxiv.org/pdf/2604.07177v1)

**作者:** Muhammad Fahim Tajwar `[一作]` (National University of Singapore), Bhojan Anand `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

研究了在不同GPU功耗限制下实时3D高斯剖分渲染的可行性，并通过在单块RTX 4090上进行低频与功耗抑制模拟多种GPU阶层，测量帧率、功耗等指标。

**💡 创新点**

提出了一种基于持续TFLOPS的GPU能力仿真方法，并首次量化不同GPU阶层在多层次细节场景中3DGS的性能‑能耗折衷。

**🔧 技术方法**

使用CUDA实现的3DGS光栅化器、分层LOD优化、4D动态高斯剖分、GPU低频/功耗抑制仿真以及能耗度量等技术。

**📊 数据集**

采用MIP‑NeRF Garden场景进行训练与评测。

**📈 对比分析**

对四个GPU阶层（RTX 4090、4070 Ti、3070、3050）在四个LOD级别（约0.6–3.4 M高斯剖分）进行两分钟帧率测量，结果显示RTX 3070在≤1 M剖分可维持≥60 fps，RTX 3050在最高LOD下帧率不足30 fps，动画导致帧率下降显著。

**⚠️ 局限性**

仿真仅考虑计算吞吐和内存频率，未完全匹配实际SM数量、缓存和带宽；仅基于单一Garden场景；未测量网络延迟；仅针对CUDA平台，无法直接迁移到移动SoC。

---

## 490. InfiniLoRA: Disaggregated Multi-LoRA Serving for Large Language Models

**arXiv ID:** 2604.07173 | [PDF](https://arxiv.org/pdf/2604.07173v1)

**作者:** Hongyu Chen `[一作]` (Shanghai Jiao Tong University), Shixuan Sun `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15510 | [OpenAlex ID](https://openalex.org/A5052957554)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种将 LoRA 适配器与基础 LLM 模型解耦的分布式推理架构 InfiniLoRA，以解决在 MoE 等大模型下 LoRA 缓存容量不足导致的尾部延迟膨胀。

**💡 创新点**

创新点包括：
1) 分离 LoRA Server 与 LLM 实例，实现 LoRA 缓存容量独立扩展；
2) 提出并实现混合并行（Pipeline+Expert）方案，平衡同步、通信与计算负载；
3) 基于服务水平目标（TTFT/TPOT）的 SLO 驱动资源配置，自动确定最小 GPU 数量；
4) 采用主机绕过、GPU 直写 RDMA 的推送通信，以及硬件专用 LoRA 内核，显著压缩解码关键路径。

**🔧 技术方法**

主要技术手段包括：LoRA 低秩更新公式、MoE 专家并行、Pipeline 并行、Hybrid 并行、Poisson 化概率模型 + 二分 + 动态规划求解最优缓存容量、GPU 直写 RDMA、TMA/warp 专化、SGMV/BGMV 变换、预取层级加载、基于 SLO 的资源自适应调度。

**📊 数据集**

使用了五种 MoE 语言模型（GPT‑OSS‑20B、Qwen3‑30B‑A3B、Mixtral‑8x7B、Scaled‑MoE、DBRX）以及对应的 LoRA 适配器（512 个，rank 64），并基于生产流量的 Zipf 分布生成请求负载。

**📈 对比分析**

与现有 S‑LoRA、S‑LoRA+SJF、S‑LoRA+Less LoRA 等基线进行对比；在相同硬件预算下，InfiniLoRA 在 P95 TTFT 与平均 TPOT 仍满足 0.25 s / 0.1 s 的 SLO，平均请求服务率提升 3.05 倍，LoRA 适配器满足 SLO 的比例提升 54%（相较 S‑LoRA），吞吐量提升 7.3%（最高 24.7%）。

**⚠️ 局限性**

局限性主要体现在：
1) 仍受 LoRA Server 缓存容量瓶颈，需进一步扩大服务器规模；
2) 现有实现仅针对 GPU 直写 RDMA，跨机通信延迟仍高；
3) 对极低延迟场景下的热加载与自适应缓存策略仍有改进空间；
4) 对 CPU‑侧 LoRA 计算的支持有限，难以兼顾大规模多租户。

---

## 491. Selective Neuron Amplification for Training-Free Task Enhancement

**arXiv ID:** 2604.07098 | [PDF](https://arxiv.org/pdf/2604.07098v1)

**作者:** Ryyan Akhtar `[一作]` `[通讯]` (Guru Gobind Singh Indraprastha University), Ryyan Akhtar (Guru Gobind Singh Indraprastha University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在推理阶段放大已存在但激活不足的神经元，提升LLM在低置信度任务（如算术、诗歌、情感分类）上的性能，且不改变模型参数。

**💡 创新点**

提出Selective Neuron Amplification (SNA) 方法，构建Three‑Zone Saturation Model 并证明基线置信度是预测 SNA 效果的关键指标；揭示层级专一化与跨任务干扰现象。

**🔧 技术方法**

差分激活分析识别任务相关神经元，使用 TransformerLens 钩子实现激活放大，完成 24,192 组合实验。

**📊 数据集**

GPT‑2 Small/Medium（117M/345M）与 12 个任务（算术、诗歌、编码、逻辑、情感分类）以及 Pythia‑160M 算术测试。

**📈 对比分析**

对比 SNA 前后提升：Zone 1（置信度<0.07）平均提升 27.85%，单例可达 70%；Pythia 算术提升 178%；SST‑2 低置信度提升 57%；高置信度无显著变化；Zone 3 仅提升 ≤7%。

**⚠️ 局限性**

仅在 GPT‑2 系列上验证，Zone 2 仍需更多实验；分类任务需要较高倍率；仅做单任务干预，未测试多任务或微调；跨模型通用性需进一步验证。

---

## 492. Improving Semantic Uncertainty Quantification in Language Model Question-Answering via Token-Level Temperature Scaling

**arXiv ID:** 2604.07172 | [PDF](https://arxiv.org/pdf/2604.07172v1)

**作者:** Tom A. Lamb `[一作]` (University of Oxford), Tim G. J. Rudner `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究了语言模型问答中的语义不确定性量化（Semantic Uncertainty Quantification），通过系统评估语义校准（calibration）和判别性（discrimination）两方面来改进模型的可靠性。

**💡 创新点**

创新点在于：①提出对语义置信度分布进行校准的首个系统评估；②证明单一全局温度缩放（Temperature Scaling, TS）比传统固定温度或更复杂的词级校准方法（ATS、Platt）在语义空间中更有效；③设计了七种语义置信度度量，包括新提出的 ML‑SC、IC‑SC、B‑SC、T‑SC、G‑SC，验证其在不同方法下的稳健性。

**🔧 技术方法**

技术上使用：温度缩放、适应性温度缩放（ATS）、Platt 缩放、负对数似然和选择性平滑损失进行参数优化；生成多个答案并通过 NLI 模型聚类得到语义群组；计算各种语义置信度分布并评估 ACE、AUROC 等指标；在 QA 任务上实现多样采样与后处理。

**📊 数据集**

数据集包括 TriviaQA、Natural Questions（NQ）用于闭本问答，SQuAD 用于开本问答；使用 Llama‑3.1‑8B‑Instruct、Ministral‑8B‑Instruct‑2410、Qwen‑2.5‑7B‑Instruct 三种大模型，并在每个数据集上划分校准集与测试集。

**📈 对比分析**

与基线（未校准 Base、固定温度 SE、ATS、Platt）对比，TS 在所有语义置信度度量上均取得 ACE 降低、AUROC 提升、语义熵判别更强、选择性准确率提升等显著优势；尤其在闭本数据集上显著改善校准，在开本数据集上显著提升判别性；实验显示 TS 在样本数 5–25 时保持稳定，且校准随样本数增大而收敛。

**⚠️ 局限性**

局限性包括：仅在短文本生成 QA 任务中验证，无法直接推广到摘要、对话等部分正确输出的任务；语义校准的定义依赖于 NLI 聚类，聚类误差会影响结果；当前方法仍需要在更大规模模型和多模态任务上进一步测试。

---

## 493. Multi-Turn Reasoning LLMs for Task Offloading in Mobile Edge Computing

**arXiv ID:** 2604.07148 | [PDF](https://arxiv.org/pdf/2604.07148v1)

**作者:** Ning Yang `[一作]` (Chinese Academy of Sciences), Haijun Zhang `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 190302 | [OpenAlex ID](https://openalex.org/A5100408669)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了基于大型语言模型（LLM）的多轮推理框架 COMLLM，用于移动边缘计算（MEC）中的任务迁移决策。

**💡 创新点**

创新点在于：①通过语义化状态序列化实现拓扑无关的输入表示；②结合 Group Relative Policy Optimization（GRPO）对 LLM 进行强化学习微调；③设计 Look‑Ahead Collaborative Simulation（LACS）奖励机制，将多步未来队列演化纳入奖励，提升远程决策能力；④实现零训练投递到更大、未知拓扑的“zero‑shot”可迁移。

**🔧 技术方法**

主要技术包括：大型语言模型（如 Qwen‑1.5B / Qwen‑7B）、语义化文本提示、监督式微调（SFT）、GRPO 强化学习、Monte‑Carlo 未来任务模拟、KL 正则化、奖励重塑。

**📊 数据集**

使用模拟生成的 MEC 环境数据，构建三类数据集：SFT 训练集（1,000 样本）、GRPO 交互集（2,000 样本）和测试集（1,000 样本），任务与服务器参数随机采样。

**📈 对比分析**

与 Random、DQN、SFT、GRPO 等基线进行对比。COMLLM 在平均延迟、任务丢弃率、性能比、负载均衡指数等指标上均优于所有基线，尤其在不同服务器数量、任务负载和语义提示扰动下保持零任务丢弃率和高公平性。

**⚠️ 局限性**

局限性：①实验仅基于仿真环境，缺乏真实网络验证；②依赖大量离线“oracle”标签进行 SFT；③模型规模较大（7B）时训练成本高；④LACS 的未来任务采样范围有限，可能在极端高峰或长时延场景下仍有欠缺。

---

## 494. Learning to Search: A Decision-Based Agent for Knowledge-Based Visual Question Answering

**arXiv ID:** 2604.07146 | [PDF](https://arxiv.org/pdf/2604.07146v1)

**作者:** Zhuohong Chen `[一作]` (Tsinghua University), Haoqian Wang `[通讯]` (Tsinghua University)

**通讯引用:** 6119 | [OpenAlex ID](https://openalex.org/A5028229824)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于搜索代理的知识驱动视觉问答框架（DBAgent），将 KB‑VQA 视为多步决策过程，动态选择图像检索、文本检索、标题生成和答案生成等工具，并在推理过程中自适应调整检索深度。

**💡 创新点**

创新点在于把检索与推理解耦为可动态规划的决策链；通过自动化生成多步轨迹数据，为模型提供决策级监督；使用 <think> 等标签实现透明的决策记录，让模型学习何时检索、何种检索方式以及何时停止。

**🔧 技术方法**

技术包括：多模态大型语言模型 Qwen2.5‑VL‑7B 作为 backbone；监督微调（SFT）配合轨迹级标签；图像检索采用 EchoSight + BGE‑m3；文本检索使用 BGE embeddings；自动化多阶段轨迹构建与线性化训练模板。

**📊 数据集**

使用的主要数据集为 InfoSeek（约 1.3M 例子，11k Wikipedia 页）和 E‑VQA（约 221K 例子，16.7k Wikipedia 页）；检索库基于 Wikipedia 构建。

**📈 对比分析**

与零射击 MLLM、经典 RAG、反射式 RAG、VLM‑PRF 等基线在 InfoSeek All、E‑VQA All 等指标上对比，DBAgent 在 InfoSeek 48.7% 与 E‑VQA 45.2% 均领先 6‑10 分，并在 Unseen‑Q/Unseen‑E 子集表现尤为突出。

**⚠️ 局限性**

局限性包括：实验在相对干净的检索环境下进行，开放世界噪声和不完整证据的鲁棒性待验证；轨迹生成依赖提示策略和基础模型，可能引入误导；动作空间仅覆盖四种工具，未涵盖知识图谱等更多工具；多步推理导致推理延迟。

---

## 495. Autopoiesis: A Self-Evolving System Paradigm for LLM Serving Under Runtime Dynamics

**arXiv ID:** 2604.07144 | [PDF](https://arxiv.org/pdf/2604.07144v1)

**作者:** Youhe Jiang `[一作]` (Hong Kong University of Science and Technology), Binhang Yuan `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 666 | [OpenAlex ID](https://openalex.org/A5002684888)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自我进化的 LLM 服役范式，通过 LLM 驱动的程序合成不断演化服役策略，使得服务计划随运行时动态自适应。

**💡 创新点**

创新点包括：① 用 LLM 与进化算法结合的程序合成工作流实现对服役策略的自动探索；② 将策略执行与策略演化拆分为数据平面与控制平面，形成闭环的自我进化架构；③ 在控制平面实现持续在线演化，实时根据监控数据更新策略；④ 在变异过程中融合运行时权衡信息与评估反馈，进行“权衡感知”变异。

**🔧 技术方法**

技术手段：大型语言模型（GLM‑4.7‑Flash 等）做代码生成；进化算法（MAP‑Elites、岛屿策略、温度调节）做搜索；评估器通过轨迹回放模拟器估算调度、迁移和服务时间；异步热交换实现无中断策略切换；多级超时控制保证演化过程可预测。

**📊 数据集**

数据集：ShareGPT、LongBench 作为请求负载；Microsoft Azure Functions (MAF) 作为集群动态轨迹；瑞士 AI 中心的实测工作负载；在不同硬件配置下（均质、异质、弹性）分别采样。

**📈 对比分析**

对比方法：与 DistServe、HexGen、SpotServe 等最先进的 LLM 服役系统在同一硬件与负载下进行基准测试；使用全流程结束时间（调度+迁移+服务）作为指标；结果显示在均质集群上最高可提升 31%，平均 27%；在异质集群上最高 33%，平均 30%；在弹性集群上最高 53%，平均 44%；演化在 10 分钟内收敛；使用 warm‑start 可将演化时间缩短约 80%。

**⚠️ 局限性**

局限性：对 LLM 的生成质量与稳定性依赖较高；演化过程需要较大算力，尽管做了超时控制但仍可能产生停滞；需要手工提供初始种子策略和超参数调优；在极端突发场景或非 LLM 工作负载下的鲁棒性尚待验证；系统在多租户、混合模型部署中的细粒度动态适配仍有待完善。

---

## 496. Language Bias under Conflicting Information in Multilingual LLMs

**arXiv ID:** 2604.07123 | [PDF](https://arxiv.org/pdf/2604.07123v1)

**作者:** Robert Östling `[一作]` (Stockholm University), Murathan Kurfalı `[通讯]` (RISE Research Institutes of Sweden)

**通讯引用:** 831 | [OpenAlex ID](https://openalex.org/A5002870078)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

作者在多语种大型语言模型中构建了冲突针-稻草堆实验，检验模型在包含互相矛盾信息的检索任务中的表现。

**💡 创新点**

创新点在于首次将针-稻草堆范式扩展至多语言场景，系统揭示不同语言在答案选择中的偏好及其跨国差异。

**🔧 技术方法**

采用多语言对照实验、二项显著性检验和层次贝叶斯模型等技术，测试了12个不同规模、不同来源的LLM的推理过程。

**📊 数据集**

实验数据来源于5种语言（中文、德语、俄语、土耳其语、英语）的公开新闻文本，手工构造并翻译了8条冲突伪事实。

**📈 对比分析**

结果显示所有模型几乎不识别冲突，且普遍忽视俄语、偏好中文；对不同模型、来源的偏好差异进行了量化比较，验证了模型间一致的语言偏好模式。

**⚠️ 局限性**

受限于样本语言和模型规模有限、仅使用英文原稿的新闻、提示设置最小CoT以及缺乏更广泛的主题与语言多样性，限制了结论的普适性。

---

## 497. Mixed-Initiative Context: Structuring and Managing Context for Human-AI Collaboration

**arXiv ID:** 2604.07121 | [PDF](https://arxiv.org/pdf/2604.07121v1)

**作者:** Haichang Li `[一作]` (George Mason University), Zhicong Lu `[通讯]` (George Mason University)

**通讯引用:** 34578 | [OpenAlex ID](https://openalex.org/A5063218435)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“混合主动性上下文（Mixed‑Initiative Context）”概念，将多轮人机交互产生的上下文视为可显式、可结构化、可操作的交互对象，并实现了探针系统 Contextify 以验证该概念；

**💡 创新点**

创新点在于：1) 把上下文从隐式、一次性堆叠改为可视化、可编辑的结构化对象；2) 在上下文层面引入混合主动性，允许人机共同决定何时、何种结构调整；3) 通过小型探针系统和用户研究系统化探讨该框架的可行性与设计空间；

**🔧 技术方法**

技术上使用大型语言模型（如 GPT‑4）结合四个后台代理（Conversation, Structure, Memory, User‑Model），实现上下文单元的创建/编辑/激活/分支等操作，并在前端采用三面板节点画布交互；

**📊 数据集**

数据集主要是实验参与者（6 名）完成两项设计任务，收集屏幕录制、日志、访谈记录；并未使用公开大规模文本或标注数据；

**📈 对比分析**

与基线 ChatGPT 的线性聊天对照，用户在 Contextify 条件下表现出更高的结构化操作频率、较好的控制感，SUS 分数约为 72；在任务完成度和满意度上无显著差异，但在主观体验和对 AI 结构性干预的接受度上有明显提升；

**⚠️ 局限性**

局限性包括：1) 仅在自然语言文本场景下验证，未扩展到代码、图像等多模态；2) 样本量小（6 人），研究偏向探索性；3) 代理与 LLM 仅通过 Prompt 调整，缺乏深度学习模型的长期学习；4) UI 采用最小节点画布，学习成本仍存在，且对高水平多用户协作支持不足。

---

## 498. TeaLeafVision: An Explainable and Robust Deep Learning Framework for Tea Leaf Disease Classification

**arXiv ID:** 2604.07182 | [PDF](https://arxiv.org/pdf/2604.07182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 499. Yale-DM-Lab at ArchEHR-QA 2026: Deterministic Grounding and Multi-Pass Evidence Alignment for EHR Question Answering

**arXiv ID:** 2604.07116 | [PDF](https://arxiv.org/pdf/2604.07116v1)

**作者:** Elyas Irankhah `[一作]`, Samah Fodeh `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

论文提出了一个四子任务的 EHR 问答系统，利用提示式少样本学习、模型集成和自一致性投票来实现患者问题的重写、证据检索、答案生成和证据对齐。

**💡 创新点**

创新点在于：①设计了统一的模块化管道，兼容 Azure 基础设施；②系统性地对集成规模、少样本数量和投票阈值进行消融实验；③在证据对齐子任务中引入全文回答上下文和嵌入召回增强，显著提升了 F1 分数。

**🔧 技术方法**

使用的技术包括：Claude Sonnet 4、GPT‑4o、GPT‑5.1/5.2、DeepSeek‑R1 等大型语言模型；基于提示的零/少样本推理；模型集成与自一致性采样；JSON 结构化输出和投票阈值调整；BERTScore 等语义评估指标。

**📊 数据集**

使用的数据集为 ArchEHR‑QA 2026，包含 167 个病例（20 个 dev、147 个 test），每个病例包括患者自由文本问题、编号句子的临床记录、临床医生解读的问题与答案以及金标准证据对齐。

**📈 对比分析**

与官方基准相比，系统在子任务 1–4 的性能分别为：问题重写 27.09（R1）、证据检索 61.90（micro F1）、答案生成 30.95（BLEU）以及证据对齐 80.41（micro F1）。实验表明，多模型集成和投票阈值调整能显著提升证据检索与对齐的鲁棒性。

**⚠️ 局限性**

主要局限包括：①数据集规模小，仅 20 个开发样本，限制了超参数调优；②对提示工程高度依赖，细微提示变化可能导致预测偏差；③各子任务独立处理，错误可能在阶段间传播；④系统依赖大规模专有模型，成本高且可复现性受限。

---

## 500. Information as Structural Alignment: A Dynamical Theory of Continual Learning

**arXiv ID:** 2604.07108 | [PDF](https://arxiv.org/pdf/2604.07108v1)

**作者:** Radu Negulescu `[一作]` `[通讯]` (Informational Buildup Foundation), Radu Negulescu (Informational Buildup Foundation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了信息对齐框架（IBF）作为一种全新的持续学习子结构，演示其在三种不同领域（自定义旋转规则世界、国际象棋、Split‑CIFAR‑100）中的记忆、代理和自我修正能力。

**💡 创新点**

核心创新在于：
- 将信息视为结构对齐，而非数据存储；
- 从两条基本方程（运动律和修改动力学）推导出记忆、代理和自我修正的机制；
- 通过局部核化的粒子机制实现“晶化”“危机”与“融合”等动态过程，完全无需外部重放、正则化或子网络分区。

**🔧 技术方法**

技术实现：
- 基于“Law of Motion” 的贝尔曼动作选择；
- “Modification Dynamics” 的局部核更新与时间分离；
- 读写路径中的可读门控、崩溃检验与激活；
- 通过核化粒子（value 与 agency 两族）实现空间局部化记忆与响应性调节；
- 训练时采用几何分辨率校准自动设定核宽度。

**📊 数据集**

数据集与环境：
- RRW：8 维随机输入与可逆规则的手工生成序列；
- Chess：12 维由 5‑head CNN 冻结的棋盘+动作表征，使用 Stockfish depth‑4 作为训练信号，depth‑8 评估；
- Split‑CIFAR‑100：冻结 ViT‑B/16+PCA 的 64 维视觉空间，68 维 class‑augmented 空间，20 个连续任务；
- Frozen‑LLM 扩展（未包含在核心实验中）。

**📈 对比分析**

评估方法与性能：
- 在 RRW 上与 MLP、Replay、EWC、Passive 进行对照；IBF 在保留率上减少 43% 的遗忘，并实现正向转移。 
- 在 Chess 上与 Stockfish 评估，IBF 对比被动基线获得平均 +88.9 cp 的行为优势，BT_A≈+35.4，显著优于 Replay。 
- 在 CIFAR‑100 上与 Replay、EWC、MLP 对照；IBF 在 20 个任务后 BT≈-0.004（几乎零遗忘），且在无任务标签下的 Class‑IL 仍保持 52.8% 的准确率，超过 Replay 的 39.2%。

**⚠️ 局限性**

局限性：
- 仅在预先冻结的编码器和基线评估器上实现，未展示端到端表征学习；
- 维度上限在 68 维，未检验更大空间的可扩展性；
- 训练过程需要外部纠错信号（如 Stockfish 或标签），缺少无监督或无阶段的连续环境；
- 仅单智能体实验，未探究多智能体互动、通信与协作；
- 机制的实现依赖显式的核化粒子，缺乏对真实硬件（如神经形态）可迁移性的实验。

---

## 501. SurFITR: A Dataset for Surveillance Image Forgery Detection and Localisation

**arXiv ID:** 2604.07101 | [PDF](https://arxiv.org/pdf/2604.07101v1)

**作者:** Qizhou Wang `[一作]` (University of Melbourne), Christopher Leckie `[通讯]` (University of Melbourne)

**通讯引用:** 12489 | [OpenAlex ID](https://openalex.org/A5076014464)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SurFITR数据集，针对监控场景的图像伪造检测与定位，利用多模态LLM驱动的自动化局部编辑技术生成137k张带像素级掩码的伪造图像；

**💡 创新点**

创新点包括①基于LLM与定位模型的语义对齐细粒度伪造生成管道；②提供多生成模型（FLUX、Stable Diffusion、LanPaint等）与跨域结构的训练与评测框架；③在监控数据上系统评估现有检测器与MLLM的零样本性能，揭示显著差距并证明SurFITR训练可显著提升。

**🔧 技术方法**

技术实现采用Qwen2.5‑VL‑72B等LLM、YOLO‑World与SAM 2.1等定位/分割模型、FLUX.1‑Fill‑dev、LanPaint、Stable Diffusion等生成模型，评估CNN/Transformer检测器（PSCC‑Net、MVSSNet、HiFi‑Net、TruFor、CAT‑Net等）以及商业/开源MLLM（Qwen3‑VL‑32B、Gemini Flash 3等）。

**📊 数据集**

数据集来源为六大监控语料库（UCF Crime、NTU Fight、ShanghaiTech、CUHK Avenue、UCSD Ped1/2），生成SurFITR Base与Transfer，并与DRCT‑2M、GenImage、FaceForensics++、IMD2020等公开数据集对比。

**📈 对比分析**

对零样本进行评估，所有基线AUROC仅0.5–0.7，F1低于0.3，像素定位P‑IoU<0.1；在SurFITR上微调后，AUROC>0.95、F1>0.8，但跨域/跨模型场景仍显著下降，显示对场景变化与小尺寸伪造的鲁棒性不足。

**⚠️ 局限性**

局限性包括：模型对场景差异和细微编辑的检测/定位不稳健；生成模型与真实分布差距仍存在；手工验证样本仅占5%，大规模自动验证可能漏检；数据集仅公开用于研究，缺乏对更复杂或隐私环境的覆盖。

---

## 502. Energy-based Tissue Manifolds for Longitudinal Multiparametric MRI Analysis

**arXiv ID:** 2604.07180 | [PDF](https://arxiv.org/pdf/2604.07180v1)

**作者:** Kartikay Tehlan `[一作]` (University Hospital Augsburg), Thomas Wendler `[通讯]` (University Hospital Augsburg)

**通讯引用:** 1777 | [OpenAlex ID](https://openalex.org/A5011729018)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出基于能量的个体化序列空间流形，用以对多参数MRI进行纵向分析，无需分割或监督分类。

**💡 创新点**

创新点在于将能量函数视为几何参考，将基线能量流形固定并在随访中评估其几何偏移，检测早期肿瘤复发。

**🔧 技术方法**

采用去噪分数匹配训练隐式神经表示（SIREN+Fourier特征）的能量模型，并计算梯度、拉普拉斯曲率等几何量。

**📊 数据集**

使用两例儿科脑肿瘤的诊断前后多参数MRI（T1、T1c、T2、FLAIR、ADC）进行验证。

**📈 对比分析**

通过比较随访扫描的能量均值和沿健康–肿瘤轴的漂移，发现复发病例出现能量升高和向肿瘤基线流形的偏移，稳定病例无明显变化；该方法在两例病例中表现出潜在的早期警示。

**⚠️ 局限性**

局限在样本量极小、仅为病例证明，缺乏大规模验证，且未结合空间信息，可能受序列空间分布变化影响。

---

## 503. Reason in Chains, Learn in Trees: Self-Rectification and Grafting for Multi-turn Agent Policy Optimization

**arXiv ID:** 2604.07165 | [PDF](https://arxiv.org/pdf/2604.07165v1)

**作者:** Yu Li `[一作]` (George Washington University), Tian Lan `[通讯]` (George Washington University)

**通讯引用:** 27347 | [OpenAlex ID](https://openalex.org/A5016639693)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Tree-structured Self-Taught Agent Rectification（T-STAR）框架，利用认知树对多步RL轨迹进行聚合、变分化奖励分配，并通过思维嫁接实现自我纠正，最终用手术式策略优化提升LLM代理性能。

**💡 创新点**

创新点在于①将看似独立的轨迹合并为认知树，恢复潜在的关联奖励结构并实现方差减小的节点优势估计；②在树中识别关键分歧点并通过思维嫁接生成针对性修正思路；③基于生成的优劣对引入Bradley‑Terry手术损失，精细化针对关键步骤的梯度更新。

**🔧 技术方法**

核心技术包括：认知树构建（功能等价与历史兼容判定、KL距离估计、节点合并）、Q-tree估值与优势回传、思维嫁接（自我纠正思路生成）、手术式策略优化（Bradley‑Terry偏好学习）。

**📊 数据集**

使用了四大类数据集：ALFWorld（文本型家居任务）、WebShop（电商导航）、搜索增强问答（单跳：NQ、TriviaQA、PopQA；多跳：HotpotQA、2WikiMultiHopQA、MusiQue、Bamboogle）、逻辑规划任务（Sokoban、Blocksworld）。

**📈 对比分析**

与GRPO、DAPO、GiGPO等基于组的RL基线以及ReAct、Reflexion提示方法、以及闭源模型（GPT‑4o、Gemini‑1.5‑Pro）进行对比。实验显示，T‑STAR在所有任务上均提升3–8%（ALFWorld）、3–5.8%（WebShop）、2.8–7.5%（多跳QA）、4.5–8.5%（中等规划）等，尤其在需要长链推理的场景中优势更明显。

**⚠️ 局限性**

局限性包括：功能等价判定依赖KL近似且易受高熵分布噪声影响；思维嫁接需模型具备足够的推理表达与纠错能力；目前仅在离散文本动作空间验证，连续动作或多模态环境的推广仍待研究。

---

## 504. Bridging MRI and PET physiology: Untangling complementarity through orthogonal representations

**arXiv ID:** 2604.07154 | [PDF](https://arxiv.org/pdf/2604.07154v1)

**作者:** Sonja Adomeit `[一作]` (University Hospital Augsburg), Thomas Wendler `[通讯]` (University Hospital Augsburg)

**通讯引用:** 1777 | [OpenAlex ID](https://openalex.org/A5011729018)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

构建一种基于正交子空间分解的多模态融合框架，将PSMA PET摄取量分为可由mpMRI解释的生理包络与与MRI特征子空间正交的残差。

**💡 创新点**

创新点在于：① 用SVD投影将预测误差显式分解为平行残差和正交残差；② 通过正交残差量化MRI无法解释的PET信息，从而给出跨模态可恢复性的理论上限；③ 采用无空间卷积的隐式神经表示（SIREN）以直接映射强度空间特征到PET信号。

**🔧 技术方法**

主要技术包括：正交子空间投影与SVD正则化；隐式神经表示网络（SIREN）与高维Fourier特征编码；mpMRI与PSMA PET的配准与标准化；以及基于MSE的损失函数。

**📊 数据集**

使用13例已确诊前列腺癌患者的mpMRI（T1w、T2w、ADC、DCE得到的Ktrans、ve、vp、TTP）与对应的PSMA PET/CT扫描，所有数据在同一参考框架下进行配准和归一化。

**📈 对比分析**

通过比较三类区域（肿瘤、非肿瘤前列腺、周围软组织）的MSE，证明正交残差在肿瘤区几乎占全部误差（≈99.9%），而平行残差保持在最低水平；在消除不同MRI模态（单模态与组模态消融）时，总MSE显著升高，尤其是去除动态血流特征导致MSE增至0.30，表明动态信息对PET包络预测最重要。

**⚠️ 局限性**

局限性包括：样本量仅13例，缺乏外部验证；仅使用强度级别特征，未考虑空间上下文；正交残差虽表征不可恢复信息，但其生物学解释仍需要进一步实验验证。

---

## 505. A Utility-preserving De-identification Pipeline for Cross-hospital Radiology Data Sharing

**arXiv ID:** 2604.07128 | [PDF](https://arxiv.org/pdf/2604.07128v1)

**作者:** Chenhao Liu `[一作]` (Shandong University), Yue Yao `[通讯]` (Shandong University)

**通讯引用:** 125408 | [OpenAlex ID](https://openalex.org/A5100743975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于生成过滤和语义约束的实用脱敏管线（UPDP），实现跨医院放射影像与报告的安全共享。

**💡 创新点**

创新点在于同时采用黑名单/白名单双重文本约束与基于扩散模型的图像重构，使脱敏数据在去除身份信息的同时保留诊断相关特征，且保持图像与文本的跨模态一致性。

**🔧 技术方法**

使用扩散模型（Latent Diffusion）、文本编码器（BiomedCLIP）、内容优化与语义对齐、黑名单/白名单约束以及低秩适配（LoRA）进行模型微调。

**📊 数据集**

在公开胸部X射线数据集MIMIC‑CXR和IU‑Xray上进行实验，构造10k子集与全量数据集两种规模。

**📈 对比分析**

通过BLEU、METEOR、ROUGE‑L、BERTScore、RadGraph F1等指标与原始数据、无脱敏数据及多种VLM（Qwen2.5‑VL、Gemma‑3‑4B‑IT、Phi‑3.5‑Vision‑Instruct）进行对比；结果显示，UPDP生成的脱敏数据在诊断准确性上与原始数据持平或略优，且在跨院迁移与混合训练中提升了报告生成质量。

**⚠️ 局限性**

局限性包括：评估指标主要为表面文本相似度，缺乏专家临床评判；隐私泄露风险未进行量化的成员推断或泄漏分析；以及对不同模态（如CT、MRI）通用性的验证尚未展开。

---

## 506. Self-Discovered Intention-aware Transformer for Multi-modal Vehicle Trajectory Prediction

**arXiv ID:** 2604.07126 | [PDF](https://arxiv.org/pdf/2604.07126v1)

**作者:** Diyi Liu `[一作]` (Beijing University of Technology), Lishan Sun `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种纯Transformer结构的多模态车辆轨迹预测模型，采用两条解码器分别预测轨迹与意图概率，利用边缘关系注意力与自回归残差提升性能。

**💡 创新点**

创新点包括：1）引入考虑车辆速度的边缘关系注意力模块；2）采用“先预测后选择”双解码器分离空间与轨迹生成；3）使用多模态残差预测技巧实现有序轨迹集合；4）通过相对偏置实现自适应的空间注意力。

**🔧 技术方法**

使用的技术主要是Transformer编码器-解码器、Multi‑Head Self‑Attention、相对偏置空间注意力、MLP、累积求和残差预测以及WTA与概率损失的组合训练。

**📊 数据集**

使用开源“Ubiquitous Traffic Eyes”数据集，涵盖合并/拆分区的复杂交通场景，包含多条子数据集（SQM‑W、SQM‑N_up等）进行训练，SQM‑N_down作为测试集。

**📈 对比分析**

通过与不同版本（单模、多人模不考虑交互、多人模考虑交互）进行消融实验，最佳模型（MV_K）在5秒预测下RMSE为5.91 m，MAE为2.45 m，显著优于单模和不考虑交互的版本。

**⚠️ 局限性**

限制包括：1）残差预测技巧在交叉路口或左转等场景下效果有限；2）未考虑道路边界与环境因素；3）缺少与其他基于图神经网络的交互模型的对比。

---

## 507. Accuracy Improvement of Semi-Supervised Segmentation Using Supervised ClassMix and Sup-Unsup Feature Discriminator

**arXiv ID:** 2604.07122 | [PDF](https://arxiv.org/pdf/2604.07122v1)

**作者:** Takahiro Mano `[一作]` (Meijo University), Kazuhiro Hotta `[通讯]` (Meijo University)

**通讯引用:** 2169 | [OpenAlex ID](https://openalex.org/A5103163418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种结合监督式ClassMix（SupMix）和特征对齐GAN（SUFD）的半监督语义分割框架。

**💡 创新点**

创新点在于将带标签的类别形状直接粘贴到无标签伪标签上以提升罕见类别学习，并通过对抗式特征判别器减少标注图像与未标注图像之间的域差距。

**🔧 技术方法**

使用的技术包括半监督伪标签生成、监督ClassMix增强、GAN对抗判别器（SUFD）以及基于DeepLabv3+的语义分割网络。

**📊 数据集**

在Chase（视网膜血管）和COVID‑19（肺部CT）两个医学图像数据集上进行实验。

**📈 对比分析**

与传统监督学习、FixMatch和UniMatch对比，平均mIoU提升约2.1%（Chase）和3.1%（COVID‑19），尤其显著提高少样本类别（视网膜血管提升3.3%，肺玻璃样影像提升10.7%）。

**⚠️ 局限性**

局限性包括对极少样本的肺气肿/胸腔积液类别仍表现不佳，且方法对标签位置的随机性可能导致过拟合。

---

## 508. Workmanship of Learning: Embedding Craftsmanship Values in AI-Integrated Educational Tools

**arXiv ID:** 2604.07118 | [PDF](https://arxiv.org/pdf/2604.07118v1)

**作者:** Tuan-Ting Huang `[一作]` (Eindhoven University of Technology), Stephan Wensveen `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 3075 | [OpenAlex ID](https://openalex.org/A5032329859)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过研究-设计方法设计并评估了一款集成AI的创意编码工具，以实现AI工匠主义框架在设计教育中的应用。

**💡 创新点**

将工匠价值（风险、节奏、关怀）嵌入AI工具的交互与界面设计，并把AI定位为中介层而非直接解决者，强调学习过程而非产出。

**🔧 技术方法**

采用p5.js编程库、约束式ChatGPT接口、代码高亮与解释、Capture/Reflection面板等技术实现工具功能。

**📊 数据集**

研究使用来自5名设计师/学生的现场互动数据、访谈转录以及代码与输出快照，未使用公开数据集。

**📈 对比分析**

通过反射性主题分析对访谈与快照进行质性评估，未量化性能指标，结果表明风险与节奏在早期出现，关怀在后期浮现，功能满足预期但缺乏数值化对比。

**⚠️ 局限性**

样本规模小、研究时长短、仅在单一课程环境，缺乏长期跟踪与多领域验证；AI约束可能限制创造性，需进一步优化。

---

## 509. Are Non-English Papers Reviewed Fairly? Language-of-Study Bias in NLP Peer Reviews

**arXiv ID:** 2604.07119 | [PDF](https://arxiv.org/pdf/2604.07119v1)

**作者:** Ehsan Barkhordar `[一作]` (Koc University), Gözde Gül Şahin `[通讯]` (Koc University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并系统化了 NLP 同行评审中因研究语言而产生的偏见（LoS bias），并构建了大型人工标注数据集 LOBSTER，提出基于 LLM 的偏见检测方法，进一步分析了负向与正向偏见在不同语言、贡献类型及偏见子类别下的分布。

**💡 创新点**

首次将 LoS bias 明确为独立的偏见类别，区分正向与负向偏见，挖掘出四种负向偏见子类型，并通过 LLM（Gemini‑3.1‑Pro）实现 87.37 % macro‑F1 的自动检测，为后续公平评审提供可复制工具。

**🔧 技术方法**

使用大语言模型（Gemini‑3.1‑Pro、Claude‑Opus‑4.6 等）进行多任务推断，结合人工标注的 529 份评审片段训练与评估；同时利用 LLM 辅助采样和语言/贡献类型识别。

**📊 数据集**

利用公开的 NLP 评审数据集 NLPEERv2（EMNLP 2023/2024、ACL 2025）与 ARR 数据收集，汇总 15,645 条评审，手工抽样 534 条进行标注，构成 LOBSTER 数据集。

**📈 对比分析**

与多数基线（随机、占多数类）比较，所有 LLM 的 macro‑F1 均显著高于基线；Gemini‑3.1‑Pro 在三类偏见分类任务中取得最高 macro‑F1 87.37%，在贡献类型与语言范围识别任务中分别达到 91.96 % 与 83.99 % 的 macro‑F1。

**⚠️ 局限性**

数据仅来自六个顶级 NLP 会议的公开评审，缺乏被拒稿样本；偏见与语言范围、贡献类型等因素相互关联，难以完全分离；LLM 性能受模型版本与提示的影响，结果可能在不同设置下变化。

---

## 510. The ATOM Report: Measuring the Open Language Model Ecosystem

**arXiv ID:** 2604.07190 | [PDF](https://arxiv.org/pdf/2604.07190v1)

**作者:** Nathan Lambert `[一作]` (Interconnects AI), Florian Brand `[通讯]` (Interconnects AI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过综合 Hugging Face 下载量、OpenRouter 推理份额、Arena Elo 及 Artificial Analysis Index 等公开数据，对 1.5K 主流开源语言模型的采纳与性能进行了系统化监测与量化；

**💡 创新点**

提出了相对采纳度指标（RAM），以模型大小分类的中位下载量为基准，兼顾模型规模与时间维度，提供更细粒度的采纳度评估；

**🔧 技术方法**

利用多源数据抓取、时序聚合、离群点过滤（2.5×IQR）、分段回归与对齐技术构建统一的采纳与性能指标；

**📊 数据集**

主要数据集为 Hugging Face Hub 的下载统计、OpenRouter 的 token 份额、Arena 的 Elo 评测和 Artificial Analysis 的整体智力指数；

**📈 对比分析**

将 RAM 与传统下载量对比，发现 Qwen 系列在 1–10B 级别遥遥领先，深度模型在 250B+ 级别占优；性能上，中文模型已于 2024 年底超过美方模型；

**⚠️ 局限性**

局限性包括：仅监测 Hugging Face 及 OpenRouter 公开流量，忽略私有云部署与模型缓存；下载量与实际使用不完全一致；以及区域归属仅按组织总部估算，可能误差较大。

---

## 511. The Impact of Steering Large Language Models with Persona Vectors in Educational Applications

**arXiv ID:** 2604.07102 | [PDF](https://arxiv.org/pdf/2604.07102v1)

**作者:** Yongchao Wu `[一作]` (Stockholm University), Aron Henriksson `[通讯]` (Stockholm University)

**通讯引用:** 1485 | [OpenAlex ID](https://openalex.org/A5046274853)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了激活式个性化对大型语言模型在教育短答生成与自动评分任务中的影响

**💡 创新点**

首次系统研究激活调度的个性向量在教育短答生成与自动评分中的任务依赖效应

**🔧 技术方法**

利用激活状态干预和个性向量提取，对模型进行正负方向的激活驱动

**📊 数据集**

使用ASAP‑SAS短答评分基准以及三大模型（Qwen3‑4B、Qwen3‑32B、gpt‑oss‑20B）

**📈 对比分析**

通过外部GPT‑5.2评估生成质量和内部模型评分，发现ELA任务对个性化更敏感，MoE模型校准偏差更大

**⚠️ 局限性**

实验规模受限于仅三模型、单一基准、七个特征和固定α值，未覆盖更广泛教育场景与多种特性

---

## 512. ClickGuard: A Trustworthy Adaptive Fusion Framework for Clickbait Detection

**arXiv ID:** 2604.07272 | [PDF](https://arxiv.org/pdf/2604.07272v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 513. Are Face Embeddings Compatible Across Deep Neural Network Models?

**arXiv ID:** 2604.07282 | [PDF](https://arxiv.org/pdf/2604.07282v1)

**作者:** Fizza Rubab `[一作]` (Michigan State University), Arun Ross `[通讯]` (Michigan State University)

**通讯引用:** 29403 | [OpenAlex ID](https://openalex.org/A5061834795)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究不同深度神经网络模型（面部特定模型与通用基础模型）产生的面部嵌入向量在几何结构上的可对齐性，评估通过简单线性变换实现跨模型身份识别与验证的效果。

**💡 创新点**

发现尽管模型在数据集、训练目标和架构上差异巨大，但其嵌入空间可通过低容量线性映射高度对齐，表明存在共享的面部身份几何结构；进一步揭示了模型间的层级组织和方向性不对称。

**🔧 技术方法**

采用三种线性对齐方法（正交Procrustes、普通线性回归、岭回归），对嵌入进行中心化、归一化、零填充后学习变换矩阵；通过排名、mAP、AUC、EER等指标评估识别与验证性能。

**📊 数据集**

使用三大公开人脸基准数据集：CFP（姿态受限）、LFW（野生环境）和CASIA‑WebFace（大规模多样性）。

**📈 对比分析**

对比未对齐基线与对齐后结果：在面部特定模型中，识别Rank‑1从≈1%提升至≈97%；在基础模型中从≈2%提升至≈70%；验证AUC从≈0.5提升至≈0.96（面部特定）或≈0.9（基础模型）。跨数据集实验显示对齐仍有效但性能下降。

**⚠️ 局限性**

局限性包括：对齐对基础模型验证的提升有限，无法满足高安全阈值；对齐效果随目标空间维度和身份分离度变化，存在方向性不对称；实验仅覆盖面部数据，未验证其他生物识别或细粒度视觉任务；对齐方案易受对齐训练集分布偏差影响。

---

## 514. Making Room for AI: Multi-GPU Molecular Dynamics with Deep Potentials in GROMACS

**arXiv ID:** 2604.07276 | [PDF](https://arxiv.org/pdf/2604.07276v1)

**作者:** Luca Pennati `[一作]` (KTH Royal Institute of Technology), Stefano Markidis `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 4865 | [OpenAlex ID](https://openalex.org/A5085178088)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在GROMACS中实现DeePMD-kit的深度潜能推理，支持多GPU和分布式域分解，完成从坐标广播到力归约的完整MD循环。

**💡 创新点**

创新点在于引入虚拟域分解层与双MPI收集，实现无需改核心代码即可在GROMACS上运行多GPU分布式推理；同时提供完整的GPU加速推理管线和可扩展的接口。

**🔧 技术方法**

使用GROMACS 2025、DeePMD-kit v3.1、PyTorch、CUDA/HIP、MPI、GPU加速深度潜能模型（DPA‑1）和自定义的MPI通信层。

**📊 数据集**

使用Unke2019PhysNet_SolvatedProtein_DPA_v3_1数据集，包含259万条水化蛋白碎片的原子位置信息、能量和力，用于训练1.6M参数的DPA‑1模型。

**📈 对比分析**

通过在NVIDIA A100与AMD MI250x上对1HCI（15,668原子）进行强/弱伸缩测试，最大32设备时强缩效率约40%，弱缩效率约50%；相较于经典MD，推理开销约三阶，GPU内存占用升至十倍。

**⚠️ 局限性**

局限性在于ghost atom数决定的最小计算量、推理时间占主导导致的负载不平衡、仅支持局部DPA‑1模型；消息传递型DP模型需要更大halo或点对点通信，且在更大规模节点时通信成本可能变得显著。

---

## 515. GenLCA: 3D Diffusion for Full-Body Avatars from In-the-Wild Videos

**arXiv ID:** 2604.07273 | [PDF](https://arxiv.org/pdf/2604.07273v1)

**作者:** Yiqian Wu `[一作]` (Zhejiang University), Junxuan Li `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了一个能够从文本和图像输入生成与编辑全身高保真、逼真的3D全身动画的扩散模型。

**💡 创新点**

创新点在于使用预训练的前馈重建网络做3D tokenizer，并提出可见性感知训练策略来处理部分可观测的2D视频数据，从数百万真实世界视频中构建大规模3D token数据集，实现了在真实视频数据上训练3D扩散模型。

**🔧 技术方法**

采用了3D高斯光散射（3D Gaussian Splatting）、流式扩散（rectified flow）、MMDiT双流块、CLIP与DINOv2特征、可见性掩码、可学习占位符、压缩器（MLP+自注意力）以及条件流匹配等技术。

**📊 数据集**

使用了约1.113M条真实世界单目视频、2.737个多视角录制视频以及1.198个移动设备全身旋转视频，共计约1.12M身份的数据集，规模远超现有合成或采集数据集。

**📈 对比分析**

与TADA、HumanGaussian、DreamWaltz‑G（SDS方法）以及TeRA、SIGMAN（直接3D扩散）等SOTA方法进行了定性、定量（BLIP‑VQA、CLIP文本评分、CLIB‑FIQA、HyperIQA、FID）和用户研究对比，模型在语义对齐、视觉质量、FID和用户偏好等多项指标上均名列第一或第二，显著优于现有技术。

**⚠️ 局限性**

主要局限在于依赖线性混合蒙皮（LBS）动画导致服装在极端姿势下出现不自然变形；未观测区域的重建仍可能出现模糊或透明，尽管使用占位符缓解，但生成质量仍受这些不足影响。

---

## 516. Non-identifiability of Explanations from Model Behavior in Deep Networks of Image Authenticity Judgments

**arXiv ID:** 2604.07254 | [PDF](https://arxiv.org/pdf/2604.07254v1)

**作者:** Icaro Re Depaolini `[一作]` (University of Trento), Uri Hasson `[通讯]` (University of Trento)

**通讯引用:** 27351 | [OpenAlex ID](https://openalex.org/A5067264638)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在预训练的视觉模型上添加轻量回归头，训练预测人类对AI生成图像真实性的评分，并使用Grad-CAM、MPM和LIME生成解释热图。

**💡 创新点**

首次系统评估不同架构在同一任务下的解释一致性，揭示了Rashomon效应并提出通过集成模型兼顾预测与解释。

**🔧 技术方法**

使用迁移学习、回归适配器、梯度归因、像素遮罩、多尺度像素遮掩、LIME、通道剪枝及Bagging/Stacking集成。

**📊 数据集**

采用AIGCIQA2023数据集（2400张AI图像，过滤后1367张）收集人类真实性、质量和对应性评分。

**📈 对比分析**

评估指标包括RMSE、PLCC、SRCC以及模型可靠性；单个模型最佳PLCC≈0.63，集成模型提升至≈0.73；VGG模型主要捕捉质量信息。

**⚠️ 局限性**

受限于样本量、仅冻结backbone、解释方法的完整性检验不足以及人类评分的噪声上限导致预测与解释的信度有限。

---

## 517. Tracking Adaptation Time: Metrics for Temporal Distribution Shift

**arXiv ID:** 2604.07266 | [PDF](https://arxiv.org/pdf/2604.07266v1)

**作者:** Lorenzo Iovine `[一作]` (Politecnico di Milano), Emanuele Della Valle `[通讯]` (Politecnico di Milano)

**通讯引用:** 4714 | [OpenAlex ID](https://openalex.org/A5015694017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对时间分布漂移中的模型鲁棒性，提出了三种后验度量（Temporal Adaptation Score、Stability Horizon、Drift Horizon），用以分离模型在时间上的适应性能与数据本身难度的混合影响；

**💡 创新点**

创新点在于引入Temporal Transfer Ratio基准化方法，将模型在未来时间点的表现与理想“oracle”相对，随后定义SH和DH两个阈值无关的时间窗口度量，能够动态解析模型随时间的适应速度与稳定性；

**🔧 技术方法**

技术实现主要是对不同时间点训练的模型在后续时间点的准确率矩阵做归一化并统计阈值穿越和累计偏差，配合已有的学习框架（ERM、CORAL、EWC、SI、A‑GEM、Fine‑Tuning、MoCo+SML）进行评估；

**📊 数据集**

数据集采用 Wild‑Time 的两大基准：Yearbook（1930‑2013 年间的性别分类）和 FMoW‑Time（多年的卫星土地利用分类）；

**📈 对比分析**

通过在这两个数据集上对比 ID/OOD、TAS、SH、DH 等指标，发现传统的 ID‑OOD 差距往往掩盖了模型真实的适应能力：例如 MoCo+SML 的 ID 较高但 TAS、SH 低；SI 与 FT 在保持较高 TAS、较长 SH 的同时，ID/OOD 也相当；整体而言，SH 平均约 5 年，DH 约 6 年，显示了模型的适应持续时间；

**⚠️ 局限性**

局限性包括：度量仅为后验，需要完整时间序列标签；未给出在线近似或自适应 retraining 的机制；目前仅在分类任务上验证，未扩展到回归或多模态场景；缺乏理论上对漂移大小与适应速度的正式保证。

---

## 518. $k$-server-bench: Automating Potential Discovery for the $k$-Server Conjecture

**arXiv ID:** 2604.07240 | [PDF](https://arxiv.org/pdf/2604.07240v1)

**作者:** Kirill Brilliantov `[一作]`, Emmanuel Abbé `[通讯]` (EPFL)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个可执行的基准，用于在圆形度量空间下搜索 k‑server conjecture 的潜能函数，提供了 Python 接口、离散化实例与自动化评估管线，并在此框架下实验多种 LLM 驱动的搜索方法。

**💡 创新点**

创新点包括：① 将潜能函数搜索转化为可执行的代码搜索任务，并提供可迭代的自动化反馈；② 在 k=4 圆形案例中突破人类设计的“unifying potential”，得到仅 3 违规点的潜能函数；③ 展示代理与人类交互的编码工作流在加速发现优质潜能函数方面的有效性。

**🔧 技术方法**

使用技术主要包括：大型语言模型（如 GPT‑5、GPT‑4）、进化/搜索框架（Best‑of‑N、ShinkaEvolve、LoongFlow）、Bellman‑Ford 负环检测、k‑taxi 扩展机制以及代理+人类交互编码流程。

**📊 数据集**

数据集为离散化圆形度量的预生成实例，包含 k=3、6、8，m=6、8 的多种尺寸，并在 k‑taxi 约束下生成数百到数十亿条边的工作函数图。

**📈 对比分析**

比较方法：以违规计数或违规比例为评分标准；在 k=3 场景中，LoongFlow 在 10 次实验中以 20% 的比例达到 0 违规；在 k=4 场景下，最佳候选者仅 3 违规，显著低于人类基准 17 违规，表明实验取得了可观进展。

**⚠️ 局限性**

局限性：实例规模随 k、m 与扩展迅速膨胀，导致完整评估成本高；基准主要适用于圆形及其可离散化度量，缺乏广泛适用的其他度量；潜能函数的 Python 表达形式受限，可能无法捕获更复杂或更高效的结构；对 canonical 形式的依赖可能抑制新颖潜能函数的探索。

---

## 519. How Does Machine Learning Manage Complexity?

**arXiv ID:** 2604.07233 | [PDF](https://arxiv.org/pdf/2604.07233v1)

**作者:** Lance Fortnow `[一作]` (Illinois Institute of Technology), Lance Fortnow `[通讯]` (Illinois Institute of Technology)

**通讯引用:** 9149 | [OpenAlex ID](https://openalex.org/A5075191567)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出将现代机器学习模型抽象为可计算且最大熵多项式界定的分布，并证明在对抗伪随机生成器数据训练时，最小化 KL 散度的可计算分布将趋近于均匀分布，从而阐明模型能量来自于随机化与可计算性约束的结合。

**💡 创新点**

创新点在于：①用计算复杂度的视角重新定义机器学习的假设空间；②将可计算分布与多项式时间、非均匀性和最大熵约束统一；③通过信息论与加密理论证明可计算分布在面对伪随机生成器时必然逼近均匀分布，揭示学习过程对复杂行为建模的本质。

**🔧 技术方法**

使用的技术包括计算复杂度理论（非均匀电路、可计算分布）、信息论（KL 散度、熵、Pinsker 不等式）、Kolmogorov 复杂度、加密原语（伪随机生成器、单向函数）以及概率论与可计算性分析。

**📊 数据集**

该工作为纯理论研究，没有使用具体数据集；讨论的分布包括伪随机生成器输出、均匀分布以及构造的可计算分布。

**📈 对比分析**

由于是理论证明而非实验，本文未进行方法对比或性能评估；其结果以信息论上“可忽略”或“相对熵小”形式给出，说明在理论上可计算分布学习对伪随机生成器的逼近优于任何非均匀可计算分布。

**⚠️ 局限性**

局限性包括：①假设学习过程可获得任意可计算分布，实际上训练难以实现；②证明基于单向函数假设，若可计算分布被逼近至均匀，可能违背加密安全；③未考虑实际模型的参数规模、梯度下降等数值限制；④对现实大模型的适用性仍需实验验证。

---

## 520. Improving Feasibility in Quantum Approximate Optimization Algorithm for Vehicle Routing via Constraint-Aware Initialization and Hybrid XY-X Mixing

**arXiv ID:** 2604.07218 | [PDF](https://arxiv.org/pdf/2604.07218v1)

**作者:** Yuan-Zheng Lei `[一作]` (University of Maryland), Nii Attoh-Okine `[通讯]` (University of Maryland)

**通讯引用:** 3479 | [OpenAlex ID](https://openalex.org/A5035496450)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对车辆路径问题的可行性感知 QAOA 框架，利用约束感知初始化与混合 XY‑X 混合器来提升可行解的搜索效率。

**💡 创新点**

创新点在于：①在初始态中嵌入部分结构化的一热约束，显著缩小搜索空间；②设计了混合 XY‑X 混合器，既保持约束信息，又保留对 Hamming 量的探索能力。

**🔧 技术方法**

使用的技术包括 QAOA、QUBO 变换、约束编码、XY‑X 混合器、经典优化器（如 COBYLA）以及在噪声模型下的量子线路实现。

**📊 数据集**

实验使用的是一个三节点、两辆车的 toy VRP 实例（距离矩阵见论文），该实例可手工写出完整的 QUBO 与 Ising 形式。

**📈 对比分析**

通过理想状态向量、有限采样与噪声采样三种实验进行对比，结果显示提出的方法在可行解概率、期望能量差和采样排名三项指标上均优于标准 QAOA，尤其在无噪声和有限采样环境下优势更为显著。

**⚠️ 局限性**

局限性包括：在噪声环境下优势减弱，混合器和初始化的实现会增加量子门和编译复杂度；在更大规模 VRP 上的可扩展性尚需进一步验证。

---

## 521. Compact Constraint Encoding for LLM Code Generation: An Empirical Study of Token Economics and Constraint Compliance

**arXiv ID:** 2604.07192 | [PDF](https://arxiv.org/pdf/2604.07192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 522. Reshaping Inclusive Interpersonal Dynamics through Smart Glasses in Mixed-Vision Social Activities

**arXiv ID:** 2604.07232 | [PDF](https://arxiv.org/pdf/2604.07232v1)

**作者:** Jieqiong Ding `[一作]` (Tsinghua University), Yang Jiao `[通讯]` (Tsinghua University)

**通讯引用:** 5231 | [OpenAlex ID](https://openalex.org/A5050461950)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究开发了一种基于智能眼镜的系统CollabLens，旨在探索智能眼镜如何影响混合视觉社交活动中的人际动态，并支持盲人和低视力（BLV）个体与有视力的同伴的包容性合作。

**💡 创新点**

创新点在于通过智能眼镜提供实时的上下文支持，增强BLV参与者的自主性和控制力，同时减少对有视力同伴的依赖，促进更具互惠性的动态。

**🔧 技术方法**

使用了大型多模态模型（LMM）作为语音助手，能够实时解释视觉信息，并通过智能眼镜进行交互。

**📊 数据集**

研究中使用了四个混合视觉工作坊的会话，参与者包括8名BLV个体和8名有视力的个体，进行桌面游戏活动。

**📈 对比分析**

通过定量和定性分析比较了BLV和有视力参与者的体验，结果显示智能眼镜在促进包容性合作和积极的人际互动方面表现良好，且两组参与者在感知上没有显著差异。

**⚠️ 局限性**

限制在于研究主要集中在结构化的工作坊环境中，未能完全反映日常社交生活中的复杂动态，且样本仅来自单一学术机构，可能未能代表更广泛的人群。

---

## 523. Designing for Accountable Agents: a Viewpoint

**arXiv ID:** 2604.07204 | [PDF](https://arxiv.org/pdf/2604.07204v1)

**作者:** Stephen Cranefield `[一作]` (University of Otago), Nir Oren `[通讯]` (University of Aberdeen)

**通讯引用:** 2707 | [OpenAlex ID](https://openalex.org/A5003931008)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出面向可问责的多智能体系统（MAS）的概念框架和研究议程，系统性综述了跨学科的问责理论，并通过地震救援场景说明问责机制的潜在价值。

**💡 创新点**

创新点在于：① 将传统的组织层面问责概念迁移到开放型MAS，强调代理可以主动作为问责方（Accountor）或被问责方（Accountee）参与完整的问责生命周期；② 结合规范推理、对话游戏和学习框架，提出可追踪责任、记录、对话、判断和补救等环节的统一模型；③ 提出可用于验证和竞赛的问责测试平台构想，填补MAS社区对问责验证的空白。

**🔧 技术方法**

技术路线主要包括：基于规范的推理与合成（norm synthesis）、对话游戏与代理沟通语言、BDI或决策理论模型、强化学习与博弈论用于效用与激励设计，以及利用大型语言模型（LLM）辅助问责对话与解释。

**📊 数据集**

本文未使用具体数据集，提出的测试平台仅为概念性构想，未来可基于模拟或真实MAS场景进行实验。

**📈 对比分析**

本文并未给出实验比较或性能评估，讨论主要聚焦在理论可行性与实现挑战上。未来工作可通过构建问责测试bed和竞赛，使用标准指标（如问责完整度、解释质量、系统性能提升）进行比较。

**⚠️ 局限性**

限制：① 研究处于概念与理论阶段，缺乏实现与实证验证；② 开放型MAS的异构性和动态性导致问责模型复杂、难以统一；③ 对话与判断阶段的自动化与人类可解释性之间存在冲突；④ 缺少统一的评估指标与基准，难以量化问责效果。

---

## 524. Beyond the Mean: Modelling Annotation Distributions in Continuous Affect Prediction

**arXiv ID:** 2604.07198 | [PDF](https://arxiv.org/pdf/2604.07198v1)

**作者:** Kosmas Pinitas `[一作]` (University of Piraeus), Ilias Maglogiannis `[通讯]` (University of Piraeus)

**通讯引用:** 7031 | [OpenAlex ID](https://openalex.org/A5013025849)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在情绪持续预测中，本文提出一种基于 Beta 分布的框架，通过仅预测标注者平均值和标准差来重构完整的注释分布，从而捕获情绪感知的中心趋势、变异性、偏度、峰度等高阶特征。

**💡 创新点**

创新点在于利用 Beta 分布的可逆矩匹配方法，既能轻量化预测仅两参数，又能闭式求得多阶分布描述，显著提升了对注释者不一致性的表征能力。

**🔧 技术方法**

采用两层 ReLU 轻量网络进行特征到 (μ,σ) 的映射，使用矩匹配推导 β 参数，并利用 KL 散度、CCC 等指标评估分布拟合度；同时对比了独立回归基线。

**📊 数据集**

实验数据集包括 RECOLA（23 参与者、6 名注释者）和 SEWA（398 参与者、3 名注释者）的持续情绪（价值-激活）标注。

**📈 对比分析**

与传统点估计回归模型相比，Beta 模型在 CCC 上基本持平或略优，且在 KL 散度上显著逼近真实注释分布，证明其在保留不确定性信息方面具有优势。

**⚠️ 局限性**

局限性包括：Beta 分布仅能表示单峰或 U 形分布，可能无法捕捉多峰或更复杂的注释分布；未对注释者身份或可靠性进行建模；以及缺乏时序建模，未充分利用情绪动态信息。

---

## 525. Efficient Learned Data Compression via Dual-Stream Feature Decoupling

**arXiv ID:** 2604.07239 | [PDF](https://arxiv.org/pdf/2604.07239v1)

**作者:** Huidong Ma `[一作]` (Nankai University), Wentong Cai `[通讯]` (Nanyang Technological University)

**通讯引用:** 6458 | [OpenAlex ID](https://openalex.org/A5036580072)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为FADE的学习式无损压缩方法，通过双流多尺度解耦、层级门控细化以及并行流水线实现压缩率与吞吐量的双提升。

**💡 创新点**

创新点在于：1）Dual-Stream Multi-Scale Decoupler将微句法与宏语义特征并行提取；2）Hierarchical Gated Refiner实现实例适应的分层门控细化；3）Concurrent Stream-Parallel Pipeline突破自回归瓶颈，实现全流水线并行。

**🔧 技术方法**

使用自回归概率预测框架，结合CNN+MLP双流网络、GeGLU+Rolling Cache、层级门控机制、双缓冲线程安全的并行管线以及数据并行微步技术。

**📊 数据集**

在七类多域数据集上进行评估，包括Enwik9、LJSpeech、TestImages、UVG、CESM、DNACorpus和Silesia（文本、音频、图像、视频、数值、基因等）。

**📈 对比分析**

与10种基线（4传统压缩+6先进LDC）进行压缩率和吞吐量对比，FADE在7个数据集上平均压缩率3.716、吞吐量最高4347 KB/min，压缩率与吞吐量均位列第一，且GPU内存占用与延迟最低。

**⚠️ 局限性**

局限性：侧重压缩率和吞吐量，导致模型参数和FLOPs略高于某些基线，但未影响实际效率；缺乏对极低资源设备或极限压缩率的评估，且主要针对CPU–GPU异构匹配，未考虑其他硬件平台。

---

## 526. Robust Quadruped Locomotion via Evolutionary Reinforcement Learning

**arXiv ID:** 2604.07224 | [PDF](https://arxiv.org/pdf/2604.07224v1)

**作者:** Brian McAteer `[一作]` (University of Galway), Karl Mason `[通讯]` (University of Galway)

**通讯引用:** 957 | [OpenAlex ID](https://openalex.org/A5088726212)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

对比深度强化学习与进化强化学习在四足机器人平地与粗糙地形上的行走策略，评估其泛化能力。

**💡 创新点**

将交叉熵方法与TD3/DDPG结合的ERL框架，证明进化搜索可提升跨域鲁棒性。

**🔧 技术方法**

使用Deep Deterministic Policy Gradient、Twin-Delayed DDPG以及两种Cross‑Entropy基因演化变体。

**📊 数据集**

采用CoppeliaSim中自建的平地与随机粗糙地形仿真环境作为实验数据集。

**📈 对比分析**

通过在训练域和未见粗糙地形上分别进行10次rollout比较，结果显示CEM‑TD3在粗糙地形上的平均奖励远高于纯梯度方法。

**⚠️ 局限性**

主要局限在奖励函数对速度的高权重导致粗糙地形奖励偏高，且未在真实硬件上验证。

---

## 527. TraceSafe: A Systematic Assessment of LLM Guardrails on Multi-Step Tool-Calling Trajectories

**arXiv ID:** 2604.07223 | [PDF](https://arxiv.org/pdf/2604.07223v1)

**作者:** Yen-Shan Chen `[一作]` (CyCraft AI Lab), Yun-Nung Chen `[通讯]` (National Taiwan University)

**通讯引用:** 3923 | [OpenAlex ID](https://openalex.org/A5076610826)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并构建了 TraceSafe‑Bench，用于评估多步骤代理工作流程中的中途执行轨迹安全。

**💡 创新点**

通过 Benign‑to‑Harmful 编辑方式生成具有精准步骤级别风险标注的静态轨迹，并揭示结构解析能力是轨迹安全的核心瓶颈。

**🔧 技术方法**

采用结构化编辑算法、12 类风险分类体系、对比评估（13 个 LLM‑as‑a‑guard + 7 专门防护），并通过多种评估模式（二分类、粗细粒度多分类）与相关性分析技术。

**📊 数据集**

以 Berkeley Function Calling Leaderboard (BFCL) 的多步轨迹为种子，经过编辑生成超过 1,000 个包含 12 种风险类别的轨迹，构成 TraceSafe‑Bench 数据集。

**📈 对比分析**

使用分类准确率、拒绝率和均衡平均准确率进行对比；结果显示一般 LLM 在二分类上偏高拒绝，专用防护偏低拒绝；粗粒度分类提升检测稳定性；细粒度分类对结构明显风险检测高，但对界面不一致等细微风险表现较差；模型性能与 RAGTruth 等结构化输入理解任务高度相关。

**⚠️ 局限性**

局限在于仅评估静态轨迹，未覆盖动态交互环境；对细粒度风险的识别仍不足；编辑过程依赖规则，可能忽视更隐蔽或非结构化的攻击方式。

---

## 528. A Systematic Study of Retrieval Pipeline Design for Retrieval-Augmented Medical Question Answering

**arXiv ID:** 2604.07274 | [PDF](https://arxiv.org/pdf/2604.07274v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 529. HIVE: Query, Hypothesize, Verify An LLM Framework for Multimodal Reasoning-Intensive Retrieval

**arXiv ID:** 2604.07220 | [PDF](https://arxiv.org/pdf/2604.07220v1)

**作者:** Mahmoud Abdalla `[一作]` (Chungbuk National University), Hyun-Soo Kang `[通讯]` (Chungbuk National University)

**通讯引用:** 693 | [OpenAlex ID](https://openalex.org/A5018214137)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 HIVE（Hypothesis-driven Iterative Visual Evidence Retrieval）框架，利用大型语言模型在检索过程中生成补偿查询并进行验证重排，以提升多模态检索的推理能力。

**💡 创新点**

创新点在于：①将 LLM 作为显式视觉推理器，通过观察检索到的候选文档和图像描述生成补偿查询；②采用四阶段递归检索（初检索、补偿查询合成、二次检索、验证重排），实现无须额外训练的插件式改进；③在不改变基检索器的前提下显著提升多模态检索性能。

**🔧 技术方法**

技术手段包括：多模态稠密检索器（可使用 CLIP、GME、Nomic-Vision、HIVE-Retriever 等）；GPT‑4o 进行图像描述、补偿查询生成和验证重排；基于余弦相似度的检索与 LLM 推理的结合；对候选集进行多轮检索和 LLM 验证。

**📊 数据集**

使用 MM‑BRIGHT 公开基准（2,803 个多模态查询，涵盖 29 个技术领域）进行实验。

**📈 对比分析**

与最强多模态基线（Nomic‑Vision 27.6）和最强文本基线（DiVeR 32.2）对比，HIVE 在聚合 nDCG@10 上达 41.7，分别高出 14.1 点和 9.5 点；在 28/29 个领域均优于所有基线，且在视觉信息密集的领域（如 Gaming、Economics、Sustainability）提升尤为显著。

**⚠️ 局限性**

局限性包括：①依赖 GPT‑4o，成本高且受限于调用次数；②在视觉信息相对抽象或信息量有限的领域（如 Quantum Computing、Cryptography）提升有限；③目前仅支持单图像到文本检索，未覆盖多图像或视频场景；④缺乏对 LLM 生成补偿查询质量的自动评估与自适应控制。

---

## 530. Why teaching resists automation in an AI-inundated era: Human judgment, non-modular work, and the limits of delegation

**arXiv ID:** 2604.07285 | [PDF](https://arxiv.org/pdf/2604.07285v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 531. Mixture Proportion Estimation and Weakly-supervised Kernel Test for Conditional Independence

**arXiv ID:** 2604.07191 | [PDF](https://arxiv.org/pdf/2604.07191v1)

**作者:** Yushi Hirose `[一作]` (Institute of Science Tokyo), Takafumi Kanamori `[通讯]` (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于条件独立性（CI）和多元条件独立性（MCI）的混合比例估计方法，并构造弱监督核检验来验证这些假设。

**💡 创新点**

用CI/MCI替代传统不可分解性假设，得到可辨识的估计器并证明其渐近正态性，同时提出能够仅用无标签数据进行条件独立性检验的WsKCI和WsKMCI方法。

**🔧 技术方法**

主要技术包括矩方法估计、核Hilbert-Schmidt独立性检验（HSIC）与核条件独立性检验（KCI）、核岭回归、黄金分割搜索以及伽玛近似来估计统计量分布。

**📊 数据集**

实验使用合成高斯数据以及UCI公开数据集（Shuttle、Wine、Dry Bean、Breast Cancer 等）进行验证。

**📈 对比分析**

与传统基于不可分解性假设的 MPE 算法（DEDPUL、EN、KM2）对比，实验显示在 CI 条件下我们的方法误差显著下降，估计精度明显优于现有方法。

**⚠️ 局限性**

局限包括对函数 g1、g2 的选择敏感、在高维特征下多重检验问题以及 MCI 检验在样本量小、分离度低时功效有限。

---

## 532. CADENCE: Context-Adaptive Depth Estimation for Navigation and Computational Efficiency

**arXiv ID:** 2604.07286 | [PDF](https://arxiv.org/pdf/2604.07286v1)

**作者:** Timothy K Johnsen `[一作]` (San Diego State University), Marco Levorato `[通讯]` (University of California Irvine)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CADENCE系统，动态调节单目深度估计网络的计算量，以实现导航与能耗效率提升。

**💡 创新点**

首次将可缩放的MDE网络与统一的导航-适配策略结合，并通过DRL实时预测动作与缩放因子，实现上下文感知的计算调节。

**🔧 技术方法**

使用Slimmable CNN、双重DQN强化学习、AirSim仿真、Jetson Orin Nano硬件加速、switch batch normalization以及滑动FIFO等技术。

**📊 数据集**

基于Microsoft AirSim生成的RGB‑Depth对（约20,000对）进行训练、验证与测试，使用AirSimNH地图进行实验。

**📈 对比分析**

与静态SoTA MDE网络对比，CADENCE在能耗下降75%、推理延迟下降74.8%、功耗降低16.1%的同时，导航准确率提升7.43%。

**⚠️ 局限性**

实验仅在仿真与HIL环境下验证，缺乏真实飞行测试；训练耗时长；对复杂环境的泛化能力待进一步验证；仅针对单目深度估计。

---

## 533. Symbolic Polyhedral-Based Energy Analysis for Nested Loop Programs

**arXiv ID:** 2604.07287 | [PDF](https://arxiv.org/pdf/2604.07287v1)

**作者:** Avinash Mahesh Nirmala `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Jürgen Teich `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 11698 | [OpenAlex ID](https://openalex.org/A5076672029)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种符号化的能量分析框架，用于估计嵌套循环程序在并行处理器阵列加速器上映射与调度后的能耗。

**💡 创新点**

创新点在于将循环体的访问量与计算量转化为多维多项式体积的闭式符号表达式，利用 Barvinok 算法与整数集合库实现无规模依赖的能耗估计，完全避免了逐周期仿真。

**🔧 技术方法**

技术包括：多维多项式体积计算（Barvinok/ISL）、多维 polyhedral 编译模型、映射与调度的符号化表达、预先表征的访问/运算能量权重。

**📊 数据集**

使用 PolyBench 基准套件中的嵌套循环 kernel（如 GESUMMV、GEMM 等）进行实验验证。

**📈 对比分析**

通过与基于 XML 的 TCPA cycle‑accurate 仿真对比，符号化方法在所有问题规模下均保持 <0.5 s 的恒定分析时间，而仿真时间随矩阵尺寸呈指数增长；两者在访问计数与总能耗上完全一致。

**⚠️ 局限性**

局限性包括：需预先测定每种内存/寄存器访问的能量消耗；对非多项式或高度不规则的访问模式适用性有限；主要针对已知映射与调度规则的阵列加速器，未在真实硬件上进行验证。

---

## 534. The Random Subsequence Model and Uniform Codes for the Deletion Channel

**arXiv ID:** 2604.07234 | [PDF](https://arxiv.org/pdf/2604.07234v1)

**作者:** Ryan Jeong `[一作]` (Stanford University), Francisco Pernice `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了植入随机子序列模型（Planted Random Subsequence Model）及其在二进制删除通道中的均匀随机码的容量，推导了植入模型的自由能下界并给出了完整的定量证明；

**💡 创新点**

创新点在于提出并严格证明了植入模型的quenched自由能与均匀码容量之间的非平凡Jensen间隙，并为植入模型给出了精确的annealed自由能解析式；

**🔧 技术方法**

主要技术包括自监督复制（Nishimori）身份、组合计数、标准差型近似、变分公式、拉格朗日乘子法、生成函数与残差计算等；

**📊 数据集**

该研究为纯理论分析，无需使用外部数据集；

**📈 对比分析**

与已有的随机编码下删码通道容量下限（如先前的非严格正下限）相比，本文得到更严格、可显式表达的正下限，证明了均匀随机码在所有删除率p<1下均能实现正比率；

**⚠️ 局限性**

限制在于仍无法得到植入模型的quenched自由能的精确解析式，导致仅能得到保守的容量下限；此外，方法对高删除率（p接近1）时的精度不足。

---

## 535. Designing Safe and Accountable GenAI as a Learning Companion with Women Banned from Formal Education

**arXiv ID:** 2604.07253 | [PDF](https://arxiv.org/pdf/2604.07253v1)

**作者:** Hamayoon Behmanush `[一作]` (Saarland University), Vikram Kamath Cannanure `[通讯]` (Saarland University)

**通讯引用:** 174 | [OpenAlex ID](https://openalex.org/A5053796170)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过参与式设计方法，邀请20名在阿富汗被禁止接受正规教育的女性，进行线上研讨、故事板创作与前后期望问卷，探讨她们如何将生成式人工智能（GenAI）视作学习伴侣以及其安全与可追溯性需求。

**💡 创新点**

创新点在于：①首次在性别受限环境中系统化研究GenAI作为学习伙伴的安全、责任与适应性；②将“曝光控制”与隐私、监控风险、文化契合度紧密结合，提出专为低资源、监控环境设计的五大责任导向设计方向；③通过参与式展望显著提升参与者的学习愿景、代理感与可行路径。

**🔧 技术方法**

主要技术方法包括：参与式设计工作坊、前后期望量表（适应性提升量表）与配对t检验、主题分析（基于故事板与访谈录音），以及对生成式AI在文稿润色中的辅助使用（非核心研究内容）。

**📊 数据集**

数据集主要为：① 140人招募问卷（人口学、技术使用）；② 20人参与式设计的前后期望问卷；③ 录音文本、故事板与情景稿。

**📈 对比分析**

比较方法：采用配对t检验评估前后期望、代理感与途径感的变化，结果均显著提升（p≤0.01）；未与现有GenAI工具或基线系统进行功能/性能对比，仅通过主观问卷和定性反馈展示设计改进的潜在价值。

**⚠️ 局限性**

局限性包括：样本量小、仅覆盖阿富汗特定性别受限情境，结果可能缺乏普适性；依赖自报数据，易受社会期望偏差影响；短期干预，缺乏长期追踪验证；未对技术实现细节（如模型准确性、延迟等）进行客观评估。

---

## 536. Weaves, Wires, and Morphisms: Formalizing and Implementing the Algebra of Deep Learning

**arXiv ID:** 2604.07242 | [PDF](https://arxiv.org/pdf/2604.07242v1)

**作者:** Vincent Abbott `[一作]`, Gioele Zardini `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

构建了一个基于范畴论的通用框架，用以在符号、图形和代码层面统一描述深度学习模型的构造与组合。

**💡 创新点**

创新点在于将集合、可测空间与Markov过程等不同数学背景统一映射到“对称单子范畴”，并引入“元素范畴”与可重排映射来区分功能性、概率性与线性三类范畴。

**🔧 技术方法**

采用范畴论中的对称单子范畴、重排（rearrangement）与块（block）等概念，结合Python代码实现了对象、态射、组合、并行与重排的具体类与方法。

**📊 数据集**

无（论文为理论性框架描述，无实验数据集）。

**📈 对比分析**

无实验对比，论文通过数学定义和代码实现演示了该框架在不同范畴中的一致性与可扩展性。

**⚠️ 局限性**

局限在于缺乏针对实际深度学习任务的性能评估与实现细节的深度验证，且对复杂模型的可视化与自动化支持仍需进一步开发。

---

## 537. PhyEdit: Towards Real-World Object Manipulation via Physically-Grounded Image Editing

**arXiv ID:** 2604.07230 | [PDF](https://arxiv.org/pdf/2604.07230v1)

**作者:** Ruihang Xu `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**通讯引用:** 81574 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于Diffusion Transformer的图像编辑框架PhyEdit，利用显式3D先验和2D-3D联合监督实现物理精确的3D空间对象操控。

**💡 创新点**

创新点在于将3D几何变换模块作为可插拔的视觉引导，并通过深度监督提升空间一致性；同时构建了真实对照图像+深度的RealManip-10K数据集与ManipEval评测基准。

**🔧 技术方法**

采用Diffusion Transformer、3D场景预测模型（如Depth‑Anything‑3）、LoRA微调、SILog深度损失等技术。

**📊 数据集**

使用RealManip‑10K数据集（含真实图像对、深度、遮罩和3D坐标）。

**📈 对比分析**

与多种开源与闭源基线对比，Ph­yEdit在DIoU、Chamfer、RA‑DINO等多维度指标上均显著优于Nano Banana Pro等强对手，整体性能领先。

**⚠️ 局限性**

局限性包括：对文本指令的3D意图捕捉仍受限，需借助参考图像；在极端光照或遮挡条件下仍可能出现误差。

---

## 538. Multiprotocol Wireless Timer Synchronization for IoT Systems

**arXiv ID:** 2604.07199 | [PDF](https://arxiv.org/pdf/2604.07199v1)

**作者:** Ziyao Zhou `[一作]` (Nanyang Technological University), Hen-Wei Huang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2225 | [OpenAlex ID](https://openalex.org/A5029613558)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出一种协议无关的无线定时器同步框架，利用多协议时隙（MPSL）发射精准时间戳信标，实现纳秒级同步精度。

**💡 创新点**

创新点在于将同步过程从上层协议栈解耦，借助硬件定时器、PPI触发和专有无线模式，在无需重传或延迟的情况下实现双向实时同步，并兼容多种无线标准。

**🔧 技术方法**

采用硬件定时器、PPI触发、Nordic nRF52832芯片、MPSL时隙管理、BLE栈以及自定义无线信标协议。

**📊 数据集**

实验使用nRF52DK开发板在不同同步频率、RSSI、BLE连接间隔和吞吐量下收集数据，形成本研究的实验数据集。

**📈 对比分析**

与CheepSync、BlueSync、传统BLE GATT、神经网络和离线校正方法对比，本文方法实现约22 ns精度，明显优于微秒级误差的现有方案，且保持双向通信能力。

**⚠️ 局限性**

局限性包括仅在两节点nRF52平台验证，未测试大规模网络下的同步稳定性和功耗，且依赖专有无线模式，导致跨平台部署时需要额外硬件支持。

---

## 539. Diffusion Processes on Implicit Manifolds

**arXiv ID:** 2604.07213 | [PDF](https://arxiv.org/pdf/2604.07213v1)

**作者:** Victor Kawasaki-Borruat `[一作]` (EPFL), Adam Gosztolai `[通讯]` (Medical University of Vienna)

**通讯引用:** 221 | [OpenAlex ID](https://openalex.org/A5074584251)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了仅利用点云数据构建隐式流形上的扩散过程的框架——Implicit Manifold-valued Diffusions (IMDs)，并给出了理论路径空间收敛性证明与数值实现。

**💡 创新点**

创新点在于：①在没有显式流形几何的情况下，通过邻接图估计扩散的无穷生成元和Carré-du-Champ，得到数据驱动的流形扩散；②证明该过程在样本无限大时在路径空间一致收敛到光滑流形扩散；③提出可行的Euler–Maruyama数值方案，并引入可选的DRGD纠正步骤。

**🔧 技术方法**

使用的技术包括：图拉普拉斯算子（随机行走图拉普拉斯）、Carré-du-Champ估计、SDE理论、欧拉-马鲁雅马积分、Score-based generative model辅助的重投影（DRGD）以及图论与谱分析方法。

**📊 数据集**

实验使用了合成数据（环面、球面、von Mises-Fisher分布）以及真实数据集MNIST。

**📈 对比分析**

与传统基于Score的重投影Langevin扩散以及理论的球面Brownian过程进行对比；在几何精度、统计分布一致性（终点分布）以及跨模态探索（模式连通性）方面均优于传统方法；数值误差随步长减小而下降，DRGD能在高维中控制误差。

**⚠️ 局限性**

局限性包括：图拉普拉斯算子计算受高维维度灾难影响；缺乏非渐近的样本误差界；仅在光滑无噪声的流形假设下适用；对大规模数据的可扩展性仍待改进。

---

## 540. Graph Neural ODE Digital Twins for Control-Oriented Reactor Thermal-Hydraulic Forecasting Under Partial Observability

**arXiv ID:** 2604.07292 | [PDF](https://arxiv.org/pdf/2604.07292v1)

**作者:** Akzhol Almukhametov `[一作]` (Texas A&M University), Yang Liu `[通讯]` (Texas A&M University)

**通讯引用:** 34247 | [OpenAlex ID](https://openalex.org/A5100356037)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于物理信息化图神经网络与连续时间神经ODE（GNN‑ODE）的数字孪生，用于在部分可观测条件下实时预测核反应堆热水力状态，并能恢复未装传感器位置的温度；

**💡 创新点**

创新点在于将图神经网络的空间消息传递与连续时间神经ODE相结合，构建可直接编码管道流动与热交换拓扑的模型；引入拓扑引导缺失节点初始化；使用层级学习率微调实现从高保真仿真到实验的快速迁移，并在保持能量守恒的同时学习可解释的流动依赖热传递系数；

**🔧 技术方法**

采用的技术包括图神经网络、控制化神经ODE、Physics‑Informed Message Passing、梯度可微RK4积分、层级学习率、教师强制与自回归训练、并行蒙特卡洛集成推理、Savitzky‑Golay滤波等；

**📊 数据集**

使用的训练数据集为基于SAM（System Analysis Module）一维热水力模拟器生成的约1,000条仿真序列（含扰动与拉普拉斯采样）、50条边缘案例用于预训练，以及实验设施记录的29个传感器序列（共30条5分钟转移序列）用于微调；还使用实验硬件不确定性分布做不确定性量化；

**📈 对比分析**

通过与SAM仿真直接对比评估：MAE在60 s预测时为0.91 K、300 s为2.18 K；未观测节点上R²可达0.995；推理速度约105×快于模拟，单GPU即可实现64成员集成推理；在实验数据上的微调后，模型在多步功率跃迁（0→10→5→7 kW）中观测节点误差≤几度，未观测节点保持平滑；与传统离散时序RNN/GRU等模型相比，表现显著提升；

**⚠️ 局限性**

局限性包括未对高阶多物理耦合（如点核动力学）建模；受限于训练数据噪声，需要滤波与预处理；对拓扑改变的泛化能力待验证；模型对极端扰动的鲁棒性有限；大规模集成推理仍需GPU或多线程优化；微调时需防止过拟合。

---

## 541. Multiple Planted Structures Below $\sqrt{n}$: An SoS Integrality Gap and an SQ Lower Bound

**arXiv ID:** 2604.07278 | [PDF](https://arxiv.org/pdf/2604.07278v1)

**作者:** Matvey Mosievskiy `[一作]` (University of Illinois Chicago), Lev Reyzin `[通讯]` (University of Illinois Chicago)

**通讯引用:** 1939 | [OpenAlex ID](https://openalex.org/A5081065957)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在随机背景下嵌入多块独立植入结构（如k-团或k×k二分团）的平均情况推理问题，给出了在总植入大小K=kt小于√n时，现有高效算法无法检测或验证这些结构的理论下界；

**💡 创新点**

创新点在于：1）将单植入的Sum-of-Squares（SoS）整合度缺口扩展到多植入情形，证明即便植入多块小团，SoS仍无法在K≲√n的范围内上界；2）在统计查询（SQ）框架下给出针对等大多植入二分团的下界，克服了传统分区嵌入导致的阈值降低问题；

**🔧 技术方法**

核心技术包括：多标签变量的伪期望构造与截断、基于傅里叶展开的校准、Ribbons分离与中间矩阵正定性证明、以及SQ统计维数（statistical dimension）与平均相关性分析；

**📊 数据集**

该工作为理论性研究，未使用实际数据集，而是依赖随机图模型G(n,1/2)和随机二分图的抽样；

**📈 对比分析**

对比方面：论文证明在K=kt≤n^1/2−O(√(d/log n))时，任意度为d的SoS解都无法在高概率下证明不存在t个k-团；SQ下界表明当kt=O(n^1/2−δ)时，任何多项式时间SQ算法需要指数级查询，说明检测难度接近信息论阈值；

**⚠️ 局限性**

局限性包括：仅给出了下界，未给出对应上界或实际算法；分析局限于k≪√n≪kt的区间；多植入结构需严格等大，否则阈值可能不同；

---

## 542. Joint Optimization of Reasoning and Dual-Memory for Self-Learning Diagnostic Agent

**arXiv ID:** 2604.07269 | [PDF](https://arxiv.org/pdf/2604.07269v1)

**作者:** Bingxuan Li `[一作]` (University of Illinois Urbana-Champaign), Yue Guo `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 21443 | [OpenAlex ID](https://openalex.org/A5067408026)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于双记忆（短期病例记忆与长期规则记忆）的自学习诊断智能体，能够在诊断过程中主动存储、提炼并复用经验；

**💡 创新点**

创新点在于将认知启发的双记忆结构与强化学习框架耦合，实现诊断推理与记忆管理的联合优化，突破传统单独处理病例的限制；

**🔧 技术方法**

技术上采用大语言模型（Qwen3-4B/8B 等）作为基线，构建可执行记忆操作的策略网络，并使用结构化奖励函数与回合级优势估计训练 RL；

**📊 数据集**

使用的评估数据集包括 MedCaseReasoning（标准评估）和 ER-Reason（长周期在线学习评估），均通过精心设计的候选诊断集合进行测试；

**📈 对比分析**

与零样本、监督微调、单一奖励 RL 及传统记忆增强方法对比，实验显示在 MedCaseReasoning 上准确率达 92.46%（+19.6%）且在 ER-Reason 上最终准确率 0.7214（ΔAcc@100 +0.35），显著优于基线；

**⚠️ 局限性**

局限性包括：规则生成的案例相关性较低（需人工评估验证）、记忆管理在某些情形下可能引入噪声导致性能波动，以及对不同模型规模和域的适用性仍需进一步验证。

---

## 543. BATON: A Multimodal Benchmark for Bidirectional Automation Transition Observation in Naturalistic Driving

**arXiv ID:** 2604.07263 | [PDF](https://arxiv.org/pdf/2604.07263v1)

**作者:** Yuhang Wang `[一作]` (University of South Florida), Hao Zhou `[通讯]` (University of South Florida)

**通讯引用:** 30533 | [OpenAlex ID](https://openalex.org/A5100344489)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了BATON数据集和基准，用于研究真实道路驾驶中司机与自动驾驶系统的双向控制转移。

**💡 创新点**

首次同步采集前视视频、车内视频、CAN、雷达和GPS等多模态数据，并设计统一的三任务基准（动作理解、交接预测、接管预测），支持跨司机评估。

**🔧 技术方法**

使用规则标注、序列模型（GRU、TCN）、树模型（XGBoost）、多模态融合策略以及零样本视觉语言模型进行实验。

**📊 数据集**

主要使用自研的BATON数据集（136.6小时、380条路线、127司机、2892事件），并与Drive&Act、DAD、AIDE、manD等公开数据集进行对比。

**📈 对比分析**

在跨司机拆分上，XGBoost/GRU在动作理解任务Macro-F1达到0.91，交接预测AUPRC为0.463，接管预测AUPRC为0.468；VLM基线表现低于训练模型，单模视频预测效果差。

**⚠️ 局限性**

存在BEV周围车辆视角缺失、驾驶时长分布不均以及基线融合方式过于简单，限制了对多模态潜力的充分挖掘。

---

## 544. A comparative analysis of machine learning models in SHAP analysis

**arXiv ID:** 2604.07258 | [PDF](https://arxiv.org/pdf/2604.07258v1)

**作者:** Justin Lin `[一作]` (Indiana University), Julia Fukuyama `[通讯]` (Indiana University)

**通讯引用:** 3009 | [OpenAlex ID](https://openalex.org/A5050341558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究SHAP解释方法在不同机器学习模型（决策树、XGBoost、神经网络）和数据集（模拟数据、MNIST图像、ADNI临床数据）上的表现，并提出多分类高维瀑布图来可视化SHAP向量

**💡 创新点**

首次将高维瀑布图扩展到多分类问题并通过SHAP向量聚类实现亚群发现，揭示不同模型决策过程与预测原因的差异

**🔧 技术方法**

SHAP值计算（TreeSHAP、Kernel SHAP）、UMAP降维、HDBSCAN聚类、PCA可视化、CNN、XGBoost、决策树、神经网络

**📊 数据集**

三种数据集：人工模拟数据（10维10类）、MNIST手写数字（784维图像）和ADNI阿尔茨海默症临床/影像/基因数据（39维）

**📈 对比分析**

通过对比三模型的预测准确率、SHAP值分布和聚类结果，发现XGBoost和神经网络在复杂任务上表现最好，SHAP向量聚类可揭示亚群，性能提升显著但可视化效果受降维方法限制

**⚠️ 局限性**

主要局限在于降维方法（如UMAP）对聚类位置的影响、CNN SHAP向量不一致导致可视化困难，以及模型解释的普适性不足

---

## 545. Geo-EVS: Geometry-Conditioned Extrapolative View Synthesis for Autonomous Driving

**arXiv ID:** 2604.07250 | [PDF](https://arxiv.org/pdf/2604.07250v1)

**作者:** Yatong Lan `[一作]` (Tsinghua University), Lei He `[通讯]` (Tsinghua University)

**通讯引用:** 32075 | [OpenAlex ID](https://openalex.org/A5102798483)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对自动驾驶稀疏监督下的外推视图合成框架 Geo-EVS，旨在从多种传感器视角生成统一的虚拟视图。

**💡 创新点**

创新点包括：① Geometry‑Aware Reprojection (GAR) 统一训练与推理的几何投影路径；② Artifact‑Guided Latent Diffusion (AGLD) 在训练时注入投影缺陷掩码以提升缺失支持下的结构恢复；③ LiDAR‑Projected Sparse‑Reference (LPSR) 评估协议，专门处理无全景监督的外推场景。

**🔧 技术方法**

采用 VGGT 进行几何特征提取与点云构造；利用 Latent Diffusion 模型（含分类器无关引导）进行条件图像生成；在 GAR 中实现深度投影、z‑buffer 光栅化；在 AGLD 中通过预计算的缺陷掩码实现结构鲁棒性训练。

**📊 数据集**

主要使用 Waymo Open Dataset（mini‑1/5 训练/验证/测试分割）进行训练与评估。

**📈 对比分析**

与 3DGS、Street Gaussians、EmerNeRF、FreeVS 等基线在：① 在轨视图 FID 仅 3.9（优于 14.62 的 FreeVS）；② 外推视图稀疏 PSNR 23.65、SSIM 0.941（均优于 3DGS、FreeVS 等）；③ 在 3D 检测任务中，使用生成视图后 mAP +1.0%、mAPH +0.8%、L1/mAPH +1.3%。

**⚠️ 局限性**

局限性包括：① 动态物体在不同视角下位置不一致导致生成不连贯；② 极端稀疏支持（<5% 有效像素）时模型易过度推断；③ 纹理在遮挡边界附近易出现泄漏；④ 缺乏时间一致性与多帧几何融合。

---

## 546. How Much LLM Does a Self-Revising Agent Actually Need?

**arXiv ID:** 2604.07236 | [PDF](https://arxiv.org/pdf/2604.07236v1)

**作者:** Seongwoo Jeong `[一作]` (Independent Researcher), Seonil Son `[通讯]` (RLWRLD.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过声明性反射运行时协议，将LLM代理的世界建模、规划、反思与稀疏LLM修订外部化，并在噪声协同Battleship游戏中对各层进行经验拆解。

**💡 创新点**

提出一种声明性反射运行时协议，使得LLM内部的预测、信心跟踪、符号反思和修订可以被显式化、可观测化，从而实现对LLM介入量的可度量拆解。

**🔧 技术方法**

使用非图灵完备DSL构建声明式运行时、MCMC后验推理、符号规划、符号反思预设以及稀疏LLM修订，构成四层代理体系。

**📊 数据集**

采用噪声协同Battleship（8×8、14船格子、40发射、15提问、ε=0.1）自生成的54局游戏（18板×3种子）进行实验。

**📈 对比分析**

将greedy+MCMC基线、WMA（世界建模规划）、MRA（符号反思）和MRA-LLM（稀疏LLM修订）四种代理在F1、胜率、提问数和LLM调用率上进行对照；结果显示：世界建模提升约+24.1pp胜率；符号反思对整体性能无显著提升；稀疏LLM修订略升F1但略降胜率。

**⚠️ 局限性**

实验仅在Battleship单一域进行，使用自制棋盘和MCMC后验，样本量有限；符号反思预设未校准；LLM修订效果表现非单调；缺乏跨域验证。

---

## 547. On the Price of Privacy for Language Identification and Generation

**arXiv ID:** 2604.07238 | [PDF](https://arxiv.org/pdf/2604.07238v1)

**作者:** Xiaoyu Li `[一作]` (University of New South Wales), Junbin Gao `[通讯]` (University of Sydney)

**通讯引用:** 11365 | [OpenAlex ID](https://openalex.org/A5015817857)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了差分隐私（DP）在语言识别和生成中的基本成本，建立了算法和匹配的下界，精确量化隐私成本。

**💡 创新点**

研究表明，在近似DP下，隐私成本几乎为零，而在纯DP下，收敛指数因子降级为min{1,}，这一结果是紧的。

**🔧 技术方法**

使用了差分隐私算法，包括指数机制和高斯机制来实现语言识别和生成的隐私保护。

**📊 数据集**

使用了来自未知分布的i.i.d.样本，具体数据集未明确说明，但涉及到语言集合和样本数据。

**📈 对比分析**

与非隐私算法相比，近似DP在语言识别和生成中恢复了非隐私的错误率，而纯DP则在收敛指数上有显著降级，表明近似DP在这两项任务中没有隐私成本。

**⚠️ 局限性**

本研究的局限性在于其信息理论性质，未考虑计算效率，且算法需要明确列举语言和元素，未来需要将这些保证与实际的DP-SGD训练管道结合。

---

## 548. Robust Hybrid Beamforming with Liquid Crystal Antennas and Liquid Neural Networks

**arXiv ID:** 2604.07219 | [PDF](https://arxiv.org/pdf/2604.07219v1)

**作者:** Xinquan Wang `[一作]` (New York University), Theodore S. Rappaport `[通讯]` (New York University)

**通讯引用:** 71079 | [OpenAlex ID](https://openalex.org/A5058193542)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于可重构液晶天线和液态神经网络（LNN）的混合波束成形框架，用于 sub‑THz 多用户 MIMO 系统的下行链路。

**💡 创新点**

创新点在于：①使用液晶天线实现无损耗的模拟波束控制，消除传统相位移器瓶颈；②引入基于 ODE 的 LNN 并配合 Sigmoid 门控与流形优化，显著提升对时变信道的自适应与对不完美 CSI 的鲁棒性；③将模拟波束与数字波束分离设计，形成两阶段优化流程。

**🔧 技术方法**

技术方法包括：液晶可重构天线代码书、离散化模拟波束选择、流形投影压缩搜索空间、基于连续时间 ODE 的液态神经网络、Adam 优化与对数损失。

**📊 数据集**

使用数据集：108 GHz 现场特定 NYURay 光线追踪生成的通道，覆盖布鲁克林 MetroTech Commons 区域。

**📈 对比分析**

与 LAGD、GRU 以及 3GPP TR 38.901 天线模型做基准比较。结果显示：在 CEE = ‑10 dB 时，LNN+LC 天线比 LAGD 提升 88.6% 的谱效率，且比 3GPP 天线高 1.9 倍；在 CEE 从 ‑20 dB 增至 0 dB 时，LNN 的 SE 下降仅 31.7%，远优于 LAGD 的 55.4%。

**⚠️ 局限性**

局限性：液晶天线响应时间慢、相位调节范围有限；实验验证仅在 108 GHz 与 48 元素阵列上完成，缺乏实测验证；用户 SE 分布、不同用户数与天线规模的深入分析未展开；模型对其他频段或更大规模部署的泛化性待进一步评估。

---

## 549. Dead Code Doesn't Talk: Authentic Requirements Elicitation in Introductory Software Engineering

**arXiv ID:** 2604.07211 | [PDF](https://arxiv.org/pdf/2604.07211v1)

**作者:** Santiago Berrezueta-Guzman `[一作]` (Technical University of Munich), Stefan Wagner `[通讯]` (Technical University of Munich)

**通讯引用:** 9237 | [OpenAlex ID](https://openalex.org/A5022333047)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在本科软件工程课程中，利用学生已完成的Java 2D迷宫游戏，设计了一个四阶段的真实需求获取活动，让20个学生团队与18位校内博士/博士后研究者进行面对面访谈、撰写SRS并完成原型演示，产生203条需求并提升学生的软技能；

**💡 创新点**

创新点在于将学生自制软件作为需求获取的“锚点”降低认知负担，并将校内研究人员作为可接触的真实客户，为缺乏行业合作的课程提供可复制的真实互动模式；

**🔧 技术方法**

采用Java 2D游戏、IEEE 830 SRS模板、用户故事图、用例图、敏捷迭代（Sprint）等工具与技术；

**📊 数据集**

数据集包括60名学生、20个团队、18位客户端、203条需求、SRS质量评分、原型演示评分以及软技能自评和反思报告；

**📈 对比分析**

通过前后自评量表和客户满意度问卷量化软技能提升（如利益相关者同理心 +1.33，谈判 +1.13），SRS平均质量为6.79/10，原型演示平均为7.21/10；

**⚠️ 局限性**

局限性：仅在一门课程、单一院校实验；客户自愿招募可能导致偏差；软技能自评存在主观性；缺乏对照组，难以单独归因；

---

## 550. VersaVogue: Visual Expert Orchestration and Preference Alignment for Unified Fashion Synthesis

**arXiv ID:** 2604.07210 | [PDF](https://arxiv.org/pdf/2604.07210v1)

**作者:** Jian Yu `[一作]` (Nanjing University of Science and Technology), Jinhui Tang `[通讯]` (Nanjing Forestry University)

**通讯引用:** 29054 | [OpenAlex ID](https://openalex.org/A5035112538)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了VersaVogue，一个统一的多条件可控时尚合成框架，支持服装生成和虚拟试穿。

**💡 创新点**

创新点在于引入了基于混合专家机制的特征路由注意力（TA）模块，实现了视觉属性的精确解耦和自适应注入，同时开发了自动化的多视角偏好优化（MPO）管道，提升了生成的现实感和可控性。

**🔧 技术方法**

使用了混合专家机制（MoE）和直接偏好优化（DPO）技术。

**📊 数据集**

使用了多个公开数据集，包括GarmentBench、VITON-HD和DressCode，涵盖了服装生成和虚拟试穿的任务。

**📈 对比分析**

与现有方法相比，VersaVogue在视觉真实感、语义一致性和细粒度可控性方面表现优越，实验结果显示其在多个基准测试中均超越了其他方法。

**⚠️ 局限性**

限制在于模型的复杂性和对多源异构条件的处理能力，可能在某些极端情况下仍然面临属性纠缠和语义干扰的问题。

---

## 551. INSPATIO-WORLD: A Real-Time 4D World Simulator via Spatiotemporal Autoregressive Modeling

**arXiv ID:** 2604.07209 | [PDF](https://arxiv.org/pdf/2604.07209v1)

**作者:** InSpatio Team `[一作]`, Ziqiang Zhao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种实时4D世界模拟框架，可从单段视频生成可交互、具有高自由度的动态场景，并支持实时导航与控制。

**💡 创新点**

核心创新包括：1）Spatio‑Temporal Autoregressive（STAR）框架，融合隐式时空缓存与显式几何约束，解决长时空一致性与精确相机控制；2）Joint Distribution Matching Distillation（JDMD）双教师策略，将合成控制任务与真实文本生成任务联合训练，通过真实数据分布对齐提升视觉质量，突破合成‑真实域差距。

**🔧 技术方法**

采用的技术包括：基于Wan2.1的Diffusion Transformer与KV缓存、RoPE位置固定、3D深度重投影与显式空间约束、双任务（V2V + T2V）联合蒸馏、Tiny‑VAE轻量化、torch.compile加速、以及多条件因果初始化等。

**📊 数据集**

实验使用公开视频数据集 RealEstate10K、Unreal Engine、ReCamMaster、OpenVid、Blender 等多来源视频与合成数据。

**📈 对比分析**

与FantasyWorld、Infinite‑World、LingBot‑World、Traj…等方法对比，本文在WorldScore‑Dynamic、RE10K‑Long 与 Camera‑Controlled Video Rerendering 任务上均实现SOTA。具体表现为：动态得分 68.72（仅 1.3B 模型 24 FPS 运行），RE10K‑Long FID 42.68、FVD 100.55，控制误差（旋转、平移）均低于竞争者；OpenVid 与 Blender 上的 VBench、FID/FVD 亦均位列首位。

**⚠️ 局限性**

主要局限：1）难以长期记忆并保持生成区域细节纹理；2）在全360°或大视角动态切换时，动态物体的多视角一致性与时空连贯性仍存在不足。

---

## 552. BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment

**arXiv ID:** 2604.07201 | [PDF](https://arxiv.org/pdf/2604.07201v1)

**作者:** Mohamed Darwish Mounis `[一作]` (High Institute for Computer and Information Systems), Hyun-Soo Kang `[通讯]` (High Institute for Computer and Information Systems)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 BRIDGE 系统，先用 FORGE 对多模态查询进行强化学习重写，随后使用 LENS 进行文本检索，彻底消除了视觉噪声对检索效果的负面影响。

**💡 创新点**

创新点在于把多模态检索瓶颈定位为查询噪声，而非编码器容量，提出了无多模态编码器、基于 RL 的查询对齐模型 FORGE 与专门针对推理需求的密集检索器 LENS，实现在文本检索上仅用文本输入即可完成多模态检索。

**🔧 技术方法**

使用了 GPT‑4o 生成图像描述、Qwen2.5‑7B‑Instruct（FORGE）与 Qwen3‑Embedding‑4B（LENS）等 LLM，强化学习（GRPO）进行查询重写优化，以及对比学习与信息熵损失训练检索器。

**📊 数据集**

主要使用 MM‑BRIGHT benchmark（2803 个多模态查询，涵盖 29 个技术领域）进行训练和评估，并在此数据集上进行实验。

**📈 对比分析**

通过与 CLIP、SigLIP、Nomic‑Embed‑Vision、BM25 等多模态与文本检索基线对比，BRIDGE 在 MM‑BRIGHT 上达到 29.7 nDCG@10，超过 Nomic‑Vision（27.6）并在某些领域进一步击败 32.2 的文本检索上限，证明了查询重写的显著提升。

**⚠️ 局限性**

局限性包括对 LLM 预训练和强化学习的依赖，查询重写质量对检索器基线的敏感性，且在视觉信息对答案贡献不大时提升有限，无法完全消除所有多模态检索差距。

---

## 553. LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics

**arXiv ID:** 2604.07193 | [PDF](https://arxiv.org/pdf/2604.07193v1)

**作者:** Kosmas Pinitas `[一作]` (University of Piraeus), Ilias Maglogiannis `[通讯]` (University of Piraeus)

**通讯引用:** 7031 | [OpenAlex ID](https://openalex.org/A5013025849)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用大型语言模型生成的语义先验，对手工构造的面部几何与声学特征进行语义调节的轻量化情感变化预测框架（LaScA）。

**💡 创新点**

创新点在于：①将结构化行为特征映射到固定的文本描述词典，再通过冻结的句子编码器生成上下文语义嵌入；②在保持特征透明度的同时，引入语言模型的高层语义抽象；③使用偏好学习的两层MLP，仅训练极少参数，极大降低模型复杂度。

**🔧 技术方法**

技术主要包括手工特征提取（面部Blendshape、MFCC）、ChatGPT 生成语义词典、句子Transformer（MPNet、MiniLM 等）生成语义嵌入、Otsu阈值做特征显著性筛选、偏好学习（pairwise difference + BCE）与多模态融合。

**📊 数据集**

数据集：Aff-Wild2（面部+音频、约280万帧）和 SEWA DB（双人对话、面部+音频、跨文化）。

**📈 对比分析**

与多种基线（VGGFace2、SwinFace、MAE-Face、Wav2Vec2、MAE-Audio、HiCMAE、MMA-DFER 等）及单纯文本/特征模型对比，LaScA 在 3s/5s 10%/20% 的阈值下，在视觉、音频和多模态任务中均达到或接近最优准确率，尤其在 SEWA 的对话场景和 5s 窗口下获得最高的 valence/ arousal 预测准确率。

**⚠️ 局限性**

局限性：①使用冻结的预训练编码器，未探索轻量化微调的潜力；②仅评估固定的 1 轮语义词典，未考虑多语言或动态更新；③仅关注短时间窗口的相邻比较，缺乏长序列情感轨迹建模；④在 SEWA 实验中未使用原始音频端到端模型，可能限制声学表达的充分利用。

---

## 554. Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training

**arXiv ID:** 2604.07279 | [PDF](https://arxiv.org/pdf/2604.07279v1)

**作者:** Changkun Liu `[一作]` (Google), Luca Ballan `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于双重记忆（隐式快权重记忆 + 显式 token 记忆）的 RNN 模型，用于长序列 3D 感知（相机姿态估计、视频深度估计与 3D 重建）。

**💡 创新点**

创新点在于将姿态跟踪与几何映射解耦：用轻量级 MLP 快权重实现姿态预测并通过 Test‑Time Training 自适应更新；用显式 token 记忆保持全局几何上下文，并通过通道级门控实现更细粒度的状态融合；兼容并可进一步提升现有的 TTT3R/TTSA3R 无需训练的状态更新策略。

**🔧 技术方法**

主要技术包括：ViT 编码器、Transformer 解码器、SwiGLU MLP 快权重模块、通道门控更新机制、Test‑Time Training、以及与 TTT3R/TTSA3R 的 plug‑and‑play 状态更新。

**📊 数据集**

在多数据集上进行评估：CO3Dv2、ARKitScenes、MegaDepth、MapFree、DL3DV、Hypersim、ScanNet、TUM Dynamics、Sintel、KITTI、Bonn、7‑Scenes 与 NRGBD。

**📈 对比分析**

与 CUT3R 及其 TTT3R/TTSA3R 版本对比，模型参数减少 19%（793M → 644M），在 500–1000 帧长序列上 Absolute Trajectory Error 降低至 39%，视频深度估计和 3D 重建的 Chamfer Distance、Normal Consistency 等指标均优于基线，且保持与 CUT3R 同等的 FPS 与常数 GPU 内存占用。

**⚠️ 局限性**

局限性包括：对超长序列仍可能出现细粒度遗忘；需要在多任务多域数据上预训练，模型对极端动态场景或极低光照的鲁棒性尚待验证；实现复杂度较单纯的 RNN 或 Transformer 方法更高。

---

## 555. Validated Intent Compilation for Constrained Routing in LEO Mega-Constellations

**arXiv ID:** 2604.07264 | [PDF](https://arxiv.org/pdf/2604.07264v1)

**作者:** Yuanhang Li `[一作]` `[通讯]`, Yuanhang Li

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向低地球轨道大型星座的端到端意图驱动约束路由系统，整合LLM意图编译、GNN成本-到-目标路由器和8通道验证器，确保安全合规。

**💡 创新点**

创新点包括：1）基于GNN的成本-到-目标路由器通过监督蒸馏实现与Dijkstra等价的路由质量且推理速度提升17倍；2）LLM（Qwen3.5-9B）意图编译器结合检证反馈循环，完成语义级约束解析，复合意图匹配率提升46.2pp；3）8通道确定性验证器实现从结构完整性到可实现性完全可追踪的安全保证。

**🔧 技术方法**

使用的技术包括图注意力网络（GAT）、大规模语言模型（Qwen3.5-9B）与few-shot提示、JSON提取与修复循环、基于BFS/Dijkstra/Edmonds-Karp的可行性证明、约束编译中间表示（ConstraintProgram）以及可视化验证流程。

**📊 数据集**

数据集为合成的Walker Delta 20×20 LEO星座，包含500个拓扑快照、240条多类别意图（单一、复合、条件、不可行），并对距离驱动与均匀延迟两种链路时延进行评测。

**📈 对比分析**

对比方法包括传统Dijkstra、规则基解析器、不同规模LLM（4B、9B）及无修复/无检证等变体。性能表现：GNN路由PDR与Dijkstra相同（99.8%）但速度提升17×；LLM编译在所有可行意图上完成率98.4%，语义匹配87.6%；验证器零误接收、检测100%结构错误，平均运行时<1.6ms。

**⚠️ 局限性**

局限性包括：1）GNN在不同倾角（SSO）上不具备跨拓扑泛化；2）验证器覆盖的可行性片段有限，复合约束如k‑disjoint+时延仍落入Abstain；3）LLM编译在新语义/多样化表达上仍需改进；4）系统对紧急子秒级响应仍依赖预编译模板；5）真实运营意图分布与合成基准可能差异大。

---

## 556. Android Coach: Improve Online Agentic Training Efficiency with Single State Multiple Actions

**arXiv ID:** 2604.07277 | [PDF](https://arxiv.org/pdf/2604.07277v1)

**作者:** Guo Gan `[一作]` (Zhejiang University), Hong Zhou `[通讯]` (Zhejiang University)

**通讯引用:** 469218 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的在线强化学习框架，采用单状态多动作（Single State Multiple Actions）范式，利用学习的 Q 函数评估同一状态下多种动作，从而提升 Android GUI 代理的训练效率和成功率。

**💡 创新点**

创新之处在于引入了 Process Reward Model 对中间步骤进行细粒度奖励，构建了 Actor‑Critic Leave‑One‑Out (ACLOO) 价值优势估计，并通过多动作评估实现了在不额外交互的前提下显著提高样本利用率。

**🔧 技术方法**

采用了基于 Vision‑Language 模型的 actor‑critic 结构、离线预训练的过程奖励模型、蒙特卡洛回报估计、以及 PPO clip 损失等技术。

**📊 数据集**

在 AndroidLab 和 AndroidWorld 两个基准任务集上进行实验，同时使用来自 AndroidControl 的离线轨迹与自收集的随机任务构建 2k 条训练样本。

**📈 对比分析**

与 UI‑TARS‑1.5‑7B、PPO、GRPO 等基线在相同训练时间下对比，实验显示在 AndroidLab 上成功率从 31.9% 提升至 39.4%，在 AndroidWorld 上提升至 41.1%，比 GRPO/PPO 提高约 1.4 倍的训练效率。

**⚠️ 局限性**

局限性包括缺乏大规模并行化系统优化、未进行预先的监督式微调、以及对 LLM 结果验证器的依赖导致奖励信号偶尔不可靠。

---

## 557. Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning

**arXiv ID:** 2604.07345 | [PDF](https://arxiv.org/pdf/2604.07345v1)

**作者:** Roberto Vercellino `[一作]` (National Laboratory of the Rockies), Juliane Mueller `[通讯]` (National Laboratory of the Rockies)

**通讯引用:** 648 | [OpenAlex ID](https://openalex.org/A5063711307)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 NVIDIA H100 GPU 节点上 GenAI 训练、微调与推理工作负载进行 0.1 s 级别功耗测量，并将测得的高分辨率功耗曲线通过离散事件仿真框架 DIPLOEE 扩展至整个数据中心，实现从节点级到设施级的能耗建模；同时公开了该功耗数据集；

**💡 创新点**

创新点在于：①将可复现的 AI 基准（MLCommons、vLLM）与实测功耗数据结合，填补了工作负载功耗与数据中心整体负载之间的空白；②提出了底层向上耦合、事件驱动的功耗扩展模型，能够根据用户行为、日周季节分布实时生成设施级负载曲线；③公开了公开可复用的高分辨率功耗数据集，为后续能源规划提供了基准；

**🔧 技术方法**

使用的技术包括：H100 GPU 节点的 NVML / RAPL 采样工具（WattAMeter）、MLCommons Training v4.0（Llama‑2 70B 微调、Stable Diffusion）和 vLLM 推理框架、Python + SimPy 的离散事件仿真、NLR Kestrel HPC 系统的工作负载调度和分布概率；

**📊 数据集**

使用的工作负载数据集包括：MLCommons Training benchmark（Llama‑2 70B 微调、Stable Diffusion），vLLM 推理 benchmark（Llama‑3 70B 离线与在线推理），以及 NLR HPC 任务日志与 Microsoft Azure 负载分布数据；

**📈 对比分析**

通过离散事件仿真对比不同节点数与负载比例下的功耗曲线与能耗；结果显示：训练与推理功耗随节点数线性增长，运行时受全局批量影响；在设施级模拟中，功耗随平均利用率上升而趋于饱和，峰值/平均比（PAR）在高利用率时下降；队列时间与功耗峰值随利用率显著增加；整体功耗仅达到设计功率的 73%–80%，表明有提升空间；

**⚠️ 局限性**

局限性包括：①测量仅在特定 H100+AMD EPYC 芯片的节点上完成，未覆盖其他加速器或网络拓扑；②未考虑冷却、UPS、变压器等辅助负载，仅算 IT 负载；③利用率与请求率分布取自外部实验室或 Azure 数据，可能不完全代表商业超大规模或企业环境；④模型未结合热力学或电力输送失效率，未来需扩展至更完整的设施级仿真。

---

## 558. Appear2Meaning: A Cross-Cultural Benchmark for Structured Cultural Metadata Inference from Images

**arXiv ID:** 2604.07338 | [PDF](https://arxiv.org/pdf/2604.07338v1)

**作者:** Yuechen Jiang `[一作]` (University of Manchester), Sophia Ananiadou `[通讯]` (University of Manchester)

**通讯引用:** 17451 | [OpenAlex ID](https://openalex.org/A5077976343)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了跨文化多类别的结构化文化元数据推断基准Appear2Meaning，并使用视觉语言模型在仅图像输入下预测文化、时期、来源与作者等属性。

**💡 创新点**

创新点在于将多维结构化元数据推断转化为可评价任务，采用LLM-as-Judge框架实现属性级别的语义一致性评估，并首次系统比较不同文化与模型的推断表现。

**🔧 技术方法**

使用现有视觉语言模型（Qwen-VL系列、Pixtral-12B、GPT-4/5、Claude Haiku）生成预测，并利用LLM判定器（GPT‑4.1‑mini）对预测与参考元数据进行精确/部分匹配与属性准确率评估。

**📊 数据集**

基于 Getty 和 Metropolitan Museum of Art 开放数据，挑选 750 件跨四个文化区域（东亚、古地中海、欧洲、美洲）和四种对象类型（陶瓷、绘画、金属制品、雕塑）的完整元数据图像，构成公开可复现的数据集。

**📈 对比分析**

通过精确匹配、部分匹配率和属性级别准确率对九个最先进模型进行比较，发现精确匹配率极低但部分匹配率相对较高，Open‑weight Qwen3‑Flash 在多数地区表现最佳，跨文化差异显著且模型在文化、时期、来源等属性上的表现远逊于标题和作者。

**⚠️ 局限性**

局限性包括模型主要靠视觉相似性与训练偏好做推断，难以一致性地恢复非可观文化属性；跨文化偏见导致某些地区表现不佳；LLM-as-Judge 的评判标准可能嵌入主流文化假设，未充分考虑多样化文化语境。

---

## 559. How to sketch a learning algorithm

**arXiv ID:** 2604.07328 | [PDF](https://arxiv.org/pdf/2604.07328v1)

**作者:** Sam Gunn `[一作]` `[通讯]`, Sam Gunn

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种数据删除方案，能够在深度学习模型中快速预测删除任意子集训练数据后模型的行为，并给出了理论误差保证。

**💡 创新点**

创新点在于：① 引入“稳定性”这一新假设，证明在此假设下可实现误差可控的数据删除；② 设计了基于高阶前向自动微分的局部 sketch 方法，利用随机方向和 median‑of‑means 取得指数收敛；③ 方案在预计算和预测阶段仅相对传统训练和推理慢 1/ε 倍，存储需求相当于 1/ε 个模型。

**🔧 技术方法**

主要技术包括：高阶前向自动微分、随机复数方向采样、median‑of‑means 估计、算术电路模型的稳定性分析，以及对局部 Taylor 近似的统计估计。

**📊 数据集**

实验使用微型 GPT（microgpt）模型及其对应的训练数据集，验证了稳定性假设在实际小规模模型上的成立。

**📈 对比分析**

与现有影响函数、数据估值或基于近似的删除方法对比，本文方案在预计算成本上与训练相当，在预测成本上仅比一次普通推理慢 d·(1/ε) 倍，误差可控制在 O(ε)。实验结果显示在 microgpt 上误差符合理论预测，且比 heuristic 方法更具一致性。

**⚠️ 局限性**

局限性包括：① 依赖稳定性假设，强稳定性假设可扩展但实证验证有限；② 对高阶导数的计算在极大模型上成本仍高；③ 方案基于算术电路抽象，实际深度网络的实现细节需进一步调优；④ 目前仅在小规模模型上验证，尚未在大型商业模型上测试。

---

## 560. Mapping Child Malnutrition and Measuring Efficiency of Community Healthcare Workers through Location Based Games in India

**arXiv ID:** 2604.07299 | [PDF](https://arxiv.org/pdf/2604.07299v1)

**作者:** Arka Majhi `[一作]` (Indian Institute of Technology Bombay), Satish B. Agnihotri `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 333 | [OpenAlex ID](https://openalex.org/A5086352343)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过共同设计的基于位置的游戏，帮助印度社区卫生工作者（CHWs）在不同地区持续、系统地收集儿童体格测量数据，进而绘制营养不良热点图。

**💡 创新点**

首次将严肃游戏与众包机制结合用于儿童营养监测，提升了CHWs的采集动力和数据质量，并为公共卫生决策提供了更及时、空间精准的证据。

**🔧 技术方法**

使用Unity引擎开发Android/ iOS游戏，嵌入测量仪器接口、自动计算Z分数及奖励机制，并通过游戏化指标监控采集效率。

**📊 数据集**

利用近200名CHWs采集的儿童身高、体重、MUAC等体格数据（基线来自NFHS‑5），与传统手工登记和手机录入数据进行对比。

**📈 对比分析**

采用准实验设计（n=94对组），通过测量效率得分进行配对t检验；游戏组在后测时平均得分显著高于对照组（p=0.00004，Cohen's D≈1.6），且三个月后保持较高的效率，表明游戏化方法提升了性能和保持性。

**⚠️ 局限性**

研究样本受限于机构化环境，未能充分观察长期持续性；游戏新奇性随时间衰退，需进一步强化激励机制；部分CHWs存在数据造假，强调后期需加入异常检测与审核。

---

## 561. MoRight: Motion Control Done Right

**arXiv ID:** 2604.07348 | [PDF](https://arxiv.org/pdf/2604.07348v1)

**作者:** Shaowei Liu `[一作]` (NVIDIA), Jun Gao `[通讯]` (NVIDIA)

**通讯引用:** 4244 | [OpenAlex ID](https://openalex.org/A5100776558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种统一框架，实现单帧图像到视频的生成，支持摄像机与物体运动的独立控制，并能进行因果推理（前向与逆向）

**💡 创新点**

创新点包括：1）dual‑stream 结构，将物体运动在固定视角下生成后通过跨视角注意力转移到任意视角，实现摄像机与物体运动的解耦；2）主动/被动运动分解与运动 dropout 训练，使模型学习动作‑结果的因果关系；3）支持从动作预测结果到动作推断的两种因果推理方式

**🔧 技术方法**

主要技术包括：latent diffusion (DiT) + VAE 编码、流匹配训练、跨视角注意力、相机与轨迹条件注入、运动 dropout 与多尺度轨迹采样

**📊 数据集**

使用的数据集包括公开视频集（Panda‑70M、Wild‑SDG‑1M、DynPose‑100K、WISA、Cooking）、同步合成数据（SyncCamMaster）以及通过合成双视角视频生成的配对数据

**📈 对比分析**

与 Gen3C、MP、ATI、WanMove 等基线在视觉质量（PSNR、SSIM、FID、FVD）、摄像机准确度（旋转/平移误差）、运动精度（EPE）、物理一致性（PC、SA）和人工评测（可控性、运动真实性、视觉逼真度）进行比较，实验显示该方法在多数指标上达到或超过基线，人工评测亦占优

**⚠️ 局限性**

局限性包括：偶尔产生不合理的交互推理；轨迹稀疏或遮挡导致运动不自然；物理一致性问题（如物体消失、异常出现）；对剧烈或快速摄像机运动的适应性有限

---

## 562. SL-FAC: A Communication-Efficient Split Learning Framework with Frequency-Aware Compression

**arXiv ID:** 2604.07316 | [PDF](https://arxiv.org/pdf/2604.07316v1)

**作者:** Zehang Lin `[一作]` (Xiamen University of Technology), John Thompson `[通讯]` (University of Edinburgh)

**通讯引用:** 121932 | [OpenAlex ID](https://openalex.org/A5035263718)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种名为SL‑FAC的通信高效拆分学习框架，旨在减少边缘设备与服务器之间的特征传输开销并提升模型收敛速度。

**💡 创新点**

采用自适应频域分解（AFD）将特征映射到频域并分离重要与冗余信息，再通过基于频谱能量的自适应量化压缩（FQC）为不同频率分量分配量化位宽，从而实现非均匀压缩。

**🔧 技术方法**

使用频域变换（离散余弦变换 DCT）、能量阈值选择、zig‑zag 采样、对数映射量化比率、线性最小–最大量化以及 IDCT 回到空间域等技术。

**📊 数据集**

在 MNIST 与 HAM10000 图像分类数据集上进行实验，分别在 IID 与 non‑IID 场景下评估。

**📈 对比分析**

与 PQ‑SL、TK‑SL、FC‑SL 等基准方法比较，SL‑FAC 在 MNIST 上 98.39%（IID）/97.65%（non‑IID）仅 15/20 通信轮次内达到最高精度，HAM10000 上 77.81%/76.46% 在 30/40 轮次内优于其他方法，整体提高 19.78% / 6.06% 的测试准确率。

**⚠️ 局限性**

仅在单模态图像分类任务上验证，未对大规模多模态模型或极端稀疏数据进行评估，且实现依赖 DCT 与量化，可能在高分辨率或实时视频场景中存在计算瓶颈。

---

## 563. A Proposed Framework for Advanced (Multi)Linear Infrastructure in Engineering and Science (FAMLIES)

**arXiv ID:** 2604.07311 | [PDF](https://arxiv.org/pdf/2604.07311v1)

**作者:** Devin A. Matthews `[一作]` (Southern Methodist University), Robert A. van de Geijn `[通讯]` (University of Texas at Austin)

**通讯引用:** 7910 | [OpenAlex ID](https://openalex.org/A5061722569)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

构建了一个垂直集成的高性能稠密线性代数与多线性张量计算框架（FAMLIES），兼容 CPU、GPU 及分布式系统，支持从 BLAS 到 LAPACK 甚至张量运算的统一接口。

**💡 创新点**

创新点在于：①将 BLAS、LAPACK 与张量算子通过控制树统一建模；②提供可扩展的算法空间与动态组合；③通过垂直集成显著降低数据移动，实现循环融合和多级缓存优化；④实现了传统 LAPACK 功能之外的张量分解与反对称矩阵因子分解等新算法。

**🔧 技术方法**

技术手段包括：BLIS 与 libflame 实现的通用算法 API、C++ 模板与元编程、控制树抽象、GPU‑aware MPI、CUTLASS、BLIS 线程通信器、张量‑矩阵映射（block‑scatter）以及动态分块与 3D/2.5D 通信策略。

**📊 数据集**

采用合成的稠密矩阵/张量数据集（不同规模、精度、实/复数），以及来自量子化学、机器学习等领域的真实案例（如 PXPᵀ = LT Lᵀ 分解、CP‑ALS、HOOI/HOSVD）。

**📈 对比分析**

通过与 Elemental、ScaLAPACK、手工优化的 Elemental 以及新搜索得到的算法族进行基准比较，实验在 BlueGene/P 8192 节点、AMD EPYC 7763 64 核 CPU、GPU 加速节点上进行，结果显示在多数关键操作上实现了 10%–50% 的加速，并在反对称分解上超过现有最佳实现。

**⚠️ 局限性**

局限性包括：①目前实现仅覆盖核心 DLA 与少数张量运算，未完成完整 LAPACK 及 ScaLAPACK 功能；②对极大规模或异构集群的可扩展性尚需进一步验证；③依赖手动选择控制树，自动调优与即时编译尚未实现；④对遗留应用的兼容性需要进一步完善。

---

## 564. Improved Implementation of Approximate Full Mass Matrix Inverse Methods into Material Point Method Simulations

**arXiv ID:** 2604.07307 | [PDF](https://arxiv.org/pdf/2604.07307v1)

**作者:** John A. Nairn `[一作]` (Oregon State University), John A. Nairn `[通讯]` (Oregon State University)

**通讯引用:** 14136 | [OpenAlex ID](https://openalex.org/A5070533171)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一种改进的全质量矩阵近似方法FMPM(k)，通过递归增量方式计算网格速度，并解决了与稀疏质量矩阵相关的边界条件、材料接触和裂缝接触等问题。

**💡 创新点**

创新点包括：①将FMPM(k)递归公式改为与k无关的增量形式；②在每一增量中加入边界条件和接触修正，彻底消除与传统MPM功能的冲突；③对时域稳定性进行系统分析，提出混合、周期性和动态FMPM(k)等提升稳定性和效率的方案；④实现了自适应动态阶数的控制机制。

**🔧 技术方法**

使用的技术主要有：Material Point Method（MPM）框架；全质量矩阵的泰勒级数近似；增量递归求解网格速度；混合质量矩阵（与FMPM(1)或FMPM(2)混合）；周期性更新；动态收敛判据；以及基于逻辑回归的接触判定和多材料接触算法。

**📊 数据集**

实验数据集包括：自由振动杆、移动壁单轴拉伸、冲击波传播、两个盘子碰撞、块体撞击等；使用的网格函数有CPDI、B2CPDI、线性/二次样条；测试采用OSParticulas和NairnMPM代码实现，网格尺寸、粒子数、时间步长等参数在论文中具体给出。

**📈 对比分析**

与传统FLIP、FMPM(1)、FMPM(2)以及不同阶数的FMPM(k)进行比较。结果表明：FMPM(k≥4)在速度边界条件误差、冲击波前沿解析度和能量损耗方面显著优于FLIP；动态和混合FMPM(k)能在一定程度上减少计算成本；周期性FMPM(k)可在保持低能量耗散的同时提升稳定性；然而随着阶数增加，时间步长需进一步减小，计算量也显著上升。

**⚠️ 局限性**

局限性包括：高阶FMPM(k)对时间步长的苛刻要求导致计算成本大幅上升；混合质量矩阵虽提升稳定性但可能引入过度耗散；动态阶数控制依赖收敛判据，选择不当可能导致高阶计算或过度耗散；接触和裂缝算法仍是近似处理，复杂接触场景中可能出现数值不稳定；不完美界面模型尚未完全集成到新的递增框架中。

---

## 565. Chatbot-Based Assessment of Code Understanding in Automated Programming Assessment Systems

**arXiv ID:** 2604.07304 | [PDF](https://arxiv.org/pdf/2604.07304v1)

**作者:** Eduard Frankford `[一作]` (University of Innsbruck), Ruth Breu `[通讯]` (University of Innsbruck)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对编程教育中基于聊天机器人的对话式评估方法进行系统性综述，并基于综述结果提出一种混合苏格拉底框架，旨在将确定性代码分析与受控对话层相结合，用于验证学生的代码理解。

**💡 创新点**

创新点在于：①提出“混合苏格拉底框架”，将规则驱动的静态/动态分析与LLM驱动的对话层双代理（提问者+评估者）分离；②通过“苏格拉底守护栏”与“知识追踪”机制，将问题与代码事实紧密绑定，减少LLM的hallucination与直接给解的风险；③提供了针对高风险评估的部署与安全保障（受控部署、随机化执行轨迹、逐步推理、局部模型选项）。

**🔧 技术方法**

技术与工具：静态代码分析（AST）、动态执行追踪、LLM（如GPT‑4、Gemini）、双代理LLM（Instructor Agent、Verifier Agent）、对话管理与知识追踪、Socratic Guardrails、可选本地Transformer模型、Streamlit 前端。

**📊 数据集**

数据集：论文主要基于文献综述（2018‑2025年共12篇核心研究），原型实验使用自建的示例代码与预生成的对话数据，未使用公开大规模代码或学生交互数据集。

**📈 对比分析**

对比方法与性能：论文未进行系统的实证评估；在讨论中引用已有研究的学习增益（如Socratic Author 43%提升）及原型功能演示，但未给出量化的评分或对比实验；因此性能评估暂缺，主要展示可行性与设计思路。

**⚠️ 局限性**

局限性：①原型仅为可行性演示，未整合完整APAS管道；②缺乏多语言支持与大规模课堂部署实验；③未进行交互质量与评分一致性评估（如交叉评审、模型漂移、延迟与成本分析）；④对高阶评估的安全与公平性仍需进一步验证；⑤模型依赖与供应商变动带来可持续性挑战。

---

## 566. Fast Spatial Memory with Elastic Test-Time Training

**arXiv ID:** 2604.07350 | [PDF](https://arxiv.org/pdf/2604.07350v1)

**作者:** Ziqiao Ma `[一作]` (MIT-IBM Watson AI Lab), Chuang Gan `[通讯]` (MIT-IBM Watson AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于弹性测试时训练的快速空间记忆模型（Fast Spatial Memory），实现对长序列动态场景的4D重建与新视角-时间合成。

**💡 创新点**

创新点包括：1) 引入弹性权重约束（Elastic Weight Consolidation）于大型块测试时训练（LaCET），使快速权重在推理过程中保持弹性平衡，避免灾难性遗忘；2) 采用多块（multi‑chunk）适配策略，实现可扩展的长上下文建模；3) 在同一框架下同时支持LVSM与LRM两种解码器，兼顾无几何与几何约束的4D重建。

**🔧 技术方法**

核心技术：弹性测试时训练（Elastic Test‑Time Training）、快速权重（Fast Weights）、大型块快速权重更新（Large‑Chunk Test‑Time Training, LaCT）、EWC‑style 正则化、EMA‑anchor 更新、SwiGLU‑MLP、Patch‑Token 化、Plücker 光线映射、时间与相机条件编码、光栅化的高斯切片渲染、以及对多种解码器的端到端训练。

**📊 数据集**

使用的训练与评估数据集包括：RealEstate10K、DL3DV、PointOdyssey、Spring、DynamicReplica、Multi‑Cam Video、Stereo4D、NVIDIA 4D 数据集以及常用的静态 3D 数据集 DL3DV‑140。

**📈 对比分析**

与现有基于渲染的 4D 重建模型（如 4D‑LRM、4D‑GS、4D‑LRM、4D‑GS‑LRM 等）以及 3D 视角合成基线（DL3DV‑140）在 Stereo4D 与 NVIDIA 评估基准上进行了对比；在 256×256 分辨率下，Fast Spatial Memory 在 PSNR、SSIM、LPIPS 等指标上均优于同类无几何约束方法，且性能逼近甚至接近需场景优化的强大方法。

**⚠️ 局限性**

局限性：1) 仅在已标定的姿态图像上训练，未解决无姿态或动态相机自定位；2) 仅使用渲染监督，缺乏显式几何约束，导致几何精度与时间一致性仍有待提升；3) 受限于可授权数据和计算资源，实验规模未能充分验证“无限长序列”能力；4) 对长时间持续推理的稳定性与泛化仍需进一步研究。

---

## 567. NIRVANA: A Comprehensive Dataset for Reproducing How Students Use Generative AI for Essay Writing

**arXiv ID:** 2604.07344 | [PDF](https://arxiv.org/pdf/2604.07344v1)

**作者:** Andrew Jelson `[一作]` (Virginia Tech), Sang Won Lee `[通讯]` (Virginia Tech)

**通讯引用:** 10804 | [OpenAlex ID](https://openalex.org/A5100444708)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集并公开了 77 名大学生在写作任务中使用 ChatGPT 的完整过程数据（键盘打字记录、复制粘贴、对话历史），并构建了可重现写作过程的 replay 界面，同时基于人类贡献比（HCR）和人类编辑比（HER）对学生的写作方式进行聚类，识别出四种写作角色。

**💡 创新点**

首次提供了包含时序细粒度交互的学生‑AI 写作数据集及其可视化分析工具，提出了 HCR 与 HER 两种量化指标来评估学生在 AI 辅助写作中的实际参与度，并通过聚类揭示了多样的写作行为模式。

**🔧 技术方法**

使用自研写作环境（CodeMirror + CodeMirror‑Record）与内置 ChatGPT 接口（OpenAI GPT‑3.5‑turbo），实现了键盘事件、复制粘贴和对话的实时记录；后端使用 OpenAI API 与前端 Web Replay 系统（React/Canvas）重现写作过程。

**📊 数据集**

NIRVANA 数据集：77 名学生在 30 分钟内完成的论说文写作过程数据，包括 382 词平均篇幅、22.5 分钟平均耗时、4.5 次平均 ChatGPT 查询以及完整的 keystroke 日志与对话记录。

**📈 对比分析**

通过 Spearman 相关、K‑means 聚类、Kruskal–Wallis 与 Dunn 后验检验等统计方法比较查询次数、写作时长、可读性与作者归属感等指标，结果显示查询频率与篇幅、时长、可读性正相关，四类写作角色在这些指标上显著差异；但未构建机器学习模型，因而无传统意义上的性能指标。

**⚠️ 局限性**

局限性包括：仅使用单一 ACT 风格论说题，数据来自 GPT‑3.5‑turbo 版本，写作时长短且无真实学术分数，样本来源自大学邮件和 Prolific，可能存在自选偏差，且数据未覆盖长周期写作或其他体裁，无法直接推断 AI 在更广泛教育情境下的影响。

---

## 568. TC-AE: Unlocking Token Capacity for Deep Compression Autoencoders

**arXiv ID:** 2604.07340 | [PDF](https://arxiv.org/pdf/2604.07340v1)

**作者:** Teng Li `[一作]` (Inclusion AI, Ant Group), Jun Zhang `[通讯]` (HKUST)

**通讯引用:** 17778 | [OpenAlex ID](https://openalex.org/A5100436731)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于ViT的深度压缩图像编码器TC-AE，利用分阶段的token压缩和自监督学习来生成更具语义结构的latent表示；

**💡 创新点**

创新点在于从token空间角度解决token-to-latent压缩瓶颈，提出分阶段token压缩来分散压缩负荷，并引入iBOT自监督目标在tokenizer训练中直接学习语义结构；

**🔧 技术方法**

使用的技术包括Vision Transformer（ViT）编码器、像素分块与多阶段压缩、iBOT自监督学习（MIM + CLS）、对抗+感知+像素重建损失以及LightningDiT扩散模型；

**📊 数据集**

在ImageNet-1K（训练）和ImageNet-50k（评估）数据集上进行训练和评估；

**📈 对比分析**

与现有深度压缩tokenizer（如DC-AE、GigaTok）以及低压缩tokenizer（ViTok、MAETok）进行对比，TC-AE在同等或更低的GFLOPs下实现了更低的gFID（例如在256×256下从26.44降至7.16，CFG下为2.57），同时显著加速扩散模型收敛（4.7×更少迭代即可达到同样gFID）；

**⚠️ 局限性**

主要限制包括：自监督目标对重建质量有轻微负面影响；在极小token数量下仍需更大压缩比时结构损失可能不可避免；目前仅在ImageNet上验证，其他域/任务的泛化仍待探讨。

---

## 569. TAMEn: Tactile-Aware Manipulation Engine for Closed-Loop Data Collection in Contact-Rich Tasks

**arXiv ID:** 2604.07335 | [PDF](https://arxiv.org/pdf/2604.07335v1)

**作者:** Longyan Wu `[一作]` (Fudan University), Hongyang Li `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一套可穿戴视觉‑触觉交互界面和闭环数据收集引擎，用于双手接触丰富的机器人操纵任务，并通过 AR 触觉反馈实现实时恢复数据采集。

**💡 创新点**

创新点包括：①双模式（MoCap+VR）可穿戴跟踪，兼顾高精度与便携性；②在线可执行性校验，显著提升演示可重放率；③AR 触觉反馈系统 tAmeR，用于收集失败恢复轨迹；④金字塔式数据结构实现分阶段学习，从大规模单臂预训练到任务特定双臂演示再到恢复数据。

**🔧 技术方法**

技术实现涵盖：可穿戴双指抓取机制、可插拔触觉传感器（GelSight、DW‑Tac 等）、结构化标记轨迹与对象跟踪、在线逆运动学可行性检查、对比学习预训练、DAgger 风格增量更新以及 AR 头显触觉视频同步。

**📊 数据集**

使用 FreeTacMan（约 3000k 视觉‑触觉对，10k 轨迹）进行大规模单臂预训练，并在四个接触丰富任务（herbal transfer、cable mounting、binder clip removal、dish washing）中自采多模态演示与恢复数据。

**📈 对比分析**

通过与仅视觉的 ACT、无预训练触觉扩展、预训练+DAgger 等方案对比，实验表明：①仅视觉 34% 成功率提升至 75%（预训练+在线恢复）；②重放成功率从 39%/12% 提升至 100%；③在未见物体、光照扰动等情况下仍保持较高成功率，证明了系统的鲁棒性与泛化能力。

**⚠️ 局限性**

局限性包括：系统仅验证在视觉‑触觉抓取上，未扩展到多指或全手指操作；跨传感器泛化效果未知；对硬件依赖较高，且在更复杂场景与任务中尚未充分验证。

---

## 570. Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment

**arXiv ID:** 2604.07306 | [PDF](https://arxiv.org/pdf/2604.07306v1)

**作者:** Huaiyuan Qin `[一作]` (Institute for Infocomm Research, A*STAR), Hongyuan Zhu `[通讯]` (Institute for Infocomm Research, A*STAR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种噪声鲁棒的动态数据剪枝模块 AlignPrune，利用动态对齐得分（DAS）对样本进行可靠排序，避免在标签噪声环境下误保留高损失噪声样本。

**💡 创新点**

创新点在于用样本损失轨迹与清洁参考集轨迹的相关性（DAS）替代传统的单轮损失排序，从而实现对噪声样本的更精准剔除，并且可作为 plug‑and‑play 模块无缝集成到现有动态剪枝框架中。

**🔧 技术方法**

核心技术包括：损失轨迹记录、与清洁验证集平均损失轨迹的 Pearson 相关性计算、基于 DAs 的样本排序与剪枝，以及与 InfoBatch/SeTa 等动态剪枝算法的集成。

**📊 数据集**

在 CIFAR‑100N、CIFAR‑10N、WebVision、Clothing‑1M、ImageNet‑1K 等多种公开基准数据集上进行实验，覆盖合成与真实标签噪声场景。

**📈 对比分析**

与静态剪枝方法（如 SmallL、Margin、Prune4ReL 等）以及动态剪枝基线（InfoBatch、SeTa）对比，AlignPrune 在 30%、50%、70% 剪枝比例下均能提升 0.1%~6.3% 的测试精度，且在大规模数据集上仍保持显著优势，训练效率亦有提升。

**⚠️ 局限性**

局限性包括：依赖一个足够干净的参考集（尽管实验显示可使用极少量或伪干净样本，但极端噪声时仍可能受限）；需要额外存储轨迹信息，增加一定内存开销；目前仅验证在 CNN/ViT 等常见网络，其他网络或任务的适用性待进一步研究。

---

## 571. OpenSpatial: A Principled Data Engine for Empowering Spatial Intelligence

**arXiv ID:** 2604.07296 | [PDF](https://arxiv.org/pdf/2604.07296v1)

**作者:** Jianhui Liu `[一作]` (Joy Future Academy), Xiaojuan Qi `[通讯]` (University of Hong Kong)

**通讯引用:** 34485 | [OpenAlex ID](https://openalex.org/A5102498323)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个开源的空间数据生成引擎 OpenSpatial，并基于该引擎构建了规模达 300 万样本、覆盖五大空间理解任务（空间测量、空间关系、相机感知、多视角一致性、场景感知）的 OpenSpatial-3M 数据集。

**💡 创新点**

创新点主要有：① 采用 3D 定向包围盒（OBB）作为统一基准，实现跨视角、跨帧的空间一致性；② 开发自动 3D 提升（3D Lifting）流水线，可将无标注的 Web 视频转化为高质量 3D 场景与 OBB；③ 通过场景图枚举自动生成多样化的问答，涵盖从测量到多视角一致性等多任务，避免“空间近视”；④ 公开完整的生成管线，支持模块化 ablation 与可扩展性。

**🔧 技术方法**

技术包括：3D 定向包围盒表示、基于 Gemini 的目标识别、SAM 实例分割、深度映射与遮挡过滤、3D Box‑Centric 生成、场景图驱动 QA 合成、并行化与特征复用的高效流水线。

**📊 数据集**

使用的数据集有：EmbodiedScan（ScanNet、Matterport3D、ARKitScenes、SUN‑RGBD）、ScanNet++、Hypersim、Web 视频；生成的 OpenSpatial‑3M；以及在实验中混合使用的 SenseNova‑800K、VST、LLaVA‑OneVision 等通用多模数据。

**📈 对比分析**

对比方法：在 InternVL、Qwen、VST 等开源 VLM 上进行单轮 SFT，使用 32 GPU、AdamW、单 epoch 训练；在空间理解基准（BLINK、AllAngles、MMSI、ERQA、VSI 等）与通用多模基准（MMStar、MMBench、MMMU）上评估。实验显示，OpenSpatial‑3M 使模型在空间任务上平均提升 14.1%（最高 19%），且在通用基准上保持稳定，未出现灾难性遗忘；单一模块 ablation 证明 box‑centric 与过滤机制显著优于点云基准。

**⚠️ 局限性**

局限性：① 目前数据多集中在室内场景，导致在室外或桌面级别的复杂环境下表现略逊；② 进一步提升空间智能需要指数级增大数据量，单纯扩展仍面临 diminishing returns；③ 多任务联合训练时某些任务组合会出现梯度干扰，导致局部性能波动；④ 现有 3D 提升仍依赖深度估计，极端光照或动态场景可能导致 3D 质量下降。

---

## 572. Toward a Tractability Frontier for Exact Relevance Certification

**arXiv ID:** 2604.07349 | [PDF](https://arxiv.org/pdf/2604.07349v1)

**作者:** Tristan Simas `[一作]` `[通讯]`, Tristan Simas

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在表示层面上是否存在针对精确相关性认证的有限优化器兼容可判定性分类器，并提出了相关不可达边界；

**💡 创新点**

创新点在于首次证明任意标签核均为优化可实现，从而表明仅凭商形状无法界定可判定边界，并给出了针对四类障碍族的元不可行性结果；

**🔧 技术方法**

作者采用了元不可行性理论、归一化谓词的可接受性类、闭合律不变性以及相同轨道不一致证明构造等技术，并在 Lean 4 中进行形式化验证；

**📊 数据集**

论文未使用任何数据集，完全基于理论证明；

**📈 对比分析**

由于缺乏实验数据，作者未进行方法比较或性能评估，所有结论均基于严格的理论不可能性证明；

**⚠️ 局限性**

局限性在于仅关注归一化谓词的可接受性类，未给出可实现的可判定方案，也缺乏经验验证来支持理论结论。

---

## 573. From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians

**arXiv ID:** 2604.07337 | [PDF](https://arxiv.org/pdf/2604.07337v1)

**作者:** Diego Gomez `[一作]` (École Polytechnique), Maks Ovsjanikov `[通讯]` (École Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于3D Gaussian Splatting的表面重建方法，通过为每个高斯原语赋予可学习的定向法向量，实现完整闭合的“Gaussian Wrapping”，从而直接从概率密度场得到占据率和法向量并提取高质量的闭合网格。

**💡 创新点**

创新点在于：① 将Gaussian视为定向概率表面元素而非对称雾点；② 设计可学习的定向法向量和递归衰减模型，使其满足Objects as Volumes框架；③ 推导闭式表达式得到占据率和法向量场；④ 提出仅需两条基准点的Pivot‑Marching‑Tetrahedra和可调分辨率的Primal Adaptive Meshing，实现细节丰富、无表面侵蚀且体积极小的网格。

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting渲染器、改进的CUDA可微光栅化器、正向/反向衰减计算、Objects as Volumes理论、法向对齐损失、密度增补策略、二叉搜索求0.5等值面、Delaunay四面体化和自适应网格细化。

**📊 数据集**

实验数据集主要为DTU、Tanks & Temples以及Mip‑NeRF 360，此外在T&T使用Mesh‑Based Rendering（MBR）和虚拟扫描评估。

**📈 对比分析**

与现有全景重建方法（GOF、SOF、MILo、RaDe‑GS、GGGS等）和前景专用方法（PGSR、2DGS）相比，本方法在DTU、T&T的Chamfer/F1分数上刷新SOTA，能够完整重建细小结构如自行车辐条；在Mip‑NeRF 360的MBR指标上同样取得领先；在传统评估中，因网格密度不敏感，其优势更显著；相比PGSR等方法，缺陷的空洞被显著减少。

**⚠️ 局限性**

局限性包括：① 适配的自适应采样策略基于均匀采样，可能对高细节区域效果有限；② 目前的衰减模型仅适用于3DGS，扩展到更精细的体渲染模型仍需研究；③ 对极大体积或动态场景的实时性能仍未验证。

---

## 574. Personalized RewardBench: Evaluating Reward Models with Human Aligned Personalization

**arXiv ID:** 2604.07343 | [PDF](https://arxiv.org/pdf/2604.07343v1)

**作者:** Qiyao Ma `[一作]` (University of California Davis), Zhe Zhao `[通讯]` (University of California Davis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Personalized RewardBench，一个针对大语言模型奖励模型的个性化评测基准。

**💡 创新点**

创新点在于用用户个人化 Rubric 构造优劣对，并保证两者在一般质量上相近，仅差在个性化匹配。

**🔧 技术方法**

采用奖励模型评估、Best‑of‑N 与 PPO 策略优化、LLM‑as‑a‑judge 评估以及规划器提取用户 Rubric 等技术。

**📊 数据集**

利用 LaMP‑QA 数据集的用户历史交互和个人化 Rubric，构成测试对。

**📈 对比分析**

与现有 Chatbot Arena‑Personalized、PRISM‑Personalized 等基准对比，奖励模型准确率最高仅 75.94%，但与下游任务的相关性明显更高（NDCG 0.9180，Spearman 0.2571）。

**⚠️ 局限性**

局限性包括仅使用单一查询、未训练专门的个性化奖励模型、对用户画像的直接注入效果不佳。

---

## 575. Robots that learn to evaluate models of collective behavior

**arXiv ID:** 2604.07303 | [PDF](https://arxiv.org/pdf/2604.07303v1)

**作者:** Mathis Hocke `[一作]` (Freie Universität Berlin), Tim Landgraf `[通讯]` (Freie Universität Berlin)

**通讯引用:** 1630 | [OpenAlex ID](https://openalex.org/A5073520168)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一个基于强化学习的框架，利用仿真训练的机器人鱼策略在真实实验中与活鱼进行闭环交互，从而评估不同鱼类行为模型的真实性。

**💡 创新点**

创新点在于将 RL 策略与仿真到实物机器人的转移相结合，通过量化仿真与现实间的 Wasserstein 距离，提供了无偏、可重复的行为模型评估方法。

**🔧 技术方法**

使用技术包括：深度强化学习（训练四种不同鱼类行为模型对应的策略）、仿真与真实机器人部署、Wasserstein 距离度量、多维行为指标（如目标达成数、个体间距离、墙壁交互等）。

**📊 数据集**

数据集主要来自：(1) 用来训练 ConvNet 的鱼对轨迹数据（斑点孔雀鱼两两配对实验），(2) 现场实验记录的活鱼轨迹，用于评估仿真到现实的转移效果。

**📈 对比分析**

评估方法是对各行为指标在仿真和现实实验中的分布计算 Wasserstein 距离；结果显示 ConvNet 模型在大多数指标上的 sim‑to‑real 差距最小，证明其行为逼真度最高。

**⚠️ 局限性**

局限性包括：模型缺乏对个体间变异和运动节律的充分描述；机器人硬件限制导致无法完全再现鱼类的冲刺与停滞行为；实验仅涉及二元交互，未涵盖更大群体的集体运动。

---

## 576. Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification

**arXiv ID:** 2604.07298 | [PDF](https://arxiv.org/pdf/2604.07298v1)

**作者:** Xin Tian `[一作]` (University of Oxford), Julian Knight `[通讯]` (University of Oxford)

**通讯引用:** 45212 | [OpenAlex ID](https://openalex.org/A5051498432)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出ROAM模型，将空间区域标记路由到专家子网络，以解决WSI分类中实例共享路径导致的泛化受限问题。

**💡 创新点**

核心创新是利用容量约束的熵正则最优传输（Sinkhorn）结合图正则化，使专家利用均衡且具空间连贯性，避免传统softmax路由导致的专家集中化。

**🔧 技术方法**

技术实现包括：网格区域化压缩patch bag、GNN构造区域-专家成本、Sinkhorn OT求解路由计划、顶k稀疏化、专家门控注意力聚合以及最终的专家融合预测。

**📊 数据集**

实验数据集涵盖四大WSI任务：TCGA的NSCLC（LUAD vs LUSC）、BRCA（IDC vs 非IDC）、CRC（COAD vs READ）以及PANDA（6级ISUP分级），并在TCGA→CPTAC的外部测试中验证泛化。

**📈 对比分析**

与MeanPool、ABMIL、CLAM、DSMIL、MAMMOTH、PAMoE等MIL与MoE基线对比，ROAM在所有基准上均接近或略优；在NSCLC外部AUC达0.845±0.019，PANDA QWK 0.917±0.003，且相较于软max路由的MoE，fold方差更低。

**⚠️ 局限性**

局限性在于依赖冻结的基础模型编码器、固定网格区域化，未对编码器进行全程微调；未来可探索自适应区域化与端到端微调以提升性能。

---

## 577. ReCodeAgent: A Multi-Agent Workflow for Language-agnostic Translation and Validation of Large-scale Repositories

**arXiv ID:** 2604.07341 | [PDF](https://arxiv.org/pdf/2604.07341v1)

**作者:** Ali Reza Ibrahimzada `[一作]` (University of Illinois Urbana-Champaign), Reyhaneh Jabbarvand `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 652 | [OpenAlex ID](https://openalex.org/A5058824250)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多智能体框架，能够对整个代码仓库进行跨语言翻译和验证，只需输入源项目和目标语言即可完成整个流程。

**💡 创新点**

创新点在于：①采用四个专门化智能体（分析、规划、翻译、验证），实现真正的 PL‑agnostic 仓库级翻译；②利用 LLM 与轻量级静态分析工具（Tree‑sitter 等）取代传统规则或语言特定工具，显著降低工程成本；③提供完整的过程记录与可重现日志，首次实现透明、过程中心的评估。

**🔧 技术方法**

技术包括：Claude Code 2.1.19 LLM、Tree‑sitter 语法分析、模型上下文协议（MCP）、多智能体交互协议，以及基于测试覆盖率的验证与自动测试生成。

**📊 数据集**

数据集：118 个真实开源项目，覆盖 6 种编程语言（C、Go、Java、JavaScript、Python、Rust），共 230K 行代码，4,583 个翻译单元，跨 4 对语言（C-Rust、Go-Rust、Java-Python、Python-JavaScript）。

**📈 对比分析**

与四种现有神经符号/智能体方法进行对比。实验显示，本框架在编译成功率 99.4%、测试通过率 86.5% 上优于对比方法，测试通过率提升约 60.8%，平均翻译成本约 15.3 美元，耗时约 57 分钟。消融实验表明移除分析、规划或验证智能体会分别使测试通过率下降 22.7%、25.3% 和 30.3%。

**⚠️ 局限性**

局限性：①仅验证了 4 对语言，跨更多语言的泛化仍待验证；②依赖 LLM 可能产生幻觉，尤其在大规模仓库长程任务中；③测试翻译依赖现有测试集，若测试不完整或不具代表性，验证结果可能不准确；④实验结果可能受 Claude 预训练数据污染影响；⑤单次实验评估，未进行多次重复以统计稳定性。

---

## 578. RoSHI: A Versatile Robot-oriented Suit for Human Data In-the-Wild

**arXiv ID:** 2604.07331 | [PDF](https://arxiv.org/pdf/2604.07331v1)

**作者:** Wenjing Margaret Mao `[一作]` (University of Pennsylvania), Antonio Loquercio `[通讯]` (University of Pennsylvania)

**通讯引用:** 3074 | [OpenAlex ID](https://openalex.org/A5075873122)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一套低成本、便携的混合传感器系统 RoSHI，利用九个消费者级 IMU 与 Project Aria 头戴式摄像头的协同工作，实时捕捉人类全身 3D 姿态、全球一致的根轨迹以及第一视角 RGB 视频，可在野外环境中无笼式、无外部摄像机完成大规模数据采集。

**💡 创新点**

创新点包括：① 通过视觉辅助的在体校准（无盒子、无规定姿势），实现 IMU 与全局坐标系的实时对齐；② 将稀疏 IMU 约束与基于 egocentric SLAM 的姿态生成 diffusion 模型结合，提升在遮挡和高速运动下的鲁棒性；③ 将捕获数据直接用于仿真-现实（sim‑to‑real）强化学习，训练可在 Unitree G1 人形机器人上成功部署的全身控制策略；④ 公开完整的硬件设计与软件栈，方便快速扩展与重用。

**🔧 技术方法**

技术方案主要包括：BNO085 低成本 9 轴 IMU 与自研固件、Project Aria 头戴摄像头 + SLAM、EgoAllo diffusion 模型进行姿态生成、骨骼方向约束与 Karcher 均值校准、基于深度强化学习（DeepMimic + BeyondMimic）实现机器人全身跟踪控制。

**📊 数据集**

数据集方面，作者收集了 11 条不同室内外运动序列（包含站立、移动、敏捷运动），并用 OptiTrack Motive 进行高精度标定，生成 SMPL‑X 关节和网格的地面真值；同时在评估时与 SAM 3D Body、IMU‑only、EgoAllo 等基线进行对比。

**📈 对比分析**

通过 MPJPE（厘米）和 JAE（度）两项指标与 OptiTrack 进行量化对比，RoSHI 在三个数据集上均取得最低 MPJPE（9.6–11.0 cm）和最佳或次优 JAE，明显优于 IMU‑only、EgoAllo 及 SAM 3D Body（后者受限于视角遮挡）。此外，利用 RoSHI 捕获的数据训练的仿真策略能够在 Unitree G1 上实现跑步、跳跃、挥手等复杂动态动作，验证了数据的可迁移性。

**⚠️ 局限性**

局限性包括：① 对不可直接观测关节的重建精度有限，尤其在复杂运动下可能出现扭曲伪影；② IMU 方向漂移仍需外部激光/视觉校准以维持长期一致性；③ 系统依赖头戴摄像头的视角，若主体完全超出视场会导致缺失；④ 低成本硬件在噪声与分辨率上不及高端 Xsens 等专业套装，适用场景受限于预算与对精度的严格要求。

---

## 579. Distilling Photon-Counting CT into Routine Chest CT through Clinically Validated Degradation Modeling

**arXiv ID:** 2604.07329 | [PDF](https://arxiv.org/pdf/2604.07329v1)

**作者:** Junqi Liu `[一作]` (Johns Hopkins University), Zongwei Zhou `[通讯]` (Johns Hopkins University)

**通讯引用:** 19832 | [OpenAlex ID](https://openalex.org/A5084104975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了一种利用模拟降质-提升方法，将高质量的光子计数CT（PCCT）图像的优势迁移到常规低质量CT（EICT）图像的技术。

**💡 创新点**

创新点在于：① 构造与临床验证的真实降质模拟器，将PCCT降至与EICT相匹配的低质量；② 在无配对扫描的前提下，学习逆向提升模型；③ 采用大规模持续学习的自动编码器和潜在扩散模型，提供可复用的CT特征表示。

**🔧 技术方法**

技术手段包括：持续学习自动编码器（训练于400k+ CT体积），潜在扩散模型（LDM）在降质后图像上训练；三种降质策略（稀疏视角、低剂量、常规降质）；像素损失、分割损失、HU一致性损失及对抗损失等多任务训练。

**📊 数据集**

数据集：14家医院、7国共计400k+ EICT扫描；约5k PCCT扫描；公开数据集（LIDC-IDRI、MIDRC、NLST、RSNA-STR、Luna16、LNDb19、DSB17 等）用于评估与下游检测；公开发布的1.5k+ PCCT‑类似质量的增强CT集，含放射科医生验证的体素级分割。

**📈 对比分析**

与多种基线（Swin2SR、Pix2Pix、SRGAN、SR3、NLM、NEED）比较，均在稀疏视角、低剂量、常规及混合降质下实现了最高+35.9% SSIM、+19.5% PSNR的提升；在三大外部检测数据集上提升检测F1最高+15.2%、AUC+10.5%；放射科医生评估显示临床可用性显著提升。

**⚠️ 局限性**

局限性：仅在胸部CT上验证；目前为二维切片处理，未实现完整三维实现；缺乏对其他解剖区域的泛化评估；模型训练依赖大规模高质量PCCT数据，实用性受限于此类数据获取。

---

## 580. Syntax Is Easy, Semantics Is Hard: Evaluating LLMs for LTL Translation

**arXiv ID:** 2604.07321 | [PDF](https://arxiv.org/pdf/2604.07321v1)

**作者:** Priscilla Kyei Danso `[一作]` (Stony Brook University), Omar Chowdhury `[通讯]` (Stony Brook University)

**通讯引用:** 1098 | [OpenAlex ID](https://openalex.org/A5070136662)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估现有大型语言模型（LLM）在将自然语言需求翻译成命题线性时序逻辑（LTL）公式的能力，提出六层评估框架并对比多种提示接口与模型。

**💡 创新点**

创新点在于：①构造公开可复现的多来源基准数据集与六阶评估流程；②将翻译任务重新表述为 Python 代码完成功能，显著提升性能；③结合 NuSMV 进行语义等价与追踪判断，实现语法与语义双重验证。

**🔧 技术方法**

技术上采用多模型（GPT‑3.5/4、Gemini、Claude 等）在三种提示方式（最小、详细、Python AST）下生成 LTL；使用 Levenshtein / Jaccard 对原子命题提取评估；利用 NuSMV 对公式等价、追踪满足性与违背性进行逻辑验证。

**📊 数据集**

使用的主要数据集包括：tricky1（Future‑LTL 306 条）、pastltl2（Past‑LTL 294 条）、book3（文本源 141 条）、trace4（对应轨迹）、prop5（命题逻辑 144 条）、syntax6（语法判定 299 条）以及 VERIFY 子集 verify7（安全需求 56 条）。

**📈 对比分析**

在最小提示下，LTL 公式的语法正确率约 48%，语义等价率仅 24%；采用详细提示可提升语法至 85%、等价至 31%；Python AST 接口进一步提高语义等价至 61‑70%，轨迹满足性分类 F1 超过 80%，轨迹生成 F1 在 55‑70% 范围。

**⚠️ 局限性**

主要局限包括：①语义等价仍低于语法正确率，主要受时间量词误置导致；②对原子命题的识别与映射仍易出错，尤其安全领域；③实验受数据泄漏与模型版本更新影响；④评测集中于短句子，未覆盖长篇、交叉引用的工业需求。

---

## 581. Evaluating In-Context Translation with Synchronous Context-Free Grammar Transduction

**arXiv ID:** 2604.07320 | [PDF](https://arxiv.org/pdf/2604.07320v1)

**作者:** Jackson Petty `[一作]` (New York University), Tal Linzen `[通讯]` (New York University)

**通讯引用:** 5545 | [OpenAlex ID](https://openalex.org/A5081824828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在低资源语言情境下的“上下文内翻译”(ICMT) 能力，采用同步上下文无关文法(SCFG)生成形式化语言对，并测评模型在仅给定文法与源句时的翻译准确性。

**💡 创新点**

创新点在于：① 用形式化 SCFG 作为可控、可验证的 ICMT 代理，精准隔离语法复杂度、句长、形态学和正字法对模型表现的影响；② 系统评估了模型在不同语法规模、句子长度、词序、形态标记和文字系统下的性能差异，揭示了非词序特征与正字法是主要瓶颈。

**🔧 技术方法**

技术上主要使用同步上下文无关文法生成句子对，并利用 GPT‑5 系列与 Gemma‑3 系列等大型语言模型进行推理；评价指标包括 exact‑match 准确率、词袋准确率以及传统字符串重叠度量。

**📊 数据集**

数据集为自定义的 SCFG 生成的句子对，规模覆盖 25–10,000 条规则，句长 3–50 词，保证与训练语料完全无交叉；未使用任何自然语言平行语料或手工标注数据。

**📈 对比分析**

比较方法是将模型生成的翻译与文法产生的金标准逐词比对；实验显示：文法规模和句子长度越大，准确率急剧下降；形态学标记与正字法差异导致性能显著衰退（如带音标的希伯来文几乎为 0%），而词序差异对性能影响不大。

**⚠️ 局限性**

局限性包括：① SCFG 仅捕捉结构化语法，未涵盖自然语言中多义、语义依赖与短语结构；② 词汇映射过于直接，缺乏多对一/一对多的自然翻译现象；③ 仅考察无示例情境，未探讨示例对模型表现的提升；④ 结果为上限，无法直接映射到实际自然语言翻译效果。

---

## 582. Intertemporal Demand Allocation for Inventory Control in Online Marketplaces

**arXiv ID:** 2604.07312 | [PDF](https://arxiv.org/pdf/2604.07312v1)

**作者:** Rene Caldentey `[一作]`, Tong Xie `[通讯]` (University of Chicago)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5101190338)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究平台如何通过对订单流的时序分配（intertemporal demand allocation）来影响第三方卖家的库存决策，并在保持公平（neutral）约束的前提下，调节卖家对平台履约（FBP）与自履约（FBM）的选择。

**💡 创新点**

创新点在于：①将信息设计的视角引入订单分配，利用对卖家订单流预测误差（root MSFE）的操控来影响安全库存；②证明在公平约束下，均匀分配给出最低可实现的MSFE，并提供通过低阶MA滤波（MA(1)或MA(2)）实现任意更高MSFE的构造；③揭示实现更高MSFE必需的非可逆性（non‑invertibility）及其对平台信息优势的意义。

**🔧 技术方法**

采用线性滤波、z 域内外因子分解（inner–outer factorization）、基于均值‑方差的库存（base‑stock）模型以及root MSFE分析；通过解析求解可实现的MSFE区间，并求平台在该区间内的最优选择。

**📊 数据集**

该研究基于理论模型，没有使用实际数据集；所有结果均为闭式推导或数值示例（如10个卖家的地理成本例子）。

**📈 对比分析**

由于是理论分析，作者没有与其它算法做实验比较；通过数值示例展示在统一分配与最优公平分配下，平台利润、FBP采用数和累计安全库存的差异，证明在公平约束下通过调节MSFE可以显著提升平台收益。

**⚠️ 局限性**

主要局限包括：假设需求为平稳、可逆；卖家采用最优一阶预测；忽略交付延迟、库存补货周期、失单与容量约束；公平约束仅限于均等均值与一阶MSFE，未考虑更广义的公平/多期预测不确定性；模型未包含卖家的战略定价或学习行为。

---

