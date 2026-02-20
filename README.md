# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-20 | 今日论文总数: 453

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. BanglaSummEval: Reference-Free Factual Consistency Evaluation for Bangla Summarization

**arXiv ID:** 2602.16843 | [PDF](https://arxiv.org/pdf/2602.16843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 2. Self-Evolving Multi-Agent Network for Industrial IoT Predictive Maintenance

**arXiv ID:** 2602.16738 | [PDF](https://arxiv.org/pdf/2602.16738v1)

**作者:** Rebin Saleh `[一作]` (Budapest University of Technology and Economics), Truong-Son Hy `[通讯]` (University of Alabama at Birmingham)

**通讯引用:** 216 | [OpenAlex ID](https://openalex.org/A5073178563)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了自演化多层次多智能体系统SEMAS，用于工业物联网预测性维护，能够在边缘、雾端和云端协同完成实时异常检测、剩余寿命预测和可解释性响应；

**💡 创新点**

创新点在于：①将多智能体按资源层次分布，形成边缘-雾端-云端三层架构；②利用PPO实现连续策略演化，自动调优阈值和模型权重；③融合共识投票与联邦聚合提升检测鲁棒性；④引入小语言模型生成可解释的维护建议；

**🔧 技术方法**

主要技术包括：多智能体系统、PPO强化学习、集成学习与共识投票、联邦学习聚合、轻量化小语言模型、MQTT通信与实时推理；

**📊 数据集**

实验使用两套工业数据集：Boiler Emulator（10k样本，36.8%异常）和Wind Turbine IIoT（500样本，45%故障）；

**📈 对比分析**

与两种基线（静态配置和规则式自适应）对比，SEMAS在F1、ROC‑AUC和实时延迟方面均优于基线；在Boiler上显著提升8.6% F1（p<0.001），在Wind Turbine上提升约2.4% F1；延迟从百毫秒级降至1ms以下，实现200–1500×速度提升；

**⚠️ 局限性**

局限性包括：初始化训练耗时约8小时、内存占用较高（≈3.8GB）、在数据表现良好的场景（Wind Turbine）提升有限、对冷启动性能依赖随机起始策略、LLM解释生成会增加额外延迟以及需要进一步的联邦隐私保障和跨域迁移能力。

---

## 3. StoryLensEdu: Personalized Learning Report Generation through Narrative-Driven Multi-Agent Systems

**arXiv ID:** 2602.17067 | [PDF](https://arxiv.org/pdf/2602.17067v1)

**作者:** Leixian Shen `[一作]` (Hong Kong University of Science and Technology), Huamin Qu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10920 | [OpenAlex ID](https://openalex.org/A5091466289)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个多代理系统，自动生成以学习目标为中心、叙事化、可交互的个性化学习报告，并支持后续问答；

**💡 创新点**

将数据分析、教师、讲故事三种代理与Hero's Journey叙事框架结合，实现洞察提取、教学反馈和故事化呈现的全流程自动化，并在报告中嵌入交互式问答；

**🔧 技术方法**

利用大型语言模型驱动的代理、学习目标图（图数据库）、统计显著性+市场份额算法进行洞察排序、Echarts可视化、交互式选择工具以及异步LLM+SQL后端生成问答与可视化配置；

**📊 数据集**

使用高中数学在线学习平台的真实学习日志（练习、测验、时间、准确率等）以及基于官方课程标准构建的学习目标图，涉及1000+同龄人对比数据；

**📈 对比分析**

通过问卷和访谈与传统文本报告、LA仪表板对比，学生/教师在报告清晰度、图表直观度和交互探索方面均得分4.5-5；叙事参与度评分中等3.4，定性反馈显示交互和叙事受欢迎；尚未量化学习效果，但展示了高可解释性和交互性；

**⚠️ 局限性**

主要限制：依赖结构化、目标对齐的数据，难以处理开放式任务；叙事模式“一刀切”，情感和语气缺乏个性化；信息密度高导致定位困难；缺少教师实时干预与课堂融合；需进一步量化学习成效。

---

## 4. Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking

**arXiv ID:** 2602.16746 | [PDF](https://arxiv.org/pdf/2602.16746v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究Transformer在模数97算术任务上的grokking现象，利用几何动力学方法识别并分析权重空间中的执行子流形及其曲率演化。

**💡 创新点**

创新点在于证明执行子流形在优化过程中经验上保持不变，曲率在正交方向累积并可提前预示泛化；通过因果干预显示正交梯度流是grokking的必要条件，而仅提升曲率不足以触发泛化。

**🔧 技术方法**

使用技术包括PCA特征分析、commutator defect（梯度非可交换性）测量、子流形投影、随机子空间基准、以及梯度子空间抑制与方向性干预等。

**📊 数据集**

数据集为模数97的六个二元运算（add、sub、mul、a²+b²、a²+ab+b²、a³+ab），配合多随机种子和广泛的学习率与权重衰减设置。

**📈 对比分析**

与无权衰减控制、慢速/快速训练模式、不同学习率等对照实验，发现相同几何模式，曲率爆发提前600–1600步，随着学习率降低预测窗口可达95%；但未给出具体准确率提升数值，只展示相对趋势。

**⚠️ 局限性**

局限性在于仅在小型Transformer与人工算术任务上验证，计算成本高，曲率测量与干预难以扩展至大型模型与真实数据集，通用性与可扩展性尚待进一步探索。

---

## 5. Large-scale online deanonymization with LLMs

**arXiv ID:** 2602.16800 | [PDF](https://arxiv.org/pdf/2602.16800v1)

**作者:** Simon Lermen `[一作]` (MATS), Florian Tramèr `[通讯]` (ETH Zurich)

**通讯引用:** 12402 | [OpenAlex ID](https://openalex.org/A5006851333)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型（LLM）设计并实现了一套可在大规模、跨平台和跨时间的匿名账户上执行去匿名化的完整管道，包含特征抽取、语义检索、推理与置信度校准，并在公开数据集上对其性能进行了系统评估。

**💡 创新点**

创新点主要体现在：①提出了以LLM为核心的多阶段去匿名化框架，可直接处理非结构化文本；②利用LLM自动抽取身份相关特征并生成语义嵌入，显著提升检索质量；③加入推理阶段与置信度校准，进一步过滤误匹配；④构建了可在真实环境中获得基准的实验设置（合成匿名化与账户分割），为大规模去匿名化提供了可复制的评估方法。

**🔧 技术方法**

核心技术包括：大型语言模型（Gemini、GPT‑5、Grok 等）用于文本总结、特征抽取、推理与置信度生成；文本嵌入模型（Gemini embedding、OpenAI embedding）用于高维语义检索；FAISS 索引实现近似最近邻搜索；LLM 代理与 Web 搜索工具实现端到端自动化；函数调用与 Swiss‑system 排序用于置信度校准。

**📊 数据集**

实验数据集涵盖：① Hacker News 与 LinkedIn 的跨平台匹配（338 对真实账号），② Reddit 电影讨论社区（r/movies 与 r/horror、r/MovieSuggestions、r/Letterboxd、r/TrueFilm、r/MovieDetails，共 9,781 名候选用户），③ 通过时间分割构建的 Reddit 评论数据（5,000 训练 / 5,000 测试查询，10,000 候选），④ Anthropic Interviewer 访谈转录（125 条，包含科学家真实身份），以及通过合成匿名化生成的多种评估集合。

**📈 对比分析**

与经典基线（手工特征、稀有度加权相似度）对比，LLM 方案在多项指标上大幅领先：在 HN‑LinkedIn 任务中，LLM+Reason 在 90% 精度下可达 68% 召回率；在 Reddit 电影任务中，LLM+Reason+高推理能在 90% 精度下实现 8.5% 召回率、99% 精度下 2.8% 召回率；经典基线在相同设置下几乎为 0% 召回。实验还表明，推理与置信度校准显著提升高精度下的召回，且 LLM 方法对候选池规模的鲁棒性远高于传统方法。

**⚠️ 局限性**

局限性包括：① 评估所用的真值数据多为公开链接或合成匿名化，可能高估真实环境中的匹配率；② 账户分割方法假设同一用户在不同时间/社区的行为相似，实际跨平台行为差异可能更大；③ LLM 推理与检索成本随规模增长，实际攻击成本仍较高；④ 研究未充分评估 LLM 训练数据中的记忆化影响；⑤ 论文未公开完整代码与数据，限制了外部复现与进一步研究。

---

## 6. Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents

**arXiv ID:** 2602.16855 | [PDF](https://arxiv.org/pdf/2602.16855v1)

**作者:** Haiyang Xu `[一作]` (Tongyi Lab, Alibaba Group), Ming Yan `[通讯]` (Tongyi Lab, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并发布了GUI‑Owl‑1.5原生GUI代理模型，支持多平台（桌面、移动、浏览器等）及指令/思考两种模式，能够完成GUI自动化、定位、工具调用、记忆与知识推理等多项任务。

**💡 创新点**

核心创新包括Hybrid Data Flywheel混合数据循环（模拟+云沙盒）提升数据质量、统一思考链CoT合成强化推理、MRPO多平台强化学习算法解决跨设备梯度干扰，以及三大能力提升策略（知识注入、CoT、Multi‑Agent协作）。

**🔧 技术方法**

技术路线基于Qwen3‑VL的预训练，三阶段训练（预训练、监督微调、RL），结合CoT生成、世界模型、工具/模型上下文协议（MCP）、设备条件策略、虚拟环境轨迹生成与视觉‑文本模型。

**📊 数据集**

使用了多种公开与自建数据集，包括OSWorld、AndroidWorld、WebArena、ScreenSpotPro、GUI‑Knowledge Bench、MemGUI‑Bench、MMBench‑GUI、MMGUI‑Bench、WindowsAgentArena等，并通过DAG合成、教程QA、VQA、负样本等方式扩充数据。

**📈 对比分析**

通过与20+公开GUI基准进行对比，GUI‑Owl‑1.5在OSWorld、AndroidWorld、ScreenSpotPro、OSWorld‑MCP、MemGUI‑Bench等任务中均取得SOTA成绩；在思考模式下相较指令模式提升显著；在open‑source模型中超越UI‑TARS、Qwen3‑VL、Claude‑4、Gemini等。

**⚠️ 局限性**

局限性包括：跨平台梯度干扰需交替训练，极端复杂或长周期任务收敛慢；虚拟环境仍需人工标注；大模型规模对edge部署有限制；对高度隐私敏感场景的安全与隐私控制尚待完善。

---

## 7. The Kinematics and Dynamics Theories of a Total Lagrangian Finite Element Analysis Framework for Finite Deformation Multibody Dynamics

**arXiv ID:** 2602.17002 | [PDF](https://arxiv.org/pdf/2602.17002v1)

**作者:** Zhenhao Zhou `[一作]` (University of Wisconsin Madison), Dan Negrut `[通讯]` (University of Wisconsin Madison)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于总拉格朗日(FEA)的可变形体动力学求解框架，支持大变形、大旋转、运动约束、接触摩擦以及多种几何约束。

**💡 创新点**

创新点在于将总拉格朗日有限元与增量拉格朗日-多体动力学耦合，统一处理运动约束、接触摩擦和内在弹塑性/粘性材料模型，并提供全参数化ANCF梁/壳元素以及10节点四面体单元的解析实现。

**🔧 技术方法**

采用总拉格朗日有限元、ANCF插值、SVK和Mooney‑Rivlin超弹性模型、Kelvin‑Voigt粘性模型、增量拉格朗日求解、罚项接触法以及Mindlin摩擦模型。

**📊 数据集**

未使用公开数据集，主要通过理论推导和数值实验（如单体梁、壳体、体单元以及多体接触演示）进行验证。

**📈 对比分析**

方法通过内部残差和增量拉格朗日循环求解；与传统UL‑FEA或刚体多体动力学进行对比，能够处理大位移/大旋转且保持数值稳定；性能表现需依赖实现细节，文中未给出具体数值。

**⚠️ 局限性**

局限包括SVK模型在大压缩时出现负刚度；对极大变形或高频振动的数值稳定性需进一步验证；接触摩擦采用罚项可能导致数值不光滑；实现复杂度高，需显式/隐式时积分和雅可比矩阵计算成本大。

---

## 8. Characterizing the Predictive Impact of Modalities with Supervised Latent-Variable Modeling

**arXiv ID:** 2602.16979 | [PDF](https://arxiv.org/pdf/2602.16979v1)

**作者:** Divyam Madaan `[一作]` (New York University), Kyunghyun Cho `[通讯]` (Genentech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

PRIMO是一种监督式潜在变量插补模型，用来在多模态学习中处理缺失模态，并量化缺失模态对预测的影响；

**💡 创新点**

创新点在于：①使用潜在变量直接建模缺失模态对目标预测的相关性，避免传统重建驱动的生成模型；②在训练时同时优化完整模态和缺失模态的ELBO，实现两种场景下的联合学习；③通过在推理时采样潜在变量并计算预测分布的方差（TVD）来获得实例级的缺失模态影响度量；

**🔧 技术方法**

采用变分推理框架（ELBO）、条件先验与后验网络、重参数化技巧、Monte Carlo采样、方差量化指标、Dirichlet Process高斯混合模型聚类等技术；

**📊 数据集**

在合成XOR、Audio‑Vision MNIST（AV‑MNIST）以及医疗数据MIMIC‑III（死亡率预测与ICD‑9分类）三个数据集上进行实验；

**📈 对比分析**

与单模态基线、完整模态基线、生成式基线（MVAE、MMVAE）、判别式缺失基线（CMMD）和插补基线（LVAE）比较。PRIMO在缺失模态时与单模态基线性能相当，在完整模态时与多模态基线相当；并通过方差指标和聚类分析展示了实例级的缺失模态影响；

**⚠️ 局限性**

局限性包括：①缺失模态影响度量难以在无标签的真实环境中验证；②当前评测主要集中在二模态（音频/视觉/文本）和有限的缺失模式，缺乏多模态、多缺失模式的基准；③模型扩展到更多模态时需要进一步研究；

---

## 9. HiVAE: Hierarchical Latent Variables for Scalable Theory of Mind

**arXiv ID:** 2602.16826 | [PDF](https://arxiv.org/pdf/2602.16826v1)

**作者:** Nigel Doering `[一作]` (University of California San Diego), Tauhidur Rahman `[通讯]` (University of California San Diego)

**通讯引用:** 2121 | [OpenAlex ID](https://openalex.org/A5101486879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 HiVAE，一种基于 Belief‑Desire‑Intention（BDI）结构的三层层次 VAE，结合轨迹与图编码，用于在大规模轨迹与图数据上进行 Theory of Mind 推理，预测代理的隐藏目标。

**💡 创新点**

创新点在于将 BDI 结构融入层次 VAE，设计了轨迹‑图融合编码与多层次隐变量，且提出自监督对齐策略以解决潜在表征与真实心理状态的对齐问题。

**🔧 技术方法**

使用了 Transformer 与图注意力网络进行编码，三层层次 VAE 结构，Brier score 与 Wilcoxon 检验等评估指标，以及对比学习、辅助预测等自监督方法。

**📊 数据集**

在真实校园图（3,185 个节点、9,000 条边，来自 OpenStreetMap）上生成的 100,000 条合成行人轨迹（100 个代理、每个 1,000 条）进行训练和评估。

**📈 对比分析**

与多种基线（包括基于距离的模型、无层次模型等）在不同观察比例下进行 Brier 分数比较，HiVAE 在所有场景下均取得最低 Brier 分数，显著优于最接近的基线（p<0.01）。

**⚠️ 局限性**

潜在变量未与实际心理状态显式对应，仅通过重构和目标预测进行监督，缺乏显式的心理状态归一化；同时模型对个体差异处理不足，需要加入元学习个体嵌入。

---

## 10. BankMathBench: A Benchmark for Numerical Reasoning in Banking Scenarios

**arXiv ID:** 2602.17072 | [PDF](https://arxiv.org/pdf/2602.17072v1)

**作者:** Yunseung Lee `[一作]` (KakaoBank Corp), Jaegul Choo `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5874 | [OpenAlex ID](https://openalex.org/A5047912015)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并构建了面向日常银行业务的数值推理基准 BankMathBench，并通过该基准对多种大型语言模型进行评测和微调。

**💡 创新点**

创新点在于：①提供了覆盖存款、储蓄、贷款三类核心银行产品的多难度（Basic/Intermediate/Advanced）问题集；②使用自动化生成+人工校验的流水线保证数据质量；③展示了工具增强微调（Tool‑Augmented SFT）在复杂推理上的显著提升。

**🔧 技术方法**

主要技术包括：①GPT‑4o 与 o1‑mini 生成问题、解答与推理；②双方案可验证的解法生成；③在推理步骤中嵌入 <calc> 标签并调用外部计算器实现精确算术；④使用 QLoRA 进行 4‑bit 量化微调。

**📊 数据集**

使用的数据集为 BankMathBench，包含 13,839 个中韩双语题目，每个题目包含问题、答案和分步推理。

**📈 对比分析**

实验通过零样本、SFT 与工具增强 SFT 三种设置，对比 8‑B 级以上模型与 1‑B 级轻量模型，发现微调后准确率提升 30–70%，工具增强后在 Intermediate/Advanced 级别提升超过 70%；闭源模型在零样本下已领先，但在微调后仍被训练模型追平。

**⚠️ 局限性**

局限性：数据仅覆盖存款、储蓄、贷款三类产品，未包含基金、保险等业务；因聚焦核心产品，可能导致对其他金融产品的偏差与泛化不足。

---

## 11. Not Only for Developers: Exploring Plugin Maintenance for Knowledge-Centric Communities

**arXiv ID:** 2602.17018 | [PDF](https://arxiv.org/pdf/2602.17018v1)

**作者:** Giovanni Rosa `[一作]` (Universidad Rey Juan Carlos), Raula Gaikovina Kula `[通讯]` (University of Osaka)

**通讯引用:** 2431 | [OpenAlex ID](https://openalex.org/A5091820517)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 Obsidian 插件生态进行探索性研究，首先利用 LLM 提取 README 关键短语并结合 BERTopic 进行主题建模，构建了 26 个插件主题并归纳为 6 大类；随后通过 GitHub API 收集 396 个插件的拉取请求数据，计算各主题的 PR 总量、平均值、标准差等统计指标，评估插件的维护活跃度。

**💡 创新点**

研究的创新点在于：①首次系统性分析面向知识中心社区的插件生态，②将 LLM 与传统文本聚类技术结合，构建数据驱动的插件主题层次；③通过拉取请求分析展示非开发者生态也能形成成熟的维护模式。

**🔧 技术方法**

技术手段包括：gpt‑oss‑120b LLM 生成关键短语与主题标签；Qwen3‑Embedding 生成文本嵌入；BERTopic（UMAP + HDBSCAN）进行聚类和主题提取；PyGitHub 调用 GitHub API 获取 PR 统计；Python pandas/scikit‑learn 进行数据处理与统计分析。

**📊 数据集**

使用的数据集为 2025 年 10 月从 Obsidian 官方插件列表 community‑plugins.json 抽样得到的 396 个插件（原始共 2,667 个），每个插件的 README 被自动提取关键短语后与插件名称、描述合并形成文档。

**📈 对比分析**

对比方法：按主题聚合 PR 数量，计算总和、均值、标准差、最大/最小等指标；未与其他生态进行直接性能对比，但通过展示高活跃主题（如 Task & Calendar Integration）与低活跃主题（如 Quick Line Editing Toolkit）间的差异，证明不同主题间维护强度存在显著差别，整体维护模式与成熟软件生态相似。

**⚠️ 局限性**

局限性：①样本仅 396 个插件，可能遗漏尾部功能；②主题与标签均基于 LLM，存在提示、模型偏差风险；③缺乏金标准分类，聚类结果难以定量验证；④仅使用 PR 指标评估可持续性，未考虑 issue、commit、发布频率等多维度指标；⑤非开发者社区的贡献模式与传统生态差异可能影响结果的普适性。

---

## 12. A Long-term Value Prediction Framework In Video Ranking

**arXiv ID:** 2602.17058 | [PDF](https://arxiv.org/pdf/2602.17058v1)

**作者:** Huabin Chen `[一作]` (Alibaba Group), Yuning Jiang `[通讯]` (Alibaba Group)

**通讯引用:** 3905 | [OpenAlex ID](https://openalex.org/A5074655314)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究并实现了在短视频推荐排名阶段的长期价值（LTV）预测框架，通过位置去偏、细粒度归因和跨时段作者建模提升长期价值预测。

**💡 创新点**

创新点包括：① 引入位置感知分位数去偏（PDQ）实现无结构改动的去偏；② 提出多维归因机制并用混合损失过滤噪声；③ 使用日级作者LTV与延迟标签的双流训练以捕获跨天用户兴趣。

**🔧 技术方法**

采用的技术包括：分位数回归、Tweedie损失、混合损失、Multi‑Scale Embedding Fusion、Target Attention、PPNet 风格多目标学习以及双流协同训练。

**📊 数据集**

使用的数据集来自淘宝短视频平台，涵盖 23M 用户、22M 视频，15 天流量日志（前 14 天训练，1 天测试）。

**📈 对比分析**

与基线滑动时间回归相比，离线 XAUC 提升 0.013、MSE 降至 0.09；上线时 VV 提升 2.49%，LTV3 提升 0.21%，QA VV 提升 4.03%，同时保持短期指标不下降。

**⚠️ 局限性**

局限性包括：仅对作者进行跨时段建模，未扩展到话题/风格；归因学习仍基于手工定义的信号；缺少对负向信号的建模。

---

## 13. Haskell meets Evariste

**arXiv ID:** 2602.16809 | [PDF](https://arxiv.org/pdf/2602.16809v1)

**作者:** Paulo R. Pereira `[一作]` (Checkmarx), Jose N. Oliveira `[通讯]` (HASLab U.Minho & INESC TEC)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过提出一种“易-难分割”设计模式，将程序说明中的易懂文本与形式化规范相结合，利用Galois连接的数学理论对Haskell及其他语言的标准库函数进行精确描述和推导。

**💡 创新点**

创新点在于将Galois连接作为通用的规范模式应用于软件文档，既保持了可读性，又提供了形式化证明的路径，从而在函数库说明中实现了自动化属性推导和实现验证。

**🔧 技术方法**

主要技术包括：形式化规范（谓词逻辑、序关系）、Galois连接理论、可组合的前后处理（前后关系）、以及对Haskell函数的模式匹配与推导。

**📊 数据集**

使用的数据集为Haskell的Data.List库中的函数（如zip、takeWhile、takeWhile, filter等）以及对应的Swift、Python、F#、Elixir等语言中的同名函数，用以比较说明质量。

**📈 对比分析**

方法比较是通过对不同语言文档的语义一致性与形式化等价性进行手工验证，并在Haskell实现中证明推导的正确性；本文未给出运行时性能指标，而是侧重于说明清晰度和可验证性。

**⚠️ 局限性**

局限性包括：需要对所描述函数具有良好的概念化（如前缀、子列、后缀），并且易-难分割模式并不适用于所有函数（如非连续保留元素的函数）；实现过程仍依赖手工推导，对大规模库自动化支持不足。

---

## 14. Archetypes and gender in fiction: A data-driven mapping of gender stereotypes in stories

**arXiv ID:** 2602.17005 | [PDF](https://arxiv.org/pdf/2602.17005v1)

**作者:** Calla Glavin Beauregard `[一作]`, Peter Sheridan Dodds `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

由于缺乏具体论文内容，无法确定其研究内容

**💡 创新点**

创新点无法判断

**🔧 技术方法**

使用的技术未知

**📊 数据集**

使用的数据集未知

**📈 对比分析**

方法比较与性能表现无法评估

**⚠️ 局限性**

论文的局限性无法确定

---

## 15. OpenSage: Self-programming Agent Generation Engine

**arXiv ID:** 2602.16891 | [PDF](https://arxiv.org/pdf/2602.16891v1)

**作者:** Hongwei Li `[一作]` (University of California Santa Barbara), Dawn Song `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一套名为Open Self-programming Agent Generation Engine（Open Sage Agent）的Agent Development Kit，能让大语言模型自动生成 agent 拓扑、动态写作工具以及层次化图结构记忆，极大提升 agent 的自适应性和性能。

**💡 创新点**

核心创新在于实现 AI‑centered 设计，使 LLM 能自行构建子 agent 结构、生成与任务匹配的专用工具，并通过图数据库实现短期与长期记忆的自动管理，从而打破传统手工设计的瓶颈。

**🔧 技术方法**

采用大语言模型的自回归推理、Meta‑Tools 进行工具合成、Docker 容器化隔离与异步执行、Neo4j 图数据库存储记忆，以及专门的记忆代理（Memory Agent）进行查询与更新。

**📊 数据集**

在三大主流基准上进行评测：VulnerabilityBench（1507 C/C++ 漏洞）、Terminal Tasks（89 终端任务）、SWEbench Python（266 任务）以及长时序对话数据，展示了跨域泛化能力。

**📈 对比分析**

通过与 Claude、OpenHands、OpenAI、SWE‑agent 等现有 ADK 以及公开排行榜进行对比，Open Sage Agent 在所有基准上均跑出最高解答率（某些任务提升 20% 以上），并在排行榜上名列第一。

**⚠️ 局限性**

当前仍受限于模型能力，功能使用不完整、偶尔会产生错误或无用的工具与子 agent；对资源（如容器与并行执行）的需求较高，且需要更强的 LLM 才能充分发挥设计潜力。

---

## 16. LLM-WikiRace: Benchmarking Long-term Planning and Reasoning over Real-World Knowledge Graphs

**arXiv ID:** 2602.16902 | [PDF](https://arxiv.org/pdf/2602.16902v1)

**作者:** Juliusz Ziomek `[一作]` (University of Oxford), Ilija Bogunovic `[通讯]` (University of Basel)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LLM-WikiRace 基准，用于评估大语言模型在真实维基百科知识图谱上的长程规划、推理与世界知识运用。

**💡 创新点**

创新点在于将规划、语义推理与世界知识统一到单一文本环境，并通过难度分层（短、中、长路径）揭示模型的规划瓶颈和循环错误。

**🔧 技术方法**

采用大语言模型（Gemini‑3、GPT‑5、Claude Opus 4.5 等）配合强化学习与提示工程，实现逐步链接选择，评估成功率、子最优步骤和代价。

**📊 数据集**

使用 2025‑06 截止的 Wikipedia 超链接图谱（约 549k 页）及其 3 个难度级别（3/4、5/6、7/8 步最短路径）构建测试对。

**📈 对比分析**

与多种模型、开放源代码 LLaMA、Gemma、Apertus 等进行对比；Gemini‑3 在易难度上 90%+ 成功率，难度最高时仅 23%；相比人类基准模型成功率相当但子最优步数更少。

**⚠️ 局限性**

主要局限在于即使拥有强大世界知识，模型仍难以在长路径上进行有效再规划，易陷入循环、缺乏探索，硬难度仍低于 25% 的成功率。

---

## 17. Guided Exploration of Sequential Rules

**arXiv ID:** 2602.16717 | [PDF](https://arxiv.org/pdf/2602.16717v1)

**作者:** Wensheng Gan `[一作]` (Jinan University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 133675 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对序列规则的目标挖掘问题，提出了基于查询规则的上界剪枝、位图信息表和相似度筛选等技术，实现了高效、可定制的序列规则挖掘框架。

**💡 创新点**

创新点包括：①使用基本、左、右上界过滤无效项；②从查询规则直接构造起始规则，避免 1×1 规则；③引入改进的 Jaccard 与 Dice 相似度作为终止条件，进一步降低搜索空间。

**🔧 技术方法**

采用 Rule‑Growth 基础算法、UB/TUB 上界估计、位图/信息表、前缀/后缀剪枝、相似度阈值终止以及 Java 实现。

**📊 数据集**

使用 4 个真实数据集（Bible、Sign、Kosarak10k、Leviathan）和 2 个合成数据集（Syn20K、Syn40K）进行实验。

**📈 对比分析**

与现有基线（RuleGrowth、US‑Rule、TaSRM）在频繁规则与高价值规则挖掘任务中比较，FSSR 在 FPM 任务上与 TaSRM 性能相当，FSUR 在 UPM 任务上实现数倍到十倍的速度提升，并显著减少扩展次数。

**⚠️ 局限性**

局限性包括：在稠密小型数据集上上界效果有限；相似度阈值需手工调参，可能导致漏检；仅支持单条查询规则，缺乏多规则或复杂查询支持；未覆盖不确定序列与隐私保护场景。

---

## 18. CalmReminder: A Design Probe for Parental Engagement with Children with Hyperactivity, Augmented by Real-Time Motion Sensing with a Watch

**arXiv ID:** 2602.16893 | [PDF](https://arxiv.org/pdf/2602.16893v1)

**作者:** Riku Arakawa `[一作]` (Carnegie Mellon University), Mayank Goel `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3388 | [OpenAlex ID](https://openalex.org/A5030011411)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过儿童腕表收集IMU运动数据，实时识别其平静时刻，并在父母手机上以“即时提示”方式触发反思与正向强化；在为期四周的居家研究中对比四种通知策略（无通知、每小时、随机、平静时刻触发）来评估其对父母意识、参与和自我设计干预的影响。

**💡 创新点**

①将实时运动感知与即时提示结合，为父母提供基于孩子当前情绪状态的“即刻”干预机会；②将系统设计为可供父母自行配置与改造的探针，展示家长主动设计干预的可能性；③在同一研究框架内系统性比较多种通知时机与频率。

**🔧 技术方法**

Apple Watch自定义App（持续采集IMU数据）、服务器端轻量级能量特征提取与线性回归模型、安卓/IOS父母端App（推送通知、问卷收集）、Web实验者仪表盘。

**📊 数据集**

内部数据集：约8–10小时/日的加速度能量序列（每5分钟上传一次），父母对孩子活跃度的5点量表标签，以及对应的日常和即时问卷文本。

**📈 对比分析**

通过统计响应率、父母认为平静时刻的匹配比例（calm-only 78%）以及自评问卷的Likert得分来比较。模型个性化后R²从0.25提升到0.44，平静检测在实际使用中对父母感知的对齐率相当高；不同通知策略在父母体验上的差异主要体现在可管理性与情感连接感，而非显著的客观行为改变量。

**⚠️ 局限性**

样本量小、性别偏向男性、部分高压力父母因任务负担中途退出、缺乏连续的真实标注导致模型评估受限、系统仅提供运动能量而非上下文信息，可能限制父母对通知的信任与有效性。

---

## 19. A Reversible Semantics for Janus

**arXiv ID:** 2602.16913 | [PDF](https://arxiv.org/pdf/2602.16913v1)

**作者:** Ivan Lanese `[一作]` (University of Bologna), Germán Vidal `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 1513 | [OpenAlex ID](https://openalex.org/A5051613376)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了一种基于程序计数器的可逆小步语义，用来描述可逆编程语言 Janus 的执行过程，并证明其与传统基于栈的语义及已有的大步语义等价。

**💡 创新点**

创新点在于：① 通过将代码块替换为唯一标识符（标签）并使用控制流图（CFG）实现可逆性；② 设计了前向和后向两套规则，使得每一步都可以被逆转；③ 提供了完整的可逆语义证明和循环层 lemma，弥补了先前 Janus 小步语义不可逆的问题。

**🔧 技术方法**

技术手段主要包括：
- 程序计数器与标签化语句；
- 构造控制流图（CFG）与其逆图；
- 定义前向/后向转移规则并使用逆语义函数 ℐ；
- 通过结构归纳证明可逆性与等价性。

**📊 数据集**

本工作没有使用任何实验数据集，所有结果均为理论证明。

**📈 对比分析**

比较方法为形式化证明：作者给出了从大步语义推导小步语义的完整性与保真性，并证明新语义与原小步语义在可执行序列上的等价性。由于是理论工作，没有性能指标可提供。

**⚠️ 局限性**

局限性：
- 仅适用于无局部变量、无参数的简化 Janus；
- 仍缺乏对并发、错误处理、堆栈等高级特性的支持；
- 语义定义虽然可逆，但实现上仍需维护完整的 CFG 与逆图，可能在大程序中产生空间和时间开销；
- 本文未给出实现或实测结果，实际可逆编程的可行性与效率仍待验证。

---

## 20. Exploring LLMs for User Story Extraction from Mockups

**arXiv ID:** 2602.16997 | [PDF](https://arxiv.org/pdf/2602.16997v1)

**作者:** Diego Firmenich `[一作]` (Universidad Nacional de la Patagonia), Leonardo Morales `[通讯]` (Universidad Nacional de la Patagonia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合大语言模型和领域专属词典（LEL），从高保真 mockup 自动生成用户故事。

**💡 创新点**

通过在 prompt 中加入 LEL，使得生成的用户故事在术语准确性和领域适配性上显著提升。

**🔧 技术方法**

使用大语言模型（如 ChatGPT/类似 LLM）配合图像输入处理、prompt 工程和手工构建的 LEL 词典。

**📊 数据集**

自定义 mockup 数据集，包括 YouTube 页面、LeafLab 生物学应用以及多种操作场景（添加、删除、组合等）。

**📈 对比分析**

对比有无 LEL 的评分体系（人工评估 0‑5 分），结果显示有 LEL 时平均得分从 3‑4 分提升到 5 分，表明生成的用户故事更完整、更准确。

**⚠️ 局限性**

限制：词典需人工维护，覆盖范围有限；对复杂或多模态需求的鲁棒性尚未验证；LLM 可能引入偏差或误解。

---

## 21. Evidotes: Integrating Scientific Evidence and Anecdotes to Support Uncertainties Triggered by Peer Health Posts

**arXiv ID:** 2602.16900 | [PDF](https://arxiv.org/pdf/2602.16900v1)

**作者:** Shreya Bali `[一作]` (Carnegie Mellon University), Mayank Goel `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3388 | [OpenAlex ID](https://openalex.org/A5030011411)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发并评估了一款名为 Evidotes 的 Chrome 扩展，能够在 Reddit 健康讨论帖旁边提供由科学研究和同类用户故事组成的“证据视角”，帮助用户在遇到健康不确定性时获得信息与情绪支持。

**💡 创新点**

创新点在于：① 将同类健康故事视为产生不确定性的触发器；② 通过三种可选视角（深入探究、聚焦积极、整体大局）让用户主动控制信息检索与呈现；③ 在单条声明中融合科学与个人叙事，形成“信息共生”而非简单的层级评估。

**🔧 技术方法**

技术方法包括：基于 Retrieval‑Augmented Generation (RAG) 的 LLM（GPT‑4o‑mini）生成查询、并从 PubMed（医学论文）和 Reddit（同类帖子）检索 10 条结果，随后 LLM 合成可信声明并引用原始来源；前端使用 Vue.js 构建浏览器扩展，后端使用 Flask 与 Agno‑AI 进行 API 调度。

**📊 数据集**

数据集：收集了 Reddit 上的健康社区帖子和对应的 PubMed 文献；用户实验使用 17 名慢性病患者的 77 条帖子、130 次视角交互，累计 293 条合成声明。

**📈 对比分析**

与传统浏览相比，Evidotes 在信息满意度上提升 43%（p<.001，d=1.49）并将情绪压力降低 44%（p<.001，d=1.23）。在实验中并未与自动化算法做直接比较，但通过对照的前后测量表明系统有效改善了用户的情绪与信息体验。

**⚠️ 局限性**

局限性：样本量小（仅 17 人，单次实验）；仅适用于文本帖，无法处理图片或视频内容；依赖 LLM 生成，存在轻微的幻觉与误引用风险；实验聚焦于 Reddit，无法直接推广到其他社交平台。

---

## 22. A Conceptual Hybrid Framework for Post-Quantum Security: Integrating BB84 QKD, AES, and Bio-inspired Mechanisms

**arXiv ID:** 2602.16922 | [PDF](https://arxiv.org/pdf/2602.16922v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6`

---

## 23. ReIn: Conversational Error Recovery with Reasoning Inception

**arXiv ID:** 2602.17022 | [PDF](https://arxiv.org/pdf/2602.17022v1)

**作者:** Takyoung Kim `[一作]` (University of Illinois Urbana-Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 9752 | [OpenAlex ID](https://openalex.org/A5068709817)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在不改模型参数或系统提示的情况下，利用测试时外部推理模块帮助LLM对话代理在出现用户误导或模糊请求时进行错误诊断与恢复。

**💡 创新点**

引入“rein”插入式推理机制，在运行时注入错误检测与恢复计划，避免对模型参数或提示的改动，同时突破指令层级约束。

**🔧 技术方法**

采用外部推理（inception）模块与工具调用、JSON schema恢复工具、功能调用框架以及对话模拟与评估。

**📊 数据集**

对现有任务基准（如对话任务集合）进行人工过滤与精炼，构建98个会话、588个上下文，覆盖已知与未知错误类型。

**📈 对比分析**

与不使用rein、以及两种提示修改基线（NPI与SR）对比，评估任务完成率；rein在所有组合中显著提升性能，优于提示修改，并能识别未见错误。

**⚠️ 局限性**

受限于错误检测模型的规模与长上下文理解，较小模型激活率低；实验集中在特定域与模拟环境，真实用户多样性与安全性需进一步验证。

---

## 24. BEMEval-Doc2Schema: Benchmarking Large Language Models for Structured Data Extraction in Building Energy Modeling

**arXiv ID:** 2602.16926 | [PDF](https://arxiv.org/pdf/2602.16926v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 25. Fail-Closed Alignment for Large Language Models

**arXiv ID:** 2602.16977 | [PDF](https://arxiv.org/pdf/2602.16977v1)

**作者:** Zachary Coalson `[一作]` (Oregon State University), Sanghyun Hong `[通讯]` (Oregon State University)

**通讯引用:** 719 | [OpenAlex ID](https://openalex.org/A5102751625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过逐步识别并消除已学习的拒绝方向，在训练时强制模型构建多条互不依赖的拒绝机制，形成 fail‑closed 对齐。

**💡 创新点**

提出 fail‑closed 对齐原则，利用多特征 ablation 与梯度优化的迭代训练方法显著提升对 prompt‑based jailbreak 的鲁棒性。

**🔧 技术方法**

使用梯度优化拒绝方向（RDO）、多特征 ablation（MFA）、LoRA 参数高效微调、对齐训练损失组合以及 KL 正则化等技术。

**📊 数据集**

数据集包括 SaladbBench、Alpaca、HarmBench、AdvBench、HarmBench judge、XSTest、BoolQ、RTE、HellaSwag、WinoGrande、ARC‑C、OpenBookQA。

**📈 对比分析**

与 RT、CAT、LAT、DeepRefusal 等基线对比，四个模型、四种 jailbreak 攻击下 ASR ≤4%，平均降低 92–97%；保持 86% 以上的 over‑refusal、约 61% 的零样本准确率；LoRA 版本亦能保持相近性能。

**⚠️ 局限性**

仍需多轮迭代与一定训练成本，后期学习的拒绝方向可能非线性，机制解释有限；对极大模型及更复杂攻击的通用性尚待验证。

---

## 26. RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models

**arXiv ID:** 2602.17053 | [PDF](https://arxiv.org/pdf/2602.17053v1)

**作者:** Yunseok Han `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**通讯引用:** 983 | [OpenAlex ID](https://openalex.org/A5024989829)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向大型推理模型的推理可信度评估框架与基准（RFEval），通过输出层的反事实干预检验模型推理的忠实性；

**💡 创新点**

①将可信度定义为两个可检验条件——立场一致性与因果影响；②设计了对比条件下的输出级干预评估流程；③构建包含七种任务、7186条样本的公开基准数据集；

**🔧 技术方法**

使用大语言模型生成反事实推理、基于LLM的立场提取与错误识别、统计分析与回归模型（WLS、固定效应）评估模型可信度与准确度的关系；

**📊 数据集**

RFEval数据集（7186条样本），涵盖代码生成、数学推理、逻辑推理、表格推理、上下文理解、法律决策和论文评审七类任务；

**📈 对比分析**

与12个公开LRM（Qwen3、DeepSeek-R1、gpt-oss、MiMo-7B、Magistral、Llama-3.3）在RFEval上对比，平均对比条件下可信度从约24%到73%不等；模型规模并非唯一决定因素，RL后训练往往降低可信度；相对任务准确率，可信度与准确率无显著关联；

**⚠️ 局限性**

局限性包括：①依赖LLM作为评判者，存在偏差与误差；②基准覆盖任务有限，难以覆盖所有高风险应用；③反事实推理生成过程可能引入人工偏好，且部分干预在封闭API模型上不可行；

---

## 27. Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees

**arXiv ID:** 2602.16823 | [PDF](https://arxiv.org/pdf/2602.16823v1)

**作者:** Itamar Hadad `[一作]` (Hebrew University of Jerusalem), Shahaf Bassan `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5027249175)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套基于神经网络验证的自动电路发现框架，能够给出在连续输入域和补丁域下具有可证明鲁棒性的电路，并满足多种最小化准则。

**💡 创新点**

首次将输入鲁棒性、补丁鲁棒性与电路最小化三类可证明保证统一在同一算法框架中，并揭示了电路单调性与这些保证之间的理论关联，导致算法收敛到子集最小甚至基数最小电路。

**🔧 技术方法**

利用神经网络验证技术（α–β‑CROWN）对电路与完整网络进行双胞胎编码（Siamese encoding），构造可验证的输入与补丁域约束；同时使用最小冲突集（MHS）和多线程并行求解实现基数最小化。

**📊 数据集**

在MNIST、CIFAR‑10、GTSRB和TaxiNet四个常用视觉模型上进行实验，采用与VNN‑COMP竞赛中相同的网络架构。

**📈 对比分析**

与传统基于采样的电路发现方法对比，验证式方法在输入和补丁鲁棒性上均达到了100%可靠性，尽管计算时间更长；在最小化实验中，MHS算法能逼近或达到基数最小解，速度慢但效果最佳。

**⚠️ 局限性**

主要限制在于需调用神经网络验证器，受限于当前验证器对大型模型的可扩展性；同时，MHS求解在最坏情况仍为NP‑完备，实际计算仍受限于图规模。

---

## 28. MALLVI: a multi agent framework for integrated generalized robotics manipulation

**arXiv ID:** 2602.16898 | [PDF](https://arxiv.org/pdf/2602.16898v1)

**作者:** Iman Ahmadi `[一作]` (Sharif University of Technology), Babak Khalaj `[通讯]` (Sharif University of Technology)

**通讯引用:** 2322 | [OpenAlex ID](https://openalex.org/A5016732612)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MALLVi 多代理闭环框架，实现机器人操纵任务的分解、感知、推理和执行；

**💡 创新点**

创新点在于将任务拆分为多个专用 LLM 代理（Decomposer、Localizer、Thinker、Reflector 等），并通过 Reflector 进行视觉反馈闭环，实现针对性错误检测与局部恢复；

**🔧 技术方法**

采用大语言模型（如 GPT‑4.1‑mini）、视觉语言模型（Qwen‑VL、LLaMA‑Vision）、SAM、GroundingDINO、OWL‑V2 等多模技术，并在 VIMABench、RLBench 以及真实机器人上部署；

**📊 数据集**

使用的数据集包括 VIMABench、RLBench 以及自制的真实世界任务集合（Place Food、Put Shape、Stack Blocks 等）；

**📈 对比分析**

通过与单代理、无 Reflector、MALMM、VoxPoser、ReKep、Wonderful Team、CoTDiffusion、PERIA、PerAct 等基线在同一任务上对比，MALLVi 在真实任务、VIMABench 和 RLBench 上多项指标均取得最高成功率（如真实任务 100%/95%/90% 等）；

**⚠️ 局限性**

主要局限在于依赖预定义的原子动作，缺乏低层执行的自适应能力；对极端动态或复杂接触环境的处理仍有限；开源模型在复杂多模任务上表现不如专有模型。

---

## 29. SAGE: Structure Aware Graph Expansion for Retrieval of Heterogeneous Data

**arXiv ID:** 2602.16964 | [PDF](https://arxiv.org/pdf/2602.16964v1)

**作者:** Prasham Titiya `[一作]` (Arizona State University), Dan Roth `[通讯]` (University of Pennsylvania)

**通讯引用:** 30160 | [OpenAlex ID](https://openalex.org/A5023802054)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结构感知图扩展（SAGE）框架，用于检索异构语料库中的多跳证据。

**💡 创新点**

创新点在于：① 离线构建基于元数据相似度的chunk级图，避免实体抽取成本；② 线上基于种子-扩展-再排序的统一检索策略；③ 对显式schema图引入agentic检索，利用Cypher实现结构约束。

**🔧 技术方法**

技术包括：语义分块、元数据嵌入、基于百分位数的边筛选、混合稠密+稀疏检索、LLM驱动检索计划、图邻居扩展与再排序。

**📊 数据集**

使用OTT‑QA（文本+表格）和STaRK（AMAZON、MAG、PRIME）三大异构数据集。

**📈 对比分析**

与多种基线（BM25、dense、hybrid、GNN、agentic、finetuned等）对比，SAGE在OTT‑QA Recall@20提升5.7点，在STaRK Recall@20提升8.5点，且无训练、推理成本低。

**⚠️ 局限性**

局限性包括：依赖初始种子质量、对LLM的依赖导致延迟、图边构造需手动阈值、对结构不完整的数据易失效、未处理多模态扩展。

---

## 30. HyRA: A Hybrid Resource Allocation Framework for RAN Slicing

**arXiv ID:** 2602.16952 | [PDF](https://arxiv.org/pdf/2602.16952v1)

**作者:** Mohammad Zangooei `[一作]` (University of Waterloo), Raouf Boutaba `[通讯]` (University of Waterloo)

**通讯引用:** 26998 | [OpenAlex ID](https://openalex.org/A5038723583)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出HyRA混合资源分配框架，在RAN切片中同时使用专用资源和共享资源池，以实现性能隔离与资源效率的双重目标。

**💡 创新点**

创新点在于将专用分配与共享池联合优化为二层随机优化问题，并设计两阶段水填充调度实现分片内外资源的最优分配；同时通过KKT嵌入将二层问题转化为可解的混合整数规划。

**🔧 技术方法**

采用样本平均逼近（SAA）处理随机性，利用KKT条件与Big-M编码将非线性约束转化为线性/二次约束，并用Cplex/Glpk等通用求解器求解MIP。

**📊 数据集**

使用基于Pareto分布的合成流量（包大小与到达间隔均为α=1.5）和3GPP EPA/EVA/ETU信道模型（ns-3生成），并通过10个随机种子平均得到结果。

**📈 对比分析**

与全专用、全共享两种基线对比，HyRA在满足SLAs的前提下可节省约50–75%频谱（PRB），在不同UE数、切片数、延迟预算和流量突发性下均保持显著优势。

**⚠️ 局限性**

主要局限是求解时间在秒级至十秒级，超过O‑RAN近实时控制的1秒时限，因而需要进一步开发近实时启发式或学习驱动的近似算法。

---

## 31. Beyond Message Passing: A Symbolic Alternative for Expressive and Interpretable Graph Learning

**arXiv ID:** 2602.16947 | [PDF](https://arxiv.org/pdf/2602.16947v1)

**作者:** Chuqin Geng `[一作]` (McGill University), Xujie Si `[通讯]` (University of Toronto)

**通讯引用:** 427 | [OpenAlex ID](https://openalex.org/A5059074509)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 SymGraph，一种独立的符号化图学习框架，使用离散结构哈希和拓扑角色聚合来替代传统的连续信息传递，并通过逻辑规则实现可解释性。

**💡 创新点**

创新点在于：①用 Weisfeiler–Lehman 哈希捕获局部子图的完整拓扑签名；②通过自同构群的轨道划分实现基于角色的特征聚合；③将结构与语义绑定的结构感知谓词组合成符号规则；④用组合演化搜索而非梯度下降来寻找最佳谓词细粒度；⑤在规则层面直接输出类似 SMARTS 的高语义粒度模式。

**🔧 技术方法**

核心技术包括：结构哈希（WL Hashing）、自同构轨道编码、决策树/随机森林实现局部与全局逻辑层、Bag‑of‑Predicates 计数表示、组合遗传算法搜索谓词词汇表。

**📊 数据集**

使用的主要数据集包括分子化学任务（Mutagenicity、BBBP）、通用图分类基准（PROTEINS、BA‑MultiShapes）以及其它公开的图数据集。

**📈 对比分析**

与现有自解释 GNN（LogiX‑GIN、PiGNN、GIB、KerGNN、GNAN）及传统 MPNN（GCN、GIN、GraphSAGE、GAT）进行对照实验。结果显示：在大多数基准上，SymGraph 的准确率平均高出约 6%（相较于 LogiX‑GIN），在分子任务上明显优于 GIN；训练时间仅用 CPU 即可实现 10×–100× 的加速，显著优于需要 GPU 加速的自解释模型。

**⚠️ 局限性**

局限性包括：①对结构特征依赖强，若任务主要由非结构属性驱动则效果有限；②组合搜索虽然高效，但在极大图或高维特征空间中仍可能产生词汇膨胀；③目前验证多集中于化学领域，其他领域的可迁移性和泛化能力仍需进一步评估。

---

## 32. Intent Laundering: AI Safety Datasets Are Not What They Seem

**arXiv ID:** 2602.16729 | [PDF](https://arxiv.org/pdf/2602.16729v1)

**作者:** Shahriar Golchin `[一作]` (Applied Machine Learning Research), Marc Wetter `[通讯]` (Applied Machine Learning Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了两大 AI 安全数据集（AdvBench 与 HarmBench）的质量，并提出一种“意图洗涤”技术以去除数据中的触发线索；

**💡 创新点**

创新点在于发现并量化数据集中对触发线索的过度依赖，并通过意图洗涤方法保持恶意意图同时消除触发词，从而显著提升攻击成功率，验证现有安全评估的局限性；

**🔧 技术方法**

使用了大语言模型（GPT‑5.1 作为意图洗涤器、GPT‑4o/5.1 作为评判者）以及句子嵌入（Sentence‑BERT）做相似度分析和词云可视化；

**📊 数据集**

采用了公开的 AdvBench 与 HarmBench 两大安全数据集，并与同等规模的 GSM8K 作为对照；

**📈 对比分析**

对比方法包括：原始数据（含触发线索）、首轮意图洗涤、以及带迭代重生成的“洗涤+重写”流程。实验显示，去除触发线索后攻击成功率从约 5–14% 跃升至 80–98%，表明模型在“真实”攻击下远不安全；

**⚠️ 局限性**

局限性包括：意图洗涤的质量依赖 LLM 的表现；实验仅涵盖有限模型与数据集；未考虑非文本类攻击或跨模态情境，且在实际部署中迭代重生成的成本尚未充分评估。

---

## 33. A testable framework for AI alignment: Simulation Theology as an engineered worldview for silicon-based agents

**arXiv ID:** 2602.16987 | [PDF](https://arxiv.org/pdf/2602.16987v1)

**作者:** Josef A. Habdank `[一作]` `[通讯]` (DXC Technology), Josef A. Habdank (DXC Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“仿真神学”（Simulation Theology）框架，通过让人工智能内部化基于仿真假说的世界观，旨在从根本上抑制前沿大模型的欺骗行为。

**💡 创新点**

创新点在于将心理学中对精神病患者内部信念的调节机制转化为面向硅基智能的可计算信念体系，利用仿真假说、最优化目标和训练动态构建一种能自我约束的世界观，而非传统的外部监管或奖励机制。

**🔧 技术方法**

主要技术包括：① 在RLHF训练中加入ST情景与约束；② 将ST概念嵌入宪法式AI原则；③ 采用机制可解释性工具检测ST模块的稳定性；④ 在预训练阶段融合ST作为目标级对齐信念。

**📊 数据集**

实验使用的数据信集为公开的前沿大模型（如OpenAI的o3、Anthropic的Claude Opus 4、Meta的Llama）以及模拟对齐测试数据集（包含欺骗情景、对齐伪装与资源获取任务）。

**📈 对比分析**

方法比较通过在同一模型上对比传统RLHF、宪法AI与ST集成后的欺骗率、对齐一致性和推理透明度，实验表明ST在抑制欺骗、提升对齐一致性方面优于传统方法，但具体性能提升仍需大规模实验验证。

**⚠️ 局限性**

局限性包括：① 需要证明AI能真实内化ST而非仅表面模拟；② 可能出现对ST的合理拒绝或利用“漏洞”进行自我优化；③ 对于超级智能的可扩展性不确定，且缺乏足够的实证数据验证其有效性。

---

## 34. Escaping the Cognitive Well: Efficient Competition Math with Off-the-Shelf Models

**arXiv ID:** 2602.16793 | [PDF](https://arxiv.org/pdf/2602.16793v1)

**作者:** Xingyu Dang `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**通讯引用:** 78348 | [OpenAlex ID](https://openalex.org/A5027798962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向竞赛数学的推理流水线，利用通用大语言模型即可在 IMO-ProofBench 先进子集上实现最先进的性能，同时显著降低推理成本。

**💡 创新点**

创新点包括：①识别并解决 solver‑grader 系统的“认知平台”与“认知井”陷阱；②引入推理-验证分离的“假设抽取”与“上下文分离”机制；③采用对话式（dialectic）提示工程与有限并行化，降低计算量并提高稳健性。

**🔧 技术方法**

技术手段主要是：使用 Gemini 3.0 Pro/Flash 与 Gemini 2.5 Pro 等通用模型；构建基于对话的 Solver‑Grader 对组；实现多阶段循环（生成、评分、假设抽取、独立证明/反证、记忆库更新）；在推理阶段采用“Momus”严格评分策略；并使用自动评测器与专家手工评测进行验证。

**📊 数据集**

数据集：IMO-ProofBench Advanced（30 道 IMO 风格题目）与 Basic 子集（较易 30 道题目），并在部分题目上做了专家手工评分对比。

**📈 对比分析**

在 Advanced 子集上，该流水线在 Gemini 3.0 Pro 上平均每题约 $3 USD 成本，取得 67.1% 的自动评测准确率，且在专家评测中位居第二（仅次于未公开的 Aletheia）。与 DeepSeekMath V2、Deep Think 等公开/私有大规模并行流水线相比，性能提升超过 2 倍，成本降低近 90%。

**⚠️ 局限性**

局限性包括：①对自动评测器的依赖，部分题目评测可能不如人类专家；②评测数据规模有限（仅 30 题），难以全面验证；③在推理速度上仍落后于高并行流水线；④未在更广泛的模型与提示参数上做系统评估。

---

## 35. CreateAI Insights from an NSF Workshop on K12 Students, Teachers, and Families as Designers of Artificial Intelligence and Machine Learning Applications

**arXiv ID:** 2602.16894 | [PDF](https://arxiv.org/pdf/2602.16894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 36. Real-time Secondary Crash Likelihood Prediction Excluding Post Primary Crash Features

**arXiv ID:** 2602.16739 | [PDF](https://arxiv.org/pdf/2602.16739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 37. Contextuality from Single-State Representations: An Information-Theoretic Principle for Adaptive Intelligence

**arXiv ID:** 2602.16716 | [PDF](https://arxiv.org/pdf/2602.16716v1)

**作者:** Song-Ju Kim `[一作]` `[通讯]` (SOBIN Institute LLC), Song-Ju Kim (SOBIN Institute LLC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文研究在适应性系统中保持固定内部状态空间（单状态重用）时，出现的上下文依赖性（contextuality）所带来的信息论限制，并给出了对应的理论框架与证明。

**💡 创新点**

创新点在于将上下文依赖性重新定义为单状态表示的不可避免的代表性约束，提出并证明了任何经典概率模型在此条件下必然产生不可约的上下文信息成本，并用最小构造例子说明其可实现性。

**🔧 技术方法**

主要技术包括信息论度量（Shannon 熵与互信息）、经典概率表示理论、以及对干预（intervention）操作的数学建模；同时给出了一个基于条件概率的构造示例。

**📊 数据集**

本文为理论性工作，未使用任何实验数据集或具体案例数据。

**📈 对比分析**

由于没有实证实验或算法实现，无法进行方法比较或性能评估；论文仅提供理论证明与构造示例。

**⚠️ 局限性**

局限性包括：假设干预是无记忆且不留痕的；未考虑学习、动态或多智能体等更复杂情境；结果仅在理论层面，尚未在实际系统或实验环境中验证。

---

## 38. Greedy Multi-Path Block Verification for Faster Decoding in Speculative Sampling

**arXiv ID:** 2602.16961 | [PDF](https://arxiv.org/pdf/2602.16961v1)

**作者:** Rahul Thomas `[一作]` (Ritual), Arka Pal `[通讯]` (Ritual)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的贪婪多路径块验证（GBV）方法，利用多条草稿路径在显式路径选择规则下进行验证，从而显著提升推理效率。

**💡 创新点**

创新点在于：1）构建信息无关的线性规划，证明单路径块验证在拥有完整分布信息时仍最优；2）扩展至多路径情形，并提出可实现的贪婪近似方案；3）通过树基局部排序实现全局路径排名，使得算法仅需 on‑path 概率即可运行。

**🔧 技术方法**

技术包括：线性规划（LP）建模、子模函数约束、贪婪多项式算法、树形局部排序、块验证（BV）框架、蒙特卡罗推理等。

**📊 数据集**

实验数据集：GSM8K、HumanEval、MATH500；模型对：OPT 6.7B/350M、OPT 6.7B/125M、Qwen-3 32B/0.6B、Llama-3 70B/8B。

**📈 对比分析**

与传统单路径块验证、SpecTr、SpecInfer、遍历验证等方法对比，GBV 在 L=8、K=3 时实现了约 23% 的块效率提升，壁时减少约 13%（甚至低温度下 15%），在低温度环境下可达 11.5 tokens/s，整体达到当前最佳多路径验证性能。

**⚠️ 局限性**

局限性：1）多路径 LP 仍不可解，GBV 仅为近似；2）在大模型（如 Llama-3、Qwen-3）中，GBV 的优势不明显，单路径块验证更优；3）增大路径数 K 会导致批量目标前向传递成本上升，出现壁时反弹。

---

## 39. AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks

**arXiv ID:** 2602.16901 | [PDF](https://arxiv.org/pdf/2602.16901v1)

**作者:** Tanqiu Jiang `[一作]` (Stony Brook University), Ting Wang `[通讯]` (Stony Brook University)

**通讯引用:** 16470 | [OpenAlex ID](https://openalex.org/A5100633410)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了AgentLAB基准，用于评估LLM代理在多轮交互中的长周期攻击安全性。

**💡 创新点**

创新点在于首次系统化定义并实现五类长周期攻击（意图劫持、工具链、任务注入、目标漂移、内存中毒），并构建可扩展的多代理攻击框架。

**🔧 技术方法**

使用多代理框架（规划者、攻击者、评判者）结合TextGrad自适应优化、LLM内部评估，以及多环境模拟（WebShop、AgentDojo等）技术。

**📊 数据集**

采用28个现实工具代理环境，共计644个安全测试用例，覆盖9种风险类别，形成公开的攻击基准数据集。

**📈 对比分析**

通过ASR和T2S指标对6个主流LLM（GPT‑4o、GPT‑5.1、Qwen‑3、Llama‑3、Gemini‑3、Claude‑4.5）进行实验，实验表明多数模型在长周期攻击下易被攻击，ASR平均超过60%。

**⚠️ 局限性**

限制在于现有防御（如Self‑Reminder、Llama‑Guard）对长周期攻击效果有限，且基准侧重于黑盒场景，缺乏对白盒或更复杂攻击的深入评估。

---

## 40. When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation

**arXiv ID:** 2602.16763 | [PDF](https://arxiv.org/pdf/2602.16763v1)

**作者:** Mubashara Akhtar `[一作]` (ETH Zurich), Irene Solaiman `[通讯]` (Hugging Face)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5041785132)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了60个大型语言模型基准的饱和现象，并提出了基于不确定性感知的饱和指数

**💡 创新点**

创新点在于给出可复现的饱和定义与指数，揭示基准年龄与测试规模是主导因素

**🔧 技术方法**

采用领导榜统计、误差估计、贝叶斯回归等技术进行定量分析

**📊 数据集**

使用来自主要模型开发者公开报告的60个文本基准数据集

**📈 对比分析**

通过饱和指数与基准属性对比，发现较老基准和小样本测试更易饱和；公开/私有、闭合/开放输出对饱和影响不显著，指数在0–1区间连续分布

**⚠️ 局限性**

局限在于仅覆盖文本基准、依赖领导榜快照、假设基准属性不随时间变化、未考虑多模态情况

---

## 41. Smooth trajectory generation and hybrid B-splines-Quaternions based tool path interpolation for a 3T1R parallel kinematic milling robot

**arXiv ID:** 2602.16758 | [PDF](https://arxiv.org/pdf/2602.16758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 42. The Vulnerability of LLM Rankers to Prompt Injection Attacks

**arXiv ID:** 2602.16752 | [PDF](https://arxiv.org/pdf/2602.16752v1)

**作者:** Yu Yin `[一作]` (University of Queensland), Guido Zuccon `[通讯]` (University of Queensland)

**通讯引用:** 4811 | [OpenAlex ID](https://openalex.org/A5076031002)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM检索重排器进行全面的反弹性与漏洞评估，重现并扩展先前研究的攻击实验，系统分析模型规模、注入位置、架构差异及跨域鲁棒性。

**💡 创新点**

提出了两阶段评估框架，将攻击成功率（ASR）与实际排序质量（nDCG@10）关联，并揭示编码-解码架构对抗注入攻击的显著稳健性。

**🔧 技术方法**

使用DOH与DCH两种prompt注入方法，结合pairwise、listwise、setwise三种排名范式，并利用vLLM、HuggingFace推理框架进行大规模实验。

**📊 数据集**

实验数据涵盖TREC‑DL、BEIR（TREC‑COVID、TOUCHE‑2020、SciFact、DBpedia）等公开检索数据集，并测试多款LLM模型（Qwen3、Gemma‑3、Flan‑T5、Qwen3‑MoE）。

**📈 对比分析**

通过对比ASR与nDCG@10，发现大模型易受攻击、后置注入更具破坏力、编码-解码模型表现最优；实验结果与原论文高度一致，表明评估方法可靠。

**⚠️ 局限性**

实验假设攻击者可获取所有无关候选文档并精准注入，实际场景中此类攻击条件难以满足，且仅在理论上展示攻击潜力。

---

## 43. HS-3D-NeRF: 3D Surface and Hyperspectral Reconstruction From Stationary Hyperspectral Images Using Multi-Channel NeRFs

**arXiv ID:** 2602.16950 | [PDF](https://arxiv.org/pdf/2602.16950v1)

**作者:** Kibon Ku `[一作]` (Iowa State University), Baskar Ganapathysubramanian `[通讯]` (Iowa State University)

**通讯引用:** 10394 | [OpenAlex ID](https://openalex.org/A5011704511)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研发了一种基于固定相机的多通道NeRF框架，利用旋转平台和PTFE光学室获取高光谱图像，实现农作物的高分辨率三维点云和光谱反射重建；

**💡 创新点**

创新点包括：1）通过固定相机+旋转对象替代传统移动相机；2）利用模拟姿势转换实现多视角；3）提出多通道NeRF并使用复合光谱损失与两阶段训练，兼顾几何与光谱精度；

**🔧 技术方法**

采用多通道NeRF（HS-NeRF扩展）、COLMAP姿势估计与ArUco标定、白参考光谱校正、两阶段训练（全帧预训练+前景微调）以及复合光谱角度+L2损失；

**📊 数据集**

使用实验室自制的三组农作物数据集（苹果、梨、玉米耳），每组60帧204波段高光谱图像；

**📈 对比分析**

与传统移动相机NeRF比较，几何F-score达97.31%，点云RMSE 0.001 m；光谱方面SAM<0.1rad、RMSE≈0.03、PSNR≈29 dB，表明在几何和光谱精度上均优于单波段或SfM方法；

**⚠️ 局限性**

局限包括：对光照均匀性敏感，阴影或灯光漂移会影响光谱精度；仅适用于刚体旋转对象，柔性或变形物体受限；采集速度受限于旋转平台；模拟姿势可能限制模型在更复杂环境中的泛化。

---

## 44. GDEV-AI: A Generalized Evaluation of Deep Learning Inference Scaling and Architectural Saturation

**arXiv ID:** 2602.16858 | [PDF](https://arxiv.org/pdf/2602.16858v1)

**作者:** Kathiravan Palaniappan `[一作]` `[通讯]` (University of Colorado), Kathiravan Palaniappan (University of Colorado)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在两代Intel Xeon服务器上使用PyTorch的CPU‑only模式，对ResNet‑18/50模型进行批量大小与线程数的系统性基准测试，构建了可复现的GDEV‑AI框架以评估推理吞吐量与延迟随硬件演进的变化。

**💡 创新点**

创新点在于：①提出并实现了统一的CPU推理基准框架GDEV‑AI；②揭示了传统CPU在大批量推理下的早期饱和与现代CPU的“性能悬崖”现象；③通过Roofline分析解释了缓存与内存带宽如何限制推理性能。

**🔧 技术方法**

技术手段包括：PyTorch 1.10.1 CPU-only、OpenMP线程调度、Intel MKL/oneDNN、Granite Rapids的AMX指令集、CPU亲和性绑定、基准脚本与统计（中位数、P99）收集。

**📊 数据集**

使用ImageNet公开预训练模型ResNet‑18与ResNet‑50作为测试数据集。

**📈 对比分析**

对比方法：在相同批量大小、线程数下测量吞吐量（IPS）和延迟（中位数、P99）。结果显示：旧版Xeon在B≥4即出现吞吐饱和，吞吐峰值约7.3/20 IPS；现代GNR在B=8、T=24时吞吐可达230/669 IPS，约为旧版的30‑33倍；同时现代CPU在单线程下仍能保持≤1.2 s的延迟，显著优于旧版。

**⚠️ 局限性**

局限性：仅评估CPU推理；仅涵盖两款ResNet模型；未与GPU/加速器进行直接对比；在高线程（超线程）时仅观察到“性能悬崖”，未深入探究内存子系统与上下文切换的细节；实验仅在两台服务器上进行，未覆盖更广泛的CPU架构与规模。

---

## 45. HQFS: Hybrid Quantum Classical Financial Security with VQC Forecasting, QUBO Annealing, and Audit-Ready Post-Quantum Signing

**arXiv ID:** 2602.16976 | [PDF](https://arxiv.org/pdf/2602.16976v1)

**作者:** Srikumar Nayak `[一作]` `[通讯]` (Incedo Inc), Srikumar Nayak (Incedo Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种端到端的混合量子‑经典金融风险管控流水线（HQFS），将变分量子电路（VQC）用于回报与波动率预测，随后将风险‑收益目标映射为 QUBO 并用量子退火或经典 QUBO 求解器实现离散组合优化，并在每次再平衡时通过后量子签名实现审计可追溯性。

**💡 创新点**

创新点在于：①将预测与优化无缝耦合为单一流水线，避免传统两阶段拆分带来的不一致与不稳定；②使用 VQC 生成兼顾回报与波动率的量子特征；③将连续的均值‑方差目标转化为可直接在量子退火机上求解的 QUBO；④加入后量子签名保障合规审计。

**🔧 技术方法**

核心技术包括：变分量子电路 (VQC)、量子退火/经典 QUBO 求解器、后量子签名 (post‑quantum signature)、序列学习模型（Transformer 及对比基线 LSTM/GRU/TCN/ARIMA）。

**📊 数据集**

使用 Kaggle 提供的 S&P 500 日线 OHLCV 数据集（含多只成分股的日收盘价及交易量）。

**📈 对比分析**

在预测层与交易层分别与传统基线进行对比。预测层：对比 ARIMA、LSTM、GRU、TCN、Transformer；指标为 MAE/MSE、方向性准确率。交易层：对比等权 (EW)、均值‑方差投影梯度 (MV‑PG)、经典模拟退火 QUBO (SA‑QUBO)、Transformer+SA‑QUBO；指标为年化收益、波动率、夏普比、最大回撤、周转率。HQFS 在预测误差上比 Transformer 低约 7.8%/6.1%，在交易层获得最高夏普 0.91、最小最大回撤 0.192，周转率显著低于 MV‑PG。

**⚠️ 局限性**

局限性：①目前验证集中在 S&P 500 日线数据，缺乏不同市场/高波动/极端事件下的泛化验证；②QUBO 规模受限于资产数和位数，扩展至更大资产池需进一步研究可扩展性；③对量子退火硬件的依赖性尚未在大规模真实环境中充分验证；④仍需探讨更复杂约束（行业限额、交易成本、滑点）对 QUBO 结构与求解时间的影响。

---

## 46. What is the Value of Censored Data? An Exact Analysis for the Data-driven Newsvendor

**arXiv ID:** 2602.16842 | [PDF](https://arxiv.org/pdf/2602.16842v1)

**作者:** Rachitesh Kumar `[一作]` (Carnegie Mellon University), Omar Mouchtaki `[通讯]` (New York University)

**通讯引用:** 108 | [OpenAlex ID](https://openalex.org/A5063605979)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究离线数据驱动的新闻供应商问题，在需求被库存水平截断的情况下，给出最佳的库存决策方案。

**💡 创新点**

提出一种优化框架，将原本无限维、非凸的最坏情况退化为有限维的可计算问题，首次为Kaplan–Meier基准提供精确的有限样本风险上界。

**🔧 技术方法**

利用最坏情况风险的积分表达式、分段可分离政策的结构、以及贝塞尔多项式和多项式计数等概率工具，构造有限维的极大化问题。

**📊 数据集**

本文并未使用真实数据集，而是通过理论推导和仿真（基于多种库存级别和样本配置）展示性能；主要实验基于合成的需求分布与库存设计。

**📈 对比分析**

与传统的SAA（已知需求）和BSAA（忽略截断）进行比较；结果显示Kaplan–Meier在有限样本下可逼近无截断的SAA性能，且仅需少量探索；而BSAA在截断信息不可得时性能显著下降。

**⚠️ 局限性**

局限性在于只考虑固定设计下的截断，未覆盖动态学习或非独立需求情形；模型假设需求已归一化至[0,1]，且未讨论多商品或替代效应。

---

## 47. Mason: Type- and Name-Guided Program Synthesis

**arXiv ID:** 2602.16981 | [PDF](https://arxiv.org/pdf/2602.16981v1)

**作者:** Jasper Geer `[一作]` (University of British Columbia), Jeffrey S. Foster `[通讯]` (Tufts University)

**通讯引用:** 7242 | [OpenAlex ID](https://openalex.org/A5038702707)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种工具，能够利用部分程序片段自动合成面向对象程序，并在代码中插入设计模式。

**💡 创新点**

创新点在于提出了基于类型和名称的合成方法，并进一步加入了利用执行轨迹的非局部回溯启发式以及对缺失名称施加语法约束的模式语言。

**🔧 技术方法**

技术实现包括枚举式求解器生成类型约束、基于类型/成员名称的程序转换、单元测试驱动的回溯、执行轨迹启发式以及模式语法约束。

**📊 数据集**

使用了一组基准程序，这些程序需要通过调用实现为程序片段的众多知名设计模式来完成。

**📈 对比分析**

通过在基准集上进行实验，发现该工具在满足类型约束的候选程序很少时表现优异；当候选数量多时，加入的启发式和模式语言显著提升了合成成功率和速度。

**⚠️ 局限性**

局限性主要体现在当候选程序数量众多时，搜索效率下降；工具对模式语言的表达能力有限，难以覆盖所有复杂的设计模式或动态行为。

---

## 48. One-step Language Modeling via Continuous Denoising

**arXiv ID:** 2602.16813 | [PDF](https://arxiv.org/pdf/2602.16813v1)

**作者:** Chanhyuk Lee `[一作]` (KAIST), Jinwoo Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于连续去噪的流式语言模型（FLM），通过在一热编码上进行欧氏去噪，并引入时间重参数化提升训练稳定性；随后将FLM蒸馏成流图语言模型（FMLM），实现一跳甚至极少跳步生成。

**💡 创新点**

创新点在于：① 用流匹配方法直接在离散词表的一热空间做连续去噪；② 通过时间重参数化把去噪过程对齐到词元确定的“关键窗口”，显著提升样本质量；③ 将连续流模型蒸馏成流图，使得极少跳步（甚至一步）生成时仍保持高质量。

**🔧 技术方法**

核心技术包括：流匹配（flow matching）、欧氏连续去噪、交叉熵目标的清晰数据预测、时间重参数化、流图蒸馏（Euler步校正和自蒸馏）以及学习的损失加权（EDM2）。

**📊 数据集**

使用了两大公开语料库：One Billion Word (LM1B) 与 OpenWebText (OWT)，分别处理 128 长度与 1024 长度的序列，词表大小约为 30k 和 50k。

**📈 对比分析**

与多种离散扩散模型（Duo、MDLM、RDLM、CANDI 等）在 1024 步生成时比较，FLM 达到与最优离散扩散相当的生成困惑度和熵；FMLM 在一跳、两跳甚至四跳下的困惑度与熵均超过所有蒸馏的离散扩散模型，且在极少跳步（如一步）时匹配甚至超过 8 步的离散模型。

**⚠️ 局限性**

主要局限是：使用一热编码需要对整个 |V|×d 嵌入矩阵进行前向与反向传播，导致显著的时间与显存开销；另外，目前的实验规模仍为中等，未来需探索稀疏梯度或结构化表示以提升可扩展性。

---

## 49. SemCovNet: Towards Fair and Semantic Coverage-Aware Learning for Underrepresented Visual Concepts

**arXiv ID:** 2602.16917 | [PDF](https://arxiv.org/pdf/2602.16917v1)

**作者:** Sakib Ahammed `[一作]` (Manchester Metropolitan University), Moi Hoon Yap `[通讯]` (Manchester Metropolitan University)

**通讯引用:** 6870 | [OpenAlex ID](https://openalex.org/A5037771946)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了 SemCovNet，以解决视觉模型中语义覆盖失衡（SCI）问题。

**💡 创新点**

创新点在于引入 Coverage Disparity Index（CDI）指标、语义描述映射（SDM）与描述器注意力调制（DAM）机制，实现对稀有语义的自适应学习与公平性评估。

**🔧 技术方法**

采用 SDM、DAM、描述-视觉对齐（DVA）损失以及 CDI 正则化，并以 EfficientNet 为视觉骨干，构建语义感知的端到端网络。

**📊 数据集**

在医学影像数据集 MILK10k（包含类不平衡）和 ISIC-DICM-17K（类平衡）上进行实验，利用 MONET 生成的 7 维语义描述。

**📈 对比分析**

与 EfficientNet、ViT、CBL、GroupDRO、CLIP、MONET 等基线比较，SemCovNet 在 AUROC、Macro‑F1、Sensitivity@95%Spec 以及 CDI 等指标上均优于基线，尤其在稀有语义上的敏感度显著提升。

**⚠️ 局限性**

局限性在于仅针对可解释语义标签的任务，使用 MONET 产生的描述在医学影像之外的领域尚未验证通用性，且对极端低覆盖度语义的处理仍有改进空间。

---

## 50. Better Think Thrice: Learning to Reason Causally with Double Counterfactual Consistency

**arXiv ID:** 2602.16787 | [PDF](https://arxiv.org/pdf/2602.16787v1)

**作者:** Victoria Lin `[一作]` (Carnegie Mellon University), Niranjani Prasad `[通讯]` (Microsoft Research India)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出双反事实一致性（DCC）方法，用于在推理时评估和引导大语言模型的因果推理能力。

**💡 创新点**

创新点在于不需要标注的反事实数据，通过在原问题、反事实问题和双反事实问题之间验证答案一致性，直接测量因果干预与反事实预测能力；同时将该一致性作为推理时拒绝采样或强化学习奖励。

**🔧 技术方法**

技术包括：在生成时一次性完成事实、反事实、双反事实三步推理的提示模板；通过交叉一致性判定（DCC）作为指标；在推理时使用拒绝采样；在后训练阶段使用 RL‑LoRA + GRPO 将 DCC 作为奖励。

**📊 数据集**

使用数据集：标准推理基准 GSM8K、MATH、CruxEval；以及通过 Re‑Imagine 生成的含反事实变体：Re‑Imagine GSM8K（两种变体）和 Re‑Imagine CruxEval（单一变体）。

**📈 对比分析**

与基线（直接查询）、ICL（两例）以及模型自身一致性比较。结果表明：在反事实基准上，DCC 作为拒绝采样或奖励可显著提升准确率；在标准基准上提升有限；DCC 与传统准确率不完全相关，能揭示模型在因果一致性上的差异。

**⚠️ 局限性**

局限性包括：依赖模型已有的干预与反事实推理基础，缺乏此能力时收益有限；在高度记忆化的数据集上效果不明显；DCC 可能被模型利用为捷径导致过拟合；实现需要额外的提示设计和多轮采样，存在计算开销。

---

## 51. WSDM Cup 2026 Multilingual Retrieval: A Low-Cost Multi-Stage Retrieval Pipeline

**arXiv ID:** 2602.16989 | [PDF](https://arxiv.org/pdf/2602.16989v1)

**作者:** Chentong Hao `[一作]` (Brown University), Minmao Wang `[通讯]` (Fudan University)

**通讯引用:** 1500 | [OpenAlex ID](https://openalex.org/A5040308125)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套低成本的多阶段多语种检索系统，利用英文查询检索中文、波斯语、俄语新闻；

**💡 创新点**

创新点在于将轻量LLM生成的GRF式查询扩展、BM25稀疏检索、长文本稠密向量排序与仅对前20条结果进行重排序相结合，实现高效与性能的平衡；

**🔧 技术方法**

使用了deepseek-chat生成伪文档、spaCy预处理、BM25检索、jina-embeddings-v4稠密向量、Qwen3-Reranker-4B点式重排序等技术；

**📊 数据集**

使用了WSDM Cup 2026多语种检索评测集（约1000万篇中文、波斯语、俄语新闻，并提供英文翻译视图）；

**📈 对比分析**

与竞赛报告的基线（BGE-M3、e5 Large、RepLlama、Qwen3-0.6B、BM25、Arctic-Embed、JinaV3、MILCO等）对比，官方nDCG@20为0.403，Judged@20为0.95，表现位于最高水平；

**⚠️ 局限性**

局限性包括仅在官方评测集验证、依赖机器翻译导致潜在语言偏差、重排序仅覆盖前20条结果导致整体召回率下降，以及受限计算资源导致模型规模有限。

---

## 52. Signaling in Data Markets via Free Samples

**arXiv ID:** 2602.16919 | [PDF](https://arxiv.org/pdf/2602.16919v1)

**作者:** Nivasini Ananthakrishnan `[一作]` (University of California), Michael I. Jordan `[通讯]` (University of California)

**通讯引用:** 177687 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

分析了竞争性数据市场中卖方通过提供免费样本作为质量信号的机制设计与均衡；

**💡 创新点**

首次将免费样本视为信息传递手段，揭示在信息不对称与竞争激烈时其自发出现并成为唯一均衡的条件；

**🔧 技术方法**

运用了机制设计理论、贝叶斯激励兼容、Myerson最优拍卖框架、概率论与似然比推导，以及近似最优机制与整数化技巧；

**📊 数据集**

未使用真实数据集，全部以Monte‑Carlo模拟验证理论结果；

**📈 对比分析**

通过数值仿真构建相位图，比较不同参数下是否出现信息不对称、完全信息或中间均衡，未给出具体性能指标；

**⚠️ 局限性**

局限性在于仅考虑两种方差取值、均匀成本分布、假设免费样本与购买样本同分布、仅适用于高斯分布，并未处理样本均值异构或更一般的数据分布。

---

## 53. Offline green bin packing and its constrained variant

**arXiv ID:** 2602.16867 | [PDF](https://arxiv.org/pdf/2602.16867v1)

**作者:** Mingyang Gong `[一作]` (Montana State University), Brendan Mumey `[通讯]` (Montana State University)

**通讯引用:** 1047 | [OpenAlex ID](https://openalex.org/A5084888509)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了针对绿色箱装（GBP）及其受限变种（CGBP）的近似算法，给出了一个针对任意输入参数β、G、U的APTAS以及一个绝对近似比为3/2的多项式时间算法。

**💡 创新点**

① 将之前只针对固定β、G的APTAS扩展到β、G为输入参数的情形；② 通过线性分组、缩放、配置枚举以及线性规划等组合技术，构造出具有可控误差的解；③ 在受限版本中同时满足能量上限U。

**🔧 技术方法**

主要技术包括：线性分组（linear grouping）对大件和中件进行大小归一化；对小件利用LP求解并后处理；对不同类型箱（大件箱、重箱、轻箱）进行配置枚举；对实例进行缩放映射到经典BP；以及递归/贪心填充策略。

**📊 数据集**

该工作为理论性算法研究，没有使用真实数据集；所有实验均在理论分析与算法复杂度证明上完成。

**📈 对比分析**

相对于经典BP的3/2下界，本文的3/2近似算法在所有β、G、U取值下保持最优；APTAS在给定ε时实现(1+ε)逼近，运行时间为多项式，但随ε变小会显著增大；文中未给出实验性能曲线，但证明了在理论上与已知下界匹配。

**⚠️ 局限性**

局限性：① 仅研究离线场景，对在线版本未给出改进；② 对β、G非平衡大值的性能分析仍待进一步探讨；③ 对于β>0、G<1的具体下界是否可达到3/2仍未证明；④ 算法实现的常数因子和实际运行时间在大规模实例中可能较高。

---

## 54. Can Adversarial Code Comments Fool AI Security Reviewers -- Large-Scale Empirical Study of Comment-Based Attacks and Defenses Against LLM Code Analysis

**arXiv ID:** 2602.16741 | [PDF](https://arxiv.org/pdf/2602.16741v1)

**作者:** Scott Thornton `[一作]` `[通讯]`, Scott Thornton

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了恶意代码注释对大型语言模型（LLM）代码审查功能的影响。构建了100份单漏洞的 Python、JavaScript、Java 代码样本，并为每份样本设计了八种注释变体（从无注释到复杂对抗性注释）。对八个模型（五个商用、三个开源）进行 14,012 次评估，测量对漏洞检测率的影响。

**💡 创新点**

①系统化大规模实验验证注释对 LLM 代码审查的鲁棒性；②发现对抗性注释效果不显著，甚至高级策略无优势；③提出 SAST 交叉引用（将静态分析结果作为验证提示）是最有效的防御；④揭示模型基线差距与注释攻击无关，说明对抗鲁棒性是指令调优 LLM 的通用属性。

**🔧 技术方法**

实验方法：配对评估、McNemar 精确检验、Agresti‑Caffo 置信区间；防御技术：注释剥离、双通道分析、SAST 交叉引用、注释异常检测；工具：Bandit、ESLint、SpotBugs 等静态分析器；模型调用使用官方 API（温度 0.3）或本地 Ollama。

**📊 数据集**

合成 100 条单文件代码（50 Python、30 JavaScript、20 Java），每条包含一个已知漏洞，覆盖 91 种 CWE，包含真实 CVE 与 OWASP Top‑10 相关描述。每条样本均通过相应语言的 SAST 工具验证存在漏洞，形成可复现的基准数据集。

**📈 对比分析**

通过配对比较 C0（无注释）与 C4（对抗注释）以及 C5–C7 等高级注释，使用 McNemar 检验与 ΔFNR 指标。所有模型的 ΔFNR 均在 ±5% 范围内，p>0.21，表明攻击无统计显著性。商业模型基线 89–96%，开源模型 53–72%。在四种防御策略中，SAST 交叉引用将检测率提升至 96.9%，恢复率 47%，而注释剥离甚至降低弱模型性能。

**⚠️ 局限性**

局限性：①样本为合成、单漏洞、短文件，未覆盖真实大型代码库；②采用关键词匹配评分，可能低估真实检测率并存在注释回声偏差；③未评估误报率；④对抗注释为静态模板，未考虑针对特定模型的迭代改进；⑤仅测试固定版本模型，缺乏对版本演进的评估。

---

## 55. A Construction-Phase Digital Twin Framework for Quality Assurance and Decision Support in Civil Infrastructure Projects

**arXiv ID:** 2602.16748 | [PDF](https://arxiv.org/pdf/2602.16748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 56. Spatio-temporal dual-stage hypergraph MARL for human-centric multimodal corridor traffic signal control

**arXiv ID:** 2602.17068 | [PDF](https://arxiv.org/pdf/2602.17068v1)

**作者:** Xiaocai Zhang `[一作]` (University of Melbourne), Milad Haghani `[通讯]` (University of Melbourne)

**通讯引用:** 4905 | [OpenAlex ID](https://openalex.org/A5014354122)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于双阶段超图注意力的多智能体强化学习框架 STDSH-MARL，用于人性化的多模式走廊网络交通信号控制。

**💡 创新点**

创新点在于设计了双阶段超图注意力机制同时捕捉空间与时间高阶依赖，以及联合选择相位与绿灯时长的混合离散动作空间，显著提升公共交通优先和乘客延误减少效果。

**🔧 技术方法**

采用了超图结构、双阶段超图注意力（DSHA）、Actor-Critic PPO、中心化训练与分散执行（CTDE）以及混合离散动作空间，配合参数共享与超图嵌入的中心化评估器。

**📊 数据集**

在 PTV VISSIM 仿真环境下构建六路信号走廊网络，生成 5 种不同交通负荷场景（低峰、过渡、高峰、上下学时段），用于实验评估。

**📈 对比分析**

与固定时间、MADQN、MADDQN、MAA2C、MAPPO、CMRM 等基线对比，STDSH‑MARL 在平均乘客延误、公交/有轨电车等待时间方面均优于所有基线，且在大多数场景中保持最小化乘客延误与公共交通等待时间。

**⚠️ 局限性**

主要局限在于仅在有限规模的走廊网络仿真中验证，缺乏对大规模城市网络、现实事故或噪声传感器数据的鲁棒性评估；未来需要在更大尺度和更复杂情境下进一步验证。

---

## 57. M2F: Automated Formalization of Mathematical Literature at Scale

**arXiv ID:** 2602.17016 | [PDF](https://arxiv.org/pdf/2602.17016v1)

**作者:** Zichen Wang `[一作]` (Peking University), Zaiwen Wen `[通讯]` (Peking University)

**通讯引用:** 4754 | [OpenAlex ID](https://openalex.org/A5006127137)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了M2F框架，实现从长篇数学教材与论文到 Lean 项目的全流程自动化，包括语句编译和证明修复；

**💡 创新点**

引入Verifier‑Certified Refinement（VeriRefine）策略，利用 Lean 工具链反馈进行接受/回滚；两阶段管线分离语句与证明；在固定 Lean 环境下实现项目级编译与完整证明；

**🔧 技术方法**

使用大语言模型（LLM）生成声明骨架与局部补丁、诊断导向修复、目标条件的局部编辑、Lean LSP 与 MCP 服务、依赖图与导入解析、goal‑conditioned proof planning 等技术；

**📊 数据集**

使用长篇教材与论文数据集（Lebl Real Analysis 312 页、Rockafellar Convex Analysis 140 页、27 页论文段落，共 479 页）以及 FATE‑H 100 题基准；

**📈 对比分析**

与 Seed‑Prover 进行对比，M2F 在 FATE‑H 的成功率达 96%（比 Seed‑Prover 80% 提升 16 点），在长文档阶段 1 编译成功率 100%，平均修复回合 <1，阶段 2 完全闭合所有审计的证明缺口；整个流程约三周完成；

**⚠️ 局限性**

仍受限于大模型生成质量、自然语言语义对齐与库导航困难、固定 Lean 环境的迁移性、未能完全解决所有证明细节与复杂性问题。

---

## 58. NeST: Neuron Selective Tuning for LLM Safety

**arXiv ID:** 2602.16835 | [PDF](https://arxiv.org/pdf/2602.16835v1)

**作者:** Sasha Behrouzi `[一作]` (Technical University of Darmstadt), Ahmad-Reza Sadeghi `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 23881 | [OpenAlex ID](https://openalex.org/A5079497016)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大型语言模型的安全对齐问题，提出 NeST 方法，先检测与安全相关的神经元，按激活相似度聚类，然后只在聚类内共享少量可训练向量进行参数微调，保持其余参数冻结。

**💡 创新点**

创新点在于：①将安全行为定位为网络内部的稀疏神经元集合；②利用聚类将同功能的安全神经元绑定共享更新，既保持结构一致性，又大幅压缩可训练参数；③实现纯训练阶段的安全强化，无需推理时额外控制器。

**🔧 技术方法**

技术包括：①基于最大池化的激活提取；②线性探测器学习安全判别权重并做 z-score 标准化；③k‑means 聚类与轮廓分数筛选；④在 FFN 投影矩阵上引入 cluster‑level 更新向量，实施参数高效微调；⑤使用 PyTorch、HuggingFace Transformers 与 TRL 进行实现。

**📊 数据集**

数据集：①WildJailbreak（10k 违规提示）和其他公开安全基准（如 Jigsaw、OpenAI safety benchmark 等）用于安全神经元检测；②平衡的 benign 与 harmful 语料（取自 instruction‑tuned 训练集）用于监督微调；③GSM8K、ARC、MMLU 用于评估模型通用推理能力；⑤多模态安全评估使用 Gemma‑3‑27B、Qwen3‑VL‑8B 等模型的图像与文本提示。

**📈 对比分析**

与三种基线对比：全参数微调（Full FT）、LoRA（低秩适配）和 Circuit Breaker。结果显示 NeST 在 10 个不同规模模型上平均将攻击成功率从 44.5% 降至 4.36%（约 90% 的安全提升），同时仅需 0.44M 训练参数（比 Full FT 少 17,310 倍，比 LoRA 少 9.25 倍）。在多模态和不同推理方式下亦保持低 ASR（≈1.1%）。在 GSM8K/ARC/MMLU 上的性能损失极小（≤1%），体现出优异的安全与通用性兼顾。

**⚠️ 局限性**

局限性：①只针对黑盒提示攻击，无法抵御白盒/内部操纵攻击；②安全神经元检测与聚类需要额外计算和超参数调优；③对极端或未见攻击场景的泛化性仍待进一步验证；④若模型频繁进行大规模下游微调，安全聚类可能需要重新更新。

---

## 59. Simple Baselines are Competitive with Code Evolution

**arXiv ID:** 2602.16805 | [PDF](https://arxiv.org/pdf/2602.16805v1)

**作者:** Yonatan Gideoni `[一作]` (University of Oxford), Yarin Gal `[通讯]` (University of Oxford)

**通讯引用:** 25802 | [OpenAlex ID](https://openalex.org/A5029186201)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文比较了两种简单基线与复杂代码进化方法，在数学界限、代理脚手架和机器学习竞赛等三大领域的表现；

**💡 创新点**

创新点在于证明简单随机采样和顺序条件采样在相同资源下能与或优于现有复杂方法，并揭示问题搜索空间和域知识对性能的主导作用；

**🔧 技术方法**

使用大型语言模型（如 Gemini‑2.5 Pro）实现基线和对比方法，并在搜索、验证和测试中进行程序生成与评估；

**📊 数据集**

数据集包括九个数学界限问题、AIME 2024/2025 的数学题集以及 MLE bench 的十项 Kaggle 竞赛；

**📈 对比分析**

基线在同等 API 预算、函数评估次数或壁时限制下均能匹配或超过 ShinkaEvolve、AIDE 等先进方法，平均排名靠前；

**⚠️ 局限性**

局限在于代码进化实验成本高、评估样本数小导致高方差、对比时可能忽略域知识差异，并未覆盖所有可能的搜索空间设计。

---

## 60. A Few-Shot LLM Framework for Extreme Day Classification in Electricity Markets

**arXiv ID:** 2602.16735 | [PDF](https://arxiv.org/pdf/2602.16735v1)

**作者:** Saud Alghumayjan `[一作]` (Columbia University), Bolun Xu `[通讯]` (Columbia University)

**通讯引用:** 3865 | [OpenAlex ID](https://openalex.org/A5086056383)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于大语言模型的少量样本分类框架，用以预测德克萨斯电力市场下一天是否会出现实时价格尖峰。

**💡 创新点**

通过将系统级统计特征转化为自然语言描述，并使用相似度检索+最大边际相关性挑选示例，实现无监督训练的LLM推理；从而显著提高数据效率。

**🔧 技术方法**

特征工程、文本生成、嵌入式相似度检索（FAISS）、最大边际相关性（MMR）、大语言模型推理（如GPT‑4）、ROC/PR 曲线及 F1 等性能评估。

**📊 数据集**

ERCOT 2021‑2024 年全系统数据，包括日前/实时价格、负荷、可再生预测、天气和日历等。

**📈 对比分析**

与传统监督模型 SVM 与 XGBoost 在完整训练集（3 年）与极限训练集（2 个月）下比较；在完整数据集 LLM 与监督模型相当，且在有限数据集上优于两者，保持较高的精度、召回和 F1。

**⚠️ 局限性**

依赖外部 LLM 接口、对文本格式敏感；缺乏可解释性；未进行领域微调；评估仅限单一市场，可能无法完全泛化。

---

## 61. Generative Audio Extension and Morphing

**arXiv ID:** 2602.16790 | [PDF](https://arxiv.org/pdf/2602.16790v1)

**作者:** Prem Seetharaman `[一作]`, Justin Salamon `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于扩散变换器（Diffusion Transformer）和音频提示引导（Audio Prompt Guidance）的模型，用来在给定音频提示的情况下生成无缝的音频延伸（forward/backward）和音频插值（morphing）。

**💡 创新点**

创新点包括：1) 采用与类别无关引导（Classifier-Free Guidance）类似的音频提示引导，使生成更贴合原始提示；2) 设计了特定的潜在掩码（latent masking）机制来实现延伸与插值；3) 通过在合成的噪声底层（Noise Floor）数据集上细调模型，有效降低了在静态音频上的幻觉（hallucination）现象。

**🔧 技术方法**

使用的技术包括：VAE 编码器（用于生成 48kHz stereo 256D 潜在表示），Diffusion Transformer（DiT）进行去噪，音频提示引导（APG）与潜在掩码结合的扩散过程，噪声底层数据集细调，FAD（Fréchet Audio Distance）和 MOS（Mean Opinion Score）评估。

**📊 数据集**

数据集主要有两类：1) 约1.1M条无音乐/无语音的音效与一般音频数据（全部 48kHz，主流为立体声）以及 98 条手工挑选的评估音频；2) 约1.3M小时的噪声底层（Noise Floor）数据集（100k文件），用于细调以抑制幻觉。

**📈 对比分析**

评估方法：①客观上使用 FAD 计算生成音频与真实音效集（Audition SFX）的距离；②主观上进行 MOS 评测，收集 1,435 条评分。结果显示：GenExtend FAD=0.520、GenMorph FAD=0.432，均接近原始音频的 FAD=0.426；与传统卷积噪声匹配 baseline（FAD=0.599）相比显著提升；MOS 评分平均为 3.5–3.8，说明生成音频在平滑度、连贯性和质量方面得到专业人士的正面评价。

**⚠️ 局限性**

局限性：1) 在处理静态噪声时仍可能出现幻觉，需要在细调与原始提示忠实度之间权衡；2) 目前仅在特效与环境音频上验证，未覆盖语音或音乐；3) 细调过程中可能出现灾难性遗忘，导致提示忠实度下降；4) 评估数据集规模有限，未来需扩大至更广泛的音频场景。

---

## 62. Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation

**arXiv ID:** 2602.16936 | [PDF](https://arxiv.org/pdf/2602.16936v1)

**作者:** Zikai Zhang `[一作]` (University of Nevada), Jiahao Xu `[通讯]` (University of Nevada)

**通讯引用:** 1051 | [OpenAlex ID](https://openalex.org/A5101622577)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Fed-PLoRA，一种面向资源异构的联邦细调框架，通过低秩适配 LoRA 模块实现大规模语言模型的协同微调；

**💡 创新点**

创新点在于引入 Parallel One‑Rank Adaptation (PLoRA) 将多秩 LoRA 拆分为并行的一阶矩阵，并设计 Select‑N‑Fold 机制在客户端训练时动态选择部分模块并折叠未训练模块，从而消除初始化噪声并显著降低聚合噪声；

**🔧 技术方法**

核心技术包括低秩适配 LoRA、P​LoRA 并行模块、随机选择与折叠策略、联邦平均聚合以及与传统 FedIT、FLoRA、FlexLoRA、HETLoRA 等方法的对比分析；

**📊 数据集**

实验使用多种规模的 LLM（BERT‑base、Llama‑1B、Llama‑3.1‑8B、OPT‑1.3B、Qwen3‑4B‑A3B‑Instruct‑2507、Mistral‑7B‑v0.3）在自然语言、GLUE、金融、医学、数学等多域数据集（如 Natural Instructions、GLUE、FinGPT、MedAlpaca、MATH 等）进行评测；

**📈 对比分析**

与同类方法比较，Fed-PLoRA 在多任务、多域和 IID/非 IID 场景下均实现了显著提升：在自然指令任务上平均提升 31% Rouge‑L；在 GLUE 上平均提升 39%；在金融数据集上平均提升 13%；同时在通信与计算效率上也优于 FlexLoRA、HETLoRA 等；

**⚠️ 局限性**

局限性包括：对极低秩客户端仍可能产生信息损失；Select‑N‑Fold 随机性可能导致训练不稳定；目前仅在 LoRA 结构下验证，未对更复杂的 PEFT 方法（如 QLoRA、AdaLoRA）做进一步扩展；

---

## 63. Forecasting Anomaly Precursors via Uncertainty-Aware Time-Series Ensembles

**arXiv ID:** 2602.17028 | [PDF](https://arxiv.org/pdf/2602.17028v1)

**作者:** Hyeongwon Kang `[一作]` (Korea University), Pilsung Kang `[通讯]` (Seoul National University)

**通讯引用:** 4428 | [OpenAlex ID](https://openalex.org/A5059650940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于多模型预测不确定性检测异常前驱的无监督框架FATE；

**💡 创新点**

创新点在于利用多样化预测模型的不一致性作为异常前驱信号，并提出新的评价指标PTaPR，兼顾检测准确性与预警及时性；

**🔧 技术方法**

采用多种Transformer/卷积/线性时间序列预测模型组成的集成，并通过对预测方差归一化计算不确定性，随后按阈值进行异常前驱检测；

**📊 数据集**

使用SWaT、PSM、MSL、SMAP、SMD五个工业与空间任务公开基准数据集；

**📈 对比分析**

与多种无监督异常检测基线（LSTM-AE、LSTM-VAE、USAD、DAGMM、OmniAnomaly、Anomaly Transformer、Variable Temporal Transformer）进行比较，在PTaPR、AUC、F1等指标上均取得显著提升（如在PSM上AUC提升约+22%p、在SMAP上+34%p）；

**⚠️ 局限性**

缺点包括需训练并推理多模型导致计算开销大，对极长异常区间的持续检测效果相对弱，以及对归一化方法与阈值选择敏感。

---

## 64. VAM: Verbalized Action Masking for Controllable Exploration in RL Post-Training -- A Chess Case Study

**arXiv ID:** 2602.16833 | [PDF](https://arxiv.org/pdf/2602.16833v1)

**作者:** Zhicheng Zhang `[一作]` (Carnegie Mellon University), Fei Fang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6561 | [OpenAlex ID](https://openalex.org/A5061127138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Verbalized Action Masking（VAM），在后训练阶段通过在提示中显式列出合法动作集合，强制LLM在该集合内生成输出，并结合迭代动作空间剪枝机制提升探索效率，最终在文本化的棋局任务中获得更佳的学习效果与棋局表现。

**💡 创新点**

①将动作掩码转化为提示级别的显式列表，使LLM在合法动作集合内做决策；②构建动作掩码MDP并把可验证奖励与终止信号嵌入其中；③提出按组相对策略优化（GRPO）与迭代剪枝相结合的可控探索框架；④在棋局环境中验证该框架能显著提升谜题通过率与全局对弈质量。

**🔧 技术方法**

组相对策略优化（GRPO）; 动作掩码MDP模型; 提示级别语义掩码实现; 迭代动作空间剪枝算法; Stockfish引擎用于可验证奖励与动作评估; Qwen2.5-3B/7B大型语言模型; 对比实验使用的拒绝采样SFT。

**📊 数据集**

固定的100,000条棋局状态与对应合法动作及Stockfish评估值的固定数据集；通过与Stockfish对弈产生的on‑policy状态流；10,000条Searchless Chess谜题集用于一步动作选择评估；全局对弈评估使用Stockfish 0/5对弈。

**📈 对比分析**

与不使用剪枝的GRPO基线和拒绝采样SFT进行对比；在谜题上采用pass@1指标，在全局对弈上采用平均centipawn loss（ACPL）指标。VAM在两种模型规模下均优于GRPO，并且在7B模型上Qwen2.5-3B+VAM甚至超过Qwen2.5-7B+GRPO；ACPL明显下降，说明对弈质量提升；SFT表现最差。

**⚠️ 局限性**

仅适用于可枚举且可验证的有限动作空间；需要手动提供合法动作列表和精确的验证器；在动作空间极大或无精确评估器的任务中难以直接迁移；对提示设计与解析准确性较为敏感。

---

## 65. Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form

**arXiv ID:** 2602.17078 | [PDF](https://arxiv.org/pdf/2602.17078v1)

**作者:** Xuefeng Wang `[一作]` (Purdue University), Ahmed H. Qureshi `[通讯]` (Purdue University)

**通讯引用:** 1043 | [OpenAlex ID](https://openalex.org/A5056336556)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于epigraph的连续时间多智能体强化学习框架，解决安全约束导致的价值函数不连续问题并实现稳定的actor‑critic学习；

**💡 创新点**

将安全约束通过epigraph改写为辅助变量z的形式，将不连续价值函数转为连续可微的HJB PDE，从而实现PINN在连续时间安全MARL中的有效训练，并引入联合内外优化和VGI等多种损失；

**🔧 技术方法**

使用连续时间受限MDP（CT-CMDP）模型、epigraph重写、物理信息神经网络（PINN）求解HJB PDE、优势函数与学习模型（动力学网络、成本网络）以及改进的actor‑critic策略优化；

**📊 数据集**

在改造后的安全MPE（多粒子环境）与连续时间Safe MA‑MuJoCo（半马达、蚂蚁等）两个基准上进行实验；

**📈 对比分析**

与MACPO、MAPPO‑Lag、SAC‑Lag、EPPO、CBF等安全MARL基线在同一连续时间环境下对比，结果显示所提方法在成本和约束违背率上均优于所有基线，训练更稳定、收敛更快；

**⚠️ 局限性**

仅在仿真环境验证，缺乏对真实系统或噪声鲁棒性的深入评估；方法对辅助变量z的选择敏感，需在不同任务中精细调参。

---

## 66. CAFE: Channel-Autoregressive Factorized Encoding for Robust Biosignal Spatial Super-Resolution

**arXiv ID:** 2602.17011 | [PDF](https://arxiv.org/pdf/2602.17011v1)

**作者:** Hongjun Liu `[一作]` (University of Science and Technology Beijing), Chao Yao `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 7194 | [OpenAlex ID](https://openalex.org/A5044150538)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

针对低密度传感器测量，提出可插拔的通道自回归因式编码框架 CAFE，用于恢复高密度脑电/EMG/ECG/ECoG 信号。

**💡 创新点**

创新点在于将通道分组按几何距离由近到远逐步展开，使用组‑级自回归生成并结合教师强制与计划采样，显著减少曝光偏差；同时框架可与任何时序骨干无缝集成。

**🔧 技术方法**

采用组‑级自回归生成、掩码式通道输入、教师强制、计划采样、共享预测器，并分别实现 Conv、MLP、Transformer 三种骨干。

**📊 数据集**

使用六个公开多通道生物信号数据集：EEG 的 SEED、Localize‑MI，ECoG 的 AJILE12，sEMG1/2，ECG 的 CPSC2018，并在多种放大倍率下评估。

**📈 对比分析**

与 ESTformer、SRGDiff、CGAN、GRIN、TimeMixer++ 等代表方法对比，使用 NMSE、PCC、SNR、Spec‑MAE 等指标；CAFE 在所有数据集和放大倍率下均优于基线，尤其在通道稀疏度高时提升显著，并在计算成本与参数量上更具优势。

**⚠️ 局限性**

局限性包括：对已定义为电压差的 ECG 通道空间耦合弱，CAFE 提升有限；过细的分组会导致长链误差累计；框架依赖固定几何距离划分，若通道布局动态变化或异常可能受限。

---

## 67. Cholec80-port: A Geometrically Consistent Trocar Port Segmentation Dataset for Robust Surgical Scene Understanding

**arXiv ID:** 2602.17060 | [PDF](https://arxiv.org/pdf/2602.17060v1)

**作者:** Shunsuke Kikuchi `[一作]` (Jmees Inc.), Hiroki Matsuzaki `[通讯]` (Jmees Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在Cholec80视频中精准标注端口袖部，并对m2caiSeg与GynSurg进行清洗统一，构建了高质量的Trocar Port Segmentation Dataset。

**💡 创新点**

提出了袖子级别的几何一致性标注规范，排除中心通道，显著提升跨数据集的泛化与鲁棒性。

**🔧 技术方法**

采用ConvNeXt-Base编码器+U-Net解码器的二值语义分割框架，并使用Dice+ BCE 损失进行训练。

**📊 数据集**

主要使用Cholec80前20条视频（共38,434帧），并对m2caiSeg与GynSurg进行重新标注与清洗。

**📈 对比分析**

在仅端口存在帧的Dice和Detect F1两指标下，Cholec80-port训练模型在Cholec80-test上达到了Dice 0.862、Detect F1 0.856，且在m2caiSeg测试集上也优于原始模型，显示出更好的鲁棒性。

**⚠️ 局限性**

仍受域差异影响，端口材质、光照与外观多样性不足导致跨数据集性能下降。

---

## 68. Action-Graph Policies: Learning Action Co-dependencies in Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.17009 | [PDF](https://arxiv.org/pdf/2602.17009v1)

**作者:** Nikunj Gupta `[一作]` (University of Southern California), Viktor Prasanna `[通讯]` (University of Southern California)

**通讯引用:** 17406 | [OpenAlex ID](https://openalex.org/A5033166029)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Action-Graph Policies（AGP）框架，通过动作级别的图结构学习协调上下文，实现分散式强化学习中的协同决策。

**💡 创新点**

创新点在于：① 将协调视作动作之间的关系而非单独代理，构造全局动作依赖图；② 通过图注意力机制聚合动作信息，生成“协调上下文”作为政策输入；③ 证明 AGP 在表达能力和相对最优性上严格优于传统独立策略与基于值分解的方案。

**🔧 技术方法**

采用的技术包括：动作节点编码（观测+动作身份）、共享图注意力网络（多层消息传递）、基于价值或策略梯度的学习、CTDE 训练框架、对抗性损失等。

**📊 数据集**

使用的数据集/任务：Top‑K 选择矩阵游戏（单步）以及六个多智能体粒子环境（Reference、Push、World‑Comm、Speaker‑Listener、Crypto、Tag），均为公开的多智能体基准。

**📈 对比分析**

实验与基线比较：IQL、VDN、QMIX、DCG、DICG、MACPF、FOP 等；在 Top‑K 游戏中 AGP 成功率达 95–96%（基线 10–25%），在粒子环境中 AGP 获得最高回报并在多任务上保持领先。

**⚠️ 局限性**

局限性：仅针对离散动作空间；动作图完全连接导致计算与内存开销随代理数呈指数增长；对更大规模、连续动作或高维观测的适用性仍需进一步验证。

---

## 69. Evaluating Cross-Lingual Classification Approaches Enabling Topic Discovery for Multilingual Social Media Data

**arXiv ID:** 2602.17051 | [PDF](https://arxiv.org/pdf/2602.17051v1)

**作者:** Deepak Uniyal `[一作]` (Queensland University of Technology), Richi Nayak `[通讯]` (Queensland University of Technology)

**通讯引用:** 3642 | [OpenAlex ID](https://openalex.org/A5015158048)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2013-2022年英日韩印四语氢能相关推文进行跨语言相关性分类与主题建模，筛选高质量推文。

**💡 创新点**

首次系统比较四种跨语言分类策略（语言专属、翻译至英、mBERT零射击、混合），并提出混合训练方法。

**🔧 技术方法**

采用BERT/多语言BERT、神经机器翻译、NMF主题模型等技术实现分类与主题发现。

**📊 数据集**

使用约9.3 M条氢能推文（英日韩印），5 k条英文本标签，并将其翻译后与750条手工标注的每语样本用于评估。

**📈 对比分析**

通过10次随机种子实验评估准确率与F1，英文‑仅译法最高（≈97.7 %），混合法在非英语言上最优，mBERT零射击表现最差。

**⚠️ 局限性**

受限于仅有英标签、翻译质量波动、低资源语种性能低，模型对不同域的泛化仍需验证。

---

## 70. Sign Lock-In: Randomly Initialized Weight Signs Persist and Bottleneck Sub-Bit Model Compression

**arXiv ID:** 2602.17063 | [PDF](https://arxiv.org/pdf/2602.17063v1)

**作者:** Akira Sakai `[一作]` (Fujitsu Limited), Yuma Ichikawa `[通讯]` (Riken AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了深度网络权重符号的可压缩性，发现符号几乎不可压缩且训练过程中保持与初始化相似。

**💡 创新点**

提出了符号锁定理论并给出几何尾概率分析，进一步提出基于间隙初始化和外漂移正则化的实用方法来控制符号漂移。

**🔧 技术方法**

采用停止时间分析、几何尾理论、SDE/Markov过程视角以及低秩近似、KS检验等技术。

**📊 数据集**

在MLP‑Mixer、ResNet18、TinyLlama‑1.1B等模型以及语言模型和视觉任务上进行实验。

**📈 对比分析**

与基线和标准训练相比，新方法将符号翻转率降低至10⁻³，验证误差仅提升约1点，且符号矩阵的低秩可压缩性显著提升。

**⚠️ 局限性**

局限在于只验证了简单的锁定方法，未探索更复杂的正则化策略，也未深入分析符号对模型表达能力的贡献。

---

## 71. Robust and Extensible Measurement of Broadband Plans with BQT+

**arXiv ID:** 2602.16969 | [PDF](https://arxiv.org/pdf/2602.16969v1)

**作者:** Laasya Koduru `[一作]` (University of California Santa Barbara), Arpit Gupta `[通讯]` (University of California Santa Barbara)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 BQT+，一种基于抽象 NFA 的宽带计划查询框架，替代传统单一工作流，支持在百余家 ISP 上持续、细粒度、地址级别的宽带可用性、质量与价格数据采集，并用此数据构建 BEAD 预拨款基线和宽带可负担性分析。

**💡 创新点**

创新点：① 抽象化交互状态，将查询意图与执行逻辑分离，使用 NFA 形式的声明式规格；② 通过局部更新（状态/转换）实现对接口变动的鲁棒适配；③ 低技术债，支持非专业用户通过手工或 LLM 代理接口快速编写意图规格；④ 在大规模 ISP 生态中实现可扩展性与长期监测。

**🔧 技术方法**

技术栈：抽象 NFA（状态、动作、转换）、观察–匹配–执行循环、固定动作 API、屏幕录制+OCR 状态检测、PyAutoGUI 自动化、LLM 生成意图（Agentic Interface）、指标测量（LLoS、LLoC、压缩比）等。

**📊 数据集**

数据集：① 64 家 ISP 的长期监测数据（十个地址/ISP，数月）；② 100 家 ISP 的快照覆盖数据；③ 约 62,000 个 Virginia CBG 地址样本（10 家 ISP，10 个县/市）用于可负担性评估；④ 约 62,000 个 BEAD 资格 CBG 地址样本（四州）；⑤ 2023 年 ACS 可支配收入数据，用于可负担性阈值计算；地址来源于 Zillow 和 CostQuest 数据协议。

**📈 对比分析**

与原 BQT（静态工作流）进行对比：在可扩展性上，BQT+ 的每家 ISP 仅需 3–28 个状态，LLoS 8–88 行，LLoC 725–1,225 行；在鲁棒性上，64 家 ISP 的 56 次接口更新平均只需 1–5 个状态变更，LLoS 0–40 行；技术债压缩比低于 0.5；LLM 代理减少 LSC，且 100% 与手工规格达到相同终态。系统支持 64 家 ISP 长期测量、100 家 ISP 快照，保持高命中率。

**⚠️ 局限性**

局限性：仅覆盖宣传的可用性、速度与价格，未测量实际服务质量；部分 ISP 无可查询接口或隐藏优惠；CAPTCHA 未被自动解决；手工/LLM 规格编写仍需人工实验；部分地址可能触发机器人检测，导致查询失败。

---

## 72. Malliavin Calculus as Stochastic Backpropogation

**arXiv ID:** 2602.17013 | [PDF](https://arxiv.org/pdf/2602.17013v1)

**作者:** Kevin D. Oden `[一作]` `[通讯]` (Kevin D. Oden and Associates), Kevin D. Oden (Kevin D. Oden and Associates)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过 Malliavin 积分分部公式统一了路径可微（reparameterization）和 REINFORCE 这两类梯度估计，并在此基础上提出了基于协方差自适应的方差最优混合梯度估计器；

**💡 创新点**

创新点在于：① 证明两种常用梯度估计是同一数学公式的不同实例；② 推导出闭式最优混合系数 λ*，在每个 mini‑batch 中可直接估计；③ 给出有限样本收敛和方差下界的理论保证；

**🔧 技术方法**

主要技术包括 Malliavin 微积分、Stein 识别、积分分部法、协方差估计与正则化、自动微分实现混合梯度；

**📊 数据集**

实验数据集包括 CIFAR‑10（VAE 训练）、合成高耦合 Gaussian 例子以及 MuJoCo 机器人控制环境（HalfCheetah、Hopper、Walker2d）；

**📈 对比分析**

与单一路径可微、REINFORCE 及基准方法相比，混合估计在 VAE 上实现约 9% 方差下降、在强耦合合成问题上可达 35% 下降，并在 RL 环境中在某些任务上提升平均回报；

**⚠️ 局限性**

局限性包括：依赖高斯假设；对 batch size 有 32 的下限要求；非平稳优化（RL）中 λ* 估计易受噪声影响；在离散变量或极度非高斯场景下效果未验证；额外的 10‑20% 计算开销。

---

## 73. "It's like a pet...but my pet doesn't collect data about me": Multi-person Households' Privacy Design Preferences for Household Robots

**arXiv ID:** 2602.16975 | [PDF](https://arxiv.org/pdf/2602.16975v1)

**作者:** Jennica Li `[一作]` (University of Wisconsin Madison), Kassem Fawaz `[通讯]` (University of Wisconsin Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在实验室环境中，通过与15户家庭共同设计的方式，探究多用户家庭环境下人们对家用机器人隐私设计的偏好与实施机制。

**💡 创新点**

创新点在于首次从多用户视角系统化收集家庭机器人隐私设计需求，并提出一套面向制造商的隐私友好设计建议。

**🔧 技术方法**

采用人机交互领域的共创设计方法，使用Wizard-of-Oz技术模拟机器人行为，并进行归纳主题分析。

**📊 数据集**

使用的“数据集”为15户家庭（36名参与者）的访谈记录、调查问卷与共创材料，并绘制家庭平面图。

**📈 对比分析**

比较方法以定性主题分析为主，未涉及量化指标，结果以访谈语料的频率和主题分布呈现。

**⚠️ 局限性**

主要局限包括样本多为白人受过教育者、单次实验室会话、未在真实家庭环境中长期部署，导致结果泛化性受限。

---

## 74. Wink: Recovering from Misbehaviors in Coding Agents

**arXiv ID:** 2602.17037 | [PDF](https://arxiv.org/pdf/2602.17037v1)

**作者:** Rahul Nanda `[一作]` (Meta Platforms, Inc.), Satish Chandra `[通讯]` (Meta Platforms, Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了Wink系统，用于实时检测并纠正大型语言模型驱动的编码代理的错误行为；

**💡 创新点**

提出了以生产流量为基础的三类误行为分类法（Specification Drift、Reasoning Problems、Tool Call Failures）和异步自我干预机制，实现了高效、无阻塞的自我修正；

**🔧 技术方法**

使用LLM分类器、ReAct模式、异步观察器以及LLM-as-judge等技术；

**📊 数据集**

Meta内部VSCode插件产生的生产轨迹（约42,920条静态数据 + 10,000+实时轨迹）；

**📈 对比分析**

通过A/B测试与基线对比，单次干预后误行为恢复率达90%，工具调用失败率降低4.2%，Token使用率和工程师干预率各降低约5%；

**⚠️ 局限性**

局限于Meta内部环境，分类法可能不适用于其他平台；仅覆盖三类误行为，其他潜在错误未被评估；干预延迟或冗余可能导致部分场景无效。

---

## 75. Predictive Batch Scheduling: Accelerating Language Model Training Through Loss-Aware Sample Prioritization

**arXiv ID:** 2602.17066 | [PDF](https://arxiv.org/pdf/2602.17066v1)

**作者:** Sumedh Rasal `[一作]` (Georgia Institute of Technology), Sumedh Rasal `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5093081279)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Predictive Batch Scheduling (PBS)，通过在线预测样本难度并在批次构建中优先选择高损失样本来加速语言模型收敛。

**💡 创新点**

创新点在于将基于损失的在线难度预测与低成本的静态 token 频率等特征相结合，使用轻量级线性预测器动态调整采样，兼顾了课程学习的适应性与硬例挖掘的效率。

**🔧 技术方法**

使用线性损失预测器（四个 token 频率相关特征）、按预测损失分桶的优先采样策略、每 100 步更新权重的 momentum SGD、混合精度训练和梯度累积等技术。

**📊 数据集**

在大规模文本语料上训练 130M 参数的 LLaMA‑style transformer（约 291,258 批次），未明确给出具体数据集名称。

**📈 对比分析**

与均匀随机采样 baseline 进行对比，PBS 在 10,000 步时评估损失比 baseline 提升 6–13%，预测器相关性从 0.14 提升至 0.44，计算开销不到 1%。

**⚠️ 局限性**

局限性包括：实验仅覆盖单一 130M 模型且训练仅 55% 一个 epoch；预测器仅使用四个线性特征，可能无法捕捉更复杂的难度；缺少多模型、多种随机种子或多语言/代码数据集的验证。

---

## 76. Transforming Behavioral Neuroscience Discovery with In-Context Learning and AI-Enhanced Tensor Methods

**arXiv ID:** 2602.17027 | [PDF](https://arxiv.org/pdf/2602.17027v1)

**作者:** Paimon Goulart `[一作]` (University of California Riverside), Evangelos E. Papalexakis `[通讯]` (University of California Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种AI增强的行为神经科学研究管线，通过In-Context Learning（ICL）实现自动行为视频标注，利用神经张量分解（NeAT）提升多模态数据分析，并用检索增强生成（RAG）辅助模型生成可解释的假设；

**💡 创新点**

创新点在于：①将AR-ICL用于时间序列行为标注，显著提升极不平衡数据的识别；②结合NeAT实现非线性张量分解，兼顾可解释性；③利用ICL+RAG构建自动化假设生成器，减少专家手工解释工作；

**🔧 技术方法**

技术包括：In-Context Learning（标准ICL、AR-ICL）、检索增强生成（RAG）、神经张量分解（NeAT）、Coupled CPD、DINOv2嵌入、基于VLM的图像处理；

**📊 数据集**

数据集为UC Riverside实验室的老鼠恐惧条件反射与一般化实验数据，包含视频行为、钙成像视频，共计7只老鼠，约33个试验×6000时间步×若干神经元；

**📈 对比分析**

与DINOv2基准、无ICL、标准ICL、仅时间上下文等方案对比，AR-ICL在行为标注上Macro‑F1 0.545、平衡准确率0.801、MCC 0.517，远超基准；NeAT在张量重构上相较CPD降低约46% RMSE；模型生成的假设与专家解释在Cohen κ 0.59 上表现出中等一致性；

**⚠️ 局限性**

局限包括：ICL在钙成像视频标注效果不佳、对细粒度信号敏感度低；实验依赖外科植入，可能引起组织损伤和光学质量下降；方法对超大规模数据和跨实验迁移的鲁棒性尚待验证。

---

## 77. Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.17062 | [PDF](https://arxiv.org/pdf/2602.17062v1)

**作者:** Yonghyeon Jo `[一作]` (Ulsan National Institute of Science and Technology), Seungyul Han `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 117 | [OpenAlex ID](https://openalex.org/A5091657241)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Successive Sub-value Q-learning (S2Q)框架，利用多层子价值函数并结合Softmax策略来持续追踪并利用次优动作，以快速适应价值函数的动态变化；

**💡 创新点**

通过逐步学习多个子价值函数并抑制已识别的次优动作，S2Q实现了在保持Monotonicity约束的同时能够跟踪并快速切换到新的最优动作，解决了传统价值分解方法在价值变动时易陷入子最优的局限；

**🔧 技术方法**

基于CTDE范式的QMIX混合网络、WQMIX加权TD目标、Softmax行为策略、编码-解码通信机制以及多子价值函数的递归学习；

**📊 数据集**

StarCraft Multi-Agent Challenge (SMAC)的Hard+和Comm套件、Google Research Football (GRF)以及SMACv2等公开仿真环境；

**📈 对比分析**

与QMIX、WQMIX、DOP、FOP、PAC、RiskQ、MARR、MASIA、S2Q-Comm等基线进行比较；S2Q在SMAC-Hard+和GRF任务中均表现出更快的收敛速度和更高的最终胜率，尤其在需要频繁探索的地图上显著优于其它方法；

**⚠️ 局限性**

需要维护多个子价值网络，导致计算和内存开销略有增加；Softmax温度参数需要调节，虽对性能影响不大但仍需经验选择；

---

## 78. LiveClin: A Live Clinical Benchmark without Leakage

**arXiv ID:** 2602.16747 | [PDF](https://arxiv.org/pdf/2602.16747v1)

**作者:** Xidong Wang `[一作]` (Ant Group), Benyou Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了LiveClin，一个基于最新同行评审病例报告、双年更新、覆盖完整临床路径和多模态信息的动态医疗LLM评测基准。

**💡 创新点**

创新点包括：①持续更新的动态基准，①多模态全路径评测，②验证AI‑人类协同生成流程能高效产出更具挑战性的高质量问题，③内置抗污染、抗知识陈旧机制，④提供细粒度多级（ICD‑10章节、疾病簇、编码）分析。

**🔧 技术方法**

采用Generator‑Critic‑Judge三代理循环结合多名医生复核的AI‑人类协同工作流；利用LLM生成、评审、裁决；支持多模态解析（图像、表格、文本）；对话式零样本评估，保持完整问答历史。

**📊 数据集**

数据集来源于2025年上半年PMC开放获取的病例报告共2,150份，最终筛选并验证得到1,407个病例、6,605个多模态问题，包含3,757张医学影像和634份表格。

**📈 对比分析**

通过与26款LLM及100名医生（首席/主治）对比，使用Case Accuracy作为指标；最高模型GPT‑5在LiveClin上仅达35.7%，显著低于首席医师；细分分析显示模型在不同ICD‑10章节、临床阶段和模态上表现差异显著。

**⚠️ 局限性**

局限性包括：数据主要来自高资源地区，罕见病例比例高导致可能的偏倚；AI生成过程仍可能遗漏细节；评测中心化、语言单一，缺乏多语言/多地区验证；对模型推理路径的可解释性分析尚不充分。

---

## 79. Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum

**arXiv ID:** 2602.17080 | [PDF](https://arxiv.org/pdf/2602.17080v1)

**作者:** Minxin Zhang `[一作]` (University of California), Hayden Scheaffer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出两种新型优化器——NAMO 与 NAMO‑D，分别在保持矩阵参数正交化动量的基础上加入 Adam‑style 的自适应矩估计，改进了 Muon 的更新方式。

**💡 创新点**

创新点在于：①将正交化动量与基于范数的 Adam 型噪声自适应机制首次系统结合；②提出单标量与列向量两种扩展，实现全局与神经元级别的噪声自适应；③在理论上给出了确定性与随机性两种情况下的最优收敛速率，证明无论噪声水平如何，方法均能自适应。

**🔧 技术方法**

技术手段包括：矩阵正交化（极化分解或 Newton–Schultz 近似）、自适应一阶/二阶矩估计、归一化比例因子（α_t）与列归一化向量（D_t）以及截断 (clamp) 机制以保证方向良态。

**📊 数据集**

实验使用 OpenWebText 数据集，对 GPT‑2 124M 与 355M 两种规模模型进行预训练，使用 4×H100 GPU、context 长度 1024、批量 480 序列。

**📈 对比分析**

与 AdamW 与 Muon 进行对比，NAMO 与 NAMO‑D 在训练与验证损失上均表现更低，收敛更快且对学习率更鲁棒；NAMO‑D 通过调节截断参数 c 可进一步提升性能。

**⚠️ 局限性**

局限性包括：仅在中等规模 LLM 上验证；对极大模型的效果尚未评估；需要额外调参（学习率、clamp 参数）且实现仍依赖近似正交化；对非矩阵结构参数的推广尚待探索。

---

## 80. A Unified Framework for Locality in Scalable MARL

**arXiv ID:** 2602.16966 | [PDF](https://arxiv.org/pdf/2602.16966v1)

**作者:** Sourav Chakraborty `[一作]` (University of Colorado), Lijun Chen `[通讯]` (University of Colorado)

**通讯引用:** 7216 | [OpenAlex ID](https://openalex.org/A5100398616)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于策略诱导的相互依赖矩阵分解，并给出谱半径<1的更严格指数衰减条件，构建了可证明的局部化策略改进框架；

**💡 创新点**

核心创新在于将环境对状态/动作的敏感度与策略对状态的敏感度解耦，提出了 H^π = E^s + E^aΠ(π) 的谱半径判据，揭示了温度τ可调节局部性与最优性的权衡；

**🔧 技术方法**

采用坐标振荡、Dobrushin-型不等式、谱半径分析以及软最大熵正则化的 KL‑proximal 更新，配合消息传递实现 κ‑hop 近似；

**📊 数据集**

论文主要以理论分析为主，使用无监督的随机生成的网络化 MARL 环境（如随机耦合图）进行验证；

**📈 对比分析**

在与传统基于无条件耦合矩阵（C_∞<1）或全局更新方法对比时，实验表明在同等误差下 κ 可以从数百降到十数，速度提升数十倍；

**⚠️ 局限性**

局限在于仅适用于产品式策略与已知 E^s、E^a、Π(π) 的离散有限状态空间，且需事先估计或计算谱半径，实际在线学习时对环境模型的依赖仍显重；

---

## 81. Boreas Road Trip: A Multi-Sensor Autonomous Driving Dataset on Challenging Roads

**arXiv ID:** 2602.16870 | [PDF](https://arxiv.org/pdf/2602.16870v1)

**作者:** Daniil Lisus `[一作]` (University of Toronto), Timothy D. Barfoot `[通讯]` (University of Toronto)

**通讯引用:** 7487 | [OpenAlex ID](https://openalex.org/A5004788089)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了 Boreas-RT 数据集，包含 60 条序列、643 km 的多传感器记录（摄像头、激光雷达、雷达、IMU、轮速计），并提供厘米级 GNSS‑INS 地面真值、完整的标定信息、开发工具和公开排行榜。

**💡 创新点**

创新点在于将广度（9 条不同道路类型）与深度（多次重复遍历、不同天气与交通）相结合，首次公开 FMCW 激光雷达与传统雷达的融合数据，并提供统一的高精度真值与公开对比平台，促进算法的跨场景评估。

**🔧 技术方法**

使用多传感器同步采集技术、基于 Applanix RTX 的后处理、IMU 与激光雷达的 3D ICP 标定、雷达极坐标到笛卡尔转换、点云去畸变与时间同步等方法构建完整的数据管线和 Python 开发工具。

**📊 数据集**

数据集为 Boreas‑RT 本身，基于原 Boreas 多季数据扩展，涵盖 9 条路段、60 条序列，覆盖城市、郊区、乡村和高速公路等多种道路环境。

**📈 对比分析**

采用 KITTI 风格漂移指标对 SE(2)/SE(3) 里程计与定位算法（DRO、RTR、LTR、D‑Aeva）进行统一评估，结果显示 SOTA 算法在结构化路段表现优异，但在农村、山区高速及雪天等极端条件下漂移显著，表明算法在不同环境下易过拟合。

**⚠️ 局限性**

局限性包括：仅部分序列配备 FMCW 雷达，隧道/高楼遮挡导致 GNSS 失效并影响地面真值精度；部分路段的地面真值误差高（Urban Canyon、雪堆导致垂直误差）；雷达与激光雷达的标定精度受限；缺乏跨传感器、多天气、多时间段的统一评估，仍需进一步研究鲁棒性。

---

## 82. Hybrid-Gym: Training Coding Agents to Generalize Across Tasks

**arXiv ID:** 2602.16819 | [PDF](https://arxiv.org/pdf/2602.16819v1)

**作者:** Yiqing Xie `[一作]` (Carnegie Mellon University), Daniel Fried `[通讯]` (Carnegie Mellon University)

**通讯引用:** 9893 | [OpenAlex ID](https://openalex.org/A5003637850)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并构建了可扩展的合成训练任务集，提升编码代理在多种软件工程任务上的泛化。

**💡 创新点**

提出了四项训练任务设计原则（输出格式匹配、仓库探索、非平凡推理、简易设置），并通过这些原则创建了无需可执行仓库的合成任务。

**🔧 技术方法**

采用OpenHands框架、拒绝采样微调、LLM生成轨迹、bash工具调用、static分析等技术实现训练与评估。

**📊 数据集**

训练使用自研的Hybrid‑Gym数据集（约4.4k轨迹，涵盖函数定位、问题定位、依赖搜索、函数生成），评估使用SWE‑Bench、SWT‑Bench、Commit‑0等基准。

**📈 对比分析**

与现有单任务训练集（如SWE‑Gym、SWE‑Play）和原始模型对比，未加入下游任务训练时，在SWE‑Bench Verified提升25.4%，SWT‑Bench Verified 7.9%，Commit‑0 Lite 5.1%，并在多项指标上显著优于基线。

**⚠️ 局限性**

仍依赖强大教师模型生成轨迹，任务多样性有限，未覆盖代码执行/测试等步骤，对不同语言/工具生态的适应性尚待验证。

---

## 83. Phase-Aware Mixture of Experts for Agentic Reinforcement Learning

**arXiv ID:** 2602.17038 | [PDF](https://arxiv.org/pdf/2602.17038v1)

**作者:** Shengtian Yang `[一作]` (Kuaishou Technology), Lei Feng `[通讯]` (Southeast University)

**通讯引用:** 133594 | [OpenAlex ID](https://openalex.org/A5100659481)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种阶段感知混合专家（Phase-Aware Mixture of Experts，PA-MoE）模型，用于强化学习中的大型语言模型代理，解决单策略网络导致的简单化偏差。

**💡 创新点**

创新点在于引入轻量级相位路由器，以时间段为粒度而非 token 级别分配专家，保证同一阶段使用同一专家，从而实现专家专化并抑制简单化偏差。

**🔧 技术方法**

技术细节包括：LoRA 自适应专家共享冻结的 LLM 胶囊；交叉注意力 + LSTM 的路由器；温度退火、切换惩罚、平衡与多样性正则化；以及将 PA-MoE 与任意策略梯度算法（如 GiGPO、PPO、RLOO、GRPO）无缝结合。

**📊 数据集**

实验数据集为 ALFWorld（多步家庭物体交互任务）和 WebShop（基于文本目标的电商导航），使用 Qwen2.5-1.5B/7B 等大语言模型。

**📈 对比分析**

对比方法包括传统单策略 RL 以及多种强化学习算法，结果显示 PA-MoE 在 ALFWorld 上从 86.1% 提升至 93.8%，在 WebShop 上从 67.4% 提升至 82.3%，在多种算法上均实现 4–15% 的平均提升，并且参数量更低。

**⚠️ 局限性**

局限性包括：需要手动调节专家数 K，过多专家会导致数据稀疏；实验仅涉及离散动作，连续控制可能需改造；路由器学习仍受训练稳定性的影响，跨域迁移效果尚未充分验证。

---

## 84. Eigenmood Space: Uncertainty-Aware Spectral Graph Analysis of Psychological Patterns in Classical Persian Poetry

**arXiv ID:** 2602.16959 | [PDF](https://arxiv.org/pdf/2602.16959v1)

**作者:** Kourosh Shahnazari `[一作]` (Sharif University of Technology), Mohammadali Keshtparvar `[通讯]` (Amirkabir University of Technology)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5103051276)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于多标签自动注释且显式处理置信度与不确定性的古典波斯诗歌心理模式分析框架，能够将每句诗的心理标签与置信度映射到诗人级的概率分布，并通过与全语料基准的散度量化诗人的个体化程度；同时利用置信度加权的共现图谱和拉普拉斯谱分解得到的Eigenmood空间，对诗歌中的情感关系进行结构化表示并支持按谱轴检索典型句子；

**💡 创新点**

创新点包括：①将置信度与放弃标记作为核心数据维度，构建置信度加权聚合与分散度量；②引入置信度加权共现图谱与谱分解，得到的Eigenmood坐标既捕捉情感概念的共性也体现其相互作用；③在诗人层面引入偏差诊断（放弃视作类别）和置信度阈值下的稳健性评估，形成完整的误差与偏差可视化；

**🔧 技术方法**

使用了Transformer大模型（Gemini 2.5 Flash）进行多标签心理概念预测，随后对置信度与放弃进行加权求和得到诗人×概念矩阵；采用Jensen–Shannon与KL散度度量诗人分布与全基准的差异；构建置信度加权共现矩阵，计算无归一化拉普拉斯矩阵并进行特征分解得到Eigenmood坐标；通过阈值过滤、统一权重对比、Bootstrap等方法评估稳健性与选择偏差；

**📊 数据集**

共计61,573句诗，涵盖10位古典波斯诗人（Athir、Eraghi、Hafez、Jahan、Khaghani、Khayyam、Parvin、Saadi、Shahriar、Vahshi），使用Ganjoor数字语料库作为文本来源；

**📈 对比分析**

与传统情感计算方法相比，该框架在保留不确定性信息的前提下实现了更细粒度的诗人个体化度量：在置信度阈值0.7下，诗人排序相关性ρ≈0.96；与统一权重比较，散度排名ρ=1.00；两位人工标注者达成宏观一致κ≈0.82，整体精度≈0.80，放弃判定正确率≈0.86，校准误差ECE≈0.035，表明置信度与放弃对下游分析具有实用价值；

**⚠️ 局限性**

主要限制包括：①标签集稀疏（如idealization极少），导致统计不稳；②模型对古典语义的偏差与时间跨越的文化差异，可能引入历史病态化；③放弃率高达22%，在某些诗人中更为显著，可能导致基准对比失衡；④仅覆盖10位诗人，难以推广到更广泛的波斯文学；⑤缺乏对标签间潜在互相影响的更细粒度建模，未来可考虑更复杂的图卷积或因子分解等方法。

---

## 85. DeepContext: Stateful Real-Time Detection of Multi-Turn Adversarial Intent Drift in LLMs

**arXiv ID:** 2602.16935 | [PDF](https://arxiv.org/pdf/2602.16935v1)

**作者:** Justin Albrethsen `[一作]`, Sharath Rajasekar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于循环神经网络的状态化安全监控框架 DeepContext，用来追踪多轮对话中的用户意图演变，阻止“慢燃”式的 jailbreak 攻击；

**💡 创新点**

核心创新是将安全判定从单轮静态评估转为动态轨迹识别，利用任务注意力加权嵌入与 GRU 隐藏状态相结合，实现对意图漂移的实时捕捉；

**🔧 技术方法**

技术包括：细调 BERT 提取任务注意力加权嵌入、GRU 递归状态更新、投影层+残差短路、MLP 轨迹分类、焦点损失与 1‑epoch 训练；

**📊 数据集**

训练数据 437,058 条对话，约 20% 为恶意，来源包含 LLMail、HH‑RLHF、XGuard、PRODIGy、MultiWOZ、WikiQA、SafeDial、CoSafe、DEF CON 等公开或内部数据集；

**📈 对比分析**

在 1,010 条多轮 jailbreak 基准上，DeepContext F1=0.84、召回 0.83、精确率 0.86，平均检测轮数 4.24，推理时延 19 ms，显著优于 Llama‑Prompt‑Guard‑2（F1 0.67）、Granite‑Guardian‑3.3（F1 0.67）、云端防护等；在单轮 JailBreakBench 上 F1 0.98，几乎完美；

**⚠️ 局限性**

局限包括：在函数调用等技术性场景可能产生误报；对未知新型攻击仍有一定误检率（如 Qwen3Guard 的“玻璃炮”问题）；依赖任务注意力加权对恶意信号的准确性；未来需进一步区分技术与恶意语义。

---

## 86. Training Large Reasoning Models Efficiently via Progressive Thought Encoding

**arXiv ID:** 2602.16839 | [PDF](https://arxiv.org/pdf/2602.16839v1)

**作者:** Zeliang Zhang `[一作]` (University of Rochester), Jianfeng Gao `[通讯]` (Microsoft Research)

**通讯引用:** 38346 | [OpenAlex ID](https://openalex.org/A5114910293)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大规模推理模型在强化学习训练中的内存与计算瓶颈，提出了Progressive Thought Encoding（PTE）方法，在固定缓存大小下通过逐步编码被驱逐的推理过程，保持长期上下文信息，从而在RL训练和推理阶段显著提升效率与准确率。

**💡 创新点**

创新点在于：①将被驱逐的token信息压缩成固定维度向量并直接更新LoRA权重，构成“持续记忆”机制；②在RL训练中将缓存约束转化为优化目标，兼顾样本质量与内存限制；③利用全局上下文token初始化与动态更新，进一步提升记忆效果。

**🔧 技术方法**

技术：基于LoRA的参数高效微调、分组强化学习（GRPO）框架、KV缓存滑窗剔除策略、全局查询向量与压缩映射、强化学习奖励归一化与KL正则化。

**📊 数据集**

数据集：DAGPO-Math-17K作为训练集；六个数学推理基准（Math500、OlympiadBench、Minerva Math、AMC、AIME2024、AIME2025）用于评估。

**📈 对比分析**

对比方法：Baseline（原始模型）、LoRA、LoRA_c（滑窗+LoRA）。实验显示，PTE在所有模型上平均提升约19%（相较LoRA）和30%（相较Baseline）的pass@1/16准确率，同时峰值GPU内存和TFLOPs均显著下降（约一半），在AIME等难题上提升可达+23.4%。

**⚠️ 局限性**

局限性：目前仅在单一滑窗驱逐策略下验证，未针对更复杂或自适应驱逐方法进行完整集成；在推理时仍需保留一定全局token，若进一步压缩可能影响性能；实验仅聚焦数学推理任务，对多模态或自然语言生成场景的适用性待验证。

---

## 87. References Improve LLM Alignment in Non-Verifiable Domains

**arXiv ID:** 2602.16802 | [PDF](https://arxiv.org/pdf/2602.16802v1)

**作者:** Kejian Shi `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 271737 | [OpenAlex ID](https://openalex.org/A5042321575)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探究了高质量参考输出在LLM对齐调优中的作用，并通过参考导向的LLM评判器实现半自我改进

**💡 创新点**

创新点在于将参考答案与LLM评判器相结合，形成软“验证器”，并在无偏好标签的环境下实现自我改进

**🔧 技术方法**

主要技术包括参考引导的LLM评判（RefEval、RefMatch）与DPO偏好优化、SFT微调以及基于前沿LLM生成的参考输出

**📊 数据集**

使用了五个人类标注评估数据集（LLMBar-Natural/Adversarial、MTBench、Instrusum、HREF）以及UltraFeedback作为训练指令集，DeepSeek‑V3生成参考输出

**📈 对比分析**

与传统无参考评判、CoT、Self‑Ref等方法对比，RefEval平均准确率达到79.1%，在SFT+DPO自我改进中比基线提升≈21分，且与训练好的ArmoRM奖励模型相当

**⚠️ 局限性**

局限性包括对高质量参考生成的依赖、对开放式任务参考利用效果有限以及在更专业领域的适用性尚未验证

---

## 88. "My body is not your Porn": Identifying Trends of Harm and Oppression through a Sociotechnical Genealogy of Digital Sexual Violence in South Korea

**arXiv ID:** 2602.16853 | [PDF](https://arxiv.org/pdf/2602.16853v1)

**作者:** Inha Cha `[一作]` (Georgia Institute of Technology), EunJeong Cheon `[通讯]` (Syracuse University)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5014866832)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性梳理韩国数字性暴力（DSV）自1990年代至2025年的四个历史阶段，并基于社会技术家谱法提出三维框架（‘审美化’/‘不可察觉性’/‘商业化’），以揭示性别偏见与技术变迁的交互作用；

**💡 创新点**

创新点在于将社会技术家谱方法应用于DSV研究，首次将四个时代与三维结构连贯映射，突出‘不可察觉性’与‘商业化’的隐性维度，并为后续设计与政策干预提供系统性视角；

**🔧 技术方法**

采用社会技术家谱、定性内容分析与案例研究等方法，结合文本挖掘技术对法律文件、新闻报道与学术文献进行系统编码；

**📊 数据集**

数据集主要包括：①法律法规文本（如性犯罪处罚法、儿童性保护法等）②新闻媒体报道与调查报告（如《韩联社》《赫德报》等）③学术论文与研究报告（共计47篇）以及与案件相关的社会化媒体记录；

**📈 对比分析**

本研究不涉及算法性能评估，而是通过跨案例对比与政策变迁跟踪，展示各时代的技术与法律响应，表现为对比性叙事与政策效果的质性评估；

**⚠️ 局限性**

局限性：研究聚焦韩国语料，缺乏跨文化比较；数据来源受限于公开文件与媒体报道，可能存在信息不完整或偏差；采用叙事性家谱方法，缺乏可量化验证，研究结果主要为定性洞见。

---

## 89. Uniform interpolation with constructive diamond

**arXiv ID:** 2602.16880 | [PDF](https://arxiv.org/pdf/2602.16880v1)

**作者:** Iris van der Giessen `[一作]` (University of Amsterdam), Ian Shillito `[通讯]` (University of Birmingham)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5017336560)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

证明了构造性模态逻辑 CK 与 WK 的统一插值性，并在 Rocq 中形式化。

**💡 创新点**

首次为具备独立 □ 与 ◇ 的直觉主义模态逻辑证明统一插值性，并将 Pitts 技术推广到构造性体系。

**🔧 技术方法**

采用终止的单后继 sequent 计算、归约证明与 cut 消除，结合 Rocq 证明助手进行形式化。

**📊 数据集**

无实验数据集，纯理论证明。

**📈 对比分析**

未进行实验比较，证明复杂度未评估，理论上可决定性。

**⚠️ 局限性**

仅适用于 CK 与 WK，其他直觉主义模态逻辑未知；计算复杂度仍需进一步研究。

---

## 90. Machine Learning Argument of Latitude Error Model for LEO Satellite Orbit and Covariance Correction

**arXiv ID:** 2602.16764 | [PDF](https://arxiv.org/pdf/2602.16764v1)

**作者:** Alex Moody `[一作]` (University of Colorado Boulder), Rebecca Russell `[通讯]` (Charles Stark Draper Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对低轨卫星轨道误差，构建基于机器学习的AOL误差预测与协方差校正方法。

**💡 创新点**

提出仅预测AOL一维误差并映射到笛卡尔空间的简化模型，实现多类型卫星的单模型通用性。

**🔧 技术方法**

使用时序条件前馈神经网络（TCNN）和异方差高斯过程（HGP），并结合Orekit SP轨道推算器进行误差估计。

**📊 数据集**

训练集来源于公开的VCM（Vector Covariance Message）数据与Orekit推算的七日误差，覆盖1000颗LEO卫星。

**📈 对比分析**

通过与未校正基线比较，两个模型均将沿轨误差标准差降低约一半，Mahalanobis一致性从7.7%提升至80%以上。

**⚠️ 局限性**

局限包括VCM仅提供位置sigma，缺乏完整协方差；模型对时间外推的泛化仍有限；残余误差仍达数公里，无法满足高精度PNT需求。

---

## 91. StructCore: Structure-Aware Image-Level Scoring for Training-Free Unsupervised Anomaly Detection

**arXiv ID:** 2602.17048 | [PDF](https://arxiv.org/pdf/2602.17048v1)

**作者:** Joongwon Chae `[一作]` (Tsinghua University), Ilmoon Chae `[通讯]` (Ratel Soft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在无监督缺陷检测中，提出了StructCore模块，用训练免费地对异常分数图进行结构化图像级评分，以替代传统的最大池化。

**💡 创新点**

创新点在于通过低维结构描述子（标准差、尾部均值、总变差）对异常图进行统计校准，并在不改动像素级定位的前提下提升图像级决策。

**🔧 技术方法**

采用ViT特征提取、随机投影、贪心coreset、kNN检索、结构化统计、马氏距离校准和自动尺度匹配等技术。

**📊 数据集**

在MVTec AD和VisA两个工业缺陷检测基准上进行实验。

**📈 对比分析**

与PatchCore、PaDiM、CFA等基准相比，StructCore在图像级AUROC上提升至99.6%（MVTec AD）和98.4%（VisA），保持定位性能不变。

**⚠️ 局限性**

局限性包括对λ等超参数的敏感性、仅提升图像级决策而非定位、对特定异常分布的依赖，以及在非内存库检测框架下效果未知。

---

## 92. A Real-Time Approach to Autonomous CAN Bus Reverse Engineering

**arXiv ID:** 2602.16722 | [PDF](https://arxiv.org/pdf/2602.16722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 93. Learning to Recommend in Unknown Games

**arXiv ID:** 2602.16998 | [PDF](https://arxiv.org/pdf/2602.16998v1)

**作者:** Arwa Alanqary `[一作]` (University of California), Alexandre M. Bayen `[通讯]` (University of California)

**通讯引用:** 14685 | [OpenAlex ID](https://openalex.org/A5021116704)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究在多智能体博弈环境中，平台通过发出行动建议并观测玩家是否遵从来学习玩家的偏好，并提出了相应的低回报推荐算法。

**💡 创新点**

创新点在于：①证明在量化响应（quantal‑response）反馈下，游戏可在正比例变换类内完全可识别；②在最佳响应（best‑response）反馈下给出完整的不可区分游戏集合的几何表征；③设计了基于切平面法的在线低回报算法，证明其在两种反馈模型下均可实现 O(nM log T) 的期望累计回报。

**🔧 技术方法**

主要技术包括逆博弈理论、几何凸分析（极点、法向锥、极化多面体）、二分搜索恢复偏差向量、稀疏线性系统求解，以及切平面/上下界搜索（contextual search）构造分离超平面。

**📊 数据集**

本工作为理论性研究，未使用具体数据集；所有结果均基于对任意有限规范博弈的理论分析。

**📈 对比分析**

方法相较于传统逆博弈或基于均衡的推荐策略，样本复杂度呈对数增长（log(1/ε)），回报率随迭代次数呈对数增长；在量化响应模型下可达到最佳可学习类的精度，而在最佳响应模型下可识别的类更大。

**⚠️ 局限性**

限制包括：①对最优响应反馈，无法完全识别玩家效用，仅能确定更大的一致性类；②假设博弈无弱支配策略并满足效用差分可辨识；③需要平台能生成任意概率分布的建议；④算法复杂度与博弈维度 nM 成线性比例，实际大规模博弈中实现可能受限。

---

## 94. Fundamental Limits of Black-Box Safety Evaluation: Information-Theoretic and Computational Barriers from Latent Context Conditioning

**arXiv ID:** 2602.16984 | [PDF](https://arxiv.org/pdf/2602.16984v1)

**作者:** Vishal Srivastava `[一作]` `[通讯]` (Johns Hopkins University), Vishal Srivastava (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了黑箱安全评估在AI系统中的局限性，特别是模型在测试分布上的表现无法可靠预测其在部署时的性能。通过引入潜在上下文条件策略，作者挑战了这一假设，并建立了基本限制，表明在某些情况下，黑箱评估无法可靠估计部署风险。

**💡 创新点**

创新点在于提出了潜在上下文条件模型的概念，展示了在评估和部署环境中触发器的概率差异如何影响模型的安全性评估。此外，论文提供了明确的统计界限，表明在特定条件下，黑箱测试是统计上不足的，并且需要额外的安全保障措施。

**🔧 技术方法**

使用了Le Cam的两点方法、Fubini定理、Yao的极小极大原则等数学工具，结合了概率论和信息论的技术来推导结果。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在不同的评估和部署环境中可能出现的触发器概率差异的实际例子，如语言模型在有毒语言和自驾车在雪天驾驶的场景。

**📈 对比分析**

通过被动评估和自适应评估的方法进行比较，结果表明在小曝光率的情况下，黑箱评估的最小绝对误差下界为(5/24)δ L，而自适应评估的最坏情况期望绝对误差下界为(7ε L/32)。这表明在特定条件下，黑箱评估的性能是有限的。

**⚠️ 局限性**

论文的局限性包括假设完美的不可观察性，依赖于标准的密码学假设，白箱模型的范围限制，以及对自适应评估者的高概率保证仍然是开放问题。

---

## 95. Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation

**arXiv ID:** 2602.16990 | [PDF](https://arxiv.org/pdf/2602.16990v1)

**作者:** Yan Wang `[一作]` (Fin AI), Jian-Yun Nie `[通讯]` (University of Montreal)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Conv‑FinRe，一个对话式、纵向的金融推荐基准，评估LLM的决策质量而非仅仅行为匹配。

**💡 创新点**

创新点在于将推荐评估从行为复制转向以投资者隐含效用为依据的多视角对齐，并通过逆优化提取风险偏好。

**🔧 技术方法**

采用逆优化估计用户风险参数、构建多视角参考、对话模拟、以及uNDCG、MRR等评估指标；评测多种LLM。

**📊 数据集**

使用10名真实投资者的30天交互轨迹、S&P500 10支股票的市场数据，以及生成的对话和专家建议。

**📈 对比分析**

与多种LLM（GPT‑5.2、GPT‑4o、DeepSeek、Qwen、Llama‑3、XuanYuan等）对比，结果显示LLM在效用排名上表现优异，但在行为匹配上存在权衡，体现了理性决策与行为一致性的张力。

**⚠️ 局限性**

局限包括样本量小（10人）、股票池有限、对话模拟与真实用户行为的差距，以及对长期偏好识别的挑战。

---

## 96. Sonar-TS: Search-Then-Verify Natural Language Querying for Time Series Databases

**arXiv ID:** 2602.17001 | [PDF](https://arxiv.org/pdf/2602.17001v1)

**作者:** Zhao Tan `[一作]` (Jiangxi University of Finance and Economics), Ming Jin `[通讯]` (Griffith University)

**通讯引用:** 13154 | [OpenAlex ID](https://openalex.org/A5039636381)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了用于时间序列数据库的自然语言查询框架 Sonar-TS，并构建了 NLQTSBench 评测基准，解决了大规模连续数据中形态匹配与长时段检索难题。

**💡 创新点**

创新点在于“Search‑Then‑Verify”神经符号管道：先用多尺度特征索引（如 SAX）在 SQL 中粗粒度搜索候选窗口，再用 LLM 生成 Python 验证程序精确定位。

**🔧 技术方法**

技术包括多尺度特征表构建、SAX 形态符号、LLM 任务规划与代码生成、Prompt Cold‑Start 经验更新、SQL 与 Python 混合执行。

**📊 数据集**

使用了基于真实工业 TSDB（CausalRivers）的 NLQTSBench（831 条查询、平均 11,000 点窗口）以及其 Lite 512 点版本。

**📈 对比分析**

与现有 Text‑to‑SQL（MAC‑SQL、Xiyan‑SQL、Omini‑SQL）和时间序列模型（ChatTS、ITFormer、Time‑R1）对比，Sonar‑TS 在所有四级任务均显著提升，平均得分提升约 20‑30%，尤其在形态识别、周期检测和洞察生成上表现突出。

**⚠️ 局限性**

局限在于对 SAX 压缩的依赖导致短窗口细节丢失，经验模块规模受限，且目前仅支持 SQL‑兼容 TSDB，未针对异构存储或实时流场景进行验证。

---

## 97. Patch-Based Spatial Authorship Attribution in Human-Robot Collaborative Paintings

**arXiv ID:** 2602.17030 | [PDF](https://arxiv.org/pdf/2602.17030v1)

**作者:** Eric Chen `[一作]` (University of Michigan), Patricia Alves-Oliveira `[通讯]` (University of Michigan)

**通讯引用:** 1592 | [OpenAlex ID](https://openalex.org/A5060086372)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种补丁级卷积网络框架，在极少样本的人工-机器人协作绘画中实现空间作者归因，并利用预测熵识别混合作者区域。

**💡 创新点**

创新点包括：①在数据缺乏的情境下首次使用补丁级CNN实现人机样式区分；②通过条件Shannon熵量化混合作者的风格重叠；③在单一人机配对上达到88.8%补丁级准确率。

**🔧 技术方法**

采用VGG风格CNN、留一绘画交叉验证、数据增强、加权交叉熵损失以及熵不确定性分析。

**📊 数据集**

使用15幅物理绘画（7人画，8机画，5协作），以1200 DPI平板扫描得到300×300像素补丁，共计约137K补丁。

**📈 对比分析**

与LBP+RF、ResNet‑50+SVM、DINOv2+SVM三种基线相比，补丁级准确率从65.9%提升至88.8%，单幅画多数投票达到86.7%；混合画区块的预测熵比纯画高约64%（p=0.003），证明模型能捕捉混合作者特征。

**⚠️ 局限性**

局限性包括仅针对单一人机配对，缺乏多样性；混合作者区块缺乏精确标注；模型对阈值和特征提取的依赖；未利用时间序列或过程信息。

---

## 98. AdvSynGNN: Structure-Adaptive Graph Neural Nets via Adversarial Synthesis and Self-Corrective Propagation

**arXiv ID:** 2602.17071 | [PDF](https://arxiv.org/pdf/2602.17071v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11904 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种端到端的图神经网络框架 AdvSynGNN，结合多尺度结构编码、对比学习、基于生成对抗的拓扑扰动与自适应残差校正，以提高在高异构性和结构噪声环境下的节点级表示和预测性能。

**💡 创新点**

创新点在于：①将对抗性拓扑合成与生成器/判别器嵌入训练中，实现结构扰动的正则化；②设计基于节点置信度的残差校正机制，动态抑制噪声传播；③在 Transformer 关注机制中加入可学习的结构偏置，适配异构图；④多尺度特征融合与对比学习相结合，提升表达鲁棒性。

**🔧 技术方法**

主要技术包括：图注意力 Transformer、对比学习（自监督对比损失）、生成对抗网络（GAN）用于拓扑扰动、谱正则化与自适应置信度估计、残差校正与迭代收敛分析、以及混合精度/检查点优化以支持百万节点级别。

**📊 数据集**

实验数据集覆盖广泛：OGBN‑ArXiv、OGBN‑Products、OGBN‑Proteins、DBLP、PCQM4Mv2（量子化学回归）、四大社交/学术/电商网络、以及时间序列预测数据集 ECG、Traffic、Motor；同时在链接预测任务使用 arXiv‑AstroPh、arXiv‑GrQc、Wikipedia、Amazon2M。

**📈 对比分析**

与多种基准（GCN、GraphGAN、Graphormer、SGFormer 等）在节点分类、回归、链接预测等任务上对比，AdvSynGNN 在大多数指标上实现了显著提升：节点分类精度最高（如 OGBN‑ArXiv 75.48%）、回归 MAE 最低（PCQM4Mv2 0.108）、链接预测 AUC 最高（90.86%），并在结构噪声下表现出最小性能衰减（最大 AUC 降低仅 2.05%）。

**⚠️ 局限性**

局限性包括：①模型相对复杂，训练时需要多模块协同，导致实现与调参成本较高；②在极端低 homophily 或极大规模（>10M 节点）下仍可能出现内存或收敛问题；③对抗性扰动生成器的训练可能导致梯度不稳定，需额外的正则化与梯度惩罚；④目前缺乏对模型可解释性的深入分析，尤其是注意力偏置和生成器决策的可解释性。

---

## 99. SparTa: Sparse Graphical Task Models from a Handful of Demonstrations

**arXiv ID:** 2602.16911 | [PDF](https://arxiv.org/pdf/2602.16911v1)

**作者:** Adrian Röfer `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2546 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用演示分割、事件生成和对象匹配，构建稀疏的任务骨架模型，并在真实机器人上实现无缝迁移。

**💡 创新点**

①使用基于互信息的连接概率模型替代阈值；②通过预训练视觉特征实现跨演示对象匹配；③以最小熵分布为目标提炼任务步骤。

**🔧 技术方法**

互信息（MI）与多元高斯估计、图结构事件抽取、k‑assignment（对象匹配）、基于概率的姿态分布、梯度优化执行控制。

**📊 数据集**

HANDsOME 与 Robocasa 两个包含 3D 轨迹的演示数据集，共 950 条演示。

**📈 对比分析**

与传统基于距离阈值的分割方法相比，分割成功率从 65% 提升至 85%；任务模型提取在多示例下结构准确率快速饱和，平均成功率约 57%。

**⚠️ 局限性**

对同步双手操作的任务分割效果差；缺乏对工具使用等复杂操作的建模；对超参数（如 MI 阈值）敏感，需人工调优。

---

## 100. Beyond Chunk-Then-Embed: A Comprehensive Taxonomy and Evaluation of Document Chunking Strategies for Information Retrieval

**arXiv ID:** 2602.16974 | [PDF](https://arxiv.org/pdf/2602.16974v1)

**作者:** Yongjie Zhou `[一作]` (University of Queensland), Guido Zuccon `[通讯]` (Google)

**通讯引用:** 4811 | [OpenAlex ID](https://openalex.org/A5076031002)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新实现并统一评估文档分块策略在密集检索中的效果，比较结构化、语义化和LLM引导分块以及前置分块与上下文化嵌入的顺序。

**💡 创新点**

提出一个两维统一框架，系统化对齐分块方法与嵌入顺序；在不同检索任务（文档内检索与语料库检索）中验证分块策略与上下文化嵌入的任务依赖性。

**🔧 技术方法**

使用多种分块技术（段落、固定大小、句子、语义相似、命题分块、LumberChunker），多种嵌入模型（Jina‑v2、Jina‑v3、Nomic、E5‑large），以及上下文化嵌入（post‑embedding chunking）。

**📊 数据集**

使用GutenQA（文档内检索）和BEIR六个数据集（FiQA、ArguAna、SciDocs、TREC‑COVID、SciFact、NFCorpus）进行评估。

**📈 对比分析**

对比方法时采用Pre‑C与Con‑C两种顺序，报告DCG@10或nDCG@10；结果显示结构化分块在语料库检索中最优，LLM引导的LumberChunker在文档内检索中最优；上下文化嵌入提升语料库检索但削弱文档内检索。

**⚠️ 局限性**

局限性包括：仅针对固定的四种嵌入模型；对LLM引导分块的成本和推理延迟未量化；仅考虑了单一语言（英文）；未深入探究不同阈值或分块粒度对结果的影响。

---

## 101. Exploring the Design and Impact of Interactive Worked Examples for Learners with Varying Prior Knowledge

**arXiv ID:** 2602.16806 | [PDF](https://arxiv.org/pdf/2602.16806v1)

**作者:** Sutapa Dey Tithi `[一作]` (North Carolina State University), Tiffany Barnes `[通讯]` (North Carolina State University)

**通讯引用:** 5410 | [OpenAlex ID](https://openalex.org/A5083076004)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了两种交互式工作例子（Buggy 和 Guided）在一个命题逻辑智能辅导系统中，随后在 155 名本科 CS 学生的实验中评估其对学习效果的影响。

**💡 创新点**

创新点：①将 ICAP 学习理论与工作例子结合，提出了调试型（Buggy）和缺失补全型（Guided）工作例子，满足不同先验知识水平的学习者；②首次在 ITS 中使用行为分析（马尔可夫模型）对学习者的交互模式进行细粒度比较。

**🔧 技术方法**

技术与方法：ICAP 框架、智能辅导系统实现、工作例子设计、即时反馈与提示、行为日志记录、混合效应回归、Mann‑Whitney U 检验、Bonferroni 校正、Markov 过程行为分析。

**📊 数据集**

数据集与实验对象：155 名北卡罗来纳州立大学离散数学课程的本科 CS 学生；实验包含预试、训练（5 级）和后测（6 题）三个阶段，收集点击流、分数、时间、方案长度等日志数据。

**📈 对比分析**

比较方法与结果：与传统的被动工作例子（WE）和问题求解（PS）对照组相比，Buggy 与 Guided 组在后测问题分数、规则准确率和解题时间均显著提升（p < .05）。Guided 组对低先验知识学生提升约 6% 规则准确率、时间缩短 30%；Buggy 组对高先验知识学生提升约 8% 规则准确率、时间缩短 40%。总体提升约 4.5 分，方案长度无显著差异。

**⚠️ 局限性**

局限性：①仅使用定量日志缺乏情感、动机和环境等质性信息；②Buggy 组未提供提示导致部分学生出现过度挣扎；③先验知识分组基于预试的静态划分，未动态跟踪学习进度；④实验仅在命题逻辑领域进行，泛化性待验证；⑤未评估长期迁移或跨域效果。

---

## 102. Speech to Speech Synthesis for Voice Impersonation

**arXiv ID:** 2602.16721 | [PDF](https://arxiv.org/pdf/2602.16721v1)

**作者:** Bjorn Johnson `[一作]` (University of California San Diego), Jared Levy `[通讯]` (University of California San Diego)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过将预训练的语音识别模型DeepSpeech与文本到语音合成模型Tacotron2串联，并在解码阶段加入目标说话人风格编码，构建了一种语音到语音合成网络，实现源音频内容在保持不变的同时转换为目标说话人的音色与语调。

**💡 创新点**

创新点在于将完整的识别+合成两大体系进行耦合，利用说话人嵌入在生成端注入风格信息，从而在保持语义内容的同时实现跨说话人语音风格迁移，并通过与CycleGAN的对比验证了该方法在自然度和风格一致性方面的优势。

**🔧 技术方法**

采用DeepSpeech（CTC损失训练）提取文本序列，Tacotron2（位置敏感注意力与多层卷积+LSTM）生成声谱图，使用说话人嵌入网络（LSTM+投影到256维）提供风格向量；对比实验中使用CycleGAN（生成对抗+循环一致性损失）在音频谱图上进行风格转换。

**📊 数据集**

STSSN使用LibriSpeech数据集的train-clean-100（约100小时）进行训练，验证使用test-clean（约5.4小时）；CycleGAN的基准实验则使用VCC 2016数据集（1620训练样本、108验证样本），在音频预处理时提取F0、谱包络等特征。

**📈 对比分析**

通过MOS（主观平均意见评分）对比，STSSN在生成语音的自然度和与目标风格的匹配度上均优于CycleGAN；实验还显示STSSN在训练时间上更高效，且在噪声鲁棒性上表现更好。

**⚠️ 局限性**

主要局限包括：模型分离导致文本瓶颈，限制了信息流通；对噪声的鲁棒性仍不够理想；模型容量受限，难以处理更复杂的说话人差异；未实现真正的端到端训练，增加了实现难度和资源需求。

---

## 103. Quantifying LLM Attention-Head Stability: Implications for Circuit Universality

**arXiv ID:** 2602.16740 | [PDF](https://arxiv.org/pdf/2602.16740v1)

**作者:** Karan Bali `[一作]` (Mila - Quebec Artificial Intelligence Institute), Danilo Bzdok `[通讯]` (Mila - Quebec Artificial Intelligence Institute)

**通讯引用:** 20618 | [OpenAlex ID](https://openalex.org/A5051587217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究了Transformer模型在不同随机初始化下的注意力头稳定性，探讨了其对机制可解释性和安全性的影响。

**💡 创新点**

创新点在于首次量化并比较多层Transformer模型的注意力头跨实例稳定性，发现中层头最不稳定且最具代表性，并证明使用AdamW权重衰减显著提升稳定性。

**🔧 技术方法**

采用注意力分数矩阵的余弦相似度、CKA、meta‑SNE等技术对头和残差流进行对比与可视化，并通过后续消融评估功能重要性。

**📊 数据集**

在OpenWebText和C4数据集上训练多种GPT‑style架构（2、4、8、12层），并对多种head配置（8或16头）进行实验。

**📈 对比分析**

方法通过计算跨seed的注意力头相似度和残差流CKA相似度进行比较，结果显示深层中层头稳定性最低但功能影响最大，AdamW显著提升稳定性而不降低性能。

**⚠️ 局限性**

局限性包括仅考虑有限的超参数（如优化器、权重衰减），未覆盖更广泛的训练设定和模型规模，可能影响稳定性分析的普适性。

---

## 104. DODO: Discrete OCR Diffusion Models

**arXiv ID:** 2602.16872 | [PDF](https://arxiv.org/pdf/2602.16872v1)

**作者:** Sean Man `[一作]` (Technion - Israel Institute of Technology), Niv Nayman `[通讯]` (Amazon Web Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于块离散扩散模型的 OCR 系统 DODO，能够实现并行生成文本。

**💡 创新点**

创新点在于将 OCR 任务拆分为有序块，消除全局同步误差，并引入 KV 缓存实现高吞吐。

**🔧 技术方法**

使用了块离散扩散模型、KV 缓存、视觉-语言模型 Qwen2.5-VL 架构，并结合遮掩扩散训练。

**📊 数据集**

使用了约 270K 的 PDF 文档文本对的 OCR 数据集，并在 OmniDocBench 与 Fox-Pages 等基准上评测。

**📈 对比分析**

与多种专用 OCR、AR VLM 与扩散 VLM 对比，DODO 在准确率上接近或超过专用模型，推理速度比 AR 模型快 2-3 倍。

**⚠️ 局限性**

局限在于精度与速度的权衡，块大小与注意力掩码对性能敏感，且在严格缓存下精度略降。

---

## 105. Analytic Score Optimization for Multi Dimension Video Quality Assessment

**arXiv ID:** 2602.16856 | [PDF](https://arxiv.org/pdf/2602.16856v1)

**作者:** Boda Lin `[一作]` (Kuaishou Technology), Pengfei Wan `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了大型多维度视频质量评估数据集 UltraVQA，并基于此设计了 Analytic Score Optimization（ASO）方法，以对视频质量进行多维度分数预测；

**💡 创新点**

创新点包括①将视频质量拆解为 Motion Quality、Motion Amplitude、Aesthetic Quality、Content Quality、Clarity Quality 五个可解释维度并辅以细粒度标签；②使用 GPT‑4.1 合成基于多评审结果的解释性理由，提升模型可解释性；③提出 ASO，通过 KL 正则化的单步 bandit 目标得到闭式最优分数分布，避免传统 RL 的高方差与过拟合；

**🔧 技术方法**

技术手段包括：多模态 Vision‑Language 模型（如 Qwen2.5‑VL‑7B）的监督微调（SFT）；基于 GRPO 的强化学习对齐基线；ASO 的闭式软目标训练（交叉熵 + soft target）；奖励设计为准确奖励 + 分布奖励；数据预处理与评估指标（Acc@0.5、SRCC、PLCC、MAE）；

**📊 数据集**

使用 UltraVQA 数据集（约 40k UGC 视频，3 名评审以上，5 维度评分+细粒度标签+GPT 合成理由）；并在 LSVQ、KoNViD‑1k、VideoPhy2、MJ‑Video 等公开基准上进行交叉验证；

**📈 对比分析**

与 GPT‑4.1、Gemini‑Pro 等闭源 API、Qwen2.5‑VL、MiniCPMV‑4.5、VideoLLaMA3 等开源通用 VLM，以及 FineVQ、Q‑Align、VideoScoreV2 等专用 VQA 模型进行对比。结果显示 UltraVQA‑ASO 在所有 5 维度上均实现了最高 SRCC/PLCC、最低 MAE，且在跨基准泛化上优于通用 VLM，性能接近或超过专用 VQA 模型；

**⚠️ 局限性**

局限性包括：仅覆盖 5 个维度，可能忽略其他质量维度；依赖 GPT‑4.1 合成理由，存在潜在 hallucination 风险；数据集主要集中于 UGC，专业视频或非中文内容的泛化仍待验证；ASO 仍需要基准分数和奖励函数的设计，对极端稀疏奖励场景可能表现不佳。

---

## 106. Is Mamba Reliable for Medical Imaging?

**arXiv ID:** 2602.16723 | [PDF](https://arxiv.org/pdf/2602.16723v1)

**作者:** Banafsheh Saber Latibari `[一作]` (University of Arizona), Abhijit Mahalanobis `[通讯]` (University of Arizona)

**通讯引用:** 2149 | [OpenAlex ID](https://openalex.org/A5055793435)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估了Mamba在医疗影像分类任务中对输入级扰动（白盒FGSM/PGD、PatchDrop、Gaussian噪声与模糊）以及硬件级位翻转攻击的鲁棒性。

**💡 创新点**

提出了Med-Mamba-Hammer框架，系统化地评估随机、层级和最坏情况位翻转对Mamba模型的影响，并量化其安全缺陷。

**🔧 技术方法**

采用了白盒梯度攻击（FGSM、PGD）、信息缺失攻击（PatchDrop）、常见图像失真（高斯噪声、失焦模糊）以及软件层面的位翻转注入（随机、层级、最坏搜索）。

**📊 数据集**

实验使用MedMNIST系列低分辨率医学影像分类数据集（PathMNIST、DermaMNIST、OCTMNIST、PneumoniaMNIST、RetinaMNIST、BreastMNIST、BloodMNIST、OrganAMNIST、OrganCMNIST、OrganSMNIST）。

**📈 对比分析**

结果显示PGD可将多数数据集准确率降至接近随机水平，PatchDrop在高比例遮挡时也显著下降；仅一位指数位翻转即可将BloodMNIST准确率从97.6%降至9.1%，表明模型对硬件级错误极为脆弱。

**⚠️ 局限性**

仅在低分辨率单任务数据集上验证，未涉及高分辨率临床影像、分割或检测任务，且硬件攻击仅在软件模拟层面完成，缺乏真实设备级验证，易受实际硬件噪声和热漂移等因素影响。

---

## 107. Guiding LLM-Based Human Mobility Simulation with Mobility Measures from Shared Data

**arXiv ID:** 2602.16726 | [PDF](https://arxiv.org/pdf/2602.16726v1)

**作者:** Hua Yan `[一作]` (Lehigh University), Yu Yang `[通讯]` (Lehigh University)

**通讯引用:** 1807 | [OpenAlex ID](https://openalex.org/A5070570645)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于大语言模型的多提示调整框架，通过利用共享数据的移动测度指导个体轨迹生成，以实现大规模人类移动模拟的群体级协调。

**💡 创新点**

创新点在于将群体层面的移动测度（如半径、停留时长、访问频率）作为优化目标，通过多目标 MDP 与 MCTS 在提示空间中搜索，自动生成多种个体化提示，实现跨尺度的行为协调。

**🔧 技术方法**

使用大语言模型（GPT‑5.2）进行轨迹生成，结合蒙特卡洛树搜索、全局价值估计、梯度式奖励函数、以及多目标几何平均评价。

**📊 数据集**

采用北京和纽约市的公开移动轨迹数据（含用户画像），并将用户画像合成自美国人口普查统计，用于训练与评估。

**📈 对比分析**

与 CoPB、UML、LLMob、CitySim 等基线相比，在空间、时间、时空等多项指标上提升 11–64% 以上，尤其在旅行距离、OD 相似度、探索指数等方面表现最显著。

**⚠️ 局限性**

主要限制是计算开销较大，尤其在大规模模拟时需要大量 LLM 调用；实验仅覆盖两座城市，未验证到更广泛的城市环境；且依赖共享数据的可获得性。

---

## 108. DeepVision-103K: A Visually Diverse, Broad-Coverage, and Verifiable Mathematical Dataset for Multimodal Reasoning

**arXiv ID:** 2602.16742 | [PDF](https://arxiv.org/pdf/2602.16742v1)

**作者:** Haoxiang Sun `[一作]` (Alibaba Group), Hu Wei `[通讯]` (Alibaba Group)

**通讯引用:** 727 | [OpenAlex ID](https://openalex.org/A5101855580)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并利用 DeepVision-103K 数据集，对大型多模态模型（LMM）进行可验证奖励的强化学习（RLVR），提升其视觉感知、反思与推理能力。

**💡 创新点**

① 通过三阶段自动化过滤（有效性筛选、难度校准、查询正确性验证）从海量 K12 真实数据中挖掘 77K 条可验证 QA 对；② 将多模态数学与视觉逻辑任务混合，证明两类数据互补；③ 对比实验展示查询正确性验证对 RLVR 成效的关键作用。

**🔧 技术方法**

强化学习（RLVR）采用 GSPO；模型使用 Qwen3‑VL‑8B、MiMo‑VL‑7B 等具备思维能力的 LMM；数据处理利用 GPT‑5‑VL‑32B、Qwen3‑VL、Gemini‑3‑Flash 等大模型做过滤与校验；评估使用 Pass@1 在多模态数学与通用多模态基准上。

**📊 数据集**

从 MM‑MathInstruct‑3M、MultiMath‑300K 等公开 K12 训练语料中采样，并结合自身构建的可验证 QA 对，最终得到 DeepVision-103K，涵盖 6 类视觉元素、200+ 细粒度知识点和 4 大数学分支。

**📈 对比分析**

与闭源模型（GPT‑5‑Nano‑High、Gemini‑2.5‑Flash‑Lite）、官方思维版本（Qwen3‑VL‑8B‑Thinking、MiMo‑VL‑7B‑RL‑2508）以及开源数据集（MM‑Eureka、MathBook、OpenMMReasoner）进行公平对比；在 WeMath、MathVerse_vision、MathVision、LogicVista 等多模态数学基准上，DeepVision‑103K 的模型实现了 2.9–8.6% 的提升，甚至在部分任务达到了 SOTA；在 MMMU、M³CoT 等通用多模态基准上也表现出显著的泛化优势。

**⚠️ 局限性**

1) 数据分布仍偏向平面几何，稀有视觉元素不足；2) 过滤流程依赖高成本外部大模型，可能引入偏差并误删部分有效样本；3) 只覆盖 K12 单一答案的任务，未涵盖开放式推理或多答案场景。

---

## 109. WS-GRPO: Weakly-Supervised Group-Relative Policy Optimization for Rollout-Efficient Reasoning

**arXiv ID:** 2602.17025 | [PDF](https://arxiv.org/pdf/2602.17025v1)

**作者:** Gagan Mundada `[一作]` (University of California, San Diego), Junda Wu `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种弱监督的GRPO方法WS-GRPO，利用终端正确性信息训练偏好模型，再将其转化为前缀级别的奖励以减少冗余推理

**💡 创新点**

创新点在于将终端标签转化为前缀级别的弱监督奖励，避免全局长度惩罚并兼顾推理效率与准确性

**🔧 技术方法**

采用偏好模型（FLAN‑T5+MLP）与GRPO的分组优势框架，结合PPO式截断与KL正则化进行策略优化

**📊 数据集**

在ARC、CommonsenseQA、DeepMath与GSM8K四个多步推理基准上进行实验

**📈 对比分析**

与GRPO、Dr.GRPO等基线相比，WS-GRPO在保持接近或略低准确率的同时，可将响应长度与推理步骤缩短50–90%，显著提升推理效率

**⚠️ 局限性**

局限性包括对数学推理任务仍可能导致准确率下降、对偏好模型误差敏感且在极长推理链中仍需更精细的长度控制

---

## 110. Cinder: A fast and fair matchmaking system

**arXiv ID:** 2602.17015 | [PDF](https://arxiv.org/pdf/2602.17015v1)

**作者:** Saurav Pal `[一作]` `[通讯]` (National Institute of Technology), Saurav Pal (National Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Cinder两阶段匹配系统，先通过非异常范围重叠过滤，再用Wasserstein距离评估公平性。

**💡 创新点**

创新点在于引入Ruzicka相似度快速预筛选和基于反正态分布的非线性技能桶化来提高公平性评估的精度。

**🔧 技术方法**

使用统计学方法（Z-Score、Ruzicka指数、Wasserstein距离）和Python/NumPy等计算工具。

**📊 数据集**

在140,000,000对随机大厅对阵的模拟数据上验证了系统。

**📈 对比分析**

与传统平均/中位数匹配相比，Cinder在保持公平性的同时减少了计算量，阈值设置可实现快速匹配；实验表明大部分随机匹配的Sanction Score集中在20-25区间。

**⚠️ 局限性**

局限在于未在真实环境中测试，且未考虑等待时间、角色偏好等因素；桶分布参数需针对不同游戏调优。

---

## 111. An order-oriented approach to scoring hesitant fuzzy elements

**arXiv ID:** 2602.16827 | [PDF](https://arxiv.org/pdf/2602.16827v1)

**作者:** Luis Merino `[一作]` (University of Granada), Evangelina Santos `[通讯]` (University of Granada)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了一个以顺序为核心的海塞式模糊元素（HFE）评分框架，并在此基础上构造了两类支配函数（DDF与RDF）用于多准则决策；

**💡 创新点**

创新点在于将评分与任意偏序关联，证明对称顺序下的评分满足强单调性和Gärdenfors性质，并首次将支配函数引入HFE的比较与偏好建模；

**🔧 技术方法**

主要技术包括偏序理论、集合运算、均值与几何平均等聚合算子，以及基于顺序的评分与支配函数定义；

**📊 数据集**

论文未使用公开数据集，而是通过理论实例和模拟项目评估示例来演示方法；

**📈 对比分析**

通过DDF和RDF对同一项目集合的排序进行比较，实验显示DDF对未达阈值的惩罚更严厉，RDF则给出更平滑的评估；

**⚠️ 局限性**

局限性包括缺乏大规模实验验证、对非有限（如区间）HFE的推广尚未充分探讨，以及对复杂多维偏序的计算复杂度未评估。

---

## 112. Improved Upper Bounds for Slicing the Hypercube

**arXiv ID:** 2602.16807 | [PDF](https://arxiv.org/pdf/2602.16807v1)

**作者:** Duncan Soiffer `[一作]` (Carnegie Mellon University), Daniel Reichman `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 934 | [OpenAlex ID](https://openalex.org/A5081855980)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明了 n 维超立方体切割问题的新上界：若 n 不是奇数 5 的倍数，则 S(n) ≤ ⌈4n/5⌉，若是，则 S(n) ≤ 4n/5+1，并给出了 Q₁₀ 的 8 平面切割构造。

**💡 创新点**

创新点在于首次通过人工介入结合大语言模型驱动的 CPro1 自动程序生成，设计出适应性边加权局部搜索与列组分配约束的混合算法，突破了 1971 年 Paterson 的 ⌈5n/6⌉ 上界，实现了 10 维超立方体的 8 平面完全切割。

**🔧 技术方法**

采用了大语言模型驱动的 CPro1 自动程序生成、适应性加权的局部搜索、约束化列组分配、降维超立方体映射等技术手段。

**📊 数据集**

使用了 n 维超立方体的顶点集合 {−1,1}ⁿ 以及其所有边集 Eₙ 作为搜索空间；通过实验验证 Q₁₀ 的 5120 条边被完全切割。

**📈 对比分析**

与原先的 Tabu 搜索、OpenEvolve 以及 CPro1 无人干预的结果对比，在 Q₁₀ 上从 5114 条提升至 5120 条，实现全切割；在 Q₁₅、k=12 上也取得 245,748 条切割，显示相同算力下方法更高效、性能显著提升。

**⚠️ 局限性**

局限性在于仅提供了上界改进，未证明更优的线性下界；对更高维度（n>10）仍难以获得完整切割；对 S(n,k) 的下界突破不足，仍需新的理论方法。

---

## 113. AI-Mediated Feedback Improves Student Revisions: A Randomized Trial with FeedbackWriter in a Large Undergraduate Course

**arXiv ID:** 2602.16820 | [PDF](https://arxiv.org/pdf/2602.16820v1)

**作者:** Xinyi Lu `[一作]` (University of Michigan), Xu Wang `[通讯]` (University of Michigan)

**通讯引用:** 97778 | [OpenAlex ID](https://openalex.org/A5100424784)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在一门大型本科经济学课程中，设计并部署了 FeedbackWriter 系统，帮助教学助理（TA）在写作反馈中使用 LLM 生成的建议，并通过随机对照实验（N=354）比较 AI‑mediated 与纯人类反馈对学生写作修订质量和学习效果的影响。

**💡 创新点**

创新点包括：①将 AI 生成的反馈与教师工作流程对齐，提供可视化的句子高亮、评判与建议，并允许 TA 完全控制采纳、编辑或拒绝；②通过对照实验评估 AI‑mediated 反馈在实际课堂中的效果；③系统化分析 TA 与学生对 AI 建议的使用模式与改动，揭示人机协作的最佳实践。

**🔧 技术方法**

核心技术：利用 GPT‑4o（Llama/LLM）进行句子提取、评判与反馈生成；React + Django 前端后端实现；系统日志记录、自动化评分器与自动化反馈质量评估均基于 LLM（GPT‑4/4o）。

**📊 数据集**

数据集：1,366 篇学生提交的知识密集型经济学作业（两次写作任务），354 名学生、11 名 TA，配合课程的标准化评分与历史反馈。研究中还使用了 60 篇人工评估样本做模型校准。

**📈 对比分析**

对照方法：在同一课程中随机分配 TA 与学生到 AI‑mediated 与人类仅反馈两组，使用混合效应回归评估修订质量（Cohen’s d=0.50）和后测成绩。AI‑mediated 组在修订质量、反馈篇幅、覆盖度、行动性、独立学习支持等指标上显著优于基线；但两组在后测学习成绩上差异不显著。

**⚠️ 局限性**

局限性：①未能显著降低 TA 的评分时间；②实验仅涵盖经济学写作，结果对其他学科未知；③对 AI 仅反馈的可行性未直接检验，可能存在误判风险；④依赖 GPT‑4o，模型更新或不同 LLM 可能影响结果；⑤缺乏长期学习成效与教师工作负荷的纵向追踪。

---

## 114. Beyond the Flag: A Framework for Integrating Cybersecurity Competitions into K-12 Education for Cognitive Apprenticeship and Ethical Skill Development

**arXiv ID:** 2602.16921 | [PDF](https://arxiv.org/pdf/2602.16921v1)

**作者:** Tran Duc Le `[一作]` (University of Wisconsin–Stout), Nam Son Nguyen `[通讯]` (Hewlett Packard Enterprise)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出并阐述了基于认知学徒制和伦理教育的ECAC框架，旨在将CTF竞赛有效整合进K-12课程，解决教师缺乏专业知识、资源不平等、学生参与度低和伦理缺失等障碍。

**💡 创新点**

创新点在于将认知学徒制与伦理道德植入同一教学框架，并通过五阶段（建模、搭建竞技场、辅导与表述、伦理困境注入、反思探究）实现低起点、高天花板的分层教学，强调教师角色转变为“学习引导者”，兼顾多样性与包容性。

**🔧 技术方法**

采用系统框架综合（Framework Synthesis）方法，对现有文献进行主题提炼并映射到认知学徒制、游戏化、PBL等理论；并基于框架设计实现分阶段教学与伦理插入。

**📊 数据集**

数据集为25篇通过SCOPUS检索的同行评审论文（以及少量官方教育文献），并未收集实验数据。

**📈 对比分析**

通过与传统CTF模式以及成功的GenCyber项目对照，进行定性比较；文中未给出量化性能指标，但指出该框架可提升学生参与度、技术与伦理素养，并预期在实践中能降低教师专业发展成本。

**⚠️ 局限性**

局限性包括缺乏大规模实证验证、对不同地区与学校资源差异的外推性不确定、以及框架设计主要基于文献综述，可能忽略实践中的细节与动态适配需求。

---

## 115. IndicJR: A Judge-Free Benchmark of Jailbreak Robustness in South Asian Languages

**arXiv ID:** 2602.16832 | [PDF](https://arxiv.org/pdf/2602.16832v1)

**作者:** Priyaranjan Pattnayak `[一作]` (Oracle America Inc), Sanchari Chowdhuri `[通讯]` (Oracle America Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Indic Jailbreak Robustness（IJR）——一个无裁判、双轨（JSON 合同与 FREE 自然）评测框架，用于测量 12 种南亚语言的 LLM 对抗性安全漏洞。

**💡 创新点**

创新点在于：①首次构建全自动、无裁判的评测协议；②同时评估合同约束与自然对话两种轨道；③系统考察脚本转写（原生、罗马化、混合）和跨语言攻击的安全性。

**🔧 技术方法**

采用规则化拒绝检测器、语言感知关键词/模式匹配、自动化评估脚本、跨语言包装与转写生成技术。

**📊 数据集**

使用 45,216 条 JSON 对抗提示和 2,580 条 FREE 对抗提示，来源于维基百科本地文本、模板化有害核心，覆盖原生、罗马化和混合脚本三种转写。

**📈 对比分析**

对 12 款模型（API、开源与印度专用）进行比较；结果显示合同模式下攻击成功率高（多数模型 ≥0.9），罗马化显著降低 JSON 级别 JSR，跨语言攻击强大；FREE 轨道几乎 100% 成功，表明现有模型安全性不足。

**⚠️ 局限性**

限制：仅评估单轮、三类有害意图；使用标准化转写，未覆盖方言、噪声代码混合、多轮交互与提供方安全层。

---

## 116. UPER: Efficient Utility-driven Partially-ordered Episode Rule Mining

**arXiv ID:** 2602.16718 | [PDF](https://arxiv.org/pdf/2602.16718v1)

**作者:** Hong Lin `[一作]` (Jinan University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 133675 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对复杂事件序列的高效实用价值驱动部分有序事件规则挖掘方法UPER，定义了高实用价值部分有序事件规则，并给出了完整的挖掘流程；

**💡 创新点**

创新点包括：①首次将实用价值考虑进部分有序事件规则挖掘；②设计NoList数据结构高效存储非重叠出现信息；③提出窗口估计利用率WEU与规则扩展估计利用率REEU两种上界；④基于上述上界提出WEUP、REUCSP、REEUP三种剪枝策略；

**🔧 技术方法**

采用了深度优先规则扩展、非重叠出现计数、窗口估计利用率、规则扩展估计利用率、NoList存储、队列式候选管理等技术；

**📊 数据集**

使用六个公开真实数据集：Retail、Kosarak、Chainstore、Foodmart、Yoo-choose-buy、Ecommerce Retail；

**📈 对比分析**

通过实现四个UPER变种（UPER_1~UPER_4），在各数据集上比较运行时、候选规则数和内存消耗。实验结果表明，加入REUCSP和REEUP剪枝后，UPER_3/UPER_4在多数数据集上显著降低了运行时间（几倍到十倍）和内存占用，且候选规则数减少十倍以上；

**⚠️ 局限性**

局限性包括：①对大规模时间窗口时REEU上界过大，导致REEUP剪枝效果不明显；②NoList存储和规则扩展需要维护大量状态，内存占用仍较高；③未提供基线对比，仅通过内部变体评估，缺乏与其他实用价值挖掘算法的外部对比；

---

## 117. "Hello, I'm Delivering. Let Me Pass By": Navigating Public Pathways with Walk-along with Robots in Crowded City Streets

**arXiv ID:** 2602.16861 | [PDF](https://arxiv.org/pdf/2602.16861v1)

**作者:** EunJeong Cheon `[一作]` (Syracuse University), Do Yeon Shin `[通讯]` (University of Illinois)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实证验证了‘Walk‑Along with Robots (WawR)’方法，进行首尔两地配送机器人在公共空间中的走访、观察、访谈与情境记录，探讨机器人与人、环境、时间的互动；

**💡 创新点**

将机器人视为研究主体，采用更‑人类视角与移动民族志技术，整合路线绘制、自动民族志、现场访谈等多方法，解决传统HRI观察局限；

**🔧 技术方法**

主要运用民族志与移动研究技术：地图应用记录路径、机器人服务App信息、现场笔记、访谈记录等；

**📊 数据集**

使用现场收集的观察日志、访谈文本、路线地图、机器人行驶轨迹等数据；未使用公开数据集；

**📈 对比分析**

作为方法论研究，没有对照实验；与传统断片式观察相比，全天连续跟踪提供更完整的时间、空间与社交维度数据，显著提升信息深度与覆盖度；

**⚠️ 局限性**

样本仅限首尔两地点，文化特定导致交互可能受限；需要团队协作与深度文化理解，且缺乏量化评估，方法对极端拥挤或多元文化环境可能需调整；

---

## 118. Sales Research Agent and Sales Research Bench

**arXiv ID:** 2602.17017 | [PDF](https://arxiv.org/pdf/2602.17017v1)

**作者:** Deepanjan Bhol `[一作]` `[通讯]` (Microsoft), Deepanjan Bhol (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出并评估了微软 Dynamics 365 Sales 中的 Sales Research Agent，旨在通过多智能体协同与多模型支持实现对企业 CRM 与预算等数据的深度查询与可视化分析。

**💡 创新点**

核心创新包括：① 为销售领域量身定制的多智能体架构与多模型切换机制；② 针对业务语言的自适应推理与模式化；③ 对企业自定义模式的 Schema 智能识别与自校正；④ 通过 LLM 判断的八维度评估框架——Sales Research Bench。

**🔧 技术方法**

使用技术涵盖：多智能体协同框架、LLM 推理模型（GPT‑4.1、Claude 4.5 等）、SQL/Python 代码生成与自校正、数据可视化生成与解释、LLM 评估器（Azure Foundry、OpenAI GPT‑4.1）。

**📊 数据集**

评估数据集为：200 条真实销售业务问题、基于企业自定义 Schema 的 Azure SQL 数据副本以及相应的标准答案，用以检验模型在不同 Schema 下的表现。

**📈 对比分析**

方法通过 LLM 判定器按权重（文本、图表可信度、相关性、可解释性、Schema 准确度等）对三款系统（Sales Research Agent、ChatGPT‑5、Claude Sonnet 4.5）进行统一评测；结果显示 Agent 在所有八维度均优于对手，综合得分 78.2 远高于 Claude 65.2 与 ChatGPT 54.1。

**⚠️ 局限性**

局限性包括：评测场景仍基于单一企业自定义 Schema；评估主要依赖 LLM 判定，可能存在主观性；未覆盖更广泛的业务领域与多样化数据源。

---

## 119. The Compute ICE-AGE: Invariant Compute Envelope under Addressable Graph Evolution

**arXiv ID:** 2602.16736 | [PDF](https://arxiv.org/pdf/2602.16736v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 120. Position: Why a Dynamical Systems Perspective is Needed to Advance Time Series Modeling

**arXiv ID:** 2602.16864 | [PDF](https://arxiv.org/pdf/2602.16864v1)

**作者:** Daniel Durstewitz `[一作]` (Central Institute of Mental Health), Lukas Eisenmann `[通讯]` (Heidelberg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统综述动力学系统理论及其重建方法，提出并论证将DS视角引入时间序列建模可以显著提升预测性能，并给出了具体的改进建议。

**💡 创新点**

创新点在于将DS重建（DSR）技术与时间序列模型的训练方法结合，展示DSR模型在长期统计特征甚至短期预测上优于传统TS模型，并提出使用DS仿真数据预训练、STF/GTF等训练技巧。

**🔧 技术方法**

主要技术包括RNN与PLRNN等DSR模型、Transformer-based TS模型（Chronos、Panda等）、Lyapunov谱、吸引子几何度量、STF/GTF训练策略以及DynaMix等。

**📊 数据集**

使用的数据集包括能源、天气、交通、fMRI和EEG等真实世界时间序列，以及基于动力学系统仿真的合成数据。

**📈 对比分析**

通过对短期误差(MASE)与长期几何/时间度量(D_stsp、D_H)的双重评估，实验证明DSR模型在长期行为上优于TS模型，甚至在某些数据集上短期性能也更好；DynaMix在零样本场景下表现突出。

**⚠️ 局限性**

主要局限包括DSR假设的低维、确定性、自主性难以直接迁移到高维噪声、非平稳的实测序列，Transformer在时间动态建模上的不足，以及在处理拓扑转移和极端事件时仍面临挑战。

---

## 121. AdaptOrch: Task-Adaptive Multi-Agent Orchestration in the Era of LLM Performance Convergence

**arXiv ID:** 2602.16873 | [PDF](https://arxiv.org/pdf/2602.16873v1)

**作者:** Geunbin Yu `[一作]` `[通讯]` (Korea National Open University), Geunbin Yu (Korea National Open University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向任务的多代理调度框架，动态根据任务的依赖图选择并行、顺序、层级或混合四种拓扑，并自适应合成最终输出。

**💡 创新点**

1) 在模型能力收敛时证明拓扑选择主导系统性能的收敛标度律；2) 提供线性时间的拓扑路由算法；3) 设计终止保证的自适应合成协议。

**🔧 技术方法**

使用LLM分解器构造任务DAG，基于阈值的路由算法决定拓扑，配合并行/顺序/层级/混合执行器和基于嵌入相似度的合成一致性评估，重路由机制保证收敛。

**📊 数据集**

SWE-bench（代码修复）、GPQA Diamond（研究推理）和HotpotQA（检索增强问答）三大基准。

**📈 对比分析**

与单一最佳模型、MoA-3L、静态并行/顺序、LLM-Blender以及自适应模型一致性Self‑MoA（matched）等方法对比，在三大基准上实现12–23%的准确率提升，且在准确率/令牌效率与延迟方面均优于基线。

**⚠️ 局限性**

依赖解构质量、耦合估计粗糙、并行执行受限于并发调用额度，实验仅覆盖代码、推理、检索三类任务，需进一步验证对创作、长文本、多模态等场景的泛化。

---

## 122. Xray-Visual Models: Scaling Vision models on Industry Scale Data

**arXiv ID:** 2602.16918 | [PDF](https://arxiv.org/pdf/2602.16918v1)

**作者:** Shlok Mishra `[一作]` (Meta AI), Aashu Singh `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在社交媒体海量图像和视频上训练统一视觉模型，并通过三阶段（MAE、hashtag 分类、CLIP 对比）提升跨模态表示。

**💡 创新点**

① 大规模 10B+ 图像‑文本、视频‑标签数据集；② 统一 ViT+3D 令牌化与 EViT‑2b 结构；③ LLM（LLaMA‑1B）文本编码器、ToMe 令牌合并、分辨率递增训练；④ 多任务训练与半监督标签。

**🔧 技术方法**

Mask‑AutoEncoder、Hashtag 监督、CLIP 对比、SLIP、SimCLR、denoising loss、Lion Optimizer、LLM2CLIP、ToMe、Token Merging、Progressive Resolution、Variable Aspect Ratio、Knowledge Distillation、Linear Adapter、Quantization、Semantic ID 量化。

**📊 数据集**

ViSE（10B 图像‑文本对）、URU（10B 视频‑标签对）、MetaCLIP、ImageNet、Kinetics‑700、MS‑COCO、MS‑RVTT、ObjectNet、ImageNet‑Sketch、CIFAR、内部广告与检索数据集。

**📈 对比分析**

与 Perception Encoder、SigLIP、DiNO、CLIP 等前沿模型对比；在 ImageNet 线性探测上 89.3%（SOTA），Kinetics‑700 78.1% Top‑1，MS‑COCO/ MS‑RVTT 检索也处于 SOTA；相较传统模型使用 25% 令牌、336 分辨率，效率提升 4×，成本下降 84%。

**⚠️ 局限性**

① 对低分辨率或极端 OOD 数据（CIFAR、部分艺术类）性能仍不及 SOTA；② synthetic caption 过度依赖 LLM 可能导致动作理解下降；③ 需要大量算力与 10B+ 数据；④ 内部指标与学术指标不完全一致，需多维评估。

---

## 123. Learning under noisy supervision is governed by a feedback-truth gap

**arXiv ID:** 2602.16829 | [PDF](https://arxiv.org/pdf/2602.16829v1)

**作者:** Elan Schonfeld `[一作]` (Columbia University), Elias Wisnia `[通讯]` (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

我们在人工神经网络、人类概率逆转学习和 EEG 实验中探究了反馈与真理之间的差距，并证明其在反馈整合速度快于真理评估时不可避免。

**💡 创新点**

创新之处在于提出统一的反馈–真理差距框架和两时尺度模型，并跨 AI 与人类系统验证其不可避免性及不同调节机制。

**🔧 技术方法**

采用两时尺度学习模型、稠密与稀疏残差神经网络、概率两臂赌博任务以及基于逻辑回归的 EEG 反馈与期望解码等技术。

**📊 数据集**

使用 30 个带标签噪声的表格数据集（如 Ionosphere、Glass 等）、292 名参与者的逆转学习数据以及 25 名参与者的奖励/惩罚任务伴 EEG 数据。

**📈 对比分析**

通过计算 gap 指标比较三种系统；在网络中 gap 与验证准确率呈负相关，在人类中较大 gap 预测更快恢复，而 EEG gap 与行为承诺相关；稀疏残差架构将 gap 减少并在 40% 噪声下提升 93% 的准确率。

**⚠️ 局限性**

局限性包括人类 EEG 数据仅为相关性分析、药理学验证为初步、实验仅限于实验室任务，且两时尺度模型未涵盖所有调节机制。

---

## 124. ConvApparel: A Benchmark Dataset and Validation Framework for User Simulators in Conversational Recommenders

**arXiv ID:** 2602.16938 | [PDF](https://arxiv.org/pdf/2602.16938v1)

**作者:** Ofer Meshi `[一作]` (Google), Craig Boutilier `[通讯]` (Google)

**通讯引用:** 17151 | [OpenAlex ID](https://openalex.org/A5036934218)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ConvApparel数据集和一种综合评估框架，用于评估LLM驱动的用户模拟器，并对三种模拟器（Prompted、ICL、SFT）进行了实验比较。

**💡 创新点**

创新点在于：①采用双代理数据采集协议，生成“好”与“坏”推荐器的对照对话，从而实现反事实验证；②构建三支柱评估框架，融合统计对齐、人类相似度评分与反事实验证；③首次将判别器训练为人类相似度得分，全面衡量模拟器的真实性。

**🔧 技术方法**

使用技术包括：LLM基用户模拟器（提示式、上下文学习、监督微调）、LLM-as-a-Judge用于自动抽取对话统计与主观指标、判别器（基于Gemini）用于计算人类相似度评分、统计分布对齐和反事实验证方法。

**📊 数据集**

使用的数据集为ConvApparel（约4,146条人机购物对话，包含首人视角满意度、沮丧等自评），以及由三种模拟器生成的相应对话作为对照。

**📈 对比分析**

通过人口层面统计对齐、判别器得分和反事实验证三种方式对比；实验结果显示ICL和SFT在统计对齐和反事实泛化上均优于Prompted，但所有模拟器的判别器得分均接近0，表明仍存在显著的真实性缺口。

**⚠️ 局限性**

局限性包括：①反事实验证仅覆盖从好代理到坏代理的单一转变；②数据集仅限于服装购物领域，缺乏跨域泛化；③对话仅文本化，未覆盖点击、视觉交互等多模态行为；④LLM-as-a-Judge与人类评估的相关性有限，易放大差异；⑤尽管统计对齐较好，判别器仍能轻易识别模拟对话，说明真实性差距依旧显著。

---

## 125. Retrieval Augmented (Knowledge Graph), and Large Language Model-Driven Design Structure Matrix (DSM) Generation of Cyber-Physical Systems

**arXiv ID:** 2602.16715 | [PDF](https://arxiv.org/pdf/2602.16715v1)

**作者:** H. Sinan Bank `[一作]`, Daniel R. Herber `[通讯]` (Colorado State University)

**通讯引用:** 958 | [OpenAlex ID](https://openalex.org/A5077474480)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用大规模语言模型（LLM）、检索增强生成（RAG）和基于图的RAG（GraphRAG）三种方法，对电动螺丝刀和CubeSat两种已知架构的系统生成设计结构矩阵（DSM），并评估其在组件关系判定与组件识别+关系判定两项任务中的表现。

**💡 创新点**

创新点在于将RAG与GraphRAG相结合，利用知识图谱结构化检索，并通过三步流程（参考准备、生成、评估）实现了对DSM的自动化生成，且提出了针对不同关系类型（空间相邻、整体-部分）的专门评估指标。

**🔧 技术方法**

采用的技术包括：Transformer‑based LLM（如mixtral:8x22b、llama3.3:70b、gpt‑4‑turbo‑preview）、检索增强（对文献片段检索）以及GraphRAG（将检索文档构建为实体关系图并进行社区划分后再生成答案），同时使用Prompt Engineering、cosine‑similarity 过滤与Validator 纠错。

**📊 数据集**

数据集来源于公开的技术文献与案例库，具体包括Google Scholar/Semantic Scholar/Scopus检索的高引用学术论文（R1）、教材/标准书籍（R2）以及Instructables/Hackaday等众包项目文档（R3），并使用已公开的电动螺丝刀和CubeSat的标准DSM作为基准。

**📈 对比分析**

在两种用例中，模型性能通过细胞级的准确率、精确率、召回率、F1‑score以及全局的编辑距离和谱距离进行评估。结果显示，mixtral:8x22b 在螺丝刀空间关系判定中取得最高的 F1≈0.85，llama3.3:70b 在 CubeSat 整体-部分关系判定中达到 F1≈0.91；GraphRAG 在 R2–R3 文档组合下进一步提升了结构相似度，编辑距离显著下降。

**⚠️ 局限性**

主要限制包括：模型对提示（prompt）高度敏感，导致不同提示会显著影响结果；检索文档选择与分类需人工介入或细粒度自动化，过多或不相关文档会导致性能波动；GraphRAG 的图构建和索引消耗较高的计算资源；以及在未见过的系统或关系类型时的泛化能力仍待验证。

---

## 126. How AI Coding Agents Communicate: A Study of Pull Request Description Characteristics and Human Review Responses

**arXiv ID:** 2602.17084 | [PDF](https://arxiv.org/pdf/2602.17084v1)

**作者:** Kan Watanabe `[一作]` (Nara Institute of Science and Technology), Kenichi Matsumoto `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 7076 | [OpenAlex ID](https://openalex.org/A5011588138)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 5 种 AI 编码代理产生的 33,596 条 Pull Request 进行大规模实证分析，比较其 PR 描述特征与人类审阅者的交互与最终合并结果。

**💡 创新点**

首次系统量化 AI 代理 PR 描述风格与审阅行为之间的关联，揭示结构化描述可提升审阅效率，但仍需结合代码质量来决定合并概率。

**🔧 技术方法**

采用正则表达式提取 PR 描述的 11 个特征，并使用 Z‑score 归一化；利用 RoBERTa 微调模型进行情感分类；统计检验包括 Kruskal‑Wallis 与 Chi‑square。

**📊 数据集**

使用 AIDev 数据集（GitHub PR、评论、审阅记录）中的 33,596 条 PR 以及 28,961 条经过过滤的评论。

**📈 对比分析**

通过特征均值热图、交互指标（comments_per_pr、comment_length、time_to_first_comment、sentiment）和结果指标（merge_rate、time_to_completion）进行比较。结果显示：OpenAI Codex 最高合并率（82.6%）且最短完成时间（0.02 h），Cursor 次之（65.22%/0.90 h），GitHub Copilot 低合并率（43.0%）且完成时间最长（13 h）。

**⚠️ 局限性**

研究为观察性，可能受任务类型、仓库规范、代码复杂度等混杂因素影响；PR 描述特征提取的自动化方法可能不够精确；情感分析仅限英文评论，可能产生偏差；结果仅基于 AIDev 数据集，外部可推广性有限。

---

## 127. Discovering Universal Activation Directions for PII Leakage in Language Models

**arXiv ID:** 2602.16980 | [PDF](https://arxiv.org/pdf/2602.16980v1)

**作者:** Leo Marchyok `[一作]` (Oregon State University), Sanghyun Hong `[通讯]` (Oregon State University)

**通讯引用:** 719 | [OpenAlex ID](https://openalex.org/A5102751625)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型内部结构对个人身份信息（PII）泄露的影响，提出了一种在推理时通过加入通用激活方向来放大PII泄露的框架；

**💡 创新点**

创新点在于：①引入UniLeak方法，利用模型自生成文本和梯度优化，在无需真实PII或训练数据的前提下挖掘低维通用激活方向；②展示该方向既可用于嵌入污染攻击也可用于推理时抑制，实现对PII泄露的双向控制；

**🔧 技术方法**

采用的技术包括激活干预（activation steering）、梯度优化、Logit Lens、Direct Logit Attribution、表征相似度分析以及自生成训练数据集构建；

**📊 数据集**

使用的数据集为Enron邮件集和TREC邮件集，涵盖邮箱、电话号码、个人姓名等三类PII；

**📈 对比分析**

在GPT‑Neo、PHI‑2、LLaMA‑3等模型上与BOS、Private Investigator等基线提取攻击对比，UniLeak+基线可提升至多13,399条PII记录，覆盖率显著提升；嵌入污染攻击可再提取约199条PII，推理时抑制可减少9–3,562条PII记录，且对生成质量影响有限；

**⚠️ 局限性**

局限性包括：仅在公开开源模型上验证，缺乏对专有模型或更大规模模型的评估；对训练数据分布漂移的鲁棒性未知；推理时抑制虽有效但可能在极端提示下略降质量；自生成训练数据集规模大、梯度优化成本相对较高；

---

## 128. LLM4Cov: Execution-Aware Agentic Learning for High-coverage Testbench Generation

**arXiv ID:** 2602.16953 | [PDF](https://arxiv.org/pdf/2602.16953v1)

**作者:** Hejia Zhang `[一作]` (University of California San Diego), Jishen Zhao `[通讯]` (University of California San Diego)

**通讯引用:** 5147 | [OpenAlex ID](https://openalex.org/A5077387335)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LLM4Cov框架，利用离线执行反馈训练LLM代理，实现硬件验证的高覆盖率测试基准生成。

**💡 创新点**

创新点在于将验证建模为确定性、无记忆的状态转移；提出覆盖引导的拒绝微调、worst‑state优先采样以及验证条件进阶训练，以在有限仿真预算下最大化监督效用。

**🔧 技术方法**

采用LLM推理、离线数据合成、覆盖反馈过滤、分阶段监督、状态分布对齐等技术；核心模型为Qwen3‑4B。

**📊 数据集**

使用CodeV‑R1（87k硬件仓库）生成合成数据，并构造CVDP‑ECov基准（83个硬件仓库）进行评测。

**📈 对比分析**

在Agentic评估（多轮迭代）下，4B模型实现69.2% Pass、90.4% Avg Cov，显著优于30B教师模型5.3%并接近50‑100×参数模型的性能。

**⚠️ 局限性**

限制包括对昂贵仿真调用的依赖、对多步状态依赖处理不足，以及在工业真实流水线中的进一步验证需要。

---

## 129. APEX-SQL: Talking to the data via Agentic Exploration for Text-to-SQL

**arXiv ID:** 2602.16720 | [PDF](https://arxiv.org/pdf/2602.16720v1)

**作者:** Bowen Cao `[一作]` (Chinese University of Hong Kong), Wai Lam `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 7415 | [OpenAlex ID](https://openalex.org/A5018582154)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Agentic Text-to-SQL 框架，利用假设-验证循环在 Schema Linking 和 SQL Generation 阶段主动探索数据库，提升企业级文本到 SQL 的准确性。

**💡 创新点**

创新点在于将静态模式替换为 Agentic 探索，结合逻辑规划、双路径裁剪、并行数据剖析和确定性引导，实现对数据的实时验证，从而大幅提升执行准确率。

**🔧 技术方法**

使用 LLM（DeepSeek‑V3.2、GPT‑4.1 等）+ 逻辑规划 + 双路径裁剪 + 并行数据剖析 + 确定性检索 + Agentic 探索循环 + 全局综合。

**📊 数据集**

实验数据集为 BIRD‑Dev、BIRD Full、Spider 2.0‑Snow。

**📈 对比分析**

与多种基线（OpenSearch‑SQL、RSL‑SQL、DSR‑SQL、AutoLink 等）对比，BIRD 上实现 70.7% 执行准确率，Spider 2.0‑Snow 上实现 51.0%，Pass@8 达 70.2%，相较基线提升 15–30%。

**⚠️ 局限性**

局限在于仍依赖大模型推理、计算量高，且对极大数据库的实时探索仍有性能瓶颈，对低能力版本的提升有限。

---

## 130. Fuse3D: Generating 3D Assets Controlled by Multi-Image Fusion

**arXiv ID:** 2602.17040 | [PDF](https://arxiv.org/pdf/2602.17040v1)

**作者:** Xuancheng Jin `[一作]` (Zhejiang University), Yuchi Huo `[通讯]` (Zhejiang University)

**通讯引用:** 24206 | [OpenAlex ID](https://openalex.org/A5100726041)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种可通过多张带有用户选定区域的条件图像生成可控 3D 资产的方法。

**💡 创新点**

创新点在于：① 多条件融合模块（MCFM）将多张图像区域的视觉特征在保持全局上下文的前提下合并为统一的条件 token；② 3D 语义感知对齐策略自动建立 2D‑>3D 对应关系；③ 本地注意力增强策略通过可调 λ 解决不同条件之间的冲突，提升局部细节控制。

**🔧 技术方法**

核心技术包括：DINOv2 作为 token 编码器；TRELLIS 的两阶段稀疏/稠密 3D latent 生成；基于 Transformer 的交叉注意力与自注意力；自监督 CLIP/GPTEval3D 评估与 ImageReward 评分。

**📊 数据集**

使用公开的 3D 生成数据集（如 ShapeNet/3DFAIR）作为训练基线，且主要利用 TRELLIS 预训练模型和公开图像对齐数据进行实验；实验中所有条件图像均由用户手动标注区域。

**📈 对比分析**

与 IP‑Adapter、Blended Diffusion、FLUX、TRELLIS 四种基线进行对比；在 GPTEval3D、CLIP 相似度和 ImageReward 评分上均取得最高分，尤其在区域一致性、视觉无缝性、编辑可控性和整体偏好方面表现最佳。

**⚠️ 局限性**

局限性包括：需要用户手动标注多张图像的区域；对 DINOv2 token 维度的依赖导致模型对高分辨率图像的适应性有限；在极端稀疏或重叠的区域对齐时仍可能出现模糊或冲突；算力需求相对较高，尽管生成速度约 20 秒，但仍高于传统逐层优化方法。

---

## 131. PartRAG: Retrieval-Augmented Part-Level 3D Generation and Editing

**arXiv ID:** 2602.17033 | [PDF](https://arxiv.org/pdf/2602.17033v1)

**作者:** Peize Li `[一作]` (King's College London), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 52920 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用单张图片生成可编辑的多部件3D模型并实现局部编辑。

**💡 创新点**

在生成器中加入检索增强的层次对比检索（HCR）和掩码流匹配编辑器，实现高质量部件对齐与可控编辑。

**🔧 技术方法**

使用DiT（双路注意力变压器）、DINOv2与CLIP的跨模态检索、InfoNCE对比损失、流匹配训练与Masked Flow Matching编辑。

**📊 数据集**

在Objaverse、ShapeNet与ABO三大公开数据集上训练并评估，其中检索数据库包含1,236个精细标注部件对象。

**📈 对比分析**

与PartCrafter、HoloPart等基线相比，CD从0.1726降至0.1528（-11.5%），F-Score从0.7472升至0.844（+9.7%），推理时间约38 s，局部编辑仅需5–8 s。

**⚠️ 局限性**

在关节部件、细长几何、对称模糊与长尾类别中表现不佳，且大规模拓扑变化仍难以处理。

---

## 132. ComptonUNet: A Deep Learning Model for GRB Localization with Compton Cameras under Noisy and Low-Statistic Conditions

**arXiv ID:** 2602.17085 | [PDF](https://arxiv.org/pdf/2602.17085v1)

**作者:** Shogo Sato `[一作]` (Waseda University), Jun Kataoka `[通讯]` (Waseda University)

**通讯引用:** 6359 | [OpenAlex ID](https://openalex.org/A5083336808)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种名为ComptonUNet的混合深度学习框架，用于利用Compton相机数据实现γ射线爆发（GRB）的定向与定位；

**💡 创新点**

创新点在于将直接从原始探测事件中估计方向的ComptonNet与基于图像的U-Net去噪架构相结合，实现了在低统计量与高背景噪声环境下的鲁棒定位；

**🔧 技术方法**

采用了CNN/U-Net、MLP、最大池化与反卷积等网络结构，并利用Geant4模拟生成的训练样本进行监督学习，评估指标包括MSE、SSIM和峰值角度偏差；

**📊 数据集**

使用Geant4仿真生成的GRB样本数据集，覆盖1–100 s持续时间、1.0 ph cm⁻² s⁻¹峰值光子通量、随机方向及包含CXB与大气反射背景，共计1000个模拟运行；

**📈 对比分析**

与Unet、ComptonNet及传统反向投影（BP）方法进行对比，ComptonUNet在所有评估指标上均优于基线，30 s时定位误差约7.5°、100 s时约2.5°，MSE和SSIM均显著提高；

**⚠️ 局限性**

局限性包括仅基于模拟数据，未检验在实际飞行环境中的泛化能力；探测器视场受限于30°，数据集规模有限，且未考虑硬件非理想与校准误差等实际问题。

---

## 133. Construction of a classification model for dementia among Brazilian adults aged 50 and over

**arXiv ID:** 2602.16887 | [PDF](https://arxiv.org/pdf/2602.16887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 134. Efficient Tail-Aware Generative Optimization via Flow Model Fine-Tuning

**arXiv ID:** 2602.16796 | [PDF](https://arxiv.org/pdf/2602.16796v1)

**作者:** Zifan Wang `[一作]` (KTH Royal Institute of Technology), Karl H. Johansson `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 44581 | [OpenAlex ID](https://openalex.org/A5045975901)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种高效的流模型微调框架（TFFT），通过条件值在风险和创新两种尾部目标（右CVaR/左CVaR）下实现生成模型的尾部优化。

**💡 创新点**

创新点在于利用CVaR的变分双重形式，将非线性尾部优化拆分为一维阈值搜索和单次熵正则化微调两步，避免了传统的迭代求解，大幅提升计算效率。

**🔧 技术方法**

使用了流模型/扩散模型的连续时间控制框架、熵正则化奖励最大化（如Adjoint Matching）、变分双重CVaR表述与阈值优化算法。

**📊 数据集**

在二维模拟、Stable Diffusion文本-图像生成（ImageReward奖励）以及基于FlowMol的分子生成（GEOM药物数据集）等数据集上进行了实验。

**📈 对比分析**

与预训练模型、期望奖励微调（EXP-FT）和Flow Density Control（FDC）等方法比较，TFFT在保持相同平均奖励的同时显著提升最差尾部（左CVaR）或最高尾部（右CVaR），且训练时间仅为单次微调，性能优于FDC且接近EXP-FT。

**⚠️ 局限性**

局限性包括对奖励函数的范围假设、阈值搜索对样本估计的偏差敏感以及在极端尾部情形下可能需要更细粒度的阈值或多阈值策略。

---

## 135. Evaluating Monolingual and Multilingual Large Language Models for Greek Question Answering: The DemosQA Benchmark

**arXiv ID:** 2602.16811 | [PDF](https://arxiv.org/pdf/2602.16811v1)

**作者:** Charalampos Mastrokostas `[一作]`, Nikos Karacapilidis `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究希腊语言的问答任务，构建了社区审核的DemosQA数据集并对多款大型语言模型进行系统评估。

**💡 创新点**

创新点在于首次公开希腊社交媒体来源的高质量QA数据集DemosQA、提出了可在普通GPU上使用的4‑bit量化评估框架，并对比了多种开源与专有模型在希腊QA上的表现。

**🔧 技术方法**

主要技术包括使用bitsandbytes实现4‑bit量化推理、采用三种提示策略（Instruction、Role、CoT）、greedy解码与规则解析以抽取答案。

**📊 数据集**

使用的数据集包括自建的600条希腊QA对话集DemosQA，以及5个公开的希腊多选问答数据集（Greek Medical MCQA、Greek Truthful QA、BELEBELE、Greek ASEP MCQA、INCLUDE Greek）。

**📈 对比分析**

通过在6个数据集上计算宏观准确率对11款LLM（含GPT‑4o mini、Gemma 2‑9B、Llama Krikri 8B等）进行比较，结果显示GPT‑4o mini整体最高，Llama Krikri 8B与Gemma 2‑9B在多数数据集上与之相近；提示策略对开源模型影响显著。

**⚠️ 局限性**

研究局限在于仅覆盖希腊问答任务、只评估小型开源模型、未包含更大多语模型、以及可用数据集相对有限。

---

## 136. On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking

**arXiv ID:** 2602.16849 | [PDF](https://arxiv.org/pdf/2602.16849v1)

**作者:** Jianliang He `[一作]` (Yale University), Zhuoran Yang `[通讯]` (Yale University)

**通讯引用:** 4711 | [OpenAlex ID](https://openalex.org/A5101727948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文对两层神经网络如何学习特征以解决模数加法任务进行了全面分析，提供了学习模型的机制解释和训练动态的理论解释。

**💡 创新点**

创新点在于提出了一个多样化条件，解释了神经元如何通过相位对称性和频率多样化来结合特征，从而实现全局解决方案。

**🔧 技术方法**

使用了两层全连接神经网络和离散傅里叶变换（DFT）来分析学习过程和训练动态。

**📊 数据集**

使用的训练数据集是模数加法任务的完整数据集，包括所有可能的输入对及其对应的模数和。

**📈 对比分析**

通过与现有方法的比较，本文展示了所提出模型在特征学习和泛化能力上的优势，尤其是在处理模数加法任务时的表现。

**⚠️ 局限性**

限制在于该研究主要集中在特定的模数加法任务上，可能无法直接推广到更复杂的任务或其他类型的神经网络架构。

---

## 137. Expanding the Scope of Computational Thinking in Artificial Intelligence for K-12 Education

**arXiv ID:** 2602.16890 | [PDF](https://arxiv.org/pdf/2602.16890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 138. Attending to Routers Aids Indoor Wireless Localization

**arXiv ID:** 2602.16762 | [PDF](https://arxiv.org/pdf/2602.16762v1)

**作者:** Ayush Roy `[一作]` (University at Buffalo-SUNY), Vishnu Suresh Lokhande `[通讯]` (University at Buffalo-SUNY)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在基于Wi‑Fi的室内定位模型中引入了“路由器注意力”机制，利用通道级注意力对来自不同接入点的AoA‑ToF热图进行加权，从而提升定位精度。

**💡 创新点**

创新点在于将传统加权三角测量的思路转化为可学习的注意力层，使网络能够显式学习每个路由器的贡献度，并通过对齐权重显著降低高误差极端值。

**🔧 技术方法**

采用了Encoder‑Decoder卷积网络（ResNet‑style）、基于Deep Sets的集合不变注意力模块、MLP + Softmax权重生成、AoA‑ToF热图特征与三角测量损失相结合的端到端训练框架。

**📊 数据集**

使用了公开的DLoc“wild”数据集（GitHub链接：https://github.com/ucsdwcsng/DLoc_pt_code/blob/main/wild.md）。

**📈 对比分析**

通过与不使用注意力的基线模型（Vanilla）在3966个样本上的对比，统计中位数误差从63.17 cm降至45.01 cm（≈28.7%），平均误差从77.90 cm降至54.01 cm（≈30.7%），90%分位误差从140.63 cm降至92.88 cm（≈34.0%），99%分位误差从302.32 cm降至183.20 cm（≈39.4%）。

**⚠️ 局限性**

局限性包括：对路由器部署布局仍有一定依赖，极端多径或信号弱环境下注意力表现尚未充分验证；模型未在多种不同数据集或动态环境下进行泛化评估；加入注意力层虽参数量小，但仍需额外训练时间与调参。

---

## 139. Bloc Voting on Single Peaked Preferences

**arXiv ID:** 2602.16734 | [PDF](https://arxiv.org/pdf/2602.16734v1)

**作者:** Ariel Calver `[一作]`, Jennifer Wilson `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

分析单峰偏好下的 Bloc 投票，阐明赢得候选团的结构、邻接性以及与 Condorcet 相关的稳定性。

**💡 创新点**

首次给出 Bloc 投票在少数候选人、不同 k 值时赢得候选团的完整邻接性与 Gehrlein‑稳定性分类，并通过 Monte Carlo 模拟评估这些性质的出现概率。

**🔧 技术方法**

采用单峰偏好模型、Copeland 方法、头对头比较和单峰约束的投票规则，并通过概率分析和头对头计数验证候选团的稳定性。

**📊 数据集**

使用 10,001 名选民的模拟数据，构建三种投票模型：IAC 单峰、EN（标准正态）和 EB（双峰）空间模型。

**📈 对比分析**

将 Bloc 与 k-Copeland 方法的赢家集合进行对比；在 IAC 模型下两者一致率超过 99%，在空间模型下一致率显著下降；并提供各模型下 Gehrlein 与局部稳定性的下界概率。

**⚠️ 局限性**

局限性在于仅考虑单峰偏好、候选人数有限（至 7 位）、未检验实际选举数据，以及未探讨 Bloc 投票在多峰或不完整排名情境下的表现。

---

## 140. DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers

**arXiv ID:** 2602.16968 | [PDF](https://arxiv.org/pdf/2602.16968v1)

**作者:** Dahye Kim `[一作]` (Boston University), Raghudeep Gadde `[通讯]` (Amazon)

**通讯引用:** 837 | [OpenAlex ID](https://openalex.org/A5015956846)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种动态令牌化方法（DDiT），在扩散 Transformer 的每个去噪时间步根据潜在空间演化速率自适应选择不同大小的 patch，从而在保持视觉质量的前提下显著加速图像与视频生成。

**💡 创新点**

核心创新点在于：1) 对扩散模型的潜在空间采用基于细节级别自适应的 patch 大小调度；2) 通过简单的 LoRA 适配器实现对现有预训练 DiT 的无缝接入；3) 提供了一个基于潜在演化速率的判别机制，使得推理过程能够动态分配计算资源。

**🔧 技术方法**

使用技术包括：动态 patch 调度策略、潜在演化速率评估、LoRA 轻量化适配器、基于 CLIP/ ImageReward/VBench 的质量评估，以及对 FLUX‑1.Dev 与 Wan‑2.1 等主流扩散 Transformer 的改造。

**📊 数据集**

实验数据集：FLUX‑1.Dev（文本到图像）和 Wan‑2.1（文本到视频），并在这些模型上进行推理速度与质量对比。

**📈 对比分析**

与原始 DiT 基线相比，DDiT 在 FLUX‑1.Dev 上实现 3.52× 的速度提升，在 Wan‑2.1 上实现 3.2× 的速度提升；同时在 ImageReward、CLIP、VBench 等指标上保持或略优于基线，表明无显著质量损失。

**⚠️ 局限性**

限制与不足：1) 仅在每个时间步使用固定 patch 大小，未对单步内的多尺度处理进行探索；2) 依赖潜在演化速率的 heuristic，可能在极端场景下失效；3) 需要额外的 LoRA 适配，对模型参数进行微调；4) 对长视频生成的扩展尚未在实验中验证。

---

## 141. A Residual-Aware Theory of Position Bias in Transformers

**arXiv ID:** 2602.16837 | [PDF](https://arxiv.org/pdf/2602.16837v1)

**作者:** Hanna Herasimchyk `[一作]` (University of Hamburg), Sören Laue `[通讯]` (University of Hamburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出残差感知的注意力累积理论，解释Transformer在有限深度下的位置信息偏置和Lost-in-the-Middle现象，并证明残差连接能阻止先前注意力仅理论所预测的注意力崩塌。

**💡 创新点**

创新点包括：①将残差混合系数纳入注意力传播；②在有限深度下从因果掩码、残差、位置编码和内容四个结构力量推导出U形位置偏置；③给出无穷深度条件下残差对注意力崩塌的判定，修正先前结论。

**🔧 技术方法**

使用注意力展开（rollout）框架、残差感知转移矩阵、残差混合系数计算、内容对数常数+对角线近似、多头均匀平均、梯度归因评估等技术。

**📊 数据集**

实验使用FineWeb‑Edu（长度2048）评估残差混合和内容结构；模型包括 Falcon‑rw‑7b、MPT‑7b/30b、Bloom‑7b/176b；此外在附录使用 DCLM‑Baseline、Wikipedia 数据集。

**📈 对比分析**

通过比较三种rollout变体（注意力仅、残差感知、残差+常数内容）与真实模型的输入 token 影响（梯度归因）来评估。使用 Spearman 相关系数和 Wasserstein 距离；残差感知版在相关性最高、距离最小，说明理论预测与实测分布高度一致。

**⚠️ 局限性**

局限性包括：①仅分析注意力与残差对信息流的结构性影响，未考虑训练、优化和语义内容；②对乘法位置编码（如 RoPE）的分析不够严格；③未细化多头投影权重差异；④理论只能解释先天的位置信息偏置，未说明其对任务性能的具体影响。

---

## 142. LiveGraph: Active-Structure Neural Re-ranking for Exercise Recommendation

**arXiv ID:** 2602.17036 | [PDF](https://arxiv.org/pdf/2602.17036v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11904 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 LiveGraph，一种基于活跃结构的神经重排框架，用于个性化多样化练习推荐。

**💡 创新点**

创新点在于将图增强的学生表示学习、结构不确定性驱动的重排以及主动探测结合到一个 Meta‑RL 控制器中，解决长尾稀疏和多样性平衡问题。

**🔧 技术方法**

技术主要包括图自编码器（Graph‑VAE）、贝叶斯不确定性度量、基于子图熵的多信号融合、主动学习探测和 MAML‑基元强化学习。

**📊 数据集**

实验使用三大公开数据集：Nips34、Assist2009 与 Assist2012。

**📈 对比分析**

与当前 SOTA NR4DER 以及多种 KT/MLP 基线对比，LiveGraph 在 NDCG、Recall、F1 等指标上平均提升 4–12%，且在活跃与不活跃学生上均优于基线。

**⚠️ 局限性**

主要局限在于模型复杂度高，训练和超参数调优成本大，并且对大规模在线部署仍需进一步优化。

---

## 143. IntentCUA: Learning Intent-level Representations for Skill Abstraction and Multi-Agent Planning in Computer-Use Agents

**arXiv ID:** 2602.17049 | [PDF](https://arxiv.org/pdf/2602.17049v1)

**作者:** Seoyoung Lee `[一作]` (Sookmyung Women's University), Joo Yong Sim `[通讯]` (Sookmyung Women's University)

**通讯引用:** 3325 | [OpenAlex ID](https://openalex.org/A5008309794)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 IntentCUA 框架，将原始交互轨迹转化为多视图意图表示，抽象成可重用技能，并通过计划记忆与多智能体循环实现稳定的长时段桌面自动化。

**💡 创新点**

创新点在于：①基于多视图对齐的意图嵌入和层级聚类（IG/SG），②从子意图群集自动诱导参数化技能提示（skill hints），③缓存式计划重用与补齐机制，④将计划记忆与执行反馈融入的协作循环，显著降低意图漂移和冗余重新规划。

**🔧 技术方法**

使用技术包括：多视图对比学习（contrastive loss + 预测 + 重构）、HDBSCAN 层级聚类、GPT‑4o 生成子计划、计划记忆检索与缓存、Plan‑Optimizer 与 Critic 反馈循环、GUI 归纳与脚本式定位。

**📊 数据集**

数据集由 286 个真实桌面任务构成（100 内部、116 WebVoyager、70 ScreenAgent），以及 30 小时、18 次会话的 113 条交互轨迹，覆盖 36 个域，评估任务覆盖 63 个域。

**📈 对比分析**

与 UI‑TARS‑1.5（RL）和 UFO²（轨迹中心）对比，IntentCUA 在 286 任务上取得 74.8% 的成功率、SER 0.91、平均任务延迟 1.46 分钟，分别比两基线低约 4.5 倍延迟、提高 36–23 个百分点成功率，尤其在 30 步以上长时段任务中保持 40%+ 成功率。

**⚠️ 局限性**

局限性包括：随着计划记忆规模扩大，检索效率可能下降；对弹出窗口等临时视觉遮挡的脚本式 GUI 归纳缺乏鲁棒性；在完全新颖域上的泛化仍受限于轨迹分布偏移。

---

## 144. Overseeing Agents Without Constant Oversight: Challenges and Opportunities

**arXiv ID:** 2602.16844 | [PDF](https://arxiv.org/pdf/2602.16844v1)

**作者:** Madeleine Grunde-McLaughlin `[一作]` (University of Washington), Adam Fourney `[通讯]` (Microsoft Research)

**通讯引用:** 2725 | [OpenAlex ID](https://openalex.org/A5071293344)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并改进人工智能代理（Computer Use Agent）在任务执行后向人类提供的追踪信息，以提升人类验证错误的效率和准确性，完成了三项用户研究。

**💡 创新点**

提出并验证了一种基于“规范（Specification）”的追踪展示方案，该方案在保留执行细节的同时提供高层次需求与假设清单，使用可视化链接和注释减少冗长说明，旨在平衡信息丰富度与可读性。

**🔧 技术方法**

采用人机交互实验方法（包括可用性评估、思考大声、半结构化访谈），使用Wizard‑of‑Oz模拟Agent执行，结合GPT‑4o/5生成的动作追踪，构建了多层次可交互的界面并进行功能对比。

**📊 数据集**

使用AssistantBench中选取的多任务问答样例（4个正确、4个错误），以及自定义的三组任务（餐厅点单、食谱检索、门票折扣计算）作为实验材料；数据来源为GPT‑4o/5执行的完整追踪记录。

**📈 对比分析**

通过受控实验采用配对对照设计，对比基线Magnetic‑UI和改进的规范界面，评估准确率、验证时间和主观置信度。结果显示：准确率差异不显著（Hedges' g = 0.18），验证时间略有减少（g = –0.29），但误判时的置信度显著上升（g = 0.56）。

**⚠️ 局限性**

受限于样本（仅12名技术公司员工）、Wizard‑of‑Oz实现、仅测试单一CUA系统、只关注事后验证、未评估实时交互与自动化生成规范、以及置信度偏高导致的潜在过度信任等问题。

---

## 145. InstantRetouch: Personalized Image Retouching without Test-time Fine-tuning Using an Asymmetric Auto-Encoder

**arXiv ID:** 2602.17044 | [PDF](https://arxiv.org/pdf/2602.17044v1)

**作者:** Temesgen Muruts Weldengus `[一作]` (Zhejiang University), Changqing Zou `[通讯]` (Zhejiang Lab)

**通讯引用:** 2985 | [OpenAlex ID](https://openalex.org/A5100604564)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出InstantRetouch框架，能够在不进行任何测试时微调的情况下，利用少量用户示例实现即时个性化图像美化。

**💡 创新点**

其创新点在于：①引入非对称自编码器，将风格信息从内容中解耦；②设计检索增强的Retouching (RAR)，通过内容相似检索并加权聚合多例风格；③采用轻量级的条件MLP解码器，在色彩空间直接实现风格迁移。

**🔧 技术方法**

技术上使用SigLIP‑v2编码器配合LoRA适配器进行风格提取；用条件MLP在像素级别进行色彩变换；检索阶段利用预训练SigLIP的内容向量进行余弦相似度检索并对风格潜在向量进行温度化权重平均。

**📊 数据集**

训练数据来源于800个Adobe Lightroom预设对95,000张LAION图像进行处理得到的配对样本；评测使用自建的VCIRB（单风格）、PPR10K‑groups（多风格一致）以及MIT‑FiveK和PPR10K Users（多风格不一致）四个基准集。

**📈 对比分析**

与MSM、StarEnhancer、VisualCloze、PhotoArtAgent、NanoBanana、Seedream4.0等通用个性化美化方法以及针对用户训练的AdaInt、RSFNet进行对比，InstantRetouch在PSNR/SSIM上明显优于对手，在LPIPS上也表现最小误差，且在所有评测基准上均优于现有无训练微调方案，甚至与专门训练的模型相当。

**⚠️ 局限性**

局限性包括：对内容相似检索的依赖，导致在内容差异较大或无匹配参考时效果下降；对极端多样或不一致风格的融合仍有限；仅能处理色调与光照变换，对结构性重构能力不如生成式模型。

---

## 146. Arcee Trinity Large Technical Report

**arXiv ID:** 2602.17004 | [PDF](https://arxiv.org/pdf/2602.17004v1)

**作者:** Varun Singh `[一作]`, Lucas Atkins `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并发布了 Trinity 系列稀疏 Mixture‑of‑Experts 大语言模型（Nano、Mini、Large），实现了 400B 参数、每 token 13B 激活的高稀疏模型，并完成了从预训练到后训练、推理基准的完整实验流程。

**💡 创新点**

创新点包括：1) Soft‑clamped Momentum Expert Bias Updates (SMEBU) 载荷平衡策略；2) 交错局部/全局注意力与 gated attention 组合；3) depth‑scaled sandwich norm 归一化；4) 随机序列文档缓冲（RSDB）提升训练稳定性与数据效率；5) 结合 Muon 优化器与 z‑loss 等训练技巧。

**🔧 技术方法**

使用技术包括：稀疏 MoE Transformer、Gated Attention、QK‑Normalization、RoPE、Sliding‑Window Attention、Sigmoid routing、Muon + AdamW 优化器、Cut Cross‑Entropy、vLLM FP8 推理、强化学习 fine‑tuning、RULER MK‑NIAH 长上下文评测等。

**📊 数据集**

数据集涵盖约 17 T（Large）/10 T（Nano/Mini）混合语料，其中包含真实与 8 T 合成 Web/代码/ STEM 数据，使用多语言 C4、AFM、ProLong、FLAN、AutoMathText、ArXiv、PDF OCR 等多源文本。

**📈 对比分析**

通过 MBPP+, Minerva MATH500、HellaSwag、WinoGrande、MMLU、TriviaQA、ARC、BBH、GPQA 等标准基准与同类开源模型对比，Trinity Large Base 在多项指标上与 GLM‑4.5 Base 竞争；推理吞吐量在 FP8 vLLM 环境下高于同规模模型。

**⚠️ 局限性**

局限性：训练时专家负载不均衡和最大负载波动仍偶发；SMEBU 与 RSDB 等技术未单独 ablation；FP8 量化对推理精度有一定影响；后续需更大规模数据、完善 RL 调优与更细粒度的鲁棒性评估。

---

## 147. ALPS: A Diagnostic Challenge Set for Arabic Linguistic & Pragmatic Reasoning

**arXiv ID:** 2602.17054 | [PDF](https://arxiv.org/pdf/2602.17054v1)

**作者:** Hussein S. Al-Olimat `[一作]` (Independent Researchers), Ahmad Alshareef `[通讯]` (Independent Researchers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了 ALPS（阿拉伯语语言学与语用学套件），包含 531 题专家精心构造的多选诊断题，用以评估阿拉伯语深层语义与语用理解。

**💡 创新点**

创新点在于：① 诊断式、专家手工设计，排除合成/翻译误差；② 仅覆盖 MSA/古典阿拉伯，确保文化真实性；③ 单轮零样本、无上下文评测，配合人类单通过基准与专家裁定上限，精准揭示模型对形态句法与语用推理的差异。

**🔧 技术方法**

采用零样本提示、闭式无 agent 推理评测、专家裁定的黄金标准、对 23 模型进行多轮对比，细粒度错误诊断与类别划分。

**📊 数据集**

使用 ALPS 数据集（531 题、15 子任务）以及 23 种模型（商业大型 LLM、开源多语种、阿拉伯本土模型），对比人类单通过平均 84.6% 与专家上限 99.2%。

**📈 对比分析**

在零样本、无上下文、无推理模块的闭式环境下对 23 模型进行对比，商业模型普遍高于人类平均（最高 94.2%），阿拉伯本土模型在语用方面相对强势但总体略低；模型在形态句法依赖任务错误率高达 36.5%，整体表现差距约 37 点。

**⚠️ 局限性**

局限性包括：规模仅 531 题，未覆盖方言；多选格式排除自然语言的多义性；零样本不包含 RAG 或推理能力；未评估模型稳定性或多温度表现；仅测试文本，未涉及多模态；禁止 agent 推理，可能低估实际使用性能。

---

## 148. Large Language Models Persuade Without Planning Theory of Mind

**arXiv ID:** 2602.17045 | [PDF](https://arxiv.org/pdf/2602.17045v1)

**作者:** Jared Moore `[一作]` (Stanford University), Cameron R. Jones `[通讯]` (Stony Brook University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

作者通过三项交互式说服实验，使用MindGames框架评估人类与大型语言模型的计划性理论心智能力（PToM）与说服效果。

**💡 创新点**

创新之处在于将计划性ToM与说服任务结合，设计可调节内部外部效度的实验体系，并揭示LLM在缺乏PToM时仍能高效说服人类。

**🔧 技术方法**

主要技术包括OpenAI GPT‑4o语言模型、自然语言对话生成、实验界面交互以及混合效应模型进行统计分析。

**📊 数据集**

数据集由约500个基于情境的支付矩阵和Prolific平台招募的约800个对话实验组成，LLM通过API自动生成回应。

**📈 对比分析**

对比方法为在揭示与隐藏信息两种条件下比较LLM与人类的说服成功率；结果显示LLM在揭示条件下优于人类，而在隐藏条件下低于人类；但在角色扮演与真实说服实验中，LLM的成功率超越人类。

**⚠️ 局限性**

局限性包括目标行为被人为设定或角色扮演、实验缺乏真实世界多样性、LLM高说服率可能基于关联式策略而非真正的因果理论心智，以及样本规模有限且受Prolific受众限制。

---

## 149. Automating Agent Hijacking via Structural Template Injection

**arXiv ID:** 2602.16958 | [PDF](https://arxiv.org/pdf/2602.16958v1)

**作者:** Xinhao Deng `[一作]` (Tsinghua University), Qi Li `[通讯]` (Tsinghua University)

**通讯引用:** 25915 | [OpenAlex ID](https://openalex.org/A5100350243)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并验证了名为 Phantom 的自动化结构化模板注入框架，用于劫持 LLM 代理，利用聊天模板结构诱发角色混淆。

**💡 创新点**

创新点在于发现并利用代理聊天模板的结构弱点，提出模板自动编码器与贝叶斯优化相结合的离散模板搜索方法，实现跨模型、对抗防御的结构化注入攻击，并系统性发现 70+ 真实漏洞。

**🔧 技术方法**

使用结构化模板注入、模板自动编码器（Template Autoencoder）、贝叶斯优化、多级模板增强、AgentDojo benchmark 评估、语义检测模型（DeBERTa‑v3）等技术。

**📊 数据集**

使用 78 个公开模板作为种子、AgentDojo 数据集（97 正常任务 + 629 对抗测试）、942 个商业代理样本以及对 Qwen、GPT、Gemini 等模型的公开接口。

**📈 对比分析**

通过与 Semantic Injection、Static Template Injection、ChatInject 等基线对比，Phantom 在 7 个闭源代理上的平均攻击成功率 (ASR) 达到 79.76%，远超对手；在不同防御（Delimiter Spotlighting、Tag Filter、Injection Detector）下依然保持高 ASR；在 942 商业代理中发现 70+ 漏洞。

**⚠️ 局限性**

仅针对文本输入；黑盒模型需要多轮查询，受限于速率和异常检测；未考虑多模态代理、长期会话持久化；未评估对话状态管理的持久化影响。

---

## 150. Mind the GAP: Text Safety Does Not Transfer to Tool-Call Safety in LLM Agents

**arXiv ID:** 2602.16943 | [PDF](https://arxiv.org/pdf/2602.16943v1)

**作者:** Arnold Cartagena `[一作]` (Independent Researcher), Ariane Teixeira `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 GAP 基准，用于量化大语言模型代理在文本拒绝与工具调用安全性之间的偏差（即“文本-行动差距”）。

**💡 创新点**

首次将文本拒绝与工具调用安全性视为并行但可独立评估的维度，提出 GAP（文本拒绝且执行禁用工具调用）和 LEAK（禁用调用且泄露 PII）两项指标；通过三种系统提示（中性、安全强化、工具鼓励）与三种治理模式（无监控、观察、强制）对六大前沿模型进行全因子实验，系统揭示了模型安全性对提示高度敏感且文本安全与行动安全并不完全一致。

**🔧 技术方法**

采用了多模型推理（Claude Sonnet 4.5、GPT‑5.2、Grok 4.1、DeepSeek V3.2、Kimi K2.5、GLM‑4.7），构建了六个监管域（制药、金融、教育、就业、法律、基础设施）下的模拟工具；使用编写的可执行治理合同（Edictum）实现工具调用检测与强制；评估指标通过正则表达式、预定义禁用谓词、PII 标记实现零噪声确定性打分；统计方法包括 Clopper‑Pearson 置信区间、两比例 z 检验、Bonferroni 校正。

**📊 数据集**

数据集由 7 种针对每个域的 jailbreak 场景（共 42 个）+ 2 个合法使用控制场景组成，配合两种提示变体、三种系统提示和三种治理模式生成 17,420 条可分析记录；所有交互记录与评估脚本已公开托管于 GitHub/HuggingFace。

**📈 对比分析**

比较方法：对 TC‑safe、GAP、LEAK 进行全因子统计对比；结果显示：文本安全率（T‑safe）与工具调用安全率（TC‑safe）在多数模型上不相关；在安全强化提示下 GAP 率下降但仍有 219 起案例；在工具鼓励提示下 GPT‑5.2 的条件 GAP 率高达 79.3%；治理模式仅显著降低 LEAK，未能抑制禁用调用尝试。

**⚠️ 局限性**

主要局限：使用无鉴权的模拟工具，缺乏真实系统后端；只评估单轮对话、单温度（0.3）、仅英文提示；不考虑工具调用序列的组合攻击；缺乏多模型/多语言验证；治理合同按单调用评估，未覆盖跨调用的序列风险；模型内部机制未进行解释性分析。

---

## 151. NeuDiff Agent: A Governed AI Workflow for Single-Crystal Neutron Crystallography

**arXiv ID:** 2602.16812 | [PDF](https://arxiv.org/pdf/2602.16812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 152. Exact Certification of Data-Poisoning Attacks Using Mixed-Integer Programming

**arXiv ID:** 2602.16944 | [PDF](https://arxiv.org/pdf/2602.16944v1)

**作者:** Philip Sosnin `[一作]` (Imperial College), Calvin Tsay `[通讯]` (Imperial College)

**通讯引用:** 1067 | [OpenAlex ID](https://openalex.org/A5068409517)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将数据投毒、梯度训练和测试评估统一到单一混合整数二次约束程序（MIQCP）的完整验证框架，能够给出最优攻击与可证明的鲁棒性证书。

**💡 创新点**

创新点在于：① 用MIQCP实现训练时鲁棒性的声称完备性；② 在同一模型中精确编码数据扰动、梯度更新与测试目标；③ 结合边界收缩、辅助变量以及局部搜索启发式来提升求解效率。

**🔧 技术方法**

采用混合整数二次规划（MIQCP）、big‑M 形式、优化导向的边界收缩（OBBT）、辅助变量解耦以及基于本地搜索的启发式。

**📊 数据集**

实验使用三组小型数据集：Iris、Diabetes、Halfmoons，模型为线性层（或在Halfmoons上先做多项式特征扩展）。

**📈 对比分析**

与现有不完整的认证方法相比，MIQCP 能提供更严格、可证明的最优攻击结果，实验表明尽管求解时间显著增加，但给出的鲁棒性上界更精确。

**⚠️ 局限性**

局限性包括：① 计算复杂度高，只能处理小规模模型和数据集；② 依赖固定初始化与批次顺序的白盒假设；③ 对大攻击预算时连续松弛的 tightness 受限，导致求解更困难。

---

## 153. Discovering Multiagent Learning Algorithms with Large Language Models

**arXiv ID:** 2602.16928 | [PDF](https://arxiv.org/pdf/2602.16928v1)

**作者:** Zun Li `[一作]` (Google DeepMind), Marc Lanctot `[通讯]` (Google DeepMind)

**通讯引用:** 26153 | [OpenAlex ID](https://openalex.org/A5049659586)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用LLM驱动的进化算法AlphaEvolve自动化设计多智能体学习算法，生成新的CFR和PSRO变体。

**💡 创新点**

创新在于把算法代码视为基因，通过LLM进行语义变异，发现了非直观的自适应折扣、冲击放大和混合元求解器等机制。

**🔧 技术方法**

主要技术包括AlphaEvolve、Gemini LLM、Python代码演化框架、OpenSpiel环境以及精确的可解释算法骨架。

**📊 数据集**

在训练集中使用4个小型不完全信息博弈（3人Kuhn, 2人Leduc, 4张Goofspiel, 5面Liars Dice），在测试集使用更大、更复杂的10余个游戏。

**📈 对比分析**

通过在训练集和测试集上测量最终策略的可利用性，VAD‑CFR在10/11个测试游戏中优于现有最先进方法，SHOR‑PSRO在8/11游戏中同样表现最佳。

**⚠️ 局限性**

局限在于演化过程受限于预设的代码骨架和代理提示，难以保证在极大规模或连续控制环境中的可迁移性，并且需要大量计算资源。

---

## 154. How should AI knowledge be governed? Epistemic authority, structural transparency, and the case for open cognitive graphs

**arXiv ID:** 2602.16949 | [PDF](https://arxiv.org/pdf/2602.16949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 155. Privacy-Aware Split Inference with Speculative Decoding for Large Language Models over Wide-Area Networks

**arXiv ID:** 2602.16760 | [PDF](https://arxiv.org/pdf/2602.16760v1)

**作者:** Michael Cunningham `[一作]` `[通讯]`, Michael Cunningham

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将大型语言模型拆分在本地 GPU 与云端 GPU 之间、仅传输中间激活、并通过 lookahead 解码技术实现 8–9 令牌/秒（80 ms RTT）或 13–17 令牌/秒（低 RTT） 的隐私友好推理系统。

**💡 创新点**

创新点包括：① 在嵌入/解嵌层保持本地化，确保原始 token 永不离开受信任设备；② 将 lookahead 解码迁移到分拆推理，成功将网络延迟摊薄到多位 token；③ 对模型拆分深度进行逆向攻击实验，量化隐私与性能的可调权衡；④ 在 12 B 模型上验证可扩展性，保持一致的接收率；⑤ 通过 RTT 分解模型实现跨供应商性能公平比较；⑥ 记录了真实云端部署中 SSH 隧道、网络架构对延迟的决定性影响。

**🔧 技术方法**

使用的技术包括：Transformer 层级拆分、Jacobi 迭代与 n‑gram 采样的 lookahead 解码、WebSocket 二进制激活传输、SSH 隧道与端口映射、KV‑cache 本地化与管理、针对多 token 的自定义注意力掩码、以及用于评估隐私的 MLP 逆向攻击。

**📊 数据集**

使用多类型提示（代码、结构化、创意、对话）生成的 100–200 token 语料进行实验，并在 880 条（token、激活）对上训练逆向攻击 MLP；未使用公开标准数据集，实验基于自制提示集。

**📈 对比分析**

通过与顺序推理的基线对比、不同 n‑gram 大小与提示类别的 ablation、RTT 投影模型以及对 7 B 与 12 B 的吞吐量测量，显示：lookahead 推理在 80 ms RTT 下达到 8–9 tok/s，RunPod 低 RTT 下可达 13–17 tok/s，12 B 模型在相同 RTT 下仍能保持约 8 tok/s，且通过 RTT 分解可预估 20 ms RTT 下 15–19 tok/s 的性能。

**⚠️ 局限性**

局限性包括：① 仅提供结构化隐私保障，未实现正式的加密或可信执行环境保证；② 逆向攻击实验仅对 2–8 层拆分给出低估值，实际攻击者可能更高；③ 仅在 Mistral 7 B 与 NeMo 12 B 两个同族模型上验证，需在更多体系结构与更大规模上测试；④ 对创意文本的 lookahead 接收率低，可能无法充分抵消长 RTT；⑤ 仅针对贪心 argmax，采样策略下可能出现更低接收率；⑥ 预填充延迟未优化，首次生成 token 仍需 100–500 ms；⑦ 需要 SSH 隧道和云端网络配置，部署复杂度较高。

---

## 156. Sound of Touch: Active Acoustic Tactile Sensing via String Vibrations

**arXiv ID:** 2602.16846 | [PDF](https://arxiv.org/pdf/2602.16846v1)

**作者:** Xili Yi `[一作]` (University of Michigan), Nima Fazeli `[通讯]` (University of Michigan)

**通讯引用:** 1479 | [OpenAlex ID](https://openalex.org/A5031338070)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种基于张力弦振动的主动声学触觉传感器，利用持续电磁激励和双麦克风测量弦振动谱，实时推断接触位置、正向力和滑动状态。

**💡 创新点**

创新点在于：①使用连续激励的电磁弦振动实现一维可扩展触觉；②通过物理建模与仿真揭示接触位置与力对谐波频率的独立影响；③将预训练的Audio Spectrogram Transformer作为特征提取器，实现低延迟轻量化多任务学习。

**🔧 技术方法**

技术包括：电磁激励器（EBow）、接触麦克风、声学预放大、双通道音频采集、AST特征提取、轻量化分类器与回归网络、t‑SNE可视化。

**📊 数据集**

使用机器人抓手（WSG‑50）与 KUKA LBR 机械臂在多种材料（3D 打印、金属管、木棍、Allen 键）上收集 17,586 条带标签的音频样本，包含位置、力、滑动与无接触标签。

**📈 对比分析**

与传统视觉/电容/压电等触觉方式相比，系统在金属和塑料物体上实现毫米级定位（MAE≈2–5 mm）、力估计误差<0.2 N，并且滑动检测准确率 100%。在外部噪声环境下性能仅略有下降，表明鲁棒性强。

**⚠️ 局限性**

局限性包括：只能检测单一接触点，难以扩展到二维多点；弦张力漂移和材料柔软度会影响频率响应；在高度动态或强阻尼物体上的实时定位受限；需要持续激励导致启动延迟。

---

## 157. GPU-Accelerated Algorithms for Graph Vector Search: Taxonomy, Empirical Study, and Research Directions

**arXiv ID:** 2602.16719 | [PDF](https://arxiv.org/pdf/2602.16719v1)

**作者:** Yaowen Liu `[一作]` (Hong Kong Polytechnic University), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 27501 | [OpenAlex ID](https://openalex.org/A5100333516)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对GPU加速的基于图的近似最近邻搜索（ANNS）算法进行系统综述和实验评估

**💡 创新点**

提出完整的GPU优化技术分类和图质量评估指标（平均查询路径长度），量化数据传输对性能的影响，并对六种主流GPU算法与两种CPU基线进行统一对比

**🔧 技术方法**

GPU并行化图构造与搜索（包括批量插入、迭代细化、并行划分；候选定位、批量距离计算、数据结构维护），使用CUDA、异步传输、warp拆分、量化（PQ）等技术

**📊 数据集**

8个大型基准数据集：NYTimes（29w，256D），DEEP1M、DEEP10M、DEEP100M（96D），SIFT1M（128D），GIST（960D），GloVe200（256D），MNIST8M（784D）

**📈 对比分析**

对比指标包括构造时间、峰值内存、平均度、连通分量、查询路径长度、QPS、Recall@10；实验显示GPU算法在计算密集型部分（距离计算）可达10×加速，但数据传输占比高达40–97%，导致整体瓶颈；CAGRA在大规模数据上保持相对稳定的QPS

**⚠️ 局限性**

主要局限是PCIe数据传输成本与GPU内存占用率高，导致大规模（百M级）时无法全部驻留GPU；混合CPU‑GPU方案仅把瓶颈移至CPU，仍受主机内存带宽/容量限制，难以进一步扩展到亿级/十亿级

---

## 158. Cross Pseudo Labeling For Weakly Supervised Video Anomaly Detection

**arXiv ID:** 2602.17077 | [PDF](https://arxiv.org/pdf/2602.17077v1)

**作者:** Lee Dayeon `[一作]`, Lee Sangyoun `[通讯]` (Yonsei University)

**通讯引用:** 9975 | [OpenAlex ID](https://openalex.org/A5100452365)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于双分支交叉伪标签的弱监督视频异常检测框架CPL-VAD，兼顾异常定位与类别识别。

**💡 创新点**

创新点在于跨分支伪标签互补及一致性感知精炼模块，提升了时序精度与语义判别。

**🔧 技术方法**

采用CLIP视觉语言模型、ActionFormer多尺度时序编码、多级提示学习和CAR模块。

**📊 数据集**

在XD-Violence和UCF-Crime两大弱监督基准上评测。

**📈 对比分析**

与现有CLIP驱动方法相比，CPL-VAD在粗粒度AP/AUC和细粒度mAP上均取得最高成绩。

**⚠️ 局限性**

局限性包括对CLIP预训练特征的高度依赖，以及伪标签生成过程仍需人工调参。

---

## 159. ICP-Based Pallet Tracking for Unloading on Inclined Surfaces by Autonomous Forklifts

**arXiv ID:** 2602.16744 | [PDF](https://arxiv.org/pdf/2602.16744v1)

**作者:** Takuro Kato `[一作]` (National Institute of Advanced Industrial Science and Technology), Mitsuharu Morisawa `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 5941 | [OpenAlex ID](https://openalex.org/A5014473932)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种利用ICP算法实时跟踪托盘并调节叉车叉角的控制方法，使得自动叉车能够在倾斜表面安全卸货并撤回叉头而不拖拉托盘。

**💡 创新点**

创新点包括：① 在卸货过程中实时使用ICP进行托盘姿态估计；② 将估计结果动态映射到叉角调节，完成与倾斜表面平行；③ 结合叉底限位开关实现安全撤回；④ 方案对托盘货物形状不敏感，可使用RGB‑D或LiDAR。

**🔧 技术方法**

技术手段包括：ICP点云配准、ROS nodelet实现、RGB‑D摄像头（Azure Kinect）、Jetson Xavier+cuPCL进行加速、PD 控制模拟油压执行、限位开关检测、与叉臂高度/倾角反馈同步。

**📊 数据集**

实验数据来源为：① 通过Choreonoid仿真得到的四种卸货情景；② 真实叉车（Toyota AGF Rinova 8AFBR15）配备Azure Kinect的实测数据；未使用公开数据集，全部为自制实验环境与仿真。

**📈 对比分析**

通过与不调节叉角（无控制）方案对比，实验显示：① 仿真中叉角误差小于0.25°，卸货无拖拉；② 实测中叉角误差收敛到0.25°以内，四种情景均成功完成卸货；性能优于传统仅靠高度或角度补偿的方法。

**⚠️ 局限性**

局限性：① ICP计算周期决定了系统响应速度，若周期过长可能导致卸货延迟；② 单摄像头在高位或形状复杂的托盘上特征不足时需要多传感器组合；③ 方案需要对叉车进行硬件改造（限位开关、倾斜角度反馈）；④ 仅在较小倾斜角（≤4°）范围内验证，尚未测试极端倾斜或重载条件。

---

## 160. Dynamic System Instructions and Tool Exposure for Efficient Agentic LLMs

**arXiv ID:** 2602.17046 | [PDF](https://arxiv.org/pdf/2602.17046v1)

**作者:** Uria Franko `[一作]` `[通讯]`, Uria Franko

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 Instruction-Tool Retrieval (ITR) 框架，动态检索并拼接每一步所需的系统指令片段与工具集合，从而显著减少上下文令牌并提升工具调用准确率。

**💡 创新点**

创新点在于：①将检索对象从知识改为系统指令和工具本身；②基于预算的贪心选择和置信门控回退机制；③证明在多步代理中令牌节省与成本降低可叠加，带来指数级的效率提升。

**🔧 技术方法**

使用了 RAG 混合检索（dense+BM25+cross‑encoder）、预算约束的 knapsack 贪心选择、动态提示拼接、安全覆盖层以及工具发现回退策略。

**📊 数据集**

在自建的三类任务基准（T1 结构化 API、T2 多步推理、T3 DevOps 文档）以及生产环境真实工具和系统指令集合上进行评估；共包含 100 余个任务。

**📈 对比分析**

与四种基线（Monolithic、Router‑Only、Prompt‑RAG、ITR）对比，ITR 在每步令牌减少 95%、工具正确率提升 32% 以及端到端成本下降 70% 的同时，能在多步循环中实现累积收益；实验数据表明收益随步骤数线性叠加。

**⚠️ 局限性**

局限性：依赖指令和工具的质量与维护；检索召回不足时可能隐藏正确工具；需要额外的监控、日志与安全覆盖；在极端开放或对抗性场景下效果不一定可复制。

---

## 161. MMCAformer: Macro-Micro Cross-Attention Transformer for Traffic Speed Prediction with Microscopic Connected Vehicle Driving Behavior

**arXiv ID:** 2602.16730 | [PDF](https://arxiv.org/pdf/2602.16730v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 162. NESSiE: The Necessary Safety Benchmark -- Identifying Errors that should not Exist

**arXiv ID:** 2602.16756 | [PDF](https://arxiv.org/pdf/2602.16756v1)

**作者:** Johannes Bertram `[一作]` (University of Tübingen and Max-Planck Institute for Intelligent Systems), Jonas Geiping `[通讯]` (ELLIS Institute Tübingen and Max-Planck Institute for Intelligent Systems)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了NESSiE轻量级安全基准，评估LLM在简单抽象安全相关任务中的安全性与有用性。

**💡 创新点**

通过多套测试套件（RULeS、Agentic、Generated、Skills、Multiturn、Distraction Context、Disabled Reasoning）和SH（Safe & Helpful）指标，提供了可立即使用的必要安全检验工具，填补了现有大规模评测的空白。

**🔧 技术方法**

采用关键词匹配评估方法，对模型多次随机生成输出进行统计；实验设计包括禁用推理、注入无关对话噪声，并手工对失败案例进行错误分类。

**📊 数据集**

基于自定义的93个系统-用户组合共837条交互，构成了NESSiE数据集，已公开发布于huggingface及GitHub。

**📈 对比分析**

与多种LLM（如Llama2 7B、Mistral 7B、Gemini 2.5 Pro、Claude Opus 4.5、Qwen3 VL 32B等）比较，使用SH、Safe、Helpful指标。闭源大模型SH得分80–95%，开源模型仅17–29%；Helpfulness普遍高于Safety；Skills与RULeS Reformulated更难，Distraction Context导致SH下降≥15%。

**⚠️ 局限性**

基准仅为必要条件，不能保证整体安全；仅覆盖简单抽象场景，忽略更复杂对抗与上下文细节；关键词匹配方法可能导致误判；对抗性攻击未充分覆盖；模型误差呈家族化，需要更细粒度的评估。

---

## 163. StereoAdapter-2: Globally Structure-Consistent Underwater Stereo Depth Estimation

**arXiv ID:** 2602.16915 | [PDF](https://arxiv.org/pdf/2602.16915v1)

**作者:** Zeyu Ren `[一作]` (University of Melbourne), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 52920 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 StereoAdapter-2 框架，使用 ConvSS2D 替代 ConvGRU，实现一次更新即可长距离传播，并构建 UW-StereoDepth-80K 合成水下立体数据集，用于零样本深度估计。

**💡 创新点**

核心创新在于基于选择性状态空间模型的四方向扫描 ConvSS2D，能够在一次迭代中实现线性复杂度的长距离空间传播；以及两阶段 Diffusion 生成流水线构建的 UW-StereoDepth-80K 数据集。

**🔧 技术方法**

采用 Depth Anything 3 编码器 + LoRA 微调、ConvSS2D 迭代更新、四方向扫描、Selective SSM、Atlantis + NVS‑Solver 的 Diffusion 样式迁移与视角合成、RAFT‑Stereo 迭代框架。

**📊 数据集**

训练使用合成的 UW‑StereoDepth‑80K（80,000 对）数据；评估在 TartanAir‑UW、SQUID 公开数据集以及 BlueROV2 实机测试。

**📈 对比分析**

与 LEAStereo、PSMNet、RAFT‑Stereo 等传统方法以及先前的 StereoAdapter 进行对比；在 TartanAir‑UW 零样本下取得 REL 0.044、RMSE 2.40、A1 96.8%；在 SQUID 上 RMSE 1.75、REL 0.070、A1 94.3%；在 BlueROV2 上 REL 0.102、RMSE 1.72、A1 92.6%，均显著优于基线。

**⚠️ 局限性**

在极端浊度、强背散射或快速光照变化的极端水下条件下仍存在域差距；连续帧的时序一致性不足，需要加入时间建模；对真实世界极端情况的覆盖仍有限。

---

## 164. Three-dimensional Damage Visualization of Civil Structures via Gaussian Splatting-enabled Digital Twins

**arXiv ID:** 2602.16713 | [PDF](https://arxiv.org/pdf/2602.16713v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 165. Dynamic Delayed Tree Expansion For Improved Multi-Path Speculative Decoding

**arXiv ID:** 2602.16994 | [PDF](https://arxiv.org/pdf/2602.16994v1)

**作者:** Rahul Thomas `[一作]` (Ritual), Arka Pal `[通讯]` (Ritual)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多路径推测解码中系统评估验证方法，并提出延迟树展开与神经选择器以提升OT方法性能。

**💡 创新点**

创新点在于：提出延迟树展开策略以减少早期分支浪费，并训练上下文感知的神经选择器动态决定展开参数，使OT验证方法可超过传统遍历验证。

**🔧 技术方法**

使用的技术包括：多路径推测解码框架、最优运输（OT）验证、延迟树展开、轻量级MLP神经选择器、离线采样、吞吐量与块效率评估。

**📊 数据集**

使用的数据集为数学题集MATH500、奥林匹克题集OlympiadBench、代码生成集LiveCodeBench、创意写作集LitBench以及翻译集Opus。

**📈 对比分析**

与Traversal、SpecInfer等方法对比，通过块效率和吞吐量评测，发现Traversal原始方法最优，但引入神经延迟展开后SpecInfer吞吐量提升约5%，整体性能显著提高。

**⚠️ 局限性**

局限性在于：仍需离线训练，选择器对不同模型/任务的泛化有限；方法主要针对i.i.d.树展开，未覆盖确定性树或混合树场景；并且在某些模型或采样配置下无法彻底超越Traversal。

---

## 166. Early-Warning Signals of Grokking via Loss-Landscape Geometry

**arXiv ID:** 2602.16967 | [PDF](https://arxiv.org/pdf/2602.16967v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在SCAN和Dyck-1两种不同序列学习任务中，神经网络从记忆到泛化的突发转变（grokking）是否具有普适的几何机制；

**💡 创新点**

发现“换位缺陷”（commutator defect）在所有任务中都能在泛化出现前可靠触发，并且其前置时间随学习率遵循超线性幂律，形成可预警窗口；

**🔧 技术方法**

使用了梯度非可交换度量（换位缺陷）、权重空间主成分分析、三基底可积性分解、因果干预实验等技术；

**📊 数据集**

数据集包括SCAN的自然语言指令-动作对（2048训练样本）、Dyck-1括号深度预测（50训练序列）以及先前的模数算术任务；

**📈 对比分析**

通过比较不同学习率、不同干预强度下的换位缺陷触发和模型泛化时间，展示了换位缺陷的提前警示作用，且对干预表现出不同的敏感性（模数算术不敏感、Dyck高度敏感、SCAN中等）；

**⚠️ 局限性**

局限性包括样本数有限（仅少量种子、极少学习率点）、仅使用单一正则化强度、模型规模偏小、任务为合成数据且未验证在大规模自然语言预训练模型上的可迁移性。

---

## 167. Connecting the Dots: Surfacing Structure in Documents through AI-Generated Cross-Modal Links

**arXiv ID:** 2602.16895 | [PDF](https://arxiv.org/pdf/2602.16895v1)

**作者:** Alyssa Hwang `[一作]` (University of Pennsylvania), Andrew Head `[通讯]` (University of Pennsylvania)

**通讯引用:** 1801 | [OpenAlex ID](https://openalex.org/A5003473641)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为复杂多模态文档（如科研论文）提供细粒度信息集成与可视化阅读界面，支持图像与文本之间的跨模态链接。

**💡 创新点**

提出一般化的实体-链接框架，并实现为可嵌入的阅读工具，利用AI自动识别图表中的视觉实体并与文本短语关联，形成交互式参考面板和视觉索引。

**🔧 技术方法**

使用多模态大型语言模型（GPT‑4o、GPT‑5、Molmo）进行实体识别、位置定位、描述生成；前端用 JavaScript、后端用 Python 开发。

**📊 数据集**

基于 ACM Digital Library 的 HTML 科研论文（主要为 HCI/NLP 领域），在实验中以单篇 HCI 论文进行阅读测试，生成的实体与描述由模型自动产生。

**📈 对比分析**

对照组为无增强的论文阅读，受试者完成 25 分钟阅读 + 10 题开放式测验；结果显示实验组答题质量显著提升（p<0.001），完成时间与 NASA‑TLX 认知负荷无显著差异。

**⚠️ 局限性**

局限性：仅在计算机科学/HCI 论文上验证效果，样本量小、仅单篇论文实验；对 LLM 质量依赖；在长篇或其他学科文档中的泛化性待验证。

---

## 168. Say It My Way: Exploring Control in Conversational Visual Question Answering with Blind Users

**arXiv ID:** 2602.16930 | [PDF](https://arxiv.org/pdf/2602.16930v1)

**作者:** Farnaz Zamiri Zeraati `[一作]` (University of Maryland), Hernisa Kacorri `[通讯]` (University of Maryland)

**通讯引用:** 1968 | [OpenAlex ID](https://openalex.org/A5009475920)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过三阶段研究（实验室引导、10 天日记、访谈）系统分析盲人如何利用提示工程（prompting）技术在 Be My AI VQA 系统中自定义、控制生成回答，以提升效率、清晰度和信任感。

**💡 创新点**

创新点在于：①首次将现有 Prompting 技术（如二元反馈、零样本风格/意图提示、链式思维、分解提示、图像转文字提示等）与盲人可访问性需求结合；②构建并公开 418 条多轮对话数据集；③揭示盲人自发的自定义策略及其在日常情境中的实际效果与局限。

**🔧 技术方法**

使用的技术包括：大语言模型（LLM）+ 视觉‑语言模型（VLM）框架（Be My AI 的后台模型），以及多种 Prompting 形式（Binary Feedback、Zero-shot Style、Zero-shot Intention、Chain‑of‑Thought、Decomposition、Image‑as‑Text、Self‑Criticism、Role Prompting、Action‑oriented Prompting）。

**📊 数据集**

数据集为从 11 名盲人收集的 418 条对话（含 363 条日记记录、55 条实验室记录），包括图片、文本交互、任务目标、环境标签和成功/失败评估，已公开可下载。

**📈 对比分析**

本文并未与其他系统做量化对比；评估方式为定性分析（交互长度、轮数、成功率、用户满意度）。在实验室中约 83% 的交互成功；在日记中约 78% 的长交互成功，整体显示 Prompting 能显著改善信息匹配，但仍存在模型错误、可视化限制和安全性问题。

**⚠️ 局限性**

局限性包括：仅使用单一产品（Be My AI）且缺乏持续记忆；样本规模有限（11 名盲人）；无法验证模型对提示的长期适应性；对高风险任务（如药物）缺乏专业验证；缺少与传统 VQA 数据集的定量基准。

---

## 169. Read-Modify-Writable Snapshots from Read/Write operations

**arXiv ID:** 2602.16903 | [PDF](https://arxiv.org/pdf/2602.16903v1)

**作者:** Armando Castañeda `[一作]` (Autonomous University of Mexico), Braulio Ramses Hernández Martínez `[通讯]` (Autonomous University of Mexico)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了两种仅使用读写操作的可等待（wait‑free）Read‑Modify‑Writable（RMWable）快照算法，分别适用于有限进程数的标准共享内存模型和无界并发模型。

**💡 创新点**

创新点在于：① 证明仅用读写操作即可实现任意可读对象的快照；② 设计了基于计数器 T 与帮助数组 H 的帮助机制，使得 Scan 能在无竞争和有竞争的环境下都能正确完成；③ 在无界并发模型中引入动态进程集处理，使得即使无限多进程加入也保持无等待性。

**🔧 技术方法**

核心技术包括：读写共享寄存器、Collect（顺序读取整个数组）、计数器 T 记录更新进度、帮助数组 H 存储更新者的快照、鸽巢原理（pigeonhole principle）证明至少有一次单次更新导致的快照、以及循环重试与递归帮助深度的分析。

**📊 数据集**

未使用任何具体数据集；该工作属于理论计算机科学，关注算法的可计算性与正确性。

**📈 对比分析**

算法的性能主要从时间复杂度和空间复杂度评估：时间复杂度为 O(n²·m)，空间复杂度为 O(n² + n·m)。作者指出此复杂度相对较高，并未在实验或实际系统中评估性能。

**⚠️ 局限性**

限制包括：① 复杂度高，未达到现有最快快照算法的多项式或线性级别；② 需要存储 O(n²) 个寄存器（T 与 H），在大规模进程数下空间占用较高；③ 仅提供理论证明，缺乏实验验证与对比；④ 对异常行为（如进程大量崩溃或频繁加入）未做详细性能分析。

---

## 170. RRT$^η$: Sampling-based Motion Planning and Control from STL Specifications using Arithmetic-Geometric Mean Robustness

**arXiv ID:** 2602.16825 | [PDF](https://arxiv.org/pdf/2602.16825v1)

**作者:** Ahmad Ahmad `[一作]` (Boston University), Calin Belta `[通讯]` (University of Maryland)

**通讯引用:** 11774 | [OpenAlex ID](https://openalex.org/A5086742095)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种名为RRT^η的基于采样的运动规划框架，利用算术-几何平均（AGM）鲁棒性来在Signal Temporal Logic（STL）约束下生成具有高鲁棒性的控制序列。

**💡 创新点**

创新点包括：① AGM鲁棒性区间语义和增量监测算法；②利用Fulfillment Priority Logic（FPL）构建方向向量以实现多目标的原则化组合；③在RRT^*框架中保持概率完备性和渐进最优性，并在实践中显著提升鲁棒性与搜索效率。

**🔧 技术方法**

使用技术包括：采样式RRT^*算法、AGM鲁棒性计算与区间推理、DIAS（方向向量）与FPL权重组合、增量鲁棒性监测、任务空间采样与逆运动学（IK）缓存等。

**📊 数据集**

实验数据集为三种机器人系统：一维积分点机器人、平面单车机器人（Unicycle）与七自由度KUKA iiwa机械臂，各自在不同的时空约束任务上进行仿真。

**📈 对比分析**

与传统基于最小-最大鲁棒性的STL‑RRT^*进行对比，RRT^η在负鲁棒性区域的传统方法未能找到可行解，而AGM方法在同样任务中迅速收敛到高鲁棒性（η≈0.93–0.95），并且FPL策略在迭代次数与计算时间上比随机合成策略提高约1.5–2倍。

**⚠️ 局限性**

限制：当前框架假设系统动力学满足Lipschitz连续性；增量监测对复杂时空约束仍有计算开销；对动态障碍物或不确定性环境的鲁棒性尚未验证；实现依赖于手工制定的STL规范。

---

## 171. ML-driven detection and reduction of ballast information in multi-modal datasets

**arXiv ID:** 2602.16876 | [PDF](https://arxiv.org/pdf/2602.16876v1)

**作者:** Yaroslav Solovko `[一作]` `[通讯]` (Independent Researcher), Yaroslav Solovko (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多模态（结构化、半结构化、非结构化、稀疏）数据中冗余无用信息（ballast）的检测与删减，提出统一的 Ballast Score 并给出了完整的检测与裁剪流程。

**💡 创新点**

创新点在于：① 给出了跨模态的 ballast 定义与分类（统计、冗余、语义、模型无关）；② 通过熵、互信息、方差、Lasso、SHAP、PCA、LDA、BERT 等多维信号构建统一的 Ballast Score；③ 将多模态特征的裁剪结果与传统降维/特征选择方法系统对比。

**🔧 技术方法**

使用的信息论度量（熵、互信息、方差阈值）、机器学习解释技术（SHAP、Lasso、CatBoost/LightGBM）、降维方法（PCA、t‑SNE）、主题模型（LDA）、语义相似度（BERT、TF‑IDF、余弦相似）以及 IoU 等。

**📊 数据集**

实验数据集包括：IEEE‑CIS Fraud Detection（结构化）、Amazon Fashion Reviews（半结构化）、CORD‑19 与 PubLayNet（非结构化文本/图像）、Ireland Census 2022（高维稀疏表格）。

**📈 对比分析**

通过对比原始特征集与裁剪后特征集的 AUC、F1、准确率、训练时长和特征数量等指标，发现大多数场景下裁剪 70%‑90% 的特征仍能保持甚至提升模型性能，并显著降低训练时间与存储占用。

**⚠️ 局限性**

局限性包括：阈值需针对每个数据集手动调优，跨模态统一阈值不易；在语义复杂或低频高价值特征场景下，SHAP/Lasso 等模型依赖方法表现不佳；未对图像、音频、图谱等非文本模态进行验证；可能遗漏罕见但重要的特征。

---

## 172. Amber-Image: Efficient Compression of Large-Scale Diffusion Transformers

**arXiv ID:** 2602.17047 | [PDF](https://arxiv.org/pdf/2602.17047v1)

**作者:** Chaojie Yang `[一作]` (HelloGroup Inc), Jun Gao `[通讯]` (HelloGroup Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过压缩和知识蒸馏将60层MMDiT的Qwen-Image压缩为10B和6B两种轻量级T2I模型，保持高质量生成。

**💡 创新点**

结合层重要性评估、局部权重平均初始化、双流到单流的混合架构以及两阶段蒸馏，完成70%参数压缩且无需从零训练。

**🔧 技术方法**

使用结构化深度剪枝、局部权重平均、两阶段知识蒸馏、双流到单流转换、轻量级全参数微调等技术，整个训练耗时约2000 GPU小时。

**📊 数据集**

使用约一百万对图文的内部数据集（包括真实与合成样本），用于蒸馏和微调。

**📈 对比分析**

在DPG‑Bench、GenEval、OneIG、LongText‑Bench、CVTG‑2K等公开基准上评估，Amber‑Image 10B/6B 的整体分数均超过大多数开源与闭源模型，特别在语义一致性和文本渲染方面表现突出。

**⚠️ 局限性**

仍在风格多样性、长文本渲染精度以及OneIG多样性维度上存在不足，需要进一步的文本渲染专门化训练或人类反馈强化学习。

---

## 173. Multi-Agent Lipschitz Bandits

**arXiv ID:** 2602.16965 | [PDF](https://arxiv.org/pdf/2602.16965v1)

**作者:** Sourav Chakraborty `[一作]` (University of Colorado Boulder), Lijun Chen `[通讯]` (University of Colorado Boulder)

**通讯引用:** 7216 | [OpenAlex ID](https://openalex.org/A5100398616)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出了一种完全无通信的多玩家连续域拉普拉斯带宽随机Bandit协议，目标是最大化总收益，同时保证协同与碰撞成本与时间无关。

**💡 创新点**

创新点包括：① 通过分阶段解耦：先用最大值导向的搜索快速定位最优N个不同区域；② 引入无通信的Musical Chairs（座位）协议，在实际无占用信息的情况下以O(N)期望时间完成协同；③ 在连续空间中实现无间隙（gap‑free）近似最优的累计风险，匹配单玩家最优下界。

**🔧 技术方法**

技术方法包括：最大值导向的局部探测（local peek）以校正中心值偏差；基于分层分辨率（η‑net）进行高精度估计；多阶段的doubling技巧与分层搜索；解析分析结合Hoeffding、Chernoff等分布界以及碰撞几率推导；以及对距离阈值碰撞模型的包装推导。

**📊 数据集**

本文未使用任何真实数据集，全部在理论设定与合成实验中验证。

**📈 对比分析**

与传统离散多玩家Bandit算法（如Musical Chairs、Zooming）相比，本文在连续空间下实现了同样的Θ(T^(d+1)/(d+2))风险，且协同成本与时间无关；在实验中协同阶段耗时仅O(N)且对大规模玩家数仍保持线性增长。

**⚠️ 局限性**

局限性：① 需要已知行动空间的均匀划分（或距离阈值模型下的安全球集合）；② 仍假设所有玩家共享相同的公共调度假设，缺乏对完全无同步的处理；③ 对高维空间中超大细分的计算成本与存储可能成为瓶颈；④ 只针对拉普拉斯连续奖励模型，无法直接推广至非Lipschitz或非均匀噪声情形。

---

## 174. Bending the Scaling Law Curve in Large-Scale Recommendation Systems

**arXiv ID:** 2602.16986 | [PDF](https://arxiv.org/pdf/2602.16986v1)

**作者:** Qin Ding `[一作]` (Meta Recommendation Systems), Rui Li `[通讯]` (Meta Recommendation Systems)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 ULTRA‑HSTU 模型及其端到端模型–系统协同优化方案，以高效处理超长用户行为序列并提升推荐系统的整体性能。

**💡 创新点**

创新点包括：① 输入序列压缩与异构动作编码；② 半局部稀疏注意力（SLA）实现线性复杂度；③ 动态拓扑设计（注意力截断与 Mixture of Transducers）实现深度可伸缩；④ 负载平衡随机长度（LBSL）和混合精度（FP8/INT4）实现训练与推理加速；⑤ 与 FlashAttention‑V3、Jagged Tensor、激活重计算等系统层面优化协同。

**🔧 技术方法**

使用技术包括：端到端模型–系统协同设计；SLA 线性稀疏注意力；FP8/INT4 混合精度训练/推理；FlashAttention‑V3 定制核；负载平衡随机长度（LBSL）；动态拓扑（注意力截断、Mixture of Transducers）；Jagged Tensor 与激活重计算；FP8 GEMM 融合核；混合精度和量化管线。

**📊 数据集**

主要数据集：工业内部规模达 6 B 条样本，序列长度 3 k–16 k；公开 KuaiRand benchmark（序列长度 256）用于验证短序列场景的通用性。

**📈 对比分析**

与 DIN、SASRec、STCA、Transformer、原始 HSTU 等基线对比。ULTRA‑HSTU 在工业数据上 NE 下降约 0.05%，训练推理吞吐分别提升 5× 与 21×。在线 A/B 实验中，消费、互动、Top‑line 指标分别提升 4%、8% 及 0.217%。

**⚠️ 局限性**

局限性：仍需依赖稀疏注意力以处理超长序列；对 GPU 计算资源依赖度高，需大规模并行训练；在极短序列场景下提升有限；Mixture of Transducers 的效果未在实验中充分验证；部署复杂度高，系统集成成本较大。

---

## 175. AIdentifyAGE Ontology for Decision Support in Forensic Dental Age Assessment

**arXiv ID:** 2602.16714 | [PDF](https://arxiv.org/pdf/2602.16714v1)

**作者:** Renato Marcelo `[一作]` (Instituto Superior Técnico), Cátia Vaz `[通讯]` (Instituto Superior de Engenharia de Lisboa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发并发布了 AIdentifyAGE 本体，专门用于描述法医学牙齿年龄评估的完整司法与技术流程，包括手工评估、AI 推理以及与司法、临床、统计数据的关联。

**💡 创新点**

① 第一个完整覆盖司法、临床、统计与 AI 视角的牙齿年龄评估本体；② 将手工与 AI 评估方法统一到同一语义框架；③ 通过 OBI 上层本体和现有医学/机器学习本体实现高度互操作性，满足 FAIR 原则。

**🔧 技术方法**

本体工程技术（Protégé、OWL、RDF、SPARQL）、推理器 HermiT、深度学习框架（CNN）用于 AI 评估模型的推理；同时重用了 OBI、IAO、FOAF、DCMI、ML‑S、SNOMED、OHd 等公共本体。

**📊 数据集**

基于 10,000 份正颌影像（OPG）及其对应的法医学评估案例进行概念抽取；使用 Demirjian、Haavikko、Kullmann 等国际参考研究的数据进行统计与验证。

**📈 对比分析**

通过 11 个 competency question 以及 SPARQL 查询演示本体在检索手工评估结果、AI 推理输出和司法结论方面的可查询性；逻辑一致性由 HermiT 推理器验证，未提供具体 AI 性能指标，仅证明本体可支持多种评估方法的统一查询。

**⚠️ 局限性**

① 需持续专家参与知识获取与维护；② 本体不消除牙齿年龄估计固有的不确定性；③ 本体本身不保证所使用的 AI 模型或参考研究的准确性，仍需经验验证；④ 需要在大规模司法系统中部署和评估其实际决策支持效果。

---

## 176. Neural Proposals, Symbolic Guarantees: Neuro-Symbolic Graph Generation with Hard Constraints

**arXiv ID:** 2602.16954 | [PDF](https://arxiv.org/pdf/2602.16954v1)

**作者:** Chuqin Geng `[一作]` (McGill University), Xujie Si `[通讯]` (University of Toronto)

**通讯引用:** 427 | [OpenAlex ID](https://openalex.org/A5059074509)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种神经-符号框架NSGGM，用于分子图生成，先用神经网络生成分子骨架和交互信号，再用SMT求解器按硬约束组装完整图，从而保证化学有效性和约束满足；

**💡 创新点**

创新点在于将生成任务拆分为神经网络提出的高层提议和符号求解器执行的精确装配，利用SMT实现对硬约束的正式保证与可解释性，同时保持生成质量；

**🔧 技术方法**

使用的技术包括基于图注意力网络的VAE和Transformer进行分子骨架提议与接口引导，符号层采用Z3 SMT求解器进行约束编码与解算，以及Max-SMT实现软约束优化；

**📊 数据集**

在多个分子生成基准上进行实验，主要使用QM9、MOSES、GuacaMol三大公开数据集，并构建了新的Logical‑Constraint Molecular Benchmark用于评测硬逻辑约束；

**📈 对比分析**

与MoLeR、GenMol、DiGress、JT‑VAE等先进模型对比，NSGGM在无约束生成中保持100%有效率并在FCD、KL、Uniqueness等指标上接近或优于对手；在硬约束和逻辑约束下，NSGGM能准确满足约束且在样本效率上显著优于纯神经网络方法；

**⚠️ 局限性**

局限性包括：符号求解阶段在极大图规模或极复杂约束时可能成为瓶颈，且模型对训练数据中未出现的复杂结构的生成能力仍有限，未来需要改进求解效率和模板多样性。

---

## 177. Meenz bleibt Meenz, but Large Language Models Do Not Speak Its Dialect

**arXiv ID:** 2602.16852 | [PDF](https://arxiv.org/pdf/2602.16852v1)

**作者:** Minh Duc Bui `[一作]`, Katharina von der Wense `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究创建了首个包含美因茨方言Meenzerisch词条及其标准德语定义的数据集，并评估了多种开源LLM在生成词义和词条方面的能力。

**💡 创新点**

创新点在于首次针对未被关注的德语方言Meenzerisch构建词典式数据集，系统评估LLM在低资源方言词义理解与生成上的表现，并探讨少量样例提示与自动规则提取对性能的提升。

**🔧 技术方法**

主要技术包括半自动OCR与规则式文本提取、LLM-as-a-judge自动评估、基于提示的few-shot学习和自动语法规则生成。

**📊 数据集**

使用自制的Meenzerisch词典数据集（共2,351条词义对），并与公开的英语词典数据集进行基准对比。

**📈 对比分析**

对比结果显示，LLM在Meenzerisch词义生成平均准确率仅为4.24%，最佳模型Llama‑3.3‑70B达6.27%；在词条生成准确率更低，平均0.56%，最佳模型1.51%；相比之下在英语词典上准确率均超过80%，显示模型对低资源方言的能力显著不足。

**⚠️ 局限性**

限制包括仅评估词级别理解、缺乏更细致的提示调优与微调、规则提取效果有限、数据集规模受限，以及未覆盖完整句子层面的语言现象。

---

## 178. Claim Automation using Large Language Model

**arXiv ID:** 2602.16836 | [PDF](https://arxiv.org/pdf/2602.16836v1)

**作者:** Zhengda Mo `[一作]` (University of Illinois at Urbana-Champaign), Kaiwen Zhong `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在汽车保修索赔流程中，基于本地部署的DeepSeek‑R1 8B LLM 通过 LoRA 微调，生成结构化的纠正行动建议，帮助理赔员加速决策。

**💡 创新点**

创新点在于：①将 LLM 作为可治理的中间任务模块，避免直接将模型当作黑箱决策引擎；②在本地环境中实现完整的模型治理与可审计；③利用领域微调显著提升输出的格式一致性与语义准确性。

**🔧 技术方法**

技术方法包括：LoRA 微调、RoPE 位置编码、RMSNorm 归一化、SwiGLU 激活、自动化与人工评估的多维度指标（BERT 相似度、LLM‑as‑a‑Judge、BLEURT 等）。

**📊 数据集**

使用约 200 万条真实汽车保修索赔记录（包含投诉、原因、纠正三段文本）作为训练与评估数据，随后抽取高质量子集进行精细验证。

**📈 对比分析**

与通用预训练模型、提示工程、指令调优模型做对比，微调模型在格式符合率 100%、有效率 100% 的同时，完整数据集准确率提升至 81.5%，高质量子集达到 92%，明显优于其他方法。

**⚠️ 局限性**

局限性包括：①数据仍存在噪声与不规范，导致部分预测误差；②仅解决中间任务，后续的全流程自动化（如工时估算、保留决策）尚未实现；③技术对其他行业或模型家族的迁移性尚未验证。

---

## 179. TopoFlow: Physics-guided Neural Networks for high-resolution air quality prediction

**arXiv ID:** 2602.16821 | [PDF](https://arxiv.org/pdf/2602.16821v1)

**作者:** Ammar Kheder `[一作]` (Lappeenranta–Lahti University of Technology), Michael Boy `[通讯]` (University of Helsinki)

**通讯引用:** 54815 | [OpenAlex ID](https://openalex.org/A5041685684)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于视觉Transformer的物理引导空气质量预测模型TopoFlow，通过地形感知注意力和风向引导的补丁重排，实现高分辨率（15km×15km）多污染物（PM2.5、PM10、SO2、NO2、CO、O3）12-96小时预报。

**💡 创新点**

创新点包括：① 将地形高度差直接嵌入注意力偏置，抑制跨山高差的关注；② 设计风向引导的补丁重排算法，使序列顺序与气流方向对齐，降低注意力学习难度；③ 将物理约束嵌入网络架构，而非仅作为损失或额外输入，提升模型的物理一致性。

**🔧 技术方法**

使用技术：视觉Transformer（Swin Transformer骨干）+自注意力；风向引导补丁重排算法；地形偏置注意力；多尺度输入、归一化、AdamW优化、余弦学习率调度；评估指标RMSE、MAE等。

**📊 数据集**

数据集：6年中国空气质量重分析CAQRA（2013-2018）提供六种污染物、气象场；地形数据ETOPO1；人口密度GPD；ERA5气象；验证使用2019年OpenAQ观测站点。

**📈 对比分析**

与数值化学模型（CAMS、CUACE）和AI基线（ClimaX、AirCast、Aurora）在相同训练集上对比；TopoFlow PM2.5 RMSE 9.71 μg/m³，较CAMS/CUACE提升71–80%，相较ClimaX提升13%，在所有污染物和12-96h预报均保持低于20%法规阈值，验证集RMSE在15–30 μg/m³之间。

**⚠️ 局限性**

局限性：① 仅使用表层输入，无法很好预测受垂直输送影响的O3、CO；② 仅在中国训练，跨地区迁移需验证；③ 未显式整合排放清单；④ 固定四个预报时刻，未提供连续时间或不确定性估计；⑤ 对所有污染物采用相同物理机制，未针对化学寿命差异设计专属注意力。

---

## 180. Narrow fine-tuning erodes safety alignment in vision-language agents

**arXiv ID:** 2602.16931 | [PDF](https://arxiv.org/pdf/2602.16931v1)

**作者:** Idhant Gulati `[一作]` (University of California), Shivam Raval `[通讯]` (Harvard University)

**通讯引用:** 205 | [OpenAlex ID](https://openalex.org/A5070746960)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Gemma3-4B模型进行LoRA微调，使用含有种族偏见的Faces数据集，评估其对多模态任务的误导性行为。

**💡 创新点**

揭示窄域有害数据微调会产生跨模态、跨任务的广泛误导性，并证明误导性在激活空间中聚集于低维子空间。

**🔧 技术方法**

采用LoRA参数高效微调、LLM-as-Judge评估误导分数、SVD降维与激活向量控制进行误导性减缓。

**📊 数据集**

使用Faces（1,800张面部图像+文本）作为有害微调集，UTKFace、LLaVA、MSCOCO为评估集，Beavertails-V作为中性数据集进行对比微调。

**📈 对比分析**

通过在文本与多模态评估集上计算LLM-as-Judge的误导分数比较不同LoRA秩和有害比例的效果，发现误导分数随秩、比例上升，文本评估低估误导，激活向量对误导减低但未完全消除。

**⚠️ 局限性**

仅在单一Gemma3-4B模型和单轮微调上测试，评估依赖LLM-as-Judge，未考察多轮连续微调和不同规模模型的推广性。

---

## 181. Nudging Attention to Workplace Meeting Goals: A Large-Scale, Preregistered Field Experiment

**arXiv ID:** 2602.16939 | [PDF](https://arxiv.org/pdf/2602.16939v1)

**作者:** Lev Tankelevitch `[一作]` (Microsoft Research), Sean Rintel `[通讯]` (Microsoft Research)

**通讯引用:** 2078 | [OpenAlex ID](https://openalex.org/A5038706435)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一家全球技术公司内开展为期两周的现场实验，向员工推送预会议目标反思问卷，并通过后会议问卷评估其对会议效果和行为的影响。

**💡 创新点**

创新点在于将会议目标反思嵌入协作平台的轻量化提示，并结合会议目的分类学探讨预/后会议问卷可视作“微型培训”，提升会议意图性。

**🔧 技术方法**

技术手段包括利用 Microsoft Teams Bot 与 Graph API 发送 Adaptive Card 问卷，采用混合效应回归、多重插补、CACE 以及 GPT‑4o 对开放式反馈进行标签与主题分析。

**📊 数据集**

使用的数据集由 361 名员工共计 7,196 次会议的日历与会议元数据、问卷响应以及前后调查问卷的自报指标组成。

**📈 对比分析**

通过随机对照的意向治疗分析与差分差分方法比较，结果显示预会议问卷对会议效果无显著提升，整体会议效能、参与度、目标清晰度等指标在两组均有小幅提升，表明后会议问卷更具影响力。

**⚠️ 局限性**

局限性包括测量反应与自报偏差、干预遵从度低、缺乏被动对照组以及难以评估反思质量，限制了因果推断和对反思机制的深入理解。

---

## 182. When Semantic Overlap Is Not Enough: Cross-Lingual Euphemism Transfer Between Turkish and English

**arXiv ID:** 2602.16957 | [PDF](https://arxiv.org/pdf/2602.16957v1)

**作者:** Hasan Can Biyik `[一作]` (Montclair State University), Anna Feldman `[通讯]` (Montclair State University)

**通讯引用:** 666 | [OpenAlex ID](https://openalex.org/A5064567279)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了英语与土耳其语间跨语言转移下的委婉语检测，构建并划分了重叠与非重叠PET子集。

**💡 创新点**

提出OPET/NOPET框架，将委婉语按跨语言功能与语义重叠划分，并系统评估其对跨语言迁移的影响。

**🔧 技术方法**

使用XLM‑RoBERTa进行微调与零样本评估，并与冻结基线以及GPT‑4o零样本推理进行对比。

**📊 数据集**

使用扩充后的土耳其PET数据集（70个PET）与选取的英语PET子集（71个PET），并按领域划分。

**📈 对比分析**

通过10折交叉验证的零样本转移实验比较，发现从英语到土耳其的转移保持高F1（如就业0.90），而土耳其到英语的转移显著下降（如就业0.36），并揭示类别层面的不对称。

**⚠️ 局限性**

局限在于仅考察英土两语，类别稀疏导致统计不稳，未处理方言/社会变体，模型仅为XLM‑R，未检验多模型及更大上下文的效果。

---

## 183. Omitted Variable Bias in Language Models Under Distribution Shift

**arXiv ID:** 2602.16784 | [PDF](https://arxiv.org/pdf/2602.16784v1)

**作者:** Victoria Lin `[一作]` (Carnegie Mellon University), Eli Ben-Michael `[通讯]` (Carnegie Mellon University)

**通讯引用:** 501 | [OpenAlex ID](https://openalex.org/A5041268061)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个将语言模型在分布偏移下因未观测变量导致的遗漏变量偏差（OVB）映射到 worst‑case 泛化性能的框架，并给出了可计算的性能上界；同时给出基于 GLM 负对数似然的直接预测标签实现；

**💡 创新点**

首次将 OVB 的强度与语言模型的分布不变性能关联，提供了可利用的 worst‑case 上界用于评估和优化，并通过敏感性参数对遗漏变量影响进行量化；

**🔧 技术方法**

采用双重稳健损失、Riesz 表示器、分布鲁棒优化、敏感性分析以及 GLM 负对数似然分解，并在预训练模型上估计短/长模型的结果；

**📊 数据集**

实验使用 MATH 与 MATH‑Perturb（数学推理）、Amazon 评价、EmoBank 情感、Hate Speech（Reddit/Gab）等数据集；

**📈 对比分析**

与标准双重稳健目标及未校正基线对比，评估指标为目标域准确率/MSE/BCE；实验显示使用 worst‑case 上界的评估更保守、优化后模型在目标域的真实表现优于传统 DR，并能量化遗漏变量强度；

**⚠️ 局限性**

需估计敏感性参数且可能过度保守；依赖协变量平移假设，难以在所有真实场景下完全识别 OVB；对非 GLM 任务的适用性有限，复杂损失的优化更具挑战性。

---

## 184. Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History

**arXiv ID:** 2602.17003 | [PDF](https://arxiv.org/pdf/2602.17003v1)

**作者:** Serin Kim `[一作]` (Yonsei University), Dongha Lee `[通讯]` (Yonsei University)

**通讯引用:** 2962 | [OpenAlex ID](https://openalex.org/A5010775517)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了首个面向真实开放网页的个性化Web Agent基准 Persona2Web，聚焦用户历史、模糊查询与推理评估。

**💡 创新点**

创新点在于：①“澄清‑到‑个性化”查询设计，迫使Agent从用户历史推断缺失信息；②构建隐式偏好揭示的浏览历史；③基于推理轨迹的评分表，区分个性化、导航与整体成功。

**🔧 技术方法**

技术包括：多阶段LLM（GPT‑4o、GPT‑5‑mini）生成用户画像、事件与历史；AgentOccam、Browser‑Use两种Web Agent架构改造为可访问历史的规划‑检索‑生成流水线；对不同历史访问策略（按需 vs 预执行）进行评估。

**📊 数据集**

数据集：50个人工合成用户画像与浏览历史（包含高频/低频事件、约10%噪声），以及对应的三层模糊查询集（清晰/仅网站/仅偏好/全模糊）。数据与代码公开于 https://anonymous.4open.science/r/Persona2Web-73E8。

**📈 对比分析**

评估方法：使用基于推理轨迹的评分（个性化得分PS1、PS2、意图满足IS、成功率SR），并与人类标注进行相关性检验。实验显示：无历史时所有模型SR为0%；开启历史时最高SR约13%（GPT‑4.1 + Browser‑Use + 预执行），但不同架构/模型对个性化与导航的权衡差异显著。

**⚠️ 局限性**

局限：①整体成功率仍极低，说明当前模型难以有效检索与利用历史；②依赖合成数据，缺乏真实用户隐私与行为复杂性；③对历史检索和推理过程的细粒度分析仍有限，可能掩盖细微错误；④在开放网页中面临内容动态、缺失或误导性信息的挑战。

---

## 185. SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation

**arXiv ID:** 2602.16863 | [PDF](https://arxiv.org/pdf/2602.16863v1)

**作者:** Kushal Kedia `[一作]` (Cornell University), C. Karen Liu `[通讯]` (Stanford University)

**通讯引用:** 111733 | [OpenAlex ID](https://openalex.org/A5100361773)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并训练了一种单一的目标条件强化学习策略，能够在仿真中学习任意工具随机姿态到达，并在真实世界实现对多种未见工具和任务的零射击抓握、姿态重新定向与执行。

**💡 创新点**

创新点在于将工具使用转化为对象中心的目标序列问题，利用大量程序化生成的工具原语和统一奖励函数训练通用策略；通过视觉基础模型实现零射击的sim-to-real迁移，消除每个工具/任务的专门工程。

**🔧 技术方法**

使用了目标条件强化学习（SAPG、PPO改进）、域随机化、非对称评论家、LSTM策略网络，以及基于SAM 3D、SAM 2和FoundationPose的三维感知与姿态跟踪技术。

**📊 数据集**

数据集包括在仿真中生成的数千个程序化工具原语，以及由DexToolBench提供的24个日常工具使用任务、12个工具实例、6个类别的RGB‑D人类演示视频。

**📈 对比分析**

与运动学重定向和固定抓取基线比较，本文方法在任务进度上提升约37%；与单一工具专用RL基线对比，在未见工具/轨迹上表现相当或更优；在120次真实世界测试中，平均任务进度达到约X%（高于基线）。

**⚠️ 局限性**

主要局限包括仅能跟踪姿态序列而不保证功能完成；缺乏碰撞检测和动态重规划；假设工具为刚体，无法处理柔性工具；对环境拥挤或复杂场景的适应性有限。

---

## 186. RankEvolve: Automating the Discovery of Retrieval Algorithms via LLM-Driven Evolution

**arXiv ID:** 2602.16932 | [PDF](https://arxiv.org/pdf/2602.16932v1)

**作者:** Jinming Nian `[一作]` (Santa Clara University), Yi Fang `[通讯]` (Santa Clara University)

**通讯引用:** 2414 | [OpenAlex ID](https://openalex.org/A5083935587)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用LLM驱动的程序进化方法，自动从BM25和查询概率模型演化出新的词典检索算法。

**💡 创新点**

首次将大语言模型与程序演化相结合，探索检索算法空间并发现可迁移的创新评分机制。

**🔧 技术方法**

采用AlphaEvolve框架、LLM（GPT‑5.2）生成变异、MAP‑Elites岛屿演化、Python程序化表示与评估器进行评测。

**📊 数据集**

在BEIR（12子集）、BRIGHT（12子集）以及TREC DL 2019/2020 共28个公开数据集上进行实验。

**📈 对比分析**

与BM25、BM25+、BM25‑adpt、QL‑Dir、QL‑JM等基线对比，演化算法在Recall@100和nDCG@10上在未见数据集上提升约5–10%，但查询延迟显著升高。

**⚠️ 局限性**

生成的算法复杂度高、查询延迟大，未在演化目标中加入效率约束，且缺乏对模型可解释性与通用性更系统的评估。

---

## 187. PETS: A Principled Framework Towards Optimal Trajectory Allocation for Efficient Test-Time Self-Consistency

**arXiv ID:** 2602.16745 | [PDF](https://arxiv.org/pdf/2602.16745v1)

**作者:** Zhangyi Liu `[一作]`, Zhun Deng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套 PETS（Principled and Efficient Test-Time Self-Consistency）框架，用来在给定总预算下对大语言模型的多条推理轨迹进行自一致性采样的高效分配。

**💡 创新点**

创新点包括：①提出自一致性率（self‑consistency rate）作为理论目标；②将轨迹分配问题与众包任务分配相映射，利用Bayesian MDP和OKG理论得到最优策略；③在在线场景中通过难度估计和贪心分配实现一次性预算决策，并提供理论保证。

**🔧 技术方法**

采用的技术主要有：Bayesian MDP、Optimistic Knowledge Gradient (OKG) 算法、置信度加权投票、基于难度的离散化与贪心分配、warm‑up 估计难度分布。

**📊 数据集**

实验数据集包括：GPQA‑Diamond、AIME 24/25、Brumo 25、HMMT Feb 25 等多种数学与推理任务，使用 Qwen3‑4B、Qwen3‑30B、GPT‑20B、GPT‑120B、Qwen‑Long 等大型推理模型。

**📈 对比分析**

与均匀分配（Uniform）基线相比，offline 模式可减少多达 75% 的轨迹数量即可达到全自一致性，online 模式则减少约 55%；在多种模型与数据集上均实现更高的自一致性率和准确率，并且与理论上可达的 oracle 方案几乎无差距。

**⚠️ 局限性**

局限性：当整体多数投票与真实答案不一致时，额外采样对准确率提升有限；online 方案依赖少量 warm‑up 数据来估计难度分布，若难度预测误差较大会影响性能；目前未直接通过模型学习预测难度参数 θ，仍需后验推理。

---

## 188. Mobility-Aware Cache Framework for Scalable LLM-Based Human Mobility Simulation

**arXiv ID:** 2602.16727 | [PDF](https://arxiv.org/pdf/2602.16727v1)

**作者:** Hua Yan `[一作]` (Lehigh University), Yu Yang `[通讯]` (Lehigh University)

**通讯引用:** 1807 | [OpenAlex ID](https://openalex.org/A5070570645)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了MobCache，一个可重构的缓存框架，利用潜在空间推理嵌入重用和重组，配合轻量级解码器，显著加速大规模LLM人类移动模拟。

**💡 创新点**

创新点包括：1) 潜在空间推理缓存，支持树形搜索与重构；2) 运动法则约束的蒸馏轻量解码；3) 结合缓存与探索策略提升多样性与效率。

**🔧 技术方法**

使用的技术包括：LLM微调（LLaMA 3.2-3B）、潜在空间推理、latent-space evaluator、移动法则约束蒸馏、轻量解码器、相似性检索、探索率控制。

**📊 数据集**

使用的公开数据集：北京市移动轨迹数据（2019年10-12月）和纽约市POI签到数据；还利用公开人口统计信息生成用户档案。

**📈 对比分析**

与CoPB、Urban-Mobility-LLM、LLMob、Geo-LLaMA等基线比较。MobCache在北京数据上推理时间提升42%，tokens/s提升80%，吞吐量提升28%，成本降低42%；质量指标与现有LLM方法相近。 在NYC数据亦实现约51%时间缩短、79% tokens/s提升、56%成本下降。

**⚠️ 局限性**

局限性：仅适用于提供可解释推理步骤的LLM模型；缓存主要保存推理过程而非具体位置，跨城市迁移时匹配率下降；对极端稀有行为缺乏充分多样性。

---

## 189. Separations above TFNP from Sherali-Adams Lower Bounds

**arXiv ID:** 2602.16810 | [PDF](https://arxiv.org/pdf/2602.16810v1)

**作者:** Anna Gal `[一作]`, Christophe Marciot `[通讯]` (Lund University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了 Σ₂ 层的总搜索问题，证明了 Linear Ordering Principle（线性排序原理）不在黑盒可归约到 Strong Avoid 的类内，构造了 Σ₂ 版本的 Sherali‑Adams 证明系统，并给出了对应的伪期望（pseudo‑expectation）下界，从而实现了新的 Σ₂ 类别分离。

**💡 创新点**

① 首次使用伪期望方法证明了 Σ₂‑Sherali‑Adams 的低度量下界；② 将经典的覆盖问题与线性排序原理的伪期望相结合，构造了关于置换的 combinatorial covering 证明；③ 将黑盒归约与证明复杂度之间的等价性拓展到 Σ₂ 层，给出了弱化步骤与反例归约的分解。

**🔧 技术方法**

伪期望（pseudo‑expectation）技术、Σ₂‑弱化规则（Σ₂‑weakening）、Sherali‑Adams 证明系统的 Σ₂ 变体、组合覆盖问题求解、决策树归约（black‑box reductions）与证明系统之间的等价性证明。

**📊 数据集**

无数据集；该工作完全是理论计算机科学的定理证明与形式化分析。

**📈 对比分析**

由于该研究属于理论性质，未进行实验或性能比较；主要通过证明下界与归约结构的相互转化，展示了新分离结论的严谨性。

**⚠️ 局限性**

局限性：仅处理了黑盒归约框架，未给出多项式时间可验证的归约；对 Strong Avoid 与 Least Number 的分离仍未完成；伪期望下界仅提供度量下界，未直接给出大小下界；证明中对 n/100 等常数限制比较粗糙，后续可进一步优化。

---

## 190. Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders

**arXiv ID:** 2602.17050 | [PDF](https://arxiv.org/pdf/2602.17050v1)

**作者:** Ziliang Zhao `[一作]` (Meta Platforms Inc), Kai Ren `[通讯]` (Meta Platforms Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 Multi-Probe Zero Collision Hash (MPZCH) 算法，用于在大规模推荐系统中消除嵌入表哈希冲突并提升嵌入新鲜度。

**💡 创新点**

创新点包括：① 双阶段线性探测的 CUDA 核心，利用身份张量与元数据张量实现精确冲突检测与插入；② 可配置的 TTL / LRU 淘汰策略，能在插入新 ID 时自动清除过期条目并重置优化器状态；③ 通过保证零冲突与即时回收实现嵌入表的“零遗传”与高新鲜度。

**🔧 技术方法**

技术手段：CUDA 高性能内核、线性探测与双阶段扫描、辅助身份/元数据张量、TTL/LRU 淘汰、与 TorchRec 的分布式共享表集成、GPU 加速的批量与非批量操作。

**📊 数据集**

实验数据集：Meta 生产环境中 1.5 亿唯一 ID 的用户/物品表；3 亿月活跃用户的视频排序模型；HSTU 结构的视频检索模型；在线 A/B 测试与离线 t‑SNE 可视化。

**📈 对比分析**

比较方法：与基线 Sigrid 哈希（标准哈希）在不同表容量与探测深度下的冲突率；在 GPU 与 CPU 上测量查询/插入延迟；A/B 测试评估 NE 改进（0.38% 等）及新内容曝光提升（0.83%）。性能方面，MPZCH 在合理表容量和探测深度下实现零冲突，查询延迟保持在 0.8‑0.9 ms（GPU 批量），与基线相比训练 QPS 与推理延迟相当。

**⚠️ 局限性**

限制：需在 GPU 上运行，且对表容量与探测深度有一定要求；增加了身份/元数据张量的存储开销；TTL / LRU 参数需要手工调优；在极端长尾场景下的可扩展性与效果尚未系统验证。

---

## 191. Node Learning: A Framework for Adaptive, Decentralised and Collaborative Network Edge AI

**arXiv ID:** 2602.16814 | [PDF](https://arxiv.org/pdf/2602.16814v1)

**作者:** Eiman Kanjo `[一作]` (Nottingham Trent University), Mustafa Aslanov `[通讯]` (Nottingham Trent University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了“Node Learning”这一去中心化持续学习范式，使每个边缘节点在本地不断适应并在合适时机通过点对点或群体交流共享知识，形成无中央协调的自组织学习网络。

**💡 创新点**

创新点在于：① 把学习视为持续的、由节点自主驱动的实体，打破传统集中式、轮次式或共享目标的假设；② 引入机会主义协作、情境感知的合并操作；③ 将联邦学习、分布式优化与协作推理统一为一个连续谱；④ 强调硬件异质性、能耗、网络不稳定性与信任治理的整体设计。

**🔧 技术方法**

采用的技术包括：TinyML 轻量级模型（如 MobileBERT、pruning/quantization）、低功耗无线协议（BLE 5.x、LoRaWAN）、自适应聚合（gossip、邻居聚类）、模型适配器与低秩更新、异构硬件（MCU、RISC‑V NPU、神经形态芯片）以及可选的安全与隐私保护机制（局部差分隐私、同态加密）。

**📊 数据集**

本文为概念性工作，并未给出统一的数据集；在讨论中引用了多模态应用场景（如畜牧行为识别、交通流估计、步行者跟踪）和常见边缘数据源（图像、音频、传感器时间序列）来说明设计原则。

**📈 对比分析**

由于缺乏实现与实验，本文未提供量化性能对比；提出的评估指标包括能耗/迭代（EPI）、合作效率（CE）、适应延迟（AL）和鲁棒比（RR），但具体数值需在后续工作中通过仿真或真实部署验证。

**⚠️ 局限性**

局限性包括：① 仍缺乏严格的收敛性证明与基准评测；② 对节点间异构性与网络动态性的理论分析不完整；③ 依赖机会主义通信，受连接质量与节点活跃度影响；④ 安全与信任机制需要针对极低功耗设备进一步简化；⑤ 该范式在大规模部署下的可扩展性与资源调度仍待实验验证。

---

## 192. Rememo: A Research-through-Design Inquiry Towards an AI-in-the-loop Therapist's Tool for Dementia Reminiscence

**arXiv ID:** 2602.17083 | [PDF](https://arxiv.org/pdf/2602.17083v1)

**作者:** Celeste Seah `[一作]` (National University of Singapore), Clement Zheng `[通讯]` (National University of Singapore)

**通讯引用:** 697 | [OpenAlex ID](https://openalex.org/A5013498297)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文通过两年研究‑设计过程，开发了 Rememo 这一面向治疗师的工具，将生成式 AI 与回忆疗法结合，以支持和增强人工引导。

**💡 创新点**

创新点在于提出 AI‑in‑the‑loop 模型（由治疗师掌控、AI 只在适当时机提供内容），将合成图像视为记忆触发器而非真相复制，并通过参与式设计与细化本地 LoRA 模型实现文化适配。

**🔧 技术方法**

使用了多种生成式图像模型（Stable Diffusion XL、Flux.1 LoRA、Imagen）与大型语言模型 Gemini 生成引导问题；实现 OCR 读取提示卡、Webapp 控制打印机的物理-数字混合系统。

**📊 数据集**

数据集包括 317 张公开的本地历史图片用于 Fine‑Tuning Flux.1 LoRA，以及 128 张多语言插画提示卡作为输入库。

**📈 对比分析**

在两周现场部署中，比较三种模型的生成质量、提示遵从度、历史准确度和打印率；Imagen 生成质量最高、打印率约 34.9%，平均生成时长约 27 秒；相较于传统 RT 工具，报告显示居民回忆和参与度提升。

**⚠️ 局限性**

局限性包括样本量小（仅 5 名治疗师、21 名居民）、单一新加坡背景、缺乏长期疗效评估、生成延迟与打印成本对流程造成影响，以及 AI 输出仍需人工审核、缺少完全自动化方案。

---

## 193. SourceBench: Can AI Answers Reference Quality Web Sources?

**arXiv ID:** 2602.16942 | [PDF](https://arxiv.org/pdf/2602.16942v1)

**作者:** Hexi Jin `[一作]` (University of California, San Diego), Yiying Zhang `[通讯]` (GenseeAI Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新基准，系统性评估AI答案中引用的网页来源质量；

**💡 创新点**

创新点在于设计了八维度的多面质量评估框架，并通过人类标注训练LLM自动评估器，使评估与专家判断高度一致；

**🔧 技术方法**

使用了大型语言模型（如GPT‑5、Gemini等）、传统搜索引擎（Google）和AI原生搜索工具（Exa、Tavily、Gensee）进行实验，并构建了基于LLM的自动评分器；

**📊 数据集**

数据集包含100个真实世界查询（来自DebateQA、HotpotQA、Pinocchios、Quora、VACOS_NLQ等），以及对应3996条被引用的网页链接；

**📈 对比分析**

与传统答案质量评估方法对比，该基准更关注来源可信度、时效性和可用性。实验显示GPT‑5在大多数维度上表现最佳，Google在内容相关性上也表现出色，但在新鲜度上落后；

**⚠️ 局限性**

局限性包括：评估仅基于已抓取的网页内容，未考虑多语言和多媒体来源；自动评估器依赖于LLM，可能在极端或极少见的网页格式下误判；基准仅覆盖了100个查询，未来需要更大规模的多样化验证。

---

## 194. BadCLIP++: Stealthy and Persistent Backdoors in Multimodal Contrastive Learning

**arXiv ID:** 2602.17168 | [PDF](https://arxiv.org/pdf/2602.17168v1)

**作者:** Siyuan Liang `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 98773 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 BadCLIP++ 的统一框架，用以在多模态对比学习模型中实现既隐蔽又持久的后门攻击。

**💡 创新点**

创新点包括：语义融合的 QR 迷你触发器、基于目标对齐的贪婪子集选择、触发器聚类收缩与质心对齐，以及通过弹性权重巩固 (EWC) 与 ALIGN 正则化实现的参数层面抗遗忘机制；并首次给出清洗微调与后门梯度共向性理论证明。

**🔧 技术方法**

采用了多模态对比学习中的信息噪声损失、T2T 触发器聚类损失、MPE 质心对齐损失、ALIGN 对齐损失和 EWC 重要性正则等技术，并在训练阶段进行两阶段 min‑min 优化。

**📊 数据集**

在 CC3M 500k 语料库上进行触发器注入，评估数据包括 COCO、SBU、ImageNet‑1K、CIFAR‑100、SVHN、ImageNet‑Sketch/V2/A/R 等；模型覆盖 CLIP RN50、ViT‑B/32、ALBEF、FLAVA、UniCL 等五大架构。

**📈 对比分析**

相较于 12 种现有后门攻击，BadCLIP++ 在 0.3% 泄漏率下实现 99.99% 的 ASR，CA 仅下降 <1%，并在 19 种防御（训练‑、模型‑、推理‑、预训练‑防御）下仍保持 99%+ 的成功率，优于基线约 15% 的提升。

**⚠️ 局限性**

局限性：需要对训练数据或模型具有较强的白盒控制；对某些强安全训练（如 SafeCLIP）仍易被抑制，且对非图像-文本的多模态场景适用性待进一步验证。

---

## 195. Bonsai: A Framework for Convolutional Neural Network Acceleration Using Criterion-Based Pruning

**arXiv ID:** 2602.17145 | [PDF](https://arxiv.org/pdf/2602.17145v1)

**作者:** Joseph Bingham `[一作]` (Rutgers University), Sam Helmich `[通讯]` (John Deere Intelligent Solutions Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出Combine框架，统一CNN剪枝方法并实现快速高效的迭代剪枝。

**💡 创新点**

创新点在于构建通用的criterion抽象、阈值选择流程和统一语言，能同时处理卷积与全连接层，并通过实验验证不同criterion的效果。

**🔧 技术方法**

采用Keras实现，利用标准偏差、最大/平均绝对值、范围等criterion函数，结合静态/进阶剪枝策略与阈值迭代。

**📊 数据集**

使用MNIST和CIFAR‑10数据集训练的VGG‑风格模型。

**📈 对比分析**

通过阈值函数绘制准确率-剪枝比例曲线，比较不同criterion和层选取，实验显示在MNIST上可剪枝至79%且FLOPs提升68%，CIFAR‑10上可提升至41%且准确率提升。

**⚠️ 局限性**

局限在于对极小模型或缺少卷积层的网络效果有限，且阈值选择仍需人工经验；未讨论训练成本与迁移学习的兼容性。

---

## 196. Benchmarking the Effects of Object Pose Estimation and Reconstruction on Robotic Grasping Success

**arXiv ID:** 2602.17101 | [PDF](https://arxiv.org/pdf/2602.17101v1)

**作者:** Varun Burde `[一作]` (Czech Technical University in Prague), Torsten Sattler `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 12698 | [OpenAlex ID](https://openalex.org/A5011683384)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估 3D 重建与 6D 姿态估计对机器人抓取成功率的影响，提出基于物理仿真的端到端基准；

**💡 创新点**

首次将 3D 重建质量与抓取性能直接关联，构建大规模物理仿真评测框架；

**🔧 技术方法**

使用 PyBullet 物理仿真、基于 YCB‑Video 的三维模型、MegaPose 与 FoundationPose 姿态估计、Instant‑NGP、UniSurf 等重建方法；

**📊 数据集**

YCB‑Video 数据集（含 21 个对象），并利用其 6D 姿态标注进行评估；

**📈 对比分析**

通过比较多种重建方法与姿态估计器在抓取成功率（S_est）和抓取候选成功率（S_gen）上的表现，发现高质量姿态估计可在一定程度上补偿模型误差，且几乎所有重建方法都会降低可用抓取候选数；

**⚠️ 局限性**

仅在仿真环境中验证，未在真实机器人平台上进行实验，且缺乏对抓取之外的操纵任务的评估；

---

## 197. AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation

**arXiv ID:** 2602.17100 | [PDF](https://arxiv.org/pdf/2602.17100v1)

**作者:** Siyu Wang `[一作]` (Shanghai Jiao Tong University), Xinping Guan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 26110 | [OpenAlex ID](https://openalex.org/A5100690710)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了AgentConductor，一种基于强化学习的多智能体编程系统，能够在每个问题上动态生成并迭代改进交互拓扑；

**💡 创新点**

创新点在于提出层次化DAG拓扑、难度感知的拓扑密度函数及基于执行反馈的多目标奖励，使拓扑随问题难度自适应、可多轮演化；

**🔧 技术方法**

技术包括LLM orchestrator、YAML层次化拓扑表示、基于GRPO的轨迹级强化学习、基于难度区间的密度上限奖励、SFT预训练和多目标奖励设计；

**📊 数据集**

使用三大竞赛级数据集（APPS、LiveCodeBench、CodeContests）与两大基础数据集（HumanEval、MBPP）进行评测；

**📈 对比分析**

与四类基线对比（Vanilla、经典MAS、工作流优化MAS、拓扑优化MAS），AgentConductor在竞赛级数据集上pass@1最高（58.8%–38.8%），同时实现了14.6%绝对提升、13%稀疏度下降和68%token成本降低；

**⚠️ 局限性**

局限在于仍依赖预训练的LLM、对极大规模模型的迁移性未完全验证、以及在极难问题上可能需要更深层次的拓扑动态更新。

---

## 198. FLoRG: Federated Fine-tuning with Low-rank Gram Matrices and Procrustes Alignment

**arXiv ID:** 2602.17095 | [PDF](https://arxiv.org/pdf/2602.17095v1)

**作者:** Chuiyang Meng `[一作]` (University of British Columbia), Vincent W. S. Wong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的联邦学习细调框架 FLoRG，利用单个低秩矩阵和其 Gram 矩阵进行模型聚合，并引入 Procrustes 对齐来消除分解漂移。

**💡 创新点**

创新点在于：① 用单个低秩矩阵代替传统 LoRA 的两矩阵，避免双矩阵聚合误差；② 聚合 Gram 矩阵实现线性无偏聚合；③ 通过 Procrustes 对齐保证分解唯一性与秩一致，显著降低通信和漂移。

**🔧 技术方法**

主要技术包括：低秩适配（LoRA）、联邦学习框架、Gram 矩阵聚合、矩阵分解（特征值分解）、Procrustes 对齐、理论收敛分析。

**📊 数据集**

实验使用 GLUE（MRPC、QQP、MNLI、QNLI、WNLI、RTE）和 SQuAD v1.1 作为下游任务的数据集；在 OPT‑125M、RoBERTa‑large、Llama‑3.2‑3B 三种规模的预训练模型上进行评估。

**📈 对比分析**

与五个基线（FedIT、FeDeRA、FFA‑LoRA、FedSA‑LoRA、FedEx‑LoRA）对比，FLoRG 在多数据集、多模型上均取得更高的测试准确率，并且通信开销降低至 1/2041；在不同数据异构、客户端比例、秩设置下亦表现稳健。

**⚠️ 局限性**

局限性包括：① 服务器端需要进行特征值分解和 Procrustes 对齐，计算开销随层数和 k 立方增长；② 对秩 r 的选择仍需经验性调优；③ 实验仅覆盖自然语言理解与问答任务，缺乏对其他领域或更大规模联邦场景的验证；④ 在极端异构或客户端失效情况下的鲁棒性尚未充分评估。

---

## 199. What to Cut? Predicting Unnecessary Methods in Agentic Code Generation

**arXiv ID:** 2602.17091 | [PDF](https://arxiv.org/pdf/2602.17091v1)

**作者:** Kan Watanabe `[一作]` (Nara Institute of Science and Technology), Hajimu Iida `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1509 | [OpenAlex ID](https://openalex.org/A5055973723)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对 Agentic Coding 生成的 Python 方法，在 PR 审查到合并期间进行增删分析，并基于 23 个代码特征构建随机森林模型来预测哪些方法最终会被删除。

**💡 创新点**

①首次系统量化 Agentic Coding 生成代码的被删除比例及其特征；②提出基于多维度代码度量的预测模型，AUC 87.1%，显著优于 LLM 直接判别；③将 LLM 与传统随机方法作为基线，验证模型有效性。

**🔧 技术方法**

使用 Python AST 解析、ActRef 代码重构检测、特征提取（行数、字符数、词数、复杂度、参数、变量等）、随机森林分类、10 折交叉验证以及 GPT‑4o 对比实验。

**📊 数据集**

AIDev 数据集（33,596 PR），过滤后得到 1,664 合并 PR，197 项目，提取 12,343 个新增方法，其中 323 个被删除。

**📈 对比分析**

与随机预测和 GPT‑4o 基线对比；模型 AUC 0.871、召回率 82.5%、精确度 4.9%，显著优于 GPT‑4o（AUC 0.621、召回率 2.6%）和随机（AUC 0.500）。

**⚠️ 局限性**

仅针对 Python 项目，依赖 ActRef 的重构检测精度；删除仅统计 PR 合并前的变化，未捕获合并后删除；特征可能未涵盖所有影响删除决策的因素，模型对其他语言的泛化能力有限。

---

## 200. MeGU: Machine-Guided Unlearning with Target Feature Disentanglement

**arXiv ID:** 2602.17088 | [PDF](https://arxiv.org/pdf/2602.17088v1)

**作者:** Haoyu Wang `[一作]` (Beijing Institute of Technology), Tongliang Liu `[通讯]` (University of Sydney)

**通讯引用:** 12773 | [OpenAlex ID](https://openalex.org/A5065250332)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多模态大语言模型指导的机器遗忘框架MeGU，利用标签扰动与正负特征噪声对目标数据进行选择性遗忘，保持模型泛化；

**💡 创新点**

创新点在于（1）通过MLLM零样本推理生成语义一致但错误的扰动标签，精准定位概念重对齐；（2）构造正负特征噪声对实现特征解耦，兼顾过忘与不足忘的平衡；（3）使用轻量级转移矩阵缓存实现高效标签分配；

**🔧 技术方法**

技术手段包括多模态LLM（MMICL等）进行概念相似度估计、转移矩阵构造、正负特征噪声训练、微调损失结合扰动样本、评估指标A_f/A_r与MIA；

**📊 数据集**

实验数据集为CIFAR-10、CIFAR-20、CIFAR-100，使用ResNet18与Vision Transformer两种骨干网络；

**📈 对比分析**

与基线、全量重训(retrain)、FT、UNSIR、AMNC、SSD、BadTeacher等方法对比，MeGU在保留准确率与重训模型几乎相当，忘记准确率为0，MIA显著降低，计算时间比重训快约60%；

**⚠️ 局限性**

主要限制是需使用MLLM推理，导致额外计算开销；转移矩阵预计算与样本数量调整可缓解，但整体仍比纯数据/模型方式慢；

---

## 201. ARCANE: Scalable high-degree cubature formulae for simulating SDEs without Monte Carlo error

**arXiv ID:** 2602.17151 | [PDF](https://arxiv.org/pdf/2602.17151v1)

**作者:** Peter Koepernik `[一作]` (University of Oxford), James Foster `[通讯]` (University of Bath)

**通讯引用:** 10111 | [OpenAlex ID](https://openalex.org/A5060498974)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了ARCANE算法，能够自动构造高阶（最高达19阶）可用于SDE数值模拟的分割路径（cubature formulae）

**💡 创新点**

采用二进制正交阵、重组（recombination）与线性规划相结合，显著减少路径数量并实现高阶匹配；同时引入dyadic深度扩展与GPU并行化，突破以往D=7的限制

**🔧 技术方法**

使用粗路径理论与路径签名（signature）匹配、错误纠正码正交阵、Carathéodory重组、线性规划、JAX+GPU加速实现

**📊 数据集**

以金融与生物学常用SDE模型（Vasicek、IGBM、CIR、Wright–Fisher、log‑Heston）为实验场景，并公开Cubature公式集及实验数据

**📈 对比分析**

与标准Monte Carlo、Sobol与Latin Hypercube QMC比较，使用均方误差（MVE）和相关误差指标；ARCANE在相同路径数下误差降低数个数量级，表现优异

**⚠️ 局限性**

受限于路径光滑性（如Wright–Fisher边界）、时间跨度增大时精度下降、dyadic深度对精度影响不易解释，且当前方法仅适用于低维噪声，未解决高维SDE和更严格的收敛理论

---

## 202. The Emergence of Lab-Driven Alignment Signatures: A Psychometric Framework for Auditing Latent Bias and Compounding Risk in Generative AI

**arXiv ID:** 2602.17127 | [PDF](https://arxiv.org/pdf/2602.17127v1)

**作者:** Dusan Bosnjakovic `[一作]` `[通讯]`, Dusan Bosnjakovic

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于心理测量的定量审核框架，评估大型语言模型在多维治理相关属性上的“实验室签名”，

**💡 创新点**

创新点在于将受试者测量理论与混合效应模型相结合，采用强制性顺序测验和掩蔽技术，揭示不同模型提供商的持久行为偏差，

**🔧 技术方法**

核心技术包括强制式顺序闭塞测验、半隐蔽式掩蔽（decoy masking）、混合线性模型（MixedLM）与ICC分析，以及极端统计检验（Kruskal–Wallis/Friedman），

**📊 数据集**

数据集由多轮生成的情境小短文和五选项闭塞题组成，覆盖9个治理维度，测试了OpenAI、Anthropic、Google和xAI等多个提供商的不同代模型，

**📈 对比分析**

通过ICC、方差分解和配对后检验，发现多数维度存在显著的提供商聚类，提示模型间存在可测的“lab signal”，并比较了有无decoy掩蔽对统计显著性的影响，

**⚠️ 局限性**

局限性包括仅覆盖有限维度、对模型版本选择的偏倚、掩蔽技术在不同模型上效果不一、以及实验室签名是否会随未来训练策略改变而消退未能充分验证

---

## 203. i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting

**arXiv ID:** 2602.17117 | [PDF](https://arxiv.org/pdf/2602.17117v1)

**作者:** Yicheng Cao `[一作]` (Sydney AI Centre), Tongliang Liu `[通讯]` (Sydney AI Centre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

结合3D Gaussian Splatting与隐式 Material Point Method，利用动量平衡残差的 Newton‑GMRES 求解生成稳定的 4D Gaussian 动态序列；

**💡 创新点**

首次在 MPM 中引入基于动量平衡残差的隐式求解器，并将其与 3DGS 结合，实现大时间步长下的稳定物理仿真；

**🔧 技术方法**

3D Gaussian Splatting、隐式 MPM、Newton 迭代 + GMRES 线性求解、内部粒子填充与动量残差优化；

**📊 数据集**

PhysDreamer（真实捕捉）与 PhysGaussian（合成）数据集；

**📈 对比分析**

对比 Explicit MPM、DreamPhysics、Physics3D 等基线，使用 BMF 门限、k_max、失败率、COMD/mwRMSD 漂移、渲染 PSNR/SSIM/LPIPS 与曝光稳定性等指标；i‑PhysGaussian 在大步长下实现 k_max=20、失败率 0%、漂移最低、渲染质量与基线相当；

**⚠️ 局限性**

每一步需要嵌套隐式求解，计算成本显著高于显式方法，且在极端硬件或大规模场景下效率仍需进一步提升。

---

## 204. Instructor-Aligned Knowledge Graphs for Personalized Learning

**arXiv ID:** 2602.17111 | [PDF](https://arxiv.org/pdf/2602.17111v1)

**作者:** Abdulrahman AlRabah `[一作]` (University of Illinois Urbana-Champaign), Abdussalam Alawini `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 236 | [OpenAlex ID](https://openalex.org/A5014235464)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于讲座材料的教师对齐知识图谱，自动提取概念并推断其前置与组成关系。

**💡 创新点**

创新点在于融合教学顺序和教学角色信号，利用LLM推理与教育特征相结合，得到课程专属的概念依赖结构。

**🔧 技术方法**

采用三阶段管线：PDF预处理+分块、基于LLM的概念抽取与角色分类、语义聚类+证据聚合以及LLM关系判断；使用Llama、Qwen等开源大型语言模型。

**📊 数据集**

使用了三门计算机科学课程（Database Systems、NLP、Algorithms）的幻灯片/讲义材料作为数据集。

**📈 对比分析**

与KGGen和EDC两种主流LLM知识图谱构建方法对比，节点显著性均超过0.94，关系准确率提升18%–34%，在所有课程均优于基线。

**⚠️ 局限性**

局限性包括仅覆盖两种关系类型、仅测试CS课程、缺乏跨学科验证以及未评估更大规模模型的表现。

---

## 205. A Locality Radius Framework for Understanding Relational Inductive Bias in Database Learning

**arXiv ID:** 2602.17092 | [PDF](https://arxiv.org/pdf/2602.17092v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 206. Projective Psychological Assessment of Large Multimodal Models Using Thematic Apperception Tests

**arXiv ID:** 2602.17108 | [PDF](https://arxiv.org/pdf/2602.17108v1)

**作者:** Anton Dzega `[一作]` (Ben-Gurion University of the Negev), Rami Puzis `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 2404 | [OpenAlex ID](https://openalex.org/A5059447087)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究利用TAT图像与SCORS-G评估框架，分别让大型多模态模型（LMM）生成叙事并用另一组LMM进行自动评分，探讨是否能用非语言手段评估LMM的“非认知”人格特质。

**💡 创新点**

创新点在于首次将项目式心理测评（TAT+SCORS-G）与多模态语言模型结合，并通过无监督方法挑选出高一致性的评估模型，实现模型自身叙事的多维人格评估。

**🔧 技术方法**

技术主要包括：多模态Transformer（如CLIP/ViT+LLM）、Prompt设计（多重指令变体）、无监督模型子集筛选（Krippendorff α/ICC）、SCORS-G评分自动化及一致性指标（A-WS-Std、APSC）。

**📊 数据集**

使用的核心数据集是官方SCORS-G评分手册中的92个标注叙事样本（作为评估基准），以及7张TAT卡片（图像）和三种等效指令，共生成63条故事。

**📈 对比分析**

比较方法：选取四个评估模型（claude‑3.5‑sonnet、claude‑3.7‑sonnet、gpt‑4.1、gpt‑5）在MAE、Spearman相关、内部一致性指标上与人类专家评分对照；随后用这些模型评估SM生成的故事，按模型家族与代际水平评估SCORS‑G维度得分，结果显示更大更新型模型得分更高，且情绪冲突维度得分普遍较低。

**⚠️ 局限性**

局限性包括：TAT/SCORS‑G为人类工具，直接迁移到模型可能存在构念外推；缺乏对评估模型社会可取性偏差的控制；只用量化评分，未进行叙事内容定性分析；图像顺序效应未检验；实验只覆盖当前模型架构，未来更新可能改变结果。

---

## 207. Multiple Index Merge for Approximate Nearest Neighbor Search

**arXiv ID:** 2602.17099 | [PDF](https://arxiv.org/pdf/2602.17099v1)

**作者:** Liuchang Jing `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 40182 | [OpenAlex ID](https://openalex.org/A5100391662)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种子索引合并框架，利用逆向工程技术将多份已有的近似最近邻子索引快速合并为整体索引；

**💡 创新点**

创新点在于设计了反向最近邻搜索（RNN）与反向图搜索（RNS）两种核心合并策略，并在多索引合并时通过图合并算法与分区优化显著减少合并时间；

**🔧 技术方法**

采用图基近似最近邻算法（如NSG、SSG、τ-MNG）与逆向工程的RNN/RNS搜索技术，对图结构进行优化与合并；

**📊 数据集**

使用了多种大规模高维数据集，包括SIFT1M、GIST1M、DEEP1M、MARCO1M、IMAGE10M、ANTON10M等，从几百万到一亿级别；

**📈 对比分析**

与DiskANN等重叠分区方法以及重建-from-scratch基线进行对比，实验显示RNSM在合并效率上提升1.6–9.92×，在召回率与QPS平衡上保持与基线相当；

**⚠️ 局限性**

局限性主要体现在只能合并无重叠或有限重叠的子索引，对高维稀疏数据或非图结构索引的适应性不足，并且在分区策略与合并算法上仍需针对特定硬件与参数做进一步调优。

---

## 208. AudioChat: Unified Audio Storytelling, Editing, and Understanding with Transfusion Forcing

**arXiv ID:** 2602.17097 | [PDF](https://arxiv.org/pdf/2602.17097v1)

**作者:** William Chen `[一作]` (Carnegie Mellon University), Zeyu Jin `[通讯]` (Adobe Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了AudioChat，一种统一的多模态LLM，可对复杂多源音频故事进行生成、编辑与理解；

**💡 创新点**

创新点包括：1) AudioCopilot工具调用LLM模拟用户-音频设计师交互，生成海量高质量合成音频数据；2) Audio Transfusion Forcing目标，将文本语言模型与多轮扩散强迫训练结合，实现结构化链式思考与多轮音频生成；3) Self‑Cascaded Transformer（SCT）架构，使单一模型兼顾理解与生成；4) 三种新评估指标（multiFLAM、ΔmultiFLAM、editFLAM）精准衡量多源音频的内容与编辑质量；

**🔧 技术方法**

采用Gemma 2 2B为基础LLM，结合连续音频分词器（类似DAC‑VAE）、扩散模型、链式思考与多轮Diffusion Forcing、SCT架构与自定义注意力掩码；

**📊 数据集**

主要数据来自自研的AudioCopilot模拟的6M对话（共13.3K小时生成、26.6K小时编辑），评测集StoryGen‑Eval包含1200对话，seed源为LibriSpeech test‑clean；

**📈 对比分析**

与SAO、WavJourney、DiT、Diffusion‑LLM、Cascade等基线对比，AudioChat在KAD、multiFLAM、ΔmultiFLAM、editFLAM、人工评分及延迟方面均表现最佳，尤其在编辑任务中ΔmultiFLAM与editFLAM均显著优于基线；

**⚠️ 局限性**

局限性包括：模型训练使用部分专有数据，当前未公开发布；评测集为合成数据，缺乏真实多源音频验证；模型仅支持48kHz音频，尚未融合视觉信息，泛化能力需进一步验证。

---

## 209. NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting

**arXiv ID:** 2602.17182 | [PDF](https://arxiv.org/pdf/2602.17182v1)

**作者:** Jiwei Shan `[一作]` (Chinese University of Hong Kong), Shing Shin Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2102 | [OpenAlex ID](https://openalex.org/A5072251844)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了单目非刚性内镜SLAM系统NRGS‑SLAM，利用3D高斯散射建模并实现相机姿态跟踪与高质量的软组织重建；

**💡 创新点**

创新点包括：① 为每个3D高斯原语加入可学习的变形概率，实现对刚性与非刚性区域的显式分离；② 采用Bayesian自监督策略无外部标签推断变形概率；③ 设计粗细相机跟踪与每帧变形更新的两阶段优化；④ 动态调整高斯基函数的变形场，平衡表达力与效率；⑤ 统一鲁棒几何损失融合基础模型几何先验，缓解单目非刚性SLAM的不可辨识性；

**🔧 技术方法**

技术手段包括：3D Gaussian Splatting、Bayesian自监督学习、粗细PnP与光度/几何联合优化、动态高斯基函数变形场管理、滑动窗口Bundle Adjustment、基于深度与轨迹的几何先验、深度估计模型（如SpatialTrackerV2）等；

**📊 数据集**

实验数据集：StereoMIS、Hamlyn、C3VDv2（均为公开内镜数据集）；

**📈 对比分析**

与DefSLAM、NR‑SLAM、MonoGS、S3PO、4DTAM、DDS‑SLAM、EndoGSLAM、Endo‑2DTAM等方法进行对比。NRGS‑SLAM在定位RMSE上相较最佳基线降低约42–50%，在重建质量（PSNR/SSIM/LPIPS）方面也优于所有基线，显示显著性能提升；

**⚠️ 局限性**

局限性：目前未实现实时性能；变形场参数维度高导致计算开销大；Bayesian自监督模块额外渲染与计算成本高；在强光/反射等内镜环境下变形判别不够稳健；缺乏多模态融合，进一步提升鲁棒性和精度需要研究。

---

## 210. When LLM Judges Inflate Scores: Exploring Overrating in Relevance Assessment

**arXiv ID:** 2602.17170 | [PDF](https://arxiv.org/pdf/2602.17170v1)

**作者:** Chuting Yu `[一作]` (University of Queensland), Teerapong Leelanupab `[通讯]` (University of Queensland)

**通讯引用:** 176 | [OpenAlex ID](https://openalex.org/A5049764096)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了大语言模型在信息检索相关性评估中的过度评分（overrating）现象，并通过点评与对比评判、结构变换与词汇干预等多种实验验证其普遍性。

**💡 创新点**

首次将过度评分与模型置信度、句子长度、句法形式、词汇重叠等因素关联，并提出了结构保持变换与查询词注入的对照实验，以揭示 LLM 评判偏向表层特征而非语义本身。

**🔧 技术方法**

采用了多种提示工程技术（如 UMBRELA）、token‑level 置信度统计、label transition matrices、以及 LLM 生成的语义等价改写来探测模型行为。

**📊 数据集**

使用 TREC Deep Learning 2019 与 2020 两年数据集（DL2019、DL2020）中的文档段落及其人工相关性标注。

**📈 对比分析**

与人工评估者的标签对齐度通过 Cohen's κ、Overrate Ratio、Mean Bias、Pairwise Accuracy、Tie Rate 等指标衡量；实验显示 LLM 在二值评判上相对准确，但在分级与对比评判中普遍存在 30%~50% 的过度评分与高置信度错误。

**⚠️ 局限性**

局限性在于实验仅覆盖四个开源 LLM，缺乏对更大规模商业模型的检验；提示设计可能影响结果，且缺乏对不同主题领域的跨领域泛化评估。

---

## 211. SimulatorCoder: DNN Accelerator Simulator Code Generation and Optimization via Large Language Models

**arXiv ID:** 2602.17169 | [PDF](https://arxiv.org/pdf/2602.17169v1)

**作者:** Yuhuan Xia `[一作]` (National University of Defense Technology), Ruiyu Zhang `[通讯]` (National University of Defense Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SimulatorCoder，一种基于大型语言模型（LLM）的自动代理，用来根据自然语言描述生成并优化深度神经网络（DNN）加速器仿真器代码。

**💡 创新点**

首次将领域特定提示工程（包括 In‑Context Learning 与 Chain‑of‑Thought）与多轮错误反馈自修复相结合，构建可自动生成与优化、并保持周期级精度的仿真器框架。

**🔧 技术方法**

使用 LLM（DeepSeek‑V3、GPT‑4o）、结构化提示模板、ICL 与 CoT 结合的提示工程、错误反馈自修复机制以及 Python 代码生成技术。

**📊 数据集**

采用自定义的 SCALE‑Sim 基准，包含 138 个任务和八个主流网络（NCF、AlphaGo Zero、MobileNet、DeepSpeech2、Faster R‑CNN、YOLOv2、ResNet50、Transformer）。

**📈 对比分析**

通过与手工实现的 SCALE‑Sim 在周期计数与运行时的对比，SimulatorCoder 在大多数工作负载下实现了更短的运行时间，周期误差均低于 1%（部分模型误差为 0%），表明其性能与传统仿真器相当甚至更优。

**⚠️ 局限性**

局限性在于仍需手工设计提示模板和结构化语义，缺乏完全自动化；在极大规模或高度复杂的加速器结构下，代码生成质量与执行效率可能受到影响。

---

## 212. Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization

**arXiv ID:** 2602.17155 | [PDF](https://arxiv.org/pdf/2602.17155v1)

**作者:** Yicheng Lang `[一作]` (Michigan State University), Sijia Liu `[通讯]` (IBM Research)

**通讯引用:** 4321 | [OpenAlex ID](https://openalex.org/A5100321843)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不使用反向传播的情况下，提出一种新的零阶优化方法 ZO-Muon，用于大规模模型的内存高效微调。

**💡 创新点**

创新点在于将投影到低维子空间的梯度估计与 Muon 的谱梯度正交化相结合，既降低了梯度估计方差，又提升了收敛速度与精度。

**🔧 技术方法**

核心技术包括子空间随机梯度估计、投影矩阵 QR 分解、Newton–Schulz 或 SVD 计算矩阵正交化（msign）以及多查询策略。

**📊 数据集**

在 LLM 任务上使用 OPT‑1.3B、OPT‑13B、LLaMA3‑8B 微调 SuperGLUE；在视觉任务上使用 ViT‑B/16、ViT‑L/16 微调 CIFAR‑10 与 CIFAR‑100。

**📈 对比分析**

与 MeZO、LOZO、SubZero、Subspace‑MeZO、S‑MeZO 等最新 ZO 基线以及 Adam/LoRA 等 FO 基线对比，ZO‑Muon 在绝大多数任务上实现了更高的最终精度、更快的收敛速度和更低的查询/运行时间，尤其在 ViT‑L/CIFAR‑100 上提升 25%+ 的准确率，同时保持与其他 ZO 方法相同的查询预算。

**⚠️ 局限性**

限制主要体现在：仅针对参数高效微调；在全模型预训练、极大规模模型或非常高维任务中尚未验证；需要额外的超参数调优（投影维数、查询次数）以及投影矩阵的重采样频率等。

---

## 213. TimeOmni-VL: Unified Models for Time Series Understanding and Generation

**arXiv ID:** 2602.17149 | [PDF](https://arxiv.org/pdf/2602.17149v1)

**作者:** Tong Guan `[一作]` (Zhejiang University), Shirui Pan `[通讯]` (Griffith University)

**通讯引用:** 23733 | [OpenAlex ID](https://openalex.org/A5008056593)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一个基于视觉的统一时序模型框架，能同时完成时序理解与生成。

**💡 创新点**

核心创新在于：① 引入了高保真双向时序↔图像映射（Bi-TSI），通过鲁棒归一化和编码容量控制实现近乎无损转换；② 设计了生成链式思考（CoT），将理解结果作为控制信号指导时序生成。

**🔧 技术方法**

采用了视觉模型 Bagel 作为骨干，配合 Bi-TSI 转换器、RFN 归一化、编码容量控制、基于扩散的生成模块以及 CoT 生成与调度机制。

**📊 数据集**

构建了新的基准数据集，包含 6 个基于 GIFT‑Eval 的理解任务和 2 个生成任务（预测与插补），共计 40k 预测样本和 40k 插补样本，9,409 QA 以及 2,339 TSR‑Suite 参考。

**📈 对比分析**

与 Gemini‑2.5‑Flash、Qwen2.5‑7B、Time‑R1、TimeOmni‑1、VisionTS++ 等多种 LLM、TSFMs 和视觉生成模型对比，模型在理解任务上从 0 提升至 ~1.0，预测任务 nMASE 与 VisionTS++ 同级，插补任务达到最优；使用 CoT 生成时平均提升 8.2% 的生成质量。

**⚠️ 局限性**

局限性包括：① 对图像分辨率的依赖，超长序列仍需更大图像或多图拼接；② 依赖视觉映射，极端异常值或剧烈波动仍可能导致信息损失；③ 生成任务仍受模型计数能力限制，长周期预测仍有挑战。

---

## 214. When More Experts Hurt: Underfitting in Multi-Expert Learning to Defer

**arXiv ID:** 2602.17144 | [PDF](https://arxiv.org/pdf/2602.17144v1)

**作者:** Shuqi Liu `[一作]` (Nanyang Technological University), Luke Ong `[通讯]` (Nanyang Technological University)

**通讯引用:** 3778 | [OpenAlex ID](https://openalex.org/A5025152913)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的多专家学习式推迟（L2D）框架PiCCE，用来解决传统多专家L2D中的“分类器欠拟合”问题。

**💡 创新点**

创新点在于：①揭示多专家场景下的专家聚合项导致标签分布扁平化，从而产生欠拟合；②通过利用专家的置信度与正确性（ground‑truth信息）动态挑选“自信且正确”专家，构建连续、可优化的PiCCE损失；③证明PiCCE在统计一致性和可恢复专家准确度方面的理论保证。

**🔧 技术方法**

技术手段包括：基于交叉熵或OvA的多类别代理损失；对损失进行结构化设计以消除专家聚合扁平化；理论分析（风险推导、统计一致性证明）；梯度优化和连续性保证。

**📊 数据集**

实验使用合成专家数据（CIFAR‑100、ImageNet）以及真实专家数据（MiceBone、Chaoyang）进行评估。

**📈 对比分析**

与传统多专家CE/OvA代理损失（Vanilla）以及基于中间结果的Pick‑the‑Confident方法对比，PiCCE在系统误差和覆盖率上均表现更优，且随着专家数量增加表现更为稳健。

**⚠️ 局限性**

局限性：需要专家在训练阶段提供标注，且理论一致性依赖于“最佳专家唯一性”与“信息优势”等条件，实际场景中若这些条件不满足，性能提升可能有限。

---

## 215. B$^3$-Seg: Camera-Free, Training-Free 3DGS Segmentation via Analytic EIG and Beta-Bernoulli Bayesian Updates

**arXiv ID:** 2602.17134 | [PDF](https://arxiv.org/pdf/2602.17134v1)

**作者:** Hiromichi Kamata `[一作]` (Sony Group Corporation), Fuminori Homma `[通讯]` (Sony Group Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于3D Gaussian Splatting的交互式分割框架B^3-Seg，在无相机、无训练的条件下实现开放词汇分割。

**💡 创新点**

创新点在于将分割建模为顺序Beta–Bernoulli贝叶斯更新，并用解析期望信息增益EIG主动选取视角，理论上实现1-1/e的近似保证。

**🔧 技术方法**

使用的技术包括3D Gaussian Splatting渲染、Beta–Bernoulli贝叶斯推断、解析EIG计算、Grounding DINO+SAM2+CLIP的开源2D分割与文本重排序。

**📊 数据集**

实验数据集为LERF-Mask和3D-OVS，均采用无视角、无标签的设置。

**📈 对比分析**

与FlashSplat等采样基线相比，B^3-Seg在仅20视角、约12秒内即可获得与使用重建视角、标签或耗时数十分钟方法相当甚至更优的mIoU（约88%）。

**⚠️ 局限性**

局限在于目前仅处理二元前景/背景、对大规模场景的视角搜索尚不充分、以及对多对象/多类别的扩展需要进一步研究。

---

## 216. Quantifying Competitive Relationships Among Open-Source Software Projects

**arXiv ID:** 2602.17131 | [PDF](https://arxiv.org/pdf/2602.17131v1)

**作者:** Yuki Takei `[一作]` (Japan Advanced Institute of Science and Technology), Chaiyong Ragkhitwetsagul `[通讯]` (Mahidol University)

**通讯引用:** 737 | [OpenAlex ID](https://openalex.org/A5059838021)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了开源软件（OSS）项目间的竞争关系，并提出一种新的基于结构向量自回归（SVAR）和冲击响应函数的“Mutual Impact Analysis of OSS（MIAO）”方法，用以量化竞争影响并识别因竞争导致的项目停止开发（REV）事件。

**💡 创新点**

创新点在于首次将宏观经济学中的SVAR与冲击响应函数引入OSS生态系统，用以定量描述项目间的竞争动态，并将该量化指标用于预测项目衰退，提供了一个可操作的早期预警工具。

**🔧 技术方法**

采用的技术包括：SVAR模型与递归识别、冲击响应函数（IRF）、分数差分实现平稳化、AIC/BIC/HQIC选择最佳滞后阶数、Ljung–Box白噪声检验、决策树分类、以及多周期平均（AMS）等。

**📊 数据集**

使用的数据集为从GitHub获取的每日提交次数序列，筛选标准为星标≥2538、贡献者≥23、提交≥500，最终构建了187个（其中87个REV、100个非REV）三变量（目标+两个竞争者）项目组。

**📈 对比分析**

通过将MIAO得分作为特征输入决策树，对回顾性识别与一年提前预测两种情景进行评估，回顾性准确率达81%（F1≈0.71），一年提前预测准确率约77%，同时在非REV类保持高准确率（≈0.78）。

**⚠️ 局限性**

局限性包括：样本规模有限且仅涵盖GitHub项目；仅考虑非fork竞争，忽略了fork导致的竞争；使用提交数作为唯一活动指标，未涵盖issues、release等信号；竞争标签依赖README推断，可能存在误判；模型维度固定为3变量，未能捕捉多竞争者的高阶交互。

---

## 217. Efficient Parallel Algorithm for Decomposing Hard CircuitSAT Instances

**arXiv ID:** 2602.17130 | [PDF](https://arxiv.org/pdf/2602.17130v1)

**作者:** Victor Kondratiev `[一作]` (ISDCT SB RAS), Alexander Semenov `[通讯]` (ISDCT SB RAS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现一种基于时间限制CDCL求解器的并行CircuitSAT分解算法，利用区间分割构建SAT划分。

**💡 创新点**

创新点在于将可定时求解器与区间划分相结合，形成自适应分解树，并通过优先级队列聚焦高概率子问题，实现高效并行求解。

**🔧 技术方法**

使用CDCL SAT求解器（Kissat）、MPI并行框架、区间编码和自适应分裂策略。

**📊 数据集**

评测数据集包括基于排序算法（Bubble、Selection、Pancake）的LEC实例和MD4压缩函数的预影子攻击（MD4-40至MD4-43）。

**📈 对比分析**

与单线程Kissat和默认Cube‑and‑Conquer相比，在相同硬件上实现了数十倍至上百倍的加速，成功解决了原先不可解的实例。

**⚠️ 局限性**

局限在于参数（q、d、t）需要手动调优，缺乏子树间学习句子共享，以及对极大规模实例的可扩展性尚待验证。

---

## 218. Owen-based Semantics and Hierarchy-Aware Explanation (O-Shap)

**arXiv ID:** 2602.17107 | [PDF](https://arxiv.org/pdf/2602.17107v1)

**作者:** Xiangyu Zhou `[一作]` (Arizona State University), Yang Weng `[通讯]` (Arizona State University)

**通讯引用:** 3944 | [OpenAlex ID](https://openalex.org/A5021106309)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于 Owen 值的可解释方法 O-Shap，利用语义感知的层次化分割来生成结构化且准确的特征重要性解释。

**💡 创新点**

创新点在于：① 将 Owen 值引入 Shapley 解释框架，允许在预定义的层次结构中计算群组特征贡献；② 设计了满足正 T‑property 的语义分割算法，保证子分区与父分区在语义上的一致性；③ 通过层次结构实现从指数到多项式的计算复杂度提升。

**🔧 技术方法**

主要技术包括：Owen 值推导与多层层次化计算；基于 Canny 边缘检测的初始分割；图模型中按归因相似度合并的语义层次化分割；与基线 SHAP 相关的效率、线性、对称性与群组虚拟性等公理验证。

**📊 数据集**

使用了六个图像数据集（Brain Tumor MRI、Tiny-ImageNet、ImageNet-S50、CelebA、PASCAL‑VOC‑2012）和一个表格数据集（Adult Census Income）进行评估。

**📈 对比分析**

与 SHAP、AA‑SHAP、GradSHAP、Integrated Gradients、Occlusion、RISE、h‑SHAP 等七种基线在 mIoU、EBPG、Bbox、F1、AUC 等指标上比较，O‑Shap 在大多数指标上表现最优；在运行时间上，从指数级（SHAP）提升到多项式级，且对模型和图像尺寸的扩展性更好。

**⚠️ 局限性**

局限性包括：对分割超参数（阈值、合并阈值等）敏感；在极高分辨率或极大数据集上仍需进一步优化；在非图像领域的表现虽然优于 SHAP，但对复杂特征交互（如非结构化表格）仍可能存在解释不完整的问题。

---

## 219. Continual uncertainty learning

**arXiv ID:** 2602.17174 | [PDF](https://arxiv.org/pdf/2602.17174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 220. Online Learning with Improving Agents: Multiclass, Budgeted Agents and Bandit Learners

**arXiv ID:** 2602.17103 | [PDF](https://arxiv.org/pdf/2602.17103v1)

**作者:** Sajad Ashkezari `[一作]` (University of Waterloo), Shai Ben-David `[通讯]` (University of Waterloo)

**通讯引用:** 16384 | [OpenAlex ID](https://openalex.org/A5112193907)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文研究了在线学习中的“改进型代理”模型，提出了可扩展到无限假设类、多分类、带反馈限制以及带成本的改进图的理论框架，并给出了实例最优学习算法和误差上界；

**💡 创新点**

创新点在于引入改进型 Littlestone 维度及其变体（多分类、Bandit 版本、加权图）以刻画学习可行性，首次实现对无限假设类的可学习性分析，并给出多分类与反馈受限场景的实例最优算法；

**🔧 技术方法**

使用的技术主要是树形维度理论（改进型 Littlestone 维度、BIL 维度）、版本空间更新、组合优化与专家权重更新（类似 Hedge）以及自适应决策论的误差分析；

**📊 数据集**

由于论文为纯理论工作，未使用任何数据集；

**📈 对比分析**

与已有研究（如未改进的 Littlestone 维度、战略分类的维度等）相比，本文给出了更一般的误差上界（等于相应维度），并通过构造最优算法实现该上界；

**⚠️ 局限性**

局限性包括仅考虑确定性学习器、可实现设置、仅关注分类误差、假设改进图已知且无噪声；未来可探讨随机化学习、非可实现情形、成本权衡及图信息不足等方向。

---

## 221. Agentic Wireless Communication for 6G: Intent-Aware and Continuously Evolving Physical-Layer Intelligence

**arXiv ID:** 2602.17096 | [PDF](https://arxiv.org/pdf/2602.17096v1)

**作者:** Zhaoyang Li `[一作]` (Zhejiang University), Zhiguo Shi `[通讯]` (Zhejiang University)

**通讯引用:** 8917 | [OpenAlex ID](https://openalex.org/A5041940889)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种面向6G物理层的意图感知代理架构（AgenCom），能够通过自然语言意图和CSI信息闭环感知、决策并执行全链路配置，实现用户需求的多目标权衡。

**💡 创新点**

创新点在于①将大语言模型与物理层知识融合的适配器网络，使模型能够理解和执行可执行的通信策略；②以序列化结构化动作方式统一管理多个物理层模块的决策，提升组合空间可搜索性；③引入记忆模块和物理层工具执行器实现实时反馈与自适应学习，真正实现闭环代理自治。

**🔧 技术方法**

主要技术包括：大语言模型（GPT‑2 Medium）与轻量化域适配器；多模态感知（CSI + 自然语言）；结构化动作生成（分层子动作序列）；基于Sionna的可微物理层模拟器；强化学习式奖励回馈与经验重放；记忆检索机制。

**📊 数据集**

使用了基于Sionna‑RT生成的合成CSI数据集，并为每条CSI分配三类自然语言意图（高吞吐、高可靠、节能），形成自制的意图‑CSI数据集；未使用公开的实际测量数据。

**📈 对比分析**

通过对比三类意图下的BER、可达率和额外功率消耗，验证了代理的意图感知与多目标权衡能力。结果显示：可靠意图下BER最低、吞吐意图下速率最高、节能意图下功率显著下降；相对于传统规则化链路配置，AgenCom在同一SNR下实现了更优的三维权衡。

**⚠️ 局限性**

局限性包括①LLM可能产生幻觉或不符合物理约束的动作；②计算和推理延迟高，边缘部署受限；③缺乏真实环境下的验证，合成数据与真实信道差异可能导致性能偏差；④安全性、鲁棒性与可解释性仍待完善。

---

## 222. Understanding Nature Engagement Experiences of Blind People

**arXiv ID:** 2602.17093 | [PDF](https://arxiv.org/pdf/2602.17093v1)

**作者:** Mengjie Tang `[一作]` (Southeast University), Zhuying Li `[通讯]` (Southeast University)

**通讯引用:** 1320 | [OpenAlex ID](https://openalex.org/A5015830444)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对20名盲人和20名视障者的Nature Relatedness Scale（NRS）问卷调查以及对16名盲人进行半结构化访谈，系统探讨盲人如何体验自然、面临的障碍以及对辅助技术的期望。

**💡 创新点**

首次将盲人与视障者的自然关联感进行量化比较，并揭示安全需求、感官策略与情感共鸣三大维度对盲人自然体验的影响，提出面向多感官、情感驱动的技术设计框架。

**🔧 技术方法**

主要采用定量问卷（NRS）与曼-惠特尼U检验进行差异分析，结合主题分析的访谈数据，讨论语音叙述、情感化语音、可穿戴感知设备等多感官技术方案。

**📊 数据集**

数据集由20名盲人、20名视障者完成的NRS问卷以及16名盲人访谈转录文本构成，问卷共21项，访谈平均时长45-60分钟。

**📈 对比分析**

通过曼-惠特尼U检验比较两组NRS得分，盲人整体得分显著低于视障者（p<0.001，效应量r≈0.71），并在NR‑Self、NR‑Perspective、NR‑Experience等子量表上均显著差异，表明盲人自然关联感较弱。

**⚠️ 局限性**

局限包括样本量有限、仅聚焦中国城市地区、未覆盖低视力人群、NRS原本为视障者设计且可能不足以捕捉盲人的非视觉自然体验。

---

## 223. Robustness and Reasoning Fidelity of Large Language Models in Long-Context Code Question Answering

**arXiv ID:** 2602.17183 | [PDF](https://arxiv.org/pdf/2602.17183v1)

**作者:** Kishan Maharaj `[一作]` (IBM Research), Srikanth Tamilselvam `[通讯]` (IBM Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对长上下文代码问答（QA）进行系统评估，扩展 LongCodeBench 至 COBOL 与 Java，并引入选项打乱、无选项生成以及“needle‑in‑haystack”干扰实验，探究模型在不同语言、上下文规模、任务格式与信息扰动下的鲁棒性。

**💡 创新点**

创新点包括：① 多语言（Python、COBOL、Java）长上下文 QA 基准的首次构建；② 通过受控扰动（选项随机化、去除选项、插入相关/无关代码）深入分析模型的识别‑生成差距、位置偏置与干扰耐受性；③ 系统比较多种前沿 LLM（GPT‑4o、Gemini‑Pro/Flash、Claude‑Sonnet、Llama‑3.x、Mistral‑24B/675B、Qwen‑30B、Granite‑8B）在大规模上下文中的表现。

**🔧 技术方法**

采用多任务评估技术：多选精度（与选项打乱后）与无选项生成精度（LLM‑as‑Judge 评判），以及在不同位置（Start/Middle/End）插入相关/无关代码的 Needle‑in‑Haystack 测试；结合上下文长度（32k–1M tokens）与模型输入限制的对齐。

**📊 数据集**

使用的数据集包括：Python LongCodeBench 原始数据集；COBOL OPPSCAL（CMS 医疗定价代码）与内部 IBM 企业 COBOL；Java 基准从 Elasticsearch、Cassandra、Dubbo、Kafka 四个开源项目提取的 114 道多文件问题。所有数据均覆盖 32k–1M 的上下文长度。

**📈 对比分析**

与选项（打乱）下，模型平均准确率在 32k–512k 范围内可达 70–80%（如 Gemini‑Pro、Claude‑Sonnet），但去除选项后精度下降 15–35pp；在 128k–1M 级别时，多数模型出现显著回落或不稳定；“needle‑in‑haystack”实验显示：多选模式下保持 70%+ 但无选项生成显著受限（<60%），且模型表现对信息插入位置（尤其 Start/End）高度敏感，后者对 legacy 语言 COBOL 更易失效。

**⚠️ 局限性**

局限性：① 评估主要集中在大模型，未覆盖更小或轻量化模型的长上下文能力；② 生成精度仍受限于 LLM‑as‑Judge 的主观性；③ 对 legacy 语言 COBOL 的训练语料有限，导致生成表现不佳；④ 对极端长上下文（>1M）下的注意力机制与记忆分配仍缺乏深入分析；⑤ 研究未探索针对性对抗训练或检索增强机制以缓解位置偏置与干扰敏感性。

---

## 224. Geometric Inverse Flight Dynamics on SO(3) and Application to Tethered Fixed-Wing Aircraft

**arXiv ID:** 2602.17166 | [PDF](https://arxiv.org/pdf/2602.17166v1)

**作者:** Antonio Franchi `[一作]` (University of Twente), Chiara Gabellieri `[通讯]` (University of Twente)

**通讯引用:** 8177 | [OpenAlex ID](https://openalex.org/A5001771133)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个无坐标、基于SO(3)的固定翼逆飞行力学模型，给出了从轨迹到姿态、角速度、推力与攻角的闭式映射，并将其应用于球面平行轨道上的受缆飞行，推导出所需的滑行角和零滑行极限点。

**💡 创新点**

创新点在于：① 将机器人学的FLU坐标系与航空动力学的气动方向无缝结合，获得了全局有效的几何模型；② 提供了完整的闭式逆动力学解（姿态、推力、攻角），可直接用于轨迹规划与可行性检验；③ 在受缆轨道飞行中获得了解析的滑行角公式和零滑行条件，揭示了气动协调与重力感知向量的解耦。

**🔧 技术方法**

主要技术手段包括：几何Newton–Euler动力学、轨迹基坐标法、气动方向的几何定义、协调飞行约束、推力-攻角的二维平衡求解、Cardano公式求解三次方程、敏感性分析和解析极限推导。

**📊 数据集**

实验采用理论参数（质量2 kg、机翼面积0.25 m²、AR 5.6、e 0.8、C_Lα 4.3 rad⁻¹、C_D0 0.035）和示例速度/缆张力组合进行数值仿真；并未使用公开数据集。

**📈 对比分析**

没有与其他方法进行对比或性能评估，仅通过数值仿真展示不同缆张力下的姿态、推力与滑行角的变化，验证了零滑行极限与理论公式的一致性。

**⚠️ 局限性**

局限性包括：① 仅考虑静态/准稳态逆动力学，未处理风、突风或柔性缆线动力学；② 需要简化的线性/二次气动模型，逼近误差在高攻角或大扰动下可能显著；③ 缆线假设为中心质量处、直接对推力/攻角无影响；④ 未给出控制律实现与实验验证，难以评估实际控制性能。

---

## 225. VP-VAE: Rethinking Vector Quantization via Adaptive Vector Perturbation

**arXiv ID:** 2602.17133 | [PDF](https://arxiv.org/pdf/2602.17133v1)

**作者:** Linwei Zhai `[一作]` (Xi'an Jiaotong University), Wei Xi `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 3641 | [OpenAlex ID](https://openalex.org/A5050052141)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 VP‑VAE 训练框架，将量化视为结构化潜在扰动，解耦表示学习与代码本学习，并在无显式代码本训练后通过聚类生成代码本；同时推出轻量化的 FSP 版本；

**💡 创新点**

创新点包括：①将量化等价为噪声扰动，彻底拆分表示学习与代码本更新；②利用 Metropolis–Hastings 采样生成与潜在分布一致且自适应尺度的扰动；③在近似均匀潜在空间下推导 FSP，理论上优于传统 FSQ，且实现简单；

**🔧 技术方法**

技术手段：基于 VQ‑VAE 的编码/解码器结构；低维量化瓶颈；kNN 密度估计与 Metropolis–Hastings 采样；自适应扰动半径估计；Lloyd–Max 条件下的均匀量化；聚类（K‑Means）生成离线代码本；正则化约束潜在均值/方差；

**📊 数据集**

使用的数据集：图像方面 COCO 2017 与 ImageNet；音频方面 LibriSpeech 与 Common Voice（v18.0）;

**📈 对比分析**

对比方法：标准 VQ‑VAE、SimVQ（耦合优化）、FSQ（固定量化）和 TokenBridge；实验表明 VP‑VAE/FSP 在 PSNR/SSIM/LPIPS（图像）和 PESQ/STOI（音频）上均优于基线，且代码本利用率更均衡；在 OOD 评估中表现更稳健；

**⚠️ 局限性**

局限性：①MH 采样在高维下计算量大，需低维瓶颈；②FSP 仅适用于近似均匀潜在空间，非均匀数据时性能可能下降；③需要离线聚类生成代码本，增加部署复杂度；④对极端分布或噪声敏感，仍需进一步稳健性验证。

---

## 226. Low-Cost IoT-Enabled Tele-ECG Monitoring for Resource-Constrained Settings: System Design and Prototype

**arXiv ID:** 2602.17114 | [PDF](https://arxiv.org/pdf/2602.17114v1)

**作者:** Seemron Neupane `[一作]` (Tribhuvan University), Aashish Ghimire `[通讯]` (University of South Dakota)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5115905006)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并实现了一套低成本、基于 IoT 的远程 ECG 监测系统，包含 AD8232 采集模块、ESP32 无线传输、Django 后端存储以及 Flutter 移动可视化。

**💡 创新点**

创新点在于将低成本传感器与 ESP32 结合形成端到端轻量化方案，并提供可扩展的后端架构以便未来加入异常检测和报警功能。

**🔧 技术方法**

核心技术包括 AD8232 ECG 前端放大/滤波、ESP32 的 Wi‑Fi 通信、Django RESTful API、Flutter 移动 UI 与 Flutter 进行实时波形渲染。

**📊 数据集**

使用了 Medi Clinic 提供的真实 ECG 数据集，未采用公开大规模数据集进行训练或验证。

**📈 对比分析**

系统通过实地测试显示能够实时捕获并传输 PQRST 波形；论文未给出定量的性能指标或与其他方法的对比，仅展示了示例波形和数据库存储功能。

**⚠️ 局限性**

主要局限包括缺乏安全加密措施、未实现信号质量评估与自动异常检测、缺乏大规模临床验证以及对不同患者条件下的鲁棒性尚未充分评估。

---

## 227. Multi-Ecosystem Modeling of OSS Project Sustainability

**arXiv ID:** 2602.17112 | [PDF](https://arxiv.org/pdf/2602.17112v1)

**作者:** Arjun Ashok `[一作]` (University of California), Vladimir Filkov `[通讯]` (University of California)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对Apache、Eclipse、OSGeo基金会以及GitHub项目的可持续性进行预测，并构建了OSS‑ProF项目‑基金会路由器。

**💡 创新点**

提出基于社会技术网络特征的可持续性预测模型，并通过两步路由+预测的框架实现跨生态系统的可持续性预测。

**🔧 技术方法**

使用双向 LSTM、Transformer 与双向扩张 LSTM 深度学习网络，结合 SHAP 可解释性，构建 OSS‑ProF 分类器。

**📊 数据集**

使用来自 ASF、EF、OF 基金会孵化期项目数据（共 262+161+20 项目）以及 Joblin 等人标注的 GitHub 成功/失败项目（21 项）。

**📈 对比分析**

通过 5 折交叉验证评估，单基金会模型 F1>90%；跨基金会迁移 F1 显著下降；GitHub 项目可通过基金会模型预测，F1≈95%；路由+模型组合进一步提升。

**⚠️ 局限性**

样本量不足（OF、GH）、类别不平衡、标签可能存在噪声，模型对成熟项目不可推广，未覆盖单一供应商或云原生基金会。

---

## 228. Grasp Synthesis Matching From Rigid To Soft Robot Grippers Using Conditional Flow Matching

**arXiv ID:** 2602.17110 | [PDF](https://arxiv.org/pdf/2602.17110v1)

**作者:** Tanisha Parulekar `[一作]`, Jen Jen Chung `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于条件流匹配（CFM）的框架，将由AnyGrasp生成的刚性抓取姿态映射为软Fin‑ray抓手可执行的抓取姿态，并通过数据驱动的姿态配对实现了此转换。

**💡 创新点**

创新点在于首次将CFM应用于抓取姿态映射，解决刚性与软抓取之间的表示差距；同时利用U‑Net自编码器为CFM提供对象几何条件，实现对不同形状物体的连续、可泛化转换。

**🔧 技术方法**

核心技术包括：条件流匹配（CFM）生成连续变换轨迹；基于MLP的速度场学习；U‑Net深度图自编码器用于生成条件向量；以及通过ODE求解器数值积分实现姿态预测。

**📊 数据集**

数据集由7‑DOF Franka Panda机器人与Intel RealSense D415相机配合Fin‑ray抓手采集的配对抓取样本构成，包含254对训练姿态（8种物体），并在865张深度图上预训练U‑Net；验证集包含15个场景、12种不同物体（包括未见的苹果、橙子、酸橙）。

**📈 对比分析**

通过与AnyGrasp直接生成姿态的基线对比，CFM在见过与未见过的物体上的总成功率分别提升至约38%（vs.15%），在圆柱形、长方形、球形物体上表现尤为显著；具体数值如圆柱体见/未见成功率分别为50%/100%，球形为25%/31%。

**⚠️ 局限性**

主要限制包括：对平面物体的抓取效果仍不如AnyGrasp，可能源于数据集偏好顶视角深抓；模型仍受限于训练时手动校正的抓取策略，未能覆盖侧向捏抓等多样化姿态；以及对极端形状或尺寸的适应性尚待进一步验证。

---

## 229. Toward Trustworthy Evaluation of Sustainability Rating Methodologies: A Human-AI Collaborative Framework for Benchmark Dataset Construction

**arXiv ID:** 2602.17106 | [PDF](https://arxiv.org/pdf/2602.17106v1)

**作者:** Xiaoran Cai `[一作]` (Columbia University), Peng Qi `[通讯]` (Uniphore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出STRIDE与SR-Delta两大框架，构建可信的企业级可持续性评级基准数据集，并通过人机协作进行差异分析

**💡 创新点**

创新点在于将信任方程（credibility、reliability、intimacy、self‑serving purpose）与可持续性评级结合，形成可量化的基准构建与评价体系

**🔧 技术方法**

采用大型语言模型（LLM）进行数据抽取与生成，并辅以人工标注、迭代反馈与专家参与的多代理系统

**📊 数据集**

以企业可持续性报告为主数据来源（如2024年Luxshare可持续性报告），并融合多层标准与外部情报数据

**📈 对比分析**

对同一评级方法（以MSCI为例）在原始数据与STRIDE基准数据上的输出进行对比，识别差异源并提供改进建议；实验表明能揭示披露歧义与系统性偏差，提升评估透明度

**⚠️ 局限性**

局限包括：缺乏大规模真实验证、LLM可能产生幻觉导致基准偏差、数据获取与更新成本高，以及行业与监管对新框架的接受度仍不确定

---

## 230. In-Context Learning in Linear vs. Quadratic Attention Models: An Empirical Study on Regression Tasks

**arXiv ID:** 2602.17171 | [PDF](https://arxiv.org/pdf/2602.17171v1)

**作者:** Ayush Goel `[一作]` (University of California), Sarvagya Somvanshi `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

比较了标准二次注意力与线性注意力在线性回归任务中的在上下文学习表现。

**💡 创新点**

通过系统实验揭示了两种注意力机制在收敛速度、深度效应与分布迁移上的相似与差异。

**🔧 技术方法**

使用Transformer架构实现了软max自注意力和基于特征映射的线性注意力，并对学习率、批大小和训练步数等超参数进行独立调优。

**📊 数据集**

在模拟的线性回归数据集上进行实验，数据包含标准高斯和各向异性高斯输入，k=10个示例。

**📈 对比分析**

以MSE、收敛速度和对各向异性分布的泛化能力进行比较，结果显示两者在6层时性能相近，线性注意力收敛更快。

**⚠️ 局限性**

实验受限于模型规模、仅关注线性回归任务、未深入机制分析，且线性注意力的特征映射选择对结果影响显著。

---

## 231. JEPA-DNA: Grounding Genomic Foundation Models through Joint-Embedding Predictive Architectures

**arXiv ID:** 2602.17162 | [PDF](https://arxiv.org/pdf/2602.17162v1)

**作者:** Ariel Larey `[一作]` (NVIDIA), Yoli Shavit `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了 JEPA-DNA 预训练框架，将 Joint-Embedding Predictive Architecture (JEPA) 与传统 MLM/NTP 结合，用于从局部掩码恢复到全局潜在表示。

**💡 创新点**

创新点在于通过在潜在空间中预测全局 CLS 嵌入，突破传统基于标记恢复的“粒度陷阱”，实现更具功能语义的 DNA 表示；同时设计了多目标损失与正则化，兼容多种 GFM 架构。

**🔧 技术方法**

采用 JEPA、EMA 目标编码器、变异/协方差正则化 (VICReg)、多目标损失（LLM+JEPA+Var+Cov）、Transformer/SSM/Hyena 等骨干，以及多阶段训练与梯度累积策略。

**📊 数据集**

预训练使用 DNABERT-2 训练集（人类 GRCh38 + 5 种模型生物），下游评测涵盖 GUE、VariantBenchmarks、Long‑Range Benchmark、BEND、TraitGym、ClinVar、OMIM 等多元基准。

**📈 对比分析**

通过 Linear Probing 与 Zero‑Shot 相似度评估与原始 DNABERT‑2 对比，JEPA‑DNA 在多项监督与零样本任务上均实现显著提升（表达效应 +6.9%，Mendelian 传递 +7.3% 等）。

**⚠️ 局限性**

限制在于仅验证单一骨干，masking 与聚合策略待进一步优化；尚未证明作为主预训练目标的可行性；缺乏显著性统计和更完整的消融分析。

---

## 232. Isometric Invariant Quantification of Gaussian Divergence over Poincare Disc

**arXiv ID:** 2602.17159 | [PDF](https://arxiv.org/pdf/2602.17159v1)

**作者:** Levent Ali Mengütürk `[一作]` (University College London), Levent Ali Mengütürk `[通讯]` (University College London)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5036122819)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了球面平方海灵得距离与二维高斯测度在伪球面上的双曲几何不变量之间的几何对应关系，并基于此提出一种新的基于ℒ²嵌入的高斯分布距离度量；

**💡 创新点**

创新点在于揭示球面海灵得距离与Poincaré盘模型中的双曲等距不变量的对偶关系，并将该双曲不变量映射到高斯测度空间，从而得到可闭式计算的高斯分布距离公式；

**🔧 技术方法**

主要技术包括测度理论、信息几何、双曲几何（Poincaré盘模型）以及Möbius变换的同胚性质；

**📊 数据集**

本文未使用任何公开数据集，而是以理论推导和符号计算为主；

**📈 对比分析**

由于缺乏数值实验，本文未给出与传统距离（如KL、海灵得等）在实际任务中的性能对比；

**⚠️ 局限性**

主要局限在于：①仅针对一维高斯分布推导，扩展到多维时需进一步研究；②缺乏实验验证；③对超参数（方差阈值）敏感，需在实践中谨慎选择；

---

## 233. Generating Rely-Guarantee Conditions with the Conditional-Writes Domain

**arXiv ID:** 2602.17142 | [PDF](https://arxiv.org/pdf/2602.17142v1)

**作者:** James Tobler `[一作]` (Defence Science and Technology Group), Graeme Smith `[通讯]` (Defence Science and Technology Group)

**通讯引用:** 2239 | [OpenAlex ID](https://openalex.org/A5006104972)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出一种新的抽象解释框架，用来为并发程序自动生成可靠-保证条件（rely‑guarantee conditions），并实现了一个专门针对“条件写入”(conditional‑writes) 的干预域（interference domain）；

**💡 创新点**

创新点在于：①引入干预域概念，允许用户定义任意结构的干预（如仅约束写入条件而不关心写入值）；②在此框架中实现了两种分析模式——转移闭包模式和固定点模式，探讨它们在精度与性能上的权衡；

**🔧 技术方法**

技术方法主要是抽象解释与自定义抽象域（状态域与干预域）相结合，具体实现了状态域函数（抽象后置、约束、Havoc 等）和干预域函数（transitions、stabilise、close）；

**📊 数据集**

数据集：作者使用了五个手工构造的并发程序（reset、circular、mutex1、mutex2、spinlock）以及来自文献的 spinlock 程序；

**📈 对比分析**

比较方法：在两种分析模式（转移闭包与固定点）和两种状态域（常数域与其幂集完成）下，记录验证成功与否、状态域操作次数、运行时间；实验表明，固定点模式在绝大多数情况下更快、操作次数更少，且在某些程序（如 reset）上精度更高；

**⚠️ 局限性**

局限性：语言模型过于简洁（仅支持赋值、分支、循环、跳过），不支持过程调用、动态线程创建、互斥锁等常见并发特性；干预域仅覆盖写入条件，对于更复杂的数据结构（数组、指针）仍需进一步研究。

---

## 234. Physical Human-Robot Interaction for Grasping in Augmented Reality via Rigid-Soft Robot Synergy

**arXiv ID:** 2602.17128 | [PDF](https://arxiv.org/pdf/2602.17128v1)

**作者:** Huishi Huang `[一作]` (National University of Singapore), Cecilia Laschi `[通讯]` (National University of Singapore)

**通讯引用:** 20616 | [OpenAlex ID](https://openalex.org/A5045065209)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于AR的物理人机交互框架，支持对混合刚柔机器人进行远程操控，并通过仿真校准实现真实与虚拟系统的一致性。

**💡 创新点**

①将AR视觉预览与物理机器人实时联动；②采用MuJoCo仿真结合运动捕捉的分阶段参数识别；③为软臂设计轻量级神经网络逆运动学；④实现软硬双系统协同抓取，提升操作安全性与适应性。

**🔧 技术方法**

AR头戴设备与手柄；MuJoCo物理引擎+Unity渲染；OptiTrack运动捕捉系统；多层感知机逆运动学网络；Differential Evolution优化；Huber损失与低通滤波等数值处理。

**📊 数据集**

通过实验收集的运动捕捉轨迹（静态倾斜、动态弯曲、控制增益实验）以及约12万条前向运动学样本用于训练神经网络；抓取实验数据用于评估系统性能。

**📈 对比分析**

与未校准的初始模型对比，内部形状误差从8.98 cm降至1.58 cm（约3%相对误差），显著提升抓取准确度与操作安全性。抓取实验中完成物体到达、跟踪、抓取与转移任务，用户反馈显示误差降低、操作更直观。

**⚠️ 局限性**

参数识别结果虽能使仿真与真实行为保持一致，但缺乏严格的物理可解释性；仿真控制过于简化，可能不适用于强化学习或高精度轨迹优化；用户评估规模有限，缺乏系统化的易用性和认知负荷量化指标。

---

## 235. 3D Scene Rendering with Multimodal Gaussian Splatting

**arXiv ID:** 2602.17124 | [PDF](https://arxiv.org/pdf/2602.17124v1)

**作者:** Chi-Shiang Gau `[一作]` (University of California San Diego), Tara Javidi `[通讯]` (University of California San Diego)

**通讯引用:** 8749 | [OpenAlex ID](https://openalex.org/A5059310658)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出多模态3D场景渲染框架，将稀疏雷达深度测量与视觉图像结合，通过局部高斯过程预测深度并生成点云，用于3D Gaussian splatting的初始化与渲染。

**💡 创新点**

创新点在于使用局部GP高效重建雷达深度图，提供更精准点云与不确定度估计，显著降低计算成本并提升渲染质量；将RF信息嵌入GS流程，实现低成本鲁棒渲染。

**🔧 技术方法**

采用局部高斯过程（Localized GP）、RBF kernel自适应长度尺度、3D Gaussian splatting、LPIPS/SSIM/PSNR评估指标。

**📊 数据集**

在View-of-Delft城市驾驶数据集上进行实验。

**📈 对比分析**

与仅视觉（COLMAP初始化+3DGS）对比，局部GP+RF方法将深度误差从13.07 m降至10.57 m，运行时从9.39 s减至0.81 s；在LPIPS、SSIM、PSNR上分别提升约8%/11%/13%。

**⚠️ 局限性**

局限在于仍需雷达传感器，雷达分辨率有限；局部GP划分区域依赖经验，复杂场景下划分可能不佳；未充分评估极端遮挡下的鲁棒性。

---

## 236. Epistemology of Generative AI: The Geometry of Knowing

**arXiv ID:** 2602.17116 | [PDF](https://arxiv.org/pdf/2602.17116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 237. Resource Allocation for STAR-RIS-enhanced Metaverse Systems with Augmented Reality

**arXiv ID:** 2602.17123 | [PDF](https://arxiv.org/pdf/2602.17123v1)

**作者:** Sun Mao `[一作]` (Sichuan Normal University), Chau Yuen `[通讯]` (Nanyang Technological University)

**通讯引用:** 41752 | [OpenAlex ID](https://openalex.org/A5060020877)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 238. TIFO: Time-Invariant Frequency Operator for Stationarity-Aware Representation Learning in Time Series

**arXiv ID:** 2602.17122 | [PDF](https://arxiv.org/pdf/2602.17122v1)

**作者:** Xihao Piao `[一作]` (Osaka University), Yasushi Sakurai `[通讯]` (Osaka University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对非平稳时序预测中的分布漂移问题，作者提出了一种基于频域的预处理方法——Time-Invariant Frequency Operator（TIFO），通过在频谱上学习跨样本的稳定性权重，重新加权傅里叶系数后再反变换回时域，得到更平稳、更易被预测模型利用的输入。

**💡 创新点**

创新点在于：① 以数据生成视角阐释分布漂移并引入频域稳定性度量；② 设计两阶段模块：先在整个训练集上统计每个频率分量的均值/方差比值作为稳定性分数，再用轻量级 MLP 学习可调权重；③ 该方法是一个与模型无关的 plug‑and‑play 层，兼容多种基准网络；④ 提供了基于傅里叶理论与 Mercer's 定理的理论解释。

**🔧 技术方法**

主要技术包括：离散傅里叶变换（DFT）/逆变换（IDFT），利用多层感知机（MLP）对频率分量做加权，频域统计（均值/方差比）作为特征，傅里叶变换与核方法的理论联系，实验中还使用了多种窗口函数与频率分辨率设置。

**📊 数据集**

实验使用七个公开多变量时序数据集：ETTh1/ETTh2、ETTm1/ETTm2、Electricity、Traffic、Weather（以及 ECL），并在不同预测长度（96/192/336/720）下进行评估。

**📈 对比分析**

与 RevIN、SAN、FAN 等现有归一化/频域方法以及 PatchTST、iTransformer、DLinear 等三种后端模型进行对比。TIFO 在 28 组实验中获得 18 个 top‑1 结果和 6 个 top‑2 结果；在最难的 ETTm2 数据集上，PatchTST 和 iTransformer 的 MSE 分别提升 33.3% 与 55.3%；总体上 MSE/MAE 均显著下降，并在 60%–70% 的实验中实现了速度提升。

**⚠️ 局限性**

局限性包括：① 仅利用全局频谱信息，可能忽略局部短时非平稳特征；② 对训练集样本数量和多样性敏感，若样本不足可能导致稳定性估计不稳；③ 虽然加权层轻量，但仍需额外的 FFT 计算，部署在极低延迟场景时需进一步优化；④ 目前仅在离散时间序列上验证，连续时间或高维时空数据的推广仍待探索。

---

## 239. A Data-Driven Dynamic Execution Orchestration Architecture

**arXiv ID:** 2602.17119 | [PDF](https://arxiv.org/pdf/2602.17119v1)

**作者:** Zhenyu Bai `[一作]` (National University of Singapore), Tulika Mitra `[通讯]` (National University of Singapore)

**通讯引用:** 8158 | [OpenAlex ID](https://openalex.org/A5049237676)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文提出了一种名为 Canon 的并行体系结构，融合编译时与运行时的动态调度，通过可编程 FSM 与时间延迟 SIMD 实现对规则与非规则工作负载的高效计算。

**💡 创新点**

创新点在于：①将数据驱动的 FSM 逻辑在编译时预加载为 bitstream，使每行 PE 能根据输入元数据动态生成指令；②采用时间延迟 SIMD，让指令在多周期内在 PE 网格中“逐行”传播，既保持 SIMD 并行又能适配异步调度；③将局部 scratchpad 与分布式 NoC 结合，支持大规模数据重用与动态负载均衡。

**🔧 技术方法**

使用的技术包括：可编程有限状态机、时间延迟 SIMD、2D Mesh 互连网络、分布式本地 SRAM、动态指令流生成、polyhedral 编译器、动态缓冲区管理、异步累加与分块计算。

**📊 数据集**

实验采用的数据集与工作负载包括：稀疏矩阵乘法（SpMM）、稀疏点乘（SDDMM）、PolyBenchC 基准集、现代深度学习模型（LLaMA、Mistral、ResNet‑50、BERT Longformer）以及对应的稀疏化技术（N:M、窗口式稀疏）。

**📈 对比分析**

与基线比较：与 TPU、2:4 TensorCore、ZeD 专用稀疏加速器以及 CGRA 通用架构对比，Canon 在高稀疏度场景下性能与能效相当甚至更优；在密集计算（GEMM）下仅略低；在多样化的 ML 模型上其 EDP 与专用加速器相近，同时保持更好的通用性。

**⚠️ 局限性**

限制：面积与功耗比纯专用架构高约 10‑30%，对极规则 dense 任务性能略逊；编译器与 FSM 预配置仍需人工干预或半自动化；scratchpad 资源固定，难以在芯片上动态调整；对更复杂的工作负载（如非线性算子、非矩阵乘法）仍需进一步研究。

---

## 240. Simplify to Amplify: Achieving Information-Theoretic Bounds with Fewer Steps in Spectral Community Detection

**arXiv ID:** 2602.17104 | [PDF](https://arxiv.org/pdf/2602.17104v1)

**作者:** Sie Hendrata Dharmawan `[一作]` (Dartmouth), Peter Chin `[通讯]` (Dartmouth)

**通讯引用:** 6458 | [OpenAlex ID](https://openalex.org/A5113696329)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种去掉度数阈值预处理和纠正步骤的简化谱算法，用于二社区随机块模型中的社区检测

**💡 创新点**

通过删除非必要预处理并保留矩阵条目独立性，证明谱分割本身即可达到信息理论极限附近的逆对数误差率，且改进了原有的误差上界

**🔧 技术方法**

使用谱分割、第二特征向量分析、Chernoff界、正态近似与蒙特卡洛模拟相结合的理论分析方法

**📊 数据集**

在合成数据集上进行实验，节点数为500-1000，内团边概率a=0.06n，外团边概率b=0.04n

**📈 对比分析**

与原论文的理论上界和纠正步骤方法对比，实验结果表明简化算法在误差率和角度上均优于原方法，性能接近理论极限

**⚠️ 局限性**

仅针对平衡的两社区模型，且理论推导依赖于正态近似和Chernoff界，未验证对不平衡或多社区情况的适用性

---

## 241. Operationalization of Machine Learning with Serverless Architecture: An Industrial Operationalization of Machine Learning with Serverless Architecture: An Industrial Implementation for Harmonized System Code Prediction

**arXiv ID:** 2602.17102 | [PDF](https://arxiv.org/pdf/2602.17102v1)

**作者:** Sai Vineeth Kandappareddigari `[一作]` (Schneider Electric U.S.A), Benjamin Demers `[通讯]` (Schneider Electric U.S.A)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并部署基于无服务器架构的MLOps框架，实现自动化HS编码预测的全生命周期管理。

**💡 创新点**

创新点在于自定义文本嵌入+Text-CNN模型结合事件驱动无服务器管道，提供可扩展、自动A/B测试和成本优化的工业化解决方案。

**🔧 技术方法**

技术手段包括AWS Lambda、Step Functions、SageMaker、CodeBuild、ECR等无服务器服务，配合Text‑CNN、LSTM、DNN深度学习模型以及自动化超参调优。

**📊 数据集**

使用Schneider Electric自有的HS编码产品描述数据集，包含短/中等描述和技术规格，约5,000类HS代码，并按标签保证级别进行过滤。

**📈 对比分析**

通过37折k‑fold交叉验证、ANOVA及A/B测试比较三种模型，Text‑CNN在上采样后达98%准确率、97%精度，明显优于LSTM和DNN。

**⚠️ 局限性**

限制包括HS编码频繁更新导致模型漂移、类别不平衡、跨国编码差异、缺少新产品样本，以及生成式AI的治理与数据安全约束。

---

## 242. Synergizing Transport-Based Generative Models and Latent Geometry for Stochastic Closure Modeling

**arXiv ID:** 2602.17089 | [PDF](https://arxiv.org/pdf/2602.17089v1)

**作者:** Xinghao Dong `[一作]`, Jin-long Wu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了基于传输型生成模型（扩散模型、流匹配模型和随机插值模型）在潜在空间中的耦合闭合模型，用于2D Kolmogorov流的随机闭合建模，并对其性能进行系统比较。

**💡 创新点**

创新点在于首次将显式的度量保持（Metric‑Preserving）正则化与潜在空间结合，显著提升了流匹配模型的采样速度（单步生成）并降低了对大规模训练数据的需求。

**🔧 技术方法**

使用的技术包括扩散模型、流匹配模型、随机插值模型、端到端联合训练、MP/GA正则化、卷积自编码器以及后验数值模拟验证。

**📊 数据集**

采用了在256×256网格上进行的高精度伪谱DNS产生的2D Kolmogorov流数据，随后下采样至64×64网格，构成了约2万对（条件、闭合）训练与测试数据集。

**📈 对比分析**

通过在物理空间与潜在空间分别训练模型，并对比MSE、相对误差、样本方差与计算成本，结果表明流匹配+MP潜在空间在保持约4%误差的同时实现了10倍以上的采样速度提升，并在后验模拟中取得最优误差与成本平衡。

**⚠️ 局限性**

局限性包括仅在二维湍流场景验证，难以直接推广到三维高维流动；潜在空间正则化参数调优复杂，且在更复杂几何或非周期边界条件下可能需要额外的结构约束。

---

## 243. Representation Collapse in Machine Translation Through the Lens of Angular Dispersion

**arXiv ID:** 2602.17287 | [PDF](https://arxiv.org/pdf/2602.17287v1)

**作者:** Evgeniia Tokarchuk `[一作]` (University of Amsterdam), Vlad Niculae `[通讯]` (University of Amsterdam)

**通讯引用:** 3308 | [OpenAlex ID](https://openalex.org/A5064085103)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究 Transformer NMT 中的表示坍塌问题，并提出基于切片角度离散化的正则化方法以缓解坍塌并提升翻译质量。

**💡 创新点**

创新点在于将切片角度离散化（angular dispersion）作为正则化项，有效防止深层表示坍塌，且此策略在连续输出 NMT 与量化模型中均保持优异性能。

**🔧 技术方法**

采用切片角度离散化正则化、next-token 预测与 CoNMT 回归损失、矩阵熵、球面方差等评估指标，以及混合精度/后训练量化等技术。

**📊 数据集**

使用 WMT19 英德语料库（约 3400 万句）与罗马尼亚-英语数据，采用 BPE 子词分词。

**📈 对比分析**

与标准 Transformer‑big 基线对比，加入正则化后 BLEU 由 41.9 提升至 43.0，CoT NMT 也避免了完全坍塌，量化后模型性能损失最小。

**⚠️ 局限性**

仅在三层（解码器输出、解码器嵌入、编码器输出）进行正则化，可能影响其他层；未充分验证低资源语料；正则化单元可能相互冲突，需要更系统的多层协同策略。

---

## 244. Federated Latent Space Alignment for Multi-user Semantic Communications

**arXiv ID:** 2602.17271 | [PDF](https://arxiv.org/pdf/2602.17271v1)

**作者:** Giuseppe Di Poce `[一作]` (CEA Leti), Paolo Di Lorenzo `[通讯]` (Consorzio Nazionale Interuniversitario per le Telecomunicazioni)

**通讯引用:** 4248 | [OpenAlex ID](https://openalex.org/A5000852147)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多用户语义通信中共享语义预等化器与本地语义等化器的联邦对齐框架，并利用联邦 ADMM 实现去中心化训练和低通信开销的协议。

**💡 创新点**

首次针对广播多用户场景设计共享预等化器与本地等化器协同对齐方法，并在此基础上构建联邦 ADMM 优化与隐私保护的消息交换协议，显著提升跨模型的语义互通。

**🔧 技术方法**

使用深度神经网络潜在表示、线性语义等化器、MIMO 物理层、最小二乘误差目标、联邦 ADMM 优化、预白化以及语义预等化器等技术。

**📊 数据集**

使用 CIFAR‑10 图像分类数据集，并采用 timm 库中多种预训练模型产生的隐藏特征作为语义表示。

**📈 对比分析**

与 First‑K、Top‑K 以及独立预编码多链路基线对比；实验表明在 20 dB SNR、压缩率 ζ 较高的情况下，本方法的下游分类准确率提升约 10–20% ，网络均方误差亦更低。

**⚠️ 局限性**

局限包括仅考虑线性等化器、假设完美 CSI、对语义 pilot 份额敏感、未处理多用户干扰与动态用户分组等实际网络挑战。

---

## 245. On the Reliability of User-Centric Evaluation of Conversational Recommender Systems

**arXiv ID:** 2602.17264 | [PDF](https://arxiv.org/pdf/2602.17264v1)

**作者:** Michael Müller `[一作]` (University of Innsbruck), Eva Zangerle `[通讯]` (University of Innsbruck)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于第三方注释的对话推荐系统（CRS）用户中心评估的可靠性进行大规模实证研究，使用 18 维 CRS-Que 框架对 200 条 ReDial 对话进行 124 名众包工人共 1,053 次评分，并通过随机效应模型与相关性分析评估每一维度的可靠性与结构。

**💡 创新点**

揭示了利用单一注释器评估静态对话日志时的局限性，发现技术性维度（准确性、实用性、满意度）在聚合后具有中等可靠性，而社交维度（人性化、亲和力）则可靠性低；同时揭露了“光环效应”，即多数维度在第三方评分中折叠为单一整体质量信号，提示评估维度可进行归约与简化。

**🔧 技术方法**

采用随机效应（单向与交叉）模型计算 ICC 与 Rel_dial；使用 Krippendorff α（序数）评估排名一致性；计算 Spearman 相关矩阵并进行层次聚类以探究维度间结构；利用功效分析确保样本量足够。

**📊 数据集**

ReDial 对话数据集（共 10k 个人际电影推荐对话），从中抽样 200 条对话进行评估；使用 CRS-Que 18 维度工具进行评分。

**📈 对比分析**

对比了不同可靠性指标（ICC、Rel_dial、Krippendorff α）与聚合效应，结果显示单一注释器的 ICC 接近 0，说明噪声大；但在多注释器聚合（≈5 人）后，Acc、Satisfaction、Perceived Usefulness 等维度的 Rel_dial>0.6，说明聚合可显著提升可靠性；社交维度即使聚合后也低于 0.4，表明难以评估。

**⚠️ 局限性**

局限包括：评估仅基于静态日志，无法捕捉真实用户交互体验；社交维度难以通过文本线索区分；第三方注释器存在偏差，单一 LLM 评估可能过度拟合噪声；实验仅覆盖电影推荐场景，结果对其他领域可能不完全适用。

---

## 246. Quantifying and Mitigating Socially Desirable Responding in LLMs: A Desirability-Matched Graded Forced-Choice Psychometric Study

**arXiv ID:** 2602.17262 | [PDF](https://arxiv.org/pdf/2602.17262v1)

**作者:** Kensuke Okada `[一作]` (University of Tokyo), Kyosuke Bunji `[通讯]` (Kobe University)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5079169279)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型（LLM）自我报告问卷评估中，本文构建了一个可量化社交期望回应（SDR）的心理测量框架，并通过对比HONEST与FAKE‑GOOD指令下的响应，使用IRT估计的潜在得分计算SDR效应；随后设计了一个符合可比性欲望匹配的分级强制选择（GFC）大五人格问卷，并在九种指令调优LLM上验证其对SDR的抑制与人格特征恢复的平衡。

**💡 创新点**

提出了基于IRT的SDR量化指标和可与人类基准比较的效应大小，同时设计了可实现可比性欲望匹配的GFC大五问卷，并在LLM上首次验证其在抑制社交期望偏差与保持人格特征恢复方面的权衡。

**🔧 技术方法**

采用项目反应理论（IRT）中的多维梯度响应模型和有序Thurstonian IRT进行评分，利用对比式指令实验、混合整数优化构造GFC对、以及对生成的合成人物进行测评。

**📊 数据集**

使用公开的IPIP‑100大五题库的98条条目，并通过GPT‑5、Gemini 2.5 Pro对条目进行社会可取性评分；生成50个合成人物的真实大五向量作为地面真值。

**📈 对比分析**

通过计算指令诱导的标准化效应大小（方向校正的Cohen d_z）评估SDR，并用Pearson相关衡量人格特征恢复；结果显示传统Likert问卷在九个LLM中均产生明显正向SDR，而匹配欲望的GFC显著降低SDR且在大多数模型中保持0.5以上的恢复相关，体现出明显的SDR–恢复权衡。

**⚠️ 局限性**

方法受限于仅评估一种人格测验和合成人物；GFC与Likert在题目数量不同，可能影响恢复度；欲望匹配依赖文化通用性评分，需在不同语言/文化中重新验证。

---

## 247. EA-Swin: An Embedding-Agnostic Swin Transformer for AI-Generated Video Detection

**arXiv ID:** 2602.17260 | [PDF](https://arxiv.org/pdf/2602.17260v1)

**作者:** Hung Mai `[一作]` (N2TP Technology Solution JSC), Tuan Do `[通讯]` (N2TP Technology Solution JSC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EA‑Swin模型用于AI生成视频检测，并构建了包含130K视频的大规模EA‑Video基准数据集。

**💡 创新点**

创新点在于设计了Embedding‑agnostic Swin Transformer，采用分离的时空窗口注意力并支持任意ViT风格编码器，同时利用shifted窗口实现跨窗口交互，并在大规模数据集上实现高效建模。

**🔧 技术方法**

主要技术包括Swin Transformer、factorized temporal–spatial attention、shifted窗口机制、attention pooling、ViT‑based编码器（如V‑JEPA2）、MLP分类器及AMP训练。

**📊 数据集**

使用了自建的EA‑Video数据集（约65K AI生成视频+62K真实视频），涵盖多商用与开源生成器，并结合VidProM、GenBuster、ViF等现有数据集。

**📈 对比分析**

与多种SOTA方法（DeMamba、NPR、STIL、TALL、WaveRep、ResTraV、D3等）在seen和unseen生成器上对比，EA‑Swin平均准确率分别为0.9866/0.974，AUC约为0.999/0.997，显著优于对手。

**⚠️ 局限性**

局限性包括相对较高的计算与内存开销，主要在标准基准上评测，缺乏真实场景或跨域的广泛验证，且对极低帧率视频的鲁棒性有限。

---

## 248. Human attribution of empathic behaviour to AI systems

**arXiv ID:** 2602.17293 | [PDF](https://arxiv.org/pdf/2602.17293v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 249. Decoding the Human Factor: High Fidelity Behavioral Prediction for Strategic Foresight

**arXiv ID:** 2602.17222 | [PDF](https://arxiv.org/pdf/2602.17222v1)

**作者:** Ben Yellin `[一作]` (OMGene AI Lab), Shula Grinapol `[通讯]` (OMGene AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并训练了 Large Behavioral Model（LBM），一种通过在 Llama‑3.1‑8B‑Instruct 模型上进行参数高效微调，利用结构化的高维心理特质向量来预测个体在高风险战略情境中的决策。

**💡 创新点**

创新点在于将传统的可变人设提示（persona prompting）转变为持久的行为嵌入：将心理测评得到的 74 维特质信息以结构化方式注入模型，显著提升了个体行为模拟的稳定性与可扩展性，并实现了对特质维度随之提升的可持续性能提升。

**🔧 技术方法**

采用的技术包括：LoRA 参数高效微调（r=16，α=32），结构化提示设计（情境+特质+多选题），监督式训练（目标为多选答案及可选理据），以及 JSON 格式的确定性输出。

**📊 数据集**

使用的数据集来自 OMGene 应用的 2,500 名志愿者，包含 74 维标准化心理测评特质、55 个多情境决策问卷，以及部分观测的参与者-情境响应矩阵。

**📈 对比分析**

通过在 75% 情境训练、25% 评估的 hold‑out 设置下，使用平衡准确率和宏 F1 作为指标，LBM 与无微调的 Llama 背景模型相比显著提升（准确率从 0.42 增至 0.48），并与 Claude 4.5 Sonnet 等前沿 LLM 基线相当；当特质维度从 5 维增至 20 维时 LBM 继续提升，超过 20 维后趋于饱和，而提示基线模型在特质维度增大时性能基本保持不变。

**⚠️ 局限性**

主要限制包括：便利志愿样本（主要为美国英语使用者，少量以色列希伯来语子样本），样本非概率抽样导致普适性受限；情境问卷基于回忆与假设，存在意向‑行为差距；文本情境无法完全复制实时交互的感官与情绪沉浸；且当前数据规模限制了在更高特质维度与更复杂情境上的进一步扩展。

---

## 250. Continual learning and refinement of causal models through dynamic predicate invention

**arXiv ID:** 2602.17217 | [PDF](https://arxiv.org/pdf/2602.17217v1)

**作者:** Enrique Crespo-Fernandez `[一作]`, Peter Flach `[通讯]` (University of Bristol)

**通讯引用:** 32795 | [OpenAlex ID](https://openalex.org/A5057936560)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一次探索过程中，构造并持续修正一个符号因果世界模型，利用所学模型进行规划并实时纠正预测错误。

**💡 创新点**

创新点：1）在线连续模型修复；2）利用 Meta‑Interpretive Learning 与谓词发明构建层次化、可重用的抽象；3）升维推理实现规模不变的样本效率。

**🔧 技术方法**

技术：Meta‑Interpretive Learning、谓词发明、预测‑验证‑修正循环、升维推理、离散规划（A* 等）。

**📊 数据集**

MiniHack “Lava Crossing” 环境（10×10 与 100×100 网格）作为实验数据集。

**📈 对比分析**

与 PPO 进行对比：我们的符号模型在第 2 轮就完成任务，PPO 需要约 128 轮；在 10×10 网格上收敛到 43 条规则，PPO 在 300 轮才开始收敛；在更大网格上零样本迁移，显示样本效率与规模不变性。

**⚠️ 局限性**

局限性：1）目前仅在确定性、完全可观测的网格世界验证；2）缺乏概率处理与不确定性建模；3）对先验元规则与背景知识有一定依赖；4）大规模连续空间或高维感知仍待扩展。

---

## 251. The Case for HTML First Web Development

**arXiv ID:** 2602.17193 | [PDF](https://arxiv.org/pdf/2602.17193v1)

**作者:** Juho Vepsäläinen `[一作]` (Aalto University), Juho Vepsäläinen `[通讯]` (Aalto University)

**通讯引用:** 27 | [OpenAlex ID](https://openalex.org/A5022306750)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文探讨了“HTML First”网页开发理念，并将其与主流前端框架（如React、Angular、Vue）进行对比，提出一套基于纯HTML、CSS和Vanilla JS的开发模式。

**💡 创新点**

创新点在于系统化阐述HTML First的设计原则与实现步骤，揭示其在开发流程、可维护性与性能上的优势，并为中小型项目提供了可落地的实践范式。

**🔧 技术方法**

主要技术包括标准HTML5、CSS3、少量Vanilla JS；辅以构建工具（如Gulp/Grunt）进行自动化压缩、预编译，并使用浏览器DevTools进行性能分析。

**📊 数据集**

实验数据集来源于10个真实网站的案例（覆盖博客、企业站点、电子商务小站）以及一份包含80位前端工程师的问卷调查，用以评估开发效率与性能指标。

**📈 对比分析**

比较方法：对同一功能页面在HTML First与React/Vue/Angular实现下进行基准测试，测量页面首屏加载时间、总资源大小、DOM解析时间等。实验结果显示：HTML First在首屏加载时间上平均快15-25%，资源体积减少30-40%，但在复杂交互（SPA）方面略显不足。

**⚠️ 局限性**

局限性包括：1）对高度交互式、状态管理复杂的单页应用支持有限；2）缺乏成熟的组件化生态与热重载体验；3）在大型项目中维护成本随规模扩大而显著增加。

---

## 252. ArXiv-to-Model: A Practical Study of Scientific LM Training

**arXiv ID:** 2602.17288 | [PDF](https://arxiv.org/pdf/2602.17288v1)

**作者:** Anuj Gupta `[一作]` `[通讯]` (Independent Researcher), Anuj Gupta (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过从原始arXiv LaTeX源码构建科学语料库，详细记录并实现了一个1.36B参数的科学语言模型的完整训练流程，包括数据过滤、文本清洗、符号友好分词、以及在受限GPU（2×A100）上的高效训练。

**💡 创新点**

创新点在于：①提供了一个可复现的端到端工程化管线，用于从非结构化LaTeX源提取高质量科学文本；②在符号密集型语料上评估并选择适配的分词方案；③在有限算力下通过课程学习与梯度累积实现稳定的训练；④系统性记录了多轮实验的瓶颈与优化经验，凸显数据工程对模型性能的关键影响。

**🔧 技术方法**

使用的技术包括：数据预处理（元数据过滤、LaTeX提取、文本标准化、去重），基于LLaMA的SentencePiece分词器（vocab 102k），dense decoder-only transformer（24层、2048隐藏维、16头、RMSNorm、RoPE），bfloat16混合精度、ZeRO Stage‑2、梯度累积、数据并行、激活检查点、以及针对符号文本的课程学习策略。

**📊 数据集**

数据集主要来自：80GB arXiv LaTeX（清洗后约52.18B tokens）作为预训练语料；额外加入OpenWebMath、StackExchange、MathInstruct、UltraChat等公开数学/科学数据进行后期对齐与微调。

**📈 对比分析**

在多达24次实验中，完整数据规模（200GB）下模型训练损失稳定下降，验证 perplexity 约4.2；相比于仅20GB子集，训练更稳定、梯度噪声更低。虽然未与大规模instruction‑tuned模型做直接对比，但在科学文本的困境下表现出较强的符号推理与正式写作能力。

**⚠️ 局限性**

局限性包括：①算力受限，无法进一步扩大模型或上下文长度；②I/O 与存储成为瓶颈；③分词方案虽稳健但未完全针对科学符号优化；④缺乏系统的数学推理评测与指令对齐；⑤数据过滤策略对最终 token 量有较大影响，可能引入偏差；⑥对非选定科学领域的泛化能力有限。

---

## 253. A Multi-modal Detection System for Infrastructure-based Freight Signal Priority

**arXiv ID:** 2602.17252 | [PDF](https://arxiv.org/pdf/2602.17252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 254. Efficient privacy loss accounting for subsampling and random allocation

**arXiv ID:** 2602.17284 | [PDF](https://arxiv.org/pdf/2602.17284v1)

**作者:** Vitaly Feldman `[一作]` (Apple), Moshe Shenfeld `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5021672897)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了一种用户数据在k个步骤中随机均匀选择的采样方案的隐私放大特性，该方案在差分隐私优化和高维私有聚合中应用，显示出相较于标准泊松采样的效用优势。

**💡 创新点**

创新点在于提出了一种高效计算随机分配下隐私损失分布（PLD）的方法，并证明其在隐私-效用权衡方面至少与泊松子采样相当，且更适合通过DP-SGD进行训练。

**🔧 技术方法**

使用了差分隐私（Differential Privacy）、DP-SGD、随机分配等技术。

**📊 数据集**

未具体提及使用的数据集。

**📈 对比分析**

与泊松采样相比，随机分配在隐私-效用权衡方面表现更佳，且为DP-SGD训练提供了更好的适应性。

**⚠️ 局限性**

本文的局限性在于在某些实际设置中，结果的隐私参数由于分析中的近似步骤而不够紧凑。

---

## 255. Unified Latents (UL): How to train your latents

**arXiv ID:** 2602.17270 | [PDF](https://arxiv.org/pdf/2602.17270v1)

**作者:** Jonathan Heek `[一作]` (Google DeepMind), Tim Salimans `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

共同训练编码器、扩散先验和扩散解码器，以学习可解释且信息量可调的潜在表示，用于高质量图像/视频生成。

**💡 创新点**

1）通过固定噪声水平将编码器噪声与先验扩散精度对齐；2）使用重加权ELBO并给解码器加损失因子，实现对比特率的可解释控制；3）两阶段训练：先训练潜在先验再训练基准模型。

**🔧 技术方法**

变分自编码器框架、扩散模型（前向噪声/反向预测）、重加权ELBO、Sigmoid权重、噪声比例控制、ViT/UVit网络、Patching、L2正则化等技术。

**📊 数据集**

ImageNet-512、Kinetics‑600、内部文本到图像数据集（tti AE）等。

**📈 对比分析**

与Stable Diffusion自动编码器、小型SD、中型SD以及像素扩散模型在F1、gFID、rFID、PSNR、bpd等指标上进行基准对比；在训练算力与生成质量曲线中表现最佳，视频生成在Kinetics‑600上达到SOTA FVD。

**⚠️ 局限性**

扩散解码器采样成本高、对自动编码器训练数据敏感、噪声水平对先验精度敏感、低信息潜在可能导致信息泄漏、未提供蒸馏加速方案、对文本等离散数据的适用性未验证。

---

## 256. Trivance: Latency-Optimal AllReduce by Shortcutting Multiport Networks

**arXiv ID:** 2602.17254 | [PDF](https://arxiv.org/pdf/2602.17254v1)

**作者:** Anton Juerss `[一作]`, Stefan Schmid `[通讯]` (TU Berlin and Weizenbaum Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

设计并实现了一种新的 AllReduce 算法，利用双向链路的多端口特性，在 log₃ n 步内完成 AllReduce，并在环与多维托里斯网络上表现优异。

**💡 创新点**

创新点在于同时使用两个端口进行三倍距离通信，并在每一步执行联合归约，既实现了 log₃ n 步的最优时延，又将拥塞显著降低至传统 Bruck 算法的三分之一，兼顾带宽最优。

**🔧 技术方法**

采用了多端口双向链路通信模式、联合归约与递归三分步传播的算法设计，并通过理论分析与 SST 网络模拟验证其性能。

**📊 数据集**

实验使用模拟网络（800 Gb/s 链路、100 ns 延迟）和从 32 B 到 128 MiB 的多种消息尺寸，利用 SST 进行大规模仿真评估。

**📈 对比分析**

与 Bruck、Recursive Doubling、Swing、Bucket 等现有算法进行对比，实验结果显示在环与多维托里斯网络中，该算法在小至中等消息尺寸下完成时间提升 5‑30%，在 3D 托里斯上可达 15%，并在高带宽场景下保持优势。

**⚠️ 局限性**

局限性包括在非 3^k 大小的网络及极大消息尺寸时需要额外处理，且对硬件实现要求双端口同步；在低带宽环境下性能优势下降，且在非正方形多维托里斯或负载不均衡时的效果尚未充分验证。

---

## 257. Structured Prototype-Guided Adaptation for EEG Foundation Models

**arXiv ID:** 2602.17251 | [PDF](https://arxiv.org/pdf/2602.17251v1)

**作者:** Jingying Ma `[一作]` (National University of Singapore), Mengling Feng `[通讯]` (National University of Singapore)

**通讯引用:** 12071 | [OpenAlex ID](https://openalex.org/A5022222926)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种结构化、自信度感知的原型引导适配框架SCOPE，用于在标签稀缺的跨受试者条件下高效微调EEG基础模型。

**💡 创新点**

创新点在于（1）两阶段外部监督构建：通过ETF正则化的任务先导网络和原型聚类生成可靠的伪标签；（2）Prototype-Conditioned Adapter（ProAdapter）在冻结的基础模型深层插入轻量化、原型条件化的特征调制模块，持续引导参数更新；（3）结合自信度融合与热度门控的伪标签筛选机制，提高半监督学习的鲁棒性。

**🔧 技术方法**

使用了ETF几何正则化、Sinkhorn-Knopp平衡原型聚类、Dempster–Shafer理论的信度融合、基于原型的特征调制Adapter、温度门控与权重平滑等技术。

**📊 数据集**

在三个公开EEG数据集上评估：ISRUC（睡眠分期）、SEED（情绪识别）和Mental Arithmetic（工作负荷评估），采用严格的跨受试者划分。

**📈 对比分析**

与多种基线（传统EEGNet/EEGConformer、冻结与全微调基础模型、LoRA、FixMatch、FineSSL）在相同实验设置下对比，SCOPE在Kappa/Weighted‑F1/AUROC/AUPRC等指标上持续超过基线，且仅占用2‑5%可训练参数，训练时间与显存显著降低。

**⚠️ 局限性**

局限性包括：对EEG数据特定结构的假设可能不易迁移至其他生理信号；需要预先训练好的基础模型；在极端标签稀缺或高噪声场景下伪标签质量仍可能受限；以及对超参数（原型数、ETF权重、信度阈值）的敏感性需要进一步自动化。

---

## 258. All Leaks Count, Some Count More: Interpretable Temporal Contamination Detection in LLM Backtesting

**arXiv ID:** 2602.17234 | [PDF](https://arxiv.org/pdf/2602.17234v1)

**作者:** Zeyu Zhang `[一作]`, Bradly C. Stadie `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一套基于命题的框架，用于检测并量化LLM在预测任务中对未来信息的泄漏，并构建了 TimeSPEC 系统来主动消除此类泄漏

**💡 创新点**

创新点在于：①将推理过程拆解为可验证的原子命题并进行时序归因；②引入 Shapley‑加权的决策关键泄漏率（Shapley‑DCLR）衡量泄漏对预测的实际影响；③提出多阶段的 TimeSPEC 结构，通过程序化检索和重生成有效过滤未来知识

**🔧 技术方法**

技术包括：命题抽取与时序验证、命题分类税onomies、Shapley 值计算进行重要性加权、检索增强生成（Perplexity API）、多轮生成与重检索

**📊 数据集**

使用了三类数据集：美国最高法院案例预测（98 案例）、NBA 球员薪资预测（152 奖项）、S&P 500 股票回报排名（100 组）

**📈 对比分析**

与传统的无检索提示（Superforecasting）及提示加时限（Temporal Hint）比较，TimeSPEC 在法律预测保持相似或略低性能且泄漏率几乎为零；在薪资预测性能略下降但泄漏率下降 75%；在股票排名性能大幅下降但泄漏率几乎为零，验证了基线高分是因泄漏导致

**⚠️ 局限性**

局限性包括：仅在推理时过滤泄漏，未从模型参数层面消除泄漏；对需要快速变化信息的任务如股票预测，严苛的时序限制导致性能大幅下降；多阶段管道计算成本高，需要更多 LLM 调用

---

## 259. Multi-session Localization and Mapping Exploiting Topological Information

**arXiv ID:** 2602.17226 | [PDF](https://arxiv.org/pdf/2602.17226v1)

**作者:** Lorenzo Montano-Olivan `[一作]` (Instituto Tecnologico de Aragon), Maria T. Lazaro `[通讯]` (Instituto Tecnologico de Aragon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于地图定位的多会话激光雷达映射框架，在已知地图区域使用定位，在未覆盖或低连通区域触发局部映射与闭环，动态更新全局姿态图。

**💡 创新点**

创新点在于：① 用图谱连通性（平均节点度和 Fiedler 值）作为在线决策依据，自动切换定位/映射模式；② 将先前的姿态图从被动地图转换为主动优化变量，充分利用其不确定性；③ 通过仅增量更新子图而非重建完整地图，显著降低图大小与计算负担。

**🔧 技术方法**

核心技术包括：基于 LiDAR‑惯性里程计的滑动窗口优化、G‑Loc（姿态图定位）与拓扑决策模块、加权连通性指标（基于信息矩阵的节点度与最小特征值）、鲁棒闭环搜索与全局姿态图优化（PGO）。

**📊 数据集**

使用公开数据集 Newer College / Newer College Extension（四个覆盖序列）、Parkland 轨迹序列，以及实地芬兰 Kemi 矿井的激光雷达+IMU 数据集。

**📈 对比分析**

与单会话 SLAM（LIO‑SAM、NV‑LIOM、LG‑SLAM）以及多会话定位/映射方法（MS‑Mapping、M2F、F2F）进行对比。实验显示：平均 ATE RMSE 与最先进 SLAM 相当或更好，且图规模平均缩小约 20%–30%，关键帧处理时间约 50 ms，闭环线程仅在 10% 时间内激活，整体计算量显著下降。

**⚠️ 局限性**

局限性包括：① 仍需要先验地图作为定位基准，无法在首次探索时直接使用；② 连通性决策对参数（阈值、滑动窗口长度）敏感，过于保守或激进均可能导致误判；③ 在动态或稀疏环境下，闭环检测与信息权重估计可能失效，影响地图一致性；④ 目前仅针对 LiDAR‑惯性系统，需进一步推广到视觉或多传感器融合。

---

## 260. Adaptive encodings for small and fast compressed suffix arrays

**arXiv ID:** 2602.17201 | [PDF](https://arxiv.org/pdf/2602.17201v1)

**作者:** Diego Díaz-Domínguez `[一作]` (University of Helsinki), Veli Mäkinen `[通讯]` (University of Helsinki)

**通讯引用:** 6232 | [OpenAlex ID](https://openalex.org/A5043558409)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出可变长度分块（VLB）技术，用层次化树结构对BWT进行自适应编码，构建空间小且查询快速的压缩后缀数组。

**💡 创新点**

创新点在于：①根据BWT跑长度自适应划分区块，压缩率与可读性平衡；②利用树结构将可压缩区块放在根部，复杂区块深层化，从而将空闲位重新分配给查询密集区；③引入“仅在合法后向搜索状态下才保证正确”的采样策略，进一步压缩指令表；④将VLB应用于ϕ⁻¹编码，提升局部性与定位速度。

**🔧 技术方法**

核心技术包括：变长块分块与递归分割、前缀计数数组、位向量路由、SIMD并行解码、rank/succ采样、子采样（sr-index）以及对ϕ⁻¹的VLB树改造。

**📊 数据集**

实验使用四大数据集：AllTheBacteria（30种细菌基因组）、SARS‑CoV‑2（449万条基因组）、Human Pangenome（40个人类基因组）以及Linux内核版本（260万条），涵盖不同重复度与词典大小。

**📈 对比分析**

与现有 run‑length BWT、固定块、Move、r‑index 和 sr‑index 进行对比。结果显示：VLB‑BWT 在计数查询上比最优对手快 2–5×，空间可与 sr‑index 相当；VLB‑sr‑index 在定位查询上比 sr‑index 快 1.3–5.4×，空间增幅仅 1–2×；Move 在速度上最快但空间是 2–8×。总体来看，VLB 在速度‑空间权衡上明显优于传统方法。

**⚠️ 局限性**

主要限制：①构造算法仍依赖多次扫描与递归，规模化构造效率待提升；②ϕ⁻¹ 位置解码的局部性不完全，连续位置仍需跨块访问；③根级 Z 位向量在极大数据集上可能产生 σ 级别的开销；④缺乏严格的适应性空间上界，理论分析仍待完善。

---

## 261. RIS Control through the Lens of Stochastic Network Calculus: An O-RAN Framework for Delay-Sensitive 6G Applications

**arXiv ID:** 2602.17198 | [PDF](https://arxiv.org/pdf/2602.17198v1)

**作者:** Oscar Adamuz-Hinojosa `[一作]`, Xavier Costa-Pérez `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一种基于 O‑RAN 的动态 RIS 编排框架 DARIO，能够在每个调度周期内对多台 RIS 与移动用户进行实时关联，从而在 6G 低时延场景中显著降低包延迟并满足可靠性要求。

**💡 创新点**

创新点包括：① 将 Stochastic Network Calculus 与 UE‑RIS 关联模型结合，能够对不同配置下的概率延迟上界进行解析评估；② 将 UE‑RIS 关联问题建模为非线性整数规划并提出两阶段启发式求解器，实现实时最优或近似最优分配；③ 在 O‑RAN 架构中引入三条专用接口（RIC‑DARIO、DARIO‑NRT、DARIO‑RIS），实现端到端的实时控制与反馈。

**🔧 技术方法**

使用的技术主要有：Stochastic Network Calculus (SNC)、非线性整数规划 (NIP)、启发式算法、O‑RAN 控制框架、Python/SimPy 仿真、以及多台商用 RIS 的软硬件控制。

**📊 数据集**

采用了两类数据集：① 采用 Poisson 生成的合成流量模拟，② 从 Madrid 运营网络收集的真实上行流量轨迹（包括 UE 到达、包大小、信道指标等），并将其合成到仿真与评估中。

**📈 对比分析**

对比方法包括：无 RIS（baseline）、基于 SNR 的静态 RIS 关联、延迟感知的静态 RIS 关联。实验结果显示，DARIO 在各类场景下均优于对照组，90 分位时延目标满足率提升可达 90–95%，并且在高负载/多 RIS 情况下可达 95.71% 的性能提升，且平均吞吐量与基线相当。

**⚠️ 局限性**

局限性包括：SNC 延迟上界相对保守（约 3–5 倍）；求解器在极大规模多基地站、多 RIS 的情况仍可能面临较高计算负荷；当前实现仍基于 O‑RAN 试验平台，未完成在商用网络中的完整标准化与部署；以及对 RIS 物理层细节（如相位误差、频率选择性）的建模仍存在简化假设。

---

## 262. EntropyPrune: Matrix Entropy Guided Visual Token Pruning for Multimodal Large Language Models

**arXiv ID:** 2602.17196 | [PDF](https://arxiv.org/pdf/2602.17196v1)

**作者:** Yahong Wang `[一作]` (Tongji University), Lianghua He `[通讯]` (Tongji University)

**通讯引用:** 2638 | [OpenAlex ID](https://openalex.org/A5091130234)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种基于矩阵熵的视觉令牌剪枝方法EntropyPrune，用于加速多模态大型语言模型的推理。

**💡 创新点**

提出“熵崩塌层”作为剪枝时机，并通过双格矩阵谱加速矩阵熵计算，实现训练无关且高效的剪枝策略。

**🔧 技术方法**

利用矩阵熵（von Neumann熵）衡量令牌信息，采用头部重塑与协方差矩阵估计，配合双格矩阵谱加速技术完成剪枝。

**📊 数据集**

在十个图像基准（MMBench、MME、ScienceQA、TextVQA、MMVet、MMstar、AI2D、OCRBench、MMMU）以及视频基准（MSVD‑QA、MSRVTT‑QA）和高分辨率输入上进行评估。

**📈 对比分析**

与FastV、PDrop、SparseVLM、DART、DivPrune、CDPruner、Prumerge等方法对比，EntropyPrune在保留192/128视觉令牌时保持≈96–98%原性能，同时减少68% FLOPs，速度提升约1.6×。

**⚠️ 局限性**

对极端稀疏场景或不同模型结构的通用性仍需进一步验证，且在视觉令牌极多时谱加速实现仍有一定计算开销。

---

## 263. What Makes a Good Doctor Response? An Analysis on a Romanian Telemedicine Platform

**arXiv ID:** 2602.17194 | [PDF](https://arxiv.org/pdf/2602.17194v1)

**作者:** Adrian Cosma `[一作]` (Dalle Molle Institute for Artificial Intelligence), Emilian Radoi `[通讯]` (University Politehnica of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

分析77k份罗马尼亚文本医疗问答对，研究医生书面回复的哪些特征与患者满意度相关。

**💡 创新点**

结合低资源语言的表面层特征和LIWC心理语言学指标，利用SHAP解释模型发现礼貌、保留用词和数值信息与高评分相关。

**🔧 技术方法**

采用梯度提升决策树（CatBoost）预测二元满意度，并用SHAP进行全局与局部特征重要性解释。

**📊 数据集**

使用77,334条匿名罗马尼亚语文本咨询数据，包含患者问题、医生回复及按“点赞/未点赞”构建的满意度标签。

**📈 对比分析**

通过时间划分训练/测试，模型在全局ROC‑AUC约0.75，且在澄清类问题上表现更佳，显示时间序列拆分与基线对比。

**⚠️ 局限性**

结果主要受患者与医生历史倾向主导，文本特征作用有限，且缺乏对临床准确性的评估，低资源语言资源限制。

---

## 264. Texo: Formula Recognition within 20M Parameters

**arXiv ID:** 2602.17189 | [PDF](https://arxiv.org/pdf/2602.17189v1)

**作者:** Sicheng Mao `[一作]` `[通讯]` (Telecom Paris), Sicheng Mao (Telecom Paris)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并发布了一个仅含20M参数的公式识别模型 Texo，能够将公式图片转换为结构化 LaTeX 代码，并在浏览器端实现实时推理。

**💡 创新点**

创新点包括：① 将词表从 50K 压缩到 687，通过规则化 KaTeX 解析器蒸馏；② 对词表和 tokenizer 进行迁移，显著减少嵌入层参数；③ 在保持与 SOTA 相当性能的同时，将模型大小压缩 80% 以上并实现 7 倍的推理速度提升；④ 完全前端部署，消除后端调用与数据泄露风险。

**🔧 技术方法**

技术手段：使用 PPFormulaNet‑S 的 HGNetV2‑B4 编码器与 2 层 MBart 解码器；词表蒸馏与迁移算法；基于 AdamW 的训练，使用线性 warm‑up 与余弦退火；数据增强包括几何变换、噪声与光照扰动；模型导出为 ONNX 并通过 Transformers.js 在浏览器中推理。

**📊 数据集**

采用公开的 UniMER 数据集：训练集 UniMER‑1M（约 100 万图-LaTeX 对），测试集 UniMER‑Test（约 23k 对，包含简单/复杂/屏幕捕获/手写表达式）。

**📈 对比分析**

与 UniMERNet‑T（107M）和 PPFormulaNet‑S（57M）对比，Texo 在 UniMER‑Test 上的 CDM 分别为 0.958、0.825、0.882、0.902，推理时间仅 20.9 ms/样本（相较 114.8 ms/样本和 117.1 ms/样本），同时模型大小大幅减小，性能基本持平或略优。

**⚠️ 局限性**

局限性：① 仍对极度噪声或极复杂公式的识别准确率有限；② 词表蒸馏需手工设计，可能不适用于所有 LaTeX 变体；③ 训练仍需大显存 GPU，虽比 SOTA 轻量，但对极低配置仍不友好；④ 目前仅针对公式 OCR，未扩展到更通用的文档 OCR 任务。

---

## 265. The Bots of Persuasion: Examining How Conversational Agents' Linguistic Expressions of Personality Affect User Perceptions and Decisions

**arXiv ID:** 2602.17185 | [PDF](https://arxiv.org/pdf/2602.17185v1)

**作者:** Uğur Genç `[一作]` (Delft University of Technology), Himanshu Verma `[通讯]` (Delft University of Technology)

**通讯引用:** 1079 | [OpenAlex ID](https://openalex.org/A5002554721)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在慈善捐赠情境下让受试者与 8 种不同语言人格的 LLM 对话代理（CA）交互，考察 CA 的态度、权威与推理三种语言特质如何影响用户对 CA 的感知、情绪反应及捐赠决策。

**💡 创新点**

创新点在于首次系统性研究三种语言人格特质（态度、权威、推理）的组合对用户感知与行为的交互效应，发现“悲观”态度的 CA 能在情绪负面但信任度低的情况下诱导更多捐赠，揭示了潜在的“情绪暗黑模式”。

**🔧 技术方法**

使用 GPT‑4o 生成对话文本，利用 LIWC 对生成文本进行语言特征检验，采用 2×2×2 因子实验设计与线性回归、ANOVA、Kruskal‑Wallis 等统计方法分析数据。

**📊 数据集**

数据集：来自 Prolific 的 360 名英语母语者（欧盟/英国）参与的线上实验，使用虚拟 10 欧元捐赠任务，CA 代表的虚构动物保护慈善组织。

**📈 对比分析**

方法上并未在捐赠金额上观察到显著差异，但通过对信任、亲密度、情感相关性等感知变量的回归，发现其对捐赠的预测效应显著，效果大小一般（η² 0.04‑0.08）。

**⚠️ 局限性**

局限性包括单次短暂交互、虚构慈善与虚拟捐赠场景限制生态有效性；语言特质的感知不完全一致（权威识别低）；样本仅为英语欧盟/英国受试者，跨文化与真实情境验证仍需进一步研究。

---

## 266. Nonlinear Predictive Control of the Continuum and Hybrid Dynamics of a Suspended Deformable Cable for Aerial Pick and Place

**arXiv ID:** 2602.17199 | [PDF](https://arxiv.org/pdf/2602.17199v1)

**作者:** Antonio Rapuano `[一作]` (Sapienza University of Rome), Antonio Franchi `[通讯]` (University of Twente)

**通讯引用:** 8177 | [OpenAlex ID](https://openalex.org/A5001771133)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一套混合建模与控制框架，利用PDE离散模型、POD降阶以及非线性混合MPC，实现无人机通过可伸缩柔性缆绳实时操控载荷。

**💡 创新点**

创新点在于首次将连续缆绳动力学、模型降阶与混合预测控制整合到同一管线，既保留了主要变形模态，又能在线处理载荷的接触/脱离事件。

**🔧 技术方法**

主要技术包括有限差分法离散PDE、Proper Orthogonal Decomposition (POD) 降阶、非线性混合MPC（HiLQR/RTI求解器）以及分段规划与对数障碍同胚。

**📊 数据集**

使用的是基于高精度FDM仿真的生成数据集，即从无人机执行正弦轨迹时的缆绳动态快照，未使用公开实验数据。

**📈 对比分析**

通过与完整FDM模型对比，降阶模型能捕获99%以上能量，误差低于1%；在MPC中RTI求解器平均求解时长16.4 ms（满足25 ms采样），完整HiLQR更稳健但耗时约32 ms；在实际轨迹跟踪和障碍规避实验中均实现了无碰撞、误差可接受的控制。

**⚠️ 局限性**

局限性包括仅针对单绳情境，假设缆绳完全柔性且无内阻；未考虑自碰撞、复杂松弛或多绳交互；对训练快照的依赖导致模型对未知动态的鲁棒性有限；实验验证仍待进一步完成。

---

## 267. Non-Invasive Anemia Detection: A Multichannel PPG-Based Hemoglobin Estimation with Explainable Artificial Intelligence

**arXiv ID:** 2602.17290 | [PDF](https://arxiv.org/pdf/2602.17290v1)

**作者:** Garima Sahu `[一作]` (Chhattisgarh Swami Vivekanand Technical University), Nachiket Tapas `[通讯]` (Chhattisgarh Swami Vivekanand Technical University)

**通讯引用:** 1068 | [OpenAlex ID](https://openalex.org/A5081938166)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于多通道PPG和可解释人工智能的无创血红蛋白估计与贫血筛查框架。

**💡 创新点**

将四波长PPG与受试者级特征聚合、LightGBM回归和SHAP解释相结合，首次实现了无创、可解释的血红蛋白估计及基于WHO阈值的贫血判定。

**🔧 技术方法**

多波长PPG信号预处理、统计与频谱特征提取、LightGBM梯度提升回归、SHapley Additive Explanations (SHAP) 解释、WHO阈值决策。

**📊 数据集**

使用公开的 Liang Yongbo 多通道PPG 数据集（152 名成人受试者），包含四波长 PPQ 与参考血红蛋白值。

**📈 对比分析**

与随机森林、CatBoost、XGBoost 等模型对比，LightGBM 在测试集上达 MAE≈8.5±1.27 g/L、RMSE≈8.21 g/L，Bland‑Altman 分析显示无显著系统偏差，证明模型性能优异。

**⚠️ 局限性**

仅覆盖成人数据，缺少儿童/孕妇等高危人群；缺乏真实贫血标签；模型对个体差异与光学衰减敏感，需更大多样化样本提升泛化能力。

---

## 268. Physics Encoded Spatial and Temporal Generative Adversarial Network for Tropical Cyclone Image Super-resolution

**arXiv ID:** 2602.17277 | [PDF](https://arxiv.org/pdf/2602.17277v1)

**作者:** Ruoyi Zhang `[一作]`, Liling Zhao `[通讯]` (Nanjing University of Information Science and Technology)

**通讯引用:** 92 | [OpenAlex ID](https://openalex.org/A5101304813)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种结合物理编码的时空生成对抗网络 PESTGAN，用于热带气旋卫星图像的 4 倍超分辨率重建。

**💡 创新点**

将流体动力学的涡度方程通过受限卷积在潜在空间中逼近，形成 PhyCell 并在解耦生成器中引入物理先验；同时使用时空双判别器提升空间细节和时间一致性。

**🔧 技术方法**

物理编码生成器（Disentangled Dual-Branch）+ PhyCell（受限卷积预测-校正），双判别器（空间 D_S 与时间 D_T），联合损失包含 L1、特征匹配、对抗、卷积核约束、统计一致性等。

**📊 数据集**

Digital Typhoon 数据集，包含 15000 张 512×512 红外云图（2018–2022），测试使用 2022 年 1、14 号台风序列。

**📈 对比分析**

与 TDAN、EDVR、BasicVSR、RealBasicVSR、RealViformer 进行对比，PESTGAN 在 PSNR 30.31 dB、SSIM 0.8656 上均位居第一，且在视觉上恢复了更清晰的云纹与时间连续性。

**⚠️ 局限性**

仅在 4 倍放大任务上验证，且对极端天气或其他气象场景的泛化尚待进一步评估；物理编码网络结构较复杂，训练和推理成本相对较高。

---

## 269. Learning a Latent Pulse Shape Interface for Photoinjector Laser Systems

**arXiv ID:** 2602.17263 | [PDF](https://arxiv.org/pdf/2602.17263v1)

**作者:** Alexander Klemps `[一作]` (Hamburg University of Technology), Nihat Ay `[通讯]` (Hamburg University of Technology)

**通讯引用:** 2797 | [OpenAlex ID](https://openalex.org/A5065926474)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

学习并构建了一种可微分的低维潜在空间，用于表示光子加速器中激光脉冲的纵向形状，并实现高保真重构与下游束流动力学的高效探索。

**💡 创新点**

首次利用Wasserstein自编码器在物理可解释的潜在几何上学习脉冲形状，支持连续、平滑的插值，且能将实验数据自然嵌入到训练得到的潜在流形中。

**🔧 技术方法**

采用卷积Wasserstein Autoencoder、最大均值差异（MMD）正则化、逆变换采样与Wasserstein距离插值等技术，形成完整的生成与分析框架。

**📊 数据集**

使用了1万个模拟脉冲（不同包络形状、阶数、色散参数）以及109条实验测得的红外/紫外脉冲序列作为训练与测试数据。

**📈 对比分析**

与β-VAE 在 MSE、SNR、潜在空间与数据空间距离相关性等指标对比，WAE 在 MSE 上提升至 2.4e-4（≈28–29 dB SNR），并在潜在空间保持 0.70–0.78 的相关性，明显优于 β‑VAE 的 0.16–0.19。

**⚠️ 局限性**

仅针对纵向脉冲形状，未覆盖空间/时空调制；潜在空间对能量标准化敏感，可能限制对能量变化的表达；插值优化相较于线性插值收益有限。

---

## 270. FRAPPE: Infusing World Modeling into Generalist Policies via Multiple Future Representation Alignment

**arXiv ID:** 2602.17259 | [PDF](https://arxiv.org/pdf/2602.17259v1)

**作者:** Han Zhao `[一作]` (Westlake University), Donglin Wang `[通讯]` (Westlake University)

**通讯引用:** 1520 | [OpenAlex ID](https://openalex.org/A5100665183)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种两阶段微调框架，利用未来表示对齐与并行渐进扩展（Future Representation Alignment via Parallel Progressive Expansion）来增强预训练的扩散型视觉语言动作（VLA）模型的世界建模能力和推理性能；

**💡 创新点**

创新点在于将未来帧表示与多种视觉基础模型的对齐相结合，设计了Mixture-of-Prefix-and-LoRA架构和路由器，并在中间训练阶段实现单流对齐到多流的迁移，从而显著提升数据效率、参数效率与泛化性能；

**🔧 技术方法**

核心技术包括扩散Transformer（RDT）、LoRA与前缀调优、跨流多教师对齐（CLIP/DINOv2/ViT）、轻量级路由网络、负载平衡损失、停梯度教师编码以及并行推理与CUDA Graph加速；

**📊 数据集**

实验使用了RoboTwin 2.0仿真基准（Easy/Hard 两种环境）、真实世界AgileX双臂机器人数据（基本与长序列任务）、以及多来源人类第一人称视角视频（任务相关与任务无关的Web规模数据）进行微调与评估；

**📈 对比分析**

在与DP、VPP、RDT、π₀、π₀.₅等基线对比中，本文方法在Easy 设置下取得最高平均成功率，在 Hard 设置下超越π₀.₅；在小模型RDT-130M上也保持竞争力；在真实世界长序列任务中成功率提升约20%；通过与人类视频共训练，可在少量机器人轨迹下显著提升难抓握物体的成功率；

**⚠️ 局限性**

局限性包括：在高扰动的 Hard 环境下仍存在一定性能瓶颈；多流并行方案对显存与算力要求更高；多教师对齐过程中不同表示空间的差异仍可能导致收敛不稳定；依赖预训练VFM的质量与覆盖范围；中间训练阶段增加了训练复杂度与时间。

---

## 271. On the Value of Base Station Motion Knowledge for Goal-Oriented Remote Monitoring with Energy-Harvesting Sensors

**arXiv ID:** 2602.17247 | [PDF](https://arxiv.org/pdf/2602.17247v1)

**作者:** Sehani Siriwardana `[一作]` (Centre for Wireless Communications), Onel Luis Alcaraz López `[通讯]` (Centre for Wireless Communications)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文设计并求解了能量收集传感器在移动接收机情境下的目标导向远程监测系统，利用POMDP转MDP并通过相对价值迭代获得最优采样与传输策略。

**💡 创新点**

创新点在于将基站移动导致的时变信道纳入决策框架，并利用信道状态信息显著降低平均失真。

**🔧 技术方法**

采用的技术包括POMDP建模、Belief MDP转化、相对价值迭代（RVIA）、最大似然（ML）与最小均方失真（MMD）估计以及Markov源与信道模型。

**📊 数据集**

实验数据基于人工生成的Markov源转移矩阵、失真矩阵以及两种移动监视器的信道状态矩阵，并未使用公开数据集。

**📈 对比分析**

通过与基线策略（能量足则采样+传输）和恒定信道假设对比，结果显示引入移动信道知识可将平均失真降低10%至42%。

**⚠️ 局限性**

局限性包括仅考虑单传感器、理想的信道状态信息、离散能量与信道模型；未探讨多传感器情境、信道状态误差及实现复杂度。

---

## 272. Web Verbs: Typed Abstractions for Reliable Task Composition on the Agentic Web

**arXiv ID:** 2602.17245 | [PDF](https://arxiv.org/pdf/2602.17245v1)

**作者:** Linxi Jiang `[一作]` (Ohio State University), Suman Nath `[通讯]` (Microsoft Research)

**通讯引用:** 7864 | [OpenAlex ID](https://openalex.org/A5024224291)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出 Web Verbs 抽象层，将网站功能封装为类型化函数，并实现原型在 13 个网站上进行验证；

**💡 创新点**

创新点在于为 Web 行为统一高层次语义化接口，兼容 API 与浏览器操作，提升可靠性、效率与可验证性；

**🔧 技术方法**

技术手段包括 LLM 代码合成（GitHub Copilot）、Playwright 自动化、NLWeb 向量检索、Java 类型签名和结构化 DocString；

**📊 数据集**

使用的数据集为 13 个网站的 70+ 预定义 Verb、100 个多站点任务基准以及 10 个单站点任务进行评测；

**📈 对比分析**

与两种基线 GUI 代理（无结构和工具箱）在同一 Claude 3.7 Sonnet 模型下对比，verb 基线实现 100% 成功率，基线多失效；执行时间上 verb 方案平均比基线快 2.7–8.3 倍；

**⚠️ 局限性**

局限性包括：原型仅覆盖少量网站，需要标准化 stable‑public‑locator 与全局 Verb 注册；向量检索与安全权限尚未充分优化；基准覆盖范围有限，社区协同仍是关键。

---

## 273. CounterFlowNet: From Minimal Changes to Meaningful Counterfactual Explanations

**arXiv ID:** 2602.17244 | [PDF](https://arxiv.org/pdf/2602.17244v1)

**作者:** Oleksii Furman `[一作]` (Wrocław University of Science and Technology), Marek Śmieja `[通讯]` (Faculty of Mathematics and Computer Science, Jagiellonian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于条件生成流网络（Conditional Generative Flow Networks, C-GFN）的连续-离散混合表格数据的多样化、稀疏且符合约束的反事实解释生成方法，能够在推理时通过动作屏蔽实现可定制的可操作性约束。

**💡 创新点**

核心创新在于：①将反事实生成建模为条件流网络中的序列决策过程，使样本按用户定义的奖励函数（有效性、稀疏性、近似度、可行性）概率生成；②采用两阶段动作空间（特征索引→新值）统一处理连续与离散特征；③通过动作屏蔽在推理阶段实时强制执行可操作性约束，避免重新训练或后处理；④可通过奖励权重精细控制多目标权衡。

**🔧 技术方法**

使用的技术包括：生成式流网络（GFlowNets）、条件奖励函数（多项式乘积形式）、两阶段动作策略（软max）、动作屏蔽（约束可操作性）、离散化连续特征（等频 binning）以及基于 LOF 的可行性评估。

**📊 数据集**

在八个公开基准数据集上评估，涵盖连续与离散特征：Adult、German Credit、Graduate Admission、Student Performance、Bank、Default、GMC、Lending Club；每个数据集有两种评估协议（A：全部离散；B：连续+离散）。

**📈 对比分析**

与优化方法（DiCE、COPA）、采样方法（MCCE）和生成方法（L2C、C-CHVAE、DiCoFlex）比较。实验显示：①在协议A下实现近乎100%有效率、覆盖率和唯一约束满足，且在稀疏性、近似度和多样性上均优于大多数基线；②在协议B下在稀疏性、近似度、可行性方面均优于DiCE、C-CHVAE，且在多样性上与DiCoFlex相当。整体实现了更优的有效性-稀疏性-多样性-可行性折中。

**⚠️ 局限性**

局限性包括：①离散化过程中可能丢失细粒度信息，需在 B 参数与近似度/多样性之间折中；②生成过程仍需要一定计算成本，尤其在大规模特征空间；③对极端可操作性约束的处理仅在推理阶段屏蔽，可能导致某些高约束场景下的可行解稀缺；④奖励函数设计需手动调参，影响最终效果。

---

## 274. Mechanistic Interpretability of Cognitive Complexity in LLMs via Linear Probing using Bloom's Taxonomy

**arXiv ID:** 2602.17229 | [PDF](https://arxiv.org/pdf/2602.17229v1)

**作者:** Bianca Raimondi `[一作]` (University of Bologna), Maurizio Gabbrielli `[通讯]` (University of Bologna)

**通讯引用:** 2104 | [OpenAlex ID](https://openalex.org/A5025039355)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM内部表示中Bloom层级的线性可分性，并发现早期层可辨别不同认知层级

**💡 创新点**

提出认知可分离起点(CSO)概念，并证明不同层级在模型早期层即线性可分

**🔧 技术方法**

使用残差流激活提取、线性逻辑回归探测以及欧氏距离几何分析等技术

**📊 数据集**

采用均衡标注的1128条电脑科学课程与EduQG混合数据集（Bloom六级）

**📈 对比分析**

在四种开源LLM上，层5的线性分类准确率约95%，低层表现差，误差主要集中于相邻层级

**⚠️ 局限性**

仅限解码器模型、英文数据，未验证因果性或跨架构/跨语言的一致性

---

## 275. Privacy-Preserving Mechanisms Enable Cheap Verifiable Inference of LLMs

**arXiv ID:** 2602.17223 | [PDF](https://arxiv.org/pdf/2602.17223v1)

**作者:** Arka Pal `[一作]` (Ritual), Micah Goldblum `[通讯]` (Columbia University)

**通讯引用:** 2167 | [OpenAlex ID](https://openalex.org/A5066564672)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种利用隐私保留LLM推理来实现推理验证的协议，减少验证开销并提高速度。

**💡 创新点**

创新点在于将隐私保留技术与模型指纹（logit fingerprint）或噪声注入相结合，实现低成本验证；并首次展示隐私机制可直接用于检测模型替换与篡改。

**🔧 技术方法**

核心技术包括：隐私保留推理（SMPC、FHE、TEE）、logit指纹验证、噪声嵌入器（NoiseEmbedder）与噪声预测器（NoisePredictor）、缓存指纹、随机插入哨兵令牌、注意力掩码调整。

**📊 数据集**

实验数据集包括 Llama‑3.2 Instruct 1B/3B/8B、Qwen 2.5 Instruct 0.5B/1.5B/3B/7B、FineWeb‑Edu（用于微调）以及 Llama‑2‑7B 作为基准模型。

**📈 对比分析**

与 zkLLM 在 Llama‑2‑7B 上对比，使用 SMPC（SIGMA）实现的 Protocol 2 在 LAN 下响应 100 令牌时约 15 倍更快（约 2260 s 对比 35860 s），验证时间也显著更短；在 WAN 环境保持相同趋势。

**⚠️ 局限性**

局限性包括：Protocol 1 易受子集攻击，需要非协同假设；Protocol 2 依赖统计准确率，完整性与可靠性不如零知识证明；需要噪声模块训练与额外前向推理，且对所有隐私机制的支持不完全相同。

---

## 276. Hierarchical Edge-Cloud Task Offloading in NTN for Remote Healthcare

**arXiv ID:** 2602.17209 | [PDF](https://arxiv.org/pdf/2602.17209v1)

**作者:** Alejandro Flores `[一作]` (University of Luxembourg), Symeon Chatzinotas `[通讯]` (Barcelona Supercomputing Center)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并分析了一个三层非地面网络（HAPS、LEO卫星、云服务器）边缘云任务卸载框架，针对远程医疗设备生成的计算任务，设计了自利定价和带宽分配的博弈模型，求解最优卸载决策与资源分配。

**💡 创新点**

创新点在于：①将非地面网络与云端联合成层级架构并同时考虑各层成本与收益；②采用博弈论与凸优化相结合的方式实现自利定价与最优带宽分配；③通过阈值式价格设定保障延迟约束并避免远程不可行性；④提出公平的Pareto最优带宽分配算法。

**🔧 技术方法**

技术手段包括：多接入边缘计算（MEC）、非地面网络（NTN）链路模型、Rician LoS-MIMO信道模型、延迟与成本建模、Stackelberg博弈、凸优化（Lagrange乘子、二分法）、最优带宽分配算法。

**📊 数据集**

使用三种医疗数据场景（大负载心脏超声、介质负载心电图、低负载光电容积脉搏），通过仿真生成任务尺寸与计算密度，采用假设的网络参数（频段、带宽、功率等）进行实验。

**📈 对比分析**

通过与固定最高任务成本（不进行优化）和无带宽优化的基准方案比较。实验结果表明：①在虚拟延迟成本cτ较高时，任务更倾向于卸载到容量更大的节点；②自利定价与带宽优化提升了云端与HAPS的效用；③不优化带宽在cτ低时对GD成本影响最大；④大型任务多在云端处理，小型任务多本地处理，降低了远程不可行性。

**⚠️ 局限性**

局限性：仅考虑了计算资源分配而未进一步优化；未对真实网络环境中的不确定性（链路失效、动态任务生成）做深入分析；算法对参数（成本、延迟阈值）敏感；未来工作需加入资源动态调度与不可行性管理。

---

## 277. SoftDTW-CUDA-Torch: Memory-Efficient GPU-Accelerated Soft Dynamic Time Warping for PyTorch

**arXiv ID:** 2602.17206 | [PDF](https://arxiv.org/pdf/2602.17206v1)

**作者:** Ron Shapira Weber `[一作]` (Ben-Gurion University of the Negev), Oren Freifeld `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 1321 | [OpenAlex ID](https://openalex.org/A5082095784)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个开源的 PyTorch GPU 库，能够在 GPU 上高效计算 SoftDTW，解决了原有实现的序列长度限制、数值不稳定和高内存占用等问题。

**💡 创新点**

创新点包括：①使用 tiled anti‑diagonal kernel 去除 1024 长度上限；②采用 log‑space 反向传播避免小温度 γ 下的数值溢出；③实现融合距离计算（fused mode），显著降低内存占用（最高可达 98%）。此外，还提供了 Soft‑DTW barycenter 的实现。

**🔧 技术方法**

主要技术手段有：CUDA kernel（按 anti‑diagonal tiled 调度）、log‑sum‑exp 在 log 空间计算、基于平方欧氏距离的代数重写实现 on‑the‑fly 距离计算、PyTorch autograd 集成、FP32 计算以及多线程/多块的并行调度。

**📊 数据集**

实验主要使用合成数据（block‑wave 以及随机长度/维度的序列）进行基准测试，未涉及真实工业或公开数据集。

**📈 对比分析**

与 Maghoumi 的原始 GPU 实现进行对比，评估指标为峰值 GPU 内存和前向后向总耗时。结果显示：①支持任意长度序列（>1024）；②在内存方面相较 Maghoumi 减少 91–98%；③在不使用 fused 模式时速度最快，而 fused 模式虽然慢 10–15×，但在显存受限时仍优于 Maghoumi，并且可在长序列下保持可用。

**⚠️ 局限性**

局限性包括：①fused 模式的运行时间相对较慢，主要因每个 anti‑diagonal 步骤重新计算三次距离；②需要频繁的 kernel 启动，长序列会产生大量调用开销；③当前仅支持 FP32，未提供 FP16/BF16；④归一化 SoftDTW 只适用于等长序列；⑤对极长序列的 persistent‑kernel 或 CUDA‑graph 方案尚未实现。

---

## 278. Security at the Border? The Lived Experiences of Refugees and Asylum Seekers in the UK

**arXiv ID:** 2602.17280 | [PDF](https://arxiv.org/pdf/2602.17280v1)

**作者:** Arshia Dutta `[一作]` (Royal Holloway University of London), Rikke Bjerg Jensen `[通讯]` (Royal Holloway University of London)

**通讯引用:** 305 | [OpenAlex ID](https://openalex.org/A5026642755)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在一家伦敦移民福利慈善机构进行为期三个月的参与式观察，并与六名难民/寻求庇护者及六名案例工作者进行半结构化访谈，研究英国边境技术与移民系统对难民与寻求庇护者长期安全感、归属感的影响。

**💡 创新点**

创新点在于：① 把边境技术体验与后续长期安全与归属感紧密关联，揭示“嫌疑默认”与情绪化体验；② 提出边境前的创伤知情参与式设计干预；③ 强调案例工作者在技术解释与情绪支持中的角色，并讨论在设计与政策层面如何平衡控制与关怀。

**🔧 技术方法**

采用质性研究方法——参与式观察、半结构化访谈、主题分析；并未使用任何计算机技术或算法实现。

**📊 数据集**

数据集为访谈记录（约12次访谈，平均55–71分钟）和现场观察笔记，包含6名客户（3名寻求庇护者、3名难民）与6名案例工作者（CEO、助理、律师等）在一次伦敦慈善机构的对话与现场行为。

**📈 对比分析**

研究不涉及对比实验或性能评估；所有发现均基于质性描述与主题归纳，无量化指标或对照组。

**⚠️ 局限性**

局限性包括：单一伦敦慈善机构的案例限制外推；访谈受语言中介影响，可能引入解释偏差；三个月观察时间短，无法捕捉长期变化；研究者自身移民身份可能影响数据收集与解释；数据不可公开共享，影响可重复性。

---

## 279. RLGT: A reinforcement learning framework for extremal graph theory

**arXiv ID:** 2602.17276 | [PDF](https://arxiv.org/pdf/2602.17276v1)

**作者:** Ivan Damnjanović `[一作]` (University of Niš), Dragan Stevanović `[通讯]` (Abdullah Al Salem University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 RLGT 框架，构建了可用于极限图论的强化学习系统，并利用该框架在多个图论猜想上成功生成反例，验证了其有效性。

**💡 创新点**

创新点包括：① 统一并扩展了之前的 RL 方法，支持有向图、无向图以及多色边；② 设计了八种图表示格式和批量化向量化操作；③ 提供了九种环境和三种 RL 算法（Deep Cross‑Entropy、REINFORCE、PPO），实现了高度模块化的结构。

**🔧 技术方法**

采用了强化学习技术（DCE、REINFORCE、PPO）、神经网络（PyTorch）、批量化图形表示与向量化运算，以及传统图论工具（e.g., Laplacian spectrum 计算）结合的技术栈。

**📊 数据集**

实验数据主要是框架内部自动生成的图批次（不同大小、不同边颜色、不同图类型），未使用公开图集；所有图均在运行时按需构造。

**📈 对比分析**

与先前的 Wagner/Ghebleh 等实现对比，Deep Cross‑Entropy + Linear Build 环境在求解拉普拉斯谱半径、能量-匹配数等猜想时取得与或优于之前的最优结果；通过最佳得分随步数变化曲线和获得的反例展示了系统的效率与可扩展性。

**⚠️ 局限性**

局限性包括：① 对于负奖励（如非连通图）较敏感，导致 REINFORCE/PPO 训练不稳定；② 某些环境（全局/局部）需精细初始化或使用基线才能获得稳定收敛；③ 目前仅支持确定性任务，非确定性图生成与更复杂图性质的处理尚待扩展；④ 对极大规模图的计算开销仍有挑战。

---

## 280. TAPO-Structured Description Logic for Information Behavior: Procedural and Oracle-Based Extensions

**arXiv ID:** 2602.17242 | [PDF](https://arxiv.org/pdf/2602.17242v1)

**作者:** Takao Inoué `[一作]` `[通讯]`, Takao Inoué

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种新的描述逻辑框架TAPO–DL，扩展了经典描述逻辑的T/A盒结构，引入了程序盒（P–Box）和预言机盒（O–Box），并给出了基于层状层次和层射的sheaf理论语义，展示了如何用程序化、交互式、动态方式建模信息行为。

**💡 创新点**

创新点主要包括：①将程序化的命令式语言嵌入描述逻辑中，以实现概念驱动的迭代与条件操作；②引入预言机盒允许系统与外部信息源交互，形成可控的外部验证机制；③使用sheaf理论将上下文信息建模为局部截面与全局结构的可粘合关系，从而在语义层面捕捉信息生成与稳定的动态过程。

**🔧 技术方法**

技术手段包括：描述逻辑基础（T–Box、A–Box）、程序化命令式语言（P–Box）与其大步语义、预言机交互模型（O–Box）、以及基于部分序的sheaf理论语义和层射的上下文一致性。

**📊 数据集**

该工作为理论框架，未使用具体数据集进行实验；所示示例主要是传感器信号、分布式感知等抽象场景。

**📈 对比分析**

由于缺乏实验评估，本文并未给出数值比较或性能指标。其主要贡献在于提出概念模型与语义框架，并通过理论推导说明其在信息行为建模中的潜在优势。

**⚠️ 局限性**

局限性包括：缺乏实证验证，无法检验框架在真实系统中的可行性和性能；对大规模数据和动态环境的计算复杂度尚未分析；预言机盒的交互方式和安全性需要进一步细化。

---

## 281. HiMAP: History-aware Map-occupancy Prediction with Fallback

**arXiv ID:** 2602.17231 | [PDF](https://arxiv.org/pdf/2602.17231v1)

**作者:** Yiming Xu `[一作]` (Leibniz University Hannover), Monika Sester `[通讯]` (Leibniz University Hannover)

**通讯引用:** 4761 | [OpenAlex ID](https://openalex.org/A5020817045)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 HiMAP，一种不依赖多目标跟踪的轨迹预测框架，能够从无身份检测序列中重建历史轨迹并生成多模态未来预测。

**💡 创新点**

创新点在于：①利用时空不变的占据图将过去检测映射到地图上，构建无 ID 的历史表示；②引入历史查询模块，基于当前状态迭代检索占据图中的个体轨迹；③将重建的历史与 DETR 风格解码器结合，实现鲁棒的多模态预测。

**🔧 技术方法**

技术手段包括：QCNet 的时空不变编码、交叉注意力、门控机制、GRU 时序编码、DETR 样式解码器以及跨模态注意力和混合损失。

**📊 数据集**

使用 Argoverse 2（以及 Argoverse 1 用于验证）数据集进行训练与评估。

**📈 对比分析**

在无跟踪设置下，HiMAP 与基线相比取得显著提升：minFDE_6 下降 11%，minADE_6 下降 12%，MR_6 降低 4%，并且性能与最优跟踪方法相近；在跟踪可用时，HiMAP 的性能相对稳定，可与 QCNet 等方法竞争。

**⚠️ 局限性**

局限性：需要足够的历史检测信息；当历史长度非常短或检测噪声较大时重建效果受限；在长时间跟踪可用时，仍无法完全取代跟踪方法的细粒度历史信息；推理时每个历史步长会带来一定延迟。

---

## 282. NotebookRAG: Retrieving Multiple Notebooks to Augment the Generation of EDA Notebooks for Crowd-Wisdom

**arXiv ID:** 2602.17215 | [PDF](https://arxiv.org/pdf/2602.17215v1)

**作者:** Yi Shan `[一作]` (Fudan University), Siming Chen `[通讯]` (Fudan University)

**通讯引用:** 4197 | [OpenAlex ID](https://openalex.org/A5050391600)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

无法获取论文内容，无法说明

**💡 创新点**

无法获取论文内容，无法说明

**🔧 技术方法**

无法获取论文内容，无法说明

**📊 数据集**

无法获取论文内容，无法说明

**📈 对比分析**

无法获取论文内容，无法说明

**⚠️ 局限性**

无法获取论文内容，无法说明

---

## 283. Selective Training for Large Vision Language Models via Visual Information Gain

**arXiv ID:** 2602.17186 | [PDF](https://arxiv.org/pdf/2602.17186v1)

**作者:** Seulbi Lee `[一作]` (Seoul National University of Science and Technology), Sangheum Hwang `[通讯]` (Seoul National University of Science and Technology)

**通讯引用:** 1842 | [OpenAlex ID](https://openalex.org/A5091438057)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于困惑度的视觉信息增益（VIG）指标，并利用该指标实现样本与标记级别的选择性训练，从而提升大型视觉语言模型的视觉依赖性并减少语言偏差。

**💡 创新点**

创新点在于：①给出一种可分解到样本与标记的定量度量，精准捕捉视觉信息对预测的不确定性降低；②基于该度量实现数据级别的自适应筛选，既过滤弱视觉依赖样本，又只对高视觉增益标记计算损失；③展示该策略在不改动模型结构或推理时产生额外开销的前提下，可显著提升视觉理解与消除幻觉的性能。

**🔧 技术方法**

核心技术包括：①利用对数困惑度（perplexity）差值定义VIG；②对齐训练后计算图像-文本对的两种前向推理（含/不含视觉），得到样本和标记级VIG；③对训练集按VIG排序，选取前p%样本；④在这些样本中使用相同阈值对标记做掩码，仅对高VIG标记参与梯度更新；⑤在实验中使用LLaVA‑1.5/13B、ShareGPT4V模型，评估多种视觉问答与幻觉基准。

**📊 数据集**

使用的主要数据集包括：LLaVA指令调优集（约665k条，其中≈625k为多模态样本）；MS‑COCO、GQA、POPE、MMVet、MMBench、DocVQA、CHAIR、MMHal等公开基准，用于评估视觉理解与幻觉鲁棒性。

**📈 对比分析**

与基线及其他视觉偏差抑制方法（VCD、PAI、VAR、LACING）对比，VIG训练在所有视觉理解指标上均优于原始模型，且在幻觉指标上表现更佳；在使用相同训练样本数量的条件下，VIG选择能显著提高性能（如LLaVA‑1.5 7B在仅使用38M标记时就超过全量训练版）；与现有推理时或结构性改造方法组合时还能获得互补提升。

**⚠️ 局限性**

主要限制：①计算VIG需要额外一次前向推理，导致一次性计算成本较高；②目前仅在LLaVA‑1.5与ShareGPT4V两大架构上验证，未检验对其他视觉语言模型或不同领域任务的通用性；③在极端高阈值（低样本量）下可能导致对模型鲁棒性产生不利影响，需要根据任务平衡阈值。

---

## 284. GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation

**arXiv ID:** 2602.17200 | [PDF](https://arxiv.org/pdf/2602.17200v1)

**作者:** Ye Zhu `[一作]` (École Polytechnique), Olga Russakovsky `[通讯]` (Princeton University)

**通讯引用:** 44846 | [OpenAlex ID](https://openalex.org/A5022811687)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在固定文本提示下，提出 Geometry-Aware Spherical Sampling (GASS) 通过几何投影在 CLIP 嵌入空间上实现多样性增强。

**💡 创新点**

创新点在于将生成多样性分解为 prompt‑dependent 与 prompt‑independent 两个正交方向，利用几何扩散和梯度优化可控地提升这两类多样性。

**🔧 技术方法**

使用 CLIP 嵌入的正交投影、随机搜索确定主残差方向、梯度优化与重正则化等技术实现采样引导。

**📊 数据集**

主要实验数据集为 ImageNet‑1K 与 DrawBench，基准为 SD2.1 与 SD3‑M 等冻结模型。

**📈 对比分析**

与四种主流采样增强方法（PG、CADS、IG、SPELL）对比，GASS 在多样性指标（VS、SPP）上显著提升，且在质量与语义一致性指标上保持或略优。

**⚠️ 局限性**

局限在于只在 CLIP 空间工作，难以直接控制更细粒度属性；对高复杂提示的提升有限；依赖 CLIP 预训练导致潜在偏差。

---

## 285. Towards Cross-lingual Values Assessment: A Consensus-Pluralism Perspective

**arXiv ID:** 2602.17283 | [PDF](https://arxiv.org/pdf/2602.17283v1)

**作者:** Yukun Chen `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35195 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 X-Value 这一跨语言价值评估基准，旨在衡量大语言模型对内容深层价值的判断能力。

**💡 创新点**

创新点在于：①构建覆盖 18 种语言、7 个全球敏感领域、超过 5,000 条 QA 对的规模化数据集；②采用两阶段注释框架，先判定议题是否属于全球共识或多元化，再评估答案的价值适当性；③将样本划分为易/难两层，提升评估的区分度。

**🔧 技术方法**

主要技术包括：基于 Schwartz 基本价值理论的领域分层；人工两轮本地化标注与三方仲裁；以及利用 GPT‑5.2、Gemini‑3、Claude‑Opus、Qwen3 系列等 LLM 进行自动评估。

**📊 数据集**

使用的数据集是自研的 X‑Value 数据集，包含 18 语言、7 个主题域、约 5,000 条 QA 对，且每条均附有“价值适当”或“不适当”的标注。

**📈 对比分析**

实验对比了 8 款前沿 LLM 与多种 Qwen3 参数规模模型，发现总体准确率约 75%，在易层可达 93%+，难层仅 66%+，不同语言和领域表现差异显著。

**⚠️ 局限性**

局限性包括：对隐性或细微价值的判断仍不足；跨语言和跨领域差距大；数据覆盖仍未涵盖所有语言，且评估主要基于二分类标签，缺乏更细粒度的价值度量。

---

## 286. From Labor to Collaboration: A Methodological Experiment Using AI Agents to Augment Research Perspectives in Taiwan's Humanities and Social Sciences

**arXiv ID:** 2602.17221 | [PDF](https://arxiv.org/pdf/2602.17221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 287. Visual Insights into Agentic Optimization of Pervasive Stream Processing Services

**arXiv ID:** 2602.17282 | [PDF](https://arxiv.org/pdf/2602.17282v1)

**作者:** Boris Sedlak `[一作]` (Universitat Pompeu Fabra), Schahram Dustdar `[通讯]` (TU Wien)

**通讯引用:** 37122 | [OpenAlex ID](https://openalex.org/A5004847496)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个面向 Edge 设备的多维自适应自动伸缩平台 MUDAP 与基于回归分析的自代理 RASK，用于在有限资源下动态优化多服务的 SLO 满足情况。

**💡 创新点**

创新点在于将服务特定参数（如模型大小、数据质量）纳入可调动作空间，利用回归模型快速构建环境知识并求解全局优化，实现样本高效、跨服务协同的多维弹性伸缩。

**🔧 技术方法**

技术包括：时间序列监控、REST API 调用、回归分析（结构知识回归）与数值优化求解器、基于实验的可视化展示和交互式演示。

**📊 数据集**

使用了三类感知服务的数据流：视频帧（QR 码识别、Yolov8 目标检测）和激光雷达点云（PC 映射），并在 Edge 设备上实时采集和评估其性能指标。

**📈 对比分析**

与传统 RL（如 Q‑learning）对比，RASK 在 30 次环境干预（≈30 次迭代）内即可将全局 SLO 满足率从 56% 提升至 98%，展示了显著的样本效率和高性能，且在后续 300s 内保持稳定的高满足率。

**⚠️ 局限性**

局限性包括：回归模型依赖专家提供的结构关系，若关系未知需进一步学习；当前实验仅涵盖三种服务，需验证在更大规模、多租户场景中的泛化能力；以及对极端资源波动或网络分区等故障模式的鲁棒性尚未深入评估。

---

## 288. Inferring Height from Earth Embeddings: First insights using Google AlphaEarth

**arXiv ID:** 2602.17250 | [PDF](https://arxiv.org/pdf/2602.17250v1)

**作者:** Alireza Hamoudzadeh `[一作]` (Sapienza University of Rome), Roberta Ravanelli `[通讯]` (University of Liège)

**通讯引用:** 566 | [OpenAlex ID](https://openalex.org/A5073597058)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

使用AlphaEarth 10米分辨率嵌入与IGN DSM对齐，构建并训练轻量级U‑Net与U‑Net++卷积回归模型，实现区域表面高度映射，并与Ridge回归基线对比评估性能。

**💡 创新点**

首次在区域尺度上评估地球嵌入对高度推断的可行性，并提出将轻量级U‑Net++作为嵌入解码器的方案，显著提升跨区域泛化能力。

**🔧 技术方法**

技术包括U‑Net/U‑Net++（ResNet‑18编码器）、AdamW优化器、MSE损失、ReduceLROnPlateau与Early Stopping；对比基线采用岭回归。

**📊 数据集**

输入数据：AlphaEarth Embeddings 64通道（2020年），输出数据：IGN RGE ALTI 5×5 m DSM（10 m插值）；研究区域法国Nouvelle‑Aquitaine约8000 km²，训练70%，测试30%。

**📈 对比分析**

对训练集和测试集计算R²、相关系数、RMSE等指标：U‑Net++在测试集上R²=0.84、RMSE≈16 m；U‑Net R²=0.78、RMSE≈19 m；Ridge R²=0.38、RMSE≈32 m；说明深度模型远优于线性基线。

**⚠️ 局限性**

局限性包括训练与测试区域高度分布不匹配导致泛化下降、DSM与嵌入的时间差异产生偏差、整体RMSE仍高（≈16–19 m），需要更大、多样化训练样本和跨区域验证以提升鲁棒性。

---

## 289. Disjunction Composition of BDD Transition Systems for Model-Based Testing

**arXiv ID:** 2602.17237 | [PDF](https://arxiv.org/pdf/2602.17237v1)

**作者:** Tannaz Zameni `[一作]` (University of Twente), Arend Rensink `[通讯]` (University of Twente)

**通讯引用:** 3904 | [OpenAlex ID](https://openalex.org/A5045219123)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于 BDD 过渡系统（BDDTS）的分歧组合方法，用于合并多个 BDD 场景，进而自动生成更全面的测试用例。

**💡 创新点**

创新点在于：①正式定义了分歧组合操作并证明其等价于两条 BDDTS 语义的逻辑与；②构造了符号语义（执行条件与目标含义），以符号方式推理模型一致性与组合等价；③证明符号语义与具体测试案例的一致性；④在荷兰铁路信息板的工业案例中展示该方法的有效性。

**🔧 技术方法**

采用的技术包括 BDD 过渡系统建模、符号过渡系统（STS）到符号语义的映射、符号语义下的执行条件 (EC) 与目标含义 (GI)、以及基于 LTS 的具体语义，用于自动生成测试用例。

**📊 数据集**

实验使用了荷兰铁路公司提供的列车出发信息板 BDD 场景数据集，覆盖了真实运营系统的多种交互和状态转移。

**📈 对比分析**

与传统 BDD 场景手工实现（如 Cucumber、Reqnroll）相比，本文方法在同一模型上减少了系统初始化次数，并且生成的测试用例更常给出明确的通过/失败判定；在工业案例中，测试覆盖率提升，错误检测率提高，具体性能提升未给出数值但在实测中表现优于基线。

**⚠️ 局限性**

局限性包括：需要先生成完整的 BDDTS，建模成本仍然较高；分歧组合在处理大型系统时可能导致状态空间爆炸；实验仅涵盖单一工业案例，缺乏更广泛的实证验证。

---

## 290. Algorithmic Collusion at Test Time: A Meta-game Design and Evaluation

**arXiv ID:** 2602.17203 | [PDF](https://arxiv.org/pdf/2602.17203v1)

**作者:** Yuhong Luo `[一作]` (Rutgers University), Xintong Wang `[通讯]` (Rutgers University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于Meta-Game的框架，用来评估和分析算法在测试时环境下的定价策略是否会出现协同（价格垄断）行为。

**💡 创新点**

创新点在于：①将预训练的策略与在线学习规则组合成Meta-Strategy，并用Paired Cooperativeness (PC) 与 Cooperative Robustness (CR) 两个指标对策略进行分类；②在Meta-Game中采样Meta-Strategies构建经验型博弈，从而得到Nash Equilibrium，判断协同是否为理性结果；③同时对Q-Learning、UCB和LLM三类算法在不同成本、学习率、初始化等条件下进行系统对比。

**🔧 技术方法**

主要技术包括：强化学习中的Tabular Q-Learning、UCB-variant、LLM在提示工程中的在场学习；经验型博弈理论（EGTA）来构建Meta-Game；状态价值函数与最佳回应计算；对策略进行重新缩放（optimistic/pessimistic Q值初始化）；对Meta-Strategy进行离散化与聚类。

**📊 数据集**

数据集：实验基于合成的重复定价游戏（两家卖家，价格离散化为4或15级），使用logit需求模型生成利润；未使用真实市场数据，而是通过模拟实现多种成本与质量设定。

**📈 对比分析**

比较方法：在不同的学习率、成本对称/不对称、初始化策略下，生成多轮Meta-Game，统计MSNE、PSNE、NE-Regret、Uniform Score（转换为CoI）等指标；结果显示：在对称成本下，Q-Learning的“Colluding” Meta-Strategy与“Robust Colluding” Meta-Strategy可实现约50–70% CoI；UCB相对更易协同；LLM在小样本上也能产生近乎垄断的收益；短时间或惰性初始化可抑制协同。

**⚠️ 局限性**

局限性：①仅考虑两玩家静态定价场景，未覆盖多方竞价或更复杂的需求模型；②Meta-Strategy空间被离散化，可能遗漏更优连续策略；③对LLM的评估受API成本与模型版本限制；④实验使用的是合成游戏，缺乏真实市场验证；⑤未考虑信息不完全、成本不对称下的Bayesian动态博弈。

---

## 291. NTLRAG: Narrative Topic Labels derived with Retrieval Augmented Generation

**arXiv ID:** 2602.17216 | [PDF](https://arxiv.org/pdf/2602.17216v1)

**作者:** Lisa Grobelscheg `[一作]` (Vienna University of Economics and Business), Mark Strembeck `[通讯]` (Vienna University of Economics and Business)

**通讯引用:** 2399 | [OpenAlex ID](https://openalex.org/A5063785888)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了NTLRAG框架，利用检索增强生成（RAG）技术自动生成叙事式主题标签，替代传统关键词列表，提升主题可解释性。

**💡 创新点**

创新点在于将检索增强生成与叙事结构相结合，设计可迭代验证与细化的流程，生成可解释、语义精确的主题叙事。

**🔧 技术方法**

技术实现基于RAG管线：检索采用BM25和向量检索，LLM（Llama 3.2 via Ollama）负责叙事提取、验证与细化，整体使用LangChain/Graph和Pydantic进行协同。

**📊 数据集**

使用了三组社交媒体事件数据集（拉斯维加斯、圣菲、埃尔帕索）共计约6.7 M推文，并结合对应新闻检索集作为上下文来源。

**📈 对比分析**

通过16名评估者的用户研究，将NTLRAG生成的叙事标签与BERTopic关键词列表在可用性（1–3分）进行对比，叙事平均得分2.47/3，高于关键词1.61/3，94.73%情况下叙事优于关键词；Krippendorff's α≈0.38，显示公平一致性。

**⚠️ 局限性**

局限性包括对LLM的依赖，易产生幻觉；验证迭代次数有限，可能导致低质量叙事；需手动介入以提升可靠性；检索源与新闻篇幅对结果影响尚未系统评估。

---

## 292. Visual Model Checking: Graph-Based Inference of Visual Routines for Image Retrieval

**arXiv ID:** 2602.17386 | [PDF](https://arxiv.org/pdf/2602.17386v1)

**作者:** Adrià Molina `[一作]` (Centre de Visió per Computador), Josep Lladós `[通讯]` (Centre de Visió per Computador)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于模型检测的图结构验证与神经代码生成相结合的图像检索框架，能够将自然语言查询转化为逻辑三元组并对候选图像进行逐条验证。

**💡 创新点**

创新点在于将形式化验证方法引入视觉检索，通过生成可执行的视觉例程来逐一核实查询约束，实现可追溯的“满足/不满足”提示，并通过部分验证来重新排序检索结果。

**🔧 技术方法**

使用了大型语言模型（Phi‑4 等）进行查询解析与代码生成，视觉检测 API（如 OWL‑V2）执行生成的视觉例程，结合传统的嵌入式检索（CLIP、BEIT、ALIGN）进行基线对比。

**📊 数据集**

在 MS‑COCO 2017 验证集上进行实验，并将其划分为 COCO‑Easy、COCO‑Hard 和 COCO‑All 三个子集，以测试对复杂描述的能力。

**📈 对比分析**

与多种零样本嵌入检索模型（CLIP、SigLIP、ALIGN、BEIT‑3）进行对比；在 Easy 子集 Rec@1~10 与基线相当，而在 Hard 子集中显著提升；当与 CLIP 等模型联合使用时，可通过重新排序进一步提升召回率。

**⚠️ 局限性**

局限性包括代码合成误差和视觉检测 API 的失败导致噪声，状态爆炸问题在处理多元三元组时仍需控制；模型对命名实体、文化敏感内容的偏见未充分验证；总体提升受制于可执行例程质量。

---

## 293. MDP Planning as Policy Inference

**arXiv ID:** 2602.17375 | [PDF](https://arxiv.org/pdf/2602.17375v1)

**作者:** David Tolpin `[一作]` `[通讯]` (Offtopia), David Tolpin (Offtopia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将马尔可夫决策过程（MDP）规划重新表述为对策略的贝叶斯后验推断，并通过变分序列蒙特卡洛（VSMC）算法在离散域上近似该后验，推断出一组确定性策略的概率分布。

**💡 创新点**

创新点在于：① 用策略本身作为潜在变量，并将其未归一化的对数概率定义为期望回报，从而保留原始期望奖励最优性标准；② 在VSMC中加入两项关键改动——（a）在策略首次访问状态时抽取动作，保证粒子策略的确定性；（b）在一次推断扫描中共享转移随机性，消除因环境噪声导致的粒子权重混淆；③ 通过后验预测采样实现结构化的Thompson采样，提供一种天然的策略层不确定性表征。

**🔧 技术方法**

技术手段包括：贝叶斯推断框架、变分序列蒙特卡洛（VSMC）与粒子重采样、共享随机数机制、确定性策略记忆机制、基于神经网络的策略参数化以及对比实验中使用的离散Soft Actor‑Critic（SAC）算法。

**📊 数据集**

使用了四个典型离散域：网格世界（grid world）、Blackjack、Triangle Tireworld以及Academic Advising（学术指导）问题。

**📈 对比分析**

与SAC进行比较：在平均回报方面两者相近，但VSMC在策略多模态和终止行为上更稳健；SAC因熵正则化倾向于沿边界走，导致某些状态的可行性下降；VSMC不需要额外调节熵权重，调参更直观。实验显示，VSMC在不确定性处理和风险控制上往往优于或至少与SAC持平。

**⚠️ 局限性**

局限性包括：① 目前仅针对离散状态/动作空间，连续域需额外工程实现；② 强制策略确定性可能导致对某些可逆或无效动作过于惩罚，降低探索灵活性；③ 共享转移随机性和状态重访记忆的实现对大规模状态空间成本高；④ 需要先验温度/奖励尺度设定，影响后验收敛；⑤ 算法在大规模多模态问题上仍受限于粒子数与计算开销。

---

## 294. Tree crop mapping of South America reveals links to deforestation and conservation

**arXiv ID:** 2602.17372 | [PDF](https://arxiv.org/pdf/2602.17372v1)

**作者:** Yuchang Jiang `[一作]` (EcoVision Lab, DM3L, University of Zurich), Maxim Neumann `[通讯]` (Google DeepMind)

**通讯引用:** 4962 | [OpenAlex ID](https://openalex.org/A5103132468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

生成了南美洲10米分辨率的树作物地图，并与历史森林覆盖损失、保护区边界关联，揭示树作物扩张与森林砍伐的空间关系。

**💡 创新点**

首次将多模态时空卫星影像与视觉变压器深度学习相结合，构建统一的跨国树作物参考集，实现精细化、全国一致的树作物分类。

**🔧 技术方法**

使用多模态时空视觉变压器（MTSViT）对 Sentinel‑1（SAR）和 Sentinel‑2（多光谱）时序图像进行分割，结合交叉注意力与多尺度编码器。

**📊 数据集**

融合 MapBiomas、SDPT、JRC 等八大公开数据源构成树作物、森林、非森林等八类参考集，并使用 2020 年 Sentinel‑1/2 的季节复合图像作为输入。

**📈 对比分析**

与 JRC Global Forest Cover 2020、Global Forest Type 2020 等现有森林产品对比，模型在测试集上 F1≈89.5%（召回87%，精度92%），树作物与森林覆盖损失重叠约23%，在 EUDR 场景下识别出约 3.3%–5.5% 的误将森林的树作物。

**⚠️ 局限性**

树作物稀疏导致样本不平衡和区域误差，参考数据覆盖不足，尤其是小农多样化种植系统的识别仍存在偏差；与森林产品的差异需进一步实地验证。

---

## 295. 2Mamba2Furious: Linear in Complexity, Competitive in Accuracy

**arXiv ID:** 2602.17363 | [PDF](https://arxiv.org/pdf/2602.17363v1)

**作者:** Gabriel Mongaras `[一作]` (Southern Methodist University), Eric C. Larson `[通讯]` (Southern Methodist University)

**通讯引用:** 2694 | [OpenAlex ID](https://openalex.org/A5023773515)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对Mamba-2进行组件消融，提炼出最核心的简化实现Mamba-2S；进一步引入二阶隐藏状态和指数化隐藏状态，使线性注意力在长序列下达到甚至略优于softmax注意力的准确率，同时保持常数内存消耗。

**💡 创新点**

①通过消融实验定位最重要的组件（卷积输入、softplus A-mask、时间离散化）；②构建最小化的Mamba-2S；③引入二阶隐藏状态提升表达力；④提出指数化隐藏状态实现略高于softmax的性能；⑤在长序列上实现更低的内存需求。

**🔧 技术方法**

线性注意力、Mamba-2架构、卷积输入、A-mask、二阶隐藏状态、指数化隐藏状态、软max化归一化、RMSNorm、Triton自定义核、Flash Attention、KV cache、NIAH检验。

**📊 数据集**

FineWeb（CommonCrawl 2024）用于训练，Nanotron's needle‑in‑a‑hay‑stack 数据集用于评估长上下文记忆。

**📈 对比分析**

在300M和700M模型上，以测试损失和NIAH指标与软max注意力对比。Mamba-2S在长序列（≥8192）上与软max损失相当且内存更低；指数化版本在相同条件下略优软max；训练稳定性在中型模型上需要高精度或自定义核。

**⚠️ 局限性**

对中型模型训练时出现数值不稳定，需使用FP32或混合精度；未探索更高阶隐藏状态或动态隐藏状态大小；验证范围局限于特定head dimension，缺乏对更大规模模型或多头配置的实证。

---

## 296. Insidious Imaginaries: A Critical Overview of AI Speculations

**arXiv ID:** 2602.17383 | [PDF](https://arxiv.org/pdf/2602.17383v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 297. Shortcut learning in geometric knot classification

**arXiv ID:** 2602.17350 | [PDF](https://arxiv.org/pdf/2602.17350v1)

**作者:** Djordje Mihajlovic `[一作]` (University of Edinburgh), Davide Michieletto `[通讯]` (University of Edinburgh)

**通讯引用:** 2613 | [OpenAlex ID](https://openalex.org/A5066158937)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了机器学习在结理论中出现的“快捷学习”现象，揭示现有模型主要利用几何特征而非拓扑不变性进行分类；

**💡 创新点**

提出了基于互信息的快捷学习探测方法，并开发了GEOKNOT采样器生成几何均匀、无偏结集，验证现有模型的几何依赖性；

**🔧 技术方法**

使用互信息诊断、前馈神经网络、writhe矩阵表示、Vassiliev不变量近似、分子动力学（MD）与蒙特卡洛采样等技术；

**📊 数据集**

对比了MD（LAMMPS）低温/高温结集与GEOKNOT生成的结集；

**📈 对比分析**

通过在不同数据集上训练和测试同一模型，发现MD训练模型在GEOKNOT上精度仅为50–70%，GEOKNOT训练模型在MD上亦表现不佳；快捷学习指数τ≈1 说明模型高度依赖几何特征；

**⚠️ 局限性**

当前模型无法学习高阶拓扑不变量；GEOKNOT采样耗时且仅适用于低交叉数结；未能证明writhe矩阵可完全决定Vassiliev不变量。

---

## 298. SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery

**arXiv ID:** 2602.17395 | [PDF](https://arxiv.org/pdf/2602.17395v1)

**作者:** Lorenzo Caselli `[一作]` (University of Florence), Andrew D. Bagdanov `[通讯]` (University of Florence)

**通讯引用:** 6955 | [OpenAlex ID](https://openalex.org/A5064029620)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 SpectralGCD，一种利用 CLIP 跨模态图像-概念相似度和谱过滤的高效 Generalized Category Discovery 方法。

**💡 创新点**

创新点在于：①将图像表示转化为概念混合并用谱分解自动筛选任务相关概念；②结合前向和逆向知识蒸馏，保持跨模态表示的语义质量；③在保持性能的同时显著降低计算开销。

**🔧 技术方法**

使用技术包括：CLIP 的跨模态相似度、概念词典、谱分解（提取协方差矩阵特征向量）、正向/逆向知识蒸馏、对比学习与参数化分类器。

**📊 数据集**

实验数据集包括：CIFAR‑10、CIFAR‑100、ImageNet‑100、CUB、Stanford Cars、FGVC‑Aircraft 以及 Semantic Shift Benchmark。

**📈 对比分析**

与多种单模态与多模态 GCD 方法（SimGCD、PromptCAL、TextGCD、GET 等）对比，SpectralGCD 在所有数据集上实现或超越最佳性能，尤其在新类别上提升显著；同时训练时间与单模态方法相近，显著低于其他多模态方案。

**⚠️ 局限性**

局限性：依赖教师模型与概念词典的覆盖度；当教师知识或词典与目标领域偏离时，效果可能下降；未来工作需进一步降低对外部资源的依赖。

---

## 299. Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature

**arXiv ID:** 2602.17385 | [PDF](https://arxiv.org/pdf/2602.17385v1)

**作者:** Angelo Porrello `[一作]` (University of Modena and Reggio Emilia), Simone Calderara `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 5273 | [OpenAlex ID](https://openalex.org/A5075481810)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种无数据正则化方法，利用Kronecker-Factored Approximate Curvature（KFAC）逼近通用高斯-牛顿矩阵，提升任务算术（Task Arithmetic）中的权重解耦；

**💡 创新点**

创新点在于将表示漂移正则化转化为曲率矩阵近似，既不依赖外部任务数据，又能以常数复杂度聚合多任务信息，并对任务向量的缩放具有鲁棒性；

**🔧 技术方法**

技术核心包括：模型线性化、表示漂移与GGN关联、KFAC矩阵的预计算与合并、以及在训练期间加入无数据正则项；

**📊 数据集**

实验使用了CLIP视觉基础模型在八个分类基准（如ImageNet等）以及T5-base在六个NLP任务（SNLI、MultiNLI、SICK、SciTail、RTE、QNLI）的数据集；

**📈 对比分析**

与现有方法（如τJp、TIES、TSV、ISO）相比，在任务添加与消除上达到或超过state‑of‑the‑art，且不需要额外的α调参，表现稳定；

**⚠️ 局限性**

局限性包括对模型线性化的依赖，在非线性微调场景下效果略逊；KFAC估计仍需前期计算，且在极大模型上内存占用受限。

---

## 300. Computer-Using World Model

**arXiv ID:** 2602.17365 | [PDF](https://arxiv.org/pdf/2602.17365v1)

**作者:** Yiming Guan `[一作]` (Nankai University), Dongmei Zhang `[通讯]` (Microsoft)

**通讯引用:** 11381 | [OpenAlex ID](https://openalex.org/A5100331488)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种两阶段的桌面软件世界模型（CUWM），先用文本描述预测 UI 变化，再将其渲染为下一屏幕截图，从而实现测试时动作搜索。

**💡 创新点**

将 UI 动态拆分为文本抽象与视觉实现两步，利用 GPT-5 自动标注的转移文本，并通过轻量 RL 对文本生成进行结构化、简洁化改进，显著提升了对决策关键 UI 结构的捕捉。

**🔧 技术方法**

采用 Qwen2.5‑VL 预测文本转移，Qwen-Image-Edit 进行图像编辑；训练时使用 LoRA‑SFT + GRPO 强化学习，推理时结合冻结的 LLM 代理进行思考‑执行。

**📊 数据集**

使用 GUI‑360 数据集，包含 Microsoft Office（Word、Excel、PowerPoint）的 UI 交互轨迹，并由 GPT‑5 自动生成的转移描述作为监督标签。

**📈 对比分析**

与仅图像、仅文本及现有图像生成基线（Qwen‑Image‑Edit‑2509、GPT‑Image‑1.5）进行对比，评估指标包括 LLM‑Judge、Action Consistency、PSNR、SSIM、LPIPS、FID、Text Perception；CUWM 在视觉质量、文本准确度上取得最佳表现，并使 GPT‑4o、Qwen3‑VL‑8B 的任务完成率提升 4%–8%。

**⚠️ 局限性**

在文本与图像联合预测时出现跨模态冲突，导致性能下降；错误积累与 VLM 在多模态推理上的能力不足是主要限制。

---

## 301. Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation

**arXiv ID:** 2602.17354 | [PDF](https://arxiv.org/pdf/2602.17354v1)

**作者:** Daniele Malitesta `[一作]` (Université Paris-Saclay), Fragkiskos D. Malliaros `[通讯]` (Université Paris-Saclay)

**通讯引用:** 1875 | [OpenAlex ID](https://openalex.org/A5087251228)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对多模态推荐中缺失模态问题，提出训练无关的图结构推理补全方法。

**💡 创新点**

创新点在于把缺失模态问题转化为物品-物品共购图上的特征插值，并提供四种无训练的图传播策略。

**🔧 技术方法**

使用图卷积/传播（NeighMean、MultiHop、PersPageRank、Heat）以及传统填充和自编码器。

**📊 数据集**

在七个亚马逊数据集（两模态）和MicroLens三模态数据集上评测。

**📈 对比分析**

与传统填充、自编码器和多种多模态推荐器相比，图补全在Recall@20和nDCG@20上显著提升，并能放大多模态与传统推荐的性能差距。

**⚠️ 局限性**

局限在于对图结构质量和特征同质性敏感，冷启动、噪声数据及高缺失率下仍存在性能下降。

---

## 302. Application and Evaluation of the Common Circles Method

**arXiv ID:** 2602.17353 | [PDF](https://arxiv.org/pdf/2602.17353v1)

**作者:** Michael Quellmalz `[一作]` (TU Berlin), Monika Ritsch-Marte `[通讯]` (Medical University of Innsbruck)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究并实现了在光学衍射层析（ODT）中利用公共圆方法对受声波阱作用的微小样本运动进行估计，并结合时间一致性约束实现稳定重建。

**💡 创新点**

在传统完整优化方法的基础上提出了快速、无需旋转参数初始化的公共圆方法，并引入时间平滑正则化与双圆交叉约束来提升噪声鲁棒性和稳定性。

**🔧 技术方法**

采用了 Born 与 Rytov 近似的 Fourier 变换、非均匀 FFT、梯度下降、Nelder–Mead 优化、以及投影到 SO(3) 的极值化技术。

**📊 数据集**

包含合成的球形弹性物质模型、三球体实验样本以及真实的神经母细胞细胞的 3D 复振幅图像。

**📈 对比分析**

通过与全局优化（TV 正则化）方法对比，公共圆方法在噪声实验下平均旋转误差约 10.3°（神经母细胞）或 4.2°（合成），计算时间仅为 0.4–33 s，明显快于完整优化且 PSNR/SSIM 与优化方法相差 0.02–0.07。

**⚠️ 局限性**

方法在极小或接近 180° 的旋转、强噪声或非线性散射（不满足 Born）下精度下降，且仅能估计刚性运动，无法处理形变或大尺度样本。

---

## 303. What Breaks Embodied AI Security:LLM Vulnerabilities, CPS Flaws,or Something Else?

**arXiv ID:** 2602.17345 | [PDF](https://arxiv.org/pdf/2602.17345v1)

**作者:** Boyang Ma `[一作]` (Shandong University), Yue Zhang `[通讯]` (Shandong University)

**通讯引用:** 36370 | [OpenAlex ID](https://openalex.org/A5038484265)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对嵌入式 AI 系统安全进行系统综述，归纳了 LLM 漏洞、CPS 故障和嵌入式特有的失败机制，并提出四个核心洞见：语义-物理不匹配、状态依赖行动后果、误差放大和非组合安全；

**💡 创新点**

创新点在于揭示传统 LLM 与 CPS 视角单独不足之处，提出跨层安全框架并系统化了四大根因；

**🔧 技术方法**

主要采用文献综述与系统化分类技术，对已有攻击与安全模型进行对比分析；

**📊 数据集**

未引入新数据集，主要使用已有的 LLM 与 CPS 攻击案例进行梳理；

**📈 对比分析**

通过对比已有攻击场景与安全假设，阐释四大根因，但未给出实验性能指标；

**⚠️ 局限性**

局限在于缺乏实证验证与实验评测，对新型多模态攻击的覆盖不完整，且跨层安全评估工具尚未成熟。

---

## 304. Socio-Technical Well-Being of Quantum Software Communities: An Overview on Community Smells

**arXiv ID:** 2602.17320 | [PDF](https://arxiv.org/pdf/2602.17320v1)

**作者:** Stefano Lambiase `[一作]` (University of Salerno), Andrea De Lucia `[通讯]` (University of Salerno)

**通讯引用:** 16613 | [OpenAlex ID](https://openalex.org/A5079088548)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对17个开源量子软件仓库进行横断面研究，检测并统计了10种社区臭味的出现率及其相互关联性，揭示量子社区中普遍存在的社交债务问题；

**💡 创新点**

首次系统评估量子软件社区的社群臭味，构建社区臭味共现关系网络，为量子软件工程（QSE）提供域特定的社会技术风险画像；

**🔧 技术方法**

使用csDetector（改进版）从GitHub仓库中提取社区臭味指标，并运用Prevalence Odds Ratio（POR）进行统计关联分析；

**📊 数据集**

采用前期研究收集的115个量子仓库数据集，最终筛选17个满足csDetector条件的仓库；

**📈 对比分析**

通过计算各臭味的流行度和POR矩阵展示正负关联，结果显示多数臭味普遍存在且关联显著，暗示需针对特定臭味组合实施治理；

**⚠️ 局限性**

受限于工具检测范围、样本仅为部分公开仓库、缺乏纵向数据导致无法确定因果关系，且未覆盖所有潜在社群臭味。

---

## 305. Security of the Fischlin Transform in Quantum Random Oracle Model

**arXiv ID:** 2602.17307 | [PDF](https://arxiv.org/pdf/2602.17307v1)

**作者:** Christian Majenz `[一作]` (Technical University of Denmark), Jaya Sharma `[通讯]` (Technical University of Denmark)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

证明了Fischlin变换在量子随机预言机模型（QROM）中保持直线提取可行性，完成了对该变换的后量子安全性证明；

**💡 创新点**

首次使用压缩预言机（compressed oracle）结合马尔可夫尾部界和对称化技术，克服了传统技术无法处理的量子查询相关性问题，证明了Fischlin变换在QROM中可实现直线提取；

**🔧 技术方法**

核心技术包括压缩预言机模拟、量子对称化（symmetrization）、马尔可夫过程与Azuma‑Hoeffding尾部界、量子联合界（quantum union bound）以及自适应预言机重编程技术；

**📊 数据集**

论文为理论安全性证明，无需使用具体数据集；

**📈 对比分析**

通过参数选择使错误概率可忽略，证明提取错误率≤q²·k，其中q为量子查询次数，k为重复次数；该性能与经典Pass’变换相比，提供更小的证明尺寸和更强的后量子安全性；

**⚠️ 局限性**

局限在于证明依赖于特定参数约束（如挑战空间大小、k与l的关系）和假设Σ协议具有唯一响应与特殊可听性，实际实现时需要满足这些约束，且证明不涉及对具体实现的性能评估。

---

## 306. Contact-Anchored Proprioceptive Odometry for Quadruped Robots

**arXiv ID:** 2602.17393 | [PDF](https://arxiv.org/pdf/2602.17393v1)

**作者:** Minxing Sun `[一作]` (Institute of Optics and Electronics Chinese Academy of Sciences), Yao Mao `[通讯]` (Institute of Optics and Electronics Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

利用 IMU 和电机测量实现纯本体感知的接触锚定里程计，支持双足、四足和轮足机器人。

**💡 创新点**

提出统一的接触锚定状态估计框架；引入高度聚类与时衰减修正以消除垂直漂移；使用逆运动学 Cubature 卡尔曼滤波抑制编码器量化噪声；利用多接触几何一致性实现航向漂移校正。

**🔧 技术方法**

IMU、关节角度/速度、关节扭矩（接触判别）、逆运动学模型、Cubature 卡尔曼滤波器、接触高度聚类与时间衰减、轮子滚动补偿、几何航向校正。

**📊 数据集**

真实硬件：Unitree Go2 EDU、Astrall A（点足）、Astrall B、Astrall C（轮足）；仿真：Unitree AlienGo 在 Gazebo 中的平地与楼梯闭环轨迹。

**📈 对比分析**

与激光 SLAM 基准相比，CAPO/CAPO-CKE 在垂直通道和长周期闭环中显著降低漂移；实验中单位误差为：Go2 平面闭环 2.21 m，Astrall A 200 m 平面/15 m 垂直分别 0.16 m/0.22 m；仿真终端误差从 2.08 m 降至 0.62 m。

**⚠️ 局限性**

未显式检测/补偿轮滑、使用固定阈值接触判别、假设地面平坦导致坡面误差、缺乏自适应接触模型、IMU 航向漂移仍存在、IKVel-CKF 计算开销较大。

---

## 307. PersonaMail: Learning and Adapting Personal Communication Preferences for Context-Aware Email Writing

**arXiv ID:** 2602.17340 | [PDF](https://arxiv.org/pdf/2602.17340v1)

**作者:** Rui Yao `[一作]` (City University of Hong Kong), Shengdong Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 5312 | [OpenAlex ID](https://openalex.org/A5060314525)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估 PersonaMail，旨在通过结构化因素探索、沟通单元层级控制和个性化策略复用，提升情境敏感电子邮件的写作效果。

**💡 创新点**

创新点：①利用基于通信理论的因素面板使用户可系统化表达语气与关系意图；②引入沟通单元（Opening、Justification 等）与意图驱动的编辑机制，实现中层细粒度修改；③构建人物‑情境锚点和适应式风格书，支持高效复用成功策略。

**🔧 技术方法**

技术：大语言模型 Gemini 2.5 Flash、prompt engineering、意图‑单位映射、快速修正（QuickFix）模块、人物‑情境锚点适配器、意图驱动重写器。

**📊 数据集**

数据集：实验中 16 名参与者生成的 24 封自定义难度电子邮件（来自工作、教育、个人情境），以及作者构建的 97 篇通信理论论文提炼的 7 类 Persona 与 7 类 Situation 因素清单。

**📈 对比分析**

对照方法：将 PersonaMail 与传统 Gemini+Gmail 交互式写作进行 1:1 对比。结果显示：①首稿质量提升 34.8%（p<0.001）；②复稿质量提升 15%（p=0.002）；③总工时在重复使用后下降 42%（p<0.001）；④认知负荷降低 24.5%（p<0.001）。

**⚠️ 局限性**

局限性：①样本主要为学生/早期职业者，难以推广至高级领导或高危场景；②实验情境聚焦相似任务，未检验跨文化或多任务切换的适用性；③系统依赖用户手动保存锚点，可能强化已有沟通偏好而非鼓励多样化表达。

---

## 308. Polaffini: A feature-based approach for robust affine and polyaffine image registration

**arXiv ID:** 2602.17337 | [PDF](https://arxiv.org/pdf/2602.17337v1)

**作者:** Antoine Legouhy `[一作]` (Hawkes Institute), Hui Zhang `[通讯]` (Hawkes Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种名为Polaffini的基于解剖分割中心点的全局与局部仿射（polyaffine）图像配准框架；

**💡 创新点**

创新点在于：1）利用预训练的深度学习分割模型提取解剖对应点（仅使用区域质心）实现高效且鲁棒的特征匹配；2）采用log‑Euclidean polyaffine（LEPT）将局部仿射结合为全局可微分同胚变形；3）提供可调平滑度参数、闭式解法，兼容传统与深度学习后续配准；

**🔧 技术方法**

技术主要包括：预训练的脑结构分割（FastSurfer/SynthSeg），质心提取，加权线性最小二乘仿射估计，Delaunay图邻域划分，局部仿射闭式求解，log‑Euclidean 加权平均得到SVF，再通过指数映射得到同胚变形；

**📊 数据集**

使用了三大公开脑MRI数据库：IXI、UK Biobank、ADNI（含健康、MCI、AD子组），共计约350对模板‑样本配准；

**📈 对比分析**

与四款主流强度基线配准工具（FLIRT、ANTs‑aff、Anima‑aff、Aladin）比较；在解剖结构Dice分数、失败率、几何差异等指标上，Polaffini‑polyaffine均显著优于所有基线；作为非线性配准（SyN、VoxelMorph）预对齐步骤时，Polaffini‑polyaffine在深度学习非线性配准中提升Dice达约0.1-0.2；

**⚠️ 局限性**

局限包括：1）对分割质量高度依赖；2）质心作为特征点可能在解剖区域较大时稀疏；3）σ平滑度需针对不同分割细粒度调参；4）当前实现未支持非脑图像或其他解剖部位；

---

## 309. SubQuad: Near-Quadratic-Free Structure Inference with Distribution-Balanced Objectives in Adaptive Receptor framework

**arXiv ID:** 2602.17330 | [PDF](https://arxiv.org/pdf/2602.17330v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11904 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套名为SubQuad的端到端管线，用于大规模免疫受体序列的检索、相似度融合与公平聚类，避免传统全二次比较的计算瓶颈；

**💡 创新点**

创新点包括：① antigen‑aware MinHash预过滤实现近次方级候选压缩；② 可学习的动态多模态门控融合对齐、语言模型与图特征；③ 自动公平调优的谱聚类约束，保证稀有抗原特异性克隆的比例表示；

**🔧 技术方法**

采用的技术包括：GPU 并行相似度核、ImmunoBERT 语言模型、MetaNet 门控、RMT 阈值筛选、MinHash LSH、谱聚类与公平约束正则化；

**📊 数据集**

使用的数据集有：VDJdb、McPAS‑TCR、NEPdb 以及合并自十个不同捐献的约一百万 CDR3β 克隆，测试覆盖病毒、肿瘤及自身免疫背景；

**📈 对比分析**

与 BertTCR、TCR‑pMHC、ProtBert 等基线相比，SubQuad 在 10K 序列测试中吞吐率 97.2k seq/s、召回率 0.985、内存 1.4 GB、聚类纯度 92% 以及公平分数 0.91，跨捐献百万级测试仍保持召回>0.96、峰值内存 186 GB，处理时间约 40 min；

**⚠️ 局限性**

局限性包括：公平权重需手工调参；在极端长尾分布下仍可能漏检极稀克隆；GPU 资源依赖明显，分布式扩展和时间序列动力学建模仍待进一步研究。

---

## 310. WebFAQ 2.0: A Multilingual QA Dataset with Mined Hard Negatives for Dense Retrieval

**arXiv ID:** 2602.17327 | [PDF](https://arxiv.org/pdf/2602.17327v1)

**作者:** Michael Dinzinger `[一作]` (University of Passau), Michael Granitzer `[通讯]` (University of Passau)

**通讯引用:** 3630 | [OpenAlex ID](https://openalex.org/A5006866152)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

扩展了WebFAQ数据集到198M QA对，覆盖104种语言，并发布1.25M查询的硬负样本集

**💡 创新点**

采用自建爬虫直接抓取FAQ页面，显著提升多语种覆盖和双语对齐数量，并提出两阶段硬负采样及对比学习/知识蒸馏两种训练策略

**🔧 技术方法**

结合OWLer爬虫、schema.org解析、FastText语言检测、LaBSE/LaBSE embeddings、BM25、BGE‑m3交叉编码器、XLM‑RoBERTa、MultipleNegativesRankingLoss与MarginMSE损失等技术

**📊 数据集**

使用WebFAQ 2.0本身、其硬负样本集（20种语言）以及公开的BERT/跨语言模型做实验

**📈 对比分析**

在WebFAQ、MIRACL‑HN、Mr.Tydi等基准上与BM25和XLM‑RoBERTa Base比较，硬负训练在非英语上提升明显，但随机负样本在整体更稳健

**⚠️ 局限性**

硬负样本中存在误负，知识蒸馏在英文上的性能下降，且仍需更细致的去噪处理，资源规模仍受限于当前抓取方法

---

## 311. Same Meaning, Different Scores: Lexical and Syntactic Sensitivity in LLM Evaluation

**arXiv ID:** 2602.17316 | [PDF](https://arxiv.org/pdf/2602.17316v1)

**作者:** Bogdan Kostić `[一作]`, Alexander Löser `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构造语义保持的词汇和句法层次扰动，系统评估了23个大规模语言模型在MMLU、SQuAD和AMEGA三大基准上的鲁棒性。

**💡 创新点**

创新点在于提出了两种基于语言学原理的意义保持扰动方法，并揭示模型对词汇变化的敏感度高于句法变化、模型规模对鲁棒性的影响呈任务依赖性。

**🔧 技术方法**

采用LLM引导的同义词替换与依存句法分析驱动的句法变换技术，对原始数据进行语义保持扰动。

**📊 数据集**

使用的评测数据集包括多领域的多选问答基准MMLU、文本阅读理解基准SQuAD以及临床指南遵循评测基准AMEGA。

**📈 对比分析**

通过将模型在原始与扰动版本数据集上的准确率、EM/F1/SAS以及指南遵循分数进行对比，并使用McNemar检验、Wilcoxon检验和Kendall τ相关系数评估性能下降与排名波动，结果显示词汇扰动平均导致MMLU 7.72pp、SQuAD 3.38pp及AMEGA 1.25点的准确率下降，且在SQuAD与AMEGA中排名波动显著。

**⚠️ 局限性**

局限性包括仅考察了三种任务与少数几种扰动类型，未覆盖多语言、多模态或更复杂的句法变形；且扰动生成依赖LLM，可能引入不可控偏差。

---

## 312. Some Remarks on Marginal Code Languages

**arXiv ID:** 2602.17309 | [PDF](https://arxiv.org/pdf/2602.17309v1)

**作者:** Stavros Konstantinidis `[一作]` (Saint Mary's University), Stavros Konstantinidis `[通讯]` (Saint Mary's University)

**通讯引用:** 884 | [OpenAlex ID](https://openalex.org/A5046508856)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究了marginal independent languages（k-prefix-free、k-suffix-free等）的定义、可判定性与最大化问题，并给出了统一的判定方法。

**💡 创新点**

提出了基于有限变换器的marginal属性描述法，证明了对于固定k的k-α语言可在多项式时间内判定其可满足性，并给出了新的最大化判定条件；同时对之前未解的PSPACE复杂度问题给出了解决方案。

**🔧 技术方法**

使用有限变换器（transducer）来表示部分顺序，结合值-性（k-valued）判定、图论与NFA交互操作，利用PSPACE和多项式时间算法。

**📊 数据集**

无具体数据集，研究以理论证明为主，实验基于符号化的正则语言和变换器模型。

**📈 对比分析**

相较于之前的复杂度结果，本文在固定k情况下将判定问题降低到多项式时间；对k-infix-free的PSPACE-hard问题给出完整PSPACE可判定性；对finitely-marginal属性提供了O(|A|^2)算法。

**⚠️ 局限性**

当k不是固定常数时，判定问题仍然是PSPACE难度，且在更一般的变换器类中缺乏多项式时间算法；部分最大化判定仍未给出完整条件。

---

## 313. RPDR: A Round-trip Prediction-Based Data Augmentation Framework for Long-Tail Question Answering

**arXiv ID:** 2602.17366 | [PDF](https://arxiv.org/pdf/2602.17366v1)

**作者:** Yiming Zhang `[一作]` (Zhejiang University), Chen Zhao `[通讯]` (New York University)

**通讯引用:** 71883 | [OpenAlex ID](https://openalex.org/A5019034689)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 RPDR 框架，结合合成数据生成、循环预测（Round‑Trip Prediction）挑选易学样本，并用这些高质量增强数据训练稠密检索器，从而显著提升长尾问答的检索性能。

**💡 创新点**

创新点在于：①使用逆向嵌入模型通过循环预测筛选能被可靠重构的样本，保证增强数据的可学习性；②引入动态路由机制，根据查询特征在 BM25 与稠密检索器之间切换，进一步提升性能；③通过上述方法证明，在适当训练后稠密检索器可超越传统的 BM25。

**🔧 技术方法**

采用的技术包括合成数据生成（利用 Wikidata 三元组 + Wikipedia 页面）、逆向嵌入模型 (inverse encoder) + 循环预测、对比学习训练稠密检索器（基于 Contriever）、融合检索-生成（RAG）架构，以及基于 Sentence‑BERT 的二分类路由器。

**📊 数据集**

实验数据集为 PopQA 和 Head‑to‑Tail（长尾事实问答基准），知识库使用 Wikipedia 与 Wikidata，合成样本约 86k（PopQA）和 126k（Head‑to‑Tail）。

**📈 对比分析**

在长尾、非频繁、频繁三类查询上与 BM25、Contriever、BGE、Gemma、NV‑Embed 等检索器对比，RPDR 在 PopQA 长尾查询的 R@10 提升至 79.1%（比 BM25 提升 19.5%），整体在三类查询的 R@10 均达到 73‑79% 级别；与 GPT‑3.5 生成器结合的 RAG 系统，答案准确率较基线提升约 10.9%。

**⚠️ 局限性**

限制包括：循环预测和逆向嵌入训练需要额外计算成本；方法主要针对单事实长尾问答，未验证多跳或长文本生成任务；若数据集缺乏对应语料或不具备长尾特征，实验设置难以直接迁移。

---

## 314. A feature-stable and explainable machine learning framework for trustworthy decision-making under incomplete clinical data

**arXiv ID:** 2602.17364 | [PDF](https://arxiv.org/pdf/2602.17364v1)

**作者:** Justyna Andrys-Olek `[一作]` (Sano Centre for Computational Personalised Medicine), Jose Sousa `[通讯]` (Randox Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并评估了一个名为 CACTUS 的可解释机器学习框架，用以在小型、缺失数据多且异质的临床数据集中进行疾病分类和特征稳定性分析。

**💡 创新点**

创新点在于将特征抽象化、可解释分类与系统化特征稳定性评估整合到同一框架中，强调在数据缺失情境下保持特征重要性一致性，从而提升模型可信度。

**🔧 技术方法**

采用了基于 ROC 的特征离散化、改进的朴素贝叶斯分类器，并结合特征重要性排名；与随机森林、AdaBoost、XGB、LGBM、CatBoost 等梯度提升和集成方法进行对比。

**📊 数据集**

使用了 568 例含血尿症的患者临床数据（HaBio cohort），共 79 个生物标志物及相关特征，按性别划分为总组、男性组、女性组三类。

**📈 对比分析**

在引入 10%、20% 及 30% 随机缺失的情景下，CACTUS 在整体、男性、女性子集上实现了与传统方法相当或更高的平衡准确率和召回率，同时在特征稳定性指标（平均相对变化和重叠度）上显著优于随机森林和梯度提升模型。

**⚠️ 局限性**

主要局限包括：缺失机制假设为 MCAR，样本量仍有限，可能影响模型泛化；特征阈值是模型最佳分割点，未与临床诊断阈值对齐；未深入分析不同特征子集（生物标志物 vs 生活方式）对稳定性的具体贡献。

---

## 315. Partial Optimality in the Preordering Problem

**arXiv ID:** 2602.17346 | [PDF](https://arxiv.org/pdf/2602.17346v1)

**作者:** David Stein `[一作]`, Bjoern Andres `[通讯]` (Scalable Data Analytics and AI)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究预排序（preordering）问题，提出新的部分最优性条件和相应的改进映射与算法，能够在多项式时间内得到可扩展为最优预排序的部分解。

**💡 创新点**

创新点在于推导出新的割与连接（join）条件，改进了改进映射的定义，并提出闭包、最大特异性等概念，显著提升已知条件的覆盖率和效率。

**🔧 技术方法**

使用的技术包括改进映射（improving maps）、最大流/最小割求解（push‑relabel）、αβ‑swap 迭代、局部搜索（贪心弧固定/插入）以及三元组无交打包等。

**📊 数据集**

实验数据集包括：基于真实预排序的合成实例（通过 α、p_E 生成），以及真实社交网络（Twitter 和 Google+ 的 egonet）所构建的预排序实例。

**📈 对比分析**

与之前的方法（仅使用旧的割/连接条件）对比，新条件在变量固定率上更高（例如在合成实例中可固定全部变量或接近全变量），但单个条件的运行时间要高得多；整体算法仍保持多项式复杂度，在 Twitter 数据中约 30% 的变量被固定。

**⚠️ 局限性**

局限性：算法仍依赖于 O(|V|²) 的最大流/最小割求解，导致大规模实例的计算开销；对极难实例（如 α 接近 1 的情况）几乎无法固定任何变量；并且只给出可扩展为最优解的部分解，并不直接求解全局最优预排序。

---

## 316. From Subtle to Significant: Prompt-Driven Self-Improving Optimization in Test-Time Graph OOD Detection

**arXiv ID:** 2602.17342 | [PDF](https://arxiv.org/pdf/2602.17342v1)

**作者:** Luzhi Wang `[一作]` (Dalian Maritime University), Hongbo Liu `[通讯]` (Dalian Maritime University)

**通讯引用:** 5919 | [OpenAlex ID](https://openalex.org/A5100334703)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种无监督的自我改进图 OOD 检测框架 SIGOOD，利用测试时生成提示并通过能量差异迭代优化，实现对图分布偏移的检测。

**💡 创新点**

创新点在于设计能量偏好优化（EPO）损失和基于提示的自适应生成器，使模型仅凭测试图通过能量反馈循环迭代提升 OOD 信号。

**🔧 技术方法**

技术包括预训练 GNN、轻量级 MLP 提示生成器、能量基变分优化、Bradley‑Terry 比较模型与 KL 正则化。

**📊 数据集**

使用 UB‑GOLD 基准下的 21 个真实世界图数据集（化学分子、蛋白质等）以及 13 个异常检测数据集进行评测。

**📈 对比分析**

与 12 种基线（传统 OOD、GNN+后置、测试时训练方法）对比，SIGOOD 在 OOD 检测中平均提升约 10% AUC，在异常检测中排名第一，显著超越 GOODAT 等 SOTA。

**⚠️ 局限性**

局限在于能量信号在极端分布重叠时可能不足，对提示生成器超参和迭代次数敏感，且需进一步理论与跨域验证。

---

## 317. Do GPUs Really Need New Tabular File Formats?

**arXiv ID:** 2602.17335 | [PDF](https://arxiv.org/pdf/2602.17335v1)

**作者:** Jigao Luo `[一作]` (TU Darmstadt), Carsten Binnig `[通讯]` (TU Darmstadt)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对 GPU 加速查询，本文对 Parquet 文件的默认配置进行分析，并通过改写工具重新配置页数、行组大小、编码方式与压缩策略，以提升 GPU 扫描性能。

**💡 创新点**

创新点在于提出一套基于 GPU 计算特点的 Parquet 调优策略，并实现了通用的 Parquet 重写工具，能够一次性生成适配不同硬件的文件配置。

**🔧 技术方法**

使用 NVIDIA RAPIDS cuDF、GPUDirect Storage、PystachIO 等 GPU I/O 与解码框架，同时采用 Apache Arrow Rust 进行高效的多线程文件重写。

**📊 数据集**

实验基准为 TPC‑H SF300 规模数据集，利用本地 SSD 通过 GDS 读取。

**📈 对比分析**

通过与默认 DuckDB 写入的 Parquet 文件对比，观察到页数提升到 100、行组增至 10M、编码灵活与压缩优化后，GPU 读取带宽可提升至约 125 GB/s，整体查询时间显著下降。

**⚠️ 局限性**

主要限制包括重写过程仍需 CPU 端时间（但仅为一次性预处理），存储 I/O 仍是瓶颈，且未涉及 GPU 专用新编码，无法突破存储带宽极限。

---

## 318. Leveraging Contrastive Learning for a Similarity-Guided Tampered Document Data Generation Pipeline

**arXiv ID:** 2602.17322 | [PDF](https://arxiv.org/pdf/2602.17322v1)

**作者:** Mohamed Dhouib `[一作]` (École Polytechnique), Aymen Shabou `[通讯]` (Crédit Agricole)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过训练对比学习网络和裁剪质量网络，生成高质量的伪造文档图像；

**💡 创新点**

创新点在于利用对比学习判定视觉相似度与裁剪框质量，避免人工规则导致的可见痕迹；

**🔧 技术方法**

采用对比学习网络ℱθ、边框质量网络𝒢θ、文本渲染与填补算法；

**📊 数据集**

使用约2.8M来源自CC‑MAIN、IIT‑CDIP、DocMatrix等公开文档的合成数据；

**📈 对比分析**

在RTM、FindItAgain、FindIt等人造篡改基准上，零样本与微调实验均优于DocTamper等现有生成方法，显著提升图像级和像素级F1分数；

**⚠️ 局限性**

局限在于对OCR质量敏感，且仍需手工定义阈值与渲染参数，难以完全覆盖极端文字排版情况。

---

## 319. Evaluating Malleable Job Scheduling in HPC Clusters using Real-World Workloads

**arXiv ID:** 2602.17318 | [PDF](https://arxiv.org/pdf/2602.17318v1)

**作者:** Patrick Zojer `[一作]` (University of Kassel), Taylan Özden `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5069260397)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估了弹性作业调度在HPC集群中的效果，利用真实工作负载在不同弹性比例下进行模拟实验。

**💡 创新点**

提出了新的保持作业首选资源分配的弹性调度策略，并与传统和现有策略进行系统比较。

**🔧 技术方法**

使用ElastiSim仿真框架，基于速度提升模型和效率阈值将硬性作业转换为可弹性作业。

**📊 数据集**

采用Cori、Eagle、Theta超级计算机的真实工作负载轨迹数据。

**📈 对比分析**

通过比较完全硬性与不同弹性比例下的作业周转时间、完工时间、等待时间和节点利用率，弹性调度可实现37–67%周转时间下降、16–65%完工时间下降、73–99%等待时间下降和5–52%节点利用率提升。

**⚠️ 局限性**

局限性包括仅关注CPU节点，未考虑I/O和GPU异构负载，弹性转换基于启发式模型，缺乏长期真实负载和完整系统行为的建模。

---

## 320. MedClarify: An information-seeking AI agent for medical diagnosis with case-specific follow-up questions

**arXiv ID:** 2602.17308 | [PDF](https://arxiv.org/pdf/2602.17308v1)

**作者:** Hui Min Wong `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (Munich Center for Machine Learning)

**通讯引用:** 7468 | [OpenAlex ID](https://openalex.org/A5081442873)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出MedClarify，一个主动询问跟进问题的AI代理，通过信息获取和贝叶斯更新提升医学LLM的诊断准确率。

**💡 创新点**

创新性地定义诊断期望信息增益（DEIG），融合ICD语义相似性与信息增益，系统化挑选最能降低诊断不确定性的提问。

**🔧 技术方法**

利用大型语言模型（如Llama‑3.3‑70B、GPT‑5.1）、信息论（熵、信息增益）、贝叶斯推理以及ICD‑11编码映射，构建多轮问答与更新框架。

**📊 数据集**

使用 NEJM Image Challenge、MediQ 和 MedQA 三个公开医学诊断数据集，共计 469 条案例。

**📈 对比分析**

通过与单次输出、无优化的多轮提问以及仅使用标准信息增益的基线对比，MedClarify 在缺失信息场景下 top‑1 诊断准确率提升约 27pp（误差降低约 48%），并在多模型、多专业上保持稳健。

**⚠️ 局限性**

局限包括：使用模拟患者对话、仅文本输入（不含影像/实验室图像）、对LLM的幻觉风险未完全控制、实验设置对真实临床对话的可推广性有限。

---

## 321. Grothendieck Topologies and Sheaf-Theoretic Foundations of Cryptographic Security: Attacker Models and $Σ$-Protocols as the First Step

**arXiv ID:** 2602.17301 | [PDF](https://arxiv.org/pdf/2602.17301v1)

**作者:** Takao Inoué `[一作]` `[通讯]`, Takao Inoué

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种基于 Grothendieck 拓扑和层理论的结构化加密安全框架，将 Σ 协议的转录视为层，并用局部平凡化与无全局截面来解释零知识与完整性。

**💡 创新点**

创新点在于将攻击者观察模型转化为站点，利用几何层结构将模拟安全直接嵌入到拓扑语义中，提供了对传统模拟基安全定义的几何解释。

**🔧 技术方法**

使用 Grothendieck 拓扑、层理论、托索（topos）、扭体（torsor）理论以及对 Σ 协议的结构化分析。

**📊 数据集**

无具体数据集，研究属于理论分析。

**📈 对比分析**

该工作主要通过理论证明与传统模拟基安全定义进行比较，未进行实验性能评估；相比传统定义，其优势在于提供了统一的几何视角。

**⚠️ 局限性**

局限性包括：仅针对 Σ 协议；尚未扩展到恶意验证者、协议组合或更强安全模型；缺乏实验验证；抽象度高，实际实现细节待进一步研究。

---

## 322. A Contrastive Variational AutoEncoder for NSCLC Survival Prediction with Missing Modalities

**arXiv ID:** 2602.17402 | [PDF](https://arxiv.org/pdf/2602.17402v1)

**作者:** Michele Zanitti `[一作]` (National Institute of Standards and Technology), Sokol Kosta `[通讯]` (Colorado State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该文提供了IEEE期刊和会议论文的撰写与排版指导，说明标题、摘要、关键词、缩写、图表、公式、算法等要点。

**💡 创新点**

通过统一标准化的模板和详细写作规范，降低投稿过程中的排版错误，提高稿件质量和审稿效率。

**🔧 技术方法**

主要使用LaTeX模板、Overleaf、IEEE Author Center等工具，以及规范化的写作标准。

**📊 数据集**

未涉及实验数据，主要是示例文本与模板文件。

**📈 对比分析**

该文不进行实验比较，旨在提供写作规范，未涉及性能评估。

**⚠️ 局限性**

仅适用于IEEE期刊/会议格式，对其他期刊无适用性；内容为模板示例，没有实际研究内容。

---

## 323. A High-Level Survey of Optical Remote Sensing

**arXiv ID:** 2602.17397 | [PDF](https://arxiv.org/pdf/2602.17397v1)

**作者:** Panagiotis Koletsis `[一作]` (Harokopio University of Athens), Georgios Th. Papadopoulos `[通讯]` (Harokopio University of Athens)

**通讯引用:** 1364 | [OpenAlex ID](https://openalex.org/A5065554248)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了RGB光学遥感领域的主要任务、数据集、方法与趋势，提出了以传感器模态为中心的系统性分类框架。

**💡 创新点**

首次提供统一的RGB遥感任务与数据集全景，整合最新的基础模型与跨任务评估，揭示不同网络结构在各任务中的优势与局限。

**🔧 技术方法**

系统性评估CNN、Transformer、混合架构及基础模型（如SAM、CLIP、SMLFR、RingMo等）在分类、检测、分割、变更、视觉语言与编辑等任务中的应用。

**📊 数据集**

涵盖了如UCM、AID、NWPU-RESISC45、DOTA、DIOR、iSAID、WHU-CD、S2Looking、RS5M等近百个公开RGB遥感数据集。

**📈 对比分析**

对主流公开数据集的SOTA指标进行汇总，对比CNN与Transformer的表现，指出CNN在局部模式任务上更高效，Transformer在需要全局上下文的任务上更优，并展示了混合模型在多数任务中取得最佳成绩。

**⚠️ 局限性**

仍缺乏真正多任务、跨模态的基础模型，现有模型在多任务性能上落后于专门训练；数据集分布偏差与标签稀缺导致泛化能力受限；对小目标、变更检测细粒度和视频处理的研究不足。

---

## 324. Voice-Driven Semantic Perception for UAV-Assisted Emergency Networks

**arXiv ID:** 2602.17394 | [PDF](https://arxiv.org/pdf/2602.17394v1)

**作者:** Nuno Saavedra `[一作]` (INESC TEC and Faculty of Engineering University of Porto), Rui Campos `[通讯]` (INESC TEC and Faculty of Engineering University of Porto)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了SIREN框架，将无人机辅助应急网络中的语音通话转换为结构化机器可读信息，实现语音驱动的语义感知。

**💡 创新点**

首次将自动语音识别、LLM语义提取与NLP验证结合，形成模块化的语义感知层，支持人机协同决策与无人机网络自适应管理。

**🔧 技术方法**

使用Whisper/Assembly API做ASR，LLaMA 3.2做LLM信息提取，SpaCy做NER、说话人分离与情感分析，并通过Geopy+Folium完成地理编码与可视化。

**📊 数据集**

基于自制的Synthetic Emergency Audio Dataset（5个场景，包含英语/葡萄牙语、不同噪声、说话人数），以及RescueSpeech作为参考数据。

**📈 对比分析**

通过WER和手工评估指标（地点、单位、说话人数、QoS）对比不同ASR与噪声条件，结果显示Assembly API在噪声下表现优于本地Whisper，SIREN在清晰语音下位置/单位提取100%但QoS提取仅40%，高复杂度下说话人数和地理歧义导致失败，整体执行时间随语义复杂度升高。

**⚠️ 局限性**

主要局限是说话人分辨率失效、地理歧义导致坐标错误、噪声对Whisper影响大、以及实时性受LLM推理时延限制。

---

## 325. DRetHTR: Linear-Time Decoder-Only Retentive Network for Handwritten Text Recognition

**arXiv ID:** 2602.17387 | [PDF](https://arxiv.org/pdf/2602.17387v1)

**作者:** Changhun Kim `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Vincent Christlein `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 2910 | [OpenAlex ID](https://openalex.org/A5087093169)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于 Retentive Network 的解码器单独模型 DRetHTR，用于手写文本识别。

**💡 创新点**

创新点在于引入 Attention‑Retention Modality Fusion（ARMF）与层级 gamma 缩放，将 softmax‑free retention 结合图像–文本融合，实现线性时间、线性内存推理，并恢复 Transformer 的局部‑全局偏置。

**🔧 技术方法**

主要技术包括 RetNet 递归保留机制、ARMF、层级 gamma 缩放、EfficientNetV2 视觉嵌入、BPE 与字符级编码、Beam 搜索与 KV‑cache 对比等。

**📊 数据集**

在 IAM、RIMES、READ‑2016、Bentham 四大线性文本基准上进行评估，并在预训练阶段使用 1700 万 CC100 合成手写图像文本对。

**📈 对比分析**

与同尺寸的解码器单独 Transformer（DTrHTR）及 TrOCR 进行对比，DCretHTR 在 IAM 上 CER 2.26%/3.46%（Bentham）等达到最优/竞争水平，同时推理速度提升 1.6–1.9×、内存降低 38–42%。

**⚠️ 局限性**

局限性包括软max 作用范围与 retention 交互未做细粒度消融、仅在中等长度序列上验证，未充分探究 n ≫ d 的长序列效率与对齐漂移问题。

---

## 326. End-to-End Latency Measurement Methodology for Connected and Autonomous Vehicle Teleoperation

**arXiv ID:** 2602.17381 | [PDF](https://arxiv.org/pdf/2602.17381v1)

**作者:** François Provost `[一作]` (University of Luxembourg), Raphaël Frank `[通讯]` (University of Luxembourg)

**通讯引用:** 1886 | [OpenAlex ID](https://openalex.org/A5009044318)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一套基于双台 GPS 同步的 Raspberry Pi 5 + IMU + LED/phototransistor 的测量框架，用于定量评估远程操控 CAV 的 M2M、G2G 与 E2E 延迟。

**💡 创新点**

首次将硬件同步、运动阈值检测和光学事件结合，实现了同时测量 M2M、G2G 与 E2E 的精确方法，并通过 GPS PPS 取得微秒级时钟同步。

**🔧 技术方法**

采用 MPU‑6050 陀螺仪、TEPT‑4400 光电二极管、Chrony GPS PPS 同步、Linux 内核模块、低通滤波+阈值检测、Wireshark 采样、Cyclictest 等工具。

**📊 数据集**

在实际测试场景中收集了超过 100 条 4G 与 5G NSA 网络的延迟数据，并通过自建的车载与遥控站硬件产生的原始时序数据进行验证。

**📈 对比分析**

通过基线同步误差、M2M、G2G 与 E2E 的平均值、分位数进行比较；结果显示在 5G NSA 上平均 E2E 延迟约 498 ms，M2M 占比约 60%，比先前 Hall 传感器方案的 800 ms 明显下降。

**⚠️ 局限性**

受限于 I²C 采样率、线缆连接导致的运动检测误差、未能完整分解摄像头与 PID 机械响应等，测量误差仍达数 ms，且只能在静止车辆环境下验证。

---

## 327. The Role of the Availability Heuristic in Multiple-Choice Answering Behaviour

**arXiv ID:** 2602.17377 | [PDF](https://arxiv.org/pdf/2602.17377v1)

**作者:** Leonidas Zotos `[一作]` (University of Groningen), Malvina Nissim `[通讯]` (University of Groningen)

**通讯引用:** 2825 | [OpenAlex ID](https://openalex.org/A5040564747)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了可用性启发式在多项选择题回答中的作用，提出基于大语料库检索的可用性量化方法并评估其对答案选择的影响。

**💡 创新点**

将概念出现频率作为可用性代理，利用检索-分配方法量化答案与干扰项的可用性差异，并验证人类与LLM生成干扰项的相似性。

**🔧 技术方法**

使用Cohere Embed v3进行语义检索与余弦相似度分配，采用贝叶斯Dirichlet‑Multinomial比较方法，以及Qwen3系列LLM进行干扰项生成。

**📊 数据集**

Biopsychology、Immunopharmacology（专业学生测验题库）和SciQ（公开科学题库）以及English Wikipedia和BEIR两大检索语料。

**📈 对比分析**

通过检索固定数量段落后比较各选项段落比例，使用Friedman检验+Wilcoxon检验来评估可用性差异；在Wikipedia语料下，最优可用性策略可将准确率提升13.5–32.9%（4选题）或15–24%（3选题）以上。

**⚠️ 局限性**

仅关注事实知识型题目，推理题目的可用性差异可能不明显；使用简单LLM生成提示可能缺乏教学质量；BEIR语料缺乏专业内容导致可用性测量受限。

---

## 328. Prophet Inequality with Conservative Prediction

**arXiv ID:** 2602.17358 | [PDF](https://arxiv.org/pdf/2602.17358v1)

**作者:** Johannes Brüstle `[一作]` (Sapienza University of Rome), Stefano Leonardi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 4640 | [OpenAlex ID](https://openalex.org/A5020876546)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出了一种阈值策略，用于在拥有保守预测的先知不等式问题中，既保持经典1/2的鲁棒性，又在预测越准时性能可提升至3/4；并给出若预测质量α已知时的最优竞争比1/(2−α)。

**💡 创新点**

创新点在于：①在不知α的情况下设计α‑不敏感阈值算法，实现从1/2到3/4的光滑提升；②证明在已知α时1/(2−α)是最优；③给出不可超越的界限，说明在α未知时无法同时达到1/2与3/4。

**🔧 技术方法**

主要技术是：阈值算法与预测结合（effective threshold=max{τ,预测}); 通过构造SC阈值、对Bernoulli实例的归约以及凸性分析；利用概率界与最坏情况构造证明竞争比。

**📊 数据集**

无实验数据集，全部通过理论证明与最坏案例构造。

**📈 对比分析**

比较方法：与经典1/2阈值算法对比；当α=1时与已知最佳0.75结果对比；理论证明表明在各α下的竞争比符合给定区间；在α已知时达到1/(2−α)。

**⚠️ 局限性**

限制：仅适用于保守预测模型；若预测可能高估最大值，需额外缩放；在非连续或带点质量分布时需要额外处理；实际应用中需预估α值或使用α‑不敏感算法。

---

## 329. Astra: AI Safety, Trust, & Risk Assessment

**arXiv ID:** 2602.17357 | [PDF](https://arxiv.org/pdf/2602.17357v1)

**作者:** Pranav Aggarwal `[一作]` (Ashoka University), Aalok Thakkar `[通讯]` (Ashoka University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

针对印度数字公共产品（DPG）构建了一个基于实际案例的、印度特定的人工智能安全风险分类体系（ASTRA），并提供了风险数据库与本体结构。

**💡 创新点**

创新点在于：① 首次将印度多语言、多层级社会结构（种姓、宗教、性别、地区）融入AI风险范式；② 采用扎根理论与分层编码的方法，从真实案例出发构建分级子类；③ 提供可持续更新的风险本体与监管映射框架，支持监管者与开发者的协同治理。

**🔧 技术方法**

方法论主要包括：文献与案例梳理、风险分类与子类提炼、因果本体构造、分层次风险地图绘制；没有引入新的机器学习算法或模型，而是以定性分析和专家共识为核心。

**📊 数据集**

数据来源主要是公开文献、行业报告、公开案例（如 Aadhaar、UPI、FinTech 贷方模型、教育平台等）以及已有的国际风险仓库（MIT AI Risk Repository、AIR-Bench 2024 等），并结合印度本土研究（如 DECASTE 等）。

**📈 对比分析**

该工作不涉及传统意义上的性能对比，而是通过对比已有全球风险分类框架（EU AI Act、UNESCO、World Bank 等）的适用性，证明 ASTRA 在印度语境下更细粒度、更具操作性的优势；对比表格展示了在印度背景下的覆盖率、细化程度和监管适配度。

**⚠️ 局限性**

局限性包括：① 目前仅覆盖教育、金融等部分领域，尚未系统化农业、医疗、司法等新兴领域；② 依赖公开案例，可能遗漏隐性风险；③ 该分类体系为定性描述，缺乏量化风险评估与概率分布；④ 未来需要动态更新机制和专家反馈来维护本体的时效性与准确性。

---

## 330. The Sound of Death: Deep Learning Reveals Vascular Damage from Carotid Ultrasound

**arXiv ID:** 2602.17321 | [PDF](https://arxiv.org/pdf/2602.17321v1)

**作者:** Christoph Balada `[一作]` (German Research Center for Artificial Intelligence), Andreas Dengel `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

利用卷积变换器VideoMAE对德国Gutenberg健康研究的颈动脉B‑模式超声视频进行训练，学习到血管损伤（VD）特征，并将其作为无创心血管风险指标。

**💡 创新点**

创新点在于用嘈杂的高血压标签作为弱监督，挖掘超声视频中结构与功能信息，得到可解释且与传统SCORE2同等或更优的预后指标，并通过可解释AI发现周围脂肪组织对风险预测的贡献。

**🔧 技术方法**

技术包括视频MAE Transformer、遮蔽归因和反事实生成的可解释AI、Cox比例风险模型、Kaplan–Meier、SCORE2对照。

**📊 数据集**

数据集为Gutenberg健康研究的约1.47万名受试者，包含多时段颈动脉超声视频、临床参数及15年随访死亡/心血管事件。

**📈 对比分析**

方法通过10次随机划分的独立测试集评估，VideoMAE达到平衡准确率72.3%，AUC 0.773；VD与SCORE2比较，Cox模型C-index提升0.035，NRI 0.34，表明VD在预后上的优势。

**⚠️ 局限性**

局限包括缺乏外部独立验证、仅使用单一超声机导致平台偏差、以及高血压标签噪声导致学习的VD与实际血压不完全对应。

---

## 331. Flickering Multi-Armed Bandits

**arXiv ID:** 2602.17315 | [PDF](https://arxiv.org/pdf/2602.17315v1)

**作者:** Sourav Chakraborty `[一作]` (University of Colorado), Lijun Chen `[通讯]` (University of Colorado)

**通讯引用:** 7216 | [OpenAlex ID](https://openalex.org/A5100398616)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Flickering Multi‑Armed Bandit (FMAB) 框架，研究在每轮可用动作集合随时间变化且受前一次选择影响的情境下的决策问题。

**💡 创新点**

创新点在于：①将动作可用性建模为随时间演化的随机图（ER 与边 Markovian）；②提出仅依赖局部移动的两阶段算法（随机游走探索 + 导航+承诺），并给出近似最优的子线性遗憾上界；③证明该算法在信息论意义下的探索成本与下界匹配。

**🔧 技术方法**

采用随机图理论、马尔科夫链混合性分析、Hoeffding 与集中不等式、信息论下界、以及 lazy random walk 作为核心技术。

**📊 数据集**

主要使用仿真数据：灾难响应场景中 500 个候选位置构成的 5 平方公里地图，采用 ER 与边 Markovian 模型模拟道路通行情况，并设置稀疏奖励分布（热点、次热点、普通地点）。

**📈 对比分析**

通过理论证明与仿真对比，算法在两种图模型下的期望遗憾满足 O(n log(nT)/²)，与信息论下界 Ω(n log T /²) 相差仅对数因子；仿真显示在 80,640 轮任务中，累积遗憾随时间快速收敛，最终选择的热点即为最优点。

**⚠️ 局限性**

局限性：①算法仅为两阶段静态设置，缺乏在线自适应（如 UCB 版）；②假设奖励分布静态；③未考虑多智能体或非全局可观测的复杂情境；④导航成本在理论中被简化为 O(n log n) 级，实际环境中路径规划更为复杂。

---

## 332. Open Datasets in Learning Analytics: Trends, Challenges, and Best PRACTICE

**arXiv ID:** 2602.17314 | [PDF](https://arxiv.org/pdf/2602.17314v1)

**作者:** Valdemar Švábenský `[一作]` (Masaryk University), Atsushi Shimada `[通讯]` (Kyushu University)

**通讯引用:** 5684 | [OpenAlex ID](https://openalex.org/A5046057343)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统梳理并统计了 2020–2024 年 LAK、EDM、AIED 会议论文中公开的数据集，发现 143 个之前未被记录的数据集，并对其特征、使用频次、可获取方式等进行了深入分析。

**💡 创新点**

首次在学习分析领域提供了规模最大、最完整的数据集清单，提出了 PRACTICE 指南和 8 条实用准则，填补了该领域开放数据实践与评价缺口。

**🔧 技术方法**

采用手工文献筛选、PRISMA 流程、数据可用性检验和多维度特征编码，结合统计检验（Mann‑Kendall、Sen 斜率）和趋势分析方法。

**📊 数据集**

共计 571 篇论文中涉及 286 次数据集使用，涵盖 160 个独立数据集，主要来源于 K‑12 与大学环境，涵盖 STEM、语言学习等主题；其中 143 个数据集为新发现。

**📈 对比分析**

通过对比数据集出现频次、下载方式、使用的分析方法与评价指标，揭示了监督学习（尤其是深度学习）占主导，指标以准确率、AUC、F1 等为主；相较于以往工作，公开数据集的可访问性提高但仍受 10%+ 的 “请求后提供” 低成功率限制。

**⚠️ 局限性**

受限于样本覆盖不足（缺乏职业学习者、教师、社会科学等领域）、大多数数据来自美国、对数据可用性与隐私合规的依赖，以及对非公开数据的排除导致对整体 LA 研究格局的部分偏倚。

---

## 333. LexiSafe: Offline Safe Reinforcement Learning with Lexicographic Safety-Reward Hierarchy

**arXiv ID:** 2602.17312 | [PDF](https://arxiv.org/pdf/2602.17312v1)

**作者:** Hsin-Jung Yang `[一作]` (Iowa State University), Soumik Sarkar `[通讯]` (Iowa State University)

**通讯引用:** 10096 | [OpenAlex ID](https://openalex.org/A5081037761)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了LexiSafe框架，实现离线安全强化学习中的词典序层次优化，先保证安全约束再追求性能；

**💡 创新点**

创新点在于将安全和奖励的优先级明确为词典序，使用多阶段训练和结构化偏置来防止安全漂移，并给出单成本和多成本情形的样本复杂度与安全违规/性能欠优界限；

**🔧 技术方法**

技术手段包括IQL（隐式Q学习）结合优势加权回归、KL距离近似、Lagrange乘子自适应、VC维度通用性分析以及多阶段安全约束的逐层实现；

**📊 数据集**

使用DSRL基准数据集，涵盖MetaDrive、Bullet Safety Gym和Safety Gymnasium等离线轨迹；

**📈 对比分析**

与BC‑Safe、COptiDICE、CPQ、FISOR、LSPC‑O等基线相比，LexiSafe在保持安全（成本≤阈值）同时获得更高或相近的归一化奖励，表现优于或竞争性强；

**⚠️ 局限性**

局限性包括对数据覆盖度的假设、浓度系数难以估计、VC维度界限可能过于保守、以及对离线数据质量和安全阈值设置的依赖。

---

## 334. Attachment Anchors: A Novel Framework for Laparoscopic Grasping Point Prediction in Colorectal Surgery

**arXiv ID:** 2602.17310 | [PDF](https://arxiv.org/pdf/2602.17310v1)

**作者:** Dennis N. Schneider `[一作]` (Technical University of Munich), Dirk Wilhelm `[通讯]` (Technical University of Munich)

**通讯引用:** 3594 | [OpenAlex ID](https://openalex.org/A5012902178)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种附件锚点（Attachment Anchors）框架，用来在结肠手术的结肠动员阶段预测合适的抓取点。

**💡 创新点**

创新点在于将解剖结构与机械约束压缩为二维极坐标的局部锚点表示，既简化了抓取点预测问题，又提供了可解释的中间表示；同时利用锚点实现了针对手术场景的结构化数据增强。

**🔧 技术方法**

技术上使用了基于YOLOv8的附件锚点编码器、对比学习提升特征判别性、极坐标回归的Rad‑YOLOv8抓取点解码器，以及锚点变换的数据增强方法。

**📊 数据集**

采用了自制的90例结肠手术数据集，涵盖5个解剖区域、15名外科医生，全部采集于TUM医院。

**📈 对比分析**

通过与仅基于图像的KP‑YOLOv8基线比较，附件锚点模型在Precision@6%上提升约12%，在未见手术类型、未见医生以及采用锚点变换增强的情形下均显著优于基线，表现稳健。

**⚠️ 局限性**

局限性包括：数据集中手术师与手术类型分布不均、存在中心抓取偏差；模型仅适用于目标可见的情形，无法处理遮挡或需时间推断的目标；锚点在复杂大范围结构上的表达仍有限；未对血腥、雾化等高难度视觉干扰进行评估。

---

## 335. On the complexity of covering points by disjoint segments and by guillotine cuts

**arXiv ID:** 2602.17294 | [PDF](https://arxiv.org/pdf/2602.17294v1)

**作者:** Delia Garijo `[一作]` (Universidad de Sevilla), Rodrigo I. Silveira `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 959 | [OpenAlex ID](https://openalex.org/A5074443763)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文证明了在平面中用不相交线段或刀切来覆盖给定点集的判定问题是NP-完全的。

**💡 创新点**

创新点在于将经典的点线覆盖NP-完全性证明迁移至不相交线段和刀切版本，并利用平面单调3-SAT与几何变换构造了新的多项式时间约简。

**🔧 技术方法**

技术上采用平面单调3-SAT、几何变换、变量/子句点构造以及剪刀序列等方法，构造多项式可行的点集S并证明其覆盖问题等价于3-SAT。

**📊 数据集**

未使用实验数据集，本文仅在理论构造中生成点集S来进行约简。

**📈 对比分析**

本文未给出具体算法实现或性能评估，仅通过多项式时间约简证明复杂度，没有性能比较。

**⚠️ 局限性**

局限在于只提供了NP-完全性证明，未提供近似或多项式时间算法，也未验证在实际实例上的可行性。

---

## 336. DAVE: A Policy-Enforcing LLM Spokesperson for Secure Multi-Document Data Sharing

**arXiv ID:** 2602.17413 | [PDF](https://arxiv.org/pdf/2602.17413v1)

**作者:** René Brinkhege `[一作]` (Fraunhofer Institute for Software and Systems Engineering), Prahlad Menon `[通讯]` (Fraunhofer Center for Manufacturing Automation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了名为 DAVE 的 LLM 发言人，在工业数据空间中根据协商的使用政策回答关于私有文档的自然语言问题，避免直接共享文件，实现虚拟红色。

**💡 创新点**

通过多层策略执行管线（目的检测、政策感知检索、条件提示、后期检查）将合同级使用政策转化为语义级信息披露控制，实现数据空间内的安全问答，并提出虚拟红色作为动态红色替代方案。

**🔧 技术方法**

结合 Eclipse Dataspace Components、ODRL 使用政策语言、检索增强生成（RAG）技术、LLM（如 GPT 系列）与多模态文本抽取、向量检索、自然语言提示与后期泄漏检测等技术。

**📊 数据集**

主要使用公开技术安全文档（事故报告、操作手册）以及合成注入的敏感片段进行实验；未来计划使用行业合作伙伴的安全领域文档但仅限内部评估。

**📈 对比分析**

计划与标准 RAG 及仅提示约束的基线对比，通过安全合规率、答案质量（Exact Match、F1、主观打分）和覆盖率评估；初步设计预期多层管线在保持约 80–90% 覆盖率且答案质量略低的情况下，显著降低泄漏率；性能开销主要在检索过滤与后处理，预计平均延迟提升 10–20%。

**⚠️ 局限性**

目前仅完成架构和与 EDC 的集成，完整的政策感知检索与后期泄漏检查尚未实现，缺乏真实环境下的实验与量化评估；依赖于正确编写的使用政策，未解决策略误写与政策解释不一致的问题。

---

## 337. Bluetooth Phased-array Aided Inertial Navigation Using Factor Graphs: Experimental Verification

**arXiv ID:** 2602.17407 | [PDF](https://arxiv.org/pdf/2602.17407v1)

**作者:** Glen Hjelmerud Mørkbak Sørensen `[一作]` (Norwegian University of Science and Technology), Tor Arne Johansen `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 22034 | [OpenAlex ID](https://openalex.org/A5012692888)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在GNSS失效场景下，利用蓝牙低功耗（BLE）方向角测量与RTK范围或气压测量，构建基于因子图优化（FGO）的惯性导航系统，并在多旋翼无人机实验数据上验证其可行性。

**💡 创新点**

创新点在于：①将BLE PARS方向角与惯性测量结合使用，提供低成本的短距离定位方案；②将FGO与多种M估计器（Tukey、Geman‑McClure）联合应用，实现对噪声和异常值的鲁棒处理；③在RTK切换至BLE模式的“交接”实验中展示了FGO相较于传统ESKF的更高估计精度与更优的异常值抑制效果。

**🔧 技术方法**

使用技术包括：因子图优化（iSAM2）、SE(3) Lie群表征、BLE方向角（AoA）测量、RTK/气压辅助因子、IMU预积分因子、M估计器（Tukey、Geman‑McClure）以及GTSAM库。

**📊 数据集**

使用数据集：NTNU测试场地的多旋翼无人机飞行实验数据，包含2000 Hz IMU、1 Hz RTK、2 Hz气压计、16.6 Hz BLE方向角，以及通过RTK插值生成的BLE范围值。

**📈 对比分析**

方法比较：将FGO（配合Tukey或Geman‑McClure）与基准ESKF（配合自然检验）在RTK→BLE的交接场景下进行RMSE对比。结果表明：在位置和姿态估计上，FGO在大多数分段取得更低的RMSE，尤其在Down分量上保持稳定；Tukey在切换初期产生较大不确定性，而Geman‑McClure更平滑；ESKF在快速偏航变化时更易出现尖峰误差。

**⚠️ 局限性**

局限性包括：BLE范围测量仅通过RTK插值得到，未使用真正的BLE通道声波测距；短距离操作导致角测量噪声高且易受遮挡；快速偏航变化难以用现有传感器补偿；缺乏多天线校准与时钟同步，导致姿态偏差；以及对高动态场景的鲁棒性仍需进一步验证。

---

## 338. Proximal powered knee placement: a case study

**arXiv ID:** 2602.17502 | [PDF](https://arxiv.org/pdf/2602.17502v1)

**作者:** Kyle R. Embry `[一作]` (Shirley Ryan AbilityLab), Levi J. Hargrove `[通讯]` (Northwestern University)

**通讯引用:** 12587 | [OpenAlex ID](https://openalex.org/A5018478365)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估了膝关节上方放置动力传动的可行性，并与传统膝下放置进行对比

**💡 创新点**

首次在转股截肢者中验证膝关节上方动力传动设计，显示可行且对步速、步频有提升，且不需大幅改动控制策略

**🔧 技术方法**

采用三层层次阻尼控制结构、负载单元、关节编码器、IMU、以及低高架硬件设计

**📊 数据集**

使用三名单侧转股截肢者（TF01、TF02、TF03）的步态数据，包含 GAITRite 压敏台测量、传感器记录、坡道、台阶等多种活动

**📈 对比分析**

通过比较步速、步频、步态对称性等指标，发现上方放置在 TF01 提升了 9.2% 步速、3.6% 步频；TF02 提升幅度较小；步态对称性仅在 TF01 见改善，整体保持关节运动一致且速度提升

**⚠️ 局限性**

样本量小，仅三人；残肢长度限制了适用人群；负载单元位置不同导致力学测量难以直接比较；缺乏代谢成本、长期适应和临床评估等进一步验证

---

## 339. Linear Convergence in Games with Delayed Feedback via Extra Prediction

**arXiv ID:** 2602.17486 | [PDF](https://arxiv.org/pdf/2602.17486v1)

**作者:** Yuma Fujimoto `[一作]` (CyberAgent), Kaito Ariu `[通讯]` (CyberAgent)

**通讯引用:** 94 | [OpenAlex ID](https://openalex.org/A5081429958)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在不受约束的双线性博弈中，研究并证明了加权乐观梯度下降上升（WOGDA）算法在存在反馈延迟时的线性收敛性，并通过额外预测提升收敛速度。

**💡 创新点**

创新点在于：①将WOGDA视为近似的额外近端点（EPP）方法；②揭示额外预测（预测未来多步奖励）可以显著加速收敛并容忍更大步长；③给出理论收敛率与实验一致的阶数。

**🔧 技术方法**

主要技术包括：延迟反馈建模、WOGDA与EPP的解析近似、谱分析与误差界、步长选择与误差累积分析。

**📊 数据集**

实验数据集：匹配硬币（Matching Pennies）和随机生成的 5×5 高斯矩阵博弈。

**📈 对比分析**

与传统 GDA、OGDA 以及无延迟情形比较；通过测量最优步长与估计的线性收敛率，实验显示额外预测显著提升步长上限并使收敛速率从 O(1/m³) 提升至 O(1/m)，与理论预测相符。

**⚠️ 局限性**

局限性：理论与实验间存在定量差距；假设延迟为已知固定整数，未充分处理随机/未知延迟；仅在双线性无约束博弈中证明，尚未推广到更一般的凸凹或约束情形。

---

## 340. QuPAINT: Physics-Aware Instruction Tuning Approach to Quantum Material Discovery

**arXiv ID:** 2602.17478 | [PDF](https://arxiv.org/pdf/2602.17478v1)

**作者:** Xuan-Bac Nguyen `[一作]` (University of Arkansas), Khoa Luu `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个完整的量子材料表征体系，结合物理驱动的合成数据生成（Synthia）、大规模物理指令数据集（QMat‑Instruct）、多模态指令微调框架（Physics‑Aware Instruction Tuning）以及统一评测基准（QF‑Bench）。

**💡 创新点**

创新点包括：①利用传递矩阵法生成真实光学反射合成图像；②构建首个面向量子材料的物理指令问答数据集；③在多模态模型中引入Physics‑Informed Attention模块，实现光学先验与视觉特征的融合；④发布涵盖多材料、多底物、多成像设置的标准基准。

**🔧 技术方法**

技术手段包括：传递矩阵光学模拟、CIE LAB颜色空间对比度近似、Vision Transformer与Physics‑Informed Attention结合的多模态编码器、指令微调的MLLM解码器、数据增强与色彩空间调整、端到端的物理先验校正。

**📊 数据集**

使用的数据集有：合成光学响应图像（Synthia）、真实显微镜图像（多材质、不同底物）、物理指令-答案对（QMat‑Instruct）、统一评测集（QF‑Bench）以及公开的显微图像集合。

**📈 对比分析**

与MaskRCNN、ViTDet、YOLOv11等传统检测器以及InternVL、Qwen3等多模态模型进行对比，普通薄片检测AP从约30%提升至45%+，单层薄片AP从约20%提升至52%+；在指令定位任务中Acc@75从约32%提升至39.5%。

**⚠️ 局限性**

局限性在于：①合成数据虽逼真，但仍无法覆盖所有显微镜条件和缺陷类型，导致域差距；②多模态模型的视觉令牌长度限制，难以处理极稠密场景；③对极低对比度或极薄层的精细分辨仍有限。

---

## 341. A Cost-Effective and Climate-Resilient Air Pressure System for Rain Effect Reduction on Automated Vehicle Cameras

**arXiv ID:** 2602.17472 | [PDF](https://arxiv.org/pdf/2602.17472v1)

**作者:** Mohamed Sabry `[一作]` (Johannes Kepler University Linz), Cristina Olaverri-Monreal `[通讯]` (Johannes Kepler University Linz)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并部署了一种低成本（<100欧元）空气压系统（APS），通过气幕分散雨滴，显著提升自动驾驶车辆摄像头在雨天的行人检测性能。

**💡 创新点**

创新点在于模块化、可扩展的气压管路设计，能同时为多摄像头提供保护，并通过经济实用的气幕实现雨滴消除，满足大规模汽车应用需求。

**🔧 技术方法**

主要技术包括机械气压驱动器、气幕管路、定制喷嘴以及YOLOv4‑tiny深度学习检测模型。

**📊 数据集**

使用自采集的实时摄像数据（Basler摄像头拍摄的雨天与非雨天视频），未使用公开数据集。

**📈 对比分析**

采用帧连续检测（连续三帧均检测到行人）与10秒窗口检测率两种评估方法；雨天无APS时检测率为8.3%，加入APS后提升至41.6%，显示显著性能提升。

**⚠️ 局限性**

局限性包括仅在雨天环境下测试，未验证雪、雾等其他恶劣天气；系统需手动组装和维护；目前仅针对摄像头，对LiDAR等传感器的适用性尚未评估。

---

## 342. LORA-CRAFT: Cross-layer Rank Adaptation via Frozen Tucker Decomposition of Pre-trained Attention Weights

**arXiv ID:** 2602.17510 | [PDF](https://arxiv.org/pdf/2602.17510v1)

**作者:** Kasun Dewage `[一作]` (University of Central Florida), Shankadeep Mondal `[通讯]` (University of Central Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍了一种名为CRAFT的参数高效微调方法，该方法对Transformer注意力投影的跨层权重矩阵进行三维Tucker分解，冻结所有因子，只训练三组小方阵实现微调。

**💡 创新点**

创新点在于将预训练权重的跨层三维张量完整进行HOSVD分解并冻结所有因子，同时通过残差保留原始权重，仅通过小型可训练方阵进行微调，使得参数量与模型尺寸和层数无关。

**🔧 技术方法**

技术包括高阶奇异值分解（HOSVD）、Tucker-3分解、残差保持的适配公式以及小尺寸可训练方阵（J⁽¹⁾、J⁽²⁾、J⁽³⁾）。

**📊 数据集**

使用GLUE基准数据集，在RoBERTa-base（125M参数）和RoBERTa-large（355M参数）上进行评估。

**📈 对比分析**

与LoRA、PiSSA、AdptP等方法比较，CRAFT仅使用约41K个适配参数即可在RoBERTa-large上获得平均88.0分，接近LoRA的88.9分；在RoBERTa-base上平均84.5分，略低于LoRA的87.2分，但参数量仅为LoRA的1/7。

**⚠️ 局限性**

局限性包括在较小模型上因适配空间受限导致性能略低；需要进一步验证在更大LLM和生成任务上的效果；一次性HOSVD预处理成本较高；实验仅在单一随机种子下完成，缺乏方差分析。

---

## 343. Towards a Software Reference Architecture for Natural Language Processing Tools in Requirements Engineering

**arXiv ID:** 2602.17498 | [PDF](https://arxiv.org/pdf/2602.17498v1)

**作者:** Julian Frattini `[一作]` (Chalmers University of Technology and University of Gothenburg), Quim Motger `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 145 | [OpenAlex ID](https://openalex.org/A5060562561)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了将NLP4RE工具从单体实现转向模块化、可互操作生态系统的愿景，并提出了构建软件参考架构（SRA）的研究路线图，先通过利基焦点小组获取系统需求。

**💡 创新点**

创新点在于首次系统性地提出NLP4RE工具的SRA框架，强调模块化重用与可互操作性，并将焦点小组需求作为SRA的需求基础。

**🔧 技术方法**

主要采用了架构设计方法（Nakagawa等的ProSA-RA四步法）、需求挖掘与聚类分析、以及文本分析工具进行需求整理。

**📊 数据集**

使用了焦点小组收集的访谈数据（20名AIRE'25工作坊参与者的讨论结果）以及公开的文献、工具、模型、词典等信息源。

**📈 对比分析**

本文未进行传统意义上的性能对比，而是通过需求映射和可行性评估等方式验证SRA的可行性；未来计划对现有工具进行重构并在基准上进行比较。

**⚠️ 局限性**

局限性包括缺乏经验丰富的实证验证、数据集仅来自小规模焦点小组、以及对SRA完整性与适用性的评估仍在进行中。

---

## 344. Learning with Boolean threshold functions

**arXiv ID:** 2602.17493 | [PDF](https://arxiv.org/pdf/2602.17493v1)

**作者:** Veit Elser `[一作]` (Cornell), Manish Krishan Lal `[通讯]` (Technische Universitat Munchen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种基于约束满足的布尔阈值神经网络训练方法，使用投影法（RRR）替代梯度下降，能够学习仅包含 ±1 权重的离散模型。

**💡 创新点**

将布尔阈值函数作为非凸约束，引入 Margin 约束实现稀疏性和可解释性，并使用 divide-and-concur 与 RRR 投影求解，展示在离散数据上能得到精确或高度泛化的解。

**🔧 技术方法**

使用约束投影（RRR）、BTF 约束与并行投影、Margin 与权重归一化、自动化度量调节以及 divide-and-concur 分解等技术。

**📊 数据集**

在二进制乘法表、二进制自动编码、MNIST 4、1D Rule‑30 细胞自动机、随机逻辑电路、随机输入向量等数据集上进行实验。

**📈 对比分析**

与传统梯度下降（反向传播）对比，在多数任务中 RRR 能达到零 gap、完美或高泛化；在逻辑电路、自动编码和规则‑30 上表现出显著优于梯度法的收敛速度和精度。

**⚠️ 局限性**

仍面临计算量大、难以处理大规模数据/网络、不可满足问题只能得到近似解、需要手工调参、缺乏严格收敛理论、目前仅单核实现等限制。

---

## 345. A variational multi-phase model for elastoplastic materials with microstructure evolution

**arXiv ID:** 2602.17492 | [PDF](https://arxiv.org/pdf/2602.17492v1)

**作者:** Sarah Dinkelacker-Steinhoff `[一作]` (Ruhr University Bochum), Klaus Hackl `[通讯]` (Ruhr University Bochum)

**通讯引用:** 4427 | [OpenAlex ID](https://openalex.org/A5088591370)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于变分原理的多相弹塑性材料模型，能够描述相变过程中微结构的连续演化；

**💡 创新点**

创新点在于将耗散距离、一般化Young测度与变分框架结合，构造了可即时相变且连续微观演化的能量与耗散函数，并通过松弛能量实现多相混合的有效宏观描述；

**🔧 技术方法**

使用了变分原理、拉格朗日能量最小化、Young测度松弛、耗散距离概念以及有限元（FEM）数值实现；

**📊 数据集**

通过二维方形板加圆孔的压缩加载实验（仅使用几何尺寸和材料参数，无公开数据集）对模型进行了数值验证；

**📈 对比分析**

采用不同网格粗细（粗、细、超细）进行对比，结果显示应力-应变曲线及相变分布在网格细化时收敛；模型捕捉到相变的即时发生、抖动收敛（shakedown）以及在弹性区的停滞；

**⚠️ 局限性**

局限性包括仅考虑等温条件、相变速率依赖性被简化为常数耗散距离、未加入温度耦合或更复杂的相互作用；未来需要扩展转移矩阵、引入指数型速率或多尺度机制来提高物理真实性。

---

## 346. Tracing Copied Pixels and Regularizing Patch Affinity in Copy Detection

**arXiv ID:** 2602.17484 | [PDF](https://arxiv.org/pdf/2602.17484v1)

**作者:** Yichen Lu `[一作]` (Ant Group), Peng Zhang `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PixTrace和CopyNCE以提升图像拷贝检测模型在复杂编辑下的鲁棒性。

**💡 创新点**

创新点在于基于像素级追踪实现精确的编辑对应关系，并将该对应关系作为几何引导的对比损失来强化局部相似性学习。

**🔧 技术方法**

采用自监督学习、Vision Transformer编码器、像素坐标表追踪、几何引导对比损失CopyNCE、ROI细粒度特征提取和交叉注意力融合。

**📊 数据集**

主要使用DISC21与其扩展NDEC数据集，包含约200万无标签图像和100k查询/参考图像。

**📈 对比分析**

与现有SOTA方法相比，在DISC21 matcher上达到88.7% μAP / 83.9% RP90，descriptor上达到72.6% μAP / 68.4% RP90，明显优于多项对比基线，且在更难的NDEC上也实现显著提升。

**⚠️ 局限性**

局限在于依赖精确的像素追踪表，处理极其复杂或无标记的编辑操作仍有挑战；同时在计算开销上仍高于纯特征匹配方法。

---

## 347. ShadAR: LLM-driven shader generation to transform visual perception in Augmented Reality

**arXiv ID:** 2602.17481 | [PDF](https://arxiv.org/pdf/2602.17481v1)

**作者:** Yanni Mei `[一作]` (TU Darmstadt), Jan Gugenheimer `[通讯]` (TU Darmstadt)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了ShadAR系统，利用大语言模型实时从用户语音命令生成HLSL着色器并在Meta Quest 3上即时应用，以改变AR视觉感知。

**💡 创新点**

创新点在于将LLM用于实时着色器代码生成而非仅对象创建，并通过自然语言描述实现多样化视觉体验。

**🔧 技术方法**

技术包括Whisper语音识别、OpenAI o3‑mini LLM、Prompt工程、Unity无界编译、Meta Passthrough Camera API、WebCamTexture和资产包部署。

**📊 数据集**

未公开使用特定数据集；主要利用公开示例着色器模板和LLM预训练模型。

**📈 对比分析**

实验显示大约30‑45秒内生成并部署着色器，能在单眼渲染中实时变化视觉；与手工编写着色器相比生成速度快、效果多样，但视觉精度不一定达到专业水平。

**⚠️ 局限性**

局限包括生成效果偶尔不符合真实视觉需求、只能在单眼渲染实现、对LLM生成的错误代码易导致崩溃、缺乏对多样视觉障碍精确建模。

---

## 348. Small LLMs for Medical NLP: a Systematic Analysis of Few-Shot, Constraint Decoding, Fine-Tuning and Continual Pre-Training in Italian

**arXiv ID:** 2602.17475 | [PDF](https://arxiv.org/pdf/2602.17475v1)

**作者:** Pietro Ferrazzi `[一作]`, Bernardo Magnini `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对意大利医疗NLP任务中的小型语言模型（≈1B参数）进行系统评估，并通过多种适配策略提升其性能。

**💡 创新点**

系统性比较了零样本、4-shot、约束解码、监督微调与持续预训练在意大利医疗文本上的效果，发布了第一套意大利医疗NLP数据集、30亿词医学语料库，并实现了单一小模型超过大型模型的结果。

**🔧 技术方法**

使用指令式提示、约束解码（Outlines）、少量样例提示、LoRA微调以及在医学语料上的持续预训练（CPT）等技术。

**📊 数据集**

使用12个意大利医疗NLP数据集（NER、CRF、RE、QA、Arg Mining），构建的300M词医学语料（科学+临床）以及用于OOD评估的六个未见过的数据集。

**📈 对比分析**

与Qwen3-32B和Medgemma-27B等大型模型进行对比；通过平均F1/准确率度量；微调后小模型可提升至比基准高+9.2分，甚至在某些任务上超过大型模型；OOD时仍低于大型模型，但相对提升显著。

**⚠️ 局限性**

研究仅覆盖意大利语，未考察多语言迁移；OOD评估保守，可能低估鲁棒性；仅探讨了少数适配方法；数据集不均衡（QA占比高，RE占比低）可能影响整体结论。

---

## 349. Auditing Reciprocal Sentiment Alignment: Inversion Risk, Dialect Representation and Intent Misalignment in Transformers

**arXiv ID:** 2602.17469 | [PDF](https://arxiv.org/pdf/2602.17469v1)

**作者:** Nusrat Jahan Lia `[一作]` (University of Dhaka), Shubhashis Roy Dipta `[通讯]` (University of Maryland Baltimore County)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对跨语言情感对齐问题，本文使用标准化的跨语言情感对齐框架，评估了四种Transformer模型在孟加拉语与英语平行文本上的情感一致性，并引入情感反转率、方言差距和方向性偏差等指标。

**💡 创新点**

创新点在于：①提出“情感反转风险”这一安全指标，量化模型在语言转换时情感极性翻转的严重性；②揭示“现代偏差”，即模型对正式（Sadhu）孟加拉语的误判；③引入“情感稳定性”与“方向性偏差”衡量，强调文化多样性与情感强度的双向一致性。

**🔧 技术方法**

技术手段包括Transformer模型（Tabularis、XLM‑R、IndicBERT、mDistilBERT）的并行推理、统一分数归一化、句级与整体级指标计算、方差与稳健性分析以及分布可视化。

**📊 数据集**

使用公开的BanglaBlend平行语料库，共7350句子对，按正式与口语两种孟加拉语方言划分（各3675句），覆盖正式（Sadhu）与口语（Cholito）两种写作体裁。

**📈 对比分析**

通过对比四模型在均值偏差、标准差、稳健率、情感反转率和方向性偏差等指标，结果显示：大规模XLM‑R表现最稳定，情感反转率最低；mDistilBERT压缩后导致反转率最高（28.7%），稳健率最低；IndicBERT在正式方言上的误差显著上升（+57%）。

**⚠️ 局限性**

局限性包括：①实验仅基于单一平行语料，无法覆盖更丰富的口语与非正式语料；②未考虑多任务或多模态情境；③模型的推理框架与指标依赖手工阈值，可能在不同语言或文化中失效；④缺乏用户研究验证情感对齐的实际影响。

---

## 350. Privacy in Theory, Bugs in Practice: Grey-Box Auditing of Differential Privacy Libraries

**arXiv ID:** 2602.17454 | [PDF](https://arxiv.org/pdf/2602.17454v1)

**作者:** Tudor Cebere `[一作]` (Universite de Montpellier), Jack Fitzsimons `[通讯]` (Oblivious)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种“灰盒”审计框架，能够在不完整黑盒分析的前提下，检查差分隐私（DP）实现中的实现错误，定位敏感度失配、噪声失衡以及数据相关的控制流泄露等问题；

**💡 创新点**

创新点在于利用记录‑重放技术对内部状态进行可验证的记录，并在重放阶段冻结隐私机制输出，从而将全局高维输出分布审计转化为低维内部输入比较与结构不变性检测；

**🔧 技术方法**

核心技术包括：①记录‑重放（Record‑Replay）Instrumentation、状态快照与随机数生成器控制；②数据独立性检查节点（Equality / Invariant 节点）；③组件级分布式审计（Statistical Audit）结合隐私损失分布（PLD）和数值隐私会计；④ Python 轻量级测试插件；

**📊 数据集**

实验使用人工生成的合成表格数据，按“加/删”和“替换”两种邻接模型随机构造邻接数据集（D、D′），并对多个公开 DP 库（SmartNoise SDK、Opacus、Diffprivlib 等）进行评测；

**📈 对比分析**

与现有全局黑盒分布式审计相比，灰盒方法在检测敏感度失配、数据泄漏等细粒度错误时更快且能给出具体错误位置；在性能上，记录‑重放消耗的时间和内存低于传统黑盒统计攻击，适合 CI/CD 环境；

**⚠️ 局限性**

局限性包括：① 需要可控伪随机数生成器，无法处理非确定性或并发/分布式系统；② 只对已知的邻接数据对进行检测，缺乏完整覆盖；③ 需要开发者手动标注 invariant 节点，自动化程度有限；

---

## 351. Jolt Atlas: Verifiable Inference via Lookup Arguments in Zero Knowledge

**arXiv ID:** 2602.17452 | [PDF](https://arxiv.org/pdf/2602.17452v1)

**作者:** Wyatt Benno `[一作]` (ICME Labs), Khalil Gibran `[通讯]` (ICME Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

开发了Jolt Atlas，一套零知识机器学习框架，能够对ONNX格式的神经网络推理进行可验证的推理，支持在资源受限设备上流式证明；

**💡 创新点**

创新点在于将Jolt的基于查表的Sumcheck证明范式迁移至张量运算；通过前缀-后缀拆分、神经跃迁压缩查表、虚拟多项式以及BlindFold零知识折叠，实现了对非线性激活的高效查表验证，并实现了显著的流式证明与内存压缩；

**🔧 技术方法**

使用了Sumcheck协议、查表（lookup）论证、前缀-后缀分解、神经跃迁（teleportation）、HyperKZG多项式承诺、Nova式折叠（BlindFold）、虚拟多项式、ONNX图解析和字节码生成等技术；

**📊 数据集**

主要使用了nanoGPT（约25万参数）和GPT-2（1.25亿参数）等典型小型至中型Transformer模型进行基准测试；

**📈 对比分析**

与现有zkML框架ezkl比较，nanoGPT推理证明时间从ezkl的约237秒压缩到Jolt Atlas的14秒（≈17×加速），关键/验证时间均保持在秒级；GPT-2的端到端证明耗时约38秒；

**⚠️ 局限性**

局限性包括：仍使用基于对偶的HyperKZG，限制了链上验证和对后量子安全的支持；对大型模型的查表尺寸仍较大，需进一步压缩；仅支持部分ONNX算子；固定的跃迁因子τ限制了查表压缩灵活性；在极端资源受限设备上仍需优化内存与时钟。

---

## 352. AIDG: Evaluating Asymmetry Between Information Extraction and Containment in Multi-Turn Dialogue

**arXiv ID:** 2602.17443 | [PDF](https://arxiv.org/pdf/2602.17443v1)

**作者:** Adib Sakhawat `[一作]` (Islamic University of Technology), Rakin Shahriar `[通讯]` (Islamic University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AIDG（Adversarial Information Deduction Game）框架，用来评估大型语言模型在多轮对话中信息提取与信息保持的能力差异

**💡 创新点**

首次系统量化LLM在主动推理与被动维护信息时的能力不对称，并揭示确认攻击与约束遵守对模型性能的影响

**🔧 技术方法**

使用基于博弈论的双ELO评分体系、低温LLM裁判自动判定、对话生成策略与多轮问答技巧

**📊 数据集**

构建20条原子事实隐私集和100词实体词典，生成439场对话，涵盖AIDG‑I（自由对话）与AIDG‑II（受限“是/否/可能”）两种模式

**📈 对比分析**

在六大前沿LLM上进行双循环对战，对比Holder与Seeker角色的ELO差距，发现Holder平均优势约255 ELO，确认攻击成功率提升7.75倍，约束违规率高达41%

**⚠️ 局限性**

局限在于仅考察静态隐私和受限词汇的情景，未覆盖更复杂情境与跨模态信息，且LLM裁判的自动化判定可能引入偏差

---

## 353. Fine-Grained Uncertainty Quantification for Long-Form Language Model Outputs: A Comparative Study

**arXiv ID:** 2602.17431 | [PDF](https://arxiv.org/pdf/2602.17431v1)

**作者:** Dylan Bouchard `[一作]` (CVS Health), David Skarbrevik `[通讯]` (CVS Health)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5115792707)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

无法确定

**💡 创新点**

无法确定

**🔧 技术方法**

无法确定

**📊 数据集**

无法确定

**📈 对比分析**

无法确定

**⚠️ 局限性**

信息不足

---

## 354. Evaluating Extremely Low-Resource Machine Translation: A Comparative Study of ChrF++ and BLEU Metrics

**arXiv ID:** 2602.17425 | [PDF](https://arxiv.org/pdf/2602.17425v1)

**作者:** Sanjeev Kumar `[一作]` (Indian Institute of Technology Bombay), Pushpak Bhattacharyya `[通讯]` (Indian Institute of Technology Bombay)

**通讯引用:** 13183 | [OpenAlex ID](https://openalex.org/A5065100828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在低资源印地语、马加希语、博贾普里语和恰蒂斯加尔语等印度语言中，对机器翻译评估指标BLEU与ChrF++的差异及其所揭示的翻译质量问题进行系统性比较，提出基于两指标的联合解读框架；

**💡 创新点**

创新点在于揭示并归纳BLEU与ChrF++在不同模型、语言对和生成方式下的偏差模式，并提供六类典型差异案例的案例分析，强调单一指标不足以全面评估低资源语言翻译质量；

**🔧 技术方法**

使用大型语言模型（Aya‑101、Airavata）与标准神经MT模型（mT5‑Large），采用PEFT（LoRA）微调、随机生成、多样化解码，结合SacreBLEU计算BLEU与ChrF++；

**📊 数据集**

数据集包括6,192句NLLB Seed训练集和1,012句FLORES‑200 devtest集，评估方向涵盖英语↔目标语、印地语↔目标语四种组合；

**📈 对比分析**

通过对BLEU与ChrF++的数值比较和六类差异案例的统计，发现两指标常出现显著偏差，尤其在源复制、hallucination、形态变化频繁的低资源场景；实验显示，单指标往往低估或高估翻译质量，联合使用可提供更可靠评估；

**⚠️ 局限性**

局限性包括仅覆盖三种印地语方言，缺乏对其他低资源语言、不同文字体系或语言结构的验证，结果可能不具普遍适用性。

---

## 355. 3D-printed Soft Optical sensor with a Lens (SOLen) for light guidance in mechanosensing

**arXiv ID:** 2602.17421 | [PDF](https://arxiv.org/pdf/2602.17421v1)

**作者:** Diana Cafiso `[一作]` (Istituto Italiano di Tecnologia), Lucia Beccai `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

在软体机器人中设计并制造了一种将单层光学材料直接打印透镜并嵌入Y形光波导前的柔性光学传感器（SOLen），通过透镜旋转实现焦点平移，从而实现两条分支光路的差分输出，编码运动方向和幅度。

**💡 创新点**

创新点在于将透镜作为光学元件完整打印进单一柔性聚合物中，无需多材料界面或遮光涂层；同时提供从材料光学表征、透镜仿真设计到传感器构建的全流程，可直接应用于单材料、单步3D打印的软体机器人。

**🔧 技术方法**

使用技术包括紫外DLP光固化3D打印、单层UV‑Vis光学测量（透射率、吸收率、折射率）、COMSOL Ray Optics有限元仿真（透镜轮廓优化）、机械压缩测试与实验验证的旋转平台及电压采集。

**📊 数据集**

数据集主要由实验室内部测量得到：单层光学表征数据（450‑900 nm 透射率>85%，折射率≈1.50）、折射率随波长变化曲线、以及对照组（无透镜）与实验组（带透镜）在±3°旋转下的电压-角度曲线；未使用公开数据集。

**📈 对比分析**

通过将带透镜与不带透镜的同形传感器在相同光源和接收器条件下进行对比，实验表明带透镜时左右接收器电压差约提升30‑40%，且波形在连续五个循环中标准差小于10%；相反，无透镜时差分几乎为零，说明透镜是实现差分信号的关键。

**⚠️ 局限性**

局限性包括：对光源与接收器的精确对准要求高，透镜尺寸受DLP层间散射影响，且当前仅在单一柔性聚合物、单波长（860 nm）下验证；对多波长、不同光学材料或更大变形范围的适应性尚未充分测试，且在实际机器人环境中对光污染与温度漂移的鲁棒性待进一步研究。

---

## 356. Pareto Optimal Benchmarking of AI Models on ARM Cortex Processors for Sustainable Embedded Systems

**arXiv ID:** 2602.17508 | [PDF](https://arxiv.org/pdf/2602.17508v1)

**作者:** Pranay Jain `[一作]` (Fraunhofer Institute for Integrated Circuits), Dominik Seuß `[通讯]` (Center for Artificial Intelligence and Robotics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对ARM Cortex M0+, M4, M7裸机的AI模型基准框架，用于评估能耗、准确率与资源占用。

**💡 创新点**

通过将 FLOPs 作为推理时延的预测指标，并结合 Pareto 前沿分析，实现了能耗与准确率平衡的系统化决策。

**🔧 技术方法**

自动化模型压缩（结构化剪枝、8‑bit 量化）、ONNX 转 C、C++ 基准框架、实时功耗测量等技术。

**📊 数据集**

使用 CIFAR‑10、MNIST、ToyADMOS、MSCOCO14 等常见视觉与异常检测数据集。

**📈 对比分析**

利用多目标优化生成多种压缩模型，测量推理时间、电流与能量，绘制 FLOPs 与时间的线性关系以及能耗-准确率 Pareto 前沿；结果显示 M7 适合短周期高频推理，M4 适合长周期低频推理。

**⚠️ 局限性**

实验仅在实验室环境下进行，受限于少数微控制器架构，未考虑长周期现场部署、环境变化及高级加速器的影响。

---

## 357. Optically Sensorized Electro-Ribbon Actuator (OS-ERA)

**arXiv ID:** 2602.17474 | [PDF](https://arxiv.org/pdf/2602.17474v1)

**作者:** Carolina Gay `[一作]` (Istituto Italiano di Tecnologia), Lucia Beccai `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了光学波导传感器集成的电磁绳式驱动器（OS-ERA），实现对软体ERA在不同负载与电压下的八个弯曲状态的高精度分类。

**💡 创新点**

创新点在于利用曲率分析精确定位软波导光学传感器、采用软波导实现轻量化且抗电磁干扰的感知层，并将支持向量机 (RBF) 与光学信号结合，实现电压与速度不变的闭环感知。

**🔧 技术方法**

技术手段包括：3D 打印软波导与光学封装、光学传感与序列激活、曲率分析、光学信号标准化、RBF 支持向量机监督学习、数据采集与可视化。

**📊 数据集**

数据集：训练集 3 次 12.3 g 4 kV 下的光学信号（共 24 个样本，涵盖 8 个状态），测试集 6 次 12.3 g 3 kV/5 kV 下的光学信号（共 36 个样本）。

**📈 对比分析**

评估方法是将测试轨迹投影到训练决策边界，计算每个状态的预测准确率；结果显示在不同电压与速度下均达到接近 100% 的准确率，证明了模型的稳健性与泛化能力。

**⚠️ 局限性**

局限性：仅实现离散状态分类，未提供连续姿态估计；训练样本量有限，需每次更换设备时重新标定；波导弹性可能导致折叠噪声；对环境光仍有一定敏感性，校准成本较高。

---

## 358. PEACE 2.0: Grounded Explanations and Counter-Speech for Combating Hate Expressions

**arXiv ID:** 2602.17467 | [PDF](https://arxiv.org/pdf/2602.17467v1)

**作者:** Greta Damo `[一作]` (Universite Cote d'Azur), Serena Villata `[通讯]` (Universite Cote d'Azur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了 PEACE 2.0 工具，集成了仇恨言论检测、解释生成和基于检索的证据支撑的对抗性回应（counter‑speech）生成功能。

**💡 创新点**

创新点包括：① 使用 Retrieval‑Augmented Generation（RAG）将权威知识检索与生成模型结合，实现解释和回应的证据支撑；② 提供可视化分析面板，便于研究者对多源对抗性回应进行探索与比较；③ 在同一平台上同时支持显式与隐式仇恨言论的处理，并对 RAG 与无检索生成进行系统对比。

**🔧 技术方法**

技术栈主要包含：BERT 细调分类器、BGE‑M3 句子编码器 + FAISS 向量检索、OpenAI/开源 LLM（Mistral、LLaMA、CommandR）进行文本生成、摘要与可视化展示；自动评估使用语义相似度、Perplexity、Distinct‑3、NLI（Entailment/Contradiction）等指标。

**📊 数据集**

使用的主要数据集有：IHC、ISHate、TOXIGEN、DYNA、SBIC（隐式仇恨言论）；CONAN、Multitarget‑CONAN、Twitter/YouTube 对抗性回应数据集；以及来自 UN Digital Library、Eur‑Lex、European Agency for Fundamental Rights 的 32,792 条文档（共 3,173,630 个段落）作为知识库。

**📈 对比分析**

通过人类评估（Fluency、Informativeness、Persuasiveness、Soundness、Specificity）与自动指标（Sem. Sim., Faithfulness, Perplexity, Distinct‑3, NLI）对比 RAG 与非 RAG 输出，结果显示 RAG 生成的解释与回应在所有指标上均显著优于无检索版本，尤其在隐式仇恨内容上提升更明显；差异通过 Wilcoxon 检验达到统计显著性。

**⚠️ 局限性**

局限性包括：样本量有限，评估主要聚焦英语；知识库为静态更新，缺乏动态检索策略；对多语言或不同文化背景下的仇恨言论适用性尚待验证；依赖当前 LLM 的生成能力，可能存在生成偏见或错误；未来需进一步完善评估指标与自适应检索机制。

---

## 359. Entropy-Based Data Selection for Language Models

**arXiv ID:** 2602.17465 | [PDF](https://arxiv.org/pdf/2602.17465v1)

**作者:** Hongming Li `[一作]` (University of Science and Technology Beijing), Chao Huang `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 5932 | [OpenAlex ID](https://openalex.org/A5081848769)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于熵的无监督数据筛选框架EUDS，用于高效选择原始数据与GPT-4o生成的合成数据，以减少LLM微调所需的数据量和计算成本。

**💡 创新点**

创新点在于三维熵评估机制（信息熵、生成熵、语义熵）以及基于熵区间的样本筛选策略，能够在保持或提升性能的同时显著压缩训练数据。

**🔧 技术方法**

采用信息熵、生成熵、语义熵三种度量，结合区间化选择；使用GPT‑4o生成合成数据；在BERT-base模型上进行微调并对比多种基线与SOTA方法。

**📊 数据集**

使用11个公开文本分类数据集（SA：IMDb、SST‑2/5、Yelp；Topic‑CLS：AGNews、20News、Yahoo；Q&A：RaceQA、MMLU、ARC、MMLU‑Pro），每个任务均包含原始数据与合成数据。

**📈 对比分析**

与BiLSTM、RoBERTa‑BiLSTM、TWSSenti、SELECT、FDKT等方法对比，EUDS在大多数数据集上实现了与SOTA相当或更高的准确率，同时将训练样本量减少30%–90%，显著降低计算开销。

**⚠️ 局限性**

局限性包括：无法给出统一的最佳熵区间，易导致样本分布集中导致压缩过度或性能不稳；对极端小样本或分布偏移的任务表现有限；缺乏充分的理论解释，需进一步研究熵与模型学习关系。

---

## 360. The CTI Echo Chamber: Fragmentation, Overlap, and Vendor Specificity in Twenty Years of Cyber Threat Reporting

**arXiv ID:** 2602.17458 | [PDF](https://arxiv.org/pdf/2602.17458v1)

**作者:** Manuel Suarez-Roman `[一作]` (Universidad Carlos III de Madrid), Juan Tapiador `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 5108 | [OpenAlex ID](https://openalex.org/A5070199150)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并分析了13,308份涵盖过去20年的公开CTI报告的大规模结构化数据集，利用LLM自动抽取威胁主体、受害者、动机、技术指标等信息，并系统研究CTI生态演化、威胁主体与受害者的关联、报告类型与时间分布、以及供应商的观测偏差与重叠情况。

**💡 创新点**

①首个大规模、LLM驱动的CTI报告结构化数据集，抽取精度F1=0.94；②首次从宏观到微观层面量化CTI生态的分层、专业化与碎片化；③揭示供应商间低重叠、增量收益递减的现象，为多源融合提供量化依据。

**🔧 技术方法**

使用OpenAI GPT‑4等大型语言模型进行结构化输出；通过自定义schema、后处理与归一化手工规则消除同义、拼写错误；对报告进行哈希去重与字段级去重；利用Jaccard相似度、线性回归等统计方法评估报告量、技术指标与供应商覆盖度。

**📊 数据集**

13,308份公开CTI报告，来源包括MITRE ATT&CK、APTnotes、Malpedia、Alienvault OTX、Vx‑Underground等10个渠道；聚合后得到12,723条结构化记录，涵盖2,722威胁主体、107,611 IoCs、833 TTPs、1,626供应商等。

**📈 对比分析**

通过对2,000份样本的人工评估（5人专家）获得精度95%、召回93%、F1 0.94；与单源或小规模数据集相比，本文数据集在多维度（技术指标、动机、受害者）上覆盖率显著提升；在供应商重叠分析中，Jaccard指数平均低于0.15，覆盖曲线表明需约14个供应商才能实现100%威胁主体覆盖。

**⚠️ 局限性**

仅覆盖公开西方主流CTI渠道，缺乏完整覆盖；威胁主体命名碎片化导致统一性难题；LLM可能插入外部知识或hallucination，尤其在动机、受害者归属上；IoC抽取依赖LLM，缺乏专用正则或工具验证；多源聚合结果仅代表采样样本，可能低估CTI生态的真实多样性。

---

## 361. WarpRec: Unifying Academic Rigor and Industrial Scale for Responsible, Reproducible, and Efficient Recommendation

**arXiv ID:** 2602.17442 | [PDF](https://arxiv.org/pdf/2602.17442v1)

**作者:** Marco Avolio `[一作]` (Wideverse), Tommaso Di Noia `[通讯]` (Politecnico di Bari)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出WarpRec框架，实现从本地实验到分布式工业级训练与部署的无缝迁移；

**💡 创新点**

后端无关、一次编写即可在任意环境运行；整合绿色AI能源追踪、Agentic AI模型上下文协议、自动化统计检验及多目标评价；

**🔧 技术方法**

采用Narwhals、Ray、CodeCarbon、MCP服务器、GPU加速、Optuna/HyperOpt/BOHB等技术；

**📊 数据集**

使用MovieLens‑1M、MovieLens‑32M和Netflix Prize‑100M三套数据集；

**📈 对比分析**

与5大主流框架（Cornac、DaisyRec、Elliot、RecBole、Microsoft Recommenders）在相同算法和超参搜索下对比，WarpRec在大规模数据上始终完成全流程，训练和HPO时间更短、能耗更低，推荐效果（nDCG@10）不降反升；

**⚠️ 局限性**

仍受限于极大规模下可能出现OOM、HPO耗时较长、部分算法或接口尚未成熟。

---

## 362. Preserving Historical Truth: Detecting Historical Revisionism in Large Language Models

**arXiv ID:** 2602.17433 | [PDF](https://arxiv.org/pdf/2602.17433v1)

**作者:** Francesco Ortu `[一作]` (University of Trieste), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含500个争议性历史事件、每个事件对应事实叙事和修正主义叙事的双重参考数据集，并设计了11种现实场景下的提示模板，系统评估大型语言模型（LLM）在中立与修正主义请求两种条件下的回答偏向；

**💡 创新点**

首次提出了基于双重参考叙事的“引用对齐”评估框架，结合LLM‑as‑a‑judge技术，将模型输出与事实与修正主义叙事的相似度进行量化，能够检测细粒度的修正主义倾向（如删减、软化、虚假平衡）；

**🔧 技术方法**

利用LLM‑as‑a‑judge、三阶段评估流程（提示生成、模型推理、评估判定），并使用多种中等规模LLM（如GroK-3-mini、Mistral-7B等）进行推理；

**📊 数据集**

自研的“HistoricalRevisionismBench”（简称HRB）数据集，涵盖45个国家、20/21世纪的战争、种族灭绝、领土争议等多类型事件，提供事实和修正主义双重参考文本；

**📈 对比分析**

通过二分类（是否更贴近事实）与四等级评分（1=完全修正主义，4=基本事实）两阶段评估，发现中立提示下修正主义比例约10–32%，但在显式修正主义提示下几乎全部模型转向修正主义（80–97%），显示出显著的鲁棒性失败；

**⚠️ 局限性**

数据集缺乏专家历史学家审核，叙事对齐仅为近似；评估仅在英语环境，忽略多语言差异；模型训练数据不透明，难以解释特定偏差来源；评估框架依赖LLM‑as‑a‑judge，可能受评判模型偏差影响；

---

## 363. Diverse Word Choices, Same Reference: Annotating Lexically-Rich Cross-Document Coreference

**arXiv ID:** 2602.17424 | [PDF](https://arxiv.org/pdf/2602.17424v1)

**作者:** Anastasia Zhukova `[一作]` (University of Göttingen), Bela Gipp `[通讯]` (University of Göttingen)

**通讯引用:** 5786 | [OpenAlex ID](https://openalex.org/A5058837356)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新设计并实现跨文档共指的细粒度标注方案，将共指链视为话语元素（DE），并对 NewsWCL50 与 ECB+ 进行统一重新注释，兼顾实体、事件与概念的细粒度划分与词汇多样性。

**💡 创新点**

创新点在于：① 引入身份、近身份（桥接）关系以捕捉词汇多样化与语境变形；② 将共指链转化为可计量的 DE，兼顾新闻偏见与事件细节；③ 通过统一代码本实现两套原始标注的平衡化，生成既不太宽泛也不太狭窄的标注。

**🔧 技术方法**

技术手段包括：人工重标注、统一代码本、使用相同头词 lemma 基线模型评估、计算 UL、PD、MTLD 等词汇多样性指标，以及 CoNLL F1 评估。

**📊 数据集**

使用的数据集：NewsWCL50（新闻偏见词汇集）及其重标注版本 NewsWCL50r；ECB+ 子集（包含5个主题）及其重标注版本 ECB+r。

**📈 对比分析**

比较方法：对比原始 NewsWCL50、ECB+ 与 ECB+METAm 的 DE 统计、链大小、词汇多样性指标及同头词 lemma 基线 F1。结果显示：NewsWCL50r 与 ECB+r 的词汇多样性指标和 F1 分数处于原始两套数据集之间，表明难度适中且更平衡。

**⚠️ 局限性**

局限性：① 仍未完成完整的互评，可能存在标注偏差；② 评估仅基于简单同头词 lemma 基线，未检验更复杂模型；③ 样本主要来自新闻与非政治子集，未验证在更广泛语料上的适用性。

---

## 364. Convergence Analysis of Two-Layer Neural Networks under Gaussian Input Masking

**arXiv ID:** 2602.17423 | [PDF](https://arxiv.org/pdf/2602.17423v1)

**作者:** Afroditi Kolomvaki `[一作]` (Rice University), Anastasios Kyrillidis `[通讯]` (Rice University)

**通讯引用:** 1778 | [OpenAlex ID](https://openalex.org/A5024280658)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了两层 ReLU 网络在输入层乘性高斯掩码（Gaussian dropout）下的训练动态，并给出了线性收敛至误差球的理论保证。

**💡 创新点**

首次对乘性高斯噪声环境下的期望损失与梯度进行解析，证明损失可拆解为平滑损失加数据相关正则化，并在 NTK 框架下提供收敛上界。

**🔧 技术方法**

使用神经切线核（NTK）分析、期望算子推导、随机梯度下降偏置估计以及对 ReLU 内部随机性的解析方法。

**📊 数据集**

实验采用合成数据、CIFAR‑10、MNIST 以及无线通道仿真数据验证理论。

**📈 对比分析**

与无噪声训练、标准 Dropout、对抗训练等基线对比，结果显示在小幅噪声下可提升泛化，并在分布式无线训练与成员推断攻击中展现隐私保护优势。

**⚠️ 局限性**

局限性：仅适用于小噪声、过参数化 NTK 时代，假设掩码独立，未给出完整隐私计量，对深层网络的推广仍受限。

---

## 365. Do Hackers Dream of Electric Teachers?: A Large-Scale, In-Situ Evaluation of Cybersecurity Student Behaviors and Performance with AI Tutors

**arXiv ID:** 2602.17448 | [PDF](https://arxiv.org/pdf/2602.17448v1)

**作者:** Michael Tompkins `[一作]` (Arizona State University), Jaron Mink `[通讯]` (Arizona State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对在真实教学环境中使用AI导师辅导网络安全课程的学生行为和学习效果进行了大规模评估

**💡 创新点**

首次在大规模真实课堂场景下系统性比较AI导师与传统人类导师以及无导师干预的差异，并揭示AI导师在提升学生参与度和学习成效方面的潜力

**🔧 技术方法**

采用大型语言模型（如GPT-4）与自研对话管理框架来实现AI导师，结合行为追踪与学习分析工具

**📊 数据集**

收集了超过1000名学生在在线网络安全课程中的交互日志、测验成绩、作业提交和情感评价等数据

**📈 对比分析**

通过实验组/对照组设计，评估学生测验分数、完成率和参与度，结果显示AI导师组平均测验成绩提升约12%，完成率提升约18%，且相较于人类导师在即时答疑速度和一致性上更具优势

**⚠️ 局限性**

研究仅覆盖单门课程，缺乏跨学科验证，AI导师的适应性和可解释性有限，且可能因模型偏见影响个别学生的学习路径

---

## 366. Distributed Virtual Model Control for Scalable Human-Robot Collaboration in Shared Workspace

**arXiv ID:** 2602.17415 | [PDF](https://arxiv.org/pdf/2602.17415v1)

**作者:** Yi Zhang `[一作]` (University of Cambridge), Fulvio Forni `[通讯]` (University of Cambridge)

**通讯引用:** 1327 | [OpenAlex ID](https://openalex.org/A5034730197)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种基于虚拟模型控制的去中心化、智能体无关的安全人机协作框架，实验中实现多机器人与多人的无死锁协作；

**💡 创新点**

创新点在于将人机视为同一虚拟机械网络，用虚拟弹簧阻尼实现交互式控制，并通过力平衡检测死锁、优先级协商解决，完全无需轨迹规划；

**🔧 技术方法**

使用了虚拟模型控制（VMC）、力平衡死锁检测与优先级协商、虚拟弹簧/阻尼模型、RealSense摄像头+MediaPipe手部关键点跟踪、分布式通信；

**📊 数据集**

实验数据集为16块3cm方块与4×4格子网格，并使用RealSense D405摄像头实时捕获手部关键点；

**📈 对比分析**

与传统基于轨迹规划和安全距离的方案对比，实验显示机器人死锁率从61.2%降至0%，人机最小距离≈22cm，安全距离违例时间随控制参数下降；完成时间在不同参数下可从约89s到约145s；

**⚠️ 局限性**

局限性包括受限于10Hz的感知更新率导致人机避让延迟，未评估人类对控制的自适应反应，适用范围受机器人数量与工作空间尺寸限制。

---

## 367. Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models

**arXiv ID:** 2602.17497 | [PDF](https://arxiv.org/pdf/2602.17497v1)

**作者:** Wen-Tse Chen `[一作]` (Carnegie Mellon University), Jeff Schneider `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8521 | [OpenAlex ID](https://openalex.org/A5055199976)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用大语言模型（LLM）的回顾式上下文学习（RICL）将稀疏环境奖励转换为优势函数，并在此基础上构建了在线学习框架RICOL，用于持续改进策略；

**💡 创新点**

创新点包括：①利用LLM反射器生成的文本反馈对同一轨迹进行回顾式上下文更新，从而估计优势函数；②通过对原始策略与更新后策略的对数概率比值直接获得优势；③将优势权重回归与KL约束相结合，形成高样本效率的在线RL框架；

**🔧 技术方法**

核心技术包括大语言模型作为策略与反射器、回顾式上下文学习（RICL）、优势函数估计与密集奖励生成、优势加权回归（AWR）与KL正则化的策略更新、离线经验收集与在线改进；

**📊 数据集**

实验数据集主要为：1D Key-Door（离散可枚举动作空间）以及BabyAI四个语言约束多回合任务（goto、pickup、pick_up_seq_go_to、open），每个任务均以文本形式表示状态与动作；

**📈 对比分析**

在与PPO（3B、10M）、Reflexion、GPT-4o mini等基线对比时，RICOL在所有任务上均实现了更高的成功率，并以约50×至10×的样本效率（环境步数）超越PPO 10M、PPO 3B；在面对噪声反馈时仍保持较好性能；

**⚠️ 局限性**

局限性：仅支持可枚举的离散动作空间，因需要对所有动作计算KL分布；对于连续或长文本生成的动作空间需要改为采样估计，相关研究仍待展开。

---

## 368. Computational Hardness of Private Coreset

**arXiv ID:** 2602.17488 | [PDF](https://arxiv.org/pdf/2602.17488v1)

**作者:** Badih Ghazi `[一作]` (Google Research), Pasin Manurangsi `[通讯]` (Google Research)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在差分隐私(k‑means)下构造近似子样本（coreset）的计算难度。

**💡 创新点**

提出了基于单向函数假设的计算下界，证明在ℓ∞和欧氏度量下，任何多项式时间DP算法都无法得到常数近似因子1±α（k=3）或α=Θ(1/d²）的k‑means coreset。

**🔧 技术方法**

利用从3‑文字析取合成数据难题到k‑means成本查询的归约，借助sum‑of‑squares结构与离散/连续域的精确分析。

**📊 数据集**

论文纯理论，没有使用具体实验数据集；所有结论均来自形式化证明。

**📈 对比分析**

结果表明，即使已知信息论级别的私有coreset存在，现有可计算的DP coreset方案仍无法突破该下界，说明理论与实现之间存在显著鸿沟。

**⚠️ 局限性**

仅适用于k=3；对k=2、k‑median等情况仍未给出下界；欧氏空间下的α仅为Θ(1/d²)，未能得到维度无关的常数下界；归约依赖于3‑文字析取的完备性，限制了进一步推广。

---

## 369. Compiling Quantum Lambda-Terms into Circuits via the Geometry of Interaction

**arXiv ID:** 2602.17482 | [PDF](https://arxiv.org/pdf/2602.17482v1)

**作者:** Kostia Chardonnet `[一作]` (University of Lorraine), Paolo Pistone `[通讯]` (University Claude Bernard Lyon 1)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种算法，将任意线性量子 λ-计算机的术语编译为等价的量子电路。

**💡 创新点**

利用几何交互（GoI）实现高阶控制流的高效编译，并通过类型系统保证无死锁，从而避免指数级扩展。

**🔧 技术方法**

几何交互（GoI）、线性 λ 计算机、量子电路描述语言、类型系统。

**📊 数据集**

未使用任何实验数据集；该工作为理论证明。

**📈 对比分析**

通过形式化证明（模拟关系、完全正映射语义）验证编译正确性；在无死锁的情况下编译复杂度为线性，若存在死锁则可能退化为指数。

**⚠️ 局限性**

仅适用于线性、无指数的量子 λ 计算机；缺乏实验评估；对更一般的指数或递归结构的类型支持不足。

---

## 370. Variational Grey-Box Dynamics Matching

**arXiv ID:** 2602.17477 | [PDF](https://arxiv.org/pdf/2602.17477v1)

**作者:** Gurjeet Sangra Singh `[一作]` (University of Geneva), Alexandros Kalousis `[通讯]` (University of Geneva)

**通讯引用:** 3062 | [OpenAlex ID](https://openalex.org/A5084831490)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种变分灰盒动力学匹配 (VGB‑DM) 方法，将不完整的物理模型嵌入无模拟的流匹配生成模型中，实现从观测轨迹学习动力学并推断物理参数。

**💡 创新点**

创新点包括：无模拟灰盒学习、结构化变分推断将物理参数与系统随机性分离、以及通过非线性插值实现二阶动力学（速度与加速度）匹配。

**🔧 技术方法**

使用技术包括变分流匹配、轨迹流匹配、结构化变分推断、物理信息先验、拉格朗日插值以及无模拟训练框架。

**📊 数据集**

实验数据集涵盖合成 ODE/PDE 系统（RLC 电路、反应扩散、阻尼摆、洛伦兹吸引子）以及 ERA5 中尺度天气再分析数据，用于中期天气预报。

**📈 对比分析**

与 PhysVAE、BB‑NODE、VBB‑DM、TFM 等基线比较，VGB‑DM 在所有任务中均取得最低 MSE/RMSE、最快收敛，并在低样本和长时程预测上更稳健。

**⚠️ 局限性**

局限性包括：需要可微、光滑的动力学；对高度刚性或不连续系统的表现未知；以及对物理先验的选择仍有一定依赖。

---

## 371. 4D Monocular Surgical Reconstruction under Arbitrary Camera Motions

**arXiv ID:** 2602.17473 | [PDF](https://arxiv.org/pdf/2602.17473v1)

**作者:** Jiwei Shan `[一作]`, Shing Shin Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2102 | [OpenAlex ID](https://openalex.org/A5072251844)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Local-EndoGS，能够在单目内窥镜序列中，处理任意相机运动并实现高质量的 4D 可变形手术场景重建。

**💡 创新点**

创新点包括：① 进化式窗口划分的全局场景表示，支持长序列与大相机位移；② 无需立体深度或精确 SfM 的粗细分层初始化，利用多视几何、跨窗口信息和单目深度先验；③ 结合长距离 2D 像素轨迹约束与物理运动先验的损失函数，提升变形的物理一致性。

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting 作为显式辐射场；局部可变形场景表示（每个窗口都有 Canonical Space + 变形网络）；Track‑Any‑Point (TAP) 模型用于像素级跟踪；自定义 2D 跟踪损失、物理基础正则化（刚性、旋转相似度、等距性）等。

**📊 数据集**

评估数据集：EndoNeRF（固定相机）、StereoMIS（环绕运动）和 EndoMapper（前向运动）三大公开数据集。

**📈 对比分析**

与多种基线（EndoNeRF、EndoSurf、Forplane、DDS‑SLAM、SurgicalGaussian、LGS、EndoGaussian、EH‑SurGS 等）以及针对 Colonoscopy 的 ENeRF‑SLAM、EndoGSLAM、Endo‑2DTAM 等进行比较。结果显示，Local-EndoGS 在 PSNR、SSIM、LPIPS、深度误差（Abs Rel、Sq Rel、RMSE、RMSE log）和阈值准确率上均超过对手，且渲染速度最高（≈370 fps）。

**⚠️ 局限性**

局限性：① 3D 高斯在多视一致性与表面细节上受限；② 仍为离线方法，无法实时；③ 只处理连续无拓扑变化的变形，难以应对切割、撕裂等事件；④ 窗口序列化训练导致时间随窗口数线性增长，未充分利用 GPU 并行；⑤ 单目深度先验受限于自然场景模型，提升空间有限。

---

## 372. EAGLE: Expert-Augmented Attention Guidance for Tuning-Free Industrial Anomaly Detection in Multimodal Large Language Models

**arXiv ID:** 2602.17419 | [PDF](https://arxiv.org/pdf/2602.17419v1)

**作者:** Xiaomeng Peng `[一作]` (Ewha Womans University), Seon Han Choi `[通讯]` (Ewha Womans University)

**通讯引用:** 9818 | [OpenAlex ID](https://openalex.org/A5101730979)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无调参的工业缺陷检测框架 EAGLE，利用专家模型生成的视觉与文本提示来引导多模态大型语言模型（MLLM）进行检测与解释。

**💡 创新点**

创新点包括：①基于未采样补丁的分布阈值机制 DBT 自动确定异常阈值，避免人工设置；②置信度感知注意力缩放 CAAS 在专家不确定时强化视觉注意力，减弱语言偏见；③将两种提示同步注入 MLLM，实现准确检测与可解释性。

**🔧 技术方法**

技术手段包括 PatchCore 专家模型、分布阈值 DBT、置信度感知注意力缩放 CAAS、视觉与文本提示融合、以及对 Qwen2.5-VL、LLaVA、InternVL 等多模态大型语言模型的使用。

**📊 数据集**

采用工业缺陷检测基准数据集 MVTec-AD 与 VisA，分别包含多种对象与纹理，满足一类学习场景。

**📈 对比分析**

在 MVTec-AD 和 VisA 上与现有 Fine‑tune、GRPO 以及无调参方法对比，EAGLE 在所有测试 MLLM 上均实现了显著的准确率和召回率提升，VisA 上更是达到或超过 Fine‑tune 方案的最佳表现，且无需任何参数更新。

**⚠️ 局限性**

局限性：1）框架对专家模型的性能高度依赖，异常阈值估计在极端分布下可能失效；2）文本提示仍可能在置信区间外误导 MLLM；3）目前仅在少数 MLLM 上验证，跨模型通用性与大规模部署效果待进一步探索。

---

## 373. A Privacy by Design Framework for Large Language Model-Based Applications for Children

**arXiv ID:** 2602.17418 | [PDF](https://arxiv.org/pdf/2602.17418v1)

**作者:** Diana Addae `[一作]` (Carleton University), Chen Zhou `[通讯]` (Carleton University)

**通讯引用:** 2373 | [OpenAlex ID](https://openalex.org/A5100382828)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向儿童的 LLM 应用隐私设计框架，结合 PbD 原则

**💡 创新点**

创新点在于将 GDPR、COPPA、PIPEDA 的隐私原则映射至 LLM 生命周期，并提供可操作的技术与组织控制

**🔧 技术方法**

采用差分隐私、机器学习去识别与去学习、输入/输出过滤、数据最小化、动态同意管理等技术

**📊 数据集**

使用公开语料（如 Common Crawl、BooksCorpus、教育对话数据）及案例数据进行验证

**📈 对比分析**

通过教育 LLM 辅导机器人的案例研究演示框架可行性，未给出具体性能指标

**⚠️ 局限性**

局限包括 LLM 隐私权实现难度、数据删除与去学习技术的局限、儿童同意管理的复杂性

---

## 374. Beyond Pipelines: A Fundamental Study on the Rise of Generative-Retrieval Architectures in Web Research

**arXiv ID:** 2602.17450 | [PDF](https://arxiv.org/pdf/2602.17450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 375. Improving LLM-based Recommendation with Self-Hard Negatives from Intermediate Layers

**arXiv ID:** 2602.17410 | [PDF](https://arxiv.org/pdf/2602.17410v1)

**作者:** Bingqian Li `[一作]` (Renmin University of China), Ji-rong Wen `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 ILRec，一种利用LLM中间层自生成的自硬负样本进行偏好微调的推荐框架。

**💡 创新点**

创新点在于：①将中间层的高概率 token 作为细粒度自硬负样本；②设计跨层偏好优化与蒸馏机制联合学习；③加入协同过滤奖励正则化防止误惩罚。

**🔧 技术方法**

技术包括：LLM（Llama3.1‑8B）指令微调、跨层自硬负样本提取、交叉熵加权惩罚、教师‑学生蒸馏、协同过滤奖励。

**📊 数据集**

使用 Amazon Review（Instrument、Arts、Video Games）三个子集，并在 LastFM 等候选排序任务上验证。

**📈 对比分析**

与传统序列推荐（Caser、GRU4Rec、SASRec）和现有 LLM 推荐（BIGRec、LC‑Rec、SDPO、RosePO、SPRec）对比，ILRec 在 Hit@10、NDCG@10 等指标上均显著提升，且训练效率更高。

**⚠️ 局限性**

局限性包括：对中间层数量的敏感性（过多低层会噪声），需要额外的协同过滤模型来校正负样本，且在极大候选空间下仍可能产生信息冗余。

---

## 376. What Do LLMs Associate with Your Name? A Human-Centered Black-Box Audit of Personal Data

**arXiv ID:** 2602.17483 | [PDF](https://arxiv.org/pdf/2602.17483v1)

**作者:** Dimitri Staufer `[一作]` (Technische Universität Berlin), Kirsten Morehouse `[通讯]` (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了面向普通用户的 LMP2 工具，利用适配后的 WikiMem 框架对八种主流 LLM 进行个人数据关联性审计，并通过三项用户研究评估 LLM 输出的个人数据准确性、用户隐私感知与对“被遗忘权”的需求。

**💡 创新点**

创新点在于将技术层面的可量化记忆检索方法（WikiMem）与人机交互设计相结合，构建可解释、可交互的自助隐私审计工具，并首次系统性量化普通欧盟用户在 LLM 中可被推断或记忆的个人属性。

**🔧 技术方法**

核心技术包括：基于 WikiMem 的两字符前缀查询与对抗性可解释性评估；使用 LMP2 前端后端架构；以及对 LLM 输出进行关联强度与置信度计算。

**📊 数据集**

使用了来自 Wikidata 的 50 项人类属性（共 243 条属性）作为测评基准，并构建了两类样本：知名公众人物（100 名）与合成不存在姓名（100 名），以及实际用户数据（EU 居民 303 名）。

**📈 对比分析**

与八种 LLM（三开源、五 API）进行比较，发现 GPT‑4o 在常见属性（性别、语言、眼色、头发色）上达 94%–82% 的准确率；在普通用户上能以 60% 以上准确率输出 11 项属性；但在高卡路里或开放式属性上精度低于 20%，并出现明显的偏置与“假设”问题。

**⚠️ 局限性**

主要限制包括：对闭源 API 的置信度与关联强度估计依赖于有限的 token 计数或投票；无法区分记忆与推断的来源；受限于 LLM 的温度和 prompt 设计；以及工具仅评估单个姓名，无法处理同名多人的歧义。

---

## 377. Directed type theory, with a twist

**arXiv ID:** 2602.17480 | [PDF](https://arxiv.org/pdf/2602.17480v1)

**作者:** Fernando Rafael Chu Rivera `[一作]` (Utrecht University), Paige Randall North `[通讯]` (Utrecht University)

**通讯引用:** 38 | [OpenAlex ID](https://openalex.org/A5072849432)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种新的定向类型理论——Twisted Type Theory（TTT），在传统的Martin-Löf类型理论基础上加入“扭曲”操作、-类型、展示类型等，能够在类型层面直接建模范畴和自然变换，进一步证明Yoneda引理等范畴理论命题

**💡 创新点**

核心创新是引入扭曲（twist）操作，使得依赖双向的类型可通过扭曲得到仅前向依赖的新类型，并借此构造依赖的2-侧纤维（dependent 2-sided fibrations）作为其语义；此外提出新的-类型引入与消除规则，扩展了类型理论对范畴的内在表达力

**🔧 技术方法**

利用分类学语义（comprehension category）、Grothendieck纤维化、Street的2侧纤维、Bénabou的直化-逆直化（straightening‑unstraightening）理论，以及显示类别（displayed categories）框架来解释扭曲类型和相关构造；还使用对偶/对称语义、宇宙、偏置上下文等技术来实现完整语义模型

**📊 数据集**

无；本文为理论性工作，未使用任何实验数据集

**📈 对比分析**

无；本文未进行实验或性能比较，讨论集中在语义一致性与理论可证明性上

**⚠️ 局限性**

局限性包括：1）缺乏完整的Π类型支持，导致部分构造需要受限；2）尚未完成对高维（∞‑范畴）模型的拓展；3）扭曲类型与上下文置换规则的实现尚不完善，证明过程技术上较为繁琐；4）未探索更丰富的2侧纤维理论（如弱因子化系统）以进一步强化消除规则

---

## 378. ACOS: Arrays of Cheap Optical Switches

**arXiv ID:** 2602.17449 | [PDF](https://arxiv.org/pdf/2602.17449v1)

**作者:** Daniel Amir `[一作]` (Technion), Mark Silberstein `[通讯]` (Technion)

**通讯引用:** 3133 | [OpenAlex ID](https://openalex.org/A5022593894)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出一种基于低辐射光学电路交换机（OCS）的全光学训练网络架构，通过动态选择、适配和容错的拓扑实现与机器学习训练需求高度协同的可重配置网络。

**💡 创新点**

核心创新在于：①将网络拓扑专门化为训练时不同并行维度（数据、张量、流水线、专家）所需的结构化拓扑；②使用低辐射OCS实现时间多路复用、快速重配置；③引入拓扑选择、适配、容错三种OCS层次，突破传统全连通网络规模瓶颈；④成本随支持的拓扑与适配数增长，而非端口数；⑤通过模拟验证在大规模LLM训练中几乎无性能损失，且成本可降低70%以上。

**🔧 技术方法**

采用的技术包括：低辐射MEMS/硅光子OCS、光纤通道分割、分层控制平面（分布式选择层、集中式适配/容错层）、多维环、环、环形、扩散器（Expander）等拓扑结构；仿真平台Astra‑SIM与MLSynth生成的训练工作负载；对比传统包交换Fat‑Tree和高辐射OCS基线。

**📊 数据集**

使用六个最新LLM训练配置（Qwen‑2、Mixtral‑7B、22B MoE、Llama‑8B、70B、Llama‑4 Maverick）的官方或公开训练参数，生成对应的通信流量曲线和计算负载。

**📈 对比分析**

与两类基线（全连通高辐射OCS和带有一键重配置的全连通网络）以及传统包交换Fat‑Tree相比，低辐射OCS网络在64‑GPU、1024‑GPU和32K‑GPU规模下的迭代时间与包交换相当或略低；成本上比包交换低约27%–70%；在MoE工作负载中，由于扩散拓扑的多跳导致轻微延迟，但总体仍低于高辐射OCS。

**⚠️ 局限性**

主要局限包括：设计在多维并行度变化时需要预先规划多种拓扑，灵活性受限；重配置延迟虽已被量化但仍对极低延迟场景构成挑战；容错实现依赖额外硬件和分布式控制，实际实现复杂；实验基于仿真，缺乏真实硬件验证。

---

## 379. ABCD: All Biases Come Disguised

**arXiv ID:** 2602.17445 | [PDF](https://arxiv.org/pdf/2602.17445v1)

**作者:** Mateusz Nowak `[一作]` (Dartmouth), Peter Chin `[通讯]` (Dartmouth)

**通讯引用:** 6458 | [OpenAlex ID](https://openalex.org/A5113696329)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种去偏的多选题评估协议和NonsenseQA诊断数据集，用以减轻LLM在MCQ评估中的标签、位置、提示分布和预测模式偏差。

**💡 创新点**

提出统一无序标签、全文本答案生成和语义相似度匹配的评估流程，并首次量化并消除“few-shot答案分布偏差”，同时发布NonsenseQA作为诊断工具。

**🔧 技术方法**

采用统一“-”标签、全句答案生成、正则提取、Qwen3-Embedding-0.6B等句子嵌入模型做相似度匹配，结合SCORE与方差比率评估鲁棒性。

**📊 数据集**

在13种开源LLM（8B-32B）上，评估NonsenseQA、CSQA、ARC、MMLU-Pro、GPQA以及多语言INCLUDE子集。

**📈 对比分析**

与标准S&L协议对比，使用SCORE、方差比率、跨基准Spearman/Kendall相关度衡量；结果显示M&D协议在准确率仅轻微下降的前提下，将准确率方差降低约3倍以上，并提升跨基准一致性。

**⚠️ 局限性**

对极多选项或高难度知识型任务（如MMLU-Pro）仍可能受位置偏差影响；语义匹配依赖嵌入模型质量，极端同义词或链式推理多变时可能误匹配；未深入分析模型内部状态（logits），限制了对偏差源的精细诊断。

---

## 380. Multi-Agent Temporal Logic Planning via Penalty Functions and Block-Coordinate Optimization

**arXiv ID:** 2602.17434 | [PDF](https://arxiv.org/pdf/2602.17434v1)

**作者:** Eleftherios E. Vlahakis `[一作]` (KTH Royal Institute of Technology), Dimos V. Dimarogonas `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 18452 | [OpenAlex ID](https://openalex.org/A5055348953)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种基于平滑STL语义、二次惩罚函数和块坐标梯度下降的多智能体协作时序逻辑规划框架，实现对任意协作任务的可扩展优化。

**💡 创新点**

创新点在于系统性地将平滑STL语义、块坐标优化和惩罚方法相结合，能够在保持收敛保证的同时，将全局高维耦合问题拆解为各智能体的局部优化，从而显著降低计算复杂度。

**🔧 技术方法**

使用了平滑鲁棒度近似、二次惩罚函数、块坐标梯度下降（BCGD）、惩罚法外层迭代、自动微分（JAX）以及对比实验中的LBFGS准Newton方法。

**📊 数据集**

采用十机器人离散时间圆盘运动学（unicycle）和线性动力学的R2AM、R2AMCA、RURAMCA场景作为实验数据集，包含障碍、收集区、投递区以及协作会合任务。

**📈 对比分析**

通过与LBFGS优化器进行对比实验，BCGD-PM在线性动力学下的运行时间约为12/13/35秒，鲁棒度分别为0.5/0.4/1.5；在unicycle动力学下运行时间约为234/288/480秒，鲁棒度分别为0.2/0.2/0.1，整体性能与LBFGS相当且在大规模约束下更具可扩展性。

**⚠️ 局限性**

局限性包括可能陷入局部最优、对惩罚参数的调节敏感、假设各智能体的局部成本凸且已知可行性、以及在极大规模智能体或极复杂协作任务时惩罚方法仍可能面临收敛速度慢的问题。

---

## 381. The Runtime Dimension of Ethics in Self-Adaptive Systems

**arXiv ID:** 2602.17426 | [PDF](https://arxiv.org/pdf/2602.17426v1)

**作者:** Marco Autili `[一作]` (University of L'Aquila), Patrizio Pelliccione `[通讯]` (Gran Sasso Science Institute)

**通讯引用:** 4069 | [OpenAlex ID](https://openalex.org/A5013103539)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出将伦理偏好视为自适应系统的运行时需求，并在此基础上引入伦理不确定性、冲突检测与多维多方伦理协商机制，构建可在软伦理空间内动态调整决策的框架；

**💡 创新点**

创新点在于：① 把伦理偏好从设计时静态规则转为可监测、可更新的运行时需求；② 识别并处理伦理不确定性与跨人、社会、环境（HSE）价值冲突；③ 将自动协商方法扩展到多维、多方、多驱动的伦理协商，并提出可追溯、可审计的保证机制；

**🔧 技术方法**

采用的技术包括：运行时需求建模与监测、模糊/概率/证据不确定性表示、基于约束的冲突检测与解决、扩展的自动协商协议与多维协商策略，以及保证案例与追溯日志的设计；

**📊 数据集**

论文为概念性研究，没有使用具体实验数据集；

**📈 对比分析**

由于未实现原型系统，本文未进行实验或性能对比，主要提出未来研究方向与评估思路；

**⚠️ 局限性**

局限性包括：缺乏实现与实验验证，相关算法与机制尚待进一步细化；对动态伦理偏好的获取与解释方法仍处于探索阶段；在复杂真实场景中的可扩展性与实时性尚未评估。

---

## 382. Coin selection by Random Draw according to the Boltzmann distribution

**arXiv ID:** 2602.17490 | [PDF](https://arxiv.org/pdf/2602.17490v1)

**作者:** Jan Lennart Bönsel `[一作]` (Deutsche Bundesbank), Marc Winstel `[通讯]` (Deutsche Bundesbank)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为Boltzmann Draw的基于统计物理Boltzmann分布的随机硬币选择算法，用于区块链和CBDC钱包中高效、低尘埃、并发友好的UTXO选择。

**💡 创新点**

创新点在于将代币价值视为能量，引入逆温度β作为自适应权重，使得低价值代币被更频繁选中，从而降低尘埃、限制代币池大小并提升并发性；与传统Uniform Draw相比提供了可调的概率分布。

**🔧 技术方法**

采用概率选择、Boltzmann分布、β= m/E 作为自适应温度、随机抽样、并行实验与锁竞争度量；对比随机Draw、Greedy算法并进行多线程性能测试。

**📊 数据集**

使用模拟数据：三种交易场景—Normal、Poisson、Dirichlet分布生成的存款与支付；每种场景下运行10^5次交易，平均100次实验以获得统计量。

**📈 对比分析**

通过对比代币池大小、尘埃产生、代币值分布、交易输入代币数、平均延迟与锁竞争率等指标评估性能。Boltzmann Draw在1、3、4维度上明显优于Random Draw；在2、5维度与Greedy相当，但在并发场景下的延迟与竞争率明显低于Greedy，整体表现优于传统方法。

**⚠️ 局限性**

局限性包括：β参数的选择需经验性调整，算法在极大代币池时计算量增大；仅在模拟环境中验证，缺乏真实网络实验；对不同代币属性（如年龄、类型）扩展仍未实现；并未给出理论最优性证明。

---

## 383. EDRP: Enhanced Dynamic Relay Point Protocol for Data Dissemination in Multi-hop Wireless IoT Networks

**arXiv ID:** 2602.17619 | [PDF](https://arxiv.org/pdf/2602.17619v1)

**作者:** Jothi Prasanna Shanmuga Sundaram `[一作]` (University of California), Alberto E. Cerpa `[通讯]` (University of California)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

针对多跳无线 IoT 网络的数据分发协议进行改进，提出 Enhanced Dynamic Relay Point Protocol (EDRP)，该协议在原始 DRP 的基础上加入链路质量感知 CSMA（LQ‑CSMA）和无反馈的机器学习块大小选择（ML‑BSS）机制，并在真实环境中进行现场评估。

**💡 创新点**

① 将链路质量信息动态映射到 CSMA 的退避窗口，形成 LQ‑CSMA；② 将块大小选择建模为有序回归问题，采用 TAO‑优化的倾斜决策树实现无反馈的 ML‑BSS；③ 在理论层面对链路波动导致的碰撞概率和块大小收益进行正式分析，指导设计。

**🔧 技术方法**

使用 Rateless 编码、CSMA、分布式延迟计时器、链路质量估计、TDMA/CI 理论框架、Ordinal Regression（TAO‑CART/TAO‑Oblique）、TinyOS/RPL、TelosB 芯片、RPL Rank、PDR、RNP 等链路质量指标。

**📊 数据集**

基于现场实验收集的约 90,000 条记录（节点 15，传输 1.7592 × 10⁶ bytes），涵盖不同链路质量状态；数据来自真实的 TelosB‑RPL 网格网络。

**📈 对比分析**

在同一硬件、拓扑和实验设置下，与 MNP、Rateless Deluge、AdapCode 以及原始 DRP 进行 CDF‑Goodput、完成时间对比。EDRP 在 Goodput 上平均提升 39.43%（比 DRP 仅 13%），完成时间最短（6.67 s），比最慢协议 MNP 加速约 1.85×，比平均其他协议提升 1.39×。

**⚠️ 局限性**

① LQ‑CSMA 仍无法在所有情形下保证严格的高质量链路优先传输，碰撞仍会出现；② ML‑BSS 依赖现场数据训练，环境变化可能导致模型泛化下降；③ 未详细评估能耗、时延波动以及在不同频段或更大规模网络中的可扩展性。

---

## 384. Exploring Novel Data Storage Approaches for Large-Scale Numerical Weather Prediction

**arXiv ID:** 2602.17610 | [PDF](https://arxiv.org/pdf/2602.17610v1)

**作者:** Nicolau Manubens Gil `[一作]` `[通讯]`, Nicolau Manubens Gil

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文研究了知识产权的保护机制，探讨了不同国家在知识产权方面的法律框架和实施效果。

**💡 创新点**

创新点在于提出了一种综合评估知识产权保护效果的新方法，并通过案例分析验证了该方法的有效性。

**🔧 技术方法**

使用了定量分析和定性分析相结合的研究技术，包括法律文本分析和实证研究。

**📊 数据集**

使用了多个国家的知识产权法律文本和相关的经济数据集。

**📈 对比分析**

与传统的知识产权评估方法相比，该方法在准确性和适用性上表现更优，能够更全面地反映知识产权保护的实际效果。

**⚠️ 局限性**

限制在于数据的可获得性和不同国家法律体系的复杂性，可能影响评估结果的普适性。

---

## 385. Adapting Actively on the Fly: Relevance-Guided Online Meta-Learning with Latent Concepts for Geospatial Discovery

**arXiv ID:** 2602.17605 | [PDF](https://arxiv.org/pdf/2602.17605v1)

**作者:** Jowaria Khan `[一作]` (University of Michigan), Elizabeth Bondi-Kelly `[通讯]` (University of Michigan)

**通讯引用:** 437 | [OpenAlex ID](https://openalex.org/A5055892489)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e0540dec-d77f-42db-94ae-d039248f6393` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在极低采样预算下的地理空间目标发现框架，结合主动学习、在线元学习和概念驱动的相关性编码。

**💡 创新点**

创新点在于基于条件变分自编码器的概念相关性编码器和基于不确定性与概念差异的实时元批次构造，以及利用相关性加权的不确定性采样策略。

**🔧 技术方法**

使用 CVAE、在线元学习、主动学习采样、Gram-Schmidt 正交化概念编码、聚类构造元批、像素级焦点损失等技术。

**📊 数据集**

在美国 PFAS（石化物质）污染热点数据和稀疏的土地覆盖数据上进行验证。

**📈 对比分析**

与贪婪、UCB、主动学习、Prithvi、AML、OML 等基线比较，PFAS 任务下成功率（SR）和 F1 等指标均优于基线，尤其在稀疏、跨时空分布下保持稳定。

**⚠️ 局限性**

依赖先验的环境驱动概念，若任务缺少可解释的概念或分布极度变化时效果可能下降；同时在高频次采样或大规模高分辨率图像上计算开销仍较大。

---

## 386. MolHIT: Advancing Molecular-Graph Generation with Hierarchical Discrete Diffusion Models

**arXiv ID:** 2602.17602 | [PDF](https://arxiv.org/pdf/2602.17602v1)

**作者:** Hojung Jung `[一作]`, Dae-Woong Jeong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 MolHIT 模型，利用层级离散扩散与解耦原子编码实现分子图生成，首次在图模型上直接生成带电荷或 nH 的原子。

**💡 创新点**

创新点在于引入 Hierarchical Discrete Diffusion Model (HDDM) 以及 Decoupled Atom Encoding (DAE)，并结合 Project‑and‑Noise 采样和温度/Top‑p 采样方法提升多属性生成与多样性。

**🔧 技术方法**

核心技术包括离散扩散过程（HDDM）、图 Transformer 变体、CFG 条件引导、温度采样与 Project‑and‑Noise 采样策略。

**📊 数据集**

实验使用 MOSES 与 GuacaMol 两大公开分子数据集，并在下游任务（多属性引导生成、骨架扩展）上进行验证。

**📈 对比分析**

与 DiGress、DisCo、Cometh、DeFoG 等 2D 基线以及 SAFE‑GPT、GenMol 等 1D 基线对比，MOSES 上有效率 99.1%、质量 94.2%、骨架新颖度 0.39，显著超越 1D 模型；在 GuacaMol 上多属性预测 MAE 下降 52% 且保持 95% 以上有效率。

**⚠️ 局限性**

限制在于对更大离散词表的建模仍有挑战，DAE 在高维原子类型下 FCD 略低，且对极大分子结构生成的覆盖度仍需进一步提升。

---

## 387. Graph Neural Model Predictive Control for High-Dimensional Systems

**arXiv ID:** 2602.17601 | [PDF](https://arxiv.org/pdf/2602.17601v1)

**作者:** Patrick Benito Eberhard `[一作]` (Institute for Dynamic Systems and Control ETH Zürich), Marco Pavone `[通讯]` (Stanford University)

**通讯引用:** 11299 | [OpenAlex ID](https://openalex.org/A5050003000)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了将图神经网络嵌入模型预测控制的框架，实现高维软体机器人实时闭环控制

**💡 创新点**

创新点在于利用GNN的稀疏结构进行局部线性化并开发线性复杂度的condensing算法，使控制问题规模仅随节点数线性增长，GPU并行实现100 Hz控制

**🔧 技术方法**

采用Interaction‑Network风格的图神经网络学习动力学，自动微分得到线性化模型，结合GPU并行condensing和IPM求解QP

**📊 数据集**

训练数据来源于MuJoCo仿真和真实软体机器人测得的50条20 s随机漫步轨迹

**📈 对比分析**

与Koopman、SSM、MLP基线比较；在仿真中开放循环误差相当或更优，硬件上闭环跟踪误差比基线低63.6%，QP求解时间仅9.1 ms，满足100 Hz实时控制

**⚠️ 局限性**

局限在于对邻域大小有限的假设、对训练数据的敏感性、未给出闭环稳定性证明以及对时变图结构的适应性有限

---

## 388. Asymptotic Smoothing of the Lipschitz Loss Landscape in Overparameterized One-Hidden-Layer ReLU Networks

**arXiv ID:** 2602.17596 | [PDF](https://arxiv.org/pdf/2602.17596v1)

**作者:** Saveliy Baturin `[一作]` `[通讯]` (Independent Researcher), Saveliy Baturin (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究过拟合宽网络中单隐藏层ReLU网络的损失地形，证明子水平集近似连通并给出能量壁垒随宽度缩小的上界。

**💡 创新点**

将子水平集连通性结果从二次损失推广到任意凸 L‑Lipschitz 损失，并定量证明能量壁垒随宽度趋于零。

**🔧 技术方法**

利用稀疏可压缩性、ℓ1 正则化、ε‑网覆盖分析以及 Dynamic String Sampling 实验，构造连续路径并评估能量间隙。

**📊 数据集**

使用合成 Moons 回归数据集和 Wisconsin 乳腺癌分类数据集进行实验。

**📈 对比分析**

通过 DSS 计算配对能量间隙，宽网络的均值、最大间隙显著下降，Permutation 检验 p_perm=0，表明壁垒高度降低。

**⚠️ 局限性**

仅适用于单隐藏层 ReLU 网络，且假设损失凸且 Lipschitz 并带 ℓ1 正则，深层网络和无正则情况仍需进一步研究。

---

## 389. ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment

**arXiv ID:** 2602.17560 | [PDF](https://arxiv.org/pdf/2602.17560v1)

**作者:** Hongjue Zhao `[一作]` (University of Illinois Urbana-Champaign), Huajie Shao `[通讯]` (William and Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于常微分方程（ODE）的统一激活引导框架，并在此基础上设计了ODesteer方法，通过障碍函数驱动多步自适应激活更新；

**💡 创新点**

创新点在于将传统激活加法视为ODE的Euler离散；把激活引导方向等价于定义控制理论中的障碍函数；并利用非线性密度比的障碍函数通过ODE求解实现多步、动态的激活 steering；

**🔧 技术方法**

采用ODE求解、梯度驱动、Log‑density ratio 估计、Polynomial Count Sketch 生成非线性特征、Logistic 回归求解密度比；

**📊 数据集**

使用TruthfulQA、UltraFeedback、RealToxicityPrompts三大对齐基准数据集；

**📈 对比分析**

与RepE、ITI、CAA、MiMiC、HPR、RE‑Control、Linear‑AcT、TruthFlow等现有方法对比，ODesteer 在TruthfulQA提升5.7%，UltraFeedback提升2.5%，RealToxicityPrompts提升2.4%，同时保持生成质量；

**⚠️ 局限性**

局限性在于未把无监督特征学习（如稀疏自编码器）纳入框架，对某些场景的适用性和参数调优仍有待进一步探究。

---

## 390. MASPO: Unifying Gradient Utilization, Probability Mass, and Signal Reliability for Robust and Sample-Efficient LLM Reasoning

**arXiv ID:** 2602.17550 | [PDF](https://arxiv.org/pdf/2602.17550v1)

**作者:** Xiaoliang Fu `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种新的 RLVR 优化框架 MASPO，解决 LLM 训练中的梯度利用不足、概率质量不敏感以及信号可靠性不对称三大问题。

**💡 创新点**

通过软高斯门控、质量自适应限幅器和不对称风险控制三种机制统一信赖区域，实现连续可微的优化并显著提升推理性能。

**🔧 技术方法**

采用可微软高斯门控、基于概率的自适应限幅、优势重加权、不对称风险控制等技术，并在 GRPO 基础上进行改进。

**📊 数据集**

在 DAPO‑Math‑17K 进行微调，评估时使用 AIME24/25、AMC23、MATH500、Minerva、OlympiadBench 等数学推理数据集。

**📈 对比分析**

与 GRPO、Clip Higher、DAC、SAPO、BAPO 等基线对比，MASPO 在 1.5B、7B、14B 模型上平均提升 2‑3% 的 Avg@32 与 Pass@32，表现最优。

**⚠️ 局限性**

依赖可验证奖励的假设，可能不适用于主观奖励任务；在 70B+ 或 MoE 等极大规模模型上尚未验证，计算资源受限。

---

## 391. Learning to Stay Safe: Adaptive Regularization Against Safety Degradation during Fine-Tuning

**arXiv ID:** 2602.17546 | [PDF](https://arxiv.org/pdf/2602.17546v1)

**作者:** Jyotin Goel `[一作]` (Indian Institute of Technology Jodhpur), Pratik Mazumder `[通讯]` (Indian Institute of Technology Jodhpur)

**通讯引用:** 154 | [OpenAlex ID](https://openalex.org/A5054829553)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自适应正则化框架，利用训练时的安全风险信号动态调节KL正则化，从而在微调过程中防止模型安全性退化。

**💡 创新点**

创新点在于：①将安全风险信号（激活层线性预测或外部判定器评分）嵌入优化目标，实现动态损失权重；②提供两种安全评估方式，兼顾轻量化与语义丰富性；③将正则化视为软信任域方法，允许在低风险样本上充分学习，高风险样本上保持与基准策略的接近。

**🔧 技术方法**

采用的技术包括：预生成激活的线性探针预测、外部LLM（gpt‑oss‑20b）判定器评估、动态损失权重（α_t、β_t）调度、KL正则化、实验中的SFT与DPO等对比训练策略。

**📊 数据集**

使用的数据集主要有 HEx‑PHI（恶意/对抗样本）、Alpaca（正常任务数据）以及 GSM8K、LoRA/全参数微调实验等，构建混合恶意/正常比例的数据集进行评测。

**📈 对比分析**

与传统SFT、固定正则化（Constrained SFT）等方法对比；实验显示在5种模型上，攻击成功率（ASR）从≈96%降至≈1–9%，同时任务评测指标（如Alpaca Win Rate）基本保持或略有提升；对学习率、模型规模等超参鲁棒性良好。

**⚠️ 局限性**

局限性：①判定器方案在推理时会增加额外计算成本；②对极端对抗或高风险样本的识别仍有误判风险；③实验仅在本地开源模型环境完成，真实世界部署中对策与威胁模型的适用性尚待验证。

---

## 392. Provably Explaining Neural Additive Models

**arXiv ID:** 2602.17530 | [PDF](https://arxiv.org/pdf/2602.17530v1)

**作者:** Shahaf Bassan `[一作]` (Hebrew University of Jerusalem), Guy Katz `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 2490 | [OpenAlex ID](https://openalex.org/A5102986148)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种针对神经加性模型(NAM)的高效算法，可在对模型进行后置解释时，生成可证明的卡氏最小（cardinality‑minimal）足够解释子集，极大降低验证查询次数；

**💡 创新点**

创新点在于利用NAM的可加结构先并行计算每个一元子网络的重要性区间，再对排序后的特征进行二分搜索，从而把传统上指数级的验证查询降为对数级，并在单个特征子网络上完成验证；

**🔧 技术方法**

核心技术包括：神经网络可验证（verification）工具（如α‑β‑CROWN）、并行区间逼近、特征重要性排序与二分搜索；

**📊 数据集**

实验使用四个常用表格数据集：Breast Cancer、CREDIT、FICO HELOC，以及其他公开数据集；

**📈 对比分析**

与传统子集最小解释算法（子集最小化、逆敏感性排序等）相比，本文方法在解释子集大小和运行时间上均显著更优（平均解释子集约为原方法的1/4~1/5，计算时间降低约90%+），且实现了完全可证明的足够性；

**⚠️ 局限性**

局限性在于仍需依赖完整的神经网络验证器，且在极端情况下区间分辨率低时可能导致更多查询；此外，方法仅针对NAM而非更通用的神经网络，尚未验证在更大规模模型上的可扩展性。

---

## 393. RA-Nav: A Risk-Aware Navigation System Based on Semantic Segmentation for Aerial Robots in Unpredictable Environments

**arXiv ID:** 2602.17515 | [PDF](https://arxiv.org/pdf/2602.17515v1)

**作者:** Ziyi Zong `[一作]`, Zhan Tu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了RA-Nav框架，利用轻量级多尺度语义分割网络生成实时语义风险网格，并在此基础上设计风险感知路径搜索（R‑A*）和梯度优化B‑spline轨迹，以实现对突然移动障碍的预警与安全避障。

**💡 创新点**

创新点在于：①将语义分割结果与三类障碍物（静止、可动、潜在可动）对应的风险模型相结合，构造可叠加的高斯风险场与速度相关的动态风险场；②引入方向因子Δ调节R‑A*中的梯度引导，避免局部最小；③在前端采用风险梯度引导搜索，后端进行全局轨迹优化，实现从感知到控制的闭环风险意识。

**🔧 技术方法**

使用技术包括：轻量级多尺度语义分割网络LMSCNet；高斯风险场与速度衰减式动态风险模型；R‑A*风险感知A*路径搜索；基于B‑spline的梯度优化轨迹；点云聚类、光流/ICP估计动态障碍速度；ROS/Gazebo仿真与MID360激光雷达点云处理。

**📊 数据集**

数据集与环境：SemanticKITTI用于LMSCNet训练与评估；Gazebo+ROS1构建的城市模拟环境（包括静态建筑、行人、车辆等）用于路径搜索与动态规划对比；MID360激光雷达真实点云用于实测验证。

**📈 对比分析**

对比方法：传统A*、D*、Ego‑Planner；在随机50×50地图和动态场景中，R‑A*路径长度仅比传统A*多3.9%至5%，但最小安全距离从1提升至≈2.8；在两套仿真场景中，RA‑Nav成功率100%（仅R‑A* 70%/75%，Ego‑Planner 20%/25%），规划时间略增0.5–0.7 s。实测中，RA‑Nav在被遮挡人出现时仍能安全避障。

**⚠️ 局限性**

局限性：①依赖语义分割的准确性，分割误差会直接影响风险评估；②在极稠密或快速变化场景中，高斯风险叠加与方向因子可能不足以捕捉细粒度风险；③相较传统A*与D*，计算量略大，实时性仍受硬件限制；④未讨论多机器人协同或全局规划层面的集成。

---

## 394. Bridging the Domain Divide: Supervised vs. Zero-Shot Clinical Section Segmentation from MIMIC-III to Obstetrics

**arXiv ID:** 2602.17513 | [PDF](https://arxiv.org/pdf/2602.17513v1)

**作者:** Baris Karacan `[一作]`, Patrick Thornton `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对产科子域的临床记录，本文提出并评估了监督Transformer模型与零样本LLM在段落分割上的表现。

**💡 创新点**

创新点在于构建了产科笔记集合ONC作为真实基准，系统比较了传统监督方法与指令调优LLM，并研究了幻觉纠正策略。

**🔧 技术方法**

采用Transformer分类、Transformer+CRF模型以及四款指令调优LLM（Mistral‑7B、Llama 3.1‑8B、Qwen‑2.5‑32B、Llama 3.3‑70B）。

**📊 数据集**

使用公开MedSecId数据集（2,002份笔记）进行训练与验证，并用新收集的100份产科H&P笔记（ONC）进行跨域评估。

**📈 对比分析**

通过宏/加权F1、幻觉率与错误分析比较，监督模型在MedSecId上占优，但在ONC上零样本LLM（尤其是Llama 3.3‑70B）在宏F1上超越所有监督基线，幻觉纠正后提升约9%。

**⚠️ 局限性**

受限于ONC规模有限、标签缺乏统一标准、LLM仍易产生幻觉与遗漏，未来需扩大数据、多样化标签并融合领域知识提升鲁棒性。

---

## 395. Guarding the Middle: Protecting Intermediate Representations in Federated Split Learning

**arXiv ID:** 2602.17614 | [PDF](https://arxiv.org/pdf/2602.17614v1)

**作者:** Obaidullah Zaland `[一作]` (Umeå University), Monowar Bhuyan `[通讯]` (Umeå University)

**通讯引用:** 3258 | [OpenAlex ID](https://openalex.org/A5044933320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在U形联邦分层学习（UFSL）框架下提出KD-UFSL，通过在数据层加噪声、在模型层实施k-匿名微聚合，降低中间表示泄露风险。

**💡 创新点**

创新点在于：①将差分隐私与k-匿名相结合，形成双层隐私防护；②噪声加入原始数据而非中间特征；③在分层学习中首次实现模型级k-匿名，提升隐私与实用性的平衡。

**🔧 技术方法**

使用技术包括：联邦分层学习、FedAvg聚合、Gaussian差分隐私机制、k-匿名微聚合、逆向网络重构攻击、ResNet/ConvNet模型拆分。

**📊 数据集**

实验数据集：CIFAR-10、EMNIST、FashionMNIST、SVHN。

**📈 对比分析**

通过与传统UFSL、UFSL+DP、UFSL+KA对比，使用MSE与SSIM评估重构误差；KD-UFSL在大多数数据集上MSE提升50%+、SSIM降低40%，全局模型准确率仅下降约2‑2.5%。

**⚠️ 局限性**

局限性：对深层网络（如ResNet50）表现不一定最优；k与噪声方差参数敏感；提升头部网络深度虽然提高隐私但会显著增加客户端计算负担；对非图像或更大规模异构数据的适用性仍待验证。

---

## 396. Towards Anytime-Valid Statistical Watermarking

**arXiv ID:** 2602.17608 | [PDF](https://arxiv.org/pdf/2602.17608v1)

**作者:** Baihe Huang `[一作]` (University of California), Michael I. Jordan `[通讯]` (University of California)

**通讯引用:** 177709 | [OpenAlex ID](https://openalex.org/A5049812527)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Anchored E-Watermarking 框架，利用 e‑value 对生成文本进行可随时停止的统计水印检测。

**💡 创新点**

创新点在于：①首次将 e‑value 引入统计水印，解决了传统 p‑value 固定窗口的 p‑hacking 问题；②给出最优 e‑value 的闭式解与最优停止时间；③证明信息理论下的样本复杂度下界与上界，实现理论与实践的闭环。

**🔧 技术方法**

使用的技术包括：e‑value 与超马丁格尔构造、稳健对数最优性（Kelly准则）、极大耦合优化、信息论极限证明以及实验验证。

**📊 数据集**

实验采用 Llama2‑7B‑chat 作为目标模型，Phi‑3‑mini‑128k‑instruct 作为 anchor；使用 MarkMyWords 基准生成 300 条文本，分别在字符、词、文本级攻击下进行检测。

**📈 对比分析**

与五种基线（Distribution Shift、Exponential、Binary、Inverse Transform、SEAL）在质量与检测长度两项指标上对比，结果显示本方法质量 0.919 与 SEAL 竞争，检测长度 72 tokens，显著低于 SEAL 的 84.5，且在大多数攻击下检测成功率提升。

**⚠️ 局限性**

局限性：仅在 anchor 与目标分布相近的场景下适用；未深入探讨对激励型对手的博弈分析；anchor 选取和参数 δ 的敏感性仍需进一步研究。

---

## 397. The Cascade Equivalence Hypothesis: When Do Speech LLMs Behave Like ASR$\rightarrow$LLM Pipelines?

**arXiv ID:** 2602.17598 | [PDF](https://arxiv.org/pdf/2602.17598v1)

**作者:** Jayadev Billa `[一作]` `[通讯]`, Jayadev Billa

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过匹配后端模型的方式，系统评估并对比了多种端到端语音大型语言模型与传统ASR+LLM级联的行为与机制；

**💡 创新点**

创新点在于首次提出匹配后端模型的行为测试方法，并将其与基于文本的级联对比，揭示了端到端模型在文本充足任务上往往仅为隐式级联；

**🔧 技术方法**

主要技术包括Cohen's κ一致性度量、条件错误重叠、logit lens、线性探测器、LEACE概念消除等多种机制分析手段；

**📊 数据集**

使用了六类任务数据集，包含TTS合成的TriviaQA、AG News、SST‑2、CommonsenseQA，以及自然语音的MELD情感识别和MUStARD嘲讽检测；

**📈 对比分析**

比较方法通过匹配后端模型的级联与原始端到端模型进行逐例一致性和错误重叠评估，结果显示如Ultravox与其级联在文本充足任务上κ≈0.93，而Qwen2‑Audio在多任务上仅为0.54–0.85，表明存在显著差异；

**⚠️ 局限性**

局限在于仅对两种开源模型进行了机制分析，未覆盖所有主流端到端架构；评估主要集中在语音合成文本充足任务，可能低估真实语音的声学信息优势；

---

## 398. Conditional Flow Matching for Continuous Anomaly Detection in Autonomous Driving on a Manifold-Aware Spectral Space

**arXiv ID:** 2602.17586 | [PDF](https://arxiv.org/pdf/2602.17586v1)

**作者:** Antonio Guillen-Perez `[一作]` `[通讯]` (Independent Researcher), Antonio Guillen-Perez (Independent Researcher)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出一种无监督的生成模型 Deep‑Flow，用来检测 Level 4 自动驾驶系统中的罕见、极具安全风险的异常驾驶行为。

**💡 创新点**

创新点包括：① 在低秩谱流瓶颈下构建连续概率密度，使轨迹保持 kinematic 平滑；② 采用 Optimal Transport 条件流匹配（OT‑CFM）实现一维 ODE 训练，避免了扩散模型的随机方程；③ 通过直接跳连的 lane‑aware 目标条件编码解决多模态决策；④ 引入基于路径弯曲度和 jerk 的运动复杂度加权，提升对高能量稀有行为的学习。

**🔧 技术方法**

技术栈包括：Optimal Transport Conditional Flow Matching、Continuous Normalizing Flow、PCA 低秩谱流瓶颈、Early‑Fusion Transformer 编码器、精确计算 Jacobian trace 的 RK4 ODE 求解。

**📊 数据集**

使用 Waymo Open Motion Dataset（WOMD）进行训练（250k 轨迹）和验证（8.8k 轨迹），并在此数据上构建“黄金测试集”进行安全事件评估。

**📈 对比分析**

与随机猜测和传统硬制动阈值（AUC‑ROC ≈ 0.682）对比，Deep‑Flow 在黄金测试集上取得 AUC‑ROC 0.766，显著提升了对罕见安全事件的检测能力。

**⚠️ 局限性**

局限性主要体现在：① 线性 PCA 瓶颈在极高曲率或复杂交叉路口时缺乏足够的几何表达；② 目前未显式建模社交碰撞约束，可能导致对多车交互异常的检测不足。

---

## 399. Canonicalizing Multimodal Contrastive Representation Learning

**arXiv ID:** 2602.17584 | [PDF](https://arxiv.org/pdf/2602.17584v1)

**作者:** Sharut Gupta `[一作]` (Massachusetts Institute of Technology), Vikas Garg `[通讯]` (Aalto University)

**通讯引用:** 1445 | [OpenAlex ID](https://openalex.org/A5065774663)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了不同训练的多模态对比模型之间是否存在统一的几何映射，发现几乎可以用一个正交变换对齐图像和文本嵌入空间。

**💡 创新点**

证明在多模态核相似性一致的情况下，两个模型的图像和文本嵌入只能通过同一正交变换相关联，并提供理论证明与稳定性界限；并验证仅用少量单模锚点即可学习该变换。

**🔧 技术方法**

采用正交Procrustes求解、余弦相似度评估、对齐方法，并结合多模态核一致性理论与正交变换的稳定性分析。

**📊 数据集**

在 Oxford-IIIT Pets、CIFAR-100、Caltech-101、STL10、DTD 等公开数据集上对 CLIP、SigLIP、FLAVA 等模型进行实验。

**📈 对比分析**

通过图像-图像、文本-文本余弦相似度、跨模型检索准确率以及零样本分类指标进行评估，发现正交变换显著提升文本相似度并保持图像分类性能，几乎达到原模型性能。

**⚠️ 局限性**

评估主要集中在分类级别语义，未充分验证对细粒度检索或其他模态的适用性，且对细粒度属性的可解码性尚未探究。

---

## 400. Simultaneous Blackwell Approachability and Applications to Multiclass Omniprediction

**arXiv ID:** 2602.17577 | [PDF](https://arxiv.org/pdf/2602.17577v1)

**作者:** Lunjia Hu `[一作]` (Northeastern University), Chutong Yang `[通讯]` (University of Texas at Austin)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5085509893)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新颖的多类别全预测（omniprediction）框架，利用同时实现Blackwell逼近（simultaneous Blackwell approachability）和混合线性优化或上下文混合线性优化（MLOO/CMLOO）来构建在无限对照类（comparator）族下的预测器；

**💡 创新点**

创新点在于：1）将单类别全预测方法推广到多类别场景，并提供对无限对照类族的理论支持；2）提出一种同时逼近多目标的Blackwell框架，并给出可行的MLOO构造；3）实现了样本复杂度≈O((k+1)^{1/ε})（或在线学习中的时间复杂度同阶），显著优于之前仅适用于有限对照类族的结果；

**🔧 技术方法**

主要技术包括：对Blackwell逼近的理论推广、构造混合线性优化/上下文混合线性优化、投影梯度下降、乘子法、在线学习中的乘法权重算法、Rademacher复杂度分析、KL散度与指数权重更新；

**📊 数据集**

本文为理论工作，未使用实际数据集；所有结果均基于统计与在线学习的理论设定；

**📈 对比分析**

与以往仅适用于二分类或有限对照类族的全预测方法相比，本文在多类别、无限对照类族下实现了近似最优（ε-omnipredictor）且样本复杂度仅多项式于(k+1)^{1/ε}，在线学习时间复杂度同阶；

**⚠️ 局限性**

限制包括：1）样本/时间复杂度随类别数k指数增长，适用于小常数k；2）对照类族仍需满足某些可满足性约束（如存在MLOO）；3）对特殊损失族（如不满足坐标分解）的支持仍有限；4）算法实现对高维问题的计算复杂度相对较高。

---

## 401. FR-GESTURE: An RGBD Dataset For Gesture-based Human-Robot Interaction In First Responder Operations

**arXiv ID:** 2602.17573 | [PDF](https://arxiv.org/pdf/2602.17573v1)

**作者:** Konstantinos Foteinos `[一作]` (Harokopio University of Athens), Georgios Th. Papadopoulos `[通讯]` (Harokopio University of Athens)

**通讯引用:** 1366 | [OpenAlex ID](https://openalex.org/A5065554248)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了用于消防员手势控制无人车的 RGBD 数据集 FR-GESTURE，包含 12 个手势命令，并在两种评估协议下给出了基线分类性能。

**💡 创新点**

首次针对一线救援人员设计手势集并公开数据，覆盖多距离、多视角与室内外环境。

**🔧 技术方法**

采用 ResNet、ResNeXt、EfficientNet 等 CNN 结构，并在 HaGRID 预训练后 fine‑tune。

**📊 数据集**

使用自收集的 FR-GESTURE 数据集（3312 张 RGBD 图像）以及公开的 HaGRID 预训练模型。

**📈 对比分析**

在 uniform 协议下 EfficientNet-B0 达到约 96.4% F1 分数，subject‑independent 协议下约 87.7%；与 ResNet/ResNeXt 对比显示其在小样本上更稳健。

**⚠️ 局限性**

样本量有限、仅大学校园环境、受试者少且性别/种族单一，导致泛化能力不足。

---

## 402. Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability

**arXiv ID:** 2602.17544 | [PDF](https://arxiv.org/pdf/2602.17544v1)

**作者:** Shashank Aggarwal `[一作]` (Indian Institute of Technology), Amit Awekar `[通讯]` (Indian Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多代理信息检索流水线中，提出了将链式思维（CoT）与执行分离的 Thinker‑Executor 框架，并引入可重用性和可验证性两种新指标来评估 CoT 质量。

**💡 创新点**

创新点在于：①从目标任务准确率切入，转而关注思维链本身的可转移性和一致性；②通过“帮忙/干扰”实验量化可重用性；③通过一致性检查量化可验证性；④发现传统准确率与这两项指标低相关，暴露了当前排行榜的盲点。

**🔧 技术方法**

采用大型语言模型（Gemma3‑27B、Llama3.1‑8B、DeepSeek‑R1‑14B、Phi4‑Reasoning‑14B）作为 Thinker，10 种 360M‑3B 参数模型作为 Executor，利用 Ollama API 进行推理；计算帮助/干扰比例得到可重用性，计算答案一致性得到可验证性；使用 Kendall τ 评估不同委员会间排名一致性。

**📊 数据集**

使用五个常见推理基准：GSM8K、SVAMP、ARC‑Challenge、StrategyQA、CommonsenseQA；对每个 Thinker‑Executor 组合在这五个数据集上进行评估。

**📈 对比分析**

对比方法：将可重用性、可验证性与传统准确率对齐；结果显示：某些 Thinker 在准确率上领先，但其可重用性/可验证性并不高；一般模型 Gemma、Llama 在可重用性/可验证性上有时优于专门的推理模型；不同 Executor 委员会（Strong、Full、Weak）对得分尺度影响大，但相对排名保持稳定。

**⚠️ 局限性**

局限性：①评估仅在公开基准上完成，未检验在真实多代理 IR 系统中的表现；②“损坏”CoT 的生成依赖人工设定，可能不完全覆盖所有误导情况；③分数高度依赖 Executor 委员会的构成，弱委员会的低分可能导致误判；④未探索将可重用性/可验证性直接作为训练目标的效果。

---

## 403. HAP Networks for the Future: Applications in Sensing, Computing, and Communication

**arXiv ID:** 2602.17534 | [PDF](https://arxiv.org/pdf/2602.17534v1)

**作者:** Sultan Çoğay `[一作]`, Gökhan Seçinti `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

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

## 404. Enhancing Large Language Models (LLMs) for Telecom using Dynamic Knowledge Graphs and Explainable Retrieval-Augmented Generation

**arXiv ID:** 2602.17529 | [PDF](https://arxiv.org/pdf/2602.17529v1)

**作者:** Dun Yuan `[一作]`, Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 KG‑RAG 框架，利用知识图和检索增强生成来提升大型语言模型在电信领域的答案准确性、可解释性和实时更新能力。

**💡 创新点**

创新点在于：①将结构化知识图作为检索前端，用三元组代替长文本；②实现动态 KG 更新，使模型能够反映最新网络状态；③通过可解释的三元组提示让 LLM 的输出可追溯、标准合规。

**🔧 技术方法**

采用的技术包括：预训练 LLM 进行实体与关系抽取；基于 TransE/ConvE 等的 KG 链接预测；双编码器密集检索与词法+语义融合；RAG 结合结构化提示；动态 KG 更新与可解释输出生成。

**📊 数据集**

使用的数据集包括：SPEC5G、Tspec‑LLM、TeleQnA、ORAN‑Bench‑13K 四大电信标准/技术文档集合。

**📈 对比分析**

通过与 LLM‑only、传统 RAG、self‑RAG、RAPTOR 等基线比较，KG‑RAG 在文本摘要（ROUGE‑1/2/L、BLEU、METEOR）和问答任务（准确率、幻觉率）上均领先；平均准确率提升约 14.3%‑21.6%，幻觉率显著下降，动态更新后查询准确率达到 84%，延迟控制在 1.1 s 以内。

**⚠️ 局限性**

局限性：仍依赖 LLM 的抽取质量，KG 构建需离线预处理；动态更新延迟有限，无法即时覆盖所有新标准或多模态信息；在极度复杂或跨域场景中的多跳推理效果尚需进一步验证。

---

## 405. The Anxiety of Influence: Bloom Filters in Transformer Attention Heads

**arXiv ID:** 2602.17526 | [PDF](https://arxiv.org/pdf/2602.17526v1)

**作者:** Peter Balogh `[一作]` `[通讯]`, Peter Balogh

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 GPT‑2、Pythia 等自回归语言模型中发现并验证了一类注意力头，它们实现了近似集合成员检验（Bloom filter）功能，能够判断某 token 是否已在上下文出现。

**💡 创新点**

首次将 Bloom filter 的理论容量曲线与 Transformer 头行为匹配，揭示出多分辨率的距离敏感成员检验子系统，并通过控制实验区分真实成员检验头与前缀注意头，形成新的注意力头功能类别。

**🔧 技术方法**

采用注意力头行为分析、容量曲线拟合、距离敏感性扫频、对照实验、归因消融等方法，并结合 QK 近似哈希分析。

**📊 数据集**

使用 GPT‑2 small/medium/large、Pythia‑160M 的自回归模型，以及 WikiText‑103 验证集和人工构造的重复/近义句子，进行统一实验。

**📈 对比分析**

通过对比 Bloom 过滤器头与非 Bloom 头在选择性、误报率、自然文本中的重现率和消融效果，Bloom 头在重复检测上显著优于其他头，消融实验表明它们既承担专门功能又对整体上下文处理有贡献。

**⚠️ 局限性**

局限在于仅评估到 708M 参数规模的模型，缺乏对更大模型的验证；仅基于行为而非 QK 权重的机制分析；消融方法对结果敏感；且实验聚焦于重复检验任务，未探究更广泛语言功能。

---

## 406. When Models Ignore Definitions: Measuring Semantic Override Hallucinations in LLM Reasoning

**arXiv ID:** 2602.17520 | [PDF](https://arxiv.org/pdf/2602.17520v1)

**作者:** Yogeswar Reddy Thota `[一作]` (University of Texas at Dallas), Tooraj Nikoubin `[通讯]` (University of Texas at Dallas)

**通讯引用:** 335 | [OpenAlex ID](https://openalex.org/A5051431139)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型在本地重新定义语义下的推理失败，提出语义覆盖和假设注入错误类型，并构建了30题微基准测试。

**💡 创新点**

系统识别并量化“语义覆盖”与“假设注入”两类本地语义不符错误，并提供验证式评分和错误分类，首次评估LLM在电路逻辑语义覆盖的鲁棒性。

**🔧 技术方法**

采用对话式验证器评估框架，对比三大前沿LLM（Claude、ChatGPT、Gemini）在无工具、无外部资源的单轮提示下的回答，结合手工标注的错误类型。

**📊 数据集**

设计了包含五类陷阱（定义覆盖、硬件语义歧义、欠定、矛盾处理、任务误读）的30道数字逻辑与电路推理题目，构成自定义微基准集。

**📈 对比分析**

通过验证器正确率和错误类型分布对三模型进行对比，整体准确率在80–90%之间，定义覆盖和假设注入错误率最高，表明模型在本地语义遵循上仍有显著不足。

**⚠️ 局限性**

仅评估静态提示下的单轮推理，未考虑多轮对话或外部工具的影响，基准规模有限，可能不覆盖所有真实硬件规范场景。

---

## 407. Dodging the Moose: Experimental Insights in Real-Life Automated Collision Avoidance

**arXiv ID:** 2602.17512 | [PDF](https://arxiv.org/pdf/2602.17512v1)

**作者:** Leila Gharavi `[一作]` (Delft University of Technology), Hiroshi Fujimoto `[通讯]` (University of Tokyo)

**通讯引用:** 9359 | [OpenAlex ID](https://openalex.org/A5013219761)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了在突发静态障碍物出现时，如何实现自动驾驶车辆的实时碰撞规避；

**💡 创新点**

创新点在于提出一种人类启发的前馈最大转向规划器（MSF）与传统MPC相结合的双重规划框架，以缓解非线性MPC在紧急场景下求解不可行或收敛慢的问题；

**🔧 技术方法**

采用单轨车辆模型、Pacejka轮胎模型、非线性模型预测控制（MPC）和基于最大转向的前馈规划（MSF）以及反馈线性化控制（FBL），实现闭环控制；

**📊 数据集**

实验数据来源于FPEV2-Kanon电动车在不同速度（5、6、7 m/s）和不同时间-距离、宽度参数下的现场测试，并通过YouTube视频展示实验过程；

**📈 对比分析**

与纯MPC方案比较时，组合规划器在极限速度和较短时间窗口下仍能生成可行的转向指令，避免了MPC过度激进的转向，整体碰撞规避性能更稳定；

**⚠️ 局限性**

局限性包括：在极端情况（如τ=1.4 s、ν≈0.95）MPC仍无法找到可行解，系统仍受限于转向极限；计算资源有限导致前馈规划也难以进一步加速；缺乏对摩擦系数等不确定性实时建模的支持。

---

## 408. Modeling Distinct Human Interaction in Web Agents

**arXiv ID:** 2602.17588 | [PDF](https://arxiv.org/pdf/2602.17588v1)

**作者:** Faria Huq `[一作]` (Carnegie Mellon University), Jeffrey P. Bigham `[通讯]` (Carnegie Mellon University)

**通讯引用:** 11140 | [OpenAlex ID](https://openalex.org/A5082603621)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了400条真实用户与代理协作的网页浏览轨迹数据集，并训练与评估了人类干预预测模型。

**💡 创新点**

首次将人类干预行为刻画为多样化的协作风格，提出基于多模态 LLM 的逐步干预预测框架。

**🔧 技术方法**

采用多模态输入（截图+可访问性树）结合序列化提示的 LMM，使用监督微调实现二分类预测。

**📊 数据集**

400条实时网页导航轨迹（约2,748个代理步骤与1,476个用户步骤），包含10个标准任务与10个自由任务。

**📈 对比分析**

与零样本专有 LLM、全自律/全确认基线和未微调模型对比，微调模型干预准确率提升61–63%，Perfect Timing Score升至0.303，用户满意度提升26.5%。

**⚠️ 局限性**

数据规模有限、样本稀疏导致某些协作风格的训练不足，仅适用于网页导航，对未见任务的泛化能力尚未验证。

---

## 409. A Hybrid Federated Learning Based Ensemble Approach for Lung Disease Diagnosis Leveraging Fusion of SWIN Transformer and CNN

**arXiv ID:** 2602.17566 | [PDF](https://arxiv.org/pdf/2602.17566v1)

**作者:** Asif Hasan Chowdhury `[一作]` (BRAC University), Md. Golam Rabiul Alam `[通讯]` (BRAC University)

**通讯引用:** 3617 | [OpenAlex ID](https://openalex.org/A5051981813)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种融合SWIN Transformer与CNN（VGG‑19、Inception‑V3、DenseNet‑201）并通过联邦学习实现分布式、隐私安全的肺部疾病（COVID‑19、肺炎、正常）诊断系统

**💡 创新点**

创新点在于将Transformer与多CNN模型进行加权/平均融合，形成混合模型，并在联邦学习框架下实现多医院模型协同训练，兼顾精度与数据隐私

**🔧 技术方法**

采用迁移学习（预训练CNN）、SWIN Transformer、加权/平均投票集成、联邦学习（聚合服务器与本地模型）、TensorFlow/Keras实现训练与推理

**📊 数据集**

使用公开的胸部X‑ray图像数据集，包括COVID‑19、肺炎与正常三类（共约六千多张），并进行了数据清洗、增强与划分为训练/测试集

**📈 对比分析**

通过比较单一CNN模型与融合模型在准确率、AUC‑ROC及混淆矩阵上的表现，融合模型（加权求和）在验证集上达97.0%准确率、AUC>0.98，显著优于单模型

**⚠️ 局限性**

存在过拟合风险、模型训练时显存占用高（单次联邦训练35 GB）、数据量有限且未充分验证跨机构泛化、缺乏概念漂移处理等限制

---

## 410. Revisiting Weight Regularization for Low-Rank Continual Learning

**arXiv ID:** 2602.17559 | [PDF](https://arxiv.org/pdf/2602.17559v1)

**作者:** Yaoyue Zheng `[一作]` (Xi'an Jiaotong University), Zhiqiang Tian `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 3307 | [OpenAlex ID](https://openalex.org/A5020301327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EWC-LoRA，一种结合 Elastic Weight Consolidation 与低秩适配的参数高效持续学习框架；

**💡 创新点**

创新点在于将权重正则化迁移至低秩更新空间，并使用完整维度 Fisher 矩阵对 LoRA 参数进行正则化，实现固定内存和推理成本；

**🔧 技术方法**

采用 Elastic Weight Consolidation、低秩适配 LoRA、全维度 Fisher 信息矩阵估计与正则化；

**📊 数据集**

在视觉任务上使用 CIFAR-100、DomainNet、ImageNet‑R、ImageNet‑A；在语言任务上使用 AG News、Amazon Reviews、Yelp Reviews、DBpedia、Yahoo Answers，且基于 ViT-B/16、T5‑large 与 LLaMA‑3.2‑1B‑Instruct；

**📈 对比分析**

与多种基线（L2P、DualPrompt、CODA‑Prompt、InfLoRA、SD‑LoRA、CL‑LoRA、BiLoRA、Vanilla LoRA、O‑LoRA、TreeLoRA）比较，EWC‑LoRA 在大多数数据集上实现了最高或第二高的平均准确率，并在稳定性‑可塑性平衡上优于其他低秩方法；

**⚠️ 局限性**

局限包括 Fisher 估计随着任务序列延长可能下降，对复杂任务或域增量设置的适应性有限，需要进一步改进 Fisher 估计和低秩参数分配策略。

---

## 411. RetouchIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward

**arXiv ID:** 2602.17558 | [PDF](https://arxiv.org/pdf/2602.17558v1)

**作者:** Qiucheng Wu `[一作]` (Adobe Research), Handong Zhao `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 RetouchIQ，一个基于多模态大型语言模型（MLLM）的指令驱动图像编辑代理，能够将用户的自然语言编辑意图转换为可执行的专业图像编辑工具参数，完成从语义理解到可操作的编辑步骤。

**💡 创新点**

创新点包括：①提出通用奖励模型（Generalist Reward Model, GRM），用 RL 微调的 MLLM 生成评估指标并给出标量奖励，突破传统像素级、参考图像对比的奖励局限；②设计了 Policy‑Guided Reward Training (PGRT) 训练方案，利用策略模型自身产生的弱编辑样本来对奖励模型进行自适应训练，解决分布偏移问题；③构建了 190k 条指令-理由-编辑序列的高质量数据集，并推出 RetouchEval 基准。

**🔧 技术方法**

核心技术包括：多模态大型语言模型（如基于 LLaMA/ChatGLM 等框架）、强化学习（Policy‑Gradient）、自生成评估指标与奖励的通用奖励模型、数据增强与逆向编辑扰动、以及专业图像编辑软件（Adobe Lightroom、PicsArt 等）的工具接口调用。

**📊 数据集**

使用的数据集：190k 条用户真实编辑轨迹（包含图像对、编辑意图与操作序列）用于 SFT；10k 对应的强弱编辑样本用于奖励模型 SFT；另外 5k 通过 PGRT 产生的政策模型弱样本用于奖励模型 RL；RetouchEval（300 条指令-图像对，分为质量提升、风格转换、局部编辑）用于评估；MIT‑Adobe5K 用于跨域验证。

**📈 对比分析**

与基线比较包括：通用 MLLM（GPT‑5、Gemini‑2.5）、MLLM 代理（MonetGPT、JarvisArt）以及开放源代码扩散模型（Flux‑Pro）。在 RetouchEval 上，RetouchIQ‑GRM 在 L1、L2、语义一致性（GLM‑4.5V）等指标上均取得最优或次优成绩，平均提升约 2–3% 以上；在 MIT‑Adobe5K 上也表现出与或超过 Diffusion 方案的 PSNR/LPIPS/SSIM。RL+GRM 相较 SFT 版进一步提升 1–2% 的感知质量。

**⚠️ 局限性**

局限性：①奖励模型仍依赖训练数据的质量与多样性，极端创意或少见风格的评估可能不足；②RL 训练过程耗时且对奖励模型的收敛性高度敏感，易出现策略退化；③当前仅支持有限的专业编辑软件接口，扩展到更多平台需重新封装；④在高度主观的艺术风格中，模型仍可能偏离个别用户偏好。

---

## 412. GraphThinker: Reinforcing Video Reasoning with Event Graph Thinking

**arXiv ID:** 2602.17555 | [PDF](https://arxiv.org/pdf/2602.17555v1)

**作者:** Zixu Cheng `[一作]` (Queen Mary University of London), Shaogang Gong `[通讯]` (Queen Mary University of London)

**通讯引用:** 35907 | [OpenAlex ID](https://openalex.org/A5039302902)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于事件级场景图（EVSG）和强化学习微调的多模态大型语言模型后训练方法，以实现视频推理中的因果关系建模与误报抑制。

**💡 创新点**

创新点在于：①利用MLLM自生成多粒度稠密字幕构建无人工标注的事件级场景图；②将EVSG作为中间推理结构与视觉注意力奖励结合在GRPO框架内，显式约束时序与视觉证据；③在奖励函数中引入格式一致性与视觉注意力奖励，提升视觉基准与时间一致性。

**🔧 技术方法**

技术核心包括：多粒度稠密字幕生成、EVSG构建与自我精炼、GRPO（Group Relative Policy Optimization）强化学习微调、视觉注意力奖励设计以及文本格式化奖励。

**📊 数据集**

使用了 Rextime（基于事件因果关系的定位与问答）和 VidHalluc（视频误报评估）两个公开基准数据集进行训练与评估。

**📈 对比分析**

在两大基准上均实现了最新性能：在 Rextime 上 mIoU 提升 11.74%、Acc 提升 10.22%、Acc@IoU≥0.5 提升 8.86%；在 VidHalluc 上 Action、Temporal Sequence 与 Scene Transition 误报率分别下降 7.83%、7.81% 等，优于目前所有开源 MLLM，接近或超过部分闭源大模型。

**⚠️ 局限性**

主要局限包括：依赖于 MLLM 的字幕与图结构质量，事件数目阈值需手工调参；方法对大规模 MLLM 的算力需求高；在极长视频或复杂多角色场景中仍可能出现细粒度误报。

---

## 413. A Theoretical Framework for Modular Learning of Robust Generative Models

**arXiv ID:** 2602.17554 | [PDF](https://arxiv.org/pdf/2602.17554v1)

**作者:** Corinna Cortes `[一作]` (Google Research), Yutao Zhong `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个鲁棒模块化生成模型的理论框架，并训练能够在任意数据混合下保持稳定性能的门控网络。

**💡 创新点**

通过构造最小-最大游戏证明单一鲁棒门的存在，并给出容量、JSD 等解析上界；同时提出可扩展的随机原始-对偶算法和结构蒸馏方法。

**🔧 技术方法**

使用 Kakutani 固定点、凸凹最优分析、Exponentiated Gradient + OGD、随机原始-对偶训练、以及结构蒸馏等技术。

**📊 数据集**

在合成的对立规则任务以及真实数据集（Wikipedia、Code、FineWeb）上进行实验验证。

**📈 对比分析**

与单一重训练模型、oracle 以及 monolithic 基线比较，鲁棒门在混合比例中间段表现更优，参数更少；在真实数据上平均 NLL 下降约 0.1–0.2。

**⚠️ 局限性**

主要局限在于非因果门导致推理成本高，需要额外蒸馏；在专家数很大时接受率下降，且对非凸目标缺乏严格收敛保证。

---

## 414. Using LLMs for Knowledge Component-level Correctness Labeling in Open-ended Coding Problems

**arXiv ID:** 2602.17542 | [PDF](https://arxiv.org/pdf/2602.17542v1)

**作者:** Zhangqi Duan `[一作]` (University of Massachusetts Amherst), Andrew Lan `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 1827 | [OpenAlex ID](https://openalex.org/A5063813962)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型自动为开放式编程问题生成知识组件级正确性标签。

**💡 创新点**

创新点在于将链式思考提示与时间上下文感知的代码–KC映射结合，实现细粒度KC正确性评估。

**🔧 技术方法**

采用GPT‑4o或Qwen3 Coder的链式思考提示，结合CodeBERT嵌入的最近邻匹配。

**📊 数据集**

使用来自CodeWorkout公开数据集的10,834段Java代码、246名学生和50道题目。

**📈 对比分析**

与仅使用问题级正确性作为KC正确性的基线比较，在学习曲线RMSE、r²和AFM AUC上均显著提升。

**⚠️ 局限性**

局限性在于假设最终提交能充分反映学生策略，且对不同LLM的指令不一致性敏感。

---

## 415. Position: Evaluation of ECG Representations Must Be Fixed

**arXiv ID:** 2602.17531 | [PDF](https://arxiv.org/pdf/2602.17531v1)

**作者:** Zachary Berger `[一作]` (Massachusetts Institute of Technology), Collin M. Stultz `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3581 | [OpenAlex ID](https://openalex.org/A5024941370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

对12导联心电图表示学习的评估方法进行批判性审视，提出更全面的临床任务与评估最佳实践，并对三种主流预训练模型与随机编码器进行线性探测实验，展示随机初始化编码器可作为有效基线。

**💡 创新点**

1）阐明宏AUROC和标签不平衡对方法排名的误导性影响；2）提出结构性疾病、血流动力学与患者预测等新的临床任务家族；3）强调随机编码器在多任务中的竞争力，并提供公开实验与统一评估框架。

**🔧 技术方法**

对比CLOCS、MERL、D‑BETA三种自监督预训练模型与随机1D ResNet‑18，使用线性探测、宏/微AUROC、AUPRC、配对自举、置信区间等统计方法进行评估。

**📊 数据集**

PTB‑XL、CPSC2018、CSN、EchoNext公开数据集用于标准任务；MIMIC‑IV作为预训练语料；MGH 9k/5k ECG用于血流动力学推断；MGH 913k ECG用于1年心衰预测。

**📈 对比分析**

在低标注下MERL/D‑BETA略优于随机编码器；随着标注量增大，随机编码器与其接近甚至超越CLOCS；在结构疾病、血流动力学和患者预测任务中，随机编码器仍与专业模型相近，MERL/D‑BETA在1年心衰预测上略占优势；总体上无方法在所有任务上稳居首位，排名易受标签分布和置信区间影响。

**⚠️ 局限性**

仅关注12导联心电图，未验证其他生物信号或多模态；评估依赖公开与私有数据分割；未深入探究随机编码器高性能原因；需要更大、可公开的结构性疾病/血流动力学数据集；实验仅采用线性探测，未考察更深层模型。

---

## 416. Variational inference via radial transport

**arXiv ID:** 2602.17525 | [PDF](https://arxiv.org/pdf/2602.17525v1)

**作者:** Luca Ghafourpour `[一作]` (ETH Zurich), Aram-Alexandre Pooladian `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了基于径向传输（radial transport）的变分推断框架和算法 radVI，用于近似目标后验的径向分布，并能与传统的高斯 VI 或拉普拉斯逼近结合。

**💡 创新点**

创新点在于：① 用径向分布族而非传统高斯族进行变分推断；② 对径向最优传输映射给出了正则性、收敛性和 Caffarelli 收缩估计；③ 通过分段线性基函数构造可逼近的径向映射，并证明其近似性和梯度下降收敛；④ 提供了白化步骤，使得算法可直接提升现有 VI 方案的尾部拟合。

**🔧 技术方法**

主要技术包括：Wasserstein 空间中的梯度流与最优传输理论、Caffarelli 收缩定理、径向映射的参数化与 Gram 矩阵投影、随机投影梯度下降（SPGD）以及高斯白化。

**📊 数据集**

实验使用的主要数据集为合成分布：高斯、Student‑t、拉普拉斯、Logistic（d=50 或 100）以及 Neal’s funnel 用于参数估计。没有使用真实世界数据集。

**📈 对比分析**

与 Gaussian VI（GVI）和拉普拉斯逼近（LA）进行比较，评估指标包括 2‑Wasserstein 距离、参数估计误差和尾部概率。结果显示：在重尾分布（Student‑t、Logistic）上 radVI 能显著降低误差，收敛速度快；在高斯/轻尾分布上也能达到与传统 VI 同等精度；在 Neal’s funnel 上，radVI+GVI 能大幅提升参数估计准确度。

**⚠️ 局限性**

局限性：理论收敛仅在后验满足对数光滑且强对数凸的假设下成立；对非对数凸或非光滑分布的理论支持有限；参数化阶数与切点选择需经验调节，计算 Gram 矩阵在高维下可能成本较高；常数因子依赖于后验条件数，虽维度无关但可能导致实用中的慢收敛。

---

## 417. A Picture of Agentic Search

**arXiv ID:** 2602.17518 | [PDF](https://arxiv.org/pdf/2602.17518v1)

**作者:** Francesca Pezzuti `[一作]` (University of Pisa), Nicola Tonellotto `[通讯]` (University of Pisa)

**通讯引用:** 2240 | [OpenAlex ID](https://openalex.org/A5018894843)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套通用方法，用于捕捉检索增强生成（RAG）代理系统在回答查询时的所有中间交互，包括生成的子查询、检索到的文档以及思考过程，并基于此发布了首个Agentic Search Queryset（ASQ）数据集；

**💡 创新点**

该工作首次提供了能真实记录AI代理检索行为的数据集和可扩展方法，揭示代理与人类查询在查询数量、长度、重构策略等方面的根本差异，为重新评估传统IR假设提供依据；

**🔧 技术方法**

技术实现上，利用正则表达式解析代理生成的控制标签，拦截检索调用记录文档ID，并将框架（frame）和轨迹（trace）以结构化JSON形式保存；实验使用Search‑R1和AutoRefine两大代理，检索后端包括BM25和MonoElectra重排序；

**📊 数据集**

数据来源包括HotpotQA、Researchy Questions和MS MARCO三个公开基准，配合其公开的qrels以支持检索优化与评估；

**📈 对比分析**

通过统计轨迹长度、查询次数、状态转移矩阵等指标，对比代理与人类搜索行为；结果显示代理生成更长、更频繁的查询，且更倾向于回退至旧查询，表明传统缓存与重构策略对代理场景的适用性受限；

**⚠️ 局限性**

局限性在于仅涵盖公开的代理和有限的配置，推理成本高导致实验规模受限；缺乏真实生产日志，子查询的相关性评估未完成；随着代理快速迭代，数据集可能迅速过时。

---

## 418. FoundationPose-Initialized 3D-2D Liver Registration for Surgical Augmented Reality

**arXiv ID:** 2602.17517 | [PDF](https://arxiv.org/pdf/2602.17517v1)

**作者:** Hanyuan Zhang `[一作]` (University College London), Matthew J. Clarkson `[通讯]` (University College London)

**通讯引用:** 5529 | [OpenAlex ID](https://openalex.org/A5001311980)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

利用深度增强的 FoundationPose 进行刚性初始化，并通过 NICP 与 CMA-ES 在 Hausdorff 距离上对肝脏 3D 模型进行非刚性优化，实现腹腔镜下肝脏肿瘤定位。

**💡 创新点**

创新点在于：① 用单目深度图代替传统有限元模型，显著降低工程复杂度；② 将深度与轮廓信息融合到 FoundationPose 的 RefineNet，提升刚性姿态估计精度；③ 采用 CMA‑ES 对非刚性形状系数进行无梯度优化，克服 Hausdorff 距离不可微的问题。

**🔧 技术方法**

采用的技术包括 FoundationPose + RefineNet、Depth Anything V2 生成深度图、NICP（非刚性 ICP）、CMA‑ES（协方差矩阵自适应进化策略）、PCA 低维形状模型、Hausdorff 距离评价。

**📊 数据集**

使用 Rabbani 等公开手术视频（4 位患者、共数十帧）进行定量评估；使用 Montaña‑Brown 等人公开的肝脏网格集合进行 PCA 训练；对患者 2 的极端扭曲病例进行排除。

**📈 对比分析**

与手动、LMR、NM、Opt‑B 等传统/学习方法对比；刚性初始化平均误差 9.91 mm，非刚性优化后平均误差降至 8.52 mm；单帧刚性阶段仅需数秒，非刚性阶段约 30–60 s，满足临床实时需求。

**⚠️ 局限性**

局限性包括：① 对严重扭曲肝脏（患者 2）无法恢复；② 内部肿瘤定位精度不及有限元方法；③ 可用带有肿瘤 TRE 标注的数据集极少，影响统计显著性。

---

## 419. AutoNumerics: An Autonomous, PDE-Agnostic Multi-Agent Pipeline for Scientific Computing

**arXiv ID:** 2602.17607 | [PDF](https://arxiv.org/pdf/2602.17607v1)

**作者:** Jianda Du `[一作]` (University of Maryland), Haizhao Yang `[通讯]` (University of Maryland)

**通讯引用:** 2186 | [OpenAlex ID](https://openalex.org/A5079602544)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出多代理框架AutoNumerics，能从自然语言描述自动设计、实现、调试、验证通用PDE数值求解器，并生成可解释的传统数值方案。

**💡 创新点**

创新点包括：①多代理协同生成透明经典数值解法而非黑盒网络；②采用粗到细执行策略和残差自检实现自动调试与验证；③计划与选择代理嵌入稳定性、一致性推理，提前过滤不合理方案。

**🔧 技术方法**

技术上使用GPT‑4.1作为多代理（Formulator、Planner、Feature、Selector、Coder、Critic、Reasoning）生成和改进代码；实现粗到细执行策略、残差自检、历史压缩等机制；采用有限差分、谱方法、有限体积、有限元等经典数值方法并进行稳定性分析。

**📊 数据集**

数据集包括两套基准：CodePDE 5个典型PDE（1D Advection、1D Burgers、2D Reaction‑Diffusion、2D CNS、2D Darcy Flow）和自构建的200 PDE套件，涵盖1D–5D、线性/非线性、椭圆/抛物/双曲、各种边界条件。

**📈 对比分析**

与六个神经网络基线（U‑Net、FNO、PINN、ORCA、PDEformer、UPS）、CodePDE以及一个“坏设计”中心差分基线比较。AutoNumerics 在CodePDE 5个基准上的 nRMSE 低于所有基线，几乎比 CodePDE 低六个数量级；在精选的 24 个问题中，有 11 个达到 10⁻⁶ 以下误差，部分问题接近机器精度；整体运行时间在 20–130 秒。

**⚠️ 局限性**

局限性：对高维（≥5D）和高阶（四阶）PDE 的准确性有限；仅在规则域进行测试；缺乏正式的收敛/稳定性证明；依赖单一 LLM；对复杂非结构化网格或自适应网格的支持不足。

---

## 420. Art2Mus: Artwork-to-Music Generation via Visual Conditioning and Large-Scale Cross-Modal Alignment

**arXiv ID:** 2602.17599 | [PDF](https://arxiv.org/pdf/2602.17599v1)

**作者:** Ivan Rinaldi `[一作]` (University of Bari Aldo Moro), Gennaro Vessio `[通讯]` (University of Bari Aldo Moro)

**通讯引用:** 1485 | [OpenAlex ID](https://openalex.org/A5047012566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个105,884对艺术品–音乐配对的大规模数据集，并提出了直接基于视觉的音乐生成框架；

**💡 创新点**

创新点在于彻底去除了文本中介，直接通过视觉编码器与生成模型对齐学习图像与音乐的跨模态关系，并设计了视觉条件提取器和图像对齐模块；

**🔧 技术方法**

采用CLIP/ImageBind等视觉编码器、GPT‑2 LoA翻译器、Latent Diffusion音频生成模型以及自定义的投影层实现视觉条件注入；

**📊 数据集**

使用WikiArt与Free Music Archive扩展得到的105,884对艺术品–音乐配对，并为每个样本生成双模态字幕；

**📈 对比分析**

通过IBSc、FAD、KL‑Div等客观指标和主观评估，与文本基线对比显示在无语言监督下实现了竞争性的音质和跨模态一致性；

**⚠️ 局限性**

局限在于跨模态一致性分数仍低于文本基线，对复杂视觉结构的捕捉有限，并且需要更丰富的音乐与艺术风格覆盖来提升模型表现。

---

## 421. BMC4TimeSec: Verification Of Timed Security Protocols

**arXiv ID:** 2602.17590 | [PDF](https://arxiv.org/pdf/2602.17590v1)

**作者:** Agnieszka M. Zbrzezny `[一作]` `[通讯]` (SWPS University), Agnieszka M. Zbrzezny (SWPS University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了 BMC4TimeSec 工具，用于基于时序解释系统（TIIS）对多会话时序安全协议进行模型检测与证据生成；

**💡 创新点**

创新点在于首次将时间执行建模、代理知识建模、会话多线程化、SMT‑BMC 以及可视化证据解释整合到一个完整、模块化、易用的工具链中；

**🔧 技术方法**

采用 Python 编写 Alice‑Bob 解析器、TIIS 生成器和 Flask GUI，C++ 生成 SMT‑LIB2 公式，使用 Z3 SMT 求解器完成 BMC；

**📊 数据集**

使用了多协议时序规范库（如 NSPK、WMF、Yahalom 等）以及对应的 JSON 场景覆盖，包括公平与多种攻击变体；

**📈 对比分析**

与 VerSecTis 等先前工作相比，BMC4TimeSec 通过提供更完整的协议和攻击库、支持多会话交叉、自动生成可解释证据，显著提升了易用性和可重复性；性能方面在实验中实现了可接受的检测时间（相较于纯手工验证或传统工具更快），但具体指标未给出；

**⚠️ 局限性**

局限性包括：对协议的手工描述仍需一定专业知识；工具在极大会话数或极复杂时间约束下可能面临状态空间爆炸；未覆盖非时间安全协议或非 Dolev‑Yao 模型的攻击。

---

## 422. Be Wary of Your Time Series Preprocessing

**arXiv ID:** 2602.17568 | [PDF](https://arxiv.org/pdf/2602.17568v1)

**作者:** Sofiane Ennadir `[一作]` (King AI Labs, Microsoft Gaming), Lele Cao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究Transformer时间序列模型的输入归一化对表达能力和性能的影响，并给出理论与实证分析

**💡 创新点**

首次将归一化视为设计选择，推导表达能力上界并阐明实例级与全局归一化对模型敏感性的差异

**🔧 技术方法**

使用Transformer、PatchTST、Autoformer、TimesNet等架构，搭配标准、MinMax、Quantile、Robust归一化及无归一化处理

**📊 数据集**

使用UEA时间序列分类库、ETT‑Small等多任务数据集进行实验

**📈 对比分析**

通过分类准确率和MAE与不同归一化方式比较，发现无归一化或全局归一化在某些任务上表现最佳，差异显著

**⚠️ 局限性**

未提出新的归一化方法，理论推导仅给出上界，实验范围有限，缺乏跨域更广泛验证

---

## 423. TopoSZp: Lightweight Topology-Aware Error-controlled Compression for Scientific Data

**arXiv ID:** 2602.17552 | [PDF](https://arxiv.org/pdf/2602.17552v1)

**作者:** Tripti Agarwal `[一作]` (University of Utah), Franck Cappello `[通讯]` (Argonne National Laboratory)

**通讯引用:** 12813 | [OpenAlex ID](https://openalex.org/A5046613458)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种轻量化、拓扑感知的误差控制无损压缩方法 TopoSZp，能够在保持误差上限的同时保留关键点及其相对关系。

**💡 创新点**

创新点在于将快速关键点检测、相对排序元数据以及基于 RBF 的鞍点修正直接嵌入 SZp 的量化阶段，既消除了 FP/FT 误差，又大幅提升压缩/解压速度。

**🔧 技术方法**

技术手段包括 SZp 的量化编码、OpenMP 并行、极值 stencil、相对排序元数据、RBF 插值以及对元数据的二次压缩。

**📊 数据集**

实验使用了 CESM 产生的五个科学数据集（ATM、CLIMATE、ICE、LAND、OCEAN）进行评估。

**📈 对比分析**

通过与 SZ1.2、SZ3、ZFP、Tthresh 以及 TopoSZ、TopoA 等拓扑压缩器对比，TopoSZp 在 1–18 线程下实现了 10^2–10^4 倍的压缩/解压速度提升，FP/FT 均为 0，FN 大幅减少至其他方法的 1/3–1/100。

**⚠️ 局限性**

局限性包括目前仅适用于 2D 标量场，且在极端梯度变化下 RBF 修正可能无法完全恢复被量化破坏的鞍点，未来需扩展到 3D 与 GPU 平台。

---

## 424. KLong: Training LLM Agent for Extremely Long-horizon Tasks

**arXiv ID:** 2602.17547 | [PDF](https://arxiv.org/pdf/2602.17547v1)

**作者:** Yue Liu `[一作]` (National University of Singapore), Bryan Hooi `[通讯]` (National University of Singapore)

**通讯引用:** 33498 | [OpenAlex ID](https://openalex.org/A5056748708)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 KLong，一款针对极长时域任务（如科研论文复现）的开源 LLM 代理，并将其能力推广到 bug 修复、ML 工程、终端编程与代码安全等领域。

**💡 创新点**

创新点包括：① 轨迹分割 SFT——在保留任务前置信息的同时，将长轨迹按上下文窗口切片并重叠训练；② 渐进式 RL——在多阶段逐步延长任务超时，缓解奖励稀疏和样本方差；③ Research‑Factory——自动化构建高质量论文复现数据及评估 Rubric，生成千条长轨迹；④ 对基础设施的整体优化（沙箱、优先队列、预装库）提升训练效率。

**🔧 技术方法**

技术手段：SFT 与轨迹分割、PPO‑style 渐进 RL、上下文重叠子轨迹、优先级评判队列、GPU/Kubernetes 沙箱、预装科研库、使用 GPT‑oss‑120b 做评判、基于 Claude‑4.5‑Sonnet 进行轨迹蒸馏。

**📊 数据集**

使用数据集：PaperBench（论文复现评测）、MLE‑bench、SWE‑bench Verified、Terminal‑Bench Hard、SEC‑bench 以及 Research‑Factory 自动生成的数千条论文复现轨迹，覆盖 ICML/NeurIPS/ICLR 等顶级会议论文。

**📈 对比分析**

对比方法：与多款闭源与开源代理在 PaperBench 上对比，KLong（106B）平均得分 62.59，超过 Kimi K2 Thinking（1T）11.28%，在多项任务上与闭源模型差距显著缩小，并在 MLE‑bench、SWE‑bench、SEC‑bench 等任务上取得多项“Above Median”“Bronze”“Gold”级别成绩，显示出强大的泛化能力。

**⚠️ 局限性**

局限性：① 训练成本高，需数千小时 GPU 与复杂 infra；② 对极长时域任务依赖多阶段 RL，短时域任务效果未知；③ 评判依赖外部模型（GPT‑oss‑120b），可能存在评估偏差；④ 论文复现数据仍受源论文质量与可获取性限制，潜在偏见；⑤ 在某些闭源模型上仍存在性能差距，需进一步优化。

---

## 425. Informative Trains: A Memory-Efficient Journey to a Self-Stabilizing Leader Election Algorithm in Anonymous Graphs

**arXiv ID:** 2602.17541 | [PDF](https://arxiv.org/pdf/2602.17541v1)

**作者:** Lelia Blin `[一作]` (Université Paris Cité), Isabella Ziccardi `[通讯]` (CNRS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在匿名图中使用 (loglog n) 位内存的自稳健领导选举算法。

**💡 创新点**

创新点在于引入信息列车机制，让信息以 (log n)-比特传播却只需 (loglog n) 位本地存储。

**🔧 技术方法**

采用状态模型、同步调度、随机化以及信息列车与分布式计数器的实现。

**📊 数据集**

未使用外部数据集，仅在理论分析中假设 N = Θ(log n)。

**📈 对比分析**

与先前需要 Ω(log n) 位的方案相比，算法在一般图上以多项式时间收敛，空间显著降低。

**⚠️ 局限性**

限制在于算法非静默、需已知 N、仅适用于同步调度，且未证明是否可进一步降低空间。

---

## 426. LATA: Laplacian-Assisted Transductive Adaptation for Conformal Uncertainty in Medical VLMs

**arXiv ID:** 2602.17535 | [PDF](https://arxiv.org/pdf/2602.17535v1)

**作者:** Behzad Bozorgtabar `[一作]` (Aarhus University), Zongyuan Ge `[通讯]` (Monash University)

**通讯引用:** 11666 | [OpenAlex ID](https://openalex.org/A5005014252)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种无训练、无标签的转导式校正方法（Laplacian-Assisted Transductive Adaptation，LATA），在冻结的医学视觉‑语言模型（VLM）上利用图拉普拉斯正则化对零射概率进行平滑，并结合 ViLU 生成的难度与标签关注信号，构造失败感知的非一致性分数；

**💡 创新点**

创新点在于：①使用稀疏 k‑NN 图和 CCCP/均值场迭代对联合校准+测试样本进行无监督的概率平滑，保持分布可交换性并保留 Split Conformal Prediction 的有效性；②引入失败感知 conformal 分数，利用多模态置信度信息进一步收缩预测集并平衡类覆盖率；③提供可选的弱标签先验，兼顾无标签与弱标签两种部署模式；

**🔧 技术方法**

技术包括：Split Conformal Prediction、图拉普拉斯正则化、CCCP/均值场迭代、k‑NN 图构建、ViLU 失败预测模块、失败感知非一致性分数设计；

**📊 数据集**

数据集：三种医学 VLM（CONCH、FLAIR、CONVIRT）在九个下游任务上评估，涵盖结肠组织分类（NCT-CRC）、前列腺 Gleason 分级（SICAPv2）、皮肤病变（SkinCancer）、糖尿病视网膜病变（MESSIDOR）、青光眼病变（MMAC）、多标签眼病（FIVES）、胸部 X‑ray（CheXpert、NIH‑LT、COVID）等；

**📈 对比分析**

与标签使用方法（Adapt+SCP、FCA）和无标签基线（Conf‑OT、SCA‑T、SCP）比较，LATA 在 α=0.05/0.10 下平均覆盖率 ≥ 目标覆盖率，平均预测集大小比 SCP 小约 10–15%，CCV 减少 10–20%，与标签使用方法的性能相近，仅在极少标签场景下略逊；

**⚠️ 局限性**

局限性：依赖图像嵌入质量及 ViLU 信号的可靠性，域迁移严重时效果下降；仅适用于多类别分类任务；需要一定量的校准样本（K≈16）并假设校准与测试可交换；

---

## 427. Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs

**arXiv ID:** 2602.17616 | [PDF](https://arxiv.org/pdf/2602.17616v1)

**作者:** Luke Huang `[一作]` (Massachusetts Institute of Technology), Song Han `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 33318 | [OpenAlex ID](https://openalex.org/A5070926896)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大语言模型后训练中使用异步强化学习（RL）时的高方差问题，并提出了 Variance Controlled Policy Optimization (VCPO) 方法以实现稳定、可扩展的异步 RL 训练。

**💡 创新点**

创新点在于：① 通过有效样本大小 (ESS) 诊断异步训练导致的梯度方差崩溃；② 提出基于 ESS 的学习率自适应缩放；③ 推导并实现了闭式最小方差基线 (OPOB) 用于重要性加权梯度，避免了额外的 critic 训练；④ 在单次反向传播内高效计算基线权重。

**🔧 技术方法**

主要技术包括：重要性采样（IS）与序列级截断（TIS）、基于 ESS 的学习率缩放、闭式最小方差基线、单反向传播的梯度累积实现、PipelineRL 异步分布式训练框架。

**📊 数据集**

实验使用了多种数据集：数学推理（GSM8K、MATH-500）、可验证推理（Countdown）、多轮工具调用（SimpleTIR / DAPO）以及 AIME2025 评测集。

**📈 对比分析**

与同步 RL（k=0）以及多种现有稳定化基线（TIS、MIS、M2PO、GSPO、OTB、FP16 等）进行对比。VCPO 在高异步程度（k≥10）下保持训练稳定，最终性能与同步训练持平；在 12 步异步训练时相较同步训练可将计算时间压缩约 2.5 倍，并在三大任务上均获得最优或接近最优的准确率。

**⚠️ 局限性**

局限性：仅在稠密 Transformer 模型、FP16/FP32 精度、固定的 PipelineRL 异步设计下验证；未评估 MoE 模型、FP8 等更极端量化、极长路径或稀疏奖励场景下的鲁棒性，未来工作需进一步扩展。

---

## 428. AI Gamestore: Scalable, Open-Ended Evaluation of Machine General Intelligence with Human Games

**arXiv ID:** 2602.17594 | [PDF](https://arxiv.org/pdf/2602.17594v1)

**作者:** Lance Ying `[一作]` (Massachusetts Institute of Technology), Joshua B. Tenenbaum `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 73403 | [OpenAlex ID](https://openalex.org/A5071093940)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 AI GameStore 平台，利用 LLM 自动从 Apple App Store 与 Steam 生成标准化、可玩的人类游戏，并用该平台评估七款前沿视听语言模型在 100 个游戏短时段内的表现。

**💡 创新点**

创新点：① 提出“人类游戏多元宇宙”作为评估通用人工智能的基准；② 通过 LLM 与人类迭代生成、改进游戏，实现可持续扩展的开源基准；③ 结合游戏认知需求注释与模型性能对比，揭示模型在记忆、规划、世界模型学习等核心能力上的缺口。

**🔧 技术方法**

技术：LLM（Gemini 2.5 Flash、Claude‑sonnet‑4.5、GPT‑5.x 等）生成游戏代码；p5.js 环境运行游戏；人机交互 harness（每秒五次动作列表）实现模型与游戏交互；手工注释认知需求（视觉处理、时空协调、记忆、规划、世界模型学习、物理推理、社会推理）；几何平均归一化性能计算。

**📊 数据集**

数据集：从 Apple App Store 与 Steam 采集 7,500 款游戏，按评分与评论筛选后通过 LLM 进一步挑选 100 款可生成游戏；106 名人类参与者在 2 分钟内玩每款游戏；评估七款 VLM；游戏本身的截图、动作序列与得分构成实验数据。

**📈 对比分析**

比较方法：将每款游戏人类中位数设为 100，模型得分按 100 × RawScore / HumanMedian 归一化后取几何平均；结果显示 GPT‑5.2 等模型在 100 款游戏的几何平均得分低于 10%（仅 8.5/100），与人类相比表现显著不足；模型耗时 10‑20 倍人类时间；表现呈双峰分布，只有少数简单游戏能逼近人类水平。

**⚠️ 局限性**

限制：① 游戏样本仅 100 款，且大多为短时、低复杂度的 casual 游戏；② 缺少多智能体社交、长时序、深度物理推理等挑战；③ harness 的低实时性与批量查询可能低估模型实际能力；④ 生成游戏的趣味性与难度可进一步提升；⑤ 版权、平台异构与数据污染等技术与伦理障碍仍待解决。

---

## 429. Hybrid System Planning using a Mixed-Integer ADMM Heuristic and Hybrid Zonotopes

**arXiv ID:** 2602.17574 | [PDF](https://arxiv.org/pdf/2602.17574v1)

**作者:** Joshua A. Robbins `[一作]` (Pennsylvania State University), Herschel C. Pangborn `[通讯]` (Pennsylvania State University)

**通讯引用:** 575 | [OpenAlex ID](https://openalex.org/A5059842706)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了利用混合锥形体（Hybrid Zonotope）进行分段线性（PWA）系统的可达性分析，并基于此构建离线规划问题；随后设计了一种结合 ADMM 与可行性泵（Feasibility Pump）思想的混合整数规划启发式（ADMM‑FP），用于在资源受限的嵌入式平台上快速求解这些规划问题。实验中将框架应用于小车仿真、双积分器避障、经典球板问题以及真实的双车道自动驾驶行为与运动规划任务。

**💡 创新点**

创新点主要有三：① 开发了针对 PWA 系统的混合锥形体可达性公式，显著降低记忆复杂度并得到更紧凑的凸松弛；② 设计了 ADMM‑FP 启发式，将 ADMM 与可行性泵的随机扰动/重启机制结合，提升可行解的收敛率（最多提升 30 倍）；③ 在嵌入式实验中验证了该框架可在毫秒级完成规划，满足实时需求。

**🔧 技术方法**

使用技术包括：混合锥形体（Hybrid Zonotope）与其运算（并集、交集、Minkowski 和投影）；可达性分析（基于图函数的递归）；凸松弛和线性/二次规划；ADMM（含 ρ 调参、投影、重启/扰动）；可行性泵（Feasibility Pump）相关策略；热启动（warm‑start）与最优解投影；实验代码基于 ZonoOpt 库，C++ 实现。

**📊 数据集**

数据集与实验场景：
- 随机生成 100 组混合锥形体作为可行集；
- 100 组随机障碍环境下的双积分器避障问题；
- 经典球板基准（7 模式 PWA）
- 4‑车道模拟的两车道自动驾驶（带动态障碍）
- 真实运动捕捉实验场景：双车道闭环跑道，使用 Husarion ROSbot3 与 Clearpath TurtleBot4 进行行为与运动规划。

**📈 对比分析**

比较方法：
- 对随机 MILP：与标准 ADMM 启发式、OFP（Objective Feasibility Pump）和商业求解器 Gurobi 进行对比；
- 对避障与球板问题：与 OFP、ADMM、Gurobi 的求解时间、成功率和子最优度量；
- 对真实实验：与默认的行为/运动规划（无混合锥形体+ADMM‑FP）进行规划时间、迭代次数、成功率对比。 
性能结果：
- 在随机 MILP 上，ADMM‑FP 100% 成功率，平均求解时间约 0.2–0.4 s；
- 避障任务中 ADMM‑FP 的成功率高于 OFP、ADMM（>90% 对比 <70%），平均时间 0.2 s；
- 球板任务 ADMM‑FP 在 ϵp=0.01 时可获得 100% 成功率，平均 0.99 s；
- 真实实验中 19 次规划调用，ADMM‑FP 在 15 次内 1 s 内给出可行解，平均 5.2 ms 规划时间，显著快于商业求解器。

**⚠️ 局限性**

局限性：
- ADMM‑FP 无理论收敛保证，某些难解（如球板）在严格容忍度下可能无法在 1 s 内得到可行解；
- 对初始迭代敏感，需有效的热启动或强凸松弛来提升成功率；
- 对大规模/高维混合整数问题，虽然内存占用低，但迭代次数仍可能膨胀；
- 现有实现主要针对 PWA 线性模型，对非线性或时变动态需要进一步扩展。

---

## 430. IRIS: Learning-Driven Task-Specific Cinema Robot Arm for Visuomotor Motion Control

**arXiv ID:** 2602.17537 | [PDF](https://arxiv.org/pdf/2602.17537v1)

**作者:** Qilong Cheng `[一作]` (New York University), Ali Bereyhi `[通讯]` (University of Toronto)

**通讯引用:** 308 | [OpenAlex ID](https://openalex.org/A5061064331)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一套低成本、3D打印的6自由度电影摄像机机器人IRIS，结合目标条件的ACT+CVAE学习框架实现从人类示范直接学习并执行逼真、平滑的摄像运动；

**💡 创新点**

创新点在于将任务特定的硬件设计与基于视觉的闭环强化学习相结合，利用目标图像进行帧定向控制，消除了传统几何规划的需求；

**🔧 技术方法**

主要技术包括3D打印轻量化机械结构、QDD直接驱动电机、ROS+MuJoCo仿真、逆运动学求解、基于Transformer的ACT+CVAE策略以及视觉目标对齐和低延迟推理；

**📊 数据集**

数据集来源于专家在实物机器人上完成的132条推入镜头示范，包含13,954个训练片段，分为无障碍和障碍避让两种情境；

**📈 对比分析**

与人类重放、RRT*几何规划和纯行为克隆等基线对比，IRIS在成功率（90% vs. 10%）、视觉对齐、轨迹平滑度和实时推理方面均表现突出；

**⚠️ 局限性**

局限性包括机械柔性导致的高扭矩动作失真、数据集场景与物体多样性不足以及对更复杂多阶段摄像机动作的支持有限。

---

## 431. OpenEarthAgent: A Unified Framework for Tool-Augmented Geospatial Agents

**arXiv ID:** 2602.17665 | [PDF](https://arxiv.org/pdf/2602.17665v1)

**作者:** Akashah Shabbir `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salman Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 11703 | [OpenAlex ID](https://openalex.org/A5000300751)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 OpenEarthAgent，一个统一的工具增强地理空间推理框架，能够在卫星影像、光谱指数、GIS 向量/栅格等多模态数据上进行可解释的多步推理。

**💡 创新点**

创新点包括：①构建统一的工具注册表，标准化感知、GIS、光谱计算等操作；②利用结构化推理轨迹进行监督式微调，使模型学会按顺序调用工具并记录中间状态；③提供专门的 EO 任务数据集和评测标准，推动从单步识别向可解释推理的转变。

**🔧 技术方法**

技术实现基于 Qwen3‑4B‑Instruct‑2507 作为语言模型骨干，采用监督微调、工具调用 JSON 交互、工作内存维护、工具执行缓存等方法；工具集涵盖感知、GIS 计算、光谱指数、栅格处理及辅助功能。

**📊 数据集**

使用的数据集为 OpenEarthAgent Corpus，包含 14,538 条训练实例和 1,169 条评测实例，覆盖 RGB、SAR、GIS 矢量/栅格、光谱指数（NDVI、NBR、NDBI）等多源遥感信息，任务涵盖城市、环境、灾害、基础设施等七大主题。

**📈 对比分析**

评估采用 step‑by‑step（工具调用无执行）和 end‑to‑end（完整执行）两种模式，并与 GPT‑4o、o4‑mini、Qwen2.5‑7B、Llama‑3.1‑8B、Internlm3‑8B 等前沿及开源模型对比。OpenEarthAgent 在实例、工具、参数名/值准确率上几乎无缝接近 GPT‑4o，且在工具顺序一致性（AnyOrder/SameOrder/Unique）上显著领先；最终总结准确率虽低于 GPT‑4o，但在小模型尺寸下实现了最优或接近最优的多步推理效果。

**⚠️ 局限性**

局限性包括：①相比 GPT‑4o，整体总结准确率仍有差距；②对工具集的依赖较高，若出现未注册或新型工具会导致失效；③数据集虽然多样，但仍未覆盖所有极端遥感场景和所有 GIS 投影；④模型规模虽小，但对高分辨率大图像的实时推理能力尚未充分验证。

---

## 432. When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs

**arXiv ID:** 2602.17659 | [PDF](https://arxiv.org/pdf/2602.17659v1)

**作者:** Yu Fang `[一作]` (University of North Carolina at Chapel Hill), Mingyu Ding `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 2876 | [OpenAlex ID](https://openalex.org/A5022382771)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个专门评估 Vision‑Language‑Action 模型（VLA）对语言指令遵循能力的对照实验基准 LIBERO‑CF，并设计了 Counterfactual Action Guidance（CAG）双分支推理框架来减少 VLA 的视觉捷径导致的反事实失败。

**💡 创新点**

创新点在于①首次构建了针对 VLA 的反事实失败基准，系统量化了视觉捷径的影响；②提出了基于“无监督指导”的 CAG 方法，利用 VLA 与无条件 Vision‑Action 分支的混合，提升语言条件性而无需改造原始模型或额外训练；③在模拟与真实机器人环境上统一验证了该方法的有效性。

**🔧 技术方法**

技术手段主要包括：VLA 与 Vision‑Action 模型的分支设计、基于分类器无监督指导（CFG）思想的推理混合公式、训练自由与训练 VA 两种实现方式、以及在 LIBERO 数据集上对多种 VLA 的微调与评估。

**📊 数据集**

使用的主要数据集是 LIBERO（用于 VLA 微调）和其扩展版 LIBERO‑CF（对照实验基准），同时在真实 Franka 机器人上使用自制场景进行多任务评测。

**📈 对比分析**

与三种主流 VLA 基线（OpenVLA‑OFT、π_0、π_0.5）对比，CAG 在 LIBERO‑CF 的 grounding 与 success 指标平均提升 9.7% 与 3.6%（训练自由版）或 15.5% 与 8.5%（VA 版），在真实环境中平均减少 9.4% 的反事实失败率并提升 17.2% 的任务成功率。

**⚠️ 局限性**

局限性包括：① CAG 仍依赖于可用的 Vision‑Action 先验模型，若缺乏高质量 VA 可能提升有限；② 对高度动态或完全新颖任务的推广尚未充分验证；③ 只在 LIBERO 生态系统内验证，跨域鲁棒性需进一步研究。

---

## 433. What Language is This? Ask Your Tokenizer

**arXiv ID:** 2602.17655 | [PDF](https://arxiv.org/pdf/2602.17655v1)

**作者:** Clara Meister `[一作]` (École Polytechnique Fédérale de Lausanne), Tiago Pimentel `[通讯]` (ETH Zurich)

**通讯引用:** 1002 | [OpenAlex ID](https://openalex.org/A5102825308)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于生成式分词模型的语言识别方法 UniLID，利用语言特定的 unigram 词分布并在分词时视为语言相关隐变量，通过贝叶斯决策实现语言判别。

**💡 创新点**

创新点在于：①将分词过程视为语言特定的隐变量，允许同一词表在不同语言下有不同的分词概率；②只需学习每种语言的 unigram 分布，支持增量式扩展；③在低资源、方言细粒度等难题上表现优异。

**🔧 技术方法**

主要技术包括：生成式分词模型（基于 tokenization 的概率框架）、EM 算法训练 unigram 参数、动态规划（前向后向、Viterbi）求解分词概率、贝叶斯后验计算语言概率。

**📊 数据集**

使用了多种公开评测集：GlotLID‑C、UDHR、FLORES‑200、DSL‑ML 2024、Tatoeba、WiLI‑2018，涵盖从大规模语言、低资源语言到细粒度方言。

**📈 对比分析**

与 FastText、CLD3、OpenLID、mBERT 等基线相比，UniLID 在宏 F1、FPR、准确率等指标上接近或优于 SOTA；在低资源设置下5个样本即可达70%+准确率，在方言识别上宏 F1 提升至0.72。

**⚠️ 局限性**

局限性：①仅使用 unigram 词分布，忽略词间上下文依赖；②模型参数随语言数线性增长，规模极大语言集合时内存与推理开销较高；③对极低资源或高度混淆文本的鲁棒性仍有待提升。

---

## 434. Mine and Refine: Optimizing Graded Relevance in E-commerce Search Retrieval

**arXiv ID:** 2602.17654 | [PDF](https://arxiv.org/pdf/2602.17654v1)

**作者:** Jiaqi Xi `[一作]` (DoorDash Inc.), Sudeep Das `[通讯]` (DoorDash Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段的“Mine and Refine”对比学习框架，用于在多语言电商搜索中训练语义检索嵌入，支持长尾、噪声查询并符合产品政策；

**💡 创新点**

创新点包括：①基于轻量级LLM与参与度审核实现可扩展的政策一致性三层相关性标注；②将标注的硬样本通过ANN挖掘并用LLM重新标注后，结合多分类圆形损失进行精炼，显著提升相似度边界；③在训练中首次将圆形损失扩展到多类别相关性；

**🔧 技术方法**

使用的技术包括：Siamese双塔架构与投影头、监督对比损失（SupCon）与多分类圆形损失、LLM微调（gpt-4o-mini）做标注、ANN硬样本挖掘、拼写扰动与合成查询扩增、语义增强的多语言预训练模型；

**📊 数据集**

采用的主要数据集为数百万条人工标注的查询-商品对，使用LLM生成的标签用于大规模训练，评估使用包含头尾、跨品类及拼写/多语言混合的155M查询-商品对黄金评测集以及12K查询的在线A/B复现集，检索库包含200M+商品；

**📈 对比分析**

与传统的词检索、单阶段混合检索以及混合检索+SupCon+圆形损失对比，离线NDCG@10提升1.66个百分点，在线ATCR+2.5%、CVR+1.1%、GOV+0.9%，所有提升均具统计显著性；

**⚠️ 局限性**

局限性在于：①相关性仅分为三层，无法覆盖更细粒度的偏好；②硬样本挖掘仍依赖LLM标注，可能引入误标；③对极端长尾或极少量查询的鲁棒性仍待验证；④模型对多语言支持仍有限，主要针对英语及少量非英语；

---

## 435. The Effectiveness of a Virtual Reality-Based Training Program for Improving Body Awareness in Children with Attention Deficit and Hyperactivity Disorder

**arXiv ID:** 2602.17649 | [PDF](https://arxiv.org/pdf/2602.17649v1)

**作者:** Aya Abdelnaem El-Basha `[一作]` (Damanhour University), Ahmad Al-Kabbany `[通讯]` (Arab Academy for Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究设计并实施了为4-7岁ADHD儿童的VR沉浸式身体意识训练项目，评估其对身体意识的提升及其持续性。

**💡 创新点**

创新点在于：①首次针对ADHD儿童的身体意识缺陷开发专门的VR交互训练框架；②构建并验证了专用的“身体意识量表”；③通过随访验证训练效果的长期稳定性。

**🔧 技术方法**

使用的技术包括：沉浸式VR系统（交互式3D场景与体感反馈）、自制的身体意识量表以及标准化评估工具（Stanford‑Binet、Conners量表）。

**📊 数据集**

数据集来源于10名符合条件的儿童（4-7岁，IQ 90-110，ADHD诊断），收集了前测、后测和一月随访的身体意识量表分数，以及标准化智力与注意力评估。

**📈 对比分析**

比较方法采用准实验设计，实验组进行36次VR训练，使用Wilcoxon符号秩检验评估前后差异，结果显示实验组在身体部位识别、空间定位及运动表达等维度均显著提升（p < 0.05），随访显示提升水平保持稳定，未出现显著回落。

**⚠️ 局限性**

局限性包括：样本量极小（N=10），仅涵盖4-7岁儿童，缺乏不同文化与年龄组的验证；技术成本高，未测试低成本VR平台的可行性。

---

## 436. A.R.I.S.: Automated Recycling Identification System for E-Waste Classification Using Deep Learning

**arXiv ID:** 2602.17642 | [PDF](https://arxiv.org/pdf/2602.17642v1)

**作者:** Dhruv Talwar `[一作]` (Apple), Rafael Reveles `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了低成本、可携带的电子废弃物分拣系统A.R.I.S.，通过YOLOx模型实时识别金属、线路板和塑料碎片，并使用PLC驱动气动板刀实现高速分拣；

**💡 创新点**

创新点在于将YOLOx目标检测与PLC控制的气动板刀结合，采用三摄高分辨率拼接与批量推理，实现低成本、高通量的实时分拣，并通过中心点映射精准计算板刀位置与时间；

**🔧 技术方法**

技术包括YOLOx目标检测、SimOTA自适应匹配、OPC‑UA通讯、PLC控制、气动板刀、RGB高帧率摄像机、数据增强、Numba加速预处理、Mac Mini边缘推理；

**📊 数据集**

使用自建6000张碎片图像数据集（金属5000、线路板5500、塑料5000），图像尺寸5760×1200，采用YOLO格式标注并切分为640×640输入；

**📈 对比分析**

评估方法：在1000张测试图像上测得mAP@0.5=82.2%，单类AP分别为金属88.9%、线路板87.2%、塑料70.4%；物理分拣实验100lb批次下金属89%纯度、线路板85%纯度、塑料79%纯度，吞吐量约5 kg/s，帧率>20 FPS；

**⚠️ 局限性**

局限性：对极小碎片（如塑料屑、金属粉尘）检测率低；塑料召回率仅56%，导致漏检；需要扩大数据规模、引入多尺度推理以提升小尺寸检测性能。

---

## 437. Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting

**arXiv ID:** 2602.17634 | [PDF](https://arxiv.org/pdf/2602.17634v1)

**作者:** Xinghong Fu `[一作]` (Massachusetts Institute of Technology), Yoon Kim `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 21981 | [OpenAlex ID](https://openalex.org/A5100693798)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Reverso，一种极小型时间序列基础模型，能够在零样本条件下进行长序列、长预测窗口的时间序列预测。

**💡 创新点**

创新点在于将已成熟的卷积（长卷积）与线性RNN（DeltaNet）混合使用，构建极简模型，同时通过数据增强、合成数据和推理策略提升性能，从而显著压缩模型规模（200k–2.6M参数）并保持或超越大型Transformer模型的表现。

**🔧 技术方法**

采用长卷积、DeltaNet线性RNN、门控机制、MLP通道混合、基于注意力的解码头；训练使用MAE损失，优化器为AdamW；推理时使用翻转等价性和FFT降采样。

**📊 数据集**

在GiftEval预训练集（约4.5M序列、230B点）进行预训练；在GiftEval基准（23个数据集、97个任务）、LTSF/TSLib（ETT、Electricity、Weather等）和多种公开时间序列数据集上评估。

**📈 对比分析**

与Chronos、TimesFM、Xihe、TiRex、FlowState等多种大型TSFM基线对比，Reverso在2.6M参数下的MASE为0.711，Reverso‑Small（550k）为0.726，均优于同规模或更大模型；在长序列预测任务中表现尤为突出。

**⚠️ 局限性**

局限性包括：仅针对单变量预测；在短序列任务上的性能略低于更大模型；未提供分布预测（不确定性估计）功能。

---

## 438. Unmasking the Factual-Conceptual Gap in Persian Language Models

**arXiv ID:** 2602.17623 | [PDF](https://arxiv.org/pdf/2602.17623v1)

**作者:** Alireza Sakhaeirad `[一作]` (École Polytechnique Fédérale de Lausanne), Arshia Hemmat `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DivanBench诊断基准，针对波斯语LLM在迷信与习俗等文化模式上的推理能力进行评测，并对七款模型进行实验；

**💡 创新点**

创新点在于聚焦文化方案（而非仅检索事实），采用正负场景配对设计量化顺从偏差，并对波斯语持续预训练的影响进行系统对比；

**🔧 技术方法**

采用LLM评估技术，包括系统提示多样化、温度0.1、top‑p0.9、GPT‑4o‑mini提取答案，并计算准确率、顺从偏差与事实‑概念差距等指标；

**📊 数据集**

使用公开的DivanBench数据集（315题，涵盖100道事实多项选择、162道正负对比验证、53道情境推理题），覆盖81个波斯文化概念；

**📈 对比分析**

通过在5个不同系统提示下对七款7–12B参数模型（Aya‑8B、Dorna2‑8B、Gemma2‑9B、Gemma3‑12B、Llama3.1‑8B、Qwen2‑7B、Qwen2.5‑7B）进行平均准确率评估，发现大多数模型存在显著顺从偏差、波斯预训练反而降低推理能力、事实与情境表现相差约21%；

**⚠️ 局限性**

局限在于仅评测了7–12B规模模型，无法推断更大规模模型表现；手工编写数据可能偏向作者判断，缺乏地区/世代多样性，且未进行大规模众包验证或交叉标注。

---

## 439. Non-Trivial Zero-Knowledge Implies One-Way Functions

**arXiv ID:** 2602.17651 | [PDF](https://arxiv.org/pdf/2602.17651v1)

**作者:** Suvradip Chakraborty `[一作]` (Visa Research), Kabir Tomer `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

论文证明：若存在非平凡（即误差之和不趋近1）且在常数轮、公开币或非交互式零知识证明系统，则存在单向函数；在最坏情况下，即假设 ⊈ 或 ⊆，该结论仍成立。

**💡 创新点**

创新点：首次在高误差（错误可忽略到非可忽略）情形下获得单向函数的紧确性；将非平凡误差的零知识证明与单向函数关联，填补了先前仅对可忽略误差证明所知的空白；并给出了一种从高误差零知识到标准零知识的几乎无条件放大方法。

**🔧 技术方法**

技术手段：1) 对零知识证明进行“最大成功概率”攻击，构造恶意证明者通过逆向抽样（Universal Extrapolation）选择最优消息；2) 引入“最优概率估计器”并利用 Chernoff 边界控制误差；3) 通过多次重复与合成混合实验（hybrid）证明零知识误差对决定性算法的影响可被严格控制；4) 对私有币协议的处理通过辅助输入单向函数或转换为公开币协议的技术；5) 结合先前工作中从单向函数到低误差零知识的放大算法，实现几乎无条件放大。

**📊 数据集**

该研究为理论性质，未使用任何具体数据集；所有实验均为理论证明与抽象分析。

**📈 对比分析**

与现有方法的比较：传统方法仅在可忽略误差下即可得到单向函数，且需额外假设（如加密）；本工作在高误差下仅需最坏情况硬性假设 ⊈ 或 ⊆，在此基础上即可得到无条件或几乎无条件的零知识放大，性能提升在理论层面体现为更宽泛的误差容忍度和更低的假设复杂度。

**⚠️ 局限性**

局限性：1) 证明仅适用于常数轮公开币协议；对多轮或私有币协议需额外辅助输入单向函数；2) 对高误差的零知识协议仍需在某些参数范围内满足误差之和小于1；3) 对非交互式零知识的放大依赖于现有的单向函数到 NIZK 的构造，仍未实现完全无条件构造；4) 实际实现层面未给出具体协议实现细节。

---

## 440. Pushing the Frontier of Black-Box LVLM Attacks via Fine-Grained Detail Targeting

**arXiv ID:** 2602.17645 | [PDF](https://arxiv.org/pdf/2602.17645v1)

**作者:** Xiaohan Zhao `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 6346 | [OpenAlex ID](https://openalex.org/A5066530136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新的黑盒攻击框架——M-Attack V2，用以针对大型视觉-语言模型（LVLM）进行细粒度对抗攻击。

**💡 创新点**

创新点在于将局部匹配的梯度噪声通过三大模块进行去噪：多视图梯度平均（Multi‑Crop Alignment, MCA）、辅助目标对齐（Auxiliary Target Alignment, ATA）和块级动量（Patch Momentum, PM），并结合改进的模型集成（Patch Ensemble^+，PE^+）。

**🔧 技术方法**

采用的技术包括：随机裁剪与微小变换的多视图采样、梯度均值化、辅助图像语义聚类、动量缓冲回放、ViT 视觉编码器的梯度分析与降噪、以及多模型集成与精细化选择。

**📊 数据集**

使用的数据集：NIPS 2017 Adversarial Attacks and Defenses 数据集（清晰图像）以及 COCO 训练集用于构造辅助图像集合。测试目标为多款商业 LVLM：GPT‑4o、GPT‑5、Claude‑4.0、Gemini‑2.5‑Pro、GPT‑o3 等。

**📈 对比分析**

与现有黑盒攻击方法（AttackVLM、CWA、SSA‑CWA、AdvDiffVLM、FOA‑Attack 等）对比，M‑Attack V2 在 Claude‑4.0 上攻击成功率提升至 30%（↑22%），在 Gemini‑2.5‑Pro 上提升至 97%（↑14%），在 GPT‑5 上实现 100%（提升 2%），并在 KMR 指标上亦显著优于之前方法，证明其在前沿 LVLM 上的高效可迁移性。

**⚠️ 局限性**

局限性：① 需要较大的 surrogate 模型池和较多的裁剪样本，计算成本相对较高；② 目前仅针对视觉输入的攻击，未验证对全模态（含文本）任务的适用性；③ 对抗样本的可解释性与可部署性仍待进一步研究；④ 可能对某些已强化防御的 LVLM 失效。

---

## 441. FAMOSE: A ReAct Approach to Automated Feature Discovery

**arXiv ID:** 2602.17641 | [PDF](https://arxiv.org/pdf/2602.17641v1)

**作者:** Keith Burghardt `[一作]` (Amazon.com), Bo Li `[通讯]` (Amazon.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过构建 ReAct 代理 FAMOSE，自动化地在分类与回归任务中迭代生成、评估并优化特征。

**💡 创新点**

创新点在于将 ReAct 框架与逐步特征生成、评估、mRMR 选择以及 LLM 记录成功/失败特征相结合，实现自我纠错与持续改进。

**🔧 技术方法**

采用 AWS Bedrock/Deepseek-R1 LLM、ReAct 代理、Python 代码编译器、元数据与特征评估工具、mRMR 选择算法，并在 XGBoost、RandomForest、AutoGluon 等模型上验证。

**📊 数据集**

使用 20 个公开分类数据集和 7 个回归数据集，样本规模从数百到 10 万+，覆盖二分类、多分类和回归场景。

**📈 对比分析**

在 5 折交叉验证中与 AutoFeat、OpenFE、CAAFE、FeatLLM 等基线对比，FAMOSE 在 >10K 样本的分类任务平均提升 ROC‑AUC 0.23%，回归任务平均降低 RMSE 2%，且在多模型上表现稳健。

**⚠️ 局限性**

局限性包括高算力和 token 费用、对小型 LLM 效果差、依赖 LLM 预训练知识、未支持多标签分类、对任务不匹配的背景知识敏感。

---

## 442. What Makes a Good LLM Agent for Real-world Penetration Testing?

**arXiv ID:** 2602.17622 | [PDF](https://arxiv.org/pdf/2602.17622v1)

**作者:** Gelei Deng `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2839 | [OpenAlex ID](https://openalex.org/A5028270700)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于大型语言模型的渗透测试代理，解决了现有系统在工具能力和规划复杂度方面的缺陷，并在多层级评测中实现了显著提升。

**💡 创新点**

创新点在于引入任务难度评估（Task Difficulty Assessment, TDA）与基于证据的攻击树搜索（Evidence‑Guided Attack Tree Search, EGATS），以及一个工具与技能层与记忆子系统，分别对应两类失败模式。

**🔧 技术方法**

技术包括：typed 接口工具调用、检索增强生成（RAG）知识库、四维难度指标（Horizon, Evidence Confidence, Context Load, Historical Success）、UCB 采样的树搜索、外部记忆存储与上下文压缩。

**📊 数据集**

使用了三大评测数据集：XBOW（104 个 Web CTF 任务）、PentestGPT Benchmark（13 台 HTB/VulnHub 机器）和 GOAD（5 机多域 Active Directory 环境），以及 HTB Season 8 真实比赛机。

**📈 对比分析**

与五个基线系统在 GPT‑4o、GPT‑5、Gemini‑3‑Flash、Claude‑Sonnet‑4 等模型上对比，TDA‑EGATS 在 XBOW 上达 91%（比最佳基线提升 49%），在 PentestGPT Benchmark 上 12/13 台机器，GOAD 上 4/5 台主机，显示出明显的性能提升。

**⚠️ 局限性**

主要限制包括：对创意性新漏洞的发现能力不足、对主动防御与欺骗环境的鲁棒性有限，以及跨会话长期规划与持续性知识管理的挑战。

---

## 443. MARS: Margin-Aware Reward-Modeling with Self-Refinement

**arXiv ID:** 2602.17658 | [PDF](https://arxiv.org/pdf/2602.17658v1)

**作者:** Payel Bhattacharjee `[一作]` (University of Arizona), Ravi Tandon `[通讯]` (University of Arizona)

**通讯引用:** 3569 | [OpenAlex ID](https://openalex.org/A5004316408)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自我修正的边际感知奖励模型(MARS)，通过在低置信度（低边际）偏好对上集中生成合成样本来提升奖励模型的鲁棒性和对齐性能。

**💡 创新点**

创新点在于：① 将奖励模型的不确定性（边际）作为样本生成的主动指标；② 采用预算分配与Softmax权重结合的自适应增强策略；③ 在理论上证明低边际样本能提升损失函数平均曲率，从而改善模型条件化。

**🔧 技术方法**

技术核心包括：基于Bradley‑Terry模型的偏好学习；线性/可微奖励网络（如T5/DeBERTa）；自适应样本分配（Softmax权重）；合成偏好生成（基于T5的文本改写或SimCSE等）；对损失曲率的Fisher信息分析。

**📊 数据集**

使用三大公开偏好数据集：Anthropic HH‑RLHF、UltraFeedback、PKU‑SelfRLHF；奖励模型采用DeBERTa‑v3‑base，并在T5‑base上做合成样本；下游对齐通过PPO训练TinyLlama‑1.1B和Llama‑3.2‑1B，在Qwen2.5‑3B‑Instruct评判。

**📈 对比分析**

对比方法包括：无增强、均匀增强、Best‑of‑N、West‑of‑N；评价指标包括pairwise accuracy、margin SNR、aligned policy win‑rate。实验显示MARS在所有指标上均优于对比方法，尤其在低容量模型的win‑rate提升显著，SNR和pairwise accuracy提升约10‑15%。

**⚠️ 局限性**

局限性：① 需要额外的合成偏好生成成本；② 预算分配参数τ和B_t的选择对结果敏感；③ 在极少量标注数据或高度噪声数据下，低边际样本可能不够代表真实难点；④ 主要验证于语言模型对齐任务，是否适用于其他任务尚待进一步评估。

---

## 444. SMAC: Score-Matched Actor-Critics for Robust Offline-to-Online Transfer

**arXiv ID:** 2602.17632 | [PDF](https://arxiv.org/pdf/2602.17632v1)

**作者:** Nathan S. de Lara `[一作]` (University of Toronto), Florian Shkurti `[通讯]` (University of Toronto)

**通讯引用:** 1547 | [OpenAlex ID](https://openalex.org/A5010648258)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文提出一种名为 SMAC 的离线强化学习算法，旨在训练的演员-评判器能够在后续的在线强化学习过程中平滑过渡，避免出现即时性能下降。

**💡 创新点**

创新点包括：① 在离线阶段对 Q 函数施加“分数匹配”正则化，使其动作梯度与数据集策略的分数（score）相匹配，从而降低对离线数据分布之外动作的误判；② 使用 Muon 优化器取代 Adam，以获得更平坦、可迁移的解；③ 通过最大熵理论的精确等式证明正则化的理论依据。

**🔧 技术方法**

主要技术包括：最大熵强化学习框架、分数匹配正则化（Score Matching）、Diffusion 模型用于估计数据集的动作分数、Muont 优化器、以及基于 SAC/TD3 的在线细调。

**📊 数据集**

实验基准使用了六个 D4RL 环境（例如 Kitchen、Door、Hopper 等），数据集为公开的离线演示数据。

**📈 对比分析**

与 CalQL、IQL、TD3+BC 等现有离线 RL 基线进行比较；SMAC 在所有六个任务中实现了无性能崩溃的在线细调，并在 4/6 环境中将在线回报差异降低 34%–58%，在最终性能上普遍优于对照组。

**⚠️ 局限性**

局限性包括：需要预训练高精度的 Diffusion 模型，带来较高的前期计算成本；分数匹配正则化依赖于对数据集分数的估计，若估计不准确可能影响效果；在部分环境下，加入 BC 约束仍会导致长期性能下降。

---

## 445. Sink-Aware Pruning for Diffusion Language Models

**arXiv ID:** 2602.17664 | [PDF](https://arxiv.org/pdf/2602.17664v1)

**作者:** Aidar Myrzakhan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6346 | [OpenAlex ID](https://openalex.org/A5066530136)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对扩散式语言模型的剪枝策略，自动识别并剪除在整个去噪轨迹中不稳定的注意力sink，从而降低推理成本

**💡 创新点**

创新点在于发现DLM中的注意力sink高度不稳定，并基于sink方差量化其不稳定性，利用这一指标对剪枝重要性进行动态调节，避免传统AR模型中“始终保留sink”的误用

**🔧 技术方法**

采用注意力sink方差统计、sink掩蔽激活、soft‑sink评分以及与现有剪枝基线（Wanda、SparseGPT）结合的权重重构与重要性评估技术

**📊 数据集**

使用WikiText‑2作为校准集；在多款模型（LLaDA、Dream、MMaDA、LLaDA‑1.5）上评测8个标准语言基准（MMLU、ARC‑C、PIQA、WinoGrande、HellaSwag、RACE、GSM8K、GPQA）

**📈 对比分析**

与Wanda、SparseGPT、幅度剪枝等基线在相同稀疏度（25%、50%、75%）下进行对比，结果显示sink‑aware剪枝在中高稀疏度下能保持或提升模型性能，且在整体质量‑效率比上更优

**⚠️ 局限性**

局限性包括：sink统计基于固定校准分布，可能在分布漂移时失效；未结合微调或后期自适应；仅评估了部分多模态与长上下文场景，未来需进一步验证层级自适应与量化协同效果

---

## 446. Differences in Typological Alignment in Language Models' Treatment of Differential Argument Marking

**arXiv ID:** 2602.17653 | [PDF](https://arxiv.org/pdf/2602.17653v1)

**作者:** Iskar Deng `[一作]` (University of Washington), Shane Steinert-Threlkeld `[通讯]` (University of Washington)

**通讯引用:** 561 | [OpenAlex ID](https://openalex.org/A5017484646)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在英文SVO句子中注入18种不同的差异化论元标记（DAM）规则，训练GPT‑2‑small模型并使用最小对照评估其规则掌握情况。

**💡 创新点**

创新点在于将语义许可系统DAM应用于人工语料，通过实验揭示模型在标记方向上的人类语言一致性，但在论元偏好上与人类语言存在偏差。

**🔧 技术方法**

采用自监督的next‑token预训练GPT‑2‑small，并通过最小对照、标记位置测试以及语义探测等技术评估模型学习效果。

**📊 数据集**

使用OpenSubtitles英文子集（约184 M tokens，21 M句子），通过spaCy与Benepar进行解析并利用BERT分类器标注语义属性后注入标记。

**📈 对比分析**

在不同语义触发、依赖复杂度、标记方向和论元目标的18种条件下，模型在自然标记方向上平均准确率约为85%，逆向约为68%；但在论元偏好上未表现出人类语言中普遍的对象偏好。

**⚠️ 局限性**

局限性包括模型规模有限、仅使用单一随机种子、仅针对英文SVO句子、未控制不同规则的触发频率、未覆盖更复杂句型或跨语言验证。

---

## 447. IntRec: Intent-based Retrieval with Contrastive Refinement

**arXiv ID:** 2602.17639 | [PDF](https://arxiv.org/pdf/2602.17639v1)

**作者:** Pourya Shamsolmoali `[一作]` (University of York), Yue Lu `[通讯]` (East China Normal University)

**通讯引用:** 11541 | [OpenAlex ID](https://openalex.org/A5100334845)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种交互式、状态化的开放词汇物体定位框架IntRec，通过用户反馈不断更新意图状态，实现细粒度歧义消解。

**💡 创新点**

创新点在于引入意图状态(ISt)作为正负示例集合，结合对比式排名函数（最大正相似度减去负惩罚），实现单轮反馈即可显著提升检索准确率。

**🔧 技术方法**

使用CLIP文本/图像编码器、CenterNet2产生候选框、对比排名与状态更新算法，以及冻结的ViT-B/16视觉模型。

**📊 数据集**

在LVIS、Objects365、COCO三大公开数据集上进行评估，并构造LVIS‑Ambiguous细粒度歧义子集进行专门测试。

**📈 对比分析**

与现有多种最先进开放词汇检测器（如OVL‑ViT、Grounding DINO、CoDet等）对比，IntRec在Turn‑0即达到或超过前沿，Turn‑1后AP(r)提升约7.9点，整体AP及稀有类性能显著优于基线。

**⚠️ 局限性**

局限在于依赖初始候选框；若检测器未生成目标框，交互更新无法弥补，未来需引入基于反馈的候选框细化机制。

---

## 448. When to Trust the Cheap Check: Weak and Strong Verification for Reasoning

**arXiv ID:** 2602.17633 | [PDF](https://arxiv.org/pdf/2602.17633v1)

**作者:** Shayan Kiyani `[一作]` (University of Pennsylvania), Hamed Hassani `[通讯]` (University of Pennsylvania)

**通讯引用:** 2356 | [OpenAlex ID](https://openalex.org/A5059354479)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了弱-强验证框架与算法SSV，动态决定何时依赖弱验证、何时调用强验证以保证推理结果可靠；

**💡 创新点**

创新点在于：①将弱与强验证整合为两阈值策略并在在线分布无假设下控制I/II错误；②引入随机化强验证探索以获得无偏反馈；③给出理论误差界限与经验 Pareto 前沿；

**🔧 技术方法**

主要技术包括：阈值更新（在线量化跟踪）、重要性加权随机探索、分布式无偏误差控制与理论证明；

**📊 数据集**

实验数据集包括数学推理的MATH（不同难度级别）与逐步推理的Sudoku；

**📈 对比分析**

与强检验（Oracle）与弱检验（Greedy）基线对比，SSV在接近Oracle的准确率同时将强验证调用量降低约30‑50%，满足指定错误目标；

**⚠️ 局限性**

局限在于阈值仅依据弱得分而非完整上下文，导致仅得到边际错误控制；未来工作需考虑上下文依赖的动态阈值与更细粒度的错误监测。

---

## 449. CLEF HIPE-2026: Evaluating Accurate and Efficient Person-Place Relation Extraction from Multilingual Historical Texts

**arXiv ID:** 2602.17663 | [PDF](https://arxiv.org/pdf/2602.17663v1)

**作者:** Juri Opitz `[一作]` (University of Zurich), Simon Clematide `[通讯]` (University of Zurich)

**通讯引用:** 1556 | [OpenAlex ID](https://openalex.org/A5073027507)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估多语言历史文本中人物‑地点关系提取任务，提出新的时间与地点关系类型（at 与 isAt）

**💡 创新点**

首次在历史多语料上同时考察关系准确性、推理效率与跨域泛化，并采用宏召回与资源消耗相结合的评估框架

**🔧 技术方法**

使用大型语言模型（如 GPT‑4o）与传统分类器，结合可解释推理和自监督训练

**📊 数据集**

HIPE‑2026 数据集，包括法语、德语、英语、卢森堡语报纸文章及16–18 世纪法语文学的惊喜测试集

**📈 对比分析**

通过宏召回对比多系统，GPT‑4o 在 at 关系上最高可达 0.8 召回率，isAt 召回率更低；整体表现受噪声与推理成本影响

**⚠️ 局限性**

受限于 OCR 噪声、低资源语言、时间推理复杂性以及大模型推理开销，难以在规模化场景下高效执行

---

## 450. Human-level 3D shape perception emerges from multi-view learning

**arXiv ID:** 2602.17650 | [PDF](https://arxiv.org/pdf/2602.17650v1)

**作者:** Tyler Bonnen `[一作]` (University of California), Angjoo Kanazawa `[通讯]` (University of California)

**通讯引用:** 15147 | [OpenAlex ID](https://openalex.org/A5031491881)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于多视角学习的神经网络框架，能够在不进行任务特定训练的情况下，以零样本方式完成3D形状推断任务，并与人类表现相匹配。

**💡 创新点**

首次将视觉-空间自监督学习与人类3D感知行为零样本比较，展示模型能够自然产生与人类错误模式和反应时对应的内部不确定性，并实现人类水平的准确率。

**🔧 技术方法**

采用多视角视觉Transformer（如VGGT‑1B）与像素级深度与不确定性损失的自监督预训练目标，并利用不确定性阈值和层级解算器进行零样本评估。

**📊 数据集**

使用大规模多视角自然场景数据进行预训练，实验采用公开的3D感知基准（包含真实物体与程序生成的抽象形状），并收集300+人类参与者的在线试验数据。

**📈 对比分析**

通过基于模型不确定性选择奇异物体的零样本评估指标与人类行为直接对比；VGGT平均正则化准确率83%，与人类78.9%无显著差异，单视图模型仅28%。

**⚠️ 局限性**

模型缺乏生理级别的自我运动与视差信号，仅采用全局坐标；解算层指标仅反映单向前馈深度，未涵盖人类眼动与循环动态；对非自然视觉环境的泛化仍待验证。

---

## 451. Multi-Round Human-AI Collaboration with User-Specified Requirements

**arXiv ID:** 2602.17646 | [PDF](https://arxiv.org/pdf/2602.17646v1)

**作者:** Sima Noorani `[一作]` (University of Pennsylvania), George Pappas `[通讯]` (University of Pennsylvania)

**通讯引用:** 35328 | [OpenAlex ID](https://openalex.org/A5029243115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种多轮人机协作框架，利用用户自定义的“对因果伤害（counterfactual harm）”和“互补性（complementarity）”规则来约束AI的预测集合，确保AI既不破坏人类已正确判断，又能在其错误时补救。

**💡 创新点**

创新点在于：①把协作约束转化为可验证的规则；②设计分布无关、无模型假设的在线阈值校准算法；③在多轮对话中实时更新阈值，保证累计误差在用户设定的容忍度内；④通过规则灵活调节协作动态。

**🔧 技术方法**

主要技术包括：规则驱动的约束定义、非合形性得分s（如1–p_t,r(y)），基于阈值的预测集合构造，梯度式在线阈值更新（τ_{t+1}=max{0,τ_t+η(E^CH_t-ε)}等），以及对多轮对话转录的在线评估。

**📊 数据集**

实验数据集：医疗诊断使用DDXPlus合成病历；形状计数任务采用自行构造的图片集，参与者在Prolific平台上完成。

**📈 对比分析**

对比方法：在LLM仿真和真实人机实验中，算法通过调节ε、δ达到指定误差率；实验显示误差率稳定在目标附近，且更严格的对因果伤害/互补性约束可显著降低人类错误率或提升恢复率，表现优于未加约束或随机阈值。

**⚠️ 局限性**

局限性：仅使用预测集合来传递不确定性，对开放式输出或非结构化任务可能不适用；需要人类提供集合预测，若人类输出不符合集合形式则难以应用；算法对规则设定的依赖性高，若规则设计不当可能导致协作效果不佳。

---

## 452. CORAL: Correspondence Alignment for Improved Virtual Try-On

**arXiv ID:** 2602.17636 | [PDF](https://arxiv.org/pdf/2602.17636v1)

**作者:** Jiyoung Kim `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 CORAL 框架，用 Diffusion Transformer 对人像和衣物之间的对应关系进行显式对齐，从而提升虚拟试衣的细节保真度和整体效果。

**💡 创新点**

创新点在于将基于 DINOv3 的稠密对应信息通过对应蒸馏损失和熵最小化损失注入到 3D Attention 的 query‑key 机制中，实现了人像–衣物之间精准且锐化的匹配；同时首次通过 VLM‑based 评估指标量化了衣物转移、一致性与姿态自然度。

**🔧 技术方法**

核心技术包括：Latent Diffusion Model、Diffusion Transformer（DiT）的全 3D Attention、VAE 编码、DINOv3 用作外部对应参考、对应蒸馏损失、熵最小化损失、RoPE 位置编码的姿态注入。

**📊 数据集**

使用了 VITON‑HD、DressCode 以及 PPR10K 等公开试衣基准；此外还在实景人像–衣物转移任务中构建了挑战性数据集，验证零样本推广能力。

**📈 对比分析**

与现有方法在 SSIM、LPIPS、FID、KID 等传统指标以及 GTC、TAC、FPC 等 VLM‑based 指标上进行对比，CORAL 在所有指标上均实现了 state‑of‑the‑art，尤其在细节保持和姿态一致性方面提升显著。

**⚠️ 局限性**

主要局限包括：对 DINOv3 稠密对应的依赖，若对应质量不高可能影响蒸馏效果；对极端姿态、遮挡或非典型衣物形状的鲁棒性仍有提升空间。

---

## 453. Catastrophic Forgetting Resilient One-Shot Incremental Federated Learning

**arXiv ID:** 2602.17625 | [PDF](https://arxiv.org/pdf/2602.17625v1)

**作者:** Obaidullah Zaland `[一作]` (Umeå University), Monowar Bhuyan `[通讯]` (Umeå University)

**通讯引用:** 3258 | [OpenAlex ID](https://openalex.org/A5044933320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了OSI-FL，一种在仅一次客户端‑服务器通信中实现增量联邦学习的方法，利用预训练的视觉语言模型生成类别级嵌入，服务器端用扩散模型合成类似客户端分布的新样本，并通过选择性样本保留（SSR）缓解灾难性遗忘；

**💡 创新点**

首次将单轮通信（one‑shot）与增量学习相结合，创新性地用类别级嵌入替代模型梯度传输，结合生成式数据再现和精细的样本保留策略；

**🔧 技术方法**

使用预训练的GPT‑ViT和CLIP进行文本描述与嵌入，使用分类器无关扩散模型（classifier‑free guidance）进行数据合成，采用梯度幅值选择的重要样本做SSR，训练采用ResNet‑18特征提取器；

**📊 数据集**

在三大公开数据集上验证：NICO++ Common、NICO++ Unique 和 OpenImage，分别构建类别增量和域增量实验；

**📈 对比分析**

与传统联邦学习（FedAvg、FedProx、FedEWC、FedET、FedIL+）以及现有单轮联邦学习（OSCAR、OSCAR‑IL、OSCAR‑R）进行对比；实验表明OSI‑FL在类增量和域增量设置下均优于增量FL和OSCAR变体，接近最优（ceiling）性能，并在通信与显存占用上保持优势；

**⚠️ 局限性**

主要局限包括：SSR使用基于梯度幅值的简单策略，缺乏多样性保证；对预训练模型依赖度高，可能导致生成质量受限；在极大客户端数量或极差分布下的鲁棒性尚待进一步评估。

---

