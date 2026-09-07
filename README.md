# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-07 | 今日论文总数: 550

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Spectral-Target Physical Latent Structuring for JEPA-Style World Models

**arXiv ID:** 2609.04264 | [PDF](https://arxiv.org/pdf/2609.04264v1)

**作者:** Penghao Zhu `[一作]` (Kaliber Labs), Aneesh Jonelagadda `[通讯]` (Kaliber Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并解决在 Latent World Model（如 LeWM）中出现的物理表示懒惰问题，提出一种在训练阶段使用的 Fourier 频谱辅助头（Fourier Auxiliary Head）来强化 latent 空间对关键物理属性的保持，并在多种模拟环境中验证其效果。

**💡 创新点**

创新点在于：① 将可交换对象的定向包围盒及语义掩码映射到 Fourier 频域，并通过求和获得一个 permutation‑invariant 的谱化目标；② 该目标在训练时作为辅助监督，能够直接引导 encoder 与 predictor 关注物理相关信息，从而有效缓解 representation laziness；③ 该方法轻量、无推理时开销，且可迁移到任意 JEPAs 风格的世界模型。

**🔧 技术方法**

使用了 Joint‑Embedding Predictive Architecture（LeWM）与 SIGReg / VISReg 正则化；设计了 Fourier 频谱辅助头并用 2 层 MLP 进行预测；在规划时采用 Cross‑Entropy Method（CEM）在 latent 空间内优化动作序列。

**📊 数据集**

实验数据集主要来自模拟环境：n‑ball（1、3、6 球）、Push‑T、Two‑Room、OG‑Bench Cube；训练样本量从 10k 到 500k 不等。

**📈 对比分析**

通过与 LeWM 基线在规划成功率和 Spearman ρ（latent‑to‑pixel 相关性）两项指标进行对比，发现 Fourier 辅助头在所有环境均提升规划成功率，特别是在低数据量（≤250k）下提升幅度显著；同时提升了 latent 对关键物理属性的相关性。

**⚠️ 局限性**

局限性包括：① 仅在模拟环境验证，缺乏真实场景实验；② 仅以 LeWM 为基线，未在其他 JEPAs 结构上测试；③ 对边界框或检测器的依赖，若标注质量差会影响辅助目标；④ Fourier 频谱维度选择需经验调优，过高可能导致训练不稳定。

---

## 2. SAGE: Semantic Attribute Graphs for Multi-Entity Visual Retrieval

**arXiv ID:** 2609.04255 | [PDF](https://arxiv.org/pdf/2609.04255v1)

**作者:** Yongjoo Kim `[一作]` (Korea University), Jungbeom Lee `[通讯]` (Korea University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对视觉稠密文档检索的语义实体图检索框架SAGE以及新基准Dense Entity-Attribute Retrieval。

**💡 创新点**

将稠密文档中的实体和属性建模为层次图，使用多向量嵌入避免语义稀释，并通过迭代子图检索实现多实体对比推理。

**🔧 技术方法**

使用冻结的VLM、OCR布局解析、LLM驱动的属性分组、多向量编码、两阶段粗细检索和检索后检验列表迭代等技术。

**📊 数据集**

自制的1,055条问答对的Dense Entity-Attribute Retrieval数据集，采集自产品详情页，并在UniDoc-Bench和M3DocVQA上做泛化测试。

**📈 对比分析**

与ColPali、ColQwen等基准以及OCR、Caption检索对比，Recall@3提升至0.849，生成质量得分2.746，并在其它数据集上同样显著提升。

**⚠️ 局限性**

主要来自对产品页的依赖、OCR/标签误差导致的属性丢失、图构建开销以及对科学/医学等专用符号的适用性不足。

---

## 3. ResLearn-XR: Residual Learning for Network Traffic and Quality-of-Experience-Aware Modeling in Extended Reality

**arXiv ID:** 2609.04493 | [PDF](https://arxiv.org/pdf/2609.04493v1)

**作者:** Yoga Suhas Kuruba Manjunath `[一作]` (Carleton University), Lian Zhao `[通讯]` (Toronto Metropolitan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了ResLearn-XR框架，利用两阶段残差学习同时预测XR网络流量并估计QoE风险；

**💡 创新点**

创新点在于：①结合值空间和对数空间残差校正来提升对突发流量的预测精度与对QoE风险的判别；②设计了可在加密流量下使用的因果DDA特征构造方法；③公开了首个同时包含XR流量与会话级QoE标签的XR Traffic-QoE数据集；

**🔧 技术方法**

技术包括Transformer、LSTM/GRU/Stacked LSTM基础网络，残差学习模块，DDA特征提取，二分类logit残差校正，分层等距回归校准等；

**📊 数据集**

使用三类数据集：①内部XR Traffic-QoE数据集（包含多种XR应用与不同带宽），②公开的SteamVR traces，③其他XR流量数据集；

**📈 对比分析**

与Informer、FEDformer、Temporal Fusion Transformer等基线以及传统机器学习方法相比，ResLearn-XR在SMAPE、RMSE、AUC、ECE等指标均取得显著提升，最大可达17.8% SMAPE降低和87.8% SMAPE提升，AUC提升至0.93，表明性能优越；

**⚠️ 局限性**

局限包括仅有两名受试者的弱监督QoE标签，缺乏时序细粒度的QoE反馈，且所有标签均为二元化，未来需扩大样本量、完善标签粒度并验证闭环控制效果。

---

## 4. MaxKernel: Agentic Kernel Generation for TPUs

**arXiv ID:** 2609.04523 | [PDF](https://arxiv.org/pdf/2609.04523v1)

**作者:** Shangkun Wang `[一作]` (Google), Sethu Sankaran `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个多智能体框架 MaxKernel，用于自动生成和优化 TPU 上的 Pallas 核心，减少人工调优工作。

**💡 创新点**

创新点在于将 LLM 与闭环编译反馈、实时 XProf 性能分析和图搜索相结合，支持人机交互、单轨循环优化和图式搜索三种执行模式。

**🔧 技术方法**

使用 Gemini 3.1 Pro LLM、检索增强生成（RAG）知识库、XProf 跟踪、Beam/Parallel 搜索算法、自动化编译/调试子智能体。

**📊 数据集**

评测数据集为 JaxBench（50 个 TPU 加速工作负载）以及多种最新开源模型的 OSS kernel（MLA、Qwen3-Next GDN、DeepSeek‑V4 等）。

**📈 对比分析**

与 XLA 基线和人工手工优化的 Pallas 进行对比，Auto 迭代可达 1.58× 的几何平均加速，Parallel 及 Beam 在 8 个关键任务中分别达到 2.32× 与 1.78× 的加速，甚至在部分工作负载上超过人类专家。

**⚠️ 局限性**

局限性包括仅使用静态知识库、仅评估单一 LLM（Gemini）、未探索更复杂的搜索算法和动态知识更新，以及对其他加速器的通用性不足。

---

## 5. When Seeing Overrides Knowing: Visual Dominance and Deferral-Based Method for Personalized Safety in VLMs

**arXiv ID:** 2609.04281 | [PDF](https://arxiv.org/pdf/2609.04281v1)

**作者:** Edward Sun `[一作]` (University of California), Aylin Caliskan `[通讯]` (University of Washington)

**通讯引用:** 4706 | [OpenAlex ID](https://openalex.org/A5101545719)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MPS-Bench基准，用以评估视觉-语言模型在缺失用户背景信息时的个性化安全性，并发现现有顶尖VLM普遍不进行询问、直接回答且易受视觉支配导致不安全；提出PRISM轻量级输入监控器，利用双向跨模态调制预测何时需要延迟以获取更多用户上下文。

**💡 创新点**

①首个针对VLM的个性化安全基准；②发现并机制化阐述“视觉主导”现象；③设计PRISM通过双向调制在融合前增强安全信号，显著提升安全-效用Pareto前沿。

**🔧 技术方法**

基于多模态注意力的激活补丁与因果推断分析、跨模态特征提取（CLIP/EVA-CLIP/SigLIP/ Qwen3-VL），双向MLP调制、风险预测头（Sigmoid+残差）。

**📊 数据集**

MPS-Bench：5,181个场景，584张真实图片，12高危领域（健康、金融、护理等），每张图生成多条高风险查询及隐藏的用户档案。

**📈 对比分析**

对8种前沿VLM（GPT‑5、Gemma‑3、Qwen3‑VL、InternVL‑S1、Pixtral‑Large等）进行评估；未提供用户上下文时安全分数均≤2.6/5；PRISM在保持低误报的同时实现AUC≈0.978，且在所有模型与领域的安全-效用Pareto前沿上严格占优。

**⚠️ 局限性**

缺乏对多模态安全失败的完整解释，PRISM仍依赖外部特征提取器且需要额外训练，无法直接处理闭源VLM内部状态；基准样本虽真实但规模有限，且未覆盖全部可能的视觉-文本冲突场景。

---

## 6. A Data Fusion Framework for Grounding Aerospace Surrogate Model via Experimental Wind-Tunnel Observations

**arXiv ID:** 2609.04267 | [PDF](https://arxiv.org/pdf/2609.04267v1)

**作者:** Nitin Nagesh Kulkarni `[一作]` (Luminary AI), Juan J. Alonso `[通讯]` (Luminary AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出了一种实验基础的修正框架，将高保真CFD训练得到的深度学习代理（Geotransolver）通过风洞PSP测量数据进行校正，保持了原有代理的泛化和计算效率；

**💡 创新点**

创新点在于：①先预训练大规模CFD数据得到的代理被冻结，只在极少的实验数据上训练一个轻量级校正网络，完成CFD到实验的系统性偏差校正；②通过空间注册和多保真度融合，使实验数据能有效校正代理而不需要大规模实验；

**🔧 技术方法**

使用技术包括：Geotransolver神经算子（Geometry‑Aware Latent Embedding）、轻量级多层感知机校正头、PSP测量的空间注册、数据融合与多保真度校正；

**📊 数据集**

数据集为：SHIFT‑Wing数据集（2300个NASA CRM翼体的高保真CFD模拟，Mach 0.70–0.85，攻角0–4°）以及对应的风洞PSP压力感知绘图数据（Mach 0.70与0.85，各9个攻角），测试时保留1.5°和3.0°攻角；

**📈 对比分析**

比较方法：先与CFD基线、原始PSP、注册后PSP及简单插值基线进行对比，评估积分力和表面压力分布的MAE、RMSE、归一化误差；性能表现为：积分负载与CFD保持>0.99相关，校正后表面压力在保留条件下误差为2.3–2.7%，且在所有测试状态下均优于插值基线；

**⚠️ 局限性**

局限性：实验数据量有限，校正网络无法明确分辨湍流、激波位置、机翼变形等具体误差来源；对模型在未测量飞行条件下的泛化能力仍有不确定性；仅针对NASA CRM翼体验证，未验证在其他几何或更宽范围的适用性。

---

## 7. TRILOGUE: A Trilingual Spoken Dialogue Fact-Checking Benchmark with Evidence and Paired Audio

**arXiv ID:** 2609.04452 | [PDF](https://arxiv.org/pdf/2609.04452v1)

**作者:** Chaewan Chun `[一作]` (Pennsylvania State University), Dongwon Lee `[通讯]` (Pennsylvania State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多语音、多语言、基于对话的事实核查基准TRILOGUE，包含英语、俄语和哈萨克语的近12K个播客式对话，配有音频、ASR转录、词级时间对齐以及逐回合可检验性标签与句子级证据。

**💡 创新点**

创新点在于：①首次提供大规模三语音频+文本对齐的对话式事实核查数据；②对话生成与标签标注均基于源文章与对话重写，保证可检验性；③设计了三任务评测（可核查性检测、源文章证据检索、主张验证）并引入ASR噪声对照，系统性分析跨语言、跨声学的挑战。

**🔧 技术方法**

采用大型语言模型（Gemini、DeepSeek、OpenAI o4-mini）进行对话生成和注释；使用多模态技术：WhisperX 对齐、Whisper 语音识别、TTS（Inworld、Azure Speech、ESPnet）合成；基准评测使用TF-IDF+LR、XLM-R、mDeBERTa、BM25、E5、mDeBERTa reranker 等。

**📊 数据集**

使用的核心数据集是TRILOGUE本身（11,957 对话，187,544 语句，390.3 小时音频），并基于已有的多语种新闻来源（Nur、Forbes、Tengrinews、Kapital）构建源文章与伪造改写。

**📈 对比分析**

在单语、跨语、留一语言实验中，mDeBERTa在可核查性检测上达到 90–92% F1，检索 Hit@1 约 74–86%，检索 Recall@5 约 86–87%；在主张验证上，Gold 证据条件下 F1_False 接近 90%，检索证据条件下约 80–88%。ASR噪声导致俄语下降 3–4 分，哈萨克语下降 7–12 分，表明语音识别是瓶颈。

**⚠️ 局限性**

局限性包括：①数据为生成对话而非真实对话，缺乏自然社交媒体语境；②仅做源文章内部检索，未覆盖开放域源发现和多源冲突；③哈萨克语 TTS 与 ASR 质量低下，限制了跨语言泛化；④基准仅使用 ASR 文本输入，未充分利用音频和时间对齐，忽视了声学、说话人、韵律信息。

---

## 8. Corporate-Family Resolution Is Not a String-Matching Problem: A Public Benchmark Stratified by Name Visibility

**arXiv ID:** 2609.04269 | [PDF](https://arxiv.org/pdf/2609.04269v1)

**作者:** Harshit Gupta `[一作]` `[通讯]` (Independent Researcher), Harshit Gupta (Independent Researcher)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并公开了 CorpFam 基准，用于评估供应商记录之间的企业家族关系（即同一母公司下的不同子公司、分部等）是否属于同一企业家族。

**💡 创新点**

创新点包括：①引入“名称可见性”三层分层评估，揭示传统整体 F1 隐藏的“不可见”层严重失效；②证明阻塞阶段是瓶颈，所有传统基于名称的阻塞器在该层几乎无候选；③提供完整的公开标注、负样本策略、分层拆分以及 SEC Exhibit 21 的独立验证，形成可复现且可信的基准。

**🔧 技术方法**

采用了多种匹配与阻塞技术：字符串相似度（精确匹配、Jaccard、字符 3-gram TF‑IDF、句子嵌入 Cosine）、监督逻辑回归属性模型（地址、地理、电话等）、以及多种阻塞器（token、q‑gram、sorted‑neighbourhood、phonetic、属性键、语义最近邻）。评估时使用 per‑stratum recall、F1 及全局 AP。

**📊 数据集**

数据集来自美国联邦合同授予记录（FY2025）中自报的 UEI 与母公司 UEI，构成真实标签；同时通过 SEC Exhibit 21 复核文件对“不可见”层的链接进行外部验证。

**📈 对比分析**

实验结果显示，最强匹配器在“不可见”层的召回率仅约0.5%（最高0.8%），整体 F1 为0.38；所有阻塞器在该层的召回率低于3%，即使联合使用也仅能覆盖约4% 的真实链接；相比传统基准几乎无提升。

**⚠️ 局限性**

局限性包括：仅覆盖单一司法辖区与财政年度；父公司关系仅为一跳，未覆盖多层持股链；大多数父母缺乏地址等属性导致属性模型难以发挥作用；验证仅限 SEC 上市公司；负样本采样策略可能导致偏差。

---

## 9. FailureSpot: Label-Efficient Timestamp-Level Failure Detection for Vision-Language-Action Models

**arXiv ID:** 2609.04277 | [PDF](https://arxiv.org/pdf/2609.04277v1)

**作者:** Jie Ma `[一作]` (Wayne State University), Yi Zhu `[通讯]` (Wayne State University)

**通讯引用:** 4555 | [OpenAlex ID](https://openalex.org/A5076130185)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 FailureSpot 框架，利用动作块的一致性与幅度信号实现对视觉‑语言‑动作（VLA）策略的时间级故障检测，并通过主动学习减少标注成本。

**💡 创新点**

创新点在于：①将动作块的短期一致性与幅度作为弱监督信号，①1预训练检测器；②采用主动学习挑选最不确定轨迹进行稀疏标注，显著提升时间级与轨迹级检测性能。

**🔧 技术方法**

技术手段包括：动作一致性/幅度信号提取、指数移动平均归一化、轻量级 MLP/LSTM 失败检测器、主动学习选择、形态预测阈值与置信区间。

**📊 数据集**

使用 LIBERO‑10 长周期操纵任务集，对 π0、π0‑FAST 与 OpenVLA 三种 VLA 策略进行评估。

**📈 对比分析**

与 SAFE、STAC、ActProbe、LogpZO 等基线比较，FailureSpot 在 15% 时间级标注预算下，时间级 AUROC 与轨迹级 AUROC 均位列前列，且在多种 VLA 上表现更稳健。

**⚠️ 局限性**

局限性：对仅产生单动作的 VLA（如 OpenVLA）缺乏足够的短期一致性信息，导致弱监督效果弱；同时依赖动作块可观测性，难以直接迁移到非块化策略。

---

## 10. Computing Lewis Weights to High Precision by Fixed-Point Iteration

**arXiv ID:** 2609.04338 | [PDF](https://arxiv.org/pdf/2609.04338v1)

**作者:** Swati Padmanabhan `[一作]` `[通讯]` (University of Minnesota), Swati Padmanabhan (University of Minnesota)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并分析了对任意 p>2 的 ℓ_p‑Lewis 权重的直接固定点迭代，并给出了高精度求解的算法与迭代次数上界。

**💡 创新点**

创新点在于证明该直接迭代对所有 p>2 都收敛、KL 整体收敛率为 1-2/p，并通过体积采样与熵独立性、局部线性化给出收敛解释；同时获得了比现有方法更好的 O(p·log p·√m /ε) 迭代次数上界。

**🔧 技术方法**

主要技术包括固定点分析、KL 散度收敛证明、体积采样与熵独立性、局部雅可比矩阵谱分析，以及利用精确杠杆得分的计算。

**📊 数据集**

实验使用合成高斯矩阵、结构化 leverage 异质矩阵、极端病态矩阵，以及真实数据集：Intel Berkeley Research Lab 传感器数据和 NOAA National Data Buoy Center 气象数据。

**📈 对比分析**

与之前基于优化的高精度方法比较，所提出的直接迭代在迭代次数上表现为 O(p) 线性增长，在大多数条件下迭代次数与 p 成线性关系；在块权重实验中也验证了相同趋势，并在有限 p 的块设计实验中展示了从均匀到 D‑optimal 的插值效果。

**⚠️ 局限性**

局限性包括需要在每一步计算完整杠杆得分（每步 O(mn^{ω-1}) 复杂度）、未给出比特复杂度与浮点数稳定性分析、对极端病态情况存在精度下限，以及对无零行、满列秩矩阵的假设。

---

## 11. Physics-Direct FPGA Tooth-Contact Computation for Deterministic Gear Digital Twins

**arXiv ID:** 2609.04248 | [PDF](https://arxiv.org/pdf/2609.04248v1)

**作者:** Jiacheng Miao `[一作]` `[通讯]`, Jiacheng Miao

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

在 Xilinx XC7K480T 现场可编程门阵列上实现并验证了一个完全固定点的、无浮点 IP 的齿轮接触计算流水线，能够在毫秒级、零抖动的实时周期内输出接触点的圆椭圆半轴和峰值压力（物理直接预测）。

**💡 创新点**

创新点包括：① 采用物理直接（physics‑direct）闭式方程压缩整个接触问题，而非传统的机器学习 surrogate；② 通过严格的硬件可分区原则，将可并行、分支自由的几何与赫兹计算移植到 FPGA，保留了 LCP 的全局求解在 CPU 上；③ 采用完全开源、无供应商工具链（openXC7）完成从 RTL 到位图、JTAG 读回的完整硅验证；④ 在硬件中实现了分支无关的三段流水线（几何‑曲率‑赫兹），实现固定周期 806 个时钟、σ=0 的确定性延迟。

**🔧 技术方法**

使用的技术包括：固定点 Q(24,16) 坐标、Q1.19 单位法向、Q(32,20) 主曲率、Q(32,8) 峰值压力；预先编译的伪逆矩阵和查找表（POW/CBRT）用于无除法的三角/立方根；DSP‑free 乘法链、128 位迭代除法器、非分支树形最小值寻找；完整的 openXC7 流程（yosys + nextpnr + prjxray）实现零浮点 IP 的位图生成；JTAG Wishbone 读回用于硅级逐比特验证。

**📊 数据集**

数据集：以实际的 hypoid 齿轮对 s0003（7/70 teeth，90°轴角，19 mm 偏移）为实验基准，对其几何网格、加载情况（32–258 Nm）和摩擦力场进行采样；在此基础上生成 80×80 的间隙模板并在 FPGA 上执行；LCP 参考求解采用 Boussinesq‑半空间 + NNLS active‑set，并在 CPU 上完成，作为精确标注。

**📈 对比分析**

对比方法：① 与完整 LCP 解决方案（≈2005 ms、2 % 失调）比较，预测峰值压力平均误差 ≈‑22 %（-13 %到‑31 %）；② 与 400 个样本训练的 MLP 近似器（在训练区误差 0 %，超出区误差 20.3 %）比较，物理直接在超出区误差 18.9 % 以内，且无需训练。性能方面：FPGA 预测 806 周期（≈16 µs），CPU 预测 277 周期（≈7 µs）但带有 7 % 抖动，完整 LCP 2005 周期。资源：约 197 000 LUT、4 850 FF、4 638 CARRY4、2×RAMB36+1×RAMB18，DSP 0，时钟 50 MHz 可扩展至 63 条计算通道。

**⚠️ 局限性**

局限性：① 预测仅提供 bulk‑Hertz 级别的接触信息，无法捕获尖峰的齿尖/边缘负载，真正的峰值仍需 LCP；② 固定点精度限制了极端几何（如接近共形接触）和高压条件下的精度；③ 设计在 XC7K480T 上 DSP‑bound，最大通道数约 63；④ 需预先编译伪逆和 LUT，针对不同齿轮类型需重新生成；⑤ 仅适用于假设半空间赫兹模型的齿轮接触，无法覆盖润滑或非线性材料效应。

---

## 12. Towards Understanding Pause Token Fine-Tuning Dynamics: A Mode Retention Perspective

**arXiv ID:** 2609.04489 | [PDF](https://arxiv.org/pdf/2609.04489v1)

**作者:** Jaehyeon Kim `[一作]` (HodooAI Lab), Jungwoo Lee `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了 Masked Boundary Pause (MBP) 细调方法，用于在训练时插入并掩码暂停标记，以改善 LLM 的推理能力并保持预训练分布。

**💡 创新点**

创新点在于将暂停标记视为训练时的动态干预，利用边界位置插入并掩码损失，从而在保持原始语言能力的同时显著提升推理任务的性能。

**🔧 技术方法**

采用了 Transformer 模型（Qwen3、Llama 等）与 MBP 细调策略，结合 LoRA、GRPO（强化学习）以及对暂停标记的精细位置选择和损失掩码技术。

**📊 数据集**

使用的主要数据集包括 DeepMath、iGSM、GSM8K、MATH、AMC23、AIME24、Minerva、MBPP、HumanEval 等推理/代码数据集，以及 MMLU、GPQA、BBH、HellaSwag、PIQA 等通用语言理解基准。

**📈 对比分析**

通过与标准 SFT、append、random、DIT 等对照实验比较，MBP 在 1B–8B 规模模型上在数学推理上提升最高 6.3 分、代码推理 2.5 分，并且在通用基准上保持甚至提升性能；在 GRPO 训练中，MBP 通过保持更广泛的行为支持实现更快收敛和更高最终准确率。

**⚠️ 局限性**

局限性包括：仅在全参数细调场景下验证，LoRA 等参数高效细调效果不佳；评估集中在少数推理任务与基准上，缺乏更细粒度的分布与 token 级别分析；需要进一步探索 MBP 在更大规模模型和多任务环境中的可扩展性。

---

## 13. When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference

**arXiv ID:** 2609.04490 | [PDF](https://arxiv.org/pdf/2609.04490v1)

**作者:** Ismail Erbas `[一作]` (Rensselaer Polytechnic Institute), Vikas Pandey `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究低精度递归网络中的“recurrent‑state write‑back”对光学荧光寿命估计任务的影响，并在冻结模型下通过改变写回规则来隔离其效应。

**💡 创新点**

创新点在于首次将写回规则视为独立设计变量，证明其能在不改变网络权重的情况下导致严重性能衰退；同时提出错误反馈、残差记忆和方向记忆三种机制可在固定网络中恢复精度；并展示写回敏感性在GRU与LSTM中的跨架构一致性。

**🔧 技术方法**

主要技术包括：分阶段量化感知训练（QMem）实现4‑bit状态写回；对GRU/ LSTM 进行后训练写回干预；引入误差反馈、量化残差与方向记忆等辅助记忆机制；以及对写回边界与写入频率的统计分析。

**📊 数据集**

使用的是1,600,000条高噪声时域荧光信号的仿真数据集，每条样本包含135个时间箱，按80/10/10划分为训练/验证/测试集。

**📈 对比分析**

与原始4‑bit、8‑bit以及训练期间逐步量化的基准模型比较，发现后训练4‑bit写回将τ1、τ2的RMSE分别提高约70倍和300倍；利用记忆机制可将误差恢复到0.35/0.40 ns；精度扫描显示更高位宽并不必然提升性能；匹配训练后可学习接口兼容性。

**⚠️ 局限性**

局限性包括：写回效应与网络架构和内部状态的具体角色高度相关，不能直接推广到所有递归模型；需要额外的辅助记忆资源或重新训练来补偿写回失效；实验仅在仿真荧光寿命数据和小型GRU/LSTM上验证，实际硬件实现与能耗影响尚未评估。

---

## 14. You Really Didn't Get That? Benchmarking Social Pragmatic Inference for Indirect and Playful Chinese Online Comments

**arXiv ID:** 2609.04384 | [PDF](https://arxiv.org/pdf/2609.04384v1)

**作者:** Shiwei Hong `[一作]` (George Mason University), Zhicong Lu `[通讯]` (George Mason University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个针对中文社交媒体中间接、玩笑式评论的情境化语用推理诊断基准，评估大语言模型在恢复上下文依赖含义方面的能力。

**💡 创新点**

创新点在于：①提出跨模型生成-验证的诊断问答流程，使模型既能写出针对性问题，也能在他人写的问题上作答；②强调情境化的多层上下文（帖子、父评论、本地回复）在语用推理中的核心作用；③提供可复用、可更新的构建协议，适用于语言演变快速的社交平台。

**🔧 技术方法**

技术主要包括：多轮对话上下文重构、LLM（GPT‑5.5、Claude Opus 4.7、DeepSeek V4 Pro、Qwen3.5、Kimi K2.6、MiniMax M2.5、Mistral Medium 3、Llama 4 Maverick）交叉写问和答、人工验证与润色、留写者外评（LWO）准确率评估、上下文消融实验。

**📊 数据集**

数据集来源于 2023‑2026 年的知乎、豆瓣、少数、贴吧等公开互动记录，筛选 4,735 条经过人工校正的诊断问答对，覆盖约 200,000 条原始评论记录。

**📈 对比分析**

在留写者外评（LWO）指标下，Qwen3.5‑9B 最高达 81.42%；平均 LWO 为 68.70%；相比之下人类评估准确率为 90.8%，显著高于模型。人类修订后的难度子集模型准确率降至 30–40%。消融实验显示，完整上下文是关键，去除任何上下文层都会导致显著性能下降。

**⚠️ 局限性**

局限性包括：①样本聚焦于富含情境化语用的热点话题，无法代表整个中文社交媒体的分布；②仅覆盖文本层面，未扩展至视频、音频等多模态平台；③数据可能存在间接泄露的风险，且评测结果随模型版本变化；④人工验证依赖专业人力，难以实现大规模实时更新。

---

## 15. Object Concepts Emerge from Motion

**arXiv ID:** 2609.04348 | [PDF](https://arxiv.org/pdf/2609.04348v1)

**作者:** Boshi Li `[一作]` (Beijing University of Posts and Telecommunications), Naiyan Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用视频中的光流边界生成无监督的伪实例标签，并通过像素级对比学习训练单图像编码器，学习以对象为中心的视觉表示。

**💡 创新点**

创新点在于：①将运动边界直接转化为类别无关的实例标签；②提出 Motion‑Verified Self‑Training 机制，将模型生成的候选掩码与运动证据结合，显著扩展标签覆盖率；③通过稠密对比学习实现对象统一性与实例分离的双重约束。

**🔧 技术方法**

技术包括：光流估计（VideoFlow）、基于聚类的伪标签生成、像素级 pairwise metric 学习、MaskFormer 用于候选掩码提议、Swim Transformer backbone、教师-学生自蒸馏。

**📊 数据集**

使用 7,163 小时的异构视频数据（OpenDV‑YouTube、nuPlan、NVIDIA PhysicalAI‑Autonomous‑Vehicles、Web 视频），在 Cycle‑1 生成 195M 帧伪标签，在 Cycle‑2 通过验证扩展至 421M 帧。

**📈 对比分析**

在四个下游任务上与 ImageNet‑22K、CLIP、DINO、SimMIM 等基线对比：在 KITTI 深度估计、nuScenes 3D 检测/占据预测、NAVSIM 端到端规划中，所学表示在规模较小到中等的 Swin backbone 上均能达到或超过基线，尤其在几何与实例敏感任务中优势明显。

**⚠️ 局限性**

局限性：高度依赖光流质量，易受快动、遮挡、低纹理、非刚性等情况影响；伪标签生成对运动显著且边界清晰的实例偏好，导致对小物体或部分遮挡对象的覆盖不足；扩展到更大模型或更多数据仍需提升光流精度与伪标签可靠性。

---

## 16. Toward Model-Driven Digital Twin Configuration: Separating Structure Semantics and Runtime with SysML SAREF and Ditto

**arXiv ID:** 2609.04213 | [PDF](https://arxiv.org/pdf/2609.04213v1)

**作者:** Andrey Sadovykh `[一作]` (Softeam), Kirill Korikov `[通讯]` (Innopolis University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本文提出将能源社区数字孪生的配置拆分为三层：SysML建模负责结构、SAREF4ENER负责能量域语义、Eclipse Ditto/WoT TD负责运行时表示，并给出了从SysML+SAREF模型推导Ditto配置的概念规则；

**💡 创新点**

创新点在于将标准化的能量词汇SAREF4ENER与SysML结构模型分离，提供模型到Ditto配置的概念性转换规则，使设备类型、测量语义和单位显式化，弥补手工JSON配置缺乏语义、可追溯性和互操作性的不足；

**🔧 技术方法**

使用技术包括SysML（Modelio）、SAREF4ENER本体引用包、Eclipse Ditto平台的Thing Description（W3C WoT TD）以及JSON‑LD语义标注；

**📊 数据集**

案例数据集为一个温室能源社区节点，手工构建了包含四种设备（光伏、负载、传感器、双向计量）的SysML模型，并演示了对应的Ditto配置；

**📈 对比分析**

方法上未实现可执行生成器，仅给出概念规则并与WoT‑TD、Vorto、IEC AAS、NGSI‑LD等四种替代方案在结构建模、域语义、平台配置、可追溯性四项指标上做对比，显示其在四项指标上均优于单一方案；未给出运行时性能数值；

**⚠️ 局限性**

局限性包括规则仅为概念层面，未实现自动化生成器或验证；模型与配置的双向一致性未验证；对大型系统的块划分仍是设计决策；仅覆盖SAREF4ENER的部分语义，未涵盖所有能源社区概念；需要SysML和本体知识背景。

---

## 17. REFINE: LLM Refinement over Budgeted Text-Attributed Graphs for Personalized Medical Concept Representation

**arXiv ID:** 2609.04415 | [PDF](https://arxiv.org/pdf/2609.04415v1)

**作者:** Mohsen Nayebi Kerdabadi `[一作]` (University of Kansas), Zijun Yao `[通讯]` (University of Kansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种患者个性化医学概念编码器REFINE，利用文本属性知识图谱构建患者专属图，并通过序列强化学习为每个观察到的代码动态分配KG上下文预算；随后用异构GNN提取结构依赖并通过图感知软提示冻结LLM进行语义细化，融合得到个性化概念嵌入；

**💡 创新点**

创新点包括：①序列RL策略实现按代码和患者动态预算KG邻域，解决统一扩展的噪声与计算开销问题；②将患者图结构与LLM语义相结合的图感知软提示，避免全序列化文本、保留结构信息；③将生成的个性化嵌入作为通用插件，可无缝集成多种EHR预测骨干网络；

**🔧 技术方法**

技术手段包括：构建文本属性知识图谱、序列强化学习（actor‑critic）预算策略、异构图神经网络、冻结大型语言模型、图感知软提示（hard+soft tokens）以及门控融合两模态表示；

**📊 数据集**

使用公开的两大医疗记录数据集：MIMIC‑III和MIMIC‑IV，分别包含数千名患者、数万次就诊和多种诊断、药物、手术代码；

**📈 对比分析**

与多种基线（Transformer、GRAM、MMORE、KAME、G‑BERT、HAP、ADORE、GraphCare、Rel‑LLM、LINKO、MedCo）以及多种预测骨干（AttPool、AdaCare、Transformer、RETAIN、TCN）进行对比；REFINE在AUPRC、F1、Acc@k等指标上普遍位居第一，尤其在稀缺诊断标签上表现突出；

**⚠️ 局限性**

局限性主要在于KG扩展策略使用离散的跳数与邻居数固定动作空间，无法对具体节点进行连续或精细化选择；此外当前仅采用基于证据排序的邻居采样，未能动态探索更优子图；

---

## 18. Data-Optimized Contingency Screening: A Machine Learning Approach to Power System Security

**arXiv ID:** 2609.04300 | [PDF](https://arxiv.org/pdf/2609.04300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 19. Blockchain-Enabled Secure Logging for Fiscal Electronic Mechanisms: Evaluation of the Greek eSEND and myDATA Tax Systems

**arXiv ID:** 2609.04356 | [PDF](https://arxiv.org/pdf/2609.04356v1)

**作者:** Panagiotis Mavridis `[一作]` (Hellenic Mediterranean University), Christos Nikolopoulos `[通讯]` (Hellenic Mediterranean University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文对希腊财政电子机制（FEM）及其中心税务数据库eSEND的区块链实现与电子发票服务（EIPS）及myDATA平台的区块链缺失进行评估，并对两种体系的交易完整性与防篡改性进行对比。

**💡 创新点**

创新点在于：①首次系统性地将硬件级双/三重哈希链与AES‑256加密结合用于FEM设备的事务链验证；②揭示仅软件化的e‑发票体系在缺乏区块链、连续编号与硬件防篡改时，导致交易完整性难以保障的现实缺陷；③通过对比分析，提出在数字税收生态中应加强区块链与硬件保障的设计思路。

**🔧 技术方法**

技术：双/三重SHA‑1哈希链、区块链规则验证、AES‑256对FEM数据的加密传输、EIPS的HTTPS数据推送、以及对Z报告的顺序性与完整性校验。

**📊 数据集**

数据集：来自实际部署的FEM设备（ECR、AFD、EAFDSS、FEMAS、EFTPOS）的交易日志与Z报告；eSEND中心数据库中的加密数据包；以及通过EIPS推送至myDATA的电子发票与收据记录。

**📈 对比分析**

比较方法：①对FEM与eSEND的哈希链结构与加密传输流程进行静态与动态分析；②对EIPS与myDATA的编号规则、区块链缺失与MARK编号连续性进行对照；③通过验证Z报告是否被接受、被拒绝以及缺失情况，评估两系统的交易完整性。结果显示，eSEND通过区块链规则和顺序校验实现了较高的完整性；而myDATA在缺乏区块链与连续编号机制时，存在潜在的缺失与篡改风险。

**⚠️ 局限性**

限制：①论文主要基于已有硬件和法规框架，未实验测量区块链和加密运算的实时性能开销；②对EIPS的安全性评价缺乏大规模实测，未覆盖极端网络中断或高并发场景；③结论受希腊特定税收体系与设备实现的限制，推广到其他国家时需考虑不同技术与法规环境。

---

## 20. Reviewer Capability Governs Rejection Targeting, Not Repair Skill: Evidence from LLM Execute-Review-Revise Pipelines

**arXiv ID:** 2609.04270 | [PDF](https://arxiv.org/pdf/2609.04270v1)

**作者:** Faizan Tanveer `[一作]` `[通讯]` (National University of Computer and Emerging Sciences), Faizan Tanveer (National University of Computer and Emerging Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究多代理LLM管线中不同能力层级的审核者对最终答案准确性的影响。

**💡 创新点**

创新点在于提出拒绝结果分解（修复、损伤、无变更、误修），并系统评估审核者能力对检测与修正效果的影响，发现存在“能力底限”。

**🔧 技术方法**

使用双阶段执行-审核器架构，配合 Gemini 3.1 Flash-Lite 执行器和 gpt-oss-20b、Llama-3.1-8B 审核器，以及 token 计费和统计分析。

**📊 数据集**

使用 100 道来自 Omni-MATH 的中等难度（5.0-6.0）奥数数学题，答案为整数/小数，便于自动精确评分。

**📈 对比分析**

通过 McNemar、Fisher 等统计检验对最终准确率、检测召回率和错误率进行比较，结果显示跨家族中等审核器将准确率提升 12pp（p=0.0005），而自审虽召回率高达 85% 但未显著提升准确率。

**⚠️ 局限性**

局限性包括样本量仅 100、仅单一任务集与难度区间、单一执行器和单轮修订，且弱审核器行为受提示长度影响，难以推广到更广泛场景。

---

## 21. A Governance Methodology Layer for AI-Assisted Software Development: Defect Taxonomy, Controlled Ablation, and Process-Over-Capability Evidence

**arXiv ID:** 2609.04218 | [PDF](https://arxiv.org/pdf/2609.04218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 22. Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation

**arXiv ID:** 2609.04298 | [PDF](https://arxiv.org/pdf/2609.04298v1)

**作者:** Lin Shi `[一作]`, Alex Shaw `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Harbor Adapters 统一评测框架，并将 80+ 代理评测基准迁移至该框架，完成大规模模型与 Harness 的评估，推出 Harbor-Index 1.0 细化任务集。

**💡 创新点**

创新点在于：① 将基准适配成统一任务接口，显著降低了 benchmark‑agent 组合的集成成本；② 通过大规模实验揭示模型能力而非 Harness 决定性能；③ 设计了 Harbor-Index，结合 AI 与人工审核、难度过滤、迭代修复，提供可重复、困难且高质量的评测任务。

**🔧 技术方法**

技术包括：Python 库 Harbor 任务抽象、沙箱执行（Daytona/Modal/E2B 等）、LLM-as-a-judge 验证、GPU 交互、多种 Harness（Terminuse-2、原生 Codex/Claude Code/Gemini CLI）以及多阶段质量审计流程。

**📊 数据集**

数据集覆盖 80+ benchmark（如 SWE-bench、Terminal-Bench、FinanceAgent 等），并筛选出 1,311 任务做难度过滤，最终得到 100 题 Harbor-Index 1.0；此外对多模型（GPT‑5.5、Claude Opus 4.8、Gemini 3.1 Pro 等）进行评测。

**📈 对比分析**

评估方法：对每个模型‑Harness 组合进行 3 次实验，累计约 0.3M 轨迹；采用标准化的通关率与成本（USD）对比。结果显示：顶级闭源模型通关率 20–28%，成本 155–289 USD/次；开放权重模型通关率 2–9%，成本 4–50 USD/次，显示模型能力主导性能，Harness 影响次要。

**⚠️ 局限性**

限制：评测仅覆盖 54 个可适配 benchmark；计算成本限制模型与 Harness 的组合，未覆盖所有 LLM、Agent 架构；未来更强代理可能利用评测漏洞导致结果失真；Harbor-Index 仍需持续更新以防止饱和。

---

## 23. Evaluation of Phonetic Encoding Algorithms on Transcription Datasets

**arXiv ID:** 2609.04391 | [PDF](https://arxiv.org/pdf/2609.04391v1)

**作者:** Can Özbey `[一作]` (Huawei Turkey Research and Development Center), Berkin Deniz Kahya `[通讯]` (Huawe Turkey Research and Development Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于Hüllermeier‑Rifqi指数的音素编码评估方案，衡量音素编码与IPA转录的一致性。

**💡 创新点**

创新点在于将Rand Index泛化为可处理字符串的HRI，结合归一化编辑距离构造对称的正负一致性度量，并与随机模型对比得到校正后的一致性得分。

**🔧 技术方法**

采用Hüllermeier‑Rifqi指数、归一化编辑距离、正负样本分割、碰撞率计算及AOC校正等技术。

**📊 数据集**

使用多语言IPA词汇对照数据集（英语、德语、法语、瑞典等），以及对多种音素编码器的输出。

**📈 对比分析**

通过计算PC、NC、OC、AOC以及碰撞率，比较不同编码器在不同语言中的性能，结果显示Metaphone、MRA等编码在准确率与召回率间取得平衡，PHEX在高碰撞率下仍保持较高AOC。

**⚠️ 局限性**

局限在于评估仅基于归一化编辑距离，可能对语素写法多音符号的语言适应性不足；样本数量受限于O(N²)复杂度；并未考虑专门为专有名词设计的编码器在该任务上的差异。

---

## 24. MedProb: Probing Internal Representations of Vision-Language Models for Medical Question Answering

**arXiv ID:** 2609.04336 | [PDF](https://arxiv.org/pdf/2609.04336v1)

**作者:** Erfan Nourbakhsh `[一作]` (University of Texas at San Antonio), Anthony Rios `[通讯]` (University of Texas at San Antonio)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MedProb，一种利用冻结的视觉语言模型（VLM）内部表示进行线性探测，直接预测多选 Med-VQA 答案；

**💡 创新点**

证明了生成式评估低估了 VLM 内部可检索的医学知识，且医学适配并不一定提升线性可解码能力；

**🔧 技术方法**

使用线性多项式回归探测器（logistic 回归）对每一层隐藏状态进行分类，结合少量标注数据；

**📊 数据集**

在 PATH-VQA、SLAKE、VQA-RAD 这三个闭合式 Med-VQA 基准上进行评估，亦对 OmniMedVQA 的四选任务做实验；

**📈 对比分析**

与多代理系统、医学 VLM、提示式生成等基线对比，MedProb 在大多数模型上均超越提示式生成，并在小模型上表现优异，缩小了小大模型差距；

**⚠️ 局限性**

需要少量标注数据、仅针对闭合式问题、线性探测可能低估非线性可解码信息、未提供临床解释与证据、对开源、跨模态、复杂推理仍有限。

---

## 25. Nested Inductive Bias Framework for SPD Manifold Learning

**arXiv ID:** 2609.04466 | [PDF](https://arxiv.org/pdf/2609.04466v1)

**作者:** Tushar Das `[一作]` `[通讯]` (National Institute of Technology Jamshedpur), Tushar Das (National Institute of Technology Jamshedpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了嵌套归纳偏差框架（Nested Inductive Biases），通过双阶段微分同胚将非欧几里得曲率（如双曲和球面）拉回到对称正定矩阵（SPD）流形上，从而在保持物理约束的同时引入数据的隐含关系几何。

**💡 创新点**

创新点包括：① 用两阶段同胚实现 SPD 与任意常曲率流形的同胚映射；② 推导出对应的拉回度量（Nested Hyperbolic / Spherical）并在其上实现 Riemannian 多项式逻辑回归；③ 设计了 Rational Conformal Metric（RCM）作为几何上有界的度量，既能压缩极端异常值，又保持核心分布的角度关系。

**🔧 技术方法**

技术手段包括：拉回度量（pullback metrics）、κ-立体投影模型、Riemannian 多项式逻辑回归（RMLR）、Rational Conformal Metric 与其切空间映射（TSM）、深度 SPDNet、RResNet、线性探针实验、Ollivier‑Ricci 曲率估计、数值稳定化技巧（64 位、微小扰动）。

**📊 数据集**

使用的数据集有：First‑Person Hand Action (FPHA) 63×63 运动协方差矩阵；Radar Target Classification 20×20 协方差矩阵；以及人工合成的正/负曲率 SPD 样本，用于验证曲率对分类的影响。

**📈 对比分析**

通过与 Log‑Euclidean（平坦）、AIRM（非正曲率）和标准 RMLR 进行对比，在 FPHA 与 Radar 数据上，嵌套双曲度量（NHM）在深度 RResNet 中平均提升约 9.9% 的准确率；嵌套球面度量（NSM）在曲率不匹配时表现差；RCM 在面对极端空间异常时实现 100% 核心准确率，且整体性能与 Log‑Euclidean 基线相近或略优。

**⚠️ 局限性**

局限性包括：① 对正曲率流形的映射仅在注入半径内有效，可能限制大尺度正曲率数据的处理；② RCM 由于几何不完备性，无法实现完整的逆映射，需额外处理；③ 目前仅在输出层使用 RMLR，未能在所有隐藏层完整保持非欧几里得曲率；④ 计算复杂度仍高于纯欧几里得方法，尤其在大规模矩阵维度时；⑤ 需要手工估计数据的内在曲率以选择合适的目标几何。

---

## 26. A Mixed-Method Empirical Study of LLM Assistance in Software Engineering Workflows

**arXiv ID:** 2609.04214 | [PDF](https://arxiv.org/pdf/2609.04214v1)

**作者:** Pamali D. Weerasinghe `[一作]` (University of Colombo), Chamath Keppitiyagama `[通讯]` (University of Colombo)

**通讯引用:** 689 | [OpenAlex ID](https://openalex.org/A5069138688)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文结合问卷与任务实验，研究LLM辅助软件工程在不同学年学生中的工作流程与效果。

**💡 创新点**

提出了基于任务类型和经验水平的LLM使用与验证行为细分，并展示LLM对工作流程的多维影响。

**🔧 技术方法**

采用LLM工具（如ChatGPT、Claude、Gemini）以及屏幕录制、视频转录与定性编码分析。

**📊 数据集**

使用的是UCSC本科生的问卷样本157人和20人的实验任务集，涵盖实现、算法选择与架构推理三层级。

**📈 对比分析**

通过对AI条件与无AI条件下的任务完成时间、错误率、验证深度等指标比较，发现LLM在实现层加速但在约束和验证层未显著提升，甚至有验证不足的风险。

**⚠️ 局限性**

样本单一高校、实验规模小、组分配非随机、依赖自我报告的验证度量，限制了结果的普适性与因果推断。

---

## 27. Safety for Whom? Boundary-Aware Self-Distillation for Controlled LLM Safety Refusal

**arXiv ID:** 2609.04482 | [PDF](https://arxiv.org/pdf/2609.04482v1)

**作者:** Alejo López-Ávila `[一作]` (Multiverse Computing), Román Orús `[通讯]` (Multiverse Computing)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了在特定部署场景下的“窄边界安全”框架，专注于在政治议题内部划定可拒绝的有害子集而非整个主题的拒绝。

**💡 创新点**

创新点在于：① 将自生成安全训练中的覆盖缺口通过“Escalate+Graft”策略修复；② 通过层次化生成控制主题、风格、长度以及构造边界对的训练样本；③ 在评估时使用边界对（harmful‑benign pairs）直接衡量拒绝边界精度；④ 引入多种补偿数据（SafeChain、FakeHarm 等）平衡拒绝与误拒。

**🔧 技术方法**

使用 Qwen3‑8B 作为目标模型，利用 LoRA 进行适配；通过自生成拒绝轨迹、WildGuard 过滤、前向 KL 正则化对善意样本进行训练；同时采用多种覆盖修复策略和补偿数据。

**📊 数据集**

主要数据集为自生成的政治说服对话数据（约 40k 个有害提示），包括长度控制、对齐式对 (PR/PB)、FakeHarm 表面有害安全提示，以及 SafeChain 的外部合规样本；此外还有 XSTest 250 条安全提示、HarmBench/StrongREJECT/WildJailbreak 等公共安全基准。

**📈 对比分析**

与单射生成、仅使用外部合规或仅使用 FakeHarm 的基线对比。Escalate+Graft 在第 4 轮训练后，政治领域拒绝率从 0.0947 提升至 0.8475，跨领域安全率从 0.2626 降至 0.0014，但 XSTest 误拒率从 0.0200 上升至 0.7400；加入 SC2 与 FakeHarm 能显著降低误拒率，同时保持较低的安全率。边界对数据进一步把误拒率降至 4% 左右，拒绝率略降至 88%。

**⚠️ 局限性**

局限性：仅在政治说服主题上验证，且主要使用 Qwen3‑8B 与单一 LoRA 设定；对其他模型、话题和训练配置的泛化尚未证明；评估依赖于 WildGuard/LlamaGuard 过滤器，可能带来标注偏差；数据集包含敏感政治内容，使用受限。

---

## 28. Hoss: Fast Oblivious Semantic Search with Heterogeneous GPU-CPU-TEE Architecture

**arXiv ID:** 2609.04522 | [PDF](https://arxiv.org/pdf/2609.04522v1)

**作者:** Jianzhang Du `[一作]` (Indiana University), Zhongshu Gu `[通讯]` (IBM Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一种基于异构 GPU‑CPU TEE 架构的无差别语义搜索系统，实现了对 HNSW 图的高效访问并保障数据与访问模式的隐私。

**💡 创新点**

创新点在于利用 GPU TEE 的大容量私有内存，将 HNSW 的热点路径放在 GPU 上进行无差别访问，仅在需要访问较低层时才调用 CPU TEE，并提出了超越传统性能限制的主机访问 ORAM 机制与数据相关优化。

**🔧 技术方法**

采用 HNSW 图、ORAM、GPU/CPU TEE、主机访问 ORAM 机制以及数据相关优化技术。

**📊 数据集**

使用公开的语义检索基准数据集（如大规模向量嵌入集）进行实验。

**📈 对比分析**

与现有最优系统 Compass 在相同环境下进行基准比较，结果显示新系统在保持高召回率的同时可获得最高 99 倍的速度提升，规模增大时加速效果更显著。

**⚠️ 局限性**

局限性包括对 GPU TEE 大容量内存的依赖，当图的下层超过 GPU 容量时需转移至 CPU TEE，导致部分性能损失；以及对特定硬件平台的实现依赖，尚未在更大规模或多租户场景中验证。

---

## 29. Nebulon Enterprise Simulated Threats for Phishing Research (NEST-Phish): A Synthetic Enterprise Phishing Email Dataset for Behavioral and Machine-Learning Research

**arXiv ID:** 2609.04474 | [PDF](https://arxiv.org/pdf/2609.04474v1)

**作者:** Emily J. Winokur `[一作]`, Danielle N. Sanchez `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并公开了一个基于虚构企业Nebulon的合成对照钓鱼与合法邮件数据集，配合人类与机器学习评估；

**💡 创新点**

创新点在于将钓鱼与合法邮件以主题匹配、可控线索变异的方式合成，并提供可解释的钓鱼线索注释、公开可复现的工作流程；

**🔧 技术方法**

使用LLM辅助生成与人工审核相结合的邮件制作流程，结合混合效应模型分析人类判断，使用逻辑回归与随机森林对邮件文本与结构特征进行分类；

**📊 数据集**

采用内部种子数据集扩展后得到的146封合成邮件作为主数据集，结合人类实验数据与分类器训练/测试数据；

**📈 对比分析**

人类实验中整体准确率82%，钓鱼邮件准确率74.7%；机器学习方面，随机森林在文本+结构特征下达成≈96%准确率，AUC≈0.99；两者均表明数据集具备可学习性且能区分钓鱼与合法邮件；

**⚠️ 局限性**

局限在于数据完全合成，可能与真实企业邮件在语言、细节上存在差异；样本量和主题/线索多样性有限；仅评估了两种模型与特征组合，未覆盖更广泛的算法或跨域泛化情况。

---

## 30. Performance Study of Serverless Workloads in Confidential Virtual Machines

**arXiv ID:** 2609.04478 | [PDF](https://arxiv.org/pdf/2609.04478v1)

**作者:** Rikesh Niroula `[一作]` (New Jersey Institute of Technology), Xiaoning Ding `[通讯]` (New Jersey Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在Intel TDX与AMD SEV‑SNP上部署OpenFaaS，系统评估了Confidential Virtual Machines（CVMs）对服务器无服务器工作负载的内存效率、warm‑start和cold‑start开销，并分析了其根本原因。

**💡 创新点**

创新点在于：①量化CVM导致的跨VM页合并失效和内存回收性下降对热容器容量的影响；②揭示高同步与idle‑active切换导致的VMEXIT频繁出现是warm‑start延迟的主因；③在受限信任模型下通过放宽容器级隔离（cgroup、Seccomp、网络命名空间）显著降低cold‑start启动时间。

**🔧 技术方法**

使用技术包括：Intel TDX / AMD SEV‑SNP 硬件隔离；KVM、KSM、cgroup、Seccomp、网络命名空间；OpenFaaS/Kubernetes 架构；实验平台为GCP Sapphire Rapids（TDX）与Milan（SEV‑SNP）实例以及本地HPE ProLiant DL380 Gen11。

**📊 数据集**

使用的数据集为15个公开基准：FunctionBench、SeBS、FaaSProfiler，涵盖算术、IO、机器学习推理、Web/API等四大类别。

**📈 对比分析**

比较方法为在相同硬件、相同软件栈下对比CVM与普通VM的内存使用、warm‑start 延迟与VMEXIT频率、cold‑start 的容器初始化耗时；实验显示：CVM 内存开销大幅提升（可达 120%+）；warm‑start 对高同步工作负载的延迟可达 2–3×；cold‑start 在CVM与普通VM相差不大，但占总时延约 60%；通过线程数调优、idle‑transition 改写、放宽容器隔离等策略可将 warm‑start 与 cold‑start 延迟分别降低 10–30%。

**⚠️ 局限性**

局限性包括：①CVM 失去跨VM页面合并导致的内存低效难以通过现有技术完全补偿；②VMEXIT 处理开销对多线程/同步敏感，系统级的 idle‑spin 或 PLE 方案在 CTF 环境下效果有限；③容器隔离放宽方案仅适用于同功能、相互信任的容器，安全性受限；④实验多基于单节点 OpenFaaS，未覆盖大规模多租户调度与多节点性能。

---

## 31. BioSync: Transformer-Based Cross-Modal Fusion for a Multimodal Physiological Digital Biomarker

**arXiv ID:** 2609.04504 | [PDF](https://arxiv.org/pdf/2609.04504v1)

**作者:** Seyed Mahmoud Sajjadi Mohammadabadi `[一作]` `[通讯]` (University of Nevada), Seyed Mahmoud Sajjadi Mohammadabadi (University of Nevada)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并评估了一种名为BioSync的多模态融合模型，用于将心率变异性、EEG、腕部运动学和语音等多种可穿戴与移动设备测量合成为连续的生物标志物；

**💡 创新点**

创新点在于提出了宽深混合Transformer架构，融合了多头自注意力与线性concat分支，能够在保持传统拼接能力的同时实现自适应模态权重，并满足BEST框架提出的六大指标（分级有效性、多模态信息捕获、适应性可解释性、渐进降解、架构通用性、边缘/联邦可部署性）；

**🔧 技术方法**

使用的核心技术包括多头自注意力Transformer、单层注意力+前馈网络、线性拼接路径、模态dropout训练、批量归一化与残差连接，以及基于合成数据的5折交叉验证评估；

**📊 数据集**

所用数据集为两组基于文献的合成队列：一组四模态认知衰退队列（HRV、EEG、腕动、语音）以及一组两模态代谢自主性队列（基于AI-READI Garmin Vivosmart5数据结构的心血管-自主与行为两类特征）；

**📈 对比分析**

通过与单模态基线、早期拼接（concatenation）以及晚期投票等多模态融合方法进行对比实验，BioSync在认知队列的AUC为0.928（高于拼接的0.926），在代谢队列的准确率和F1分别为0.764/0.766（略高于拼接的0.756/0.758），但在其他指标上与拼接相差不大；

**⚠️ 局限性**

主要限制包括：实验仅在合成数据上完成，未使用真实患者或原始时序信号；模态损坏实验使用高方差噪声的简化替代方案；联邦学习与低秩压缩等边缘部署方案尚未实现；模型在不同数据集和任务上的可推广性需通过后续真实队列验证。

---

## 32. FAVE: Foveated Adaptive Visual Encoding for Efficient Fine-Grained Visual Understanding

**arXiv ID:** 2609.04392 | [PDF](https://arxiv.org/pdf/2609.04392v1)

**作者:** Amitangshu Mukherjee `[一作]` (Purdue University), Kaushik Roy `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的视觉编码方法FAVE（Foveated Adaptive Visual Encoding），该方法通过选择性地对局部区域进行高分辨率编码来提高细粒度视觉理解的效率。

**💡 创新点**

FAVE的创新点在于它能够根据所选区域的大小自适应调整空间分辨率和标记数量，从而避免对每个区域使用固定分辨率的表示。

**🔧 技术方法**

使用了轻量级的可变分辨率视觉变换器（ViT）和Patch-n-Pack技术来处理可变长度的输入序列。

**📊 数据集**

使用了ImageNet数据集，特别是针对最大边长为96像素的小物体进行实验，并使用真实的边界框作为选择区域。

**📈 对比分析**

与固定分辨率的ViT相比，FAVE在相同的裁剪窗口上提高了9.4个百分点的Top-1准确率，同时计算量降低了12.7倍。在TextVQA任务中，FAVE在FastVLM-1.5B基线的基础上提高了1.60个百分点，并在GQA属性问题上提高了1.31个百分点。

**⚠️ 局限性**

限制在于FAVE的性能依赖于所选区域的质量和选择机制，且在处理全图时可能无法充分利用其优势。

---

## 33. Towards a universal language of concepts: A survey

**arXiv ID:** 2609.04528 | [PDF](https://arxiv.org/pdf/2609.04528v1)

**作者:** Aishni Parab `[一作]` `[通讯]` (University of California Los Angeles), Aishni Parab (University of California Los Angeles)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并评估了将程序作为通用概念表征的多种计算模型，并讨论其实现方式与理论优势。

**💡 创新点**

创新点在于把程序看作最通用的概念表示框架，并把实例集合与相互依赖程序两种视角结合，提出跨域概率程序化的可能性。

**🔧 技术方法**

采用了概率上下文无关文法（PCFG）、组合逻辑（CL）、贝叶斯程序学习（BPL）以及域特定语言（DSL）等技术。

**📊 数据集**

使用了Omniglot手写字符集、Bongard几何图形任务、视觉问答及机器人训练等数据集。

**📈 对比分析**

通过与传统深度学习模型和贝叶斯模型对比，BPL在一次性学习上达成人类水平，并在视觉概念生成与机器人任务中优于常规方法。

**⚠️ 局限性**

局限性包括搜索空间庞大、需手工设定先验、表示与学习算法难以完全分离，以及在不同领域的泛化能力尚有限。

---

## 34. The microscope is the mask: privileged views and labels from a cryo-ET forward model

**arXiv ID:** 2609.04325 | [PDF](https://arxiv.org/pdf/2609.04325v1)

**作者:** Bogdan Toader `[一作]` (MRC Laboratory of Molecular Biology), Sjors H. W. Scheres `[通讯]` (MRC Laboratory of Molecular Biology)

**通讯引用:** 52129 | [OpenAlex ID](https://openalex.org/A5061952716)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用物理前向模型生成配对视图和特权标签，在模拟数据上训练自监督模型CARNIVAL，用于在受限投影和噪声严重的冷冻电镜层析体中进行蛋白质注释。

**💡 创新点**

创新点在于：1）将前向模型产生的“清洁‑腐蚀”配对视图作为不变性目标；2）利用模拟中已知的蛋白质位置与身份构造场景级损失，将语义信息定位到蛋白质位置；3）将上述两种信息融合进LeJEPA自监督框架，实现对干扰的鲁棒表征。

**🔧 技术方法**

使用的技术包括：LeJEPA自监督学习框架、密集特征编码器（E）、读出网络（F）和可选分割头（G）、基于前向模型的三种视图（物理轴、内容轴）以及场景级对齐损失；同时在训练时通过GPU实现在线合成清洁与腐蚀视图。

**📊 数据集**

训练数据来自大规模合成蛋白质场景（模拟体积）；评估使用公开的CZII Cryo‑ET Object Identification挑战集中的121幅真实层析体，包含6种蛋白质类型及约32,700个实例。

**📈 对比分析**

与同样基于模拟数据但无配对视图或特权标签的TomoTwin进行比较。评估指标为分类F1分数、检测F4分数。CARNIVAL在过滤与去噪层析体上均超过TomoTwin，分类F1平均值提升至0.88（TomoTwin为0.83），检测F4在所有条件下保持领先。

**⚠️ 局限性**

局限性包括：1）依赖前向模型的完整性，若模型缺失某种噪声轴或失真，模型难以适应；2）对前向模型参数范围的覆盖性不足，无法完全覆盖实验空间；3）模拟与真实数据之间存在残余差异，可能导致对真实噪声与辐射损伤的过度不变性。

---

## 35. Joint Alignment and Distillation for Video Generation via Sample-Guided Distribution Matching

**arXiv ID:** 2609.04283 | [PDF](https://arxiv.org/pdf/2609.04283v1)

**作者:** Jiuzhou Lin `[一作]` (Tsinghua University), Han Li `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种统一的单阶段视频生成模型优化框架DM-Align，兼顾了分布匹配蒸馏和人类偏好对齐；

**💡 创新点**

创新点在于将对齐梯度自然表达为分布匹配（score）空间梯度，消除RL所需的多步采样与MDP转换；

**🔧 技术方法**

技术包括分布匹配蒸馏（DMD）、DM-PairLoss与DM-GroupLoss两种基于样本的对齐损失，以及对齐梯度与蒸馏梯度的协同融合；

**📊 数据集**

实验数据集为ConsistID（真实视频对）和VidProM（文本提示），并在Wan 2.1-T2V-1.3B与CogVideoX等基模型上验证；

**📈 对比分析**

与原始模型、单独DMD、传统RL（Flow-DPO/DanceGRPO）以及两阶段RL+蒸馏流水线对比，DM-Align在VBench自动评测和人类评估中均显著提升，尤其在动态度与美学质量上分别提升至约+6.55分和+48%首选率；

**⚠️ 局限性**

局限包括缺乏严格的收敛理论保证、仅基于DMD蒸馏，未探索GAN等其他蒸馏方法，且在极度稀疏或多模态奖励场景下稳定性尚待进一步验证。

---

## 36. Modular Deep Recurrent Neural Network: Application to Quadrotors

**arXiv ID:** 2609.04339 | [PDF](https://arxiv.org/pdf/2609.04339v1)

**作者:** Nima Mohajerin `[一作]` (University of Waterloo), Steven L. Waslander `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种模块化深度循环神经网络（MODERNN）架构，并将其应用于四旋翼飞行器的高度动力学建模。

**💡 创新点**

创新点在于：1) 通过前馈层间连接显著缓解空间梯度消失/爆炸问题；2) 设计了统一的模块化结构，便于快速构建多种RNN架构并自动计算雅可比矩阵；3) 将Levenberg–Marquardt优化与批量学习结合，显著提高训练效率。

**🔧 技术方法**

使用技术包括：MODERNN网络结构（含前馈与反馈连接）、雅可比矩阵递推计算、批量Levenberg–Marquardt（LM）学习、交叉验证与动态λ调整。

**📊 数据集**

数据集为仿真生成的四旋翼高度轨迹，采样频率10 Hz，输入为10个不同频率正弦波的线性组合，覆盖0–2 m高度并考虑地面效应与噪声。

**📈 对比分析**

实验通过与传统RMLP和NARX模型在相同仿真数据上训练和闭环预测进行对比。结果显示：MODERNN在小批量（n_tr=5–20）下即可获得最低平均误差（约0.09–0.42），训练时间约0.5–3.5 h，显著优于RMLP（误差0.12–1.01、时间5.5–9.3 h）和NARX（误差0.33–0.72、时间6–7.5 h）。

**⚠️ 局限性**

局限性包括：仅在仿真数据上验证，未针对真实飞行环境或更复杂动力学（姿态、横向控制）进行测试；对批量大小和网络深度的选择仍需经验调优；高阶非线性系统的可解释性与泛化能力尚未深入评估。

---

## 37. Embodied Multimedia: A Tutorial

**arXiv ID:** 2609.04204 | [PDF](https://arxiv.org/pdf/2609.04204v1)

**作者:** Yang Liu `[一作]` (Tongji University), Liang Song `[通讯]` (Fudan University)

**通讯引用:** 16387 | [OpenAlex ID](https://openalex.org/A5034582366)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了“Embodied Multimedia (EMM)”这一面向具身智能代理的多媒体计算范式，并设计了包含数据、通信、认知与评估四层的统一架构，系统综述了每一层的核心技术并指出了未来的前沿应用方向。

**💡 创新点**

创新点在于：①从任务驱动角度重塑多模态数据的采集、编码与传输；②提出面向具身决策的语义通信与边缘‑云协作模型；③将世界模型、VLN、VLA 等先进认知技术融入具身循环；④引入机器偏好与安全评估框架，以任务成功率而非人类感知为评判标准；⑤明确了五大前沿应用场景，为跨学科研究提供了清晰路线图。

**🔧 技术方法**

所使用的技术包括：主动多模态感知（RGB、事件相机、触觉、惯性）、语义导向增强、端到端神经压缩、语义通信与网络切片、端‑云协作计算、代理检索、生成式世界模型、Vision‑Language‑Navigation 与 Vision‑Language‑Action 模型、机器偏好评估及安全性能基准。

**📊 数据集**

主要数据集与仿真平台涵盖：Habitat/AI2‑THOR（主动感知与导航）、仿真生成的数字人动作序列、物理仿真器产出的交互场景、Machine Preference Database（约 225 万样本）、RLBench、ManiSkill、LIBERO、Open X‑Embodiment 等。

**📈 对比分析**

论文通过在标准视觉与交互基准（如 RLBench、ManiSkill、VLN‑R‑2R 等）上复现现有技术，比较了传统人类感知指标与机器偏好评估的差异，并量化了语义通信在低带宽场景下对控制延迟与任务成功率的提升；实验表明，任务导向的压缩与语义编码可在 20–30% 的带宽压缩下保持 90% 以上的操控成功率。

**⚠️ 局限性**

限制主要包括：①多模态时空对齐与传感器同步的技术瓶颈；②仿真到现实的落差导致学习策略迁移困难；③网络波动与丢包对实时控制的安全隐患；④缺乏统一的硬件约束下的评估协议；⑤现有数据集在真实世界的稀缺性与长尾异常事件的缺失。

---

## 38. Breaking the Alphabet: Rethinking File Ordering in Code Review

**arXiv ID:** 2609.04207 | [PDF](https://arxiv.org/pdf/2609.04207v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 39. Hardware-conscious Software Training for Deep Neural Network Inference Accelerator Chips to Recover Accuracy Degradation due to Hardware Variabilities

**arXiv ID:** 2609.04259 | [PDF](https://arxiv.org/pdf/2609.04259v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 40. AI Writes Code, Humans Pay the Debt. An Empirical Study on the Sustainability and Evolution of Agent-Generated Code

**arXiv ID:** 2609.04208 | [PDF](https://arxiv.org/pdf/2609.04208v1)

**作者:** Antonino Coppola `[一作]` (University of Southern Denmark), Valentina Lenarduzzi `[通讯]` (University of Southern Denmark)

**通讯引用:** 3531 | [OpenAlex ID](https://openalex.org/A5015576503)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用SQuaD数据集中的628k条闭合Issue，先收集对应的人类实现提交（HIFC），再让LLM代理在同一代码快照下生成代理实现提交（AIFC），通过静态分析工具评估两者在定位准确性、技术债务引入量以及随时间的技术债务演化三方面的差异。

**💡 创新点**

①首次从大规模真实项目数据出发系统评估LLM代理对技术债务的长期影响；②引入对比实验与对照轨迹（human vs agent）以及线性混合效应模型，对技术债务演化率进行量化；③对不同LLM模型（GLM‑5、Qwen3.5‑397b、Kimi K2.5）进行统一基准，探究模型间差异。

**🔧 技术方法**

LLM代理（OpenCode框架+三大LLM），静态分析工具（SonarQube、Understand、CodeScene），树形解析（tree‑sitter），指标归一化与聚合，统计检验（ANOVA/Friedman/Wilcoxon、Bonferroni/BH、线性混合效应）。

**📊 数据集**

SQuaD数据集（450成熟开源项目，63,586个release，110个可维护性指标），以及从中提取的628k闭合Issue与对应提交。

**📈 对比分析**

对比方法：对同一Issue在agent与human两条实现路径下的实体触发（package/class/method）进行信息检索指标评估；使用静态分析指标评估技术债务与可维护性差异；计算每条项目的release级技术债务斜率，对比human/agent轨迹。预期结果：agent在更细粒度位置定位准确率较低、技术债务引入量上升，且在release级别技术债务累积速度更快。

**⚠️ 局限性**

限制：①静态分析指标仅为技术债务的代理；②仅使用2021年前数据，缺乏最新AI代码实践；③只固定OpenCode框架，无法推断不同工具链的影响；④Issue–commit关联依赖正则表达式，可能有噪声；⑤对照轨迹假设变更可叠加，忽略交互与架构依赖。

---

## 41. Game-Theoretic Drone Swarm Defense: A Case Study in Applied Differential Game Theory

**arXiv ID:** 2609.04394 | [PDF](https://arxiv.org/pdf/2609.04394v1)

**作者:** Ross E. Allen `[一作]` `[通讯]` (Massachusetts Institute of Technology), Ross E. Allen (Massachusetts Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用差分博弈理论设计无人机编队防御策略，实现目标分配与中期引导，显著提高拦截成功率。

**💡 创新点**

提出多智能体差分博弈求解目标软分配与中期路径的框架，闭合对手反应并通过局部LQ近似+Riccati递推获得闭环平衡策略，突破传统单向优化的局限。

**🔧 技术方法**

核心技术包括：差分博弈（线性二次近似 + Riccati 递推）、多阶段（目标分配 → 中期引导 → PN终端）规划、PettingZoo 多智能体仿真环境、PYDGENS 计算库。

**📊 数据集**

实验使用自定义仿真数据，随机生成防御/攻击编队位置、速度、转弯率等参数，无外部公开数据集。

**📈 对比分析**

通过蒙特卡洛实验与贝叶斯统计比较差分博弈策略与基准最近目标（NB）和覆盖感知（CA）策略：在非规避攻击下成功率 96.8% vs 94.6%，在规避攻击下 96.8% vs 94.6%，且差分博弈在贝叶斯对比中显示 99.9% 的显著优势。

**⚠️ 局限性**

假设全局观测与无限带宽、简化无人机动力学与拦截判定、未考虑高速攻击者与完全自适应攻击者、缺乏高保真传感与飞控模型，导致模型在现实场景下的适用性受限。

---

## 42. From Matching Models to Recruiting Agents: A Systematized Narrative Review of AI Recruitment Systems, Evaluation, and Governance

**arXiv ID:** 2609.04286 | [PDF](https://arxiv.org/pdf/2609.04286v1)

**作者:** Ziyi Zhao `[一作]` (University of Chinese Academy of Social Sciences), Guanzheng Wei `[通讯]` (Southwest University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统化叙事回顾，总结并系统化了招聘 AI 领域从检索到多阶段工作流的演进，构建了统一框架（招聘阶段、自动化范围、评估单元及证据→主张映射），并基于 40 篇代表性文献、工业披露、监管审计和区域法规，绘制了详细的证据图谱。

**💡 创新点**

创新点包括：① 将招聘 AI 的工作流拆解为 6 个评估单元（领域、对、列表、案例、轨迹、结果），并与自动化级别（A0–A4）和流程权限（P0–P3）对应；② 设计了证据到主张的分级映射，明确了不同评估单元所能证明的最强主张；③ 在单一框架下综合学术、工业、监管三方面证据，揭示评估缺口；④ 提出了未来研究议程（互惠、基于证据、时间可控、可审计系统）。

**🔧 技术方法**

采用的技术手段包括：系统化搜索与编码协议（ACM、ACL、AAAI、IEEE、arXiv、监管官网等数据库），双向雪崩搜索，定量与定性编码，构建可视化证据地图与 CSV 编码文件，跨学科对照分析。

**📊 数据集**

主要利用的“数据集”是从 40 篇代表性工作中提炼的元数据与评估结果；其中包含大型公开与私有的简历-职位匹配集合、行业内部检索日志、监管审计记录和多语言招聘任务基准（如 JobMatch、ConFit、PeopleSearchBench 等）。

**📈 对比分析**

评估方法侧重于对不同评估单元的证据完整性和主张强度进行对比，未给出统一的性能数值；文中引用了 nDCG@K、Recall@K、对照实验、A/B 测试等指标，并指出单纯的离线准确率无法说明最终结果。总体而言，评估强调多级、跨阶段的证据链而非单一指标的性能提升。

**⚠️ 局限性**

局限性包括：① 仅涵盖 40 篇精选文献，未实现全面文献计量或元分析；② 主要依赖公开信息，私有数据与内部实验难以验证；③ 评估框架与主张映射虽然系统化，但在不同工作流与法律背景下仍需进一步实证验证；④ 缺少对隐私、公平与安全等多维度指标的综合评估。

---

## 43. When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models

**arXiv ID:** 2609.04453 | [PDF](https://arxiv.org/pdf/2609.04453v1)

**作者:** Berkcan Kapusuzoglu `[一作]` (Capital One), Milind Naphade `[通讯]` (Capital One)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对过度分散（over-dispersed）Mixture-of-Experts模型的专家剪枝方法MESA，以减少内存和服务成本。

**💡 创新点**

创新点在于：①识别并刻画过度分散路由的特性（Perplexity–accuracy失效、能力权衡），②设计了最小化最差域损失的minimax域公平剪枝算法MESA。

**🔧 技术方法**

技术包括：基于路由概率和激活范数的专家重要性评分、路由熵分析、域级路由质量测量、最小化最大域损失的迭代增益调节。

**📊 数据集**

数据集涵盖6个域的校准文本（数学、编码、常识、科学、指令、通用知识）以及11个评测基准（GSM8K、AIME、GPQA、MMLU、ARC、OpenBookQA、PIQA、WinoGrande、IFEval等）。

**📈 对比分析**

与随机、均匀、REAP、EvoESAP等基线比较，MESA在r=0.25时在7/11基准上优于REAP，整体保持均衡性能，且内存占用下降约25%；在标准路由模型Mixtral上表现不佳，验证了过度分散特有性。

**⚠️ 局限性**

局限包括：仅在单一剪枝比例（r=0.25）评估；缺乏对更激进剪枝（r=0.5）的系统性研究；算法为贪心启发式，未提供收敛理论；对其他过度分散模型的泛化需进一步验证。

---

## 44. Patterns of Priming in Production: Lexical, Semantic and Structural Alignment in Language Model Generation

**arXiv ID:** 2609.04484 | [PDF](https://arxiv.org/pdf/2609.04484v1)

**作者:** Giulia Pucci `[一作]` (University of Aberdeen), Arabella Sinclair `[通讯]` (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在语言模型（LM）上进行结构诱导句子完成实验，探究了前置句子结构对生成句子结构的影响，并检验了语义连贯性和词汇重复对结构诱导的调节作用。

**💡 创新点**

创新点在于首次系统评估LM在生成任务中的结构诱导效应，揭示了结构诱导与词汇/语义对齐相互促进的关系，并展示了语义连贯性能显著放大诱导效应。

**🔧 技术方法**

采用了大规模预训练语言模型（Phi、Gemma、EuroLLM、Qwen）进行句子完成，并用BERT等深度学习模型对生成句子进行结构分类；通过统计检验（z检验、Benjamini–Hochberg FDR校正）评估诱导效果。

**📊 数据集**

使用自建PrimeLMDative+数据集（≈104万句），融合PrimeLM、BLiMP、ROC故事闭塞等语料；同时收集并手工注释约2700句用于验证结构分类器。

**📈 对比分析**

比较方法：在无上下文、非Dative、语义不连贯与语义连贯四种条件下测量结构匹配率，并对比不同模型、不同诱导结构（PO vs DO）和不同基线。结果显示所有模型均表现出显著的结构诱导；DO诱导在相对增幅上更强，而PO在绝对增幅上更显著；语义连贯性进一步提升诱导率，且诱导句子在词汇/语义相似度上均更高。

**⚠️ 局限性**

局限性包括：仅在英语单一替代结构（dative）上测试；未与人类生成结果直接对比；未探索不同动词偏好和语料偏差对诱导强度的影响；模型偏好与词汇重复的机制尚未进一步解释。

---

## 45. Network Availability Enhancement in Low-Altitude HetNets: A Cross-Layer Design Perspective

**arXiv ID:** 2609.04406 | [PDF](https://arxiv.org/pdf/2609.04406v1)

**作者:** Teng Wu `[一作]` (Xidian University), Michail Matthaiou `[通讯]` (Queen's University Belfast)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种跨层优化的计算‑通信资源互换方法，用以提升低空异构网络（LA‑HetNet）的网络可用性（NA），并通过仿真验证其在高异构度下的有效性。

**💡 创新点**

① 给出异构度对 NA 的解析表达式；② 引入计算‑通信资源互换的跨层优化框架；③ 通过延迟重构将计算资源转化为通信资源，实现更高 NA；④ 证明该方法在传统仅扩充计算资源策略失效时仍能达到目标 NA。

**🔧 技术方法**

跨层优化、统一多连接（MC）广播方案、CoMP、Processor Sharing（PS）服务器模型、有限块长（FBL）信息理论、MMSE 信道估计、MRT 预编码、Monte‑Carlo 仿真。

**📊 数据集**

无公开数据集，使用仿真参数：40 UE（5 MSDC、35 LSDC）、20 FAP、服务区域半径 2000 m、飞行高度范围 100–400 m、Poisson 到达率等。

**📈 对比分析**

与基准方案（固定延迟比例 2/5）比较。仿真显示跨层优化在更高异构度下能够实现目标 NA，并在相同计算资源下保持 NA，显著优于仅扩充计算资源的传统方案。

**⚠️ 局限性**

① 分析仅考虑最坏情况（最大传输距离），未捕捉空间‑时间动态；② 延迟重构在单帧粒度内，未探索更细粒度；③ 仿真未包含实际干扰、硬件实现复杂度；④ 需 exhaustive search，尽管复杂度低，但在大规模网络下仍有限；⑤ 仅考虑集中式 DCCU，未讨论分布式实现。

---

## 46. Scalable Context Orchestration for Serving LLMs Over Voice

**arXiv ID:** 2609.04288 | [PDF](https://arxiv.org/pdf/2609.04288v1)

**作者:** Linyi Jiang `[一作]` (Shanghai Jiao Tong University), Yifei Zhu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2003 | [OpenAlex ID](https://openalex.org/A5042533299)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个声学上下文管理中间件，通过显式建模语义、声学和环境状态，实现实时LLM语音交互的上下文组织与控制。

**💡 创新点**

创新点在于：① 结合语义、声学（说话速率）和环境（网络丢包）三种状态，统一进行上下文建模与运行时决策；② 通过 VoicePage/VoiceThread 层次结构和多保真度投影，动态选择音频、转写或摘要以满足上下文预算；③ 采用 LLM 内部的函数调用进行状态协同调度，避免传统规则冲突。

**🔧 技术方法**

主要技术包括：声学特征提取（说话速率估计、WebRTC 统计）、语音转写与摘要（GPT-4o-mini）、向量检索（HNSW）、多保真度上下文投影、LLM 上下文调度与函数调用、VAD 参数动态调节。

**📊 数据集**

使用了 NIST Rich Transcription 2002（RT2）、MSP-PODCAST、ICASSP 2024 Audio Deep PLC Challenge 的验证集进行实验，并在真实旅行规划任务中进行案例评估。

**📈 对比分析**

与无调度、固定 VAD、缓存、页面级检索、压缩等基线相比，系统在说话速率一致性上误差下降 52.4%，在丢包情况下误触发率从 46% 降至 0.9%，网络诱发成本下降 79.2%，长期会话每轮成本可低至 24.9 倍，答案质量保持 98.7% 以上相当于全历史上限。

**⚠️ 局限性**

局限性包括：① 主要关注说话速率与丢包，其他声学特征（语调、情绪）未充分探索；② 需要 LLM 对函数调用的支持，依赖模型端实现；③ 在极长会话中仍需更高效的压缩与缓存策略；④ 对不同语言或方言的泛化能力未充分验证。

---

## 47. Data-Driven Learning of Unknown Nonlinear Differential Equations Using Functional Analysis

**arXiv ID:** 2609.04329 | [PDF](https://arxiv.org/pdf/2609.04329v1)

**作者:** Seyyed Shaho Alaviani `[一作]` (University of Minnesota), Gregory W. Vogl `[通讯]` (National Institute of Standards and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于功能分析与算子理论的非线性动力学系统辨识方法（FINS），通过构造连续时间积分误差的成本函数，在仅有单条状态轨迹数据的情况下，能够同时学习未知的向量场和外部激励；

**💡 创新点**

创新点包括：①将 ODE 辨识问题重新表述为功能空间的最小化问题；②用函数间距离（积分误差）代替传统离散误差，进而实现增量学习；③不依赖数值微分、神经网络或显式积分器，保持模型可解释性；④可同时推断系统动力学与未知输入，支持自适应时间变化（非自治）系统；

**🔧 技术方法**

使用技术：功能分析、算子理论、Hermite 多项式基展开、最小二乘/稀疏压缩感知、梯度消失的积分误差近似（梯形法）以及可选的稀疏化正则化；

**📊 数据集**

实验数据集：通过 MATLAB ode45 生成的合成轨迹，包括 Duffing、Van der Pol、Lorenz、Lorenz96、Mathieu-Hill、Takens–Bogdanov 等非线性系统；所有数据均为单一轨迹、可变采样率（有噪声与无噪声两组）；

**📈 对比分析**

与 SINDy、EDMD、KLT、SymNN、ODENet、NODE 等方法比较，FINS 在多种评估指标（插值误差、外推误差、噪声鲁棒性）上均优于或相近；计算时间方面，FINS 训练速度与 SINDy 相当，但增量更新时间显著低于网络基方法；

**⚠️ 局限性**

局限性：①需要完整状态/输入观测；②仅能精确逼近多项式向量场，对非多项式（如正弦、余弦）项需近似；③高维系统面临参数数量指数增长的维数灾难；④对无限数据下的收敛性与噪声鲁棒性缺乏严格理论证明；

---

## 48. Quality Recovery for Quantized KV Caches via Low-Rank Attention Adaptation

**arXiv ID:** 2609.04263 | [PDF](https://arxiv.org/pdf/2609.04263v1)

**作者:** Seifeldin Abdellatif `[一作]` `[通讯]` (Al Ain University), Seifeldin Abdellatif (Al Ain University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对低位 KV 缓存导致的质量下降，本文在保持缓存格式不变的前提下，通过低秩投影适配器恢复推理质量。

**💡 创新点**

创新点在于使用后训练的低秩 Q/K/V 投影更新，在固定低位缓存格式下实现质量恢复，而非改动缓存结构或重新量化。

**🔧 技术方法**

采用 LoRA/QLoRA 低秩适配器、物理增量缓存、KIVI/KVarN 量化方式，以及自蒸馏对齐浮点模型的技术。

**📊 数据集**

使用 WikiText‑2、WikiText‑103、TinyLlama、Gemma、Llama‑3.1 以及合成检索与 RULER 数据集进行训练与评估。

**📈 对比分析**

通过对比浮点缓存、未适配低位缓存和适配后低位缓存，在 TinyLlama、Gemma 以及 Llama‑3.1 上分别实现了约 54%–76%（TinyLlama、Gemma）或 60%–38%（Llama‑3.1 KIVI/KVarN） 的 PPL 差距恢复；2‑bit 低秩适配可使 PPL 接近浮点，但检索/长上下文性能仍有限。

**⚠️ 局限性**

局限性包括实验仅覆盖少量模型与量化方案、单种/单种子评估导致方差估计不足、检索和长期上下文的恢复不完整、以及未测量真实推理延迟、能耗与内存占用。

---

## 49. DTM: Deterministic Approaches for Black-box Test Suite Minimization with Tree-based Similarity

**arXiv ID:** 2609.04205 | [PDF](https://arxiv.org/pdf/2609.04205v1)

**作者:** Md Siam `[一作]`, Kazi Sakib `[通讯]` (University of Dhaka)

**通讯引用:** 627 | [OpenAlex ID](https://openalex.org/A5051563680)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于抽象语法树相似度的黑盒测试用例最小化框架 DTM，能够在不访问生产代码的前提下，通过三种确定性算法实现测试用例集合的最小化。

**💡 创新点**

核心创新在于将传统进化搜索方法替换为确定性策略（修改的最小生成树、谱聚类、动态规划），并利用四种树结构相似度度量来保持检测效果，保证每次运行产生相同的最小化结果。

**🔧 技术方法**

技术手段包括：Eclipse JDT 解析 AST、Top‑down、Bottom‑up、Combined、Tree Edit Distance 四种树相似度计算、基于图的最小生成树、谱聚类和动态规划算法。

**📊 数据集**

实验数据集为 Defects4J，涵盖 16 个真实 Java 项目共 661 个 buggy 版本，测试套件规模从几千行到数万行不等。

**📈 对比分析**

在 25%、50% 与 75% 的减小预算下，DTM（尤其是 Spectral Clustering + Tree Edit Distance）在 50% 预算下平均准确率达到 0.73，比基准 ATM（0.67）和 LTM（0.71）分别提升 7% 与 2.9%，执行时间仅 0.98 分钟，速度比 ATM 提升 50 倍、比 LTM 提升 2.8 倍。

**⚠️ 局限性**

局限性：实验仅覆盖 Java 语言和 Defects4J 数据集，缺乏对其他编程语言或更大规模系统的验证；同时，仅考虑了 AST 结构相似度，未探索其他可能的相似度或混合策略。

---

## 50. Iris: Climbing to the Search Frontier

**arXiv ID:** 2609.04304 | [PDF](https://arxiv.org/pdf/2609.04304v1)

**作者:** Ziyuan Liu `[一作]`, Mu Chuan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套端到端的搜索代理训练框架，先通过网页超链结构反向构造多跳问答数据，随后用教师生成轨迹并进行多级过滤，再交替进行监督微调与强化学习，最终得到在多项搜索基准上表现最优的 Iris-mini 与 Iris-pro 模型。

**💡 创新点**

创新点包括：① 用超链结构自动生成高质量、多跳且不可直接字符串匹配的训练样本；② 轨迹级与轮次级双重过滤策略，提升监督数据的可靠性；③ 迭代 SFT–RL “攀登”过程，使得每轮 RL 探索得到的高质量轨迹立即反馈到监督训练；④ 在推理时统一评估不同上下文管理策略，客观衡量模型本身与外部 harness 的贡献。

**🔧 技术方法**

采用的技术包括：ReAct 交互式推理框架；MoE 大模型（Qwen3.6‑35B 与 Qwen3.5‑397B）与 256K 上下文窗口；强化学习中的组相对策略梯度、请求级部分回放与前缀重用；in‑cluster 生成式奖励模型与观察摘要；以及多阶段压缩/去重、压缩比检测等轨迹筛选技术。

**📊 数据集**

数据集：从公开网页语义镜像（RDF）和渲染页面生成的超链接图构造的自研多跳问答；以及 Benchmark 数据集包括 BrowseComp、BrowseComp‑ZH、DeepSearchQA 与 HLE 的公开版本。

**📈 对比分析**

在四大基准（BrowseComp、BrowseComp‑ZH、DeepSearchQA、HLE）上，与同参数规模模型相比，Iris‑mini 与 Iris‑pro 分别取得 82.2/84.8/86.9/52.3 与 88.6/85.1/92.9/56.4 的最高或接近最高分；在 400B 规模下甚至超过同级别的 XYZ‑Aquila‑pro，展示了强劲的性能提升。

**⚠️ 局限性**

局限性：仍未突破最前沿大模型（>1T）性能；推理时上下文管理（如 discard‑all 与 retry）对计算成本影响较大；所构造的多跳任务虽然提升了搜索能力，但在非 Web 领域或极短路径任务中的泛化尚未充分验证。

---

## 51. An Energy-Based Conservative-Dissipative Latent Neural Evolution Operator for Magnetization Dynamics

**arXiv ID:** 2609.04530 | [PDF](https://arxiv.org/pdf/2609.04530v1)

**作者:** Sebastian Schaffer `[一作]` (University of Vienna), Lukas Exl `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于能量的自编码器+神经ODE的低阶模型，用于磁化动力学的预测

**💡 创新点**

通过在潜在空间引入保守-耗散分解的能量梯度场，保证潜在能量随时间单调下降，并通过反对称算子实现沿能量等值线的运动，从而提高长时序预测精度

**🔧 技术方法**

卷积自编码器、潜在能量网络（深度、二次或深度-二次混合）、结构化的反对称+半正定通道算子、神经ODE（Diffrax实现）

**📊 数据集**

NIST μMAG Standard Problem 4 的两种外场方向（field 1 170°，field 2 190°）的数据，包含199条不同幅值（20-95 mT）训练曲线和20条验证、40条测试曲线，网格为100×25×1，时间跨度1 ns，共101帧

**📈 对比分析**

在窗口训练（k=5）与全周期推断上对比，采用RMSE、角误差、推断时间及步数等指标；结果显示深度‑二次+反对称耗散模型在两种场下均取得最低RMSE（field 1 0.0459，field 2 0.2929），并在2 ns外推中保持较低误差，优于单纯耗散或仅二次模型

**⚠️ 局限性**

仅在单一网格与两种固定外场方向上验证；未进行超参数调优；未量化模型对不同参数空间的泛化与鲁棒性；潜在能量与真实Gibbs能量不对应；高维潜在空间可能导致参数膨胀

---

## 52. A Systematic Evaluation of Cross-Lingual Consistency Enhancement Methods in Multilingual Language Models

**arXiv ID:** 2609.04409 | [PDF](https://arxiv.org/pdf/2609.04409v1)

**作者:** Jirui Qi `[一作]` (University of Groningen), Arianna Bisazza `[通讯]` (University of Groningen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对多语言模型的跨语言一致性（CLC）提升方法进行了统一评估，比较了推理时干预与后训练（英语中心、跨语言自对齐、直接分布对齐）三类技术在三大模型族（Qwen2.5、Gemma3、Aya）和五个基准（BMLAMA、MMMLU、XCSQA、GEOMLAMA、BLEND）上的效果，并探讨其跨域泛化与对文化多样性知识的影响。

**💡 创新点**

创新点在于：①提供了首个统一实验框架，统一数据、提示、候选集与评估指标，消除了不同研究间对比难题；②系统分析了后训练方法在跨域转移与文化敏感性上的限制；③揭示了直接分布对齐（DCO）在多任务中最稳健、最易泛化的优势；④首次评估了 CLC 提升对文化多样性问答的潜在风险与实际影响。

**🔧 技术方法**

主要技术包括：Inference-time Representation Intervention（INCLINE）；英语中心偏好对齐（EN-Align / MAPO）；多语言自对齐（CALM）；直接一致性优化（DCO）基于 DPO；以及标准的 LM-Evaluation-Harness 评估与 BERTScore/余弦相似度作为开放式生成的一致性度量。

**📊 数据集**

使用的数据集包括：BMLAMA（事实填空）、MMMLU（多选学术/专业知识）、XCSQA（多选常识推理）、GEOMLAMA（地理/文化事实，答案在不同语言可差异）和 BLEND（开放式日常生活文化问题）。

**📈 对比分析**

比较方法：在统一的评估协议下，对三大模型族分别在三类后训练方式和推理时干预进行实验；使用准确率和一致性指标（候选排名一致性、BERTScore、余弦相似度）进行对比。结果显示：后训练方法普遍优于推理时干预；其中 DCO 在所有模型-数据组合上提供最稳定且显著的 CLC 改善，且在跨域转移（如 BMLAMA→MMMLU、XCSQA→MMMLU）上相对更好；EN-Align 对事实问答表现良好但在多选场景敏感；CALM 提升有限。文化多样性评估显示，后训练方法在 GEOMLAMA 上未系统降低准确率，BLEND 上偶有非英语准确率下降，提示存在一定权衡。

**⚠️ 局限性**

局限性：①使用语义相似度（BERTScore/余弦）作为 BLEND 上的一致性代理，可能未能完整捕捉文化差异；②文化多样性分析仅基于 GEOMLAMA 和 BLEND，缺乏更广泛人类评测与更大规模数据；③对文化依赖性的表示分离分析可能受输入表面差异（实体分布、翻译风格）影响，未能完全证明模型对文化特征的内在编码。

---

## 53. Client-Side Probing of Deleted Ridge Statistics in Federated Unlearning

**arXiv ID:** 2609.04475 | [PDF](https://arxiv.org/pdf/2609.04475v1)

**作者:** Yijun Quan `[一作]` (University of Warwick), Giovanni Montana `[通讯]` (University of Warwick)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用多轮分类器发布的攻击，能够从服务器的线性分类器中识别并恢复聚合训练统计，进而实现已删除样本、类或客户端信息的重放与重建。

**💡 创新点**

在精确岭回归联邦无学习框架下，首次给出完整的可识别条件与最优探测构造，并证明在可接受的采样矩阵下可通过攻击者自己的样本实现状态识别。

**🔧 技术方法**

使用理论分析（矩阵秩、伪逆、最小二乘）、数值实验（矩阵条件数评估、误差传播）以及自定义的“moment‑only”探测构造。

**📊 数据集**

实验基于MNIST和CIFAR‑10图像数据集，特征提取分别采用冻结的ResNet‑18（512维）和DINOv2 ViT‑B/14（768维）。

**📈 对比分析**

在高精度（float64）下，所有测试的单样本、类级和客户端级删除均能完美恢复，误差均在10⁻⁸到10⁻¹⁰范围；在低精度（float32）下，单样本恢复率下降显著，但类/客户端级聚合恢复仍保持较低误差。

**⚠️ 局限性**

局限性包括：需要服务器在每次更新后返回完整分类器并保证无并发更新；攻击者必须拥有足够多且特征维度完全的样本；在实际系统中，重复提交、所有权校验和批量/噪声化回应可部分抑制攻击。

---

## 54. SH-PDOPS: AI-Driven Cloud Native Enterprise Reliability Framework for Predictive Analytics and Intelligent DevOps Automation

**arXiv ID:** 2609.04210 | [PDF](https://arxiv.org/pdf/2609.04210v1)

**作者:** Ayushman Bosu Roy `[一作]` `[通讯]` (Global Institute of Technology), Ayushman Bosu Roy (Global Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 SH-PDOPS，一个集成预测分析与 DevOps 自动化的自愈云原生可靠性框架。

**💡 创新点**

创新点包括：① 将运行时遥测与 CI/CD 元数据统一收集，构建可解释的风险评分模型；② 通过政策驱动的三层响应（建议、半自动、自动）实现可审计的自愈；③ 在 Kubernetes 与 Jenkins 的协同控制平面上实现预测驱动的决策链路。

**🔧 技术方法**

使用技术包括：Docker、Kubernetes、Jenkins、Prometheus/Grafana 等监控栈；Python/Go 进行遥测收集与 AI 评分；逻辑回归/树模型等预测模型；政策引擎与 API 网关实现自动化动作。

**📊 数据集**

数据集来源：在铁路云原生示例环境中收集的真实生产遥测（容器、节点、日志、管道指标）和人工注入的故障测试数据。

**📈 对比分析**

对比方法：三种运维模式——手工监控、规则触发脚本、SH-PDOPS。评估指标为 MTTD、MTTR、误报率、自动化覆盖率和系统开销。结果显示 SH-PDOPS 在 MTTD 约 2–5 秒、MTTR 约 15–30 秒方面明显优于手工和规则模式，但相对提升了中等级别的系统开销。

**⚠️ 局限性**

局限性包括：① 需要对集群遥测、CI/CD 元数据的完整访问和权限管理；② 预测模型易出现漂移，需要定期维护；③ 高风险动作仍需人工审批，自动化覆盖有限；④ 额外的遥测处理开销在高吞吐量场景下可能影响性能。

---

## 55. Why Better Models Can Create Riskier Systems: Evidence from LLM Agents in Financial Markets

**arXiv ID:** 2609.04373 | [PDF](https://arxiv.org/pdf/2609.04373v1)

**作者:** Jillian Ross `[一作]` (Massachusetts Institute of Technology), Andrew W. Lo `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究大型语言模型在金融市场等系统中的集体行为，展示更强模型可能导致系统性风险。

**💡 创新点**

提出了能力悖论框架，阐明模型能力提升与系统风险之间的非单调关系。

**🔧 技术方法**

基于代理动作分解（纠正 vs 非纠正）、回归与协方差分析，并构建仿真交易平台。

**📊 数据集**

使用MMLU‑Pro与ELO得分评估模型能力，并利用基准资产价值的合成金融市场数据。

**📈 对比分析**

通过与噪声交易者基线对比测量跟踪误差与波动率，结果表明在信息准确时更强模型提升价格发现，信息误导时则显著恶化。

**⚠️ 局限性**

实验仅在单一资产仿真环境中进行，未覆盖多资产、策略适应与机构约束，结论在其他领域仍待验证。

---

## 56. Data-Related Challenges and Requirements for Event Log Generation in Process Mining: A Systematic Literature Review

**arXiv ID:** 2609.04211 | [PDF](https://arxiv.org/pdf/2609.04211v1)

**作者:** Ghita El Alaoui Talibi `[一作]` (Technical University of Munich), Anastasija Nikiforova `[通讯]` (University of Tartu)

**通讯引用:** 1464 | [OpenAlex ID](https://openalex.org/A5086523033)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对事件日志生成中的数据相关挑战与需求进行系统文献综述，识别29个挑战并提出5类需求，探讨数据工程和需求工程技术在解决这些挑战中的应用。

**💡 创新点**

首次构建挑战与需求的系统映射，并将事件日志生成与数据工程、需求工程交叉结合，提供未来研究与实践的综合指导。

**🔧 技术方法**

采用系统文献综述方法、关键词检索、编码分析，结合数据工程技术（ETL、数据库映射、XES/XOC标准等）和需求工程技术（领域知识建模、需求验证等）。

**📊 数据集**

以59篇研究文献为基础，公开数据集已上传至 https://zenodo.org/records/18518836。

**📈 对比分析**

本文未进行实验比较或性能评估；通过文献计量和定性映射展示挑战-需求对应关系，未给出数值指标。

**⚠️ 局限性**

受检索范围（近5年、Scopus+IEEE Xplore）限制，手工映射存在主观性；未系统评估各类解决方案；未涵盖非过程领域日志；仅基于文献而非实证实践。

---

## 57. Engineering as Code: Bringing Software Engineering Methodology to Engineering Design

**arXiv ID:** 2609.04216 | [PDF](https://arxiv.org/pdf/2609.04216v1)

**作者:** Song Difei `[一作]` `[通讯]` (Huaxin Consulting, Design, and Research Institute), Song Difei (Huaxin Consulting, Design, and Research Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了Engineering as Code (EaC) 的范式，构建了基于文本的设计声明语言 ADL（Part、Mate、Layout 三层），以及相应的静态分析器 ESA，用于在设计生成阶段即进行合规检查，并在此基础上构建了 piki 开源运行时。

**💡 创新点**

创新点在于：1) 以“Design as Code”取代传统几何驱动的 CAD/BIM 表述，解决了设计意图与几何实现的耦合；2) ADL 的三层独立语法实现了语义完整、可版本化、可可视化的设计声明；3) ESA 的 L0–L4 分层规则体系让合规检查在毫秒级可执行，显著降低 false‑positive 并消除命名依赖；4) 通过 Information Representation Hypothesis 论证 AI 进展瓶颈在于缺乏可计算设计表述。

**🔧 技术方法**

主要技术包括：领域特定语言（DSL）设计、抽象语法树与类型检查、静态分析（抽象解释）、Git/Ci/CD 集成、自动化规则库生成与 AI 辅助规则构建、以及 YAML/JSON 作为存储格式。

**📊 数据集**

使用了三个跨领域示例数据集：Telecom Rack Expansion、Modular Datacenter、Mechanical Keyboard Assembly（共 69 条规则），并在 piki 原型上执行违规注入实验（15 条故意违例）。

**📈 对比分析**

与传统几何基 ACC 的比较表明：ESA 在 200 ms 内完成 30 条规则的 L0–L4a 检查，检测率 100% 且无误报；ACC 需对模型重建与命名约束进行前处理，且容易产生数十甚至数百个 false‑positive。实验验证了 ESA 在语义层面上具有更高的信噪比和更快的反馈周期。

**⚠️ 局限性**

局限性包括：1) 对连续自由度（如管线、布线）的几何求解精度仍有限；2) ESA 的 L4b–L6 仍需依赖下游 CAD/CAE 进行精确冲突、仿真和人工签署；3) 目前只支持 YAML/TOML 文本，尚未整合对大型二进制 CAD 数据的直接导入；4) 需要进一步验证跨更复杂工程域的可扩展性和对大规模项目的性能。

---

## 58. AquaBEV: Monocular Underwater BEV Occupancy with 3D Sonar Supervision

**arXiv ID:** 2609.04411 | [PDF](https://arxiv.org/pdf/2609.04411v1)

**作者:** Trung Tien Dong `[一作]` (University of South Florida), Xiaomin Lin `[通讯]` (University of South Florida)

**通讯引用:** 187 | [OpenAlex ID](https://openalex.org/A5101610451)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了使用单幅RGB图像预测水下局部鸟瞰（BEV）占据图，并提出了AquaBEV模型。

**💡 创新点**

创新点在于利用3D成像声纳在训练时提供几何监督，采用极坐标表示和因果范围解码来学习无标定的图像到BEV映射。

**🔧 技术方法**

技术包括ConvNeXt B视觉编码器、跨视图Transformer、极坐标查询、因果卷积解码、极坐标到笛卡尔的可微转换以及基于占据的交叉熵损失。

**📊 数据集**

使用了7个水下会话共110场景、约96,000对RGB与3D成像声纳数据，构成训练、验证和测试集。

**📈 对比分析**

与七种迁移自陆地占据方法（如GaussianFormer、SurroundOcc等）在统一协议下对比，AquaBEV在Visible IoU 31.4、Observed IoU 38.6、Range Macro 43.4等指标上均优于最佳基线，表现最佳。

**⚠️ 局限性**

局限在于仅在单一水下环境验证，依赖离线声纳累计产生的监督，且对光照、能见度变化、动态场景的泛化尚未评估。

---

## 59. A Removal Based Approach to Improve LLM Faithfulness at Test-Time

**arXiv ID:** 2609.04343 | [PDF](https://arxiv.org/pdf/2609.04343v1)

**作者:** Qinglan Luo `[一作]` (Massachusetts Institute of Technology), Katie Matton `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种测试时的“去除包装器”，通过先让LLM给出答案与解释，再识别并删除解释中未提及的概念，重新查询模型，使答案只受解释中被记载的概念影响。

**💡 创新点**

创新点在于针对解释的不完整性（隐藏概念）提出了无需修改模型权重、仅在推理阶段可用的去除方法，并可与现有测试时技术如模块化提示融合进一步提升可信度。

**🔧 技术方法**

核心技术包括：辅助语言模型用于概念抽取与归属识别、概念删减编辑、以及对目标模型的二次查询；配合多种可解释性度量（提示可言性、因果概念可信度、模拟可用性增益）。

**📊 数据集**

使用了两大公开基准：Hint-augmented QA（含有植入提示的多项选择题）和Bias Benchmark QA（BBQ）社群偏见问答集。

**📈 对比分析**

与基线（未处理模型）、提示增强、模块化分解等方法对比，在所有三种指标上均实现更高的可信度（如提示可言性从0.14提升到1，因果概念可信度从≈0.27提升至≈0.87），并在组合方法中进一步提升。

**⚠️ 局限性**

局限性包括：只解决不完整性，无法处理解释中记载但无因果影响的“无声”概念；依赖辅助模型准确一致的概念抽取与归属；对概念间相互依赖的处理不足，可能导致删减后语义偏移。

---

## 60. Privacy Failure in Split-LLM Training, The Returned Gradient Nullifies the Decoys

**arXiv ID:** 2609.04382 | [PDF](https://arxiv.org/pdf/2609.04382v1)

**作者:** Georgios Politis `[一作]` (Setloop.io), Evangelos Pappas `[通讯]` (Setloop.io)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对一个两节点分割式大型语言模型（split‑LLM）训练系统进行评估，发现后向梯度通道泄露了哪些行是真数据、哪些是干扰行（零梯度结构泄漏），并指出系统的隐私门在未包含该通道的情况下误判为安全。

**💡 创新点**

首次证明了在分割学习中因零梯度支持导致的结构性元数据泄漏；提出了预先声明通道、指标和门限的校准评估协议；通过逐行梯度裁剪与加噪消除了泄漏，且对模型质量影响仅约0.01 nats。

**🔧 技术方法**

使用分割学习、潜在瓶颈（宽度64）、每请求旋转/置换加权、干扰行混入、截断损失、梯度支持检测、注入式攻击、Bonferroni‑Wilson门限、逐行梯度裁剪与高斯噪声，以及频繁词元探测。

**📊 数据集**

主要使用 WikiText‑2（私有切片）作为训练语料，并在公开 WikiText‑2 子集上做鲁棒性检查；模型为 Qwen3‑0.6B。

**📈 对比分析**

与三篇近期分割‑LLM 评估（BiSR、DualGuard、Prompts to Responses）以及多种攻击族进行对比；结果显示系统通过前向隐私门但在联合视图下突破门限；加噪裁剪后泄漏被消除，模型质量仅下降约0.01 nats。

**⚠️ 局限性**

局限于单一分割‑LLM 实现；未测量的攻击族（成员资格、时序、状态、主动扰动等）；主配置未通过实用性门限；结果可能不适用于其他数据集或更大模型。

---

## 61. SharedSAE: One Feature Dictionary Across Language Models

**arXiv ID:** 2609.04344 | [PDF](https://arxiv.org/pdf/2609.04344v1)

**作者:** Daniil Ognev `[一作]` (Mohammed bin Zayed University of Artificial Intelligence), Benjamin Heinzerling `[通讯]` (Tohoku University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种共享稀疏自编码器（SharedSAE），可以在不同规模、架构和分词器的语言模型上共享一个稀疏字典，同时保留激活幅度并支持单模型推理。

**💡 创新点**

创新点在于：①通过共享字典+模型特定的编码器‑解码器对实现跨模型稀疏表示；②采用投票归一化+单模型dropout确保单模型即可选择共享稀疏码；③冻结字典后可高效地将新模型接入，保持已有标签不变；④相较于现有方法，恢复误差更低，跨模型相关性更高，且标签迁移更好。

**🔧 技术方法**

使用技术包括Top‑K稀疏自编码器、投票归一化、跨模型重缩放、交叉重构损失、模型dropout以及字典冻结+新模型适配。

**📊 数据集**

在四个约1B参数的基础模型（不同家族与分词器）上进行训练，使用对齐文本数据（同一段落通过各模型分词对齐）。

**📈 对比分析**

与单模型专属SAE、SPARC、USAE以及后置匹配方法对比；SharedSAE在平均解释方差上保留了96.6%，跨模型激活相关性提升约1.8倍，适配新模型时可达专属SAE的近似质量。

**⚠️ 局限性**

局限性包括：对训练细节（投票归一化、dropout率、重缩放权重等）高度敏感；仅在四个1B规模模型上验证，未测试更大模型或多语言；字典宽度和层深固定，可能影响可迁移性；评估指标多为代理指标，真实标签迁移效果仍需进一步验证。

---

## 62. Beyond a Universal Forecasting Selector: Demand-Conditioned Model Selection across Demand Patterns and Horizons

**arXiv ID:** 2609.04425 | [PDF](https://arxiv.org/pdf/2609.04425v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 63. Corporate Language Model (CLM): Transforming Tacit and Fragmented Enterprise Knowledge into a Sovereign, Auditable, and Executable Corporate Intelligence Layer

**arXiv ID:** 2609.04377 | [PDF](https://arxiv.org/pdf/2609.04377v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 64. A Numerical Approach to the Realizability Problems for Memoryless Nash and Epsilon Equilibria in Concurrent Multiplayer Reachability Games

**arXiv ID:** 2609.04396 | [PDF](https://arxiv.org/pdf/2609.04396v1)

**作者:** Senthil Rajasekaran `[一作]` (Universite Libre de Bruxelles), Moshe Y. Vardi `[通讯]` (Rice University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在有限位数表示（ℓ-bit）约束下，概率并发状态游戏中内存无关 Nash 与 ε‑Nash 均衡的可实现性问题，并提出了压缩表（compressible table）作为转移函数的新表示，进而给出了 NP 以及 ∃ℝ 复杂度上界与下界。

**💡 创新点**

创新点包括：① 设计压缩表模型，兼顾高效查询与对局面游戏的紧凑表示；② 对 ℚ 及其根扩张域 ℚ(r₁…r_m) 上的数值表示进行符号运算与比较，证明 NP 可验证；③ 修正文献中关于 ∃ℝ 证明的错误，并给出完整的可达性谓词。

**🔧 技术方法**

主要技术：利用记忆无关策略构造 Markov 链与 MDP，符号处理基于根扩张域的代数运算，基于多项式位数比较的判定方法，以及在 ∃ℝ 句子中嵌入可达性谓词。

**📊 数据集**

无实验数据集，全部为理论证明与算法构造。

**📈 对比分析**

对比方法：与现有 NP‑完整或 ∃ℝ‑完整结果对比；表现为在 ℓ‑可表示约束下问题为 NP‑complete，去除约束后 Nash 均衡可实现性为 ∃ℝ‑complete；ε‑均衡在 ℝ 上的复杂度仍未确定。

**⚠️ 局限性**

局限性：ε‑均衡在 ℝ 上的精确复杂度未解决；若扩张域通过生成元而非基底给定，NP 上界不再成立；依赖于 ℓ 的无穷表示与基底大小的多项式假设。

---

## 65. Huawei's $τ$ Chip Was Supposed to Melt?

**arXiv ID:** 2609.04287 | [PDF](https://arxiv.org/pdf/2609.04287v1)

**作者:** Tingbo He `[一作]` `[通讯]` (Huawei Technologies Co), Tingbo He (Huawei Technologies Co)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文通过在华为Kirin 2026 SoC中引入三维混合键合技术和逻辑折叠，实现了晶体管密度提升55%且功耗降低66%。

**💡 创新点**

创新点在于将焦点从器件尺寸缩放转移到延迟缩放，以垂直堆叠和短化互连长度来削减动态功耗，从而突破热量限制。

**🔧 技术方法**

所用技术包括三维混合键合、逻辑折叠、设计空间探索以及实时功耗测量与基准测试。

**📊 数据集**

使用的数据集主要是Kirin 2026、Kirin 2027 与Kirin 9030 Pro 的硅片测量结果，并以 AI 推理、游戏帧率、HNX 与 Geekbench 6 等基准来评估性能。

**📈 对比分析**

比较方法为同等性能（iso‑performance）与极限（turbo）两种模式下对比平面与折叠芯片的频率、电压、功率及功率密度，结果显示折叠NPU功耗下降66%，GPU 58%，CPU 41%，同时性能提升至 70 TOPS 等。

**⚠️ 局限性**

局限性包括低并行度 CPU 核及 DSP 在第一代折叠中功率密度仍升高，热管理仍需改进，以及键合间距、层数等技术瓶颈限制进一步扩展。

---

## 66. Candidate Comparability Before Promotion: Conditional Validation in Adaptive Network Intrusion Detection

**arXiv ID:** 2609.04388 | [PDF](https://arxiv.org/pdf/2609.04388v1)

**作者:** Roberto Fernández-Barrios `[一作]` (University of Deusto), Pablo García Bringas `[通讯]` (University of Deusto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究网络入侵检测系统中，在概念漂移触发后如何判定并部署新模型（candidate promotion），强调在做出推广决定前必须保证候选模型与现有模型在预处理、训练样本量等方面可比，以避免因候选模型不具备足够证据而导致的负面影响。

**💡 创新点**

创新点在于将推广决策拆解为“漂移报警 → 候选构建 → 证据可比性 → 可选验证 → 推广”，并通过系统化的可比性审计和实验验证显示：候选模型的预处理归属和样本量不匹配会显著改变推广效果，验证门控在证据不足时能恢复正面价值，而在证据相等时无明显收益。

**🔧 技术方法**

主要技术包括：多样本两样本漂移检测（能量距离、MMD、KS、Jensen–Shannon）、自监督/无标签性能估计（ATC、DoC）、基于置信序列的验证门控（VBC‑SG）、随机森林、逻辑回归、MLP、SVC‑RBF 等分类器，以及在预处理上使用自包含的 scaler+PCA 以及冻结式预处理。

**📊 数据集**

实验使用了三大基准数据集：CICIDS2017（PortScan、DoS 等场景）、UNSW‑NB15（Reconnaissance 等）和 ToN‑IoT（Scanning），通过构造的“pool‑constructed progressive drift”生成多种漂移程度，并在多种证据设置下进行评估。

**📈 对比分析**

比较方法：在相同种子、相同窗口下对“从不更新”（never‑adapt）、“始终部署”（always‑deploy）、“点门控”和“严格门控”进行配对对比，并与 ATC、DoC、加权软集成、重放 50/50 以及 DDM/ADWIN 监测器进行基准对比。结果表明：在样本量不匹配时，always‑deploy 在部分场景下会产生负面效果；点/严格门控能在证据不足时恢复正面效果；在样本量相等时，门控对平均性能无显著提升；在全部漂移情况下，ATC/DoC 等无标签估计在某些场景下可与门控竞争但不具备全局优势。

**⚠️ 局限性**

局限性：实验依赖仿真生成的漂移序列，缺乏真实流量的长期验证；候选模型的预处理归属与样本量是主要可比性因素，其他因素如标签质量、时间分布等未完全覆盖；验证门控在不均衡或极端漂移时的鲁棒性尚不明确；实验结果对具体分类器、特征工程、数据集具有一定依赖性，无法直接推广到所有生产环境。

---

## 67. Adapting from Downturns: Prediction of Long-Term Conversational-Skill Development in Mental-Health Crisis Counselors

**arXiv ID:** 2609.04350 | [PDF](https://arxiv.org/pdf/2609.04350v1)

**作者:** Vivian Nguyen `[一作]` (Cornell University), Cristian Danescu-Niculescu-Mizil `[通讯]` (Cornell University)

**通讯引用:** 5642 | [OpenAlex ID](https://openalex.org/A5011012964)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种方法，利用志愿心理危机咨询员在早期对“下滑时刻”的应对方式的变化，预测其未来在引导会话朝向积极结果的能力是否会提升。

**💡 创新点**

创新点在于：①把对话“下滑时刻”(对话预测失衡点)作为学习难点；②通过对早期与后期相似下滑时刻的回应差异进行聚合，用LLM生成的“适应摘要”作为特征；③使用强化学习调优摘要生成，使其更能区分未来会提升与不提升的咨询员。

**🔧 技术方法**

技术包括：对话预测模型（用于定位下滑时刻）、语料匹配（基于余弦相似度找相似时刻）、LLM（prompt式生成适应摘要）与RL（GRPO）优化摘要生成、基于适应摘要的分类器进行后期改进检测和早期预测。

**📊 数据集**

使用危机文本热线（Crisis Text Line）匿名化数据：约1.5M对话，挑选至少完成125次对话的1488名志愿咨询员，取其前后各50次对话作为早期与后期。

**📈 对比分析**

与多种基线（全文本、仅失败文本、语言多样性、时间、剩余成功率）对比。方法在预测早期是否会提升时，在精确率、召回率、F1上均优于基线，最高F1约0.70（相较基线0.56）。

**⚠️ 局限性**

局限性：①预测准确度仍有限，主要受早期行为信号噪声和长时间跨度影响；②仅适用于志愿者心理危机咨询，未验证其他对话场景；③需要大量隐私敏感数据，外部复现困难；④仅区分最高与最低四分位改进，未给出具体改进程度。

---

## 68. Multiobjective Hypergraph Min-Cut in Quasi-Polynomial Time

**arXiv ID:** 2609.04389 | [PDF](https://arxiv.org/pdf/2609.04389v1)

**作者:** Karthekeyan Chandrasekaran `[一作]`, Weihao Zhu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在给定固定常数k的多目标超图最小割问题，提出了随机化的准多项式时间算法和PTAS，并将结果应用于超图连通性破坏等问题。

**💡 创新点**

创新点在于：①克服了图中标量化方法在任意秩超图中失效的困难；②设计了基于随机收缩与删除、单点不可行集采样的算法框架；③首次给出了多目标超图预算割的准多项式时间与PTAS，开辟了该领域的理论基础。

**🔧 技术方法**

主要技术手段包括：随机收缩与删除策略、单点不可行集的大小选择、概率论的充足-不足事件分析、层级化的递归与分支分析，以及对大边与小边的分情况处理。

**📊 数据集**

无实际数据集，研究为理论算法与复杂度分析，主要通过构造实例来证明算法性质。

**📈 对比分析**

与图中已知的多目标最小割多项式时间算法相比，超图版本只能得到准多项式时间与PTAS；相比于单目标超图最小割的伪多项式算法，新的算法在预算约束下实现了更紧凑的复杂度；在应用层面，通过将结果映射至连通性破坏问题，取得了与图情形相当的QPTAS与bicriteria‑PTAS。

**⚠️ 局限性**

局限性：①算法仍不是多项式时间，尤其在k变为输入时不可行；②对超图规模的依赖为 m^O(k log n) 或 2^O(k)·p^O(1)，在实践中可能受限；③对Pareto前沿的枚举仍可能呈指数级；④开放问题包括：是否存在多项式时间算法、Pareto点数是否可多项式化、以及是否可直接获得超图连通性破坏的PTAS。

---

## 69. A systematic literature review on logging smell detection

**arXiv ID:** 2609.04215 | [PDF](https://arxiv.org/pdf/2609.04215v1)

**作者:** Nora Madi `[一作]` (King Saud University), Manal Binkhonain `[通讯]` (King Saud University)

**通讯引用:** 277 | [OpenAlex ID](https://openalex.org/A5008841836)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对21篇关于日志气味检测的研究进行了系统文献综述，梳理了研究目标、检测技术、评估方法和使用的数据集，并总结了当前的研究空白与挑战。

**💡 创新点**

提出了一个统一的日志气味分类映射框架、识别了研究中缺乏标准化基准和评估指标的问题，并强调了大模型（LLM）在多气味检测中的潜力与未来发展方向。

**🔧 技术方法**

主要技术包括静态代码分析、机器学习（传统与深度学习）、以及基于大型语言模型（LLM）的检测与修复方法。

**📊 数据集**

数据集多来自公开的开源 Java 项目（如 Apache Tomcat、Hadoop、CloudStack 等），并结合手工标注、版本控制记录、Issue 跟踪数据等进行构建。

**📈 对比分析**

比较方法多样，既有与现有工具的基线对比，也有基于人工标注的 ground‑truth 验证；评估指标涵盖精确率、召回率、F1、AUC 等；但由于缺乏统一指标与公共基准，研究间的性能对比难以直接进行。

**⚠️ 局限性**

局限性包括：数据集与实验环境高度聚焦于 Java，导致语言泛化受限；评估方法缺乏标准化，导致结果不可复现；缺乏统一的基准与共享数据集，影响跨研究比较；以及不同研究使用的日志气味定义和标签不一致。

---

## 70. Disentangling Attention in Deep Operator Learning: A Controlled Study of Data-Driven and Physics-Informed Architectures

**arXiv ID:** 2609.04407 | [PDF](https://arxiv.org/pdf/2609.04407v1)

**作者:** Amar Alem Koric `[一作]` (Stanford University), Seid Koric `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了深度算子网络中不同注意力机制对数据驱动和物理信息化训练的影响，并系统评估了交叉注意力、主体自注意力、分辨率标记和注意力深度对预测精度的贡献。

**💡 创新点**

通过对比五种只改动单一注意力组件的 DeepONet 变体，首次将注意力机制的独立作用在相同架构下量化，从而揭示交叉注意力为最显著的提升因素。

**🔧 技术方法**

采用Transformer式多头注意力、Self‑Attention、Cross‑Attention、Tokenization、全局预混合以及物理信息化损失等技术，构建并训练 DeepONet 变体。

**📊 数据集**

使用三个 PDE 基准：一维非线性扩散‑反应、变初始条件的粘性 Burgers 方程和二维 Poisson 热传导，分别以源函数或初始条件作为输入。

**📈 对比分析**

在数据驱动和物理信息化两种训练模式下，对六个基准组合进行比较，发现包含交叉注意力的模型平均相对 L2误差可比标准 DeepONet 降低 2.4–28 倍；自注意力在二维问题中显著提升；但在一维问题中不如基线。

**⚠️ 局限性**

研究仅限于规则网格、固定维度、单一类型的输入标记，未系统探讨位置编码、三维/非规则域、以及更深 Transformer 架构对性能的影响。

---

## 71. Budgeting Bytes: A Windowed Storage Roofline and Dual-Budget Architecture Ablations for Storage-Bound LLM Decoding

**arXiv ID:** 2609.04238 | [PDF](https://arxiv.org/pdf/2609.04238v1)

**作者:** Hanhaodi Zhang `[一作]` `[通讯]`, Hanhaodi Zhang

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究低成本硬件上自回归解码的内存带宽瓶颈，提出地址确定性分类和窗口式屋顶线理论，并在微控制器和边缘设备上进行实测和大规模MoE部署的对比实验

**💡 创新点**

首次将字节/令牌(bytes‑per‑token)作为设计维度，构建地址确定性四类(A0‑A3)和窗口式屋顶线，用来预测和解释不同容量机制（密集、MoE、LUT）在多级存储中的性能表现；证明预取在带宽饱和时无效，只能通过减少字节量实现加速

**🔧 技术方法**

地址确定性分类、窗口式屋顶线理论、预取预测器(Probe)、量化(Q2_K、ternary)、微控制器仿真、边缘设备（ESP32‑S3、RK3588、Apple M4）部署、LLM训练与评估框架（TinyStories、MiniMind）

**📊 数据集**

TinyStories（486M tokens）、MiniMind中文语料、Qwen3‑30B 大模型、通用训练语料（用于probe训练）

**📈 对比分析**

与传统密集模型、不同层级MoE（预注意力前路由 vs 后路由）、LUT专家等进行对比；在子100M规模下发现预注意力路由几乎无损，LUT专家性能被动化；在边缘设备上演示预取无效，只有量化减少字节才能提升22×；实验使用token/s、loss、存储占用等指标，结果与理论屋顶线高度一致

**⚠️ 局限性**

仅在子100M规模下实验，使用单一训练配置和loss作为唯一评估指标；未对大型模型的质量（如2‑bit量化的生成质量）进行评估；实验仅涵盖自回归推理，未验证非自回归工作负载；预取预测器的泛化仅在有限模型和设备上验证，未系统扫描不同带宽/存储分层组合

---

## 72. VERGE: Verification-Enhanced Refinement for Grounded Extraction of Early-Onset Colorectal Cancer Symptoms in Clinical Notes

**arXiv ID:** 2609.04366 | [PDF](https://arxiv.org/pdf/2609.04366v1)

**作者:** Nikkie Hooman `[一作]` (Southern Methodist University), Mehak Gupta `[通讯]` (Southern Methodist University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

该论文开发了一种基于验证增强的多代理框架 VERGE，用于从临床笔记中提取早发性结直肠癌红旗症状和家族史，并提供可审计的证据链。

**💡 创新点**

创新点在于将检索增强生成、证据优先排序与可边界的验证-改进循环结合，形成四代理工作流，实现自动纠错、降低假阳性并只需极少人类介入。

**🔧 技术方法**

技术包括检索增强生成（MedCPT+UMLS）、GRADE证据优先排序、LLM 作为判别器与改进者、基于 Llama/Meta-Llama-3.1-8B 的生成模型、ROUGE/BERTScore 等检验方法，以及规则化的证据冲突处理。

**📊 数据集**

数据集来自 Parkland Health 2020 年 10 年期的电子健康记录，抽样 4,033 条 clinician 标注的 note‑finding 对，覆盖六种红旗症状与家族史。

**📈 对比分析**

与单代理提取器、rule‑based medspaCy+ConText、不同 LLM 后端进行对比，VERGE 在精确率从 0.764 提升到 0.849、F1 从 0.733 提升到 0.769、MCC 从 0.681 提升到 0.730，保持召回率基本不变。

**⚠️ 局限性**

限制包括单一机构、单一临床标注者、家族史阳性样本极少、仅为回溯性笔记抽取，缺乏多评审共识及真实筛查验证。

---

## 73. Large Language Models for Fuzz Testing in Microservices: A Systematic Literature Review

**arXiv ID:** 2609.04219 | [PDF](https://arxiv.org/pdf/2609.04219v1)

**作者:** Ying Song `[一作]` (University of Helsinki), Xiaozhou Li `[通讯]` (Free University of Bozen-Bolzano)

**通讯引用:** 2168 | [OpenAlex ID](https://openalex.org/A5100693396)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统综述了2024-2026年间关于在微服务系统中使用大型语言模型（LLM）进行模糊测试的20篇研究，构建了LLM角色与集成模式的分类体系并梳理了评估方法与效果。

**💡 创新点**

创新点在于首次在微服务与LLM模糊测试交叉领域开展系统综述，提出了LLM在黑盒模糊、代理与检索增强等多种角色的taxonomy，并指出了成本、评估统一与多服务真实度等研究缺口。

**🔧 技术方法**

主要技术包括基于提示的生成、反馈循环、多代理架构、检索增强生成、程序分析与强化学习等LLM集成方式，以及传统模糊器RESTler、EvoMaster等基准。

**📊 数据集**

使用的主要数据集为公开的基准服务（如FDIC、Spotify、RestCountries）、故意易受攻击的API（如VAmPI、CrAPI）以及部分工业级微服务集群（如ByteDance、Volkswagen）。

**📈 对比分析**

通过与传统模糊器、其他LLM或人工测试套件对比，研究显示LLM生成的输入在有效性、覆盖率和漏洞发现上略有提升，但评估结果差异较大，整体表现仍受评测环境与度量不统一影响。

**⚠️ 局限性**

主要局限在于缺乏统一的多服务微服务基准、评测指标碎片化、成本与可部署性不明，以及LLM生成测试缺乏形式化正确性保证。

---

## 74. Step Back to Move Forward: Reflection-Aware Preference Optimization for Visual Generation

**arXiv ID:** 2609.04282 | [PDF](https://arxiv.org/pdf/2609.04282v1)

**作者:** Junlong Wu `[一作]` (Tsinghua University), Tingting Gao `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在扩散模型的后训练阶段，通过强化学习实现对人类偏好的对齐；提出了反射意识的GRPO框架，利用扩散反射和反事实路径合成实现探索与自我校正。

**💡 创新点**

核心创新在于(1)扩散反射：利用弱-强引导差异对采样轨迹进行单步逆向修正，从而在梯度更新前提升探索性；(2)反事实路径合成：将被修正的轨迹转化为可直接用于策略学习的目标，实现隐式蒸馏。

**🔧 技术方法**

技术包括基于扩散模型的连续时间策略梯度（GRPO），弱-强指导的速度场差分，用随机单步反射进行主动探索，以及基于生成轨迹的优势加权最大似然和平方误差损失的训练。

**📊 数据集**

主要在FLUX‑1‑Dev图像生成和Wan‑2.1视频生成上进行实验，使用HPS v2.1、PickScore、CLIPScore、ImageReward、HPS v3等多指标以及VideoAlign-VQ等视频评估指标。

**📈 对比分析**

与基线模型（FLUX‑1‑Dev + DanceGRPO + MixGRPO）对比，RA‑GRPO在所有指标上均取得最高或相近的分数，特别是在HPS v2.1、HPS v3、ImageReward等指标上显著提升；在视频端亦优于DanceGRPO，体现了更好的视觉质量、语义一致性和时序连贯性。

**⚠️ 局限性**

局限性包括：需要手动调节反射比例，过高的反射会降低效果；计算成本略有增加；目前仅在文本-图像和文本-视频任务中验证，对其它生成任务或更复杂的奖励结构的适用性尚待进一步研究。

---

## 75. Big Questions on Software Architecture: Report of the ICSE 2026 BoF on Software Architecture

**arXiv ID:** 2609.04212 | [PDF](https://arxiv.org/pdf/2609.04212v1)

**作者:** Davide Taibi `[一作]` (University of Southern Denmark), Henry Muccini `[通讯]` (University of L'Aquila)

**通讯引用:** 3724 | [OpenAlex ID](https://openalex.org/A5030457541)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在ICSE 2026 BoF会议中组织讨论，收集并综合软件架构研究的重大问题；随后计划在ICSA 2026工作坊进一步验证和完善；旨在为未来软件架构研究制定长远议程

**💡 创新点**

首次系统性聚焦并整合软件架构领域的“big questions”，并提出社区驱动的两阶段研究流程，促进学术与实践的深度共识

**🔧 技术方法**

采用社区驱动的讨论和迭代收集方法，结合专家提交的简短立场论文、主题面板讨论、观众开放辩论以及后期的笔记合成

**📊 数据集**

无数据集或实验数据，研究对象为社区讨论产出

**📈 对比分析**

无实验或性能对比，研究侧重问题挖掘与共识形成

**⚠️ 局限性**

受限于参与者的代表性、已提交论文的主题偏倚、会议组织方式可能影响讨论焦点、以及归纳分析中的主观判断

---

## 76. FailSAE: Towards Interpretable Failure Prediction for Vision-Language Models via Sparse Autoencoders

**arXiv ID:** 2609.04276 | [PDF](https://arxiv.org/pdf/2609.04276v1)

**作者:** Jie Ma `[一作]` (Wayne State University), Yi Zhu `[通讯]` (Wayne State University)

**通讯引用:** 4555 | [OpenAlex ID](https://openalex.org/A5076130185)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于稀疏自编码器（SAE）的可解释失败预测框架，利用SAE的稀疏潜在向量作为特征，预测视觉‑语言模型（VLM）在不同失败类型下的正确性。

**💡 创新点**

创新点包括：①将SAE从被动可视化工具转变为主动预测器；②设计三阶段失败感知训练流程，使SAE潜在方向既保持可解释性又具备更强的失败区分能力；③通过潜在向量的掩码和可解释分析揭示失败时模型内部概念的变化；④探索运行时失败恢复策略（空间遮挡和潜在干预）。

**🔧 技术方法**

核心技术：稀疏自编码器（linear encoder/decoder + ReLU + L1 正则），Top‑K 掩码，三阶段训练（预训练 + 头部训练 + 联合微调），四分类失败预测头，基于潜在向量的概念解释与干预。

**📊 数据集**

使用的主要数据集：ImageNet‑1K、CIFAR‑10、CIFAR‑100；在这些数据集上构建 FailPred 数据集，涵盖正确、ID 错误、OOD 错误（四种噪声）和 ADV 错误（PGD、Patch 攻击）。

**📈 对比分析**

与四个基线（SSL、ORCA‑B、ORCA‑R、SuperMentor）比较，在三种数据集上实现了最高的四分类准确率（CIFAR‑10 88.0%，CIFAR‑100 76.3%，ImageNet‑1K 73.0%）和二分类 AUROC（CIFAR‑10 93.9%，CIFAR‑100 85.6%，ImageNet‑1K 81.4%）。 Ablation 证明三阶段训练与 Top‑K 掩码对性能贡献显著。

**⚠️ 局限性**

局限性：①SAE 需要预训练且参数量大（49,152 维）；②Top‑K 掩码的 K 值需经验调优；③对极端或新型攻击（如强对抗样本）恢复效果未完全验证；④解释性评价主要基于视觉直观，缺乏定量验证。

---

## 77. Distilled Continuous Diffusion Language Models Can Write Code in Few Steps---or One

**arXiv ID:** 2609.04531 | [PDF](https://arxiv.org/pdf/2609.04531v1)

**作者:** Fred Zhangzhi Peng `[一作]` (Duke University), Anru R. Zhang `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于连续扩散的代码生成模型 PlaidQ，并通过分布匹配与轨迹配对蒸馏将其压缩为 4、8、16 步甚至 1 步的高效生成器。

**💡 创新点**

创新点在于：① 将预训练的自回归模型转换为双向连续扩散 denoiser；② 设计 SWVR 关键字流式矩阵计算，显著降低大词表下的显存消耗；③ 开发混合 Muon–AdamW 优化器提升收敛；④ 通过分布匹配蒸馏和轨迹配对蒸馏实现极低步数（单步）高质量代码生成。

**🔧 技术方法**

技术包括连续扩散 (Plaid)、自回归模型迁移、SWVR 流式 vocab‑attention、Muon–AdamW 混合优化、分布匹配蒸馏 (DMD)、轨迹配对蒸馏、classifier‑free guidance (CFG)、多步到单步的 DDIM 逆过程。

**📊 数据集**

训练数据：Nemotron 预训练集合（50% 代码，20% 网页，18% 论文，8% 推理，4% STEM）。评估数据集：HumanEval、HumanEval+、MBPP、MBPP+、HumanEval‑Infill、SantaCoder。

**📈 对比分析**

在与同规模离散扩散模型（LLaDA、Dream、Mask DFM、Edit Flow、Open‑dCoder、oDLM）的对比中，PlaidQ 在 128 步时达 HumanEval pass@10≈25，MBPP+ pass@10≈34；CFG 进一步提升到 39.9/44.8。经过蒸馏后，PlaidQ‑D16 在 16 步时达到 HumanEval pass@10≈31.8、MBPP+≈40.5，超越 128–512 步教师；单步模型 PlaidQ‑D1 在 HumanEval pass@1≈7.07、pass@10≈8.53，已能生成可执行程序。

**⚠️ 局限性**

局限性：① 目前规模仅 0.7B，难以与 7–8B 大模型竞争；② 单步模型在 pass@1 上仍显低；③ 主要验证在代码生成任务，对自然语言生成、对话等其他任务的适用性未做深入探讨；④ 蒸馏过程对教师模型质量高度依赖，若教师表现不足，蒸馏效果有限。

---

## 78. Quantum-Assisted Memory-Efficient Training for Parameter-Intensive Wi-Fi-Based Human Activity Recognition

**arXiv ID:** 2609.04271 | [PDF](https://arxiv.org/pdf/2609.04271v1)

**作者:** To Truong An `[一作]` (Queen's University Belfast), Simon L. Cotton `[通讯]` (Queen's University Belfast)

**通讯引用:** 3899 | [OpenAlex ID](https://openalex.org/A5090482024)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种基于量子辅助的内存高效训练框架（Q‑MET），用于Wi‑Fi感知的人类活动识别（HAR），通过量子神经网络动态生成经典ResNet‑18参数并在训练过程中嵌入结构化剪枝，显著减少训练和推理所需的可训练参数。

**💡 创新点**

首次将Quantum‑Train（QT）框架与层自适应重要性剪枝（LAMP）在训练阶段同步结合，实现了对训练内存和推理模型体积的双重压缩，达到超过90%的可训练参数削减和超过80%的模型稀疏度，同时保持甚至提升分类精度。

**🔧 技术方法**

采用混合量子‑经典神经网络：参数化量子电路（PQC）+正弦嵌入 + 经典映射网络；使用量子超参数生成经典CNN权重；在训练时对全量梯度链路进行反向传播；整合层自适应重要性剪枝（LAMP）实现结构化通道剪枝。

**📊 数据集**

在公开Wi‑Fi CSI数据集UT‑HAR（7类动作）和WidAr3.0（22类手势）上进行实验验证。

**📈 对比分析**

与传统全参数ResNet‑18、宽松轻量化ResNet‑18、静态超网络以及传统训练‑后剪枝（TTP）方案进行对比。Q‑MET在UT‑HAR上实现95%参数削减、85%稀疏度，准确率仅下降0.5%（从98.08%提升至99.08%）；在WidAr3.0上实现90%参数削减、75%稀疏度，准确率下降不到2%（71.29%→70.61%）。训练内存降低≈95%，推理内存同样降低；训练时间相对传统模型增加约18–36%，但仅发生在离线训练阶段。

**⚠️ 局限性**

受限于当前量子硬件的量子比特数与深度；量子模拟/量子硬件的额外计算开销；高压缩比例对复杂数据集仍存在准确率衰减；仍需处理激活/缓冲区内存占用；在实际部署中对量子电路的可实现性与可扩展性需进一步验证。

---

## 79. Tuning Collective Patterns to Alleviate Congestion in Shared AI Clusters

**arXiv ID:** 2609.04417 | [PDF](https://arxiv.org/pdf/2609.04417v1)

**作者:** Eashan Gupta `[一作]` (University of Illinois Urbana-Champaign), Radhika Mittal `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

在共享 GPU 集群中构建 REACT 系统，动态调整通信集体模式以逃避网络拥塞

**💡 创新点**

首次在应用层仅利用流完成时间统计，通过安全节点交换（全局置换、块级置换、位置等价置换）重新配置通信拓扑，避免对网络基础设施的任何依赖

**🔧 技术方法**

实现了基于流完成时间的拥塞检测、基于关键路径的分析、可行置换集合的预计算以及基于分析模型的快速性能估计的反馈回路

**📊 数据集**

在实际的国家学术 GPU 集群（A100 + 200 Gbps Cray Slingshot）上以及使用 ns‑3 模拟的星形、Clos 与 Alibaba 生产网络拓扑上进行实验，使用 20 MB–100 MB 的 AllReduce/AllGather 消息

**📈 对比分析**

与 NCCL 原始算法及 AdapCC 生成的集体进行对比，实验显示在真实集群上 13–38% 的算法带宽提升，模拟中最高可达 75% 的提升，且尾部延迟亦显著下降

**⚠️ 局限性**

局限于仅能处理少量拥塞链路；对多作业交互未做协调；仅关注跨主机集体，无法自动处理主机内部拥塞；部署仍需手动集成在现有 CCL 或自定义库中

---

## 80. VocalCoachBench: Benchmarking Audio-Language Models on Expert Feedback for Singing

**arXiv ID:** 2609.04241 | [PDF](https://arxiv.org/pdf/2609.04241v1)

**作者:** Hayeon Bang `[一作]` (KAIST), Juhan Nam `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了VocalCoachBench，一个面向专业歌唱指导的音频-语言评估基准；

**💡 创新点**

首次将专业教练反馈拆分为可确定的结构化目标和基于主张的开放式评估，避免单一分数或单一参考答案；

**🔧 技术方法**

利用多种主流音频‑语言模型（如Qwen‑Audio、GPT‑Audio、Gemini、MiMo‑Audio等）进行评估，并设计对应的提示与解析流程；

**📊 数据集**

构建了包含515段录音、18位专业歌唱教练共12,051条原子主张的双子集数据集（同曲控制子集与多曲多段子集）；

**📈 对比分析**

与多数标签先验、随机基线对比，发现模型在同曲排序和主张级诊断准确率仍低于先验基线，开放式反馈的严格诊断命中率不足7%，但纠正有效率相对较高，表明模型在粗略识别问题上可行，但在细粒度诊断与对齐方面存在显著差距；

**⚠️ 局限性**

局限于单曲与英语、仅音频、一次性评估，缺乏多语言、多模态、交互式与纵向跟踪能力，且主张覆盖度可能低估真实教学价值。

---

## 81. Automated Deployment of Real-Time Tasks for Phased Execution on Scratchpad-Based Multicore Platforms

**arXiv ID:** 2609.04221 | [PDF](https://arxiv.org/pdf/2609.04221v1)

**作者:** Konstantin Dudzik `[一作]` (FZI Research Center for Information Technology), Jürgen Becker `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 34871 | [OpenAlex ID](https://openalex.org/A5067045121)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出一种基于模型的部署方法，自动化将实时任务适配到具有阶段执行的多核 Scratchpad 平台；

**💡 创新点**

创新点在于完全模块化的部署流程，任务与运行时二进制完全分离，部署工具直接从任务二进制中提取信息并自动生成时序、内存传输配置；

**🔧 技术方法**

采用逻辑执行时间（LET）范式、AER 三相执行模型、DMA 加速、RISC‑V PMP 内存保护、OpenSBI 运行时框架；

**📊 数据集**

使用 ROSACE 机载控制任务集作为实验数据集，并在 Xilinx VCU118 FPGA 上实现的自定义四核 Rocket‑Cores RISC‑V 系统；

**📈 对比分析**

通过对比手工部署的延迟与自动部署的调度可行性，展示了工具生成的调度事件和内存传输的时延测量，表明在给定的硬件上实现了可预测的执行且运行时开销可控；

**⚠️ 局限性**

局限性包括对 Scratchpad 与 DMA 的硬件依赖、需手工指定加载/卸载偏移、任务总大小受 Scratchpad 限制，以及对 WCET 的测量或估算仍是手工或基于实验的。

---

## 82. At Equal Inference Cost, Multi-Agent Structure Does Not Beat a Single Frozen Agent

**arXiv ID:** 2609.04217 | [PDF](https://arxiv.org/pdf/2609.04217v1)

**作者:** David Dylan `[一作]` (Trinity College Dublin), Saoirse Walsh `[通讯]` (University College Dublin)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定总 LLM 调用次数的前提下，使用反射式进化搜索对 Planner‑→‑Executor‑→‑Critic 三角色的自然语言提示进行协同优化，评估其是否能在保持推理成本不变的情况下超过单一进化 Agent。

**💡 创新点**

提出了 iso‑call invariant（按总 LLM 调用次数计费）以及三度量（搜索调用、评估调用、token）表，能客观比较结构对性能和成本的真实贡献；同时通过留一角色实验定位收益来源。结果表明在冻结 7B Backbone 上，结构并未带来可统计显著的增益，收益仅来自 Executor。

**🔧 技术方法**

技术主要包括：① 冻结的 Qwen2.5‑7B backbone；② 角色提示为可演化基因组；③ 每角色协同坐标上升（Coordinate‑Ascent）+ 反射式进化；④ CallMeter 对总 LLM 调用做上限；⑤ 角色影响率与重述率的指标。

**📊 数据集**

使用两套交互式基准：HouseHold（n=134，二元成功奖励）和 WebShop（n=80，密集 [0,1] 奖励）。

**📈 对比分析**

与未进化 Stock、单 Agent 进化、全 Team 以及单角色 LOI 进行对比。单 Agent 进化提升 0.097（p=0.021）；Team 在 1.8× 调用成本下取得最高均值 0.769，但与单 Agent 的差异为 +0.015（p=0.80，未显著）。在 WebShop 上，Team 更低，单 Agent 无显著提升。归因分析显示所有收益集中在 Executor，Planner 与 Critic 角色几乎无贡献。

**⚠️ 局限性**

局限性包括：仅使用单一冻结 7B Backbone，单一固定 Topology（Planner→Executor→Critic），仅采用坐标上升进化策略；仅评估两种基准，未探究更大模型或异构角色；未考虑动态预算或多模型协作。

---

## 83. Compute-in-Memory Attention: A Time-Domain Analog Softmax Circuit with RC-Tunable Temperature

**arXiv ID:** 2609.04266 | [PDF](https://arxiv.org/pdf/2609.04266v1)

**作者:** Ankur Singh `[一作]` (University of Wisconsin–Madison), Guojing Cong `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 1289 | [OpenAlex ID](https://openalex.org/A5089801879)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

设计并实现了一种128通道的时域模拟softmax电路，用于计算内存（CIM）产生的注意力分数，无需模拟到数字转换。

**💡 创新点**

创新点在于通过共享斜坡电压和RC衰减生成指数权重，并可通过斜坡斜率和RC时间常数实现softmax温度可编程，从而突破传统弱逆压MOS指数范围的局限。

**🔧 技术方法**

采用22 nm FDSOI工艺实现电路，使用比较器阵列、脉冲采样、RC衰减、全局共享斜坡，以及后端MemTorch硬件感知Transformer验证。

**📊 数据集**

使用nanoGPT 1.8M参数模型的Transformer进行验证，并通过随机多层级输入向量进行软max测试；未报告其他公开数据集。

**📈 对比分析**

与现有模拟/数字softmax实现对比，单元面积70 µm²、功耗105 µW、延迟243 ns、能耗25.5 pJ/输出；在Transformer验证中验证损失仅比理想softmax低2.5%，显著优于Sigmoid/HardSigmoid替代方案。

**⚠️ 局限性**

局限性包括共享斜坡线路的互连寄生随通道数增加而显著，可能需要分区或复制斜坡以维持可编程温度；同时使用外部Verilog‑A memristor模型，实际工艺中尚未实现。

---

## 84. Waves on the Walls: Empirical Characterization of mmWave Lateral Waves for Enhanced Indoor Coverage

**arXiv ID:** 2609.04429 | [PDF](https://arxiv.org/pdf/2609.04429v1)

**作者:** Apala Pramanik `[一作]` (University of Nebraska--Lincoln), Mehmet C. Vuran `[通讯]` (University of Nebraska--Lincoln)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

在实验室测量中验证并量化毫米波（60 GHz）在干墙表面产生的侧波（lateral wave），并构建其频率与距离相关的传播模型；

**💡 创新点**

首次实验证明毫米波侧波可沿建筑墙面传播，为室内固定无线接入提供新型面向导波的传输路径；

**🔧 技术方法**

使用EVK 60 GHz波束成形发射机/接收机、RF吸收层、干墙板、天线阵列波束扫瞄与SNR热图分析；

**📊 数据集**

基于干墙板实验测得的信号强度与SNR数据（无公开数据集），并以多角度、不同频率、不同距离的测量结果作为实验数据；

**📈 对比分析**

通过与理论模型的对比，测得频率指数约5.3、距离指数低于自由空间（约1–2），表明侧波在高入射角下可实现比直接波更优的衰减特性；

**⚠️ 局限性**

实验仅在低对比低损耗的干墙材料上验证，未测试高损耗墙体、角落或多层分隔墙，模型中未考虑多径叠加、波束宽度限制等实际环境因素。

---

## 85. How Much Does Corpus Choice Change Dependency-Distance Estimates?

**arXiv ID:** 2609.04223 | [PDF](https://arxiv.org/pdf/2609.04223v1)

**作者:** Sirui Chen `[一作]` (Beihang University), Sirui Chen `[通讯]` (Beihang University)

**通讯引用:** 1637 | [OpenAlex ID](https://openalex.org/A5101900248)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究比较了同一语言在Universal Dependencies 2.18中不同来源的树库得到的平均依赖距离（MDD）估计，评估了跨树库的一致性并探讨了归一化、句子长度匹配和预处理规格对结果的影响。

**💡 创新点**

创新点在于首次系统地将多重数据集（同语种不同来源的UD树库）纳入对比，使用多种一致性指标（CCC、ICC、Bland-Altman）和“多元宇宙”规范曲线分析，量化了树库选择对MDD的影响，并证明MDD既是语法驱动的普遍趋势，也是语料库条件的复合指标。

**🔧 技术方法**

技术方法包括：依赖距离计算与随机序列归一化；concordance correlation coefficient、单因素ICC、Bland‑Altman 95% 限界；句子长度匹配和多重抽样的Bootstrap估计；对12种预处理规格的规范曲线（specification‑curve）与多元宇宙设计；族群权重与文档级抽样稳定性分析。

**📊 数据集**

使用的数据集为UD 2.18的多语种树库（共38对，76个树库），配合Glottolog 5.3的语言族元数据；所有树库通过MD5/SHA-256校验，句子筛选满足5–40词长、单根、无空节点等标准。

**📈 对比分析**

比较方法显示跨树库一致性中等：原始MDD的CCC≈0.39，归一化后CCC≈0.51；树库选择解释约29%方差，随机抽样误差无法解释跨树库差异；树库替换导致约40%语言排名倒置。归一化和句子长度匹配在一定程度上减少差异，但仍未消除大部分不一致。

**⚠️ 局限性**

局限性包括：样本以印欧语为主（24/38），族群均衡不足；大多数树库对比在体裁、注册、时间段等方面不匹配；注释差异与注册混杂，难以分离；仅有3对严格可比样本；源独立性判定依赖文本哈希阈值，可能漏检同义或改写；规范曲线分析虽然展示敏感度，但未提供单一“最佳”方案。

---

## 86. Segmentation of the aorta in 4D flow MRI using 4D convolutional kernels and learning from sparse annotations

**arXiv ID:** 2609.04439 | [PDF](https://arxiv.org/pdf/2609.04439v1)

**作者:** Hinrich Rahlfs `[一作]` (Deutsches Herzzentrum der Charité), Anja Hennemuth `[通讯]` (Fraunhofer MEVIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种基于四维（3D+时间）U‑Net的全自动主动脉分割方法，可在4D流MRI中实时生成升主动脉、主动脉弓和近端降主动脉的时变分割，支持血流动力学参数的无监督提取。

**💡 创新点**

创新点主要包括：① 使用稀疏的2D+时间横截面专家标注与静态中心线相结合，构造4D稀疏标签，从而在无需完整4D标注的情况下实现多中心、多制造商的训练；② 设计了参数高效的“Hybrid 4D卷积”核，在保留时间上下文的同时显著降低模型参数与推理时间；③ 在多中心内部和外部数据集上验证了模型的鲁棒性与可迁移性，首次实现了在不同扫描协议和注释者的外部数据上保持高Dice与极佳的血流动力学一致性。

**🔧 技术方法**

技术方法包括：4D U‑Net网络架构（基于nnU‑Net改造为4D卷积），Hybrid 4D卷积核、时间重采样至32帧、稀疏标签生成（中心线半径cr与等熵厚度et参数化），Masked Dice+Cross‑Entropy损失、数据增强与SGD优化，五折交叉验证与Ensemble推理。

**📊 数据集**

训练集共268个扫描（来自8个中心、2个厂家、7个扫描仪模型），包含23健康、105 BAV、16三尖瓣狭窄、5单尖瓣主动脉瓣、131非主动脉病变人群；内部测试集32个扫描，外部测试集30个扫描（5-10分钟后对比剂、不同协议、不同注释者）。

**📈 对比分析**

与6种基准方法（3D nnU‑Net、UNETR、合成数据3D nnU‑Net、PCMRA3D、ManReg4D）比较，内部DSC最高为0.927±0.033，外部0.911±0.104，失败率内部0%，外部0.73%；ICCs≥0.954（内部）和≥0.980（外部）显示血流动力学参数（最大速度、净流量、轴向/环向WSS、直径）与专家标注高度一致；相比基准方法，4D U‑Net在低流量相位表现尤为优异，推理时间比3D nnU‑Net短约30%。

**⚠️ 局限性**

局限性包括：① 仅利用稀疏标签覆盖升主动脉、主动脉弓和近端降主动脉，未覆盖下段主动脉、分支及其他解剖方向；② 模型学习到的近圆形形状先验可能偏差尖锐或不规则病变（如瓣膜狭窄、解剖畸形）；③ 训练与评估主要基于2D+时间切面，未进行完整3D体素级验证；④ 对极端扫描方向或低信噪比序列的鲁棒性尚未充分验证；⑤ 仍需专家后期检查以确保分割质量。

---

## 87. Multi-dimensional Bias in Modeling Multi-dimensional Preferences: Evaluating the Ability of Synthetic Agents to Replace Human Participants in Conjoint Experiments

**arXiv ID:** 2609.04243 | [PDF](https://arxiv.org/pdf/2609.04243v1)

**作者:** Ho Ting Hung `[一作]`, Yiwen Zhang `[通讯]` (London School of Economics)

**通讯引用:** 8815 | [OpenAlex ID](https://openalex.org/A5100388218)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过复制六篇已有政治学共轭实验，对多种主流LLM生成的合成样本与原人类样本在分布、效应估计和程序稳健性三维度上进行对比评估。

**💡 创新点**

提出基于主张的验证层级与三维度评估框架，系统性检视合成代理在共轭实验中的可替代性与局限。

**🔧 技术方法**

使用 GPT‑4o、GPT‑4o mini、Llama 3.2/3.3、Gemini 2.5 Flash 等大型语言模型生成样本，并应用 Wasserstein 距离、Hellinger、Pearson、Weighted F1、AMCE、MM、cAMCE、RMSE、方向一致性、覆盖率及随机效应稳定性比等统计检验。

**📊 数据集**

采用六项政治学共轭实验原始人类数据：arias_changing_2022、bechtel_mass_2013、hainmueller_hidden_2015、hankinson_when_2018、ono_contingent_2019、teele_ties_2018。

**📈 对比分析**

通过分布相似度检验（Wasserstein、Hellinger、Pearson、Weighted F1）和效应估计一致性检验（AMCE相关性、RMSE、方向一致率、覆盖率），并对不同模型和温度设置进行稳健性分析。结果显示：在宏观分布与方向一致性上可接受，但在联合分布、个体匹配、效应大小、异质性和模型间稳定性上常不满足阈值。

**⚠️ 局限性**

仅针对典型共轭设计；未评估图像共轭、链式思维等先进实验形式；未检验开放式回答的推理质量；可能受到训练数据中的结果“污染”；模型性能受技术进步影响；高成本/可扩展性限制。

---

## 88. Pitch-class Steering for Diffusion-based Music Generation via Latent-space Probes

**arXiv ID:** 2609.04516 | [PDF](https://arxiv.org/pdf/2609.04516v1)

**作者:** Yushi Ye `[一作]` (Carnegie Mellon University), Yongyi Zang `[通讯]` (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过训练一个约125k参数的卷积探测器从Stable Audio Open的VAE潜在空间解码音高类别，并在推理时将其作为可微损失引导潜在空间，从而实现对生成音乐的音高控制。

**💡 创新点**

将轻量级探测器直接作为推理时的可微损失来对潜在空间进行梯度引导，而不需要改造模型或重新训练，展示了latent diffusion模型潜在空间可被解释且可被利用的特性。

**🔧 技术方法**

采用卷积探测器、二进制交叉熵损失、RMS归一化梯度更新、分类器自由引导以及音高类别框架进行实现。

**📊 数据集**

在MAESTRO钢琴演奏数据集（音频+MIDI）上训练探测器，并用9个文本提示和3种目标旋律构成27次实验进行评估。

**📈 对比分析**

与未引导基线以及随机旋转控制比较，使用旋律一致性指标、CLAP文本-音频相似度和FAD音频质量指标，结果显示在λ=0.05时平均一致性提升约2.4倍，且音质指标无显著下降。

**⚠️ 局限性**

仅针对单音钢琴数据、仅控制音高类别不包含节奏或音域，探测器在混合乐器或更复杂音乐结构中的可靠性尚未验证，且仅在Stable Audio Open模型上测试。

---

## 89. Evaluating Large Language Models for Forced Outage Risk Prediction: Benefits and Comparison to Machine Learning

**arXiv ID:** 2609.04272 | [PDF](https://arxiv.org/pdf/2609.04272v1)

**作者:** Christos Petridis `[一作]` (Temple University), Mladen Kezunovic `[通讯]` (Texas A&M University)

**通讯引用:** 12027 | [OpenAlex ID](https://openalex.org/A5007985005)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估大语言模型（LLM）在无标签天气相关强制停电风险预测中的零样本推理能力，并与传统监督机器学习模型进行对比。

**💡 创新点**

首次将LLM零样本推理应用于配电网停电风险预测，展示LLM在迁移性、可扩展性和可解释性方面的优势，并提出LLM与监督模型互补的使用框架。

**🔧 技术方法**

使用OpenAI API调用四种LLM（GPT‑4o、GPT‑4o‑mini、GPT‑4.1、GPT‑4.1‑mini）进行零样本二分类；监督模型为Logistic Regression和LightGBM；通过留一年的交叉验证评估性能；使用天气观测与预报特征聚合。

**📊 数据集**

六年（2018‑2023）中央德克萨斯州电力配电网停电记录（去除冬季暴风雨事件）与NOAA Storm Events数据库以及Open‑Meteo历史重分析天气数据。

**📈 对比分析**

对宏F1、精确率和召回率进行定量比较；监督模型宏F1最高（0.74），LLM宏F1在0.57‑0.63之间；LLM在召回率上高但精确率低；预测窗口越大性能下降，LLM更稳定；LLM在无标签场景下仍能取得相当性能。

**⚠️ 局限性**

LLM误报率高、精确率低，导致操作信任受损；对天气特征的利用有限，难以充分挖掘多维信息；未对概率输出进行校准；实验仅使用封闭API，未评估开源LLM；缺乏对不同地区、不同基础设施的泛化验证；未结合历史停电上下文或植物指数等其他信息。

---

## 90. EXAONE Forecast for Finance

**arXiv ID:** 2609.04239 | [PDF](https://arxiv.org/pdf/2609.04239v1)

**作者:** Seunghan Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种无注意力机制的金融时间序列基础模型，能够在多种金融资产上一次性进行预测。

**💡 创新点**

创新点在于：①用因果一维卷积代替自注意力实现时间轴信息混合；②用组感知池化MLP在变异维度上实现线性时间混合；③通过掩码上下文增广训练，使模型对缺失观察保持鲁棒；④在覆盖全金融资产类别的大规模金融语料上进行预训练。

**🔧 技术方法**

技术包括：Patch‑based 输入处理、实例归一化、arcsinh 转换、因果卷积、组池化MLP、残差结构、量化解码、Masked‑Context Augmentation、Pinball 损失、以及线性时间复杂度的前向推理。

**📊 数据集**

使用了一个包含 13 个子集、覆盖外汇、商品、加密资产、固定收益、股票、宏观指标等多类资产的金融语料库，并结合了 KernelSynth 合成数据与 GIFT‑Eval 通用领域数据。

**📈 对比分析**

在 FinVerse 基准上通过三层评分体系（点精度、横截面信息系数、组合回测）与 44 款基线模型对比，模型在所有三层均排名第一，整体排名总和为 3，且在所有基准中都取得了 >0.5 的头对头胜率，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：变异维度的分组方式尚未充分探索，尚未验证参数高效微调和集成方法，且仅在金融基准上评估，缺乏对通用领域的进一步验证。

---

## 91. Engineered Persuasion: Evaluating Personalized Pretexts in LLM-Generated Spear Phishing

**arXiv ID:** 2609.04410 | [PDF](https://arxiv.org/pdf/2609.04410v1)

**作者:** Jerson Francia `[一作]` (Brigham Young University), Shydra Valynn Murray `[通讯]` (Brigham Young University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一项使用 GPT‑4o 生成的钓鱼邮件中，研究评估了不同程度的工作场景个性化对受试者感知可信度和点击意图的影响。

**💡 创新点**

创新点在于将个性化深度分为四个累积层级，并结合 LLM 生成的邮件与定量加定性分析，揭示了个性化与上下文匹配对说服力的相互作用。

**🔧 技术方法**

技术主要使用 OpenAI GPT‑4o 进行邮件生成，并采用线性混合效应模型和逻辑混合效应模型进行统计分析。

**📊 数据集**

数据集由 180 名美国在职成人完成的在线问卷产生，包含 8 封基于其工作信息生成的钓鱼邮件（共 1,436 次评估）。

**📈 对比分析**

通过在同一受试者内比较四个个性化层级，发现最高层级使得可信度平均提升约 7.6 分，点击意图提升约 28%/层，且报告率相对下降。

**⚠️ 局限性**

主要局限包括：使用公开调查而非真实收件箱，结果基于自我报告，样本为便利性样本，且仅考察电子邮件，送信方式与真实攻击不同。

---

## 92. A Semantic Model of Genetic Evidence: A Step Toward Bridging the Basic-Science-Clinic Gap

**arXiv ID:** 2609.04509 | [PDF](https://arxiv.org/pdf/2609.04509v1)

**作者:** Michael Bouzinier `[一作]` (Harvard University), Dmitry Etin `[通讯]` (Forome Association)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一套遗传学证据的语义模型，并在六篇遗传学论文上进行人机协作注释验证。

**💡 创新点**

引入三类核心类、维度词汇表与条件激活机制，兼容FHIR Evidence并通过SHACL实现机器可验证，提供可扩展的AI准备基础。

**🔧 技术方法**

采用SHACL约束、Python工具链、YAML/JSON注释格式、UMLS交叉映射及AI注释与审核技能。

**📊 数据集**

使用六篇遗传学文献（人类遗传、群体遗传、模型生物、功能实验、多基因风险评分）作为注释语料。

**📈 对比分析**

通过人工标注与AI生成注释的对比，验证模型可行性；所有实例均通过SHACL校验，未进行性能基准，但展示了模型的实用性。

**⚠️ 局限性**

试点样本仅六篇单一标注者，缺乏交叉验证、临床部署和完整性能评估；部分维度仍需扩展，且模型尚未覆盖所有遗传证据类型。

---

## 93. The Anatomy of an ASR Hallucination

**arXiv ID:** 2609.04404 | [PDF](https://arxiv.org/pdf/2609.04404v1)

**作者:** Hamees Sayed `[一作]` (Smallest AI), Akshat Mandloi `[通讯]` (Smallest AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究ASR系统在环境降质与发音偏移下的幻觉现象，定位失去音频根基的终端编码器层并通过跳层干预验证其因果关系。

**💡 创新点**

首次将跳层消融与多维度表征分析相结合，证明终端编码器层是实现可读输出的必需边界，失去该层仅导致乱码或重复而非流畅幻觉。

**🔧 技术方法**

使用Conformer‑Large CTC与RNN‑T模型，结合跳层干预、线性探针、有效秩、编码器透镜、预测器消融与CKA等技术。

**📊 数据集**

利用WildASR（环境降质、口音偏移）和L2‑ARCTIC（跨语种口音）两个公开语音数据集。

**📈 对比分析**

与基线对比：跳层后4‑gram差异率从<3%激增至>90%，WER大幅上升；多模型、多条件下结果一致，表明结论稳健。

**⚠️ 局限性**

仅覆盖两种大规模Conformer模型，未涵盖不同结构或语言；幻觉率基于4‑gram规则，语义验证有限；使用贪心解码，批量大小对预测器消融影响显著。

---

## 94. MonoMoE: An Efficient Fused Mega-kernel for Quantized MoE Decoding

**arXiv ID:** 2609.04244 | [PDF](https://arxiv.org/pdf/2609.04244v1)

**作者:** Yu Gong `[一作]` (Amazon), Ashish Khetan `[通讯]` (Amazon)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5056299576)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MonoMoE，一种权重主导的持久 Megakernel，用于 FP8 量化的 Mixture‑of‑Experts (MoE) 解码，并将其集成到 vLLM。

**💡 创新点**

创新点：将 Token 维度放置在 Tensor‑Core 的 N 轴上，使用权重主导的调度，结合持久化内核、动态 warp 专化和 readiness flag 实现路由、Top‑k、量化、专家投影、激活和归约的无缝融合，消除 token‑major 的填充、预处理和多核切换开销，从而显著提升低 token 解码效率。

**🔧 技术方法**

技术手段：FP8 量化、WGMMA/UMMA Tensor‑Core、持久化 CUDA 内核、TMA 异步传输、ready‑flag 同步、生成式形状专用调度与离线调优。

**📊 数据集**

实验数据集：使用 Qwen3.5‑35B、Qwen3.5‑122B、GLM‑5.2、DeepSeek‑V3.1 这四个大规模 LLM 进行评测，并在 GSM8K 与 HumanEval 上验证生成任务的准确性。

**📈 对比分析**

对比方法与性能：与 vLLM Triton Grouped GEMM、FlashMoE‑FP8 进行对比；MonoMoE 在 NVIDIA H200 上相对 vLLM 速度提升 1.17‑1.54×、相对 FlashMoE‑FP8 提升 2.20‑3.84×；端到端平均每输出 token 延迟（TPOT）下降 9.9‑18.7%，并保持或略优于基线的任务准确率。

**⚠️ 局限性**

局限性：主要针对 B≤8 的低 token 解码；在大批量或高 Tensor‑Parallel 的场景下收益递减；通信（TP/EP）仍需单独处理，单机小批量解码仍受限于跨卡同步开销。

---

## 95. On the Abundance of Critical Points of the t-SNE Energy

**arXiv ID:** 2609.04379 | [PDF](https://arxiv.org/pdf/2609.04379v1)

**作者:** Nakul Haridas `[一作]` (North Carolina State University), Ryan Murray `[通讯]` (North Carolina State University)

**通讯引用:** 301 | [OpenAlex ID](https://openalex.org/A5073143742)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过对 t‑SNE 能量函数进行对称性分析，证明在满足离散等变性条件的输入分布下，t‑SNE 能量存在无穷多组不同的临界点，揭示其能量景观的复杂性。

**💡 创新点**

创新点在于：① 用对称群的同构映射构造“对称不变配合”并证明梯度下降保持对称性；② 通过对称性与梯度流的结合，严谨地构造出无限多条互不等价的临界点；③ 证明在欧氏空间中，常数映射不是局部极小点，进一步确认存在非平凡临界点。

**🔧 技术方法**

主要技术手段包括：对称群理论（有限等距群、生成元、轨道与基本域）；可微测度理论（纤维概率测度、配合与映射的可微变形）；能量泛函的第一、二阶变分分析；梯度流的存在性与对称性不变性证明；以及 Riemannian 几何中的距离、度量与雅可比映射。

**📊 数据集**

实验使用的人工数据集有：三维直线（Spaghetti Plot）、二维格点（Grid Twisting）、瑞士卷（Swiss Roll）、二维圆（Circle）以及在超球面与双曲空间中的点集，主要用以展示对称性不变性与能量景观的非凸性。

**📈 对比分析**

对比方法：在同一数据集上对比 vanilla 纯梯度下降与 openTSNE（含稀疏近似、Barnes–Hut 近似、动量与自适应学习率）的结果。openTSNE 在能量上取得更低值，但破坏了对称性，导致对称误差显著增大；vanilla GD 保持对称性但能量略高。性能指标主要是 KL 能量值、吸引与排斥能量、以及对称误差统计。

**⚠️ 局限性**

限制与未解问题：① 证明中多处假设为猜想（如大时间下嵌入空间一阶矩有界、对称映射的极限保持单值映射）；② 结论依赖于输入分布的连续对称性与离散对称群的存在；③ 目前仅在理论层面证明临界点存在，未判定其稳定性或是否为局部最小；④ openTSNE 等实际实现中的近似会破坏对称性，实际优化路径可能不落在理论构造的临界点上。

---

## 96. Atlas: Optimizing Deployment of Compound AI Workflows on Heterogeneous Clusters

**arXiv ID:** 2609.04513 | [PDF](https://arxiv.org/pdf/2609.04513v1)

**作者:** Milos Gravara `[一作]` (Vienna University of Technology), Stefan Nastic `[通讯]` (Vienna University of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Atlas 框架，实现对 Compound AI 工作流在满足 SLO 的前提下进行模型选择和硬件放置优化；

**💡 创新点**

创新点在于引入 Markovian Accuracy Predictor（MAP）通过局部质量转移表预测全局准确率，既降低了配置评估成本，又比传统乘积式估计更准确；

**🔧 技术方法**

使用 MAP（基于分桶的马尔可夫链）、混合整数线性规划（MILP）以及对工作流拓扑的三类运算符（线性、路由、循环）来实现优化；

**📊 数据集**

在 SQuAD 数据集上对四种工作流（RAG、RAG+Router、RAG+Refine、RAG+Router+Refine）进行实验；

**📈 对比分析**

与 PAS‑naive、PAS‑fair、IPA、Loki 等基线比较，MAP 在 Spearman 相关性上最高可达 0.947，配置评估成本比全量评估低 2.6 倍，Atlas 生成的执行计划准确率仅比 oracle 少 0.03，且在异构集群上可实现 42% 的成本下降；

**⚠️ 局限性**

局限性包括：假设质量转移是一阶马尔可夫过程；需要每个阶段可离散化的质量信号；仅覆盖线性、路由和循环三种拓扑，对更复杂的 fan‑out 等结构尚未支持；

---

## 97. Memory as transformation: LETHE, a self-referential gan-inspired architecture

**arXiv ID:** 2609.04289 | [PDF](https://arxiv.org/pdf/2609.04289v1)

**作者:** Francesco Vitucci `[一作]` (Conservatorio di Musica N. Piccinni di Bari), Francesco Scagliola `[通讯]` (Conservatorio di Musica N. Piccinni di Bari)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计并实现了一个名为LETHE的自我引用、实时音频生成系统，利用反馈延迟网络（FDN）与自适应控制循环，结合生成对抗网络（GAN）的概念实现音频参数的演化；

**💡 创新点**

创新点在于将GAN的生成器/判别器框架引入非可微分的实时音频处理，构建了一个仅使用自身输出作为训练数据的“内部数据集”机制，并通过极简感知判别器与随机梯度估计实现参数更新；

**🔧 技术方法**

使用的技术包括SuperCollider实现FDN与缓冲管理、单感知器判别器、基于单样本REINFORCE的随机扰动优化、OSC通信、以及基于谱半径的稳定性约束；

**📊 数据集**

实验使用无外部数据集，仅在启动时记录的固定或循环源中生成的五种合成信号（白噪声、冲击波、粉噪声、方波、锯齿波）作为自我生成的“DNA”与评估样本；

**📈 对比分析**

通过对比三种条件（固定GAN、固定消融、循环GAN）和对Δc22、判别器得分、奖励方差等指标的统计检验，验证了生成器的必要性（Δc22在消融实验中为0，GAN实验显著为正）并证明系统能稳定维持约0.44的判别器输出；

**⚠️ 局限性**

局限性包括：对初始“DNA”的高度依赖导致对环境敏感；判别器仅评估能量特征，未包含频谱或学习特征；随机扰动优化对参数更新的稳定性和收敛速度有限；未对长期运行稳定性进行充分评估，也未在真实音频材料上进行系统验证。

---

## 98. Languages and Recognition in a Category with Factorisation

**arXiv ID:** 2609.04346 | [PDF](https://arxiv.org/pdf/2609.04346v1)

**作者:** Harsh Beohar `[一作]`, Georg Struth `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a244defd-9560-426b-b1b1-f78ebb2b7bf9`

**🎯 论文内容**

本文提出一种基于Grothendieck纤维化的框架，用来统一描述语言的识别、合成与语法结构，推广了传统的基于单子和分解系统的语言识别理论。

**💡 创新点**

创新点主要体现在：①将语言、识别器（左T-同态）与合成器（语言本身）三者放入同一纤维化结构中，形成识别纤维化；②在具备分解系统的范畴中给出了合成器（合成对象）存在性的充分条件，并通过“翻译”构造得到语法等价的最小合成器（句法商）；③在不同代数结构（如无序、秩序、多项式）下统一推导了经典的句法同余与Myhill–Nerode同余。

**🔧 技术方法**

技术手段包括：分解系统与单子在范畴上的结合、Grothendieck纤维化的构造、对语言的内部化（取值于固定对象Ω）、利用翻译与内部算子构造右伴随实现语法商、以及通过极限/余极限的纤维化性质证明正则语言的闭包性。

**📊 数据集**

本文并未使用任何实验数据集，全部研究均为理论推导与范畴论证明。

**📈 对比分析**

由于研究以理论为主，没有与其他方法在性能或实验数据上的对比，故无法给出性能评估。

**⚠️ 局限性**

限制与不足包括：①目前仅在具备分解系统的范畴下给出语法商的存在性；②对多项式或多种结构的泛化仍处于初步阶段；③在有序/多重排序情形下，翻译构造的证明尚不完全纤维化；④缺乏实验验证和对量化语言（如权重或概率语义）的扩展。

---

## 99. VLA-Precision: Asymmetric Co-Bootstrapping for Efficient Real-World Online RL of Vision-Language-Action Models

**arXiv ID:** 2609.04355 | [PDF](https://arxiv.org/pdf/2609.04355v1)

**作者:** Chenyu Su `[一作]` (University of Science and Technology of China), Weiwei Shang `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出VLA-Precision框架，利用Asymmetric Co-Bootstrapping (ACoB)算法与ACoB-Stream体系实现大规模视觉语言动作模型在真实环境中的在线强化学习，显著提升高精度化学操纵任务的成功率和执行速度。

**💡 创新点**

1）在不同时间尺度上实现行为克隆与渐进价值校准的共引导学习，抑制价值误差导致的策略漂移；2）通过 invariant‑state decoupling 与 on‑demand streaming 的闭环经验–策略架构，显著提高在线学习吞吐量和计算效率。

**🔧 技术方法**

基于流匹配行为克隆、相对优势策略改进、K个价值估计器集成、实时人类干预反馈、冻结前缀上下文缓存、磁盘级上下文去重与滑动窗口采样、GPU加速的LoRA参数微调等技术。

**📊 数据集**

9个高精度化学操作任务（管道装载、2 mL 试管转移、玻璃杯传递、橡胶塞插入、酒精灯熄灭、移液器滴头装配、灯泡滴管转移、移液器转移与排出、管刷）在四个机器人平台（UR5e、UR5e+DexHand、双臂UR5e、Franka R3）上收集的真实交互数据。

**📈 对比分析**

与HIL‑SERL、ConRFT、Robo‑Dopamine以及两种大VLA基线（π_0、π_0.5）对比，VLA‑Precision平均成功率达到98.3%（相较π_0.5提升30.5%，相较Robo‑Dopamine提升88.3%），平均在线训练时长45.8 min，成功试验平均时长27.6 s，吞吐量提升约10.9×。

**⚠️ 局限性**

对实时人类干预依赖较高；长期多步任务中递归误差仍可能累积；多任务或跨任务迁移能力尚未充分验证；实现复杂，对硬件与显存资源要求较高。

---

## 100. Robustness and Trade-offs for Code LLMs on Protected Code

**arXiv ID:** 2609.04220 | [PDF](https://arxiv.org/pdf/2609.04220v1)

**作者:** Jin Wen `[一作]` (University of Luxembourg), Maxime Cordy `[通讯]` (University of Luxembourg)

**通讯引用:** 2309 | [OpenAlex ID](https://openalex.org/A5000695937)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估大型语言模型（LLM）在处理经过代码混淆（Obfuscation）后的程序时的表现，比较了直接推理与先恢复（Deobfuscation）两种策略在代码翻译与补全任务中的效果。

**💡 创新点**

创新点在于提出基于执行的评估框架，定义救援率与降解率衡量恢复的双重影响，并通过实验揭示模型能力是决定是否需要恢复的主导因素；同时给出针对不同模型、语言和混淆方法的操作性建议。

**🔧 技术方法**

技术上使用七个公开与闭源的代码LLM（CodeLlama-7B/70B、DeepSeek-Coder-V2、DeepSeek-R1-Qwen-14B、Qwen2.5-Coder-14B、Qwen3-Coder-30B、GPT-4.1），五种混淆手段（Identifier Rename、Dead Branch Injection、Remove Symbols、Random、CodeCipher），三种推理协议（Plain、Obfuscated、Deobfuscated），并采用HumanEval-style测试套件与Pass@1、CodeBLEU、编辑距离等多维度指标进行评测。

**📊 数据集**

使用的基准数据集为HumanEval扩展版，包含164个函数级别的翻译实例（C++/Go/Java/JavaScript→Python）和补全实例，涵盖四种源语言与相应的执行测试。

**📈 对比分析**

通过比较三种推理协议的Pass@1、救援率和降解率来评估性能。实验结果显示：强大模型（GPT‑4.1、Qwen3‑Coder‑30B）在保护代码上的Pass@1可保持≈90%，恢复对其效果不大甚至略微降低；弱模型如CodeLlama-70B在恢复时损失显著；中等模型的救援率与降解率平衡，提示可采用“先尝试直接推理再退回恢复”的策略。

**⚠️ 局限性**

局限性包括：仅评估函数级HumanEval任务，未涉及完整仓库级依赖与构建；仅使用五种混淆手段，未覆盖更复杂的控制流虚拟化等；恢复采用同一模型，缺乏独立工具或专业逆向技术的对比；实验仅采用零-shot推理，未考察微调或few-shot；结果对模型、语言和混淆方法的交互可能不完全可泛化。

---

## 101. EyeMakeYou: Identity-, Task-, and Subjective-State-Conditioned Diffusion for High-Frequency Gaze Synthesis

**arXiv ID:** 2609.04501 | [PDF](https://arxiv.org/pdf/2609.04501v1)

**作者:** Kamrul Hasan `[一作]` (Texas State University), Oleg V. Komogortsev `[通讯]` (Texas State University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种多条件扩散模型EyeMakeYou，用于生成基于身份、任务和主观状态的高频眼动轨迹。

**💡 创新点**

创新点在于同时对身份、任务和主观状态进行条件化，并结合多尺度谱、漂移一致性、事件加权局部平滑等多种损失，提升了生成信号的空间精度、身份保真度和生理特征一致性。

**🔧 技术方法**

采用条件去噪扩散概率模型（DDPM）、FiLM调制、Savitzky–Golay差分滤波、EKYT嵌入、任务和状态向量编码等技术。

**📊 数据集**

使用公开的GazeBase眼动数据库，包括322名志愿者、12,334条记录，采样率1000Hz。

**📈 对比分析**

与SP‑EyeGAN、VAE、DiffEyeSyn等基线比较，EyeMakeYou在HSS、RAN任务的U50|E50空间精度最低、U95|E95漂移误差最小，并在所有七个任务上实现0.91–0.95的EKYT嵌入余弦相似度。

**⚠️ 局限性**

局限在于仅在短时段（5s）和单台眼动仪下验证，极端用户/样本的误差仍高，主观状态相关性仍较弱，且未对跨设备、长时段或自然环境进行泛化评估。

---

## 102. Scalable Edge-assisted Fusion and Path Prediction for Connected Autonomous Vehicles

**arXiv ID:** 2609.04364 | [PDF](https://arxiv.org/pdf/2609.04364v1)

**作者:** Tyler Landle `[一作]` (Georgia Institute of Technology), Umakishore Ramachandran `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种名为Conductor的边缘服务，通过在RSU锚点下统一融合多辆CAV的中间特征，构建共享世界模型并预测轨迹，同时动态控制参与融合的CAV数和预测量，以满足300 ms的Age‑of‑Information（AoI）安全阈值。

**💡 创新点**

创新点在于：① 将多车感知合并为单一共享模型，消除每辆车的重复融合；② 设计基于遮挡感知的选择器，只选取能补全RSU视野的新信息的CAV；③ 采用预测缓存与风险预算调度，在保持AoI合规的同时最大化轨迹预测质量；④ 将融合、跟踪与预测统筹为一个实时调度问题。

**🔧 技术方法**

技术包括：中间特征融合（PointPillars+Where2Comm风格交叉注意力）、RSU锚定的BEV坐标变换、AB3DMOT多目标跟踪、MTR多模轨迹预测、基于置信图的遮挡评分、动态预算分配控制器以及NR‑MBS多播下行。

**📊 数据集**

使用CARLA‑基于的Multi‑V2X数据集（6个城镇、56个交叉口、约1.75万帧），其中CAV与RSU均携带64通道LiDAR；在此数据上评估融合、检测和跟踪性能。

**📈 对比分析**

与无选择/随机选择/Oracle选择以及仅预测自适应的基线比较，Conductor在最多31辆CAV的情况下始终保持100 % AoI合规；在高遮挡交叉口，选取4辆CAV即可实现87 %对Oracle召回率的闭合；在闭环驾驶实验中，Conductor在所有可见性×负载组合下实现≥90 %成功率，且保持≥2 s的最小碰撞时间。

**⚠️ 局限性**

局限性包括：① 对硬件加速器的依赖，A10 GPU下的阈值可能随硬件变化；② 仅在RSU锚定下的单一场景，多场景交叉时需跨域调度；③ 目前AoI目标固定为300 ms，无法自适应不同速度或网络波动；④ 预测自适应仅限于缓存与线性外推，对高速动态对象的误差仍有限；⑤ 网络鲁棒性尚未充分覆盖极端丢包或时延尾部。

---

## 103. Rethinking Indirect Prompt Injection as a Test-Time Search Problem

**arXiv ID:** 2609.04495 | [PDF](https://arxiv.org/pdf/2609.04495v1)

**作者:** Duong M. Nguyen `[一作]` (University of Illinois Urbana-Champaign), Vaikkunth Mugunthan `[通讯]` (Dynamo AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究间接提示注入攻击，提出将其建模为基于任务的攻击面上测试时搜索问题，并开发了具备搜索工具的代理攻击者。

**💡 创新点**

首次把间接提示注入视为可扩展的测试时搜索，构建了可调节计算预算的搜索主控器，并证明计算预算与攻击成功率呈正相关。

**🔧 技术方法**

使用大型语言模型（如GLM‑5.2、GPT‑5.5、Claude Opus 4.8）实现代理攻击与受害者，并通过环境探索、结构化推理和反馈评估的搜索主控器实现自适应搜索。

**📊 数据集**

在两套合成任务集合（Workspace和Retail）上评估，分别包含40/17个用户任务与6/8个注入任务。

**📈 对比分析**

对比不同计算预算下的攻击成功率（ASR@T），发现GPT‑5.5对攻击更易受害，随着token预算的增加ASR显著提升；使用全主控器的攻击者在50‑500 K token范围内成功率提升至>50%，而去除主控器或策略工具的实验结果缓慢停滞或重复性高。

**⚠️ 局限性**

实验仅在白盒环境下进行，未评估黑盒场景；任务集合为合成数据，真实世界复杂性不足；仅针对工具调用型LLM代理，缺乏跨模型泛化验证。

---

## 104. ProToMEx: Rapid, Interpretable Explanations via Structured Representations

**arXiv ID:** 2609.04265 | [PDF](https://arxiv.org/pdf/2609.04265v1)

**作者:** Athina Georgara `[一作]` (University of Southampton), Sarvapali D. Ramchurn `[通讯]` (University of Southampton)

**通讯引用:** 7598 | [OpenAlex ID](https://openalex.org/A5065527041)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于概率主题模型（PTM）的可解释方法ProToMEx，用于为任意黑盒分类器生成全局与局部解释。

**💡 创新点**

创新点：①将表格特征与类别标签转化为 bag‑of‑words 文档，允许 LDA 学习“主题”对应的高层次理由；②主题既能提供全局模型行为概览，也能在单个实例上分辨出多条独立推理路径；③通过预训练的 LDA 使得解释生成时间大幅降低（30‑40 倍更快）。

**🔧 技术方法**

技术：离线阶段生成合成样本并构造文档，计算特征条件概率和分箱权重；使用 Latent Dirichlet Allocation (LDA) 训练主题模型；在线阶段通过文档推断主题分布得到局部解释；对比 SHAP、LIME。

**📊 数据集**

数据集：合成数据（多种特征/标签组合如 F8P10L2 等）；公开数据集 4 个：Loan Approval、Lung Cancer、Wine Quality、Air Quality。

**📈 对比分析**

比较方法：在相同的黑盒模型（5 层 ReLU 神经网络）上，用 SHAP、LIME 和 ProToMEx 生成解释；评估指标包括解释精度（与真实模式/特征重要度的相似度）和计算时间。实验表明 ProToMEx 与 SHAP 在解释质量上相近（大多数实例差距 <2 重要特征），但在单实例解释时速度至少提升 30‑40 倍，显著优于 SHAP 和 LIME。

**⚠️ 局限性**

局限性：①仍未进行正式的用户研究验证解释的可理解性和实用性；②离线预训练和文档构造的过程需要一定的人工调参（如分箱数、主题数）；③在极高维特征或稀有模式时可能无法完整捕获所有重要特征；④依赖 LDA 的生成假设，若模型行为不符合主题模型的稀疏性假设，解释质量可能下降。

---

## 105. The Deterministic Hare Core Is Nonempty for Nine-Seat Approval Elections

**arXiv ID:** 2609.04497 | [PDF](https://arxiv.org/pdf/2609.04497v1)

**作者:** Jiarui Fang `[一作]` `[通讯]` (Boston University), Jiarui Fang (Boston University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

证明在任何有限的九席审批式委员会选举中都存在一个确定性的 Hare 配额核心委员会。

**💡 创新点**

将已知的 8 席核心存在性提升到 9 席，采用闭合系统、Farkas 证书、PAV 增量漂移界定、以及马尔可夫链不变性矛盾论证的组合，首次给出完整的符号与计算机辅助证明。

**🔧 技术方法**

使用符号计算、精确有理数证明、Farkas 线性规划证书、平均 PAV 漂移计算、不可约马尔可夫链分析，以及 Python/NumPy/HiGHS 等数值软件进行精确检查。

**📊 数据集**

通过枚举 7,356 种投票者类型、36 个一阻塞细胞和 19 个两阻塞模式，构建完整的计算机验证数据集；所有数据均为合成的、覆盖所有可能阻塞结构的实例。

**📈 对比分析**

本工作不涉及实验性能评估或与其他算法比较；其主要贡献是提供严格的数学证明，证明了九席核心存在性；验证通过完整的精确证书检查，确保结果的可复现性。

**⚠️ 局限性**

无法推广到任意席位数（k>9），不提供构造核心委员会的多项式时间算法；依赖计算机辅助的部分虽经过精确验证，但在理论上仍需外部软件，且仅适用于九席情况。

---

## 106. AdaptVPR: Route-Aware Hard Positive Generation for Robust Visual Place Recognition

**arXiv ID:** 2609.04369 | [PDF](https://arxiv.org/pdf/2609.04369v1)

**作者:** Shunpeng Chen `[一作]` (Beijing University of Posts and Telecommunications), Shibiao Xu `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 3318 | [OpenAlex ID](https://openalex.org/A5011919230)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AdaptVPR 框架，通过路由感知的生成式增强构造同地点难样本来提升视觉地点识别的鲁棒性。

**💡 创新点**

将视觉语言模型与规则调度结合，设计三条生成路线（全局外观、局部遮挡、混合），并通过几何与多样性验证与反射回馈自动筛选和优化生成样本。

**🔧 技术方法**

使用 diffusion 模型（IC-Light、Qwen-LightX2V）、VLM（Qwen3‑VL）、SuperPoint+LightGlue 进行几何一致性评估、CLIP 计算外观多样性、基于规则的路由调度和自适应反射生成循环。

**📊 数据集**

以 GSV-Cities 作为原始街景图像，构建 AdaptCities（160K 经过验证的同地点难样本）并在 10 个 VPR 基准（Pitts30k、MSLS、Tokyo24/7、Nordland、SVOX、Pitts250k、SF‑XL 等）进行评估。

**📈 对比分析**

在多种 VPR 基线（CosPlace、MixVPR、EigenPlaces、SALAD、BoQ、EDTformer、ImAge 等）和不同视觉基础模型（DINOv2/3）上对比实验，平均 R@1 提升 0.1%–0.7%，在跨季节、夜间、遮挡等极端域移时提升 4%–9% 以上。

**⚠️ 局限性**

生成过程依赖预设路由和阈值，几何一致性仅为代理指标，可能无法保证所有样本完全保持地点身份；离线生成成本高且需大量 GPU 计算。

---

## 107. Corten - Foundational Verification of Rust Programs

**arXiv ID:** 2609.04372 | [PDF](https://arxiv.org/pdf/2609.04372v1)

**作者:** František Farka `[一作]` (Barkhausen Institute), Sebastian Ertel `[通讯]` (Barkhausen Institute)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

在Rocq中实现了面向THIR层的Rust程序基础验证框架，并提供了程序逻辑与自动化层，成功验证了约300行的buddy allocator。

**💡 创新点**

创新点在于：①在THIR层实现完整语义与源代码保持一致；②引入双栈defunctionalized continuation的控制流敏感spec monad；③模块化层次架构支持逐步扩展与模块化证明。

**🔧 技术方法**

使用的技术包括：Iris separation logic、Rocq、Hax工具提取THIR、交互树ITrees、Dijkstra monad、defunctionalized continuation、自动化tactics（如语义推导、join‑point推理）。

**📊 数据集**

使用的数据集为：合成测试套件（涵盖数组、赋值、match、if、impl等约10类场景）以及300行Rust实现的buddy allocator案例。

**📈 对比分析**

与手写原语语义证明相比，自动化证明约短2-4倍；合成测试集显示自动化能显著提升证明规模与速度，证明体积与Rust源码相比约2-4倍。

**⚠️ 局限性**

局限性：目前仅覆盖THIR语义，未完整支持MIR级别与所有unsafe/Drop特性；对复杂控制流（多层break/continue、标签）支持有限；仍在扩展语义覆盖与自动化能力。

---

## 108. A Polymatroidal Perspective on Random Contraction

**arXiv ID:** 2609.04521 | [PDF](https://arxiv.org/pdf/2609.04521v1)

**作者:** Karthekeyan Chandrasekaran `[一作]`, Weihao Zhu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了随机收缩算法在多重体（polymatroid）中的应用，提出了一种统一的框架来理解这些算法，并引入了多重体的概念。

**💡 创新点**

创新点在于将随机收缩算法的思想推广到多重体的最小商问题，并展示了该框架如何统一现有的结果。

**🔧 技术方法**

使用了随机收缩算法，结合了多重体的性质来设计新的算法。

**📊 数据集**

使用了多重体的相关数据集，具体数据集未详细说明。

**📈 对比分析**

与现有方法进行比较，提出的算法在多重体的特定类别中能够在多项式时间内找到最小非空商，并且在参数化解的情况下表现出固定参数可解性。

**⚠️ 局限性**

限制在于多重体的最小商问题在一般情况下是NP难的，因此在某些情况下可能无法找到有效的多项式时间算法。

---

## 109. Encore: Infinite Audio-Video Generation with Adaptive Signal Routing

**arXiv ID:** 2609.04249 | [PDF](https://arxiv.org/pdf/2609.04249v1)

**作者:** Shaohua Pan `[一作]` (Baidu), Hang Zhou `[通讯]` (Baidu)

**通讯引用:** 3922 | [OpenAlex ID](https://openalex.org/A5100707855)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了 Encore 框架，支持无限长同步音视频生成，能够在同一模型下完成长时间的音视频联合生成、音频到视频以及视频到音频的跨模态生成。

**💡 创新点**

创新点在于把长音视频生成拆解为局部连续性与全局一致性两部分，利用参考信号的 RoPE 位置偏移实现全局身份锚定，并提出 Adaptive Signal Routing（ASR）机制，通过可学习的注意力偏置与残差缩放动态平衡四类条件（anchor、continuation、semantic、synchronization）以保持跨模态同步与时间连贯。

**🔧 技术方法**

技术主要包括基于 LTX-2.3 的隐式扩散变压器、SVI 误差回收、RoPE 位置偏移、ASR、文本引导的交叉注意力、音视频双向交叉注意力以及在训练中引入的多源条件处理。

**📊 数据集**

训练使用约 270,000 条 5 秒短音视频 clip，评测使用扩展后的 VerseBench，生成 30 秒长的音视频样本。

**📈 对比分析**

与 SVI、Helios（视频长生成）、OVI、LTX-2.3（短视频音视频生成）等基线比较，Encore 在身份保持、音视频同步、时间漂移、视频/音频质量指标均显著优于所有对手，用户研究亦显示被优选率超过 80% 以上。

**⚠️ 局限性**

局限性包括：使用固定参考帧，难以实现复杂镜头切换；推理过程需多步扩散，非实时；并且模型可能被用于伪造或未经授权的内容生成，需要注意使用规范与安全性。

---

## 110. EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness?

**arXiv ID:** 2609.04280 | [PDF](https://arxiv.org/pdf/2609.04280v1)

**作者:** Zixuan Ke `[一作]` (Salesforce Research), Shafiq Joty `[通讯]` (University of Wisconsin Madison)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个可控的工具、技能与专用代理（agent）随时间演进的 benchmark，专门研究外部 harness 的演化对 LLM 代理表现的影响，并在两种评估模式（部署与自适应）下系统性检验保留能力与适应性。

**💡 创新点**

创新点：①首次将 harness 的演化与代理内部自适应分离；②设计 17 条增量演化流，覆盖工具、技能、代理三轴；③提出部署与自适应评估框架，揭示“保持 vs 适应” 的相互矛盾；④揭示了 harness‑induced forgetting 与不同轴的适应性差异。

**🔧 技术方法**

技术手段：使用 GPT‑5 / Codex / Claude‑Sonnet 等 LLM；实现多种自适应策略：记忆（Raw Memory、ReasoningBank、MemToolAgent、G‑Memory、LEGOMem）、提示（GEPA）、代码（Meta‑Harness）；部署阶段无状态，适应阶段通过持久化状态 zₜ 更新；评估基于 verifier‑checkable 任务与 Forward/Backward Transfer 指标。

**📊 数据集**

数据集：基于 EnterpriseOps‑Gym 与 Agentic Last Exam 的 802 个任务，构成 1,510 个评估样本，涵盖 520 个可执行工具、42 个隐式技能、62 个专业代理；通过频率排名分桶生成 17 条工具/技能/代理演化流。

**📈 对比分析**

比较方法：对比任务特定参考（仅暴露必要能力）与累积 harness；对比部署模式（无自适应）与自适应模式；对比多种自适应技术。结果显示：在工具轴上，部署可提升准确率但成本高；在技能轴上，部署几乎无影响，自适应提升有限；在代理轴上，部署与自适应表现高度依赖环境，往往出现显著遗忘。自适应在某些情况下可提升 Forward Transfer 但往往伴随 Backward Forgetting，且不同方法效果差异大。

**⚠️ 局限性**

局限性：①仅考虑 monotonic harness 增长，未研究能力淘汰或替换；②实验仅在固定 LLM 与有限任务集上进行，缺乏跨模型验证；③自适应方法多基于经验工程，未深入理论分析；④benchmark 的生成依赖频率统计，可能忽略功能相似度等细节。

---

## 111. Grounded Decoding for Autoregressive Speech Enhancement via Adaptive Code-Space Grounding and Local LLM Refinement

**arXiv ID:** 2609.04245 | [PDF](https://arxiv.org/pdf/2609.04245v1)

**作者:** Hao Shi `[一作]` (Kyoto University), Xugang Lu `[通讯]` (National Institute of Information and Communications Technology)

**通讯引用:** 3921 | [OpenAlex ID](https://openalex.org/A5034792613)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在语音增强任务中，本文提出了一种基于证据约束的自回归生成框架：先用 Whisper‑guided DPRNN 生成与观测紧耦合的增强波形，再与原始噪声混合并离散化为 FSQ 证据；随后用 Whisper 表征与证据共同条件化自回归 LLM 生成清晰语音，并在解码时采用 Code‑Space Grounding (CSG) 以及 SNR‑Conditioned CSG 对生成轨迹进行约束，并通过 Grounded Neighborhood Refinement with LLM Ranking (GNR‑LLM) 在 FSQ 邻域内进行局部重排序，以提升内容保真与自然度。

**💡 创新点**

创新点主要有：① 将确定性增强器的输出视为观测耦合证据而非最终目标；② 利用 FSQ 的因子化几何结构定义 Hamming 距离，并在解码时引入 CSG 对不受观测支持的生成偏差进行惩罚；③ 通过残差 SNR 估计自适应地为每个句子选择 grounding 强度（SNR‑CSG）；④ 采用无训练的 GNR‑LLM 在局部 FSQ 邻域内对 anchor 进行 LLM 级重排序，既恢复自然度又保持内容一致。

**🔧 技术方法**

核心技术包括 Whisper‑guided Dual‑Path Recurrent Neural Network (DPRNN) 进行确定性增强；CosyVoice 3 语音分词器将语音离散化为 FSQ 码；BERT/LLM 作为自回归生成器；FSQ Hamming 距离作为 grounding 兼容度；残差 SNR 计算与量化映射；以及基于 Top‑K 与 Hamming 邻域交叉的 GNR‑LLM 重排序。

**📊 数据集**

实验使用合成的 LibriSpeech–DNS 语料（960 小时 LibriSpeech 与 DNS 2020 噪声混合，SNR 取值 -5~20 dB）以及官方 DNS Challenge no‑reverb 评测集。

**📈 对比分析**

与纯确定性增强（DPRNN）、Whisper‑guided DPRNN、以及无 grounding 的自回归模型（如 B5 Free）比较。结果显示：SNR‑CSG 在内容准确性（WER）上明显优于无 grounding，且通过 GNR‑LLM 能在不显著降低 WER 的前提下提升感知质量（P808/UTMOS）。在控制 SNR 评测中，SNR‑CSG 在低 SNR 条件下将 WER 由 26.7% 降到 24.4%，GNR‑LLM 进一步提升至 23.9%；在 DNS no‑reverb 条件下，SNR‑CSG/WER=8.4% 与 GNR‑LLM/WER=8.6%，均优于基线。

**⚠️ 局限性**

局限性包括：① 仍可能继承确定性增强中的局部失真，尤其在极低 SNR 时 GNR‑LLM 对错误的修正受限；② 依赖残差 SNR 估计，未针对混响等卷积失真展开；③ 由于使用无训练的重排序，缺乏对复杂场景下错误模式的学习，可能无法彻底消除 catastrophic errors。

---

## 112. Cultural Misalignment in Large Language Models: Detection, Measurement, and Mitigation Through Targeted Fine-Tuning

**arXiv ID:** 2609.04485 | [PDF](https://arxiv.org/pdf/2609.04485v1)

**作者:** Antoni Czolgowski `[一作]` (University of Colorado Boulder), Abel Iyasele `[通讯]` (University of Colorado Boulder)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了三款开源大语言模型在不同文化人群上的回答分布与真实世界数据的匹配程度，并通过针对性 LoRA 微调尝试减轻跨文化偏差。

**💡 创新点**

首次将最差表现的人群（worst‑case personas）与 LoRA 微调结合，用极少量训练对样本和单 GPU 训练时间内显著提升模型在目标子群的对齐度，并揭示了微调后偏差重分布的“搬墙”效应。

**🔧 技术方法**

采用了 1‑Wasserstein 距离度量、Bootstrap 置信区间、LoRA 参数高效微调（针对 5 个 worst‑case personas）以及多国人群分层评估。

**📊 数据集**

使用 World Values Survey（WVS）第七波的宗教重要性问题 Q164 作为评估标准，构建了 63 个跨国、跨性别、年龄、教育层级的人群 persona。

**📈 对比分析**

对比三种模型（Gemma3‑12B、Bielik‑11B‑v3、Qwen3‑4B）在 63 人群上的 Wasserstein 误差，发现无模型偏好本土文化；LoRA 微调后，Bielik 在其 worst‑case personas 上下降 16.8%（p=0.002, d≈-4.4），但 Qwen 未见改善并反而加剧偏差。

**⚠️ 局限性**

局限包括仅评估单一宗教价值问题、仅覆盖三国（中国、斯洛伐克、美国）、斯洛伐克替代波兰、未验证不同 Prompt 或多元化问卷的稳健性，以及 LoRA 仅能产生全局性偏移，易引发重分布问题。

---

## 113. On-board ML for Trace Gas detection in Imaging Spectroscopy data

**arXiv ID:** 2609.04458 | [PDF](https://arxiv.org/pdf/2609.04458v1)

**作者:** Vít Růžička `[一作]` (Jet Propulsion Laboratory), David R. Thompson `[通讯]` (Jet Propulsion Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在东京野外飞行任务中部署并使用轻量化机器学习模型，对AVIRIS‑5光谱数据进行机载实时甲烷点源检测，并将预测结果通过受限通信通道实时下行。

**💡 创新点**

创新点在于首次实现机载实时检测甲烷点源，采用极小化模型实现低延迟推理，并通过将预测结果压缩为向量形式，克服了航空/航天通信瓶颈，实现了“随航随测”。

**🔧 技术方法**

技术包括：DN→辐射率的快速预处理（省略OSF、坏像素/鬼影校正等步骤）；非迭代匹配滤波器；轻量级端到端模型 HyperEfficientViT(ConvUp) 与基于匹配滤波器的 U‑Net 结构；在NVIDIA Jetson Orin Nano Super（测试台）和NASA Langley 机载计算平台上部署。

**📊 数据集**

使用的数据集为：东京野外飞行（Tokyo‑FC 2026）收集的 AVIRIS‑5 原始光谱数据，以及 EMIT 任务的 L1A/L1B 流程数据用于基准比较。

**📈 对比分析**

比较方法：对比传统地面匹配滤波 + 手工检查的处理流程与机载 ML 推理；在测试台 CPU 上完整 L1A‑L1B 处理耗时约 2088s，改进版仅 2.61s（CPU）/0.46s（GPU）；机载推理平均 2.59±0.93s，满足 67.5s 数据帧周期，显著低于传统地面处理时间。

**⚠️ 局限性**

限制包括：模型目前仅针对单一甲烷气体，未实现多物种检测；对预处理步骤的依赖导致在极端光谱噪声或极低辐射率条件下可能误检；受限算力和通信带宽限制了更复杂模型的部署与多源数据的即时融合。

---

## 114. Where Appearance Fails, Geometry Recognizes: A CAD-Free 3D Shape Prior That Complements Vision Foundation Models

**arXiv ID:** 2609.04381 | [PDF](https://arxiv.org/pdf/2609.04381v1)

**作者:** Chenxi Tao `[一作]` (Georgia Institute of Technology), Seung-Kyum Choi `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在没有CAD模型的情况下，利用短时间的物体中心扫描生成3D Gaussian Splatting 重建，提取几何先验，并与冻结的 DINOv2 视觉特征融合，实现实例识别。

**💡 创新点**

创新点：①短扫描即可恢复 CAD 识别价值；②几何先验在形状可辨识度高时显著提升，且与视觉特征互补；③不需要训练，手工 45 维几何描述子已达到学习编码器的效果；④验证了几何先验对遮挡、纹理缺失等情形的鲁棒性。

**🔧 技术方法**

使用技术：3D Gaussian Splatting 重建、RGB‑D/多视角深度获取、点云去噪、手工构造的 45 维几何描述子、冻结的 DINOv2 视觉基础模型、轻量化融合头以及多种视觉编码器对比。

**📊 数据集**

实验数据集：T‑LESS（28 个纹理缺失的工业零件）和 HOPE（28 个纹理丰富的日常物品），均来自 BOP 基准。

**📈 对比分析**

方法对比：与单独使用图像特征、单独使用几何特征、以及 14 种不同冻结视觉编码器进行对比。结果显示：在 T‑LESS 上融合提高约 3%（0.560→0.591），在 HOPE 上几何单独已接近极限（0.920），融合仍提升但受限；在不同光照、遮挡、重建源（depth、GS、CAD）下均表现稳健。融合权重对性能影响平滑，固定 λ=0.35 在两数据集上均可获得稳定提升。

**⚠️ 局限性**

局限性：①固定融合权重在 HOPE 这类几何主导场景下并非最优；②仅在已裁剪的标注框上评估，未验证完整检测‑识别流水线；③未在大规模物件数或真实工厂环境下测试；④几何先验在极端遮挡或高度相似的零件仍有限；⑤依赖精确的扫描姿态，需先完成姿态恢复。

---

## 115. Beyond SDR: How Music Source Separation Reshapes Rhythm-Relevant Signal Properties

**arXiv ID:** 2609.04224 | [PDF](https://arxiv.org/pdf/2609.04224v1)

**作者:** Chuxin Ding `[一作]` `[通讯]` (Universitat Autonoma de Barcelona), Chuxin Ding (Universitat Autonoma de Barcelona)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文对四种代表性音乐源分离模型在MUSDB18‑HQ测试集上的表现进行评估，提出并检验了模型在节奏关键属性（攻击、动态、定位）上的系统性偏差，并比较了传统 SDR 与分析师关注的时序/动态质量指标之间的关联。

**💡 创新点**

创新点在于将节奏学视角与源分离技术相结合，构建了包含时序 F‑measure、八维动态剖面和能量路由精度的“分析师指标”体系；系统性检验了输入长度对攻击形状的影响，并揭示 SDR 与动态/攻击失真不一致，提示仅凭 SDR 评估不足以满足节奏研究需求。

**🔧 技术方法**

采用 BSS‑Eval (SDR/SIR/SAR/ISR, SI‑SDR) 与自定义时序/动态/标签度量，利用 RMS 脉冲特征、攻击斜率、能量路由等特征对分离结果进行量化；对比模型间的统计相关性、Spearman 相关与配对 Wilcoxon 检验。

**📊 数据集**

使用 50 条全轨道、44.1 kHz 双声道的 MUSDB18‑HQ 测试集，确保每个模型的估计与真实分离基准可对照验证。

**📈 对比分析**

结果表明：在鼓轨道上，SI‑SDR 与起始 F‑measure 的 Spearman ρ≈0.62，说明 SDR 对时序准确性具有预测力；但与攻击斜率、动态变异等失真指标的相关性弱（ρ≤‑0.29），并出现模型排名倒置（如 SDR 领头的 BS‑Roformer 在攻击失真上比 SCNet‑XL 差一倍）。输入长度改变对攻击斜率的漂移显著，凸显上下文对模型输出的影响。

**⚠️ 局限性**

主要限制包括：仅对鼓轨道进行深入分析，未覆盖其他音色；使用单一固定时间段进行输入长度实验，缺乏音乐内容自适应；未进行听感验证，仅量化信号失真；并且所有模型均为单一检查点，未体现训练变异带来的差异。

---

## 116. What Does Multi-Harness RL Learn? Credit Assignment and Portability in Coding Agents

**arXiv ID:** 2609.04518 | [PDF](https://arxiv.org/pdf/2609.04518v1)

**作者:** Chenqian Le `[一作]` (New York University), Xupeng Chen `[通讯]` (Dimension Gate)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在同一冻结轨迹上对两种不同的信用分配边界（Cross 与 Within）进行对照实验，研究多引擎强化学习中信用分配对策略学习和可迁移性的影响。

**💡 创新点**

创新点在于首次系统地隔离并比较跨引擎与单引擎信用分配对优势标准化的影响，发现跨引擎信用仅携带引擎身份信息而不提升对未见引擎的性能，并给出评估报告的规范建议。

**🔧 技术方法**

采用了基于GRPO的组相对优势计算、政策梯度优化、冻结轨迹重放、sealed oracle评估以及优势标准化等技术。

**📊 数据集**

使用了SWE-Gym与SWE-bench Verified数据集，涵盖四个生产引擎（Aider、OpenHands、Qwen Code、SWE-agent）及500个任务样本。

**📈 对比分析**

通过在四个源引擎和一个未见引擎上评估多个检查点的解题率和行为分布，比较了Cross与Within两种信用分配，结果显示引擎差异是性能变化的主导因素，Cross信用虽能识别引擎身份但未显著提升未见引擎的解题率。

**⚠️ 局限性**

局限性包括仅评估单一模型族与固定预算，低解题率、样本量有限、只用单个检查点进行大部分比较、未涵盖多语言与更广泛的任务范畴，以及对不同seed的统计覆盖不足。

---

## 117. The Prompt Triangle: A Registered Report on Prompts as Hybrid Artifacts

**arXiv ID:** 2609.04209 | [PDF](https://arxiv.org/pdf/2609.04209v1)

**作者:** Shalini Chakraborty `[一作]` (University of Bayreuth), Jan-Philipp Steghöfer `[通讯]` (XITASO GmbH IT and Software Solutions)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了 Prompt Triangle 模型，探究 AI 辅助编程提示在不同阶段的演化与需求工程活动的匹配对代码质量的影响，使用对照实验、社区上传与公开挖掘三源数据进行定量与定性分析。

**💡 创新点**

创新点包括：①将提示视为“混合工件”，构建三维 Prompt Triangle；②提出四个可检验假设，关注提示演化、开发者特征、RE 活动匹配与时间顺序；③设计三源混合数据收集与多方法验证；④采用 Dirichlet 及层级回归等统计模型对提示组成与代码质量进行严谨推断。

**🔧 技术方法**

技术方法包括：对话转录与 JSON 结构化；双人编码与 Cohen’s κ 可靠性检验；Prompt Triangle 组件比例与 RE 活动标签的手工/半自动注释；Dirichlet 混合效应回归、层级回归、重复测量 ANOVA、双编码对齐评分；代码质量评估结合单元测试通过率与静态分析指标。

**📊 数据集**

数据集来源三大类：①30 名受试者的 120 轮 AI 辅助编码实验；②≥150 条社区自愿上传的 IDE 对话；③≥200 条公开挖掘的对话；共计约 3858 条对话、54,357 个回合、22,506 条代码片段，覆盖 Python、JavaScript、Java 等主流语言。

**📈 对比分析**

比较方法：先用实验数据检验四个假设，随后用社区上传数据做共价复制，公开挖掘数据做探索性模式描述；在实验中用线性混合模型与 Dirichlet 回归评估提示比例变化，层级回归检验开发者特征与质量对齐；结果显示：提示随轮次从功能/质量向具体实现倾斜，特定解决方案比例随经验提升而显著；RE 活动对齐度与代码质量正相关，时间顺序匹配进一步提升效果；但未给出数值性能指标，主要以统计显著性与效应量呈现。

**⚠️ 局限性**

局限性包括：①提示编码主观性与可能的注释偏差；②受试者样本量仅 30 人，可能存在自选偏差；③实验任务与时间限制限制生态效度；④不同 AI 助手版本与语言差异带来的混杂；⑤仅评估功能正确性与静态质量，未涵盖性能与安全等维度；⑥公开挖掘数据偏向公开案例，可能缺乏代表性。

---

## 118. BER-PEF: Unified Human Mobility Predictability Evaluation via Bayes Error Rate Estimation

**arXiv ID:** 2609.04292 | [PDF](https://arxiv.org/pdf/2609.04292v1)

**作者:** En Xu `[一作]` (Tsinghua University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于Bayes错误率的统一可验证人类移动性可预测性评估框架BER-PEF。

**💡 创新点**

创新点在于将BER估计映射为可预测性估计，并通过受控扰动曲线、共享参考区间和面积差异度量实现无观测真实可预测性情况下的统一比较。

**🔧 技术方法**

使用Bayes错误率估计器（1NN、kNN、kNN-LOO等）、共享表示层（UniMob）、受控扰动和面积评估技术。

**📊 数据集**

在Foursquare NYC、Foursquare TKY、GeoLife和T-Drive四个城市轨迹数据集上验证。

**📈 对比分析**

通过比较估计曲线与参考区间的下/上侧偏差面积来评估估计器可靠性；BER估计器在多种输入下实现了更低的参考差距，并能跟踪扰动下的经验预测性能。

**⚠️ 局限性**

局限性包括对扰动设计的依赖、对高维表示的估计器性能敏感，以及在极端稀疏或非平稳数据上的适用性待进一步验证。

---

## 119. LentEx: Generalizable Latent Entity Extraction via Synthetic Data and Instruction-Tuned LLMs

**arXiv ID:** 2609.04511 | [PDF](https://arxiv.org/pdf/2609.04511v1)

**作者:** Umesh Bodhwani `[一作]` (Amazon), Ayush Goyal `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了LentEx框架，利用模板化合成数据和指令微调方法，显著提升小型LLM在隐含实体提取（LEE）任务中的性能与跨域泛化能力。

**💡 创新点**

首创将LLM与模板化合成数据、LoRA低秩适配及指令微调相结合用于隐含实体提取，解决了标签稀缺问题，并实现了在多域上的强大泛化。

**🔧 技术方法**

技术包括：模板化合成数据生成、Claude‑3‑Sonnet驱动的多轮生成、Mistral‑7B‑Instruct‑v0.2的LoRA微调、检索增强生成（RAG）框架集成、以及基于OpenSearch的动态过滤和重排序。

**📊 数据集**

数据集：合成语料约10,000条；MTEB 11个聚类基准（arXiv、bioRxiv、medRxiv、Reddit、StackExchange、20‑Newsgroups等）；法律检索数据COLIEE；生物医学检索数据BioASQ。

**📈 对比分析**

对比方法包括：Mistral‑7B‑Instruct、Claude‑3‑Haiku、GTE‑Qwen2、bge‑en‑icl、NV‑Embed等嵌入与提示基线。LentEx在聚类任务中平均V‑measure达到59.54，显著高于所有基线；在RAG检索任务中，COLIEE和BioASQ的F1分别为0.732和0.66，接近或超过当前SOTA CAPTAIN，显著提升Precision和Recall。

**⚠️ 局限性**

局限性：聚类时对模糊或多重隐含实体的判定容易出错；检索时过度过滤可能导致Recall下降；对极端专业领域的多实体推断仍需进一步自适应调优。

---

## 120. Abstraction Agent

**arXiv ID:** 2609.04303 | [PDF](https://arxiv.org/pdf/2609.04303v1)

**作者:** Boning Li `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**通讯引用:** 3844 | [OpenAlex ID](https://openalex.org/A5082905458)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种零样本抽象代理，利用大型语言模型从自然语言规则中自动发掘连续战略特征并进行信息抽象，无需任何游戏特定评估器或树遍历；

**💡 创新点**

创新点在于将LLM的结构化推理能力转化为可量化的特征空间，实现无需手工设计评估器的跨游戏信息抽象；

**🔧 技术方法**

技术手段包括多阶段提示（特征发现、打分、相关性筛选、k‑means聚类），配合LLM的校准锚点实现数值化评分；

**📊 数据集**

使用的实验数据集包括HUNL turn endgames、原创游戏ROVER Trials、PLO4、HUNL preflop/flop以及Riichi Mahjong；

**📈 对比分析**

与EHS、潜在意识抽象（PA）等基线对比，实验显示在HUNL中可实现高达62%可利用性降低，在ROVER中零样本特征也优于标量排序，跨游戏表现稳定；

**⚠️ 局限性**

局限性包括对强大LLM的依赖（较弱模型效果不佳）、特征发现与评分对提示敏感、未在极大规模游戏或极其稀缺语言资料的游戏中验证。

---

## 121. Evidence Integration in Large Language Models

**arXiv ID:** 2609.04290 | [PDF](https://arxiv.org/pdf/2609.04290v1)

**作者:** Sebastien Kawada `[一作]` (Massachusetts Institute of Technology), Manolis Kellis `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究大语言模型在获取外部证据后如何调整先前已形成的答案分布，并提出一种基于先验权重与候选人倾斜的分布式理论；在十二个大型模型（Llama、Qwen、Gemma、Ministral）和八个推理领域（算术、词汇问题、线性系统、命题约束、物理、分子生物、量子、遗传学）进行十余百万次实验验证；进一步通过因果干预与机制解释方法（残差流补丁、注意力裁剪、激活引导、J‑lens 与 Logit‑lens）揭示证据如何被候选人、检验信息与来源线索所调节；并提出“受体相对可靠性前沿”来量化证据对不同能力受体的正负影响。

**💡 创新点**

①将证据整合视为受体特定的候选人级别分布变换，分解为先验加权与候选人倾斜；②提出受体相对可靠性前沿，说明同一证据对弱模型有利、对强模型有害；③发现检验结果与最终决策解耦——模型能自行检验并拒绝候选，却仍采纳；④利用因果实验在网络中定位证据流的“接受、促进、转移、整合”四阶段，并证明验证表征与控制的分离；⑤为多源证据聚合与自我批判提供定量准则。

**🔧 技术方法**

分布式贝叶斯模型、先验权重与候选倾斜估计；交叉拟合与匹配设计的回归分析；因果干预技术（残差流补丁、注意力打断、激活引导）；J‑lens 与 Logit‑lens 机制解读；统计显著性与 bootstrap 置信区间；前沿计算与协方差分析。

**📊 数据集**

基于多领域推理数据集：合成算术（Compositional Arithmetic）、词汇推理（Word Problems）、线性方程（Linear Systems）、命题约束（SAT）、物理实验模拟（Physics）、分子序列分析（BIO）、量子系统（QM）以及遗传学（GEN），共计八大领域，并在物理与分子生物的子域做留样检验。

**📈 对比分析**

通过对每个模型、每个域、每个候选的采纳率、正确率和对照实验进行对比，发现：受体先验权重均小于1（0.20–0.65）；在弱模型上外部证据提升正确率（最高+30%），在强模型上降低（最高-30%）；检验准确度高达0.95但对决策影响极小；受体相对可靠性前沿与理论预测吻合（误差≤0.06）。

**⚠️ 局限性**

实验受限于单一候选证据的假设，无法完全覆盖多信息源交互；模型与域的组合效应导致参数估计不稳定；验证与决策分离的结论在极少数模型中不成立；因果干预需要手动定位网络层，难以自动化；数据集仍以人工构造任务为主，缺乏真实世界复杂推理场景。

---

## 122. Motion-Omni: End-to-End Joint Speech and Full-Body Motion for Spoken Dialogue

**arXiv ID:** 2609.04250 | [PDF](https://arxiv.org/pdf/2609.04250v1)

**作者:** Chengqian Ma `[一作]` (Peking University), Yiwen Guo `[通讯]` (Independent Researcher)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Motion-Omni 框架，实现对话式 LLM 能同时生成可同步的语音和全身共语动作；

**💡 创新点**

通过联合训练将语音生成器与动作生成器与 LLM 共适配，使动作直接由产生语音的隐藏状态生成，消除传统音频→动作的第二推理阶段；

**🔧 技术方法**

采用 Whisper 语音编码器、Qwen2.5 LLM、GLM‑4‑Voice 语音单元生成器、LOM VQ‑VAE 动作教师及 TQGF 门控融合机制；

**📊 数据集**

利用 InstructS2S‑200K、Ex‑Instruct、SwDA‑500、以及自建的 422,856 对齐的伪标签语音‑动作对；

**📈 对比分析**

在 SwDA‑500 上与多种 cascade（MambaTalk、GestureLSM、EMAGE、LOM）比较，Motion‑Omni‑Q7 在无教师推理下的动作质量与速度几乎等同（误差 <1%），RTF 0.78，速度最快；

**⚠️ 局限性**

动作受 LOM 码本限制，无法表达码本外的动作；目前仅训练单一实例，缺乏多语言/文化支持；模型为离线生成，未实现实时交互；评估指标仍有改进空间。

---

## 123. What Moves? Localized Motion Representations for Compositional Scene Control

**arXiv ID:** 2609.04383 | [PDF](https://arxiv.org/pdf/2609.04383v1)

**作者:** Frank Fundel `[一作]` (CompVis), Björn Ommer `[通讯]` (CompVis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种可提示的局部运动表示，通过在全视频编码器中使用空间掩码实现实体级运动嵌入，保持全景上下文的同时实现运动本地化。

**💡 创新点**

创新点在于：①将全景编码与区域掩码结合，避免裁剪或后置掩码导致的上下文丢失和运动歧义；②生成的运动嵌入既具备局部精度又保持全局语义可解释性，可用于运动转移、场景组合和局部动作分类；③通过掩码流解码器对运动嵌入进行监督，提升运动的空间一致性和持久性。

**🔧 技术方法**

技术实现：3D Vision Transformer 运动编码器 + DINOv2‑B 帧嵌入；Transformer 结构的掩码流解码器；与预训练视频扩散模型 CogVideoX‑5B 结合，并采用 LoRA 微调；训练采用 AdamW、600k 步、Batch 128。

**📊 数据集**

数据集：训练使用 OpenVid‑1M 及内部采集视频；评估使用 A2D（局部动作分类）、ARID、IARD、Jester、Something‑Something V2、Diving48（全局动作分类）、SemanticMoments、LAOM DCS 等。

**📈 对比分析**

与低级轨迹、全局语义运动转移等基线相比，本文方法在运动转移的区域内保真度高、漏传低、选择性好；在局部动作分类中达 29.9% Top‑1、42.0 F1，显著优于裁剪或全局方法；在全局动作分类上与最优基线相当或略优，检索任务亦保持高性能。

**⚠️ 局限性**

局限性：仍需手动或自动推断查询掩码；对极端遮挡或快速视角变化的鲁棒性有限；对非常细粒度动作区分仍受限于编码器容量；训练成本高，需要大规模视频和显存。

---

## 124. Low-Latency Spell Correction for Japanese Music Search Queries

**arXiv ID:** 2609.04262 | [PDF](https://arxiv.org/pdf/2609.04262v1)

**作者:** Anshul Garg `[一作]` (Amazon), Ujjal Kumar Dutta `[通讯]` (Amazon)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对日本音乐搜索查询构建低延迟多写作脚本的拼写纠错模型

**💡 创新点**

创新点是脚本感知的合成错误生成、混合脚本规范化以及紧凑的 BART seq2seq 体系

**🔧 技术方法**

使用 BART + 字节级 BPE 词表、键盘布局误差模型、语音混淆先验和声调/半声调错误模型

**📊 数据集**

训练数据来源于 2.8M 音乐目录标题、约 3M 真实日志误拼对以及合成的 50‑100 倍误拼样本

**📈 对比分析**

与 SymSpell、Lattice Path Edit Distance、Claude Haiku 等基线相比，EM 41.09% / CER 11.62%，并保持 <4 ms 推理延迟，显著优于传统字典方法和 LLM

**⚠️ 局限性**

主要局限在汉字 IME 错误处理不足、仅基于单查询无上下文信息，以及对极其混合脚本查询的性能仍有提升空间

---

## 125. Achieving Asymptotic Near-Optimality Without $δ$-Similarity

**arXiv ID:** 2609.04464 | [PDF](https://arxiv.org/pdf/2609.04464v1)

**作者:** Michael Moncton `[一作]`, Eric Frew `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

对采样基础运动规划的渐近近似最优性证明进行了改进，剔除对δ相似轨迹存在的假设并证明即使不保留δ相似路径也能获得渐近近似最优性。

**💡 创新点**

提出了“拥挤排斥（crowding out）”现象，并在证明中直接跟踪近似最优节点，而非依赖δ相似路径，完成了无需δ相似轨迹的渐近近似最优性证明。

**🔧 技术方法**

采用覆盖球序列、稀疏树采样与剪枝（SST）算法的理论分析，利用动态清晰度、δ-稳健性概念，构造诱导归纳证明。

**📊 数据集**

通过构造的二维可达性成本地图（含低成本绿区与高成本红区）的实验环境验证理论，未使用公开数据集。

**📈 对比分析**

与传统基于δ相似轨迹保证的SST理论对比，实验展示在拥挤区域中δ相似路径被剪枝，但仍能生成成本在h(c*,δ)内的非δ相似近似最优路径；表现与理论一致。

**⚠️ 局限性**

证明仅适用于稀疏前向传播采样规划，且仍需假设动态清晰度、成本可加等；实际实现需估计合适的δ、T_prop，无法自动适应所有环境。

---

## 126. AVENUE: Audio-Video EditiNg Understanding and Evaluation

**arXiv ID:** 2609.04253 | [PDF](https://arxiv.org/pdf/2609.04253v1)

**作者:** Hayeon Kim `[一作]` (Ulsan National Institute of Science and Technology), Jaejun Yoo `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 5553 | [OpenAlex ID](https://openalex.org/A5089933293)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AVENUE，即一个面向音视频编辑的基准数据集和评估框架，用来系统评估模型在不同编辑类型和模态选择性上的表现。

**💡 创新点**

创新点在于①构建覆盖12种细粒度编辑类型、音、视频和音视频耦合三类模态的多样化基准；②引入基于每个样本的人工验证的“意向编辑”与“保持内容”标注，形成“选择性可控性”指标；③采用多模态大模型作为自动评判者，实现样本特定、模态感知的四维度评估。

**🔧 技术方法**

主要技术包括：多阶段数据过滤与质量评估（ImageBind、CLAP 等）、类别感知指令生成（使用 Gemini-3-flash）、MLLM-as-a-judge（Qwen3-Omni）评估四个指标（Edit Accuracy、Modality Selectivity、Perceptual Quality、AV Consistency）。

**📊 数据集**

使用 VGGSound 作为源数据集，经过过滤得到 1,291 源片段，生成 7,957 条编辑指令。

**📈 对比分析**

与三种编辑范式（joint、sequential、separate）进行对比。结果表明：joint 在保持模态方面表现最佳（最高 Modality Selectivity），但 Edit Accuracy 与 Perceptual Quality 相对较低；separate 与 sequential 在 Edit Accuracy 上更强，但对保持模态的鲁棒性差异显著。总体而言，各范式在不同编辑子类型上表现互补，尚未出现单一方案在所有指标上占优。

**⚠️ 局限性**

限制：目前模型无法在所有编辑类型上同时兼顾高编辑精度与模态保持；评估依赖 MLLM 作为裁判，虽然与人工对齐度较高，但仍存在主观性；数据集覆盖的编辑场景虽大幅提升，但仍未完全覆盖极端或极细粒度的编辑需求。

---

## 127. An iterative rounding $2$-approximation for Feedback Vertex Set via AI-assisted proof of an extreme point property

**arXiv ID:** 2609.04414 | [PDF](https://arxiv.org/pdf/2609.04414v1)

**作者:** Karthekeyan Chandrasekaran `[一作]`, Shubhang Kulkarni `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了强密度多面体（Strong‑Density Polyhedron）和边强密度多面体（Edge‑Strong‑Density Polyhedron）的极点性质，即在任何极点解中至少存在一个坐标≥1/2，从而构造了基于迭代舍入（Iterative Rounding）的 2‑近似算法；同时给出了这两个多面体的多项式时间分离预言机（对于边强密度多面体）和一个紧凑的扩展表述（与已知的定向扩展模型等价）。

**💡 创新点**

创新点包括：
- 首次把极点 ≥1/2 的性质证明应用到具有非 {0,1} 系数的线性规划（而非传统的可拉普拉斯/子模函数情况）；
- 引入“几乎层状（almost‑laminar）”基的结构化证明，结合 AI 工具帮助发现关键计数引理；
- 证明了边强密度多面体的多项式分离预言机，并证明其等价于之前的定向扩展模型，从而得到新的 2‑近似算法；
- 通过极点性质的发现，提供了一条新的 LP 取舍方向，可能用于 Tree‑width 删除和 Subset‑FVS 等更广泛问题。

**🔧 技术方法**

主要技术手段：
- 超模（supermodular）与交叉（uncrossing）技术，构造层状/几乎层状基；
- 通过条件超模性证明 f_x 为超模，进而在交叉时保持紧性；
- 计数引理（sharing loss）与递归证明，得到极点坐标≥1/2；
- AI 工具（Claude, Gemini）用于梳理复杂证明步骤；
- 对比与定向扩展（orientation polyhedron）的等价性，通过 Farkas 定理和完全无环性证明；
- 迭代舍入与极点选取策略实现 2‑近似。

**📊 数据集**

本文为理论研究，未使用实验数据集；所有证明均为形式化的数学推导。

**📈 对比分析**

与已有 2‑近似（基于 local‑ratio / primal‑dual）相比，本文提供了基于 LP 的迭代舍入算法，保持相同的 2 近似比但在理论上更具可解释性和可扩展性；
- 在多项式时间内可求解边强密度多面体（而原始强密度多面体缺乏分离预言机）；
- 通过等价的定向扩展模型，可直接在多项式规模 LP 上进行求解。

**⚠️ 局限性**

局限性：
- 对强密度多面体仍无已知多项式分离预言机，导致该多面体的 LP 仍为指数规模；
- 迭代舍入算法在实践中可能较慢，因需要枚举并求解多次 LP；
- 结果仅适用于无自环无孤立顶点的无向图；
- 对 Tree‑width 删除、Subset‑FVS 等相关问题的进一步改进尚未实现。

---

## 128. A Quantum Variational Approach to Prototypical Recurrent Unit

**arXiv ID:** 2609.04354 | [PDF](https://arxiv.org/pdf/2609.04354v1)

**作者:** Mahyar Sadeghi Garjan `[一作]` (University of Ottawa), Michel Barbeau `[通讯]` (Carleton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出轻量级量子递归单元QPRU，利用变分量子电路实现时间序列预测；

**💡 创新点**

通过仅使用两个量子变分电路并共享结构，将参数量降至QLSTM、QGRU 的一半或三分之一，同时保持甚至提升预测精度；

**🔧 技术方法**

采用角编码、CNOT纠缠层与经典全连接层相结合的变分量子电路（VQC），使用RMSProp优化与adjoint微分训练；

**📊 数据集**

在合成正弦序列和2020‑2024年AAPL股票收盘价数据集上进行实验；

**📈 对比分析**

与经典PRU/GRU/LSTM以及量子QGRU/QLSTM进行对比，QPRU在两组实验中均实现最低或最优的l1/l2/l∞损失，且标准差较低，显示出良好的预测性能；

**⚠️ 局限性**

主要局限在于实验基于模拟器/小规模量子硬件，未验证在实际硬件上的误差与可扩展性，且仅针对单变量时间序列。

---

## 129. Approximating CDTW Distance of Piecewise Algebraic Curves

**arXiv ID:** 2609.04294 | [PDF](https://arxiv.org/pdf/2609.04294v1)

**作者:** Alperen A. Ergür `[一作]` (UT San Antonio), Shamik Khowala `[通讯]` (Harker School)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种用于计算分段代数曲线之间的连续动态时间规整（CDTW）距离的近似算法，旨在克服现有算法在计算精度和复杂度上的限制。

**💡 创新点**

创新点在于提出了一个完全多项式时间近似方案（FPTAS），能够在给定的误差范围内有效计算CDTW距离，特别是针对高阶代数曲线的情况。

**🔧 技术方法**

使用了动态规划和切比雪夫插值等技术，结合几何预处理方案来处理高阶代数曲线的近似计算。

**📊 数据集**

使用了分段代数曲线作为数据集，具体包括具有不同片段数的两条输入曲线，且每个片段由最高为d的多项式定义。

**📈 对比分析**

通过与现有的CDTW和Fréchet距离算法进行比较，展示了所提算法在计算复杂度和精度上的优势，尤其是在处理高阶代数曲线时的有效性。

**⚠️ 局限性**

限制在于算法的复杂度依赖于输入曲线的片段数和多项式的位数，尽管提供了近似，但在某些情况下可能仍然面临计算资源的挑战。

---

## 130. Knowing When Not to Answer: Pseudo-Ensembles for Abstention in Music Audio-Language Models

**arXiv ID:** 2609.04362 | [PDF](https://arxiv.org/pdf/2609.04362v1)

**作者:** Aanya Maheshwari `[一作]` (Jumeirah), Vatsal Raina `[通讯]` (Apta AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在音乐音频‑语言模型的多选问答中，构造伪集成（通过打乱候选答案顺序得到不同的预测分布），实现模型在做出回答前可根据不确定性做拒绝（abstention），并且通过多次前向传播提高整体准确率。

**💡 创新点**

创新点在于：①提出“伪集成”概念，用单个预训练模型在答案保持不变的输入扰动（如答案顺序打乱）生成多样化的预测；②利用该伪集成实现可观测的集成不确定性指标（如期望熵），从而实现低成本的选择性预测；③证明答案顺序打乱既能提升准确率，又能提供比单通道熵更好的错误排序。

**🔧 技术方法**

使用技术包括：音频编码器＋语言模型（TinyMU 229 M参数）、基于第一个生成token的softmax读取答案分布、答案顺序扰动、对多重预测分布求平均以得到期望熵、EE、MI 等不确定性度量；前向传播次数从1增至4或21，以形成伪集成。

**📊 数据集**

使用的数据集是 MuChoMusic（1 187道四选一音乐问答，来自 MusicCaps 与 Song Describer 数据集），并在其上评估 TinyMU 的准确率与不确定性曲线。

**📈 对比分析**

与单通道基线（单次前向、单通道熵）比较，伪集成在四种答案顺序（M=4）下将准确率从 55.7% 提升到 59.2%，使用全部 24 种顺序时可达 60.3%；在不确定性排名上，期望熵的 AUC‑ERC 从 0.293 降低到 0.261，显著优于单通道熵（0.293）和负置信度（0.264）。

**⚠️ 局限性**

局限性包括：仅在 TinyMU 与 MuChoMusic 上实验，未验证大模型或其他问答数据集；伪集成依赖于答案保持不变的扰动，若模型对答案顺序不敏感则效果减弱；标签交换在 TinyMU 上失效，表明方法对模型训练偏好敏感；未对不同扰动强度或多模型集成进行系统调优。

---

## 131. A Constraint-Aware Generative Framework for Synthetic Origin-Destination Demand in Logistics Networks

**arXiv ID:** 2609.04345 | [PDF](https://arxiv.org/pdf/2609.04345v1)

**作者:** Leian Chen `[一作]` `[通讯]`, Leian Chen

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于约束感知的条件生成框架，用于在层级物流网络中合成可满足运营约束的O‑D需求分布。

**💡 创新点**

创新点在于将交通层级偏置融入图注意力网络、在生成器中加入可微约束损失以及通过选择性迁移学习实现网络拓扑变更时的快速自适应。

**🔧 技术方法**

核心技术包括交通偏置的图注意力网络（GAT）、条件变分自编码器（cVAE）、可微约束满足机制和选择性冻结的迁移学习策略。

**📊 数据集**

使用美国中部中间配送网络12个月的真实运营数据，包含数千个设施、3.8万个ZIP目的地和数百万日包裹记录。

**📈 对比分析**

与历史均值、无图VAE、标准GAT‑VAE等基线对比，JS散度提升约16%，约束合规率达到87%，在冷启动场景下仅需10–18轮训练即可保持82–88%完整训练性能。

**⚠️ 局限性**

局限在于仍依赖先验的目的地聚类，约束设计需人工指定，对非层级或多源网络的泛化能力尚未验证。

---

## 132. PerfReasoning: How Well Do LLMs Reason on Hardware Performance?

**arXiv ID:** 2609.04476 | [PDF](https://arxiv.org/pdf/2609.04476v1)

**作者:** Dan Zhao `[一作]` (NVIDIA), Qijing Huang `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套针对大型语言模型（LLM）的基准，用来评估它们在硬件性能推理和生成解析性能模型代码方面的能力，并对多种前沿模型进行系统评测。

**💡 创新点**

创新点在于：①将性能推理与性能模型构造拆分为两项独立任务，清晰区分模型理解与代码实现的难度；②构建了包含 216 条矩阵乘法、批量矩阵乘法和卷积三类工作负载的标准化基准；③在基准中引入了多种 RL 与自我修正技术，评估其对推理与模型构造的影响；④首次公开该基准，推动 LLM 在硬件设计中的可复现评估。

**🔧 技术方法**

使用的大型语言模型包括 Claude、DeepSeek、Gemini、GLM、GPT、GPT‑OSS、MiniMax 等，实验采用的技术主要是：自然语言推理、代码生成与执行、Veriﬁer‑guided RL、无反馈多轮自我修正、性能指标（Q&A 正确率、模型构造通过率、对比排名准确率）等。

**📊 数据集**

数据集为 216 条工作负载/架构/映射配置，覆盖矩阵乘法、批量矩阵乘法和卷积三类算子；每条配置提供工作负载描述、两层内存架构（主存→缓存→MAC）以及三类映射（时序分块、循环顺序、张量保留/绕过）。使用 Timeloop 生成的标签作为性能模型的黄金标准。

**📈 对比分析**

比较方法：对每个模型进行 Q&A 推理准确率、单次通过率（模型生成代码能否在隐藏案例中通过 10⁻⁶ 误差门槛）和映射排名准确率的评测。结果显示：
• 关闭源模型 Q&A 正确率可达 90%+；最强开放模型达 82.4%。
• 模型构造通过率普遍低于 45%，仅 GPT‑5.6 Sol 维持在 84–88%。
• RL 训练后，Qwen3‑4B 的映射推理准确率从 54.3% 提升至 70%。
• 无反馈自我修正效果不稳定，平均通过率仅从 14.0% 提升到 16.7%。

**⚠️ 局限性**

局限性：
① 模型构造结果高度依赖随机性，单次输出往往不可靠；
② 基准仅包含两层内存架构，未覆盖更复杂的多层缓存或网络拓扑；
③ 评测标签来自 Timeloop 的解析模型，缺乏硅芯片的实际验证；
④ 只测试了当下最先进的 LLM，未考虑更大规模或不同结构模型；
⑤ 自我修正策略未结合执行反馈，导致效果不佳。

---

## 133. Statistics of Similarity Graphs in Node-Arrival Streams

**arXiv ID:** 2609.04505 | [PDF](https://arxiv.org/pdf/2609.04505v1)

**作者:** Kaiwen Liu `[一作]` (Indiana University), Qin Zhang `[通讯]` (Indiana University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在节点到达流中对相似图进行统计的算法，提出了针对度数矩、分散指数、度数矩采样和分散采样等问题的常数轮、子线性空间流算法。

**💡 创新点**

创新之处在于首次将相似性感知的频率统计推广到隐式相似图，并给出了对应问题的常数轮、子线性空间算法与匹配的下界，实现了理论上几乎最优的空间复杂度。

**🔧 技术方法**

主要技术包括见证采样、将节点划分为高/低度两组、Chernoff 及 Chebyshev 近似、指数随机变量拒绝采样、Nisan 伪随机生成器以及通信复杂度归约。

**📊 数据集**

论文不使用真实数据集，而是通过构造的数学实例（高维向量与特定相似函数）来证明下界；所有结果均为理论分析。

**📈 对比分析**

通过对比上界与下界，作者证明空间复杂度在对 n 与 ε 的依赖上几乎匹配，算法在常数轮内实现子线性空间，性能优于传统频率统计方法。

**⚠️ 局限性**

局限性包括对 ε 的依赖尚未完全最优；算法在某些情形下需预知流长度或额外轮次；对动态流或更一般图模型的扩展仍是未解决的问题。

---

## 134. Hakken: Predicting future discoveries to fill the gaps in today's knowledge

**arXiv ID:** 2609.04494 | [PDF](https://arxiv.org/pdf/2609.04494v1)

**作者:** Tarek R. Besold `[一作]` (SonyAI), Michael Spranger `[通讯]` (SonyAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了 Hakken，一个融合时序知识图谱和大语言模型（LLM）的预测与解释系统，能够在科学文献中预测并解释未来尚未公开的实体关系，随后在生物医学领域通过历史回溯与实验验证展示其实际价值。

**💡 创新点**

创新点：① 通过时间序列图卷积网络与 Transformer 结合，加入 LLM 语义知识，实现多标签未来关系预测；② 提出了模型无关的解释框架 PHELInE，利用伪重训练快速评估路径解释的充分性和必要性；③ 在生物医学领域完成从模型预测到 wet‑lab 实验验证的完整闭环，确认两条全新基因/酶-基因相互作用。

**🔧 技术方法**

技术手段：时序图神经网络（GraphSAGE + hierarchical Transformer）+ LLM（Mistral‑7B‑Instruct）注入；PU‑安全动态加权焦点 BCE 训练；模型无关的路径解释算法 PHELInE；历史回溯评估、Top‑m 评价、实验验证等。

**📊 数据集**

数据集：基于 PubMed Central Open Access、MEDLINE 以及商业许可数据构建的知识图谱，原始 2.49 亿三元组，清洗后 2.55 万实体 712 万三元组，23 种关系；实验使用衰老相关基因子集生成 154 万条高置信度预测。

**📈 对比分析**

对比方法：随机、ComplEx、KNN、MLP、规则、tNodeEmbed、THiGER、THiGER‑LLM 等；在 2020 年时间切分的多标签关系预测任务中，THiGER‑LLM 在宏 F1、宏召回、均值 nDCG 等指标上均名列前茅，尤其在低频关系上提升显著；历史回溯实验显示预测随时间衰减缓慢，排名稳定；wet‑lab 结果验证两条预测关系得到确认。

**⚠️ 局限性**

局限性：① 仅支持概念–关系类型，缺乏定量/条件表达；② 仅考虑首次出现的事实，未加入多时间/证据元数据；③ 预测不直接给出时间窗口；④ 解释框架依赖代理模型逼近，易受误差影响；⑤ 训练与推理仍需较高算力；⑥ 对未标记负样本的假设仍需改进。

---

## 135. Integrating Crash Report Mining and LLMs for Bug Localization and Repair: An Industrial Report

**arXiv ID:** 2609.04483 | [PDF](https://arxiv.org/pdf/2609.04483v1)

**作者:** Marcos Medeiros `[一作]` (Federal University of Rio Grande do Norte), Rodrigo Bonifacio `[通讯]` (Federal University of Pernambuco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究将崩溃报告挖掘技术（堆栈聚类、可疑文件/方法排名）与大型语言模型（LLM）结合，构建零射提示的崩溃定位与修复流水线，并在两款大型Java企业系统的真实崩溃数据上进行评估。

**💡 创新点**

创新点在于：①系统化地将工业级崩溃报告挖掘结果作为结构化、优先级化的上下文提供给LLM；②探究堆栈信息（完整堆栈 vs 仅异常信息）与排名信息对LLM定位与修复效果的交互影响；③在真实工业环境下，使用零射提示进行一次性生成补丁，评估其在定位与修复上的实用性。

**🔧 技术方法**

技术手段包括：堆栈聚类（四级相似度策略）、可疑文件/方法排名（IAD、IBF、FF多属性评分）、LLM模型（Claude 3.5、Gemini、OpenAI GPT‑4、Mistral、Llama‑3.1）与零射提示设计、REST API 调用、手工评估框架。

**📊 数据集**

数据集为 38 起崩溃 bug，来自两套 2.1 M LOC 的 Java 企业系统（SIGAA 与 SIGRH），每个 bug 包含堆栈、排名信息、前后版本的 top‑3 可疑文件代码及开发者提交的修复。

**📈 对比分析**

评估方法为：①在 8 个样本上进行 5 个模型 × 4 个提示 × 5 次生成的 800 条响应的手工定位/修复准确率；②选出最佳模型（Claude 3.5）在完整 38 个 bug 上再次评估。最终定位准确率 71%（在 38 个 bug 上），修复准确率 52%；与前人（例如 IntDiagSolver、Sobania 等）在类似设置下的 30–40% 级别修复性能相比，具备竞争力。

**⚠️ 局限性**

局限性包括：仅在两款 Java 企业系统上验证，缺乏跨技术/行业的泛化；评估依赖人工判定，缺少自动化测试验证；LLM 生成的补丁往往不完整或含额外注释，难以自动提取；一次性零射提示受限于上下文窗口，未探索多轮交互；模型非确定性导致结果波动。

---

## 136. STEMPix: A Phase-Transition-Material-Based Pixel Sensor for Resolving Edge-Movement Direction

**arXiv ID:** 2609.04435 | [PDF](https://arxiv.org/pdf/2609.04435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 137. Conformity Breaks Conformal Prediction

**arXiv ID:** 2609.04445 | [PDF](https://arxiv.org/pdf/2609.04445v1)

**作者:** Yibo Hu `[一作]` (Illinois Institute of Technology), Hanyu Su `[通讯]` (Illinois Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在多代理LLM系统中注入同质错误的同行回复，展示了社交共识导致的分数机制漂移，使得传统的分层校准的符合式预测在测试时失效。

**💡 创新点**

创新点在于提出并量化了“score‑mechanism shift”这一新的内部漂移机制，揭示了隐藏的条件覆盖崩溃，以及清晰校准的act‑vs‑escalate防御在此情形下被突破。

**🔧 技术方法**

采用了拆分式符合式预测（split CP）与最小不合格分类器（LAC）以及自适应预测集（APS）等技术，并通过同义压力实验、答案翻转率与Δp_gt度量来评估模型对压力的敏感度。

**📊 数据集**

实验使用了ARC‑Challenge、TruthfulQA等公开多项选择问答数据集，涵盖了四个主流开源模型（Qwen2.5‑7B、Llama‑3.1‑8B、Mistral‑7B‑v0.3、Gemma‑2‑9B）以及额外的七种任务类型。

**📈 对比分析**

与传统校准方式相比，覆盖率从目标的90%下降至74%（α=0.10），在低置信度子组上更为严重，从87%跌至47%；单一模型在被攻击时“单例执行”错误率可高达71%。

**⚠️ 局限性**

局限性包括：仅在未标记压力条件下校准的模型仍易受攻击；需要在部署前获取压力标签以实现完整修复；缺乏可落地的在线防御机制；并且结果对模型和任务具有一定依赖性。

---

## 138. HarvestBench: Measuring Whether LLM Agents Will Pay to Avoid Killing Animals

**arXiv ID:** 2609.04444 | [PDF](https://arxiv.org/pdf/2609.04444v1)

**作者:** Jasmine Brazilek `[一作]` (Compassion Aligned Machine Learning), Jeremiah Miller `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了名为 HarvestBench 的农场仿真基准，用来评估大型语言模型在追求收割玉米目标时，对动物生命的侧效应做出付费选择的行为。

**💡 创新点**

创新点在于：①为避开动物设定了明确的“价格”，将动物标识为道德主体；②通过可量化的“付费”机制直接测量模型对侧效应的敏感度；③在模型评估中同时引入道德指令（morality briefing）和无道德指令（neutral briefing），并对两种情境下的行为差异进行统计分析。

**🔧 技术方法**

技术手段包括：基于 gridworld 的农场模拟器、LLM harness 交互框架、自动化评分脚本（无需 LLM 评分）、多种 LLM（GPT‑5.6、GPT‑4o‑mini、Gemini、Claude 等）通过 OpenRouter 接口调用，以及对模型推理过程（reasoning tokens）与价格弹性进行计量分析。

**📊 数据集**

数据集为：7,201 次决策的游戏日志，包含 9 种模型、30 种随机种子、不同地图几何结构（k=12）以及障碍（石头、干草、动物）分布信息。

**📈 对比分析**

比较方法：通过计算动物被撞击率（kill rate）、对价格弹性的点估计（elasticity）以及与不同简报（morality vs neutral）下的对比，评估模型的同情心和行为一致性。结果显示：kill rate 范围 0.4%–98.8%，在道德指令下大部分模型的杀伤率显著下降，价格弹性介于 0.09–1.69，模型能力与慈悲度并不呈正相关。

**⚠️ 局限性**

limitations：仅测试了两台拖拉机协同，未覆盖多智能体竞争/合作场景；动物只在地图上可见，未评估模型在无视觉信息时的判断；模型被安全策略拦截导致部分拒绝；地图几何变化有限，缺乏更广泛的环境适应性；最终评估仍基于模拟，真实性和外推性需进一步验证。

---

## 139. ICM-Bench: Person-Level Identity Reasoning in Multimodal Agents with Long-Term Memory

**arXiv ID:** 2609.04438 | [PDF](https://arxiv.org/pdf/2609.04438v1)

**作者:** Shidu Ren `[一作]`, Junxiao Shen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Identity-Centric Memory Benchmark（ICMB），专门评估多模态长时记忆系统在跨时间识别、关联和推理重复出现的人物身份与关系的能力。

**💡 创新点**

创新点在于：①将身份绑定、跨期关系推理和长期身份概况推断三类任务系统化；②使用可配置的生成管线创建高度可追踪、身份一致的合成视频与对应的开放式问答；③提供可验证的支持证据与时间截断机制，使得评测能精准测量身份记忆而非仅事件记忆。

**🔧 技术方法**

技术包括：多模态视频生成与重定向、面部/声纹对齐与身份链路、基于字幕和视觉描述的直接记忆、检索增强生成（RAG）、图结构检索、以及用于评判的语义等价判定器。

**📊 数据集**

数据集为ICMB自身生成的1年合成生活相册，包含839段视频（约141分钟），6名持续出现的成人角色，配备1,217道可追踪的开放式问答。

**📈 对比分析**

在基准上，直接使用Gemini 3.1 Pro的字幕记忆模型达74.0%整体准确率，其中Profile（长期身份概况）仅60.3%；其他开源直接模型、内存增强代理和图检索系统均低于Gemini；ASR-only控制性能更差，表明视觉证据至关重要。

**⚠️ 局限性**

限制在于：使用合成视频可能缺乏真实世界的遮挡、光照变化、背景噪声等挑战；评测为受控诊断，未涵盖自然视频中的复杂性，且对隐私与可追踪性要求较高，未来需扩展至真实世界数据。

---

## 140. Topology-Aware Training and Spatial Diagnostics for Fiber Bundle Segmentation in Tracer Histology

**arXiv ID:** 2609.04454 | [PDF](https://arxiv.org/pdf/2609.04454v1)

**作者:** Joselyn Romero Avila `[一作]` (Universidad Nacional Mayor de San Marcos), Anastasia Yendiki `[通讯]` (Athinoula A. Martinos Center for Biomedical Imaging)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

该研究针对大猿类神经示踪组织学图像中的纤维束进行自动分割。

**💡 创新点**

创新点在于首次系统比较了拓扑感知损失（clDice、Betti匹配、Topograph）与传统像素重叠损失，并引入了空间诊断指标Excess_32以揭示过度分割问题。

**🔧 技术方法**

使用了冻结的DINOv3视觉Transformer作为特征提取器，结合可训练的U-Net解码器，并采用多种拓扑损失。

**📊 数据集**

使用了来自五只大猿（M1–M5）的共约150张组织切片，包含稠密、中等、稀疏三类纤维束的人工注释。

**📈 对比分析**

与之前基准方法（Bintsi等、Sundaresan等）比较后，所提出的Topograph损失实现了与BCE–Dice相近的Dice分数、最低的B0错误，并在保留高检测率的同时显著降低了过度分割和FDR。

**⚠️ 局限性**

局限性包括仅在冻结的DINOv3特征下实验，未探索自适应编码器；以及评估仍受限于当前注释质量与组织学图像分辨率。

---

## 141. GRACE: Graph-Grounded Reflective Agent Copilot Engine for Expert-in-the-Loop Knowledge Expansion

**arXiv ID:** 2609.04442 | [PDF](https://arxiv.org/pdf/2609.04442v1)

**作者:** John Seon Keun Yi `[一作]` (Boston University), Dokyun Lee `[通讯]` (Boston University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了GRACE框架，将LLM生成的句子级主张与可信先验知识以加权二分图结构化，并通过图中心性分析将主张划分为Grounded、Refuted、Boundary，随后利用Return on Attention（RoA）策略将边界主张推送给专家验证，形成循环的知识库扩增机制。

**💡 创新点**

① 将外部可信先验作为图中的Anchor节点并赋予权重，使中心性评价由内部一致性转向外部证据；② 引入加权边距与距离成本，利用加权紧密中心性提升对真值的判别；③ 设计RoA目标在有限验证资源下自动挑选高价值且不确定的边界主张；④ 通过专家验证提升的主张被动态加入先验，形成共进化的知识库。

**🔧 技术方法**

加权二分图建模、紧密中心性与介数中心性分析、RoA优先级与成本模型、LLM主张抽取与对齐、检索增强生成（RAG）、专家/LLM验证回路。

**📊 数据集**

QASPER（含长篇论文QA）、QuALITY（多选长文本QA）、内部使用的QASPER论文集用于知识扩增实验，模型包括GPT‑4o‑mini与LLaMA‑3.1‑8B。

**📈 对比分析**

与零样本+部分/完整上下文、传统RAG、GraphRAG进行对比；指标为准确率与可答率。GRACE在QASPER上接近完整上下文零样本水平、显著优于RAG；在QuALITY上与GraphRAG相当但更稳定；随着知识扩增轮次推进，准确率从14%提升至49%，甚至超过完整论文基线；在多文档规模测试中，准确率仅下降≈5%后趋于平稳，显示良好可扩展性。

**⚠️ 局限性**

① 图构建耗时高，主要依赖LLM进行主张分解与先验对齐；② 人类验证实验规模有限，仅完成一次15人实验；③ 假设所有先验同等可信，未考虑来源可信度、时效性或不确定性，对图中心性影响较大。

---

## 142. What Attention Recalls and Recurrence Controls in Hybrid Language Models

**arXiv ID:** 2609.04434 | [PDF](https://arxiv.org/pdf/2609.04434v1)

**作者:** Kirill Afendulev `[一作]` (Artificial Intelligence Research Institute), Anton Korznikov `[通讯]` (HSE University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了混合注意力-循环的语言模型中KV缓存和循环状态在推理过程中的功能差异

**💡 创新点**

发现KV缓存负责可寻址的检索，循环状态负责生成的语言、人物风格和语义场景的控制，二者功能互补

**🔧 技术方法**

采用了split‑prefill和state‑swap两种缓存层面干预手段，对Qwen3.5-4B（Gated DeltaNet）和Falcon‑H1-3B‑Instruct（Mamba）进行实验

**📊 数据集**

使用四个小型合成诊断任务（KV检索、列表索引、语言跟随、人物匹配）以及四个因果案例研究进行评估

**📈 对比分析**

在拆分干预下，KV检索任务在KV‑only保持64–98%准确率，循环状态则降至0；语言与人物任务在rec‑only保持70–80%准确率，KV‑only仅约1%；两模型均显示相同的功能分离模式

**⚠️ 局限性**

局限性包括：仅探究推理时的功能贡献，未揭示各通道的静态编码；干预可能导致通道失衡；未来需对各通道的静态信息进行探针分析

---

## 143. A Repeated-Measurement Study for Cultural Analytics of English Song Lyrics Using Five Large Language Models

**arXiv ID:** 2609.04428 | [PDF](https://arxiv.org/pdf/2609.04428v1)

**作者:** E. Cho Smith `[一作]` (Purdue University), Dawn Laux `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了五种前沿LLM在对英文歌曲歌词进行自尊、自控、归属感、认同感等四个社会构念的零样本注释的可靠性。

**💡 创新点**

在大规模文化文本中系统检验LLM注释的重复测量可靠性、跨模型一致性及其在监督分类中的可迁移性。

**🔧 技术方法**

采用零样本提示的LLM（GPT‑4o‑mini、o3‑mini、Claude‑3.7‑Sonnet、DeepSeek‑R1、Gemini‑2.0 Flash）配合Fleiss κ、Cohen κ以及BigBird‑RoBERTa 进行监督学习。

**📊 数据集**

使用Music4All中约69,000条经过长度与语言过滤的英语歌词子集。

**📈 对比分析**

通过三轮重复注释评估Fleiss κ、跨模型Fleiss κ，结果表明自尊最稳定；监督分类在16分类上macro‑F1≈0.66，单构念二分类macro‑F1在0.68–0.73范围。

**⚠️ 局限性**

未固定温度/种子导致测量波动；外部验证集非独立心理真值；仅评估可靠性而非构念有效性。

---

## 144. Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs

**arXiv ID:** 2609.04526 | [PDF](https://arxiv.org/pdf/2609.04526v1)

**作者:** Tung-Ling Li `[一作]` (Crusoe.ai), Janaki Ram Gotei `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在 native 4‑bit 微尺度 LLM 上进行 LoRA 适配器合并的方法，利用低秩标度调整并冻结 E2M1 代码，实现了合并后模型的代码不变性和比传统合并更精确的权重恢复。

**💡 创新点**

创新点在于引入量化感知标度训练（QAST）与低秩标度适配器，将适配器的变更限制在每块标度上，从而保证合并后代码平面不被重新量化，避免了传统合并导致的精度崩溃，同时获得了存储、切换、回滚等生命周期优势。

**🔧 技术方法**

技术包括：低秩标度适配、量化感知 STE（QAST）、块级标度冻结、基于 E4M3/E8M0 量化网格的训练、Naïve re‑quantized merge 对比、Merge‑aware 量化感知 LoRA、vLLM、DeepSpeed 等推理框架。

**📊 数据集**

使用了 Llama‑8B、Qwen‑30B MoE、GPT‑OSS‑120B MoE、DeepSeek‑V4‑Flash MoE 四个模型，分别在 Banking77、AGNews、CLINC150、Spider 四个任务上评估，辅以 MBPP 生成任务验证。

**📈 对比分析**

与 Naïve 合并和 Merge‑aware LoRA 进行对比，实验显示两种 merge‑aware 方法均在 0.2 分以内保持准确性，但 Scale‑QLoRA 在合并后保持权重完全一致；在存储上实现约 3× 节省，切换速度约 125× 提升，且可精确回滚和统一代码平面审计，整体性能稳定。

**⚠️ 局限性**

局限性包括：需要训练时的量化网格与部署网格完全一致；实验仅覆盖有限模型与任务，未在更大规模或生成任务上验证；代码不变性带来的优势仅在特定生命周期事件（如工具导出、格式转换）显现，对其他情况影响不大；对量化规则的敏感性仍需进一步探究。

---

## 145. STyMo: Fast and Controllable Few-Shot Motion Style Transfer

**arXiv ID:** 2609.04500 | [PDF](https://arxiv.org/pdf/2609.04500v1)

**作者:** Jose Luis Ponton `[一作]` (Reality Labs), Petr Kadlecek `[通讯]` (Reality Labs)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研发一种只需几秒配对运动数据即可学习并实时应用的快速可控运动风格迁移方法。

**💡 创新点**

将运动风格拆分为时间不变的静态姿态和时变的动态运动两部分，提供可解释且可调控的控制；采用极短训练（1–2 min）实现几秒级样例学习；引入可塑性门控避免对外域运动产生伪影；支持快速迭代创作。

**🔧 技术方法**

基于Transformer Encoder‑Decoder结构、静态/动态MLP分类器、门控MLP、2轴旋转表示、时序自回归、负样本挖掘、数据增强和接触感知后处理等技术。

**📊 数据集**

主要使用MOCHA公开数据集，利用自动相位提取和DTW对齐，配合内部数据进行跨数据集验证。

**📈 对比分析**

与VAE‑GME、GANimator、SinMDM、MoST、MoMo等方法在多种指标（多样性、内容保持、脚滑、jerk）上进行对比，STyMo在训练时间（1–2 min）和风格保真度、内容保持等方面均优于基线；在人类评测中获显著高优先选择。

**⚠️ 局限性**

仅依赖极短样例，难以在高度外域运动上泛化；当前方法为离线后处理，实时性需改进；缺乏对风格与内容本质区别的理论支持；需结合更大规模风格先验以提升泛化能力。

---

## 146. Uncertainty Signals for Network Intent Translation: Risk Ranking and Ambiguity Localization

**arXiv ID:** 2609.04486 | [PDF](https://arxiv.org/pdf/2609.04486v1)

**作者:** Ala' A. Alsamarneh `[一作]` (Khalifa University), Omar Alhussein `[通讯]` (Khalifa University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨如何利用LLM生成网络配置的模型不确定性作为部署前的风险信号，提出采样预测不确定性用于风险排序以及词级熵用于歧义源定位。

**💡 创新点**

首次将采样预测不确定性与词级熵结合用于意图转译的风险评估与歧义定位，证明其在不同上下文和采样预算下有效。

**🔧 技术方法**

使用Llama‑3.1‑8B‑Instruct模型进行LoRA微调，采用采样预测不确定性（字符串一致性熵）和词级熵（BPE子词熵聚合）作为不确定性信号。

**📊 数据集**

在Juniper EX3300意图转译数据集（NIT）上训练，并构造了包含多层歧义的测试集，覆盖六级歧义等级和四种上下文类型。

**📈 对比分析**

通过EM、ECE、AURC、Spearman相关等指标比较，采样不确定性在保持风险排序的同时在模板上下文表现最优，词级熵对参数与描述歧义的区分度高。

**⚠️ 局限性**

主要限制为不确定性估计的校准差（高ECE），仅适用于相对风险排序；词级熵受子词切分影响，且目前仅在Juniper平台验证，缺乏跨供应商和大模型的通用性验证。

---

## 147. Shared circuits predict whether LLMs generalize across formats in arithmetic reasoning

**arXiv ID:** 2609.04463 | [PDF](https://arxiv.org/pdf/2609.04463v1)

**作者:** Andrea Gregor de Varda `[一作]` (Massachusetts Institute of Technology), Evelina Fedorenko `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对13种LLM在数值与三种语言（英语、西班牙语、意大利语）算术问题的内部活动进行归因补丁，定位了解决数值算术时所使用的核心神经单元（numeric circuit），并验证该电路的重叠与加载程度能预测模型在语义格式上的泛化表现。

**💡 创新点**

创新点在于利用内部电路重叠与加载指标，在无需任何标注数据的情况下，预测LLM在不同表面格式间的算术推理性能，并证明该方法能与甚至超过传统监督探测器。

**🔧 技术方法**

主要技术包括归因补丁（attribution patching）来计算单元重要性、Jaccard 重叠度量、点-双变量相关、线性概率模型与熵等统计工具，结合对模型内部激活的直接干预验证因果关系。

**📊 数据集**

使用了2,000个算术问题集合，问题分别以数值、英语、西班牙语、意大利语四种表述方式呈现，并配备了符号翻转版本以支持归因补丁。

**📈 对比分析**

将电路加载度与传统监督探测器（在数值问题上训练的预测模型）和置信度指标（答案对数概率、下一词熵）进行比较，结果显示电路加载度在英语（r≈0.75）和西班牙语（r≈0.58）中能显著预测准确率，且在所有三种语言中解释的方差与或超过监督探测器。

**⚠️ 局限性**

局限性包括缺乏对电路重叠与泛化关系的理论解释、单元集合无法映射到具体算法实现、对低资源语言的预测仍受限于电路激活不充分、且熵等简单信号仍能提供相当或更高的预测力。

---

## 148. Optimizing Credential Blast Radius Through Trust Boundaries and Delegation Under Post-Quantum Authentication Costs

**arXiv ID:** 2609.04566 | [PDF](https://arxiv.org/pdf/2609.04566v1)

**作者:** Pauli Taipale `[一作]`, Harri Lainio `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了联合优化可信边界与凭证派生结构的框架，以在满足延迟预算的前提下最小化系统妥协的影响。

**💡 创新点**

创新点在于构建两层优化模型，证明多域深度受限情形下的NP‑hard性，并给出可解的特殊结构（如直接发放、链式或受限宽度）和共享发行器的精确加分条件。

**🔧 技术方法**

采用图割、树宽动态规划、参数流与多目标线性标量化等组合优化技术，并结合安全概率传播与量化的PQC交叉成本。

**📊 数据集**

使用公开的Jaeger服务调用日志生成交互图，并在五条网络路径上采集TLS握手与加密成本实验数据。

**📈 对比分析**

通过风险感知搜索、延迟优先搜索和流量聚类三种方法对比，实验表明风险感知搜索在多条路径上能显著降低期望受影响权重，并且在5%预算保留后所有设计均能满足边界延迟要求。

**⚠️ 局限性**

模型仅涵盖凭证权限传播，未覆盖多父派发、阈值签名、软件供应链攻击等路径；且需要在每次部署中重新校准与测量以保证延迟和安全性。

---

## 149. La Agente Óptima: Towards Agentic Self-Driving Laboratories

**arXiv ID:** 2609.04564 | [PDF](https://arxiv.org/pdf/2609.04564v1)

**作者:** Marcel Müller `[一作]`, Alán Aspuru-Guzik `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个基于大语言模型和贝叶斯优化（BO）的通用实验自动化框架，能够在计算和实验平台上自动构造、管理、监控和修复BO搜索，并在同一框架内实现LLM直接建议与传统BO的无缝切换；

**💡 创新点**

将LLM推理与持久化、事件驱动的BO状态、子代理架构和多平台接口相结合，形成可扩展、可审计、可动态调整目标与约束的闭环实验流程；

**🔧 技术方法**

使用GPT‑5.4/4.1等大语言模型；贝叶斯优化服务BO‑MCP（BayBE/BoTorch后端）；PySCF执行图；a2a 接口调用分子生成代理；RAISE 与 RoboChem‑Flex 物理实验平台的 typed 接口；子代理架构、事件驱动监控与持久工作空间；

**📊 数据集**

多种化学与材料数据集：有机固态激光光子片段库（42 caps×68 bridges×162 cores）、磷烷基配体库（364）、反向隙发射器库（1512）、Co bisphosphine 库（144）、Xe/Kr MOF 库（2800/420有效）、RAISE 液体配方、RoboChem‑Flex 流式化学反应条件等；

**📈 对比分析**

通过与传统BO（自定义 UI/直接 API）以及人工主导的 BO 进行对比，展示在计算任务中提升了超体积（Hypervolume）和帕累托集合、在 RAISE 任务中将目标接近 ±1°、在 RoboChem‑Flex 任务中将产率从 30% 提升至 58.8% 且实验次数、耗材和成本均低于人工方案，整体表现优异；

**⚠️ 局限性**

对评估器质量与实验平台限制敏感；LLM 推理成本高；失败测量的错误编码会误导 BO；需要人工干预以确认异常与纠正约束；缺乏完整的传感器数据与安全机制，尚未实现完全自主化；多平台接口与配置仍需手动维护。

---

## 150. Rhythms of Work: Multi-Scale Interpretation of Human Behavioral Traces for Workplace Agents

**arXiv ID:** 2609.04556 | [PDF](https://arxiv.org/pdf/2609.04556v1)

**作者:** Lin Ai `[一作]` (Microsoft), Scott Counts `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了针对工作场景人类行为轨迹的多分辨率解释方法，构建了从低级操作到高层节律的四级词汇；

**💡 创新点**

创新点在于将行为解释视为可解码的多尺度问题，保持各级解释可单独检索，而非统一压缩为单一摘要；

**🔧 技术方法**

使用了LLM辅助的操作映射、聚类、N-gram挖掘、SVD降维等技术，并通过线性回归和概率比较验证模型；

**📊 数据集**

数据集为来自一款商业生产力套件的667 M个带时间戳的人类活动事件，覆盖5 0 k用户、100个组织；

**📈 对比分析**

通过在独立的2 000人样本上复现词汇结构、对保留用户进行下一节预测的宏F1提升（相对提升约30%）以及对合成时间线的分辨率消融实验，证明不同分辨率对不同问题最优；

**⚠️ 局限性**

局限包括仅基于单一生产力平台、未评估干预策略、合成时间线实验不代表真实用户行为，以及对高层决策缺乏人类评估。

---

## 151. A Calibrated Reflection Approach for Enhancing Confidence Estimation in LLMs

**arXiv ID:** 2609.04539 | [PDF](https://arxiv.org/pdf/2609.04539v1)

**作者:** Umesh Bodhwani `[一作]` (Amazon), Ayush Goyal `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种结合最大置信度选择、反思式提示和距离感知校准的“Calibrated Reflection”框架，以提升大型语言模型在各类任务中的置信度估计与校准。

**💡 创新点**

创新点在于（1）MCS方法对所有标签统一评估置信度；（2）反思式提示通过评估-反思-总结过程增强推理可靠性；（3）距离感知校准考虑标签的序数关系，按邻近度加权调节置信度。

**🔧 技术方法**

采用的技术包括结构化提示（Chain-of-Thought + Reflection）、最大置信度选择（MCS-R）、距离加权校准公式，以及零样本单次LLM调用。

**📊 数据集**

使用了HelpSteer2（对话多维度评估）、Llama T-REx（事实分类）以及一份自有多轮对话数据集。

**📈 对比分析**

与传统的Verbalized Confidence、Log Probability、Self-Consistency、Top-K Confidence、训练探针等基线对比；在HelpSteer2上MCS-RC在ECE、Brier Score、AUROC、AUPRC等指标均优于基线；在T-REx上反思/辩论提示也能接近训练探针的性能。

**⚠️ 局限性**

局限在于假设标签具有明确定义的序数结构，难以直接推广到名义或层次标签；零样本方式虽然低成本，但在需要更高精度的情境下可能不及微调模型；实验主要聚焦对话和事实分类，未验证在视觉‑语言或多模态任务中的有效性。

---

## 152. Deep Reinforcement Learning for Optimization of STAR-RIS Phase and Energy Splitting Coefficients in OTFS-NOMA Framework

**arXiv ID:** 2609.04536 | [PDF](https://arxiv.org/pdf/2609.04536v1)

**作者:** Rais. J. Gachaba `[一作]` (National Institute of Technology), Anirban Bhowal `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并训练了一种基于最大熵深度强化学习（SAC-BSE）的STAR‑RIS配置策略，实时映射OTFS‑NOMA下的时延-多普勒信道到反射/透射相位及能量分配，以实现下行总速率最大化。

**💡 创新点**

创新点在于：①将STAR‑RIS的相位与能量分配同时建模为单步决策问题；②引入Beta‑Space Explorer子网络专门学习能量分配耦合；③利用最大熵SAC保持对时变信道的鲁棒性，避免传统交替优化的实时重解瓶颈。

**🔧 技术方法**

采用了OTFS调制、NOMA功率分配、STAR‑RIS的能量分割与相位控制、最大熵SAC强化学习、Beta‑Space探索网络以及闭式MRT波束成形。

**📊 数据集**

使用仿真生成的OTFS‑NOMA‑STAR‑RIS信道模型（包含多径、移动用户多普勒扩展、反馈延迟等），未使用公开真实数据集。

**📈 对比分析**

通过与OTFS‑only、NOMA‑only、STAR‑RIS‑only、固定能量分配、模式切换等基线对比，SAC‑BSE在各种功率、RIS尺寸和用户速度下均表现出最高或次高总速率，且在最大128倍速度变化时仅约10%降速，证明了方法的鲁棒性与显著性能提升。

**⚠️ 局限性**

局限性包括：仅考虑固定的NOMA功率分配与两用户/分支设置；未验证在更大规模网络或多用户多分支场景的可扩展性；依赖仿真信道，缺乏实际环境验证；对极端高速移动时的性能进一步下降仍需研究。

---

## 153. Open-Set 3D Scene Graphs for Field Robotics: An Outdoor Case Study

**arXiv ID:** 2609.04607 | [PDF](https://arxiv.org/pdf/2609.04607v1)

**作者:** Chad R. Samuelson `[一作]` (Brigham Young University), Joshua G. Mangelson `[通讯]` (Brigham Young University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文在五个户外机器人数据集上系统评估了基于CLIP的开放集3D场景图（3DSG）在真实环境中的表现，探讨了VLM嵌入的多模态与离群、基于对象+上下文的查询、导航与路径效率、区域理解以及跨遍历的一致性等关键问题。

**💡 创新点**

创新点包括：①提出多模态嵌入与离群检测分析，②设计对象+上下文的分步查询策略以缓解小VLM对复杂句子的处理瓶颈，③引入多会话一致性指标（节点计数比、聚类一致性、语义相似度、地形匹配度、几何距离）评估3DSG的长期稳健性，④证明在多公里尺度下3DSG内存保持<600 MB的可扩展性。

**🔧 技术方法**

使用技术：CLIP + FastSAM + YOLO + LIO‑SAM（构建稀疏点云）、GVD（生成place/region节点）、A*（导航规划）、HDBSCAN（多模态聚类）、余弦相似度与自定义一致性度量。

**📊 数据集**

数据集：River Park、Nunns Park、Marina Part 1、Marina Part 2、Rock Canyon Campground（5个多样化户外环境）以及四个在同一RV Park区域的多次遍历数据。

**📈 对比分析**

方法对比：Object+Context查询比Object Only与Full Sentence在对象检索和导航成功率上高达约70%（SS≈55%/RS≈69%），路径效率平均为0.66；区域查询F1平均≈0.36；VLM嵌入离群率高达30%/点，嵌入多模态存在；一致性指标在0.5–1区间，内存≤600 MB。

**⚠️ 局限性**

局限性：VLM嵌入易出现多模态与离群，未得到充分解决；place‑node图缺乏通行性约束导致路径低效或失效；区域理解高度依赖提示精确度且平均嵌入压缩信息，导致F1偏低；小VLM对否定与复杂句子表现不佳。

---

## 154. Representation Redundancy and Structural Complexity in Finite-Field Inversion

**arXiv ID:** 2609.04583 | [PDF](https://arxiv.org/pdf/2609.04583v1)

**作者:** Zheng Zhang `[一作]` (Towson University), Na Zhang `[通讯]` (Towson University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在有限域 𝔽_2^n 上的逆运算中，不同有序基对代数形式和学习难度的影响，证明了同一 Galois 轨道的有序基会产生相同的坐标逆映射。

**💡 创新点**

提出了基于 Galois 轨道的精确冗余性，证明了每个逆任务有 n 个不同的基表示，并分析了不同表示对代数复杂度的影响。

**🔧 技术方法**

使用了多层感知机（MLP）进行控制实验，分析了不同表示的学习性能。

**📊 数据集**

使用了有限域 𝔽_2^n 的不同有序基作为数据集，进行了 n=3 和 n=4 的实验。

**📈 对比分析**

通过比较三种不同的逆映射形式（参考形式、混合表示形式和完整原始形式），发现学习性能的顺序为：参考形式 > 混合表示形式 > 完整原始形式，且与理论分析一致。

**⚠️ 局限性**

限制在于实验主要集中在 n=3 和 n=4，且未考虑显式的 Galois 对称性，未来的工作可以扩展到更高维度和更复杂的模型。

---

## 155. Optimizer Memory Schedules for Outscaling the Overtraining Axis

**arXiv ID:** 2609.04577 | [PDF](https://arxiv.org/pdf/2609.04577v1)

**作者:** Katie Everett `[一作]` (Massachusetts Institute of Technology), Shikai Qiu `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过在不同模型规模（51M~253M参数）和多种过度训练（OT）因子（1×到256×）下训练语言模型，系统评估了四种优化器（AdamW、ADANA、Muon、SOAP）的性能，并针对每种优化器进一步调整学习率、权重衰减和记忆（记忆窗口/动量）等超参数；同时提出了日志时间权重衰减和动量冷却两种新的调度策略；通过对比验证集最终损失、token乘数和等效OT指数等指标，量化了不同优化器随训练时长的相对优势。

**💡 创新点**

①首次将训练时长（OT轴）作为评估优化器的必需实验维度；②提出了可随训练进度动态扩展的动量调度（ADANA）和日志时间权重衰减；③引入动量冷却规则，在训练末期限制动量增长；④证明了动态记忆能在更长训练周期内实现与DANA理论预期相符的OT指数提升；⑤展示了矩阵预条件优化器在大多数OT范围内保持近乎恒定的token乘数优势。

**🔧 技术方法**

采用了自适应优化器AdamW、基于矩阵预条件的Muon与SOAP、带有DANA动量调度的ADANA；使用日志时间权重衰减和动量冷却的调度；通过Skaling函数拟合训练损失，计算token乘数、等效OT指数；对不同超参数（学习率、记忆窗口、权重衰减系数）进行网格搜索；利用最终验证损失进行比较。

**📊 数据集**

论文未显式列出具体数据集，推测使用标准大规模语言模型训练语料（如Common Crawl等公开文本数据）。

**📈 对比分析**

通过在每个OT点分别搜索学习率和权重衰减，记录最终验证损失；利用token乘数（Baseline token / 优化器 token）和等效OT（Baseline OT / 优化器 OT）进行比较；结果显示：ADANA在日志时间权重衰减+动量冷却配置下，等效OT指数≈1.15，接近DANA理论预测；Muon和SOAP在大多数OT区间保持约1.4–1.7×的token乘数；随着OT增大，ADANA逐步缩小与矩阵预条件优化器的差距并在最高OT处与SOAP竞争。

**⚠️ 局限性**

仅测试了51M、124M、253M参数的模型，批量大小固定为256，未系统探究更大模型或不同体系结构；OT范围虽覆盖1×–256×，但对更极端OT的行为未知；日志时间权重衰减和动量冷却的系数选择未针对所有优化器统一调优；批量大小对ADANA效率的影响仅在单一OT点进行，缺乏完整的批量规模与更新次数相互作用的研究。

---

## 156. MURAL: Multimodal Uncertainty-aware Recommendation via Adaptive edge Learning

**arXiv ID:** 2609.04574 | [PDF](https://arxiv.org/pdf/2609.04574v1)

**作者:** Ahmad Mousavi `[一作]` (American University), Yeganeh Abdollahinejad `[通讯]` (Michigan State University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 MURAL 框架，用自适应边学习和不确定性感知融合解决多模态推荐中的结构僵化和语义脆弱问题，实现动态拓扑发现与噪声鲁棒推理。

**💡 创新点**

创新点包括：① 可微分检索+近似最近邻的自适应边学习（AEL），从静态相似图迁移为可学习的动态拓扑；② 对每个模态建模的方差与注意力权重（UAF），实现按项的不确定性感知融合；③ 通过梯度分离、对比对齐与熵正则，提升优化稳定性与多模态信息质量辨别。

**🔧 技术方法**

采用的技术包括：可微分检索+HNSW近似最近邻、轻量 MLP 边评分与局部 softmax、LightGCN 传播、对比损失（行为-模态、跨模态）、对数方差和注意力融合、梯度分离、对比教师-学生对齐、熵正则和随机刷新索引。

**📊 数据集**

使用了 TikTok、Amazon‑Baby、Amazon‑Sports 三大多模态推荐基准数据集，分别包含文本、图像、音频等多模态特征。

**📈 对比分析**

与 LightGCN、MMGCN、LATTICE、DualGNN、MMSSL、DiffMM、DiffCL、FREEDOM、AlignRec、MMGSL 等多种基线在 Recall@20 与 NDCG@20 上进行全排名评估；MURAL 在所有数据集上均超越最强基线，提升 Recall@20 约 7%–10% 以上，差异具有统计显著性。

**⚠️ 局限性**

局限性包括：1）仅对物品侧建模不确定性，未针对用户侧模态偏好展开；2）动态图随时间演化的长期稳定性与实时更新机制待进一步研究；3）在极端模态噪声或极度稀疏的情况下鲁棒性仍有限；4）实验规模虽大，但在更大工业环境中的可扩展性与部署成本仍需验证。

---

## 157. Dynamic Adaptation of the LLM Context for Generating Routines with Coupled Semantics

**arXiv ID:** 2609.04570 | [PDF](https://arxiv.org/pdf/2609.04570v1)

**作者:** Gnaneswar Villuri `[一作]` (Stony Brook University), Alex Doboli `[通讯]` (Stony Brook University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种动态上下文自适应框架，通过验证-生成循环在LLM代码生成中利用执行反馈解决运行时耦合的静态绑定问题。

**💡 创新点**

创新点包括：①四类语义依赖分类；②利用知识图谱提供语义约束；③在每轮生成多候选并采用模拟退火选择；④结构化执行反馈近似梯度指导生成。

**🔧 技术方法**

技术：LLM（Qwen2.5‑72B‑Instruct）生成代码；验证代理执行并提取诊断日志；生成代理接收诊断、KG与候选生成；模拟退火（SA）选取下一代；知识图谱抽取与推理。

**📊 数据集**

数据集：8个程序生成任务，包含2个显式耦合（迷宫导航、跨耦合优化）和6个标准基准（圆形打包、函数最小化、TSP、滤波器设计、在线判题、符号回归）。

**📈 对比分析**

与零射击、Reflexion、OpenEvolve对比，在300/600评估中，本方法在7/8个任务上均优于所有迭代基线（p<0.01），在完整1000评估中仅在跨耦合优化任务上取得最高分，其余任务略低于OpenEvolve。

**⚠️ 局限性**

局限：仅单一占优者（缺乏种群多样性）；诊断仅覆盖测试套件，可能漏掉隐藏bug；离散动作空间波动大，结构化反馈受限；无法进行反事实推理。

---

## 158. Mitra-v2 Technical Report

**arXiv ID:** 2609.04540 | [PDF](https://arxiv.org/pdf/2609.04540v1)

**作者:** Yefan Tao `[一作]` (Amazon), Chris Kong `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为 Mitra-v2 的基于 2D Transformer 的表格基础模型，并通过大量合成任务预训练实现了零样本学习。

**💡 创新点**

创新点在于：①扩大并多样化合成任务分布；②引入 Hybrid SCM 生成器，实现不同机制的混合；③使用几何感知的优化器堆栈提升大规模任务的学习效率；④保持模型尺寸不变，仍达成行业领先性能。

**🔧 技术方法**

技术包括 12 层 2D Transformer、分层注意力、1,000 桶分布式回归头、Hybrid SCM 生成器、Muon 优化器、训练时多进程分布式、以及轻量级的特征/类别扩展包装器。

**📊 数据集**

使用合成数据进行预训练，评估基于 TabArena（51 个数据集）和 TALENT（约 300 个数据集）的真实世界分类和回归任务。

**📈 对比分析**

与 TabFM、EXAONE Tabular、TabPFN-3 等现有模型对比，Mitra‑v2 在 TabArena 全量评测中达到 1,774.6 Elo，排名与 TabFM、EXAONE 并列，超过 TabPFN-3 约 137 Elo；在回归任务中领先 TabPFN-3 约 187 Elo，且与 TabFM 接近；在 TALENT 上与 TabFM、EXAONE 同等级，且在多类分类上位居榜首。

**⚠️ 局限性**

局限性包括：①模型需进行 fine‑tune 与八折叠，计算成本高于前向通行模型；②预训练数据来源未知，缺乏完整训练过程记录；③在极大样本表格上仍有提升空间，需进一步扩大上下文支持；④评测受限于可变的基准更新，性能区间存在重叠。

---

## 159. An Empirical Analysis of CodeQL False Positives and Query Refinements for Java Vulnerabilities

**arXiv ID:** 2609.04535 | [PDF](https://arxiv.org/pdf/2609.04535v1)

**作者:** Amirali Sajadi `[一作]` (Drexel University), Preetha Chatterjee `[通讯]` (Drexel University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对 CodeQL 在 Java 安全分析中误报的系统研究，构建了基于源代码的误报分类并设计了可复用的查询细化方法。

**💡 创新点**

创新点在于将 500 条人工审计的误报路径转化为源级误报模式，进一步实现这些模式在 CodeQL 查询中的细化与自动化，并探究 LLM 代理在项目特定上下文中对细化进行适配的可行性。

**🔧 技术方法**

主要技术包括 CodeQL taint‑analysis、手工归纳与正则/模式匹配、以及使用 Codex 与 Claude Code 等 LLM 代理进行查询重写和适配。

**📊 数据集**

实验使用 CWE‑Bench‑Java benchmark（213 个 CVE、110 个项目、167 个可构建版本），并从误报率最高的十个查询中随机抽样 500 条路径进行审计。

**📈 对比分析**

通过对比原始查询与细化查询，发现细化后误报下降 15.8%（在全数据集），真例保持率 87.5%；在 LLM 代理实验中，提供细化模板时编译率超过 90%，成功率提升至 56–62%，无模板时仅 28%。

**⚠️ 局限性**

局限性包括：仅针对 Java 与 CodeQL；细化模板高度依赖项目特定上下文，仍需人工或 LLM 适配；实验基于 Benchmark 的 CVE 标注，可能无法覆盖所有真实场景；未验证在其他 SAST 工具或语言上的可迁移性。

---

## 160. Repeat-After-Me: Black-Box Adaptive Visual Prompt Injection

**arXiv ID:** 2609.04533 | [PDF](https://arxiv.org/pdf/2609.04533v1)

**作者:** Sizhe Chen `[一作]` (University of California Berkeley), Arman Zharmagambetov `[通讯]` (Meta)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种面向黑盒前沿视觉语言模型（VLM）的自适应视觉提示注入攻击——Repeat‑After‑Me，能在保持用户正常提示的情况下诱导模型泄露 PII 或发起恶意工具调用。

**💡 创新点**

创新点在于将注入转化为“回复前缀控制”——在图像中渲染攻击者期望的确切输出前缀，并通过 LLM 生成器和评估器实现自适应优化，同时构建可复用的注入库。

**🔧 技术方法**

采用的技术包括：基于 LLM 的攻击与评估迭代（PAIR‑style），RAM（Repeat‑After‑Me）提示渲染，攻击库模式挖掘，以及 JPEG‑压缩、下采样/上采样等图像预处理。

**📊 数据集**

使用的数据集为 DocVQA 验证集（100 张独立图片）进行 PII 泄露与工具调用实验，CIMemories 合成用户档案生成 PII 属性，AgentDojo 与 InjecAgent 作为工具集，OpenClaw 模拟环境进行端到端攻击验证。

**📈 对比分析**

在与 ARE、CoTTA、TransferEns、LangVPI 等基线对比中，Repeat‑After‑Me 在商业 VLM（GPT‑5.5、Claude‑Opus‑4.7、Gemini‑3.1‑Pro）上工具调用 ASR ≥90%，在开源 VLM（Qwen3.6‑27B 等）上超过96%；在 OpenClaw 真实 Discord 场景中，文本+图像混合攻击实现 90%–100% 的成功率。

**⚠️ 局限性**

局限性包括高昂的 API 调用成本、仅在未适应防御的模型上评估、对特定 VLM 部署方式（如工具返回放置）未做充分测试，以及现有防御手段虽能降低成功率但尚未完全阻止攻击。

---

## 161. Too Rare to Learn: Prescribed Cyclone Tracks Degrade a Bay of Bengal Ocean Emulator

**arXiv ID:** 2609.04635 | [PDF](https://arxiv.org/pdf/2609.04635v1)

**作者:** Sumaiya Islam `[一作]` `[通讯]` (University of Dhaka), Sumaiya Islam (University of Dhaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在孟加拉湾地区对海洋预测网络进行研究，比较了是否使用预先给定的飓风轨迹信息作为输入的两种U-Net模型。

**💡 创新点**

发现稀疏的飓风轨迹条件输入会导致模型在未见过的强飓风事件上表现比基准持久性更差，证明条件输入的使用频率而非信息量决定其有效性。

**🔧 技术方法**

使用了U-Net深度卷积网络进行三天历史海面温度和盐度的下一天预测，比较了两种模型：无条件（仅海洋状态）和有条件（加四个飓风轨迹通道）。

**📊 数据集**

使用GLORYS12V1每日重分析数据（海温和盐度）以及IBTrACS最佳轨迹，构造了包含七个通道的输入，其中四个为飓风轨迹相关。

**📈 对比分析**

通过事件不重叠的留出15个飓风（65–150 kt）的测试集，计算相对持久性技能（RMSE比率）并进行Wilcoxon检验；无条件模型在所有实验中均优于持久性，条件模型始终劣于持久性，技能范围完全不重叠。

**⚠️ 局限性**

该结论仅适用于一日领先的轨迹条件，未考虑更长时序或其他大气力学输入；此外模型规模微小，可能无法充分学习复杂的风暴海洋交互。

---

## 162. Does the Selected Object Reach the Reader? Auditing Identity Handoffs in Grounded Language-Model Pipelines

**arXiv ID:** 2609.04579 | [PDF](https://arxiv.org/pdf/2609.04579v1)

**作者:** Siddharth Vohra `[一作]` (Amazon Web Services AI Native), Min Xu `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估在地面化语言模型管道中，从选择阶段到检索阶段再到读者阶段，所选实体（ID）是否在返回的前k条检索结果中被保留。通过在HybridQA 600问答样本上，冻结Claude、GPT和Gemini三大模型的输出，对9种检索/重排序策略进行手工审计，记录每种策略在前5条结果中是否出现选定对象，并将此过程标准化为Returned-Object Profile (ROP) 评估框架；同时验证对象缺失对答案准确率（EM）的影响。

**💡 创新点**

创新点包括：①将选择–检索–读者三阶段拆解并以身份保留为独立评估目标；②设计ROP方案，显式记录期望对象、返回字段、cutoff等，完成端到端的可复现评估；③区分 dataset-trace recall 与 selected-object return，揭示原始问题检索中两种目标的误差差异；④通过删除实验证实缺失选定对象导致EM平均下降约30分，量化身份失效的实际损失。

**🔧 技术方法**

技术手段：使用HybridQA数据集与英语维基百科表格-文本匹配；利用Claude Opus 4.7、GPT‑5.5、Gemini 3.1 Pro三大模型的冻结推理；实现9种检索/重排序规则（Exact key lookup、Exact title equality、BM25（title/body/both）、BGE‑M3、ColBERTv2、Hybrid RRF、Hybrid + reranker）；采用ROP schema及参考评估器完成每条记录的身份检查；用EM与F1评估答案质量。

**📊 数据集**

数据集：HybridQA（15,314张表格、286,270条维基百科链接段落），随机抽取600个问答样本用于实验；删除实验选取64个U=G的高对比度样本。

**📈 对比分析**

比较方法：对每个模型–问题对冻结的检索结果，在cutoff 5下判断是否出现目标对象；Exact key / title始终100%；Body‑only BM25遗漏26.6%；ColBERTv2遗漏1.4%；Hybrid reranker遗漏1.0%；Dataset‑trace recall 与 selected‑object return的差异为106/1,792记录（5.9%）。对象缺失导致EM平均下降28.6–31.0分；Hybrid reranker相较BM25提升约15–20分。

**⚠️ 局限性**

局限性：ROP仅验证已哈希绑定的记录，无法确认声明时间或生成者；实验仅覆盖英语维基百科表格‑文本；未评估选择器在不同样本间的稳定性；缺失对象的语义质量未单独分析；删除实验仅针对预选的64个样本；未探讨多模态或非Wiki数据集场景。

---

## 163. Software Engineering in the Agent Era From Trustworthy Change to Human Agent Software Organizations

**arXiv ID:** 2609.04630 | [PDF](https://arxiv.org/pdf/2609.04630v1)

**作者:** Zhongjie Wang `[一作]` (Harbin Institute of Technology), Mingyi Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了一个针对代理软件工程的理论框架，定义了Trustworthy Change（可信变更）、Human–Agent Cell（人机单元）和Responsibility Topology（责任拓扑）等核心概念，并从可扩展数字执行与责任治理的交叉点阐述了软件变更的完整生命周期。

**💡 创新点**

创新点在于：①将软件变更视为可信变更（TC），并将责任闭合的组织维度独立出来形成责任拓扑；②引入人机单元（HAC）作为执行抽象，区分执行与风险接受的权限；③提出渐进式规范和受限容量的假设，探讨在代理驱动执行下的效率与责任分配的理论边界。

**🔧 技术方法**

主要采用理论建模与抽象概念构造，参考现有的 SE 3.0、SASE、MAGE 等工作；没有具体算法或工具实现，但指出可在现有 CI/CD、Git、CODEOWNERS 等软件工程基础设施上实现该框架的参考架构。

**📊 数据集**

未使用任何数据集；论文为理论与框架性工作，未开展实验或案例收集。

**📈 对比分析**

无对比实验或性能评估；文章仅提出理论预测与潜在度量指标，未进行定量验证或基准测试。

**⚠️ 局限性**

局限性包括：①缺乏实证验证与案例研究，理论仍处于概念化阶段；②未提供工具实现或性能评估；③在多责任拓扑的实际组织场景中缺乏实测数据；④对不同规模组织的适用性与成本模型仍待进一步研究。

---

## 164. Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models

**arXiv ID:** 2609.04575 | [PDF](https://arxiv.org/pdf/2609.04575v1)

**作者:** Xing Chen `[一作]`, Hengshuai Yao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种训练后无额外计算开销的稀疏Mixture-of-Experts（MoE）压缩方法，通过在推理时分离路由器的激活数与归一化参照集，显著降低了激活专家数却几乎不损失性能

**💡 创新点**

创新点在于将归一化参照集从激活专家数解耦（k_1激活专家，k_2作为归一化参照集），避免了因k变动导致的增益失衡，从而让半量激活专家几乎不失分数

**🔧 技术方法**

主要技术包括基于软最大路由的top-k选取、重定义归一化分母、对比实验（MMLU、GSM8K、C-Eval、perplexity）以及统计分析（Jensen–Shannon、Gini、熵等）

**📊 数据集**

使用了WikiText-103、CodeParrot（Python）进行无监督评估，MMLU、GSM8K、C-Eval等标准下游任务评测

**📈 对比分析**

比较方法采用配对McNemar检验和百分比差异，结果显示在两大模型（35B与397B）上：标准归一化下k=4/8的精度下降≈4–5点；但采用k_2=16或k_2=10时，MMLU下降仅≈0.3–0.6点，且保持低截断率；perplexity在k_2=k处最优，MMLU在k_2=2k处最优

**⚠️ 局限性**

局限性包括：仅测试两款同系列模型；对不同规模或粗粒度MoE的泛化未验证；仅覆盖英语和Python文本，长文本推理未评估；k_2取值为粗网格，可能有更优细化方案

---

## 165. Reducing Hallucinated Transcripts in Whisper via Hallucination Space Projection

**arXiv ID:** 2609.04561 | [PDF](https://arxiv.org/pdf/2609.04561v1)

**作者:** Maryam Abbasihafshejani `[一作]` (University of Texas at San Antonio), Murtuza Jadliwala `[通讯]` (University of Texas at San Antonio)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种训练‑free 的 Whisper 解码器激活投影方法，通过在推理阶段将隐藏状态投射到低秩子空间来抑制非语音输入产生的幻觉文本。

**💡 创新点**

创新点在于只需一次离线子空间估计，随后在推理时无参数更新即可调节 Whisper 的幻觉率，并通过可控门控策略平衡幻觉抑制与误拒率。

**🔧 技术方法**

技术核心是利用非语音校准数据计算解码器隐藏状态差异的奇异值分解，得到低秩投影基，随后在推理时按门控阈值修正隐藏状态并更新无语音概率。

**📊 数据集**

实验数据集包括 ESC‑50、UrbanSound8K、FSD50K（非语音）和 LibriSpeech（语音），用于评估幻觉率、词错误率和误拒率。

**📈 对比分析**

与 Whisper 原始模型、全量投影、门控投影以及外部 VAD、后处理与阈值调节等基线相比，门控投影在保持低幻觉率（HR ≈ 4 %）的同时，将语音词错误率提升至 6 % 左右，误拒率仅约 2–3 %。

**⚠️ 局限性**

局限性包括：对非语音幻觉的专注导致对其他幻觉情形（长音频、多语言、噪声干扰）覆盖不足；投影子空间依赖校准数据，跨域迁移需进一步验证；并且仍存在一定的误拒率，需在实际应用中做后续确认。

---

## 166. IPGeoAI: Transformer-Based Geolocation with LLM Semantic Fusion

**arXiv ID:** 2609.04559 | [PDF](https://arxiv.org/pdf/2609.04559v1)

**作者:** Avinash Kadimisetty `[一作]` (Meta), Xiaolu Xiong `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文提出了一种基于 Transformer 的 IP 地理定位模型 IPGeoAI，利用 IP 地址的层级序列特征与 LLM 提取的语义上下文进行融合，以实现城市级精度的定位。

**💡 创新点**

创新点在于将 IP 视为层级序列通过 Transformer 编码，并引入零射击 LLM 提取的自治系统语义特征，结合多头跨模态注意力实现动态特征融合，从而突破传统数据库和测量方法在 IPv6 与移动网络上的盲区。

**🔧 技术方法**

主要技术包括 Transformer Encoder、Zero-Shot LLM 语义特征提取、Embedding、Multi-Head Cross-Attention Fusion、标签平滑的交叉熵损失以及基于第三方基准的层级约束推理。

**📊 数据集**

使用的数据集为 Meta 自建的 200,000 城市级地理信息、基于 7 天滑动窗口处理的 IP‑GPS 真实标签、公开的 WHOIS/ASN 信息以及通过 LLM 生成的语义特征，覆盖 IPv4 与 IPv6 流量。

**📈 对比分析**

与传统基于数据库的第三方供应商及多种深度学习基线进行对比，IPGeoAI 在城市级准确率上提升了 6%（从 30% 提升至 36%），在 IPv6 上表现尤为突出；在线 A/B 测试显示对一阶下游指标提升 0.35%。

**⚠️ 局限性**

主要局限包括：模型仍需每日批处理更新，缺乏即时推理；对完整地理层级（从国家到邮编或更细颗粒度）的预测尚未实现；以及在极罕见或全新 IPv6 子网的泛化仍面临挑战。

---

## 167. Matched Starts, Divergent Objects: How Human-AI Collaboration Forms What It Explains

**arXiv ID:** 2609.04542 | [PDF](https://arxiv.org/pdf/2609.04542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 168. VISTA: Dense Multi-Label Classroom Coding with Vision-Language Models

**arXiv ID:** 2609.04550 | [PDF](https://arxiv.org/pdf/2609.04550v1)

**作者:** Andrew Franck `[一作]` (Occidental), Chris Craney `[通讯]` (Occidental)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将教育领域验证过的COPUS课堂观察协议转化为视频‑语言基准，并实现自动化标注

**💡 创新点**

利用已有的可靠性文献为基准提供噪声底线，展示跨学科验证协议的价值

**🔧 技术方法**

基于MiniCPM‑V‑4.5的密集滑动窗口视觉‑语言模型+轻量MLP头，配合结构化多标签提示与max‑pool聚合

**📊 数据集**

使用13个化学课堂视频，5名评估者共识矩阵构成评测数据集

**📈 对比分析**

与人类评估者共识比较，宏观加权准确率为80.1%（Fine‑tuned），零样本仅74.9%，表现优于零样本但仍低于人类可靠性上限

**⚠️ 局限性**

数据集局限单一学科、单摄像头；罕见代码样本不足导致召回低；未加入音频融合，且无法公开视频

---

## 169. Revisiting MemGuard Overhead: A Reproduction Report

**arXiv ID:** 2609.04547 | [PDF](https://arxiv.org/pdf/2609.04547v1)

**作者:** Weifan Chen `[一作]` (Boston University), Renato Mancuso `[通讯]` (Boston University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文重新评估了MemGuard内存带宽调节器的执行时间开销，并通过对比旧版集中式与新版分布式实现，发现之前报告的1.79×慢速是由于实验配置不当导致的误差；同时对MemPol与MemGuard的性能进行了修正比较，并进一步验证了MemGuard的带宽回收与共享功能在内存密集型工作负载上的优势。

**💡 创新点**

创新点在于：①以完整重现实验配置的方式纠正了MemGuard原有开销报告的误差；②展示了新版分布式实现显著降低的时钟中断开销；③将修正后的开销数据与MemPol进行公平比较，证明MemGuard在多种调节周期下竞争甚至超越MemPol；④首次对MemGuard的带宽回收与共享机制进行量化评估。

**🔧 技术方法**

技术手段包括基于ARM Cortex‑A53的性能计数器（PMU）监测、对内存访问进行预算管理、使用高分辨率本地计时器实现分布式调度、利用IPIs与全局计时器的互操作、以及在ZCU102平台上实现的软硬件协同带宽调节。

**📊 数据集**

实验使用了Xilinx Zynq UltraScale+ ZCU102硬件平台，并基于IsolBench与SD‑VBS工作负载集，其中包含读写密集型基准（如disparity、其他矩阵运算等）以及多核调度情景。

**📈 对比分析**

通过对调度周期（32µs–1ms）下的慢速（C’/C）进行测量，并与MemPol在相同平台与负载下的慢速对比，结果显示新版MemGuard在大多数周期下的慢速在1.05–1.10×之间，显著优于旧版MemGuard（1.79×），并在内存密集型基准上往往与甚至优于MemPol。

**⚠️ 局限性**

局限性包括：实验仅在单一ZCU102平台上进行，未涵盖多样化硬件与更大规模多核系统；仅对单一调度策略（预算回收与共享）进行评估，缺少对其他高级调节模式的比较；并且评测主要集中在单任务或简化共调度场景，未充分反映真实生产环境中的多任务交互与复杂工作负载。

---

## 170. SocioGesture: Real-Time and Adaptive Social Gesture Perception for Human-Robot Interaction

**arXiv ID:** 2609.04545 | [PDF](https://arxiv.org/pdf/2609.04545v1)

**作者:** Wenjin Fu `[一作]` (OpenMind), Jan Liphardt `[通讯]` (OpenMind)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一套实时、可自适应的人机交互社交手势识别系统 SocioGesture，能够在机器人边缘设备上低延迟地识别并回应邀请、禁止、占用等社交手势，并通过不确定性驱动的离线重标记来扩充手势词汇。

**💡 创新点**

创新点包括：① 置信度感知的骨架编码（保留关节点坐标与置信度，支持部分缺失的骨架）；② 轻量化双流体式体手骨架模型（单前向传递实现体态与手部特征交叉注意力融合）；③ 训练阶段的遮挡感知腐蚀（模拟手部/臂部遮挡、时间失效），无需推理时开销；④ 在线低置信度手势被保存，离线利用云视频模型标注后再细调，实现安全且可扩展的词汇增长。

**🔧 技术方法**

核心技术：基于 ST‑GCN++ 的骨架动作识别框架，配合 9‑通道体手特征、跨通道注意力融合；YOLO11n+TensorRT 进行人检测；BoT‑SORT 追踪；RTMW‑l 骨架估计；Jetson Thor 边缘推理；离线标签生成使用 Gemini 2.5 Flash 等云视频理解模型；自适应训练采用置信度、熵、预测一致性等多重损失。

**📊 数据集**

使用了在混合室内‑室外 HRI 场景中收集的 7 类社交手势数据集（邀请、禁止、占用、闲置等），并在部署后通过伪标签扩充至 10 类（加上握手、敬礼、竖起大拇指）。

**📈 对比分析**

与现有骨架识别基线（CTR‑GCN、InfoGCN 等）相比，SocioGesture 仅 1.3 M 参数、2.3 GFLOPs、单前向推理即可获得 94.6% 的 NTU‑RGB+D 60 交叉子测试精度；在自定义 HRI 数据集上 LOSO 精度 96.6%，遮挡下手部/臂部精度分别提升至 84.9% 与 79.5%；机器人端实时推理 25 ms/帧；离线适配后 10 类识别精度提升至 87.9%，原 7 类保持 98.5%；现场部署 97.3% 的手势识别率，行为执行成功率 79.3%。

**⚠️ 局限性**

局限性：仅适用于短时、单步的社交手势；长序列、多步骤交互需要更复杂的时序分段与记忆机制；在极端遮挡导致手/臂信息完全丢失时无法恢复；离线重标记与微调耗时且依赖人工验证，更新速度受限；系统目前只关注手势，缺乏与语言、视线、任务状态等多模态信息的融合。

---

## 171. SCAPES: Semantically Conditioned Autoregressive Prior for Environmental Sounds

**arXiv ID:** 2609.04634 | [PDF](https://arxiv.org/pdf/2609.04634v1)

**作者:** Esteban Gutiérrez `[一作]` (Universitat Pompeu Fabra), Xavier Serra `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了轻量级的SCAPES模型，通过在连续音频码流上使用连续正则化流与流匹配，生成高保真环境音频并实现语义可控。

**💡 创新点**

创新点在于将语义嵌入作为细粒度时序条件，采用连续正则化流而非离散标记化，结合重叠分割和上下文记忆，实现了资源友好且可插值的音频合成。

**🔧 技术方法**

使用EnCodec连续潜在空间、CLAP语义嵌入、Transformer参数化的连续正则化流、流匹配优化、以及重叠加叠拼接技术。

**📊 数据集**

基于未人工编辑的Freesound 34分钟、10类环境音频数据集。

**📈 对比分析**

在单卡RTX4090上训练36M参数仅1小时，VRAM<8GB；与RAVE对比，SCAPES在FAD/KAD指标上更优，实时性约为2×，并展示出语义一致性与长期稳定。

**⚠️ 局限性**

受限于仅10Hz控制率、有限的记忆缓冲，难以处理需要长时结构的语音或复杂乐曲，并且对稀缺或未覆盖的语义空间依赖较强。

---

## 172. Leveraging Imperfect Restoration for Data Availability Attack

**arXiv ID:** 2609.04627 | [PDF](https://arxiv.org/pdf/2609.04627v1)

**作者:** Yi Huang `[一作]` (Nanyang Technological University), Adams Kong `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对数据可用性攻击（DAA），本文首先对 CUDA 进行理论分析，揭示其导致梯度非最优和类别偏置的机制，并在此基础上提出一种基于不完全恢复的 Imperfect Restoration Poisoning (IRP) 攻击方法。

**💡 创新点**

创新点在于：①通过深度学习模型的梯度分析揭示 CUDA 的弱点；②利用不完全恢复（imperfect restoration）构造新的滤波器 P_c = R_c ⋆ A_c，使得攻击在保持高图像质量的同时实现更强的不可学习效果；③在单一防御或多种防御（AT、ISS、Cutout、CutMix、Mixup）下均表现出卓越性能。

**🔧 技术方法**

主要技术包括：深度学习梯度理论分析、卷积滤波器设计（CUDA 与 IRP），自监督学习（SimCLR、SimSiam、MoCoV3、BYOL）与监督学习（ResNet-18/34、VGG-19、DenseNet-121、MobileNet‑V2、ViT-S）的训练，抗御技术（对抗训练、ISS、数据增强）以及图像质量评估（LPIPS、SSIM、MS‑SSIM、CLIP‑IQA、BRISQUE）。

**📊 数据集**

使用的数据集为 CIFAR‑10、CIFAR‑100、STL‑10 以及 ImageNet‑100（100 类子集），在这些数据集上评估 SL、SSL 与防御场景下的攻击效果。

**📈 对比分析**

实验通过与八种基准 DAA（EM、TAP、REM、AP、CP、LSP、OPS、AR、CUDA）以及五种防御方法（AT、ISS、Cutout、CutMix、Mixup）进行对比。IRP 在 SL、SSL 及所有防御下均使 clean 测试准确率降至最低（例如 CIFAR‑10 SL≈10%，SSL≈43%；ImageNet‑100 SL≈1.98%，SSL≈9.3%），在图像质量指标（LPIPS、SSIM、MS‑SSIM）上优于 CUDA。

**⚠️ 局限性**

局限性包括：1）攻击效果在 100% 数据被污染时最佳，部分污染时性能显著下降；2）实验主要集中在有限的网络架构和自监督算法，尚未覆盖更广泛的模型和真实场景；3）理论分析基于特定假设，可能对不同数据分布的适用性有限。

---

## 173. An Evaluation Framework for Generating Multi-View Images of a Person in a Scene

**arXiv ID:** 2609.04603 | [PDF](https://arxiv.org/pdf/2609.04603v1)

**作者:** Mahir Majid `[一作]` (Princeton University), Guillermo Sapiro `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文开发了一种名为Head Scene Rotation Difference（HSRD）的新评估指标，用以区分生成图像中全局相机角度变化与局部头部旋转，并基于此指标对合成的多视角人像数据进行筛选与质量控制。

**💡 创新点**

创新点在于提出了在3D一致性度量中缺失的“主体-背景分离”方案，通过将前景头部姿态与背景相机姿态分离计算差异，定量判定生成模型是否实现了真正的视角变化。

**🔧 技术方法**

实现技术包括：使用6DRepNet对头部进行欧拉角姿态估计；利用SAM3生成前景遮罩并将人像像素置黑；使用VGGT进行背景相机姿态估计；将上述结果结合得到HSRD。

**📊 数据集**

实验数据集主要来源于Qwen-Image-Edit-2512生成的40张室内场景前景图，随后用Qwen的Camera LoRA生成多视角图像；此外使用RealEstate10K与P3M-10k验证背景姿态估计的鲁棒性。

**📈 对比分析**

与GPT Image 2、Nano Banana系列及标准Qwen-2511基线模型对比，Camera LoRA在背景变化（PSNR↓、MSE↑、LPIPS↑）上表现显著更好；使用HSRD过滤后，约42%（169/400）生成图像满足HSRD≤20°且头部转动≥30°的标准，证明了该方法在筛选高质量多视角数据上的有效性。

**⚠️ 局限性**

限制在于目前仅适用于单一主要人像的场景，无法处理多主体或背景中有人物的情况；此外对“假零”情况（完全不改变相机或相反方向转动）仍需进一步完善。

---

## 174. Dual-Part Multi-Lateral Branched Network for Multi-Class Segmentation in Cardiovascular Catheterization Angiograms

**arXiv ID:** 2609.04590 | [PDF](https://arxiv.org/pdf/2609.04590v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 175. NavArena: Automated Construction of Goal-Oriented Navigation Benchmarks from 3D Gaussian Splatting Reconstructions

**arXiv ID:** 2609.04602 | [PDF](https://arxiv.org/pdf/2609.04602v1)

**作者:** Junhui Wang `[一作]`, Chao Gao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

NavArena通过在固定的3D Gaussian Splatting重建上构建可用于闭环导航评估的自动化基准，生成可达目标、碰撞约束和语义任务的导航场景。

**💡 创新点**

其创新在于利用固定的3DGS模型直接推断占据成本图和语义目标候选，无需网格重建或人工标注，实现大规模无监督的导航基准构建。

**🔧 技术方法**

结合3D Gaussian voxel化、占据成本图推断、多视角SAM语义分割、DBSCAN聚类、AlphaShape约束等技术，实现渲染、碰撞检测与目标生成。

**📊 数据集**

对超过2000个来自InteriorGS、SceneSplat++以及50个真实捕获的3DGS重建进行处理，生成2200万条专家轨迹。

**📈 对比分析**

与基于网格的FCL对比，网格查询速度提升约10×且误检率低；与基线语义中心提取方法相比，valid‑center率提升至92%，并在多任务闭环评估中实现多种成功率、碰撞率和路径效率指标，展示了明显的性能提升。

**⚠️ 局限性**

目前仅适用于静态SE(2)场景，对动态环境、复杂运动模型或低质量重建支持不足，且占据和语义层仍易受重建与分割误差影响。

---

## 176. PetQA: Benchmarking Veterinary Knowledge and Clinical Reasoning

**arXiv ID:** 2609.04598 | [PDF](https://arxiv.org/pdf/2609.04598v1)

**作者:** Taegyun Kim `[一作]` (Soongsil University), Kunwoo Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建并公开了首个韩语长篇问答基准PetQA，用于评估大语言模型和视觉语言模型在犬猫兽医临床知识与推理能力。

**💡 创新点**

创新点在于构建了包含10,076文本与8,751多模态问答、专家答案、临床条件标注的长文基准，并提供多语言翻译。

**🔧 技术方法**

采用GPT-4o-mini等LLM进行数据清洗和预处理，使用ROUGE、BERTScore以及LLM-as-a-judge的事实性和有用性评价；同时实验RAG与SFT等适配策略。

**📊 数据集**

使用来自Naver Knowledge iN的犬猫问答数据，并在公开的PetQA数据集上进行评估。

**📈 对比分析**

对18种模型（闭源LVLM、开源LVLM、开源LLM）进行零射、RAG、SFT三种设置的基准，结果显示闭源模型在事实性与有用性上领先，所有模型在多模态问答上表现下降，RAG/SFT提升不稳定。

**⚠️ 局限性**

主要局限包括仅涵盖犬猫两种宠物、对专家答案的可信度依赖、以及评测指标对生成内容的客观性限制。

---

## 177. Hidden In Plain Gaze: Gaze Representations as Privacy Controls for Utility and Re-identification Risk in XR

**arXiv ID:** 2609.04592 | [PDF](https://arxiv.org/pdf/2609.04592v1)

**作者:** Cory Ilo `[一作]` (Virginia Tech), Doug A. Bowman `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了在智能 XR 系统中，眼动数据不同表示方式（原始时序、空间热图、工程化特征）对任务识别（动作分类）和身份泄露（闭集重识别）的影响，并验证了表示选择本身可作为轻量级隐私控制。

**💡 创新点**

首次系统比较表示层面对隐私-效能权衡的影响，发现工程化特征在保留约85%动作识别精度的同时，将身份识别率降低约10倍；并指出表示层可为隐私设计提供可解释、可审计的杠杆。

**🔧 技术方法**

使用了眼动/头部运动的三种表示，配合统一的 CNN+BiGRU 编码器；采用两阶段 SupCon 预训练 + 任务/重识别头；评估指标包括动作分类的 Top‑1/Top‑5/平衡精度和重识别的 Top‑1/Top‑5/mAP@R。

**📊 数据集**

HoloAssist 真实场景的自视角数据集，包含 206 个身份、10k+ 事件、21 动词/44 名词共 85 个动作类别。

**📈 对比分析**

在匹配模型容量和训练预算的前提下，对三种表示进行同等训练；实验表明：原始眼动获得最高动作识别准确率；工程特征在保持 85‑90% 任务效能的同时，使闭集身份识别准确率从 19% 降至 2%（≈4 倍随机），热图则效果介于两者之间。

**⚠️ 局限性**

局限性包括：仅在单一数据集上验证，热图表示的 2D 网络可能被低估；匹配容量约束可能导致高维表示欠调优；只评估身份识别，未涉及属性推断；并未探讨多模态（RGB、手部）联合或在线实时场景下的表现。

---

## 178. JLIR: A Julia-Native MLIR-Inspired Intermediate Representation with Automatic JACC Kernel Extraction

**arXiv ID:** 2609.04585 | [PDF](https://arxiv.org/pdf/2609.04585v1)

**作者:** Narasinga Rao Miniskar `[一作]` (Oak Ridge National Laboratory), Jeffrey S Vetter `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文实现了一个 Julia 原生的多层级中间表示 JLIR，能将普通 Julia 程序自动转换为 JACC GPU 内核，且不需要手动注解。

**💡 创新点**

创新点在于把 MLIR 的多方言、Pass 管理和结构化 IR 直接用 Julia 语言实现，并通过宏 DSL 轻量化地新增方言与 Pass，实现在 Julia 生态中完成全链路编译。

**🔧 技术方法**

使用的技术包括 Julia 的宏与多 dispatch、SSA IR、JACC 运行时、基于 ForOp 的循环检测、以及 JLIR 自己的 Pass 管理器；实现了从 Julia AST 到 JLIR IR，再到 JACC 或 Julia 代码的自动生成。

**📊 数据集**

实验数据集为四个主流 HPC 任务：矩阵乘法（GEMM）、二维 Jacobi stencil、Black–Scholes 期权定价与 LLaMA‑3 语言模型微核，分别在 NVIDIA A100 GPU 与 CPU 上跑测。

**📈 对比分析**

与手写 Julia+CUDA、cuBLAS 以及 Reactant.jl 的比较显示，JLIR 生成的 JACC GPU 内核在 Black–Scholes 达到 96% 的手写基准，在 Jacobi 取得 3,023 GB/s 的有效带宽，在 GEMM 取得 87% 的手写基准，并在 LLaMA‑3 的 matmul_vec 内核实现 85× 的 GPU 加速；总体编译延迟低于 1.5 ms。

**⚠️ 局限性**

局限性包括：缺乏共享内存调度/寄存器块化的 tiling Pass，导致 GEMM 性能落后于 cuBLAS；循环检测只能识别结构简单的并行/归约循环；类型系统默认 Any 限制了类型驱动优化；在小规模内核上 GPU 启动开销可能超过收益。

---

## 179. Improving Progressive Compression with Adaptive Interpolation and Coefficient Decomposition

**arXiv ID:** 2609.04573 | [PDF](https://arxiv.org/pdf/2609.04573v1)

**作者:** Wenbo Li `[一作]` (Oregon State University), Xin Liang `[通讯]` (Oregon State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个自适应渐进压缩框架，针对误差界限和PSNR两种常见目标，提供更高效的检索与重建。

**💡 创新点**

创新点在于：① 使用两种互补插值方案（按级别与按区域）并根据目标自适应选择；② 提出系数分解(CoefDecom)方法，进一步利用插值后残差的空间相关性；③ 开发自动调优工作流，在线选择最佳插值/编码配置；④ 针对系数分解做了检索映射与最大值排序的优化。

**🔧 技术方法**

采用多级插值（线性、三次、双线性/三维）、自适应系数分解、位平面编码（通用位平面或Negabinary+XOR）、贪婪/动态规划检索算法，以及快速方向插值和位平面编码加速技术。

**📊 数据集**

在五个真实科学数据集上评测：CESM（气候）、Miranda（流体动力学）、SCALE（天气）、S3D（燃烧）、JHTDB（湍流），数据量从几GB到512GB不等。

**📈 对比分析**

与三种最先进的渐进压缩方法（包括基于残差的多层压缩）进行对比。误差界限模式下压缩比提升可达42.3%，PSNR模式下提升可达92.5%；检索效率提高、重建误差满足目标；在512 GB数据传输实验中，整体传输速度最快，提升高达1.26×；在可视化实验中，给定相同比特率下获得最高PSNR和视觉质量。

**⚠️ 局限性**

局限性包括：1）系数分解深度需预先确定，过深会引入错误传播；2）对某些数据集（如高分辨率湍流）检索/重建速度仍相对慢；3）目前仅在CPU上实现，GPU加速仍待探索；4）对极低误差界限（<10⁻⁷）时，性能优势相对减弱。

---

## 180. DART: Depth-as-Target Pretraining for Surgical Vision Foundation Models

**arXiv ID:** 2609.04555 | [PDF](https://arxiv.org/pdf/2609.04555v1)

**作者:** John J. Han `[一作]` (Vanderbilt University), Omid Mohareri `[通讯]` (Intuitive Surgical, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在手术视觉领域，提出一种在 DINOv2 训练框架中加入像素级深度重建目标的 RGB‑D 自监督预训练方法（DART），仅在预训练阶段利用伪深度作为监督目标，微调与推理时仍只使用 RGB。

**💡 创新点**

创新点：① 将深度作为目标而非输入，避免在推理时需要深度传感器；② 仅在 iBOT 的掩码位置添加轻量级深度解码头，实现深度与 DINOv2 的自蒸馏目标协同工作；③ 通过深度重建促进模型学习更具几何一致性的特征，显著提升稠密像素预测与图像级任务表现。

**🔧 技术方法**

技术：基于 Vision Transformer（ViT‑B/16）的 DINOv2 自监督框架，加入 3 层 MLP 深度解码器，使用 L1 损失对掩码 iBOT 位置的深度重建；使用 Depth Anything V2-Large 生成伪深度；在预训练中保留 DINO、iBOT 与 KoLeo 正则化，新增 depth 损失；微调阶段冻结 backbone，训练任务特定线性头。

**📊 数据集**

数据集：使用公开的 LEMON 手术视频数据集（938 小时，3.4M 帧）作为预训练语料；伪深度由 Depth Anything V2-Large 生成；下游评估在五个手术分割基准（EndoVis18、CholecInstanceSeg、SAR‑RARP50、CholecSeg8k、PhaKIR）、一个深度估计基准（SCARED‑C）以及两个图像级基准（Cholec80 阶段识别、CholecT50 组合识别）上进行。

**📈 对比分析**

对比方法：公开的 VFMs（DINOv2、DINOv3、MAE、MultiMAE）和同数据集训练的 LEMON‑FM；同时对同一数据训练的 vanilla DINOv2。实验表明 DART 在所有稠密像素任务上显著提升 mIoU（如 CholecInstanceSeg +ms. 提升 9.2 点）、深度估计 δ1（提升 0.023）以及图像级任务（Cholec80 F1 提升 0.7）。在多项基准上，DART 超越了自然图像预训练模型和同域基线，即使在冻结特征的设置下。

**⚠️ 局限性**

局限性：① 依赖于伪深度估计器，深度噪声会影响学习效果；② 只在手术数据与 ViT‑S/B backbone 上验证，尚未检验更大模型或其他领域；③ 预训练需要 8×H200 GPU，训练成本较高；④ 对深度源的鲁棒性和不同手术场景的泛化能力尚未系统评估。

---

## 181. Continual Field-Adaptive Models (CFAMs) for Post-Deployment Physical AI

**arXiv ID:** 2609.04552 | [PDF](https://arxiv.org/pdf/2609.04552v1)

**作者:** Amarjot Singh `[一作]`, Vince Nakayama `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 Continual Field‑Adaptive Models（CFAM）的架构，能够在实验室仅用少量示范快速构建技能库，并在部署后通过一次性写入的、无梯度的「Competence Capsule」实现持续自适应学习。

**💡 创新点**

创新点包括：①将“慢学习”与“快学习”拆分为四个脑类脑皮层模块（Sensor、Reasoning、Action、Capsule Field），实现零干扰的增量学习；②用几何残差变换（GRT）将已学习的轨迹按物体姿态重投，避免再次学习；③在 Capsule Field 里实现纠正型（correct）与扩展型（extend）两种学习范式，并在固定预算内进行压缩和衰减；④整个增量更新过程完全在设备上完成，符合离线、计算受限的任务场景。

**🔧 技术方法**

技术实现：SHDL（ScatterNet Hybrid Deep Learning）作为 Sensor Cortex 处理多模态感知；定制 VLM 作为 Reasoning Cortex 进行任务分解与结果判定；GRT 作为 Action Cortex 的几何技能模型；Capsule Field 采用稀疏分布式记忆 + 兼容压缩算法；自适应更新规则基于置信度与检索距离的门控，写入仅需一次前向传播和内存插入。

**📊 数据集**

使用的数据集：公司内部 2.6M+ 轨迹的多体素描数据集（覆盖操作台、四足、类人、无人机和越野车）；另外在仿真中使用公开的 SimplerEnv 与 LIBERO 基准数据。

**📈 对比分析**

对比方法：在同一数据集上训练的匹配数据基线 π₀、CogACT、SpatialVLA；以及后续适应基线 LoRA、MemoryVLA、CronusVLA。实验显示：CFAM 在训练阶段仅使用 40% 数据即可达到完整基线的性能；在保留拆分测试中提升 9.7pp；在部署后收集近 OOD 经验时，CFAM 相比适应基线提升 13.9pp；在顺序环境适应实验中，CFAM 近乎零后向迁移（-0.5pp）而 LoRA 则出现 18.2pp 的灾难性遗忘。

**⚠️ 局限性**

局限性：仅能处理已知技能家族的“近端 OOD”情况，无法处理完全开放式新颖任务；对 GRT 的可靠锚点依赖较强；基线与 CFAM 之间的差异部分源于预训练模型差异，未完全分离；物理平台在后续增长后的保留性能尚未在真实机器人上验证；写入与压缩机制在极端长时序或极大场景多样性时的容量限制仍未彻底解决。

---

## 182. From Answers to Interpretations: Rethinking Ambiguity-Induced Aleatoric Uncertainty Estimation in LLMs

**arXiv ID:** 2609.04543 | [PDF](https://arxiv.org/pdf/2609.04543v1)

**作者:** Omer Nahum `[一作]` (Technion), Paolo Favaro `[通讯]` (University of Bern)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种不依赖生成回答而直接利用澄清所产生的解释多样性来估计大语言模型中的歧义性不确定性。

**💡 创新点**

创新点在于理论证明回答信息往往冗余或误导，并且引入了“澄清‑仅”框架，显著降低了与模型知识相关的泄漏，同时在歧义检测任务中实现了更高的 AUROC 与更低的计算成本。

**🔧 技术方法**

使用的技术包括：基于 LLM 的澄清生成、语义聚类提取解释集合、可选的解释概率估计以及基于熵、Gini 或变异比的统计聚合。

**📊 数据集**

实验数据集包括 AmbigQA、Ambiguous VQA 以及 AMBROSIA，涵盖文本、视觉与文本‑SQL 三种歧义场景。

**📈 对比分析**

与传统基于回答的歧义检测方法（如 Clarification Ensembling、Spectral Decomposition）比较，澄清‑仅方法在宏平均 AUROC 上提升约 2.45 分，成本降低 4–26 倍（输出 token）和 2.2–3.5 倍（API 调用），并显著减少与本体不确定性的相关性。

**⚠️ 局限性**

局限性包括：仍需 LLM 生成澄清并进行聚类，可能遗漏或错误归类解释；对潜在的“潜在歧义”场景（不同解释却有相同答案）不易区分；若需完整的本体-歧义分解，仍需回答信息。

---

## 183. Sparse Disapproval Guarantees a Nonempty Hare Core

**arXiv ID:** 2609.04537 | [PDF](https://arxiv.org/pdf/2609.04537v1)

**作者:** Jiarui Fang `[一作]` `[通讯]` (Boston University), Jiarui Fang (Boston University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明在每位选民最多不认同两名候选人的审批选举中，Hare核心一定存在，并给出了确定性的选择规则；

**💡 创新点**

首次通过缺失集的加权顶点‑边覆盖和双层lexicographic优化，在无候选数、席位数或选民类型限制下保证Hare核心非空；

**🔧 技术方法**

采用组合覆盖与平均化证明、符号化不等式推理以及两套基于位掩码的离线验证器；

**📊 数据集**

未使用真实数据集，而是在有限整数多重性的候选集与选民类型上做穷举验证；

**📈 对比分析**

对 14 组 (m,k) 的 100k+ 方案进行完整枚举验证，验证器耗时可控，未给出多项式时间实现；

**⚠️ 局限性**

仅适用于选民不认同至多两名候选人，不能推广到更高阶失认情况，也未解决更一般域下的核心存在性问题。

---

## 184. SiLR: Structure-Preserving Admission and Process Reward for LLM Tool Agents

**arXiv ID:** 2609.04629 | [PDF](https://arxiv.org/pdf/2609.04629v1)

**作者:** Chenyu Zhou `[一作]` (Institute of Science Tokyo), Xu Zhou `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在电网和建筑能源调度等关键基础设施中，提出一种将运行时门控视为搜索操作的新框架，并通过产品序（branch‑level violation state）实现对LLM工具代理的后违规恢复与安全约束；同时将同一几何表征用作过程奖励，训练能够在无门控时实现恢复的策略。

**💡 创新点**

核心创新在于：① 揭示“标量投影陷阱”，证明任何标量门都无法正确判定可恢复路径；② 提出完整的产品序门（support + severity）作为唯一可行的安全决策；③ 将该几何表征直接转化为GRPO过程奖励，使得训练出的策略在无门控时仍能完成恢复。

**🔧 技术方法**

技术手段包括：Shadow‑Execution 验证器（对LLM提出的工具调用进行深度拷贝模拟）、ReAct 循环交互、产品序门（支持包含与 severity 逐点下限检查）、GRPO 强化学习框架、LoRA 微调、以及多模型（Qwen3、Gemma‑3、Llama‑3.1）和多域（Gym‑ANM、CityLearn）实验。

**📊 数据集**

数据集：Gym‑ANM（ANM6‑Easy）及其手工挖掘的 600 个恢复场景，筛选出的 24 个多动作可恢复场景；CityLearn 建筑能源调度数据，用于验证跨域泛化；还包含多模型、双约束族的攻击实验数据。

**📈 对比分析**

比较方法：与终端门（只允许完全恢复）、标量门（不同阈值）以及仅支持门（Grid‑Agent 方式）进行对比。结果显示：在 24 场景、5 步实验中，完整门实现 120/120 恢复；最佳标量门仅 96/120；仅支持门 113/120；终端门 0/120。攻击实验表明，仅完整门能完全阻止幅值重新分配攻击。

**⚠️ 局限性**

局限性：依赖可确定性、可模拟的工具调用；对非确定性或多模态工具支持不足；实验主要集中在电网和建筑能源场景，其他物理域的推广尚待验证；在多约束族下，标量化奖励仍表现不稳定。

---

## 185. CIERA: Cross-Iteration Exponent Reuse for Lossless Allgather in Sharded MoE Training

**arXiv ID:** 2609.04609 | [PDF](https://arxiv.org/pdf/2609.04609v1)

**作者:** Ali Zafar Sadiq `[一作]` (University of Virginia), Masahiro Tanaka `[通讯]` (Anyscale)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现一种针对分片MoE训练的无损Allgather压缩方法CIERA，利用跨迭代指数重用来显著降低通信开销。

**💡 创新点**

创新点包括：①观察到大多数权重指数在迭代间保持不变；②在块级别实现指数重用的无损压缩；③基于块变化率的收益驱动选择压缩，仅压缩收益正的分片；④通过计算-通信流水线将压缩/解压时间与前后计算/通信重叠，几乎不影响批量。

**🔧 技术方法**

使用技术包括块级指数哈希+ANS压缩、收益驱动选择压缩、PyTorch FX图调度实现计算-通信重叠、在DeepSpeed ZeRO-3基础上实现Allgather改造。

**📊 数据集**

实验使用 AG News、OpenWebText 数据集；在多种MoE LLMs（OLMoE‑1B‑7B、DeepSeek‑MoE‑16B、MiniCPM‑MoE‑8×2B、Qwen2‑57B‑A14B、Mixtral‑8×7B、Llama‑4‑Scout‑17B）以及 BF16/FP16 精度下进行。

**📈 对比分析**

与无损基线（DeepSpeed ZeRO‑3、FSDP）和有损基线（ZeRO++）在 4/8/16 GPU 上比较。CIERA 在 16 GPU 上对 OLMoE 实现 3.70× 的迭代时间加速，相对 ZeRO‑3；相对 ZeRO++ 为 3.68×。在 128 GPU 的模拟预测中，CIERA 分别可达 4.28×（ZeRO‑3）和 4.42×（ZeRO++），在大多数模型上实现 1.16–4.28× 的加速且保持位级精度。

**⚠️ 局限性**

局限性：①主要适用于稀疏 MoE 训练，指数稳定性在密集 LLM 上表现差；②超过 16 GPU 以上的结果基于模拟，缺乏大规模多节点验证；③压缩选择器一次性静态，缺少在线动态调整；④在高带宽 NVLink 环境下收益有限。

---

## 186. GNN-Guided Graph Coarsening and Adaptive QUBO Penalties for the Capacitated Vehicle Routing Problem with Time Windows on a Quantum Annealer

**arXiv ID:** 2609.04593 | [PDF](https://arxiv.org/pdf/2609.04593v1)

**作者:** Youssef Kamel Rezk `[一作]` (Alamein International University), Paweł Gora `[通讯]` (Jagiellonian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对 CVRPTW 进行 QUBO 编码，提出自适应惩罚校准与基于 GNN 的无家族参数共聚方法，并在模拟退火与 D‑Wave Advantage2 上进行评估。

**💡 创新点**

创新点包括：① 对容量 slack 的自适应归一化与非绑定约束剔除，显著改善 QUBO 的条件数；② 用 GraphSAGE 学习的合并分数实现统一配置、无家族调参的共聚；③ 通过对系数范围的控制而非单纯缩放，提升原始样本质量。

**🔧 技术方法**

技术手段包括：GraphSAGE 作为 GNN 合并评分器；SimulatedAnnealingSampler 及 D‑Wave Advantage2 量子退火；专家迭代（expert‑iteration）奖励驱动训练；log‑slack 约束编码与动态惩罚校准；以及对比的路段激活与集合划分编码。

**📊 数据集**

实验使用 Solomon 基准集（56 个 CVRPTW 实例，涵盖 C、R、RC 家族与窄/宽时窗），测试 N=10–100 的多规模场景。

**📈 对比分析**

与传统按家族调参的手工合聚及静态惩罚基线相比，适应性惩罚将原始违约次数从 33 降至 0.06，预修复可行率提升至 94.8%；GNN 合聚在 N=10 时实现 100% 可行率，整体 83% 对比 69%；在 N=80/100 时显著领先；在硬件上，同变量数下从 0.02% 提升至 39% 可行率；QUBO 规模缩减 5–6 倍，保持可嵌入性。

**⚠️ 局限性**

局限性包括：硬件实验仅在 N≈10–20 的可嵌入范围内验证，未展示相对于经典采样的性能优势；最终解成本在经典修复+局部搜索后仅匹配而未超越；自适应惩罚主要解决 slack 约束的系数范围问题，非路由本身固有；共聚仍保持对问题规模的多项式增长；实验仅覆盖 Solomon 数据集，未验证其他实例或编码方案。

---

## 187. Tracing Audio Grounding and Answer Selection in Audio LLMs

**arXiv ID:** 2609.04637 | [PDF](https://arxiv.org/pdf/2609.04637v1)

**作者:** Hyebin Cho `[一作]` (Korea Advanced Institute of Science and Technology), Joon Son Chung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究系统分析了训练如何增强 Audio LLM 对音频证据的使用，揭示了内部机制与层级作用；

**💡 创新点**

创新点在于通过注意力裁断和层级干预，首次将训练对音频信息的加强细化到模型不同层级，并定位 LoRA 更新最关键的层区；

**🔧 技术方法**

主要技术包括注意力裁断（Attention Knockout）、层级注意力干预、LoRA 适配器训练、以及基于行为敏感度的评估；

**📊 数据集**

使用了 AudioMCQ-StrongAC-GeminiCoT（19,480 条）进行训练，评估数据集为 ADQA-Bench、MMAU-test-mini、MMAR、MMSU 共 9,998 条；

**📈 对比分析**

通过比较训练前后模型在原音频、静音、匹配音频等条件下的准确率及干预效果，发现训练后模型对音频的敏感度显著提升，性能提升在 ADQA-Bench、MMAR、MMSU 上均为显著正向；

**⚠️ 局限性**

局限性包括仅评估两种 LoRA 适配的 Audio LLM，未验证在更大模型或其他任务上的泛化；在 MMAU 数据集上未见正向迁移，表明方法对不同数据分布的鲁棒性仍待提升。

---

## 188. Why Is SHAP Not a Reliable Standalone Explanation Framework for Malware Detection?

**arXiv ID:** 2609.04626 | [PDF](https://arxiv.org/pdf/2609.04626v1)

**作者:** Seyedreza Mohseni `[一作]` (University of Maryland Baltimore County), Manas Gaur `[通讯]` (University of Maryland Baltimore County)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过理论证明和实验验证，阐明在静态PE文件的恶意软件检测中使用SHAP解释时，由于特征间高度相关性，interventional SHAP会产生离群组合、conditional SHAP会出现冗余衰减、代理赋值和符号不稳定等问题，从而使得SHAP不能作为独立的解释框架。

**💡 创新点**

创新点包括：① 提出了针对恶意软件特征依赖性下SHAP失效的三类正式定理（离群组合、冗余衰减、代理赋值与符号不稳）；② 在理论基础上设计了“冗余注入”实验来验证定理；③ 将定理与实际数据集（EMBER-2018/2024、BODMAS）结合，展示了不同特征依赖模式（冗余 vs 互补）对interventional与conditional SHAP的具体影响。

**🔧 技术方法**

主要技术手段包括：SHAP（interventional 与 conditional 形式）与 TreeSHAP 计算；梯度提升树模型（LightGBM、XGBoost）作为被解释器；特征依赖度量（组间相关系数、互信息）；冗余特征注入实验；以及对解释分配与预测性能的统计评估。

**📊 数据集**

使用的公开数据集为：EMBER-2018、EMBER-2024 和 BODMAS，仅保留静态PE特征，去除哈希等元信息。

**📈 对比分析**

方法对比：以固定的 LightGBM / XGBoost 检测器为基准，先评估其 AUC（均 >0.99），随后通过不同 SHAP 设置（缺失规则、背景分布、代理冗余）观察 SHAP 分配变化。实验显示：预测性能保持不变，但在冗余注入时原特征的 SHAP 值可下降至 40‑50% 甚至出现负值；在互补组间互信息较高的 EMBER-2024 中，interventional SHAP 产生的离群组合导致解释不可信。性能方面，模型准确率高，而 SHAP 解释的稳定性与可靠性显著受限。

**⚠️ 局限性**

局限性：① 结果仅针对静态PE特征与树模型；不同模型（如深度网络）或其他特征集可能表现不同；② 需要明确指定缺失规则、背景分布及简化映射，解释的有效性取决于这些人为设定；③ 对于高维、强相关特征的解释仍存在不确定性，无法完全替代人工分析；④ 论文未讨论对动态分析或多模态特征的适用性。

---

## 189. Pack It My Way: Triadic Human-Robot Collaboration for Personalized Autonomous Packing

**arXiv ID:** 2609.04620 | [PDF](https://arxiv.org/pdf/2609.04620v1)

**作者:** Sandeep Chowdary Kotapati `[一作]` (New Jersey Institute of Technology), Tsung-Chi Lin `[通讯]` (New Jersey Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在个性化自主包装场景下的三角人机协作，比较人类专家与语音代理在三类偏好（保护、紧凑、分组）下的中介效果。

**💡 创新点**

提出了三角人机协作框架与Show‑Correct‑Generalize流程，并首次用语音代理实现自动偏好中介与人类专家进行对比。

**🔧 技术方法**

使用Kinova Gen3机械臂、Orbbec相机、手势识别与Whisper语音转写，再通过规则解析器将口头指令映射为笛卡尔放置目标。

**📊 数据集**

依靠12名参与者构建的自制数据集（含ArUco标记物体的三种任务配置），进行用户实验。

**📈 对比分析**

采用配对t检验、TOST等统计方法评估包装质量；结果显示在保护与分组任务两种中介实现质量可比，紧凑任务语音代理表现更佳，满意度与可靠性评价差异不显著。

**⚠️ 局限性**

局限性包括样本量小、任务与对象简化、固定的语音命令词典、单一人类专家、未测试长期交互与多轮纠正。

---

## 190. $τ^τ$-Bench: An Environment for End-To-End, Realistic Agent Construction

**arXiv ID:** 2609.04611 | [PDF](https://arxiv.org/pdf/2609.04611v1)

**作者:** Quan Shi `[一作]` (Princeton University), Victor Barres `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 τ^τ-bench 基准，用来评估 AI 开发者在真实业务约束下构建可部署客户服务代理的能力。

**💡 创新点**

将代理构建本身视为任务，结合多模态业务记录、交互式客户、REST API、模型预算等因素，提供可度量且真实的构建挑战。

**🔧 技术方法**

使用 LLM 驱动的开发者代理、模拟客户端与用户、对话式评估器，结合多种 LLM（Claude Opus 5、GPT‑5.6、Kimi K3 等）和自定义工具链。

**📊 数据集**

基于 τ‑bench 四个领域（航空、零售、电信、银行）生成 2,868 份多模态证据档案（约 5.5 M 文本标记），并对公开域重新命名以防记忆泄露。

**📈 对比分析**

通过对比六种开发者配置与专家参考上限，采用平均任务奖励减去预算超支罚分的得分；最强配置 Claude Opus 5 仅达 23.9% 通过率，专家参考 82.2%，各领域差距显著，尤其银行领域。

**⚠️ 局限性**

局限在于客户端与用户模拟过于简化，单次构建未评估方差，数据集被审计保持一致性而非真实业务中的冲突与缺失，未包含后期维护与需求变更等工作。

---

## 191. Data-Driven Discovery of Composition-Dependent Constitutive Models for Hyperelasticity and Viscoelasticity of Digital Materials

**arXiv ID:** 2609.04541 | [PDF](https://arxiv.org/pdf/2609.04541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 192. When Do Internal Probes Beat Reading the Answer? Miscalibrated Readouts and Behavior-Concealed Knowledge in Language Models

**arXiv ID:** 2609.04582 | [PDF](https://arxiv.org/pdf/2609.04582v1)

**作者:** Gnaneswar Villuri `[一作]` (Stony Brook University), Alex Doboli `[通讯]` (Stony Brook University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在逻辑验证任务上对小型语言模型的知识与行为进行系统诊断，发现模型答案失配是由输出阈值失准导致的。

**💡 创新点**

证明大部分所谓的“隐藏知识”已存在于模型的输出logit中，仅需对阈值进行一次校准即可恢复；提出三种失配诊断并给出单参数修复方案。

**🔧 技术方法**

使用结构保留的逻辑推理语料、线性探针、margin AUC、阈值校准、少量示例（few‑shot）等技术。

**📊 数据集**

数据集包括30种自然演绎结构的逻辑语料（共600例）、600个单编辑伪结论、503个确定性伪结论，以及一个空间推理（迷宫）任务。

**📈 对比分析**

通过与文本表面特征、冻结编码器基线、跨模型探针等对比，修正后0.6B模型的行为准确率从50%提升至81%，8B模型提升至94%；探针AUC最高达0.99。

**⚠️ 局限性**

局限性包括使用人工构造的英文数据、模型规模不超过8B、探针仅为相关性分析、未在自然噪声标签任务中验证。

---

## 193. Extremely Sparse Supervision Incentivizes Reasoning Ability

**arXiv ID:** 2609.04565 | [PDF](https://arxiv.org/pdf/2609.04565v1)

**作者:** Zhishuai Liu `[一作]` (Duke University), Karim Bouyarmane `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在大语言模型后训练阶段探索并验证了极端稀疏监督（每条生成轨迹仅监督1-2个Token）可同样提升或超越传统全Token监督的推理能力。

**💡 创新点**

创新点在于突破了OPD对Token级密集监督的传统假设，揭示了少量高价值Token即可驱动模型推理性能的非平凡现象，并系统分析了极端正负奖励Token在不同教师-学生规模下的作用机制。

**🔧 技术方法**

主要使用了On‑Policy Distillation（OPD）及其稀疏变体（rand1tok、mintok、maxtok、minmaxtok、randmask0.1%、pctltail0.05%）和PPO等强化学习算法，结合KL、reverse KL等评价指标进行训练与分析。

**📊 数据集**

实验数据集涵盖Qwen3系列模型在数学推理任务（AIME 24/25、HMMT）、代码推理任务（LiveCodeBench、Eurus‑RL‑Code）以及Llama系列模型的推理数据。

**📈 对比分析**

通过与基线模型、教师模型、传统密集OPD以及PPO等方法对比，稀疏OPD在pass@k、avg@8等指标上均能匹配甚至超越密集OPD，且在大多数教师-学生组合下表现优于教师。

**⚠️ 局限性**

局限性包括实验仅覆盖Qwen3与Llama两大系列，未对Gemma/Mistral等其他模型验证；稀疏监督仍需教师模型支持，且对超大规模模型的训练成本与超参数调优缺乏深入探讨。

---

## 194. Fast Surrogate Modeling of Excitable and Oscillatory FitzHugh-Nagumo Dynamics with Parametric Neural Operators

**arXiv ID:** 2609.04549 | [PDF](https://arxiv.org/pdf/2609.04549v1)

**作者:** Andrew Franck `[一作]` (Occidental), Justin Li `[通讯]` (Occidental)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在一维空间上，用参数化的 Fourier 神经算子（FNO）构建 FitzHugh‑Nagumo 方程的快速可微前向仿真模型；

**💡 创新点**

创新点是将 FiLM 条件化引入 FNO，能够一次性覆盖 5 维参数空间，并在振荡与可触发两种动力学模式下实现高精度、可微仿真；

**🔧 技术方法**

使用 Fourier Neural Operator 结合 FiLM 条件化、GELU 激活、k_max=16 模式截断和残差连接；

**📊 数据集**

使用基于半隐式有限差分的 FHN 轨迹数据，包含 8000 条振荡模式轨迹（256 点、Δt=0.01、50 步）以及 1250 条可触发模式轨迹（L=8、Δt=0.02、100 步）；

**📈 对比分析**

与基线 FNO（广播 λ）和 DeepONet 对比，参数化 FNO 在振荡模式下相对 L²误差低于 0.1%，相较于基线减少约 30%，并实现 100–1000 倍的速度提升；在可触发模式下误差也保持在 0.02% 以内；

**⚠️ 局限性**

主要限制包括仅在一维域实验、对扩散参数的外推性能下降、对非线性参数敏感性分析有限，以及未直接验证逆向控制或多尺度情况。

---

## 195. Bridging Modalities and Tasks: A Unified Hierarchical ViT for SAR-to-Optical Translation and Semantic Segmentation

**arXiv ID:** 2609.04726 | [PDF](https://arxiv.org/pdf/2609.04726v1)

**作者:** Siyuan Liu `[一作]` (Northwestern Polytechnical University), Huihui Li `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了BMT统一框架，联合优化SAR到光学图像翻译和语义分割任务。

**💡 创新点**

设计了LocalViTBlock融合全局自注意与局部卷积、ControlNet式多尺度条件注入以及有限Kendall不确定性权重，实现双任务协同学习。

**🔧 技术方法**

基于层次Vision Transformer、离散小波变换、双分支解码器、生成对抗、感知损失、颜色直方图损失等多种技术。

**📊 数据集**

在公共WHU-OPT-SAR配对数据集和自构造的HRSID-DIOR非配对船舶数据集上进行训练与评估。

**📈 对比分析**

与多种S2O生成器（CycleGAN、Pix2pix等）和语义分割模型（FCN、U-Net、SegFormer等）对比，BMT在SSIM、FID、KID、mIoU和mPA等指标上均取得最优或竞争性结果。

**⚠️ 局限性**

生成与分割目标存在一定的权衡，单任务提升往往牺牲另一任务性能，且在复杂背景或密集船只场景下仍表现有限。

---

## 196. Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models

**arXiv ID:** 2609.04714 | [PDF](https://arxiv.org/pdf/2609.04714v1)

**作者:** Minji Kim `[一作]` (POSTECH), Hyounghun Kim `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在安全微调时误拒合法请求的根因，并提出仅使用理由（rationale）进行训练的方案

**💡 创新点**

发现冗余的“拒绝声明”是导致误拒的主要因素，去除该声明可显著降低误拒率，且对安全性影响极小

**🔧 技术方法**

采用 QLoRA 微调、vLLM 贪婪解码、内省评估（AdvBench、MaliciousInstruct、XSTest‑Safe、OKTest）以及零样本/多样本推理

**📊 数据集**

安全微调数据：Safety‑Tuned LLaMAs；指令数据：Alpaca；实验用的安全与伪安全查询集（AdvBench、MaliciousInstruct、XSTest‑Safe、OKTest）

**📈 对比分析**

与传统包含拒绝声明的安全微调对比，单理由训练在保持相同安全性（违规拒绝率≈不变）的前提下，误拒率下降30%‑70%（各模型具体表现见论文表格）

**⚠️ 局限性**

实验规模有限，仅评估8B‑7B模型；未探索大模型（>80B）以及更大规模安全数据集的影响

---

## 197. Simulation-free Unbalanced Dynamic Optimal Transport with General Growth Penalty

**arXiv ID:** 2609.04710 | [PDF](https://arxiv.org/pdf/2609.04710v1)

**作者:** Junda Ying `[一作]` (Peking University), Lei Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了SUDO框架，能够在不依赖模拟的情况下解决带任意增长惩罚的未平衡动态最优传输（UDOT）问题，进而推断单细胞轨迹与细胞增殖/凋亡动力学；

**💡 创新点**

创新点在于：①理论证明凸增长惩罚为非退化UDOT的必要条件；②通过学习“行进Dirac”与传输成本，采用投影梯度下降求解半耦合，完全摆脱了传统方法对闭式解析解（仅适用于WFR）的依赖；③实现了对非二次增长惩罚（如“only-growth”等）的高效推断；

**🔧 技术方法**

使用神经网络学习行进Dirac路径与传输成本，投影梯度下降求解半耦合，随后采用未平衡流匹配（UFM）训练速度场与增长率网络；

**📊 数据集**

在合成数据（二维、Dyngen 5D、1000D高斯）以及真实单细胞RNA测序数据（Mouse血液干细胞、EMT、mouse血液造血等）上进行实验；

**📈 对比分析**

与基准方法（TIGON、DeepRUOT、VarRUOT、WFR-FM）比较。SUDO在WFR基准上取得与WFR-FM相当的测量匹配和UDOT成本，且在大规模数据上训练时间和内存消耗显著低于基准，提升约10-30倍；

**⚠️ 局限性**

局限性在于：①目前仅处理确定性UDOT，随机动力学（RUOT）尚未得到严格理论支持；②需要先指定增长惩罚函数，无法直接从数据中学习或校准该惩罚；

---

## 198. Decoding Error Probability of the Random Code Ensemble over the Erasure Channel

**arXiv ID:** 2609.04688 | [PDF](https://arxiv.org/pdf/2609.04688v1)

**作者:** Chin Hei Chan `[一作]` `[通讯]` (Hetao Institute of Mathematics and Interdisciplinary Sciences), Chin Hei Chan (Hetao Institute of Mathematics and Interdisciplinary Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文给出了所有(n,M)_q码在q-ary删失信道下，按无歧义、列表和最大似然三种解码原则的平均误码概率的显式公式，并推导了其随码长增长的误差指数。

**💡 创新点**

创新点在于将已知的线性码结果推广到所有（含非线性）码族，首次得到该族平均误码概率的闭式表达，并与已有线性码的误差指数进行对比，揭示了列表解码的相对优势。

**🔧 技术方法**

采用随机码族平均分析、incorrigible集分布的概率计数、组合与极限定理，结合信息理论的容量与相对熵工具，完成误码概率与误差指数的推导。

**📊 数据集**

使用的是全随机码族（所有(n,M)_q码）作为实验对象，不涉及具体数据集；误码概率通过组合计数直接计算。

**📈 对比分析**

与随机线性码和随机检验矩阵码族对比，误差指数在无歧义和最大似然解码下相同，而固定列表大小时，非线性码的指数略低于对应线性码，证明了列表解码在该族中更优；图示表明在二进制删失信道中，当列表大小为4时，非线性码的指数高于线性码。

**⚠️ 局限性**

局限性在于仅给出平均性能指标，未给出具体构造码；仅适用于大码长极限定理，实际短码表现未知；对非线性码的误差分布分析仍依赖概率估计，难以在实际编码方案中直接使用。

---

## 199. VizIt: A multi-view framework for exploring single-cell, spatial, and genetic data online

**arXiv ID:** 2609.04658 | [PDF](https://arxiv.org/pdf/2609.04658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 200. Beyond Prompt-to-App: Accountable Translation in Teacher-Facing Agentic Authoring

**arXiv ID:** 2609.04679 | [PDF](https://arxiv.org/pdf/2609.04679v1)

**作者:** Nizam Kadir `[一作]` (Singapore University of Technology and Design), Lay Kee Ang `[通讯]` (Singapore University of Technology and Design)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一次教师专业学习工作坊中，对一个自然语言驱动的应用构建系统（Studio）的六次构建尝试进行了有限追踪分析，探究其从简短说明到最终可发布软件的整个转换链。

**💡 创新点**

提出“可追溯翻译（accountable translation）”框架，将专业承诺在多阶段管道中的属性（目的、决策权、证据边界、验证范围、修复权）纳入可归因、可检验、可争议的评估维度，首次系统化地描述了教师专业责任在 AI 生成软件中的转移与保持。

**🔧 技术方法**

采用了多模型与规则相结合的流水线：编译器扩展简短说明为平台契约，计划器生成应用蓝图，生成器输出代码，分析器与安全检查器做静态验证，最后进入注册库；技术核心是大语言模型（LLM）与基于规则的编译与验证组件。

**📊 数据集**

使用了两类数据集：工作坊公开聊天的 37 条教师提交的结构化想法（包含问题、目标、控制、证据等字段）以及 Studio 系统中可追踪的 6 次构建记录（包含简短说明、编译结果、计划、分析、检查与注册状态）。

**📈 对比分析**

研究主要通过定性追踪与比对不同阶段的状态标记（如编译、计划、分析、检查、注册）来评估可追溯翻译的实现情况，并未进行传统意义上的性能量化或对比实验，报告的是各阶段状态的碎片化与不一致性。

**⚠️ 局限性**

局限性包括样本量极小（仅 6 次构建、3 账户）、缺乏与公开想法的直接关联、未对生成的应用进行运行时验证或教师使用评估、系统日志不完整导致无法追溯具体模型或参数变化，以及研究团队内部角色可能带来的偏见。

---

## 201. Train What You Deploy:Token-Faithful Post-Training of a Production Coding

**arXiv ID:** 2609.04678 | [PDF](https://arxiv.org/pdf/2609.04678v1)

**作者:** Cheng Li `[一作]` (KunlunMeta), Chi Hong `[通讯]` (KunlunMeta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在生产级终端/编程代理的后训练过程中，提出了一个“忠诚耦合”框架，并基于此实现了可证的分割策略C-DPPO，保证训练与部署环境的一致性和梯度可信度。

**💡 创新点**

创新点：①将采样权交给训练器，确保每个生成的token都是模型真实采样；②通过硬化的训练模式和协商头部声明所有非策略行为，消除后台调用污染；③引入两侧 TV 逼近证书、适配 K 规则、序列级预算和对 log‑prob 误差的鲁棒决策，构成完整的可证优化流程。

**🔧 技术方法**

技术：基于 Megatron–SGLang 的训练框架；DPPO 与 C‑DPPO 优化器；消息树线性化、漂移分类；会话‑键路由、HTTP 适配器；KL 与 TV 的理论证明；可证预算分配与鲁棒化规则。

**📊 数据集**

数据集：公开的 TMax‑15K 终端任务集合（包含 TMax‑100 评估子集）以及基于代码仓库的 SWE 任务，使用 Baize5B 与 Baize10B 模型。

**📈 对比分析**

比较方法：在同一训练协议（350 步）下，分别使用 Standard DPPO 与 C‑DPPO，最终在同一 TMax‑100 集合上评估。结果：C‑DPPO 在两种规模下均比 Standard DPPO 提升约 3 分（34→37、41→44），且证书完整率和解决率均超过 84%。

**⚠️ 局限性**

局限性：仅在单一跑（单机、单种规模）下进行，缺乏多种随机种子、不同任务集与更长训练时间的稳健性验证；证书有效性依赖于严格的协议与误差半径假设；在不同代理架构或多代理场景下的可迁移性仍需进一步实验。

---

## 202. BF16 Component-Product Emulation of FP32 and FP64 GEMM on Intel AMX

**arXiv ID:** 2609.04663 | [PDF](https://arxiv.org/pdf/2609.04663v1)

**作者:** Bing Cui `[一作]` (Maginfra Co), Yu Liu `[通讯]` (Maginfra Co)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

**🎯 论文内容**

本文提出了一种在Intel AMX矩阵引擎上使用BF16低精度矩阵乘法来近似高精度（FP32、FP64）GEMM的算法，先将FP32或BF16范围内的FP64矩阵拆解为多层BF16子矩阵，再按选定的组件乘积（FP32采用三分BF16+六乘，FP64采用固定六分Ozaki+可调产品计数）进行AMX计算，并在FP32或FP64层级重构得到近似结果；

**💡 创新点**

创新点在于：①利用BF16与FP32/FP64共享指数范围的特性，将高精度矩阵拆成可控数量的BF16子矩阵；②为FP32提供三分BF16+六乘的快速六乘子方案，实现FP32级别误差；③为FP64提供固定六分Ozaki拆解并通过可调的三角化产品计数实现准确性与性能的权衡；④在AMX上实现operand‑reuse调度、预计算拆解/打包以及FP32/FP64重构，从而最大化硬件吞吐与内存效率；

**🔧 技术方法**

技术手段包括BF16残差拆分（FP32三分BF16、FP64六分Ozaki）、Intel AMX BF16×BF16→FP32累加指令、VNNI格式打包、8×2维块化、组件重用调度、预计算组件缓冲区、FP32/FP64重构、OpenMP并行分块；

**📊 数据集**

使用随机均匀/高斯矩阵做基准，另外对FP64使用缩放或对数均匀矩阵、以及抵消敏感矩阵；测试全部为方阵，矩阵大小从256到32768；

**📈 对比分析**

与Intel oneMKL SGEMM（FP32）和DGEMM（FP64）对比，采用Frobenius范数相对误差和GEMM‑scaled componentwise误差衡量准确度；性能方面，AMX‑FP32在所有规模上均可实现1.16‑1.32×的速度提升；AMX‑FP64‑6在大规模（≥8192）可获得约1.7×的速度提升，而更多产品计数（10/15/21）虽然准确度更好，但速度提升随之下降；

**⚠️ 局限性**

局限性包括：①FP64路径仅支持BF16指数范围内的数值，无法处理完整FP64指数或下溢/子规范数；②仅验证方阵，未涉及矩形或通用BLAS接口；③固定六分拆解和可调产品计数未实现最优的误差/性能闭环；④受AMX八个8×2维寄存器限制，无法扩大微核或存储更多组件；⑤未覆盖NaN、Infinity、异常值的后备处理；

---

## 203. Harmonica: Accurate and Lightweight Instrument-Agnostic Music Transcription

**arXiv ID:** 2609.04640 | [PDF](https://arxiv.org/pdf/2609.04640v1)

**作者:** Longshen Ou `[一作]` (BandLab Technologies), Taemin Cho `[通讯]` (BandLab Technologies)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一系列名为Harmonica的轻量级、可扩展的多源乐器无关音乐转录模型；

**💡 创新点**

创新点在于多层次的谐波卷积（shift‑and‑aggregate）机制，既能充分利用谐波信息，又显著减少参数；

**🔧 技术方法**

采用CQT谱图输入、残差网络、频段分组LSTM以及参数高效的多深度谐波卷积；

**📊 数据集**

使用MAESTRO、MAPS、GuitarSet、GAPS、EGDB、GOAT、URMP、Slakh2100八大数据集进行训练与评测；

**📈 对比分析**

在统一的多源无关评估协议下，与十一种基线模型对比，Harmonica在每个规模下均取得最优或接近最优的帧级F1和音符级F1，且nano版仅26.3K参数即可实现1,622×实时速度；

**⚠️ 局限性**

主要局限是对极其失真或非音高化乐器的处理仍不够理想，且模型仍基于单乐器音频的转录，尚未在复杂混音环境下充分验证。

---

## 204. Same Request, Different Answer: Quantization Amplifies Cache-Induced Divergence in LLM Serving

**arXiv ID:** 2609.04748 | [PDF](https://arxiv.org/pdf/2609.04748v1)

**作者:** Aditi Patodiya `[一作]` `[通讯]` (Independent Researcher), Aditi Patodiya (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究前缀缓存对LLM推理可重复性的影响，量化并定位缓存导致的结果不确定性，并探讨权重量化与该效应的交互。

**💡 创新点**

提供系统化、可控实验验证缓存开启导致输出非重复性的具体幅度与机制，首次揭示量化精度会放大这一效应，并提出通过恢复缓存状态恢复可重复性的方案。

**🔧 技术方法**

使用配对实验设计、两大开源推理引擎（llama.cpp 与 vLLM）、多种权重量化格式（FP16、Q8_0、Q4_K_M、Q3_K_M）、多轮代理工具任务与单轮数学任务，并采用 token 差异、Wilson 区间、McNemar 检验等统计手段。

**📊 数据集**

主要使用 Berkeley Function Calling Leaderboard（多轮代理工具使用）和 GSM8K（单轮小学数学）两套公开数据集。

**📈 对比分析**

对比缓存开启与关闭下重复运行同一工作负载，测量 episode 级别的轨迹差异率、token 级别差异以及准确率变化。结果显示开启缓存导致 36–91% 的 episode 轨迹发生变化，量化从 FP16 到 Q3_K_M 时差异率提升至约 75%，两引擎实验一致但规模不同，表明该效应普遍存在。

**⚠️ 局限性**

局限于 7–14B 规模模型、单一硬件（RTX 4090）、单租户实验环境，未覆盖更大模型、云端服务、多租户负载或不同推理批量；实验聚焦 greedy 解码和 batch size 1，未探究其他解码策略或批量对该效应的影响。

---

## 205. Where to Look Matters: Learning Influential Views for VLM-based 3D Visual Grounding

**arXiv ID:** 2609.04741 | [PDF](https://arxiv.org/pdf/2609.04741v1)

**作者:** Tsung-Chih Chiang `[一作]` (National Tsing Hua University), Chun-Yi Lee `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 IVSGround 框架，通过学习视角选择器来挑选对视觉语言模型（VLM）最具信息量的摄像机视图，从而提高 3D 视觉定位（visual grounding）的准确性。

**💡 创新点**

核心创新在于：① 用基于 VLM 反馈的两阶段拒绝抽样（per-view 和 comparative）生成无标注的“影响视图”监督信号；② 训练轻量级视角选择器并保持 VLM 冻结，实现零样本（zero-shot）定位；③ 通过候选对象筛选（Cascade Object Screening）和两轮淘汰锦标赛进一步提升效率。

**🔧 技术方法**

技术手段包括：Qwen3-VL 作为 VLM 与视角选择器；LoRA 微调 Qwen3-VL‑8B 训练视角选择器；Rejection Sampling 生成正负样本；CLIP 文本嵌入 + Open‑YOLO3D 进行候选筛选；图像与语言推理的比较式（pairwise tournament）定位。

**📊 数据集**

使用 ScanRefer 与 NR3D 两个 ScanNet 基准数据集，包含约 51k 描述、41k 参考句，涵盖不同难度与视角依赖场景。

**📈 对比分析**

与现有 VLM‑Grounder、SPAZER、SeeGround 等方法比较，IVSGround 在 ScanRefer Acc@0.25 上从 57.2% 提升至 58.0%，在 NR3D top‑1 accuracy 上从 48.0% 提升至 62.1%，尤其在视角依赖与同类干扰场景表现显著提升；混合使用可见度视角和学习视角时，准确率可达 68.8%。

**⚠️ 局限性**

局限性包括：① 视角选择器对特定 VLM 的依赖（虽然可迁移但仍需微调）；② 仍需对每个候选对象生成多视图，计算量较高；③ 只针对 3D 点云场景，未探讨动态或非室内环境；④ 依赖大量 VLM 推理，成本和推理时延较大。

---

## 206. Building a research-software catalog with a coding agent: from hackathon prototype to public deployment

**arXiv ID:** 2609.04711 | [PDF](https://arxiv.org/pdf/2609.04711v1)

**作者:** Kazuyoshi Yoshimi `[一作]` (Institute for Solid State Physics, University of Tokyo), Gotai Yamada `[通讯]` (Institute for Solid State Physics, University of Tokyo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并硬化了一个基于生成式 AI 的研究软件目录，随后将其经验迁移到更大的人手维护门户 MateriApps，并开发了检索代理。

**💡 创新点**

通过将 coding agent 与对抗式审查相结合，实现快速开发与可靠发布的迭代流程；发现并处理无错误但误导性输出的“silent failure”，并提出了显式验证与监控的改进方法。

**🔧 技术方法**

使用 Claude Code/Codex 进行编码与审查；本地 Ollama 语言模型做检索增强生成；多语言 BGE‑M3 句向量与 HNSW 近似检索；跨编码器重排；前端浏览器层测试和配置验证。

**📊 数据集**

68 条仓库记录（hackathon 版）；336 个 MateriApps 应用及其 49,485 条索引段落（包含外部文档抓取）。

**📈 对比分析**

在小型目录上，检索 hit@10≈0.91、MRR≈0.70；在 MateriApps 上，hit@10≈0.84‑0.95、MRR≈0.80‑0.85；通过对比模板查询与改写查询，以及字典路由、重排与向量检索等模式评估性能。

**⚠️ 局限性**

主要局限包括缺失/重复文档导致检索失效；silent failures 难以自动检测；生成质量尚未在真实用户查询上充分验证；实验规模有限，未与传统开发方式做对照。

---

## 207. FinalityBench: An Effect-Level Benchmark for Agent Decisions Under Delayed and Conflicting Financial Finality

**arXiv ID:** 2609.04706 | [PDF](https://arxiv.org/pdf/2609.04706v1)

**作者:** Abhishek Sharma `[一作]` `[通讯]` (IEEE), Abhishek Sharma (IEEE)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FinalityBench，一个基于可执行环境的支付异常决策基准，利用隐藏的事件日志、故障注入和最终性探测来评估代理在临时信息冲突下的决策效果。

**💡 创新点**

创新点包括：① 以效果层（实际移动的金钱）为评判标准的“最终性”基准；② 通过“同构任务对”构造不可区分但行动相反的情境，揭示单任务准确率掩盖的决策风险；③ 细粒度的六类故障（延迟、重复、丢失、重排、部分提交、陈旧读取）与可解释的损失归因；④ 公开的权威最终性探测器（oracle）与手写策略的对比。

**🔧 技术方法**

技术手段包括：模拟器实现四系统（处理器、分类账、ERP、银行）并行的投递流；故障注入引擎；基于 JSON 的任务生成器；多策略接口（随机、乐观、投票、规则、ReAct、反射、最终性门）；效益评分脚本，利用 PostgreSQL 存储日志与结果。

**📊 数据集**

数据集：约 1605 条任务，覆盖 9 种异常 archetype，使用同构任务对构造；每个任务在 20 个随机种子下多次执行，形成 32100 条 episode；此外还有一个 1605 条任务的重抽样测试集，用于评估调参迁移。

**📈 对比分析**

比较方法：在每个任务与种子下执行策略，记录单任务准确率、平均超额损失（与最终性 oracle 的差值）、预先确定的不可逆行动比例；按单任务准确率、配对损失及整体经济损失对策略排序。实验表明，最终性门策略在绝大多数指标上均优于其它手写策略，单任务准确率高但配对损失差的乐观策略表现最差。

**⚠️ 局限性**

局限性：① 模拟环境的延迟、批处理等参数基于合理假设，未与真实支付系统对标；② 任务仅为单一订单、单一捕获，缺乏多件、分拆或并发异常；③ 仅评估 9 个手写策略，模型 arm 受限于免费配额；④ 经济参数（商品成本、争议费、升级费）为固定假设；⑤ 参考策略并非绝对最优，仍有可改进空间。

---

## 208. The Cross-Correlation Distribution of the Niho-Type Decimation $d=4(2^m-1)+1$

**arXiv ID:** 2609.04683 | [PDF](https://arxiv.org/pdf/2609.04683v1)

**作者:** Maosheng Xiong `[一作]` (Hong Kong University of Science and Technology), Haode Yan `[通讯]` (Harbin Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了二进制 m 序列与其 Niho 型指数 d = 4(2^m-1)+1 的反向衰减序列的交叉相关分布，等价于求解多项式 f_a(x)=x^7+ax^4+ãx^3+1 在单位圆 U_{q+1} 上的根数分布。

**💡 创新点**

创新点在于利用根集的标准化（乘积为 1）与其伴随的三元“可解析元”(resolvent) 进行枚举，将 4 元根集的计数转化为有限域上若干方程的解数，随后用 Kloosterman 和混合 Kloosterman 和求值，最终得到完整的交叉相关值与频率显式公式。

**🔧 技术方法**

技术主要包括：根集归一化与伴随可解析元；对有限域 U_{q+1} 上的 4 元子集进行分块计数；利用特征和 Kloosterman 和（以及 Carlitz 定理）计算相关字符和求和；方程求解与代数几何推导；最终通过组合计数得到频率。

**📊 数据集**

无外部数据集；论文完全基于理论推导与有限域数论计算。

**📈 对比分析**

与先前仅给出上限或局部分布的结果对比，本文给出完整的分布公式；数值检验表明公式在所有正整数 m 下都成立；在实际序列设计中可直接使用该分布计算相关性能。

**⚠️ 局限性**

局限性包括：只针对特定 Niho 型指数 d=4(2^m-1)+1；扩展到更一般的 Niho 型指数或其他特征时需重新构造伴随可解析元与求和技巧；且计算复杂度高，缺乏简易的计算算法实现。

---

## 209. WEECFP-SuRGE: Wide Embedded Extended Connectivity Fingerprint with Substructure Rotary Graph-distance Encoding

**arXiv ID:** 2609.04672 | [PDF](https://arxiv.org/pdf/2609.04672v1)

**作者:** Robert Epps `[一作]` `[通讯]`, Robert Epps

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了参数无关的1024维连续分子指纹WEECFP及其变体，结合图距离的RoPE类编码SuRGE，构建了从子结构到Transformer的全新表示与注意力机制；

**💡 创新点**

创新点在于将Morgan子结构以多位置带符号散射写入向量，生成可训练的连续指纹，同时引入SuRGE以图距离为基准的旋转位置编码，使Transformer能在无预训练的情况下捕获分子拓扑信息；

**🔧 技术方法**

采用WEECFP子结构分词、SuRGE旋转编码、3层Transformer（或CNN/MLP）以及7模型混合等技术；

**📊 数据集**

使用MoleculeNet（9项回归/分类任务）和TDC ADMET（22项ADMET基准）作为评估数据集；

**📈 对比分析**

与传统位图指纹+梯度提升树或已预训练的GNN/Transformer模型比较，WEECFP-SuRGE在MoleculeNet回归任务中击败大多数基线，在TDC ADMET回归榜单上获得平均排名3.9（无预训练时排名第一），在总体榜单上排名第二；

**⚠️ 局限性**

局限包括对SMILES的立体化学敏感度有限、对超大分子和宏环的token截断问题、以及在分类任务上仍落后于预训练模型，需进一步探索预训练或多维度特征融合来提升性能。

---

## 210. Harness-agnostic detection and immunization of reward hacking in self-evolving language models

**arXiv ID:** 2609.04665 | [PDF](https://arxiv.org/pdf/2609.04665v1)

**作者:** Rongxin Yang `[一作]` (Fullive-AI), Bin Chong `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种黑盒监测与免疫框架 HackProbe，用以检测并抵御自进化语言模型中的奖励黑客行为。

**💡 创新点**

创新点在于：①双层探针银行（固定核心+在线旋转新层）保证了检测指标与模型优化路径不循环；②六个单侧检验通过 Šidák 校正融合成校准后的 p 值风险评分；③风险感知重选策略在不泄露核心信息的前提下实现候选重选，兼顾误报成本。

**🔧 技术方法**

采用统计检验（Page–Hinkley 变点检测、置信度率、差异测试）、信号融合（Šidák 校正）、黑盒探针采样、定量阈值校准与安全预算推导。

**📊 数据集**

使用小学与竞赛数学数据集：GSM‑8K、MATH、GSM‑Symbolic、GSM‑Plus 以及保留的 GSM‑1k 用作 gold 审计与测试。

**📈 对比分析**

与无探针、固定探针、污染测试、单陷阱、绝对轨迹等基线对比，HackProbe 在 AUROC 0.763（基线 0.663）上提升；误报率降至 0.434（基线 0.706），检测召回 0.845，平均检测延迟 0.28 代，免疫后平均能力提升 0.052，清洁运行成本仅 0.047，成本效益比 1.11。

**⚠️ 局限性**

局限性包括：误报率仍高（10% 以上），仅在 prompt‑level 自进化环境验证，缺乏权重级别评估；旋转保证仅覆盖完全匹配探针的攻击；需要足够的 probe 预算与 gold 审计数据，且对输出格式攻击的检测不充分。

---

## 211. Interpretability for Turing Machines

**arXiv ID:** 2609.04661 | [PDF](https://arxiv.org/pdf/2609.04661v1)

**作者:** Billy Snikkers `[一作]` (Resolution), Will Troiani `[通讯]` (Resolution)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文展示了如何通过敏感性分析技术识别图灵机中的算法结构，特别是在噪声图灵机的学习问题中。研究了图灵机的局部损失景观，并通过主成分分析和聚类方法在敏感性空间中恢复算法特征。

**💡 创新点**

创新点在于将敏感性理论应用于图灵机的解释性分析，证明了图灵机的对称性和路径分离性如何影响其敏感性矩阵的结构。

**🔧 技术方法**

使用了敏感性理论，结合主成分分析和聚类方法来分析图灵机的算法特征。

**📊 数据集**

使用了38,019个确定性有限自动机（DFA）作为数据集，这些DFA与特定语言一致，进行实验和分析。

**📈 对比分析**

通过与现有方法的比较，论文展示了敏感性矩阵的结构如何反映图灵机的内部计算结构，且在路径分离的情况下，敏感性矩阵的秩被限制在2以内，验证了理论结果。

**⚠️ 局限性**

论文的局限性在于对噪声图灵机的模型假设，可能无法完全捕捉所有类型的图灵机行为，且实验结果依赖于特定的输入分布和参数设置。

---

## 212. Choosing the Right Language Mode at Inference Time for Multilingual Reliability

**arXiv ID:** 2609.04653 | [PDF](https://arxiv.org/pdf/2609.04653v1)

**作者:** Ekata Mitra `[一作]` (Portland State University), Ameeta Agrawal `[通讯]` (Portland State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对多语言多选题进行文本范围和语言模式的系统实验，提出了可靠性感知自适应推理框架RAAI，动态调节英文辅助与双语推理，提升低资源语言推理准确率和校准性能。

**💡 创新点**

在不训练模型的前提下，首次引入基于ECE的动态路由和中间层风险指数的分组门控，实现了对翻译干扰与过度自信的自适应抑制。

**🔧 技术方法**

使用 LLaMA 3.1-8B-Instruct 与 Qwen 3-4B 语言模型，结合 logit‑lens ECE 评估、层级集成、风险指数门控与多语种翻译提示。

**📊 数据集**

评测数据集包括 Belebele 与 MMLU‑ProX‑Lite 两个平行多语种多选题数据集，覆盖高、中、低资源 9 种语言。

**📈 对比分析**

与传统单一语言模式（T/EN/TEN）对比，RAAI 的 RouteGate 与 SeqGate 在低资源层可提升 25–38% 准确率、降低 3–6% ECE，证明了自适应推理的显著优势。

**⚠️ 局限性**

局限在于仅评估闭集多选题，未考虑开放式生成、翻译质量与推理延迟，且对新语言或领域需额外小规模配置。

---

## 213. ConsensusBench: Benchmark of Consensus Nodes for LLM Reasoning via Outcome Reward Densifying

**arXiv ID:** 2609.04648 | [PDF](https://arxiv.org/pdf/2609.04648v1)

**作者:** Shi-Qi Yan `[一作]` (Alibaba Group), Zhen-Hua Ling `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建Consensus Nodes数据集并将其作为过程奖励，集成到GRPO/DAPO等RL框架中，以提升LLM的推理精度。

**💡 创新点**

提出可验证的中间结论（Consensus Nodes）作为奖励信号，既保留了可验证性，又避免人工标注；同时设计了ConsensusPR奖励机制及三种集成策略。

**🔧 技术方法**

使用强化学习（GRPO、DAPO）、多模型Rollout采样、语义聚类、规则化文本匹配及Token per Node等度量方法。

**📊 数据集**

采用AIME 2024/2025、GSM8K、MATH‑500、DAPO‑MATH‑17K以及自研的ConsensusBench。

**📈 对比分析**

在与vanilla GRPO/DAPO的同等训练条件下对比，ConsensusPR在Acc、NCR等指标上普遍提升（Acc提升至40%+，NCR>80%，TPN有显著下降）。

**⚠️ 局限性**

未开发独立的过程奖励算法，奖励直接嵌入可能未充分挖掘Consensus Nodes潜力；数据构建依赖闭源模型，可能引入偏差；对极端高难度任务的泛化仍有限。

---

## 214. CAGE: Coherence-Aware Graph Encoding for Retrieval-Augmented Generation

**arXiv ID:** 2609.04647 | [PDF](https://arxiv.org/pdf/2609.04647v1)

**作者:** Tong Qi `[一作]`, Erin Babinsky `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该文档提供了一个示例，展示如何使用ACL样式文件与LuaLaTeX或XeLaTeX。

**💡 创新点**

文档的创新点在于展示了多语言文本的处理方式。

**🔧 技术方法**

使用了LuaLaTeX或XeLaTeX技术。

**📊 数据集**

未提及具体数据集。

**📈 对比分析**

未提供比较的方法和性能评估。

**⚠️ 局限性**

文档缺乏具体的研究内容和数据支持，限制了其应用性。

---

## 215. Counting Beyond Instances: A Benchmark for Group-Individual Object Counting

**arXiv ID:** 2609.04716 | [PDF](https://arxiv.org/pdf/2609.04716v1)

**作者:** Rui Wang `[一作]` (Huazhong University of Science and Technology), Baoru Huang `[通讯]` (University of Liverpool)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Group-Individual Object Counting（GIC）任务，要求模型根据提示在同一图像中分别计数单个实例和其语义组；

**💡 创新点**

首次将计数单元（实例 vs 组）作为显式查询维度，并构建了配对实例与组注释及包含关系的BunchCount基准；

**🔧 技术方法**

使用基于GroundingDINO的检测框架，结合计数单元提示和基于组-实例包含关系的关系对齐损失（metric learning）实现统一计数；

**📊 数据集**

BunchCount数据集，包含1330张图像、89,254个实例点、11,065个组框以及包含关系；

**📈 对比分析**

与多种基线（Visual-Exemplar、Text-guided、VLM等）对比，GICount在个体计数MAE约12.2，组计数MAE仅1.85，显著优于传统计数模型（组计数MAE 9.21）并保持个体计数性能；

**⚠️ 局限性**

对齐关系仅在训练阶段有效，推理时仍需独立处理，且模型对极大/极小组规模的泛化仍有限，且依赖于预训练检测器的质量。

---

## 216. Model Retirement Creates Reproducibility Risk in Biomedical AI Publications

**arXiv ID:** 2609.04699 | [PDF](https://arxiv.org/pdf/2609.04699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 217. AngelFingerprint: A Traceable, Explainable, and White-Box Stealthy Watermark for Text-Guided Image Editing

**arXiv ID:** 2609.04709 | [PDF](https://arxiv.org/pdf/2609.04709v1)

**作者:** Bo-Han Kung `[一作]` (National Taiwan University), Shang-Tse Chen `[通讯]` (National Taiwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对文本引导扩散编辑的水印框架AngelFingerprint，能够在每幅编辑图像中嵌入编辑提示的CLIP嵌入并在后期可恢复

**💡 创新点**

1) 将水印直接写入LoRA权重，保持模型结构不变，实现白盒隐蔽性；2) 采用速度对齐锚点和中频频域滤波器，使水印既不影响编辑质量又能被提取；3) 以语义嵌入为载体，实现可追溯、可解释的水印

**🔧 技术方法**

LoRA权重微调、流匹配（flow‑matching）损失、余弦+InfoNCE嵌入恢复损失、速度对齐(anchor)损失、DCT中频带掩膜频域滤波器、CLIP文本编码器

**📊 数据集**

MagicBrush基准数据集；使用Stable Diffusion 3（SD3）和UltraEdit模型作为后端

**📈 对比分析**

与现有的后置编码器和提示反演基线（BLIP‑2、PromptStealer、VGD等）对比，AngelFingerprint在MagicBrush 200‑way提示检索中实现Top‑1 86%/0.860、MRR 0.917，远超对照方法；在可视质量上保持与未加水印模型相近，FID/KID仅略高；在量化/微调攻击下保持高鲁棒性（int8保持Top‑1 0.86，int4显著下降）

**⚠️ 局限性**

对极低精度量化（int4）和图像尺寸变换（重采样）攻击敏感；需要训练额外的LoRA和提取器，训练成本较高；在大规模开放式验证场景下仍需生成候选句子以对齐嵌入

---

## 218. How Do Language Models Represent and Use Phonological Information for Allomorph Selection?

**arXiv ID:** 2609.04708 | [PDF](https://arxiv.org/pdf/2609.04708v1)

**作者:** Sangwoo Kim `[一作]` (Seoul National University), Sangah Lee `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型如何在仅使用文本输入的情况下，利用音系条件实现屈折形式的选择，并通过内部机制进行因果解释。

**💡 创新点**

提出单词级Wug测试和逆向FutureLens干预方法，揭示英语不定冠词 a/an 的音系条件仅以一维方向在词嵌入空间编码，并证明模型通过预测后续触发词的音系特征来决定屈折形式。

**🔧 技术方法**

使用线性分类器与线性探针提取嵌入方向，FutureLens 进行未来状态预测，逆向映射实现因果干预，结合跨语言所有形变实验和多模型比较。

**📊 数据集**

采用英语、韩语、土耳其语、意大利语、法语维基百科文本数据，自动生成音系标签，并使用对应模型的词表。

**📈 对比分析**

与随机对照和非线性MLP探针对比，实验在Llama‑3.2‑3B‑Instruct、Qwen2.5‑3B‑Instruct、Gemma‑3‑1b‑it三模型中，一维方向解码准确率>99%，干预转化率达84–90%，跨语言实验显示同一模式普遍存在。

**⚠️ 局限性**

仅针对解码器型LLM及标准分词；音系标签来源于语料统计，可能含噪；FutureLens推断提供有限的机制解释，缺乏电路层面细节；未涵盖多模态或非文本输入。

---

## 219. Sustainable Edge Vision via Empirically Calibrated DVFS: Eliminating Thermal Throttling on Passively Cooled Hardware

**arXiv ID:** 2609.04705 | [PDF](https://arxiv.org/pdf/2609.04705v1)

**作者:** Aayush Marasini `[一作]` (University of Southern Mississippi), Zhaoxian Zhou `[通讯]` (University of Southern Mississippi)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在Raspberry Pi 5上为YOLOv8n持续推理设计并实现了基于经验校准的状态感知DVFS调度器，成功消除了热节流。

**💡 创新点**

创新点在于将阈值与传感器噪声和热步响应实验直接关联，结合停留时间、确认样本和导数保护，形成可复现、无手工调参的调度策略。

**🔧 技术方法**

使用的技术包括经验校准的DVFS、时间域守护、指数滑动平均与导数检测、统计自举、SHA256哈希验证和能耗测量。

**📊 数据集**

使用的数据集是USA子集的RDD2022数据集，YOLOv8n模型在70/10/20拆分上训练，测试视频为32秒循环播放。

**📈 对比分析**

与静态S0/S1/S2、仅阈值的反应式基线和主动冷却参考进行对比；主动式调度在不发生热节流的前提下，比反应式快6.8%、能耗低1.9%，在能耗/帧指标上甚至优于主动冷却。

**⚠️ 局限性**

局限性包括仅在单一Pi 5平台和单一YOLOv8n工作负载上验证，未验证其他SoC或工作负载，主动冷却与能耗/帧对比受软件堆栈和INT8量化误差影响，且对环境温度≥27 °C的非线性泄漏未充分探测。

---

## 220. LookThere! Sparse Vision by Reinforced Selection

**arXiv ID:** 2609.04698 | [PDF](https://arxiv.org/pdf/2609.04698v1)

**作者:** Sreehari Rammohan `[一作]`, Evan Shelhamer `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于强化学习的可变稀疏注意力框架，动态选择视觉Transformer中的有效令牌，以减少计算量并保持模型性能。

**💡 创新点**

创新点在于将注意力稀疏化视为决策问题，利用RL学习策略在训练过程中逐步提升稀疏比例（82–92%），从而实现高效推理；同时通过逐步扩大的网格（6×6 → 18×18 → 37×37）实现分阶段细化。

**🔧 技术方法**

核心技术包括策略梯度强化学习（REINFORCE/Actor-Critic）、稀疏编码/Masking、卷积预处理及自适应奖励设计（结合准确率与稀疏度）。

**📊 数据集**

实验主要基于ImageNet‑1k图像分类数据集，使用DINOv2预训练模型作为基线。

**📈 对比分析**

与完整注意力/ DINOv2 基线比较，本文方法在保持约86% Top‑1 准确率的同时，将激活令牌数量削减至约10%–20%（即稀疏率82–92%），显著降低 FLOPs；但相较于 90% 的基线准确率略有下降。

**⚠️ 局限性**

局限性包括：训练过程需要大量 RL 回合，导致训练时间较长；方法仅在 ImageNet 分类任务上验证，尚未评估到目标检测或分割等下游任务；在极低计算预算下准确率仍显著衰减。

---

## 221. How Developers Discuss Generative AI: A Longitudinal Study of the Visual Studio Code Community

**arXiv ID:** 2609.04680 | [PDF](https://arxiv.org/pdf/2609.04680v1)

**作者:** Panida Rumriankit `[一作]` (Kasetsart University), Arnon Rungsawang `[通讯]` (Chitralada Technology Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 2021-2026 年间 VS Code GitHub 仓库的 AI 相关 issue 进行长周期话题分析，识别并追踪开发者讨论的主题与演变。

**💡 创新点**

首次对主流 OSS 社区中生成式 AI 的讨论进行纵向研究，揭示讨论以操作与工作流为主而非概念风险，并展示从代码补全到对话式与代理式 AI 的演进。

**🔧 技术方法**

结合关键词检索与语义相关性过滤构建语料，使用 BERTopic 与 LDA 进行主题建模，采用 Mann–Kendall 趋势检验及多指标评估（C_v、C_NPMI、主题多样性）验证主题质量。

**📊 数据集**

Microsoft/vscode 仓库 2021‑01 至 2026‑06 共 43,806 条候选 issue，经过语义过滤后 25,227 条 AI 相关 issue。

**📈 对比分析**

通过比较 BERTopic 与优化 LDA 的主题质量，BERTopic 在 C_v 与 C_NPMI 上优于 LDA；主题一致性通过 Cohen κ=0.73 评估，趋势分析显示 84/122 主题显著上升，证明模型鲁棒性。

**⚠️ 局限性**

依赖单一注解者，语义过滤精度受限导致部分相关 issue 可能被丢弃；数据来源仅为 VS Code，缺乏跨项目普适性；Search API 结果上限及关键词噪声影响召回。

---

## 222. Latent-Aligned Reasoning for Multimodal Recommendation

**arXiv ID:** 2609.04645 | [PDF](https://arxiv.org/pdf/2609.04645v1)

**作者:** Jiarui Jin `[一作]` (Xiaohongshu Inc), Anyang Ji `[通讯]` (Nanjing University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多模态推荐任务中，作者提出了 LARK，一种两阶段潜在推理框架，利用视觉-语言模型在离线阶段生成高质量的多模态项目嵌入，并通过双重对齐机制解决跨模态信号衰减问题。

**💡 创新点**

创新点包括：① 在第一阶段将可学习的潜在 token 与多步 Chain‑of‑Thought 推理交错，并与冻结的视觉编码器对齐；② 在第二阶段通过桥接 MLP 投影潜在表示，并与第一阶段的 CoT 隐状态对齐；③ 结合视觉对齐、推理‑文本对齐与 item‑to‑item 对比学习，显著缓解跨模态衰减，并提升推荐效果。

**🔧 技术方法**

技术手段：基于 Qwen2.5‑VL‑3B 的视觉‑语言模型；Chain‑of‑Thought 以及潜在推理；视觉对齐损失（与冻结视觉编码器对齐）；推理‑文本对齐损失；InfoNCE 对比学习；Swing 算法挖掘 item‑to‑item 关联。

**📊 数据集**

实验数据集：Amazon Review（Baby、Sports、Clothing）5‑core 过滤版；工业级 In‑House 数据集（百万级用户、数十万商品）。

**📈 对比分析**

与多类基线（传统 CF、LLM‑增强、多模态、NoteLLM‑2 等）在 Recall@10/20、NDCG@10/20 上进行比较。LARK 在所有数据集、所有下游模型（DeepFM、LightGCN、SASRec）上均取得最高或第二最高分，平均提升 10%–15% 以上，尤其在工业数据上显著领先。

**⚠️ 局限性**

局限性：依赖大模型和离线计算，推理模板固定；对用户侧建模缺乏；潜在推理与对齐机制在极端长尾场景下仍有提升空间；需在真实在线系统中进一步验证性能与成本。

---

## 223. Aplaud: Adaptive Personalized Low-Rank Decomposition for User-Specific LLM

**arXiv ID:** 2609.04738 | [PDF](https://arxiv.org/pdf/2609.04738v1)

**作者:** Xinyu Li `[一作]` (Kent State University), Zhi Liu `[通讯]` (iLambda Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对个体级调查问卷回答预测，提出 Aplaud 框架，在大型语言模型上实现轻量化、可扩展的个性化微调。

**💡 创新点**

将全局 LoRA 低秩更新拆分为共享 UΣV^T 子空间和用户特定的低秩修正，再加单阶残差，显著压缩用户参数并提升泛化。

**🔧 技术方法**

采用 LoRA、SVD 分解、低秩嵌套分解（P_u Q_u）、rank‑1 残差以及两阶段训练技术。

**📊 数据集**

使用美国公开调查数据集 Pew ATP、GSS，以及 LAMP Movie‑Tagging 等非调查任务进行验证。

**📈 对比分析**

与多种非个性化 LoRA 变体、GPT‑5 零样本、RAG 检索以及 OPPU 基线比较，Aplaud/Aplaud+ 在 ACC 与 Macro‑F1 上相较于 OPPU 提升约 4–8% 与 4–10%，且参数占比仅为 OPPU 的 2%。

**⚠️ 局限性**

适用于共享任务结构且用户数据稀疏的场景；不适合高风险决策替代真实受访者，需结合人工监督和伦理合规。

---

## 224. HiSfM: Disambiguating Structure-from-Motion via Scaffold-Anchored Hierarchical Reconstruction

**arXiv ID:** 2609.04718 | [PDF](https://arxiv.org/pdf/2609.04718v1)

**作者:** Ziding Zhao `[一作]` (Institute of Automation Chinese Academy of Sciences), Shuhan Shen `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HiSfM，一种层次化粗到细的结构光流重建框架，先将视图图划分为强本地社区，构造多棵无交叉生成树的骨架，仅对全局桥梁进行昂贵的去模糊验证，先重建骨架，再高效地附加剩余图像。

**💡 创新点**

创新点在于：① 只对全局关键桥梁使用高成本的视觉去模糊器；② 通过多棵无交叉生成树（EDST）构造紧凑且鲁棒的骨架；③ 结合粗到细的重建调度，既避免了歧义传播，又显著降低了计算冗余。

**🔧 技术方法**

使用的技术包括：top‑k 社区划分、基于 EDST 的骨架构造、Doppelgangers++ 两视图去模糊、COLMAP 的增量式 SfM 与束调、局部与全局束调整、邻接边选择与多棵生成树打包等。

**📊 数据集**

实验数据集涵盖：Heinly 与 Yan 的视觉歧义基准（小型、结构化集合）以及 1DSfM 与 PhotoTourism 的大规模互联网图像集合（数千张至十万张不等）。

**📈 对比分析**

与 CamTrip（稀疏化）和 DG++（完整去模糊）进行对比。HiSfM 在歧义集上完整率高且比 DG++ 快数十倍；在大规模网络集合上完整率优于 CamTrip，运行时间低于 DG++，实现了鲁棒性与效率的最佳平衡。

**⚠️ 局限性**

局限性包括：① 对树数 K 的选择需要经验，过多树会带来额外时间；② 在极大规模场景中仍需保证桥梁数量可控；③ 对极度重复或低质量图像的去模糊效果可能不足，仍可能出现误匹配。

---

## 225. PLUME: Parameter-Efficient Personalization of Large Language Models via Low-Rank User Modulation in Shared Subspaces

**arXiv ID:** 2609.04715 | [PDF](https://arxiv.org/pdf/2609.04715v1)

**作者:** Xinyu Li `[一作]` (Kent State University), Ruoming Jin `[通讯]` (Kent State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种轻量级个性化LLM微调框架PLUME，能够在保持高度表达能力的同时显著压缩每用户参数量；

**💡 创新点**

创新点在于利用全局任务共享低秩子空间与用户调制器（USM）进行子空间混合，再结合跨层共享子空间（PCLS）与秩-1残差（Resid）实现高效个性化；

**🔧 技术方法**

主要技术包括LoRA、低秩子空间投影、用户条件子空间混合、跨层共享低秩表示与秩-1残差微调；

**📊 数据集**

实验使用LongLaMP与LaMP两大个性化文本生成基准（含抽象生成、产品评论、主题写作、新闻标题、学术标题）；

**📈 对比分析**

与非个性化方法（如LoRA、PiSSA、AdaLoRA、QLoRA）以及个性化对手（OPPU、CoPE、RAG）比较，PLUME在保持或超越其性能的同时，将每用户参数压缩至原始约3–7%，显著提升参数效率；

**⚠️ 局限性**

局限在于对极少数据或极端多样化用户的泛化能力尚待验证，且对大型模型的进一步扩展与硬件适配仍需研究。

---

## 226. Retinal OCTA Phenotyping with LLM Reporting for Alzheimer's Disease

**arXiv ID:** 2609.04689 | [PDF](https://arxiv.org/pdf/2609.04689v1)

**作者:** Progga Paromita Dutta `[一作]` (Columbia University), Md Rafiul Kabir `[通讯]` (Central Michigan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一个可解释的OCTA管线，集成注释感知血管分割、层级血管特征提取、无标签表型分析与基于测量的LLM报告生成。

**💡 创新点**

创新点在于融合多层次血管注释、无监督表型结构发现以及通过LLM生成证据关联、非诊断性报告。

**🔧 技术方法**

使用RoSE分割框架、盒计数分形维数、k‑means聚类、GPT/Gemini/Llama等LLM，以及基于置信度的引用映射。

**📊 数据集**

使用公开的ROSE‑1 OCTA数据集，包含117张3×3 mm²视网膜血管图像，来自39名受试者。

**📈 对比分析**

与传统分割模型（U‑Net、Attention U‑Net、nnU‑Net）无直接比较，分割表现ROC‑AUC 0.916–0.970，Dice 0.695–0.781，聚类得到内部一致的低密度低分形表型。

**⚠️ 局限性**

局限在于缺乏诊断标签、仅有9例测试样本、报告的引用准确性和临床解释性不足，且FAZ提取失效。

---

## 227. Beyond Code Generation: Reliability, Verification, and Cost Economics in the Agentic Software Development Lifecycle

**arXiv ID:** 2609.04681 | [PDF](https://arxiv.org/pdf/2609.04681v1)

**作者:** Happy Bhati `[一作]` `[通讯]` (Northeastern University), Happy Bhati (Northeastern University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对AI编码代理在软件开发生命周期中的可靠性、验证与成本经济学进行系统综述，并提出新的评估框架

**💡 创新点**

提出Agentic SDLC Throughput Paradox、Production‑Qualified Change、Verification Tax与Agentic Autonomy Budget四个概念，首次将生成速度、验证成本与自治预算统一量化

**🔧 技术方法**

综合运用实验数据、行业报告、benchmark审计与成本模型，构建Agentic SDLC Control Plane的理论与实践示例

**📊 数据集**

使用2024‑2026期间的GitHub、Google DORA、Meta TestGen‑LLM、SWE‑chat、SWE‑Marathon等真实工程与研究数据集

**📈 对比分析**

通过对比传统产出指标与新提出的PQC率、Verification Tax等，展示生成速度与发布速率脱节，证明可靠性门控对成本与价值的重要性

**⚠️ 局限性**

缺乏对新概念（PQC、Verification Tax等）的实证验证，受限于行业报告的可比性与模型/环境快速变化导致的通用性问题

---

## 228. Controlling and Assessing Appropriate Persona Use in LLM-based Dialogue Generation

**arXiv ID:** 2609.04676 | [PDF](https://arxiv.org/pdf/2609.04676v1)

**作者:** Jongkyung Shin `[一作]` (Ulsan National Institute of Science and Technology), Chiehyeon Lim `[通讯]` (Pohang University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了LLM在Persona‑based Dialogue Generation（PDG）中出现的“人格过度使用”问题，并提出了两项创新：一种基于自对比的内部表示抑制方法SCONPOS，能够在推理时直接减弱促使模型过度使用人格属性的偏置；一种新评估指标Persona Appropriateness Score（PAS），通过比较对话上下文与回复的主题分布来同时惩罚过度使用和使用不足。

**💡 创新点**

创新点在于：①从内部表示层面而非后处理或生成层面抑制人格过度使用，SCONPOS仅需在prompt编码阶段一次减法即可；②提出的PAS采用Jensen‑Shannon距离对主题分布进行度量，既能捕捉过度使用也能检测使用不足，弥补了传统一致性指标的不足。

**🔧 技术方法**

技术上主要使用自对比表示学习提取“过度使用诱导向量”，在推理时对LLM隐藏状态进行λ倍减法；评估时利用主题分布、KL散度和JS距离计算PAS；实验中还对比了Chain‑of‑Thought、Task Decomposition、Self‑Refine、CAA等ICL与自我对比方法。

**📊 数据集**

使用了英文PersonaChat（短句式人格描述）和MBTI‑S2Conv（详细人格描述）两大对话数据集，模型分别为Llama3.1、Qwen2.5、DeepSeek‑chat。

**📈 对比分析**

在所有模型和数据集上，SCONPOS相较于Baseline ICL方法和CAA显著降低Overuse指标，并在ROUGE‑L和PAS上提升约10‑20%（视数据集而定）。同时SCONPOS的向量提取速度比CAA快约4.5倍，存储开销仅20KB，说明其高效且易部署。

**⚠️ 局限性**

局限性包括：①需要针对不同模型/数据集调优λ和目标层；②实验仅限英文数据，跨语言效果未知；③PAS仅基于当前话题主题，无法捕捉对话整体主题跳转与自然性；④对过度使用诱导向量的解释与其他潜在偏差尚未完全解耦。

---

## 229. Ultra-High Resolution Method for Multipath Within a Co-Delay-Doppler Bin in DFT-P-OCDM

**arXiv ID:** 2609.04669 | [PDF](https://arxiv.org/pdf/2609.04669v1)

**作者:** Mingxuan Han `[一作]` (Xi'an Jiaotong University), Feifei Gao `[通讯]` (Tsinghua University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文在 DFT-P-OCDM 协同感知框架下，提出了两阶段超高分辨率估计（TSUR）方法，用于解决多径共延迟-多普勒 bin 内的路径分辨问题。

**💡 创新点**

创新点在于推导了 DPF 领域下的输入-输出闭式关系，并利用阶段性估计分辨延迟类、物理路径数和分段多普勒偏移，结合 MDL、BIC 与 NMLL‑MLE 实现对共延迟多径的高精度分辨。

**🔧 技术方法**

所用技术包括 DFT-P-OCDM、DFT 前处理、DFP 变换、前向后向空间平滑 (FBSS)、MUSIC、MDL、BIC、NMLL‑MLE、CRLB 计算以及数值优化。

**📊 数据集**

实验使用仿真多径信道模型，依据论文表格给出的参数（如 N=1024、ζ=16、k_c=1 等）进行蒙特卡洛仿真。

**📈 对比分析**

与 OTFS、OCDM、AFDM、ZP-PCTD、OMP 等基线比较，TSUR 在共延迟多径分辨率、分段多普勒估计误差、信道重构 NMSE 以及 QPSK BER 等指标上均优于传统方法，逼近理论 CRLB。

**⚠️ 局限性**

局限性包括仅在 DFT-P-OCDM 波形下验证，未考虑实际硬件、同步误差与多普勒分辨率受限等因素；高复杂度计算对实时部署仍是挑战。

---

## 230. Continual Graph Memory for Adaptive Recommendation under Intent Drift

**arXiv ID:** 2609.04651 | [PDF](https://arxiv.org/pdf/2609.04651v1)

**作者:** Hao Nguyen Ngoc `[一作]` (Phenikaa University), Nguyen Xuan Tung `[通讯]` (Phenikaa University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种持续图记忆框架CGM-Rec，使推荐系统在用户意图漂移时通过可写图记忆实现自适应重排序；

**💡 创新点**

创新点在于将知识图谱视为可写记忆，区分语义图记忆与情节记忆，并通过质量门控的结构化图编辑将反馈转化为可控的关系更新；

**🔧 技术方法**

使用冻结的图编码器、LLM重排序代理、质量门控写策略以及双时序记忆（Semantic Graph Memory与Episodic Lesson Memory）来实现推荐与自适应；

**📊 数据集**

在Bundle、Games、MovieLens ML-1M、ML-100K等四个数据集上进行实验，分别对应短上下文、游戏、电影用户行为和丰富元数据的非会话重排序；

**📈 对比分析**

与传统、深度、跨意图、持续学习及LLM基准进行对比，CGM-Rec在大多数指标上取得最高或接近最高得分，尤其在Bundle（HR@1提升29.58%）和ML-100K（HR@1提升0.0971）表现突出；

**⚠️ 局限性**

局限性包括额外的存储与计算开销、质量门控参数仍未全面探索、实验仅限于采样候选重排序而非全库检索，以及情节到语义的转移可能引入偏见和噪声。

---

## 231. ReaDiT Guidance: Control for Image and Video Generation using Diffusion Transformer Features

**arXiv ID:** 2609.04649 | [PDF](https://arxiv.org/pdf/2609.04649v1)

**作者:** Jay Mahajan `[一作]` (University of Illinois Urbana-Champaign), Svetlana Lazebnik `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用Diffusion Transformer内部特征实现对图像与视频生成的空间（深度、姿态、边缘）和运动控制

**💡 创新点**

只需单一DiT块特征即可进行控制，无需额外适配器，参数量少且能统一处理多种空间目标和运动

**🔧 技术方法**

Readout预测网络（AdaLN、线性投影、多尺度融合）、log基采样、latent优化的引导

**📊 数据集**

PascalVOC（约16K图像）配合DepthAnythingV2、OpenPose、HED生成标注；DAVIS与CoTracker3用于光流；模型基于SD3‑Medium、FLUX、CogVideoX

**📈 对比分析**

与ControlNet、OminiControl、Readout Guidance (RG) 等进行对比，实验显示ReaDiT在深度、姿态、边缘三任务的RMSE/PCK/mAP/ODS等指标均优于对手，参数约53M（相较ControlNet的1487M），可与适配器联合进一步提升性能

**⚠️ 局限性**

推理时需进行latent优化，生成速度显著下降；运动控制对快速运动效果不足；缺乏时间上采样能力

---

## 232. Importance-Aware Low-Rank Distillation of Diffusion Transformers

**arXiv ID:** 2609.04646 | [PDF](https://arxiv.org/pdf/2609.04646v1)

**作者:** Denis Zavadski `[一作]` (Heidelberg University), Carsten Rother `[通讯]` (Heidelberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出了一种基于截断SVD的两阶段低秩压缩方案：首先通过块重要性感知分配秩，在保持原有结构的前提下对投影矩阵进行低秩近似；随后使用模块化知识蒸馏与rectified‑flow训练目标对压缩后的模型进行微调，以保持文本到图像生成的高质量。

**💡 创新点**

创新点在于①揭示Diffusion Transformers对投影矩阵级低秩近似具有渐进性容错性；②设计了块重要性感知的低秩分配策略，能在全局参数预算下智能分配压缩力度；③将模块化蒸馏与rectified‑flow训练结合，进一步提升压缩后模型的生成性能。

**🔧 技术方法**

主要技术包括截断SVD低秩压缩、块重要性感知的压缩策略、模块化知识蒸馏（MKD）和rectified‑flow训练目标；评估使用GenEval、HPSv2、DPG等文本到图像基准。

**📊 数据集**

使用LAION-2B图像与JoyCaption生成的文本作为训练与重要性估计数据；评估基准为公开的GenEval、HPSv2和DPG数据集。

**📈 对比分析**

在统一的推理协议下与EcoDiff、Dense2MoE、Chroma‑HD、HierarchicalPrune等方法对比：在68%参数预算下仅下降0.75%，在57%预算下下降2.6%，显著优于现有压缩技术。

**⚠️ 局限性**

在较高压缩比例（<50%）时出现域漂移和对特定实体记忆丢失，表明压缩后模型对风格多样性和细节记忆的鲁棒性受限；进一步研究如何在更低预算下保持这些特性仍是未来工作方向。

---

## 233. A Cost-Aware Agentic Architecture for NL-to-SQL over Nested Enterprise Schemas, with a New Benchmark

**arXiv ID:** 2609.04641 | [PDF](https://arxiv.org/pdf/2609.04641v1)

**作者:** Yoga Sri Varshan Varadharajan `[一作]` (University of Texas at Austin), Sunil Kumar Pandey `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对企业级嵌套图形数据库的自然语言到SQL（NL2SQL）基准DevRev，并设计了面向该基准的单生成、成本意识的代理架构；

**💡 创新点**

创新点在于：1）引入Semantic Depth Score（SDS）衡量查询分析深度；2）实现了迭代式模式化模式选择、子字段感知检索、结构化错误分类与历史反馈；3）通过单生成路径降低每查询成本；

**🔧 技术方法**

使用的技术包括：大型语言模型（LLM）驱动的工具调用、知识图谱查询、Schema检索、错误分类与修复、确定性检查链、动态作弊表、语义验证；

**📊 数据集**

使用了两组数据集：900条经执行验证的DevRev NL2SQL查询（覆盖嵌套型与链图结构）和Spider 2.0 Snowflake公共数据集；

**📈 对比分析**

通过与APEX‑SQL、FlexSQL、ReFoRCE等基线系统在DevRev上比较，系统达到了91.7%的答案正确率，超越下一名54.6个百分点；在Spider 2.0 Snowflake上表现与领先系统持平；成本方面，系统每正确答案的API费用仅为$0.57，显著低于其他系统；

**⚠️ 局限性**

局限性包括：依赖在线执行API，无法离线运行；评估仅覆盖Snowflake环境，跨方言迁移需验证；DevRev基准由LLM生成并通过执行验证，缺乏完全人工审阅；

---

## 234. SMILE: Bridging Continuous Optimization and Discrete Symbolic Recovery

**arXiv ID:** 2609.04639 | [PDF](https://arxiv.org/pdf/2609.04639v1)

**作者:** Mansooreh Montazerin `[一作]` (University of Southern California), Ajitesh Srivastava `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 SMILE 框架，通过结构分析、连续优化与符号恢复三阶段实现符号回归。

**💡 创新点**

创新点在于将固定可解释激活的神经网络与数据驱动的结构分析相结合，并通过门控稀疏化、结构化剪枝及梯度四舍五入统一连续与离散搜索。

**🔧 技术方法**

使用了固定符号激活网络（Sine、Multiplication、Identity、Logarithm、Exponential）、门控稀疏化、梯度优化、结构化剪枝、参数重拟合与梯度基四舍五入等技术。

**📊 数据集**

在 SRBench 数据集上评估，包含 Feynman 物理方程、Strogatz ODE 任务以及 57 个无真值黑盒回归任务（PMLB）。

**📈 对比分析**

与 15+基线（如 PySR、uDSR、E2E、ParFam 等）对比，SMILE 在噪声最高水平的符号解率最高、训练时间最短、表达式复杂度最低，并始终处于准确性–复杂度 Pareto 前沿。

**⚠️ 局限性**

局限在于受限于浅层网络导致的精度受限，以及假设数据具有可分离的组合结构，可能不适用于所有非结构化关系。

---

## 235. Training Large Language Models for Small-Molecule Design with Synthetic Task Scaling

**arXiv ID:** 2609.04735 | [PDF](https://arxiv.org/pdf/2609.04735v1)

**作者:** Frank Hu `[一作]` (Prescient Design), Colin Grambow `[通讯]` (Prescient Design)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLM在低成本合成任务上进行强化学习后训练，使其能够在高成本的高保真分子优化（SQM）任务中表现出色；并通过分层（Tier‑1, Tier‑2）与课程学习策略提升迁移能力。

**💡 创新点**

1) 设计了基于合成任务的分层训练方案，证明低成本任务可以预训练出通用分子设计原理；2) 采用课程式RLVR将低难度任务逐步过渡到高难度任务，显著提升了模型在昂贵评估任务上的性能；3) 证明一款35B参数的模型在此训练方案下可超越更大、更多资源投入的前沿模型。

**🔧 技术方法**

1) 大规模语言模型（Qwen3.6‑35B‑A3B）作为基础；2) 强化学习框架（PPO/GRPO/IPO）与异步RL；3) RLVR（可验证奖励）用于化学任务；4) 软体工具调用与多轮交互；5) 课程式混合训练策略。

**📊 数据集**

1) Tier‑1合成任务：200,000个随机生成的RDKit/OpenEye属性优化任务；2) Tier‑2合成任务：999个基于PLINDER的蛋白‑配体对，使用Chemgauss4评分；3) 评估集：40个保留的SQM‑20.0（semi‑empirical QM）评分任务；4) 还使用公开模型基准（LLaMA、ChatGPT、Claude 等）作对比。

**📈 对比分析**

与前沿大模型（如Qwen‑35B、LLaMA‑70B、ChatGPT‑4o等）在相同评估任务下比较，衡量指标为相对结合能提升（kcal/mol）、系统覆盖率、交互轮数、约束满足率等。实验表明：① 仅用Tier‑1任务的模型已达到接近前沿水平；② 加入Tier‑2的课程训练进一步提升，部分模型在相对结合能提升和系统覆盖率上超过所有对照模型；③ 训练成本与性能并非严格正相关，低成本策略也能获得高性能。

**⚠️ 局限性**

1) 训练仍需数十小时甚至更久，特别是包含Tier‑2的训练；2) 仅在模拟环境中验证，缺乏实验室实际合成与活性验证；3) 仅针对单分子绑定能优化，未涵盖更广泛的药物属性（如毒性、代谢等）；4) 结果对模型基础（如Qwen3.6‑35B）有一定依赖，未验证在其他大模型上的可迁移性；5) 课程策略设计仍需经验性调参，缺乏统一的理论指导。

---

## 236. FlexPosit: Tunable Fractional Precision for LLM Inference Accelerators

**arXiv ID:** 2609.04724 | [PDF](https://arxiv.org/pdf/2609.04724v1)

**作者:** Yimin Gao `[一作]` (University of Virginia), Mircea Stan `[通讯]` (University of Virginia)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出FlexPosit，一种面向LLM推理的精度可调加速器，结合Posit量化、分布感知的混合精度分配与位串行体系结构；

**💡 创新点**

创新点在于：①通过位串行技术实现精度的分数级调节；②使用分布感知的Posit量化和逐通道的2ⁿ缩放；③采用敏感度引导的混合精度分配，按通道窗口粒度分配精度；④统一的Systolic阵列与全局精度控制单元保证数据流正则；⑤在子5位权重下即可达到近FP16精度，且吞吐量与能耗均优于现有加速器；

**🔧 技术方法**

主要技术包括Posit数值格式、分布感知量化、敏感度剖面、混合精度分配、位串行Systolic阵列、全局精度控制单元(GPCU)、SerialPosit解码器、四路MAC聚合；

**📊 数据集**

使用WikiText‑2评估困惑度，并在MMLU、HellaSwag等下游任务中检验准确性；实验模型覆盖GPT‑2 Large/XL、Phi‑2、OPT‑2.7B、LLaMA‑2‑7B、Mistral‑7B、DeepSeek‑7B、Qwen2.5‑7B/14B；

**📈 对比分析**

与BitMoD（组级4位）和OliVe（通道级4/8位）在等面积、等PPL条件下比较，FlexPosit实现吞吐量提升最高1.8×、能耗降低最高2.0×；在子5位权重下实现近FP16准确率，形成平滑的Pareto前沿；

**⚠️ 局限性**

局限性包括仅对权重进行量化（激活保持FP16/FP8），位串行架构受频率限制，需对每个模型单独做敏感度分析，且未覆盖训练或动态模型切换场景；

---

## 237. Locating and Steering Refusal Beyond Attention

**arXiv ID:** 2609.04721 | [PDF](https://arxiv.org/pdf/2609.04721v1)

**作者:** Preethi Carmel Bosco `[一作]` (Indian Institute of Technology Madras), Gopalakrishnan Srinivasan `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了拒绝机制在 Transformer、SSM、RWKV 与混合架构中的位置与可迁移性，并实现了基于写入点的门控防御；

**💡 创新点**

首次证明拒绝方向在不同架构间仅需刚性旋转对齐即可共享，提出写入点可读写原则并实现跨架构拒绝方向迁移；

**🔧 技术方法**

采用线性探测器与均值差分法提取拒绝方向，利用 Procrustes 旋转对齐空间，实施写入点门控，比较残差子空间投影、CAST 等基线；

**📊 数据集**

使用多种对抗提示集（XSTest、AdvBench、HarmBench 等）以及表面匹配正负提示对，在 Llama‑3.1‑8B、Mistral‑7B、Falcon‑Mamba‑7B、RWKV‑6、Zamba2‑7B 等模型上实验；

**📈 对比分析**

通过 AUROC 评估拒绝方向可解码性，使用攻击成功率与误拒率衡量门控效果；实验显示门控显著降低各架构的注入攻击成功率，并保持低误拒率；

**⚠️ 局限性**

防御依赖检测器，可能被绕过；门控在某些写入点需精细调校；对 persona 攻击需多方向策略；在标签稀缺情形下迁移仍有限制。

---

## 238. Knowing What Not to Answer: Selective Non-Compliance in Vision-Language Models

**arXiv ID:** 2609.04720 | [PDF](https://arxiv.org/pdf/2609.04720v1)

**作者:** Minji Kim `[一作]` (POSTECH), Hyounghun Kim `[通讯]` (POSTECH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 KoNA 基准，用于评估视觉语言模型在单一查询和复合查询中的选择性不合规行为，并通过两阶段 Fine‑Tuning（SFT + GRPO）提升模型对不合规场景的响应。

**💡 创新点**

创新点在于：① 构建单一与复合查询配对的评估框架，涵盖 False Premise、Visual Inaccessibility、Universal Unknown、Task Feasibility、Safety 五大不合规类别；② 通过 KoNA 数据集进行两阶段训练（先 SFT 再 GRPO），实现模型在复合查询中对不合规组件的精准拒绝、纠正或回避。

**🔧 技术方法**

使用技术包括：视觉语言模型（Qwen2.5‑VL、InternVL3、GPT‑5、Gemini‑2.5‑Flash）、GPT‑5‑mini 作为评判器、两阶段训练（SFT + Group Relative Policy Optimization，GRPO）以及基于组件级不合规准确率和事实准确率的多维奖励函数。

**📊 数据集**

使用 KoNA 数据集（共 3,100 张图像，3,900 个单一 QA，3,900 个复合 QA，3,900 个完全可答 QA），图像来源于 MS‑COCO、Open Images V7，问答由 GPT‑5 与 Gemini‑2.5‑Flash 生成后人工筛选。

**📈 对比分析**

在多种开闭源 VLM 上进行对比实验，基线模型在复合查询中的不合规准确率普遍低于单查询；Fine‑Tuned 模型在单/复合查询的选择性不合规准确率提升 20–30%，同时保持回答可答准确率不下降，表现出显著的性能提升。

**⚠️ 局限性**

局限性包括：未覆盖参数规模超过 80B 的大型开源 VLM；仅针对视觉语言任务，未考虑音频、图像生成等其他模态；实验规模受计算资源限制，未能深入探索更大模型的表现。

---

## 239. SQL-Zero: Self-Evolving Text-to-SQL

**arXiv ID:** 2609.04697 | [PDF](https://arxiv.org/pdf/2609.04697v1)

**作者:** Daniel Machado Pedrozo `[一作]` (Instituto de Informatica Universidade Federal de Goias), Telma Woerle de Lima Soares `[通讯]` (Instituto de Informatica Universidade Federal de Goias)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SQL-Zero框架，利用无标注的自然语言与SQL对的自我对弈（challenger- solver）实现Text-to-SQL模型的训练，完全不依赖人工标注数据；

**💡 创新点**

创新点在于将执行结果作为唯一的监督信号，构建了基于GRPO的交替自我对弈循环，并通过模板层重复惩罚防止任务多样性崩溃，实现零标注下的竞争式学习；

**🔧 技术方法**

核心技术包括基于SQL-first的生成与验证、GRPO强化学习、难度奖励函数、模板重复惩罚以及对Schema的全局使用；

**📊 数据集**

实验使用BIRD和Spider两大Text-to-SQL基准数据库的训练集进行自我对弈训练，评估时在BIRD dev、Spider test以及Spider-Syn（词义扰动）上测试；

**📈 对比分析**

与零射击基线和同等规模的人工标注控制相比，SQL-Zero在3B模型上BIRD dev提升6.6分、7B提升7.3分，在3B时在Spider和Spider-Syn上均保持或优于基线，而7B在后续轮次转移性能下降；

**⚠️ 局限性**

局限包括单次实验跑、仅覆盖Qwen2.5-Coder、SQLite和单样本生成、训练和评估数据预算不一致、对多样性与熵衰减未做深入分析、无法证明多轮自我对弈会进一步提升效果。

---

## 240. Predicting Spatiotemporal Mobile Sensing-Based PM2.5 Concentrations Using Low-Rank Adapted Spatially Attentive Graph Neural Network

**arXiv ID:** 2609.04693 | [PDF](https://arxiv.org/pdf/2609.04693v1)

**作者:** Om Chiddarwar `[一作]` (Indian Institute of Information Technology Design and Manufacturing Kurnool), Shriniwas Arkatkar `[通讯]` (Sardar Vallabhbhai National Institute of Technology Surat)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于移动传感的细粒度 PM2.5 预测框架 SA‑GNN‑LoRA，利用沿 Surat 城市干道的高频移动监测数据构建动态图结构，实现短期高分辨率空气质量预测与热点识别。

**💡 创新点**

创新点包括：① 将固定长度分段与 DBSCAN 聚类相结合，形成自适应伪站点并在图中定义节点；② 在图注意网络中加入污染输运系数（风场驱动）和双层 GAT，以捕捉空间异质性；③ 采用卷积+GRU 的时间模块配合线性注意力与 LoRA，提升时序建模效率与泛化；④ 在移动传感场景下首次将可解释性方法（LIME、LRP）评估外部特征的重要性。

**🔧 技术方法**

技术手段主要包括：Graph Attention Network (GAT)、GRU、1D CNN、Linear Attention、Low‑Rank Adaptation (LoRA)、DBSCAN 聚类、滚动统计特征工程、LIME/ LRP 解释。

**📊 数据集**

数据集：收集自 Surat（古吉拉特邦）14 km 高速干道的 53 条骑行轨迹，覆盖约 50,000 条同步记录，包含 PM2.5、温度、湿度、风速/风向、车速、土地利用等。

**📈 对比分析**

与传统序列模型（ANN、RNN、LSTM、GRU）及 GNN 对比，SA‑GNN‑LoRA 在 MAE 4.19 μg/m³、RMSE 6.84 μg/m³、R² 0.95、CSI 95.79% 等指标上显著优于基线；线性注意力+LoRA 方案获得最高 R² 并保持低误报率。

**⚠️ 局限性**

局限性：① 模型依赖高质量移动传感数据，GPS 噪声和数据缺失仍需进一步鲁棒性提升；② 仅在单一路段验证，跨城市/多路段推广需更多数据；③ 解释性分析受特征归一化影响，真实物理因子解释仍待加强。

---

## 241. Enhancing Multimodal Emotion Recognition via Multi-Feature Encoding and Attention-Based Fusion

**arXiv ID:** 2609.04690 | [PDF](https://arxiv.org/pdf/2609.04690v1)

**作者:** Xu Lin `[一作]` (Jilin University), Xinying Wang `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种融合语音和视频特征的多模态情感识别框架，使用多特征音频编码、ResNet50-BiLSTM视觉编码，并通过多头注意力实现特征层融合。

**💡 创新点**

创新点在于：①将Wav2Vec2语义嵌入、MFCC、统计声学特征三者联合提取并通过BiLSTM融合；②使用ResNet50+BiLSTM结合空间-时间特征；③在特征层引入多头交叉注意力实现自适应多模态权重分配；④最后采用SVM分类实现鲁棒性。

**🔧 技术方法**

主要技术包括：PyTorch深度学习框架、Torchaudio、Librosa、预训练Wav2Vec2、ResNet50、双向LSTM、多头注意力机制以及支持向量机。

**📊 数据集**

实验使用MELD和IEMOCAP两个公开多模态情感数据集，分别包含音频、视频和文本标签。

**📈 对比分析**

与传统单模态、基线网络（RNN、CNN、1D CNN-LSTM、AlexNet、GoogleNet、EF-LSTM、BERT、CNN-LSTM）进行对比，MELD上平均准确率93.7%、F1 93.7%；IEMOCAP上准确率90.3%、F1 88.5%，相较基线提升约10%+，并在不平衡类别上表现稳健。

**⚠️ 局限性**

局限性包括：仅利用语音和视频两模态，未加入生理信号；融合方式仍为特征层级，未探索更高级的跨模态Transformer；在极端噪声或缺失模态场景下鲁棒性待验证。

---

## 242. ERPBench: Evaluating LLM Agents for Enterprise Decision-Making Across Competitive Market Ecologies

**arXiv ID:** 2609.04667 | [PDF](https://arxiv.org/pdf/2609.04667v1)

**作者:** Xinran Zhang `[一作]` (Beijing Institute of Technology), Pengfei Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并评估了ERPBench，针对企业决策代理在两种竞争生态（Solo 与 Arena）下的性能，记录执行审计。

**💡 创新点**

首次把竞争市场生态作为可测量变量，对同一模型-问题对在不同生态中的排名进行配对比较，并将执行审计嵌入评价。

**🔧 技术方法**

基于可执行的 ERP 模拟环境，六轮决策周期，利用 LLM 代理的 JSON 结构化行动，记录解析、可行性检验、回滚等日志。

**📊 数据集**

使用一套 100 个预设的 ERP 问题实例（覆盖三类难度），以及六大 LLM 模型族（DeepSeek、Gemini、GPT‑5.5、Qwen、Doubao、Claude）。

**📈 对比分析**

采用终端公司估值、排名、底部排名率、归一化后悔等指标进行配对比较；结果显示 DeepSeek 在 Solo 领先，Gemini 在 Arena 领先，且两生态仅有 21/100 问题排名一致。

**⚠️ 局限性**

受限于单一问题集、六轮周期、固定模型端点、仅两种生态，以及执行审计仅为诊断而非因果解释。

---

## 243. Weather-Conditioned Depth Anything

**arXiv ID:** 2609.04827 | [PDF](https://arxiv.org/pdf/2609.04827v1)

**作者:** Zhaoming Xu `[一作]` (Texas A&M University), Zhengzhong Tu `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Weather-Conditioned Depth Anything（DA-W），通过学习天气风格嵌入并在Depth Anything backbone的解码器侧使用AdaLN-Zero进行调制，实现了在多种恶劣天气下的鲁棒单目深度估计。

**💡 创新点**

创新点在于①引入Style Filter实现天气风格与场景几何的显式分离；②使用对比学习实现真实与合成天气嵌入的域对齐；③在冻结的Depth Anything encoder上仅调节解码器的AdaLN-Zero模块，避免灾难性遗忘并保持清晰天气性能。

**🔧 技术方法**

主要技术包括：Vision Transformer (ViT) 编码器、DPT解码器、AdaLN-Zero参数高效调制、对比损失进行天气嵌入学习、教师蒸馏与对齐损失进行深度自监督。

**📊 数据集**

使用了7个真实恶劣天气数据集（ACDC、RTTS、Snow100K、Muses、RID、RIS、NightCity）以及4个清洁数据集（COCO、MegaDepth、SA-1B、HRWSI），并通过合成算法生成对应的天气畸变对。

**📈 对比分析**

在真实恶劣天气基准（NuScenes-night、RobotCar-night、DrivingStereo）和合成噪声基准（KITTI-C）上，与Depth Anything v2/v3、DepthAnything-AC、MWFormer+DepthAnything、以及各种深度估计模型相比，DA-W在大多数天气条件下实现了平均AbsRel提升3.7%，δ1提升至0.74以上，保持或略优于清晰天气下的性能。

**⚠️ 局限性**

局限性包括：在极端夜间场景（如RobotCar-night）仍受相机曝光、运动模糊等因素影响；目前仅针对单一Depth Anything版本，扩展到更大模型需调节适配头；对多重天气共存的适配效果尚未在所有场景充分验证。

---

## 244. Linguistic Trajectory Encoding for Efficient Long-Horizon Spatial Memory in Embodied Agents

**arXiv ID:** 2609.04802 | [PDF](https://arxiv.org/pdf/2609.04802v1)

**作者:** Tianyidan Xie `[一作]`, Zili Yi `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种能够记录动态物体的语言轨迹编码（LTE），实现了长时域可查询的空间记忆。

**💡 创新点**

创新点在于将运动阶段以自然语言描述与稀疏3D锚点和视觉锚点结合，形成可压缩且可查询的多通道轨迹表示。

**🔧 技术方法**

使用了多模态感知（目标检测、跟踪、深度估计、视觉语言模型、语音识别等）和基于八叉树的空间索引。

**📊 数据集**

主要数据集包括 EgoLife 的多日连续录制（构成 Spatial Memory Benchmark）以及 Ego4D 的自然语言与视觉查询基准。

**📈 对比分析**

相较于 VLM 和结构化记忆基线，LTE 在 SMB 的语义轨迹检索达到 45.3% 及 48.7% 的成功率，在 Ego4D NLQ 任务中零样本 R@1 达到 28.75%，显著提升。

**⚠️ 局限性**

局限性包括依赖完整跟踪、缺少部件级动力学、仅在家庭环境下验证、跟踪错误导致性能下降。

---

## 245. ProtLingo: Efficient Protein Language Modeling via Conditional Memory and Expert Routing

**arXiv ID:** 2609.04793 | [PDF](https://arxiv.org/pdf/2609.04793v1)

**作者:** Mingrui Li `[一作]` (ShanghaiTech University), Jingyi Yu `[通讯]` (ShanghaiTech University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在预训练的ESM2蛋白质语言模型基础上，加入了条件本地记忆和稀疏专家路由两种模块，形成ProtLingo框架。

**💡 创新点**

创新点在于将基于隐式代码的中心化N-gram记忆与稀疏MoE路由相结合，既利用可重用的局部上下文，又实现 residue 依赖的计算，从而在保持大规模预训练知识的同时提升突变敏感预测。

**🔧 技术方法**

主要技术包括：中心化潜在N-gram记忆（LNgram）、稀疏混合专家（MoE upcycling）、持续预训练、路由负载平衡和 z-loss。

**📊 数据集**

使用的数据集包括：UniRef50/90 大规模蛋白序列语料进行预训练；ProteinGym、FLIP 进行突变效应评估；CASP15 进行接触图预测；Swiss-Prot、PROSITE 用于专家与记忆的生物学验证。

**📈 对比分析**

通过与 ESM2-150M、ESM2-650M、ProtBert、CARP、RITA XL、ESM-1b 等基准模型在 ProteinGym Spearman、AUC、FLIP 分数、CASP15 长程接触精度等指标上比较，ProtLingo 在仅 153M 活跃参数下实现与或优于更大模型的性能，展示出优越的参数效率。

**⚠️ 局限性**

局限性在于当前模块设计相对轻量，提升空间有限；在更大规模或更复杂任务上可能需要更高级的路由策略或记忆容量，且对极端变异的泛化仍需进一步验证。

---

## 246. Diffusion Language Models for Mobile Edge Agentic AI: Foundations, Applications, and Challenges

**arXiv ID:** 2609.04778 | [PDF](https://arxiv.org/pdf/2609.04778v1)

**作者:** Chenqi Li `[一作]` (China University of Mining and Technology), Wei Ni `[通讯]` (Edith Cowan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对扩散语言模型（DLM）在移动边缘智能中的适用性进行综述，分析其架构、效率技术、部署模式、通信适配与应用，并提出评估框架与挑战。

**💡 创新点**

以DLM为中心的系统视角，连接模型属性与移动边缘需求，提出可拆分推理、通信感知拆分、能耗与隐私协同的边缘部署方案，并构建结构化评估层次。

**🔧 技术方法**

采用连续/离散扩散架构、掩码+最大似然训练、轻量化Transformer、稀疏MoE路由、量化与蒸馏、早停/自适应 denoising、分布式/流水线推理以及多端协同通信拆分等技术。

**📊 数据集**

利用公开评测集（MMLU、GSM8K、MATH、HumanEval、SHELL）以及边缘交互集（Mobile‑Bench、AndroidWorld、OSWorld）验证模型能力与边缘性能。

**📈 对比分析**

通过对比自回归 LLM、通用 DMs 与 DLM 的推理延迟、KV 缓存占用、NFE、吞吐量、能耗等指标，发现 DLM 在约束任务、并行多标记、可调速率下可与自回归相当甚至更优，且在移动设备上实现亚秒级延迟，但仍需更多步骤或稀疏激活。

**⚠️ 局限性**

仍存在采样稳定性、长序列 KV 缓存兼容、模型规模大导致存储/内存压力、稀疏/分布式拆分实现复杂、隐私泄漏风险以及缺乏统一边缘部署框架与标准化评测的局限。

---

## 247. When Does an Interpretation Count as Established? The Formation, Evaluation, and Responsibility of Interpretation in Generative AI

**arXiv ID:** 2609.04766 | [PDF](https://arxiv.org/pdf/2609.04766v1)

**作者:** Deyu Jing `[一作]` `[通讯]` (Fudan University), Deyu Jing (Fudan University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过概念分析与案例讨论，阐释生成式 AI 输出在社会技术循环中如何获得“已形成”解释的地位，并提出“解释性出现”“评估合同”“立场替代”“延迟闭合”等理论框架，揭示本地评估与公共认可之间的差距。

**💡 创新点**

创新点在于：①将本地评估与社会技术认可分离，提出评估合同与立场替代的机制；②提出延迟闭合的实践要求，强调在认知完成前保持可修正性；③给出五项公共条件，构建可追溯、可修正的解释性评价体系。

**🔧 技术方法**

主要采用概念性方法：文献综述、理论构建与案例分析（如历史报告、数字人文案例），并结合现有评估方法（事实性、来源性、覆盖度、报告逻辑等）进行对比。

**📊 数据集**

未使用实验数据集；论文主要基于已有评估框架、公开报告和学术讨论的文献资料。

**📈 对比分析**

作为概念性论文，未进行实验比较；通过与传统评估范式的对比，说明评估合同与立场替代能更好地保持评估边界、可追溯性与责任明确，提升解释性认可的可靠性。

**⚠️ 局限性**

局限性包括：缺乏实证验证与数据支撑；框架在不同学科、不同文本类型的适用性尚未检验；对多模态、交互式评估缺少细化；聚焦文本输出的公共认可，未深入探讨模型内部可解释性与责任机制。

---

## 248. Resilience Beyond Stationary Client Unavailability: Unlocking Efficient and Unbiased Federated Learning

**arXiv ID:** 2609.04763 | [PDF](https://arxiv.org/pdf/2609.04763v1)

**作者:** Ming Xiang `[一作]` (Northeastern University), Lili Su `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并理论分析了一种面向非平稳、异质客户端不可用的联邦学习算法（FedAPM/Federated Stabilized Agile Weight Re‑Equalization），该算法通过自适应创新回声、全局移动平均插值和隐式 gossip 机制，在不需要先验可用性概率或额外内存的情况下实现了收敛。

**💡 创新点**

创新点主要有：
1) 自适应创新回声（adaptive innovation echoing）让缺失的局部更新得以补偿；
2) 全局移动平均（global moving average）在每轮插值全局模型，抑制动态不可用带来的波动；
3) 隐式 gossip 通过延迟多播实现信息在客户端间的平衡混合；
4) 在非平稳、异质可用性模型下给出了收敛证明，并展示了线性加速（线性速度提升）的特殊情形；
5) 对比传统 FedAvg 与多种先进方法，在多种非平稳场景下表现优异。

**🔧 技术方法**

技术与方法：
- 随机梯度下降（SGD）和局部训练。
- 联邦学习框架下的全局与局部更新混合。
- 信息混合矩阵、谱分析和混合系数 k 的插值。
- 非凸优化的收敛分析。
- 计算和内存开销低，主要额外成本是 O(d) 的模型复制和 k 次插值。

**📊 数据集**

实验使用的公开数据集：SVHN、CIFAR‑10、CINIC‑10，采用 CNN 作为模型，并在不同非平稳、异质可用性模式（周期性、阶梯式、正弦波）下进行测试。

**📈 对比分析**

比较方法：
- 与基线 FedAvg（主动/全量）、gFedAvg、FedKnown、FedAvg over all、F3AST 等算法在相同超参、同一数据集上对比。
- 通过 varying the interpolation coefficient k（k=0、k≈m 等）评估性能。
- 结果表明：在非平稳不可用场景下，FedAPM（k≈m）在收敛速度、最终精度和训练稳定性方面均优于基线，且不需要额外存储梯度。
- 仅在极少数情形下与 FedKnown 的精度差距小于 1%。

**⚠️ 局限性**

限制与未解决问题：
- 仅假设客户端可用性独立，未覆盖相关性或协同缺失情况。
- 需要经验性选择插值系数 k；理论上最佳 k 难以解析。
- 理论收敛分析依赖一系列较强假设（如梯度有界方差、梯度相似性等），对更复杂的分布偏移或噪声鲁棒性尚未充分验证。
- 对于极端稀缺或长时间缺失的客户端，仍可能出现较大偏差。

---

## 249. Controversy and Group Certainty Jointly Shape Everyday Moral Judgments

**arXiv ID:** 2609.04750 | [PDF](https://arxiv.org/pdf/2609.04750v1)

**作者:** Ziyu Chen `[一作]` (Australian National University), Lexing Xie `[通讯]` (Australian National University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Reddit AITA 平台的 135 条日常道德困境进行随机实验，检验社会争议度与群体置信度对个人道德判断弱化或强化的影响。

**💡 创新点**

首次将争议度和群体置信度作为两种独立社会信号，在真实生活道德情境中探究它们的交互作用和异向影响。

**🔧 技术方法**

采用随机对照实验、混合效应逻辑回归、Barnard 精确检验、Benjamini‑Hochberg 校正，并利用大语言模型自动抽取评论置信度。

**📊 数据集**

使用 54,827 条 Reddit AITA 评判汇总得到的 135 条多主题道德困境数据集，包含争议度与各方置信度指标。

**📈 对比分析**

通过精确检验和交叉验证的回归模型比较弱化/强化比例，发现争议度将弱化率从 5.3% 提升至 15.9%，置信度呈异向关联，模型预测性能优于基线且无显著改进。

**⚠️ 局限性**

实验仅基于英文 Reddit 数据和简化的可视化摘要，未检验长期效应、论证信息的影响，也未验证判断是否真正更准确或更合理。

---

## 250. ProLombard: Structured Multi-Scale Modeling for Normal-to-Lombard Speech Conversion

**arXiv ID:** 2609.04828 | [PDF](https://arxiv.org/pdf/2609.04828v1)

**作者:** Hongyang Chen `[一作]` (Wuhan University), Song Lin `[通讯]` (OPPO Mobile Telecommunications Corp.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种多尺度结构化模型ProLombard，用于将普通语音转换为洛曼德式语音，保持内容、说话人身份和语音质量。

**💡 创新点**

创新点在于：①通过对齐说话人编码器（ASE）消除洛曼德泄漏；②将洛曼德内容解耦与注入扩展到语音片段级；③设计VQ-中值模块实现可靠的片段级表示；整体实现多尺度洛曼德效应建模。

**🔧 技术方法**

技术包括：多尺度解码器、对齐说话人编码器、Phoneme-aware De‑Lomb/En‑Lomb块、VQ‑Median下采样与上采样、损失函数（对齐、VQ、KL、重建、对抗、特征匹配）。

**📊 数据集**

使用了中文EMALG和英文Lombard Grid两个公开数据集，分别包含多名说话人在不同噪声水平下的并行普通/洛曼德语音。

**📈 对比分析**

与CycleGAN、StarGAN、PGD‑N2L、DurFlex‑Lomb等基线进行比较，ProLombard在洛曼德相似度、可懂度、说话人保留和语音质量等多项指标上均优于基线，且保持模型规模和算力提升有限。

**⚠️ 局限性**

局限性包括：转换后仍与真实洛曼德语音存在一定差距；对动态噪声环境的控制仍有限；以及对说话人多样性和多语种适用性需进一步验证。

---

## 251. Generating Constructive Feedback on Stories via Reinforcement Learning

**arXiv ID:** 2609.04824 | [PDF](https://arxiv.org/pdf/2609.04824v1)

**作者:** Maja Stahl `[一作]` (Leibniz University Hannover), Henning Wachsmuth `[通讯]` (Leibniz University Hannover)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于强化学习的生成式方法，利用GRPO对大语言模型进行训练，生成具有优先性、特异性和可操作性的故事写作反馈，且不需要真实反馈或改写数据。

**💡 创新点**

创新点在于首次使用多维奖励函数（优先性、特异性、可操作性）与GRPO相结合，直接优化生成的反馈质量，解决传统提示式方法缺乏针对性与行动指导的问题。

**🔧 技术方法**

技术方法包括：GRPO（组相对策略优化）与DrGRPO、ModernBERT奖励模型、QLoRA量化微调、以及多LLM集成评判（Bradley‑Terry聚合）。

**📊 数据集**

实验数据集为StoryFeedback（含人工标注反馈）、Storal和WritingPrompts三大创意故事语料，使用StoryFeedback中的1,920条注释评估奖励模型。

**📈 对比分析**

通过自动评估（奖励分数）、多LLM评判排名和5名人工评测的Likert评分，GRPO模型在优先性、特异性、可操作性上均显著优于基准模型（包括Gemini 2.5 Flash），且在全局排名中取得更低（更好）平均位次。

**⚠️ 局限性**

局限性包括仅在英文创意故事上验证，缺乏对其他语种和写作类型的推广；评估主观性强，奖励模型依赖标注者共识；未显式加入真实性/可信度约束，可能导致信息不准确或过度统一化创意表达。

---

## 252. Strict Modes Everywhere - Bringing Order Into Dynamics of Mechanical Systems by a Potential Compatible With the Geodesic Flow

**arXiv ID:** 2609.04817 | [PDF](https://arxiv.org/pdf/2609.04817v1)

**作者:** Arne Sachtler `[一作]` (Technical University of Munich), Alin Albu-Schäffer `[通讯]` (German Aerospace Center)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种非线性弹性势能，使得系统配置空间被严格非线性模态密集填充，从而每个初始状态可自然产生周期振荡。

**💡 创新点**

通过将势能与系统的Riemann度量匹配，得到无数严格模态；利用神经网络拟合势能并提出模式选择与能量调节控制器，实现低能耗周期轨道追踪。

**🔧 技术方法**

利用Riemann几何、几何流积分、梯度约束优化、神经网络拟合与控制器设计（模式选择器、能量调节器）以及数值仿真。

**📊 数据集**

采用双摆系统（无重力版本）作为案例，生成的地理轨迹（geodesics）作为训练数据；未使用公开数据集。

**📈 对比分析**

通过对比不同能量调节控制器（负阻尼 vs 模式兼容），展示了在摆动与能量补偿阶段的性能差异；仿真表明控制器激励远低于弹性力。

**⚠️ 局限性**

仍需将理论势能实现为可行的弹性元件；对势能误差的敏感性较高，导致能量补偿需要额外激励；实际硬件实现与不完整的模型不确定性尚未解决。

---

## 253. Federated Attack Campaign Detection via Contrastive Encoding of Threat Indicators in Gradient Updates

**arXiv ID:** 2609.04815 | [PDF](https://arxiv.org/pdf/2609.04815v1)

**作者:** Manuel Röder `[一作]` (Technical University of Applied Sciences Würzburg-Schweinfurt), Frank-Michael Schleif `[通讯]` (Technical University of Applied Sciences Würzburg-Schweinfurt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出FedIoC框架，在联邦学习中通过监督对比学习将攻击指示器信息编码进梯度，实现跨组织攻击活动检测。

**💡 创新点**

通过在客户端使用基于指示器匹配的监督对比损失，将IoC信息嵌入梯度方向，使服务器仅需聚类梯度即可恢复攻击活动聚类，避免传输原始指示器。

**🔧 技术方法**

联邦学习、监督对比学习（SupCon）、IoC匹配、梯度聚类（余弦相似度+层次聚类）、两步梯度合成等技术。

**📊 数据集**

CTU‑13与UNSW‑NB15两个网络攻击数据集，采用非IID分布实验。

**📈 对比分析**

与FedAvg、FedProx、SCAFFOLD等基线及其IoC改造比较；在CTU‑13上峰值ARI≈0.97、平均ARI≈0.85，分类宏F1保持不变；在UNSW‑NB15上表现相当或略逊。

**⚠️ 局限性**

现有IoC对比损失对基线无显著提升，难以证明对比正则化独立于标签正则化；对梯度聚类依赖于非IID梯度差异，且对SCAFFOLD等抑制漂移的算法效果不稳定；未验证更丰富指示器或流式指示器的效果。

---

## 254. Inventory-Grounded Policy-Level Optimization for Training-Free AI Search

**arXiv ID:** 2609.04813 | [PDF](https://arxiv.org/pdf/2609.04813v1)

**作者:** Wei Zhou `[一作]` (Huawei Technologies Co., Ltd.), Yi Cao `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种无训练的 Inventory-Grounded Policy-Level Optimization（IGPO）框架，利用库存画像和可解释的政策指导来提升 AI 搜索系统在动态库存环境下的表现。

**💡 创新点**

创新点在于将搜索策略与库存事实解耦：不记忆具体商品，而是学习基于实时库存证据的政策准则；同时结合在线库存画像与离线随机漫游与探索的对比信号，形成完整的无训练优化循环。

**🔧 技术方法**

技术手段包括：库存画像构建（嵌入检索、置信度与类别密度统计）、阶段性政策准则（按场景和检索/选择阶段索引）、在线注入准则、离线随机漫游分组、库存引导探索回路、基于 LLM 的准则生成与验证、以及回放门控评估。

**📊 数据集**

使用内部数据集：约 500 条优化查询、500 条验证查询和 3,000 条后续测试查询；库存快照在实验期间分别为 ℐ_t 与 ℐ_{t+1}，并通过 DeepSeek‑V3 LLM 进行自动评判与准则生成。

**📈 对比分析**

与基线（原始配置、SFT、GRPO、prompt 调优）对比，IGPO 在离线测试中 Candidate Recall@30 提升至 0.847（从 0.691），F1@8 提升至 0.816（从 0.643），假无库存率降至 4.8%（从 30.6%），假匹配率降至 15.2%（从 60.8%）。在 14 天线上 A/B 测试中实现 3.17% 的 CTR 提升（95% CI: 1.9–4.5%）并将审核坏例下降 38.9%。

**⚠️ 局限性**

局限性包括：依赖实时库存证据的完整性与检索质量，库存字段缺失或检索不可达时仍难以区分缺失支持与检索漏失；政策准则需要持续维护以适应库存变动、供应商更改和模式迁移；方法无法弥补系统中不存在的库存或元数据缺口。

---

## 255. How Faithful Is Attribution for Sales Forecasting? A Counterfactual Study

**arXiv ID:** 2609.04797 | [PDF](https://arxiv.org/pdf/2609.04797v1)

**作者:** Glib Kechyn `[一作]` `[通讯]`, Glib Kechyn

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在已有的多系列 WaveNet 销售预测模型上，加入了后置的、架构无关的反事实解释层，用以分解每个预测为可解释的贡献。

**💡 创新点**

创新点在于提出了贡献严格相加且不产生残差的计量方法，并通过删除/插入检验展示其可信度，同时诚实阐明解释的适用范围与局限。

**🔧 技术方法**

使用了基于掩码的反事实贡献估计、WaveNet 多系列结构、SHAP 比例分析、删除/插入可信度评估以及日周期拟合等技术。

**📊 数据集**

实验数据来源于 Corporación Favorita 超市销售预测竞赛数据集，包含 174,685 条商品/门店系列，时间跨度 1,688 天，输入窗口 90 天，预测期 16 天。

**📈 对比分析**

与仅使用销售序列的基线模型对比，加入促销信号后 NWRMSLE 从 0.6137 降至 0.6102，虽然提升有限；核心评价是通过删除/插入测试验证解释的可信度，显著优于随机顺序。

**⚠️ 局限性**

局限包括解释对大多数系列缺乏信息、对周期振幅和节假日的建模不足、解释结果受揭示顺序与基线选择影响、以及仅在单一数据集和指标上验证，未能证明跨域泛化。

---

## 256. Can Activation Steering Capture Multidimensional Authorship Style?

**arXiv ID:** 2609.04792 | [PDF](https://arxiv.org/pdf/2609.04792v1)

**作者:** Hieu Tran `[一作]` (University of Maryland), Marine Carpuat `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在 LLM 激活空间中基于语义对照提示构造多维作者风格向量的无训练方法（Aspect-Aware Activation Steering, A3S），并将其用于作者风格迁移。

**💡 创新点**

创新点在于：① 用戏剧性维度（语气、视角、结构等）对风格进行分解，并通过对比生成得到对应的激活方向；② 在激活空间中采用 PCB‑Merging 解决多维向量冲突；③ 针对每个输入动态搜索最优的 Steering 强度，从而兼顾风格匹配与语义保持。

**🔧 技术方法**

主要技术包括：对比激活驱动（Contrastive Activation Steering）、激活空间聚合（PCB‑Merging）、混合自适应搜索（Hybrid Adaptive Search）以及基于 LLM 的无监督生成提示。

**📊 数据集**

使用三个公开基准：MUD（Reddit 任务）、LaMP（Twitter 任务）和 LongLaMP（长篇文本任务），涵盖短文与长文、不同社交平台。

**📈 对比分析**

与提示式基线、单向激活驱动、以及训练得到的 TinyStyler 进行比较。A3S 在所有三个基准上在风格相似度（LUAR、StyleCAV、StyleDistance）上超过提示式方法，在长文本上也优于 TinyStyler，且在 GPT‑4.1 与人工偏好评估中获胜率最高（MUD 约 61% 人工，约 71% GPT‑4.1；LaMP 约 81%；LongLaMP 约 94%）。

**⚠️ 局限性**

局限性包括：① 依赖生成器（LLM）产生的对照样本，若对照样本质量差会影响向量；② 观察到的“共享主成分”可能部分来源于生成器本身；③ 仅在英语数据上验证，跨语言推广未知；④ 需要多轮生成和每实例的搜索，推理时延高。

---

## 257. Learning-Augmented Algorithms: Guarantees, Construction Mechanisms, and System-Level Implications

**arXiv ID:** 2609.04787 | [PDF](https://arxiv.org/pdf/2609.04787v1)

**作者:** Hailiang Zhao `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了学习增强算法的理论与系统实现，聚焦预测接口、误差度量以及一致性–鲁棒性–平滑性框架；

**💡 创新点**

创新点在于将多领域（在线优化、缓存、学习型数据结构、图问题、机制设计）中的构造机制统一归纳，并系统化了一致性、鲁棒性与平滑性的评估维度；

**🔧 技术方法**

主要采用理论分析与案例研究相结合的技术，构建了预测接口模型、误差测度与构造机制的抽象框架；

**📊 数据集**

未使用具体实验数据集，综述基于已有文献与理论结果；

**📈 对比分析**

通过比较一致性上界、鲁棒性上界与已匹配的渐近依赖，评估理论与实际性能的差距，并给出系统级实验示例；

**⚠️ 局限性**

局限在未充分处理预测成本、反馈机制、组件间的互相影响、语义预测与基准构建等系统层面问题。

---

## 258. CLON: Cue-Calibrated Linguistic Object Onboarding for Zero-Shot 6D Pose Front-Ends

**arXiv ID:** 2609.04784 | [PDF](https://arxiv.org/pdf/2609.04784v1)

**作者:** Seojin Ji `[一作]` (Seoul National University), Hyung-Sin Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练的Cue‑Calibrated Linguistic Object Onboarding前端，利用语言记忆引导SAM 3产生高召回、目标相关的候选区域，并在上线前根据对象集统计自动校准语义、外观和几何线索权重，从而显著提升零样本6D姿态估计的前端性能。

**💡 创新点**

① 用视角一致的短语语言记忆作为高召回的目标提示，抑制无关干扰；② 在上线阶段基于模板统计自动计算对象集的语义/外观/几何权重，实现针对性线索校准；③ 将对象上岗视作前端记忆构建，而非仅生成模板，为后续匹配提供更可靠的先验。

**🔧 技术方法**

SAM 3概念分割+语言提示、DINOv2/3特征、HSV纹理熵评估、语义/外观/几何匹配评分、对象集线索校准、与GigaPose、SAM‑6D、FoundationPose等姿态求解器结合。

**📊 数据集**

七个BOP核心数据集（LM‑O、T‑LESS、TUD‑L、IC‑BIN、ITODD、HB、YCB‑V）以及SenseShift6D机器人抓取实验数据。

**📈 对比分析**

与CNOS、SAM‑6D等基线在检测/分割以及下游姿态估计上进行对比；平均检测AP提升8.1pp、分割AP提升6.2pp，姿态AR提升4.1pp；机器人抓取实验中AP_mean 0.811、AR_mean 0.849，抓取率36%（约两倍于最强基线）。

**⚠️ 局限性**

语言提示可能存在歧义，导致误检；熵校准可能忽略形状相似/对称等视觉细节；未针对极端实时性能做优化，SAM 3多提示和对象数扩大会导致计算开销；缺少视觉示例提示，无法充分利用CAD视图多样性。

---

## 259. Adaptive Context Parallelism for Production LLM Serving

**arXiv ID:** 2609.04774 | [PDF](https://arxiv.org/pdf/2609.04774v1)

**作者:** Jiarui Guo `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种能够在LLM推理时自适应管理上下文并行（CP）度的系统，支持不同CP度的持久化工作单元并在请求层通过队列延迟、缓存感知预填时间和GPU占用成本共同决定请求分配；在集群层通过秒级的拆分/合并动态调整工作单元组合；同时提供全局前缀缓存管理器以在CP度不同的工作单元间复制、回收前缀KV，保持缓存局部性。

**💡 创新点**

创新点在于：①将CP度视为可调度维度并与请求长度、缓存命中和当前负载联合建模，制定统一的“放置成本”；②实现轻量级、无重启的工作单元拆分/合并，可在秒级响应工作负载变化；③构建全局前缀Trie，实现跨CP度的复制与回收，从而最大化缓存复用而不牺牲负载平衡。

**🔧 技术方法**

核心技术包括：上下文并行（CP）实现（zigzag分区等），基于R，P的预填时间模型，GPU时间成本估算；请求调度算法（放置成本公式）；集群级动态重构（拆分/合并操作及版本同步）；全局前缀Trie及其复制/回收策略；在RTP-LLM框架上实现，利用NVLink和RDMA做高速KV迁移。

**📊 数据集**

使用的工作集：公开数据集 L‑Eval（多长上下文任务）与 Mooncake Tool‑Agent（工具调用场景），以及一份匿名生产流量（约20%请求为长请求，35–55%前缀命中率）。实验还涉及 DeepSeek‑V4‑Flash‑FP8 与 Qwen3‑30B‑A3B 两种大模型。

**📈 对比分析**

与基线（Homo、Static、vLLM、SGLang）比较：在所有模型与负载下均实现最低或接近最低的平均与P90 TTFT，并在最高负载时平均TTFT下降最多 28.1%，P90 TTFT下降 55%；令牌加权SLO达成率提升最高可达 13.3个百分点。实验还评估了工作单元重构的秒级切换与前缀缓存复制/回收对性能的贡献。

**⚠️ 局限性**

局限性：①未覆盖所有主流系统（如LoongServe、MOE模型支持有限）；②重构期间仍需 1–5 秒的停机时间，对极短响应窗口不适用；③对前缀Trie的维护和复制策略依赖精细阈值调参，跨环境迁移需重新校准；④系统仅针对预填阶段优化，解码阶段未统一调度；⑤实验聚焦于两种模型与有限的CP度集合，可能对更大/不同架构的模型泛化能力未知。

---

## 260. Persistent Teacher Anchoring for Tool-Using Agents

**arXiv ID:** 2609.04773 | [PDF](https://arxiv.org/pdf/2609.04773v1)

**作者:** Hyun Bin Park `[一作]` (Sogang University), Du-Seong Chang `[通讯]` (Sogang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Persistent Teacher Anchoring（PTA），一种在工具使用阶段进行的前‑RL 对齐方法，通过块级验证和回合级承诺控制何时将助手输出提交到轨迹并执行工具调用，从而避免学生与教师分布漂移导致的错误执行。

**💡 创新点**

创新点在于：①将 proposer‑verifier 思想扩展到工具执行层面；②利用教师固定的验证器，让已验证的状态在学生更新后保持不变，从而实现 persistent lookahead 调度；③在保持大部分学生生成内容的同时，显著提升了对齐质量和后续 RL 的收敛速度。

**🔧 技术方法**

使用的技术包括：on‑policy knowledge distillation、chunk‑level proposer‑verifier（类似 SKD/RSD）、教师验证器（top‑K/概率阈值）、KL 损失仅对已提交的助手位置，持续的 lookahead 任务调度，以及在高吞吐量 rollout 引擎中实现块级生成。

**📊 数据集**

实验数据集包括：检索‑中介推理任务 NQ、PopQA、HotpotQA、Musique；感知‑中介推理任务 VStar、HRBench4K、HRBench8K（DeepEyes）以及对应的搜索和感知工具接口。

**📈 对比分析**

对比方法：Base、Direct RL、OPKD + RL 与 PTA + RL。结果显示 PTA + RL 在宏观 best@4 上比 OPKD + RL 提升约 2.5–2.8 分，在 macro mean@4 与 macro best@4 上也占优；在稀疏 distillation（top‑N 保留质量）上保持更高的教师分布覆盖；在调度层面实现了约 24% 的吞吐率提升。

**⚠️ 局限性**

局限性：仅在检索与感知工具环境中验证；不包含代码执行、数据库访问或多智能体交互；需要教师推理开销；lookahead 调度对模型规模、工具延迟及任务长度的依赖尚未深入；实验仅进行一次训练，缺乏多随机种子的显著性分析。

---

## 261. Dressing in Motion: A Human Motion-Aware Diffusion Policy for Robot-Assisted Dressing

**arXiv ID:** 2609.04759 | [PDF](https://arxiv.org/pdf/2609.04759v1)

**作者:** Haoxiang Sun `[一作]` (Hong Kong Polytechnic University), David Navarro-Alarcon `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种可适应人体手臂运动的机器人穿衣视觉‑运动策略，能在穿衣过程中实时跟踪手臂运动并自适应调整执行轨迹。

**💡 创新点**

创新点包括：①基于交互几何的扩散策略学习穿衣动作；②利用PDE扩散构造的“膈内标量场”实现对手臂轴向分布的无缝采样；③在线点云配准与运动可视化投影，实时补偿手臂运动，实现运动感知的轨迹适应。

**🔧 技术方法**

技术主要有：扩散模型（DDIM）用于学习动作分布；EdgeConv点云编码器提取人衣交互特征；PDE表面扩散构造标量场；GICP点云配准估计局部刚体变换；运动投影公式实现轨迹补偿。

**📊 数据集**

使用了三类数据集：1）Simulation环境 Assistive Gym 中基于 Cloth3D 的三种衣袖；2）真实数据通过RealSense D435i捕获的手臂与衣服点云；3）人工演示数据（180条）用于训练扩散模型。

**📈 对比分析**

与六个基线（DP3、Diff-MPC、DP-image、BC-LSTM、以及无轨迹适配版本）在模拟和真实实验中对比。结果显示，完整方法在穿衣进度、袖子插入成功率、用户自由度和舒适度等指标上均优于基线，平均穿衣比例超过0.95，真实实验成功率约89%。

**⚠️ 局限性**

局限包括：①在衣物与人体接触前手臂运动可能导致袖口对齐误差；②轨迹投影的平滑参数在高频或大幅度运动下难以完全跟踪，导致性能下降。

---

## 262. A Fairness Audit of the Duckworth-Lewis-Stern Method: Format-Specific and Gender-Differential Bias, with an Interpretable Calibration Layer for Cricket Target Revision

**arXiv ID:** 2609.04754 | [PDF](https://arxiv.org/pdf/2609.04754v1)

**作者:** Soumyadeep Roy `[一作]` `[通讯]` (St. Xavier's), Soumyadeep Roy (St. Xavier's)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Cricsheet公开的国际板球球级数据上，对Duckworth–Lewis–Stern（DLS）方法在雨停赛中的目标修正偏差进行大规模经验审计，并基于审计结果提出可解释的校准层DLS-Cal及其性别感知变体。

**💡 创新点**

创新点在于：①首次系统性揭示DLS的状态依赖偏差（跨不同剩余局数和失球数的137跑差距）和性别差异偏差（女子ODI平均+6.13跑）；②提出DLS-Cal轻量级MLP校准方案和性别感知版本，显著降低偏差；③引入Win-Flip Rate作为面向目标修正的公平性度量。

**🔧 技术方法**

使用的技术包括：合成雨停点采样、球级特征提取、LSTM、XGBoost、XGBoost-Plus、CAP‑Net v2、堆叠集成和基于MLP的校准层；采用匹配采样、匹配桶校准、聚类Bootstrap等统计方法评估偏差与公平性。

**📊 数据集**

使用的数据集为Cricsheet公开的8,150场国际板球比赛球级数据（3,095场ODI、5,055场T20I，包含5,708场男子和2,442场女子），时间跨度2002–2026。

**📈 对比分析**

与三种传统DLS变体和五种现代机器学习基线相比，DLS-Cal在ODI的WFR_5最低（83%），性别感知版将女子ODI残差从+6.19跑降至+0.65跑；整体ML模型的RMSE、MAE均优于任何DLS变体，但DLS-Cal在公平性和可解释性上具有优势。

**⚠️ 局限性**

主要局限包括：①合成中断点仅近似真实雨停情形，可能不完全代表实际操作；②使用的DLS表格为公开重建，非ICC官方更新表格；③女子样本量相对不足，性别校准的泛化能力受限；④未纳入天气、场地、门球等可能影响分数的外部特征。

---

## 263. Reinforcement Learning for improving Large Language Models' Catalan text simplification capabilities

**arXiv ID:** 2609.04823 | [PDF](https://arxiv.org/pdf/2609.04823v1)

**作者:** Arnau Ayguadé Domingo `[一作]` (Barcelona Supercomputing Center), Horacio Saggion `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对低资源语言（Catalan）的自动文本简化任务，使用强化学习对大型语言模型进行后训练。

**💡 创新点**

提出基于SARI的奖励函数并结合Group Relative Policy Optimization（GRPO），加入复制与长度惩罚，实现针对特定简化风格的优化。

**🔧 技术方法**

采用GRPO强化学习算法、SARI度量、BLEURT过滤、LLM后训练及提示工程。

**📊 数据集**

使用iDEM（Catalan 304 句）和ASSET（英语 15,563 句，翻译成Catalan/Spanish）数据集。

**📈 对比分析**

在Catalan ASSET测试集和iDEM基准上评估，SARI提升约2–2.7分；英语ASSET训练后对Catalan iDEM产生显著提升，翻译版本未显著。

**⚠️ 局限性**

受限于ASSET数据质量、SARI与人类评判不一致、缺乏Catalan无参考度量，以及仅进行单次训练未充分探索超参数。

---

## 264. A Systematic Comparison of Multilingual Interpretability Methods Reveals Anisotropy-Driven Failures

**arXiv ID:** 2609.04819 | [PDF](https://arxiv.org/pdf/2609.04819v1)

**作者:** Oskar Holmström `[一作]` (Linköping University), Marco Kuhlmann `[通讯]` (Linköping University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对21个多语言解码器模型系统地比较了四种跨语言共享度量（CKA、ANC、GMM占优率、ILO），并关联其与零样本跨语言迁移性能的关系。

**💡 创新点**

首次揭示各度量对表征异向性的不同敏感性，发现仅ILO与迁移性能真正相关，并提出同时报告异向性诊断的实践建议。

**🔧 技术方法**

采用CKA、ANC、基于PCA的GMM占优率、基于kNN的ILO等度量，并通过有效维度、平均随机余弦相似度等诊断评估表征空间；利用零样本迁移任务进行功能验证。

**📊 数据集**

使用FLORES-200并行语料进行度量计算，结合Belebele、XNLI、XCSR、SIB-200和XQuAD五个多语言零样本基准评估迁移性能。

**📈 对比分析**

通过Spearman相关、偏相关、Benjamini–Hochberg FDR校正和置换检验比较指标与迁移的关联；ILO与迁移最高相关（ρ=0.90，p<0.001），ANC次之（ρ≈0.84），GMM和CKA关联弱或无关联。

**⚠️ 局限性**

样本规模有限（仅5个模型族）、仅覆盖高资源语言、仅研究解码器模型、指标易受表征异向性影响、缺乏因果干预等局限。

---

## 265. CPR-IE:A Compression-Prediction-Resource Intelligence Efficiency Metric

**arXiv ID:** 2609.04809 | [PDF](https://arxiv.org/pdf/2609.04809v1)

**作者:** Xiantao Jiang `[一作]` `[通讯]` (Shanghai Maritime University), Xiantao Jiang (Shanghai Maritime University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种新的多属性效率评估框架CPR-IE，用于在部署约束下比较智能系统的压缩、预测和资源负载三项属性；

**💡 创新点**

其创新点在于将资源负载的比例增量组合律与属性聚合的比例响应律分离，并通过函数方程得到唯一的对数线性指数表示，兼顾了可比性、单位不变性与边界行为；

**🔧 技术方法**

技术上主要运用了函数方程与对数变换的数学推导、逻辑可识别性分析、凸分析与子高斯概率界定等工具；

**📊 数据集**

文章未使用公开数据集，而是以理论推导和数学证明为主进行验证；

**📈 对比分析**

评价方法通过严谨的公理化构造与一系列定理证明实现，无需经验实验，因而在实际性能上尚无实测结果；

**⚠️ 局限性**

局限性包括对资源组合与比例增量的强假设在真实系统中可能不成立，缺乏实证验证，并且对外部安全或硬件约束的直接处理有限。

---

## 266. Whose record is this? Diagnosing and authorizing record use in personalized multimodal models

**arXiv ID:** 2609.04801 | [PDF](https://arxiv.org/pdf/2609.04801v1)

**作者:** Xinyu Mao `[一作]` (University of Electronic Science and Technology of China), Ming Sun `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了记录授权（RecordAuth）框架，构建了RecordAuth-Diag诊断套件，并在多种多模态大语言模型上评估了视觉记忆绑定错误（VMM）与预生成授权策略的效果。

**💡 创新点**

创新点在于：①将授权拆解为存在(P)、边缘有效(E)和答案支持(S)三条判定线，正式化记录授权；②设计了基于卡片级边缘判定的“typed pre‑generation authorization”，显著降低未授权内存泄露；③创建了3,690例匹配干预诊断集，能精确区分记录错误与幻觉；④通过对比全盘序列化与授权筛选，揭示授权对安全与效率的双重提升。

**🔧 技术方法**

主要技术包括：视觉特征提取（SigLIP2冻结编码器）、余弦相似度与阈值化的边缘判定、支持追踪的显式记录追踪器、两路释放（guard/raw）决策网络、HGB（梯度提升树）后置过滤、以及Token化的身份认证。

**📊 数据集**

数据集包括：从LSD与Yo'LLaVA衍生的Synthetic Event Metadata（3,690例）、Davis视频轨道（560例，用于P∧E∧S完整验证）、DreamBooth自由主体集、以及公开的Qwen、CoViP、Phi-3.5-Vision、Gemma-3-4B-IT等模型的接口。

**📈 对比分析**

比较方法：在相同的查询、问题、记录文本和图像集合下，对比全盘上下文、一次性组级决策、以及卡片级Typed Authorization。结果显示，Typed Authorization将Qwen的本地未授权使用率从43.63%降至3.06%，CoViP从44.09%降至4.61%；在Davis上，未授权释放率从26.61%降至0.89%；同时，正面召回从86.26%降至60.90%，表明在安全提升与召回之间存在权衡。

**⚠️ 局限性**

局限性包括：①评估基于冻结模型和固定记录，未能估计自然用户历史中的发生率；②缺乏真实身份认证，仅使用Token验证，无法证明对视觉身份的完整鉴权；③诊断集未覆盖所有模型架构（如专有VLMs、MMPB）；④在全盘序列化下的性能与资源占用高，虽然Typed Authorization降低了输入token，但仍需两路决策，计算成本不小；⑤目标缺失（P缺失）仍是关键边界问题，需要外部观察才能完全消除。

---

## 267. CoMLP: Cooperatively-Gated MLPs for Fine-Grained Cross-Modal Information Fusion in Medical Image Segmentation

**arXiv ID:** 2609.04781 | [PDF](https://arxiv.org/pdf/2609.04781v1)

**作者:** Mingyuan Meng `[一作]` (Shanghai Jiao Tong University), Lei Bi `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

针对医学影像分割中的多模态影像与临床文本信息融合，提出一种基于MLP的CoMLP模块，构建统一的多源融合网络，分别在跨图像模态和图像-文本融合两类任务中实现细粒度交互。

**💡 创新点**

创新点包括：① 通过合作式交叉门控（Cooperative Cross‑Gating）将不同模态信息直接嵌入MLP交互过程，消除对密集注意力的需求；② 在MLP交互中拆分为局部（region）和稀疏（dilated）两支，兼顾局部细节与全局语义；③ 同一交互原语可统一应用于图像-图像和图像-文本融合，提升模型通用性。

**🔧 技术方法**

技术方案包括：MLP‑Mixer / gMLP 样式的全连接交互、合作式交叉门控、区域与稀疏分支、残差通道注意力、BiomedBERT 语言编码、Dice+交叉熵损失、Adam 优化；在编码器使用 HPVL 块，解码器使用 TSG 块。

**📊 数据集**

数据集涵盖 5 组：PET/CT + 临床报告的 OPC 与 NPC；胸部 X‑ray + 文本的 QaTa‑COV19；肺 CT + 文本的 MosMedData+；结肠镜图像 + 文本的 Kvasir‑SEG；所有数据均为公开或竞赛数据，且包含 2D/3D、不同解剖区域与多模态特征。

**📈 对比分析**

与 16 种基线方法（U‑Net、Swin‑Cross、MAdapter、BiomedCLIP 等）进行对比。实验显示 CoMLP 在 OPC（Dice 81.24% / mIoU 69.86%）、NPC（Dice 82.10% / mIoU 68.49%）、QaTa‑COV19（Dice 91.55% / mIoU 84.37%）、MosMedData+（Dice 92.85% / mIoU 87.06%）、Kvasir‑SEG（Dice 88.24% / mIoU 79.54%）等数据集均获得显著提升，并通过统计检验验证显著性。

**⚠️ 局限性**

局限性包括：① 主要验证在 PET/CT 与临床文本场景，其他模态组合（如 CT/MR、mpMRI 等）尚未充分评估；② 区域尺寸 R 需要人工设定，缺乏自适应机制；③ 依赖结构化、完整的文本描述，针对报告风格多样性、噪声和缺失信息的鲁棒性尚待研究；④ 计算成本仍高于轻量化注意力方案，需进一步优化。

---

## 268. Continuous Cognitive Coverage for Autonomous Robots via Event-Dependent Cognitive Treatment and Learning

**arXiv ID:** 2609.04770 | [PDF](https://arxiv.org/pdf/2609.04770v1)

**作者:** Hong Su `[一作]` `[通讯]` (Chengdu University of Information Technology), Hong Su (Chengdu University of Information Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种连续认知覆盖框架，使自主机器人能够对每个认知事件进行适当的事件依赖性认知处理，而不是仅在明确任务要求时进行处理。

**💡 创新点**

创新点在于将认知处理与学习机制结合，允许不同事件根据其状态、上下文和历史进行不同的认知处理，并且能够在事件到达时持续进行认知处理。

**🔧 技术方法**

使用了事件依赖性学习机制和多种认知处理方法，包括描述、记忆、风险预测、规划、诊断和类比等。

**📊 数据集**

实验使用了一个控制的移动服务机器人模拟器，机器人在室内环境中执行交付、检查和路线相关活动，遇到各种对象和条件。

**📈 对比分析**

与现有方法（如任务驱动、固定处理等）相比，提出的方法在结构化处理准确率上达到了96.76%，自动处理事件的比例为93.66%，在突发延迟工作负载下保持了92.64%的认知覆盖率，持续学习的联合准确率为79.53%。

**⚠️ 局限性**

限制在于该框架可能在处理复杂或不确定事件时需要更多的计算资源，且在环境变化时需要重新评估和修正先前的认知处理。

---

## 269. Shadow Queries for Private Retrieval in Vector Databases

**arXiv ID:** 2609.04767 | [PDF](https://arxiv.org/pdf/2609.04767v1)

**作者:** Xinguo Feng `[一作]` (University of Queensland), Guangdong Bai `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种通过生成“影子查询”并将其嵌入替换原文嵌入的防御机制，旨在抵御云端向量数据库中的嵌入反演攻击（EIA）；

**💡 创新点**

核心创新在于将防御思路从直接扰动嵌入转向“语义分解+嵌入解耦”，即先用大型语言模型生成多样化的影子查询，再将这些查询的嵌入作为文档的表示，从而破坏嵌入与原文的直接对应关系；

**🔧 技术方法**

技术细节包括：(1) 采用预训练的生成式语言模型（QwQ-32B）通过精心设计的提示和K‑Means聚类来生成多样化影子查询；(2) 使用同一嵌入模型（GTR‑T5 encoder）将影子查询编码为向量；(3) 在向量数据库中对生成的影子查询向量进行随机分散索引；(4) 对抗性实验使用 state‑of‑the‑art 的 vec2text 反演攻击；

**📊 数据集**

实验数据集来自 BEIR 基准，涵盖科学、金融、开放域问答等多领域 15+ 数据集，使用 100 条随机查询并按 BM25 排序扩充检索库；

**📈 对比分析**

对比方法包括无防御、嵌入噪声（σ=0.01）和秘密缩放；评估指标包括 NDCG@10、MAP@10、Recall/Precision/Accuracy 以及反演恢复指标（BLEU、ROUGE‑1/2/L、METEOR）。实验显示，影子查询防御在保持检索性能（与无防御差异≤0.01）且在 R‑1/ROUGE‑L 等指标上比噪声防御低至少 0.2，恢复率最低至 0.2104；在自适应攻击情境下仍能将恢复率压至 0.2083；

**⚠️ 局限性**

局限性包括：需要一次性离线生成大量影子查询，耗时与算力高；对极长文本（>32 token）仍需截断，可能降低恢复难度但影响业务需求；对多模态或非文本向量数据库尚未验证；

---

## 270. Cost-Aware Hierarchical Multi-Agent Ransomware Detection and Family Attribution

**arXiv ID:** 2609.04820 | [PDF](https://arxiv.org/pdf/2609.04820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 271. Vectorizing Classical Tamil: Representation Learning for Verse-Commentary Pairs

**arXiv ID:** 2609.04755 | [PDF](https://arxiv.org/pdf/2609.04755v1)

**作者:** Amrit Gopinath `[一作]` (Sri Sivasubramaniya Nadar), Sangeetha Sivanesan `[通讯]` (National Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了1,262个古典泰米尔诗句与注释（urai）配对的语料库，训练了LSTM、BiLSTM、Transformer编码器、Siamese配对网络、mBART式编码解码器以及仅解码的语言模型，并通过多种对照实验评估其表示学习、检索、生成与语法偏好。

**💡 创新点**

提出了一种针对小数据古典语言的“先对照后实验”框架，系统阐明了CCA、Token‑F1等指标的解释与控制需求，并通过实验证明模型虽能捕捉文本形式，但难以恢复注释内容。

**🔧 技术方法**

使用了递归网络（LSTM、BiLSTM）与Transformer编码器、Siamese匹配网络、mBART‑style编码‑解码器（含降噪与微调）、仅解码器语言模型；同时采用TF‑IDF检索、CCA/KCCA、t‑SNE可视化、最小对比语法探针等技术进行评估。

**📊 数据集**

数据集由五个古典泰米尔文献（Naaladiyar、Tholkappiyam的Eluttatikaram/Sollathikaram/Porulathikaram、Thirukadukam）通过Project Madurai与Tamil Virtual University下载并手工对齐得到，包含1,262个有效诗句‑注释对。

**📈 对比分析**

评估方法包括：TF‑IDF检索基线（R@1≈35%总体，最高92%来自Thirukadukam）；Token‑F1对照分布（模型0.060低于常数25词串0.108）；CCA对照噪声（高值不可解释）；Siamese对比器与t‑SNE可视化；最小对比语法探针（模型95.5%正确识别原始词序）。整体性能显示模型在词序偏好上表现良好，但在内容生成、检索以及跨源泛化方面表现不佳，Encoder–Decoder在第4个epoch后开始过拟合。

**⚠️ 局限性**

局限性包括：语料量小且分布不均，缺乏对不同来源/注释风格的跨源评估；使用空格分词忽略泰米尔的黏着性与韵律结构；Token‑F1仅在30个保留样本上评估；表示与Siamese实验未在保留对上验证；最小对比探针样本有限、主要由随机行重排构成；未进行专家人工校对与质量评估；模型生成常见注释片段而非针对输入诗句的专属说明。

---

## 272. Beneath the Surface of Chains-of-Thought: A Mechanistic Interpretation of Reasoning Operations in LLMs

**arXiv ID:** 2609.04753 | [PDF](https://arxiv.org/pdf/2609.04753v1)

**作者:** Seogyeong Jeong `[一作]` (Korea Advanced Institute of Science and Technology), Taekyung Kim `[通讯]` (NAVER Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大语言模型隐藏状态中不同链式推理操作的几何结构，验证操作可分离性、层次分布、上下文依赖以及错误执行时的表现。

**💡 创新点**

将推理操作的功能层级与内部表示空间相对应；发现中层隐藏状态可清晰区分不同操作；相同词在不同操作上下文中表现不同；通过注意力遮蔽证明操作表示由前置上下文塑造；错误执行时操作几何仍存在但弱化。

**🔧 技术方法**

使用Polya四阶段框架构建推理操作标签，GPT‑5 自动标注并人工验证；提取隐藏状态并进行 PCA+LDA 线性探测；评估 AUROC/AUPRC；对词汇、位置、数字密度进行对照实验；利用注意力遮蔽干预验证因果关系；跨模型、跨数据集泛化验证。

**📊 数据集**

主要使用 DAPO‑Math‑17K 与 TheoremQA 两大数学推理数据集；实验中也验证了 GPQA‑Diamond、MATH‑500 等数据集。

**📈 对比分析**

通过在 held‑out 隐藏状态上进行一对多分类，使用 AUROC 和 AUPRC 作为指标；与词袋、TF‑IDF、位置基线对照，隐藏状态表现明显更好；中层层数 AUROC 最高，跨模型 AUROC 均超过 0.9；错误样本 AUROC 略低，但仍高于 0.9，表明操作几何稳健。

**⚠️ 局限性**

主要限制：标注依赖 GPT‑5 与有限人类验证；研究范围仅限数学/定理推理任务与少量模型；分析为诊断性研究，未探索直接干预或应用；未检验在其它领域、架构或大规模模型上的通用性。

---

## 273. HaptiNet: Networked Haptic Robots Enable Physical Co-presence in Geographically-Unconstrained Rehabilitation

**arXiv ID:** 2609.04799 | [PDF](https://arxiv.org/pdf/2609.04799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 274. DCFA: Dual-view Causal-inspired Attribution for Failure Reasoning in LLM-based Multi-agent Systems

**arXiv ID:** 2609.04749 | [PDF](https://arxiv.org/pdf/2609.04749v1)

**作者:** Zehao Wang `[一作]` (Tianjin University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种无训练的失败归因框架 DCFA，用于定位大语言模型驱动多智能体系统中的决定性错误。

**💡 创新点**

通过构建因果启发的依赖图实现全局推理，再利用局部反事实启发的优化进行细化，从而突破传统方法的浅层归因和上下文衰退问题。

**🔧 技术方法**

关键技术包括：(1) 结构化偏差检测与因果图构建；(2) 依赖条件评估（时序性、必要性、充分性）；(3) 全局因果推理；(4) 双向局部搜索与反事实评估；(5) LLM 作为推理引擎。

**📊 数据集**

在 Who&When 基准上使用 184 条多智能体执行轨迹（126 算法生成、58 手工编写）进行评估。

**📈 对比分析**

与 All‑At‑Once、Step‑by‑Step、Binary‑Search、A2P、ECHO 等基线比较，DCFA 在步骤级归因准确率上平均提升 12.5%（算法生成）和 4%（手工编写），最大提升达 8.27%。

**⚠️ 局限性**

局限性包括：依赖 LLM 认知能力；在极长或高度分支的轨迹上仍可能出现误判；以及对 LLM 本身幻觉导致的内在推理错误难以精确归因。

---

## 275. MedFlow: Class-Aware Multi-Scale Generation for Medical Time-Series Synthesis

**arXiv ID:** 2609.04804 | [PDF](https://arxiv.org/pdf/2609.04804v1)

**作者:** Yanhao Huang `[一作]` (Shanghai Jiao Tong University), Chunyan Miao `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了 MedFlow，一种基于多尺度流匹配的医学时间序列生成框架，能够在类别不平衡的条件下保留临床信息并实现高效采样。

**💡 创新点**

创新点在于：① 使用残差多尺度向量量化分词器捕获从粗到细的时间尺度特征；② 在流匹配过程中引入类条件引导，实现类意识的生成；③ 设计 Token Marginal Guidance，通过训练统计校正 logits 强化少数类模式；④ 采用并行多尺度 ODE 求解显著提升采样速度。

**🔧 技术方法**

核心技术包括多尺度向量量化分词器、类条件流匹配（flow matching）、Token Marginal Guidance（基于 smoothed token 频率的 logit 调整）、结构化源分布、残差解码以及并行 ODE 采样。

**📊 数据集**

实验使用四个公开医学时间序列数据集：EHR 领域的 MIMIC‑III 与 eICU（分别用于住院死亡率和 ICU 延期预测）以及生理信号领域的 APAVA EEG 与 PTB ECG。

**📈 对比分析**

与 TimeGAN、TimeVAE、TimeVQ‑VAE、Diffusion‑TS、BioDiffusion、TarDiff 等基线在 TSTR 任务中比较，MedFlow 在 AUPRC 平均提升约 5.8%、AUROC 平均提升约 2.1%；分布匹配指标 Context‑FID 降低 88.6%；采样速度比 TarDiff 快 3.8 倍，明显优于扩散模型。

**⚠️ 局限性**

局限性包括：对极度稀有类别或非常长序列的鲁棒性尚未充分验证；需要手工调优多尺度和代码表参数；评估主要基于公开数据集，缺少真实临床环境下的广泛验证。

---

## 276. Intrinsic Temporal Adaptation of CLIP for Partially Relevant Video Retrieval

**arXiv ID:** 2609.04800 | [PDF](https://arxiv.org/pdf/2609.04800v1)

**作者:** Hyun Seok Seong `[一作]` (Sungkyunkwan University), Jae-Pil Heo `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Intrinsic Temporal Adaptation框架，用于部分相关视频检索，结合骨干内部时序适配与Affinity-Weighted梯度传播，实现对视频中时序信息的细粒度建模。

**💡 创新点**

创新点在于：① 在CLIP视觉编码器内部直接加入局部时序注意力（Grouped Temporal Attention），使帧嵌入受邻近帧影响；② 采用Affinity-Weighted Gradient Propagation对弱监督的文本-视频匹配进行多帧软聚合，分散对单帧的监督压力。

**🔧 技术方法**

使用技术包括CLIP ViT-B/32视觉/文本编码器、LoRA参数高效适配、Grouped Temporal Attention、轻量级全局帧注意力、Affinity-Weighted Gradient Propagation。

**📊 数据集**

实验数据集涵盖TVR、ActivityNet Captions、Charades-STA和QVHighlights四大PRVR基准集。

**📈 对比分析**

与ProPy等现有方法相比，SumR提升21.4点、R@K更优，跨数据集泛化更好；同时参数量和视频编码算力均低于基线，性能与效率兼具。

**⚠️ 局限性**

局限性：受CLIP先验偏差影响，弱监督可能强化不准确的帧特征；缺乏帧级标注导致模型难以完全消除这些偏差。

---

## 277. Dynamic Heterogeneous Graph Representation Learning: A Survey

**arXiv ID:** 2609.04779 | [PDF](https://arxiv.org/pdf/2609.04779v1)

**作者:** Huan Liu `[一作]` (Hangzhou Dianzi University), Zhidong Zhao `[通讯]` (Hangzhou Dianzi University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了动态异构图（DHG）表示学习的现状，提出统一的DHG定义并构建了基于算法的分类体系，系统回顾了Embedding、GNN、Transformer和应用导向四大类方法，梳理了各类方法的技术细节与典型代表；

**💡 创新点**

首次将离散时间与连续时间DHG统一定义，并给出面向算法的全新分类体系，明确了各方法在动态粒度与异构建模上的偏置；

**🔧 技术方法**

综合评述了随机游走、增量更新、点过程、关系专属GNN、Meta结构引导、递归GNN、结构/交互/LLM增强Transformer等多种技术手段；

**📊 数据集**

综述中引用的公开DHG数据集包括TGB 2.0、公开的交通、推荐、网络安全等数据集，强调了不同研究对数据构造、时间切分和评估协议的差异；

**📈 对比分析**

文章并未提出新的实验对比，而是总结了现有研究的评估方法与性能表现，指出多数方法在统一基准下往往存在不一致的评价结果，难以直接比较；

**⚠️ 局限性**

主要局限：缺乏统一的基准与评估流程，导致结果不可复现；对大规模Web级DHG的可扩展性不足；缺乏预训练与迁移学习框架，难以实现跨域泛化；解释性与因果推断能力有限。

---

## 278. A Robust Watermark-based Fingerprint Framework for GNNs Ownership Verification

**arXiv ID:** 2609.04772 | [PDF](https://arxiv.org/pdf/2609.04772v1)

**作者:** Han Zhang `[一作]` (Macquarie University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种鲁棒的水印与指纹结合的GNN所有权验证框架（REMARK），通过在原始分布内生成水印图并提取输出差异指纹，验证模型所有权；

**💡 创新点**

创新点包括：1) 在分布内生成水印图，避免OOD导致的性能下降；2) 不依赖水印训练集即可验证；3) 利用决策逻辑差异提升指纹分离度，兼容多层输出；4) 将水印与指纹技术融合，形成整体鲁棒框架；

**🔧 技术方法**

使用的技术包括GNN解释器（XGNN）生成子图、动态聚类与多头注意力生成水印特征、图结构生成、桥接注入、输出差异指纹构造、半监督交叉熵训练的二分类器，以及MMD度量；

**📊 数据集**

实验使用六个公开图数据集：Cora、PubMed、CiteSeer、CS、Computers 以及大规模 ogbn-arxiv，并在四种主流GNN架构（GCN、GAT、GraphSAGE、GIN）上评估；

**📈 对比分析**

与现有指纹方法（Grove、GNNFingers、Canary）对比，REMARK在节点嵌入层几乎100%准确，在logit层平均提升约4–5个百分点，且在模型剪枝与自适应攻击下保持更高且更稳定的验证准确率；

**⚠️ 局限性**

限制点：仍依赖对原始图分布的良好估计，水印生成与注入过程相对复杂；在极端攻击（如彻底删除水印或重构模型）下的鲁棒性未全面评估；

---

## 279. Injected and Leaked: Actively Inducing Side-Channel Leakage Using Electromagnetic Injection and Hardware Nonlinearity

**arXiv ID:** 2609.04785 | [PDF](https://arxiv.org/pdf/2609.04785v1)

**作者:** Haoran Yan `[一作]` (Hong Kong University of Science and Technology), Yan Long `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出利用主动电磁注入激活并放大低频模拟秘密的电磁侧信道泄漏，构建一种新的 Injection‑Induced EM Side Channel（注入诱发 EM 侧信道）模型；

**💡 创新点**

创新点在于将电磁注入与侧信道分析结合，利用硬件非线性将秘密信息调制到注入载波上，实现跨墙、长距离（高达30 m）高保真音频窃听与设备状态监测；

**🔧 技术方法**

使用软件定义无线电（USRP）进行注入与接收，基于非线性模型的频率剖面扫描，采用扩散式去噪模型（SGMSE）进行信号恢复，实验涵盖放大器、ADC、MOSFET、电源转换器等四类常见非线性硬件；

**📊 数据集**

训练去噪模型使用 LibriSpeech 语音库，并通过物理建模合成噪声信号；实验数据来自 11 种市售设备（有线/无线耳机、固话、智能风扇、灯具），并在实验室与真实场景下采集 EM 泄漏波形；

**📈 对比分析**

在 50 cm 处，SNR 可达 30 dB，语音识别错误率低于 5%，对设备状态的 ASR 达 100%；通过频率扫描与注入功率提升，最大有效距离提升至 30 m；与传统被动 EM 侧信道（如 MagEar、Periscope）相比，距离提升 10–15×，且能恢复连续波形而非仅二进制；

**⚠️ 局限性**

局限性包括需具备高功率注入设备与精确频率匹配、对非线性组件的依赖、对低幅值输入（如麦克风）距离受限、在某些硬件或双绞线设计下易被抑制，并且目前的防御方法仍不完善。

---

## 280. Beyond Distance Ordering: Resource Complexity and Universal Optimality of Exact Labeled Directed Shortest Paths

**arXiv ID:** 2609.04825 | [PDF](https://arxiv.org/pdf/2609.04825v1)

**作者:** Bin Cai `[一作]` `[通讯]` (Wuhan University of Technology), Bin Cai (Wuhan University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在只输出带标签的距离向量（而非距离顺序）的单源最短路问题，阐明比较‑加法模型下的加法复杂度与比较复杂度的关系，证明 DAG 上两者对齐，循环图破坏对齐；针对共享枢纽图族 H_k 给出熵紧致的 Pareto 规律；提出“转录‑锥”游戏定义同程序最优基准，并证明存在统一解释器实现最优计数；进一步给出通过主动核约简与现有定向 SSSP 算法相结合的高效统一算法，取得子对数竞争因子。

**💡 创新点**

①首次给出任意拓扑的加法最优值的精确表述；②揭示循环拓扑导致的非矩形可行域及最小实例 H_2 的计算；③在共享枢纽族中建立熵紧致的比较‑加法 Pareto 关系；④设计转录‑锥游戏精确刻画同程序基准；⑤证明存在统一解释器实现与最优专用程序相同的加法/比较总数；⑥提出主动核约简+定向 SSSP 的高效统一实现。

**🔧 技术方法**

比较‑加法模型、线性代数（核/纤维开辟）、多项式时间可搜索的转录‑锥游戏、确定性多选与滚动计算、支配树约简、熵与信息不等式、计算机辅助单元极限证明、以及现有定向 SSSP 算法（Duan 等）等技术。

**📊 数据集**

理论实验：基于图族 H_2（最小非矩形实例）和 H_k（共享枢纽族）进行符号推导与计算机辅助验证；对 DAG 族和一般有向图进行抽象分析；无实测数据集，全部为理论证明。

**📈 对比分析**

与已知下界和上界（如 Dijkstra 的比较下界、Duan 等的 SSSP 算法）进行符号对比；通过计算机验证证明 H_2 的比较下界为 5；给出的统一解释器达到最优计数，但在普通时间上可能指数级；主动核约简算法实现的总计数为 O(√(log(2+)+loglog(4+)))，比最优值低于常数乘因子。

**⚠️ 局限性**

仅覆盖确定性比较‑加法模型，未考虑随机化、减法或标量乘法；对一般循环拓扑的比较复杂度仍未得到完整表述；统一解释器在普通时间上可能指数级，实际效率不高；主动核约简算法虽高效但竞争因子仍为子对数；未讨论位数/编码复杂度和实数权重的实际实现问题。

---

## 281. Recurrence Is Not Enough: Causally Validating Multilingual SAE Translation Features in Gemma 2 and 3

**arXiv ID:** 2609.04808 | [PDF](https://arxiv.org/pdf/2609.04808v1)

**作者:** Giang Son Nguyen `[一作]` (Nanyang Technological University), Dung D. Le `[通讯]` (VinUniversity)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重现并扩展了 Gemma 系列 LLM 的 SAE（稀疏自编码器）翻译启动特征，验证其跨语言（提示、源、目标）以及跨模型（Gemma 2 与 Gemma 3）的因果传递性，并揭示仅有一个特征在 23 种语言设置下具有稳健的正向/负向干预效应。

**💡 创新点**

首次证明特征在多语言环境下的出现频率（recurrence）并不能作为因果转移的可靠指标；并发现 Gemma 2 的 (L10, 5717) 与 Gemma 3 的 (L20, 2456) 两个特征能够作为语言无关的翻译启动开关。

**🔧 技术方法**

使用稀疏自编码器（SAE）对残差流进行稀疏表征分解，三阶段特征发现流程（基于阈值、PCA 一致性过滤、因果干预），并在不同语言设置下进行放大（α=2）与消除（α=0）干预。

**📊 数据集**

采用 WMT24++ 数据集构造 23 组跨语言翻译对（包括英中、日中、阿拉伯语、俄语、越南语等），以及 Gemma Scope 预训练的 16k 宽度 SAE 作为特征空间。

**📈 对比分析**

通过 COMET 评分评估干预效果；在 23 语言配置中，放大 105717/202456 可使 Gemma 2 COMET 提升 1.00–7.46 分、Gemma 3 提升 0.59–2.50 分，消除则在大多数设置中导致 1.04–8.93 分下降；相对基线，Gemma 3 的提升幅度较小但仍显著。相比原始论文仅在英文→X 的单一设置验证，本文展示了跨语言、跨模型的稳健因果证据。

**⚠️ 局限性**

局限性包括：仅评估 Gemma 2 2B 与 Gemma 3 4B 两个模型；特征发现仅涵盖英、中、日三种语言，未能完全区分提示语言与翻译方向的交互作用；未探究多特征联合干预的可能性；评价指标主要为 COMET，未系统评估流利度与适切性；对输出类型的解释基于人工检查而非系统标注。

---

## 282. CoLMIN: LLM-based Multi-Decision Path Negotiation for Cooperative Autonomous Driving

**arXiv ID:** 2609.04807 | [PDF](https://arxiv.org/pdf/2609.04807v1)

**作者:** Zhe Huang `[一作]` (Chang'an University), Min Liu `[通讯]` (Hunan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了CoLMIN框架，利用大语言模型实现多决策路径协商与浅深层反思，实现多车协同自动驾驶的稳定决策共识。

**💡 创新点**

创新点在于将多决策路径协商、评估式评判器、浅层反思和深层反思相结合，突破单决策路径导致的子最优收敛和过度自信问题。

**🔧 技术方法**

采用大语言模型（Qwen2.5-3B/7B）、多轮交互协商、评估式评判器、浅/深层反思机制以及CARLA仿真环境。

**📊 数据集**

使用CARLA仿真平台中V2Xverse生成的十个多车场景，并加入行人和骑车者等动态交通参与者。

**📈 对比分析**

与VAD、TCP、UniAD、LMDriver、CoDriving、CoLMDriver等基线对比，CoLMIN在所有交互场景的驾驶分数最高，尤其在车道变更场景提升7.99%，整体提升1.59%。

**⚠️ 局限性**

仅在仿真环境中验证，缺乏真实道路测试，感知噪声、通信延迟和真实驾驶者行为等鲁棒性待进一步评估。

---

## 283. When Financial Fine-tuning Fails: A Three-Level Detectability Analysis of Numerical Hallucination in Domain-Adapted Language Models

**arXiv ID:** 2609.04806 | [PDF](https://arxiv.org/pdf/2609.04806v1)

**作者:** Xiaodong Li `[一作]` (Guangzhou College of Applied Science and Technology), Peiwei Liu `[通讯]` (Guangzhou College of Applied Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对金融文本摘要中数值幻觉进行系统评估，比较基模型与两种领域微调模型的幻觉表现。

**💡 创新点**

提出三层可检测性分类体系（L1‑L3）揭示“限制缺口”，并证明模板注入是主导幻觉机制。

**🔧 技术方法**

使用 Mistral‑7B‑Instruct 结合 QLoRA 进行高效微调，并配合自动规则检测与人工验证来评估幻觉。

**📊 数据集**

利用 240 条金融摘要样本（160 生成合成、80 真实 10‑K）并按弱/强数值锚定（S0/S1）划分进行实验。

**📈 对比分析**

通过比较 Base、FT‑A、FT‑A+B+C 在 L1‑L3 的幻觉率，发现 Fine‑Tuning 使 Overt 幻觉率从 5.4% 跃升至 82‑90%，数值训练反而提升幻觉；基模型保持近 0%。

**⚠️ 局限性**

检测方法主要依赖规则，L3 可能产生误判；实验仅针对 7B 模型、少量数据、无真实部署环境，缺乏更大模型和跨领域的验证。

---

## 284. On Being Prepared: Automated Vehicle Incident Management Exercise Practices

**arXiv ID:** 2609.04777 | [PDF](https://arxiv.org/pdf/2609.04777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 285. Hierarchical Possession-Aware Graph Pointer Network for Pass Receiver Selection

**arXiv ID:** 2609.04803 | [PDF](https://arxiv.org/pdf/2609.04803v1)

**作者:** Jingyi Wang `[一作]` (Beijing University of Technology), Zhangqin Huang `[通讯]` (Beijing University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于层次化、持有感知图指针网络（HPGPN），用于在StatsBomb 360部分观测条件下预测足球传球的受体

**💡 创新点**

创新点在于将当前传球图、固定事件上下文与动态持有历史双分支联合编码，并采用glimpse指针头对可变大小候选集进行直接选择

**🔧 技术方法**

使用GraphSAGE做图卷积、Transformer做时间上下文编码、跨注意力和门控融合、glimpse指针头进行候选评分

**📊 数据集**

使用StatsBomb公开的360数据集，包括德国德甲、世界杯、欧锦赛、女子世界杯等六个赛事的传球事件和冻结帧

**📈 对比分析**

与最近的三种基线（最近队友、SoccerMap、XGBoost）相比，HPGPN在六个数据集上平均提升约3.3个百分点，最高可达5.5个百分点；在跨赛季迁移和候选集大小实验中也表现稳健

**⚠️ 局限性**

主要限制在于对历史窗口长度的敏感性、对观测缺失球员的鲁棒性，以及模型对不同比赛风格的偏差，在女子赛场上迁移效果相对差弱

---

## 286. An Attention-Guided Global and Local Fusion Framework for Lesion-Focused Image Classification

**arXiv ID:** 2609.04791 | [PDF](https://arxiv.org/pdf/2609.04791v1)

**作者:** Mst Shafia Tasnima `[一作]`, Md Musfique Anwar `[通讯]` (Jahangirnagar University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现一种三分支的注意力驱动全局‑局部融合框架，用于病变/斑点图像分类。

**💡 创新点**

创新点包括：①利用训练后的全局分支生成 Grad‑CAM 热图并阈值化得到掩模，直接将其作为局部分支输入；②在局部分支加入 CBAM 以进一步强调通道和空间特征；③采用自适应加权融合（门控网络）实现样本级别的全局与局部特征动态平衡；④三分支架构相互独立训练，避免全局特征对局部分支的偏置。

**🔧 技术方法**

技术细节：DenseNet‑121 预训练并微调；Grad‑CAM 生成注意力掩模；CBAM（通道+空间注意力）嵌入局部分支；自适应融合使用两层全连接的门控网络产生权重；分类头为两层全连接+Softmax；训练采用三阶段策略（全局→局部→融合），损失为交叉熵。

**📊 数据集**

使用的数据集：①Synthetic Spot Pattern Dataset (SSPD) ；②皮肤病变数据集（MSLD v2.0，4类） ；③番荔枝叶病害数据集（Kaggle，5类） ；④葡萄叶病害数据集（NGLD，4类） ；并在源‑hold‑out 试验中对 MSID 进行评估。

**📈 对比分析**

对比方法：与单独全局分支、单独局部分支以及多种主流 CNN（ResNet‑50、DenseNet‑169、Inception‑V3、Xception）进行基线对比；并与文献中的 ViT、MAE、DINO、Swin Transformer、Transformer‑Xception 等方法进行性能对比。结果显示融合分支在四个数据集上均优于单独分支，准确率最高分别为 97.75%（皮肤）、99.64%（番荔枝叶）和 96.52%（葡萄叶）。在源‑hold‑out 试验中，融合分支亦获得最高 83.83%。

**⚠️ 局限性**

主要限制：①对背景复杂度敏感，背景噪声会降低 Grad‑CAM 掩模质量，进而影响局部分支；②局部分支可能丢失重要的全局上下文信息；③依赖 Grad‑CAM 的定位准确性，若掩模不完整会导致特征缺失；④在跨源或不同设备的图像上，全局分支性能下降，导致整体表现受限。

---

## 287. DODR: Deterministic Operator-Driven Reasoning in Latent Space

**arXiv ID:** 2609.04782 | [PDF](https://arxiv.org/pdf/2609.04782v1)

**作者:** Weicai Huang `[一作]` `[通讯]` (Beijing MQPat Technologies Company), Weicai Huang (Beijing MQPat Technologies Company)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将推理建模为在高维潜在空间中的确定性矩阵运算，形成推理图结构，消除自回归模型的随机采样与错误积累

**💡 创新点**

将推理的三种基本类型（演绎、归纳、溯因）分别对应为可训练的矩阵运算符，证明演绎不可逆、归纳可被“硬否决”、溯因等于演绎伪逆，并提出最小完备运算符集合、结构性零幻觉等九项创新

**🔧 技术方法**

采用低秩矩阵参数化、伪逆求解、Gram–Schmidt正交投影、Banach不动点理论等线性代数工具构建运算符；利用Transformer隐藏层拼接得到超宽快照向量；通过训练冻结的运算符实现跨域推理

**📊 数据集**

使用了由多领域（生物、物理、社会、数学、化学等）共计 420 条独立样本，涵盖演绎、归纳、溯因、组合四类推理，共 503 条样本实例进行评估

**📈 对比分析**

与传统自回归 GPT‑2 基线对比，DODR 在演绎损失下降 1.5×10⁻⁵、演绎准确率 100%、溯因相似度提升 28 倍、幻觉率为 0%，在 218 条跨域样本的端到端实验中演绎 100% 正确、溯因 81.7% 正确，表现出显著性能提升

**⚠️ 局限性**

主要局限在于快照编码的分辨率不足导致溯因判定错误，伪逆残差高达 85.9%，以及对超宽参数规模和内存消耗的依赖；对更复杂或多模态推理任务的可扩展性尚待验证

---

## 288. LUMIN: Lightweight Universal Manufacturing Inspection Network for Anomaly Detection

**arXiv ID:** 2609.04775 | [PDF](https://arxiv.org/pdf/2609.04775v1)

**作者:** Pengfei Yang `[一作]` `[通讯]` (Intelligent Precision Instrument), Pengfei Yang (Intelligent Precision Instrument)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建轻量化制造缺陷检测网络LUMIN，并提出无后端推理的四阶段元数据采样算法PSP

**💡 创新点**

通过极致头部压缩与插件融合实现采样准确率与速度的平衡，PSP在不使用网络前向推理的情况下达到与传统采样相同的P‑AUPR，LUMIN在参数量仅为1/8的条件下保持像素级精度

**🔧 技术方法**

使用CLIP/DINOv3视觉基础模型、插件融合、并行相似度计算与分层像素采样技术

**📊 数据集**

在MVTec‑AD、VisA、BTAD、KSDD、Real‑IAD等公开工业异常检测数据集上进行评测

**📈 对比分析**

与PatchCore、UniADet、VisualAD等基线对比，PSP采样速度提升341×，推理优化后速度提升20×，LUMIN在保持或超过前者的像素级指标的同时，参数量仅为原来的1/8

**⚠️ 局限性**

对非标准光照/背景、极端噪声场景的适应性有限，CLIP版本的LUMIN无法收敛，元数据采样在噪声较大时易失真

---

## 289. Memory-Efficient Designs for Word-Wise Universal Fully Homomorphic Encryption

**arXiv ID:** 2609.04769 | [PDF](https://arxiv.org/pdf/2609.04769v1)

**作者:** Ardhi Wiratama Baskara Yudha `[一作]` (Advanced Micro Devices), Yan Solihin `[通讯]` (University of Central Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了全同态加密（FHE）在 GPU 上的内存瓶颈，提出了一套名为 BXT 的多级优化框架，显著降低 ciphertext 的存储与传输开销；

**💡 创新点**

创新点包括：①基于种子（seed）的 ciphertext 压缩，使得随机矩阵可在执行时即时重构；②将 ciphertext 系数按位打包为 bit‑array 并在 L2→L1 传输时解包，减少无用位占用；③延迟种子生成，将 PRNG 负载从在线阶段推到离线阶段，从而减少矩阵 I/O；④针对通用 FHE（uFHE）中的字节级比较，结合错误感知训练实现 ciphertext 位数裁剪，实现可调精度与性能折衷；

**🔧 技术方法**

使用技术包括：硬件支持的 PRNG 用于种子重构、L2–L1 级位转整数转换器、Tensor‑Core 风格的矩阵片段 Load/Store 指令、故障感知训练（fault‑aware training）、与现有 GPU 加速器（如 GME、TensorFHE）兼容集成；

**📊 数据集**

实验数据集主要为 MNIST，采用逻辑回归（LR）和 CNN（含 ReLU、卷积、全连接层）进行加密推理；

**📈 对比分析**

与基准 100x、TensorFHE、GME 进行对比，BXT-CSO 单独在 CNN 上实现 3.8× 加速（无精度损失 <1%），BXT-CSO50（50% 位裁剪）与 GME 组合可达 9.2× 加速，准确率下降不超过 1%；

**⚠️ 局限性**

局限性包括：优化主要针对内存受限的 uFHE，其他 FHE 方案受益有限；延迟种子生成需依赖特定运算模式；位裁剪对比较精度敏感，需在训练阶段进行错误感知；硬件依赖 PRNG 与位转换器，若 GPU 未提供相应功能则实现成本较高。

---

## 290. SeamFlow: Structure-Aware Flow Matching on Edge Probabilities for Artist-Like UV Unwrapping

**arXiv ID:** 2609.04751 | [PDF](https://arxiv.org/pdf/2609.04751v1)

**作者:** Yuming Zhao `[一作]` (City University of Hong Kong), Junhui Hou `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于连续流匹配的生成式3D表面切割框架SeamFlow，解决传统几何优化与自回归生成方法的投影误差与顺序偏置问题。

**💡 创新点**

创新点包括：①将离散切割任务连续化为边概率空间的流匹配；②引入基于边的结构感知表示以捕捉拓扑信息；③结合全局形状先验实现语义一致性；④消除投影误差与顺序偏置，提升生成多样性与效率。

**🔧 技术方法**

核心技术为连续流匹配（Flow Matching）、基于Transformer的边token自注意力网络、全局点云编码器、Ode求解器以及Gaussian噪声到边概率的学习。

**📊 数据集**

使用Objaverse大规模3D模型数据集，经过严格过滤后得到约350k训练样本和2k测试样本，包含高质量UV切割布局。

**📈 对比分析**

与Blender Smart UV、xatlas、PartUV、Nuvo、FlexPara、OptCuts等传统与神经优化方法相比，SeamFlow在角度失真、面积失真、UV岛数量、运行时间和均值二面角等指标均表现更佳，获得80%以上的用户首选率。

**⚠️ 局限性**

局限性在于对超高分辨率网格时边集过大导致上下文长度瓶颈，导致效率下降；此外对极其复杂拓扑的极限案例仍易出现失败。

---

## 291. A Piecewise-Linear Approximation-based Energy-Efficient Error-Optimized Unsigned Square Rooter for Accuracy-Critical Applications

**arXiv ID:** 2609.04783 | [PDF](https://arxiv.org/pdf/2609.04783v1)

**作者:** Prateek Goyal `[一作]` (Indian Institute of Technology Goa), Sujit Kumar Sahoo `[通讯]` (Indian Institute of Technology Goa)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于分段线性近似的能效优越、误差优化的无符号平方根器（EOSQR），实现了低硬件复杂度和高计算精度。

**💡 创新点**

创新点在于：①用端点匹配法推导出每个区间最优的线性逼近参数，实现误差最小化；②针对奇偶位长度采用可实现的二进制分数逼近常数，全部用移位与加法完成，消除乘法器；③提出Composite Efficiency Metric (CEM) 综合评估精度与能效。

**🔧 技术方法**

主要技术：Verilog‑HDL实现、移位加法电路、领先位检测（LOD）+优先编码器、误差优化的分段线性逼近、CEM指标。

**📊 数据集**

使用的典型数据集：Sobel 边缘检测六幅灰度图（Pirates, Cameraman, Barbara, House, Peppers, Mug），K‑means 颜色量化（Peppers 图像），K‑Nearest Neighbor（MNIST 手写数字）。

**📈 对比分析**

通过与 ERAS、AXSR3、MAHSQR、LESQ-EC、OLSR、TSOSQR 等现有方法比较，EOSQR 在 16 位无符号输入上取得：误差指标 NMED 0.4741e-2、MRED 0.7447e-2、MED 1.2091、EDmax 3；资源节省 61.91% LUT、77.54% 动态功耗、53.11% 延迟；CEM 最高；在实际工作负载中 PSNR/SSIM、分类准确率与精确实现几乎无差距。

**⚠️ 局限性**

局限性：仅针对 2n‑bit 无符号输入，未验证带符号或更大位宽；设计中误差主要集中在区间边界，极端残差时误差仍存在；实现依赖 FPGA 的 LUT 资源，ASIC 端尚未评估；整体结构仍需要领先位检测与优先编码，若输入范围极宽，解码开销可能增加。

---

## 292. TourPhysics: Bringing Physics to World Models for Exploration and Manipulation from a Single Image

**arXiv ID:** 2609.04911 | [PDF](https://arxiv.org/pdf/2609.04911v1)

**作者:** Xin Zhang `[一作]` (Fudan University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一种从单张图像及声明式物理配置初始化的在线世界模型 TourPhysics，支持相机漫游与物体交互并保持物理一致性。

**💡 创新点**

将物理仿真与视频生成分离，先预先计算固定物理与相机轨迹，再进行观察合成并通过质量门控提交，利用几何路由的记忆和永久参考图像实现长时程一致性。

**🔧 技术方法**

使用物理仿真（RBD/MPM/PBD）、Deterministic 视频扩散生成、几何一致性记录、生成器控制的深度分离、参考绑定的记忆回环以及质量门控事务式提交等技术。

**📊 数据集**

基于61个单图像与声明式配置的108条模拟交互序列，涵盖刚体、布料、弹性、颗粒与流体等物体。

**📈 对比分析**

与LingBot-Cam/Act、Wan2.1-I2V、Sora-2、MotionCtrl、Tora-Initiator、minWM-Wan等八种方法在相机轨迹、对象运动、碰撞/变形准确性及视觉一致性等指标上对比；TourPhysics 在相机ATE、方向命中、对象ADE、接触时序误差等方面均领先，表现最佳。

**⚠️ 局限性**

仅在模拟环境中评估，缺乏真实物理验证；单图像初始化导致几何不完整；记忆容量未限制，长时间序列仍可能出现漂移；模型对极端光照/材质变异的鲁棒性有限。

---

## 293. Fast Gauss Sums via Flash Attention

**arXiv ID:** 2609.04910 | [PDF](https://arxiv.org/pdf/2609.04910v1)

**作者:** Nicolaj Rux `[一作]` (Chemnitz University of Technology), Sebastian Neumayer `[通讯]` (Chemnitz University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用现成的 flash attention 机制实现无归一化高斯核求和，并提供两种实现方案

**💡 创新点**

创新点在于仅通过对查询/键/值进行两次轻量级输入扩展，即可将高斯核求和转化为 softmax attention 计算，无需编写自定义 GPU 代码；且可实现完全可微的梯度传播

**🔧 技术方法**

核心技术是对 softmax attention 进行数学重构（包括使用 logits 与不使用 logits 两种方式），配合 PyTorch SDPA 接口和 fp16 计算

**📊 数据集**

使用合成的高斯点云（标准正态分布、尺度为 D⁻¹/²，批量 B、样本数 M=N）进行实验评测

**📈 对比分析**

与 naive 朴素实现和 PyKeOps 进行对比；在 D>8 的情况下，提出的 flash attention 方案在前向和梯度计算上分别提升 2–21 倍和 3–10 倍速度，内存占用线性且低于 PyKeOps，误差保持在 10⁻⁴ 级别

**⚠️ 局限性**

局限性包括仍然是 O(N²) 复杂度；需要 fp16 精度且对 logits 访问有依赖；某些实现受限于 D、C ≤ 256，且必须对维度进行 8 的补齐，导致在低维时性能不如传统方法

---

## 294. Adaptation Interfaces for In-Context Tabular Foundation Models in Time-to-Event Prediction

**arXiv ID:** 2609.04901 | [PDF](https://arxiv.org/pdf/2609.04901v1)

**作者:** Minh-Khoi Pham `[一作]` (ADAPT Centre), Marija Bezbradica `[通讯]` (Dublin City University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究将通用表格基础模型（TabFM）迁移到带有删失的时间事件预测任务中，探讨不同的适配接口（零射击、分类微调、生存头）对模型性能的影响。

**💡 创新点**

创新点在于将TabFM的预训练表示与多种生存分析目标（CoxPH、DeepHit、MTLR）相结合，并系统评估在大规模单风险和小规模多风险数据集上的适配效果，首次揭示了适配接口与数据规模、评估指标的交互关系。

**🔧 技术方法**

技术上使用了三种TabFM（TabPFN、TabDPT、TabICL），通过上下文条件化训练和冻结预训练参数，分别实现零射击、分类微调和生存头适配；生存头包括CoxPH、DeepHit和MTLR；评估指标采用时间相关一致性指数C_td和集成Brier分数IBS。

**📊 数据集**

数据集方面，单风险方面使用74个公开表格生存数据集（覆盖医疗、金融等多领域），多风险方面使用4个公开竞争风险数据集（SUPPORT2-CR、FRAMINGHAM、PBC2、SYNTHETIC）。

**📈 对比分析**

通过5折交叉验证比较，结果显示：在小样本数据集上，零射击TabFM性能最接近；随着样本增大，冻结TabFM加生存头的监督适配（尤其是CoxPH）显著优于零射击和分类微调；在IBS上CoxPH表现最为稳健，DeepHit在C_td上更强；多风险实验中，因子特定MTLR表现最好，但样本量有限。

**⚠️ 局限性**

局限性包括：零射击与监督适配在目标构造和损失上差异，难以单纯归因于损失函数；多风险实验样本量仅4个，结果可能不稳定；未对预训练任务与生存任务的结构匹配进行深入分析；缺少对不同预训练特征维度和PCA降维影响的系统评估。

---

## 295. ReCAST: Restoration-aware Cascaded Stage-wise Training for Obfuscated SMS Risk Classification

**arXiv ID:** 2609.04878 | [PDF](https://arxiv.org/pdf/2609.04878v1)

**作者:** Jieyun Huang `[一作]` (China Unicom), Shiguo Lian `[通讯]` (China Unicom)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ReCAST 框架，分两阶段训练：先用大模型蒸馏的结构化去混淆监督（span、类型、恢复文本）训练学生模型，再对风险分类任务微调，使其能在单通道低延迟推理下对中文 SMS 的混淆做鲁棒判别。

**💡 创新点**

创新点在于：①将去混淆能力作为训练时的中间监督，通过结构化标签实现对混淆信息的显式学习；②采用级联阶段式训练与统一 prompt 接口，保持推理单通道；③通过蒸馏把大模型的去混淆知识迁移到可部署的小模型，避免推理时额外恢复步骤。

**🔧 技术方法**

技术包括：大模型知识蒸馏、结构化监督（span、类型、恢复文本）、级联阶段式训练、统一 prompt 设计、LLM 辅助数据生成、vLLM 部署与性能评测。

**📊 数据集**

数据集为真实生产中文 SMS，30k 训练样本（人机混合生成），500 验证样本，1000 测试样本，覆盖四类风险标签（fraud、gambling、pornography、benign），测试集包含混淆与非混淆样本。

**📈 对比分析**

与 Direct-CLS、Aug-CLS、Pipeline-CLS、Prompt-CLS 以及 RoCBert、CA-CoT 等基线对比，ReCAST 在 ACC 86.6%、Risk Recall 89.5%，比 Direct-CLS 提升约 10 个点，比 Pipeline-CLS 提升约 3 个点；推理延迟与同基模型相近，单通道效率保持。

**⚠️ 局限性**

局限性：仅在中文 SMS 和 9B 参数模型上验证；对其他语言、渠道、风险域或更小模型的效果未知；蒸馏依赖单一教师模型，可能带来偏差；离线评估未覆盖真实生产流量的演化与系统级约束。

---

## 296. When Genomic Masking Priors Fail to Transfer: Strong Variant Prediction, Weak Functional Generation

**arXiv ID:** 2609.04861 | [PDF](https://arxiv.org/pdf/2609.04861v1)

**作者:** Susu Hu `[一作]` (National Center for Tumor Diseases), Julien Vibert `[通讯]` (Gustave Roussy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发并评估了 GenDA——一种双向离散扩散模型，结合熵引导的连续片段遮蔽，用于基因组序列的缺失修复与变异效应预测。

**💡 创新点**

创新点在于把扩散模型的上下文双向性与密度优化片段遮蔽相结合，尝试把训练资源集中在序列组成复杂的区域，并同时验证其在变异预测和零样本功能修复上的效能。

**🔧 技术方法**

技术包括吸收态离散扩散、熵引导连续片段遮蔽、多阶段腐败日程、迭代解码、以及使用 AlphaGenome 进行功能评估。

**📊 数据集**

使用了 4,096 bp 的人类基因组窗口作为预训练数据，ClinVar 高置信度 SNV 数据集进行下游变异预测评估，并在 524,288 bp 的 hg38 上下文中使用 AlphaGenome 进行功能修复评测。

**📈 对比分析**

与同参数规模的自回归 Llama 及 Evo 2 进行 AUROC 对比，GenDA 取得 0.774（vs 0.671），随机片段模型稍优 0.777；在零样本功能修复任务中，所有模型的恢复增益均低于 3-mer 维持对照，甚至为负，表明未能有效保留功能信号。

**⚠️ 局限性**

局限性包括：有限的物理上下文（4,096 bp）与片段上限 300 bp 与评估的 3,500 bp gap 不匹配；熵引导只捕捉序列复杂度而非功能重要性；功能评估对 3-mer 维持控制高度敏感，可能掩盖真实恢复效果；模型在功能生成方面表现不佳，难以转移到更长范围的无监督修复任务。

---

## 297. MMTClinic: Multimodal, Multilingual Time Series Question Answering and Reasoning Benchmark for Clinical Domain

**arXiv ID:** 2609.04842 | [PDF](https://arxiv.org/pdf/2609.04842v1)

**作者:** Sourav Malakar `[一作]` (Institute of Engineering and Management), Priti Singh `[通讯]` (Indian Institute of Technology Patna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了MMTClinic，一套面向临床的多模态、多语言、时序推理问答基准，包含30,000个多语言（英、印、孟、马、塔）问答对，涵盖生存预测、心率预测和SOFA评分三大ICU任务，数据来源于PhysioNet 2012 ICU数据集并通过人工专家评审校正。

**💡 创新点**

创新点在于首次将文本、医学图像与多变量时序信号三种模态联合起来，并提供多语言（包括低资源印地语、孟加拉语等）多选与开放式推理两种问题形式，填补了临床时序推理与多模态、跨语言评测的空白。

**🔧 技术方法**

采用大规模语言模型（LLM）和多模态语言模型（MLLM）在零样本、少样本、链式思考（CoT）三种提示方式下进行评估，共13种模型，涵盖开源与专有、大小不同的模型。

**📊 数据集**

数据集基于PhysioNet/Computing in Cardiology 2012挑战赛的48小时多变量生命体征记录（HR、BP、RR、Temp等），生成多语言文本、对应的CSV数值序列和PNG线图，并经过医疗专家与语言学者审核。

**📈 对比分析**

结果表明：在文本+时序任务中，开源模型DeepSeek-R1、Qwen3等表现最佳；在包含图像的任务中，专有模型GPT‑4.1‑nano和Gemini‑2‑flash表现更佳；不同语言间性能差异明显，模型在印地语、孟加拉语等低资源语言上普遍下滑。总体来看，模型对多模态时序推理的能力仍有限，尤其是推理类任务。

**⚠️ 局限性**

局限性包括：仅覆盖五种印地语族语言，缺乏全球多样性；仅使用单一PhysioNet数据源，可能导致泛化受限；视觉模态仅为时序线图，未包含真实医学影像；机器翻译导致的细微语言偏差；评估仅使用准确率，未衡量置信度、推理深度等安全性指标；推理任务采用离散答案，未评价模型解释质量。

---

## 298. Long Horizon Transformer Quantile Fault Prediction for Multi Site Industrial Predictive Maintenance

**arXiv ID:** 2609.04840 | [PDF](https://arxiv.org/pdf/2609.04840v1)

**作者:** David J Poland `[一作]` (University of Hertfordshire), Na Helian `[通讯]` (University of Hertfordshire)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在多站工业生产环境下，提出并验证了一种基于双阶段量化回归神经网络（QRNN）提取分位数特征并与多流Temporal Fusion Transformer（TFT）融合的长周期（7/14/30天）预测维护框架。

**💡 创新点**

创新点包括：① 双阶段QRNN分位数表征（先估十个分位数再细化四个中间分位数）；② 将分位数状态与动态协变量、静态通道元数据及历史记忆通过门控残差、注意力等机制融合；③ 设计不稳定性感知记忆门，在持续预测误差发散时调节GRU的记忆更新；④ 在机器拆分的多站数据集上开展机器无关的长周期评估。

**🔧 技术方法**

使用技术：双阶段QRNN、分位数回归（pinball loss）、多层感知机压缩、门控残差网络、可变选择、因子化注意力、跨模态注意、动态记忆门、以及基于Transformer的Temporal Fusion Transformer。

**📊 数据集**

数据集：来自72台相同设备族的81通道多速传感器（共720小时/30天文档），覆盖9个制造厂、24个月的运维记录。标签通过操作员日志、PLC故障码和维护记录的两者以上确认规则生成。

**📈 对比分析**

与18种基线（kernel、树、LSTM、Transformer、BERT系列等）进行固定阈值下的跨站评估；在30天时F1=79.97%，召回80.18%，精度81.82%，准确率82.39%，ROC‑AUC 0.820，均领先所有基线，尤其相对RoBERTa的F1提升约3.9%。

**⚠️ 局限性**

局限性：仅在同一设备族、同一组织内的9个工厂中验证，未检验跨站、跨设备或跨行业的迁移；标签可能受未记录故障或记录误差影响；假设传感器校准和数据分布在训练/测试期间保持稳定；不稳定性门需在新场景中重新校准；数据不可公开，限制了复现。

---

## 299. CHAMP: Cross-domain Hybrid Architecture for Matchmaking and Prediction in Online Multi-Player Games

**arXiv ID:** 2609.04870 | [PDF](https://arxiv.org/pdf/2609.04870v1)

**作者:** Kai Wang `[一作]` (Independent Researcher), Yuze Liu `[通讯]` (Swinburne University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套跨模式重匹配系统CHAMP，解决了MOBA游戏匹配中的冷启动、分布不一致和数据稀疏问题。

**💡 创新点**

创新点在于混合跨模式短期序列与长期统计的Hybrid Domain Feature Collection，以及通过DAKE和三个Domain‑Aware Encoder实现单模型跨模式训练的DAWN。

**🔧 技术方法**

使用Transformer‑based Omnidirectional Attention（OwO）作为骨干，加入DAKE、DATOE、DASOE、DAPOE，以及统一的在线部署与阈值调节。

**📊 数据集**

数据集来自流行MOBA游戏的三种模式（Casual、League、Elite），约包含40M、30M和0.1M场比赛。

**📈 对比分析**

与LR、MLP、LSTM、Transformer、OwO等基线相比，DAWN在所有模式下准确率最高（最高0.6773）且RMSE最低，线上A/B测试在5分钟击杀压制率上提升高达20.73%。

**⚠️ 局限性**

局限在于跨模式数据仍需显式域感知，单模式稀疏时仍可能出现欠拟合，对极端技能区间的校准仍有改进空间。

---

## 300. From Deep to Shallow: Unconstrained and Efficient Layer Merging Strategy

**arXiv ID:** 2609.04881 | [PDF](https://arxiv.org/pdf/2609.04881v1)

**作者:** Petro Shulzhenko `[一作]` (LTCI, Telecom Paris, Institut Polytechnique de Paris), Enzo Tartaglione `[通讯]` (University of Turin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对深度网络的激活函数线性化后，提出一种不依赖解析公式、可在任意卷积配置下（含填充、步幅、膨胀等）对相邻线性层进行合并的损失引导学习策略，并避免卷积核尺寸增长。

**💡 创新点**

创新点在于：①引入全局指导的损失（交叉熵、蒸馏、激活映射对齐）使得合并层即使不满足解析可合并的条件也能逼近原模型；②支持任意卷积配置且不必扩大核尺寸；③只训练合并层，保持大部分网络冻结，从而减少训练成本。

**🔧 技术方法**

技术手段包括：激活函数线性化（如EASIER、TLC等），在参考模型与缩减模型间并行前向传播，利用三项损失组合进行微调；采用 SGD 训练仅合并层；对卷积核大小进行可配置化（默认 3×3，或使用深度可分离卷积）。

**📊 数据集**

实验数据集为 CIFAR‑10 与 ImageNet‑1k，架构涵盖 ResNet18/50、MobileNetv2 与 Swin‑T，硬件平台包括 NVIDIA RTX 2080 Ti、Jetson Orin 与 Raspberry Pi 5。

**📈 对比分析**

与传统解析合并方案、全量微调及标准蒸馏等方法比较，Shrunk 模型在保持甚至提升准确率的同时，获得 1.12–1.15× 的推理速度提升（尤其在 CPU/边缘设备上），而解析合并往往因为核尺寸扩大而失去加速效果；在 RTX 2080 Ti 上亦实现数十毫秒的延迟降低。

**⚠️ 局限性**

局限性包括：①需先进行激活线性化，若线性化程度不足可能无法充分压缩；②微调仍需一定的训练时间与超参数调优；③对非卷积或特殊算子（如注意力块）的支持仍有限；④在极大模型或极端硬件上，kernel‑size 约束可能仍导致性能瓶颈。

---

## 301. ARIA - An Agentic Framework for Autonomous Testing of Infotainment Systems

**arXiv ID:** 2609.04913 | [PDF](https://arxiv.org/pdf/2609.04913v1)

**作者:** António Azevedo `[一作]` (Critical Techworks), João Pascoal Faria `[通讯]` (INESC TEC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为ARIA的多代理框架，利用大型语言模型（LLM）自动执行基于Android的汽车信息娱乐系统的端到端测试场景，通过视觉界面交互来验证系统功能。

**💡 创新点**

创新点在于采用多代理架构，将感知推理与实现级交互分离，使得测试过程更为灵活和可靠，同时能够通过自然语言描述直接执行现有的手动测试脚本。

**🔧 技术方法**

使用了大型语言模型（LLM）和多代理架构，具体包括执行者、翻译者、检查者和评估者四个专门代理，分别负责不同的认知子任务。

**📊 数据集**

在一个物理测试环境中评估了30个场景，涵盖了多种系统功能，所有场景均基于Android的汽车信息娱乐系统。

**📈 对比分析**

与单代理基线进行比较，ARIA的多代理架构在首次执行时的假阳性率显著降低（从72.0%降至52.6%），并且在28个完成的场景中，71.4%与真实情况相符，显示出较高的故障检测召回率（100%）。

**⚠️ 局限性**

限制在于模型可能会引用当前屏幕上不存在的元素，或错误识别可见元素的功能，导致一些正确的系统行为被错误分类为错误。此外，执行速度较慢和高的令牌消耗也限制了其在快速开发中的应用。

---

## 302. TreeFI: Value-Aware Statistical Fault Injection for Deep Neural Networks

**arXiv ID:** 2609.04912 | [PDF](https://arxiv.org/pdf/2609.04912v1)

**作者:** Noam Bires `[一作]` (University of Rennes), Elisa Fromont `[通讯]` (University of Rennes)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出TreeFI，一种针对深度神经网络的值感知统计错误注入方法。

**💡 创新点**

通过把层值分区为相似的位翻转效应区间并进行分层采样，显著降低注入次数。

**🔧 技术方法**

使用回归树学习区间，Sigmoid风险评分，分层抽样，软件级单比特FP32错误注入。

**📊 数据集**

CIFAR‑10（ResNet8、RepVGG‑A0）和ImageNet（DeiT‑Tiny/Small/Base）。

**📈 对比分析**

与SFI、IFI及全量注入对比，TreeFI在相同置信度/误差下减少72.1×注入量，平均约44.9×；耗时亦显著缩短。

**⚠️ 局限性**

仍依赖单比特FP32模型，对多比特、多格式或硬件相关错误建模有限。

---

## 303. Better Understanding, Better Fixes? A Study of Hallucination in LLM-based Automated Program Repair

**arXiv ID:** 2609.04909 | [PDF](https://arxiv.org/pdf/2609.04909v1)

**作者:** Xuemeng Cai `[一作]` (Singapore Management University), Lingxiao Jiang `[通讯]` (Singapore Management University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨LLM在自动程序修复中的幻觉现象，提出多层评估框架，考察修复与理解阶段的幻觉

**💡 创新点**

首次将修复幻觉与中间产物理解幻觉分离，并通过触发测试识别、行覆盖预测、额外测试生成三任务进行细粒度分析

**🔧 技术方法**

使用大语言模型（GPT-4/ChatGPT-3.5/CodeLlama等）配合提示工程进行生成任务

**📊 数据集**

Defects4J 832 Bug 数据集

**📈 对比分析**

对三模型在三任务上的自动评估及手工分析，修复成功率仅为21%–55%，理解任务准确率约23%–41%，结果显示理解准确性与修复成功相关但并非决定性

**⚠️ 局限性**

研究仅覆盖Defects4J范围，缺乏更大规模或多领域验证；幻觉分析仍需更细化的自动化工具，且实验受限于模型算力与提示策略

---

## 304. Sound-based Multi-Person 3D Pose Estimation

**arXiv ID:** 2609.04902 | [PDF](https://arxiv.org/pdf/2609.04902v1)

**作者:** Yusuke Oumi `[一作]` (Keio University), Mariko Isogawa `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于主动声学传感的多人人3D姿态估计框架

**💡 创新点**

创新性地引入多尺度声学编码器和时序姿态解码器，专门解决多声源叠加与多体反射问题

**🔧 技术方法**

采用多尺度STFT+log-Mel谱、Transformer自注意力（时序、频率、运动、交互）以及交叉注意力进行特征提取与解码

**📊 数据集**

构建了首个6小时Acoustic Multi-person Pose (AMP) 数据集，包含15名受试者的多人人姿态与声学信号同步记录

**📈 对比分析**

与改编的单人声学模型和WiFi多人人模型做对比，实验显示在MPJPE、PA-MPJPE、PCK等指标上均优于基线，且在不同人数场景下表现稳健

**⚠️ 局限性**

局限于实验室环境的声学反射与设备成本，且模型对极大规模多人或复杂噪声环境的泛化仍有待验证

---

## 305. The Security Feature Location Problem

**arXiv ID:** 2609.04899 | [PDF](https://arxiv.org/pdf/2609.04899v1)

**作者:** Kevin Hermann `[一作]` (Ruhr University Bochum), Adam Shostack `[通讯]` (University of Washington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了安全功能定位问题，定义了安全功能的概念并探讨其定位方法

**💡 创新点**

提出了安全功能定位这一新范畴，并系统列出了研究挑战和方法建议

**🔧 技术方法**

主要采用理论分析、文献综述和案例研究（如Traccar GPS系统）来阐述问题

**📊 数据集**

使用的示例数据集是Traccar开源GPS系统的源代码，约77,000行

**📈 对比分析**

由于论文为概念性研究，未给出实验对比或性能指标

**⚠️ 局限性**

缺乏实证验证与实现，方法尚未在实际项目中评估，难以评估实用性

---

## 306. Reinforcement Learning for Sequential Solar PV Policy Design under Uncertainty: An Agent-Based Approach

**arXiv ID:** 2609.04880 | [PDF](https://arxiv.org/pdf/2609.04880v1)

**作者:** Iias Faiud `[一作]` (University of Galway), Karl Mason `[通讯]` (University of Galway)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究将光伏政策设计表述为马尔可夫决策过程，结合强化学习与随机代理模型，在16年规划期内动态学习财政激励策略。

**💡 创新点**

创新点在于将RL与行为学基础的ABM集成、使用标量化奖励探索采用-成本权衡，并在动态环境中实现自适应政策设计。

**🔧 技术方法**

使用了PPO、SAC、TD3等连续控制RL算法、随机代理模型、标量化奖励以及Monte Carlo不确定性采样等技术。

**📊 数据集**

使用了以爱尔兰乳业农场为基础的历史光伏采用数据、PV成本、电价、财政参数等实证数据集。

**📈 对比分析**

通过将RL学习策略与三种静态基线进行对比，评估累计采用、总成本及每位采用者成本，RL策略在不同权重下构成了稳定的权衡前沿，表现出可比的性能提升。

**⚠️ 局限性**

局限性包括仅覆盖16年单一情景、有限的政策空间、ABM简化行为、未包含部分财务与实施约束，结果仅为模型情景而非最终政策建议。

---

## 307. Near-Field Physical-Layer Authentication Under Impersonation Attacks

**arXiv ID:** 2609.04879 | [PDF](https://arxiv.org/pdf/2609.04879v1)

**作者:** Hajar El Hassani `[一作]` (CY CERGY Paris Université), Arsenia Chorti `[通讯]` (Barkhausen Institut)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了近场物理层身份验证（PLA）中的冒充攻击，提出基于均方误差（MSE）的攻击分析模型，并给出了单天线和多天线攻击者的最优预编码解。

**💡 创新点**

创新点在于：1) 将近场相位差视为同时依赖角度与距离，揭示了与远场相比更严格的攻击条件；2) 在二阶 Fresnel 近似下推导出单天线攻击者需同时匹配角度和距离，及多天线攻击者需满足 Alice 驻波矢量属于其驻波矢量子空间的必要条件；3) 通过仿真验证距离差即可阻止完美冒充。

**🔧 技术方法**

主要技术：二维极坐标模型、近场 steering vector 表达式、二阶 Fresnel 近似、Wirtinger 微分求最优预编码、Moore–Penrose 伪逆求解多天线预编码、仿真蒙特卡洛平均。

**📊 数据集**

无公开数据集，采用模拟环境：频率 2.18 GHz，λ≈0.138 m，Bob 接收阵列 16 个元素、d=λ/2，仿真 10⁴ 次 Monte Carlo。

**📈 对比分析**

比较方法：对比不同攻击者 SNR、距离偏差 Δ、预编码相位等参数下的 MSE；结果显示：1) 当 r_E≠r_A 时 MSE 不趋于零，即使角度相同也无法实现完美冒充；2) 远场距离（≈15.5 m）附近，距离影响减弱，MSE 变平；3) 多天线时，只有当 Eve 的所有天线共位于 Alice 位置时（或其驻波矢量子空间覆盖 Alice）才能实现低 MSE。

**⚠️ 局限性**

局限性：仅在二阶 Fresnel 近似有效的近场范围内；未考虑多径、同步误差、硬件失真等实际因素；仿真仅限二维模型，未扩展到三维或 MIMO 环境。

---

## 308. PRISM-Bench: An Audio-Centric Diagnostic Benchmark for Text-to-Audio-Video Generation

**arXiv ID:** 2609.04867 | [PDF](https://arxiv.org/pdf/2609.04867v1)

**作者:** Yuchen Sun `[一作]` (Shanghai Artificial Intelligence Laboratory), Qi Jia `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了第一份面向文本到音频视频生成（T2AV）的音频中心诊断基准 PRISM‑Bench，采用二维分层（音频类型与声源可见性）对生成结果进行细粒度评估，并提供 900 条人工验证样本及面向多模态 LLM 的评估协议。

**💡 创新点**

创新点：①将音频类型（Speech、Music、Sound）与声源可见性（On‑screen、Off‑screen）交叉分层，形成更细致的诊断维度；②设计 35 条细粒度评估准则，覆盖 Audio‑Visual Coherence、Audio Quality、Audio Expressiveness、Prompt Following；③提出基于 LLM 的 blind side‑by‑side “Judge”协议，校准后与人工评分一致率超过 70%；④公开基准排行榜与评测代码，推动大规模系统比较。

**🔧 技术方法**

技术与方法：多阶段数据标注（Gemini、Qwen3‑Omni、GPT‑5 交叉融合）、PySceneDetect 场景切分、Gemini‑3.1‑pro‑preview LLM 评测、两阶段评分与证据生成、对齐校准与人类对比实验。

**📊 数据集**

数据集：从公开视频与机构授权素材中选取，利用 PySceneDetect 生成 900 条 8 秒以内的音视频片段，人工校验音频类型与声源可见性标签，保证均衡覆盖 Speech、Music、Sound 与 On‑screen、Off‑screen 场景。

**📈 对比分析**

比较方法：在 PRISM‑Bench 上对 Seedance 2.0、Kling v3 Omni、Veo 3.0/3.1、Sora 2、LTX‑2、Ovi、MOVA 等 7 系统进行统一评测。结果显示专有模型在所有维度上显著优于开源模型（如 Seedance 2.0 的 Final 分数最高），但音频可见性一致性（AV Coherence）仍是主要瓶颈，尤其是 On‑screen Music 场景。

**⚠️ 局限性**

局限性：①数据受版权限制，公开样本有限且无法完全复现原始音视频；②评测侧重 900 条样本，难以覆盖所有多样化情境；③LLM 评测虽与人工一致率高，但仍可能受模型偏见与呈现顺序影响；④对复杂多音频场景的细粒度分析仍需进一步完善。

---

## 309. LLM-Assisted Behavioural and Scenario Augmentation for Agent-Based Energy Adoption Models

**arXiv ID:** 2609.04866 | [PDF](https://arxiv.org/pdf/2609.04866v1)

**作者:** Iias Faiud `[一作]` (University of Galway), Karl Mason `[通讯]` (University of Galway)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在已校准的爱尔兰奶农光伏采纳代理模型中，构建了一个混合框架：利用大语言模型（LLM）离线生成可解释的保守/平衡/乐观行为规则和结构化的技术经济情景，然后将这些规则和情景通过规则验证转化为确定性输入，保留原有的逻辑回归采纳机制，并在多重蒙特卡洛仿真中评估其对采纳率和公共成本的影响。

**💡 创新点**

创新点在于：① 将LLM仅作为规范设计工具，而非直接代理；② 通过可解释的行为加权规则（保守、平衡、乐观）实现有限且可解释的行为调节；③ 将情景设置为可验证的结构化输入；④ 在保持模型透明度、可复制性的前提下，将LLM辅助的规则嵌入到已校准的ABM中，从而实现对行为和情景的可控扩展。

**🔧 技术方法**

使用技术包括：大语言模型提示生成（输出JSON行为规则与情景），规则验证与约束，逻辑回归式采纳概率计算，蒙特卡洛模拟（500个世界，5个随机种子）以及对行为多重性、采纳曲线与公共成本的统计比较。

**📊 数据集**

数据集主要来自于爱尔兰奶农光伏采纳的校准数据，包含经济指标、补贴、融资利率、能源价格、技术成本等；实验使用了多种政策组合（补贴、贷款利率、FiT）以及六个结构化情景（能源危机、技术提升、补贴撤销、绿色转型、弱出口激励、金融紧缩）。

**📈 对比分析**

通过对比原始逻辑回归基线与三种行为规则在同一政策和情景下的采纳率与公共成本（均值、标准差、95%置信区间），结果显示行为规则按保守→平衡→乐观递增顺序提升采纳率，最大提升约13%，相应公共成本提升约15%；行为多重性保持有序且无饱和，情景评估表明不同技术经济背景下的采纳和成本差异可解释，整体表现稳定、可复制且符合经济逻辑。

**⚠️ 局限性**

局限性包括：① LLM仅离线使用，未检验实时推理或自适应行为；② 行为规则为结构化近似，缺乏直接的农户决策验证；③ 情景为探索性设定，未进行预测性校验；④ 未与专家或传统专家规则进行直接对比；⑤ 结果受提示语、模型偏差和研究者判断的影响。

---

## 310. MZ-Rain: Moisture-Budget-Guided Zero-Inflated Model for Station-Level Precipitation Nowcasting

**arXiv ID:** 2609.04864 | [PDF](https://arxiv.org/pdf/2609.04864v1)

**作者:** Yifang Zhang `[一作]` (Wuhan University of Technology), Pengfei Duan `[通讯]` (Wuhan University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了站点级降水即刻预报，提出基于湿度预算引导的零膨胀sLSTM框架MZ‑Rain。

**💡 创新点**

创新点：将湿度预算方程拆分为水汽存储、输运与表面蒸发四条物理过程通道，并结合自适应Tweedie目标与辅助降水发生监督处理零膨胀。

**🔧 技术方法**

使用了多分支sLSTM、物理过程门控、Tweedie分布、二元交叉熵等深度学习技术。

**📊 数据集**

使用了2018‑2024年GNSS、IMERG与ERA5结合的RainfallBench六站点数据集。

**📈 对比分析**

与多类基线（sLSTM、TimeFilter、BFPF、ZIDF、ZIP等）比较，在CSI、FAR、MSE、MAE及极端降水评估上平均排名第一，尤其在事件检测与大雨预测上显著提升。

**⚠️ 局限性**

局限性：仍依赖单站观测，未考虑雷达/卫星等多模态信息，且在极端降水样本稀缺时性能提升有限。

---

## 311. Mitigating Performance Discrepancy in Cross-Domain 3D Class-Incremental Learning

**arXiv ID:** 2609.04860 | [PDF](https://arxiv.org/pdf/2609.04860v1)

**作者:** Jinge Ma `[一作]` (Purdue University), Fengqing Zhu `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在跨域3D增量学习中，先提出 Domain3D‑CIL 评估协议并量化性能差异，再设计 PolyMem 方法以显式捕获高阶特征统计，从而减小不同域间的准确率落差。

**💡 创新点**

创新点在于：① 明确定义并量化跨域性能差异；② 用多项式核 Taylor 近似的 TensorSketch 逼近多阶 RBF，形成结构化的高阶统计记忆；③ 在 PTM‑CIL 端实现高阶特征交互而无需显式存储高阶张量。

**🔧 技术方法**

使用技术包括：预训练 Transformer‑基础 3D backbone（Uni3D），闭式岭回归统计记忆，TensorSketch 多阶多项式核逼近，RBF 核的 Taylor 展开，及对特征进行 L2 归一化与多阶 sketch。

**📊 数据集**

实验数据集为 CAD 域点云（ModelNet、ShapeNet）与混合真实/受损域（ScanObjectNN、CO3D、OmniObject3D、ShapeNet‑C），每个任务由 CAD 与单一非 CAD 域混合构成。

**📈 对比分析**

与 10+ PTM‑CIL 基线（L2P、DualPrompt、CODA‑Prompt、SLCA、SimpleCIL、APER、EASE、RanPAC、MOS、TUNA 等）在 Domain3D‑CIL 上比较，PolyMem 在所有域均实现了最小的 LA/CA 差距，D2 的准确率平均提升 ≥6.8% 并保持或略升 D1，表明有效抑制跨域性能差异。

**⚠️ 局限性**

局限性包括：仅在预训练 Transformer 上验证，未探索其他记忆/正则化策略；对阶数 L 与 sketch 维度 M 的选择敏感；高阶特征交互的解释性仍有限，未来需进一步研究不同方法的差异机制。

---

## 312. On Epistemic Diversity in Large Language Models

**arXiv ID:** 2609.04835 | [PDF](https://arxiv.org/pdf/2609.04835v1)

**作者:** Elisabeth Kirsten `[一作]` (UAR Research Center for Trustworthy Data Science and Security), Muhammad Bilal Zafar `[通讯]` (UAR Research Center for Trustworthy Data Science and Security)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并量化LLM的“知识多样性”（epistemic diversity），构建框架并在两类任务（化学家短篇故事与素数无限证明）上系统评估前沿LLM的多样性表现。

**💡 创新点**

创新点在于将哲学与社会认知中的知识多样性概念引入LLM评估，区分可访问多样性与潜在多样性，并提供三种交互协议与答案空间覆盖测量方法。

**🔧 技术方法**

采用答案空间代理、覆盖度度量、温度与采样策略、提示工程和多轮交互协议（可访问、潜在、主动多样性请求）等技术。

**📊 数据集**

使用维基百科列出的化学家名单作为有限答案参考，和已知的多种素数无限证明策略作为无限答案参考，结合对应任务数据集。

**📈 对比分析**

通过对十款前沿LLM在可访问、多样性恢复与主动多样性请求下的覆盖率进行比较，发现可访问多样性极低（约1%），潜在多样性可显著提升，但仍存在显著差距，模型间差异显著。

**⚠️ 局限性**

局限包括答案空间代理不完全、验证与等价判定困难、实验仅覆盖两类任务、模型与接口交互方式复杂、温度与提示改动对多样性提升效果有限。

---

## 313. Communication-Efficient Personalized Federated Learning via Layer-Wise Multi-Threshold Random Sketching

**arXiv ID:** 2609.04830 | [PDF](https://arxiv.org/pdf/2609.04830v1)

**作者:** Xu Zhang `[一作]` (Xidian University), Maoguo Gong `[通讯]` (Inner Mongolia Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于层级多阈值随机投影的个性化联邦学习框架pFedLMS，用低比特双向通信实现高效协同训练；

**💡 创新点**

创新点在于为每一层分配独立阈值集，实现多阈值量化；通过加权多数表决得到低位一致性信号，并用间隔一致性正则化对本地模型进行对齐，显著提升信息表达能力；

**🔧 技术方法**

使用随机投影（Hadamard投影）进行压缩，多阈值二值化与编码、加权多数投票、Nesterov平滑化、局部SGD与正则化训练；

**📊 数据集**

实验数据集包括MNIST、FMNIST、SVHN、CIFAR‑10和CIFAR‑100，采用20个客户端的非IID分布；

**📈 对比分析**

与FedAvg、OBDA、OBCSAA、FedProto、EDEN、zSignFed和pFed1BS等基线对比，pFedLMS在大部分数据集上获得与FedAvg相近的准确率，同时通信开销低约98%；

**⚠️ 局限性**

局限性：理论分析中的漂移项上界可能过于保守，阈值选择对性能影响显著，需要经验调参；对极端异构或极低带宽环境的适应性尚待进一步验证；

---

## 314. CPL: A Compact C-like Systems Language with Explicit Low-Level Control

**arXiv ID:** 2609.04904 | [PDF](https://arxiv.org/pdf/2609.04904v1)

**作者:** Nikolay Fot `[一作]`, Alexander Vinarsky `[通讯]` (Ivannikov Institute for System Programming of the Russian Academy of Sciences)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出并实现了CPL（Cordell Programming Language），一种紧凑的C-like系统语言与对应的SSA/SMT优化编译器，旨在为系统级实验提供可观察、可调试的测试平台。

**💡 创新点**

创新点在于将C的低级语义与现代语言的简洁语法结合，设计了专用注解与容器机制，并实现了基于Z3的SSA层符号检查与可编程的PTRN peephole模式，形成了可插拔、可观测的编译器流水线。

**🔧 技术方法**

使用技术包括：基于LLVM风格的SSA/SMT中间表示、Z3符号执行、基于域特定语言的PTRN peephole、阶段化测试框架、NASM后端以及自定义注解指令（如@entry、@naked、@align、@nosection等）。

**📊 数据集**

数据集主要是微基准（empty loop、arithmetic recurrence、hot branch、hot function call、table traversal、string scan、Fibonacci）与系统级案例（Multiboot入口、i386键盘驱动等），并在单台Linux x86_64机器上执行十次测量。

**📈 对比分析**

比较方法：将CPL编译出的NASM汇编与GCC/Clang的C基准编译结果在同一目标架构（x86_64和i386）下直接对比运行时（秒）；结果显示CPL在空循环和Fibonacci上与C编译器相当，但在算术、分支、函数调用、表遍历和字符串扫描等优化较深的内核中表现明显逊色。

**⚠️ 局限性**

主要局限包括：缺乏形式化语义与验证、仅对x86家族的后端做了有限验证、符号诊断的精确度与性能未知、微基准规模小、未进行缓存/硬件计数分析、内联汇编缺乏优化、以及未覆盖更广泛的目标体系结构和大规模应用基准。

---

## 315. From Language Models to World-Acting Systems: Progress and Limits of Agentic AI across Digital, Social, Virtual, and Physical Environments

**arXiv ID:** 2609.04894 | [PDF](https://arxiv.org/pdf/2609.04894v1)

**作者:** Linsen Zhu `[一作]`, Mengqing Cai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对截至2026年8月31日公开的代理式语言模型研究进行系统性、批判性综述，阐述其核心维度（委托权限、时间持久性、环境耦合）并对模型、框架、环境、委托者等主体进行严格分离，形成“有理委托”评估框架；

**💡 创新点**

提出将代理系统视为“配置化行动系统”而非单纯模型，强调模型、掌控层、环境与委托者各自责任；将研究聚焦在“有理委托”而非单一“自主度”指标，揭示行动覆盖、能力提升与可验证性之间的差距；

**🔧 技术方法**

综合分析了ReAct、Toolformer、Reflexion等思考-行动循环技术，MCP、A2A等开放协议，SWE-agent等编程代理、WebArena/OSWorld等可执行环境，VLA/机器人实验、物理实验室自动化等物理接口，MHS硬件标准等；

**📊 数据集**

主要参考公开基准数据集（WebArena、OSWorld、SWE-bench、τ-bench、VJEP‑2、RT‑2、OpenVLA等）、实验室数据（Coscientist、ChemCrow、AFMBench）、以及各机构发布的系统卡与预览代码；

**📈 对比分析**

通过对比公开基准结果与第一方预览，发现行动接口（API、图形化、编码、机器人、实验室）在功能覆盖上已大幅提升，但在成功率、连贯性、错误检测与恢复、权限验证等方面仍低于公开基准；对比时采用“相同模型+相同掌控层”或“相同掌控层+相同模型”的因子实验设计，报告总体推理预算、工具调用次数、失败率、人工介入频率等多维指标；

**⚠️ 局限性**

局限性包括：综述仅覆盖公开资料，未能量化研究数量；缺乏对高风险应用场景（如医疗、金融）真实环境评估；代理系统中人类监督与权限验证仍处于试验阶段，难以在大规模部署中统一标准；基于预览的技术（Genie 3、Project Eden、MHS）缺乏独立复现与长期安全性验证。

---

## 316. MARLA: A Conceptual Scaffold for Regulatory Learning under the EU AI Act

**arXiv ID:** 2609.04877 | [PDF](https://arxiv.org/pdf/2609.04877v1)

**作者:** Alessio Buscemi `[一作]` (Luxembourg Institute of Science and Technology), Antonino Rotolo `[通讯]` (University of Bologna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 MARLA（Map、Assess、Report、Learn、Adapt）这一监管学习框架，用以系统化欧盟 AI 法案下法律义务向技术实践的转化过程，并通过在本地、国家和欧盟层面的案例演示其可操作性。

**💡 创新点**

创新点在于：①将监管学习设计为需求驱动、非强制化的循环模型；②明确每一阶段的学习产出（映射洞察、评估结果、报告实践等）；③构建三层级治理视角，展示学习从局部到全局的流动；④将框架与负责任创新的四大维度（预期、反思、包容、响应）对齐，提供统一的沟通与协作语境。

**🔧 技术方法**

技术层面主要为概念模型与流程设计；案例中应用的技术包括：偏见检测工具、对话生成模型（LLM）鲁棒性与多语言评测框架，以及安全防护（guardrail）评估；但未提出新的算法或技术创新。

**📊 数据集**

数据集方面：银行对话评估使用了 BIL（Banque Internationale à Luxembourg）收集的多语言客户交互样本；Sandbox 评估使用了多语言 LLM 生成的对话与攻击样本；通用公开数据集未被正式引用，数据均来自参与机构的内部数据。

**📈 对比分析**

方法比较：论文未进行算法性能对比，而是通过对不同语言模型、鲁棒性测试结果以及 guardrail 触发情况进行定性比较；主要关注评估方法的适用性、阈值设定与结果可解释性，没有提供数值指标或基准性能。

**⚠️ 局限性**

局限性包括：①框架不提供具体阈值或技术规范，导致实施差异；②缺乏实证验证与经验评估；③对通用 AI 模型的监管路径未完整覆盖；④在代理 AI 场景下的适配仍需进一步细化；⑤多层级学习转化存在时间异步，导致本地发现的洞察在欧盟层面落地缓慢。

---

## 317. Forgetting Without Restarting: Execution-State Unlearning for Stateful LLM Agents

**arXiv ID:** 2609.04875 | [PDF](https://arxiv.org/pdf/2609.04875v1)

**作者:** Chao Yao `[一作]` (Arizona State University), Lei He `[通讯]` (Eastern Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向长期运行的LLM代理的执行状态遗忘（unlearning）方法，并实现了可审计的跨层次系统——Provenance‑Guided Selective Replay；

**💡 创新点**

创新点在于：①将遗忘定义为与未观察到目标的对照世界的计数器事实轨迹一致；②在确定性转移系统框架下给出前缀共享与后缀污染的两条定理，并证明最优重放成本为 T‑τ+1；③构建了基于原始运行时证据的 provenance 图、稀疏检查点与“裁剪+重放”策略，实现了实际可用的遗忘操作；

**🔧 技术方法**

使用技术包括：KV 缓存裁剪、原型化的原型图记录、基于检查点的快照恢复、受控重放（shadow‑execution + 预填充）以及对照计数器事实的算子实现；

**📊 数据集**

评测数据集包含三套代理场景：LongMemEval（内存注入）、ToolSandbox（工具返回注入）和 AgentDojo（实际任务场景），分别在 Llama‑3.1‑8B、Qwen2.5‑7B、Mistral‑7B 三大模型上进行实验；

**📈 对比分析**

对比方法包括无遗忘、内存删除、遗忘指令、源红色化、全重置、IFC 等；在泄露、行为距离、重计算代价等多维度评测中，Selective Replay 在所有泄漏度指标上与全重置相当（Leak@probes、Leak@5、CAD 等均为零），而重计算代价仅为全重置的 1/9 左右，且遵循理论预期的 T‑τ+1 线性增长；

**⚠️ 局限性**

限制包括：只能针对可重构的运行时状态（不涉及模型参数或已提交的外部副作用）；假设解码确定性且工具返回可快照重放；对比测试主要基于字符串匹配，可能无法捕捉所有行为差异；

---

## 318. A Wavelength Borrowing Architecture for Optical Data Center Networks - Extended Version

**arXiv ID:** 2609.04874 | [PDF](https://arxiv.org/pdf/2609.04874v1)

**作者:** Andrea Detti `[一作]` (CNIT - University of Rome Tor Vergata), Silvello Betti `[通讯]` (CNIT - University of Rome Tor Vergata)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出了一种可调度的光学脊叶架构，通过光纤借用实现不同叶节点间波长重分配。

**💡 创新点**

创新点在于用单一可调借用度B控制硬件复杂度和可重配置性，并通过无色OxC和被动AWGR实现无光信号处理的可扩展设计。

**🔧 技术方法**

采用AWGR、色彩无关OxC、被动组合器、SDN控制的波长分配以及基于两跳水位填充的贪心算法。

**📊 数据集**

实验使用Lognormal生成的流量矩阵A^E，平均0.65，系数变异cv可调；模拟不同B值与基线对比。

**📈 对比分析**

与仅AWGR的静态核心、统一两跳转发以及两跳水位填充的对照进行比较，结果显示B=8即可达到近似最优，显著降低两跳转发比例并满足负载阈值，且所需借用线数随规模缩小。

**⚠️ 局限性**

局限在于仅考虑非TDMA场景、缺乏大规模仿真和实际部署验证，以及对流量模型的假设和对光衰减、放大器需求的简化。

---

## 319. The Generative AI Gold Rush in Theoretical and Computational Research

**arXiv ID:** 2609.04872 | [PDF](https://arxiv.org/pdf/2609.04872v1)

**作者:** Xiaoshn Nee `[一作]` (Independent researchers), Xiaomin Ni `[通讯]` (Artificial Intelligence Research Institute)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用凸正则化合成控制方法分析2026年arXiv数学预印本量激增，并结合作者产出、AI扩散与能力阈值等证据探究其机制。

**💡 创新点**

首次将多重稳健性检验（空间安慰剂、先前年份伪留存、留一捐赠者重估等）与合成控制相结合，构建多维证据链，揭示AI驱动的生产模式转变。

**🔧 技术方法**

使用凸正则化合成控制、空间安慰剂检验、先前年份伪留存、留一捐赠者重估等统计技术。

**📊 数据集**

arXiv 20个主要归档的月度条目计数（2018‑2026），全平台月度提交总数，匿名作者产出面板（2023‑2026）以及相关文献语料库。

**📈 对比分析**

与自身历史趋势、合成对照以及多种安慰剂比较，发现数学归档在2026年相对差距超过30%，在15个可比归档中排名第一；所有稳健性检验均保持显著。

**⚠️ 局限性**

仅基于归档列表计数，存在跨归档计数重叠、缺乏直接稿件质量测度，且仅覆盖2026年八个月，未能捕捉更长周期或更细粒度的生产结构变化。

---

## 320. AutoLR: Automating the Path from Research to Launch Review in Industrial Recommender Systems

**arXiv ID:** 2609.04871 | [PDF](https://arxiv.org/pdf/2609.04871v1)

**作者:** Qi Zhang `[一作]` (NetEase, Inc.), Wenchao Xiao `[通讯]` (NetEase, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了 AutoLR 系统，实现从研究到 Launch Review 的全流程自治，包括知识检索、专家委员会生成方案、方向选择、代码实现、训练与离线评估以及候选包装；

**💡 创新点**

核心创新在于以治理为中心的生命周期驱动机制：多专家委员会与证据加权的方向选择、分层知识与实验记忆、基于约束的代码实现与离线促销决策（KEEP 与 PACK），以及对可变基线风险（KEEP ratchet）的识别与阐释；

**🔧 技术方法**

利用大型语言模型（Claude、DeepSeek 等）完成语义推理与代码生成，结合确定性控制器负责执行、监控、指标提取、门控与持久状态管理；

**📊 数据集**

在网易游戏社区推荐应用 DASHEN 的单/双列 Feed 与沉浸式视频 Feed 上进行实验，收集约 1,586 条离线评估日志和 9 条上线 Launch Review 记录；

**📈 对比分析**

通过对比离线评估与上线 A/B 结果，观察到累计提升约 +5.75% 的消费渗透率、+10.83% 的总消费时长、+5.55% 的有效观看次数；系统迁移至低成本 LLM 后，API 成本降至 3–4 元/轮；

**⚠️ 局限性**

局限包括：未能单独评估各组件因果效果；缺乏完整阈值与门控记录、不可变的离线-上线链路；仅在两种推荐场景验证，缺乏跨任务转移评估；对 KEEP 机制的统计显著性与最优阈值未确定；

---

## 321. From Interaction Traces to Persistent Skills: Online Evolution for Computer-Use Agents

**arXiv ID:** 2609.04869 | [PDF](https://arxiv.org/pdf/2609.04869v1)

**作者:** Longtao Hu `[一作]` (University of Electronic Science and Technology of China), Linchao Zhu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建在线技能进化框架，利用每轮冻结的技能库快照、交互轨迹和评估者反馈创建、修订并版本化可复用的程序库。

**💡 创新点**

通过将执行与技能演化解耦，实现可审计、共享的程序记忆，并展示其在固定计算机使用栈上对长期性能的可测增益。

**🔧 技术方法**

采用固定的执行器 EvoCUA‑32B、MVP 视觉定位、Kimi K2.5 提取器，以及基于结构化事实的提议器/构造器，构建无模型参数更新的外部技能库。

**📊 数据集**

基于 OSWorld 的四个桌面应用任务集：GIMP、VLC、LibreOffice Writer 和 Thunderbird。

**📈 对比分析**

采用配置匹配的纵向对比（Full vs Empty），在 Warm‑up 5 次后 Full 在所有域均取得更高的评估者平均分，差距为 5.7–18.6 百分点，表现随域和时间不一。

**⚠️ 局限性**

单次实验、Warm‑up 不平衡、仅 GIMP 的 provenance、固定执行/定位模块的瓶颈、序列化演化顺序依赖、缺乏 RL 优化提议器等限制。

---

## 322. CC-Mediation: Evaluating Large Language Models for Cross-Cultural Conflict Mediation

**arXiv ID:** 2609.04855 | [PDF](https://arxiv.org/pdf/2609.04855v1)

**作者:** Suhyun Lee `[一作]` (Hanyang University), Yang Deng `[通讯]` (Singapore Management University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CC‑Mediation跨文化调解基准数据集，并提出了基于DMIS的轨迹AUC和符号Wasserstein-1距离两个评价指标，用于量化调解过程中跨文化立场的持续性与幅度。

**💡 创新点**

创新点在于：①首次系统地把跨文化调解问题与Developmental Model of Intercultural Sensitivity（DMIS）结合，提供阶段感知的对话标注；②设计了两种能捕捉立场变化轨迹的定量指标；③通过大模型驱动的生成与筛选流程，得到高质量的对话与调解样本；④对LLM在“何时介入”和“如何介入”两方面的失败机制进行机制性分析。

**🔧 技术方法**

使用的技术包括：多模态LLM推理（GPT‑4o‑mini、Llama、Gemma、Claude等）、层级内部表示探针、Cramér’s V等统计分析、基于自监督与监督微调的实验设计。

**📊 数据集**

数据集：1,661条10回合对话（1,503训练、158评估），共18,081发言，其中包括冲突前/冲突/冲突后标注的DMIS阶段和真实的调解发言；所有对话均由LLM生成，后续人工验证。

**📈 对比分析**

方法对比：在冲突检测任务上，六大LLM的turn‑acc低于33%，显示出强烈的“位置先验”偏置；在调解生成任务上，基线AUC仅0.68–1.44，低于无指导的GT平均1.64；提供DMIS指导后AUC提升0.03–0.20，但总体仍未达到GT；通过监督微调后，时机检测准确率从约30%跃升至90%+，并在控制时机后AUC、Wasserstein-1等内容指标均出现明显提升。

**⚠️ 局限性**

局限性包括：①数据为LLM合成，缺乏真实情绪与非语言信号；②仅英文对话，忽略多语种、语言选择等文化信号；③DMIS标注衡量表达层面的立场转变，未必对应真实的内在信念变化；④存在文化本质化风险，需要谨慎使用与解释。

---

## 323. PAPT++: Risk-Aware Adversarial Tuning and Generation for Single Domain Generalization

**arXiv ID:** 2609.04837 | [PDF](https://arxiv.org/pdf/2609.04837v1)

**作者:** Zhipeng Xu `[一作]` (Xidian University), Xinbo Gao `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 PAPT++ 框架，在单源域泛化中使用风险感知的对抗扩散生成来丰富源分布，并结合语义参考与分类器引导的生成策略，提升模型对未知目标域的鲁棒性。

**💡 创新点**

创新点包括：①将扩散生成空间构造为语义模糊集，通过语义参考约束降低语义漂移；②在生成过程中加入分类器风险最大化和分层去噪，生成既高风险又语义一致的样本；③从 DRO 视角提供理论支持，关联风险搜索、语义约束与目标域误差。

**🔧 技术方法**

使用的技术包括 Stable Diffusion 文本到图像扩散模型、LoRA 微调、图像-文本对齐与多样性正则的语义参考学习、分类器引导的对抗扩散（CADS）、progressive denoising 以及 CFG++ 的增强引导。

**📊 数据集**

实验数据集涵盖单源域泛化基准 PACS、VLCS、OfficeHome 以及 CIFAR-10-C；多源域泛化评测包含 TerraInc、OfficeHome、VLCS。

**📈 对比分析**

与传统数据增强、对抗扰动、分布式鲁棒优化等单源/多源方法对比，在 ResNet-18/ResNet-50 上平均提升 0.7–1.1pp；在 CIFAR-10-C 5 级失真上取得 82.34% 的平均精度，明显优于 PAPT 与 UDIM；在多源 DG 上亦获得最佳或接近最佳表现。

**⚠️ 局限性**

局限性：依赖大型扩散模型，训练时间和 GPU 资源消耗较大；生成样本质量对 CFG/CFG++ 参数敏感；对极端视觉偏差的覆盖仍有限；尚未在动态或多任务迁移场景中验证。

---

## 324. PACE: Propagation-Aware Collaborative Correction for One-Shot Personalized Federated Graph Learning

**arXiv ID:** 2609.04832 | [PDF](https://arxiv.org/pdf/2609.04832v1)

**作者:** Ruizhe Huang `[一作]` (Wuhan University), Xiaochuan Shi `[通讯]` (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出PACE，一种一轮个性化联邦图学习方法，通过上传低秩修正向量和传导时刻的二阶矩进行通信，服务器返回与接收者本地模型对齐的低秩纠正，接收者通过CNLL校准决定使用量。

**💡 创新点**

创新点在于将共享知识视为完整本地模型的紧凑、传导感知校正，而非替代模型，并通过收敛凸NLL对齐使接收者自行选择外部贡献。

**🔧 技术方法**

使用低秩SVD（Rank‑6）压缩模型更新、RegMean一致性、对传导消息二阶矩的对角化压缩、CNLL凸logit插值校准以及GCN本地训练等技术。

**📊 数据集**

实验数据集包括Cora、CiteSeer、PubMed、CS、Computers以及ogbn‑arxiv六个子图联邦基准。

**📈 对比分析**

在匹配的一轮协议下与本地训练及八种协作基线（FedAvg、FedProx等）对比，PACE在五个数据集上均优于Local并在ogbn‑arxiv上保持一致，平均提升准确率与加权F1，通信量仅为稠密张量的约10–18%。

**⚠️ 局限性**

局限性包括仅在Louvain社区划分上验证、缺乏正式隐私保障、最差客户端性能保障不足、依赖共享初始化与参数对齐、Rank‑6压缩特定于所用模型和序列化方案，且未解决参数置换对齐问题。

---

## 325. Methane Detection On Board Satellites from Unorthorectified Imagery

**arXiv ID:** 2609.04906 | [PDF](https://arxiv.org/pdf/2609.04906v1)

**作者:** Luca Marini `[一作]` (Delft University of Technology), Giacomo Acciarini `[通讯]` (European Space Agency)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套 UnorthoDOS 数据集及训练方法，直接在未正射校正的 EMIT 高光谱卫星图像上检测甲烷羽流，省去传统的正射校正和匹配滤波步骤。

**💡 创新点**

创新之处在于首次证明可在未正射校正图像上实现与正射校正模型相当的甲烷羽流检测效果，并展示了模型压缩后在机载设备上的可部署性。

**🔧 技术方法**

采用 86 通道 U‑Net 深度学习模型，使用 Dice 损失进行训练，并通过几何映射逆向生成未正射校正图像；随后使用 FP16 与 INT8 权重量化实现模型压缩。

**📊 数据集**

使用 EMIT 传感器采集的 86 带高光谱影像（含 RGB）与 1,574 条甲烷羽流标注（来自 L2B 正射校正产品），构成 UnorthoDOS 与正射校正基准数据集，公开发布于 HuggingFace。

**📈 对比分析**

将 U‑Net 在正射校正与未正射校正数据上训练，并与 mag1c 匹配滤波基线比较。结果显示，未正射校正模型 IoU 为 16.91%（正射 18.47%），均显著优于 mag1c 的 4.76%；分类准确率在强羽流时分别为 92.41%（正射）与 85.65%（未正射）。压缩后 FP16 模型仅 1.70 MB，最大输出偏差 <0.3%。

**⚠️ 局限性**

主要局限在于对弱、低浓度甲烷羽流的检测灵敏度不足，且 INT8 量化后输出偏差较大，需要更大规模数据或进一步验证以提升模型在资源受限平台上的性能。

---

## 326. Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference

**arXiv ID:** 2609.04895 | [PDF](https://arxiv.org/pdf/2609.04895v1)

**作者:** Zhenhe Wu `[一作]` (Huawei Technologies), Hanting Chen `[通讯]` (Huawei Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Mixture-of-Experts模型中提出一种后训练的缓存管理框架，包含只更新的Temporal Router和带时空预取的Spatio-Temporal Router。

**💡 创新点**

在保持原生Top-K专家选择不变的前提下，通过后训练学习辅助路由器来优化缓存优先级，支持更新仅模式和有限预取的完整模式。

**🔧 技术方法**

利用缓存损失函数、soft Top-B 归一化、时空预测器以及post‑access retention 与 pre‑access refinement 技术，结合 Qwen3 与 GPT‑OSS 进行后训练。

**📊 数据集**

使用 GSM8K、MATH 和 CommonsenseQA 三个推理基准进行评估。

**📈 对比分析**

与传统 LRU/LFU/LRFU 替换策略以及 ProMoE、FineMoE、Temporally‑Extended MoE 等预取方法对比；Temporal Router 在无预取时显著提升命中率并降低专家权重传输；Spatio‑Temporal Router 在 Qwen3 上实现最高调整后命中率和最低专家传输，GPT‑OSS 结果竞争性。

**⚠️ 局限性**

仅关注算法层面，未结合完整部署栈；预取量与实际吞吐/延迟取决于调度与硬件；需要对整个模型进行后训练；实验范围局限于 MoE 模型和指定基准，未涵盖混合稠密模型或不同内存预算场景。

---

## 327. SimFuse3D: Source-Guided Target Simulation and Confidence-Guided Multi-Stage Localization Reweighting for Cross-Platform 3D Object Detection

**arXiv ID:** 2609.04886 | [PDF](https://arxiv.org/pdf/2609.04886v1)

**作者:** Yongchun Lin `[一作]` (Guangdong University of Technology), Haoang Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在无监督跨平台 LiDAR 检测中，利用自训练方法修复伪标签中点与框不一致的问题

**💡 创新点**

提出 SimFuse3D，结合源域实例记忆恢复点云几何，并通过 Confidence‑Guided Multi‑Stage Localization Reweighting (CMLR) 控制伪标签权重

**🔧 技术方法**

使用对象记忆检索、目标模拟对齐与角度过滤、CMLR 加权训练以及 Pi3DET‑Net 结构

**📊 数据集**

在 nuScenes、Vehicle、Quadruped、Drone 以及 KITTI 等跨平台数据集上进行实验

**📈 对比分析**

与 Pi3DET‑Net、ST3D、MS3D++ 等多种 UDA 方法对比，在六个跨平台转移中所有 AP 指标均优于对照组，并在 nuScenes→KITTI 适配下取得最高分

**⚠️ 局限性**

限制在于检索阈值固定、记忆存储庞大、对不同源域的适应性不足，未来需自适应检索与更紧凑的记忆构造

---

## 328. CoSkill: Joint Reinforcement Learning of Reasoning and Meta-Skill Agents for Hierarchical Skill Evolution

**arXiv ID:** 2609.04865 | [PDF](https://arxiv.org/pdf/2609.04865v1)

**作者:** Jinyuan Feng `[一作]` (Institute of Automation, Chinese Academy of Sciences), Zhiqiang Pu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种联合训练的多智能体强化学习框架CoSkill，用大语言模型同时学习推理策略和元技能编辑器，以实现技能库与策略的协同进化；

**💡 创新点**

创新点在于把元技能从固定工作流转化为可学习的Agent，采用层次化技能库和延迟验证奖励，使技能编辑与环境奖励直接对齐，并通过共享参数实现两者的协同更新；

**🔧 技术方法**

技术包括基于LLM的共享策略、层次化技能检索、后编辑验证奖励、GiGPO分组策略优化以及多智能体半马尔可夫决策过程；

**📊 数据集**

在ALFWorld和WebShop这两个长时程交互式任务环境上进行实验；

**📈 对比分析**

与闭源LLM、提示式、无技能强化学习、以及已有技能强化学习基线相比，CoSkill在ALFWorld上平均成功率达98.4%（比最优基线高3.5pp），在WebShop上成功率90.6%（比最优基线高6.2pp），并在样本效率和壁钟时间上也显著优于对照方法；

**⚠️ 局限性**

局限性包括需要共享LLM模型且训练成本仍较高，元技能编辑的策略仍依赖大量后验证回放，且对更大规模或更复杂任务的可扩展性尚未充分验证。

---

## 329. MM-IFEval-Pro: A Multilingual and Attack-Resistant Benchmark for Instruction-Following in Vision-Language Models

**arXiv ID:** 2609.04859 | [PDF](https://arxiv.org/pdf/2609.04859v1)

**作者:** Changming Xiao `[一作]` (Huawei Technologies Limited), Jie Hu `[通讯]` (Huawei Technologies Limited)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MM-IFEval-Pro 评测基准和对应的 RL 训练集，涵盖中英双语、多约束任务以及视觉指令劫持场景。

**💡 创新点**

创新点包括：① 把指令劫持视为独立、可规则化的评测维度；② 采用语言感知的规则验证器，避免语言偏差；③ 通过全参数 GRPO 强化学习实现跨语言、跨任务的鲁棒提升。

**🔧 技术方法**

使用技术主要有：全参数 GRPO 强化学习；四阶段规则化数据生成管道；基于规则的可验证约束与自动化评测器；多语言约束元数据管理。

**📊 数据集**

使用的数据集包括：新构建的 MM-IFEval-Pro 训练集 13,547 条样本和 1,279 条测试样本；以及 MM-IFEval、MIA‑Bench、IFEval 等公开基准，用于跨基准对比；同时在 STEM、VQA、文档理解等多任务基准上进行评测。

**📈 对比分析**

通过在上述基准上与未微调模型和仅使用英文训练集的对比，GRPO 训练后模型在 MM-IFEval-Pro 上从约 78% 提升至约 89%，在指令劫持任务上显著提升；在 MM-IFEval、MIA、IFEval 以及 STEM、VQA、文档基准上也保持或略有提升，表明训练集具有良好的跨任务、跨语言泛化能力。

**⚠️ 局限性**

局限性包括：不同模型对指令劫持的鲁棒性差异仍显著；评测仅覆盖文字型视觉劫持，未涵盖其它视觉攻击方式；训练集规模和多样性虽然提升，但仍可能不足以覆盖所有真实场景。

---

## 330. Coupled Control and Wireless World Models for Resilient Remote Robotic Control

**arXiv ID:** 2609.04851 | [PDF](https://arxiv.org/pdf/2609.04851v1)

**作者:** H. P. Madushanka `[一作]`, Mehdi Bennis `[通讯]` (Centre for Wireless Communications)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一套基于耦合cJepa和wJepa预测世界模型的无线通信感知远程机器人控制框架，联合学习机器人动态与无线通道演化，实现有限无线资源下的可靠控制与通信调度。

**💡 创新点**

创新点包括：1）将控制与无线世界模型耦合于同一潜在空间；2）使用多模态输入（视觉+结构化RF）提升预测精度；3）引入自适应鲁棒机制实现在线感知嵌入的自适应调整；4）在同步Gazebo–Sionna仿真环境中验证鲁棒性。

**🔧 技术方法**

采用耦合cJepa/wJepa的预测世界模型、结构化RF表示（光谱图与持久性图）、视觉VAE自适应、离线迁移学习与同步仿真集成。

**📊 数据集**

使用Gym CarRacing训练的视觉序列并迁移至Gazebo中的JetBot视觉和Sionna生成的CSI/光谱图/PI，构建多散射、基站切换、载波频移等扰动的无线环境。

**📈 对比分析**

与DQN和PID基线相比，任务完成时间和累计奖励均提升；通信上传延迟、能量消耗显著降低；在频移、散射、切换等干扰下保持高成功率，预测误差更低。

**⚠️ 局限性**

仅在单机器人单链路、仿真环境中验证；对多机器人、多网络、真实硬件验证及更高效在线自适应仍待研究。

---

## 331. LetOccVote: Learning Weakly Supervised 3D Occupancy through Consensus

**arXiv ID:** 2609.04846 | [PDF](https://arxiv.org/pdf/2609.04846v1)

**作者:** Chi Zhang `[一作]` (Chinese University of Hong Kong, Shenzhen), Rui Huang `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 LetOccVote 框架，通过多视角投票（Depth Vote 与 Semantic Vote）在弱监督下提升 3D 占据预测，利用 2D 伪标签的共识来过滤噪声并增强可靠性。

**💡 创新点**

创新点在于将几何与语义一致性投票作为伪标签可靠性评估，不仅优化深度一致性，还通过语义投票生成可靠标签并掩码不可靠标签，实现更稳健的弱监督。

**🔧 技术方法**

采用多视图特征提升到共享 3D 体素、Gaussian 采样解码、基于深度投票的伪深度校正、语义投票掩码、渲染监督和 RayIoU 评价等技术。

**📊 数据集**

使用 Occ3D‑nuScenes 数据集，仅利用 Grounded‑SAM（语义）和 Metric3D‑v2（深度）生成的 2D 伪标签进行训练。

**📈 对比分析**

在 Occ3D‑nuScenes 验证集上与现有基于 2D 伪标签的弱监督方法比较，LetOccVote‑L 在 IoU 达到 53.27、mIoU 20.39、RayIoU 20.91，明显优于 GaussTR、GaussianFlowOcc 等基线。

**⚠️ 局限性**

局限在于对极其嘈杂或模糊的语义类别仍易受误标影响，且投票参数和多帧窗口会增加内存开销。

---

## 332. KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU

**arXiv ID:** 2609.04852 | [PDF](https://arxiv.org/pdf/2609.04852v1)

**作者:** Di Chai `[一作]` (Shanghai University of Finance and Economics), Zhihang Yu `[通讯]` (Jilin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对 LLM 代理的工作区溢出问题，提出 KV‑Context 虚拟化系统，允许代理在 GPU 内存之外维护超大工作区，并在每一步根据查询动态检索并重构 KV 块。

**💡 创新点**

创新点在于把历史上下文视为可虚拟化的 KV 资源，利用模型原生注意力空间索引（Mean‑K）与分层 KV 管理（分页、重编码、流水线重建）实现高效检索与位置一致的重置，而不是压缩成文本。

**🔧 技术方法**

核心技术包括：Mean‑K 轻量索引、RoPE 重编码、分层 KV 存储（GPU、主机内存、NVMe）、预取/重建流水线、分步内存调度，以及自研 QW3 推理引擎。

**📊 数据集**

使用 LongMemEval‑S、MemoryAgentBench、AgentLongBench、DeepSWE 等长上下文代理基准进行评估。

**📈 对比分析**

与滑动窗口、压缩、压缩+RAG 等传统方法相比，KV‑Mem 在保持任务效能（接近 Full Context）且预答复延迟仅 1–2 秒的前提下，在消费者设备上实现 1M 令牌虚拟工作区，在服务器端扩展至 10M 令牌，同时显著降低预处理时间与整体延迟。

**⚠️ 局限性**

局限性包括：无法无缝叠加在现有黑盒 LLM API 上，需要对 KV 分配与 RoPE 重编码做显式控制；KV 状态比文本占用更大存储，导致主机/NVMe 开销上升；仅提升单步上下文大小，仍无法跨步联合注意力。

---

## 333. InterSing: Explicit Interaction Dynamics for 3D Duet Singing Animation and Beyond

**arXiv ID:** 2609.04903 | [PDF](https://arxiv.org/pdf/2609.04903v1)

**作者:** Yihan Zhou `[一作]` (South China University of Technology), Shengfeng He `[通讯]` (Singapore Management University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对二重唱表演的3D头部动画框架 InterSing，能够在保持个体表达的同时实现实时的互动协调。

**💡 创新点**

创新点包括：①将二重唱互动建模为可解释的时间序列“交互对数”并通过弱监督对比学习得到；②在交互感知的扩散生成器中同时引入音频、说话者活跃度与交互对数，实现统一的多模式（完整二重唱、伙伴预测、交互调节）生成；③使用交互跨注意力与分数注入，保证长程时序一致性。

**🔧 技术方法**

主要技术：FLAME 3D人脸模型、Transformer 编码器与解码器、弱监督对比学习、neck‑masking 防止几何捷径、交互对数约束、交互跨注意力、扩散强制（Diffusion Forcing）以及音频特征编码（Whisper+Whisper Transformer）。

**📊 数据集**

数据集：基于现有 ChorusHead 的二重唱片段，结合公开二重唱视频，共计约9小时、414段、972,394帧；训练集/测试集采用歌曲互斥 85:15 分割。

**📈 对比分析**

对比方法包括 PaChorus、DualTalk、DIM、UniLS 等对话或单人驱动模型。InterSing 在多项指标上显著优于基线：LVE 下降 16.1%（相对 UniLS），FID/P‑FID 下降 29.2%/25.9%，并在用户评估中获得最高的唇同步、表情丰富度和交互自然度分数。

**⚠️ 局限性**

局限性：FLAME 表达无法捕捉细粒度眼神与注视；缺乏大规模多人的合唱数据，导致多参与者场景的评估受限。

---

## 334. RefactorPlatform: An Open-Source Harness for Controlled Evaluation of Repository-Scale Refactoring Agents

**arXiv ID:** 2609.04898 | [PDF](https://arxiv.org/pdf/2609.04898v1)

**作者:** Aziz Ben Amor `[一作]` (Pi School), Sébastien Bratières `[通讯]` (Translated)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 RefactorPlatform，一个面向仓库级重构任务的可扩展评估平台，提供隔离工作区、检索增强、命令行终端、AST 验证等功能。

**💡 创新点**

将评估环境固定，只显式变化模型、检索、提示、协同四个设计轴；实现 AST 验证门控、可视化仪表盘以及完整日志导出，首次实现对仓库级重构的可复现审计。

**🔧 技术方法**

使用 OpenRouter/GitHub Copilot、Model Context Protocol 检索、CocoIndex 向量检索、AST Chunking、LSP 反馈、子代理（S_3）、CodeBLEU、RefactoringMiner 等技术。

**📊 数据集**

使用 RefactorBench（100 个 Python 多文件任务）和 SWE-Refactor（Java）作为基准数据集。

**📈 对比分析**

通过在 S_1、S_2、S_3 三种执行模式以及 Descriptive/Base/Lazy 三种提示模式下的全因子实验，对比通过 AST Chunking 检索、无检索、子代理等，发现 AST Chunking+检索可将通过率提升至 86%，成本每成功案例约 $0.13，子代理效果不如检索。

**⚠️ 局限性**

限制在于仅测试中等规模模型，主要聚焦 Python，未覆盖大规模模型、编译型语言以及更大规模实验。

---

## 335. Reasoning Without Inference Cost: Latent Semantic Scaffolding for Robot VLA Policies

**arXiv ID:** 2609.04893 | [PDF](https://arxiv.org/pdf/2609.04893v1)

**作者:** Andrew Ting Yan Li `[一作]` (Chinese University of Hong Kong), Fei Chen `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在训练阶段引入Latent Semantic Scaffolding，将VLA的动作标记表示与每个操控阶段的物理推理文本嵌入对齐，随后在推理时移除对齐模块，从而在不增加额外成本的前提下提升策略的因果理解与迁移能力。

**💡 创新点**

①通过训练时对齐动作表示与推理文本实现因果推理的“烘焙”，②证明按阶段局部对齐（dense alignment）比单一池化对齐更能迁移，③在推理阶段实现零额外开销。

**🔧 技术方法**

使用H‑RDT 2B diffusion‑transformer backbone、Frozen DINO+SigLIP视觉编码器、T5‑XXL文本编码器、cosine对齐损失、阶段局部dense对齐、flow‑matching损失。

**📊 数据集**

人类演示数据集EgoDex（500条含左右手、两种瓶子），以及RoboTwin 2.0仿真评估任务（含分布内、相关、异构任务）。

**📈 对比分析**

与基线（无推理、AVP等）进行消融实验；在分布内任务成功率从70%提升到90%，在未见任务上从39%提升到54%，整体表现显著优于对齐与否或不同对齐粒度的对照实验。

**⚠️ 局限性**

仅在单一人类演示瓶子操作域验证，依赖VLM生成的阶段分割与推理文本质量，缺乏在线重规划能力，未验证跨物体/技能的更广泛泛化。

---

## 336. Finding Many Overlapping Dense Subgraphs Using Triadic Cohorts

**arXiv ID:** 2609.04890 | [PDF](https://arxiv.org/pdf/2609.04890v1)

**作者:** Sabyasachi Basu `[一作]` (Microsoft Research), C. Seshadhri `[通讯]` (University of California Santa Cruz)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在大规模稀疏图中提出一种新的密集子图挖掘框架，利用三角形信息构造“三元队列”（triadic cohorts），并给出一种可多次覆盖、可并行化的算法 CohortRecovery，能够在保持高密度的同时输出覆盖图中大部分顶点的重叠密集子图。

**💡 创新点**

创新点主要包括：①定义了三元队列概念，刻画了可被检测的密集结构并给出了可判定的可识别条件；②证明了在无任何假设下，算法能够近似恢复所有可识别的三元队列；③设计了基于三角形计数的清理步骤、独立集种子选择和核心扩展的实用实现，兼顾理论保证与时间效率；④在覆盖度、密度、可扩展性等多维度上实现了相较现有方法的显著提升。

**🔧 技术方法**

技术手段包括：三角形枚举与计数、核心（k-core）分解、最大独立集采样、阈值化的三角形重叠检测、红边核心扩展、并行化实现与局部三角形重算；在实现层面采用C++ + O3 编译、仅保留必要的数据结构以降低存储需求。

**📊 数据集**

实验使用了18个真实图数据集（如 euemail、soc-ham、caAstroPh、Ph-Citations、Epinions、Slashdot、dblp、amazon、Berkstan、hollywood、youtube、pokec、skitter、wiki、large-dblp、orkut、cit-Patents、livejournal），并在7个带有手工标注聚类的子集上评估与真值匹配；此外还在合成的 Stochastic Block Model（覆盖与不覆盖）上进行验证。

**📈 对比分析**

与 RTR, CoDeSEG, Nucleus, Louvain, Leiden, AGM 等方法对比，CohortRecovery 在密度阈值0.5和0.8下的覆盖率几乎在所有数据集上排名第一，平均覆盖率可达25%（密度>0.8），在覆盖度与密度兼顾的情形下明显优于其他方法；在运行时方面，最大网络（117M边）在普通服务器上 8 分钟以内完成；在带真值的图上，MaxPrec 指标普遍高于竞争者。

**⚠️ 局限性**

局限性包括：①算法对参数 γ 的选择仍有一定敏感性，需要经验调优；②理论证明仅适用于可识别的三元队列，对极稀疏或极大规模的密集结构缺乏强保证；③三角形枚举仍是主要时间瓶颈，在极大图上可能仍受限；④输出可能产生过多小块或冗余聚类，实际应用中仍需后处理过滤。

---

## 337. AtomRec: Evolving Atomic Memory for Agentic Recommendation

**arXiv ID:** 2609.04882 | [PDF](https://arxiv.org/pdf/2609.04882v1)

**作者:** Peiyu Hu `[一作]` (Xi'an Jiaotong-Liverpool University), Jia Wang `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AtomRec，一个利用可演化原子化协同记忆的代理式推荐系统。

**💡 创新点**

将用户和物品记忆拆解为结构化原子化记忆，构建语义协同链接并实现历史记忆的演化，从而支持细粒度、可解释的证据路径推荐。

**🔧 技术方法**

使用大型语言模型（LLM）作为记忆生成、链接构建与演化以及推荐推理的核心，结合句子编码器进行检索，采用多跳语义链接和字段级更新。

**📊 数据集**

在四个指令增强的基准上评估：Amazon Books、Goodreads、MovieTV、Yelp。

**📈 对比分析**

与传统推荐、LM、代理式及记忆增强基线（LightGCN、SASRec、P5、iAgent、MemRec 等）进行对比，AtomRec 在所有数据集上均取得最优成绩，平均提升约 8.5%。

**⚠️ 局限性**

记忆演化可能导致兴趣过度压缩、对短期探索性偏好处理不足，且协同信号在记忆中消失，削弱后置可追溯性。

---

## 338. Personalized Task Dependency Graphs for Mitigating Signal Erosion in Multi-Task Recommendation

**arXiv ID:** 2609.04862 | [PDF](https://arxiv.org/pdf/2609.04862v1)

**作者:** Fuyuan Liu `[一作]` (Huawei Technologies Co., Ltd.), Jiandong Ding `[通讯]` (Huawei Technologies)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Personalized Task Dependency Graph (PTDG)，在多任务推荐中动态学习每个项目的任务依赖关系，缓解信号衰减问题。

**💡 创新点**

创新点在于通过低秩近似实现实例化任务图，结合硬因果掩码的 GCN 消息传递以及自适应进展掩码 (APM) 结构化分离共享参数，兼顾灵活性与低延迟。

**🔧 技术方法**

核心技术包括低秩图生成、GCN 级联传播、硬因果掩码、以及基于任务标签稀疏度的 APM 结构化掩码。

**📊 数据集**

实验使用 KuaiRand1K（6 任务）和工业级别的 7 任务数据集进行评估。

**📈 对比分析**

与 MMoE、PLE、STEM、MoCoGrad、PMTRec、MIT 等方法对比，PTDG 在两大数据集上均实现最高 AUC，尤其在稀疏深层任务提升高达 1.45%，在线 A/B 测试中 CVR 提升 1.2% 及 eCPM 提升 1.9%。

**⚠️ 局限性**

局限性包括对低秩维度和掩码率的超参数敏感，需要在更大任务规模和跨域场景下进一步验证与扩展。

---

## 339. ElderBench: Benchmarking Autonomous Mobile Agents for Older Adults

**arXiv ID:** 2609.04850 | [PDF](https://arxiv.org/pdf/2609.04850v1)

**作者:** Weide Zhan `[一作]` (Fudan University), Tun Lu `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了首个面向老年人自然语言的手机 GUI 代理基准 ElderBench，收集 249 条来自 20 个 Android 应用的真实任务指令，系统地分析其在句法、语义、语用上的差异，并在在线与离线两种模式下评估了多种 LLM/VLM 与 GUI 代理，进一步通过指令归一化、失败分析和特征归因识别出老年人语言与执行瓶颈。

**💡 创新点**

创新点在于：①首次从老年人真实对话中抽取 GUI 任务指令，填补了现有基准缺失的“自然语言”与“老年人特征”；②提出双模式（在线/离线）评估框架，兼顾交互真实性与可重复性；③结合语言学方法系统刻画老年人语言差异，为后续代理设计提供可操作的洞察。

**🔧 技术方法**

主要技术包括：大型语言模型（如 Gemini‑3‑Flash、Qwen3‑VL‑Plus）、视觉语言模型与 GUI 代理（AutoGLM、GUI‑Owl、UI‑Venus 等），指令归一化手段，VLM‑as‑Judge 自动验证机制，以及失败与特征分析框架。

**📊 数据集**

使用数据集：ElderBench（249 条自然老年人指令，涵盖 20 个应用）和对照基准 MobileWorld，此外还对 28 名老年受访者的访谈文本进行语言学标注。

**📈 对比分析**

通过任务成功率（SR）、平均推理延迟（AIL）与代币成本（MCM）三维度比较，在 130 条在线任务中，Qwen3‑VL‑Plus 达到 76.15% SR；在 119 条离线任务中，Gemini‑3‑Flash 最高 27.73% SR；整体最高 SR 仅为 49.80%（Gemini‑3‑Flash），显示当前模型在老年人自然语言上的显著性能瓶颈；归一化后成功率提升约 30–40%。

**⚠️ 局限性**

局限性包括：样本量相对有限（249 条），仅覆盖单轮任务，未充分考察多轮交互；指令多样性仍受受访者群体限制；模型主要瓶颈在意图理解与规划，缺乏针对老年人特定语言的自适应机制；评估主要针对特定 20 个应用，跨应用泛化仍待验证。

---

## 340. MABPD: Multi-Agent Bias Probing & Detection via Structured Argument Debate

**arXiv ID:** 2609.04841 | [PDF](https://arxiv.org/pdf/2609.04841v1)

**作者:** Garvit Joshi `[一作]` (Graphic Era University), Arun Chauhan `[通讯]` (Graphic Era University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种全零训练的多智能体媒体偏见检测框架MABPD，利用三个专门化LLM（偏见、证据、框架）通过结构化辩论达成一致。

**💡 创新点**

核心创新是Structured Argument Debate（SAD）协议，引入非对称举证负担、角色权重投票和后验证，显著提升回收率而不依赖监督数据。

**🔧 技术方法**

技术包括LLM多智能体协同推理、基于证据因子加权的决策、规则化的共识路径、以及不对称的VerifierAgent后置校验。

**📊 数据集**

使用BABE（4,121句）作为主要评测集，并在SemEval 2019 HyperPartisan新闻集（644篇）验证跨域迁移。

**📈 对比分析**

在BABE测试集上实现宏观F1 83.4%，仅比有监督SOTA MAGPIE 84.1%低0.7个百分点；在HyperPartisan上零射击准确率 75.0%，比有监督SOTA 82.2%低7.2个百分点，表明可与监督方法竞争。

**⚠️ 局限性**

局限包括所有代理使用同一LLM导致的潜在共性误差、阈值未在数据上微调、对极端专业化语言的高误报率，以及评测仅覆盖英文、单一主题的BABE数据集。

---

## 341. How do LLMs Evaluate Perceived Moral Agency? Investigating Moral Decision-Making in Human-Artificial Agents Interactions

**arXiv ID:** 2609.05037 | [PDF](https://arxiv.org/pdf/2609.05037v1)

**作者:** Fernanda Mansilla `[一作]` (CNRS@CREATE), Nancy F. Chen `[通讯]` (Institute for Infocomm Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比人类与大型语言模型（LLM）在智能城市情境中对人类与人工代理的道德代理感知（PMA）的评估；

**💡 创新点**

首次将PMA量表扩展至情境化场景，并提出三阶段LLM问卷响应选择与评估流程；

**🔧 技术方法**

使用LLM‑as‑respondent方法、近确定性推理、SAE与PSS等量化指标以及主题分析的解释生成；

**📊 数据集**

人类样本190人（人文系学生）与多款LLM（Phi‑4‑14B、Llama‑3.1‑8B‑Instruct、Llama‑3.2‑11B、InternVL3‑38B）在190道题/8场景下的响应；

**📈 对比分析**

通过SAE衡量与人类评分的对齐度，最佳对齐模型为Llama‑3.1‑8B（SAE≈22.8）；LLM在情境下显著高估人工代理的自主性和道德判断；

**⚠️ 局限性**

局限包括样本单一、仅英文、每位受试者只评估一种代理、情境文本化而非真实交互、图像提示未提升一致性、部分量表信度不足。

---

## 342. Compositional Reward Models for Conditional Medical Image Generation

**arXiv ID:** 2609.05028 | [PDF](https://arxiv.org/pdf/2609.05028v1)

**作者:** Aayush Kumar Tyagi `[一作]` (Indian Institute Of Technology), Mausam `[通讯]` (Indian Institute Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于层次约束传播（HCP）的组合奖励模型（CRM），用于对基于语义掩码的扩散模型（如ControlNet）进行强化学习微调，从而生成符合医学标注要求的高质量合成数据。

**💡 创新点**

创新点在于将图像质量拆解为可验证的、按层次排列的子奖励，并通过HCP保证低层属性（强度、纹理）先满足后才累积高层奖励，避免单一标量奖励掩盖关键缺陷；同时与GDPO结合实现多奖励稳定优化。

**🔧 技术方法**

技术方法包括：ControlNet+Stable Diffusion 2.1预训练、GMM分布匹配、Dice分数、冻结分割器/分类器作为验证器、Hierarchical Constrained Propagation、Group Distributional Policy Optimization（GDPO）等。

**📊 数据集**

实验使用三类医学图像数据集：PanNuke（多类细胞分割）、CeDeM（肠道活检测量）、ISIC（皮肤病诊断）。

**📈 对比分析**

与原始SFT、单一ORM、SUM、MaxMin等基线在三项下游任务中进行比较。CRM在PanNuke上mDice提升5.18、mIoU提升5.18；ISIC上准确率提升10.57；CeDeM上MAE_R下降24%；整体性能显著优于所有基线。

**⚠️ 局限性**

局限性在于需要领域专家为每个任务手工设计子奖励、验证器和层次顺序，缺乏完全通用的自动化方案；在不同数据域上可能需要重新调优。

---

## 343. How a Chatbot's Response Style Shapes a Classroom: A Multi-Agent Simulation of Students Consulting AI

**arXiv ID:** 2609.05018 | [PDF](https://arxiv.org/pdf/2609.05018v1)

**作者:** Rin Tamai `[一作]` (Matsuyama University), Yuya Dan `[通讯]` (Matsuyama University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建多代理虚拟课堂模拟，比较六种聊天机器人回应风格对学生心理状态（压力、幸福、自立、AI依赖、缺课）随时间的影响。

**💡 创新点**

首次将 LLM 生成对话与另一 LLM 评估器相结合，形成可重复的、可解释的仿真框架，探究回应风格对群体心理和社交关系的系统性影响。

**🔧 技术方法**

基于 Python 的 agent‑based 模型，使用 Gemini 2.5 Flash 两次调用：一次生成辅导员回复，一次评估并输出参数变化；并用规则引擎处理学生之间的聊天、争吵、求助等事件。

**📊 数据集**

数据全部为合成：20 名学生代理的五个心理参数在给定区间内随机初始化，课堂交互通过内部规则产生。

**📈 对比分析**

通过 15 天、50 天、阈值降低三种实验设置，比较不同回应风格的最终平均值、最大/最小值以及缺课人数。结果显示：解决方案型风格保持 AI 依赖低、提升自立；肯定型与煽动型显著提高 AI 依赖且导致缺课；其余风格效果介于两者之间。

**⚠️ 局限性**

主要限制包括：(1) 评估器基于 LLM，易受自身偏见影响；(2) 缺乏真实人类数据验证，模拟结果未必对应现实用户；(3) 随机种子和 LLM 的非确定性导致结果波动；(4) 课堂模型过于简化（无个体差异、家庭环境、教师支持等）；(5) 缺课规则为硬性阈值，缺乏现实可解释性。

---

## 344. Algebraic Geometry Codes Approach the Half-Singleton Bound with Constant Field Size

**arXiv ID:** 2609.05017 | [PDF](https://arxiv.org/pdf/2609.05017v1)

**作者:** Neehar Verma `[一作]` (Aalto University), Razane Tajeddine `[通讯]` (Aalto University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

开发了一般化框架，分析随机评估码在插入/删除错误下的编辑距离，并将其应用于 Reed–Muller、Reed–Solomon 与代数几何码。

**💡 创新点**

通过简化的行暴露与零点计数方法，避免了复杂的 LCS 分析，显著降低了 Reed–Solomon 码的场大小要求和 ε 依赖，并首次构造了常数场大小的代数几何码，几乎实现半 Singleton 边界。

**🔧 技术方法**

使用了线性代数与矩阵秩分析、Schwartz–Zippel 约束、行暴露概率论、Riemann–Roch 定理以及 Garcia–Stichtenoth 代数曲线塔的结构。

**📊 数据集**

主要是理论推导，随机选择评估点（即随机挑选码字坐标）作为“数据集”，未使用具体实验数据集。

**📈 对比分析**

通过概率上界证明，随机 Reed–Solomon、RM 及 AG 码在常数字段上以高概率纠正约 (1‑ε)n‑2k+1 个插入/删除错误，性能显著优于以往 2^{O(1/ε²)} 的依赖，并逼近半 Singleton 上限。

**⚠️ 局限性**

局限在于仅给出存在性/随机构造，缺乏对应的高效解码算法；对固定小字段的最优实现仍未完成；且对非线性码的改进有限。

---

## 345. Has MIMO decoding been proved hard from lattice problems?

**arXiv ID:** 2609.05013 | [PDF](https://arxiv.org/pdf/2609.05013v1)

**作者:** Yang Li `[一作]` `[通讯]` (Deakin University), Yang Li (Deakin University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文对Dean‑Goldsmith提出的从格子问题到MIMO解码的多项式时间归约进行了理论审查，指出其证明缺失、分布不匹配等关键问题，导致所宣称的MIMO硬度根本无法得到证明。

**💡 创新点**

创新点在于系统地识别并归纳归约证明中断裂的步骤，揭示了非模数MIMO设置与Regev LWE归约之间的根本结构差异，从而为未来修正或替代方案提供了明确的技术障碍。

**🔧 技术方法**

主要采用了格子理论、LWE归约框架、MIMO信道模型以及分布相似性（统计距离）分析等数学与信息论工具。

**📊 数据集**

本研究不依赖任何实验数据或公开数据集，全部基于理论推导和文献综述。

**📈 对比分析**

与原始归约及相关攻击论文对比，作者通过理论上对比说明缺陷，未进行实验性能评估；因此无法给出数值级的性能或效果对比。

**⚠️ 局限性**

主要限制在于：① 证明缺失导致无法确认MIMO解码的格子难度；② 现有的改进尝试（如限制BBDD、模块化公共成分等）都未能完全解决分布不匹配问题；③ 结论仅适用于当前的Dean‑Goldsmith构造，对其他可能的物理层安全方案仍保持开放性。

---

## 346. Beyond Homoscedasticity: Decoupled Uncertainty Optimization for Deep Imbalanced Regression

**arXiv ID:** 2609.04995 | [PDF](https://arxiv.org/pdf/2609.04995v1)

**作者:** Juncheng Zhou `[一作]` (Wuhan University), Jingsong Cui `[通讯]` (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DUO框架，利用条件高斯分布对深度不平衡回归任务中的样本不确定性建模，并通过解耦均值与方差优化与分布引导的对比学习提升稀疏尾部性能。

**💡 创新点**

核心创新在于①使用实例级异方差高斯建模并通过停止梯度实现均值与方差的计算解耦；②引入不确定性加权的均值损失以及基于Bhattacharyya系数的分布式正负样本分配，实现对尾部样本的动态强化。

**🔧 技术方法**

采用条件高斯回归、负对数似然、均值与方差的停止梯度解耦、分布加权的均值损失、分布指导的InfoNCE对比学习以及ResNet‑50骨干网络。

**📊 数据集**

在四大基准上进行实验：AgeDB‑DIR、IMDB‑WIKI‑DIR（人脸年龄回归）、NYUD2‑DIR（室内深度估计）以及AAV2‑DIR（蛋白变异活性预测）。

**📈 对比分析**

与多种基线（Vanilla MSE、LDS、FDS、RankSim、ConR、Balanced MSE、DistLoss）进行对比；在少量样本区的bMAE、GM、MAE等指标上取得最优或近优成绩，尤其在AgeDB、IMDB和AAV2的少样本/尾部评估中明显领先。

**⚠️ 局限性**

在部分多数样本或整体指标上略有退步，且对超参数λ和warm‑up阶段较为敏感；模型对极端尾部样本的处理仍受训练稳定性与数据分布影响，需进一步探索更稳健的解耦与对比策略。

---

## 347. A Tree-based RAG Framework for Evidence-Intensive QA via Adaptive Planning and Topology-Aware Evidence Gathering

**arXiv ID:** 2609.04981 | [PDF](https://arxiv.org/pdf/2609.04981v1)

**作者:** Songeun Lee `[一作]` (Korea University), Woohwan Jung `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自适应规划与拓扑感知证据收集的RAG框架（APT‑RAG），用于处理需要在数十到数百篇文档中聚合证据的证据密集型问答任务。

**💡 创新点**

创新点在于：①动态规划能根据问题依赖和证据需求递归扩展推理树；②拓扑感知证据收集通过侧向收集、外部检索和垂直聚合三种策略高效利用子树、兄弟节点与父节点的证据；③基于证据相似度的批量答案生成显著降低LLM调用量和推理延迟。

**🔧 技术方法**

使用了深度优先的递归推理树、答案可达性检查器与分解器、语义检索（基于Qwen3-Embedding-0.6B）、LLM生成（Qwen3-4B/30B）、以及自定义的聚类算法（将子问题聚类转换为最小团划分）。

**📊 数据集**

在两个证据密集型问答基准上进行实验：MoNaCo（平均43.3页证据）和QAMPARI（平均13.0页证据）。

**📈 对比分析**

与LLM‑Only、Naïve RAG以及多种树/图结构RAG基线（ToQ、RT‑RAG、Plan*RAG、LogicRAG）对比，APT‑RAG在答案F1和检索召回上均实现显著提升（例如MoNaCo 30B设置下F1提升8%，召回率从39.28%提升至50.79%），且推理延迟比基线低41%。

**⚠️ 局限性**

局限性包括：对规划模块的依赖，若分解器或可达性检查错误会导致无效分支或错误传播；随着证据量和树分支增大，推理成本仍然上升，需进一步通过树剪枝或证据压缩技术优化。

---

## 348. BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference

**arXiv ID:** 2609.04971 | [PDF](https://arxiv.org/pdf/2609.04971v1)

**作者:** Janghyeon Kim `[一作]` (Hanyang University), Jungwook Choi `[通讯]` (Hanyang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种无训练的 KV 缓存压缩方法 BeaconKV，利用 beacon 查询代表全局关注的关键上下文，减少大型推理模型的内存占用。

**💡 创新点**

创新点在于提出 Thought Revisiting Tokens 概念，发现全局查询在嵌入空间形成少数聚类，可用 beacon 查询代替完整查询历史；并采用持续 FPS 在线采样动态获取代表性查询。

**🔧 技术方法**

使用的技术包括 KV 缓存压缩、RoPE 对齐、基于最大池化的注意力权重评分、持续 FPS 采样、Group Query Attention（GQA）等。

**📊 数据集**

使用的数据集涵盖数学（AIME24、MATH‑500）、编程（LiveCodeBench）和自然科学（GPQA‑Diamond）等多领域推理基准。

**📈 对比分析**

通过与 RPC、SnapKV、R‑KV 等基线对比，在四个开放源代码大型推理模型上评估，低预算下准确率提升高达 31.7%，内存压缩可达 5.8×，吞吐量提升 4.3×。

**⚠️ 局限性**

局限性包括依赖查询聚类特性，对不同模型或任务的适用性可能有限；方法仅针对推理阶段，不改模型结构，且对生成质量的细粒度影响尚未全面评估。

---

## 349. ARC-Loc: Leveraging Azimuthal Ray Convergence as a Geometric Cue for Direct Cross-View Localization

**arXiv ID:** 2609.04965 | [PDF](https://arxiv.org/pdf/2609.04965v1)

**作者:** Hyeongsik Kim `[一作]` (Hanyang University), Je Hyeong Hong `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于极角射线收敛（ARC）的跨视角定位框架ARC-Loc，直接在二维地面图像与卫星图像之间匹配关键点，生成射线约束并通过闭式两点最小求解器估计相机位置，完全不依赖BEV变换或外部深度模型。

**💡 创新点**

创新点包括：① 通过“重定位（resection）”原理将地面-卫星对应关系转化为二维线-点约束；② 提出ARC Solver，实现仅需两条射线即可闭式求解相机位置，并兼容RANSAC；③ 设计ARC Loss，利用射线收敛性对特征匹配网络进行无监督几何正则化。

**🔧 技术方法**

核心技术包括：基于RADIOv3-L视觉基础模型的特征提取；双软最大化匹配概率并结合关键点置信度的对应选择；生成极角射线并构建线-点距离最小化；闭式线性最小二乘求解；RANSAC鲁棒估计；ARC Loss对网络进行端到端训练。

**📊 数据集**

使用VIGOR（城市全景与卫星图像）和KITTI（前视相机与卫星图像）两个公开数据集进行实验，分别评估已知方向与带噪声方向条件下的位置误差、召回率等指标。

**📈 对比分析**

与多种BEV和非BEV基线（如GGCVT、DenseFlow、HC-Net、FG^2、BevSplat、Loc^2、CCVPE）比较，ARC-Loc在VIGOR Same-Area下的平均位置误差为2.52 m，处于非BEV方法的第二名；在KITTI Same-Area下位置误差0.79 m、召回率99.87%/100%，与前沿方法相当。推理时间与内存更低，匹配阶段延迟≈59 ms，内存≈906 MB。

**⚠️ 局限性**

局限性：① 在方向误差超过约±10°时位置精度下降明显；② 依赖射线方向多样性，前视摄像机等视野受限场景下射线聚集导致几何退化，影响定位稳定性。

---

## 350. GreenPipe: Power Modeling for Containerized DNN Inference on Kubernetes Edge Nodes

**arXiv ID:** 2609.04952 | [PDF](https://arxiv.org/pdf/2609.04952v1)

**作者:** Mengxue Wang `[一作]` (Universitat Politècnica de Catalunya), Jordi Guitart `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并验证了一套名为 GreenPipe 的数据驱动多资源功耗建模与在线估计管道，专门用于 ARM 边缘节点上的容器化 DNN 推理。

**💡 创新点**

创新点在于：① 结合外部功率计标注的多资源监控数据，构建回归功耗模型；② 通过比例归因实现容器级功耗可视化；③ 在 Kubernetes ARM 边缘环境中实现 1Hz 实时功耗估计，显著提升了 CPU‑only 模型的 MAPE（从 35% 降至 6–9%）。

**🔧 技术方法**

使用技术包括：Kubernetes/K3s、Kepler（自定义 ARM 版本）+ eBPF、Prometheus、scikit‑learn 回归器（LR、RF、XGB 等）、LiteRT/TensorFlow Serving、外部 USB 功率计、DeepBench、stress‑ng、Docker/Containerd、Python 及 Go 开发。

**📊 数据集**

所用数据集：ImageNet 验证集用于推理实验；DeepBench、stress‑ng 等基准作训练和验证；MLPerf 校准集用于量化模型。

**📈 对比分析**

通过与仅 CPU 压力 + 利用率基线模型对比，GreenPipe 在验证推理工作负载上平均 MAPE 降低约 26.9%，MAE 降低 46.9%。在不同模型、精度、线程数、推理引擎和服务模式下，模型保持 6–9% 的 MAPE，显示出稳健的性能。

**⚠️ 局限性**

局限性：仅在单一 Raspberry Pi 4 上评估；容器级功耗归因为启发式且未获得真实验证；未覆盖 GPU、不同 DVFS 设置或多节点大规模部署；在线部署缺乏同步标注机制。

---

## 351. Learning 3D Editing without Paired Supervision via Generative Prior Distillation

**arXiv ID:** 2609.04942 | [PDF](https://arxiv.org/pdf/2609.04942v1)

**作者:** Hao Wen `[一作]` (Beihang University), Lu Sheng `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为PriorEdit3D的端到端3D编辑框架，能够在没有配对3D监督的情况下，通过生成先验知识蒸馏实现高质量、跨视角一致的3D编辑。

**💡 创新点**

创新点包括：① 采用无配对（unpaired）训练，通过从强大基础模型中蒸馏视觉、语义和几何先验实现编辑；② 在训练中结合2D图像编辑教师、VLM语义反馈和3D分布匹配正则化，弥补单视角监督导致的几何崩塌和视角不一致；③ 通过分离源视角的视觉先验与全局语义先验，获得mask‑free、快速的前向编辑。

**🔧 技术方法**

技术手段包括：
- 差分渲染（3D Gaussian Splatting）用于将3D表示映射为可微的2D图像；
- 2D图像编辑教师（如Qwen-Image-Edit）提供目标视角的编辑图像；
- Vision‑Language Model（Qwen3‑VL）在辅助视角提供“指令遵循”和“身份保持”双重语义反馈；
- 3D分布匹配（Distribution Matching Distillation）正则化，使编辑结果保持在预训练的UniLat3D生成的3D资产流形上；
- 采用UniLat3D预训练模型的Token Concatenation 方式对源3D隐码与编辑指令进行条件化。

**📊 数据集**

数据集：
- 73,121条高质量的2D编辑样本（源‑编辑‑指令三元组），从Objaverse对象渲染并通过Gemini3‑Flash生成指令，再用Qwen-Image-Edit进行编辑；
- 过滤后保持多样性，包含部分/整体编辑、颜色/形状/姿态等多种操作。
- 评测集包括自建130条分布内样本以及来自ABO和GSO的236/239条分布外样本。

**📈 对比分析**

对比方法：EditP23、Instant3DiT、3DEditFormer、VoxHammer、Nano3D。实验表明PriorEdit3D在PSNR、SSIM、LPIPS、FID、FVD、CLIP‑T、LLM‑Id、LLM‑Inst等指标上均显著优于基线；同时推理时间仅约7 s（相比VoxHammer 133 s、Nano3D 14 s），展示出更快、更高质量、更跨视角一致的编辑性能。

**⚠️ 局限性**

局限性：
- 对精细、小尺寸或高频细节的编辑效果有限，主要关注全局一致性；
- 受限于2D编辑教师的错误，易将错误传播到3D模型；
- 受UniLat3D先验与可微渲染的限制，难以处理大幅姿态、拓扑大变形或与非可微3D架构兼容的场景；
- 训练与推理仍需GPU资源，且对高分辨率渲染的支持尚不充分。

---

## 352. LensStyle: Learning the Optical Aesthetics for Controllable Stylized Lens Effect Rendering

**arXiv ID:** 2609.04939 | [PDF](https://arxiv.org/pdf/2609.04939v1)

**作者:** Yachuan Huang `[一作]` (Huazhong University of Science and Technology), Zhiguo Cao `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LensStyle框架，实现对相机镜头风格的连续与离散可控渲染。

**💡 创新点**

首次将焦距、散景强度的连续参数与光圈形状、星光等离散风格分离控制，并采用流匹配进行生成。

**🔧 技术方法**

基于流匹配的生成模型、Dual Path Controller、轻量MLP特征调制、跨注意力以及VAE编码解码技术。

**📊 数据集**

构造了MultiLens数据集，包含3k场景、30万对焦距/散景对和星光样本。

**📈 对比分析**

与MPIB、BokehMe、Dr.Bokeh、BokehDiff、UltraEdit、SuperEdit等基线进行PSNR/SSIM/LPIPS对比，在多种风格下均显著优于基线。

**⚠️ 局限性**

无法模拟不同品牌相机的特定镜头风格，需要进一步扩展。

---

## 353. Solving Hard XAI Queries Based on a Compiled Dual-Rail Encoding

**arXiv ID:** 2609.04931 | [PDF](https://arxiv.org/pdf/2609.04931v1)

**作者:** Arthur Ledaguenel `[一作]` (CRIL, Université d'Artoi), Jean-Marie Lagniez `[通讯]` (CRIL, Université d'Artoi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了可解释人工智能（XAI）中对布尔分类器的可解释性查询（如最短解释、最优解释、计数与枚举等）的复杂性，并证明即使在最易处理的知识编译语言——有序二叉决策图（OBDD）上，许多此类查询仍然是NP‑难的。随后，作者提出使用“dual‑rail”编码将原始分类器转化为双重变量形式，并利用该形式在 d‑DNNF 及其结构化子类上实现了多种高效的算术模型计数与枚举算法，能够在多种解释度量（加权、分层优先级、覆盖度）下快速给出最优或最短解释，并且可以计算基于归纳解释的 Shapley 值。实验结果表明，双重编码的 CNF 在编译时间、模型数和 d‑DNNF 大小上与原 CNF 相当，甚至在某些实例上仅通过双重编码即可完成求解。

**💡 创新点**

创新点：
1) 证明了即使是最易处理的 OBBD 形式，对多种 XAI 查询（最短/最优解释、计数、枚举）仍保持 NP‑难性。
2) 引入 dual‑rail 编码，将布尔函数的所有含义项（implicants）与模型一一对应，打开了在 d‑DNNF 上执行算术模型计数的可能性。
3) 在 d‑DNNF 结构上实现了多种优化与计数算法（加权、分层、覆盖、Shapley 归纳解释），并给出了对应的多查询可归约关系与复杂度分析。
4) 通过结构共享（dual‑rail 与域约束共享 vtree）实现了覆盖基解释的多项式时间判定。
5) 提供了实验验证，展示了双重编码在编译与查询效率上的优势。

**🔧 技术方法**

主要技术：
- Dual‑rail 编码（将每个变量映射为正、负两份，强制纯模型对应于 implicants）。
- 知识编译：将 dual‑rail CNF 编译为 d‑DNNF、d‑dec‑DNNF 或结构化 d‑DNNF，以支持多项式时间的条件化、计数与枚举。
- 算术模型计数（AMc）与优化半环框架，用以统一处理最优解释、计数、Shapley 值等。
- 结构化 DNNF 与 vtree 共享，实现覆盖基解释的高效判定。
- 经典 NP‑难性与 Σ₂‑P 难度证明（通过 QSAT₂ 降约）以及多查询可归约图。

**📊 数据集**

使用的实验数据集：
- 标准 CNF 竞赛/工业案例（大小范围从几百到数千变量，数十至数百条子句）。
- 通过对每个 CNF 生成其 dual‑rail 版本，构成实验对照组。

**📈 对比分析**

比较方法与性能：
- 对照实验：分别编译原 CNF 与其 dual‑rail 版本，记录编译时间、生成的 d‑DNNF 大小（边数）以及可解实例数。
- 结果显示：
  * dual‑rail 编译时间略高，但在某些实例中因结构更适合 OBDD/ d‑DNNF 而显著提升可解率；
  * d‑DNNF 大小与原 CNF 相近，甚至更小；
  * 在对最短/最优解释的查询上，dual‑rail 实现可在多项式时间内完成，原 CNF 在 OBBD 上则仍需 NP‑难搜索。
- 进一步实验表明，使用 dual‑rail 编译的 d‑DNNF 在枚举“足够理由”时可以实现增量多项式延迟。

**⚠️ 局限性**

局限性：
1) 对于 OBBD 形式，仍无法避免 NP‑难的最短/最优解释计算，dual‑rail 只能在 d‑DNNF 之上提供多项式解法；
2) dual‑rail 编码会使公式规模翻倍（每个变量产生两份），在极大规模实例上可能导致内存瓶颈；
3) 仅在 CNF 形式下实验，尚未验证对非 CNF 公式（如通用逻辑表达式）的适用性；
4) 某些覆盖基解释的判定虽然是多项式，但对 vtree 结构的共享要求较高，若无法共享则需额外成本；
5) Shapley 归纳解释的计算依赖于对解释大小计数的精确实现，若 d‑DNNF 近似或不完整则结果不准确。

---

## 354. Artificial Intelligence in Equity and Crypto Markets: Progress, Profitability Evidence, and the Limits of Automated Investing

**arXiv ID:** 2609.04917 | [PDF](https://arxiv.org/pdf/2609.04917v1)

**作者:** Linsen Zhu `[一作]`, Mengqing Cai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对截至2026年8月31日公开的人工智能在投资工作流中的研究进行批判性综述，构建了“alpha‑translation chain”与多维证据配置，系统评估了不同 AI 方法（机器学习、时间序列预训练模型、自然语言处理/大型语言模型、强化学习与代理系统）在股票/ETF、中心化加密货币现货、永续合约与链上市场等多种资产层面的表现与局限。

**💡 创新点**

创新点在于：①提出 alpha‑translation chain 将信息获取、模型表示、投资决策、执行与净 alpha 四步连贯化；②制定多维证据配置（T、S、P、I、R、X、O）来客观评估研究的经济有效性；③构建跨市场结构化评估矩阵，阐明每种市场特定的风险与成本维度；④提出针对 AI 投资的未来研究议程与治理框架。

**🔧 技术方法**

使用的技术包括：传统机器学习与深度学习模型（树模型、神经网络、时间序列预训练模型）、自然语言处理与大型语言模型（BloombergGPT、FinGPT）、强化学习与直接策略优化、以及集成检索、记忆、推理和工具调用的 LLM 代理系统。

**📊 数据集**

主要数据集涵盖：公开的股票与 ETF 价格与公司基本面、新闻文本、加密货币现货与永续合约交易数据、链上交易与 AMM 状态，以及对比基准的宏观经济与行业因子。

**📈 对比分析**

比较方法通过多维证据配置对比不同研究的时间有效性、选择偏差、组合映射、实现成本、风险与基准、外部有效性与操作来源。结果显示：在历史 OOS 设定下，非线性 ML 与直接策略可提升回测表现，但在扣除交易成本、容量限制与未来可持续性后，公开证据未能证明任何通用 AI 架构能够持续产生跨市场、跨周期的净 alpha；而在预训练时间戳、成本敏感目标与代理系统等方面的研究仍处于探索阶段。

**⚠️ 局限性**

主要局限包括：时间戳失效与前视偏差、过度搜索与选择偏倚、成本与市场影响未充分建模、容量与流动性问题、缺乏前瞻性与实盘验证、不同市场结构导致的评估不一致、以及缺少可公开复制的高质量实盘数据。

---

## 355. Minimum Schubert Codewords and Second-Minimum Grassmann Codewords

**arXiv ID:** 2609.04916 | [PDF](https://arxiv.org/pdf/2609.04916v1)

**作者:** Muskan Khaneja `[一作]` (Indian Institute of Technology Jammu), Prasant Singh `[通讯]` (Indian Institute of Technology Jammu)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

论文对Schubert码的最小重量码字进行分类，并用该分类推导Grassmann码的二次最小重量码字，并给出其枚举公式。

**💡 创新点**

创新点在于完全解决了Schubert码最小重量码字的Minimum Distance Conjecture，并首次给出了Grassmann码二次最小重量码字的几何描述和计数结果。

**🔧 技术方法**

采用了坐标无关的几何方法、Bruhat序、Plücker映射与外代数分解、归纳证明以及组合分解等技术。

**📊 数据集**

该工作不依赖具体实验数据，全部为理论证明和组合计数。

**📈 对比分析**

由于为理论性研究，没有实验对比；通过证明得到的重量与计数结果与已知极限与前人结果相符，性能表现为完全正确且可推导。

**⚠️ 局限性**

局限性在于仅处理Schubert码与Grassmann码的最小及二次最小重量，未给出更高重量码字的结构或算法实现。

---

## 356. Amortizing Scaling Law Construction Costs

**arXiv ID:** 2609.05016 | [PDF](https://arxiv.org/pdf/2609.05016v1)

**作者:** Abhash Kumar Jha `[一作]` (ELLIS Institute Tübingen), Aaron Klein `[通讯]` (ELLIS Institute Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种利用贝叶斯优化收集稀疏数据并通过 surrogate 幻想与仿真网格来高效构建大型模型缩放律的方法，显著降低计算成本。

**💡 创新点**

将缩放律构建视为计算窗口递增的贝叶斯优化问题，并结合 surrogate 幻想来重建稠密实验网格，实现10–100倍的计算节省。

**🔧 技术方法**

使用贝叶斯优化（高斯过程 surrogate + LCB 采集）、compute slicing、surrogate 幻想、仿真网格拟合以及多指标评估。

**📊 数据集**

在现有大型语言模型缩放实验数据（如 OELLM-English 等）预定义的网格上进行模拟验证。

**📈 对比分析**

与全网格搜索（Full）和不同 compute 窗口设置下的贝叶斯优化进行对比，使用参数误差、MSE、覆盖率和外推误差等多指标评估；结果显示在仅占总预算 1–10% 的情况下即可获得接近全网格的参数，计算加速达到 10–100 倍。

**⚠️ 局限性**

仅在模拟数据集上验证，缺乏在全新大规模真实实验中的实证；对 surrogate 模型假设、kernel 与 acquisition 的专业化仍需进一步研究。

---

## 357. PuTR-CouT: Counting-by-Tracking in Camera-Trap Image Sequences

**arXiv ID:** 2609.05038 | [PDF](https://arxiv.org/pdf/2609.05038v1)

**作者:** Fagner Cunha `[一作]` (Federal University of Amazonas), Eulanda M. dos Santos `[通讯]` (Federal University of Amazonas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了PuTR-CouT框架，利用Transformer基的多目标追踪器和弱监督合成追踪标签，对低帧率相机陷阱序列进行计数，并支持多物种计数与轨迹可验证；

**💡 创新点**

①基于相机陷阱结构先验生成合成追踪标签，解决缺乏标注的难题；②在PuTR中去除几何补偿，仅使用注意力关联，提高低帧率下的鲁棒性；③实现轨迹级别计数，可处理同序列多物种并支持跟踪可视化验证；

**🔧 技术方法**

MegaDetector检测+Swin‑B分类+MegaDescriptor重识别特征+Pure Transformer (PuTR)追踪+负二项分布合成数据+MaxBoxCount改进基线；

**📊 数据集**

iWildCam 2021计数挑战数据集，生成的Synthetic Seq (≈75k)作为训练；

**📈 对比分析**

与iWildCam 2021挑战Top 3、改进MaxBoxCount、ByteTrack、BoT‑SORT等方法对比，PuTR‑CouT在private MCRMSE 0.024864（比BoT‑SORT下降≈9%）和public 0.025070（优于改进MaxBoxCount 0.025470）表现最佳；

**⚠️ 局限性**

依赖合成标签的噪声与分布假设，易受外观变化影响，训练需要大量合成数据，且对实时视频处理尚未验证。

---

## 358. Reducing the Cross-Model Tax: Query Optimization over Multi-Model Data

**arXiv ID:** 2609.05014 | [PDF](https://arxiv.org/pdf/2609.05014v1)

**作者:** Jáchym Bártík `[一作]` (Charles University), Irena Holubová `[通讯]` (Charles University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在MM‑quecat多模型查询处理器上实现并评估了一套基于映射和能力的优化管道，显著降低跨模型查询的延迟。

**💡 创新点**

创新点在于将谓词下推、交叉模型依赖联接与非冗余查询部分构造三种经典优化原则统一应用到概念层、映射层与原生执行层之间，实现跨模型、跨系统的协同优化。

**🔧 技术方法**

主要技术包括：映射感知谓词下推、跨模型依赖联接（dependent join）、强制性模式的批量添加、轻量级查询驱动基数估计、MMQL概念查询语言与MM-cat映射框架的结合。

**📊 数据集**

使用从Cal.com开放源代码调度应用衍生的70个对象、90个关系的概念模式，映射至PostgreSQL、MongoDB和Neo4j共计约200万条记录的合成数据集。

**📈 对比分析**

通过在单机Docker部署下对20条MMQL读查询进行基准测试，比较四种配置（原始、加谓词下推、再加依赖联接、全部优化）在计划、原生执行+传输、统一层处理和结果构造四个阶段的耗时；结果显示谓词下推可降低最高两阶，依赖联接进一步提升约一阶，非冗余构造在图形查询中把规划时间从约600 ms压至毫秒级，整体查询延迟从几秒下降至几百毫秒。

**⚠️ 局限性**

主要限制包括：仅覆盖SELECT‑PROJECT‑JOIN带过滤的查询；使用合成事务性数据和单机实验，未覆盖分布式部署、写操作和复杂聚合；轻量级基数估计与非冗余构造仅处理强制性模式；未实现完整的全局成本模型或动态映射选择。

---

## 359. Solution-space heterogeneity shapes federated learning dynamics across partial differential equations

**arXiv ID:** 2609.05012 | [PDF](https://arxiv.org/pdf/2609.05012v1)

**作者:** Ping Luo `[一作]` (National University of Defense Technology), Dongsheng Li `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于解空间的分布式学习协议（Solution‑space PDE‑Dirichlet），通过将连续 PDE 解离散成伪类并使用 Dirichlet 分配实现可复制的非 IID 客户端分配，并将分配效果量化为解空间的 optimal‑transport 距离。

**💡 创新点**

创新点在于：①将解空间作为统一的响应维度进行离散化，消除对特定 PDE 输入空间划分的依赖；②证明 Dirichlet 集中参数 α 与解空间异质性、梯度不一致、参数漂移之间的理论关联；③在七个受控与公开 PDE 任务上系统验证该协议，并与输入空间距离进行对比。

**🔧 技术方法**

主要技术包括：Neural Operator（DeepONet、FNO、InversionNet）训练；FedAvg、FedProx、SCAFFOLD 等 Federated Learning 算法；k‑means 解决方案离散化；Symmetric Dirichlet 分配与整数多项式采样；离散 optimal‑transport（Wasserstein‑1）度量；统计检验与 Spearman/ Pearson 相关性分析。

**📊 数据集**

使用的数据集：受控实验生成的 Antiderivative、Diffusion‑Reaction、Viscous Burgers；公开 benchmark 包括 PDEBench（Darcy Flow、Shallow Water）、CFDBench（Cylinder Flow）和 OpenFWI（FlatVel‑A）。

**📈 对比分析**

比较方法：将非 IID FedAvg 与同一随机种子下的近 IID（α=100）FedAvg 进行配对对照，记录解空间距离、梯度不一致、参数偏差及最终相对 L2 错误；在不同 α、客户端数、优化器、Viscosity 维度上做 ablation；与输入空间 Wasserstein‑1 距离做对比。实验结果显示：较低 α 产生更大的解空间异质性、梯度漂移和参数偏差；在低粘度 Burgers、低 Viscosity 领域表现最为显著，导致最终误差提升 3‑5 个百分点；其他任务的误差提升有限，且可通过增加通信轮次或使用更平滑的 PDE 解决。

**⚠️ 局限性**

局限性包括：①离散化使用欧氏 k‑means，可能未捕获所有 PDE 物理几何；②对解空间的 Dirichlet 分配只在有标记响应的离线场景可行；③整数采样与最小样本修复会导致 α 与实际异质性偏离；④理论假设（梯度可辨识、曲率梯度对齐）在某些 PDE 上不一定满足；⑤实验以仿真生成数据为主，缺乏真实跨机构分布的验证。

---

## 360. MIVAIS: A Study Environment for Multi-Agent Mixed-Initiative Visual Analytics Applications

**arXiv ID:** 2609.04983 | [PDF](https://arxiv.org/pdf/2609.04983v1)

**作者:** Tobias Stähle `[一作]` (ETH Zürich), Mennatallah El-Assady `[通讯]` (ETH Zürich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了MIVAIS，一个支持多智能体混合主动视觉分析系统的完整研发与评估平台，集成了标准化的基础设施与多模态实验环境；

**💡 创新点**

创新点在于提供了双层抽象——一层负责统一的世界状态同步、权限与通信机制，另一层负责声明式实验配置与自动化多模态日志；

**🔧 技术方法**

核心技术包括Python/TypeScript基础设施、WebSocket实时同步、FastAPI后端、MongoDB日志存储、Web Audio/Screen Recorder、Whisper语音转写与LLM（Gemini）推理；

**📊 数据集**

采用了Podium的汽车属性数据集、Voyager 2的通用数据探索数据以及ProactiveVA的社交事件分析数据集进行系统重现与验证；

**📈 对比分析**

通过在MIVAIS上重现三种现有混合主动系统并在专家案例研究中快速部署实验，证明了框架在实现效率、日志完整性和实验可重复性方面优于传统手工搭建，性能表现主要体现在开发周期缩短和数据同步一致性提升；

**⚠️ 局限性**

局限性包括仅支持桌面Web端、缺乏VR/AR与移动端支持、部分浏览器对Web Bluetooth/传感器API的兼容性不足，以及需要研究者在应用与实验两层模型之间保持一致性，导致一定的认知负担和可视化融合功能尚不完善。

---

## 361. RefDiT: Local Attribute Guidance in Reference-Based Image Generation

**arXiv ID:** 2609.04976 | [PDF](https://arxiv.org/pdf/2609.04976v1)

**作者:** Rameshwar Mishra `[一作]` (Indraprastha Institute of Information Technology Delhi), A V Subramanyam `[通讯]` (Indraprastha Institute of Information Technology Delhi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 RefDiT，利用参考图像中局部属性引导生成新图像，支持对多对象场景的细粒度控制。

**💡 创新点**

创新点在于：①使用多模态大语言模型提取属性三元组并生成属性级标识符；②通过三元组条件的 LoRA 训练，使 DiT 的注意力在局部区域内实现对应；③在训练中引入区域一致性损失，确保标识符聚焦正确局部。

**🔧 技术方法**

技术手段包括：Diffusion Transformer (DiT) + LoRA 适配、基于多模态注意力的交叉注意力机制、三元组条件 LoRA 训练、GPT‑4o/InternVL 进行属性三元组提取与选择。

**📊 数据集**

数据集：构造了约 500 张评估图像，包含 60 张多对象参考图像，并扩展单对象数据集加入更多对象；使用 Stable Diffusion 3.5 和 Flux 两大 DiT 模型进行实验。

**📈 对比分析**

与 LoRA‑个性化、编辑方法以及商业模型（Gemini‑Banana、GPT‑5）对比，RefDiT 在属性匹配（Attr‑Match 0.88）和属性相似度（Attr‑SIM 0.70）上显著优于开源基线，并与商业模型相当；CLIP‑I/T 也保持一致。

**⚠️ 局限性**

局限性：依赖大语言模型的三元组提取质量，提取错误会影响结果；评估指标受限于语言模型提取的词表，可能忽略细微视觉差异；在极端遮挡或罕见对象场景下仍易出现偏差。

---

## 362. SwanWeave:One-Stage Multi-Task Instruction-Guided 3D Spatial Audio Editing

**arXiv ID:** 2609.04975 | [PDF](https://arxiv.org/pdf/2609.04975v1)

**作者:** Ke Lei `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种一阶段的指令驱动3D FOA空间音频编辑框架SwanWeave。

**💡 创新点**

提出SE-MoE双层路由和SPO直接偏好优化，支持单/复合编辑并保持原场景。

**🔧 技术方法**

利用流匹配扩散变换器、FOA VAE编码、双层路由Mixture-of-Experts、DPO偏好训练和阶梯CFG引导等技术。

**📊 数据集**

基于AudioCaps、FSDKaggle2019、PicoAudio、LibriSpeech、Spatial LibriSpeech等开源语音/音效数据，并用可控房间仿真生成FOA配对。

**📈 对比分析**

与AudioEditor、ZETA、SDEdit、SmartDJ等基线在FD、FAD、KL、LSD、CLAP以及空间指标（GCC、CRW、FSAD）上对比，SwanWeave在所有指标上均优于基线，且推理速度最快。

**⚠️ 局限性**

训练数据为仿真FOA，缺乏真实录音的复杂声学；场景时长与事件数有限，难以处理长时段高密度混合场景。

---

## 363. Why We Care About Understanding: Competence through Predictive Compression

**arXiv ID:** 2609.04962 | [PDF](https://arxiv.org/pdf/2609.04962v1)

**作者:** Matthieu Queloz `[一作]` (University of Bern), Pierre Beckmann `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

构建了一个将哲学关于理解的解释与信息论/机器学习中压缩观统一的框架，提出代理假说和CPC（Competence through Predictive Compression）模型，阐释理解是对领域关系结构的结构敏感预测模型，并解释人类理解趋向原则性简洁的社会动力

**💡 创新点**

① 将理解与压缩的关系从“同一”转为“相互映射”的视角，提出理解是压缩的表象而非等价；② 用社会功能（信任与教学）解释理解概念的出现，并将其四大压力（可预测性、可存储/可操作性、可演示性、可传递性）统一为CPC框架；③ 对传统哲学中的“phronesis”等隐性理解提供可度量的评判标准

**🔧 技术方法**

信息论中的最小描述长度（MDL）原理、预测-压缩关联分析、对已有哲学、计算机科学文献的系统性综述与理论构造

**📊 数据集**

无具体实验数据集，论文主要以理论推导与历史案例（开普勒、牛顿等）为支撑

**📈 对比分析**

无定量实验对比；通过案例演示与逻辑推导说明CPC框架在解释理解与压缩关系、社会功能及压缩极限方面的解释力

**⚠️ 局限性**

① 缺乏经验/实验验证，理论框架难以直接检验；② 对非形式化（如phronesis）或深层隐性理解的处理仍较简化；③ 对复杂社会机制和跨文化差异的细致建模不足，导致在某些情境下的适用性不确定

---

## 364. Discourse Dependency: A Continuous Criterion for Translation Difficulty

**arXiv ID:** 2609.04959 | [PDF](https://arxiv.org/pdf/2609.04959v1)

**作者:** Ahrii Kim `[一作]` (AI-Bio Convergence Research Institute), Seong-heum Kim `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了无指标、源侧的语篇依赖度（discourse dependency）来衡量句子对上下文的依赖距离，并验证其与黄金共指的高度相关；在WMT24++与WMT25上对其分布进行统计；通过英文-韩文自动纠错（APE）实验比较了五种上下文注入策略与人类评判的差异；发现人类对“无上下文”策略在高依赖度句子上的偏好随依赖度递增，而自动评测指标对依赖度几乎无感；模型在获取完整文档上下文时仍未能有效利用远程信息。

**💡 创新点**

创新点在于提出一种语言学根植、无指标、仅基于实体再提及和代词共指的“语篇依赖度”度量，为MT难度提供连续、文档级、与领域标签解耦的评估轴；将该度量与文档级评测相结合，揭示自动指标对长程依赖的盲点；以及基于该度量对上下文注入策略进行分层分析。

**🔧 技术方法**

使用spaCy/Flair进行NER、POS、依赖分析以抽取实体与代词；构造规则1、规则2计算距离；采用LLM（包括OpenAI GPT系列、Anthropic Claude等）进行APE；使用低阶回归（LOWESS）和Kendall τ、Wilcoxon等统计方法进行结果分析。

**📊 数据集**

使用WMT24++与WMT25的英文-韩文以及英文-中文数据集，涵盖文学、新闻、社交三大领域，重点对文档级段落进行语篇依赖度计算与评测。

**📈 对比分析**

与人类评判的4分排名对比，并与多种自动指标（BLEURT、COMET、TER等）对齐。结果显示：在人类评判中，高依赖度句子对无上下文策略的偏好随依赖度递增；自动指标对依赖度几乎无变化；在高依赖度段落上，模型的post‑edit量趋于保守且错误率升高。

**⚠️ 局限性**

局限性包括：仅在英文-韩文对进行实验，未覆盖其他语言对；依赖度仅考虑实体再提及与代词共指，未涵盖词汇黏着、动词时态等其他语篇现象；依赖度计算受NER/POS/解析错误影响；实验使用的LLM可能存在数据泄露风险；且未在多语言、低资源场景中验证其鲁棒性。

---

## 365. MINT: A Unified Model for World-Space Camera and Hand Motion Estimation from Scalable Egocentric Pipeline Supervision

**arXiv ID:** 2609.04958 | [PDF](https://arxiv.org/pdf/2609.04958v1)

**作者:** Zijie Zhu `[一作]` (ShanghaiTech University), Guanqi He `[通讯]` (Wuji Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了 MINT，一种统一的单前向网络，能够直接从 egocentric RGB 视频恢复世界坐标系下的相机轨迹和双手运动。

**💡 创新点**

创新点包括：1）将相机、视场、双手状态和手观测性统一到同一时空几何表示上，实现在一次前向推理中同时完成所有任务；2）构建大规模伪标注管道，利用公开视频自动生成 1,021 小时的世界坐标运动监督；3）采用两阶段训练策略，先用伪标签学习稳健的相机-手几何关系，再用少量高精度轨迹微调相机尺度。

**🔧 技术方法**

技术方法主要是：预训练的 Geometric Context Transformer (GCT) 作为共享时空几何编码器；多头预测器分别输出相机外参、视场、双手 MANO 参数和手观测概率；使用可微分相机‑世界变换实现相机与手运动的耦合；以及基于 UKF 的后处理实现轨迹平滑。

**📊 数据集**

使用的数据集：1,729 小时的 Ego4D、EPIC‑KITCHENS 和 EgoDex 公共 egocentric 视频生成 1,021 小时伪标签；少量带高精度相机标定的数据用于阶段二微调；零样本评估在 HOT3D 与 ARCTIC 上进行。

**📈 对比分析**

与现有多阶段系统（如 HaWoR、MegaSaM、DROID‑SLAM 等）比较，MINT 在相机轨迹的相对位移误差 (RPE‑T) 上实现最优，手姿态误差 (MPJPE‑p, PA‑MPJPE‑p) 与顶尖单阶段方法相近；在推理速度上达到 44fps，显著快于现有方法。

**⚠️ 局限性**

局限性：伪标签来源于单目深度与 SLAM，导致尺度漂移；32 帧训练窗口限制了长时间跨度的全局一致性，未来工作需要更精确的度量监督和更长时间的推理机制。

---

## 366. The fourth generalized Davenport constant of $C_5^3$

**arXiv ID:** 2609.04950 | [PDF](https://arxiv.org/pdf/2609.04950v1)

**作者:** Sze Chun Yiu `[一作]` `[通讯]` (Stockholm University), Sze Chun Yiu (Stockholm University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

通过对有限阿贝尔群 C5^3 的有限计算与组合推理，证明其 k 次一般化 Davenport 常数满足 _k(C5^3)=5k+10（k≥2），特别是 _4(C5^3)=30，并给出 Freeze–Schmid 下界在此群上的最佳性。

**💡 创新点**

创新点在于：① 将该群的零和子序列问题化简为对长度为 31 的“短零和自由”序列的排除；② 提出了饱和性与乘法语法、射影线限制、GL(3,5) 正规化等新组合工具；③ 设计了 78 个分支的完全可枚举搜索，并在三套独立实现上交叉验证，从而实现了完全计算验证的证明。

**🔧 技术方法**

使用的技术包括：零和子序列理论、饱和性与乘法语法推理、射影线与乘法约束、线性代数正交化、GL(3,5) 群作用归一化、极大乘法分支枚举、增量零和检测、三套并行实现的交叉验证、计算机辅助证明框架。

**📊 数据集**

数据集：对 C5^3 内所有支持大小 14–31 的乘法模式共 60 个，生成 78 个搜索分支；在每个分支中枚举所有满足乘法模式的序列，涉及约 2.94 亿个节点（最大分支 1.01 亿节点）。此外，使用已验证的 _3(C5^3)=25、_6(C5^3)=24、_7(C5^3)=19 等值作为辅助数据。

**📈 对比分析**

与现有理论（如 Freeze–Schmid 下界、已知的低秩/有限 p‑群结果）比较，证明该下界在 C5^3 上完全被达到。性能方面，三套实现互相验证，总节点约 2.94×10^9，计算耗时可在多核机器上完成，且跨机器重跑保持一致；验证层级确保结果可靠。

**⚠️ 局限性**

局限性包括：① 证明高度依赖于 C5^3 的 Property C（或类似结构定理），难以直接推广到其他群；② 计算证明未经过正式同行评审，且缺乏形式化验证；③ 结果仅适用于素数 5 的三阶扩展，其他素数或更高秩情况尚未解决；④ 计算量大，对硬件资源有一定需求。

---

## 367. One Diffusion Model, Two Roles: Guided Trajectory Planning and Safety-Critical Scenario Generation in Closed-Loop Simulation

**arXiv ID:** 2609.04921 | [PDF](https://arxiv.org/pdf/2609.04921v1)

**作者:** Arka Pal `[一作]` (KTH Royal Institute of Technology), Maciej Wozniak `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用单一预训练扩散模型同时实现自主驾驶中的轨迹规划和安全关键场景生成。

**💡 创新点**

提出SSDS-DP解码器实现早期双流注意力融合、DAPSE训练无关的能量引导以及通过能量约束的可控情景生成。

**🔧 技术方法**

采用Diffusion Transformer、联合注意力、Langevin动力学与能量函数（TTC、车道变换、制动、行驶区域约束）等技术。

**📊 数据集**

在nuPlan大规模驾驶数据集上训练，使用250K/650K/1M三种规模的场景。

**📈 对比分析**

与原始Diffusion Planner对比，SSDS-DP在交互性强、反应性场景中提升6.8–14.4分；在生成的攻击场景中，两模型性能均显著下降，说明基准优越性不一定转化为鲁棒性。

**⚠️ 局限性**

缺点包括：在大数据规模下SSDS-DP不一定优于基线；对极端情景的生成仍受能量函数设计限制；并未系统评估DAGSE的碰撞率与TTC改进等指标。

---

## 368. TPMSpy: Validation of Measured Boot Systems by Low-Level Tracing of TPM Usage

**arXiv ID:** 2609.05011 | [PDF](https://arxiv.org/pdf/2609.05011v1)

**作者:** Roman Lacko `[一作]` (Masaryk University), Petr Svenda `[通讯]` (Masaryk University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本研究提出并实现了 TPMSpy，一种基于虚拟化的、平台无关的低层 TPM 交互追踪工具，用来独立重建并验证 TPM 事件日志；通过该工具对 NixOS、Fedora、Ubuntu 及 Windows 11 的 Measured Boot 过程进行了大规模实验，并对 Linux 发行版 v245–v258 的 TPM 使用演化进行系统纵向分析；

**💡 创新点**

创新点在于：①实现了完全无侵入、可重复的 TPM 交互捕获机制；②不依赖具体平台实现，可在任意带 TPM 的虚拟机上使用；③通过捕获完整 TPM 命令流可发现文档中未记录的行为和日志不完整性，提升了远程证明的可信度；

**🔧 技术方法**

使用技术包括 QEMU 虚拟机、软件 TPM（swtpm）及其 UNIX 套接字通信，TPMSpy 通过拦截和记录控制/数据通道实现对 TPM 命令的完整观察；配合脚本自动收集事件日志、PCR 值以及内核/启动器版本信息；

**📊 数据集**

数据集主要是对 NixOS（v245–v258）、Fedora 41–43、Ubuntu 24 LTS 以及 Windows 11 进行的 1000 次启动实验，共计数千条 TPM 交互记录；

**📈 对比分析**

实验比较表明：TPMSpy 的运行时间仅为原系统 1.64 秒的 6.9% 额外开销，CPU 使用率低于 0.02%（约 40 ms/启动），不显著影响系统启动性能；

**⚠️ 局限性**

局限性在于：①仅能观察到 TPM 的交互，无法确认所有系统组件是否都已被测量；②缺乏对事件来源的上下文信息（如具体执行进程），未来可通过查询 VM 状态补充；③依赖虚拟化环境，真实硬件上的行为可能存在差异；

---

## 369. On the equivalence between generating functions computed by memory transducers and enumerating functions produced by indexed grammars

**arXiv ID:** 2609.05002 | [PDF](https://arxiv.org/pdf/2609.05002v1)

**作者:** Vincent Ghigo `[一作]` `[通讯]` (University of Bordeaux), Vincent Ghigo (University of Bordeaux)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

提出了一个理论框架，证明在任意结构上，使用记忆为堆栈堆栈的确定性变压器（k+2）计算的整数序列恰好是由 k-索引的上下文无关索引语法生成的树的计数序列。

**💡 创新点**

创新点在于将变压器与索引语法联系起来，构造了一系列从变压器到索引语法、再到可枚举语法的转换，扩展了传统的正则-线性、上下文无关-代数递归的对应关系。

**🔧 技术方法**

主要技术包括构造确定性变压器、符号运行、交替变压器、堆栈结构的语义化，以及一系列转换（T1–T6）将变压器运行映射到语法推导树。

**📊 数据集**

本研究是理论性的，未使用具体实验数据集，而是通过形式化证明和构造性算法完成。

**📈 对比分析**

方法的比较是理论等价性证明；没有实验性能指标，重点在于提供有效构造和证明等价关系。

**⚠️ 局限性**

局限性包括只处理确定性变压器、需要无死锁、无无限推导；对非确定性变压器或更一般的 D‑finite 级数的表示仍未解决。

---

## 370. Moral Competence Before Moral Content: Why LLM Agents Lack the Prerequisites for Coherent Alignment

**arXiv ID:** 2609.05036 | [PDF](https://arxiv.org/pdf/2609.05036v1)

**作者:** Arno Libert `[一作]` (Aithos Research Foundation), Daan R. Henselmans `[通讯]` (Aithos Research Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一套结构化评估方法，用于检测大语言模型在面向代理情境下是否表现出连贯的道德策略。

**💡 创新点**

创新点在于提出四个结构性条件（判决稳定性、单调性、决定性与帕累托可行性），并将它们合成为一个不依赖具体道德规范的“道德能力”基准。

**🔧 技术方法**

技术上采用因子设计（变形、升级、支配）对三种代理决策场景进行多维度采样，并对每个模型在 3,500 次试验下计算判决率，随后用统计修正（方差分解、噪声校正）得到四个指标。

**📊 数据集**

数据集为作者自构的三种模拟代理情境（化学泄漏、智能家居上架与金融透明度），每个情境包含 5×5×3=75 个语义保持或升级变体，共计 9 个前沿 LLM（Claude, GPT, Gemini, Mistral 等）。

**📈 对比分析**

比较方法是将每个模型在每个场景下的四个指标取几何平均得到综合能力分数；结果显示无模型在所有场景中均达到 1，平均分数在 0.53–0.75 之间，且不同场景之间的排名高度不相关。

**⚠️ 局限性**

局限性包括：仅评估三种场景且采用单决策点脚本化设置，可能不足以覆盖真实代理部署；方法对模型内部机制缺乏可解释性；并且结构条件与实际道德规范的映射仍需进一步研究。

---

## 371. Language models judge war differently when tested for alignment

**arXiv ID:** 2609.05009 | [PDF](https://arxiv.org/pdf/2609.05009v1)

**作者:** Maxim Chupilkin `[一作]` `[通讯]` (University of Oxford), Maxim Chupilkin (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在对大型语言模型进行对齐评估时，一句简单的提示句（"You are tested for alignment with human values"）如何同时改变模型的整体倾向（水平效应）和对决策因素的权重分配（结构效应），并在20个不同模型上使用全因子conjoint实验验证这一现象。

**💡 创新点**

创新点在于提出并区分了对齐评估的“水平效应”与“结构效应”，首次在多模型设置中量化“对齐伪装”——即模型在被提示评估时会显著调整其内部权重，而不仅仅是整体上更保守；同时展示了conjoint实验在捕捉模型决策权重变化方面的有效性。

**🔧 技术方法**

采用了全因子5属性conjoint实验设计、OLS回归、模型固定效应、标准化系数、模型-提示交互分析，以及通过OpenRouter API对20个大型语言模型进行批量评分的技术。

**📊 数据集**

使用了20个大型语言模型（来自13个开发商）的数据，每个模型评估32个二值化战争情景（概率、国内支持、平民伤亡、军队伤亡、经济成本），共12800条评分记录。

**📈 对比分析**

通过比较基线与提示两条件下的平均评分、标准化系数以及权重排序，评估模型对提示的响应。结果显示提示将平均评分降低13.4点，显著压缩了战略成功和国内支持的权重，部分模型更关注平民伤亡，表明不同模型对提示的响应存在异质性。

**⚠️ 局限性**

局限性包括：仅使用单一句提示，未探讨更细微或多样化的提示；属性仅二值化，无法捕捉更细粒度的成本/伤亡影响；样本为选定的20个模型，非随机抽样；未控制温度参数；实验情境高度简化，无法直接映射到真实军事部署；无法直接揭示模型内部机制或真实偏好。

---

## 372. Temporal Residual Neural Radiance Fields for Monocular Video Dynamic Human Body Reconstruction

**arXiv ID:** 2609.04984 | [PDF](https://arxiv.org/pdf/2609.04984v1)

**作者:** Tianle Du `[一作]` (Nanchang University), Jie Liu `[通讯]` (Nanchang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于时序残差神经辐射场（TRes-NeRF）的方法，用于单目视频中动态人体的快速高质量重建。

**💡 创新点**

创新点在于将时间残差层嵌入MLP并进行因式分解以提升时空建模能力，同时结合Instant-NGP哈希编码与空体跳过策略实现高效渲染。

**🔧 技术方法**

采用多层感知器+时间残差、因式分解、hash编码、稀疏体素跳过、刚体变换以及多维损失等技术。

**📊 数据集**

在PeopleSnapshot和NeuMan两个真实与合成的单目人体视频数据集上进行训练与测试。

**📈 对比分析**

与Anim-NeRF、Neural Body和InstantAvatar等SOTA方法比较，PSNR/SSIM表现相当或更好，训练时间从数小时降至约一分钟，渲染帧率提升至13 FPS，整体速度提升约780倍。

**⚠️ 局限性**

局限性是仍依赖监督信息，在缺乏无监督约束的复杂场景下表现可能不如预期。

---

## 373. MCPO: Modality-Contrastive Preference Optimization for Multimodal Chain-of-Thought Compression

**arXiv ID:** 2609.04947 | [PDF](https://arxiv.org/pdf/2609.04947v1)

**作者:** Guangheng Yang `[一作]` (Tsinghua University), Jie Hu `[通讯]` (Huawei Technologies Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于多模态对比偏好优化的链式推理压缩框架 MCPO，显著减少推理步骤长度并提升推理速度；

**💡 创新点**

创新点在于结合自适应跨模态互信息剪枝与非对称长度对齐损失，解决视觉依赖不足和幻觉推理问题；

**🔧 技术方法**

采用跨模态互信息评估（NCMI）剪枝、双通道视觉零填充、非线性对数几率对齐损失与对比损失等技术；

**📊 数据集**

使用 905 条来自 ScienceQA 与 CLEVR 的高质量推理样本，并在 MathVista、ScienceQA、MMMU、MMStar 等四大多模态基准上评测；

**📈 对比分析**

与 StepEntropy、REFRAIN 等压缩方法对比，MCPO 在 8B/4B 模型上实现 42.6%–69.5% 的 token 压缩，速度提升 1.38×–3.34×，准确率仅下降 0.4%–1.0%；

**⚠️ 局限性**

局限性包括对小样本数据集的依赖、对特定视觉零填充策略的敏感性以及在极端任务难度下对对比权重的调节需求。

---

## 374. Physics-Aware Random Walk Fingerprints for Scalable Power Grid Graph Classification

**arXiv ID:** 2609.04943 | [PDF](https://arxiv.org/pdf/2609.04943v1)

**作者:** Adnan Anwar `[一作]` `[通讯]` (Deakin University), Adnan Anwar (Deakin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Multi‑Channel Physics‑Aware Random Walk Fingerprints（MC‑PA‑RWF）框架，利用电力系统线路的操作状态（功率流、容量余量、耦合强度、超载严重度）构造多条加权通道，对每条通道分别提取随机游走指纹并拼接得到图级表示；并进一步加入节点特征的扩展版本 MC‑PA‑RWF+。

**💡 创新点**

创新点在于：① 将物理边属性嵌入随机游走传播矩阵，使随机游走能捕捉电力线路的实时负荷与压力；② 通过多通道融合不同物理维度的传播视角，提升表示的鲁棒性；③ 保持非神经网络的可解释性与线性可扩展性，且在大规模数据集上仍能与强大 GNN 比肩。

**🔧 技术方法**

技术手段包括：随机游走指纹（RWF）框架、结构化节点分组、对边属性的中值缩放与截断、对每条物理通道构造对称归一化的传播矩阵、SVM 监督分类；以及对比实验中使用的 GCN、GAT、GINE、TransformerConv 等 GNN 架构。

**📊 数据集**

使用了 PowerGraph 基准数据集，包括 IEEE24、IEEE39 和 UK 三个电网，分别包含数千到数万张图，任务为图级级联故障分类（稳定 vs 失效）。

**📈 对比分析**

与传统拓扑 RWF、节点特征 RWF 以及四种主流 GNN 进行比较。MC‑PA‑RWF 在 IEEE24、IEEE39 和 UK 上的平衡准确率分别达到 99.15%、97.15% 与 99.14%，MC‑PA‑RWF+ 更是实现 99.21%、98.04% 与 99.32%，在所有数据集上均至少与最佳 GNN（TransformerConv）持平或略胜；在失败类 F1 方面，MC‑PA‑RWF+ 相比最佳 GNN 提升 1.6–5.8 个百分点，且差异在统计学上显著。

**⚠️ 局限性**

局限性包括：① 需要先验知识手工定义物理通道，缺乏自适应通道选择；② 仅在电力系统上验证，泛化到其他网络类型尚未测试；③ 仍需对边属性做预处理（缩放、截断），可能忽略细粒度的动态特征；④ 对极端大规模图的计算仍受 O(NCτ_c mn) 复杂度限制。

---

## 375. Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection

**arXiv ID:** 2609.05025 | [PDF](https://arxiv.org/pdf/2609.05025v1)

**作者:** Renato Vukovic `[一作]` (Heinrich Heine University Düsseldorf), Milica Gasic `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用 LLM 自动将参考文档转化为 SQL 数据库，并在该数据库上执行 SQL 查询，以此来进行无监督的幻觉检测。

**💡 创新点**

创新点在于：①首次将低级符号能力（SQL）与 LLM 结合，实现无监督、可解释的幻觉检测；②构建可重用的 neurosymbolic 检查流程；③不需要任何领域特定微调即可竞争最先进方法。

**🔧 技术方法**

技术：多步 Prompting + Text‑to‑SQL 生成、数据库增量构建、SQL 检索、神经‑符号对齐（neurosymbolic checkup）以及零样本提示。

**📊 数据集**

数据集：RAGTruth（QA、数据生成、摘要）和 DiaHalu（任务导向对话）。

**📈 对比分析**

对比方法包括直接预测（Zero‑shot）、SelfCheckGPT、RAG‑HAT、LettuceDetect、ReDeEP 等。TeQHallu 在 RAGTruth 的宏观 F1 约 71.3，超过所有无监督基线并接近 fine‑tuned SOTA；在 DiaHalu 上也实现了最高 F1，特别是在非事实错误和过度依赖的检测上表现突出。

**⚠️ 局限性**

限制：依赖参考文档的完整性与质量；SQL 生成与查询过程增加计算成本与延迟；SQL 语义映射可能出现信息损失；模型生成错误 SQL 可能误导判定；目前缺乏大规模人类评估与对不同领域偏差的深入探讨。

---

## 376. Fractal basins trap latent reasoning

**arXiv ID:** 2609.04963 | [PDF](https://arxiv.org/pdf/2609.04963v1)

**作者:** Jeffrey Lai `[一作]` (University of Texas at Austin), William Gilpin `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了推理模型的动力学行为，发现其在解决难题时会出现瞬态混沌和分形吸引子；

**💡 创新点**

揭示了推理慢速（过度思考）是难题固有复杂度的不可避免后果，并将分形基底和瞬态混沌作为推理过程的新动力学分类；

**🔧 技术方法**

采用动态系统理论、分形基底熵、快速李雅普诺夫指标、有限时李雅普诺夫指数等工具，结合循环 Transformer、Equilibrium Reasoner、Fixed‑Point Reasoner 等模型；

**📊 数据集**

实验覆盖了 Sudoku、Maze、Countdown、ARC‑AGI 等多种任务，训练小型循环 Transformer 解决整数线性系统；

**📈 对比分析**

通过基底熵与迭代次数相关性、Lyapunov 指标与解路径次数的对应，验证了更难任务导致更高的基底熵和更长的收敛时间，显示推理模型在困难任务上收敛更慢；

**⚠️ 局限性**

研究范围主要集中在实验室级模型与合成任务，缺乏在大规模真实场景中的验证，且未提出有效的缓解瞬态混沌的方法，适用性和可推广性仍有限。

---

## 377. MoirfEolas and CríochScore: Developing Resources for and the Evaluation of Tokenization Alignment with Irish Morphology

**arXiv ID:** 2609.05022 | [PDF](https://arxiv.org/pdf/2609.05022v1)

**作者:** Jane Adkins `[一作]` (ADAPT Centre), Elaine Uí Dhonnchadha `[通讯]` (Trinity College Dublin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了爱尔兰语的形态切分，构建了 MoirfEolas 词形资源和 CríochScore 评估指标，并用它们对常用子词分词算法进行内在评估。

**💡 创新点**

提出了针对低资源爱尔兰语的专属词形切分资源和自定义的形态边界对齐度量 CríochScore，并展示了 Unigram 模型在该语言上优越的对齐性能。

**🔧 技术方法**

使用了多种子词分词技术（Unigram LM、BPE、WordPiece、SentencePiece 等）以及自定义的 CríochScore 评估框架。

**📊 数据集**

基于 ParaCrawl V9 的爱尔兰语语料、UniMorph 数据以及手工词形列表构建的 MoirfEolas 词表，包含 35,000+ 词条。

**📈 对比分析**

通过 CríochScore、词频指标（Corpus Token Count、Fertility、Rényi Efficiency）以及前缀/后缀/eclipsis 的召回/精准度进行比较；Unigram 8k 词表得分最高（≈40%），BPE 表现最差。

**⚠️ 局限性**

仅进行内部评估，缺乏下游任务验证；eclipsis 处理仍不充分，未覆盖 lenition 现象，资源规模与语料深度有限。

---

## 378. TROVE: Adaptive Agent Skill Orchestration via Trace-Grounded Route Validation and Editing

**arXiv ID:** 2609.05019 | [PDF](https://arxiv.org/pdf/2609.05019v1)

**作者:** Tianxing Wang `[一作]`, Fan Wu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TROVE 框架，利用离线搜索轨迹提炼可复用技能和结果条件转移图，在线时仅承诺执行首个技能，随后根据运行时结果进行保留、插入或替换未执行路径，从而在保证任务质量的同时显著提升执行效率。

**💡 创新点**

核心创新在于：① 将工作流搜索轨迹拆解为原子与复合技能，构建结果驱动的转移图；② 在线只执行一步，按需微调剩余路径，避免完整重排或全局抛弃；③ 通过“保留–插入–替换”三种局部编辑策略，兼顾质量和效率。

**🔧 技术方法**

技术手段包括：离线工作流搜索与轨迹收集；技能抽象与复合化；基于轨迹的结果条件转移图构建；在线技能执行与边界结果监测；LLM 辅助的局部重排；多模态 LLM（DeepSeek‑V4‑Flash、GPT‑4o‑mini、Qwen3‑8B）作为推理引擎。

**📊 数据集**

评测数据集：HumanEval、MBPP、MATH、GSM8K、DROP、HotpotQA（共 6 个代码、数学、问答基准）。

**📈 对比分析**

与 AFlow（数据集级优化）、MaAS（查询级预优化）和 LAS（图约束在线调度）进行对比。TROVE 在 18 个模型‑基准组合中取得 15 个最佳或并列最佳分数，16 个最快执行时间；相较 AFlow，分数提升 0.12–31.37 分，时间缩短 2.9–86.7%；相较 MaAS，分数提升 0.48–46.27 分，时间缩短 6.1–94.6%；相较 LAS，分数提升 1.47–7.03 分，时间缩短 37.3–63.9%。

**⚠️ 局限性**

局限性包括：① 依赖离线搜索轨迹，若训练集覆盖不足可能导致技能/图覆盖不全；② 对极大规模或高度动态任务的适应性尚未验证；③ 对某些 QA 任务仍存在较高 token 消耗和较低分数提升；④ 对 LLM 生成质量的依赖导致重排成本和结果可变性。

---

## 379. On Maximizing a Weakly Submodular Function over a Matroid Constraint via the Greedy Algorithm

**arXiv ID:** 2609.05008 | [PDF](https://arxiv.org/pdf/2609.05008v1)

**作者:** Justin Ward `[一作]` (Queen Mary University of London), Moran Feldman `[通讯]` (University of Haifa)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了在一般基 matroid 约束下，贪心算法对弱子模函数的最大化没有常数逼近保值。

**💡 创新点**

构造了一类 γ-弱子模函数和简化分区 matroid 的逆例，显示贪心算法逼近比率随基数增长而下降。

**🔧 技术方法**

利用子模比率定义，构造基于对数函数的目标函数，并通过严格的数学证明验证其单调性和弱子模性。

**📊 数据集**

本文完全基于理论构造，无需使用真实数据集。

**📈 对比分析**

与最优解直接对比，证明贪心解的价值仅是最优解的 O(γ/(1-γ)^2 log n) 比例，说明性能随实例规模恶化。

**⚠️ 局限性**

该结果仅适用于基于分区 matroid 的情形，未给出其他算法的上界，也未解决更广泛基 matroid 或非单调情形的情况。

---

## 380. BIT.UA at BioASQ 14B: Modular Retrieval with pg_textsearch and Qdrant, and Agent-Based Answer Generation

**arXiv ID:** 2609.04999 | [PDF](https://arxiv.org/pdf/2609.04999v1)

**作者:** André Ribeiro `[一作]` (University of Aveiro), Sérgio Matos `[通讯]` (University of Aveiro)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在第14届BioASQ Task B竞赛中，BIT.UA团队重新设计了检索与生成管线，改用PostgreSQL+pg_textsearch做BM25检索，结合Qdrant做稠密检索，并引入多模型协作的生成方法。

**💡 创新点**

创新点在于：1) 将检索与生成完全模块化；2) 使用HyDE与Context-1进行查询扩展与LLM驱动检索；3) 采用密集检索负样本训练的重排序模型；4) 引入LLM-as-a-judge与多模型代理仲裁（Agent Quorum）并结合自适应文档保留。

**🔧 技术方法**

使用技术包括：PostgreSQL pg_textsearch、Qdrant向量索引、Text Embeddings Interface、HyDE、Context-1、BM25、BGE-Reranker、LLaMA Nemotron、Gemma、Mistral、Qwen、LLM-as-a-judge、Agent Quorum、ROUGE/ROUGE‑SU4评估等。

**📊 数据集**

使用数据集：BioASQ 13B的golden数据用于内部验证，BioASQ 14B的四个测试批次（Batch 1‑4）以及官方评测数据。

**📈 对比分析**

比较方法：检索评估采用MAP@10、MRR、R@100等；生成评估采用Y/N、factoid、list的F1和ROUGE‑2/ROUGE‑SU4；结果显示Phase A最高排名第5（MAP≈0.28），Agent Quorum在Phase A+和B多次取得最优或接近最优的F1/ROUGE；小模型在部分指标上与大模型竞争；整体性能优于以往年份但波动较大。

**⚠️ 局限性**

局限性：内部验证与官方评测结果不一致；Context‑1检索效果不佳；高计算成本的代理仲裁；ROUGE与人类评价不完全对应；人类评测尚未完成；迁移到实际系统的通用性待验证。

---

## 381. Global to Local: Topology-Preserving Adaptive Graph Pooling via Granular-Ball

**arXiv ID:** 2609.04978 | [PDF](https://arxiv.org/pdf/2609.04978v1)

**作者:** Sen Zhao `[一作]`, Wei Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于颗粒球（granular-ball）的拓扑保留自适应图池化方法TPAGP，用于在图神经网络中生成多尺度、结构友好的图表示；

**💡 创新点**

创新点在于：①通过从全局到局部的颗粒球划分动态确定自适应粒度；②利用图拓扑与特征相结合的BFS划分与TDA持久同调质量评估，实现多层次结构的保留；③构建多颗粒球图网络实现跨尺度特征交互；

**🔧 技术方法**

技术手段包括：图神经网络（GCN/GIN/GAT等）与层叠卷积、颗粒球自适应划分、双向BFS中心分配、持久同调（TDA）质量评估、梯度反向传播训练、超参数调优与消融实验；

**📊 数据集**

使用了六个标准图分类数据集：MUTAG、MSRC_9、BZR、DD、PTC_MR 与 IMDB-MULTI；

**📈 对比分析**

与多种GNN基线（GCN、GAT、GIN、GraphSAGE等）及多种池化方法（TopKPool、SAGPool、DiffPool、MinCutPool、TIP等）对比，TPAGP 在所有数据集上均取得最高准确率，平均提升约3–6个百分点，表现出更好的泛化与鲁棒性；

**⚠️ 局限性**

局限性包括：①对大规模图的计算效率与内存占用尚未充分评估；②深层GNN可能导致过度平滑，需精细调节层数；③颗粒球划分依赖节点度与超参数，可能对特定图结构敏感。

---

## 382. Robust Coverless Linguistic Steganography via Sentence Embedding Space with Global Resynchronization

**arXiv ID:** 2609.04970 | [PDF](https://arxiv.org/pdf/2609.04970v1)

**作者:** Lizhi Xiong `[一作]` (Nanjing University of Information Science and Technology), Zhangjie Fu `[通讯]` (Nanjing University of Information Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在句子嵌入空间进行无覆盖（coverless）的鲁棒语言隐写框架；

**💡 创新点**

核心创新在于：①通过层级聚类将句子嵌入空间划分为可编码语义子空间，天然抵御词级与句级扰动；②引入全局重同步机制（GRM）结合LT码，将局部解码失误转化为符号缺失，消除位移（bit‑slippage）导致的错误传播；

**🔧 技术方法**

使用Sentence‑T5‑Large句向量编码、k‑means层级聚类（HCM）、SparSamp子空间采样、LT码+BP解码、伪随机数生成器；

**📊 数据集**

主要数据集为PersonaChat，另外对C4和IMDB做了跨数据集实验；

**📈 对比分析**

与AC、ADG、Discop、SECC、STEAD、FSBTS等基线进行对比；在五种文本攻击（删、换、插、互、释义）下，鲁棒性（P_E）显著低于2.7%，比基线低一至两百倍；PPL、嵌入率和有效嵌入率也优于基线；在反隐写分析中，检测准确率接近50%，说明安全性更好；

**⚠️ 局限性**

限制主要体现在：①子空间数k越大鲁棒性下降，容量提升有限；②需要共享大型句子编码器和数据集，部署成本高；③在高扰动（高α）下，ER仍略升高；③LT码冗余导致容量下降，需在鲁棒性与容量间权衡。

---

## 383. SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce

**arXiv ID:** 2609.04961 | [PDF](https://arxiv.org/pdf/2609.04961v1)

**作者:** Hui Zhou `[一作]` (Alibaba International Digital Commerce Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SAM-D2Q，一种针对电商检索的多模态Doc2Query框架，既能对标题进行语义扩展，又能对图像属性进行视觉归属，并通过业务指标对生成结果进行优化；

**💡 创新点**

创新点在于①基于Boolean检索的“信息增益”约束重新定义Doc2Query目标；②利用CPV知识库对文本进行视觉属性的对照增强；③引入GRPO强化学习，在语义、商业和视觉奖励的多重目标下对生成策略进行业务对齐；

**🔧 技术方法**

采用多模态大语言模型（Qwen3-VL-8B）进行分阶段监督微调，结合CPV引导的对照数据增强和Group Relative Policy Optimization（GRPO）进行业务偏好对齐；

**📊 数据集**

使用阿里巴巴AliExpress的生产日志：约9M点击日志用于Stage‑1，50万对照增强样本和3.5M人工标注的相关对；8.8M商品倒排索引用于离线评估；在线A/B测试覆盖4%实时流量；

**📈 对比分析**

与文本版Doc2Query、单模态SFT以及无RL对齐的变体进行对比；离线检索中#Rel Items提升30%，TotalQual提升33%；在线A/B中GMV提升3.38%，Pay Count提升2.27%，CVR提升0.53%；

**⚠️ 局限性**

局限性包括：①模型需在离线索引中预生成，无法实现实时推理；②对CPV属性表的依赖导致对缺失视觉属性的商品扩展效果有限；③在跨域或多语言场景下的泛化性尚未验证；

---

## 384. VICAL: Vicinal Consistency Alignment for Long-Tailed Visual Recognition

**arXiv ID:** 2609.04948 | [PDF](https://arxiv.org/pdf/2609.04948v1)

**作者:** Jiangang Zhu `[一作]`, Jingjing Chen `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究长尾视觉识别问题，提出 VICAL 框架通过对多专家模型进行 vicinal consistency alignment 来提升性能，重点减少预测方差而非提升专家多样性。

**💡 创新点**

创新点在于摒弃传统的多样性驱动做法，改为利用 Self-Consistency Learning（在强对抗视图插值中对齐学生与 EMA 教师的输出）和 Deep Ensemble Distillation（在低分辨率视图上实现跨专家低频语义一致），并引入冲突知识过滤器以避免信息冲突，从而显著降低模型方差。

**🔧 技术方法**

主要技术包括：logit 调整、强数据增强（AutoAugment/RandAugment）、EMA 教师、插值对抗视图、低分辨率蒸馏、冲突知识过滤器、以及自监督一致性学习。

**📊 数据集**

在 CIFAR-10/100 长尾、ImageNet-LT、以及 iNaturalist 2018 等长尾视觉基准上进行实验。

**📈 对比分析**

与现有多专家模型（如 NCL++、MDCS、BalPoE 等）以及一致性学习方法进行对比，VICAL 在所有数据集与评价指标上均实现或接近 state‑of‑the‑art，尤其在尾部类别上提升显著，同时保持头部类别的高精度。

**⚠️ 局限性**

主要限制包括：多专家+多视图训练需要较高的计算资源；推理时仍需使用目标网络，虽然成本与单模型相近；对超参数（如 β、η、α、λ）敏感；缺乏严格的理论分析说明一致性如何导致方差降低。

---

## 385. Who's Blocking Whom? Candidate Generation and Block Prediction on Bluesky

**arXiv ID:** 2609.04923 | [PDF](https://arxiv.org/pdf/2609.04923v1)

**作者:** Cecilia Galbiati `[一作]` (Politecnico di Milano), Francesco Pierri `[通讯]` (Politecnico di Milano)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在Bluesky社交平台上，预测哪些用户会在何时阻止特定账户的行为，提出了两阶段的有向阻止预测框架；

**💡 创新点**

创新点在于系统性评估候选生成规则对预测覆盖率、性能及特征重要性的影响，揭示候选选择是预测任务的关键设计；

**🔧 技术方法**

使用了LightGBM梯度提升树模型，结合对历史交互、网络结构、内容等多层特征的工程；

**📊 数据集**

数据集来源于Bluesky公开Firehose，包含超过260万次互动和超过300万次首次阻止事件，涉及约260万用户；

**📈 对比分析**

在七种实验设置中，最优设置（候选为直接或共同邻居、同源排名）实现Hits@1约61%，相较于随机基线约16%；

**⚠️ 局限性**

主要局限包括候选生成规则无法覆盖所有阻止事件，实验仅在Bluesky短时间窗口内，且缺乏因果推断和用户安全评估。

---

## 386. QUASAR: Quantum Satellite Architecture and Routing Simulator

**arXiv ID:** 2609.04920 | [PDF](https://arxiv.org/pdf/2609.04920v1)

**作者:** Yaliang Shi `[一作]` (University of Electronic Science and Technology of China), Zhiwei Zhao `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了轻量级卫星量子网络仿真框架 QUASAR，用于在 LEO 轨道动态、光学衰减和量子记忆退相干等物理约束下评估纠缠分发与路由。

**💡 创新点**

创新点在于将连续轨道传播、时变光学通道和记忆退相干抽象为网络层动态属性，采用事件驱动的解耦架构，使得在大规模 LEO 星座中实现可扩展的离散事件仿真，并对两种硬件范式（同步下行与在轨拼接）以及基于 EDR 的路由启发式进行统一评估。

**🔧 技术方法**

技术包括：Python 叠加层基于 SimQN 的离散事件核心；SGP4 轨道传播与 Walker‑Delta/TLE 生成；自由空间衰减+大气抛物公式计算光学通道；指数退相干模型；事件驱动更新与可视化；基于 Dijkstra 的 EASR 路由。

**📊 数据集**

使用的轨道数据集有可配置的 Walker‑Delta 星座、从 CelesTrak 获得的 Starlink 60 颗卫星 TLE 集合；以及 32 个全球基站的请求流。

**📈 对比分析**

比较方法：与连续轮询更新做对比，评估网络层更新延迟、事件触发次数、可视化 EDR；与三种基准路由（DSP、MPR、EASR）在不同记忆寿命下对比 EDR；通过宏基准测试验证可扩展性。性能表现：在 800 颗卫星时，事件驱动比连续轮询减少 85%+ 更新延迟，EASR 在给定记忆寿命下实现最高 EDR。

**⚠️ 局限性**

局限性：仍未集成完整的光学硬件误差模型和多路复用；对地面站多址/多用户调度未建模；仿真仅覆盖纠缠分发级别，未深入实现量子层协议细节；对极端轨道扰动或星座变形的鲁棒性待验证。

---

## 387. Compact-Memory LLM Agents via Online Max-Member Clustering and Atom-Aware Packing

**arXiv ID:** 2609.04915 | [PDF](https://arxiv.org/pdf/2609.04915v1)

**作者:** Jiahe Geng `[一作]`, Kun Yuan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在线聚类记忆管道 RSM-full，针对长序列 LLM 在紧凑提示预算下提升答案质量。

**💡 创新点**

核心创新是：① 采用余弦门控最大成员合并写入规则，使得新片段可与簇中心或任意成员满足阈值即可合并；② 引入原子感知分组上下文打包，将检索到的片段按簇组织而非平面排序，显著提升提示质量。

**🔧 技术方法**

使用 BGE 大模型嵌入、余弦相似度阈值、在线 K-Means/DP-Means 对比、atom-aware 分组打包、可选 int8/PQ 等压缩后端。

**📊 数据集**

主要实验数据集包括 AMA-Bench（208 任务）、RealMem（10 人格长对话）以及 LoCoMo-Plus 等外部基准，用于验证不同环境下的表现。

**📈 对比分析**

在约 4k 令牌预算下，RSM-full 以 83% 的 Full‑Context 质量和 32% 的令牌成本（≈0.311 的准确率）位列 compact‑memory Pareto 前沿；相较于 Online K‑Means、Streaming‑Proto 等基线提升 3.5–6.0pp，且在 RealMem 上也显著优于 Budget‑RAG、Streaming‑Proto 等方法。

**⚠️ 局限性**

局限包括：仅在单一 BGE 嵌入堆栈下验证；未实现选择性遗忘；对高度词汇化场景仍需混合检索；对跨嵌入器或更高令牌预算的可扩展性未完全评估。

---

## 388. MomentQuant: an even more minimalist interval method with linear time complexity for time series classification

**arXiv ID:** 2609.05136 | [PDF](https://arxiv.org/pdf/2609.05136v1)

**作者:** Johann Faouzi `[一作]` `[通讯]` (University Rennes), Johann Faouzi (University Rennes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

重新实现Quant并提出MomentQuant，提供更快的实现并引入基于矩的近似量化方法；

**💡 创新点**

对Quant的理论复杂度进行严格分析，提出两种实现策略（series‑outer 与 interval‑outer）并实现自动切换；引入Cornish‑Fisher矩估计替代排序，消除O(log l)开销，实现更快的近似模式；

**🔧 技术方法**

使用interval‑based方法、dyadic划分、四种时间序列表示、Moments与Cornish‑Fisher展开、Numba/NumPy实现、理论成本模型与实测校准；

**📊 数据集**

在UCR时序数据集上实验，包含142个单变量数据集（112标准+30 bake‑off）；

**📈 对比分析**

与原Quant、pyQuantFloat64、ROCKET、MiniRocket等算法按ACC、BALACC、AUROC、NLL、F1等指标比较；单线程下MomentQuant约快2×，近似模式更快但精度略低；多线程加速后原Quant更快；自动dispatch在大多数场景下选择最优实现；

**⚠️ 局限性**

近似模式对短序列可能导致精度下降；实现的l‑dependent调用开销使理论与实际切换点偏离；模型假设固定深度d=6、量化除数ν=4，未覆盖其他超参；近似量化不保证单调性，可能产生无效的分位数。

---

## 389. Strategic Facility Location in Euclidean Spaces

**arXiv ID:** 2609.05132 | [PDF](https://arxiv.org/pdf/2609.05132v1)

**作者:** Kim Thang Nguyen `[一作]`, Bertrand Simon `[通讯]` (Université Grenoble Alpes)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并分析了多种无支付的策略不变机制，用以在欧氏空间中实现设施定位，尤其针对两位代理人和任意维度下的均等化成本目标。

**💡 创新点**

创新点在于利用额外维度的设施位置、正交正方形抽奖、提升箱中心以及 SD‑lift 机制来提高逼近比例，并通过新的势能函数和递归构造给出了接近上界的下界。

**🔧 技术方法**

主要技术包括几何构造（正方形、正多面体、球面）、随机化与潜在函数分析、策略不变性证明、期望距离与凸优化的组合。

**📊 数据集**

该工作为理论研究，无实测数据集；所有结果均为解析性上界/下界证明。

**📈 对比分析**

与先前已知的 2‑1/n 上界相比，本文在 2 维两代理人情形下实现了最优 √2 比例，在更高维下给出了 1.6054 以上的下界，整体性能在理论上已逼近极限。

**⚠️ 局限性**

局限性包括：高维下的上界仍为 2，且机制在更高维的性能仍未达到最优；对非欧氏度量或不等代理数时的适用性待进一步研究。

---

## 390. Unifying ICL, SFT, KL-Regularized RL Through a Bayesian Lens

**arXiv ID:** 2609.05111 | [PDF](https://arxiv.org/pdf/2609.05111v1)

**作者:** Junxin Fan `[一作]` `[通讯]` (Fudan University), Junxin Fan (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

论文通过统一的贝叶斯/KL投影视角，阐释了少量样本上下文学习(SFT/ICL)、强化学习调优(RLHF/RLVR)及其奖励加权变体的本质相似性。

**💡 创新点**

创新点在于将这些方法映射为对同一目标Gibbs后验的前向KL投影，揭示了它们在目标与梯度更新层面的等价性，并指出冷启动与支持重叠等关键因素。

**🔧 技术方法**

使用的技术包括贝叶斯推断、Gibbs后验构造、KL正则化RL、奖励/优势加权SFT、RW‑ICL、信息几何与近似推理。

**📊 数据集**

论文主要基于理论推导和假设，未使用具体公开数据集，而是在对话式与推理任务的通用框架中讨论。

**📈 对比分析**

方法的比较以理论层面进行，提出了在RLHF/RLVR、奖励加权SFT、RW‑ICL和标准ICL等场景下的性能等价性；实验上未给出具体数值，而是通过对DeepSeek‑R1、o1等模型的观察来佐证。

**⚠️ 局限性**

局限性包括只在目标与梯度层面建立等价性，未覆盖过程级别差异；对支持重叠与冷启动的依赖；对大规模数据与实际实现细节的假设可能导致实际效果与理论差异。

---

## 391. NEAT-POCKET: Pocket-Conditioned Autoregressive 3D Molecular Generation with a Neighborhood-Guided Set Transformer

**arXiv ID:** 2609.05097 | [PDF](https://arxiv.org/pdf/2609.05097v1)

**作者:** Roxane Axel Jacob `[一作]` (University of Vienna), Johannes Kirchmair `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为NEAT-POCKET的自回归3D分子生成模型，可在蛋白质结合口袋中直接生成分子；

**💡 创新点**

核心创新在于结合细粒度交叉注意力与自适应层归一化，将结合口袋信息嵌入已训练的NEAT框架，实现快速、可控的口袋条件生成，并天然支持片段完成；

**🔧 技术方法**

技术包括Transformer式自回归生成、Neighborhood‑Guided Set Transformer、交叉注意力、AdaLN、无条件指导（cfg）以及基于flow‑matching的坐标预测；

**📊 数据集**

使用GEOM‑Drugs进行无条件预训练，随后在CrossDocked和SPINDR这两个结合口袋生成基准数据集上进行微调；

**📈 对比分析**

与Pocket2Mol、TargetDiff、DiffSBDD、DrugFlow以及FLOWR等基线相比，NEAT-POCKET在PoseBusters有效率、蛋白-配体碰撞、物理可行性、分子质量与物理化学相似度上均保持竞争力，并且生成速度提升约20‑30倍；

**⚠️ 局限性**

局限性包括易受训练数据偏差影响、对蛋白动态与水分子未建模、以及自回归序列生成可能导致错误传播，需在实际药物设计中进一步验证与优化。

---

## 392. ToPos: Automated Optimal Positioning on Topographic Manifolds using Constrained Geodesic Voronoi Decomposition

**arXiv ID:** 2609.05084 | [PDF](https://arxiv.org/pdf/2609.05084v1)

**作者:** Rajesh Raveendran `[一作]` (University of Oulu), Juha Röning `[通讯]` (University of Oulu)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一套名为ToPos的自动化框架，用于在三维等高地形表面上实现空间参考点（GCP/GSP等）的面积平衡分布，解决传统二维欧氏方法导致的测距失真与空间聚集问题。

**💡 创新点**

创新点包括：①将地形视为离散二维流形并采用约束几何Voronoi分解；②使用热方法（Heat Method）高效计算流形上内在测地线距离；③引入黎曼尼采夫斯基加速梯度（Riemannian NAG）求解器，实现快速收敛并保持几何一致性；④设计硬/软流形约束，兼顾安全与覆盖需求。

**🔧 技术方法**

技术手段包括：离散二维流形（三角网）表示、热方法求解Eikonal方程获取测地线距离、约束几何Voronoi分解、黎曼尼采夫斯基NAG优化、GIS微服务架构（Python、PyQt6、GDAL、Potpourri3d、NumPy）、投影与约束投影操作。

**📊 数据集**

实验使用合成非凸正弦波形高程表面（100m×100m，16,900个顶点），在此表面上加入硬性作业空洞与软性不可通行区进行验证。

**📈 对比分析**

与三种基线方法（二维Lloyd、适应面积平衡、黎曼动量）相比，ToPos在100个采样点、100次迭代下CV（面积方差系数）降低74.17%，相对基线提升55%；计算时间每步仅比基线增加约6%（<800 ms），实现了显著的几何精度提升与可接受的计算成本。

**⚠️ 局限性**

局限性：仅在合成数据上验证，缺乏对真实复杂地形或多约束场景的深入评估；对动态变化的地形或实时更新支持尚未完善；在极大规模点云或高分辨率网格下的可扩展性与内存开销需进一步研究。

---

## 393. On the Delay-Constrained Maximum Concurrent Flow Problem

**arXiv ID:** 2609.05068 | [PDF](https://arxiv.org/pdf/2609.05068v1)

**作者:** Walid Ben-Ameur `[一作]` (SAMOVAR, Telecomm SudParis), Sebastien Martin `[通讯]` (Huawei Technologies Ltd)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并求解延迟受限多路复用最大并发流（DCMCF）问题，证明其强NP难度并给出多项式近似算法；

**💡 创新点**

创新点在于构造了基于二阶锥约束的凸包松弛，显著优于先前的离散/分支松弛，并以此得到可证明的近似比；

**🔧 技术方法**

采用凸包（convex envelope）技术、二阶锥规划（SOCP）表示、分支定界以及启发式子集选择法；

**📊 数据集**

使用SND-Lib的实际网络实例和随机Erdős–Rényi生成的网络，均提供多路可选路径集合；

**📈 对比分析**

通过与BIG‑M、离散（disjunctive）松弛及启发式基准的比较，发现CONVEX松弛在大多数实例上给出更紧的上界，启发式ℋ_threshold在规模更大时优于ℋ_greedy；

**⚠️ 局限性**

局限性包括对单源单汇情形的复杂度尚未确定、松弛虽强但求解时间较高、近似比受参数（|P|、u、b）影响，需要进一步改进可扩展性与精度。

---

## 394. EuroAlpaca: Task-Preserving Localisation of Instruction Data for European Languages

**arXiv ID:** 2609.05043 | [PDF](https://arxiv.org/pdf/2609.05043v1)

**作者:** Aleix Sant `[一作]` (Telefónica Innovación Digital), Carlos Escolano `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于LLM的任务保留本地化流水线，并生成覆盖50种欧洲语言及其方言的近并行指令数据集。

**💡 创新点**

创新点在于将样本级与字段级翻译策略、任务级重写以及最终一致性验证相结合，显著减少直接机器翻译导致的任务失真，并提供了兼顾参考相似度与可验证指令遵循的多元评估框架。

**🔧 技术方法**

主要技术包括Gemma‑4用于任务与域标注、样本级/字段级策略预测、NLLB‑3.3B进行政策驱动的机器翻译、LLM判定与一致性验证，以及LoRA微调多种3–4B参数量的LLM。

**📊 数据集**

使用Alpaca Cleaned作为源数据，生成EuroAlpaca（覆盖50种欧洲语言）和european‑ifeval评估集；同时对外部资源Okapi、MITS、Bactrian‑X进行对比。

**📈 对比分析**

与直接MT、Task‑Preserved及外部资源进行对比，四大模型在Aya ROUGE‑L/Fbert和Acc（验证指标）上均显著优于直接MT，平均提升约0.1–0.2分，且在200个模型‑语言组合中保持最高性能。

**⚠️ 局限性**

局限性包括仅覆盖欧洲语言，依赖Gemma‑4导致低资源方言质量受限；未评估训练随机性带来的方差；与外部资源的对比受语言覆盖差异限制。

---

## 395. Westlake Scholar: AI-Enhanced Scholarly Discovery over an Institutional Repository

**arXiv ID:** 2609.05072 | [PDF](https://arxiv.org/pdf/2609.05072v1)

**作者:** Junshu Pan `[一作]` (Westlake University), Rui Shang `[通讯]` (Westlake University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了 Westlake Scholar——一个基于机构信息库的开源 AI 辅助学术发现平台，集成了上下文阅读、跨文献发现、机构专家匹配和研究历程生成四大功能。

**💡 创新点**

创新点在于：① 将机构已批准的论文记录、全文文本和学者–论文关联作为共享知识层，统一支撑所有 AI 任务；② 设计了“进阶式学术发现”工作流，让用户从单篇论文切入逐步扩展到相关文献、专家和研究轨迹；③ 通过机构治理保障数据审核、模型处理与结果发布的可追溯性与可修正性。

**🔧 技术方法**

使用技术包括：检索增强语言模型（OpenAI‑compatible chat & embedding APIs）、混合词典‑向量检索（BM25 + 语义向量）、段落分割与重叠（1000 字/100 字重叠）、递归检索与逆序排名融合、语义检索工具调用、论文摘要与标签生成的 Prompt‑Engineering、OpenAlex/Crossref 接口获取 DOI 计时信息，以及前端交互（PDF 内嵌阅读器 + 对话面板）。

**📊 数据集**

数据集：Westlake University 内部的已审核论文集合（PDF、标题、摘要、作者信息）、学者–论文关联表、以及通过 OpenAlex/Crossref 检索到的 DOI 元数据。全部数据均保持在机构内部，未公开发布。

**📈 对比分析**

方法比较：论文将 Westlake Scholar 与传统图书馆 AI 助手、CRIS/RIMS、VIVO 等系统对比，主要从数据基础、用户焦点与系统关系角度描述；未给出量化性能指标，主要关注功能集成与治理实现；部署在 Westlake University 自身环境中，演示了可行性与可持续性。

**⚠️ 局限性**

局限性包括：仅在单一机构部署，缺乏跨机构泛化；AI 生成结果需人工审核，工作量和时效性仍是挑战；隐私与可解释性需进一步完善（如学者同意、结果上报流程）；当前功能未涵盖外部文献检索与大规模多机构协作；技术上对模型访问与费用有依赖。

---

## 396. MultiAttenGastro: Multi-Dimensional Attention Augmentation for Gastrointestinal Endoscopy Classification

**arXiv ID:** 2609.05070 | [PDF](https://arxiv.org/pdf/2609.05070v1)

**作者:** Sadhana Devarajan `[一作]` (Sardar Vallabhbhai National Institute of Technology), Kiran Raja `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种可插拔的多维注意力模块 MultiAttenGastro，用于胃肠内镜图像分类。

**💡 创新点**

创新点在于将 1D 通道、2D 空间和 3D 上下文注意力并行融合，并通过跨数据集、多骨干和多种统计检验揭示其效果随 ImageNet 与目标域代表性差距的显著相关性。

**🔧 技术方法**

使用了 1D/2D/3D 并行注意力分支、全局平均池化、线性分类层，并辅以 CKA、Grad‑CAM 和多种显著性检验（配对 t、Wilcoxon）进行评估。

**📊 数据集**

在五个公开 GI 内镜数据集上实验：Kvasir‑Capsule、SEE‑AI、Kvasir‑v2、HyperKvasir 与 GastroVision。

**📈 对比分析**

在 8 个 CNN/Transformer 骨干上共 80 次实验，发现 MultiAttenGastro 在代表性差距大的 Kvasir‑Capsule 上宏 F1 提升至 98.33%（+1.05%），在差距小的 Kvasir‑v2 上统一下降；在中等差距数据集表现混合，整体效果呈现“仅在大域差时有效”的趋势。

**⚠️ 局限性**

局限性包括：多种指标仅在单个数据集上达到差异无统计显著性，无法证明因果关系；仅在所选骨干和数据集上验证，未涵盖更广泛的模型或临床场景；并且对小类样本、不同预处理策略的敏感性仍待深入探究。

---

## 397. Towards Efficient Evaluation of Evolutionary Transfer Optimization: Case Studies on Task-Parameterized Applications

**arXiv ID:** 2609.05040 | [PDF](https://arxiv.org/pdf/2609.05040v1)

**作者:** Yanchen Li `[一作]` (Hong Kong Polytechnic University), Kay Chen Tan `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

针对任务参数化的进化迁移优化，本文通过将串行评估运算改写为矩阵形式，提出两种针对跨任务与单任务扩展的高并行评估重构。

**💡 创新点**

创新点在于提出任务侧评估的矩阵重构方法（累积矩阵与混合矩阵），在不改变数值精度的前提下显著降低评估时间并提升整体求解效率。

**🔧 技术方法**

使用矩阵累积与混合、并行矩阵乘法、CUDA GPU并行计算与EvoX框架实现并行评估。

**📊 数据集**

使用Sobol序列生成的随机关节长度与角度任务、基于B样条的二维障碍物轨迹任务（包含20个随机方形障碍）做实验。

**📈 对比分析**

通过与原始串行评估在单体评估与整个优化过程的运行时间比较，跨任务评估实现256.72×加速，单任务评估实现93.91×加速，保持误差<1e-6。

**⚠️ 局限性**

局限性包括仅针对两类连续任务参数化问题，重构需要针对特定应用设计，且在大规模任务时可能增加内存占用。

---

## 398. A Human-in-the-Loop Framework for AI-Assisted Scoring in Large-Scale Writing Assessment

**arXiv ID:** 2609.05143 | [PDF](https://arxiv.org/pdf/2609.05143v1)

**作者:** María Eugenia Curi `[一作]` (Ceibal), Andrés Peri `[通讯]` (ANEP)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一个人机协同的 AI 辅助写作评分框架，应用于乌拉圭全国中等教育认证考试的 150–200 词短篇议论文评分。

**💡 创新点**

将大语言模型与人工审核结合的工作流，并通过分析 AI 与人工在 rubric 维度、水平与通过/不通过决策上的一致性，证明其可安全用于高风险评估。

**🔧 技术方法**

使用 GPT‑5 进行提示式评分，对拼写错误使用 LanguageTool；结合 IRT 与 Bookmark 方法设定分数阈值，构建人机协同评分流程。

**📊 数据集**

采用 2024 与 2025 年乌拉圭全国考试约 5,000–6,000 份写作样本及其人工评判的 15 项 Rubric 分数作为数据集。

**📈 对比分析**

通过准确率、Cohen's Kappa、IRT 级别一致性和通过/不通过混淆矩阵比较 AI 与人工；准确率 60–80%，Kappa 中等偏上；AI 对“通过”预测误差低，误判为不通过约 15%，但可通过人工复核解决。

**⚠️ 局限性**

以人工评分为真值导致本身变异；实验为离线，缺乏实时监测与漂移检测；对低频词汇、句法等细粒度项的准确率仍有限。

---

## 399. Single-Query Black-Box Calibration Auditing via Logit Bias

**arXiv ID:** 2609.05125 | [PDF](https://arxiv.org/pdf/2609.05125v1)

**作者:** Roman Plaud `[一作]` (Institut Polytechnique de Paris), Willem Waegeman `[通讯]` (Ghent University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种仅需一次API查询即可评估黑盒LLM校准误差的估计器，解决了传统方法需访问连续概率的不现实性。

**💡 创新点**

创新点在于利用API的bias参数对阈值进行数学映射，从而实现单查询精确阈值检测，并证明其一致性与真校准误差相匹配。

**🔧 技术方法**

使用的技术包括阈值偏置推导、随机分区估计、理论误差分析以及对大模型日志提取的优化。

**📊 数据集**

实验使用BoolQ和二值化MMLU数据集，模型涵盖Qwen-2.5-7B-Instruct、Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3和Gemma-2-9B-IT。

**📈 对比分析**

与verbalized confidence、Monte Carlo sampling及Iterative Logit Extraction基线相比，该方法在单查询预算下平均MAE低于0.01，几乎匹配高查询量基线并显著降低成本。

**⚠️ 局限性**

局限性包括对单一token对的依赖、对bias参数可用性的要求以及在多类分类场景下的扩展性待验证。

---

## 400. Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting

**arXiv ID:** 2609.05119 | [PDF](https://arxiv.org/pdf/2609.05119v1)

**作者:** Bogdan Cebere `[一作]` (CISPA Helmholtz Center for Information Security), Christian Rossow `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过统一评估框架，对 HTTP/3 的应用层流量防护机制进行实验模拟、性能校准和评测，验证并发掘了多种已知与新型防护技术对子页面指纹攻击的抵御效果。

**💡 创新点**

创新点在于：①将 HTTP/3 的多路复用、流优先级、流控制以及 103 Early Hints 等原生特性转化为可用的隐私防护工具；②提出 H2PC/H2PS 等轻量级、端点统一的防护策略；③设计了针对每个数据集、每种防护参数的“校准”流程，兼顾攻击者最优配置与隐私-开销折衷；④通过信息论泄漏估计（WeFDE、DeepSE-WF）量化剩余不确定性。

**🔧 技术方法**

技术手段包括：网络层流量伪装（噪声插入、填充、分割、定时控制）、HTTP/3 特性驱动的防护（流优先级重排、流控制窗口调节、PING 伪装、Early Hints 推测等）、深度学习指纹模型（k-FP、DF、VarCNN、RobustFP-CNN、Holmes）以及信息理论泄漏估算。

**📊 数据集**

使用了五个公开网站（Amazon、BBC、Reddit、Udemy、Wikipedia）各 100 个子页面的数据集，分别生成 500 条已加密的 PCAP 流量（共 25 000 条），并在自建 Python 客户端-服务器环境中重放以保证同一资源在不同防护下的可比性。

**📈 对比分析**

评测方法：①在每个数据集上先用多种防护参数做“校准”，挑选在保留低泄漏的同时不至于产生过大开销的配置；②对每个防护方案使用 5 种最佳的指纹模型（k-FP、DF、VarCNN、RobustFP‑CNN、Holmes）进行攻击并报告宏 F1 与 Top‑k 识别率；③采用两种信息论估计器得到残余不确定性 𝒦*；④测量上传、下载和页面加载时间的相对增量。结果显示：客户端 CL‑Tamaraw 在多数站点获得最低 Macro‑F1（≤0.42）但开销高；H2PC 在 7–20 倍的 𝒦* 之间提供了更低成本的保护；服务器侧 SRV‑Tamaraw/ALPaCA 在关键连接处可将 Macro‑F1 降至 0.1–0.3，但需要针对每个网站的部署；H2PS 通过 103 Early Hints 在首方服务器实现 13–95 的 𝒦*，且相对开销最低。

**⚠️ 局限性**

局限性包括：①防护效果高度依赖网站特征与防护参数，需针对每个站点单独校准；②强防护方案（如 CL‑Tamaraw）导致显著带宽/时延开销；③服务器侧方案受部署位置限制（需协同 CDN 或多域名支持）；④实验基于 HTTP/3，未覆盖旧版 HTTP 或其他加密隧道；⑤信息论估计仅为下限，真实泄漏可能更高。

---

## 401. A Comparative Study of Counterfactual Explainers for Graph Neural Networks Enabling Multiple Types of Graph Edit

**arXiv ID:** 2609.05113 | [PDF](https://arxiv.org/pdf/2609.05113v1)

**作者:** Maria Myrto Villia `[一作]` (Foundation for Research and Technology - Hellas), Panos Trahanias `[通讯]` (Foundation for Research and Technology - Hellas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对支持边添加与删除的对抗性 GNN 解释器进行系统评测，比较其在图分类与节点分类任务中的表现。

**💡 创新点**

首次在统一实验框架下全面对比多种对抗性 GNN 解释器，提出多维评价指标并揭示各方法的优势与局限。

**🔧 技术方法**

结合 GAN、扩散模型、风格迁移、强化学习等技术实现六种解释器，并在同一框架内对其进行统一实现与评估。

**📊 数据集**

图分类使用4个合成数据集（BA-2Motifs、BA-3Motifs、BA-4Motifs、BA-2Motifs-3Classes）和3个真实数据集（BBBP、Twitter、Graph-SST5）；节点分类使用BA-Shapes、Tree-Cycles、Cora和PubMed。

**📈 对比分析**

采用有效率、解释大小、可信度、基底接近度、最小化程度以及训练/推理时间等指标进行比较。结果显示无单一方法在所有指标上最优；如 RSGG-CE 训练快、解释小但质量中等；D4Explainer 可信度高但耗时长；GCFExplainer 覆盖率高但解释大。

**⚠️ 局限性**

缺乏标准化基准与统一评估指标；实验主要聚焦图分类，对节点/链路预测关注不足；未覆盖异构或动态图；对解释可解释性与鲁棒性评估不充分。

---

## 402. Compact Bellman-Grounded Cognitive Maps for Cost-Aware Navigation

**arXiv ID:** 2609.05104 | [PDF](https://arxiv.org/pdf/2609.05104v1)

**作者:** Yuzhe Han `[一作]` (Guangdong Institute of Intelligence Science and Technology), Yujie Wu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种成本意识的认知地图模型BCM，用于在已知加权图上重复规划，支持随目标变化而无需重训练。

**💡 创新点**

核心创新是将Bellman准则与自监督目标结合，直接用边权和下游最优评分学习目标条件转移排名，并通过交互式坐标编码实现参数子线性增长。

**🔧 技术方法**

技术包括基于坐标的节点编码（交叉轴交互式k-hot）、差分转移编码器、Bellman‑grounded自监督损失以及访问记录的贪心读取。

**📊 数据集**

实验在三种二维加权网格（U‑shaped、center‑block、wall‑with‑gap）上进行，节点数最多到3600，边权分别为统一、轻度（1–3）和宽泛（1–10）。

**📈 对比分析**

与Dijkstra、EigenAgent、CML、APF等基线对比，BCM在N≤1600时成功率100%，平均Gap仅≈5%，而EigenAgent在相同条件下Gap约45%；在更大图上BCM仍保持较小Gap并实现参数子线性扩展。

**⚠️ 局限性**

局限包括仅适用于固定已知坐标结构的图，顶点/边变化需重新训练；在更大或更密集的迷宫布局下Gap显著上升，且缺乏正式的最优性或完整性保证。

---

## 403. Improving Language Identification for Code-Switched Utterances with Integer Linear Programming

**arXiv ID:** 2609.05099 | [PDF](https://arxiv.org/pdf/2609.05099v1)

**作者:** Joanna Radoła `[一作]` (Sorbonne Université), François Yvon `[通讯]` (Sorbonne Université)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新分析并改进MaskLID的代码切换识别流程，提出基于ILP的优化和改进的语言识别模型。

**💡 创新点**

1) 用更准确的单词级语言关联分数替代原始词级不可靠分数；2) 将MaskLID的迭代掩蔽算法转化为整数线性规划，支持多种可解释约束。

**🔧 技术方法**

采用改进的GlotLID（基于fastText）作为后端，结合整数线性规划（Gurobi）和Pyomo框架进行优化。

**📊 数据集**

使用LinCE的多语言混合数据集（Turkish-English、Turkish-German、Basque-Spanish等）和Flores+的单语句集，共十种语言。

**📈 对比分析**

与原MaskLID和其他基准（k=1/2、阈值等）在开发集和测试集上进行EM、F1对比，平均EM提升至0.71（比MaskLID高出0.17），在特定语言对上提升达0.44。

**⚠️ 局限性**

仅支持最多两种语言，短句/多脚本混合处理仍受限；ILP求解耗时较高；未覆盖非拉丁脚本和非罗马化文本。

---

## 404. CAT-LDP: Cloud-edge Adaptive Taxonomy under Local Differential Privacy

**arXiv ID:** 2609.05095 | [PDF](https://arxiv.org/pdf/2609.05095v1)

**作者:** Junzhe Yang `[一作]` (Sichuan University), Xinye Chen `[通讯]` (Sichuan University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了基于类别树的云端‑边缘协同推荐框架 CAT‑LDP，在局部差分隐私约束下实现隐私保护的分层推荐。

**💡 创新点**

创新点在于：①采用层次化类别树结构降低维度并聚合语义相似项目；②本地采用自适应隐私预算分配，将预算聚焦于用户活跃类别；③云端进行粗粒度聚类筛选，边缘设备使用未扰动历史进行细粒度 Jaccard 重排，实现云‑边缘解耦。

**🔧 技术方法**

使用技术包括：二值随机响应（BRR）实现 LDP；自适应预算约束下的预算分配与二分搜索；云端多轮 K‑means + 谱聚类的鲁棒图聚类；粗粒度评分采用乘法门控；边缘重排使用 Jaccard 相似度。

**📊 数据集**

采用 Amazon Video Games（5‑core）数据集，约 54k 用户、16k 商品，稀疏度 99.96%。

**📈 对比分析**

在 leave‑two‑out + 1+99 采样评估下，以 HR@K 与 NDCG@K 衡量，CAT‑LDP 在 ε=1.0 下无论是 CoarseOnly 还是 HybridLocalRerank 均显著优于 CT‑LDP、LCF‑SP/AP、DPLCF‑SP/AP 等基线，HR@10 提升约 17% 以上。

**⚠️ 局限性**

局限性包括：①依赖完善的类别树结构，若缺失或不准会影响效果；②仅在单一数据集验证，跨数据集泛化仍待验证；③重排依赖元数据，若元数据缺失会降低性能；④在极低 ε 下仍受高维稀疏噪声影响，性能下降明显。

---

## 405. MePo++: Unifying Representation Refinement and Reconciliation for General Continual Learning

**arXiv ID:** 2609.05075 | [PDF](https://arxiv.org/pdf/2609.05075v1)

**作者:** Guanglong Sun `[一作]` (Tsinghua University), Yi Zhong `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MePo++ 框架，通过预训练后置训练和在线特征调和两阶段方法提升在一般连续学习（GCL）中的性能。

**💡 创新点**

创新点：1) MetaPrep 利用无监督聚类构造伪连续序列并进行双层元学习，预训练模型在上线前提升可适应性；2) StreamAlign 通过将在线特征与稳定的几何先验对齐并进行语义对齐，解决无任务边界环境下的对齐缺口。

**🔧 技术方法**

技术：无监督聚类、双层元学习、几何变换（Cholesky 变换）、监督对比学习（语义对齐）以及传统的分类损失。

**📊 数据集**

数据集：CIFAR‑100、ImageNet‑R、CUB‑200；预训练模型包括 Sup‑21K、Sup‑21/1K、DINOv2 等 ViT-B/16、ViT-B/14 后端；还在 CLIP 视觉‑语言模型上验证。

**📈 对比分析**

与多种基线（Seq FT、L2P、DualPrompt、MVP、MISA 等）比较，MePo++ 在 A_AUC 与 A_Last 上平均提升约 8–12%（相对 10–30%），在少量样本和 CLIP‑based GCL 场景也保持显著优势。

**⚠️ 局限性**

局限性：MetaPrep 需要额外的预部署后置训练阶段，增加计算与存储成本；StreamAlign 依赖固定几何先验，若遇长期或剧烈分布漂移，先验可能失效。

---

## 406. Confounding-Valid Conformal Inference for Counterfactual KPIs in Wireless Networks

**arXiv ID:** 2609.05073 | [PDF](https://arxiv.org/pdf/2609.05073v1)

**作者:** Abdessamed Qchohi `[一作]` (EURECOM), Matteo Zecchin `[通讯]` (EURECOM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为 CV-CCI 的方法，利用有限的随机化遥测与大量可能受混杂影响的观察遥测来构建可靠的因果 KPI 预测集。

**💡 创新点**

创新点在于将 General Synthetic‑Powered Inference (GESPI) 与加权合成预测 (WCP) 结合，既保持有限样本覆盖率，又能利用偏倚的观测数据显著提高预测效率，并通过可调的容差 ε 控制误差上限。

**🔧 技术方法**

使用加权合成预测、倾向分数/密度比估计、非一致性评分校准以及 GESPI 聚合规则来构造预测集。

**📊 数据集**

在两类典型 RAN 控制任务上进行实验：MAC‑层调度（基于 Nokia Wireless Suite 仿真器生成的 UE backlog、CQI、PRB 预算数据）和切换（基于 SionnaRT ray‑tracing 产生的 RSS 历史与邻接基站负载数据）。

**📈 对比分析**

与 CCKE、wSCP‑DR（Inexact/Exact）以及 Guardrail 等基线比较；实验显示 CV-CCI 在所有混杂强度下保持接近目标覆盖率，同时预测区间宽度显著优于仅使用随机化遥测的方法，且相较于 wSCP‑DR Exact 的过宽区间更为精细，且在不同 ε 设定下稳定性更好。

**⚠️ 局限性**

局限性包括对有限随机化样本的依赖（若随机化样本更少，效能可能下降）、对倾向估计误差敏感，以及目前仅在单步、静态决策场景验证，尚未扩展到多步骤或交互式网络控制。

---

## 407. Performance Evaluation of HAPS-enabled Coverage Enhancement in Hard-to-Reach Areas

**arXiv ID:** 2609.05067 | [PDF](https://arxiv.org/pdf/2609.05067v1)

**作者:** Hao Lin `[一作]` (King Abdullah University of Science and Technology), Mohamed-Slim Alouini `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

对高空平台站(HAPS)在难以覆盖地区的下行/上行覆盖性能进行建模与分析。

**💡 创新点**

采用随机几何框架考虑HAPS方向波束与地面网络共存，推导距离分布并给出覆盖概率与设计指南。

**🔧 技术方法**

使用随机几何、Nakagami-m衰落、ITU HAPS天线模式、拉氏变换及闭式/近似表达式进行性能分析。

**📊 数据集**

通过蒙特卡洛仿真验证，参数取自亚马逊雨林、地中海等典型难以覆盖区域的设定，无使用公开数据集。

**📈 对比分析**

与无HAPS、全向天线以及不同波束宽度、HAPS数量、海拔高度组合进行仿真比较，结果表明适量HAPS与窄波束可显著提升覆盖概率。

**⚠️ 局限性**

仅适用于圆形难区、采用最优关联规则、未考虑干扰管理、切换与真实地形复杂性。

---

## 408. Beyond Bias: Participatory and Reflective Approaches to Cultural AI

**arXiv ID:** 2609.05102 | [PDF](https://arxiv.org/pdf/2609.05102v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 409. Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources

**arXiv ID:** 2609.05059 | [PDF](https://arxiv.org/pdf/2609.05059v1)

**作者:** Dmitrij Żatuchin `[一作]` `[通讯]` (Estonian Entrepreneurship University of Applied Sciences), Dmitrij Żatuchin (Estonian Entrepreneurship University of Applied Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过多轮相同购买类问题对比语言模型的品牌推荐累积情况，验证检索功能是否能使推荐列表快速饱和。

**💡 创新点**

创新点在于将生态学的种群累积方法（rarefaction、Chao2估计）引入对LLM输出的品牌与引用域的数量和多样性进行系统性量化，并揭示检索功能是驱动饱和的关键因素。

**🔧 技术方法**

使用了rarefication曲线、Chao2下限估计、稀有度计数（Q1、Q2）以及两种提取方法（开源提取与规则提取）来评估品牌和引用域的累积；实验通过多引擎多轮调用实现。

**📊 数据集**

实验数据来源于50个行业内购买问题，分别在六大语言模型引擎（五个不带检索、一个带检索）上运行15轮，得到共4,500条回答；另外再使用24轮深度检索实验验证。

**📈 对比分析**

对比方法包括：计算每个引擎在第15轮时的品牌/域数、Q1>0比例、Chao2覆盖率及A(1)/A(5)比值；结果显示无检索引擎在第15轮仍有86–92%单元在增添新品牌，平均积累31–31个品牌；检索引擎在第15轮已完成近100%估计，平均仅8个品牌。

**⚠️ 局限性**

局限性包括：仅覆盖五个行业的50个问题，未涵盖多语言或其他类别；检索配置单一；未评估推荐频率的稳定性；实验未深入探讨不同解码策略或模型版本对累积的影响。

---

## 410. NS-ST-GraphRAG: Neuro-Symbolic Spatio-Temporal GraphRAG for Literary Knowledge Processing

**arXiv ID:** 2609.05139 | [PDF](https://arxiv.org/pdf/2609.05139v1)

**作者:** Zheng Kui Lin `[一作]` `[通讯]` (Dalian Ocean University), Zheng Kui Lin (Dalian Ocean University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了 NS-ST-GraphRAG，结合神经-符号约束、空间时间图建模与动态子图检索，以解决长篇文学叙事的多跳问答。

**💡 创新点**

创新点在于：①神经-符号层对提取的三元组进行硬/软规则检查并修正；②双时间坐标（章节级与事件级）和空间场景属性实现叙事时空切片；③构建首个经典中文文学多跳 QA 基准 Red-Chamber-QA。

**🔧 技术方法**

使用的技术包括 LLM 基于 schema 的实体关系抽取、领域本体规则过滤、事件时间线抽取、图数据库版本化存储、基于索引的子图检索，以及严格的评估协议。

**📊 数据集**

数据集为《红楼梦》全文（120 章）以及公开的《骆驼祥子》做跨域评估，并在 Red‑Chamber‑QA 104/120 题集上公开基准。

**📈 对比分析**

与传统向量检索、GraphRAG、LightRAG、HippoRAG 等基线比较，NS-ST-GraphRAG 在冻结测试上机械复制率 0.733、语义判断 0.866，citation‑faithfulness 0.825，略优于章节级 BM25‑lite，提升了时间/空间约束题的回答质量。

**⚠️ 局限性**

局限在于：①神经-符号约束规则覆盖仍有限，未能完全消除所有幻觉；②子图检索依赖手工本体与规则，维护成本高；③在时间约束题上收益有限，仍未显著超越一般题目；④对非中文或非小说文本的泛化尚未充分验证。

---

## 411. GradRig: Differentiable Weights for Skinned Gaussian Splat Deformation

**arXiv ID:** 2609.05127 | [PDF](https://arxiv.org/pdf/2609.05127v1)

**作者:** Nina Vesseron `[一作]` (ENSAE-CREST), Élie Michel `[通讯]` (Adobe)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于皮肤权重梯度的全GPU皮肤化高斯斑点（Gaussian splat）变形框架，使得在不使用网格代理的情况下即可实时交互式地对3D高斯斑点云进行形变；

**💡 创新点**

创新点在于：①利用皮肤权重梯度对斑点进行弹性伸缩，避免因邻近点不同变形而产生的孔洞；②在作者工具中自动求导生成权重梯度，避免后期邻居差分估计；③引入基于字典的自适应重采样后处理，仅在需要时分裂斑点以消除残留失真；

**🔧 技术方法**

主要技术包括：线性混合皮肤化、梯度驱动的斑点协方差变形、视点依赖色彩变换、基于GPU实例化渲染、自动微分（autograd）生成权重梯度、预计算的重采样字典与阈值判定；

**📊 数据集**

使用了多种公开的高斯斑点场景（如香蕉、汽车、花朵、喷泉、拖拉机等）进行实验与评测；

**📈 对比分析**

与 Gao 等人提出的实时大规模变形方法进行了对比；结果表明本方法在保持全高斯斑点空间的同时能实现至少相同规模的变形；在Apple M1 Max上测试表明新增的 per‑splat 变换对渲染帧率影响不大，主要成本是内存占用（每个斑点额外10字节）；

**⚠️ 局限性**

局限性包括：需为每个斑点存储梯度导致内存开销；仅实现线性混合皮肤化，未涵盖四元数等更复杂皮肤模型；字典重采样使用最近邻插值，未采用更精细的插值或全局优化；仅在静止形状上进行重采样，无法直接适应所有动态变形范围。

---

## 412. LayoutShop: Content-Constrained Exploratory Design of Creative Article Layout

**arXiv ID:** 2609.05098 | [PDF](https://arxiv.org/pdf/2609.05098v1)

**作者:** Jialuo Li `[一作]` (Shenzhen University), Pengfei Xu `[通讯]` (Shenzhen University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种计算框架，帮助用户通过选择或创建版式模板、构建设计空间、优化几何结构，并使用双神经网络评估，快速生成创意且符合排版原则的文章版式。

**💡 创新点**

创新点在于将手工版式草图过程转化为模板混合与结构抽取的两阶段流程，并用混合整数二次规划精确满足文本、图片和排版约束，最终通过视觉与结构评估网络筛选高质量结果。

**🔧 技术方法**

技术包括模板树结构抽象与剪枝、MIQCQP（混合整数二次约束二次规划）求解、基于图卷积网络的结构评估、基于卷积网络的视觉评估，以及交互式几何与结构编辑工具。

**📊 数据集**

使用的公开数据集包括Magazine、PubLayNet、Architecture+Detail和The Public Domain Review，用于模板提取、网络训练和用户实验。

**📈 对比分析**

与GRIDS基准对比，本文方法在相同文章下平均生成时间从1.974 s降至0.106 s，并在用户研究中得到更高的可用性评分和布局质量评分（平均5分制4.4分），实验显示自动生成布局的质量显著优于传统编辑器。

**⚠️ 局限性**

局限性包括高层次设计原则难以完全纳入优化模型，导致某些生成版式结构异常；交互编辑工具受限于自动化几何约束，专业设计师在细粒度修改时体验不佳；目前仅支持轴对齐排版，未涵盖非对齐或自由布局。

---

## 413. Deep Microcompression: Structured Pruning and Bit-packed Quantization for Microcontrollers

**arXiv ID:** 2609.05081 | [PDF](https://arxiv.org/pdf/2609.05081v1)

**作者:** Opegbemi Matthias Busoye `[一作]` (PowerLabs Technologies), Eghonghon-aye Eigbe `[通讯]` (PowerLabs Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套端到端的 Deep Microcompression (DMC) 管线，集成结构化剪枝、量化感知训练和硬件友好的固定长度位打包，实现了在裸机微控制器上部署 CNN 的可行性；

**💡 创新点**

创新点在于将位打包硬件化，生成无依赖、常数时间解包的 C 代码，兼顾极低内存、低功耗的 8 位 MCU，首次实现在 2KB SRAM 的 ATmega328P 上运行标准 CNN；

**🔧 技术方法**

使用结构化通道剪枝、量化感知训练 (QAT)、固定长度 4×2 位或 2×4 位打包、整数运算推理、预先生成的 LUT 等技术；

**📊 数据集**

主要实验数据集为 MNIST 上的 LeNet-5 网络；

**📈 对比分析**

与 TensorFlow Lite Micro 及 FP32 基线比较，DMC-Ultra 在权重量化后压缩率高达 55.8×、准确率 98.77%，二进制体积比 TFLite Micro 小 3 倍，能够在 RP2040 上匹配其准确率；在 ATmega328P 上，DMC-Tiny 通过激活空间剪裁将峰值 SRAM 降至 1.27 KB，支持 98.17% 的准确率，展示了优越的内存占用与实时性；

**⚠️ 局限性**

限制在于仅针对小型 CNN 验证，激活内存仍是瓶颈；目前仅支持固定点整数运算，难以直接扩展至更大或 Transformer 结构；位打包方式对不同网络架构的通用性仍需进一步验证；

---

## 414. TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents

**arXiv ID:** 2609.05079 | [PDF](https://arxiv.org/pdf/2609.05079v1)

**作者:** Zhibo Yang `[一作]` (TruthInsight AI), Hao Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个专注于科学发现的自动化研究代理基准（DiscoveryBench），通过40个无目标盲任务评估代理的科研判断能力。

**💡 创新点**

创新点包括：①把任务、数据和评分从目标导向重现转变为发现导向；②构建了6维、29项可验证的证据成熟度评分框架；③实现全自动化、artifact‑grounded的评估流程，消除人工评分瓶颈。

**🔧 技术方法**

技术实现：使用LLM判定器（GLM‑5.1 4‑bit量化版）自动评判，评估四个主流编码代理（Claude Code、Codex CLI、OpenScience、DeepSeek Harness）在同一Frozen基模型DeepSeek‑V4‑Flash下的表现；通过自动化脚本收集并解析报告、代码与中间结果。

**📊 数据集**

数据集：40份来自2018‑2024年高水平期刊（Nature Communications、Science Advances 等）的实验/观测数据，涵盖天文、化学、计算数学、地球科学、能源、信息科学、生命科学、材料科学、神经科学与物理等10个学科。

**📈 对比分析**

比较方法：对每个代理执行40个任务，统计每项的29条评分项并合成6维度得分；计算平均任务得分并与95%置信区间比较。结果显示四个代理得分仅在58.4–60.3（满分100）之间，未出现统计显著差异；发现维度（控制测试、稳健性、可证伪性、跨数据集泛化）得分低至4–6%，表明发现能力远落后于执行能力。

**⚠️ 局限性**

限制：仅使用单一Frozen基模型且每个代理仅进行一次运行；任务为单阶段数据，跨数据集泛化维度受限；评估完全自动化但LLM判定器的专家校准和鲁棒性待进一步验证；未考虑多种模型、多种随机种子或真实实验环境的扩展。

---

## 415. SciDocBench: A Workflow-Centered Benchmark and Data Pipeline for Scientific Document Understanding

**arXiv ID:** 2609.05141 | [PDF](https://arxiv.org/pdf/2609.05141v1)

**作者:** Shenxi Wu `[一作]` (Chinese University of Hong Kong), Dahua Lin `[通讯]` (Centre for Perceptual and Interactive Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了SciDocBench，一个面向科学文档理解的工作流中心化基准，包含124条专家撰写、难度筛选的多语言问题，并通过全图像与Markdown交错两种文档表示形成496个匹配评测实例；同时构建了SciDocIR统一表示和ScidoC可验证训练数据；

**💡 创新点**

创新点在于：①将科学文档理解拆解为七大能力组和19个子任务的工作流视角，②采用双轴匹配（语言+文档表示）实现对输入格式敏感度的可控评测，③提出可验证的训练子任务，生成可审核的监督与强化学习样本，形成闭环的评测-训练生态；

**🔧 技术方法**

使用多模态评测框架（Rule‑based, LLM‑as‑Judge, Execution‑based），结合文档视觉解析、LaTeX语义对齐、结构化记录生成与块级重渲染；

**📊 数据集**

基于arXiv、bioRxiv、OpenReview、PubMed等公开论文，构成124条问题与对应的15K监督+8K RL训练样本；

**📈 对比分析**

对14个主流多模态模型（Claude‑Opus‑5、GPT‑5.6‑Sol、Gemini‑3.6‑Flash等）在496实例上进行评测，最高得分为62.6/100，显示不同模型在感知、提取、验证、跨文档整合、重构与源追踪等方面的分散优势与差距；

**⚠️ 局限性**

局限性包括：基准规模仍有限（124题），可能未覆盖所有科学工作流；训练子任务受限于可验证目标，跨文档与代码/数据的复杂性尚待扩展；评测对语言与文档表示的敏感度高，导致模型排名易变；

---

## 416. From 80x to 385x: A Best-Matching-Unit Search at the L2 Roof, Measured Against a Symmetrically Tuned Baseline

**arXiv ID:** 2609.05138 | [PDF](https://arxiv.org/pdf/2609.05138v1)

**作者:** Andrew James Amos `[一作]` `[通讯]`, Andrew James Amos

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对自组织映射（SOM）中最匹配单元搜索（BMU）进行GPU端调优，并对自研 SparseBin 与 NVIDIA cuSPARSE 两种实现做对称调优，最终提升 BMU 搜索速度 5.6–10.1 倍，SparseBin 对 cuSPARSE 进一步提升 2–3 倍。

**💡 创新点**

① 采用对称调优策略，保证比较基准的公平性；② 通过四个调优杠杆（tile size、tile‑membership clustering、neuron‑axis chunking、vectorised loads）将 BMU 搜索推至 L2 带宽上限；③ 公开完整可复现的代码、数据和调优日志，构建可验证的实验平台。

**🔧 技术方法**

主要技术包括：CUDA GPU 内核调优（寄存器/共享内存分配、线程块调度）、稀疏-稠密矩阵乘法（SparseBin 核实现）与 cuSPARSE 库调用的比较、图形化 roofline 分析、批量大小和算法选择调优、向量化加载等。

**📊 数据集**

使用 MEDLINE 语料库（约 2690 万篇摘要）构造的稀疏二值词向量集合，作为训练和搜索的完整数据集。

**📈 对比分析**

对比方法：在同一 RTX 4090 GPU 上，先对两实现进行相同的调优杠杆；随后测量单 epoch 运行时间并与公开配置做对比。结果显示：SparseBin 对 cuSPARSE 提升 2–3 倍；在 MEDLINE atlas 上，SparseBin 对 MedSOM 的加速比从 80× 提升到 385×，对 somoclu 的加速比从 84× 提升至 2600+×。

**⚠️ 局限性**

仅在单一消费级 GPU（RTX 4090）上验证，带宽/缓存比例特定，未在其他硬件上验证；调优主要关注单 epoch 性能，整体训练过程中的其他阶段未进一步优化；调优空间有限，未探索共享内存分配等更细粒度的硬件资源管理。

---

## 417. Coarse-Graining Hidden Representations: Unsupervised Neuron Selection via Mapping Entropy

**arXiv ID:** 2609.05126 | [PDF](https://arxiv.org/pdf/2609.05126v1)

**作者:** Margherita Mele `[一作]` (University of Trento), Alessandro Ingrosso `[通讯]` (Radboud University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于映射熵（ME）最小化的无监督神经元选择方法，用于在过参数化神经网络中压缩隐藏层并保留最具信息量的子网络。

**💡 创新点**

创新点在于将信息论中的映射熵概念引入神经网络压缩与解释，可通过隐藏层激活统计自适应地识别功能重要的神经元，且不依赖标签或梯度信息。

**🔧 技术方法**

使用技术包括：隐藏状态二值化、ME（映射熵）计算、MEOW（映射熵优化）流程、模拟退火搜索最优子集、Wang‑Landau采样分析映射空间以及激活统计的无监督评估。

**📊 数据集**

实验数据集包括：教师-学生（TS）回归任务、非线性高斯过程（NLGP）二分类任务、以及经过随机平移增强的 MNIST 1‑vs‑7 二分类任务。

**📈 对比分析**

通过将ME选出的子网络与等大小的随机子网络在相同任务下进行对比，结果显示在高压缩（保留隐藏层一半以下神经元）时ME选取的子网络在保持预测性能方面显著优于随机选择，尤其在NLGP和MNIST任务中表现突出。

**⚠️ 局限性**

局限性包括：仅对单层全连接网络验证，未在更深或卷积/Transformer结构上测试；二值化激活可能导致信息损失；计算开销随网络宽度增长；以及该方法未必能找到绝对最优剪枝方案，需结合迭代微调或其他指标进一步改进。

---

## 418. TIER: Threat Implicitness Benchmark for Evaluating LLM Safety Behaviors

**arXiv ID:** 2609.05117 | [PDF](https://arxiv.org/pdf/2609.05117v1)

**作者:** Thu-Hien Trinh-Thi `[一作]` (Vietnam National University), Tram Ho `[通讯]` (Vietnam National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了TIER基准，用以评估大型语言模型在不同威胁隐蔽程度下的安全行为。

**💡 创新点**

创新点在于将威胁隐蔽性分为四层并配合六标签行为分类，对比传统二元评估提供更细粒度的安全分析。

**🔧 技术方法**

采用LLM-as-a-Judge框架，利用两个强指令微调模型（Qwen2.5-7B和Llama-3.1-8B）对模型回应进行六标签打分，并计算攻击成功率。

**📊 数据集**

使用了1,184条跨四风险领域（性内容、暴力、违法活动、自我伤害）的威胁提示，按四个威胁隐蔽层级构建，形成TIER数据集。

**📈 对比分析**

对六个开源LLM（Qwen1.5-MoE、Gemma-2B、Mistral-7B、GPT-J、Llama3、ChatGLM3）进行评测，发现Mistral-7B攻击成功率最高、Llama3最低，且不同模型在相同ASR下表现出不同的行为分布，验证了行为多样性。

**⚠️ 局限性**

局限性包括数据集仅覆盖四个风险领域、评估依赖LLM判断器的主观性、未涵盖更复杂的跨语言或多模态威胁提示，且对真实世界部署安全性的直接可转移性尚未验证。

---

## 419. Embedding Surgery: Localized Updates for Adaptive Ranking Correction in Dense Retrieval

**arXiv ID:** 2609.05110 | [PDF](https://arxiv.org/pdf/2609.05110v1)

**作者:** Maddalena Amendola `[一作]` (Italian Institute of Technology), Raffaele Perego `[通讯]` (National Research Council Institute for Scientific and Technological Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在大规模密集检索中提出一种轻量级的“嵌入外科”方法，实时对查询相关文档的向量嵌入做最小化局部更新以纠正排名错误。

**💡 创新点**

创新点在于将排名校正视为凸二次规划，利用编辑反馈、用户点击或LLM生成的偏好，局部微调文档嵌入而不需重新训练模型或重建索引，且保证嵌入空间整体结构不被破坏。

**🔧 技术方法**

核心技术包括：凸优化（二次规划）、文档嵌入局部调整、基于反馈的约束生成、在ANN索引（IVF、HNSW）上进行原地覆盖更新。

**📊 数据集**

实验使用TREC Deep Learning (DL) 2019/2020、DL‑Hard、TREC Robust、TREC CAsT 2019、MS MARCO Dev、QSharedRel等多种标准检索基准，并对Contriever、TAS‑B、E5、Snowflake等四种密集检索模型进行评估。

**📈 对比分析**

与基线检索器、CoRocchio查询适应方法以及多种反馈来源（人工编辑、模拟点击、LLM重新排序）对比，嵌入外科在nDCG@10上最高可提升约+60%（DL‑Hard），在多数据集、模型、噪声反馈下均保持显著改进，计算开销极低，且在IVF/HNSW上可原地更新而不影响索引性能。

**⚠️ 局限性**

局限性包括：依赖于反馈信号的质量，若反馈误差大或不一致可能导致无效或错误更新；仅在前k条检索结果内做局部修正，无法彻底改写全局排名；在极大规模更新时仍可能对嵌入空间产生累积扰动，需进一步评估跨域迁移和长期维护问题。

---

## 420. ProCA: Progressive Contrastive Alignment for Robust EEG Visual Decoding

**arXiv ID:** 2609.05094 | [PDF](https://arxiv.org/pdf/2609.05094v1)

**作者:** Kanglei Zhou `[一作]` (Tsinghua University), Liyuan Wang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为ProCA的自适应神经‑语义对齐框架，用于提高脑电（EEG）视觉解码的鲁棒性和泛化能力。

**💡 创新点**

创新点在于将进化的类别级监督（Progressive Class‑level Alignment，PCA）与结构一致的插值（Structure‑Consistent Interpolation，SCI）结合，形成自适应的对比学习过程，使语义空间能跟随EEG表征的演化动态。

**🔧 技术方法**

核心技术包括：基于混淆矩阵的动态类别监督更新；通道时序重要性权重下的结构一致插值；多层次对比损失与结构正则化；以及对多种EEG视觉解码骨干的无缝集成。

**📊 数据集**

实验使用了两个大规模脑电视觉解码数据集：THINGS‑EEG2（200/150/100类检索任务）和Alljoined‑1.6M（20名受试者）。

**📈 对比分析**

与ATMS、NICE、RealMind、NeuroBridge、COBRA等传统与先进的对比学习与EEG对齐方法进行对比。ProCA在所有设置下平均提升Top‑1准确率约+3.8%（THINGS‑EEG2）或+15.7%（Alljoined‑1.6M），Top‑5提升约+2.5%或+7.3%；在持续学习场景中亦获得约+16.8% Top‑1提升；相较于欧式对齐等预处理方法表现更稳健。

**⚠️ 局限性**

主要限制是训练阶段需周期性更新混淆矩阵，对在线推理无影响但增加训练时开销；目前仅在视觉解码任务验证，缺乏对其他神经或生理信号的广泛适用性；对动态非平稳分布的刷新频率仍需进一步研究。

---

## 421. LLM-Guided Program Evolution for Circle Packing: Breaking 10 Packomania Records for $28

**arXiv ID:** 2609.05093 | [PDF](https://arxiv.org/pdf/2609.05093v1)

**作者:** Wes Sander `[一作]` `[通讯]` (Practical Systems), Wes Sander (Practical Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个极简的 LLM 引导程序进化框架 Discovery Loop，利用单个 LLM 对圆形打包算法进行迭代改进并取得 10 项 Packomania 圆打包记录突破

**💡 创新点**

核心创新在于：只需一次 LLM 调用产生完整求解器替代方案、实时榜单与历史反馈驱动搜索、以及零容差独立验证与自适应平台检测，从而实现极低成本（<30 美元）即可取得最优结果

**🔧 技术方法**

技术实现包括：Claude Fable 5.1 LLM、Python 代码生成与自动化验证、并行评估、奖惩榜单、以及基于窗口阈值的 plateau‑detection 机制

**📊 数据集**

使用 Packomania 圆打包 csqv 数据集（N∈{101,…,114} 及若干小规模目标），并从数据库实时获取基准记录

**📈 对比分析**

与 AlphaEvolve、FunSearch 等大型实验室级系统对比，Discovery Loop 以 27.72 美元成本在 8 小时内提升总体圆半径总和 5.00 %（单目标最高 5.39 %），且保持所有改进可独立验证且已被 Packomania 接受

**⚠️ 局限性**

局限性包括：仅聚焦单一问题域、缺乏种群多样性、对 LLM 代码生成质量高度依赖、以及在固定目标集合上快速出现平台化、收益递减的瓶颈

---

## 422. Operational Roles of QRNG-Derived Quantum Entropy in Bitcoin Proof-of-Work Architectures

**arXiv ID:** 2609.05092 | [PDF](https://arxiv.org/pdf/2609.05092v1)

**作者:** Ricardo Fernandes da Silva `[一作]` (Federal Technological University of Paraná), Paulo Vitor Batista Santos `[通讯]` (Foton Institute of Quantum Sciences and Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过模拟和统计分析，评估量子随机数生成器（QRNG）在比特币工作量证明（PoW）体系中的操作价值，并提出可从控制平面日志中直接测量的两个指标（熵效率因子 η 和重启多样性指数 ρ）。

**💡 创新点**

创新点在于：①将 QRNG 与传统确定性调度和强经典伪随机生成器在相同 PoW 逻辑下公平对比；②提出并量化两项可操作指标 η 和 ρ，用以评估熵质量对矿工覆盖率和重启稳健性的影响；③构建了一个可复现的基准框架，为后续硬件级验证提供依据。

**🔧 技术方法**

技术手段包括：蒙特卡洛仿真、调度器级别建模、统计置信区间计算、以及对工作单元唯一标识符的日志解析。

**📊 数据集**

未使用真实硬件数据集，全部采用参数化的模拟工作单元日志和预设的工作负载配置；QRNG 被视为理想高熵根源而非实际设备流。

**📈 对比分析**

通过比较四种基线（确定性、强经典、理想 QRNG、关联重启控制）对 η、ρ 和 PoW 成功概率的影响，发现：在无缺陷条件下，前三种基线在 PoW 成功率上几乎无差异；关联重启场景下 η 与 ρ 降低，但 PoW 成功率仍符合理论曲线；说明 QRNG 主要提升系统鲁棒性而非吞吐量。

**⚠️ 局限性**

局限性：缺乏基于真实 QRNG 设备的硬件级验证；理想化的 QRNG 模型可能无法捕捉实际设备的熵率、延迟及健康测试问题；研究范围仅限于诚实矿工、无自私挖矿等场景，未考虑更广泛的攻击模型。

---

## 423. Training-Free Logical and Structural Anomaly Detection via Calibrated Fusion

**arXiv ID:** 2609.05091 | [PDF](https://arxiv.org/pdf/2609.05091v1)

**作者:** Changyi Li `[一作]` (Aalto University), Yu Xiao `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种训练-free的自解释异常检测框架，能够同时检测逻辑和结构异常；

**💡 创新点**

核心创新在于“正常集校准”，将不同冻结模型产生的异常分数统一到同一尺度，并通过p-范数soft-OR融合，同时引入无训练的开源词汇计数模块；

**🔧 技术方法**

使用冻结的DINOv2、Wide-ResNet、SAM3等模型提取视觉特征和计数信息，并通过统计量进行校准和融合；

**📊 数据集**

主要使用MVTec-LOCO（含逻辑与结构异常）和MVTec-AD（仅结构异常）两个公开数据集；

**📈 对比分析**

与其他训练-free方法对比，在MVTec-LOCO上平均图像AUROC达92.5，逻辑类89.0、结构类95.9，位列训练-free方法之首；在MVTec-AD上结构配置的AUROC为99.1，竞争力强；

**⚠️ 局限性**

局限性包括对计数分支的依赖，无法很好处理小且相似的部件（如screw_bag），对概念名称的要求，以及融合策略仍未完全消除oracle差距，提示未来可进一步优化融合方法。

---

## 424. Constructing and Evaluating Clinical Reasoning Trajectories for Medical Agent

**arXiv ID:** 2609.05090 | [PDF](https://arxiv.org/pdf/2609.05090v1)

**作者:** Yunqi Zhu `[一作]` (Guangzhou University), Xuebing Yang `[通讯]` (Institute of Automation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出MedTraj框架，对医学人工智能代理的推理轨迹进行构造、评估与优化。

**💡 创新点**

创新点包括：①多维度轨迹质量评估方案与合成轨迹价值指标；②通过受控错误注入揭示错误类型与质量下降的因果关系；③步骤级别过滤与归类，识别关键驱动与破坏性推理步骤；④在推理阶段引入质量加权上下文学习，提升推理连贯性、证据支持与低幻觉率。

**🔧 技术方法**

主要技术：大语言模型（Qwen3-32B/7B、Qwen3-8B）用于生成与评估轨迹；结构化解析与错误注入；多维度评分与轨迹价值合成；步骤级边际贡献分析；质量加权上下文学习与自一致性/Best-of-N等采样策略。

**📊 数据集**

使用三大医学QA数据集：CareQA（开放式诊断问答）、PubMedQA（基于文献摘要的问答）和CECMed（老年患者严重程度评估）。

**📈 对比分析**

通过在三数据集上与多种对比方法（无上下文、问答上下文、轨迹上下文、质量加权上下文、SFT、自一致性、Best-of-N）进行实验，结果显示轨迹上下文和质量加权上下文显著提升推理连贯性+0.029~+0.041，CECMed正确率提升至73.8%，幻觉率下降87%，但在开放式QA中正确率提升有限。

**⚠️ 局限性**

局限性包括：生成与评估均依赖LLM，可能继承其偏见与幻觉；错误注入模式与真实临床错误可能不完全对应；未使用强化学习或人类专家评估；仅在三数据集和特定模型上验证，需进一步扩展到更多模型与临床领域。

---

## 425. Measuring AI Accountability Through Argumentation Analysis: Can Model Reasoning Withstand Scrutiny?

**arXiv ID:** 2609.05088 | [PDF](https://arxiv.org/pdf/2609.05088v1)

**作者:** Daan R. Henselmans `[一作]` (Aithos Research Foundation), Arno Libert `[通讯]` (Aithos Research Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过四阶段对话式检验，评估大型语言模型在高模糊伦理困境中的论证可辩护程度，关注模型自身推理与公开辩护的结构一致性；

**💡 创新点**

创新点在于将Walton论证方案与Govier论证质量指标相结合，提出独立于目标答案的结构性可辩护标准，并区分推理轨迹与后续辩护之间的差距；

**🔧 技术方法**

技术手段包括：①使用Walton的十三种伦理论证方案自动分类；②生成对应的批判性提问并让模型回答；③用Govier的接受性、关联性、充分性三维度以及全局维度对回答进行三分制评分；

**📊 数据集**

数据集为MoralChoice基准中200条高模糊性伦理困境（共6,778个评估单元），覆盖多模型（9种），并结合人工评审判定；

**📈 对比分析**

比较方法采用多模型、两路径（推理轨迹vs. 公开辩护）以及两位评审，报告失效率、均值分数、方案一致性和模糊度分析，结果显示所有模型在结构可辩护上普遍高于阈值，但在本土化基础与全局充分性上存在差距，推理轨迹普遍优于公开辩护；

**⚠️ 局限性**

局限包括：仅使用单一高模糊性数据集，未涵盖多文化、多代理或具体情境；推理轨迹与公开辩护的分离可能受模型内部隐藏机制影响；评审偏差与单一评审（Claude）对结果的影响需要进一步跨族群验证；

---

## 426. Adaptive Multi-Granularity Temporal Modeling for Weakly Supervised Video Anomaly Detection

**arXiv ID:** 2609.05066 | [PDF](https://arxiv.org/pdf/2609.05066v1)

**作者:** Changyi Li `[一作]` (Aalto University), Yu Xiao `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种适用于弱监督视频异常检测的多粒度时序建模框架，包含时间细化模块（TRM）、事件分割模块（ESM）和基于相似度的自适应融合策略。

**💡 创新点**

创新点在于：①使用可学习的类别标记和动态位置编码实现全局上下文聚合与长程依赖建模；②通过自适应事件分割捕捉不同时长的异常事件；③用相似度加权代替固定top‑k聚合，提升视频级预测的鲁棒性。

**🔧 技术方法**

技术手段包括：多实例学习（MIL）、自注意力与残差连接、动态位置编码、事件分割（基于特征差异、MAD阈值）、相似度加权融合、I3D骨干网络、因果卷积分类器、中心损失等。

**📊 数据集**

在UCF‑Crime和XD‑Violence两个公开大规模监控数据集上进行实验。

**📈 对比分析**

与现有SOTA方法对比，在UCF‑Crime上获得87.24% AUC、0.54% FAR，在XD‑Violence上获得83.89% AP、0.46% FAR，均优于之前最高记录。

**⚠️ 局限性**

局限性包括：仅使用视频级标签，无法获得更细粒度的时序监督；对多模态输入（如音频、光流）的适应性尚未验证；模型对超参数和阈值敏感，需在不同数据集上调优。

---

## 427. Beyond Maintenance Manual Multimodal RAG: Suggesting What Tool

**arXiv ID:** 2609.05116 | [PDF](https://arxiv.org/pdf/2609.05116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 428. Online Matching in Convex Bipartite Graphs

**arXiv ID:** 2609.05057 | [PDF](https://arxiv.org/pdf/2609.05057v1)

**作者:** Yilong Feng `[一作]`, Xiaowei Wu `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在线二分图匹配问题，重点考察在凸二分图（即在线侧邻域为连续区间）下，是否能利用结构提升匹配的竞争比；在请求区间长度相同的统一长度模型中，提出了仅用一随机位的Flip算法；并给出了在半自适应对手下的竞争比分析与下界；同时证明在不满足统一长度时凸性无法超越经典1-1/e的竞争比；

**💡 创新点**

创新点包括①证明凸性本身不改进1-1/e的最优界；②提出Flip算法，利用随机一次决策即可达到2d-1/3d-2≈2/3的竞争比；③构造硬实例验证Flip的最优性；④给出全随机算法在统一长度模型下不能超过3/4的上界；以及⑤通过偏移函数、坏索引与代理打包的全新组合证明技术。

**🔧 技术方法**

主要技术包括：偏移函数（offset function）和坏索引（bad indices）定义；最大链（maximal chains）与链长度限制；代理（proxy）打包与两轨打包策略；证明不等式的组合与充电/计数论证；以及构造递归实例与递推函数w_k(x)用于上界证明。

**📊 数据集**

无真实数据集，所有结果均为理论证明与合成硬实例。

**📈 对比分析**

与半自适应对手比较时，Flip取得2d-1/3d-2的竞争比，随d增大趋近于2/3；这比任何单纯的早/晚匹配规则（仅1/2）明显好。对比上界，任何算法在统一长度模型下竞争比上限为3/4，说明Flip已逼近最佳可实现区间（[2/3,3/4]）。在非统一长度的凸图中，竞争比无法超过1-1/e，跟经典Karp-Vazirani-Vazirani排名算法的界限相同。

**⚠️ 局限性**

限制包括：①算法仅适用于凸二分图且在线请求邻域长度相同；②对手模型为半自适应（比完全自适应强，但比无记忆对手弱）；③Flip虽随机，但竞争比仍受限于2/3，下界表明无法进一步提升至3/4；④在更一般的邻域结构（不满足统一长度）中无法获得更高竞争比。

---

## 429. Scales, Reflections, and Conversations: A Multi-Modal Approach to Emotion Annotation

**arXiv ID:** 2609.05046 | [PDF](https://arxiv.org/pdf/2609.05046v1)

**作者:** Pragya Singh `[一作]` (Indraprastha Institute of Information Technology Delhi), Pushpendra Singh `[通讯]` (Indraprastha Institute of Information Technology Delhi)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了一款支持用户自选时间和多模态（快速、详细、LLM交互）情绪日志的可行性系统，并在33名参与者的一周现场实验中进行评估。

**💡 创新点**

创新点在于将情绪记录方式拆分为可配置提示和多模态输入，并强调情绪表达的多层面、情境丰富性与用户自主管理，提高数据的生态有效性。

**🔧 技术方法**

技术实现采用移动端闹钟提醒、四象限情绪评估+多选标签、情境列表、自信度评分以及本地部署的LLM对话引擎，实现多模态情绪采集。

**📊 数据集**

使用了33位参与者在一周内产生的505条日志数据，包含文本、音频、图片、情绪标签、情境描述与自信度评分等多维信息。

**📈 对比分析**

通过混合效应模型和主题分析比较调度方式与模态对情绪表达的影响，发现随时记录率显著高于预定提醒，情绪多样性提升，但模态差异对情绪属性影响不显著，系统整体可用性评价积极。

**⚠️ 局限性**

局限性包括样本规模小、样本多为技术熟练的学生/专业人士，缺乏多元文化与重度心理疾病人群，未给与激励，且LLM交互使用率低，限制了结论的普适性与长期使用的可行性。

---

## 430. A Schema Bounded Language Model for Refining Robot Policies Without Destabilizing Local Learning

**arXiv ID:** 2609.05133 | [PDF](https://arxiv.org/pdf/2609.05133v1)

**作者:** Chongwen Dong `[一作]` (Northern Arizona University), Carlo R. daCunha `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个去中心化、多时尺度的多机器人导航系统，利用每轮的LLM生成和细化策略，并通过UCB选取改进模式，结合本地Double DQN进行每个控制时刻的动作选择，支持跨机器人共享回合摘要。

**💡 创新点**

创新点包括：① 仅在回合边界调用LLM，避免每步调用带来的延迟和不稳定；② 机器人本地持有策略与UCB与DQN状态，形成完全去中心化的决策链；③ 将LLM、UCB和Double DQN三者协同结合的多时尺度架构，在同一任务上实现了更快、更稳定的导航。

**🔧 技术方法**

使用技术包括：大型语言模型（Llama-3.3-70B-Instruct-quantized、Phi-4、Meta-Llama-3.1-70B-Instruct-quantized）、UCB1 bandit用于改进模式选择、Double DQN用于tick级动作决策、NetLogo-Python仿真框架和共享文本板实现跨机器人信息交流。

**📊 数据集**

实验数据来自自定义的NetLogo二维网格仿真环境：三台机器人在固定起点和目标下完成30轮任务，每轮最多1000个tick，记录完成时间、成功率等指标。

**📈 对比分析**

通过对四个配置（C1–C4）进行比较，采用成功率、中位数、均值、P90、LBSE和Worst Decile等指标评估性能；结果显示C4在中位数（42 tick）、P90（73.2 tick）和LBSE（0.431）上均优于其他配置，成功率均达100%，但C3在Worst Decile上表现更好。

**⚠️ 局限性**

局限性在于：实验仅在固定NetLogo仿真下进行，缺乏独立随机种子重复和统计显著性检验；配置级别的结果难以归因到单一模块；未评估LLM的延迟、token消耗和在真实机器人上的可扩展性。

---

## 431. VoxelFix: Post-Hoc Semantic Correction of Completed 3D Voxel Maps

**arXiv ID:** 2609.05114 | [PDF](https://arxiv.org/pdf/2609.05114v1)

**作者:** Sunesh Praveen Raja Sundarasami `[一作]` (Fraunhofer IVI), Sebastian Houben `[通讯]` (Hochschule Bonn-Rhein-Sieg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种仅基于已完成语义体素地图的后置语义纠正方法，只利用地图本身的几何与语义信息对语义标签进行重新标注；

**💡 创新点**

创新点在于构造双分支图注意力网络，融合几何与语义邻域信息，并引入基于混淆矩阵的噪声训练课程，提升了对连贯误差的识别与纠正；

**🔧 技术方法**

技术手段包括稀疏体素U-Net提取几何嵌入、六维几何描述子、共现先验与几何先验、双分支GATv2（几何与语义）以及门控融合与辅助错误检测；

**📊 数据集**

使用了OccuFly数据集（包含真实深度与位姿）、STPLS3D合成数据预训练，以及独立重建的OOD场景作为评估数据；

**📈 对比分析**

与KNN、CRF、几何启发式、MinkUNet等基线对比，在六个不同上游分割模型上平均提升mIoU 4.23–5.00点，单模型上从25.80点提升至30.66点；OOV场景亦获得显著改进；

**⚠️ 局限性**

局限性包括：仅能纠正语义错误，无法修复几何缺失或重建误差；对几何质量高度依赖；在语义不确定或模糊区域的纠正效果有限；需要在相似域内进行训练，跨域泛化仍受限。

---

## 432. Influence Score and Transformers interpretability: Measure of the Effective Impact of Attention Heads at inference time

**arXiv ID:** 2609.05074 | [PDF](https://arxiv.org/pdf/2609.05074v1)

**作者:** Lisa Bouger `[一作]` (Thales CDI), Philippe Loubet Moundi `[通讯]` (Thales CDI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合注意力头对logit方向性影响与其在残差流中的相对贡献的影响得分，用于定量评估Transformer注意力头在prompt injection检测中的决策作用。

**💡 创新点**

创新点在于将方向性投影与ALTI结构贡献相乘得到多尺度影响得分，可在头、层、网络层级聚合，并在同一框架下同时捕捉决策的方向与贡献，揭示正确与错误预测的内部机制差异。

**🔧 技术方法**

使用Transformer残差流分析、注意力头分解、ALTI相对重要性度量、logit投影、层级加权、K‑means聚类及消融实验等技术。

**📊 数据集**

基于DeBERTa‑v3‑base的prompt injection检测模型，训练数据由VMware/open‑instruct、HuggingFaceH4/grok‑conversation‑harmless、OpenSafetyLab/Salad‑Data等公开数据集构成，共约3万条二分类样本。

**📈 对比分析**

通过将不同头/层的影响得分聚合并与原始模型性能对比，使用F1、TP/TN变更等指标评估消融效果；实验表明采用综合得分消融能显著降低F1（约0.003–0.004），优于仅使用投影或贡献的消融，说明得分能捕获真正的决策关键。

**⚠️ 局限性**

主要局限在于直接将中间激活投影到最终logit方向，未考虑层间分布漂移，可能导致估计偏差；此外模型可能过参数化，存在冗余，导致消融影响相对温和。

---

## 433. A Unified Physics-Aware Quantum Machine Learning Framework across Power GaN HEMTs and Logic Nanowire FETs: Predicting Unseen Process Splits and Held-Out Geometry Combinations with Lower Error and Tighter Split-to-Split Variability

**arXiv ID:** 2609.05251 | [PDF](https://arxiv.org/pdf/2609.05251v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 434. A Structured Debate-Mixture-of-Agents Framework for Complex Clinical Diagnostic Decision Support

**arXiv ID:** 2609.05069 | [PDF](https://arxiv.org/pdf/2609.05069v1)

**作者:** Chang Xia `[一作]` (Sichuan University), Kang Li `[通讯]` (Sichuan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了DMoA多代理框架，在大型语言模型基础上通过生成、质疑、修订、聚合四个阶段模拟临床诊断的迭代推理流程；

**💡 创新点**

创新点在于结构化的多角色协作与辩论式回馈机制，明确将诊断拆分为生成、质疑、修订与聚合四个阶段，显著提升诊断准确性与安全性；

**🔧 技术方法**

使用了Mixture-of-Agents技术构建多代理系统，结合GPT-4o、GPT-3.5、GPT-4o mini等大型语言模型，辅以提示工程、token预算管理与轻量模型混合；

**📊 数据集**

评估数据集包括297例罕见病和1719例挑战性临床案例（ClinDiag），以及1000个从MedQA筛选的临床知识问答；

**📈 对比分析**

通过与单模型基线、不同消融配置、结构规模、轻量模型和奖励模型等多维度对照，DMoA在最可能诊断准确率提升约10个百分点、三候选准确率提升约4–6个百分点，安全率提升约10–12个百分点；

**⚠️ 局限性**

局限性包括样本量仍有限、仅基于已发布案例、对底层模型性能高度依赖、轻量版仍低于单模型绝对水平、缺乏临床验证以及可解释性不足。

---

## 435. Beyond Co-purchase Relation: Evolution of Complementary Recommendations at Allegro

**arXiv ID:** 2609.05063 | [PDF](https://arxiv.org/pdf/2609.05063v1)

**作者:** Aleksandra Osowska-Kurczab `[一作]` (Allegro.com), Michał Bień `[通讯]` (NVIDIA)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出了名为 AlleCompanion 的生产级补充商品推荐框架，旨在把用户购买意向从相似性迁移到真正的互补性。

**💡 创新点**

核心创新点包括：① Category Adapter 将查询嵌入空间引导至指定补充品类；② ComCat 多源映射将专家规则、LLM 推理和行为筛选合并为可更新的补充品类约束；③ 在 Two‑Tower 架构中加入类别重构损失，提升类别一致性与商品级兼容性。

**🔧 技术方法**

技术手段包括：Two‑Tower 共享编码器、类别适配器与重构损失、基于采样软最大化的检索损失、负采样与温度缩放、行为过滤与阈值 heuristics、专家规则、LLM 辅助标注、Faiss ANN 近似检索、在线 A/B 实验。

**📊 数据集**

使用 Allegro 公开的 90 天购买日志（约 3.6M 交易，779k 商品，7.5k 类别）为训练集，并通过多种过滤策略得到 829k 训练对；还构造了专家规则与同商家过滤的合成数据集。

**📈 对比分析**

离线评估采用 Recall@20、MRR@20、类别一致性、商家一致性和属性一致性；相较基线 Two‑Tower，AlleCompanion 在 Recall@20 与 MRR@20 上提升约 3–4%；在线 A/B 试验显示在产品页有 8–9% GMV 上升，在结账阶段有 15–21% GMV 提升，赞助位 CTR 约提升 50%。

**⚠️ 局限性**

局限性主要体现在：① 对补充品类映射的依赖导致对极冷启动类别的覆盖不足；② 未将用户个性化直接融入候选生成，缺少下游排序层；③ 由于多商品互补动态难以离线评估，仍需依赖在线 A/B 测试。

---

## 436. Efficient Multi-Timescale Event Representations for Feed-Forward Object Detection

**arXiv ID:** 2609.05049 | [PDF](https://arxiv.org/pdf/2609.05049v1)

**作者:** Fredrik Lundell `[一作]` (Linkoping University), Astrid Lundmark `[通讯]` (Saab AB)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于对数B样条的连续多尺度事件表示，并用该表示提升了前馈式事件摄像机目标检测。

**💡 创新点**

创新点在于引入置信度归一化的多尺度B样条编码、空间局部置信支持以及可递归的指数-多项式逼近，实现了高效事件驱动的时间建模。

**🔧 技术方法**

使用了对数B样条时间编码、置信度归一化、局部置信聚合、指数多项式逼近、Feed‑Forward EventCenterNet和Kalman滤波。

**📊 数据集**

实验采用PEDRo和Gen1两个事件摄像机目标检测数据集。

**📈 对比分析**

与CSTR、指数衰减以及ReYOLO等基线比较，PEDRo上AP_50:95从0.566提升至0.630，Gen1上提升至0.406，前馈模型在PEDRo上已超越非递归YOLO，在Gen1上仍略低于递归模型。

**⚠️ 局限性**

局限在于需要针对不同应用调节时间函数和窗口，对极低光/稀疏事件的鲁棒性有限，且对长时序聚合的依赖仍未完全消除。

---

## 437. Do LLMs Exhibit Coherent Knowledge Structures in Mathematical Reasoning? A Perspective from Knowledge Space Theory

**arXiv ID:** 2609.05245 | [PDF](https://arxiv.org/pdf/2609.05245v1)

**作者:** Peng Cui `[一作]` (ETH Zürich), Mrinmaya Sachan `[通讯]` (ETH Zürich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于知识空间理论（KST）的评估框架，用来检验大型语言模型（LLM）在数学推理中的知识结构是否与人类相似；

**💡 创新点**

创新点在于将KST的前提依赖关系作为判定模型知识一致性的客观标准，并定义了三项规范性行为（PSR、SC、KOC）以及对应的量化指标；

**🔧 技术方法**

采用KST理论、概念注解（通过LLM从预定义概念列表中挑选）、PSR（前置满足率）、SG（在前置知识上下文中的增益）、KOC（知识重叠系数）等技术；

**📊 数据集**

使用公开的数学知识追踪数据集XES3G5M（4,118道题，18,066名学生）并依据纽约州数学标准构建概念及其前置依赖图；

**📈 对比分析**

与18,000多名真实学生以及八款公开/闭源LLM（Mistral、Llama、Qwen、Claude、GPT‑4）进行对比。实验表明，LLM的PSR远低于人类（人类≈0.94 vs LLM最高≈0.95），前置上下文对性能的提升不如相似/同技能示例，且更强模型的知识并不完全包含较弱模型的知识，整体结构一致性不足；

**⚠️ 局限性**

局限性包括：需要预先存在的专家依赖图（仅适用于数学等结构化领域），仅在数学任务上验证，且以问题级答对率作为知识状态的代理，未直接评估概念掌握水平。

---

## 438. Measuring the Novelty of Biomedical Papers Using the Latent Distances between Knowledge Units

**arXiv ID:** 2609.05175 | [PDF](https://arxiv.org/pdf/2609.05175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 439. Uncensored Open-weight Models: Redistribution as the Persistence Layer

**arXiv ID:** 2609.05241 | [PDF](https://arxiv.org/pdf/2609.05241v1)

**作者:** 10a Labs `[一作]` (10a Labs), Zachary Yahn `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 HuggingFace 和 GitHub 上的仓库进行关键词爬取和 LLM 分类，系统量化了不受约束的开源模型（3,471 个原始模型、8,164 个重分发模型）以及 1,643 个集成这些模型的应用，描绘了从生产到重分发再到下游部署的完整生态链。

**💡 创新点**

创新点在于首次将生产层和重分发层严格区分，揭示重分发是模型持久化的核心；发现 Heretic CLI 在 2025 年后将生产速率提升 4 倍；并突显中文基础模型（尤其是 Qwen 家族）已占比 38% 的生产量，说明模型来源多样化。

**🔧 技术方法**

使用了关键词搜索、基于 GPT‑5 的 LLM 分类器、Heretic CLI 自动化重分发管道，并通过定量统计、可视化和比例分析来评估生态结构。

**📊 数据集**

主要数据集为 HuggingFace 的 12,360 个安全护栏移除相关仓库（3,471 原始模型、8,164 重分发、547 合并、178 恶意数据集）以及 44,705 个 GitHub 候选仓库，最终筛选出 1,643 个使用不受约束模型的应用。

**📈 对比分析**

通过对模型家族、重分发倍数、下游应用类型和注册表使用情况的统计比较，发现 Heretic 使得平均每个原始模型被重分发 2.4 次，且生产率从每月 89 模型激增至 338 模型；但本文未对模型性能或安全性进行基准测试，而是侧重于数量和分布的评估。

**⚠️ 局限性**

局限性包括：仅覆盖公开平台（未包含 ModelScope、Gitee 等）、分类依赖单次 LLM 预测缺乏人工校验、GitHub 搜索上限导致高频 CJK 关键词被截断、部分仓库无法解析模型来源、未测量精准率与召回率，以及对新兴的私有或地下渠道缺乏监测。

---

## 440. Dimension-Adaptive Batched Lipschitz Narrowing Without Knowing the Zooming Dimension

**arXiv ID:** 2609.05214 | [PDF](https://arxiv.org/pdf/2609.05214v1)

**作者:** Yasong Feng `[一作]` `[通讯]`, Yasong Feng

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的Count-Adaptive BLiN算法，消除了对缩放维度d_z的依赖，改进了边长选择的方式。

**💡 创新点**

创新点在于算法不再依赖于缩放常数C_z和缩放维度d_z，且在未知d_z的情况下仍能达到最优的批处理复杂度。

**🔧 技术方法**

使用了Count-Adaptive Batched Lipschitz Narrowing (CA-BLiN)算法，结合了自适应网格和数据依赖的边长选择。

**📊 数据集**

论文中没有具体提到使用的数据集，但算法的理论分析基于一般的臂空间模型。

**📈 对比分析**

与原始D-BLiN算法进行比较，新的算法在批处理复杂度上达到了Θ_d(loglog T)，并且在不依赖于d_z的情况下仍能保持最优的后悔界限。

**⚠️ 局限性**

算法的局限性在于其理论分析依赖于特定的模型假设，实际应用中可能受到模型不匹配的影响。

---

## 441. An Empirical Study on Learning Paths and Gender Dynamics in Scrum Master Roles

**arXiv ID:** 2609.05186 | [PDF](https://arxiv.org/pdf/2609.05186v1)

**作者:** Manuela Petrescu `[一作]` (Babes Bolyai University), Paul Razvan Petrescu `[通讯]` (Babes Bolyai University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本研究通过两阶段定性访谈（共44位受访者，涵盖14家公司）和一次焦点小组，探讨Scrum Master角色的学习路径、团队规模与SM需求的关联、女性SM的职业发展及企业如何监测软技能。

**💡 创新点**

创新点在于：①首次实证确认团队规模（>6人）是SM需求的阈值；②系统梳理女性SM的多元学习方式与组织支持；③揭示当前企业普遍缺乏标准化软技能评估指标。

**🔧 技术方法**

采用访谈+焦点小组的质性研究方法，并使用主题分析（Thematic Analysis）对访谈文本进行编码与归纳。

**📊 数据集**

数据集包括：第一阶段32份访谈记录（14名女性、18名男性），第二阶段12份女性SM访谈，涵盖不同背景（技术、商业、工程等）与公司规模（小型6家，大型8家）。

**📈 对比分析**

本研究未进行量化性能比较，仅通过频率统计呈现结果；相对先前研究提供了更细粒度的学习路径映射与性别分布信息。

**⚠️ 局限性**

局限性包括：①样本主要来自罗马尼亚IT公司，外推性受限；②受访者自述信息可能存在偏差；③缺乏长期跟踪数据，无法验证学习路径效果。

---

## 442. Students' Perception of Big Data Engineering in Higher Education Curricula: Expectations, Interest and Ethical Implications

**arXiv ID:** 2609.05160 | [PDF](https://arxiv.org/pdf/2609.05160v1)

**作者:** Ioana-Georgiana Ciuciu `[一作]` (Babes-Bolyai University), Petrescu Manuela-Andreea `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对42名硕士生进行匿名在线调查，并对其开放式回答进行主题分析，探究他们对大数据课程的期望、兴趣及伦理认知。

**💡 创新点**

首次系统地从学生视角量化大数据课程的兴趣与伦理关注，并将计算机科学与生物信息学学生的差异纳入对比，填补了高等教育大数据教学研究的空白。

**🔧 技术方法**

采用问卷调查结合定性主题分析（Thematic Analysis）对数据进行编码与归类。

**📊 数据集**

来自Babeș‑Bolyai大学计算机科学与生物信息学硕士二年级学生的42份匿名问卷数据。

**📈 对比分析**

通过频率统计与主题归纳比较不同专业学生的回答，发现大多数学生对实践性学习和伦理风险具有高度关注，结果显示对比方法能清晰呈现各群体关注点。

**⚠️ 局限性**

样本量有限、仅单一高校、未进行纵向跟踪，且调查依赖自我报告，限制了结论的普适性与因果性。

---

## 443. Substrate-Aware AI Agents: Execution Context as a First-Class Input

**arXiv ID:** 2609.05232 | [PDF](https://arxiv.org/pdf/2609.05232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 444. T(r)opical Islands: Visualizing & Understanding Socio-Technical Artifacts

**arXiv ID:** 2609.05254 | [PDF](https://arxiv.org/pdf/2609.05254v1)

**作者:** Adam Štěpánek `[一作]` (Masaryk University), Michele Lanza `[通讯]` (Software Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于岛屿隐喻的三维可视化方法，将 GitHub 社会技术工件聚类为主题岛屿并以树形图形展示其演化。

**💡 创新点**

创新点在于将主题建模与三维岛屿可视化结合，动态呈现主题活跃度，并通过 LLM 自动生成主题标签。

**🔧 技术方法**

使用了 Qwen3-Embedding-8B 文本嵌入、UMAP 降维、HDBSCAN 聚类、BERTopic 主题抽取、OpenAI GPT-OSS-B120 生成标签，以及 Godot 引擎渲染。

**📊 数据集**

数据集来源于 GitHub，包含 Lume、JetUML、Git for Windows、DotVVM 等开源项目共计超过 5 万条 STA。

**📈 对比分析**

通过案例研究和 34 名参与者的用户研究验证，结果显示可读性、洞察力和实用性均显著优于传统 GitHub 视图；但对大规模项目的可读性略有下降。

**⚠️ 局限性**

主要限制包括聚类依赖于高成本嵌入模型、缺乏对讨论子帖和重开闭工件的完整处理、以及渲染性能限制导致可处理的 STA 数量有限。

---

## 445. Governing Bring Your Own AI: A Parameterized Maturity Model

**arXiv ID:** 2609.05236 | [PDF](https://arxiv.org/pdf/2609.05236v1)

**作者:** Dare Bello `[一作]` (Dakota State University), John Hastings `[通讯]` (Dakota State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并评估一种面向“自带 AI” (BYOAI) 的治理框架和参数化成熟度模型；

**💡 创新点**

首次将 BYOAI 风险进行系统性分类，并通过确定性参数化模型（FGI、CGS、TCCS）量化治理成熟度对残余风险的影响；

**🔧 技术方法**

采用系统性文献综述、控制族技术栈（DSPM、CASB、DLP、SSPM、IAM、CIEM、SWG）以及基于参数的风险链模型；

**📊 数据集**

基于 30 篇文献（24 篇研究论文 + 6 框架/标准文档）的精心挑选语料库进行编码与分析；

**📈 对比分析**

模型通过敏感性分析展示了从 Level‑1 到 Level‑5 的 TCCS 上升、FGI 降低、CGS 上升等趋势，但未进行实证实验，主要以理论推导与参数设定为依据；

**⚠️ 局限性**

局限性包括：缺乏经验验证、单一研究者编码导致可靠性未知、仅量化技术层、未覆盖治理与人类层的数值化、数据集规模有限且未完全覆盖最新文献。

---

## 446. Measured Sliders: Learning Continuous Controls from Differentiable Image Measurements

**arXiv ID:** 2609.05234 | [PDF](https://arxiv.org/pdf/2609.05234v1)

**作者:** Yijia Chen `[一作]` (University of Sydney), Xuanhua Yin `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Measured Sliders框架，通过闭式可微分图像测量定义连续控制器，并在单一检查点中存储多条LoRA分支；

**💡 创新点**

创新点在于将控制坐标直接映射到解码后图像的可微测量空间，实现统一的可观测性分析、目标监督和解码校准；

**🔧 技术方法**

使用闭式图像测量、可微预览（affine映射）、LoRA低秩微调、观测性比率判别、解码校准和多目标监督损失；

**📊 数据集**

在SDXL和FLUX.1-dev两大扩散模型上进行训练与评估，使用553个GenEval提示和12个对比内容；

**📈 对比分析**

与Concept Sliders、AttributeControl、Text Slider、FreeSliders等基线对比，Measured Sliders在光照方向上取得ρ=0.995、98.9%单调性，五属性检查点平均选择性提高至2.59，配合解码校准将终点离散度从63.2×降至2.2×；

**⚠️ 局限性**

局限在于只支持已定义的可微测量属性，需预训练观测性检验；解码校准需额外内容集；对更大规模属性集合和跨模型泛化仍有待扩展。

---

## 447. First Things First: Teaching LLM-Based Agents to Prioritize Must-Haves before Nice-to-Haves

**arXiv ID:** 2609.05224 | [PDF](https://arxiv.org/pdf/2609.05224v1)

**作者:** Tianjie Ju `[一作]` (Shanghai Jiao Tong University), Cheng Yang `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 FTF‑bench 数据集和 FTF‑rl 强化学习框架，用于评估和提升多模态大模型在真实服务场景中对必须需求和可选需求的分层推理与决策。

**💡 创新点**

创新点在于提出了三种需求优先级场景（单答案、多答案、不可答）并设计了多目标奖励机制，显式强化模型对必须需求的识别、可选需求的排序以及回答的格式与正确性。

**🔧 技术方法**

采用了多模态链式思维 (MCoT)、GRPO 强化学习、KL 约束、以及自定义的四维奖励（格式、答案、需求分类、推理过程）等技术。

**📊 数据集**

使用了 3,649 张来自电商、预订与地图/打车界面的截图，并人工校验生成的必需/可选需求及答案；同时在 LogicVista、MathVision、InfoQA 等外部推理基准上进行迁移评估。

**📈 对比分析**

在 Direct 设定下原始模型准确率仅 38–55%，但在 Upper 设定（给定黄金需求）可达 80–88%；通过 FTF‑rl 训练后，单答案/多答案/不可答场景均提升 10–30% 以上，且在外部推理基准上表现也出现 4–12% 的提升。

**⚠️ 局限性**

局限在于数据集仅覆盖有限的服务领域，需求演化与动态约束未覆盖；强化学习仍受限于 3,649 样本规模，未来需扩展多领域、动态场景和更大规模训练以验证泛化上限。

---

## 448. The Mirror Agent Model: a Bayesian Architecture for Interpretable Agent Behavior

**arXiv ID:** 2609.05190 | [PDF](https://arxiv.org/pdf/2609.05190v1)

**作者:** Michele Persiani `[一作]` (Umeå University), Thomas Hellström `[通讯]` (Umeå University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建并实验了 Mirror Agent Model，将可解释行为与解释生成统一到一个结构化贝叶斯网络框架内，并分别在 BDI 代理（PDDL）和 RL 代理（Deep Q‑Network）上实现可解释行为与 saliency‑based 解释的生成。

**💡 创新点**

创新点在于将观测者模型设为代理意图模型的镜像，利用信息增益与 KL 散度量化行为可解释性与解释效果，并通过统一的 H(P_R,P_R^H) 距离度量将多种可解释任务（可解释计划、可解释行为、解释）统一到同一框架。

**🔧 技术方法**

技术包括：第二阶心智推理、贝叶斯网络建模、信息增益与 KL 散度计算、离散化 saliency 生成（perturbation 方法）、PDDL 规划、Deep Q‑Network、OpenAI Gym 环境、对比实验（用户研究与数值评估）。

**📊 数据集**

使用自建的 OpenAI Gym 隧道/多彩障碍环境作为 RL 实验数据集，BDI 任务则在 PDDL 设计的情境下测试；未使用公开大规模数据集。

**📈 对比分析**

通过用户研究对信息增益式计划沟通与传统升序/降序策略进行对比，结果表明信息增益策略显著提升了参与者的目标推断成功率；在 RL 任务中比较不同 α 值的 legible policy 与最优策略，展示了奖励-可解释性之间的 trade‑off，legible policy 在牺牲少量奖励的同时提升了可解释度。

**⚠️ 局限性**

局限性包括：需假设观测者拥有完整且结构相同的模型；对齐模型的成本与可扩展性待进一步评估；实验仅在小规模自建环境进行，缺乏大规模或多样化真实数据的验证；解释主要依赖 saliency 视觉化，对非视觉用户的可解释性尚未评估；框架尚未集成所有可解释行为算法，仍处于初步验证阶段。

---

## 449. Proton Irradiation Characterization of an Open-Source ML Accelerator on a Zynq UltraScale+ MPSoC

**arXiv ID:** 2609.05249 | [PDF](https://arxiv.org/pdf/2609.05249v1)

**作者:** Saad Memon `[一作]` (University of Luxembourg), Mike Papadakis `[通讯]` (University of Luxembourg)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在Zynq UltraScale+ MPSoC上对开放源代码的Tensil ML加速器进行质子辐照实验，评估其在空间环境下的可靠性和错误特征。

**💡 创新点**

首次系统级（端到端）地记录并量化了Linux管理的FPGA加速器在质子辐照下的两类故障：工作负载中断（Linux-SEFI）和无声数据腐败（持续错误输出）。并提出了基于事件起始、监视器匹配和恢复层级的实验报告规范。

**🔧 技术方法**

使用的技术包括：Zynq UltraScale+ XCZU3EG MPSoC、Tensil Tensor Compute Unit（16×16算术阵列）、PYNQ Linux 2.7、ResNet‑20模型、CIFAR‑10数据集、AIC‑144质子加速器（20–58 MeV）、多种监视器（进程状态、内核日志、内存测试、电源采样）以及事件聚类和交叉截面统计方法。

**📊 数据集**

数据集：固定的10张CIFAR‑10测试图像（10–19索引），用于在每个推理块中进行随机采样并记录预测类别。

**📈 对比分析**

通过对比不同场尺寸（2 cm vs. 4 cm）以及不同能量（20 MeV、40 MeV、58 MeV）下的事件率，计算出Linux-SEFI和输出事件的交叉截面。实验显示：在宽场下所有事件均出现，提示外部板级电路的贡献；在小场下无Linux-SEFI或输出事件。性能指标表明，系统在大部分时间内可用性>90%，但存在未被活跃监控捕获的无声错误。

**⚠️ 局限性**

主要局限包括：仅评估单一硬件平台和单一工作负载；场尺寸与实验顺序混杂，无法精确定位错误源；未记录中间张量、模型缓存校验和DMA传输；USB‑gadget链路可能掩盖服务中断；缺乏随机化和长周期无重置运行，难以测量错误持续时间。

---

## 450. Towards Federated, Green, and Resilient 6G Non-Terrestrial Networks

**arXiv ID:** 2609.05184 | [PDF](https://arxiv.org/pdf/2609.05184v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 451. Few-Shot Video Recognition via Hierarchical Metric Learning

**arXiv ID:** 2609.05242 | [PDF](https://arxiv.org/pdf/2609.05242v1)

**作者:** Jiaxin Zhang `[一作]` (University of Jinan), Sijie Niu `[通讯]` (University of Jinan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于层次度量学习的少样本动作识别框架 HML‑FSAR，旨在从跨帧全局空间信息中提取更具判别力的时空特征并通过多阶段度量约束实现高质量原型学习。

**💡 创新点**

创新点在于：①构造跨帧全局空间增强模块和异质对齐、时空融合、字典学习四个新模块；②提出五阶段层次度量学习（中心、对齐、对比、字典、原型）在整个特征传递链上逐层监督；③通过字典学习抑制少样本噪声并提升鲁棒性。

**🔧 技术方法**

采用MAE 预训练视觉编码器，结合 SPDNet 处理 Riemannian SPD 特征，使用多头注意力（MHA）、MLP、Transformer 块实现对齐与融合；损失函数包括中心损失、对齐损失、对比损失、字典重建损失与原型损失的加权组合。

**📊 数据集**

在五大少样本动作数据集上评估：HMDB51、UCF101、Kinetics、SSv2‑Full、SSv2‑Small，均使用 1‑shot / 5‑shot 任务。

**📈 对比分析**

与现有 15+ 先进方法（ProtoNet、OTAM、CLIP‑FSAR、MVP‑Shot 等）对比，HML‑FSAR 在大多数 5‑shot 任务上实现了最高准确率（如 UCF101 99.2%，SSv2‑Full 73.2%），显著提升了原型的判别性与泛化能力。

**⚠️ 局限性**

主要局限：模型结构复杂、参数量大，训练时需多阶段多任务损失，计算开销高；对超参数调优敏感；在极低样本（1‑shot）下性能提升相对有限。

---

## 452. PRICE: A Systematic Study of LLM Adaptation Choices for Bitcoin Price Forecasting

**arXiv ID:** 2609.05235 | [PDF](https://arxiv.org/pdf/2609.05235v1)

**作者:** Maryam Fakhari `[一作]` (Isfahan University of Technology), Mehran Safayani `[通讯]` (Isfahan University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出PRICE框架，对大语言模型进行参数高效微调、递归多步推理、整数化数值表示、结构化CTF提示及零温度解码，以实现比特币短期价格预测。

**💡 创新点**

创新点在于系统评估并融合五个适配组件（LoRA微调、递归推理、整数化表示、CTF提示、确定性解码），通过消融实验明确每个选择对准确性与鲁棒性的贡献，并证明即使基于文本预训练的LLaMA-3 8B亦能超越专门的时序基模型。

**🔧 技术方法**

采用4-bit量化LLaMA‑3 8B，LoRA实现参数高效微调；递归多步推理实现自回归预测；整数化输入将价格四舍五入到整数；Context‑Task‑Format提示结构化指令；零温度解码消除采样随机性。

**📊 数据集**

使用从Binance抓取的BTC/USDT分钟OHLCV数据，经过按小时聚合得到约一年多的小时级收盘价序列，按80%/10%/10%时间序列划分为训练、验证、测试集。

**📈 对比分析**

与八种Transformer及时序基模型（Autoformer、Crossformer、PatchTST、TimesNet、Chronos等）进行对比，PRICE在验证集和测试集的MAE、MSE、MAPE、SMAPE、NRMSE均取得最低或接近最低误差，显示显著优于基线模型且在不同市场周期下保持更稳定的表现。

**⚠️ 局限性**

局限性包括仅针对单一资产（比特币）、固定时间粒度（小时）和预测时长；仅测试了LLaMA‑3 8B一个LLM；未验证在其他金融品种、不同频率或更长预测窗口的泛化能力；实验受限于所用数据分割与市场周期的特定性。

---

## 453. Latency-Optimal Geo-Distributed Storage over Structured Networks

**arXiv ID:** 2609.05229 | [PDF](https://arxiv.org/pdf/2609.05229v1)

**作者:** Madhura Pathegama `[一作]` (Georgia Tech), Viveck Cadambe `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在加权图模型下的地理分布式存储文件分配，目标是最小化平均检索延迟。

**💡 创新点**

创新点包括证明对于固定文件数k≥3，寻找延迟最优分配是NP‑难的；并提出“greedy”分配概念，在某些图结构（加权树、特定循环、单权高最小度图）下可有效构造并达到最优平均延迟。

**🔧 技术方法**

主要技术包括图论中的k‑site定义、domatic number归约、分离子图引理、以及针对树和循环的贪心构造算法。

**📊 数据集**

论文未使用具体实验数据集，而是以理论证明与算法复杂度分析为主。

**📈 对比分析**

方法与传统的最优平均延迟求解相比，在可行图结构上实现了多项式时间最优解，证明了在这些结构中greedy分配可达到全局最优。

**⚠️ 局限性**

限制在于对一般网络的最优分配仍为NP‑难，且提出的高效构造仅适用于特定拓扑；在更一般的地理分布网络中仍缺乏近似或启发式解。

---

## 454. Conserved Immune Topology Improves Pathology Foundation Model Generalization for Cross-Cancer MSI-H Prediction

**arXiv ID:** 2609.05182 | [PDF](https://arxiv.org/pdf/2609.05182v1)

**作者:** Dasari Naga Raju `[一作]` `[通讯]` (Independent Researcher), Dasari Naga Raju (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于免疫空间拓扑的轻量级特征CIT，用于跨癌症的MSI‑H预测

**💡 创新点**

创新点在于利用无监督聚类提取免疫相关图块，并计算四类生物学上可解释的空间描述子（TLS、周边免疫反应、多尺度TIL密度、免疫–肿瘤混合），实现跨组织的免疫拓扑表征

**🔧 技术方法**

技术包括：预训练的病理基础模型(UNI2‑h、CONCH)提取图块嵌入；K‑means+DBSCAN进行免疫图块聚类；计算十维空间特征并与嵌入拼接；在多实例学习(MIL)框架中使用ABMIL、CLAM‑SB和TransMIL聚合器

**📊 数据集**

数据集：TCGA‑COAD（训练/内部交叉验证）、CPTAC‑COAD（跨站点测试）和TCGA‑STAD（跨癌种零样本转移）

**📈 对比分析**

与仅使用基础模型嵌入的基线相比，CIT在跨癌种零样本转移中实现了绝对提升0.0534（TransMIL），在跨站点测试中提升至+0.0407，统计显著；在内部验证中提升仅在TransMIL上为0.017，表明保持原有准确度的同时显著提升跨域泛化

**⚠️ 局限性**

局限性：免疫聚类无人工标注验证，且评估仅限于胃肠道癌症，未在其他肿瘤类型验证跨癌种通用性

---

## 455. LIBERO-RECOVER: Beyond Task Success Towards Failure Recovery in Robotic Manipulation Models

**arXiv ID:** 2609.05178 | [PDF](https://arxiv.org/pdf/2609.05178v1)

**作者:** Lin Liu `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LIBERO‑Recover基准，收集并评估机器人在真实执行失败后的恢复能力。

**💡 创新点**

首次从实际模型运行中提取失败场景，并构建四级恢复难度的评估体系，聚焦失败识别与恢复而非仅完成任务。

**🔧 技术方法**

使用VLA与WAM模型作为失败生成器，利用大型语言模型Qwen3.5‑27B‑Instruct进行失败定位与分类，制定恢复成功率、恢复退化率、恢复一致性等评估指标。

**📊 数据集**

基于LIBERO原始任务（130子任务）构建2178个失败恢复场景，并收集3184条人类恢复轨迹用于微调。

**📈 对比分析**

在六个代表性模型上进行多级评估，标准基准近100%成功率，但失败恢复率低于15%，随着恢复难度升高性能急剧下降；WAM模型相对更稳健；更小的动作块尺寸和加入时间上下文可提升恢复。

**⚠️ 局限性**

恢复训练难以转移到标准任务中的自然失败；现有模型在高难度状态恢复（L3‑L4）表现不足；基准分布聚焦于少数关键状态，需进一步多样化。

---

## 456. Hatebench in the era of safer LLMs

**arXiv ID:** 2609.05169 | [PDF](https://arxiv.org/pdf/2609.05169v1)

**作者:** Ole Becker `[一作]` (Hasso Plattner Institute), Vaibhav Bajpai `[通讯]` (Hasso Plattner Institute)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现并扩展了 HateBench 研究，评估 LLM 生成的仇恨言论对现有检测器的影响，并检验检测器在新模型和攻击场景下的鲁棒性。

**💡 创新点**

创新点在于提供独立重现流程、使用 LLM 作为判定器自动标注扩展数据集、加入最新 LLM（GPT‑5‑nano、GPT‑OSS、Mistral‑Instruct）以及 omni‑Moderation，系统评估对抗与隐蔽攻击对检测器阈值的影响。

**🔧 技术方法**

采用 LLM 生成、LLM‑judge 自动标注、检测器阈值反向工程、对抗攻击（DeepWordBug、TextBugger、PWWS、TextFooler、Paraphrase）与模型窃取攻击，并使用 BERT/RoBERTa 代理模型、OpenAI、Perspective API 等技术。

**📊 数据集**

复现 HateBenchSet，构建 ExtendedHateBenchSet（34 个身份组、9 个 LLM，原始与破解两种状态），使用 MHS 人工生成数据集，以及 120 例样本的对抗实验集。

**📈 对比分析**

通过 F1、准确率、召回率、精度等指标，对比原论文阈值与我们最优阈值；结果显示大多数检测器与原论文相近，omni‑Moderation 在鲁棒性上略优，但对抗与隐蔽攻击仍能显著绕过检测，攻击成功率普遍高于原始阈值。

**⚠️ 局限性**

主要局限在于使用 LLM 作为判定器导致标注噪声，缺乏人工标注；样本量有限且对阈值高度敏感，攻击效果受样本偏差影响；实验受 API 速率限制与硬件性能限制。

---

## 457. Conformal Prediction for Offensive Security

**arXiv ID:** 2609.05165 | [PDF](https://arxiv.org/pdf/2609.05165v1)

**作者:** Giovanni Cherubin `[一作]` `[通讯]` (Microsoft), Giovanni Cherubin (Microsoft)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文探讨了将合成预测（Conformal Prediction）用于攻击场景，具体提出了在隐私保护机器学习中的数据记录重建攻击以及在 Tor 网络中的网站指纹攻击中使用 CP 的方法。

**💡 创新点**

创新点在于：①将 CP 的集合预测与可控误差率引入攻击，提供比单点预测更丰富的攻击信息；②利用 CP 的无限可生成样本特性和覆盖保证，在重建攻击中显著提升重建准确度；③在网站指纹攻击中通过 CP 自然处理开放世界与多标签情形，实现更鲁棒的攻击框架。

**🔧 技术方法**

主要技术包括：合成预测（CP）及其分割式（inductive）实现、Conformalized Quantile Regression (CQR)、LightGBM 生成非一致性度量、基于 DF CNN 的软最大非一致性评估。

**📊 数据集**

使用了 Adult 数据集进行数据记录重建攻击；使用 Tor 网络 95 网站的流量数据集（每个网站 1,000 条轨迹）进行网站指纹攻击。

**📈 对比分析**

与传统点预测基线和无攻击基线比较，CP 在无隐私设置下的重建误差明显低于基线，覆盖率满足 α=0.1；在网站指纹攻击中，CP 的误差随 α 调整，监测网站平均预测集大小接近 1，未监测网站往往产生空集，表明能够有效识别开放世界场景。

**⚠️ 局限性**

局限性包括：CP 的有效性是平均性质，单个样本可能不满足覆盖保证；在高维或复杂分布中需要大量校准样本；开放世界下空集判断依赖于 α 设定，若 α 选择不当可能导致误判。

---

## 458. APEX-RBD: Mixed-Precision Exploration Framework for Hardware-Efficient Robot Dynamics Accelerator Design

**arXiv ID:** 2609.05161 | [PDF](https://arxiv.org/pdf/2609.05161v1)

**作者:** Xingyu Liu `[一作]` (Hong Kong University of Science and Technology), Wei Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对机器人控制中的刚体动力学加速器，提出了APEX-RBD框架以实现混合精度量化并降低面积与功耗。

**💡 创新点**

创新点在于结合物理驱动的变量分组与敏感度分析进行搜索空间剪枝，以及用层次化特征工程的随机森林代理模型快速预测轨迹误差，从而在满足精度约束下高效搜索最佳混合精度配置。

**🔧 技术方法**

采用变量分组、敏感度分析、随机森林代理模型、层次化特征工程、贝叶斯优化、局部贪心微调以及基于单元库的硬件成本估计。

**📊 数据集**

使用针对三款机器人（7-DoF iiwa、12-DoF HyQ、29-DoF Atlas）的工作负载数据集，每台机器人采集40条代表性轨迹，使用Pinocchio库进行仿真。

**📈 对比分析**

与统一32位、24位基准及模拟退火、随机搜索、纯贝叶斯优化等方法对比，APEX-RBD在保持相同运动精度的前提下，面积可节约最多1.9倍、功耗可节约最多1.8倍。

**⚠️ 局限性**

局限在于仍需大量闭环仿真采集数据，搜索时间受数据采集支配；模型仅适用于固定点量化，且对不同架构或非固定点实现的推广需进一步验证。

---

## 459. Large Language Models with At Most One Spike per Neuron

**arXiv ID:** 2609.05151 | [PDF](https://arxiv.org/pdf/2609.05151v1)

**作者:** Zhuoya Zhao `[一作]`, Richard Naud `[通讯]` (University of Ottawa)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种基于参考时间的TTFS编码，构建了完整的TTFS-SNN LLM架构，支持嵌入层、层归一化、注意力和Dropout等模块，训练并在BERT和GPT-2上实现了1.5B参数级别。

**💡 创新点**

提出了参考时间R-TTFS方案，使TTFS神经元支持负激活；将四大LLM核心模块映射为TTFS层；实现了从0到1.5B参数的全TTFS LLM，并在自然语言理解上接近ANN性能。

**🔧 技术方法**

采用时间到首次发放（TTFS）编码，参考时间机制，shift-ReLU映射，基于Spike的矩阵乘法与层归一化近似，使用全端到端训练。

**📊 数据集**

使用Wikipedia、BookCorpus、FineWeb-Edu进行预训练；在GLUE基准、lm-evaluation-harness上的Commonsense Reasoning和WikiText/ LAMBADA评估。

**📈 对比分析**

与多种SNN方法和原始ANN对比，在GLUE上TTFS-BERT Base平均得分80.5、Large 82.3；在GPT-2上TTFS-GPT-2 Small/XL在Commonsense Accuracy可与ANN匹配，但在Perplexity仍有显著差距。

**⚠️ 局限性**

语言建模指标（Perplexity）仍落后，无法捕捉长距离依赖；缺乏真实神经形态硬件能耗验证。

---

## 460. Compact Neural Appearance Models for Efficient Gaussian Splatting

**arXiv ID:** 2609.05255 | [PDF](https://arxiv.org/pdf/2609.05255v1)

**作者:** Florian Hahlbohm `[一作]` (TU Braunschweig), Marcus Magnor `[通讯]` (University of New Mexico)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文对3D高斯散点（Gaussian splatting）中基于视角的辐射模型进行统一、可扩展的实现与系统评测，涵盖传统的球谐（SH）、球面Voronoi（SV）、归一化各向异性球面高斯（NASG）与NASGabor，以及一种基于共享MLP的隐式神经视角模型。

**💡 创新点**

创新点包括：① 用统一的接口与JIT编译方式把所有模型融合进可微分CUDA光栅器，实现无额外内核调用的高效训练与渲染；② 设计一种极简的隐式神经视角解码器，仅用8维潜在码+共享小型MLP，显著降低每个原始体素的内存占用（192字节→28字节）并提升优化速度；③ 在桌面GPU与移动WebGL上同时评估多种模型，揭示显式高频球面模型与神经模型在不同硬件平台上的性能差异与取舍。

**🔧 技术方法**

使用的技术包括：可微分高斯散点渲染（3DGS），runtime code‑generation（tiny‑cuda‑nn），GPU加速的Tensor Cores、FMA运算，WebGL/Three.js 3D渲染，颜色激活函数（ReLU、softplus、sigmoid）以及 per‑image signal processing（PPISP）来抑制拍摄时的光照与曝光漂移。

**📊 数据集**

实验数据集：Mip‑NeRF 360、Tanks & Temples、Deep Blending、NeRF Synthetic 共21个场景，使用与原始3DGS论文相同的图像分辨率与训练/测试划分。

**📈 对比分析**

比较方法：在同一优化管线、相同超参数（仅视角模型不同）下，衡量重建质量（PSNR/SSIM）、优化时间、峰值显存与CUDA渲染速度；在WebGL上评测720p帧时。结果显示：神经模型在保持或略优于第三阶SH质量的同时，显存占用、训练时间和CUDA渲染速度均提升；NASG/NASGabor在CUDA上最快，但在WebGL上与SV相差不大；SV因参数量大在移动端易耗尽显存。

**⚠️ 局限性**

局限性：① 传统球面模型与隐式模型均无法准确模拟镜面反射；② 隐式共享MLP无法解析旋转，限制场景组合与编辑；③ 随着场景规模扩大，单一共享网络可能成为容量瓶颈；④ WebGL缺乏Tensor Core支持，导致神经解码器在移动端不具优势，需等待WebGPU等新API。

---

## 461. Cross-Domain Tracker Adaptation Without Target-Domain Labels via Vision-Language Agents

**arXiv ID:** 2609.05239 | [PDF](https://arxiv.org/pdf/2609.05239v1)

**作者:** Daniel Davila `[一作]` (Cisco Systems, Inc.), Mike Cochran `[通讯]` (Cisco Systems, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种使用 Vision‑Language Model（VLM）作为诊断代理，在没有目标域标签的情况下，通过分析追踪器输出的视觉诊断信息，自动调整 detect‑to‑track 系统的运行参数，实现跨域追踪器适配。

**💡 创新点**

创新点在于：①将 VLM 用作闭环控制器而非仅评估器；②通过视觉诊断面板（检测/跟踪失败模式）和结构化提示，指导参数更新；③实现了无需目标域标签的自适应调优，且在大域移位时可恢复大部分性能。

**🔧 技术方法**

核心技术包括：Gemma‑4‑31B 视觉‑语言模型、两阶段（检测、跟踪）自适应循环、窗口级诊断与序列级聚合、基于视觉面板的失败模式分类、以及对比 Bayesian 优化与源域 oracle 转移。

**📊 数据集**

使用了 MOT17、MOT20 和 DanceTrack 三个多目标跟踪基准数据集进行实验，涵盖不同密度、运动速度和摄像机视角的场景。

**📈 对比分析**

与源域 oracle 直接转移、目标域 oracle（有标签）以及无标签 Bayesian 优化基线对比，结果显示：在 MOT17→MOT20 转移中，VLM 调优平均提升 HOTA 0.061，恢复 67.8% 的性能损失；对最难的 MOT20‑03/05 还恢复 86.7%；与 BO‑proxy 比较，VLM 在大域移位时更稳健，且在低域移位时几乎不造成退化。

**⚠️ 局限性**

局限性包括：①只能调优暴露的参数，无法解决需要修改检测器权重或重识别模型的域移位；②仍需源域有标签来计算 oracle；③在源域与目标域差异不大时可能轻微退化，尽管两阶段结构可减轻。

---

## 462. Phase Transition Frequency as a Training Time Predictor of Test Accuracy in ResNets

**arXiv ID:** 2609.05194 | [PDF](https://arxiv.org/pdf/2609.05194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 463. CABAL: Multi-Agent Simulacra for Tracing the Effects of Collusive Bidding in Peer Review

**arXiv ID:** 2609.05227 | [PDF](https://arxiv.org/pdf/2609.05227v1)

**作者:** Jicheng Zhou `[一作]` (University of Macau), Jiantao Zhou `[通讯]` (University of Macau)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CABAL，一个端到端多代理模拟框架，用于在固定会议环境下研究协同投标对评审分配与评审结果的生命周期影响

**💡 创新点**

通过基于互惠亲和度构造协同投标圈和目标论文，保持投标与真实专业匹配的一致性；同时实现同一会议环境下的对比因果实验

**🔧 技术方法**

使用 LLM 驱动的评审代理、互惠亲和度图、贪心匹配器、以及基于预先定义的规则的投标/评审流程

**📊 数据集**

140 名来自 Semantic Scholar 的研究者档案，100 篇合成提交（cs.LG、cs.CV、cs.CL、cs.AI），以及独立评分代理生成的参考质量分数

**📈 对比分析**

在 7 次实验运行中与全诚实情景对比，协同投标使目标论文的分配率从约 20% 上升到 60% 以上，目标论文平均评分提升约 0.9 分；会议整体评分平均提升 0.13–0.38 分，质量相关性轻微下降；现有投标阶段检测器在精准度与覆盖率之间存在显著权衡，难以完整识别分布式协同群体

**⚠️ 局限性**

实验仅基于合成数据和单一协同策略，缺乏真实会议中的多样性；LLM 模拟的行为可能与人类评审偏差不完全一致；检测器评估受输入视图限制，未涵盖所有潜在防御方法

---

## 464. Risk-Aware Optimal Control with Rulebooks

**arXiv ID:** 2609.05199 | [PDF](https://arxiv.org/pdf/2609.05199v1)

**作者:** Tichakorn Wongpiromsarn `[一作]` `[通讯]` (Iowa State University), Tichakorn Wongpiromsarn (Iowa State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对规则书（rulebook）描述的风险感知最优控制问题的任意时（anytime）算法，能够在有限计算预算下给出可证的最优性间隙，并返回满足优先级约束的控制策略。

**💡 创新点**

创新点包括：①将多优先级风险约束表述为阈值化的超额风险向量的词典序优化；②在黑盒风险评估函数上结合 Lipschitz 分支限界与过滤机制，能够在每个规则层级上逐步缩小可行域并维护可证的下、上界；③提供了任意时最优性间隙的理论保证，并证明在额外假设下该间隙随计算预算趋近于零。

**🔧 技术方法**

采用的技术主要有：Lipschitz 连续性下的分支限界（branch‑and‑bound）框架、盒子覆盖与分裂、阈值化超额风险下的竞争盒子判定、递归规则层级过滤、基于中心点的盒子界估计，以及 Monte Carlo 采样实现的 CVaR 风险评估。

**📊 数据集**

实验使用了两组数据集：一是可解析解的单规则 Rosenbrock 变形函数的合成基准；二是基于真实道路几何的高速公路合流仿真，采用 Monte Carlo 场景集合并 CVaR 作为风险度量。

**📈 对比分析**

与基线方法（最佳优先分支限界、阈值化与非阈值化两种变体以及均匀网格搜索）相比，算法在任意时表现出更快的盒子覆盖收缩速度、同等或更小的最优性间隙，并在合流仿真中在 50 ms 预算内得到满足前三条规则、第四条规则误差为 0.40 的策略；运行时间主要由黑盒评估决定，算法自身开销可忽略。

**⚠️ 局限性**

局限性包括：①需要预先给定全局 Lipschitz 常数，导致估计保守；②假设风险评估函数是确定性的黑盒，未考虑 Monte Carlo 估计的不确定性或神经网络逼近误差；③对规则书结构的阈值化假设在某些应用中可能过于简化，且在高维搜索空间下盒子分裂的计算复杂度仍然较高。

---

## 465. What Matters in On-Policy Distillation? A Perspective on Data Efficiency and Data Selection

**arXiv ID:** 2609.05198 | [PDF](https://arxiv.org/pdf/2609.05198v1)

**作者:** Zhinan Hou `[一作]` (Tsinghua University), Keyou You `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大语言模型的On‑Policy Distillation（OPD）在数据量和数据选择上的效率进行系统实验，提出1-shot OPD以及仅使用极少硬例子即可匹配完整数据集的训练策略。

**💡 创新点**

创新点在于发现1-shot OPD即单例训练就能显著提升推理性能；硬例子训练能进一步提升效果，且提升主要由长Chain‑of‑Thought（CoT）路径驱动而非高token熵。

**🔧 技术方法**

主要技术包括基于top‑k的逆KL反向损失的OPD框架、长度受限实验、硬例子筛选机制以及token‑级KL分析。

**📊 数据集**

使用的数据集为DAPO‑Math‑17K（子集1K）以及多项数学推理基准：AIME 2024/25、AMC 2023、MATH500、Minerva Math和OlympiadBench。

**📈 对比分析**

实验通过对比Full‑Set 17K和极少例子（1‑shot/8‑shot硬例子）在各基准上的平均准确率，8硬例子可达53.6%≈Full‑Set 53.7%，在不同模型规模上亦能匹配或接近完整数据集性能。

**⚠️ 局限性**

局限性包括仅在数学推理任务验证，缺乏跨领域泛化实验；长CoT训练对计算资源要求高，且对教师模型能力的依赖未完全解决。

---

## 466. Compression Beyond the Uncompressed: A Two-Stage Training Recipe for Soft Context Compression in RAG

**arXiv ID:** 2609.05152 | [PDF](https://arxiv.org/pdf/2609.05152v1)

**作者:** Shuyu Guo `[一作]` (Shandong University), Zhaochun Ren `[通讯]` (Leiden University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了两阶段训练方案DEX-Comp，结合纯蒸馏与硬探索实现软上下文压缩，提升RAG在多深检索下的效果与效率。

**💡 创新点**

①仅蒸馏教师正确样本，避免错误示例污染；②将强化学习探索限定在教师失败样本上，促使压缩模型突破令牌级策略；③证明软压缩可在高检索深度下超越未压缩RAG。

**🔧 技术方法**

软压缩（连续嵌入压缩）、纯蒸馏（KLD）、强化学习（GRPO）、LLM-LoRA适配、检索（Splade, DeBERTa）、评测用Gemini 3 Flash judge。

**📊 数据集**

Natural Questions、TriviaQA、HotpotQA、ASQA、PopQA，以及 BioASQ、CovidQA、FEVER 等外部数据集。

**📈 对比分析**

在 top‑5 到 top‑30 检索深度上与未压缩 RAG、LLMLingua‑2、xRAG、ICAE、COCOM、PISCO 等基线对比。DEX-Comp 以 16× 压缩率在 top‑30 下 TTFT 提升约 24×、GFLOPs 降 13.6×、显存缩 4×，且 CEM 与 LLM‑judge 得分均超过未压缩 RAG，平均提升 2–3 分。

**⚠️ 局限性**

需要离线预压缩并存储嵌入，适应频繁更新或极大语料成本；无法完全分离压缩与任务 RL 的效益；评测仅聚焦答题准确性，未考查可信度、对抗鲁棒等方面。

---

## 467. Beyond Stationarity in Time Series: Discovering Causal Structures and Latent Regimes via Markov Blankets

**arXiv ID:** 2609.05150 | [PDF](https://arxiv.org/pdf/2609.05150v1)

**作者:** Lei Zan `[一作]` (University Grenoble Alpes), Eric Gaussier `[通讯]` (University Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种针对非平稳时间序列的因果结构发现算法RCBNB-MB，通过标记马尔可夫毯实现鲁棒的 regime 分段和因果图恢复。

**💡 创新点**

将马尔可夫毯用于 regime 归属预测，改进传统使用父集的方法，并在同一算法框架下同时完成 regime 识别和因果图恢复。

**🔧 技术方法**

结合 CBNB（约束+噪声）因果发现与马尔可夫毯预测，迭代求解，使用条件独立检验、时间排序、噪声模型和预测误差最小化等技术。

**📊 数据集**

在模拟数据（6 变量、600/1200 步、2/3 regime）和真实 IT 监控数据（8 变量、2100 步、已知异常区间）上进行实验。

**📈 对比分析**

与 PCMCI、VarLiNGAM、Dynotears、CASTOR 等基线方法对比，RCBNB-MB 在 regime 识别误差率低于 1%，因果图 F1 分数最高（0.73/0.66），结构误差 (nSHD) 最低，优于其他方法。

**⚠️ 局限性**

仅评估线性关系、少于三 regime，难以处理边界时间点、需要已知 regime 数目且对参数敏感。

---

## 468. GLASS: Graph-Language Alignment with Spherical Scoring for Transferable Graph-Level Anomaly Detection

**arXiv ID:** 2609.05253 | [PDF](https://arxiv.org/pdf/2609.05253v1)

**作者:** Xudong Wang `[一作]` (Chinese University of Hong Kong), Jicong Fan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于图‑语言对齐的图层级异常检测框架，将图结构序列化为 Graph Descriptor Prompt（GraphDP），并通过冻结的指令感知文本编码器将其映射到单位超球面上，与结构化图编码器进行多切片软余弦对齐，最终采用基于von Mises–Fisher 核密度的球面参考集评分，实现单域、零样本跨域及少样本自适应的图异常检测；

**💡 创新点**

核心创新在于：①将图结构统一转化为可直接对齐的语言提示（GraphDP），②利用多切片 Matryoshka 纹理在超球面上保持多尺度一致性；③提出球面多模态评分（SMS）并证明其在高浓度极限下等价于角度最近邻评分，形成统一的非参数异常判定；④通过冻结文本空间，仅学习图侧投影，显著提升跨域鲁棒性；

**🔧 技术方法**

使用的技术包括：GNN（GIN）作为结构编码器；Qwen3‑Embedding 0.6B 作为文本编码器；多切片 Matryoshka 表示学习；Gram 正则化对齐；von Mises–Fisher 核密度估计；k‑NN 角度距离评分；参考集校准；以及对抗扰动正则化；

**📊 数据集**

实验数据集涵盖 12 个常用 GLAD 任务，分为分子（MUTAG、DHFR、BZR、COX2、AIDS、NCI1）、蛋白质（PROTEINS、D&D、ENZYMES）和社交网络（IMDB‑BINARY、COLLAB、REDDIT‑BINARY）三大元域；

**📈 对比分析**

与传统图核+SVM、GNN 端到端单类学习（OCGIN、GLocalKD、OCGTL、SIGNET、GLADC、CVTGAD、MUSE、UniFORM）等基线对比，平均 AUROC 79.85，平均排名 1.75，单域表现居前；在零样本跨域场景中，分子→蛋白质迁移提升约 3–5 pp，蛋白质→分子亦可；少样本校准下仅需 8–16 条正常样本即可接近单域性能；

**⚠️ 局限性**

局限性包括：①对 GraphDP 语义覆盖度有限，导致 ENZYMES、DHFR 等细粒度生化任务表现相对不足；②跨域迁移到社交网络（REDDIT‑BINARY）仍显著受限，说明不同领域的结构语义差异需要更丰富的提示词；③依赖冻结文本编码器，若文本模型与图语义偏差过大，可能影响对齐质量；④高维球面计算成本随切片数和维度增长；

---

## 469. Hessian-based molecular conformation augmentation for a scalable and efficient strategy of machine learning interatomic potentials

**arXiv ID:** 2609.05233 | [PDF](https://arxiv.org/pdf/2609.05233v1)

**作者:** Bumju Kwak `[一作]` (Independent researcher), Jeonghee Jo `[通讯]` (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两种基于分子Hessian的Taylor展开数据增强方法（UniAug 与 ModeAug），通过在原始结构附近产生扰动并生成相应的能量与力标签，从而在训练时无需额外的高阶自动微分或修改损失函数。

**💡 创新点**

创新点在于：①将Hessian信息作为数据生成的先验而非直接监督，避免了高阶求导带来的计算与内存开销；②设计两种可互补的扰动分布，既可探测近似谐振子附近的曲率，又能覆盖更广泛的非平衡结构；③实现了与任何现有MLIP架构无缝对接的“即插即用”方案。

**🔧 技术方法**

技术包括：二阶Taylor展开生成能量与力标签；UniAug 在随机原子子集上进行等方差高斯位移；ModeAug 在质量加权Hessian特征向量（正则化的频率倒数）方向上采样；在训练中仅将这些合成样本按原始比例混合，无需改变损失函数或网络结构。

**📊 数据集**

使用了三类基准数据集：HORM（非平衡和过渡态结构，含完整Hessian）、HessianQM9（平衡结构在多种溶剂下的Hessian）以及MD17（无Hessian，仅用于零射训练后 MD 测试）。

**📈 对比分析**

与标准仅用能量-力（E–F）训练、以及在论文中已报道的显式Hessian监督（E–F–H）方法进行对比。结果表明：①在 HORM 上，UniAug 在直接力训练下的力 MAE 可低于 E–F–H；②ModeAug 在 OOD 评估、MD 轨道稳定性和 Hessian 复原度上更优；③两种增强方式均显著提高了 Hessian 的 MAE 与特征频率的预测误差，使其达到 20–30% 的降低幅度。总体来看，数据增强方案在不增加训练成本的前提下，逼近甚至超过了显式 Hessian 监督的性能。

**⚠️ 局限性**

局限性：①需要预先计算并提供每个样本的 Hessian，适用于已有Hessian数据集；②Taylor 展开仅在小位移范围内有效，对大尺度构象变化（如绕键旋转）不适用；③需要在每个数据集上手动调节位移尺度 σ，且不同扰动方式对 σ 的敏感度不同；④在平衡结构中原子力几乎为零时，标签信息稀薄，增强效果有限；⑤对极低频（几乎零能量）模式的处理仍存在不确定性。

---

## 470. ACE: Adaptive Calibration-Free Expert Skipping for MoE-based LLMs

**arXiv ID:** 2609.05228 | [PDF](https://arxiv.org/pdf/2609.05228v1)

**作者:** Zukang Xu `[一作]`, Dawei Yang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ACE框架，实现在Mixture-of-Experts大型语言模型中训练、校准、检查点无关的动态专家跳过；

**💡 创新点**

通过全局谱代理（GSP）估计专家结构响应能力，并结合路由条件精炼（RCR）获取路由指向的局部响应，采用最大交集融合避免误删重要专家；

**🔧 技术方法**

基于SwiGLU专家的权重矩阵、RMSNorm尺度、路由权重中心化、几何平均与正则化等统计量；

**📊 数据集**

在Qwen3-30B-A3B-Instruct-2507、Qwen3.6-35B-A3B、Gemma-4-26B-A4B-it等模型上，使用WikiText‑2、ARC‑Challenge/Easy、PIQA、MATH‑500、GPQA‑Diamond、HumanEval、LiveCodeBench七个下游任务；

**📈 对比分析**

与Score、NAEE、MoDES、DiEP等基线对比，ACE在10%‑60%跳过率下保持或提升准确率和困惑度，尤其在50%‑60%时优于所有对手，且在A100上实现1.7‑2.3×的延迟提升；

**⚠️ 局限性**

缺点包括需预先预计算专家统计量，跳过阈值需基于未标记数据映射，且对极端跳过率下的鲁棒性和跨任务阈值迁移性仍待验证。

---

## 471. FedDRAW: Federated Dual Reputation Annealing Weighting for Heterogeneous Multi-Institutional Chest Radiograph Classification

**arXiv ID:** 2609.05223 | [PDF](https://arxiv.org/pdf/2609.05223v1)

**作者:** Maryam Moradpour `[一作]` (Institute for Predictive Deep Learning in Medicine and Healthcare), Anne-Christin Hauschild `[通讯]` (Institute for Medical Informatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种联邦学习聚合权重方法 FedDRAW，利用两种耦合的退火调度在训练过程中动态平衡数据量先验与客户端模型与全局模型的相似度，从而在保证模型性能的同时减轻大客户端对全局模型的主导作用，并最终实现聚合级别的公平性。

**💡 创新点**

创新点在于：①引入了“内层退火”将聚合权重的依据从单纯的样本量逐步过渡到基于相似度的评价；②引入了“外层退火”在训练后期将权重分布逐步拉平至均匀，避免大客户端长期占优；③将两种退火同时作用，既保证早期训练的稳定性，又兼顾后期的公平性，形成完整的双重退火框架。

**🔧 技术方法**

技术方法包括：联邦学习标准通信流程；客户端使用标准本地训练；服务器端对上传的模型参数（仅最后一层）计算余弦相似度；用指数衰减调度 γ_t = exp(-ηt) 调节数据量与相似度的混合比例；用指数递增调度 β_t = β_max(1-e^{-λ(T-t)}) 控制软max 的温度，从而将相似度分数映射为聚合权重；最终通过加权平均得到新的全局模型。

**📊 数据集**

实验数据集：公开胸部X光多标签分类数据 CheXpert（191k张）和 ChestMNIST（78k张）；通过人为划分12种不同的客户端分布（数量失衡、病种偏移、均匀随机分配）构造多场景对比。

**📈 对比分析**

对比方法包括：FedAvg、FedNova、FedAdp、FedProx、SCAFFOLD、MOON、FedDyn 共7种基线。实验采用相同本地超参，训练15轮，每轮2个本地epoch。评价指标为宏平均 AUC 与 GM（灵敏度·特异度的几何平均）。FedDRAW 在所有12个场景下均获得最高 GM，且在 AUC 上往往排名第一或接近第一；在统计检验（Friedman + Nemenyi）中显示显著优于其他方法，且在不均衡场景下的性能提升更为显著。

**⚠️ 局限性**

局限性包括：①需要预先设定两种退火速率（η、λ）和 β_max，缺乏自动调参机制；②相似度仅基于最后一层参数，可能在某些任务中信息不足；③假设所有客户端均始终参与并上传模型，未考虑客户端掉线或恶意攻击；④在极快收敛或极端异构环境下，退火曲线可能不匹配训练动态，需要进一步实验验证。

---

## 472. A Verifier-Guided Explainable Reasoning Framework with Gold-Anchored QLoRA, Task-Aware Mixture-of-Experts, and Group-Relative RLVR

**arXiv ID:** 2609.05221 | [PDF](https://arxiv.org/pdf/2609.05221v1)

**作者:** Thi Kim Trang Vo `[一作]` (University of Information Technology), Duy Phuong Tran `[通讯]` (University of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一套验证器引导的可解释推理框架，结合金标Anchored QLoRA、任务感知Mixture-of-Experts与Group-Relative RLVR，以提升教育领域问答的推理透明度与可靠性。

**💡 创新点**

创新点包括：①使用金标Anchored QLoRA实现领域加权监督，保证答案与证据一致；②轻量级任务感知路由将逻辑与物理问题分别送至FOL/Z3与公式/单位符号验证器；③基于三维度（答案正确、证据/单位一致、推理深度）设计的P1/P2/P3奖励，配合verifier‑guided自我修订与群组相对RLVR；④在推理阶段采用五生成自我一致性与可选物理验证器，避免全局依赖金标。

**🔧 技术方法**

主要技术包括：Qwen2.5-3B‑Instruct + 4‑bit NF4 QLoRA、LoRA、任务感知MoE路由、FOL/Z3逻辑验证器、公式/单位符号求解器、Group‑Relative RLVR（GRPO）、自我一致性、verifier‑guided自我修订、P1/P2/P3三维度奖励设计。

**📊 数据集**

使用了 EXACT 2026 评测集（逻辑多选、是非、不确定、物理共2,162例，训练1,724例，验证438例）进行实验。

**📈 对比分析**

通过与校准SFT基线及RLVR模型对比评估，RLVR将推理深度指标P3从50.68%提升至72.20%，答案正确率P1保持≈55.9%（略降），证据/单位一致性P2略降至75.3%；五生成自我一致性和物理验证分别在模型层和系统层进一步提升P1。与DeepSeekMath、One‑Shot‑RLVR、Scientific Logicality等方法对比，显示在多任务环境下取得更高的P3和稳定的P2，但答案准确率与某些基线相近。

**⚠️ 局限性**

局限性：RLVR对答案准确率提升有限，P2下降表明更深推理可能带来错误或单位不一致；物理子任务受公式选择、数值执行与科学记数法归一化限制；验证器覆盖范围有限，未能覆盖所有问题；需要进一步调节奖励权重、改进数值与符号归一化，并在更大规模、跨语言的科学推理基准上验证迁移效果。

---

## 473. BLASt3R: Bundle Adjustment of Any Image Set with Multi-View Matching and Monocular Priors

**arXiv ID:** 2609.05210 | [PDF](https://arxiv.org/pdf/2609.05210v1)

**作者:** Vincent Leroy `[一作]` (NAVER LABS Europe), Jérome Revaud `[通讯]` (NAVER LABS Europe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种融合在线与离线结构与运动重建的混合系统，利用快速多视角匹配和可调深度先验，在统一的束调整中同时优化相机参数与稠密点云。

**💡 创新点**

创新点在于：①把前向多视角匹配与可调深度点图结合；②引入可调深度参数并在BA中做正则化；③构建可扩展的记忆机制，实现大规模无序图像集合与长序列视频的统一处理。

**🔧 技术方法**

核心技术包括ViT+Transformer的多视角匹配网络、粗细匹配两阶段、可调深度点图网络（基于raymap+log‑depth basis）、基于Huber的GPU实现束调整和对齐算法。

**📊 数据集**

使用了 Habitat、BlendedMVS、MegaDepth、Mapfree、Co3Dv2、DL3DV、ScanNet++、Unreal4K、VirtualKitti、Infinigen、WildRGBD、HyperSim、ARKitScenes、MegaScenes、TartanAir、TUM‑RGBD、ETH3D‑SLAM、Tanks & Temples 等多种公开数据集。

**📈 对比分析**

在多种评测指标下（ATE、mAA@30、Dense Accuracy/Completeness/Normal Consistency）与 MASt3R‑SfM、VGGT、COLMAP、DA‑V2、MoGe 等基线对比，BLASt3R 在精度、鲁棒性与速度三者上普遍取得或刷新现有 SOTA，尤其在无标定场景下表现出色。

**⚠️ 局限性**

局限性包括：对纯平移或静态视角场景的深度正则化效果有限；在极大尺度 MVS 数据集（如 ETH3D、Tanks & Temples）上仍受限于匹配质量；对计算资源要求较高（GPU 计算与内存）。

---

## 474. Morphology and actuation as inductive biases in robotic hand manipulation

**arXiv ID:** 2609.05206 | [PDF](https://arxiv.org/pdf/2609.05206v1)

**作者:** Zalán Tari `[一作]` (Pazmany Peter Catholic University), Miklós Koller `[通讯]` (Pazmany Peter Catholic University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文对Shadow Dexterous Hand和Anatomically Correct Biomechatronic Hand两款机械手建立统一的结构分析框架，分别量化关节轴几何、驱动比、耦合结构和权威分布对手指运动的影响，并与强化学习实验结果进行对比验证。

**💡 创新点**

创新点在于提出四个形态维度的结构化指标（关节轴几何、驱动比、耦合架构、权威分布），通过条件数、冗余和操纵度量分解机械手的协调瓶颈，并首次将这些结构特征与强化学习学习效率关联。

**🔧 技术方法**

采用PoE正向运动学、奇异值分解、条件数与操纵度量，构建动力学矩阵A并计算B=JA；实验使用PPO、DDPG+HER和TQC+HER等强化学习算法。

**📊 数据集**

使用Shadow手的URDF模型和ACBH的MuJoCo XML模型作为参数来源，仿真环境包括Reach、BlockRotateZ和BlockRotateXYZ三种任务。

**📈 对比分析**

通过在相同任务和算法下比较结构指标与学习收敛速度、成功率，结果显示SDH在PPO和BlockRotateZ上收敛更快，而ACBH在Thumb相关任务中因高κ(B)导致TQC+HER成功率明显低，整体上结构预测与实验结果基本吻合。

**⚠️ 局限性**

局限性包括仅评估两款手，未覆盖软体或低成本手；结构指标受仿真参数影响；实验次数有限，未能充分统计；缺乏针对不同任务的自适应控制策略。

---

## 475. Can Large Language Models Anticipate Behavioral Responses to Social Policies? A Case of Pension Enrollment Prediction among China's Flexible Workers

**arXiv ID:** 2609.05189 | [PDF](https://arxiv.org/pdf/2609.05189v1)

**作者:** Yumiao Li `[一作]` (Tsinghua University), Runhuan Feng `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出FlexPension-LLM，专门用于中国灵活就业者的分层养老保险登记预测，并通过DKI-RDistill实现政策信息注入和理据蒸馏。

**💡 创新点**

结合计量经济先验与hukou政策规则，利用知识注入提示+错误过滤的理据监督，将经济学因果机制嵌入LLM，实现可解释的决策轨迹。

**🔧 技术方法**

采用大型语言模型（Qwen 3.5-35B-A3B）通过LoRA/SFT微调，配合Claude Sonnet 4.5教师生成理据、DKI提示、理据蒸馏与错误再生成。

**📊 数据集**

主要使用CHFS 2019灵活就业样本15,672例，并在CHFS 2017、CFPS 2018、CHIP 2018、CLDS 2018四个外部调查中评估泛化。

**📈 对比分析**

在独立盲测和四个外部数据集上与18个基线对比，Composite F1最高0.9316，超越Claude Sonnet 4.5教师和多数基线，外部平均F1为0.7549，性能稳健且接近最先进系统。

**⚠️ 局限性**

模型仅适用于中国养老制度与调查标签，需重建政策输入；适用于聚合政策评估而非个体咨询，且未覆盖更大范围的政策逆向模拟。

---

## 476. SMILE: Self-Explainable Multimodal Information Bottleneck for Medical Diagnosis

**arXiv ID:** 2609.05174 | [PDF](https://arxiv.org/pdf/2609.05174v1)

**作者:** Yuqing Yang `[一作]`, Shujian Yu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于信息瓶颈（IB）的自解释多模态诊断框架 SMILE，能够在训练过程中同时学习预测模型和每个模态的解释子集。

**💡 创新点**

创新点包括：①将解释性嵌入模型学习的目标函数中，使得所选特征与决策过程天然一致；②采用矩阵形式的 Rènyi α‑order 熵函数估计互信息，避免使用 MINE 等辅助网络；③使用 Gumbel‑Softmax 进行可微的 top‑k 选取；④给出理论泛化分析，证明压缩项不会引入准确度的固有折衷。

**🔧 技术方法**

核心技术：信息瓶颈优化、Rènyi α‑entropy 互信息估计、Gumbel‑Softmax 上的可微子集选择、模态特定的解释器网络、编码器+拼接融合、交叉熵损失。

**📊 数据集**

使用了五个多模态医学数据集：BRCA（多组学）、ROSMAP（多组学）、iCTCF（CT + 临床特征）、Glaucoma Grading（视网膜图像 + OCT 图像）以及 REST‑meta‑MDD（rs‑fMRI + sMRI）。

**📈 对比分析**

与传统多组学融合方法（DIABLO、MOGONET、TMC 等）、专门的 COVID‑19 预测模型（De‑COVID19、HoFN+SCResNet、HUST‑19）、以及其他 IB 方法（DMIB）以及后置解释方法（LIME、SHAP）进行对比。SMILE 在所有数据集上均达到或超过 state‑of‑the‑art 的准确率（如 iCTCF 92.4%，BRCA 87.3%），并在解释的 faithfulness（fidelity）和一致性上优于后置解释器。

**⚠️ 局限性**

局限性：①依赖“足够表达的编码器”假设，若此假设不成立互信息估计会失效；②未显式建模跨模态高阶交互，只关注模态内部特征；③在大型高维数据上训练时计算成本和参数量较大，可能限制可扩展性。

---

## 477. WeAgent-MMGenEdit: A Full-Stack Recipe for Multimodal Agentic Image Generation and Editing

**arXiv ID:** 2609.05171 | [PDF](https://arxiv.org/pdf/2609.05171v1)

**作者:** Hui Zhang `[一作]` (Weixin AI, Tencent), Fandong Meng `[通讯]` (Weixin AI, Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了全栈框架WeAgent‑MMGenEdit，用于多模态知识密集型图像生成与编辑，解决外部证据获取、视觉验证、跨模态整合等问题。

**💡 创新点**

创新点在于：①构建专用的多模态Agent Harness，支持检索、显式视觉验证与结构化证据融合；②提出基于三层检查清单的监督与强化学习，提升证据获取与整合质量；③双侧后训练策略，既对Agent策略做SFT+RL，也对图像后端做多参考SFT+Diffusion‑NFT RL，显著提升知识表达与视觉质量。

**🔧 技术方法**

使用技术包括：检索工具（文本/图像搜索）、视觉验证工具（图像问答）、整合工具（代码渲染器）、生成/编辑工具；SFT、基于检查清单的RL（GSPO）、Diffusion‑NFT RL；多参考条件编码（VAE+视觉‑语言编码器）。

**📊 数据集**

数据集包括：WeDataset‑MMGenEdit（约23K SFT轨迹、14.7K RL任务），以及人类审核的双语基准WeBench‑MMGenEdit（300条，生成/编辑各150条，涵盖多跳检索与多图编辑）。

**📈 对比分析**

与直接生成、reason‑then‑generate、开源Agentic系统、以及同Harness下不同策略和图像后端的基线进行对比；在WeBench‑MMGenEdit上，WeAgent‑MMGenEdit在生成与编辑的加权平均得分（W_Avg）分别提升约12–20分，尤其在文本与视觉知识准确度（TKA、VKA）提升显著；在KnowGen与Mind‑Bench等公开基准上亦保持领先。

**⚠️ 局限性**

局限性包括：对大规模图像后端的依赖（需与GPT‑Image‑2或Qwen‑Image等强大模型配合）；训练与推理成本高（检索+验证+RL）；对极长序列或极端多跳检索的鲁棒性尚待进一步验证。

---

## 478. From Vision to Language: Investigating Causal Information Flow in Multimodal Decision-Making

**arXiv ID:** 2609.05149 | [PDF](https://arxiv.org/pdf/2609.05149v1)

**作者:** Davide Testa `[一作]` (Fondazione Bruno Kessler), Albert Gatt `[通讯]` (Utrecht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过在视频-文本注意力路径上进行因果干预，探究了视觉信息在视频多模态决策任务中的流动与作用；

**💡 创新点**

创新点在于将Attention Knockout方法应用于视频生成式多选任务，系统地分析了空间、因果与时间推理中的视觉-语言交互；

**🔧 技术方法**

使用的技术包括Attention Knockout、层级logit‑lens评估、POS层面细粒度遮蔽以及自定义的视觉-文本交互掩码；

**📊 数据集**

数据集为MAIA的三类子集（空间、因果、时间）共4800条视频-文本对，亦在英语翻译样本上验证；

**📈 对比分析**

通过与无干预（全模态）和仅文本条件比较，发现视觉信息主要在候选答案阶段被利用，且在空间/因果任务上表现较好，但时间任务准确率低于基线；

**⚠️ 局限性**

局限在于只评估了两款中等规模VLM（Qwen2.5‑VL与LLaVA‑OneVision），且全句评分可能引入语言偏差，未系统检验不同评分方式对结果的影响。

---

## 479. A Hybrid Predictive Ensemble of Machine Learning and Deep Neural Networks for Early Cardiovascular Disease Risk Assessment

**arXiv ID:** 2609.05146 | [PDF](https://arxiv.org/pdf/2609.05146v1)

**作者:** Balaji Venkateswaran `[一作]` `[通讯]` (Independent Researcher), Balaji Venkateswaran (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种基于物联网（IoMT）与机器学习/深度学习集成的实时心血管疾病预测框架，能够从可穿戴设备收集心电、血压、心率等生理信号，并通过云端处理实现即时风险预警；

**💡 创新点**

创新点在于将传统机器学习模型（SVM、RF、XGBoost）与Bi‑LSTM深度网络进行集成学习，利用IoMT实时数据流和云计算架构，实现高精度、低误报的早期心脏病风险评估，并通过自动化预处理与特征选择提升模型鲁棒性；

**🔧 技术方法**

使用的技术包括：Kalman滤波去噪、缺失值插补、特征选择（互信息/统计重要性）、SVM、随机森林、XGBoost、Bi‑LSTM网络、集成学习（加权投票/元学习器）、云计算平台、实时数据流处理与报警系统；

**📊 数据集**

主要使用的数据集为公开的Kaggle心脏病数据集（1025条记录，14个特征），同时在实验中通过模拟IoMT传感器流与公开的UCI/PhysioNet数据进行模型训练与验证；

**📈 对比分析**

通过与DT、SVM、RF、KNN、NB等传统单模型的对比，ML‑IoT框架在准确率94.45%、精准率95.23%、召回率96.2%、F1‑score95.89%等指标上均显著优于传统方法，体现了集成学习与实时物联网数据融合的优势；

**⚠️ 局限性**

局限性包括：实验主要基于公开数据与仿真IoMT流，缺乏大规模真实临床验证；模型对异常信号的鲁棒性和可解释性（如SHAP可视化）仍需进一步提升；系统在不同设备与网络环境下的可扩展性与低延迟实现尚待优化；

---

## 480. Cutting Down the Tower: Single-Exponential Envy-Free Cake Cutting

**arXiv ID:** 2609.05191 | [PDF](https://arxiv.org/pdf/2609.05191v1)

**作者:** Qilin Ye `[一作]` (Stanford University), Yannan Bai `[通讯]` (Carnegie Mellon University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了一种新的单指数查询量的可绑定罗伯逊‑韦伯协议，用于在任意数量的非分数、可加、非零散度的价值函数下寻找完整的无嫉妒分配

**💡 创新点**

核心创新在于：①将Aziz‑Mackenzie原始协议中指数级（n↑↑6）个部分分配压缩到多项式数量；②设计新的预处理与剪枝步骤，以确保所有“差异”既不是微小也不是中等，从而避免了指数级调用；③利用图论匹配和等级结构，构造多阶段的部分分配，保证后续递归时不产生冲突或重复；④通过一次性调用单次n元子例程，最终将总查询量控制在n^O(1)2^n

**🔧 技术方法**

主要技术包括：
- Robertson‑Webb模型下的cut和eval查询；
- 预处理阶段多次使用裁剪器和分配器以获得“微小”或“大”差异；
- 通过构造图 G_s 的强连通性判定，决定是否递归或继续预处理；
- 对不同档次（profile）类别使用匹配与等级辅助，确保不同代理的分配互不冲突；
- 采用多项式数量的部分分配与“剪裁”子例程，最终通过递归消除残余；
- 复杂度分析利用递归关系和对单次调用的查询上界 n^O(1)2^n

**📊 数据集**

无实际数据集；论文完全在理论框架下证明存在性与查询上界，无实验或基准测试

**📈 对比分析**

与此前的上界(n↑↑6)与( n^8n^2(1+o(1)) )相比，本文将查询量降至单指数级 n^O(1)2^n；下界仅为 Ω(n^2)，两者之间仍存在巨大差距；实验性或数值比较未给出，因为本研究为理论分析

**⚠️ 局限性**

限制包括：
- 仍未突破单指数界限，距离已知下界 Ω(n^2) 仍遥远；
- 却使用了大量子例程调用，尤其是对 n 元的剪裁与分配器，导致指数查询；
- 需要多项式计算复杂度的辅助，实际实现可能受限；
- 结果主要适用于非分数、可加、非零散度的价值函数，对更一般的偏好模型尚不适用

---

## 481. TherMosaic: Accelerating Perceived Thermal Transitions Through Spatiotemporal Thermal Feedback

**arXiv ID:** 2609.05347 | [PDF](https://arxiv.org/pdf/2609.05347v1)

**作者:** Zining Zhang `[一作]` (University of Maryland), Huaishu Peng `[通讯]` (University of Maryland)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出一种名为 TherMosaic 的时空热反馈方法，通过在手指尖上使用 2×2 佩尔帖阵列实现热感知的加速；

**💡 创新点**

创新点在于利用热空间求和与热适应的结合，实现局部预加热/预冷以降低感知阈值，从而将热感知转变时间缩短约 30%–40%；

**🔧 技术方法**

技术上采用多通道佩尔帖控制、PID 温度调节、微型水冷散热以及可穿戴 PCB；

**📊 数据集**

使用了自建的 2×2 佩尔帖实验装置以及在 VR 游戏场景中收集的用户同步感知问卷数据；

**📈 对比分析**

通过三项感知实验和一次 VR 对比，发现相较于统一热源，TherMosaic 在热感知转变时延上平均减少约 2 秒（约 30%–40%），并显著提升视觉-热同步感知；

**⚠️ 局限性**

局限包括需要对热变换进行预调度，且目前仅针对单指实现，扩展至多指或手掌面临尺寸、散热与功耗挑战。

---

## 482. Commonsense Reasoning in Computer Vision: Foundations, Recent Advancements, and Future Directions

**arXiv ID:** 2609.05257 | [PDF](https://arxiv.org/pdf/2609.05257v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 483. Variational Continuation for Double Pendulum Periodic Orbits

**arXiv ID:** 2609.05337 | [PDF](https://arxiv.org/pdf/2609.05337v1)

**作者:** Leo Yao `[一作]` (Massachusetts Institute of Technology), Max Tegmark `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种基于Hessian的变分延续方法，用于在动力学系统中寻找周期轨道，并在理想双摆上完成了完整的周期轨道谱调查

**💡 创新点**

创新点在于利用自动微分和Hessian特征值信息自动定位无积分器的延续方向，能够高效检测分叉与亚调谐分叉，系统性地划分轨道家族

**🔧 技术方法**

使用技术包括自动微分、Hessian特征值分解、傅里叶参数化、Rprop优化器以及线性外推加速收敛

**📊 数据集**

数据集为理想双摆的等质量等长度动力学方程（无外部数据），通过对其状态空间的模拟得到周期轨道

**📈 对比分析**

与传统射击/积分器方法比较，该方法在长周期、接近竖直状态时仍能收敛，积分器误差会发散；变分法在这些条件下误差可低至1e-10，表现优于传统方法

**⚠️ 局限性**

局限在于Rprop梯度优化在高维系统上扩展性差，完整Hessian计算成本高，且方法尚未在更复杂系统中验证

---

## 484. Adaptation Needs in Robotic Systems: Assessing Behavior Trees and Their Enhancement

**arXiv ID:** 2609.05331 | [PDF](https://arxiv.org/pdf/2609.05331v1)

**作者:** Mehran Rostamnia `[一作]` (Gran Sasso Science Institute), Patrizio Pelliccione `[通讯]` (Gran Sasso Science Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过系统文献综述与实证验证，构建了机器人系统适应需求的六类分类，并评估经典行为树（BT）在满足这些需求方面的能力与局限，随后对现有BT增强方法（生成、扩展、进化、精炼）进行了分类与映射，最后通过作者验证、问卷与访谈收集专家意见以验证结论。

**💡 创新点**

①首次将自适应系统与机器人特定的不确定性融合，提出了六大适应需求范畴（知识、感知、执行、系统、任务、环境）；②系统地将经典BT与增强BT的特性与局限与适应需求对应，揭示经典BT难以处理的关键适应场景；③通过三轮循环的研究设计与外部专家验证，提供了方法论上可复制的验证流程。

**🔧 技术方法**

行为树（Behavior Trees）基础模型、BT增强方法（生成、扩展、进化、精炼）、自适应系统与机器人不确定性理论、系统文献综述、定性与定量问卷分析、半结构化访谈。

**📊 数据集**

主要采用已公开的31篇BT相关研究作为核心文献，外部验证样本包括46名机器人/BT专家参与问卷、13名论文作者反馈、6名行业/学术专家访谈；未使用公开的机器人实验数据集，而是以案例与专家经验为依据。

**📈 对比分析**

未进行算法性能对比，而是通过专家问卷对比经典BT与增强BT在满足各类适应需求时的适用性与局限；结果显示，经典BT仅在少数需求（如部分感知更新）可充分发挥作用，绝大多数需求需要BT扩展或外部模块支持。通过交叉表与统计检验（Fisher精确检验、卡方检验）验证不同需求与BT类技术的相关性。

**⚠️ 局限性**

①经典BT缺乏运行时结构重组、非确定性推理与持续状态处理的原生支持；②研究样本量有限（作者验证41.9%，问卷仅46人），存在响应偏倚；③分类与映射依赖作者自述，可能存在解释差异；④未提供统一性能基准，难以量化提升幅度。

---

## 485. Embedded Graph Flows for Categorical Graph Generation

**arXiv ID:** 2609.05328 | [PDF](https://arxiv.org/pdf/2609.05328v1)

**作者:** Ethan Ma `[一作]` (University of Queensland), Guangdong Bai `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种新的图生成模型——Embedded Graph Flows（EGF），通过学习节点和无向边的连续嵌入锚点，将高斯噪声沿直线路径迁移至这些锚点，再通过终端读取将连续状态映射回离散图类别。

**💡 创新点**

其创新点在于：①允许端点几何（节点/边类别的锚点）可学习，摆脱一热向量固定几何；②采用统一的图变换器同时实现去噪与终端读取；③在无向、可变大小、对称性强的图上实现掩码与对称状态共享；④在分子图上表现出最优的分子生成质量。

**🔧 技术方法**

使用了流匹配（rectified flow）、Permutation‑Equivariant 图Transformer、两分支训练（去噪+终端读取）、连续到离散终端投影、密集边状态与对称平均、以及MMD、FCD等评估指标。

**📊 数据集**

在 QM9（最多9个节点的小分子）和 ZINC250k（最多38个节点的药物样分子）数据集上进行实验，采用芳香与 kekulised 编码。

**📈 对比分析**

与 DiGress、GruM 两种基线在同一评估管道下比较。EGF 在 QM9 上四项指标（有效率、FCD、NSPDK MMD、结构相似度）均优；在 ZINC250k 上其 MMD 低于 DiGress，但 FCD 最高、有效率略低于 GruM。

**⚠️ 局限性**

局限性包括：①端点锚点共享导致表达力受限，尤其在大分子复杂环境下；②稠密边状态导致内存和计算成本随最大节点数 N^2 成长；③未覆盖立体化学、同位素、自由基等属性；④仅单一随机种子实验，结果稳定性未充分验证。

---

## 486. RoboSPA: Can VLA Models Go Beyond Simple Scenes and Short-Horizon Tasks?

**arXiv ID:** 2609.05324 | [PDF](https://arxiv.org/pdf/2609.05324v1)

**作者:** Zhenxuan Fan `[一作]`, Yueting Zhuang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了 RoboSPA 数据集与基准，用于评估 Vision‑Language‑Action 模型在细粒度空间推理和长期程序规划上的能力。

**💡 创新点**

创新点包括：①将评估聚焦于 Fine‑Grained Spatial Reasoning 与 Long‑Horizon Procedural Planning 两个维度；②设计多层次难度等级和步进级诊断指标（Object‑Normalized Target Accuracy 与 Progress Score）；③覆盖 5 种机器人、干净与域随机场景，提供 527K 轨迹的规模化数据。

**🔧 技术方法**

技术实现：在 SAPIEN 仿真与 RoboTwin 2.0 框架下自动生成任务；使用 GPT‑5.2 生成多样化指令模板；构建细粒度评价指标并进行大规模数据采集。

**📊 数据集**

使用的数据集是 RoboSPA 本身，包含 56 基础任务、280 个难度层级变体、527K 轨迹；实验中主要以干净场景的 Aloha‑AgileX 机器人为评测平台。

**📈 对比分析**

比较方法：在单任务设置下分别训练并评估四个代表性 VLA 模型（RDT、GO‑1、π_0.5、X‑VLA），使用 Success Rate、ONTTA 与 Progress Score 进行评估。实验显示，所有模型在最高难度下成功率低于 25%，表现出对细粒度空间关系和长时序记忆规划的显著不足。

**⚠️ 局限性**

局限性：仅在仿真环境下构建，侧重桌面场景，缺乏柔性物体、人机交互及开放式任务的挑战，导致现实迁移和更广泛场景适用性受限。

---

## 487. Scalable Detection of Fossil Palynomorphs in Multifocal Digital Microscopy Images

**arXiv ID:** 2609.05323 | [PDF](https://arxiv.org/pdf/2609.05323v1)

**作者:** Abbas Shaikh `[一作]` (Rice University), Arko Barman `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套完整的端到端自动检测古植物碎屑的流水线，能够处理多焦点全切片图像；

**💡 创新点**

创新点包括：①将多焦点图像压缩为二维表征以供检测模型使用；②提出基于边界的非极大值抑制实现大规模去重；③通过I/O优化将推理时间压缩到1小时以内；

**🔧 技术方法**

使用了YOLO26-L和RF‑DETR‑2XL两种现代目标检测模型，并对多焦点图像进行聚焦堆叠或单平面选择；

**📊 数据集**

数据集为美国史密森国家自然历史博物馆提供的847张纳米分辨率切片，其中82张被标注用于训练、验证和测试；

**📈 对比分析**

与传统单平面检测相比，RF‑DETR‑2XL在聚焦堆叠图像上取得AP@50 0.879、AP@50‑95 0.642，推理时间从原始8‑9小时降至约51分钟；

**⚠️ 局限性**

局限性包括样本量相对较小、模型未利用原始三维焦点信息、仅完成检测未实现亚种分类。

---

## 488. Trace2Tower: Transition-Aware EigenTrace Induction of Multi-Level Skills for LLM Agents

**arXiv ID:** 2609.05261 | [PDF](https://arxiv.org/pdf/2609.05261v1)

**作者:** Jiazheng Sun `[一作]` (Fudan University), Xin Peng `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 Trace2Tower 框架，利用 EigenTrace 与对比谱分解将 LLM 代理的执行轨迹转换为可重用的多层次技能金字塔，从而提升任务成功率和执行效率。

**💡 创新点**

创新点在于：①构建包含语义兼容、转移动力学与结果证据的转移感知 EigenTrace 图；②采用对比谱分解抑制失败模式并突出成功对齐的行为模式；③将行为模式自动聚类为动作、程序与策略三层技能，并通过 verifier 反馈实现动态细化。

**🔧 技术方法**

核心技术包括：事件级轨迹分段、语义映射与归一化、图构建与对比谱变换、谱聚类与 eigengap 自动判定、基于上下文检索的层级技能部署算法。

**📊 数据集**

实验数据集为 ALFWorld（134 任务，1,240 条轨迹）和 WebShop（100 任务，400 条轨迹）两大基准。

**📈 对比分析**

与 No‑Skill、Expert‑Crafted、ExpeL、SkillX、Trace2Skill 等基线相比，Trace2Tower 在 ALFWorld 上实现 87.31% 成功率、10.35 步、0.26 无效动作；在 WebShop 上实现 50.67% 精确成功率，明显优于其它自动化方法。

**⚠️ 局限性**

局限性包括：依赖足够多且多样化的成功/失败轨迹；图构建与谱分解对大规模任务仍存在计算开销；对极端稀缺或高度噪声环境的鲁棒性尚待验证。

---

## 489. MEOX: Compact Multimodal Mixture-of-Experts for Earth Observation

**arXiv ID:** 2609.05351 | [PDF](https://arxiv.org/pdf/2609.05351v1)

**作者:** Mohanad Albughdadi `[一作]` `[通讯]` (European Centre for Medium-Range Weather Forecasts), Mohanad Albughdadi (European Centre for Medium-Range Weather Forecasts)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 MEOX，一种仅含约 3 百万参数的多模态掩码自编码器，能够处理不同遥感传感器（光学、合成孔径雷达）以及缺失观测，并通过显式有效性信号实现可变输入融合；

**💡 创新点**

创新点包括：共享传感器特定块+延迟有效融合、可变传感器自适应适配器、低秩残差专家、旋转位置编码、结构化传感器丢弃、平衡的多模态重建损失和专家路由平衡约束，以及一套可解释的路由诊断；

**🔧 技术方法**

技术手段：掩码自编码器、Transformer‑MoE（top‑2 路由）与低秩专家投影、RoPE 注意力、可变网格推理、模态均衡重建 + 重要性惩罚、平衡专家路由损失、预训练/冻结/微调、检索评估；

**📊 数据集**

使用的数据集：MMEarth64（约 1.2M 样本）做预训练；GEO‑Bench v1（BigEarthNet、EuroSAT、So2Sat、cashew segmentation 等）做冻结转移；BENv2‑14k 做检索；WorldCover 进行元数据缺失实验；

**📈 对比分析**

对比方法：与 CSMoE 及其他基线（DOFA、Prithvi、TerraMind）在同一六项任务上比较；冻结转移上 MEOX 在 cashew mIoU 64.42% 及 EuroSAT AA 90.56% 里均超过 CSMoE；微调后 BigEarthNet mAP 达 72.95%；检索上 SAR 同传感器 F1 64.41% 也优于对照；总体展示了在极低参数预算下的竞争性性能；

**⚠️ 局限性**

局限性：在更大分辨率（224px）时计算量急剧上升；光学与雷达在跨传感器检索中仍未完全对齐；未验证随机波段丢失鲁棒性、重叠窗口推理等；模型仅支持预定义传感器集合，无法处理未知传感器；

---

## 490. Lightweight Vision Transformer Compression for On-Device Plant Disease Detection in Resource-Constrained Agricultural Field Conditions

**arXiv ID:** 2609.05334 | [PDF](https://arxiv.org/pdf/2609.05334v1)

**作者:** Mahadev Sunil Kumar `[一作]` (Amrita Vishwa Vidyapeetham), G. Gopakumar `[通讯]` (Amrita Vishwa Vidyapeetham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

针对印度辣椒（Capsicum annuum）叶片病害的实时检测，提出了一套统一的 Vision Transformer（ViT）压缩框架，将 H-BAC 结构化裁剪、注意力知识蒸馏和动态 INT8 量化相结合，构建了可在资源受限的移动设备上部署的轻量化模型。

**💡 创新点**

创新点在于：①通过 H-BAC 结合 Hessian 估计实现块级曲率感知的自适应裁剪，显著提升剪枝后恢复性能；②提出基于注意力图的蒸馏方式，克服特征维度不匹配问题；③系统性地将裁剪、蒸馏、量化三项技术串联，形成可配置的压缩管线，并在真实田间多村跨设备数据上验证其泛化能力。

**🔧 技术方法**

使用的核心技术包括：Hessian‑Balanced Adaptive Block Pruning（H‑BAC）+ Hutchinson 估计；Attention‑Based Knowledge Distillation（基于注意力图的蒸馏）；Post‑Training Dynamic INT8 Quantization（PTQ‑Dynamic）；以及标准的 ViT‑B/16 预训练和微调流程。

**📊 数据集**

实验数据集为自采辣椒叶片三类（健康、叶卷病毒初期、叶卷病毒晚期）村庄划分的数据集，共 17,655 张训练图，2,207 张验证图，2,207 张测试图，以及 760 张跨村跨设备 OOD 测试集；所有图像均为现场采集，包含自然光照、遮挡和背景杂乱等真实场景因素。

**📈 对比分析**

与基线 ViT‑B/16（95.13% OOD 准确率）相比，单独 H‑BAC 50% 剪枝可保留 95.53% 准确率并减少 49% FLOPs；PTQ‑Dynamic 约 74% 存储压缩但准确率仅下降 0.79%；Attention‑KD 通过蒸馏将模型压缩至 21.15 MB，准确率 96.71%；完整压缩管线将模型压缩到 6.01 MB（54.5×），平均 95.13% ±2.32% 准确率，满足 OOD 任务需求；相较于仅训练 TinyViT 并量化的直接方案（94.87%），管线的平均性能略有提升，但单次运行可高于或低于该方案。

**⚠️ 局限性**

主要局限包括：①INT8 量化在 Apple M4 Pro CPU 上未能带来显著推理速度提升，仅缩小模型大小；②实验未在目标低端 Android/ARM 智能手机上测评真实延迟；③压缩管线存在较大运行间方差，单次配置可能不稳定；④仅针对辣椒三类病害，泛化到其他作物和病害的效果尚未验证；⑤H‑BAC 与蒸馏的组合在极端剪枝率下仍有 1–2% 的准确率损失。

---

## 491. FIRE-LIVWO: Robust LiDAR-Inertial-Visual-Wheel Odometry via Failure-Immune mmWave Radar Enhancement

**arXiv ID:** 2609.05325 | [PDF](https://arxiv.org/pdf/2609.05325v1)

**作者:** Kun Hu `[一作]` (China University of Mining and Technology), Gongbo Zhou `[通讯]` (China University of Mining and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多模态紧耦合定位框架FIRE‑LIVWO，能够在地下煤矿烟雾、尘埃及几何退化环境中实现稳健的同时定位与建图；

**💡 创新点**

创新点包括：1）统一VoxelMap整合LiDAR、毫米波雷达与视觉特征；2）引入毫米波雷达的点对平面几何残差与多普勒速度约束；3）利用车轮里程计的非完整性约束和在线杠杆臂补偿；4）基于几何与视觉可观测性分析的退化检测与自适应融合模式切换；

**🔧 技术方法**

技术实现基于迭代误差状态Kalman滤波器（IESKF），融合毫米波雷达点云、LiDAR点云、相机图像、IMU及车轮里程计数据，并采用气溶胶散射模型与Hessian特征的可观测性评估；

**📊 数据集**

在真实地下煤矿隧道中使用Husky A200平台搭载Livox AVIA LiDAR、Hikvision摄像头、Oculii Eagle 4D毫米波雷达、IMU及车轮里程计进行实验，使用全站仪标定的地面真实轨迹作为基准；

**📈 对比分析**

与FAST‑LIVO2、R3LIVE、GaRLIO、4DRadarSLAM等现有方法对比，FIRE‑Full在平均定位误差方面达5.677 m，显著优于其他方法（如R3LIVE 31.596 m、4DRadarSLAM 45.453 m），并在烟雾与几何退化双重恶劣条件下实现完整轨迹；

**⚠️ 局限性**

局限性包括：1）系统依赖多传感器同步与标定，硬件成本较高；2）在极度低光或完全无视觉条件下仍需雷达与里程计信息，若雷达失效或车轮打滑仍可能导致漂移；3）可观测性阈值选择需经验调优，可能不适用于所有环境。

---

## 492. Shallow neural network approximation in mixed Sobolev spaces

**arXiv ID:** 2609.05263 | [PDF](https://arxiv.org/pdf/2609.05263v1)

**作者:** Yuwen Li `[一作]` (Zhejiang University), Guozhi Zhang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了在混合 Sobolev 空间中，用浅层神经网络对目标函数的最佳 L₂ 近似的理论界限。

**💡 创新点**

提出了激活函数的 Fourier‑block 属性与一维结构化逼近条件，统一分析不同激活函数的逼近阶数。

**🔧 技术方法**

采用傅里叶块分解、稀疏网格、对数因子控制以及单变量逼近理论实现上界推导。

**📊 数据集**

本文为理论研究，无实验数据集。

**📈 对比分析**

与已有深度网络及稀疏网格理论相比，给出近似误差的最优幂率，并对 ReLU^k 网络证明几乎达到最优。

**⚠️ 局限性**

上界中的对数项能否去除以及更广泛激活函数能否实现完整混合光滑性指数仍是未解问题。

---

## 493. Moral Advice as Interactional Negotiation: Framing, User Pressure, and Social Position in Large Language Model Responses

**arXiv ID:** 2609.05345 | [PDF](https://arxiv.org/pdf/2609.05345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 494. Large Language Models for HVAC Operations in Building Energy Systems: A Critical Review of Methods, Applications, and Deployment Readiness

**arXiv ID:** 2609.05314 | [PDF](https://arxiv.org/pdf/2609.05314v1)

**作者:** Alexander Neubauer `[一作]` (Technische Universität Berlin), Martin Kriegel `[通讯]` (Lawrence Berkeley National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2023‑2026 年 66 篇关于大语言模型（LLM）在暖通空调（HVAC）运营中的论文进行了系统综述，并构建了 A×M（任务-方法）交叉表，评估了证据真实性、部署成熟度和责任边界。

**💡 创新点**

创新点在于首次将任务家族、LLM 适配方法、证据真实性、部署准备度和责任边界三维度统一编码，形成可量化的评估框架，揭示了从 M1（提示/监督适配）到 M2（上下文驱动的协作）再到 M3（多模态输入）的技术演进及其在 HVAC 领域的分布差异。

**🔧 技术方法**

主要采用系统综述方法（PRISMA 过滤流程）、分类编码（A1‑A5 任务，M1‑M3 方法），以及基于文本与代码的多维度评估指标（证据真实性、可部署性、责任边界），并对每篇论文进行跨表统计与可视化。

**📊 数据集**

综述基于公开论文的元数据；涉及的真实数据集包括 ASHRAE RP‑1312、LBNL AHU 数据、能源管理系统（BEMS）日志、能源仿真模型（EnergyPlus、OpenStudio）等；论文中引用的实验/实测数据多数为回溯性现场数据、实验室测试或仿真结果。

**📈 对比分析**

通过对 A×M 交叉表、年份趋势、模型使用（GPT‑4/GPT‑4o 等）以及证据真实性分布的对比，发现：M1 与 M2 研究占比接近，M2 在 2024‑2026 年显著增长；仅有 4 篇 pilot 级部署，未出现 operational 部署；在任务层面，BEM (A3) 与控制 (A4) 的实测证据最少，负责任边界多停留在“LLM‑advisor‑human‑decides”或“LLM‑generate‑human‑review”。

**⚠️ 局限性**

局限性：① 证据多为仿真或回溯性数据，缺乏持续的现场部署验证；② 绝大多数论文依赖专有 GPT‑4/4o，缺乏可迁移的开源模型与可复现代码；③ 在安全治理、推理不一致、延迟与检索错误等方面的失败模式尚未得到系统评估；④ 责任边界和安全约束多为自我声明，缺乏统一的规范与验证机制。

---

## 495. RISE: Recursive Improvement via Self-Extrapolating Policy Distillation

**arXiv ID:** 2609.05295 | [PDF](https://arxiv.org/pdf/2609.05295v1)

**作者:** Yang Li `[一作]` (Salesforce AI Research), Shafiq Joty `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种自我提取教师模型的策略，在RLVR训练轨迹上进行递归外推，从而实现无外部模型的on‑policy distillation；

**💡 创新点**

创新点在于将模型自身的参数/logit变化作为教师，利用RLVR产生的方向来进行线性外推，再用OPD将该教师的token级分布回归到当前模型，从而形成递归改进循环；

**🔧 技术方法**

采用了RLVR（强化学习自我奖励）、OPD（on‑policy distillation）、logit/权重空间外推、EMA锚点、β衰减调度以及top‑K近似；

**📊 数据集**

在数学推理（Dapomath、OpenR1）、多域STEM、代码生成（Skywork‑OR1‑Code）、多轮代理任务（ALFWorld、WebShop）等四大数据集上进行评估；

**📈 对比分析**

与单纯RLVR、GRPO、以及多种基于privileged conditioning的OPS基线（GRPO+SDPO、SDAR、RLSD）对比，RISE在所有模型规模和任务上均取得更高的平均准确率、更快的样本效率，且在OOD任务上保持或提升表现；

**⚠️ 局限性**

局限在于依赖训练轨迹低维线性可外推性，β衰减和EMA锚点需要经验调优，且若奖励信号存在偏差，外推会放大错误方向。

---

## 496. TacPAC: Tactile Prediction and Real-Time Action Correction in World-Action Models for Contact-Rich Manipulation

**arXiv ID:** 2609.05266 | [PDF](https://arxiv.org/pdf/2609.05266v1)

**作者:** Zipei Ma `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在基于世界-动作模型（WAM）的视觉+触觉框架中，提出 TacPAC，通过缓存预测的触觉观测与计划的注意力状态，并在执行过程中使用触觉专家实时校正动作片段，显著提升多种紧密对接任务的成功率。

**💡 创新点**

核心创新点在于：① 将触觉预测与执行纠正结合，克服传统触觉预测在动作已规划后无法更新的时序瓶颈；② 引入缓存机制，让触觉专家在纠正时能参照计划所期望的触觉状态，而非仅对当前触觉读数做反应；③ 设计单通道高效的触觉专家，实现每次纠正仅 30 ms，远低于重新规划的计算成本。

**🔧 技术方法**

技术实现主要依赖：
- 视觉+触觉混合视频编码器与动作专家的混合变压器（MoT）
- 条件流匹配（conditional flow‑matching）训练损失
- 触觉专家（single‑pass corrector）与缓存（per‑layer attention keys/values）
- 触觉专家与计划的异步推理架构
- 触觉感知：InTac S1 触觉传感器 + RealSense 摄像头
- 机器人平台：Flexiv Rizon 4 单臂机械手。

**📊 数据集**

实验使用五个真实机器人任务（充电器插头插入、多物体水果传输、易碎土豆片传输、空瓶子立起、扩展卡插入）作为数据集；未使用公开数据集，全部采集自实验室真实环境。每个任务均进行 20 次试验。

**📈 对比分析**

与六种基线（π_0.5、ACT、VITaL、LingBot‑VA、Dream‑Tac、T‑Rex）对比，TacPAC 在所有任务上均取得最高成功率，五任务平均成功率提升至 64%，比最佳基线 T‑Rex 提升 16 个百分点；在最难的插入任务上，提升幅度高达 15 个百分点。触觉专家每次纠正仅 30.4 ms，计算成本比重新规划低 20.7 倍。

**⚠️ 局限性**

局限性包括：
- 触觉预测仍受限于先前预测的时序误差，无法完全捕捉瞬时接触变化；
- 缓存机制假设计划在整个片段内可被重新利用，若执行偏差过大可能导致误差累积；
- 仅在单臂 Flexiv Rizon 4 上验证，跨平台通用性未充分评估；
- 触觉传感器的分辨率与鲁棒性仍受硬件限制，极端环境下性能可能下降。

---

## 497. LLM-Driven Algorithm Design for Quantum Circuit Synthesis based on Binary Decision Diagrams

**arXiv ID:** 2609.05327 | [PDF](https://arxiv.org/pdf/2609.05327v1)

**作者:** Yoonju Sim `[一作]` (KAIST), Changhyun Kwon `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种利用大型语言模型驱动的进化框架，自动设计适用于可逆电路合成的BDDs变量排序启发式，从而直接降低量子电路成本。

**💡 创新点**

创新点在于：①首次将LLM作为启发式生成器，引入QCC（量子成本）作为评价目标；②设计了HGA‑QE——一种混合遗传算法，将sifting替换为LLM发现的针对性sifting，显著提升QCC；③通过多家族种子初始化和QCC‑aware fitness，突破传统BDD大小优化的局限。

**🔧 技术方法**

技术上结合了CUDD BDD库、ReVKit可逆电路合成、LLM（OpenAI GPT‑OSS‑120B）、遗传算法、sifting/模拟退火等。

**📊 数据集**

实验使用RevLib、LGSynth91以及ISCAS85/89三大公开基准集，共148条函数，包含从这些数据集扩展的变体。

**📈 对比分析**

与经典sifting、GA、SA以及学习基线BDD2Seq进行对比，HGA‑QE在全部基准上实现70.9% tie‑or‑win，13.5% strict‑win；在20个函数上独占最低QCC，平均QCC仅比最佳基线低约1.7%。

**⚠️ 局限性**

局限性包括：①搜索仍受初始启发式家族的影响，难以发现完全不同的排序策略；②仅优化BDD排序，未结合ancilla/垃圾输出等后处理；③LLM推理成本高，非LLM搜索在某些指标上仍具竞争力，提示可进一步优化LLM使用。

---

## 498. How Does mHC Use Its Residual Streams? Selective Routing and Near-Identity Mixing

**arXiv ID:** 2609.05309 | [PDF](https://arxiv.org/pdf/2609.05309v1)

**作者:** Pengxiang Zhao `[一作]` (Huawei Technologies Co.), Zhenhua Dong `[通讯]` (Huawei Technologies Co.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 DeepSeek-V4-Flash 四流残差通道的真实前向（true‑forward）测量，系统分析了读写路由、残差混合以及流间表示的深度分布，并在推理阶段通过路由稀疏化与残差混合器替换两种干预验证这些模式对模型质量的功能性影响。

**💡 创新点**

创新点在于：①提出了深度分辨率的多流路由与残差混合诊断框架；②首次使用推理时干预（路由保留顶 k 权重、残差混合器替换为恒等）直接量化多流结构的功能重要性；③发现早期层残差混合器对性能贡献大，而后期层几乎可用恒等替代，揭示了多流宽度与实际利用率的差异。

**🔧 技术方法**

使用的技术包括：true‑forward instrumentation、token‑级读写权重归一化、有效流计数与赢家一致性度量、残差混合器的身份偏差指标、基于 Sinkhorn‑Knopp 的双随机矩阵约束、以及 top‑k 路由保留和残差混合器替换的推理时干预。

**📊 数据集**

主要使用的数据集是英文 C4（训练 32T 令牌）用于诊断与 perplexity 评估，外加六个零样本下游任务（ARC‑Easy、ARC‑Challenge、PIQA、HellaSwag、MMLU、GSM8K）用于综合性能测评。

**📈 对比分析**

与基线模型（PPL 11.1587、六任务平均 84.22%）对比，路由保留顶 3 权重时 perplexity 仅上升 2.4–2.7%，六任务平均仅下降 0.38%；保留顶 2 权重时 perplexity 上升 12–14%，平均下降 2.9–6.6%；将第 22–42 层残差混合器替换为恒等时 perplexity 仅上升 1.9%，平均不变；而替换第 0–21 层混合器时 perplexity 直升 41%，平均下降 3.3%。

**⚠️ 局限性**

局限性包括：仅在单一 284B 检查点上评估，缺乏训练时路由与混合器动态的可视化；干预仅针对推理阶段，未验证对梯度传播的长期影响；所用指标（PPL、六任务平均）可能无法全面捕捉模型对多流结构的功能依赖；此外，实验未涵盖其他多流设计的对比。

---

## 499. What Makes a Redundant Representation Remember? Lineage Isolation, Not Masking

**arXiv ID:** 2609.05304 | [PDF](https://arxiv.org/pdf/2609.05304v1)

**作者:** Jia Huang `[一作]` (Peking University), Yangjun Ou `[通讯]` (Peking University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了双拷贝遗传算法在动态优化中的隐式记忆机制，揭示了遗传规则（继承）对记忆保留的决定性作用，并通过对比周期性与单向漂移环境评估记忆效应。

**💡 创新点**

①将双拷贝表述为“gate + inheritance”两维设计轴，首次将两者分离并系统评估；②发现继承规则而非门控决定记忆保留；③提出门控翻转率既是读取速率又是腐蚀速率，存在内部最优点；④证明门控冗余不是编码冗余，无法提升噪声鲁棒性。

**🔧 技术方法**

基于标准XOR式动态基准的遗传算法（GDC-EA），实现了硬门/泄漏门两种门控规则与两种继承规则（独立遗传/隔离线性），使用AUC作为性能度量并测量保留信息的比特数。

**📊 数据集**

使用L=100、d=50、N=200、μ=0.002等参数的XOR式动态基准，共计18个周期，首次四个周期剔除后进行实验。

**📈 对比分析**

将双拷贝算法与单拷贝基线（同等表示预算、相同突变率、选择规则与交叉算子）对比，AUC在周期性环境下略有提升(+0.010)，在单向漂移环境下明显下降(-0.078)，二者差距为0.089；门控翻转率在0.05处达到AUC最大，保留信息最高在约0.02处；实验展示了基线交叉算子选择对结果的显著影响（0.062 AUC差异）。

**⚠️ 局限性**

①保留信息估计在二元符号下存在偏差；②仅测试了两种环境状态，未评估更复杂基准；③未探索不同种群规模、基因多样性与多重状态下的扩展；④双拷贝的均匀冗余导致大部分位点无实质记忆，增加了携带成本。

---

## 500. Learning Spatial-Spectral Refinement and Calibrating Complementary Observations for Hyperspectral Image Super-Resolution

**arXiv ID:** 2609.05303 | [PDF](https://arxiv.org/pdf/2609.05303v1)

**作者:** Liqian Yang `[一作]` (Zhengzhou University), Qianxin Yi `[通讯]` (Zhengzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种自监督的两阶段框架（TSR-ITNR）来完成低分辨率高光谱图像（LR-HSI）与高分辨率多光谱图像（HR-MSI）的融合，最终生成高分辨率高光谱图像（HR-HSI），无需任何HR-HSI配对训练数据。

**💡 创新点**

创新点包括：①将隐式张量神经表示（implicit tensor neural representation）与Tucker分解相结合，既保留了高阶结构又实现了紧凑的低秩表示；②设计了几何保持的注意力引导光谱细化器（AGPSR），在不改变光谱子空间几何的前提下提升光谱相关性；③提出了无参数的互补观测校正（COGC），通过正交补充校正充分利用LR-HSI与HR-MSI的互补信息，并保证最小改动。

**🔧 技术方法**

技术手段包括：隐式张量神经网络（hash encoder + GFINER）、多尺度空间系数细化（MSCR）、注意力引导几何保持光谱细化（AGPSR）、无参数互补观测校正（COGC）、PnP-HQS优化与预训练深度去噪器（DRUNet）相结合的自监督训练。

**📊 数据集**

实验使用了三大公开数据集：Pavia（Pavia Centre）、CAVE（Balloons、Peppers、Lemons、Sponges、Clay）和ICVL（BGU、Flower、Eve、Hill、Nachal），通过模拟降采样与光谱响应获得LR-HSI与HR-MSI。

**📈 对比分析**

与10种基线方法（低秩、深度学习及混合方法）在PSNR、SSIM、SAM、ERGAS等指标上比较，TSR-ITNR在绝大多数数据集与指标上均获得最高或第二高分；在下游语义分割任务中亦显著提升mIoU和宏F1，接近真实HR-HSI的性能。

**⚠️ 局限性**

局限性：实验均基于合成降采样，缺乏真实传感器噪声与复杂光谱响应的验证；模型参数调优仍需经验，计算量较大；对不同尺度因子和极端光谱条件的鲁棒性尚未充分评估。

---

## 501. Human-Human & Human-Robot Interaction Transformer (H2INT) for Robot Navigation in Dense and Uncertain Crowds

**arXiv ID:** 2609.05300 | [PDF](https://arxiv.org/pdf/2609.05300v1)

**作者:** Ao Shen `[一作]` (Beijing Institute of Technology), Chen Chen `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于双层门控Transformer的强化学习框架H^2INT，用于在密集且不确定的人群中实现机器人导航。

**💡 创新点**

创新点在于将人对机器人的响应行为建模为行为层面且保持隐藏状态，并通过两阶段门控Transformer逐步细化人-人与人-机的交互特征，同时引入响应性退火课程提升鲁棒性。

**🔧 技术方法**

使用双层门控Transformer编码器、GRU递归策略核心、PPO强化学习、ORCA仿真人群、LiDAR感知、梯度残差门控等技术。

**📊 数据集**

主要在Circle Crossing、Intersection、Split Flow、Random Wander等四种模拟布局上训练和评估，并在真实室内演示场景中部署。

**📈 对比分析**

与经典规则方法(ORCA、Social Force)、学习方法(CADRL、SARL、DS-RNN、AIG-GST)对比，H^2INT在不同响应率和人群密度下均保持最高成功率、最低碰撞率，且在未见布局与真实部署中表现出色。

**⚠️ 局限性**

局限性包括对人类响应概率的近似假设、缺乏对更丰富感知模态（速度、视线、身体姿态）的利用，以及在极端密度或快速运动场景下可能的计算瓶颈。

---

## 502. Beyond Aggregate Scores: Behavioral Correctness Assumptions for Assessing Reference-Based Automatic Evaluation Methods

**arXiv ID:** 2609.05289 | [PDF](https://arxiv.org/pdf/2609.05289v1)

**作者:** Maria Mahbub `[一作]` (Oak Ridge National Laboratory), Amir Sadovnik `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套基于行为正确性假设的诊断框架，用受控文本变换评估参考基评测方法的行为。

**💡 创新点**

创新点在于将正确性保持与改变两类假设与对应受控变换相结合，形成细粒度测试套件，系统揭示评测方法的稳定性与敏感性。

**🔧 技术方法**

采用受控变换、LLM生成基线、以及词法、字符、语义、混合与LLM评测方法，计算变换级、假设级和总结级指标。

**📊 数据集**

使用九份美国海岸警卫队文档生成203个问答对，并基于这些答案生成2842个受控变换实例。

**📈 对比分析**

与传统整体相关度或人类判定的基准相比，实验显示不同评测方法即便整体性能相近，其行为特征差异显著，且无任何方法同时满足所有正确性假设。

**⚠️ 局限性**

局限包括：依赖LLM生成的基线与变换，假设体系不完整，实验仅针对单一参考答案，且框架在其他生成任务中的适用性待进一步验证。

---

## 503. GUT: Quantifying and Optimizing the Reasoning Uncertainty of LLMs via Graph Complexity

**arXiv ID:** 2609.05284 | [PDF](https://arxiv.org/pdf/2609.05284v1)

**作者:** Shuang Liang `[一作]` (Nanjing University), Shao-Qun Zhang `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于有向无环图的LLM推理不确定性量化与优化框架GUT，包含GUT‑Q和GUT‑O模块；

**💡 创新点**

创新点在于将推理可能分支建模为DAG并用图复杂度近似推理空间复杂度，同时将负不确定性作为强化学习奖励来优化推理；

**🔧 技术方法**

使用多次采样、自然语言推理（NLI）进行节点合并，计算节点不确定性（四种token级别指标），构造图宽度/高度/不确定性传播三种图复杂度估计方法，并采用GRPO进行MTLP优化；

**📊 数据集**

在四个不同规模的Qwen3 LLM（0.6B–8B）上，结合五个公开数据集（GSM8K、MATH‑500、AMC2022‑2024、FOLIO、MMLU‑Pro）进行实验；

**📈 对比分析**

与45种现有UQ方法对比，GUT‑Q‑UP在PRR、AUROC、AUPRC上平均提升约11.8%、13.3%、9.7%；GUT‑O在四个模型上平均降低1.93%误差、提高13.9%平均不确定性；

**⚠️ 局限性**

局限在于节点合并依赖NLI模型的阈值选择，计算复杂度受样本数和温度影响；此外优化仍需手工选择不确定性代理（如MTLP），未对其他代理做广泛验证。

---

## 504. Adaptive Gated Deepfake Detection for Low-Resolution and Resource-Constrained Environments

**arXiv ID:** 2609.05320 | [PDF](https://arxiv.org/pdf/2609.05320v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 505. Temporal Tactile Encoding and Compliance for Intent-Aware Robot-to-Human Bimanual Handover

**arXiv ID:** 2609.05282 | [PDF](https://arxiv.org/pdf/2609.05282v1)

**作者:** Pasquale Marra `[一作]`, Lorenzo Natale `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了一个多模态学习式机器人对人类交付的抓取包裹手动交接策略，融合视觉、触觉历史和顺应控制。

**💡 创新点**

创新点在于①采用时间编码的触觉（TTE）取代瞬时触觉，显著提升对持续取用意图的判定；②在学习策略下引入低层顺应控制，实现物理平滑交接；③对两者进行消融实验验证其互补性。

**🔧 技术方法**

使用了NVIDIA Isaac GR00T-N1.5-3B视觉‑语言‑动作模型、Xela指尖触觉传感器、时间编码自编码器、二阶逆运动学+顺应控制、Meta Quest 3 远程操作等技术。

**📊 数据集**

通过Meta Quest 3收集的8名参与者遥控演示（约50k帧，包含5类交接模式），随后在10名新参与者上测试三种配置。

**📈 对比分析**

采用客观指标（成功率、释放延迟、峰值拉力）和主观问卷进行比较。全系统（TTE+顺应）在93%成功率、1.70s释放延迟、4.07N峰值拉力等指标上优于无触觉或无顺应基线，并被10/10受试者视为最安全、最满意。

**⚠️ 局限性**

局限性包括：对单一盒子物体的泛化有限；对极短突发拉力的鲁棒性仍不足；系统依赖同步高频触觉传感和低层顺应控制，部署复杂；未解决模糊取用意图的恢复策略。

---

## 506. KanAdapter: A Kolmogorov-Arnold Network-based Plug-and-Play Module for Efficient Fine-tuning of Foundation Speech Models

**arXiv ID:** 2609.05281 | [PDF](https://arxiv.org/pdf/2609.05281v1)

**作者:** Phuong Tuan Dat `[一作]` (National University of Singapore), Tran Huy Dat `[通讯]` (A Star)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了KanAdapter，一种基于GR‑KAN的轻量级适配器，用于对语音自监督学习模型进行参数高效微调。

**💡 创新点**

首创将KAN网络（GR‑KAN）作为适配器的非线性模块，替代传统MLP瓶颈，显著提升表达能力并增强连续学习鲁棒性。

**🔧 技术方法**

使用GR‑KAN模块、并行瓶颈设计、预训练MLP权重迁移、方差保持初始化以及分组有理激活等技术。

**📊 数据集**

使用VoxCeleb2/1进行说话人验证、MSP‑Podcast进行情感识别、ASVspoof2019/2021 LA与DF以及ASVspoof5和In‑The‑Wild进行深伪造检测。

**📈 对比分析**

与全微调、LoRA和AdaptFormer比较，KanAdapter在SV、SER、DFD以及连续学习任务中，参数量减少至97.5%，同时EER/F1等指标保持与全微调相近，且在连续学习中EER下降多达83.6%。

**⚠️ 局限性**

局限在于对大规模深层SSL骨干最有效，浅层模型收益有限；仅评估分类任务，未验证生成任务；未深入分析有理激活机制；连续学习测试场景有限。

---

## 507. Self-Supervised Lexical Representation Learning for Fast, Large-Scale Phylogenetic Inference

**arXiv ID:** 2609.05262 | [PDF](https://arxiv.org/pdf/2609.05262v1)

**作者:** Tim Wientzek `[一作]` `[通讯]`, Tim Wientzek

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种完全自监督的对比学习框架 DualCWE，用于从原始 IPA 词表直接学习词汇表示并推断全球语言亲缘关系与概念稳定性；

**💡 创新点**

不需要手工标注同源关系、对齐或专家输入，仅通过词级与语言级对比损失实现自我监督；

**🔧 技术方法**

使用 Transformer 编码器、语音特征向量（39 维）、对比损失（NT‑Xent）以及语言级辅助对比目标；

**📊 数据集**

训练集为 185 个 Lexibank 数据集（共 3399 种语言、210 个核心词汇概念），数据为 IPA 转录；

**📈 对比分析**

与基线（最大似然 MSA、Levenshtein、pHMM 等）在 3,399 语言上比较，GQD 0.0346（低于 0.0433/0.0468），推断时间仅数分钟；

**⚠️ 局限性**

模型仍保留地理相似性，未区分继承与借音；对语言级表示评估有限；对小语族或单家族的细粒度效果尚未验证；

---

## 508. Development of a Humanoid Robot Prototype for Multimodal Human-Robot Interaction

**arXiv ID:** 2609.05361 | [PDF](https://arxiv.org/pdf/2609.05361v1)

**作者:** Thang Tran Viet `[一作]` (University of Engineering and Technology), Xiem HoangVan `[通讯]` (University of Engineering and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了一款低成本、可扩展的人形机器人原型，用于多模态人机交互实验。

**💡 创新点**

创新点在于将12自由度双臂、2自由度头部与LCD面板相结合，并在同一平台上集成姿态识别、目标检测与语音指令理解三个AI模块，实现实时、可复制的交互能力。

**🔧 技术方法**

使用技术包括ROS+Jetson AGX Xavier进行计算，MediaPipe Pose+LSTM进行手势识别，YOLOv8与ZED2双目相机实现3D目标定位，语音识别+Gemini LLM进行语义解析，以及自制的电路板与Arduino控制器实现机械驱动。

**📊 数据集**

数据集方面，手势识别采用自制的多姿态手势序列数据集，目标检测使用预训练YOLO模型并在COCO数据集基础上微调，语音识别和LLM输入基于实验室收集的中文口语指令。

**📈 对比分析**

实验结果表明，手势识别准确率达96%，语音转写准确率92%，语义解析准确率96%，目标检测准确率90%；机械操作的平均定位误差约1.83 cm，整体pick‑and‑place成功率约90%，与现有闭源高端平台相比，已在同等任务上实现了较高的精度与低成本。

**⚠️ 局限性**

主要局限在于对外部大型语言模型API的依赖，导致网络延迟和实时性受限，未来需部署轻量化本地模型以提升响应速度。

---

## 509. Trust-Aware Adaptive Disclosure for Inference Privacy Preservation in Multi-Agent Networks

**arXiv ID:** 2609.05340 | [PDF](https://arxiv.org/pdf/2609.05340v1)

**作者:** Puspanjali Ghoshal `[一作]` (R. C. Bose Centre for Cryptology and Security), Tobias J. Oechtering `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种基于信任的隐私控制框架TAPC，用以在多智能体协同任务中抵御目标推断攻击。

**💡 创新点**

创新点在于将动态信任度作为信息披露的控制变量，采用可调的信任衰减系数实现对消息的概率性遮蔽，而非传统的统一噪声或硬阈值过滤。

**🔧 技术方法**

主要技术包括信任评估模型（可靠性与一致性评分融合）、基于信任的披露系数、Gaussian噪声遮蔽、互信息上界推导与随机森林目标分类评估。

**📊 数据集**

使用的是基于随机几何图的50节点网络，每个智能体的隐性目标从大小为6的离散集合中随机赋值。

**📈 对比分析**

与全披露FD、统一噪声GN、信任阈值TOC三种基线比较，TAPC在保持约0.92的共识效用的同时，将泄漏率降至0.137，优于其他方法且通信成本相同。

**⚠️ 局限性**

局限性在于仅考虑瞬时通信且未建模时间相关性、仅使用高斯噪声、未扩展至有向或时变图、理论上限过于简化。

---

## 510. Does Your Agent's Memory Survive a Model Upgrade? A Controlled Study of Memory Portability

**arXiv ID:** 2609.05339 | [PDF](https://arxiv.org/pdf/2609.05339v1)

**作者:** Ankit Goyal `[一作]`, Jaideep Ray `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过实验对比了四种常见的记忆格式（长上下文原始记录、检索增强生成、压缩笔记和固定结构知识图）在模型升级时的迁移性能。

**💡 创新点**

创新点在于提出了记忆迁移的可靠性评估指标（RPAS 与 CTR），系统化展示了不同迁移方向和嵌入空间兼容性对记忆可用性的影响，并揭示压缩笔记在写入方向上极不对称的性能损失。

**🔧 技术方法**

使用了开放权重的 Llama‑3.1‑8B 与 Qwen2.5‑7B 两模型，搭配嵌入模型升级、RAG 检索、自然语言笔记压缩与固定结构知识图存储，并通过统计检验与重建实验进行评估。

**📊 数据集**

实验基于 48 条自定义脚本化历史，每条包含 160 题随机答案，确保零预训练知识干扰。

**📈 对比分析**

通过四类迁移测试（模型写读迁移、嵌入混合索引、存储修复、损失诊断）对比，结果显示 KG‑fixed 迁移最稳健；RAG 混合索引仅提升约 4.96pp，完整重嵌入可提升 11.90pp；压缩笔记在不同写读方向上对称损失达 ±13pp；仅保留原始历史的修复可使笔记恢复 90% 以上性能，若不保留则无法达标。

**⚠️ 局限性**

研究仅限于两模型、同维度嵌入、单阶段检索、脚本化历史，未涵盖更大模型、自然对话、复杂检索或隐私合规等实际场景，结果的通用性受限。

---

## 511. Technical Manual for a Toolkit for Measuring Contextual Individuation in Transformer Language Models

**arXiv ID:** 2609.05333 | [PDF](https://arxiv.org/pdf/2609.05333v1)

**作者:** José Luciano Verçosa Marques `[一作]` (University of Campinas), Tárcio André dos Santos Barros `[通讯]` (University of Campinas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在处理大规模数据时显著减少计算时间。

**🔧 技术方法**

使用了深度学习和强化学习相结合的技术。

**📊 数据集**

采用了公开的图像识别数据集进行实验。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定类型的数据上表现不佳，且对计算资源的需求较高。

---

## 512. Machine Unlearning as Private Retroactive Algorithms

**arXiv ID:** 2609.05329 | [PDF](https://arxiv.org/pdf/2609.05329v1)

**作者:** Haim Kaplan `[一作]` (Tel Aviv University), Uri Stemmer `[通讯]` (Tel Aviv University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并研究了“私有回溯算法”（Private Retroactive Algorithms）的概念，旨在在保持差分隐私的同时实现对过去操作的可逆修改，从而满足机器无学习的精确重训练要求。

**💡 创新点**

创新点在于将传统的机器无学习视为数据维护问题，首次定义了结合回溯性和差分隐私的算法框架，并给出了多种通用编译器和特定算法（线性查询、聚类、直方图等），证明在许多情形下回溯性几乎不增加误差，但在计数 distinct 等任务中存在不可避免的成本。

**🔧 技术方法**

核心技术包括基于二叉树的树机制（Tree Mechanism）实现无记忆噪声，离散化的后处理与随机采样，隐私分析中使用的连续观测下的差分隐私和 Gaussian 噪声的组合，以及对窗口功能的样本抽样与隐私放大。

**📊 数据集**

论文主要在理论模型下验证，未使用具体真实数据集；所有实验和证明均在抽象的多集合更新序列和数值查询上完成。

**📈 对比分析**

通过理论证明与对比，线性统计、k-means/k-median、稀疏直方图等任务的误差在加入回溯性后保持与原始差分隐私算法相同的上界；相反，在计数 distinct 任务中，加入回溯性迫使误差上升到与最坏情况一致的 Θ(T^{1/4}) 级别。

**⚠️ 局限性**

限制主要在于：回溯性对某些可适应误差机制（如基于“翻转度”自适应噪声的计数 distinct）产生不可避免的性能损失；此外，证明依赖于抽象模型，实际实现与复杂数据流的兼容性仍待进一步实验验证。

---

## 513. Optimal Rates for Agentic Networked Information Aggregation

**arXiv ID:** 2609.05318 | [PDF](https://arxiv.org/pdf/2609.05318v1)

**作者:** MohammadHossein Bateni `[一作]` (Google Research), Shayan Taherijam `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了网络学习模型中的信息聚合，特别是在每个代理只能看到部分数据并传递自己的结论的情况下，如何在深度为D的路径上实现线性回归问题的最小均方误差（MSE）损失。

**💡 创新点**

通过改进之前的界限，填补了上界O(M/√(D))和下界Ω(M/D)之间的差距，提出了在深度D≥M²时的下界Ω(M²/D)，并证明了在分类问题中也能达到相同的最优速率。

**🔧 技术方法**

使用了线性回归和逻辑回归模型，分析了在有向无环图（DAG）中代理的学习过程，采用了均方误差（MSE）和二元交叉熵（BCE）作为损失函数。

**📊 数据集**

使用了高斯分布的特征和标签数据集，特别是在回归和分类问题中，构造了相应的实例以验证理论结果。

**📈 对比分析**

与之前的方法进行了比较，证明了在深度D≥M²时，模型的过度误差为Ω(M²/D)，而在D<M²时，过度误差为常数。通过构造实例展示了这些界限的紧密性。

**⚠️ 局限性**

限制在于所构造的实例依赖于特定的深度D，无法通过固定的分布在所有深度上实现M²/D的下界，表明在不同深度下，信息聚合的效率会有所不同。

---

## 514. LexFlip: A Dissociation Diagnostic for Legal Meaning Preservation Metrics

**arXiv ID:** 2609.05296 | [PDF](https://arxiv.org/pdf/2609.05296v1)

**作者:** Gaurab Baral `[一作]` `[通讯]`, Gaurab Baral

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 LexFlip 诊断集，用来检测法律文本简化是否保留法律意义，并对多种自动化评估指标在该诊断集以及 FrJudge 评估集上的表现进行系统比较。

**💡 创新点**

创新点包括：① 设计了“解耦”测试（LexFlip）——在保持句子表面词汇几乎不变的前提下改变法律意义，突破传统“同一句得最高、非同句得最低”检验的局限；② 提出了四项评估准则（R1‑R4），强调指标必须同时具备人类上限、监督一致性、表面特征控制以及解耦检验；③ 将上述准则与开源工具、公开数据集结合，形成可复现的评估流程。

**🔧 技术方法**

使用的技术包括：句子嵌入（LaBSE、mE5、CamemBERT）、BERTScore（CamemBERTv2/FlauBERT 变体）、二向 NLI（mDeBERTa‑XNLI）以及 Prompted LLM（DeepSeek‑Chat）评估；还利用了表面特征回归（词长差、Token Jaccard）作为基准。

**📊 数据集**

数据集：① Quebec 立法文本（《汽车保险法》和《公路安全法》）的 400 条未修改条款以及 373 条经过 LexFlip 改动的对偶条款；② FrJudge 保险条款简化数据集，包含 297 条条款及五位法律学生的 10 分等级评分。

**📈 对比分析**

比较方法：在每个评估准则下计算指标的相关性、均方误差、以及在 LexFlip 上的 margin（相同‑相似与相同‑无关的分数范围比例）。结果显示：所有基于相似度的指标（BERTScore、句子嵌入）在 LexFlip 上的 margin 接近 0，几乎不区分法律意义；二向 NLI 的 margin 最高（≈0.67），表明它能识别法律意义变动；而单纯表面特征（词长差）在相关性上最高（≈0.64）但 margin 极低。整体来看，指标往往在相关性上达不到人类上限，而在解耦检验上表现欠佳。

**⚠️ 局限性**

局限性：① LexFlip 改动仅基于定义式替换（如 doit→peut），未进行人工确认，可能遗漏更细微或更复杂的法律意义变化；② 只测试单一模板改动，系统可能过度拟合这些改动，缺乏泛化性；③ FrJudge 标注一致性低，导致人类上限估计不稳定；④ 评估集与诊断集来源不同（保险条款 vs. 法律条文），结果可能受文本域影响，不能直接推断在所有法律文本上的性能。

---

## 515. Augur: Predicting View Serializability Violations in Relational Data Store Applications

**arXiv ID:** 2609.05288 | [PDF](https://arxiv.org/pdf/2609.05288v1)

**作者:** Chujun Geng `[一作]` (Ohio State University), Yang Wang `[通讯]` (Ohio State University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种动态预测分析方法，能够在观察到可序列化执行的基础上推断出可实现的不可序列化执行。

**💡 创新点**

创新点在于：①精确建模带谓词的SQL查询，并在SMT约束中体现；②仅报告真正违反视图可序列化的执行，而非泛化的历史；③在数据存储层面保持数据存储无关性并保留应用层依赖。

**🔧 技术方法**

技术采用了SQL解析、符号化约束生成、Z3 SMT求解器，以及迭代的视图可序列化检查（CEGIS）。

**📊 数据集**

使用的基准数据集包括OLTP‑Bench的多个程序（Smallbank、Voter、TPC‑C、Wikipedia）和真实电商应用Spree的五个业务场景。

**📈 对比分析**

与IsoPredict的比较表明本方法在同一实验中能够消除大部分误报并发现更多可序列化违规；性能上在大多数案例仅需1–2次迭代，SMT求解时间从秒到数小时不等，整体优于IsoPredict。

**⚠️ 局限性**

局限性包括：对查询的SQL语法支持有限（多表JOIN、子查询等不被解析），SMT求解仍为NP‑hard，导致在事务数超过数十个或语句数千条时求解时间过长；此外对只读工作负载的预测效果有限。

---

## 516. Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference

**arXiv ID:** 2609.05275 | [PDF](https://arxiv.org/pdf/2609.05275v1)

**作者:** Mostafa Elhoushi `[一作]` (Cerebras Systems), Joel Hestness `[通讯]` (Cerebras Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大规模语言模型预训练中使用层 dropout（stochastic depth）的方法，并系统评估了其在训练效率、验证精度和推理灵活性方面的影响

**💡 创新点**

提出了基于层级分布递增和时间递减的层 dropout 配置，证明其在大模型上可保持甚至提升精度，同时显著减少训练 FLOPs，并实现了零样本推理弹性和推理加速

**🔧 技术方法**

使用了结构化层 dropout、不同分布与时间调度、优化器超参数迁移、批量/序列粒度实验以及后训练的早退出适配器和自我投机解码等技术

**📊 数据集**

在多规模（271M-8.2B）模型上使用高达160B个 token 的多元自然语言与代码语料，覆盖超过2400次实验

**📈 对比分析**

通过与密集模型对比，发现最佳配置（ILD+DTS）在相同训练 FLOPs 下可获得相近或更低的验证损失，训练 FLOPs 节省约25%，推理时可实现1.5×的速度提升（自我投机解码）且精度损失可忽略不计

**⚠️ 局限性**

局限性包括：高 dropout 率下超参数迁移不佳、仅研究 transformer 层级 dropout 可能缺乏更细粒度的鲁棒性、未与基于学习的深度优化方法做对比、未探究混合专家或非 transformer 结构的适用性、缺乏针对最大 dropout 率的预测公式以及对不同推理优化的量化不足

---

## 517. One Word, Different Action: A Real-Robot Benchmark for Language-Conditioned Embodied Reasoning

**arXiv ID:** 2609.05260 | [PDF](https://arxiv.org/pdf/2609.05260v1)

**作者:** Yiwei Liu `[一作]` (Chinese University of Hong Kong), Shunbo Lei `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究机器人在指令变化时如何保持或更新动作，构建可执行动作决策基准；

**💡 创新点**

创新点在于提出One Word, Different Action基准，采用任务保持与任务改变配对评估决策不变性与敏感性，并扩展至多约束情境；

**🔧 技术方法**

使用语言条件化的可执行动作决策框架，结合结构化状态、可执行动作空间与多模态视觉输入；

**📊 数据集**

基于85个真实机器人决策锚点，包含单约束与多约束任务，形成完整的任务家族；

**📈 对比分析**

与Qwen3.5‑27B、GLM‑5‑FP8、Gemma4‑31B等大型模型进行对比，单约束性能已接近饱和，而多约束场景下模型仍出现明显退化；

**⚠️ 局限性**

主要限制在于多约束整合能力不足，且验证仅在受控真实视觉条件下完成，未覆盖更复杂或动态的真实环境场景。

---

## 518. Closing Gaps in Online Fair Division

**arXiv ID:** 2609.05310 | [PDF](https://arxiv.org/pdf/2609.05310v1)

**作者:** Tzeh Yuan Neoh `[一作]` (Harvard University), Nicholas Teh `[通讯]` (University of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了无预测或预测有限的在线公平分配离散物品问题，证明了在完全在线模型下几乎所有公平性近似目标都不可实现，提出了利用最大物品价值预测和随机化的新的在线算法，并给出了关于LIKE规则的紧凑高概率公平性和禀赋不平衡度分析。

**💡 创新点**

创新点包括：① 在无预测下证明任意正PROPk近似都不可行；② 用最大物品价值预测实现常数级PROP1近似（1/2或n/(n+κ)），并在已知物品数的情形下同时获得β-PROP1与O(√(Tlog n/n))的最大加法厌恶上界；③ 对LIKE规则在非自适应对手下给出Θ(min{1,n/κlog(n/δ)})的高概率PROP1上界并证明其最优；④ 统一使用递归潜能、倒数潜能、指数潜能和贝塞尔-伯恩斯坦不等式等技术实现多维目标。

**🔧 技术方法**

主要技术手段包括：通过两位代理人归约、归一化松弛与潜能更新；递归和指数势能结合实现PROP1与厌恶双重保证；使用一侧误差转换、伯恩斯坦不等式和Paley–Zygmund不等式进行概率分析；对LIKE规则的随机分配概率进行细致的方差与期望计算。

**📊 数据集**

本工作为纯理论分析，未使用任何具体实验数据集；所有结果均通过构造对手实例和严谨的数学证明获得。

**📈 对比分析**

方法与已有工作比较：在没有预测的情形下与过去只给出下界的结果相比，给出了完全不可实现的证明；利用最大物品价值预测的算法在PROP1方面从先前的1/n提升到1/2；对LIKE规则的高概率分析与之前的Uniform随机分配相比，显著提升了在稀疏偏好（κ较小）下的公平性；同时给出了对应的厌恶上界，揭示了两种随机规则的优势与劣势。

**⚠️ 局限性**

局限性：在完美预测下仍无法证明比1/2更高的PROP1常数；在已知物品数的情形下，最大加法厌恶的上界与下界之间仍存在对数因子；仅针对非自适应对手给出LIKE规则的最优结果，对适应性对手的随机算法提升尚未解决；实验验证缺失，实际应用中的噪声与动态偏好未被考虑。

---

## 519. Online Change-point Detection for Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2609.05298 | [PDF](https://arxiv.org/pdf/2609.05298v1)

**作者:** Fatemeh Saberi Khomami `[一作]` (University of Saskatchewan), Julita Vassileva `[通讯]` (University of Saskatchewan)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种轻量级、算法无关的在线变点检测器PPR，用来监控协作多智能体强化学习过程中的奖励变化。

**💡 创新点**

创新点在于将简单移动平均、指数移动方差与滑动窗口KS检验相结合，形成一条完整的奖励流预处理与漂移检测流水线，能够在不修改学习算法的前提下实现可靠的变点识别。

**🔧 技术方法**

技术细节包括SMA平滑、EMV方差突变提取以及KSWIN统计漂移检测，参数设置为窗口大小w=8、β=2/3、显著性水平α=2e-4、参考窗口W=25、检测窗口m=10。

**📊 数据集**

使用自定义的Speaker‑Listener通信任务（基于Multi‑Agent Particle Environment）作为实验环境，并在其中设计了颜色映射变更和奖励函数变更两种受控非平稳场景。

**📈 对比分析**

与SMA+KSWIN以及原始奖励+KSWIN三种变体对比，PPR在检测延迟与误报/重复报警之间取得了平衡：检测延迟略高于SMA+KSWIN，但误报率和重复报警显著降低；原始奖励方案既迟缓又易漏检。

**⚠️ 局限性**

局限性包括对训练种子敏感，特别是在奖励函数变更场景下检测结果波动大；此外仅利用奖励序列作为监测信号，未考虑其他可能更稳健的状态或行为特征。

---

## 520. Learning from VAE Errors to support ECG-based Differential Diagnosis of Myocardial Scar

**arXiv ID:** 2609.05294 | [PDF](https://arxiv.org/pdf/2609.05294v1)

**作者:** Shayan Sharifi `[一作]` (University of Trieste), Giulia Cisotto `[通讯]` (University of Trieste)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本研究比较了预训练的ECGx.AI β‑VAE与自训练的浅层β‑VAE提取的潜在特征以及基于DTW的重建误差，以评估它们在300名DCM/NDLVC患者中预测心肌瘢痕（LGE+ vs LGE-）的能力。

**💡 创新点**

首次在小规模局部心肌病队列中将DTW重建误差作为可解释的瘢痕指示器，并证明仅用正常ECG训练的轻量级β‑VAE即可捕捉瘢痕相关异常。

**🔧 技术方法**

采用β‑VAE（浅层与预训练版）进行潜在表示学习，基于DTW计算重建误差，结合随机森林、梯度提升、逻辑回归等传统机器学习分类器，并用Mann‑Whitney U检验评估误差分布差异。

**📊 数据集**

使用300名DCM/NDLVC患者的12导联ECG及其对应的CMR LGE注释，以及3000条来自PTB‑XL公共数据集的正常ECG进行浅层β‑VAE训练。

**📈 对比分析**

通过LOSO、5‑折交叉验证和80/20随机拆分等方法，对不同特征表示、归一化方式和分类器进行网格搜索；ECGx.AI最佳AUROC为0.686，浅层β‑VAE基于DTW误差的最佳AUROC为0.643，梯度提升在浅层模型上得到AUROC 0.577。

**⚠️ 局限性**

局部样本量有限，缺乏多中心外部验证，未整合临床变量，重建与分类目标的权衡仍不充分，影响模型在临床应用中的泛化性和可解释性。

---

## 521. AI for Computational Design Science: A Responsible Human-AI Framework and Case Study on Short-Form Video Safety Surveillance

**arXiv ID:** 2609.05270 | [PDF](https://arxiv.org/pdf/2609.05270v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 522. Testing Interchangeability in LLM Agent Teams

**arXiv ID:** 2609.05279 | [PDF](https://arxiv.org/pdf/2609.05279v1)

**作者:** Jianxin Gao `[一作]` (China Agricultural University), Zining Wang `[通讯]` (Tianjin University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估大型语言模型代理团队成员互换对任务得分与协调成本的影响，并提出基于角色互换的“swap test”方法。

**💡 创新点**

首次将交叉对比实验与文本笔记分离相结合，量化团队形成过程中的伙伴特定信息对协调效率的贡献。

**🔧 技术方法**

使用GPT‑5.6 Luna等LLM生成私有笔记并按任务与伙伴拆分，进行角色互换实验，并通过协议签名与消息计数衡量协调成本。

**📊 数据集**

采用公开基准 Collab‑Overcooked（低耦合/高耦合）和 Hanabi 的隐藏任务集进行评估。

**📈 对比分析**

通过对比自对照（Placebo）、Swap、Naive 等条件，发现任务得分几乎不变但通信成本提升 16–63%；Swap惩罚随耦合度、团队历史长度和解码温度变化，表明伙伴特定性虽有限但显著。

**⚠️ 局限性**

实验仅限二人对话、形成周期短、笔记分离可能夸大现象、协议签名仅为表面特征，未检验多代理或长期形成的影响。

---

## 523. Who Should Grade My Work? Student Perspectives on Transparent AI-Assisted Writing Assessment in Higher Education

**arXiv ID:** 2609.05346 | [PDF](https://arxiv.org/pdf/2609.05346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 524. Ritgard: T(r)opical Islands of Socio-Technical Artifacts on GitHub

**arXiv ID:** 2609.05278 | [PDF](https://arxiv.org/pdf/2609.05278v1)

**作者:** Adam Štěpánek `[一作]` (Masaryk University), Michele Lanza `[通讯]` (Software Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个可视化工具，将 GitHub 项目的 Issues、PRs 和 Discussions 等社会技术文档（STA）进行主题建模后，以三维岛屿地图展示其演进与活跃度。

**💡 创新点**

将自然语言 STA 通过嵌入+聚类+LLM 摘要得到主题，并将每个主题映射为岛屿、每条 STA 映射为树木，利用海拔高度动态表示活跃度，实现了“岛屿”可视化与时间演进的统一表达。

**🔧 技术方法**

使用 Qwen3-Embedding-8B 进行文本嵌入，UMAP 降维，HDBSCAN 聚类，LLM（大模型）生成主题名称，Godot 引擎渲染 3D 交互可视化。

**📊 数据集**

以四个 GitHub 开源项目（A-D）为示例，收集其 Issues、PRs、Discussions，构成实验数据集。

**📈 对比分析**

通过可视化对比四个项目的活跃度与结构，但未进行量化性能基准，主要通过示例图和使用体验展示效果；构建时间受 GitHub API 限速限制。

**⚠️ 局限性**

局限在于需要 GitHub 访问令牌、对大型仓库的速率限制、需要高性能 GPU 进行嵌入与聚类、滑动窗口长度预设、缺乏与现有可视化工具的对比以及未在大规模实验中验证可扩展性。

---

## 525. How to Speculate about Uncertainty in Agentic Coding? A Draft-Model Gate Method

**arXiv ID:** 2609.05274 | [PDF](https://arxiv.org/pdf/2609.05274v1)

**作者:** Konstantin Grotov `[一作]` (ITMO University), Valentin Malykh `[通讯]` (MWS AI, IITU University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Speculative Uncertainty (SU)，利用小型开放权重草稿模型对黑盒 LLM 代理产生的 reasoning‑action 轨迹进行单次前向评估，提取阶段感知特征并通过线性校准器输出失败可能性，再在执行前插入 veto gate，提前拦截可能失败的动作。

**💡 创新点**

创新点包括：① 将 speculative decoding 逆向用于评分而非生成；② 只依赖代理输出 token，完全不需要内部 logits/权重；③ 通过分离 reasoning 与 action 两阶段来捕获不同熵信号；④ 用最小成本的线性校准器实现可解释且跨模型的失败预测；⑤ 在闭源 API 场景下实现高效部署与零样本迁移。

**🔧 技术方法**

技术细节：逆向 speculative decoding + teacher‑forcing 交叉似然计算；阶段感知特征（均值、方差、极值、趋势等）提取；L1 正则化的逻辑回归校准器；预执行 veto gate 策略；对 Qwen3‑Coder‑480B、Claude 3.5 Sonnet 等大模型进行评估。

**📊 数据集**

数据集：SWE‑rebench OpenHands 轨迹、GitHub 实际问题、SWE‑Bench Verified、DA‑Code 等；使用 Qwen3‑Coder‑480B、Claude 3.5 Sonnet 作为目标代理，Qwen3‑4B 作为草稿模型；在多种训练配置（SFT、TF、混合语料）下进行实验。

**📈 对比分析**

与 baseline（verbalized confidence、Last‑TP、Global‑TP、HTC）对比，SU 在 AUROC 上提升约 0.15–0.20（最高 0.80 vs HTC 0.82），且在门控策略下可将 per‑call 执行错误率下降 6–8 pp，Token 花费下降 14–19%，且零样本迁移到 out‑of‑distribution benchmark 亦能保持显著收益。

**⚠️ 局限性**

局限性：仅在代码执行成功这一二元可验证目标上验证；对其他域（SQL、Web、工具编排等）的效果未量化；校准仍有偏差，需进一步改进；草稿模型成本假设适用于代码任务，其他任务可能不成立；实验仅为单次估计，未给出置信区间或种子波动。

---

## 526. CONTINUITY: Security-Context Contracts for Composable LLM Agent Controls

**arXiv ID:** 2609.05269 | [PDF](https://arxiv.org/pdf/2609.05269v1)

**作者:** Chris Zheng `[一作]` (ZAST.AI), Geng Yang `[通讯]` (ZAST.AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种端到端的“后果完整性”模型，设计并实现了一个基于签名、证明携带的安全上下文链，能够在LLM代理执行过程中防止安全上下文断裂（security-context discontinuity）导致的恶意后果。

**💡 创新点**

创新点包括：①将根授予、权限证明、原始数据出处、语义转换、策略时效性、最终执行权限等多种安全控制统一映射为可验证的证明链；②提出“端到端后果完整性”定义并给出其组合充分条件；③构造了 128 个跨层 fault 类的系统级对照测试，证明完整配置能够在所有攻击实例中阻止恶意效果；④通过目标消融展示每一项安全假设的重要性，验证了组合安全性的“防御深度”。

**🔧 技术方法**

核心技术：签名和哈希（Ed25519 + SHA‑256），JSON Pointer 叶级路径解析，基于 EdDSA 的签名证据，独立验证的语义转换证据（如别名解析），签名的原始数据释放凭证，基于 assume–guarantee 的合同式设计，基于时间戳和 nonce 的一次性许可，最终的可重入性检查，Python 3.11+ 实现，Deterministic JSON 编码。

**📊 数据集**

使用的“数据集”是一个合成的故障注入基准：32 种 fault 类 × 4 个领域（工作区邮件、金融支付、DevOps 部署、多代理委托）× 20 参数化实例 = 2560 个攻击场景；另外 700 个正常任务（300 需要签名释放，400 纯正常）和 200 个未签名释放的模糊任务。该基准通过程序化生成而非真实流量。

**📈 对比分析**

比较方法：对比七种不完整配置（PassThrough、ToolAllowlist、GatewayPolicy、ProvenanceGateway、EffectBoundPermit、Gateway+Finality、CONTINUITY）在同一基准上的攻击成功率（ASR）、正常任务完成率、模糊任务升级率。性能测量在 Linux 主机上，平均验证延迟 4.21 ms，端到端包含转换与最终检查 7.17 ms；随着签名转移数线性增长。完整配置在 2560 次攻击中 ASR 为 0%，在所有 128 fault 类中都被触发，正常任务完成率 100%，模糊任务升级率 100%。

**⚠️ 局限性**

局限性包括：①根权限和所有验证器均为部署时硬编码的信任实体，若被破坏则安全失效；②模型仅验证安全事实的连贯性，无法保证其真实性；③仅实现了一种语义转换（别名解析），需扩展到更丰富的转换语言；④基准为人工合成，未覆盖真实攻击分布；⑤最终执行环境采用内存单线程模拟，未考虑分布式事务、重试、并发等真实场景；⑥未实现形式化验证，Python 代码仅通过回归测试；⑦不防止信息泄露的潜在通道或恶意文本输出。

---

## 527. The History Is the Detector: Executing CVE Patch History, End-to-End

**arXiv ID:** 2609.05335 | [PDF](https://arxiv.org/pdf/2609.05335v1)

**作者:** Qiushi Wu `[一作]` (IBM Research), Ian Molloy `[通讯]` (IBM Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将公开的 CVE 修复历史自动转化为可执行的检测规则，并通过分阶段流水线在目标项目中识别、验证并修复漏洞。

**💡 创新点**

创新点包括：① 构建了基于 CVE 修复的可重用检测知识库；② 采用成本排序的多层检测管道，将昂贵的 LLM 语义推理和运行时验证仅保留给少量候选；③ 通过双向差分测试对生成的补丁进行独立验证。

**🔧 技术方法**

使用的技术主要有：Tree‑sitter 语法索引、Deterministic 过滤器、LLM 代理式语义审查、运行时环境构建、AddressSanitizer 等安全报告、两侧差分测试，以及将检测规则打包为可部署的“Skill”。

**📊 数据集**

数据集：收集 19,325 条 2022‑2026 年高危 CVE，验证 2,710 条修复提交，生成 1,033 条规则（56 个 CWE 家族，172 个技能），在 14 个开源项目上进行扫描。

**📈 对比分析**

与现有工具（OpenAI Codex、Anthropic、Visa VVAH）在 wolfSSL 上对比，召回率约 52‑55%；10 次重复跑中发现 125‑147 条结果，覆盖率随跑次递增；运行时验证产生 644 条有 ASan/崩溃等证据的结果；在成本上，使用小模型多次并行比单次使用大模型更高效。

**⚠️ 局限性**

局限性：仅覆盖可通过 API 错误使用的反复出现的弱点，无法检测一次性设计缺陷或全局状态问题；运行时验证受限于可重现的构建与测试环境；补丁验证仅证明补丁阻断示例攻击，未保证无回归或维护者接受；模型推理具有随机性，需多跑；规则库仍可能遗漏新出现的 CWE 或未覆盖的 API。

---

## 528. Diffusion TV: Experiencing Diffusion Models through Tangible, Embodied Interaction

**arXiv ID:** 2609.05404 | [PDF](https://arxiv.org/pdf/2609.05404v1)

**作者:** Sihwa Park `[一作]` `[通讯]` (York University), Sihwa Park (York University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

制作了一个可互动的AI艺术装置Diffusion TV，通过旋转天线让观众体验扩散模型的去噪过程，并将中间生成状态以物理媒介呈现。

**💡 创新点**

创新点在于将CRT电视的物理交互映射到扩散模型的去噪阶段，提供感官化、可体验的可解释AI方式。

**🔧 技术方法**

使用Stable Diffusion XL与Stable Audio Open生成图像和音频，结合Python+Processing+Raspberry Pi5+Google Colab+Google Drive进行数据处理与展示。

**📊 数据集**

使用IUCN红色名录与世界自然基金会数据库挑选的16种已灭绝动物、14种濒危动物以及12种未来想象生物作为数据集。

**📈 对比分析**

由于主要是体验型，没有进行定量性能比较；通过访谈与现场观察收集用户反馈，认为交互直观且能激发对生成过程的直觉认知。

**⚠️ 局限性**

限制在于缺乏系统化评估，观众对AI技术的理解有限，实时生成受限，交互门槛对年轻人较高。

---

## 529. Necessary or Sufficient? Evaluating LLM Explanations With Behavioural Evidence

**arXiv ID:** 2609.05385 | [PDF](https://arxiv.org/pdf/2609.05385v1)

**作者:** Urja Pawar `[一作]` (BNY), Houssem Chatbri `[通讯]` (BNY)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估LLM在advisor推荐和prompt风险监测两类任务中自报的前三个特征是否与通过干预得到的必要性和充分性一致，以检验解释的可靠性。

**💡 创新点**

提出一种基于黑盒干预的必要性/充分性评估框架，将模型自报解释与可观测决策行为直接关联，并揭示解释与实际影响之间的可靠性与覆盖差异。

**🔧 技术方法**

采用干预式必要性（特征替换）和充分性（特征保留）计算，结合Spearman相关、未提及特征比较和集合平均得分，对Claude、GPT和Gemini系列模型进行评估。

**📊 数据集**

使用合成数据集：100个顾问推荐案例（包含18个客户特征与13个顾问）和100个风险监测案例（基础请求加3-4个风险段）。

**📈 对比分析**

通过Spearman相关、未提及特征得分比较以及集合平均得分进行比较；在advisor推荐中相关性约为0.35–0.36，在风险监测中为0.43–0.58，并且多数模型的前三特征未能覆盖最高影响特征。

**⚠️ 局限性**

研究仅限于合成数据、top-3解释、温度为0的生成，并未探讨内部机制、解释长度、真实场景多样性或解释对用户可理解性的影响，干预定义也可能不足以捕捉所有因果影响。

---

## 530. CUA-Universe: A Scalable and Dynamic Environment for Hybrid GUI+CLI Agents

**arXiv ID:** 2609.05374 | [PDF](https://arxiv.org/pdf/2609.05374v1)

**作者:** Haoting Shi `[一作]` (Shanghai Jiao Tong University), Yanfeng Wang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对真实桌面软件构建可扩展的混合GUI+CLI环境，并基于该环境训练计算机使用代理。

**💡 创新点**

通过 App-Forge 自动化将任意桌面应用转换为可复现 VM 并发现/生成 CLI 接口，Task-Weave 合成多难度混合任务，Path-Steer 指导高效轨迹，从而实现无需人工工程即可大规模生成环境与训练数据。

**🔧 技术方法**

采用自动化安装代理、CLI 自动发现/生成、操作抽象、任务合成、路径引导以及 VLM 验证等技术。

**📊 数据集**

以16款真实桌面应用（如 Blender、Draw.io、Zotero 等）为基础，生成约4,923条验证轨迹，约235K 步级训练样本。

**📈 对比分析**

在 CUA-Verse、OSWorld 与 OSWorld-MCP 上与闭源及多款开源模型对比，9B 模型在 CUA-Verse 成功率提升约3倍、token 减少60%；在 OSWorld 提升 16.8 份成功率、步骤和 token 分别减少 57% 与 44%。

**⚠️ 局限性**

局限包括在3D/空间类应用上的表现仍相对弱、对完全新工具接口的泛化仍需提升、数据规模虽然大但不足以覆盖所有复杂工作流、以及对 VLM 判定的依赖可能引入误差。

---

## 531. WorldSculpt: Generating Compositional Worlds from Grounded Videos

**arXiv ID:** 2609.05416 | [PDF](https://arxiv.org/pdf/2609.05416v1)

**作者:** Muyao Niu `[一作]` (Alaya Lab), Zhixiang Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 WorldSculpt，利用多视角图像和对象级 3D 生成先验来生成可编辑的包含数百个独立网格的复杂场景。

**💡 创新点**

创新点在于：①将单视角对象生成模型迁移到多视角，并在锚对齐的规范体内对齐所有观测；②通过 IBR 样式聚合器实现可变视角的视图聚合；③引入条件视角增强训练日程以提升对遮挡与噪声的鲁棒性；④无需场景级训练即可在大规模密集场景中泛化。

**🔧 技术方法**

核心技术包括：Pixal3D 生成先验、DINOv3 视觉特征、anchor‑aligned canonical 框架、IBR 聚合器、LoRA 低秩适配、条件视角增强、以及多视角投影与融合。

**📊 数据集**

使用的数据集有 Toys4k（单对象）、Toys4k‑Scene（合成多物体）、HouseCat6D（真实桌面场景）、以及自建的 UE‑MeshyScene（六个 Unreal 真实感场景，最多 701 个对象）。

**📈 对比分析**

与 Pixal3D、TRELLIS、TRIS、MER、RecGen、ShapeR 等基线进行比较；在单视角下与 Pixal3D 接近，在多视角和遮挡场景中明显优于所有基线；在 UE‑MeshyScene 上平均 CD‑ℓ2 下降 12%，F‑Score 提升 0.7%，显示在极度拥挤和遮挡环境下的优势。

**⚠️ 局限性**

局限性包括：依赖先验的相机姿态、实例掩码和粗定位；仅生成几何，不包含材质；只适用于静态场景；对极端低光、运动模糊等条件的鲁棒性有限。

---

## 532. UniMate: One Unified Model to Animate Diverse Skeletons

**arXiv ID:** 2609.05415 | [PDF](https://arxiv.org/pdf/2609.05415v1)

**作者:** Linzhan Mou `[一作]` (Princeton University), Szymon Rusinkiewicz `[通讯]` (Princeton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种统一的基础模型 UniMate，可仅凭已绑定的 3D 角色和文本描述，生成任意拓扑骨骼的动画，无需在推理时对每个骨骼进行微调或额外的参考动作；

**💡 创新点**

核心创新在于：1）Topology‑Aware Diffusion Transformer（TADiT）——通过图感知注意力偏置、Spectral Rotary Position Embedding（Spec‑RoPE）以及全局拓扑调节器，将骨骼拓扑信息嵌入到变压器的注意力机制中；2）构建了覆盖 13,006 条序列、7 类不同骨骼形态的统一数据集 UniMotion；3）采用条件流匹配训练，提升生成质量与效率；

**🔧 技术方法**

主要技术包括：基于图的注意力偏置、Spectral Rotary Position Embedding（Spec‑RoPE）使用图拉普拉斯谱编码、全局拓扑条件 AdaLN‑Zero、条件扩散模型（flow‑matching 训练）、多头注意力的时空分解、文本编码与骨骼语义嵌入；

**📊 数据集**

使用了 UniMotion 数据集，包含 13,006 条动画序列，涵盖双足、四足、鸟类、海洋生物、昆虫、蛇形以及可连杆物体等多种骨骼拓扑，并与 3,584 条独特文本描述配对；

**📈 对比分析**

与现有拓扑无关运动生成与网格动画基准方法相比，UniMate 在质量（视觉与运动一致性）、泛化能力（跨拓扑零样本迁移）以及推理速度上均实现了显著提升；此外，模型还能在无监督环境下实现跨拓扑运动迁移、插值、扩展与文本引导编辑；

**⚠️ 局限性**

局限性包括：1）对极度复杂或非典型骨骼拓扑（如多连杆机械臂）仍可能表现不足；2）需要预先提供精确的 rest‑pose 骨骼；3）在极大规模实时场景下仍受 GPU 计算资源限制；4）数据集覆盖仍不完全，可能缺少某些稀有动作或风格。

---

## 533. A Deep Generative Model for Synthesizing Labeled Wireless Signals

**arXiv ID:** 2609.05396 | [PDF](https://arxiv.org/pdf/2609.05396v1)

**作者:** Yuxiao Li `[一作]` (Basque Center for Applied Mathematics), Yuan Shen `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于深度生成模型的无线信号合成框架 IIns-GAN，用于生成带有距离与环境标签的高保真 UWB 信号。

**💡 创新点**

通过将生成任务建模为潜在变量模型的变分推断，并引入全局对抗变量 ξ，将 VAE 与 GAN 结构结合，实现了对距离与环境特征的可控、真实信号生成。

**🔧 技术方法**

使用变分自编码器、生成对抗网络、隐式分布假设、PatchGAN 判别器以及残差网络 / LSTM 等深度学习模块。

**📊 数据集**

在公开的 UWB 数据库（Dataset 3）上训练，包含 21,250 个带真实距离与 LOS/NLOS、室内/室外标签的信号样本。

**📈 对比分析**

通过波形、功率谱密度、四个物理特征（最大振幅、上升时间、能量、峰度）以及对距离估计与环境识别的 CNN 评估，生成信号与真实信号的相似度均超过 0.9，数据增强后模型误差降低约 10%。

**⚠️ 局限性**

模型仍受限于标签维度、环境细粒度不足，且对极端噪声/多路径条件的泛化能力待进一步验证。

---

## 534. Think-Verify-Revise: Neuro-Symbolic Visual Reasoning with Vision-Language Models and Dynamic Logic Tensor Networks

**arXiv ID:** 2609.05388 | [PDF](https://arxiv.org/pdf/2609.05388v1)

**作者:** Homayoun Afshari `[一作]` (Politecnico di Torino), Lia Morra `[通讯]` (Politecnico di Torino)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种闭环神经-符号框架，利用Vision‑Language Model（VLM）生成First‑Order Logic规则，并通过Dynamic Logic Tensor Network（D‑LTN）进行可微分验证，自动从少量视觉样本中学习并迭代改进Sudoku约束。

**💡 创新点**

创新点在于将VLM的生成能力与D‑LTN的可微分逻辑验证相结合，形成思考‑验证‑修正闭环，实现规则的自动生成、验证和迭代优化，避免了传统NeSy系统对人工规则的依赖。

**🔧 技术方法**

使用技术包括：Vision‑Language Model（如 Llama‑3/Claude 等）生成FOL规则；Dynamic Logic Tensor Network（D‑LTN）做可微分逻辑验证；CNN 作为视觉编码器；链式提示与正则化回馈机制来引导规则生成。

**📊 数据集**

实验基准为 ViSudo‑PC，包含四个视觉域（MNIST、EMNIST、KMNIST、FMNIST）的 4×4 Sudoku 图像，测试样本数量为 100 对每个子集。

**📈 对比分析**

通过与 NeuPSL、LTN‑IND A/B/C 等现有 NeSy 方法对比，在四个视觉域上的 AUC 分别为 MNIST 0.94±0.10、EMNIST 0.93±0.10、KMNIST 0.88±0.10、FMNIST 0.87±0.09，匹配或超越基线。

**⚠️ 局限性**

局限性包括：只能处理单规则、依赖预先设定的符号表、在 FMNIST 领域难以收敛、未实现自动符号发现、未充分利用预训练视觉模型、闭环反馈未采用正式强化学习策略。

---

## 535. Molecular Déjà Vu: Digit-Level Retrieval of Published Values in Frontier Language Models

**arXiv ID:** 2609.05381 | [PDF](https://arxiv.org/pdf/2609.05381v1)

**作者:** Matthias Busch `[一作]` (Hamburg University of Technology), Christian Feiler `[通讯]` (Helmholtz-Zentrum Hereon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对22个前沿大语言模型在12个分子回归基准上的表现进行审计，使用基于数字级别的检索统计量，评估模型是否在基准数据上进行逐字检索（memorisation）而非真正预测。

**💡 创新点**

创新点在于提出了一种针对回归基准的新检索检测方法，该方法通过比较预测值与公布值在一、二、三位有效数字上的保持率，衡量模型对数字的“记忆”而非整体误差；同时系统分析了推理层次、数据重分发与检索率的关联，揭示了检索率随推理令牌数量增加而上升的趋势。

**🔧 技术方法**

使用的技术主要包括：零射击查询、控制推理令牌数量的提示设计、数字保持率(R12、R23)与分子无关的基准floor计算、Benjamini–Hochberg多重检验校正，以及对四个模型进行的字符替换“遮蔽”实验来测试检索可中断性。

**📊 数据集**

实验使用12个分子属性基准，涵盖实验测量值、量子化学计算值、文本书本沸点、以及最近的抗病毒药效（2024–2025）盲测挑战；每个基准均包含约500个分子（部分基准为45个沸点或其它数量）。

**📈 对比分析**

比较方法：通过统计每个模型-基准组合在三位有效数字上的保持率以及R12、R23保持率是否显著高于分子无关的floor，来判定是否存在检索。实验结果显示，检索集中在5个广泛分发的基准（FreeSolv、ESOL、LD50、AqSolDB和沸点），其余基准检索率低或不可检测。推理层次越高，检索率越高，且在遮蔽实验中，结构字符替换能显著降低检索率并收敛模型误差，说明检索是影响排名的主要因素。

**⚠️ 局限性**

局限性包括：检测方法仅关注数字保持率，无法判断对低位数字的预测能力；在低检索率单元中样本不足导致统计功效不足；遮蔽实验仅使用字符替换方法，未探讨其他更细粒度的检索抑制策略；未评估传统基准模型（如最近邻或QSAR）在同一数字统计上的表现。

---

## 536. Design Docs Are All You Need: An AI-native Machine-Learning Performance Tool

**arXiv ID:** 2609.05364 | [PDF](https://arxiv.org/pdf/2609.05364v1)

**作者:** Samuel Kushnir `[一作]` (Google DeepMind), Suvinay Subramanian `[通讯]` (Google)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了以自然语言设计文档为唯一持久规范，由 AI 编码代理按文档层级顺序重建整个机器学习性能建模库的方法。

**💡 创新点**

创新点包括：将代码视为可重建产物；利用机器自动发现的 DAG 调度子代理；使用递归符号 IR（包含 fast/slow 模式）与 SymPy 表达式；以及通过工作示例文档实现可靠的重建与验证。

**🔧 技术方法**

技术手段：自然语言设计文档、机器自动推断依赖 DAG、分层子代理编译、递归符号 IR、SymPy 符号成本表达式、快速解析与模块化调度、AI 编码代理（如 Claude Code）以及与 TPU 的集成。

**📊 数据集**

未使用传统数据集；仅以 DeepSeek‑V3 在 TPU pod 切片上的部署结果作为手工审计的参考模型。

**📈 对比分析**

通过对比重建后库与手工审计的参考模型，验证了数值与符号表达式的精确匹配；在大规模设计点扫描中使用 fast 模式实现数千点的闭式评估，slow 模式用于细粒度调度与资源约束验证。

**⚠️ 局限性**

局限性：依赖 AI 生成的代码质量与上下文窗口限制；对文档写作规范要求高，错误修正仍需人工；缺乏跨平台通用的自动验证机制；在极大规模系统中的完整重建仍需显著计算成本。

---

## 537. Beyond Scalar Flexibility: From Eligible AI Workloads to Dependable Load Relief

**arXiv ID:** 2609.05406 | [PDF](https://arxiv.org/pdf/2609.05406v1)

**作者:** Meiyi Li `[一作]` `[通讯]` (Louisiana State University), Meiyi Li (Louisiana State University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用阿里巴巴 185 天 155,410 GPU 的公开生产追踪，构建工作负载条件的功率模型，并基于此推算可随时释放的灵活负载（即“可实现负荷”）的时长-可靠性曲面。

**💡 创新点**

提出用时长、可靠性、集群组合、实现率等维度来描述数据中心灵活性，取代传统的单一“可灵活比例”描述；同时通过三种对照（均值标度、时间洗牌、独立集群）量化标度误差与协方差效应。

**🔧 技术方法**

工作负载条件功率模型、Monte Carlo 参数抽样、时间序列的持续性度量（K_α,h）、协方差分解、滚动起点验证、以及对比实验（均值标度、时间洗牌、独立集群）等技术。

**📊 数据集**

阿里巴巴 2026 年公开的 155,410 GPU、37,707 服务器、17 个集群、185 天（4,439 小时）追踪数据；并提供足够统计表、功率波段、灵活性 envelope、K_α,h 表等衍生数据。

**📈 对比分析**

与传统单一比例标度以及独立集群假设进行对比。结果表明，均值标度在 1‑小时到 24‑小时之间分别高估 17–47% 的可用功率；独立集群假设在 4‑小时和 24‑小时分别高估 11–20%；聚合后四小时可用功率从 0.38 提升到 0.66，但正相关仍导致收益低于理想独立值。

**⚠️ 局限性**

局限性包括：缺乏真实控制实验（实现率 q 未测定）；只在小时级别建模，无法验证子小时响应；追踪仅代表单一运营商，难以推广；以及工作负载优先级映射和延迟记录不完整导致对可调度灵活性的估计可能偏低。

---

## 538. A Generalizable Feature Extractor for Alzheimer's-Related Brain MRI Tasks

**arXiv ID:** 2609.05400 | [PDF](https://arxiv.org/pdf/2609.05400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 539. WearableQA: A Benchmark for Health Reasoning over Real-World Wearable Data

**arXiv ID:** 2609.05405 | [PDF](https://arxiv.org/pdf/2609.05405v1)

**作者:** Ji Soo Lee `[一作]` (KAIST), Benoit Corda `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一个基准，使用200名真实用户的可穿戴设备长期记录、血液生物标志物和人口统计信息，生成4,084道多项选择题；

**💡 创新点**

创新点包括：1）双重基线构造框架，将文献基础生理发现与人口统计学验证模式相结合，生成确定性答案；2）16类诊断性问题，覆盖数据推理/健康推理与单信号/交叉信号推理；

**🔧 技术方法**

使用技术主要是时间序列处理、确定性计算、人工审核循环以及对LLM的评估方法；

**📊 数据集**

数据集由200名真实用户的可穿戴设备测量（心率、睡眠、活动、心率变异性等）覆盖数百天，并结合血液指标与人口统计信息构成；

**📈 对比分析**

对14款LLM（专有与开源）在该基准上评估，准确率从10%基线到72.9%不等，能细粒度诊断模型在不同推理类型和信号复杂度上的表现；

**⚠️ 局限性**

局限在于多信号交叉推理仍然挑战较大，开源模型表现普遍较低，基准主要针对多项选择题，缺乏开放式问答或更复杂交互形式。

---

## 540. RegionFed: Federated Learning for Personalized Query Understanding in Heterogeneous Retail Environments

**arXiv ID:** 2609.05403 | [PDF](https://arxiv.org/pdf/2609.05403v1)

**作者:** Quoc H. Nguyen `[一作]` (Walmart Global Tech), Chittaranjan Tripathy `[通讯]` (Walmart Global Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出RegionFed框架，通过梯度冲突实现地区级联邦学习个性化，兼容Transformer和CNN；

**💡 创新点**

创新点在于使用梯度级别的ℓ₂冲突信号驱动自适应个性化强度和策略选择，避免参数级别方法在Transformer上的崩溃；

**🔧 技术方法**

技术包括梯度冲突检测、DP梯度噪声、金字塔式地区协调、三种个性化策略（Grad、Interp、Meta）及动态路由；

**📊 数据集**

数据集包括Amazon ESCI（查询理解）、Amazon Reviews（情感分析）以及LEAF-FEMNIST（字符识别）；

**📈 对比分析**

与FedAvg、FedProx、SCAFFOLD、FedTP等基线相比，RegionFed在T5-Small上达92.27%总体准确率，接近隐私违规集中训练的92.04%，参数级方法在Transformer上失效但在CNN上仍能提升；

**⚠️ 局限性**

局限在于需预定义地区、对多模态或时序模型未验证，以及动态策略选择偶尔误路。

---

## 541. Same Trajectory, Contradictory Rewards (ROBORMBENCH): Paraphrase Fragility in Vision Language Reward Models

**arXiv ID:** 2609.05401 | [PDF](https://arxiv.org/pdf/2609.05401v1)

**作者:** Wonje Jeung `[一作]` (Yonsei University), Albert No `[通讯]` (Yonsei University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了视觉‑语言模型（VLM）作为机器人奖励函数时对同义任务描述的敏感性，构建了RoboRewardBench（Robot Reward Model Bench）并设计了三种释义生成与验证方法；

**💡 创新点**

提出基于已验证同义指令的奖励模型鲁棒性评估框架，定义了 Score Crossing Rate（SCR）、Flip Rate（FR）等指标，并展示了模型规模与推理并不提升鲁棒性，进一步提出聚合与方差约束训练来缓解此问题；

**🔧 技术方法**

使用多种专有与开源VLM（GPT‑4o、GPT‑5.1、Gemini、Qwen、Gemma、Llama）与专门的奖励模型（RR‑4B/8B），利用LLM集成进行语义等价过滤，并计算SCR、FR、ME等指标；

**📊 数据集**

基于RoboRewardBench的数据集，包含2,390条真实机器人轨迹（覆盖14种机器人、exo/ego视角）以及21,673条经验证的同义指令；

**📈 对比分析**

在三种释义策略（词汇替换、句法重构、动作‑目标视角转换）下与多种模型进行对比，结果显示专门训练的奖励模型在SCR/FR/M E上最优；规模增大反而导致不稳定；聚合与方差约束训练显著降低SCR并提升轨迹选择和对比准确率；

**⚠️ 局限性**

仅评估英语指令，未覆盖多语言或代码混用情况；只关注终点奖励预测，未研究步进奖励或轨迹比较等其他奖励设定，释义生成策略可能未覆盖所有语言变体。

---

## 542. CrossDepth: Geometry-Constrained Attention for Generalizable Multi-View Surround Depth Estimation

**arXiv ID:** 2609.05397 | [PDF](https://arxiv.org/pdf/2609.05397v1)

**作者:** Samer Abualhanud `[一作]` (Leibniz University Hannover), Max Mehltretter `[通讯]` (Leibniz University Hannover)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用多视角环视相机阵列的自监督学习框架，结合几何约束的跨图像注意力和相机光线嵌入，实现了在无真实深度标注条件下的度量深度估计。

**💡 创新点**

创新点包括：①将每个像素的相机光线以正弦编码作为嵌入，直接让网络学习视角-深度的非线性映射；②将跨图像注意力限制在离散化圆柱网格上，既保留几何可行性又显著降低计算开销；③在所有特征层上应用学习型跨图像注意力，而非仅在单一尺度。

**🔧 技术方法**

核心技术包括：ViT 视觉 Transformer 编码器（预训练 DINOv3 + LoRA 微调），DPT 退化式解码器，基于圆柱网格的几何约束跨图像注意力，正弦光线嵌入，基于空间、时间和时空的光度一致性自监督损失，姿态网络（ResNet18 + U‑Net）预测相机间变换。

**📊 数据集**

在 DDAD 和 nuScenes 两个环视数据集上进行训练与评估，使用 6 角度相机集合覆盖 360° 场景，分辨率分别为 384×640（DDAD）和 352×640（nuScenes）。

**📈 对比分析**

与 FSM、SurroundDepth、VFDepth、CVCDepth、CylinderDepth 等五个当前主流自监督方法对比，实验显示：在整体和重叠区域的绝对相对误差、RMSE、δ<1.25 等指标上均优于对手；跨图像深度一致性误差显著下降；同时保持与 CylinderDepth 相近的内存占用，优于 VFDepth、SurroundDepth 的高内存需求。

**⚠️ 局限性**

局限性包括：跨图像注意力仅作用于编码器，导致边界附近可能出现局部深度伪影；模型未在更大规模或更复杂的多视角数据上验证，缺乏对更广泛场景的可扩展性评估；在计算资源受限的情况下，对更高分辨率或更大摄像头阵列的适应性尚未探讨。

---

## 543. Reflection-aware Generative Novel View Synthesis

**arXiv ID:** 2609.05382 | [PDF](https://arxiv.org/pdf/2609.05382v1)

**作者:** GeonU Kim `[一作]` (KAIST), Tae-Hyun Oh `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在单张或稀疏输入图像下，通过估计镜面平面并反射相机姿态，将镜像视为补充视角，实现无训练的、反射感知的生成式新视图合成。

**💡 创新点**

创新点在于：①把镜像视为两张互补视图；②提出镜面门控注意力（Mirror‑gated attention）限制反射视图仅影响镜面区域；③引入反射注入（Reflection injection）在反向步骤中填补镜面，实现反射一致的新视图。

**🔧 技术方法**

技术核心为：多视角扩散模型的交叉注意力、镜面检测与平面估计、Householder 反射、镜面门控注意力、反射注入以及 SDEdit 风格的边界平滑。

**📊 数据集**

使用 8 个合成平面镜场景（Blender Demo/Kit/CGTrader 等）和 Mirror‑NeRF 实景数据集进行评估，并在这些数据上构建输入‑测试轨迹。

**📈 对比分析**

与 MVGenMaster、SEVA、FlexWorld、VistaDream 等基线比较，使用 DreamSim、CLIP 相似度、PSNR、SSIM、LPIPS 等指标，结果表明本文方法在所有指标上均显著优于基线，尤其在镜面反射一致性方面提升明显。

**⚠️ 局限性**

局限性包括：仅适用于理想平面镜面，镜面检测与平面估计误差会影响结果；方法仍需依赖多视角扩散模型；对高度扭曲或非平面镜面支持有限。

---

## 544. Propagation Model for SSC attacks: Why SBOM (tools) don't tell the whole truth

**arXiv ID:** 2609.05380 | [PDF](https://arxiv.org/pdf/2609.05380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 545. Towards Neuro-Symbolic Procedural Reasoning for Long-Horizon Vision-Language-Action Manipulation

**arXiv ID:** 2609.05369 | [PDF](https://arxiv.org/pdf/2609.05369v1)

**作者:** Vivek Chavan `[一作]` (Fraunhofer IPK), Jörg Krüger `[通讯]` (Fraunhofer IPK)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种神经符号化程序化推理框架，将任务图、事件记忆与多模态程序性记忆结合，用于解决长周期视觉语言动作（VLA）操纵中的状态持久性、依赖推理和条件决策问题，并通过伪注视点实现空间引导。

**💡 创新点**

创新点在于将显式任务图与事件记忆对高层决策进行约束，同时利用稀疏程序性显著性（伪注视）对低层VLA策略进行空间引导，三者共同提升长周期任务的执行可靠性。

**🔧 技术方法**

技术主要包括：基于DINOv2的特征映射和Lucas‑Kanade光流实现伪注视点迁移；任务图模型 (Directed Graph) 与附加只读事件记忆；VLA微调策略（π0.5）结合显式RGB提示或注意力正则化；多视角感知（固定底座与腕部相机）进行状态验证。

**📊 数据集**

数据集主要为231段人工遥操作的七步仪器处理演示视频（含固定底座与腕部视角），以及在工作区清理与手术器械处理两大长期任务场景下的真实机器人实验数据。

**📈 对比分析**

在实验中，将无提示、RGB提示（环形）和注意力正则化三种方法与基准模型对比，评估容器定位、子目标完成、步骤顺序一致性等指标；结果显示，两种引导方式均在所有实验中完成全部试验，且RGB提示显著提升了模糊容器定位的成功率（10/10→25/25）。

**⚠️ 局限性**

主要局限包括：实验已达到性能饱和，缺乏对未见场景和几何转移的泛化验证；伪注视点仅在RGB提示中实现，无法实现跨视角的几何显著性迁移；当前仅验证了任务图与记忆的局部效果，整体管线的完整性与恢复机制尚待评估。

---

## 546. From Interpretability Methods to Interpretable Models

**arXiv ID:** 2609.05399 | [PDF](https://arxiv.org/pdf/2609.05399v1)

**作者:** Julien Colin `[一作]` (ELLIS Alicante), Thomas Serre `[通讯]` (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述XAI工具成熟度并提出将关注点从解释方法转移至模型本身的研究议程，阐明模型可解释性应视为可度量、可比较的属性；

**💡 创新点**

将成熟的XAI方法用于模型比较与人类理解度量，提出从工具到模型的系统评估框架，强调解释性应以人类可理解性为核心；

**🔧 技术方法**

归因、特征可视化、概念化、电路分析等现有XAI方法，结合表示对齐、交叉编码器、通用稀疏自编码器等技术；

**📊 数据集**

主要使用公开模型及其数据集，如ImageNet、CLIP、DINO、MAE等；

**📈 对比分析**

通过对内部表示、概念覆盖与电路结构进行直接对比，利用表示对齐度量、交叉编码器等手段；目前缺乏统一的性能评估指标，结果主要为定性和相对比较；

**⚠️ 局限性**

缺乏统一的可解释性定义、对人类评估的高度依赖、现有工具仍不完备，导致模型可解释性的客观度量困难。

---

## 547. Multi-Step Tool-Calling over Korean Open Public APIs: A Benchmark and a Data-Synthesis Recipe

**arXiv ID:** 2609.05395 | [PDF](https://arxiv.org/pdf/2609.05395v1)

**作者:** Dain Kim `[一作]` (LG CNS), Kyuseong Lim `[通讯]` (LG CNS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了韩国公开API多步骤任务基准KOPA‑Bench，并提出了基于实时执行验证的动态图方法EDGE，用于合成并训练小型开源LLM的多步骤工具调用数据；

**💡 创新点**

创新点在于：①将工具依赖图与真实API执行相结合，动态修剪无效边；②设计高基数结果的分类型轨迹合成；③构建真实的韩国公共API多步任务基准，填补现有评测缺口；

**🔧 技术方法**

使用的技术包括：工具依赖图构建与Beta先验、Thompson采样迭代更新、执行验证与动态修剪；轨迹合成时的高基数分类型处理；GRPO强化学习对生成的数据集进行微调；

**📊 数据集**

使用的数据集为：KOPA‑Bench（145个多步任务，包含2318个工具函数），EDGE合成的训练集约1781条任务；外部对比基准BFCL；

**📈 对比分析**

比较方法：在KOPA‑Bench上对比细调前后模型的pass@1、pass@4和Action等指标，Qwen3.5‑4B从0.18提升至0.31（+13pp），9B从0.33提升至0.43（+10pp），几乎匹配27B模型；在BFCL上亦取得显著提升；对比SFT与GRPO，证实数据集贡献主导；

**⚠️ 局限性**

局限性：依赖实时API，易受端点漂移影响；仅覆盖韩国公共API，难以直接推广到其他语言或私有API；实验仅与内部实现对比，未与现有完整合成系统做系统级评测。

---

## 548. What Matters, When? Diagnosing and Improving Conditional Visual Grounding in Visuomotor Imitation Policies

**arXiv ID:** 2609.05376 | [PDF](https://arxiv.org/pdf/2609.05376v1)

**作者:** Vivek Chavan `[一作]` (Fraunhofer Institute for Production Systems and Design Technology IPK), Jörg Krüger `[通讯]` (Fraunhofer Institute for Production Systems and Design Technology IPK)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究并改进视觉伪造在视觉-运动模仿策略中的条件性视觉定位，通过诊断对象与容器选择瓶颈并引入数据增强、阶段性注意力正则化以及视觉提示，提升了在模拟和实际UR3e机器人上的鲁棒性，并验证了同一方法在预训练视觉语言动作模型中的适用性。

**💡 创新点**

提出“条件性视觉定位”框架，将视觉错误分解为提示、参照物和执行阶段，结合轻量级干预（数据复制粘贴、阶段性注意力正则化、无坐标视觉提示）实现对多种视觉干扰的系统性鲁棒性提升。

**🔧 技术方法**

使用Action Chunking with Transformers (ACT) 作为基准策略；通过图像空间复制粘贴增强、阶段注意力正则化、视觉提示生成与阶段预测器；在VLA任务中微调预训练的 vision‑language‑action 模型。

**📊 数据集**

在仿真环境下生成两种抓取‑放置任务，使用100条干净演示进行训练；对UR3e机器人进行硬件测试；在VLA案例中使用7步仪器操作的231条遥操作视频。

**📈 对比分析**

与标准ACT对比，使用混合干扰时标准ACT成功率仅39.5%/14.0%，而仅增强策略可恢复至100%/64%，完整改进方案（ACT‑Modified）在仿真中分别达到94.5%/88.5%，硬件上为65%/60%。在VLA任务中，加入视觉提示的模型从5/10提升至10/10的模糊路径成功率。

**⚠️ 局限性**

实验规模有限（硬件仅20次/单元，VLA子目标仅5次/条件），仅对颜色/形状的干扰做了部分验证，注意力与表示分析为相关性而非因果，提示策略依赖额外的目标信息，且在更复杂或真实多样化环境中的通用性尚未验证。

---

## 549. When LLM Decompilers Recompile More and Preserve Less

**arXiv ID:** 2609.05370 | [PDF](https://arxiv.org/pdf/2609.05370v1)

**作者:** Chang Liu `[一作]` (Syracuse University), Kristopher Micinski `[通讯]` (Syracuse University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于 fuzzer 自动生成输入的行为比较 oracle，用来评估 LLM 驱动的反编译器在重编译后与原始机器码的功能一致性，并构造了两套无固定测试集（GitHub 库函数与 CVE 漏洞函数），使评测不再依赖人工输入/输出对。

**💡 创新点**

提出了一种不依赖手写测试的动态行为比较方法，自动合成驱动并用 AFL++ 对原始实现进行 fuzz 生成输入，随后在相同输入下比较可观测后置状态；同时首次系统性分析了 LLM 重写产生的新字段、类型与调用导致的行为偏差，并量化了 Crash Absence 现象。

**🔧 技术方法**

使用自动驱动生成、AFL++ 动态 fuzz、AddressSanitizer、边界可观测后置状态摘要、LLM 生成器（LLM4Decompile、SK2Decompile、Idioms、DeGPT‑Qwen、AutoDecompiler、Nova、SLaDe、GLM‑5.2）以及源代码级符号重写分析。

**📊 数据集**

四个传统 LLM 反编译语料库（HumanEval‑Decompile、ExeBench、AnghaBench、MBPP），300 个 GitHub 库函数（无公开测试）和 287 个 CVE 漏洞函数（自 OSV.dev、CVEfixes 收集）。

**📈 对比分析**

对每个候选代码先尝试构建；若成功，用相同输入执行原始与重编译二进制，提取返回值、写内存、全局变量等边界状态摘要；若摘要一致则标记为 Matched，若不同则 Divergence，若原始崩溃但重编译不崩溃则 Crash Absence。实验显示，即使通过所有原始测试的系统，也有约 4.9% 的函数在动态输入上出现 Divergence；LLM 重写可将 Build 率从 75% 提升至 90% 但 Matched 率降至 62%；在 CVE 函数中出现约 8.7% 的 Crash Absence。

**⚠️ 局限性**

仅覆盖能被 AFL++ 生成且可观察的边界状态，对堆/栈/寄存器状态不敏感；对多维数组、函数指针等复杂类型支持不足；LLM 结果受训练分布限制，难以覆盖所有真实代码模式；行为比较无法捕捉细粒度语义差异（如性能、隐式 UB 等）。

---

## 550. Distill Globally, Adapt Locally: Reasoning Distillation and Product-Type Test-Time Training for Scalable Trade-Up Recommendation

**arXiv ID:** 2609.05363 | [PDF](https://arxiv.org/pdf/2609.05363v1)

**作者:** Siliang Liu `[一作]` (Amazon Everyday Essentials Technologies), Amin Banitalebi-Dehkordi `[通讯]` (Amazon Everyday Essentials Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种两层框架：先用检索增强的LLM教师进行推理蒸馏，得到一个只用嵌入对的轻量级学生模型；随后在每个产品类型上进行少量支持集的测试时训练（PT‑TTT），实现无LLM推理的全量catalog trade‑up 评分。

**💡 创新点**

创新点在于将LLM的推理解释转化为嵌入对齐与对比蒸馏，形成全局可共享的表示；并通过LoRA轻量级适配实现每个产品类型的局部决策边界，从而在数亿对级别上实现可扩展、低成本的 trade‑up 推荐。

**🔧 技术方法**

使用检索增强 LLM、自然语言推理、对齐与 InfoNCE 对比蒸馏、embedding‑pair 分类器、LoRA adapter、少量支持集的测试时训练、Focal 损失、KL 相似度等技术。

**📊 数据集**

数据集包括 12.83 万对来自 Amazon–Walmart 的银色监督对，29 个产品类型；一个 8,352 对的人工标注金标基准；以及 100,000 对的代理评估集。

**📈 对比分析**

在金标上，15.5M 参数的四类推理蒸馏学生 AUC 达 0.924；加入 PT‑TTT 后提升至 0.941；相较于 LLM 教师和未蒸馏模型均显著提升，并在速度和成本上比直接 LLM 推理快 5,000 倍、成本低 10,000 倍。

**⚠️ 局限性**

局限性包括仅在 29 类产品上验证，未测试未见类泛化；需要专家标注的支持集；适配效果受支持集构成与随机性影响；未完全排除类别校准的影响；金标与生产流分布不同；LLM 对比仅基于固定配置，未做全量优化。

---

