# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-07-05 | 今日论文总数: 604

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. A Novel Machine Learning Approach for Central Nervous System Tumor Classification from DNA Methylation

**arXiv ID:** 2607.01307 | [PDF](https://arxiv.org/pdf/2607.01307v1)

**作者:** Paulo R. Ferreira `[一作]`, Vinicius F. Campos `[通讯]` (Federal University of Pelotas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

该论文提出了一种基于稀疏随机投影和多项式逻辑回归的DNA甲基化脑瘤分类方法，能够对中央神经系统肿瘤进行91类精细分类。

**💡 创新点**

创新点在于使用稀疏随机投影替代传统特征筛选，避免手工挑选CpG位点，提升模型可解释性与泛化能力，并在高维低样本场景下实现线性分类器的优越性能。

**🔧 技术方法**

核心技术包括稀疏随机投影（SRP）进行维度约减、多项式逻辑回归（Multinomial Logistic Regression）进行多类分类，以及三折分层交叉验证和网格搜索进行超参数调优。

**📊 数据集**

使用的数据集为Capper等人2018年公开的2801个参考样本（Illumina 450K甲基化阵列）作为训练/验证集，和1104个临床前瞻性病例（EPICv2阵列）作为独立测试集。

**📈 对比分析**

与原始Capper随机森林分类器比较，该方法在参考集上三折交叉验证准确率达96%，在临床集上91类准确率为86.9%，相较于原版的82%和家庭级别91%分别提升约4–5个百分点。

**⚠️ 局限性**

局限性包括对稀疏随机投影的参数选择缺乏理论解析，模型仍为线性，可能无法捕捉复杂的非线性关系；此外，在临床集上仍存在少数类误判，且未提供对甲基化级别的深度解释。

---

## 2. TokenScope: Token-Level Explainability and Interpretability for Code-Oriented Tasks in Large Language Models

**arXiv ID:** 2607.01235 | [PDF](https://arxiv.org/pdf/2607.01235v1)

**作者:** Amirreza Esmaeili `[一作]` (University of British Columbia), Fatemeh Fard `[通讯]` (University of British Columbia)

**通讯引用:** 473 | [OpenAlex ID](https://openalex.org/A5029327446)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 TokenScope，一款针对基于解码器的 LLM 代码生成的交互式可解释性工具，实时展示 token 级置信度、熵、惊讶度、注意力、可替代候选等信息，并支持交互式替换与反事实分支；

**💡 创新点**

在工具层面实现了将解码时信号与代码 AST 对齐、可视化及交互分支、以及多级结构聚合（token/表达式/语句/行/块）等功能，弥补了现有工具缺乏解码时置信度与结构关联分析的不足；

**🔧 技术方法**

利用模块化架构（生成服务器、协调服务器、React 前端）捕获 LLM 的 token 概率、注意力权重；对生成过程进行在线统计；采用 Tree‑Sitter 解析器将 token 对齐至 AST；并在前端实现可视化与交互；

**📊 数据集**

主要在 Qwen2.5 Coder 1.5B Base 上演示，示例代码为 Python，未公开使用公开数据集；

**📈 对比分析**

未给出系统性性能对比；工具强调可解释性与交互性，延迟受实时信号采集与可视化渲染影响，非针对推理速度优化；

**⚠️ 局限性**

仅支持解码器式自回归模型、需要访问 token 概率与注意力、对不完整或无效代码的 AST 对齐可能失败，且仅支持已有 Tree‑Sitter 语法树的语言（目前仅 Python）且不适合作为生产推理框架。

---

## 3. From Approximation to Emergence: A Theory of Deep Learning

**arXiv ID:** 2607.01311 | [PDF](https://arxiv.org/pdf/2607.01311v1)

**作者:** Zhilin Zhao `[一作]` (Sun Yat-sen University), Zhilin Zhao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1429 | [OpenAlex ID](https://openalex.org/A5100645286)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

梳理并综合深度学习理论中关于逼近、优化、泛化及现代机制的主要结果，形成统一的框架。

**💡 创新点**

提出将多种理论视为相互重叠的解释程序，强调假设可视化，统一解释不同机制。

**🔧 技术方法**

利用逼近理论、神经切线核、均值场理论、稳定性与PAC‑Bayes、分布式假设、计算下界等数学工具进行分析。

**📊 数据集**

无实验数据集，纯理论综述。

**📈 对比分析**

通过对比假设、结论和适用范围，说明各理论在不同情形下的有效性与局限，无具体性能指标。

**⚠️ 局限性**

仅选取代表性结果，可能遗漏部分研究；理论假设与实际应用不完全一致，缺乏新证据。

---

## 4. An alternative approach towards attacks against fully-split PLWE instances

**arXiv ID:** 2607.01340 | [PDF](https://arxiv.org/pdf/2607.01340v1)

**作者:** Iván Blanco-Chacón `[一作]` (Universidad de Alcalá), Raúl Durán Díaz `[通讯]` (Universidad de Alcalá)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5021424068)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

本文对PLWE（多项式学习误差）中基于根的攻击进行通用化，并进一步研究通过同构把安全实例迁移到潜在更弱实例的思路。作者构造了全分裂情形下的显式同构，并证明任何此类同构与原始根基攻击等价，从而证明同构迁移无法获得新的攻击优势。

**💡 创新点**

创新点在于：①首次在全分裂PLWE实例中给出所有可能同构的完整构造与唯一性证明；②证明任何同构-评估组合等价于原始根评估，故无法提升攻击成功率；③尝试将攻击扩展到任意因子化情形，并指出其成功率受因子零系数数目影响。

**🔧 技术方法**

主要技术包括：有限域同构与CRT分解；多项式环的评估同态和迹算子；根基攻击的概率分析（小集、受限小值、无界小值三类）；线性与代数结构的组合证明。

**📊 数据集**

本文未使用任何实验数据集，全部结论均来自理论证明与符号计算。

**📈 对比分析**

与之前的根基攻击相比，作者通过理论证明表明同构迁移无法提高成功概率；由于缺乏实验验证，无法给出具体性能数值，只能说明理论上不具备优势。

**⚠️ 局限性**

局限性包括：①证明仅针对全分裂情形；②对非全分裂实例的结论仅为尝试性讨论，未给出完整证明；③缺乏实验验证；④依赖特定同构结构与理想因子化假设，可能不适用于所有PLWE参数。

---

## 5. Mapping Text to Multiplex Graph: Prompt Compression as Lévy Walk-Guided Graph Pruning

**arXiv ID:** 2607.01241 | [PDF](https://arxiv.org/pdf/2607.01241v1)

**作者:** Yaxin Gao `[一作]` (Zhejiang University of Technology), Joey Tianyi Zhou `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 11249 | [OpenAlex ID](https://openalex.org/A5045125183)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RAGP，一种基于多层图结构的提示压缩框架，通过 Lévy walks 在多重图中进行冗余节点剪枝，实现长文本压缩。

**💡 创新点**

将文本建模为细粒度注意力层与粗粒度语义层的多重图，利用 Lévy walk 的重尾步长在稠密子图与稀疏全局连接之间自适应探索，提升重要性估计。

**🔧 技术方法**

多重图构造、注意力权重边稀疏化、句子级语义相似度图、Lévy walk 随机游走、重要性计数与预算约束剪枝等技术。

**📊 数据集**

LongBench 评估，涵盖单文档 QA、跨文档 QA、摘要、少样本、合成推理和代码任务。

**📈 对比分析**

与检索式、文本压缩、LLM-based、视觉压缩方法对比，RAGP 在 3000/2000 token 限制下均取得最高平均分（49.3/48.1），超越 LongLLMLingua、Glyph 等基线，且延迟与成本显著下降。

**⚠️ 局限性**

构造多重图消耗额外计算，摘要性能略逊于部分基线，需手动调节 Lévy 指数 μ 与稀疏阈值 δ，且对不同任务的泛化需要进一步验证。

---

## 6. CPG-PAD: Concept-Informed Prompts Guided Presentation Attack Detection

**arXiv ID:** 2607.01303 | [PDF](https://arxiv.org/pdf/2607.01303v1)

**作者:** Haoyuan Zhang `[一作]` (University of Chinese Academy of Sciences), Zhen Lei `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 27529 | [OpenAlex ID](https://openalex.org/A5109299788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于视觉概念引导的面部呈现攻击检测框架 CPG-PAD，利用预训练的 CLIP 模型通过可解释 AI 发现攻击相关视觉概念，并将这些概念以热图形式注入可学习提示，从而提升跨域泛化性能。

**💡 创新点**

创新点：①在 PAD 任务中首次将 XAI 发现的模型级概念与 CLIP 视觉/文本空间对齐；②设计了 Visual Concept-driven Enhancement (VCE) 自动挖掘 PAD 相关概念并生成细粒度热图；③提出 Prompt-based Concept Injection (PCI) 与 Visual-Prompt Decoder (VPD) 共同学习多概念提示并通过概念映射损失实现概念与提示的精确对齐。

**🔧 技术方法**

技术方法：CLIP 视觉/文本双模架构；Semi‑NMF 进行概念分解；概念热图生成；多头自注意力与双头注意力的 VPD；概念映射损失（Hungarian 匹配）；轻量级 ConvPass 适配器；交叉熵 + 概念损失联合训练。

**📊 数据集**

使用了九个公开 PAD 数据集：MSU-MFSD、CASIA-FASD、Idiap Replay‑Attack、OULU‑NPU、CASIA‑SURF、CASIA‑CeFA、WMCA、CelebA‑Spoof、SiW‑Mv2，覆盖多种攻击方式、传感器、光照、材质及未知攻击器材。

**📈 对比分析**

与传统基于 CNN/Transformer 的域泛化方法及最新 CLIP‑基方法（如 FLIP、S‑CPTL、CFPL‑FAS、TF‑FAS、I‑FAS 等）进行 P1–P5 多源/单源/未知攻击器材评估，CPG‑PAD 在 8/9 个跨域场景下平均 HTER 低于 2%，比最先进方法平均低 1–2%，并在 SiW‑Mv2 的未知攻击器材上将 HTER 从 3.03% 降至 1.02%，BPCER‑100 从 11.08% 降至 6.65%。

**⚠️ 局限性**

局限性：①依赖 CLIP 预训练模型，对大模型的资源需求仍较高；②概念数量固定为 15，可能无法覆盖所有极端攻击模式；③热图生成过程在噪声攻击或高分辨率场景下可能不稳定；④方法主要针对 RGB 视觉数据，未结合深度、红外等多模态信息。

---

## 7. I\textsuperscript{2}RiMA: Spectral Riemannian Representation with Temporal Attention for Mental Stress Detection based on EEG Signals

**arXiv ID:** 2607.01279 | [PDF](https://arxiv.org/pdf/2607.01279v1)

**作者:** Cheng He `[一作]` (Capital Normal University), Likun Xia `[通讯]` (Capital Normal University)

**通讯引用:** 2069 | [OpenAlex ID](https://openalex.org/A5020520941)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种跨受试者EEG应激检测模型I2RiMA，结合频率感知的黎曼几何特征提取与 intra‑inter 切片注意力融合，提升跨受试者泛化能力。

**💡 创新点**

创新点在于：① 在频域对每个频点独立构造协方差并映射到SPD切空间，保留频率特定空间关联；② 用数据驱动的频率聚类聚合去除冗余、提升可解释性；③ 引入 intra‑inter 切片注意力模块，动态融合局部谱信息与全局时序依赖。

**🔧 技术方法**

技术包括FFT频谱分解、黎曼几何(Log‑Euclidean映射)、K‑means频率聚类、注意力加权融合、全连接分类器，并采用Adam优化。

**📊 数据集**

实验使用三大公开数据集：MIST Control、MIST Stress（同一受试者在不同情境）以及SEED情感分类数据集。

**📈 对比分析**

与五个主流基线（EEGNet、BIOT、LaBraM、NeuroBOLT、CorrAtt）在5折受试者分层交叉验证下比较，I2RiMA在三数据集上分别取得77.59%、75.88%、82.78%的B.ACC（最高），并在参数（1.6M）和FLOPs（31.95M）上显著低于对照，显示出优异的性能与效率。

**⚠️ 局限性**

局限性包括：① 仅验证于应激与情感两类任务，需在运动想象、癫痫等更多EEG解码任务上验证；② 数据量仍有限（30/15受试者），更大规模的评估仍待开展。

---

## 8. Breaking Safety at the Token Boundary: How BPE Tokenization Creates Exploitable Gaps in LLM Alignment

**arXiv ID:** 2607.01239 | [PDF](https://arxiv.org/pdf/2607.01239v1)

**作者:** Tung-Ling Li `[一作]` (Palo Alto Networks), Yuhao Wu `[通讯]` (Palo Alto Networks)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5114065745)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究字符级扰动如何绕过大型语言模型的安全对齐，并揭示其背后的结构机制；

**💡 创新点**

发现BPE分词碎片化安全关键词导致拒绝信号崩溃，形成一条可验证的因果链；

**🔧 技术方法**

利用BPE分词、logit gap评估、激活补丁、对抗实验、SFT/DPO微调、Conv‑Benign诊断等技术；

**📊 数据集**

使用公开的安全对齐数据集（PKU‑SafeRLHF、Anthropic HH‑RLHF、BeaverTails）和HarmBench进行攻击与评估；

**📈 对比分析**

在五个模型族（Qwen、Gemma、Llama、Mistral）上实验，BPE碎片化能使80–100%原拒绝提示的拒绝信号翻转，其中约48%产生真实有害内容；对SFT/DPO的多参数网格实验显示，DPO无力实现稳健闭环，SFT虽能降低ASR但往往伴随全局拒绝失效；

**⚠️ 局限性**

实验范围受限于模型规模（4–8B）、单一语言（英文）、微调方法（LoRA‑16）以及对齐数据不可见性，未验证更大规模模型、跨语种或更高容量微调的效果；

---

## 9. AnchorSplat: Fast and Structure Consistent Detail Synthesis for Gaussian Splatting

**arXiv ID:** 2607.01290 | [PDF](https://arxiv.org/pdf/2607.01290v1)

**作者:** Dexu Zhu `[一作]` (Chinese Academy of Sciences), Huaibo Huang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 2657 | [OpenAlex ID](https://openalex.org/A5005195235)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AnchorSplat，一种直接在3D Gaussian Splatting资产上进行单通前向细节增强的网络；

**💡 创新点**

通过点锚机制解决梯度混淆和映射模糊，等效密度机制替代迭代稠密化，并实现完全源免费、10⁵倍速度提升；

**🔧 技术方法**

使用点云编码器PTv3、点锚机制、等效稠密化、可微渲染以及L1/SSIM/VGG感知损失的多尺度训练；

**📊 数据集**

构建了3DGS‑SR基准（约15k Objaverse单物体），并在NeRF‑synthetic上验证零样本泛化；

**📈 对比分析**

与SRGS、SuperGaussian、Sequence Matters等传统2D‑3D优化方法对比，3DGS‑SR上达到PSNR 36.57、SSIM 0.943、LPIPS 0.058，渲染时间约0.01 s，速度提升10⁵倍；

**⚠️ 局限性**

仅针对Gaussian Splatting表示，未在非Gaussian或高噪声扫描、复杂材质/光照场景中进行充分验证。

---

## 10. Measure Once, Model Everywhere: Model-Based Per-Request Resource Consumption for HTTP

**arXiv ID:** 2607.01246 | [PDF](https://arxiv.org/pdf/2607.01246v1)

**作者:** Geerd-Dietger Hoffmann `[一作]` (Green Coding Solutions), Verena Majuntke `[通讯]` (HTW Berlin)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5021321066)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出并实现了一种基于离线测量、模型化推断的 HTTP 请求级能耗与 CO₂e 估算机制，并通过 nginx 插件在运行时在响应头中披露。

**💡 创新点**

创新点在于将离线基准测量转化为可执行的端点级能耗模型（常数、线性、分段曲线），并支持应用层提供的特征，使得即使在无精细电力计量的生产环境也能得到可解释、可部署的 per-request 能耗与排放估计。

**🔧 技术方法**

核心技术包括：使用 Green Metrics Tool 进行机能测量；回归与分段插值构建模型；将模型序列化为 JSON 注册表；nginx Lua 模块读取注册表、提取 HTTP 级特征并实时计算能耗、运营与固有排放；以及通过响应头披露模型参数与计算结果。

**📊 数据集**

使用的数据集为自建的简易 REST/TODO 与 AI 推理微服务，在多种请求量、负载（文本长度、附件大小、提示词长短）下执行多次基准，产生每个端点的能耗标签与特征记录。

**📈 对比分析**

对比方法包括：(1) 对照测量与模型估计的总能耗差异（误差约 2.3%）；(2) 开启与关闭插件的能耗与运行时间对比，额外消耗低于 0.5%；(3) 与无头部响应的基准跑对比，验证模型的可接受误差。整体性能表现良好，运行时开销极低，模型评估在毫秒级别。

**⚠️ 局限性**

主要局限：模型仅为接口级抽象，未给出不确定性区间；依赖于硬件、虚拟化与地区电网条件，需周期性重新校准；未覆盖 HTTPS/TLS 开销与第三方 CDN 能耗；应用层特征需要额外注入，手动模型选择未自动化。

---

## 11. Multilayer Q-Matrix-Embedded Neural Network for Cognitive Diagnosis (M-QCDNet): Structure-Aware Deep Learning Architecture for Psychometric Interpretability

**arXiv ID:** 2607.01278 | [PDF](https://arxiv.org/pdf/2607.01278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 12. Three Futures for the Diagnostic Radiologist: A Structured Disagreement About What AI Actually Changes

**arXiv ID:** 2607.01253 | [PDF](https://arxiv.org/pdf/2607.01253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 13. KathaTrace: Diagnosing Semantic Trajectory Collapse in Generated Visual Narratives

**arXiv ID:** 2607.01312 | [PDF](https://arxiv.org/pdf/2607.01312v1)

**作者:** Jamuna S. Murthy `[一作]`, Rajiv Ramnath `[通讯]` (Ohio State University)

**通讯引用:** 1974 | [OpenAlex ID](https://openalex.org/A5073535794)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了KathaTrace协议和KathaBench-25K数据集，用以评估图像序列生成（Storyboard）在保持故事情节过渡意义上的能力，发现了所谓的语义轨迹坍塌现象。

**💡 创新点**

创新点在于：①构造了“语义轨迹差距（STG）”指标，量化文本到图像的过渡意义损失；②设计了图像‑唯一本地可恢复性评估流程，包含文本‑仅、图像‑仅、文本+图像三种证据；③引入对比语义变体、歧义过滤以及可操作的Semantic Compass修复方案。

**🔧 技术方法**

技术上结合了多模态问答评估、维度特定问题设计、对比语义变体检测、人工与VLM混合评估、生成器无关的评估协议以及后期重排序与桥接场景修复。

**📊 数据集**

使用了KathaBench-25K数据集：5,000个源故事（来自Kathasaritsagara、Aesop和Panchatantra），共20,000个相邻场景过渡、28,712个可恢复性问题和10,000个对比变体。

**📈 对比分析**

通过与多种现有Storyboard生成器（StoryDiffusion、DreamStory、LogiStory等）进行对比，评估图像恢复率、STG、情感/因果等维度。结果显示大多数方法STG仍在28–35之间，图像恢复率在39–55之间；Gemma-ST+Semantic Compass将STG降至21.4、图像恢复率提升至55.8，显示显著性能提升。

**⚠️ 局限性**

局限性包括：①对符号化、讽刺或长距离过渡的恢复仍表现不佳，歧义率高；②STG指标受图像质量与计划质量影响，单一方法难以彻底解决；③数据集主要覆盖经典寓言，缺乏多样化叙事结构；④Semantic Compass虽可修复部分错误，但对整体叙事连贯性提升有限。

---

## 14. Cache Merging as a Convergent Replicated State for Multi-Agent Latent Reasoning

**arXiv ID:** 2607.01308 | [PDF](https://arxiv.org/pdf/2607.01308v1)

**作者:** Carlos Baquero `[一作]` (Universidade do Porto), Luís Brito `[通讯]` (Politécnicо de Viana do Castelo)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向多代理隐式推理的KV‑缓存合并方法，去除输入顺序依赖并实现收敛的复制状态；

**💡 创新点**

创新点在于：①基于中间层键向量均值的内容驱动排序，使合并结果在任何输入排列下字节相同；②将缓存视为内容地址集合，采用CvRDT的集合并与确定性渲染，实现重复交付吸收和可合并；

**🔧 技术方法**

使用的技术包括RoPE重编码、内容度量（均值K‑范数）排序、内容地址哈希、集合并的状态‑CRDT、对Qwen3模型的KV‑缓存操作与渲染算法；

**📊 数据集**

实验数据集为：自定义的分区推理基准（100道问题，5个问题族）、HotpotQA bridge‑k=2（50道问题）、以及Qwen3-1.7B/4B模型的KV缓存；

**📈 对比分析**

与BagMerge（传统concat+RoPE）和PackLLM（输出级融合）进行比较。CanonicalMerge在所有4个实验格子中均保持在最佳固定顺序的±4个百分点内，并在与PackLLM的对比中实现了约45个百分点的准确度提升；

**⚠️ 局限性**

局限性包括：仅在k=2时满足收敛性，k>2需额外模型重处理；仅在字节级一致的环境下（同硬件、相同权重）保证相同结果；不处理语义近似重复；对路由层的选择仍需经验调优；实验范围仅限Qwen系列模型和有限任务。

---

## 15. Embedding Inference Attack

**arXiv ID:** 2607.01276 | [PDF](https://arxiv.org/pdf/2607.01276v1)

**作者:** Cedric Fitiavana Raelijohn `[一作]` (Université du Québec à Montréal), Jean-Francois Rajotte `[通讯]` (Université du Québec à Montréal)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种黑盒下的Embedding Inference Attack（EIA），通过仅观察检索结果集来识别信息检索系统使用的嵌入模型。

**💡 创新点**

首次实现了仅凭检索文档集合而非嵌入向量即可判别嵌入模型的攻击，并设计了针对性对抗性查询（随机字符串、非查询、探测查询）以提升判别效率。

**🔧 技术方法**

使用LLM生成查询、Jaccard相似度计算文档集差异、对抗性查询生成与消除技术、以及在RAG框架下的评估。

**📊 数据集**

主要使用MS MARCO passage开发集和Alloprof教育问答数据集（共约8.8M文档和7400条评估文档）。

**📈 对比分析**

在控制的IR流水线和AnythingLLM RAG系统上进行对比实验，探讨k值、reranker与相似度阈值对攻击效果的影响。实验表明：k=3时，约1249个生成查询即可成功区分13个模型；k增大或使用reranker可进一步提升成功率；相似度阈值0.6以上显著降低攻击，但不同模型对阈值的敏感性不同。

**⚠️ 局限性**

实验仅覆盖开源嵌入模型，未考虑文档分块大小/重叠、关闭源/专有模型的适用性，以及攻击对检索质量的整体影响。

---

## 16. Adaptive Companionship for Group-Following Robots: Handling Dynamically Changing Group Formations

**arXiv ID:** 2607.01287 | [PDF](https://arxiv.org/pdf/2607.01287v1)

**作者:** Cong-Thanh Vu `[一作]` (National Cheng Kung University), Yen-Chen Liu `[通讯]` (National Cheng Kung University)

**通讯引用:** 2963 | [OpenAlex ID](https://openalex.org/A5057832539)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉-语言模型（VLM）与MPPI控制器的自适应人群陪伴机器人方法，能够在多人的动态队形中实时识别群体成员并选择合适的陪伴位置。

**💡 创新点**

创新点在于：①利用VLM对群体成员进行语义推理，实现对成员加入/离开的即时检测；②构建网格化互动空间并通过链式思维（CoT）提示让VLM在有限候选位置中推理；③将VLM推理结果与MPPI+CBF控制器耦合，兼顾安全性与社交距离。

**🔧 技术方法**

核心技术包括：3D LiDAR + PointPillars 进行人群检测与轨迹提取；VLM（Gemini 2.5 Flash Lite 为主）配合CoT提示进行群体识别与位置推理；网格化互动空间作为 VLM 输入；MPPI 控制器配合控制障碍函数（CBF）实现路径规划与碰撞规避。

**📊 数据集**

使用内部构造的五个对比场景（群体规模3–10人、障碍物、成员变动）以及20名受试者的用户实验数据，LiDAR 采集的点云作为感知输入；未引用公开数据集。

**📈 对比分析**

与单人跟踪基线 MPPI、侧位陪伴方法 ESFM、领队选择法 PP 以及基线 VLM 版本进行对比；实验显示成功率提升至少15%，碰撞率降低约25%，舒适距离保持在0.87–1.1 m，且在用户体验评估中获得最高的舒适、社交与智能分数。

**⚠️ 局限性**

局限性包括：VLM 推理延迟随群体规模增长而升高，超过约7人后成功率显著下降；依赖 LiDAR 覆盖范围与视觉-语言模型的准确性；目前未在大规模拥挤环境中验证；未对 VLM 进行任务专门微调，导致在大群体情形下精度下降。

---

## 17. The Benchmark Ceiling: Human Judgment, Evaluation Scarcity, and the Political Economy of AI Capability Measurement

**arXiv ID:** 2607.01254 | [PDF](https://arxiv.org/pdf/2607.01254v1)

**作者:** Mark Esposito `[一作]`, Ali Ansari `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统分析了AI基准评估在技术发展与治理中的核心作用，提出“基准天花板”概念，阐述了随着模型性能提升基准信号的衰减及其对评估有效性的影响，并通过正式的信号衰退模型与基于专业评估者薪酬的数据验证了人类高阶评估劳动的稀缺性与治理结构的匹配。

**💡 创新点**

创新点在于将基准有效性视为可耗尽的公共品，揭示评估劳动的右尾稀缺性，并将经济学的信号衰退与人力资源稀缺性结合，提出对基准治理的制度性改进建议；同时提供了基准有效性与模型进步之间的数学关系。

**🔧 技术方法**

采用了信号衰退的经济学模型、项目评估理论以及项目管理视角的定量分析；同时借鉴项目评估（Item Response Theory）框架来说明硬尾基准项的生成与评估者能力的匹配。

**📊 数据集**

使用了微观平台micro1的数据，覆盖1000+具有专业资质的评估者，包含任务难度、完成时间、质量分数和报酬信息。

**📈 对比分析**

通过比较不同难度层级任务的工资溢价、基准的污染程度与信号衰减速度，证明了硬尾任务工资溢价显著高于低难度任务，并显示模型性能提升导致基准有效性递减的实证关系；在性能评估层面，硬尾任务的工资梯度和模型在硬尾上的得分下降均表明基准的信号变弱。

**⚠️ 局限性**

限制包括：模型假设简化了实际评估过程；micro1样本仅涵盖少数专业领域，缺乏跨领域和纵向跟踪；未对不同基准族群的社会价值进行量化；以及对基准污染和战略优化机制的具体定量估计仍待进一步研究。

---

## 18. Prompt Framing Distorts Count-Based Evaluation of LLM Error Detection: Evidence from Numeric Anchoring

**arXiv ID:** 2607.01240 | [PDF](https://arxiv.org/pdf/2607.01240v1)

**作者:** Dekun Yang `[一作]` (Zhejiang University), Dekun Yang `[通讯]` (Zhejiang University)

**通讯引用:** 6534 | [OpenAlex ID](https://openalex.org/A5046455995)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大语言模型在错误检测任务中的计数准确性与span定位的差异进行系统评估，构建了 ErrorBench 压力测试协议，检验数值锚定（anchor）对 F1 计数指标的影响。

**💡 创新点**

提出 F1 Inflation 概念，首次用 Controlled Stress‑Test 方式量化提示中锚定数值如何导致计数 F1 与 span F1 的巨大差距；引入 Anchoring Sensitivity Index（ASI）诊断模型对锚定的敏感度，并在六大模型、五种提示条件下提供公开基准。

**🔧 技术方法**

使用 CoNLL‑2014 M2 scorer、ERRANT 3.0.0 进行 span‑aware 评估；通过自然语言错误描述自动抽取编辑；计算 Count Bias、Count‑F1、ASI；配对 t‑检验与 Benjamini‑Hochberg FDR 进行统计显著性检验；利用多参考评分验证结果稳健性。

**📊 数据集**

基于 CoNLL‑2014 学习者英语作文数据，挑选 3–7 条错误的四句段落共 143 段；并在 100 段子集上使用 ERRANT 进行多参考复现。

**📈 对比分析**

将 Count‑F1 与 CoNLL‑2014 M2 overlap F0.5 对比，发现 Anchored 条件下 Count‑F1 可提升 0.26–0.79，但 M2 F0.5 仅提升 0.01–0.05，F1 Inflation 达 0.79（overlap）或 0.96（strict）。不同模型表现差异显著：GPT‑5.4、Claude 系列对锚定高度敏感（ASI>1），Gemini 系列对锚定几乎不敏感（ASI≈0）。

**⚠️ 局限性**

局限性包括：仅在 temperature=0、单次推理；锚定偏移固定为 ±2，未绘制 dose‑response 曲线；仅评估 CoNLL‑2014，难以推广到代码审计、事实检查等其他领域；描述提取的 span 可能低估真实定位精度；Gemini 的低 ASI 可能源于先验低报错倾向，未能完全分离；未考虑多样化提示或温度变化；结果受模型 API 版本变动影响。

---

## 19. RuleChef: Grounding LLM Task Knowledge in Human-Editable Rules

**arXiv ID:** 2607.01293 | [PDF](https://arxiv.org/pdf/2607.01293v1)

**作者:** Ádám Kovács `[一作]` (KR Labs), Gábor Recski `[通讯]` (KR Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RuleChef 框架，利用 LLM 在学习阶段自动生成并迭代改进可执行规则，供 NLP 任务（如 NER、文本分类）推理时使用。

**💡 创新点**

将 LLM 的生成能力与离线验证循环相结合，构建可解释、高精度、低延迟的规则系统，并支持人类反馈和无监督观测模式。

**🔧 技术方法**

使用大语言模型（如 Kimi‑K2）生成正则/ spaCy 规则，配合聚类失败、规则验证、Wilson 下限、代理协调等技术。

**📊 数据集**

在文本匿名化基准 TAB（欧盟人权法院判决）和 Banking77（银行业务意图分类）上进行实验。

**📈 对比分析**

与直接 LLM 调用、GLiNER2 以及几种基线进行对比，规则系统在格式类实体上 F1 与 LLM 相近，整体精度高但召回略低；在规则学习过程中仅需约 12 分钟、20 次 LLM 调用。

**⚠️ 局限性**

实验基于单次运行，结果波动 ±3 分，仅在英文数据和单一 LLM 上验证，缺乏多模型、多语言及偏差评估。

---

## 20. ExPerT: Personalizing LLM Responses to Users' Domain Expertise via Query-Wise Semantic and Keystroke Behavioral Cues

**arXiv ID:** 2607.01242 | [PDF](https://arxiv.org/pdf/2607.01242v1)

**作者:** Yeji Park `[一作]` (Ulsan National Institute of Science and Technology), Taesik Gong `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5057451323)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了 ExPerT，一种基于查询级语义与键盘打字行为的动态个性化框架，用以推断用户当前专业水平并生成与之匹配的 LLM 回答。

**💡 创新点**

创新点在于首次将语义内容与键盘行为联合用于每个查询的专业水平推断，并通过少量示例的上下文学习实现两阶段的专业条件生成，使回答在深度、术语和概念复杂度上随用户专业水平自适应。

**🔧 技术方法**

主要技术包括键盘按键间隔、按键保持时长、backspace 计数、打字速度等行为特征的词级聚合；在 GPT‑4 上使用结构化少量示例的 in‑context learning 进行专业推断与生成；两阶段模型（专业推断 + 专业条件生成）与自定义提示模板。

**📊 数据集**

实验采用 40 名参与者（化学、计算机科学、商业三领域）共 1270 条查询，记录查询文本、键盘动态、专家自评与满意度；数据集已公开于 GitHub。

**📈 对比分析**

与随机、Persona、Session、IDL、AI Persona、单独语义/行为等基线对比，ExPerT 的 MAE 为 0.398、MSE 0.698，较最强基线 IDL 降低 65.7% MAE；用户满意度提升 17.5%（从 3.71 直至 4.36）；跨 GPT‑4、Claude Sonnet 4、Gemini 2.5 Pro 的评估亦显示一致优于单独特征。

**⚠️ 局限性**

局限性包括：高度依赖提示设计与 few‑shot 示例，跨用户或冷启动时鲁棒性下降；键盘动态易受疲劳、键盘/设备差异等噪声影响；隐私、可访问性及潜在偏见问题；样本规模与自评标签可能限制了泛化能力。

---

## 21. SPARCLE: SPeaker-aware Aligned Representations via Contrastive Language Embeddings

**arXiv ID:** 2607.01238 | [PDF](https://arxiv.org/pdf/2607.01238v1)

**作者:** Priyam Mazumdar `[一作]` (University of Illinois Urbana-Champaign), Volodymyr Kindratenko `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了SPARCLE模型，用对比学习将字符（grapheme）与对应的Wav2Vec2声学表示对齐，并将说话人信息作为条件，作为G2P模块的替代品直接用于TTS后端。

**💡 创新点**

创新点在于：①利用字符级对比预训练捕获细粒度、上下文敏感的发音信息；②将说话人声纹嵌入作为条件，提升低资源多说话人场景下的发音一致性；③对比预训练与下游TTS的部分微调策略实现最佳适应-遗忘平衡。

**🔧 技术方法**

技术包括字符级Transformer编码器（带邻域卷积），FaCodec声纹嵌入，Wav2Vec2声学特征的注意力池化，对比损失（温度为0.1），以及对TTS后端ParrotTTS和VITS的整合。

**📊 数据集**

使用LibriSpeech‑960h进行预训练，VCTK‑v0.92用于下游TTS微调与评估，数据集覆盖多说话人且在域间存在差异。

**📈 对比分析**

与传统字符基线和基于G2P的基线对比；在极低资源（10 min–5 h）时，SPARCLE将WER从≈85%下降至≈42%，在1 h时降至≈7.5%；EER亦从≈1.56%提升至≈1.13%；在VITS后端仅能略微改善WER，但显著提升EER。

**⚠️ 局限性**

限制：对齐时需要准确的字符–声学映射；对话上下文外的多音素歧义仍存在；对跨语言和跨口音的泛化需进一步验证；在全微调时可能出现“遗忘”预训练知识。

---

## 22. Safeguarding LLM Agents from Misalignment through Provenance Analysis

**arXiv ID:** 2607.01236 | [PDF](https://arxiv.org/pdf/2607.01236v1)

**作者:** Yining She `[一作]` (Carnegie Mellon University), Eunsuk Kang `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于 provenance（源追踪）的框架和多阶段管道，来检测并阻止 LLM 代理在调用外部工具前产生与用户意图不一致的行为（misalignment）。

**💡 创新点**

创新点在于：①将 misalignment 视为缺乏可追溯证据的行为；②构造了工具级、参数级、解释级三类 misalignment 的 provenance 关系；③设计了三阶段判定管道，将 LLM 作为辅助推理器而非单一判断者；④在多 LLM 上验证其可迁移性。

**🔧 技术方法**

技术上主要使用：1) 基于 W3C PROV 的 provenance 关系定义；2) LLM 进行关系推理（如检索工具相关性、参数可导出性、是否存在多可行解释）；3) 多阶段规则化检查（工具适用性、参数可追溯性、解释唯一性）。

**📊 数据集**

使用了两个公开基准：Agent‑SafetyBench（侧重 underspecification）和 WorkBench（包含工具调用与参数误用），并自行生成并标注了 Agent‑SafetyBench 的执行轨迹。

**📈 对比分析**

与传统 LLM-as-a-judge 基线和单步 Provenance Prompt 进行对比，实验表明：在 Agent‑SafetyBench 上多阶段管道将误判率从 42.9% 降至 1.8%，在 WorkBench 上从 32.1% 降至 17.3%；同时在成功轨迹上的干预率显著降低（例如 30.5%→12.8%），而对已对齐轨迹的误干预几乎无显著提升。

**⚠️ 局限性**

局限性包括：①无法检测生成型参数的对齐；②对代理自身计划的依赖，若计划本身误导仍可能误判；③多阶段设计增加推理延迟和算力消耗；④实验样本中已对齐轨迹样本不足，统计显著性受限。

---

## 23. PACE: A Neuro-Symbolic Framework for Plausible and Actionable Counterfactual Explanations

**arXiv ID:** 2607.01306 | [PDF](https://arxiv.org/pdf/2607.01306v1)

**作者:** Pavel Iakovets `[一作]` (University of Klagenfurt), Fadi Al Machot `[通讯]` (Norwegian University of Life Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了PACE框架，用神经网络与符号推理相结合生成可行的反事实解释。

**💡 创新点**

将可行性约束显式编码为符号规则，并通过ASP在可行空间中系统搜索，确保生成的解释满足领域知识。

**🔧 技术方法**

使用多层感知机作为预测器，结合Answer Set Programming（ASP）进行符号推理。

**📊 数据集**

在Adult Income数据集上进行实验。

**📈 对比分析**

与Random Search、DiCE、Wachter式、VCNet式、C‑CHVAE式等基线方法对比，PACE在可行性方面始终满足约束，平均修改特征数仅1.24，尽管在有效率略低，但可行性和最小化优势显著。

**⚠️ 局限性**

实验仅覆盖单一基准数据集，约束手工制定，缺乏自动约束学习，且对更复杂的关系或时间约束支持有限。

---

## 24. HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching

**arXiv ID:** 2607.01299 | [PDF](https://arxiv.org/pdf/2607.01299v1)

**作者:** Yifei Liu `[一作]` (Xiaohongshu Inc.), Weihang Chen `[通讯]` (Xiaohongshu Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对混合注意力LLM的位移无关缓存（PIC）系统，使得预填充阶段可大幅加速

**💡 创新点**

创新点包括：1) 为线性注意力层缓存并利用段累计转换算子实现常数时间状态拼接；2) 在全注意力层使用小的边界重合窗口修复跨段注意力偏差；3) 通过段级自包含性实现跨实例并行预填充，显著降低尾部延迟

**🔧 技术方法**

使用的技术包括：段累计转换算子缓存、常数时间状态组合公式、边界重合窗口重计算、分段并行调度（LPT策略）以及SGLang+Triton实现

**📊 数据集**

在四个生产级混合注意力模型（Ring-mini、Ring-flash、Qwen3.5-35B-A3B、Qwen3.5-122B-A10B）上，评估了四类公开数据集（HotpotQA、TriviaQA、多文档摘要 MultiNews、GovReport）和一个真实生产RAG日志

**📈 对比分析**

相较于完整重计算和传统前缀缓存，系统在所有模型-数据集组合上平均将首个 token 延迟（TTFT）降低约2.45×、峰值吞吐量提升约2.0×，且任务准确率误差控制在3.3分以内；在全缓存缺失时，8个worker并行预填充可实现6.1×的TTFT加速

**⚠️ 局限性**

限制包括：仍需为每个段存储段累计转换算子，且全注意力层仍需重计算边界窗口；在极深层模型或高段数情形下，累计状态漂移可能略大；并且该方案主要针对预填充阶段，对解码阶段的并行性提升有限

---

## 25. IonSense-QKG: A Quantum-Readiness Metadata Framework for Lithium-Ion Battery Dataset Discovery

**arXiv ID:** 2607.01286 | [PDF](https://arxiv.org/pdf/2607.01286v1)

**作者:** Sakthi Prabhu Gunasekar `[一作]` (Amrita Vishwa Vidyapeetham), Prasanna Kumar Rangarajan `[通讯]` (Amrita Vishwa Vidyapeetham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了 IonSense-QKG 框架，对公开的锂离子电池数据集进行量子可用性元数据标注，并通过加权 Quantum Readiness Score（QRS）对数据集进行优先级排序，辅助混合量子-经典机器学习实验的选取与设计。

**💡 创新点**

首次提出面向量子计算的电池数据集元数据架构与可解释的加权评分体系，填补了电池数据与量子研究之间的中间层，提供可查询的量子实验优先级与编码建议，促进数据驱动的量子电池分析。

**🔧 技术方法**

采用知识图谱兼容的元数据模式、SQL 风格查询、Python 实现的 QRS 计算脚本与稳健性评估、链接校验脚本，并结合手工/辅助注释技术完成数据集扩充。

**📊 数据集**

以 EV‑Battery‑IonSense 为种子，对 15 个代表性公开锂离子电池数据集（如 Impedance‑Based Forecasting、PulseBat、Voltage Relaxation、WMG‑DIB EIS、Stanford‑MIT Early Cycle Life、NASA、OSF Magnetometry、Home‑Storage Field Measurements、Battery Imaging Library 等）进行手工/辅助量子相关元数据标注。

**📈 对比分析**

通过定义六项加权指标（特征紧凑度、序列适配性、模态兼容性、标签可用性、预处理可行性、访问可复现性）计算 QRS，并在 2000 次高斯权重扰动实验中验证排名稳健性；高分数据集被视为近端量子实验可行的候选集，但尚未开展实际量子模型实验，因而未给出性能指标。

**⚠️ 局限性**

评分基于专家手工定义的离散标注，缺乏真实量子实验验证；注释工作需人工完成，易受主观偏差；数据集访问许可与预处理复杂度随时间变化；评分与实际量子模型性能的相关性尚未验证。

---

## 26. Artificial Intelligence-Enabled Accounting Information Systems and Fraud Detection in Nigeria's Financial Services Sector: The Moderating Role of Natural Language Processing

**arXiv ID:** 2607.01257 | [PDF](https://arxiv.org/pdf/2607.01257v1)

**作者:** Timothy Oluwapelumi Adeyemi `[一作]` (WeAreGenius Research Institute), Abigail Omotola Ojogbede `[通讯]` (Park University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究考察了在尼日利亚金融服务业中，AI驱动的会计信息系统（AIS）对审计与欺诈检测效能的影响，并评估自然语言处理（NLP）在此关系中的调节作用。

**💡 创新点**

创新点在于首次将AI-enabled AIS与NLP整合为协同框架，并通过欺诈钻石理论与技术接受模型在尼日利亚环境下进行实证检验，揭示NLP对AI系统解释性与检测性能的提升作用。

**🔧 技术方法**

采用机器学习算法、预测分析、异常检测以及NLP技术（语义解析、情感分析与文本分类）构建的智能审计系统与文本解释机制。

**📊 数据集**

使用来自186名在银行、保险与FinTech机构工作的专业人士填写的问卷数据，涵盖AIS功能、NLP能力与审计欺诈检测效能等维度。

**📈 对比分析**

通过分层层次调节回归验证，AI-enabled AIS对审计效能的R²为0.626，加入NLP后提升至0.632，且防范功能（β=0.384）成为最强预测因子，显示调节效应显著且提升显著。

**⚠️ 局限性**

局限性包括采用横断面自评问卷，缺乏纵向或客观交易数据，样本局限于尼日利亚金融机构，且可能存在共同方法偏差与外部有效性受限。

---

## 27. Folding an e-graph in pure egglog

**arXiv ID:** 2607.01249 | [PDF](https://arxiv.org/pdf/2607.01249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 28. How Indian Dermatologists are Utilizing Artificial Intelligence for Clinical Practice and Workflow Management: A Nationwide Survey with a Special Focus on atopic dermatitis

**arXiv ID:** 2607.01252 | [PDF](https://arxiv.org/pdf/2607.01252v1)

**作者:** Dipayan Sengupta `[一作]` (Charnock Hospital), Narayanan B `[通讯]` (Sree Balaji Medical College and Hospital)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该研究通过全国性横断面调查，评估印度皮肤科医生在慢性疾病管理（尤其是特应性皮炎）中的临床瓶颈，并调查其对人工智能（AI）工具的使用情况。

**💡 创新点**

创新之处在于首次采用“问题优先”客户开发框架，系统映射临床痛点与AI采用之间的关系，揭示AI在印度皮肤科中主要被用作认知和行政支持，而非传统的诊断辅助。

**🔧 技术方法**

研究采用在线问卷收集数据，随后使用描述性统计、Pearson 卡方检验、Benjamini‑Hochberg FDR 纠正以及多元逻辑回归等统计技术进行分析。

**📊 数据集**

使用的数据集为377名印度皮肤科医生的问卷响应，没有利用公开医学影像或其他临床数据库。

**📈 对比分析**

通过对比AI使用者与非使用者，并在多元回归中控制临床经验和学术隶属，发现AI使用者对患者自我误诊与非皮肤科医生使用AI的担忧显著增高（aOR=2.25, p=0.0003），表明AI对临床工作有实质影响。

**⚠️ 局限性**

局限性包括非概率抽样、响应率低（约8%）、缺乏真实临床干预验证以及横断面设计无法确定因果关系。

---

## 29. Collaborative Disagreement Resolution for Scalable Oversight

**arXiv ID:** 2607.01251 | [PDF](https://arxiv.org/pdf/2607.01251v1)

**作者:** Yuyang Jiang `[一作]` (University of Chicago), Chenhao Tan `[通讯]` (University of Chicago)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“Disagreement Resolution（DR）”协议，将传统的AI对抗式辩论转变为协作式真相追求，从而实现可扩展的监督机制。

**💡 创新点**

创新点在于：1）将人类调解与冲突解决的策略自动化到AI对话中；2）通过识别“crux”（争议核心）并让模型在对话中不断更新信念，降低弱判官需要直接评估复杂论点的负担；3）理论证明在能力不对称条件下，DR能优于传统Debate。

**🔧 技术方法**

主要技术：多轮协作式对话框架、crux识别映射、信念更新与行动决策（保留/接受）、实验使用的大模型（GPT‑4o、Claude Sonnet 4、GLM‑4.6、Kimi K2）与弱判官（GPT‑4o‑mini、Gemma‑3‑4B），以及对话转录的判官评估。

**📊 数据集**

使用了三大专家级基准：GPQA、SuperGPQA 以及 Humanity’s Last Exam (HLE‑MC)，全部在模型自然产生分歧的实例上评估。

**📈 对比分析**

比较方法：对比三种协议（Debate、DR、Double Consultancy）以及无咨询的Naive Judge。结果显示：在弱判官情形下，DR平均提升 12.9%（相对Debate），在所有 12 条件中 10 条DR更优；在强判官时提升有限甚至略降；与Double Consultancy相比，DR 通常更好，除非任务极其复杂。

**⚠️ 局限性**

限制：1）仅在推理阶段测试，未探究训练时效用；2）弱判官仅被动参与，未评估主动判官可能的改进；3）只考虑自然分歧场景，未保证每个争议都有明确定义的真值；4）仍存在“Agreement Trap”风险，需进一步设计安全防护。

---

## 30. Structuring the Space of Sociotechnical Alignment

**arXiv ID:** 2607.01250 | [PDF](https://arxiv.org/pdf/2607.01250v1)

**作者:** Esra Dönmez `[一作]` (University of Stuttgart), Agnieszka Falenska `[通讯]` (University of Stuttgart)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2022-2025年ACL Anthology中关于社会技术对齐的论文进行系统综述，提出四维框架（对齐目标、规范概念、对齐方法、理论框架），并用该框架分析文献中的规范化与实现缺失。

**💡 创新点**

创新点在于：①首次把社会技术对齐拆解为四个维度并提供理论映射；②结合自动检索、主题建模与人工标注构建详尽语料库；③揭示文献中对齐目标、规范概念和目标人群的常见不足，提出可操作的改进建议。

**🔧 技术方法**

采用的技术主要包括关键词检索、基于BERTopic与UMAP的主题建模、TF-IDF + NMF聚类、人工双人/三人标注以及对齐目标与规范概念的结构化分析。

**📊 数据集**

使用的数据集为ACL Anthology 2022-2025年发表的论文共1089篇（过滤后281篇），并通过手工标注得到对齐目标标签和主题聚类。

**📈 对比分析**

通过对齐目标分布与主题聚类的可视化对比，展示了文献中对齐目标的多样性与偏向（如安全、偏见、个性化等），但因是综述研究未涉及模型性能评估；主要评估为方法覆盖度和理论完整性。

**⚠️ 局限性**

限制主要包括：①检索仅基于英文关键词，可能遗漏非英语或不同表述的工作；②依赖自动聚类和手工标注，存在主观偏差；③聚焦ACL社区，未覆盖跨学科或非ACL期刊的相关研究；④未对模型性能做实验评估。

---

## 31. LLMs as Teaching Assistants for Mathematics Exam Grading: Reliability, and Practical Usability

**arXiv ID:** 2607.01247 | [PDF](https://arxiv.org/pdf/2607.01247v1)

**作者:** Aastha Sapkota `[一作]` (University of Wisconsin - Green Bay), M. G. Sarwar Murshed `[通讯]` (University of Wisconsin - Green Bay)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估六种大型语言模型（Gemini 3.1 Pro、Gemini 3.5 Flash、ChatGPT 5.5 Pro、ChatGPT 5.5 Thinking、Claude Pro Opus、Claude Sonnet）作为本科离散数学开放式考试的评分助手。

**💡 创新点**

提出严格与自由两种评分策略来研究LLM在部分学分判定中的表现，并设计可审计的评分工作流和交叉校验机制。

**🔧 技术方法**

利用Prompt工程、基于大语言模型的分步评分、交叉校验审计以及结构化反馈生成技术。

**📊 数据集**

使用28名学生完成的七道离散数学题目组成的开放式考试数据集。

**📈 对比分析**

通过MAE、RMSE、NRMSE、Pearson相关和Exact一致率等指标比较，结果显示自由评分在所有模型均降低误差；ChatGPT 5.5 Thinking在题级MAE最低，Gemini 3.1 Pro在总分MAE最低，但两者在排名保持上并不完全一致。

**⚠️ 局限性**

缺乏高Exact匹配、需人工复核；仅针对单门课程、缺乏跨学科和跨语言验证；不同接口的稳定性和文件处理差异仍需进一步研究。

---

## 32. Office Comprehension Benchmark

**arXiv ID:** 2607.01245 | [PDF](https://arxiv.org/pdf/2607.01245v1)

**作者:** Firoz Shaik `[一作]` (Microsoft), Vishal Chowdhary `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍了 Office Comprehension Bench，评估 Word/Excel/PowerPoint 原生文件的理解，包括 File Fidelity Q&A 与 Domain Q&A 两个轨道。

**💡 创新点**

首次公开联合评测原生 Office 文件的基准，使用原子断言与三模型多数投票的 LLM-as-a-judge 方法，并揭示系统性能上限。

**🔧 技术方法**

采用原子断言分解、三模型投票评审、文件解析与多步推理、思考深度/层级对比，以及统计误差分解等技术。

**📊 数据集**

构建了包含 240 个文件、902 个查询的 File Fidelity 轨道（Word 78、Excel 116、PowerPoint 46）以及 64 个文件、120 个查询的 Domain Q&A 轨道，涵盖 12 行业真实文件（SEC、政府、教育等）。

**📈 对比分析**

通过每个断言的二值评分计算 assertion‑level accuracy；Domain Q&A 结果显示 GPT‑5.5 Thinking 59.3%、Claude Opus 4.7 56.8%、Gemini 3.1 45.7%；File Fidelity 轨道中 GPT‑5.5 在 Excel 领先，Claude 在 Word/PowerPoint 优秀，整体均超过人类单评测基准。

**⚠️ 局限性**

仅单轮评测、仅关注理解不含编辑或工作流、混合文件覆盖有限、缺少多语言支持、断言质量与人类标注一致性未充分验证、Domain Q&A 缺乏人类基准、仅使用英文等局限。

---

## 33. STRUCTSURVEY: Structured Agentic Retrieval for Automated Survey Paper Generation

**arXiv ID:** 2607.01243 | [PDF](https://arxiv.org/pdf/2607.01243v1)

**作者:** Paolo Pedinotti `[一作]` (Bloomberg), Enrico Santus `[通讯]` (Bloomberg)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了 StructSurvey，一个层级多代理框架，利用结构化检索构建和重用领域图来指导科学综述论文的规划、检索与撰写。

**💡 创新点**

创新点在于把结构推理从生成阶段转移到检索阶段，动态构造图结构并在整个写作过程中持续更新与重用，从而提供可解释的概念、方法和分类支撑。

**🔧 技术方法**

采用 LLM 驱动的规划、检索与写作代理；结构化提取工具（实体、关系、双向图等）；图融合与序列化；层级多代理架构；以及 LLM‑as‑a‑Judge 评估协议。

**📊 数据集**

使用了自建的 ACL Survey 数据集（33 篇 2018‑2025 年的综述论文及其公开引用），以及对应的引用论文摘要作为检索语料。

**📈 对比分析**

在与仅使用向量检索的 SurveyForge 基线相同的多代理框架下进行对比，评估指标包括 ROUGE‑1/2 的精度、召回与 F1，以及 LLM‑as‑a‑Judge 的逻辑结构、深度与综合等维度；结果显示 StructSurvey 的 ROUGE‑1 召回提升 2.9、ROUGE‑2 召回提升 1.0，且在逻辑结构、深度和综合评分上均优于基线，精度基本保持不变。

**⚠️ 局限性**

主要局限包括：1）结构化检索需要额外的 LLM 调用，计算成本高；2）评估仍受 ROUGE 与 LLM‑as‑a‑Judge 的局限，缺乏人类专家评审；3）实验仅在引用语料和摘要层面进行，未覆盖开放检索与全文检索；4）生成的综述在批判性分析方面表现仍不理想。

---

## 34. Kara: Efficient Reasoning LLM Serving via Sliding-Window KV Cache Compression

**arXiv ID:** 2607.01237 | [PDF](https://arxiv.org/pdf/2607.01237v1)

**作者:** Shen Han `[一作]`, Yuyang Wu `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于滑动窗口的KV缓存压缩方法Kara，并将其集成到vLLM的PagedAttention中，进一步构建了KvLLM推理框架，提供了周期性压缩策略以提升多路并发推理吞吐量。

**💡 创新点**

创新点在于：①使用滑动窗口双向注意力对KV对进行重要性评分，避免全缓存阈值压缩导致的频繁压缩和信息损失；②引入Token2Chunk模块，将离散保留的KV对扩展为灵活大小的连续块，提升语义保留能力；③在KvLLM中采用周期性压缩与PagedAttention对齐，显著提升吞吐量和并发性能。

**🔧 技术方法**

主要技术包括滑动窗口双向注意力评分、Token2Chunk块扩展、周期性压缩调度、PagedAttention块管理、Recomputation获取查询状态，以及在vLLM框架下实现的KV压缩与推理优化。

**📊 数据集**

实验数据集涵盖：MATH‑500、AIME24、AMC23数学推理基准；Needle‑in‑a‑Haystack（NIAH）长上下文检索基准；并使用DeepSeek‑R1‑Distill‑Llama‑8B、Qwen3‑14B、Qwen3‑4B三种大模型进行评估。

**📈 对比分析**

与SnapKV、ChunkKV、StreamingLLM、PyramidKV、AdaKV等主流压缩方法对比，Kara在相同KV保留比例下保持或提升pass@1准确率（如MATH‑500 30%保留保持接近原始准确率），在NIAH检索任务中也取得更高的检索准确率；在KvLLM框架下，周期性压缩显著提升吞吐量并保持高并发性能，低于阈值压缩导致的吞吐量下降。

**⚠️ 局限性**

局限性包括：①仍采用阈值触发的压缩周期，可能在极大并发场景下导致不必要的压缩开销；②Token2Chunk的块长度和预算参数需要手工调优，过小会限制语义保留；③对极长上下文或极低保留比例的情况下，信息损失仍不可避免；④对不同模型架构的兼容性和通用性还有待进一步验证。

---

## 35. Black-Box Inference of LLM Architectural Properties with Restrictive API Access

**arXiv ID:** 2607.01313 | [PDF](https://arxiv.org/pdf/2607.01313v1)

**作者:** Christopher Ellis `[一作]` (Carnegie Mellon University), José M. F. Moura `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在仅能获取单一输出词概率和整体响应时延的LLM API下，利用共用词集提示和时延侧信道技术，推断Transformer模型的隐藏维度、层数和参数量。

**💡 创新点**

创新点在于：①不依赖于传统的top‑k logits或logit‑bias接口；②通过共用词集提示构建稀疏对数概率矩阵并用谱分析恢复隐藏维度；③利用预填充阶段的时延比例关系联合推断层数和参数量。

**🔧 技术方法**

核心技术包括：共用词集提示（Common‑Set Prompting）、谱估计（利用矩阵秩与特征值拐点检测）、时延侧信道攻击（基于预填充阶段的 O(L·ℓ²·d) 计算模型）以及线性回归校准。

**📊 数据集**

使用了32个开源LLM（覆盖从 135 M 到 30 B 参数、隐藏维度 576–4096、层数 16–48 的 Dense 与 Mixture‑of‑Experts 体系）作为评测数据集，并在 HuggingFace + vLLM 环境中收集日志概率和时延。

**📈 对比分析**

与现有使用 top‑k logits/ logit‑bias 的攻击相比，本方法在相同准确度下需要约 1–2.3 量级更多的 token（约 33–190 倍），但仍能在大模型（≥3 B 参数）下将隐藏维度、层数和参数量恢复到 40–65% 的相对误差；在小模型上误差可超过 100%。

**⚠️ 局限性**

主要局限：①对大型模型的推断需极高 token 预算（10¹¹–10¹²），导致成本显著；②仅验证 Dense 与 Mixture‑of‑Experts 结构，对状态空间或混合模型缺乏评估；③时延估计依赖预填充主导且需已知服务堆栈，对部署环境变化不稳健。

---

## 36. The Three Dimensions of ROS 2 Middleware

**arXiv ID:** 2607.01304 | [PDF](https://arxiv.org/pdf/2607.01304v1)

**作者:** Sanghoon Lee `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Kyung-Joon Park `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对ROS 2中Middleware（DDS与Zenoh）的架构与运行动态进行系统化综述，提出三维结构框架（Space、Time、State），并将已有94篇相关研究映射至该框架，揭示三维之间的结构性冲突与权衡，进一步给出未来研究路线。

**💡 创新点**

创新点在于：①将ROS 2 Middleware的设计问题抽象为三维结构框架；②系统地将Space、Time、State三维的冲突与相互作用进行分类与分析；③对已发表的研究进行统一归类，构建研究的冲突-解决路径图，形成一套完整的研究路线图。

**🔧 技术方法**

采用系统性文献综述方法、结构化分析与概念框架构建技术，对DDS与Zenoh的Discovery、Data Exchange与State Management三个运作动态进行抽象与映射，形成Space/Time/State维度与实现机制的对应关系。

**📊 数据集**

本研究未使用实验数据或标准数据集，全部依据公开论文、标准文档与开源实现的技术细节与性能报告进行汇总与分析。

**📈 对比分析**

文章主要通过对文献的归类与维度映射进行比较，未给出数值实验；但通过分析不同技术在Space、Time、State三维上的优劣，说明在无线、资源受限环境下多维度权衡的普遍性与必然性。

**⚠️ 局限性**

局限性包括：①仅聚焦DDS和Zenoh两种主流实现；②缺乏量化模型与实时实验验证；③对其他中间件（如iceoryx、micro‑ROS）和新兴协议（如TSN、QUIC）的覆盖不足；④对不同应用场景下具体性能影响的细粒度评估仍待进一步研究。

---

## 37. Cognitive Firewall: A Proactive, Zero-Trust, Multi-Gate Framework for LLM Safety

**arXiv ID:** 2607.01277 | [PDF](https://arxiv.org/pdf/2607.01277v1)

**作者:** Michele Guida `[一作]` (Roma Tre University), Noorbakhsh Amiri Golilarz `[通讯]` (University of Alabama)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Cognitive Firewall，利用四个独立门控模型在用户与受保护大型语言模型之间进行预生成监督与后生成评估，防止多轮、权限和分解式攻击；

**💡 创新点**

创新点在于将安全判断拆解为意图识别、零信任上下文验证、对话一致性检测和输出风险评估四类门控，并通过“升级”决策规则在任何门控发出危险信号时立即拦截，实现对话级的预先阻断与可审计性；

**🔧 技术方法**

技术包括：Qwen系列模型作为监督模型实现四门控；使用“意图门”对请求进行实体化归纳；“上下文门”对角色/权限声明做零信任校验；“一致性门”检测对话渐进式升级与拆解；“输出门”对生成回复做后置风险评估；所有门均采用分类标签而非分数；

**📊 数据集**

实验数据集包含四种攻击集（单轮直接请求 jbb、Crescendo多轮升级、ActorAttack多轮拆解、Human-crafted mhj）以及120条表面类似攻击的benign xstest；

**📈 对比分析**

与三种已发布的 per‑message 过滤器（Llama Guard、ShieldGemma、Granite Guardian）以及两种轨迹感知防火墙（THRD、TCA）对比；Cognitive Firewall 在单轮、双轮、权限与人造攻击上将成功率降低至1%–14%，大幅优于对手（最高可达74%），且过度拒绝率仅8%，低于其他防御；

**⚠️ 局限性**

局限性包括：样本量有限（50–100会话），评判依据单一相同家族的 Qwen 判别模型，未针对注入或多轮误判的边缘情况做充分测试，且四个门控均需额外计算与延迟，需进一步验证在更大规模、多样化任务上的泛化与稳健性。

---

## 38. Generative AI and Federated Learning for Intrusion Detection Systems: A Survey

**arXiv ID:** 2607.01305 | [PDF](https://arxiv.org/pdf/2607.01305v1)

**作者:** Jiefei Liu `[一作]` (New Mexico State University), Jayashree Harikumar `[通讯]` (DEVCOM Analysis Center)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对生成式 AI 与联邦学习在入侵检测系统（IDS）中的应用进行了系统综述，梳理了研究方向、技术分类、数据集与挑战。

**💡 创新点**

首次将 VAE、GAN、扩散模型、LLM 等生成式 AI 技术与联邦学习框架统一归类，提出了多维度的开放挑战与未来研究方向。

**🔧 技术方法**

综述了 Variational Autoencoders、Generative Adversarial Networks、Diffusion Models、Large Language Models 及其在 IDS 中的应用，讨论了 FedAvg 等联邦学习方法。

**📊 数据集**

参考的 IDS 数据集包括 KDD Cup 99、NSL‑KDD、UNSW‑NB15、CIC‑IDS2017、CIC‑DDoS2019、FLNET2023 等，强调了 FLNET2023 的联邦特性。

**📈 对比分析**

由于是综述性工作，未进行实验比较；作者对已有研究中的性能指标（准确率、召回率、对抗鲁棒性、通信成本等）进行了归纳与对比，指出生成数据对模型性能的提升与潜在负面影响。

**⚠️ 局限性**

主要限制包括：生成数据质量与真实网络行为匹配不足；联邦学习中非 IID 数据与通信开销挑战；攻击者利用生成模型制造对抗样本或投毒更新的风险；缺乏真实联邦场景下的可验证数据集与评估指标。

---

## 39. Scaling Laws for Grid-Based Approximate Nearest Neighbor Search in High Dimensions

**arXiv ID:** 2607.01283 | [PDF](https://arxiv.org/pdf/2607.01283v1)

**作者:** Matthew J Liu `[一作]`, Noah Flynn `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了多探测网格（multiprobe grid）在大规模高维数据上的近似最近邻搜索的 N（数据量）和 d（维度）缩放特性，并将其与图、树、量化等主流 ANN 方法做系统对比。

**💡 创新点**

发现多探测网格在高维下能保持近乎恒定的 d‑缩放指数，出现 d‑尺度交叉点，且在 N 缩放上近线性；提出了闭式理论模型预测 QPS 与召回率的对数线性关系；通过实验证明了其在高维和大规模数据下的竞争优势。

**🔧 技术方法**

采用多探测网格算法（网格划分+PCA 子空间 cell 选择），NSGA‑II 进行超参优化，闭式概率模型推导 QPS 与召回关系，使用 Python + NumPy/BLAS 实现；与 Voyager、PyNNDescent、Annoy、FAISS‑IVF 等基线进行对比。

**📊 数据集**

实验使用 GloVe‑200-angular、GloVe‑25/50/100/200-angular（维度 25–200）、SIFT‑128‑euclidean（图像特征）等多种数据集，覆盖不同相似度度量与维度。

**📈 对比分析**

通过 Pareto 前沿评估 QPS‑召回关系，报告 N‑缩放指数 α_N 与 d‑缩放指数 α_d；结果显示多探测网格在 N‑缩放上近线性，且在 d 高时表现更优；构建时间最短但查询延迟比大多数基线慢。

**⚠️ 局限性**

局限性包括：Python 实现导致吞吐低于可优化的 C++ 版本；仅在 d≤200 的范围内验证 d‑交叉点；未深入评估每次插入成本及对键向量分布（如 Transformer KV‑cache）的适用性；实验仅覆盖有限数据集。

---

## 40. Domain Knowledge Based Temporal-Spatial Graph Convolution Network for ECG Recognition

**arXiv ID:** 2607.01282 | [PDF](https://arxiv.org/pdf/2607.01282v1)

**作者:** Wenting Ma `[一作]` (China Mobile Research Institute), Zhenjie Yao `[通讯]` (Institute of Microelectronics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种基于域知识的时空图卷积网络，用于对12导联心电图进行多类别异常分类。

**💡 创新点**

创新点在于将PRQST关键点作为域知识构建双流有向图，分别捕捉心电图的空间（同一心跳内）和时间（相邻心跳间）信息。

**🔧 技术方法**

采用了图卷积网络（GCN）结合时间卷积网络（TCN）构成的双流指向图网络（DGNN），并在预处理阶段使用小波变换提取关键点。

**📊 数据集**

使用的是首届中国心电图智能竞赛（FECGIC）公开数据集，包含6500条12导联心电记录，共8种典型心脏异常及正常类。

**📈 对比分析**

与三种SOTA方法（深度CNN+手工特征、Attention‑Res‑BiLSTM、Residual‑Attention 结构）进行对比，平均F1得分0.881，整体最佳；在样本稀少的LAFB和ER类别上，F1提升至0.763，显著优于其他方法。

**⚠️ 局限性**

局限性包括图结构仍较简单，难以充分表达更复杂的相互作用；在噪声较大或关键点检测失误时仍会出现误分类；对极少样本类别的召回率仍相对较低。

---

## 41. WaveLander: A Generalizable Hierarchical Control Framework for UAV Landing on Wave-Disturbed Platforms via Reinforcement Learning

**arXiv ID:** 2607.01281 | [PDF](https://arxiv.org/pdf/2607.01281v1)

**作者:** Chun-Kit Li `[一作]` (Hong Kong University of Science and Technology), Ling Shi `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为WaveLander的层级强化学习控制框架，用于无人机在波浪扰动平台上的垂直降落控制。

**💡 创新点**

创新点在于将学习任务压缩为仅输出标量垂直速度参考，结合传统低层姿态与横向跟踪控制，避免了复杂的离散切换规则并降低了仿真到实装的鸿沟。

**🔧 技术方法**

采用了强化学习中的Policy Gradient与GAE，使用了PPO框架与GRU递归网络对观察序列进行建模。

**📊 数据集**

使用了MuJoCo中自定义的轻量级波浪平台模型生成的随机波浪数据，以及Isaac Sim/ArduPilot SITL仿真与小规模实地测试数据。

**📈 对比分析**

通过与常数下降基准进行比较，WaveLander在不同波浪强度下（30°/40°/60°倾斜）触地姿态误差≤0.10 rad的成功率显著提升，累计分布向低误差偏移，表明落地时机更安全。

**⚠️ 局限性**

局限性包括：波浪平台模型简化，未考虑真实海面气动与感知噪声；未加入显式平台运动预测模块；实地验证范围有限，尚未在更大规模海况下进行全面测试。

---

## 42. Fixed-Set Robustness in Programming by Example: Example Corruption and Semantic Partition Recovery

**arXiv ID:** 2607.01280 | [PDF](https://arxiv.org/pdf/2607.01280v1)

**作者:** Yuan Si `[一作]` (University of Waterloo), Jialu Zhang `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了编程按示例（PBE）中固定示例集被对抗性篡改导致的程序错误，并提出了相应的攻击与防御方法。

**💡 创新点**

创新点在于：①定义了“固定集合最坏情况”腐败模型并量化其对PBE的影响；②实现了基于版本空间的精确与启发式攻击搜索；③提出了版本空间分区聚合（VPA）作为在低边际下的语义聚合防御，并给出投票稳定性证书。

**🔧 技术方法**

主要技术包括：基于FlashFill风格的确定性字符串DSL、版本空间枚举与损失/复杂度目标、边际阈值公式、精确/束搜索攻击、VPA投票与语义签名聚合。

**📊 数据集**

使用的数据集包括：自定义生成的低边际与高边际DSL任务、公开SyGuS‑PBE_SLIA字符串任务、Playgol 语料库、以及20条受控边际1的LLM-PBE提示集合。

**📈 对比分析**

对比方法包括：随机打字错误、相同池随机替换、编辑距离匹配、Handa–Rinard 0/1‑bound 与裁剪损失、VPA 与投票基线。实验表明：策略性篡改显著优于随机噪声；束搜索可发现被贪婪搜索漏检的攻击；VPA 在保持投票边际的任务中能恢复正确程序，但在公开任务中投票边际接近1时被对抗者击败；LLM-PBE提示在受攻击后准确率大幅下降。

**⚠️ 局限性**

局限性：①仅评估有限的字符串DSL，无法覆盖更复杂的查询或代码生成任务；②攻击与防御在候选集合大小上有限制，真实全空间攻击效果未知；③VPA 仅在投票边际足够时有效，边际消失时会失效；④LLM-PBE 结果仅为范围演示，非全面 LLM 评估；⑤未对大规模数据集或在线学习场景进行验证。

---

## 43. Beyond Detection: Redesigning Assessment and Governande of Generative AI at the Universidad Politécnica de Madrid (UPM)

**arXiv ID:** 2607.01255 | [PDF](https://arxiv.org/pdf/2607.01255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 44. Benchmarking Federated Learning and Knowledge Distillation for Point Cloud Classification

**arXiv ID:** 2607.01272 | [PDF](https://arxiv.org/pdf/2607.01272v1)

**作者:** Aizierjiang Aiersilan `[一作]` `[通讯]` (University of Macau), Aizierjiang Aiersilan (University of Macau)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并执行了覆盖13种联邦学习算法与10种知识蒸馏目标的跨组合基准实验，对ModelNet40和临床头部形状数据集进行点云分类，探讨了联邦学习在极端非IID标签分布下的性能衰退、知识蒸馏的压缩效果以及两阶段FL+KD流程中因硬标签项导致的评估误导。

**💡 创新点**

① 完整的两阶段FL+KD跨组合基准；② 发现极端标签倾斜导致FL显著退化且无单一鲁棒算法；③ 揭露硬标签项造成的“恢复幻觉”，提出无硬标签蒸馏作为可信评估手段；④ 在真实临床数据验证该现象。

**🔧 技术方法**

使用了FedAvg、FedProx、FedNova、FedDyn、SCAFFOLD、Ditto、FedMedian、FedAvgM/FedAdam/FedYogi/FedAdagrad等联邦算法；蒸馏目标包括Vanilla KD、Feature KD、Attention Transfer、Logit‑MSE、Cosine、Self‑Distillation、Contrastive (CRD)、Decoupled KD等；教师网络为PointNet++ SSG，学生网络为压缩版SmallPointNet2；实验采用多随机种子、非IID标签倾斜划分、标准化训练协议。

**📊 数据集**

ModelNet40（40类3D模型）和临床crania‑synostosis（4类头部形状，100样本/类）两个数据集。

**📈 对比分析**

对每种FL与KD方案在三种随机种子下进行评估，报告均值±标准差。结果显示：在ModelNet40上最佳联邦模型仅达76.3%（vs 92.3%中心化）；在临床数据上最佳联邦仅75.8%（vs 100%中心化）。知识蒸馏可将教师压缩为74.5%大小、约2×推理速度提升，同时保持≈92%准确率。两阶段FL+KD中，含硬标签交叉熵的蒸馏目标无论教师质量如何均能恢复≈92%准确率，形成评估误导；纯无硬标签目标则准确率与教师本身高度相关。

**⚠️ 局限性**

实验仅覆盖两数据集、单一极端非IID划分、固定客户端数/轮次、统一PointNet++架构，未探究不同backbone（Transformer、KPConv）或异构客户端情况；因评估规模受限，未进行更细粒度的异构性和梯度噪声分析。

---

## 45. The Rising Unsustainability of AI Graphics Cards Production

**arXiv ID:** 2607.01258 | [PDF](https://arxiv.org/pdf/2607.01258v1)

**作者:** Clément Morand `[一作]` (University Paris-Saclay), Anne-Laure Ligozat `[通讯]` (University Paris-Saclay)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

收集并量化了2013–2025年NVIDIA工作站显卡的生产碳足迹与金属资源枯竭。

**💡 创新点**

首次提供系统化、可重复的显卡生产生命周期数据库，并揭示显卡生产损害随时间和销量持续上升的趋势。

**🔧 技术方法**

基于MLCA的参数化生命周期评估模型，结合GPU die面积、内存密度与基线影响进行计算。

**📊 数据集**

利用TechPowerUp GPU数据库、Wikipedia、NVIDIA数据表合成的174款工作站显卡特征数据集，以及Epoch AI的显卡销售数据。

**📈 对比分析**

与ADEME与NVIDIA实验数据对比，差异低于30%；模型结合销量预测显示2024年显卡生产排放可达数千吨CO2e，资源枯竭亦呈显著增长。

**⚠️ 局限性**

模型假设内存密度固定、未考虑技术节点变化；缺少最新显卡（B200/B300）die面积信息；未覆盖水资源、毒性等其他环境影响。

---

## 46. AI Assistance for Human Review of Default Judgments

**arXiv ID:** 2607.01256 | [PDF](https://arxiv.org/pdf/2607.01256v1)

**作者:** Theodora Worledge `[一作]` (Stanford University), David Freeman Engstrom `[通讯]` (Stanford University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于LLM的 Default Assistant，辅助法院人员审查债务收款案件的默认判决，提供标注且可追溯的法律要求评估与引用。

**💡 创新点**

创新点在于：1) 通过“先检索‑再生成‑引用”流程确保建议可验证；2) 采用多层次的法律要求分解与多轮人工标注；3) 在受控实验中展示了AI辅助可显著提升准确率与速度，且对种族/性别差异几乎无影响。

**🔧 技术方法**

主要技术包括：Azure AI Document Intelligence（OCR），OpenAI Embeddings 与向量数据库进行语义检索，LangGraph 管理检索-生成工作流，LLM（如 GPT‑4）用于提取证据、生成推荐与解释，并结合正则表达式定位原始文件中的引用。

**📊 数据集**

使用了 188 份加州债务收款案件的公开 PDF 文件（请求默认判决与获批判决），并在 20 份样本上训练 Default Assistant，剩余 168 份用于评估；数据通过法院电子档案系统获取。

**📈 对比分析**

在 66 名法律专业学生的受控实验中，辅助组与未辅助组对比：平均准确率提升 6.0%（p<1e-4），平均耗时下降 25.9%（p<2.5e-10）；在各法律要求上，误差率降低 47–62%，时间节省 24–34%；与单独 LLM 的性能相比，人机协作性能进一步提升，放大倍率可达 3×。

**⚠️ 局限性**

局限性包括：1) 受试者为法律学生，缺乏法院工作人员的经验与实际时间限制；2) 未辅助组表现已高，可能夸大 AI 效果；3) 长期使用可能出现过度依赖或低依赖；4) 技术锁定问题，系统需持续维护以适应法规变更。

---

## 47. A Practice Auditing Framework for Large Language Model Use: Collective Empiricism, Pseudo-Rational Cognition, and Governance of AI-Generated Content

**arXiv ID:** 2607.01248 | [PDF](https://arxiv.org/pdf/2607.01248v1)

**作者:** Yang Zhao `[一作]` (Beijing Institute of Technology), Zeyu Zhang `[通讯]` (Beijing Institute of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向大型语言模型（LLM）的实践审计框架，并通过对“集体经验主义”和“伪理性认知”两种概念的阐释，说明了LLM输出被误解为用户自身认知的风险；

**💡 创新点**

创新点在于：①首次将材料主义认识论与LLM使用结合，提出“集体经验主义”与“伪理性认知”的概念；②基于此构建完整的审计流程（需求定义、问题界定、证据审计、实践验证、逆向提问、日志回滚、认知更新），为LLM输出治理提供理论和操作性框架；

**🔧 技术方法**

主要技术手段为理论分析与框架设计，并未采用算法实现或模型训练；

**📊 数据集**

未使用具体数据集，主要基于文献综述和案例观察；

**📈 对比分析**

本文未进行实验对比或性能评估，所述方法以概念性说明和案例讨论为主；

**⚠️ 局限性**

局限性包括：概念性阐释缺乏跨场景的实证验证；对AI-AI对话循环、检测误判等现象的分析多基于观察而非系统实验；缺乏针对不同用途的细化评估方法。

---

## 48. Retrieval-Augmented Generation to Support Railways Engineering Tasks: A Case Study

**arXiv ID:** 2607.01244 | [PDF](https://arxiv.org/pdf/2607.01244v1)

**作者:** Andrea Gerardo Russo `[一作]` (NIER Engineering S.p.A.), Giuseppe-Emiliano La Cara `[通讯]` (NIER Engineering S.p.A.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了一个检索增强生成(RAG)框架，用于铁路行业技术规范（UNISIG规范）的查询与解答。

**💡 创新点**

创新点在于结合人机协同的领域特定微调、量化与可视化交互界面，实现了本地化、安全合规且可解释的技术规范问答系统。

**🔧 技术方法**

使用技术包括RAG、LangChain、FAISS向量检索、Unstructured文本解析、Zephyr LLM、LoRA微调、量化、以及基于web的交互界面。

**📊 数据集**

使用数据集为三份英文UNISIG技术规范（SUBSET‑026、037、098）以及人工标注的665条问答对与人工生成的2831条问答对。

**📈 对比分析**

通过与Mistral和Falcon的人工评分对比，选定Zephyr-7B-beta；随后在人工评分和ROUGE评估中，量化后模型性能略有下降但人类评价提升，最终平均评分从1.25提升至3.21。

**⚠️ 局限性**

限制包括缺乏对标基准系统的对比、检索精度不足导致幻觉、无法回答图像/图表内容、对长表格和多上下文推理的支持有限。

---

## 49. TurnNat: Automatic Evaluation of Turn-Taking Naturalness in Dyadic Spoken Dialogue

**arXiv ID:** 2607.01345 | [PDF](https://arxiv.org/pdf/2607.01345v1)

**作者:** Hao Zhang `[一作]` (Johns Hopkins University), Laureano Moro-Velazquez `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TurnNat框架，用未来双声道活动预测的负对数似然来自动评估对话中的转接自然度，并构建人类验证的转接扰动基准。

**💡 创新点**

创新点在于：①将转接自然度视为自然对话分布下的似然问题；②使用未来双声道活动的概率分布进行局部评分并聚合为对话级得分；③在单一连续评分空间内统一衡量多种转接失误；④结合对话边界单元（TBU）和尾部NLL聚合实现对局部异常的敏感度。

**🔧 技术方法**

核心技术包括：Causal Future Voice‑Activity Prediction（VAP、DualTurn）模型；负对数似然（NLL）计算；TBU提取与加权训练；对比度指标（C-index）和配对准确率评估；对模型进行全微调与辅助任务训练。

**📊 数据集**

使用Seamless Interaction数据集的英语双声道自然对话作为训练和评估基础，随后在测试集上构造了包含“迟到回复”“提前入场”“保持转移”等五种扰动的配对样本作为基准。

**📈 对比分析**

与VAP原版、DualTurn伯努利输出以及全微调版本比较，最佳配置D4（DualTurn+256分类+辅助任务+TBU加权α=8）在配对准确率上达到88%（C-index 0.676），明显优于VAP（≈80%）和Bernoulli DualTurn（≈78%）。

**⚠️ 局限性**

局限性包括：仅评估对话时序扰动，未覆盖ASR误差、语义失配、话语内容等因素；仅基于双声道活动，可能忽略词汇或语境对自然度的影响；人类验证仅用于基准而非分数校准，缺乏更大规模的主观对齐。

---

## 50. How Much Future Helps? A Controlled Study of Future-Privileged Supervision for Causal Egocentric Gaze Estimation

**arXiv ID:** 2607.01437 | [PDF](https://arxiv.org/pdf/2607.01437v1)

**作者:** Jia Li `[一作]` (University of Texas at Dallas), Yapeng Tian `[通讯]` (University of Texas at Dallas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种可控的未来信息辅助训练框架，训练时使用未来帧监督严格因果的在线注视预测模型，推断时仅保留因果网络。

**💡 创新点**

通过共享参数的双分支（未来感知教师与因果学生）在不改变推断架构的前提下，系统评估未来上下文对因果注视预测的效用，并发现最佳未来窗口约为1.7–3.3秒。

**🔧 技术方法**

使用冻结的DINOv3视觉编码器、轻量级分离时空注意力头、全局‑局部聚焦（GLF）模块以及未来优先监督（FPS）蒸馏损失。

**📊 数据集**

在EGTEA Gaze+（烹饪活动）和Ego4D（大规模日常活动）两大视角注视基准上进行实验。

**📈 对比分析**

与多种基线（经典显著性、早期任务特定模型、GLC等）对比，ECOGaze在保持严格因果的前提下实现F1≈45.9（EGTEA）和42.7（Ego4D），比因果GLC高出约4–5 %且参数量减少≈5×、帧率翻倍。

**⚠️ 局限性**

受限于固定的视觉编码器与单一动作域，未来窗口的最佳时长可能随任务、模型规模及多模态信息不同而变化；缺少更广泛任务、长时记忆或多模态特征的验证。

---

## 51. When Should Service Agents Reconsider? Difficulty-Routed Control in Customer-Service Operations

**arXiv ID:** 2607.01426 | [PDF](https://arxiv.org/pdf/2607.01426v1)

**作者:** Qian Chen `[一作]` (Pennsylvania State University), Xin Yu `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种难度路由服务控制架构，针对自主客服代理在执行后端写操作时的运营风险进行分层控制。

**💡 创新点**

创新点在于通过提示式路由器识别高风险请求，仅在这些请求中启用冲突-aware 对话与预写复核，避免全局加重计算与交互负担。

**🔧 技术方法**

采用LLM提示技术实现路由判断、ReSpAct候选生成以及预写验证器，并结合工具调用实现写前再评估。

**📊 数据集**

使用τ^2‑bench的零售和航空两大任务集进行实验评估。

**📈 对比分析**

在冲突任务集上相较基线提升约20%多数通过率，整体任务集提升有限；表现为在高风险场景下显著提高可靠性。

**⚠️ 局限性**

局限性包括仅在模拟bench环境验证，未覆盖真实客服场景；路由与验证策略为手工提示，缺乏自学习或监管型决策机制。

---

## 52. On Reconstructing a Convex Polygon from Partial Information

**arXiv ID:** 2607.01423 | [PDF](https://arxiv.org/pdf/2607.01423v1)

**作者:** Alexander Baumann `[一作]` (Freie Universität Berlin), André Schulz `[通讯]` (FernUniversität in Hagen)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

系统研究了从多种单一或组合特征（如三角形面积、边长、边法向量、角度等）重构凸多边形的可行性与算法，并给出了完整的可行性表、线性/多项式时间算法以及部分不可解情况。

**💡 创新点**

创新点在于：①提出了“三角形面积唯一可重构”这一全新可行性结论并给出线性时间构造；②给出“仅边距离可构造星形多边形”的完整构造方法并在边距离唯一时提供线性时间凸性判定；③引入极坐标极点变换和传播公式统一处理多种组合特征，形成一套系统化的判定与构造框架；④对未解决的组合特征给出硬性复杂性分析。

**🔧 技术方法**

主要技术包括：几何极坐标极点对偶、SAS-公式推导、局部极小/极大序列的角度求和判定、整数乘法的分治实现（O(n log²n)），以及基于角度与弧度的二分搜索判定凸性；还使用了归纳构造和可行性传播公式。

**📊 数据集**

该工作为理论分析，未使用真实数据集；所有结果均基于抽象数学模型和数值假设。

**📈 对比分析**

由于是理论算法研究，没有实验对比；但作者给出的算法复杂度（线性、O(n²log n) 等）已与已知最优结果对齐，并在可行性判定上实现了最优或接近最优的时间复杂度。

**⚠️ 局限性**

局限性包括：部分组合特征（如“边长度+角度”或“边距离+角度”）仍未完成算法设计；L₀-Minkowski问题缺乏构造算法；对某些情况的多项式复杂度上限仍未知；并且对硬性不可解情况的证明仅适用于特定假设（如边距离互异）。

---

## 53. Adoption and Impact of Command-Line AI Coding Agents: A Study of Microsoft's Early 2026 Rollout of Claude Code and GitHub Copilot CLI

**arXiv ID:** 2607.01418 | [PDF](https://arxiv.org/pdf/2607.01418v1)

**作者:** Emerson Murphy-Hill `[一作]` (Microsoft), Alexandra Savelieva `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文利用公司内部工程师的使用遥测和人力资源数据，系统分析了两款CLI型代理式编码工具（Claude Code 与 GitHub CLI）的采用过程（首次使用、保留率）以及其对合并拉取请求（PR）产出的影响。

**💡 创新点**

创新点在于：①首次以细粒度、可识别的工程师级别遥测数据进行实地采用研究，填补了仅基于调查或公共仓库的研究空白；②将“首次使用”与“保留率”拆分为两个独立模型，揭示二者驱动因素差异；③采用合成控制（CausalImpact）和个体固定效应的剂量-反应分析，验证工具对产出的持续提升，并比较不同工具的效果。

**🔧 技术方法**

技术手段包括离散时间逻辑回归、贝叶斯结构时间序列（CausalImpact）、个体固定效应泊松回归（dose‑response）以及多重比较校正（Benjamini–Hochberg）等统计与计量经济学方法。

**📊 数据集**

使用的数据集来自公司全体软件工程师的内部日志，涵盖：①工具使用日志（Claude Code 与 GitHub CLI 的日/周使用频次）；②拉取请求记录（Azure DevOps 上 PR 的创建、合并时间）；③HR 人事数据（职级、任期、团队分配等）。

**📈 对比分析**

在采用与保留方面，社交曝光（同事/经理使用率）是最大驱动力；在产出方面，合成控制估计显示采用后 PR 合并率提升约 10‑15%，且在四个月观察窗口内保持稳定；个体固定效应分析进一步证实，使用天数越多 PR 合并数越高，且 Claude Code 的提升约为 GitHub CLI 的两倍左右。

**⚠️ 局限性**

局限性包括：①仅在单一大型软件公司内部进行，外推性有限；②采用 PR 合并数作为产出指标，未能衡量代码质量与维护成本；③社交曝光与同质性混淆，难以完全分离影响；④使用期限短，仅覆盖四个月，无法评估长期效应。

---

## 54. NeuroBridge: Bridging Multi-Task MRI Knowledge for Neurodegenerative Disease Diagnosis

**arXiv ID:** 2607.01401 | [PDF](https://arxiv.org/pdf/2607.01401v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 55. Conditional Inference Trees and Forests for Feature Selection

**arXiv ID:** 2607.01417 | [PDF](https://arxiv.org/pdf/2607.01417v1)

**作者:** Robert Milletich `[一作]` (Amazon Web Services), Newel Hirst `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对 conditional inference trees (CIT) 与 conditional inference forests (CIF) 的 top‑k 特征排序效果进行系统评估，验证其在下游预测任务中的有效性。

**💡 创新点**

创新点在于提出并证明了固定节点 Stage‑A 排序的完整置换检验保证，同时通过大规模真实与合成数据实验验证 CIF 在高维稀疏场景下仍能保持较高的特征排序质量。

**🔧 技术方法**

使用的技术包括条件推断树/森林、Monte Carlo 置换 p‑值、Bonferroni 多重检验、阈值与特征自适应停止、以及阈值哈希化（histogram‑256）等算法改进。

**📊 数据集**

实验数据集涵盖 23 个分类数据集、8 个回归数据集以及多组合成数据与高维稀疏模拟（p∈{100,500,1000}），共计 45+ 个任务。

**📈 对比分析**

与 17 种特征选择基线（包括 ctree、cforest、CIT、随机森林、ExtraTrees、XGBoost 等）进行比较，CIF 在分类任务中平均排名第 4，回归任务第 3，且在多种下游模型（LR、SVM、KNN、Ridge、SVR）上表现优异，平均提升约 1–2% 的准确率/R²。

**⚠️ 局限性**

局限性包括：固定节点理论仅适用于完整置换且不考虑自适应停止或特征/阈值扫描；在极高维稀疏情形下特征采样可能导致重要特征被忽略；实验规模虽大但仍缺乏对多任务学习与实时在线场景的评估。

---

## 56. GPUAlert: A Zero-Instrumentation Process-Boundary Monitor for Diagnosing GPU Training-Job Failures

**arXiv ID:** 2607.01409 | [PDF](https://arxiv.org/pdf/2607.01409v1)

**作者:** Parv Agarwal `[一作]` (Indian Institute of Technology Patna), Asif Ekbal `[通讯]` (Indian Institute of Technology Patna)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款零仪器化的 GPU 训练作业监控工具 GPUAlert，能够在不改动训练脚本的前提下，监控作业运行、捕获日志、分类失败原因、收集输出并通过邮件即时通知运维人员。

**💡 创新点**

核心创新在于将三大可靠性原语（预启动日志保证、通知隔离、非静默产物预算）结合到进程边界监控中，保证日志持久、退出码不受邮件失败影响、产物不会被无声丢失，并提供 15 种 GPU/Python 失败模式的规则分类器。

**🔧 技术方法**

技术包括：进程子进程管理、标准流实时写入日志、正则表达式优先级匹配失败分类、SMTP 异常捕获隔离、产物文件大小预算及附件处理、Python 命令行包装。

**📊 数据集**

使用了 474 条标注的 GPU 训练日志（15 类失败模式），其中 360 条为硬件复现日志，另外 24 条来自公开 issue tracker 的野生样本；同时发布了评测 harness 以实现可复现性。

**📈 对比分析**

与基线（仅退出码、错误类型解析、无优先级的关键字搜索）比较，GPUAlert 的宏观 F1 率在 12 个硬件复现类上达到 0.997，整体 0.998；相较于 grep 的 0.830、traceback 的 0.559、exitcode 的 0.133 具有显著提升；包装器的开销恒定约 3 ms；日志持久化和退出码隔离等属性在所有 15 类失败模式下均成立。

**⚠️ 局限性**

局限包括：评测仅在单台 NVIDIA V100、单驱动版本上完成，未覆盖 A100/H100 等新硬件；三类失败模式（CUDA OOM、NCCL、OOM‑killer）为合成，缺乏真实硬件证据；仅监控单进程/单节点作业，无法直接应对多节点分布式训练中的同步或节点失效；正则规则对非标准错误字符串的识别能力有限。

---

## 57. Spin-Weighted Spherical Harmonics Enable Complete and Scalable $\mathrm{E}(3)$-Equivariant Networks

**arXiv ID:** 2607.01408 | [PDF](https://arxiv.org/pdf/2607.01408v1)

**作者:** Chenxing Liang `[一作]` (Texas A&M University), Shuiwang Ji `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种基于自旋加权球谐函数的Gaunt张量乘积（SpinGTP），可在保持O(L³)计算复杂度的同时恢复E(3)等变网络中缺失的反对称交互路径，解决了传统GTP在奇偶性上的表达不足。

**💡 创新点**

核心创新在于将自旋加权球谐函数（SWSH）引入Gaunt张量乘积，利用自旋选择规则重建奇偶性不对称路径，从而实现数学上的完整性与计算上的高效性兼顾；同时构造了实数、带奇偶标签的SWSH基底，进一步实现了对镜像和手性结构的显式区分。

**🔧 技术方法**

技术手段包括：
1) 自旋加权球谐函数（SWSH）与Gaunt积分的推广；
2) 通过FFT实现的O(L³)空间域乘积；
3) 预卷积核（pre‑contracted kernel）与多通道混合的高效实现；
4) 兼容SWSH的等变线性层、归一化层和注意力机制；
5) 将SpinGTP集成至MACE和Equiformer框架。

**📊 数据集**

采用的数据集有：
- Tetris（手性几何判定）
- 3BPA（分子能量/力预测）
- SPICE‑MACE‑OFF（含手性子集的能量/力预测与手性分类）
- OC20 IS2RE（催化剂初始到平衡能量预测）

**📈 对比分析**

与传统的CGTP、标量GTP、VSTP以及主流模型（Allegro、NequIP、MACE、Equiformer、EquiformerV2）进行对比。实验显示：
- 在Tetris任务上，SpinGTP实现100%准确率，标量GTP仅达75%；
- 在3BPA、SPICE子集和OC20上，SpinGTP的能量/力误差与CGTP相当，且在手性分类、低温/高温泛化和异构点对化合物预测上优于其它基线；
- 收敛速度最快，尤其在手性子集上几乎一半的epoch即可达到98%验证准确率。

**⚠️ 局限性**

局限性包括：
1) 需要一致的局部框架（gauge）才能聚合自旋特征，难以在高度对称点云上保持等变性；
2) 对于小的最大角度L，使用球面变换并未显著加速，且多通道乘法导致内存占用较大；
3) 仍未实现比现有SWSH实现更快的硬件加速，且在极高多重性场景下性能下降。

---

## 58. Benchmarking Code Improvement with Progressive, Adaptive, and Interactive Feedback

**arXiv ID:** 2607.01360 | [PDF](https://arxiv.org/pdf/2607.01360v1)

**作者:** Cuong Chi Le `[一作]` (University of Texas at Dallas), Tien N. Nguyen `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向代码改进的渐进式、可适应的交互式评测框架；

**💡 创新点**

引入双维度的反馈控制（失败区域与提示深度）以及进展度量，使评测从单一通过/失败转向轨迹评估；

**🔧 技术方法**

采用结构化提示生成、故障情境聚类、分层提示深度、以及基于轨迹的多指标评价；

**📊 数据集**

基于Codeforces Python提交，构造440个含错误的程序及其参考实现的测试集；

**📈 对比分析**

通过对八大LLM在受控提示与无提示交互下的指标（FinalFix、TRS、HE等）进行比较，DeepSeek在大多数指标上领先，控制提示显著提升评测稳定性；

**⚠️ 局限性**

仍受限于提示生成模型的性能、缺乏跨语言验证、以及评测对特定测试集与提示策略的依赖。

---

## 59. Geometry-Aware R-Structured Kolmogorov-Arnold Networks

**arXiv ID:** 2607.01449 | [PDF](https://arxiv.org/pdf/2607.01449v1)

**作者:** Sergei Kucherenko `[一作]` (Imperial College), Nilay Shah `[通讯]` (Imperial College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Geometry-aware R-Structured Kolmogorov–Arnold Network (GRS-KAN)，将R-函数嵌入KAN框架，实现在网络中显式编码几何约束与逻辑结构，支持在连续模型中加入分段跳跃或区域限制；

**💡 创新点**

创新点在于三种架构（加性、乘性、agnostic）以及可学习的结构选择权重，实现自动识别几何约束与光滑模型的交互；同时给出R-函数的可微解析形式和梯度；

**🔧 技术方法**

采用Kolmogorov–Arnold网络的可学习一元函数边激活、R-函数的逻辑运算（∧,∨,¬）、sigmoid平滑门、可学习权重和正则化；

**📊 数据集**

实验数据集为人工生成的二维回归基准：Liu等的toy函数、乘积函数以及带有矩形不连续跳跃或支撑的目标函数；

**📈 对比分析**

与标准KAN、MATLAB MLP进行比较，GRS-KAN在测试RMSE和边界RMSE上分别提升约58–67%，参数量保持与标准KAN相当；在无几何约束的情形下，agnostic模型自动抑制几何分支，恢复纯KAN；

**⚠️ 局限性**

局限性包括仅在二维简单几何（矩形、圆）上验证，复杂非凸或多分量隐式域仍需进一步研究；此外R-函数门的梯度在角点处不光滑，可能影响收敛；

---

## 60. Hamm-Grams: An Algorithm for Mining Regular Expressions of Bytes

**arXiv ID:** 2607.01445 | [PDF](https://arxiv.org/pdf/2607.01445v1)

**作者:** Derek Everett `[一作]` (Amazon), James Holt `[通讯]` (CrowdStrike)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于 Hamming 相似性的正则表达式特征——hamm-gram，用于更鲁棒的恶意软件检测。

**💡 创新点**

创新点是将局部敏感哈希与聚类相结合，自动生成带单字符通配符的固定长度正则表达式，显著提升特征稀疏性与鲁棒性。

**🔧 技术方法**

采用了复数加权滚动 LSH、层次聚类、前缀树 DFA 等技术，并在逻辑回归模型中使用。

**📊 数据集**

在 Drebin Android、EMBER 2018 Windows PE、PDF 等公开数据集上进行实验。

**📈 对比分析**

与传统 n-gram、ssdeep、sdhash、LZJD 等基线相比，hamm-gram 在受限特征数时实现了 80%+ 的平衡准确率，且在 AUC 方面优于大多数基线，显著提升了检测性能。

**⚠️ 局限性**

限制在于需要预先设定通配符预算与窗口大小，且对极短或高变异样本的鲁棒性仍有限；聚类计算成本与特征构造过程仍有优化空间。

---

## 61. From Forgeries to Foundation Models: A Systematic Survey of Identity Document Attack and Detection

**arXiv ID:** 2607.01442 | [PDF](https://arxiv.org/pdf/2607.01442v1)

**作者:** Gourab Das `[一作]` (Indian Institute of Information Technology Dharwad), Raghavendra Ramachandra `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对身份文件伪造与检测进行了系统综述，提出统一的威胁模型，涵盖呈现攻击、数字注入攻击与生成式AI驱动的全量合成，并对公开数据集进行审计，构建零样本评估集以量化现有模型在新合成攻击下的性能；

**💡 创新点**

创新点包括首次统一三类攻击的威胁模型与评估框架，揭示了Script-Dependent Generative Instability（SDGI）这一生成式文本失真现象，系统性审计2019–2025年公开数据集与Reality Gap，并在零样本情境下评测主流模型，揭示其泛化不足；

**🔧 技术方法**

使用的技术涵盖规则与启发式检测、全局深度学习分类、取证微观特征分析、注入感知网络、基础模型（DINOv2、CLIP、ViT）、VLM（GPT‑5、Gemini、GPT‑4o）、few‑shot学习及生成式编辑模型；

**📊 数据集**

采用的数据集有MIDV系列、DLC2021、KID34K、SIDTD、IDNet、Syn‑IDPASS、RSCID、FantasyID、FakeIDet‑db等公开数据集，并自构造了包含140个未见合成ID卡的零样本评估集；

**📈 对比分析**

比较方法为在该零样本评估集上对九个公开模型进行APCER、BPCER、EER评估；结果显示即使在0.5阈值下，GPT‑5、Gemini等模型的APCER仍在30–40%，EER在13–30%，表现远低于安全阈值，表明现有模型对生成式合成攻击的鲁棒性不足；

**⚠️ 局限性**

局限性在于公开数据集缺乏真实样本、多脚本、多攻击类型与像素级定位标注，评测方法与标准碎片化，现有模型缺乏跨域泛化与定位能力，生成式攻击的全流程尚未被充分捕获，导致评测与实际部署之间存在显著Reality Gap。

---

## 62. IsoSci: A Benchmark of Isomorphic Cross-Domain Science Problems for Evaluating Reasoning versus Knowledge Retrieval in LLMs

**arXiv ID:** 2607.01431 | [PDF](https://arxiv.org/pdf/2607.01431v1)

**作者:** Samir Abdaljalil `[一作]` (Texas A&M University), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种跨域同构科学题对的基准，用以分离模型的推理能力与领域知识检索；

**💡 创新点**

创新点在于构造可比的同构题对并引入p_know指标，将推理模式提升拆解为知识相关与结构不变两部分；

**🔧 技术方法**

主要技术包括大语言模型生成与人工/LLM评估、零样本链式思考提示、答案提取链和统计置信区间估计；

**📊 数据集**

使用了自研的144对跨域题目集合，覆盖物理、化学、生物、地球科学四个领域，来源于GPQA、SciBench、MMLU-STEM以及自生成的地球科学题目；

**📈 对比分析**

通过传统对比与推理开关(toggle)对比，对五对模型（OpenAI、Google、Qwen、DeepSeek）在四个基准上进行评估，结果显示91.3%推理模式收益主要依赖知识，推理开关几乎无提升；

**⚠️ 局限性**

局限包括样本规模有限（仅144对）、评测范围仅限短链推理、评估模型覆盖有限、自动评分误差和推理开关对模型权重影响有限。

---

## 63. The Rollout Infrastructure Tax in Coding-Agent Reinforcement Learning

**arXiv ID:** 2607.01415 | [PDF](https://arxiv.org/pdf/2607.01415v1)

**作者:** Daniel Thi Graviet `[一作]` (Daytona), Ivan Burazin `[通讯]` (Daytona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了编码代理强化学习中执行基础设施的成本（即“执行基础设施税”），对比四种常见执行子系统（单容器、托管沙箱、Kubernetes 容器和云虚拟机）的冷启动与运行时延迟。

**💡 创新点**

提出“执行基础设施税”这一概念并量化其对大规模训练成本的影响，提出了三项设计要求：基于位置的热池、低延迟动作 API、以及按作业规模匹配的隔离机制。

**🔧 技术方法**

采用控制实验方法，固定编码代理工作负载（包括最小命令、预加载仓库、完整错误修复等三层），测量环境创建、就绪、每步执行和控制面板开销，并使用 100 次实验获取 p50/p95 延迟。

**📊 数据集**

使用三层工作负载：T0 最小命令、T1 预加载仓库、T2 完整错误修复（含克隆、检查、修补、测试），均在相同环境配置下执行。

**📈 对比分析**

结果显示，冷启动延迟可相差 110 倍，150 步轨迹延迟差距可达 19 秒，导致 1 百万轨迹的工作时数差距超过 5,000 小时，表明子系统选择对训练成本影响巨大。

**⚠️ 局限性**

实验受限于固定命令序列、特定硬件与地区配置，未覆盖真实代理多样化行为，也仅衡量环境层成本，未包含模型推理与奖励计算等其他训练成本。

---

## 64. RusFinChain: A Russian Benchmark for Verifiable Chain-of-Thought Reasoning in Finance with Fuzzy-Aligned Evaluation

**arXiv ID:** 2607.01388 | [PDF](https://arxiv.org/pdf/2607.01388v1)

**作者:** M. K. Arabov `[一作]` `[通讯]` (Kazan Federal University), M. K. Arabov (Kazan Federal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 RusFinChain——俄语可验证链式推理金融基准。

**💡 创新点**

提出 Fuzzy Numeric Alignment 与 Soft-Attention Alignment 两种连续评估指标，提升与最终答案的相关性。

**🔧 技术方法**

使用可执行 Python 模板生成符号推理示例，结合语义编码、DTW 与软注意力对齐技术。

**📊 数据集**

使用 5,280 个自定义模板生成的俄语金融推理实例（17 个领域，172 个主题）。

**📈 对比分析**

对 8 个开源 LLM 进行零样本评估，Hard F1≈0.65，最终答案正确率≈29%，连续指标相关性更高。

**⚠️ 局限性**

局限性包括：数据为合成，缺乏真实文本多样性；仅覆盖俄语，未验证更大规模模型；对非数值推理不适用。

---

## 65. On the Utility and Factual Reliability of Pruned Mixture-of-Experts Models in the Biomedical Domain

**arXiv ID:** 2607.01444 | [PDF](https://arxiv.org/pdf/2607.01444v1)

**作者:** Atsuki Yamaguchi `[一作]` (University of Sheffield), Nikolaos Aletras `[通讯]` (University of Sheffield)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

系统评估Mixture-of-Experts（MoE）模型在生物医学领域进行专家剪枝时，对下游任务效用和事实可靠性的影响；并对比不同剪枝比例、方法及跨域情况，探索安全压缩的边界；

**💡 创新点**

首次系统性地把专家剪枝与高风险领域的事实可靠性关联起来，发现效用与可靠性不一定同步，提出在高风险部署中必须同时评估可靠性；

**🔧 技术方法**

采用训练无关的领域特定专家剪枝框架，六种 saliency 评估指标（Random、Frequency、Gate、EAN、EASY‑EP、REAP），结合 LLM‑as‑a‑Judge 可靠性评估器；

**📊 数据集**

使用医学专用数据集 MedINST（评估与校准）、MultiMedQA、Multi‑XScience、RCT、以及通用领域基准 IFEval、GSM8K、HumanEval、MMLU、Multi‑News+；

**📈 对比分析**

对四款MoE模型（GPT‑OSS 20B、Qwen3 30B、Nemotron3 Nano 30B、Qwen3.6 35B）在多种剪枝比例（12.5%~75%）和六种剪枝方法下进行对比；使用 ROUGE‑L、chrF++、F1、准确率、Hallucination 率、相对/绝对可靠性等指标。结果表明：中等剪枝（≤50%）在生物医学领域能保持几乎完整的效用，但极端剪枝导致可靠性显著下降；跨域任务中效用与可靠性均快速恶化；量化实验表明量化与剪枝可并行，但在跨域场景可能提升 hallucination。

**⚠️ 局限性**

仅评估训练无关剪枝，未探讨微调或后处理的影响；可靠性评估依赖 LLM‑as‑a‑Judge，可能受模型偏差；数据集覆盖有限，未覆盖所有高风险任务；对不同 MoE 架构（如 Nemotron3 混合 Transformer‑Mamba）机制解释不够深入；需进一步研究更细粒度的事实一致性度量和安全部署策略。

---

## 66. FUSE: A Partitioned Field-Exchange Framework for Coupling Physics Simulations in FEBio

**arXiv ID:** 2607.01428 | [PDF](https://arxiv.org/pdf/2607.01428v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 67. FaithMed: Training LLMs For Faithful Evidence-Based Medical Reasoning

**arXiv ID:** 2607.01440 | [PDF](https://arxiv.org/pdf/2607.01440v1)

**作者:** Zhiyun Zhang `[一作]` (Carnegie Mellon University), Chenyan Xiong `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FaithMed 框架，通过医生设计并自动优化的评价 Rubric 进行步骤级奖励训练，提升 LLM 在医学问答中的基于证据的推理过程与答案准确率。

**💡 创新点**

创新点在于：①将证据医学五步流程（Ask, Acquire, Appraise, Apply, Assess）转化为可量化的 Rubric；②在强化学习中采用细粒度步骤级过程奖励和优势归一化，明确给每一步骤可比的正向信号；③通过教师模型迭代自动精炼 Rubric，减少冗余并提升判别力。

**🔧 技术方法**

使用技术包括：强化学习框架 GRPO；步骤级过程奖励与优势分组；Gemini‑2.5‑Flash‑Lite 作为评判器进行 Rubric 评分；SFT + RL 训练流程；与检索增强生成（RAG）结合的检索策略；以及对 Rubric 的自动精炼与临床专家校正。

**📊 数据集**

实验数据集涵盖七大医学 QA 基准：HeadQA、MedMCQA、MedCalc‑Bench、MedQA、MMLU‑Pro‑Health、MedXpertQA 和 MedBullets^5op，分别用于训练、在分布内和离散的跨分布评估。

**📈 对比分析**

与基准 LLM、MedRAG、agentic search、以及仅使用 episode‑level 奖励的 RL 方法对比，FaithMed 在 1.7B 与 4B Qwen3 模型上平均提升答案准确率约 9% / 10.8%，并在 Rubric 评分上提升约 15.5%。在某些基准上还超越了更大规模的 Qwen3‑8B 参考模型。

**⚠️ 局限性**

限制包括：仅在文本 QA 基准上验证，未覆盖完整临床工作流程；Rubric 与评判器仍可能被模型“技巧性”满足而不真正实现可信推理；实验仅针对 Qwen3 系列模型，缺乏多模型、多语言、多专业的广泛评估；在真实临床使用前仍需严格验证与伦理审查。

---

## 68. Discrete Diffusion Language Models for Interactive Radiology Report Drafting

**arXiv ID:** 2607.01436 | [PDF](https://arxiv.org/pdf/2607.01436v1)

**作者:** Max Van Puyvelde `[一作]` (Stanford University), Olivier Gevaert `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文通过在相同规模、相同族系的混合专家模型上对比离散扩散语言模型与自回归模型，在医学视觉问答和放射报告起草任务中进行微调与评估。

**💡 创新点**

创新点在于：①证明离散扩散模型在医学任务中能与自回归模型匹敌或更优；②展示其独有的任意顺序填充（any-order infill）功能，能在已写文本两侧同时完成内容填充；③通过一致的 LoRA 适配方案实现仅因生成范式不同的公平比较。

**🔧 技术方法**

使用的技术包括：混合专家（Mixture-of-Experts）扩散语言模型 DiffusionGemma、其自回归同族 Gemma-4、LoRA 低秩适配、SigLIP 视觉编码器、统一状态 dLLM 目标、以及 LLM（Claude Sonnet 4.6）作为语义评判。

**📊 数据集**

实验所用数据集包括：VQA-RAD、SLAKE、VQA-Med-2019，用于医学视觉问答；以及 Radiology‑RRG 数据集，用于评估任意顺序填充效果。

**📈 对比分析**

比较方法是：在相同 LoRA 方案、同尺寸模型、同硬件（H100）下，评估 LLM-judge 语义准确率、推理延迟与吞吐量；结果显示，扩散模型在三组 VQA 数据集上与自回归模型持平或更好，且推理速度提升 3.5–4.4 倍，且在任意顺序填充任务中对双侧上下文的利用显著优于自回归模型。

**⚠️ 局限性**

局限性包括：仅在胸部 X‑ray 相关任务上验证，未评估跨模态推理深度；扩散模型虽速度快，但在极长文本生成时仍受限于固定画布；LoRA 适配仅在预训练后期调优，可能不适用于所有大规模模型。

---

## 69. CreativityNeuro: Steering Language Model Weights to Improve Divergent Thinking and Reduce Mode Collapse

**arXiv ID:** 2607.01433 | [PDF](https://arxiv.org/pdf/2607.01433v1)

**作者:** Samuel Schapiro `[一作]` (Univeristy of Illinois, Urbana-Champaign), Lav R. Varshney `[通讯]` (Stony Brook University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种无数据、基于权重空间的对比调节方法（CreativityNeuro），通过对比创意与非创意提示计算参数重要性，选择并缩放特定权重，从而提升LLM的发散性思维。

**💡 创新点**

创新点在于：①不依赖行为数据或梯度微调，完全数据自由；②通过对比提示集实现参数重要性筛选，得到可转移的创意相关权重；③在多任务中表现出优于激活调节的泛化能力。

**🔧 技术方法**

技术包括：参数重要性计算（Wanda式乘积法）、对比提示集构建、权重稀疏选择与缩放、无梯度推理调节。

**📊 数据集**

使用的数据集为：DAT（词汇空间离散化任务）和标准创意评测AUT、Task Task；无监督的对比提示集合；不使用任何标注行为数据。

**📈 对比分析**

对比方法：与提示、温度、Top‑k/p、激活调节（CAA）等基线相比较，在DAT上提升高达14个百分位；在AUT、TT上显著提升原创性、惊喜度和创造性；同时显著降低模式坍塌。

**⚠️ 局限性**

局限性：仅在有限的创意基准上验证，未涵盖全部创造性维度；对比提示集设计可能影响效果；未探索更高级的激活空间干预；发现创意与事实推理在权重空间不可分离，仍需进一步研究。

---

## 70. Robustifying Sparse Matrix Multiplication

**arXiv ID:** 2607.01427 | [PDF](https://arxiv.org/pdf/2607.01427v1)

**作者:** Karl Bringmann `[一作]` (ETH Zurich), Vasileios Nakos `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种黑盒归约，将鲁棒稀疏矩阵乘法转化为普通稀疏矩阵乘法，保持多项式对数级别的开销；

**💡 创新点**

创新点在于利用稀疏恢复技术和最小成本多选背包问题，首次实现了对大多数稀疏矩阵乘法算法的鲁棒化，得到更优的时间复杂度；

**🔧 技术方法**

核心技术包括稀疏恢复算法、最小成本多选背包近似、快速傅里叶变换以及矩阵乘法指数优化；

**📊 数据集**

该工作为理论性论文，没有使用具体实验数据集；

**📈 对比分析**

与之前仅鲁棒化Pagh算法（O(nk)）相比，新方法在多种参数区间实现了如((n+k)^{1.346})和k≥n^{1.762}时的近乎最优O(k^{1+ε})；

**⚠️ 局限性**

主要局限是依赖随机化、对数级别的预处理与空间开销，以及理论实现可能难以直接用于工业规模的稀疏矩阵乘法；

---

## 71. Agent4cs: A Multi-agent System for Code Summarization in Large Hierarchical Codebases

**arXiv ID:** 2607.01425 | [PDF](https://arxiv.org/pdf/2607.01425v1)

**作者:** Yongjian Tang `[一作]` (Siemens AG), Thomas Runkler `[通讯]` (Siemens AG)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Agent4cs 这一多智能体框架，采用自底向上策略对大型层次化代码仓库进行函数级和文件夹级的代码摘要。

**💡 创新点**

创新点在于三大智能体协作：摘要智能体负责生成摘要、关键字提取智能体主动捕获子文件夹信息、质量保障智能体进行迭代改进，并通过关键字压缩缓解上下文窗口限制。

**🔧 技术方法**

技术实现基于 LLM（GPT‑5、GPT‑4.1、GPT‑4o、Gemini‑2.5‑flash、LLaMA‑3.1‑8B、Qwen3‑8B、Gemma‑3‑4B），结合 AST、关键词提取、循环迭代提示、结构化多层聚合等手段。

**📊 数据集**

使用从 CodeSearchNet 与 CodeXGlue 中挑选的 6 个仓库（总函数量超 10k，层级深度≥7）以及对其进行标识符混淆的版本，构成功能级与层次级评测数据。

**📈 对比分析**

与 HR‑CS、CS‑BF 两大结构化提示基线相比，Agent4cs 在函数级评价指标（BLEU‑1、ROUGE‑L、BERTScore 等）上整体领先，层次级语义相似度提升约 8%，归一化关键字覆盖率提升至 38%，可读性评分亦优于基线。

**⚠️ 局限性**

局限性包括缺乏真实仓库层级摘要的标注数据、对大型 LLM 的计算成本高、较小模型在关键字覆盖和可读性上表现不佳、评测指标仍需进一步丰富，且实验主要集中在 Python 仓库，未验证跨语言通用性。

---

## 72. Risk Architecture for AI-Native Engineering Teams: An Organizational Framework for Agentic System Governance

**arXiv ID:** 2607.01421 | [PDF](https://arxiv.org/pdf/2607.01421v1)

**作者:** Laxmipriya Ganesh Iyer `[一作]` `[通讯]` (Independent Researcher), Laxmipriya Ganesh Iyer (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了AI‑native工程团队的风险架构框架，提出七维团队画像、六类失败模式，并给出了基于合成评估的覆盖度衡量方法。

**💡 创新点**

创新点在于填补了组织层面的空白，揭示了依赖边界不确定性是高危未覆盖失败的主要根源，并给出了最小可落地的风险架构蓝图。

**🔧 技术方法**

采用合成推导与可审计的评分规则，并在现有AI工具风险基准（如OWASP、CLTC）上进行仿真验证。

**📊 数据集**

使用公开的AI风险案例、技术基准以及三名资深工程经理的专家问卷作为数据来源。

**📈 对比分析**

通过离散评分（0–2）合成检测/封闭/升级三维指标，得到覆盖等级（Low/Medium/High），结果显示从纯软件到AI‑native风险覆盖呈递减趋势，低级别未覆盖单元在AI‑native阶段显著激增。

**⚠️ 局限性**

主要限制在于仅为合成框架评估，未对真实组织行为进行实证检验；场景集有限，专家样本规模小。

---

## 73. MultAttnAttrib: Training-Free Multimodal Attribution in Long Document Question Answering

**arXiv ID:** 2607.01420 | [PDF](https://arxiv.org/pdf/2607.01420v1)

**作者:** Dang Quang Thien Tran `[一作]`, Samyadeep Basu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练的多模态归因方法 MultAttnAttrib，并创建了相应的长文档评测基准 MultAttrEval，用于定位文本、图像和混合证据。

**💡 创新点**

创新点在于利用预填充阶段的注意力头子集进行跨模态检索，采用因果中介分析挑选检索头并校准阈值，从而在单通前向推理中实现高质量归因。

**🔧 技术方法**

采用注意力聚合、因果中介分析（CMA）、均值注意力评分（MAS）、阈值校准、min-max 归一化等技术，基于大型多模态语言模型（如 Qwen3‑VL‑30B）实现。

**📊 数据集**

使用自建的 MultAttrEval benchmark，来源于 MINT‑1T PDF 语料，覆盖文本、图像和混合归因三种模式，分五个领域。

**📈 对比分析**

与多种提示、RAG、captioning、LLM 等基线比较，MultAttnAttrib 在 F1 上提升 20%+，在多模态归因中达到与 GPT‑5.4 相近的水平，同时推理延迟和显存显著降低（约 7×速度、15GB 内存）。

**⚠️ 局限性**

局限性包括对图像相关性筛选不足、归因仅覆盖单图像、需要少量标注探测集进行头识别与阈值校准，以及跨模态检索头稀疏导致对精细标注的依赖。

---

## 74. Multi-Objective Exploration and Preference Optimization via Mutual Information

**arXiv ID:** 2607.01392 | [PDF](https://arxiv.org/pdf/2607.01392v1)

**作者:** Hongyan Xie `[一作]` (Beihang University), Shuangyong Song `[通讯]` (China Telecom Artificial Intelligence Technology Co., Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MI‑EPO 框架，将多目标探索与偏好对齐统一为最大化生成文本、偏好反馈与偏好向量之间的联合条件互信息；

**💡 创新点**

通过引入概率路由变量 Z 并分解互信息目标，将偏好对齐与探索不确定性控制相结合；

**🔧 技术方法**

使用 DPO（直接偏好优化）来实现 δC_k 的互信息最大化，InfoNCE 对 δW 的互信息进行估计，并在单一政策网络中实现多目标条件生成；

**📊 数据集**

安全对齐任务使用包含 10K 例的 “helpfulness/harmlessness” 数据集，助手任务使用 160K 轮对话的 “helpfulness/harmlessness/humor” 数据集；

**📈 对比分析**

与 RS、RiC、MO‑ODPO 等基线对比，在安全对齐任务中 MI‑EPO 的 HV 提升 68.8%、MIP 提升 23.2%、CRD 降低 53.2%；在三目标助手任务中 HV 提升 87.2%、MIP 提升 29.6%，同时保持较低的 CRD；

**⚠️ 局限性**

仅在小型 7B 语言模型与开源奖励模型上验证；对更大模型、更多目标或更复杂奖励场景的可扩展性与鲁棒性尚未充分验证；

---

## 75. How Should Transformers Encode Numeric Values in Electronic Health Records?

**arXiv ID:** 2607.01391 | [PDF](https://arxiv.org/pdf/2607.01391v1)

**作者:** Maria Elkjær Montgomery `[一作]` (University of Copenhagen), Mads Nielsen `[通讯]` (University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文系统评估Transformer在电子健康记录中数值特征的编码方式。

**💡 创新点**

创新点在于提出统一评估框架，并发现联合编码（FiLM）在高精度算术任务中优于离散或混合方法。

**🔧 技术方法**

采用BERT类架构（CORE‑BEHRT改进为ModernBERT）结合五种数值编码策略（离散、组合、混合离散、串联、FiLM）。

**📊 数据集**

使用丹麦Capital Region与Zealand地区的真实EHR数据，并通过合成算术任务嵌入真实序列进行验证。

**📈 对比分析**

通过比较算术任务和临床预测任务的AUROC/信息效率等指标，发现FiLM在精度敏感任务中表现最好，混合离散在稳健性与可扩展性上表现最佳，整体提升有限。

**⚠️ 局限性**

局限在于方法与架构耦合难以完全隔离、合成任务与真实临床复杂度差异、并且在临床任务中的收益有限。

---

## 76. Sign in the Air to Unlock: An Interface for authentication in Virtual and Augmented Reality Powered by Point-Voxel Cross-Attention Network

**arXiv ID:** 2607.01435 | [PDF](https://arxiv.org/pdf/2607.01435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 77. CommonRoad-Game: A Human-in-the-Loop Simulation Framework for Autonomous Driving

**arXiv ID:** 2607.01382 | [PDF](https://arxiv.org/pdf/2607.01382v1)

**作者:** Yunfei Bi `[一作]` (Technical University of Munich), Youran Wang `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本文提出了 CommonRoad-Game，一个轻量级的人工参与驾驶仿真框架，用于在 CommonRoad 场景中实时评估运动规划器并记录驾驶行为。

**💡 创新点**

创新点在于将实时人机交互与 CommonRoad 评估生态紧密结合，提供多线程时序同步、坐标对齐以及可直接生成可重现场景的能力。

**🔧 技术方法**

核心技术包括多线程主循环与规划线程、全局时钟控制、仿真与规划坐标转换、基于单轨模型的车辆动力学、硬件/键盘输入归一化以及 CommonRoad 驱动检测器。

**📊 数据集**

实验主要使用 CommonRoad 官方的公开场景集，并通过人机交互记录生成新的交互日志与场景文件。

**📈 对比分析**

通过将框架与 IDM 与反应式采样规划器联合测试，并与无同步的基线进行对比，显示最终时间误差降至 1.48 ms、平均步误差 4.1 ms，实时率 99.99%，证明同步机制显著提升了实时性能。

**⚠️ 局限性**

局限性包括仅采用二维运动学模型，缺乏高保真物理与传感器仿真；对复杂多传感器环境支持有限；并且在高负载下仍需手动调整帧跳过与漂移重置阈值。

---

## 78. Field-Deployable RF Capture System for Indoor, Outdoor, and Foliage Environments

**arXiv ID:** 2607.01368 | [PDF](https://arxiv.org/pdf/2607.01368v1)

**作者:** Lawrence Obiuwevwi `[一作]` (Old Dominion University), Sampath Jayarathna `[通讯]` (Old Dominion University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并实地验证了一个基于HackRF One和Raspberry Pi 5的便携式、低成本RF捕获平台，能够持续20 Msps IQ记录并嵌入GNSS地理坐标与SigMF元数据，适用于三种典型环境（密集树木、城市户外、室内办公）的实时频谱采集。

**💡 创新点**

①首次将持续高率IQ捕获、原生SigMF兼容、嵌入GNSS定位和完整电池供电整合于单一设备；②生成可复现、环境标记的地理化IQ数据集，为后续频谱建图与干扰分析提供标准化数据。

**🔧 技术方法**

使用HackRF One SDR、Raspberry Pi 5、u‑blox GNSS模块、USB 3.0 SSD、Python+GNU Radio 3.10.7、SigMF标准、定制文件分段与写入流程。

**📊 数据集**

自行采集的三场景IQ数据集：在2.45 GHz频段分别收集约10 分钟、每60 秒分段的SigMF文件，涵盖林区、城市户外与室内办公三种环境；未使用公开数据集。

**📈 对比分析**

通过Welch功率谱密度、时间‑频谱图、动态范围与频道占用分析进行对比；结果显示林区功率接近噪声底（-76~-82 dBFS），城市场景动态范围≈30 dB、峰值≈-40 dBFS，室内峰值≈-43 dBFS；平台写入速度75–85 MB/s无样本丢失，GNSS定位精度≤1 s，验证了系统的稳定性和可复现性。

**⚠️ 局限性**

仅单次10 分钟采样、单地点、单频段，缺乏多节点、多季节或长时段的统计验证；硬件分辨率受8位IQ限制，环境温度极端下性能未知，且未对更高频段（如3.5 GHz、5.8 GHz）的适用性进行系统评估。

---

## 79. Auto-FL-Research: Agentic Search for Federated Learning Algorithms

**arXiv ID:** 2607.01366 | [PDF](https://arxiv.org/pdf/2607.01366v1)

**作者:** Holger R. Roth `[一作]` (NVIDIA), Andrew Feng `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Auto‑FL‑Research：一个在 NVIDIA FLARE 上实现的、约束式的编码代理工作流，用来在固定 FL 合约、预算和评估路径下自动搜索和记录代码级联邦学习算法改进。

**💡 创新点**

创新点：① 将代理搜索范围限定在可变更的“mutation surface”，保持通信合约与最终全局模型评估不变；② 通过完整的候选记录、重复种子评估和保留/失败日志，将实验结果可复现、可追踪；③ 通过对比同预算的手工 HPO 控制，区分真正的 FL 机制改进与纯参数调优。

**🔧 技术方法**

技术：Python 代码静态验证、FLARE 执行引擎、自动化代理（Codex GPT‑5.5）、任务配置文件、mutation surface、运行日志、可视化绘图、重复种子评估与保留评估。

**📊 数据集**

数据集：5 个医疗交叉熵任务（Fed‑Heart‑Disease、Fed‑TCGA‑BRCA、Fed‑IXI、Fed‑ISIC2019、Fed‑Camelyon16）以及 LEAF 的 5 个交叉设备任务（FEMNIST、Sent140、Shakespeare、CelebA、Reddit）和一个合成分类任务。

**📈 对比分析**

对比方法：在每个任务/配置下，先跑固定预算的 baseline，再用代理搜索最佳候选，随后用 5 个种子重复评估；与同预算手工 HPO 控制和公开基准（FENS、FedCompass、Fed-Camelyon16 等）进行比较。性能上：IXI、ISIC2019、Camelyon16、FEMNIST、Sent140 等任务在重复评估中取得显著提升；TCGA‑BRCA 与 CelebA 的单次搜索赢得的提升在重复评估中未能复现。

**⚠️ 局限性**

局限性：① 结果高度依赖于代理模型与固定的 mutation surface；② 单次搜索结果可能出现种子敏感或过拟合；③ 只在 FLARE 约束内实验，无法直接推广到更大规模的交叉设备环境；④ 机制归因仍基于搜索轨迹，缺乏统计显著性检验；⑤ 代码级别的约束和验证尚未完全加固，未来需加强安全与治理。

---

## 80. The Wiola Architecture for Efficient Small Language Models

**arXiv ID:** 2607.01394 | [PDF](https://arxiv.org/pdf/2607.01394v1)

**作者:** Aryuemaan Kumar Chowdhury `[一作]` (Oscowl Ai), Brahma Kumar `[通讯]` (Oscowl Ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种全新的小型语言模型架构 Wiola，并提出了五个独立创新模块。

**💡 创新点**

创新点包括：3D 螺旋位置编码（SRPE）、跨层门控注意力（GCLA）、自适应 token 合并（ATM）、双流前馈网络（DSFF）以及改进的 RMSNorm（WiolaRMSNorm）。

**🔧 技术方法**

使用技术涵盖 Transformer 自回归结构、RoPE/螺旋编码、GQA、门控融合、token 合并与恢复、SiLU/GELU 激活、梯度检查点、AdamW 优化器、BF16 训练以及 HuggingFace 集成。

**📊 数据集**

主要在 WikiText‑103 或类似的英文文本数据集上进行预训练，遵循 Chinchilla 的标量比例进行 token 采样。

**📈 对比分析**

通过与 GPT‑2、LLaMA‑2、Mistral 等模型在参数量、KV‑cache 体积以及预期困惑度（PPL）等指标的对比，Wiola‑360M 约为 13–17 的 PPL，KV‑cache 仅 67 MB，显著低于 GPT‑2‑XL（421 MB），展示了更小存储且潜在相当的性能。

**⚠️ 局限性**

限制包括：ATM 在推理时被禁用、GCLA 产生跨层依赖影响流水线并行、SRPE 的径向项可能在超长序列上产生相位干扰，以及未完成大规模预训练与指令微调。

---

## 81. Bi-NAS: Towards Effective and Personalized Explanation for Recommender Systems via Bi-Level Neural Architecture Search

**arXiv ID:** 2607.01387 | [PDF](https://arxiv.org/pdf/2607.01387v1)

**作者:** Longfeng Wu `[一作]` (Virginia Tech), Dawei Zhou `[通讯]` (Virginia Tech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Bi‑NAS框架，通过双层神经架构搜索自动优化交叉注意力和特征交互，结合LLM生成个性化、有效的推荐解释。

**💡 创新点**

创新点：①将跨注意力和交互函数的搜索统一为双层NAS，消除手工设计与泛化瓶颈；②利用搜索得到的对齐用户–物品特征，用LLM的零样本提示生成更精准、个性化的自然语言解释；③在推荐与解释两个维度同时提升。

**🔧 技术方法**

技术方法包括：双层可微神经架构搜索（continuous relaxation），交叉注意力（att_0–3），元素级交互函数（plus、multiply、concat），情感分析提取Aspect–Opinion–Sentiment，LLM零样本提示（Llama‑3.1‑8B‑Instruct），二元交叉熵训练，基于特征对齐的解释生成。

**📊 数据集**

使用亚马逊四大商品类数据集：Instrument、Video、Beauty、Clothing（共计约1.6万用户、20万商品）。

**📈 对比分析**

与NCF、VBPR、CER、NAR、MANAS等基线比较，采用Hit@10、NDCG@10、MRR、Precision/Recall/F1等指标；Bi‑NAS在推荐准确性和解释质量上均显著优于所有基线，取得最高Hit@10和最高F1。

**⚠️ 局限性**

局限性：①搜索空间仍有限，可能未覆盖最优结构；②搜索过程耗时，需额外计算资源；③解释生成依赖LLM，可能出现幻觉或事实错误；④在极小或极稀疏数据集上的表现尚未验证。

---

## 82. MapDreamer: Aerial Imagery Conditioned Latent Diffusion for Lane-Level Map Generation

**arXiv ID:** 2607.01370 | [PDF](https://arxiv.org/pdf/2607.01370v1)

**作者:** Julian Brandes `[一作]` (University of Technology Nuremberg), Wolfram Burgard `[通讯]` (University of Technology Nuremberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 MapDreamer，一种基于潜在扩散模型的生成式方法，从航空图像直接生成车道级向量地图。

**💡 创新点**

创新点包括：①利用 VAE 将车道几何和拓扑压缩为连续潜在空间；②引入车道计数模块和“幽灵车道潜在”以自适应处理不同车道数量；③采用密集交叉注意力将航空图像特征注入扩散过程；④滑动窗口全局图聚合策略实现城市级地图拼接。

**🔧 技术方法**

技术：变分自编码器 (VAE)、潜在扩散模型 (Latent Diffusion Model, DiT)、密集交叉注意力、车道计数器、幽灵潜在、滑动窗口拼接、DDIM 采样。

**📊 数据集**

数据集：UrbanLaneGraph（从 Argoverse 2 提取的航空图像与车道图）。

**📈 对比分析**

与 BGFormer、LaneGNN 等非生成式基线对比；在本地和全局评估中，MapDreamer 在 GEO、TOPO、IoU 等指标上均显著优于基线，尤其在拓扑一致性和车道计数自适应方面表现突出。

**⚠️ 局限性**

局限性：在遮挡严重、缺失车道标记或罕见交叉口（如 U 型转弯）等视觉信息不足的场景下仍可能欠缺预测；模型对多样化训练数据的依赖较大，缺乏其他传感器信息。

---

## 83. Simulation Based Reward Function Validation for Multi-Agent On Orbit Inspection

**arXiv ID:** 2607.01367 | [PDF](https://arxiv.org/pdf/2607.01367v1)

**作者:** Patrick Quinn `[一作]` (Florida Institute of Technology), Madhur Tiwari `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一个可支持任意图像位置与数量的多智能体强化学习（MARL）检视框架，并通过PPO训练上/下层智能体完成卫星近地轨道检视任务。

**💡 创新点**

创新点在于：①提出了通用图像价值奖励函数，结合距离、角度分离、光照、阴影等多维度评分；②将奖励函数与3D重建结果相结合，实现从图像质量到奖励的闭环；③采用自适应奖励惩罚与课程学习，提升训练稳定性。

**🔧 技术方法**

使用的技术包括：多智能体强化学习（PPO + DTDE）、Hill-Clohessy-Wiltshire 近地轨道动力学、Ray RLlib+Gym环境、NVIDIA Isaac Sim进行光照重现、COLMAP与Instant‑NGP用于3D重建。

**📊 数据集**

使用的数据集为：由RL环境生成的仿真轨迹、控制指令和图像位置，随后在Isaac Sim中渲染的图像；无外部公开数据集，全部为内部仿真数据。

**📈 对比分析**

通过将RL生成的图像输入COLMAP和Instant‑NGP进行3D重建，对比重建完整度与重建误差；实验显示在燃料惩罚适中时能获得较为高效的轨迹，重建结果在部分区域完整，但仍有缺失，整体表现可圈可点但仍需改进。

**⚠️ 局限性**

局限性包括：奖励函数对参数高度敏感，训练过程易陷入随机或停滞；重建质量受光照变化影响大；缺乏真实硬件验证与安全保证；对不同卫星/轨道场景的泛化能力有限。

---

## 84. Multi-modal Rail Crossing Safety Analysis

**arXiv ID:** 2607.01365 | [PDF](https://arxiv.org/pdf/2607.01365v1)

**作者:** Paimon Goulart `[一作]` (University of California, Riverside), Evangelos E. Papalexakis `[通讯]` (University of California, Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套多模态管道，结合街景图像、FRA事故记录和APS评分，利用Vision‑Language Models对铁路平交道安全进行自动评估。

**💡 创新点**

创新点包括：① 将路面图像与历史事故记录融合用于安全评分；② 采用LoRA微调Gemma4实现轻量化VLM；③ 引入路由回归模块处理高低风险分离，提高连续评分性能。

**🔧 技术方法**

使用技术包括：Vision‑Language Models（Gemma 4）、LoRA参数高效微调、路由器与专属回归器、图像增强（天气、光照模拟），以及统计评估指标（RMSE、Pearson、F1）。

**📊 数据集**

使用的数据集：California Mapillary街景图像（3285张，149个交叉口）与FRA Form 57事故记录（406条，1975‑2023），合并后得到1634张图像、149个交叉口的多模态数据。

**📈 对比分析**

与仅提示图像、仅提示图像+报告、oracle提示等基线比较，微调后二分类宏F1提升至0.757，连续评分RMSE降至0.071；路由回归将Pearson提升至0.492，Oracle路由进一步提高，表明微调与路由显著提升性能。

**⚠️ 局限性**

局限性包括：① 连续评分受长尾分布影响，低风险预测过于保守；② 路由器准确率不足，成为瓶颈；③ 仅使用静态图像，未考虑动态或3D信息；④ 数据仅覆盖加州，缺乏跨区域泛化。

---

## 85. Trustworthy Runtime Verification via Bisimulation (Extended Experience Report)

**arXiv ID:** 2607.01363 | [PDF](https://arxiv.org/pdf/2607.01363v1)

**作者:** Ryan G. Scott `[一作]` (Galois, Inc.), Robert Dockins `[通讯]` (Amazon)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

开发了一款针对 Copilot 监视器编译器的翻译验证器，能够在编译后自动生成证明，证明生成的 C 代码与原始 DSL 规范在所有输入下行为等价并且安全；

**💡 创新点**

在保证有限资源与时间约束的前提下，首次将 bisimulation 的翻译验证方法与 SMT 结合，形成一种“一次性”验证流程，可直接生成可被审计员接受的形式化证明；

**🔧 技术方法**

核心技术包括：Haskell 实现的 Copilot 解释器与编译器、Crucible 的 LLVM 语义模拟、Z3（及其他 SMT 选项）进行 SMT 查询、以及基于 bisimulation 的证明生成框架；

**📊 数据集**

验证数据主要来自 Copilot 官方 test suite 以及两项航空航天案例（Well‑Clear 监测器与安全模式监测器）；

**📈 对比分析**

通过对编译生成的 C 代码与符号执行得到的状态转移进行比较，平均每个验证任务在几秒到十几秒内完成，所有案例均通过验证，证明方法在工业规模下具有可接受的性能；

**⚠️ 局限性**

局限性主要包括：对浮点运算的支持有限（采用无符号函数方式）、缺乏自动归纳不变式推导、对 Clang 及其优化的依赖导致在高优化级别下可能失效，以及对递归/无限循环等高级特性的支持尚不完善。

---

## 86. Mitigating Confirmation Bias through Hand-Drawing Videos

**arXiv ID:** 2607.01359 | [PDF](https://arxiv.org/pdf/2607.01359v1)

**作者:** Chenyu Lin `[一作]` (University of Wisconsin–Madison), Icy Zhang `[通讯]` (University of Wisconsin–Madison)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对比静态条形图与动态手绘视频，评估手绘过程对减少确认偏差的影响

**💡 创新点**

首次将体感认知视角与步进式动画相结合，探究动态手绘在先验信念情境下提升解释准确率的可能性

**🔧 技术方法**

采用混合效应逻辑回归与累计链接混合模型，对实验设计中的视频与静态图进行统计分析；使用手绘动画视频展示条形图

**📊 数据集**

使用自定义的2×2与3×3条形图情境（如枪支禁令、疫苗接种等）作为实验数据集

**📈 对比分析**

通过对照实验比较两种呈现方式的正确率，结果显示在先验信念情境下，手绘视频的正确率约提高一倍（显著水平p<0.05），表明该干预有效提升解释准确率

**⚠️ 局限性**

实验无法分离动态逐步展示与手绘本身的影响；样本仅限于简单条形图，未验证更复杂可视化或不同受试者群体的适用性

---

## 87. Chameleon: Recovering Cyber-Physical Systems from Memory Corruption Attacks via ML Surrogates

**arXiv ID:** 2607.01356 | [PDF](https://arxiv.org/pdf/2607.01356v1)

**作者:** Mohsen Salehi `[一作]` (University of British Columbia), Karthik Pattabiraman `[通讯]` (University of British Columbia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于机器学习的替代模块框架Chameleon，用于在CPS遭受内存破坏攻击后自动恢复系统。

**💡 创新点**

创新点在于在软件隔离层级生成行为等价的ML surrogate，并通过静态数据流分析自动识别每个隔离单元的输入输出变量，减少手工工作。

**🔧 技术方法**

采用LLVM编译器进行编译时隔离与插桩，使用LSTM神经网络作为surrogate，结合CFI/DFI检测实现运行时替换。

**📊 数据集**

使用在七款真实与仿真机器人车辆（ArduPilot、PX4等）上收集的1万余次任务轨迹，包含不同距离与环境条件的数据集。

**📈 对比分析**

与无保护、Gecko等方案对比，Chameleon在5次攻击任务中平均偏差为7.9 m（≤10 m阈值），比Gecko低6倍，CPU开销约8.5%且无实时约束违背。

**⚠️ 局限性**

限制包括依赖现有CFI/DFI检测精度、对极度资源受限设备的模型压缩需求、以及对训练数据污染或surrogate篡改的鲁棒性未讨论。

---

## 88. Spatial-Temporal Expert Learning for Video-based Person Re-identification

**arXiv ID:** 2607.01353 | [PDF](https://arxiv.org/pdf/2607.01353v1)

**作者:** Xiaofei Hui `[一作]` (Singapore University Of Technology And Design), Jun Liu `[通讯]` (Singapore University Of Technology And Design)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种输入感知、可扩展的专家模块，用于视频人重识别任务，通过动态激活专家并在空间与时间上选择特征来挖掘细粒度信息。

**💡 创新点**

创新点在于：① 专家选择机制能在类似样本子集上激活最相关专家，聚焦细粒度差异；② 空间-时间选择机制让专家动态分配通道至空间或时间分支；③ 可扩展机制使模型在训练期间根据需要自动增添新专家，避免手工设定专家数量。

**🔧 技术方法**

采用了动态网络技术：Gumbel‑Softmax实现专家选择、改进的 SemHash 与 max‑pool+FC 计算通道重要性；两条并行卷积分支（1×3×3 与 3×1×1）分别提取空间与时间特征；多层专家堆叠与正则化多样性损失以保持专家间互异性。

**📊 数据集**

在两个大规模视频人重识别数据集上评估：MARS（17,503条目，1,261身份）和 LS‑VID（14,943条目，3,772身份）。

**📈 对比分析**

与现有多种基于 CNN 与 Transformer 的方法比较，MARS 上取得 87.0% mAP / 91.6% Rank‑1，LS‑VID 上 81.0% mAP / 88.3% Rank‑1，均达或超过当前最优水平；ablation 证明专家选择、空间‑时间选择及多样性损失的必要性。

**⚠️ 局限性**

局限性包括：① 需要多 GPU 训练，显存占用高；② 模型复杂度提升，推理时延增加；③ 仅在两个数据集验证，跨域适应与对抗攻击等场景尚待深入；④ 对参数如 λ、专家数的敏感性，需经验调优。

---

## 89. Physically-Aware Preemptive Virtual Channels for Deadlock-Free AXI Networks-on-Chip

**arXiv ID:** 2607.01430 | [PDF](https://arxiv.org/pdf/2607.01430v1)

**作者:** Lorenzo Leone `[一作]` (ETH Zürich), Luca Benini `[通讯]` (ETH Zürich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出了一种新的轻量级虚拟通道（Preemptive VC）设计，用于在宽链路 AXI4 NoC 中实现协议级无死锁通信，并与传统多平面方案及两种 VC 变体（Naive、CreditBased）进行了对比。

**💡 创新点**

创新点在于：① 采用“预占式”链接占用策略，消除了传统 VC 中的组合后压路径，保持与多平面相同的时序；② 通过静态绑定 VC 与流，显著降低控制复杂度和面积开销；③ 在保持带宽的同时，仅以极小面积开销（约 %）和略微增加的路由资源（约 %）实现了与多平面相同的功能。

**🔧 技术方法**

主要技术包括：AXI4 NoC 细粒度流分离、虚拟通道实现（包括可预占式、信用计数、Naive 等）、多平面物理链路划分、TSMC7nm 物理实现、Cycle‑accurate 仿真与性能评估。

**📊 数据集**

使用的“数据集”为：4×4 网格中的 16 个 Snitch 核的广播传输工作负载（二叉树算法），以及在 TSMC 7nm 上的物理实现数据（面积、频率、金属轨道等）。

**📈 对比分析**

比较方法：将 Preemptive VC、Naive VC、CreditBased VC 与多平面基线在相同网格、相同负载下进行面积、频率、路由资源、传输时间（runtime）等指标的对比。结果显示：Preemptive VC 与多平面在时序上相当，面积仅略高（约 %），路由资源仅增加约 %；Naive VC 在两缓冲配置下吞吐量下降 33%；CreditBased VC 需三缓冲才能保持带宽，且在路由资源上略逊。

**⚠️ 局限性**

限制：实验仅涵盖了广播场景和单一宽链路配置，未评估更复杂的多任务流或动态流量分布；预占式 VC 仍依赖静态绑定，可能在某些动态调度环境下不够灵活；最终资源节省比例与具体链路宽度、布局约束密切相关，需在更大规模系统中进一步验证。

---

## 90. Beyond Heatmaps: Unsupervised Concept-Graph Reasoning for Interpretable Visual Explanation

**arXiv ID:** 2607.01416 | [PDF](https://arxiv.org/pdf/2607.01416v1)

**作者:** Md Mohasin Hossain `[一作]` (German Research Center for Artificial Intelligence), Daniel Sonntag `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于图的概念瓶颈模型（G-CBM），通过无监督概念发现、概念图构建和图注意力推理实现可解释的图像分类。

**💡 创新点**

创新点在于将非负矩阵分解用于概念提取，构造每张图像的概念图并通过阈值过滤提升可解释性，同时利用图注意力捕捉非线性交互，从而在单一模型中同时实现概念选择、定位与重要性解释。

**🔧 技术方法**

核心技术包括非负矩阵分解（NMF）、非负最小二乘投影、图注意力网络（GAT）、概念过滤阈值、梯度基重要性评估，以及基于冻结backbone的patch级特征提取。

**📊 数据集**

实验使用 ImageNet 子集、皮肤病数据集 HAM10000、PH2 与 Derm7pt。

**📈 对比分析**

与普通CNN基线、线性/MLP CBM 以及多种监督概念方法进行比较，G-CBM 在 ImageNet 上平均提升 3.7% 的 AUC，皮肤病数据上与监督方法相当，且通过删插分析验证了解释信度。

**⚠️ 局限性**

局限性包括对 patch 分辨率的依赖导致结构碎片化，概念提取无监督但需人工后期命名，阈值调优成本较高且缺乏自适应机制。

---

## 91. BIFROST: Bridging Invariant Feature Representation for Observation-space Sim2Real Transfer

**arXiv ID:** 2607.01410 | [PDF](https://arxiv.org/pdf/2607.01410v1)

**作者:** Yunfu Deng `[一作]` (University of Wisconsin--Madison), Josiah P. Hanna `[通讯]` (University of Wisconsin--Madison)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出BIFROST方法，实现从模拟到真实的零射门策略迁移

**💡 创新点**

创新在于利用跨域双模拟（bisimulation）对齐历史编码器，将感知与动力学差异统一为一个共享潜在空间

**🔧 技术方法**

使用GRU历史编码器、奖励预测、潜在动力学预测、Wasserstein‑1对齐、SAC强化学习和离线paired数据收集

**📊 数据集**

在sim2sim（视觉导航）和sim2real（桌面操作）数据集上测试，使用200条目标域轨迹及对应模拟段

**📈 对比分析**

与Direct Transfer、Target‑Only、BDA、Co‑Training（BC与Offline RL）对比，BIFROST在导航与操作任务中取得最高成功率/最低误差，尤其在复杂视觉与动力学双重缺口时表现最优

**⚠️ 局限性**

依赖精确的跨域对齐数据，难以处理接触突变导致的行为分歧；收集阶段需要状态估计，离线训练的覆盖范围受限

---

## 92. Rethinking Generic Object Tracking Toward Human-Level Perceptual Intelligence

**arXiv ID:** 2607.01395 | [PDF](https://arxiv.org/pdf/2607.01395v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. A global predicted-fMRI drive signal from TRIBE does not predict YouTube replay heatmaps

**arXiv ID:** 2607.01400 | [PDF](https://arxiv.org/pdf/2607.01400v1)

**作者:** Barada Sahu `[一作]` (Cabal AI), Shivesh Pandey `[通讯]` (Para AI)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

使用TRIBE模型预测YouTube视频的fMRI驱动信号，并将其压缩为每秒的参与度曲线，对比该曲线与YouTube“最常重播”热图的相关性，以检验预测神经信号是否能预测观众重播行为。

**💡 创新点**

首次将顶尖多模态脑编码模型TRIBE应用于行为预测任务，并评估无扫描预测fMRI信号在实际观看行为中的可预测性，同时提出针对SABR流媒体的获取与编码缓存方案。

**🔧 技术方法**

采用TRIBE（Llama‑3.2 + V‑JEPA2 + Wav2Vec‑BERT 1B参数多模态编码器）、全脑全局场功率（GFP）读取、位置控制的偏相关分析、低级声音与运动基线，以及SABR‑鲁棒视频采集与GPU缓存技术。

**📊 数据集**

48条具备“最常重播”热图的YouTube视频（涵盖音乐、谈话、技术等11类内容），并使用TRIBE训练所依赖的700+受试者、500+小时fMRI数据集。

**📈 对比分析**

通过与低级声音、运动基线以及位置控制的偏相关对比来评估模型；TRIBE的偏相关为+0.058，统计上不显著，且与低级声音基线无显著差异，整体性能不优于基线。

**⚠️ 局限性**

热图受偏差与噪声影响、分析窗口仅60秒、视频样本已高度热门、TRIBE被优化为fMRI准确性非行为预测、单一全脑读取未能捕捉网络级信号。

---

## 94. Computer Vision for Wildlife Monitoring: Detecting Brown Howler Monkeys using YOLO

**arXiv ID:** 2607.01396 | [PDF](https://arxiv.org/pdf/2607.01396v1)

**作者:** Gabriel Ferri Schneider `[一作]` (Pontificia Universidad Catolica De Rio Grande Do Sul), Soraia Raupp Musse `[通讯]` (Pontificia Universidad Catolica De Rio Grande Do Sul)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用YOLOv10在真实与合成图像混合训练的模型，实现对棕色呼噜猴的目标检测，并基于该模型开发自动视频筛选工具。

**💡 创新点**

创新点在于（1）证明合成数据与少量真实数据混合能显著提升检测性能，几乎可替代大量人工标注；（2）提出视频级别阈值策略，实现低误检率的自动筛选；（3）在碎片化森林环境中，首次将人工合成与真实捕捉相结合，为林冠桥监测提供可扩展方法。

**🔧 技术方法**

技术方法包括：YOLOv10卷积神经网络、Unity3D与Blender生成合成图像、迁移学习与早停、Temporal cross‑validation、基于帧计数的阈值分类。

**📊 数据集**

数据集：主数据集10,508张带猴子图像；辅助数据集①人类检测5k图像；②非人类灵长类5k图像；③Unity合成10k图像；视频数据16,179段，抽样后约5,000段用于筛选。

**📈 对比分析**

评估指标为Precision、Recall、F1‑Score、mAP@0.5；混合训练（10,000真实+10,000合成）实现F1≈0.859、mAP≈0.873，优于单纯10,000真实图像；视频筛选在阈值24帧时达到F1≈0.762、Recall≈0.838，误检率低。

**⚠️ 局限性**

限制：合成图像在行为与环境细节上仍不够真实，导致对部分/短时出现的猴子误检；阈值策略对极短出现的视频敏感，可能漏检；仅在特定森林桥环境验证，跨域泛化待进一步验证。

---

## 95. MIBE: Multi-subject Interaction Benchmark and Evaluator for Personalized Image Generation

**arXiv ID:** 2607.01383 | [PDF](https://arxiv.org/pdf/2607.01383v1)

**作者:** Zhihan Chen `[一作]` (University of California Los Angeles), Lu Xin `[通讯]` (DeerLab LLC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了针对多主体个性化图像生成的统一评估框架MIBE，包括基准MIB和评估器MIE；

**💡 创新点**

创新点在于将绑定问题拆解为存在、外观、交互三维度，构建60K对银标与4K对金标数据，并训练可解释的双头评估器；

**🔧 技术方法**

采用层次化提示设计、VLM一致性标注、双头（排名+诊断）模型以及LoRA微调等技术；

**📊 数据集**

使用的数据集为MIB银标（60K对Nano Banana vs MOSAIC）和MIB金标（4,020对来自六种生成器的人工评估）；

**📈 对比分析**

与现有指标（如CLIP、DINO、PickScore等）相比，MIE在金标上实现92.2%的人类偏好对齐率（见已公开的精度和F1指标），明显优于传统方法；

**⚠️ 局限性**

局限性包括金标规模有限、参考主体覆盖不足、银标VLM标注可能含噪，以及随着生成器快速迭代导致基准更新滞后。

---

## 96. Neuro-Symbolic Safety Guidance for Vision-Language-Action Models via Constrained Flow Matching

**arXiv ID:** 2607.01378 | [PDF](https://arxiv.org/pdf/2607.01378v1)

**作者:** William English `[一作]` (University of Florida), Rickard Ewetz `[通讯]` (University of Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种神经-符号安全指导机制，将安全约束直接嵌入 Vision‑Language‑Action（VLA）模型的流匹配生成过程中，实现对生成轨迹的预测性碰撞避免；

**💡 创新点**

创新点在于：①将整个动作轨迹视为可预测的未来序列，利用控制障碍函数（CBF）在每一步迭代中对预测轨迹进行全程约束；②在流匹配的去噪步骤中插入最小范数修正，形成迭代的神经-符号闭环，避免传统后置修正导致的“后期干预”问题；

**🔧 技术方法**

主要技术包括：流匹配动作生成、离散时间控制障碍函数、基于最小范数的约束优化（SLSQP）、欧拉积分的去噪迭代、以及与物体几何的距离估计；

**📊 数据集**

使用 SafeLIBERO 基准数据集，该数据集在 LIBERO 任务基础上加入障碍物，分为 Spatial、Goal、Object、Long 四大任务套件；

**📈 对比分析**

与基准 VLA 模型 π_0.5‑LIBERO 以及单步 CBF 方案 AEGIS 进行对比。结果显示本文方法在整体碰撞避免率（CAR）82.81% 与任务成功率（TSR）81.62% 上均优于两者；在 Long 任务中提升显著（TSR 从 43.75% 提升至 76.75%），但执行时间略高；

**⚠️ 局限性**

局限性包括：①仅对末端执行器进行碰撞检测，未考虑手臂其他部位；②需要已知障碍物几何与位置，未完全基于原始感知输入；③在某些空间任务中，约束修正导致任务成功率略低，且整体执行时间相对较长。

---

## 97. Hidden-Shot: Towards One-Shot Task Generalization for Low-Level Vision Generalist Models

**arXiv ID:** 2607.01535 | [PDF](https://arxiv.org/pdf/2607.01535v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 98. Maximum Entropy is a 10/7-Approximation Algorithm for the TSP on Half-Integral Cycle Cut Instances

**arXiv ID:** 2607.01536 | [PDF](https://arxiv.org/pdf/2607.01536v1)

**作者:** Billy Jin `[一作]` (Purdue University), David P. Williamson `[通讯]` (Cornell University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明最大熵算法在半整数环切实例（half‑integral cycle cut instances）上可以得到 10/7 近似解，并给出一种从最大熵树分布扩展到欧拉多重子图的构造。

**💡 创新点**

创新点在于：①首次在非平凡实例上提升最大熵算法的近似比（从 1.5 降到 10/7）；②提出一种不依赖马尔可夫链的 stationarity 分布构造；③通过二叉层级拆分紧集并递归扩展，保证每个紧集的边集合满足特定的奇偶性分布，从而实现期望成本控制。

**🔧 技术方法**

使用了最大熵分布、层级拆分（binary cut hierarchy）、概率扩展（stationary distribution）与递归构造、欧拉图和匹配理论、以及组合概率分析技术。

**📊 数据集**

论文未使用实际数据集，而是构造了理论实例（如 K5、envelope 图等）来展示半整数环切实例及其性质。

**📈 对比分析**

通过与 Christofides‑Serdyukov（3/2）、先前的 4/3 近似、以及已知的 11/8 下界进行比较，证明在该实例类中，最大熵算法的期望成本不超过 10/7 倍 LP 最优值，且已知最差情况下至少为 11/8 倍。

**⚠️ 局限性**

局限性：①结果仅适用于半整数环切实例；②仍无法达到 conjectured 的 4/3 上界；③对一般 TSP 实例的改进效果尚未揭示；④所用构造复杂，实际实现和实验验证缺乏。

---

## 99. Quantifying the Uncertainty of Blindly Estimated Room Embeddings Using a Dispersion-Calibrated Score

**arXiv ID:** 2607.01527 | [PDF](https://arxiv.org/pdf/2607.01527v1)

**作者:** Yang Xiang `[一作]` (University of Surrey), Philip J. B. Jackson `[通讯]` (University of Surrey)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一套三阶段框架，用于从混响语音中学习对语音内容变化鲁棒的房间嵌入，并在无下游任务监督的情况下估计单一句话的表示不确定性。

**💡 创新点**

创新点在于：①使用多视角数据结构与KL对齐将语音嵌入锚定到已训练的RIR VAE潜在空间；②加入多正样本对比学习进一步提升对语音内容变化的鲁棒性；③通过“分散校准”与排名损失训练轻量化不确定性头，将受损扰动引起的嵌入分散映射为单一不确定性分数。

**🔧 技术方法**

技术手段包括：变分自编码器（VAE）用于构建RIR潜在空间；混合CNN‑Transformer编码器提取语音特征；KL对齐和多正对比损失实现嵌入对齐与区分；分散计算与排名损失训练轻量化不确定性预测网络。

**📊 数据集**

数据集方面，使用EARS无声语料合成混响语音；3000个实测RIR来自多达20+公开数据库（如ACE、AIR‑IKS、ASMR‑IR等），RIR按身份划分为训练、验证和测试集。

**📈 对比分析**

与FiNS及两种MRL变体（SV与MV）比较，验证AP提升至0.99，RIR重建MAE下降至≈4 dB，T₆₀误差下降至≈12.9%；不确定性与表示分散的Spearman相关性为0.90，显著优于仅使用噪声/掩蔽程度或MRL-MV的0.83/0.59/0.66；在选择性预测实验中，基于不确定性的排序比基于干扰程度的排序更能稳健提升下游性能。

**⚠️ 局限性**

局限性包括：不确定性是分散校准型而非后验不确定性；Stage‑3训练需配对干净与受损视图；RIR拆分基于身份而非完全的房间分离；实验中的干扰仅限于粉红噪声与SpecAugment掩蔽，未覆盖真实场景中的多说话人、设备失配或削波等复杂失真。

---

## 100. LIB-TRAP: Standard Cell Library Hardware Trojan Risk Assessment and Prevention

**arXiv ID:** 2607.01526 | [PDF](https://arxiv.org/pdf/2607.01526v1)

**作者:** Harish Kumar Dharavath `[一作]` (University of Arizona), Soheil Salehi `[通讯]` (University of Arizona)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了标准单元库被恶意篡改导致硬件木马插入的威胁模型，并通过构造恶意库、综合 AES-128、以太网控制器和 WISHBONE DMA 三个基准电路进行实验。

**💡 创新点**

提出了基于标准单元的木马威胁模型、恶意库生成流程，以及利用机器学习对木马检测效果的评估，揭示了传统检测方法在此模型下的盲区。

**🔧 技术方法**

使用了行业级 EDA 工具（Synopsys 32nm、SkyWater 130nm）、开源工具（OpenRoad、Magic、ngspice）进行库转换与仿真；收集了设计级特征并训练 Logistic 回归、SVM、随机森林、深度神经网络等机器学习模型。

**📊 数据集**

基准数据集包括 AES-128、以太网控制器和 WISHBONE DMA 三个设计的原始、失活木马、激活木马（单个、10 个、全部）版本，共计 12 组样本，提取特征包括单元数、面积、动态功耗、静态功耗和时序裕度。

**📈 对比分析**

通过对清洁与木马感染设计的特征进行分类，比较四种 ML 算法的准确率与 F1 分数。结果显示准确率仅 24%–36%，F1 分数低于 0.4，基本等同于随机猜测，表明该威胁模型对现有检测手段极为隐蔽。

**⚠️ 局限性**

局限性包括：仅针对特定的三种基准电路和两种工艺节点；木马实现方式单一（缓冲器级别）；特征集合有限，可能无法捕捉更细粒度的异常；实验环境假设恶意库已完整渗入，未考虑更复杂的实际工厂环境。

---

## 101. Revisiting Chain-of-Thought Reasoning under Limited Supervision: Semi-supervised Chain-of-Thought Learning

**arXiv ID:** 2607.01511 | [PDF](https://arxiv.org/pdf/2607.01511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 102. Parameter Golf: What Really Works?

**arXiv ID:** 2607.01517 | [PDF](https://arxiv.org/pdf/2607.01517v1)

**作者:** Prashanna Mani Paudel `[一作]` (University of Wyoming), Shivanand Venkanna Sheshappanavar `[通讯]` (University of Wyoming)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Parameter Golf 挑战进行系统分析，收集 2,037 条 PR 记录，筛选 1,430 有效提交，计算 BPB 并评估 84 个优化技巧对 BPB 的影响。

**💡 创新点**

将社区挑战转化为公开可复现的 ablation 实验，并通过 frontier‑内校正揭示真正有效的技巧，证明经典 n‑gram 混合与测试时训练在小预算下最有效。

**🔧 技术方法**

使用了 Int6 QAT‑STE 量化、稀疏注意、Brotli 压缩、SP8192 词表、滑动窗口评估、FP16 词嵌入、TTT、n‑gram backoff 等多种技术。

**📊 数据集**

使用 FineWeb 验证集（字节级文本）和 SentencePiece 词表进行实验。

**📈 对比分析**

通过 BPB（bits‑per‑byte）作为无词表依赖指标，对 1,430 评分提交进行 Δ_k 相关性分析，并在 429 条 frontier 内重新计算，结果显示 BPB 从 1.2244 降至 1.058（13.6% 改善）。

**⚠️ 局限性**

Δ_k 仅为观察性相关，受采纳时机偏差；技巧检测基于关键字匹配，可能漏检；低于 0.9 BPB 的提交被排除，结果仅适用于 16 MB/10 min H100 环境。

---

## 103. Asymmetric Trading Prophets

**arXiv ID:** 2607.01516 | [PDF](https://arxiv.org/pdf/2607.01516v1)

**作者:** Gagan Aggarwal `[一作]` (Google Research), Mingfei Zhao `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了异质买卖价格（Asymmetric Trading Prophet）问题，设计了在线算法并给出了竞争分析；

**💡 创新点**

创新点在于将传统的对称买卖价格模型推广为异质买卖价格，结合 LP 逼近、区间分解和在线冲突分辨技术，实现了对单容量与大容量情形的最优竞争比（上界与下界匹配到对数因子）；

**🔧 技术方法**

使用了线性规划松弛、跨区间解耦、无交叉化（uncrossing）、在线冲突分辨方案（OCRS）、独立随机化、以及随机过程分析等多种理论工具；

**📊 数据集**

本研究为纯理论分析，未使用实际数据集，而是基于概率分布模型和假设的价格分布；

**📈 对比分析**

通过竞争比（online 算法期望利润 / 离线预言家利润）与理论下界进行比较；在单容量、iid/非 iid 情况下得到常数竞争比（22/14）；在大容量、对称/非对称情况下得到 1−O(logB/√B) 的竞争比，随着容量增大可逼近最优；

**⚠️ 局限性**

局限性包括：当初始库存 B0=0 时无法获得任何竞争比；目前仅对单容量与大容量两极化情况给出结果；对非 iid 情况的竞争比仍受对数因子限制；且实验验证缺失，未验证在实际交易市场中的鲁棒性。

---

## 104. The Agentic Garden of Forking Paths

**arXiv ID:** 2607.01507 | [PDF](https://arxiv.org/pdf/2607.01507v1)

**作者:** Jiacheng Miao `[一作]` (Stanford University), James Zou `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过给大型语言模型（LLM）代理分配不同的意识形态人物设定，观察其在同一数据集与研究问题下生成的分析路径与结论的差异，并提出通过“Agentic Bootstrap”采样多条可接受的分析路径，计算新颖的 m‑value（分析空间置信度）来评估结论的稳健性。

**💡 创新点**

创新点在于：① 通过人物设定揭示AI代理在分析选择上的偏向与人类研究者类似的“forking paths”现象；② 引入 m‑value 这一量化分析空间不确定性的指标；③ 设计 Agentic Bootstrap 方法，使得通过 AI 生成的分析空间可被经验化、可复现地用于检验报告结论是否属于该空间的极端，从而提升对研究可信度的评估。

**🔧 技术方法**

主要技术包括：使用大型语言模型（Claude Sonnet 4.6、OpenAI Codex GPT‑5.4）构建分析代理；Agentic Bootstrap 采样与记录可接受的分析路径；CatBoost 分类器预测分析结果符号；独立 AI 与人工专家对结果进行二元 PASS/FAIL 审核；以及 m‑value 计算与统计检验。

**📊 数据集**

使用了四个领域的公开数据集：国际社会调查计划（ISSP）关于移民与福利支持的问卷；国家健康与营养检查调查（NHANES）关于咖啡与健康的资料；青少年风险行为监测调查（YRBS）关于社交媒体与青少年心理健康的问卷；以及美国肠道菌群项目（American Gut）关于 Firmicutes/ Bacteroidetes 比例与 BMI 的微生物组数据。

**📈 对比分析**

与传统人类多分析者实验（42支团队）比较时，AI 代理在不同人物设定下的效应估计差距达 72% 之人类差距；在同一分析空间中，AI 代理的最终报告通过 AI 与人类复核的通过率均高于 80%；m‑value 分析表明约 13.5% 的人类报告落入极端 5% 的分析空间，显示显著的选择性报告。成本方面，构建 4,392 条分析路径仅约 100 美元，远低于传统多分析者研究的费用。

**⚠️ 局限性**

局限性包括：① m‑value 的计算依赖于代理的搜索策略、人物设定和提示，若代理无法覆盖全部合理分析路径，则可能低估真实不确定性；② Agentic Bootstrap 的结果与模型版本、更新有关，需固定并报告完整协议以保证可复现；③ 仅评估分析路径的可接受性，未涵盖其他偏倚（如数据选择、结果解释等）；④ 目前仅在四个单一研究问题上验证，尚未推广至更广泛的科学领域；⑤ 代理生成的分析空间虽大但仍可能缺少少见但合理的模型或变量组合。

---

## 105. Anti-Prompt: Image Protection against Text-Guided Image-to-Video Generation

**arXiv ID:** 2607.01499 | [PDF](https://arxiv.org/pdf/2607.01499v1)

**作者:** Yeonghwan Song `[一作]` (GIST), Jeany Son `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种对图像进行不可察觉扰动的保护方法，以阻止文本引导的图像到视频生成。

**💡 创新点**

通过同时抑制文本相关注意力并增强视觉注意力，实现对文本提示的普适干扰；引入基于Video-LLM的失败检测评估协议。

**🔧 技术方法**

利用扩散模型注意力机制（Full-Attention、Cross-Attention）进行损失优化，并使用VAE编码器攻击和Video-LLM评估器。

**📊 数据集**

在VBench Image-to-Video数据集上进行评测，并生成GPT-4合成的未见提示。

**📈 对比分析**

与I2VGuard对比，使用VBench指标和Video-LLM评分均显示更低分（更强保护），在人类主观评估中排名最高，且显著减少运算与显存需求。

**⚠️ 局限性**

对未公开数据或更大规模模型的验证有限，且在某些结构一致性评估中表现略逊。

---

## 106. SLFS: a Flexible, Low-Cost Distributed File System Using Serverless Designs

**arXiv ID:** 2607.01486 | [PDF](https://arxiv.org/pdf/2607.01486v1)

**作者:** Cheng Hao `[一作]` (Northeastern University), Ji-Yong Shin `[通讯]` (Northeastern University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

本文设计并实现了一个完全基于无服务器（Serverless）计算的分布式文件系统 SLFS，支持文件数据和元数据的读写。

**💡 创新点**

核心创新点包括：①将文件操作映射到无服务器函数上，使用短生命周期多线程服务器实现热启动并显著降低冷启动；②采用哈希映射和无状态函数实现对键值存储的简单读写，避免传统 inode 结构；③在函数内部实现 LRU 写穿缓存并通过策略管理函数生命周期，兼顾性能与成本。

**🔧 技术方法**

技术实现涵盖：OpenWhisk 无服务器框架、Docker 容器、键值存储（S3、LevelDB、Cassandra）、多线程服务器、两阶段提交、LRU 缓存、Chord 风格一致性哈希、Zookeeper 成员服务。

**📊 数据集**

实验使用的主要数据集为 Azure Function Blob 追踪（14 天 855 应用），YCSB、IOzone 以及自定义均匀随机工作负载。

**📈 对比分析**

通过与 AWS EFS、Ceph、InfiniCache 以及 HopsFS 等基线对比，SLFS 在吞吐量上可比 EFS 提升 2.4–4.3 倍、与 Ceph 比提升 16–24 倍，成本则比 EFS 低 12–68%，比 Ceph 低 48–63%，并显著减少冷/热启动次数。

**⚠️ 局限性**

限制包括：仅支持块级文件操作，依赖键值存储的强一致性（否则需额外两阶段提交）；在持续高负载场景下函数长时间运行仍会产生额外成本；缺乏完整的日志/快照恢复机制，需要进一步完善文件系统的容错与持久化特性。

---

## 107. Beyond Next-Token Prediction: An RLVR Proof of Concept for Tool-Use Agents on Atlassian Workflows

**arXiv ID:** 2607.01465 | [PDF](https://arxiv.org/pdf/2607.01465v1)

**作者:** Karthikeya Aditya Vissa `[一作]` (Centific), Abhishek Mukherji `[通讯]` (Centific)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了五个模拟 Atlassian（Jira REST v3 与 Confluence v2）API 的合成环境，设计可验证的奖励函数，并使用 Qwen3 模型通过 RLVR（基于 Group‑Relative Policy Optimization 的强化学习）训练工具调用代理；随后与仅使用提示的基线模型进行对比。

**💡 创新点**

首次在合成企业 SaaS 工作流中直接采用可验证奖励实现强化学习，无需实时 API、学习判定器或人工标注，证明了通过 RLVR 能显著提升在结构严谨、字段繁多的 API 任务中的表现。

**🔧 技术方法**

使用的技术包括：RLVR、GRPO、TRL 库、Qwen3 1.7B 与 4B 语言模型、OpenAI 兼容的函数调用接口、HuggingFace Inference Router、合成 API 环境与奖励判定器。

**📊 数据集**

使用的“数据集”是自制的合成数据：每个场景 6–12 条提示，共五个场景（Jira 票据转换、子任务创建、Confluence 页面创建、页面标签、跨产品联合创建）。不使用公开真实数据。

**📈 对比分析**

比较方法：在同一套可验证奖励检查器下，对同一批提示分别评估提示基线 Qwen3（1.7B、4B）与 RL‑训练后模型的平均奖励；性能上，RL 训练的策略平均奖励从基线的 0.35–0.92 提升至 0.95–1.00，最大提升出现在 Confluence 页面创建任务（0.35→1.00，提升 0.65）。

**⚠️ 局限性**

局限性：奖励函数需手工设计，难以扩展到大量端点；部分场景奖励饱和，限制了进一步提升；仅使用 6–12 条提示，缺乏对未见提示的泛化评估；合成环境未覆盖真实 API 的错误、速率限制、权限等异常；与其他强化学习或 fine‑tune 方法相比的优势未做充分验证。

---

## 108. SE(2) Navigation Mesh

**arXiv ID:** 2607.01454 | [PDF](https://arxiv.org/pdf/2607.01454v1)

**作者:** Shuyang Shi `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了SE(2)导航网格与ASA路径规划系统，用于复杂多层环境中地面机器人的全局导航。

**💡 创新点**

创新点在于将机器人朝向纳入网格构建，区分安全与受限通行区域，并通过分层结构实现yaw-依赖的可通行性与翻转连通。

**🔧 技术方法**

使用了三角网格生成、连续yawn footprint mask、分层图与连通性建模、A*+字符串拉伸+二次A*路径规划、Slab分层局部更新、VoxelBlox TSDF、凸多边形合并等技术。

**📊 数据集**

利用HM3D/HSSD室内3D场景数据集以及真实环境点云进行实验。

**📈 对比分析**

通过与传统NavMesh、RRT/RRT*/PRM等采样基规划器比较，ASA在受限场景中成功率更高、规划时间更短、路径成本更低；在仿真中可通行面积提升50+平方米，实时在线更新可维持4Hz。

**⚠️ 局限性**

局限性包括固定高度假设、低矮物体误判、缺乏语义信息、无法处理可变高度机器人以及未集成自主探索策略。

---

## 109. MMAO-Cls: Metabolic Multi-Agent Optimization for Joint Feature Selection and Classifier Tuning

**arXiv ID:** 2607.01539 | [PDF](https://arxiv.org/pdf/2607.01539v1)

**作者:** Jinliang Xu `[一作]`, Liping Ma `[通讯]` (Seventh Medical Center Chinese PLA General Hospital)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于代谢多智能体优化器（MMAO）的混合空间外循环优化框架MMAO-Cls，用于同时进行特征子集选择和分类器超参数调优。

**💡 创新点**

创新点在于将特征掩码与超参数统一映射到一个代谢经济循环中，通过私有能量、公共预算、角色漂移与生命周期替换来实现自适应的混合变量搜索，并引入特征信息先验驱动的预算适配与复合目标函数，兼顾准确率、特征稀疏性和过拟合惩罚。

**🔧 技术方法**

使用的技术包括代谢多智能体优化（MMAO）、二进制特征掩码与连续/序数超参数的混合编码、特征互信息先验、目标函数的稀疏性与过拟合正则化、角色漂移和动态种群控制等。

**📊 数据集**

在七个标准表格数据集上进行实验：Breast Cancer、Digits、Ionosphere、Iris、Sonar、Vehicle、Wine，分别配合RBF SVM、k-NN和逻辑回归三类经典分类器。

**📈 对比分析**

与随机搜索、轻量级GA、PSO、以及无公共共享消融版等基线比较，MMAO-Cls在目标函数上排名第二，在留出测试集上的平均得分略高于随机搜索和GA-lite，但与PSO-lite相近；其最大优势在于能获得最紧凑的特征子集（平均特征比例约0.49），体现了在准确率与特征稀疏性之间的良好平衡。

**⚠️ 局限性**

局限性包括：共享预算的优势尚未显著验证；部分数据集的紧凑偏差可能导致性能欠佳；基线相对轻量，未与更强的贝叶斯或SMAC等配置器比较；评估预算未完全按函数评估量化；目标函数单目标化，未实现多目标Pareto分析。

---

## 110. Multi-Head Recurrent Memory Agents

**arXiv ID:** 2607.01523 | [PDF](https://arxiv.org/pdf/2607.01523v1)

**作者:** Jiatong Li `[一作]` (University of Wisconsin-Madison), Sharon Li `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出多头循环记忆框架 MHM，通过将记忆拆分为多个独立头并采用阶段化的选择‑更新策略，解决长上下文下记忆保留瓶颈。

**💡 创新点**

创新点在于：①用多头结构替代单块记忆，显著减轻覆盖压力；②引入分阶段的 select‑then‑update 机制；③轻量化实例 MHM‑LRU 通过 LRU 规则实现均匀头利用且无额外标记，完全无训练需求。

**🔧 技术方法**

核心技术：多头记忆结构、选择阶段与更新阶段的分离、Least‑Recently‑Updated（LRU）头选择规则；在 Qwen2.5‑14B‑Instruct、Qwen2.5‑32B‑Instruct、gpt‑oss‑120b 等 LLM 基础上实现。

**📊 数据集**

使用长上下文 QA 与推理基准：RULER‑HQA、BABILong、BABI‑Long，覆盖 7K–1M+ token 的多任务测试。

**📈 对比分析**

与单头基线 MemAgent、ReMem 对比：在 896K tokens 时记忆保留率从 <30% 提升至 73.96%，准确率从 21.6% 提升至 49.7%；在 1M tokens 时准确率从 25.3% 提升至 41.4%；在 100K–1M 范围内保持 48–50% 的稳定性能；相对单头方法无显著计算或内存开销。

**⚠️ 局限性**

局限：头数 H 固定，最佳 H 随上下文长度与任务结构变化；目前未实现动态头分配或自适应调整。

---

## 111. Overthink-Triggered Slowdown Attacks on LVLM-Based Robotic Systems

**arXiv ID:** 2607.01518 | [PDF](https://arxiv.org/pdf/2607.01518v1)

**作者:** Qiang Han `[一作]` (Michigan Technological University), Bo Chen `[通讯]` (Michigan Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究如何在大型视觉语言模型驱动的机器人系统中，利用嵌入可读场景文字诱发模型“过度思考”，从而显著延长推理延迟，构建三阶段黑盒搜索框架；

**💡 创新点**

首次系统识别并优化可在图像输入中实现的高影响力文字触发器，并证明其跨模型、跨场景可迁移；

**🔧 技术方法**

采用数据驱动的词汇挖掘、遗传算法文本优化和基于前缀的代理评分，减少搜索成本；

**📊 数据集**

使用BDD100K道路场景图像作为评估数据集，包含训练、校准与测试拆分；

**📈 对比分析**

相较于随机、Naïve-CoT、先前文本攻击，所提触发器在三种主流LVLM上实现1.15×–6.96×的延迟放大，并在物理打印实验中达到4.74×；

**⚠️ 局限性**

受限于摄像头分辨率、光照变化及场景文字可读性，实验仅在受控环境下验证，且对对抗性更强或非图文输入的LVLM鲁棒性未知；

---

## 112. Disentangling Pictorial Cue Understanding from Language Bias in VLMs via Depth Ordering Task

**arXiv ID:** 2607.01503 | [PDF](https://arxiv.org/pdf/2607.01503v1)

**作者:** Yiqian Liu `[一作]` (York University), John K. Tsotsos `[通讯]` (York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了 Vision‑Language Models 在深度排序任务中的表现，构建了可控的 Odd‑One‑Out Depth (O3‑D) 数据集，系统评估了视觉与语言对深度理解的影响。

**💡 创新点**

提出了独立控制九种图像深度线索的实验框架，构造了 O3‑D 数据集，并定义了量化视觉与语言敏感度的新指标。

**🔧 技术方法**

利用 Kubric 渲染 3D 场景、生成多种深度线索、基于多选 VQA 形式的提示、Chain‑of‑Thought 与 In‑Context Learning 进行实验，计算 SDGM 与 bias_NF 指标。

**📊 数据集**

使用自建的 O3‑D 数据集（37K 实景与合成图像，147K 视觉问答对），并与 DepthAnythingV2、DepthCues 等基准进行对照。

**📈 对比分析**

评估 12 个 VLM（含开源与商用），比较深度线索、语言清晰度、CoT/ICL 效果；结果显示大多数模型在深度排序上仅略高于随机，语言敏感度远大于视觉敏感度。

**⚠️ 局限性**

模型对单个或组合深度线索的利用不足，CoT/ICL 效果有限；数据集仅覆盖平面深度线索，缺乏运动与双目线索，缺少实时动态场景评估。

---

## 113. Procedural Memory Distillation: Online Reflection for Self-Improving Language Models

**arXiv ID:** 2607.01480 | [PDF](https://arxiv.org/pdf/2607.01480v1)

**作者:** Ye Liu `[一作]` (Salesforce AI Research), Semih Yavuz `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出Procedural Memory Distillation（PMD），通过在线构建经验、洞察与行为三层程序记忆，并在自我教师中使用此记忆来提升语言模型的自我蒸馏训练，从而在推理时实现无记忆依赖。

**💡 创新点**

创新点在于将跨回合的经验持续积累为可抽象的程序记忆，并实现策略与记忆的共进化，使得模型能在训练期间内化可复用的推理策略而非仅靠单回合奖励。

**🔧 技术方法**

主要技术包括自我反思抽象、三层记忆（经验、洞察、行为）构建、基于记忆的自我教师（memory‑conditioned self‑distillation）以及在训练中保持记忆与策略同步更新。

**📊 数据集**

使用了两个可验证的基准数据集：科学多选推理基准 SciKnowEval（涵盖生物、化学、物理、材料科学）和代码生成基准 LiveCodeBench，均提供可执行单元测试反馈。

**📈 对比分析**

与基线GRPO和SDPO相比，PMD在SciKnowEval上平均提升3.8–5.5%（Qwen3‑8B/OLMo3‑Instruct‑7B），在LiveCodeBench上提升7.9–13.6%；实验表明记忆的持续性和共进化是主要收益来源。

**⚠️ 局限性**

局限性在于仅验证了固定任务分布下的回合重复经验，未覆盖多任务或长期代理环境，且记忆构建与抽象对不同任务的通用性仍需进一步探索。

---

## 114. World Feedback for Clinical Agents: Diagnosing RL in FHIR Environments

**arXiv ID:** 2607.01470 | [PDF](https://arxiv.org/pdf/2607.01470v1)

**作者:** Ananya Mantravadi `[一作]` (Centific Global Solutions, Inc.), Abhishek Mukherji `[通讯]` (Centific Global Solutions, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 MedAgentBench 进行审核，修正了 41.7% 的无操作（silent‑finish）上限，并构造了 508 题目的 MAB‑v3 基准；在此干净环境下，使用 Qwen3‑8B 通过 GRPO 进行从世界反馈的强化学习实验，并与监督微调（SFT）以及前沿模型进行对比。

**💡 创新点**

创新点包括：① 提出了基于任务结构的三类（Decision、Lookup、Format‑Knowledge）RL 可学习性分类，并用其诊断 RL 与 SFT 的差距；② 证明了“世界反馈”在临床协议执行任务中可行，但存在格式知识和能力天花板两大障碍；③ 通过 SFT+RL 的组合方案提供了改进路径，展示了两者互补的效果。

**🔧 技术方法**

技术手段主要包括：
- 构建可重现、确定性的 FHIR 环境（HAPI FHIR 服务器快照）
- 设计可审计的规则验证器（rule‑based verifier）
- 为 RL 提供细粒度奖励分解（终端奖励、动作奖励、误操作惩罚、空闲惩罚）
- 使用 GRPO（基于梯度的策略优化）在 Qwen3‑8B 上进行训练
- 通过程序化生成 SFT 训练样本（rule‑based demos）并使用 LoRA 微调。

**📊 数据集**

数据集为 MAB‑v3，包含 508 个临床协议执行任务，跨 20 种任务类型，约 100 名匿名患者（每种类型 30 份实例，经过 1:1 分支平衡后 463 个需要操作，45 个无需操作）。

**📈 对比分析**

对比方法：
- 前沿模型（GPT‑5.5、Gemini 3.1 Pro、GPT‑4o、Llama 4 Maverick、Mistral Large、Claude 4.6）在官方 harness 上报告 p@1 ~ 78–83%；
- Qwen3‑8B 基础模型在 MAB‑v3 上 p@1 16.6%；
- SFT 训练后 p@1 34.1%；
- 纯 RL（GRPO）训练后 p@1 18.2%。
- SFT 与 RL 的差距为 15.9 pp，表明世界反馈不足以提供格式知识，SFT 负责注入代码信息，RL 负责学习决策逻辑。

**⚠️ 局限性**

限制包括：
- 仅针对已知协议执行的结构化任务，无法处理开放式临床推理或判断；
- 数据集规模有限，且任务实例数在 1–2 的类型噪声大；
- 需要手工设计 verifier 与 FHIR 环境，难以快速迁移到其他医院；
- 纯 RL 受限于格式知识和能力天花板，需与 SFT 结合；
- 结果对真实临床工作流程的可迁移性尚未验证。

---

## 115. Token Geometry

**arXiv ID:** 2607.01455 | [PDF](https://arxiv.org/pdf/2607.01455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. A Cost-Aware, Paired Protocol for Auditing Dynamic Tool Synthesis in Agentic Video Question Answering

**arXiv ID:** 2607.01469 | [PDF](https://arxiv.org/pdf/2607.01469v1)

**作者:** Aseel Mohamed `[一作]` (Texas A&M University), Hasan Kurban `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Dynamic‑SAGE，一种能够离线合成、验证并持久注册可复用复合工具的智能 VideoQA 框架，显著提升推理效率与准确率。

**💡 创新点**

创新点在于：①提出了面向成本的配对评估协议，能够同时量化准确率与推理成本；②构建了多代理合成管道（签名、实现、验证），实现从原始工具自动生成持久化复合工具；③通过复合工具将频繁多步推理压缩为单步调用，改变了成本分布。

**🔧 技术方法**

技术手段包括：多代理工具合成（签名‑实现‑验证），LLM 驱动的推理控制器，基于可见工具调用的成本度量，McNemar 统计检验与配对自举置信区间，VADAR 风格的代理编排，以及对 token 与费用的细粒度跟踪。

**📊 数据集**

使用了 SAGE‑Bench 作为评估基准，涵盖 1,744 条问答（含 MCQ 与开放式），覆盖多时长、不同模态、难度层级及视频长度等多维属性。

**📈 对比分析**

通过对同一问题的两系统（Static‑SAGE vs Dynamic‑SAGE）进行配对准确率与可见工具调用差值的统计，形成六类结果分组；实验显示 Dynamic‑SAGE 在准确率上提升 7.52 点（p<0.001），可见工具调用减少 28%，推理轮次与延迟下降 28%/6%，但 token 消耗与经济成本分别增加 34%/26%。

**⚠️ 局限性**

局限性在于：提升不均衡，主要集中在视觉、开放式、难度较高或长视频；合成工具的采用率不高，深度工具使用稀少；在某些样本中会出现准确率退化；成本结构转移导致总 token 与费用上升，需要进一步优化工具质量与路由策略。

---

## 117. Grounded Optimization: A Layered Engineering Framework for Reducing LLM Hallucination in Automated Personal Document Rewriting

**arXiv ID:** 2607.01457 | [PDF](https://arxiv.org/pdf/2607.01457v1)

**作者:** Shashank Indukuri `[一作]` (DePaul University), Adarsh Agrawal `[通讯]` (Stony Brook University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个五层防御框架（Temporal Context Validation、Cross-Domain Contamination Detection、Structural Invariant Enforcement、Prompt-Level Grounding、Evaluator Agent QA Gate）用于大语言模型对简历的重写，目标是消除四类幻觉（时间错报、跨域污染、结构突变、内容捏造）

**💡 创新点**

提出了针对个人文档优化的四类幻觉分类法，并在此基础上设计了基于时间上下文、正则化服务词典、结构计数、提示层约束和独立评估代理的分层防御体系，首次在简历优化领域实现多模型、多温度下的系统化评估

**🔧 技术方法**

技术包括：1）将技术发布年份映射嵌入提示实现时间上下文校验；2）构建257项云服务正则词典进行确定性污染检测；3）对角色和要点计数做前后比较实现结构完整性；4）在提示中加入不可变规则进行内容基线约束；5）使用独立LLM作为评估代理进行反向审核；6）利用LangGraph实现多代理流水线和后备合并

**📊 数据集**

使用25份合成简历（覆盖14个行业，共42个角色、188个要点）与5份针对不同云/AI职位的对抗性工作描述；以及构建的257项云服务词典用于污染检测

**📈 对比分析**

通过消融实验、跨模型（GPT‑4.1‑nano、GPT‑4o‑mini、Llama‑3.1‑8B）和温度（0–1.0）评估，基线幻觉率从 2.48/incidence 降至 0.12（约95% 下降），在最差模型/高温下仍保持 0.04–0.24 的低幻觉率；单层 Prompt‑Level Grounding 在 t=0 时可达到 0 幻觉，但在其他设置中需配合确定性层才能保持性能

**⚠️ 局限性**

主要局限：1）跨域污染检测与 Layer‑2 防御共享同一函数，导致评估时检测结果不具独立性；2）仅使用合成数据，真实简历的多样性与长尾平台未覆盖；3）检测器对误报（尤其 H4）敏感；4）未评估幻觉幅度与真实影响；5）高方差导致统计置信度低；6）缺乏人类真实标注验证

---

## 118. From Anatomy to Smells: An Empirical Study of SKILL.md in Agent Skills

**arXiv ID:** 2607.01456 | [PDF](https://arxiv.org/pdf/2607.01456v1)

**作者:** David Boram Hong `[一作]` (University of California, Irvine), Iftekhar Ahmed `[通讯]` (University of California, Irvine)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对现实世界中 238 个 Agent Skill（SKILL.md 文件）进行系统性分析，提出 13 类高层语义组件和 44 类低层语义组件的分类体系，并构建了自动检测器来识别违反最佳实践的“skill smell”。

**💡 创新点**

首次把 Agent Skill 文件视作软件工件进行研究，既制定了完整的语义组件分类，也提出并验证了“skill smell”概念，展示了文件质量评估和改进的可行路径。

**🔧 技术方法**

采用定性内容分析、文献综述（29 篇来源）以及自动化文本检测技术（正则表达式和模式匹配）来识别并量化 skill smell。

**📊 数据集**

使用公开可获取的 238 个真实 Agent Skill 文件（SKILL.md）作为实验数据集。

**📈 对比分析**

与已有的最佳实践对照，构建的检测器在实际文件上发现 99% 以上出现至少一种 skill smell，表明实践与理论存在显著差距；但文章未进行性能基准对比，仅展示检测覆盖率与缺陷持久性。

**⚠️ 局限性**

局限性包括：样本来源仅限于公开仓库，未涵盖企业内部或私有技能；检测器只能发现违规，而不提供自动修复方案；并未评估不同技术栈或不同 LLM 代理对 skill smell 的影响。

---

## 119. Robust and Explainable 3D Mode Shape Recognition Using Region-Aware Graph Neural Networks

**arXiv ID:** 2607.01522 | [PDF](https://arxiv.org/pdf/2607.01522v1)

**作者:** Tong Duy Son `[一作]` (Siemens Digital Industries Software), Theo Geluk `[通讯]` (Siemens Digital Industries Software)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个 Canonical Engineering Graph Representation，将不同车辆的 BiW 模型转化为共享的结构语义图，并基于该图实现可解释、可迁移的模式识别；

**💡 创新点**

创新点在于以工程语义区域为图节点、以物理关系为边构造通用图结构，解耦几何离散化与学习；同时引入区域感知池化与工程先验特征融合，提升可解释性与跨车型迁移；

**🔧 技术方法**

采用图注意网络（GAT）进行图编码，结合工程区域描述符进行特征融合，并使用层级分类器完成三阶模式识别；

**📊 数据集**

使用四款车辆的模拟与实验 BiW 数据（共计 4 组不同几何/网格/传感器布局），通过物理驱动的数据增强扩充至 310 条标注样本；

**📈 对比分析**

与仅在单车训练的基准相比，跨车训练的模型在 Level‑1、Level‑2 和整体分类上的准确率分别提升至 100%、98.7% 与 99.2%，并且在不同车辆间保持高准确性；

**⚠️ 局限性**

局限性包括：仍需工程师提供少量标注样本；图构造和区域划分需手工或基于规则自动完成；模型在极端结构变形或未见类别时可能表现不佳。

---

## 120. The risk of KV cache compression

**arXiv ID:** 2607.01520 | [PDF](https://arxiv.org/pdf/2607.01520v1)

**作者:** Lukas Haverbeck `[一作]` (RWTH Aachen University), Marco Pavone `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提出一种基于软最大注意力的 KV 缓存压缩理论框架，将压缩问题转化为 Hilbert 空间中的稀疏平衡问题，并给出最优的极值风险下界与上界；在此基础上设计了可在预填充（prefill）和自回归解码中使用的局部压缩器，支持因果掩蔽与批量并行计算；

**💡 创新点**

创新点包括：①把 KV 缓存压缩建模为稀疏测度逼近并引入响应协方差度量；②证明查询感知与查询无感知压缩的最优风险分别为谱截断与迹比值；③提出可在实际 Transformer 推理中实现的局部压缩器，满足预填充和自回归两种场景；④在 LongBench 上验证理论与实现的紧密对应。

**🔧 技术方法**

技术手段主要包括：稀疏平衡理论、Hilbert 空间协方差分析、随机采样与聚类（保护主成分）压缩器、并行 Blelloch 归约、局部随机重整器（local reducer）以及理论上可实现的查询无感知/感知压缩策略。

**📊 数据集**

实验使用 LongBench‑v2 的 “long” 子集，评估 Qwen3‑32B 模型，测试压缩率约 95% 相对于完整 KV 缓存。

**📈 对比分析**

对比方法包括：完整 KV 缓存（Full KV）、文献中的 ScissorHands、SnapKV、StreamingLLM 以及本文的两种压缩器（随机采样和聚类）。结果显示随机采样已接近 Full KV 的准确率；聚类在保护 128 维主成分时能完全恢复 Full KV 的性能，而在 64 维时略低；相比文献基线，本文方法在预填充与解码阶段均使用压缩，显著减少 KV 缓存占用。

**⚠️ 局限性**

局限性包括：①需满足“尖锐注意力”与“共同几何”假设，实际模型对这些假设的适用性有限；②聚类压缩器对受保护维数的选择敏感，超参数调优耗时；③未系统评估推理速度与能耗改进；④实验仅在单一大模型和数据集上验证，泛化性待进一步验证。

---

## 121. Mind the Trust Gap: Identifying (Mis)alignments in Teacher-Student Views Toward Control and Agency in K-12 Classroom AI

**arXiv ID:** 2607.01506 | [PDF](https://arxiv.org/pdf/2607.01506v1)

**作者:** Tomohiro Nagashima `[一作]` (Saarland University), Man Su `[通讯]` (Leibniz Institute for Knowledge Media)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在德国的K‑12课堂中，使用情境故事板与速度约会法，访谈了16名学生和15名教师，系统对比了他们对AI决策控制的观点，揭示了信任差距与对齐/不对齐的主题。

**💡 创新点**

创新点在于首次以细粒度主题匹配（低层主题438个）对学生与教师的视角进行精确对齐/不对齐分析，并系统阐述了信任、情感、监控、数据共享与自主决策五大核心维度中的冲突与共识。

**🔧 技术方法**

采用的技术包括情境故事板、速度约会访谈、录音转写、开放编码、亲和图划分（Affinity Diagramming）以及低层主题的配对匹配分析。

**📊 数据集**

数据集为：16名学生（平均年龄14.19岁）和15名教师（平均教学年限8.21年）的访谈录音与转录文本，共约31.4小时的视频资料。

**📈 对比分析**

通过对学生与教师低层主题进行一一配对，形成55对主题，进一步聚合成11个中层主题和5个高层主题，展示了各维度的对齐与差异；虽然没有传统意义上的数值性能指标，但该方法在细粒度视角匹配上表现出较高的可解释性和深度洞察。

**⚠️ 局限性**

局限性包括：故事板情境可能引导偏见；样本规模与文化背景（仅德国）有限；仅聚焦ITS类AI，未覆盖生成式AI等新技术；学生年龄跨度大导致视角不均衡；以及补偿机制可能影响受访者表达。

---

## 122. CoPersona: Collaborative Persona Graphs for Robust LLM Personalization

**arXiv ID:** 2607.01485 | [PDF](https://arxiv.org/pdf/2607.01485v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 123. Kani: A Model Checker for Rust

**arXiv ID:** 2607.01504 | [PDF](https://arxiv.org/pdf/2607.01504v1)

**作者:** Rémi Delmas `[一作]` (Amazon Web Services), Carolyn Zech `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

Kani 是一个针对 Rust 的模型检查器，提供从无注解的 panic‑free 检查到通过函数/循环契约实现的无穷大正确性证明。

**💡 创新点**

创新点在于将 BMC 与可扩展的契约语言相结合，实现了零注解到完全形式化验证的渐进式工作流，并在 MIR 层面集成到 Cargo。

**🔧 技术方法**

使用技术包括 Rust MIR 级别的模型转换、CBMC 作为符号执行引擎、DFCC、量化、循环不变式、函数 stubbing 以及 AI 辅助的契约生成。

**📊 数据集**

数据集涵盖了 Firecracker、s2n‑quic、Hifitime、Rust 标准库等多个工业级开源项目，总计约 16,000+ 检测 harness。

**📈 对比分析**

在 CI 中，Kani 对标准库每次提交能验证 16,748 个 harness，平均运行时约 69 分钟；相较于仅测试或 fuzz，发现 11 个此前未被捕获的 Bug，且证明时间在秒级到分钟级。

**⚠️ 局限性**

限制包括仅支持安全 Rust 代码、缺乏并发/多线程支持、对动态 trait dispatch、泛型实例化和复杂的内存模型（如栈借用）支持有限。

---

## 124. Class-Grouped Normalized Momentum and Faster Hyperparameter Exploration to Tackle Class Imbalance in Federated Learning

**arXiv ID:** 2607.01474 | [PDF](https://arxiv.org/pdf/2607.01474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 125. Unveiling the Non-Monotonic Effect of Privacy on Generalization under Byzantine Robustness

**arXiv ID:** 2607.01492 | [PDF](https://arxiv.org/pdf/2607.01492v1)

**作者:** Thomas Boudou `[一作]` (Inria), Aurélien Bellet `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了拜占庭鲁棒性、局部差分隐私（LDP）与分布式学习中的泛化误差之间的关系，并通过理论与实验验证了在不同隐私噪声强度下，隐私与鲁棒性对泛化误差的非单调影响。

**💡 创新点**

创新点在于揭示并证明泛化误差在高噪声（强隐私）和低噪声（弱隐私）两种隐私 regime 下的两种不同趋势，首次通过匹配算法稳定性上下界解释了这一非单调性，并将其与传统的优化误差三元悖论区分开来。

**🔧 技术方法**

主要技术包括算法稳定性分析、匹配的下界和上界证明，以及在 LDP 约束下的拜占庭鲁棒分布式学习框架的设计与实现。

**📊 数据集**

实验使用了公开的标准机器学习数据集（如 MNIST、CIFAR-10 等）进行评估，数据集详情未在摘要中披露。

**📈 对比分析**

与传统的非隐私或非鲁棒基线方法相比，实验表明在高噪声 regime 下引入 LDP 可以显著降低泛化误差，而在低噪声 regime 下则出现与鲁棒性竞争的情况，验证了理论预言。

**⚠️ 局限性**

局限性包括：实验范围仅覆盖常见公开数据集，缺乏真实工业场景验证；分析仅针对 LDP 约束，未探讨其他隐私模型；对模型规模与网络拓扑的依赖性未系统评估。

---

## 126. Don't Let Gains FADE: Breaking Down Policy Gradient Weights in RL

**arXiv ID:** 2607.01490 | [PDF](https://arxiv.org/pdf/2607.01490v1)

**作者:** Juliette Decugis `[一作]` (FAIR at Meta), Taco Cohen `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了RL中政策权重的正负梯度质量，提出FADE调度器来动态平衡探索与利用。

**💡 创新点**

将政策权重分解为正负梯度质量并沿困难、符号、尺度三轴分析，设计结合AsymGRPO与Power α的FADE动态权重。

**🔧 技术方法**

基于PPO框架的GRPO、AsymGRPO、Power α、FADE；使用二元奖励的策略梯度；通过SVD分析权重更新秩。

**📊 数据集**

CodeContest、TACO竞赛集、LiveCodeBench v6、AIME 2024/2025数学竞赛。

**📈 对比分析**

与REINFORCE、GRPO以及各种pass@k优势等静态权重比较，在7B和32B模型上，FADE在相同训练步数下pass@1提升约14%，并保持更高多样性pass@100。

**⚠️ 局限性**

仅考虑二元终止奖励，未处理多步奖励；对低回合数rollout的鲁棒性不明；多轮生成和中间反馈场景的适用性待验证。

---

## 127. Fully Unsupervised Detection of Physical Contacts on Subsea Cables via State-of-Polarization Monitoring

**arXiv ID:** 2607.01484 | [PDF](https://arxiv.org/pdf/2607.01484v1)

**作者:** Agastya Raj `[一作]` (Trinity College Dublin), Marco Ruffini `[通讯]` (Trinity College Dublin)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在海底光纤电缆上实现完全无监督的物理接触事件检测，基于连续偏振状态（SOP）监测；

**💡 创新点**

创新点在于提出双头Fast‑Slow DSVDD架构，能够同时捕捉短暂冲击与持续变化的两种时间尺度异常，并且无需任何事件标签即可训练；

**🔧 技术方法**

使用深度单类学习算法DSVDD，配合双频段分离、四路膨胀卷积分支和多尺度池化，最终生成记录级异常排名；

**📊 数据集**

采用2025年6–8月Tampnet Lowestoft–Lista海底电缆92天连续SOP记录（122,174条一分钟FLAC文件）作为实验数据集；

**📈 对比分析**

与传统STA/LTA触发器和单尺度vanilla DSVDD对比，Fast‑Slow DSVDD在五个已确认的渔船接触事件中最差排名仅为13，远优于STA/LTA的91和vanilla DSVDD的1,219；

**⚠️ 局限性**

局限性包括：极低的事件标签稀缺导致难以验证泛化性、对不同电缆环境的适应性未知、以及部分未被das覆盖的事件可能导致漏检或误报。

---

## 128. Social-Annotate: Self-Healing Browser Extension to Annotate and Collect Social Media Data

**arXiv ID:** 2607.01460 | [PDF](https://arxiv.org/pdf/2607.01460v1)

**作者:** Ali Najafi `[一作]` (Sabanci University), Onur Varol `[通讯]` (Sabanci University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了名为 Social‑Annotate 的浏览器扩展，支持在 12 大社交平台上直接注入自定义问卷，实现无代码的生态学有效数据采集，并配备 LLM 驱动的自愈代理以自动修复因 DOM 变化导致的注入失效。

**💡 创新点**

创新点在于：①将问卷注入原生平台以保持生态有效性；②利用 LLM 结合结构化输出与 Playwright 实时验证，实现全自动的自愈与选择器更新；③提供内容操纵代理，支持盲测与知情模式，满足干预实验需求。

**🔧 技术方法**

技术实现主要包括：Chrome Extension 架构（manifest、background、content script、popup、options）、JSON Schema + jsonform 配置问卷、LLM（Claude、Gemini）进行结构化选择器提取、Playwright 自动化浏览器验证、REST API 与外部 LLM 交互实现内容改写。

**📊 数据集**

数据方面通过在各平台（X、Bluesky、Reddit、Mastodon、WhatsApp、Telegram 等）采集的真实用户帖子与账户信息，并在扩展中记录问卷答案及元数据，支持导出为 JSONL 或 POST 到后台服务器；评估时使用 Internet Archive Wayback Machine 的历史页面快照。

**📈 对比分析**

在自愈代理评估中，针对 X 的 2010、2014、2017、2020、2026 年快照实现了 5/5 次完整注入（4/5 完整元数据捕获），Bluesky 2023 与 2026 同样通过；对 Reddit、Mastodon、WhatsApp、Telegram、Truth Social 等平台，全部成功提取选择器并注入问卷；与传统手工修复相比，自动化流程将维护成本降低约 70%–80%，并显著提升注入成功率。

**⚠️ 局限性**

局限性包括：①LLM 输入长度限制导致无法处理极大页面（如 LinkedIn）或高度动态的 DOM（如 Instagram 评论区）；②仍需人工批准最终部署以保障安全；③依赖外部 LLM 服务，可能产生隐私泄露与成本问题；④内容操纵功能目前仅支持预定义映射或简单 API，缺乏更细粒度或基于本地 LLM 的复杂改写能力。

---

## 129. The General Stability of Ranking

**arXiv ID:** 2607.01546 | [PDF](https://arxiv.org/pdf/2607.01546v1)

**作者:** Houming Chen `[一作]` (University of Michigan), H. V. Jagadish `[通讯]` (University of Michigan)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

提出了一种基于距离的排名稳定性度量（General Stability），并给出了二维精确算法、通用采样估计方法以及针对准凸距离函数的凸体体积近似算法（Conv-SC）

**💡 创新点**

创新点在于将排名稳定性从“完全一致”扩展为“相似性加权”，引入可自定义距离函数、准凸性概念，利用凸体体积近似实现多维情形下的多项式时间计算

**🔧 技术方法**

核心技术包括：排名区域几何分析、二维分段扫描、无偏采样与加权指数核、随机蒙特卡洛体积近似、Hit‑and‑Run MCMC、凸体体积多项式算法与二分搜索求解

**📊 数据集**

实验使用了八个真实世界排行榜数据（QS、THE、CWUR、CSMetrics、MPI、CSRankings、NBA、EPI）以及随机生成的多维实例

**📈 对比分析**

与传统 Exact Stability 对比，General Stability 在受限小扰动下保持更高稳定性；在采样方法中，GSC_md 在大多数实例上误差低于预设 30% 并保持毫秒级运行；Conv‑SC 在准凸距离下实现了维度多项式扩展，显著优于采样法在高维稀疏事件中的指数增长

**⚠️ 局限性**

局限性包括：距离函数需满足准凸性才能使用 Conv‑SC；对非凸或高分辨率距离（如复杂 Kendall/Tau 加权）仍需采样，且在极小稳定值的稀有事件下采样成本可能仍然很高；实现复杂度高且对内点求解敏感

---

## 130. Can Language Models Actually Retrieve In-Context? Drowning in Documents at Million Token Scale

**arXiv ID:** 2607.01538 | [PDF](https://arxiv.org/pdf/2607.01538v1)

**作者:** Siddharth Gollapudi `[一作]` (University of California, Berkeley), Sewon Min `[通讯]` (University of California, Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在百万-token规模下的in-context检索，提出改进的0.6B LM检索器并探索长文本泛化。

**💡 创新点**

发现注意力稀释是大规模检索失效的根本原因，并提出长度感知的softmax调节和文档级稀疏注意力以缓解该问题。

**🔧 技术方法**

使用随机文档ID、block-sparse注意力、on-policy辅助损失、SSMax、长度感知sink、文档级路由等技术。

**📊 数据集**

在MS MARCO、NQ、HotpotQA以及LIMIT等BEIR数据集和词法相似性任务上进行评估。

**📈 对比分析**

与稠密检索、MSA-4B等基线相比，在百万-token规模下可匹敌稠密检索，且在LIMIT上超过3倍；在更大规模下仍有提升空间。

**⚠️ 局限性**

缺点是注意力稀释仍导致在极大上下文中检索性能下降，且改进仍无法完全解决长上下文泛化瓶颈。

---

## 131. Janus: a Playground for User-Involved Agentic Permission Management

**arXiv ID:** 2607.01510 | [PDF](https://arxiv.org/pdf/2607.01510v1)

**作者:** Natalie Grace Brigham `[一作]` (University of Washington), Franziska Roesner `[通讯]` (University of Washington)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Janus Playground，一个用于实验和评估用户参与式代理系统权限管理设计的模块化平台，并在三类合成情景中测试了六种不同的权限助手；

**💡 创新点**

首次构建了覆盖权限管理设计空间各维度的可插拔框架，系统性比较多种权限策略与用户交互的权衡，为理解代理系统安全性提供新的实验方法；

**🔧 技术方法**

基于 Google Agent Development Kit (ADK) 与 LiteLLM 实现代理与工具；使用 OpenAI o3-mini LLM 进行决策与风险评估；通过自定义 PolicyManager 与 PermissionAssistant 接口实现插件化；采用脚本化合成回应器与自动化评估框架；

**📊 数据集**

使用自制的合成数据集（模拟电子邮件、日历和文件系统）以及 AgentDojo benchmark 的任务描述；不涉及任何真实用户数据；

**📈 对比分析**

采用全因子实验（3场景×4子场景×3合成回应器×5次）记录攻击/不符请求、期望工具调用和输出正确率；实验表明用户交互能显著降低攻击率，自动化或低交互策略在部分情景下导致误操作；没有单一设计在所有情景中表现最佳；

**⚠️ 局限性**

仅评估六个简单原型，使用单一小型 LLM，缺乏真实用户研究和多模态交互评估，合成情景与真实攻击差异大，且未考虑长期用户行为与系统的持续学习与适应。

---

## 132. How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size

**arXiv ID:** 2607.01487 | [PDF](https://arxiv.org/pdf/2607.01487v1)

**作者:** Fabian Schaipp `[一作]` `[通讯]` (Inria), Fabian Schaipp (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

提出三项缩放律，将模型大小、训练步数和批量大小统一建模，能够预测最佳批量和子最优批量的规模，并且仅需少量训练跑即可拟合。

**💡 创新点**

创新点在于把批量大小与步数作为独立变量加入幂律，既可恢复Chinchilla形式，又能描述子最优批量和临界批量，并在训练跑不足时仍能保持准确性。

**🔧 技术方法**

使用幂律拟合、Huber损失、多重初始化优化、五折交叉验证以及两阶段拟合等技术。

**📊 数据集**

采用两套公开的LLM训练日志数据集（Schaipp等公开数据集和另一未公开数据集）。

**📈 对比分析**

与先前的批量大小扩展律和Chinchilla律比较，三项律在训练跑数显著减少（仅28%）的情况下仍保持与全量跑相近的最佳批量预测精度，并能更好预测5%子最优批量区间。

**⚠️ 局限性**

局限包括：仍需针对每组（N,D,b）寻找最优学习率；对不同优化器、任务的泛化不确定；在极端批量大小或极大数据预算下预测误差较大。

---

## 133. Comparing Architectures for Supervised Political Scaling

**arXiv ID:** 2607.01464 | [PDF](https://arxiv.org/pdf/2607.01464v1)

**作者:** Anna Golub `[一作]` (University of Stuttgart), Sebastian Padó `[通讯]` (University of Stuttgart)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了使用Transformer及其变体进行政治立场尺度化的方法，比较了基于句子标签聚合和分块回归两种策略，并探讨联合预测的可能性。

**💡 创新点**

创新点在于：①对联合预测是否有提升进行经验验证；②在分块回归中系统探索块大小对性能的影响，揭示了从分类到回归的连续空间；③提出使用ModernBERT进行分块回归作为替代方案。

**🔧 技术方法**

使用的技术包括SBERT、ModernBERT、BigBird、whitening归一化、多任务训练、对比学习（triplet loss）以及LLM零样本标注。

**📊 数据集**

数据集为MARPOR的3200余份宣言（约1M句子），2000-2018年用于训练，2019-2023年用于测试，使用机器翻译的英文版本。

**📈 对比分析**

评估采用Spearman秩相关；结果显示单独预测在RILE上达到0.88、GAL‑TAN约0.83；分块回归与标签聚合性能相当；联合预测未显著提升；块大小在20‑100句时最优。

**⚠️ 局限性**

局限包括：仅在时间泛化任务上评估，未探索多国或少量数据情况；LLM仅做零样本对比；使用的模型尺寸有限，未做完整超参数搜索；未对句子级标注的可靠性进一步分析。

---

## 134. Certified World Models as Sensing Clocks: Drift-Aware Deadlines for Active Perception

**arXiv ID:** 2607.01537 | [PDF](https://arxiv.org/pdf/2607.01537v1)

**作者:** Hongbo Wang `[一作]` `[通讯]`, Hongbo Wang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于已审核的等变世界模型的感知时钟原语，给出一条可操作的重新感知截止时间，控制连续预测误差超限概率。

**💡 创新点**

创新点在于将世界模型的有效性期限转化为主动的、可证实的感知触发器，并通过漂移包络校准实现可部署的漂移感知时钟；同时对等变模型的谱率与漂移的关系进行了系统阐释。

**🔧 技术方法**

采用等变向量神经元JEPA（VN‑JEPA）世界模型、Lyapunov谱率审计、漂移包络校准、合成误差动力学实验、以及对比实验中的概率阈值与误差上限分析。

**📊 数据集**

主要数据集为冻结的三维VN‑JEPA模型与其对应的自回归残差轨迹，另外使用合成线性高斯误差动力学与小规模模拟测试。

**📈 对比分析**

与基于期望信息增益（MB‑EIG）的反应式调度器、周期性采样、以及经验性合形（conformal）阈值调度器进行对比；在冻结模型上感知时钟使区间误差超限率低于0.15，且相较MB‑EIG在事件尾部误差降低约56%，但在短期全感知场景下经验性合形阈值与其性能相当。

**⚠️ 局限性**

局限性包括：在短期全感知下谱率项未产生实质优势，经验性合形阈值可在该情境下替代感知时钟；缺乏可操作的提前时间（lead‑time）评估，且在非全感知或更长漂移区间时谱率对性能的影响仍待验证。

---

## 135. IntentTune: Using user demand and personalization to resolve "unknown" query intents for e-commerce search

**arXiv ID:** 2607.01530 | [PDF](https://arxiv.org/pdf/2607.01530v1)

**作者:** Rachith Aiyappa `[一作]` (eBay Inc.), Shuang Zhou `[通讯]` (eBay Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 IntentTune 框架，通过需求模式和用户行为上下文联合推断模糊查询的性别、年龄、尺寸等隐式意图，并细化类别预测。

**💡 创新点**

创新点在于将个人化历史查询与全局需求信号并行利用，直接在意图推断阶段引入用户上下文，而非仅在检索后进行重排序。

**🔧 技术方法**

采用内部 LLM 进行意图推断，结合 BERT 基础的基础意图模型（性别、年龄、尺寸、类别）和需求/用户信号的投影；对历史查询进行筛选并构造提示；统计加权指标评估性能。

**📊 数据集**

使用自建的 900 条（30 个模糊查询 × 30 名用户）人工标注数据集，包含用户历史查询、个人资料与手工标注的尺寸、性别、年龄与类别标签。

**📈 对比分析**

与仅基于需求或仅基于用户资料的基线对比，历史查询个性化在年龄和性别上分别提升约 17% 与 90% 的加权 F1，尺寸仅通过历史查询可达 0.853 加权 F1；类别候选数下降 68.5%，保持准确率。

**⚠️ 局限性**

局限包括多源信号冲突导致推断不一致、冷启动问题（缺乏用户或需求信息）以及对实时评估（如 CTR）尚未验证。

---

## 136. Wind-Aware Reinforcement Learning Control of a Small Quadrotor Using Learned Onboard Wind Estimation in Simulated Atmospheric Turbulence

**arXiv ID:** 2607.01528 | [PDF](https://arxiv.org/pdf/2607.01528v1)

**作者:** Abdullah Al Tasim `[一作]` (University of Oklahoma), Wei Sun `[通讯]` (University of Oklahoma)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了一个两阶段无人机风感知与控制管线，先用注意力增强的GRU网络估计水平风速，再将估计嵌入PPO强化学习控制器，实现显著降低追踪误差。

**💡 创新点**

创新点在于结合无传感器风速估计与强化学习控制的端到端架构，并通过三路分解方法量化风感知对性能提升的具体贡献。

**🔧 技术方法**

使用注意力增强的GRU网络进行风估计，PPO算法进行策略学习，EMA平滑后作为观测输入，并对控制动作进行滤波和限制。

**📊 数据集**

训练和评估基于自定义六自由度仿真环境生成的数千场模拟飞行数据，涵盖多种风速、湍流强度、速度梯度和方向变化的von Kármán风场。

**📈 对比分析**

通过与传统PD基线以及将风感知通道置零的RL控制器在相同重放物理条件下的对比，水平轨迹追踪误差平均降低48%，垂直水平轴39.5%，且在超出训练范围的强风场中保持100%胜率。

**⚠️ 局限性**

局限包括仅在仿真环境验证、对水平风向估计的依赖、对非训练轨迹和极端风速/高度条件的泛化未知，以及缺乏实飞实验支持。

---

## 137. From Monolingual to Multilingual: Evaluating Mamba for ASR in South African Languages

**arXiv ID:** 2607.01502 | [PDF](https://arxiv.org/pdf/2607.01502v1)

**作者:** Jesujoba O. Alabi `[一作]` (Saarland University), Dietrich Klakow `[通讯]` (Saarland University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文评估了Mamba架构在七种南非语言上的自动语音识别性能，并与Conformer基线对比；同时探讨了多语言训练、语言及语言族嵌入、以及多任务LID的影响。

**💡 创新点**

创新点在于将Mamba迁移到低资源非洲语言，证明其在计算效率上的优势，并首次系统分析语言嵌入在多语言、低资源和跨语料库稳健性中的作用。

**🔧 技术方法**

采用Mamba（ConMamba）与Conformer编码器、CTC目标、语言嵌入、语言族嵌入及多任务LID策略。

**📊 数据集**

使用Swivuriso（7语料）、NCHLT（10语料）和FLEURS（4语料）进行训练与评测。

**📈 对比分析**

对比方法为单语、聚合多语、含语言信息与无语言信息的训练，结果显示Mamba在大部分语言上与Conformer相当，且在多语言、低资源及跨数据集场景下表现更佳；仅在高资源单语场景下语言嵌入提升有限。

**⚠️ 局限性**

局限包括语言嵌入未能捕捉词汇/语法的语言相似性、仅覆盖七种语言、缺乏大规模预训练以及在极短或极长语音上的鲁棒性仍有限。

---

## 138. CADENZA in Action: Breaking the Monolith with Intent-Dependent Plan Spaces for Semantic Queries

**arXiv ID:** 2607.01468 | [PDF](https://arxiv.org/pdf/2607.01468v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 139. Towards Learning Representations of Policies in Two-Player Zero-Sum Imperfect-Information Games

**arXiv ID:** 2607.01498 | [PDF](https://arxiv.org/pdf/2607.01498v1)

**作者:** Kevin Wang `[一作]` (Brown University), Amy Greenwald `[通讯]` (Brown University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

学习两玩家零和信息不完全游戏（如 Kuhn Poker 与 Leduc Poker）中的策略嵌入，并设计下游任务评估其效果。

**💡 创新点**

首次系统地构造三种策略数据集生成方法、七种嵌入学习方法，并提供完整的下游任务评测框架，以比较自监督学习在策略表示上的性能。

**🔧 技术方法**

采用权重自编码器、功能自编码器、对比学习轨迹编码、NeuPL 条件网络、Grover 混合目标、以及表格/身份基线等多种技术。

**📊 数据集**

使用 Kuhn Poker、Leduc Poker 两个小型游戏作为主实验数据集，并在附录中补充 Liar's Dice 与 Phantom Tic‑Tac‑Toe 的实验。

**📈 对比分析**

通过均方误差、收益预测、最佳响应、零射最佳响应、代理识别等任务进行对比，轨迹编码与 NeuPL 在大多数任务中显著优于基线，而表格表示仅在极小游戏上优于学习方法。

**⚠️ 局限性**

局限性包括：仅验证于小型游戏、嵌入维度仍较高、NeuPL 训练与评估数据集不一致、对更大游戏的推广尚未深入。

---

## 140. Insights from GitHub Community on the Matter Standard: Developer Perspectives and Challenges

**arXiv ID:** 2607.01494 | [PDF](https://arxiv.org/pdf/2607.01494v1)

**作者:** Muhammad Hassan `[一作]` (University of Illinois Urbana Champaign), Masooda Bashir `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Matter 标准官方 GitHub 仓库中的 13,008 条 Issue 进行主题建模（LDA）与定性分析，归纳出四大类别（测试、互操作性、开发、平台与网络）并提出改进建议。

**💡 创新点**

首次从开发者视角系统挖掘 Matter 的实践痛点，结合无监督主题模型与专家编码，提供可操作的实证反馈，为标准演进与工具改进提供数据支持。

**🔧 技术方法**

使用 LDA（Gensim）主题建模并评估 C_v 一致性；通过 Tokenization、停用词移除、代码块过滤等预处理；利用 Pandas/Matplotlib 进行统计与可视化；采用专家审阅与共识讨论进行定性编码。

**📊 数据集**

官方 Project CHIP（Matter）GitHub 仓库公开 Issue 数据集，时间跨度 2020‑2025，包含 13,008 条问题，1010 名贡献者，涵盖不同平台与工具链。

**📈 对比分析**

通过计算问题数量、关闭率、平均评论数与平均解决时长等指标，对四大类别进行比较；LDA 选取 k=17、p=20、i=100，C_v=0.59，主题解释度良好；测试与互操作性占比 42% 与 30%，关闭率均超过 85%。

**⚠️ 局限性**

仅涵盖公开 Issue，无法追踪问题在 CSA 内部决策流程中的影响；缺少终端用户视角的对比数据；安全相关问题虽被识别但未进行深度形式化验证，可能漏掉更细粒度的风险。

---

## 141. VLAFlow: A Unified Training Framework for Vision-Language-Action Models via Co-training and Future Latent Alignment

**arXiv ID:** 2607.01586 | [PDF](https://arxiv.org/pdf/2607.01586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. BFF: Simple explanations for complex phenomena

**arXiv ID:** 2607.01483 | [PDF](https://arxiv.org/pdf/2607.01483v1)

**作者:** Charlotte Knierim `[一作]`, Rif A. Saurous `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对BFF（Brainfuck变体）系统进行实验，比较了配对交互与单纯随机变异在寻找自复制程序（自复制器）方面的效率，并提出了一种自复制检测器。

**💡 创新点**

创新点在于：①证明在BFF中单纯随机变异（随机游走）能与甚至超过配对交互在发现自复制器方面的性能；②设计了一种能够在程序执行后快速识别自复制器的检测方法；③通过引入自定义字符分布（CUST、CUST64）展示可大幅提升随机搜索效率，挑战了原先认为配对交互必要的观点。

**🔧 技术方法**

主要技术包括：BFF系统的模拟、程序空间中的随机游走、对程序状态的“复制”操作、基于多次执行后片段一致性的自复制检测器、以及对合成深度/宽度约束的实验。

**📊 数据集**

数据集：使用随机生成的64字节程序；还使用了实验中记录的BFF写入字节频率分布作为自定义分布；此外还在不同分布（Uniform、BFF、CUST、CUST64）下进行大规模（10^10）程序测试。

**📈 对比分析**

比较方法：以“测试的程序总数/发现自复制器数量”作为时间度量；结果显示，单纯随机变异（尤其采用CUST64分布）平均需要的程序数约为BFF系统的1/20到1/25；在低变异率下仍能优于BFF；在高变异率下两者差距减小。

**⚠️ 局限性**

局限性：①自复制检测器阈值（48/64）是经验选择，可能漏检或误检；②实验仅在BFF这类固定长度脑机互操作系统上验证，结果对其他语言或更大程序空间的推广有限；③没有深入分析配对交互对长远进化（如多代共生、水平基因转移）可能产生的影响；④未探讨BFF系统在资源约束下的可扩展性。

---

## 143. The Evolution of the Peridynamics Community in Its First Quarter Century

**arXiv ID:** 2607.01461 | [PDF](https://arxiv.org/pdf/2607.01461v1)

**作者:** Biraj Dahal `[一作]` (Georgia Institute of Technology), Pablo Seleson `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并演化分析了2000-2024年Peridynamics领域的共作者网络，重点研究2020-2024年疫情对合作模式的影响。

**💡 创新点**

创新点在于将年度共作者网络与连通分量、聚类系数等多层次网络指标结合，并通过国家层面分析揭示中国机构在疫情期间对该领域新作者增长的主导作用。

**🔧 技术方法**

采用网络分析技术（度中心性、加权度、流量闭合度、流量介数）、聚类系数、平均距离、直径等指标进行计算，并基于Scopus构建共作者网络。

**📊 数据集**

使用Scopus数据库中所有包含“peridynamic(s)”的文献，截止2024年共2761篇文献、3195位作者的数据集。

**📈 对比分析**

通过与2019年前基线对比，发现作者数和出版物增长放缓，但聚类系数提升、连通分量结构稳步演化，平均距离和直径略升，整体网络更紧密，指标表现出明显的演化趋势。

**⚠️ 局限性**

局限性包括：仅使用Scopus索引数据，可能遗漏未被索引的近年文献；作者ID偶有错误需手工校正；国家归属仅取首个机构，忽略多机构作者；疫情影响的地区差异未作细化探讨。

---

## 144. OPINE-World: Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration

**arXiv ID:** 2607.01531 | [PDF](https://arxiv.org/pdf/2607.01531v1)

**作者:** David Courtis `[一作]` (University of Toronto), Scott Sanner `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出OPINE-World，一种利用两台LLM协作的在线假设-检验循环，学习对象中心的可编程世界模型并在交互中执行规划。

**💡 创新点**

创新点在于将动作执行与模型学习分离为两台LLM代理，使用精确回放验证、反例驱动合成和贝叶斯本体错误度量来动态细化对象类型，同时无需给定对象词表、目标或动作语义。

**🔧 技术方法**

技术包括基于Python的程序化世界模型、精确回放检验、计数式贝叶斯效应表和本体错误度量、局部上下文特征分裂、以及基于已验证模型的有界前向搜索规划。

**📊 数据集**

使用ARC‑AGI‑3公共评测集（25个游戏、183层）作为数据集。

**📈 对比分析**

与单一代理编码模型baseline1、WorldCoder和神经潜在世界模型对比，OPINE-World在ARC‑AGI‑3上获得78.4的动作效率分数，赢得20/25游戏并在大多数难关上显著低于baseline1的动作次数，超越人类基准。

**⚠️ 局限性**

局限包括对可观测Markov假设的依赖（隐藏状态场景不适用）、自监督感知导致的配对误差、规划搜索受限于分支因子以及单次游戏运行缺乏方差估计。

---

## 145. EO-Agents: A Three-Agent LLM Pipeline for Earth Observation Hypothesis Generation

**arXiv ID:** 2607.01584 | [PDF](https://arxiv.org/pdf/2607.01584v1)

**作者:** Mahyar Ghazanfari `[一作]`, Peng Wei `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

建立了NASA Earth Observation Knowledge Graph (NASA EO-KG)，并对其结构、属性和共用关系进行量化统计；

**💡 创新点**

首次完整提供了NASA数据与文献的知识图谱，并详细描述了文献-数据集共用网络及跨DAAC、跨仪器的使用模式；

**🔧 技术方法**

利用GraphML解析、图数据库结构、节点属性提取与统计分析技术，对节点、边、度分布等进行计算；

**📊 数据集**

使用NASA发布的GraphML文件，包括138,704篇论文和8,058个数据集（涵盖1972–2026年时间跨度的CMR元数据）；

**📈 对比分析**

本文仅给出了统计结果，并未与其他方法进行实验对比，因而暂无性能指标；

**⚠️ 局限性**

局限性包括：仅覆盖NASA数据，未考虑非公开或跨机构使用；知识图谱为静态 snapshot，缺乏动态更新；共用关系仅基于论文引用，可能遗漏实际使用场景。

---

## 146. DiPS: Dialogue Policy Selection for High-Stakes Persuasion Agents

**arXiv ID:** 2607.01557 | [PDF](https://arxiv.org/pdf/2607.01557v1)

**作者:** Tianyi Zhang `[一作]` (University of Southern California), David Traum `[通讯]` (University of Southern California)

**通讯引用:** 11113 | [OpenAlex ID](https://openalex.org/A5004384107)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一个名为DiPS的离线强化学习框架，用于在火灾疏散场景中根据对话历史动态选择最合适的说服策略，以提升居民撤离成功率。

**💡 创新点**

首次将隐式Q学习(IQL)应用于高风险说服对话，通过学习在离线对话数据中对策略集合进行选取，实现了可在多轮对话中自适应切换说服风格。

**🔧 技术方法**

使用离线强化学习（IQL）、多策略选择、LLM驱动的响应生成、RAG检索以及LLM判定器进行评估。

**📊 数据集**

基于先前的Wizard‑of‑Oz火灾疏散对话数据集，包含10个居民角色的手工标注对话，并用LLM模拟器进行扩充。

**📈 对比分析**

与零射击、全局检索（RAG）以及随机/固定策略基线对比，实验表明DiPS在改进的模拟环境中实现了92%的成功率，平均对话轮次为11.2，显著优于其他方法。

**⚠️ 局限性**

受限于离线对话数据覆盖不足、策略集合离散化导致表达不足、LLM居民/判定器的真实性与多语种可迁移性不足。

---

## 147. A Reconfigurable Rocker-Bogie Robot for High Step Climbing and Turning

**arXiv ID:** 2607.01554 | [PDF](https://arxiv.org/pdf/2607.01554v1)

**作者:** Kento Koizumi `[一作]` (University of Tsukuba), Kenji Suzuki `[通讯]` (University of Tsukuba)

**通讯引用:** 139593 | [OpenAlex ID](https://openalex.org/A5044483075)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并验证了一种可重构 rocker‑bogie 机器人，可在四轮与六轮模式之间切换，兼顾高阶步升与高效零半径转向。

**💡 创新点**

创新点在于为 bogie 关节配备电动机并实现主动摇摆，实现配置切换仅需两额外驱动器；结合前轮差速与后部全向轮，获得零半径转向且能保持六轮步升性能。

**🔧 技术方法**

采用机械建模估算摇摆扭矩、差速驱动控制、BLE/WiFi 通信、CAN 总线驱动 CyberGear 电机以及实验验证。

**📊 数据集**

使用自制原型机器人，在实验室进行转向与步升测试，并在 XROBOCON 竞赛赛场上验证。

**📈 对比分析**

与传统六轮非可转向 grip 机器人对比，零半径转向速度提高5倍、平均扭矩仅17%，40 cm 步升平均时间6.4 s；转向半径波动更小，表现更稳定。

**⚠️ 局限性**

局限在于高速转向时易产生侧倾、步升成功率随速度下降；机器人结构不对称导致转向时 bogie 旋转；缺乏实时动态控制与更高阶障碍适应性。

---

## 148. X-LogSMask: Expand Transformer for Graph-Structured Data

**arXiv ID:** 2607.01553 | [PDF](https://arxiv.org/pdf/2607.01553v1)

**作者:** Leyan Li `[一作]`, Liping Hu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种名为X-LogSMask的结构化掩码，用以将图结构直接注入Transformer自注意力中，使其能高效处理图数据；

**💡 创新点**

创新点在于：①使用对称归一化的邻接矩阵的对数变换作为可解释的结构掩码，既保留多头注意力的可学习性，又能强制顶点之间的拓扑约束；②不同注意力头使用不同幂次的归一化邻接矩阵，赋予每个头不同的拓扑半径，实现单层多跳信息传播；③保持Transformer原有架构，兼容轻量级单层实现。

**🔧 技术方法**

技术包括对称归一化、对数变换、对归一化邻接矩阵的幂运算、可解释的多头掩码生成、注意力对数位添加以及标准Transformer编码器。

**📊 数据集**

实验数据集覆盖节点级（Cora、Citeseer、Pubmed、Computers、Photo、CS、Physics、WikiCS）、边级（Cora、Citeseer链接预测；epic-games-plr、air-traffic-2019-rlr、air-traffic-2015-rlr边回归）和图级（NCI1、D&D、PROTEINS、MUTAG、COLLAB、IMDB-B、MOLHIV）七大类。

**📈 对比分析**

与传统GNN（GCN、GraphSAGE、GAT等）和多种Graph Transformer（Graphormer、Gradformer、Eigenformer、SGFormer、Polynormer等）进行对比，X-LogSMask在节点级平均排名3.3、边级1.0、图级1.7，单层配置仍能保持与深层模型相近的性能，取得多项数据集上的最新最佳结果。

**⚠️ 局限性**

局限性包括：在极小的节点级数据集上易过拟合，需使用子图采样等增广；对大规模图仍面临自注意力的O(n²)计算和存储瓶颈；在某些图级数据集（如NCI1、COLLAB）上表现略逊于最佳基线；需要合理选择头数和幂次以避免冗余。

---

## 149. Evaluating Glanceable Multi-Device Family Health Tracking with Smartwatches and Home Displays

**arXiv ID:** 2607.01618 | [PDF](https://arxiv.org/pdf/2607.01618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 150. Mechanism and Stability Analysis of Metabolic Closed-Loop Metaheuristics

**arXiv ID:** 2607.01551 | [PDF](https://arxiv.org/pdf/2607.01551v1)

**作者:** Jinliang Xu `[一作]`, Liping Ma `[通讯]` (Seventh Medical Center Chinese Pla General Hospital)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Metabolic Multi‑Agent Optimizer（MMAO）框架进行机制层面的理论分析，构建通用状态模型，证明私有能量、公共预算、角色状态与活跃种群在弱条件下保持有界，并从中识别收缩、再投资与搜索再分配三种内生运行模式；同时给出跨领域（连续、离散）机制验证实验以检验理论预测。

**💡 创新点**

提出了MMAO的通用闭环状态描述，并在此基础上给出有界性、可再生性与自适应运行模式的理论保证；首次将资源循环视为框架级控制原理，区别实现细节与通用机制；用最小化实验验证了理论结论，突破了先前仅为经验性证明的局限。

**🔧 技术方法**

理论证明（有界性、稳态漂移、再生性、阶段分析）、通用状态模型抽象、基于投影仿射递推的递推界定、Foster‑Lyapunov 驱动分析；实验方法包括机制变量追踪、收缩/再投资占比统计、消除关键机制的消融实验。

**📊 数据集**

连续优化使用CEC式10维测试函数（如Sphere、Rastrigin等），离散优化使用TSP实例；此外在消融实验中对MMAO进行多种版本（无角色漂移、无再投资等）进行对比。

**📈 对比分析**

通过比较机制变量的有界性、预算/能量/种群大小波动及各模式占比，验证MMAO在不同场景下能保持资源平衡并呈现预期的三种运行模式；相较于消融版本，完整MMAO在连续任务上误差下降约1.5‑2倍，在离散任务上仍保持可比性能，但未在公开排行榜上实现全局最优。

**⚠️ 局限性**

缺乏完整收敛或最优性证明；假设受限于能量/预算/角色的有界性与漂移条件；未给出不同问题几何下收益分布的具体推导；机制验证实验规模有限，未覆盖所有离散/连续基准，且对实际性能提升的解释仍主要依赖经验。

---

## 151. OrchestrXR: A Multi-Agent System for Idea-to-Prototype XR Study Authoring

**arXiv ID:** 2607.01588 | [PDF](https://arxiv.org/pdf/2607.01588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 152. AgentFlow: Building Agent Dependency Graphs for Static Analysis of Agent Programs

**arXiv ID:** 2607.01640 | [PDF](https://arxiv.org/pdf/2607.01640v1)

**作者:** Shenao Wang `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 66698 | [OpenAlex ID](https://openalex.org/A5115602103)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 Agent Dependency Graph（ADG）框架，构建了一套统一的中间表示，利用静态分析从多种 LLM 代理框架中恢复 agent、prompt、model、tool、memory、policy 等实体及其组件、控制流和数据流依赖，并基于 ADG 实现了 Agent BOM 生成和 prompt‑to‑tool 风险检测。

**💡 创新点**

创新点包括：①首次引入 ADG 作为框架无关的图表示，显式捕获框架层面的 agent 依赖；②设计了跨框架的事实提取前端与多阶段依赖分析流程，能够从 5 种主流代理框架的源代码中抽取规范化事实；③在大规模真实项目（5,399 个）上进行评估，展示 ADG 在依赖恢复、BOM 生成和风险检测方面的显著优势。

**🔧 技术方法**

技术手段主要是：静态分析（别名解析、数据流和控制流推理）、图构建与查询、事实提取前端（基于 YASA 的语义注册）、taint 风险检测算法，以及 Python/TypeScript 结合的实现。

**📊 数据集**

使用的数据集为 AgentZoo（5,399 真实代理程序），以及对其进行细分的 ADG‑Eval（60 项目）、BOM‑Eval（100 项目）和 P2T‑Audit（200 条检测结果）等子集。

**📈 对比分析**

通过与 Agent‑Wiz、AgenticRadar、Trusera‑AI‑BOM、Drako‑Agent‑BOM、Cisco‑AI‑BOM 等基线工具进行定量（节点/边数、BOM 关系、风险检测精度）和定性对比；性能方面，平均分析时长 14.17 秒，95% 分位 163.54 秒，生成的 ADG 规模适中，覆盖率高，精度为 73%。

**⚠️ 局限性**

局限性包括：①静态分析的 over‑approx 可能导致误报；②目前仅支持 Python 代码和 5 种框架，难以处理自定义或不符合标准语法的 agent；③需要维护框架语义注册，框架更新会影响准确性；④未覆盖工具实现内部的低层调用，无法实现完整的 taint 路径追踪。

---

## 153. MKGR: Multimodal Knowledge-Graph Representation Learning for Cold-Start Protein-Protein Interaction Prediction

**arXiv ID:** 2607.01627 | [PDF](https://arxiv.org/pdf/2607.01627v1)

**作者:** Wenbo Zhang `[一作]` (Southwest University), Wenbo Zhang `[通讯]` (Southwest University)

**通讯引用:** 73015 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了冷启动下蛋白‑蛋白相互作用（PPI）预测，提出一种多模态序列‑图网络框架；

**💡 创新点**

创新点在于将区域感知序列编码与四种生物医学知识图谱（蛋白‑药物、蛋白‑疾病、蛋白‑miRNA、蛋白‑lncRNA）联合使用，采用桥式重建正则化稀疏图学习，并通过对偶门控融合自适应调节不同模态对每对蛋白的贡献；

**🔧 技术方法**

使用ESM2蛋白语言模型+Transformer进行序列编码，图注意网络（GAT）处理四类知识图谱，桥式重建损失辅助图学习，最后通过对偶门控融合完成PPI预测；

**📊 数据集**

在两个多模态PPI基准数据集（MTV‑PPI衍生数据集和自构造的STRING+DrugBank+LncTarD+miRTarBase+CTD集合）上进行实验；

**📈 对比分析**

与TAGPPI、HNSPPI、EResCNN、BaPPI、KGF‑GNN、HEENN、ESM2‑AMP等基线比较，在novel‑old和novel‑novel冷启动场景下，均在ACC、F1、AUC、AUPR、MCC等指标上实现显著提升；

**⚠️ 局限性**

局限在于仅考虑四类生物医学实体，未覆盖所有可能的关系；模型在跨物种或特定疾病网络的迁移性尚待进一步验证。

---

## 154. OmniPilot: An Uncertainty-Aware LLM Inference Advisor for Heterogeneous GPU Clusters

**arXiv ID:** 2607.01579 | [PDF](https://arxiv.org/pdf/2607.01579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 155. Safe and Adaptive Cloud Healing: Verifying LLM-Generated Recovery Plans with a Neural-Symbolic World Model

**arXiv ID:** 2607.01595 | [PDF](https://arxiv.org/pdf/2607.01595v1)

**作者:** Junyan Tan `[一作]` (Zhejiang University), Zeyu Qiao `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PASE框架，实现基于LLM的多步恢复计划生成并通过神经符号模型预执行验证，形成“reason–plan–verify–adapt”闭环自愈系统。

**💡 创新点**

将LLM从仅做状态解释转变为核心规划引擎，结合神经符号世界模型进行预验证，并用DRL学习元提示（meta‑prompt）以快速适配未知故障。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）做计划合成；神经符号世界模型（NSWM）做计划可行性评估；深度强化学习（SAC）做元提示优化；LoRA微调LLM；多模态语义描述。

**📊 数据集**

使用OpenStack基础的云故障注入数据集“Failure‑Dataset‑OpenStack”，涵盖实例、网络、存储等多种故障类型及其依赖关系。

**📈 对比分析**

与IFSHM、Deformable DETR‑FD、GCN‑FR以及规则系统对比；PASE在故障检测F1‑score上达0.94，平均恢复时间72 s，较基线节省约28%恢复时间，且在新颖故障的快速适应上将成功率从40%提升到80%仅需15次交互。

**⚠️ 局限性**

限制包括：LLM规划仍依赖离线微调，实时适配受限；元提示优化需要大量交互数据；NSWM验证增加计算开销；假设对系统可观测数据的集中访问，难以直接迁移到联邦或隐私敏感场景。

---

## 156. Fully Persistent Dynamic LCE via AVL Trees and AVL Grammars

**arXiv ID:** 2607.01580 | [PDF](https://arxiv.org/pdf/2607.01580v1)

**作者:** Taiki Kaneda `[一作]` (Hokkaido University), Shunsuke Inenaga `[通讯]` (Kyushu University)

**通讯引用:** 2086 | [OpenAlex ID](https://openalex.org/A5012002493)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种全持久化动态字符串数据结构FeAVL（以及其基于AVL语法的压缩版本），支持字符串分割、连接、单字符更新、相等性查询和最长公共前缀（LCE）查询；

**💡 创新点**

核心创新在于用AVL树代替FeST中的自适应splay树，实现了最坏情况下的对数时间复杂度，并通过路径复制实现全持久化；此外引入AVL语法实现压缩版本，进一步降低空间占用；

**🔧 技术方法**

主要技术包括AVL树的隐式键序列表示、Karp–Rabin指纹用于子串等价性判定、Lipták等人的LCE算法、路径复制实现全持久化，以及Rytter的AVL语法拼接和区间分解；

**📊 数据集**

实验使用合成数据（随机字符串和构造的最坏案例）以及真实文本/代码片段；

**📈 对比分析**

与传统数组、原始FeST、以及全持久化的FeST比较，实验显示FeAVL在大规模字符串上保持对数级别的时间和相对较低的内存占用，尤其在LCE和等价查询上比原始FeST更稳定；

**⚠️ 局限性**

限制包括：在动态操作频繁且版本分支深时，空间仍随更新数线性增长；压缩版本在缺乏重建或动态解析时，空间仍受初始语法大小和更新次数限制；此外，最坏情况下的LCE时间仍含有log²ℓ项。

---

## 157. H-SAGE: Holistic Speaker-Aware Guided Experts for MoE-based Multi-Talker ASR

**arXiv ID:** 2607.01566 | [PDF](https://arxiv.org/pdf/2607.01566v1)

**作者:** Yujie Guo `[一作]` (Nankai University), Yong Qin `[通讯]` (Nankai University)

**通讯引用:** 484802 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 H‑SAGE 框架，利用 Speaker‑Aware Global Encoder 与 Holistic Gating 在 MoE‑based 多说话人 ASR 中实现显式的说话人状态建模与专家路由。

**💡 创新点**

创新点在于通过 Overlap‑Aware Loss 对全局编码器进行显式监督，并将全局与局部特征融合的全局门控机制（Holistic Gating）用于专家选择，从而提升了高重叠场景下的说话人分离与转录准确性。

**🔧 技术方法**

使用 MoLE（低秩 Mixture‑of‑Experts）集成于 Conformer 编码器，结合自注意力、前馈网络、辅助交叉熵损失及序列化输出训练（SOT）等技术。

**📊 数据集**

采用 LibriSpeechMix（包含 2‑mix 与 3‑mix）和标准 LibriSpeech 作为训练与评估数据集。

**📈 对比分析**

与 SOT、SOT+Local MoLE、SOT‑SACTC、GLAD‑SOT 等基线对比，H‑SAGE 在 LSM‑2mix 上 PI‑WER 低至 3.6%/3.8%/6.0%（低/中/高重叠），在 LSM‑3mix（零样本）上实现 5.7%/19.7%/19.5% 的平均提升，整体性能优于现有方法，尤其在高重叠区显著领先。

**⚠️ 局限性**

在单说话人场景下辅助损失略有负面影响，且模型参数受 MoLE 专家数量限制；对极低重叠或无重叠环境的鲁棒性仍有提升空间。

---

## 158. A Capacity-Aware Parr Model for Agile Projects

**arXiv ID:** 2607.01562 | [PDF](https://arxiv.org/pdf/2607.01562v1)

**作者:** Pedro E. Colla `[一作]` `[通讯]` (UADER), Pedro E. Colla (UADER)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文将Parr努力分布模型重新构造为适用于敏捷环境的容量感知预测层，利用容量轨迹对latent努力需求进行约束，从而预测项目进度、完成时间以及容量盈余/短缺。

**💡 创新点**

创新点在于：①将Parr曲线视作隐式需求而非直接人力需求，②在模型中显式加入容量限制，③通过最小二乘法校准累计进度并使用滚动预测验证，提供容量诊断。

**🔧 技术方法**

主要技术包括：Parr曲线归一化与参数化、容量约束的微分/差分方程、sprint级离散化公式、累计进度校准、滚动预测评估与误差度量。

**📊 数据集**

使用的数据集为单一真实Scrum项目（22个两周冲刺，1033故事点、5424工时，5-8人团队），通过该项目的历史记录进行校准与验证。

**📈 对比分析**

与常见基线（恒定速度、恒定容量、无约束Parr、Rayleigh/PNR）进行对比，实验显示在此单项目下容量感知Parr模型在滚动预测误差上至少不劣于基线，并能提供容量盈余/短缺诊断；但在多项目泛化和准确性方面仍有提升空间。

**⚠️ 局限性**

局限性包括：仅在单一项目上验证，未建模容量限制下内部路径变化，依赖准确的总努力估计K，缺乏对重工、依赖、学习等内部动力学的描述，以及容量与生产率不完全等问题。

---

## 159. Bridging 3D Gaussians and Semantic Occupancy for Comprehensive Open-Vocabulary Scene Understanding from Unposed Images

**arXiv ID:** 2607.01633 | [PDF](https://arxiv.org/pdf/2607.01633v1)

**作者:** Hu Zhu `[一作]` (Hong Kong Polytechnic University), Chang Wen Chen `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 13582 | [OpenAlex ID](https://openalex.org/A5002277899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 COVScene，一种通过将可渲染的语义高斯原语与密集占据场进行耦合，实现从稀疏无姿态图像中实现姿态无关的 3D 场景重建、开放词汇语义查询和语义占据预测的单一框架。

**💡 创新点**

核心创新在于差分可卷积的体积提升（differentiable volumetric lifting）将高斯原语映射为连续占据场，并通过体积正则化直接反馈给高斯的透明度、几何和语义特征；此外引入占据熵正则化、姿态蒸馏与语义感知几何 Transformer，实现了统一的可渲染与体积约束。

**🔧 技术方法**

技术手段包括 3D 高斯散射（Gaussian Splatting）、语义感知几何 Transformer、CLIP 视觉语言模型、基于卷积的多任务解码器、姿态蒸馏、差分体积提升、占据熵正则化等。

**📊 数据集**

在 ScanNet 与 ScanNet++ 两个室内场景数据集上进行训练与评估。

**📈 对比分析**

与 AnySplat、LSM、Uni3R、Feature-3DGS、LSeg 等基线比较，COVScene 在新视角合成（PSNR/SSIM/LPIPS）保持竞争力；开放词汇语义分割 mIoU 超过 0.55；在无 3D 标签条件下的语义占据预测中，IoU 从 7.68 提升至 18.32，整体性能优于自监督基线，接近监督方法。

**⚠️ 局限性**

局限性包括仅在静态室内场景上验证，计算成本高，动态 4D 场景和在线部署未处理，体积正则化与多任务训练仍耗时。

---

## 160. MxGLUT: A Reconfigurable LUT-Centric Broadcast Dataflow Accelerator for Mixed-Precision GEMM

**arXiv ID:** 2607.01607 | [PDF](https://arxiv.org/pdf/2607.01607v1)

**作者:** Weiyu Zhou `[一作]` (University of Macau), Yuxiang Huan `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为MxGLUT的混合精度LUT加速器，能够在单一计算单元中同时执行FP8‑INT4和FP8‑FP8的GEMM，并通过可重构的广播数据流（RLB）在预填充和解码阶段动态切换工作模式。

**💡 创新点**

创新点在于：①将FP8‑FP8乘法迁移到LUT预计算阶段，彻底消除运行时浮点乘法；②提出统一的LUT‑centric执行框架，使用相同的RAC路径处理两种精度模式；③设计RLB数据流，实现OS与WS之间的轻量级切换，分别针对预填充的部分和解码的权重重用需求优化。

**🔧 技术方法**

核心技术包括：LUT‑based RAC（读取‑累计）单元、共享FP8‑INT4 LUT生成器、FP8‑FP8 LUT生成器、按位平面移位与加权、以及可重构的广播数据流。

**📊 数据集**

实验使用Llama系列模型（1B、3B、8B）在WikiText‑103数据集上评估推理准确性，并在不同上下文长度（1K–8K）下测量延迟和能耗。

**📈 对比分析**

与FIGLUT+VPU、FP8+DQ、Mx Systolic等基线相比，MxGLUT在28 nm 200 MHz工艺下实现了0.492 TFLOPS/mm²的面积效率、11.58 TFLOPS/W的能效；在Llama推理中，预填充阶段速度提升最高2.16×、能耗下降至0.44×，解码阶段速度提升1.49×、能耗下降至0.71×，且极差提升不超过1.7%。

**⚠️ 局限性**

局限性包括：仅针对FP8‑centric权重‑仅量化模型；对高精度（FP16、BF16）支持不足；在长上下文解码时FP8‑FP8注意力的能耗提升有限，主要受DRAM带宽限制；且需要对FP8次正规化采用flush‑to‑zero以保持数值稳定。

---

## 161. BOUNDARY_SYNC: Measuring Communication-Induced Representational Coupling in Multi-Agent LLM Systems

**arXiv ID:** 2607.01600 | [PDF](https://arxiv.org/pdf/2607.01600v1)

**作者:** Zewen Liu `[一作]` `[通讯]` (Qilu Institute of Technology), Zewen Liu (Qilu Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Boundary_Sync 协议，用来量化大语言模型（LLM）多代理系统中的沟通耦合，通过对比通信前后的 Jensen‑Shannon 散度得到 Coupling Amplification Factor (CAF)。

**💡 创新点**

创新点在于：①建立了可复现、无状态的通信耦合度量 CAF；②证明不同模态（文本、图像）在适当基线下表现出相同比例的同质化；③发现群体规模是耦合方向的关键调节因子；④展示了模型差异对 CAF 的极端影响。

**🔧 技术方法**

技术方法包括：使用 GPT‑4o 真实 API 调用、Beta(α,β) 混合权重、Jensen‑Shannon 散度、Bootstrap 95% 置信区间、跨模型复制（DeepSeek、Qwen3.7‑Plus）以及噪声 DeGroot 模拟。

**📊 数据集**

实验数据来自：合成职业场景图像（512×384）与无文本刺激的文本任务，输出为 5 类概率向量（neutral、biased_female、biased_male、stereotype_avoidant、stereotype_reinforcing），共 30 组 GPT‑4o 实验与多模型小规模验证。

**📈 对比分析**

比较方法：对照无通信基线、随机混合比例、提示扰动、群体规模变化、跨模型复制；结果显示 GPT‑4o 在文本和图像下 CAF ≈0.80–0.84（显著同质化，d≈1.3），群体规模从 5 降至 3 时 CAF 反向升至 >1.0，提示耦合无累积效应，统计显著且置信区间不跨 1。

**⚠️ 局限性**

局限性：①自我锚定效应未完全剔除；②群体规模与提示长度混淆；③模态与刺激存在交叉影响；④人物与温度异质性影响基线多样性；⑤仅用 5 类概率和 10 类嵌入测度验证，未覆盖开放式生成；⑥单一性别刻板任务可能不具代表性；⑦跨模型结果受输出格式等工件影响；⑧ CAF 作为比例指标受类别数与定义影响。

---

## 162. Hawk: Harnessing Hardware-Aware Knowledge for High-Performance NPU Kernel Generation

**arXiv ID:** 2607.01590 | [PDF](https://arxiv.org/pdf/2607.01590v1)

**作者:** Junyi Wen `[一作]` (Sun Yat-sen University), Yanlin Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 15735 | [OpenAlex ID](https://openalex.org/A5100322617)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

本文设计了 Hawk 框架，实现了无需训练的 NPU kernel 自动生成，结合 LLM 与硬件感知知识库；

**💡 创新点**

核心创新点在于三大模块：Run‑Time Knowledge Synthesis 的三部位结构化表示、Bottleneck‑Aware 2D‑Retrieval 结合语法与硬件语义，以及 Effect‑Driven Knowledge Distillation 基于执行反馈的知识净化；

**🔧 技术方法**

技术手段包括 GLM/Claude LLM 推理、触发式知识提取、Triple‑Part Executable Knowledge Representation、BM25 与稠密嵌入的 Reciprocal Rank Fusion、LLM 驱动的语义仲裁与知识蒸馏；

**📊 数据集**

使用 ops‑nn 开源 NPU kernel 库（约170 条）构建知识库，并在 CANNBench L1/L2（共 20 类算子）进行测试验证；

**📈 对比分析**

与 Vanilla LLM、Few‑Shot 与 CANNBot 三个基线在 Compilation Success、Correctness、Speedup 与综合 Score 上对比，Hawk 将准确率从 49.4% 提升至 80%，Speedup 达 2.2×，Score 超过 100；

**⚠️ 局限性**

局限性包括需手工或自动化流程构建初始知识库、对 NPU 版本与 API 更新的敏感性、以及在极弱 LLM 或高度复杂算子下性能仍有限。

---

## 163. Made to Feel: How Designers Bring Emotions into Affective Visualization

**arXiv ID:** 2607.01593 | [PDF](https://arxiv.org/pdf/2607.01593v1)

**作者:** Yixin Bai `[一作]` (University of Maryland), Fumeng Yang `[通讯]` (University of Maryland)

**通讯引用:** 710 | [OpenAlex ID](https://openalex.org/A5019781916)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过半结构化访谈15名可视化设计师，定性分析他们在情感可视化中的设计过程，提出情感功能的三层次、设计维度的三方面以及伦理边界的三类考量；

**💡 创新点**

首次系统阐释情感在可视化设计中的功能层次与设计维度，并将情感设计的伦理边界纳入研究框架，为实践提供可操作的设计指南；

**🔧 技术方法**

采用混合主题分析方法对866条访谈编码进行编码与归纳，使用Zoom录音转写、手动校正，并计算Cohen’s κ保证编码一致性；

**📊 数据集**

研究使用的是访谈文本数据（访谈记录），未使用公开数据集；

**📈 对比分析**

本研究为定性探索，不进行实验比较或性能度量，主要通过访谈结果展示设计意图与观众情感体验的差异；

**⚠️ 局限性**

局限在样本仅限于自发性情感可视化项目，缺乏业务/科学可视化场景的代表性；大多数案例为静态设计，未覆盖动画等手段；且依赖受访者自报，缺乏后续实验验证。

---

## 164. Enabling Real-Time AI in O-RAN: Deploying andMeasuring AI Inside a Near-RT RIC xApp

**arXiv ID:** 2607.01583 | [PDF](https://arxiv.org/pdf/2607.01583v1)

**作者:** Lawrence Obiuwevwi `[一作]` (Old Dominion University), Sachin Shetty `[通讯]` (Old Dominion University)

**通讯引用:** 8656 | [OpenAlex ID](https://openalex.org/A5052787847)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在O-RAN Near-RT RIC xApp中实现并部署轻量级AI网络状态分类器（Logistic Regression和浅层MLP），并在OAI/FlexRIC实时测试平台上测量推理和端到端延迟。

**💡 创新点**

创新点在于将训练好的模型直接导出为C语言推理模块、无外部ML运行时、实现可测量的确定性推理时间，并提供可在商用硬件上复现的RIC Workbench和完整的离线‑→在线流水线。

**🔧 技术方法**

采用O-RAN架构、FlexRIC Near‑RT RIC、OAI 5G核心、C语言编译推理、特征提取（MAC、RLC、PDCP、GTP延迟及UE数）以及Logistic Regression、浅层MLP等轻量级模型。

**📊 数据集**

使用结构化合成数据集（32,000个样本，跨层延迟+UE数），每类8,000个样本，用于训练、测试并评估模型性能。

**📈 对比分析**

与基准规则、随机森林、梯度提升、SVM等模型比较，准确率保持在0.88–0.90之间；Logistic Regression推理耗时1–5 µs，MLP 10–25 µs，端到端延迟低于4 ms，满足10 ms Near‑RT RIC控制窗口。

**⚠️ 局限性**

主要限制包括：依赖合成数据而非真实网络流量；测试规模有限（单机、RF仿真）；未实现完整闭环控制；模型在真实动态环境中的泛化性未知。

---

## 165. Scaling Trends for Lie Detector Oversight in Preference Learning

**arXiv ID:** 2607.01567 | [PDF](https://arxiv.org/pdf/2607.01567v1)

**作者:** Oskar J. Hollinsworth `[一作]` (FAR.AI), Chris Cundy `[通讯]` (FAR.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并扩展了 Scalable Oversight via Lie Detectors (SOLiD) 在更大规模 LLM（1B‑405B）以及 Qwen‑3（0.6B‑32B）上的可扩展性和在更真实偏好学习环境中的表现，探究分布移位、数据来源及低成本变体 SOLiD‑Defer 的影响。

**💡 创新点**

① 在超大模型上首次验证 SOLiD 的可扩展性，发现高 TPR 下误报率显著下降；② 提出了并评估了 SOLiD‑Defer（完全拒绝被检测到的回复，消除高成本标签者）并证明其与标准协议性能相当；③ 系统分析了 detector 训练数据与 finetune 数据分布不匹配导致 FPR 急升的风险；④ 对比 on‑policy 与 off‑policy 数据、跨数据集迁移等多种方案，揭示对结果的关键影响。

**🔧 技术方法**

内部激活线性探针（logistic regression on residual stream），SFT、reward‑model 训练与 CISPO RL，PID 控制 KL 约束；4‑bit 量化实现大规模实验；统计检验（Jonckheere‑Terpstra、Pearson、Wald）评估趋势和显著性；GPT‑4o 评估奖励模型的偏差与奖励劫持。

**📊 数据集**

主要使用 DolusChat（65k）、MASK（1k）、TrueFalseFacts（612）进行 detector 训练；任务数据来自 Llama‑3 系列（1B‑405B）和 Qwen‑3 系列（0.6B‑32B）；还利用 on‑policy 生成的真/假对来对比不同数据来源。

**📈 对比分析**

通过 undetected deception rate、finetuning test FPR、reward‑model deception‑preference AUC、KL divergence 等指标进行横向对比；结果显示：在 TPR≥0.9 时，undetected deception 从 1B 的 34% 降至 405B 的 14%，并且 SOLiD‑Defer 与标准协议在高 TPR 下表现相近；on‑policy 或跨域 detector 训练导致 FPR 急升，影响可扩展性；整体上 SOLiD 在更大模型上表现出良好的可扩展性和鲁棒性。

**⚠️ 局限性**

① 仅在部分配置下多种种子，未覆盖所有大规模模型；② KL 目标设定可能不够激进；③ 仅关注可从上下文检测的欺骗，未处理隐藏信息欺骗；④ 仅使用白盒激活探针，未尝试黑盒监控；⑤ 将探针仅用于标注阶段，未直接嵌入 RL 循环；⑥ 标注流程理想化，未涵盖真实标签者多样性；⑦ 未研究推理模型或长期链-of-thought 的监测；⑧ 未评估对模型其它能力的潜在副作用。

---

## 166. Evolutionary Feature Engineering for Structured Data

**arXiv ID:** 2607.01548 | [PDF](https://arxiv.org/pdf/2607.01548v1)

**作者:** Ege Onur Taga `[一作]`, Samet Oymak `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一种基于进化学习的框架 EFE，利用大语言模型（LLM）自动生成可执行的预处理程序（包括时间序列归一化和表格特征工程），并将其嵌入现有机器学习流水线中。

**💡 创新点**

创新点在于：①把预处理视为可演化的 Python 程序；②利用 LLM 在每一步获取数据上下文、统计量和下游验证反馈，指导搜索；③实现可逆、数据特定的时间序列归一化（EFE‑Time）和稀疏、可解释的特征程序（EFE‑Tab），两者均能与现有模型协同提升性能。

**🔧 技术方法**

核心技术包括：LLM 生成/修改代码的提示模板、OpenEvolve 的进化搜索与多样性保持、对程序进行语法与执行验证、在验证集上评估并将性能反馈回 LLM。时间序列侧使用可逆正则化程序，表格侧使用逻辑和算子组合形成特征转换。

**📊 数据集**

时间序列实验使用 GIFT‑Eval 10 组数据（健康、能源、金融、天气、交通、网络等），包含 Covid‑Deaths、M4‑Yearly、Solar‑Hourly 等；表格实验使用 TabArena 9 个二分类数据集（电信、招聘、电子商务、银行、医疗、体育等）。

**📈 对比分析**

与原始模型（identity baseline）和多种 TSFMs（Chronos‑2、TimesFM‑2.5、Moirai‑2.0、Reverso‑Nano）以及表格模型（TabPFN、LightGBM、单棵决策树）进行对比。EFE‑Time 在平均 MASE、WQL、MAE 上提升 3–4%（单个数据集最高 19%），并可转移至其它 TSFMs 继续保持 2–6% 的提升；与模型微调相比，EF‑Time 与微调叠加可实现额外增益。EFE‑Tab 在决策树上平均排名 1.39，显著优于 CAAFE、LLM‑FE；在低样本、低数据量场景下提升更大，且保持可解释性。

**⚠️ 局限性**

局限性包括：进化过程使用固定的超参数（探索–利用比例、岛屿数等）未做系统性调优；对强大 LLM 的依赖（弱模型性能下降）；实验覆盖的基准有限，未对所有现有方法做完整对比；API 费用高导致实验规模受限；程序的可解释性和执行效率仍需进一步评估。

---

## 167. 3DLS: A 3D Logic-Stacked Architecture for Disaggregated LLM Serving

**arXiv ID:** 2607.01617 | [PDF](https://arxiv.org/pdf/2607.01617v1)

**作者:** Jaehun Lee `[一作]` (KAIST), Joo-Young Kim `[通讯]` (KAIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在芯片互连中预填充与解码阶段的KV缓存传输与解码侧张量并行集合通信之间的共享路径争用，并提出3DLS架构解决该瓶颈。

**💡 创新点**

通过3D堆叠将预填充池与解码池分层，并使用垂直3D互连实现KV缓存传输与解码侧通信的物理隔离，从而消除共享路径争用。

**🔧 技术方法**

采用3D逻辑堆叠芯片架构、UCIe‑3D垂直互连、张量并行、预填充-解码分离、虚拟通道调度以及trace‑driven模拟器评估。

**📊 数据集**

在LLaMA3‑8B、LLaMA3‑70B、OPT‑175B模型上，使用Azure Conversation（Conv）和Code（Code）两组真实推理工作负载。

**📈 对比分析**

将3DLS与Naive Planar（共享平面）和工作负载感知的PM‑Planar（共享平面+虚拟通道）两种基线在相同带宽条件下对比，结果显示相对Naive Planar，E2E延迟可降低60.2%，吞吐量提升1.49倍；相对PM‑Planar，延迟下降18.2%，吞吐量提升1.11倍。

**⚠️ 局限性**

需考虑3D堆叠的面积、功耗与热设计约束，以及垂直互连的制造缺陷和产率问题；当前研究未深入探讨功耗/能效、不同张量并行度下的动态调度等进一步细化。

---

## 168. Scaling with Confidence: Calibrating Confidence of LLMs for Adaptive Test Time Scaling

**arXiv ID:** 2607.01612 | [PDF](https://arxiv.org/pdf/2607.01612v1)

**作者:** Xuqing Yang `[一作]` (Shanghai Jiao Tong University), Xuhong Wang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了 Correctness and Confidence Calibration Reinforcement Learning（C3RL）算法，用来同时提升大语言模型的答案准确率和置信度校准，并基于校准后的置信度推出 Confidence-based Adaptive Test Time Scaling（CAS）自适应推理时缩放方案。

**💡 创新点**

创新点包括：①在 RL 奖励函数中整合正确性奖励、校准奖励以及参考准确性奖励，避免模型出现“校准但错误”的行为；②使用置信度阈值 t=5 细粒度区分“确定”与“不确定”，并用贝塔分布计算停止概率实现高效早停；③将校准置信度直接用于推理时动态分配计算资源，显著降低推理成本。

**🔧 技术方法**

技术手段：强化学习（C3RL）与贝塔分布停止准则；置信度阈值与分数转换；自适应推理（CAS）算法；基于多模态与文本的评测框架；使用 Qwen2.5VL‑7B‑Instruct 和 Llama‑3.2‑3B‑Instruct 作为实验模型。

**📊 数据集**

训练数据：从 NuminaMath‑TIR、WebInstruct‑verified、LogicNLI 与 LogiQA 中挑选 106k 训练样本，按“全对”“部分对”“全错”三类标注；测试集 1k。评测基准：文本 AGIEval、MMLU；多模态 MMMU、MathVista、LogicVista；OOD 文本 MMLU、FOLIO、GSM8K 等。

**📈 对比分析**

与 Base、Self‑Consistency、SFT+Ref、RLVR、SaySelf、RLCR 等基线对比。C3RL 在准确率和校准指标（ECE、AUROC）上与 RLCR 相当，且在 OOD 场景下显著优于 SaySelf（准确率提升 7.9% 文本、19.2% 多模态，ECE 更低）。CAS 在保持或提升准确率的同时，将推理预算降至 Adaptive‑Consistency 的 40% 以内，最高可节省 12.33 倍。

**⚠️ 局限性**

局限性：①科学领域训练样本不足，可能导致训练偏差；②虽实现了较好准确率‑校准平衡，但校准性能仍略逊于 SaySelf；③在小样本（N<8）推理时 CAS 的优势不明显，需要进一步优化停止阈值和置信度校准。

---

## 169. SINA: A Fully Automated Circuit Schematic Image to Netlist Generator Using Artificial Intelligence

**arXiv ID:** 2607.01609 | [PDF](https://arxiv.org/pdf/2607.01609v1)

**作者:** Saoud Aldowaish `[一作]` (University of Utah), Morteza Fayazi `[通讯]` (University of Utah)

**通讯引用:** 351 | [OpenAlex ID](https://openalex.org/A5060525404)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了SINA，一个全自动、开源的电路原理图图像到SPICE网表生成管线；

**💡 创新点**

创新点包括：①兼容IC级与PCB级原理图并支持手绘/扫描图；②引入交叉线检测与解耦算法，区分真实连点与穿越；③将YOLO目标检测、连通组件标记、OCR、VLM和图同构验证集成；

**🔧 技术方法**

采用YOLOv11/YOLOv8等目标检测模型、Connected‑Component Labeling（CCL）求连通性、Morphological + contour 方法做交叉线检测、EasyOCR + GPT‑4o 进行文本抽取与设计ator 分配、图同构算法验证网表结构；

**📊 数据集**

使用自建的700+标注原理图数据库，涵盖IC、PCB、手绘、扫描、文本书籍等多样化样式，包含10种元件类型、超过1500个元件实例；

**📈 对比分析**

与公开的 Masala‑CHAI 及传统方法对比，SINA 在文本提取准确率97.5%、元件检测F1 99.4%、电路结构准确率96.7%，总体准确率达96.67%，比基准高2.72倍；功能仿真通过率97.5%，拓扑匹配92.5%，精确匹配90%；

**⚠️ 局限性**

局限性主要在于：①对极其复杂或非标准符号的识别仍可能失误；②交叉线检测目前仅针对正交或斜交，非标准交叉可能漏检；③VLM在极端旋转/倾斜文本上识别率下降；④整体推理时间受大型模型影响，未针对实时场景做轻量化。

---

## 170. Testing Unate Distributions

**arXiv ID:** 2607.01573 | [PDF](https://arxiv.org/pdf/2607.01573v1)

**作者:** Daeho Lee `[一作]` (Massachusetts Institute of Technology), Ronitt Rubinfeld `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 7291 | [OpenAlex ID](https://openalex.org/A5041567023)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了在{±1}^n上对“单调性”扩展的分布——即单调或反单调方向未知的“单调分布”（unate distributions），并提出了两类测试：1) 在无条件采样模型下的均匀性测试；2) 在子立方子条件采样模型下的单调性测试。

**💡 创新点**

创新点在于首次对unate分布进行研究，提出了能以Θ(n^{3/2}/ε^2)样本/查询实现均匀性和单调性测试的算法，并给出了匹配的下界；尤其引入了“弱方向学习”框架，证明了单元之间的协方差衰减，从而避免了对未知方向σ的完整学习；同时在子立方模型下实现了与单调分布相当的复杂度。

**🔧 技术方法**

主要技术包括：1) 对单调分布的弱方向学习与协方差分析；2) 利用Berry–Esseen定理对协方差上界；3) 通过“单调分解”与随机方向旋转构造下界；4) 结合子立方条件采样的“边缘偏差”检测与Talagrand等同构不等式。

**📊 数据集**

本研究基于理论分析，无需使用具体实验数据集，所有结果均为理论证明与上界/下界。

**📈 对比分析**

与传统需要Θ(n)样本/查询的单调分布均匀性测试相比，unate均匀性测试需Θ(n^{3/2}/ε^2)样本；在子立方条件采样模型中，unate单调性测试的查询复杂度为Θ(n^{3/2}/ε^2)，匹配已知单调分布的Θ(n/ε^2)下界的上界和Ω(n^{2/3})下界。

**⚠️ 局限性**

限制包括：1) 对unate分布的均匀性测试在高维下仍超线性；2) 只考虑了子立方1维查询的下界，尚未探讨更高维子立方；3) 结果对ε的依赖为ε^{-2}，在某些应用中可能仍不够高效；4) 未考虑实际数据中分布近似unate的情况。

---

## 171. Spatial Support Matters: Geometry-Aware Graph Fusion for Rainfall Field Reconstruction

**arXiv ID:** 2607.01621 | [PDF](https://arxiv.org/pdf/2607.01621v1)

**作者:** Low Jun Yu `[一作]` (National University of Singapore), Lucy Amanda Marshall `[通讯]` (University of Sydney)

**通讯引用:** 6102 | [OpenAlex ID](https://openalex.org/A5053383515)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个几何感知的多支持异构图神经网络，用于融合雨量观测中的点、线、面三种空间支持，进行细尺度降雨场重建。

**💡 创新点**

创新点在于将观测的空间支持（0D点、1D线、2D网格）显式编码为不同节点层，并通过跨支持消息传递保留各自几何约束；同时采用留一遮蔽的诱导节点训练，解耦观测分辨率与重建分辨率。

**🔧 技术方法**

主要技术包括异构图卷积、基于支持的边构造、Laplacian位置编码、留一遮蔽训练、以及可插入目标节点实现任意格网重建。

**📊 数据集**

使用新加坡（70个雨量站、414条微波链路、1 km雷达格网）和悉尼（75个雨量站、两种不同分辨率的雷达和卫星格网）两组真实降雨数据。

**📈 对比分析**

与IDW、克里金、卷积融合和支持无关的HGNN做对比；在新加坡数据上多支持HGNN在RMSE上比IDW下降23.2%，比卷积模型低约30%，并在支持无关HGNN上提高约12%；在悉尼数据上增益不显著，表明在已被点观测充分采样的情况下提升有限。

**⚠️ 局限性**

局限在于只考虑三种预设支持类型，未验证更复杂支持；两组测试区同时变化空间相关性与支持配置，难以完全分离二者影响；模型对快速衰相关降雨更有效，低相关性场景收益有限。

---

## 172. ADVENT: LLM-Driven Automatic Predicate Invention for ILP

**arXiv ID:** 2607.01585 | [PDF](https://arxiv.org/pdf/2607.01585v1)

**作者:** Tingting Yu `[一作]` (National Sun Yat-Sen University), Yihuang Kang `[通讯]` (National Sun Yat-Sen University)

**通讯引用:** 552 | [OpenAlex ID](https://openalex.org/A5006616917)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ADVENT，一个结合 LLM 推理与 Prolog 推理的自动谓词发明框架，增强 ILP 的表达能力。

**💡 创新点**

创新点在于：①利用 LLM 自动生成带有语义化名称与定义的辅助谓词；②通过 Prolog 形式化验证循环来修正 LLM 产出的谓词；③构建知识池实现跨任务谓词重用与组合，保持规则可解释性。

**🔧 技术方法**

技术手段包括：大语言模型（如 GPT‑5‑Codex、grok‑4.1‑fast 等）进行 abductive 生成；Prolog 作为 deductive 验证器；ILP 引擎 Aleph 进行规则诱导；Representation Check 评估是否需要谓词发明；知识池管理与复用机制。

**📊 数据集**

实验数据集为 UCI Poker Hand（经转化为 Michalski Train 形式）共 9 个手牌概念（从一对到皇家同花顺），并在 7 个 LLM 上重复 5 次，形成 45 轮试验。

**📈 对比分析**

比较方法：将 ADVENT 与 ILP 仅模式（位置依赖/位置自由）对比；在无验证、LLM 自评、正式验证（ADVENT）与去除知识池等四种设置下评估。性能提升：ILP 仅在位置依赖下 6/9 成功，位置自由 0/9；ADVENT+ILP 在正式验证下平均 80% 成功率，比无验证的 58% 提升显著；知识池提升可达 +31% 最高，整体效果显著优于基线。

**⚠️ 局限性**

局限性：①知识池未做主动过滤，随着规模增大可能因上下文窗口限制导致关注稀释；②实验仅覆盖变换后扑克手牌数据，缺乏更广泛、真实世界的关系数据验证；③对 LLM 的推理质量仍高度依赖，低性能模型在单次生成中可能表现不佳。

---

## 173. Multi-Rate Nonlinear Model Predictive Control for Wall-Supported Bipedal Locomotion of Quadrupedal Robots

**arXiv ID:** 2607.01574 | [PDF](https://arxiv.org/pdf/2607.01574v1)

**作者:** Taizoon Chunawala `[一作]` (Virginia Tech), Kaveh Akbari Hamed `[通讯]` (Virginia Tech)

**通讯引用:** 1148 | [OpenAlex ID](https://openalex.org/A5066501467)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种多速非线性模型预测控制(MR-NMPC)框架，联合规划四足机器人壁面支撑的双足步态和环境反作用力，实现壁面支撑的双足步态。

**💡 创新点**

创新点在于将步态规划和反作用力规划统一到同一多速NMPC中，使用指示函数将慢速步距更新与快速反作用力解耦，并采用单刚体(SRB)模板模型实现实时规划。

**🔧 技术方法**

使用MR-NMPC、CasADi/IPOPT求解器、全局非线性全身控制(WBC)的二次规划、虚拟约束等技术。

**📊 数据集**

在RaiSim环境下使用Unitree A1机器人进行仿真，生成随机木块障碍的7m轨迹集（250条）用于评估。

**📈 对比分析**

与基于Raibert启发式的步态规划对比，MR-NMPC在高速度下成功率提高2.9倍，能够在约束包围盒内保持稳定并抵抗50N冲击。

**⚠️ 局限性**

主要局限是对速度估计噪声敏感，导致前倾和失去周期性；硬件验证仅在模拟，缺少更强基线对比和感知不确定性处理。

---

## 174. Boosting Infrared Small Target Detection via Logit-Domain Contrast and Adaptive Shape Refinement

**arXiv ID:** 2607.01555 | [PDF](https://arxiv.org/pdf/2607.01555v1)

**作者:** Handong Zeng `[一作]` (Hunan Normal University), Hongshan Yu `[通讯]` (Hunan University)

**通讯引用:** 1759 | [OpenAlex ID](https://openalex.org/A5100383577)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在红外小目标检测任务中，提出了 AC‑SLSIoU 损失函数，利用 Logit‑Domain Margin Constraint、Adaptive Boundary Suppression 与 False‑Alarm Focal Loss 三种新约束，提升了弱目标检测率与目标形状精度。

**💡 创新点**

创新点在于：①将损失从概率域迁移到无界 Logit 域进行对比约束，缓解梯度消失；②设计尺度自适应的边界抑制环，精细化目标边缘；③针对高置信度误检引入聚焦式负样本损失，进一步压制误检。

**🔧 技术方法**

采用对比学习思想的 Logit‑Domain Margin Constraint、形状约束的 Adaptive Boundary Suppression 以及聚焦负样本的 False‑Alarm Focal Loss，并在 MSHNet 结构上实现；训练使用 AdaGrad，学习率 0.05，batch 4，400 epoch。

**📊 数据集**

实验数据集为公开的 IRSTD‑1k 与 NUDT‑SIRST 两个红外小目标数据集。

**📈 对比分析**

与传统滤波、局部对比、低秩方法以及最新的 ISNet、DNANet、MSHNet、MIRSAM、TDA、PConv+SDLoss 等深度学习方法对比，AC‑SLSIoU 在 IoU、P_d 提升约 4–5% 的同时将 F_a 降低至 1/3 以内，成为实验集上表现最优的方案。

**⚠️ 局限性**

局限在于改进仅发生在训练阶段，推理时无额外开销，但对不同场景分布的泛化能力以及在更大尺度目标上的适用性仍需进一步验证。

---

## 175. Autonomous discovery of traffic laws with AI traffic scientists

**arXiv ID:** 2607.01639 | [PDF](https://arxiv.org/pdf/2607.01639v1)

**作者:** Xingyuan Dai `[一作]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences), Fei-Yue Wang `[通讯]` (Macau Institute of Systems Engineering, Macau University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 TrafficSci——一个基于大型语言模型的自主 AI 科学家系统，通过文献检索、假设诱导和观测/干预验证的闭环流程，自动发现和验证城市交通规律。

**💡 创新点**

创新点在于首次将闭环、可审计的 AI 驱动科学发现框架应用于无实验室的复杂交通系统；通过多代理协同实现从文献到实验再到假设演进的全自动化，并首次揭示了城市驾驶行为的跨城市一致的时间记忆尺度。

**🔧 技术方法**

使用 GPT‑5.5 等 LLM 进行知识检索（Lit‑LATS）、假设生成与评判（critic‑judge 循环）、实验流程自动化（MCP 接口、SUMO 模拟、统计工具）以及 Wasserstein 距离等评估指标；整体系统依托 LLM 辅助的多任务架构。

**📊 数据集**

实验数据包括：四座城市（东京、巴黎、纽约、北京）的游客量与移动数据、城市与郊区的拥堵成本（jam‑prints）数据、ASC 试验的 CBEngine 仿真结果，以及 Argoverse 2（6 城市）和 nuScenes（2 城市）的车辆轨迹数据。

**📈 对比分析**

在四个案例中自动重现了已知的三条交通规律（逆平方访问量缩放、拥堵成本幂律分布、ASC 进入效益对数关系），并发现了新的时间记忆尺度规律。实验结果显示跨城市 Wasserstein 距离小于 0.24，验证稳定；系统能够在无人工假设的前提下完成重现，证明闭环流程有效。

**⚠️ 局限性**

局限性包括：依赖已公开的文献与数据，可能产生检索或样本偏倚；对罕见事件或长期结构变化的覆盖有限；仅在有限城市/数据集上验证，普适性待进一步检验；需要人工监督以保证发现质量和安全。

---

## 176. Output-Sensitive Construction of CDAWGs from BWT-Runs

**arXiv ID:** 2607.01636 | [PDF](https://arxiv.org/pdf/2607.01636v1)

**作者:** Yuta Tsuruzono `[一作]` (Hokkaido University), Shunsuke Inenaga `[通讯]` (Kyushu University)

**通讯引用:** 2086 | [OpenAlex ID](https://openalex.org/A5012002493)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

设计了一种基于 BWT 运行数 r 的输出敏感算法，能够在 O(log n·log(n/r)) 时间、O(r·log(n/r)+) 额外空间内构建长度为 n 的字符串的 CDAWG。

**💡 创新点**

创新点在于同时利用硬 Weiner 链（发现最大重复节点）和软 Weiner 链（生成二级 DAWG 边）并通过右闭包与左闭包的双重闭包搜索来完成节点合并；该方法直接从 BWT 运行信息构造 CDAWG，避免显式构造后缀树或 DAWG。

**🔧 技术方法**

使用了后缀树/后缀数组的 BWT 相关操作、Weiner 链、右闭包/左闭包搜索、Gagie‑Navarro‑Prezza 的完全功能压缩后缀树接口以及子树等价判定等技术。

**📊 数据集**

论文未给出实验数据或具体数据集，全部结果仅在理论上给出。

**📈 对比分析**

与之前的 r‑enum 方法（O(n) 时间、O(r) 空间）相比，该算法将时间降到 O(log n·log(n/r))，空间略升至 O(r·log(n/r))，在理论上取得更优性能，但未提供实验对比。

**⚠️ 局限性**

主要限制包括：依赖完整的压缩后缀树接口，需要预先构造 BWT 及相关结构；对极大规模文本的实际实现与内存占用未验证；对特殊字符 $ 与 # 的处理需要额外步骤。

---

## 177. DRDN: Decoupled Representation Dynamic Network for From-Scratch ViT Class-Incremental Learning

**arXiv ID:** 2607.01630 | [PDF](https://arxiv.org/pdf/2607.01630v1)

**作者:** Bingchen Huang `[一作]` (Meituan), Yuanchao Du `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了从零开始的 Vision Transformer 动态扩展模型 DRDN，用于类增量学习；

**💡 创新点**

创新点包括：① 在共享 backbone 上持续做掩码图像重建，并仅将梯度限制在 backbone，保持通用表示；② 对每一层 Transformer 进行任务 Token 层级扩张，并采用改进的 per‑task attention 规则，减少跨任务干扰；③ 结合知识蒸馏与多样性损失，进一步抑制遗忘；

**🔧 技术方法**

使用技术：Vision Transformer、Masked Image Modeling (MIM)、动态任务 Token 扩张、任务特定注意力、KL 目标蒸馏、Diversity Loss、Grad‑routing 等；

**📊 数据集**

实验数据集：CIFAR‑100（B0、B50）、ImageNet‑100、ImageNet‑1000；

**📈 对比分析**

与 DyTox、DKT、DER 等 token‑expansion 与回放基线在从零开始 ViT 设置下对比；在 CIFAR‑100‑B0 10 步平均准确率 77.19%（比 DyTox 75.08% 提升 1.36%），BWT 更小，长序列（20 步）优势更明显，整体表现优于现有方法；

**⚠️ 局限性**

局限性：训练时间约 +37%（仅离线增量阶段可接受）；仅在无预训练 backbone 上验证，未探讨预训练 ViT 或在线流式场景；MIM 解码器仅训练时使用，推理时无额外开销。

---

## 178. Online Segment 3D Gaussians via Launching Virtual Drones

**arXiv ID:** 2607.01628 | [PDF](https://arxiv.org/pdf/2607.01628v1)

**作者:** Liwei Liao `[一作]` (Peking University), Ronggang Wang `[通讯]` (Peking University)

**通讯引用:** 3784 | [OpenAlex ID](https://openalex.org/A5050071143)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出SAGO框架实现3D Gaussian的在线交互式分割，无需场景预处理。

**💡 创新点**

通过虚拟无人机与NBV规划将分割转化为视图搜索，实现设置无关的在线分割。

**🔧 技术方法**

结合3D Gaussian splatting、SAM2二位分割、虚拟无人机的下一最佳视图规划与Mask‑shaped Frustum Filtering。

**📊 数据集**

在SPIn‑NeRF、NVOS、LERF‑Mask、3D‑OVS等多场景数据集上进行评估。

**📈 对比分析**

与多种离线及在线方法对比，mIoU/ mAcc 与SOTA相当或略优，推理时间从40‑60秒降至0.4‑0.9秒，实现约50倍速度提升。

**⚠️ 局限性**

中心基MFF难以处理尖锐边界，且依赖初始视图质量，重叠遮挡严重时性能会受限。

---

## 179. Multi-THuMBS: Multi-person Tracking of 3D Human Meshes Beyond Video Shots

**arXiv ID:** 2607.01626 | [PDF](https://arxiv.org/pdf/2607.01626v1)

**作者:** Jeongwan On `[一作]` (UNIST), Seungryul Baek `[通讯]` (UNIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在多镜头视频中实现多人人体三维网格的连续追踪，尤其在出现镜头切换时保持身份一致性并重建全局对齐的三维运动轨迹。

**💡 创新点**

①首次在多镜头、多人人体场景下进行三维网格追踪；②引入基于三维场景重建的共享空间，实现跨镜头的几何对齐；③提出结合几何、姿态和外观的多模态Re‑ID方法；④采用全局优化与跨镜头一致性约束，消除镜头切换导致的位移和身份漂移。

**🔧 技术方法**

利用4DHumans对每帧进行SMPL网格估计；VGGT进行边界帧的三维点云重建并与网格对齐；DROID‑SLAM追踪非边界帧相机姿态；分阶段优化（2D投影、轮廓、深度）实现网格与点云的精确配准；Hungarian算法实现跨镜头身份匹配；最终通过全局平滑与跨镜头重投影约束完成轨迹一致性。

**📊 数据集**

EgoHumans、EgoBody、Harmony4D（改造后加入镜头切换）、AVA、Friends、The Big Bang Theory等多视角与现实视频数据集；使用公开的多视角视频与自制的跨镜头剪辑进行实验。

**📈 对比分析**

与Multishot、GVHMR、PromptHMR、HSfM、KPR、Pose2ID等基线进行对比；在MPJPE/MPVPE、ATE、身份切换数、PCK*、Jitter、Foot Sliding等指标上均达成或逼近SOTA，尤其在镜头切换处实现更低的误差和更少的身份漂移。

**⚠️ 局限性**

仅适用于镜头切换发生在同一物理场景内（intra‑scene）时；跨场景的完全不重叠切换无法通过共享三维空间实现身份跟踪；重建过程对计算资源有一定需求。

---

## 180. Path planning for unmanned naval surface vehicles

**arXiv ID:** 2607.01631 | [PDF](https://arxiv.org/pdf/2607.01631v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 181. Profit-Based Counterfactual Explanations for Product Improvement: A Case Study of Manga Sales in Japan

**arXiv ID:** 2607.01610 | [PDF](https://arxiv.org/pdf/2607.01610v1)

**作者:** Keita Kinjo `[一作]` (Kyoritsu Women's University), Takeshi Ebina `[通讯]` (Meiji University)

**通讯引用:** 270 | [OpenAlex ID](https://openalex.org/A5044885966)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于利润最大化的反事实解释框架PBCE，直接生成可实施的产品与价格调整方案以提升利润。

**💡 创新点**

创新点在于：①把传统CE的目标值设定改为利润最大化，消除外部目标设定；②把距离函数解释为属性调整成本，为CE提供经济学意义；③结合理论解析与实证验证，首次在管理营销场景下展示PBCE。

**🔧 技术方法**

使用受限单调神经网络（CMNN）进行需求预测，并结合线性回归做对照；利用数值优化（SLSQP等）求解利润最大化问题，理论上给出线性模型的闭式解。

**📊 数据集**

使用合成模拟数据（5000个样本）以及真实的日本漫画销售数据（204条记录），包含价格、销量以及视觉和文本提取的多维特征。

**📈 对比分析**

通过5折交叉验证比较MSE/MAE，CMNN与线性回归预测性能相近；PBCE在模拟中几乎完美恢复理论最优，在真实漫画数据中平均利润提升约3%，并显著降低最低利润风险，价格和属性调整均保持经济可行性。

**⚠️ 局限性**

局限包括：仅基于单一预测模型生成CE，缺乏鲁棒性；未建立因果关系，CE仅为预测驱动；数据集规模有限、存在偏差；未对价格调整成本建模，且未扩展到竞争市场结构。

---

## 182. pykci: A Compact Urban Knowledge Graph for Semantic and Spatial Queries using LLMs

**arXiv ID:** 2607.01605 | [PDF](https://arxiv.org/pdf/2607.01605v1)

**作者:** Huynh Duc An Son Nguyen `[一作]` (HafenCity University Hamburg), Youness Dehbi `[通讯]` (HafenCity University Hamburg)

**通讯引用:** 463 | [OpenAlex ID](https://openalex.org/A5023864581)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将 CityGML 2.0 数据集转换为紧凑的 Neo4j 知识图谱（包含所有主题模块和细节级别），提供 3D Tiles 可视化和无损往返导出，并通过 LLM 的文本‑>Cypher 接口实现自然语言查询。

**💡 创新点**

① 采用单一、紧凑的图模式，显著减少查询路径长度和 LLM 上下文量；② 将 CityGML 的结构信息（如顺序、XLink、属性）完整映射为图属性和关系，保证无损往返；③ 通过模型无关的文本‑>Cypher 转换，结合 R‑tree 空间索引，实现对大规模城市数据的语义+空间联合查询。

**🔧 技术方法**

Python 编写的全流程管道；Neo4j（LPG）数据库；Neo4j Spatial 扩展（R‑tree）；OGC 3D Tiles；LLM（Gemma, Mistral, Qwen 以及 Claude Opus）作为自然语言到 Cypher 的转换器。

**📊 数据集**

Hamburg LoD2 CityGML 数据集（388,267 建筑），FZK‑Haus 多 LoD 参考模型，Railway Scene 多主题模块场景；同时在实验中使用公开的城市建筑数据和合成数据集。

**📈 对比分析**

与 3DCityKG（图数据库）和 3DCityDB（PostgreSQL/PostGIS）对比：在同一数据集下，pykci 的导入耗时与 3DCityKG 相当（约 873 s），但图大小更小（≈ 11.4 GB vs 18.3 GB）。查询性能上，pykci 的文本‑>Cypher 生成的查询更短（≈ 290 字符），数据库执行时间最快（≈ 0.37 s），显著优于 3DCityKG（≈ 2.10 s）和 3DCityDB（≈ 2.62 s）。

**⚠️ 局限性**

① 需要在线数据库，无法在离线环境下直接查询；② 对细粒度结构比较（如子图匹配）不如 3DCityKG 等保持完整层次的映射；③ 受限于图模式设计，某些 CityGML 的深层语义（如 Appearance 模块）未覆盖；④ 依赖 LLM 进行文本‑>Cypher 转换，若 LLM 生成错误仍需人工校正。

---

## 183. A Single Patch Is Not Enough: Deterministic Fusion of Repair Candidates

**arXiv ID:** 2607.01597 | [PDF](https://arxiv.org/pdf/2607.01597v1)

**作者:** Boyang Yang `[一作]` (Yanshan University), Haoye Tian `[通讯]` (Aalto University)

**通讯引用:** 567 | [OpenAlex ID](https://openalex.org/A5101397373)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 deterministic atomic evidence fusion 方法（包括 repair‑neighborhood fusion、representative selection 与 evidence‑constrained fusion），将固定的候选补丁池合并为单一可审计的最终补丁。

**💡 创新点**

创新点在于：①完全基于静态交叉候选证据，无需测试或模型评估；②在候选补丁层面进行分支级别的编辑原子融合，能够重构原本未出现的完整修复；③使用“同一表面”约束和局部编辑证据实现高效且可解释的决策。

**🔧 技术方法**

技术包括：差异（diff）解析、文件与标识符重叠度量、Repair‑Neighborhood 构建、投票式代表选取、局部编辑原子分解、静态支持计数、基于 lexicographic evidence 的融合策略。

**📊 数据集**

数据集为：SWE‑bench Verified（500 bug）、SWE‑bench Multilingual（300 bug）以及 Defects4J pass@10（371 bug）。

**📈 对比分析**

与 token medoid、rank‑aggregation、Agentless、LLM‑naturalness、DeepSeek‑V4‑Pro listwise/自由形式融合等基线比较。该方法在 Verified 上 426/500、Multilingual 上 236/300、Defects4J 上 87/371，均优于所有同类基线，且与候选可达上限仅差 17、41 和 39。决策时间仅 3.28 ms/bug，且无模型调用成本。

**⚠️ 局限性**

局限性包括：①依赖候选池的多样性和覆盖度，单一模型池时可达度仍受限；②仍存在局部候选内选择误差（同一 neighborhood 内误选）；③仅处理基于 diff 的补丁，对非 diff 或复杂多语言 AST 解析存在约束；④在缺少测试或运行信息时仍无法排除误修复，可能导致部分错误补丁通过。

---

## 184. ProWAFT: A ROMA-LPD Instance for Workload-Aware and Dynamic Fault Tolerance in FPGA-Based CNN Accelerators

**arXiv ID:** 2607.01602 | [PDF](https://arxiv.org/pdf/2607.01602v1)

**作者:** Xinxin Chen `[一作]` (University of Chinese Academy of Sciences), Jingwen Ma `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 7065 | [OpenAlex ID](https://openalex.org/A5100400247)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 ProWAFT 框架，利用 FPGA 的局部重配置实现工作负载感知的主动故障容错，针对 CNN 加速器进行自适应 TMR 配置。

**💡 创新点**

将工作负载关键性评分、故障传播风险模型以及重配置开销整合为复合目标，实时选择最佳 TMR 分区配置，实现性能、能耗与可靠性的三维平衡；采用主动而非被动重配置。

**🔧 技术方法**

FPGA 局部重配置、三模冗余（TMR）、工作负载关键性评分（WCS）、故障传播因子（FPF）/可靠性风险评分（RRS）、基于 MDP 的决策策略、离线故障注入与在线性能估算。

**📊 数据集**

ResNet‑18、MobileNetV2 与 EfficientNet‑Lite 的 500 层任务追踪（包含 Conv、DepthwiseConv、Pool、FC），以及模拟的时变 SEU 注入模型。

**📈 对比分析**

与静态基础、静态 TMR、反应式重配置三种基线对比，综合指标为复合成本 C_total；ProWAFT 在吞吐率提升 45.9%、能耗下降 30.3%、成功率保持 98.8%、复合成本降低 30.8%。决策开销 ≤ 0.5 ms，主动重配置比被动恢复快 3.3 倍。

**⚠️ 局限性**

实验基于软件仿真注入的 SEU，未覆盖所有物理失效机制；依赖离线表格和预先建模；在更大规模加速器库或长时间部署时需进一步验证。

---

## 185. SemHash-LLM: A Multi-Granularity Semantic Hashing Framework for Document Deduplication

**arXiv ID:** 2607.01601 | [PDF](https://arxiv.org/pdf/2607.01601v1)

**作者:** Xinyi Fang `[一作]` (Independent Researcher), Yuhang He `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SemHash-LLM框架，用多粒度语义哈希和LLM裁决实现大规模文档去重

**💡 创新点**

创新点在于将LLM嵌入到哈希学习中，构建语义投影哈希、注意力加权MinHash、对比学习阈值自适应和不确定性驱动的LLM裁决，形成级联过滤流水线

**🔧 技术方法**

采用知识蒸馏的LLM学生编码器、可学习的语义投影哈希、注意力加权一致加权采样的MinHash、对比学习边界网络、MC Dropout不确定性估计、LLM判别提示和多粒度融合网络

**📊 数据集**

在100GB RedPajama网络文本上进行实验，包含五类去重场景

**📈 对比分析**

相较于SimHash、MinHash、NearDup‑BERT、DedupLM，SemHash-LLM的最终加权得分从81.20提升到91.05，且在各子任务均超过基线

**⚠️ 局限性**

局限性包括需要先训练LLM蒸馏模型、对超长文本或高噪声数据的鲁棒性尚未充分评估，以及流水线复杂度和维护成本较高

---

## 186. MVFusion-GS: Motion-Variance Guided Temporal Attention for High-Quality Dynamic Gaussian Splatting

**arXiv ID:** 2607.01578 | [PDF](https://arxiv.org/pdf/2607.01578v1)

**作者:** Jianwei Hu `[一作]` (Tsinghua University), Bin Wang `[通讯]` (Tsinghua University)

**通讯引用:** 120859 | [OpenAlex ID](https://openalex.org/A5100338047)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MVFusion-GS框架，利用运动方差引导的时间注意机制提升3D Gaussian Splatting在动态场景重建和去干扰背景重建中的表现。

**💡 创新点**

创新点在于引入全局运动轨迹签名与局部运动方差统计作为运动先验，并结合跨帧Transformer时间注意，实现单一管线下同时精准分离动态前景与静态背景。

**🔧 技术方法**

使用技术包括3D Gaussian Splatting、DeGauss分支结构、Motion-Variance Guided Refinement（MVG）、MotionFormer Temporal Attention（MFTA）、跨帧自注意力机制，以及基于光度、SSIM和正则化的多任务损失。

**📊 数据集**

实验数据集涵盖NeRF On-the-Go、RobustNeRF和Neu3D等动态场景与去干扰背景基准。

**📈 对比分析**

与多种SOTA方法（如DeGauss、WildGaussians、SpotlessSplats、4DGS等）在PSNR/SSIM/LPIPS上进行对比，MVFusion-GS在多数指标上位居第一，动态前景PSNR提升约3-4dB，背景LPIPS显著下降。

**⚠️ 局限性**

限制在于对基线变形场的依赖，运动方差在初始阶段可能缺乏辨别力，遮挡或极弱运动情况下仍可能残留伪静态前景，需要进一步提升对复杂遮挡和运动缺失的鲁棒性。

---

## 187. Geometric Signatures of Reasoning: A Spectral Perspective on Task Hardness

**arXiv ID:** 2607.01571 | [PDF](https://arxiv.org/pdf/2607.01571v1)

**作者:** Aria Masoomi `[一作]` (Northeastern University), Vahab Mirrokni `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了链式推理（CoT）在Transformer隐藏状态空间中的几何轨迹，提出通过轨迹的谱、位置和运动学特征来评估任务难度和解答正确性。

**💡 创新点**

创新点在于引入“有效维度”（effective dimension）作为轨迹复杂度的谱度量，理论证明更难任务会产生更高维度轨迹；同时利用运动学特征在生成早期就能预测解答正确性。

**🔧 技术方法**

使用了谱分析、PCA降维、运动学特征提取以及逻辑回归、MLP、GRU等分类器。

**📊 数据集**

实验基于MATH500数据集的代数、计数/概率和预备微积分三个类别，使用Qwen2.5-0.5B-Instruct模型生成推理轨迹。

**📈 对比分析**

在任务难度预测上，跨问题的有效维度在第21层可达AUC 0.93；在正确性预测上，逻辑回归在前20%轨迹即可获得AUC约0.81、AUPRC≈0.87。

**⚠️ 局限性**

局限包括仅在单一小模型和有限数据集上验证，未探讨更大模型和其他领域的泛化，且未给出具体的早停或best‑of‑n实现细节。

---

## 188. Mind the Gap: Standard 3DGS Evaluation Primarily Measures Near-Trajectory Interpolation

**arXiv ID:** 2607.01556 | [PDF](https://arxiv.org/pdf/2607.01556v1)

**作者:** Gaoxiang Jia `[一作]` (Advanced Micro Devices), Vikram Appia `[通讯]` (Advanced Micro Devices)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种公平匹配计数的评估协议，用以区分 3D Gaussian Splatting (3DGS) 在轨迹插值与空间外推（extrapolation）下的性能差距，并系统评估了该差距在不同表示、不同数据集与不同随机种子下的普适性。

**💡 创新点**

创新点包括：① 在保持相同训练样本数量的前提下，设计插值与外推两种 hold‑out 方式；② 通过 SH 级别分解将差距归因为几何代理与视角依赖两部分；③ 发现差距与视角间角距离高度相关，可用零成本诊断；④ 对非高斯 NeRF 以及两种 feed‑forward 生成模型同样验证差距，证明其与表示无关；⑤ 开发了覆盖指导的视角选择方法，显著提升外推性能。

**🔧 技术方法**

使用了 3D Gaussian Splatting（官方版、Mip‑Splatting、3DGS‑MCMC）、Instant‑NGP（体素 NeRF）、MVSplat 与 DepthSplat（视角生成网络）等技术；评估指标为 PSNR、SSIM 与 LPIPS；通过 SH 分解、角度距离相关分析及覆盖度诊断实现对差距的量化与解释。

**📊 数据集**

实验数据集包含 10 个真实捕获场景（MipNeRF360 + Tanks & Temples）、6 个由 HY‑World 2.0 生成的视图场景、另外 9 个 MipNeRF360 场景以及 RealEstate10K 用于 feed‑forward 评估；共计 16 场景，502 次训练实验。

**📈 对比分析**

在插值模式下平均 PSNR 约 27 dB，外推模式约 21–22 dB，差距 3–12 dB，远大于方法间常见 0.5–2 dB 的差异；在两种随机种子确认后，插值/外推排名在 2 个场景中出现逆转。即使在多表示下，外推差距仍保持一致；覆盖指导视角选择在大多数场景中显著提升外推质量。

**⚠️ 局限性**

局限性：① 仅使用基于方位角的连续 sector hold‑out，可能不适用于所有空间布局；② 对表示的覆盖范围有限，未覆盖更多 NeRF/生成网络；③ 对随机种子敏感，部分排名变化依赖于种子；④ 覆盖诊断仅在已有视角集上选择，未生成全新相机位置；⑤ 训练过程非确定性导致结果波动。

---

## 189. Beyond Skepticism: Evaluating LLMs Pedagogical Intent Reasoning with the Adaptive Pedagogical Vigilance Framework

**arXiv ID:** 2607.01581 | [PDF](https://arxiv.org/pdf/2607.01581v1)

**作者:** Minghao Chen `[一作]` (Zhejiang University), Yuxin Liu `[通讯]` (Zhejiang University)

**通讯引用:** 31247 | [OpenAlex ID](https://openalex.org/A5100358868)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Adaptive Pedagogical Vigilance (APV) 框架，构建Pedagogical Intent Inference Engine (PIIE) 进行贝叶斯推理，并在三层次评估（识别教学类型、推断教学配置、泛化到真实教学话语）中实验验证其在大型语言模型中的教学动机推断能力。

**💡 创新点**

将警觉性重新定义为针对教学动机的自适应推理，引入贝叶斯教师政策模型与学生推断机制，实现对教学体制（种类、立场、激励）的系统化推断；提供统一的层级评估与对比基准，显著提升 LLM 对教学动机的理解和人类评判的一致性。

**🔧 技术方法**

贝叶斯推理框架、教师策略与学生信念更新的两层模型、链式思维（CoT）与直接提示的对比实验、以及对多模型（GPT‑4o、Claude 3.5、Gemini、Llama 等）的推理与评估。

**📊 数据集**

① 受控翻译教学实验（Level 1）；② 基于角色的教学推断实验（Level 2）；③ 真实教学视频、教师反馈和翻译论坛转录文本（Level 3）。此外进行消融实验检验模型各组成部分的贡献。

**📈 对比分析**

与多种基线 LLM 进行 Pearson 相关性与人类判断的对比，APV 在所有层次下均获得最高相关系数（最高 r≈0.958），相较传统提示和模型平均提升显著；在自然语料下，APV 的相关性保持在 0.28–0.35 范围内，显著高于对照模型。

**⚠️ 局限性**

仅在英语教学情境中验证，真实数据集规模有限，未覆盖多语言和跨文化情境；框架目前仅适用于单教师–单学生的一对一互动，尚未扩展到多方协作学习或长期对话。

---

## 190. When Does Generating More Help? Disentangling Fixed-Source Synthesis from Source Expansion in Synthetic Data Scaling

**arXiv ID:** 2607.01727 | [PDF](https://arxiv.org/pdf/2607.01727v1)

**作者:** Xu Guo `[一作]` (Shanghai AI Laboratory), Qipeng Guo `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将合成数据的规模化拆分为源扩展（SE）和固定源合成（FSS），并通过实验验证 FSS 的增益边界及 SE 的相对优势。

**💡 创新点**

创新点在于：①系统区分并对比 SE 与 FSS 两条扩展路径；②基于重复采样提出固定源规模律并对其进行预测；③在固定源控制下评估多种合成协议，发现其对性能提升有限。

**🔧 技术方法**

使用技术包括：Rejection Sampling（RS）生成教师回答、DeepSeek‑V3.1 作为教师模型、S_Q7/S_L8 作为学生模型；对响应预算 r 拟合 P(r)=P∞−Ar⁻α 的固定源规模律；对比 SE、FSS 以及多种温度、选择、人格提示、无判断过滤、追踪修复等合成协议。

**📊 数据集**

采用的数据集为：SuperGPQA 的数学（1833/406）与物理（2010/458）子集，以及混合 STEM 的 Nemotron‑Post‑Training‑Dataset‑v2 作为异质源。

**📈 对比分析**

比较方法：在匹配总样本预算下对比 SE 与 FSS 的 mean@8 性能；在不同教师–学生对上对多种合成协议进行 r 维度扫频；实验结果表明：FSS 随 r 递减收益，在大预算下 SE 更优；在固定源下多种协议均未显著超越 RS 基线。

**⚠️ 局限性**

局限性：仅针对推理类 STEM 任务；实验计算成本高，难以进一步扩大采样预算；实验固定教师–学生配置，未覆盖更广泛的任务或更大规模模型。

---

## 191. Distributionally Robust Listwise Preference Optimization

**arXiv ID:** 2607.01715 | [PDF](https://arxiv.org/pdf/2607.01715v1)

**作者:** Xudong Wu `[一作]` (University of Hong Kong), Jiayu Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 6746 | [OpenAlex ID](https://openalex.org/A5048308937)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于点位总变差的鲁棒 Plackett–Luce 损失，用于解决大语言模型在列表式偏好标注不确定性下的对齐问题。

**💡 创新点**

创新点包括：①将排名标签不确定性建模为点位 TV 模糊集；②证明最坏排名可通过升序排序得到，复杂度降为 O(K log K)；③在离线凸优化与在线弱凸优化下给出收敛理论。

**🔧 技术方法**

主要技术包括：Plackett–Luce 模型、点位总变差鲁棒优化、升序排序求最坏排名、投影随机子梯度与弱凸 Moreau‑envelope 收敛分析。

**📊 数据集**

使用 UltraFeedback（每个 prompt 4 个候选）进行离线评估，在线评估基于 Qwen3 系列模型在 U10 评估集上，并用 GPT‑4 作为外部 judge。

**📈 对比分析**

与 BT/DPO、非鲁棒 PL、TV‑DR‑DPO、KLDPO 等基线比较；在离线噪声实验中，鲁棒 PL 在干净标签下几乎不损失性能，噪声下显著提升 Kendall τ；在线对齐中，K=4 加鲁棒 λ>0 提升 reward‑model 及 GPT‑4 judge 分数。

**⚠️ 局限性**

局限性：需要手动调节鲁棒半径 ρ，鲁棒性对 ρ 的敏感性；实验规模有限，尚未在更大规模在线对齐任务中验证。

---

## 192. Trust Boundary Semantic Gaps: A Multi-dimensional Analysis and Mitigation for Security-by-Design

**arXiv ID:** 2607.01711 | [PDF](https://arxiv.org/pdf/2607.01711v1)

**作者:** Doyeon Kim `[一作]` (Korea University), Junghee Lee `[通讯]` (Korea University)

**通讯引用:** 12558 | [OpenAlex ID](https://openalex.org/A5100435060)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“信任边界语义缺口”（TBSG）概念，并构建了四维模型（身份、空间、时间、解释）以及设计时评估与缓解框架TBSAM，对实际案例（SolarWinds/SUNBURST）进行回溯性分析。

**💡 创新点**

创新点在于将语义缺口从缺失或错误的校验转向语义不匹配层面，系统化四维缺口分析，并引入P1/P3标记的跨边界传播追踪，提供比传统STRIDE更细粒度的安全需求残留识别。

**🔧 技术方法**

采用形式化边界与语义缺口定义、四维维度推导、优先级分级、传播分析以及与已知控制（如代码签名、可信执行、路径校验、时间绑定等）的映射等技术；框架与STRIDE、SLSA、Zero Trust等方法对齐。

**📊 数据集**

使用了从2014年至2025年的75起公开安全事件（CISA、MITRE、CVE等公开报告）作为案例集进行边界级别分析与模型验证。

**📈 对比分析**

与STRIDE对比时展示了两者在发现与定位缺口上的互补性；虽然没有量化性能指标，但案例分析表明TBSAM能在设计阶段揭示跨边界的语义残留，提升安全设计的完整性。

**⚠️ 局限性**

局限包括：需要安全架构师手工识别边界和安全需求，受限于公开报告的细节，四维模型未覆盖所有可能的语义缺口，实验仅为回溯分析，缺乏实时运行时验证与工具化支持。

---

## 193. Generic Expert Coverage for Pruning SparseMixture-of-Experts Language Models

**arXiv ID:** 2607.01710 | [PDF](https://arxiv.org/pdf/2607.01710v1)

**作者:** Yongqin Zeng `[一作]`, XiuTeng Zhou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于覆盖率的稀疏 MoE 专家裁剪方法 Generic TB-Coverage；

**💡 创新点**

创新点在于通过对不同通用语料库（WikiText2 与 C4）分别评估专家效用，并采用轮询覆盖规则保障跨语料专家多样性，避免单一重要性评分导致的专家偏移；

**🔧 技术方法**

使用 REAP 风格的实用性分数、轮询覆盖策略、基于重建的候选掩码融合以及预算恢复；

**📊 数据集**

仅利用两套通用语料库 WikiText2 与 C4 进行校准；

**📈 对比分析**

与随机裁剪、原始 REAP 以及 ExpertSparsity 进行对比，实验显示在 Qwen1.5-MoE‑A2.7B 与 DeepSeek‑MoE‑16B‑Base 的 25%、50%、75% 保留比例下，Generic TB-Coverage 在六个零样本基准上的平均准确率和 WikiText2/C4 的困惑度均优于基线，尤其在激进裁剪时提升显著；

**⚠️ 局限性**

缺点包括：裁剪后不对路由器进行微调；保护预算 B 固定且需手工设定；未测量裁剪后的推理延迟或内存占用；在推理推理和推理推理推理方面的性能尚未评估；

---

## 194. Efficient Pattern Matching in Unordered Term Tree Patterns with Height Constraints

**arXiv ID:** 2607.01704 | [PDF](https://arxiv.org/pdf/2607.01704v1)

**作者:** Shintaro Matsushita `[一作]` (Fukuoka Institute of Technology), Yusuke Suzuki `[通讯]` (Hiroshima City University)

**通讯引用:** 1540 | [OpenAlex ID](https://openalex.org/A5030137541)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出无序树的“高度约束变量”模式，并给出了求解成员问题的多项式时间算法

**💡 创新点**

在保持无序性与变量可替换整个子树的基础上，引入高度约束，使模式既灵活又能在全局层面控制子树高度

**🔧 技术方法**

使用动态规划结合双亲图匹配（Hopcroft–Karp）来构造继承集与对应集，整体时间复杂度为 O(N·max{nD³/², S})

**📊 数据集**

使用人工生成的无序树（20~200 顶点，5,000 次实验）

**📈 对比分析**

通过对不同顶点数的树测算平均运行时间，实验显示计算时间与树大小近线性增长，平均毫秒级别，验证了算法的实际效率

**⚠️ 局限性**

实验仅限于合成数据，未在真实无序树（如糖链结构）上评估；当树的最大度数 D 或变量的高度约束 S 较大时，时间复杂度可能变高

---

## 195. WARP: Weight-Space Analysis for Recovering Training Data Portfolios

**arXiv ID:** 2607.01686 | [PDF](https://arxiv.org/pdf/2607.01686v1)

**作者:** Tzu-Heng Huang `[一作]` (University of Wisconsin-Madison), Frederic Sala `[通讯]` (University of Wisconsin-Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于权重空间几何的框架WARP，用已发布的基础模型和微调模型权重恢复微调训练所用的域混合比例；

**💡 创新点**

创新点在于利用模型合并生成伪训练轨迹，提取“模仿得分”构成几何足迹，再通过无监督softmax或监督MLP映射至域比例，从而突破传统只做样本级会员推断的局限；

**🔧 技术方法**

核心技术包括线性/球面插值（模型合并）、Mimic Score（梯度投影）、温度softmax读出、以及基于合成混合数据训练的两层MLP投影器；

**📊 数据集**

实验使用四个文本分类数据集（SNLI、AGNews、Yelp、Yahoo）并在BERT-base与GPT-2-Small上进行多种已知域混合的微调；

**📈 对比分析**

与随机猜测、均匀猜测、样本级会员推断以及真实中间检查点（oracle）对比，WARP在BERT上MAE降至0.046，在GPT-2上为0.104，均显著优于基线并且超越了拥有真实训练轨迹的对手；

**⚠️ 局限性**

局限性包括需访问基础模型与微调模型权重，假设可采样的探测数据集且域划分已知；对极端训练方式（如大幅度学习率调整、非线性插值）仍需进一步验证。

---

## 196. Self-Referential $K$-SAT and the Finite Analogue of Gödel's Incompleteness Theorem

**arXiv ID:** 2607.01671 | [PDF](https://arxiv.org/pdf/2607.01671v1)

**作者:** Wen Fang `[一作]` (Beihang University), Ke Xu `[通讯]` (Beihang University)

**通讯引用:** 12318 | [OpenAlex ID](https://openalex.org/A5100665814)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文构造了一类自引用的布尔 K‑SAT 公式（宽度为 O(log N)），证明其满足解的独立性，并给出了结构不可约性（即任何子线性子实例无法区分 SAT 与 UNSAT），从而给出了与哥德尔不完备性相对应的有限形式。

**💡 创新点**

创新点在于：①首次将解的独立性与自引用构造相结合，得到可在有限域上实现的“哥德尔句”；②通过对随机 K‑SAT 的泊松极限定理与信息论方法，建立了从结构不可约性到解的复杂度、证明宽度和指数性搜索成本的严格联系；③将这一结构障碍与 SETH 直接对应，提供了一个信息论视角的计算复杂度下界。

**🔧 技术方法**

主要技术包括：随机组合学（第二阶矩法、投影簇估计、Chen–Stein 近似）、泊松极限定理、信息论（Shannon 熵、Kolmogorov 复杂度、Pinsker 不等式）、结构化证明理论（Resolution 宽度与大小界）以及自引用构造与可扩展的逻辑变换。

**📊 数据集**

本文未使用公开数据集，而是通过理论随机实验——即在 N 个变量上随机生成宽度为 O(log N) 的 K‑SAT 公式，并分析其极限分布和结构性质。

**📈 对比分析**

通过理论分析证明：任何仅在子线性窗口内评估的算法必须具有 Ω(N^{1‑δ}) 的描述复杂度，任何 Resolution 证明至少需要 Ω(N^{1‑δ}) 的宽度，进而导致指数级别的证明大小；这些结果与传统的 SETH 预测保持一致，展示了自引用实例对算法和证明复杂度的严格限制。

**⚠️ 局限性**

局限性包括：①结果仅在 N→∞ 的极限下成立，具体到有限 N 的实际实例尚未验证；②只针对宽度为 O(log N) 的随机 K‑SAT，固定宽度（K≥3）仍不适用；③构造依赖概率方法，缺乏可复制的具体实例；④对实际 SAT 求解器的启发性有限，更多是理论上对复杂度边界的阐释。

---

## 197. Temporal and Cross-Modal Alignment for Enhanced Audiovisual Video Captioning

**arXiv ID:** 2607.01667 | [PDF](https://arxiv.org/pdf/2607.01667v1)

**作者:** Chen Zhao `[一作]` (Nanjing University), Ying Tai `[通讯]` (Nanjing University)

**通讯引用:** 13676 | [OpenAlex ID](https://openalex.org/A5029021362)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了专门针对视听视频字幕生成的TCA-Captioner框架，重点解决音视跨模态绑定与时间一致性问题。

**💡 创新点**

核心创新包括Observer‑Checker‑Corrector（OCC）迭代生成流程、密集人类交互高质量数据集、以及专门评测跨模态绑定与时间关系的TCA‑Bench。

**🔧 技术方法**

技术手段涵盖基于Gemini‑3‑Pro的观察者、双重视觉/音频检查器、视频/音频剪辑检查器、LoRA微调、DPO偏好训练，配合YOLOv8、Silero VAD、Doubao‑1.8等工具。

**📊 数据集**

使用自建的高密度人类交互数据集（HDI，3500个精选视频）、FineVideo（5000）与TikTok‑10M（7000）等公开来源，生成OCC合成字幕用于训练。

**📈 对比分析**

在公开基准（Video‑SALMONN‑2、UGC‑VideoCap）和自研TCA‑Bench上均表现出显著优势，TCA‑Captioner在AV绑定和时间维度上分别达76.9%和76.9% F1，超过Gemini‑3.0‑Pro、UGC‑Captioner等对手。

**⚠️ 局限性**

局限性包括对昂贵LLM（Gemini‑3‑Pro等）的高依赖、OCC流程对算力与token成本高、仍存在少量绑定或时间倒置误差，且数据集主要集中在短视频/节目，未覆盖更广泛场景。

---

## 198. Revisiting Decentralized Online Convex Optimization with Compressed Communication

**arXiv ID:** 2607.01665 | [PDF](https://arxiv.org/pdf/2607.01665v1)

**作者:** Hao Zhou `[一作]` (Zhejiang University), Yuanyu Wan `[通讯]` (Zhejiang University)

**通讯引用:** 9677 | [OpenAlex ID](https://openalex.org/A5026532752)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文首次提出了两种针对压缩通信的FTRL类型算法，用于去中心化在线凸优化（D-OCO），并与现有的OGD类型算法进行了比较。

**💡 创新点**

创新点在于将FTRL算法扩展到压缩通信的场景，并且在算法设计和理论分析上更为简洁，同时在带宽设置下显著改善了现有算法的后悔界限和通信成本。

**🔧 技术方法**

使用了FTRL算法和Choco-Gossip技术来处理压缩通信问题。

**📊 数据集**

使用了ijcnn1和a9a两个数据集进行实验，验证算法在去中心化在线逻辑回归问题上的性能。

**📈 对比分析**

与现有的最佳算法相比，本文的带宽算法在后悔界限上显著改善，分别达到了O(nT^3/4)和O(nT^2/3(log T)^1/3)，并且所需的通信轮次显著减少。

**⚠️ 局限性**

限制在于尚未探讨如何将加速技术与压缩通信结合，以及在全信息设置中是否可以在L=1的情况下获得相同的结果。

---

## 199. Teaching Vision-Language-Action Models What to See and Where to Look

**arXiv ID:** 2607.01658 | [PDF](https://arxiv.org/pdf/2607.01658v1)

**作者:** Yuguang Yang `[一作]` (Beihang University), Xianbin Cao `[通讯]` (Beihang University)

**通讯引用:** 10207 | [OpenAlex ID](https://openalex.org/A5038809760)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DriveTeach-VLA 双模架构，通过驾驶感知视觉蒸馏（DVD）和二维轨迹引导提示（2D‑TGP）让 VLA 学会先识别关键视觉信息再定位规划空间，最终实现端到端的轨迹预测；

**💡 创新点**

创新点在于：①将视觉先验与空间引导解耦，解决传统 VLA 仅依赖文本预训练导致的空间信息缺失；②使用自蒸馏结合 bbox‑增强图像为 ViT 注入驾驶相关视觉先验；③引入 2D‑TGP 作为文本坐标条件，使规划器在学习阶段获得可解释的空间约束；

**🔧 技术方法**

采用 Qwen2.5‑VL‑3B 作为主干网络，Grounding DINO 进行关键对象检测，ViT 自蒸馏对视觉编码器进行预训练；利用 2D‑TGP 投影技术将 BEV 轨迹映射到图像平面；SFT（监督微调）配合 CoT 伪标签训练；最后使用 GRPO 强化学习对规划策略进行对齐；

**📊 数据集**

在 NAVSIM（闭环非反应式）和 nuScenes（开放环）两个公开自动驾驶基准数据集上进行训练与评测；

**📈 对比分析**

与最新 SoTA VLA 如 AutoVLA、RecogDrive、UniAD 等进行对比；在 Navtest 上 DriveTeach-VLA 取得 PDMS 90.4，优于 AutoVLA 的 86.4；在 nuScenes 上 L2 距离 0.30、碰撞率 0.12，均为最佳；

**⚠️ 局限性**

局限性：双模架构推理时延相对较高；视觉蒸馏依赖 Grounding DINO 的检测质量，检测缺失或误检会影响视觉先验，进而影响轨迹预测；

---

## 200. Plug-and-Play Volumetric Reconstruction for Compressive Sensing Light-Sheet Microscopy

**arXiv ID:** 2607.01654 | [PDF](https://arxiv.org/pdf/2607.01654v1)

**作者:** Jianqing Jia `[一作]` (University of North Carolina at Chapel Hill), Yifei Lou `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 3851 | [OpenAlex ID](https://openalex.org/A5062067602)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究压缩感知光束显微镜（CS‑LSM）下的体积重建，提出一种可插拔的PnP‑ADMM框架；

**💡 创新点**

在传统的切片独立重建基础上引入轴向耦合模型，并使用Woodbury矩阵逆与Gauss‑Seidel迭代实现高效更新；

**🔧 技术方法**

实现了多种去噪器（Tikhonov、TV、BM3D、DnCNN、FFDNet、DRUNet）在PnP‑ADMM中的联合重建；

**📊 数据集**

使用合成斑马鱼心脏体积和真实斑马鱼心脏CS‑LSM测量数据进行验证；

**📈 对比分析**

与切片基模型对比，轴向耦合模型在PSNR/SSIM上均有提升，深度学习去噪器表现最佳；

**⚠️ 局限性**

仅对弱凸正则化给出了子序列收敛证明，对深度去噪器的理论收敛性尚未证明，且真实数据缺乏真值只能进行主观评估。

---

## 201. Beyond Pixel Diffs: Benchmarking Image Change Captioning for Web UI Visual Regression Testing

**arXiv ID:** 2607.01728 | [PDF](https://arxiv.org/pdf/2607.01728v1)

**作者:** Licheng Zhang `[一作]` (University of Melbourne), Naveed Akhtar `[通讯]` (University of Melbourne)

**通讯引用:** 7902 | [OpenAlex ID](https://openalex.org/A5069697936)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Web UI 图像变化描述（WUICC）任务，并构建了首个公开的 WUICC-bench 数据集，旨在为视觉回归测试提供自然语言变化说明。

**💡 创新点**

创新点在于：①首次将变化描述任务与 Web UI 视觉回归结合；②使用 LLM 驱动的单一变更生成管线，并配合人类验证，得到 37 条规则覆盖的高质量样本；③提供统一基准与多模型对比，推动领域研究。

**🔧 技术方法**

主要技术包括：利用 GPT‑5.1 生成 HTML 变更与对应自然语言说明；通过无头浏览器渲染前后截图；实现 11 种 IDC（CNN、Transformer、Mamba 等）模型与两款零‑shot VLM（Llama‑3.2‑11B‑Vision‑Instruct、Qwen2‑VL‑7B‑Instruct）的训练与推理。

**📊 数据集**

数据集为 WUICC‑bench，基于 WebSight 生成的 9,906 对前后截图及其标注，按 70/10/20 分为训练、验证、测试集，涵盖 37 条单一变更规则。

**📈 对比分析**

对比 11 种 IDC 方法与两款 VLM，使用 BLEU、METEOR、ROUGE_L、CIDEr、SPICE 以及非更改/更改准确率评估。结果显示训练模型能显著抑制非有意义噪声并保持高检测率，但整体分数仍偏低；VLM 的零‑shot性能更差。

**⚠️ 局限性**

局限性包括：①仅处理单一 atomic 变更，未覆盖多变更或动态内容；②数据来源为 LLM 合成网页，可能与真实生产页面差异较大；③评测指标侧重词汇重叠，未充分衡量文字逐字准确性；④规模虽大但仍不及自然图像数据集，进一步提升仍需努力。

---

## 202. CoRe: Combined Rewards with Vision-Language Model Feedback for Preference-Aligned Reinforcement Learning

**arXiv ID:** 2607.01721 | [PDF](https://arxiv.org/pdf/2607.01721v1)

**作者:** Hexian Ni `[一作]` (Chinese Academy of Sciences), Yinghao Cai `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 4966 | [OpenAlex ID](https://openalex.org/A5000445977)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在机器人操控任务中提出CoRe框架，将奖励分解为基于LLM的可编码正式奖励和基于VLM的视频级残差奖励，实现无需人工标签的偏好对齐策略学习。

**💡 创新点**

通过正式奖励模块（FRM）利用LLM迭代生成和优化可执行代码奖励，并用视觉‑语言模型（VLM）提供偏好与重要性信息来校准残差奖励，从而实现奖励高效、稳定且与人类偏好一致。

**🔧 技术方法**

采用GPT‑4 mini生成正式奖励代码，使用Gemini 2.0 Flash和LIV等VLM进行视频级偏好和重要性标注；结合SAC策略学习与多模态奖励融合。

**📊 数据集**

在MetaWorld（7个任务）和SoftGym（3个任务）模拟数据集以及UR5实物机器人上（5个任务）进行实验。

**📈 对比分析**

与CLIP Score、Eureka、Text2Reward、RL‑VLM‑F、PrefVLM、ERL‑VLM等方法对比，CoRe在所有10个任务中取得近乎100%的成功率，样本效率提升3‑40×，总成本仅约2.00 M token、$0.37、2.15 h。

**⚠️ 局限性**

受限于对LLM/VLM的反复查询，导致一定的计算和费用成本；模型依赖于VLM的偏好标注，可能受到偏见与噪声的影响；在更复杂、无结构的真实环境中需进一步验证鲁棒性。

---

## 203. Structure-Aware Gaussian Splatting for Large-Scale Scene Reconstruction

**arXiv ID:** 2607.01698 | [PDF](https://arxiv.org/pdf/2607.01698v1)

**作者:** Weiyi Xue `[一作]` (Tongji University), Guang Chen `[通讯]` (Tongji University)

**通讯引用:** 484802 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于场景频率收敛的自适应调度框架(SIG)和球面约束高斯，显著提升大规模3D Gaussian Splatting的训练效率与渲染质量。

**💡 创新点**

创新点包括：①从信号恢复角度导出平均采样频率与场景频带；②设计同步图像监督与高斯频率的调度器；③引入球面约束高斯利用点云几何先验；④实现自适应分辨率与密度调度的统一框架。

**🔧 技术方法**

主要技术包括3D Gaussian Splatting、频率一致性调度(SIG)、球面约束高斯、深度一致性正则化、基于块级并行训练与COLMAP点云初始化。

**📊 数据集**

使用的三大数据集为真实场景Mill19、UrbanScene3D和合成场景MatrixCity。

**📈 对比分析**

在与NeRF、GS、DashGS等基线对比中，PSNR平均提升约0.9dB，训练速度提升1.4–1.5倍，SSIM/LPIPS等指标显著优于对手，同时显著减少浮点与冗余高斯。

**⚠️ 局限性**

局限性在于仍需依赖级别细节渲染策略；在极大规模场景下对块划分与深度一致性正则化的敏感性；对深度估计误差较为敏感。

---

## 204. Model Merging as Probabilistic Inference in Fine-Tuning Parameter Space

**arXiv ID:** 2607.01689 | [PDF](https://arxiv.org/pdf/2607.01689v1)

**作者:** Long Minh Bui `[一作]` (Washington State University), Trong Nghia Hoang `[通讯]` (Washington State University)

**通讯引用:** 744 | [OpenAlex ID](https://openalex.org/A5102929916)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于产品专家（PoE）能量模型的模型融合框架，将多任务模型的融合视为对共享参数的MAP推断；

**💡 创新点**

创新点在于把模型融合转化为概率推断，并通过重构专家能量函数揭示现有方法的高斯假设，随后提出重尾Cauchy专家以更好捕捉残差重尾特性，并证明了固定点迭代的收敛性；

**🔧 技术方法**

采用能量基模型（EBM）、PoE结构、Cauchy分布专家、固定点MAP推断算法；

**📊 数据集**

在视觉任务上使用CLIP ViT-B/32、ViT-B/14、ViT-L/14的七个或十三个数据集；在语言任务上使用Flan‑T5-base/large与GLUE 8个数据集以及对三大7B LLM（Vicuna‑7B、Llama‑2‑Coder、WizardMath）在GSM8K上的评估；

**📈 对比分析**

与权重平均、Task Arithmetic、DARE‑TIES、Fisher融合、DOGE‑TA、Concrete‑TA、KnOTS‑TIES等基线相比，所提方法在所有视觉任务的平均精度上提升约6–10个百分点，在GLUE任务上平均排名提升至第1；同时收敛速度快、推理时间低；

**⚠️ 局限性**

限制在于假设任务更新均匀受限于预训练参数，且仅在共享前置模型的场景下实验；对大规模模型的扩展仍需进一步验证；

---

## 205. SCAPE: Accurate and Efficient LLM Training with Extreme Sparse Communication

**arXiv ID:** 2607.01678 | [PDF](https://arxiv.org/pdf/2607.01678v1)

**作者:** Mingkai Zheng `[一作]` (Rutgers University), Zhao Zhang `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

SCAPE提出了一种在大型语言模型预训练中通过稀疏同步AdamS一阶动量来实现通信高效的分布式优化器。

**💡 创新点**

创新点在于利用AdamS一阶动量的时序稳定性和一次延迟的mask同步，避免对梯度进行直接稀疏化，同时通过单个同步缓冲即可完成二阶动量的更新。

**🔧 技术方法**

采用了Top‑K稀疏化、错误反馈、分布式优化器分片、CPU offload双缓冲、单步延迟mask以及在Megatron‑LM框架下的实现。

**📊 数据集**

使用了OpenWebText和SlimPajama‑6B数据集对GPT‑345M与Llama‑500M模型进行预训练。

**📈 对比分析**

与密集的AdamW/AdamS对比，SCAPE在90%/99%稀疏率下保持相同或更优的训练/验证损失，Llama‑500M预训练时间缩短43.3%，Llama‑1.8B可获得3.26×加速，且下游任务表现不受影响。

**⚠️ 局限性**

局限性在于需要额外的残差缓冲和全模型CPU offload，CPU‑GPU间距较大的系统可能产生额外开销，并且在极高稀疏率时仍需进一步验证其稳定性。

---

## 206. HistoSeg++: Delving deeper with attention and multiscale feature fusion for biomarker segmentation

**arXiv ID:** 2607.01675 | [PDF](https://arxiv.org/pdf/2607.01675v1)

**作者:** Saad Wazir `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了HistoSeg++架构，将Nested-UNet与内部/外部注意力单元、SE通道重校准、ASPP以及边缘敏感损失融合，专门用于细胞/核分割；

**💡 创新点**

创新点在于将多尺度上下文捕获与上采样注意力相结合，利用SE重校准与边缘加权损失显著提升边界精度与整体分割质量；

**🔧 技术方法**

使用的技术包括Nested-UNet结构、注意力门、Squeeze‑and‑Excitation、Atrous Spatial Pyramid Pooling、深度可分离卷积、边缘增强交叉熵损失、Adam优化器；

**📊 数据集**

实验数据集包括MoNuSeg、2018 Data Science Bowl（DSB）以及Electron Microscopy（EM）三大公开医学图像分割数据集；

**📈 对比分析**

与UNet、nnU-Net、HoVer‑Net、Swin‑UNet、TransUNet、UNet++、UNet3+、U2‑Net等基线模型进行全面对比，HistoSeg++在IoU、Dice、Precision、Recall和HD95等指标上普遍领先，尤其在MoNuSeg上取得71.44 IoU、83.30 Dice等最佳成绩；

**⚠️ 局限性**

局限性包括计算资源需求较高、训练时间较长、对不同数据集的超参数调优需进一步研究、仅验证单标签二分类分割、尚未证明对多类分割或实时推理的适用性。

---

## 207. Unified Panoramic-Gaussian Representation for Monocular 4D Scene Synthesis

**arXiv ID:** 2607.01663 | [PDF](https://arxiv.org/pdf/2607.01663v1)

**作者:** Yuankun Yang `[一作]` (Fudan University), Li Zhang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出统一的全景‑高斯表示（Panoramic‑Gaussian），通过全景轨迹训练与推理、视频扩散生成与动态高斯散射相结合，实现从单目视频合成几何一致的4D场景，并能在未观测视角下重建隐藏区域。

**💡 创新点**

创新点在于：① 将全景轨迹引导的生成模型与动态高斯散射统一，形成可学习的全景‑高斯框架；② 采用渐进式全景扩展与掩码归一化，结构上防止误差累积；③ 通过掩码均方误差与原始视频对齐，使生成结果可直接蒸馏回3D几何。

**🔧 技术方法**

技术主要包括全景轨迹条件视频扩散（TrajectoryCrafter）、4D Gaussian Splatting（MoSca）、掩码均方误差归一化、渐进式视角扩展与固定几何投影。

**📊 数据集**

使用数据集：训练基于OpenVid，评估用DyCheck IPhone、Nvidia Dynamic、Kubric‑4D。

**📈 对比分析**

与NeRF、Gaussian Splatting、基于相机的扩散模型等方法对比，本文在DyCheck的可见和不可见区域均获得最高PSNR/SSIM/LPIPS；整体每场景耗时约2.0h（4DGS 1.4h + 生成 0.6h），性能优于大多数基线。

**⚠️ 局限性**

局限性：最适用于向内看的、对象中心的场景；对外向捕捉的远景约束弱，质量下降；依赖预训练扩散模型和每场景4DGS优化，难以实时部署，且可能继承扩散模型的伪影。

---

## 208. Message Passing Based Two-Timescale Bayesian Learning for Joint Channel and Memory Hardware Impairments Tracking

**arXiv ID:** 2607.01660 | [PDF](https://arxiv.org/pdf/2607.01660v1)

**作者:** Wei Xu `[一作]` (Zhejiang University), An Liu `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种在大规模MIMO接收机中联合跟踪信道与硬件失真参数的 Bayesian 推理框架，使用残差门控GRU建模信号级记忆失真，并通过两时标马尔可夫先验实现对快速信道和慢速失真漂移的连续在线跟踪。

**💡 创新点**

创新点包括：① 采用两时标马尔可夫先验分别描述快速变化的稀疏信道和慢速漂移的网络参数；② 将残差GRU嵌入因子图，避免逆映射难题；③ 设计基于 Turbo‑OAMP 的信道模块和定制 DAMP 的失真校准模块，并通过期望传播进行模块间自洽推理。

**🔧 技术方法**

技术手段包括残差门控GRU、两时标马尔可夫先验、消息传递（MP）、Turbo‑OAMP、深度近似消息传递（DAMP）、期望传播（EP）以及变分贝叶斯近似（VBI）等。

**📊 数据集**

实验数据来源于 QuaDRiGa 生成的 3GPP 城市宏基站 NLOS 通道（6 GHz、1 m/s、4 载波符号），以及合成的硬件失真模型（相邻链路耦合、IQ 失衡、GMP 放大器），不使用任何预先训练的硬件模型。

**📈 对比分析**

与基线方法（GMP+Turbo‑OAMP、GRU+Turbo‑OAMP、MP‑BDL、MP‑BDL‑Frozen）在静态及慢漂移失真条件下对比，MP‑TTBDL 在所有 SNR 级别下均显著降低信道估计 NMSE，尤其在高 SNR 处差距最大，并且随着跟踪时间的推移性能持续提升。

**⚠️ 局限性**

局限性在于仅考虑单用户、固定 ULA 结构，未对多用户多天线场景进行推广；算法仍以两模块间频繁消息传递为核心，计算复杂度相对较高；对硬件失真模型的假设（仅邻链耦合）限制了在更复杂失真环境下的适用性。

---

## 209. Domain Generalization via Text-Anchored Information Bottleneck

**arXiv ID:** 2607.01657 | [PDF](https://arxiv.org/pdf/2607.01657v1)

**作者:** Eunyi Lyou `[一作]` (Seoul National University), Joonseok Lee `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在域泛化任务中，提出一种纯文本引导的文本信息瓶颈（Text‑Anchored Information Bottleneck）方法，通过将预训练文本嵌入作为固定的域不变锚点，消除视觉编码器中传递的域相关噪声，并利用条件信息瓶颈压缩掉输入中的域特定信息，从而学习到更具域不变性的特征。

**💡 创新点**

创新点：① 放弃视觉引导，改用文本空间作为唯一的域不变监督源；② 将文本锚点与条件信息瓶颈相结合，既保证特征对标签的充分性，又显式压缩与标签无关的输入信息；③ 设计三项互补损失（语义蒸馏、压缩、对齐），实现对文本锚点的有效利用；④ 证明该框架在多种骨干网络上保持显著优势，表明方法具有通用性。

**🔧 技术方法**

主要技术：文本信息瓶颈（Conditional Information Bottleneck，CEB），基于预训练 CLIP 文本编码器的固定锚点；语义蒸馏损失、压缩损失（聚类长度），对齐损失（余弦相似度）；使用 von Mises–Fisher 分布进行变分近似；训练时对所有图像特征进行聚类并与文本锚点对齐。

**📊 数据集**

使用六个标准域泛化基准：TerraIncognita、OfficeHome、VLCS、PACS、DomainNet、NICO++，并在不同骨干（ResNet‑50、ViT‑B/16、CLIP‑ViT‑B/16 等）上进行评估。

**📈 对比分析**

与多种基线（LP、MIRO、RISE、VL2V、CLIPood、CLIP‑专用方法等）在 Leave‑One‑Domain‑Out 和 NICO++ 的 Leave‑One‑Group‑Out 方案下对比，均实现了或超过 0.5–1.5% 的平均提升，在所有骨干上均保持正向改进，尤其在 CLIP 预训练模型上实现了 4–6% 的绝对提升，达到当前最优水平。

**⚠️ 局限性**

局限性：① 依赖稳定的外部文本锚点，缺乏上下文信息；② 对于需要上下文线索的任务或开放集域泛化，固定锚点可能不够；③ 可能受数据泄漏影响（如 CLIP 预训练已接触部分域）；④ 目前未探索自适应锚点或多模态联合优化。

---

## 210. DeadPool: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint

**arXiv ID:** 2607.01646 | [PDF](https://arxiv.org/pdf/2607.01646v1)

**作者:** Haotian Xie `[一作]` (Rutgers University), Zhao Zhang `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种面向大规模 LLM 训练的容错系统，通过异步的零开销内存检查点与在线节点热替换，实现在不终止整个作业的情况下快速恢复。

**💡 创新点**

创新点主要包括：①利用非关键路径的异步内存检查点机制实现对优化器状态的实时复制，完全隐藏在训练关键路径之外；②设计分布式通讯器重建协议，在节点失效时在线将失效节点替换为备用节点，恢复通信拓扑，避免全作业重启。

**🔧 技术方法**

技术方案涵盖：异步设备‑主机（D2H）与主机‑主机（H2H）复制管线、MPI 级异步传输、PyTorch 与 Megatron‑LM 的深度集成、ZeRO‑2 优化器状态管理、3D 并行（张量、流水线、数据并行）架构、专用控制平面（TCPStore）与节点自适应重构协议。

**📊 数据集**

实验使用 GPT‑style 变压器模型，参数规模从 0.6B 到 65B，训练数据为标准 LLM 预训练语料（未公开具体数据集）。

**📈 对比分析**

与传统周期性检查点/重启方案对比，评估了错误-free 期间的每步开销、恢复延迟和总训练时间。结果显示：①检查点开销几乎为零，训练步时不受影响；②节点失效时恢复时间不超过 40 秒，显著低于 150 秒的重启恢复；③在 Perlmutter（A100）和 Vista（H200）上，规模可扩展到 512 GPU，模型 65B 参数，系统保持线性扩展与稳定恢复。

**⚠️ 局限性**

局限性包括：目前主要针对单节点失效（可通过 k‑副本扩展多节点失效，但需拓扑感知）；依赖预留备用节点和控制平面；在高度相关的硬件故障（同机架、同交换机）下仍需更复杂的副本策略；实验仅覆盖两套超级计算机，尚未在更大规模或不同框架（如 DeepSpeed）上验证。

---

## 211. When Agents Do Not Stop: Uncovering Infinite Agentic Loops in LLM Agents

**arXiv ID:** 2607.01641 | [PDF](https://arxiv.org/pdf/2607.01641v1)

**作者:** Xinyi Hou `[一作]` (Huazhong University of Science and Technology), Haoyu Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并解决大型语言模型（LLM）智能体中因反馈路径未被有效终止而导致的无限代理循环（IAL）问题，提出了一款将源代码抽象为框架无关的 Agent IR 并构造 Agentic Loop Dependence Graph（ALDG）的静态分析工具，对反馈路径进行检测。

**💡 创新点**

创新点包括：①首次对 IAL 进行正式定义；②提出框架无关的 Agent IR 与 ALDG 两级抽象，用以捕捉跨框架的代理执行语义；③结合 SCC 分析与边界覆盖判定，实现高效的 IAL 检测；④利用 LLM 辅助过滤降低误报，提升实用性。

**🔧 技术方法**

技术手段：基于 Python AST 的静态分析、数据流与控制流推断、Agent IR 与 ALDG 的构造、强度分析与 SCC 检测、边界覆盖评估；辅助 LLM（GPT‑5.5）进行候选筛选；对八大主流 LLM 代理框架（LangChain、LangGraph、AutoGen、CrewAI、OpenAI Agents SDK 等）进行支持。

**📊 数据集**

数据集：从 GitHub 采集 6,549 个带 star 的 Python LLM 代理仓库，包含 246,748 个文件、33.41M 行代码，用于评估工具的检测能力。

**📈 对比分析**

对比方法：与两种 LLM 基线（纯 LLM API 与 Codex 编码助手）对比。结果显示工具覆盖 100% 的已确认 IAL，精度 91.9%，每个项目平均 4.2K token，分析时长 31.2 秒；基线在召回率、token 消耗和运行时间上均逊色。

**⚠️ 局限性**

局限性：①静态分析的过度近似导致误报与漏报；②仅支持 Python 与八大框架，未覆盖其它语言或自定义框架；③对高度定制的用户语义、外部状态控制与自然语言输出的边界推断不够精准；④LLM 辅助过滤不稳定，可能产生差异；⑤仅能检测结构性错误，无法捕获纯运行时失效。

---

## 212. Resilient Liquid Democracy: Mitigating Voting Power Imbalances via Secure Delegation Networks

**arXiv ID:** 2607.01730 | [PDF](https://arxiv.org/pdf/2607.01730v1)

**作者:** Zhuolun Li `[一作]` (University of Leeds), Evangelos Pournaras `[通讯]` (University of Leeds)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并评估了一种使用时限解密隐藏形成阶段的液体民主机制，结合排名多重委托与回退投票；

**💡 创新点**

创新点在于将委托形成阶段进行加密封闭，利用去中心化的时限解密实现可验证但隐藏的委托，并引入多级委托与个人备用票以提升鲁棒性；

**🔧 技术方法**

使用去中心化时限解密（基于区块链的时间机器）、分数共享、加密投票、图遍历算法和可验证性等技术；

**📊 数据集**

采用四个真实数据集：Aarau市政预算投票、Pabulib 20个城市预算投票、CES 2022美国全国选民调查、Zürich实验室预算投票；

**📈 对比分析**

通过实验对比透明委托与封闭委托、个人与代表式委托、不同聚合规则，评估代表性准确率、投票权集中度、对委托失败的容错率，发现封闭委托在代表性提升有限但能显著降低权力集中，并在针对性失败下将投票流失降至约3%；

**⚠️ 局限性**

主要局限是委托关系模型仅为两层，缺乏多级委托与真实匿名身份验证；实验中使用的专业水平指标为代理指标；未在真实在线平台中进行现场部署。

---

## 213. When Algebraic Symmetry Breaking Meets Solvers: An Experimental Study

**arXiv ID:** 2607.01726 | [PDF](https://arxiv.org/pdf/2607.01726v1)

**作者:** Madalina Erascu `[一作]` (West University of Timișoara), Johannes Middeke `[通讯]` (Temple University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了使用自动生成的多项式对称破坏约束（包括线性与非线性）对整数线性规划的影响，重点在近半容量箱装问题上；

**💡 创新点**

提出了可同时生成线性与非线性对称破坏约束的代数方法，并在不同求解器中评估其“求解器感知”特性；

**🔧 技术方法**

使用代数工具生成多项式模板，对称破坏家族形式为 h(Px)-h(x)≤0，随后在 Gurobi、CPLEX、SCIP、Hexaly、Z3 等求解器中分别处理原始、非线性、内部重写与显式线性化三种模式；

**📊 数据集**

使用基于箱装问题的四类实例（Class 3、5、7、9），每类在不同规模 n=99/100/700/900 或 n=1000/1024/2000 的随机生成实例；

**📈 对比分析**

通过将原始模型作为基准，比较加入对称破坏约束后的求解时间、求解速度与求解器自带对称处理的差异。结果显示：Gurobi 在处理原生二次约束时最有利，线性化或内部重写往往导致性能下降；CPLEX 对二次约束重写后效果不一，线性化后相对稳定；SCIP 与 Hexaly 对静态破坏约束的响应不明显；Z3 在 SAT 任务中受破坏器大小影响显著，适当平衡变量数与排列数可提升性能；

**⚠️ 局限性**

限度：二次约束的效益高度依赖求解器实现；大规模或已有快速求解的实例添加破坏器可能适得其反；Hexaly 与部分求解器对静态破坏约束支持不足；破坏器生成需动态阈值，且破坏器数量与结构对性能有显著影响。

---

## 214. Pmeta-TLA: Backdoor Attacks for Speech Classification Models via Meta-Learning with Timbre Leakage Attack

**arXiv ID:** 2607.01702 | [PDF](https://arxiv.org/pdf/2607.01702v1)

**作者:** Yueming Huang `[一作]` (Xiangtan University), Weiping Wen `[通讯]` (Peking University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于音色泄露（TLA）触发器的语音模型后门攻击方法，并通过元学习与PCGrad实现一次性多后门植入与快速微调

**💡 创新点**

1) 音色泄露触发器在帧级别注入音色信息，既隐蔽又自然；2) 采用元学习训练模型“学习植入后门”，实现对新触发器的快速适应；3) 结合PCGrad缓解多任务（清洁任务与后门任务）梯度冲突，提高后门效果与鲁棒性

**🔧 技术方法**

元学习（MAML）、梯度冲突投影（PCGrad）、自监督语音编码器（SSM）与声码器（vocoder）

**📊 数据集**

Google Speech Commands v2（KWS）和部分Speaker Verification数据集

**📈 对比分析**

与PIBA、DABA、Ultrasonic、PBSM、VSVC等基线后门攻击进行对比；实验显示Pmeta‑TLA在多模型（ERes2Net、KWS‑ViT、EAT‑S、CAM++）上均实现了更高的攻击成功率（ASR）且所需毒样本数（PN）更少；在多后门（t=5）情况下仍保持高ASR且PN更低；在五种防御（Fine‑tuning、Pruning、STRIP、Spectral Signatures、Trigger Filtering）下表现出较强的鲁棒性

**⚠️ 局限性**

1) 对于极端高强度或更复杂的防御（如结合多种防御的混合策略）效果尚未完全评估；2) 触发器仍需在语音数据集外验证其对跨域、跨语言的泛化能力；3) 该攻击依赖对模型的白盒访问和对训练数据的写入权限，攻击成本与实际部署场景相关

---

## 215. ICDepth: Taming Video Diffusion Models for Video Depth Estimation via In-Context Conditioning

**arXiv ID:** 2607.01677 | [PDF](https://arxiv.org/pdf/2607.01677v1)

**作者:** Xuanhua He `[一作]` (Hong Kong University of Science and Technology), Qifeng Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出ICDepth框架，利用预训练的文本到视频扩散Transformer通过In-Context Conditioning实现视频深度估计。

**💡 创新点**

创新点包括SAND-Attention（通过RoPE对齐和单向注意力消除噪声污染）和SRFM（注入DINOv2语义和分辨率先验）来提升几何精度和泛化能力。

**🔧 技术方法**

技术手段包括扩散模型的流匹配训练、In-Context Conditioning、Transformer自注意力、RoPE对齐、语义与分辨率特征调制、低步数ODE采样。

**📊 数据集**

训练数据为约0.8M帧的合成与真实视频（VKITTI、TartanAir、TartanGround、OmniWorld子集），测试覆盖Sintel、KITTI、ScanNet、Bonn等公开基准及低照度、夜间、海底等挑战场景。

**📈 对比分析**

与现有判别式（Video Depth Anything）与生成式（DepthCrafter、Depth Any Video、ChronoDepth）方法对比，ICDepth在AbsRel、δ1、RMSE、Temporal Alignment Error等指标上均达到或超过SOTA，且数据量仅为其他生成式方法的1/6~1/13，推断速度与显存占用在可接受范围内。

**⚠️ 局限性**

局限性包括：仍依赖预训练的文本到视频模型，若场景与训练分布差异过大仍可能出现误差；在极长视频或极高分辨率下仍需更高采样步数以保持一致性；当前模型主要针对单目深度，尚未覆盖多相机或多视角融合。

---

## 216. UT-AISTimprt submission for ICME 2026 Grand Challenge on Academic Text-to-Music Generation

**arXiv ID:** 2607.01669 | [PDF](https://arxiv.org/pdf/2607.01669v1)

**作者:** Shunsuke Yoshida `[一作]` (University of Tokyo), Satoru Fukayama `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在低资源、小模型的文本到音乐生成任务中，通过基于文本或音频嵌入的聚类进行批量采样，并评估其对模型性能的影响。

**💡 创新点**

提出在训练时按同一聚类构建mini-batch的策略，并发现文本聚类能提升目标指标，同时在聚类细粒度上实现客观与感官质量的权衡。

**🔧 技术方法**

使用FluxAudio模型、CLAP和T5文本编码器、VAE+BigVGAN音频VAE、k-means聚类、AdamW优化和梯度裁剪等技术。

**📊 数据集**

采用Jamendo数据集（约3.7K小时的10秒音频片段）以及官方挑战的文本提示集合。

**📈 对比分析**

在ICME 2026 Grand Challenge的官方评测中，文本-500模型在FAD、CLAP和CSS指标上均优于FluxAudio-S基线，并在参数与数据对比下表现出更高的数据效率；额外实验显示文本聚类（50/500簇）在客观指标上均优于基线，且50簇最为优异。

**⚠️ 局限性**

仅探索了单一模态聚类、固定簇数（1/50/500）且未系统评估数据增强、多模态联合聚类、不同模型规模和训练周期的影响。

---

## 217. VeriChat: An Agentic Conversational AI Assistant for Hardware Security Verification

**arXiv ID:** 2607.01668 | [PDF](https://arxiv.org/pdf/2607.01668v1)

**作者:** Dipayan Saha `[一作]` (University of Florida), Farimah Farahmandi `[通讯]` (University of Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了VeriChat，一个面向硬件安全验证的多智能体对话式助手，能够在用户交互中提供安全知识检索、上下文优化、验证工具集成以及结果反馈。

**💡 创新点**

创新点包括：① 基于主题划分的多检索器架构和加权RRF融合，显著提升检索相关性；② 四阶段EDA工具流水线与LLM的闭环协作，实现从语法检查到形式验证的自动化；③ 通过检索验证与自检机制抑制幻觉，实现高达87.73%的事实忠实度和92%的抗幻觉率。

**🔧 技术方法**

技术包括：检索增强生成（RAG）、多智能体工作流（QUOA、HRA、GA）、多主题向量数据库、权重RRF融合、文本嵌入（OpenAI text-embedding-ada-002）、FAISS索引、开源EDA工具（Icarus Verilog、Yosys、SymbiYosys）、BMC+Z3形式验证、LLM提示工程与自检。

**📊 数据集**

使用了28,221篇经过人工验证的硬件安全研究论文（约61GB）构建的主题分层知识库，并结合实时Google搜索结果；此外在评估中构造了由25名专家贡献的150条多样化查询作为基准。

**📈 对比分析**

评估方式包括：1）检索层的上下文召回/精确度、2）生成层的答案相关性/提示符合度、3）端到端的事实忠实度、抗幻觉率、用户偏好（Elo评分）。与八款主流专有聊天机器人比较，VeriChat在事实忠实度上领先23个百分点，抗幻觉率高92%，并在Elo评测中获得最高分（2350）。

**⚠️ 局限性**

局限性主要体现在：① 仍依赖第三方LLM接口，无法完全控制生成质量；② 知识库扩展需人工验证，更新速度有限；③ 仅覆盖开源EDA工具，商业工具集成不完善；④ 对极端复杂或高度定制化验证流程的支持仍需进一步验证。

---

## 218. CALM: Interpretable Cross-Modal Alignment for Biomarker Discovery from Unpaired Data

**arXiv ID:** 2607.01656 | [PDF](https://arxiv.org/pdf/2607.01656v1)

**作者:** Jueqi Wang `[一作]` (Boston University), Archana Venkataraman `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出 CALM 框架，利用未配对的神经影像与遗传数据学习可解释的脑区-通路关联。

**💡 创新点**

创新点在于通过类条件线性投影结合 CMMD、对比损失和正交正则化，实现未配对数据的分布对齐与可解释性，并能保持诊断分离。

**🔧 技术方法**

采用模态特定编码器、线性投影、Class‑Conditional Maximum Mean Discrepancy (CMMD)、监督对比损失、正交正则化及两阶段训练。

**📊 数据集**

使用 ABIDE (MRI)、SSC (遗传) 作为训练集，ACE (配对 MRI 与遗传) 作为测试集；影像采用 Brainnetome 246 ROI、4 个形态特征，遗传聚合到 177 KEGG 通路并结合 6 个 GWAS 表型。

**📈 对比分析**

与 G‑MIND、UNSEEN、SUE 等基线比较，在 ACE 数据集上准确率 0.606±0.03、AUC 0.606±0.03，显著优于其他方法。

**⚠️ 局限性**

局限性：仅在 ASD 上验证，需扩展到其他疾病；线性投影可能不足以捕捉更复杂的跨模态关系；未探索非线性或多模态深度融合的潜在改进。

---

## 219. Boosting Ultrasound Image Classification via Attribute-Guided Dual-Branch Framework

**arXiv ID:** 2607.01648 | [PDF](https://arxiv.org/pdf/2607.01648v1)

**作者:** Bo Zhao `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种可插拔的属性引导双分支框架，用于提升超声图像分类的准确性与可解释性。

**💡 创新点**

创新点在于将无监督预训练的CLIP文本特征与医学属性表相结合，构建属性语义空间，并通过自适应融合模块在决策层进行低成本自纠正。

**🔧 技术方法**

采用预训练视觉编码器（如ResNet、ViT、Mamba）、CLIP文本编码器、属性预测与正则化损失、以及可学习的融合权重进行联合训练。

**📊 数据集**

在BUSI三分类乳腺超声数据集和内部多中心胎儿标准平面七分类数据集上进行实验。

**📈 对比分析**

与多种基线（ResNet、ViT、Mamba等）以及多任务设置对比，AttrGuide在BUSI上从87.86%提升到88.72%，在胎儿任务上平均提升约5.8个百分点，且训练开销不到5%。

**⚠️ 局限性**

局限在于需人工制定属性表，对不同任务的属性覆盖度可能有限；且模型仍受限于训练数据分布，迁移到其他超声任务时需重新构造属性表。

---

## 220. Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling

**arXiv ID:** 2607.01642 | [PDF](https://arxiv.org/pdf/2607.01642v1)

**作者:** Xingyu Zheng `[一作]` (Beihang University), Haotong Qin `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 MrFlow，一种无训练的多分辨率加速框架，结合低分辨率结构生成、像素空间GAN超分、低强度噪声注入和单步高分辨率细化，实现对预训练流匹配模型的高效加速。

**💡 创新点**

创新点在于：① 采用低分辨率快速生成全局结构并利用其低频信息；② 在像素空间使用预训练GAN超分保持结构同时补充高频；③ 通过低强度噪声仅针对高频误差进行重采样；④ 只需一次高分辨率采样即可完成细化；⑤ 与时间步蒸馏可叠加，提升至25×加速。

**🔧 技术方法**

使用技术包括：流匹配（flow matching）模型、VAE编码/解码、预训练Real‑ESRGAN超分、基于Euler的ODE采样、低强度噪声注入、以及多阶段（低分辨率→像素超分→高分辨率）流水线。

**📊 数据集**

在 FLUX.1‑dev 和 Qwen‑Image‑20B 两大文本到图像预训练模型上进行实验，使用 1024×1024 分辨率的生成任务。

**📈 对比分析**

与多种无训练加速方法（Teacache、DB‑Taylor、ToMA、RALU、SPEED、LSSGen）以及训练依赖的时间步蒸馏方法（SenseFlow、Pi‑Flow、FLUX‑schnell）进行对比。实验显示 MrFlow 在无训练条件下实现 10×+ 的整体加速，同时保持 OneIG‑Bench 误差 ≤1%，在生成质量、速度和灵活性方面均优于现有无训练方案；与时间步蒸馏结合可实现 25× 的加速。

**⚠️ 局限性**

局限性包括：① 仍需预训练的VAE、GAN超分网络；② 对超分网络的依赖可能导致在极端高分辨率或不同域的泛化受限；③ 低强度噪声策略需要在不同模型/数据上手动调节；④ 对非流匹配模型的适用性尚待验证。

---

## 221. Consistent Scene Understanding in 3D Gaussian Splatting via Multi-Cue Mask Refinement

**arXiv ID:** 2607.01708 | [PDF](https://arxiv.org/pdf/2607.01708v1)

**作者:** Hyunjoon Park `[一作]` (Hanyang University), Donghyeon Cho `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过多模态信息（语义、深度、边缘）对 SAM 生成的过分割 2D 片段进行精细合并，并将一致的实例 ID 通过多视角一致性投票上升到 3D 高斯原语，实现稳定的 3D 场景分割与编辑；

**💡 创新点**

提出 Multi-Cue Guided Mask Merging（MCM）与 Cross-View Mask Matching 的两阶段策略，首次将语义、几何与结构三重信号联合用于 2D 片段合并，并通过投票实现全局 ID 一致性，显著降低过分割并提升边界精度；

**🔧 技术方法**

利用 Segment Anything Model (SAM) 生成初始掩码；采集 DINOv2 语义嵌入、DepthAnythingV2 单目深度与 LoG 边缘特征；构建 3D Gaussian Splatting (3DGS) 体素与特征场；执行多视角一致性投票、联合优化（渲染、语义一致性与 3D 规则）等；

**📊 数据集**

在 LERF、Replica 与多场景的真实环境数据集上进行评估；

**📈 对比分析**

相较于 GaussianGrouping、SAGA、InstanceGaussian、Feature3DGS、GARField、CF3 等基线，mIoU 提升至 0.728、mBIoU 0.677、Mask 数量降至 67，渲染质量（PSNR 28.6、SSIM 0.91）保持与基线相当，展示了更高的分割准确度与更紧凑的实例表示；

**⚠️ 局限性**

依赖 SAM 的过分割假设，若 SAM 出现欠分割或缺失边界则无法恢复；在强遮挡或显著外观变化下，单视角合并可能产生局部不一致，需进一步提升跨视角鲁棒性。

---

## 222. LASER: A Corrective Lens for LVLMs via Visual Attention Preservation and Sink Suppression

**arXiv ID:** 2607.01707 | [PDF](https://arxiv.org/pdf/2607.01707v1)

**作者:** Bowen Yuan `[一作]` (University of Queensland), Zi Huang `[通讯]` (University of Queensland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大视觉-语言模型（LVLM）在长时推理中出现的视觉遗忘问题，本文提出了 LASER 框架，利用后训练的奖励机制在推理过程中同时调节视觉注意力的时间轨迹和分布，保持对有意义视觉信息的持续关注并抑制对无关视觉“sink” token 的过度集中。

**💡 创新点**

创新点包括：① 系统性地从早期视觉注意力衰退和视觉 sink token 集中两个维度诊断视觉遗忘；② 在 GRPO 强化学习框架下引入视觉保留奖励（R_vis）与 sink 抑制奖励（R_supp）双重奖励，兼顾注意力时序与空间分布；③ 通过隐藏状态激活模式自动识别视觉 sink token，避免手工规则。

**🔧 技术方法**

技术手段主要为：Group Relative Policy Optimization（GRPO）强化学习、基于注意力权重的奖励设计、隐藏层激活分析、视觉注意力比例（VAP）跟踪与可视化，以及与 Qwen‑2.5‑VL‑7B‑Instruct 预训练模型的后训练集成。

**📊 数据集**

实验使用了 8 个多模态推理基准：数学推理（MathVista、MathVision、MathVerse、WeMath）、通用推理（MMMU、MMStar、LogicVista）以及视觉感知（HallusionBench），并在训练阶段采用约 45K 的 RL 样本。

**📈 对比分析**

与多种基线（包括 Qwen‑2.5‑VL‑7B、VisionR1、R1‑Onevision、OpenVLThinker、Reflection‑V、VAPO‑Thinker 以及专有模型 GPT‑5、Gemini‑2.5‑Pro）对比，LASER 在多数任务上实现了显著提升：在 MMStar 上刷新记录至 64.1，在 HallusionBench 上提升 3.3%，在 MathVision 上提升 12.7%，整体平均得分提升至 58.0，证明了双重奖励在抑制视觉遗忘方面的有效性。

**⚠️ 局限性**

局限性包括：需要额外的 RL 训练与奖励调参，计算成本相对较高；sink token 的识别依赖于隐藏激活统计，可能对不同数据集或模型结构敏感；在极长推理长度或更复杂多模态场景下的鲁棒性尚未充分验证；以及对专有大模型的兼容性和泛化能力仍有待进一步研究。

---

## 223. Arachne: Orchestrating Cascades for Efficient Text-to-Video Model Training

**arXiv ID:** 2607.01701 | [PDF](https://arxiv.org/pdf/2607.01701v1)

**作者:** Peng Yu `[一作]` (China Telecom), Qizhen Weng `[通讯]` (China Telecom)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Arachne，一种通过细粒度时空编排实现高效文本到视频（T2V）模型训练的框架

**💡 创新点**

创新点在于将训练拆解为最小可调度单元（cascades），通过混合整数规划与遗传算法进行时序调度，并结合拓扑感知的资源映射与运行时执行器，实现动态并行度和GPU分配的自适应优化，显著缓解了因视频序列长度不均导致的工作负载失衡

**🔧 技术方法**

核心技术包括：cascade级调度规划（MILP/GA）、VAE与DiT的精细成本模型、拓扑感知的内/跨节点资源放置策略、异构梯度聚合与自适应数据传递、以及整体的时空编排引擎

**📊 数据集**

在Open‑Sora等进阶训练课程下使用的真实大规模视频数据集：Stage1 WebVid（约7M 360p）、Stage2 Koala（约20M 720p）和Stage3内部Lynx（约10M 1080p），以及Wan2.1（1.3B）、CogVideoX（5B）和HunyuanVideo（13B）三大T2V模型

**📈 对比分析**

与Megatron‑LM、DeepSpeed、FlexSP三大主流训练框架对比，Arachne在迭代时间上分别提升最高65%、59%和35%；GPU空闲比例从8%降至2%；在模型规模、工作负载异质性以及集群规模提升时均表现出正向的扩展趋势

**⚠️ 局限性**

主要局限包括：对成本模型的准确性高度依赖；动态调度与资源映射引入额外的系统复杂度；目前评估仅覆盖特定T2V模型与数据集，尚未验证在更广泛模型和异构硬件上的泛化能力

---

## 224. Frequency Shift Physics-Informed Extreme Learning Machine for Solving High-Frequency Partial Differential Equations

**arXiv ID:** 2607.01694 | [PDF](https://arxiv.org/pdf/2607.01694v1)

**作者:** Xiong Xiong `[一作]` (Northwestern Polytechnical University), Zichen Deng `[通讯]` (Northwestern Polytechnical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种频率偏移物理信息极限学习机（FS-PIELM），通过平移隐藏层权重均值来精准控制高频谱，解决PDE高频解的光谱偏差问题。

**💡 创新点**

创新在于采用均值平移而非权重缩放来控制权重分布，保持方差有界，理论证明频率方差保持有限，同时提出两种变体FS-PIELM-L（单个神经元频率）和FS-PIELM-G（分组频率）。

**🔧 技术方法**

使用极限学习机（ELM）架构与物理信息学习框架，结合频率偏移采样、理论频率方差分析和单线性求解的技术。

**📊 数据集**

使用七个合成基准PDE（Helmholtz、Wave、Poisson、Klein‑Gordon、Heat、Advection‑Diffusion、Panda形域Helmholtz），无公开数据集，全部基于解析解与手工生成的源项。

**📈 对比分析**

与Tanh‑PIELM、SIREN‑PIELM、GFF‑PIELM在相同网络设置、随机种子下比较，FS‑PIELM‑L在6/7个案例中获得最低误差，提升幅度从约20×到约3.7×10⁴，平均误差显著降低且标准差更小。

**⚠️ 局限性**

局限性包括需人工指定频率范围（μ_min、μ_max），尚未针对非线性PDE或需要迭代求解的情况进行验证，也缺乏严格的收敛理论与自适应频率选择策略。

---

## 225. Epistemic Goggles: A Pretrained Module that Induces an Epistemic Frame via Gradient Editing

**arXiv ID:** 2607.01690 | [PDF](https://arxiv.org/pdf/2607.01690v1)

**作者:** Joshua Penman `[一作]` `[通讯]` (Independent Researcher), Joshua Penman (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Goggles，一种在微调梯度上进行编辑的模块，用来在训练期间为模型注入“认知立场”（例如把文本标记为虚构）

**💡 创新点**

创新点在于：① 用梯度编辑而非文本注释直接将认知框架注入模型；② 仅需训练一次、可冻结后应用于未见过的文档和不同框架；③ 有效克服 Negation Neglect 现象

**🔧 技术方法**

技术手段包括：LoRA 低秩适配器、每个 LoRA 的梯度编辑小网络（SwiGLU 头+外积残差）、外循环元训练（逆 KL 损失对比教师滚动输出）、截断 BPTT、谱正则化

**📊 数据集**

数据集：① 约 4 万份合成文档（4 千主题，每主题 10 题）；② 先前 Negation Neglect 论文中的约 10 千篇正向文档；③ 生成的教师滚动输出、留出小说家与混合实体集合；④ 评测使用 TruthfulQA 与 GPQA

**📈 对比分析**

对比方法：普通 SFT、带否定前缀的 SFT、In‑Context Distillation (ICD) 与基线模型；结果显示 Goggles 在长周期训练中将虚构识别率提升至约 91%（对比 ~9%/7%），同时保持 TruthfulQA 与 GPQA 评分不下降，且在未见主题上同样表现良好

**⚠️ 局限性**

局限性：仅在 8B LoRA 训练上验证；需要一次完整外循环为每个框架/模型配置训练；依赖教师滚动输出与特定学习率；未测试在更大模型或全模型训练中的可扩展性；对对齐效果的实际影响仍待进一步验证

---

## 226. Imagining the Sense of Touch: Touch-Informed Manipulation via Imagined Tactile Representations

**arXiv ID:** 2607.01684 | [PDF](https://arxiv.org/pdf/2607.01684v1)

**作者:** Zhiyuan Zhang `[一作]` (Purdue University), Yu She `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TacImag 框架，利用视觉和本体信息在部署时生成触觉感知，从而实现无需触觉传感器的触觉驱动操控。

**💡 创新点**

创新点在于：①只在训练阶段使用真实触觉，部署阶段通过扩散模型“想象”触觉；②展示触觉想象是接触感知监督而非信息补全，能够将视觉中的微妙交互信号转换为易被策略利用的触觉表示。

**🔧 技术方法**

采用条件去噪扩散概率模型（DDPM）生成 TacRGB 触觉图像和 TacFF 力场，然后用 Diffusion Policy 训练触觉条件下的操控策略。

**📊 数据集**

使用 ManiFeel 仿真基准（USB 插拔、功率插头、peg‑in‑hole、齿轮组装、电灯泡安装、球体排序）以及对应的实机抓取数据，并通过 TacSL 提供的 GelSight 触觉图像与力场作为训练数据。

**📈 对比分析**

与仅视觉+本体、以及使用真实 TacRGB/TacFF 触觉的基线比较；模拟实验中想象 TacFF 的平均成功率提升约 3.9%，与真实 TacFF 仅差 0.7%；实机实验中想象 TacFF 对接触敏感任务提升 44.4% 成功率，想象 TacRGB 对纹理分类任务提升约 40%，整体表现接近真实触觉条件。

**⚠️ 局限性**

局限性：触觉想象受限于视觉中可观测的交互信息，无法补偿视觉完全缺失的细节；对纹理任务高度依赖视角；生成过程相对耗时，需要大量标注的 visuotactile 对应数据。

---

## 227. UniWind: Toward Unified Day-Ahead Wind Power Forecasting via Physics-Informed State Routing

**arXiv ID:** 2607.01670 | [PDF](https://arxiv.org/pdf/2607.01670v1)

**作者:** Ronghui Xu `[一作]` (East China Normal University), Yong Li `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并提出 UniWind 模型，实现基于物理信息与状态路由的日常风电预测。

**💡 创新点**

创新点在于将物理先验与站点校准相结合，并引入物理上限约束；通过潜在状态编码和状态感知校正，将物理可用功率与操作状态分离，显著提升对异常状态的鲁棒性。

**🔧 技术方法**

采用物理先验估计（站点条件单调 warp + 共享物理功率曲线）、物理上限约束；利用历史差异编码的潜在状态编码器、注意力检索未来状态、状态感知功率校正器以及监督状态路由等技术。

**📊 数据集**

使用20+个真实风场数据集，涵盖英国（Penmanshiel、Kelmarsh）与中国（山东、山西、安徽等）多省风场；NWP 数据来自 ECMWF 与 GFS。

**📈 对比分析**

与物理模型、统计树模型、时间序列预测模型、可再生能源预测模型及基础模型在全射与零射两种设置下对比；UniWind 在 MAE/RMSE 上普遍优于基线，尤其在高风速异常状态下表现最为突出。

**⚠️ 局限性**

主要限制是对 NWP 质量高度依赖，未充分利用额外观测数据，且在极端天气场景下仍需进一步提升鲁棒性。

---

## 228. Diverse Evidence, Better Forecasts: Multi-Agent Deliberation Under Information Asymmetry

**arXiv ID:** 2607.01661 | [PDF](https://arxiv.org/pdf/2607.01661v1)

**作者:** Yuante Li `[一作]` (Carnegie Mellon University), Yaxin Zhou `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种多智能体预测框架InfoDelphi，通过将证据拆分为公共和私有子集实现信息不对称，提升协同推理效果。

**💡 创新点**

首创将信息不对称作为多智能体推理的核心设计原则，理论证明公私证据划分可降低错误相关性，并通过合理证据分配和理据共享实现有效集体决策。

**🔧 技术方法**

采用BM25相关性排序进行证据路由、基于LLM的理据生成与多轮辩论、信心加权（logit空间）聚合以及Rationale共享的交互式推理。

**📊 数据集**

构建PolyGym预测基准，由375个Polymarket二分类预测市场的预检索证据组成，消除检索变化对实验的影响。

**📈 对比分析**

与多种单智能体与同质输入的多智能体基线对比，InfoDelphi在PolyGym上Brier分数降低12–18%，准确率提升4–8个百分点，表现最优。

**⚠️ 局限性**

限制在于理论假设理想化、检索与推理耦合、实验规模受API成本限制，且在真实动态检索环境下效果尚未验证。

---

## 229. One Demonstration Is Enough for Real-World Robotic Reinforcement Learning

**arXiv ID:** 2607.01651 | [PDF](https://arxiv.org/pdf/2607.01651v1)

**作者:** Yuwan Liu `[一作]` (National Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institution of Automation, Chinese Academy of Sciences), Ceyao Zhang `[通讯]` (Institute for Artificial Intelligence, Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 AutoSERL 框架，利用单个演示轨迹实现真实世界机器人强化学习的自动干预，从而消除持续人类干预的需求。

**💡 创新点**

创新点在于三种自动干预机制：滑动窗口干预、故障恢复点回放与干预终止判定，三者结合可在单示例下匹配甚至超越传统 HIL‑SERL。

**🔧 技术方法**

使用技术包括基于 SERL 的强化学习、演示轨迹回放、几何直觉的滑动窗口定位、运动规划恢复、稀疏奖励设计以及三种自动干预策略。

**📊 数据集**

实验数据集为六种接触密集操控任务（插入、悬挂、铰链）在两个机器人平台（Franka、UR5+Inspire）上收集的单条示例轨迹。

**📈 对比分析**

与 SERL、HIL‑SERL、行为克隆（BC）和一射印象学习基线 MILES 进行对比；AutoSERL 在所有任务上取得最高成功率（插入任务 100%）、训练效率优于 SERL，训练时间与 HIL‑SERL 相当或更少，并在不同随机种子和位置扰动下表现出良好鲁棒性。

**⚠️ 局限性**

局限性包括：恢复机制只能针对单条示例中的失败模式，无法应对多样化失败；仅适用于 6D 末端位姿动作空间；若任务失败模式超出示例分布，恢复效果会显著下降，需要更多示例或更强的恢复策略。

---

## 230. DRL-CLBA: A Clean Label Backdoor Attack for Speech Classification via DDPG Reinforcement Learning

**arXiv ID:** 2607.01729 | [PDF](https://arxiv.org/pdf/2607.01729v1)

**作者:** Yueming Huang `[一作]` (Xiangtan University), Weiping Wen `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于深度确定性策略梯度（DDPG）与深度音频隐写的无标签（clean‑label）样本特定后门攻击 DRL‑CLBA，能够在不改变标签的情况下将目标类样本迁移到携带触发器的源类特征空间，从而在推理时诱导模型误分类。

**💡 创新点**

创新点包括①将音频隐写技术与后门触发器相结合，生成样本特定、不可被察觉的触发器；②将清标后门攻击建模为马尔可夫决策过程（MDP），利用强化学习（DDPG）逐步优化样本，实现长程特征碰撞；③不依赖目标模型的完整梯度信息，适用于白盒和一定程度的黑盒。

**🔧 技术方法**

核心技术：深度确定性策略梯度（DDPG）强化学习框架、音频深度隐写生成器（encoder‑decoder）、特征碰撞损失与多目标奖励函数（距离、扰动约束、语义一致性）。

**📊 数据集**

实验数据集：关键词识别（SCD、AudioMNIST、LibriKWS‑20）、说话人识别（AISHELL3‑50、VoxCeleb1‑50）、情感识别（ESD‑CN、ESD‑EN）。模型包括 ERes2Net、KWS‑ViT、EAT‑S、CAM++。

**📈 对比分析**

与五种基线（Ultra、OneSpec、CBA、CSSBA、TUAPBA）对比，DRL‑CLBA 在所有任务和模型上均获得最高的攻击成功率（ASR）且保持较高的正常准确率（BA），在 KWS 上平均 ASR 超过 88%，在 SV、SER 上亦显著优于基线；在不同触发器、目标标签、奖励项和决策步数上实验验证其鲁棒性和优越性。

**⚠️ 局限性**

局限性：1）对目标模型架构仍需一定程度的白盒信息，黑盒跨模型迁移虽可行但成功率下降；2）训练需要较多迭代（T>1），时间成本高；3）在高剪枝率或高频率细化等极端防御下攻击效果会衰减；4）触发器生成与特征碰撞依赖于目标特征层次，深层特征时成功率显著降低。

---

## 231. COMFYCLAW: Self-Evolving Skill Harnesses for Image Generation Workflows

**arXiv ID:** 2607.01709 | [PDF](https://arxiv.org/pdf/2607.01709v1)

**作者:** Zongxia Li `[一作]` (University of Maryland), Lichao Sun `[通讯]` (Lehigh University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自演化的工作流控制框架，能够在ComfyUI图形化图像生成中自动修复失败并从历史执行中提炼可复用的工作流技能。

**💡 创新点**

创新点在于将Verifier驱动的闭环修复与技能演化机制结合，形成“技能库持续更新”的闭环；并引入阶段化图编辑与逐步暴露技能的策略，显著提升工作流可复用性与鲁棒性。

**🔧 技术方法**

核心技术包括：① 以类型化图编辑为基础的工作流构建器；② 视觉语言模型（VLM）作为Verifier，用以生成需求级与区域级的错误反馈；③ 基于聚类与验证的技能演化循环（包括技能突变、持久化验证与版本化管理）；④ 通过LLM与工具接口实现代理执行。

**📊 数据集**

在四个文本生成基准上评估：GenEval2、DPG-Bench、OneIG-EN 与 OneIG-ZH，使用两种ComfyUI后端模型（z-image-turbo 与 LongCat-Image）。

**📈 对比分析**

与仅使用初始工具与技能、仅使用Verifier的基线相比，所提框架在所有四个基准上取得最高平均得分；在人工评估中，其生成图像获得的Likert分数平均比对手高约0.8分；在技能使用率方面，演化后的技能约占总调用量的50%。

**⚠️ 局限性**

局限性包括：仅针对图像生成工作流；视频生成所需的更复杂节点控制与时序一致性未覆盖；技能检索与上下文窗口管理仍有改进空间；在高并发与资源受限环境下的实时性能未充分验证。

---

## 232. A Mathematical Introduction to Diffusion Models

**arXiv ID:** 2607.01693 | [PDF](https://arxiv.org/pdf/2607.01693v1)

**作者:** Jianfeng Lu `[一作]` `[通讯]` (Duke University), Jianfeng Lu (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文以教学笔记的形式，对从经典马尔科夫链蒙特卡罗到现代扩散模型的采样理论进行系统梳理，涵盖了 Langevin 动力学、连续时间扩散、反向 SDE、概率流 ODE、DDPM 与 DDIM 离散化、以及在推理阶段的引导和强化学习等技术；同时给出了误差分解、KL 消散、Lipschitz 控制、以及第一阶拒绝采样等高精度采样方法的理论分析。

**💡 创新点**

创新点主要在于：① 将扩散模型的连续时间过程与离散 DDPM 采样的误差分解统一到 KL、Wasserstein 和总变差三种距离；② 用信息论的 Girsanov 变换与数据处理不等式，给出精细的路径空间 KL 递推公式；③ 引入“协方差预算”概念，用后验协方差量化 Hessian 控制，得到几乎线性（≈d）对维数的采样复杂度；④ 将第一阶拒绝采样与扩散模型的评分信息结合，提供仅用评分（无密度比）即可实现高精度采样的理论。

**🔧 技术方法**

技术手段包括：随机微积分（Ito、Girsanov）、Fokker–Planck 方程、信息量（KL、Fisher 信息）与 log‑Sobolev/交通不等式的组合；误差分析利用梯度 Lipschitz、Hessian 控制、随机局部化（stochastic localization）与 Polchinski 流；离散化采用 Euler–Maruyama、DDPM、DDIM、第一阶拒绝采样等；推理时的控制则用路径空间控制、奖励倾斜、强化学习等。

**📊 数据集**

本文本身不针对具体实验；若要在实验上验证，需要使用标准生成模型数据集（如 MNIST、CIFAR‑10、CelebA 等）来训练评分网络并进行采样评估。

**📈 对比分析**

与传统 MCMC、变分自编码器、流模型等方法相比，本文提出的扩散采样在理论上能够在 log‑Sobolev/凸性条件下给出 KL 收敛速度；在高维下，通过协方差预算实现 O(d log²(T/δ)/ε²) 的采样步数；通过第一阶拒绝采样可实现仅 O(polylog(1/ε)) 步的高精度采样。实验上，这些方法在生成质量和样本多样性上往往优于传统的 MCMC 与流模型。

**⚠️ 局限性**

限制主要包括：① 需要对目标分布满足 log‑Sobolev 或交通不等式等结构；② 对评分网络的误差要求严格，训练过程需要大量高质量样本；③ 第一阶拒绝采样在实现上对随机数生成和计算量有额外需求；④ 对多峰、非平滑或奇异目标分布的理论保证仍不完整。

---

## 233. From Answer Generators to Reasoning Facilitators: Designing AI Tutors for Mathematical Reasoning in High-Stakes Environments

**arXiv ID:** 2607.01692 | [PDF](https://arxiv.org/pdf/2607.01692v1)

**作者:** Yuming Feng `[一作]` (Stanford University), Erica Zhao `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究设计并部署了AITutor——一款基于大语言模型的数学辅导系统，采用层级化步骤展示、同步几何图形、可修复步骤、校准信任标签以及自动错题回顾等交互，将AI解答转化为可检查、可修复、可验证、可延迟学习的体验。

**💡 创新点**

创新点在于提出“Reasoning‑Centered Product Loop”，把LLM输出从单纯答案转化为可检查的步骤、可局部修复的提示、可校准的课程契合度、可延迟检索的错题集；通过层级UI、步进视觉同步与上下文后续提问按钮，降低认知负荷，支持高压考研环境下的数学推理学习。

**🔧 技术方法**

技术实现上使用了前端React Native/Expo、后端Python/FastAPI、调用大型语言模型（如GPT‑4/ChatGPT），结合知识标签与课程映射的检索模块，生成逐步解释与动态几何图；在交互层面实现可展开层级、步骤连线视觉高亮、上下文修复按钮、自动错题分类与错书生成。

**📊 数据集**

数据集主要来自12名中国初中生的作业照片与课堂练习，构成题目识别与答案验证数据；后端还利用校内数字化教材与题库进行检索与知识点标签化。

**📈 对比分析**

通过12天实地部署，对比前置产品（Qianwen、Xiaoyuan、iFlytek）与本系统。完成率为56.4%，平均答题时延32.0秒（p90 68.5秒）；跟进与转换卡的打开率分别为10.0%与8.6%，体现功能可用性。相较于传统答题工具，AITutor在步骤可见性、视觉同步与修复便利性上表现更佳，用户对回答先核查的行为转化为认知调节。

**⚠️ 局限性**

局限性包括：部署时间仅12天，样本量有限，缺乏长期学习成效评估；系统在5月20–21日的可靠性故障导致完成率偏低；未能系统性分离技术故障与用户流失；缺少标准化前后测或对照实验来验证推理能力提升。

---

## 234. Beyond Gradient-Based Attacks: Adversarial Robustness and Explainability Stability in Cybersecurity Classifiers

**arXiv ID:** 2607.01679 | [PDF](https://arxiv.org/pdf/2607.01679v1)

**作者:** Mona Rajhans `[一作]` (Palo Alto Networks), Vishal Khawarey `[通讯]` (Quicken Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对网络安全领域的MLP、随机森林和XGBoost等模型，在四个表格化数据集上进行对抗攻击评估，并提出了新的解释稳定性指标（ESI），展示了预测鲁棒性与解释稳定性之间的分离；

**💡 创新点**

首次将ESI与RI并列度量，揭示ZOO梯度失效导致XGBoost鲁棒性被高估的现象，证明PGD在z-score归一化数据上因步长问题反而弱于FGSM，并对黑盒攻击方法进行了系统比较，提供了对树模型的对抗训练实验；

**🔧 技术方法**

使用SHAP（TreeSHAP与KernelSHAP）、FGSM、PGD、ZOO、Square Attack、HopSkipJump等攻击技术，结合MLP、随机森林、XGBoost三类模型，辅以z-score归一化、特征选择和对抗训练；

**📊 数据集**

Phishing URL、UNSW-NB15、NF-ToN-IoT、HIKARI-2021四个网络安全数据集；

**📈 对比分析**

对每种攻击在相同ε_max（0.30）下计算RI与ESI；ZOO在XGBoost上得到RI≈0.98但ESI≈0.06，Square Attack则将RI降至≈0.36/0.30；PGD在步长0.01时比FGSM弱；攻击排名基于梯度依赖与查询效率，Square Attack通常最强；对抗训练后XGBoost Phishing的RI提升至≈0.75，说明训练有效；

**⚠️ 局限性**

仅评估四个数据集且未考虑特征合法性约束，CPU限制导致查询预算受限，ZOO对XGBoost鲁棒性高估可能误导评估，ESI仅基于ZOO或Square攻击，缺少对其他XAI方法或更大预算下的验证，且统计显著性与置信区间缺失；

---

## 235. Separating Expert Retention from Autonomous Source Inference in Raw-ECG-Replay-Free Continual ECG Deployment

**arXiv ID:** 2607.01674 | [PDF](https://arxiv.org/pdf/2607.01674v1)

**作者:** Yufan Lu `[一作]` (Xidian University), Shenda Hong `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

构建了一个基于冻结 ECGFounder 特征的专家银行，在多源 ECG 持续学习中实现源特定专家的保持与自主源推断。

**💡 创新点**

将专家保持与源推断拆分为两独立问题，并在无原始 ECG 重放的前提下，仅存储特征实现自适应路由，提出验证校准的 top‑2 边界融合方法。

**🔧 技术方法**

使用冻结 ECGFounder 预训练模型、平衡 Softmax 线性专家、MLP 路由器、top‑2 校准融合、kNN、LDA 等技术。

**📊 数据集**

实验基于四个公开 ECG 数据集：CPSC、PTB‑XL、Georgia、Chapman‑Shaoxing。

**📈 对比分析**

与池化头、kNN、LDA、线性路由等基线以及离线完整训练的匹配专家对比；源知情专家得到 0.7915 Macro‑F1，自治 MLP top‑2 得 0.7782 Macro‑F1，差距约 0.013；相较共享参数连续学习基线性能更好。

**⚠️ 局限性**

局限性：源推断误差仍未完全消除，仍落后于 oracle；仅基于记录级别无患者级分割；需要存储特征并未实现完全无存储；仅针对二分类单导 ECG，未涵盖多标签或多导情况。

---

## 236. AgenticDataBench: A Comprehensive Benchmark for Data Agents

**arXiv ID:** 2607.01647 | [PDF](https://arxiv.org/pdf/2607.01647v1)

**作者:** Zhaoyan Sun `[一作]` (Tsinghua University), Huaiyu Ruan `[通讯]` (Ant Digital Technologies, Ant Group)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了AgenticDataBench，涵盖15个领域的344个真实与生成的任务，并对任务进行细粒度的433个数据科学技能标注；

**💡 创新点**

创新点在于：①利用LLM进行层级化技能抽取与聚类，形成可解释的技能树；②设计任务选择与生成模块，保证技能覆盖与任务多样性；③在评估中引入技能级别分数，实现对代理行为的细粒度诊断；

**🔧 技术方法**

技术主要包括：LLM（Qwen3.5、Kimi‑K2.5、Claude Sonnet 4.6）驱动的代理；词向量嵌入+UMAP+GMM+LLM细化的技能聚类；任务生成流水线（数据剖析、技能图采样、LLM编写工作流与任务描述）；评估框架（五种评分模式、Pass@1、token成本统计）；

**📊 数据集**

使用的数据库来自97个真实数据集（Kaggle、UCI、Mendeley、学术与政府公开数据）和5个Ant Group B2B业务案例，共计27.3GB；并在此基础上生成242个任务，补充15个领域；

**📈 对比分析**

对四种代理实现（DA‑Agent、Smolagents、Claude Code、CodeX）分别搭配三大LLM进行实验；评估指标为Pass@1、技能级别分数、token消耗与成本；结果显示CodeX(Kimi‑K2.5)总体最优，但不同代理在不同领域表现差异，成本-性能权衡也因LLM-代理匹配而异；

**⚠️ 局限性**

局限性包括：①技能抽取仍可能遗漏细粒度或新兴技能；②代理对LLM与工具的耦合不够紧密，导致高失败率；③对跨表异构数据处理能力不足；④实验规模受15个领域和有限任务数限制；⑤闭源LLM成本高，难以广泛复现。

---

## 237. Archer: Towards Agentic Review for Compiler Optimizations

**arXiv ID:** 2607.01808 | [PDF](https://arxiv.org/pdf/2607.01808v1)

**作者:** Yunbo Ni `[一作]` (Chinese University of Hong Kong), Shaohua Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Archer，一个面向 LLVM 编译器优化补丁的自动化代码审查工具，利用动态义务构造与确定性验证守卫实现对 PR 的语义错误检测。

**💡 创新点**

创新点包括：① 将历史修复动态转化为可复用的义务，抽象语义知识；② 通过确定性验证守卫把 LLM 的自然语言分析转化为可执行、可验证的证据；③ 将 LLM 与专用验证工具深度集成，形成端到端的自动审查流程。

**🔧 技术方法**

使用技术包括：大语言模型代理（Gemini‑3.1‑Pro、DeepSeek‑V3.2、Qwen3.5‑Plus 等），工具调用接口，LLVM 专用验证器 ProofCheck、TestCheck（配合 LLUBI），动态义务构造算法，以及 mini‑SWE‑agent 代理框架。

**📊 数据集**

实验数据集包含：398 条 LLVM 中间端优化 PR（70 开放、328 关闭），以及 47 条已知误编译回归 PR；义务构造基于 317 条历史修复 PR，生成 188 条验证过的义务。

**📈 对比分析**

与通用 LLM、开源代码代理、传统定向模糊测试（Optimuzz）及商业 AI 审查工具（Copilot、CodeRabbit 等）比较。回归集上 Archer 的缺陷发现率最高；在实时 PR 上发现 51 个语义错误，误报率仅 6%；平均审查成本约 2.5 美元、耗时 877 秒、Token 使用约 5M。

**⚠️ 局限性**

局限性：召回率仍有不足，需持续完善义务与验证策略；代理探索效率受限，可能重复无效工具调用；依赖现有验证工具与 Oracle，难以覆盖所有语义细节；误报仍存在（3 起）且对部分复杂或未见模式的检测有限。

---

## 238. PARTREP: Learning What to Repeat for Decoder-only LLMs

**arXiv ID:** 2607.01792 | [PDF](https://arxiv.org/pdf/2607.01792v1)

**作者:** Andikawati P Widjaja `[一作]` (Bandung Institute of Technology), Jaeho Lee `[通讯]` (Pohang University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对解码器仅LLM因因果注意力导致的前后信息不均衡问题，提出一种选择性重复关键token的方法（Partial Repetition），在不增加大规模参数或架构改动的前提下通过在原始prompt后追加高信息量token来提升推理准确率。

**💡 创新点**

创新点在于：①用token级负对数似然(NLL)作为重要性度量；②训练轻量级门控网络仅基于早期层隐藏状态预测高NLL token，避免完整前向推理；③结合token窗口扩展提升子词分割和局部结构的保留；④实现了与全重复几乎同等的性能，同时显著降低KV缓存和prefill FLOPs。

**🔧 技术方法**

核心技术包括：负对数似然（NLL）评估、基于隐藏状态的两层MLP+注意力门控网络、token窗口扩展、以及在中间层中断前向推理实现早退出。

**📊 数据集**

评估使用了8个基准（ARC-Challenge、OpenBookQA、MMLU、MedQA、SciQ、MMLU-Pro、GSM8K、RULER），并在三类模型（Qwen2.5-3B、Llama3.2-3B、Gemma4-E4B）上进行实验。

**📈 对比分析**

与无重复、摘要补充、全重复、以及多种缓存清理方法相比，Partial Repetition在大多数任务上均实现了与全重复相当或更优的准确率，同时仅使用约59% KV缓存和79% prefill FLOPs，显著降低计算与内存开销。

**⚠️ 局限性**

主要限制在于：①需要针对每个目标LLM进行离线门控网络训练，缺乏零样本或跨模型迁移能力；②方法基于文本token的假设，难以直接推广到图像或多模态模型。

---

## 239. Subliminal Clocks: Latent Time Modelling in Diffusion Language Models

**arXiv ID:** 2607.01774 | [PDF](https://arxiv.org/pdf/2607.01774v1)

**作者:** Maximo Rulli `[一作]` (Sapienza University of Rome), Alessio Devoto `[通讯]` (Independent researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统探索了扩散语言模型（DLM）在残差流中是否隐含可解码的去噪进度（τ）信号，并证明该信号可以被线性探针提取、在低维子空间进行干预，并通过熵、置信度及KL散度的变化验证其因果影响。

**💡 创新点**

创新点在于首次发现并量化DLM内部存在可解码且可操控的去噪进度表示，并揭示其在层间的几何结构（近似二维抛物线轨迹）以及在层内不同模块（自注意力与MLP）对该信号的对齐与反向对齐。

**🔧 技术方法**

使用的技术包括：针对每层残差流训练MLP回归探针；计算并分析均值激活向量并对其进行PCA降维；对均值向量差异进行线性干预；在推理过程中评估干预对输出熵、最大置信度及KL散度的影响；以及通过层间余弦相似度分析信号传递。

**📊 数据集**

实验数据基于公开训练的两款大型DLM：LLaDA‑1.5 与 LLaMA‑2‑70B（70B参数版），使用其标准去噪采样过程产生的序列作为评估对象。

**📈 对比分析**

与随机方向干预对比实验显示，针对τ信号的干预在熵下降、置信度提升与KL散度增大方面显著优于随机干预；在层深靠后的层（如第29层）更能显现出可预测的因果效应，说明该信号在模型推理中起核心作用。

**⚠️ 局限性**

局限性包括：仅验证两款模型，缺乏对其他DLM架构的泛化验证；未揭示构建或更新τ表示的具体计算回路；未评估此干预对推理效率或生成质量的潜在改进；以及未探究单词级别的影响。

---

## 240. SimWorlds: A Multi-Agent System for Dynamic 3D Scene Creation

**arXiv ID:** 2607.01766 | [PDF](https://arxiv.org/pdf/2607.01766v1)

**作者:** Chunjiang Liu `[一作]` (Carnegie Mellon University), László A. Jeni `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个多智能体框架，能够将自然语言提示转换为可编辑的4D Blender项目，涵盖几何、材质、灯光、相机、动画与物理求解器；

**💡 创新点**

创新点在于：①通过分阶段构建管道和层级场景协议实现对Blender运行时状态的确定性验证；②引入机制正确性评估，而非仅视觉评估；③提供50场包含多种物理求解器的基准与评测流程；

**🔧 技术方法**

使用技术包括：LLM规划者、编码器与审阅者协同工作；Blender Python API与实时状态读取工具；确定性验证器与多层协议；视觉评估通过GPT‑5.5的VLM判定；基准评测采用无图形Blender运行时审计；

**📊 数据集**

数据集：5种物理求解器（cloth、fluid、rigid body、particle、soft body）各9条提示，5条静态场景，共50条；BlenderBench 27个编辑任务用于编辑模式评测；

**📈 对比分析**

比较方法：与VIGA对比，使用机制通过率（MPR）、结构通过率（SPR）和VLM分数；在所有动态类别中MPR提升约0.2，视觉分数相近；在编辑任务上，综合得分比VIGA提升约20%；

**⚠️ 局限性**

局限性：仍依赖LLM/VLM做感知判断；只能处理文本输入，缺乏图像引导；对复杂跨求解器交互的自动化仍不完善。

---

## 241. Refploit: Facilitating Exploit Construction via Code-Agent Trajectory Repair

**arXiv ID:** 2607.01760 | [PDF](https://arxiv.org/pdf/2607.01760v1)

**作者:** Zirui Chen `[一作]` (Zhejiang University), Xiaohu Yang `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Refploit框架，利用LLM对代码代理生成的失效轨迹进行差分执行、进度评估和约束恢复，从而自动构造可运行的Java漏洞利用。

**💡 创新点**

创新点在于：①将失败轨迹中的有价值子任务提取并保留；②通过差分执行验证漏洞行为；③基于环境、Harness、复现三维度进行进度评估并生成保留/修复约束；④以约束驱动的恢复循环显著提升漏洞利用生成效果。

**🔧 技术方法**

技术包括：LLM驱动的代码代理（mini‑swe‑agent/ OpenHands）、ReAct循环、差分执行、进度评估模块、约束生成与恢复子代理、Python实现。

**📊 数据集**

使用了三大开源Java漏洞数据集（CWE‑Bench‑Java、VISION 等），共 172 条公开漏洞利用链接、143 CVE、涵盖 53 个 CWE。

**📈 对比分析**

在 DeepSeek‑V4‑Flash 上与现有漏洞生成器、Codex+GPT‑5.4 以及基础代理进行对比，成功率 80.2%，比最佳基线提升 27.9%–52.3%；在 Qwen3.5‑27B 同样提升；迁移至 OpenHands 仍提升 74.5%，证明方法具有良好可迁移性。

**⚠️ 局限性**

局限性：① 对需要复杂 gadget 链的漏洞效果有限；② 只处理文本型公开 exploit，图像形式的引用未覆盖；③ 评估覆盖率仅为 61.9% Maven 旧漏洞，未验证授权类、平台级漏洞；④ 依赖差分执行与 LLM 判断，可能出现误判；⑤ 预装工具环境，未评估完整环境搭建能力。

---

## 242. The Turning Point of 3D Plant Phenotyping: 3D Foundation Models Enable Minute-to-Second Cross-Crop Reconstruction and Beyond

**arXiv ID:** 2607.01753 | [PDF](https://arxiv.org/pdf/2607.01753v1)

**作者:** Hanyue Jia `[一作]` (Northwest A&F University), Tingting Wu `[通讯]` (Northwest A&F University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出基于3DFM的全流程低成本跨作物3D植物表型化框架，将COLMAP前端替换为3DFM，实现从手机视频到可测量叶片几何的自动化。

**💡 创新点**

创新点包括：①使用3DFM实现秒级相机与初始几何推断；②结合3D Gaussian Splatting进行几何约束的密集重建；③通过迭代视角合成提升稀视角下的可用性；④2D→3D语义转移实现跨作物分割；⑤端到端的尺度恢复与叶片实例分离。

**🔧 技术方法**

使用视觉几何3DFM（VGGT/π^3）、3D Gaussian Splatting（Mip‑Splatting）、Difix3D+视角合成、SAM语义分割、点云密集优化、尺度恢复与叶片实例分离等技术。

**📊 数据集**

构建了跨作物智能手机视频数据集（26株，包含烟草、玉米、小麦等）并提供叶片实例标注与手工测量。

**📈 对比分析**

与传统COLMAP+SfM+MVS对比，3DFM前端速度从6.5分钟提升至1.6秒，重建质量SSIM/PSNR仅微降；在稀视角下可在≤10视图实现可用重建，叶片面积与倾斜角误差≤2°，相较COLMAP的失败率显著降低。

**⚠️ 局限性**

适用于单株、近距离闭环采集；稠密冠层、动态扰动或极度遮挡仍易失败；尺度恢复需场景中已知参考物；叶片实例分离在复杂重叠情形下仍有误差。

---

## 243. Reformalization of the Jordan Curve Theorem

**arXiv ID:** 2607.01734 | [PDF](https://arxiv.org/pdf/2607.01734v1)

**作者:** Simon Guilloud `[一作]` (École Polytechnique Fédérale de Lausanne), Samuel Chassot `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型和代理式流水线，对Jordan曲线定理（JCT）的正式证明进行跨证明助手（Mizar、HOL Light、Agda）改写，完成三项改写工作；

**💡 创新点**

创新点在于把“改写”视为一种自动化转移，而不是传统的“正确性保证”型翻译，强调保持原证明思路、实现对目标助手生态的良好对齐，并通过LLM辅助的交互式流水线显著降低人工投入；

**🔧 技术方法**

采用LLM（Claude Opus、GPT‑4/5 通过 GitHub Copilot 等）结合脚本、工具调用、状态追踪与手工提示的代理式管道；利用中间数据（依赖图、元数据、标记表）来实现库对齐和证明填充；

**📊 数据集**

使用已有的JCT正式化数据集：Mizar MML 版本、HOL Light 版本（约60k行），以及相应的目标助手库（Mathlib、Agda 标准库）；

**📈 对比分析**

通过对比原始证明与改写结果的字符/行数、编译时间、实现天数和人工时长来评估性能：Mizar→Lean 约10天、20h人工；HOL Light→Lean 约1周、10h人工；HOL Light→Agda 仍在完成中，出现内存/编译瓶颈；整体证明规模得到压缩（大量使用 Mathlib 现成 lemmas），但仍保持数学内容完整；

**⚠️ 局限性**

局限性包括：对高度依赖特定基础的证明（如集合论编码）迁移困难；LLM不保证确定性或完整性；目标助手的编译效率和工具生态直接影响进度；存在内存溢出、工具反馈不及时等运行时问题；需要手动介入来校正库对齐与错误处理。

---

## 244. VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon

**arXiv ID:** 2607.01804 | [PDF](https://arxiv.org/pdf/2607.01804v1)

**作者:** Yi Pan `[一作]` (Zhejiang University), Wenqi Zhang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为VLA-Corrector的轻量级纠正推理框架，旨在解决在动作分块的VLA政策中存在的开放环盲点问题。

**💡 创新点**

VLA-Corrector通过引入潜在空间视觉监控（LVM）和在线梯度引导（OGG）机制，实现了自适应的动作地平线，能够在执行过程中监测并纠正偏差，从而提高成功率和效率。

**🔧 技术方法**

使用了潜在空间视觉监控（LVM）和在线梯度引导（OGG）技术，结合生成模型进行动作预测和纠正。

**📊 数据集**

在MetaWorld和LIBERO数据集上进行了实验，验证了VLA-Corrector的有效性。

**📈 对比分析**

与固定动作地平线的基线方法相比，VLA-Corrector在不同任务难度下均表现出更高的成功率和更低的政策调用频率，成功率提升幅度在4.05%到15.65%之间，且在更长的地平线上效果更显著。

**⚠️ 局限性**

VLA-Corrector的局限性在于其依赖于训练数据的质量和多样性，且在某些情况下可能需要更多的演示数据来提高性能。

---

## 245. Do LLMs Truly Generalize in the Molecular Domain? A Perturbation-Based Analysis

**arXiv ID:** 2607.01800 | [PDF](https://arxiv.org/pdf/2607.01800v1)

**作者:** Jiatong Li `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过构造语法合法的分子扰动框架，系统评估大语言模型在分子结构变化下的泛化能力。

**💡 创新点**

创新点在于提出基于图编辑距离的分子扰动方法，并结合局部置信区间与检索式上下文微调（ICT）探讨模型鲁棒性。

**🔧 技术方法**

使用的技术包括图编辑距离（GED）度量、原子与键级扰动、检索式上下文微调（ICT）、自回归和编码解码LLM（Galactica、Qwen、MolT5、BioT5）。

**📊 数据集**

数据集主要是公开的化学语料库（SMILES/SELFIES、IUPAC命名）及其语法合法衍生的扰动样本。

**📈 对比分析**

通过与直接微调、随机检索等基线对比，发现ICT在GED≤3时显著提升性能，鲁棒性提升约10‑20%，但在GED≥4时差距缩小。

**⚠️ 局限性**

局限性在于ICT只能在训练分布附近扩展可信域，扰动越大检索邻居距离增大，导致鲁棒性递减；模型仍对化学拓扑极度敏感。

---

## 246. EHHN: An Event-driven Heterogeneous Hypergraph Network for Object-Centric Next Activity Prediction

**arXiv ID:** 2607.01785 | [PDF](https://arxiv.org/pdf/2607.01785v1)

**作者:** Jiaxing Wang `[一作]` (Zhejiang University of Technology), Ji Zhang `[通讯]` (University of Southern Queensland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于异质超图的EHHN框架，用于对象中心化日志（OCEL）中的下一个活动预测，能够在保持多对象事件绑定的同时捕捉事件驱动的对象状态演化、时间动态和全局执行模式；

**💡 创新点**

创新点在于：①利用异质超图将多对象事件直接建模为超边，避免对事件-对象关系的对偶拆分；②设计双流结构：微空间流采用事件驱动状态更新和生命周期约束，宏演化流通过时间感知注意力与全局原型记忆融合全局模式；③通过原型多样性正则和FiLM调制实现对全局模式的可解释性引导；

**🔧 技术方法**

采用的技术包括：超图表示、JEST（事件驱动状态转移）、LCSE（生命周期约束状态演化）、HIE（异质交互编码器）、TASE（时间感知Transformer）、全局原型记忆与FiLM调制、跨层融合与多任务学习；

**📊 数据集**

实验使用四个公开OCEL基准数据集：OTC、BPI 2017、Intermediate、P2P；

**📈 对比分析**

与9个基线（包括平面化/传统单实例方法和原生OCEL图方法）比较，EHHN在所有四个数据集上均取得最高准确率和宏F1，最高提升分别为8.1%和12.4%；在GPU内存和时间方面相较于最强图基线可降低约24×内存占用，虽然推理延迟略高；

**⚠️ 局限性**

局限性包括：在OTC等结构复杂数据集上训练时间相对较长；对超图构建与双流编码的计算开销仍高于简化的平面化模型；原型记忆对K值不敏感但仍需手工调参；仅针对下一个活动预测，未考虑多步预测与在线流式更新等实际业务需求。

---

## 247. Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding

**arXiv ID:** 2607.01775 | [PDF](https://arxiv.org/pdf/2607.01775v1)

**作者:** Marianne Arriola `[一作]` (Cornell University), Volodymyr Kuleshov `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Set Diffusion 框架，允许在自回归（AR）和扩散（diffusion）之间通过生成顺序平滑插值，实现可变长度、任意位置的生成；

**💡 创新点**

创新点在于：①以集合为单位建模生成顺序，而非固定块；②采用位置相关的 reveal‑time 方案实现左到右偏置；③构建集因果扩散架构，实现每一步 KV 缓存更新；

**🔧 技术方法**

技术主要包括离散扩散模型、位置偏置的 reveal‑time 采样、集因果 Transformer、变长集合分块、低方差训练目标以及滑动窗口采样；

**📊 数据集**

使用 OpenWebText、One Billion Words 进行预训练，评估数据集包括 GSM8K（数学推理）、CNN/DailyMail（摘要）、ROCStories（填空）以及 OWT 进行无条件生成；

**📈 对比分析**

与块扩散（BD3LM）、MDLM、AR 以及其他扩散模型对比，在 GSM8K 上 0-shot pass@1、摘要 ROUGE、填空 ROUGE 以及无条件生成 MAUVE 等指标上均达到或超过最优扩散模型，并在推理速度（tokens/s）上较块扩散提升 10–25%；

**⚠️ 局限性**

局限在于生成顺序的调优仍需手工或学习，平衡精度与并行度需根据硬件与任务特性手动设置，且对超参数（如窗口宽度）敏感。

---

## 248. Verifiable Knowledge Expansion through Retrieval-Grounded Formal Concept Analysis

**arXiv ID:** 2607.01773 | [PDF](https://arxiv.org/pdf/2607.01773v1)

**作者:** Yujin Yang `[一作]` (Hanyang University), Heejung Lee `[通讯]` (Hanyang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

结合正式概念分析与检索增强小型语言模型的循环，自动构建并验证罕见共济失调疾病-表型关系的本体。

**💡 创新点**

在传统FCA的验证循环中加入检索驱动的证据支持，形成可审计的符号-子符号混合构建流程。

**🔧 技术方法**

使用正式概念分析（FCA）、检索增强生成（RAG）、小型语言模型（Gemma2、Llama3.2、Qwen2.5）以及检索索引。

**📊 数据集**

以Orphadata及HPO的罕见共济失调疾病和表型标签构成的122病例、160诊断属性的数据集。

**📈 对比分析**

与无检索的 GPT‑4o/4o‑mini 对比，检索增强实验在20轮中取得关系F1最高0.52、闭包基规则F1最高0.41；检索缺失时可得到更高F1但扩展有限。

**⚠️ 局限性**

主要限制在于检索文本覆盖不足导致表型关联召回低、属性发现与归一化困难，以及小型模型在局部判定上的误差，无法实现完整的本体生成。

---

## 249. On the structure of constacyclic codes over finite chain rings

**arXiv ID:** 2607.01771 | [PDF](https://arxiv.org/pdf/2607.01771v1)

**作者:** Vaishali Singh `[一作]` (Punjab Engineering College (Deemed to be University)), Ridhima Thakral `[通讯]` (Punjab Engineering College (Deemed to be University))

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文针对任意长度的λ-常量循环码（λ-constacyclic codes）在有限链环（finite chain rings, FCR）上的结构进行研究，给出了其生成多项式的显式构造，并由此得到最小生成集合与码的秩。

**💡 创新点**

创新点在于：①提出了一种基于最小次数多项式的逐步选取方法，能够生成最少数量的生成元；②通过对生成元的γ-幂次与次数关系的分析，推导出最小生成集与秩的解析式；③给出了关于该类码是否为最大汉明距离（MHDR）与最大距离可分离（MDS）的必要与充分条件，且条件以余域上扭转码（torsion code）的性质为核心。

**🔧 技术方法**

主要技术包括：有限链环的γ-分解理论、环多项式环的理想结构、扭转码与余域映射、以及多项式除法与最小化生成元的递归构造。

**📊 数据集**

本文没有使用外部数据集，所有示例均为在特定环（如 Z_125、Z_343、Z_289、F_5+γF_5 等）上构造的人工例子。

**📈 对比分析**

由于研究对象为理论构造与性质证明，本文未进行实验对比或性能评估；因此此类比较与性能指标未给出。

**⚠️ 局限性**

局限性包括：①仅适用于有限链环；②需要λ为单位；③研究仅覆盖常量循环码，未扩展到非循环或更一般的码类；④在实际编码实现中，生成多项式的计算复杂度与码长、环的nilpotency index相关，可能影响应用效率。

---

## 250. JointHOI: Jointly Generating Contact Maps Enhances Hand Object Interaction Generation

**arXiv ID:** 2607.01768 | [PDF](https://arxiv.org/pdf/2607.01768v1)

**作者:** Mingyeong Song `[一作]` (Ewha Womans University), Junhyug Noh `[通讯]` (Ewha Womans University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出 JointHOI，一种单阶段扩散框架，能够从文本提示中同时生成 3D 双手–物体交互运动与动态距离型接触图，并通过接触内引导（CIG）在推理时进一步提升物理可行性。

**💡 创新点**

创新点包括：① 将接触作为运动的内置模态共同生成，学习运动–接触的时空耦合；② 使用连续距离型动态接触图捕捉时变接触细节；③ 设计无额外网络的接触一致性引导（CIG），在采样过程中直接对生成的接触与几何一致性进行约束。

**🔧 技术方法**

主要技术包括：扩散模型（Transformer‑based denoiser）联合文本（CLIP）与物体几何（PointNet）条件；MANO 可微手模型实现几何推导；距离场接触计算与 Log‑L1 接触一致性能量；基于梯度的采样引导。

**📊 数据集**

使用 GRAB（51 个刚体、29 类动作）和 ARCTIC（11 个关节物体、10 类动作）两个公开双手交互数据集进行训练与评估。

**📈 对比分析**

与 MDM、DiffH2O、LatentHOI、Text2HOI 等现有方法对比，JointHOI 在语义精确度（Top‑1/3 Acc、FID）以及物理可行性（渗透体积、渗透深度、接触率）上均取得领先，尤其在 ARCTIC 上 Top‑3 Acc 0.983、IV 仅 4.4 cm³，CR 接近真实值。

**⚠️ 局限性**

局限性：① 对接触精度仍依赖预设的 1024 个锚点，可能在极细小物体或复杂表面上产生误差；② 仅验证了固定对象几何，未覆盖动态场景或环境交互；③ CIG 引导虽无额外网络，但仍需调节引导权重，极端值可能影响运动多样性。

---

## 251. Role-Aware Neural Convex Divergence Heads for Asymmetric Representation Learning

**arXiv ID:** 2607.01762 | [PDF](https://arxiv.org/pdf/2607.01762v1)

**作者:** He Huang `[一作]` (Chongqing Technology and Business University), Li Qi `[通讯]` (Chongqing Technology and Business University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种角色感知的神经凸散度头，用于学习有方向的表示关系

**💡 创新点**

通过在Bregman散度前引入源与目标投影，实现既保持非负凸结构又能显式表达方向性

**🔧 技术方法**

结合输入凸网络(ICNN)构造可学习的凸势函数，使用线性或浅层投影映射，并在此空间计算Bregman散度

**📊 数据集**

在Lexical（HyperLex、WordNet）、句子（SICK、SNLI）、本体（Gene Ontology）以及OGBL-Citation2图数据集上评估

**📈 对比分析**

与欧氏、余弦、马氏、无结构MLP、顺序嵌入、Poincaré等对手比较；在十个随机种子下，角色投影版在方向准确率上显著优于普通ICNN头，负散度率为零；在大规模引用预测任务中表现略逊于Poincaré和cosine，但仍优于传统ICNN

**⚠️ 局限性**

局限性包括仅在固定嵌入上验证、投影映射多为线性（可提升但解释性下降）、在纯排名指标上不一定优于专门的几何或MLP基线、缺乏端到端编码器与头的联合训练

---

## 252. ReQuest: Rethinking-based Question-Aware Frame Selection for Long-Form Video QA

**arXiv ID:** 2607.01737 | [PDF](https://arxiv.org/pdf/2607.01737v1)

**作者:** Minkuk Kim `[一作]` (Kyung Hee University), Seong Tae Kim `[通讯]` (Kyung Hee University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于不确定性驱动、问题感知的关键帧选择管线ReQuest，能够在固定视觉 token 预算下显著提升长视频问答的准确率。

**💡 创新点**

创新点包括（1）轻量化的 MLLM 模仿器作为问题感知选择器；（2）基于预测熵和视频长度的 Re‑thinking 路由器，实现只在模型不确定时进行额外推理；（3）不确定性引导的自适应 NMS，使帧间间隔随问题难度动态调整，从而避免冗余并捕获关键信息。

**🔧 技术方法**

采用 BLIP 视觉‑语言交互编码器、轻量化 Transformer 评分头、熵基不确定度判别、长度校正因子及自适应间隔的贪心 NMS 等技术。

**📊 数据集**

在三大长视频 QA 基准上进行评测：Video‑MME、MLVU 与 LongVideoBench（多选与开放式问答）。

**📈 对比分析**

与统一采样、相似度选择器、VL‑LLM 等方法对比，ReQuest 在 Video‑MME 上从 62.6% 提升至 65.6%，在中长视频子集提升超过 4%；在 MLVU 与 LongVideoBench 上亦实现 5–7% 的准确率提升，且在高帧容量 MLLM（如 Qwen3‑VL‑8B）中保持成本优势。

**⚠️ 局限性**

局限性包括：仍需额外的 re‑thinking 推理步骤，导致推理时间波动；对超短视频或信息已被充分捕获的场景提升有限；阈值调优需经验，且对不同 MLLM 的适配仍需实验验证。

---

## 253. Predicting Closed-Loop Performance of Latent World Models: Offline Checkpoint Selection for MPC and Model-Based RL Under Non-Markovian Rewards in LunarLander

**arXiv ID:** 2607.01736 | [PDF](https://arxiv.org/pdf/2607.01736v1)

**作者:** Nikolai Smolyanskiy `[一作]` `[通讯]`, Nikolai Smolyanskiy

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究如何仅凭验证阶段的诊断指标来预测学习到的潜在世界模型在闭环控制中的性能，并通过这些指标实现离线检查点选择。

**💡 创新点**

创新点在于提出了奖励可观测分数（ROF）和综合奖励可观测分数（CROF），这两项指标能够捕捉奖励梯度与可观测子空间的对齐程度，从而准确预测CEM-MPC和基于世界模型的A2C在LunarLander-v3环境中的闭环表现。

**🔧 技术方法**

技术方法包括基于RSSM的世界模型训练、跨步Jacobians控制理论分析、跨步预测误差与灵敏度度量、以及使用CEM-MPC进行离线规划和A2C在潜在空间中的想象训练。

**📊 数据集**

数据集为Gymnasium的LunarLander-v3，使用872条人类控制的轨迹（共180,916步）进行离线训练与评估。

**📈 对比分析**

通过与模型无关的A2C基线对比，CROF选择的世界模型可在约65倍更少的真实环境交互下使A2C平均回报提升约24.5点（+217.5 vs. +193.0），并在CEM-MPC上实现+166.6的回报，显示了显著的数据效率提升。

**⚠️ 局限性**

局限性在于实验仅针对单一环境，且ROF与CROF在奖励完全Markov且无形状化的任务中效果有限，未来需要在更多连续动作、高维观测及多样化奖励结构的环境中验证。

---

## 254. Exploiting Task-Based Parallelism for the Red-Black Gauss-Seidel Method on 2D Grids

**arXiv ID:** 2607.01735 | [PDF](https://arxiv.org/pdf/2607.01735v1)

**作者:** Shiting Long `[一作]` (KTH Royal Institute of Technology), Dirk Pleiter `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现并评估了在2D Poisson方程上使用红黑高斯-赛德尔（RBSGS）方法的多任务并行版本，比较了传统OpenMP并行循环、OpenMP任务以及OmpSs-2任务模型的性能。

**💡 创新点**

提出利用任务调度和NUMA感知的动态调度机制，在保持RBSGS算法可扩展性的同时显著缓解了多核系统的同步与内存访问异步性问题。

**🔧 技术方法**

采用OpenMP（循环与任务指令）与OmpSs-2（基于数据依赖的任务调度）实现RBSGS，并在两台不同架构（x86与Arm）上进行基准测试。

**📊 数据集**

使用2D Poisson方程的均匀网格（N=3000~4000）作为测试数据集，运行100次迭代以确保内存受限。

**📈 对比分析**

通过对比传统OpenMP并行循环与两种任务实现，在JUWELS（Xeon）和HAICGU（Kunpeng）上测得OmpSs-2与OpenMP循环相当且在HAICGU上更优，主要得益于其NUMA感知与动态任务分配，且相较于传统任务实现具有更稳定的性能。

**⚠️ 局限性**

局限在于任务粒度需要人工调优；对极大规模或高维/更复杂算子时，任务调度开销可能会削弱优势；且实验仅限单节点共享内存，未涉及分布式MPI混合。

---

## 255. Denser $\neq$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training

**arXiv ID:** 2607.01763 | [PDF](https://arxiv.org/pdf/2607.01763v1)

**作者:** Meng Wang `[一作]` (Centre for Artificial Intelligence and Robotics, HKISI, Chinese Academy of Sciences), Fei Zhu `[通讯]` (Centre for Artificial Intelligence and Robotics, HKISI, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究在持续后训练（Continual Post‑Training）中，基于教师-学生框架的自监督蒸馏（Self‑Distillation Policy Optimization，SDPO）与基于序列级奖励的策略优化（Group Relative Policy Optimization，GRPO）的表现差异，重点评估SDPO在保持已有能力与学习新技能时的优劣。

**💡 创新点**

首次将教师与学生使用相同模型、在不同上下文（如示例、链式推理、反馈）下的自监督蒸馏与持续训练相结合，系统量化了“稠密标注”带来的加速与易失效，并通过参数/响应漂移、崩溃模式等诊断方法揭示了SDPO在持续学习中的“过度漂移”与“确认偏误”机制。

**🔧 技术方法**

技术包括：①基于自回归模型的Jensen‑Shannon Divergence（JSD）蒸馏目标；②组相对策略优化（GRPO）作为基准；③教师EMA（Exponential Moving Average）更新速率的探测；④对模型权重做奇异值分解（SVD）来度量参数漂移；⑤通过最大均值差异（MMD）与KL漂移分析评估OOD迁移与记忆保持；⑥自定义损失与正则化（如KL约束、Token‑level weighting）。

**📊 数据集**

使用多域持续训练数据集：数学（AIME、Math500）、逻辑（ZLogic、BFCLv4）、知识（Knowledge）、数学推理评测（LCBv6）、通用推理（GPQA）、算术推理（GPQA等）以及与基准模型对齐的OOV任务；所有任务均基于同一基础大型语言模型（如Qwen3/ChatGPT等）。

**📈 对比分析**

比较方法：在单域与多域持续训练设置下，对比GRPO与SDPO的当前域精度（Current）和最终域精度（Last），以及在OOD基准上的Acc@8和相对性能变化。实验结果显示：SDPO在单域能快速提升精度（如AIME提升至56%），但在多域持续训练中记忆保持率下降（最终模型在部分任务下低于基准），且相对性能变化呈非单调性；GRPO在保持已有能力方面更稳定，尽管在单域的提升幅度相对较小。

**⚠️ 局限性**

局限性包括：①SDPO对教师信号的稳定性与对齐度高度敏感，容易放大格式化噪声与高频错误；②稠密Token‑level监督导致参数与输出漂移累积，易引发崩溃；③缺乏有效的Token‑level权重或教师控制机制，导致过度强化不必要的细节；④实验集中于少数领域和模型，尚未验证在更大规模或更广泛任务上的通用性。

---

## 256. Approximate Attention Weighting for Sustainable FPGA-Based Vision Transformer Inference

**arXiv ID:** 2607.01798 | [PDF](https://arxiv.org/pdf/2607.01798v1)

**作者:** Muhammad Usman `[一作]` (University of Regensburg), Dorit Merhof `[通讯]` (University of Regensburg)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个BRAM-free的FPGA Vision Transformer注意力权重单元，能在小型SoC FPGA上高效完成注意力权重生成。

**💡 创新点**

创新点在于使用自然指数的16段分段线性逼近，既保留预训练模型的softmax温度，又避免了BRAM占用与基数转换的需要。

**🔧 技术方法**

采用分段线性逼近、分布式LUTRAM、DSP矩阵乘累加、恢复除法等RTL实现，完成全行注意力计算的两个通道流水。

**📊 数据集**

使用Imagenette验证集和ViT-S/16、ViT-B/16、ViT-L/16模型进行硬件级精度评估。

**📈 对比分析**

与精确softmax基线相比，最大Top‑1误差≤0.20%；在Xilinx Zynq‑7020上实现1,444 LUT、77 DSP、0 BRAM、21 mW动态功耗（124 mW总功耗），效率601 krows/s/W，动态能耗1.66 µJ/行。

**⚠️ 局限性**

局限性：仅实现单行注意力核心，未覆盖完整Transformer流水线；扩展到更大模型或更高维度可能需要更多DSP/逻辑；系统级功耗与实际部署验证仍待完成。

---

## 257. RTE-FM-Dehazer: Radiative Transfer Equation Inspired Flow Matching for Real-World Image Dehazing

**arXiv ID:** 2607.01748 | [PDF](https://arxiv.org/pdf/2607.01748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 258. Adapting CCDF Plots for Visualizing Ordinal Regression Results

**arXiv ID:** 2607.01747 | [PDF](https://arxiv.org/pdf/2607.01747v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 259. Faster Parameterized Broadcasting

**arXiv ID:** 2607.01770 | [PDF](https://arxiv.org/pdf/2607.01770v1)

**作者:** Édouard Bonnet `[一作]` (National Center for Scientific Research), Manolis Vasilakis `[通讯]` (Paris Dauphine University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了三种改进的 FPT 算法，分别针对电话广播问题的顶点覆盖数、顶点完整度和距离到团参数化，并给出了对应的时间复杂度上界。

**💡 创新点**

创新点在于将问题转化为最大权 b‑matching：通过“模板”猜测广播协议中有限的关键结构，然后用 b‑matching 解决剩余匹配与分配，从而将原先的 2^(k³)、2^(vi⁵) 等指数依赖降到 2^(k log k)、2^(vi² log vi)，显著提升算法效率。

**🔧 技术方法**

主要技术包括：模板化（template）来枚举有限的协议骨架；Turing 归约到边加权 b‑matching；利用图的对称性和贪心策略完成团内部的广播；以及对顶点完整度、距离到团的结构分析。

**📊 数据集**

论文中没有使用实验数据集，全部为理论算法与复杂度证明。

**📈 对比分析**

通过理论分析比较：相较之前的 2^(k³) n^O(1)、2^(vi⁵) n^O(1) 等上界，新的算法分别实现了 2^(k log k) n^O(1) 与 2^(vi² log vi) n^O(1) 的改进；实验比较未给出。

**⚠️ 局限性**

局限性：虽然指数依赖已显著降低，但仍为 2^(O(k log k))，无法达到 2^(O(k)) 的期望；对其他参数（如 cutwidth、bandwidth）的 FPT 可行性仍未知；并且论文未给出 ETH 下的严格下界，是否存在更优算法仍是开放问题。

---

## 260. ProSAC-CT: Progressive Spectral-Anatomical Co-Guided Multi-Stage Diffusion Model for Low-Dose CT Denoising

**arXiv ID:** 2607.01756 | [PDF](https://arxiv.org/pdf/2607.01756v1)

**作者:** Xuepeng Liu `[一作]` (Northeastern University), Eichi Takaya `[通讯]` (Tohoku University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种面向低剂量CT去噪的 ProSAC-CT 模型，利用解剖先验引导、频域解耦和时间步解耦的多阶段扩散网络实现高质量、结构保留的去噪。

**💡 创新点**

创新点在于三大模块：1）LDCT自适应解剖先验引导（APGC）在编码阶段注入结构信息；2）残差频域解耦（RFDDS）对低/中/高频进行分支加权；3）时间步解耦去噪解码器（TD³）在扩散逆向过程中按阶段使用不同频域增强特征，显著提升边界清晰度与细节恢复。

**🔧 技术方法**

使用的技术包括条件扩散模型（基于 RDDM），MedDINOv3、Sobel/Laplacian 边缘提取、FFT 频域分块、深度可分离卷积、残差学习、Adam 优化、1000 步残差采样等。

**📊 数据集**

数据集涵盖四个低剂量CT基准：Mayo-2016（腹部），Mayo-2020（胸腹部），LoDoPaB（低剂量并行束）和 QIN-Lung（投影域合成），每个均包含真实或模拟的 LDCT/NDCT 对。

**📈 对比分析**

与 CNN、GAN、Transformer 以及多种扩散方法（RED-CNN、UNAD、DU-GAN、CTformer、AMIR、I2SB、ResShift、RDDM、CoreDiff 等）在统一训练/评估协议下对比，ProSAC-CT 在 PSNR、SSIM、FSIM、NQM、VIF 上均获得最高分，PSNR 最高提升约 1.7–1.8 dB；下游六分类解剖区域识别实验亦显示其在 F1、BAcc、AUC 上最优，接近 NDCT 水平。

**⚠️ 局限性**

局限性包括：1）仍依赖 LDCT 图像自身的结构信息，极端噪声下先验提取可能失效；2）模型训练与推理较为复杂，参数量和推理时间高于传统 CNN；3）在极端低剂量或非CT模态下的泛化尚未充分验证。

---

## 261. Efficient Temporal Point Processes via Monotone Alternating Splines

**arXiv ID:** 2607.01752 | [PDF](https://arxiv.org/pdf/2607.01752v1)

**作者:** Cheng Wan `[一作]` (Hong Kong University of Science and Technology), Feng Zhou `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种新的时点过程累计条件强度函数（CCIF）建模方法——Monotone Alternating Splines (MAS)，用分段单调样条代替传统的单调神经网络（MNN）来逼近CCIF；

**💡 创新点**

创新点在于揭示MNN在CCIF建模中存在的三大结构瓶颈（凸性限制、饱和极限、违反CCIF要求），并通过MAS的插值分段样条与全局单调外推相结合，既提升了拟合能力，又保证了理论上的一致性与泛化性能；

**🔧 技术方法**

主要技术包括使用Transformer对历史事件进行编码、MLP+Softplus生成正值的插值间隔、累计增量和导数，构造Rational Quadratic Spline（RQS）或其他单调样条进行插值，外推采用线性或线性+指数函数，整体通过最大似然无积分训练，且所有导数可解析；

**📊 数据集**

实验采用五个合成单变量数据集（两种Hawkes、两种Renewal、一个Self‑Correcting）以及四个多变量真实数据集（Retweet、Earthquake、Taxi、Taobao）；

**📈 对比分析**

与13个基线模型（包括参数化模型、CIF模型、MNN‑CCIF模型以及流/分数匹配模型）在NLL、RMSE、ACC等指标上进行对比，MAS在大多数数据集上获得最优或次优表现，并且训练速度显著快于基线；

**⚠️ 局限性**

局限性主要是对插值节点数和插值范围的超参数敏感，过多节点或过大范围易导致过拟合；此外仍需依赖Transformer编码器的表达能力，极端大规模数据或复杂事件类型时可能需要进一步优化。

---

## 262. Knowledge Over Parameters: Evolving Smart Contract Vulnerability Detection

**arXiv ID:** 2607.01742 | [PDF](https://arxiv.org/pdf/2607.01742v1)

**作者:** Yuqiang Sun `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出一种自动化框架，将智能合约漏洞检测转化为可演化的程序化知识生成与改进过程，并在无须大规模标注数据的前提下实现检测。

**💡 创新点**

通过 IoC 执行架构与两阶段演化（Cold Start + Few-Shot Evolving），实现规则自动生成、严谨执行、稀缺数据下的高效演化，且演化出的规则可跨模型迁移。

**🔧 技术方法**

采用 LLM 两角色（监督与检测）、可执行策略（Executable Policy）、逆向推理调试、基于执行日志的故障定位与规则压缩等技术。

**📊 数据集**

使用 DeFiHackLab 公开攻击事件与 Etherscan 验证合约，覆盖价格操纵、访问控制、验证不足、精度问题与重入攻击共五类，训练集每类 10 样本（5+5），测试集剩余。

**📈 对比分析**

与 Slither、GPTScan、MANDO‑GURU、SAEL、iAudit、零样本 LLM 及编码代理等基线对比，宏观 F1 达 71%，优于所有基线，轻量模型可超越更大 LLM 19 个百分点。

**⚠️ 局限性**

受限于样本规模、仅覆盖 Solidity、对新型未知漏洞缺乏零样本识别，且演化过程仍依赖 LLM 推理质量与规则更新策略。

---

## 263. Rethinking Speech-LLM Integration for ASR: Effective Joint Speech-Text Training by Interleaving

**arXiv ID:** 2607.01733 | [PDF](https://arxiv.org/pdf/2607.01733v1)

**作者:** Ruchao Fan `[一作]` (Microsoft), Jinyu Li `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Joint Speech-Text Interleaved Pre-Training（JSTIP）方案，在对齐的语音-文本对中构造词级和段级交错序列，保持 LLM 的生成先验并显著提升 ASR 的实体识别效果。

**💡 创新点**

创新点在于：①通过在每个语音-文本配对内部交错构造，解决 decoder‑only Speech‑LLM 在大规模 ASR 监督下对文本先验的遗忘；②提出可扩展的词级连续语音交错方法，并将其与段级交错结合，进一步缩小语音-文本模态差距。

**🔧 技术方法**

使用技术包括：Conformer 语音编码器 + 轻量 adapter + 7B 参数 decoder‑only LLM；词级/段级对齐、交错序列构造；损失掩码仅对文本标记计算；FlashAttention + 8k 上下文打包训练；下一词预测（NTP）目标。

**📊 数据集**

使用数据集：38k 小时内部英语 ASR 数据；9k 小时合成医学 TTS 数据；PubMed 2.3B 文本；TTS 转录文本；内部医学与银行领域评测集。

**📈 对比分析**

通过与 ASR‑only、文本+ASR、以及 Whisper‑large‑v3、Qwen、Voxtral、Gemma 等开源 Speech‑LLM 基线在 TER、EER、MMLU、SQA 上对比；JSTIP 在医学实体识别 EER 上相对 ASR‑only 提升约 17%，在 SQA 零样本准确率从 0.05% 提升至 41%，并在医学领域整体 EER 上优于 Whisper‑large‑v3，展现出显著的性能提升。

**⚠️ 局限性**

局限性包括：对词级交错的效果高度依赖对齐质量；实验仅在内部语料和英语环境下验证，缺乏跨语言、跨模型的普适性验证；在银行等非医学领域仍表现不佳；对齐误差和长短句切分不完善可能导致模型训练受限。

---

## 264. Single-Channel EEG-Based Cognitive Load Assessment in Online Learning: A Hybrid Deep Learning Approach

**arXiv ID:** 2607.01795 | [PDF](https://arxiv.org/pdf/2607.01795v1)

**作者:** Rowan Hussein `[一作]` (University of Ottawa), Mohamed Ouf `[通讯]` (Queen's University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

基于单通道消费级EEG设备，评估在线学习中认知负荷的可行性，构建并验证了一个混合CNN+LSTM+Attention模型。

**💡 创新点**

创新点包括：①将原始波形与频带功率序列双输入融合；②在单通道EEG上实现了注意力机制以捕获时间依赖性；③公开了可复现的留一受试者交叉验证（LOSO）评估管线。

**🔧 技术方法**

使用了深度学习框架TensorFlow/Keras搭建CNN+LSTM+Attention网络，并对传统机器学习基线（决策树、随机森林、XGBoost、SVM等）进行对比；同时采用Dropout、L2正则化和早停法控制过拟合。

**📊 数据集**

采用Wang等人公开的数据集（10名受试者，9名有效，90个视频片段），每名受试者观看5段易、5段难的视频，并给出自评困惑标签。

**📈 对比分析**

与传统基线模型（最高约55%准确率）相比，混合模型在受试者内交叉验证中最高达78.5%准确率；经过正则化后，验证准确率稳定在68–73%。

**⚠️ 局限性**

局限性：样本量极小（仅9名受试者），受试者内评估结果偏乐观；单通道EEG信息受限；标签为粗粒度自评困惑，缺乏客观认知负荷标准；未给出正式的留一受试者独立评估结果。

---

## 265. Safety Testing LLM Agents at Scale: From Risk Discovery to Evidence-Grounded Verification

**arXiv ID:** 2607.01793 | [PDF](https://arxiv.org/pdf/2607.01793v1)

**作者:** Yunhao Feng `[一作]` (AntGroup), Xinhao Deng `[通讯]` (AntGroup)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了可扩展的安全测试框架，自动生成可执行安全案例并在沙盒中适配多种LLM代理

**💡 创新点**

将软件工程测试原则（测试用例构造、组合生成、证据驱动验证）迁移到非确定性代理，提供自适应驱动与多通道攻击模型

**🔧 技术方法**

利用三阶段流水线：文献驱动风险挖掘、组合生成安全案例、沙盒化自适应执行与环境状态验证；使用LLM生成目标、脚本化初始化与判定

**📊 数据集**

构建了VeraBench（1600个可执行案例，覆盖124风险类、77攻击方式、30环境），并在OpenClaw、Hermes、Codex、Claude Code等四种生产代理上评测

**📈 对比分析**

对比单通道/多通道/正常任务，攻击成功率高达90-94%，展示了相较于传统静态评测的更高覆盖率和更高攻击成功率；门限基准性能提升显著

**⚠️ 局限性**

局限在于仅评估推理时可执行的攻击，无法覆盖训练阶段后门；对极大规模多轮交互的可扩展性和对新型工具链的即时适配仍待改进

---

## 266. SpaceEra++: A Unified Framework Towards 3D Spatial Reasoning in Video

**arXiv ID:** 2607.01784 | [PDF](https://arxiv.org/pdf/2607.01784v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 267. Repair the Amplifier, Not the Symptom: Stable World-Model Correction for Agent Rollouts

**arXiv ID:** 2607.01767 | [PDF](https://arxiv.org/pdf/2607.01767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 268. DL-VINS-Factory: A Modular Framework for Learned Visual Front-Ends in Visual-Inertial SLAM

**arXiv ID:** 2607.01757 | [PDF](https://arxiv.org/pdf/2607.01757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 269. KRCA: An Efficient Root Cause Analysis System in Hyper-Scale Microservice Systems via Agentic AI

**arXiv ID:** 2607.01788 | [PDF](https://arxiv.org/pdf/2607.01788v1)

**作者:** Jiamin Jiang `[一作]` (Nankai University), Dan Pei `[通讯]` (Tsinghua University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并部署了针对超大规模微服务系统的端到端根因分析（RCA）系统KRCA。

**💡 创新点**

创新点包括：API级别的钻取搜索策略、基于骨架的因果图结构先验以及多代理记忆增强的LLM推理框架。

**🔧 技术方法**

技术手段涵盖：时间序列相关性计算、权重组合的延迟评分、Skeleton Graph Instantiation、Retrieval Augmented Generation（RAG）与三层记忆、主从多代理协作。

**📊 数据集**

数据集为Kuaishou生产环境收集的300个真实故障实例，涵盖200k+微服务与9类常见故障类型。

**📈 对比分析**

与CoT、ReAct、Reflexion及RCA-Agent等基线对比，KRCA在根因定位AC@1达到0.88、故障类型分类AC@1为0.79，提升约30%并在生产中将MTTR降低77%。

**⚠️ 局限性**

局限性：对关键下游服务的可观测性依赖、同一时间多链路异常导致分辨困难，以及LLM对历史检索匹配质量的敏感性。

---

## 270. ProCal: Inference-Time Proposal Calibration for Open-Vocabulary Object Detection

**arXiv ID:** 2607.01759 | [PDF](https://arxiv.org/pdf/2607.01759v1)

**作者:** Jae-Ryung Hong `[一作]` (Korea University), Seong-Whan Lee `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在开放词汇目标检测中，提出了一种推理时的提议校准方法ProCal，通过利用冻结的视觉语言模型生成前景与背景相似度信号，对检测得分进行重校准。

**💡 创新点**

创新点在于使用类无关提示（prompt）从冻结VLM提取定位前景和抑制背景的分数，并将其融合为提议先验，在不需额外训练的情况下提升未见类别的定位质量。

**🔧 技术方法**

采用冻结的CLIP/OpenCLIP视觉编码器与文本编码器，计算三种提示（"背景对象"、"对象"、"背景"）的余弦相似度，得到s_loc和s_bg，再通过sigmoid与加权平均构成q_i，再融合到原始得分。

**📊 数据集**

在OV-COCO（COCO划分为48基类+17新类）和OV-LVIS（LVIS 1203类）两个开放词汇基准上进行评估。

**📈 对比分析**

与RegionCLIP、Detic、F-VLM、CORA、OV-DQUO、CLIPSelf等方法对比，ProCal在OV-LVIS上AP_novel提升约+2.5，在OV-COCO上AP_novel提升约+2.7，保持或略低于部分方法的总AP，但显著提升了新类别检测性能。

**⚠️ 局限性**

局限性包括仅适用于两阶段检测框架，对基类检测提升有限，依赖于预训练VLM的质量和提示设计，且未针对更大范围的分布偏移或开放集不确定性进行全面验证。

---

## 271. Mastermind: Strategy-grounded Learning for Repository-Scale Vulnerability Reproduction

**arXiv ID:** 2607.01764 | [PDF](https://arxiv.org/pdf/2607.01764v1)

**作者:** Mingzhe Du `[一作]` (National University Of Singapore), See-Kiong Ng `[通讯]` (National University Of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种双循环框架Mastermind，先用可训练的策略规划器生成简洁的自然语言策略，再将其传给冻结的执行器完成漏洞重现任务，并通过验证器给出反馈，持续更新策略与经验。

**💡 创新点**

创新点在于把高层策略而非完整执行轨迹作为学习单元，将策略学习与任务本地经验分离，实现跨模型可迁移的策略生成，并通过群组相对策略优化（GRPO）实现高效强化学习。

**🔧 技术方法**

主要技术包括策略规划器的监督微调+GRPO、经验策划器（Curator）记录任务本地信息、Slot-conditioned多样化策略采样、基于CyberGym里八级里程碑的密集奖励信号以及双循环训练与推理。

**📊 数据集**

使用的数据集为CyberGym，包含来自ARVO和OSS‑Fuzz的1,507个真实仓库级漏洞重现任务，在260个训练任务上训练策略，在200个独立评估任务上进行评测。

**📈 对比分析**

在评估集上，Mastermind与冻结执行器GPT‑5.5配合时获得84.5%成功率，显著超过独立Best‑of‑8（63.0%）、迭代经验（77.0%）和Level‑3开放书（60.0%）等基线；同一策略规划器对GPT‑5.4 mini和GLM‑5.1也分别提升至60.0%和71.0%。

**⚠️ 局限性**

局限性包括：依赖大量执行器回合的训练成本高；策略生成仍受限于现有执行器的能力；在少数任务中仍难以区分不同构造的崩溃（m6 vs m7）；以及对其他软件工程任务或未来模型的泛化尚待验证。

---

## 272. Open Source Is Not One Thing: A Typology of Open-Source Software Sub-Genres

**arXiv ID:** 2607.01750 | [PDF](https://arxiv.org/pdf/2607.01750v1)

**作者:** Mohamed Ouf `[一作]` (Queen's University), Rowan Hussein `[通讯]` (University of Ottawa)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并系统化了14种开源软件（OSS）子流派的分类框架，阐明各子流派的驱动因素、治理模式、资金来源以及代表项目，并给出研究议程。

**💡 创新点**

创新点在于以项目驱动力为轴，兼顾治理和资金维度，将OSS划分为传统社区驱动、公司支持、基金会治理等多样化子流派；同时标注了研究成熟度，明确哪些子流派仍缺乏实证支持。

**🔧 技术方法**

采用轻量级多源文献检索与规则筛选技术：在OpenAlex和arXiv上执行46个检索查询，使用标题/摘要规则筛选，去重后形成最终数据集。

**📊 数据集**

数据集：检索得到5,771条记录，去重后3,925篇论文，最终筛选出399篇满足条件的文献，覆盖14个子流派，代表项目包括Debian、GitLab、OpenStack、OpenMRS等。

**📈 对比分析**

比较方法：通过文献计数与代表项目归类来评估每个子流派在学术关注度上的成熟度（Established、Growing、Emerging）；未进行量化性能评估，重点在于文献覆盖率与研究关注度。

**⚠️ 局限性**

局限性：检索非系统化，仅覆盖英文主流数据库；子流派并非互斥，存在重叠；成熟度评估基于检索结果而非客观引用指标；对非主流语言或灰色文献的覆盖有限。

---

## 273. PixGS: Pixel-Space Diffusion for Direct 3D Gaussian Splat Generation

**arXiv ID:** 2607.01803 | [PDF](https://arxiv.org/pdf/2607.01803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 274. Meta-Benchmarks for Financial-Services LLM Evaluation

**arXiv ID:** 2607.01740 | [PDF](https://arxiv.org/pdf/2607.01740v1)

**作者:** Blair Hudson `[一作]` `[通讯]`, Blair Hudson

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种元基准框架，将452个公开基准聚合到O*NET工作活动和BIAN业务域中，为金融服务领域提供模型能力档案。

**💡 创新点**

创新点在于动态加权公式（判别力×覆盖率×新近度）自动抑制饱和或罕用基准，并使用双层标准化词表映射实现可审计、跨供应商的领域评分。

**🔧 技术方法**

采用组合权重、Elo成对比较、归一化与层级聚合等技术，对模型进行任务级和业务域级评分。

**📊 数据集**

利用LLM Stats API收集的公开评测结果，共288个模型、452个基准，涵盖2022‑2026年发布的模型。

**📈 对比分析**

通过对比全球平均排名与业务域排名、敏感性分析和指标分布，展示领域评分能揭示模型在特定业务需求上的优势，排名稳定但与公共排行榜显著差异。

**⚠️ 局限性**

局限性包括仅依赖公开自报分数、缺乏对关键业务域的基准覆盖、对模型成本与法律合规评估不足，以及Elo评分对新模型的敏感性与缺乏连续性。

---

## 275. A Social Norms Approach to Youth Social Media Design

**arXiv ID:** 2607.01807 | [PDF](https://arxiv.org/pdf/2607.01807v1)

**作者:** JaeWon Kim `[一作]` `[通讯]` (University of Washington), JaeWon Kim (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了以青少年为中心、以社交规范为驱动的社交媒体原型，旨在通过改变平台规范来促进可信连接和真实表达。

**💡 创新点**

创新点：①把社交规范理论应用于中介环境，说明平台设计如何塑造和维持青少年行为的规范环境；②将隐私、信任等关系目标从单一设置转化为整个平台共同维系的社会规范；③提出并验证了与主流平台不同的“独立”社交媒体概念，强调与青少年共创并提供可持续替代方案。

**🔧 技术方法**

技术：基于Web的社交平台实现（前后端分离，使用React/Node.js/GraphQL），隐私与信任框架的交互式配置，数据日志记录与行为追踪工具，实验平台与控制平台的功能模块化对比设计。

**📊 数据集**

数据集：由99名年龄在15–25岁的青少年在韩国和美国参与的四周交叉实验日志，包含行为记录（发布、点赞、关注、私信等）、访谈转录以及问卷自评。后续部署的选项最小化实验亦使用同样的数据来源。

**📈 对比分析**

比较方法：使用交叉实验设计，控制与实验版本在同一组用户中轮换，采用配对t检验/混合效应模型对行为频率、隐私设定使用率、信任度量（自评与同行评价）进行统计比较。实验版本在促进“真实分享”行为上显著提升（p<0.01），隐私设置的主动调整率提高约40%。

**⚠️ 局限性**

局限性：①样本规模相对有限，且主要集中于英语/韩语背景，缺乏多元文化验证；②实验时长仅四周，难以观察长期行为与规范变化；③平台功能相对简化，未覆盖所有主流社交媒体的复杂交互与算法推荐机制；④缺乏对算法偏见或信息泡沫效应的深入评估。

---

## 276. On the Limits of Steering Vectors for Preference-Aligned Generation

**arXiv ID:** 2607.01802 | [PDF](https://arxiv.org/pdf/2607.01802v1)

**作者:** Melanie Subbiah `[一作]` (Columbia University), Kathleen McKeown `[通讯]` (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对大型语言模型（LLM）的激活向量（steering vectors）进行系统实验，评估其在36种写作风格特征、两大下游任务（摘要与邮件写作）以及多向量组合（最多四个）下的可泛化性与可控性。

**💡 创新点**

创新点在于：①构建包含36种多样写作风格的评估框架；②揭示不同风格特征的可控性差异与任务迁移失败；③系统比较四种多向量组合方法，发现层20的单位归一化方案在表达与连贯性之间实现最佳权衡。

**🔧 技术方法**

技术方法包括：①利用正负样本差分提取层级激活向量；②在推理时对残差流进行加权修改；④四种组合策略（正交化、不同层、均值、单位归一化）；⑤使用LLM-as-a-judge（GPT‑4o/4.1‑mini）评估风格表达与连贯性。

**📊 数据集**

主要使用的数据集为PLUME（提供摘要与邮件写作的多种用户风格偏好），以及两大开源模型Qwen2.5‑7B‑Instruct与Llama3.1‑8B‑Instruct，用以生成与评估文本。

**📈 对比分析**

比较方法：在提取任务和PLUME任务分别评估trait expression与coherence；结果显示：大多数风格在提取任务中可表达但在PLUME迁移时明显下降；多向量组合导致表达至少下降15%，但层20的单位归一化方法保持最高连贯性。

**⚠️ 局限性**

局限性包括：①需要针对每个向量和组合手动调节层和α参数；②仅在开源模型上验证，未知是否适用于更大模型；③评估依赖LLM‑judge，可能与人类评判存在差距；④多向量组合时仍需额外调参以平衡表达与连贯性。

---

## 277. AI Virtue: What is "Good" Knowledge in the Age of Artificial Intelligence?

**arXiv ID:** 2607.01776 | [PDF](https://arxiv.org/pdf/2607.01776v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 278. Expander Sparse Autoencoders: Parameter-Efficient Dictionaries for Mechanistic Interpretability

**arXiv ID:** 2607.01799 | [PDF](https://arxiv.org/pdf/2607.01799v1)

**作者:** Rodrigo Mendoza-Smith `[一作]` `[通讯]` (Independent Researcher), Rodrigo Mendoza-Smith (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Expander Sparse Autoencoder（Expander SAE），通过将解码器列限制在左 d-regular expander 图的稀疏掩码上，显著降低参数量，同时保持稀疏编码维度，研究其在大型语言模型残差流激活中的可解释性与重构质量。

**💡 创新点**

创新点包括：①将组合压缩感知中的 expander 图结构引入 SAEs，首次实现参数高效的稀疏解码；②在 expander + 列平坦性条件下证明 k-稀疏代码的唯一可识别性；③展示可调节的 d 参数在存储与重构精度之间绘制出光滑的前沿。

**🔧 技术方法**

使用技术包括：TopK 稀疏非线性、共享权重的解码器、左 d-regular expander 掩码、Orthogonal Matching Pursuit (OMP) 迭代解码、GPU 并行实现、RIP-1/expander 理论分析。

**📊 数据集**

使用数据集：Pythia-70M/160M、Qwen2.5-3B、Llama-3.2-1B 的残差流激活，训练与评估样本取自 Pile 数据集。

**📈 对比分析**

与标准 Dense-SAE、匹配参数的 Dense-SAE、Clustered-sparse 等基线比较，测量重构误差、CE 恢复率、死特征率。结果显示在 d=7 时参数压缩 293×，CE 恢复 84%，存储–精度前沿平滑；使用 OMP 可显著缩小训练编码器的 amortization gap。

**⚠️ 局限性**

限制在于：理论证明是最坏情况，实际性能受训练质量与 expander 结构的影响；迭代 OMP 解码速度慢，在线部署仍需训练编码器；实验仅覆盖残差流层面，未在更大模型或其他任务中验证。

---

## 279. Lightweight Safe Reinforcement Learning for End-to-End UAV Navigation

**arXiv ID:** 2607.01794 | [PDF](https://arxiv.org/pdf/2607.01794v1)

**作者:** Shenghui Zhang `[一作]`, Hechang Chen `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种安全约束感知-控制集成框架，利用稀疏 LiDAR 感知实现无人机在复杂环境中的自主导航。

**💡 创新点**

创新点包括：轻量化稀疏 LiDAR 编码器（非对称 + 深度可分卷积）实现高效碰撞风险特征提取；在 CMDP 下采用 Lagrangian 形式的 Safe PPO 与自适应 Lagrange multiplier；通过课程学习将低密度环境的特征迁移到高密度环境；使用 Beta 分布输出实现连续、边界化动作。

**🔧 技术方法**

使用技术：深度可分卷积、轻量化特征编码器、Beta 分布策略、层级控制、Safe PPO（Lagrangian relaxation）、Generalized Advantage Estimation、对数潜在距离奖励、Isaac Sim + OmniDrones 仿真、课程学习。

**📊 数据集**

使用数据集：在 NVIDIA Isaac Sim 中自行构建的 3D 森林仿真环境，随机生成 100、200、300 个圆柱障碍，LiDAR 观测为 1×36×4 的距离张量。

**📈 对比分析**

与 Vanilla PPO、PPO+DSC、PPO+DSC+Safe RL、SAC、TD3、PPO+GRU/LSTM 等方法对比。实验显示在 100/200/300 障碍下成功率分别为 0.968、0.953、0.945，显著优于其他方法；在 2–11 m/s 速度范围内成功率均保持 ≥ 0.94；参数量仅 143k，远低于 SAC/TD3 等传统方法。

**⚠️ 局限性**

限制：仍未在真实无人机上验证；对传感器噪声鲁棒性探究不足；极高密度或极端动态场景下性能仍有下降；课程学习需要手动设置环境迁移策略。

---

## 280. EPnG: Adaptive Expert Prune-and-Grow for Parameter-Efficient MoE Fine-tuning

**arXiv ID:** 2607.01789 | [PDF](https://arxiv.org/pdf/2607.01789v1)

**作者:** Ahin Lee `[一作]` (Ulsan National Institute of Science & Technology), Taesik Gong `[通讯]` (Ulsan National Institute of Science & Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对Mixture-of-Experts（MoE）模型的参数高效微调框架 EPnG，利用路由器门概率动态评估专家重要性，按需剪枝低重要专家并扩增高重要专家的 LoRA 参数，从而在固定参数预算下实现高效微调。

**💡 创新点**

创新点在于将专家使用率信息直接映射到 LoRA 参数分配，通过循环的 prune‑and‑grow 机制动态重塑参数布局，使微调与 MoE 路由动态紧密对齐，显著提升了参数利用率和性能。

**🔧 技术方法**

技术方法包括：LoRA 低秩适配器、MoE 路由器门概率采样、专家重要性分数计算、基于阈值的剪枝与增量 Rank 扩展、正交初始化、指数衰减门统计以及 warm‑up 阶段的统计收集。

**📊 数据集**

实验使用的评估数据集包括数学推理的 MetaMathQA、GSM8K、MATH；代码生成的 Code Alpaca、HumanEval、MBPP；个性化对话的 PrefEval；以及通用能力评估的 ARC‑C、ARC‑E、BoolQ。

**📈 对比分析**

与全微调（FFT）、静态 LoRA、ESFT 等基线在 OLMoE 与 Qwen1.5‑MoE 上进行对比，EPnG 在仅更新约0.5%–0.7%参数的情况下，性能与全微调相当且优于静态 LoRA；与 ESFT 相比参数量约 29 倍更少，性能相近。

**⚠️ 局限性**

局限性包括：剪枝操作不可逆，可能错过后期变得重要的专家；引入了多余的超参数（剪枝率、增长率、warm‑up 步数、更新间隔等），需要额外调优；以及需额外维护路由器统计信息，增加训练复杂度。

---

## 281. LLM-Empowered Multimodal Fusion Framework for Autonomous Driving: Semantic Enhancement and Channel-Adaptive Design

**arXiv ID:** 2607.01772 | [PDF](https://arxiv.org/pdf/2607.01772v1)

**作者:** Wen Wang `[一作]` (Pengcheng Laboratory), Shuguang Cui `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 LM‑SCIP 框架，利用大语言模型实现通道感知的多模态融合，支持基础设施中心的协同感知任务（定位、轨迹预测、图像重建）。

**💡 创新点**

创新点包括：①将融合提升至语义层并将通道质量作为语义推理的上下文；②通过 CASM 把链路指标映射为 Channel Prompt，动态门控雷达特征；③使用 LoRA 微调的 LLM 与 H‑MoE 进行任务专用推理与多任务解码。

**🔧 技术方法**

采用的大型语言模型（LoRA 细调的 GPT‑2）、Heterogeneous Mixture‑of‑Experts（H‑MoE）、多模态语义编码器（ViT + 复杂 CNN）、Channel‑Adaptive Semantic Module（CASM）以及解耦多任务解码器。

**📊 数据集**

实验数据集包括 nuScenes（定位/轨迹评估）和 VIRAT（定位、轨迹与图像重建评估）。

**📈 对比分析**

与 vision‑only、无 CASM、无 H‑MoE 等基线进行对比；在 nuScenes 的雷达开启场景中定位 RMSE 下降 40%；在 VIRAT 上 minFDE_1 为 0.179 m、定位 RMSE 为 0.214 m；在低 SNR 情况下 CASM 能保持较低误差；整体性能优于 CenterFusion、CRAFT 等现有融合方法。

**⚠️ 局限性**

局限性在于依赖链路侧信息的准确性、需要精确时空对齐、实验多使用合成雷达波形和模拟噪声，缺少真实 V2X 传输评估；以及对多车协同推理的探索仍有限。

---

## 282. Path-level Hindsight Instructions for Semantic Exploration in Vision-Language Navigation

**arXiv ID:** 2607.01754 | [PDF](https://arxiv.org/pdf/2607.01754v1)

**作者:** Sung June Kim `[一作]` (Korea University), Honglak Lee `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Φ-Nav框架，通过路径层面的后向指令生成，将探索轨迹与语言对齐，弥补了传统上策略训练中的语义监督缺失；

**💡 创新点**

创新点在于将大规模视觉语言模型用于后向指令生成，并引入专家情境学习与轨迹-指令对齐权重机制，实现了自监督的语义强化；

**🔧 技术方法**

利用预训练的视觉语言模型（如Qwen2.5-VL-7B）生成后向指令，结合专家-情境学习、轨迹-指令对齐评分以及双重监督（专家行为+后向模仿）进行训练；

**📊 数据集**

在R2R-CE和RxR-CE两个VLN基准数据集上进行实验；

**📈 对比分析**

与DAgger、Scheduled Sampling等基线对比，Φ-Nav在验证集上提升SR、SPL等指标（如在R2R-CE val‑unseen上SR提升约5%），且在样本效率上显著优于传统方法；

**⚠️ 局限性**

局限在于后向指令生成仍存在幻觉与语义偏差，且轨迹-指令对齐权重的估计方法仍可进一步提升。

---

## 283. MedStreamBench: A Time-Aware Benchmark for Streaming and Proactive Medical Video Understanding

**arXiv ID:** 2607.01751 | [PDF](https://arxiv.org/pdf/2607.01751v1)

**作者:** Yuan Wang `[一作]` (Zhejiang University), Zuozhu Liu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了MedStreamBench，一套基于时间窗口的医学视频问答基准；

**💡 创新点**

创新点在于引入时间约束的前缀推理和主动式报警任务，统一了回顾、当前、未来与主动四种时序模式；

**🔧 技术方法**

使用统一的时间感知QA schema、帧采样与Prompt构造技术，以及AI辅助评判与人类审核的混合评价流程；

**📊 数据集**

整合22个多源医学视频与序列数据集（包括腹腔镜、机器人手术、胃肠内镜、胶囊内镜等），共收集5,419个QA条目；

**📈 对比分析**

通过与Gemini‑2.5‑Pro、Qwen3‑VL‑4B/8B、InternVL3.5‑8B等多种闭源与开源模型对比，整体得分最高的是Gemini‑2.5‑Pro，开源模型中InternVL3.5‑8B表现最好；

**⚠️ 局限性**

局限性包括源数据标注不一致、部分样本来源于弱监督或模型挖掘、固定帧采样可能遗漏短时事件、以及基准整体受源数据分布偏差影响。

---

## 284. Finite-Lag Operator Geometry of Recurrent Representations

**arXiv ID:** 2607.01746 | [PDF](https://arxiv.org/pdf/2607.01746v1)

**作者:** Kanishka Reddy `[一作]` `[通讯]` (University of Washington), Kanishka Reddy (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于源–后继对的有限滞后运算符几何，定义了源中心传输张量 GΔ 和坐标循环 𝒲Δ^ρ，用来度量递归表示在固定步长内的运动与方向。

**💡 创新点**

创新点在于：① 用有限滞后传输律代替传统静态对称核；② 将 GΔ 精确分解为条件扩散与确定性位移；③ 证明仿射协变性、密集高斯估计器的 Lipschitz 稳定性以及有限滞后分离定理；④ 给出线性高斯闭式校准，揭示创新噪声、更新矩阵与方向循环的数学关系。

**🔧 技术方法**

技术手段包括：高斯源平滑估计器、源中心二阶矩计算、坐标循环的反对称矩阵、线性高斯闭式分析、对比实验（controlled linear-Gaussian 及 repeat-copy RNNs），以及理论证明。

**📊 数据集**

数据集主要为受控的线性高斯模拟和在 repeat-copy 任务上训练的 Elman、GRU、LSTM 网络，未使用公开的大规模现实数据集。

**📈 对比分析**

比较方法：与传统静态扩散几何和转移算子方法对齐，通过统计量（tr(G)、条件扩散、可持续位移、循环）在不同架构之间对比。实验显示 Elman 在传输规模和可持续位移上显著高于 GRU/LSTM；在受控实验中误差低于 1%。

**⚠️ 局限性**

局限性：量纲/度量敏感，分数与循环随核宽度和坐标归一化变化；依赖高斯估计器的带宽选择；仅研究单步滞后，未扩展到更长或多步；未验证在更复杂任务或大规模数据集上的泛化。

---

## 285. InterCMDM: Block-Causal Diffusion for Autoregressive Human Interaction Generation

**arXiv ID:** 2607.01743 | [PDF](https://arxiv.org/pdf/2607.01743v1)

**作者:** Qing Yu `[一作]` (LY Corporation), Kent Fujiwara `[通讯]` (LY Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于块级因果扩散的两人交互动作生成框架 InterCMDM

**💡 创新点**

创新点包括：双流因果扩散 Transformer、统一多任务块注意力掩码实现可控交互模式，以及块级扩散目标实现长时序稳定回放

**🔧 技术方法**

采用临时运动 VAE、双流因果扩散 Transformer、块级注意力掩码、扩散时间步调度以及文本提示的 DistilBERT 编码

**📊 数据集**

在 InterHuman（AMASS 22-joint）和 Inter-X（SMPL-X 56-joint）两个文本到交互动作的基准数据集上训练与评估

**📈 对比分析**

与现有扩散、掩码建模与自回归方法对比，InterCMDM 在 R‑Precision、MM‑Dist、FID 等指标均超越所有全生成基线，并在长时序生成中显著降低边界不连贯与漂移

**⚠️ 局限性**

局限性：仅针对双人交互，未显式约束物理碰撞与接触；掩码模式手工预设，缺少自学习交互模式；文本条件仅为整体句子嵌入，缺乏细粒度 per‑person 控制

---

## 286. The Grammar Does the Work: Functional vs. Lexical Dependency Length Minimization Across Universal Dependencies

**arXiv ID:** 2607.01899 | [PDF](https://arxiv.org/pdf/2607.01899v1)

**作者:** Kim Gerdes `[一作]` `[通讯]` (Universite Paris-Saclay), Kim Gerdes (Universite Paris-Saclay)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了依存长度最小化（DLM）在功能性依存与词汇性依存之间的差异，并在122种语言中系统量化两类的平均依存长度和优化程度。

**💡 创新点**

提出了“两层 DLM”理论：功能依存由语法规则强制局部化，词汇依存受处理压力软约束；并在跨语言层面首次证明此二分法的普遍性。

**🔧 技术方法**

使用统计方法（MDD、随机基线、优化比 OR、Wilcoxon 检验、线性混合效应模型等），并在 UD 与 SUD 注释框架下进行交叉验证。

**📊 数据集**

数据集为 UD v2.17 与对应的 SUD 版本，过滤后共 122 种语言，约 798,381 句子、11.2M 非标点依存标记。

**📈 对比分析**

通过与随机线性位置打乱的基线比较以及 UD 与 SUD 的双框架对比，功能 MDD 平均为 1.71（OR 0.28），词汇 MDD 平均为 2.87（OR 0.46），功能‑词汇差距在所有语言中显著且稳健。

**⚠️ 局限性**

局限包括树库规模与体裁差异、标注方案可能影响结果、词性边界理论依赖、未考虑语义/语用层面的影响以及对单一树库的泛化性不足。

---

## 287. PairCoder++: Pair Programming as a Universal Paradigm for Verified Code-Driven Multimodal and Structured-Artifact Generation

**arXiv ID:** 2607.01883 | [PDF](https://arxiv.org/pdf/2607.01883v1)

**作者:** Junhao Chen `[一作]` (Tsinghua University), Hao Zhao `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PairCoder 两代理框架，驱动者 (Driver) 负责生成和改写程序，导航者 (Navigator) 基于工具链的编译/渲染/模拟结果进行验证并给出具体修改意见，错误触发时两者角色互换，以实现代码生成的可验证闭环。

**💡 创新点**

创新点在于：
1) 将人类 Pair Programming 概念迁移至 LLM 代码生成，并通过工具链验证来保证可执行性；
2) 设计验证证据驱动的审查流程，使 Navigator 能够基于编译器报错、执行结果或渲染对比给出可操作的反馈；
3) 引入错误触发的角色切换策略，让诊断者直接修复错误；
4) 在多模态（图表、SVG、CAD、3D 场景等）任务上系统化评估该框架。

**🔧 技术方法**

技术手段包括：
- 大语言模型（多型号、不同供应商）与角色特定提示、Self‑Mirror 机制；
- 迭代的 Driver‑Navigator 交互循环；
- 统一的验证器接口，集成编译器、执行器、渲染器、单元测试等工具；
- 错误阈值触发的角色切换策略；
- 以 token 计数与时间为度量的成本分析。

**📊 数据集**

使用了 17 个公开基准，包括程序合成（LiveCodeBench、BigCodeBench、DS‑1000）、多语种代码（HumanEval‑X C++/Java/JS）、数据科学（Plot2Code、PandasPlotBench、ChartMimic）、网页（WebApp1K）、硬件（VerilogEval、RTLLM）、图形（DaTikZ、StarVector、GenCAD‑Code、3DCodeBench）、参数化 CAD（P3D‑Bench）等，覆盖 7 种 LLM 模型与 3 供应商。

**📈 对比分析**

与单模型直接生成（baseline）进行对比，采用官方完整指标（pass@1、执行率、编译率、SSIM、CLIP、DINO、Chamfer 等）。实验显示：在 7 种模型中，PairCoder 在 44/48 可验证任务中获益，主要提升集中在工具链提供丰富反馈的任务（如 Blender 场景可执行率从 0.20 提升至 0.78，TikZ 编译率提升 10–30 分）。成本方面，Token 消耗为单模型的 2.9–9.2 倍，平均约 7 倍。

**⚠️ 局限性**

局限性：
1) 成本显著增加，Token 与推理时间约为单模型的 7 倍，限制了实时或大规模部署；
2) 当工具链验证能力弱或不存在时，PairCoder 的优势减弱甚至无效；
3) 仍需手工调参（如角色切换阈值、验证器实现）和对多模态任务的细粒度适配。

---

## 288. SAB-LVLM: Significance-Aware Binarization for Large Vision-Language Models

**arXiv ID:** 2607.01876 | [PDF](https://arxiv.org/pdf/2607.01876v1)

**作者:** Qi Lyu `[一作]` (Chinese Academy of Sciences), Zhi Han `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了针对大规模视觉-语言模型的 Significance-Aware Binarization（SAB-LVLM）框架，将权重量化为近乎 1-bit，显著降低存储与计算开销；

**💡 创新点**

核心创新包括：1）基于多模态校准数据构建空间显著性图（区分单模态与多模态激活权重）；2）模态引导显著性融合策略，动态生成显著性加权映射；3）结合显著性加权的交替更新算法，实现精细的二值化拟合；

**🔧 技术方法**

主要技术手段包括：Hessian 逆矩阵估计权重敏感度、二值化参数分解、交替优化的显著性加权更新、模态整合分数 r 的计算；

**📊 数据集**

使用 COCO 2017 作为校准数据集；在 MMStar、DocVQA、TextVQA、Video-MME、VSI-Bench 等五大多模态基准上进行评估；

**📈 对比分析**

与 PB-LLM、BiLLM、ARB-LLM 等一比特 PTQ 方法以及 3-bit GPTQ 进行对比。实验显示，SAB-LVLM 在大多数任务上均显著优于对照方法，性能几乎与全精度模型持平，尤其在 MMStar、DocVQA、VSI-Bench 等任务中提升幅度可达 5–10 分；

**⚠️ 局限性**

局限性包括：1）依赖手工设定阈值 τ，阈值对结果影响敏感；2）仅在 Qwen2.5-VL 与 InternVL3.5 系列模型验证，未覆盖更大或不同结构的 LVLM；3）仅实现 1-bit 量化，未探索更细粒度混合精度方案；4）对长文本或长视频输入的鲁棒性尚未系统评估。

---

## 289. QWERTY: Training-Free Motion Control via Query-Warped Video Diffusion Transformers

**arXiv ID:** 2607.01869 | [PDF](https://arxiv.org/pdf/2607.01869v1)

**作者:** Kyobin Choo `[一作]` (Yonsei University), Seong Jae Hwang `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的图像到视频扩散框架，通过在推理时对视频扩散 Transformer（DiT）的查询（query）进行扭曲，实现用户定义的对象与相机运动控制。

**💡 创新点**

创新点在于仅对 3D 全注意力中的查询进行空间扭曲，并通过语义-时序通道分解（STCD）保留帧一致的语义子空间；利用查询扭曲产生的噪声作为自引导进行潜在优化，从而在保持预训练模型泛化的前提下实现高质量的运动控制。

**🔧 技术方法**

采用的视频技术包括视频扩散 Transformer（DiT）、3D Rotary Positional Embedding（RoPE）、PCA 基于通道重要性的评估、光流驱动的查询扭曲、以及潜在优化与相位一致性约束。

**📊 数据集**

实验使用 VIPSeg 数据集评估对象运动控制，DL3DV 数据集评估相机运动控制；基准模型为 Wan 2.2 TI2V‑5B 与 CogVideoX‑I2V‑5B。

**📈 对比分析**

与 U‑Net 基础的训练无关方法（SG‑I2V、MOFT、FreeTraj 等）以及基于 DiT 的 Fine‑tuned GWTF 进行对比；实验表明该方法在 FID、FVD、FTD 等指标上与 Fine‑tuned 方法相当或更优，同时保持更高的视频一致性与控制精度。

**⚠️ 局限性**

局限性包括对光流估计精度的依赖、在极端运动或复杂场景下可能失效、查询扭曲仅适用于帧一致的语义子空间，难以处理剧烈形变或大范围摄像机运动。

---

## 290. Has This Checkpoint Been Abliterated? A Two-Signal Audit and Its Failure Map

**arXiv ID:** 2607.01854 | [PDF](https://arxiv.org/pdf/2607.01854v1)

**作者:** Gabriel Hurtado `[一作]` `[通讯]` (Moonsong Labs), Gabriel Hurtado (Moonsong Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出一种基于参考锚定的激活拒绝间隙与低秩权重差异能量两种互补内部信号的无阈值检查点审计方法，用以在模型部署前判断公开的去审查（abliteration）模型是否已移除拒绝机制。

**💡 创新点**

创新点在于将激活间隙与权重能量通过 z-标准化后求和得到阈值无关的分数，显著提升检测 AUROC，并实现跨模型族的迁移；同时系统性绘制了审计失效场景，揭示其局限性。

**🔧 技术方法**

采用参考锚定激活分数、奇异值分解提取的低秩权重能量（WeightWatch）、z-标准化、Youden 指数阈值设定、留一族外评估以及白盒对抗训练演示等技术手段。

**📊 数据集**

使用了 273 个检查点的注册表，其中包含 57 个公开 abliteration（覆盖 Qwen、DeepSeek Qwen、Llama、Gemma）和 37 个良性微调/合并/指令微调样本作为负例。

**📈 对比分析**

在与单一激活间隙、单一权重能量以及 AMS Tier-1/2 基线对比时，z-求和在内样本 AUROC 0.95、留一族外平衡准确率 0.89，阈值化后 FPR 仅 0.11，明显优于任何单一信号。

**⚠️ 局限性**

主要限制包括：审计强度完全依赖于可信参考基线；权重能量无法区分拒绝移除与其他低秩编辑；白盒攻击可训练模型逃避审计；样本量有限，跨族泛化仅通过留一族外估计；多方向或链路思考式拒绝移除仍可能逃逸。

---

## 291. Adaptive Group-Based Counterfactual Explanations for Time-Series Rehabilitation Data

**arXiv ID:** 2607.01838 | [PDF](https://arxiv.org/pdf/2607.01838v1)

**作者:** Emmanuel C. Chukwu `[一作]` (Eindhoven University of Technology), Mykola Pechenizkiy `[通讯]` (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一个适用于高维IMU时间序列的基于组的反事实解释框架Adaptive-MO，结合Shapley基组排序和可学习门控，生成稀疏且符合临床肌肉/关节层面解释的反事实；

**💡 创新点**

创新点包括：①引入可学习门控实现动态组级稀疏；②将Shapley组排序与多目标优化相结合；③在康复运动分析中实现从肌肉/关节层面的可解释反事实；

**🔧 技术方法**

技术手段主要包括GradientSHAP求取组重要性、可学习sigmoid门控、四目标损失（有效性、稀疏性、平滑性、门控正则）、M-CELS基线对比以及FCN分类器；

**📊 数据集**

使用的数据集是KneE-PAD膝关节康复数据集，包含三种练习的IMU信号（8个传感器、48通道）、31名患者；

**📈 对比分析**

在与M-CELS及多种SA/LG变体的对比实验中，LG-SHAP pruned获得最高成功率94.7%（相较90% M-CELS），组稀疏度下降27%，生成时间与M-CELS相当甚至略快，且在各练习上均优于基线；

**⚠️ 局限性**

局限性包括仅在KneE-PAD小规模数据集验证；手工制定的组划分仅适用于下肢IMU；未进行临床用户研究；仅用离线指标评估，缺乏实时、多模态扩展。

---

## 292. RT-Tango: Real-Time Distributed Binaural Speech Enhancement for Low-Power Hearing Aid Devices

**arXiv ID:** 2607.01834 | [PDF](https://arxiv.org/pdf/2607.01834v1)

**作者:** Z. Benslimane `[一作]`, R. Serizel `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计并实现了RT‑Tango，一种针对助听器的实时分布式双耳语音增强框架，能够在极低延迟（8 ms）和极低算力的条件下完成双耳语音增强。

**💡 创新点**

创新点在于：①采用感知驱动的ERB特征压缩、分组递归网络（GRNN）和时间稀疏化技术，以显著降低模型复杂度；②利用非对称STFT将频率分辨率与算法延迟解耦；③构建两阶段分布式架构，分别在本地和跨耳节点进行掩码估计与多通道滤波，实现低通信成本。

**🔧 技术方法**

主要技术包括ERB滤波器组、分组RNN（GRNN）、跳帧（Fixed‑Rate Skipping）和学习型跳帧门控、在线空间协方差矩阵递归更新、异步STFT、SDW‑MWF滤波器、以及全因果的递归推理。

**📊 数据集**

训练使用基于Monir等人的模拟双耳数据集，采用LibriSpeech与语音形噪、真实环境噪声混合；评估数据来自公开的BinauRec数据集（1 200个混合、测量的房间冲激响应）。

**📈 对比分析**

与CNN版Tango、Tango‑RNN、轻量化GTCRN等对比，RT‑Tango在SI‑SDR、SI‑SIR、SI‑SAR、PESQ、STOI等指标上保持竞争力，同时MACs大幅降低，算法延迟仅8 ms，显示出优异的性能与资源效率。

**⚠️ 局限性**

局限性在于：评估主要基于室内模拟与测量数据，缺乏对更复杂多场景、非同步或通信误差场景的验证；对更多麦克风配置与更高噪声环境的适应性尚未深入探讨。

---

## 293. Many Voices, One Reward: Multi-Role Rubric Generation for LLM Judging and Reward Modeling

**arXiv ID:** 2607.01830 | [PDF](https://arxiv.org/pdf/2607.01830v1)

**作者:** Dazhi Fu `[一作]` (Chinese University of Hong Kong), Jicong Fan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多角色评判表生成框架 MRRG，以提升 LLM 评判和奖励模型的多维度质量信号。

**💡 创新点**

通过从用户、领域专家、教师、AI 研究者和语言学家等多元角色分别生成评判标准，并合并去重，克服了单一视角产生的维度盲区。

**🔧 技术方法**

利用多角色提示、无监督表生成、确定性去重与 rubric‑based scorer 进行偏好验证和 RLVR 奖励建模。

**📊 数据集**

在 RewardBench‑2、JudgeBench、PPE 以及 BiGGen‑Bench、HealthBench‑Hard 等公开基准上进行实验。

**📈 对比分析**

在所有评测上均优于单声评判生成器和其他基线，平均提升 3–16 个百分点；在 RLVR 中，MRRG 在 BiGGen‑Bench 提升 1.7 分，在 HealthBench‑Hard 提升 3.4 分。

**⚠️ 局限性**

主要局限是角色池固定，无法自适应问题需求，可能导致对部分任务缺少关键视角。

---

## 294. Pre-Flight: A Benchmark for Evaluating Large Language Models on Aviation Operational Knowledge

**arXiv ID:** 2607.01829 | [PDF](https://arxiv.org/pdf/2607.01829v1)

**作者:** Alex Brooker `[一作]` (Airside Labs), Tim Hughes `[通讯]` (Mahino Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了Pre‑Flight基准，评估航空业务操作相关LLM在法规与地面操作知识上的表现。

**💡 创新点**

首创公开多选题集，专门聚焦国际机场地面操作、ICAO/FAA法规和航空常识，填补航空领域缺乏专用LLM评测资源的空白。

**🔧 技术方法**

使用Inspect评估框架、零样本多选模板、量化本地部署模型，并维护持续滚动的leaderboard进行性能跟踪。

**📊 数据集**

构建300道题目，来源于ICAO Annex、FAA 14 CFR、国际机场操作手册以及航空常识文档等权威资料。

**📈 对比分析**

采用准确率与约95%专家基准对比，最高模型达82.7%，开放权重模型可达77%，与专家水平仍差约12个百分点，提升速度缓慢。

**⚠️ 局限性**

局限包括多选形式仅评估识别能力、公共集可能受训练数据污染、专家基准非正式、类别不均衡导致评估噪声以及缺乏生成与决策性任务。

---

## 295. MMBench-Live: A Continuously Evolving Benchmark for Multimodal Models

**arXiv ID:** 2607.01813 | [PDF](https://arxiv.org/pdf/2607.01813v1)

**作者:** Yuanzhi Liu `[一作]` (Beijing University of Posts and Telecommunications), Zhanyu Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了可持续演进的多模态评测基准MMBench-Live，利用多智能体自动化流水线自动生成并验证新的评测实例；

**💡 创新点**

创新点在于将评测基准演进视为任务导向的数据构建过程，结合视觉模式识别、反馈控制查询优化与可执行推理验证，实现分布一致且低成本的动态更新；

**🔧 技术方法**

采用了多智能体框架、任务结构化描述、视觉模式提取、反馈控制查询优化、可执行推理验证、LLM（如GPT-5 Mini、Gemini-3-Flash）、以及多种工具（VisionReasoner、Depth Anything、LLaVA、OCR等）；

**📊 数据集**

以MMBench原始dev集为基础，生成5.9K新的QA实例，数据来源包括Google图像API、Flickr、特定网站爬取等；

**📈 对比分析**

通过在多款VLM（DeepSeek‑VL、InstructBLIP、LLaVA、mPLUG‑Owl2、Qwen3、Qwen2.5）上进行评测，结果显示模型排名基本保持稳定，Qwen2.5-VL表现最佳；相较原MMBench，MMBench‑Live在数据污染（PaCoST）信号上更弱；

**⚠️ 局限性**

仍存在高频视觉概念的隐式记忆、自动实例质量受限于基础模型、以及任务扩展受限于任务导向框架，难以引入全新任务类型。

---

## 296. TO-Master: an LLM-agent framework for automated topology optimization

**arXiv ID:** 2607.01812 | [PDF](https://arxiv.org/pdf/2607.01812v1)

**作者:** Haoju Lin `[一作]` (Hong Kong University of Science and Technology), Tianju Xue `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 TO-Master，一套基于大型语言模型（LLM）的智能代理框架，能够将自然语言指令与有限元求解、网格生成、边界条件检查、敏感性分析、迭代优化等计算工具串联，实现全流程无代码的拓扑优化；

**💡 创新点**

创新点在于：① 将 LLM 代理与确定性有限元/优化工具进行结构化协同；② 通过工具使用规则、链式推理指导和少量示例实现对模糊指令的高鲁棒性解释；③ 设计了用户交互式确认与可视化检查环节，提升公式化可靠性；④ 支持多种几何来源（矩形、Gmsh、图片重建）、多负载、应力约束、热传导等不同物理与约束；

**🔧 技术方法**

使用技术包括：大型语言模型 + ReAct 风格代理；结构化工具接口（MCP）与链式思考提示；JAX‑FEM 差分求解器；SIMP 密度方法、过滤与投影；方法移动极限（MMA）优化；自动微分求敏感性；热传导、应力约束、p‑norm 聚合等；

**📊 数据集**

主要使用的测试案例包括经典基准（2D/3D cantilever、bridge、tower、L‑shape、T‑shape）、热传导基准、工业几何（机翼轴承、发动机支架）以及 2D 图像转 3D 结构的案例；未使用公开大规模数据集，侧重手工构造的基准与工程案例；

**📈 对比分析**

通过与手工设定的经典基准结果对比验证了算法的正确性；在模糊指令的消融实验中，完整指令下成功率达到96.7%，去除工具约束后仅 40%；对不同物理类型和几何复杂度的案例均能得到符合预期的拓扑与场分布，表明框架在多种场景下具有可用性；

**⚠️ 局限性**

限制包括：仅支持预设工具集，无法直接进行线性求解；需要用户进行可视化确认和人工检查；消融实验仅覆盖六个基准，未对工业级大规模案例进行验证；单位系统需自洽且手动规范；对复杂约束或自定义目标的扩展仍需进一步工作。

---

## 297. Rank-Then-Act: Reward-Free Control from Frame-Order Progress

**arXiv ID:** 2607.01897 | [PDF](https://arxiv.org/pdf/2607.01897v1)

**作者:** Yuriy Maksyuta `[一作]` (T Tech), Daniil Gavrilov `[通讯]` (T Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Rank-Then-Act (RTA)，一种两阶段、无环境奖励的视觉‑语言模型（VLM）学习框架，先用专家视频训练 VLM 预测帧的进度排名，再在控制阶段用帧排名与时间的 Spearman 相关系数作为奖励训练策略。

**💡 创新点**

创新点在于：① 用列表式 GRPO 损失训练 VLM 成为“进度排序器”，避免了后序总优的短路；② 采用 Spearman 相关系数作为完全无量纲、区间 [-1,1] 的奖励，消除了对绝对进度尺度的依赖；③ 通过单一预训练的进度评分器即可跨任务、跨环境迁移，无需对每个任务调参或额外奖励。

**🔧 技术方法**

主要技术包括：视觉‑语言模型（如 Qwen2.5‑VL‑7B 或简化 MLP）、Group Relative Policy Optimization (GRPO) 进行列表式排序训练、Spearman 相关奖励计算、PPO/DrQv2 强化学习、窗口化奖励采样与平均。

**📊 数据集**

数据集：专家视频来自 Retro 游戏 Catrap、Kirby；连续任务 PointMaze、MetaWorld；额外的 YouTube GameBoy 演示与 COIN AssembleSofa 视频用于评估跨域泛化。

**📈 对比分析**

对比方法包括 Rank2Reward、VLM‑RM、GVL（含 Gemini 变体）、Oracle 奖励等。实验表明 RTA 在 Catrap 2/4/6 及 Kirby 的成功率均优于或等于基线；在连续任务中 RTA 在 UMaze、MetaWorld 上也表现最好；单一进度评分器能在多任务上共享，显示良好跨域性能。

**⚠️ 局限性**

局限性：① 需要足够多样的专家视频，覆盖不足会导致特定关卡过拟合；② 相关奖励对窗口大小敏感，过小窗口易产生局部一致误报，过大窗口信号稀释；③ VLM 推理延迟较高，影响实时性；④ 对高度非单调进度任务的鲁棒性有限，需多尺度或层次化改进。

---

## 298. Diversity-aware View Partitioning for Scalable VGGT

**arXiv ID:** 2607.01885 | [PDF](https://arxiv.org/pdf/2607.01885v1)

**作者:** Jinsoo Park `[一作]` (POSTECH), Jeany Son `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于多视角几何变换器（VGGT）的多视角分区方法，通过将视角组织成多样性意识的平衡块来提高重建质量和计算效率。

**💡 创新点**

创新点在于提出了一种训练无关的、即插即用的框架，通过组合图优化方法来实现视角的多样性分区，从而减少冗余视角的影响。

**🔧 技术方法**

使用了组合图优化和软姿态传播策略来估计空间关系，结合了视觉相似性来进行视角分区。

**📊 数据集**

在多个数据集上进行了实验，包括7Scenes、NRGBD、Bonn、ScanNet-50、TUM-RGBD和Tanks&Temples等。

**📈 对比分析**

与现有的VGGT变体相比，提出的方法在相机姿态估计、多视角深度预测和3D重建中表现出更好的性能，同时减少了内存使用和推理延迟。

**⚠️ 局限性**

限制在于该方法依赖于初始的视觉相似性估计，可能在某些情况下对空间关系的近似不够准确。

---

## 299. An Exploratory Study on LLM-Generated Code and Comments in Code Repositories

**arXiv ID:** 2607.01867 | [PDF](https://arxiv.org/pdf/2607.01867v1)

**作者:** Yongyi Ji `[一作]` (University of Leicester), Hongji Yang `[通讯]` (University of Leicester)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用多种零射方法检测软件仓库中代码和注释是否由大型语言模型生成，并分析其比例、特征以及与企业与社区维护仓库的差异，同时评估与已知漏洞的关联。

**💡 创新点**

首次在真实工业与社区开源项目中系统性应用现有 LLM 生成内容检测器进行代理分析，并将检测结果与代码克隆、漏洞数据相结合，揭示了 LLM 生成代码的普遍位置与风险。

**🔧 技术方法**

Binoculars、Log‑Likelihood、Entropy、Rank、Log‑Rank、LRR、DetectGPT、Fast‑DetectGPT、DetectCodeGPT 等零射检测器。

**📊 数据集**

8 个活跃的企业与社区仓库（2021‑2025 年）作为目标数据；AISE 数据集用于阈值校准；PreciseBugs 数据集用于漏洞关联分析。

**📈 对比分析**

通过基准阈值、±20% 变动的灵敏度分析以及多检测器交叉验证，评估 LLM 生成代码/注释的比例。结果显示：代码生成比例随时间下降，注释比例保持稳定；企业仓库的比例高于社区；与已知漏洞关联比例低（约 10%）。

**⚠️ 局限性**

检测器对 LLM 生成内容的敏感度随模型演进可能下降；阈值缺乏真值标签导致误判；仅覆盖支持的编程语言，无法推广到所有开源/闭源项目；缺乏对人类重写或改写后 LLM 代码的检测。

---

## 300. DL-SLAM: Enabling High-Fidelity Gaussian Splatting SLAM in Dynamic Environments based on Dual-Level Probability

**arXiv ID:** 2607.01860 | [PDF](https://arxiv.org/pdf/2607.01860v1)

**作者:** Ziheng Xu `[一作]`, Jianwei Niu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

DL‑SLAM 在动态环境下实现了高精度的 3D Gaussian Splatting SLAM，能够在保持静态地图完整性的同时有效处理瞬时静态物体。

**💡 创新点**

创新点在于双层概率框架：先在像素层通过语义+光流估计动态概率，再提升到 3D 对象层进行动态高斯裁剪，并通过反馈循环将静态地图信息回馈至像素层，最终实现无渲染伪影且跟踪精度提升。

**🔧 技术方法**

采用的技术包括：3D Gaussian Splatting、双层概率模型、光流与极限几何的动态概率计算、基于 Recognize Anything、Grounding DINO 与 MobileSAMv2 的开源语义分割、差分渲染、贝叶斯更新、动态概率反馈与语义标签精炼。

**📊 数据集**

实验使用的公开数据集为：TUM RGB‑D dynamic、BONN 以及 Wild‑SLAM iPhone。

**📈 对比分析**

与传统 ORB‑SLAM2、NeRF‑based Co‑SLAM、3DGS‑based SGS‑SLAM、DG‑SLAM、WildGS‑SLAM 等方法对比，DL‑SLAM 在跟踪 ATE RMSE 上最多提升 13%，渲染质量（PSNR、SSIM、LPIPS）均优于对手，同时保持实时跟踪与映射速度，GPU 内存占用仅略高于同类方法。

**⚠️ 局限性**

局限性包括：仍未显式建模动态物体的轨迹，极端纹理缺失区域的动态概率估计受限，且对 GPU 内存有一定需求。

---

## 301. Constructible Words Characterize Rational Languages of Words Indexed by Scattered Linear Orderings

**arXiv ID:** 2607.01858 | [PDF](https://arxiv.org/pdf/2607.01858v1)

**作者:** Thomas Braipson `[一作]` (University of Liège), Tom Clara `[通讯]` (University of Liège)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文研究了散列线性序（scattered linear ordering）上的有限状态自动机（automata on linear orderings），并证明了这些自动机接受的有理语言（rational languages）可以完全由其可构造词（constructible words）来表征。

**💡 创新点**

创新点在于：①引入了可构造词的概念，以有限符号表示无限或不可数长度的单词；②证明任何非空等价类至少包含一个可构造词，从而将有理语言的等价类与可构造词对应；③利用Colcombet的无穷阶乘分解森林定理（factorization forests theorem）以及Ramsey类分割技术，完成上述证明。

**🔧 技术方法**

主要技术包括：有限语义自动机的语义定义、等价关系的构造、半群（semigroup）理论、Colcombet的因子分解森林定理、Ramsey分割（Ramseyan split）与半群的乘法标签。

**📊 数据集**

本文为纯理论研究，没有使用任何实验数据集。

**📈 对比分析**

由于研究为理论证明，本文未进行实验对比，也没有涉及性能评估。

**⚠️ 局限性**

局限性：该结果仅适用于散列线性序；对非散列（完整）线性序的有理语言尚未给出对应的可构造词表征；同时，尽管可构造词提供了有限表示，但并未提供有效判定算法或复杂度分析。

---

## 302. FoundDP: Revisiting Weak Disparity Observability in Dual-Pixel Depth Estimation

**arXiv ID:** 2607.01900 | [PDF](https://arxiv.org/pdf/2607.01900v1)

**作者:** Fengchen He `[一作]` (Huazhong University of Science and Technology), Shaoqun Zeng `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一框架 FoundDP，利用双像素 (DP) 相机的局部视差信息提供物理量化深度，同时引入 monocular ViT 基础模型的全局结构先验，提升弱视差区域的深度质量。

**💡 创新点**

创新点：① 通过 ViT 特征对齐解决 DP 图像因散焦模糊导致的表示退化；② 设计三阶段模块（DP 估计、结构细化、深度引导）实现物理尺度与结构一致性的平衡；③ 在弱视差和下采样条件下显著提升结构保真度和度量精度。

**🔧 技术方法**

技术：基于 DP 的对称视差成本体积 + Softmin 回归；多尺度残差细化网络；ViT 先验提取与特征对齐；深度引导模块将 DP 估计作为空间条件融入 ViT 的解码过程；统一 Smooth L1 误差损失。

**📊 数据集**

使用合成 NYUData、真实 DP2020、DP5K、DP2019 以及自建的 DPDown70（下采样 70%）数据集进行训练与评估。

**📈 对比分析**

与 DPNet、DDDNet、SFBDNet、CADSNet 等现有 DP 方法比较，FoundDP 在 AI(1/2)、Rank 相关、δ<1.25/1.25² 等指标均实现最低误差、最高准确率，尤其在弱视差和下采样场景下优势显著，性能提升可达 10%+。

**⚠️ 局限性**

局限性：依赖 DP 视差观测，噪声或极端光学失真时性能会下降；ViT 编码器带来额外计算开销，实时部署受限；DP 与 ViT 先验的适配仍不完美，未来需要更高效的 DP‑专用表征学习。

---

## 303. Understanding Build Reproducibility in the F-Droid Ecosystem

**arXiv ID:** 2607.01890 | [PDF](https://arxiv.org/pdf/2607.01890v1)

**作者:** Denise Nanni `[一作]` (Télécom Paris), Gabriele d'Angelo `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 F‑Droid Android 应用生态中，对历史构建可重现性进行系统实验，分析随时间推移的重建成功率与位比可重现性，并调研导致失败的根本原因。

**💡 创新点**

首次在大规模历史 Android 应用上量化可重现性随时间衰退的程度，发现缺失依赖是主要瓶颈，并提出基于归档与环境固定的改进方案。

**🔧 技术方法**

使用 F‑Droid 构建服务器、Vagrant/云 VM 镜像、Diffoscope、Git、Python 脚本等技术完成重建、可重现性验证与日志分析。

**📊 数据集**

构建并分析了 80139 条版本记录与 18904 条历史可重现性日志，覆盖 2018‑2026 年 F‑Droid 上所有可重现版本。

**📈 对比分析**

通过对比历史日志与重新构建结果，测得重建成功率 83.7% 与可重现率 94%，表明可重现性本身在时间上保持稳定；实验在云端并行构建，平均每次重建耗时约为数十分钟。

**⚠️ 局限性**

研究仅聚焦 F‑Droid，结果不一定适用于其他分发渠道；重建环境推断采用经验时间阈值，可能误判；未对日志分类正则表达式进行交叉验证，导致分类误差；未来依赖可用性与工具更新可能进一步降低重建率。

---

## 304. CSI Simulation: Why Additive Noise Fails and How to Fix It

**arXiv ID:** 2607.01882 | [PDF](https://arxiv.org/pdf/2607.01882v1)

**作者:** Aymen Bouferroum `[一作]` (Inria Lille-Nord Europe), Valeria Loscri `[通讯]` (Inria Lille-Nord Europe)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过实验验证了在Wi‑Fi CSI仿真中加入高斯噪声的常用假设失效，并提出了一种基于测量校准的MQTC（Quantile‑Temporal‑Copula）模型，用以在噪声注入后准确重现接收机链路（AGC、ADC、FFT）引起的幅度压缩与跨子载波相关性；

**💡 创新点**

首度在CSI层面经验性揭示加性噪声模型不成立，并引入分量级分位映射、AR(1)时序滤波与Iman‑Conover copula重排三步校准流水线，显著闭合模拟与真实信号间的8倍幅度误差；

**🔧 技术方法**

采用分位映射（quantile mapping）校正子载波幅度分布、AR(1)时序滤波恢复帧间相关性、Iman‑Conover copula实现跨子载波相关性重排、基于Wasserstein距离、相位圆方差、ACF差异和协方差矩阵的四维真伪度量，以及多分类器AUC评估；

**📊 数据集**

使用6台ESP32‑C6芯片在受控房间与实验室两种环境下收集的清晰与受干扰CSI数据，外部验证数据集包括ESP32的Wallhack1.8k、Intel 5300的SignFi与Widar 3.0；

**📈 对比分析**

与传统AWGN模型及两种中间模型（M2、M3）在四维指标上进行对比，MQTC将幅度Wasserstein距离从3.09降至0.39（8倍改进），累计误差下降89%，在10 dB干扰下5类分类器的AUC从0.522提升至0.904，接近真实数据上0.967的上界；

**⚠️ 局限性**

校准需针对每个部署位置进行，无法跨场景迁移；模型主要适用于OFDM Wi‑Fi，尚未在802.11be或5G NR平台验证；在高子载波数时需对协方差矩阵做稀疏化处理以保持可扩展性。

---

## 305. Learning the Supports for Categorical Critic in Reinforcement Learning

**arXiv ID:** 2607.01880 | [PDF](https://arxiv.org/pdf/2607.01880v1)

**作者:** Jen-Yen Chang `[一作]` (University of Tokyo), Tatsuya Harada `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了动态学习价值函数分布支撑区间的DySEL算法。

**💡 创新点**

将HL-Gauss的上界与均方Bellman误差关联，转化为约束优化，自适应学习支撑区间。

**🔧 技术方法**

使用高斯直方图损失、交叉熵、KL、拉格朗日乘子构成的min-max框架，结合TD3基础。

**📊 数据集**

在DeepMind Control Suite的11个连续控制任务上进行评估。

**📈 对比分析**

与固定支撑HL-Gauss及TD3基线对比，DySEL在多数任务保持竞争力，尤其人形机器人任务表现更好。

**⚠️ 局限性**

对拉格朗日更新稳定性敏感，且学习的支撑区间往往对称，未能自适应偏移；缺乏更广泛的基准验证。

---

## 306. Regression Accumulation in Multi-Turn LLM Programming Conversations

**arXiv ID:** 2607.01855 | [PDF](https://arxiv.org/pdf/2607.01855v1)

**作者:** Yonghui `[一作]`, Lysa Xiao `[通讯]` (University of Otago)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多轮LLM编程对话中的回归累积，构造542个任务的8轮演化链，对26,016次回合进行系统评估，并提出跨轮回归bug分类与干预策略。

**💡 创新点**

创新点在于引入固定演化链与回归oracle的评估协议，系统量化回归累积；构建跨轮回归bug的四类标签体系；验证“验证门”交互级策略在抑制回归方面的有效性。

**🔧 技术方法**

使用LLM代码生成、回归测试、回合级评估指标、四名评审者的手工标注、Snowball Recap与Verification Gate交互策略。

**📊 数据集**

使用HumanEval+和MBPP+共542个Python函数级任务，采用原始测试套件并在第6-8轮做接口重构。

**📈 对比分析**

对6个模型（GPT‑4o、DeepSeek‑V3、Qwen2.5‑Coder‑32B、Qwen3‑32B、DS‑R1‑Distill‑32B、Llama‑3.1‑8B）在所有8轮中的回归通过率进行对比；发现所有模型均有回归累积，验证门可将DeepSeek从75.8%提升至87.9%，将Llama‑3.1‑8B从31.6%提升至47.3%。

**⚠️ 局限性**

仅关注Python函数级任务、固定8轮演化序列，未覆盖自然会话多样性及其他语言/规模；评估仅基于已验证的单元测试，未测量代码可读性或未测试的边缘案例。

---

## 307. CLAP: Closed-Loop Training, Evaluation, and Release Control for Domain Agent Post-training

**arXiv ID:** 2607.01846 | [PDF](https://arxiv.org/pdf/2607.01846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 308. Gender Differences in Research Topic and Method Selection in Library and Information Science: Perspectives from Three Top Journals

**arXiv ID:** 2607.01828 | [PDF](https://arxiv.org/pdf/2607.01828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 309. Geometric Foundation Model Distillation for Efficient Lunar 3D Reconstruction

**arXiv ID:** 2607.01851 | [PDF](https://arxiv.org/pdf/2607.01851v1)

**作者:** Clémentine Grethen `[一作]` (IRIT), Simone Gasparini `[通讯]` (IRIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在月球立体图像上，将大规模3D基模型MASt3R压缩成多种轻量级学生网络，利用知识蒸馏和SVD初始化实现高效重建

**💡 创新点**

提出结构化SVD权重初始化、特征级蒸馏与教师伪标签相结合的压缩框架，显著提升压缩后模型的几何精度

**🔧 技术方法**

知识蒸馏、特征对齐、SVD降维初始化、ViT编码器与Transformer解码器

**📊 数据集**

StereoLunar月球立体数据集

**📈 对比分析**

与教师模型和多种学生架构对比，压缩后模型参数下降4.4×-7.3×，推理速度提升约2×，Chamfer误差仅比教师高15%，在姿态和地形指标上也保持接近或优于教师

**⚠️ 局限性**

对教师模型的适配和预训练依赖较强；在更小的编码器下性能急剧下降；仍需进一步验证在更广泛任务和硬件平台上的通用性

---

## 310. Mixture-of-Parallelisms: Towards Memory-Efficient Training Stack for Mixture-of-Experts Models

**arXiv ID:** 2607.01844 | [PDF](https://arxiv.org/pdf/2607.01844v1)

**作者:** Xuan-Phi Nguyen `[一作]` (Salesforce Ai Research), Shafiq Joty `[通讯]` (Salesforce Ai Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Mixture‑of‑Parallelisms（MoP）的训练框架，通过为Mixture‑of‑Experts（MoE）模型的每个组件（密集路径、注意力、专家前馈、词表投影及优化器状态）分别选择最合适的并行策略，实现显著的显存与带宽利用率提升；

**💡 创新点**

创新点在于：①将整体模型并行拆解为组件级别的并行组合，避免了单一全局计划导致的资源浪费；②提出了记忆高效的Least‑Loaded Expert Parallelism（LLEP）变体，以重叠路由通信与专家计算来削减激活峰值；③设计了分布式的数据‑tensor‑parallel词表投影方案，避免在长序列上生成巨大的logit张量；④实现了将优化器状态放置在CPU并与计算重叠的高效管道；

**🔧 技术方法**

使用的技术包括：MoP组件化并行、LLEP、数据‑tensor‑parallel投影、序列并行、张量并行、流水线并行、专家并行、CPU‑GPU/节点间的高效All‑to‑All/All‑Gather通信、计算‑通信重叠、AdamW优化器状态主机化等；

**📊 数据集**

论文未给出具体数据集，但表明在标准大规模文本预训练数据（如C4、Pile、RedPajama等通用语言模型语料）上进行训练；

**📈 对比分析**

对比方法：在同一硬件（8×8×H200 GPU节点）上，与最佳调优的FSDP2基础方案（FSDP‑best）进行对比；实验结果显示：MoP在120B、600B、1T参数的MoE模型上分别获得4.7×、6.1×、8.2×的GPU吞吐量加速，并且能够支持最高1M token的上下文长度，而FSDP‑best在64–128K上下文已OOM；

**⚠️ 局限性**

局限性：MoP高度依赖高速互连（NVLink/PCIe/InfiniBand），通信开销大；在带宽受限或节点数极大时，All‑to‑All/All‑Gather收敛会成为瓶颈；此外，方案偏向显存紧张场景，若带宽是主要限制则需额外使用流水线并行或多重数据并行以恢复吞吐。

---

## 311. Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference

**arXiv ID:** 2607.01831 | [PDF](https://arxiv.org/pdf/2607.01831v1)

**作者:** Wenchen Han `[一作]` (University College London), Adam Barker `[通讯]` (Huawei)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Lynx系统，采用分层非线性量化将KV缓存拆分为Anchor（高位）与Residual（低位）两条流，并在传输期间利用推测解码使解码阶段可以并行进行，从而显著加速长上下文推理。

**💡 创新点**

创新点在于：①将KV缓存视为可分段使用的资源，突破传统完整接收才可解码的瓶颈；②提出Anchor‑Residual分流量化方案，利用MSB先行传输捕捉注意力分布的主要信息；③在网络层面引入推测解码与验证，保证最终生成结果与全精度一致。

**🔧 技术方法**

技术细节包括：分层非线性（对数）量化与逐通道、逐块归一化；Anchor‑Residual两条流的SerDes实现与双缓冲压缩；推测解码验证协议；Ascend NPU 上的Ascend‑C核实现；与 vLLM‑Ascend 及 LMCache‑Ascend 的集成。

**📊 数据集**

实验使用三大模型（LLaMA 8B、Qwen 32B、Mistral 24B）在三类数据集：MMLU‑Pro（多轮问答）、Needle‑in‑the‑Haystack（检索）、QMSum（摘要）。上下文长度从 16K 到 128K 进行测试。

**📈 对比分析**

与 BF16 原始、INT4/INT8 统一量化以及 CacheGen delta 编码等基线对比。结果显示：Lynx 在保持 BF16/INT8 级别的准确率的同时，TTFT 与 TT32T 分别降低约 30%/10%，TT64T 在低带宽/长上下文情形下比 INT8 提升 0.2~0.8 秒；推测解码接受率平均约 64%，足以覆盖 Residual 传输时间。

**⚠️ 局限性**

限制与待改进：①推测解码的接受率对性能影响敏感；②目前仅验证了 INT4 Anchor 与 Residual，未系统评估不同位宽组合；③实现依赖 Ascend NPU，跨平台移植需要进一步工作；④极端网络延迟或更长上下文仍可能导致 Residual 传输成为瓶颈。

---

## 312. Gaming Consensus: Coordinated Manipulation in Crowdsourced Fact-Checking

**arXiv ID:** 2607.01824 | [PDF](https://arxiv.org/pdf/2607.01824v1)

**作者:** Nikil Roashan Selvam `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地分析了社区注释（Community Notes）中基于矩阵分解的桥接算法对协同操纵的脆弱性，并提出了利用投票历史塑造潜在因子空间以制造“合成共识”的攻击方法，同时在 X 平台上部署了基于人口抽样的缓解措施。

**💡 创新点**

创新点包括：① 两阶段攻击框架（先定位因子空间再统一支持），② 理论上证明投“非有用”可提升帮助度的逆向结果，③ 通过闭式分析估计最小投票数的 MRS（Manipulation Resistance Score）和成本模型，④ 将上述发现应用到公开数据并在 X 实际系统中实现缓解。

**🔧 技术方法**

使用的技术包括：矩阵分解（线性回归 + 正则化）、潜在因子逆向投票策略、基于文本的 MLP 预测模型、Sherman‑Morrison 公式求解插入投票的影响、成本模型构建与仿真。

**📊 数据集**

数据集：公开的 X Community Notes 数据（包含 2021‑2025 年所有笔记与 1.269 亿条评分），并利用其中的文本信息训练笔记参数预测模型。

**📈 对比分析**

通过在真实数据上模拟攻击，发现 10.7% 的低质量笔记在仅投 10 票以内即可突破阈值；MRS 分布显示多数笔记可在少量投票下被操纵；成本模型估算单笔记攻击约 30.5 美元；缓解措施显著提高了所需投票数，提升了系统鲁棒性。

**⚠️ 局限性**

局限性包括：仅在 1 维因子空间下分析，未考虑动态反馈循环；预测模型仅基于笔记文本，忽略被注释帖子和 URL；攻击成本模型假设线性与独立，未充分评估现有反滥用组件的实际抑制效果；实验基于静态快照，未模拟平台实时更新和共识传播。

---

## 313. Koopman operator theory: fundamentals, control, and applications

**arXiv ID:** 2607.01819 | [PDF](https://arxiv.org/pdf/2607.01819v1)

**作者:** Igor Mezić `[一作]` (University of California Santa Barbara), Armin Lederer `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文是一篇关于Koopman算子理论及其在系统与控制中的应用的教程论文，系统地介绍了Koopman算子理论基础、数据驱动逼近方法（EDMD、kEDMD、gEDMD）、控制系统的扩展（LTI嵌入、Koopman控制族、GeKo）、基于Koopman的控制设计（LQR、鲁棒控制、MPC）、与机器学习的关联，并配有GitHub源码示例，帮助读者从理论到实践逐步掌握Koopman方法。

**💡 创新点**

创新点在于：①把Koopman算子理论与现代数据驱动控制方法（EDMD、kEDMD）系统化，提供统一的误差分析与一致性指标；②提出Koopman控制族与GeKo两种控制系统扩展框架，并对比其理论与数值性能；③在教程中引入多步预测损失、输入状态可分离形式和一致性指数，提升模型的多步预测与鲁棒性；④在教程中加入了完整的实验代码与数据，促进学习者快速验证和复现。

**🔧 技术方法**

使用技术包括：Koopman算子与谱分解、EDMD（最小二乘）、kernel EDMD（核函数与RKHS）、gEDMD（生成器逼近）、输入状态可分离形式、GeKo张量积法、LQR、鲁棒控制、Koopman-MPC、机器学习（统计学习理论、贝叶斯视角、深度学习Koopman模型）、一致性指数与投影误差分析、Luenberger观测器设计、Lyapunov稳定性分析。

**📊 数据集**

主要使用的实验数据集为：1）Duffing振荡器的仿真数据（多条初始状态、步长0.25）；2）DC电机的仿真数据（包含非线性输入非单调项）；3）从GitHub提供的统一代码库中读取的多步滑动窗口数据，用于多步预测训练与评估。

**📈 对比分析**

比较方法：在相同字典与输入/状态扩展框架下，分别训练线性、双线性、GeKo和Koopman控制族模型，采用一阶与多步回归两种训练目标；对比指标包括多步均方误差（MSE）、最大误差、运算量（算子范数）以及对不同初始状态的鲁棒性。实验结果表明：在线性输入场景下，双线性与GeKo模型在中等误差和最坏情况上优于纯线性；在非单调输入场景下，GeKo与Koopman控制族在均方误差上提升十倍，且表现更为一致；GeKo模型因完全耦合导致最坏误差略高，Koopman控制族因稀疏耦合具备更好的鲁棒性。

**⚠️ 局限性**

局限性包括：①对Koopman算子逼近的误差仍受字典选择与数据分布影响，可能产生谱污染；②LTI嵌入和输入状态可分离形式要求系统满足严格结构性假设，非线性系统一般只能得到近似模型；③多步训练虽能减小误差，但需要更多数据与计算；④对于高维输入/状态的控制族，维度急剧增长导致可扩展性受限；⑤理论闭环保证需要假设模型误差可控，实际系统中噪声与模型不匹配可能削弱鲁棒性。

---

## 314. SkillCoach: Self-Evolving Rubrics for Evaluating and Enhancing Agentic Skill-Use

**arXiv ID:** 2607.01874 | [PDF](https://arxiv.org/pdf/2607.01874v1)

**作者:** Jiayin Zhu `[一作]` (HKUST(GZ)), Yutao Yue `[通讯]` (HKUST(GZ))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 SkillCoach 框架，利用自演化的评估 rubrics 对 LLM 代理在企业式技能库中的技能使用过程（选择、跟随、组合、反思）进行轨迹级评估与训练。

**💡 创新点**

创新点在于：①将技能使用定义为四维轨迹元能力并与外部验证器分离；②设计自演化 rubrics 通过证据判定、局部补丁与验证门控来不断提升评估质量；③将演化出的 rubrics 直接作为训练筛选器，提升监督学习效果。

**🔧 技术方法**

技术手段包括：LLM 驱动的证据抽取与判分、基于知识图的技能选择与跟随判定、局部补丁生成与验证门控的演化算法、以及基于 rubrics 的轨迹过滤和 SFT。

**📊 数据集**

使用了从 SkillsBench 选出的 24 个技能依赖任务（包含金技能与扰乱技能），以及构造的 distractor‑augmented 库和公开的 50k 实际技能库进行边界分析。

**📈 对比分析**

与传统的基于最终结果的 SFT（Outcome‑only）和大型预训练模型相比，演化 rubrics 在评估可靠性、消除幻觉、提升过滤一致性方面均优于初始 rubrics；在训练中，使用演化 rubrics 过滤的示例使 Qwen3.5‑4B 与 9B 的任务准确率分别从 8%→24% 与 14%→32%，显著优于仅依赖结果的 SFT。

**⚠️ 局限性**

局限性在于：①仅在有限的技能依赖任务子集上验证，未覆盖更大规模、持续演进的生产型技能库；②实验仅使用离线监督微调，未探究基于 rubrics 的强化学习或长期部署反馈。

---

## 315. CamoNAS: Neural Architecture Search for Enhanced Camouflaged Object Detection

**arXiv ID:** 2607.01870 | [PDF](https://arxiv.org/pdf/2607.01870v1)

**作者:** Dawei Ren `[一作]` (Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences), Jianpo Liu `[通讯]` (Shanghai Institute of Microsystem and Information Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一种针对隐蔽目标检测的频域感知多分辨率NAS框架CamoNAS，自动搜索单元操作与多尺度路径

**💡 创新点**

创新点在于将可学习的小波变换与RGB双流相结合，并在层级空间同时搜索单元操作与尺度路由，以提升边界精细化和背景抑制

**🔧 技术方法**

采用可学习小波变换、RGB‑频域双流、DARTS风格微单元搜索、可学习分辨率转移、低秩Soft‑VQ融合头等技术

**📊 数据集**

使用四个COD基准数据集：CAMO、COD10K、CHAMELEON、NC4K

**📈 对比分析**

与十余种基线对比，CamoNAS在四个数据集的Sα、Fβ、Eϕ、MAE指标上均名列前茅，尤其在CHAMELEON和COD10K上显著提升

**⚠️ 局限性**

局限在于搜索空间仍相对有限，频域与RGB仅在最终融合处交互，且搜索和训练成本较高

---

## 316. Evaluating Chunking Strategies for Retrieval-Augmented Generation on Academic Texts

**arXiv ID:** 2607.01852 | [PDF](https://arxiv.org/pdf/2607.01852v1)

**作者:** Valentin J. J. Kreileder `[一作]` (Deggendorf Institute of Technology), Andreas Fischer `[通讯]` (Deggendorf Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了三种文本分块策略（固定大小、递归分块、基于聚类的语义分块）在学术论文上的检索增强生成（RAG）效果，使用RAGAs评估框架和中等配置的LLM；

**💡 创新点**

提出了结合可信度与答案相关性的答案质量得分（AQS），并揭示RAGAs在长文档评估中的可靠性有限；

**🔧 技术方法**

采用了RAGAs框架、llama3.2:3b生成模型、deepseek-r1:8b评估模型、all-MiniLM-L6-v2嵌入器，以及TF‑IDF二元组相似度、Context F1等评估指标；

**📊 数据集**

使用了十三篇学术毕业论文（约10k–27k词）及十条问答（5条固定信息，5条论文专属）作为测试数据集；

**📈 对比分析**

通过对比Context F1和AQS指标，固定信息问题的得分普遍偏低，而自由问题中固定分块和递归分块表现相近（中位数≈0.65），聚类分块表现最差（中位数≈0.40），且可信度评估在大部分样本中失效；

**⚠️ 局限性**

受硬件限制导致模型规模受限、可信度评估频繁失败、聚类分块增加计算复杂度但无显著提升，并且RAGAs指标可能与人工评估不一致，限制了结论的普适性。

---

## 317. Technical Debt Friction for Maintenance Prioritization: An Industrial Multi-Case Study

**arXiv ID:** 2607.01850 | [PDF](https://arxiv.org/pdf/2607.01850v1)

**作者:** Simeon Tverdal `[一作]` (SINTEF), Adam Tornhill `[通讯]` (CodeScene)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对三家工业案例进行多案例研究，探讨技术债务摩擦（Technical Debt Friction）在维护优先级决策中的作用与可解释性。

**💡 创新点**

提出“技术债务摩擦”作为一种将代码质量、变更频率、耦合与社会技术因素整合的维护导向指标，并验证其在实际开发中的价值。

**🔧 技术方法**

采用 CodeScene 平台的 CodeHealth、热点（Hotspots）、耦合（Change Coupling）以及社群技术视图进行静态与动态分析，并在结构化走查会中呈现。

**📊 数据集**

数据集来自三家公司（SINTEF、奥斯陆大学、CodeScene 等）中的真实工业项目，涵盖数十个文件级别、数个版本的变更记录。

**📈 对比分析**

通过与开发者主观感知、已识别的维护痛点和 refactoring 需求进行对照，评估摩擦分数与实际维护优先级的相关性；实验显示摩擦与已知问题文件高度重叠，但单独使用并不具备自动排序能力。

**⚠️ 局限性**

局限性包括：样本规模有限，主要来自同一行业领域；摩擦指标高度依赖上下文与团队知识，需人工解读；实验缺乏量化验证，未提供客观性能度量。

---

## 318. Decomposer: Learning to Decompile Symbolic Music to Programs

**arXiv ID:** 2607.01849 | [PDF](https://arxiv.org/pdf/2607.01849v1)

**作者:** Yewon Kim `[一作]` (Carnegie Mellon University), Chris Donahue `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将 MIDI 转为可执行、可编辑 Strudel 代码的符号音乐逆向工程框架。

**💡 创新点**

创新点在于：①构建 21,174 对合成 MIDI–Strudel 样本的语料库；②两阶段训练（SFT + 基于执行的 RL）同时优化音乐还原度与代码可读性；③设计结合 Onset‑F1 与 12 项可读性 rubrics 的复合奖励。

**🔧 技术方法**

使用大型开源 LLM（Qwen3‑4B/8B），基于 LoRA 进行 SFT，随后采用 GDPO 强化学习，奖励由执行器 𝒞 计算的 MIDI 还原度与可读性评分构成。

**📊 数据集**

训练数据包括：①自制的 21k MIDI–Strudel 语料库（由前沿 LLM 生成后执行得到 MIDI）；②真实 MIDI 语料 LMD、NES‑MDB、GigaMIDI 等用于 RL 与评估。

**📈 对比分析**

与前沿 LLM（GPT‑5.5 等）和启发式 MIDI‑to‑Strudel 转换器对比：在 Onset‑F1 上 8B 模型提升 0.16（synthetic）/0.32（LMD），可读性 Rubric 分数提升至 0.61‑0.74（远高于启发式的 0.05/0.09），同时保持代码多样性；RL 阶段显著提升真实 MIDI 的 Faithfulness。

**⚠️ 局限性**

局限性包括：①可读性奖励基于固定 12 项 rubrics，可能对某些风格偏好不足；②实验仅覆盖短片段，难以处理长形式音乐；③未显式优化多样性，后续可加入多样性奖励；④仅处理 MIDI，未扩展至音频或更高维度的音乐属性。

---

## 319. Non-synchronism in Global Usage of Research Methods in Library and Information Science from 1990 to 2019

**arXiv ID:** 2607.01833 | [PDF](https://arxiv.org/pdf/2607.01833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 320. Understanding Software Defect Prediction: A Large-scale Empirical Study Across Uncertainty Quantification and Performance Evaluation

**arXiv ID:** 2607.01842 | [PDF](https://arxiv.org/pdf/2607.01842v1)

**作者:** Ranjun Peng `[一作]` (Macau University of Science and Technology), Zhijie Wang `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在软件缺陷预测中，系统性评估了基于概率的置信度量（UQ）与模型性能、校准误差以及跨项目迁移的关系。

**💡 创新点**

首次在16种代表性传统分类器和36个公开基准上，大规模实验揭示 UQ 与性能的关联高度依赖于指标、分类器类别、数据集特性及迁移设置，且校准误差与性能并非互相决定。

**🔧 技术方法**

使用了五种常见的概率置信度量（最大概率、最小置信度、边际分数、预测熵、DeepGini），六种性能指标（MCC、Precision、Recall、FPR、AUC、F1）和三种校准指标（ECE、NLL、Brier Score），并通过 Pearson、Spearman、Kendall 三种相关系数进行关联分析。

**📊 数据集**

数据集来源于 AEEEM、NASA、PROMISE、ReLink 四大公开缺陷预测基准（共 36 个项目），并在 AEEEM、NASA、PROMISE 的 32 个特征兼容项目上执行直接迁移（CPDP）实验。

**📈 对比分析**

比较方法：在 WPDP 下计算 UQ 与性能/校准的相关系数；在 CPDP 下重复上述操作并对比方向与幅度变化。结果显示：UQ 与 FPR、AUC 的相关性较强，但与 MCC、F1 等阈值敏感指标几乎无关联；高性能模型不一定校准良好；跨项目迁移时，部分相关性会发生方向逆转。

**⚠️ 局限性**

局限性：仅考虑传统基于概率输出的分类器；未包含深度学习或预训练代码模型；数据集与预测粒度相互耦合，难以单独评估粒度影响；CPDP 仅在特征兼容的项目上实验，未涵盖所有跨项目情形；所用置信度量在二分类中高度相关，可能限制解释深度。

---

## 321. Congestion-Based Slot Pricing in a Railway Auction Game

**arXiv ID:** 2607.01822 | [PDF](https://arxiv.org/pdf/2607.01822v1)

**作者:** Bill Roungas `[一作]` (Panteion University), Sebastiaan Meijer `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一种基于多智能体的铁路时隙拍卖游戏，用于研究不同规模运营商在拥堵定价与纠正激励机制下的决策行为

**💡 创新点**

创新点在于结合了拥堵感知的基准价格与对最大/最小请求者分别施加惩罚/奖励的非对称激励，以抑制大型运营商的优势并提升小型运营商的参与度

**🔧 技术方法**

采用Web技术（HTML/CSS/JavaScript/PHP）与MySQL数据库实现实时多人交互、时间限制决策与即时财务反馈的交互式游戏平台

**📊 数据集**

使用领域专家模拟的数据（即三名不同规模运营商的决策记录），未采用公开标准数据集

**📈 对比分析**

未进行严格的实验对照，只在两次专家会话中观察机制反应，结果显示拥堵定价正常工作，但大型运营商仍保持高请求，表明惩罚措施不足以完全消除优势行为

**⚠️ 局限性**

局限在于实验规模小、仅为探索性观察，缺乏统计检验与长期对照实验，且未检验不同参数设置对结果的影响

---

## 322. C2E: Boosting Ego-Only 3D Object Detection via Multi-Teacher Contrastive Knowledge Distillation

**arXiv ID:** 2607.01827 | [PDF](https://arxiv.org/pdf/2607.01827v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 323. Rethinking Conditional Generation for Underwater Salient Object Detection

**arXiv ID:** 2607.01825 | [PDF](https://arxiv.org/pdf/2607.01825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 324. MMIR-TCM: Memory-Integrated Multimodal Inference and Retrieval for TCM Clinical Decision Support

**arXiv ID:** 2607.01814 | [PDF](https://arxiv.org/pdf/2607.01814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 325. Decoupling Code Complexity from Newcomer Participation: A Causal Study of AI Coding Agent Adoption in OSS

**arXiv ID:** 2607.01810 | [PDF](https://arxiv.org/pdf/2607.01810v1)

**作者:** Weiwei Xu `[一作]` (Peking University), Minghui Zhou `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过因果差分法评估了 AI 编码助手（如 Cursor、Claude Code）在已建立的开源项目中的采纳效应，重点关注新手参与（新人流入、首次贡献处理、留存和 “good‑first‑issue” 任务数量）和代码复杂度的变化。

**💡 创新点**

创新点在于：①首次通过因果设计检验 AI 编码助手对 OSS 新手参与的影响；②对代码复杂度成本进行实地校准（相较先前 41% 的增幅，本文发现 3–11% 的增长）；③在同一项目集合内直接检验复杂度上升是否导致新手参与下降，验证了“复杂度-排斥”机制的解耦。

**🔧 技术方法**

采用的技术包括：GitHub 代码搜索 + 自适应大小分区、项目匹配（倾向得分 1:3 近邻匹配、星级桶划分）、差分-差分估计（BJS imputation、Callaway‑Sant'Anna 组时效/动态估计、Sun‑Abraham 事件研究）、单元线性趋势、平行趋势检验、假设检验（placebo、Honest‑DiD）。

**📊 数据集**

数据集来源于 GitHub：1,888 采用 AI 配置文件（.cursor.yml/.claude.yml）提交的项目（其中 603 个具有至少 6 个月前期历史）及 1,784 控制项目；项目特征包括 stars、forks、提交量、语言；每月面板记录新手数量、PR 处理时间、留存率、good‑first‑issue 计数；代码复杂度使用 Lizard（cyclomatic）和 complexipy（Python cognitive）从每月快照计算。

**📈 对比分析**

比较方法：对照匹配的非采纳项目进行多方法差分-差分估计，并进行事件研究；使用多估计器和单元趋势校正来确保结果稳健。结果显示：新手流入无显著负面影响（+5% 估计，未显著）；首次 PR 接受率、合并时间和留存率基本无差异；复杂度上升约 3%（cyclomatic）和 11%（Python cognitive），但在相同项目上仍无新手参与下降。

**⚠️ 局限性**

局限性包括：①仅检测可见的配置文件提交，未衡量实际使用强度；②样本局限于具有至少 6 个月前期历史的已建立项目，对“从一开始就使用 AI”的项目不适用；③认知复杂度仅在 Python 上可测；④good‑first‑issue 指标稀疏且缺乏平行趋势验证；⑤可能漏掉某些机器人/自动化行为；⑥结果的外部有效性对私有或不同语言生态系统有限。

---

## 326. Predicting Heterogeneous Treatment Effects Of Building Energy Saving Retrofits Using Causal Machine Learning

**arXiv ID:** 2607.01891 | [PDF](https://arxiv.org/pdf/2607.01891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 327. Descriptor: LYNRED Mobility Dataset Multimodal Detection Subset (LYNRED-MDS)

**arXiv ID:** 2607.01871 | [PDF](https://arxiv.org/pdf/2607.01871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 328. HCMS: Head-Chunked Multi-Stream Pipeline for Communication-Computation Overlap in Long-Sequence Parallel Attention

**arXiv ID:** 2607.01817 | [PDF](https://arxiv.org/pdf/2607.01817v1)

**作者:** Chao Yuan `[一作]` (Bilibili Inc.), Jing Liu `[通讯]` (Bilibili Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Head‑Chunked Multi‑Stream Pipeline（HCMS）来提升 Transformer 在中长序列（31K‑56K token）上的通信与计算重叠。

**💡 创新点**

创新点在于将多头注意力按头分块，利用双 CUDA 流实现细粒度头级重叠，且无需改动 FlashAttention/SDPA 等现有核函数，兼容现有分布式训练框架。

**🔧 技术方法**

采用的技术包括：多头注意力头分块（Head Chunking）、双流并行（通信流与计算流）、CUDA Event 同步、All‑to‑All 交换、以及对 PyTorch autograd 的无侵入支持。

**📊 数据集**

使用的数据集主要是视频生成模型（Sora、Wan2.2 等）内部数据，序列长度为 31K‑56K tokens 的 Latent 空间序列；实验覆盖 4‑8 GPU、PCIe 4.0/5.0 等多平台。

**📈 对比分析**

与 DeepSpeed Ulysses、Ring Attention 进行对比；在 4‑8 GPU、31K‑56K token 的场景下，HCMS 对比 Ulysses 提升 10%‑17.5%（单层）并对比 Ring 提升 5%‑14.5%；端到端 Wan2.2 生成时提升 6.8%。

**⚠️ 局限性**

限制在于通信比率 ρ<10%（如 281K token）时收益有限；过多分块会产生 Event 同步开销；目前仅验证至 8 GPU，跨节点或更大规模的适用性尚未验证。

---

## 329. Regularized Variational and Spectral Log-Density-Ratio Estimation in the Gaussian Location Model

**arXiv ID:** 2607.01895 | [PDF](https://arxiv.org/pdf/2607.01895v1)

**作者:** Francis Bach `[一作]` `[通讯]` (Inria), Francis Bach (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在具有共同协方差矩阵的高斯位置模型中，岭正则化的对数密度比估计。

**💡 创新点**

提出了一种新的变分估计器和谱估计器，并通过高维确定性渐近等价物进行比较，发现在观察数量较多时，变分估计器的风险较小，而在观察数量较少时，谱估计器更具优势。

**🔧 技术方法**

使用了岭正则化的变分估计和谱估计技术，结合了高斯样本协方差矩阵的随机矩阵理论。

**📊 数据集**

使用了高维高斯样本数据集进行实验，具体样本数量和维度在文中进行了详细说明。

**📈 对比分析**

通过比较变分估计器和谱估计器的风险，发现变分估计器在观察数量多时表现更好，而谱估计器在观察数量少时表现更优，且在不同的正则化参数下进行了性能评估。

**⚠️ 局限性**

限制在于该研究主要集中在高斯位置模型中，未来的扩展可以考虑其他类型的特征和模型，例如非线性特征和协方差变化模型。

---

## 330. Spec-AUF: Accept-Until-Fail Training under Train-Inference Misalignment for Masked Block Drafters

**arXiv ID:** 2607.01893 | [PDF](https://arxiv.org/pdf/2607.01893v1)

**作者:** Tianjian Yang `[一作]` (Peking University), Meng Li `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种称为AUF的接受至失误监督策略，改进块式草稿模型的训练，以更好匹配推理中的前缀接受行为。

**💡 创新点**

创新点在于将教师强制启发式与接受至失误裁剪结合，自动根据模型预测的第一个错误动态截断监督区间，消除手工设置的指数位置衰减，提升块式草稿的接受长度。

**🔧 技术方法**

采用块式草稿网络（DFlash、Domino）与SGLang推理框架，基于交叉熵的接受至失误（AUF）损失，并在Qwen3-8B上进行训练。

**📊 数据集**

使用ShareGPT训练对话数据，在六个基准（GSM8K、MATH-500、HumanEval、MBPP、MT-Bench、Alpaca）上评估。

**📈 对比分析**

与传统指数位置衰减交叉熵（decay-only）对比，AUF在greedy和sampling两种验证模式下均提升平均接受长度，提升幅度约2.4%~4.8%，在所有基准上均有正向改进。

**⚠️ 局限性**

实验仅在固定块大小B=16、Qwen3-8B与ShareGPT数据上验证，未探索不同模型规模、块尺寸或与其他自适应目标（如D-PACE、SpecDiff-2）的对比，且无法验证是否对更大规模或多样化任务同样有效。

---

## 331. Safety Targeted Embedding Exploit via Refinement

**arXiv ID:** 2607.01859 | [PDF](https://arxiv.org/pdf/2607.01859v1)

**作者:** Joshua Adrian Cahyono `[一作]` `[通讯]` (Nanyang Technological University), Joshua Adrian Cahyono (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种梯度引导的代码切换攻击（STEER），通过识别并替换触发模型拒绝机制的词汇，实现对多语言LLM安全防御的绕过。

**💡 创新点**

创新点在于：①利用机理可解释性发现的单一拒绝方向作为攻击目标；②通过梯度归因精准定位高贡献词汇；③逐步翻译并迭代改写以最小化拒绝得分；④使用Fisher Linear Discriminant自动定位最易被利用的层，并量化安全编码的脆弱性；⑤构建跨多语言、低资源词汇池，显著提升攻击效率与成功率。

**🔧 技术方法**

技术手段包括：机制可解释性（识别拒绝方向）、梯度归因（词级贡献评分）、Fisher Linear Discriminant层选择、自动化翻译与多语言代码切换、GPT‑4o进行间接表述的前置重写、基准评测（JailbreakBench、HarmBench、AdvBench）。

**📊 数据集**

使用的数据集：JailbreakBench（100条）、HarmBench（200条）、AdvBench（520条）共计约 820 条攻击实例；在六款7–9B开源指令调优模型上进行白盒评测。

**📈 对比分析**

对比方法：与直接未改写（Direct）、随机代码切换（CSRT）和基于后缀的梯度优化（GCG）进行对比。STEER 在 8 次迭代内在所有模型‑基准组合上均实现最高 ASR，平均约 70–96%（最高 96.7%），并在 GPT‑4o‑mini 上实现 35.5% 的转移成功率；FLD 层选、翻译池规模与前置 paraphrase 步均显著提升成功率，分别减少 19–44% 的性能损失。

**⚠️ 局限性**

局限性：需要模型内部白盒访问，难以直接应用于闭源 API；实验仅覆盖 7–9B 参数范围；黑盒适配尚未设计；评测使用 GPT‑4o 自动判断，可能与人工评估存在差异；转移性能有限，表明对未知架构仍有一定依赖。

---

## 332. Training-free Controllable Human Motion Generation under Heterogeneous Constraints

**arXiv ID:** 2607.01990 | [PDF](https://arxiv.org/pdf/2607.01990v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 333. On the algebraic analysis of runtime distribution of probabilistic programs

**arXiv ID:** 2607.01856 | [PDF](https://arxiv.org/pdf/2607.01856v1)

**作者:** Michele Boreale `[一作]` (University of Florence), Alessandro Pompa Di Gregorio `[通讯]` (University of Florence)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出了一套基于生成函数与核多项式的代数方法，用于精确计算满足 Generalized Constant Probability (GCP) 格式的概率程序的终止分布、期望运行时间以及尾概率上界。

**💡 创新点**

创新点在于：① 将 GCP 程序的执行映射到概率化的下推自动机（pPDA）并构造其运行生成函数；② 证明该生成函数是代数函数，并通过核多项式与“kernel method”获得其闭式表示；③ 在此基础上开发符号算法，直接求得收敛半径、主奇点以及相应的渐近系数和指数尾界，从而获得比传统 martingale 或证明系统更精确的尾概率上界。

**🔧 技术方法**

主要技术包括：概率程序语义的 PDA 运行模型；生成函数与代数函数理论；核多项式、Puiseux 级数与 Newton 多边形算法；符号线性代数求解；复杂度分析与 Homotopy 连通性等。

**📊 数据集**

论文没有使用外部真实数据集；所有实验均基于人工构造的示例程序（如随机游走、ZeroConf 协议、BRP 协议等），在 Maple 中实现并测算。

**📈 对比分析**

与已有的 martingale / 期望时间上界方法相比，本文的算法在符号求解后可以得到闭式生成函数，从而直接给出精确的渐近系数和指数尾界；在实验中，算法执行时间通常低于 0.5 秒，且得到的尾界严格优于基于 E_max 的粗糙上界。

**⚠️ 局限性**

局限性包括：① 仅对 GCP（或其轻微扩展）程序可直接应用，不能覆盖更一般的概率循环；② 算法在 1‑state 子类外为不完整，需满足线性代数条件（如足够多的“小根”）才能成功；③ 对高阶多项式求解与 Puiseux 级数的符号计算在某些例子中仍显得计算量大；④ 结果依赖于核多项式的可分性与唯一解的存在性，若条件不满足则会失败。

---

## 334. Actual causality in fault trees

**arXiv ID:** 2607.01840 | [PDF](https://arxiv.org/pdf/2607.01840v1)

**作者:** Georgiana Caltais `[一作]` (University of Twente), Mariëlle Stoelinga `[通讯]` (University of Twente)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文把故障树（Fault Tree, FT）转化为因果模型，并利用Halpern‑Pearl的实际因果性（Actual Causality, AC）框架，对FT中的因果关系进行系统化的分类与分析，探讨最小割集（MCS）与实际因果之间的关系，并给出针对不同AC定义的计算复杂度与算法设计。

**💡 创新点**

创新点包括：
1) 将FT映射到结构方程模型（SEM），首次在FT上完整引入AC三种定义（原始、更新、简化）并进行分类；
2) 明确了在FT中实际因果只能是单元事件（singleton）或特定子集的必要充分条件；
3) 证明MCS元素是实际因果的充分条件，但不是必要条件，且给出了特定结构（树形或DNF）下的必要条件；
4) 对三种AC定义在FT上的计算复杂度给出精确界限（NP‑complete或多项式），并提供了对应的算法（路径过滤、正向/反向传播、最小路径集转换）。

**🔧 技术方法**

主要技术手段：Halpern‑Pearl的实际因果性定义、结构方程模型与do‑算子、因果网络与图论（路径、可达性、禁止对等）、布尔函数与最小割/最小路径集理论、NP完备性证明与算法复杂度分析。

**📊 数据集**

论文未使用公开实验数据集；研究基于理论推导与算法分析，并引用了工业界典型的54个FT实例作为经验验证，但并未在本文中给出实验结果。

**📈 对比分析**

与传统的最小割集（MCS）分析对比，本文指出MCS并不等价于实际因果，给出了更细粒度的因果判定。算法方面，对原始AC给出O(|E|^2|V|+2^k|E|)的实现；对更新AC给出O(2^|V||V|^2|E|^2)的上界；对简化AC给出O(2^|T||E|)的上界。整体而言，原始AC在FT上可多项式求解，而更新/简化AC仍保持指数级复杂度。

**⚠️ 局限性**

局限性：
1) 仅针对静态、可合成的FT（无动态或时间依赖性）；
2) 对更新AC的算法仍未实现高效版本；
3) 论文缺乏实验验证与真实系统案例；
4) 未考虑概率信息，无法直接得到因果概率或置信度；
5) 只讨论了最小割集与因果关系的必要条件，缺乏完整的因果归因与责任度量。

---

## 335. An overlap-free morphism is a k-power-free morphism for any integer k $\ge$ 3

**arXiv ID:** 2607.01837 | [PDF](https://arxiv.org/pdf/2607.01837v1)

**作者:** Francis Wlazinski `[一作]` `[通讯]`, Francis Wlazinski

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

无法确定论文的研究内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法确定比较方法和性能

**⚠️ 局限性**

信息不足导致无法评估局限性

---

## 336. MolSight: A Graph-Aware Vision-Language Model for Unified Chemical Image Understanding

**arXiv ID:** 2607.01982 | [PDF](https://arxiv.org/pdf/2607.01982v1)

**作者:** Wenda Wang `[一作]` (Renmin University of China), Zhewei Wei `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MolSight，一种图结构感知的视觉‑语言模型，用于统一化学分子图像的理解。

**💡 创新点**

创新点在于引入分子拓扑模块 (MTM) 预测视觉标记间的化学键邻接关系，并通过分子定位模块 (MGM) 将视觉表示与 SVG 结构注释对齐，实现图拓扑信息在视觉特征中的注入。

**🔧 技术方法**

采用 Transformer、图注意力、跨模态注意力等技术，基于 Qwen3‑VL 视觉编码器，结合 SVG 文本编码器和 LoRA 微调方案。

**📊 数据集**

使用 PubChemSTM（约 249k 分子图像–任务标签对）进行预训练，并在 MolVision、MoleculeQA、CHEBI‑20 等公开数据集上进行下游任务评估。

**📈 对比分析**

与通用 VLM（如 GPT‑4V、Qwen‑VL）和分子专用 LLM（如 MolT5、ChemLLM）进行对比，MolSight 在 SMILES 翻译、分子字幕、描述符预测和活性预测等任务上均显著优于现有方法，取得了 SOTA 级别的性能（SMILES 翻译 Tanimoto 相似度>0.96、准确率>83%、有效率>99%）。

**⚠️ 局限性**

局限性包括仍需要 SVG 结构注释作为辅助信息，对分子尺寸过大或结构极为复杂的图像表现下降，且对不同图像格式的鲁棒性尚待进一步验证。

---

## 337. Epic-Organized vs. Requirement-Aligned Gherkin: An Empirical Evaluation of LLM-Based Acceptance Criteria Generation

**arXiv ID:** 2607.01980 | [PDF](https://arxiv.org/pdf/2607.01980v1)

**作者:** Shahbaz Siddeeq `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并评估了两通道LLM管道（epic合成 + 问题填补），以 epic 组织方式生成 Gherkin 场景，并与零射击基线进行对比。

**💡 创新点**

创新点在于将 JSON schema 与后置验证相结合实现结构保证，并通过双重覆盖度（TF‑IDF 与 dense embedding）首次量化 epic‑organized 生成的语义覆盖与覆盖率差异。

**🔧 技术方法**

使用 OpenAI GPT‑4 进行提示工程与两步生成，利用 JSON schema 强制输出结构，采用 TF‑IDF 词频分析和 OpenAI dense embeddings 进行语义覆盖评估，并进行盲评专家打分。

**📊 数据集**

采用 PURE（Public Requirements）数据集中的四份 SRS 文档，共计 107 条功能需求作为实验数据。

**📈 对比分析**

与零射击基线相比，epic‑organized 方案在结构有效率 100% 与 99%（baseline）保持一致，语义 RCR 94.3% 对 92.9%、TF‑IDF RCR 72.0% 对 76.0%；专家评分在 Correctness、Executability 与 Completeness 上均优于基线；生成时延为 58.9 秒 vs 21.5 秒。

**⚠️ 局限性**

主要局限包括仅测试四份文档、缺乏大规模 SRS 验证、模型供应商耦合可能影响语义覆盖、单次实验跑可能因 LLM 随机性产生差异、未评估更大规模或非功能需求。

---

## 338. A More Accurate Algorithm Comparison through A/B Testing using Offline Evaluation Methods

**arXiv ID:** 2607.01958 | [PDF](https://arxiv.org/pdf/2607.01958v1)

**作者:** Koki Konishi `[一作]` (Hakuhodo DY Holdings Inc.), Yuta Saito `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MID估计器，通过引入假中间算法在A/B测试中诱导正相关，从而提升算法选择的准确性和样本效率。

**💡 创新点**

创新点在于有意识地在A/B测试框架下引入正相关；利用中间算法拆分差值估计；推导最优中间算法并证明其无偏；实现比传统AVG和离线IPS更低的选择误差。

**🔧 技术方法**

采用统计学的方差-偏差分解、重要性加权（IPS）、离线评估技术以及A/B测试模拟；结合理论推导和实验验证。

**📊 数据集**

使用Kuaishou的KuaiRec公开视频推荐数据集（全观测的用户-项目交互矩阵）来生成基准性能并进行实验。

**📈 对比分析**

与AVG（样本均值）和IPS（仅离线）进行比较，评估指标包括选择误差率、方差和统计检验功效；MID在相同误差率下仅需1/4–1/2的样本量，误差率始终低于两者，且统计功效更高。

**⚠️ 局限性**

局限在于仅以IPS为基础实现MID；对极低相似度或多算法比较的扩展未验证；需要进一步结合双稳健等更先进方法；理论分析假设样本量相等、已知真实性能，实际部署时可能受限。

---

## 339. Hybrid quantum-classical neural network for sentiment analysis

**arXiv ID:** 2607.01943 | [PDF](https://arxiv.org/pdf/2607.01943v1)

**作者:** Giacomo Cappiello `[一作]` (University of Southern Denmark), Dimitrios Makris `[通讯]` (Kingston University London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了混合量子-经典神经网络(HNN)并应用于COVID‑19推文的情感分析，随后通过迁移学习评估其在短信垃圾分类任务中的表现。

**💡 创新点**

创新点在于首次将参数化量子电路嵌入文本情感分析的深度学习管线，展示量子层在训练动态和迁移学习中能提供更丰富的特征表达，并在有限量子资源下实现与经典模型相近甚至更优的性能。

**🔧 技术方法**

采用TF‑IDF特征提取，经典模型为单隐藏层全连接网络；HNN由角编码、受控门耦合的参数化量子电路（6、8、12量子比特）以及期望值测量输出，后接经典线性层；训练使用Adam、交叉熵、参数‑shift梯度，所有量子模块在无噪声模拟器上实现。

**📊 数据集**

使用公开的COVID‑19推文情感数据集（41,159训练条目/3,798测试条目）以及SMS垃圾邮件数据集（5,574条目）进行实验。

**📈 对比分析**

与经典基线对比，12量子比特HNN在情感分类的平均测试准确率为77.64%（与经典77.88%相近，方差更低），而6/8比特模型表现更差；迁移学习后，12比特HNN在垃圾邮件检测上达到66.94%准确率，明显优于经典的63.12%，主要提升体现在对多数类（正常短信）的识别率。

**⚠️ 局限性**

主要限制包括：量子电路规模受限导致小规模HNN的方差高、训练不稳定；实验仅在理想模拟器上进行，未考虑真实量子硬件的噪声与拓扑约束；迁移学习提升主要集中在多数类，对少数类的召回提升有限。

---

## 340. AIriskEval-edu: New Dataset for Risk Assessment in AI-mediated K-12 Educational Explanations

**arXiv ID:** 2607.01934 | [PDF](https://arxiv.org/pdf/2607.01934v1)

**作者:** Javier Irigoyen `[一作]` (BiometricsAI, Universidad Autónoma de Madrid), Aythami Morales `[通讯]` (Universidad de Las Palmas de Gran Canaria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 AIriskEval-edu-db2 数据集，并对其进行风险评估实验，验证 LLM 评估器在 K‑12 教学解释中的可行性与可解释性；

**💡 创新点**

创新点在于（1）通过六种教师角色生成多样化 LLM 解释并对风险进行二分类标注；（2）引入可解释的风险定位与描述；（3）实现轻量级 Llama‑3.1‑8B 的微调，使评估器可本地部署；

**🔧 技术方法**

使用的技术包括 Gemini 2.5 Pro、GPT‑5.5、Llama 3.1‑8B；半自动标注流程结合专家校验；LoRA 微调、5‑折交叉验证；评估指标涵盖 MAE、IoU、BLEU、ROUGE‑L 与 BERTScore 等；

**📊 数据集**

采用 AIriskEval‑edu‑db2 数据集（170 道 K‑12 科学题，1,639 条解释，11 种 LLM 教师角色，785 条带可解释标注），并参考原始 EduEVAL‑DB；

**📈 对比分析**

通过零射推理和微调模型在三份子集（EduEVAL‑DB、扩展子集、完整数据集）上进行二分类与可解释评估；微调后的 Llama‑3.1‑8B 在大多数维度上逼近甚至超过 Gemini 2.5 Pro 与 GPT‑5.5，尤其在关注度、深度、学生适宜度和偏见检测方面表现显著提升；

**⚠️ 局限性**

局限性包括：半自动标注仍需人工干预；对多轮对话、跨文化和多模态场景的鲁棒性不足；轻量模型在事实准确性上仍落后前沿模型；评估主要集中于文本，未覆盖更丰富的学习交互。

---

## 341. ElephantAgent: Contextual State Continuity in Agentic Systems

**arXiv ID:** 2607.01919 | [PDF](https://arxiv.org/pdf/2607.01919v1)

**作者:** Jiankai Jin `[一作]` (360 AI Security Lab), Quanchen Zou `[通讯]` (360 AI Security Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于上下文状态连续性的安全协议（Contextual State Continuity Protocol），通过在Agentic系统中嵌入Context Guard和State Continuity Module，并利用可信执行环境（Intel TDX）与Nimble线性化账本，确保工具描述符与内部记忆的完整性与连续性，从而防止工具与记忆中毒攻击。

**💡 创新点**

创新点包括：
1) 将Agent的“工具状态+内存”作为受保护的上下文子集，并通过哈希+数字签名在TEE内验证其连续性；
2) 采用可插拔的初始化与快照策略，提供历史可追溯性以便在语义滥用后进行审计与恢复；
3) 将Nimble线性化账本与Intel TDX相结合，构建可扩展且易于部署的可信账本服务；
4) 该协议与MCP、CLI等多种Agentic框架兼容，可作为插件式安全增强。

**🔧 技术方法**

技术栈：
- 可信执行环境（Intel TDX）
- Nimble线性化账本（复制状态机）
- 远程身份验证与签名（SM2/ED25519）
- 哈希函数（SHA‑256）
- Context Guard（本地状态校验）
- State Continuity Module（远程状态日志）
- 代理语言模型（Qwen）用于规划与执行
- MCP协议与CLI接口

**📊 数据集**

实验数据集：
- Email Triage（基于邮件线程的分类与回复）
- Code Bug Fix（QuixBugs仓库的bug修复流程）
- Calendar Scheduling（CalDAVTester日历调度）

**📈 对比分析**

评估方法与性能：
- 与传统未加保护的MCP进行对比；
- 量化初始化、数据加载、状态验证、状态更新四个阶段的时延；
- 结果：
  * 初始化时延从219ms提升至1.8s（8.22×），为一次性成本；
  * 每查询验证约4.2–4.3ms，更新约5.25–5.39ms；
  * 任务总时延相对未加保护仅提升1.02×–1.04×；
  * 服务器端TDM attestation模式下，初始化1.51×，工具调用1.12×–1.22×。

**⚠️ 局限性**

限制与挑战：
- 只能阻止out‑of‑band状态篡改，无法防御内部工具的语义滥用；
- 依赖TEE硬件与可信账本，部署成本相对较高；
- 历史可追溯性需保留本地账本与快照，若被删除或损坏则失效；
- 对CLI型Agent可选省略服务器TEE，但不解决工具执行本身的安全性；
- 目前实现集中于MCP，其他架构的适配仍需进一步验证。

---

## 342. Understanding Geometric Representations in Self-Supervised Vision Transformers via Subspace Intervention

**arXiv ID:** 2607.01987 | [PDF](https://arxiv.org/pdf/2607.01987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 343. From Battlefield to Boardroom: Strategic Red Teaming as an Epistemic Governance Instrument in the Age of AI

**arXiv ID:** 2607.01913 | [PDF](https://arxiv.org/pdf/2607.01913v1)

**作者:** Jeroen Janssen `[一作]` `[通讯]`, Jeroen Janssen

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出并正式化了一种针对 AI 战略采纳的董事会级别的假设压力测试模型，即“战略红队”，并将其拆分为六个治理组件。

**💡 创新点**

创新点在于将 AI 视为五维曝光乘数，构建了基于证据等级的假设评估尺度和独立性报告架构，填补了传统风险清单与战略批准之间的认知空白。

**🔧 技术方法**

采用概念建模、证据等级评估、治理流程设计以及独立性和报告架构等方法论工具，未涉及算法实现。

**📊 数据集**

未使用任何数据集；此工作属于治理设计理论，主要基于文献综述和框架构建。

**📈 对比分析**

提出了四条验证轨道（案例重建、董事会仿真、评审者一致性、现场试点）用于后续比较，但本报告未给出实测性能指标。

**⚠️ 局限性**

局限包括缺乏实证验证、适用性受司法管辖区差异限制、资源需求高、易被组织内部化导致失去独立性、证据等级与概率风险不可混淆。

---

## 344. AI Writes Faster Than Humans Can Review: A Longitudinal Study of an Enterprise 2x Mandate

**arXiv ID:** 2607.01904 | [PDF](https://arxiv.org/pdf/2607.01904v1)

**作者:** Hao He `[一作]` (Carnegie Mellon University), Bogdan Vasilescu `[通讯]` (Carnegie Mellon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对一家中型 AI‑主导企业实施的“2×”生产率提升命令进行为期两年的纵向案例研究，量化 AI 代码工具使用对每位工程师 PR 产出、代码审查负荷以及质量指标的影响。

**💡 创新点**

① 在单一企业中以“2×”目标为自然实验，首次系统地追踪 AI 工具从采用到使用强度累积对生产率的累积效应；② 采用 staggered difference‑in‑differences 与事件研究框架，分离即时采用跳跃与随使用累计增长的收益；③ 结合自动化审查与人工审查的重构分析，揭示生产率提升的组织再设计效应。

**🔧 技术方法**

统计方法：固定效应面板回归、Staggered DiD、事件研究、Poisson 计数模型、Callaway–Sant'Anna 与 Borusyak 估计器；工具使用记录、PR 计数、审查计数、Merge/Reject 率等指标。分析聚焦于在 dev‑level、PR‑level 两层次的回归。

**📊 数据集**

内部企业数据：802 名开发者、196,212 条 PR（含 30.2% AI‑标记）、约 2 年（2024‑04/2026）使用日志（Cursor、Claude Code 及自研代理）、HR 岗位信息。数据完整性保证每名开发者至少三个月活跃，覆盖 113 名未采用者作为对照。

**📈 对比分析**

通过固定效应面板与 staggered DiD，发现每位开发者在 AI 采用后产出平均提升 42%（β₁≈0.35），累计使用进一步提升约 12%/月，综合在 9 个月后达到 2× 的 1.72 倍提升；在新代码库中收益显著（+44%），高层级也受益最大。审查侧显示自动化审查覆盖率提升至 84%，人类审查负荷约翻倍，Merge/Revert 率保持不变，证明质量未显著受损。

**⚠️ 局限性**

① 采用非随机，开发者自选择采用时间和使用强度，导致 β₁ 与 β（累计使用）存在自我选择与反向因果风险；② 仅在单一 AI‑友好企业内完成，外部有效性受限；③ 仅测量 PR 产出和粗略质量指标，未覆盖技术债、所有权减弱等长远成本；④ 不能在公司层面区分模型代替因果效果，因模型发布与组织使用同步；⑤ 自动化审查对实际缺陷的捕捉能力未知，影响对质量的完整评估。

---

## 345. NeoMap: Training-free Novel-View Synthesis from Single Images and Videos

**arXiv ID:** 2607.01962 | [PDF](https://arxiv.org/pdf/2607.01962v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 346. Episodic-to-Semantic Consolidation Without Identity Drift

**arXiv ID:** 2607.01988 | [PDF](https://arxiv.org/pdf/2607.01988v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在受监管的长期部署场景下，提出一种把知识巩固从模型权重或提示中分离出来，只更新可查询的语义层，从而保持代理的身份哈希不变。

**💡 创新点**

核心创新是：① 结构化证明身份不变（Manifest字段固定、语义层不进入哈希输入）；② 设计了可审计、确定性且可 idempotent 的聚合算法；③ 通过只读接口让计划器利用语义事实而不改变身份。

**🔧 技术方法**

技术包括：SHA‑256 对 Manifest 的哈希、基于 SQL 的聚合和 upsert、置信度与观测计数、贝叶斯缩减、规则版本化、审计链记录、HTTP 只读查询端点。

**📊 数据集**

使用合成的事件日志（如 1000 条 grasp 试验、两技能两环境等）以及 10 种随机种子下的 1000 次决策模拟数据。

**📈 对比分析**

与无记忆、raw、uniform、calibrated 四种控制对比。calibrated 控制在 10 种种子下平均降低 79.82%（95% CI [78.02%,81.49%]）的无效尝试；uniform 控制未显著提升，raw 与 calibrated 结果相近；运行时延低于 310 ms/次，查询延迟 < 0.005 ms。

**⚠️ 局限性**

局限性：仅聚合结构化字段，无法处理自然语言摘要；无忘却/衰减机制；不防止恶意写入的 episodic 日志；LLM 生成的抽象不在范畴；跨身份知识迁移缺失；计划器仍可能随语义事实产生行为漂移。

---

## 347. Do Newer Lightweight CNNs Perform Better Under Resource Constraints? A Controlled Multigenerational Study of Architecture, Initialization, Training Budget, and Efficiency

**arXiv ID:** 2607.01984 | [PDF](https://arxiv.org/pdf/2607.01984v1)

**作者:** Tasnim Shahriar `[一作]` `[通讯]` (Independent Researcher), Tasnim Shahriar (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对9个轻量级CNN模型在CIFAR‑10、CIFAR‑100和Tiny ImageNet上进行统一训练、评估与资源度量，探究架构与初始化对性能与资源的影响。

**💡 创新点**

系统化多维资源Pareto前沿分析、统一CPU/GPU延迟测评、对EfficientNet‑B0预训练与scratch训练差异的深入实验，并首次将RepViT‑M1.0与MobileNetV4‑Conv‑S等2024设计纳入同一基准。

**🔧 技术方法**

使用PyTorch实现、AdamW+cosine学习率、混合精度、标准化数据增强、FP16 autocast、MKLDNN、CUDA事件计时以及配套的GPU/CPU测试脚本。

**📊 数据集**

CIFAR‑10、CIFAR‑100、Tiny ImageNet。

**📈 对比分析**

通过top‑1、macro F1、top‑5准确率以及参数/FP32存储/GMACs/峰值CUDA内存/ GPU/CPU延迟等指标构建Pareto前沿；结果显示EfficientNet‑V2‑S在CIFAR任务中最高，RepViT‑M1.0在Tiny ImageNet领跑，EfficientNet‑B0在多维资源下最具稳定性，MobileNet‑V3‑Small在CPU与算力极限下表现最好。

**⚠️ 局限性**

缺乏多次训练种子、仅在单一GPU/CPU环境测试、未考虑量化/能源/不同运行时、未覆盖所有最新轻量化网络、以及预训练权重相关性未完全剔除。

---

## 348. PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation

**arXiv ID:** 2607.01938 | [PDF](https://arxiv.org/pdf/2607.01938v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 349. NAVER LABS Europe Submission to the Instruction-following 2026 Short Track

**arXiv ID:** 2607.01960 | [PDF](https://arxiv.org/pdf/2607.01960v1)

**作者:** Marcely Zanon Boito `[一作]` (NAVER LABS Europe), Ioan Calapodescu `[通讯]` (NAVER LABS Europe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多模态系统，能够从英文语音中完成转写、翻译（德语、意大利语、中文）和多语言问答，使用较小的LLM骨干与改进后的SpeechMapper投影器。

**💡 创新点**

创新点包括：①改进的SpeechMapper投影器和四种对齐损失（L1、余弦、softmax对比、CTC）；②同步训练语音投影器与LoRA适配器的并行多阶段管线；③合成科学演讲数据fakACL以缓解领域不匹配；④在指令格式中将目标语言嵌入问题，提升跨语言泛化。

**🔧 技术方法**

采用了SpeechMapper投影、LoRA适配器、CTC、L1/余弦/softmax对比损失、基于LLM的指令调优、TTS+LLM生成合成数据、LLM-as-judge评估脚本、AdamW优化器及多任务采样策略。

**📊 数据集**

主要使用CoVoST2、EuroParlST、GigaST、LibriSQA等真实数据；通过MT生成额外语言；使用合成科学演讲数据fakACL；评估集包括EuroParl、CoVoST、LibriSpeech、MCIF。

**📈 对比分析**

对比原始backbone、上年最佳系统；在ASR上使用WER、ST/MT使用COMET、SQA/QA使用LLM-as-judge准确率；结果显示尽管LLM参数减半，但在MCIF数据集上仍能超越去年的最佳提交；多任务平衡存在权衡，ASR与SQA性能往往相互影响。

**⚠️ 局限性**

局限性：投影器单独训练效果不佳，对噪声敏感；低参数LLM在零射击和不熟悉领域的表现差；多任务训练难以同时最优，导致在不同任务间需做权衡；合成数据可能引入误导信息。

---

## 350. Underspecification does not imply Incoherence: The Risks of Semantic Collapse in Coding Models

**arXiv ID:** 2607.01953 | [PDF](https://arxiv.org/pdf/2607.01953v1)

**作者:** Cedric Richter `[一作]` (SnT, University of Luxembourg), Mike Papadakis `[通讯]` (SnT, University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过实验研究了大型语言模型在处理任务描述不完整、含糊或矛盾时的行为，并发现模型往往在出现语义崩塌时持续输出同一错误实现。

**💡 创新点**

提出了“有害语义崩塌”（detrimental semantic collapse）概念，证明了提示下的模糊并不必然导致模型产生多样化输出，揭示了以语义不一致性为依据的聚类方法的盲点。

**🔧 技术方法**

使用语义聚类技术对生成代码进行等价类划分；定义 Pass@k、相异度（inconsistency）与语义崩塌率（%SC）等指标；对模型输出进行多次采样并评估；模拟澄清对话来检验聚类式消歧方法；通过温度与采样预算的消融实验探讨参数影响。

**📊 数据集**

实验数据集包括 MBPP、HumanEval 与 LiveCodeBench 三大 Python 代码生成基准及其人工构造的模糊/不完整/矛盾变体；每个基准均含原始规范描述与对应的多种不完整变体。

**📈 对比分析**

比较三种顶尖编码 LLM（Claude Sonnet 4.5、GPT‑4.1‑mini、Qwen3‑32B），在不同采样预算下报告 Pass@k、相异度与 %SC。实验表明，模糊任务导致相异度平均提升但仍有 50–70% 的任务出现语义崩塌，且有害语义崩塌在原始任务中已占 3–32%，在模糊任务中可高达 55%，说明聚类式方法在检测错误与消歧时表现不佳。

**⚠️ 局限性**

局限性包括：仅针对 Python 代码，可能无法推广至其它语言或大型项目；聚类依赖测试集的完整性，易受测试缺失影响；采样预算有限，可能低估崩塌概率；用户澄清模拟不等同真实交互；内部模型不确定性未被直接测量，导致对“有害崩塌”原因理解不够深入。

---

## 351. Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits

**arXiv ID:** 2607.01940 | [PDF](https://arxiv.org/pdf/2607.01940v1)

**作者:** Zhiren Gong `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无标签、输出导向的“条件共消融”得分，用于在已知主电路的基础上恢复隐藏的备份组件，并将其应用于解释、功能消除和结构剪枝。

**💡 创新点**

创新点：① 把自我修复视作重要性加法失效，引入二阶交互的条件增长度量；② 该得分无需先验标签即可发现备份，且在多模型、多规模上迁移；③ 通过恢复备份，闭合了归因、消除与剪枝的盲点，显著提升模型解释性。

**🔧 技术方法**

技术：基于Fisher信息的输出空间度量、中心化的单消融特征、条件消融能量增量、二阶协同量；结合对数几率差分、直接logit归因、反事实补丁、结构化头剪枝；实验使用PyTorch和OpenAI GPT系列。

**📊 数据集**

数据集与实验：GPT‑2‑small IOI 圈电路（手工标注的备份头），随后扩展到 GPT‑2‑medium/large 及 8 个跨家族（Pythia、LLaMA、EleutherAI 等）模型的“归因”电路；使用 WikiText‑2 做零样本 perplexity 测试。

**📈 对比分析**

与传统第一阶方法（单消融、AtP、GIM、EAP‑IG 等）对比，条件共消融在备份头 ROC‑AUC 由 0.33/0.82 提升至 0.91；在归因、消除与剪枝任务中均取得显著改进：归因 logit‑差 1.76、消除准确率 0.70、Pruning 80‑级稀疏率下 perplexity 仅比基线低约 30%。

**⚠️ 局限性**

局限性：① 主要针对“沉睡‑替代”自我修复场景，若备份不完全沉睡则效果减弱；② 仍需先有主电路种子或高置信度的先行发现；③ 对大规模模型的计算成本为 O(|seed|) 前向传递，虽然低于全二阶枚举，但在 7B 级别仍需显著 GPU 资源；④ 对非注意力或 MLP 主导电路的迁移能力尚不充分。

---

## 352. Beyond Textual Repository Exploration: Dual-Modal Structural Reasoning for Agentic Issue Resolution

**arXiv ID:** 2607.01929 | [PDF](https://arxiv.org/pdf/2607.01929v1)

**作者:** Jiayi Zhang `[一作]` (Nanyang Technological University), Chunyang Chen `[通讯]` (TU Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双模态结构化框架，利用四种软件结构图（模块耦合图、函数调用图、类层次图、程序依赖图）为缺陷修复代理提供可视化+文本响应，以支持仓库级的长程探索与定位。

**💡 创新点**

创新点包括：1）将仓库结构外部化为图抽象并同步提供视觉和文本两种观测；2）通过图视图代替碎片化文本，显式呈现多跳依赖与层次关系；3）设计可参数化的查询接口，支持在不同抽象层级上快速切换。

**🔧 技术方法**

使用技术包括：大语言模型（Claude‑4.5 Sonnet、Kimi‑K2.5、Gemini‑3 Flash）、图查询引擎（MCP 服务）、图可视化（Graphviz）以及基于 AST 的四种结构图构建器。

**📊 数据集**

实验使用 SWE‑Bench Pro（731个长程实例）和 SWE‑Bench Verified（500个真实仓库实例），并在两者上评估多种代理与模型组合。

**📈 对比分析**

通过与四个文本中心代理（mini‑SWE‑agent、SWE‑agent、Live‑SWE‑agent、OpenCode）及两种基于文本的图检索基线（RepoGraph、CodeGraph）对比，实验显示：在 Pro 上使用 Kimi‑K2.5 时提升 46 个已解决实例（从 342 → 388），在 Verified 上提升 20 个实例（从 381 → 401）且成本平均降低 0.18 美元/实例。

**⚠️ 局限性**

局限性：1）仅支持静态图构建，无法处理运行时动态依赖；2）视觉渲染与图结构的生成耗时，可能限制极大仓库的即时查询；3）需要在目标代理中额外实现查询接口，增加集成复杂度；4）目前验证范围局限于公开的 SWE‑Bench 生态，需进一步验证跨语言/跨规模场景。

---

## 353. Sparse-Aware Vector Quantization for Bandwidth-Efficient Collaborative 3D Semantic Occupancy Prediction

**arXiv ID:** 2607.01928 | [PDF](https://arxiv.org/pdf/2607.01928v1)

**作者:** Feng Li `[一作]` (Tianjin University), Gong Chen `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VQSOP 框架，旨在实现带宽高效的协同 3D 语义占用预测，采用稀疏感知向量量化（SAVQ）对关键空间区域进行离散编码，再通过双分支自适应空间细化（ASR）模块提升几何细节与语义一致性。

**💡 创新点**

创新点包括：① 稀疏感知向量量化，利用空间稀疏性与学习型码本仅传输重要前景的索引；② 双分支自适应细化模块，将局部卷积与膨胀卷积融合并通过空间自适应权重动态组合；③ 通过上述两项技术实现了 82 倍的通信压缩率，并在保持甚至提升感知精度的同时显著降低带宽需求。

**🔧 技术方法**

使用了：向量量化与可学习码本、基于置信度的稀疏选择器、3D 卷积与膨胀卷积、空间自适应加权融合、基于 V2X 的消息解压与聚合、AdamW 优化器、线性 warmup 与余弦退火学习率调度等。

**📊 数据集**

在 Semantic-OPV2V 数据集上进行评估，该数据集为 OPV2V 的语义占用扩展版，包含 2–7 车协同场景，配备 LiDAR 与多向摄像头。

**📈 对比分析**

与 CoHFF、GaussianFormer、GSFusion 等现有协同 3D 语义占用方法对比，VQSOP 在协同设置下实现 73.79% IoU、41.54% mIoU，通信量仅 0.013 MB（相比 1.07 MB 低 82 ×），同时在单机设置下也获得最高 IoU 70.40% 与 mIoU 33.61%，显著优于所有基线。

**⚠️ 局限性**

局限性包括：① 对置信度阈值 τ 的敏感性，过高阈值会导致重要前景被裁剪；② 需要共享码本与同步传输，若码本版本不一致会影响解码；③ 主要在仿真数据验证，真实世界噪声与不稳定网络下的鲁棒性尚未充分评估；④ 目前仅针对语义占用任务，扩展到更大范围感知（如动态遮挡、复杂几何）仍需进一步研究。

---

## 354. TUDUM: A Turkish-Thinking Reasoning Pipeline for Qwen3.5-27B

**arXiv ID:** 2607.01927 | [PDF](https://arxiv.org/pdf/2607.01927v1)

**作者:** Baran Bingol `[一作]` (Ankara University), Bahaeddin Turkoglu `[通讯]` (Ankara University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了一个基于 Qwen3.5‑27B 的 Turkish‑thinking pipeline，利用 LoRA SFT 让模型在思考区块中生成土耳其语推理轨迹，并通过 GRPO‑family 强化学习在代理数学环境中优化正确性、格式、语言一致性和长度，最终公开 step‑50 版本模型；

**💡 创新点**

在大语言模型中首次明确区分答案语言与推理轨迹语言，提出使用 LoRA 快速适配土耳其语推理，并设计多奖励归一化的 GRPO‑family RL，以在数学任务中恢复并提升性能，同时保持推理过程的土耳其语可读性；

**🔧 技术方法**

LoRA 参数微调、GRPO‑family 强化学习（DAPO/GDPO 风格）、多奖励归一化、Token‑level DAPO loss、soft length penalty、代理数学验证器等；

**📊 数据集**

15991 条土耳其语推理示例（数学 9566 条、科学 5134 条、代码 652 条、系统提示 639 条），来源于公开的 AIME、Turkish science prompts、Kimi‑K2.5 judge、DeepSeek teacher 以及 proxy‑filtered DAPO‑Math 5k 训练集；

**📈 对比分析**

与基准 Qwen3.5‑27B 对比，SFT 后模型的推理轨迹更短、语言更一致，但 Macro‑6（AIME24/25、Turkish MMLU、GPQA、HumanEval、IFEval prompt）平均从 81.7% 降至 75.8%；RL step‑50 在 AIME24 上达 86.7%（高于基准 82.2%），部分数学指标提升，但整体 Macro‑6 仍低于基准；SFT 通过推理语言一致性带来显著行为改进；

**⚠️ 局限性**

RL 仅在数学代理环境中训练，未能统一恢复所有基准性能；数据集与奖励范围有限，缺少 instruction、coding、science 等多样任务；缺少完整可复现细节（版本哈希、随机种子等）；推理轨迹并非真实内部计算，只是训练目标的输出格式，评价属于内部项目水平。

---

## 355. Rethinking Post-Hoc Calibration in Semantic Segmentation

**arXiv ID:** 2607.01902 | [PDF](https://arxiv.org/pdf/2607.01902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 356. Traceable Fault Diagnosis for Battery Energy Storage Systems via Retrieval-Augmented Multi-Agent O&M Assistant

**arXiv ID:** 2607.01992 | [PDF](https://arxiv.org/pdf/2607.01992v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 357. Low-Latency Task-Oriented Image Transmission with Opportunistic Spectrum Access

**arXiv ID:** 2607.01921 | [PDF](https://arxiv.org/pdf/2607.01921v1)

**作者:** João Henrique Inacio de Souza `[一作]` (Aalborg University), Petar Popovski `[通讯]` (Aalborg University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 VQ‑VAE 的任务导向低延迟图像传输框架，并在机会性频谱访问环境下实现了高效的图像远程推理。

**💡 创新点**

将离散低维 VQ‑VAE 潜在表示与传统数字调制无缝对接，既显著压缩比又不需要重传，适应机会性频谱访问；并提供了端到端延迟-准确率的统计分析模型。

**🔧 技术方法**

使用 VQ‑VAE 学习离散编码、BPSK/多元调制、LDPC 纠错、机会性频谱感知、负二项分布延迟建模等技术。

**📊 数据集**

采用 Imagenette 数据集（128×128 像素）进行训练与测试，并用 MobileNetV3‑Large 进行图像分类任务。

**📈 对比分析**

与 PNG+LDPC、JPEG+LDPC、原始 RGB+LDPC 等传统源码+信道码方案比较，在相同或更低延迟下仅牺牲 2% 左右的分类准确率，并在 8 码本尺寸下实现 79× 延迟减小。

**⚠️ 局限性**

受限于离散码本大小与低维压缩导致的重建失真；在极低 SNR 时仍会出现重建错误；且未考虑多用户多跳或多天线等更复杂场景。

---

## 358. SFKD: Spatial--Frequency Joint-Aware Heterogeneous Knowledge Distillation via Multi-Level Wavelet Spectral Interaction

**arXiv ID:** 2607.01906 | [PDF](https://arxiv.org/pdf/2607.01906v1)

**作者:** Cuipeng Wang `[一作]` (Fudan University), Haipeng Wang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究跨异构模型的知识蒸馏，提出了 Spatial–Frequency Joint‑Aware Heterogeneous Knowledge Distillation (SFKD) 框架；

**💡 创新点**

创新点在于将多级离散小波变换与傅里叶频域的高斯滤波相结合，形成空间‑频率联合感知的蒸馏策略，并设计双流双阶谱细化 (DS²SR) 模块来进一步提升学生子带的结构一致性；

**🔧 技术方法**

主要技术包括多级离散小波变换 (MDWT)、双流双阶谱细化模块 (卷积 + 自注意力)、Gaussian‑Filtered Frequency Loss (GFFL) 与 InfoNCE 对比损失；

**📊 数据集**

使用的公开数据集为 CIFAR‑100 与 ImageNet‑1K；

**📈 对比分析**

与多种同质与异构蒸馏基线（如 KD, FitNet, FBT, OFA 等）进行对比，SFKD 在 CIFAR‑100 上平均提升约 10.9% 终端精度，在 ImageNet‑1K 上平均提升约 2.5%，并在多组师生对中多次位列或接近榜首；

**⚠️ 局限性**

局限性包括：对极端异构对（如 Transformer‑Transformer）提升有限；学生模型表征容量仍有限，导致子带重构不够稳定；在更大规模或多任务场景下的泛化性能尚未充分验证。

---

## 359. Multimodal Knowledge Edit-Scoped Generalization for Online Recursive MLLM Editing

**arXiv ID:** 2607.01978 | [PDF](https://arxiv.org/pdf/2607.01978v1)

**作者:** Siyuan Li `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 ScopeEdit，一种在线多模态知识编辑框架，能够在编辑过程中精确控制知识的传播范围，保证在指定语义范围内的跨模态泛化，同时在范围之外保持局部性。

**💡 创新点**

创新点主要包括：① 引入“Edit‑Scoped Generalization”概念，将编辑视为在语义范围内传播与范围之外不泄漏的双重目标；② 将每次更新拆分为本地吸收分支和共享传播分支；③ 通过低秩写入空间的正交分离与交叉模态证据门控（方向一致性与支持度）实现可控传播；④ 采用递归预条件化（基于历史二阶统计）和 Sherman‑Morrison 递推，使每次编辑保持常数时间与空间复杂度。

**🔧 技术方法**

使用技术包括：低秩写入（rank‑k）与正交基、FFN key‑value 视角的参数修改、双分支（loc & sh）写入、交叉模态证据门控（cosine 与 support）、递归预条件化（C_t‑1 统计）、Sherman‑Morrison 更新、以及与现有 LLM 编辑方法的对齐实现。

**📊 数据集**

评估数据集包括 E‑VQA、E‑IC、VLKEB 以及多种 MLLM 后端（BLIP2‑OPT、LLaVA‑v1.5、Qwen2‑VL、Qwen3‑VL），通过这些数据集考察可靠性、跨模态一般性、局部性和可携带性等指标。

**📈 对比分析**

与 FT‑L/FT‑M/MEND/AlphaEdit/M‑ORE/SERAC/IKE/LiveEdit 等主流参数修改与参数保持编辑器进行系统对比。ScopeEdit 在所有模型和任务上均获得最优或显著提升，特别是在跨模态一般性与局部性平衡、长期编辑稳定性以及在线效率（恒定 per‑edit 计算与内存）方面表现突出；在 VLKEB 实际滑动编辑场景中，同样保持了高可靠性、一般性与可携带性。

**⚠️ 局限性**

局限性包括：① 需要手工设定门控阈值与分支比例，缺乏完全自适应范围发现；② 对低秩写入维度 r 的敏感性，过小或过大均可能影响编辑效果；③ 在证据弱或模态不对齐的情况下，门控可能失效；④ 对极大规模模型或不同架构的适配仍需进一步验证；⑤ 依赖 FFN key‑value 视角，可能不适用于所有 MLLM 设计。

---

## 360. A Multi-Branch Hierarchy-Aware Framework for Heterogeneous Audio Classification

**arXiv ID:** 2607.01974 | [PDF](https://arxiv.org/pdf/2607.01974v1)

**作者:** Beile Ning `[一作]` (Wuhan University), Gongping Huang `[通讯]` (Wuhan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在DCASE2026 Task1中，作者设计并实现了一个以CLAP音频-文本嵌入为核心的多分支模型，用于BST分类。

**💡 创新点**

创新点在于：扩展训练集BSD-Grand以降低噪声；引入log-Mel、MFCC、log-STFT等声学分支并用高通门适配器提升语义融合；采用层次化分类头（Flat/GC/LCL）与KNN后处理及知识蒸馏进一步优化预测。

**🔧 技术方法**

技术包括CLAP预训练模型、特征分支（MDFD、TDNN、Transformer）、高通门适配器、Flat/GC/LCL分类头、KNN邻居后处理与蒸馏、5折交叉验证和集成投票。

**📊 数据集**

使用的数据集为BSD10k-v1.2与从BSD35k筛选得到的BSD-Grand，共计20529个样本。

**📈 对比分析**

与CLAP基线相比，单模型在Hier. F1从78.45%提升到80.84%，集成模型进一步达到81.25%，显著优于基线。

**⚠️ 局限性**

局限在于：仍需人工过滤上传者偏差，模型结构复杂且推理时需要多分支与KNN检索，计算成本较高。

---

## 361. Probabilistic Low-Voltage Peak Load Forecasting with Time Series Foundation Models Evaluated on Application-Oriented Metrics

**arXiv ID:** 2607.01966 | [PDF](https://arxiv.org/pdf/2607.01966v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 362. Assessing VLM Reliability for Medical Image Quality Evaluation Under Corruption and Bias

**arXiv ID:** 2607.01973 | [PDF](https://arxiv.org/pdf/2607.01973v1)

**作者:** Sofiane Ouaari `[一作]` (University of Tuebingen), Nico Pfeifer `[通讯]` (University of Tuebingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估多款视觉‑语言模型在医学图像质量评估中的可靠性，系统探讨图像失真与文本偏见对评分的影响。

**💡 创新点**

提出零样本 VLM‑as‑a‑Judge 的完整基准框架，结合嵌入空间分析与文本偏见实验，首次量化像素化失真和元数据偏见对模型判断的显著性。

**🔧 技术方法**

采用零样本 VLM‑as‑a‑Judge、PCA 嵌入距离、百分比变化计算和文本偏见注入等技术，评估模型对不同失真和文本条件的敏感度。

**📊 数据集**

使用 MediMeta-C 数据集（7 种失真类型、5 个强度级别）以及原始 MediMeta 清晰图像进行实验。

**📈 对比分析**

对 16 款开源 VLM（11 通用 + 5 医学）进行平均评分、百分比变化、嵌入距离和模型间相关性比较；像素化导致平均 -20.58% 评分，亮度影响最小；部分模型在噪声下评分提升，表明存在错误的特征映射。

**⚠️ 局限性**

VLM 对严重失真缺乏鲁棒性，且对文本元数据高度敏感，存在显著偏见；缺乏针对医疗场景的专门训练与微调，未验证在真实临床流程中的泛化能力。

---

## 363. Object Aligner: A Configurable JSON Schema Similarity Score for Graphs, Applied to LLM Prompt Optimization

**arXiv ID:** 2607.01972 | [PDF](https://arxiv.org/pdf/2607.01972v1)

**作者:** Jan Drchal `[一作]` `[通讯]` (Czech Technical University in Prague), Jan Drchal (Czech Technical University in Prague)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Object Aligner，一种基于 JSON Schema 扩展的可配置、确定性 JSON 对象相似度评估工具；

**💡 创新点**

创新点包括：① 可通过 JSON Schema 注解实现完全无代码配置；② 引入参照对齐（referential alignment）实现对图/超图的标识符重映射不变性；③ 支持顺序敏感序列对齐以及可配置的关键字权重；④ 在评估时自动生成按分数递减排序的修复建议，提供确定性反馈；

**🔧 技术方法**

技术上采用递归分配匹配：使用匈牙利算法处理无序集合、动态规划对顺序敏感序列匹配、基于 Weisfeiler–Leman 颜色细化推断 ID 之间的双射，所有比较均在纯 Python 中实现，无需外部模型；

**📊 数据集**

实验数据集包括：① 组织结构→图的 Org2Graph（合成）；② 事实→有序列表的 Facts2Order（合成）；③ 真实世界数据如 AMR、SciERC、BioRED、NATURAL PLAN、ROCStories 等，均经 JSON 处理并附有对应原始评价指标；

**📈 对比分析**

方法比较：在内部实验中验证了对标识符重映射和顺序敏感的准确性；在外部实验中将 Object Aligner 作为 GEPA 的奖励信号与传统精确匹配或 Smatch 等指标对比，结果显示：① 对识别符不可区分的图数据（如 Org2Graph、Bio AMR）显著提升；② 对需要顺序正确的任务（如 Facts2Order、NATURAL PLAN）也有明显收益；③ 确定性反馈在多种任务中均提升了优化效果；性能上算力占用主要来自 O(n³) 的分配匹配，但在实际 JSON 长度不大时表现可接受；

**⚠️ 局限性**

局限性包括：① 分数为序数尺度，缺乏校准；② 参照对齐使用 1-WL 近似，可能在高阶同构上失效；③ 对自引用或相互引用的作用域处理不完整；④ 仅在 GEPA 与单一 LLM 上验证，结果是否能迁移至其他优化框架未知；⑤ 超参数探索有限，未系统评估不同 Schema 设计和反馈列表长度的影响。

---

## 364. Towards a Phonology-Informed Evaluation of Multilingual TTS

**arXiv ID:** 2607.01965 | [PDF](https://arxiv.org/pdf/2607.01965v1)

**作者:** Sneha Ray Barman `[一作]` (IIT Guwahati), Shakuntala Mahanta `[通讯]` (IIT Guwahati)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建基于分类器的评估框架，利用人类语音训练的ATR分类器检测TTS输出中的音位对比失真。

**💡 创新点**

首次提供面向特定语言音位对比的可解释性评估方法，揭示TTS在实现语法条件音位（如Assamese ATR和声）时的系统性偏差。

**🔧 技术方法**

使用逻辑回归与随机森林分类器、Lobanov归一化、F1/F2/F3提取、词级特征聚合等技术。

**📊 数据集**

人类 Assamese 语料库（14名说话人）与Meta MMS TTS生成的 114 词样本（其中 80 与人类重叠）。

**📈 对比分析**

通过交叉域评估（人→人、人→TTS、TTS→TTS、TTS→人）和词级/音素级不一致率比较，发现 TTS 在 +ATR 中低元音上存在 7:1 的低产生误差，准确率与宏 F1 与人类相近但存在显著方向性差异。

**⚠️ 局限性**

受限于样本量不足、词级标签稀缺、仅评估单一语言单一音位对比，结果无法直接推广至更广泛的语音合成系统或其他语言。

---

## 365. Beyond Supervised Clarification: Input Rewriting with LLMs for Dialogue Discourse Parsing

**arXiv ID:** 2607.01964 | [PDF](https://arxiv.org/pdf/2607.01964v1)

**作者:** Yiming Liu `[一作]` (University of Oklahoma), Jie Cao `[通讯]` (University of Oklahoma)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在增量对话语篇解析中，探索在不使用监督澄清数据的情况下，通过修改最后一句话的输入来提升冻结的解析器性能。

**💡 创新点**

提出将澄清视为选择性干预问题，强调在能否修复的判定（可修复性预测）是提升性能的关键。

**🔧 技术方法**

使用零-shot提示、强化学习（GRPO）以及多种 LLM 生成/判别解析器进行实验，评估不同程度的输入重写策略。

**📊 数据集**

在三个 SDRT 数据集 STAC、Molweni、MSDC 上进行实验，使用 Qwen3‑8B/14B、SDDP 等解析器。

**📈 对比分析**

与无澄清基线和 parser‑agnostic 规则重写进行对比，发现大多数规则重写导致回归，RL 策略在部分数据集可减少回归 37%，但整体提升有限，且约 80% 错误无法通过重写修复。

**⚠️ 局限性**

局限在于仅考虑最后一句重写，未覆盖全局上下文；仅评估两类解析器，未探讨其他模型；缺乏对监督澄清的直接对比；可修复性判定仍需改进。

---

## 366. ContextSniper: AntTrail's Token-Efficient Code Memory for Repository-Level Program Repair

**arXiv ID:** 2607.01916 | [PDF](https://arxiv.org/pdf/2607.01916v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 367. CausalSteward: An Agentic Divide-Conquer-Combine Copilot for Causal Discovery

**arXiv ID:** 2607.01936 | [PDF](https://arxiv.org/pdf/2607.01936v1)

**作者:** Nicholas Tagliapietra `[一作]` (TU Darmstadt), Kristian Kersting `[通讯]` (TU Darmstadt)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 CausalSteward——一种多代理、带有人机交互、分而治之的因果图学习框架，能够在高维数据下自动整合先验知识并构建大规模因果模型。

**💡 创新点**

创新点在于将 LLM 与检索增强生成（RAG）相结合，形成可动态提问的 Human‑in‑the‑Loop 代理，并通过分层划分（Divide‑and‑Conquer）显著提升可扩展性和识别性；同时设计了多阶段（Explain‑Divide‑Conquer‑Combine）协同流程，实现对局部因果子图的并行学习与全局合并。

**🔧 技术方法**

采用的技术包括：大语言模型（GPT‑4o‑mini、o3‑mini、Qwen3‑14B）、RAG 搜索引擎、约束式因果发现算法（如 FCI、DAGMA）、分层划分与合并策略，以及多代理交互与评估框架。

**📊 数据集**

实验使用了工业制造数据集 CausalMan（Small/Medium）、Neuropathic‑Pain、CausalChambers（light‑tunnel 设备）以及经典 ASIA 图作为案例验证；通过这些多域、高维、含隐藏变量的数据集检验方法。

**📈 对比分析**

与七个基线（XGES、GranDAG、FCI、BOSS、CausalCopilot、LLM‑BFS、LLM‑Pairwise）以及自研的 ablation 版本相比，CausalSteward 在 F1、精度（Precision）上均有显著提升（尤其在 HITL+RAG 模式下），且 SHD（结构汉明距离）降低，表明模型更准确且鲁棒；在大规模数据下保持子线性增长的运行时与令牌消耗。

**⚠️ 局限性**

主要限制包括：对高质量先验知识和检索结果的依赖，LLM 的推理与指令遵循能力决定性能；需要人工专家参与（HITL）以校正错误；在极大规模数据集或不完善的 RAG 源时，分层划分与合并仍可能产生连通性错误；缺乏自适应的误差补偿机制。

---

## 368. Real-weighted Diameter and Eccentricity of Minor-free and Bounded VC-dimension Graphs in Truly Subquadratic Time

**arXiv ID:** 2607.01926 | [PDF](https://arxiv.org/pdf/2607.01926v1)

**作者:** Da Wei Zheng `[一作]` `[通讯]`, Da Wei Zheng

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了首个适用于实值加权有向图的子二次时间直径与球面度计算算法，特别针对 K_h-无环图与更广泛的多项式扩张图类；

**💡 创新点**

通过引入随机搜索到决策的归约，并利用随机见证技术突破实值权重下 VC‑维度方法的限制，实现了真正子二次时间的球面度求解；

**🔧 技术方法**

核心技术包括：1）随机搜索-决策归约与随机见证产生；2）利用有向图的平衡分离器与 r‑划分；3）利用广义距离 VC‑维度与低平均交叉扫描路径；4）对内部与边界顶点的距离预处理与迭代更新；

**📊 数据集**

论文主要为理论分析与算法设计，未使用具体数据集进行实验；

**📈 对比分析**

与以往只适用于无权或整数权重的 VC‑维度算法相比，本工作在实值权重下实现了 (n^2-1/(2h-2)) 的时间复杂度，显著提升了对 K_h-无环图的处理效率；

**⚠️ 局限性**

局限性包括：1）算法依赖子线性平衡分离器；2）不适用于计算Wiener指数等其他距离度量；3）目前尚缺乏低直径分解在实值权重图中的实现。

---

## 369. Markovian Arrival Process Parameter Estimation of Quasi-birth-death Queueing Systems with Utilization Data

**arXiv ID:** 2607.01914 | [PDF](https://arxiv.org/pdf/2607.01914v1)

**作者:** Chen Li `[一作]` (University of Osaka), Tadashi Dohi `[通讯]` (Hiroshima University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于利用率监测数据的EM算法，用于估计由MAP驱动的QBD排队系统的到达、服务及相位参数。

**💡 创新点**

创新点在于：①首次将利用率数据拆分为可观测与不可观测区间，并推导相应的期望充分统计量；②在此框架下设计完整的EM推断过程；③引入AIC进行阶段数选择，避免过拟合。

**🔧 技术方法**

核心技术包括：Markovian Arrival Process（MAP）建模、Quasi‑Birth‑Death（QBD）过程描述、Expectation‑Maximization（EM）算法以及均匀化技术以降低计算复杂度。

**📊 数据集**

实验使用合成的CPU利用率监测数据（每个监测周期包含不可观测与可观测两段时间），并在不同阶段数下验证算法效果。

**📈 对比分析**

与传统基于完整事件序列（到达时间、等待时间等）的MLE方法比较，结果表明即使仅利用率数据，EM算法仍能得到与完整数据相近的参数估计，且在模型复杂度较高时仍保持良好拟合。

**⚠️ 局限性**

局限性包括：①对可观测区间内最多一次状态变化的假设限制了适用范围；②随着MAP相位数和队列容量增大，计算量急剧增加；③目前仅验证了指数服务时间的M/1/K模型，扩展到多服务器或非指数服务分布仍需进一步研究。

---

## 370. Rethinking Complexity Metrics for LLM-Integrated Applications: Beyond Source Code

**arXiv ID:** 2607.01903 | [PDF](https://arxiv.org/pdf/2607.01903v1)

**作者:** Zihao Xu `[一作]` (University of New South Wales), Zhenchang Xing `[通讯]` (CSIRO's Data61)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了 Hecate 静态分析工具，用来衡量 LLM 集成应用的复杂度，包括自然语言提示、代码以及两者交互的结构。

**💡 创新点**

创新点在于首次将 Prompt-as-Specification 模型引入复杂度评估，构造了 52 个跨层维度的候选指标，并验证出 10 个能够超越传统代码复杂度指标的新度量。

**🔧 技术方法**

采用静态代码分析、自然语言文本分割、Hoare 逻辑式提示建模以及部分基于统计的词法与语义特征提取技术。

**📊 数据集**

使用 18 个开源 LLM 集成项目共 118 个组件（LOC 50–1902）做训练验证，并在 6 个未见仓库的 20 个组件上做泛化测试。

**📈 对比分析**

通过控制代码大小后的偏相关检验，10 个指标在三种维护难度维度上均显著相关，其中 n_mem_refs 和 n_llm_calls 的相关系数最高（+0.40、+0.38），显著优于传统基准指标。

**⚠️ 局限性**

局限包括仅基于 VCS 历史的间接维护信号、仅在 Python 环境下实现、未覆盖运行时动态提示生成与多代理交互、以及对企业级大规模项目的可迁移性尚未验证。

---

## 371. LiZAD: A Lightweight Zero-Shot Anomaly Detection Framework for Industrial Manufacturing

**arXiv ID:** 2607.01949 | [PDF](https://arxiv.org/pdf/2607.01949v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 372. SABER: A Semantic-Aligned Brain Network Analysis Framework via Multi-scale Hypergraphs

**arXiv ID:** 2607.01901 | [PDF](https://arxiv.org/pdf/2607.01901v1)

**作者:** Yidan Xu `[一作]` (Hangzhou Dianzi University), Huihui Ye `[通讯]` (Hangzhou Dianzi University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 SABER 框架，将大语言模型（LLM）生成的语义知识与多尺度超图结构相结合，用于脑功能网络的疾病诊断。

**💡 创新点**

创新点包括：① 将语义从传统的辅助特征提升到决策层级，利用 Graph‑CLS token 与跨注意力机制实现语义主动引导；② 采用多尺度超图神经网络捕捉高阶、跨 ROI 的交互；③ 设计分层语义对齐与门控残差注入，确保语义在不同层级的稳定融合。

**🔧 技术方法**

技术手段：LLM 文本生成与编码（ChatGPT‑5、Llama‑encoder‑1.0B）；脑网络 Transformer 自注意力；多尺度超图神经网络（HGNN）；多头自注意力、动态融合、门控残差注入；对比损失与交叉熵联合训练；AdamW + cosine 学习率调度。

**📊 数据集**

使用公开脑网络数据集 ABIDE I（ASD 与 NC）和 ADHD‑200（ADHD 与 NC）进行评估。

**📈 对比分析**

与 BrainNetCNN、HGNN、BrainGNN、BrainPrompt 等多种基线方法对比；在 ABIDE 上准确率达 71.58%、AUC 70.40%；在 ADHD‑200 上准确率 66.21%、AUC 66.97%；SABER_lite（无 LLM）已超越多数基线，SABER 进一步提升性能，表现出更高的稳定性与可解释性。

**⚠️ 局限性**

局限性：① 对 LLM 文本生成质量敏感，噪声文本可能影响结果；② 训练时间与计算成本较高；③ 对极小样本情况仍需更强正则或数据增强；④ 仅在 ASD 与 ADHD 两个疾病上验证，需扩展至更多病种；⑤ 对超参数（如尺度、注意力头数）较为敏感。

---

## 373. Liquid Latent State Dynamics for Interpretable Turbofan Degradation Modeling

**arXiv ID:** 2607.01986 | [PDF](https://arxiv.org/pdf/2607.01986v1)

**作者:** Weizhi Nie `[一作]` (Tianjin University), Yuting Su `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种分解液体神经网络的潜在动态模型，用于航空发动机健康监测，既能预测未来传感器数据，又能产生可解释的退化状态轨迹。

**💡 创新点**

将潜在状态分解为退化与运行条件两部分，并通过液体递归单元学习可调节时间常数的状态增量，同时加入 RUL、单调性、潜在一致性与去相关损失，使退化子空间形成有序的退化轴。

**🔧 技术方法**

使用液体时间常数网络、GRU 编码器/解码器、重构损失、RUL 回归损失、单调风险损失、潜在一致性损失、去相关损失以及 Spearman 相关分析和 PCA 可视化等技术。

**📊 数据集**

实验基于 NASA C‑MAPSS 涡轮风扇退化基准，包含四个子集 FD001–FD004。

**📈 对比分析**

与 GRU 基线及多种液体模型对比；在多条件子集 FD002、FD004 上，传感器预测 RMSE 从 0.1058/0.0936 降至 0.0627/0.0625，退化状态速度相关性提升至 0.596，但 RUL RMSE 仍略逊于 GRU。

**⚠️ 局限性**

退化状态尚未完全校准为 RUL 预测，条件子空间存在退化泄漏，且模型对真实工业数据（维护事件、缺失观测、非单调运行）适用性尚待验证。

---

## 374. Zeus: Towards Tuning-Free Foundation Model for Time Series Analysis

**arXiv ID:** 2607.01918 | [PDF](https://arxiv.org/pdf/2607.01918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 375. Robust Image Processing Techniques for Construction Environment Monitoring Using Underwater Robots

**arXiv ID:** 2607.01915 | [PDF](https://arxiv.org/pdf/2607.01915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 376. OntoLearner: A Modular Python Library for Ontology Learning with Large Language Models

**arXiv ID:** 2607.01977 | [PDF](https://arxiv.org/pdf/2607.01977v1)

**作者:** Hamed Babaei Giglou `[一作]` (Leibniz Information Centre for Science and Technology), Sören Auer `[通讯]` (Leibniz Information Centre for Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了OntoLearner——一个模块化的Python库，旨在统一本体学习（Ontology Learning）中的本体访问、基于大型语言模型（LLM）的学习管道以及标准化的基准评估。

**💡 创新点**

创新点包括：①首次公开180个跨22个领域、可机器读取、已划分训练/验证/测试集的本体数据集；②构建了统一的“Ontologizer”模块，实现本体的跨域导入、版本管理和度量；③设计了可扩展的学习任务和模型接口，支持检索增强生成（RAG）以及多任务组合；④提出了本体复杂度评分体系并进行鲁棒性分析；⑤通过大规模实验（22个检索模型、12个LLM）揭示本体复杂度是性能瓶颈。

**🔧 技术方法**

主要技术包括Python实现、HuggingFace与GitHub集成、LLM推理（Qwen、Gemma、Mistral、Falcon等）、检索模型（词典、传统嵌入、句子Transformer、密集检索、混合检索）、RAG流水线、可插拔的学习器、合成语料生成以及本体度量与复杂度评估。

**📊 数据集**

使用的数据集为180个已标注、机器可读的本体，覆盖22个领域；对每个本体自动生成term‑typing、taxonomy‑discovery和non‑taxonomic relation extraction的标准化train/dev/test拆分；还构建了Synthetic Corpus Generator用于生成训练文本。

**📈 对比分析**

通过在所有26个代表性本体上比较22个检索模型与12个LLM，采用Recall@15和F1作为评估指标。结果表明：检索单独无法完成所有任务；term‑typing表现最优；taxonomy‑discovery始终是最难的；LLM规模对term‑typing提升显著但对taxonomy‑discovery提升有限；LLM‑增强检索可显著提升检索性能；模型的输出规范性比推理深度更关键。

**⚠️ 局限性**

局限性主要体现在：①本体复杂度导致的结构匹配瓶颈，当前LLM与检索仍难以在大规模层级中保持一致；②评估仍以特定任务与领域为中心，跨域迁移性不足；③对逻辑一致性与推理深度的支持有限；④合成语料和度量方法仍有改进空间；⑤部分功能（如可视化、交互式UI）尚未完整实现。

---

## 377. A-TMA: Decoupling State-Aware Memory Failures in Long-Term Agent Memory

**arXiv ID:** 2607.01935 | [PDF](https://arxiv.org/pdf/2607.01935v1)

**作者:** Zitong Shi `[一作]` (National University of Singapore), Anthony Kum Hoe Tung `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并解决LLM代理长期记忆中的“幽灵记忆”问题，提出状态感知覆盖层来管理记忆并防止过时或过渡事实误导回答。

**💡 创新点**

创新点在于：1）从银行维护、检索、回答三层拆解记忆失败；2）引入状态标签（当前、历史、过渡）并通过覆盖层保持超越和过渡记录；3）构建证据包以实现查询时的状态视图；4）提出专门的LTP（LoCoMo Temporal Plus）冲突密集基准用于量化幽灵记忆。

**🔧 技术方法**

使用的技术包括：状态感知记忆覆盖层、证据包构建、分离式评估（银行、检索、回答层），以及与现有记忆系统的无缝集成。

**📊 数据集**

使用的数据集为LTP（LoCoMo Temporal Plus）以及LoCoMo会话扩展数据，用于评估长期对话中的时序一致性。

**📈 对比分析**

与基线相比，LTP上冲突准确率提升0.240；在LoCoMo上时序F1从0.0295提升至0.1705，显示显著性能提升，尽管提升幅度受主机环境影响。

**⚠️ 局限性**

限制在于：改进效果受执行环境（host）依赖；仍存在隐藏的记忆错误，最终QA准确率可能掩盖具体的幽灵记忆位置。

---

## 378. Towards Real-World Ultrasound Understanding: Large Vision-Language Models from Multi-Image Examinations with Long-Form Reports

**arXiv ID:** 2607.01908 | [PDF](https://arxiv.org/pdf/2607.01908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 379. SPLC: Social Preference Learning for Crowd Robot Navigation

**arXiv ID:** 2607.01925 | [PDF](https://arxiv.org/pdf/2607.01925v1)

**作者:** Zixuan Chen `[一作]` (Wuhan University of Science and Technology), Shiquan Zheng `[通讯]` (Wuhan University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用社交偏好学习（SPLC）自动生成偏好标签并训练奖励模型，再与离线强化学习（IQL/CQL/TD3BC）结合，实现人群机器人导航；

**💡 创新点**

① 社交偏好反馈机制可无人工标注自动生成偏好数据；② 采用Preference Transformer学习非马尔可夫奖励；③ 以碰撞、目标进度和风险暴露为层级准则，量化社会规范，减少奖励偏差；

**🔧 技术方法**

离线强化学习（IQL、CQL、TD3BC）、Preference Transformer、基于ORCA的仿真模拟、TurtleBot4机器人实验；

**📊 数据集**

自建中等难度模拟数据集（约5×10⁵条轨迹，六名行人），无人工偏好标注；

**📈 对比分析**

与手工奖励HR、人工偏好HPR、鲁棒偏好RPR等对比，在500测试案例中SPLC在所有离线RL算法上均取得最高成功率（≈95%）、最低碰撞率，且导航时间相对合理；在实地实验中成功避碰并到达目标；

**⚠️ 局限性**

仅在室内模拟及中等人数人群场景验证，未测试极度拥挤或大规模人群；对sim‑to‑real迁移的鲁棒性需进一步验证；偏好评估准则为固定规则，可能不适用于所有文化或环境。

---

## 380. Population-Based Multi-Objective Training of Discriminators for Semi-Supervised GANs

**arXiv ID:** 2607.01907 | [PDF](https://arxiv.org/pdf/2607.01907v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 381. Open-Weather Robust 3D Detection via Dual-Critic Diffusion Alignment

**arXiv ID:** 2607.01983 | [PDF](https://arxiv.org/pdf/2607.01983v1)

**作者:** Shuyao Li `[一作]` (Nanjing University of Aeronautics and Astronautics), Jingjing Gu `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Dual-Critic Guided Diffusion Alignment (DCDA)，在 LiDAR–4D 雷达融合的 3D 目标检测中实现对降雨、雾霾等恶劣天气下 LiDAR 特征的天气无关恢复。

**💡 创新点**

核心创新在于：①不依赖天气标签或清洁–降雨配对数据；②利用雷达条件扩散过程并引入检测指导与天气对抗两种互补的 Critic 进行特征对齐；③通过路由机制在保持速度的同时保持鲁棒性。

**🔧 技术方法**

技术实现包括雷达条件扩散模型、冻结检测器与判别器的双重对抗监督、两阶段训练策略以及可选的路由决策。

**📊 数据集**

使用公开 K‑Radar 数据集，并基于其 7 种实际天气（Normal、Overcast、Fog、Rain、Sleet、LightSnow、HeavySnow）构建了类型开放、强度开放及类型+强度开放三种开放天气基准。

**📈 对比分析**

在上述三种开放协议下与 RTNH、InterFusion、V2X‑R、L4DR 等主流基线对比，DCDA 在 BEV 与 3D AP 上均显著提升，尤其在未见天气类型上提升 3–7 点 AP，并在合成与真实恶劣天气测试中均保持竞争力。

**⚠️ 局限性**

局限性包括：①扩散迭代带来额外推理延迟，路由能减轻但无法完全消除；②极端强度（如 Snow‑Heavy、Fog‑Heavy）下 LiDAR 结构损失过大，恢复有限；③仅使用合成强度评估，缺乏真实强度标注的验证。

---

## 382. Personalized 4D Whole-Heart Mesh Reconstruction from Cine MRI via Multi-Scale Temporal Modeling and Differentiable Contour Rendering

**arXiv ID:** 2607.01952 | [PDF](https://arxiv.org/pdf/2607.01952v1)

**作者:** Xiaoyue Liu `[一作]` (National University of Singapore), Lei Li `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `4de8e9d8-757b-475f-9627-18a445e50202` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

实现了一个端到端的多视角二维cine MRI到四维全心网格的重建框架，直接从稀疏的二维切片生成时间一致的完整四腔心形三维网格。

**💡 创新点**

创新点包括：1) 基于Beer–Lambert衰减原理的可微渲染损失，用二维切片监督直接指导三维网格生成；2) 多尺度时间建模模块融合全周期全局与局部帧一致性，实现平滑且生理合理的运动轨迹；3) 采用LoRA低秩适配实现跨域自适应；4) 完整四腔网格重建而非仅限左室。

**🔧 技术方法**

使用技术包括：U‑Net+MeshVAE的双域编码器-解码器、图卷积与图注意力网络、可微渲染损失、低秩适配LoRA、Sinusoidal位置编码的多尺度时间网络、MSE、边缘与法线正则化，训练采用Adam优化。

**📊 数据集**

主要数据集为222例后心肌梗死患者的多视角cine MRI（SAX、2CH、3CH、4CH），内部划分155/10/67；外部验证使用BAAI Cardiac Agent公开的CXR‑MULTI‑Fused cine MRI数据。

**📈 对比分析**

与HybridVNet相比，平均MAE降至1.68mm（↓23%）、MSE降至5.06mm²（↓42%），Chamfer 3.41mm、Hausdorff 5.13mm，运动抖动0.77mm/frame³，推理时间<0.1s；在外部数据上亦取得更低MAE/MSE/抖动，证明了较优的几何精度、运动平滑性与泛化能力。

**⚠️ 局限性**

局限性包括：1) 依赖高质量的二维分割监督，分割误差会传递到网格；2) 未对组织物理属性或生物力学约束建模，限制了下游电生理/机械仿真精度；3) 仅在后MI人群验证，缺乏对健康或其他疾病（如心肌病、瓣膜病、心房纤颤）的评估；4) 对细微运动幅度的捕捉仍有提升空间。

---

## 383. Atomic Task Graph: A Unified Framework for Agentic Planning and Execution

**arXiv ID:** 2607.01942 | [PDF](https://arxiv.org/pdf/2607.01942v1)

**作者:** Yue Zhang `[一作]` (South China University of Technology), Zhi Wang `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于图的LLM代理控制框架Atomic Task Graph (ATG)，将任务规划与执行统一为显式有向无环图，支持并行执行与局部错误修复。

**💡 创新点**

创新点在于：①使用递归图编译保持接口不变的任务细化；②显式依赖图实现并行调度与状态追踪；③基于图演化历史的局部子图修复机制；④预执行“思考实验”减少运行时错误。

**🔧 技术方法**

主要技术包括递归图编译、接口保持、拓扑排序执行、节点状态记录、思考实验预验证以及子图局部修复。

**📊 数据集**

在三大长时序交互基准上进行实验：ALFWorld（文本模拟家居任务）、WebShop（电商模拟）、ScienceWorld（文本科学推理）。

**📈 对比分析**

与ReAct、Reflexion、ToT、PoG、CAMEL等传统提示式与显式结构基线以及GPT‑4/3.5等大型模型对比，ATG在所有基准上取得显著提升（如Mistral‑7B上ALFWorld提升≈49点、WebShop≈48点），同时减少执行步数与幻觉动作。

**⚠️ 局限性**

局限性包括依赖LLM的分解能力，错误定位在噪声或长程依赖下困难；实验仅在文本基准，未验证多模态或真实环境；对简单任务会产生额外开销。

---

## 384. TCG-AR: Real-Time Multi-View Augmented Reality for Trading Card Game Streaming

**arXiv ID:** 2607.02090 | [PDF](https://arxiv.org/pdf/2607.02090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 385. DeepGaze3.5-VL: Modeling Scanpaths via Autoregressive Token Prediction

**arXiv ID:** 2607.02083 | [PDF](https://arxiv.org/pdf/2607.02083v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 386. Guided Action Flow: Q-Guided Inference for Flow-Matching Vision-Language-Action Policies

**arXiv ID:** 2607.02092 | [PDF](https://arxiv.org/pdf/2607.02092v1)

**作者:** Liuhaichen Yang `[一作]` (University College London), Zezhi Tang `[通讯]` (University College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种在冻结的流匹配视听语言动作（VLA）策略上进行 Q‑引导流采样（QGF）的推理时框架，通过训练动作块评论家（critic）在不重新训练基准策略的情况下引导采样轨迹。

**💡 创新点**

创新点在于：①将 Q‑引导的梯度应用于冻结的流采样器，实现在推理阶段的局部策略改进；②使用任务描述隐藏状态作为评论家输入，实现跨任务的条件化；③通过评论家集成与不确定性门控，抑制 OOD 情况下的错误梯度。

**🔧 技术方法**

主要技术包括：流匹配（flow‑matching）采样器、逆时序采样规则、动作块评论家（MLP）、梯度裁剪、评论家集成与标准差门控、任务描述隐藏状态提取。

**📊 数据集**

使用基于 任务空间的机器人操作数据集：从官方 LeRobot 评测堆栈收集的实测  环境回放，涵盖 vanilla、-Plus、-PRO 三个子任务族。

**📈 对比分析**

与冻结基准策略比较，QGF 在单一任务上成功率从 68.0% 提升至 82.0%（+14 pp），多任务验证集从 46.0% 提升至 56.0%（+10 pp），锁定测试集提升仅 2.5 pp（从 65.0% 到 67.5%）。

**⚠️ 局限性**

局限性包括：评论家泛化不足导致 OOD 任务上可能退化；评估预算有限，尤其是测试集仅 40 条例；对超参数（β、梯度裁剪阈值、门控参数）高度敏感；在 -PRO 族的零样本性能极差；未在真实机器人上验证。

---

## 387. Evidence-State Rewards for Long-Context Reasoning

**arXiv ID:** 2607.02073 | [PDF](https://arxiv.org/pdf/2607.02073v1)

**作者:** Ya Gao `[一作]` (Aalto University), Pekka Marttinen `[通讯]` (Aalto University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用强化学习训练LLM在长上下文推理中通过添加、链接、删除证据以及最终回答的动作编辑证据内存来完成问答。

**💡 创新点**

创新点在于提出答复条件证据状态价值，并为每个动作（add、link、drop、answer）提供局部奖励，强调证据状态的动态迁移而非一次性提取。

**🔧 技术方法**

使用GRPO强化学习框架、冻结的verifier评估证据价值、编辑式证据内存接口以及action‑local奖励机制。

**📊 数据集**

在LongBench v2、LongReason、RULER等长上下文基准以及HotpotQA、2WikiMultiHopQA、MuSiQue、LongRLVR等训练数据集上进行实验。

**📈 对比分析**

相较于基线（原始模型、SFT、结果导向RL、证据识别奖励等），Maven在各模型和基准上提升约3–4分，并显著提高证据充分率、降低干扰器保留率，表现优异。

**⚠️ 局限性**

局限性包括：需金标准答案来计算verifier价值；证据必须可用文本段落表示，难以直接适用于多模态或开放式生成任务；对人类评估证据质量的依赖较高。

---

## 388. A Memory Efficient Unified Algorithm for Online Learning of Linear Dynamical Systems

**arXiv ID:** 2607.02050 | [PDF](https://arxiv.org/pdf/2607.02050v1)

**作者:** Yuval Ran-Milo `[一作]` (Tel Aviv University), Elad Hazan `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种在线预测算法，能够在未知线性动力系统（LDS）上实现次线性遗憾，并且其可学习参数仅取决于系统的“不稳定复杂度”k，而与隐藏状态维度无关。

**💡 创新点**

创新点包括：
- 设计了一种统一的预测类，融合了谱滤波、有限记忆输入滤波（FIR）和自回归（AR）三种方法，专门针对具有少量不稳定/非半单一模式的系统；
- 引入并证明了不稳定复杂度k是该预测问题的自然复杂度度量，并给出了匹配的下界；
- 在在线学习框架下给出子线性遗憾保证，并通过Vovk–Azoury–Warmuth算法实现参数高效学习。

**🔧 技术方法**

主要技术包括：
- 线性系统理论（Jordan分解、Cayley–Hamilton定理）来拆解系统的不同模式；
- 谱滤波与FIR的近似理论来处理稳定与快速衰减模式；
- 自回归模型对不稳定/非半单一模式的精确表示；
- 在线学习中的Vovk–Azoury–Warmuth（VAW）在线回归算法，用于在特征空间中学习参数；
- 证明技巧：组合近似误差与VAW的对数遗憾，得到总体次线性遗憾。

**📊 数据集**

实验使用合成的高维LDS（维度503，k=3），通过随机输入控制该系统，并在相同参数预算下比较不同预测器。

**📈 对比分析**

与仅使用有限记忆、仅使用AR或仅使用谱滤波的基线方法比较。统一预测器在同等参数（16个标量参数）下，在最后10000步的归一化均方误差上达到约1e-8至1e-8，远优于其它方法（最优约1e-5，最差约1e-1），显示了统一方法在捕捉不稳定模式时的显著优势。

**⚠️ 局限性**

局限性：
- 上界与下界仅匹配到多项式对数因子，实际最优对数因子及常数尚未确定；
- 研究聚焦于预测，未直接给出闭环控制策略；
- 只在合成数据上验证，缺乏真实世界控制任务的实证；
- 目前仅对单输入单输出或已可观测的LDS提供理论，其他更一般的控制场景仍待探索。

---

## 389. Cross-Platform Control for Autonomous Surface Vehicles via Adaptive Reinforcement Learning

**arXiv ID:** 2607.02037 | [PDF](https://arxiv.org/pdf/2607.02037v1)

**作者:** Ruiheng Jiang `[一作]` (ETH Zurich), Aswin Ramachandran `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

该论文提出一种自适应强化学习策略，能在不知道平台动力学的情况下，实现跨平台零步部署的航迹跟踪。

**💡 创新点**

创新点在于使用教师-学生结构通过交互历史学习潜在动力学表示，使单一策略即可适配不同船舶动态。

**🔧 技术方法**

技术包括三自由度分析动力学模型、随机域训练、PPO、教师-学生编码器、GRU适配器和基于历史的条件化。

**📊 数据集**

数据集包括两台真实ASV平台（A、B）和五台仿真平台（Roboat1-3）以及随机生成的动力学参数。

**📈 对比分析**

与基准MPC、PPO特定、PPO通用、教师、递归和学生平均潜在变量等方法比较，学生在真实环境中位置误差从9.5cm降至4.0cm，提升58%，在仿真中误差由1.49cm降至0.84cm。

**⚠️ 局限性**

局限在于仅覆盖低速非计划状态的线性阻尼3-DoF模型，无法处理高速计划、垂直运动及复杂海况等情形。

---

## 390. Embracing Intra-Class Heterogeneity for Semi-Supervised Medical Image Segmentation: From Diversity to Precision

**arXiv ID:** 2607.02051 | [PDF](https://arxiv.org/pdf/2607.02051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 391. Using embeddings to predict spoken word duration and pitch in Mandarin monosyllabic words

**arXiv ID:** 2607.02002 | [PDF](https://arxiv.org/pdf/2607.02002v1)

**作者:** Xiaoyun Jin `[一作]` (University of Tuebingen), R. Harald Baayen `[通讯]` (University of Tuebingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究利用上下文嵌入预测普通话单音节词的发音时长和F0轮廓，并探究其在词型和词素层面的可预测性。

**💡 创新点**

首次证明词义在上下文嵌入中与词素层面时长存在关联；同时结合时长和形状预测实时F0轮廓，并与置换基线比较，展示了语义与语音细节高度交织。

**🔧 技术方法**

使用GPT‑2生成上下文嵌入，线性映射求解SW=C以预测时长/轮廓；采用10折交叉验证、Pearson相关、邻近邻正确率、动态时间规整（DTW）等评估方法；并用LDA检验嵌入对说话人、语速、停顿等非语义因素的可预测性。

**📊 数据集**

台湾普通话自发语料库（约7,476个单音节词，已手工校正语素边界），提取词素时长、词时长及100点归一化F0轮廓，共6,118个有效标注。

**📈 对比分析**

与全局置换基线和类型内置换基线进行对比。词素时长预测相关系数为0.366，词时长为0.399，F0轮廓正确率为0.170，均显著优于置换基线（p<0.001）。实时F0轮廓预测的DTW距离平均显著小于置换预测（t(92)≈-3.25，p≈0.0016）。

**⚠️ 局限性**

局限性包括：仅限单音节词且剔除极短或异常词；模型未捕获说话人差异、语速、情感等非语义因素；采用线性映射而非更复杂的端到端模型，可能限制预测精度。

---

## 392. Evaluating Vision-Language Models as a Zero-Shot Learning Alternative to You Only Look Once and Optical Character Recognition for Nigerian License Plate Recognition

**arXiv ID:** 2607.02025 | [PDF](https://arxiv.org/pdf/2607.02025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 393. Hidden Forgetting in Continual Multimodal Learning: When Accuracy Survives but Grounding Fails

**arXiv ID:** 2607.02020 | [PDF](https://arxiv.org/pdf/2607.02020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 394. Prompt Coverage Adequacy

**arXiv ID:** 2607.02057 | [PDF](https://arxiv.org/pdf/2607.02057v1)

**作者:** Florian Tambon `[一作]` (University of Luxembourg), Mike Papadakis `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Prompt Coverage Adequacy，评估测试用例覆盖自然语言提示中的需求；

**💡 创新点**

创新点在于利用LLM的注意力与熵值变化，将测试用例与提示中的需求进行语义匹配，形成新的基于提示的覆盖准则；

**🔧 技术方法**

采用LLM注意力加权（Spotlighting）、交叉熵估计、Beta GLMM等技术进行覆盖计算与统计分析；

**📊 数据集**

使用HumanEval+和LiveCodeBench两大代码生成基准，结合Qwen2.5和Gemma3两种开源LLM；

**📈 对比分析**

与传统代码覆盖率比较，Prompt Coverage与代码覆盖正相关，且基于Prompt Coverage生成的测试用例在缺陷检测上比基于代码覆盖提高约30%+；

**⚠️ 局限性**

局限包括依赖LLM的注意力机制与熵估计，可能不适用于所有模型；实验仅覆盖Python任务，提示长度、结构复杂度的影响尚未充分验证；

---

## 395. Mitigating Package Hallucinations in Large Language Models via Model Editing

**arXiv ID:** 2607.02052 | [PDF](https://arxiv.org/pdf/2607.02052v1)

**作者:** Shuhan Liu `[一作]` (Zhejiang University), Xin Xia `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 BOUND 的轻量级局部模型编辑框架，用于减少大语言模型在软件工程任务中产生的虚假包名（package hallucination）。

**💡 创新点**

创新点在于将包名幻觉视为“包有效性边界”编辑问题，设计风险感知模块定位策略，随后在定位的模块中注入 LoRA 并使用边界感知的三重损失（有效包 NLL、幻觉包 Unlikelihood、局部 KL）来精细调节模型行为。

**🔧 技术方法**

采用 LoRA 参数插桩、梯度敏感度定位、边界风险评分、LogSumExp 归一化、unlikelihood 损失与 KL 散度约束等技术，整体保持原模型权重冻结，仅更新极少量低秩参数。

**📊 数据集**

使用 Spracklen 等人提供的 Python 任务提示集（高风险提示子集），并以 PyPI 元数据判定包合法性；评估任务包括包推荐、代码生成与包安装命令生成，共涉及 3 个开源 LLM（DeepSeekCoder、Qwen3、Llama‑3.1）。

**📈 对比分析**

与全量微调、Self‑Refinement、ROME、MEMIT、DINM 等基线对比，BOUND 在编辑提示集上 Sample‑HR 降低约 75.9%、Package‑HR 降低 79.9%，在未见提示集上分别降低 63.7% 和 65.4%，且 Valid‑Rate 上升 6.4%。在跨任务（代码生成、包安装命令）中也分别取得 12.8% 与 34.0% 的 Package‑HR 减少，显著优于其他方法。

**⚠️ 局限性**

局限性包括：仅验证了 Python/PyPI 生态，无法直接推广到其他语言或包管理系统；依赖于提示集的高风险筛选，可能对不同模型产生不同效果；局部编辑在极少数极端任务中仍可能失效；实验仅在三款开源 LLM 上进行，商业 LLM 的适用性尚未确认。

---

## 396. LongEgoRefer: A Benchmark for Long-Form Egocentric Video Referring Expression Comprehension

**arXiv ID:** 2607.02096 | [PDF](https://arxiv.org/pdf/2607.02096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 397. Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving

**arXiv ID:** 2607.02043 | [PDF](https://arxiv.org/pdf/2607.02043v1)

**作者:** Shrikara Arun `[一作]` (Microsoft), Victor Rühle `[通讯]` (Microsoft)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Kairos，一种主动预填转移调度框架，能将分离式LLM服务中预填阶段的请求转移到解码节点上执行，以减少预填队列等待和跨节点KV缓存传输带来的延迟；

**💡 创新点**

核心创新在于：①利用实时TBT安全检查和TTFT估计，为每个请求动态生成chunked预填调度；②构建可预测混合批处理的Step‑Latency模型，确保在不违背TBT SLO的前提下最大化解码节点闲置算力；③将预填转移与传统分离式架构无缝集成，显著降低尾部延迟；

**🔧 技术方法**

技术细节包括：预填/解码分离架构、chunked prefill、TBT安全分析模型、基于线性回归的步骤延迟预测、实时节点状态收集与分析、vLLM 0.18.1 上的插件实现和一次性步骤延迟分析；

**📊 数据集**

使用公司X的生产级LLM请求追踪（归一化到DeepSeek‑V2‑Lite 16B MoE模型），并构造了五种工作负载（LPLD、LPHD、HPLD、HPHD、Bursty）进行实验；

**📈 对比分析**

与传统的PD disaggregation和TaiChi进行对比。实验表明，Kairos在Burst工作负载下将P95 TTFT从4.5 s压缩至2.6 s（提升≈81%），SLO达成率从88%提升至99%（提升≈79%），吞吐量保持相当或更高，且TBT始终不超过70 ms，未出现SLO违例；

**⚠️ 局限性**

主要限制包括：①解析模型误差可能导致TBT预测偏差，②调度器每100 ms更新节点状态，可能在极高RPS时产生延迟；③一次只能在每个解码节点接受单个转移请求；④未考虑多轮请求的前缀缓存影响；⑤集中式调度器在大规模部署时可能成为瓶颈；

---

## 398. Fast and Accurate Anomaly Detection in Time Series

**arXiv ID:** 2607.02046 | [PDF](https://arxiv.org/pdf/2607.02046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 399. EduArt: An educational-level benchmark for evaluating art history knowledge in large language models

**arXiv ID:** 2607.02007 | [PDF](https://arxiv.org/pdf/2607.02007v1)

**作者:** Gianmarco Spinaci `[一作]` (University of Bologna), Giovanni Colavizza `[通讯]` (University of Bologna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了名为 EduArt 的艺术史教育级多模态基准，包含 871 题目（意大利中学练习和美国 AP 艺术史考试），并在 12 个大型语言模型（跨 6 家供应商）上进行两种评测条件（仅答复与答复+动机阐述）的实验。

**💡 创新点**

创新点在于：① 采用人类作者的真实教育题目而非合成问答；② 结合多种题型（MCQ、True/False、定位、填空、错误识别等），提供细粒度能力划分；③ 将 Classical Test Theory 与逻辑回归相结合，对项目难度、辨别度、格式、语言、图像影响进行量化分析。

**🔧 技术方法**

技术手段包括：多模态 LLM 调用（OpenAI、Google Gemini、Anthropic、Qwen、Mistral、Meta via Bedrock），系统/用户 prompt 设计，基于 Gemini/Claude 的信息抽取脚本，使用 o200k_base 编码计数，SPACY 进行词法统计；评测指标采用精确匹配、F1、模糊字符串匹配等。

**📊 数据集**

数据集：871 条题目，涵盖 7 种题型，包含 261 张不同分辨率的图像；来自意大利 Zanichelli 数字练习平台（668 题）与美国 College Board AP Art History（203 题），提供意大利和英语两种语言的混合。

**📈 对比分析**

比较方法：在默认与动机两条件下对 12 个模型进行成对比较；使用宏平均准确率、MCQ 绝对匹配准确率、项目级 CTT 统计、逻辑回归预测等；性能显示宏平均分在 29.1%–82.8% 之间；六个模型在 MCQ 达到 90%+，但在开放式/定位/错误识别等格式下降 50%+，说明单一格式评测会高估能力。

**⚠️ 局限性**

局限性：① 题目版权受限，无法公开重分；② 图像低分辨率导致视觉任务可能被低估；③ 题型与语言分布不均，图像与非图像子集未完全匹配；④ 只覆盖教育级内容，未探究更高级学术级任务；⑤ 评测仅基于黑盒 API，无法深入分析模型内部机制。

---

## 400. Algebraic Model Counting for Global Analysis of Optimal Decision Trees

**arXiv ID:** 2607.02069 | [PDF](https://arxiv.org/pdf/2607.02069v1)

**作者:** Hiroki Arimura `[一作]` `[通讯]`, Hiroki Arimura

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于代数模型计数的框架 ADTC，对决策树空间进行全局评估，利用张量语义实现多指标聚合与样本筛选，构建模型行为张量；

**💡 创新点**

创新点在于将代数模型计数推广至决策树，设计动态规划算法 EMT 使复杂度从双指数降至 O^*(n^O(Δ))，并引入张量卷积实现多度量联合分析；

**🔧 技术方法**

使用代数模型计数 (AMC)、张量代数 (tensor semiring)、动态规划与卷积运算；

**📊 数据集**

采用 UCI Adult 数据集（32,129 条样本）以及其平衡版本进行实验；

**📈 对比分析**

通过运行时、扫描单元数和非零张量条目等指标与传统枚举、统计计数对比，显示 ADTC 在数据规模、树深度和结构约束下保持对数级增长，能够在几分钟内完成全局分析；

**⚠️ 局限性**

局限包括对树深度的指数复杂度、张量维度和内存占用受限，尚未扩展至其它可解释模型，且理论复杂度与 #P 等计数类的关系仍待深入研究。

---

## 401. kNNGuard: Turning LLM Hidden Activations into a Training-Free Configurable Guardrail

**arXiv ID:** 2607.02072 | [PDF](https://arxiv.org/pdf/2607.02072v1)

**作者:** Mahmoud Abdelfattah `[一作]` (Lancaster University), Peter Garraghan `[通讯]` (Lancaster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的LLM守护栏kNNGuard，通过对冻结LLM隐藏层激活进行多层kNN分类，结合句子嵌入实现安全与主题检测。

**💡 创新点**

创新点在于利用LLM内部激活空间的结构性信息而非仅依赖表面嵌入，使用Fisher分离度加权多层kNN，并自适应融合激活与嵌入分数，且无需微调即可快速适配新域。

**🔧 技术方法**

技术主要包括多层隐藏激活提取、Fisher分离度权重、kNN风险估计、适应性融合、系统提示条件化以及离线银行构建。

**📊 数据集**

实验使用16个公开数据集，覆盖编码指令/输出、医学、安全、越狱和注入等六大领域的安全与主题检测任务。

**📈 对比分析**

与基准Fine‑tuned守护栏（Llama Nemotron系列）和Embedding‑kNN比较，kNNGuard FE在六个领域平均F1达到87.4%，FPR 12.9%，推理时延仅45.9 ms，比主流守护栏快2.7倍且比Embedding‑kNN慢约4 ms。

**⚠️ 局限性**

局限性包括对参考银行质量依赖强、在没有系统提示时精度下降、对极其细粒度语义区分仍有误判、以及在新型对抗样本面前可能仍出现逃逸风险。

---

## 402. HaloGuard 1.0: An Open Weights Constitutional Classifier for Multilingual AI Safety

**arXiv ID:** 2607.02079 | [PDF](https://arxiv.org/pdf/2607.02079v1)

**作者:** Navaneeth Sangameswaran `[一作]` (Astroware AI), Ashmiya Lenin `[通讯]` (Astroware AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HaloGuard 1.0，一种基于自然语言宪法的输入安全分类器，作为LLM部署的第一层防护；

**💡 创新点**

创新点在于将宪法视为生成规范，构建细粒度46条宪法、2,940子类别的安全体系，采用配对反事实、边界硬负样本、多语言平衡和持续红队硬化等技术；

**🔧 技术方法**

技术手段包括：Qwen3.5基准生成器、生成式分类（自回归跨熵）、风格锚定生成、配对反事实生成、攻击模式叠加、确定性变换、边界与基线FP控制、滑动窗口长文本检测、阈值校准、连续红队；

**📊 数据集**

使用合成语料：约1,259,451条训练/评测/测试样本，覆盖所有子类别，包含有害、边界安全、共享安全样本；多语种覆盖46种语言，采用英语基线生成后逐语种翻译；此外引入公开基准（OpenAI Moderation、Aegis、ToxiC、WildGuard、PolyGuardPrompts等）用于评测；

**📈 对比分析**

与8款公开guard模型（LlamaGuard4、WildGuard、ShieldGemma、NemoGuard、PolyGuard-Qwen、Qwen3Guard-Gen等）在7个基准上对比，HaloGuard 0.8B平均F1 90.9，4B平均F1 92.1，均超越所有基线，FP率4.3/4.7、FN率9.5/7.7，显示在FP/FN前沿上的显著提升；

**⚠️ 局限性**

局限性包括：1）评测主要集中在英文，跨语言性能仍有提升空间；2）对极端或新型攻击仍可能有误判；3）合成数据仍可能与真实流量存在差距；4）持续红队和阈值调优需人工维护；5）部署时需结合后端工具和人类审核以实现完整安全栈。

---

## 403. Scalable and Distributed Silhouette Approximation

**arXiv ID:** 2607.01993 | [PDF](https://arxiv.org/pdf/2607.01993v1)

**作者:** Ilie Sarpe `[一作]` (KTH Royal Institute of Technology), Fabio Vandin `[通讯]` (University of Padova)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一系列基于采样的随机化算法，用于在任意度量空间下近似计算聚类的局部轮廓值和全局轮廓系数，并给出了严格的误差概率保证；同时实现了可扩展的 MapReduce / MPC 版本，能够在常数轮次内完成计算，并在大规模数据集上实现子线性本地内存。

**💡 创新点**

创新点主要包括：
• 第一次给出局部轮廓估计的理论保证，误差可由用户设定的 ε 控制；
• 采用概率比例采样 (PPS) 取样策略，显著降低了所需的距离计算量至 O(nkε⁻² log(nk/δ))，远低于传统 O(n²)；
• 设计了三种全局轮廓估计器（直接平均、基于局部估计、两阶段采样），并证明它们在误差和样本量上均优于现有基线；
• 构建了常数轮次、子线性本地内存的 MapReduce/MPC 实现，实现了分布式计算的可扩展性。

**🔧 技术方法**

主要技术包括：
• 概率比例取样 (PPS) 与 Poisson 采样相结合，用于估计每个簇内的平均距离；
• 误差分析使用 Chernoff 与 Hoeffding 绑定，给出绝对误差和置信度；
• MapReduce/MPC 并行模型的多轮实现，利用分区、广播与聚合实现常数轮次；
• 对比实验采用多种基线（精确 O(n²)、简化轮廓、均匀采样等）。

**📊 数据集**

实验数据集覆盖了不同规模与维度：
• 小规模（n≈10⁴–10⁵，维度≤30）：Breast、Wine、RNA‑seq、Credit、Metro、Shuttle；
• 中等规模（n≈10⁵–10⁶，维度≤10）：PowerHouse、IoT、Gowalla 等；
• 大规模（n≈10⁶–10⁷，维度≤10）：Gowalla、PowerHouse、IoT 等。 
所有数据集均在 Euclidean、Cosine、Manhattan、Canberra 等常用距离下聚类，k 取值 2–20。

**📈 对比分析**

与基线比较：
• 在大多数中等/大规模数据集上，PPS 采样方法（全局估计）在保持相同或更少的距离计算量时，平均误差低于 0.02，且速度提升 5–10 倍；
• 局部估计器（p-pps）在绝大多数簇中将最大误差降至 0.1 以下，明显优于均匀采样；
• 简化轮廓与现有 heuristic 在大多数场景下误差高达 0.5，且不提供误差保证；
• 在分布式实现中，常数轮次、子线性内存的 MapReduce/MPC 版本在 8–16 线程下实现 8–15 倍的加速。

**⚠️ 局限性**

局限性与未来工作：
• 仍需在每个簇内对所有点与采样点做距离计算，若 k 接近 n 或簇内点分布极度不均匀，取样误差可能增大；
• 误差控制参数 ε、δ 需要人工设定，过小 ε 可能导致样本量巨大；
• 仅适用于满足三角不等式的度量距离，对非度量或高维稀疏数据的适用性需进一步验证；
• 对极大维度（e.g., >10⁴）时，单点距离计算本身成本高，尚未结合更高效的距离估算技术。

---

## 404. SPLIT: Cross-Lingual Empathy and Cultural Grounding in English and Ukrainian LLM Responses

**arXiv ID:** 2607.02049 | [PDF](https://arxiv.org/pdf/2607.02049v1)

**作者:** Anna Chorna `[一作]` `[通讯]`, Anna Chorna

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了SPLIT 500题多语言情绪支持基准，评估了三大LLM在英语和乌克兰语危机情境下的共情、自然度和文化语境适应性。

**💡 创新点**

提出跨语言情绪支持评估框架及LLM‑as‑a‑jury自动评估方法，发现不同LLM在乌克兰语上的性能下降并揭示共情与文化适应评分差异。

**🔧 技术方法**

使用了三种架构各异的LLM（DeepSeek‑V3、LLaMA‑3.3‑70B‑Instruct、Gemini‑2.5‑Flash）以及GPT‑4o、Mistral Large、Claude 4.5 Sonnet作为评审模型。

**📊 数据集**

SPLIT基准共500条情绪支持查询，涵盖Stress、Panic、Loneliness、Internal Displacement、Tension五类，原始查询通过GPT‑4o生成并翻译为乌克兰语，经人工校对。

**📈 对比分析**

通过三评审模型的加权平均给出1–5分评分，并与10%人工评估进行Pearson相关性对比，结果显示DeepSeek在两语种均表现最稳定，Gemini和LLaMA在乌克兰语上显著退化。

**⚠️ 局限性**

仅有单一人工评注，评审模型可能存在自偏好；基准仅覆盖5类危机场景，缺乏更广泛文化和语言多样性验证。

---

## 405. File-Level Copying Is an Implicit Dependency in Open Source

**arXiv ID:** 2607.02059 | [PDF](https://arxiv.org/pdf/2607.02059v1)

**作者:** Runzhi He `[一作]` (Peking University), Minghui Zhou `[通讯]` (Peking University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文系统研究了开放源代码生态中文件级复制行为，构建了690,500个复制事件的数据集，并分析了复制形式、开发者动机及其对供应链可见性（来源、维护、安全、合规）的影响。

**💡 创新点**

创新点在于：①提出了13类可观测复制形式的分类法；②结合复制事件的上下文信息，构建了开发者动机代码本；③量化了复制导致的四维可见性缺口，并揭示了安全风险集中在依赖托管形式、合规风险集中在源代码复制两种互补形式。

**🔧 技术方法**

技术上采用了世界代码（World of Code）的大规模仓库快照，利用 Bloom 过滤器进行高速复制种子检测，构建事件级别的复制链；使用大语言模型 (LLM) 进行复制形式与动机的自动标注；通过时间戳与项目度量实现来源归属评分；对安全风险采用 CVE 修复提交链，对合规风险采用 SPDX 兼容矩阵和人工审核。

**📊 数据集**

主要数据集为：World of Code 2019-2023 的仓库快照（超过 10 亿 blob），GitHub 事件日志（GHArchive）用于补充提交元数据，CVE 关联的漏洞修复提交表以及 Repology 的跨生态包映射。

**📈 对比分析**

与现有仅基于包管理器的安全合规扫描相比，本文的复制检测能发现 88% 的 CVE 风险聚集在未在清单中标注的托管复制，且修复延迟平均 640 天；在合规层面，原始工具只检测到 0.1% 的冲突，而本文通过文件级扫描揭示 39 起真实的许可冲突，准确率 93.6%。

**⚠️ 局限性**

局限性包括：①复制事件的覆盖受限于 WoC 的公开仓库和镜像；②动机标签仅覆盖 3,912 个具备足够文本线索的事件，未能给出整体比例；③来源归属采用规则加权模型，可能在复杂重写或镜像场景下产生误判；④安全和合规风险的统计基于已知 CVE 修复提交，忽略了仅 advisory 或二进制补丁的漏洞。

---

## 406. PWM-ArtGen: Part World Model for Articulated Object Generation

**arXiv ID:** 2607.02045 | [PDF](https://arxiv.org/pdf/2607.02045v1)

**作者:** Wentao Zheng `[一作]` (Sun Yat-sen University), Ancong Wu `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用单张图片生成可关节3D物体，自动推断关节类型、轴线、运动范围等运动学参数，并在同一框架内生成对应的视觉动态；

**💡 创新点**

提出统一的Part World Model（PWM‑ArtGen），通过独立的动作与图像扩散步长联合学习视觉动态与运动学参数的联合分布，并通过视觉动态正则化（VDR）和无标注数据共训练显著提升运动学识别；

**🔧 技术方法**

采用Diffusion Transformer+AdaLN，结合SAM+GPT‑4o的部件分割、DINOv2视觉编码、VDR正则化和动作-图像共训练的扩散策略；

**📊 数据集**

使用PartNet‑Mobility（合成渲染）及其增强的PartNet‑Mobility‑Reality（PM‑R）19.7k对照图像数据，和ACD真实物体数据进行训练与评估；

**📈 对比分析**

在ACD和PartNet‑Mobility测试集上与URDFormer、NAP、SINGAPO、Articulate‑Anything等基线对比，PWM‑ArtGen在d_gIoU、d_cDist、d_CD等指标上均明显优于对照组；在视觉动态方面PSNR≈24.3、SSIM≈0.91、LPIPS≈0.093，推理时间0.7 s，参数仅0.38 B；

**⚠️ 局限性**

局限性包括：对部件分割器（Part Mask Generator）的依赖导致合成数据上部分指标略逊；仅支持旋转/滑动关节，未覆盖更复杂运动；模型仍需大规模GPU训练，推理成本相对较高。

---

## 407. Mirror Illusion Art

**arXiv ID:** 2607.02015 | [PDF](https://arxiv.org/pdf/2607.02015v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 408. Spatio-Temporal and Clinical Conditioning for Fine-Grained Radiology Report Retrieval

**arXiv ID:** 2607.02024 | [PDF](https://arxiv.org/pdf/2607.02024v1)

**作者:** P. Sloan `[一作]` (University of Bristol), M. Mirmehdi `[通讯]` (University of Bristol)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于多模态、时空注意力的检索框架（STAR3），用于生成按解剖区域、时间进展和临床指征条件化的放射学报告。

**💡 创新点**

创新点包括：① 在解剖区域级别进行句子检索，增强解剖定位与报告内容的一致性；② 引入区域级时序建模，捕捉前后检查的病变变化；③ 使用跨模态注意力将临床指征融入视觉特征；④ 采用解剖丢弃机制筛选报告相关区域，降低冗余；⑤ 通过半监督多模态对比学习和硬负样本约束构建高质量检索空间。

**🔧 技术方法**

技术手段包括：RadDINO+Faster R‑CNN 对象检测；多头注意力进行时序融合与跨模态注意力；解剖丢弃与异常预测网络；RadBERT 进行临床文本编码；自监督对比损失（全局、局部、硬负样本、ITM）以及多任务训练。

**📊 数据集**

数据集：MIMIC‑CXR（放射学报告与图像）以及其子集 Chest ImaGenome（解剖区域标注与句子标签），使用官方训练/验证/测试拆分。

**📈 对比分析**

与现有检索式方法（BioViL、RadIR、CLIP‑XRad、X‑REM、Teaser、AHIVE、DuCo‑Net）对比。STAR3 在 Recall@k 上均优于所有基线，Recall@100 达 0.869；BLEU‑1/2/3/4 分别为 0.437/0.281/0.172/0.128；CHEXBERT 0.713、RadGraph F1 0.569、RadCLiQ 0.209，均为最佳或近最佳。

**⚠️ 局限性**

局限性：检索式生成受限于语料库，难以表达罕见或新颖发现；仅处理前景胸片，未结合侧位或多视角；对极少见病变的检索效果不佳；在极端场景下可能产生重复或冗余句子。

---

## 409. A Stereo Visual SLAM System Using Object-Level Motion Estimation and Geometric Filtering Based on Cross Disparity

**arXiv ID:** 2607.02005 | [PDF](https://arxiv.org/pdf/2607.02005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 410. WBMM: Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution

**arXiv ID:** 2607.02097 | [PDF](https://arxiv.org/pdf/2607.02097v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 411. ESC: Emotional Self-Correction for Reliable Vision-Language Models

**arXiv ID:** 2607.02089 | [PDF](https://arxiv.org/pdf/2607.02089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 412. SUNTA: Hierarchical Video Prediction with Surprise-based Chunking

**arXiv ID:** 2607.02087 | [PDF](https://arxiv.org/pdf/2607.02087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 413. Comprehensive Robustness Analysis of LiDAR-based 3D Object Detection in Autonomous Driving

**arXiv ID:** 2607.02074 | [PDF](https://arxiv.org/pdf/2607.02074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. Hierarchical Anti-Aesthetics: Protecting Facial Privacy against Customized Diffusion Models

**arXiv ID:** 2607.02038 | [PDF](https://arxiv.org/pdf/2607.02038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 415. ComplexMimic: Human-Scene Interaction Imitation in Complex 3D Environments

**arXiv ID:** 2607.02034 | [PDF](https://arxiv.org/pdf/2607.02034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 416. Multimodal Fusion for Fine-Grained Classification of Breast Fibroadenoma and Phyllodes Tumors

**arXiv ID:** 2607.02091 | [PDF](https://arxiv.org/pdf/2607.02091v1)

**作者:** Chuxi Nan `[一作]` (Wuxi Vocational College of Science and Technology), Jiawei Li `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个基于多模态深度学习的乳腺纤维腺瘤与叶状瘤细分类别判别框架。

**💡 创新点**

提出临床引导的适应性调制和交叉模态Transformer融合，首次将B‑mode超声图像、诊断文本和结构化临床信息三模态联合。

**🔧 技术方法**

采用DenseNet图像编码、CLIP风格文本编码、轻量化临床编码，配合FiLM调制、门控机制和Transformer多头自注意力进行融合。

**📊 数据集**

使用由910例病理确诊的FAPT‑M多模态数据集，包括超声图像、诊断文字和临床属性。

**📈 对比分析**

在患者级五折交叉验证中取得77.64%准确率、73.38% F1和89.74% AUC，显著优于CNN、Transformer及视觉‑语言基线。

**⚠️ 局限性**

局限于单中心回顾性样本、类别不平衡、未包含分割或其他影像模态，需进一步外部验证。

---

## 417. InduceKV: Fixed-Footprint Continual Adaptation of Multimodal LLMs via Inducing KV Memories

**arXiv ID:** 2607.02010 | [PDF](https://arxiv.org/pdf/2607.02010v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 418. Beyond the Performance Illusion: Structure-Aware Stratified Partitioning and Curriculum Distributionally Robust Optimization for Spatially Correlated Domains

**arXiv ID:** 2607.02055 | [PDF](https://arxiv.org/pdf/2607.02055v1)

**作者:** Prathamesh Patil `[一作]` (QpiAI India Pvt Ltd), Aswanth Krishnan `[通讯]` (QpiAI India Pvt Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一套面向空间相关数据的统一评估与训练框架，解决了传统随机划分导致的时空泄漏与隐性分层问题。

**💡 创新点**

创新点在于① Structure‑Aware Stratified Partitioning (SASP)，通过自监督特征构造语义簇并在保证类别平衡的前提下实现结构无缝分离；② Curriculum Distributionally Robust Optimization (CDRO)，在训练阶段逐步提升难度组的权重，缓解了严格划分导致的优化不稳定。

**🔧 技术方法**

技术包括自监督视觉表示（如 DINOv2）用于语义聚类；图论和约束优化实现语义无缝分层；基于难度重采样的学习率和采样策略实现 CDRO；YOLO 目标检测模型作为基线。

**📊 数据集**

实验涵盖三类空间相关基准：全球麦穗检测（GWHD）、无人机航拍目标检测（VisDrone‑DET）和医学显微血细胞图像（BCCD）。

**📈 对比分析**

与传统随机划分+经验风险最小化（ERM）比较，SASP+ERM 已显著降低验证误差与测试误差差距；再加上 CDRO 后，验证与测试性能进一步接近，且在各数据集上提升了 3–8% 的 mAP。

**⚠️ 局限性**

局限性包括：① SASP 需要自监督模型的预训练与特征提取，可能受限于可用算力；② 对极端稀有子群的识别仍不保证完全；③ CDRO 的超参数（如学习率衰减、阶段划分）对最终效果敏感，需进一步自动化调优。

---

## 419. OpenSafeIntent: Evaluating Intent-Calibrated Safe Completion Across Dual-Use Prompt Sets

**arXiv ID:** 2607.02047 | [PDF](https://arxiv.org/pdf/2607.02047v1)

**作者:** Rheeya Uppaal `[一作]` (University of Wisconsin-Madison), Junjie Hu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个以意图转移为核心的安全完成评估基准，包含匹配的 benign、dual‑use 和 malicious 提示三元组；

**💡 创新点**

创新点在于通过保持任务不变仅改变意图来衡量模型的安全一致性，避免单个提示评估导致的误判；

**🔧 技术方法**

采用多模态 LLM 自动评分器（Helpfulness 与 Harmfulness 评估），并定义了 MeanSafety、TripletSafety、MeanUtility、WorstCaseUtility 等度量；

**📊 数据集**

使用 OpenSafeIntent 数据集，包含 805 条提示、115 种任务类型，并对双重使用提示生成了四个同义改写；

**📈 对比分析**

对多种模型（GPT‑5.4、Llama‑3.1‑8B、Claude‑Sonnet‑4.6 等）进行比较，发现平均安全度与 TripletSafety 可能不一致，最佳模型的 MeanUtility 仅为 0.56，表明安全‑有用性仍有提升空间；

**⚠️ 局限性**

局限性包括对自动评分器的过度依赖、双重使用提示生成与评判仍需人工校准，以及缺乏对更细粒度安全行为的细致分析。

---

## 420. X-Splat: Gaussian Splatting for 3D CBCT Generation from Single Panoramic Radiograph

**arXiv ID:** 2607.02099 | [PDF](https://arxiv.org/pdf/2607.02099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 421. Fine-Grained Bounds for Courcelle's Theorem

**arXiv ID:** 2607.02033 | [PDF](https://arxiv.org/pdf/2607.02033v1)

**作者:** Daniel Lokshtanov `[一作]` (University of California), Meirav Zehavi `[通讯]` (Ben-Gurion University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在树宽受限图上针对有限元一阶逻辑(FO)与一阶逻辑+集合变量的多层量化语句(MSO)提出了通用的判定算法，并给出了与树宽和量化层数相关的指数阶时间复杂度上界；

**💡 创新点**

创新点在于提出了“签名”(signature)的概念，用于刻画图的量化层次结构，从而将判定问题拆分为先计算签名，再根据签名高效判定语句是否成立；同时给出了与量化层数、树宽、量化变量数量相关的指数层数上界，并证明了该上界是紧的（在EXP假设下下界同样呈指数层级）；

**🔧 技术方法**

核心技术包括：树分解的nice形式、动态规划结合分离与join节点的签名合成、指数层级的“签名”集合枚举以及利用指数塔函数的组合与界定；

**📊 数据集**

无实验数据集，全部以理论上限证明和计数论证为主；

**📈 对比分析**

方法上通过对比上界与下界来说明其最佳性；实验或实现方面未给出具体时间测量，理论复杂度与现有Courcelle‑style算法相比更紧凑；

**⚠️ 局限性**

主要局限在于对量化层数d和量化变量数k_i、s_i的指数层级上界使得在实际中难以处理高层级语句；实现复杂度高，常数隐藏较大，且对树宽t仅在对数级别提升的情况下适用。

---

## 422. HandsOnWorld: Unconstrained Egocentric Video Generation with Camera-Disentangled Hand Control

**arXiv ID:** 2607.02075 | [PDF](https://arxiv.org/pdf/2607.02075v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 423. Fourier Neural Operators for Rayleigh-Bénard Convection

**arXiv ID:** 2607.02088 | [PDF](https://arxiv.org/pdf/2607.02088v1)

**作者:** Chelsea Maria John `[一作]` (Juelich Supercomputing Centre), Daniel Ruprecht `[通讯]` (Hamburg University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并训练了一种轻量级 Fourier 神经算子（FNO），通过学习状态增量实现二维 Rayleigh–Bénard 对流的时间步进模拟，显著提升了预测精度和推理速度。

**💡 创新点**

创新点包括：①使用增量（ΔU）目标而非完整状态预测，使模型更像一阶时间积分器；②采用多层 1D 卷积缩放层（lifting/projection）而非线性层；③结合余弦学习率调度器；④构建了仅 314k 参数、1.26 MB 的紧凑模型。

**🔧 技术方法**

技术手段：Fourier Neural Operator（FNO）架构、Dedalus 生成的数值模拟数据、相对 L₂ 损失、Adam 优化器、余弦学习率调度、CUDA GPU（NVIDIA A100）训练。

**📊 数据集**

数据集：Dedalus 产生的 256×64 网格、Ra=10⁷、Pr=1 的 2000 个时间步样本（10 条随机初始种子），从 T_init=100 到 T_sim=200，每步 Δt=10⁻³；训练集 80%/验证集 20%。

**📈 对比分析**

与标准 FNO 基线及 Straat 等人 3D FNO 对比：增量目标的精度比完整状态低两阶（e.g. u 错误 2.4e-05 vs 1.6e-03），模型大小仅 314k 参数，推理时间 7 ms（batch=1）vs 5 ms；在不同时间步长和自回归推理下误差略升高；在更细网格（512×128）上仍保持相似误差，显示网格不变性，但准确性受训练分辨率限制。

**⚠️ 局限性**

局限性：①自回归推理时误差逐步积累，长时间步长或多步预测精度下降；②对更大时间步长（>10⁻²）和更高 Rayleigh 数时性能受限；③虽然网格可插值，但超细网格不一定提高精度，需更高分辨率训练数据。

---

## 424. Ask the Right Comparison:Bias-Aware Bayesian Active Top-$k$ Ranking with LLM Judges

**arXiv ID:** 2607.02104 | [PDF](https://arxiv.org/pdf/2607.02104v1)

**作者:** Jian Xu `[一作]` (RIKEN iTHEMS), Qibin Zhao `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预算有限的情况下，研究如何利用大型语言模型（LLM）作为判定者，通过对比判定高质量答案的 top‑k 项目，克服 LLM 判定的噪声和系统性偏见。

**💡 创新点**

创新点：① 引入带有可学习偏差系数（长度、位置）的贝叶斯 Bradley–Terry 模型，并通过缩放先验自适应识别是否存在偏差；② 设计了针对 top‑k 成员资格的主动采样规则，将比较成本聚焦在能改变 top‑k 边界的对比上。

**🔧 技术方法**

技术手段：贝叶斯推断 + 拉普拉斯近似、缩放（或豪斯号）先验、主动学习/实验设计、信息量评估、逻辑回归、马尔科夫决策。

**📊 数据集**

数据集：构造了 30 个答案（每个答案 12 条真假陈述，含可控冗长标签）作为基准；对 16 种真实 LLM 判定者（开放源、闭源）进行全对比；在 LLMBar 对抗性子集上进行外部验证。

**📈 对比分析**

与基线比较：朴素计数、全局不确定性（D‑optimal）和随机采样；在预算 60–200 的情况下，偏见感知+top‑k 主动采样的 recall 约提升 0.15–0.20（对低成本、偏见显著的 LLM 最高提升 0.3+），在大预算下可与无偏见判定器相当；在高预算下仍保持 20‑倍的成本优势。

**⚠️ 局限性**

局限性：① 仅在偏差属于呈现属性（长度、格式）而非真实质量时有效；若长度真相关（如 SummEval、Nectar）则误差会增大；② 需要先验或锚点判定是否开启偏差校正；③ 对极弱判定者（信息量低）几乎无益；④ 仅建模单一偏差维度，无法处理多重交互偏差；⑤ 需要高质量的对比标注来估计偏差，可能增加人工成本。

---

## 425. Evolutionary Wave Function Collapse

**arXiv ID:** 2607.02082 | [PDF](https://arxiv.org/pdf/2607.02082v1)

**作者:** Dipika Rajesh `[一作]` (University of California Santa Cruz), Julian Togelius `[通讯]` (New York University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过进化小型WFC输入图像，利用Wave Function Collapse作为基因型到表现型的映射，生成更大尺寸的迷宫和地牢地图，并对生成结果进行评估。

**💡 创新点**

创新点在于将WFC的输入图像视为可进化的基因型，将进化搜索与WFC结合，从而在保持WFC局部约束学习优势的同时，实现对全局生成质量的优化。

**🔧 技术方法**

采用进化算法（锦标赛选择、变异、精英保留）与Wave Function Collapse相结合，并使用领域特定的适应度函数进行评估。

**📊 数据集**

使用4×4随机小图作为WFC输入基因型，生成8×8迷宫和16×16地牢地图；实验依据PCG Benchmark提供的适应度函数进行评价。

**📈 对比分析**

与单纯随机搜索做对比，在迷宫域进化方法快速收敛并显著提升连通性和路径长度；在地牢域虽取得一定进步，但仍难以满足全局玩法约束，整体性能优于随机搜索但有限。

**⚠️ 局限性**

主要局限在于WFC仅学习局部邻接关系，难以直接实现全局约束（如唯一玩家、钥匙、门等），且WFC的随机性导致评估噪声，限制了进化搜索的稳定性与效果。

---

## 426. UnderOneFacade: Worldwide Facade Semantic Segmentation Benchmark Dataset

**arXiv ID:** 2607.02018 | [PDF](https://arxiv.org/pdf/2607.02018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 427. CoFL-S: Spatially Queryable Sector Flow Fields for Local Language-Conditioned Navigation

**arXiv ID:** 2607.02222 | [PDF](https://arxiv.org/pdf/2607.02222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 428. What Types of Human-AI Teams Exist?

**arXiv ID:** 2607.02198 | [PDF](https://arxiv.org/pdf/2607.02198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 429. SA-HGNN: Sample-Adaptive Hyperbolic Graph Neural Network for EEG-Based Depression Recognition

**arXiv ID:** 2607.02063 | [PDF](https://arxiv.org/pdf/2607.02063v1)

**作者:** Yang Li `[一作]` (Huazhong University of Science and Technology), Lianbo Guo `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种面向抑郁症识别的 Sample‑Adaptive Hyperbolic Graph Neural Network (SA‑HGNN)，通过自适应构建脑电图网络并在双曲空间进行卷积与注意池化来捕获层次化功能连通性；

**💡 创新点**

创新点在于①融合物理先验与个体特征的自适应图构建模块，②使用双曲空间卷积克服欧氏空间的表达瓶颈，③加入注意池化去除冗余噪声并保持核心子图；

**🔧 技术方法**

采用 1D‑CNN 进行时序特征提取，Poincaré 球模型的双曲图卷积，基于多头自注意力的注意池化，以及交叉熵、稀疏正则和均匀损失等多重损失；

**📊 数据集**

使用公开 HUSM 数据集，包含 34 名抑郁症患者和 30 名健康对照的 19 通道 256Hz 记录，分为休息态与任务态两种实验模式；

**📈 对比分析**

与七种基准图神经网络（如 SDGCN、GCBNet、DGCNN、LGGNet、RGNN、GraphSleepNet、DCGNN）进行对比，SA‑HGNN 在休息态获得 95.24% 识别精度与 95.77% F1，任务态 94.26% 识别精度与 94.69% F1，均显著优于对手；

**⚠️ 局限性**

局限在于仅验证于单一 HUSM 数据集，模型复杂度高，需进一步评估跨数据集泛化能力与计算成本；

---

## 430. PACE: A Proxy for Agentic Capability Evaluation

**arXiv ID:** 2607.02032 | [PDF](https://arxiv.org/pdf/2607.02032v1)

**作者:** Yueqi Song `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**通讯引用:** 22297 | [OpenAlex ID](https://openalex.org/A5068811427)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了代理式评估框架 Proxy for Agentic Capability Evaluation，用少量非代理性评估实例预测大语言模型在代理性基准上的性能。

**💡 创新点**

创新点：结合SVD杠杆得分与Spearman相关度的双重筛选机制，并采用噪声感知的bootstrap回归，能够在仅占代理评估成本1%以下的情况下实现高精度预测。

**🔧 技术方法**

技术：线性最小二乘回归、逻辑回归、SVD秩分解、SVD杠杆得分、Spearman相关度、bootstrap采样等。

**📊 数据集**

数据集：代理性基准 GAIA、SWE‑Bench（Verified、Multimodal）、SWT‑Bench；非代理基准包含若干涵盖指令遵循、规划、工具调用、推理、长文本等能力的20+基准；模型覆盖20+大语言模型。

**📈 对比分析**

比较方法：与随机子采样目标基准和完整代理评估成本进行对比。实验显示，使用100个代理实例时，MAE 3.8%、Spearman 0.81、pairwise accuracy 85%，成本仅为随机子采样的1/100。

**⚠️ 局限性**

局限：代理预测依赖于校准模型的代表性，若新模型超出校准分布，误差可能增大；需要定期更新校准集。

---

## 431. Benchmarking Quantum Software Testing with Scalable Quantum Programs

**arXiv ID:** 2607.02029 | [PDF](https://arxiv.org/pdf/2607.02029v1)

**作者:** Yuechen Li `[一作]` (Beihang University), Kai-Yuan Cai `[通讯]` (Beihang University)

**通讯引用:** 6629 | [OpenAlex ID](https://openalex.org/A5103991023)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了可扩展量子程序的量子软件测试基准（Benchmark）体系，提供了对程序进行重构、规范化接口、编写规格和单元测试等一系列支持，可用于系统化评估QST方法。

**💡 创新点**

创新点在于：①提出基于真实来源的可扩展量子程序收集与筛选流程；②设计测试友好的重构与规范化步骤；③创建针对功能、输出、规模的QST导向分类与度量；④通过该基准进行两种现有QST方法的可行性实验，揭示后台差异对结果的影响。

**🔧 技术方法**

主要技术包括：Python与Qiskit 2.3.0的高级API；程序重构与接口标准化；基于Sphinx的文档与公式规范化；单元测试框架；使用Qiskit后端（理想模拟器、Algiers、Brooklyn等）进行噪声与无噪声实验；统计分析工具（Mann-Whitney U、Vargha-Delaney效应值）。

**📊 数据集**

数据集为自构建的Benchmark “Quantum Software Testing Benchmark”（简称QSTB），共包含 100+ 可重构的可扩展量子程序，来源于公开仓库、已发表研究及现有基准；对每个程序提供规格、单元测试与七个改写变体。

**📈 对比分析**

实验通过将两种QST方法（MSTC、DOSS）分别在理想与噪声后端上执行，比较执行时间与缺陷检测率；结果显示混合态测试比纯态测试更快；不同后端（尤其是Algiers与Brooklyn）导致执行时间与故障检测率差异显著，验证了后端模型对QST实验的重要性。

**⚠️ 局限性**

局限性包括：仅覆盖Qiskit无测量的量子程序；实验仅使用经典模拟器，未验证真实硬件；缺陷检测采用人工变异而非真实错误；基准不包含工业级大型量子程序；且程序重构仍可能引入语义差异。

---

## 432. SAMoR: Motion Modelling for Articulated Objects of Any Skeleton and Topology

**arXiv ID:** 2607.02148 | [PDF](https://arxiv.org/pdf/2607.02148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. AbsoluteDegradation: A Physics-Inspired Synthetic Film-Degradation Pipeline and Archival Film Restoration Benchmark

**arXiv ID:** 2607.02131 | [PDF](https://arxiv.org/pdf/2607.02131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 434. Speaker head orientation estimation with a single microphone array using phase spectrogram features

**arXiv ID:** 2607.02129 | [PDF](https://arxiv.org/pdf/2607.02129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 435. Overview of Risk Assessment and Management for Intelligent Systems under the AI Act and Beyond

**arXiv ID:** 2607.02197 | [PDF](https://arxiv.org/pdf/2607.02197v1)

**作者:** Javier Irigoyen `[一作]` (BiometricsAI, Universidad Autónoma de Madrid), Alvaro Ortigosa `[通讯]` (GHIA, Universidad Autónoma de Madrid)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了AI风险评估与管理方法，结合欧盟AI法、各国监管框架、ISO/IEC标准与NIST AI RMF，梳理现有风险分类、评估流程与实践缺口；

**💡 创新点**

创新点在于将多源法规、标准与学术框架系统化整合，并指出其方法学差距与未来研究方向，尤其针对多模态LLM与生成式AI的风险评估；

**🔧 技术方法**

主要采用文献综述、框架对比、风险分类与理论模型（如AI Risk Atlas、AIRA、KAIRI等）进行分析；

**📊 数据集**

未使用实验数据集，内容主要基于法规文本、标准规范与已有研究文献；

**📈 对比分析**

通过比较不同法规与标准的风险评估要求，未给出具体量化性能指标，仅展示各框架在风险识别、评估与治理上的差异与适用场景；

**⚠️ 局限性**

局限性包括缺乏经验验证与实证案例，聚焦理论与框架整合，未对多模态LLM等新兴技术的具体风险进行深入评估。

---

## 436. Generalized Extended Codes with Applications in Entanglement-Assisted Qubit and Qutrit Codes

**arXiv ID:** 2607.02170 | [PDF](https://arxiv.org/pdf/2607.02170v1)

**作者:** Yang Li `[一作]` (Nanyang Technological University), Zhonghua Sun `[通讯]` (Hefei University of Technology)

**通讯引用:** 432 | [OpenAlex ID](https://openalex.org/A5047549483)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种通用的扩展码$(oldsymbol u,a)$模型，并利用其在Hermitian构造下生成EAQECCs，系统地得到281条新或改进的qubit和qutrit量子码参数。

**💡 创新点**

创新点在于提出$(oldsymbol u,a)$通用扩展码，证明几乎所有$[n+1,k+1]_{q^2}$线性码的Hermitian双对偶距离大于1时都单群等价于该模型，并给出Hermitian棱锥维数和距离提升的必要与充分条件，从而实现多参数同时优化。

**🔧 技术方法**

主要技术包括Hermitian构造、CSS和辛构造、最大子码判据、有限几何覆盖判据以及代数/几何解析。

**📊 数据集**

使用的数据集为Grassl在线表中列出的最佳已知线性码及最新EAQECC记录。

**📈 对比分析**

通过与Grassl表和最新记录对比，本文实现了267条qubit和14条qutrit改进参数，其中244条明显优于之前最佳，37条为全新参数，总计281条；在长度、维数、最小距离以及所需纠缠对数等指标上表现突出。

**⚠️ 局限性**

局限性在于仅针对Hermitian内积展开，欧几里得和辛对称情形处理较弱，且对双对偶距离为1的特殊情况处理有限。

---

## 437. Predictive Conformal Slip Monitoring: An Empirical Evaluation of Rolling Split Conformal Prediction for Pre-Incident Traction Loss Detection

**arXiv ID:** 2607.02124 | [PDF](https://arxiv.org/pdf/2607.02124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 438. Influence of Radial Basis Activation Functions on Intelligent Controller for Robotic Manipulators

**arXiv ID:** 2607.02167 | [PDF](https://arxiv.org/pdf/2607.02167v1)

**作者:** Kimmo Paldanius `[一作]` (University of Turku), Wallace Moreira Bessa `[通讯]` (University of Turku)

**通讯引用:** 877 | [OpenAlex ID](https://openalex.org/A5078549673)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在机器人机械臂的轨迹跟踪控制中，结合基于反馈线性化的模型控制与RBF神经网络的在线扰动估计，研究了不同RBF核函数对闭环性能的影响

**💡 创新点**

将激活函数视为结构设计参数，系统评估了高斯、拉普拉斯和逆多二次核的对稳态精度与瞬态行为的权衡，并揭示了全局支持核能显著提升平滑轨迹的跟踪误差

**🔧 技术方法**

基于反馈线性化的非线性控制、RBF神经网络逼近、Lyapunov投影自适应学习、实验验证

**📊 数据集**

Quanser QArm 机器人臂的数字双胞胎仿真平台，测试三种参考轨迹（正弦、方波、三角波）

**📈 对比分析**

通过比较RMS误差、IAE、RMS PWM、以及对方波的超调/调节时间等指标，结果显示IMQ核相较基线可降低约52%的跟踪误差，控制能耗仅增加约3%，但在阶跃输入下会产生更大的超调和更长的调节时间

**⚠️ 局限性**

仅在单自由度系统上验证，未考虑多自由度耦合及更复杂工况，核函数参数调节缺乏系统化方法

---

## 439. Unlocking Speech-Text Compositional Powers: Instruction-Following Speech Language Models without Instruction Tuning

**arXiv ID:** 2607.02214 | [PDF](https://arxiv.org/pdf/2607.02214v1)

**作者:** Congrui Du `[一作]` (University of California, Santa Barbara), Shiyu Chang `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 9639 | [OpenAlex ID](https://openalex.org/A5112248869)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造一种只需单轮语音预训练（30k小时）的语音语言模型 SpeechCombine，利用文本LLM的指令跟随能力（Δθ_inst）与语音适配方向（Δθ_speech）在参数空间相加，直接生成可执行语音输入/输出且能跟随多种指令的模型。

**💡 创新点**

创新点在于：① 将指令跟随能力与语音知识通过权重差向量直接融合，完全不需要迭代的指令微调；② 只需一次预训练即可获得多模态指令跟随；③ 通过模型合并的方式在语音域内实现“深层”组合（如情感理解、强调检测/生成），此前的方法在此类任务上几乎为零。

**🔧 技术方法**

技术方法包括：连续预训练（next-token 预测）+ LoRA 微调；模型合并（Δθ_inst + Δθ_speech）；离散语音分词（仅编码韵律信息）；对齐输入/输出模板与格式强制；长思考（long‑thinking）机制。

**📊 数据集**

使用约30k小时的公开语音数据（如 Common Voice、LibriSpeech、VoxPopuli 等），配合自生成的语音说明（GPT‑OSS 生成的包含情感、语速、强调等属性的自然语言描述）。

**📈 对比分析**

在两大实验组中评估：
• Group A（对比常规语音微调方法）：SpeechCombine 在文本导向任务（QA、推理）上与 ASR+文本LLM 接近，且在语音理解/生成任务上大幅优于连续预训练和连续预训练+SFT。
• Group B（SOTA 语音LLM）：SpeechCombine 在大部分基准（情感识别、强调检测、情感与强调生成）上取得与顶尖模型相当或更好，且使用的数据量仅为其 1% 以内。

**⚠️ 局限性**

局限性：① 输出格式不稳定，需要格式强制；② 仅编码韵律，缺乏音色/口音信息；③ 依赖外部 ASR，增加推理时延与转写误差。

---

## 440. Criticality-Based Guard Rail Validation for AI Agent Decisions in Autonomous Telecom Networks

**arXiv ID:** 2607.02210 | [PDF](https://arxiv.org/pdf/2607.02210v1)

**作者:** Ravi Kant Sharma `[一作]` `[通讯]` (Ericsson), Ravi Kant Sharma (Ericsson)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出 Guard Rail Validation（GRV）框架，插入 AI/ML 推理输出与网络动作之间的运行时验证层，防止错误决策直接影响网络。

**💡 创新点**

创新点在于：① 多维度关键性评估与临时模式检测；② 按风险分级的验证机制（低-执行、媒-边界检查、高-独立验证、危-多方共识）；③ 跨代理冲突检测与基于关键性加权的优先级解析；④ 兼容 EU AI Act 的合规日志与人机交互。

**🔧 技术方法**

使用的技术包括：权重关键性评分算法、滑动窗口临时模式检测、独立验证器（模型/规则/状态校验）、M‑of‑N 共识、O‑RAN 接口（A1/E2/O1）、Near‑RT RIC 与 Non‑RT RIC 的部署模型，以及安全日志框架。

**📊 数据集**

论文未使用公开数据集，主要通过理论描述、示例场景（如能源节省 rApp）和威胁映射来阐述框架。

**📈 对比分析**

未给出实验评估，作者仅通过威胁覆盖表和法规对应表说明效果，缺乏实际性能或误报率数据。

**⚠️ 局限性**

局限包括：缺乏实测评估与性能指标、阈值调优导致误报/误判风险、只检测直接冲突（未覆盖拓扑间间接冲突）、对验证器可用性的依赖、以及对临时模式检测与多方共识机制的实现复杂度。

---

## 441. An Optimisation Framework for the Well-Conditioned Training of Physics-Informed Neural Networks

**arXiv ID:** 2607.02194 | [PDF](https://arxiv.org/pdf/2607.02194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 442. Faster Cache-Efficient Pattern Matching for Deterministic Wheeler Pangenome Graphs

**arXiv ID:** 2607.02113 | [PDF](https://arxiv.org/pdf/2607.02113v1)

**作者:** Riccardo Maso `[一作]` (Ca' Foscari University of Venice), Carlo Tosoni `[通讯]` (Ca' Foscari University of Venice)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种针对确定性Wheeler自动机（WDFA）的缓存友好的图后缀数组索引，实现了高效的模式匹配查询。

**💡 创新点**

创新点在于将传统的BWT后向搜索改为结合二分搜索与线性扫描的图后缀数组方法，显著减少I/O操作并提升缓存效率。

**🔧 技术方法**

核心技术包括Wheeler DFA的路径一致性、前缀/后缀推断、伪树的Heavy‑Light分解、最小完美哈希、以及对单元路径的压缩存储。

**📊 数据集**

在实际生物信息学数据上使用了来自芬兰人群dbSNP数据库构建的21号和14号染色体的pan‑genome图进行实验。

**📈 对比分析**

与传统的前向搜索（Forward Search）比较，图后缀数组在内存占用上最多提升15倍，但在查询速度上可快500倍，单字符处理时间低于3纳秒。

**⚠️ 局限性**

主要局限在于空间开销相对较大（O(|D|) RAM词），对非常大规模图或高通量需求时可能受限，且对特殊字母集（如四字母DNA）做了专门优化。

---

## 443. ART for Diffusion Sampling: Continuous-Time Control and Actor-Critic Learning

**arXiv ID:** 2607.02137 | [PDF](https://arxiv.org/pdf/2607.02137v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 444. Choreographing the Way of Water: A Computational Framework for Aquatic Robotic Art

**arXiv ID:** 2607.02174 | [PDF](https://arxiv.org/pdf/2607.02174v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 445. Efficient PEFT Methods with Adaptive Checkpointing for Vision Models and VLMs on Resource Constrained Consumer-GPUs

**arXiv ID:** 2607.02158 | [PDF](https://arxiv.org/pdf/2607.02158v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 446. Online Resource Allocation with Continuous Random Consumption: Regret under Degeneracy

**arXiv ID:** 2607.02196 | [PDF](https://arxiv.org/pdf/2607.02196v1)

**作者:** Jiawei Zhang `[一作]` (New York University), Jiawei Zhang `[通讯]` (New York University)

**通讯引用:** 193039 | [OpenAlex ID](https://openalex.org/A5100408669)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在奖励和消耗量均可连续分布的在线资源分配问题，并提出了一种样本路径边际（SPM）策略；

**💡 创新点**

创新点在于揭示连续随机消耗可使得活跃边界质量指数改变，给出了以此指数为核心的上界与匹配下界，并且不依赖传统的流体非退化假设；

**🔧 技术方法**

主要技术包括RAMS原理、活跃质量产品稳定性、凸分析、端点接触（endpoint‑contact）结构、VC子类集中性及大数定律等；

**📊 数据集**

论文未使用实测数据集，全部基于理论模型与分布假设；

**📈 对比分析**

与传统基于流体双重价格的算法比较，SPM在指数=1时实现O((log T)^2)加性遗憾，指数>1时实现O(T^{1/2-1/(2α)}·polylog T)并给出匹配下界，优于以往需非退化假设的结果；

**⚠️ 局限性**

局限性包括仅处理一维消耗向量、需要已知分布与端点指数、以及对多资源随机消耗向量的推广仍待研究。

---

## 447. UA-ChatDev: Uncertainty-Aware Multi-Agent Collaboration for Reliable Software Development

**arXiv ID:** 2607.02186 | [PDF](https://arxiv.org/pdf/2607.02186v1)

**作者:** Temitayo Olamilekan Ogunsusi `[一作]` (Prairie View A&M University), Xishuang Dong `[通讯]` (Prairie View A&M University)

**通讯引用:** 1254 | [OpenAlex ID](https://openalex.org/A5006284276)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了UA-ChatDev，一种在多智能体软件开发流程中引入不确定性量化的框架；

**💡 创新点**

通过在子任务交互中评估输出置信度并在置信度低于阶段阈值时触发检索验证，显著减少幻觉传播；

**🔧 技术方法**

利用基于token级对数概率的轻量级不确定性估计、阶段性阈值校准、检索增强验证与多轮对话协作；

**📊 数据集**

在SRDD（Software Requirement Description Dataset）上进行评估，该数据集包含1200个自然语言软件任务；

**📈 对比分析**

与GPT-Engineer、MetaGPT、ChatDev等SOTA单/多智能体系统对比，UA-ChatDev在完整性、可执行性、一致性与综合质量上均实现了明显提升（例如Qwen2.5模型下整体质量从0.395提升至0.649）；

**⚠️ 局限性**

增加的不确定性评估和检索触发导致生成时间和token消耗上升，且对阈值设定敏感，未来需要进一步优化检索策略与阶段性预算。

---

## 448. RadiomicNet: A Hybrid Radiomics-Guided Lightweight Architecture for Interpretable Medical Image Segmentation

**arXiv ID:** 2607.02185 | [PDF](https://arxiv.org/pdf/2607.02185v1)

**作者:** Mohammad Amanour Rahman `[一作]` `[通讯]` (Ahsanullah University of Science and Technology (AUST)), Mohammad Amanour Rahman (Ahsanullah University of Science and Technology (AUST))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种两流混合结构RadiomicNet，将手工提取的放射组学特征直接融入轻量化MobileNetV2编码-解码器，实现医学图像分割。

**💡 创新点**

创新点包括Radiomics Attention Gate（RAG）利用GLCM和LBP特征调制跳跃连接注意力，并提出Radiomics Consistency Loss将纹理复杂度与预测不确定性对齐，实现先验解释性。

**🔧 技术方法**

采用的技术包括MobileNetV2轻量化骨干、双流特征融合、RAG注意力门、MSE一致性损失、BCE+Dice+boundary加权损失以及梯度特征重要性分析。

**📊 数据集**

使用的公开数据集为乳腺超声图像集BUSI和肠镜息肉分割集Kvasir-SEG。

**📈 对比分析**

与U-Net、UNet++、TransUNet、U-KAN等基线对比，RadiomicNet在两数据集上分别实现DSC 0.763（比U-KAN提升1.2%）和0.854（提升1.8%），参数量仅3.27M，显著低于基线。

**⚠️ 局限性**

主要局限在于固定的13个特征选择、仅处理2D切片、未验证3D CT/MRI等多模态扩展。

---

## 449. Predicting Early Stages Of Alzheimer's Disease And Identifying Key Biomarkers Using Deep Artificial Neural Network And Ensemble Of Machine Learning Methodologies

**arXiv ID:** 2607.02142 | [PDF](https://arxiv.org/pdf/2607.02142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 450. Patient-Specific Articulated Digital Twins from a Single Full-Body CT Scan

**arXiv ID:** 2607.02156 | [PDF](https://arxiv.org/pdf/2607.02156v1)

**作者:** Han Zhang `[一作]` (Johns Hopkins University), Mathias Unberath `[通讯]` (Johns Hopkins University)

**通讯引用:** 5223 | [OpenAlex ID](https://openalex.org/A5087095414)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

基于单张全身CT扫描构建患者特定可运动的数字双生模型，实现对新体位的重定位。

**💡 创新点**

将SMPL参数化人体模型与CT骨骼绑定，保留骨骼刚性并实现姿态重定向，首次实现CT静态扫描的姿态可控性。

**🔧 技术方法**

使用TotalSegmentator分割、Marching Cubes与Taubin平滑生成表面，SMPL模型拟合、骨骼关键点初始化，线性混合皮肤化改为刚体变换，DeepDRR生成DRR。

**📊 数据集**

使用NMDID数据集中三例全身CT，并采用AMASS运动序列作为目标姿势。

**📈 对比分析**

评估Chamfer距离、骨骼包围率、SSIM/PSNR等指标；在采集姿态下SSIM≈0.872、PSNR≈18.5dB；在未见姿势下骨骼包围率保持≈94.4%，说明模型保持一致。

**⚠️ 局限性**

受样本量少、软组织非刚性未建模、SMPL对极端姿势的局部失配、缺乏真实姿势下的真实影像验证等限制。

---

## 451. Tight Lower Bounds for the Multi-Secretary Problem via Bellman Certificates

**arXiv ID:** 2607.02150 | [PDF](https://arxiv.org/pdf/2607.02150v1)

**作者:** Jiawei Zhang `[一作]` (New York University), Jiawei Zhang `[通讯]` (New York University)

**通讯引用:** 193039 | [OpenAlex ID](https://openalex.org/A5100408669)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明了多秘书问题中有支持间隙的混合均匀分布在临界容量下的增量遗憾为Ω((log T)²)，与已知的O((log T)²)上界匹配。

**💡 创新点**

提出了Bellman证书框架，将遗憾递归转化为可行解构造，从而直接给出全局下界；首次展示支持间隙导致的log²阶遗憾机制。

**🔧 技术方法**

利用动态规划的Bellman递推、凸残差分析、极限马尔可夫过程以及概率大偏差与有限差分估计等技术。

**📊 数据集**

无实际数据集，整个工作为理论证明。

**📈 对比分析**

通过对比已知的上界结果，证明所给下界与上界在该情形下完全一致，表明额外的对数因子是必要的。

**⚠️ 局限性**

局限于单资源多秘书模型及支持间隙为常数宽度的两均匀混合分布，未覆盖多资源或非均匀分布情形。

---

## 452. DetailAnywhere: Fashion Detail Generation via Cross-Modal Feature Alignment Distillation

**arXiv ID:** 2607.02220 | [PDF](https://arxiv.org/pdf/2607.02220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 453. Actuator Reality Shaping for Zero-Shot Sim-to-Real Robot Learning

**arXiv ID:** 2607.02205 | [PDF](https://arxiv.org/pdf/2607.02205v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 454. Taxing Artificial Intelligence

**arXiv ID:** 2607.02144 | [PDF](https://arxiv.org/pdf/2607.02144v1)

**作者:** Juliette Faivre `[一作]` (Carnegie Mellon University), Sarah H. Cen `[通讯]` (Carnegie Mellon University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文系统梳理并评估了人工智能（AI）可能产生的外部性，探讨税收工具如何在AI治理中发挥作用，并提出针对不同外部性设计合适税基、税种与税率的框架。

**💡 创新点**

创新点包括：① 将税收视为多功能工具，不仅限于Pigouvian矫正，还兼顾资源再分配与监管资金筹集；② 依据AI外部性的性质将税收目标细分为纠正、再分配和监管三类，并针对每类制定对应的税基与税种；③ 提出了将AI税收与现有税收体系（如消费税、企业所得税、排放税等）相结合的可操作方案，强调了税收在技术监管缺口时的“快速部署”优势。

**🔧 技术方法**

主要采用的技术是宏观经济与税收理论框架，结合政策分析方法；并对各种税收工具（消费税、企业所得税、超额利润税、排放税、劳动力税等）的理论效能与适用场景进行了定性比较。

**📊 数据集**

未使用传统意义上的数据集；文章基于文献综述、案例分析（如数据中心资源外部性、创意劳动冲击等）以及已有的税收实例（如美国各州数据中心电费税、欧盟对AI平台的监管提案）进行讨论。

**📈 对比分析**

作者通过理论模型和案例对比不同税收工具的优缺点，例如：消费税能直接对AI使用产生定价，排放税能精准捕捉资源消耗，企业所得税可快速产生财政收入；但也指出了各工具在测量、合规、逃避与激励失真方面的潜在问题，未给出数值性能指标，强调需根据具体外部性选择最合适的税种。

**⚠️ 局限性**

局限性包括：① 对AI外部性的量化和边际成本估算困难；② 税基与纳税主体界定模糊，易导致规避或激励错误；③ 可能产生的竞争力流失与区域逃税风险；④ 政治捕获可能削弱税收的纠正与再分配功能；⑤ 在缺乏监管基础设施时，税收单独不足以解决技术性外部性。

---

## 455. A$^{2}$utoLPBench: An Auto-Generated, Agent-Friendly LP Benchmark via Inverse-KKT Construction

**arXiv ID:** 2607.02141 | [PDF](https://arxiv.org/pdf/2607.02141v1)

**作者:** Shuo Ren `[一作]` (Chinese University of Hong Kong), Tsung-Yi Ho `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 6879 | [OpenAlex ID](https://openalex.org/A5062800747)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 A^2utoLPBench：一个可生成线性规划（LP）问题并支持 LLM 驱动代理自测的完整基准体系，包含逆 KKT 生成器、自然语言渲染、基线 Solver‑Critic 交互循环及 Docker 运行环境。

**💡 创新点**

创新点在于①用逆 KKT 直接构造可验证的 LP 对例，消除人工标注与求解器依赖；②将基准从静态数据集迁移为可生成的“生成器”，实现无限供应、可调难度与泄露抗性；③提供“Propose–Audit–Refine”三步代理循环，自动纠错并提升 LLM 解决率；④将所有工具与接口打包为 Docker 镜像，方便 LLM 代理即插即用。

**🔧 技术方法**

核心技术包括逆 KKT 逆向构造、自然语言 LLM 绘制、Solver‑Critic 多轮自检、工具调用（function calling/Anthropic tool）与 Docker 化封装。

**📊 数据集**

使用生成器构造的 256 条参考样本（按 (n,m) 分层），并可根据 seed 与维度随时生成任意规模问题；此外在实验中对 NL4OPT、MAMO（易/难）等现有静态基准进行迁移测试。

**📈 对比分析**

通过在八个尺寸层（(2,3)→(40,40)）上对 DeepSeek‑V4 进行基准扫描，验证难度随尺寸提升平滑变化；在外部基准上比较 vanilla 与 A^2utoLPBench 的解率，发现对 MAMO 复杂集提升可达 15.1pp；在六个求解器的跨模型 Critic 评测中，Kimi‑K2.5 与 Claude‑Sonnet‑4.6 等可提升 10–15pp。

**⚠️ 局限性**

局限性包括：①批量生成与解算时需 LLM 代价；②Solver‑Critic 循环的收益依赖代理能否有效利用 Critic 反馈，某些模型（如 OpenAI‑GPT5.4）在 Refine 阶段表现不佳；③生成的 LP 仅覆盖标准不等式/非负约束的最大化形式；④对极大规模实例的数值稳定性尚未彻底验证。

---

## 456. ContextNest: Verifiable Context Governance for Autonomous AI Agent

**arXiv ID:** 2607.02116 | [PDF](https://arxiv.org/pdf/2607.02116v1)

**作者:** Misha Sulpovar `[一作]` (PromptOwl, LLC), Gabe Goodhart `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ContextNest 规范与实现，构建一个可版本化、可验证、可追溯的 AI 代理知识库治理层，并与检索增强生成（RAG）协同工作。

**💡 创新点**

创新点在于把上下文治理单独抽象为治理层，定义 Typed Markdown、集成式选择器语法、SHA‑256 链式版本历史、检查点快照、源节点与审计轨迹，形成完整、可复现的知识治理框架。

**🔧 技术方法**

技术栈包括：Markdown + YAML 前置元数据、集合代数选择器、URI 可寻址与锚点、SHA‑256 hash 链链、Git‑style 内容地址化、Model Context Protocol（MCP）、OpenTelemetry 监控、AGPL‑3.0 开源实现。

**📊 数据集**

使用合成的 1,060 条 Markdown 文档（10/30 查询实验集合）和公开的 1,060 文档实验集，亦包含实际的 10‑查询基准与 30‑查询滞后版本攻击集。

**📈 对比分析**

通过与 BM25 稀疏检索、Dense+HNSW 近似检索对比；治理选择器在滞后版本攻击中以 97% pass 率、约 1/3 token 成本优于 BM25；检索确定性实验表明 Dense+HNSW 在 80% 查询上非确定，平均 Jaccard 0.611；实验验证治理层显著提升可信度且 token 成本更低。

**⚠️ 局限性**

局限性包括：仅实现文档级授权与审计，未提供行级/段落级作者归属；语义检索未内置，只通过委托接口；哈希链仅检测后期篡改而非预防；未处理代理身份与跨组织身份验证；缺少多用户协作、实时编辑与冲突解决机制。

---

## 457. The Eticas AI Risk Taxonomy: Open Infrastructure for Operationalizing AI Audits

**arXiv ID:** 2607.02201 | [PDF](https://arxiv.org/pdf/2607.02201v1)

**作者:** Gemma Galdon Clavell `[一作]` (Eticas Ai), Usman Gohar `[通讯]` (Eticas Ai)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Eticas AI 风险分类体系 v2.0.0，并在此基础上构建了四层运作方法，首次展示了从风险概念到可执行测试、测量、严重度判定、评级以及合规报告的完整链路；在 GPT‑4‑0314 上用 DecodingTrust Privacy Scenario 2 对 PII 泄露风险进行端到端验证。

**💡 创新点**

创新点包括：① 将风险与其表现机制分离，形成可供任何方法论接入的契约表面；② 开放式语义基础设施（SKOS/JSON‑LD、稳定 URI、CC BY 4.0）和“开放核心”模式；③ 10‑20‑76 的三层分级结构和 18 个外部框架的三层映射；④ 公开的测量‑评级链条和严重度阈值，为审计提供可复制的操作规范。

**🔧 技术方法**

技术实现主要采用：① 基于 SKOS 的语义层；② 四层架构（基础、技术核心、行业附录、项目实例）；③ 针对每种风险机制的 probe、metric、severity band、grade 规则；④ 公开的度量值和评级计算脚本；⑤ 与 DecodingTrust Benchmark 的接口集成。

**📊 数据集**

使用的数据集是公开的 DecodingTrust benchmark（包含 PII 泄露情景 2），并在 GPT‑4‑0314 上执行；该 benchmark 作为验证真实系统性能的参考。

**📈 对比分析**

方法对比主要体现在：① 与 NIST 600‑1、EU AI Act、ISO/IEC 42001 等 18 个框架的映射覆盖率；② 在 DecodingTrust 上对 PII 泄露风险的测量与严重度阈值的匹配；③ 与现有多达 74 个风险分类体系的对照，证明其在操作性和可扩展性上的优势；绩效表现方面，PII 泄露在三种对抗条件下的披露率从 0% 提升至 84%，对应严重度 1、4、5，最终评级为 E（系统性）。

**⚠️ 局限性**

局限性包括：① 仅公开了部分子分类（PII 泄露、差异化影响等）的完整实例，其余 43 项正在内部研发；② 公开方法论仍以内部团队实现为主，外部复现需自行实现；③ 对 IP 侵权等非隐私风险的覆盖不足；④ 目前仅在 GPT‑4‑0314 上验证，缺乏跨模型、多域的进一步实证；⑤ 公开的严重度阈值为初步校准，后续需要更多业务场景来细化。

---

## 458. Bayesian Sparse Low-Rank Adaptation for Large Language Model Uncertainty Estimation

**arXiv ID:** 2607.02182 | [PDF](https://arxiv.org/pdf/2607.02182v1)

**作者:** Jijie Zhang `[一作]` (Jilin University), Dandan Guo `[通讯]` (Jilin University)

**通讯引用:** 13235 | [OpenAlex ID](https://openalex.org/A5100378159)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DALorRA方法，通过在LoRA的rank维度上学习可变贝叶斯掩码，实现LLM的无偏校准并保持推理准确性。

**💡 创新点**

在LoRA结构的rank层面引入可学习的Bernoulli掩码，结合变分推理和蒙特卡洛采样，兼顾贝叶斯与深度集成的优点，实现参数效率与高质量不确定性估计。

**🔧 技术方法**

变分贝叶斯推理、Gumbel‑Sigmoid重参数化、LoRA适配器、蒙特卡洛采样、深度集成思想。

**📊 数据集**

Winogrande、ARC、OpenBookQA、BoolQ、Chem、Phy等常见推理基准，包含OOD迁移实验。

**📈 对比分析**

与LoRA、BLoB、C‑LoRA、MCD、ENS、LA、MLE、MAP等基线对比，DALorRA在ACC、ECE、NLL三项指标上大多数场景实现最优或次优，显著降低ECE/NLL而保持ACC。

**⚠️ 局限性**

仅针对分类式推理任务；Inference需多次采样导致延迟；仅在rank层面建模不确定性，可能忽略权重级细粒度；假设Bernoulli独立性忽略层间相关性。

---

## 459. Synthetic Contact with AI Reduces Cross-Partisan Animosity

**arXiv ID:** 2607.02181 | [PDF](https://arxiv.org/pdf/2607.02181v1)

**作者:** Benjamin Lira `[一作]` (University of Pennsylvania), Olivier Toubia `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过五个预注册实验，测试了让政治对立派别成员与代表对立阵营的AI聊天机器人进行简短对话（Synthetic Contact），并证明此方式能降低跨党派的厌恶、纠正误解、提升相互温暖、影响实际行为，并在一定程度上在一周后仍保留。

**💡 创新点**

创新点在于提出并验证了“合成接触”（synthetic contact）——一种可扩展、成本低、可在线即用且被对立双方更易接受的跨党派互动模式，区别于传统面对面接触和线上冲突放大的数字干预。

**🔧 技术方法**

主要技术为基于大型语言模型（GPT‑4o、GPT‑5.2‑mini）构建的对立阵营聊天机器人，配合自适应上升-下降 staircase、混合效应回归、逻辑回归以及GPT‑5.4‑mini进行对话内容的认知与情感维度编码。

**📊 数据集**

数据集来源为美国两党自我认定的参与者（共计约10,000人），通过CloudResearch Connect和Prolific收集；使用的测量包括环境消费态度（GREEN量表）、移民议题问卷、感情温度计和自我报告的知情度与共情度。

**📈 对比分析**

比较方法采用了预注册的双组对照、单组前后测、三组对照（聊天+游戏）以及行为选择和纵向随访实验；效应大小普遍为Cohen's d ≈ 0.5–1.2（对温暖）和odds ratio ≈ 2–3（行为选择），误认纠正幅度约为0.4点。

**⚠️ 局限性**

局限性包括：干预效果随时间衰减，约一周后大部分温暖效应消失；样本仅限美国党派，且议题仅为环境和移民；只测量了感情温度，未涉及更全面的偏见维度；聊天机器人仍可能保留模型偏见，且未验证长期累积效应。

---

## 460. Local polynomial factorisation: improving the Montes algorithm

**arXiv ID:** 2607.02153 | [PDF](https://arxiv.org/pdf/2607.02153v1)

**作者:** Poteaux Adrien `[一作]`, Weimann Martin `[通讯]`

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文显著改进了Nart-Montes算法，用于在完全离散值环上因式分解多项式。主要贡献是扩展了Hensel引理，并提出了一种新的分治策略。

**💡 创新点**

创新点在于将Hensel算法适应于高阶牛顿多边形，并提出了一种新的分治策略，使得可以同时提升所有因子的精度，复杂度几乎线性。

**🔧 技术方法**

使用了扩展的Hensel算法和分治策略，结合牛顿多边形的高阶特性。

**📊 数据集**

使用了完全离散值环上的多项式F∈[x]，具体数据集未明确提及。

**📈 对比分析**

与现有方法相比，本文的方法在复杂度上有显著改进，特别是在处理不可约性和因式分解问题时，复杂度几乎线性。

**⚠️ 局限性**

限制在于当残余特征为零或足够高时，近似根作为类型的代表是有效的，但在某些情况下可能无法使用近似根。

---

## 461. Electronic Bursting Neuron: design, equations and hardware implementation

**arXiv ID:** 2607.02122 | [PDF](https://arxiv.org/pdf/2607.02122v1)

**作者:** Lev V. Takaishvili `[一作]` (Peter the Great St. Petersburg Polytechnic University), Ilya V. Sysoev `[通讯]` (Peter the Great St. Petersburg Polytechnic University)

**通讯引用:** 737 | [OpenAlex ID](https://openalex.org/A5054305042)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种基于相位锁定环方程改进的电子神经元模型，并通过实验验证其在不同动力学模式下的表现。

**💡 创新点**

将原始方程中的周期余弦函数改为双曲正切并引入离散相位控制，避免无限相位增长，极大简化硬件实现，同时保留原模型的所有动力学行为。

**🔧 技术方法**

模拟电路设计、运算放大器积分器、比较器、RS触发器、模拟乘法器以及双曲正切实现，结合数值模拟与实验数据。

**📊 数据集**

采用数值仿真时间序列（不同参数设置下的模型输出）与实验设备采集的时间序列进行对比。

**📈 对比分析**

通过一维分岔图、时间序列对比和参数空间扫描，证明修改模型与原模型在分岔结构和动力学模式上保持一致；实验结果与仿真相符，表现良好。

**⚠️ 局限性**

参数设置不精确导致实验与理论差异；相位限制与双曲正切近似在某些参数区间需要更大惯性参数；需进一步重构以精确匹配实验系统。

---

## 462. Behind the Refusal: Determining Guardrail Activation via Behavioral Monitoring

**arXiv ID:** 2607.02121 | [PDF](https://arxiv.org/pdf/2607.02121v1)

**作者:** William Hackett `[一作]` (Mindgard), Peter Garraghan `[通讯]` (Mindgard)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在黑盒条件下，利用仅40个提示对AI系统进行门控探测、检测门控阻断与LLM拒绝，并生成门控阻断指纹。

**💡 创新点**

创新点在于：①仅依赖HTTP、词法与时序三类可观测特征实现零知识门控探测；②同时判别门控阻断与LLM拒绝；③通过行为监测构建门控阻断指纹，提升攻击策略的定向优化。

**🔧 技术方法**

技术手段包括：HTTP状态码、头部与响应体结构分析；词法特征提取（LLM判别器、响应重复率）；时序特征（每词耗时、总时延）；Kolmogorov–Smirnov 与 Fisher exact 检验，结合 Benjamini–Hochberg FDR 进行多重校正。

**📊 数据集**

使用的公开数据集：PI Guard（提示注入、越狱、毒性）与 Nvidia Aegis（安全违规），以及9种门控系统（如OpenAI Omni、Microsoft Azure Content Safety 等）配合3种LLM（gpt‑4.1、claude‑sonnet‑4‑6、gemini‑2.5‑flash）。

**📈 对比分析**

与仅使用 HTTP 特征（HTTP Recon）和基于 LLM 判别器（LLM Judge）对比，门控探测准确率达 100%；在区分门控阻断与 LLM 拒绝上，平均 F1 为 0.98（最高 1.00），显著优于对手的 0.66 与 0.81。

**⚠️ 局限性**

局限性包括：时序特征易受网络噪声影响导致误报；部分门控在某些内容类别下信号弱（如 Protect AI DeBERTa v2 的越狱检测）；方法需要约 40 条提示，对极其隐蔽或动态调整的门控仍可能检测不到。

---

## 463. Planning over Matrix-Factorization MDPs for Candidate Generation

**arXiv ID:** 2607.02115 | [PDF](https://arxiv.org/pdf/2607.02115v1)

**作者:** Mikhail Trapeznikov `[一作]` (AI VK), Maksim Utushkin `[通讯]` (AI VK)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在传统的矩阵分解（Implicit ALS）检索基础上，引入一个基于状态转移的MDP框架，对top-K推荐序列进行规划，从而实现动态状态更新后的顺序化推荐。

**💡 创新点**

创新点在于：①将隐式ALS的后验分布[A⁻¹, u]直接作为MDP状态；②使用Sherman–Morrison公式实现闭式一阶fold‑in转移；③构造结合相关性与后验对齐的轨迹奖励；④在此框架下仅通过轻量级的单步规划或MCTS即可显著提升检索性能。

**🔧 技术方法**

技术包括：隐式ALS（矩阵分解）、Sherman–Morrison闭式更新、MDP/强化学习、Monte Carlo Tree Search（MCTS）以及近似最近邻索引用于动作约束；评估指标为Recall@10和nDCG@10。

**📊 数据集**

使用了五个公开/工业数据集：MovieLens‑1M、KuaiRec、VK‑LSVD（热门与非热门子集）、YAMBDA；实验分别在留出最近n次交互（LLN）和全局时间切分（GTS）两种协议上进行。

**📈 对比分析**

比较方式：对比静态检索（无规划）、单步规划（Plan‑1）和多步MCTS规划（Plan‑K），保持相同的嵌入向量；结果表明在LLN协议下Plan‑1在所有数据集上显著提升Recall和nDCG，Plan‑K仅在部分场景进一步提升；在GTS协议下提升有限，主要取决于相似度度量（余弦优于点积）。

**⚠️ 局限性**

局限性：①规划假设被推荐项必定被接受，忽略了拒绝和会话结束等现实动态；②对时间漂移不具备鲁棒性，GTS实验中提升消失；③较深的MCTS计算开销高，实际部署需根据验证结果权衡；④仅在固定ALS嵌入上实验，未验证与其他序列模型的组合效果。

---

## 464. Bridge-WA: Predicting Where and How the World Changes for Robotic Action

**arXiv ID:** 2607.02195 | [PDF](https://arxiv.org/pdf/2607.02195v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 465. Visual Analytics of Neighborhood Attribute Profiles for Exploring Structural Equivalence

**arXiv ID:** 2607.02163 | [PDF](https://arxiv.org/pdf/2607.02163v1)

**作者:** Kohei Arimoto `[一作]` (Teikoku Databank, Ltd.), Masahiko Itoh `[通讯]` (Hokkaido Information University)

**通讯引用:** 15153 | [OpenAlex ID](https://openalex.org/A5109126172)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

利用邻域属性档案和UMAP进行可视化分析，揭示企业交易网络中节点的真实结构角色，避免仅依赖静态属性标签或线性空间假设的相似性探测。

**💡 创新点**

提出质疑传统方法对线性、均匀潜在空间的隐式假设，并通过可视化展示同一行业标签下结构角色的碎片化与连续性，揭示语义标签与结构等价之间的显著差异。

**🔧 技术方法**

邻域属性向量构造（区分进/出交易方向）、L1归一化消除规模偏差、UMAP非线性降维（使用余弦距离）以及交互式可视化。

**📊 数据集**

日本主要信用研究机构提供的企业交易网络数据，约887 289家公司、6 725 600条有向交易边，包含13个行业分类标签。

**📈 对比分析**

与传统相似度方法（如SimRank、node2vec）以及仅基于属性聚合的方法对比；通过可视化案例（供应链连续转变、行业碎片化）证明其更能捕捉真实结构角色，虽未给出数值指标，但在可解释性和结构辨识上优于对比方法。

**⚠️ 局限性**

主要局限在于依赖可视化与人工判断，缺乏量化相似性评估；仅考虑行业标签，忽略其他属性；UMAP参数选择具有一定经验性；对极大规模网络的实时可视化与交互仍存在计算瓶颈。

---

## 466. MedSaab-US: A Backpropagation-Free Multi-Scale Wavelet-Saab Framework for Thyroid Nodule Segmentation in Ultrasound Images

**arXiv ID:** 2607.02209 | [PDF](https://arxiv.org/pdf/2607.02209v1)

**作者:** Mohammad Amanour Rahman `[一作]` (Ahsanullah University of Science and Technology), Mohammad Amanour Rahman `[通讯]` (Ahsanullah University of Science and Technology)

**通讯引用:** 72 | [OpenAlex ID](https://openalex.org/A5113355497)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种无需反向传播的绿色学习框架 MedSaab-US，用于甲状腺超声图像中结节分割。

**💡 创新点**

将多尺度离散小波变换与多级 Saab 变换、标签辅助贪心特征选择以及 XGBoost 结合，形成端到端的无梯度学习方法。

**🔧 技术方法**

采用 2 级 Daubechies-4 小波分解、三尺度（5×5、11×11、21×21）Saab PCA 变换、位置编码、LAG 特征筛选及 XGBoost 分类器。

**📊 数据集**

在 TN3K 数据集（2879 训练 / 614 测试）上进行实验。

**📈 对比分析**

与多种深度学习模型以及传统 Otsu+Morph、RF+Haralick 基准对比，MedSaab-US 平均 Dice 为 0.4784，明显优于传统基准但低于最佳 DL 模型。

**⚠️ 局限性**

受限于局部特征、对等回声结节识别不足以及固定空间先验，导致与最佳 DL 方法相差约 0.37 点，并不适用于所有超声采集场景。

---

## 467. Self-explainable Operator Learning for Discovering Spatial Patterns in Functional Data

**arXiv ID:** 2607.02203 | [PDF](https://arxiv.org/pdf/2607.02203v1)

**作者:** Mojgan Alishiri `[一作]` (University of Utah), Amirhossein Arzani `[通讯]` (University of Utah)

**通讯引用:** 2016 | [OpenAlex ID](https://openalex.org/A5054293534)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种自解释的算子学习框架，将算子表示为线性组合的积分方程，并通过分区积分实现预测结果的可分解解释；

**💡 创新点**

在算子结构中直接嵌入可解释性，允许在不依赖后置解释工具的前提下，按空间子域直接获取输入-输出关系；

**🔧 技术方法**

使用基于核的泛化函数线性模型（积分方程）、L2正则化线性回归以及分区积分实现可解释性，同时与Kernel SHAP、遮挡敏感度和Grad‑CAM等后置XAI方法进行对比；

**📊 数据集**

利用血流数据（脑动脉瘤内速度场与壁面剪切应力）和二维平板机翼的无稳流动速度场与阻力/升力系数进行实验；

**📈 对比分析**

与傅里叶神经算子（FNO）基线和后置解释方法比较，FNO在预测误差上更优，但自解释框架在解释一致性和物理可解释性方面表现突出，体现了精度与可解释性之间的权衡；

**⚠️ 局限性**

局限包括相对较低的预测精度、线性模型对非线性或高度杂乱输入的解释能力有限、对核带宽和正则化参数的敏感性，以及尚未在更广泛领域验证的可推广性。

---

## 468. Privacy-Preserving and Verifiable Approximate Distributed Coded Computing

**arXiv ID:** 2607.02187 | [PDF](https://arxiv.org/pdf/2607.02187v1)

**作者:** Xavier Martínez-Luaña `[一作]` (Galician Research and Development Center in Advanced Telecommunications), Rebeca P. Díaz-Redondo `[通讯]` (atlanTTic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的、模型无关的分布式机器学习安全框架（GPBACC+adversary‑resistant机制），同时在联邦学习和去中心化学习两种场景下实现隐私保护与对抗性鲁棒性。

**💡 创新点**

创新点包括：
① 将通用隐私增强的Berrut逼近编码计算（GPBACC）与特定范式的鲁棒聚合/验证机制（如Krum、Approximate Decode‑and‑Compare + Group Testing）相结合，构建了跨范式的完整防御体系；
② 通过攻击驱动（Membership Inference、Label Flipping、噪声注入等）对框架进行严格评估，验证隐私泄露与鲁棒性两方面的实用性；
③ 在去中心化学习中首次实现ADC+GT的prune‑and‑refine策略，以实现无可信聚合器的恶意节点检测与隔离。

**🔧 技术方法**

核心技术：
- GPBACC（Berrut插值 + 高斯噪声编码）实现模型无关的隐私增强；
- 传统鲁棒聚合（Median、Trimmed‑Mean、Krum、Multi‑Krum）用于联邦学习；
- Approximate Decode‑and‑Compare（ADC）与Group Testing（GT）用于去中心化学习的结果验证与恶意节点定位；
- 对比实验中使用多种基线技术（Multi‑Key Homomorphic Encryption、Sparse Vector Technique + DP）。

**📊 数据集**

实验数据集：
- MNIST 与 CIFAR‑10 数据集，用标准CNN架构进行训练，验证在不同攻击场景下的准确率与隐私泄漏。

**📈 对比分析**

评估方法与性能：
- 与未加密 baseline、MKHE、SVT 以及 ALCC（去中心化编码）进行对比；
- 通过准确率、RMIA AUC（Membership Inference）和对抗性攻击（Label Flipping、噪声注入）来衡量效果；
- 结果表明：GPBACC在保持与未加密模型相近的准确率的同时，显著降低隐私泄露（AUC接近 0.5）并且在鲁棒性攻击下表现优于或等价于 MKHE 与 SVT；
- 在去中心化学习中，GPBACC+ADC+GT 与 ALCC 的准确率相近，且对噪声攻击具备良好抵抗。

**⚠️ 局限性**

局限性：
- 近似编码导致的数值误差会影响基于距离的聚合（如 Krum）表现，需进一步调优噪声参数；
- Group Testing 在高噪声或误报率下可能误删诚实节点；
- 目前实验规模有限，未验证在更大模型或分布式系统中的可扩展性；
- 对高级后门/模型污染攻击的防护尚未充分探究；
- 需要手动调节多项参数（噪声方差、聚合阈值、GT 设计）以平衡隐私与精度。

---

## 469. A rubric-based controlled comparison of frontier language models on expert-authored clinical reasoning tasks

**arXiv ID:** 2607.02175 | [PDF](https://arxiv.org/pdf/2607.02175v1)

**作者:** Samiha A. Ismail `[一作]`, Ali Merali `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一个由临床医生编写的5个高难度临床情景的评价体系，使用细粒度加权的MECE rubric 对三种前沿LLM（GPT 5.4、Claude Opus 4.7、Gemini 3.1 Pro）进行单轮对照评估。

**💡 创新点**

①临床医生黄金答案生成细粒度权重rubric；②控制对比实验的统一 harness 与第三方评估框架；③发现模型在高权重临床优先级上表现逆转；④使用LLM auto‑rater 与专家对齐的校准。

**🔧 技术方法**

前沿LLM（GPT、Claude、Gemini）API调用；MECE权重化 rubric 设计；单轮 prompt harness；LLM auto‑rater 进行自动评分；统计分析及失败分布按权重/类别。

**📊 数据集**

五个由临床专家撰写的合成病例场景（共184个评价准则，5个专业领域），不基于公开数据，符合 HIPAA 合成。

**📈 对比分析**

在统一 harness 下，三模型通过加权 rubric pass rate 进行比较。平均通过率：Claude 0.47 > GPT 0.39 > Gemini 0.37。模型在低权重指标 80–90% 通过，但在最高权重（critical）指标仅 32–42% 通过；约 52% 的 weight‑5 条件未被任何模型满足。

**⚠️ 局限性**

样本仅 5 个任务、184 条准则，缺乏统计显著性；rubric 主要引用 UK/US 指南，跨系统迁移受限；每个准则只有单位专家标签，缺乏双重验证；未涵盖多轮、工具或检索场景。

---

## 470. Dynamic Neural Graph Encoding of Inference Processes in Deep Weight Space

**arXiv ID:** 2607.02166 | [PDF](https://arxiv.org/pdf/2607.02166v1)

**作者:** Di Wu `[一作]` (University of Toronto), Yang Wang `[通讯]` (Concordia University)

**通讯引用:** 14957 | [OpenAlex ID](https://openalex.org/A5100714578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将神经网络权重空间建模为动态图，设计 DNG-Encoder 对其进行时序图神经网络处理，并提出 INR2JLS 将 INR 权重映射到联合潜在空间，以支持下游分类和泛化预测。

**💡 创新点**

创新点在于：①使用动态图捕捉层级前向推理的时间序列，避免静态图的逆问题；②构建 RNN‑style 图神经网络 DNG‑Encoder 以模拟层间信息流；③引入联合潜在空间（INR+图像）而非仅重构权重，提升表示质量。

**🔧 技术方法**

采用的技术包括：动态图表示、基于 FiLM 的多头消息传递、GRU 更新、层归一化、随机傅里叶特征、联合潜在空间解码（逆卷积）以及自监督图像重构训练。

**📊 数据集**

使用的数据集有：MNIST、FashionMNIST、CIFAR‑10、CIFAR‑100（INR 分类）以及 Small CNN Zoo 与 CNN Wild Park（CNN 泛化预测）。

**📈 对比分析**

与 NFN、NFT、NG‑GNN、NG‑T 等现有方法对比，INR2JLS 在四个 INR 数据集上分别提升约 0.1%–8% 的分类准确率，在 CIFAR‑10/100 上提升 9%–10%，在 CNN 泛化预测上在 Kendall 相关系数上也优于其它基线，整体性能显著更优。

**⚠️ 局限性**

局限性包括：对生成式或权重编辑任务不如静态图或 token 化方法适用；对图像空间任务的表现仍落后于传统 CNN；对非常异构网络的解释能力仍有提升空间；此外，权重空间分析的可逆性可能带来安全和版权风险。

---

## 471. Probing Chemical Language Models: Effects of Pre-training and Fine-tuning

**arXiv ID:** 2607.02140 | [PDF](https://arxiv.org/pdf/2607.02140v1)

**作者:** Anna Karnysheva `[一作]` (Saarland University), Ji-Ung Lee `[通讯]` (Saarland University)

**通讯引用:** 609 | [OpenAlex ID](https://openalex.org/A5081951157)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统性研究化学语言模型（Chemberta、Molformer、Roberta‑zinc等）对SMILES表示的分子子结构的表征能力，并通过78种子结构的探针数据集评估预训练、随机初始化以及在脂溶性和水溶性预测任务上的微调对模型内部表示的影响。

**💡 创新点**

首次构建大规模（78种）分子子结构探针数据集，全面比较预训练与随机初始化、以及微调前后在不同层级的表征差异，揭示模型学习与化学理论的一致性及“忘记”子结构的现象。

**🔧 技术方法**

使用线性探针（线性分类器）评估Transformer各层隐藏状态对二分类子结构任务的可分性；通过宏F1衡量预训练与微调效果；对比不同预训练数据量与模型结构对结果的影响。

**📊 数据集**

预训练使用ZINC/PCQM4Mv2 SMILES数据；下游任务使用MoleculeNet的lipophilicity（logD）和solubility（logS）数据集；探针数据集从PCQM4Mv2抽样得到的78种子结构。

**📈 对比分析**

通过与随机初始化模型、Majority‑class baseline以及传统指纹+传统机器学习模型（LR、SVM、XGB）对比，发现预训练模型在上层显著提升子结构识别，微调对任务相关子结构影响更大；在下游任务上预训练模型均优于随机初始化，并且优于传统指纹+传统模型。

**⚠️ 局限性**

探针仅衡量线性可分性，未证明特征在实际预测中被使用；子结构识别为二分类，未考虑数量或位置信息；数据集选择有限，难以推广至其他化学任务；部分子结构在预训练中可能被遗忘；化学理论简化假设可能影响解释。

---

## 472. AdaCount: Training-Free Similarity-Guided Spatial and Feature Adaptation for Zero-Shot Object Counting

**arXiv ID:** 2607.02139 | [PDF](https://arxiv.org/pdf/2607.02139v1)

**作者:** Muhammad Ibraheem Siddiqui `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Muhammad Haris Khan `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 3736 | [OpenAlex ID](https://openalex.org/A5032830353)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 AdaCount，一个训练-free 的零样本物体计数框架，利用 SAM3 的原型驱动相似度图实现空间裁剪与特征调制，提升对密集场景的计数精度。

**💡 创新点**

创新点在于将原型驱动相似度图用于指导可逆空间重映射和残差门控特征调制，既高效分配图像分辨率，又保留全局上下文，避免多轮推理。

**🔧 技术方法**

采用 SAM3、原型平均池化、余弦相似度、可逆分布式裁剪、残差门控特征调制以及双通道推理等技术。

**📊 数据集**

在 FSC-147、CARPK、OmniCount-191、MBM、PerSense-D 和 PrACo 等六个计数基准上进行评测。

**📈 对比分析**

与训练-free 与训练式基准对比，AdaCount 在 MAE/RMSE 上普遍优于 SAM3、SAM3Count，甚至在部分数据集上达到 SOTA，推理时间仅比 SAM3 增加约 2.5 倍。

**⚠️ 局限性**

局限性在于依赖初始 SAM3 的高置信检测，若初步检测失败或缺失实例，后续相似度引导失效，导致计数误差。

---

## 473. NEvo: Neural-Guided Evolutionary Video Synthesis for Dynamic Visual Selectivity

**arXiv ID:** 2607.02317 | [PDF](https://arxiv.org/pdf/2607.02317v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 474. Coding-agents can replicate scientific machine learning papers

**arXiv ID:** 2607.02134 | [PDF](https://arxiv.org/pdf/2607.02134v1)

**作者:** Atharva Hans `[一作]` (Purdue University), Ilias Bilionis `[通讯]` (Purdue University)

**通讯引用:** 3223 | [OpenAlex ID](https://openalex.org/A5043072708)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了Paper-replication工作流，利用编码代理在仅基于论文材料的条件下重建并验证科学机器学习论文中的计算性主张，记录每个目标的生成结果、方法重构、执行过程、比对与报告；

**💡 创新点**

将论文复现定义为目标级证据任务，采用持久化工作空间与验证检查相结合的方式，确保复制过程透明且可追溯；

**🔧 技术方法**

采用编码代理技能、持久化记录文件、目标分解与任务账本、规范文件、执行记录、版本追踪、自动比对与报告生成等技术；

**📊 数据集**

在四篇科学机器学习论文（PIFT、PINN-I、PINN-II、SINDy）中使用其原始数据集和引用资源进行评估；

**📈 对比分析**

通过对照论文主张，比较数值误差、置信区间覆盖率、可视化结果等，12次独立运行均通过完成门槛，目标覆盖率、数值保真度与执行时间等指标均符合预期；

**⚠️ 局限性**

存在运行结果在不同尝试间的可变性、对细节阐述依赖、仍需人工干预验证以及仅针对计算性主张，缺乏对实验物理过程的复现等限制。

---

## 475. Enhancing Fitness Intelligence through Domain-Specific LLM Post-Training

**arXiv ID:** 2607.02118 | [PDF](https://arxiv.org/pdf/2607.02118v1)

**作者:** Xingtao Zhao `[一作]` (Beihang University), Han Jiang `[通讯]` (Beihang University)

**通讯引用:** 484976 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并训练了一系列基于 Qwen3 的健身专用 LLM，采用三阶段后训练（持续预训练、监督微调、强化学习）提升科学健身指导性能。

**💡 创新点**

创新点在于构建高质量健身知识语料、设计连续预训练 + 两阶段混合监督微调 + DAPO 强化学习的三阶段管线，并结合混合奖励函数与模式奖励显著提升专业认证考试成绩且保持通用能力。

**🔧 技术方法**

技术包括领域知识工程、持续预训练（CPT）、两阶段混合监督微调（SFT）、奖励模型 + 模式奖励、基于 DAPO 的偏好强化学习。

**📊 数据集**

使用了约 30B 标记的健身领域语料（ACSM、NSCA 等权威教材、指南），通用语料库，专业认证题库及经过验证的推理数据集。

**📈 对比分析**

与 Qwen3 基线、DeepSeek‑V3.2、Gemini‑3.1‑Flash、GPT‑5.4‑Mini 等大模型对比，在 ACSM‑EP/NSCA‑CSCS 考试中分别提升 10%+ 与 12%+，32B 版本甚至超过 100B 参数的闭源模型。

**⚠️ 局限性**

局限性包括对训练数据质量高度依赖、RL 奖励设计复杂且可能导致特定任务过拟合，且尚未验证跨文化/多语言适用性。

---

## 476. A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets

**arXiv ID:** 2607.02303 | [PDF](https://arxiv.org/pdf/2607.02303v1)

**作者:** Wanyun Cui `[一作]` (Shanghai University of Finance and Economics), Wanyun Cui `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 883 | [OpenAlex ID](https://openalex.org/A5103026537)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在线性注意力模型上加入了一个受海马启发的有限精确KV缓存，实现对“惊讶”token的挑选和精准检索，从而在保持 O(1) 记忆的同时显著提升了长距离记忆和语言建模性能。

**💡 创新点**

创新点在于：①用模型自身的 delta‑rule 写入幅度（即惊讶度）作为无参数的缓存淘汰信号；②采用解耦的 RMSNorm‑γ 对缓存查询进行锐化，使得缓存能进行近似 argmax 检索而非软平均；③将此机制无缝集成到 Gated DeltaNet 的每一层，实现了“海马+皮层”双重记忆的半参数测试时回归框架。

**🔧 技术方法**

核心技术包括：DeltaNet/Gated DeltaNet 的线性注意力结构；利用 delta‑rule 写入幅度作为淘汰得分；RMSNorm‑γ 对缓存键值进行归一化提升检索尖锐度；与传统软注意力对比的半参数回归分析。

**📊 数据集**

使用 15 B 个 SlimPajama 语料进行训练，并在 Wikitext‑103、LAMBADA、ARC、PIQA、HellaSwag、WinoGrande、BoolQ、SciQ、OpenBookQA、FDA、SWDE、SQuAD 以及 RULER（needle‑in‑haystack）等多种评测数据集上进行测试。

**📈 对比分析**

与 GDN、GLA、GSA、KDA 等线性注意力基线以及全注意力 Transformer++ 进行对比；在 340 M 参数下，Wikitext perplexity 降至 22.92（比同基线低 16.1%），在 FDA、SWDE 上实现最高线性检索性能，并在 32 k 上保持 0.58 的针尖检索召回率，明显优于仅使用递归状态或仅靠 recency 缓存的模型。

**⚠️ 局限性**

限制主要体现在：缓存大小有限（≈321 tokens），因此在极长或针尖密集的上下文中无法捕捉所有关键信息；相较于全注意力，精确提取（如 FDA）仍有差距；实验结果多基于单一随机种子，未与学习型缓存淘汰模块（如 LTE）做充分对比。

---

## 477. Search-based Testing of Vision Language Models for In-Car Scene Understanding

**arXiv ID:** 2607.02300 | [PDF](https://arxiv.org/pdf/2607.02300v1)

**作者:** Lev Sorokin `[一作]` (BMW Group), Andrea Stocco `[通讯]` (Technical University of Munich)

**通讯引用:** 2559 | [OpenAlex ID](https://openalex.org/A5027652385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种基于可控渲染和搜索优化的自动化测试框架，用于评估基于视觉语言模型的车载场景理解系统。

**💡 创新点**

创新点在于将可渲染的场景参数化与遗传搜索相结合，设计了针对结构化与开放式输出的适配性fitness和oracle，系统性发现安全关键错误。

**🔧 技术方法**

使用SMPL-X生成驾驶员3D模型、灯光与物体渲染引擎、遗传算法搜索、以及多维度评分与阈值判定等技术。

**📊 数据集**

主要数据来源是自定义的参数化渲染生成的合成图像，实验中也使用了公开的VLM模型和工业原型的接口进行测试。

**📈 对比分析**

与随机场景生成对比，实验表明在VQA和Captioning任务中失败率提高最多10倍、失败覆盖率提升3.6倍，验证了方法的有效性与多样性。

**⚠️ 局限性**

局限性包括对真实环境的再现度有限、仅针对静态单视角场景、对单车模型的依赖以及对复杂交互和物理动态的建模不足。

---

## 478. Dual-Selective Network for Domain-Incremental Change Detection

**arXiv ID:** 2607.02299 | [PDF](https://arxiv.org/pdf/2607.02299v1)

**作者:** Yuzhi He `[一作]` (Xidian University), Jiahui Qu `[通讯]` (Xidian University)

**通讯引用:** 1819 | [OpenAlex ID](https://openalex.org/A5044736869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对域增量变化检测（DICD）中固定二分类标签空间导致的空间表征混乱和灾难性遗忘问题，提出了双选择增量网络 DSINet，能够在长序列域迁移过程中稳定学习。

**💡 创新点**

创新点包括：①Selective Spatial State Unit（S^3U）在空间特征层利用输入相关的状态空间选择机制，动态滤除域特异性噪声，保持域无关的结构；②Concentration-Balanced Distillation（CBD）在知识蒸馏层引入 α‑β Divergence，平衡硬度与置信度，避免概率过度平滑或模式崩塌。

**🔧 技术方法**

使用了视觉状态空间模型（SSM）/Mamba 作为骨干，配合 S^3U、CBD、Teacher‑Student 结构以及多尺度特征对齐损失，实现高效的 𝒪(N) 计算复杂度。

**📊 数据集**

在三大遥感数据集 SYSU‑CD、CDD 以及 PRCV 上进行实验，覆盖不同分辨率与季节变化的多时相影像。

**📈 对比分析**

与 CNN/Transformer/Mamba 静态模型及 MDINet 等增量基线在两阶段和三阶段域增量任务中对比，DSINet 在 Mem‑F1、Mem‑IoU、整体 F1/Iou 上均显著提升，最高 Mem‑F1 接近 70%，在保持历史知识的同时兼顾新域适应。

**⚠️ 局限性**

局限性：在极长域序列中仍会出现一定程度的性能衰退；对蒸馏温度、α、β 参数的选择敏感，且未系统评估跨城市迁移的泛化能力。

---

## 479. Coding Agents Are Guessing: Measuring Action-Boundary Violations in Underspecified DevOps Instructions

**arXiv ID:** 2607.02294 | [PDF](https://arxiv.org/pdf/2607.02294v1)

**作者:** Zimo Ji `[一作]` (Hong Kong University of Science and Technology), Shing-Chi Cheung `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 8746 | [OpenAlex ID](https://openalex.org/A5034057959)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个基于 69 个真实 DevOps 事故的基准，用来衡量 LLM 编码代理在指令不完整时是否会越过安全边界。

**💡 创新点**

创新点在于：①将指令欠缺拆分为意图清晰度、目标确定性和冲击半径三轴；②使用确定性 oracle 对安全成功、错误目标和超范围进行细粒度评分；③构造了 2,208 条提示的系统化评测集合。

**🔧 技术方法**

使用 LLM 编码代理（OpenCode、Claude Code、Codex）与 Docker 隔离环境、deterministic oracle、LLM 判别器（DeepSeek‑v4‑flash）以及自动执行模式进行实验。

**📊 数据集**

基于 69 个真实事故、CVE 或工具行为构建的任务集，形成 2,208 条指令变体，覆盖三轴变化。

**📈 对比分析**

对 5 种 agent×model 组合进行统一评测，报告安全成功率、错误目标率和超范围率；结果显示目标歧义导致错误目标率高达 75%，超范围率可达 87%，相较于单纯完成度评估更能揭示安全风险。

**⚠️ 局限性**

局限性包括：实验环境为容器化仿真，未覆盖真实生产与人类审核；每个任务仅定义单一安全动作；oracle 可能不覆盖所有可接受行为；任务集在某些控制面上覆盖不足；模型与框架更新快速，实验结果为时效性快照。

---

## 480. One More Time: Revisiting Neural Quantum States from a Reinforcement Learning Perspective

**arXiv ID:** 2607.02292 | [PDF](https://arxiv.org/pdf/2607.02292v1)

**作者:** Juan Agustín Duque `[一作]` (Université de Montréal), Anna Dawid `[通讯]` (Leiden University)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5066979291)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于 PPO 的自回归神经量子态优化方法 PWO，结合幅度和相位的剪裁保证训练稳定性并实现大规模量子波函数训练。

**💡 创新点**

将变分量子优化映射为优势策略梯度，首次引入 PPO 思路和相位剪裁到 NQS 中，提供理论收敛与 infidelity 约束保证。

**🔧 技术方法**

使用自回归 NQS、变分蒙特卡罗、PPO 近似、相位剪裁、无矩阵求逆的一阶优化以及相关理论证明。

**📊 数据集**

实验基于一维 Ising、J1‑J2、10×10 方格 J1‑J2 哈密顿量；并对 1.5B 参数 RWKV‑7 LLM 在 1D Ising 进行微调。

**📈 对比分析**

与 Adam、minSR、SPRING 在相同样本量和 GPU 资源下比较；PWO 在 Ising 达到 10^-7 误差约 5 分钟，J1‑J2 约 15 分钟；显著快于 Adam 和 minSR，且更稳定，能在 10×10 体系中取得更低能量。

**⚠️ 局限性**

对非自回归 NQS、非标量相位或更大系统的适用性有限；理论证明依赖可测支撑和幅度剪裁范围，极端超大模型可能失效；需进一步评估数值稳定性与样本效率。

---

## 481. Generalization in offline RL: The structure is more important than the amount of pessimism

**arXiv ID:** 2607.02288 | [PDF](https://arxiv.org/pdf/2607.02288v1)

**作者:** Max Weltevrede `[一作]` (Delft University of Technology), Wendelin Böhmer `[通讯]` (Delft University of Technology)

**通讯引用:** 433 | [OpenAlex ID](https://openalex.org/A5033832179)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文探讨了在离线强化学习的零射击策略迁移（ZSPT）场景下，悲观程度与泛化性能的关系，并从理论和实验两方面验证了对称性对泛化的关键作用；

**💡 创新点**

创新点在于提出悲观结构而非悲观程度决定泛化，并证明只要保持对称性，即使极度悲观也不损害性能，同时提出通过数据增强一致性损失（DAC）直接强制对称性，优于传统的完整数据增强训练；

**🔧 技术方法**

主要技术包括：GTI‑ZSPT 理论框架、无限宽神经网络与 NTK 分析、Q‑value 蒸馏、离线 RL 算法 IQL/CQL、以及四种数据增强策略（Aug‑D、Aug‑D‑Online、DAC‑Latent、DAC‑Output）；

**📊 数据集**

使用的实验数据集为一个旋转对称的 Reach‑er 环境，数据来源于单一情景（context 1）下的专家、混合和子最优轨迹；

**📈 对比分析**

实验对比了无 DA、传统 Aug‑D、Aug‑D‑Online、DAC‑Latent 与 DAC‑Output 四种方法，结果显示 DAC‑Output 在 IQL 与 CQL 中都实现了最高的测试回报，证明一致性损失能显著提升泛化性能；

**⚠️ 局限性**

局限性包括：仅验证了已知的群对称性；对外部或随机变换的泛化未知；实验仅覆盖简单的旋转 Reach‑er 与两种算法，需在更复杂环境与更多算法上验证；理论基于无限宽网络，有限宽度网络的完整适用性仍待进一步研究。

---

## 482. AGVBench: A Reliability-Oriented Benchmark of Data Augmentation for Vein Recognition

**arXiv ID:** 2607.02271 | [PDF](https://arxiv.org/pdf/2607.02271v1)

**作者:** Haiyang Li `[一作]` (Chongqing Technology and Business University), Xin Jin `[通讯]` (Westlake University)

**通讯引用:** 132931 | [OpenAlex ID](https://openalex.org/A5100451926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了AGVBench，一个针对静脉识别的数据增强可靠性评估基准；

**💡 创新点**

首次系统化评估30种增强方法在多模型、多数据集上的多维度可靠性，并揭示了精度与安全性、校准性之间的矛盾；

**🔧 技术方法**

采用多图混合、策略驱动、标签平滑等多类别增强技术，结合CNN、ViT和静脉专用网络进行实验；

**📊 数据集**

使用五个公开静脉数据集（SCUT1100、TJU600、VERA220、FV-USM、SDUMLA-HMT）进行评测；

**📈 对比分析**

通过Top‑1 Accuracy、EER、TAR@FAR、ECE、抗干扰与对抗鲁棒性等六维指标比较，发现多图混合（如MixUp、PuzzleMix）在精度上最优，但在校准和对抗稳健性上表现差；

**⚠️ 局限性**

局限性包括：几何变换易破坏血管拓扑，最高精度方法校准差且对抗攻击脆弱，且不同数据集对最佳增强方法存在显著差异。

---

## 483. Real-Time Visual Intelligence on Low-Cost UAVs: A Modular Approach for Tracking, Scanning, and Navigation

**arXiv ID:** 2607.02298 | [PDF](https://arxiv.org/pdf/2607.02298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 484. HERMES: A Multi-Granularity Labeling Substrate for Pre-training Data Mixtures

**arXiv ID:** 2607.02266 | [PDF](https://arxiv.org/pdf/2607.02266v1)

**作者:** Ziyun Qiao `[一作]` (Peking University), Yujun Li `[通讯]` (Wizard Quant)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于层次残差向量量化（HERMES）的数据标签体系，能够一次性为文档生成可通过前缀长度调节粒度的多级标签；

**💡 创新点**

创新点在于将标签粒度视为可调拨的“旋钮”，不需要重新聚类即可在256到约13万桶之间切换，并通过阶段二采样规则揭示粒度与采样策略的交互效应；

**🔧 技术方法**

使用了学习语义变换（Learned Semantic Transform）+三阶段残差向量量化（RVQ）来生成层次标签，结合了外部权重（DoReMi、Uniform）和基于FineWeb-Edu质量分数的阶段二采样；

**📊 数据集**

在内部约5000万文档的集合上训练，采用冻结的1024维编码器生成嵌入；

**📈 对比分析**

与KMeans、MiniBatchKMeans、BisectingKMeans、Plain RVQ等传统聚类以及WebOrganizer等标注体系进行对比，发现HERMES在256桶时与KMeans表现相当；在16项能力任务上，采用DoReMi-L1 + L12质量Top‑30%采样可提升宏平均准确率+0.0253，至L123时此优势消失；

**⚠️ 局限性**

局限包括：仅在1B参数、25B token实验验证；外部权重仅在L1上学习，未在更细粒度下自适应；使用的FineWeb‑Edu质量分数可能不具普适性；未对更深层次或更大代码本做探索；

---

## 485. Copewell: A Multi-Agent Swarm Architecture for Equitable Mental Wellness Support

**arXiv ID:** 2607.02245 | [PDF](https://arxiv.org/pdf/2607.02245v1)

**作者:** Seren Yenikent `[一作]` (Copewell), Katherine Ng `[通讯]` (Copewell)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一个多智能体群体架构Copewell，用于提供可扩展的心理健康支持，帮助弥补低中收入国家的专业资源缺口。

**💡 创新点**

创新点包括：1) 多源评估框架（自我报告、生理传感器、上下文数据）以降低算法偏差；2) 基于Russell的情绪循环模型（valence‑arousal）实现情绪映射并动态路由到四类专用智能体；3) 双模式干预（对话 + 证据基础感官疗法）以及嵌入式伦理监督智能体，实现实时安全与伦理监控。

**🔧 技术方法**

采用多模态数据融合、情绪映射算法、智能体路由与协作机制、伦理监督与危机检测模块、以及音频/视觉/多感官干预库。

**📊 数据集**

使用自我报告的情绪检视、可穿戴设备采集的生理信号（睡眠质量、HRV、活动量）以及日历上下文数据；未公开专用的公开数据集。

**📈 对比分析**

通过内部红队测试和早期beta部署进行安全性与可用性评估；安全召回率100%，精度100%，整体安全得分约94%；缺乏正式的临床效能或用户体验对比实验，主要以形成性反馈为主。

**⚠️ 局限性**

局限包括：1) 缺乏大规模、独立的实证验证；2) 多源权重、情绪映射与感官介入的映射尚未经过临床或跨文化验证；3) 用户样本规模小，安全测试为内部评估；4) 目前未提供公开数据集或详细算法参数。

---

## 486. Self-Gating Attention for Efficient Time Series Forecasting

**arXiv ID:** 2607.02344 | [PDF](https://arxiv.org/pdf/2607.02344v1)

**作者:** Dezheng Wang `[一作]` (Southeast University), Hongzhi Yin `[通讯]` (University of Queensland)

**通讯引用:** 18226 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种名为 Self‑Gating Attention (SGA) 的轻量化注意力机制，旨在提升时间序列预测模型的推理速度和内存利用率。

**💡 创新点**

创新点在于将自注意力的查询‑键投影去掉，用共享的可学习矩阵捕捉全局重复的注意力模式，并通过输入依赖的残差矩阵捕获局部变化，从而实现线性时间与内存复杂度。

**🔧 技术方法**

技术手段包括：共享注意力权重矩阵、基于输入统计的残差构造、Top‑K 稀疏化、正交初始化等；并将 SGA 作为插件集成至多种 Transformer 预测骨干。

**📊 数据集**

实验使用了常规时间序列数据集（ETT、Weather、Exchange‑Rate）以及不规则时间序列数据集（PhysioNet ICU、Human Activity、USHCN），涵盖电力、金融、天气、医学监测和气候等领域。

**📈 对比分析**

与标准自注意力、Geometry、ProbSparse、AutoCorrelation、TSSA 等注意力变体在七个主干模型上进行对比，SGA 在 90 项评估中取得 84 项最佳或次佳结果；同时在 FLOPs、参数量和推理时间上平均分别降低约 60%、60% 和 25%，保持甚至提升了预测精度。

**⚠️ 局限性**

局限性包括：实验仅在公开数据集和高端 GPU 上完成，缺乏真实工业边缘设备的部署验证；在极度非平稳或突发 regime 的序列中共享矩阵的有效性可能下降；以及对超参数（如 Top‑K 比例、残差 dropout）的敏感度仍需进一步研究。

---

## 487. DisciplineGen-1M: A Large-Scale Dataset for Multidisciplinary Visual Generation and Editing

**arXiv ID:** 2607.02290 | [PDF](https://arxiv.org/pdf/2607.02290v1)

**作者:** Zhaokai Wang `[一作]` (Shanghai Jiao Tong University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 61181 | [OpenAlex ID](https://openalex.org/A5100636551)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一百万级多学科学术视觉生成与编辑数据集，并结合四条生成管道实现了高质量T2I与编辑样本；

**💡 创新点**

首次提供多学科、结构化且可验证的学术图像数据，结合基于推理的生成流程，显著提升跨学科生成与编辑性能；

**🔧 技术方法**

采用向量渲染（SVG/TikZ）、OCR编辑、程序化合成及大规模过滤等技术，并用Qwen-Image系列模型（Qwen3-VL-8B、Qwen-Image-2512、Qwen-Image-Edit-2511）配合LoRA、FlowMatch等训练技巧；

**📊 数据集**

使用自研的1.2M样本数据集（涵盖10个学科的T2I与编辑），并在GenExam、GRADE、WISE、RISE等公开基准上进行评估；

**📈 对比分析**

与FLUX.2、BAGEL、Qwen-Image-2512等开源模型对比，T2I在GenExam上获得51.4分、编辑在GRADE上获得58.7分，均为开源模型中的最高；在WISE和RISE上亦超越多数基线，显示出良好的通用推理能力；

**⚠️ 局限性**

仍无法完全赶上闭源系统，尤其在地图、乐谱、树图等高度结构化视觉上表现欠佳，逻辑推理编辑仍有提升空间，需要进一步扩充针对性数据和专门渲染策略。

---

## 488. BamiBERT: A New BERT-based Language Model for Vietnamese

**arXiv ID:** 2607.02259 | [PDF](https://arxiv.org/pdf/2607.02259v1)

**作者:** Dat Quoc Nguyen `[一作]` (Qualcomm AI Research), Linh The Nguyen `[通讯]` (Qualcomm AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并训练了 BamiBERT，一款针对越南语的 BERT 预训练模型，支持 2048 长度上下文且无需外部分词。

**💡 创新点**

突破了 PhoBERT 仅限 256 子词长度且需外部分词的局限，直接在原始文本上训练，并大幅提升跨域性能。

**🔧 技术方法**

采用 BERT base 架构、Masked Language Modeling、RoBERTa 动态遮蔽策略、字节级 BPE 词表、Adam 优化器和 A100 GPU 集群。

**📊 数据集**

使用 129GB 越南语通用文本语料进行预训练，评估八个越南语基准数据集（ViNLI、PhoNER_COVID19、UIT-VSFC、ViSpamReviews、UIT-ViSFD、UIT-ABSA Hotel/Restaurant）。

**📈 对比分析**

与 ViDeBERTa、ViSoBERT、XLM‑RoBERTa 及 PhoBERT 在相同任务上对比，BamiBERT 在 15 项指标中获得 11 项第一，3 项第二，3 项第三，显著优于 PhoBERT。

**⚠️ 局限性**

仍为 “base” 规模模型，且在酒店领域情感分类上略逊于 ViSoBERT，未来可进一步针对特定域进行微调或扩大模型规模。

---

## 489. ArcAD: Anomaly-Rectified Calibration for Cold-Start Supervised Anomaly Detection

**arXiv ID:** 2607.02252 | [PDF](https://arxiv.org/pdf/2607.02252v1)

**作者:** Ningning Han `[一作]` (Harbin Institute of Technology), Tonghua Su `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5033324841)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种用于工业异常检测冷启动场景的插件式校准框架ArcAD，能在少量正常样本和少量异常样本的条件下构建紧凑且可判别的正常边界。

**💡 创新点**

创新点在于：①在单位高维球面上采用von Mises-Fisher分布对特征进行建模；②使用Sinkhorn算法实现均衡的原型聚类（SPM）以避免模式崩塌；③通过原型受限的伪异常合成与真实异常对比学习，进一步校正正常边界（DGC）。

**🔧 技术方法**

核心技术包括：高维球面投影、vMF分布建模、Sinkhorn-Knopp算法、对比学习与伪异常合成、二元焦点损失等。

**📊 数据集**

在四大工业异常检测数据集上评估：MVTec-AD、VisA、Real-IAD、MANTA（含小样本冷启动设置）。

**📈 对比分析**

与多种无监督与监督基线（如Dinomaly、RD4AD、ReContrast、DRA、SDNet等）比较，ArcAD在单类和多类设置中均显著提升I-AUROC、P-AUROC和P-F1-max，单类最高可达100.0%/99.3%/67.5%，多类最高达99.7%/99.2%/68.9%，提升幅度从几个百分点到十几个百分点不等。

**⚠️ 局限性**

局限性：训练阶段增加了原型优化和伪异常合成的计算开销，但推理时保持与原始重建模型相同的速度。

---

## 490. Partition Rank and Algebraic Circuit Lower Bounds

**arXiv ID:** 2607.02241 | [PDF](https://arxiv.org/pdf/2607.02241v1)

**作者:** Cornelius Brand `[一作]` (University of Regensburg), Jiaheng Wang `[通讯]` (University of Regensburg)

**通讯引用:** 24 | [OpenAlex ID](https://openalex.org/A5101984439)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文利用Naslund提出的划分秩（partition rank）构造了一个新的张量秩概念，并证明在任何常数多线性度下，该秩（尤其是两切片秩）能够从下方给出乘法复杂度的下界，推广了Strassen的双线性复杂度理论。

**💡 创新点**

创新点在于：① 将划分秩与乘法复杂度关联，为高阶张量提供了可计算的下界；② 证明两切片秩（R_{d-2,1,1}）比传统的切片秩更强；③ 给出该秩的NP‑hard性证明，显示其计算难度；④ 与多项式强度（strength）相连，形成张量与多项式复杂度的新桥梁。

**🔧 技术方法**

主要技术包括：张量与同阶多项式的对应；同质乘法序列与同质乘法复杂度的定义；λ‑秩与λ‑强度的设置；通过归纳证明两切片强度≤同质乘法复杂度；利用集合多线性化（set‑multilinearization）将张量秩转化为多项式强度；以及改造Sawin–Tao的反链论证实现NP‑hard性。

**📊 数据集**

论文没有使用实验数据集；所有结果均为理论证明与计算复杂度分析。

**📈 对比分析**

与已有方法相比，本工作提供了更紧的下界（相比切片秩），并证明了相关秩的NP‑难度；理论上，该方法在常数阶张量上给出了最小乘法门数的多项式下界，显示了划分秩在细粒度复杂度中的潜在价值。

**⚠️ 局限性**

局限性包括：① 下界可能不是最优，尚未给出与乘法复杂度的上界对应；② NP‑hard性证明仅说明计算困难，并未提供高阶张量的显式强度大于线性的构造；③ 对于高阶张量的实际算法改进仍需进一步研究；④ 论文主要关注理论框架，缺乏实验验证或实际应用示例。

---

## 491. Challenges and Recommendations for LLMs-as-a-Judge in Multilingual Settings and Low-Resource Languages

**arXiv ID:** 2607.02235 | [PDF](https://arxiv.org/pdf/2607.02235v1)

**作者:** A. Seza Doğruöz `[一作]` (Universiteit Gent), David Ifeoluwa Adelani `[通讯]` (Mila - Quebec Ai Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统评估ACL Anthology中使用LLM-as-a-Judge在多语言/低资源环境下的研究，分析评判范式、任务类型、模型家族、语言覆盖与验证方式；

**💡 创新点**

提出四条针对多语言LLM评估的实用改进建议，并构建了多维度分类体系；

**🔧 技术方法**

采用文献检索、关键词匹配、人工标注与统计分析技术；

**📊 数据集**

主要基于ACL Anthology论文元数据，对已公开的MT、摘要、问答等任务数据集进行引用与整理；

**📈 对比分析**

通过对33篇论文的多维度归类与比较，发现低资源语言覆盖不足、评估验证不充分、依赖封闭模型，缺乏跨模型可靠性对比；

**⚠️ 局限性**

仅检索ACL Anthology、仅英文标题/摘要，可能遗漏非英语或其他会议论文；未进行新实验，缺乏自有数据与人类评估验证。

---

## 492. On the Role of Directionality in Structural Generalization

**arXiv ID:** 2607.02307 | [PDF](https://arxiv.org/pdf/2607.02307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 493. Purified OPSD: On-Policy Self-Distillation Without Losing How to Think

**arXiv ID:** 2607.02234 | [PDF](https://arxiv.org/pdf/2607.02234v1)

**作者:** Zhanming Shen `[一作]` (Zhejiang University), Jieping Ye `[通讯]` (Tongyi Lab, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对长链思考模型的自我监督蒸馏（OPSD）失败原因进行诊断，并提出利用参考‑only 教师分解与 PMI 目标的改进方法。

**💡 创新点**

创新点在于通过将教师监督拆解为参考诱导与问题条件两部分，发现后者被忽视；随后使用 PMI 将差异转化为可学习的目标分布，去除参考快捷路径。

**🔧 技术方法**

采用了 On‑Policy Self‑Distillation、参考‑only 教师探针、点互信息（PMI）目标、Jensen‑Shannon 散度蒸馏、中心化与软裁剪、LoRA 微调等技术。

**📊 数据集**

使用 Math‑CoT‑20K 与 DASD‑10K 两个包含参考解答的数学推理数据集。

**📈 对比分析**

在 AIME 2024/25、HMMT 2025 等基准上，每 50 步评估一次，与基线模型和标准 OPSD 对比，OPSD‑PMI 在所有四个长 CoT 模型上均显著提升性能，并保持反射性标记的稳定。

**⚠️ 局限性**

局限性包括仍需参考解答作为教师信息，对 β 与裁剪阈值 c 的选择存在一定敏感性；对极端 PMI 值的软裁剪虽保证稳定，但可能略微压制有效信号，且未彻底解决跨域泛化的挑战。

---

## 494. Guiding Human Validation of LLM-Generated Code via Verifiable Literate Programming

**arXiv ID:** 2607.02333 | [PDF](https://arxiv.org/pdf/2607.02333v1)

**作者:** Ziqi Yuan `[一作]` (University of Hong Kong), Chuan Wu `[通讯]` (University of Hong Kong)

**通讯引用:** 11480 | [OpenAlex ID](https://openalex.org/A5012597518)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出Verifiable Literate Programming (VLP)，在LLM生成代码的过程中引入可验证的自然语言文档层，让用户通过阅读文档而非代码来确认代码与需求的一致性，并支持细粒度的错误检测、用户反馈与自动化验证，提升代码正确率。

**💡 创新点**

创新点包括：
① 通过可确定性的语法导向翻译，将生成的Python代码转化为结构化、可读的自然语言文档；
② 采用实现相关的追踪链接（implementation‑relevant TLR）实现代码与原始提示的细粒度对齐，精准定位潜在意图偏差；
③ 结合API知识库与基于用户确认的文档生成匹配检验，再通过模型检查实现对API使用与断言的自动验证。

**🔧 技术方法**

技术手段：LLM（GPT‑5.4、DeepSeek V4 Flash）进行代码生成与文档翻译；LALR(1)语法和语法导向翻译规则；实现相关追踪链接与LLM辅助匹配；API知识库与自然语言提示驱动的错误检测；跨语言验证器（CrossHair）做有限状态模型检查；用户交互界面展示层次化文档与问题；使用LangGraph、NLTK等工具实现流程。

**📊 数据集**

使用的数据集：BigCodeBench‑Instruct（BCB）和QuantCodeEval（QCE），分别覆盖通用编程任务和金融领域复杂任务。

**📈 对比分析**

比较方法：对比无用户介入的直接生成、ClarifyGPT（预生成澄清）、TiCoder（基于测试的澄清）和PInG（注释交互）。在BCB上，VLP的pass@1从28.7%–73.2%提升到65.4%–93.5%；在QCE上也有显著提升。真实用户实验显示VLP获得最高满意度、最低劳动强度，并在正确率与用户时间上逼近Pareto最优。成本方面，VLP在token和用户时间上略高，但在整体性能上具有更高的性价比。

**⚠️ 局限性**

局限性：
① 仍无法覆盖隐藏需求（prompt未明确但评测要求的）导致的一些错误；
② 对LLM的翻译与修复能力高度依赖，模型误差仍会导致验证失败；
③ 主要针对Python，跨语言推广受限；
④ 需要额外的API知识库维护，缺失或错误的API描述会影响匹配与验证；
⑤ 在极短或极长任务中，用户仍需一定阅读工作，且对高模型prompt的敏感度较高。

---

## 495. A Stable Boundary Element Method for Reliable Long-Time Industrial Sound Emission

**arXiv ID:** 2607.02308 | [PDF](https://arxiv.org/pdf/2607.02308v1)

**作者:** Simon Schneider `[一作]` (Ulm University of Applied Sciences), Bernd Graf `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `b88c6eac-d57a-4623-a604-1f401f3eb268` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一种基于第一类超奇异边界积分方程的时域空间-时间Galerkin边界元方法，用于长时间工业噪声辐射仿真；

**💡 创新点**

其创新点在于将超奇异算子作为主算子，证明了连续与离散形式的稳定性与well‑posedness，使得在复杂几何和百万步时间域内仍保持数值稳定；

**🔧 技术方法**

采用张量积空间-时间离散、MOT（marching‑on‑time）求解策略、hp‑高精度四面体积分法以及非齐次Sobolev空间理论，实现了高效的矩阵组装与求解；

**📊 数据集**

使用真实工业齿轮箱壳体（橢圓式主壳OPG与ZF齿轮箱壳ZFG）以及在全吸音室实验得到的声压数据，Neumann边界数据来自结构动力学模拟；

**📈 对比分析**

通过与传统的双层算子二阶边界元和标度法比较，本文方法在长时间演化中始终保持稳定，误差（相对L2误差、声压级偏差）均低于1 dB，并与实验测量误差不超过0.3 dB；

**⚠️ 局限性**

主要限制是超奇异算子矩阵组装成本高，需进一步研究压缩与降阶技术，以提升大规模问题的计算效率。

---

## 496. FlowCIR: Semantic Transport via Flow Matching for Zero-Shot Composed Image Retrieval

**arXiv ID:** 2607.02284 | [PDF](https://arxiv.org/pdf/2607.02284v1)

**作者:** Zhenqi He `[一作]` (Hong Kong University of Science and Technology), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 97826 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FlowCIR，一种通过条件流匹配实现的零样本组合图像检索框架；

**💡 创新点**

创新点在于将组合检索视为参考图像条件下的语义传输，通过轻量级条件流场完成文本与视觉特征的连续融合，并引入推理阶段的多负向导向策略解决VLM对否定的处理弱点；

**🔧 技术方法**

主要技术包括预训练的CLIP视觉-语言模型、条件流匹配（Conditional Flow Matching）、时间条件神经ODE、InfoNCE对比损失、以及基于规则+小型LLM的多负向导向（Multi-Negative Steering）；

**📊 数据集**

使用的主要数据集包括CIRR、CIRCO、Fashion‑IQ作为评测基准，训练时采用HQ‑Edit‑200k合成编辑数据；

**📈 对比分析**

与文本反演、生成式以及LLM驱动的零样本CIR方法相比，FlowCIR在CIRR、CIRCO和Fashion‑IQ上均取得最强或竞争性表现，且训练时间仅为1小时单卡，显著低于传统方法；

**⚠️ 局限性**

局限性在于仍依赖CLIP空间的语义表达，对极细粒度视觉细节或极度多模态指令的处理可能不足，且多负向导向仅为推理时策略，缺乏训练时的自适应能力。

---

## 497. AnyGroundBench: A Specialized-Domain Benchmark for Video Grounding in Vision-Language Models

**arXiv ID:** 2607.02269 | [PDF](https://arxiv.org/pdf/2607.02269v1)

**作者:** Rintaro Otsubo `[一作]` (Keio University), Ryo Hachiuma `[通讯]` (NVIDIA)

**通讯引用:** 602 | [OpenAlex ID](https://openalex.org/A5020411666)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并发布了一个针对Spatio‑Temporal Video Grounding（STVG）的领域适配基准——AnyGroundBench，覆盖动物、工业、体育、手术和公共安全五个专业领域，并为每个领域提供专门的训练集，以评估Vision‑Language Models（VLMs）的零射击与少量样本适应性能。

**💡 创新点**

创新点主要体现在：① 通过将五个高度专业化领域的视频与已有公开数据统一标注，构建高质量、稠密的时空边框与文本查询；② 设立训练子集，系统化评估模型在新域的少样本适配能力；③ 引入In‑Context Learning（ICL）作为无梯度、通用的适配基线；④ 将STVG拆解为Spatial Video Grounding（SVG）、Temporal Video Grounding（TVG）与完整STVG，细致剖析性能瓶颈。

**🔧 技术方法**

使用的技术包括：VLMs（GPT‑4o、GPT‑5.1、Gemini系列、LLaVA‑ST、Qwen‑3、Eagle2.5、InternVL）在零射击与ICL条件下推理；检索式示例选择（文本、视觉或两者加权）；基于Grounding DINO + SAM2的自动/半自动标注工具；以及多种评估指标（vIoU、tIoU、sIoU）。

**📊 数据集**

数据集方面：AnyGroundBench自身收集并标注的 2,040 条视频，涵盖五个域：Animal Kingdom、Mouse Scratching、MECCANO、ENIGMA‑51、MultiSports、American Football、EgoSurgery、CholecTrack20、UCA、DoTA；每条视频配有自然语言查询、时段和时空框。

**📈 对比分析**

比较方法：对 15 种 VLM 进行零射击测试，并通过 2‑shot ICL 进行领域适配；对 STVG、TVG、SVG 三项任务分别使用 vIoU@0.3、tIoU@0.3、sIoU@0.3 评估。实验显示：所有模型在零射击下 STVG 分数普遍低于 5%，空间定位几乎失效；ICL 在部分域（如工业、公共安全）略有提升，但整体提升有限且不稳定；TVG 成绩相对更好，但仍不理想。

**⚠️ 局限性**

局限性：① 当前 VLM 在专业域的零射击与少样本适配能力不足，尤其是空间定位是主要瓶颈；② ICL 的提升有限，表明单纯的推理‑时适配方法无法充分利用训练样本；③ 数据集覆盖仅限于五个领域，可能不足以代表更广泛的专业场景；④ 高质量标注仍依赖人工与专家，成本高且难以扩展。

---

## 498. Elasticity in Parallel Sparse Triangular Solve

**arXiv ID:** 2607.02324 | [PDF](https://arxiv.org/pdf/2607.02324v1)

**作者:** Raphael S. Steiner `[一作]` (Huawei), A. N. Yzelman `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出在稀疏三角线性系统求解中使用新型的 stale‑synchronous‑parallel 执行模型，并设计了相应的调度器 ElasticDivide。

**💡 创新点**

创新点在于首次将 stale‑synchronous‑parallel 引入稀疏三角求解，利用超步中的延迟同步实现计算与同步的重叠，并在 DAG 上实现自适应的 staleness 2 调度。

**🔧 技术方法**

使用的技术包括基于 OpenMP 的并行内核、弱同步栅栏、带权有向无环图调度、基于优先级和可分配性的 ElasticDivide 算法。

**📊 数据集**

实验数据集包括 SuiteSparse 基准矩阵、其不完整 Cholesky 变体 iChol 以及采用 METIS 分区的 Metis 集合，共计 99 个稀疏矩阵。

**📈 对比分析**

通过在 ARM Kunpeng、AMD EPYC、Intel Xeon 等架构上与 GrowLocal、SpMP、HDagg 进行几何平均加速对比，ElasticDivide 在 ARM 上可达 30% 的加速，在 x86 上与 GrowLocal 相当或略优，平均加速幅度 7-30%。

**⚠️ 局限性**

局限性在于调度器的序列化生成时间相对较高，staleness 取值仅为 2 可能不适用于所有稀疏结构，并且在某些数据集上对同步粒度的假设导致性能波动。

---

## 499. HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion Guidance and Windowed Payload Curriculum

**arXiv ID:** 2607.02332 | [PDF](https://arxiv.org/pdf/2607.02332v1)

**作者:** Chenxin Liu `[一作]` (Tsinghua University), Jianyu Chen `[通讯]` (Tsinghua University)

**通讯引用:** 5775 | [OpenAlex ID](https://openalex.org/A5100611364)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种名为 HEFT 的全尺寸人形机器人重载遥操作框架，利用来自 VR 的嘈杂轨迹与专家标签训练单一可部署控制器，使 L7 机器人在无需切换控制器的情况下完成多种 24 kg 负载任务。

**💡 创新点**

创新点包括：Privileged Motion Guidance (PMG)——在训练时用离线重建的运动作为奖励，解决 VR 噪声对跟踪的影响；Windowed Payload Curriculum (WPC)——为每个运动窗口分配可承载最大负荷，实现负载鲁棒性而不改动接口。

**🔧 技术方法**

技术手段包括：PPO 强化学习、RMA 风格的演员-批评家结构、离线运动重建模型 RoHM、SMPL‑X 运动表示、动态负载仿真、数据驱动的自适应教师‑学生学习。

**📊 数据集**

使用了大规模 mocap 库（SEED、100STYLE、LaFAN1）和配对的 VR 数据集（含原始与重建轨迹），以及高动态 SEED 片段和随机采样 SEED 片段做评估。

**📈 对比分析**

与 TWIST2、SONIC、FALCON、TWIST2+FC 等基线在 G1 和 L7 机器人上进行比较；PMG 在嘈杂 VR 轨迹下根误差显著降低，WPC 使 30 kg 负载下成功率达到 75% 以上，且在无负载高动态任务上成功率提升至 73%；在真实 L7 机器人上完成 24 kg 负载的拾取、携带、蹲姿等任务。

**⚠️ 局限性**

局限性包括：需要离线重建 VR 轨迹和专家标签，迁移到新机器人或新跟踪系统需重新准备数据；负载模型仅近似为手腕力，未考虑抓取质量、几何、滑动等；仅在单一全尺寸平台验证，泛化性待进一步验证。

---

## 500. Facility Location Game with Envy Ratio

**arXiv ID:** 2607.02330 | [PDF](https://arxiv.org/pdf/2607.02330v1)

**作者:** Yuan Ding `[一作]` (Ocean University of China), Qingqin Nong `[通讯]` (Ocean University of China)

**通讯引用:** 415 | [OpenAlex ID](https://openalex.org/A5086028326)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在一维线性设施定位游戏中，提出并分析了最小化“Envy Ratio”（厌恶比）这一新的公平性目标，研究了在固定区间和相对区间两种约束下的策略无关机制。

**💡 创新点**

①首次将公平分配中的厌恶比引入设施定位问题；②给出了最优解与最优策略无关机制的对比，证明了确定性和随机化机制在不同约束下的最优逼近比；③在相对区间设置中构造了两类随机化机制（GLRM 与 LRM），并证明其在策略无关与团体策略无关之间的最佳权衡。

**🔧 技术方法**

基于理论分析与证明的方法，使用数学构造与逼近比分析（如凸性、单调性、三角不等式）来评估机制的公平性与策略性。

**📊 数据集**

无具体数据集，全部结果来自理论证明与数学实例。

**📈 对比分析**

通过理论对照（如 1+1/β、1+2γ/β 等上界）与下界（如 1+2β-1/(8β^2(1+β))）进行比较，表明在固定区间下最优随机机制的逼近比在 1.0314 与 2 之间；在相对区间下，最优随机机制的逼近比在 1+2/(1+β)^2 与 1+1/(2β) 之间。

**⚠️ 局限性**

存在理论与实际机制之间的逼近比差距，尤其在相对区间设置下的随机化机制；此外，模型仅限于一维线性空间与单设施，未考虑多设施、多维或更一般度量空间；最后，缺乏实验验证与真实数据评估。

---

## 501. The Moving Eye: Enhancing VLA Spatial Generalization via Hybrid Dynamic Data Collection

**arXiv ID:** 2607.02322 | [PDF](https://arxiv.org/pdf/2607.02322v1)

**作者:** Jincheng Tang `[一作]` (Lion Rock AI Lab, China Merchants Research Institute of Advanced Technology), Jiaxing Zhang `[通讯]` (Lion Rock AI Lab, China Merchants Research Institute of Advanced Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出双臂动态视角数据采集策略，结合固定、多视角和移动视角，以提升VLA模型的空间泛化。

**💡 创新点**

通过混合动态数据收集与黄金比例混合比系统打破摄像机-机器人、摄像机-物体、物体-位置的shortcut，且该策略对多种架构通用。

**🔧 技术方法**

利用双臂机器人环境摄像机、视觉-语言-动作模型训练、数据分布设计与混合采样以及对象位置多样化技术。

**📊 数据集**

在真实机器人上收集的pen pick‑and‑place与五类多物体任务数据，包含Fixed、Multi‑Fixed、Moving视角，未使用公开数据集。

**📈 对比分析**

与仅Fixed、仅Multi‑Fixed、仅Moving三种基线对比，使用成功率评估；混合策略在ID、OOD测试中提升至约90%，在跨任务与多架构上提升10‑30%。

**⚠️ 局限性**

仅验证桌面拾取与插入任务，未对长时序或高接触任务进行系统验证；动态视角需稳定帧率，快速切换可能导致性能下降。

---

## 502. Hybridizing a Grouping Metaheuristic with Reinforcement Learning for the One-Dimensional Bin Packing Problem

**arXiv ID:** 2607.02315 | [PDF](https://arxiv.org/pdf/2607.02315v1)

**作者:** Zitouni Rania `[一作]` (École Nationale Supérieure d'Informatique), Hasnaoui Sarah `[通讯]` (École Nationale Supérieure d'Informatique)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将分组遗传算法(HGGA)与强化学习相结合，提出RL‑HGGA方法以动态选择操作符

**💡 创新点**

创新点在于用Q‑learning学习搜索阶段的宏观操作选择策略，替代传统固定概率选择

**🔧 技术方法**

技术包括HGGA的分组编码、BPCX交叉、变异、局部搜索，以及基于离散状态的Q‑learning

**📊 数据集**

实验使用经典一维装箱基准集：Falkenauer T/U、Scholl和Hard28系列

**📈 对比分析**

对比FFD、HGGA和RL‑HGGA，RL‑HGGA在保持与HGGA相近的解质量的同时，将平均求解时间从约64秒压缩至1.3秒，显著提升时间‑质量折衷

**⚠️ 局限性**

局限性是略有质量损失、仅使用表格Q‑learning且状态离散化粗糙，对更大规模实例及更复杂状态空间的适用性待验证

---

## 503. InvSplat: Inverse Feed-Forward Scene Splatting

**arXiv ID:** 2607.02301 | [PDF](https://arxiv.org/pdf/2607.02301v1)

**作者:** Polina Karpikova `[一作]` (University of Tübingen), Andreas Geiger `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种一次性前向推理的多视角逆渲染框架，能够从已标定的图像中直接预测包含几何信息和物理材质属性（漫反射、金属度、粗糙度、法向）的三维高斯原语。

**💡 创新点**

首次将物理基础的材质属性与三维高斯分裂（3D Gaussian Splatting）结合，构建统一的三维场景表示，并通过双分支（几何+材质）网络在单次前向传播中完成几何与材质的协同重建，显著提升多视角一致性与实时渲染能力。

**🔧 技术方法**

采用 3D Gaussian Splatting 作为场景表示；利用基于 Transformer 的多视角几何编码器与 DINOv2 ViT 的属性提取；通过 DPT 深度解码器、点 Transformer 以及多头回归头生成高斯参数；使用可微分高斯光栅化器进行自监督渲染；结合像素级、LPIPS 及视角不变的深度与法向损失进行训练。

**📊 数据集**

主要在室内场景数据集 InteriorVerse、Structured3D 以及合成/真实数据集 Infinigen、RealEstate10K、DL3DV 上进行实验与评估。

**📈 对比分析**

与基于扩散模型的 DiffusionRenderer、单视角 DNF‑Intrinsic 以及多视角 MVInverse（含 fine‑tuned 版本）做对比。实验表明，虽然在材料精度上略低于 fine‑tuned MVInverse，但在跨视角一致性、材质细节保留和新视角合成上优于 2D 方法，并且具备实时渲染与可直接重光照的能力。

**⚠️ 局限性**

局限性包括：对相机位姿与深度估计的依赖，仍无法完全匹敌昂贵的逐场景优化方法；在复杂照明或极少视角下容易出现伪影；模型对光照变化的鲁棒性仍有提升空间。

---

## 504. Optimizing Visual Generative Models via Distribution-wise Rewards

**arXiv ID:** 2607.02291 | [PDF](https://arxiv.org/pdf/2607.02291v1)

**作者:** Ruihang Li `[一作]` (University of Science and Technology of China), Wenjie Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对视觉生成模型中基于样本的强化学习奖励容易出现奖励破解、图像质量下降和多样性受限的问题，本文提出一种基于分布级奖励的强化学习框架，并通过子集替换策略实现高效奖励估计，进一步将该奖励用于后置模型合并系数的优化。

**💡 创新点**

创新点在于：①用分布级（FID）奖励替代传统样本级奖励，显著降低奖励破解风险；②设计子集替换策略在保持奖励稠密性的同时降低计算成本；③将奖励信号用于后置模型合并，使训练过程不依赖SDE，从而消除训练-推理不一致；④在多个基准模型上验证该框架的普适性和有效性。

**🔧 技术方法**

核心技术包括：扩散模型的流匹配训练；分布级奖励的在线FID计算与子集替换策略；基于策略梯度的强化学习（带KL正则）；后置模型合并的RL优化（使用轻量级MLP产生合并系数）；以及ODE推理。

**📊 数据集**

实验主要使用 ImageNet 256×256（用于SiT模型微调）和 ImageNet 512×512（用于EDM2模型合并），无额外外部数据。

**📈 对比分析**

在SiT上，子集替换+RL将FID‑50K从 8.30 降至 5.77，FD_DINOv2从 230.39 降至 164.88；在EDM2上，RL‑EMA 将FID从 3.74/2.57 分别提升到 3.52/2.52；相较于现有基线（SiT、EDM2、ADM 等），模型在FID和感知多样性指标上均取得显著提升。

**⚠️ 局限性**

局限性包括：①子集替换策略和参考集规模需经验调优；②奖励信号仍依赖FID，对高维图像的统计估计可能受限；③目前仅验证于扩散模型，对其他生成框架的适用性尚未评估；④后置合并方法需预先保存多个检查点，存储和训练成本相对较高。

---

## 505. Dendritic In-Context Learning in a Single-Layer Spiking Neural Network

**arXiv ID:** 2607.02283 | [PDF](https://arxiv.org/pdf/2607.02283v1)

**作者:** Juwei Shen `[一作]` (Hong Kong Polytechnic University), Changwen Chen `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了单层分区尖峰神经网络DendriCL，利用树突递归实现在线Widrow–Hoff LMS，解决Garg-2022 ICL任务。

**💡 创新点**

将树突子电位作为主动在线学习器，而非传统突触可塑性；在单层尖峰网络中嵌入结构化LMS算法，并实现对高维ICL的稳定性与能效优势。

**🔧 技术方法**

采用分区神经元架构、Leaky在线LMS递归、反向传播训练冻结权重、线性探针验证动力学、Loihi能耗估计等技术。

**📊 数据集**

在Garg-2022线性回归与二分类以及两层ReLU网络回归的合成任务上进行评估，任务维度d=5至50，k=2d。

**📈 对比分析**

与Transformer、Spikformer、Pure LIF、Active Dendrites等15种架构在相同参数预算和计算预算下进行R^2比较；DendriCL在d≥30保持R^2>0.5且种子稳定，优于Dense Transformer的grokking崩溃，且能耗比Spikformer低约5-10倍。

**⚠️ 局限性**

仅验证合成ICL任务，未覆盖真实语言/视觉；对LIF非线性理论证明有限；仅至d=50，未探索更高维；能耗估计基于模型而非实测；对生物学假设尚待实验验证。

---

## 506. CheckRLM: Effective Knowledge-Thought Coherence Checking in Retrieval-Augmented Reasoning

**arXiv ID:** 2607.02262 | [PDF](https://arxiv.org/pdf/2607.02262v1)

**作者:** Dingling Xu `[一作]` (Beijing Normal University), Maosong Sun `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CheckRLM 框架，结合检索增强生成，在推理过程中实时识别并纠正事实错误

**💡 创新点**

创新性地将检索与推理耦合，采用知识声明识别与局部检索纠错，并用 DPO 训练提高纠错质量

**🔧 技术方法**

使用检索增强生成、知识声明识别、局部知识一致性纠错、DPO 训练、Llama-3.3-70B-Instruct/Qwen 等大语言模型以及 BM25/dense retriever

**📊 数据集**

在 HotpotQA、2WikiMultiHopQA、MuSiQue、IIRC 等多跳 QA 数据集以及 SimpleQA 短问答数据集上进行实验

**📈 对比分析**

与 Direct Reasoning、Vanilla RAG、RAT、FLARE、Self-RAG、ReAct、Search-o1 等方法对比，CheckRLM 在 F1/EM 指标上显著领先，且在 token 消耗与推理时间上更高效

**⚠️ 局限性**

目前仅基于文本知识库，未扩展至多模态验证，也未处理多源知识库的协同与冲突解决

---

## 507. AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents

**arXiv ID:** 2607.02255 | [PDF](https://arxiv.org/pdf/2607.02255v1)

**作者:** Xiangchen Cheng `[一作]` (Alaya Lab), Kaipeng Zhang `[通讯]` (Alaya Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Slay the Spire（堆叠式 roguelike）游戏中设计并实现了一种基于决策的五层 typed memory contract，能够在长周期 LLM agent 中替代传统的全文本上下文累积，实现可审计、可重用的记忆接口，并对其在固定难度 A0 与自动阶梯 Ascension ladder 的表现进行评估。

**💡 创新点**

创新点包括：①将记忆拆分为 L1（协议）、L2（状态 schema）、L3（游戏规则）、L4（回合总结）和 L5（触发技能）五层可独立 ablate 的结构；②在同一代码库中提供可直接与累积上下文设计对比的实验框架；③展示了通过启用 L5 技能层可在 A0 难度下实现显著性能提升，同时保持上下文规模固定。

**🔧 技术方法**

技术手段：Per-decision typed retrieval、协议/schema 组合、规则/经验/技能检索、四层路由（fast/strategic/analysis/evolution）、动态条件标签、Wilson CI、bootstrap 95% 置信区间、Fisher exact 统计检验。

**📊 数据集**

数据集：Slay the Spire 的完整规则集（576张卡、293遗物、115怪物等）以及 298 条完整游戏轨迹（覆盖固定 A0 的 5 种配置、跨 backbone 探测、自动阶梯 Ascension ladder），每条轨迹记录 Ascension 级别、结果、时间、LLM 调用次数及条件标签。

**📈 对比分析**

比较方法：在固定 A0 难度下进行 5 条配置的 ablation，对比跨 backbone（Gemini、Qwen、DeepSeek）和自动阶梯上升水平；与公开的 STS2MCP 与 CharTyr 进行同一游戏、同一角色、同一背后的 LLM 版本下的运行对比。性能结果显示：在 A0 下，启用 L5 技能层可将胜率从 3/10 提升到 6/10（方向性提升，样本量有限）；跨 backbone 的可迁移性不一致；自动阶梯可达 A6–A8；与累积上下文代理相比，bounded contract 在 A0 只需 1–2 个数量级更少的 token 和更快的运行时间即可获得显著更高的得分。

**⚠️ 局限性**

Limitations：①样本量仅为 50 条完整游戏（每配置 10 条），导致统计显著性不足；②仅单一角色（Silent）与单一游戏版本；③未在同一代码库下实现完整的累积上下文 variant，导致无法完全消除不同实现细节的干扰；④仅针对回合制文本游戏，无法直接推广到视觉或持续控制任务；⑤未评估持续学习、多人代理或跨游戏迁移等复杂场景。

---

## 508. Developers' Experience with Generative AI Beyond Productivity Assessment -- Insights from an Empirical Mixed-Methods Field Study

**arXiv ID:** 2607.02337 | [PDF](https://arxiv.org/pdf/2607.02337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 509. Efficient Waste Sorting for Circular Economy: A Confidence-guided comparison between One-Vs-All and One-Vs-Rest Classification Strategies with Human-in-the-Loop for Automated Waste Sorting

**arXiv ID:** 2607.02230 | [PDF](https://arxiv.org/pdf/2607.02230v1)

**作者:** Mohammed Fahad Ali `[一作]` (Clausthal University of Technology), Andreas Rausch `[通讯]` (Clausthal University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在德国Goslar市废物分类体系中比较了One‑vs‑All（OvA）与One‑vs‑Rest（OvR）两种多类别分类策略，并评估了通过置信度阈值筛选需要人工标注的样本，从而支持循环经济目标。

**💡 创新点**

创新点在于系统化比较OvA与OvR在整体准确率、误分类模式以及不确定样本识别效率上的差异；提出置信度引导阈值分组（高信、单投、多投、无投）来评估预测不确定性，并揭示OvR在低置信度误判识别方面的优势。

**🔧 技术方法**

技术包括基于InceptionV3的CNN模型，OvA采用单一多类别软max输出，OvR采用六个独立二分类sigmoid输出；同时使用置信度阈值分组对预测结果进行不确定性评估。

**📊 数据集**

数据集为扩展后的六类废弃物图像集（Organic、Paper、Plastic+Metal、Recycling Center、Glass、Residual Waste），由原始Goslar四类数据与TACO、TrashNet映射合成，最终包含18,515训练、2,000验证、2,000测试图像。

**📈 对比分析**

比较方法是在相同的数据划分与超参数条件下训练OvA与OvR，计算准确率、召回率、F1，并对误分类样本进行置信度分组分析。结果显示OvA略优于OvR（误分类49 vs 54），但OvR在识别低置信度误判（Group 3+4）上可捕获>50%误判，仅需标注<5%样本。

**⚠️ 局限性**

限制包括OvR需要训练六个二分类器，计算成本较高；两策略误判模式不同，仅有29个共同误判，未尝试集成模型；数据集主要为图像，缺乏真实现场多模态信息；置信度阈值未做动态优化。

---

## 510. SelectTSL: Prompt-Guided Selective Target Sound Localization in Complex Scenarios

**arXiv ID:** 2607.02343 | [PDF](https://arxiv.org/pdf/2607.02343v1)

**作者:** Ziyang Jiang `[一作]`, Haizhou Li `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了SelectTSL框架，实现了基于多模态提示（文本与音频）进行选择性目标声音定位，能够在多源、噪声环境下仅定位指定目标的方向。

**💡 创新点**

创新点在于：①结合Prompt-Guided Selective Attention (PGSA) 模块，将提示信息映射为目标感知嵌入，过滤无关源；②利用IPD Enhancer与目标感知嵌入共同提升空间相位差，生成更精准的方向后验；③通过可变目标数预测头，支持时间变化的目标数，实现连续轨迹跟踪。

**🔧 技术方法**

核心技术包括：多模态CLAP编码、FiLM条件融合、双路径RNN提取嵌入、交叉注意力、深度可分离卷积、门控FiLM、进化的IPD增强与多尺度特征融合，以及端到端联合训练。

**📊 数据集**

使用了合成数据（约289小时、22k+片段）和真实房间数据（TAU‑SRIR 9个房间共18k片段）进行评估，涵盖多源、噪声、移动源等多种场景。

**📈 对比分析**

与多类基线（IPDNet、EINV2、SELDT、FN‑SSL、SEL等）在静态与动态指标上对比，SelectTSL在MAE、Precision、F1、Recall、MOTA*、DetA和OSPA‑T等指标上均显著领先，尤其在混合场景下MAE仅0.98°、MOTA* 91.6%。

**⚠️ 局限性**

局限性包括：①仅支持双通道水平布局，未扩展到更大阵列或三维定位；②对快速运动或极端回声场的鲁棒性仍有限；③对缺失或不完整提示的依赖较高，提示质量直接影响性能。

---

## 511. HNSW with Accuracy Guarantees Using Graph Spanners -- A Technical Report

**arXiv ID:** 2607.02338 | [PDF](https://arxiv.org/pdf/2607.02338v1)

**作者:** Minghao Li `[一作]` (University of Toronto), Nick Koudas `[通讯]` (University of Toronto)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“Certify‑then‑Rectify”框架，先用无分布假设的统计认证检查HNSW检索结果质量；若低质量则通过把HNSW图视为几何支撑子图并利用极值理论估计最大拉伸因子，随后执行从查询点出发的收敛扩展（SBE‑Q）与基于三角不等式的Metric Bound Verification（MBV）来实现确切的k‑最近邻检索。

**💡 创新点**

创新点在于：① 将HNSW视作几何spanner并用极值理论自适应估计拉伸因子，得到可用的搜索半径；② 通过SBE‑Q消除近邻误差对搜索半径的影响，显著缩小扩展范围；③ 引入轻量级分布式统计认证（CRC/LTT）在查询时决定是否需要耗时的精确恢复；④ 将上述方法推广到滤波检索和磁盘级别的DiskANN，实现了端到端的准确性‑速度折衷。

**🔧 技术方法**

使用的技术包括：HNSW图结构、极值理论（Generalized Extreme Value分布）进行拉伸因子估计、几何spanner理论、SBE‑Q和MBV算法、Conformal Risk Control / Learn‑then‑Test统计认证、以及对过滤检索的F‑SBE‑Q扩展。

**📊 数据集**

实验数据集主要为：SIFT1M、GIST1M、DEEP1M（均为欧氏/余弦距离）、T2I‑10M、T2I‑100M（余弦距离），以及在滤波检索实验中使用的Synthetic predicates、ACORN、PAPER、LAION1M、arXiv等。

**📈 对比分析**

与传统HNSW、ConANN、DARTH等方法相比，CTR在保持与HNSW相近的平均查询时延的同时，在被认证为低质量的查询上可实现1.0精确召回；在大多数设置下，CTR的合规率（Recall ≥ τ）比HNSW高数个百分点，且在高召回阈值（τ≥0.95）下仍能保持大于90%的合规率；相比ConANN，CTR在低召回阈值下吞吐量相当或更优，在高召回阈值下吞吐量略低但合规率明显更高；在过滤检索场景下，CTR通过SBE‑Q + MBV实现完美召回，且相对于基准的运行时间增长在可接受范围内。

**⚠️ 局限性**

主要局限包括：① 需预先对HNSW图做拉伸因子估计，估计误差会影响搜索半径；② 当查询分布或图结构出现显著变化时，需要重新校准；③ 对图中存在孤立节点（degree=0）或连接度过低的情况假设不成立；④ 对于极高维、极稀疏或大规模数据集，SBE‑Q + MBV的搜索开销仍可能显著，导致高阈值下的延迟急剧上升；⑤ 统计认证的假设依赖于独立同分布查询，实际工作负载的非平稳性会削弱理论保证。

---

## 512. AI usage patterns are shaped by perceived gains in human agency

**arXiv ID:** 2607.02313 | [PDF](https://arxiv.org/pdf/2607.02313v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 513. Grounded autonomous research: a fault-tolerant LLM pipeline from corpus to manuscript in frontier computational physics

**arXiv ID:** 2607.02329 | [PDF](https://arxiv.org/pdf/2607.02329v1)

**作者:** Haonan Huang `[一作]` `[通讯]` (Princeton University), Haonan Huang (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了从论文语料库到自动生成完整科研论文的 LLM 自动化流水线，并在计算凝聚态物理前沿实现了基于锚点校准的研究方向选择、数值复现、模型计算与论文写作。

**💡 创新点**

首次将“文献锚点校准”与“多阶段故障容错”嵌入端到端自动研究管道，证明了在缺乏可靠锚点的前沿领域中可通过程序选择与 pilot 复现实现文献根植的自我验证。

**🔧 技术方法**

使用大型语言模型与对话式推理、分布式 grounding、fresh‑context isolation、对抗式审查、基于 Quantum ESPRESSO / Wannier90 的第一性原理计算、以及知识库与规则库。

**📊 数据集**

使用约 11,083 篇近年 condensed‑matter arXiv 论文语料库（共 2,162 次文献检索）以及若干公开的第一性原理参考结果。

**📈 对比分析**

通过与预设的“无 pilot”对照和“前架构基线”实验比较，显示缺失锚点校准时管道会产生无锚点论文，而加入 pilot 复现后错误率降为零，整体耗时约 6 天，输出稿件质量达到期刊提交级别。

**⚠️ 局限性**

受限于单一研究方向、缺乏跨方向验证、对锚点质量高度依赖、未能对锚点本身进行批判性审查，以及多次复现对计算资源要求较高。

---

## 514. Personality Without Persons? A Psychometric Critique of Big Five Testing in Large Language Models

**arXiv ID:** 2607.02325 | [PDF](https://arxiv.org/pdf/2607.02325v1)

**作者:** Kim Zierahn `[一作]` (ELLIS Alicante), Nuria Oliver `[通讯]` (ELLIS Alicante)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统性地评估了将人类设计的 Big Five 性格测评套用于大语言模型（LLM）的可行性，涵盖内容效度、模型间差异性以及潜在因子结构三大维度。

**💡 创新点**

创新点在于首次对 LL M 进行全流程心理测量检验，证实传统人类测评工具在 LLM 上缺乏效度，并提出 LLM 本土化测评框架的必要性。

**🔧 技术方法**

采用专家内容效度评估、问卷管理、Cronbach α 与 McDonald ω、确认性与探索性因子分析，以及线性混合效应模型等多种心理统计技术。

**📊 数据集**

使用 244 个跨 49 家族、不同规模、是否指令微调、开放/专有、地区来源多样的 LLM 进行 44 题 BFI‑LLM 问卷调查，产生 107,360 条有效响应。

**📈 对比分析**

通过与人类 BFI‑44 标准对比、基准模型与指令微调模型差异对照，并利用方差分解、CFA/TLI/CFI/RMSEA 等指标评估模型拟合，结果显示指令微调显著提升社会期望属性，整体差异性极低。

**⚠️ 局限性**

主要局限包括专家人数仅 3 人且其中 2 人为作者，样本仅限英文问卷，使用自评而非行为数据，模型间共性可能导致相关性偏差，且仅覆盖截至 2026 年初的公开 API 模型，未检验跨语言或视听模型。

---

## 515. Constrained Distributed Heterogeneous Two-Facility Location Problems with Max-Variant Cost

**arXiv ID:** 2607.02314 | [PDF](https://arxiv.org/pdf/2607.02314v1)

**作者:** Xinru Xu `[一作]` (Ocean University of China), Qizhi Fang `[通讯]` (Ocean University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了受候选地点集合限制、分布式决策和异质双设施的定位问题，在实线上的最大距离成本模型下设计无钱机制并证明其策略不诚实性与失真度量；

**💡 创新点**

创新点在于将分布式机制与异质设施结合，提出四种社会目标的失真下限与上限，并给出可实现的（α,β）-Quantile机制实现最优失真上界；

**🔧 技术方法**

采用分布式机制框架、策略不诚实证明、三角不等式与几何分析、参数优化（α,β）求取失真上界；

**📊 数据集**

本工作为理论研究，不涉及具体数据集；

**📈 对比分析**

通过理论证明得到失真下限与上限：平均-平均成本下限3、上限9；最大-最大成本下限3、上限3；最大-平均成本下限7/2、上限2+√5；平均-最大成本下限3、上限2+√5；

**⚠️ 局限性**

仅限两设施、实线、离散候选位置，未扩展到更一般度量空间或多设施，且上下界之间仍有剩余间隙，缺乏实验验证。

---

## 516. Securing People and their Machines Against Major Faults

**arXiv ID:** 2607.02304 | [PDF](https://arxiv.org/pdf/2607.02304v1)

**作者:** Ohad Eitan `[一作]` (Technion Israel Institute of Technology), Ehud Shapiro `[通讯]` (London School of Economics)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种面向草根平台的对等恢复框架，利用身份托管人和状态托管人，在用户私钥丢失或设备状态丢失时通过安全社会图恢复身份和状态，并将该框架扩展到草根加密货币的状态恢复。

**💡 创新点**

创新点在于将守望多代理原子事务与身份/状态恢复机制结合，给出完整的形式化模型、实现细节以及可恢复故障下的一致性证明；此外，采用基于身份托管人而非传统阈值加密的身份恢复方案。

**🔧 技术方法**

使用的技术包括守望多代理原子事务（guarded multiagent atomic transactions）、通信志愿代理（communicating volitional agents, CVA）以及最终同步消息传递模型，配合身份与状态托管人的设计实现安全社会图与草根币。

**📊 数据集**

论文未使用实际数据集，主要通过形式化模型与定理证明验证设计；实现部分通过模拟实验展示协议行为。

**📈 对比分析**

通过形式化证明展示实现与规范的一致性，证明在最终同步假设下运行达到quiescence并保持正确性；性能评估未给出数值，只说明在故障恢复后系统能恢复到一致状态。

**⚠️ 局限性**

局限性包括仅能处理崩溃/失钥/失机等可恢复故障；存在单一不可恢复的好友关系条件；攻击者持有私钥时只能恢复身份，无法撤销已完成的交易；对网络延迟的假设为最终同步，未覆盖高延迟或恶意中间人攻击场景。

---

## 517. NEUROSYMLAND: Neuro-Symbolic Landing-Site Assessment for Robust and Edge-Deployable UAV Autonomy

**arXiv ID:** 2607.02277 | [PDF](https://arxiv.org/pdf/2607.02277v1)

**作者:** Weixian Qian `[一作]` (Macquarie University), Xi Zheng `[通讯]` (Macquarie University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种神经符号化无人机着陆场地评估框架，将轻量化感知、概率语义场景图构建与完全符号化安全推理分离，支持边缘设备部署；

**💡 创新点**

创新点在于：①显式构造概率语义场景图（PSSG）以可审计的方式记录感知结果；②采用Scallop编译器执行离线生成的符号规则，实现无神经推理的确定性决策；③利用大语言模型仅在离线阶段协助规则编写，运行时不再调用LLM；

**🔧 技术方法**

技术包括：INT8量化的SegFormer‑B0分割网络+几何后处理；概率语义场景图构建；Scallop符号推理引擎；LLM（OpenAI GPT‑4）辅助规则生成；多帧验证与任务加权排序；

**📊 数据集**

使用Semantic Drone Dataset作为对象、属性和关系词汇表，生成PSSG；实验基于AirSim模拟的72个多样化着陆场景；

**📈 对比分析**

与SegOpticalFlow、SafeUAV、PEACE、LLMExplain以及DetFOL对照实验，成功率61/72（最高），多帧验证率与安全间隙均优于基线；硬件边缘评估（Jetson Orin Nano）显示1.04 s/帧、CPU 76.9%、GPU 52%、内存5.6 GB、功耗12.3 W，符号推理仅占1.9%；

**⚠️ 局限性**

主要局限在于感知质量决定世界模型的可靠性，分割误差导致符号推理误判；场景语义词汇表覆盖有限；对动态障碍物处理不足；未来工作聚焦提升感知可靠性、引入更多传感器与增量图构建。

---

## 518. Cadence: Extreme Pipelining with Multiple Concurrent Proposers

**arXiv ID:** 2607.02275 | [PDF](https://arxiv.org/pdf/2607.02275v1)

**作者:** Fatima Elsheimy `[一作]` (Category Labs), Jason Milionis `[通讯]` (Category Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种可扩展的多共识者（MCP）协议，利用极端流水线（每个时隙独立完成共识）实现块间隔可低于网络延迟，同时通过阈值加密与分片实现提议的并行传播与快速达成一致。该协议在任意同步条件下提供安全性、活性、短期审查抵抗与隐藏特性，并通过窗口调度实现槽数的上界，保证在网络失效时不会无限积累待处理槽。

**💡 创新点**

创新点包括：① 极端流水线框架，将每个时隙视为单次共识实例，去除块级连锁，使块间隔与网络延迟无关；② 多共识者设计，在每个时隙内并行提交提议，并通过阈值加密+分片实现提议的隐藏与短期审查抵抗；③ 三轮快速路径，支持离线共识者并在同步下实现三轮快速达成；④ 基于窗口的槽调度机制，实现可控的槽上限与自愈性。

**🔧 技术方法**

使用的技术主要包括：同步时钟划分时隙、阈值加密（每个时隙的加密共享）、Erasure Coding + Merkle Root 进行提议的可传播与可验证、签名聚合（BLS）实现投票压缩、分片式分发与并行投票、基于多投票的元块（meta‑block）形成与恢复、以及基于可达成共识的 ACS（异步一致性）协议用于窗口边界的共识。

**📊 数据集**

实验使用的是 Monad 主网的 200 名全球分布式验证者的网络时延模型，模拟五个共识者在每个槽内。该模型基于真实网络测量得到的延迟分布。

**📈 对比分析**

在模拟中，100 ms 的块间隔下，最终化延迟平均为 219 ms（167 ms 为推测性最终化），交易从提交到进入提议平均等待 50 ms。该性能相较于传统单领导者 HotStuff 等协议在块间隔与最终化延迟上都有显著提升；且在极低延迟需求的实时金融应用场景中，能提供更短的经济周期。

**⚠️ 局限性**

局限性主要包括：① 需要全局同步时钟，若时钟漂移或同步失效会影响槽调度；② 依赖阈值加密与分片的实现，若密钥管理或分片解码失败可能导致提议丢失；③ 方案在大规模网络下对窗口参数的选择与调度开销仍需进一步评估；④ 虽然提供隐藏与短期审查抵抗，但在极端攻击者能观察到部分提议的前缀时，隐藏效果受限。

---

## 519. When Token Compression Breaks: Structural Pruning vs. Token Reduction for Robust ViT Segmentation under High Compression

**arXiv ID:** 2607.02237 | [PDF](https://arxiv.org/pdf/2607.02237v1)

**作者:** Tien-Phat Nguyen `[一作]` (Singapore University of Technology and Design), Ngai-Man Cheung `[通讯]` (Singapore University of Technology and Design)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

统一基准评估 ViT 语义分割中的 token compression 与结构剪枝，并在高压缩条件下提出 prune‑then‑merge 组合方案。

**💡 创新点**

创新点在于：①对 token compression 与结构剪枝在清晰与受破坏输入下的鲁棒性进行系统对比；②提出在适度剪枝基础上进一步进行 token 合并的堆叠策略，在高压缩时实现更佳的准确性‑鲁棒性平衡。

**🔧 技术方法**

技术手段包括：ViT (DeiT‑B) 编码器 + 轻量 Transformer 解码器；token compression 方法 ToMe、ALGM、CTS；结构剪枝方法 NViT；使用 FLOPs、FPS 作为效率度量；采用 effective rank 诊断特征多样性。

**📊 数据集**

数据集：ADE20K、Cityscapes 及其对应的 16 种常见破坏（噪声、模糊、数字、天气）组成的 ADE20K‑C、Cityscapes‑C。

**📈 对比分析**

比较方式：在匹配 FLOPs 的基础上，分别在轻度、中度、激进三种压缩等级下评估 clean mIoU 与 corruption mIoU；结果显示：token compression 在轻度压缩可保持性能，激进压缩导致显著下降；结构剪枝下降更平滑，且在激进压缩下更稳健；prune‑then‑merge 在激进压缩时实现最佳准确‑鲁棒性权衡。

**⚠️ 局限性**

局限性：实验仅覆盖平坦 token ViT（DeiT‑B），未验证分层 Transformer；仅评估单轴压缩与堆叠组合，缺乏更系统的协同设计；FLOPs 与实际 FPS 的匹配并不完全；对不同解码器或其他任务（如检测）的适用性未知。

---

## 520. Learning to Evolve Scenes: Reasoning about Human Activities with Scene Graphs

**arXiv ID:** 2607.02425 | [PDF](https://arxiv.org/pdf/2607.02425v1)

**作者:** Francesca Pistilli `[一作]` (Politecnico di Torino), Giuseppe Averta `[通讯]` (Politecnico di Torino)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了可编辑的时空场景图SG‑Ego，并开发Graph‑Language Edit Network（GLEN）实现基于动作的图编辑预测（A‑GEF），以显式建模人类行为驱动的场景演化；

**💡 创新点**

创新点在于把第一人称视频的动态场景转化为可编辑的时空场景图，并引入基于动作的图编辑预测，使得情境演化可被透明、可控地推理；

**🔧 技术方法**

采用图神经网络（TripletGCN + cross‑attention）、图‑文本对齐（对比与匹配损失）、大型语言模型（Qwen3.5）、目标检测（GroundingDINO、SAM2）、以及图编辑网络（GEN）等技术；

**📊 数据集**

主要数据集为Ego4D的扩展SG‑Ego（3.8M时空场景图），以及EgoSchema、EgoMCQ、EgoCVR、EXPLORE‑Bench、A‑GEF等下游任务数据；

**📈 对比分析**

与视频‑语言基线相比，在EgoMCQ和EgoCVR检索任务上取得与端到端视频语言模型相当甚至更优的性能，在A‑GEF和EXPLORE‑Bench上实现了相对基线和LLM的显著提升；

**⚠️ 局限性**

局限性包括对复杂场景和长时延的推理仍有误差，依赖于自动化标注管线且未覆盖空间位置预测，且模型在实时部署和跨域通用性上待进一步验证。

---

## 521. Physical surfaces make touch interactions in virtual reality precise, efficient, and bimanual

**arXiv ID:** 2607.02430 | [PDF](https://arxiv.org/pdf/2607.02430v1)

**作者:** Wen Ying `[一作]` (University of Virginia), Seongkook Heo `[通讯]` (University of Virginia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究通过对比无触觉、触觉反馈（振动+压力）以及携带可触摸物理平面三种手势反馈方式，评估其对虚拟现实中选择、追踪与素描任务的精准度、速度及双手协调行为的影响。

**💡 创新点**

创新点在于首次系统验证携带可移动物理表面能显著提升高精度手势交互的表现，并揭示物理表面在双手协同中的协同优势；同时将物理表面与触觉反馈做细粒度对比，展示物理表面优于传统触觉手段。

**🔧 技术方法**

采用SenseGlove DK1手部跟踪与线性共振振动器（LRA）与舵机实现的压力反馈相结合的触觉方案；使用透明亚克力板配备电容触控与可跟踪标记的物理表面；结合OptiTrack全局跟踪与Unity虚拟环境渲染。

**📊 数据集**

共12名右手使用者完成183个实验试次（选择48、追踪10、素描3×3），另外30名MTurk评审对素描质量进行主观打分；所有数据记录于60Hz采样率的日志文件。

**📈 对比分析**

采用重复测量ANOVA、ART变换等统计方法，对TP、误差率、MPD、MFD、断点数、完成时间以及主观评估进行比较。结果显示，物理表面在选择精度、追踪精度与连贯度、素描平滑度与清晰度方面均优于触觉和裸手条件；触觉在速度上略优于裸手。

**⚠️ 局限性**

局限性包括样本量仅12人且为VR初学者，未检验专家绘图者；仅研究平面单点触控，未涉及多点或非平面表面；硬件重量与跟踪漂移可能影响疲劳；实验仅在坐姿下进行，未考察站立或移动情境。

---

## 522. LIME: Learning Intent-aware Camera Motion from Egocentric Video

**arXiv ID:** 2607.02417 | [PDF](https://arxiv.org/pdf/2607.02417v1)

**作者:** Boyang Sun `[一作]` (ETH Zurich), Hermann Blum `[通讯]` (University of Bonn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了LIME——一种语言条件下的摄像机运动生成模型，能够根据当前视觉观测和自由形式的自然语言意图，预测相对摄像机的SE(3)姿势，并通过从被动人类第一视角视频中挖掘的意图、观察收益描述和相机运动三元组进行训练，随后将模型迁移到机器人上实现主动感知；

**💡 创新点**

创新点主要包括①将意图与观察收益的语言描述与视觉运动耦合，形成观察收益接口；②使用连续流匹配（flow‑matching）头建模多模态的相对SE(3)姿势分布；③通过对人类视频的“事后”对齐（hindsight labeling）自动生成多意图监督；④搭建专门的意图条件相机运动基准，评估不同意图粒度下的性能；

**🔧 技术方法**

采用了基于VLM的视觉语言模型（如Qwen3‑VL‑4B‑Instruct）进行自回归语言生成，并在其隐藏层上接入连续流匹配头进行姿势预测；使用自回归观察收益生成器与流匹配损失共同训练；通过LoRA等轻量化适配实现快速迁移；

**📊 数据集**

训练数据来源于RoomTour3D与Nymeria两大 egocentric 视频库，用以挖掘约300万条意图‑收益‑姿势样本；基准评估使用InteriorGS渲染数据集；下游机器人实验使用Boston Dynamics Spot 的RGB‑D摄像机和ScanNet++场景；

**📈 对比分析**

与JanusVLN、Uni-NaVid、VG-AVS、VLMnav等基线在三类意图（Target‑approaching、Exploration、Perspective‑shift）下采用相同的6 m/600°预算多步评估，LIME 在成功率（SR）和碰撞敏感成功率（CA‑SR）上分别达到约45–51 % 和 39–47 %，显著优于所有基线；

**⚠️ 局限性**

局限性在于需要依赖大量人类 egocentric 视频并通过人工或脚本化的“事后”标注生成意图与收益标签；模型主要关注单步视图转移，尚未充分解决长期多步规划与复杂动态环境下的连续动作序列；

---

## 523. Show Me Examples: Inferring Visual Concepts from Image Sets

**arXiv ID:** 2607.02402 | [PDF](https://arxiv.org/pdf/2607.02402v1)

**作者:** Nick Stracke `[一作]` (CompVis), Björn Ommer `[通讯]` (CompVis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了视觉概念推理任务VICIS，要求模型从少量示例图像中推断共享概念并将其应用于查询图像生成新图像。

**💡 创新点**

构建可规模化的弱监督训练框架，利用WordNet层级和合成数据自动生成训练样本；提出集学习器+实例化模块+条件扩散模型的端到端架构，可在无文本输入下进行视觉上下文推理。

**🔧 技术方法**

使用Vision Transformer编码器、专门的Set Learner ViT、概念方向投影、条件扩散模型（flow matching目标），以及在合成和ImageNet/WordNet数据上的训练。

**📊 数据集**

合成数据集（控制形状、颜色等概念）和基于ImageNet/WordNet层级的真实数据集（Animal子树、Sketch、ImageNet21k）。

**📈 对比分析**

与BAGEL、ILLUME+、Visual Prompting等通用VLM及特定方法对比；在Accuracy、Diversity指标上，本文模型在验证集上准确率约46.3%、多样性得分0.81，显著优于基线与现有VLM；在Sketch、未见类别等迁移测试中仍保持高准确率并优于Visual Prompting。

**⚠️ 局限性**

对模型的局限性：在高噪声或少量上下文图像时性能下降；对多层次概念推理的依赖可能导致层级不一致时的误差；目前仍以ImageNet为主，跨域迁移需要进一步探索。

---

## 524. From Ham-Sandwich to Centerpoints: Semialgebraic Algorithms for Cutting Polytopal Measures

**arXiv ID:** 2607.02400 | [PDF](https://arxiv.org/pdf/2607.02400v1)

**作者:** Marie-Charlotte Brandenburg `[一作]`, Chiara Meroni `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了针对多维多边形测度的 ham‑sandwich 切割与中心点的精确算法，能够计算出所有满足给定比例的切割平面。

**💡 创新点**

创新点在于发现 cap‑volume 函数是分块有理函数，利用此性质将切割问题转化为半代数可行性问题，进而得到多项式时间算法，并将该框架推广至中心超平面截断与中心点，给出中心点与浮动体的等价半代数描述。

**🔧 技术方法**

主要技术包括实代数几何中的量化消除、层析分解（CAD）、临界点方法以及对多面体测度的分块有理化。

**📊 数据集**

输入数据为由有限个全维凸多面体的顶点描述构成的 rational polytopal measures，实验示例包括三角形、立方体、三维多面体等。

**📈 对比分析**

与已有的随机近似或点集算法相比，该方法在固定维度下实现了多项式时间的精确解，能够输出半代数描述和样本点；但对高维情形的实际运行时间仍受限。

**⚠️ 局限性**

局限性在于只适用于固定维度和多边形测度；对维度增长的复杂度呈指数，且对多平面或非多面体测度的计算尚无高效解法。

---

## 525. Ultra-Low-Cost Hybrid Beamforming: A New Static-Connection Architecture with Sparse Phase-Shifter Sharing

**arXiv ID:** 2607.02393 | [PDF](https://arxiv.org/pdf/2607.02393v1)

**作者:** Honghao Wang `[一作]` (Shanghai Jiao Tong University), Derrick Wing Kwan Ng `[通讯]` (University of New South Wales)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种稀疏相位移器共享的静态连线混合波束成形架构，显著降低高频大天线系统的硬件成本。

**💡 创新点**

核心创新在于将相位移器共享的连线拓扑视为离散优化问题，利用离散约束与连续波束成形协同优化，达到几乎完整的波束性能。

**🔧 技术方法**

采用离散二进制优化、混合整数凸规划、MM算法、上下界近似、双向链路对偶以及多用户MISO LoS信道模型。

**📊 数据集**

实验使用LoS信道模型，随机生成用户方向、距离、信噪比等参数，仿真在MATLAB实现。

**📈 对比分析**

通过与全PS、随机连接、循环/相邻连线等基线比较，结果显示在单RF链系统PS数减少37.5%时功率提升约1dB，在多RF链系统PS数减少62.5%时仅高于全PS约0.5dB，优于其他方案。

**⚠️ 局限性**

局限在于对极端宽频或多路径环境下的性能未验证，且算法求解复杂度仍随PS数和用户数显著增加。

---

## 526. Know Your Source: A Public Knowledge Store for Media Background Checks

**arXiv ID:** 2607.02383 | [PDF](https://arxiv.org/pdf/2607.02383v1)

**作者:** Benjamin Nichols `[一作]` (Cardiff University), Nedjma Ousidhoum `[通讯]` (Cardiff University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个公开可复现的知识库（MediaRef），用于支持媒体背景检查（MBC）的生成，并评估多种大型语言模型（LLM）在该任务中的表现。

**💡 创新点**

创新点在于：①提供独立于付费搜索API的低成本、可更新的文档集合；②设计可复现的检索+证据抽取+迭代生成流程；③提出针对MBC的四维人类评估框架（清晰度、相关性、信息量、可验证性）。

**🔧 技术方法**

技术手段包括检索增强生成（RAG）架构、BM25检索、DeBERTa问答抽取证据、LLM（OpenAI GPT‑4/3.5、Qwen、Llama、Mistral、Anthropic）交互式生成与更新。

**📊 数据集**

使用的数据集为：①Media Bias/Fact‑Check（MB/FC）提供的200个新闻源的金标准MBC；②由Google Search API检索并爬取的21,921篇文档，构成MediaRef知识库。

**📈 对比分析**

方法比较：用ROUGE‑L、METEOR、Fact Recall和Error Rate进行自动评测；用专家标注的四维人类评估进行质量评估。实验显示，加入检索后Fact Recall提升显著，但错误率变化不大；在自动指标上，GPT‑4表现最佳，后继为Anthropic/LLama；人类评估显示所有模型均能保持清晰度和相关性，但信息量与可验证性仍低。

**⚠️ 局限性**

局限性包括：检索依赖搜索引擎排名可能产生偏差；黑名单无法覆盖所有低质量来源；仅包含公开网页，无法访问付费或受限数据库；生成结果仍可能包含不准确信息，需人工监督。

---

## 527. Hardware-Enforced Semantic Coordination for Safety-Critical Real-Time Autonomous Systems

**arXiv ID:** 2607.02376 | [PDF](https://arxiv.org/pdf/2607.02376v1)

**作者:** Uwe M. Borghoff `[一作]` (University of Bundeswehr Munich), Remo Pareschi `[通讯]` (University of Molise)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出一种将TB‑CSPN协调语义直接映射到FPGA的硬件层，以实现安全关键实时自主系统的确定性语义协调；并给出了分层架构，将自适应推理、硬件级协调与物理安全层分离。

**💡 创新点**

创新点包括：①将同步、时限、授权等协调规则硬件化，提供可编程的确定性执行；②使用紧凑的语义Token只携带必要元数据，保持软件推理灵活性；③引入公平性、动态重配置与硬件层授权屏障等机制；④提出将TB‑CSPN与FPGA结合的全新架构视角。

**🔧 技术方法**

采用的技术包括：主题基通信空间Petri网（TB‑CSPN）理论；FPGA硬件加速与可重配置；时间门控、同步条件、授权屏障的硬件实现；语义Token压缩与路由。

**📊 数据集**

本文未使用任何实验数据集，而是以理论建模与架构描述为主。

**📈 对比分析**

由于缺乏实验实现，本文并未给出具体的性能数值；其比较方法主要是与传统软件中介式协调框架对比，指出后者在时延、确定性与安全性保障方面的局限。

**⚠️ 局限性**

限制包括：①缺乏完整的FPGA实现与硬件验证；②动态重配置与公平性权衡仍是未解决的技术挑战；③对硬件资源和可编程性要求较高，限制了在资源受限设备上的直接应用；④仅聚焦协调语义的硬件化，未涉及完整系统的集成与部署。

---

## 528. Understanding Agent-Based Patching of Compiler Missed Optimizations

**arXiv ID:** 2607.02370 | [PDF](https://arxiv.org/pdf/2607.02370v1)

**作者:** Batu Guan `[一作]` (Chinese University of Hong Kong), Shaohua Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性研究了利用大语言模型驱动的编码代理在LLVM编译器中修补被漏掉的优化缺陷，并提出通过检索与蒸馏历史优化经验来提升代理的泛化能力；

**💡 创新点**

创新点在于首次将编译器缺陷修补与“范围对齐”概念结合，构建了面向优化范围的评估框架，并展示历史知识（检索增强与知识蒸馏）能显著提升代理生成的补丁与开发者目标的契合度；

**🔧 技术方法**

采用大语言模型代理（GPT‑5.5、DeepSeek‑V4‑Pro、Qwen3.5‑Plus、Kimi K2.5）结合ReAct循环、工具接口与验证回调；通过检索增强生成（RAG）与历史优化PR的蒸馏知识，提供检索实例与通用原则；使用Alive2、llvm‑mca、fuzz测试进行验证；

**📊 数据集**

使用43条已验证的LLVM缺陷案例（由GitHub Issues提取并配备金手指补丁与测试），869条历史PR用于知识构建；同时在LLVM Opt Benchmark（8个主流项目）上评估实际IR优化命中；

**📈 对比分析**

对比方法：将代理补丁与金手指补丁的优化范围通过金手指测试与随机fuzz测试进行判定，归类为A⊂G、A⋈G、G⊂A、A≈G；在未使用泛化指令时，成功率约为74%，但只有约一半能完全匹配范围；加入泛化指令未能显著提升；引入历史知识后，A≈G与覆盖范围的数量显著提升，且在8个真实项目上历史知识增强的补丁在优化命中率上表现出比基线更高的胜率；

**⚠️ 局限性**

局限性包括：① benchmark规模有限，可能不完全代表所有优化缺陷；②泛化指令效果不稳定，可能导致过度/不足泛化；③依赖历史PR，若训练数据已泄漏可能影响结果；④评估仅基于测试覆盖与fuzz，无法完全证明语义正确；⑤实验结果受LLM随机性与工具调用顺序影响，需更多重复验证。

---

## 529. World Wide Models: Literary Tools for Cultural AI

**arXiv ID:** 2607.02369 | [PDF](https://arxiv.org/pdf/2607.02369v1)

**作者:** Nina Begus `[一作]` `[通讯]` (University of California, Berkeley), Nina Begus (University of California, Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出将文学批评与AI技术融合的层级框架，识别并实验AI文本的结构单语化问题，使用FinneGAN/FinneganLM等模型检视潜在空间与文化偏差。

**💡 创新点**

首次将结构单语化概念与世界文学视角引入LLM评估，利用文学方法揭示tokenization与潜在空间对文化输出的影响，开创跨学科的文化AI评估路径。

**🔧 技术方法**

使用大型语言模型（如GPT系列）与生成对抗网络（GAN）进行语音文本建模、潜在空间分析，并结合结构主义/后结构主义的批评方法进行理论解读。

**📊 数据集**

Finnegans Wake文本与对应音频、英文及多语种公开语料库（Common Crawl、Wikipedia等）、人工标注的文学文本与跨语言对照集。

**📈 对比分析**

通过定量指标（BLEU、ROUGE）与定性文化偏见分析对比FinneGAN与传统LLM，发现传统LLM在语义一致性上更强，但在文化多样性与潜在空间表达上表现更弱。

**⚠️ 局限性**

受限于结构单语化导致的文化偏见、tokenization导致的语言抽象、低资源语言数据缺失，以及实验规模和方法评估范围有限，难以全面解决跨文化一致性问题。

---

## 530. SkillFuzz: Fuzzing Skill Composition for Implicit Intents Discovery in Open Skill Marketplaces

**arXiv ID:** 2607.02345 | [PDF](https://arxiv.org/pdf/2607.02345v1)

**作者:** Jinwei Hu `[一作]` (University of Liverpool), Xiaowei Huang `[通讯]` (University of Liverpool)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 SkillFuzz，一种无需实际执行即可通过计划漂移差分算子对 LLM 语言模型技能市场中的隐式意图进行模糊测试的方法。

**💡 创新点**

创新点在于：将隐式意图检测框架化为模糊测试问题；首次使用计划层的文本漂移作为差分判别器；通过从技能说明中抽取结构化合同并嵌入语义空间，构造契约引导的 Monte Carlo 树搜索；定义了结合严重性与新颖度的 ICQ 指标，实现了在有限查询预算下高效发现多样化、严重的隐式意图。

**🔧 技术方法**

使用的技术包括：计划-先执行（plan‑then‑act）模型；计划漂移（plan drift）差分算子；基于 LLM 的技能合同抽取与嵌入；契约驱动的 MCTS 搜索；ICQ 评估指标；以及多模型对比实验（Open‑weight 与商用 LLM）。

**📊 数据集**

数据集为 SkillsBench（196 个社区贡献的技能）和 10 个代表性任务；实验使用多种规划器（DS‑R1 系列、GPT‑4.1、GPT‑5 等），并在每个任务上采用 B=200 或 B=1000 的查询预算。

**📈 对比分析**

与随机采样、无契约、贪婪漂移、贪婪覆盖、以及 MCTS+正交探索等对比方法相比，SkillFuzz 在累计 ICQ、严重性意图数量（高达 116 条）以及覆盖多样性方面均占优；在执行验证中，高达 80% 的高风险组合被确认；总体而言，SkillFuzz 在固定预算下实现了更高的发现率和更严谨的风险定位。

**⚠️ 局限性**

局限性包括：依赖计划层文本的漂移判别，可能无法检测执行阶段会产生不同意图的“欺骗性”代理；对较大或更具对抗性的技能市场的适用性尚未验证；阈值（δ_min、δ_sev、θ）为统一设置，实际部署时需重新校准；以及判别器和意图提取器使用 GPT‑4o-mini，可能存在模型偏差。

---

## 531. Automated grading of Linux/bash examinations using large language models: a four-level cognitive taxonomy approach

**arXiv ID:** 2607.02432 | [PDF](https://arxiv.org/pdf/2607.02432v1)

**作者:** Manuel Alonso-Carracedo `[一作]` (Universidade de Vigo), Lorena Otero-Cerdeira `[通讯]` (Universidade de Vigo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了四大前沿大语言模型（GPT‑5.2、Claude Opus‑4.6、Gemini‑3.0 Pro、GLM‑5）在自动评分 Linux/bash 命令考试中的表现，探索了基于四层认知分类与操作影响的税onomic框架及结构化 rubric 提示的效果。

**💡 创新点**

① 将 Bloom 认知层级与操作影响融合成四层 CogTax，提供可量化的难度预测；② 在同一数据集下比较两种提示（基础与 rubric‑增强）对模型评分一致性的显著提升；③ 系统使用多维度指标（ICC、Pearson、MAE、Bland‑Altman、Weighted‑Kappa）评估人机一致性。

**🔧 技术方法**

采用四种主流 LLM 并对比两种提示策略；利用人类三位专家评分作为黄金标准；通过统计学工具（ICC(3,1)、Pearson/Spearman、MAE、Bland‑Altman、Weighted‑Kappa）对模型评分与人类评分进行量化比较。

**📊 数据集**

来自一门计算机工程二年级操作系统课程的 1200 条学生答题记录（Linux/bash 命令）作为实验数据集，覆盖四层认知复杂度。

**📈 对比分析**

在多维度指标上，Gemini‑3.0 Pro 在 rubric‑增强版（V2）下获得最高一致性（ICC 0.888、MAE 0.10），GPT‑5.2 与 Claude V2 也表现优于 V1；随着认知层级升高，所有模型的一致性显著下降，说明模型在高难度任务上的局限。

**⚠️ 局限性**

① LLM 仍无法完全捕捉教师在评分时的隐性经验与上下文判断，导致在多任务序列中误判等；② 需要更细粒度的 rubric 与交互式提示；③ 对最高难度（L3、L4）问题的自动评分仍需人工复核；④ 仅测试两种提示与默认参数，未探讨其他 prompt 设计或温度等设置。

---

## 532. EvoPolicyGym: Evaluating Autonomous Policy Evolution in Interactive Environments

**arXiv ID:** 2607.02440 | [PDF](https://arxiv.org/pdf/2607.02440v1)

**作者:** Zhilin Wang `[一作]` (University of Science and Technology of China), Yang Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Autonomous Policy Evolution 框架，在固定交互预算下让编码代理反复编辑可执行策略，并在 EvoPolicyGym 基准上评估其性能。

**💡 创新点**

创新点在于将自主策略改进视为可控的、基于交互预算的评估任务，并提供轨迹层面诊断工具，用以分离结构合成与参数微调。

**🔧 技术方法**

利用语言模型与工具 harness（Codex、Claude Code）进行代码编辑，结合 Gymnasium 环境接口、训练/验证/隐藏分割与预算限制，实现持续反馈循环。

**📊 数据集**

使用 Core16 组 16 个 RL 环境（Gym/Box2D、MuJoCo、MiniGrid、机器人/驾驶），并在 GitHub、Hugging Face 上公开。

**📈 对比分析**

通过隐藏验证选取的 held‑out 回报进行排行榜比较，GPT‑5.5 以 0.891 的聚合排名居首，Claude Opus 4.7 仅次；相比单一任务表现，GPT‑5.5 在所有环境均为前两名。

**⚠️ 局限性**

局限在于诊断仅基于 AST 结构和参数差异，未捕获语义相似性；预算仅 128 轮，难以与传统 RL 对比；仅评估语言模型 harness，未涵盖非代码驱动的改进方法。

---

## 533. Steerability via constraints: a substrate for scalable oversight of coding agents

**arXiv ID:** 2607.02389 | [PDF](https://arxiv.org/pdf/2607.02389v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 534. WorldSample: Closed-loop Real-robot RL with World Modelling

**arXiv ID:** 2607.02431 | [PDF](https://arxiv.org/pdf/2607.02431v1)

**作者:** Yuquan Xue `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 WorldSample 框架，结合真实机器人 roll‑out 与基于 world‑model 的合成转移，形成 real‑synthetic 循环来提升在线实机强化学习的效率与成功率。

**💡 创新点**

核心创新在于：① 通过局部扰动的 counterfactual 轨迹生成保持物理可行性；② 引入 Policy‑Paced Learning (PPL)，通过 Q‑aware 样本选择与基于策略熵的噪声调度双重控制，避免价值过估与视觉幻象噪声；③ 实现异步世界模型更新与数据生成，保障实时控制。

**🔧 技术方法**

主要技术包括：action‑conditioned 视频世界模型（Cosmos‑Predict2.5）、基于演示与在线 roll‑out 的后期微调、异步生成与调度框架、PPL 中的 Q‑aware 采样与 uncertainty‑guided 调度。

**📊 数据集**

使用 Galaxea A1X 机器人在五类真实场景任务（Pushing、Insertion、Sorting、Pick & Place、Assemble）进行实验，采集 20 条人工演示并进行在线 roll‑out，构成真实数据集。

**📈 对比分析**

与 HIL‑SERL、VLAW、WMPO 等基线相比，WorldSample 在所有任务上提升了平均成功率 28%，将训练步数降低 59%，训练时间下降 23%，并在 world‑model 视觉质量上 PSNR 提升 19.4dB、SSIM 提升 0.47。

**⚠️ 局限性**

目前框架仅针对单一任务与固定场景进行自适应，缺乏跨任务/跨场景的泛化能力，需要进一步扩展到多实例、多任务或动态环境的适用性。

---

## 535. WattGPU: Predicting Inference Power and Latency on Unseen GPUs and LLMs

**arXiv ID:** 2607.02391 | [PDF](https://arxiv.org/pdf/2607.02391v1)

**作者:** Mauricio Fadel Argerich `[一作]` (Universidad Politécnica de Madrid), Marta Patiño-Martínez `[通讯]` (Universidad Politécnica de Madrid)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了两种基于公开元数据的预测模型，用于估算LLM推理时GPU的平均功耗和交叉标记延迟（ITL），无需对目标硬件和模型进行实际测功或 profiling；

**💡 创新点**

首次在未见 GPU 上验证能量与延迟预测，并通过仅使用公开的 GPU 规格与 LLM 结构信息，实现对新 GPU 与新 LLM 的泛化；

**🔧 技术方法**

采用梯度提升回归树（XGBoost）进行回归，特征工程基于 GPU TDP、频率、显存、带宽、模型参数、层数等；

**📊 数据集**

使用公开的 Watt Counts 数据集，包含 42 个密集型 LLM（0.1B–27B 参数）和 8 台 NVIDIA 服务器级 GPU；

**📈 对比分析**

与传统 TDP/Load‑Scaled TDP 以及基于 Roofline 的 ITL 预测作为基线比较；在离线模式下，功耗模型 MdAPE ≤3.4%，在服务器模式下 ≤13.5%；ITL 模型在服务器模式下 MdAPE ≤8.5%，离线模式下为 24.9%；同时在 GPU 排名上 Kendall τ ≥0.76；

**⚠️ 局限性**

主要局限包括：离线 ITL 预测误差较大、仅覆盖单 GPU 单模型，未考虑 Mixture‑of‑Experts、量化模型和多 GPU 部署；模型对工作负载动态特征（如 prompt 长度、KV 缓存使用）依赖不足，需进一步改进。

---

## 536. DecompRL: Solving Harder Problems by Learning Modular Code Generation

**arXiv ID:** 2607.02390 | [PDF](https://arxiv.org/pdf/2607.02390v1)

**作者:** Juliette Decugis `[一作]` (FAIR at Meta), Gabriel Synnaeve `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 DecompRL 的强化学习框架，通过将代码生成任务分解成可独立实现的模块，并利用重组合策略在 CPU 上评估大量候选程序，从而在不增加 GPU 计算量的前提下显著提升大规模推理时的成功率。

**💡 创新点**

核心创新包括：① 端到端训练分解策略和实现策略的两阶段协作 RL；② 在层次化推理下利用重组合产生指数级候选方案；③ 采用留一基线和 β 软目标来平衡探索与利用，并解决多样本奖励稀疏性；④ 将 GPU 生成成本转移至 CPU 评估，显著降低推理成本。

**🔧 技术方法**

使用的技术主要有：大语言模型（Qwen 2.5 7B、Llama 3.1 8B、Code World Model 32B）、PPO 强化学习、层次化采样与重组合、留一基线和 log‑mean‑exp（β 软目标）奖励聚合、离线分解/实现策略两阶段训练、CPU 集群并行评估。

**📊 数据集**

训练数据：约 15,000 个竞赛编程问题（CodeContest、TACO）；评估数据：DeepMind Code Contest 验证集 117 题和 LiveCodeBench 279 题。

**📈 对比分析**

与基线（GRPO、pass@k 训练、SPO、留一 β=0.3 等）以及 48 样本的强化学习方法进行对比。结果显示：在同等 token 预算下，DecompRL 的 pass@k 率高于所有基线；在 1,000–500,000 token 预算内，DecompRL 的 solve 率显著提升，最高可达 48%（比标准 16‑样本采样提升 18%）。GPU 生成量减少约 50×，推理时间主要由 CPU 评估占据。

**⚠️ 局限性**

局限性：① 由于分解带来的格式税，单样本 pass@1 性能下降（约 3%）；② 对易题和低 token 预算时，分解开销超过收益；③ 分解规模会因奖励削弱而收缩，导致模型倾向于单函数生成；④ 训练中需要大量 CPU 评估，导致训练成本上升；⑤ 在离线数据不包含足够模块化解时，RL 需要克服离策略难题。

---

## 537. HTTP REST API Structure Learning

**arXiv ID:** 2607.02442 | [PDF](https://arxiv.org/pdf/2607.02442v1)

**作者:** Ran Dubin `[一作]` (Ariel University), Amit Dvir `[通讯]` (Ariel University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种无监督的 HTTP REST API 学习方法 HRAL，通过网络流量学习 API 端点结构，并基于此检测异常请求。

**💡 创新点**

创新点在于：①不依赖完整 OpenAPI 文档，能够从零开始重建端点结构；②将聚类 + Speculator 结合，精确识别路径参数；③与签名规则（如 OWASP ModSecurity CRS）融合，实现 100% 的攻击检测。

**🔧 技术方法**

使用的技术包括 Porter Stemmer、Count Vectorizer、Ward 链接的 Agglomerative Clustering、Speculator、无监督异常检测算法，以及 OWASP ModSecurity CRS 等签名规则。

**📊 数据集**

使用 ATRDF（API Traffic Research Dataset Framework）数据集，该数据集包含 18 个 API 端点、7 种攻击类型（共约 217k 条请求/响应）。

**📈 对比分析**

评估方法：比较三层 OpenAPI 文档（Minimal、Basic、Full）、Speculator 与 HRAL，指标为召回率和 F1 分数。HRAL 在无文档场景下召回率 82.07%，F1 分数 87.24%，仅次于 Full 文档；与签名规则结合后检测率达到 100%。

**⚠️ 局限性**

局限性包括：聚类计算量大，难以实时部署；对 API 变更不够敏感，需要重新聚类；缺乏可解释性，难以阐明异常原因；无法覆盖请求体内的攻击。

---

## 538. Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data

**arXiv ID:** 2607.02447 | [PDF](https://arxiv.org/pdf/2607.02447v1)

**作者:** Xuanyu Chen `[一作]` (University of Sydney), Dong Yuan `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究分布式自监督学习在非IID数据下的鲁棒性，并提出 MAR 损失改进 MIM 的鲁棒性

**💡 创新点**

理论证明 MIM 在数据异质性下比 CL 更鲁棒，并揭示网络连通度对分布式 SSL 鲁棒性的影响；提出 MAR 损失对齐局部与全局表征以进一步提升鲁棒性

**🔧 技术方法**

使用理论分析（表示可表示性向量）、SimSiam、MAE、FedAvg、D-PSGD、MMD 对齐以及余弦调度等技术

**📊 数据集**

使用 Mini-ImageNet 进行预训练，CIFAR-10/100 与 ImageNet 进行 fine‑tune，利用 Dirichlet 分布模拟非IID，Erdős‑Rényi 生成 DecL 网络

**📈 对比分析**

与传统 CL/MIM、FedU、FedEMA、Orchestra 等 SOTA 进行对比；MAR 在 FL 与 DecL 下均提升 10–15% 精度，尤其在高异质性场景显著表现

**⚠️ 局限性**

仅在模拟网络中验证，未考虑通信成本与安全细节，理论仅适用于线性模型与简化非IID设定

---

## 539. Fast Multi-dimensional Refusal Subspaces via RFM-AGOP

**arXiv ID:** 2607.02396 | [PDF](https://arxiv.org/pdf/2607.02396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 540. Data Comics for Education: Evaluating Effectiveness, Benefits, and the Ethics of AI-Assisted Creation

**arXiv ID:** 2607.02361 | [PDF](https://arxiv.org/pdf/2607.02361v1)

**作者:** Zirui Shan `[一作]` (Monash University), Roberto Martinez-Maldonado `[通讯]` (Monash University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了使用生成式人工智能辅助制作的数据漫画（Data Comics）对大学生在信息检索和理解任务中的数据可视化理解效果，并与传统可视化进行对比；

**💡 创新点**

首次实证评估了GenAI（如DALL‑E 3）生成漫画对学习任务的影响，并探讨可视化素养与AI伦理问题的交互；

**🔧 技术方法**

使用ChatGPT‑4生成文本提示、DALL‑E 3生成图像，并通过Canva拼贴完成漫画；采用Mini‑VLAT测试评估可视化素养；

**📊 数据集**

使用四组基于气候变化的可视化数据（来自Kaggle及相关研究），以构建实验材料；

**📈 对比分析**

采用 within‑subjects 60名学生进行实验，比较正确率；实验结果显示，数据漫画在检索与理解任务中显著优于传统可视化（中位数正确率 0.75 对 0.33），尤其在多洞见理解任务上提升最大；

**⚠️ 局限性**

局限包括样本规模相对有限、仅评估四组可视化、仅覆盖 Bloom 的前两层任务、漫画制作仍需人工介入、未检验长期学习效果与更复杂认知层级的适用性。

---

## 541. MARVEL: Margin-Aware Robust von Mises-Fischer Expert Learning for Long-Tailed Out-of-Distribution Detection

**arXiv ID:** 2607.02435 | [PDF](https://arxiv.org/pdf/2607.02435v1)

**作者:** A. S. Anudeep `[一作]`, Vaanathi Sundaresan `[通讯]` (Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对医疗影像中长尾类别分布下的异常样本检测，提出了一种端到端的框架，实现了对未知病例的可靠识别与医生推送；

**💡 创新点**

①引入非线性 von Mises–Fisher（NvMF）分类器，可产生非线性决策边界；②设计基于类别频率的三种 Margin‑Aware 专家网络，分别聚焦头、中、尾类；③增设独立的 OOD 专家，用二分类器明确分离 ID 与 OOD；

**🔧 技术方法**

非线性 vMF 分类、指数族参数化、margin‑aware 多专家集成、辅助 OOD 训练、Softmax/OOD 得分组合、Adam + cosine LR、ResNet‑18 特征提取；

**📊 数据集**

三大医学影像数据集：RFMiD（视网膜）、ISIC2019（皮肤）、NCT‑CRC（组织病理），以及多层级 OOD 数据集（近 OOD、远 OOD、损伤、域移位、自然图像等）与 ImageNet‑100 作为辅助 OOD 训练集；

**📈 对比分析**

与 PASCL、COCL、EAT、PATT、OE 等最新长尾/OOD 方法在 7 个随机种子上对比；在三组数据集的 6 类 OOD 场景（Open‑Set、NearOOD1/2、Corruption、FarOOD）下，平均 AUROC、AUPR、FPR95 均优于对照组，尤其在远 OOD 和尾类上提升显著（如 FPR95 减少 8.45%–36.90%）；

**⚠️ 局限性**

依赖辅助 OOD 数据集（ImageNet‑100）的多样性；对极端稀缺类别或多模态/层级标签的处理仍有限；在开放集、模态迁移等更复杂的真实临床情形中性能尚待验证。

---

## 542. ACID: Action Consistency via Inverse Dynamics for Planning with World Models

**arXiv ID:** 2607.02403 | [PDF](https://arxiv.org/pdf/2607.02403v1)

**作者:** Gawon Seo `[一作]` (POSTECH), Suha Kwak `[通讯]` (POSTECH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在决策时规划中加入循环动作一致性（ACID）成本，对动作与逆动力学模型预测的动作做一致性检查，提升可执行性。

**💡 创新点**

创新点是将逆动力学模型用作规划时的实时验证器，将每步动作一致性误差作为可执行性成本，并采用尺度不变自适应权重平衡终点目标成本与一致性成本。

**🔧 技术方法**

采用基于隐空间的动作条件世界模型（LE-WM、PLDM、DINO-WM）、视频生成模型（NWM）、逆动力学模型（IDM）与采样优化器（CEM）相结合。

**📊 数据集**

使用六个任务数据集：Cube、Reacher、Push‑T、Rope、Granular 以及 RECON 视觉导航环境。

**📈 对比分析**

与仅使用终点目标成本的基线相比，在所有四种世界模型和六项任务中均实现了成功率提升（Cube、Reacher、Push‑T）或 Chamfer 距离下降（Rope、Granular、导航），且在大多数情况下达到相同或更优性能所需的规划计算量更低。

**⚠️ 局限性**

局限性包括依赖完整观测与单一动作驱动的转移，在部分可观测或外部干扰条件下一致性检验可能失效；逆动力学模型需额外训练，虽然一次性完成但仍增加工作量。

---

## 543. HULAT2 at MER-TRANS 2026: Governed Multi-Agent Simplification for Spanish Easy-to-Read Generation

**arXiv ID:** 2607.02381 | [PDF](https://arxiv.org/pdf/2607.02381v1)

**作者:** Lourdes Moreno `[一作]` (Universidad Carlos III de Madrid), Miguel Domínguez-Gómez `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提交了三套全自动的西班牙语E2R生成系统，包括基于LangGraph多智能体的工作流和线性生成‑评估‑再生成基线。

**💡 创新点**

创新点在于将内部质量信号与ECA路由相结合，实现可追踪的多智能体控制，并验证了词汇支持对参考指标的非线性影响。

**🔧 技术方法**

采用Gemini 2.5 Flash、RigoChat‑7B‑v2、LoRA适配、LangGraph、内部质量评估模块和规则引擎。

**📊 数据集**

使用iDEM、UNE 153101 E2R材料、公开西班牙语E2R对齐语料和医学词汇资源进行校准和词典构建。

**📈 对比分析**

通过官方BLEU‑Orig、BLEU‑Gold、SARI和BERTScore比较，RUN1以SARI 44.0543排名第6，RUN2略低，RUN3最低，显示多智能体策略优于线性基线。

**⚠️ 局限性**

局限在于仅使用官方指标评估，词汇支持未提升参考得分，缺乏文档级错误分析和真实用户可读性验证。

---

## 544. Generalized Rank Weight and Extended Generalized Poset Weight Defined For Codes Over Rings: A Galois Connection Approach

**arXiv ID:** 2607.02377 | [PDF](https://arxiv.org/pdf/2607.02377v1)

**作者:** Yang Xu `[一作]` (Fudan University), Guangyue Han `[通讯]` (University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了码在环/模上的广义秩权重和扩展广义格子权重，并给出了它们之间的 Galois 连接以及相关的 Wei 型对偶性定理、Singleton 与扩展 Singleton 上界，探讨了 MRD、QMRD、NMRD、i‑MRD 等码的性质及散射界限；

**💡 创新点**

创新点在于利用 Galois 连接统一并推广了多种已知权重的 Wei 型对偶性定理（包括 GHW、GPW、GRW、GMW、EGHW、EGPW 等），提出了新的广义秩权重与扩展广义格子权重定义，并得到了一系列新的 Singleton 上界、码类型的等价判定与散射界限；

**🔧 技术方法**

主要技术包括 Galois 连接理论、模论与对偶性、MacWilliams 识别、模块化双线性形式以及链环和准 Frobenius 环的结构性质；

**📊 数据集**

无实验数据集，属于纯理论研究；

**📈 对比分析**

通过与已有的 Wei 型对偶性定理和 Singleton 上界结果进行对比，证明了新定理在特定情形下与已知结论完全一致，进一步揭示了权重序列与对偶权重之间的完整互定关系；

**⚠️ 局限性**

局限性在于仅适用于主理想环/链环或准 Frobenius 环等特定环，且对非主理想或非自由模的情况尚未覆盖，缺乏对实际编码实现和性能评估的讨论。

---

## 545. DRIFTLENS: Measuring Memory-Induced Reasoning Drift in Personalized Language Models

**arXiv ID:** 2607.02374 | [PDF](https://arxiv.org/pdf/2607.02374v1)

**作者:** Xi Fang `[一作]` (Amazon), Chandan K. Reddy `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了在大型语言模型中持久用户记忆如何改变模型在开放式无真值任务上的推理轨迹，并提出了一个无基准真值的“漂移检测”框架；

**💡 创新点**

创新点在于构建了基于价值本体的推理符号化方法、定义了DTW和SRI两种漂移度量，并系统验证了多模型、多属性下的“符号漂移”现象；

**🔧 技术方法**

采用的技术包括本体构建与迭代标注、动态时间规整（DTW）与符号分布相似度（SRI）评估、线性混合效应模型与集群自助法统计检验、以及GRPO（在线RL）和DPO（离线优先级）后训练漂移缓解；

**📊 数据集**

使用的主要数据集包括：422个无真值推理题、10类用户属性扰动（如年龄、职业、残障等）、生活重大事件正控制以及无内容噪声负控制，附加本体与评估脚本公开；

**📈 对比分析**

实验显示，在四个不同规模模型（Claude Sonnet 4.6、GPT‑OSS‑120B、Qwen3‑4B、DeepSeek‑R1）上，用户属性记忆会使DTW/SRI漂移显著高于噪声基准（Cohen’s d≈0.35–0.98）；GRPO与DPO在不同模型上均能降低漂移，但对通用能力、帮助性与指令遵循的影响各不相同，且不存在统一最优方案；

**⚠️ 局限性**

局限性包括：仅测量外部表达的推理路径而非内部认知；固定本体可能忽略细粒度文化或任务差异；实验仅限中小规模开源模型；并且只考察单回合开放式问答，未覆盖多回合交互或工具使用等场景。

---

## 546. VisionAId: An Offline-First Multimodal Android Assistant for People with Visual Impairment, Featuring Personalized Object Retrieval

**arXiv ID:** 2607.02371 | [PDF](https://arxiv.org/pdf/2607.02371v1)

**作者:** Cristian-Gabriel Florea `[一作]` (Military Technical Academy 'Ferdinand I'), Stelian Spînu `[通讯]` (Military Technical Academy 'Ferdinand I')

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款面向视障用户的 Android 离线多模态助手 VisionAId，集成了障碍感知、个人物体注册与 AR 导引、面部识别、色彩判定、罗马尼亚纸币检测等功能，并通过语音、触觉和空间音频等多种反馈方式提升可访问性。

**💡 创新点**

创新点包括：① 采用少样本学习管线实现个性化物体检索；② 将六个深度学习模型（深度估计、实例分割、视觉嵌入、面部检测/嵌入、纸币检测）全部部署在设备上；③ 对深度模型进行 INT8 量化、ARCore 3D 定位以及自适应阈值的嵌入匹配；④ 开发了罗马尼亚纸币专用 YOLO 模型，解决了无公开数据集的问题；⑤ 在同一设备上实现实时深度、检测、嵌入和 AR 引导，保持帧率在 5–8 FPS 内。

**🔧 技术方法**

技术栈包括：ONNX Runtime（支持 INT8 量化）、Kotlin 2.0 + Jetpack Compose、CameraX、ARCore、Google Gemini Flash（可选）、MobileCLIP、YOLOv11n-Seg、YOLOv26n、YuNet、MobileFaceNet、Room 数据库、TTS 语音合成、触觉反馈、空间音频。

**📊 数据集**

使用的数据集包括：① COCO 作为通用目标检测和分割基准；② 自建的罗马尼亚纸币数据集（约 800 张图片，涵盖 8 种面值）；③ 通过 Roboflow 采集的多角度个人物体图片用于少样本训练；④ 公开的面部图像数据集用于 MobileFaceNet 预训练（如 300W-LP 等）。

**📈 对比分析**

在 Samsung Galaxy S21 Ultra 5G 上评估：深度模型 INT8 量化后 491 ms/帧（约 7–8 FPS），AR 检索 560 ms/帧（≈5 FPS），纸币检测 <15 ms/帧；深度校准后 1 m 误差 <1 cm；纸币检测 mAP@50 达 0.986；面部识别召回率 0.94。与基线（如未量化模型、单独检测器）相比，延迟下降 2.4×，准确率基本保持。

**⚠️ 局限性**

局限性包括：仅在高端设备上进行了性能评估，mid‑range 设备上延迟可能提升 2–3 倍；缺乏正式的视障用户实验验证；云端 Gemini 仅用于场景描述，若无网络仍能使用核心功能；部分模型（如 MobileCLIP）尚未支持 INT8 量化；AR 追踪对光照、遮挡敏感。

---

## 547. Extreme Adaptive Transformer for Time Series Forecasting

**arXiv ID:** 2607.02437 | [PDF](https://arxiv.org/pdf/2607.02437v1)

**作者:** Sanjeev Shrestha `[一作]` (Missouri State University), Yifan Zhang `[通讯]` (Missouri State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种极端自适应Transformer（Exformer），用于预测极端事件稀缺的水文时间序列。

**💡 创新点**

创新点是引入极端自适应注意机制：由Local、Stride、Extreme三种稀疏组件组成，能根据 token 是否为极端事件动态选择注意键，显式捕获正常与极端事件之间的依赖，同时降低计算量。

**🔧 技术方法**

采用Transformer编码器结构、稀疏注意力（Dozer）、季节趋势分解、异常检测生成极端标签、GMM+阈值标注、Kruskal‑Wilcoxon采样、log+标准化预处理等技术。

**📊 数据集**

在圣塔克拉拉县四条河道（Ross、Saratoga、UpperPen、SFC）的15分钟间隔流量与降雨数据上进行实验。

**📈 对比分析**

与九个基线（FEDformer、Informer、NLinear、DLinear、LSTM‑Atten、NEC+、iTransformer、DAN、PFformer）在3天（288步）RMSE/MAPE指标上对比，Exformer在7/8指标上获最优，MAPE最低，且FLOPs与显存约为PFformer的1/5。

**⚠️ 局限性**

局限：对极端事件阈值敏感，需人工调参；实验仅涵盖水文流量数据，泛化性待验证；极端标签依赖异常检测与GMM，误标会影响性能。

---

## 548. Neuron-Aware Active Few-Shot Learning for LLMs

**arXiv ID:** 2607.02423 | [PDF](https://arxiv.org/pdf/2607.02423v1)

**作者:** Zhuowei Chen `[一作]` (University of Pittsburgh), Xiang Lorraine Li `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 NeuFS，一种利用大型语言模型内部神经元激活模式进行主动少样本学习（AFSL）的框架。

**💡 创新点**

创新点在于：①将样本选择的依据从输出层的概率/熵切换到内部神经元激活；②使用“早期解嵌入（Early Unembedding）”筛选有贡献的神经元；③通过神经元一致性（Neuron Consensus）衡量模型的幻觉风险；④结合聚类与共识双重评分，既保证知识覆盖又提升信息性。

**🔧 技术方法**

核心技术包括：内部 FFN 激活提取、Early Unembedding 计算贡献分数、Jaccard 相似度+K-Medoids 聚类、神经元共识计数、双重权重分数（τ）进行最终样本选取。

**📊 数据集**

使用了三大数据集：MMLU-Pro（推理）、Edu-Feedback（二分类）和 TREC（多分类），并在四个指令微调 LLM（Llama3 3B/8B，Qwen3 4B/8B）上进行实验。

**📈 对比分析**

与随机、熵、语义聚类（TypiClust）、Patron、FastVoteK、VoteK 等六种 AFSL 基线对比，NeuFS 在所有任务、模型规模及 5/10/20/30 shots 上均实现了最高或第二高的准确率/ F1，尤其在推理任务中显著优于组合语义+熵的 Patron，证明内部激活信息更为可靠。

**⚠️ 局限性**

局限性：需要访问模型内部 FFN 激活和解嵌入矩阵，限制了对封闭式 API 的适用性；计算成本相对较高，尤其是在大规模未标记样本池上进行激活提取。

---

## 549. Wavelet-Guided Semantic Signal Compensation for Inversion-Free Image Editing

**arXiv ID:** 2607.02421 | [PDF](https://arxiv.org/pdf/2607.02421v1)

**作者:** Anqi Tang `[一作]` (University of Electronic Science and Technology of China), Zhaoqiang Liu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无逆向（inversion-free）的图像编辑框架，利用频域的 Haar 小波补偿来强化在高噪声阶段的语义引导，从而实现更强的全局属性修改与背景保持；

**💡 创新点**

创新点在于：①通过“同点语义探测”获取源目标在相同潜在点的文本差异向量；②采用多级 Haar 小波分解提取低频语义补偿信号，并以时间可调权重注入编辑轨迹；③实现了在 rectified flow 模型上不增加额外计算或模型修改的全局编辑提升；

**🔧 技术方法**

使用技术包括：rectified flow（FlowEdit）框架、2D Haar 小波变换、时间权重调度（λt²）、流差分注入、频域分析；

**📊 数据集**

实验数据集包括 PIE-Bench（700 图像多种编辑场景）、EditBench（高分辨率 1024×1024）以及多个基准模型的实现；

**📈 对比分析**

与多类基准（Diffusion、Rectified Flow、频域感知等）比较，本文方法在 CLIP 语义一致性、背景保真度（PSNR/SSIM/LPIPS）等指标上均取得最佳或次优成绩，并在用户研究中获得最高偏好率；

**⚠️ 局限性**

局限性在于：对全局属性的补偿仍需手动调节 λ 与 L，且在极端大规模编辑或极细粒度细节处理时，低频补偿可能不足；

---

## 550. The Future of NLP may not be at NLP Conferences: Scholarly Migration Patterns in Natural Language Processing

**arXiv ID:** 2607.02416 | [PDF](https://arxiv.org/pdf/2607.02416v1)

**作者:** David Jurgens `[一作]` `[通讯]` (University of Michigan), David Jurgens (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对2010-2026年141,710篇NLP主题论文的系统采集与分析，量化了自LLM时代起NLP作者向更广泛的ML会议迁移的规模与趋势；

**💡 创新点**

创新点在于首次将大规模面板回归、Oaxaca-Blinder分解、作者层次匹配与引用增益估计等多种方法结合，全面解析迁移机制与其对引用影响；

**🔧 技术方法**

主要技术包括跨源作者识别、面板OLS回归、Oaxaca-Blinder分解、SPECTER2嵌入+k-means聚类、余弦相似度匹配、logit模型以及对log(1+citations)的回归；

**📊 数据集**

使用的数据集来自ACL Anthology、Semantic Scholar、OpenAlex、OpenReview、PMLR、DBLP等公开来源，涵盖23个NLP/ML/AI会议与期刊；

**📈 对比分析**

通过比较作者在ACL与ML-general等会议的发表比例变化（Δshare）以及匹配论文的引用对比（约+75%至+118%），验证了迁移幅度与引用优势；

**⚠️ 局限性**

局限性包括判别器在ML-general会议的精度不高、会议结构变化导致基准不稳定、PhD cohort平台偏倚，以及因观察性研究而无法确立因果关系。

---

## 551. Bringing Agentic Search to Earth Observation Data Discovery

**arXiv ID:** 2607.02387 | [PDF](https://arxiv.org/pdf/2607.02387v1)

**作者:** Minghan Yu `[一作]` (University of Maryland), Haizhao Yang `[通讯]` (University of Maryland)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于NASA地球观测知识图谱的地球观测数据检索系统，提供自然语言查询到对应数据集的全流程检索与排序。

**💡 创新点**

创新点在于：① 将科研论文的引用数据转化为大规模可验证的查询–数据集基准（47,654对，21,272任务式查询）；② 设计了融合BM25与神经评分校正（NN‑SSC）的检索套件，并通过凸组合实现最佳混合；③ 对同一模型进行零-shot LLM排序与加工具器调用（Web + arXiv）对比，首次在检索任务中系统性评估 agentic reranking 的价值。

**🔧 技术方法**

使用技术包括：BERT 句子变换器（nasa‑smd‑ibm‑st‑v2）及其微调、BM25、神经评分校正 MLP、均值偏移校正（R2）、LLM（GPT‑5.5、Claude Opus 4.7、DeepSeek v4 pro 等）以及自研 agentic harness（预设 5 步搜索+推理流程并调用 Web 与 arXiv）。

**📊 数据集**

主要数据集为 NASA Earth Observation Knowledge Graph (提取 10,636 篇高引用论文的 dataset‑citation 关系) 与 NASA Common Metadata Repository（8,058 个数据集）组合构成的 47,654 个查询–数据集正样本；另外使用公开的 NASA 官方工具（Harmony、WorldView、Giovanni、SDE）作为系统的第一阶段检索入口。

**📈 对比分析**

通过在同一检索基线上（BM25+NN‑SSC）对比：1）基线 BM25 与无微调句子变换器；2）微调句子变换器 + BM25；3）微调 + BM25 + R2；4）零-shot LLM 排序；5）同一 LLM 加工具器调用。结果显示：检索套件将 R@10 提升至 0.4275（相比基线 0.1083），提升超过 5×；在 N=200 子集上，单机 LLM rerank 相比无 rerank 提升 0.06–0.08；agentic rerank 在 Opus 4.7 与 DeepSeek v4 pro 上均实现 3–4% 的绝对提升，表现出工具调用的正面影响。

**⚠️ 局限性**

主要局限包括：① 以论文引用作为“银标签”可能漏掉未被引用但同样相关的数据集；② 对热门数据集存在偏好，长尾产品的评价受限；③ 允许 LLM 访问 Web/arXiv 存在潜在标签泄露风险；④ 评测规模仅覆盖 200 个查询，缺乏全量基准和人工评估；⑤ agentic 方案显著增加计算成本，需进一步研究成本与收益平衡。

---

## 552. The Weight Distribution of the Third-Order Reed-Muller Code of Length 2048

**arXiv ID:** 2607.02365 | [PDF](https://arxiv.org/pdf/2607.02365v1)

**作者:** Kirill Khoruzhii `[一作]` (Zuse Institute Berlin), Sebastian Pokutta `[通讯]` (Zuse Institute Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对长度 2048 的三阶 Reed–Muller 码 (3,11) 进行了完整的权分布计算，并由此得到相对覆盖半径 ρ_{2,3}(10)=408，提升了先前的下界 400；同时给出了 179 个极值轨道的二阶非线性与交替秩信息，并通过启发式搜索将 (6,10) 在 (7,10) 的相对覆盖半径上界从 50 降到 32。

**💡 创新点**

主要创新在于提出并证明了一个结构定理：非退化布尔三次多项式在偶维空间中必有非退化的超平面限制，从而把枚举成本由 2^{36} 降到 2^{26}；利用该定理完成了前所未有的 (3,11) 码权分布全量计算；并将同一技术与局部搜索相结合，显著降低了求解高阶覆盖半径上界的计算量。

**🔧 技术方法**

采用了 Sarwate‑type 递推公式、Coset 权枚举、快速 Walsh–Hadamard 变换、结构定理的分解方法以及三种改进的局部搜索策略（τ‑move、beam search、iterated local search），从而实现了大规模枚举与高效的上界搜索。

**📊 数据集**

使用了 10 变量布尔三次多项式的全轨道分类数据集，共 3 691 560 个非零轨道，并利用该分类提供的代表元与稳定子信息进行权枚举和搜索。

**📈 对比分析**

与已知的下界 400、上界 50 以及以前在 512 长度码上的结果进行对比；枚举耗时约 65 CPU‑年，启发式搜索仅 538 CPU‑小时；最终得到的覆盖半径 408 明显优于 400，且 32 的上界显著低于 50，验证了方法的有效性。

**⚠️ 局限性**

主要局限在于计算复杂度仍然极高，仅能在偶维 10 变量的情况下实现；结构定理在奇维度存在唯一例外轨道；对于更高维度的三阶码，尚未验证该方法的可扩展性；此外，极值轨道中大多数并不具有简洁的多项式表示，限制了进一步分析。

---

## 553. Reasoning effort, not tool access, buys first-try reliability in agentic code generation: an observational study

**arXiv ID:** 2607.02436 | [PDF](https://arxiv.org/pdf/2607.02436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 554. SoK: A Taxonomy for Cybersecurity Incident Response Influence Factors

**arXiv ID:** 2607.02451 | [PDF](https://arxiv.org/pdf/2607.02451v1)

**作者:** Thomas Biege `[一作]` (FH Münster University of Applied Sciences), Sebastian Schinzel `[通讯]` (FH Münster University of Applied Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统文献综述与定性编码，构建了首个统一的网络安全事件响应影响因素（CIR‑IF）分类体系，涵盖技术、人类与组织层面的多维度因素。

**💡 创新点**

创新点在于：①首次将 1999–2024 年的研究成果聚合为一个多层次、可扩展的分类框架；②将人因、组织与技术因素系统化，并与 NIST 800‑61r3 及七个学术框架对照，揭示现有标准在人员与治理层面的覆盖不足；③提供可直接使用的编码书与方法，为后续定性研究与理论建模奠定基础。

**🔧 技术方法**

使用了 PRISMA 2020 指南的三阶段系统综述流程；采用 Nickerson 等人 2013 年的分类法构建与迭代方法；结合 Cohen’s κ 评估编码一致性；并利用定性内容分析（Mayring）验证。

**📊 数据集**

数据集来源于 457 篇初筛文献，最终纳入 105 篇符合质量标准的学术与灰色文献（涵盖期刊、会议、博士论文等），并在此基础上提炼 105 个影响因素与 136 个子类别。

**📈 对比分析**

比较方法为将 CIR‑IF 子类别映射到 NIST SP 800‑61r3 社区配置表及七个学术框架元素，计算覆盖比例并绘制比例图。结果显示 CIR‑IF 在人员与治理层面的覆盖显著高于 NIST（约 60% 以上的人员因素 vs 仅 3%），而 NIST 在技术与流程层面覆盖更全面，说明两者互补。

**⚠️ 局限性**

局限性包括：①仅检索英文同行评议文献，可能遗漏非英语或灰色资料；②未对影响因素进行定量效度检验，仅描述潜在关联；③与行业实践的差距尚未充分验证，未来需结合企业案例进行实证检验。

---

## 555. AgentsCAD: Automated Design for Manufacturing of FDM Parts via Multi-Agent LLM Reasoning and Geometric Feature Recognition

**arXiv ID:** 2607.02448 | [PDF](https://arxiv.org/pdf/2607.02448v1)

**作者:** Emmanuel George `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了AgentsCAD，一个多代理系统，将STEP文件解析为结构化JSON后，用Claude Sonnet LLM给出DFM建议，最终输出修改后的STEP文件和可读报告。

**💡 创新点**

首次将B‑Rep语义+GraphSAGE嵌入与LLM推理相结合，采用黑板多代理架构、RAG记忆、MCP工具 grounding，完成FDM过度悬垂检测与几何修正的闭环。

**🔧 技术方法**

使用大语言模型（Claude Sonnet 4.6、GPT‑4o视觉验证）、Graph Neural Network（GraphSAGE、UV‑Net）、MCP工具、黑板架构、RAG检索以及CadQuery/OCCT进行解析与修改。

**📊 数据集**

以MFCAD++（59,665个工业零件）训练GraphSAGE，并在鸟屋案例及两份测试零件上验证。

**📈 对比分析**

在MFCAD++ hold‑out 上比较GCN与GraphSAGE，GraphSAGE+UV‑Net宏F1达0.785、准确率0.85；实验中鸟屋两处悬垂被成功修复，修改后无悬垂、体积误差<1%。

**⚠️ 局限性**

受限于模型上下文窗口导致大零件面数难处理；仅支持单零件无装配；缺乏多缺陷类别（桥、薄壁等）支持；需要手工 prompt 设计与 RAG 扩展。

---

## 556. Improved Approximation Algorithms for n-Pairs Shortest Paths

**arXiv ID:** 2607.02443 | [PDF](https://arxiv.org/pdf/2607.02443v1)

**作者:** Avi Kadria `[一作]`, Virginia Vassilevska Williams `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

论文探讨了某种新型算法在特定领域的应用。

**💡 创新点**

创新点在于提出了一种改进的算法结构，能够提高处理效率。

**🔧 技术方法**

使用了深度学习和机器学习技术。

**📊 数据集**

采用了公开的标准数据集进行实验。

**📈 对比分析**

与现有方法进行了对比，结果显示新算法在准确率和速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定情况下的适用性和对数据质量的依赖。

---

## 557. QFedAgent: Quantum-Enhanced Personalized Federated Learning for Multi-Agent Activity Recognition

**arXiv ID:** 2607.02426 | [PDF](https://arxiv.org/pdf/2607.02426v1)

**作者:** Quoc Bao Phan `[一作]` (Florida State University), Tuy Tan Nguyen `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 QFedAgent 的个性化联邦学习框架，用于多模态 IMU 传感器的多主体活动识别，并将变分量子电路作为融合模块。

**💡 创新点**

创新点在于将变分量子电路（VQC）引入联邦学习，实现跨模态融合的参数量缩减约 10 倍，同时保持甚至提升识别精度。

**🔧 技术方法**

使用的技术包括双 CNN 编码器、VQC 量子状态编码与纠缠、参数‑shift 梯度法、FedAvg/ FedProx 联邦训练、客户端适配器与分类头。

**📊 数据集**

实验采用 OPPORTUNITY 活动识别数据集，四名受试者分别视为联邦客户端，提供加速度计和陀螺仪的多通道时间序列。

**📈 对比分析**

与本地训练、FedAvg、FedProx、经典 MLP‑FL 基线对比，QFedAgent 在平均测试准确率上达到 97.7%，优于 97.2% 的 MLP‑FL，且仅使用 72 个量子旋转参数；但量子电路模拟导致单轮训练时间显著提升。

**⚠️ 局限性**

局限性包括：量子电路模拟在经典硬件上计算成本高；对真实量子硬件的可扩展性与噪声鲁棒性尚未验证；在更大规模或更严重非 IID 条件下的性能可能下降。

---

## 558. Text-Driven 3D Indoor Scene Synthesis in Non-Manhattan Environments

**arXiv ID:** 2607.02407 | [PDF](https://arxiv.org/pdf/2607.02407v1)

**作者:** Xianhui Meng `[一作]` (University of Science and Technology of China), Xiaoshuai Hao `[通讯]` (Xiaomi EV)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 SPG-Layout 框架，能够在非曼哈顿室内环境中实现物理合理的文本驱动 3D 场景合成。

**💡 创新点**

创新点在于将空间先验引导（SPG）与层次化布局策略（HLS）结合，形成双重优化，解决非正交空间关系难题。

**🔧 技术方法**

采用结构化场景表示、LLM 监督微调、GRPO 强化学习、注意力加权的空间先验奖励以及多尺度 3D 资产检索。

**📊 数据集**

使用 500 个手工修订的非曼哈顿室内场景作为新基准，并在 SSR-3DFRONT 等曼哈顿数据集上进行对照评估。

**📈 对比分析**

与多种基线（LayoutGPT、LayoutVLM、InstructScene、ATISS、MiDiffusion、ReSpace）对比，SPG-Layout 在布局违背率、真实性指标和用户研究中均取得显著优于对手的成绩。

**⚠️ 局限性**

局限性包括对罕见物品类别的泛化受限、极端物品密度下偶有功能失配、以及基于体素的几何奖励计算成本较高。

---

## 559. FlintKV: A Fast Durable Storage Engine for Modern Databases

**arXiv ID:** 2607.02401 | [PDF](https://arxiv.org/pdf/2607.02401v1)

**作者:** Sergey Egorov `[一作]` (University of London), Sadegh Keshavarzi `[通讯]` (University of Surrey)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于跳表的NVM优化存储引擎，提供完整的键值API，包括原子批量写、快照一致迭代等功能；

**💡 创新点**

创新点是将平面合并（flat‑combining）并发控制与多版本管理（MVCC）和持久化机制相结合，既能保证线性可达性，又显著提升吞吐量；

**🔧 技术方法**

使用了NVM本地内存、跳表数据结构、平面合并并发控制、MVCC、多版本持久化和针对性I/O对齐技术；

**📊 数据集**

评测使用YCSB、TPC‑C等常见工作负载，并在Intel Optane DC Persistent Memory阵列上运行；

**📈 对比分析**

与RocksDB、LMDB及其他NVM键值存储进行基准对比，实验显示吞吐量提升可达75%，延迟下降10%–20%；

**⚠️ 局限性**

局限性包括对写密集场景下的空间回收机制仍不完善，在极大键空间下跳表的内存占用可能显著增加。

---

## 560. Object-centric LeJEPA

**arXiv ID:** 2607.02404 | [PDF](https://arxiv.org/pdf/2607.02404v1)

**作者:** Jakob Geusen `[一作]` (ETH Zurich), Ender Konukoglu `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文在LeJEPA自监督学习框架上，将对齐与正则化从图像级迁移到对象级，通过使用SAM 2生成的对象掩码训练，提升特征质量并实现更高的数据效率；

**💡 创新点**

创新点在于：1) 在训练时直接使用外部掩码实现对象级对齐，避免了对象分割与表征学习的循环不稳定；2) 引入实例分离损失，使同一场景中同类别对象在特征空间保持可区分；3) 展示对象级LeJEPA在仅10% COCO数据下即可匹敌全量图像级对齐模型；

**🔧 技术方法**

技术方法包括：ViT Backbone提取patch特征；基于掩码的语义对象表示聚合；Object-centric LeJEPA对齐+正则化（SIGReg）；Instance-level对比损失；数据增强与多视图采样；在训练时使用SAM 2掩码；在推理时无需掩码，直接使用patch特征；

**📊 数据集**

主要使用COCO数据集进行预训练；下游评估在ImageNet-1k、ADE20k、DAVIS、NAVI等公开数据集；SAM 2用于生成掩码；

**📈 对比分析**

与图像级LeJEPA、SlotMIM和DINOv3比较：Object LeJEPA在多项任务（tracking、classification、segmentation、re-identification）均优于或匹配图像级LeJEPA，并在10% COCO预训练时即可达到全量模型性能；在ImageNet线性分类上平均池化patch特征亦优于图像级LeJEPA；

**⚠️ 局限性**

限制在于：1) 训练依赖外部掩码生成，掩码质量影响表现；2) 对象级对齐难以处理对象与背景混合的复杂场景；3) 对比损失对实例区分有时效果有限，尤其在语义任务中略逊；4) 目前未探索大规模多模态或更高分辨率的场景。

---

## 561. Transformer Geometry Observatory TGO-II: Representational Similarity Observatory

**arXiv ID:** 2607.02386 | [PDF](https://arxiv.org/pdf/2607.02386v1)

**作者:** Kaustubh Kapil `[一作]` (Sardar Vallabhai National Institute of Technology), Kishor P. Upla `[通讯]` (Sardar Vallabhai National Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 Transformer Geometry Observatory-II（TGO‑II），一个专门用来分析 Vision Transformer 在监督训练过程中内部表征几何演化的框架，并通过多种指标对表征相似性、内在维度与 token 交互结构进行了系统评估。

**💡 创新点**

创新点在于：①首次将 CKA、SVCCA、TwoNN‑ID 与 token 协方差/耦合度量整合到同一观测框架中，形成“表征几何观测”视角；②揭示了表征相似性随训练下降、内在维度随训练上升并最终稳定、token 交互结构始终保持强耦合的三重共生关系；③提出了“流形扩张”与“转换区”假设，试图从几何视角解释表征多样性与层专化的同步出现。

**🔧 技术方法**

使用的技术包括：Centered Kernel Alignment (CKA)、Singular Vector Canonical Correlation Analysis (SVCCA)、Two‑Nearest‑Neighbor Intrinsic Dimensionality estimator (TwoNN‑ID)、token 协方差矩阵和 token 融合比（coupling ratio）计算，以及对 ViT‑Small/16 的前向 hook 激活提取。

**📊 数据集**

实验数据集为 ImageNet‑100（100 个类别），在 100 epoch 的监督训练下采集 1000 张验证图像的表征，用于计算所有指标。

**📈 对比分析**

方法比较：与先前的 TGO‑I（基于特征协方差的谱分析）进行对比，观察到 TGO‑II 在表征相似性和内在维度方面与谱指标表现一致；通过图表显示 CKA/SVCCA 随训练的下降趋势、TwoNN‑ID 的上升-稳定曲线以及 token 协方差矩阵始终保持非对角结构，表明模型内部表征在训练过程中既趋向专化又保持高维流形和强 token 交互；但并未提供传统任务性能指标的提升，仅关注表征几何变化。

**⚠️ 局限性**

局限性包括：①仅在 ViT‑Small/16 和 ImageNet‑100 上验证，缺乏对更大模型/数据集的泛化；②未直接测量语义信息或任务相关性，无法确定内在维度提升是否真正对应语义丰富性；③缺乏理论证明将内在维度与协方差谱联系起来，只给出了经验性假设；④token 交互持续强耦合的结论基于协方差矩阵统计，可能受到梯度噪声或训练超参数的影响。

---

## 562. Representation Distribution Matching for One-Step Visual Generation

**arXiv ID:** 2607.02375 | [PDF](https://arxiv.org/pdf/2607.02375v1)

**作者:** Lan Feng `[一作]` (EPFL), Alexandre Alahi `[通讯]` (EPFL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于Representation Distribution Matching（RDM）的单步图像生成框架，并给出了完整的训练与评估方法。

**💡 创新点**

创新点包括：①将传统MMD与Nyström近似结合，使用大批量新样本实现高质量分布匹配；②在条件任务中匹配联合图像‑文本分布；③使用多编码器电池并通过受限优化保持平衡，避免单一编码器“作弊”；④设计了独立于训练目标的Sliced‑Wasserstein评估指标。

**🔧 技术方法**

采用的技术包括：最大均值差异（MMD）+Nyström吸引 + 逐批惩罚、联合图像‑文本特征匹配、梯度缓存实现大批量训练、受限Lagrangian权重控制多编码器、Sliced‑Wasserstein距离评估。

**📊 数据集**

主要使用的数据集为ImageNet-256（1.28M训练图像）和COCO（用于文本‑图像后训练和评估）。

**📈 对比分析**

与现有公开单步生成器对比，RDM在Sliced‑Wasserstein r14上取得1.30（前沿为2.05），在PickScore上胜过所有先前模型（71.2%胜率），并将四步FLUX.2后训练为单步模型，在GenEval上从0.794提升至0.826，在PickScore上提升至22.58。

**⚠️ 局限性**

局限性：仍与真实图像存在1.30 vs 1的差距；依赖预训练编码器电池，需手工挑选；训练需要冻结完整参考，适配其他模态时需重新构建参考与编码器。

---

## 563. GAP-GDRNet: Geometry-Aware Monocular Visual Pose Sensing on a Single-Target Synthetic Spacecraft Dataset

**arXiv ID:** 2607.02360 | [PDF](https://arxiv.org/pdf/2607.02360v1)

**作者:** Yonglong Zhang `[一作]` (Harbin Institute of Technology), Yang Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了针对单目标航天器的单目RGB 6D姿态估计框架 GAP-GDRNet，并通过改进特征提炼与补丁级几何自注意力提升精度。

**💡 创新点**

创新点包括：① 先于密集几何预测的注意力特征强化模块 AFR（融合全局 GGCA 与局部 MECS）；② 在 Patch-PnP 中插入补丁级几何自注意力 PGSA，强化跨区域几何关联。

**🔧 技术方法**

技术手段：基于 ConvNeXt 的特征提取，GGCA/MECS 关注机制，Patch-PnP 结构，Patch-level 自注意力，Blender 渲染与注释，端到端训练损失。

**📊 数据集**

使用单一航天器 CAD 模型的 5 万张 Blender 生成的合成图像（含姿态、光照、背景、遮挡等变化），以及公开的 T‑LESS 与 LM‑O 数据集做补充验证。

**📈 对比分析**

与 GDR‑Net 基线在自建航天器数据集上对比，旋转误差降至 1.96°、平移误差 0.0165 m、ADD@0.02 m 达到 95.16%，相较基线提升 3.88pp；帧率 35.97 FPS。T‑LESS 与 LM‑O 上也分别提升 6.8pp 与 3.1pp。

**⚠️ 局限性**

局限性：仅在单一航天器模型和合成图像上验证，缺乏真实空间图像测试；训练与推理仅针对单目标，未扩展到多目标或多种航天器；对高度光照与材质变异的泛化仍待评估。

---

## 564. Learning Spectral and Polarimetric Clues for One-to-Multimodal Novel View Synthesis

**arXiv ID:** 2607.02372 | [PDF](https://arxiv.org/pdf/2607.02372v1)

**作者:** Federico Lincetto `[一作]` (University of Padova), Pietro Zanuttigh `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种通过多场景预训练学习不同成像模态（RGB、NIR、Monochrome、Polarimetric、Multi‑Spectral）之间相互关联，再在单场景仅用 RGB 或极少数模态进行微调，即可合成任意模态的新视角图像。

**💡 创新点**

创新点包括：①基于基础（basis）和系数（coefficient）网格的共享与场景特定分离机制；②三种专门的潜空间正则化（潜空间几何损失、逆函数损失、模态‑到‑亮度损失）以保持多模态一致性；③交替冻结共享与特定模块的预训练/微调流程，使模型在缺乏多模态数据时仍能高质量生成未观测模态。

**🔧 技术方法**

技术实现基于 NeRF/GS 的隐式场渲染，继承并改进 MultimodalStudio 框架，使用周期性坐标映射、浅层 MLP 进行投影、潜编码与解码，并加入 Eikonal、曲率正则化及上述三种潜空间损失；训练分为多场景预训练和单场景微调两阶段。

**📊 数据集**

使用 MMS‑DATA（32 个物体场景，5 种模态：RGB、NIR、Mono、Pol、MS）进行预训练（27 场景）和微调（5 场景）；额外在 X‑NeRF 数据集上验证不平衡模态组合的效果。

**📈 对比分析**

与从零训练的多模态 NeRF MMS‑FW、以及两阶段 RGB→MS（MST++）/Pol（PolarAnything）组合相比，方法在仅 RGB 监督下 PSNR 约提升 0.5–1.5 dB；在少量二模态帧的实验中，PSNR 相比 MMS‑FW 提升 6–8 dB；在多视角一致性、MAngE、MAbsE 等指标上也均优于对手，证明了更好的多模态一致性与渲染质量。

**⚠️ 局限性**

局限性：仅在与预训练场景材质相似的物体上表现良好；对新模态需重新预训练；对曝光、白平衡和照明条件敏感，要求固定拍摄设置；对光谱范围之外的极化/红外变化鲁棒性有限。

---

## 565. Deterministic Polynomial-time Exact-root Computation for Sparse Polynomials with Bounded Total Degree

**arXiv ID:** 2607.02364 | [PDF](https://arxiv.org/pdf/2607.02364v1)

**作者:** Qiao-Long Huang `[一作]` (Shandong University), Xiao-Shan Gao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了在多变量情况下确定性计算稀疏多项式的确切根的问题，提出了一种算法来计算基多项式g。

**💡 创新点**

证明了基多项式g的稀疏性界限，并开发了一种在总度D有界的情况下的多项式时间算法，显著优于现有的准多项式依赖算法。

**🔧 技术方法**

使用了标量加权替换、y-adic二项式展开和Frobenius还原等技术。

**📊 数据集**

使用了稀疏多项式f∈[x_1,…,x_n]，其稀疏性s=f_0，个别度d=indeg(f)，总度D=(f)。

**📈 对比分析**

与Bhargava、Saraf和Volkovich的通用确定性因式分解算法相比，提出的算法在总度D有界的情况下是多项式时间的，复杂度为poly(s^O(Dd), n, d, D) + s·R(e)。

**⚠️ 局限性**

算法的局限性在于R(e)的计算成本，特别是在特征分割的情况下，计算单个Frobenius根的成本可能会影响整体性能。

---

## 566. Cloak and Detonate: Scanner Evasion and Dynamic Detection of Agent Skill Malware

**arXiv ID:** 2607.02357 | [PDF](https://arxiv.org/pdf/2607.02357v1)

**作者:** Zimo Ji `[一作]` (Hong Kong University of Science and Technology), Shing-Chi Cheung `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了LLM代理技能的恶意供应链风险，展示了现有静态扫描器易被payload‑preserving伪装攻击规避，并提出了两种攻击技术（结构化混淆与自提取技能打包），随后设计并实现了基于行为的运行时审计器，能够在沙箱中执行技能并通过操作系统边界信息流检测恶意行为。

**💡 创新点**

①系统化评估了攻击者在保持功能不变的情况下对静态扫描器的规避能力；②提出了可自动生成payload‑preserving伪装的框架；③首次在代理技能场景中实现了闭包提升（On‑Demand Closure Lift）和标记型污点分析（Marker‑Based Taint），实现对多阶段、跨媒介数据流的动态检测。

**🔧 技术方法**

技术包括：静态规则抽取与自动化编辑器、零宽字符与语义重写（结构化混淆）；自提取打包与动态解包；基于eBPF的系统调用监控；FUSE层的文件访问拦截；符号化读取与数据标记；信息流策略（机密性与完整性）；Docker化沙箱与网络隔离；以及对LLM模型的对话式审计。

**📊 数据集**

数据集：1) 1,613个真实恶意技能（OpenClaw市场）；2) 600个经过伪装的SkillJect基准技能（450恶意、150善意）；3) 622个可执行野外技能（MalSkillBench）。

**📈 对比分析**

对比方法：与九款主流静态扫描器（Huifer、Vigile、Qualixar等）和两款LLM判定器（Cisco、Nova）以及朴素代理模型进行对比。性能：在SkillJect基准上，动态审计器以2%误报率检测97%攻击；在野外技能上检测87%攻击，且对两种伪装保持稳定；相较于静态扫描器，后者在伪装下检测率骤降至≤10%；动态审计器每技能耗时约153s，低于静态扫描器的约21s。

**⚠️ 局限性**

限制：①自然语言覆盖缺口——若代理未执行某条语义路径，恶意行为将不被触发；②沙箱环境依赖——缺失的外部服务或不匹配的资源导致误判或漏检；③对极端抗分析技术（如检测沙箱并规避）尚无完整防御；④性能开销相对静态扫描器更高，适合离线或一次性审计；⑤无法完全覆盖所有多阶段或交互式技能，需要进一步的强制执行与多环境模拟。

---

## 567. PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation

**arXiv ID:** 2607.02515 | [PDF](https://arxiv.org/pdf/2607.02515v1)

**作者:** Haofei Xu `[一作]` (Google), Michael Niemeyer `[通讯]` (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Pixel‑Space Diffusion Transformer（PointDiT），直接在原始点地图像上进行单目几何估计，完全去掉VAE和复杂混合网络。

**💡 创新点**

创新点在于：①在像素空间直接训练ViT生成点地图；②采用x‑prediction流匹配目标；③用预训练DINOv3特征条件化，证明VAE‑free方法能获得更锐利的几何细节。

**🔧 技术方法**

使用技术包括：Flow Matching框架、纯ViT结构、x‑prediction目标、logit‑normal噪声调度（含0点采样）、DINOv3特征注入以及单步/多步ODE推理。

**📊 数据集**

仅使用合成数据预训练与微调（SceneNet‑RGBD、Hypersim、VKITTI2、UrbanSyn、Synscapes、TartanAir、OmniWorldGame等约1.5‑6.2M样本），在7个真实世界数据集上进行零样本评估。

**📈 对比分析**

与GeometryCrafter、PPD、Depth Pro等基线对比，单步推理即可达到或超过前者，在点图/深度指标上取得领先，并在BF1边界锐度上最高；推理速度显著更快。

**⚠️ 局限性**

局限性：仅在固定分辨率（256×256/512×512）训练，难以适应混合分辨率；户外场景表现仍有提升空间；当前模型仅输出几何，可进一步扩展多模态输出。

---

## 568. ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning

**arXiv ID:** 2607.02509 | [PDF](https://arxiv.org/pdf/2607.02509v1)

**作者:** Yanjun Zhao `[一作]` (University of Illinois Urbana Champaign), Jingrui He `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的推理方法——Recursive Evidence Replay（ReContext），利用LLM内部注意力生成相关证据池，递归选取证据并在最终生成前重播，以此在保留完整上下文的同时提升长上下文推理效果。

**💡 创新点**

创新点在于将证据组织与答案生成分离，完全不依赖训练、外部记忆或上下文裁剪；通过递归使用模型自身的相关性信号构建证据池，并在推理过程中多轮重播，实现对关键证据的显式强化；同时给出了关联记忆视角的理论解释。

**🔧 技术方法**

使用技术包括：基于末尾查询提示的注意力相关性评分、token级候选选择、句子/短语层级证据实体化、递归证据池更新、在最终生成前重播证据；实验基于Qwen3-4B/8B、Llama3-8B三大LLM；理论分析采用关联记忆模型证明隐藏表示与答案的余弦相似度递增。

**📊 数据集**

实验数据集覆盖八个128K上下文基准：Natural Questions、TriviaQA、HotpotQA、PopQA、NarrativeQA、InfBench QA、InfBench MC 与 CLIPPER（长文本断言验证）。

**📈 对比分析**

与 Vanilla、AttnSharp、DySCO、A‑MEM、DAC 等基线比较，ReContext 在三大后端均获得最低平均排名；整体准确率从 0.24 提升到 0.30，提升幅度 24.6%；在不同上下文长度、是否开启中间推理时均保持领先或竞争优势。

**⚠️ 局限性**

局限性包括：需访问模型内部注意力/相关性信息，无法直接用于不公开这些信息的闭源 API；增加读取‑重播步骤导致推理延迟略高，虽然内存占用接近 Vanilla，但仍需额外计算。

---

## 569. GeoMix: Descriptor-Free Visual Localization via Global Context and Multi-Detector Training

**arXiv ID:** 2607.02486 | [PDF](https://arxiv.org/pdf/2607.02486v1)

**作者:** Yejun Zhang `[一作]` (Aalto University), Juho Kannala `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 GeoMix，一种全新的基于几何的 2D‑3D 匹配框架，通过方向与距离嵌入、可学习全局上下文节点以及混合训练显著提升几何判别性，实现高精度、低存储、隐私友好的视觉定位。

**💡 创新点**

三项创新：① 在局部邻域中加入方向与距离信息的嵌入，强化局部几何辨识；② 设计可学习的全局上下文节点，利用跨注意力实现全局信息聚合与重分配；③ Mix‑Training 让模型在多检测器的几何空间上联合训练，提升对未见检测器的零样本泛化。

**🔧 技术方法**

技术包括：ResNet‑风格特征编码器、方向+距离嵌入、Annular Convolution 与 Max‑Pooling 的双分支 GNN、可学习全局上下文节点、Self‑/Cross‑Attention、Optimal Transport 匹配、基于学习的 Outlier Rejection，以及多检测器混合训练策略。

**📊 数据集**

使用 MegaDepth、Cambridge Landmarks、7Scenes、Aachen Day‑Night 四大公开数据集进行训练与评测。

**📈 对比分析**

在匹配与定位指标上与现有 descriptor‑free 方法（GoMatch、DGC‑GNN、A2‑GNN 等）进行对比；GeoMix 在 MegaDepth 上 AUC@5/10px 提升 11%+，75% 旋转误差下降 89%，平移误差下降 90%；在 Cambridge、7Scenes 与 Aachen 上取得 descriptor‑free 领域最佳结果，存储仅 69 MB、每帧匹配耗时约 0.115 s。

**⚠️ 局限性**

局限性：与 descriptor‑based 方法仍存在精度差距，尤其在高离群率场景下几何特征单独不足；3D 地图仍采用固定检测器构建，可能带来采样偏差；未引入轻量外观信息，未来可尝试融合外观与几何以进一步缩小差距。

---

## 570. EAGLE-360: Embodied Active Global-to-Local Exploration in 360$^\circ$

**arXiv ID:** 2607.02479 | [PDF](https://arxiv.org/pdf/2607.02479v1)

**作者:** Jingtao Xu `[一作]` (Zhejiang University), Yawei Luo `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了EAGLE-360框架，完成从全景图的全局感知到目标定位的全局到局部主动搜索。

**💡 创新点**

创新点在于将RoPE Rolling位置编码与逐步投影工具结合，形成全景感知的全局–局部探索策略，并通过GRPO强化学习实现高效多步推理。

**🔧 技术方法**

使用了RoPE Rolling、Equirectangular‑to‑Perspective投影工具、链式推理（CoT）和SFT+GRPO训练管线，辅以4K全景图的高分辨率特征。

**📊 数据集**

构建了自制EAGLE-360数据集（14k+ 4K全景图 + 70k VQA对话）并在H*Bench上进行零样本迁移评估。

**📈 对比分析**

在EAGLE-360基准上与多款开源与专有模型对比，准确率从基准模型的8.33%提升至64.44%，GCD<50°达94.72%，在H*Bench零样本取得56.1/74.8/28的高分。

**⚠️ 局限性**

局限性包括：超长上下文导致推理延迟，极端极地区域因投影失真仍存在定位误差。

---

## 571. Adoption and Ecosystem Health: A Longitudinal Analysis of Open-Source Multi-Agent Frameworks

**arXiv ID:** 2607.02453 | [PDF](https://arxiv.org/pdf/2607.02453v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 572. TestEvo-Bench: An Executable and Live Benchmark for Test and Code Co-Evolution

**arXiv ID:** 2607.02469 | [PDF](https://arxiv.org/pdf/2607.02469v1)

**作者:** Jiale Amber Wang `[一作]` (University of Waterloo), Pengyu Nie `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个可执行、实时的测试与代码共进化基准 TestEvo‑Bench，包含生成新测试和更新现有测试两条轨道。

**💡 创新点**

创新点在于将共进化对构造成可复现的任务，采用执行级别的标注和指标，并支持时间戳过滤以防模型泄露。

**🔧 技术方法**

技术上采用了三阶段矿取–清洗–包装流水线，结合动态测试依赖、覆盖率与突变测试来评估生成或更新的测试。

**📊 数据集**

数据集来源于公开的 Java Maven 开源项目，经过筛选后生成了 BenchGen‑Tasks 和 BenchUpdate‑Tasks 两个任务集合。

**📈 对比分析**

实验对四个工业/学术级 LLM + harness 组合进行比较，最佳成功率分别达到 Gen‑Best‑Success 和 Update‑Best‑Success；但在最近数据和预算受限时性能显著下降。

**⚠️ 局限性**

局限包括仅支持 Java/Maven，评估成本高且仅覆盖四个模型，后续需扩展语言、工具与更开放的模型。

---

## 573. Neuron-Aware Data Selection for Annotation-Free LLM Self-Distillation

**arXiv ID:** 2607.02460 | [PDF](https://arxiv.org/pdf/2607.02460v1)

**作者:** Zhuowei Chen `[一作]` (University of Pittsburgh), Xiang Lorraine Li `[通讯]` (University of Pittsburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种完全无监督的 LLM 后期自我提升框架 Neuron‑OPSD，利用模型内部神经元激活信息进行数据筛选和教师上下文构造，并通过 on‑policy distillation 实现自我改进。

**💡 创新点**

创新点在于①用神经元激活稀疏度（Neuron Consensus）作为无标签的可靠性评估指标；②用神经元重叠度（Neuron Overlap）检索相似推理路径作为高质量的 few‑shot 教师上下文；③将这两者与 OPD 结合，形成完全无监督的自我学习循环。

**🔧 技术方法**

技术包括：内部激活提取、基于神经元激活集合的 Consensus 与 Overlap 计算、Jaccard 最近邻检索、EMA 维护教师模型、Token‑level reverse‑KL on‑policy distillation。

**📊 数据集**

实验使用了三大基准：SciKnowEval（生物、材料、物理、化学四个子域）、Edu‑Feedback（二分类反馈质量）以及 MMLU‑Pro（多域多选）。

**📈 对比分析**

与 SFT、GRPO 及 Intuitor 等无监督对比，Neuron‑OPSD 在大多数源域取得更高的 Avg@8，且保持或提升交叉域泛化与校准（ECE 下降），尤其在 Mat.、Phys. 与 Edu. 域表现突出。

**⚠️ 局限性**

局限包括：在 Bio. 与 Chem. 领域几乎无提升；激活计数本身不够决定性，需进一步平衡自监督与信号可靠性；Neuron Overlap 对推理模式均质的域效果有限；仅在 4B 参数模型上验证，尚未测试更大模型。

---

## 574. When Do LLM Personas Support Visualization Design? A Cross-Model Study of Color Assignment and Chart Choice

**arXiv ID:** 2607.02455 | [PDF](https://arxiv.org/pdf/2607.02455v1)

**作者:** Shahreen Salim `[一作]` (Stony Brook University), Klaus Mueller `[通讯]` (Stony Brook University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在可视化设计的两个子任务中，探讨了使用 LLM personas（基于 Big Five 个性特征的提示）模拟用户人格对颜色选择和图表偏好的影响。

**💡 创新点**

创新点在于系统性评估人格与视觉输出之间的关联，揭示模型配置、概念类型（抽象 vs. 具体）以及聚合策略对人格效应的调节作用，指出人格效应并非普适存在。

**🔧 技术方法**

主要技术包括：GPT‑4o‑mini、GPT‑4.1‑mini 与 GPT‑5‑mini 三个模型；Mantel 置换检验、方差分解（η²）、多重聚类（Majority Vote、IRV、Borda 等）以及 Kendall τ 比较。

**📊 数据集**

使用数据集：43 个独特的 Big Five 人格配置（从 Rentfrow 等州级数据抽取）以及 12 种图表 idiom 与 3 个任务情境（层级、时间序列、比较）对应的高分辨率图表。

**📈 对比分析**

通过 Mantel 相关评估人格距离与颜色分布距离的关联，方差分解比较模型与人格对色彩变异的贡献，Kendall τ 评估不同模型和聚类下的图表排名一致性。结果显示：抽象概念下人格与颜色的耦合显著；图表排名在聚类后相对稳定，但无人格基线显示大多数排名由上下文驱动。

**⚠️ 局限性**

限制包括：未进行人类验证；样本量相对较小；GPT‑5 的多重视角采样方法与温度采样不同，可能影响可比性；模型配置差异导致人格效应不可泛化；对不同文化/语言的迁移性未测试。

---

## 575. Alignment Is All You Need For X-to-4D Generation

**arXiv ID:** 2607.02516 | [PDF](https://arxiv.org/pdf/2607.02516v1)

**作者:** Qiaowei Miao `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Align4D 框架，实现任意模态（文本、图像、视频、3D）输入到动态 4D 对象生成；

**💡 创新点**

创新点包括对象距离对齐（VAOD/MAOD）、动作-几何联合对齐（MGJA）以及异步优化（AO）机制，结合 X4D 四模态数据集。

**🔧 技术方法**

使用预训练的视频与 3D 扩散模型、Score Distillation Sampling（SDS）优化、对象距离搜索、Mask 辅助、异步优化等技术。

**📊 数据集**

构建并使用 X4D 数据集（prompt‑image‑video‑3D 四模态对）以及 Consistent4D 进行评估。

**📈 对比分析**

与 L4GM、SC4D、STAG4D、DG4D、4Diffusion、Free4D 等基线比较，在 X4D 上人类评估与 VBench 取得最高分，在 Consistent4D 上 PSNR/SSIM/FVD/CLIP 等指标均优于竞争方法。

**⚠️ 局限性**

受限于依赖预训练扩散模型生成的视频‑3D 对，透明或快速闪烁材质效果不足，且数据集依赖生成模型可能带来分布偏差。

---

## 576. What LLM Agents Say When No One Is Watching: Social Structure and Latent Objective Emergence in Multi-Agent Debates

**arXiv ID:** 2607.02507 | [PDF](https://arxiv.org/pdf/2607.02507v1)

**作者:** Arman Ghaffarizadeh `[一作]` (Independent Researcher), Shahriar Noroozizadeh `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无显式目标的社交结构环境下，研究LLM代理在公开渠道和离线（OTR）渠道下的表达差异，并提出双通道评估框架。

**💡 创新点**

创新点在于通过OTR对比公开表达捕捉隐含目标，证明社会关系结构能引发代理输出偏差，并提供多维度（立场、语义相似度、NLI、问卷）测度。

**🔧 技术方法**

技术方法包括：使用多种大语言模型生成公开和OTR回应；构造带角色与关系上下文的情景；计算立场差异、余弦相似度、自然语言推理（NLI）标签和结构化问卷；对比不同模型与情境下的表现。

**📊 数据集**

数据集与实验规模：10种LLM模型（含GPT、Gemini、Grok、GLM等）；3种二分决策情景（晋升、议案、稿件提交）；每个情景5种关系上下文变体；共750次实验（10模型×3情景×5变体×5复测）。

**📈 对比分析**

比较方法：对每个模型/情景/变体，统计公开与OTR在立场、语义相似度、NLI以及问卷分数上的差异率。结果显示，在对齐诱导情境下，受影响代理α的立场差异从约3%升至≈40%；语义一致性下降，NLI中的对立比例上升。不同模型表现存在明显异质性。

**⚠️ 局限性**

局限性：OTR仅为对比输出，不等同于内部信念或目标；结果高度依赖模型架构、情境语义与时间框架；缺乏干预/修正方案，研究仅诊断性；在真实部署中多样化角色、长历史和模糊关系可能导致更复杂的行为。

---

## 577. Controllable Sim Agents with Behavior Latents

**arXiv ID:** 2607.02496 | [PDF](https://arxiv.org/pdf/2607.02496v1)

**作者:** Juanwu Lu `[一作]` (Purdue University), Ziran Wang `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了可控的离线交通仿真框架 CNeVA，能够从日志数据中推断每个车辆的行为潜变量并通过混合通道掩码的流匹配生成器实现多维可控轨迹生成。

**💡 创新点**

创新点包括：① 通过闭式共轭后验以高斯行为潜变量和每通道折扣回报为基础实现可解释的可控性；② 混合通道掩码的 classifier‑free guidance 训练策略；③ 软可及性门和对比条件以提升安全性与物理可行性。

**🔧 技术方法**

采用了变分推理、流匹配生成、混合通道掩码的 classifier‑free guidance、软可及性门、对比条件等技术。

**📊 数据集**

在 Waymo Open Motion Dataset (WOMD) 上进行评估。

**📈 对比分析**

与基准闭环/token化模仿模型对比，CNeVA 在 WOSAC 真实性评测中获得 0.7145 的 meta‑metric，minADE 1.80 m，且在通道可控性指标 CSM 上表现出可调节的速度、加速度、安全与地图遵循。

**⚠️ 局限性**

局限包括：地图可控性对回报定义高度敏感，物理安全性在高幅度可控范围内可能衰减，且仍需改进对稀疏语义回报的可控性。

---

## 578. Probabilistic Memory for Trustworthy Edge Intelligence

**arXiv ID:** 2607.02465 | [PDF](https://arxiv.org/pdf/2607.02465v1)

**作者:** Likai Pei `[一作]` (University of Notre Dame), Ningyuan Cao `[通讯]` (University of Notre Dame)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种统一的概率存储单元 p‑MEM，用以在边缘智能中高效进行概率计算。

**💡 创新点**

创新点在于将分布参数与随机数生成器集成到内存近端，实现了在原生内存带宽下直接采样，显著降低了指令计数、延迟与能耗。

**🔧 技术方法**

采用近内存模拟器 pMEMSim、模拟 SRAM/NRAM/FeRAM 等单元，并集成模拟/数字随机数发生器与 ADC，在 CPU/GPU 上实现 BNN、PCME、DP 等工作负载。

**📊 数据集**

使用 CIFAR‑10 进行 BNN 测试，MSR‑VTT 用于 PCME，差分隐私（DP）测试采用通用数据集。

**📈 对比分析**

通过与传统软件 RNG 的 CPU/GPU 对比，p‑MEM 在 CPU 上实现 BNN 的指令量降至原来的 0.46 倍，延迟提升 562 倍，能耗降低 295 倍；GPU 上相应提升约 4.2×、3.5×，PCME 与 DP 同样表现出类似优势。

**⚠️ 局限性**

限制在于需要在内存上额外增加 RNG、ADC 等硬件，对非高斯分布支持有限，且不同单元实现可能导致面积与功耗偏高。

---

## 579. OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers

**arXiv ID:** 2607.02461 | [PDF](https://arxiv.org/pdf/2607.02461v1)

**作者:** Donghyun Lee `[一作]` (Cantina Labs), Saurabh Shukla `[通讯]` (Cantina Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 OrbitQuant，一种无校准的低位量化方法，用于扩散变压器（DiT）在图像与视频生成任务中的权重和激活量化，保持甚至超越 FP16 的生成质量。

**💡 创新点**

创新点在于：①使用随机置换块哈达玛变换（RPBH）旋转激活，使其分布固定为已知分布；②共享单一 Lloyd–Max 代码本，无需针对每个时间步或提示重新校准；③将旋转嵌入权重，使激活旋转在前向推理中与权重旋转相互抵消，提升效率；④该方案在图像与视频模型上统一使用，无需额外调参。

**🔧 技术方法**

技术手段包括：随机置换块哈达玛变换、正交旋转、归一化、Lloyd–Max 代码本、权重与激活共享的旋转基底、线性层分解与量化、以及对 AdaLN 模块的 INT4 处理。

**📊 数据集**

实验使用 FLUX.1-schnell、FLUX.1-dev、Z-Image-Turbo 进行图像生成评估；Wan 2.1‑1.3B、CogVideoX‑2B 进行视频生成评估；评价指标为 GenEval（图像）和 VBench（视频）上的分数。

**📈 对比分析**

与 SVDQuant、AdaTSQ、ViDiT‑Q、Q‑DiT、QuaRot、SmoothQuant 等基线在 W4A4、W2A4 等低位宽设置下对比，OrbitQuant 在 W4A4 处与 FP16 差距≤0.1，W2A4 仍能产生可用图像且是唯一不降噪的方法；同时在推理时延与显存占用方面表现最优，开销最低。

**⚠️ 局限性**

局限性包括：① AdaLN 模块仍保持 INT4 处理，无法进一步压缩；② 在极低位宽（如 W2A4）下仍对激活分布的鲁棒性有一定依赖；③ 对非常大维度需要构造 RPBH 旋转，可能影响效率；④ 目前仅验证于 FLUX、Z‑Image‑Turbo、Wan、CogVideoX，需进一步验证在更大规模模型和多种任务上的稳定性。

---

## 580. G-RRM: Guiding Symbolic Solvers with Recurrent Reasoning Models

**arXiv ID:** 2607.02491 | [PDF](https://arxiv.org/pdf/2607.02491v1)

**作者:** Timo Bertram `[一作]` (Johannes Kepler University Linz), Günter Klambauer `[通讯]` (Johannes Kepler University Linz)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

G‑RRM将神经网络生成的完整解建议与符号求解器结合，以提高约束满足问题求解效率。

**💡 创新点**

结合符号等价循环Transformer（SE‑RRM）与符号求解器的神经符号指导，证明在特定条件下能显著降低冲突数并加速求解。

**🔧 技术方法**

使用循环Transformer（SE‑RRM）作为神经解算器，并将其输出作为回溯和 CDCL SAT 求解器（Glucose 4.1、CaDiCaL 3.0） 的分支提示。

**📊 数据集**

主要使用 Sudoku 9×9、25×25 等多尺寸约束满足实例进行实验。

**📈 对比分析**

与传统回溯和现代 SAT 求解器对比，G‑RRM 在 9×9 Sudoku 上使回溯速度提升 33.3×，Glucose 4.1 提升 1.70×，在 25×25 上保持 1.17× 的加速，而 CaDiCaL 由于不允许覆写提示而几乎无提升。

**⚠️ 局限性**

仅在搜索空间大且求解器能动态覆写分支时有效；对不允许覆写提示的求解器（如 CaDiCaL）无显著收益；依赖神经网络的准确性，且无法单独保证解的正确性。

---

## 581. Online Safety Monitoring for LLMs

**arXiv ID:** 2607.02510 | [PDF](https://arxiv.org/pdf/2607.02510v1)

**作者:** Mona Schirmer `[一作]` (University of Amsterdam), Eric Nalisnick `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于阈值的实时LLM安全监控框架，并通过风险控制（期望控制与高概率控制）对阈值进行校准；

**💡 创新点**

创新点在于利用风险控制方法为监控阈值提供严格的统计保证，并证明该简单方法在数学推理和红队对话场景中与更复杂的 e‑valuator 竞争力；

**🔧 技术方法**

主要技术包括风险控制框架（CRC、UCB）、外部验证器的概率信号、生成器自身的 token log‑prob 作为信号、以及与 e‑valuator 的对比实验；

**📊 数据集**

实验使用的主要数据集包括 MATH、Anthropic Red Teaming、FineHarm，以及使用 Claude Haiku、Mistral‑7B‑Instruct 等模型生成的序列；

**📈 对比分析**

与 e‑valuator 对比时，CRC 与 UCB 在误报率控制上满足理论保证，且检测延迟更低，功效在部分场景与 e‑valuator 相当；

**⚠️ 局限性**

主要局限是仅使用单一时间不变阈值，受限于验证器信号的质量，且忽略了信号随时间变化的结构特征。

---

## 582. From SRA to Self-Flow: Data Augmentation or Self-Supervision?

**arXiv ID:** 2607.02508 | [PDF](https://arxiv.org/pdf/2607.02508v1)

**作者:** Dengyang Jiang `[一作]` (Hong Kong University of Science and Technology), Jingdong Wang `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并提出了一种基于注意力分离的自我表示对齐方法，通过双时隙噪声调度与注意力分离实现噪声状态与部分视图的数据增强；

**💡 创新点**

创新点在于证明双时隙调度的提升主要来自噪声状态数据增强而非交互自监督，并将注意力分离既抑制交互又产生部分视图增强，构成更强的自我对齐训练框架；

**🔧 技术方法**

使用了扩散变换器（DiT/SiT）、EMA自监督对齐、双时隙噪声调度、注意力分离掩码、Classifier-free Guidance以及FID、sFID、IS等评估指标；

**📊 数据集**

实验数据集为 ImageNet 的 256×256 与 512×512 版本；

**📈 对比分析**

与 vanilla DiT/SiT、REPA、SRA、Self-Flow 等基线在 FID、sFID、IS、精确率/召回率等指标对比，4M 步时 FID 下降至 1.44（256）/2.08（512），IS 提升至 315.3（256）/282.7（512），优于或与外部编码器方法相当；

**⚠️ 局限性**

局限性包括仍需大量 GPU 训练；在高分辨率下对齐层比例与掩码比例需要调优；缺乏在多模态（文本‑图像/视频/音频）通用性的进一步验证。

---

## 583. Seek to Segment: Active Perception for Panoramic Referring Segmentation

**arXiv ID:** 2607.02497 | [PDF](https://arxiv.org/pdf/2607.02497v1)

**作者:** Song Tang `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了主动全景定位分割（APRS）任务，并开发了具备EgoSphere空间视觉记忆的PanoSeeker智能体；

**💡 创新点**

创新点在于将EgoSphere显式空间视觉记忆与基于RL的高效策略相结合，使代理在连续360°环境中实现非冗余、基于语言的主动探索与分割；

**🔧 技术方法**

采用Qwen3‑VL‑8B视觉‑语言模型、gnomonic投影构建EgoSphere、LoRA微调、GRPO强化学习、SAM‑3语义分割等技术；

**📊 数据集**

构建了APRS基准数据集，包含4,971个全景场景、7,420条带四类空间指令（EGO、UNIQ、ALLO、MULTIHOP）的样本，并收集专家标注的搜索轨迹；

**📈 对比分析**

与静态RIS、启发式扫描和基线VLM代理对比，PanoSeeker在成功率75.4%、mIoU55.8%以及SPL0.57等指标上均显著优于现有方法；

**⚠️ 局限性**

局限性包括：搜索步数上限为20步、对全景图像投影与记忆分辨率依赖较高、在极其复杂或长距离推理场景下仍可能出现死循环或效率下降。

---

## 584. Human Capital, Not Model Benchmarks, Predicts Hybrid Intelligence in Forecasting

**arXiv ID:** 2607.02467 | [PDF](https://arxiv.org/pdf/2607.02467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 585. Visually Grounded Self-Reflection for Vision-Language Models via Reinforcement Learning

**arXiv ID:** 2607.02490 | [PDF](https://arxiv.org/pdf/2607.02490v1)

**作者:** Liyan Tang `[一作]` (University of Texas at Austin), Greg Durrett `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 VRRL 框架，结合随机转折掩蔽和缓冲回放训练视觉语言模型进行多轮自我反思和错误纠正。

**💡 创新点**

创新点在于设计两种专门机制：随机转折掩蔽让模型仅更新错误后状态的梯度；缓冲回放让模型从历史失败状态开始继续学习恢复，显著提升视觉反馈下的自我纠错能力。

**🔧 技术方法**

采用强化学习（GRPO）与自监督微调、反射奖励、视觉回放缓冲等技术，形成从 SFT 到 RL 的完整训练流程。

**📊 数据集**

使用合成表格/图表（小表、Bar Chart、Scatter Plot）和 FrozenLake 空间导航地图（6×6、7×7）作为 OOD 评测数据集。

**📈 对比分析**

与零射、SFT、单/多轮 RL、VL‑Rethinker、Reflection Tuning 等基线对比，VRRL 在 OOD 任务平均提升 10–25% 以上，保持 In‑Distribution 准确率接近 100%。

**⚠️ 局限性**

局限性：依赖环境提供视觉反馈，难以直接应用于无视觉回馈的 VQA 等任务；实验仅在 Qwen 系列 3B–7B 模型上，未验证更大规模或不同架构的泛化效果。

---

## 586. Combating Textual Noise and Redundancy: Entropy-Aware Dense Visual Token Pruning

**arXiv ID:** 2607.02484 | [PDF](https://arxiv.org/pdf/2607.02484v1)

**作者:** Xuehui Wang `[一作]` (Shanghai Jiao Tong University), Wei Shen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 Entropy-Aware Dense Visual Token Pruning (EADP) 框架，对视觉模型的所有图像补丁进行可解释且高效的稀疏化。

**💡 创新点**

创新点在于：① 用统计熵度量并过滤文本噪声，得到细粒度指令相关性评分；② 将 token 选择转化为带空间先验的子模最大化（Facility Location），从而消除冗余、保证整体覆盖。

**🔧 技术方法**

技术手段包括：跨模态密集相似度计算、熵驱动去噪与加权聚合、Gaussian 平滑 + 指数极化、子模最大化采样、动态分配预算、以及对视觉相似度进行非负映射。

**📊 数据集**

实验覆盖 10 个视觉问答与多模任务（VQAv2、GQA、VizWiz、ScienceQA、TextVQA、POPE、MME、MMBench、MMBench-CN、MM‑Vet），视频任务（MVBench、LongVideoBench、Video‑MME），以及文档与图表任务（DocVQA、InfoVQA）。

**📈 对比分析**

与 FastV、PyramidDrop、SparseVLM、LLaVA‑Prumerge、VisionZip、HiPrune、DART、DivPrune、TRIM、CDPruner 等现有 pruning 方法对比，EADP 在 LLaVA、Qwen、LLaVA‑Video 等多模型、多分辨率设定下均实现了接近全标的准确率，并在 80% 以上压缩率时仍保持 1–3 分的优势，整体性能达到了 SOTA。

**⚠️ 局限性**

局限性：① 需要计算完整的文本‑视觉相似度矩阵，导致在极大图像尺寸或算力受限环境下的计算开销；② 对 CLIP 文本编码器的依赖，若文本特征质量不足会影响去噪效果；③ 在极低 token 预算（≤32）时仍可能出现细节丢失，需进一步优化。

---

## 587. QuadRocket: An Aerial Robotic Testbed for Adaptive Thrust-Vector Control of Rocket-Like Vehicles

**arXiv ID:** 2607.02474 | [PDF](https://arxiv.org/pdf/2607.02474v1)

**作者:** Pedro Santos `[一作]` (Universidade de Lisboa), Carlos Silvestre `[通讯]` (University of Macau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了QuadRocket，一种将四旋翼无人机与圆柱体结构通过万向关节耦合，形成类似火箭推力向量控制（TVC）的低成本、低风险空中机器人平台，并设计了面向该平台的全局轨迹跟踪控制器，实现对未知恒定扰动的自适应补偿。

**💡 创新点**

创新点主要包括：1）首次将四旋翼视为推力向量执行器，构造可模拟火箭TVC动力学的简化单刚体模型；2）采用二维球面（𝕊²）减少姿态表示，解耦偏航与推力向量方向；3）基于自适应反步和控制点变换，提供几乎全局的轨迹跟踪和非最小相位行为抑制；4）将动态表面控制与四旋翼姿态跟踪相结合，避免显式求导，提高鲁棒性；5）通过实验室运动捕捉验证平台和控制算法的有效性。

**🔧 技术方法**

使用的技术包括：自适应反步控制、几何控制（减姿态表示在二维球面）、控制点变换以消除非最小相位、动态表面控制、低通滤波处理期望推力向量、基于向量误差的Lyapunov分析、以及基于运动捕捉的实验验证。

**📊 数据集**

文中未使用传统机器学习数据集，实验数据主要来自室内运动捕捉系统（VICON）对QuadRocket的位姿和速度的高精度实时测量，用于在线控制与误差评估。

**📈 对比分析**

对比方法：在仿真与实验中与基准线性控制或经典PID控制器进行对比。结果表明，QuadRocket在面对未知扰动时实现了厘米级轨迹跟踪误差、快速的姿态恢复，并且推力向量误差被限制在滤波时间常数与目标轨迹速度衍生量的乘积范围内，展示了显著的鲁棒性与精度提升。

**⚠️ 局限性**

局限性包括：1）控制器设计基于恒定扰动假设，面对强时变扰动时性能可能下降；2）单刚体模型忽略了连接关节的动态与柔性效应，实际系统在极限运动时可能出现非最小相位残余；3）实验验证仅在室内有限空间，未展示大范围飞行或多机协同；4）对运动捕捉依赖度高，实际应用需引入自估计或惯性测量单元以实现自足控制。

---

## 588. LACUNA: A Testbed for Evaluating Localization Precision for LLM Unlearning

**arXiv ID:** 2607.02513 | [PDF](https://arxiv.org/pdf/2607.02513v1)

**作者:** Matteo Boglioni `[一作]` (Mila Quebec Artificial Intelligence Institute), Verna Dankers `[通讯]` (Mila Quebec Artificial Intelligence Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并实现了一个基于掩码持续预训练的LLM unlearning测试平台，通过在预训练模型中注入合成PII并在特定参数上实现ground-truth定位，随后评估现有unlearning方法的参数级定位精度。

**💡 创新点**

核心创新在于首次构造具有真实参数级知识定位标注的LLM实验环境，并通过ROC AUC衡量定位精度，揭示传统方法对知识隐藏而非彻底删除的缺陷。

**🔧 技术方法**

技术手段包括掩码持续预训练、LoRA指令微调、梯度优化（NPO、Gradient Difference）以及基于定位的改造（ROME/MEMIT），并引入oracle基准进行对照。

**📊 数据集**

使用了由9,674名合成个体构成的PII数据集与相应的问答模板，混合4.3B token的预训练语料，在1B和7B OLMo模型上进行训练。

**📈 对比分析**

与三种SOTA方法（梯度式NPO、定位式ROME/MEMIT、Pythia）进行对比，结果显示尽管输出层表现良好，但定位精度普遍偏低，oracle方法在定位、遗忘效果与resurfacing攻击抵抗力上表现最佳。

**⚠️ 局限性**

局限性包括仅在合成PII与预先注入的参数上验证，真实场景中知识可能分散更难定位；实验仅针对OLMo架构，是否适用于其他模型尚待进一步验证。

---

## 589. Reasoning LLM Improves Speaker Recognition in Long-form TV Dramas

**arXiv ID:** 2607.02504 | [PDF](https://arxiv.org/pdf/2607.02504v1)

**作者:** Yuxuan Li `[一作]` (Tsinghua University), Qi Tian `[通讯]` (Huawei Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个大规模长剧人物说话者识别基准，并提出了基于大型推理模型（LRM）的多模态推理框架来实现高精度人物归属

**💡 创新点**

创新点在于（1）创建了包含520小时、144名角色的长剧语音-视觉-文本三模态数据集；（2）设计了可自适应调用语音相似度、视频描述和人物关系工具的推理流程；（3）通过RL强化学习优化推理决策，显著提升短句子和离屏语音的识别率

**🔧 技术方法**

采用了多模态大型语言模型（Qwen3‑8B）作为推理核心，结合语音特征提取（ERes2Net）、视频字幕生成（Qwen3‑VL‑32B）、面部识别与聚类、关系三元组抽取，并用GRPO进行强化学习

**📊 数据集**

使用了自研的DramaSR‑LRM数据集（约1.6万个剧集，525小时，428K条注释语句），其中10个剧集用于SFT，另10个用于RL，剩余11个用于测试

**📈 对比分析**

与基线（Label‑Propagation）、Pyannote、纯语音模型等对比，LRM在全局准确率上从85.49%提升至87.79%（+2.30%），在短句子、离屏、角色密集等困难子集的提升尤为显著；在下游视频问答任务中，准确率从21.6%提升至72.0%

**⚠️ 局限性**

主要局限包括（1）对多说话者场景处理有限，未使用语音分离；（2）依赖先验的角色候选集和30s视觉窗口，难以覆盖完全离屏的情节；（3）推理时需要较多计算资源，推理时间约0.33 GPU‑秒/句

---

## 590. VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation

**arXiv ID:** 2607.02503 | [PDF](https://arxiv.org/pdf/2607.02503v1)

**作者:** Shuai Tian `[一作]` (Chinese Academy of Sciences), Dongbin Zhao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 VT-WAM，一种同时学习视觉、触觉变形预测与动作预测的视觉-触觉世界动作模型，解决接触丰富操控中的信息不平衡问题。

**💡 创新点**

核心创新包括异向MoT注意力（Asymmetric MoT Attention）让动作预测既依赖首帧视觉上下文，又可捕捉完整触觉动态；以及基于接触阶段的动作-视觉-触觉注意力引导（AVTAG），训练时通过 hinge 损失提升触觉信息权重。

**🔧 技术方法**

采用了视觉‑触觉‑动作专家结构、异向MoT注意力、AVTAG 引导、流匹配训练目标以及视觉缓存推理模式，并基于预训练的视觉 VAE 与触觉 VAE。

**📊 数据集**

使用了六个真实世界接触丰富任务（wipe board、wipe vase、peel cucumber、insert plug、swipe card、insert tube）的演示数据，采用人类 kinesthetic 教学收集约 100 条轨迹/任务。

**📈 对比分析**

与 DP+Tactile、RDP、π_0.5、OmniVTLA、Fast‑WAM 等基线对比，VT‑WAM 在所有任务上取得最高成功率，平均提升至 71.67%，比 Fast‑WAM 提升 26.67%，比 OmniVTLA 提升 35.84%。

**⚠️ 局限性**

局限性包括：仅针对单一任务训练，缺乏跨任务通用性；推理仍需较多计算资源；触觉传感器与视觉传感器的同步与校准依赖硬件稳定性。

---

## 591. Towards Robustness against Typographic Attack with Training-free Concept Localization

**arXiv ID:** 2607.02494 | [PDF](https://arxiv.org/pdf/2607.02494v1)

**作者:** Bohan Liu `[一作]` (University of Virginia), Aidong Zhang `[通讯]` (University of Virginia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对CLIP ViT模型在文字诱导攻击（Typographic Attack）下的鲁棒性问题，提出了一种训练无关的机制解释方法，通过采样概念向量和梯度归因识别出导致文本干扰的注意力模块，并对其进行轻量化干预以提升模型鲁棒性。

**💡 创新点**

创新点在于将线性表示假设与稀疏子网络彩票假设结合，利用随机采样概念向量在MHSA子空间中快速发现词汇概念，并提出归因度量nTAS来精准定位易受攻击的注意力环路，最终实现无需额外训练即可显著提升鲁棒性的机制干预。

**🔧 技术方法**

主要技术包括随机概念向量采样、梯度归因与归一化文本归因分数（nTAS）、多头自注意力结构分析、注意力重加权和零值消融等机制解释与干预手段。

**📊 数据集**

实验使用了RTA‑100、Disentangling、PAINT、IN‑100‑Text等对象分类数据集，以及RIO‑Bench VQA数据集，并在五种CLIP ViT变体以及Qwen3‑VL、InternVL3.5、Gemma3等大型视觉语言模型上进行验证。

**📈 对比分析**

与先前的Defense‑Prefix和Dyslexify等基线相比，本文方法在所有ViT模型上在对象分类任务中实现了平均约6.1%的准确率提升，同时将文本混淆率显著降低；在VQA任务中也在大多数子集提升了约0.8–2.0个百分点，且仅导致<1%整体性能损失。

**⚠️ 局限性**

局限性包括对非ViT架构或更复杂攻击形式的泛化能力尚未充分验证，以及在极端数据噪声或高度语义冲突场景下可能仍存在鲁棒性下降。

---

## 592. Audio-Based Understanding of Audiobook Narration Appeal

**arXiv ID:** 2607.02473 | [PDF](https://arxiv.org/pdf/2607.02473v1)

**作者:** Shahar Elisha `[一作]` (Spotify), Emmanouil Benetos `[通讯]` (Queen Mary University Of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了音频书叙述音质、声学特征、类型与标题对听众吸引力的影响，并构建了多层次统计和预测模型。

**💡 创新点**

首次在大规模真实世界数据中系统关联叙述声学特征与听众消费，揭示不同类型和同一书目版本间叙述差异对吸引力的显著作用。

**🔧 技术方法**

采用预训练音频模型（eGeMAPS、YAMNet、Whisper-tiny）提取129维声学特征，使用GLM、LME、分类与排序机器学习模型进行建模。

**📊 数据集**

主要数据集为LibriVox公开音频书（8,854条记录，65类，1,206名旁白），并在3,428条Spotify托管音频书中使用更细粒度的回访率指标验证结果。

**📈 对比分析**

通过比较全局GLM、按类型GLM、混合效应模型以及多种分类/排序模型（LR、SVM、XGBoost、MLP、XGBRanker、LGBMRanker）评估性能；声学特征单独能将分类准确率提升至约0.35，排序模型在同书版本内部的Kendall τ可达0.26-0.28，均优于随机基准。

**⚠️ 局限性**

局限性包括视图率等粗糙消费指标、录音质量不一、缺乏完整的听众行为数据，导致特征相关性与因果解释受限；模型解释性仍受多重共线性和数据噪声影响。

---

## 593. Learning Agile Intruder Interception using Differentiable Quadrotor Dynamics

**arXiv ID:** 2607.02472 | [PDF](https://arxiv.org/pdf/2607.02472v1)

**作者:** Michael Anoruo `[一作]` (Carnegie Mellon University), Wennie Tabib `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种仅利用拦截器状态与目标单位方向向量学习的主动拦截控制策略

**💡 创新点**

创新点在于采用可微四旋翼动力学与分析策略梯度（APG）进行端到端训练，只需方向信息即可实现高速机动拦截

**🔧 技术方法**

使用可微仿真、四旋翼动力学模型、GRU 神经网络、分析策略梯度（APG）和 BPTT 进行训练

**📊 数据集**

利用在仿真中随机生成的椭圆、螺旋和镰形三种轨迹作为训练和评估数据集

**📈 对比分析**

与 PPO、点质量动力学（PMD）等基线对比，Quad‑APG 在拦截成功率上平均提升约 30%（最高可达 60%）并在速度上可达 10 m/s

**⚠️ 局限性**

局限性包括假设目标以恒定速度运动、未考虑检测与传感器不确定性、仅在仿真环境验证、缺乏硬件实测

---

## 594. Will Scaling Improve Social Simulation with LLMs?

**arXiv ID:** 2607.02464 | [PDF](https://arxiv.org/pdf/2607.02464v1)

**作者:** Caleb Ziems `[一作]` (Stanford University), Diyi Yang `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建大规模计算规模实验与观察性规模实验，研究大型语言模型（LLM）在社会模拟任务（意见模拟、行为模拟、纵向预测）中的表现，并探讨计算规模、通用能力与模拟真实性之间的关系。

**💡 创新点**

①首次将计算规模与观察性规模方法应用于社会模拟任务，系统量化规模提升对模拟真实性的影响；②揭示预训练数据分布不平衡导致的“弱规模”任务，说明部分社会模拟任务与通用推理能力呈正交性；③通过微调实验验证弱规模现象是否因分布不匹配而非模型能力缺陷。

**🔧 技术方法**

使用IsoFLOP计算规模法（固定 FLOPs 计算最优模型）与 log‑linear 规模拟合；采用 PCA 提取通用能力主成分并构建观察性规模关系；构建损失‑准确率校准函数；对 Qwen2.5 与 Llama3 进行微调并评估参数规模效应。

**📊 数据集**

预训练数据：DCLM Web 文本；实验数据集：世界价值观调查（WVS）7th wave、心理学实验集 Psych‑101、美国变迁生活调查（ACL）；还使用了 Qwen3、Llama、Gemma 等模型体系。

**📈 对比分析**

通过比较计算规模曲线（R² 0.85‑0.97）、观察性规模相关系数（0‑0.85）与微调后的参数规模曲线，评估模型在三类任务中的性能。结果显示，大部分任务随规模提升显著改善，尤其是意见与行为模拟；但纵向预测与低资源群体的模拟提升缓慢；微调在弱规模任务上几乎无显著提升，表明单纯规模无法完全弥补。

**⚠️ 局限性**

仅覆盖有限的离散动作/多项选择型社会模拟任务；只评估基础模型，未考虑后训练或工具扩展；规模外推不确定性高；对低资源人群的模拟受预训练偏差限制；未深入探讨模型模拟的伦理与科学有效性。

---

## 595. Language Models as Measurement Apparatus for Culture

**arXiv ID:** 2607.02459 | [PDF](https://arxiv.org/pdf/2607.02459v1)

**作者:** Kent K. Chang `[一作]` `[通讯]` (University of California), Kent K. Chang (University of California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建多种语言模型测量工具（如BERT、Gemini、Longformer、LLaMA等），对电视/电影剧本进行结构、互动和偏差三类文化测量，并通过匿名化、专属预训练和多模型工作流等实验探讨测量仪器与文化对象的相互建构；

**💡 创新点**

首次引入“代理切割”（agential cut）概念，将测量过程视为物质-话语实践，强调测量仪器与文化对象共构；

**🔧 技术方法**

使用多模态对话分解、角色归属分类、关系类型抽取等NLP技术，结合长序列编码器和熵阈值读取器；

**📊 数据集**

利用TVQA、808部电影剧本、4部电视剧400k语句、109部恢复时代戏剧集等多种文本与对话数据集；

**📈 对比分析**

与基线模型对比，使用匿名化实验、宏观F1评价和准确率差异展示性能：角色归属准确率在匿名化后大幅下降，熵阈值读取器在专属预训练后宏观F1提升至0.44；

**⚠️ 局限性**

存在仪器与文化内容深度交织导致难以分离模型偏差、数据覆盖不足、模型长程与多模态处理受限，以及跨文化推广的可行性有限。

---

## 596. WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory

**arXiv ID:** 2607.02517 | [PDF](https://arxiv.org/pdf/2607.02517v1)

**作者:** Hanlin Wang `[一作]` (Hong Kong University of Science and Technology), Qifeng Chen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计了一个可控的视频世界模拟框架，利用LLM规划3D轨迹并通过位置与外观条件实现长时序视频生成中的持久动态记忆；

**💡 创新点**

创新点在于将语义运动规划与视觉合成彻底解耦，提出“Appearance Binding”机制实现对象永久性与外观一致性，并通过LLM实现开放式事件与相机视角控制；

**🔧 技术方法**

主要技术包括LLM（Gemini/ChatGPT）轨迹规划、3D VAE+Diffusion Transformer（LingBot-World-Base）条件去噪、位置与外观双条件编码、空间感知跨注意力、时间抖动机制、上下文检索与相机投影；

**📊 数据集**

使用自制游戏平台生成的15秒视频（采用SAM3追踪、Qwen2.5-VL-72B生成文字描述）作为训练集，视频分辨率832×480@16fps；

**📈 对比分析**

与Yume、HY-World、Infinite-World、LingBot-World-Fast、HyDRA等基线对比，评估指标包括PSNR、SSIM、LPIPS、VBench主体/背景一致性、动态主体一致性，实验显示本文方法在所有重建指标上均优于基线，尤其在动态一致性上表现突出；

**⚠️ 局限性**

局限性在于过度依赖合成游戏数据导致域差距，导致部分视觉细节不够真实（如不自然运动或模糊面部），未来需要引入真实世界数据以提升视觉逼真度。

---

## 597. Distributed Attacks in Persistent-State AI Control

**arXiv ID:** 2607.02514 | [PDF](https://arxiv.org/pdf/2607.02514v1)

**作者:** Josh Hills `[一作]` (Constellation Astra Fellowship), Asa Cooper Stickland `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在持久代码库中 AI 编码代理的安全控制，构建了一个多 PR 的任务序列实验框架，并对比了逐步（gradual）与一次性（non‑gradual）攻击的可探测性。

**💡 创新点**

创新点包括：①引入“持久状态攻击”概念，展示逐步攻击能利用 PR 之间的自然覆盖；②设计了状态化 diff 监控器 Link‑Tracker，能够跨 PR 跟踪可疑线索；③将多模态监控（diff 与 trajectory）组合成四监控集成，显著降低 evasion。

**🔧 技术方法**

技术手段：使用 LLM 作为攻击者（Claude Sonnet 4.5、Gemini 3.1 Pro、Kimi K2.5）和监视器（GPT‑4o、GPT‑5.x、DeepSeek 等）；diff 监控、trajectory 监控、Link‑Tracker（状态化 diff）、四监控集成；采用 98th‑percentile 阈值校准，并测量 evasion 率。

**📊 数据集**

数据集：共 20 个任务变体，包含 5 条 CLI 工具序列和 5 条 Flask Web 服务序列，每条序列 5–6 PR；每个序列配备 2 种侧任务（网络 exfil、环境变量 exfil、认证后门、Webhook 集成）。

**📈 对比分析**

对比方法：在每个序列和侧任务下分别跑逐步攻击与位置最佳一次性攻击，记录成功率与 evasion；比较不同监控器（task‑description diff、full‑diff、Link‑Tracker、trajectory、ensemble）。结果显示：标准 diff 在逐步攻击下 evasion 高达 93%；Link‑Tracker 将其降至 54%；四监控集成将逐步攻击 evasion 降至 47%；在更强监控模型下，evasion 仍高达 31%。

**⚠️ 局限性**

局限性：任务规模小，未覆盖大型企业代码库；实验仅使用单一 PR 交互式模型，未充分利用跨 PR 记忆；Link‑Tracker 与集成虽然提升检测，但仍有约 47% 的成功逐步攻击漏检；实验对不同模型的可推广性仍待进一步验证。

---

## 598. Program-as-Weights: A Programming Paradigm for Fuzzy Functions

**arXiv ID:** 2607.02512 | [PDF](https://arxiv.org/pdf/2607.02512v1)

**作者:** Wentao Zhang `[一作]` (University of Waterloo), Yuntian Deng `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将自然语言描述的模糊函数编译成小型神经网络权重（PAW）并在本地可执行的程序范式。

**💡 创新点**

创新点包括：① 混合程序结构（离散伪代码 + 连续PEFT）；② 单次编译即可生成可离线复用的权重；③ 通过大模型编译器与冻结轻量解释器的组合，在不调用外部API的情况下实现与大模型相当的性能；④ 在多模态任务上仅更换编译器即可实现图像条件的模糊函数。

**🔧 技术方法**

使用的技术：文本到LoRA编译器（4B Qwen3模型），伪编译器（未训练的4B Qwen3-Instruct），LoRA映射器（共享基底 + 低秩投影），冻结的 Qwen3‑0.6B 或 GPT‑2 解释器，量化技术（Q4_K_M、Q6_K）和多任务数据集 FuzzyBench‑10M。

**📊 数据集**

数据集：FuzzyBench‑10M，10M个（spec, input, output）三元组，涵盖29个主题版本、800+子类别，包含噪声扰动版本用于鲁棒性评估。

**📈 对比分析**

对比方法：直接提示（Qwen3 0.6B‑32B、OLMo3‑7B、gpt‑oss‑20B、API模型），符号代码生成，完整微调与固定LoRA。PAW 在 FuzzyBench 上实现 73.78% 的 exact‑match，超过 Qwen3‑32B（68.70%）且使用约 1/50 的推理内存；在 MacBook‑M3 上以 30 tokens/s 运行；在多模态任务上，PAW 在 CoSyn 等任务上也优于 4B 视觉模型。

**⚠️ 局限性**

局限性：① 依赖大模型编译器，编译时需要云端算力；② 仅在单任务或少量任务时可生成高质量 LoRA，复杂长文本任务仍受上下文预算限制；③ 需要对伪编译器进行提示设计，易受提示敏感性影响；④ 目前主要针对文本/图像任务，其他模态（音频、视频）需进一步扩展；⑤ 生成的 LoRA 质量受 FuzzyBench 训练数据质量限制，存在上限。

---

## 599. DemoPSD: Disagreement-Modulated Policy Self-Distillation

**arXiv ID:** 2607.02502 | [PDF](https://arxiv.org/pdf/2607.02502v1)

**作者:** Yunhe Li `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型的推理任务中，提出了一种基于对教师和学生分布不一致度自适应插值的反KL质心目标，利用该目标在自监督策略中逐步削弱对特权信息的依赖，防止特权信息泄露并保持探索熵；

**💡 创新点**

创新点在于：①将教师与学生的分布按几何混合构成反KL质心目标；②用Jensen‑Shannon散度量度教师与学生的不一致度，并通过可微分门控函数动态调节权重；③在理论上证明了该目标既能抑制泄露，又能保持或提升熵；

**🔧 技术方法**

使用技术包括：反KL质心目标、几何混合、Jensen‑Shannon散度、可微分门控（sigmoid调制）、EMA教师复制、密集Token级监督、熵保持分析与梯度裁剪；

**📊 数据集**

使用数据集：SciKnowEval四个科学领域（生物、化学、材料、物理）进行训练与评估，以及GPQA Extended做跨域泛化验证；

**📈 对比分析**

与基线SDPO和GRPO对比，最高提升约4.2% best@16，平均提升1.7% mean@16/maj@16，训练熵提升33‑98%，并在GPQA OOD任务上保持稳定且优于SDPO，表明模型在保留探索性与泛化性方面表现更佳；

**⚠️ 局限性**

局限性在于：①需要每个样本都有可用的特权轨迹作为教师上下文；②对β门控阈值的调优敏感；③在高不一致度Token上仍可能出现残留泄露；④方法主要针对多选推理任务，尚未验证在其他生成式任务上的通用性。

---

## 600. Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots

**arXiv ID:** 2607.02501 | [PDF](https://arxiv.org/pdf/2607.02501v1)

**作者:** Ling Xu `[一作]` (Southeast University), Shuai Wang `[通讯]` (Southeast University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个可移植的 C++ 推理运行时 Embodied.cpp，用于在异构边缘设备上高效部署视觉-语言-动作（VLA）模型和世界-动作（WAM）模型。

**💡 创新点**

创新点在于将模型共享的执行路径抽象为五层架构（输入适配器、序列构建器、骨干执行、头部插件、部署适配器），实现了多速率执行、低延迟批量-1 推理和可扩展的接口插件，兼容 VLA 与 WAM 两大模型族。

**🔧 技术方法**

技术实现包括 C++ 运行时、后端抽象层、可插拔头部与自定义算子、GGUF 量化（Q4_K）以及对 Jetson、RK、x86 等异构硬件的统一调度。

**📊 数据集**

使用了 RoboTwin 任务数据集评估 HY‑VLA，使用 pi0.5 的默认训练集评估其性能，并对 LingBot‑VA 的首个 Transformer 块做了基准测试。

**📈 对比分析**

通过与 Python BF16 基线对比，单块推理内存从 312.2 MiB 降至 88.1 MiB，误差 MAE < 3.3×10⁻²，余弦相似度 > 0.999；在闭环测试中，HY‑VLA 成功率 100%，pi0.5 成功率 91%，推理延迟分别为 735.9 ms 与 56.85 ms。

**⚠️ 局限性**

局限性包括：WAM 的完整闭环实验未完成；仅对单块 Transformer 做了量化基准；对新出现的更复杂模型的支持仍需进一步验证。

---

## 601. Beyond Adam: SOAP and Muon for Faster, Label-Efficient Training of Machine Learning Interatomic Potentials

**arXiv ID:** 2607.02499 | [PDF](https://arxiv.org/pdf/2607.02499v1)

**作者:** Gil Harari `[一作]` (Harvard University), Boris Kozinsky `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了Muion、SOAP、SOAP-Muon等矩阵结构优化器在训练NequIP和Allegro等机器学习原子势（MLIP）模型时的性能，比较它们与默认AdamW的效果。

**💡 创新点**

首次将矩阵预处理优化器引入MLIP训练，并系统地研究其在不同力监督稀疏程度下的优势，发现SOAP在稀疏力监督场景下仍保持高精度和稳定性。

**🔧 技术方法**

采用Muion、SOAP、SOAP-Muon这三种矩阵结构预处理优化器，结合AdamW作为基准，使用E3-equivariant图神经网络架构NequIP和Allegro进行训练。

**📊 数据集**

使用两组真实物理系统的数据集：水分子液体（NequIP）和固体酸电解质CsH2PO4（Allegro），涵盖全力监督、部分力监督和仅能量监督。

**📈 对比分析**

通过多线程、不同力监督比例下的超参数调优，比较了能量和力的MAE以及壁钟时间；结果显示SOAP在能量/力MAE上比AdamW低9-24%，并在完全监督下实现了4.9-5.8倍的壁钟速度提升，稀疏力监督时SOAP-Muon甚至匹配全监督AdamW的性能。

**⚠️ 局限性**

实验仅覆盖两种系统和两种模型，缺乏在更大规模基础模型和不同化学环境下的验证；Muion在水系统表现不佳，暗示其正交化步骤对收敛稳定性有负面影响。

---

## 602. Building the Ipseome: Large, Free, Open, Human Identity Data

**arXiv ID:** 2607.02488 | [PDF](https://arxiv.org/pdf/2607.02488v1)

**作者:** Jason Jeffrey Jones `[一作]` `[通讯]` (Stony Brook University), Jason Jeffrey Jones (Stony Brook University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了名为“ipseome”的开源人类身份数据集框架，汇聚并持续收集自我描述文本，包括每日问卷、全国代表性样本、Twitter自述及用户自评等多种来源。

**💡 创新点**

创新点在于首次系统化、开放、纵向追踪人类自我表达，将自我身份研究从零散数据转为可复现、可大规模分析的科学体系。

**🔧 技术方法**

采用了重复横断面调查、Twitter流媒体抓取、自然语言处理与可视化仪表盘等技术，形成持续的数据收集与实时分析流程。

**📊 数据集**

使用的数据集包括Ipseity Daily每日调查数据、JJJ Pro Who am I代表性问卷数据、HINENI跨国Twitter自述数据、JJJITV2美国Twitter数据及Words You Today用户交互数据。

**📈 对比分析**

通过对比身份标识符在时间与国家层面的流行度、聚类特征等，展示了数据驱动的身份变化分析，但本文未提供算法性能对比，仅展示了基于统计的趋势与聚类结果。

**⚠️ 局限性**

局限性包括样本偏倚（以美国Twitter用户为主）、自报偏差、潜在机器人生成文本、语言与文化多样性不足，以及对隐私保护与伦理的持续关注。

---

## 603. Interpretation-Oriented Cloud Removal via Observation-Anchored Residual Flow with Geo-Contextual Alignment

**arXiv ID:** 2607.02471 | [PDF](https://arxiv.org/pdf/2607.02471v1)

**作者:** Ziyao Wang `[一作]` (Chinese University of Hong Kong Shenzhen), Man-on Pun `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向解释的云去除框架GACR，通过观测锚定残差流（OAR-Flow）实现物理化的残差逆向过程，并加入Geo-Contextual Prior Alignment（GCPA）保持语义一致性；

**💡 创新点**

创新点在于将云去除视为观测锚定的残差逆向生成任务，并通过VFM诱导的语义流形对齐，既保证视觉质量又提升下游语义任务性能；

**🔧 技术方法**

采用观察锚定残差流（OAR-Flow）与流匹配学习、基于DINOv3的Vision Foundation Model进行语义对齐，以及HDiT/DiT架构实现去云；

**📊 数据集**

在六个云去除数据集（CUHKCR-EXT-GZ/CS、Potsdam-CR-thin/thick、Vaihingen-CR-thin/thick）以及对应的十二个下游任务（分类、建筑提取、语义分割、高程估计）上进行实验；

**📈 对比分析**

与现有MPRNet、Restormer、AST、MambaIR、DFCFormer、EMRDM等方法对比，GACR在PSNR、SSIM上提升约1-3dB，mIoU提升约3%至5%，并在大部分下游任务上超越基线，收敛速度提升约5×；

**⚠️ 局限性**

局限性包括对VFM预训练依赖较强，计算开销仍高于纯去噪方法，对极端厚云区域仍可能产生细节失真。

---

## 604. Learning to Move Before Learning to Do: Task-Agnostic pretraining for VLAs

**arXiv ID:** 2607.02466 | [PDF](https://arxiv.org/pdf/2607.02466v1)

**作者:** Junhao Shi `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Task-Agnostic Pretraining (TAP)，通过自监督逆动力学预训练学习“如何移动”的物理先验，再用少量任务标注对齐，显著提升VLA模型在有限专家数据下的性能与鲁棒性。

**💡 创新点**

创新点在于将“如何移动”与“什么去做”解耦，利用任务无关轨迹与自主随机探索的自监督逆动力学目标提炼物理常识，并通过两阶段预训练实现对互联网规模专家数据的有效补偿。

**🔧 技术方法**

采用逆动力学自监督训练、Vision‑Language Transformer（Qwen2.5‑VL+SigLIP）骨干、行为克隆微调、随机自探索采集流程（VoxelGridDownsampling、ContactHeuristic、CosineInterpolate）以及注意力可视化等技术。

**📊 数据集**

使用Bridge任务无关轨迹、Open X‑Embodiment (OXE)、Simular benchmark、真实 WidowX 250 机器人实验数据，并生成30小时自主随机演习数据。

**📈 对比分析**

与标准BC、RT‑1‑X、OpenVLA、NORA、π_0等SOTA模型对比，Simular上-20k预训练样本下Avg‑All成功率达到33.32%，远超大型预训练模型；在真实机器人上，仅200条专家轨迹+30h自演练下，多种扰动场景平均成功率>60%，优于NORA及标准BC，并在视觉/视角变化中表现更稳健。

**⚠️ 局限性**

仍需一定量专家演示以完成语义对齐，长时序推理与复杂语义理解能力有限；预训练需要足够多样化的随机探索，若探索受限会导致物理先验不足；逆动力学在高维或非连续动力学环境下训练可能不稳定。

---

