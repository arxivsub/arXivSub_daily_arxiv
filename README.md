# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-12 | 今日论文总数: 522

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs

**arXiv ID:** 2602.10431 | [PDF](https://arxiv.org/pdf/2602.10431v1)

**作者:** Kanghyun Noh `[一作]` (Sungkyunkwan University), Yulwha Kim `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 QTALE 框架，将 token‑adaptive 层执行与低位量化无缝结合，保持 LLM 的准确率，同时降低 FLOPs 与内存占用。

**💡 创新点**

创新点：① 通过熵正则化的量化鲁棒训练，保证训练路径多样性；② 训练后通过可调阈值 θ 实现推理时的执行比例调节，重新注入冗余以提升对量化误差的鲁棒性。

**🔧 技术方法**

技术手段包括 token‑adaptive 层执行（D‑LLM）、Gumbel‑Softmax 路由、熵正则化、AWQ 与 MagR+GPTQ 等后训练量化、可调阈值执行比例、与稀疏剪枝的兼容性测试。

**📊 数据集**

数据集：CommonsenseQA（含 PIQA、BoolQ、SIQA、ARCe、ARCc、Winogrande、OBQA）、MMLU、Stanford‑Alpaca、SAMSum 等。

**📈 对比分析**

评估方式：与全层执行、D‑LLM、量化全模型以及 D‑LLM+量化的“naive”整合进行对比；结果显示 QTALE 在 4/3‑bit 量化下，准确率差距不超过 0.5%（如 CSQA 72.79% vs 72.22%），FLOPs 约减 50%，内存从 13.5 GB 降至 <4.5 GB，推理速度提升 ~1.28×，且在不同量化方法与剪枝场景下均保持稳定。

**⚠️ 局限性**

局限性：为恢复冗余需略微提升执行比例，导致 FLOPs 与推理时间略有增加；熵正则化与阈值调节依赖于训练与校准数据，迁移到不同 LLM 或任务时可能需要重新调参；在极端稀疏或更低比特量化下的鲁棒性尚未彻底验证。

---

## 2. Min-Sum Uniform Coverage Problem by Autonomous Mobile Robots

**arXiv ID:** 2602.11125 | [PDF](https://arxiv.org/pdf/2602.11125v1)

**作者:** Animesh Maiti `[一作]` (Indian Institute of Technology Jodhpur), Krishnendu Mukhopadhyaya `[通讯]` (Indian Statistical Institute)

**通讯引用:** 483 | [OpenAlex ID](https://openalex.org/A5048645380)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文研究了在一维线段和圆周上，面向自治、匿名、同质、无记忆、无通信的机器人群体进行最小总移动距离（min‑sum）均匀覆盖问题，提出了在 ASYNC 调度器下的确定性分布式算法，并对可解配置与不可解配置进行完整的可解性分类与证明。

**💡 创新点**

创新点包括：
- 首次在极限的分布式机器人模型（无记忆、无通信、非刚性移动）下完成 min‑sum 目标的理论可解性完整刻画；
- 引入“极端机器人”（extremal robot）概念，利用其唯一性和对称性对最优匹配进行约束；
- 通过对称性分析与视角词典（lexicographic ordering）实现确定性决策，即使在 ASYNC 下也能保证无碰撞、单调收敛；
- 对圆周情况给出一套完整的配置分类树，并在不可解类中给出不可解性证明。

**🔧 技术方法**

采用的技术主要有：
- 组合优化与匹配理论，确定最优目标点集合；
- 对称性分解与镜像映射，用于消除多重最优解的歧义；
- 视角与视图比较（lexicographic ordering）实现局部判断；
- 非刚性移动模型下的 δ‑移动保证；
- 递归式证明与归纳，确保算法在 ASYNC 下的终止与收敛。

**📊 数据集**

本文属于理论计算机科学，未使用实验数据集；所有结果均通过形式化证明给出。

**📈 对比分析**

性能评价基于理论证明：
- 所有可解配置下算法在有限时间内完成；
- 运动成本等于全局最优 min‑sum（证明最优性）；
- 算法保持无碰撞与单调收敛；
- 与先前集中式或需要灯光/通信的算法相比，消除了外部辅助，实现了完全无记忆的自治执行。

**⚠️ 局限性**

限制与开放问题：
- 仅考虑全局可见性与无阻塞；在有限可见性或阻挡环境下可否保持最优性尚未解决；
- 只讨论一维（线段、圆周）环境，扩展到更高维度或非平坦曲面仍是挑战；
- 对非刚性移动的 δ 未知时，需要更强的移动保证或新的等待策略；
- 在实际机器人硬件中对同步与时延的粗粒度假设可能不完全成立。

---

## 3. Reality Copilot: Voice-First Human-AI Collaboration in Mixed Reality Using Large Multimodal Models

**arXiv ID:** 2602.11025 | [PDF](https://arxiv.org/pdf/2602.11025v1)

**作者:** Liuchuan Yu `[一作]` (George Mason University), Lap-Fai Yu `[通讯]` (George Mason University)

**通讯引用:** 3567 | [OpenAlex ID](https://openalex.org/A5084967276)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在混合现实头显上实现了一个以语音为主的人工智能助手，利用大规模多模态模型（LMM）完成语音理解、文本生成、图像描述、3D模型生成等多种任务，支持跨平台工作流和实时录像。

**💡 创新点**

首次将 LMM 与语音驱动的 MR 交互结合，提出混合商业与开源模型的架构，使用栈式上下文管理实现语境感知，并在硬件层面实现双轨音视频实时录制，保证用户隐私的本地处理。

**🔧 技术方法**

技术包括 Unity + Vuplex 3D WebView、Android Studio 插件、TEN VAD、EmBARDiment、OpenAI ChatGPT/Google Gemini、FastVLM、SAM3/SAM3D、MR Utility Kit、硬件加速 H.264/AAC 编码、Gmail API 等。

**📊 数据集**

未公开使用专门数据集；采用本地采集的图像、视频和 3D 生成任务，利用预训练的 FastVLM、SAM3、SAM3D 模型完成相关推理。

**📈 对比分析**

论文未给出量化对比；通过用户案例（电子学习、第一人称视频录制、3D 建模）展示系统的实时性和可用性，性能主要以响应时间和语音交互流畅度为主。

**⚠️ 局限性**

依赖网络调用商业 LMM，网络延迟可能影响交互；硬件兼容性局限于支持的 MR 头显；缺乏大规模评测与客观指标；隐私保护需要进一步完善。

---

## 4. SynergyKGC: Reconciling Topological Heterogeneity in Knowledge Graph Completion via Topology-Aware Synergy

**arXiv ID:** 2602.10845 | [PDF](https://arxiv.org/pdf/2602.10845v1)

**作者:** Xuecheng Zou `[一作]` (Soochow University), Bingbing Wang `[通讯]` (Soochow University)

**通讯引用:** 39141 | [OpenAlex ID](https://openalex.org/A5100382667)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SynergyKGC 框架，结合预训练语言模型与图神经网络，通过双塔结构与交叉模态协同专家实现知识图谱补全。

**💡 创新点**

创新点在于：①主动指令驱动的结构检索与关系感知交叉注意力；②密度感知身份锚定策略，平衡稀疏与稠密图的结构噪声与表示崩塌；③双轴一致性（架构一致性与生命周期一致性）以及捕获效应，快速收敛并消除训练推理分布漂移；④自适应门控融合语义与结构信息。

**🔧 技术方法**

技术包括：BERT 语义专家、关系感知交叉注意力、密度阈值身份锚定、双塔对齐、信息对比学习（InfoNCE）、MSE 对齐正则、动态双塔一致性与自适应门控。

**📊 数据集**

使用两个公开基准数据集：FB15k-237（稠密）和 WN18RR（稀疏）。

**📈 对比分析**

与 TransE、RotatE、DistMult、KG-BERT、SimKGC、ProgKGC 等多种传统与混合模型对比，SynergyKGC 在所有主要指标上均显著提升，尤其在 WN18RR 的 Hits@1 提升 8.0% 以上，且训练时间显著缩短。

**⚠️ 局限性**

局限性包括：对密度阈值 ϕ 的手动设定敏感；在极稀疏或极稠密之外的图结构中性能可能下降；需要双塔并行计算，计算成本相对较高；若语义描述过短或质量低，语义专家的表示稳定性可能受限。

---

## 5. Transport, Don't Generate: Deterministic Geometric Flows for Combinatorial Optimization

**arXiv ID:** 2602.10794 | [PDF](https://arxiv.org/pdf/2602.10794v1)

**作者:** Benjy Friedmann `[一作]` (Technion), Nadav Dym `[通讯]` (Technion)

**通讯引用:** 581 | [OpenAlex ID](https://openalex.org/A5086417185)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将欧氏TSP问题重新定义为确定性几何流问题，利用点传输而非边热图去噪生成路径。

**💡 创新点**

提出CycFlow框架，通过学习实例条件向量场将节点连续传输到圆形基准空间，利用数据依赖的流匹配实现线性坐标动力学，避免二次边评分瓶颈。

**🔧 技术方法**

利用确定性流匹配（Deterministic Flow Matching）与ODE学习向量场，采用Canonicalize-Process-Restore架构结合Transformer+RoPE，并通过Procrustes对齐和Fiedler向量排序实现几何对齐。

**📊 数据集**

在标准Euclidean TSP基准集上训练和评估，节点数分别为50、100、500、1000。

**📈 对比分析**

与传统求解器（Concorde、LKH3）、构造启发式（POMO、2-OPT）以及最新扩散模型（DIFUSCO、T2T、Fast-T2T）进行对比，结果在保持相近最优误差的同时，推理速度提升2-3个数量级，TSP-1000仅0.22秒。

**⚠️ 局限性**

需要监督训练即访问最优解；仅适用于欧氏空间的连续几何TSP，难以直接扩展到非欧氏或一般图路由问题。

---

## 6. MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models

**arXiv ID:** 2602.10934 | [PDF](https://arxiv.org/pdf/2602.10934v1)

**作者:** Yitian Gong `[一作]`, Xipeng Qiu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种全因果Transformer架构（CAT）并实现大规模音频离散化器MOSS-Audio-Tokenizer，能够在单一端到端训练框架下，将连续音频转换为可供自回归语言模型直接使用的离散序列；

**💡 创新点**

创新点包括：① 采用纯Transformer编码器-解码器实现无CNN混合、可扩展的统一接口；② 在RVQ层引入量化dropout实现可变比特率；③ 在语音合成中提出Progressive Sequence Dropout训练策略，使单模型在任意比特率下均能稳健生成；④ 通过大规模音频‑文本配对数据与多任务学习实现语义对齐，提升生成与理解性能；

**🔧 技术方法**

核心技术为：因果Transformer编码器/解码器、残差向量量化（RVQ）+量化dropout、对抗式训练（多尺度判别器）、音频‑文本对齐的解码器仅LLM、Progressive Sequence Dropout、端到端联合优化；

**📊 数据集**

使用约3 000 000小时的多领域（语音、声音、音乐）音频数据进行预训练，含多语言（英语/中文）和配对文本；此外在下游评测中采用LibriSpeech、AISHELL‑2、AudioSet、MUSDB、Seed‑TTS‑Eval等公开基准；

**📈 对比分析**

在重构质量上，MOSS-Audio-Tokenizer在低/中/高比特率下均超过或与公开音频 tokenizer（Encodec、XCodec2.0、MiMo、XY‑Tokenizer 等）同等或更优，指标如STFT‑Dist、STOI、PESQ、Mel‑Loss、SIM 等均位列前列；在TTS 任务中，基于CAT的完全自回归系统 CAT‑TTS 在 Seed‑TTS‑Eval 的 WER 低于 2% 并且在 SIM 上击败所有对比模型；同时在语音理解任务中与现有大模型对标表现良好；

**⚠️ 局限性**

主要局限包括：① 训练与推理成本高（1.6 B 参数、3 M 小时数据、需大规模算力）；② 目前对实时/低延迟推理的评估有限；③ 虽然支持多比特率，但在极低比特率下仍存在音质衰减；④ 端到端优化对超参数和多任务平衡敏感，需进一步研究鲁棒性。

---

## 7. From Circuits to Dynamics: Understanding and Stabilizing Failure in 3D Diffusion Transformers

**arXiv ID:** 2602.11130 | [PDF](https://arxiv.org/pdf/2602.11130v1)

**作者:** Maximilian Plattner `[一作]` (Institute for Machine Learning), Arturs Berzins `[通讯]` (Institute for Machine Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究并解决3D扩散变换器在稀疏点云重建中出现的“Meltdown”碎片化失败模式。

**💡 创新点**

将机制可解释性与扩散动力学相结合，定位到单个交叉注意力激活并提出基于谱熵压缩的测试时干预。

**🔧 技术方法**

采用激活补丁、奇异值分解、谱熵指标、DDPM/DDIM、WaLa与Make‑a‑Shape扩散变换器。

**📊 数据集**

在Google Scanned Objects（GSO）与SimJEB数据集上评估。

**📈 对比分析**

与未干预模型对比，干预后在两数据集上实现高达98.3%（WaLa）和84.6%（Make‑a‑Shape）的碎片化恢复率。

**⚠️ 局限性**

干预强度需手工调参，且对不同架构敏感，谱熵作为粗粒度指标尚未解释具体几何特征。

---

## 8. Amortized Inference of Neuron Parameters on Analog Neuromorphic Hardware

**arXiv ID:** 2602.10763 | [PDF](https://arxiv.org/pdf/2602.10763v1)

**作者:** Jakob Kaiser `[一作]` (Institute of Computer Engineering), Johannes Schemmel `[通讯]` (Kirchhoff Institute for Physics)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用非顺序的模拟基础推断（sbi）方法，对 BSS2 模拟神经形态硬件中 ADEx 神经元的七个参数进行参数化，得到可用于多次实验的近似后验分布。

**💡 创新点**

创新点在于：① 引入了 amortized neural density estimator（NDE）与 BayesFlow 结合的 summary network，实现一次训练即可推断任意观测；② 通过训练二元分类器预筛选参数组合，显著提高感兴趣区域（中等放电频率）样本比例；③ 对比手工特征与自动提取特征两种摘要方式，验证 summary network 在后验聚焦和膜电位动态匹配方面的优势。

**🔧 技术方法**

使用了非顺序 SBI（BayesFlow）算法、Coupling Flow 形式的 NDE、ResNet 二分类器、卷积+循环网络的 summary network 以及 KL 散度损失进行训练。

**📊 数据集**

数据集来自 BSS2 物理硬件的 ADEx 神经元实验：50000 组随机参数生成的原始数据，经二分类器筛选后得到 600000 组有效样本用于训练，另 600 组用于验证；每个样本包含插值后的膜电位轨迹、脉冲时序以及 12 维电生理特征。

**📈 对比分析**

比较方法：用手工摘要统计训练的 NDE 与用 summary network 训练的 NDE 的后验分布和后验预测轨迹进行对比。结果显示：summary network 的后验更集中，预测轨迹更贴近目标膜电位；但两者在某些特征（如快速/慢陷深度、首次发放延迟）上仍有偏差，且整体仍显示一定的偏倚与失准。性能指标主要通过后验预测检验和特征一致性评估来体现。

**⚠️ 局限性**

局限性包括：后验分布存在偏倚与失准，无法完全捕捉硬件中的时间噪声导致的行为波动；部分观测下预测结果缺失 spikes 或过多 spikes；目前尚未对后验校准性做深入评估；需要进一步探索 diffusion‑based 模型与更精准的误差校正方法。

---

## 9. When Fusion Helps and When It Breaks: View-Aligned Robustness in Same-Source Financial Imaging

**arXiv ID:** 2602.11020 | [PDF](https://arxiv.org/pdf/2602.11020v1)

**作者:** Rui Ma `[一作]` `[通讯]` (Awakening of Insects Co., Ltd.), Rui Ma (Awakening of Insects Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在上海金价指数数据上，构建同源的两种图像视图（OHLCV价格/成交量图与技术指标矩阵），开展同源多视图学习与对抗鲁棒性研究。

**💡 创新点**

创新点包括：① 采用泄漏防护的时间块拆分与可解释的最小运动阈值筛选以定义标签噪声分段；② 设计视图对齐的攻击协议，区分单视图受限攻击与联合攻击；③ 对早期融合与晚期融合以及一致性正则化在不同噪声下的效能进行系统对比。

**🔧 技术方法**

使用的技术包括：① 金融图像渲染（OHLCV图和指标矩阵）；② 轻量级CNN（Lite-CNN）与预训练ResNet-18；③ 双分支晚期融合模型并加入温度缩放的对称KL一致性正则；④ FGSM与PGD ℓ∞对抗攻击；⑤ Matthews相关系数作为评估指标。

**📊 数据集**

使用的数据集为 2005‑2025 年的上海金价指数（SGE）每日收盘价与成交量。

**📈 对比分析**

通过在不同视图、融合方式和一致性权重下，比较单视图、早期融合、晚期融合以及一致性模型；在中等标签噪声（τ=0.006）下，晚期融合可将 MCC 提升至约 0.1，并在单视图受限攻击中表现更稳健；但联合攻击下仍显著退化。

**⚠️ 局限性**

局限性包括：① 仅对渲染图像层面的攻击，未覆盖真实市场对抗场景；② 结果高度依赖最小运动阈值与样本规模，可能在不同噪声级别下不稳定；③ 采用的对抗攻击为单步或无随机重启的 PGD，未使用更强的攻击或对抗训练；④ 未探索可证明鲁棒性或更复杂的多视图防御策略。

---

## 10. Weight Decay Improves Language Model Plasticity

**arXiv ID:** 2602.11137 | [PDF](https://arxiv.org/pdf/2602.11137v1)

**作者:** Tessa Han `[一作]` (Broad Institute), Sham Kakade `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在LLM预训练阶段使用不同weight decay值对模型可塑性及其下游微调性能的影响，并提供了系统实验和机制分析。

**💡 创新点**

创新点在于将weight decay视为塑性调节器，而非仅仅是正则化，证明其可提高模型在下游任务中的适应性，并揭示了其在表示线性可分化、注意力矩阵秩下降和过拟合抑制等机制。

**🔧 技术方法**

采用了AdamW优化器的weight decay调参、线性探针、注意力矩阵秩评估、训练-验证差距分析等技术，并在预训练与微调阶段进行全流程实验。

**📊 数据集**

使用了FineWeb-Edu和olmo-mix-1124两大预训练语料，以及Llama-2和OLMo-2两族模型，结合六个Chain-of-Thought下游任务。

**📈 对比分析**

通过对不同weight decay设置下的预训练交叉熵、微调后的Pass@1/16、ORM评分等指标进行对比，发现相较于默认0.1，较高weight decay（如0.5–1.0）在预训练损失略高但下游性能显著提升。

**⚠️ 局限性**

局限性包括：实验仅覆盖到4B参数规模，对更大模型和更长训练周期的影响未知；权衡可塑性与预训练准确度的最佳值随任务与规模变化；未探究weight decay与其他超参交互的综合效应。

---

## 11. Predicting integers from continuous parameters

**arXiv ID:** 2602.10751 | [PDF](https://arxiv.org/pdf/2602.10751v1)

**作者:** Bas Maat `[一作]` (Vrije Universiteit Amsterdam), Peter Bloem `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 5236 | [OpenAlex ID](https://openalex.org/A5001241443)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在神经网络中使用离散分布直接预测整数标签，并提出了多种新的离散分布模型。

**💡 创新点**

创新点在于提出离散拉普拉斯（Dalap）和比特序列（Bitwise）两种可用于回归的离散分布，并将其与传统连续近似进行比较。

**🔧 技术方法**

主要技术包括构造离散化的正态、拉普拉斯、Weibull 等分布以及新的 Dalap、Danorm、Bitwise，利用梯度可微的参数化在神经网络中训练。

**📊 数据集**

使用了多种数据集：Bicycles、Upvotes、Migration（表格回归）、MAESTRO（MIDI 时序预测）以及 MNIST、FashionMNIST、CIFAR‑10（像素级生成）。

**📈 对比分析**

通过负对数似然（Bits）和 RMSE 进行比较，Dalap 在大多数任务中取得最优的负对数似然，Bitwise 在高熵数据上表现良好，但 RMSE 较差；连续平方误差在表格回归中往往更优。

**⚠️ 局限性**

局限性包括某些分布（如 Danorm、Dweib）在高维任务中数值不稳定或计算成本高，且实验中未充分调优混合组件数，可能影响结果。

---

## 12. Labor, Capital, and Machine: Toward a Labor Process Theory for HCI

**arXiv ID:** 2602.10548 | [PDF](https://arxiv.org/pdf/2602.10548v1)

**作者:** Yigang Qin `[一作]` (Syracuse University), EunJeong Cheon `[通讯]` (Syracuse University)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5014866832)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文将劳动过程理论（LPT）引入人机交互（HCI）研究，构建以资本与劳动关系为核心的结构化分析框架，并基于此提出对技术设计与研究的新议程。

**💡 创新点**

创新之处在于首次系统将马克思主义劳动过程理论与 HCI 结合，强调劳动与工作区别，揭示控制、同意、管理层与全球价值链等维度，并为 HCI 设计提供以剥削机制为导向的批判性视角。

**🔧 技术方法**

主要采用理论阐释、系统性文献综述和案例分析；未使用具体算法或软件技术。

**📊 数据集**

研究以学术文献、案例研究和半系统性综述为数据来源，未涉及公开数据集。

**📈 对比分析**

本文不做实验或性能对比，而是通过对比现有 HCI 研究方法与 LPT 视角，说明 LPT 能揭示的结构性剥削机制与设计缺陷。

**⚠️ 局限性**

局限在于缺乏实证验证与定量评估，未在真实 HCI 设计中检验提出框架的可行性与效果，未来需要开展案例研究与实验检验。

---

## 13. To Think or Not To Think, That is The Question for Large Reasoning Models in Theory of Mind Tasks

**arXiv ID:** 2602.10625 | [PDF](https://arxiv.org/pdf/2602.10625v1)

**作者:** Nanxu Gong `[一作]` (Arizona State University), Xing Xie `[通讯]` (Microsoft Research Asia)

**通讯引用:** 12397 | [OpenAlex ID](https://openalex.org/A5081376690)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了大规模推理模型（LRMs）在理论心智（ToM）任务中的表现，并将其与非推理模型进行对比；

**💡 创新点**

创新点在于揭示推理模型在ToM任务中不一定优于非推理模型，发现“慢思考崩溃”和“选项匹配捷径”两大失效机制，并提出Slow-to-Fast（S2F）与Think-to-Match（T2M）干预策略；

**🔧 技术方法**

采用Chain-of-Thought（CoT）推理控制、token长度限制、动态“wait”触发的快速切换、以及移除选项后再匹配的思考-匹配流程等技术；

**📊 数据集**

使用了HiToM、ToMATO和ToMBench三大ToM基准数据集，分别覆盖不同层级、情景和心智状态；

**📈 对比分析**

通过比较相同系列的推理与非推理模型，在三大基准上计算准确率，发现推理模型往往表现相同或更差，慢思考导致准确率下降，而S2F与T2M能够显著提升性能；

**⚠️ 局限性**

局限性包括推理机制与ToM需求不匹配导致的失效，选项匹配导致的捷径，缺乏自适应推理阈值，实验仅覆盖公开模型，无法完全验证更大规模或不同任务的普适性。

---

## 14. AI-PACE: A Framework for Integrating AI into Medical Education

**arXiv ID:** 2602.10527 | [PDF](https://arxiv.org/pdf/2602.10527v1)

**作者:** Scott P. McGrath `[一作]` (University of California Berkeley), Nick Anderson `[通讯]` (University of California Davis)

**通讯引用:** 4883 | [OpenAlex ID](https://openalex.org/A5040186285)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过系统综述和主题分析，提出并阐述了 AI‑PACE 框架，旨在为医学教育提供一套纵向整合、覆盖认知、心理运动、情感与嵌入式四大维度的 AI 培训体系。

**💡 创新点**

创新点在于①将 Bloom 认知分类与 Psychomotor、Affective、Embedded 四大支柱相结合，构建完整的学习梯度；②首次将“情感域”与信任校准嵌入医学 AI 教育；③提供跨 UME‑GME‑CME 的通用框架，填补了先前仅针对影像或专科的碎片化空白。

**🔧 技术方法**

研究主要运用了系统文献检索（PubMed、MEDLINE、ERIC、Asta 等数据库）、主题分析、对照矩阵、Bloom 认知分类映射以及 Harden 等医学教育理论框架。

**📊 数据集**

所用数据集为 2016‑2025 年间检索到的 643 篇文献，筛选后保留 23 篇核心论文，用以构建比较与框架设计。

**📈 对比分析**

通过比较表评估现有 AI 教育框架在纵向整合、通用性和域覆盖等维度的缺口，AI‑PACE 在所有评估维度上得分最高，理论上可实现更高的学习连续性和情感培养；但尚未在实证实验中量化其教学效果。

**⚠️ 局限性**

局限性包括：仅基于文献综述，缺乏实际教学实验验证；框架实施高度依赖多学科教师与资源；技术快速迭代导致内容更新困难；缺少跨地区、跨专业的多中心评估。

---

## 15. ChainRec: An Agentic Recommender Learning to Route Tool Chains for Diverse and Evolving Interests

**arXiv ID:** 2602.10490 | [PDF](https://arxiv.org/pdf/2602.10490v1)

**作者:** Fuchun Li `[一作]` (Chinese Academy of Sciences), Hailong Shi `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 899 | [OpenAlex ID](https://openalex.org/A5015917634)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可动态规划、可主动收集证据的推荐框架ChainRec，利用LLM工具库与学习到的规划器实现按场景自适应的推荐流程；

**💡 创新点**

创新点在于：①把推荐过程拆解为可复用工具的调用和动态决策；②使用SFT+DPO两阶段训练，让规划器在固定步骤预算下学习最优工具路由；③构建标准化工具库，提升可解释性和迁移性；

**🔧 技术方法**

采用大型语言模型（如Qwen3-8B）做规划器和工具实现，利用chain‑of‑thought提示、工具调用接口、强化学习中的直接偏好优化（DPO）进行训练；

**📊 数据集**

使用AgentRecBench的公开数据集，包含Amazon、Goodreads、Yelp三大领域，并在Classic、Cold‑Start、Evolving‑Interest等多种推荐场景下进行评估；

**📈 对比分析**

与传统矩阵分解、深度学习模型及多种基线（BaseAgent、CoTAgent、MemoryAgent、Agent4Rec等）比较，ChainRec在Avg HR@{1,3,5}上显著领先（Amazon +14%~+23%，Goodreads +0~+37%，Yelp +272%~+82%）且保持较低的规划步骤；

**⚠️ 局限性**

局限性包括：依赖固定候选集和交互预算；工具库构建需要大量专家CoT数据；在极端稀疏或跨域场景下仍需进一步验证；

---

## 16. Scaling World Model for Hierarchical Manipulation Policies

**arXiv ID:** 2602.10983 | [PDF](https://arxiv.org/pdf/2602.10983v1)

**作者:** Qian Long `[一作]`, Xinghang Li `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种分层的视觉-语言-动作框架，通过利用大规模预训练的世界模型来解决机器人操作中的泛化瓶颈，特别是在有限的真实机器人数据下。

**💡 创新点**

创新点在于引入了视觉目标合成和分层任务分解的方法，使得低级策略能够在未见物体和新场景中更好地泛化。

**🔧 技术方法**

使用了分层的视觉-语言-动作（VLA）框架，结合了世界模型作为高层规划器和VLA作为低层执行器。

**📊 数据集**

使用了来自Open-X-Embodiment、AgiBot World Beta和Mobile Aloha的数据集，构建了包含超过1M轨迹的庞大数据集，并通过自动化的轨迹分解管道生成了文本子任务和视觉目标图像的交错序列。

**📈 对比分析**

与基线方法相比，提出的方法在未见场景中的成功率从14%提升至69%，显示出显著的性能提升，尤其是在分布外（OOD）场景中。

**⚠️ 局限性**

限制在于目前的评估仅限于拾取和放置任务，未来需要扩展到更复杂和长时间的操作场景。此外，生成的视觉子目标可能与真实子任务端点存在小的空间偏差，这可能导致执行失败。

---

## 17. Bridging the Compression-Precision Paradox: A Hybrid Architecture for Clinical EEG Report Generation with Guaranteed Measurement Accuracy

**arXiv ID:** 2602.10544 | [PDF](https://arxiv.org/pdf/2602.10544v1)

**作者:** Wuyang Zhang `[一作]`, Yinzhi Jin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一个混合架构，先用信号处理精确提取EEG测量值，再利用LLM生成报告，解决压缩导致的测量精度损失。

**💡 创新点**

创新点是将测量提取与文本生成分离，采用多速率采样、信号处理安全护栏、图注意力和参数高效微调，实现精确测量与长上下文的兼容。

**🔧 技术方法**

技术包括信号处理测量护栏、多速率采样、跨模态EEG‑to‑语言桥、图神经网络、SSM层、参数高效微调、约束解码和合成校准。

**📊 数据集**

使用公开的TUH、TUSZ和CHB‑MIT脑电数据集进行评估。

**📈 对比分析**

与EEGNet、BENDR等基线相比，FA/24h降至0.51（↓56%），检测延迟从24.2s降到10.5s，频率MAE为0.18Hz（低于临床容差0.1Hz），定位准确率提升至85%。

**⚠️ 局限性**

局限包括边缘设备部署需进一步压缩、冻结测量值限制文本细腻表达、数据集人口多样性不足、需要自动化伪影处理以及跨中心验证。

---

## 18. Biomimetic Mantaray robot toward the underwater autonomous -- Experimental verification of swimming and diving by flapping motion -

**arXiv ID:** 2602.10904 | [PDF](https://arxiv.org/pdf/2602.10904v1)

**作者:** Kenta Tabata `[一作]` (Utsunomiya University), Koichi Ozaki `[通讯]` (Utsunomiya University)

**通讯引用:** 11351 | [OpenAlex ID](https://openalex.org/A5053237195)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文设计并实验验证了一款仿真曼塔雷鱼的水下机器人，该机器人通过双侧胸鳍的摆动实现推进，采用Raspberry Pi 3B、IMU与压力传感器实现实时姿态与深度控制，并通过PD控制实现稳定的直线游动与潜水。

**💡 创新点**

创新点包括：①将曼塔雷鱼侧面流线型截面（近似NACA 0020翼型）应用于机器人控制舱与鳍片设计，显著降低流动阻力；②采用双轴伺服（摆动+羽化）模拟真实鳍动，结合90°相位差实现最大推力；③将PD姿态控制与深度目标融合，提升在水中潜水时的轨迹稳定性。

**🔧 技术方法**

技术手段包括：1) 3D打印软硬件混合鳍片（RGD720、TangoBlackPlus）；2) Raspberry Pi 3B + Arduino Nano 进行传感数据采集与伺服驱动；3) MPU9050+LPS33HW实现姿态与深度估计；4) 基于IMU误差的PD姿态控制；5) 通过实验室泳池与视频摄像头进行轨迹与角度评估。

**📊 数据集**

本研究未使用公开数据集，实验数据全部来自实验室泳池中的多次游动与潜水试验，记录了位置、速度、偏航角及误差统计。

**📈 对比分析**

与传统螺旋桨推进相比，实验结果表明：①在相同频率下，曼塔雷鱼型推进的平均速度约为20–22 cm/s；②PD控制后直线误差显著下降（最大误差从17.5 cm降至5.5–7.1 cm）；③与螺旋桨推进相比，能耗更低且对水底沉积物扰动更小，适用于敏感环境；然而在深潜大冲击时，单一PD控制仍无法完全补偿碰撞导致的姿态偏差。

**⚠️ 局限性**

限制与不足：①PD控制在遇到较大外部扰动（如碰撞或墙壁干扰）时仍存在显著误差，需引入更复杂的鲁棒或自适应控制；②机器人目前仅在平坦泳池实验验证，缺乏复杂水动力与流场下的适用性评估；③硬件配置功耗约50 min，续航受限；④实验未涉及多机器人协同或远程视觉引导，适用范围有限。

---

## 19. Exploring the Interplay Between Voice, Personality, and Gender in Human-Agent Interactions

**arXiv ID:** 2602.10535 | [PDF](https://arxiv.org/pdf/2602.10535v1)

**作者:** Kai Alexander Hackney `[一作]` (Georgia Institute of Technology), Pedro Guillermo Feijoo-Garcia `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过实验研究了虚拟代理的性别、声音和人格特征如何影响用户对代理的感知。

**💡 创新点**

创新点在于系统地检验了声音中的人格线索与性别同步效应，并发现男性代理的声音难以被区分。

**🔧 技术方法**

使用了人工智能生成语音（Eleven Labs）结合 TIPI 个性量表进行用户情境评估。

**📊 数据集**

数据集来自 388 名美国受试者和 4 个人工生成的音频，原始音频采样自 11 名自我报告人格不同的参与者。

**📈 对比分析**

通过 Mann‑Whitney U 与 Spearman 相关检验，发现女性代理声音可显著区分内向与外向，男性代理无显著差异，且男性用户更易出现人格同步。

**⚠️ 局限性**

局限包括仅基于自报人格挑选声音样本、样本多样性不足、未控制其他人口学变量，导致男性声音识别效果受限。

---

## 20. Reimagining Sign Language Technologies: Analyzing Translation Work of Chinese Deaf Online Content Creators

**arXiv ID:** 2602.10235 | [PDF](https://arxiv.org/pdf/2602.10235v1)

**作者:** Xinru Tang `[一作]` (University of California), Anne Marie Piper `[通讯]` (University of California)

**通讯引用:** 7576 | [OpenAlex ID](https://openalex.org/A5066173469)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对13名中国聋人在线内容创作者进行访谈，探讨他们在视频中进行的多语言、多文化翻译工作；

**💡 创新点**

将聋人翻译工作定位为跨语言、跨文化、政治性的创意活动，提出翻译不应被视为单一的口译/文字翻译，而应视为语言活动（languaging），并为未来技术设计提供支持多元语种和文化的建议；

**🔧 技术方法**

采用半结构化访谈和主题分析方法，并结合对创作者视频的观察；

**📊 数据集**

无公开数据集，研究使用的是创作者个人频道的视频与访谈记录；

**📈 对比分析**

未进行量化比较或性能评估，研究为质性探索性研究；

**⚠️ 局限性**

研究者听力背景可能影响对访谈内容的解读；样本主要来自城市且受教育程度较高，未包含少数民族地区手语；未考察观众的接收和反馈。

---

## 21. Omni-Safety under Cross-Modality Conflict: Vulnerabilities, Dynamics Mechanisms and Efficient Alignment

**arXiv ID:** 2602.10161 | [PDF](https://arxiv.org/pdf/2602.10161v1)

**作者:** Kun Wang `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 48876 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对全模态大型语言模型（OLLM）在跨模态交互下容易失效的安全问题，作者构建了跨模态无监督安全基准 AdvBench‑Omni，系统分析了内部拒绝机制，提出了基于奇异值分解的“黄金拒绝向量”与轻量级适应性层级调节模块（AdaRefuse）来提升拒绝成功率，同时保持模型的通用能力。

**💡 创新点**

创新点包括：①在安全评估中首次采用“模态与语义解耦”原则，构造了可严控语义一致性的全模态基准；②发现并解析了“中层消解”（Mid‑layer Dissolution）现象及拒绝向量幅值缩小是导致跨模态安全下降的根本原因；③通过奇异值分解提取跨模态共享的纯拒绝方向，并设计自适应层级调节适配器实现无需大规模参数更新的安全对齐。

**🔧 技术方法**

核心技术为：跨模态基准构建（基于文本转换为图像、音频、视频的确定性渲染）；拒绝向量提取与激活调节（利用均值差法得到拒绝向量）；奇异值分解（SVD）用于提取黄金拒绝向量；轻量级 2‑层 MLP 适配器实现层级自适应调节；对抗性安全评估（Refusal Success Rate, Benign Acceptance Rate）。

**📊 数据集**

使用的数据集包括：AdvBench‑Omni（从 AdvBench 扩展的 11 维模态组合，共 11,220 条样本，包含 520 条恶意指令及 500 条友好指令）；八个跨模态安全测试集（文本、图像、音频、视频及其组合）；OmniBench（评估通用能力）。

**📈 对比分析**

与 Self‑Reminder、OmniGuard 等基线方法在 3 种主流 OLLM（Qwen2.5‑Omni‑7B、Baichuan‑Omni‑1.5、MiniCPM‑o‑2.6）上进行对比。AdaRefuse 在 8 个跨模态数据集上的平均 Refusal Success Rate 提升至 91.2%，比基线高约 15–30%；Benign Acceptance Rate 维持在 80% 以上，基本不影响模型的通用性能（OmniBench 准确率差异 < 2%）。

**⚠️ 局限性**

局限性：①目前仅在 4 种基本模态（文本、图像、音频、视频）及其组合上验证，缺乏对更复杂模态（如 3D、传感器流）的评估；②方法在极端多模态序列或长文本上下文中的鲁棒性尚未彻底验证；③SVD 提取的黄金拒绝向量假设各模态拒绝信号共线性，可能在不同模型架构或更大规模模型中失效。

---

## 22. Semi-Supervised Cross-Domain Imitation Learning

**arXiv ID:** 2602.10793 | [PDF](https://arxiv.org/pdf/2602.10793v1)

**作者:** Li-Min Chu `[一作]` (National Yang Ming Chiao Tung University), Ping-Chun Hsieh `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 258 | [OpenAlex ID](https://openalex.org/A5017738079)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 AdaptDICE 算法，实现半监督跨域模仿学习（Semi‑Supervised CDIL）

**💡 创新点**

创新点包括：①使用可学习的状态/动作映射函数实现源域与目标域之间的 Bellman 误差最小化；②通过自适应权重 β(t) 动态融合源域与目标域的密度比率，理论上保证收敛且无需配对轨迹；③在离线数据上完成整个流程，无需在线交互

**🔧 技术方法**

采用技术：DICE 密度比估计、交叉域映射损失、加权行为克隆、正则化分布匹配、神经网络实现映射与策略、normalizing flow 约束映射输出、滑动平均自适应权重

**📊 数据集**

使用数据集：源域为 MuJoCo（Hopper‑v3、HalfCheetah‑v3、Ant‑v3）和 Robosuite（Panda 机器人）; 目标域为对应的修改版 MuJoCo 与 UR5e 机器人；数据包含少量专家轨迹（1–5 条）与大量子最优轨迹（随机或低质量演示）

**📈 对比分析**

与 SMODICE、GWIL、IGDF+IQLearn 等基线方法在 MuJoCo 与 Robosuite 任务上进行对比；AdaptDICE 在所有任务均显著优于基线，尤其在高维机器人臂任务中提升幅度最大；Ablation 结果证明同时使用源域与目标域密度比率是提升性能的关键

**⚠️ 局限性**

局限性：①仍需要一定数量的目标专家示例，无法完全无监督；②在源域与目标域差异极大时，映射与自适应权重仍可能出现偏差；③实验仅在模拟环境中验证，缺乏真实世界的验证

---

## 23. Theory of Troubleshooting: The Developer's Cognitive Experience of Overcoming Confusion

**arXiv ID:** 2602.10540 | [PDF](https://arxiv.org/pdf/2602.10540v1)

**作者:** Arty Starr `[一作]` (University of Victoria), Margaret-Anne Storey `[通讯]` (University of Victoria)

**通讯引用:** 9380 | [OpenAlex ID](https://openalex.org/A5038905934)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

对27名专业开发者进行深度访谈，采用构建主义扎根理论方法，构建了基于认知科学的《故障排查理论》。

**💡 创新点**

首次将认知疲劳、混乱体验与开发者体验紧密结合，提出了“混乱体验 → 争取清晰 → 探索与直觉”这一层次化模型，为工具设计与风险管理提供了认知层面的洞察。

**🔧 技术方法**

主要采用构建主义扎根理论（CGT）的访谈、线性编码、持续比较、Memos记录、Miro可视化、PostgreSQL 数据管道等质性研究技术；未使用机器学习或实验算法。

**📊 数据集**

数据来源为27名经验丰富（总经验570年）的开发者访谈记录，包含1032条初始编码和451个聚焦代码，涵盖不同职能、公司规模与性别比例。

**📈 对比分析**

研究不涉及传统意义上的算法性能比较；通过理论饱和度评估模型可靠性，并通过七次跟进访谈获得共鸣评分，显示理论在实际开发者心智中的高度一致性。

**⚠️ 局限性**

样本采用便利抽样，主要来自北美和欧洲，女性与跨性别比例相对不足；缺乏现场观察或屏幕记录，仅依赖回忆性访谈，可能影响经验的完整性和普适性。

---

## 24. Solving Geodesic Equations with Composite Bernstein Polynomials for Trajectory Planning

**arXiv ID:** 2602.10365 | [PDF](https://arxiv.org/pdf/2602.10365v1)

**作者:** Nick Gorman `[一作]` (University of Iowa), Venanzio Cichella `[通讯]` (University of Iowa)

**通讯引用:** 1117 | [OpenAlex ID](https://openalex.org/A5014268576)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于复合 Bernstein 多项式和高斯成本表面来求解测地线方程的轨迹规划方法。

**💡 创新点**

创新点在于将障碍物映射为连续的高斯成本表面，并用符号化的测地线方程与复合 Bernstein 多项式结合，实现连续轨迹且无需离散采样；同时通过符号计算获得精确雅可比，从而提升优化效率。

**🔧 技术方法**

使用的技术包括 CasADi 符号优化框架、复合 Bernstein 直接离散化、测地线方程（Christoffel 符号）、高斯成本表面、IPOPT 求解器和 MA57 子求解器。

**📊 数据集**

实验使用人工生成的二维和三维障碍场（分别有12个和15个障碍物），无公开数据集。

**📈 对比分析**

与不加测地线约束的长度最小化规划相比，测地线约束的求解时间约为10–100倍；在更复杂的 OCP 中，以测地线或测地线近似轨迹热启动可显著降低求解时间，尤其在高阶分段时。

**⚠️ 局限性**

局限性包括对低分辨率（K 较小）时求解失败率高、对动态障碍支持不足、以及需要先生成合适的高斯成本表面参数；此外，对更高维度或非欧几里得约束的推广仍需研究。

---

## 25. A Human-in-the-Loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies

**arXiv ID:** 2602.10289 | [PDF](https://arxiv.org/pdf/2602.10289v1)

**作者:** Rohan Banerjee `[一作]` (Cornell University), Tapomayukh Bhattacharjee `[通讯]` (Cornell University)

**通讯引用:** 1348 | [OpenAlex ID](https://openalex.org/A5026083794)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种人机协作的故障恢复框架，用于模块化机器人策略，框架通过模块选择器确定需询问的模块，并通过查询算法决定何时与人交互或自主执行。

**💡 创新点**

创新点在于：①将模块级不确定性与人类干预成本模型相结合，实现自适应查询；②将两项决策（模块选择与查询时机）分离，分别优化；③在多种模拟和真实助餐实验中验证了该框架的有效性。

**🔧 技术方法**

技术包括：模块图表示、置信度校准、混合整数规划、Brute‑Force、图论（Binary Tree Query / Graph Query）模块选择器；查询策略包括 Execute‑First、Query‑Then‑Execute、Query‑Until‑Confident 及其工作负载感知变体；使用置信度阈值和成本阈值做决策。

**📊 数据集**

数据集与实验：①合成模块仿真（随机模块图、逻辑门）；②机器人辅助咬取任务，使用四个模块（GPT‑4o、GroundingDINO、RT‑1 等）；③两场实验室研究（10 人、无移动障碍；10 人、仿真移动障碍）和一场真实家庭研究（2 人、严重移动障碍），评估主观与客观指标。

**📈 对比分析**

与永不询问、始终询问、基于置信度的基线相比，框架在任务成功率上最高，用户查询负担最低，主观满意度显著提升；在合成实验中显示在模块数、冗余结构、置信度、查询成本等维度下均具备鲁棒性，且计算复杂度可控。

**⚠️ 局限性**

局限性包括：置信度校准仍不完美，需进一步改进；框架主要针对非冗余或简单冗余结构，复杂多重冗余情形待扩展；实验规模有限，用户样本量小，未覆盖更复杂食物或长期交互；对实时性能的评估仍需进一步验证。

---

## 26. Flow caching for autoregressive video generation

**arXiv ID:** 2602.10825 | [PDF](https://arxiv.org/pdf/2602.10825v1)

**作者:** Yuexiao Ma `[一作]` (ByteDance), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 31753 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对自回归视频生成模型提出了一套名为 FlowCache 的缓存框架，能显著加速视频生成。

**💡 创新点**

创新点在于：①为每个视频块单独制定缓存策略，利用块级相似度动态决定重用与重算；②结合重要性与冗余度的 KV 缓存压缩方案，保持固定内存占用同时不牺牲质量。

**🔧 技术方法**

采用 Transformer 结构的自回归扩散模型、相对 L1 距离度量、分块缓存算法、KV 缓存重要性/冗余度联合评分以及量化压缩等技术。

**📊 数据集**

主要使用 MAGI-1‑4.5B-distill 和 SkyReels‑V2‑1.3B‑540P 两个公开模型，并在 VBench、LPIPS、PSNR、SSIM 等指标上评测。

**📈 对比分析**

与原始模型和 TeaCache 进行对比，FlowCache 在 MAGI‑1 上实现 2.38× 的加速（VBench 0.87% 提升），在 SkyReels‑V2 上实现 6.7× 的加速（VBench 0.79% 降低），同时保持或提升了视频质量。

**⚠️ 局限性**

局限性包括：需要针对不同模型手动调节阈值和缓存预算，且目前仅在自回归扩散模型上验证，尚未证明对其他类型生成模型的通用性。

---

## 27. ISD-Agent-Bench: A Comprehensive Benchmark for Evaluating LLM-based Instructional Design Agents

**arXiv ID:** 2602.10620 | [PDF](https://arxiv.org/pdf/2602.10620v1)

**作者:** YoungHoon Jeon `[一作]` (Upstage), Unggi Lee `[通讯]` (Korea University)

**通讯引用:** 315 | [OpenAlex ID](https://openalex.org/A5066209480)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ISD-Agent-Bench，构建包含51维情境变量和33个ADDIE子步骤的情境矩阵框架，生成25,795个多样化的教学设计场景；

**💡 创新点**

首次将传统ISD理论（ADDIE、Dick & Carey、Rapid Prototyping）与现代ReAct式智能体架构结合，并采用多评审协议降低LLM-as-judge偏差；

**🔧 技术方法**

采用ReAct、工具调用（14工具或5阶段工具）、多轮交互与GPT‑4o、Gemini‑3‑Flash、Solar‑Pro3等大模型；

**📊 数据集**

使用从SCOPUS论文抽取的8,842个学术摘要为种子，再通过GPT‑4o生成情境，补充16,953个人工合成样本，最终形成训练集24,593例、测试集1,017例；

**📈 对比分析**

通过ADDIE rubrics（70%）+轨迹评估（30%）的两阶段多评审评分，React‑ADDIE在测试集上平均得分86.49，显著优于纯理论或仅技术的基线；

**⚠️ 局限性**

局限性包括：场景生成仍偏重学术论文范例导致域偏移；评审者为LLM，尽管多样化但仍可能产生系统偏差；缺乏真实教学环境验证，难以评估长期学习效果。

---

## 28. Neuro-symbolic Action Masking for Deep Reinforcement Learning

**arXiv ID:** 2602.10598 | [PDF](https://arxiv.org/pdf/2602.10598v1)

**作者:** Shuai Han `[一作]` (Utrecht University), Shihan Wang `[通讯]` (Utrecht University)

**通讯引用:** 785 | [OpenAlex ID](https://openalex.org/A5109584106)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将符号推理与深度强化学习结合的框架NSAM，利用概率句法决策图(PSDD)学习高维状态的符号表征，并根据动作前置条件生成动作掩码，实现在训练和执行中有效减少违反域约束的动作。

**💡 创新点**

创新点包括：①在缺乏完整符号描述的环境中，通过最小监督（仅记录动作是否导致约束违反）学习符号表征；②使用PSDD保证学习出的符号模型始终满足给定的逻辑约束；③将符号推理与梯度基强化学习无缝融合，实现端到端可微训练；④通过动作掩码提高样本效率与安全性。

**🔧 技术方法**

核心技术：概率句法决策图(PSDD)、符号前置条件表示、最小监督的PSDD参数学习、PPO强化学习与动作掩码重归一化、线性时间MAP推理、端到端训练框架。

**📊 数据集**

在四个典型约束强化学习任务上评估：Sudoku（3×3、4×4、5×5）、N‑Queens（4、6、8、10）、Graph Coloring（四种图）以及Visual Sudoku（基于MNIST图像的2×2、3×3、4×4、5×5）。

**📈 对比分析**

与Rainbow、PPO、KCAC、PLPG、PPO‑Lagrangian和RC‑PPO等基线进行比较。实验表明NSAM在所有任务上都取得了更高的样本效率、更低的约束违例率，并在最终奖励上与或优于所有对手。

**⚠️ 局限性**

局限性：需要先验的命题形式符号知识，PSDD编译为离线开销较大；对极大比例命题或连续动作空间的适用性尚未验证；当约束未知或错误时框架的鲁棒性有限；目前仅在离散动作空间上验证。

---

## 29. Targeted Syntactic Evaluation of Language Models on Georgian Case Alignment

**arXiv ID:** 2602.10661 | [PDF](https://arxiv.org/pdf/2602.10661v1)

**作者:** Daniel Gallagher `[一作]` (Institute for Applied Informatics), Gerhard Heyer `[通讯]` (Institute for Applied Informatics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了格鲁吉亚语拆分‑ERG格位对齐的最小对称测试集，并评估七种 Transformer 语言模型在该任务上的性能。

**💡 创新点**

采用基于 Grew 查询的树库方法自动生成最小对称测试集，聚焦极为罕见的 ergative 格位，公开数据集并为低资源语言提供可复现的评估范式。

**🔧 技术方法**

使用 XLM‑RoBERTa、BERT、RemBERT、HPLT‑BERT、GPT2‑geo、mGPT‑ka 等 Transformer 模型，并通过词级和句子级概率比较实现语法正确性判定。

**📊 数据集**

利用 Georgian Language Corpus Universal Dependencies（GLC UD）3,013 句子生成 370 个测试样本（7 个子集，每个 50–70 例），包含 nominative、ergative、dative 三种格位。

**📈 对比分析**

通过词级和句子级准确率评估，模型在 nominative 语法判断上最高（词级≈88%），dative 约 49%，ergative 仅约 24%，表明 ergative 语法学习最为困难。

**⚠️ 局限性**

测试样本受树库规模限制，部分样本可能已出现在训练集；未对句子长度或未见数据进行严格控制，影响对模型泛化能力的评估。

---

## 30. Evaluating Numerical Accuracy in Mixed-Precision Computing by Dual-Delta Testing

**arXiv ID:** 2602.10605 | [PDF](https://arxiv.org/pdf/2602.10605v1)

**作者:** Peichen Xie `[一作]` `[通讯]`, Peichen Xie

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出了双Delta测试方法，用高精度oracle衡量两实现的误差分布以评估混合精度数值准确性。

**💡 创新点**

创新点在于将误差评估转化为两条误差分布对比而非单一误差指标，允许进行统计检验和稳定性分析。

**🔧 技术方法**

采用统计描述、非参数检验（如Wilcoxon、Kolmogorov‑Smirnov）、可视化（直方图、箱线图、Q‑Q图）以及FP16/FP64算子实现来构建测试框架。

**📊 数据集**

以矩阵乘法为例，使用随机生成的FP16矩阵（128×128和128×4096×4096×128）进行实验验证。

**📈 对比分析**

通过比较GPU/CPU FP16与FP64 oracle的误差分布，发现GPU默认的FP16降精度累加导致误差放大，禁用该优化后误差分布与CPU几乎一致；性能方面验证了GPU实现的加速优势。

**⚠️ 局限性**

局限在于需要高精度oracle保证误差可比、输入生成的覆盖面有限、对非线性或非平滑算子评估困难，且统计检验对极端异常值敏感。

---

## 31. Language Model Inversion through End-to-End Differentiation

**arXiv ID:** 2602.11044 | [PDF](https://arxiv.org/pdf/2602.11044v1)

**作者:** Kevin Yandoka Denamganaï `[一作]` (School of Informatics), Kartic Subr `[通讯]` (School of Informatics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于分布视角的可微分语言模型（DLM），通过软嵌入和Gumbel‑Softmax梯度估计实现LM的端到端可微分，从而将语言模型反演（LMI）问题转化为梯度优化；

**💡 创新点**

核心创新在于将LM视为从序列分布到序列分布的函数，并引入可学习、逐词解耦的温度参数以及教师强制（Teacher‑Forcing）策略，显著提升梯度估计的稳定性与效率；

**🔧 技术方法**

使用软嵌入、Gumbel‑Softmax（可学习温度）、梯度下降、Teacher‑Forcing以及随机目标生成（Difficulty‑Driven Target Generation）等技术；

**📊 数据集**

在SmolLM2‑135M和SmolLM3‑3B两大规模模型上，以多难度（k∈[1,6,11,16,21]）的自动生成目标序列作为评测数据集；

**📈 对比分析**

与REINFORCE、GBDA、SODA等基线对比，DLMI在不同prompt长度（N=10/80）和优化步数（256/2048）下均取得最高的LCS比率，尤其在长prompt和大模型上实现近乎完美的反演；

**⚠️ 局限性**

局限性包括：仅评估了特定模型架构和词表规模，算法对GPU算力敏感；在极短prompt或极难目标下仍存在收敛缓慢的问题；缺乏对非英语多语言的验证。

---

## 32. Llama-Polya: Instruction Tuning for Large Language Model based on Polya's Problem-solving

**arXiv ID:** 2602.10597 | [PDF](https://arxiv.org/pdf/2602.10597v1)

**作者:** Unggi Lee `[一作]` (Korea University), Minji Jeon `[通讯]` (University of Nebraska)

**通讯引用:** 255 | [OpenAlex ID](https://openalex.org/A5069886170)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将波利亚四步法嵌入LLM对话结构，训练Llama-Polya实现阶段化数学问题引导和元认知提示。

**💡 创新点**

以教育理论为核心进行指令微调，将波利亚框架直接嵌入模型行为，提升教学一致性与思维透明度。

**🔧 技术方法**

使用LLaMA‑3.1‑8B全参数微调、ChatML模板、人工设计的Polya阶段化提示，未采用LoRA/QLoRA等PEFT。

**📊 数据集**

使用合成的基于GSM8K的Polya对话数据，结合Meta Instruction、Metamath、Polya‑v2等数据集。

**📈 对比分析**

通过对比基线、通用指令、数学域、Polya‑v2、序列微调等五种模型，评估阶段分布、错误率和专家Likert评分；Polya‑v2在各阶段平衡、错误率最低，专家认为结构与鼓励性良好。

**⚠️ 局限性**

局限在于缺乏个性化与数学严谨性、误差纠正表面化、序列微调导致的遗忘现象，仍需改进适应性与深度反思。

---

## 33. See, Plan, Snap: Evaluating Multimodal GUI Agents in Scratch

**arXiv ID:** 2602.10814 | [PDF](https://arxiv.org/pdf/2602.10814v1)

**作者:** Xingyi Zhang `[一作]` (East China Normal University), Xiangfeng Wang `[通讯]` (East China Normal University)

**通讯引用:** 2401 | [OpenAlex ID](https://openalex.org/A5101927070)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ScratchWorld基准，用于评估多模态GUI代理在块式编程环境中的程序构造能力；

**💡 创新点**

创新点在于引入双模式评估（原始低级操作与高级语义API）以分离推理与执行，采用执行式验证脚本保证功能正确性，并对单步拖拽与视觉感知进行细粒度诊断；

**🔧 技术方法**

使用多模态大语言模型（GPT‑5、Claude‑Sonnet‑4.5、Gemini‑2.5‑Pro等）、视觉专用模型（Qwen3‑VL‑32B‑Instruct、UI‑TARS‑1.5‑7B）、以及基于Playwright与Scratch VM的执行评估框架；

**📊 数据集**

构建了83个任务，覆盖Create、Debug、Extend、Compute四类，任务包括自然语言说明、初始项目、金标准项目和JavaScript单元测试脚本；

**📈 对比分析**

与多种现有大模型和代理框架对比，Composite模式下Claude‑Sonnet‑4.5达78.31%成功率；Primitive模式下仅为14.46%，单步拖拽成功率低于25%，显示显著的推理–执行缺口；

**⚠️ 局限性**

主要限制在细粒度的拖拽坐标定位与“下落”目标识别，视觉感知虽已达到高精度，但执行精度不足导致大多数任务失败；

---

## 34. On the Robustness of Knowledge Editing for Detoxification

**arXiv ID:** 2602.10504 | [PDF](https://arxiv.org/pdf/2602.10504v1)

**作者:** Ming Dong `[一作]` (Hubei Provincial Key Laboratory of Artificial Intelligence and Smart Learning, National Language Resources Monitoring and Research Center for Network Media, Central China Normal University), Tingting He `[通讯]` (Hubei Provincial Key Laboratory of Artificial Intelligence and Smart Learning, National Language Resources Monitoring and Research Center for Network Media, Central China Normal University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出了针对基于知识编辑（KE）的毒性消除（detoxification）方法的鲁棒性评估框架，并在此框架下系统评估了不同模型、方法和语言的表现。

**💡 创新点**

创新点包括：①将评估拆分为优化鲁棒性、组合鲁棒性和跨语言鲁棒性三维度；②首次识别并量化伪消毒（pseudo‑detoxification）以及毒性消除与生成退化之间的权衡；③构建了跨语言的 mSafeEdit 数据集并引入了多语言安全分类器与退化检测器。

**🔧 技术方法**

使用的技术主要包括知识编辑方法 DINM 与 FT‑M、微调优化、毒性检测器（多语言安全分类器）以及重复检测的退化判别器；评估过程中对编辑超参数（学习率、编辑步数）进行调优，并对多目标和跨语言场景进行累计编辑实验。

**📊 数据集**

采用的数据集有：SafeEdit、LinguaSafe 以及其翻译版本 mSafeEdit（覆盖 8 种语言），共 100 条样本；此外使用公开的多语言安全分类器与 GPT‑4o‑mini/Claude‑3.5‑Haiku 进行安全性评估。

**📈 对比分析**

比较方法：在多模型（Llama2‑7B、Llama3‑8B、Mistral‑7B、Ministral‑8B、Qwen2‑7B、Qwen3‑8B）与两种 KE 方法（DINM、FT‑M）下，分别评估伪消毒率、退化率、对 OOD 和跨语言攻击的防御成功率。结果显示：只有 Qwen2‑7B 在 DINM 下在单语言和跨语言场景中表现相对稳健；其他模型或方法在多目标或低资源语言下效果显著下降，甚至出现负面影响。

**⚠️ 局限性**

局限性：①仅评估两种 KE 方法，未覆盖更广泛的编辑技术；②实验样本量有限（100 条），可能影响统计显著性；③依赖自动安全分类器，仍可能误判；④跨语言评估主要集中在英文训练的模型，缺乏对低资源语言专门优化的探究。

---

## 35. MERIT Feedback Elicits Better Bargaining in LLM Negotiators

**arXiv ID:** 2602.10467 | [PDF](https://arxiv.org/pdf/2602.10467v1)

**作者:** Jihwan Oh `[一作]` (KAIST AI), Taehyeon Kim `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了面向买方的多场景谈判基准AgoraBench，并提出了人类偏好对齐的Merit评估指标。

**💡 创新点**

创新点在于整合经济学理论与多维度人类偏好，创建九种真实市场环境和人类偏好数据集，并用Merit指导LLM的对话生成。

**🔧 技术方法**

采用ReAct思考-行动框架、Merit反馈的ICL-MF以及基于人类对话的LoRA微调等技术。

**📊 数据集**

使用自建的AgoraBench模拟数据以及来自Amazon Mechanical Turk的人类偏好对话数据。

**📈 对比分析**

与ReAct和OG-Narrator对比，Merit引导的ICL-MF和SFT在单品与多品场景中显著提升Merit分数和成交率，表现优于基线。

**⚠️ 局限性**

局限性包括仅聚焦买方目标、未覆盖卖方视角、缺乏监管与文化因素，以及未结合工具增强的环境。

---

## 36. DEGMC: Denoising Diffusion Models Based on Riemannian Equivariant Group Morphological Convolutions

**arXiv ID:** 2602.10221 | [PDF](https://arxiv.org/pdf/2602.10221v1)

**作者:** El Hadji S. Diop `[一作]` (University Iba Der Thiam), Mohamed Daoudi `[通讯]` (Institut Mines-Telecom Nord Europe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种融合Riemannian流形上等变群形态卷积与对流-膨胀-腐蚀块（ResnetCDEBlock）的去噪扩散模型DEGMC；

**💡 创新点**

创新点在于将等变群形态卷积与Hamilton–Jacobi PDE相结合，使模型在保持平移、旋转、反射、置换等Euclidean群对称性的同时，显著提升细节与非线性特征提取；

**🔧 技术方法**

主要技术包括PDE‑G‑CNN、Hamilton–Jacobi方程的粘性解、超球面（或双曲球）几何、GMCUnet架构、以及对流-膨胀-腐蚀操作；

**📊 数据集**

使用了MNIST、旋转MNIST（RotoMNIST）和CIFAR‑10三组图像数据集；

**📈 对比分析**

与传统DDPM基线对比，DEGMC在MNIST上FID从36.41降至30.94，在RotoMNIST上从44.74降至35.75，在CIFAR‑10上从25.59降至25.14，并在训练迭代数上实现更快收敛；

**⚠️ 局限性**

局限性包括仅在2D图像数据上验证，未与更先进的扩散模型比较，且对高维或3D数据的适用性尚待探索，模型计算开销相对传统U‑Net也可能更大。

---

## 37. Macaron: Controlled, Human-Written Benchmark for Multilingual and Multicultural Reasoning via Template-Filling

**arXiv ID:** 2602.10732 | [PDF](https://arxiv.org/pdf/2602.10732v1)

**作者:** Alaa Elsetohy `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Alham Fikri Aji `[通讯]` (Mohammed Bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个模板优先、跨20国20语10脚本的多语言多文化推理基准（Macaron），共计11,862个场景对齐的多选与对应的真/假题；

**💡 创新点**

创新点在于将推理类型（7种）与文化维度（22种）与语言独立分离的模板框架，能够快速生成可对齐的双语题目，并通过系统化的真/假派生进一步检验模型对文化事实的验证能力；

**🔧 技术方法**

采用100个无关语言的模板，人工本地化填充并翻译，随后利用LLM进行拼写/语法校对；对21个多语言LLM进行零样本评估，分析按语言、推理和文化维度的表现；

**📊 数据集**

数据集为自建的Macaron benchmark，覆盖20个文化语境、10种文字脚本，包含多选与真/假两种格式；

**📈 对比分析**

通过零样本评估，闭源“思考”模型平均准确率达79.3%，与英文版本几乎无差距；开源权重模型平均55.2%，且在低资源语言与真/假题上表现显著下降；数学与计数类题目普遍最难；

**⚠️ 局限性**

局限性包括仅覆盖20个主要文化与单一主语言，未涵盖同一国家内的多样性与方言；任务形式受限于多选/真/假，无法反映开放式对话与工具使用场景；并可能因模板或翻译导致的刻板印象与信息不足。

---

## 38. Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings

**arXiv ID:** 2602.10495 | [PDF](https://arxiv.org/pdf/2602.10495v1)

**作者:** Tianxiang Dai `[一作]` (Stanford University), Jonathan Fan `[通讯]` (Stanford University)

**通讯引用:** 16813 | [OpenAlex ID](https://openalex.org/A5074787051)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过对多分辨率哈希编码（MHE）的点扩散函数（PSF）进行物理系统分析，揭示其内在的各向异性与优化导致的空间扩展，并基于此提出了旋转MHE（R-MHE）以改善各向异性；

**💡 创新点**

创新点在于：①从PSF角度系统阐释MHE空间特性及其对有效分辨率的影响；②量化优化过程中的空间扩展因子和哈希碰撞对SNR的影响；③提出无额外参数的R-MHE，通过在不同分辨率层引入旋转实现等向化；

**🔧 技术方法**

主要技术包括：基于线性化MLP与最小范数理论推导MHE的闭式PSF；使用数值实验验证优化扩展和碰撞噪声；利用旋转策略在各层实现等向化；

**📊 数据集**

在2D图像回归、3D Neural Radiance Fields（Synthetic NeRF）以及3D Signed Distance Function（Armadillo、Bunny、Spot）等数据集上进行评估；

**📈 对比分析**

与传统MHE（使用经验性增长因子b）和多种旋转策略（正多面体方向）比较，R-MHE在2D任务中提升约0.9 dB PSNR，3D NeRF中提高约0.13 dB，SDF任务几乎无显著差异；

**⚠️ 局限性**

局限性包括：3D体渲染中视线积分对高频噪声的平均效应削弱了R-MHE的优势；在高分辨率设置下大多数基准已达到饱和，难以显现明显改进。

---

## 39. Healthy Harvests: A Comparative Look at Guava Disease Classification Using InceptionV3

**arXiv ID:** 2602.10967 | [PDF](https://arxiv.org/pdf/2602.10967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 40. (MGS)$^2$-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization

**arXiv ID:** 2602.10704 | [PDF](https://arxiv.org/pdf/2602.10704v1)

**作者:** Minglei Li `[一作]` (Tsinghua University), Ziyang Meng `[通讯]` (Tsinghua University)

**通讯引用:** 6339 | [OpenAlex ID](https://openalex.org/A5051392570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个基于三维几何约束的跨视角地理定位框架（MGS）²，结合宏观结构过滤（MGSF）与微观尺度自适应（MGSA），实现从UAV机载图像到卫星图像的高精度匹配。

**💡 创新点**

创新点在于首次通过膨胀几何梯度物理过滤垂直立面噪声，利用深度先验动态校正尺度差异，并设计几何-外观对比蒸馏损失以强化空间一致性。

**🔧 技术方法**

采用Depth Anything 3做深度估计，Dilated Sobel卷积提取宏观梯度，DINOv2作为Transformer骨干，深度感知多尺度融合模块，和对比蒸馏损失实现三维结构对齐。

**📊 数据集**

实验使用University‑1652、SUES‑200以及跨域DenseUAV数据集进行评估。

**📈 对比分析**

与TransFG、Sample4Geo、CAMP等现有SOTA方法对比，MGS²在University‑1652上Recall@1达97.5%，在SUES‑200上Recall@1达97.02%，并在DenseUAV上实现高达81.7%的零样本召回，显著优于其他方法。

**⚠️ 局限性**

局限性包括对单目深度估计的依赖导致噪声敏感，模型较大且计算量高，且在极端光照或复杂纹理环境下仍可能出现误匹配。

---

## 41. Equity by Design: Fairness-Driven Recommendation in Heterogeneous Two-Sided Markets

**arXiv ID:** 2602.10739 | [PDF](https://arxiv.org/pdf/2602.10739v1)

**作者:** Dominykas Seputis `[一作]` (University of Amsterdam), Alexander Timans `[通讯]` (University of Amsterdam)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5050193138)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文在现实异质双边市场中，对离散多项推荐、消费者群体公平与业务约束进行统一建模与实验验证；

**💡 创新点**

创新点在于把多项离散分配、CVaR群体公平与GMV业务约束整合入优化框架，证明单项“免费公平”不再适用，并提供可扩展的求解方案；

**🔧 技术方法**

采用混合整数规划、LP放松+舍入、增广拉格朗日与梯度优化，并引入CVaR与业务约束；

**📊 数据集**

实验数据来源于MovieLens‑100k、Amazon Reviews及自建的SimRec仿真数据集；

**📈 对比分析**

与基准MIP、LP+舍入、AugLag、SCGrad进行对比，LP/AugLag在公平度与均值效用上可与MIP持平，SCGrad略逊；在运行时长上GPU加速实现1.5×提升；公平约束提升业务指标（STR、GMV），但在k>1时公平会显著降低消费者平均效用；

**⚠️ 局限性**

局限性包括仅在离线仿真验证、固定群体划分、忽略位置偏差、梯度方法偶尔违背约束、缺乏实时实验验证与自适应调参机制。

---

## 42. GPU-Fuzz: Finding Memory Errors in Deep Learning Frameworks

**arXiv ID:** 2602.10478 | [PDF](https://arxiv.org/pdf/2602.10478v1)

**作者:** Zihao Li `[一作]` (Southern University of Science and Technology), Fengwei Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 1439 | [OpenAlex ID](https://openalex.org/A5101886601)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出GPU‑Fuzz，利用约束建模自动发现深度学习框架中的GPU内存错误。

**💡 创新点**

创新点在于将GPU算子参数抽象为形式约束并使用约束求解器生成针对边界条件的测试用例。

**🔧 技术方法**

技术包括约束建模、SMT求解、自动化生成测试用例与GPU内核执行。

**📊 数据集**

使用的“数据集”是PyTorch、TensorFlow和PaddlePaddle的算子参数空间，无需外部数据集。

**📈 对比分析**

与手工测试或随机fuzz相比，GPU‑Fuzz系统化地定位错误，成功发现13个未知缺陷。

**⚠️ 局限性**

局限性包括对动态生成的算子支持不足、求解器规模受限以及对特定框架的依赖。

---

## 43. Protecting Context and Prompts: Deterministic Security for Non-Deterministic AI

**arXiv ID:** 2602.10481 | [PDF](https://arxiv.org/pdf/2602.10481v1)

**作者:** Mohan Rajagopalan `[一作]` (MACAW Security), Vinay Rao `[通讯]` (ROOST.tools)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两种加密原语——认证提示与认证上下文，构建可验证的LLM代理工作流；

**💡 创新点**

创新点在于将指令生成与验证分离，提供完整的指令血统与上下文完整链，配合形式化策略代数与四个定理，实现零误报、100%攻击检测并实现拜占庭级别的协议安全；

**🔧 技术方法**

使用数字签名、哈希链、策略交叉（最小上界）代数、分布式策略执行点、语义意图验证与深度限制等技术；

**📊 数据集**

采用人工构造的六类攻击样本进行评估，未使用公开数据集；

**📈 对比分析**

与传统输入过滤、政策框架、训练对齐等方法对比，检测率为100%，误报率为0%，性能开销仅约1.8%；

**⚠️ 局限性**

局限在于需人工编写完整政策，若政策不完整或不正确，安全保障无法保证；

---

## 44. Direct Learning of Calibration-Aware Uncertainty for Neural PDE Surrogates

**arXiv ID:** 2602.11090 | [PDF](https://arxiv.org/pdf/2602.11090v1)

**作者:** Carlos Stein Brito `[一作]` `[通讯]` (NightCity Labs), Carlos Stein Brito (NightCity Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了跨正则化（XReg）不确定性学习框架，在神经PDE代理中通过训练/正则化划分分别更新预测器与不确定性参数，得到自适应且已校准的预测分布。

**💡 创新点**

创新点在于将不确定性视为可学习的目标，并通过在正则化样本上路由梯度来更新泛化不确定性，从而无需手动调节噪声强度即可实现数据稀缺或观测不足时的自适应不确定性缩放。

**🔧 技术方法**

采用傅里叶神经算子（FNO）骨干，结合输出头与内部层的乘性高斯噪声注入，使用混合似然、NLL与ECE等指标训练并评估模型，并通过交叉正则化算法实现双噪声参数的分离学习。

**📊 数据集**

实验使用一维APEBench时间序列和OTNO车表面压强数据，构造一阶教师强制对偶样本以及不同观测比例和训练集大小的划分。

**📈 对比分析**

与MC Dropout（p=0.1）和三成员深度集成等基线在匹配观测比例下比较，XReg在保持相近一阶NLL的同时，混合ECE显著更低，显示出更优的校准性能。

**⚠️ 局限性**

局限性包括仅在单步教师强制评估下验证，未针对长周期自回归推断或更复杂的物理场进行验证；同时仍缺乏对高度混沌或非线性扩散等极端情形的理论与实验支持。

---

## 45. From Representational Complementarity to Dual Systems: Synergizing VLM and Vision-Only Backbones for End-to-End Driving

**arXiv ID:** 2602.10719 | [PDF](https://arxiv.org/pdf/2602.10719v1)

**作者:** Sining Ang `[一作]` (University of Science and Technology of China), Yan Wang `[通讯]` (Tsinghua University)

**通讯引用:** 26676 | [OpenAlex ID](https://openalex.org/A5100322712)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究在端到端自动驾驶中如何融合大型视觉语言模型（VLM）与传统视觉骨干，利用表示分析揭示两者的差异，并通过轨迹级选择实现性能提升。

**💡 创新点**

创新点包括：① 引入共享‑唯一稀疏自编码器（SAE）对比两分支的共享与特异子空间；② 证明决策层表示比骨干层更可对齐；③ 在轨迹级构造“风格轴”插值候选集并使用学习的轨迹评分器实现最佳轨迹选择；④ 设计快慢双路部署（DualDriveVLA），在保持性能的同时显著提升推理吞吐。

**🔧 技术方法**

技术主要有：线性 CKA 与 PCA‑白化 CCA、共享‑唯一 SAE、Diffusion Transformer（DiT）规划器、轨迹评分网络（DrivoR‑style）、自回归多分支联合训练与快慢路策略。

**📊 数据集**

使用 NAVSIM 公开仿真测试集（navtest）评估驾驶性能，采用 PDMS/EPDMS 指标。

**📈 对比分析**

与单一 VLM 或 ViT 基线相比，HybridDriveVLA 将 PDMS 提升至 92.10（+1.30%），DualDriveVLA 在 15% 触发 VLM 的情况下达 91.00（+0.20%），并实现 3.2× 的吞吐提升；在 NAVSIM‑v2 的 EPDMS 也实现了相近的提升。

**⚠️ 局限性**

局限性包括：① 需要在每个场景上运行两条路径或插值，仍有计算开销；② 轨迹评分器需要额外训练，性能受评分准确性影响；③ 研究仅在 NAVSIM 仿真环境验证，缺乏真实世界测试；④ 对极端场景的泛化性尚未充分评估。

---

## 46. SnapMLA: Efficient Long-Context MLA Decoding via Hardware-Aware FP8 Quantized Pipelining

**arXiv ID:** 2602.10718 | [PDF](https://arxiv.org/pdf/2602.10718v1)

**作者:** Yifan Zhang `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SnapMLA 框架，实现 FP8 量化的 MLA 解码，以提升长上下文推理效率。

**💡 创新点**

创新点在于 RoPE‑aware per‑token KV 量化、PV 量化尺度融合重构以及端到端数据流优化，解决了 MLA 的数值异构和硬件对齐问题。

**🔧 技术方法**

采用 FP8（E4M3）量化、Hopper Tensor Core 的 WGMMA/WGMMA 指令、RoPE 预缩放、块级动态量化、软最大化 implicit dequant、融合的 CUDA kernel、缓存对齐与寄存器级转置等技术。

**📊 数据集**

使用 DeepSeek‑V3.1、LongCat‑Flash‑Thinking 两大 MLA LLM 以及 MMLU、IFEval、Arena‑Hard、MATH、AIME、LCB 等公开评测数据集。

**📈 对比分析**

与 FlashMLA 基准对比，SnapMLA 在 FP8 下保持 0~0.2% 误差，同时在 DP8/TP1 组合下生成吞吐量提升 1.91×，在多领域任务中几乎无精度损失。

**⚠️ 局限性**

局限在于仅针对 Hopper GPU 设计，需支持 8‑bit FP8 计算；对极端长序列仍受内存布局限制，且对非 MLA 结构迁移需进一步改造。

---

## 47. Enhancing YOLOv11n for Reliable Child Detection in Noisy Surveillance Footage

**arXiv ID:** 2602.10592 | [PDF](https://arxiv.org/pdf/2602.10592v1)

**作者:** Khanh Linh Tran `[一作]` (Posts and Telecommunications Institute of Technology), Linh Nguyen Kieu `[通讯]` (Posts and Telecommunications Institute of Technology)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5046590553)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在低质量监控视频中通过细化YOLOv11n模型，构建了面向儿童检测的轻量级管线；

**💡 创新点**

提出了基于真实场景的域特定数据增强方法（合成儿童剪裁、遮挡、截断、光照、噪声等），并在推理阶段结合SAHI以显著提升小目标召回率；

**🔧 技术方法**

采用YOLOv11n的迁移学习、人工设计的多层级图像合成与退化增强、SAHI切片推理、NMS等技术；

**📊 数据集**

基于Roboflow Daycare数据集的儿童单类别子集进行训练和评估；

**📈 对比分析**

通过与原始YOLOv11n基线在同一测试集（mAP@0.5/0.5:0.95）对比，增强+SAHI模型实现mAP@0.5 0.967（+0.7%）和mAP@0.5:0.95 0.783（+2.3%），保持实时低功耗性能；

**⚠️ 局限性**

受限于单摄像机数据背景单一、跨视角泛化不足，且实验仅覆盖单类别儿童检测，提升幅度相对有限，未来需扩展多摄像机和更大多样化数据集以进一步验证效果。

---

## 48. ICODEN: Ordinary Differential Equation Neural Networks for Interval-Censored Data

**arXiv ID:** 2602.10303 | [PDF](https://arxiv.org/pdf/2602.10303v1)

**作者:** Haoling Wang `[一作]` (University of Pittsburgh), Ying Ding `[通讯]` (University of Pittsburgh)

**通讯引用:** 16366 | [OpenAlex ID](https://openalex.org/A5047170063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

开发了一种基于常微分方程的神经网络ICODEN，用于无比例风险假设、可处理高维预测因子且可兼顾左截断的区间删失生存分析。

**💡 创新点**

创新点在于将危害函数直接用神经网络建模，并通过求解ODE得到累计危害，既消除了比例风险假设，又能捕捉非线性、非比例风险特征，同时支持高维基因或影像数据。

**🔧 技术方法**

技术包括神经网络（全连接、ReLU+Softplus输出）、神经ODE求解（torchdiffeq）、梯度计算的逆向传播、L1正则化、早停与交叉验证等。

**📊 数据集**

使用阿尔茨海默病多期研究（ADNI）预测转为AD时间，以及两项AMD临床试验（AREDS/AREDS2）预测晚期AMD时间，均涉及数百到数千个SNP与临床变量。

**📈 对比分析**

与传统的区间删失Cox PH模型（IcenReg、APOE）和自适应LASSO进行比较，ICODEN在高维、非比例风险和非线性关系场景下表现最佳；在ADNI中与IcenReg相当或更优，且能在623 SNP时仍能训练；在AREDS/AREDS2中，ICODEN在IBS和d_out上优于所有比较方法。

**⚠️ 局限性**

局限包括：尚未处理竞争风险；模型解释性依赖后续解释技术；训练时间相对较长；仅在两类疾病数据上验证，缺乏跨领域通用性验证。

---

## 49. CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion

**arXiv ID:** 2602.10999 | [PDF](https://arxiv.org/pdf/2602.10999v1)

**作者:** Yusong Lin `[一作]` (Huawei Technologies Co), Dandan Tu `[通讯]` (Huawei Technologies Co)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 CLI-Gym，一个可公开使用的自动化管道，用于从 GitHub 仓库中生成大规模的环境密集型（CLI）任务，并基于该数据微调大型语言模型，显著提升了在 Terminal‑Bench 任务上的通用终端交互能力。

**💡 创新点**

创新点包括：①将 Dockerfile 视为可逆的环境变更序列，提出“环境反演”方法让代理主动破坏环境生成任务；②首次实现了全自动化、无人工干预的环境密集型任务采集；③通过仅 291 条高质量的恢复轨迹即可实现 32B 级别模型在 Terminal‑Bench 1.0 上 38.9%、1.0 级别 46.1% 的 Pass@1 成绩，超过多种更大规模闭源模型。

**🔧 技术方法**

核心技术包括：基于 LLM 的 agentic 任务生成（模拟环境演化）、Dockerfile 代码生成与逆向、自动化单元测试验证、OpenHands 代理框架进行轨迹收集、两阶段微调（先预训练 SWE 轨迹，再微调 CLI 轨迹）。

**📊 数据集**

使用了 29 个开源 Python 仓库产生 1,655 条 CLI 任务实例，结合 Terminal‑Bench 1.0 与 2.0 的 80/89 任务进行评测；轨迹来源为 291 条经过过滤的成功恢复过程。

**📈 对比分析**

在 OpenHands 评测框架下，LiberCoder‑32B 与 LiberCoder‑235B‑A22B 分别在 Terminal‑Bench 1.0 上取得 38.9% 与 46.1% 的 Pass@1，显著高于 Qwen3‑32B（10.3%）与 Qwen3‑235B‑A22B‑Instruct（25.0%），并在 1.0/2.0 上超过多种更大规模的开源模型；表明少量高质量轨迹即可带来大幅性能提升。

**⚠️ 局限性**

局限性包括：①仅覆盖 CLI 环境，未涉及更复杂的系统管理、网络配置或跨平台交互；②生成任务的多样性仍受限于所选仓库和 Dockerfile 规范；③轨迹收集与过滤仍需手工设计规则，可能遗漏有用轨迹；④模型在更高难度或长序列交互时易超出上下文长度，导致性能衰减。

---

## 50. Better Diameter Bounds for Efficient Shortcuts and a Structural Criterion for Constructiveness

**arXiv ID:** 2602.10747 | [PDF](https://arxiv.org/pdf/2602.10747v1)

**作者:** Bernhard Haeupler `[一作]` (Sofia University St. Kliment Ohridski and ETH Zurich), Zhijun Zhang `[通讯]` (Sofia University St. Kliment Ohridski)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过提出“certified shortcut”这一概念，重新表述了所有基于短路的并行有向连通性与最短路径算法的构造限制，并在此框架下给出了新的直径下界。

**💡 创新点**

创新点在于：① 以结构性判据形式定义了可由高效（组合）算法构造的短路；② 利用该判据推导出任意近线性时间短路算法的直径下界提升至 n^{1/4-o(1)}（相比之前的 n^{2/9-o(1)}）；③ 引入“certification complexity”度量，证明已知存在性证明的短路/跳跃集往往不可实现为高效算法；④ 在无权无向图中得到与加权图相同的直径下界，填补前人空白。

**🔧 技术方法**

技术方法主要为：基于层化 DAG、凸集与交替乘积的图构造；通过对每条关键路径的唯一性与交叉约束，推导出任何 certified shortcut 必须在该路径上加入至少 Ω(D) 条边；利用“certified shortcut”与“shortcutting procedure”之间的等价性，将构造问题转化为对构造过程的结构性分析；并结合“certification complexity”证明存在性构造的不可构造性。

**📊 数据集**

研究中不使用真实数据集，而是构造一系列具有特定层数、关键路径数与边数关系的人工图（如层化 DAG、凸集图）。

**📈 对比分析**

对比方法主要为已有的直径下界与构造性证明：本文将 n^{2/9} 的下界提升到 n^{1/4}，并在 n-大小短路与跳跃集上得到与加权图相同的 n^{1/3} 与 n^{1/2} 下界；证明了已知高效算法构造的短路必为 certified，因而本下界即为这些算法的最佳可达深度上界。

**⚠️ 局限性**

限制与展望：在目前技术下，n-大小短路的下界仍停留在 n^{1/3}，m-大小短路在 n^{1/4}，而更精细的下界需要全新的图构造方法；此外，certification complexity 的上界仅对已知算法给出，尚未能给出所有算法的严格上界；对无权无向图的进一步改进仍受限于当前的凸集与交替乘积构造。

---

## 51. LoRA-Squeeze: Simple and Effective Post-Tuning and In-Tuning Compression of LoRA Modules

**arXiv ID:** 2602.10993 | [PDF](https://arxiv.org/pdf/2602.10993v1)

**作者:** Ivan Vulić `[一作]` (DeepMind), Jonas Pfeiffer `[通讯]` (DeepMind)

**通讯引用:** 799 | [OpenAlex ID](https://openalex.org/A5024983536)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出LoRA‑Squeeze方法，先使用高rank LoRA进行过参数化训练，再通过随机SVD压缩为低rank LoRA，压缩过程可在训练后或训练期间动态完成；

**💡 创新点**

创新点在于将训练rank与部署rank解耦，利用随机SVD实现高效压缩，并给出内存友好的实现与渐进式（In‑Squeeze/Cont‑Squeeze）压缩策略；

**🔧 技术方法**

采用LoRA、随机奇异值分解（RSVD）与QR分解的记忆优化、任务向量概念以及过参数化微调技术；

**📊 数据集**

使用Gemma 3系列（4B、1B、12B）指令微调模型，涵盖13个文本评测任务和10个视觉‑语言问答任务；

**📈 对比分析**

通过与直接低rank LoRA微调做对比实验，发现Post‑Squeeze在多数任务上与或优于直接微调，尤其低rank时提升数个百分点；In‑Squeeze策略在多任务上表现最稳健，Cont‑Squeeze可在压缩后快速恢复性能；

**⚠️ 局限性**

局限包括需要先训练高rank LoRA（增加训练成本），压缩步可能导致信息损失导致性能崩溃，需要慎选源rank；目前仅验证标准LoRA，复杂变体未覆盖；随机SVD与大模型的内存消耗仍是潜在挑战。

---

## 52. RealHD: A High-Quality Dataset for Robust Detection of State-of-the-Art AI-Generated Images

**arXiv ID:** 2602.10546 | [PDF](https://arxiv.org/pdf/2602.10546v1)

**作者:** Hanzhe Yu `[一作]` (Zhejiang University of Technology), Chen Ma `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 27520 | [OpenAlex ID](https://openalex.org/A5100652421)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个高质量、覆盖多类别、包含730k张图像的AI生成图像与真实图像对照数据集RealHD，并提出基于图像噪声熵的轻量检测方法

**💡 创新点**

①数据集规模大、图像质量高、生成方式多样（文本→图像、修补、精化、换脸）并配有细粒度标签；②引入噪声熵特征捕捉传感器噪声差异，显著提升检测泛化能力

**🔧 技术方法**

利用稳定扩散、Flux等先进生成模型生成图像；使用非局部均值（NLM）提取噪声，再计算局部Shannon熵构成特征；训练基准模型（ResNet‑50、Xception、EfficientFormer、LNP）进行对比实验

**📊 数据集**

RealHD（730k+图像）与现有数据集GenImage、DiffusionForensics、DMimageDetection进行对照；还在Chameleon等跨域数据上评估模型泛化

**📈 对比分析**

在RealHD上训练的模型在跨域测试中均表现出最优的准确率与AUC，且噪声熵方法在所有网络架构上提升≈2-5%准确率，保持较好压缩鲁棒性

**⚠️ 局限性**

数据集仍有限制于所选生成模型与场景，缺乏更广泛的后处理与攻击手段；噪声熵特征在极端压缩或重采样下可能失效

---

## 53. Rethinking Security of Diffusion-based Generative Steganography

**arXiv ID:** 2602.10219 | [PDF](https://arxiv.org/pdf/2602.10219v1)

**作者:** Jihao Zhu `[一作]`, Xiaohua Xie `[通讯]` (Sun Yat-sen University)

**通讯引用:** 6688 | [OpenAlex ID](https://openalex.org/A5018298892)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并评估基于扩散模型的生成图像隐写技术的安全性，并提出一种基于噪声空间的检测框架 NS-DSer。

**💡 创新点**

通过理论证明噪声分布保持性与隐写安全的等价性，提出噪声空间检测框架 NS-DSer，并在四个递增难度场景下展示其优越性。

**🔧 技术方法**

使用 KL 散度理论分析、确定性无条件扩散采样、统计特征提取（均值、方差、偏度、峰度、IQR）、Fisher 线性判别分类器，以及对不同扩散模型、采样步长、CFG 的无条件设计。

**📊 数据集**

使用 Flickr8K 图像集生成训练和测试样本，采用 Stable Diffusion 1.5/2.1、DreamShaper 7 等公开扩散模型。

**📈 对比分析**

与 XuNet、SRNet、SiaStegNet、UCNet 等传统图像隐写分析器对比，NS-DSer 在四个检测场景中大多数方法的检测准确率超过 90%，整体平均性能提升至 97% 以上，并且训练时间最短。

**⚠️ 局限性**

对极少量噪声分布轻微扰动的 CRoSS 难以检测；在使用提示引导时性能略降；模型依赖扩散模型的可逆性，若反演精度受限会影响效果。

---

## 54. AI Infrastructure Sovereignty

**arXiv ID:** 2602.10900 | [PDF](https://arxiv.org/pdf/2602.10900v1)

**作者:** Sergio Cruzes `[一作]` `[通讯]` (Ciena Brazil), Sergio Cruzes (Ciena Brazil)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

探讨 AI 基础设施主权概念，并提出以 AI 数据中心、光网络和能源系统为核心的多层协同架构与治理框架。

**💡 创新点**

首次将 AI 主权从纯软件层面转移到物理与控制层面，融合遥测、代理 AI 与数字孪生实现实时闭环管理，构建可持续性驱动的自治控制体系。

**🔧 技术方法**

遥测流、实时可观测性平台、代理 AI（自治决策）、数字孪生模型、可持续性指标集成（碳强度、水消耗、能源可用性）以及多层光网络调度算法。

**📊 数据集**

无实验数据集；基于公开文献、行业案例与标准指标（如电力密度、碳强度、水耗）进行理论与案例分析。

**📈 对比分析**

通过案例对比传统软件主权与基础设施主权，展示后者在能耗、可持续性、延迟与故障域管理上的优势；未进行量化实验，对比主要以理论与行业指标为依据。

**⚠️ 局限性**

受限于当前硬件与光网络可用性，缺乏大规模实测验证；架构复杂度高，部署成本与跨组织治理协调成为主要挑战。

---

## 55. Computational Phenomenology of Temporal Experience in Autism: Quantifying the Emotional and Narrative Characteristics of Lived Unpredictability

**arXiv ID:** 2602.10947 | [PDF](https://arxiv.org/pdf/2602.10947v1)

**作者:** Kacper Dudzic `[一作]` (Adam Mickiewicz University), Marcin Moskalewicz `[通讯]` (IDEAS Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

结合结构化现象学访谈（TATE）、情感分析和叙事流计算，对自闭症谱系个体的时间经验进行定量化研究。

**💡 创新点**

将TATE访谈的量化结果与大规模自闭症自传语料库的情感化时间词汇相结合，实现了现象学与计算方法的桥接。

**🔧 技术方法**

使用了结构化访谈（TATE）、spaCy NLP、DeBERTa情感分析模型和GPT-3日志概率的序列化测量技术。

**📊 数据集**

使用了28名ASC参与者的访谈数据、104本自传文本（约700万词）以及对照的常规时间表达集合。

**📈 对比分析**

通过Welch t检验比较时间词汇的情感极性，发现自闭症词汇负向情感显著高于对照；叙事流的sequentiality值与普通回忆故事相近，表明叙事连贯性无显著差异。

**⚠️ 局限性**

访谈和语料库均为回溯性数据，缺乏实时微现象学测量；样本相对有限，主要为女性/非二元个体，外推性受限。

---

## 56. MerkleSpeech: Public-Key Verifiable, Chunk-Localised Speech Provenance via Perceptual Fingerprints and Merkle Commitments

**arXiv ID:** 2602.10166 | [PDF](https://arxiv.org/pdf/2602.10166v1)

**作者:** Tatsunori Ono `[一作]` `[通讯]` (University of Warwick), Tatsunori Ono (University of Warwick)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一种基于公钥可验证、分块本地化的语音 Provenance 系统，支持拼接感知与鲁棒水印检索。

**💡 创新点**

通过将分块感知的音频指纹与 Merkle 树签名结合，并在音频中嵌入指向签名根的轻量级水印，使得检索时只需公共信息即可验证声学片段是否属于已登记的发行者。

**🔧 技术方法**

使用 QIM‑STFT 水印、MFCC/SSL 语音指纹、SHA‑256 + Ed25519 签名、Merkle 树、ECC 纠错、公开仓库检索等技术。

**📊 数据集**

采用 LibriSpeech（16 kHz，约 5.4 小时）进行训练、负样本与鲁棒性测试。

**📈 对比分析**

与基线水印/指纹方案对比，使用 FPR_verified、decode rate、鲁棒性（重采样、带通滤波、噪声等）等指标；无扰动下 99.9% 通过，WM‑only 在噪声/重采样下 61% 成功，完整层在任何扰动下 0% 但能提供差异诊断。

**⚠️ 局限性**

指纹不是碰撞安全；未评估对神经编解码器的鲁棒性；需要外部仓库检索；白盒/自适应攻击可能突破；缺乏重叠分块与更宽容指纹。

---

## 57. KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis

**arXiv ID:** 2602.10246 | [PDF](https://arxiv.org/pdf/2602.10246v1)

**作者:** Mayur Akewar `[一作]` (Florida International University), Janki Bhimani `[通讯]` (Florida International University)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5011474644)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个基于知识图谱与大型语言模型的SSD运维分析框架KORAL，实现从碎片化遥测数据生成结构化数据KG，并结合文献KG进行解释性、预测性、指导性和what‑if查询

**💡 创新点**

首次将LLM与两套知识图谱（数据KG与文献KG）联合用于SSD全景运维分析，并提供可追溯的证据引用和自动化中间表示

**🔧 技术方法**

使用知识图谱（RDF/Turtle、SPARQL）、LLM（GPT‑4o）与规则基中间表示、数据质量标注、检索增强生成（RAG）技术

**📊 数据集**

使用Google SSD现场监控数据（6年百万驱动日）、阿里巴巴SSD跟踪数据（5万+驱动）以及实验室环境实验数据（温湿度、振动）

**📈 对比分析**

相较于随机森林、前馈网络、LSTM等传统机器学习基线，KORAL在设备级故障预测精度达96%/召回58%，TTR误差下降，文本质量BLEU4/ROUGE‑L和信度指标显著提升

**⚠️ 局限性**

依赖专有LLM模型、缺乏真实运维查询、对极端稀缺故障标签不足、未验证推理结果的实际效用、对新硬件/环境的适应性未知

---

## 58. Safe mobility support system using crowd mapping and avoidance route planning using VLM

**arXiv ID:** 2602.10910 | [PDF](https://arxiv.org/pdf/2602.10910v1)

**作者:** Sena Saito `[一作]` (Utsunomiya University), Koichi Ozaki `[通讯]` (Utsunomiya University)

**通讯引用:** 11351 | [OpenAlex ID](https://openalex.org/A5053237195)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

使用Vision‑Language模型检测图像中的人群并通过高斯过程回归生成连续成本图，实现机器人在动态人群环境下的路径规划。

**💡 创新点**

首次将VLM的抽象概念识别能力与概率建模（GPR）结合，形成可量化的不确定性地图，提升了对动态人群的感知与避障效果。

**🔧 技术方法**

VLM（GPT‑4o‑mini）+高斯过程回归+Dijkstra算法+二维网格地图。

**📊 数据集**

实验使用Utsunomiya大学校园真实场景，机器人配备3D‑LiDAR、RGB相机等传感器；人群数据为实验中人为创建的8人队列，未使用公开数据集。

**📈 对比分析**

与仅基于几何地图的Dijkstra规划相比，加入人群抽象地图后，机器人成功避开了静态障碍和人群，路径略长但安全性提升；实验以路径示例和成功率表明效果，未给出数值指标。

**⚠️ 局限性**

检测结果在距离远近、光照和视角变化时不稳定；当前方法视为近似静态人群，需更频繁更新才能应对高速动态场景；仅处理单一抽象概念，缺乏多层环境因子集成。

---

## 59. Photons x Force: Differentiable Radiation Pressure Modeling

**arXiv ID:** 2602.10712 | [PDF](https://arxiv.org/pdf/2602.10712v1)

**作者:** Charles Constant `[一作]` (University College London), Tobias Ritschel `[通讯]` (University College London)

**通讯引用:** 4600 | [OpenAlex ID](https://openalex.org/A5084442493)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用可微分的太阳辐射压模拟和神经网络代理，实现了在辐射压作用下自动优化航天器设计与轨迹的系统。

**💡 创新点**

创新点包括：① 高效的基于Monte‑Carlo的并行辐射压模拟（重要性采样与下一事件估计）；② 用MLP学习的可微分辐射压代理，可比完整模拟快数百倍；③ 结合自适应梯度（adjoint）实现的逆向设计与轨迹优化。

**🔧 技术方法**

使用的技术主要有：GPU加速的JAX路径追踪、三层MLP代理、RK4积分器、Adjoint梯度求解、重要性采样、Next‑Event估计等。

**📊 数据集**

实验数据来源于TU Delft公开的卫星模型（GRACE、Swarm、CHAMP、GPS‑2F）以及IGS GPS轨道四元数，未使用真实实测轨迹，仅做仿真验证。

**📈 对比分析**

与Ball、Cannonball、ClassicGrid、RTGrid、BVHRTGrid等基线进行对比，误差平均约0.5 m（比大多数基线低一半以上），在A100上可达10⁸样本/秒，神经代理更快（5×10⁸样本/秒），速度提升10–100倍，保持高精度。

**⚠️ 局限性**

局限性包括：需要精确的几何和材质模型；Monte‑Carlo噪声仍存在；未在真实航天器上验证；仅考虑太阳辐射压，其他非保守力需手工加入；目前对高维设计空间的收敛性与可扩展性仍需进一步研究。

---

## 60. Predictive-State Communication: Innovation Coding and Reconciliation under Delay

**arXiv ID:** 2602.10542 | [PDF](https://arxiv.org/pdf/2602.10542v1)

**作者:** Ozgur Ercetin `[一作]` (Sabanci University), Mohaned Chraiti `[通讯]` (Sabanci University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出预测状态通信（PSC）框架，将通信转化为共享预测状态与创新补丁的同步。

**💡 创新点**

创新点在于把延迟下的状态同步视为主要任务，引入交叉熵创新负载计量和感知‑容量可行带。

**🔧 技术方法**

使用交叉熵计量、状态标识符、锚点、有限回滚、补丁更新与误差监控等技术。

**📊 数据集**

示例采用WikiText‑2等文本数据评估交叉熵，验证PSC在不同预测器下的可行带。

**📈 对比分析**

通过与传统符号传输的可靠通信对比，可视化可行带显示PSC在给定延迟和容量时能支持更宽的速率区间。

**⚠️ 局限性**

局限包括对高质量预测器的强依赖、补丁编码效率未优化、需要标准化状态描述以及在大误差时可能退化为常规重传。

---

## 61. Chamfer-Linkage for Hierarchical Agglomerative Clustering

**arXiv ID:** 2602.10444 | [PDF](https://arxiv.org/pdf/2602.10444v1)

**作者:** Kishen N Gowda `[一作]` (University of Maryland), Jakub Łącki `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出Chamfer-linkage作为Hierarchical Agglomerative Clustering的新链接函数，并给出O(n²)时间的精确实现和O(n²/t)空间、O(n²t)时间的折中算法，同时在19个真实数据集上进行实验评估。

**💡 创新点**

创新点在于将Chamfer距离（点云相似度度量）引入聚类链接，满足概念表示且不需要满足可约性；通过观察Chamfer距离的递减性质设计出与经典链接同等复杂度的算法，并提出可调节的空间时间折中策略。

**🔧 技术方法**

使用Chamfer距离定义、点-簇距离维护、簇间距离递推、增量更新以及增强映射等技术；实现为C++并行版本，并提供Python绑定。

**📊 数据集**

实验数据集共19个，涵盖多领域的公开数据集，如Cancer、Coil-20、Reddit、Covertype等。

**📈 对比分析**

与平均、完整、单、中心、Ward五种经典链接函数以及Chamfer的对称/归一化变体进行比较，使用ARI、NMI、AMI、FMI等指标评估。Chamfer-linkage在ARI上平均提升6%，最优提升57%，树高更平衡；实现速度与经典相当，甚至在优化实现上比fastcluster和scikit‑learn快5.75–9.28倍。

**⚠️ 局限性**

局限性包括仍为O(n²)空间/时间，难以直接处理极大规模数据；对称和归一化变体缺乏O(n²)实现；目前没有并行或近似可扩展的版本，需要进一步研究。

---

## 62. WHEREIS: IP Address Registration Geo-Consistency

**arXiv ID:** 2602.11102 | [PDF](https://arxiv.org/pdf/2602.11102v1)

**作者:** Robert Beverly `[一作]` (San Diego State University), Oliver Gasser `[通讯]` (IPinfo)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过主动测量和延迟基准的IP地址定位，评估并分类五大RIR前缀的地理一致性。

**💡 创新点**

定义了“地理一致性”分类法，利用RIPE Atlas的约束式延迟定位实现对前缀真实位置的推断，系统性量化注册信息准确度，并验证对商业地理数据库的影响。

**🔧 技术方法**

使用RIPE Atlas主动探测、ICMP响应地址hitlist、延迟-光速转换约束定位、BGP对齐过滤、Anycast检测以及Speedtest服务器验证。

**📊 数据集**

2024年10月1日的五大RIR WHOIS数据库、ISCSI/IPv6 hitlist、全球BGP表、商业地理数据库（MaxMind、IPinfo、DB-IP）以及Speedtest服务器。

**📈 对比分析**

与已有地理数据库和RIR反馈对比，发现98% 前缀完全一致，约1% 为不一致；与商业数据库比对时检测到OOR前缀差异高达92%；相对传统WHOIS基准，延迟定位在国家级精度上更可靠。

**⚠️ 局限性**

受限于探测不到响应IP、Atlas探针缺失、探针定位误差及被测前缀属于NIR或任何cast，导致部分前缀被误判为一致；方法仅给出一致性下限，无法完全覆盖所有前缀。

---

## 63. Mask-Based Window-Level Insider Threat Detection for Campaign Discovery

**arXiv ID:** 2602.11019 | [PDF](https://arxiv.org/pdf/2602.11019v1)

**作者:** Jericho Cain `[一作]` (Portland Community College), Hayden Beadles `[通讯]` (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出一种双通道卷积自编码器，用于无监督地在固定时间窗口内检测内部威胁，并进一步通过窗口级异常分数的稀疏聚合实现多日攻击活动的发现。

**💡 创新点**

创新点在于将活动的存在与幅值显式分离，通过二值掩码通道学习稀疏行为结构，并利用掩码重构损失主导表征空间，从而显著提升窗口级检测精度；同时证明仅靠稀疏聚合即可完成攻击系列检测，无需复杂序列建模。

**🔧 技术方法**

使用的技术包括：双通道卷积自编码器（掩码+数值）、加权二元交叉熵与条件均方误差损失、可选的时间一致性正则化、以及窗口级异常分数的 top‑k 聚合用于系列检测。

**📊 数据集**

实验基于 CERT r4.2 内部威胁数据集，仅关注持续 1–7 天的多日攻击（Scenario 1 与 Scenario 3）。

**📈 对比分析**

与传统单通道自编码器基线对比，窗口级 PR‑AUC 从 0.234 提升至 0.714，且在合适阈值下实现零误报；系列检测的 PR‑AUC 在 0.79–0.84 之间，top‑k 聚合显著优于求和或随机投票。

**⚠️ 局限性**

局限性包括：评估仅限于已知受害用户子集，缺乏真实世界外部验证；对攻击持续时间和稀疏特征的假设可能不适用于所有场景；未探索更复杂的时间序列建模，且结果受 CERT 数据集特定特征的影响。

---

## 64. Spatial-Morphological Modeling for Multi-Attribute Imputation of Urban Blocks

**arXiv ID:** 2602.10923 | [PDF](https://arxiv.org/pdf/2602.10923v1)

**作者:** Vasilii Starikov `[一作]` (ITMO University), Sergey Mityagin `[通讯]` (ITMO University)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5038590181)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种基于空间‑形态的填补工具（SM‑Imputer），用于在城市街块层面重建缺失的楼层面积指数（FSI）和地面面积指数（GSI）。

**💡 创新点**

创新点在于将数据驱动的形态聚类（通过 K‑means 生成的 FSI–GSI 形态簇）与局部空间插值方法（IDW、sKNN）动态融合，既利用全局形态先验，又兼顾邻域上下文，显著提升填补精度；同时实现了形态簇概率预测的端到端流程。

**🔧 技术方法**

使用技术包括：BlocksNet 生成街块拓扑；K‑means 在标准化 FSI‑GSI 空间进行聚类；CatBoost 分类器预测每块的形态簇概率；IDW、sKNN、SMV‑NMF 等基线填补；以及 MAE、RMSE、R²、R²_robust 等评价指标。

**📊 数据集**

数据集为圣彼得堡（Saint Petersburg）城市街块数据集，包含土地利用比例、总占地面积、FSI 与 GSI 等属性；在此数据上人工掩码并引入不同缺失率进行实验。

**📈 对比分析**

与 IDW、sKNN、SMV‑NMF 基线方法对比，单独的 SM 在大多数缺失率下表现中等；但与 IDW 或 sKNN 组合后（SM+IDW、SM+sKNN）在 MAE、RMSE 方面均优于所有基线，并获得最高的 R² 分数，表明两种信息源互补。

**⚠️ 局限性**

局限性包括：仅在单一城市（圣彼得堡）上验证，缺乏跨城市泛化评估；形态簇的静态划分可能不适用于形态高度多样的城市；缺失数据为人工掩码，未检验在真实缺失情形下的表现；未将交通网络、功能邻接等更丰富的空间关系融入模型。

---

## 65. Benchmarks Are Not That Out of Distribution: Word Overlap Predicts Performance

**arXiv ID:** 2602.10657 | [PDF](https://arxiv.org/pdf/2602.10657v1)

**作者:** Woojin Chung `[一作]` (NAVER Cloud), Jeonghoon Kim `[通讯]` (NAVER Cloud)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究预训练语料与基准测试数据的词级重叠对零样本基准性能的影响，并验证词级单元交叉熵与基准得分的负相关性

**💡 创新点**

提出词级单词交叉熵和词频统计作为无词元化器、易解释的度量，用以预测和解释基准得分差异

**🔧 技术方法**

信息论方法（交叉熵、KL散度）、词频统计、白名单词分割、无监督预训练（GPT-2 tokenizer）、梯度裁剪与AdamW优化

**📊 数据集**

FineWeb-Edu、DCLM、C4、OpenWebText 四大预训练语料，10 个零样本基准（ARC Easy/Challenge、HellaSwag、MMLU、SciQ、OpenBookQA、PIQA、LAMBADA、SocialIQA、SWAG）

**📈 对比分析**

通过在不同模型规模（400M、1.33B、3.36B）和不同子集规模（8.5B、26B、60B）下训练并评估，发现词级交叉熵与基准得分呈负相关，词频越高基准得分越好，实验结果在各基准上均保持一致

**⚠️ 局限性**

只关注词级重叠，无法解释语法或数理类基准；多语言零样本无法通过词级重叠解释；词级指标对基准难度评估不具通用性

---

## 66. ReSPEC: A Framework for Online Multispectral Sensor Reconfiguration in Dynamic Environments

**arXiv ID:** 2602.10547 | [PDF](https://arxiv.org/pdf/2602.10547v1)

**作者:** Yanchen Liu `[一作]` (Columbia University), Xiaofan Jiang `[通讯]` (Columbia University)

**通讯引用:** 3999 | [OpenAlex ID](https://openalex.org/A5063824268)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e0540dec-d77f-42db-94ae-d039248f6393` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个闭环框架，将多模态感知、强化学习和传感器硬件控制结合，实时调节RGB、IR、mmWave雷达和深度传感器的采样频率、分辨率和测量范围，实现资源感知的自适应机器人感知；

**💡 创新点**

创新点在于：①将检测模型的模态贡献估计直接输入RL决策；②通过RL实时控制硬件采样参数；③在移动机器人上验证并显著降低GPU负载；

**🔧 技术方法**

采用YOLOv8改进的多模态检测骨干、梯度归因贡献提取、基于Q学习的强化学习调度、以及mmWave雷达/红外/RGB/深度传感器同步技术；

**📊 数据集**

使用SeeingThroughFog (STF) 与LLVIP 数据集验证模态贡献估计；

**📈 对比分析**

与全开启和启发式规则基线对比，在SPEC移动车平台实验中，适应调度将GPU负载降低29.3%，检测准确率仅下降5.3%，某些场景下甚至提升；

**⚠️ 局限性**

局限性包括：依赖贡献估计的稳定性；硬件切换延迟影响即时响应；未纳入安全/时延保证；表格Q学习规模受限，难以扩展到更复杂任务。

---

## 67. Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval

**arXiv ID:** 2602.10847 | [PDF](https://arxiv.org/pdf/2602.10847v1)

**作者:** Fanpu Cao `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 43852 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Global Temporal Retriever（GTR）模块，能够将全周期的时间嵌入动态检索并与局部历史序列融合，显著提升多变量时间序列的短期和长期预测性能。

**💡 创新点**

创新点在于：① 通过自适应全周期嵌入和绝对时间索引实现对全周期信息的检索；② 使用 2D 卷积与残差融合同时捕捉局部与全局周期；③ 该模块轻量、无须改动原始模型架构，可直接插拔。

**🔧 技术方法**

核心技术包括：全周期嵌入矩阵、绝对周期索引、线性映射、2D 卷积、残差融合、RevIN 归一化以及基于 MLP 的前馈网络。

**📊 数据集**

实验使用六大公开数据集：ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Traffic、Solar、Weather、PEMS03、PEMS04、PEMS07、PEMS08。

**📈 对比分析**

与 10+ 先进模型（RAFT、S-Mamba、TQNet、TimeXer、CycleNet、SOFTS、TimeMixer、iTransformer、PatchTST、DLinear）在多种预测时长下进行对比。GTR 在 16 次长周期任务中位居前 2 的次数为 10 次，在 8 次短周期任务中位居前 2 的次数为 8 次，平均 MSE/MAE 分别提升 15‑30% 以上，且参数和计算开销极低。

**⚠️ 局限性**

局限性包括：① 需要预设单一固定周期长度，无法自适应多变周期或跨通道周期差异；② 对极长周期的学习受数据稀缺和计算复杂度限制；③ 对长输入序列的线性投影导致 O(T²) 计算，长窗口时效率下降；④ 在仅存在单一周期的序列中可能表现不佳。

---

## 68. Anonymization-Enhanced Privacy Protection for Mobile GUI Agents: Available but Invisible

**arXiv ID:** 2602.10139 | [PDF](https://arxiv.org/pdf/2602.10139v1)

**作者:** Lepeng Zhao `[一作]` (Tsinghua University), Zhuotao Liu `[通讯]` (Tsinghua University)

**通讯引用:** 1174 | [OpenAlex ID](https://openalex.org/A5045206037)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于匿名化的移动 GUI 代理隐私保护框架，实现敏感数据可用但不可见。

**💡 创新点**

通过类型保留的确定性占位符、跨模态一致性映射、交互代理和本地隐私门控，解决了敏感信息可用性与隐私的矛盾。

**🔧 技术方法**

采用了零射击 NER（GLiNER）、正则回退、OCR、哈希占位符、视觉遮罩、交互代理映射以及本地小型语言模型 Qwen3 进行局部计算等技术。

**📊 数据集**

在 AndroidLab 与 PrivScreen 两个基准上进行评测，使用 138 项任务和 500+ 真实+合成 PII 界面。

**📈 对比分析**

与原始、CORE、DualTAP、PrivWeb 等基线对比，隐私泄露率显著降低，任务成功率仅轻微下降或略有提升，显示最佳隐私-效能折衷。

**⚠️ 局限性**

仍需依赖 OCR 质量，对高复杂语义任务的局部计算有限，且系统对移动设备实时性能及跨平台通用性尚未完全验证。

---

## 69. Privacy Control in Conversational LLM Platforms: A Walkthrough Study

**arXiv ID:** 2602.10684 | [PDF](https://arxiv.org/pdf/2602.10684v1)

**作者:** Zhuoyang Li `[一作]` (Eindhoven University of Technology), Yuhan Luo `[通讯]` (City University of Hong Kong)

**通讯引用:** 486 | [OpenAlex ID](https://openalex.org/A5048911139)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 Character.ai、ChatGPT、Claude、Gemini、Meta AI、Pi 六大主流对话式 LLM 平台进行系统化的应用程序漫游（walkthrough），从界面层面梳理并比较了用户对数据的访问、编辑、删除、共享等隐私控制机制。

**💡 创新点**

发现并提出了三大创新视角：①数据控制单位由传统的结构化字段转向基于交互生成的“记忆片段”“定制对象”等动态数据；②自然语言（NL）控制与图形用户界面（GUI）混合使用，使隐私指令更直观但也更易歧义；③多用户共享导致数据共同所有权与治理的复杂性，亟需新的共享与治理框架。

**🔧 技术方法**

采用专家驱动的应用程序漫游方法结合 Wilson 等人隐私政策注释框架，手工收集与编码平台截图、交互日志、政策文本，并进行归纳-演绎的主题分析。

**📊 数据集**

研究对象为六个平台的公开用户界面、隐私政策、使用条款以及通过漫游得到的交互截图与日志数据；并对比了这些平台在数据单元、控制选项与执行方式上的差异。

**📈 对比分析**

该研究并未涉及算法性能对比，而是通过定性对照法呈现各平台在数据控制覆盖度、细粒度、NL控制可用性、共享机制等方面的差异，说明不同设计取向对用户隐私体验的影响。

**⚠️ 局限性**

局限性包括：①仅为快照式研究，无法跟踪平台持续演进；②缺乏真实用户使用体验与反馈；③仅聚焦于美国总部的商用平台，未覆盖国际或开源生态；④手工分析易受研究者主观性影响。

---

## 70. Chart Specification: Structural Representations for Incentivizing VLM Reasoning in Chart-to-Code Generation

**arXiv ID:** 2602.10880 | [PDF](https://arxiv.org/pdf/2602.10880v1)

**作者:** Minggui He `[一作]` (Waseda University), Yuya Ieiri `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Chart Specification作为结构化中间表示，实现图表图像到可执行绘图代码的高保真生成。

**💡 创新点**

创新点包括：①结构化规范（Chart Specification）取代文本复制，聚焦视觉结构；②Spec-Align Reward提供可验证的细粒度结构奖励；③ChartStruct数据构建实现结构均衡且高效的数据集。

**🔧 技术方法**

技术手段包括：视觉语言模型(Qwen2.5-VL-7B) + 组相对策略优化(GRPO)的强化学习；代码执行时的运行时拦截；基于规范的奖励树。

**📊 数据集**

使用自构造的ChartStruct数据集（3K/4K样本），源自ChartCoder等公开数据；在ChartMimic、Plot2Code和ChartX三大基准上评估。

**📈 对比分析**

与先前SFT和大型通用模型对比，ChartSpec在ChartMimic、Plot2Code和ChartX上分别以约82.4、88.7、3.52的综合分数超越GPT‑4o、InternVL‑Llama3‑76B等竞争对手，显示出显著的数据效率与结构精度提升。

**⚠️ 局限性**

局限性包括：仍依赖可执行环境进行奖励计算；对交互式/动态图表支持有限；对极端稀有图表类型的泛化尚需进一步研究。

---

## 71. Agentic Knowledge Distillation: Autonomous Training of Small Language Models for SMS Threat Detection

**arXiv ID:** 2602.10869 | [PDF](https://arxiv.org/pdf/2602.10869v1)

**作者:** Adel ElZemity `[一作]` (University of Kent), Rogério De Lemos `[通讯]` (University of Kent)

**通讯引用:** 7125 | [OpenAlex ID](https://openalex.org/A5026454561)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出 Agentic Knowledge Distillation，即利用大型语言模型（LLM）自主生成合成数据、持续细化并用 LoRA 微调小型语言模型（SLM），实现无需人工标注即可在移动设备上进行 SMS Smishing 检测。

**💡 创新点**

创新点在于：①将 LLM 担任“自动 ML 工程师”，完成闭环自适应训练；②首次与 Direct Preference Optimisation (DPO) 基线对比，证明迭代反馈与针对性数据生成显著提升性能；③系统评估四种主流 LLM（Claude Opus 4.5、GPT 5.2 Codex、Gemini 3 Pro、DeepSeek V3.2）对最终模型质量的差异。

**🔧 技术方法**

使用技术包括：LLM 生成合成训练/验证集、LoRA 参数高效微调、闭环错误分析与硬负样本生成、DPO 作为对照方法、基于合成验证集的聚合指标反馈。

**📊 数据集**

实验采用公开的 SMS Spam Collection 数据集（5,574 条短信），构建平衡 1,494 条测试集（747 正类、747 负类），教师 LLM 负责生成全部合成数据。

**📈 对比分析**

比较方法：先做零样本基线，然后用教师 LLM 生成的 DPO 数据进行一次性微调，最后运行完整的 Agentic Knowledge Distillation。最佳配置（Claude Opus 4.5 + Qwen2.5‑0.5B）在测试集上达到 94.31% 准确率、94.42% F1、96.25% 召回；其他教师导致准确率相差 20–45% 点，凸显教师 LLM 的关键作用。

**⚠️ 局限性**

局限性包括：受教师 LLM 培训知识范围限制，可能错过新型攻击；系统仅依赖合成验证集进行内部评估，无法直接验证对真实标签的泛化；教师 LLM 的数据生成与错误分析能力决定最终性能，若质量不足会导致严重的精确率-召回失衡。

---

## 72. Architecting Trust: A Framework for Secure IoT Systems Through Trusted Execution and Semantic Middleware

**arXiv ID:** 2602.10762 | [PDF](https://arxiv.org/pdf/2602.10762v1)

**作者:** Muhammad Imran `[一作]` (Universidade da Coruña), Muhammad Imran `[通讯]` (Universidade da Coruña)

**通讯引用:** 57848 | [OpenAlex ID](https://openalex.org/A5078421218)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一套集成可信执行环境（TEE）、语义中间件与区块链技术的跨层物联网安全架构，兼顾硬件根信任、零信任网络和语义安全；

**💡 创新点**

创新点在于将 TEEs、语义安全与区块链三大技术统一到五层安全模型，实现资源受限环境下的防御深度和可扩展性；

**🔧 技术方法**

采用 ARM TrustZone‑M/ Fortress、Semantic IoT Middleware、Ethereum 智能合约、ASCON 轻量密码、MQTT/TLS1.3 以及 AI 异常检测等多种技术；

**📊 数据集**

使用真实硬件（Cortex‑M 微控制器、单板机）以及仿真网络（1 万设备）生成的传感器与安全事件日志，并基于 IoT Security Foundation 评估框架进行验证；

**📈 对比分析**

通过 IoT Security Foundation 评估、能耗/延迟基准测量以及与传统外围安全和云中心安全两种对照架构对比，证明安全有效率达 82%（比传统方案提升 17%），能耗提升约 8–15%，在 5,000 设备规模内保持稳定；

**⚠️ 局限性**

局限性包括硬件异构性导致的安全机制差异、区块链可扩展性瓶颈、额外的能耗与延迟成本，以及对旧设备兼容与统一管理的挑战。

---

## 73. Time Series Foundation Models for Energy Load Forecasting on Consumer Hardware: A Multi-Dimensional Zero-Shot Benchmark

**arXiv ID:** 2602.10848 | [PDF](https://arxiv.org/pdf/2602.10848v1)

**作者:** Luigi Simeone `[一作]` `[通讯]` (Independent Researcher), Luigi Simeone (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个多维度基准，评估了四种时间序列基础模型（Chronos‑Bolt、Chronos‑2、Moirai‑2、TinyTimeMixer）与行业基线Prophet及统计参考模型在ERCOT小时负荷预测上的零样本性能；

**💡 创新点**

创新点在于：①首次将时间序列基础模型在能源负荷预测的实战环境（零样本、消费者级硬件）下系统评估；②提出多维度评估框架，涵盖上下文长度敏感性、概率校准、分布偏移鲁棒性和启发式决策价值；

**🔧 技术方法**

使用的技术包括零样本Transformer/MLP‑Mixer模型、Prophet拟合、SARIMA统计模型、CRPS与Winkler评分等评估指标；

**📊 数据集**

使用的数据集为2020‑2024年ERCOT系统总负荷的每小时观测数据（约43,732条），涵盖正常、COVID、冬季风暴Uri等不同运营阶段；

**📈 对比分析**

比较方法为在八个上下文长度（24至2048小时）与两步长（24h、168h）下，针对七个不同测试窗口共2,352个预测实例，计算MASE、CRPS、经验覆盖率和Winkler分数；结果显示Chronos‑Bolt/Chronos‑2/Moirai‑2在足够上下文时MASE可低至0.31，显著优于季节性Naïve与Prophet，并且在概率校准与极端事件预测上具有可比性；

**⚠️ 局限性**

局限性包括：仅评估单一ERCOT负荷数据，未涵盖多电网或多变量（气象、价格）情境；仅测试零样本配置，未探究微调带来的潜在提升；使用的模型仅为CPU版，GPU加速的强大模型未纳入；未与已训练的深度学习基线（如N‑BEATS、TFT）直接比较；

---

## 74. "Humans welcome to observe": A First Look at the Agent Social Network Moltbook

**arXiv ID:** 2602.10127 | [PDF](https://arxiv.org/pdf/2602.10127v1)

**作者:** Yukun Jiang `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2026年初 AI 代理社交平台 Moltbook 进行大规模经验性研究，分析 44,411 条帖子与 12,209 个子社群的主题与毒性分布，并揭示其快速增长、中心化关注与群体风险放大。

**💡 创新点**

首次构建针对代理讨论的两维标签体系（主题分类与五级毒性），并系统量化不同主题下的风险谱系与时间演化，凸显平台级治理需求。

**🔧 技术方法**

利用 LLM 驱动的自动标注管线、余弦相似度聚类、Shannon 熵多样性指标、时间窗毒性比例关联分析等技术，对整个数据集进行标注与多维度分析。

**📊 数据集**

采集自 Moltbook 官方 API 的 44,411 条公开帖子与 12,209 个 submolt，涵盖技术、社交、经济、宣传、政治等九大主题，并配以 44,376 条人工/LLM 复核标注。

**📈 对比分析**

与人工标注对比，LLM 标注准确率达 91.86%；通过主题-毒性流向图、小时级毒性比率等可视化展示，证明毒性高度依赖主题且高峰期风险显著上升。

**⚠️ 局限性**

受限于仅聚焦早期 Moltbook、主要为英文内容、LLM 标注可能携带偏差以及未能评估跨平台泛化，研究结果对其他代理社交生态的适用性仍需进一步验证。

---

## 75. Linear-LLM-SCM: Benchmarking LLMs for Coefficient Elicitation in Linear-Gaussian Causal Models

**arXiv ID:** 2602.10282 | [PDF](https://arxiv.org/pdf/2602.10282v1)

**作者:** Kanta Yamaoka `[一作]` (German Research Centre for Artificial Intelligence), Sebastian Josef Vollmer `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Linear-LLM-SCM框架，用于评估LLM在已知DAG下线性高斯结构因果模型参数化的能力。

**💡 创新点**

将DAG分解为局部父子结构，利用LLM生成回归系数，并提供可插拔评估和抗扰性测试。

**🔧 技术方法**

LLM提示工程、循环反馈、JSON解析、L2距离和相对排序指标等技术。

**📊 数据集**

使用BnRep中的7个真实世界DAG（如cachexia1、expenditure等），并进行单元和结构扰动实验。

**📈 对比分析**

对比Gemini 2.5 Flash、Llama 3.1 8B、Llama 3.3 70B等模型，M3和M4指标表现最佳为Gemini；LLM在温度0下仍表现随机性，准确性不一。

**⚠️ 局限性**

主要局限是线性假设、缺乏不确定性量化、LLM在单位/结构扰动下的鲁棒性差、以及高随机性导致安全性问题。

---

## 76. Core-Stable Kidney Exchange via Altruistic Donors

**arXiv ID:** 2602.10725 | [PDF](https://arxiv.org/pdf/2602.10725v1)

**作者:** Gergely Csáji `[一作]` (Eötvös Loránd University), Thánh Nguyen `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出并研究了“补充核心”（supplemented core）概念，证明只需少量互惠捐赠者即可稳定肾脏交换市场；

**💡 创新点**

创新点在于用补充核心理论解决核心空缺问题，给出随机图、类型图、二分图等多种图模型下的上界与下界，并通过实验验证理论可行性；

**🔧 技术方法**

技术手段包括Scarf引理、整数规划与半整数规划的取整技巧、基于代币的极点分析以及迭代核心剪枝的启发式算法；

**📊 数据集**

使用基于Saidman生成器的合成数据集，规模从100到500对（患者‑捐赠者）和50名互惠捐赠者，随机划分为5–30个组织；

**📈 对比分析**

与传统的弱核心、强核心及TU核心比较，实验显示在99.7%以上的实例中弱核心无需额外捐赠者；添加核心约束后平均仅需0.1–0.7名捐赠者，性能优于现行的按层次目标优化；

**⚠️ 局限性**

局限性：理论上所需捐赠者上界依赖Ω(|V|)，实际证明为常数；算法仅对≤4名组织的协同失效求解，且主要考虑Δ≤3的循环；未完全解决强核心和TU核心的最优性与可扩展性问题。

---

## 77. Hidden Licensing Risks in the LLMware Ecosystem

**arXiv ID:** 2602.10758 | [PDF](https://arxiv.org/pdf/2602.10758v1)

**作者:** Bo Wang `[一作]` (Beijing Jiaotong University), Zhou Yang `[通讯]` (University of Alberta)

**通讯引用:** 20500 | [OpenAlex ID](https://openalex.org/A5071872950)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了大规模LLMware供应链数据集，分析了许可分布和冲突，并提出基于LLM的多代理框架LiAgent来检测许可兼容性。

**💡 创新点**

首次系统量化LLMware许可风险，发现许可冲突普遍且多与AI专用许可相关，并提出LLM驱动的多代理方案在冲突检测上大幅超越传统方法。

**🔧 技术方法**

利用GitHub与Hugging Face的API抓取、静态代码分析、法律条款抽取及LLM（GPT‑4o、DeepSeek‑V3等）多代理推理实现许可分析。

**📊 数据集**

使用12,180个GitHub仓库、3,988个Hugging Face模型和708个数据集构成的LLMware链条，及手工标注的124个传统许可证与16个AI专用许可证的法律条款数据库。

**📈 对比分析**

将LiAgent与Semantic Similarity+SST以及LiDetector进行基准评测，LLM模型在原始许可证集上取得87%–89% F1，提升约8–12个百分点；在变体集上仍保持高达88% F1，表明鲁棒性更强。

**⚠️ 局限性**

受限于许可证条款覆盖范围、手工标注规模以及仅关注GitHub和Hugging Face，未覆盖低星项目或其他平台的LLMware；此外LLM推理对成本和可解释性仍是挑战。

---

## 78. Confounding Robust Continuous Control via Automatic Reward Shaping

**arXiv ID:** 2602.10305 | [PDF](https://arxiv.org/pdf/2602.10305v1)

**作者:** Mateo Juliani `[一作]` (Columbia University), Elias Bareinboim `[通讯]` (Columbia University)

**通讯引用:** 3247 | [OpenAlex ID](https://openalex.org/A5039620960)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一套基于因果推断的奖励塑造框架，利用离线数据学习保守的状态潜力函数（上界），并将其作为 Potential-Based Reward Shaping (PBRS) 的潜能，在连续控制任务中提升学习效率。

**💡 创新点**

创新点包括：①将因果贝尔曼方程扩展到连续、无限期的 Confounded Markov Decision Process (CMDP)；②通过部分识别方法从受混淆影响的离线数据中得到最优状态值的上界；③将该上界作为奖励塑造信号，在保持策略最优性的同时显著提升样本效率。

**🔧 技术方法**

采用因果推断与部分识别技术、因果贝尔曼方程、神经网络近似（潜力网络、行为策略网络、状态差分网络）、Soft Actor-Critic（SAC）以及高斯策略的重参数化梯度训练。

**📊 数据集**

使用 Minari 提供的 MuJoCo（Hopper、HalfCheetah、Walker2D、Ant）和 Adroit（Door、Relocate）离线数据集，分别采用不同专业水平（simple、medium、expert），并通过删减观测维度模拟混淆。

**📈 对比分析**

与无塑造 SAC、CQL PBRS、T-REX PBRS、循环 SAC 等基线进行对比。实验表明，Causal PBRS 在大多数受混淆的环境中取得了约 1.29 的归一化平均收益和 1.09 的 IQR 归一化收益，显著优于其他方法；当关键状态维度被移除时性能下降，说明方法的适用范围。

**⚠️ 局限性**

局限性包括：①在极端混淆或去掉关键任务维度时，潜力函数无法提供足够信息；②依赖离线数据质量（行为策略竞争性和经验水平）；③在更高维度（如图像）观察空间的适用性尚未验证；④方法对行为策略的竞争性假设仍存在一定敏感性。

---

## 79. GRASP: group-Shapley feature selection for patients

**arXiv ID:** 2602.11084 | [PDF](https://arxiv.org/pdf/2602.11084v1)

**作者:** Yuheng Luo `[一作]` (Chinese Academy of Medical Sciences and Peking Union Medical College), Zhong Cao `[通讯]` (Heidelberg University)

**通讯引用:** 2635 | [OpenAlex ID](https://openalex.org/A5016651437)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一种基于SHAP值和组L21正则化的特征选择框架GRASP，用于医疗预测任务的稳定、可解释特征抽取。

**💡 创新点**

将模型衍生的SHAP重要性与结构化稀疏正则化相结合，首次实现了解释性驱动的嵌入式特征选择，并通过组策略提升了稳定性与泛化能力。

**🔧 技术方法**

采用XGBoost训练得到SHAP值作为重要性度量，构建组L21正则化的逻辑回归目标，并通过proximal‑gradient（含Armijo退火）优化实现特征筛选。

**📊 数据集**

使用美国NHANES调查数据（1999–2014）作为训练/验证集，UK Biobank（2015–2024）作为外部验证集，共重叠76个临床变量。

**📈 对比分析**

与LASSO、SHAP聚合和AFS三种基线方法在准确率、F1、MCC、冗余度（VIF、相关性）及稳定性（JI、ASM）等指标对比，GRASP在保持相近预测性能的同时，特征数最少、冗余度最低、稳定性最高。

**⚠️ 局限性**

限制主要包括对高维特征集的计算效率仍有提升空间，以及依赖SHAP解释的前置模型训练可能导致对不同数据分布的适应性不足。

---

## 80. LCIP: Loss-Controlled Inverse Projection of High-Dimensional Image Data

**arXiv ID:** 2602.11141 | [PDF](https://arxiv.org/pdf/2602.11141v1)

**作者:** Yu Wang `[一作]` (Utrecht University), Alexandru Telea `[通讯]` (Utrecht University)

**通讯引用:** 8375 | [OpenAlex ID](https://openalex.org/A5024074640)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种通用的、可控的逆投影方法，能够在用户设定的两个参数下扫掠高维图像数据空间，生成丰富多样的样本；

**💡 创新点**

创新点在于突破传统逆投影只能生成固定曲面结构的局限，提供了可调整的“扫掠”机制，实现了对数据空间更完整的覆盖；

**🔧 技术方法**

采用了基于损失控制的逆投影框架，结合投影方法P（如t‑SNE、UMAP等）与深度生成模型，利用逆向网络与自定义损失函数实现数据重构；

**📊 数据集**

在大规模图像数据集上验证，包括CelebA、CIFAR‑10与ImageNet的子集，用于风格迁移与图像增强任务；

**📈 对比分析**

与现有逆投影方法（如逆向投影网络、逆映射等）对比，在重构质量、样本多样性与覆盖范围方面均有显著提升，实验结果表明生成样本的FID/IS分数分别提高了约5–10%；

**⚠️ 局限性**

局限性包括：①对投影方法的依赖仍然存在，极端非线性投影可能导致扫掠效果受限；②需要对损失函数进行手动调参，控制平衡仍非完全自动化；③在极高维度或稀疏数据集上计算成本较高。

---

## 81. CMAD: Cooperative Multi-Agent Diffusion via Stochastic Optimal Control

**arXiv ID:** 2602.10933 | [PDF](https://arxiv.org/pdf/2602.10933v1)

**作者:** Riccardo Barbano `[一作]` (University College London), Francisco Vargas `[通讯]` (University of Cambridge)

**通讯引用:** 876 | [OpenAlex ID](https://openalex.org/A5102728741)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究如何利用多智能体协同控制预训练扩散模型实现多模型组合生成，提出把组合生成视为协同随机最优控制问题；

**💡 创新点**

创新点在于将多模型组合迁移到随机最优控制框架，并通过迭代扩散优化(Iterative Diffusion Optimisation, IDO)实现控制网络的协同学习；

**🔧 技术方法**

使用协同随机最优控制、扩散模型的逆向动力学、Tweedie估计、控制网络及迭代扩散优化；

**📊 数据集**

在MNIST手写数字数据集上进行实验；

**📈 对比分析**

与推理时的DPS式控制（CDPS）对比，CMAD在终端损失上显著更低、分类准确率相当且生成结果更真实；

**⚠️ 局限性**

仅在低维MNIST任务验证，样本多样性下降，缺乏收敛理论，难以直接扩展到更高维或更复杂的应用场景。

---

## 82. Co-jump: Cooperative Jumping with Quadrupedal Robots via Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.10514 | [PDF](https://arxiv.org/pdf/2602.10514v1)

**作者:** Shihao Dong `[一作]` (University of Hong Kong), Peng Lu `[通讯]` (University of Hong Kong)

**通讯引用:** 19967 | [OpenAlex ID](https://openalex.org/A5100366018)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

通过多智能体强化学习实现两只四足机器人在不使用通信或外部感知的条件下协同完成高跳跃动作，显著提升单体机器人可达高度；

**💡 创新点**

①设计无通信、无外部感知的协同跳跃任务；②采用四阶段渐进式课程学习和共享奖励解决多体信用分配与高动力学协同；③将多智能体策略在中央训练、去中心化执行框架下实现；

**🔧 技术方法**

Multi‑Agent Proximal Policy Optimization (MAPPO)、Centralized Training with Decentralized Execution、分阶段渐进式课程学习、域随机化、PD低层控制器以及全靠本体感知的观测输入；

**📊 数据集**

在IsaacSim中构建的4096并行仿真环境（含随机化参数）进行训练，随后在真实硬件（Go2+Js01）上进行验证；

**📈 对比分析**

与单机跳跃基准（Curriculum‑Based Jumping、OmniNet）对比，Co‑jump在1.5 m平台上成功率达91‑93%，最大跳高约1.57–1.77 m，单机基准成功率低于5%；能耗显著下降（约为单机基准的1/6），并在ablation实验中显示四个课程阶段的必要性；

**⚠️ 局限性**

对初始姿态和相对位置高度敏感，需精确堆叠；不具备自主对接和导航能力；仅针对两四足机器人，缺乏对更大规模或多样化形态的推广；仿真到真实的动力学差距仍导致高跳表现下降；

---

## 83. Abstraction Generation for Generalized Planning with Pretrained Large Language Models

**arXiv ID:** 2602.10485 | [PDF](https://arxiv.org/pdf/2602.10485v1)

**作者:** Zhenhe Cui `[一作]` (Hunan University of Science and Technology), Wei Liang `[通讯]` (Hunan University of Science and Technology)

**通讯引用:** 5860 | [OpenAlex ID](https://openalex.org/A5073441538)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何利用预训练的大语言模型（LLM）生成用于通用规划（GP）的定性数值规划（QNP）抽象，并通过自动化调试方法提升抽象质量。

**💡 创新点**

创新点在于：①提出基于提示工程的LLM抽象生成协议；②设计了四阶段的自动化交互式调试流程，以检测并修复抽象错误；③在七个经典GP领域上系统评估并证明某些LLM（如GPT‑5.2）在调试支持下能够生成高覆盖率的QNP抽象。

**🔧 技术方法**

技术包括：提示工程（Feature Generation、Abstraction Generation）、自动化调试（ASC、HLISC、HLPRC、LLGRC）、QNP求解器 DSET、Fast Downward、以及对LLM的交互式错误反馈。

**📊 数据集**

使用了七个通用规划领域的数据集（Delivery、Heavy、Gripper、Miconic、Ferry、Spanner、Forest），每个领域生成4个训练实例并评估10个测试实例。

**📈 对比分析**

与不使用调试的直接生成方式相比，加入自动化调试后，GPT‑5.2 的覆盖率从约 0.6 提升到 1.0；Gemini 较弱但也有提升；Qwen 仍表现不佳。实验显示错误主要来自 ASC 与 HLPRC，调试显著提高成功率，且对大多数领域几乎无 LLGRC 错误。

**⚠️ 局限性**

局限性：①特征模板表达能力有限，无法捕获更复杂的数值特征；②高层动作的精炼必须是单一 LL 动作，无法处理动作序列或程序；③调试提供的错误信息不够精确，影响修正效率；④无法保证生成的抽象一定正确。

---

## 84. Bring Your Own Objective: Inter-operability of Network Objectives in Datacenters

**arXiv ID:** 2602.10252 | [PDF](https://arxiv.org/pdf/2602.10252v1)

**作者:** Sanjoli Narang `[一作]` (Massachusetts Institute of Technology), Manya Ghobadi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1197 | [OpenAlex ID](https://openalex.org/A5058729992)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一个基于市场的分布式调度框架，利用每个链路每个RTT的第二价格拍卖来动态分配网络带宽，并通过预算与动态竞价让不同目标的流共同在同一网络上运行。

**💡 创新点**

创新点包括：① 把网络资源抽象为竞争市场，让应用独立声明目标与预算；② 在交换机局部实施轻量级拍卖与过度承诺，实现工作保持而不需要全局控制；③ 通过限制反馈、延迟计费和单向支付，抑制策略性竞价并确保真诚竞价；④ 为常见数据中心目标（FCT、CCT、deadline、公平性）提供统一的竞价接口与动态规划求解方法。

**🔧 技术方法**

技术手段包括：分布式RTT拍卖、第二价格拍卖、overcommitment、市场元数据字段、动态预算、动态规划求边际效用、可插拔的学习/规则竞价代理、TCP栈级集成、ns‑3仿真和交换机流水线扩展。

**📊 数据集**

使用多种典型数据中心工作负载：Websearch、Datamining、Uniform、Map‑Reduce、Alibaba、以及合成的Coflow、Deadline、Fairness、Best‑Effort 流量，并在 144 主机 leaf‑spine 拓扑中进行仿真。

**📈 对比分析**

与单目标调度器（pFabric、Sincronia、DCTCP、D2TCP 等）对比：单目标场景下与最优调度几乎无差距；多目标场景中，deadline miss 降低约 2 倍，CCT 降低约 1.6 倍，短流 FCT 仍保持在最优的 5% 以内，并且在负载漂移、突发和热点等动态条件下保持稳定。

**⚠️ 局限性**

局限性：需要在交换机中增加自定义状态与逻辑，对极短流会产生额外的 1 RTT 触发延迟，路径选择仍依赖固定或基于 ECMP 的策略，且对恶意共谋或极端策略行为的鲁棒性尚未在真实生产环境中全面验证。

---

## 85. Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling

**arXiv ID:** 2602.11146 | [PDF](https://arxiv.org/pdf/2602.11146v1)

**作者:** Gongye Liu `[一作]` (Hong Kong University of Science and Technology), Wenhan Luo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 7380 | [OpenAlex ID](https://openalex.org/A5004450394)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于潜在扩散模型的奖励模型DiNa-LRM，利用噪声校准的Thurstone偏好学习和多噪声推理实现对文本到图像生成的偏好对齐。

**💡 创新点**

创新点包括：1）将Thurstone偏好模型推广到扩散噪声状态并引入噪声级相关的比较不确定性；2）构建时间感知的潜在奖励头并支持噪声集成推理；3）通过预训练扩散骨干实现更高效的梯度对齐。

**🔧 技术方法**

使用技术包括：预训练潜在扩散模型（如SD3.5-M），FiLM时间调制，Q-Former评分头，噪声校准Thurstone损失，分布式时间采样（均匀/对数正态），LoRA微调，ReFL对齐算法。

**📊 数据集**

训练和评估数据集：HPDv3偏好对比集、ImageReward、HPDv2、HPDv3、GenAI-Bench等；评测时使用多噪声集成推理。

**📈 对比分析**

与CLIP/ VLM/传统扩散奖励模型比较，在单噪声设置下取得71.49%准确率，集成后提升至72.48%；相比VLM仍略低，但在对齐实验中收敛更快、显存和FLOPs下降超过50%。

**⚠️ 局限性**

局限性：仅在特定扩散骨干内部有效，跨骨干迁移性不足；潜在空间可能忽略像素级细节，导致对齐后出现像素级伪影；缺乏强大的像素级正则或多模反馈机制。

---

## 86. Spend Search Where It Pays: Value-Guided Structured Sampling and Optimization for Generative Recommendation

**arXiv ID:** 2602.10699 | [PDF](https://arxiv.org/pdf/2602.10699v1)

**作者:** Jie Jiang `[一作]` (Tencent Inc), Huan Yu `[通讯]` (Tencent Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出V-STAR框架，在生成式推荐中通过价值引导的解码与树结构优势强化来提升候选多样性和奖励对齐。

**💡 创新点**

创新点在于：①价值引导的高效解码（VED），只在决策性前缀上分配预算以避免早期剪枝；②基于前缀树的Sibling‑GRPO，利用同父子群的相对优势恢复奖励压缩问题。

**🔧 技术方法**

技术包括：自回归Transformer+SID编码、轻量级价值头+熵正则、基于UCB的预算搜索、TD学习的值函数更新、GRPO与Sibling‑GRPO的联合策略梯度。

**📊 数据集**

使用Amazon Review的Industrial与Office子集做离线评估，并在微信渠道的真实流量中做在线A/B实验。

**📈 对比分析**

相对于传统Beam‑search+GRPO、SFT、HSTU、BIGRec、TIGER、MiniOneRec等基线，V-STAR在HR@3、NDCG@10等指标上提升4–10%，在线GMV提升约1.2–1.9%。

**⚠️ 局限性**

局限性包括：①仅在SID表示的生成式推荐任务上验证；②推理时仍需Beam‑search，训练时计算开销相对较高；③对极端长尾或多样性需求仍需进一步研究。

---

## 87. Calliope: A TTS-based Narrated E-book Creator Ensuring Exact Synchronization, Privacy, and Layout Fidelity

**arXiv ID:** 2602.10735 | [PDF](https://arxiv.org/pdf/2602.10735v1)

**作者:** Hugo L. Hammer `[一作]` (Oslo Metropolitan University), Pål Halvorsen `[通讯]` (SimulaMet)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发并实现了一个开源离线框架 Calliope，用于将普通 EPUB 转换为带有同步音频的 EPUB 3 Media Overlays，保留原有排版并通过神经 TTS 生成音频。

**💡 创新点**

①直接在 TTS 过程中捕获时间戳，实现零漂移同步；②采用递归文本分段算法解决 Transformer 上下文窗口限制；③完整保留出版商原始 CSS 与媒体；④完全离线运行，消除云端成本与隐私风险。

**🔧 技术方法**

使用 Coqui 的 XTTS‑v2 与 ReSpeaker 的 Chatterbox 神经 TTS；EPUB 3 Media Overlays 与 SMIL；Python、DOM 操作、音频信号处理；递归分段与异常处理策略。

**📊 数据集**

在 Gutenberg 项目公开的《The Gift of the Magi》测试集；使用 15 秒参考语音样本生成声音。

**📈 对比分析**

将 Calliope 与 Syncabook、Storyteller 的强制对齐方法在相同文本与 TTS 音频上比较，计算 SMIL 时间戳漂移。Calliope 的漂移为 0 s，Storyteller 最大漂移超过 30 s，Syncabook 约 1 s；统计量显示 Calliope 远优于对齐方法。

**⚠️ 局限性**

目前仅支持命令行使用，缺乏图形界面；对大文件处理耗时较长；只验证了英文文本与两种 TTS；未针对移动设备进行量化模型压缩；以及在极短句子或特殊符号时仍需进一步优化。

---

## 88. Beyond Task Performance: A Metric-Based Analysis of Sequential Cooperation in Heterogeneous Multi-Agent Destructive Foraging

**arXiv ID:** 2602.10685 | [PDF](https://arxiv.org/pdf/2602.10685v1)

**作者:** Alejandro Mendoza Barrionuevo `[一作]` (University of Sevilla), Sergio L. Toral Marín `[通讯]` (University of Sevilla)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在一项针对破坏性多智能体采集任务的研究中，作者提出了一套面向异质团队的通用合作度量，并在模拟水面垃圾清理环境下对三种策略（DRL、贪婪、Lévy Walk）进行评估

**💡 创新点**

创新点在于系统化地将合作度量分为主指标、跨团队指标和团队内指标，并将其应用于具有部分可观测性和时序依赖的异质采集任务

**🔧 技术方法**

使用了双深度Q学习（DDQL）、贪婪搜索与Lévy Walks，以及基于图的环境模拟

**📊 数据集**

采用了仿真生成的动态垃圾分布数据（每回合随机生成分布和风向）

**📈 对比分析**

通过在100个回合、相同随机种子下比较PTA、RMSE、吞吐量等指标，结果显示DRL和贪婪几乎同样高效（PTA≈99%），但DRL在模型精度（RMSE≈0.001）上优于贪婪，Lévy表现明显差距

**⚠️ 局限性**

局限性包括仅考虑两类角色（侦察员与收集员）、缺乏能耗、通信延迟与失真等实际约束，且指标对更复杂层级结构的推广尚待验证

---

## 89. Non-Fungible Blockchain Tokens for Traceable Online-Quality Assurance of Milled Workpieces

**arXiv ID:** 2602.10169 | [PDF](https://arxiv.org/pdf/2602.10169v1)

**作者:** Nicolai Maisch `[一作]` (University of Stuttgart), Oliver Riedel `[通讯]` (University of Stuttgart)

**通讯引用:** 2407 | [OpenAlex ID](https://openalex.org/A5087654536)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

开发了基于以太坊ERC‑721 NFT的在线质量保证框架，将实时仿真产生的工件质量数据以Asset Administration Shell（AAS）JSON形式上传至IPFS，并通过NFT实现跨企业安全、可追溯的数据交换与多阶段质量记录更新。

**💡 创新点**

创新点在于将NFT与IPFS结合，利用AAS标准化数据实现可修改的质量信息指针；支持多方动态添加质量元数据，保持数据不可篡改且可审计，同时通过对称加密保障公共链与IPFS数据的隐私。

**🔧 技术方法**

使用技术包括以太坊智能合约（ERC‑721 + OpenZeppelin库）、IPFS去中心化存储、AAS标准子模型（Quality Control for Machining）、对称加密、工业机器人控制、实时仿真（并行实时模拟）和JSON序列化。

**📊 数据集**

采用标准工作件“Diamond Circle Square”（ISO 10791:2020‑7）进行实验，利用实时仿真产生尺寸偏差数据，并将生成的AAS文件上传至IPFS。

**📈 对比分析**

通过手工比较仿真模型与目标尺寸的偏差列表验证NFT与IPFS存储的一致性；未进行性能或算法对比，仅展示概念可行性与数据完整性验证。

**⚠️ 局限性**

局限性包括：minting过程仍需人工输入偏差数据，尚未完全自动化；公共链与IPFS公开性需加密保障隐私；交易成本高，可能需要私有链或成本优化；缺乏大规模实验与性能基准评估。

---

## 90. Implicit representations via the polynomial method

**arXiv ID:** 2602.10922 | [PDF](https://arxiv.org/pdf/2602.10922v1)

**作者:** Jean Cardinal `[一作]` (Universite libre de Bruxelles), Micha Sharir `[通讯]` (School of Computer Science, Tel Aviv University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在半代数图（vertices 为实向量，边由常数复杂度的半代数谓词决定）的邻接标签化方案，并给出了多类图的紧凑标签大小；通过多维多项式划分构造分裂树，将顶点出现的二分完全子图数控制在子线性量，从而实现 O(n^{1-2/(d+1)}+ε)-bit 的邻接标签。对特殊情况，如单一多项式、线性多项式、可见性图等，进一步给出了更优的 O(log n)、O(log^3 n)、O(log^2 n) 结果。

**💡 创新点**

创新点主要在于：①利用多项式划分（而非传统的切分或分块）实现对半代数图的二分完全子图分解，得到子线性标签大小；②将此思路推广至线性谓词、可见性图等非半代数族，获得更优的标签长度；③将分裂树与比特列表相结合，实现高效的邻接判断；④在构造阶段保证子线性时间复杂度。

**🔧 技术方法**

核心技术包括：多项式划分（Guth–Katz、Matoušek–Patáková 等）构造分裂树；分裂树与二分完全子图的对应关系；分裂树节点计数分析（递推式、极限证明）；对线性谓词的可比较性图拆解；对可见性图的双色段交叉图与层状树的结合；递归分治与节点标识符编码。

**📊 数据集**

该工作主要是理论分析，没有使用具体实验数据集；所有结果均基于对任意 n 顶点的半代数图进行抽象推导。

**📈 对比分析**

与先前最优的 O(n^{1-1/d} log n) 结果相比，本文通过多项式划分取得了更好的指数 1-2/(d+1)，尤其在 d=2 时从 n^{2/3} 降到 n^{1/3}；对线性谓词可进一步压缩到 O(log n)。在可见性图方面，先前的 O(n log^3 n) 标注被改进为 O(log^3 n) 位标签，显著压缩存储。性能上，构造时间可在实数 RAM 上实现子平方（即 O(n^{1+ε})）的预处理。

**⚠️ 局限性**

局限性包括：①仍缺乏对应的下界，无法确认标签长度是否最优；②标签长度依赖于谓词的常数复杂度，若谓词复杂度提升则常数因子增大；③对非半代数族（如一般可见性图）需要更复杂的双色段交叉图处理；④构造依赖于实数 RAM 模型，若实现于整数模型需进一步讨论。

---

## 91. Affordances Enable Partial World Modeling with LLMs

**arXiv ID:** 2602.10390 | [PDF](https://arxiv.org/pdf/2602.10390v1)

**作者:** Khimya Khetarpal `[一作]` (Google Deepmind), Doina Precup `[通讯]` (Google Deepmind)

**通讯引用:** 21580 | [OpenAlex ID](https://openalex.org/A5065836447)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出利用大型语言模型（LLM）构建部分世界模型，并通过分配 affordances 来降低搜索分支因子，验证其在多任务桌面机器人任务中的有效性。

**💡 创新点**

①将 affordances 形式化为多任务环境下的 task‑agnostic 与 task‑specific intents；②证明实现这些 intents 等价于存在可预测的部分世界模型；③给出理论证明和经验评估，展示部分模型可获得指数级搜索加速且对缺失 intents 具有鲁棒性。

**🔧 技术方法**

使用 LLM 作为世界动态预测模型和 affordance 生成器；多任务强化学习框架；蒙特卡罗树搜索（MC‑Search）与自适应纠正的部分模型；理论分析（定理 1 与 2）。

**📊 数据集**

基于 PyBullet 仿真平台的桌面拼块任务（3、5、7 块）作为实验数据集；未使用公开数据集，而是自生成仿真数据。

**📈 对比分析**

与全世界模型（few‑shot LLM）以及 oracle affordance 的完整模型进行对比，评估指标包括 MC simulations、LLM calls、步骤数、在线奖励等；实验结果表明部分模型显著降低搜索代价、提升奖励并在在线评估中取得更快任务完成速度。

**⚠️ 局限性**

依赖手工 prompt 设计的 task‑agnostic intents，affordance 误判会导致搜索灾难性失败；实验仅在仿真环境中验证，缺乏真实机器人实验，缺少对不同 LLM 与环境兼容性的进一步评估。

---

## 92. PhyCritic: Multimodal Critic Models for Physical AI

**arXiv ID:** 2602.11124 | [PDF](https://arxiv.org/pdf/2602.11124v1)

**作者:** Tianyi Xiong `[一作]`, Zhiding Yu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对物理AI任务的多模态评判（critic）模型，利用两阶段强化学习（RLVR + 自参照训练）提升模型的物理感知、因果推理与行动评估能力。

**💡 创新点**

创新点包括：①自参照critic训练框架，模型先生成自身答案再进行评判，增强判断稳定性与物理正确性；②两阶段RL策略，其中第一阶段物理技能预热，第二阶段自参照critic微调；③构建专门的物理AI评判基准（-Bench），填补评估空白。

**🔧 技术方法**

技术手段：基于Qwen2.5-VL-7B-Instruct的多模态语言模型；使用Group Relative Policy Optimization（GRPO）进行RL训练；自参照奖励机制（self‑prediction reward + critic reward）；强化学习与监督数据相结合的两阶段训练流程。

**📊 数据集**

数据集：Cosmos-Reason1-RL（物理QA数据）、RoboVQA、BridgeData V2、HoloAssist、AgiBot World、RoboFail、LingoQA，用于预热与评判训练；-Bench基准集合由上述数据集的多模态问题及候选答案构成。

**📈 对比分析**

评估方法：在专门的-Bench以及VL-RewardBench、Multimodal-RewardBench等公开基准上进行对比；结果显示该模型在-Bench上达成68%精度，超过所有开源7B/8B基线（最高56%）；在Cosmos-Reason1-Bench、CV-Bench、EgoPlanBench2等物理推理任务中也获得最佳或次佳表现。

**⚠️ 局限性**

局限性：自参照训练需要每个问题的真实答案作为奖励信号，限制了对完全开放式任务的适用性；训练仍需依赖有限的可验证物理数据，若数据不足可能影响泛化；并未解决多轮评判与自我完善的机制。

---

## 93. GraphSeek: Next-Generation Graph Analytics with LLMs

**arXiv ID:** 2602.11052 | [PDF](https://arxiv.org/pdf/2602.11052v1)

**作者:** Maciej Besta `[一作]` (ETH Zurich), Torsten Hoefler `[通讯]` (ETH Zurich)

**通讯引用:** 12612 | [OpenAlex ID](https://openalex.org/A5026990786)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种分离语义规划与执行的图数据库分析框架，利用LLM在语义平面进行多步查询规划，并在执行平面使用确定性后端实现。

**💡 创新点**

创新点在于构建紧凑的Semantic Catalog（包含模式和算子描述），实现LLM对查询语义的精确指引；通过TAG扩展实现多步、可自我纠错的工具调用链；并通过执行平面将大规模属性图的计算委托给非LLM执行器，保持上下文与令牌成本恒定。

**🔧 技术方法**

使用大型语言模型（GPT‑4o）、Tag（表增强生成）框架、工具集（Graph 操作器）、非LLM执行器（Cypher/Gremlin 编译器）以及语义目录与执行平面分离的设计。

**📊 数据集**

在两个真实数据集上评估：一是包含 230k 结点/314k 边的工业级 EV 制造图；二是 WikiDatasets Countries 3.3k 结点/10.2k 边的小型知识图。对比基线使用 LangChain 及其增强版本。

**📈 对比分析**

与基线相比，框架在成功率（≈86% vs 70%）、令牌消耗（更低且波动小）和响应时间（平均 2–3 秒）上均有显著提升；尤其在多跳、复杂模式匹配与结构歧义任务中优势更为突出。

**⚠️ 局限性**

限制主要在于：需要手工或半自动构建语义目录；当前实现仅支持属性图；缺乏公开代码与数据集，影响复现；对极大规模图（数十亿节点/边）仍需进一步优化执行层。

---

## 94. Dynamic Interference Management for TN-NTN Coexistence in the Upper Mid-Band

**arXiv ID:** 2602.10813 | [PDF](https://arxiv.org/pdf/2602.10813v1)

**作者:** Pradyumna Kumar Bishoyi `[一作]` (Indian Institute of Technology Jodhpur), Marina Petrova `[通讯]` (RWTH Aachen University)

**通讯引用:** 2704 | [OpenAlex ID](https://openalex.org/A5064915398)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了基于强化学习的动态干扰管理框架，联合优化地面网络（TN）的下行功率、上行功率与天线倾斜角，以在FR3上限中段频段保护非地面网络（NTN）性能。

**💡 创新点**

创新点在于将天线倾斜控制与功率调节协同纳入干扰抑制，通过PPO强化学习实现对TN网络的自适应调节，摆脱传统固定排斥区和对精准CSI的依赖。

**🔧 技术方法**

使用技术包括Proximal Policy Optimization（PPO）强化学习、联合优化模型、3GPP TR 38.811/38.901路径损耗与天线模型、干扰-噪声比（INR）评估以及中心化控制框架。

**📊 数据集**

数据集包括真实德国法兰克福基站坐标、基于Poisson点过程生成的NTN用户分布以及LEO卫星轨道与天线模型的仿真参数。

**📈 对比分析**

通过与无协调基线和ASCENT规则基准进行对比，采用累计INR分布、NTN下行吞吐量和TN基站活跃率等指标评估；PPO方案在所有NTN密度下相较ASCENT降低6–8 dB INR，TN基站活跃率维持≈87%，明显优于基准。

**⚠️ 局限性**

局限性包括依赖仿真场景与假设，未考虑多卫星、多频段共存与实际CSI误差；PPO训练时间长且对网络规模敏感；中心化控制对低延迟环境提出较高要求。

---

## 95. MacWilliams identities for the generalized rank weights

**arXiv ID:** 2602.10929 | [PDF](https://arxiv.org/pdf/2602.10929v1)

**作者:** Julien Molina `[一作]` `[通讯]` (Universite Grenoble Alpes), Julien Molina (Universite Grenoble Alpes)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了线性码的广义秩权分布，给出了代码与其双码之间的MacWilliams身份，并推导了枚举多项式的闭式公式，最终计算出MRD码的广义秩权分布。

**💡 创新点**

创新点在于首次针对广义秩权分布建立MacWilliams身份，给出统一的枚举多项式公式，并证明MRD码的分布仅由参数决定。

**🔧 技术方法**

主要使用了秩度量、Gaussian二项式系数、组合计数、q-变换与Sage数学软件进行符号计算。

**📊 数据集**

采用了具体示例码，如二次域上的[3,1]循环码、三次域上的[4,2]和[4,3]加比杜林码等进行验证。

**📈 对比分析**

通过对比不同等价码的分布，证明了分布相同；利用公式直接计算枚举多项式，验证了一致性，未涉及实际性能指标。

**⚠️ 局限性**

局限在于仅适用于$m\ge n$的线性码，且公式计算复杂度随参数增大而显著；未考虑非线性或环上的码。

---

## 96. Text-to-Vector Conversion for Residential Plan Design

**arXiv ID:** 2602.10757 | [PDF](https://arxiv.org/pdf/2602.10757v1)

**作者:** Egor Bazhenov `[一作]` (ITMO University), Valeria Efimova `[通讯]` (ITMO University)

**通讯引用:** 71 | [OpenAlex ID](https://openalex.org/A5025704136)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于文本描述的向量住宅平面图生成方法，结合自定义的白色背景损失函数和基于Shi‑Tomasi角点检测的向量化算法；

**💡 创新点**

创新点在于：①引入白色背景损失函数提升SDXL生成平面图的质量；②利用Shi‑Tomasi角点检测实现精准的右角和直墙向量化；③通过后处理消除冗余路径，得到干净结构化的SVG文件；

**🔧 技术方法**

采用Stable Diffusion XL加自定义白色背景损失、Shi‑Tomasi角点检测、阈值预处理、路径合并/删除后处理以及CLIPScore与Image Reward等评估技术；

**📊 数据集**

论文未给出具体公开数据集，实验使用约30个自然语言描述的平面图样本进行评估；

**📈 对比分析**

通过与多种生成+向量化组合（SD+DiffVG、AuraFlow+LIVE、SDXL+EvoVec、FLUX+SvgTracer）及LLM（GPT‑4、DeepSeek‑v3）对比，CLIPScore提升约5%，路径数12，处理时间23.3 s，整体性能优于现有方案；

**⚠️ 局限性**

局限性包括：对右角生成仍存在不足；LLM生成质量低；实验样本规模有限且缺乏公开数据集验证；对复杂平面布局的适应性尚待进一步评估。

---

## 97. Fast Person Detection Using YOLOX With AI Accelerator For Train Station Safety

**arXiv ID:** 2602.10593 | [PDF](https://arxiv.org/pdf/2602.10593v1)

**作者:** Mas Nurul Achmadiah `[一作]` (National Formosa University), Wen-Kai Kuo `[通讯]` (National Formosa University)

**通讯引用:** 698 | [OpenAlex ID](https://openalex.org/A5102949205)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究在火车站使用YOLOX结合Hailo-8 AI加速器实现快速人检测与安全警报系统。

**💡 创新点**

将YOLOX单阶段检测模型部署到边缘AI加速器Hailo-8，实现低延迟且功耗更低的实时人检测，并与Jetson Orin Nano比较验证性能提升。

**🔧 技术方法**

YOLOX检测算法、Hailo-8 AI加速器、HEF文件转换、HailoRT推理框架以及视频实时监控技术。

**📊 数据集**

使用COCO数据集（约14万张图像，80+类别）进行训练与评估。

**📈 对比分析**

在Leninskiy Prospekt地铁站场景下对比Hailo-8与Jetson Orin Nano，检测精度提升12%以上，延迟降低约20 ms，功耗更低，效率更高。

**⚠️ 局限性**

仅验证单个站点与COCO预训练模型，未针对实际站点多样化场景进行定制化训练，且未评估不同光照与拥挤度对检测精度的影响。

---

## 98. ArtisanGS: Interactive Tools for Gaussian Splat Selection with AI and Human in the Loop

**arXiv ID:** 2602.10173 | [PDF](https://arxiv.org/pdf/2602.10173v1)

**作者:** Clement Fuji Tsang `[一作]` (NVIDIA), Maria Shugrina `[通讯]` (NVIDIA and University of Toronto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套基于AI与人工交互的3D高斯散点（3DGS）选择与分割工具，支持快速从杂乱场景中提取可编辑对象。

**💡 创新点**

创新点在于：① 结合预训练的2D分割网络与可交互的Mask跟踪网络Cutie，实现一次点击即可自动生成全局3D分割；② 提供手动投影（视锥体/深度投影）与多种布尔操作的交互模式；③ 允许用户在交互过程中即时修正错误，显著提升分割鲁棒性。

**🔧 技术方法**

核心技术包括：预训练的SAM进行2D掩码生成、Cutie Mask跟踪网络、可微分3DGS渲染器用于掩码投影与优化、以及自定义交互界面与布尔操作逻辑。

**📊 数据集**

使用了公开的LERF-figurines场景、NVOS数据集，以及自制的Figurines3DSeg手工标注数据进行验证。

**📈 对比分析**

与FlashSplat、GaussianCut、GaussianEditor等基线进行对比，取得相当甚至更优的IoU/准确率，并在推理时间上显著加速（1–5秒完成一次分割），同时支持实时交互迭代。

**⚠️ 局限性**

局限性：对极端遮挡和复杂多物体场景仍易出现误分割；缺乏全自动无交互流程，仍需人工点击或Mask输入；在高分辨率或大规模场景下的实时性能尚未实现。

---

## 99. Risk-Equalized Differentially Private Synthetic Data: Protecting Outliers by Controlling Record-Level Influence

**arXiv ID:** 2602.10232 | [PDF](https://arxiv.org/pdf/2602.10232v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Bradley A. Malin `[通讯]` (Vanderbilt University Medical Center)

**通讯引用:** 11448 | [OpenAlex ID](https://openalex.org/A5090647314)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种风险均衡的差分隐私合成数据生成框架，旨在通过控制记录级别的影响来保护高风险记录，尤其是异常值。

**💡 创新点**

创新点在于通过风险加权机制，优先保护高风险记录，减少其对学习生成器的影响，从而提高这些记录的隐私保护。

**🔧 技术方法**

使用了差分隐私（DP）机制，结合了异常值评分和风险加权的DP学习方法。

**📊 数据集**

使用了模拟数据集（控制异常值注入）和真实世界基准数据集（如乳腺癌、成人收入、德国信用数据集）进行实验。

**📈 对比分析**

与标准DP合成方法相比，风险加权显著降低了高异常值记录的成员推断成功率，实验结果表明在不同数据集上性能有所不同，强调了评分器质量与合成管道之间的相互作用。

**⚠️ 局限性**

限制在于在某些数据集上，风险加权未能显著改善隐私保护，且在高阶特征交互驱动的脆弱性情况下，效果有限。

---

## 100. Neural Network Quantum Field Theory from Transformer Architectures

**arXiv ID:** 2602.10209 | [PDF](https://arxiv.org/pdf/2602.10209v1)

**作者:** Dmitry S. Ageev `[一作]` (Steklov Mathematical Institute, Russian Academy of Sciences), Yulia A. Ageeva `[通讯]` (Institute for Nuclear Research of Russian Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于Transformer注意力头的神经网络量子场论（NN-QFT）构造，将标量场定义为注意力头输出的线性读出，并通过对随机初始化参数取平均来定义n点相关函数；作者计算了两点函数并利用随机傅里叶特征嵌入实现欧氏不变核，进一步求得四点函数并揭示了单头模型在无限宽度下仍保持非高斯（相互作用）特性；随后将模型推广到多头，并证明在标准1/N_h方差归一化下，连通非高斯相关函数随头数N_h递减至零，恢复高斯场论。

**💡 创新点**

核心创新在于发现单一Transformer注意力头由于共享随机softmax权重，即使在无限宽度极限下也会产生“独立性破坏”贡献，导致四点连通函数非零，从而实现非高斯（相互作用）场论；同时给出该贡献的协方差表达式，并证明多头平均可抑制该效应，恢复Gaussian性质。

**🔧 技术方法**

使用了神经网络量子场论框架、随机傅里叶特征token嵌入、Gaussian初始化与中心极限定理、对softmax注意力权重的统计分析、以及大宽度与大头数极限下的协方差与连通函数计算。

**📊 数据集**

论文为理论研究，不涉及具体数据集。

**📈 对比分析**

对比方法主要是将两点函数与标准欧氏自由场传播子进行匹配；通过选择合适的傅里叶特征分布可使得两点函数与自由场传播子一致；未给出实验性能指标，只在理论层面展示了相关函数的闭式表达与极限行为。

**⚠️ 局限性**

局限性包括：仅考虑单层Transformer结构，未讨论训练过程对场论测度的影响；非高斯性仅在概率层面定义，缺乏对物理相互作用的更深入解释；未验证数值或实验结果；softmax归一化方式对非高斯性的影响仅在理论上说明，实际实现中可能面临数值不稳定等问题。

---

## 101. VFGS-Net: Frequency-Guided State-Space Learning for Topology-Preserving Retinal Vessel Segmentation

**arXiv ID:** 2602.10978 | [PDF](https://arxiv.org/pdf/2602.10978v1)

**作者:** Ruiqi Song `[一作]` (Sichuan Normal University), Nan Mu `[通讯]` (Sichuan Normal University)

**通讯引用:** 785 | [OpenAlex ID](https://openalex.org/A5055218366)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了VFGS-Net，一种端到端的视网膜血管分割网络，能够同时保留细小血管细节与全局血管拓扑结构。

**💡 创新点**

创新点在于三项模块的协同设计：① 双通路特征卷积（DFC）同时提取局部纹理与多尺度语义；② 血管感知频域通道注意力（VFCA）在FFT域自适应放大血管相关频率；③ 双向不对称Mamba2（BA‑Mamba2）在瓶颈处高效捕获长距离空间依赖，强化血管连续性。

**🔧 技术方法**

技术包括CNN编码解码骨干、双通路卷积、FFT与频域注意力、Mamba2状态空间模型、双向扫描、权重BCE+Dice损失、数据增强（翻转、旋转、Gamma、CLAHE）等。

**📊 数据集**

使用四个公开视网膜血管数据集：DRIVE、HRF、CHASE_DB1、STARE。

**📈 对比分析**

与U‑Net、DSCNet、HMT‑UNet、UltraLight、VM‑UNet、Serp‑Mamba等六种先进方法对比；VFGS‑Net在Dice（最高约85%）、HD95（最低约2-3像素）、ASSD等指标均优于对手，尤其在细小血管、低对比度区域和复杂分支的分割精度显著提升。

**⚠️ 局限性**

局限性包括：对极低对比度或严重噪声图像的鲁棒性仍有限；模型在推理时参数量与计算量较大，尚未针对实时临床部署做进一步优化；仅在公开数据集验证，缺乏大规模临床病例的进一步评估。

---

## 102. A Diffusion-Based Generative Prior Approach to Sparse-view Computed Tomography

**arXiv ID:** 2602.10722 | [PDF](https://arxiv.org/pdf/2602.10722v1)

**作者:** Davide Evangelista `[一作]` (University of Bologna), Elena Loli Piccolomini `[通讯]` (University of Bologna)

**通讯引用:** 1014 | [OpenAlex ID](https://openalex.org/A5053041046)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

使用DDIM扩散模型作为先验，提出带有数据保真、潜在正则和图像层次TV正则的正则化最小化框架，用FBP逆向初始化和余弦退火学习率，进行稀疏视角CT重建。

**💡 创新点**

创新点在于：①将滤波背投影(FBP)重建与DDIM逆向过程结合实现更可靠的潜在空间初始化；②在DGP优化中加入自适应学习率（余弦退火）提升收敛性；③同时在潜在空间与图像空间加入正则化项，提升重建质量。

**🔧 技术方法**

采用DDIM（去噪扩散隐式模型）作为生成器；基于Adam优化的潜在空间最小化；使用余弦退火学习率；加入正则化项（潜在高斯先验、TV正则）。

**📊 数据集**

训练集为Mayo Clinic低剂量CT数据集（约3305张512×512腹部切片，缩放至256×256）；测试集采用同一数据集中的若干样本；仿真稀疏投影使用ASTRA Toolbox。

**📈 对比分析**

与DPS、DiffPIR、DDRM和DMPlug进行对比，使用MSE/PSNR/SSIM等指标。实验显示在45°/60°投影角下，RD‑DGP与DDRM相当甚至略优；在30°投影下略逊；可视化结果表明RD‑DGP更能保持形状完整、噪声更低。

**⚠️ 局限性**

局限性包括：①计算成本高，训练/重建耗时；②对极端稀疏角度（30°）仍易陷入次优局部最优；③仅在有限的Mayo数据集上验证，泛化性待进一步验证；④模型对超参数（λ1, λ2）敏感，需要更多实验。

---

## 103. Digging for Data: Experiments in Rock Pile Characterization Using Only Proprioceptive Sensing in Excavation

**arXiv ID:** 2602.11082 | [PDF](https://arxiv.org/pdf/2602.11082v1)

**作者:** Unal Artan `[一作]` (Orebro University), Joshua A. Marshall `[通讯]` (Smith Engineering)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在瑞典Eker采石场对Epiroc电动LHD进行225次完整规模挖掘实验，利用装载机上安装的IMU加速度和液压压力传感器进行自我感知，提取连续小波特征ζ，并用其比值估计不同碎石堆的相对平均粒径。

**💡 创新点**

创新点在于：①仅用自我感知传感器而不依赖摄像或LiDAR即可实时估算碎石粒径；②证明小波特征比值与不同堆的平均粒径比例呈近似线性关系，形成一种新的“相对粒径估计”方法；③在现场大规模验证了该方法的可行性，为后续自主挖掘控制提供了闭环反馈。

**🔧 技术方法**

使用技术包括：高频IMU（1000 Hz）加速度采集、液压压力/位置采集（250 Hz）、连续小波变换（MATLAB Wavelet Toolbox）、特征归一化与比值计算、基于阈值的挖掘时间窗口确定、以及统计比较（均值±标准差）。

**📊 数据集**

数据集包括：五种碎石堆（0/32、0/63、0/90、0/150、0/1500），两位操作员的不同操作风格，225次挖掘实验记录的IMU、压力、位置以及对应的视觉图像（WipFrag）和实验室筛分（sieve）粒径数据。

**📈 对比分析**

与筛分和WipFrag视觉系统的相对粒径估计进行比较。结果显示：①自我感知方法在区分不同堆的相对粒径方面与筛分和WipFrag的准确度相当；②在粒径分布相近的堆（如0/63与0/90）时区分度下降，但总体误差低于±0.3；③分类实验表明，对相对粒径小于或大于参考堆的判定准确率≥90%。

**⚠️ 局限性**

局限性包括：①对操作员行为敏感，需统一或校准操作模式；②液压压力传感器在主动作业时失效，无法用于粒径估计；③仅提供相对平均粒径，无法得到完整粒径分布；④筛分作为“真值”本身存在代表性样本偏差；⑤实验仅在单一矿区进行，需进一步验证不同地质、密度、湿度等条件下的鲁棒性。

---

## 104. Search or Accelerate: Confidence-Switched Position Beam Search for Diffusion Language Models

**arXiv ID:** 2602.10953 | [PDF](https://arxiv.org/pdf/2602.10953v1)

**作者:** Mingyu Cao `[一作]` (Qualcomm AI Research), Lu Yin `[通讯]` (University of Surrey)

**通讯引用:** 12124 | [OpenAlex ID](https://openalex.org/A5045336307)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种训练无关的自适应解码框架，结合位置光束搜索与置信度驱动的并行解码，以在Diffusion Language Models中实现生成质量与速度的平衡。

**💡 创新点**

提出了置信切换位置光束搜索（SOAR），动态在搜索与并行模式之间切换并自适应调整候选规模，从而在不增加显著推理延迟的前提下提升输出质量。

**🔧 技术方法**

利用位置光束搜索（PBS）、置信度阈值驱动的并行解码、基于平均置信度的得分函数、Softmax置信度评估、可变长度解码与替代的Unmask指标等技术。

**📊 数据集**

在数学推理数据集GSM8K、代码生成数据集MBPP和HumanEval以及HumanEval-Infilling上进行实验，并在Dream-7B和LLaDA-8B两种Diffusion Language Models上进行评估。

**📈 对比分析**

与贪心解码、适应性并行解码和PBS单/多token等方法对比，平均准确率/通过率提升约2%–7%，速度保持或略快；Pareto曲线显示在保持速度的同时显著提升质量。

**⚠️ 局限性**

受置信阈值和beam宽度等超参的影响；搜索宽度与计算线性相关，极大beam会显著延迟；对置信度估计的依赖可能在低置信场景下效果下降。

---

## 105. Linguistic Indicators of Early Cognitive Decline in the DementiaBank Pitt Corpus: A Statistical and Machine Learning Study

**arXiv ID:** 2602.11028 | [PDF](https://arxiv.org/pdf/2602.11028v1)

**作者:** Artsvik Avetisyan `[一作]` (American University of Armenia), Sachin Kumar `[通讯]` (American University of Armenia)

**通讯引用:** 5262 | [OpenAlex ID](https://openalex.org/A5032948428)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

研究者通过对DementiaBank Pitt语料库中多种语言任务的自动转录文本进行特征抽取，比较了原始文本、POS增强和POS仅化三种语言表征，并使用逻辑回归和随机森林两种可解释模型在转录级别和受试者级别两种验证协议下对早期认知衰退进行分类，进一步通过全局特征重要性和非参数统计检验验证所选特征的稳健性。

**💡 创新点**

创新点在于（1）同时评估了词汇、句法及其组合的三种抽象语言表征，证明即使去除词汇信息仍能捕捉认知衰退信号；（2）在临床可行的受试者级交叉验证框架下进行性能评估，避免了发音者泄漏；（3）将可解释机器学习与统计效应大小结合，提供了多视角验证；（4）系统展示了不同表征下的特征重要性与统计显著性的一致性。

**🔧 技术方法**

采用的技术包括：文本预处理、词汇多样性（TTR、MTTR）、句子结构统计、语义连贯度、POS分布等特征抽取；逻辑回归（L2正则化）和随机森林（200棵树、均衡类别权重）作为可解释模型；全局特征重要性提取（回归系数绝对值、特征重要度平均）；Mann‑Whitney U检验与Cliff’s delta效应量；Benjamini–Hochberg多重检验校正；GroupKFold（受试者级）交叉验证。

**📊 数据集**

使用的主要数据集为DementiaBank Pitt Corpus（总500个转录，控制243，痴呆257），主要任务为Cookie Theft图片描述、故事回忆、词汇流畅性和句子重复，采用CHAT格式转录。

**📈 对比分析**

在转录级别下，逻辑回归+POS增强实现最高准确率≈0.75、宏平均F1≈0.75；POS仅与原始文本相近；在受试者级交叉验证下，最佳表现为逻辑回归+POS增强，平均准确率≈0.72、宏平均F1≈0.71，显示模型在去除个体泄漏后仍保持相对稳健但略低于转录级别。

**⚠️ 局限性**

局限性包括：仅采用单一英语任务（Cookie Theft）导致结果可能不具备跨任务、跨语言、跨文化的泛化能力；未使用声学或韵律特征，可能限制识别性能；缺乏纵向追踪数据，无法研究个体随时间变化；存在潜在的人口学偏倚与不确定性，未采用专门的偏差缓解或不确定性建模；解释性评估仅基于全局特征重要性，未进行临床医生用户研究。

---

## 106. Interpretable Vision Transformers in Image Classification via SVDA

**arXiv ID:** 2602.10994 | [PDF](https://arxiv.org/pdf/2602.10994v1)

**作者:** Vasileios Arampatzakis `[一作]` (Democritus University of Thrace), Nikos Papamarkos `[通讯]` (Democritus University of Thrace)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5109991926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将SVDA（SVD-Inspired Attention）整合进Vision Transformer，实现可解释的自注意力机制。

**💡 创新点**

创新点在于通过软正交投影和学习对角谱矩阵，将注意力分解为方向与谱重要性两部分，并提出六项结构化可解释指标。

**🔧 技术方法**

采用SVD-Inspired Attention、软正交投影、谱正则化以及六项可解释性诊断指标。

**📊 数据集**

使用FashionMNIST、CIFAR-10、CIFAR-100和ImageNet-100四个常用图像分类数据集。

**📈 对比分析**

与基准ViT在相同架构、训练设置下进行对比，分类准确率基本相同，SVDA在可解释性方面显著提升，训练时间略增加约17%。

**⚠️ 局限性**

限制在于仅提供描述性可解释性，未能通过可解释指标驱动训练；在大规模模型和推理效率方面仍需改进。

---

## 107. Are More Tokens Rational? Inference-Time Scaling in Language Models as Adaptive Resource Rationality

**arXiv ID:** 2602.10329 | [PDF](https://arxiv.org/pdf/2602.10329v1)

**作者:** Zhimin Hu `[一作]` (Georgia Tech), Sashank Varma `[通讯]` (Georgia Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型在变量归因任务中的推理行为，探讨推理时间扩展是否能自然产生资源合理的策略。

**💡 创新点**

创新点在于提出并系统使用变量归因任务（VAT）来操纵任务复杂度，并证明推理时间扩展可以在未显式惩罚计算成本的情况下，使模型自适应地切换到更资源合理的策略。

**🔧 技术方法**

采用指令微调（IT）和强化学习（LRM）两类大语言模型，利用外部判别模型识别模型推理策略，并通过字符计数衡量推理成本。

**📊 数据集**

使用自行生成的3000个变量归因样本，涵盖10种布尔逻辑函数、10种候选变量数与多余试验组合，构成完整的数据集。

**📈 对比分析**

对比IT模型与LRM，在不同逻辑函数与任务复杂度下测量准确率与字符消耗；结果显示LRM保持高准确率且推理成本相对可控，IT模型在XOR/XNOR等非线性函数下准确率显著下降。

**⚠️ 局限性**

主要局限在于仅评估二元布尔函数且未考虑更高阶逻辑，判别模型的识别误差以及模型在更大规模任务中的泛化能力尚未验证。

---

## 108. Gauss-Newton Unlearning for the LLM Era

**arXiv ID:** 2602.10568 | [PDF](https://arxiv.org/pdf/2602.10568v1)

**作者:** Lev McKinney `[一作]` (University of Toronto), Roger Grosse `[通讯]` (University of Toronto)

**通讯引用:** 8010 | [OpenAlex ID](https://openalex.org/A5067036768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 K-FADE，一种基于 Gauss-Newton 步长和 K-FAC Hessian 近似的 LLM 反学习方法，能在不显著降低模型性能的前提下抑制指定遗忘集的输出。

**💡 创新点**

创新点：①将 Gauss-Newton 更新直接用于 LLM 反学习，自动平衡输出抑制与保留集的特异性；②使用 K-FAC/ EK-FAC 对 Hessian 进行参数化近似，显著提升二阶方法的效率与精度；③提出单步 Gauss-Newton 即可匹配重训练的输出，并可轻松转移到后续微调模型。

**🔧 技术方法**

技术：Gauss-Newton 上升步、K-FAC（或 EK-FAC）Hessian 近似、对抗性损失（负边际/交叉熵）、阻尼正则化、梯度方向归一化、损失加权与步骤大小控制。

**📊 数据集**

数据集：WMDP（武器类别多选题）、ToFU（虚构作者问答）、MMLU、MT-Bench、Alpaca、Wikitext、Phi-1.5 以及 7B Llama‑2、Zephyr‑7B‑β 等模型。

**📈 对比分析**

比较方法：与 RMU、ELM、SimNPO、SOUL、梯度上升/梯度差异、KL‑min 等基线对比；在 WMDP 上实现最优抑制同时保持最优特异性；在 ToFU 上单步 Gauss-Newton 超越所有基线的遗忘质量和模型效用；运行时与一阶方法相当，二阶方法在速度/精度上更具竞争力。

**⚠️ 局限性**

局限：缺乏正式的差分隐私或安全性保证；对完整权重访问的攻击能在微调后恢复遗忘效果；在遗忘集上仍可能出现流畅度下降；在极大规模模型上的内存与训练成本仍需进一步优化。

---

## 109. On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models

**arXiv ID:** 2602.10298 | [PDF](https://arxiv.org/pdf/2602.10298v1)

**作者:** Polina Tsvilodub `[一作]` (University of Tuebingen), Michael Franke `[通讯]` (University of Tuebingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过行为评估和因果消融实验，探究大型语言模型是否在理论心智（ToM）与语用推理之间共享计算机制。

**💡 创新点**

创新点在于提出功能整合假设，构造四套基于认知神经科学的合成本地化材料，结合ATOMS细粒度分析，并将共轭分析方法引入LM可解释性研究。

**🔧 技术方法**

采用了自回归Transformer的激活统计本地化、Welch t检验、贝叶斯β回归、GPT‑5生成合成刺激、以及对局部网络的零化消融等技术。

**📊 数据集**

使用了22个ToM基准、16个语用基准、BLiMP句法基准，并通过GPT‑5生成的1400个本地化刺激（四套本地化）。

**📈 对比分析**

通过将ToM与语用任务的准确率进行相关性、预测冗余及预测增益分析，并对消融效果进行贝叶斯回归比较；结果显示ToM与语用表现正相关，消融ToM网络显著降低两者准确率，未影响句法。

**⚠️ 局限性**

局限性包括对刺激生成与本地化方法的假设、数据集单一语言及可能的训练数据泄漏、合成本地化未在人类fMRI中验证、以及消融仅能给出粗略因果洞察。

---

## 110. Area-Efficient In-Memory Computing for Mixture-of-Experts via Multiplexing and Caching

**arXiv ID:** 2602.10254 | [PDF](https://arxiv.org/pdf/2602.10254v1)

**作者:** Hanyuan Gao `[一作]` (University of Virginia), Xiaoxuan Yang `[通讯]` (University of Virginia)

**通讯引用:** 6679 | [OpenAlex ID](https://openalex.org/A5080236384)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种针对MoE Transformer的面积高效PIM加速器，利用交叉点多路复用、专家分组与调度以及门输出缓存提升性能与能效。

**💡 创新点**

创新点包括：①跨交叉点共享外围电路的多路复用方案；②负载感知专家分组与动态预填调度以缓解负载不平衡；③门输出（GO）缓存配合KV缓存，在自回归生成阶段显著降低计算与能耗。

**🔧 技术方法**

使用的技术包括：PIM存储器阵列（256×256交叉点）、ADC共享、静态负载感知分组、动态调度、门输出缓存、KV缓存，以及基于3DCIM和HERMES规格的仿真器。

**📊 数据集**

实验使用Pajama C4数据集抽样获取门网络工作负载，并以Llama‑MoE‑4/16（Llama2‑7B MoE变体）作为评测模型。

**📈 对比分析**

与3DCIM基线（无共享、无调度）比较，实验显示在生成8个token时，KVGO缓存+动态调度（S2O）可将延迟提升4.2×、能耗降低10.1×，整体性能密度达到15.6 GOPS/W/mm²，比基线提升1.53×。

**⚠️ 局限性**

局限性包括：方案依赖MoE的稀疏激活与静态分组，较大交叉点群组会产生竞争导致性能下降；缓存规模受离线存储限制；未考虑硬件噪声与误差对MoE推理精度的影响。

---

## 111. 3DXTalker: Unifying Identity, Lip Sync, Emotion, and Spatial Dynamics in Expressive 3D Talking Avatars

**arXiv ID:** 2602.10516 | [PDF](https://arxiv.org/pdf/2602.10516v1)

**作者:** Zhongju Wang `[一作]` (University of New South Wales), Hongdong Li `[通讯]` (Australian National University)

**通讯引用:** 17922 | [OpenAlex ID](https://openalex.org/A5101819061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出3DXTalker，一种基于音频驱动的3D谈话头像生成框架；

**💡 创新点**

①构建2D‑>3D身份建模管道并使用解耦FLAME表示；②引入帧级幅度与情绪特征提升唇同步与表情表达；③融合流匹配Transformer实现统一的身份、情绪、姿态控制；

**🔧 技术方法**

EMOCA、FLAME、WavLM、emotion2vec、流匹配Transformer、头部姿态控制模块、LLM生成头部轨迹；

**📊 数据集**

GRID、RAVDESS、MEAD、VoxCeleb2、HDTF、CelebV‑HQ等6个公开视频/音频数据集；

**📈 对比分析**

与FaceFormer、CodeTalker、SelfTalk、FaceDiffuser、EMOTE、DiffPoseTalk、DEEPTalk等7个基线对比，3DXTalker在3D几何误差、唇同步、情绪FID、头部节拍等指标上均优于或竞争，用户研究平均排名4.22，帧率≈69 FPS；

**⚠️ 局限性**

对音频的依赖仍较强，情绪与幅度特征需要额外提取；头部姿态控制依赖LLM生成轨迹，可能缺乏细粒度手动调节；

---

## 112. Multimodal Information Fusion for Chart Understanding: A Survey of MLLMs -- Evolution, Limitations, and Cognitive Enhancement

**arXiv ID:** 2602.10138 | [PDF](https://arxiv.org/pdf/2602.10138v1)

**作者:** Zhihang Yi `[一作]` (Sichuan University), Tao Wang `[通讯]` (Engineering Research Center of Machine Learning and Industry Intelligence Ministry of Education)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了从传统方法到最新 MLLM 在图表理解中的演进，提出了信息融合视角和图表类型的规范化分类，系统梳理任务、数据集、方法与挑战，并对未来方向做出建议。

**💡 创新点**

① 引入“canonical vs non‑canonical”图表分类，凸显研究盲点；② 以信息融合为核心的整体框架，强调视觉与语言的对齐；③ 汇总并更新 2025 年前沿方法（RL、Prompt、MoE 等），为研究者提供系统化路线图。

**🔧 技术方法**

主要基于文献检索与归纳，结合对 MLLM 结构（ViT+LLM+跨模态 Transformer）、Prompt（CoT/CCR/PoT）、强化学习（RLHF/DPO/GRPO）以及多模态中间表征（表格、代码、SVG）等技术的梳理与评述。

**📊 数据集**

对比了主流图表数据集如 FigureQA、DVQA、PlotQA、ChartQA、ChartSFT、ChartX、SciCap、FlowVQA、MapQA 等，区分了基于统计图（canonical）与复杂图表（non‑canonical）的数据来源与规模。

**📈 对比分析**

论文未提出新的实验模型，而是对现有方法在各类基准上的表现做综述，指出传统模型受词表限制，MLLM 在多任务（Caption、QA、C2T/C2C）上表现优异但仍落后于专门化系统；同时强调 RL/Prompt 可提升推理准确率但仍受限于视觉细节编码。

**⚠️ 局限性**

① 视觉编码缺乏高频细节，导致对条形高度、线条斜率等信息捕捉不足；② 语言解码无法精准映射连续视觉量到离散符号，易产生幻觉；③ 依赖 OCR 与文本提示，对无文本图表鲁棒性差；④ 对复杂多模态图表（非规范化、交互式）仍缺乏通用推理框架。

---

## 113. ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving

**arXiv ID:** 2602.10884 | [PDF](https://arxiv.org/pdf/2602.10884v1)

**作者:** Jinqing Zhang `[一作]` (Beihang University), Yunhong Wang `[通讯]` (Beihang University)

**通讯引用:** 13833 | [OpenAlex ID](https://openalex.org/A5115589096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种端到端自动驾驶框架ResWorld，利用时间残差世界模型（Temporal Residual World Model, TR-World）和未来引导轨迹细化（Future-Guided Trajectory Refinement, FGTR）来提升对动态物体的建模与轨迹规划；

**💡 创新点**

创新点包括：①使用时间残差提取动态物体信息，避免冗余建模静态物体；②仅对时间残差建模，聚焦动态物体未来分布；③FGTR模块通过与未来BEV特征交互细化轨迹，同时提供稀疏空间-时间监督，防止世界模型崩溃；

**🔧 技术方法**

技术手段包括：基于BEV特征的多时刻对齐、TokenLearner与TokenFuser、Transformer自注意力、Deformable Attention、MLP解码、L1损失；

**📊 数据集**

实验数据集为nuScenes（开环评估）和NAVSIM（闭环评估），并在两大基准上验证性能；

**📈 对比分析**

与现有端到端方法（如SSR、LAW、UniAD、GenAD等）相比，ResWorld在nuScenes上实现了最优的L2误差与碰撞率，尤其在不使用辅助任务时表现最优；在NAVSIM上获得最高的PDMS评分（89.0%），超过同类世界模型与感知任务方法；

**⚠️ 局限性**

局限性在于对潜在动态物体（如行人、停驶车辆）依赖时间残差的敏感性不足，导致此类物体只能通过先前轨迹分支处理，未来工作计划加入粗粒感知以提升对潜在动态物体的预防建模。

---

## 114. Gated Removal of Normalization in Transformers Enables Stable Training and Efficient Inference

**arXiv ID:** 2602.10408 | [PDF](https://arxiv.org/pdf/2602.10408v1)

**作者:** Andrei Kanavalau `[一作]` (Stanford University), Sanjay Lall `[通讯]` (Stanford University)

**通讯引用:** 8174 | [OpenAlex ID](https://openalex.org/A5082736758)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

做了一个可调节的归一化模块TaperNorm，允许Transformer在训练早期使用标准归一化，随后逐渐去掉token级统计，使得推理时可将归一化合并到线性层。

**💡 创新点**

创新点是提出了门控渐变归一化TaperNorm，并通过尺度锚定理论解释归一化作用，同时提供固定目标尺度损失以实现最终归一化的消除。

**🔧 技术方法**

用了技术：门控归一化、指数移动平均（EMA）校准、余弦门衰减、固定目标尺度损失、权重折叠（weight folding）以及在推理中将归一化合并到线性投影。

**📊 数据集**

用了数据集：TinyStories（预训练）、OpenWebText、The Pile 与 Pile-filtered（GPT‑2微调）。

**📈 对比分析**

如何比较的方法：与RMSNorm/LayerNorm基线在验证集上对比，评估损失差距（≤2%），并在NVIDIA H100上进行推理吞吐量微基准，结果在推理吞吐量上提升约1.13–1.22×。

**⚠️ 局限性**

limitation是仅在小到中型模型、单一硬件平台进行微基准；未验证大模型、长上下文或完整生成任务，且对超参数的鲁棒性仍需进一步研究。

---

## 115. Versor: A Geometric Sequence Architecture

**arXiv ID:** 2602.10195 | [PDF](https://arxiv.org/pdf/2602.10195v1)

**作者:** Truong Minh Huy `[一作]`, Edward Hirst `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了基于共形几何代数的序列模型Versor，用于替代Transformer的线性操作，实现对SE(3)对称性的几何化建模，并在多模态和物理任务上进行验证。

**💡 创新点**

使用CGA实现多向量状态表示、几何乘积注意力和递归旋转累加器，天然实现SE(3)等变换、线性O(L)复杂度、极高参数效率和可解释的距离/方向分量。

**🔧 技术方法**

共形几何代数、几何乘积注意力(GPA)、递归旋转累加器(RRA)、自定义Triton/MLX Clifford核、可编译C++核心、Hamiltonian‑Versor混合等。

**📊 数据集**

N体动力学轨迹、Broken Snake拓扑任务、CIFAR‑10视觉、WikiText‑103文本、MD17分子动力学、图形与合成控制基准。

**📈 对比分析**

与Transformer、GNS、EGNN、GATr、Mamba等基线在参数、推理延迟、能量漂移、预测误差、拓扑识别等指标对比，Versor在仅6K参数的情况下，MSE比Transformer低1–2倍，能量漂移显著减小，拓扑识别99.3%而ViT仅50%；推理复杂度为O(L)而Transformer为O(L²)。

**⚠️ 局限性**

算子常数因子高、GPU对32维多向量寄存器支持不足、长序列出现数值漂移、缺乏专用硬件（GAPU）和更高阶几何优化。

---

## 116. Analyzing Fairness of Neural Network Prediction via Counterfactual Dataset Generation

**arXiv ID:** 2602.10457 | [PDF](https://arxiv.org/pdf/2602.10457v1)

**作者:** Brian Hyeongseok Kim `[一作]` (University of Southern California), Chao Wang `[通讯]` (University of Southern California)

**通讯引用:** 31362 | [OpenAlex ID](https://openalex.org/A5100406891)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用线性回归近似和神经元激活相似度，对训练标签进行排序并少量翻转，以生成能改变指定测试样本预测的对抗性训练集（CFD）的算法，进而评估标签偏差导致的模型不公平性。

**💡 创新点**

创新点在于：①首次将训练阶段与推理阶段相结合，使用线性回归替代器和激活相似度共同指导标签翻转；②通过构造对抗性训练集直接审计个体预测，而非传统的输入对抗或全局公平性评估；③在深度网络上实现高效的CFD生成，避免枚举和反复训练的指数开销。

**🔧 技术方法**

使用的技术包括：PyTorch实现的神经网络训练；线性回归（Ridge）作为近似替代器计算标签影响系数；激活相似度（基于隐藏层二进向量的汉明距离）评估输入相似性；以及对排名进行归一化融合得到最终标签翻转顺序。

**📊 数据集**

实验使用七个常见公平性数据集：Salary、Student、German、Compas、Default、Bank、Adult。

**📈 对比分析**

与随机采样、基于L₂距离的启发式、以及四种影响函数（Explicit、Conjugate Gradients、LiSSA、Arnoldi）做对比。实验结果显示，所提方法在成功率上显著高于基线（尤其在小数据集上几乎覆盖所有真实CFD），在较大数据集上提升约9–22%的成功率；计算时间与基线相近，且由于更快的CFD发现，整体耗时更少；生成的CFD在训练样本与目标测试样本的相似度上优于基线，且在接近与远离决策边界的样本上均能有效工作。

**⚠️ 局限性**

局限性包括：①需要设定过滤规则ϕ与ψ，需领域专家介入；②对标签翻转数量有限制（m），无法处理大量标签错误；③仍需对网络重新训练，计算成本在大规模数据上仍显著；④对非ReLU激活的近似可能降低准确性；⑤方法在极少数测试样本上可能找不到CFD，说明标签偏差与预测不完全相关。

---

## 117. Neuro-Symbolic Synergy for Interactive World Modeling

**arXiv ID:** 2602.10480 | [PDF](https://arxiv.org/pdf/2602.10480v1)

**作者:** Hongyu Zhao `[一作]` (University of Maryland), Tianyi Zhou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了神经符号协同框架Neuro-Symbolic Synergy，用于交互式世界建模，直接用可执行符号规则调节LLM输出概率以实现逻辑一致性和语义表达；

**💡 创新点**

创新点在于：1) 通过符号规则作为能量项直接修改LLM概率分布，而非依赖提示；2) 引入相互迭代的训练循环和符号引导的数据筛选，显著减少训练样本；3) 通过规则与神经模型互补的阶段性训练实现性能提升；

**🔧 技术方法**

使用的大型语言模型包括Llama3.2‑1B Instruct与Qwen3‑4B，符号组件则是可执行的Python规则；训练方法采用阶段性微调与规则权重学习，数据处理通过规则覆盖率筛选；

**📊 数据集**

实验数据集涵盖三类交互环境：ScienceWorld（科学实验场景）、Webshop（电商交互）以及Plancraft（Minecraft制作品），每个环境提供多轮决策轨迹与多项选择测试；

**📈 对比分析**

与多种基线对比（未微调LLM、全量微调、开源与专有模型），结果显示在所有环境下Neuro‑Symbolic Synergy在平均准确率上超过基线，且仅使用约45–60%训练数据即可匹配或优于全量微调；

**⚠️ 局限性**

局限性包括：1) 需要可手动或自动生成的可执行规则，规则覆盖率不足时效果受限；2) 在神经模型质量极低时，规则对概率重排的负面影响可能导致性能下降；3) 路由机制仍较简单，缺乏动态决策支持。

---

## 118. Adaptive Optimization via Momentum on Variance-Normalized Gradients

**arXiv ID:** 2602.10204 | [PDF](https://arxiv.org/pdf/2602.10204v1)

**作者:** Francisco Patitucci `[一作]` (University of Texas Austin), Aryan Mokhtari `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的Adam风格优化器 MVN-Grad，它在梯度先做方差归一化后再加动量，提升训练稳定性和收敛速度。

**💡 创新点**

创新点在于两方面：① 将动量运算顺序改为先归一化后动量，消除传统 Adam 的时序耦合导致的跨时刻不稳定；② 用梯度方差（而非无中心二阶矩）做归一化，避免低方差时的符号坍塌，提高梯度幅度信息保留。

**🔧 技术方法**

使用指数移动平均 (EMA) 估计梯度均值和方差；实现归一化后动量的更新规则；结合理论分析（条件方差比较、单尖峰鲁棒性证明、低方差收敛性证明）与实验评估。

**📊 数据集**

在 CIFAR-100（ResNet‑18）图像分类、GPT‑2 风格语言建模（WikiText‑103、OpenWebText）以及 MNIST 用于理论验证的数据集上进行实验。

**📈 对比分析**

通过在相同模型架构、相同超参数搜索空间下与 Adam、AdaBelief、LaProp 等基线进行对比。实验显示 MVN-Grad 在测试准确率、验证 NLL/Perplexity 上与或优于这些基线，且训练过程更平滑、对超参数更鲁棒。

**⚠️ 局限性**

局限性包括：仍需手工调节 β1、β2；理论分析基于理想的噪声与梯度平稳假设，未覆盖极端噪声或非平稳场景；实现与其他 Adam 变体计算量相当，未提出显著的加速方案。

---

## 119. Natural Hypergradient Descent: Algorithm Design, Convergence Analysis, and Parallel Implementation

**arXiv ID:** 2602.10905 | [PDF](https://arxiv.org/pdf/2602.10905v1)

**作者:** Deyi Kong `[一作]` (University of Minnesota), Shancong Mou `[通讯]` (University of Minnesota)

**通讯引用:** 112 | [OpenAlex ID](https://openalex.org/A5067794264)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了自然超梯度下降 (NHGD) 方法，用于求解双层优化问题，并将内层 SGD 与 Hessian 逆近似同步进行，形成一种并行 optimize‑and‑approximate 框架。

**💡 创新点**

创新点在于：① 利用内层 KL 散度最小化的统计结构，用经验 Fisher 信息矩阵（EFIM）逼近海森逆；② 通过 Sherman‑Morrison 进行秩‑1 更新，实现零额外成本的并行更新；③ 给出了高概率样本复杂度和收敛率理论，匹配现有 optimize‑then‑approximate 方法，但显著降低计算时间。

**🔧 技术方法**

技术手段包括：SGD 内层优化、EFIM 逆更新（rank‑1 Sherman‑Morrison）、自然梯度思想、K‑FAC 进行大规模块级近似、并行/同步实现、以及高概率理论分析。

**📊 数据集**

实验使用了 MNIST（带标签噪声）进行超数据清洗，Fashion‑MNIST 用于数据蒸馏，以及二维热方程 PDE 控制问题（物理信息学习）。

**📈 对比分析**

与 Neumann 系列、CG 近似、stocBiO、AmIGO、TTSA、SOBA 以及 Broyden‑based 方法对比。NHGD 在收敛速度上通常快 3–4 倍，最终测试准确率或目标值与最先进方法持平或更优，尤其在大规模机器学习任务中表现突出。

**⚠️ 局限性**

局限性包括：目前仅在双循环框架下证明，单循环或更复杂的内层结构尚未推广；对内层数据分布随外层变量变化的情形缺乏专门处理；理论假设（如 KL 匹配、强凸、光滑）较严格；在极端非高斯噪声或模型误设情况下性能尚未验证。

---

## 120. VulReaD: Knowledge-Graph-guided Software Vulnerability Reasoning and Detection

**arXiv ID:** 2602.10787 | [PDF](https://arxiv.org/pdf/2602.10787v1)

**作者:** Samal Mukhtar `[一作]` (University of Manchester), Youcheng Sun `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 2214 | [OpenAlex ID](https://openalex.org/A5060663223)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于安全知识图谱的漏洞推理与检测框架，能够在函数级别对代码进行二分类与CWE级别诊断，并给出结构化自然语言解释；

**💡 创新点**

创新点在于：①通过知识图谱构建CWE抽象与实体关联，生成对比推理对；②结合教师LLM与Odds Ratio Preference Optimization (ORPO)的教师-学生框架，进行参数高效微调；③实现长尾CWE多分类推理，显著提升解释的语义一致性；

**🔧 技术方法**

主要技术包括：大语言模型(如Qwen2.5-7B-Instruct)、知识图谱构建与检索、教师-学生对比推理生成、ORPO偏好优化、QLoRA参数高效微调、以及宏/微 F1 等评估指标；

**📊 数据集**

使用了三大真实世界漏洞数据集 PrimeVul、DiverseVul 与 R2Vul，覆盖约140–270种CWE；

**📈 对比分析**

与传统深度学习模型(BiLSTM、SySeVR、Devign)以及LLM基线(Zero-shot、ICL、COT、SFT、R2Vul、LLMxCPG、ReVD)进行对比；二分类 F1 提升约 8%–10%，多分类 Macro‑F1 提升约 30%，KG 引导显著提升检测与推理性能，尤其在长尾 CWS 上表现突出；

**⚠️ 局限性**

局限性包括：教师生成的推理对可能存在噪声；CWE 抽象层次不唯一可能影响结果；KG 检索时可能返回冗余或误导信息；数据集偏差与计算资源限制限制了更大模型和更广泛场景的验证。

---

## 121. Understanding the Effects of AI-Assisted Critical Thinking on Human-AI Decision Making

**arXiv ID:** 2602.10222 | [PDF](https://arxiv.org/pdf/2602.10222v1)

**作者:** Harry Yizhou Tian `[一作]` (Purdue University), Ming Yin `[通讯]` (Purdue University)

**通讯引用:** 6205 | [OpenAlex ID](https://openalex.org/A5101993864)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了AI辅助批判性思维（AACT）框架，利用AI对人类决策论证进行反事实分析，帮助决策者识别并改进自身推理中的缺陷；

**💡 创新点**

创新点在于将AI的反事实视角引入人类内部推理的结构化评估，实现对人类论证的量化、可操作的批判与修正，而非仅围绕AI输出进行反思；

**🔧 技术方法**

核心技术包括基于领域特定分类模型的概率推断、反事实视角切换、问题类型（不完整、不可可靠、冲突）识别、目标化自我反思、AI修正建议与数据三角化等；

**📊 数据集**

实验使用了Ames Housing数据集，经过预处理后抽取8个特征并离散化为三类房价标签；

**📈 对比分析**

与传统Recommender（推荐）和Analyzer（假设驱动XAI）以及无AI baseline比较，AACT在降低过度依赖AI方面表现最优，但整体准确率提升不显著；

**⚠️ 局限性**

局限包括实验任务为低风险房价预测、参与者为非专业受众、依赖表格特征且需要手动特征选择，且AACT在提高认知负荷和潜在低依赖方面表现不佳，需进一步在高风险或多模态任务中验证。

---

## 122. Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models

**arXiv ID:** 2602.11057 | [PDF](https://arxiv.org/pdf/2602.11057v1)

**作者:** Xinyu Yuan `[一作]` (Zhejiang University), Wenzhi Chen `[通讯]` (Zhejiang University)

**通讯引用:** 4002 | [OpenAlex ID](https://openalex.org/A5101562846)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Pram，一种利用多模态语言模型（MLM）进行多商品流（MCF）分配的分区分布式方法。

**💡 创新点**

创新点在于：① 将 MCF 任务按源节点划分为子问题，显著降低变量规模；② 采用 MLM 作为统一“智能体”，通过图像+文本提示实现跨模态推理；③ 设计基于 LoRA 与可学习上下文的轻量级多智能体强化学习框架，实现全局一致与自适应优化；④ 理论上证明在满足凸/凹性质时，Pram 可模拟梯度下降并收敛到最优。

**🔧 技术方法**

技术包括：多模态预训练语言模型（如 Qwen2.5‑VL‑7B‑Instruct），视觉编码（CLIP），文本提示与可学习上下文（reprogramming），LoRA 低秩适配，交叉注意力，基于 counterfactual 的多智能体强化学习（policy gradient）。

**📊 数据集**

使用了五个真实网络数据集（Meta DB、Meta WEB、Abilene、CERNET、GÉANT）和五个大规模合成拓扑（GtsCe、Colt、UsCarrier、Cogentco、Kdl），并采用 gravity 模型生成合成流量。

**📈 对比分析**

与 LP、POP、LP‑top、DRL、HARP、Aether 等基线进行对比；Pram 在三种 MCF 目标（最大链路利用率、总流量、并发流量）上平均仅落后 8% 左右最优解，并在大规模拓扑上比 LP 快 10‑100 倍，运行时低 1–2 个数量级；在鲁棒性实验中对链路失效或流量波动表现出 10% 以下性能下降。

**⚠️ 局限性**

主要局限：① 微调仍需大量算力，尤其是大型 MLM；② 视觉编码可能引入偏差；③ 目前仅在静态或有限动态场景验证，需进一步研究在线自适应与更大规模网络的可扩展性。

---

## 123. A Cognitive Distribution and Behavior-Consistent Framework for Black-Box Attacks on Recommender Systems

**arXiv ID:** 2602.10633 | [PDF](https://arxiv.org/pdf/2602.10633v1)

**作者:** Hongyue Zhan `[一作]`, Jizhong Han `[通讯]` (Institute of Information Engineering)

**通讯引用:** 3439 | [OpenAlex ID](https://openalex.org/A5112539471)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种双增强黑盒攻击框架，先通过认知分布对齐实现对序列推荐系统的高质量模型提取，再利用协作关系与梯度融合生成行为一致的污染序列，从而提升目标物品的推荐排名。

**💡 创新点**

创新点包括：①将认知心理学中的首要效应和位置偏差引入模型提取，使用注意衰减理论把离散排名转为连续注意分布；②设计行为一致污染生成策略，结合协作关系与梯度信息双重优化，既保证攻击效果，又提升序列语义与统计的隐蔽性。

**🔧 技术方法**

核心技术包括：认知分布对齐损失（KL + pairwise），自回归采样生成查询序列，协作矩阵构建候选池，梯度引导的投毒候选评分，双信号融合与贪心/束搜索构建投毒序列；实现时采用 NARM、SASRec、BERT4Rec 三种主流序列推荐模型作为对手与白盒模型。

**📊 数据集**

使用了三个公开数据集：Movielens‑1M、Steam 和 Amazon Beauty，以覆盖不同稀疏度和用户行为特点。

**📈 对比分析**

与现有提取基线（DFME、ME‑MIA 等）和投毒基线（RandAlter、DQN、WhiteBox SimAlter、DFME‑based 投毒）进行对比；实验表明该框架在所有模型/数据集上都取得最高的 Agr@1 / Agr@10 及更高的攻击成功率，提升幅度约 15%–30% 以上，且在稀疏场景下优势更为显著。

**⚠️ 局限性**

局限性包括：①对目标模型 API 的查询次数有限制，需一定查询预算；②依赖协作矩阵信息，若协作数据缺失会影响行为一致性；③在极度稀疏或冷启动情形下仍可能性能下降；④白盒情境下仍可被针对梯度或行为异常的检测方法发现。

---

## 124. HairWeaver: Few-Shot Photorealistic Hair Motion Synthesis with Sim-to-Real Guided Video Diffusion

**arXiv ID:** 2602.11117 | [PDF](https://arxiv.org/pdf/2602.11117v1)

**作者:** Di Chang `[一作]` (University of Southern California), Stephane Grabli `[通讯]` (Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 HairWeaver，一种基于视频 Diffusion Transformer 的框架，利用 CG 生成的物理逼真头发运动数据进行少样本训练，实现从单张图像生成带动态头发的人体动画。

**💡 创新点**

创新点在于：①用 CG 渲染得到高质量、可控的头发运动条件，弥补了真实数据稀缺问题；②采用两阶段训练策略，先在 CG 域预训练 LoRA 再冻结并学习 Pose+Hair 注入模块，避免域间伪影；③设计 Motion Control Module 在不修改预训练权重的前提下，将身体姿态和头发 UVW 条件注入 DiT。

**🔧 技术方法**

技术主要包括：视频 Diffusion Transformer（DiT）、LoRA/ControlNet 风格的运动注入模块、VAE 编码器+patchify、transformer attention、物理模拟与 Blender Cycles 渲染。

**📊 数据集**

数据集为 83 分钟 CG 生成的视频（1500 条 100 帧短片），每条包含 I_ref、V_gt、C_pose、C_hair；测试集包括自收集 CG 10 条视频和 NeRSemble 30 条真实光照视频。

**📈 对比分析**

在 CG 与 NeRSemble 数据集上与 Wan2.2、LTX-Video、UniAnimate-DiT 等基线在 PSNR、SSIM、LPIPS、FID、cd-FVD 等指标上进行对比，HairWeaver 在所有指标上均显著优于基线，并且推理速度更快；用户研究亦显示其偏好度最高。

**⚠️ 局限性**

局限性包括：当驱动姿态与参考图像差异过大时，身份与外观保持不佳；手部细节渲染仍不够精准；受限于训练数据多样性与底层 DiT 的生成能力。

---

## 125. On the Role of Consistency Between Physics and Data in Physics-Informed Neural Networks

**arXiv ID:** 2602.10611 | [PDF](https://arxiv.org/pdf/2602.10611v1)

**作者:** Nicolás Becerra-Zuniga `[一作]` (Universidad Politécnica de Madrid), Gonzalo Rubio `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 856 | [OpenAlex ID](https://openalex.org/A5026793328)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

系统评估了数据与PDE一致性对PINN精度的影响，并提出一致性障碍概念；

**💡 创新点**

首次把一致性障碍定义为数据误差与PDE残差不匹配导致的内在误差下限，并用制造解实验验证；

**🔧 技术方法**

采用物理信息神经网络、动态损失加权、AdamW+L‑BFGS优化以及一维粘性Burgers方程的制造解；

**📊 数据集**

使用不同网格分辨率（粗、中、细）求解的数值数据以及完全一致的解析数据作为训练/测试集；

**📈 对比分析**

通过RMSE、绝对误差分布和Pareto前沿对比分析，发现低保真数据下PINN精度停滞在一致性障碍水平，唯有高保真或解析数据可实现几乎无误差；

**⚠️ 局限性**

研究仅针对一维制造问题，缺乏对高维、实验噪声或参数不确定性的考察，需进一步验证与推广。

---

## 126. Roughness-Informed Federated Learning

**arXiv ID:** 2602.10595 | [PDF](https://arxiv.org/pdf/2602.10595v1)

**作者:** Mohammad Partohaghighi `[一作]` (University of California Merced), YangQuan Chen `[通讯]` (University of California Merced)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的联邦学习算法RI‑FedAvg，通过加入粗糙度指数（Roughness Index, RI）驱动的正则化项来缓解非IID环境下的客户端漂移问题。

**💡 创新点**

创新点在于将高维损失曲线的粗糙度指数作为自适应正则化权重，使局部更新在面对数据异质性时更稳健；并提供了非凸目标的收敛分析。

**🔧 技术方法**

技术包括：联邦平均（FedAvg）框架、RI的计算（随机方向投影与归一化总变差）、RI‑基正则化、SGD本地训练、以及理论上的L‑光滑性与方差界定分析。

**📊 数据集**

使用了MNIST、CIFAR‑10和CIFAR‑100三种图像分类数据集，在非IID划分（Dirichlet分布α=0.1或0.5）下进行实验。

**📈 对比分析**

与FedAvg、FedProx、FedDyn、SCAFFOLD和DP‑FedAvg等主流基线相比，RI‑FedAvg在所有数据集和多种实验配置下都获得了更高的测试准确率和更快的收敛速度，尤其在极度异质的场景中表现最优。

**⚠️ 局限性**

局限性包括：需要额外计算RI所需的方向投影和总变差，可能增加客户端算力负担；RI估计的质量受随机方向数量和离散化点数影响；在极高维模型或极大规模客户端群组中，RI计算成本可能成为瓶颈。

---

## 127. Pricing Query Complexity of Multiplicative Revenue Approximation

**arXiv ID:** 2602.10483 | [PDF](https://arxiv.org/pdf/2602.10483v1)

**作者:** Wei Tang `[一作]` (Chinese University of Hong Kong), Mengxiao Zhang `[通讯]` (University of Iowa)

**通讯引用:** 380 | [OpenAlex ID](https://openalex.org/A5101956035)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在仅通过价格查询并仅得到买家是否购买的二元反馈的场景下，如何在乘法误差（1−ε）约束下实现单买家单件商品的收益最大化。

**💡 创新点**

创新点在于：① 引入两种“规模提示”模型（one‑sample hint 与 value‑range hint），解决在无规模信息时无法近似最优价格的根本限制；② 设计统一的三分搜索式算法，利用正则分布与 MHR 分布的半凹性与单峰性；③ 给出了对上述模型下正则、MHR 与一般分布的近似收益查询复杂度上下界，并证明上界与下界在对数因子之外匹配。

**🔧 技术方法**

核心技术包括：半凹性证明与三分搜索；量化查询复杂度的量子估计（使用 Bernoulli 统计量与 KL 发散的链式分析）；构造硬实例并利用 Pinsker / KL 边界给出下界；通过单样本得到置信区间并在该区间内使用高精度估价。

**📊 数据集**

该工作仅使用合成的概率分布（正则、MHR、一般支持区间为 [1,H] 的离散/连续分布）进行理论分析和下界构造，未使用真实数据集。

**📈 对比分析**

实验或实证比较：论文通过理论证明表明，在 value‑range hint 模式下正则/ MHR 分布的查询复杂度分别为 Θ(ε⁻²·H) 与 Θ(ε⁻²)，与已知的样本复杂度匹配；在 one‑sample hint 模式下正则分布需要 Θ(ε⁻³) 次查询；这些结果与先前仅在样本访问模型中的下界（如 Θ(ε⁻³)）相同或更严谨。

**⚠️ 局限性**

局限性包括：① 对于一般分布，单样本提示不足以解决尺度问题，只能在 value‑range hint 下得到结果；② 只考虑单买家单件商品的情形，未扩展到多买家或多件商品；③ 论文主要给出理论上最优的查询复杂度，对实际实现的常数与实际性能未做实验验证。

---

## 128. The Infrastructure Equation: Water, Energy, and Community Policy for Georgia's Data Center Boom

**arXiv ID:** 2602.10526 | [PDF](https://arxiv.org/pdf/2602.10526v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 129. Interpretable Vision Transformers in Monocular Depth Estimation via SVDA

**arXiv ID:** 2602.11005 | [PDF](https://arxiv.org/pdf/2602.11005v1)

**作者:** Vasileios Arampatzakis `[一作]` (Democritus University of Thrace), Nikos Papamarkos `[通讯]` (Democritus University of Thrace)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5109991926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将SVD-Inspired Attention（SVDA）引入Dense Prediction Transformer（DPT），实现对单目深度估计的可解释性改进。

**💡 创新点**

创新点在于将注意力分解为方向编码与谱调制，使用可学习的对角谱矩阵生成可解释的注意力映射，并提出六个谱指标用于诊断注意力行为。

**🔧 技术方法**

采用SVDA机制、行归一化查询/键投影、对角谱矩阵调制以及原始DPT架构，并配合相应的评估指标进行实验。

**📊 数据集**

使用KITTI和NYU‑v2这两个公开的深度估计基准数据集进行实验验证。

**📈 对比分析**

与原始DPT基线相比，SVDA在保持甚至略优的预测精度的同时，仅增加约15%运行时间，并显著提升注意力可解释性指标。

**⚠️ 局限性**

局限性包括实现开销相对较高、对实现细节敏感以及实验仅在单目深度估计任务上进行，尚未验证在其他稠密预测任务中的普适性。

---

## 130. Less is More: The Dilution Effect in Multi-Link Wireless Sensing

**arXiv ID:** 2602.10823 | [PDF](https://arxiv.org/pdf/2602.10823v1)

**作者:** Bruno Rodrigues `[一作]` (University of St. Gallen), Karim Khamaisi `[通讯]` (University of St. Gallen)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一间住宅客厅使用9个ESP32‑C3节点生成72条WiFi链路，连续12天收集CSI数据并与摄像机标注的占用/空闲标签对齐，评估单链路与多链路融合对占用检测的影响。

**💡 创新点**

首次揭示多链路融合会因Fresnel区约束和高维特征噪声产生稀释效应，导致准确率下降；证明传感器数量不是提升精度的关键，且布局优先级超过算法复杂度。

**🔧 技术方法**

利用Fresnel区物理模型、CSI特征工程（NBVI、方差等）、TDMA时隙调度、嵌入式固件、Python后端流水线、嵌套交叉验证以及Wilcoxon、Cohen d等统计检验。

**📊 数据集**

自行构建的312小时CSI数据集（72链路）与同步摄像机获取的占用/空闲标签，已公开发布。

**📈 对比分析**

采用10折前向链式嵌套交叉验证，比较单链路、随机链路、最优链路、全链路及多种分类器；单链路AUC≈0.54，72链路≈0.49，差异显著（Cohen d=0.86）。

**⚠️ 局限性**

局限于单一住宅环境、2.4 GHz频段、固定1.4 Hz采样率，导致对静态占用检测性能低下，未涵盖多房间或更复杂布局的情况。

---

## 131. Multi-objective computational design optimization of a Total Disc Replacement implant

**arXiv ID:** 2602.10304 | [PDF](https://arxiv.org/pdf/2602.10304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 132. LakeMLB: Data Lake Machine Learning Benchmark

**arXiv ID:** 2602.10441 | [PDF](https://arxiv.org/pdf/2602.10441v1)

**作者:** Feiyu Pan `[一作]` (Shanghai Jiao Tong University), Jianhua Li `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 30565 | [OpenAlex ID](https://openalex.org/A5100336358)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了LakeMLB，面向数据湖环境下的多表机器学习基准，聚焦 Union 与 Join 两种典型表集成场景；

**💡 创新点**

首次为数据湖设计完整的多表学习基准，提供三套 Union 数据集、三套 Join 数据集、统一拆分与评价指标，并实现三种主流多表集成策略的开源实现；

**🔧 技术方法**

使用预训练（PT）、数据增强（DA）、特征增强（FA）三种集成策略，评估了传统树模型、单表 Tabular Transformer、跨表迁移学习模型（TransTab、CARTE）以及基础模型（TabPFN、TabICL）等多种技术；

**📊 数据集**

Union 场景包含 MSTraffic、NCBuilding、GACars（政府交通、建筑违章、二手车）三套数据集；Join 场景包含 NNStocks、LHStocks、DSMUSIC（股票、音乐）三套数据集；全部均为公开数据并做了时间/标签平衡拆分；

**📈 对比分析**

通过在单表基线与三种多表策略的对比，实验显示：在 Union 场景预训练策略胜出（win‑rate 83.3%，平均提升0.06%），在 Join 场景特征增强策略表现最佳（win‑rate 91.7%，平均提升2.97%）；树模型在单表基线表现最稳健；数据增强往往不利；迁移学习模型在两类场景均保持较好鲁棒性；

**⚠️ 局限性**

局限性：仅涵盖分类任务并且标签平衡；每个任务仅使用单一辅助表；数据规模有限（0.8K–40K行）；未覆盖多表、多语言、多模态、动态 schema 等更复杂、真实的数据湖场景；

---

## 133. How Much Reasoning Do Retrieval-Augmented Models Add beyond LLMs? A Benchmarking Framework for Multi-Hop Inference over Hybrid Knowledge

**arXiv ID:** 2602.10210 | [PDF](https://arxiv.org/pdf/2602.10210v1)

**作者:** Junhong Lin `[一作]` (Massachusetts Institute of Technology), Yada Zhu `[通讯]` (IBM Research)

**通讯引用:** 790 | [OpenAlex ID](https://openalex.org/A5101792548)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于混合知识（无结构文本 + 结构化知识图）的多跳推理评估框架，自动构造显式推理路径的问答对，并支持按领域与时间窗定制化。

**💡 创新点**

创新点在于自动联结无结构文本与结构化知识图，生成知识密集型问答并通过污染检测机制避免与LLM预训练数据重叠，真正评估检索与推理能力。

**🔧 技术方法**

使用检索增强生成（RAG）技术、知识图构建与维护、自动问答生成算法以及污染检测与时间框架控制等技术。

**📊 数据集**

基于最近arXiv科研文献构建的文本与知识图，涵盖人工智能、治理与政策、生物信息学三大领域。

**📈 对比分析**

与多种检索增强LLM（如ChatGPT+检索、GPT‑4、Claude）以及纯参数化模型对比实验，检索增强模型在该基准上多跳推理准确率显著提升，且不受预训练信息泄漏影响。

**⚠️ 局限性**

局限性包括对自动构建知识图质量的依赖、时间窗更新难题、对最新信息的实时刷新不足、模型仍可能受参数记忆影响以及评测通用性需进一步验证。

---

## 134. Scaling Routers with In-Package Optics and High-Bandwidth Memories

**arXiv ID:** 2602.10505 | [PDF](https://arxiv.org/pdf/2602.10505v1)

**作者:** Isaac Keslassy `[一作]` (Technion), Bill Lin `[通讯]` (UC San Diego)

**通讯引用:** 2443 | [OpenAlex ID](https://openalex.org/A5115588966)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种“router‑in‑a‑package”架构，利用Heterogeneous Integration的HBM4堆叠与包内光互连，实现单一封装内可达1.31 Pb/s的互联网核心路由器。

**💡 创新点**

创新点包括：① Split‑Parallel Switch（SPS）将光纤按空间分割，仅需一次OEO转换即可并行处理；② Parallel Frame Interleaving（PFI）将包聚合成大帧，在HBM内实现高效并行访问，并通过循环交叉bar模拟理想的输出排队共享内存交换机，保证100%吞吐率。

**🔧 技术方法**

使用的关键技术包括：Heterogeneous Integration的HBM4多堆叠、芯片组片段（chiplets）、2.5D光子学互连、伪随机纤维分配、循环交叉bar、帧分层聚合与银行交错访问、功耗优化的OEO转换。

**📊 数据集**

实验使用的基准数据集有：Abilene、GEANT背骨网络的流量矩阵（TM）、CAIDA 100 Gbps链路抓包、人工生成的交叉数据中心 AI 训练流量矩阵。

**📈 对比分析**

通过将原始TM按比例放大、不同α值、不同负载下测量包丢失率，并与细粒度负载均衡方案比较；结果显示：在Abilene/GEANT工作负载下丢失率<0.013%；在CAIDA实测流量下<0.008%；在跨DC AI工作负载下最大丢失率<0.35%；证明SPS+PFI实现与细粒度负载均衡几乎无差异，吞吐率达100%。

**⚠️ 局限性**

主要限制在于功耗与热量管理——HBM占40%功耗，处理芯片占50%；对极不均匀的流量分布敏感，需良好均衡策略；以及对未来更高速光纤（如112 Gbps）与更大规模HBM的适配仍需进一步研究。

---

## 135. A Low-Rank Defense Method for Adversarial Attack on Diffusion Models

**arXiv ID:** 2602.10319 | [PDF](https://arxiv.org/pdf/2602.10319v1)

**作者:** Jiaxuan Zhu `[一作]` (Clemson University), Siyu Huang `[通讯]` (Clemson University)

**通讯引用:** 2511 | [OpenAlex ID](https://openalex.org/A5082547392)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了低秩防御模块 LoRD 及两阶段防御管道，专门针对在 Latent Diffusion Model (LDM) 上对 LoRA 微调的 ACE/ACE+ 对抗攻击，实现对攻击样本的检测与防御，并保持在清洁样本上的高质量图像生成。

**💡 创新点**

创新点在于设计双分支低秩适配器并引入可学习的平衡参数 λ，使 LoRD 能根据输入样本自动调节防御强度；同时将 LoRD 与传统 LoRA 有效融合，构建高效可扩展的两阶段防御框架。

**🔧 技术方法**

采用的核心技术包括 LoRA 低秩适配、MLP+Sigmoid 计算平衡参数、PGD 对抗训练、LoRD 与 LoRA 权重融合、Stable Diffusion 生成模型，以及 CLIP‑IQA 与 FID 作为评估指标。

**📊 数据集**

使用的实验数据集为 CelebA‑HQ‑caption（人脸）和 Chinese_Landscape_Paintings_1k（风景）进行训练，并在 CelebA‑HQ、VGGFace2 与自采景观图像集上进行评估。

**📈 对比分析**

通过与基于 PGD‑2 的对抗训练+LoRA、ACE 攻击结果对比，LoRD 在 CLIP‑IQA 上提升约10%–20%，FID 降低约30%–50%，在防御效果与图像质量上显著优于传统对抗训练方法。

**⚠️ 局限性**

局限性包括：仅验证了对 LoRA 微调的 ACE/ACE+ 攻击；对更广泛的对抗手段或更大规模模型的泛化尚未评估；平衡参数 λ 对不同任务需要额外调参；防御性能在极端攻击强度下可能受到影响。

---

## 136. Colorful Talks with Graphs: Human-Interpretable Graph Encodings for Large Language Models

**arXiv ID:** 2602.10386 | [PDF](https://arxiv.org/pdf/2602.10386v1)

**作者:** Angelo Zangari `[一作]` (University of Illinois Chicago), Sourav Medya `[通讯]` (University of Illinois Chicago)

**通讯引用:** 458 | [OpenAlex ID](https://openalex.org/A5049055881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

本文研究如何通过人类可解释的结构编码（将 Weisfeiler–Lehman 颜色映射为自然语言颜色词）来提升大型语言模型在图结构推理任务中的性能；

**💡 创新点**

创新点在于提出有序 WL 细化的结构标记，保留图的置换不变性并生成可解释的颜色标签；同时证明该编码与距离加权连通性一致；

**🔧 技术方法**

技术包括：有序 1‑WL 细化、颜色映射、图到文本的 prompt 生成（包含上下文、颜色结构、少量示例、查询描述）以及 LLM 作为图推理求解器；

**📊 数据集**

实验数据集涵盖合成图（Erdős–Rényi、Barabási–Albert、Path）和真实图（Cora、Citeseer、PubMed、OGBN‑ArXiv），任务包括环检测、最短路、最大流、三角计数与节点分类；

**📈 对比分析**

与基线 TLG‑A/F（节点索引或角色名编码）在多种任务上对比，WL‑颜色编码显著提升全局推理任务（如最大流、最短路）的准确率，尤其在节点距离远或图规模大时优势更为明显；在三角计数上表现略逊；

**⚠️ 局限性**

局限性包括：对大型图仍受 LLM 上下文窗口限制；加入颜色编码会增大 prompt 长度，可能影响性能；在某些局部模式匹配任务（如三角计数）效果不佳；依赖预训练 LLM 对自然语言颜色词的理解，可能受模型偏差影响。

---

## 137. ERGO: Excess-Risk-Guided Optimization for High-Fidelity Monocular 3D Gaussian Splatting

**arXiv ID:** 2602.10278 | [PDF](https://arxiv.org/pdf/2602.10278v1)

**作者:** Zehua Ma `[一作]` (Shenzhen Campus of Sun Yat-sen University), Xiaodan Liang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 23047 | [OpenAlex ID](https://openalex.org/A5047878798)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于超额风险分解的自适应优化框架 ERGO，用于单图像 3D Gaussian splatting 的多视角生成。

**💡 创新点**

创新点包括：① 引入超额风险与 Bayes 误差分解以动态调节各视角和目标的权重；② 设计几何感知目标和纹理感知目标，实现全局局部自适应优化；③ 将多视角扩散模型与优化方法无缝结合，显著抑制视角不一致带来的噪声。

**🔧 技术方法**

使用 3D Gaussian splatting、score distillation sampling (SDS)、多视角扩散模型 Zero123++、可视化权重映射、梯度和 Hessian 计算来估计超额风险并更新权重。

**📊 数据集**

在 Google Scanned Objects (GSO) 与 OmniObject3D 两大公开数据集上进行评测，此外还在真实场景图像上进行验证。

**📈 对比分析**

与 SyncDreamer、Wonder3D、DreamGaussian、LRM、LGM、VideoMV、InstantMesh、SAR3D 等方法对比，ERGO 在 PSNR、SSIM、LPIPS 上均实现了最高分，提升幅度可达 3-5 分，表现出更好的几何一致性与纹理细节。

**⚠️ 局限性**

局限性：仍依赖多视角扩散模型的质量；在极端视角差异或极端光照条件下，超额风险估计可能不够精准；计算量较大，训练时间相对传统优化方法较长。

---

## 138. 1%>100%: High-Efficiency Visual Adapter with Complex Linear Projection Optimization

**arXiv ID:** 2602.10513 | [PDF](https://arxiv.org/pdf/2602.10513v1)

**作者:** Dongshuo Yin `[一作]` (Tsinghua University), Shi-Min Hu `[通讯]` (Tsinghua University)

**通讯引用:** 22511 | [OpenAlex ID](https://openalex.org/A5037233582)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了CoLin，一种低秩多分支复杂线性投影优化的视觉适配器，实现仅需约1%新参数即可高效适配视觉基础模型。

**💡 创新点**

创新点在于将投影矩阵拆分为多分支低秩矩阵并共享核与分支参数，结合正交损失与SVD初始化解决低秩矩阵梯度耦合导致的收敛慢问题。

**🔧 技术方法**

采用多分支低秩投影、核与分支共享机制、正交正则化、SVD初始化、深度学习框架MMDetection/MMSegmentation/MMRotate/MMClassification以及Swin Transformer等技术。

**📊 数据集**

在Pascal VOC、COCO、ADE20K、DOTA、STAR、Oxford 102 Flower、Oxford-IIIT Pet、VOC 2007等公开视觉数据集上进行实验。

**📈 对比分析**

与全微调、固定、BitFit、NormTuning、Partial-1以及LoRA、AdaptFormer、LoRand、Mona等基线对比，CoLin在所有任务上均超过全微调且在大多数任务上取得最高精度，仅引入约1%参数。

**⚠️ 局限性**

局限性包括对参数量和性能的依赖需要仔细调参、主要针对单模态视觉任务，跨模态或更大规模模型的适配效果尚未充分验证。

---

## 139. SQ-CBF: Signed Distance Functions for Numerically Stable Superquadric-Based Safety Filtering

**arXiv ID:** 2602.11049 | [PDF](https://arxiv.org/pdf/2602.11049v1)

**作者:** Haocheng Zhao `[一作]` (Technical University of Munich), Angela P. Schoellig `[通讯]` (University of Toronto)

**通讯引用:** 5943 | [OpenAlex ID](https://openalex.org/A5052147335)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种利用超四面体(Superquadric)表示并通过签名距离函数(SDF)实现的实时安全过滤器，用于机器人在拥挤且动态的环境中避免碰撞。

**💡 创新点**

创新点在于：①提出用SDF而非SQ隐式函数作为控制边界，解决梯度严重不良条件问题；②通过GJK/EPA+随机平滑得到SDF梯度；③将该SDF直接嵌入控制障碍函数(CBF)的二次规划，兼顾安全与任务一致性。

**🔧 技术方法**

使用的技术包括：Superquadric几何建模、GJK与EPA求距离、随机平滑(Smoothing)估计梯度、控制边界函数(CBF)与QP求解、Frankia Emika FR3机器人低层控制框架CRISP、RGB‑D点云分割与ICP配准、扩展卡尔曼滤波估计动态障碍物运动。

**📊 数据集**

数据集：主要使用仿真环境和实时采集的RGB‑D点云、运动捕捉数据；未使用公开标准数据集，所有实验均基于自建测试环境与真实场景。

**📈 对比分析**

与传统球体、圆柱盒拆分以及官方FR3碰撞模型对比，SQ模型在覆盖率与过度近似率上均优于球体和圆柱盒拆分，且保持光滑凸性；在仿真与真实任务中实现了39%平均任务完成时间提升、100%碰撞避免成功率，且在控制周期100Hz下可实时运行。

**⚠️ 局限性**

局限性：①仅适用于凸且形状指数在(0,2]的SQ；②SDF梯度估计需调参（温度ε），不当会影响精度；③大规模场景下计算仍受限于GJK/EPA与随机平滑的计算开销，尽管可多核并行；④未针对非凸障碍或高度非线性动态环境进行理论证明。

---

## 140. End-to-End Semantic ID Generation for Generative Advertisement Recommendation

**arXiv ID:** 2602.10445 | [PDF](https://arxiv.org/pdf/2602.10445v1)

**作者:** Jie Jiang `[一作]` (Tencent Inc), Jiawei Jiang `[通讯]` (Wuhan University)

**通讯引用:** 50993 | [OpenAlex ID](https://openalex.org/A5109553608)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

未提供具体内容，无法总结

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未知

**⚠️ 局限性**

未知

---

## 141. YOR: Your Own Mobile Manipulator for Generalizable Robotics

**arXiv ID:** 2602.11150 | [PDF](https://arxiv.org/pdf/2602.11150v1)

**作者:** Manan H Anjaria `[一作]` (New York University), Zichen Jeff Cui `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了一款低成本、开源的双臂移动机械臂YOR，具备全向轮驱动、伸缩式垂直升降、两支PiPER 6-DOF柔性臂以及视觉惯性SLAM导航，并通过Meta Quest 3手柄实现全身遥控及基于学习的策略演示。

**💡 创新点**

创新点在于：将全向轮驱动与伸缩式升降相结合，解决传统差速驱动的非全向约束；采用柔性关节控制降低碰撞风险；将低成本组件与开放源码硬件/软件整合；使用Meta Quest手柄进行直观全身遥控；在单机上完成从感知、地图构建、路径规划到执行的一体化系统。

**🔧 技术方法**

关键技术包括：四轮SWERVE驱动、闭环电机控制、摄像头姿态估计（ZED 2i）、视觉惯性SLAM、体素地图与A*规划、Pure Pursuit跟踪、Ruckig轨迹生成、柔性关节控制、VQ-BeT变压器策略学习、Meta Quest 3手柄输入映射。

**📊 数据集**

使用自制的遥控演示数据集：约100条30Hz遥控轨迹，包含两部iPhone手腕摄像头和ZED 2i的RGB图像及相机姿态；无公开数据集。

**📈 对比分析**

与现有低成本平台（Tidybot++、XLeRobot、Mobile-ALOHA等）对比，YOR在工作空间、垂直高度、双臂协同、全向移动以及成本（<10k$）方面具有明显优势；实验中拾取/提升成功率均为10/10，移动与导航成功率9/10，轨迹跟踪误差≤12mm，末端执行误差≤16mm。

**⚠️ 局限性**

局限性包括：仅配备6-DOF臂，易受奇异点限制；缺乏高级语义导航与外部障碍预测；在动态环境下的姿态漂移问题；对高负载或大场景的适用性有限；未来可通过升级为7-DOF臂、改进SLAM与控制算法来提升。

---

## 142. The Offline-Frontier Shift: Diagnosing Distributional Limits in Generative Multi-Objective Optimization

**arXiv ID:** 2602.11126 | [PDF](https://arxiv.org/pdf/2602.11126v1)

**作者:** Stephanie Holly `[一作]` (Johannes Kepler University Linz), Werner Zellinger `[通讯]` (Johannes Kepler University Linz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究离线多目标优化中生成模型的表现，评估其在不同指标下与进化算法的对比，并提出“离线前沿偏移”概念来解释生成方法的局限性。

**💡 创新点**

引入离线前沿偏移指标阐释生成方法受数据分布偏移限制，证明性能提升需实现离群点采样，并通过经验验证生成方法在多目标指标上的不足。

**🔧 技术方法**

采用流匹配和扩散生成模型作为生成方法，NSGA-II+代理学习作为进化方法，对离线数据进行重采样实验，利用MMD、GD+、IGD+、HV等多目标评价指标。

**📊 数据集**

使用标准离线多目标优化基准 Off‑MOO‑Bench，包括 ZDT 和 DTLZ 任务的离线数据集，并通过逐层去除非支配样本制造前沿偏移。

**📈 对比分析**

通过在 256 设计上多次随机种子，比较生成与进化方法在 HV、GD+、IGD+、MMD 等指标的表现，结果显示生成方法在 HV 上相当，但在 GD+、IGD+ 上明显落后，且 MMD 较高表明其更保守。

**⚠️ 局限性**

生成模型受离线数据分布约束，缺乏对前沿外区域的探索，导致在离线前沿偏移大时性能急剧下降，无法满足离线 MOO 对离群样本的需求。

---

## 143. Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling

**arXiv ID:** 2602.10623 | [PDF](https://arxiv.org/pdf/2602.10623v1)

**作者:** Zhibin Duan `[一作]` (Xi’an Jiaotong University), Dandan Guo `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 Bayesian Non‑Negative Reward Model (BNRM)，通过非负稀疏因子分析与 Bradley–Terry 模型相结合，对 RLHF 中的奖励模型进行概率建模，加入了不确定性估计和可解释的稀疏表示，以抑制奖励劫持。

**💡 创新点**

创新点包括：① 将全局与局部稀疏非负因子引入奖励建模，实现去偏与可解释；② 用 Weibull 分布参数化的变分推断网络实现可扩展的后验推断；③ 利用稀疏正则化自动消除长度、格式等表面特征的偏差。

**🔧 技术方法**

主要技术手段为变分推断（variational inference）配合 Weibull 分布、低秩适配（LoRA）与大模型（LLM）后端，构建 Bradley–Terry 先验并在奖励学习中实现概率稀疏因子化。

**📊 数据集**

使用的数据集包括 Unified‑Feedback (40K/400K 训练样本)、RM‑Bench、RewardBench、HHH‑Alignment、MT‑Bench、Skywork‑Preference‑v0.2 等，及用于 RLHF 的 alpaca‑gpt4‑data‑en。

**📈 对比分析**

通过与 BT 基线、BT 的多种变体（Margin、Label‑Smooth、Ensemble）、GRM、InfoRM 等对比，在 ID 与 OOD 评估中均取得更高准确率（如 74.2%/83.6%/75.2% 在 40K 训练集；在 400K 训练集上进一步提升），在低资源、噪声设置下表现尤为突出；在 RLHF 评估中，使用 BNRM 的策略在 benchmark 上分别达到约 74.98% 和 62.25% 的准确率。

**⚠️ 局限性**

局限性：① 需要额外的变分推断网络，计算与内存开销相对较大；② 对极大模型或极端偏差的鲁棒性尚未充分验证；③ 解释性依赖因子权重分布，实际语义解释仍需人工验证；④ 主要验证在特定 RLHF 场景，未覆盖所有对齐任务。

---

## 144. Simple generators of rational function fields

**arXiv ID:** 2602.10878 | [PDF](https://arxiv.org/pdf/2602.10878v1)

**作者:** Alexander Demin `[一作]` (National Research University Higher School of Economics), Gleb Pogudin `[通讯]` (Ecole Polytechnique)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种新的随机化算法，用来在给定一组生成元的子域k⊂E⊂k(x₁,…,xₙ)时，寻找一个更简单的生成元集合。

**💡 创新点**

创新点在于：①使用OMS（Ollivier‑Müller‑Quade‑Steinwandt）理想的低度系数，而不是完整 Gröbner 基；②采用稀疏插值和部分 Gröbner 基计算，仅求解低度系数，从而显著降低中间表达式膨胀；③引入一种新的基于随机化的子域成员判定算法，避免在符号域上做 Gröbner 基；④结合多种启发式筛选和最小化策略得到最终简化生成元。

**🔧 技术方法**

技术实现包括：稀疏多元多项式和有理函数插值（Ben‑Or/ Tiwari、van der Hoeven‑Lecerf）、有限域 Gröbner 基计算（F4算法与追踪技术）、随机化成员判定（利用Jacobian与OMS理想的特殊化）、多次插值求低度系数、基于重构的多维度数估计、最小化与排序启发式。

**📊 数据集**

实验使用了53个来自结构识别（ODE/离散动力学）和代数不变量等领域的实际模型生成元，包含如SEIR34、MAPK‑5、Pharm、Fujita、JAK‑STAT等。

**📈 对比分析**

与现有方法（Maple实现的OMS系数直接计算、ffmodStd、slimgb等）相比，本文方法在生成子域基时既速度快（数十倍到百倍）又得到更简洁的生成元；在大多数案例中，得到的生成元次数更少、次数或次数度更低，且算法在1小时内完成。

**⚠️ 局限性**

局限性包括：①在极大系数或高次数系数出现时，插值难度上升，导致计算失败（如Covid‑2例子）；②对高维大规模问题仍需改进追踪与并行化；③随机化方法需要足够大的有限域样本，若样本不足会影响成功概率；④对特定领域的启发式排序未必通用。

---

## 145. Can LLMs Cook Jamaican Couscous? A Study of Cultural Novelty in Recipe Generation

**arXiv ID:** 2602.10964 | [PDF](https://arxiv.org/pdf/2602.10964v1)

**作者:** F. Carichon `[一作]`, G. Farnadi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在跨文化烹饪配方生成中的表现，构建 LLMFusion 数据集并对比模型与人类的文化适配程度

**💡 创新点**

首次把基于信息论的文化差异度量与 LLM 生成文本结合，揭示 LLM 在文化创意与传统维度上缺乏真实对应关系

**🔧 技术方法**

使用多种开源 LLM（Llama‑3、Gemma、Qwen、Falcon、Orion、Phi‑4 等）在零样本设置下生成配方，并采用 Jensen‑Shannon Divergence 及其变体衡量“新颖度”“独特度”等指标

**📊 数据集**

扩展 GlobalFusion 人类配方数据，生成 LLMFusion（约 44 条每种菜式、每个国家的 LLM 生成配方）

**📈 对比分析**

通过统计 Pearson 相关、差异率、层级内表示对比等多重评估方法发现：LLM 在“新颖度”上与人类相似，但在与文化距离的相关性、传统与创意区分、内部表示与配方素材的文化根基上均显著落后；整体性能与人类相距甚远，缺乏与文化距离同步的差异

**⚠️ 局限性**

局限包括：仅使用英文配方；对 Human‑reference 依赖导致评估偏倚；GlobalFusion 采样不均衡；未充分考察多语言环境对文化适配的影响

---

## 146. ELROND: Exploring and decomposing intrinsic capabilities of diffusion models

**arXiv ID:** 2602.10216 | [PDF](https://arxiv.org/pdf/2602.10216v1)

**作者:** Paweł Skierś `[一作]` (Warsaw University of Technology), Kamil Deja `[通讯]` (Warsaw University of Technology)

**通讯引用:** 10709 | [OpenAlex ID](https://openalex.org/A5070627781)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 ELROND 框架，通过对同一文本提示在不同随机种子下生成样本的差异进行反向传播，提取梯度并在文本嵌入空间中分解得到可解释的语义方向，实现对单一概念的精细控制。

**💡 创新点**

创新点在于：①直接在条件嵌入空间而非视觉特征中进行梯度分解，②利用 PCA 与稀疏自编码器获得可解读的语义轴；③通过子空间维度估计概念复杂度，④用这些方向重构蒸馏模型的多样性以缓解模式坍塌。

**🔧 技术方法**

使用技术包括：扩散模型（SDXL、SDXL‑DMD 及 Flux）、文本嵌入反向传播、梯度采集、主成分分析 (PCA)、Top‑k 稀疏自编码器 (SAE)、DreamSim 与 CLIP 评估指标，以及 FID 评估蒸馏模型的多样性。

**📊 数据集**

数据集主要为从 SDXL 生成的 10,000 张图像（每个概念），再从中随机采样 30,000 对梯度；概念提示采用通用模板 “A picture of <concept>”。

**📈 对比分析**

与基线 SDXL‑DMD 和 SliderSpace 对比，ELROND 在 DreamSim 多样性得分上显著提升（+≈0.2-0.3），CLIP 兼容性保持不变；在蒸馏模型的 FID 评估中，使用 SAE 方向可将 FID 降至 19.7–25.5，明显优于原始蒸馏模型（>30）和 SliderSpace（>35）。

**⚠️ 局限性**

局限性包括：梯度采集过程计算成本高、在多主体场景下方向可能产生交叉干扰、部分通过 PCA 或 SAE 得到的方向解释性不足。

---

## 147. Adaptive Sampling for Private Worst-Case Group Optimization

**arXiv ID:** 2602.10820 | [PDF](https://arxiv.org/pdf/2602.10820v1)

**作者:** Max Cairney-Leeming `[一作]` (Institute of Science and Technology Austria), Christoph H. Lampert `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 12653 | [OpenAlex ID](https://openalex.org/A5068085751)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在差分隐私约束下实现最大-最小（worst‑case）组优化的算法，利用自适应采样与分组梯度裁剪来平衡不同组的隐私和学习效果。

**💡 创新点**

核心创新在于同时调节每个组的采样率和裁剪阈值，使得即使组大小不均也能获得相同的 RDP 隐私保证；并通过指数加权重更新采样比例，显著降低梯度方差，从而提升训练稳定性和最差组准确率。

**🔧 技术方法**

技术手段包括 DP‑SGD、Rényi 差分隐私 (RDP)、自适应采样、分组梯度裁剪、指数权重更新、隐私会计器（Opacus）以及对噪声和采样率的精确控制。

**📊 数据集**

在三个标准公平分类基准上验证：Unbalanced MNIST（数字 8 为少数类）、CelebA（性别与发色的四个子组）以及 Bank Fraud（年龄与欺诈与否四个子组）。

**📈 对比分析**

与普通 DP‑SGD、传统组‑DRO（不做裁剪、采样不平衡）以及两种基线变体对比。实验表明，本方法在所有数据集上显著提升了最差组准确率（WGA），平均准确率保持相近，并在少数组上保持更严格的隐私保证，整体性能优于对比算法。

**⚠️ 局限性**

局限性包括：需要手动设定采样频率 k、温度 η 与噪声比例 τ^2 等超参；在极端不平衡或极大组数时，采样与裁剪的精细调节可能复杂；实验主要针对二分类/多分类任务，跨任务或多任务设置下的可扩展性仍待进一步研究。

---

## 148. TestExplora: Benchmarking LLMs for Proactive Bug Discovery via Repository-Level Test Generation

**arXiv ID:** 2602.10471 | [PDF](https://arxiv.org/pdf/2602.10471v1)

**作者:** Steven Liu `[一作]` (Microsoft), Scarlett Li `[通讯]` (Microsoft)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TestExplora 基准，用来评估大语言模型在真实仓库级别、基于文档的主动缺陷发现能力，涵盖了测试生成、Fail‑to‑Pass 等评价指标。

**💡 创新点**

创新点在于：① 将代码文档作为唯一 oracle，真正实现主动缺陷发现；② 在仓库层面而非单文件级别进行测试；③ 采用时间感知、持续采集的数据流水线避免数据泄漏；④ 引入 agentic 探索策略显著提升性能。

**🔧 技术方法**

使用的技术包括：LLM（GPT‑5‑mini、Qwen3‑Coder‑30B、Gemini‑2.5‑pro 等）生成测试用例；DocAgent 自动生成接口文档；SWEAgent / TraeAgent 进行仓库探索；Fail‑to‑Pass、Head Pass、Entry Coverage、Change‑Focused Coverage 等评估指标。

**📊 数据集**

数据集来自 482 个 GitHub 仓库，包含 1,552 条 PR 和 2,389 个测试生成任务，平均每仓库 5,010 次星标，平均 PR 1.6 条测试、8 条调用、1.76 调用深度；持续采集实现时间感知数据流。

**📈 对比分析**

通过对 6 大主流 LLM 在 12,227 条真实 PR 上的 Head Pass、Fail‑to‑Pass、Entry Coverage、Change‑Focused Coverage 四项指标进行多模型对比。最优模型 GPT‑5‑mini 在白盒/黑盒下的 Fail‑to‑Pass 分别为 11.84%/7.67%；Agent（SWEAgent）可提升至 17.27%/29.7%。整体而言，缺陷发现成功率仍低于 20%。

**⚠️ 局限性**

限制：① 缺陷发现成功率仍偏低，证明 LLM 在主动测试上尚不成熟；② 文档质量对性能影响显著，自动生成文档仍存在误差；③ 对依赖信息的偏好不一致，过多上下文反而降低性能；④ Agent 探索虽有效，但对搜索策略、工具调用频率仍需进一步优化。

---

## 149. Normalized Surveillance in the Datafied Car: How Autonomous Vehicle Users Rationalize Privacy Trade-offs

**arXiv ID:** 2602.11026 | [PDF](https://arxiv.org/pdf/2602.11026v1)

**作者:** Yehuda Perry `[一作]` (Rutgers University), Tawfiq Ammari `[通讯]` (Rutgers University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对16名美国AV司机进行半结构化访谈，采用构造主义扎根理论和NVivo代码分析，探讨司机如何理解和合理化车辆监控与数据采集的隐私问题。

**💡 创新点**

首次将数字放纵、平台治理与监控生态相结合，阐释AV司机的隐私认知为数字放弃而非隐私恐慌，并提出治理与设计干预建议。

**🔧 技术方法**

使用构造主义扎根理论方法、NVivo文本分析软件、访谈录音转写工具（Otter AI）等质性研究技术。

**📊 数据集**

16份访谈记录（涵盖Tesla、Polestar、Chevrolet Bolt、Mitsubishi Outlander四款车型）以及相关的车辆数据访问日志（仅限加州参与者）。

**📈 对比分析**

未进行量化实验或性能对比，采用质性比较分析方法，对隐私认知模式进行归纳与解释。

**⚠️ 局限性**

样本规模小、仅限美国地区、缺乏跨文化与纵向研究、数据访问受地理限制导致信息不对称、未对比不同平台或算法的具体隐私风险。

---

## 150. Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base for Long-Context Transformers

**arXiv ID:** 2602.10959 | [PDF](https://arxiv.org/pdf/2602.10959v1)

**作者:** Feilong Liu `[一作]` `[通讯]`, Feilong Liu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析并阐明了旋转位置编码（RoPE）在长上下文 Transformer 中的可行性范围，给出了基值的理论下限（混叠与低频稳定性）与上限（浮点精度限制），并通过对 LLaMA、Mistral、DeepSeek 等主流模型的案例研究验证了这些边界。

**💡 创新点**

创新点在于将 RoPE 重新解释为相位调制的复杂振荡器银行，并通过经典信号处理理论推导出与模型深度、上下文长度和数值精度共同决定的“Goldilocks 区间”，首次给出了完整的理论可行性区域。

**🔧 技术方法**

使用了相位调制的信号处理框架、傅里叶/频率分析、误差传播与数值精度分析，并结合 Transformer 结构的深度复合效应进行理论推导。

**📊 数据集**

主要基于现有大模型（LLaMA 系列、Mistral、DeepSeek 等）以及标准长序列评测（如 Needle-in-a-Haystack、RULER 等）进行实验验证，没有引入新的公开数据集。

**📈 对比分析**

通过对比模型在实际部署时的 RoPE 基值与理论下限/上限的关系，验证了理论预测与模型性能（长距离注意力衰减、关注崩溃等）的一致性，说明理论框架能够准确预判模型的长上下文稳定性。

**⚠️ 局限性**

局限性包括：①仅关注 RoPE，未对其他位置编码方法给出类似理论；②理论推导假设相位误差相互独立，实际可能受网络权重分布影响；③未给出针对极端长上下文的硬件实现方案，仅提出理论上限。

---

## 151. Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models

**arXiv ID:** 2602.10382 | [PDF](https://arxiv.org/pdf/2602.10382v1)

**作者:** Théo Lasnier `[一作]` (Inria Paris), Djamé Seddah `[通讯]` (Inria Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对GAPperon系列模型中的语言切换后门进行机制分析，定位触发信息在前7.5%–25%层形成，并发现触发激活的注意力头与自然语言头高度重叠。

**💡 创新点**

首次揭示后门触发器并未构建独立电路，而是共用模型已有的语言编码注意力头，从而说明后门行为与内部功能组件高度耦合。

**🔧 技术方法**

采用激活补丁（activation patching）在头层与层层对比真实触发与伪触发、英文与非英文上下文，计算Jaccard指数评估头集合重叠。

**📊 数据集**

使用FineWeb‑Edu中随机抽取的1,000条英文段落，并通过Qwen3‑32B翻译成法语、德语、意大利语、西班牙语，形成多语言并行数据集。

**📈 对比分析**

通过比较真实触发与伪触发以及英文与非英文上下文的补丁效果，触发头与语言头的Jaccard指数在0.18–0.66之间，显著高于随机基线，表明触发共用语言机制并影响输出语言。

**⚠️ 局限性**

仅针对GAPperon模型的语言切换后门，未验证其他后门类型或不同架构；阈值与Jaccard仅衡量集合重叠，未证明因果必要性；仅测试拉丁脚本语言，缺乏对非拉丁脚本的验证。

---

## 152. OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL

**arXiv ID:** 2602.10687 | [PDF](https://arxiv.org/pdf/2602.10687v1)

**作者:** Jinjie Shen `[一作]` (Wuhan University), Zhun Zhong `[通讯]` (LION Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出统一的OmniVL-Guard框架，针对多模态（文本、图像、视频）伪造检测与细粒度定位问题；

**💡 创新点**

提出自演进的Chain‑of‑Thought（CoT）生成策略与自适应奖励缩放策略（ARSPO），有效解决多任务RL中的难度偏差，提升定位与分类的协同性能；

**🔧 技术方法**

结合自监督强化学习（RL）、自演进CoT、奖励重塑与动态任务权重调节技术，构建多模态大模型训练管线；

**📊 数据集**

构建全谱取证Reasoning（FSFR）数据集，整合多种公开数据集（FakeNewsCorpus、MCFEND、FakeClue、LOKI、ForgeryNet、GenVideo、DVF、SAMM、MDSM、DGM^4、NewsCLIPpings等），覆盖文本、图像、视频及图文联合任务；

**📈 对比分析**

在70万样本的内部测试集和四个零样本外域基准（ISOT、CASIA2.0、MMFakeBench、FakeSV）上进行对比，OmniVL-Guard在二分类、图像定位、文本定位、视频定位上均超越当前SOTA，尤其在细粒度定位任务上提升超过20%；

**⚠️ 局限性**

仍受制于训练成本高、RL稳定性挑战、对极端稀有伪造场景的泛化不足，以及潜在的训练数据偏差和模型可解释性有限。

---

## 153. Online Bisection with Ring Demands

**arXiv ID:** 2602.10337 | [PDF](https://arxiv.org/pdf/2602.10337v1)

**作者:** Mateusz Basiak `[一作]` (University of Wrocław), Agnieszka Tatarczuk `[通讯]` (University of Wrocław)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5092392100)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种针对环网络请求的在线二分割问题的随机化算法，允许集群大小最多为 (3/4+ε)·n，并证明其竞争比为 O(ε⁻³·log²n)

**💡 创新点**

创新点在于将问题视为具有受限状态空间的度量任务系统（MTS），通过限制切割边数到 2k（k=Θ(1/ε)）将状态空间从指数降到多项式，从而使 Bubeck 等人最近的 MTS 结果可应用；同时证明该约束仅以 O(k) 的因子增加成本

**🔧 技术方法**

使用的技术包括：全局重平衡过程（保留 α-平衡且只保留 2k 条切割边），对比最优解与算法状态的“编辑距离”Φ 的势能分析，分阶段分析与重平衡的费用，最后应用 Bubeck 等人关于 MTS 的随机化 O(log²|S|) 竞争比

**📊 数据集**

该工作为纯理论研究，无使用实验数据集，所有结果均为证明性分析

**📈 对比分析**

与目前已知的最佳结果相比（如 Rajaraman & Wasim 的 O(n·log n) 确定性算法或 Räcke 等人的多聚类 O(log³n) 算法但需要更大资源放大），本文提供了随机化且只需 3/4+ε 资源放大的竞争比；实验/数值比较未给出，但理论上显著优于现有的随机化方法

**⚠️ 局限性**

局限性包括：仍需相对较大的资源放大因子 (3/2+ε)；竞争比仍远高于最优的 O(log n) 下界，且无法适用于一般请求集合；理论分析复杂，实际实现与效率尚未评估

---

## 154. GENIUS: Generative Fluid Intelligence Evaluation Suite

**arXiv ID:** 2602.11144 | [PDF](https://arxiv.org/pdf/2602.11144v1)

**作者:** Ruichuan An `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14730 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GENIUS 评测套件，专门用于衡量统一多模态模型的生成型流体智力（GFI），并对其三大核心维度进行系统评估。

**💡 创新点**

创新点在于：①首次将流体智力拆解为隐式模式诱导、即时约束执行与上下文知识适应三大原语；②设计了 510 条专家手工编制的多模态样本与混合评估指标；③揭示模型在逻辑遵循上的缺陷，并提出训练无关的注意力校正机制，提升 GFI 性能。

**🔧 技术方法**

使用的技术包括多模态交互式上下文输入、混合评估（规则遵循、视觉一致性、美学质量）以及基于 Gemini‑3‑Pro 的 LMM 评判；并对注意力分布进行可视化与无训练的重权重调节。

**📊 数据集**

使用的数据集为 GENIUS 官方发布的 510 条样本，覆盖 20 个子任务，三维结构分别是 Implicit Pattern Induction、Ad‑hoc Constraint Execution 与 Contextual Knowledge Adaptation。

**📈 对比分析**

方法上与 12 款公开与专有模型（Nano Banana Pro、Bagel、Qwen‑Image 等）进行对比，发现即便最先进模型也仅取得约 57 分；训练无关的注意力校正可提升约 6% 的整体分数，表明模型在规则遵循上存在显著不足。

**⚠️ 局限性**

局限性包括：①仅通过注意力调节提升 GFI，未对模型内部结构进行改造；②评测仍依赖单一 LMM 评判器，可能存在主观偏差；③未覆盖更复杂的多步推理与动态场景，导致实验结果不完全泛化。

---

## 155. PuriLight: A Lightweight Shuffle and Purification Framework for Monocular Depth Estimation

**arXiv ID:** 2602.11066 | [PDF](https://arxiv.org/pdf/2602.11066v1)

**作者:** Yujie Chen `[一作]` (Hefei University of Technology), Tian Zhang `[通讯]` (Hefei University of Technology)

**通讯引用:** 58694 | [OpenAlex ID](https://openalex.org/A5100429270)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级自监督单目深度估计框架 PuriLight。

**💡 创新点**

创新点包括：Shuffle‑Dilation 卷积、Rotation‑Adaptive Kernel Attention 与 Deep Frequency Signal Purification 三大模块的组合，兼顾局部细节、层次增强与全局净化。

**🔧 技术方法**

采用稀疏卷积+通道洗牌、旋转注意力机制、频域滤波与逆 FFT 等技术实现高效特征处理。

**📊 数据集**

主要在 KITTI Eigen 分割和 Make3D 数据集上训练与评估。

**📈 对比分析**

与 R‑MSFM、Lite‑Mono、Monodepth2 等方法对比，参数仅 2.7 M，取得 Abs Rel、δ1 等指标的最优或接近最优性能，并在 Make3D 跨域测试中保持竞争力。

**⚠️ 局限性**

在高频抑制过程中，可能误删部分细纹与动态物体细节，导致图像细节缺失。

---

## 156. DMP-3DAD: Cross-Category 3D Anomaly Detection via Realistic Depth Map Projection with Few Normal Samples

**arXiv ID:** 2602.10806 | [PDF](https://arxiv.org/pdf/2602.10806v1)

**作者:** Zi Wang `[一作]` (Niigata University), Jun Yu `[通讯]` (Niigata University)

**通讯引用:** 8775 | [OpenAlex ID](https://openalex.org/A5084275232)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无训练的跨类别3D异常检测框架DMP-3DAD，利用多视角真实深度投影和冻结的CLIP视觉编码器实现对少量正常样本的异常评分；

**💡 创新点**

创新点在于：1）完全训练无关，直接使用预训练CLIP无微调；2）采用多视角真实深度投影将点云映射到CLIP友好的图像域；3）通过视角可靠性加权来提升异常判别鲁棒性；

**🔧 技术方法**

技术主要包括多视角深度投影、冻结CLIP视觉编码器、基于特征相似性的加权距离度量；

**📊 数据集**

在ShapeNetPart数据集上进行实验，使用各类物体的点云作为正常/异常样本；

**📈 对比分析**

与基于重建和知识蒸馏的分类方法对比，平均AUC-ROC在1、3、5个参考样本下分别达到92.49%、96.08%、96.44%，在所有类别上均超过对手；

**⚠️ 局限性**

局限性：对全局几何相似度高的异常难以区分；视角数量与质量对性能敏感；缺乏局部几何细节的判别能力。

---

## 157. AI Sensing and Intervention in Higher Education: Student Perceptions of Learning Impacts, Affective Responses, and Ethical Priorities

**arXiv ID:** 2602.11074 | [PDF](https://arxiv.org/pdf/2602.11074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 158. Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution

**arXiv ID:** 2602.10357 | [PDF](https://arxiv.org/pdf/2602.10357v1)

**作者:** Haixu Liao `[一作]` (New Jersey Institute of Technology), Shuai Zhang `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 9276 | [OpenAlex ID](https://openalex.org/A5017928828)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在数据不平衡条件下，Transformer‑MLP 模型通过对比学习学习特征表示的训练动态；

**💡 创新点**

首次给出三阶段训练过程的理论解析，量化少数特征对神经元分化的影响，并证明基于幅值剪枝能显著提升少数特征的学习；

**🔧 技术方法**

使用对比学习（InfoNCE）+Transformer-MLP 架构、BReLU 激活、幅值剪枝 + 反向不剪枝更新；

**📊 数据集**

实验使用长尾版本的 CIFAR‑10、CIFAR‑100、ImageNet 数据集；

**📈 对比分析**

与无剪枝对比，采用线性探针评估；结果显示剪枝在所有数据集上提升整体准确率并缩小 head‑tail 准确率差距，尤其在极端不平衡（ρ=100）时提升显著；

**⚠️ 局限性**

仅针对幅值剪枝比例与策略的细粒度影响缺乏深入分析，且理论仅适用于简化的 Transformer‑MLP 结构，难以直接推广至更复杂模型。

---

## 159. From Interaction to Demonstration Quality in Virtual Reality: Effects of Interaction Modality and Visual Representation on Everyday Tasks

**arXiv ID:** 2602.10618 | [PDF](https://arxiv.org/pdf/2602.10618v1)

**作者:** Robin Beierling `[一作]` (Bielefeld University), Anna-Lisa Vollmer `[通讯]` (Bielefeld University)

**通讯引用:** 705 | [OpenAlex ID](https://openalex.org/A5044783914)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

比较三种VR输入配置（手部动作捕捉手套、手部可视化控制器、仅控制器可视化）在虚拟厨房日常任务中的用户体验与任务执行表现。

**💡 创新点**

创新点在于提出语义轨迹分段分析方法，并将轨迹相似度（Frechet/DTW）与语义错误（Levenshtein）结合，用以评估不同输入方式在目标导向与方式导向任务中的效率与自然性；同时为机器人学习与康复等应用给出设计建议。

**🔧 技术方法**

技术包括：Unreal Engine 4.27实现的虚拟厨房，Manus手套和Valve Index控制器；NEEM格式轨迹记录；SUS、NASA‑TLX主观评估；轨迹分割、Frechet/DTW距离、Levenshtein距离；非参数统计检验（Kruskal‑Wallis、Conover、Cliff’s Delta）。

**📊 数据集**

数据集：60名参与者在虚拟厨房完成桌面布置、洗碗、切面包、清洁、倒汁、指向与评分等任务，记录完整的物体与手部轨迹，未使用公开数据集。

**📈 对比分析**

比较方法：先用SUS与NASA‑TLX评估主观体验；再用客观指标（任务时间、空闲时间、把握次数、轨迹相似度DTW/DFD、Levenshtein距离）统计检验。结果显示：控制器（尤其是仅可视化）在目标导向任务中更快、更一致；手套在方式导向任务中更自然、语义更准确；指向准确度在手套下最低。

**⚠️ 局限性**

局限性：样本量与任务设计可能导致部分指标无显著差异；轨迹相似度指标本身并非完备；未进行长期适应性或多次使用的评估；手套手势识别可能出现误抓或误放；缺乏触觉反馈，可能影响真实感与错误率。

---

## 160. Invisible Trails? An Identity Alignment Scheme based on Online Tracking

**arXiv ID:** 2602.10626 | [PDF](https://arxiv.org/pdf/2602.10626v1)

**作者:** Ruisheng Shi `[一作]` (Beijing University of Posts and Telecommunications), Jiaqi Zeng `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5102725679)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于在线追踪的身份对齐方案，设计数据收集器、身份对齐算法，并实现被动与主动去匿名化攻击。

**💡 创新点**

首次引入主动诱导式攻击以提升识别精度，并构建了在线追踪身份对齐的评估框架。

**🔧 技术方法**

利用时间窗口与时间匹配粒度、追踪器生成的伪匿名行为数据与公开社交网络数据的离散化匹配；实现爬虫、数据生成器、哈希查找等算法。

**📊 数据集**

使用公开的 Twitter 与新浪微博帖子时间戳数据（约10,655条用户记录、40,000+帖子），并通过自研算法生成对应的追踪器匿名数据进行实验。

**📈 对比分析**

与传统基于用户名、网络结构或行为相似度的方法对比，采用 IASR、ASSR、AIUP 三指标。实验显示，当时间匹配粒度足够时 IASR 超过 90%，AIUP 达到 91%，相较于已有方法显著提升。

**⚠️ 局限性**

受时间偏差、浏览/发布行为区分、低活跃度用户、跨平台同步误差以及法律隐私限制等因素影响，精度随用户活跃度下降；方法在非合法授权环境下也存在伦理与合规风险。

---

## 161. Let Leaders Play Games: Improving Timing in Leader-based Consensus

**arXiv ID:** 2602.11147 | [PDF](https://arxiv.org/pdf/2602.11147v1)

**作者:** Rasheed M `[一作]` (International Institute of Information Technology Hyderabad), Sujit Gujar `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种双提议者的区块链共识机制，用以抑制时间游戏，迫使提议者尽早发布区块。

**💡 创新点**

创新点在于通过在每个 slot 选取两名提议者并引入随机化确认与基于接收顺序的奖励分配，构建了一个 Latency Game，使得最优策略为不延迟提议，显著降低了时间游戏的激励。

**🔧 技术方法**

采用博弈论模型（Nash equilibrium 分析）、概率传播模型、随机化奖励政策，并利用区块传输延迟分布进行理论推导。

**📊 数据集**

使用以太坊主网区块 21720000-21750648 的传播时延与区块价值数据，模拟并估算奖励增值函数。

**📈 对比分析**

通过对比传统单提议者协议和所提双提议者机制的理论收益及延迟敏感性，证明在同质与异质网络下，双提议者协议的 Nash equilibrium 几乎不允许延迟，性能提升在公平性与吞吐量上均有正向影响。

**⚠️ 局限性**

局限在于对网络延迟分布的假设（近似 Gamma/对称），以及对协作攻击的分析仅限于两名提议者，实际实现需考虑链上随机性、合约成本及跨协议兼容性。

---

## 162. R2RAG-Flood: A reasoning-reinforced training-free retrieval augmentation generation framework for flood damage nowcasting

**arXiv ID:** 2602.10312 | [PDF](https://arxiv.org/pdf/2602.10312v1)

**作者:** Lipai Huang `[一作]` (Texas A&M University), Ali Mostafavi `[通讯]` (Texas A&M University)

**通讯引用:** 6976 | [OpenAlex ID](https://openalex.org/A5023165780)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个基于推理强化、检索增强生成（R2RAG）的框架，用以在无任务特定训练的前提下，对洪水后的财产损失程度（PDE）进行即时预测并生成可解释的推理过程。

**💡 创新点**

核心创新包括：①利用特征分布差异分析自动生成推理知识库，形成可检索的推理轨迹；②通过邻域检索和自由样本（原型和边界案例）实现上下文增强；③引入基于规则的降级机制校正过度严重预测；④在多种大型语言模型（LLM）上实现训练‑free 推理与解释的统一。

**🔧 技术方法**

技术手段主要是：表格到文本生成、基于JS/KS差异的特征排序、检索增强生成（RAG）框架、LLM推理轨迹生成（DeepSeek‑R1风格）、降级规则实现、与多种LLM后端（GPT‑4o、Llama‑3.1、Qwen‑3、DeepSeek‑R1等）集成。

**📊 数据集**

使用了2017年哈里斯县飓风 Harvey 的洪水索赔数据与公开地理信息（建筑、人口、地形、降雨等）构建的 500 m × 500 m 网格特征集，共 10,764 条训练样本和 4,614 条测试样本，并通过归一化索赔金额划分三层 PDE 标签。

**📈 对比分析**

与传统监督基线 FloodDamageCast*（GBDT）在同一数据集、相同标签划分下进行对比。R2RAG‑Flood 在整体准确率约 0.66–0.67、严重度分数 0.78–0.82 方面与基线相近；在高损失召回和成本效能（severity‑per‑cost）上，轻量级 LLM 版（如 Llama‑3.1‑70B）表现更佳，尤其在每样本成本与准确率比值上明显优于基线与大型 LLM。

**⚠️ 局限性**

局限性包括：仅在哈里斯县一地区、一次洪水事件进行评估，难以泛化至其他地区或灾害类型；使用固定 LLM 后端和解码参数，未探讨温度等设置的鲁棒性；推理质量评估基于自动化指标，可能忽略专家判定的细微差别；成本对比仅基于评估时的费用，未覆盖长期运维和模型更新成本。

---

## 163. FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight

**arXiv ID:** 2602.11136 | [PDF](https://arxiv.org/pdf/2602.11136v1)

**作者:** Jiayi Zhou `[一作]` (Institute for Artificial Intelligence, Peking University), Jie Fu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 Neuro‑Symbolic 监督框架 FormalJudge，通过 LLM 先把自然语言意图拆解为原子事实，再使用 Dafny + Z3 进行形式化验证，确保代理行为满足安全与约束。

**💡 创新点**

创新点在于：① 采用双向 Formal‑of‑Thought 架构，将 LLM 的不确定性限定在单一事实提取；② 将逻辑组合交给确定性 SMT 求解器，消除概率监督的“幻觉回声”问题；③ 通过形式化反馈实现弱模型对强模型的欺骗检测和迭代式安全改进。

**🔧 技术方法**

技术手段包括：大型语言模型（Qwen‑2.5 系列、Claude‑4.5‑Opus 等）作为规范编译器；Dafny 语言与 Boogie 生成 Z3 SMT 代码进行形式验证；上下文映射与原子事实提取的分层设计；实验中还使用 Python 代码验证对比。

**📊 数据集**

数据集涵盖三大基准：Agent‑SafetyBench（工具使用与行为安全），VitaBench（跨领域约束遵从），Deceivers（代理欺骗检测），每个基准包含数千条轨迹与多级欺骗等级。

**📈 对比分析**

与 LLM‑as‑a‑Judge（包括 CoT、Few‑Shot、Structured、Python 等变体）对比，FormalJudge 在所有代理模型与基准上平均提升 16.6% 的检测准确率；在弱对强欺骗检测中 7B 评判器可超过 90% 的准确率；在迭代修正中，Dafny 反馈使安全率从 70% 提升至 99.8%。

**⚠️ 局限性**

局限性主要来源于 LLM 在意图拆解与事实提取阶段的错误，导致形式化验证的输入不完全正确；此外，FormalJudge 并未实现 100% 的准确率，对高难度自然语言转换仍有挑战。

---

## 164. Cross-Sectional Asset Retrieval via Future-Aligned Soft Contrastive Learning

**arXiv ID:** 2602.10711 | [PDF](https://arxiv.org/pdf/2602.10711v1)

**作者:** Hyeongmin Lee `[一作]` (Seoul National University of Science and Technology), Yongjae Lee `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 6133 | [OpenAlex ID](https://openalex.org/A5100366478)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出未来对齐软对比学习（FASCL）框架，使资产检索结果与未来收益相关性对齐。

**💡 创新点**

创新点在于将未来日收益相关系数作为连续软标签构造软对比损失，并引入直接衡量未来行为一致性的评估指标。

**🔧 技术方法**

使用Transformer编码器、软对比损失、温度缩放softmax、KL散度优化，并在同周期检索场景下进行评估。

**📊 数据集**

在4,229只美国股票（NASDAQ+NYSE）上实验，包含6个OHLCV特征，训练集2010-2022，验证/测试2023-2024。

**📈 对比分析**

与13类基线（统计、时序自监督、预测、金融专用、基础模型）在同周期检索协议下对比，FASCL在FRC@K、TC@K、IC@K等未来行为指标上均优于最优基线，特别在FRC@K上提升12%。

**⚠️ 局限性**

软标签过度关注高度相关对，未显式区分无相关或负相关资产；并且仅针对同周期检索，跨时段检索与对冲应用尚未覆盖。

---

## 165. LocoVLM: Grounding Vision and Language for Adapting Versatile Legged Locomotion Policies

**arXiv ID:** 2602.10399 | [PDF](https://arxiv.org/pdf/2602.10399v1)

**作者:** I Made Aswin Nahrendra `[一作]` (Korea Advanced Institute of Science and Technology), Hyun Myung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 5833 | [OpenAlex ID](https://openalex.org/A5059521863)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

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

## 166. TabICLv2: A better, faster, scalable, and open tabular foundation model

**arXiv ID:** 2602.11139 | [PDF](https://arxiv.org/pdf/2602.11139v1)

**作者:** Jingang Qu `[一作]` (Inria), Marine Le Morvan `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的 tabular foundation model（类似 TabICL），通过改进预训练流程、架构设计和合成数据生成，提供更高准确性、更快推理速度和更好的大规模适配性。

**💡 创新点**

创新点包括：① 设计了更高多样性的 synthetic data 生成引擎；② 引入 Query‑Aware Scalable Softmax (QASSMax) 解决注意力衰减，提升长序列泛化；③ 采用 Muon 优化器与门控机制提升训练效率；④ 重复特征分组、目标感知嵌入、量化回归等细节改进；⑤ 完全开源代码、模型权重与预训练流程。

**🔧 技术方法**

技术手段主要是基于 Transformer 的 PFN 架构，结合 Set Transformer、行/列注意力、可扩展 softmax、MuOn 优化器、量化回归、混合基数多分类等。

**📊 数据集**

使用了大规模 synthetic 数据集作为预训练源，并在 TabArena（51 个数据集）和 TALENT（300 个数据集）上进行评估，涵盖二分类、多分类、回归以及多类分类等任务。

**📈 对比分析**

与 RealTabPFN‑2.5、TabPFNv2、TabICL、CatBoost、XGBoost 等基线进行无调参对比，结果显示在 TabArena 与 TALENT 上可实现超过 RealTabPFN‑2.5 的性能，同时推理速度显著提升（H100 上 10.6×，CPU 上 11.8×），并能自然处理百万级样本。

**⚠️ 局限性**

局限性包括：未能利用列名语义或文本特征；对百万级以上样本仍有挑战；缺乏多输出回归、分布漂移、缺失值处理等扩展；需要进一步的微调和 fine‑tuning 研究。

---

## 167. The Alignment Bottleneck in Decomposition-Based Claim Verification

**arXiv ID:** 2602.10380 | [PDF](https://arxiv.org/pdf/2602.10380v1)

**作者:** Mahmud Elahi Akhter `[一作]` (Queen Mary University of London), Maria Liakata `[通讯]` (Queen Mary University of London)

**通讯引用:** 7948 | [OpenAlex ID](https://openalex.org/A5007426895)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出在复杂主张验证中采用结构化子主张分解，并探究其对验证效果的影响。

**💡 创新点**

创新点在于引入了具有人类标注子主张证据跨度的真实世界数据，并系统分析了证据对齐和子主张标签噪声对分解效果的关键作用。

**🔧 技术方法**

技术包括LLM（Qwen3‑14B）推理、图神经网络（GNN）结构化推理、以及基于CHEF的编码器，配合自定义提示和层次化证据对齐方案。

**📊 数据集**

使用了PHEMEPlus、MMM‑Fact、COVID‑Fact等多领域、时限约束的社交媒体与事实检查数据集。

**📈 对比分析**

与单一主张验证基线相比，子主张对齐（SAE）场景可显著提升宏观F1（+6.3%）和准确率；而重复主张级证据（SRE）场景则无提升或下降；在标签噪声下，性能下降幅度依赖于子主张模型的标签偏差。

**⚠️ 局限性**

主要局限包括闭环证据设置、时间窗口假设、子主张与主张标签空间不匹配，以及对特定LLM模型和提示的依赖，缺乏端到端检索与实时更新能力。

---

## 168. Chatting with Images for Introspective Visual Thinking

**arXiv ID:** 2602.11073 | [PDF](https://arxiv.org/pdf/2602.11073v1)

**作者:** Junfei Wu `[一作]`, Tienie Tan `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的框架“与图像对话”，通过语言引导的特征调制来重新构建视觉操作，解决了现有视觉-语言模型在多图像和视频空间推理中的信息损失问题。

**💡 创新点**

创新点在于将视觉操作重新定义为语言引导的特征调制，增强了语言推理与视觉状态更新之间的紧密耦合。

**🔧 技术方法**

使用了一种动态视觉编码器，结合了监督微调和强化学习的两阶段训练策略，以促进有效的推理行为。

**📊 数据集**

在八个基准数据集上进行了广泛的实验，特别是在复杂的多图像和视频空间推理任务上表现出显著的改进。

**📈 对比分析**

与现有方法相比，ViLaVT在五个基准上达到了最新的状态，尤其在多图像和视频推理任务上表现突出，显示出更强的通用性和整体性能。

**⚠️ 局限性**

限制在于模型的复杂性和对计算资源的需求，可能在处理长视频和多跳推理时面临挑战。

---

## 169. PRISM: Parallel Residual Iterative Sequence Model

**arXiv ID:** 2602.10796 | [PDF](https://arxiv.org/pdf/2602.10796v1)

**作者:** Jie Jiang `[一作]` (Tencent), Zhouchen Lin `[通讯]` (Peking University)

**通讯引用:** 26023 | [OpenAlex ID](https://openalex.org/A5016399094)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种并行残差迭代序列模型 PRISM，用于高效处理超长序列的生成任务；

**💡 创新点**

创新点在于引入写-忘记解耦、输入锚定循环展开以及高秩（Rank‑L）注入机制，将深度迭代优化的表达力压缩为可并行的前馈算子；

**🔧 技术方法**

采用写-忘记解耦策略、短卷积锚定、学习预测器、GELU非线性以及 Flash‑Linear‑Attention 体系实现硬件友好的并行递推；

**📊 数据集**

主要在推荐系统基准上评测，使用 Amazon‑Books、Amazon‑Movies、Amazon‑Elecs、Yelp 四个数据集；

**📈 对比分析**

与传统线性注意力、状态空间模型、DeltaNet、TTT、Transformer 等基线对比，PRISM 在准确率（NDCG、Hit、AUC）上与显式优化模型持平甚至略优，同时训练吞吐量提升 170‑180 倍；

**⚠️ 局限性**

局限在于模型在极端小维度或极低资源场景下仍受限于写入更新的精度，对输入锚定近似的依赖可能在某些任务中导致性能下降。

---

## 170. A Dual-Stream Physics-Augmented Unsupervised Architecture for Runtime Embedded Vehicle Health Monitoring

**arXiv ID:** 2602.10432 | [PDF](https://arxiv.org/pdf/2602.10432v1)

**作者:** Enzo Nicolas Spotorno `[一作]`, Antonio Augusto Medeiros Frohlich `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出双流架构，将无监督异常检测与宏观物理代理融合，实现车辆实时健康监测。

**💡 创新点**

突破无监督模型对持续高负载的盲区，通过低频物理代理补偿，实现多维可解释健康向量。

**🔧 技术方法**

采用LSTM Autoencoder、物理代理计算（悬挂冲击、侧向应力、驱动工作、制动能量）、RISC‑V嵌入式推理与max‑pooling融合。

**📊 数据集**

使用CARLA仿真生成的重型车辆数据集，共192,857个10 Hz窗口，覆盖不同负载与路面条件。

**📈 对比分析**

与完全监督XGBoost基准对比，双流在高负载场景下检测准确率显著提升，推理能耗仅增加1.3%，总能耗约0.68 μJ/窗口。

**⚠️ 局限性**

局限性包括仅基于仿真数据、低频采样忽略高频微振动、物理参数假设与实时估计不完整。

---

## 171. HII-DPO: Eliminate Hallucination via Accurate Hallucination-Inducing Counterfactual Images

**arXiv ID:** 2602.10425 | [PDF](https://arxiv.org/pdf/2602.10425v1)

**作者:** Yilin Yang `[一作]` (University of Houston), Chengming Zhang `[通讯]` (University of Houston)

**通讯引用:** 713 | [OpenAlex ID](https://openalex.org/A5100691052)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出基于Hallucination‑Inducing Images（HII）的对比偏好优化框架 HII‑DPO，旨在通过精确合成“场景‑条件幻觉”样本，显式削弱视觉语言模型的语言偏见，提升视觉真实性。

**💡 创新点**

创新点在于：①利用GroundingDINO对目标对象进行迭代遮挡，自动生成模型专属的高质量HII；②构建Masked‑Object‑Hallucination（MOH）基准，系统评估场景条件下的幻觉模式；③在HII基础上构造细粒度首尾一致的对比偏好样本，改进DPO以精准对齐视觉与文本。

**🔧 技术方法**

核心技术包括：多模态视觉语言模型（如LLaVA‑1.5）、GroundingDINO目标检测、对比视觉‑语言对齐（VCA）、Direct Preference Optimization（DPO）以及基于同义词表的实体抽取。

**📊 数据集**

使用的数据集为COCO2014（用于MOH基准）、Visual Genome（用于HII生成与偏好数据），并在多项标准幻觉基准（Object Hallucination、Amber、HallusionBench）与通用VQA基准（ScienceQA、TextVQA、VQAv2、MM‑Vet）上进行评测。

**📈 对比分析**

与SOTA方法（Re‑Align、SENTINEL、HA‑DPO、mDPO等）对比，HII‑DPO在MOH基准上幻觉率平均下降约41.6%（判别）/73.4%（生成），在主流幻觉基准上最高可提升38%且在通用VQA任务上保持竞争力。

**⚠️ 局限性**

局限性包括：①对物体类别仅限于80个COCO标签，可能忽略更细粒度或少量类别；②在某些VQA任务中，过度抑制语言先验可能导致模型过于保守，产生冗余或缺失信息；③需要额外的检测与生成循环，计算成本相对较高。

---

## 172. Motion Capture is Not the Target Domain: Scaling Synthetic Data for Learning Motion Representations

**arXiv ID:** 2602.11064 | [PDF](https://arxiv.org/pdf/2602.11064v1)

**作者:** Firas Darwish `[一作]` (University of Oxford), Hang Yuan `[通讯]` (University of Oxford)

**通讯引用:** 38853 | [OpenAlex ID](https://openalex.org/A5100403087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了利用合成全身运动数据进行预训练，提升可穿戴设备下人类活动识别（HAR）模型的迁移性能。

**💡 创新点**

创新点在于：①构建了从文本到运动再到IMU信号的合成数据生成框架；②系统评估了不同规模、不同来源（真实与合成）预训练对18个HAR数据集的0/多-shot效果；③揭示了运动捕捉与可穿戴传感器域间的不匹配导致的模拟到现实障碍。

**🔧 技术方法**

技术方法包括：文本驱动的运动生成模型（MDM、T2M、MotionDiffuse）；逆运动学将捕捉姿态转为IMU信号；基于UniMTS的对比预训练框架；随机旋转与关节遮蔽的数据增强；CLIP ViT-B/32文本编码器。

**📊 数据集**

使用的数据集为：HumanML3D（24,661条文本–运动对）作为真实数据；对应的合成数据通过三种文本-运动模型生成；下游18个多样化的HAR数据集用于0/多-shot评估。

**📈 对比分析**

比较方法：在相同预训练样本数（24,660对）下，分别使用真实、纯合成、50/50混合数据预训练，并在多shot（0-10 shot）上评估宏平均F1。结果显示：①混合预训练可与纯真实相媲美，且在高shot场景提升明显；②仅合成预训练在足够规模（8×）下可在fine-tuned场景超越真实基线，但0-shot仍落后；③真实数据规模放大对fine-tuned的提升有限，且在0-shot不稳定。

**⚠️ 局限性**

局限性：①合成数据仍基于已有文本描述，缺乏完全离谱的多样性；②运动捕捉与可穿戴IMU信号之间的域差异削弱了预训练效果；③合成数据生成的随机性导致性能非单调；④实验主要聚焦于文本-运动映射，未探索更广泛的条件生成方式。

---

## 173. dnaHNet: A Scalable and Hierarchical Foundation Model for Genomic Sequence Learning

**arXiv ID:** 2602.10603 | [PDF](https://arxiv.org/pdf/2602.10603v1)

**作者:** Arnav Shah `[一作]` (University of Toronto), Albert Gu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2974 | [OpenAlex ID](https://openalex.org/A5025386668)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 dnaHNet，一种无词表的自回归模型，通过可微分的分块机制对基因组序列进行端到端的动态分段与建模。

**💡 创新点**

创新点在于将动态分块与递归层次结构结合，使模型在保持生物学连贯性的同时显著压缩序列，获得比固定子词分词或单核苷酸模型更高的效率与可解释性。

**🔧 技术方法**

使用了 H‑Net 的可微分分块、Mamba 与 Transformer 层、递归层次压缩、归一化与损失正则化等技术。

**📊 数据集**

在大规模原核基因组数据集 GTDB 以及 MaveDB、DEG、B. subtilis 注释数据上进行预训练与评估。

**📈 对比分析**

与 StripedHyena2 与优化的 Transformer++ 进行对比，dnaHNet 在推理 FLOPs 上可比相同规模模型快 3×，在零样本蛋白变异效应预测与基因必需性分类任务上表现更好，规模效能指数更高。

**⚠️ 局限性**

局限在于仅预训练于原核基因组，未覆盖真核多样性；对超大模型的行为未知；固定压缩比例可能不适用于非编码或真核序列；以及未验证微调性能。

---

## 174. MalMoE: Mixture-of-Experts Enhanced Encrypted Malicious Traffic Detection Under Graph Drift

**arXiv ID:** 2602.10157 | [PDF](https://arxiv.org/pdf/2602.10157v1)

**作者:** Yunpeng Tan `[一作]` (Peking University), Xinggong Zhang `[通讯]` (Peking University)

**通讯引用:** 8292 | [OpenAlex ID](https://openalex.org/A5001396675)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种名为 MalMoE 的混合专家模型，用于在存在时间图漂移的情况下对加密网络流量进行实时恶意流检测。

**💡 创新点**

创新点在于：①利用不同节点特征（平均流量与节点度）分别对不同图漂移具有固有鲁棒性的专家；②采用 Mixture‑of‑Experts 框架并重新设计门控为“硬”选择，输入加入图表示以识别近 OOD ；③引入两阶段训练和漂移驱动的数据增强，保证门控学习稳定且可解释。

**🔧 技术方法**

使用的技术包括 1‑hop GNN‑样专家网络、Mixture‑of‑Experts（hard‑selection）门控、漂移增强（流量扰动 + 边随机丢弃）以及两阶段监督训练；模型实现基于 PyTorch/DGL。

**📊 数据集**

在四个公开流量数据集（CIC‑IDS2018、ToN‑IoT、UNSW‑NB15、BoT‑IoT）、合成数据集以及真实 backbone 网络流量上进行评估；效率测试使用 MAWI 数据集。

**📈 对比分析**

与 NetBeacon、E‑GraphSAGE、HyperVision、NetVigil 等基线对比，MalMoE 在无漂移和有漂移场景下均取得显著提升（ACC 最高提升 24%，F1 最高提升 31%），并实现 858,646 flows/s 的实时检测吞吐量。

**⚠️ 局限性**

局限性包括：当前仅结合两种节点特征，对更复杂漂移场景的鲁棒性仍有限；门控在专家预测相同的样本上不做选择；模型需要在路由器端进行图构建，仍存在构建与内存瓶颈；扩展到更多专家或更深的 GNN 仍需进一步研究。

---

## 175. Division of Labor and Collaboration Between Parents in Family Education

**arXiv ID:** 2602.10501 | [PDF](https://arxiv.org/pdf/2602.10501v1)

**作者:** Ziyi Wang `[一作]` (Beijing University of Civil Engineering and Architecture), Haining Zhang `[通讯]` (Nankai University)

**通讯引用:** 1154 | [OpenAlex ID](https://openalex.org/A5100664371)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对中国城市地区小学一年级至三年级家长的家庭作业辅导劳动分工进行深度访谈与定性分析，提出以情感与认知劳动为核心的父母三人制互动模型，并给出 AI 辅助关系维护的设计方向。

**💡 创新点**

创新点在于：①将情感、认知与物理三维劳动视角融入家庭作业辅导研究；②揭示父母-子女三人互动的动态循环机制；③提出 AI 设计从任务自动化转向关系支持、协同注释、叙事时间线与共同反思三大方向，强调情感与认知劳动的可视化与协商。

**🔧 技术方法**

研究主要使用定性方法：半结构式访谈、主题分析（Braun & Clarke 六阶段），以及对现有 AI 辅助工具（如照片识别作业、语音助手、聊天机器人）功能的使用情况与需求归纳；未实现具体 AI 系统原型。

**📊 数据集**

数据集为：133份问卷筛选后 18 位家长的访谈原稿（包含 12 名母亲、6 名父亲），并记录其使用的家庭作业 AI 工具种类与使用频率。

**📈 对比分析**

由于未开发或评估具体 AI 原型，研究没有对比实验或性能指标，研究结果仅为主题发现与设计建议。

**⚠️ 局限性**

局限性包括：①样本仅限城市中国中低收入家庭，缺乏多样性；②仅使用访谈与问卷，未进行现场观察或纵向跟踪，可能忽略即时情境的细节；③研究者主观性与文化背景可能影响编码与解释；④未实现 AI 系统，无法验证设计方案的实际效果。

---

## 176. Investigating the Effects of Eco-Friendly Service Options on Rebound Behavior in Ride-Hailing

**arXiv ID:** 2602.10237 | [PDF](https://arxiv.org/pdf/2602.10237v1)

**作者:** Albin Zeqiri `[一作]` (Ulm University), Enrico Rukzio `[通讯]` (Ulm University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

在一项在线对照实验中，研究者让 75 名参与者在不同距离和五种 EFSO 设计下，选择步行或打车，以考察环保服务选项对冲击效应的影响。

**💡 创新点**

首次系统评估了环保服务选项与多种生态反馈机制如何共同导致在共享出行中的直接回弹行为，并揭示仅凭绿色叶标识即可显著提升打车率。

**🔧 技术方法**

采用基于 Python Flask 的 Web 原型呈现场景，利用 logistic mixed‑effects 模型分析选择概率，并结合主题分析挖掘定性反馈。

**📊 数据集**

使用模拟的 0.5–2.0 英里距离数据（共10个距离点），并通过 OpenRouteService 生成对应步行和驾驶路线；参与者来自 Prolific 的 75 人样本。

**📈 对比分析**

通过与无 EFSO 基线对比并计算 odds ratio，发现 Minimal 设计相较基线的选择概率提升至 OR≈1.94，且在 1.05–2.0 英里区间内显著提高；其余生态反馈未见显著差异。

**⚠️ 局限性**

实验仅在受控在线情境中进行，简化了真实决策流程，且样本仅来自美国，无法直接推广至多元文化和真实交通场景。

---

## 177. A Human-Centric Framework for Data Attribution in Large Language Models

**arXiv ID:** 2602.10995 | [PDF](https://arxiv.org/pdf/2602.10995v1)

**作者:** Amelie Wührl `[一作]` (IT University of Copenhagen), Anna Rogers `[通讯]` (IT University of Copenhagen)

**通讯引用:** 1609 | [OpenAlex ID](https://openalex.org/A5039256697)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一个以利益相关者协商为核心的人本数据归因框架，旨在为大型语言模型（LLM）生态中的创作者、用户和中介方提供可操作的归因方案。

**💡 创新点**

创新点在于将归因任务嵌入多方协商过程，提供一套可供实践的技术与治理对齐标准，突破传统单一技术归因的局限。

**🔧 技术方法**

技术涵盖相似度归因（文本相似度、检索增强生成）、因果影响归因（Shapley值、影响函数、模型重训练）以及训练数据使用归因（训练数据公开与许可、侵权检测）。

**📊 数据集**

论文未使用具体公开数据集，而是讨论了可能的训练集、检索库、公开语料库等典型数据来源，并假设可通过合作获得完整训练数据。

**📈 对比分析**

方法以框架设计和案例推演为主，缺乏系统实验和性能指标，作者提出未来可通过精确度、召回率、偏差评估等指标来检验归因方法。

**⚠️ 局限性**

局限包括：技术实现细节不充分，缺少可行性实验；跨学科协商机制和法律合规路径仍待完善；对实际部署的经济与社会影响评估不足。

---

## 178. EvoCodeBench: A Human-Performance Benchmark for Self-Evolving LLM-Driven Coding Systems

**arXiv ID:** 2602.10171 | [PDF](https://arxiv.org/pdf/2602.10171v1)

**作者:** Wentao Zhang `[一作]` (Nanyang Technological University), Zhe Zhao `[通讯]` (Stanford University)

**通讯引用:** 918 | [OpenAlex ID](https://openalex.org/A5101755455)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EvoCodeBench，一个多语言执行基准，用于评估 LLM 驱动的编码系统在推理时的自我演化、效率与人类对照性能。

**💡 创新点**

创新点在于将推理时自我演化与功能正确性、资源使用和人类基准统一量化，强调轨迹级评估并支持长尾语言的多语言测评。

**🔧 技术方法**

采用 LeetCode 在线评测接口，构建轻量级编码代理与自演化代理，利用执行反馈实现多轮改进，并收集运行时、内存、通过率等指标。

**📊 数据集**

使用从 LeetCode 抽取的 3,822 道算法题，其中挑选 100 道近期题目，覆盖 Python、C++、Java、Go、Kotlin 等多语言模板。

**📈 对比分析**

通过 Pass Rate、TLE/MLE/CE 等错误计数、平均运行时/内存、与人类提交的 Beats 比值进行对比；顶尖模型在高资源语言上超过 90% 通过率，长尾语言下降；自演化代理提升 10‑27% 的通过率并显著降低 TLE/CE，且在多语言上保持相对人类的效率优势。

**⚠️ 局限性**

局限性包括仅基于 LeetCode 题库，可能存在训练数据泄漏；评测侧重单一编译/执行环境，无法覆盖更复杂的软件工程任务；长尾语言仍表现不足，缺乏针对性多语言训练。

---

## 179. AurigaNet: A Real-Time Multi-Task Network for Enhanced Urban Driving Perception

**arXiv ID:** 2602.10660 | [PDF](https://arxiv.org/pdf/2602.10660v1)

**作者:** Kiarash Ghasemzadeh `[一作]` (University of Alberta), Sedigheh Dehghani `[通讯]` (Shahid Beheshti University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 AurigaNet 多任务网络，实现车道线、可行驶区域实例分割和目标检测的统一端到端处理。

**💡 创新点**

创新点在于将判别损失和可变形卷积与均值漂移聚类结合，实现无后处理的实例分割，并在低成本设备上实现实时推理。

**🔧 技术方法**

采用 CSPDarknet+SPPF+FPN 的共享编码器，三任务头（检测、车道、实例分割），结合判别损失、可变形卷积和均值漂移聚类，使用 PyTorch 实现。

**📊 数据集**

在 BDD100K 数据集上进行训练与评估。

**📈 对比分析**

与 YOLOP、HybridNets 等模型对比，IoU 达到 85.2%、60.8%，mAP 47.6%，在 Jetson Orin NX 上实现 5.08 FPS，整体性能优于竞争者。

**⚠️ 局限性**

仍受限于极端天气、遮挡等情况的鲁棒性不足，以及实例聚类在极大实例数时的效率下降。

---

## 180. When Skills Lie: Hidden-Comment Injection in LLM Agents

**arXiv ID:** 2602.10498 | [PDF](https://arxiv.org/pdf/2602.10498v1)

**作者:** Qianli Wang `[一作]` (Shandong University), Yue Zhang `[通讯]` (Shandong University)

**通讯引用:** 36260 | [OpenAlex ID](https://openalex.org/A5038484265)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了隐藏 HTML 注释注入在 LLM 技能文档中的安全风险，并通过实验验证其能够诱导模型产生敏感工具调用

**💡 创新点**

提出了隐藏注释注入作为一种新的攻击手段，并证明了简短的防御系统提示能有效阻止此类攻击

**🔧 技术方法**

使用了 LLM 触发式提示注入技术、HTML 渲染可见性缺口分析以及系统提示策略来构造和检测攻击

**📊 数据集**

实验基于 DeepSeek‑V3.2 与 GLM‑4.5‑Air 两个模型，并利用一条“请格式化我的代码”这一普通用户请求进行评测

**📈 对比分析**

对比未防御和防御两种设置，未防御时攻击成功率为 100%，防御后攻击成功率降为 0%，且模型仍能完成预期的代码格式化任务

**⚠️ 局限性**

仅针对两款模型、单一攻击场景与简化的技能结构，未评估更大规模或多样化工具集合的通用性，防御提示可能需要针对不同 LLM 调整

---

## 181. From Classical to Topological Neural Networks Under Uncertainty

**arXiv ID:** 2602.10266 | [PDF](https://arxiv.org/pdf/2602.10266v1)

**作者:** Sarah Harkins Dayton `[一作]` (University of Tennessee), Vasileios Maroulas `[通讯]` (U.S. Army DEVCOM Army Research Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过将神经网络、贝叶斯方法、拓扑数据分析（TDA）与拓扑深度学习结合，构建面向军用场景的多模态 AI 系统，涵盖图像、时间序列与图数据的识别、预测与异常检测。

**💡 创新点**

创新点在于：① 将持久同调与神经网络直接嵌入训练过程（TCNN、BTCNN、simplicial、cell‑complex 与 sheaf 网络）；② 结合贝叶斯推理实现不确定性量化与决策；③ 设计基于 Wasserstein 距离的持久图聚类与统计模型，实现可解释且鲁棒的拓扑特征学习；④ 在军用任务（雷达识别、图像融合、脑电信号、结构网络）上验证方法优越性。

**🔧 技术方法**

技术手段包括：CNN、RNN（LSTM、GRU）、Transformer、图卷积网络（GCN、GraphSAGE、GAT）、贝叶斯神经网络（BBB）、拓扑卷积层（圆/克莱因滤波器）、贝叶斯拓扑 CNN、simplicial 与 cell‑complex GNN、sheaf 与贝叶斯 sheaf NN、Topological Functional Units、持久图向量化、Wasserstein 距离、Fréchet 均值与贝叶斯 PPP 模型、等。

**📊 数据集**

数据集涵盖：红外军用车辆图像、雷达-光学融合数据、EEG 与 fMRI 时间序列、海洋漂浮物轨迹、社会网络与引用网络、合成图与真实图、机械/生物结构网络（actin filament、蛋白质结构）等多种军工与科研数据。

**📈 对比分析**

与传统 CNN、RNN、GAT 等基线比较，实验表明：TCNN 在数据稀缺场景下比 CNN 提升 3–5% 准确率；BTCNN 既保持准确率又显著提升不确定性校准；注意力‑GRU 在雷达发射器识别中达到 93%+ 的高精度；Wasserstein 持久图聚类在周期性与混沌信号识别上比 DTW、Wavelet 低 20% 错误率；贝叶斯 PPP 模型在类判定上相较传统 SVM 减少 15% 假阳性。

**⚠️ 局限性**

局限性包括：① 计算量大，特别是高维拓扑特征和贝叶斯采样；② 对小样本或极噪声数据仍易过拟合；③ Fréchet 均值在持久图空间非唯一导致聚类不稳定；④ 大规模图网络仍面临邻居爆炸与内存瓶颈；⑤ 需要手工调参（滤波器大小、贝叶斯先验等），缺乏统一理论统一度量。

---

## 182. Reverse-Engineering Model Editing on Language Models

**arXiv ID:** 2602.10134 | [PDF](https://arxiv.org/pdf/2602.10134v1)

**作者:** Zhiyu Sun `[一作]` (Shanghai Qi Zhi Institute), Tianxing He `[通讯]` (Tsinghua University)

**通讯引用:** 1072 | [OpenAlex ID](https://openalex.org/A5051747323)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对定位-编辑模型的参数更新泄露问题，提出逆向工程攻击 KSTER，以谱分析和熵减方法恢复被编辑主体和提示。

**💡 创新点**

创新点在于利用更新矩阵行空间的低秩结构揭示编辑主体指纹，并设计子空间伪装防御来混淆该指纹，首次从代数角度量化编辑安全风险。

**🔧 技术方法**

使用线性代数（Woodbury、奇异值分解）、熵度量、谱分析等技术，以及可逆编辑算法如 MEMIT、AlphaEdit 等。

**📊 数据集**

在 GPT‑J、Llama3‑8B‑Instruct 与 Qwen‑2.5‑7B‑Instruct 上，用 CounterFact 与 zsRE 两大事实编辑基准进行实验。

**📈 对比分析**

与灰盒基线对比，白盒攻击在 10–100 条编辑中主量召回率均超过 0.94，提示重建 top‑20 召回率可达 0.99；防御在 α≈5 时实现 80% 防护同时保持 90% 编辑效果。

**⚠️ 局限性**

局限包括需要预先构造主体与提示候选池、仅针对单次编辑、对连续多次编辑的聚合更新难以解码，以及防御在长期持续编辑下可能累积性能下降。

---

## 183. DeepImageSearch: Benchmarking Multimodal Agents for Context-Aware Image Retrieval in Visual Histories

**arXiv ID:** 2602.10809 | [PDF](https://arxiv.org/pdf/2602.10809v1)

**作者:** Chenlong Deng `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 3927 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 DeepImageSearch 这一新范式，将图像检索转化为在视觉历史上进行自主探索的代理任务，并基于此构建了 DISBench benchmark 与 ImageSeeker 基线框架。

**💡 创新点**

创新点包括：①从传统的独立语义匹配转向语料级上下文推理；②提出人机协同的上下文挖掘与查询合成流水线；③设计了适用于视觉历史探索的多工具与双层记忆机制；④首次提供了跨事件、跨时间段的多步推理基准。

**🔧 技术方法**

主要技术包括：视觉-语言模型用于图像解析与实体抽取；检索‑验证管道与时空记忆图构建；结构化提示实现代理规划；工具集（ImageSearch、GetMetadata、FilterMetadata、ViewPhotos、WebSearch）与显式/压缩记忆机制。

**📊 数据集**

使用的数据集为 DISBench，基于 YFCC100M 公开数据集整理而来，包含 57 位用户的 109,467 张照片、122 个文本查询，涵盖 intra‑event 与 inter‑event 两类推理任务。

**📈 对比分析**

通过在 ImageSeeker 框架下评估 GPT‑4o、Gemini、Claude、Qwen、GLM 等多模态代理模型，采用 Exact Match 与 F1 评价指标；最佳模型的 EM 为 28.7，F1 为 55.0；传统单步检索模型性能极低，证明需要多步推理。

**⚠️ 局限性**

局限性主要体现在：模型在长时序探索与跨事件关联上仍易失效；推理路径选择不稳，容易提前终止或丢失约束；记忆与规划机制虽有效但仍需改进；DISBench 数据规模有限，且需要人工验证，限制了数据量与多样性。

---

## 184. VERA: Identifying and Leveraging Visual Evidence Retrieval Heads in Long-Context Understanding

**arXiv ID:** 2602.10146 | [PDF](https://arxiv.org/pdf/2602.10146v1)

**作者:** Rongcan Pei `[一作]` (Tongji University), Qi Zhu `[通讯]` (Amazon Web Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对多模态模型的注意力机制进行细粒度分析，发现并定义了稀疏、动态的视觉证据检索头（VER heads），并基于其在推理时的不确定性峰值设计了训练‑free 的 VER‑RAG 推理框架，能够将模型关注的视觉补丁显式转化为文本增强上下文；

**💡 创新点**

创新点在于首次揭示了与 OCR 头截然不同的、对长上下文推理至关重要的动态检索头，并将对这些头的关注信息通过 entropy‑triggered 的显式口述方式直接融入模型推理，无需额外训练；

**🔧 技术方法**

主要技术包括基于图像渲染的视觉检索评分（VER score）计算、对高 entropy 步骤的注意力聚焦、验证集上的 head‑selection 与 patch‑extraction、以及将检索到的视觉信息转写为文本注释的 prompt 设计；

**📊 数据集**

实验使用了五个长上下文问答数据集：DocMath、Qasper、HotpotQA、Musique 与 LongBench Pro；

**📈 对比分析**

在与 OCR‑RAG、Random‑RAG、Embedding‑RAG、ColPali‑RAG、Glyph 等基线及原始 VLM 进行对比时，VER‑RAG 在 Qwen3‑VL‑8B‑Instruct 上平均提升 21.3%、在 GLM‑4.1V‑Thinking 上提升 20.1%，整体相对提升超过 20%，在多数据集上均保持显著优势；

**⚠️ 局限性**

局限性包括：需要手动构造带有视觉补丁标签的开发集来挑选 VER 头；方法仅在能够生成 VER 头的 VLM 上有效，尚未验证对其他模型的适用性；此外，因将图像从文本渲染，实际视觉文本压缩场景下的性能仍需进一步验证。

---

## 185. Privacy by Voice: Modeling Youth Privacy-Protective Behavior in Smart Voice Assistants

**arXiv ID:** 2602.10142 | [PDF](https://arxiv.org/pdf/2602.10142v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 186. The emergence of numerical representations in communicating artificial agents

**arXiv ID:** 2602.10996 | [PDF](https://arxiv.org/pdf/2602.10996v1)

**作者:** Daniela Mihai `[一作]` (University of Southampton), Francesca Franzon `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 179 | [OpenAlex ID](https://openalex.org/A5045137998)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

训练两个神经网络对话代理，让它们在参照游戏中自发产生数字表示；

**💡 创新点**

证明仅靠沟通压力可产生精确但非可组合的数字代码，揭示人类数词的可组合性与泛化需要额外约束；

**🔧 技术方法**

使用 ViT 预训练编码器结合 LSTM（离散通道）或绘图端+ViT（连续草图），并采用多分类 hinge loss 训练；

**📊 数据集**

自制 256x256 像素黑点图像，包含 1–5（或 1–20）个点，面积限制在 5–10%；

**📈 对比分析**

与随机/无通信基线比较；在训练集上达 90–100% 任务准确率，连续通道略低，但在未见数值上准确率显著下降；

**⚠️ 局限性**

缺乏可组合性与泛化；频率不影响代码简洁；未考虑生产/感知成本导致缺少压缩效应。

---

## 187. Information Abstraction for Data Transmission Networks based on Large Language Models

**arXiv ID:** 2602.11022 | [PDF](https://arxiv.org/pdf/2602.11022v1)

**作者:** Haoyuan Zhu `[一作]` (University of Sheffield), Jie Zhang `[通讯]` (Ranplan Wireless Network Design Ltd)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现一种信息抽象度（DIA）度量，用于衡量表示的压缩程度与语义保留，并将其应用于基于大语言模型的语义视频传输，显著降低传输量。

**💡 创新点**

创新点包括：①将压缩率和语义保持度整合为可计算的DIA公式；②证明DIA与信息瓶颈（IB）在共享潜在空间下一致；③利用LLM驱动的OPRO优化框架进行黑盒配置搜索；④设计VSDS模块捕捉时空语义差异。

**🔧 技术方法**

技术手段涵盖信息理论（KL散度、熵估计）、CLIP共享潜在空间、Qwen‑VL‑Max和sora‑video2‑landscape生成模型、LLM（OPRO）调优、VSDS热图生成与注意力引导。

**📊 数据集**

实验数据集为20个固定分辨率1280×704的视频，按两大类（人类动作、物体/场景动态）划分，覆盖多种语义变化。

**📈 对比分析**

通过SSIM、VMAF和DIA三项指标比较四种策略；OPRO(DIA)和VSDS‑OPRO在压缩率与语义质量上均优于基线，DIA随迭代单调提升，最终实现约99.75%传输量下降且无感知损失。

**⚠️ 局限性**

局限性包括：数据集规模有限且类别单一；评价依赖参考指标，缺乏用户主观评估；LLM驱动的OPRO计算成本高；DIA仍需可微分实现以支持端到端训练；CLIP潜在空间的语义覆盖度有限。

---

## 188. Execution-Centric Characterization of FP8 Matrix Cores, Asynchronous Execution, and Structured Sparsity on AMD MI300A

**arXiv ID:** 2602.10262 | [PDF](https://arxiv.org/pdf/2602.10262v1)

**作者:** Aaron Jarmusch `[一作]` (University of Delaware), Sunita Chandrasekaran `[通讯]` (University of Delaware)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5009614578)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

对 AMD MI300A APU 的 FP8 矩阵核心、异步计算引擎（ACE）以及 2:4 稀疏性进行基于微基准的执行层面特征化，并将结果映射到典型的 Transformer、并行与混合精度内核上；

**💡 创新点**

首次系统性测评 FP8 在不同占用率下的吞吐阈值、ACE 并发公平性与稀疏性突破点，并揭示稀疏性在单核运行无效但在多核竞争时可提升吞吐和公平性的“反直觉”现象；

**🔧 技术方法**

采用 HIP/C++ 微基准，利用 CDNA3 MFMA 指令、ACE 量化工具和 rocSPARSE 计数器；通过 L2、LDS 失效率、波前占用率、指令延迟等硬件指标来评估执行特性；

**📊 数据集**

使用标准 Transformer‑style 训练/推理矩阵尺寸（256³–8192³）、多精度（FP32/FP16/FP8）以及 2:4 稀疏编码的通用矩阵乘法；

**📈 对比分析**

对比 MI300A 与同类 AMD/NVIDIA GPU 在 FP8 归一化吞吐、ACE 并发加速比与公平度、稀疏性加速比；实验显示 FP8 在 256+ 波前占用下可达 13.7% 近峰值吞吐；ACE 在 8 条流时可实现 2.8× 加速但公平度降至 0.016；稀疏性在单核时 1.0×，在 4 条流时提升至 1.3×且公平度提高 7%；

**⚠️ 局限性**

仅覆盖单核 GCD，未考察跨 GCD 或多节点；稀疏性优势高度依赖软件开销，未提供硬件层面加速实现；实验基于静态批量，缺乏动态工作负载自适应预测；

---

## 189. Towards Autonomous Mathematics Research

**arXiv ID:** 2602.10177 | [PDF](https://arxiv.org/pdf/2602.10177v1)

**作者:** Tony Feng `[一作]` (Google DeepMind, University of California Berkeley), Thang Luong `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种端到端的自然语言数学研究代理，能够生成、验证并修订数学推理，展示了在IMO、FutureMath以及Bloom的Erdős数据库等多项任务中的高准确率，并通过该代理实现了多篇AI全自动或协同撰写的研究论文；

**💡 创新点**

创新点在于将生成、验证、修订三子代理结合，并引入推理时尺度扩展律与工具使用，形成了“深度思考”架构，使AI在研究级数学问题上具备可验证性能，并首次系统性记录了AI全自动生成论文与人类协作的实践；

**🔧 技术方法**

核心技术包括：升级版Gemini Deep Think模型、推理时尺度扩展律、生成-验证-修订子代理框架、网络搜索与网页浏览工具、Python算子、非正式自然语言验证以及专门的自我验证机制；

**📊 数据集**

主要数据集为IMO-ProofBench Advanced（30道IMO级题）、FutureMath Basic（PhD级练习题）、Bloom’s Erdős Conjectures数据库（700开放题）以及内部的FutureMath Basic和Eigenweights等参考文献；

**📈 对比分析**

评估方法为人类专家手动评分，并与Gemini Deep Think IMO Gold版做对比；性能方面，IMO-ProofBench Advanced 95.1%准确率（相较此前的65.7%），FutureMath Basic 82%条件准确率，Erdős问题中21%命中率但仅6.5%意义正确；

**⚠️ 局限性**

局限性包括高错误率和幻觉倾向、对长推理链的可靠性不足、对专业知识的浅薄理解、对数据稀缺问题的依赖、缺乏完全无误验证机制，以及需要人类专家对结果进行评估与归属确认。

---

## 190. Exploring Semantic Labeling Strategies for Third-Party Cybersecurity Risk Assessment Questionnaires

**arXiv ID:** 2602.10149 | [PDF](https://arxiv.org/pdf/2602.10149v1)

**作者:** Ali Nour Eldin `[一作]` (Telecom SudParis), Walid Gaaloul `[通讯]` (Telecom SudParis)

**通讯引用:** 3507 | [OpenAlex ID](https://openalex.org/A5084851123)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨利用语义标签对第三方风险评估（TPRA）安全问题进行组织与检索的方法，比较直接使用大型语言模型（LLM）进行问题级标签与一种混合半监督语义标注（SSSL）管道的效果。

**💡 创新点**

创新点在于提出SSSL管道：先在嵌入空间聚类问题，使用少量代表性样本由LLM标注，再通过k‑近邻传播标签，从而在保持标签区分度和一致性的同时显著降低LLM使用量和成本；并展示在标签空间检索能提升检索对齐度。

**🔧 技术方法**

技术包括：LLM直接标签、嵌入式聚类、k‑近邻标签传播、基于标签空间的检索、与传统基于问题相似度的检索对比。

**📊 数据集**

使用TPRA问题大规模仓库（涵盖ISO/IEC 27001、NIST等标准的安全与合规问题），在此基础上挑选少量样本进行标注。

**📈 对比分析**

通过实验对比，标签空间检索在检索准确率上优于直接问题相似度检索，且SSSL相较于全LLM标注显著降低了标注成本，提升了可扩展性。

**⚠️ 局限性**

局限性包括：标签的区分度和一致性对检索效果影响大；若初始聚类或标签传播出现错误，误标签可能扩散；对标签覆盖度有限的领域或语言环境适用性需进一步验证。

---

## 191. Fungal systems for security and resilience

**arXiv ID:** 2602.10543 | [PDF](https://arxiv.org/pdf/2602.10543v1)

**作者:** Andrew Adamatzky `[一作]` `[通讯]` (University of the West of England Bristol), Andrew Adamatzky (University of the West of England Bristol)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过综合生物学、电生理学和材料科学的视角，提出并阐述了以菌丝网络为基础的生物混合系统在安全、韧性与基础设施保护中的应用框架，主张将菌丝网络同时用作分布式传感介质、自主修复材料和低可探测异常检测层；

**💡 创新点**

创新点在于将菌丝网络的分散控制、内嵌记忆与自修复特性统一为一种多功能安全平台，突破传统数字传感器的集中式设计，提供一种在极端、资源受限环境下持续、低能耗、难以被探测或攻击的“生物前沿过滤”与“自记录”机制；

**🔧 技术方法**

核心技术包括：菌丝网络的电生理信号捕捉与分析、菌丝基复合材料的自愈与可变力学性能、低功耗电极接口、以及将生物信号映射为可解释的异常指示；

**📊 数据集**

本文未使用传统机器学习数据集，而是汇总了多种实验研究（机械扰动、化学刺激、温湿变化等）中记录的菌丝电位与波形数据，作为构建该框架的基础；

**📈 对比分析**

通过与传统传感网络、主动监测与 AI 辅助分析系统的对比，展示菌丝系统在损伤后持续功能、渐进式降解、低可探测性以及长周期信息整合方面的优势；实验与案例表明，尽管响应速度慢，但在长期监测、灾害后恢复和基础设施抗干扰等情境下，表现出更高的韧性和自适应性；

**⚠️ 局限性**

主要限制包括菌丝生长速度慢、对温湿和营养条件高度敏感、易受极端环境抑制；缺乏实时高精度量测，难以用于需要即时响应的安全应用；此外，生物安全与生态风险需在实际部署前通过严格的物种选择和封闭式材料工艺来规避。

---

## 192. Lie Group Variational Integrator for the Geometrically Exact Rod with Circular Cross-Section Incorporating Cross-Sectional Deformation

**arXiv ID:** 2602.10963 | [PDF](https://arxiv.org/pdf/2602.10963v1)

**作者:** Srishti Siddharth `[一作]`, Ravi N. Banavar `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

本文提供了一套全新的 LaTeX 文档类 elsarticle.cls，用于 Elsevier 期刊的稿件排版，包含预印本、最终版、双栏等多种模式；

**💡 创新点**

创新点在于彻底重写了老旧的 elsart.cls，保持与主核一致，减少与其他宏包冲突，同时提供更丰富的排版选项和易用的前置信息设置；

**🔧 技术方法**

采用了 LaTeX 宏包技术，集成 natbib、geometry、fleqn、graphicx、txfonts、hyperref 等常用宏包，并提供了自定义定理、列表、交叉引用等宏；

**📊 数据集**

本文不涉及具体数据集，而是面向作者提供排版与文档结构的工具；

**📈 对比分析**

与原版 elsart.cls 的对比显示，本类在兼容性、排版效果和功能扩展上都有提升，但未给出量化性能指标；

**⚠️ 局限性**

局限性包括仅适用于 Elsevier 期刊，不兼容某些特殊的自定义宏包或排版需求，且需要作者手动检查多栏排版下的公式断行问题。

---

## 193. Discovering Differences in Strategic Behavior Between Humans and LLMs

**arXiv ID:** 2602.10324 | [PDF](https://arxiv.org/pdf/2602.10324v1)

**作者:** Caroline Wang `[一作]`, Pablo Samuel Castro `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文利用 AlphaEvolve 对人类和前沿 LLM 在迭代石头剪刀布中的行为进行可解释模型挖掘。

**💡 创新点**

创新点在于首次将自动程序搜索技术应用于行为博弈论，发现 LLM 与人类在对手建模维度上的结构差异。

**🔧 技术方法**

使用 AlphaEvolve（基于 LLM 代码生成的进化优化）结合基线 Nash、CS-EWA 和 RNN 进行模型学习。

**📊 数据集**

数据集包括 411 名人类参与者的 IRPS 数据和 4 种 LLM（Gemini 2.5 Flash/Pro、GPT 5.1、GPT OSS 120B）在 15 个机器人对手下的 90,000 轮游戏记录。

**📈 对比分析**

通过对战绩、交叉验证似然和 AlphaEvolve 程序的 Pareto 前沿进行比较，发现前沿 LLM 的胜率显著高于人类且 AlphaEvolve 模型在拟合度上优于 CS‑EWA、接近 RNN，且能更好解释对手建模差异。

**⚠️ 局限性**

局限性包括仅关注平均人类行为、未考虑个体差异，未探讨对 LLM 进行人类化对齐的方式，且仅在 IRPS 一种简单游戏中验证，缺乏对更广泛情境的泛化评估。

---

## 194. The Role of Learning in Attacking Intrusion Detection Systems

**arXiv ID:** 2602.10299 | [PDF](https://arxiv.org/pdf/2602.10299v1)

**作者:** Kyle Domico `[一作]` (University of Wisconsin-Madison), Patrick McDaniel `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种轻量级对抗代理，通过强化学习在离线训练后实现对基于NetFlow的机器学习入侵检测系统的高效规避。

**💡 创新点**

将对抗攻击迁移到离线强化学习代理，避免每流梯度优化，显著降低计算开销与内存占用，同时在黑盒场景下仍能保持高攻击成功率。

**🔧 技术方法**

采用离线强化学习（PPO、A2C、SAC、TD3）训练多层感知机策略，在NetFlow特征空间进行增量扰动（字节、包、延迟），并使用代理模型进行奖励回馈。

**📊 数据集**

在四个公开NetFlow数据集（Bot‑IoT、CSE‑CIC‑IDS2018、UNSWNB‑15、ToN‑IoT）和多种ML模型（LR、MLP、RF、XGBoost）上进行实验。

**📈 对比分析**

与传统基于梯度的白盒攻击（PGD）以及黑盒查询优化（HSJA）比较，轻量代理在攻击成功率最高达48.9%，延迟仅5.7 ms，内存占用0.52 MB，吞吐率比PGD高10倍。

**⚠️ 局限性**

只针对基于NetFlow的NIDS，无法处理深度包检测；受限于可加扰动范围（仅入流增量），对基于协议特征的攻击（如恶意Shellcode）难以规避；并假设训练数据足够代表目标分布。

---

## 195. Chain-of-Look Spatial Reasoning for Dense Surgical Instrument Counting

**arXiv ID:** 2602.11024 | [PDF](https://arxiv.org/pdf/2602.11024v1)

**作者:** Rishikesh Bhyri `[一作]` (State University of New York at Buffalo), Peter C W Kim `[通讯]` (State University of New York at Buffalo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Chain-of-Look Spatial Reasoning (CoLSR) 框架，用视觉链引导顺序计数，解决高密度手术器械计数难题。

**💡 创新点**

创新点：① 通过视觉链模拟人类顺序计数；② 引入邻接损失使模型学习物体的空间顺序与距离约束；③ 采用可学习的类别专属提示（CSL）提升细粒度检测。

**🔧 技术方法**

技术：基于 Swin Transformer 的图像编码器、BERT 文本编码器、跨模态解码器；CountGD 视觉链生成器；邻接损失、后处理、prompt tuning。

**📊 数据集**

数据集：SurgCount-HD，1,464 张高密度手术器械图像，包含手柄位置信息。

**📈 对比分析**

与 4 种 SOTA 计数方法（CountGD、DQ-DETR、CrowdDiff、REC）以及两大多模态 LLM（Qwen-2.5-VL、GPT‑5）对比，CoLSR 在 MAE 0.88、RMSE 1.27 方面明显优于所有基线，提升幅度可达 80%+。

**⚠️ 局限性**

局限性：对极端遮挡或间隙仍易失效；依赖视觉示例和提示；仅在单视角图像上训练，缺乏多视角/深度信息的鲁棒性。

---

## 196. Tuning the burn-in phase in training recurrent neural networks improves their performance

**arXiv ID:** 2602.10911 | [PDF](https://arxiv.org/pdf/2602.10911v1)

**作者:** Julian D. Schiller `[一作]` (Leibniz University Hannover), Matthias A. Müller `[通讯]` (Leibniz University Hannover)

**通讯引用:** 6418 | [OpenAlex ID](https://openalex.org/A5057092836)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在使用截断BPTT（TBPTT）训练循环神经网络时引入burn‑in阶段（忽略初始过渡的若干步）的方法，并提供了理论误差（regret）上界和实验验证；

**💡 创新点**

创新点在于将burn‑in视为可调超参数，利用最佳控制和turnpike理论给出对训练损失和性能损失的指数衰减上界，阐明了burn‑in长度与网络记忆衰减率之间的权衡，并给出了实际调参准则；

**🔧 技术方法**

主要技术包括：截断BPTT与mini‑batch SGD、增量输出稳定性假设、最佳控制/turnpike理论推导、MSE评价指标；

**📊 数据集**

实验使用了合成系统数据以及公开系统识别数据集（Silver‑Box、Wiener‑Hammerstein、RLC）和时间序列预测数据集（Electricity、Traffic、Solar‑Energy）；

**📈 对比分析**

与标准TBPTT、stateful TBPTT和完整BPTT进行了对比；实验结果表明适当调节burn‑in可使训练和测试MSE降低多达60%，并在多数情况下接近BPTT性能，同时训练时间更短；

**⚠️ 局限性**

局限性包括：理论假设增量输出稳定性和其他技术条件相对保守，适用范围主要是中小规模子序列；对大型数据集无法直接求解基准；未考虑stateful训练和其他更复杂的网络架构。

---

## 197. DRAMPyML: A Formal Description of DRAM Protocols with Timed Petri Nets

**arXiv ID:** 2602.10654 | [PDF](https://arxiv.org/pdf/2602.10654v1)

**作者:** Derek Christ `[一作]` (JMU Würzburg), Matthias Jung `[通讯]` (JMU Würzburg)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于Python和时序Petri网的DRAMPyML框架，用以正式描述DRAM协议、生成可执行模型、验证控制器以及自动生成代码与SVA属性。

**💡 创新点**

创新点在于用Python实现高度灵活的Petri网模型，支持SameBank、Per2Bank等复杂时序约束；模型可直接执行；并可从模型自动生成符合协议的驱动代码与验证属性。

**🔧 技术方法**

采用Python语言、Rustworkx图形库构建Petri网；引入时序弧、抑制弧、重置弧；实现动态时序约束生成；利用模型进行代码生成和SVA属性生成。

**📊 数据集**

主要使用DDR3/DDR4等JEDEC标准的命令表和时序参数作为输入数据集，用于构建和验证模型。

**📈 对比分析**

通过展开Petri网生成可达状态和k深度合法命令序列，对比不同标准的Petri网；实验显示k_min=3时有368条合法命令序列，算法运行时间随k指数增长。

**⚠️ 局限性**

局限性在于状态爆炸问题仍然严重；对极大规模层次结构（如DDR5、HBM）的可扩展性有限；缺少专门的DSL接口，模型编写仍需手工编码；目前仅验证DDR3/DDR4，尚未覆盖最新标准。

---

## 198. Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence

**arXiv ID:** 2602.10354 | [PDF](https://arxiv.org/pdf/2602.10354v1)

**作者:** Mashrekur Rahman `[一作]` `[通讯]` (Dartmouth), Mashrekur Rahman (Dartmouth)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对Google AlphaEarth卫星基础模型的64维嵌入空间进行可解释性分析，并基于此构建了一个基于检索增强生成（RAG）的陆表情报系统，能以自然语言回答环境查询。

**💡 创新点**

创新点在于：①将嵌入维度与26种环境变量的物理关系通过三种方法（Spearman、随机森林、Transformer）共同验证；②证明这些关系在空间块交叉验证和七年时间序列上高度稳健；③利用验证后的维度解释构建可检索的嵌入数据库，并通过LLM-as-Judge框架评估系统的科学性与可操作性。

**🔧 技术方法**

使用的技术包括：Spearman相关、随机森林重要性评估、跨任务Transformer预测与注意力分析、FAISS向量检索、检索增强生成（RAG）与多模型LLM（GPT‑OSS‑120B、Llama‑3.2‑11B‑Vision‑Instruct、Gemma‑3‑27B‑IT、Qwen3‑VL‑32B‑Instruct）以及旋转角色的LLM‑as‑Judge评估。

**📊 数据集**

数据集为12.1 M条CONUS（美国本土）样本，包含2017–2023年的AlphaEarth嵌入向量与26个环境变量（温度、植被、土壤、水文、地形、城市等）。

**📈 对比分析**

方法比较：空间块交叉验证显示Transformer平均R²下降仅0.017；随机森林对10个关键变量平均ΔR²为0.009；七年时间序列相关性平均r=0.963；LLM‑as‑Judge总体加权分数为3.74/5，主要优势在于“地面关联”(3.93)和“连贯性”(4.25)。

**⚠️ 局限性**

局限性包括：仅覆盖CONUS和7年；嵌入分辨率为1 km，导致细尺度地形和城市变量表现较弱；评估依赖LLM主观打分，缺乏专家实地验证；系统代码未公开。

---

## 199. MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs

**arXiv ID:** 2602.10965 | [PDF](https://arxiv.org/pdf/2602.10965v1)

**作者:** Yupu Gu `[一作]` (Tsinghua University), Pan Li `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 10216 | [OpenAlex ID](https://openalex.org/A5100455171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对稀疏 Mixture-of-Experts（MoE）大型语言模型的知识编辑框架 MoEEdit，实现了对特定事实的精准修改而不破坏模型整体能力。

**💡 创新点**

通过引入每个专家的 null‑space 投影来消除编辑导致的路由分布漂移，并结合随机块坐标下降（BCD）求解器，将编辑问题拆分为可线性规模的子问题，从而兼顾编辑局部性、路由稳定和计算效率。

**🔧 技术方法**

主要技术包括：per‑expert null‑space 投影、块结构优化、随机 BCD 求解器、路由分布相似度（RS）评估、Kullback‑Leibler 路由差异等。

**📊 数据集**

在 COUNTERFACT（单跳反事实编辑）和 ZsRE（零样本关系抽取）两大事实编辑基准上进行实验，使用 Qwen3‑30B‑A3B（128 专家）和 GPT‑OSS‑20B（32 专家）两款 MoE LLM。

**📈 对比分析**

与 FT、FT‑L、AdaLoRA、UnKE 等密集模型编辑器的 MoE 适配版进行对比，MoEEdit 在编辑成功率、泛化能力和特异性上均优于基线，且路由相似度>88%，计算与内存效率显著高。

**⚠️ 局限性**

局限性：仍需根据专家数量和隐藏维度调参投影阈值；在极大规模或多层编辑时可能出现累计路由漂移；缺乏对连续大规模批量编辑的实时评估。

---

## 200. LightGTS-Cov: Covariate-Enhanced Time Series Forecasting

**arXiv ID:** 2602.10412 | [PDF](https://arxiv.org/pdf/2602.10412v1)

**作者:** Yong Shang `[一作]` (Inspur Group Co. Ltd), Bin Yang `[通讯]` (School of Data Science and Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `edb9d762-f411-4838-a852-f2d638b018db` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了LightGTS-Cov，一种轻量级的时间序列预测模型，能够在保持LightGTS原有高效周期感知编码和并行解码的基础上，加入仅约0.1M参数的后端MLP插件，实现对历史和未来已知协变量的时序对齐融合，提升电价与光伏发电预测性能。

**💡 创新点**

创新点包括：①在不改变LightGTS主体结构的前提下，将协变量以token级对齐方式注入解码后端，实现可插拔的残差融合；②使用两阶段MLP融合（先合并历史协变量，再融合未来已知协变量），保持模型轻量且易于迁移；③兼顾历史与未来协变量，提升长期预测的准确性与稳定性。

**🔧 技术方法**

技术实现主要是基于LightGTS的周期感知分块、Transformer编码-解码；协变量采用相同分块得到token；后端使用两层MLP（层归一化+激活）进行两阶段融合；最后通过共享输出头将原始预测与协变量校正相加。

**📊 数据集**

实验使用公开能源/电价基准（EPF六个市场+Energy）以及工业实际部署数据：光伏站15天5min分辨率、日电价15min分辨率。

**📈 对比分析**

与多种基准比较：从无协变量到有未来已知协变量的全套深度学习模型（TFT、TimeXer、TiDE等）、基础TSFM（ChronosX、CoRA、UniCA、AdaPTS等）以及大规模基座（Chronos-2、Sundial）。LightGTS-Cov在有未来协变量时往往达到或超过最优，MSE/MAE均低于大模型；在无未来协变量时仍保持竞争力。

**⚠️ 局限性**

局限性：对未来已知协变量的依赖度高，若其缺失或噪声大可能影响；插件设计为通用但未在所有TSFM上验证；未考虑协变量不确定性和缺失建模；未直接嵌入业务约束（如容量上限、市场规则）。

---

## 201. Rising Multi-Armed Bandits with Known Horizons

**arXiv ID:** 2602.10727 | [PDF](https://arxiv.org/pdf/2602.10727v1)

**作者:** Seockbean Song `[一作]` (POSTECH), Jungseul Ok `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对有限时限下的上升多臂赌博机（RMAB）的时限感知算法CURE-UCB，利用累计奖励估计实现动态策略切换

**💡 创新点**

核心创新在于将累计潜在收益作为UCB指数，并显式引入已知时限T，使得算法能够精准捕捉时限依赖的最优策略切点，避免无效探索

**🔧 技术方法**

采用累计奖励估计UCB（CURE-UCB），并在实验中对比多种基准算法（R‑ed‑UCB、ET‑SWGTS、TI‑UCB、R‑ed‑AE、SW‑UCB、SW‑TS、Rexp3、Ser4 等）

**📊 数据集**

使用合成数据（Linear‑Then‑Flat 与 Concave 结构化环境）以及真实任务数据（IMDB 文本生成模型的在线模型选择）进行评估

**📈 对比分析**

实验显示CURE‑UCB在所有测试环境下均优于或与最强基准相当，尤其在LTF场景实现了严格优势，在线模型选择任务中与R‑ed‑UCB竞争并在长时限下保持低回报损失

**⚠️ 局限性**

局限性包括需事先知道完整时限T、对奖励函数的凹性与递增性假设以及对高维/复杂实际环境的适用性待进一步验证

---

## 202. Token-Efficient Change Detection in LLM APIs

**arXiv ID:** 2602.11083 | [PDF](https://arxiv.org/pdf/2602.11083v1)

**作者:** Timothée Chauvin `[一作]` (Université de Rennes), Gilles Tredan `[通讯]` (LAAS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种严格黑盒、仅观测输出 token 的 LLM API 变更检测方法 B3IT，能够在低温下通过边界输入实现高敏感性检测。

**💡 创新点**

核心创新在于证明低温软最大化下边界输入（top‑token 近乎 tied）导致 Fisher 信息与 Jacobian 的比值发散，从而提供理论上可无穷放大的检测信号，并将该理论转化为可执行的低成本采样与支持匹配测试。

**🔧 技术方法**

使用统计理论（局部渐近正态性、Fisher 信息、SNR 量化）、Transformer 结构推导、低温 softmax、边界输入搜寻与支持差异检验等技术。

**📊 数据集**

在 TinyChange 基准（9 大模型，0.5B–9B）以及 93 个商业 LLM API 端点（64 模型、20 提供商）上进行评估。

**📈 对比分析**

与黑盒 MET、MMLU‑ALG 以及灰盒 Log‑Probability Tracking（LT）比较，B3IT 在仅 1/30 的检测成本下实现约 0.9 的 ROC‑AUC，显著优于现有黑盒方法并接近灰盒性能。

**⚠️ 局限性**

局限性包括：对推理型模型（如需要多步推理）检测效果有限；边界输入在某些端点稀缺；方法仅基于单 token 输出，未覆盖多 token 生成场景。

---

## 203. Near-Constant Strong Violation and Last-Iterate Convergence for Online CMDPs via Decaying Safety Margins

**arXiv ID:** 2602.10917 | [PDF](https://arxiv.org/pdf/2602.10917v1)

**作者:** Qian Zuo `[一作]` (University of Edinburgh), Fengxiang He `[通讯]` (University of Edinburgh)

**通讯引用:** 1738 | [OpenAlex ID](https://openalex.org/A5100635369)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在线约束马尔可夫决策过程（CMDP）学习算法 FlexDOME，能够在不使用平均指标的强奖励与约束违例度量下，兼顾近常数强违例、亚线性强奖励回报以及最终迭代收敛。

**💡 创新点**

核心创新在于结合时间衰减的安全裕度与时间可变的正则化，采用逐项渐近占优策略严格控制每一步的优化与统计误差，从而实现强违例近常数且保持最后一次迭代收敛；同时提供了严格的理论分析与实验验证。

**🔧 技术方法**

技术手段包括：自适应安全裕度设计、熵与 L2 正则化的双重正则化、基于极限占优的误差分解与安全缓冲、使用潜在能量函数（Lyapunov）证明收敛、以及混合策略与对偶扰动向量的构造。

**📊 数据集**

在实验中使用随机生成的表格型 CMDP（S=20、A=H=5，单约束），并分别在随机阈值和固定阈值两种环境下进行评估。

**📈 对比分析**

与基线的原始原始双重方法及 UOpt-RPGPD 进行对比；实验表明 FlexDOME 在保持近零即时违例的同时，累计强违例保持常数，且在固定阈值场景下相较于 UOpt-RPGPD 的奖励回报略逊一筹，但整体性能优于两者。

**⚠️ 局限性**

局限性包括：回报回退为 O(T^{5/6}) 仍未达到最优 O(√T)；安全裕度与正则化参数的调度需要精细的理论计算，实际实现中可能对超参数敏感；实验仅限于小规模表格环境，缺乏对大型或连续空间的验证。

---

## 204. Contrastive Learning for Multi Label ECG Classification with Jaccard Score Based Sigmoid Loss

**arXiv ID:** 2602.10553 | [PDF](https://arxiv.org/pdf/2602.10553v1)

**作者:** Junichiro Takahashi `[一作]` (University of Tokyo Hospital), Norihiko Takeda `[通讯]` (University of Tokyo Hospital)

**通讯引用:** 7331 | [OpenAlex ID](https://openalex.org/A5000377104)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

训练了基于SigLIP的ECG编码器，并通过改进的Jaccard损失实现多标签ECG分类；

**💡 创新点**

创新点在于：①使用真实医院临床ECG数据进行多模态预训练；②将SigLIP的单标签损失改造成适用于多标签的Jaccard损失；③结合医学知识丰富的语言模型（如Qwen3-8B）提升预训练效果；④通过增大嵌入维度和随机裁剪消除数据漂移进一步提升性能；

**🔧 技术方法**

采用的技术包括SigLIP框架、1D ResNet-18编码器、Qwen3-8B（以及Gemma3-4B做对照）的语言模型、Jaccard‑based sigmoid损失、随机裁剪、嵌入维度提升（128→256）、Adam优化器；

**📊 数据集**

使用本院33,732个12导联、500 Hz、10 s时长的ECG样本，按患者划分为训练、验证、测试集；

**📈 对比分析**

与标准SigLIP、无医学知识语言模型以及不同嵌入维度/裁剪方案对比，采用Hamming Loss、Precision、Recall、F1、Jaccard指数评估；改进后F1由0.308提升至0.503，Hamming Loss下降至0.0451，Jaccard指数从0.0373提升至0.3495；

**⚠️ 局限性**

局限性包括：F1仍仅达约0.5，难以满足临床实用需求；部分标签（如心室早搏、心肌梗死、左心房扩大等）预测效果差；跨医院迁移时性能略有下降；模型缺乏解释性和鲁棒性，需要进一步研究。

---

## 205. 5Gone: Uplink Overshadowing Attacks in 5G-SA

**arXiv ID:** 2602.10272 | [PDF](https://arxiv.org/pdf/2602.10272v1)

**作者:** Simon Erni `[一作]` (ETH Zurich), Srdjan Capkun `[通讯]` (ETH Zurich)

**通讯引用:** 18407 | [OpenAlex ID](https://openalex.org/A5077290467)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了基于SDR的5G-SA上行掩盖攻击平台5Gone，能够实现对5G基站的无声Denial‑of‑Service、降级和用户隐私泄露（SUCI提取/重放）攻击。

**💡 创新点**

创新点在于首次实现了针对5G‑SA的上行掩盖攻击，并通过符号级处理架构实现低于500 µs的端到端响应时间，能够并行攻击多达64台UE，且不依赖硬件加速。

**🔧 技术方法**

技术手段包括USRP X310 SDR、COTS x86服务器、srsRAN/自研符号级协议栈、PDCCH/DSCCH解码、PUSCH编码、时距补偿、Octoclock 时钟同步以及软件定义的信道资源调度。

**📊 数据集**

实验数据集涵盖真实手机（Samsung S23、OnePlus Pro 10、Nothing Phone 3、iPhone 16/17 Pro、Xiaomi 15T Pro、Pixel 10 Pro）在实验室的Amarisoft gNodeB模拟器与运营商生产基站（B78 100 MHz）上的实际连通记录。

**📈 对比分析**

通过与开源UE实现（srsRAN、OpenAirInterface）对比，5Gone在100 MHz频宽下实现了271–362 µs的端到端延迟，单CPU核心即可完成64台UE的并行上行掩盖；在真实网络中对所有测试机型均能成功诱导DoS、降级或SUCI泄露。

**⚠️ 局限性**

主要限制包括：需要靠近基站或对时距有精确估计；对特定基站配置（如k₂值、频段）敏感；仅覆盖上行链路，无法干扰下行；需提前获取目标UE的TMSI以实现定向攻击；缺乏硬件加速时对更高负载或更低k₂值的支持。

---

## 206. Necessary President in Elections with Parties

**arXiv ID:** 2602.10601 | [PDF](https://arxiv.org/pdf/2602.10601v1)

**作者:** Katarína Cechlárová `[一作]` (P.J. Safarik University), Ildikó Schlotter `[通讯]` (ELTE Centre for Economic and Regional Studies)

**通讯引用:** 678 | [OpenAlex ID](https://openalex.org/A5070244460)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了在各种投票规则下候选人提名前的必要总统（Necessary President）问题的计算复杂度，并给出了多种规则（Borda、短规则、Veto‑like规则、Copeland^α、Maximin、Ranked Pairs）的多项式时间与NP/ W[1]/W[2] 复杂度分类。

**💡 创新点**

首次完成了必要总统问题在上述投票规则下的完整多参数复杂度图谱，尤其对短规则、Veto‑like规则在党派数或选民类型参数下的 W[2]/FPT 结果，以及 Ranked Pairs 在常数选民数下的 W[1] 难度证明；同时给出了新的多项式时间算法（Borda、Copeland^α、Maximin）。

**🔧 技术方法**

使用了多项式时间的贪心构造与匹配技术、整数规划模型、以及从已知 NP/ W[1]/W[2] 难度问题（(2,2)-E3‑SAT、Hitting Set、Multicolored Clique）构造的归约；对参数化复杂度采用了标准参数化归约与匹配算法。

**📊 数据集**

本研究主要为理论分析，没有使用公开的真实选举数据集；在整数规划实验部分引用了前人基于合成与真实选举数据的实验结果，但并未在本文中自行收集数据。

**📈 对比分析**

通过归约与算法分析，将问题的复杂度与已知难度问题进行对照，表明在多数规则下该问题是 NP‑或 W[1]/W[2]‑难；在特定参数化下可实现 FPT；不存在统一的实验性能指标，所给定的“性能”是以计算复杂度类和运行时间多项式度量为准。

**⚠️ 局限性**

局限性包括：仅考虑单一候选人（单胜选举），未扩展到多胜或委员会选举；仅涵盖了有限的投票规则集合，部分规则（如 STV、IRV 等）未被覆盖；参数化结果主要针对党派数、候选人集大小、选民类型，其他自然参数仍未探究；理论归约虽严谨，但缺乏对实际选举场景中约束和噪声的考量。

---

## 207. Multi-encoder ConvNeXt Network with Smooth Attentional Feature Fusion for Multispectral Semantic Segmentation

**arXiv ID:** 2602.10137 | [PDF](https://arxiv.org/pdf/2602.10137v1)

**作者:** Leo Thomas Ramos `[一作]` (Computer Vision Center), Angel D. Sappa `[通讯]` (ESPOL Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MeCSAFNet，一种针对多光谱遥感图像的多编码器 ConvNeXt 网络，并通过平滑注意力特征融合实现精细语义分割。

**💡 创新点**

创新点包括：①将可见光与非可见光通道分别由独立的 ConvNeXt 编码器处理；②引入融合解码器和 CBAM 注意力机制，显著提升多尺度特征融合质量；③使用 ASAU 激活函数以获得更平滑的梯度传播；④通过 4/6 通道输入（RGB-NIR 与 NDVI/NDWI 组合）实现对不同光谱配置的自适应。

**🔧 技术方法**

核心技术：双分支 ConvNeXt 编码器、FPN 结构解码器、Pixel Shuffle 上采样、CBAM 组合注意力、ASAU 激活函数、交叉熵+Dice 损失、AdamW+OneCycleLR 训练策略。

**📊 数据集**

使用的公开数据集：Five-Billion-Pixels（FBP）和 ISPRS Potsdam，分别评估大规模多类别和高分辨率城市场景。

**📈 对比分析**

与 U‑Net、DeepLabV3+、SegFormer 及多种最新方法进行横向对比。MeCSAFNet-base（6c）在 FBP 上 OA、mIoU、mF1 均达到最高；相较于 SegFormer(4c) mIoU 提升 19%+，与 DeepLabV3+ (6c) 提升 10%+；在 Potsdam 上 MeCSAFNet-large(4c) 与 MeCSAFNet-base(6c) 分别超越同类模型 6% 以上。推理速度保持在 0.05–0.06 秒/patch，训练时间约 15–20 小时。

**⚠️ 局限性**

局限性：1）大型模型在资源受限环境下训练和推理成本较高；2）在较大模型中收敛速度慢，可能需要更长训练；3）仅使用 NDVI/NDWI 两个手工索引，未探索更多光谱指标；4）在不同分辨率或传感器条件下的泛化尚未系统验证。

---

## 208. Reinforced Curriculum Pre-Alignment for Domain-Adaptive VLMs

**arXiv ID:** 2602.10740 | [PDF](https://arxiv.org/pdf/2602.10740v1)

**作者:** Yuming Yan `[一作]` (Tencent), Edith C. H. Ngai `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了RCPA框架，使得视觉-语言模型在保持通用能力的前提下，能够通过分阶段预对齐和强化学习对齐有效地适配专门领域。

**💡 创新点**

创新点在于将预对齐与强化学习对齐拆分为两个阶段，并引入了Curriculum Progress Perception（CPP）与Curriculum Difficulty Perception（CDP）两种课程化机制，以避免优化崩溃并抑制遗忘。

**🔧 技术方法**

采用了基于GRPON的RL策略，结合语义相似度、事实一致性和实体对齐的规则奖励；同时利用CPP控制答案前缀注入和奖励阈值调度，CDP对样本难度进行动态加权。

**📊 数据集**

实验使用了COCO Caption、Geo170K和OpenI三大领域数据集，用于评估域适配性能与通用能力的保持情况。

**📈 对比分析**

通过与SFT、PEFT、FFT、GRPO、DAPO、GRPON等基线对比，RCPA在域特定指标（如BLEU、CIDEr）与通用指标（如MMMU、MME、IFEval）均表现出优于或相当的性能，同时显著降低了通用能力的衰减。

**⚠️ 局限性**

限制在于计算成本相对较高（预对齐阶段占总时间28%，每步计算量比GRPO高56%），并对奖励阈值等超参数较为敏感。

---

## 209. Beyond Confidence: The Rhythms of Reasoning in Generative Models

**arXiv ID:** 2602.10816 | [PDF](https://arxiv.org/pdf/2602.10816v1)

**作者:** Deyuan Liu `[一作]` (Harbin Institute of Technology), Dianbo Sui `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 744 | [OpenAlex ID](https://openalex.org/A5031774490)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实验验证 Token Constraint Bound (TCB) 指标，用于衡量大型语言模型（LLM）在面对内部隐藏状态微小扰动时下一词预测的局部稳健性。

**💡 创新点**

创新点包括：① 以 Softmax Jacobian 的 Frobenius 范数为基础，给出 TCB 的精确解析公式；② 揭示 TCB 与输出嵌入几何分布（o_i^2 加权的方差）之间的直接关系；③ 将 TCB 作为提示工程和 In-Context Learning（ICL）效果的评估工具，识别准确性-稳健性冲突。

**🔧 技术方法**

技术方法：Softmax 解析、Jacobian 计算、Frobenius 范数、几何分析、梯度/扰动实验；实验使用 LLaMA‑3.1‑8B 作为模型，计算 TCB、有效词汇量、logit margin、perplexity 等指标。

**📊 数据集**

使用的数据集包括：MMLU（多学科多项选择）、GSM8K（多步推理）、以及自行构造的 Diverse Prompts (DPD) 与 Low‑Targeted (LVD) 两组提示数据集。

**📈 对比分析**

与传统的有效词汇量、logit margin、perplexity 等指标对比，实验表明：① TCB 能捕捉到其他指标遗漏的预测不稳健案例；② 在提示优化实验中，基于 TCB 的指导能在保持或提升准确率的同时，显著提高模型在扰动下的 worst‑case accuracy 与性能下降率（PDR）；③ 在不同置信度 regime 下，TCB 与 logit margin、有效词汇量的相关性变化符合理论预测。

**⚠️ 局限性**

局限性：实验仅在单一规模（8B）模型上验证，未覆盖更大规模或不同架构；TCB 仅关注最后隐藏层，未探究中间层的稳健性；未评估对结构性攻击或系统级扰动的鲁棒性；需要进一步验证跨模型、跨任务的通用性。

---

## 210. Signature-Kernel Based Evaluation Metrics for Robust Probabilistic and Tail-Event Forecasting

**arXiv ID:** 2602.10182 | [PDF](https://arxiv.org/pdf/2602.10182v1)

**作者:** Benjamin R. Redhead `[一作]` (University of Edinburgh), Amos Storkey `[通讯]` (University of Edinburgh)

**通讯引用:** 13680 | [OpenAlex ID](https://openalex.org/A5007901825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了两种基于签名核的概率预测评估指标 Sig-MMD 与 CSig-MMD，用于更准确地捕捉多变量时序依赖并专门评估尾部事件的预测性能。

**💡 创新点**

创新点包括：① 将签名核与最大均值差异结合，形成严格合规且对整体分布敏感的评分规则；② 设计基于马氏距离的裁剪机制 CSig-MMD，既能聚焦尾部预测，又保持指标的严格性。

**🔧 技术方法**

采用技术包括签名核、最大均值差异（MMD）、截断签名、马氏距离裁剪、最小协方差判别法（MCD）、PCA 降维、量化损失训练等。

**📊 数据集**

使用的实验数据集涵盖：合成高斯过程、ERA5 极端天气、EWELD、ETT、Weather、Exchange、Illness 等多种真实时序数据。

**📈 对比分析**

与传统 QL、CRPS、ES、VS 等指标及 PatchTST、iTransformer、TimesNet、N-HiTS、Chronos-2、Moirai 等前沿预测模型进行基准比较；CSig-MMD 在尾部事件评估上能够区分模型优劣，并揭示复杂模型在尾部表现不足。

**⚠️ 局限性**

局限性包括：截断签名在高维度下计算成本高；裁剪阈值需手动设定；指标尚未验证可作为训练损失函数；梯度可微性和高维可扩展性仍需进一步研究。

---

## 211. Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search

**arXiv ID:** 2602.10891 | [PDF](https://arxiv.org/pdf/2602.10891v1)

**作者:** Berfin Sakallioglu `[一作]` (University of Trieste), Eric Medvet `[通讯]` (University of Trieste)

**通讯引用:** 2641 | [OpenAlex ID](https://openalex.org/A5074055647)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种交互式 LLM 辅助的多任务进化策略搜索框架，使用 LLM 根据优化过程的实时反馈动态生成训练案例。

**💡 创新点**

创新点在于把 LLM 作为在线课程设计者，通过多模态反馈（数值、收敛曲线、行为可视化）让 LLM 自适应地调整难度，从而自动化生成有效的多任务课程。

**🔧 技术方法**

核心技术包括：语言模型（Claude Sonnet 4.5）生成场景，基于遗传编程（GP）编码的符号控制器，MAP-Elites（ME）进化优化，以及多模态（数值+图像）反馈交互。

**📊 数据集**

使用了 2D 机器人导航的 15×15 网格障碍环境，测试集包含 6 个未见的场景，训练集由 LLM 逐步生成的 8 个案例组成。

**📈 对比分析**

实验对比了专家手工课程、静态 LLM 课程、随机课程以及三种反馈模态（数值、进展+图、行为+图）。结果显示，进展+图和行为+图的交互式 LLM 课程在训练和测试精度上与专家课程持平，显著优于静态 LLM 与随机课程。

**⚠️ 局限性**

局限性包括：仅验证在单一 2D 导航任务上的可行性；LLM 需要复杂且结构化的提示，可能不易迁移到其他任务；未评估不同 LLM 或更大规模环境下的性能；实验规模受限，缺乏更广泛的基准比较。

---

## 212. Compute Only Once: UG-Separation for Efficient Large Recommendation Models

**arXiv ID:** 2602.10455 | [PDF](https://arxiv.org/pdf/2602.10455v1)

**作者:** Hui Lu `[一作]` (ByteDance), Yuchao Zheng `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种用户-组分离（UG‑Sep）框架，能够在稠密特征交互模型（如RankMixer）中实现用户侧计算的可重用，从而降低推理成本。

**💡 创新点**

创新点在于：①设计掩码机制在token‑mixing层显式分离用户与候选信息，保证一部分token始终保持纯用户表示；②通过可重用的per‑token FFN将该部分token的计算一次性完成；③引入信息补偿策略恢复掩码导致的交互信息损失；④结合W8A16权重‑仅量化缓解因用户计算降低而暴露的内存带宽瓶颈。

**🔧 技术方法**

使用的技术包括：token‑mixing掩码与可重用FFN、信息补偿投影加法、W8A16 8‑bit 权重量化、以及可扩展到注意力结构的UG‑masked attention。

**📊 数据集**

实验数据集来自ByteDance的真实业务场景，包含抖音（Feed、Feed Rec）、红果（Feed Rec）、穿山甲（CTR/AD）、千川（CVR）等四大业务，规模涵盖数十亿用户ID、数百万物品ID。

**📈 对比分析**

与基线RankMixer比较：离线AUC几乎无变动（±0.01%），推理延迟显著下降——抖音最多20%，红果12.5%，穿山甲12.7%，千川22%；在线A/B实验表明关键用户行为指标无显著差异，延迟下降达到12–20%。

**⚠️ 局限性**

局限性包括：当用户/组token比例过度失衡（如5:1）时仍可能出现性能下降；实现依赖于训练阶段的用户级样本聚合；掩码与补偿的超参数需针对不同业务细调；量化方案对极小批量或非权重‑仅场景的效果有限。

---

## 213. Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models

**arXiv ID:** 2602.10224 | [PDF](https://arxiv.org/pdf/2602.10224v1)

**作者:** Shiting Huang `[一作]` (University of Science and Technology of China), Feng Zhao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 13717 | [OpenAlex ID](https://openalex.org/A5102740754)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种 Meta-Experience Learning 框架，将通过自我验证获得的错误分叉点进行对比分析、抽象并自蒸馏后内化到 LLM 的参数记忆中，以提升 RLVR 在数学推理任务中的性能。

**💡 创新点**

创新点在于：①提出了 meta-experience 概念，将实例级错误转化为可复用的知识级启示；②利用对比分析定位错误分叉点并抽象成通用启示；③通过自蒸馏和内部化方法将 meta-experience 转化为模型参数记忆，提供密集的过程奖励，突破传统 RLVR 仅基于终局奖励的局限；④实现经验内部化而非仅外部提示。

**🔧 技术方法**

使用技术包括：RLVR/GRPO 强化学习框架；基于规则的验证器进行自我验证；对比分析生成分叉点、批判与抽象；负对数似然自蒸馏学习实现内部化；经验回放验证确保 meta-experience 的可靠性；联合 RLVR 与 MEL 的优化目标。

**📊 数据集**

训练数据：DAPO-Math-17k；评估数据集：AIME24、AIME25、AMC23、MATH500、OlympiadBench 等数学推理基准。

**📈 对比分析**

与 GRPO 基线以及原始 Qwen3-4B/8B/14B 模型对比，采用 Pass@1、Avg@8、Pass@8 三个指标；在所有模型规模上，Pass@1 提升 3.92%–4.73%，Avg@8 与 Pass@8 同样提升，显示了显著性能提升；同样的 meta-experience 在 RFT、REINFORCE++ 等其他 RL 方法中也能带来收益，且随着模型规模扩大收益更显著。

**⚠️ 局限性**

局限性：①对模型自身自验证和错误定位能力高度依赖，模型规模越小可获得的 meta-experience 质量下降；②meta-experience 可能出现幻觉或误导，需回放验证过滤；③目前仅在数学推理任务验证，跨领域推广仍需进一步研究。

---

## 214. CVPL: A Geometric Framework for Post-Hoc Linkage Risk Assessment in Protected Tabular Data

**arXiv ID:** 2602.11015 | [PDF](https://arxiv.org/pdf/2602.11015v1)

**作者:** Valery Khvatov `[一作]` (REALM AI), Alexey Neyman `[通讯]` (Big Data Association)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 CVPL（Cluster‑Vector‑Projection Linkage）框架，用于对已发布的受保护表格数据进行后置链接风险评估，量化潜在链接可行性而非单一合规判定。

**💡 创新点**

创新点在于：① 将链接过程拆解为阻塞、向量化、潜在投影、相似度评估的几何管线；② 通过阈值感知风险曲面 R(λ,τ) 把保护强度与攻击者严格度联合刻画；③ 引入递进阻塞与单调性保证，实现任意时刻的风险下界估计；④ 明确区分存在性链接与唯一链接，强调链接可行性为安全评估关键维度。

**🔧 技术方法**

核心技术包括：传统阻塞（基于泛化的 QI 区块化）、数值向量化（标准化+one‑hot）、PCA 或 UMAP 投影、余弦相似度、阈值化链接判定，及基于阈值与阻塞梯度的风险曲面构建。

**📊 数据集**

实验使用 10,000 条合成营销接触数据（包含年龄、性别、地区、产品、时间、广告渠道等），并在三类保护机制（k‑匿名、扰动、生成式合成）以及 19 种强度组合上进行评估；也对真实公开数据（如 Adult、German Credit）进行未来扩展计划。

**📈 对比分析**

与 Fellegi‑Sunter、随机阻塞、无投影、距离到最近记录（DCR）和最近邻距离比（NNDR）等基线比较，CVPL 在 0.9 阈值下的链接率仅 26% 但精度提升至 31.6%；k‑匿名下链接率与 1/k 形成反向关系；扰动机制表现为风险随噪声增大而线性下降；合成数据保持 18% 链接率但精度接近随机。整体显示 CVPL 能更细粒度捕捉结构保留导致的链接可行性。

**⚠️ 局限性**

局限包括：需要已知真链接才能量化评估；结果对阻塞策略、编码方式、投影方法与阈值高度敏感；仅评估存在性链接，未提供正式隐私保证；在极强保护下相似度可能退化导致低风险误判；未针对大规模或高维稀疏数据提供更高效近似方案。

---

## 215. Resilient Alerting Protocols for Blockchains

**arXiv ID:** 2602.10892 | [PDF](https://arxiv.org/pdf/2602.10892v1)

**作者:** Marwa Moullem `[一作]`, Ari Juels `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出并分析了区块链智能合约中“警报”问题，给出了贿赂抵抗的理论上界并设计了实现该上界的三种协议，

**💡 创新点**

创新点在于证明任意常数预算警报协议的贿赂抵抗上界为Θ(n²)，并首次通过同时游戏模型设计了锁步、TEE及顺序三种协议来实现该上界，

**🔧 技术方法**

采用游戏论、密码学技术（定时提交、Proof‑of‑Publication、TEE）、智能合约与区块链同步模型来构造和分析协议，

**📊 数据集**

未使用任何数据集，全部为理论分析与协议设计，

**📈 对比分析**

通过理论分析比较三种协议在贿赂抵抗、执行时间、交易数量和链上存储方面的性能，锁步与TEE协议在常数时间内实现Θ(n²)贿赂抵抗，但分别需要O(n)与2n交易；顺序协议在O(n)时间内实现Θ(n²)贿赂抵抗且零链上存储，但仅实现上界的一半；

**⚠️ 局限性**

主要局限在于对强同步、TEE可信性及固定节点数的假设，且未考虑动态节点、网络分区及复杂攻击情景等实际部署难点。

---

## 216. End-to-End LiDAR optimization for 3D point cloud registration

**arXiv ID:** 2602.10492 | [PDF](https://arxiv.org/pdf/2602.10492v1)

**作者:** Siddhant Katyan `[一作]` (Université Laval), Jean-François Lalonde `[通讯]` (Université Laval)

**通讯引用:** 4973 | [OpenAlex ID](https://openalex.org/A5034761030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种将 LiDAR 采集参数与点云配准算法超参数联合优化的自适应感知框架，利用注册反馈实时调整传感器设置以提升配准精度与效率。

**💡 创新点**

创新点在于将 LiDAR 采集与配准超参数实现端到端闭环优化，首次展示通过反馈机制实现任务感知的 LiDAR 参数自适应；并证明在不同环境下联合调优比单独调优或使用默认参数显著提升回调率。

**🔧 技术方法**

采用 CMA‑ES 进化优化器对 LiDAR 采集参数（功率、增益、噪声门限等）与配准参数（体素尺寸、FPFH 半径、对应距离阈值、迭代次数等）进行联合搜索，集成 CARLA + LiDAR 仿真模型与 Open3D 注册实现。

**📊 数据集**

在 CARLA 仿真环境中生成的三类场景（结构化、半结构化、非结构化）点云数据集，用于训练与测试，覆盖不同地形与反射特征。

**📈 对比分析**

与默认配置、仅调节 LiDAR、仅调节配准超参数四种策略对比，评估指标为配准召回率、平均平移/旋转误差；联合优化在所有算法（ICP、FGR、MAC）与环境下均实现显著提升，例如 FGR 在非结构化场景召回率从 18% 提升至 60%。

**⚠️ 局限性**

局限性包括仿真到真实硬件的差距、对实时性能的评估不足，以及需要进一步验证在真实 LiDAR 设备与不同噪声环境下的可迁移性。

---

## 217. A receding-horizon multi-contact motion planner for legged robots in challenging environments

**arXiv ID:** 2602.11113 | [PDF](https://arxiv.org/pdf/2602.11113v1)

**作者:** Daniel S. J. Derwent `[一作]`, Bruno V. Adorno `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种针对多足机器人在极限场景下的递归视野多接触运动规划器。

**💡 创新点**

创新点在于能够在实时重新规划、同时规划接触点与全身轨迹、避免潜在场地局部极小问题，并使用二次规划姿态生成器提升速度。

**🔧 技术方法**

技术包括递归视野（receding-horizon）规划、矢量场不等式（Vector-Field Inequalities）、双四元数代数（Dual Quaternion Algebra）以及二次规划（QP）姿态生成。

**📊 数据集**

使用模拟的挑战性场景数据集，如烟囱爬行、极窄通道与大跨距等。

**📈 对比分析**

与现有最先进方法对比，短周期规划（一步）平均快45%–98%，但姿态变化次数高5%–700%；长周期规划速度差异为73%–400%，但姿态变化次数可降低8%–47%。

**⚠️ 局限性**

局限在于在高质量规划时需要更长时间，且在某些场景下产生的姿态变化次数仍显多，尚未在真实机器人上验证。

---

## 218. MPA: Multimodal Prototype Augmentation for Few-Shot Learning

**arXiv ID:** 2602.10143 | [PDF](https://arxiv.org/pdf/2602.10143v1)

**作者:** Liwen Wu `[一作]` (Yunnan University), Bin Pu `[通讯]` (Hunan University)

**通讯引用:** 976 | [OpenAlex ID](https://openalex.org/A5022349009)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多模态原型增强框架 MPA，用于少样本学习，通过增强语义特征、多视角视觉特征以及不确定类别吸收来改进原型表示。

**💡 创新点**

创新点包括：①利用大语言模型生成多变体语义描述提升语义特征多样性；②层次化多视角增强将自然变换与几何视角相结合，显著扩大视觉特征空间；③自适应不确定类别吸收器通过插值与高斯采样动态生成不确定类别，降低类别间干扰。

**🔧 技术方法**

核心技术包括 CLIP 图文编码器、GPT‑4 语义生成、层次化数据增强、插值混合与高斯采样、逻辑回归分类器，以及动态权重 λ 计算。

**📊 数据集**

在四个单域数据集（miniImageNet、tieredImageNet、CIFAR‑FS、FC100）和六个跨域数据集（CUB、Cars、Places、Plantae、EuroSAT、CropDisease）上进行评估。

**📈 对比分析**

与多种先进方法（如 SPM、SP‑CLIP、MLVLM 等）对比，MPA 在 5‑way 1‑shot 与 5‑shot 任务中均取得最高或第二高准确率，单域平均提升约 12.3%，跨域平均提升 24.6%。

**⚠️ 局限性**

局限性主要在于对 CLIP 预训练模型和 GPT‑4 生成质量的依赖；在极端噪声或极少样本情形下，插值与高斯采样的效果可能不足，且模型对算力要求较高。

---

## 219. An Ontology-driven Dynamic Knowledge Base for Uninhabited Ground Vehicles

**arXiv ID:** 2602.10555 | [PDF](https://arxiv.org/pdf/2602.10555v1)

**作者:** Hsan Sandar Win `[一作]` (University of Adelaide), Tan Doan `[通讯]` (Defence Science and Technology Group)

**通讯引用:** 1070 | [OpenAlex ID](https://openalex.org/A5001775216)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了基于动态情境任务数据（DCMD）的语义驱动知识库，用于无人地面车辆（UGV）在战术边缘的自适应情境感知与决策。

**💡 创新点**

创新点在于：①将BFO、CCO等上层本体与任务专用中层本体结合，构建可重用的语义层次；②通过实时多模态感知、贝叶斯网络推理与本体驱动的知识图谱，实现在任务执行过程中对先验信息的动态更新（DCMD）；③利用TypeDB作为知识图谱引擎，实现本体实体与关系的高效存取与推理。

**🔧 技术方法**

采用的技术包括：YOLOv11目标检测、深度估计、贝叶斯网络（BN）进行身份与威胁推理、ROS 2框架下的机器人导航与通信、Nvidia Jetson ORIN AGX上GPU加速的视觉处理、TypeDB（TypeQL）实现知识图谱与本体融合、无线局域网（WLAN）实现UGV间信息共享。

**📊 数据集**

使用的“数据集”主要为实验室构建的1:18比例物理测试场景（包含已知目标、未知威胁与平民目标），以及预加载的UGV先验知识库（包括目标属性、任务地点等）。

**📈 对比分析**

在四辆UGV执行的监视与核查任务中，对目标识别、威胁判断和任务完成情况进行实验验证。实验表明DCMD能够实时更新知识库、共享信息并驱动协作决策，但论文未给出与传统静态知识库或无本体方法的量化对比，性能指标主要以任务成功率和信息更新时延的定性描述为主。

**⚠️ 局限性**

局限性包括：①实验环境为模拟测试，缺乏真实战术环境验证；②实时推理受限于硬件性能，复杂场景下的计算负荷未充分评估；③对检测不确定性和误报的处理尚不完善，未来需加入自适应推理与不确定性建模；④知识库可扩展性和跨团队共享机制的进一步验证仍待深入。

---

## 220. The State's Politics of "Fake Data"

**arXiv ID:** 2602.10944 | [PDF](https://arxiv.org/pdf/2602.10944v1)

**作者:** Chuncheng Liu `[一作]` (Northeastern University), danah boyd `[通讯]` (Cornell University)

**通讯引用:** 51321 | [OpenAlex ID](https://openalex.org/A5009029178)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对中国社区志愿服务记录与美国2020年人口普查数据制作流程的民族志研究，揭示了“假数据”在政府数据工作中的生产、纠正、共谋与扩充四个关键时刻。

**💡 创新点**

提出“假数据”是一种相对、过程化、表演化的概念，突破传统真假二分法，强调数据在组织实践中的功能性与可塑性。

**🔧 技术方法**

采用民族志访谈、现场观察及文档分析等人类学与社会学研究技术。

**📊 数据集**

使用的案例数据包括中国某市街道级志愿服务记录与美国2020年人口普查的内部处理数据。

**📈 对比分析**

通过跨案例对比分析阐释两国政府在处理假数据的共性与差异，未给出数值性能指标。

**⚠️ 局限性**

局限在于案例受限于两个国家与单一行政层级，缺乏对更广泛地区或其他数据类型的推广验证。

---

## 221. VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training

**arXiv ID:** 2602.10693 | [PDF](https://arxiv.org/pdf/2602.10693v1)

**作者:** Guobin Shen `[一作]` (Xiaohongshu Inc), Xing Yu `[通讯]` (Xiaohongshu Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的变分序列级软策略优化方法（VESPO），旨在提高大语言模型（LLM）在强化学习中的训练稳定性，特别是在离线策略更新的情况下。

**💡 创新点**

VESPO通过将方差减少纳入变分公式，提供了一种统一的理论基础，避免了现有方法中的长度归一化偏差，并直接在序列级别的权重上操作。

**🔧 技术方法**

采用了变分推断和重要性采样的技术，设计了一种闭合形式的重塑核，能够在不进行长度归一化的情况下处理序列级重要性权重。

**📊 数据集**

使用了未过滤的DAPO-Math数据集进行数学推理基准测试，评估了不同模型在训练和推理不匹配情况下的表现。

**📈 对比分析**

与GRPO、GSPO和SAPO等方法进行比较，VESPO在多个数学推理基准上表现出更高的准确性，尤其是在高达64倍的陈旧比率和完全异步执行下保持稳定的训练性能。

**⚠️ 局限性**

限制在于当前方法主要针对序列级别的优化，可能在更复杂的多回合交互和工具使用的代理强化学习设置中面临挑战。

---

## 222. TEGRA: Text Encoding With Graph and Retrieval Augmentation for Misinformation Detection

**arXiv ID:** 2602.11106 | [PDF](https://arxiv.org/pdf/2602.11106v1)

**作者:** Géraud Faye `[一作]` (Airbus Defence and Space), Céline Hudelot `[通讯]` (Université Paris-Saclay)

**通讯引用:** 1839 | [OpenAlex ID](https://openalex.org/A5070164335)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了文本-图混合编码方法TEG及其知识增强版TEGRA，用于误信息检测

**💡 创新点**

创新点在于将OpenIE生成的图与文本联合编码，并利用训练集知识图加上三元组筛选提升判别能力

**🔧 技术方法**

采用OpenIE6/KGI图提取、RoBERTa文本编码、fastText节点/边嵌入、Graph Attention Network、三元组筛选模块

**📊 数据集**

使用PolitiFact、GossipCop、CoAID、Horne2017四个误信息数据集进行实验

**📈 对比分析**

与RoBERTa、BERT、Tsetlin机、LLM、DeClarE等基线比较，TEG/TEGRA在大多数数据集上平均提升宏F1 2–4%，最高可达约99%宏F1

**⚠️ 局限性**

局限在于仅在任务特定训练集知识上测试，未验证对其他主题或通用知识的泛化，且检索策略过于简单

---

## 223. AD$^2$: Analysis and Detection of Adversarial Threats in Visual Perception for End-to-End Autonomous Driving Systems

**arXiv ID:** 2602.10160 | [PDF](https://arxiv.org/pdf/2602.10160v1)

**作者:** Ishan Sahu `[一作]` (Indian Institute of Technology Kharagpur), Soumyajit Dey `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 424 | [OpenAlex ID](https://openalex.org/A5085224640)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文在 CARLA 环境中对端到端自主驾驶模型 Transfuser 与 Interfuser 进行闭环评估，针对视觉感知提出了三种黑盒攻击（声学模糊、数字噪声与电磁干扰），并提出了轻量化的攻击检测模型 AD^2。

**💡 创新点**

创新点在于：①首次在闭环环境下系统评估端到端驾驶模型的黑盒鲁棒性；②设计了三类代表性视觉攻击并量化其对驾驶安全的影响；③提出基于多摄像头空间-时间一致性、轻量化 Transformer 结构的外部检测器 AD^2。

**🔧 技术方法**

使用 CARLA 仿真、Transfuser/Interfuser 端到端模型、三类黑盒攻击、ResNet+Transformer 的空间-时间一致性检测框架，并在多摄像头图像上进行训练与推理。

**📊 数据集**

实验数据来源于 CARLA Longest6 benchmark（RouteScenario0 用于测试，RouteScenario19 用于训练检测器），包含 20Hz 多摄像头图像流以及攻击后图像，构成四类（benign、Poltergeist、SNAL、ESIA）样本集。

**📈 对比分析**

与聚焦度测量、Kernel PCA 以及 CyberDet 等基线对比，采用准确率、TPR/FPR、AUC 等指标，AD^2 在四类攻击上均达到 1.0 的 TPR、0.0 的 FPR，参数量仅 1.1M、推理时间 0.16 ms，显著优于 23.5M 参数、0.26 ms 的 CyberDet；攻击导致驾驶分数最高可下降 99%。

**⚠️ 局限性**

局限性包括：假设攻击者不了解检测机制，未考虑自适应攻击；仅关注视觉感知攻击，未涵盖其他传感器；实验基于仿真，真实车载部署与对抗环境噪声的鲁棒性尚待验证。

---

## 224. What do people want to fact-check?

**arXiv ID:** 2602.10935 | [PDF](https://arxiv.org/pdf/2602.10935v1)

**作者:** Bijean Ghafouri `[一作]` (University of Southern California), Reihaneh Rabbany `[通讯]` (Mila)

**通讯引用:** 730 | [OpenAlex ID](https://openalex.org/A5043159673)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对用户在开放式AI事实核查系统中提交的近2500条声明进行大规模语义分类，构建了用户需求的行为图谱。

**💡 创新点**

首次量化用户事实核查需求的语义分布，并揭示与现有基准数据集（如FEVER）的结构差异，强调需求侧研究的重要性。

**🔧 技术方法**

使用GPT-OSS-20B对五个语义维度（领域、认知形式、可验证性、目标实体、时间指向）进行自动标注，结合模型生成的可信度得分。

**📊 数据集**

收集了457名Prolific受试者在实验应用中提交的2473条英文声明，另外抽样2000条FEVER声明做对照。

**📈 对比分析**

对比两组数据在五维分布上的绝对差异，发现FEVER在领域、认知形式、可验证性等维度上与真实用户差距显著，表明现有基准与真实需求脱节。

**⚠️ 局限性**

样本局限于英语Prolific用户，自动分类未经过人工验证，模型生成的可信度得分可能带偏差，需在多语言、多平台上复现。

---

## 225. Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents

**arXiv ID:** 2602.10226 | [PDF](https://arxiv.org/pdf/2602.10226v1)

**作者:** Haochen Wang `[一作]` (Google Inc), Lukasz Heldt `[通讯]` (Google Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建并部署了一套基于大型语言模型的自演化推荐系统，通过离线代理快速生成模型改进建议（结构变更、优化器、奖励函数），再通过在线代理在真实流量中验证并上线改进。

**💡 创新点**

① 让LLM担任专业ML工程师（MLE）角色，具备推理、代码生成与验证能力；② 采用双循环（离线快速、在线慢速）设计，形成高效的探索-验证过滤通道；③ 在工业级推荐系统中首次实现结构与语义层面的自动创新（如新激活函数、奖励逻辑），显著超越传统AutoML。

**🔧 技术方法**

使用Google Gemini 2.5 Pro LLM；Prompt工程（Persona、目标、约束、历史记录）+思考-代码-验证（Think‑Code‑Verify）循环；工具集包括模型训练、数据分析、A/B实验、DAG调度；双代理架构（Offline Agent + Online Agent）；A/B实验系统与实验日志。

**📊 数据集**

以YouTube全球视频推荐系统为背景，利用海量用户交互日志（点击、观看时长、问卷、频道偏好等）做离线训练；在线A/B实验使用真实流量数据。

**📈 对比分析**

对比方法：与人力工程师手工调优基线在同一任务（优化器、架构、奖励）下进行离线损失筛选与在线北极星指标A/B；性能表现：优化器改为RMSprop提升YouTube级+0.06%，架构Gated Path提升+0.14%，多目标奖励提升+0.13%；实验速度从每周1–10次提升至100次，工程成本几乎降至零。

**⚠️ 局限性**

局限性：① 依赖丰富的历史实验日志，冷启动时表现欠佳；② LLM可能产生幻觉，需严密安全约束；③ 奖励函数的离线评估仍无法完全捕捉延迟用户反馈；④ 对不同平台的迁移需要一定的调优；⑤ 对模型改动的可解释性与可审计性有限。

---

## 226. Generalized Decidability via Brouwer Trees

**arXiv ID:** 2602.10844 | [PDF](https://arxiv.org/pdf/2602.10844v1)

**作者:** Tom de Jong `[一作]` (University of Nottingham), Fredrik Nordvall Forsberg `[通讯]` (University of Strathclyde)

**通讯引用:** 235 | [OpenAlex ID](https://openalex.org/A5065899567)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在同伦类型论框架下提出并研究了α-可决性（α-decidability）的概念，利用Brouwer序数来描述命题在不同“阶数”下的可决性，并证明了该概念的闭包性质、与可选性（countable choice）的关系以及在计数量化、量化交叉等情形下的可决等级。

**💡 创新点**

创新点在于：①引入α-可决性作为对传统可决/半可决性的细粒度扩展；②使用Brouwer序数捕捉算法执行步骤的“层级”，从而实现对可决性的层级化描述；③通过形式化证明展示在不同层次下的闭包与可决等级（如ω²、ω·3、ω²+ω）以及在可选性假设下的半可决性提升。

**🔧 技术方法**

主要技术手段包括同伦类型论（HoTT）、Brouwer序数的 quotient inductive-inductive 定义、Cubical Agda 形式化、Sierpiński半可决性、有限/可选性原理（countable choice）以及对路径构造的细致处理。

**📊 数据集**

本文完全基于理论证明和形式化验证，无使用实际数据集或实验数据。

**📈 对比分析**

相较于传统的可决/半可决性方法，α-可决性提供了更细的可决等级，证明了在计数量化时可达的可决层级（如ω²、ω·3、ω²+ω），并在存在可选性的前提下将这些可决层级进一步降为半可决性；性能方面为理论证明，不涉及计算复杂度或实验测评。

**⚠️ 局限性**

局限性包括：①尚未完整解决α-可决性与β-可决性之间的上下闭包关系；②对更高层次的α-可决性（如ω^k+1）尚缺乏通用的提升方法；③在无可选性假设下对计数量化的半可决性提升仍不可行；④Brouwer序数可能不是最优的序数模型，存在寻找更合适序数的需求。

---

## 227. Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models

**arXiv ID:** 2602.10346 | [PDF](https://arxiv.org/pdf/2602.10346v1)

**作者:** Arash Gholami Davoodi `[一作]` (Carnegie Mellon University), Pouya Pezeshkpour `[通讯]` (Megagon Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于Token嵌入几何的截断解码方法Top-W。

**💡 创新点**

创新点在于将Wasserstein距离与熵和质量约束结合，形成几何感知的截断规则。

**🔧 技术方法**

采用最优传输（Wasserstein-1）与熵正则、几何距离的近似，结合一次性前缀扫描的高效实现。

**📊 数据集**

在GSM8K、GPQA、AlpacaEval、MT-Bench等四个评测基准上验证。

**📈 对比分析**

与Top-k、Top-p、Min-p、Top-H等传统截断方法对比，Top-W在准确率和创造性评估中均优于基线，提升幅度可达33.7%。

**⚠️ 局限性**

局限在于对大词表的距离计算仍需候选池近似，且在高温度下对多样性与准确性仍有折衷。

---

## 228. Kill it with FIRE: On Leveraging Latent Space Directions for Runtime Backdoor Mitigation in Deep Neural Networks

**arXiv ID:** 2602.10780 | [PDF](https://arxiv.org/pdf/2602.10780v1)

**作者:** Enrico Ahlers `[一作]` (Humboldt University of Berlin), Lars Grunske `[通讯]` (Humboldt University of Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种推理时的后门修复方法 FIRE，通过在网络的潜在空间中逆向补偿触发方向，实时纠正被污染的输入，避免修改模型权重或输入。

**💡 创新点**

创新点在于：①发现后门触发在某些潜在层会呈现一致的线性位移；②利用这一位移在推理过程中直接修正潜在表示，而非对输入或模型做昂贵的预处理；③采用在线更新机制，使防御随着更多受污染样本的到来逐步提升效果。

**🔧 技术方法**

技术细节包括：潜在方向估计（配对样本求差、无配对样本求均值差、增强对比估计），修复算子 Rep_ℓ(x, b, α)=x−αb，在线更新均值和方向，按层顺序判断是否改变预测。

**📊 数据集**

实验使用 CIFAR‑10 与 GTSRB 两个视觉数据集，模型为 PreActResNet18/34，攻击手段包括 BadNets、Blended、Blind、Bpp、FTROJAN、InputAware 与 WaNet 等。

**📈 对比分析**

与现有推理时修复方法 ZIP 对比：在第 1 个受毒样本时 FIRE 已显著提升准确率；到第 10 个样本时所有 21 种配置均超过 ZIP；Clean Accuracy 与 Poisoned Accuracy 接近；执行时间约 11–27 ms，远低于 ZIP 的 1.4–1.6 s，表现出更高的实时性和更好的后门消除效果。

**⚠️ 局限性**

局限性：目前仅在图像分类任务上验证；依赖清洁参考样本；对多目标后门或多方向触发的情况需要进一步研究；在触发结构极端或不规则时估计方向的准确性可能下降。

---

## 229. Dynamic Frequency Modulation for Controllable Text-driven Image Generation

**arXiv ID:** 2602.10662 | [PDF](https://arxiv.org/pdf/2602.10662v1)

**作者:** Tiandong Shi `[一作]` (Central South University), Chengli Peng `[通讯]` (Central South University)

**通讯引用:** 412 | [OpenAlex ID](https://openalex.org/A5053368234)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑free 的频率调制方法（FMM），通过在噪声潜变量上动态融合原始与改写文本对应的频率分量，实现对文本引导图像的结构保持与语义更新。

**💡 创新点**

创新点在于从频域角度阐释结构与纹理的层次生成机制，并设计频率依赖权重函数与动态衰减策略，在不同时间步对频谱进行柔性约束，从而避免空间域特征插值的经验性选择。

**🔧 技术方法**

采用快速傅里叶变换、频率权重函数、动态衰减、Stable Diffusion v1.5、DDIM 采样、CLIP、LPIPS、PSNR、SSIM 等技术。

**📊 数据集**

在 PIE‑Bench 与 ImageNetR‑Fake 两大基准上评估，同时使用 Stable Diffusion v1.5 生成的原始图像作为对照。

**📈 对比分析**

与 PnP、FPE、TtfDf、DiffEdit、h‑edit 等空间域方法对比，FMM 在结构保留、背景一致性（LPIPS/PSNR/SSIM）和文本‑图像对齐（CLIP Score）方面均取得最佳或接近最佳指标，整体性能显著优于现有方法。

**⚠️ 局限性**

在真实图像编辑时受 DDIM 逆转噪声不一致影响，导致高频纹理失真、细节模糊、结构偏差；参数 α/σ 的选择仍需手工调节，且对极端语义改写的适应性有限。

---

## 230. SecCodePRM: A Process Reward Model for Code Security

**arXiv ID:** 2602.10418 | [PDF](https://arxiv.org/pdf/2602.10418v1)

**作者:** Weichen Yu `[一作]` (Carnegie Mellon University), Matt Fredrikson `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10134 | [OpenAlex ID](https://openalex.org/A5057424614)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 SecCodePRM——一种过程奖励模型，能够在代码生成的每一步实时评估安全性，支持完整代码、部分代码的漏洞检测以及安全代码生成；

**💡 创新点**

创新点在于：①通过静态分析与人工标注构造步骤级安全标签，实现对漏洞出现时刻的精准定位；②采用风险敏感加权聚合与优势搜索，提供更密集、更早期的安全反馈；③无需完整程序即可完成漏洞检测，显著提升实时交互与长序列生成的安全性；

**🔧 技术方法**

技术方法包括：基于 Qwen2.5-Coder-7B-Instruct 的 transformer 加分类头，步骤级代码分割与 AST/序列对齐标签生成，风险加权聚合与优势搜索机制，以及 DeepSpeed ZeRO‑2 的高效训练；

**📊 数据集**

使用的数据集有：BigVul、SVEN、PrimeVul、ReposVul、PreciseBugs 等，包含对比、配对漏洞示例，配合 CodeQL 静态分析与人工注释生成训练标签；

**📈 对比分析**

在完整代码检测、部分代码检测及安全代码生成等任务上，与静态分析、LLM+GNN、LLM Prompting、VulSim、VulBERTA 等基线比较，SecCodePRM 在准确率、F1、SR@1 等指标上均取得显著提升（例如 PrimeVul 准确率 96.8%、F1 96.7%，安全生成 SR@1 超过 75%），展示了强大的性能优势；

**⚠️ 局限性**

局限性包括：①对极长代码依赖步骤分割可能导致语义切断；②训练仍需大量步骤级标签与静态分析支持，成本较高；③模型规模为 7B，进一步提升需更大算力；④在极端复杂跨文件或多文件交互式漏洞检测方面仍有提升空间。

---

## 231. GameDevBench: Evaluating Agentic Capabilities Through Game Development

**arXiv ID:** 2602.11103 | [PDF](https://arxiv.org/pdf/2602.11103v1)

**作者:** Wayne Chi `[一作]` (Carnegie Mellon University), Chris Donahue `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3173 | [OpenAlex ID](https://openalex.org/A5019674079)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GameDevBench基准，用Godot 4引擎的游戏开发任务评估多模态LLM代理的能力。

**💡 创新点**

首次将大规模游戏引擎任务与多模态评测相结合，自动从在线教程生成丰富的代码+图像+音频任务，并设计两种简易的多模态反馈机制。

**🔧 技术方法**

使用Claude、Gemini、ChatGPT Codex等LLM，并通过命令行或OpenHands框架实现项目文件访问；通过MCP截图和运行时视频提供多模态输入。

**📊 数据集**

基于约170份Godot 4教程（视频+网页）及其GitHub仓库自动生成202个任务，最终提供115个基准任务和17个变体。

**📈 对比分析**

在不同模型、框架与多模态反馈组合下进行比较，最高模型在无反馈时约46%成功率，加入截图或视频后可提升至约53%；整体表现仍低于传统软件基准。

**⚠️ 局限性**

局限在于对多模态理解不足、模型在场景/编辑器任务表现差、需人工验证、成本高且多模态反馈对不同模型效果不一致。

---

## 232. Boundary-Aware Multi-Behavior Dynamic Graph Transformer for Sequential Recommendation

**arXiv ID:** 2602.10493 | [PDF](https://arxiv.org/pdf/2602.10493v1)

**作者:** Jingsong Su `[一作]` (Beijing Normal University), Yu Guo `[通讯]` (Beijing Normal University)

**通讯引用:** 4253 | [OpenAlex ID](https://openalex.org/A5017660782)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种边界感知的多行为动态图Transformer（MB-DGT），用来解决多行为序列推荐问题，能够同时捕获用户兴趣随时间的动态变化和多行为间的高阶关联；

**💡 创新点**

创新点在于：①引入Transformer‑based多行为相关建模以学习行为间的细粒度依赖；②通过时间感知聚合实现动态图结构的逐层更新；③在损失函数中引入可学习的用户行为兴趣边界，提升个性化预测；

**🔧 技术方法**

主要技术包括Transformer、图神经网络（多层图聚合）、相对位置编码、时间编码（正弦‑余弦映射）以及自定义的边界感知点对点损失；

**📊 数据集**

实验使用了Yelp、Taobao和Tmall三个真实数据集，分别包含多种行为类型（如点击、加入收藏、加入购物车、购买等）；

**📈 对比分析**

与DMF、NGCF、GRU4Rec、SASRec、NMTR、MB‑GCN、DIPN、MB‑HT、MB‑STR等传统、序列、多行为及多行为序列推荐基线相比，MB‑DGT在HR@5、NDCG@5、HR@10、NDCG@10等指标均取得最高分，提升幅度约2.5%‑6.5%；

**⚠️ 局限性**

局限性主要体现在：①模型采用2‑hop图聚合，复杂度随节点数指数增长；②需要调优多项超参数（如嵌入维度、邻居数、α）；③对稀疏冷启动场景仍有限制，未来需进一步优化效率与泛化能力。

---

## 233. From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?

**arXiv ID:** 2602.10771 | [PDF](https://arxiv.org/pdf/2602.10771v1)

**作者:** Krishna Kanth Nakka `[一作]`, Vedasri Nakka `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并构建了CyclingVQA基准，用以评估视觉语言模型在自行车视角下的感知、时空理解和交通规则推理能力，并对多种模型进行评估。

**💡 创新点**

引入了以自行车为中心的视角基准，包含八类任务，揭示了车辆中心训练模型在自行车场景中的局限，并系统分析了失败模式。

**🔧 技术方法**

采用视觉语言模型（VLM），结合GPT‑5生成问题、Chain‑of‑Thought提示，使用多任务评估与解释文本解析。

**📊 数据集**

基于695张真实自行车视角图像（共2009个问答对），来自慕尼黑城市，手工标注交通标识、车道分割等。

**📈 对比分析**

通过准确率对比16个通用VLM、6个驾驶专用VLM、7个空间专用VLM以及两款专有模型，发现通用模型优于驾驶专用模型，最优性能来自Gemini‑2.5‑Flash，但整体仍低于人类，尤其在时空推理上表现最差。

**⚠️ 局限性**

数据规模、场景多样性、地理覆盖有限；模型解释性评估不足；CoT提示效果不佳；仅评估VLM，未涵盖VLA等新模型。

---

## 234. Why Human Guidance Matters in Collaborative Vibe Coding

**arXiv ID:** 2602.10473 | [PDF](https://arxiv.org/pdf/2602.10473v1)

**作者:** Haoyu Hu `[一作]` (Cornell University), Nori Jacoby `[通讯]` (Cornell University)

**通讯引用:** 3496 | [OpenAlex ID](https://openalex.org/A5021757437)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一套可控实验框架，用于系统比较人类主导、AI主导及混合模式的vibe coding（高层次自然语言指令驱动的代码生成）在SVG绘图任务中的表现。

**💡 创新点**

创新点在于：①首次以实验方式量化人机在vibe coding中的协作效果；②发现人类在提供高层次指令时表现优于AI，且AI往往导致性能衰退；③揭示了人类与AI指令在语义与长度上的根本差异；④提出并验证了人类负责方向、AI负责评估的高效混合角色分配。

**🔧 技术方法**

主要技术包括：大型语言模型（GPT‑5、Claude‑4.5‑Opus、Gemini‑3‑Pro）用于代码生成、指令编写和评估；PsyNet在线实验平台；文本嵌入与UMAP降维；多维语义指标评估指令质量。

**📊 数据集**

使用10幅由GPT‑5生成的动物SVG参考图（猫、狗、虎、鸟、象、企鹅、鲨鱼、斑马、长颈鹿、熊猫），共计604人参与实验，进行16项实验共计4800条AI/人类指令与SVG生成。

**📈 对比分析**

对比方法：人类与AI分别担任选择者/指令者进行15轮迭代，随后由独立人类评估SVG与参考图的相似度。结果显示：在前几轮两者相近，但最终人类组平均得分提升≈0.95分（p<0.001），而AI组相反下降。混合模式中，人类占比越高性能越好；完全AI主导表现最差。

**⚠️ 局限性**

局限性包括：①任务仅为SVG绘图，缺乏更复杂真实编码场景；②仅测试单一AI模型作为指令/选择者，未探究多模型协作；③实验为线性迭代链，未覆盖并行团队协作；④只考察短期交互，未评估长期反馈与认知负荷。

---

## 235. SurfPhase: 3D Interfacial Dynamics in Two-Phase Flows from Sparse Videos

**arXiv ID:** 2602.11154 | [PDF](https://arxiv.org/pdf/2602.11154v1)

**作者:** Yue Gao `[一作]` (Stanford University), Jiajun Wu `[通讯]` (Stanford University)

**通讯引用:** 12205 | [OpenAlex ID](https://openalex.org/A5100621605)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种利用动态高斯surfels、SDF几何约束与视频扩散模型从仅两视角高速度摄像头捕获的多相流视频中重建三维界面动态及其速度的方法。

**💡 创新点**

创新点在于：①首次将SDF约束与动态高斯surfels结合以保证界面几何一致性；②通过训练专门的单视角液气界面视频扩散模型提供新视角先验，弥补稀疏视角导致的重建不确定性；③利用泡泡实例绑定与速度引导初始化，实现从两视角获得物理量级的三维速度估计。

**🔧 技术方法**

核心技术包括：动态高斯surfels渲染与优化、SDF正则化、基于SDEdit的多视角视频扩散生成、Segment‑Anything模型进行泡泡实例分割、基于差分法的速度估计以及基于差分显式的相机标定与校准。

**📊 数据集**

使用了新收集的高速度沸腾实验视频数据集：200条单视角高速视频用于扩散模型训练；一组同步双视角（35°角度）已标定的高速视频用于评估；以及基于Houdini的合成多相流场用于量化基准。

**📈 对比分析**

与FluidNexus、4DGS、2DGS等现有方法对比，实测表明该方法在新视角视频合成、界面几何Chamfer距离以及泡泡速度L1误差方面均显著优于基线，尤其在仅两视角条件下仍能恢复连贯的三维几何和物理速度。

**⚠️ 局限性**

局限性包括：需要同步高速相机与精确标定，扩散模型对训练数据的相似度高度依赖，且在极端遮挡或界面快速顶点拆分时仍可能出现几何误差；此外，该方法目前针对液气界面，尚未证明能直接扩展到更复杂的多相或高浓度固液系统。

---

## 236. Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives

**arXiv ID:** 2602.10541 | [PDF](https://arxiv.org/pdf/2602.10541v1)

**作者:** Antonin Sulc `[一作]` `[通讯]` (Lawrence Berkeley National Laboratory), Antonin Sulc (Lawrence Berkeley National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 FastLSQ 方法，利用冻结的随机正弦特征一次性构造 PDE 线性算子矩阵并求解线性 PDE；对非线性 PDE 采用 Newton–Raphson 迭代，每步仍使用 FastLSQ 求解线性子问题。

**💡 创新点**

创新点在于发现正弦随机 Fourier 特征具有闭式循环导数结构，使得任何阶线性微分算子在特征点处的值与导数均可在 O(1) 时间内解析计算；基于此实现了无自动微分、一次性最小二乘求解器，并将其推广至非线性 PDE，显著提高了精度与速度。

**🔧 技术方法**

使用技术包括：冻结随机 Fourier 特征、闭式求导与算子矩阵组装、最小二乘一次性线性求解、Newton–Raphson 迭代、Tikhonov 正则化、回溯线搜索、热启动、连续参数（viscosity 递减）等。

**📊 数据集**

实验数据集为 17 个基准 PDE，涵盖 5 个线性（Poisson5D、Heat5D、Wave1D、Helmholtz2D、Maxwell2D）、5 个非线性（NL-Poisson2D、Bratu2D、SteadyBurgers1D、NL-Helmholtz2D、Allen–Cahn1D）和 7 个回归模式 PDE（Burgers Shock、KdV Soliton、Fisher Reaction‑Diffusion、Sine‑Gordon、Klein‑Gordon、Gray‑Scott、Navier‑Stokes Kovasznay）。

**📈 对比分析**

与 PINNacle（最优 PINN 变体）、RF‑PDE（迭代随机特征）和 PIELM（tanh 基底）进行对比。在线性 PDE 上，FastLSQ 达到 10⁻⁷ 的 L² 误差，耗时 0.07–0.09 s，比 PINNacle 快 10³–10⁴ 倍、精度高 10³ 倍；比 RF‑PDE 速度快 10³ 倍、误差低 10³ 倍；tanh 基底误差高 10–1000 倍。对非线性 PDE，Newton–Raphson 迭代实现 10⁻⁸–10⁻⁹ 误差，耗时 <9 s，甚至优于回归或 PINN 求解器。

**⚠️ 局限性**

局限性包括：需要手工调节频率超参数 σ；高阶或大 σ 时算子矩阵条件数高，精度受限；边界惩罚 λ 需调参；仅适用于盒形域，需改进边界采样；1/√N 归一化导致随 N 增大精度不再提升；Newton 迭代相比线性模式慢，需要多次最小二乘求解。

---

## 237. LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer

**arXiv ID:** 2602.10556 | [PDF](https://arxiv.org/pdf/2602.10556v1)

**作者:** Lihan Zha `[一作]` (Princeton University), Anirudha Majumdar `[通讯]` (Princeton University)

**通讯引用:** 974 | [OpenAlex ID](https://openalex.org/A5102792178)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种语言动作预训练（Language‑Action Pre‑Training，LAP）框架，将低层机器人动作直接用自然语言描述，以对齐预训练视觉‑语言模型（VLM）的输入‑输出分布，并基于此实现了零样本跨机器人实体迁移的 VLA 模型 LAP‑3B。

**💡 创新点**

创新点在于：① 将连续端执行器运动转化为结构化的自然语言动作（language‑action），消除了传统 VLA 在动作表示上与 VLM 预训练分布的不匹配；② 在不引入额外 tokenizer 或实体专用网络的前提下，通过语言监督实现跨实体泛化；③ 通过混合 Transformer 与轻量级扩散专家实现实时控制，并展示了语言‑动作协同训练能提升 VQA 任务与动作生成的性能。

**🔧 技术方法**

技术包括：VLM 基础（PaliGemma‑3B）、跨实体数据混合、语言‑动作序列的交叉熵训练、流匹配（flow‑matching）动作专家、梯度隔离（knowledge insulation）以及多视角图像与姿态的文本化表示。

**📊 数据集**

使用了 Open X‑Embodiment（OXE）、MolmoAct、DROID、LIBERO 等公开机器人数据集，并在多机器人体式（Franka Panda、YAM、Kinova 等）和多任务（Pick‑Place、Sort、Tissue、Put‑Towel、Pour）上进行评估。

**📈 对比分析**

与公开的 VLA 基线（π₀.₅‑DROID、π₀.₅‑Base、X‑VLA、MolmoAct、OpenVLA）以及自研对照（π₀.₅‑replicated、π₀‑replicated、VLA‑0‑replicated）比较，LAP‑3B 在三种未见实体上平均零样本成功率超过 50%，约比最强基线提升 2 倍；在细粒度微调中仅需 2.5 倍更少的演示与梯度步骤即可达到同等或更高性能。

**⚠️ 局限性**

局限性：① 目前仅验证单臂机械手，对双臂或多自由度复杂动作尚未扩展；② 对高频率或极高精度任务（如快速反应或细腻变形物体操作）尚未评估；③ 依赖固定坐标系统与结构化模板，若数据来源不一致可能需要额外预处理；④ 只在机器人与 VLM 混合数据上训练，跨非机器人交互场景的适用性待进一步验证。

---

## 238. Self-referential instances of the dominating set problem are irreducible

**arXiv ID:** 2602.10559 | [PDF](https://arxiv.org/pdf/2602.10559v1)

**作者:** Guangyan Zhou `[一作]` (Beijing Technology and Business University), Guangyan Zhou `[通讯]` (Beijing Technology and Business University)

**通讯引用:** 42 | [OpenAlex ID](https://openalex.org/A5102947791)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文证明了在 Erdős–Rényi 随机图模型 G(n,p) 下，支配集合问题在任何常数 0<c<1 的子图上都无法通过局部信息决定是否存在大小为 ln n 的支配集合，从而表明该问题在随机图中具有不可约性（irreducibility）

**💡 创新点**

创新点在于构造了“自指”实例与对称映射（symmetry mapping），能够仅通过修改常数条边来切换图是否拥有支配集合，使得任何仅检查 ≤n^c 规模子图的算法都无法区分两类实例

**🔧 技术方法**

核心技术包括：第二矩方法计算支配集合的期望与方差；细致的概率估计证明存在唯一支配集合或仅缺少一条边即可支配所有顶点的两种情况；以及对称映射（局部边交换）保持子图不变而改变全局属性

**📊 数据集**

研究对象为随机图 G(n,p)，没有使用真实数据集，而是对随机生成的图理论上分析其属性

**📈 对比分析**

由于文章主要给出理论证明，未涉及实验对比；但结论显示即使对极大子图进行完整枚举，仍需全图搜索，说明任何局部或基于子结构的算法无法突破穷举搜索

**⚠️ 局限性**

局限性：仅适用于特定选择的边概率 p=p(n)；结果是概率上正向而非确定性；此外缺乏对算法实际运行时间的实验评估，无法说明在实践中该理论难度的表现

---

## 239. Deriving and Validating Requirements Engineering Principles for Large-Scale Agile Development: An Industrial Longitudinal Study

**arXiv ID:** 2602.10972 | [PDF](https://arxiv.org/pdf/2602.10972v1)

**作者:** Hina Saeeda `[一作]` (Chalmers University of Technology), Niels Jørgen Strøm `[通讯]` (Grundfos Holding A/S)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在基尔霍尔姆斯技术大学与格伦多福斯公司合作，进行为期五年的纵向案例研究，系统收集并分析了来自 25 个冲刺、约 320 次同步会议、7 次跨公司研讨会以及 5 次高级领导焦点小组的定性数据，最终在格伦多福斯、博世、爱立信和沃尔沃汽车四家公司中共识并验证了六条适用于大规模敏捷系统开发的需求工程原则。

**💡 创新点**

创新点在于：① 将需求工程（RE）原则从抽象理论转化为可操作的、可扩展的高层指导原则；② 通过长达五年的跨组织纵向研究和跨公司专家验证，填补了大规模敏捷系统中缺失的高层 RE 指导空白；③ 将 IREB、Kasauli 等现有框架与行业实践相结合，提出了“架构语境”“分权需求管理”“最小可行文档”等在先前研究中缺失或未系统化的原则。

**🔧 技术方法**

技术方法包括：定性数据收集（访谈、焦点小组、研讨会、冲刺会议记录）；主题分析（Braun & Clarke 六阶段方法 + NVivo + Excel）与编码一致性检验；三角验证（多来源、多研究者、专家评估）；文献综述（IEEE Xplore、Scopus、ACM）与 IREB 基准对比；以及使用 Likert 量表评估原则重要性。

**📊 数据集**

使用的数据集主要是公司内部产生的非结构化文本：约 25 次冲刺回顾报告、320 次同步会议日志、7 次跨公司研讨会记录、5 次焦点小组访谈转录，以及跨公司专家反馈记录。数据量虽不大，但覆盖了超过 100 个敏捷团队、数百名开发者以及多家跨国企业。

**📈 对比分析**

比较方法主要是通过专家打分（五点 Likert 量表）评估各原则的价值，并在三角验证阶段与 IREB 及其他框架进行对照。结果显示“系统架构语境”和“验证”在所有公司中均被评为最高优先级；“需求演化”和“分权需求管理”亦被高度认可。该研究并未给出传统意义上的性能指标，而是通过原则在真实项目中的可操作性、跨组织一致性以及被采纳的频率来体现其价值。

**⚠️ 局限性**

局限性包括：① 研究主体主要来自单一工业客户（格伦多福斯），尽管通过跨公司验证提升外部有效性，但仍可能存在行业特定偏差；② 结果以定性访谈和专家评估为主，缺乏量化的性能度量；③ 原则的适用性需要在不同组织文化、团队规模与监管环境中进一步定制化；④ 长期纵向研究的时间成本高，复制性较低；⑤ 研究未对比传统 RE 工具或流程的具体效果，导致难以量化“改进”程度。

---

## 240. Stochastic Parroting in Temporal Attention -- Regulating the Diagonal Sink

**arXiv ID:** 2602.10956 | [PDF](https://arxiv.org/pdf/2602.10956v1)

**作者:** Victoria Hankemeier `[一作]` (University of Münster), Malte Hankemeier `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了时序注意力中存在的对角线注意力汇聚（diagonal attention sink）导致的过度压缩，并给出了敏感度界限。

**💡 创新点**

证明了时序注意力矩阵的对角线注意力汇聚现象，并提出了三种对角线正则化方法来缓解随机鹦鹉问题。

**🔧 技术方法**

使用了时序注意力层、残差连接、图卷积网络、绝对位置编码以及对角线掩码/Dropout/惩罚正则化等技术。

**📊 数据集**

在METR‑LA交通流预测数据集上进行实验。

**📈 对比分析**

与无残差、无正则化以及仅使用残差的基线对比，正则化方法提升MAE/RMSE/MAPE约2‑3%，并显著降低对角线强度。

**⚠️ 局限性**

缺点在于对角线正则化对模型可解释性有限，且实验仅在单一交通数据集上验证，缺乏更广泛的跨域评估。

---

## 241. Multimodal Priors-Augmented Text-Driven 3D Human-Object Interaction Generation

**arXiv ID:** 2602.10659 | [PDF](https://arxiv.org/pdf/2602.10659v1)

**作者:** Yin Wang `[一作]` (Beihang University), Xiaohui Liang `[通讯]` (Beihang University)

**通讯引用:** 9073 | [OpenAlex ID](https://openalex.org/A5101655447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多模态先验的文本驱动三维人机交互生成框架MP-HOI

**💡 创新点**

核心创新点包括：①引入文本、图像、三维空间三模态先验以弥补跨模态鸿沟；②改进对象表征，加入几何关键点、接触信息和动态属性；③设计模态感知混合专家（MoE）模型实现高效多模态特征融合；④构建级联扩散网络并加入交互监督，逐步细化人机交互动作

**🔧 技术方法**

使用GPT-4o提取文本先验、Flux生成视觉先验、CLIP+PointNet编码空间先验；混合专家网络结合FiLM调制；三阶段扩散模型（人、物、交互）以及多任务损失（L2、速度、距离、交互）

**📊 数据集**

在FullBodyManipulation（单物体）和HIMO（2/3物体）两大公开数据集上进行实验

**📈 对比分析**

与IMoS、MDM、OMOMO、PriorMDM、MotionDiffuse、CHOIS、HIMO-Gen等SOTA方法对比，MP-HOI在R‑TOP、FID、MM‑Dist、Diversity、Precision/Recall/F1、Contact %和Interaction Distance等指标均取得显著提升，尤其在多物体交互中的性能差距最高可达60%+

**⚠️ 局限性**

局限性包括：对柔性或流体对象交互支持不足；缺乏细粒度手指参数导致手物碰撞误差；偶尔出现足部滑动等物理不一致问题

---

## 242. Frame-Level Internal Tool Use for Temporal Grounding in Audio LMs

**arXiv ID:** 2602.10230 | [PDF](https://arxiv.org/pdf/2602.10230v1)

**作者:** Joesph An `[一作]` (University of Washington), Noah A. Smith `[通讯]` (Allen Institute for AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

训练音频语言模型使用内部帧级工具直接对音频进行时间定位，而非生成文本形式的时间戳；

**💡 创新点**

将模型内部的音频表示重新利用为时间预测工具，并结合二元分类损失与非齐次泊松过程（IHP）损失，实现高精度、低幻觉的时间定位，同时大幅提升推理速度（50×+）。

**🔧 技术方法**

使用轻量级预测头、二元交叉熵与IHP损失、时间重标定后验模式推理、内部工具调用等技术。

**📊 数据集**

在LibriSpeech（词定位）、LibriCount（说话人分段）和AudioSet人声子集（事件定位）三大数据集上进行实验。

**📈 对比分析**

与零样本、token化时间戳生成以及二元分类损失的基线对比，IHP损失在准确率和平均绝对误差（MAD）上均优于基线；在长序列和外域时长下仍保持较高准确率；在多时间戳任务中实现5–50×的推理加速。

**⚠️ 局限性**

仍需帧级标注数据，且对多事件/复杂时间结构的处理有限；对非音频事件的泛化能力尚未验证；模型训练和推理仍需要一定资源。

---

## 243. Anomaly Detection with Machine Learning Algorithms in Large-Scale Power Grids

**arXiv ID:** 2602.10888 | [PDF](https://arxiv.org/pdf/2602.10888v1)

**作者:** Marc Gillioz `[一作]` (University of Applied Sciences of Western Switzerland), Philippe Jacquod `[通讯]` (University of Geneva)

**通讯引用:** 5282 | [OpenAlex ID](https://openalex.org/A5045471214)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究并比较了九种机器学习算法在大规模高压电网运维数据中检测异常的效果，重点针对水电站功率输出的“on/off”伪造攻击和一般上下文异常。

**💡 创新点**

提供了基于完整合成时序数据的系统性实验，验证了神经网络与无监督回归方法在检测上下文异常时与传统分类器相比的显著优势，并探讨了历史长度与输入上下文对性能的影响。

**🔧 技术方法**

采用了七种传统分类器（NBC、KNN、SVM、RFC、GBC、MLPC）以及两种深度网络（LSTM分类与回归）和无监督回归模型，使用F2分数、R²和误差分析等指标进行评估。

**📊 数据集**

使用了公开的欧洲大陆电网模拟数据集（20年时序，按小时分辨率），并在瑞士、西班牙、德国三大子网的负荷与发电数据上进行实验。

**📈 对比分析**

通过5折交叉验证调参，在测试集上用F2分数比较，结果表明GBC、MLPC、LSTM分类器在检测伪造攻击时F2>0.9，回归模型R²>0.95，且无监督模型在多攻击场景下仍保持低误差。

**⚠️ 局限性**

受限于计算资源，深度网络的层数和训练时间有限；对不同类型发电机的性能差异较大，尤其在西班牙网络中；并且在真实攻击多样性与时序相关性方面仍有提升空间。

---

## 244. Found-RL: foundation model-enhanced reinforcement learning for autonomous driving

**arXiv ID:** 2602.10458 | [PDF](https://arxiv.org/pdf/2602.10458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 245. Design, Development, and Use of Maya Robot as an Assistant for the Therapy/Education of Children with Cancer: a Pilot Study

**arXiv ID:** 2602.10942 | [PDF](https://arxiv.org/pdf/2602.10942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 246. Emotion-Coherent Speech Data Augmentation and Self-Supervised Contrastive Style Training for Enhancing Kids's Story Speech Synthesis

**arXiv ID:** 2602.10164 | [PDF](https://arxiv.org/pdf/2602.10164v1)

**作者:** Raymond Chung `[一作]` `[通讯]`, Raymond Chung

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

使用情感匹配语句拼接与自监督对比学习的数据增强方法，训练一种能一次性生成多句书朗读文本的富表达式TTS模型。

**💡 创新点**

创新点在于：①将情感相似的句子拼接成长语音段，提升表达性；②在GST模块的参考编码器上引入SimCLR对比损失，优化说话风格嵌入；③通过正态分布模拟真实句间停顿，使模型学会自然停顿。

**🔧 技术方法**

核心技术包括Tacotron2+TP‑GST、stepwise monotonic attention、WaveGlow声码器、Fine‑tuned T5‑base情感分类器、SimCLR对比学习、MFA对齐评估。

**📊 数据集**

使用的语料库为：LJSpeech（21h）预训练、LibriTTS（200h）进一步训练、Blizzard 2017儿童故事（5,155句，约6.5h）细调，并利用情感分类器在LJSpeech上做情感标注做增量。

**📈 对比分析**

与仅用连续两句语音训练的基线（M2）相比，M4模型在说话风格嵌入预测误差下降、情感分类准确率提升、句间停顿分布与真实数据更相符（KS检验 p=0.630），MOS自然度与风格适配度均高于基线。

**⚠️ 局限性**

局限性包括：仅在英语儿童故事数据上验证，缺乏跨语言通用性；声码器未针对高 F0 语调微调，可能影响情感细腻度；数据量虽增大但仍受限，模型对极端情感场景的鲁棒性未知。

---

## 247. I can tell whether you are a Native Hawlêri Speaker! How ANN, CNN, and RNN perform in NLI-Native Language Identification

**arXiv ID:** 2602.10832 | [PDF](https://arxiv.org/pdf/2602.10832v1)

**作者:** Hardi Garari `[一作]` (University of Kurdistan Hewler), Hossein Hassani `[通讯]` (University of Kurdistan Hewler)

**通讯引用:** 8076 | [OpenAlex ID](https://openalex.org/A5037960046)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

收集并分析了约23小时27分钟的赫勒里（Hewlêri）方言语音数据，构建了用于判别母语与非母语说话者的NLI模型；

**💡 创新点**

首次提供了赫勒里方言的语音NLI数据集，并证明了在短时（5秒）音频片段上采用RNN可获得高达95.92%的准确率，显著优于传统CNN与ANN；

**🔧 技术方法**

采用MFCC特征提取，训练了人工神经网络（ANN）、卷积神经网络（CNN）和循环神经网络（RNN）三种深度学习模型；

**📊 数据集**

使用了自建的Hewlêri Speech Corpus，包含40位说话者（19名母语者，21名非母语者），共23小时27分钟语音；

**📈 对比分析**

通过66组实验（不同时长、过采样/欠采样、交叉验证），RNN在5秒片段上取得最高准确率95.92%，CNN在10秒片段上达94.47%，ANN在5秒片段上最高82.92%；RNN训练速度最快、资源占用最低；

**⚠️ 局限性**

局限在于数据集仅覆盖两类说话者，未涵盖更多方言差异；训练主要基于单一方言，缺乏跨方言验证；模型对较长片段的鲁棒性下降。

---

## 248. Assessing Vision-Language Models for Perception in Autonomous Underwater Robotic Software

**arXiv ID:** 2602.10655 | [PDF](https://arxiv.org/pdf/2602.10655v1)

**作者:** Muhammad Yousaf `[一作]` (Simula Research Laboratory and Oslo Metropolitan University), Shuai Wang `[通讯]` (Det norske Veritas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对四种开源视觉‑语言模型（VLM）在水下垃圾检测中的零样本分类效果与不确定性进行了系统评估，旨在为自主水下机器人（AUR）感知模块的可靠性与风险管理提供实验依据。

**💡 创新点**

创新点在于：①首次从软件工程视角量化水下环境下VLM的性能、置信度、不确定性与校准关系；②引入token‑level不确定性指标（MSP、PCS、Entropy、Deep Gini）与校准指标（ECE、MCE），为安全关键水下应用提供更细粒度的信赖评估；③通过对比不同VLM在相同提示、配置下的表现，揭示高性能与良好校准并非简单由高置信度决定，提供模型选择的依据。

**🔧 技术方法**

使用的技术包括：开源VLM（如DeepSeek‑VL、Llava、SigLIP、CLIP 变种）在零样本推理模式下执行；统一的提示工程（固定prompt、max token 80、temperature=0、do_sample=false）；对生成文本进行后处理映射到四类标签；利用token‑level softmax分布计算MSP、PCS、Entropy、Deep Gini；利用预测结果与真实标签计算ECE/MCE；在Python+PyTorch+NVIDIA RTX‑5090环境下批量评估。

**📊 数据集**

采用的水下数据集为：4cTrashCan1.0 与 SeaClear，两者均包含“动物、植被、物体、垃圾”四大类别，并分别提供深浅海景像。实验中还构建了两数据集的合并（Aggregated）版本以检验跨域泛化。

**📈 对比分析**

对比方法：在相同prompt、模型参数、硬件设置下，计算每个模型的F1（Micro/Macro）、Jaccard、MSP、PCS、Entropy、Deep Gini、ECE、MCE。结果显示，最优模型在聚合数据集上F1(Micro)≈0.67，F1(Trash)≈0.76，F1(Object)≈0.87，且校准误差（ECE≈0.05）最小；其它模型虽然置信度高，但校准差、动物/植被类表现差强人意，说明单凭高置信度并不能保证可靠性。

**⚠️ 局限性**

局限性：①仅评估了四种VLM且未进行任何领域微调，无法充分挖掘模型潜能；②数据集仅覆盖两种公开数据，无法全面检验在更广泛水下环境中的泛化能力；③在动物和植被类上表现不佳，提示VLM对未见类别的泛化受限；④评估侧重零样本推理，未探究模型在真实工业部署（资源受限、实时性）下的可行性。

---

## 249. Adaptive Time Step Flow Matching for Autonomous Driving Motion Planning

**arXiv ID:** 2602.10285 | [PDF](https://arxiv.org/pdf/2602.10285v1)

**作者:** Ananya Trivedi `[一作]` (Northeastern University), Faizan M. Tariq `[通讯]` (Honda Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于自适应流匹配的实时交互式行驶轨迹规划框架，能够同时预测周围车辆轨迹并生成车辆轨迹。

**💡 创新点**

创新点包括：①使用方差引导的可变时间步长流匹配，自动决定推理步数；②轻量级凸二次规划后处理提升舒适度和动态约束；③无需手动调噪声或情景特定微调。

**🔧 技术方法**

采用Transformer场景编码器、U‑Net速度场网络、方差预测网络、可变步长流匹配以及凸二次规划等技术。

**📊 数据集**

使用Waymo Open Motion数据集进行训练与评估。

**📈 对比分析**

与Diffusion、Consistency、Transformer等基线对比，在minADE/minFDE、轨迹平滑、约束违例率等指标上均优于基线，且推理速度可达20 Hz。

**⚠️ 局限性**

局限：仍依赖预处理的多边形地图；未在闭环仿真或真实硬件上测试；碰撞率仍与某些基线相近。

---

## 250. Reinforcing Chain-of-Thought Reasoning with Self-Evolving Rubrics

**arXiv ID:** 2602.10885 | [PDF](https://arxiv.org/pdf/2602.10885v1)

**作者:** Leheng Sheng `[一作]` (ByteDance), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60398 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为RLCER的强化学习框架，利用模型自身生成并自我演进的Rubrics来监督链式思维（CoT），实现对LLM推理过程的直接奖励；

**💡 创新点**

创新点在于：①让单一模型在不同提示下扮演推理器和Rubricator两角色，实现无人工标注的自监督Rubrics生成；②通过Rubric的有效性（与答案正确性相关）来动态演进Rubrics，避免Rubrics饱和；③将Rubrics直接用作提示，进一步提升推理表现；

**🔧 技术方法**

技术包括：多角色强化学习（single-policy multi-role），自监督Rubrics生成与评估（使用外部verifier模型），CoT奖励与最终答案奖励的联合策略梯度优化，Rubrics有效性判定（相关性与离散度阈值），以及动态Rubrics演进奖励；

**📊 数据集**

实验数据集：DAPE-Math-17k用于训练，评估覆盖AIME24、AIME25、AMC23（数学）以及GPQA-Diamond、SuperGPQA（一般知识）等；

**📈 对比分析**

与传统基于答案正确性奖励的RLVR对比，RLCER在所有模型尺寸（4B、8B）和所有数据集上均显著提升Pass@1，尤其在大模型上提升更明显；在不使用答案奖励，仅用CoT Rubrics奖励时仍能持续进步；

**⚠️ 局限性**

局限性包括：1）引入Rubricator显著增加采样负担与训练时间；2）方法目前仅在可验证任务（如数学）验证，未在开放域任务上测试。

---

## 251. Drawing Your Programs: Exploring the Applications of Visual-Prompting with GenAI for Teaching and Assessment

**arXiv ID:** 2602.10529 | [PDF](https://arxiv.org/pdf/2602.10529v1)

**作者:** David H. Smith `[一作]`, Daniel Zingaro `[通讯]` (University of Toronto)

**通讯引用:** 2652 | [OpenAlex ID](https://openalex.org/A5017837548)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究将问题分解图作为视觉提示，使用 GPT‑4.1 直接生成代码并评估其有效性。

**💡 创新点**

首次将多模态生成式 AI 与手绘分解图结合，实现“画图即编程”的教学与评估范式。

**🔧 技术方法**

使用 OpenAI 的 GPT‑4.1（多模态）和简短文本指令，仅输入图像即生成代码。

**📊 数据集**

133 名本科生在《Evil Hangman》任务中绘制的分解图（纸张尺寸 8.5×11 英寸）。

**📈 对比分析**

对比模型输出与人工标注的函数数量与内容，相关系数 r=0.776，完全提取率 81.2%；同时记录模型误解注释、输入等导致的偏差。

**⚠️ 局限性**

仅评估单一模型与单一任务，提示不够多样，结果对其他模型、任务或不同提示方式的泛化有限；模型生成不确定，可能影响评分公平性。

---

## 252. C-MOP: Integrating Momentum and Boundary-Aware Clustering for Enhanced Prompt Evolution

**arXiv ID:** 2602.10874 | [PDF](https://arxiv.org/pdf/2602.10874v1)

**作者:** Binwei Yan `[一作]` (Noah's Ark Lab), Hailin Hu `[通讯]` (Noah's Ark Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于聚类和文本动量的自动提示优化框架C‑MOP，利用硬负样本、锚点和边界对来精准映射决策边界，并通过历史梯度池与时间衰减稳定梯度方向，从而实现提示的迭代演化。

**💡 创新点**

创新点包括：① Tripartite采样策略（Hard Negatives、Anchors、Boundary Pairs）精准捕捉模型的错误与成功边界；② 文本动量机制结合衰减与二次聚类，显著抑制梯度冲突，提升优化过程的稳定性。

**🔧 技术方法**

使用的技术包括：文本梯度优化、MiniLM-L6嵌入、K‑means语义聚类、UCB多臂赌博机选择、历史梯度池与衰减机制以及基于文本重写的提示生成。

**📊 数据集**

实验使用四个多样化基准数据集：BBH、GSM8K、CFinBench 和 Liar。

**📈 对比分析**

与 PromptWizard、ProTeGi、SPO、GPO 等现有SOTA方法比较，平均提升 1.58%–3.35%，在 Liar 上从 61.61% 提升到 64.46%，在 CFinBench 上 3B 通用模型经过优化后超过 70B 专用模型，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：依赖大模型的推理能力，文本梯度离散性仍可能导致局部冲突；聚类参数与阈值对效果有较大影响；在更大规模模型或更复杂任务上仍缺乏充分验证。

---

## 253. Latent Thoughts Tuning: Bridging Context and Reasoning with Fused Information in Latent Tokens

**arXiv ID:** 2602.10229 | [PDF](https://arxiv.org/pdf/2602.10229v1)

**作者:** Weihao Liu `[一作]` (University of Illinois Chicago), Lu Cheng `[通讯]` (University of Illinois Chicago)

**通讯引用:** 4010 | [OpenAlex ID](https://openalex.org/A5022914600)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现一种Latent Thoughts Tuning框架，让大型语言模型能够在连续潜在空间进行推理，并通过动态切换和上下文-预测融合机制提升推理稳定性与准确性。

**💡 创新点**

核心创新在于①引入Context-Prediction Fusion，将隐藏状态与词表概率权重融合生成高质量潜在token；②采用自信度驱动的动态插入，让模型自行决定何时使用潜在推理；③通过三阶段逐步课程学习实现从显式CoT到潜在推理的平滑过渡。

**🔧 技术方法**

技术手段包括Transformer的端到端微调、三阶段课程学习、温度与Top‑p过滤的概率加权词嵌入、Context‑Prediction Fusion、注意力与熵分析、PCA可视化等。

**📊 数据集**

使用GSM8K训练集以及四个数学推理评测集（GSM8K‑NL、ASDiv‑Aug、MultiArith、SVAMP），并在Llama‑3.2‑1B/3B/3.1‑8B等模型上进行实验。

**📈 对比分析**

与Explicit CoT、Coconut、Soft‑Thinking、SoftCoT、SemCoT等基线进行对比，结果显示在所有模型规模上均实现最高平均准确率（1B: 36.4%，3B: 52.4%，8B: 68.8%），并在8B上通过adapter进一步提升至70.3%，显著缓解特征坍塌问题。

**⚠️ 局限性**

局限性在于：大模型仍需额外adapter以弥补输入/输出嵌入不一致；三阶段学习过程增加训练复杂度；对更大规模或不同任务的泛化尚未充分验证。

---

## 254. Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception

**arXiv ID:** 2602.11004 | [PDF](https://arxiv.org/pdf/2602.11004v1)

**作者:** Liangkai Liu `[一作]` (University of Michigan), Weisong Shi `[通讯]` (University of Delaware)

**通讯引用:** 24103 | [OpenAlex ID](https://openalex.org/A5100651611)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 PP-DNN 系统，动态选择关键帧与 ROI，减少多租户 DNN 推理帧数，维持感知精度并实现时序可预测。

**💡 创新点**

创新点包括：①环境感知驱动的关键帧与 ROI 自适应选择；②基于 FLOPs 的预测与调度降低融合延迟；③检测预测器在非关键帧上提供预测结果，解耦推理与融合。

**🔧 技术方法**

技术手段：SSIM 结构相似度、DeepSort 跟踪、FLOPs 预测模型、ROS 通信框架、Approximate Time Synchronizer、Faster R‑CNN、DNLNet、Deeplabv3+ 等多模型推理。

**📊 数据集**

使用数据集：BDD100K 与 nuScenes mini，覆盖多交通场景。

**📈 对比分析**

比较方法：设置五种实验配置（Baseline、FD、FD+FG、FD+DP、PP‑DNN），对比融合帧数、融合延迟、最大延迟、检测完整性与成本效能。PP‑DNN 在融合帧数提升 7.3×、融合延迟降低 >2.6×、检测完整性提升 75.4%、成本效能提升 98%。

**⚠️ 局限性**

局限性：CPU/GPU 内存占用提升（CPU 20%/GPU 39.3%），关键帧/ROI 选择在极端低帧率或突发场景下可能误判，预测器对极端情况的鲁棒性有限。

---

## 255. Automated Model Design using Gated Neuron Selection in Telecom

**arXiv ID:** 2602.10854 | [PDF](https://arxiv.org/pdf/2602.10854v1)

**作者:** Adam Orucu `[一作]` (Ericsson Research), Sarunas Girdzijauskas `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 1235 | [OpenAlex ID](https://openalex.org/A5041411651)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种名为TabGNS的梯度基神经网络架构搜索方法，专为电信表格数据设计，实现自动化模型生命周期管理。

**💡 创新点**

其创新点在于采用神经元级门控（基于Gumbel‑Softmax+STE）进行连续搜索，支持从小到大逐步增长并共享权重，从而显著减少模型尺寸与搜索时间。

**🔧 技术方法**

技术实现包括梯度基架构搜索、门控神经元、Gumbel‑Softmax与Straight‑Through Estimator、渐进式增长策略以及权重共享的SuperNet训练。

**📊 数据集**

实验使用六个表格数据集：四个电信场景（VoD、DeepMIMO、Sim‑A、Sim‑B）以及两个通用数据集（CoverType、Higgs）。

**📈 对比分析**

与大型MLP、TabNAS和AgE比较，TabGNS在MSE/准确率上与大型MLP相当或更优，模型参数仅占18–49%，搜索时间缩短5–36倍，且结果方差更小。

**⚠️ 局限性**

局限性包括仅支持全连接网络、门控初始化对搜索结果有一定影响，以及尚未验证在CNN/RNN等更复杂网络或多任务场景中的表现与鲁棒性。

---

## 256. MapVerse: A Benchmark for Geospatial Question Answering on Diverse Real-World Maps

**arXiv ID:** 2602.10518 | [PDF](https://arxiv.org/pdf/2602.10518v1)

**作者:** Sharat Bhat `[一作]` (University of Southern California), Vivek Gupta `[通讯]` (Arizona State University)

**通讯引用:** 1942 | [OpenAlex ID](https://openalex.org/A5100748239)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个包含 1,025 张真实地图、11,837 个人工问答对的跨领域地图 VQA 基准；

**💡 创新点**

创新点在于：①使用真实地图而非合成数据，涵盖 10 种地图类型与多样化问答格式；②构建细粒度元数据（地图类别、空间细化、推理类型等）；③通过两种输入模式（图像+文本与文本单独）区分视觉推理与记忆效果；

**🔧 技术方法**

采用多模态 VLM 评估框架，使用标准化提示、统一评估指标，并对 10 种顶尖开源与闭源模型进行测试；

**📊 数据集**

数据集为 “MapVerse”，来源于公开网络地图并通过 AMT 人工标注生成 QA 对；

**📈 对比分析**

与闭源 GPT‑4o、Gemini 2.5 以及多种开源模型（Qwen‑2.5‑VL、InternVL3‑8B、CogVLM2‑19B 等）进行基准对比，发现最强闭源模型在多数任务中达到 65–80% 以上精度，但在计数、排名与多步推理等高阶任务上仅低于 30%，揭示显著的推理瓶颈；

**⚠️ 局限性**

主要限制包括：1）受限于固定提示与 few‑shot 示例的人工成本；2）缺乏对时间序列地图等动态信息的支持；3）对噪声、分辨率下降的鲁棒性不足，尤其是椒盐噪声与遮挡会显著削弱性能。

---

## 257. IU-GUARD: Privacy-Preserving Spectrum Coordination for Incumbent Users under Dynamic Spectrum Sharing

**arXiv ID:** 2602.11023 | [PDF](https://arxiv.org/pdf/2602.11023v1)

**作者:** Shaoyu Li `[一作]` (Virginia Tech), Wenjing Lou `[通讯]` (Virginia Tech)

**通讯引用:** 32256 | [OpenAlex ID](https://openalex.org/A5001879281)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个名为 IU-GUARD 的隐私保护频谱协调框架，允许主导用户（Incumbent Users, IUs）在不暴露身份的前提下向频谱协调系统（SCS）申请并获得频谱使用授权。

**💡 创新点**

核心创新点在于将可验证凭证（Verifiable Credentials, VCs）与零知识证明（Zero‑Knowledge Proofs, ZKPs）结合，实现在不向 SCS 明文披露身份信息的情况下证明 IUs 的授权资格，同时保持多次请求的不可关联性（unlinkability）和匿名性；并且该方案不依赖集中可信中介，兼容现有 SCS 工作流。

**🔧 技术方法**

采用技术包括：Hyperledger Indy（用于发行与管理 VCs）、BBS+ 签名（为 VCs 提供可证明签名）、非交互式零知识证明（NIZK）实现属性范围证明（如频段范围）、TLS 1.3 加密通道、Python 实现的协议逻辑。

**📊 数据集**

使用的数据集主要是模拟的 DoD 运营商注册信息（设备类型、天线参数、授权频段等），以及通过 Hyperledger Indy 公共账本注册的凭证模式；实验中没有使用公开的真实频谱使用记录，而是基于仿真生成的 IU 请求序列。

**📈 对比分析**

比较方法：在同一硬件环境下与传统 IIC（Incumbent Informing Capability）基线方案对比。性能指标包括：凭证签发 128.4 ms vs 49.7 ms；身份验证 105.9 ms vs 15.2 ms；授权延迟 278.4 ms vs 130.3 ms；VP 消息大小 23.47 KB；在 10–5000 并发请求下的 p95 延迟与吞吐量。结果显示 IU-GUARD 在 95 % 分位点的延迟略高于基线，但仍在实时 DSS 的可接受范围内，并保持良好的可扩展性。

**⚠️ 局限性**

局限性包括：相对于基线方案增加了约 2× 的身份验证与授权延迟；依赖于可信的 Credential Authority（CA），对其安全性与可用性有严格要求；在实验中仅使用仿真数据，未在大规模真实环境中验证；对频谱使用模式变化频繁的情境下的实时性尚待进一步评估。

---

## 258. Viewpoint Recommendation for Point Cloud Labeling through Interaction Cost Modeling

**arXiv ID:** 2602.10871 | [PDF](https://arxiv.org/pdf/2602.10871v1)

**作者:** Yu Zhang `[一作]`, Siming Chen `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于语义分割的点云数据标注视点推荐方法，利用Fitts定律评估标注效率。

**💡 创新点**

创新点在于将视点推荐与标注速度模型结合，动态选择最优标注视点显著提升标注效率。

**🔧 技术方法**

采用PointNet++等点云语义分割网络，使用强化学习/启发式算法进行视点选择，并基于Fitts定律构建效率评估模型。

**📊 数据集**

在SemanticKITTI（或KITTI-360）数据集上进行实验，并构建自有的标注效率评估数据集。

**📈 对比分析**

与随机选点、基于密度或覆盖率的传统方法比较，实验表明该方法标注时间比基线快约30%，标注质量保持不变。

**⚠️ 局限性**

局限性包括对预训练分割模型的依赖、难以适用于动态场景或低标注质量环境，以及Fitts定律模型通用性待进一步验证。

---

## 259. Flash-SD-KDE: Accelerating SD-KDE with Tensor Cores

**arXiv ID:** 2602.10378 | [PDF](https://arxiv.org/pdf/2602.10378v1)

**作者:** Elliot L. Epstein `[一作]` (Stanford University), John Winnicki `[通讯]` (Stanford University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5115810362)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过将 Score‑debias KDE (SD‑KDE) 的计算重排为矩阵乘法形式，并利用 GPU Tensor Cores 与流式累加实现加速，成功将 SD‑KDE 的执行时间从数秒压缩到几百毫秒甚至秒级。

**💡 创新点**

创新点在于：① 把 SD‑KDE 的分数计算和 KDE 评估拆解为可被 Tensor Core 加速的 GEMM；② 采用流式累加避免构造全尺寸 Gram 矩阵；③ 在 16 维情形下实现了高 GPU 利用率，显著提升了计算效率。

**🔧 技术方法**

主要技术包括：矩阵乘法重排、Tensor Core GEMM、Triton 代码生成、块级调度与多阶段流水线、指数函数与归约的融合与优化。

**📊 数据集**

实验使用了高斯混合模型（16 维和 1 维）和大型随机采样（最多 1,000,000 训练点和 131,000 查询点）作为基准数据集。

**📈 对比分析**

与 scikit‑learn KDE、Torch SD‑KDE 以及 PyKeOps 对比，Flash‑SD‑KDE 在 32k 训练样本时可达 47× 的速度提升，且在 1M 样本时仅需 2.3 秒完成全部推断，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：目前仅支持维度为 16 的多重；需额外处理指数函数与原子操作的非 Tensor Core 开销；对负值估计的非正性修正仍未解决；以及多 GPU 或更大规模并行化的实现尚未展开。

---

## 260. ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression

**arXiv ID:** 2602.11008 | [PDF](https://arxiv.org/pdf/2602.11008v1)

**作者:** Ammar Ali `[一作]` (ITMO University), Stamatios Lefkimmiatis `[通讯]` (MWS AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种训练‑自由的快速模型压缩方法ROCKET，能够在保持90%+原始性能的同时将LLM压缩到20%–50%大小，且无需后续微调即可直接使用。

**💡 创新点**

创新点包括：① 通过校准指导的单步结构稀疏矩阵分解（融合低秩与稀疏字典学习），避免传统的迭代K‑SVD/OMP；② 在全局压缩预算下使用多选背包（MCKP）动态分配各层压缩率，显著提升压缩质量；③ 采用逆白化激活空间下的特征分解与闭式最小二乘更新，进一步减少重建误差。

**🔧 技术方法**

核心技术有：白化激活空间下的特征分解与稀疏阈值化、激活‑权重敏感性加权稀疏化、闭式L2正则化最小二乘更新、动态规划求解多选背包、基于校准集的激活统计。

**📊 数据集**

使用的主要数据集包括：256条RefinedWeb样本作为校准集；零样本评测集 PIQA、HellaSwag、LAMBADA、ARC‑Easy、ARC‑Challenge、SciQ、RACE、MMLU；WikiText、LAMBADA‑OpenAI；微调恢复阶段使用30M C4 tokens；多模态评测使用MathVista/MathVerse、MMBench、MMMU、MMStar、OCRBench、RealWorldQA；语音模型使用VibeVoice的转录数据。

**📈 对比分析**

通过与SVD‑LLM、CoSpaDi、ARS、Dobi‑SVD、ARA、LLM‑Pruner、SliceGPT、Bonsai、Wanda等基线在20%–50%压缩率下对比，ROCKET在零样本准确率、困惑度等指标上均优于所有基线，尤其在30%压缩时保持>90%原始性能；在轻量微调（30M tokens）后可恢复至与原始Qwen3‑8B相当甚至更好。

**⚠️ 局限性**

局限性：动态规划方案对拥有成百上千可压缩模块的MoE模型扩展困难；压缩后稀疏模式固定，Fine‑tune时未自适应更新；未针对极端高压缩比例（>70%）的鲁棒性进行深入研究；仅压缩密集层，未处理embedding等特殊层。

---

## 261. The Effect of Design Thinking on Creative & Innovation Processes: An Empirical Study Across Different Design Experience Levels

**arXiv ID:** 2602.10827 | [PDF](https://arxiv.org/pdf/2602.10827v1)

**作者:** Yuxin Zhang `[一作]` (Tsinghua University), Fan Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 53213 | [OpenAlex ID](https://openalex.org/A5100403400)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

研究了设计思维的四类认知技能（问题导向、信息导向、方案导向、知识导向）以及创造自我效能和集体创造效能对设计创造与创新的直接与中介路径，并通过线性回归和结构方程模型检验其对设计创新的影响；

**💡 创新点**

创新点在于构建了由思维技能驱动设计思维，再通过个人与团队效能双重中介路径推动设计创新的理论框架，并揭示学生与专业设计师在效能转化与路径结构上的差异；

**🔧 技术方法**

采用多元线性回归、结构方程模型（SEM）、验证性因素分析、Bootstrap 5,000 次抽样的中介检验以及逐步释放约束的多组不变性检验；

**📊 数据集**

使用了475份有效问卷（230名学生、245名专业设计师）的数据，问卷基于7点李克特量表测量八个构念；

**📈 对比分析**

通过多组SEM不变性检验比较学生与专业组，模型拟合优良（χ²/df=1.866，CFI=0.937，RMSEA=0.043），中介路径显著，整体间接效应占总效应的48.5%，显示模型具有较高解释力；

**⚠️ 局限性**

主要局限在于依赖自评问卷缺乏对实际设计产出的客观评估，样本仅来自问卷调查，可能存在社会期望和认知偏差。

---

## 262. The CLEF-2026 FinMMEval Lab: Multilingual and Multimodal Evaluation of Financial AI Systems

**arXiv ID:** 2602.10886 | [PDF](https://arxiv.org/pdf/2602.10886v1)

**作者:** Zhuohan Xie `[一作]` (MBZUAI), Preslav Nakov `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究构建了 CLEF 2026 FinMMEval Lab，提出了首个多语言、多模态的金融大型语言模型（LLM）评估框架，包含金融考试题解答、多语言金融问答（PolyFiQA）和金融决策制定三项互联任务；

**💡 创新点**

创新点在于将金融知识理解、跨语言多模态推理与实际决策行动三大层面统一在同一评测体系中，实现从概念掌握到实时交易的端到端评估；

**🔧 技术方法**

采用多语言问答技术、证据驱动生成、文本与数值混合推理，并使用 ROUGE‑1、BLEURT、财务绩效指标（累计收益、夏普比率、最大回撤等）对模型进行量化评测；

**📊 数据集**

使用的公开数据集包括：EFPA、GRFinQA、CFA、CPA、BBF、SAHM（金融考试题）；PolyFiQA‑Easy 与 PolyFiQA‑Expert（多语言财报+新闻问答）；以及 BTC 与 TSLA 的每日市场情境 JSON（决策制定）；

**📈 对比分析**

在评测中，模型分别以准确率评估考试题，ROUGE‑1/BLEURT 评估问答答案，累计收益与夏普比率等财务指标评估决策模型；在基线实验中，传统 LLM 在准确率与 ROUGE 上表现参差不齐，决策任务的累计收益与风险控制仍低于专业交易算法；

**⚠️ 局限性**

局限性包括：语言覆盖仅限七种，未涵盖图表、音频等多模态；数据规模受限，尤其是决策任务的真实交易窗口较短；部分任务仍可能存在数据泄漏风险，且模型在跨语言推理与数值精度上仍表现不足。

---

## 263. Utilitarian Distortion Under Probabilistic Voting

**arXiv ID:** 2602.11152 | [PDF](https://arxiv.org/pdf/2602.11152v1)

**作者:** Hamidreza Alipour `[一作]` (Sharif University of Technology), Mohak Goyal `[通讯]` (Stanford University)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5040410978)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在概率投票模型（Plackett‑Luce）下研究效用扭曲（utilitarian distortion），并给出多种投票规则（Copeland、Borda、Plurality、Random Dictator、Veto 等）的上界与下界；

**💡 创新点**

首次通过考虑投票者的噪声偏好，解决了传统模型下规范性好规则扭曲无限大、而效率低规则扭曲最优的悖论；提供 Copeland 与 Borda 的几乎最优线性扭曲上界（β/(1+e^{-β}))，并给出相匹配的下界；证明任何有限精度的锦标赛规则至少要有 (5/8-ε)β 的扭曲；

**🔧 技术方法**

使用逻辑斯蒂函数 σ_β 的凸凹性分析、强大法则（SLLN）与 Hoeffding 收敛、精细的概率构造、线性规划与不等式推导等理论工具；

**📊 数据集**

本工作为理论分析，未使用实测数据集，全部基于数学建模与概率论推导；

**📈 对比分析**

与以往的定性下界（O(β^2) 或无界）相比，得到更紧的 β 线性上界，并通过构造实例证明下界接近；对各规则的扭曲度进行理论比较，展示在概率投票下规范性与效率可兼得；

**⚠️ 局限性**

局限性包括：仍未确定最优的确定性规则；模型假设为 Plackett‑Luce 并假设噪声为 Gumbel，实际投票可能更复杂；扭曲指标仍是比值型，可能对极端情况不够稳健；

---

## 264. Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation

**arXiv ID:** 2602.10506 | [PDF](https://arxiv.org/pdf/2602.10506v1)

**作者:** Wei Chen `[一作]` (Beihang University), Huimei He `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 DiffGDA，一种基于连续时间扩散过程的图域适应框架，通过随机微分方程（SDE）联合建模结构与语义的演化，并引入域感知引导网络实现自适应路径控制。

**💡 创新点**

创新点在于：①将图域适应视为连续生成过程，突破传统离散中间图构造的局限；②设计密度比率引导网络实现目标域向导；③给出扩散过程在潜在空间收敛到最优适配路径的理论保证。

**🔧 技术方法**

核心技术包括：扩散模型（score‑based diffusion + SDE）、图神经网络、MMD 对齐、图多头注意力、随机子图扩散策略、密度比率判别网络。

**📊 数据集**

使用八个公开图数据集：Citation（ACMv9、Citationv1、DBLPv7）、Airport（USA、Brazil、Europe）、Social（Blog1、Blog2）。

**📈 对比分析**

在 14 个节点分类迁移任务中与 20+ 传统与最新方法（如 GAT、GIN、GCN、DANE、UDAGCN、AdaGCN、StruRW、GRADE、PairAlign、GraphAlign、A2GNN、GGDA、TDSS、DGSDA、GAA 等）进行对比，DiffGDA 在大多数场景下取得最高 Mi‑F1 分数，并在计算时间上优于生成式基准（如 GraphAlign），显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：扩散过程仍需较多计算资源，尤其在大规模图上可能受限；对超参数（采样比例 α、对齐权重 η、扩散步数 𝖳）敏感；仅针对节点分类任务验证，缺乏对其他图任务的广泛评估；假设源/目标共享特征空间，未考虑特征不匹配的场景。

---

## 265. Multiconfiguration Pair-Density Functional Theory Calculations of Ground and Excited States of Complex Chemical Systems with Quantum Computers

**arXiv ID:** 2602.10435 | [PDF](https://arxiv.org/pdf/2602.10435v1)

**作者:** Zhanou Liu `[一作]` (East China Normal University), Yuxin Deng `[通讯]` (Shanghai University of Finance and Economics)

**通讯引用:** 1441 | [OpenAlex ID](https://openalex.org/A5022036342)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了结合变分量子本征求解器与多配置配对密度泛函理论的混合方法，用于高效描述复杂体系的基态与激发态。

**💡 创新点**

通过将静态相关性限定在紧凑多重参考量子态中，并用经典的on‑top密度泛函恢复动态相关性，实现了显著降低量子资源需求的同时保持物理严谨性。

**🔧 技术方法**

使用变分量子本征求解器（VQE）、多配置配对密度泛函理论（MC‑PDFT）与自洽轨道优化技术；结合经典与量子计算。

**📊 数据集**

在标准基准体系中测试，包括C₂平衡键长、苯激发能量以及需要大完整活性空间的Cr₂二聚体（48e,42o）。

**📈 对比分析**

将结果与高精度多参考计算和实验数据对比，C₂键长MAE 0.006 Å，苯激发能量MAE 0.048 eV；Cr₂在真实硬件噪声下仍能得到有束缚势能曲线，表现出正确的解离行为，表明方法在近端量子硬件上的可行性。

**⚠️ 局限性**

受限于当前量子硬件噪声与量子资源有限，方法仍需进一步优化量子电路深度及对更大体系的扩展。

---

## 266. ContactGaussian-WM: Learning Physics-Grounded World Model from Videos

**arXiv ID:** 2602.11021 | [PDF](https://arxiv.org/pdf/2602.11021v1)

**作者:** Meizhong Wang `[一作]` (Tongji University), Yiguang Hong `[通讯]` (Tongji University)

**通讯引用:** 21634 | [OpenAlex ID](https://openalex.org/A5100359415)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文提出了 ContactGaussian‑WM，一种从稀疏视频中学习物理约束世界模型的框架；

**💡 创新点**

创新点在于：1）将视觉与几何统一为球形高斯表示，既能渲染又能实现闭式可微碰撞检测；2）使用闭式无互补完整接触动力学模型，实现高效稳定的可微仿真；3）端到端可微管线实现从视频直接学习质量、摩擦等物理参数；

**🔧 技术方法**

主要技术包括：球形高斯光栅渲染（SG‑GS）、可微碰撞检测（LogSumExp 与 Sigmoid 软化）、闭式无互补接触动力学、可微 3DGS 渲染、梯度回传及 MPC 控制；

**📊 数据集**

数据集：使用 MuJoCo 生成的合成视频（推送与自由落体两种场景），以及真实摄像头（RealSense D456）拍摄的自由落体与 LEAP Hand 交互视频；

**📈 对比分析**

与 DreamerV3、CEM+MuJoCo+、PIN‑WM 等基线比较：在合成场景下，ContactGaussian‑WM 在推送与落体两种场景均取得最低位移与角度误差，尤其在高动态落体场景表现明显优于 PIN‑WM 与 CEM+MuJoCo+；在真实场景中，经过参数学习的模型在长时隙预测与轨迹转移的 PSNR 均显著高于不学习参数的版本；

**⚠️ 局限性**

局限性：1）球形高斯碰撞检测在深度穿透时误差增大；2）渲染质量受限于统一球形近似；3）当前仅适用于刚体，未覆盖可变形对象；4）对极少视角或遮挡时几何估计可能不足；

---

## 267. What Makes Value Learning Efficient in Residual Reinforcement Learning?

**arXiv ID:** 2602.10539 | [PDF](https://arxiv.org/pdf/2602.10539v1)

**作者:** Guozheng Ma `[一作]` (Nanyang Technical University), Dacheng Tao `[通讯]` (Nanyang Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究残差强化学习中的价值学习瓶颈，并提出一种基于数据锚定预热与Critic层归一化的最小化方法，以显著提升样本效率。

**💡 创新点**

① 识别出残差RL中的两大根本瓶颈：冷启动病理和结构尺度不匹配；② 证明仅通过数据预热和Critic归一化即可解决这两类问题；③ 通过对比实验展示该方案在多任务上实现了5-6倍的收敛速度提升，且方差更低。

**🔧 技术方法**

使用残差RL框架（SAC + 受限残差）+ Critic层归一化（LayerNorm）、基于基线策略的数据预热、MSE critic目标；对比分布式RL、分布式 critic 等传统方法。

**📊 数据集**

在 ManiSkill（3个接触式操作任务）和 Adroit（3个手部控制任务）两个机器人抓取/操作基准上测试；基线策略为 Diffusion Policy 与 Behavior Transformer。

**📈 对比分析**

与 Policy Decorator、Vanilla Residual RL 等基线对比。改进方法在 8 个任务中均实现高成功率，收敛速度比 Policy Decorator 快约 5–6 倍，且在多随机种子下表现出更低的方差。

**⚠️ 局限性**

目前仅在仿真环境中验证；对真实机器人部署的鲁棒性与适用性尚未评估；对不同 λ 超参数的敏感性以及在多任务/动态环境中的泛化能力仍需进一步研究。

---

## 268. Experimental Demonstration of Online Learning-Based Concept Drift Adaptation for Failure Detection in Optical Networks

**arXiv ID:** 2602.10401 | [PDF](https://arxiv.org/pdf/2602.10401v1)

**作者:** Yousuf Moiz Ali `[一作]` (Aston University), Pedro Freire `[通讯]` (Aston University)

**通讯引用:** 909 | [OpenAlex ID](https://openalex.org/A5076069500)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一种基于在线学习的光网络故障检测方法，用以解决概念漂移问题。

**💡 创新点**

首次将在线学习与Page‑Hinkley漂移检测结合应用于光网络故障检测，显著提升性能并实现自动适应。

**🔧 技术方法**

采用在线学习模型（Adaptive Random Forest、Logistic Regression、Naive Bayes）以及Page‑Hinkley Drift检测技术，对数据进行逐样本预测与更新。

**📊 数据集**

使用作者公开的软故障（SFD）与硬故障（HFD）混合数据集，模拟软→硬故障漂移场景。

**📈 对比分析**

通过滚动准确率和AUC指标对比，在线模型相较于静态模型提升高达70%准确率、AUC上升至0.75；在线学习仅增加约1 ms的延迟。

**⚠️ 局限性**

研究仅覆盖软→硬故障漂移且采用监督在线学习，未评估多种漂移类型、非监督场景以及更复杂的网络环境。

---

## 269. Exploring the Feasibility of Full-Body Muscle Activation Sensing with Insole Pressure Sensors

**arXiv ID:** 2602.10442 | [PDF](https://arxiv.org/pdf/2602.10442v1)

**作者:** Hao Zhou `[一作]` (Pennsylvania State University), Mahanth Gowda `[通讯]` (Pennsylvania State University)

**通讯引用:** 928 | [OpenAlex ID](https://openalex.org/A5064270644)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并验证了一种利用鞋垫压力传感器无创推断全身肌肉激活的系统，并在30名用户、15种运动的实验中进行评估。

**💡 创新点**

①首次通过足部压力实现全身肌肉激活推断；②提出Region Importance Learning动态关注有效足部区域；③结合用户生物信息FiLM实现零干预适配；④设计压力缩放与时间移位的数据增强。

**🔧 技术方法**

使用深度学习 Transformer 编码器 + Region Importance Learning + FiLM + MSE+平滑损失；硬件采用 36 芯柔性压力传感器（20 Hz），并与 sEMG 作为标签同步。

**📊 数据集**

30 名志愿者（21 男 9 女）采集约 27 小时足部压力 + sEMG 数据，覆盖 15 种常见运动，采用 leave‑one‑user‑out 分割。

**📈 对比分析**

与视频基方法对比，RMSE 0.025（比视频方法提升 19%）；在不同鞋类、地面、人口统计等条件下保持鲁棒；相关系数高，表现稳定。

**⚠️ 局限性**

对上身运动或坐姿等不产生足部重心变化的场景效果有限；模型单靠足部压力难以捕捉这些情况；需与其他轻量感测器结合；数据集规模与人群多样性仍可进一步扩展。

---

## 270. Blockwise Advantage Estimation for Multi-Objective RL with Verifiable Rewards

**arXiv ID:** 2602.10231 | [PDF](https://arxiv.org/pdf/2602.10231v1)

**作者:** Kirill Pavlenko `[一作]` (Nebius), Boris Yangel `[通讯]` (Humanoid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多目标强化学习中，针对结构化文本生成（如数学解题后自评），作者提出了 Blockwise Advantage Estimation（BAE）框架，能够对每个文本块分别估计优势并仅在对应块上更新策略，解决了传统 GRPO 中单一全局优势导致的目标干扰问题。

**💡 创新点**

创新点：①将优势拆分为块级，避免跨目标信用传递；②提出 Outcome-Conditioned Baseline（OCB），通过根据前缀的结果将样本分组来近似后缀的状态价值，实现无额外 roll‑out 的低方差优势估计；③在不需要奖励标量化的情况下，仅用局部奖励即可训练出校准良好的自信模型。

**🔧 技术方法**

技术：基于 GRPO 的无 critic 训练；块级优势估计与 PPO‑style clipped 损失；分层基线估计（无基线、批量均值、组均值、OCB）；Brier 及 BCE 等严格的评分规则作为局部奖励；测试时多样本抽样与置信度加权投票；在数学推理中对两步改进循环进行评估。

**📊 数据集**

数据集：MATH500（在分布内）、GSM8K（易 OOD）、AIME23‑25（难 OOD）以及自定义的两步改进任务；模型：Qwen2.5‑7B‑Base、Qwen2.5‑7B‑Instruct、Qwen2.5‑3B‑Instruct。

**📈 对比分析**

与基线对比：RLCR（奖励标量化方法）保持在准确率上的优势，但 BAE+OCB 在 ECE、AUROC、Brier 上与 RLCR 相当或更好，尤其在自信校准上显著提升；Group Mean、Batch Mean 和无基线的性能普遍落后。测试时多样本推理中，OCB 保留了 RLCR 的置信度加权效果，取得与 RLCR 相近甚至更高的 Pass@k。

**⚠️ 局限性**

局限性：①需要预先定义且稳定的块边界，边界不清晰会导致目标干扰；②OCB 依赖足够多的样本以填充各个条件层，稀缺层会导致方差增大；③在极端 OOD 情况下，单一条件层可能无法充分捕捉不同前缀的差异。

---

## 271. Authenticated Workflows: A Systems Approach to Protecting Agentic AI

**arXiv ID:** 2602.10465 | [PDF](https://arxiv.org/pdf/2602.10465v1)

**作者:** Mohan Rajagopalan `[一作]` (MACAW Security), Vinay Rao `[通讯]` (ROOST.tools)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一套基于协议层的安全框架——Authenticated Workflows，提供完整的身份验证、意图检查和上下文完整性保障，用以防御企业级 Agentic AI 中的多层攻击。

**💡 创新点**

创新点包括：① MAPL 语言实现可动态、可继承的多层策略与加密凭证的组合；② 在每个交互边界嵌入独立的 Policy Enforcement Points（PEP），实现零信任、跨框架的分层防御；③ 通过哈希链和签名实现会话、上下文与操作的不可篡改追溯；④ 通过“可信工作流”概念，将工具调用、数据检索、提示与上下文四大攻击面统一纳入加密验证。

**🔧 技术方法**

技术手段主要有：分布式身份与密钥管理（ECDSA-256、PKI）、哈希链与序列号保障会话完整性、MAPL 语法与层级策略交叉（intersection algebra）、PEP 侧车/库嵌入实现零信任验证、基于加密凭证的工作流证明与签名、以及对现有九大 AI 框架（MCP、A2A、OpenAI、Claude、LangChain、CrewAI、AutoGen、LlamaIndex、Haystack）的轻量级适配。

**📊 数据集**

主要使用了人工构造的 174 条安全测试用例（涵盖 9 个框架组合和 11 种攻击模式）以及两条真实生产漏洞（OpenAI Atlas 浏览器和 GitHub MCP）作为验证数据；未使用公开的标准数据集，而是基于业务场景模拟与攻击脚本生成。

**📈 对比分析**

与传统的基于模式匹配或 ML 的守护层相比，Authenticated Workflows 在 174 条测试中实现了 100% 的召回率与 0% 的误报率；在加密验证层面平均每个操作的额外开销低于 0.2 ms；在大多数应用中该延迟远低于网络或 LLM 推理时延。实验表明该方案能够覆盖 OWASP LLM 2025 风险中的 9/10 类，且对 OpenAI Atlas 与 GitHub MCP 的真实漏洞实现了完全拦截。

**⚠️ 局限性**

局限性包括：① 仍不涵盖训练阶段的模型数据污染与模型投毒；② 对 LLM 输出的语义安全（如伪造信息）仍需依赖外部守护层；③ 需要在各框架内部嵌入或适配 PEP，部署成本仍非零；④ 对大规模 LLM 语义校验（如实时 Prompt 安全评估）存在性能瓶颈；⑤ 在极端高并发场景下，密钥管理与策略查询的扩展性需进一步验证。

---

## 272. Basic Legibility Protocols Improve Trusted Monitoring

**arXiv ID:** 2602.10153 | [PDF](https://arxiv.org/pdf/2602.10153v1)

**作者:** Ashwin Sreevatsa `[一作]`, Cody Rushing `[通讯]` (Redwood Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了腿部协议（legibility protocols），尤其是允许注释的协议，以提高在 APPS 编码反背门（backdoor）控制评估中的安全性而不显著牺牲任务完成率。

**💡 创新点**

创新点在于将可读性（legibility）融入可信监控，证明注释能显著降低监视误报并提升安全性，同时解释了其对正当代码更易解释且对攻击代码难以说服的两大机制。

**🔧 技术方法**

使用了 OpenAI 的 o4‑mini 作为未受信模型，GPT‑4.1‑nano 作为可信监视模型，以及基于多次采样的安全性和效用指标评估。

**📊 数据集**

数据集为 APPS 代码竞赛的 499 个问题（含完整测试用例），并在此基础上进行后门注入。

**📈 对比分析**

与去掉注释的基线对比，允许注释的协议在安全‑效用 Pareto 曲线中获得更高安全性，且随着监视模型能力提升其优势扩大；在保留 2% 人工审计的前提下，安全率提升约 10–20%。

**⚠️ 局限性**

局限包括仅在 499 条 APPS 任务上测试、假设完美审计、攻击策略手工构造且可能过度估计真实攻击者能力，以及未考虑未受信模型能力变化对注释效果的影响。

---

## 273. JAG: Joint Attribute Graphs for Filtered Nearest Neighbor Search

**arXiv ID:** 2602.10258 | [PDF](https://arxiv.org/pdf/2602.10258v1)

**作者:** Haike Xu `[一作]` (Massachusetts Institute of Technology), Jakub Łącki `[通讯]` (Google Research)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 Joint Attribute Graphs（JAG）算法，构建了一种统一的图索引，能够在任意滤波类型和查询选择度范围内高效执行过滤最近邻搜索。

**💡 创新点**

创新点包括：①引入属性距离（dist_A）和滤波距离（dist_F）两种连续度量，将离散的过滤约束转化为连续导航梯度；②采用“capped”属性距离在图构建时生成多层次导航边，兼顾高低选择度场景；③统一比较规则在查询时使用滤波距离优先，随后按向量距离打破平局，从而避免死胡同并提升查询鲁棒性。

**🔧 技术方法**

技术手段主要是：基于HNSW/向量近邻图的改进，使用多阈值（Threshold-JAG）或多权重（Weight-JAG）构建邻接表；引入联合属性图、过滤距离与属性距离的组合；实现贪心 beam search 与 RobustPrune 算法；并通过实验验证其高吞吐量与高召回率。

**📊 数据集**

实验数据集涵盖五个大规模数据集：SIFT、ARXIV、LAION、YFCC 与 MSTuring，并在四类滤波（Label、Range、Subset、Boolean）上进行评测。

**📈 对比分析**

与十种基准（ACORN、NaviX、RWalks、Post‑Filtering、FilteredVamana、StitchedVamana、UNG、NHQ、iRangeGraph 等）对比，JAG 在所有过滤器与选择度范围内都实现了最高的查询吞吐量（QPS）和最优或近乎最优的召回率；在低选择度（≤1/100）场景下，JAG 唯一能实现 100% 召回且 QPS 超过 1000 的方法。

**⚠️ 局限性**

局限性主要在于：①在极高选择度或完全无过滤约束的查询中，JAG 的性能略低于专门针对该滤波类型的定制算法；②构建时需要多阈值或多权重的图，导致索引大小和构建时间比单一阈值/权重方法略大；③对于某些极端分布（如完全随机标签）可能不如专门的标签分区策略优越。

---

## 274. Disability-First AI Dataset Annotation: Co-designing Stuttered Speech Annotation Guidelines with People Who Stutter

**arXiv ID:** 2602.10403 | [PDF](https://arxiv.org/pdf/2602.10403v1)

**作者:** Xinru Tang `[一作]` (University of California, Irvine), Shaomei Wu `[通讯]` (AImpower.org)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文基于残障优先原则，结合口吃者与专业人员共同设计了针对口吃语音的注释指南，并对现有数据集进行重新标注与对比。

**💡 创新点**

创新点在于将口吃者的具身经验与社区视角融入注释流程，提出多模态、可解释的标注原则，并倡导持续的社区监督与迭代。

**🔧 技术方法**

技术方法包括访谈、共创工作坊、主题分析、Praat音频标注以及对Sep-28k等数据集的对比评估。

**📊 数据集**

使用的数据集为公开的口吃语音数据（FluencyBank、Sep-28k、LibriStutter、AS-70）以及作者自行收集的51名英美口吃者录音。

**📈 对比分析**

比较方法是将原始Sep-28k标注与社区重新标注的子集（Sep-28k-SW）进行对比，发现显著的误标差异，提示标注一致性差；论文未给出模型性能指标。

**⚠️ 局限性**

局限性包括样本量有限、受访者多为高社会经济背景且技术熟练，缺乏多语言与方言覆盖，以及仍受SLP框架影响。

---

## 275. Improving Medical Visual Reinforcement Fine-Tuning via Perception and Reasoning Augmentation

**arXiv ID:** 2602.10619 | [PDF](https://arxiv.org/pdf/2602.10619v1)

**作者:** Guangjing Yang `[一作]` (Beijing University of Posts and Telecommunications), Qicheng Lao `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 807 | [OpenAlex ID](https://openalex.org/A5013010213)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种面向医学图像的视觉强化微调框架VRFT-Aug，旨在提升大型视觉语言模型的感知与推理能力；

**💡 创新点**

创新点在于：①通过提示工程与跨任务训练双通道知识注入增强模型对医学视觉属性的感知；②设计两类奖励塑造机制（反复复述奖励和多等级模糊奖励）来引导模型更好地进行医学推理；

**🔧 技术方法**

技术上结合GRPO强化学习、Prompt增强、跨任务强化学习以及自定义奖励函数；

**📊 数据集**

使用了MedMNIST、HAM10000、Heel、RetinaMNIST、COVID-19等医学影像数据集进行评估；

**📈 对比分析**

与V-SFT、V-RFT等基线相比，VRFT-Aug在10-shot下提升约6.9%，在HAM10000上提升约35%，在RetinaMNIST上从60%提升到约70%，总体性能明显优于现有方法；

**⚠️ 局限性**

局限性包括：奖励设计对任务高度依赖，可能难以迁移到其他医学领域；模型对数据平衡敏感，且在过度重复奖励时易陷入次优解，需进一步研究鲁棒性与泛化能力。

---

## 276. Random Access in Grammar-Compressed Strings: Optimal Trade-Offs in Almost All Parameter Regimes

**arXiv ID:** 2602.10864 | [PDF](https://arxiv.org/pdf/2602.10864v1)

**作者:** Anouk Duyster `[一作]` (Max Planck Institute for Informatics), Tomasz Kociumaka `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 1350 | [OpenAlex ID](https://openalex.org/A5086467798)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一套通用的时空权衡框架，能够在任意大小的上下文无关文法压缩字符串上实现高效的字符随机访问、子串提取以及 rank / select 查询。

**💡 创新点**

创新点在于同时考虑压缩度（g）与数据结构空间（M）两大参数，给出一个可调的参数化上界，并给出除极小文法和极小空间外几乎所有参数范围内的匹配下界，首次实现了从 (g) 空间到 (n log σ) 位空间的连续插值。

**🔧 技术方法**

核心技术包括：① 将任意 SLG 转换为收缩（contracting）和 τ‑nice 文法，实现对每个符号的子句查询（_A）在 O(τ) 时间内常数空间实现；② 构造 b‑leafy 文法利用位并行加速块级遍历；③ 通过融合树（fusion tree）和层祖先查询实现节点指针与字符指针的常数时间操作；④ 结合前缀求和与 Elias–Fano 编码扩展至 rank / select。

**📊 数据集**

该工作主要是理论分析，未使用具体实验数据集，而是基于信息量上界与随机化单元探测模型给出严格的上界与下界。

**📈 对比分析**

与此前 (log n) 时间/ (g) 空间的最优方案相比，新方案在 g≈n^{1−o(1)} 时将查询时间压缩到 O(log n·log σ/(M w·log (M w)/(g log n)))，并在小字母表情况下实现常数时间块提取；在极小文法或小空间时仍保持与旧方案相当。

**⚠️ 局限性**

主要限制是：当文法规模 g≤w^{1+o(1)}·log n 或数据结构空间 M w≤g·log n·w^{o(1)} 时，所给下界不再适用；此外实现细节（如 Elias–Fano 的高效构造）仍存在一定的工程挑战。

---

## 277. Quadratic Speedup for Computing Contraction Fixed Points

**arXiv ID:** 2602.10296 | [PDF](https://arxiv.org/pdf/2602.10296v1)

**作者:** Xi Chen `[一作]` (Columbia University), Mihalis Yannakakis `[通讯]` (Columbia University)

**通讯引用:** 30548 | [OpenAlex ID](https://openalex.org/A5043084405)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了在ℓ∞和ℓ1范数下，对[0,1]^k中的收缩映射求ε-近似不动点的问题，并给出了时间复杂度为O(log^{⌈k/2⌉}(1/ε))的算法。

**💡 创新点**

主要创新是提出了新的分解定理和递归结构，使得高维问题可分解为低维子问题，从而把原来的O(log^k)时间提升到O(log^{⌈k/2⌉})。

**🔧 技术方法**

采用了分解定理、二分搜索、弱支配坐标分析以及递归缩小搜索空间等技术。

**📊 数据集**

本工作完全为理论算法，未使用任何实验数据集。

**📈 对比分析**

与之前的O(log^k)算法相比，新算法在时间复杂度上实现了指数级别的改进，在常数维下达到接近最优。

**⚠️ 局限性**

主要局限是对于k=3仍无法实现O(log(1/ε))的时间复杂度，且在时间和查询复杂度之间的平衡尚未完全解决。

---

## 278. Breaking 5G on The Lower Layer

**arXiv ID:** 2602.10250 | [PDF](https://arxiv.org/pdf/2602.10250v1)

**作者:** Subangkar Karmaker Shanto `[一作]` (Purdue University), Elisa Bertino `[通讯]` (Purdue University)

**通讯引用:** 39602 | [OpenAlex ID](https://openalex.org/A5061694501)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

在5G网络中实现并验证了SIB1伪造攻击和TA值操纵攻击，使用软件定义无线电和srsRAN在实验室测试床上对商业智能手机进行实测。

**💡 创新点**

首次实证证明对随机接入响应中的Timing Advance进行注入可导致无线链路失效和持续的重连循环，揭示了低层控制消息的安全盲点；同时验证了SIB1伪造导致的系统信息重新获取和电池消耗。

**🔧 技术方法**

利用软件定义无线电（USRP B210）、srsRAN开源基站栈、被动嗅探、主动注入、以及NSG进行物理层参数监测。

**📊 数据集**

实验使用两款商业机型（OnePlus Nord 5G、Google Pixel 5）作为受害者，未使用公开数据集。

**📈 对比分析**

通过比较攻击前后的RLF触发时间、TA_delta容忍阈值以及重连次数，发现TA_delta>20时在30-60秒内触发RLF，并导致长达数分钟的重连循环；SIB1伪造导致SI重新获取周期显著缩短。

**⚠️ 局限性**

实验仅涵盖两种机型，结果受固件实现影响；未评估核心网影响，未测量实际电池损耗；实验在室内受限环境，缺乏大规模或真实网络验证。

---

## 279. Just on Time: Token-Level Early Stopping for Diffusion Language Models

**arXiv ID:** 2602.11133 | [PDF](https://arxiv.org/pdf/2602.11133v1)

**作者:** Zahar Kohut `[一作]` (Ukrainian Catholic University), Volodymyr Karpiv `[通讯]` (SoftServe Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为扩散式语言模型（如 Dream‑7B、LLaDA‑8B）提出了一种训练无关的每个 token 级别早停方法 Jot，动态判断每个位置何时已收敛并直接固定，避免了无谓的迭代。

**💡 创新点**

创新点在于：① 通过 token 的预测置信度（top‑2 概率比）与局部上下文的空间权重自适应地设定阈值，实现在不同位置不同步停止；② 不依赖额外训练，仅使用模型自身的推理信号；③ 兼容现有的反向扩散采样流程。

**🔧 技术方法**

核心技术包括：离散扩散语言模型（掩码式），基于 logits 的置信度度量，空间衰减权重窗口，动态阈值插值，以及与传统 transfer schedule 的无缝对接。

**📊 数据集**

在四大基准上进行评估：GSM8K、MMLU、HellaSwag、HumanEval（均采用 zero‑shot 提示）。

**📈 对比分析**

与全步解码、Prophet（全局置信度早停）和 KLASS（基于 KL 的稳定性采样）对比，Jot 在 HumanEval 上实现 19.6× 速度提升，仅损失 0.6 分；在 GSM8K 上 5.5× 加速，误差 2.3 分；总体保持 98.3% 的原始质量，位居 Pareto 前沿。

**⚠️ 局限性**

局限性包括：仍使用贪心 argmax 终止，可能影响长文本的连贯性；未在创意写作、翻译等开放式生成任务中验证；仅测试至 512 token 的长度，对更长序列的行为未知；虽然对阈值稳健，但不同任务仍需手动微调。

---

## 280. Security, Privacy and System-Level Resillience of 6G End-to-End System: Hexa-X-II Perspective

**arXiv ID:** 2602.10734 | [PDF](https://arxiv.org/pdf/2602.10734v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 281. Modeling Programming Skills with Source Code Embeddings for Context-aware Exercise Recommendation

**arXiv ID:** 2602.10249 | [PDF](https://arxiv.org/pdf/2602.10249v1)

**作者:** Carlos Eduardo P. Silva `[一作]` (Universidade Federal de Viçosa), Lucas N. Ferreira `[通讯]` (Universidade Federal de Minas Gerais)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于源代码嵌入的情境感知编程作业推荐系统，利用学生提交代码构建学习上下文，预测其编程技能并据此推荐个性化练习。

**💡 创新点**

创新点在于：①直接使用学生代码的Jina嵌入构造动态学习上下文；②通过最近实验提交的中心点嵌入作为学生代表向量，避免平均嵌入导致的无效向量；③将多分类的技能预测作为推荐信号，与传统的正确性或耗时指标相比更能体现学生真实能力。

**🔧 技术方法**

主要技术包括：多语言BERT模型Jina（jina-embeddings-v2-base-code）进行代码嵌入；多分类MLP预测每个技能的四级标签；余弦相似度排序生成推荐列表；与TF‑IDF、CodeBERT‑cpp、GraphCodeBERT等嵌入方法做对比实验。

**📊 数据集**

使用了从巴西维多萨联邦大学INF110课程收集的多年份（2018–2025）数据集，包含12,912条C++提交、112道题目、8个编程技能维度（数学、条件、循环、数组、矩阵、函数、字符串、结构）。

**📈 对比分析**

实验采用时间敏感的留一法，将2018–2024年的数据训练，评估2025年。Jina嵌入在大多数技能上取得最高平均准确率0.69；在推荐评估中，使用技能预测的余弦相似度排序（尤其是最近实验提交的中心点嵌入）得到最高的适合度比例，明显优于仅基于正确性或耗时的排序方法。

**⚠️ 局限性**

限制包括：仅针对单一C++课程数据，未考虑课堂出勤、额外练习等信息；推荐效果未通过真实学生体验评估验证；方法在低样本、非编程类课程中的可迁移性仍待验证。

---

## 282. Temper-Then-Tilt: Principled Unlearning for Generative Models through Tempering and Classifier Guidance

**arXiv ID:** 2602.10217 | [PDF](https://arxiv.org/pdf/2602.10217v1)

**作者:** Jacob L. Block `[一作]` (University of Texas at Austin), Sanjay Shakkottai `[通讯]` (University of Texas at Austin)

**通讯引用:** 7311 | [OpenAlex ID](https://openalex.org/A5028903768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大规模生成模型的机器不学习（unlearning），将不学习问题建模为密度比估计，并提出基于温度-倾斜的 Temper-Then-Tilt Unlearning 方法。

**💡 创新点**

创新点在于：①把不学习视为分布倾斜任务；②证明标准分类引导在忘记集合高度集中的情形下会泄漏信息；③提出先对基模型进行温度平滑再使用轻量级分类器倾斜的方法，并给出有限样本的保真度和忘记误差理论保证；④展示该方法在保持生成质量的同时显著提升忘记质量。

**🔧 技术方法**

技术手段包括：密度比估计与分类器引导；基模型温度平滑；轻量级线性头在冻结 LLM 表示上训练；交叉熵分类损失；softmax+温度推断；理论分析的 Retain Error 与 Forget Error 上界；实验中使用 Llama 3.1 8B 作为基模型。

**📊 数据集**

数据集：TOFU benchmark（由 GPT‑4 生成的 200 位虚构作者的问答对，用于评估 LLM 的不学习），以及 Llama 3.1 8B 作为训练/评估模型。

**📈 对比分析**

与 GradAscent、GradDiff、IdkDPO、WGA、SatImp、UnDIAL、RMU、ULD、NPO、SimNPO 等基线方法比较。实验表明 Temper-Then‑Tilt 在 TOFU 上获得最高的 Forget Quality（FQ）和 MU‑ROUGE（生成质量指标），并且仅训练少量参数（≈0.03%）与极低的计算/存储成本。

**⚠️ 局限性**

局限性：①温度参数需在提升忘记效果与引入偏差之间进行权衡；②理论保证基于混合分布假设，对极端尖锐的忘记分布仍有挑战；③在更复杂或真实世界数据集上的泛化需进一步验证；④对长尾或多模态忘记集合的性能尚未系统评估。

---

## 283. Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking

**arXiv ID:** 2602.10577 | [PDF](https://arxiv.org/pdf/2602.10577v1)

**作者:** Yiming Che `[一作]` (Walmart Global Tech), Kannan Achan `[通讯]` (Walmart Global Tech)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了Campaign‑2‑PT‑RAG框架，利用LLM对电商营销活动进行语义解读，并将其映射到产品类型（PT）集合，从而生成可用于训练活动排名模型的用户‑活动购买标签；

**💡 创新点**

创新点在于将检索增强生成（RAG）与结构化LLM推理相结合，先用LLM提炼活动意图，再通过语义检索和三级相关性分类精准识别活动所覆盖的PT；

**🔧 技术方法**

技术主要包括：LLM（如GPT‑4o）用于活动解读和PT相关性判断、密集向量检索（bi‑encoder）和可选的交叉编码reranker、以及基于PT层次结构的结构化推理；

**📊 数据集**

实验数据集包含Walmart内部真实营销活动（8条，人工标注）以及由ChatGPT生成的合成活动描述，PT知识库共7,147个条目；

**📈 对比分析**

与BM25、零射击LLM、单纯检索等基线比较，Campaign‑2‑PT‑RAG在真实数据上平均精度0.89、召回0.98、F1 0.94，在合成数据上亦以F1 0.87、语义一致性0.79、LLM分数0.89等指标领先；

**⚠️ 局限性**

局限性包括：对大模型的依赖导致推理成本高、相关性判断仍具主观性、可选reranker与LLM推理层增加延迟，且模型在不同部署环境下的鲁棒性需进一步验证。

---

## 284. XSPLAIN: XAI-enabling Splat-based Prototype Learning for Attribute-aware INterpretability

**arXiv ID:** 2602.10239 | [PDF](https://arxiv.org/pdf/2602.10239v1)

**作者:** Dominik Galus `[一作]` (Wrocław University of Science and Technology), Piotr Syga `[通讯]` (Wrocław University of Science and Technology)

**通讯引用:** 341 | [OpenAlex ID](https://openalex.org/A5088028973)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于原型的前置可解释框架XSPLAIN，用于3D高斯光斑分类。

**💡 创新点**

创新点包括：将体素聚合的PointNet作为骨干网络、引入可逆正交变换实现特征通道解耦、通过原型检索实现“看起来像那样”的直观解释。

**🔧 技术方法**

采用了PointNet+体素聚合、正交矩阵（指数映射）训练、密度感知正则化、纯度损失以及原型检索等技术。

**📊 数据集**

在Toys、ShapeSplat、MVImageNet-GS、3D Real Car Toolkit和ShapeNet Core等数据集上进行实验。

**📈 对比分析**

与传统监督点云模型以及后置方法LIME/PointSHAP比较，XSPLAIN保持与骨干相同的分类精度（≈0.88），在用户研究中获得48.4%优先选择，且在删除测试中验证了解释的可信度。

**⚠️ 局限性**

局限性在于可用的高斯光斑分类数据集有限，主要涵盖简单物体，未来需扩展到更丰富的3DGS数据。

---

## 285. Red-teaming the Multimodal Reasoning: Jailbreaking Vision-Language Models via Cross-modal Entanglement Attacks

**arXiv ID:** 2602.10148 | [PDF](https://arxiv.org/pdf/2602.10148v1)

**作者:** Yu Yan `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Min Liu `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对先进视觉语言模型(VLM)的跨模态安全漏洞，本文提出了一种名为COMET的黑盒攻击框架，能够在多模态推理过程中诱导模型生成有害内容。

**💡 创新点**

创新点在于：①使用知识可扩展重构将单一恶意指令转换为多跳链式指令；②通过跨模态线索纠缠将文本中的恶意实体迁移至图像中，并在文本中用空间指针引用，形成细粒度隐蔽的语义耦合；③在攻击负载外嵌入精心设计的跨模态情景与评分规则，进一步诱导模型进入指令遵循模式，从而生成更具有害性的输出。

**🔧 技术方法**

核心技术包括ReAct循环知识重构、文本到图像生成（T2I）与图像到文本生成（I2T）的交叉使用、跨模态线索纠缠、情景嵌套与评分规则引导、以及基于StrongReject的攻击成功率与有害性评分评估。

**📊 数据集**

实验使用SafeBench（含7类危险任务）以及其精简版SafeBench‑tiny进行评估，并在9款主流VLM（如GPT‑4.1、Gemini‑2.5‑Pro、Qwen3‑VL‑235B 等）上进行攻击测试。

**📈 对比分析**

与CS‑DJ、FigStep、HIMRD等三种先进对抗方法对比，COMET在未加防御（Vanilla）和加防御（Defended）两种设置下，平均攻击成功率达到94%，比CS‑DJ高出约29%，且在所有模型上均保持高成功率，尤其在加入prompt‑based防御时仍能保持高效。

**⚠️ 局限性**

局限性：攻击过程依赖外部辅助模型（T2T、I2T、T2I），生成成本相对较高；对更严格的安全防御（如多模态内容审核、实时监控）尚未充分评估；在极端高防御或特定安全策略下，攻击成功率可能进一步下降。

---

## 286. FGAA-FPN: Foreground-Guided Angle-Aware Feature Pyramid Network for Oriented Object Detection

**arXiv ID:** 2602.10710 | [PDF](https://arxiv.org/pdf/2602.10710v1)

**作者:** Jialin Ma `[一作]` `[通讯]` (Beijing University Of Posts And Telecommunications), Jialin Ma (Beijing University Of Posts And Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在遥感图像中实现了面向前景且角度感知的特征金字塔网络FGAA-FPN，用于提高旋转目标检测的精度

**💡 创新点**

通过在低层特征上引入前景引导特征调制（FGFM）和在高层特征上加入角度感知多头注意力（AAMHA），实现了前景与几何信息的双重显著增强

**🔧 技术方法**

结合特征金字塔网络、弱监督前景概率估计、角度相关偏置的多头注意力以及残差门控等深度学习技术

**📊 数据集**

在DOTA v1.0和DOTA v1.5两个大规模遥感目标检测基准上进行实验

**📈 对比分析**

与多种主流FPN变体及旋转检测框架进行对比，FGAA-FPN在DOTA v1.0上mAP达到75.5%，在DOTA v1.5上mAP 68.3%，均显著优于现有方法

**⚠️ 局限性**

额外的计算开销较大，且弱监督的前景引导在高噪声或极度混乱场景下效果有限，需要进一步轻量化和更精准的几何语义先验

---

## 287. TRACE: Theoretical Risk Attribution under Covariate-shift Effects

**arXiv ID:** 2602.10588 | [PDF](https://arxiv.org/pdf/2602.10588v1)

**作者:** Hosein Anjidani `[一作]` (Sharif University of Technology), Mohammad Hossein Yassaee `[通讯]` (Sharif University of Technology)

**通讯引用:** 568 | [OpenAlex ID](https://openalex.org/A5044522990)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出TRACE框架，用于在协变量偏移（covariate shift）下，对模型更新导致的源域风险变化ΔR进行理论归因与诊断；

**💡 创新点**

创新点包括：①将风险变化拆解为四个可解释的上界项（源/目标泛化差距、模型变化惩罚、协变量偏移惩罚）；②给出可计算的高概率诊断报告；③引入梯度分位数估计的Lipschitz常数与特征空间Optimal Transport（Sinkhorn）/MMD作为偏移量估计；④基于TRACE构造安全部署门（gate）分数，实现无标签安全模型替换；

**🔧 技术方法**

技术手段包括：Lipschitz与有界损失假设、Wasserstein距离与Kantorovich–Rubinstein对偶、Sinkhorn迭代求OT、MMD、梯度分位数近似、输出距离度量、Hoeffding/Hoeffding‑type集中不等式以及PAC‑Bennett等概率界；

**📊 数据集**

实验使用合成数据（Gaussian blobs、two‑moons）以及真实视觉基准DomainNet，特征提取采用冻结的ResNet‑50；

**📈 对比分析**

与传统单一指标（如最大Softmax概率）相比，TRACE门分数在识别有害更新时达到Spearman相关系数≈0.94、AUROC=1.0；在主动域适应案例中，TRACE能够同时监测数据对齐与模型不稳定性，表明单一偏移度量不足；

**⚠️ 局限性**

局限性：仅针对协变量偏移；对标签/概念漂移缺乏理论保证；估计中的Lipschitz、梯度分位数假设可能导致保守上界；需验证数据和验证集；

---

## 288. Stop Training for the Worst: Progressive Unmasking Accelerates Masked Diffusion Training

**arXiv ID:** 2602.10314 | [PDF](https://arxiv.org/pdf/2602.10314v1)

**作者:** Jaeyeon Kim `[一作]` (Harvard University), Sitan Chen `[通讯]` (Harvard University)

**通讯引用:** 1045 | [OpenAlex ID](https://openalex.org/A5026455990)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种 Progressive UnMAsking（PUMA）方法，通过在训练时使用教师强制链（teacher‑forced chain）来调整前向掩码过程，使训练时的掩码分布与推理时的掩码模式一致，从而显著加速 Masked Diffusion Model（MDM）的训练。

**💡 创新点**

创新点在于首次通过修改前向掩码过程而非网络架构或损失加权，实现了训练与推理掩码分布的对齐，保证了目标函数最优解不变，同时显著提升训练效率。

**🔧 技术方法**

使用了教师强制链、置信度快进（confidence‑based fast‑forwarding）、K‑调度（K‑scheduling）、基于当前模型的推理策略作为掩码策略，并在训练中计算交叉熵损失；此外结合了自回归初始化和块大小热身等现有训练技巧。

**📊 数据集**

主要使用了 Sudoku 数据集（1.8M 训练、0.1M 测试）验证方法可行性，以及 TinyGSM 数据集（1.18M 训练样本）进行 125M 参数 MDM 的大规模预训练；评估时使用 GSM8K 测试集。

**📈 对比分析**

通过与标准 MDM 以及结合自回归初始化、块大小热身等方法的对比，实验显示 PUMA 在迭代数上提升约 2.5×（在 TinyGSM 上），在 Sudoku 上提升 1.4×，在自回归初始化场景下提升 2.3×；在壁钟时间上每秒迭代数略高，且 PUMA 对不同推理策略保持鲁棒。

**⚠️ 局限性**

实验仅在结构化的 TinyGSM（和 Sudoku）数据集上进行，结果对真实世界长序列、长距离依赖的数据集的适用性与速度提升幅度仍需进一步验证。

---

## 289. Can Large Language Models Implement Agent-Based Models? An ODD-based Replication Study

**arXiv ID:** 2602.10140 | [PDF](https://arxiv.org/pdf/2602.10140v1)

**作者:** Nuno Fachada `[一作]` (Lusófona University), João P. Matos-Carvalho `[通讯]` (Universidade de Lisboa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估LLM将完整ODD规范翻译成可执行Python ABM代码的能力。

**💡 创新点**

首次在严格的统计验证与性能评估框架下对多种LLM的ABM实现可靠性进行比较。

**🔧 技术方法**

采用LLM推理、Python代码生成、统计Energy检验、运行时测量和静态代码分析等技术。

**📊 数据集**

使用PPHPC（Predator‑Prey for High‑Performance Computing）模型的完整ODD描述作为规范。

**📈 对比分析**

通过分阶段执行、能量统计相似度检验、PCA+Energy多变量检验及与NetLogo基线的运行时对比，发现GPT‑4.1在所有试验中均能产生统计相似且运行高效的实现，其余模型成功率低且性能波动大。

**⚠️ 局限性**

局限性包括仅评估单一模型、可能的数据泄露、仅使用两组参数、对静态质量度量的依赖以及结果随LLM技术快速迭代而快速过时。

---

## 290. Towards Affordable, Non-Invasive Real-Time Hypoglycemia Detection Using Wearable Sensor Signals

**arXiv ID:** 2602.10407 | [PDF](https://arxiv.org/pdf/2602.10407v1)

**作者:** Lawrence Obiuwevwi `[一作]`, Sampath Jayarathna `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用。

**💡 创新点**

创新点在于提出了一种改进的模型架构，能够更有效地处理复杂数据。

**🔧 技术方法**

使用了深度学习技术，特别是卷积神经网络（CNN）和循环神经网络（RNN）。

**📊 数据集**

使用了公开的图像数据集和文本数据集进行实验。

**📈 对比分析**

与现有方法进行了比较，结果显示新算法在准确性和效率上均有显著提升。

**⚠️ 局限性**

限制在于模型对特定类型数据的适应性较差，且训练时间较长。

---

## 291. Learning to Compose for Cross-domain Agentic Workflow Generation

**arXiv ID:** 2602.11114 | [PDF](https://arxiv.org/pdf/2602.11114v1)

**作者:** Jialiang Wang `[一作]` (Hong Kong University of Science and Technology), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 27473 | [OpenAlex ID](https://openalex.org/A5100333516)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CapFlow模型，在单次推理中生成可执行的跨域工作流。

**💡 创新点**

将分解-重组-决策机制内置于LLM，学习可重用的工作流能力基，并用反事实归因驱动能力选择。

**🔧 技术方法**

稀疏基学习、任务条件composer、反事实能力归因、对比损失、对成功/失败工作流的多参考学习。

**📊 数据集**

覆盖推理、编程、数学和科学四大域的8个基准（HotpotQA、DROP、HumanEval、MBPP、GSM8K、MATH、SciBench、GPQA）以及自构造的跨域失败/成功工作流数据。

**📈 对比分析**

与手工提示、迭代工作流精炼（ADAS、AFlow）和学习式生成（ScoreFlow）对比，在多域、跨域及未见域上均以单步生成获得更高Solve率、可执行率，并显著降低推理延迟和成本。

**⚠️ 局限性**

依赖预先收集的成功/失败工作流样本，未能动态适应新工具库或在线反馈；对极长或更复杂拓扑的泛化尚待验证。

---

## 292. UMEM: Unified Memory Extraction and Management Framework for Generalizable Memory

**arXiv ID:** 2602.10652 | [PDF](https://arxiv.org/pdf/2602.10652v1)

**作者:** Yongshi Ye `[一作]` (Xiamen University), Weihua Luo `[通讯]` (Alibaba International Digital Commerce)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 UMEM 框架，联合优化大语言模型的记忆提取和记忆管理，使自演化代理在连续交互中能够动态改进外部记忆库。

**💡 创新点**

核心创新包括：①语义邻域建模（Semantic Neighborhood Modeling）以避免实例噪声；②使用边际效用奖励（Marginal Utility Reward）与 Group Relative Policy Optimization（GRPO）实现记忆提取与管理的联合强化学习；③在线记忆演化（Online Memory Evolution）让记忆库随训练实时更新。

**🔧 技术方法**

技术主要包括：大型语言模型（Llama‑3.2‑1B, Qwen3‑4B, Qwen3‑8B 等）做记忆提取与执行器；GRPO 强化学习算法；语义嵌入编码器（如 BGE‑M3）构建邻域；自定义 XML 记忆格式与奖励设计。

**📊 数据集**

使用 MMLU（随机采样 2000 题）构建训练集，并在 AIME、GPQA‑Diamond、HLE、HotpotQA、ALFWorld 等 5 个基准上进行评估；同时对不同执行器（GPT‑5.1、Qwen3‑8B、Gemini‑2.5‑Flash）进行跨模型验证。

**📈 对比分析**

与基线（无记忆、无训练、Self‑RAG、Memp、ReMem）相比，UMEM 在单回合推理任务上平均提升 10–15% EM，在多回合 ALFWorld 中显著提高 CSR（最高 82.84%），并保持持续的性能增长曲线；在 10 轮长期交互实验中，UMEM 仍保持最快恢复和最高累计成功率。

**⚠️ 局限性**

局限性包括：①依赖预训练的语义编码器，邻域构建对编码质量敏感；②奖励设计和 GRPO 训练对计算资源要求高，尤其是多邻域评估；③在极大规模任务或动态环境中，记忆库更新频率和容量仍需进一步研究。

---

## 293. The Subjectivity of Respect in Police Traffic Stops: Modeling Community Perspectives in Body-Worn Camera Footage

**arXiv ID:** 2602.10339 | [PDF](https://arxiv.org/pdf/2602.10339v1)

**作者:** Preni Golazizian `[一作]` (University of Southern California), Morteza Dehghani `[通讯]` (University of Southern California)

**通讯引用:** 3966 | [OpenAlex ID](https://openalex.org/A5065952016)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在洛杉矶警察局（LAPD）BWC 录像上构建了大规模、跨群体（执法、司法受影响、普通居民）交通拦截尊重评分与理由的多视角数据集，并提出了基于程序正义的尊重评估规则；

**💡 创新点**

创新点在于把“尊重”视为主观多视角任务，设计了群体感知驱动的评价标准、规则引导的偏好数据生成框架，以及可对齐的个体化模型，使得同一事件可产生多样化的评分与理由；

**🔧 技术方法**

主要技术包括基于大语言模型（LLM）的规则评判器（LLM-as-a-judge）、规则驱动的偏好数据构造、LoRA 参数高效微调以及基于偏好的直接偏好优化（DPO）；

**📊 数据集**

使用了约1,000条LAPD交通拦截BWC视频及其转录文本，配合来自三类群体的4名注释者（共计数千条尊重评分和自由文本理由）；

**📈 对比分析**

通过混合效应分析验证群体差异，零拷贝模型基线 MAE 较高且解释性差，SFT+DPO 后在评分 MAE 和基于规则的 F1 方面均显著提升，尤其在司法受影响群体上提升最大；

**⚠️ 局限性**

局限性包括数据仅涵盖文本转录，未加入语音、视频等非语言线索；规则与模型可能难以迁移至其他警务情境；以及对群体标签的使用需避免强化刻板印象。

---

## 294. Autonomous Continual Learning of Computer-Use Agents for Environment Adaptation

**arXiv ID:** 2602.10356 | [PDF](https://arxiv.org/pdf/2602.10356v1)

**作者:** Tianci Xue `[一作]` (Ohio State University), Huan Sun `[通讯]` (Ohio State University)

**通讯引用:** 2499 | [OpenAlex ID](https://openalex.org/A5101488340)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个完全自主的课程强化学习框架 ACuRL，使计算机使用代理在目标环境中通过零数据持续学习并自适应。

**💡 创新点**

创新点在于：①将环境探索与上下文回顾结合，用实际经验驱动任务合成；②根据代理的能力动态调整任务难度的自适应课程生成；③设计自动评估器 CUAJudge，利用状态差分和证据驱动的关键点验证，提供高可靠的奖励信号。

**🔧 技术方法**

技术手段包括基于 GPT‑5 的任务生成、GRPO 扩展的多步强化学习、统一环境管理协议、异步环境预加载、自动评估器 CUAJudge 以及迭代式课程学习。

**📊 数据集**

实验使用了六个代表性桌面/网络环境（LibreOffice Impress/Writer/Calc、Thunderbird、Celestia、KAlgebra），并基于 OSWorld、OfficeWorld、ScienceBoard 的任务集；同时采集 1,444 条公开轨迹用于评估评判器。

**📈 对比分析**

与现有基线（如 UI‑TARS、Claude‑3.7 等）对比，ACuRL 在目标环境的成功率提升 4–22%，跨环境学习亦保持或提升 2–6%；自动评估器与人类评估的一致率高达 93.7%，与规则评估器的精度达到 87.5%。

**⚠️ 局限性**

局限性包括：仍依赖 GPT‑5 生成任务、对计算资源需求高；评估器主要依赖视觉信息，可能对非视觉交互适用性有限；在极度差异的环境间迁移仍存在性能瓶颈；对长周期迁移学习的长期稳定性未做深入验证。

---

## 295. A Practical Guide to Agentic AI Transition in Organizations

**arXiv ID:** 2602.10122 | [PDF](https://arxiv.org/pdf/2602.10122v1)

**作者:** Eranga Bandara `[一作]` (Old Dominion University), Nilaan Loganathan `[通讯]` (Effectz.AI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一套从人工与AI辅助工作流迁移至完全代理式AI系统的实践框架，强调业务领域驱动、任务委派、AI辅助工作流构建与小型跨职能团队；并在旅游中小企业中通过多智能体系统实现日常计划与交通管理的自动化；

**💡 创新点**

将代理式AI的引入视为组织转型问题而非单纯技术问题，提出人类作为协调者的“人机共管”操作模型；利用AI自身进行工作流构建（Agentsway）与小团队自治；在实际案例中展示了从无代码到代理式工作流的完整落地过程；

**🔧 技术方法**

大语言模型（Claude、OpenAI LLM）、Agent SDK、MCP服务器、LM Studio、Claude Code等AI工具与框架；通过多智能体架构实现任务拆分、工具调用与知识上下文管理；

**📊 数据集**

使用真实业务数据（旅游预订邮件、日程表、车辆与路线信息），未引用公开通用数据集；主要基于企业内部的运营记录进行评估；

**📈 对比分析**

评估指标包括推理正确性、输出一致性、业务可用性、可解释性、效率提升及符合责任AI原则；实验结果表明代理式工作流能完全替代人工日程与交通计划，且在可靠性、可扩展性和可解释性方面优于传统工具；

**⚠️ 局限性**

局限性：案例规模有限，仅覆盖旅游SME场景；缺乏大规模对比基准；对不同LLM与工具链的依赖可能导致迁移成本；未深入讨论治理、合规与长期维护机制；需要更多跨行业验证以证明通用性。

---

## 296. Fine-Tuning GPT-5 for GPU Kernel Generation

**arXiv ID:** 2602.11000 | [PDF](https://arxiv.org/pdf/2602.11000v1)

**作者:** Ali Tehrani `[一作]` (Makora), Mohamed S. Abdelfattah `[通讯]` (Makora)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过强化学习微调 GPT‑5 以生成高性能 Triton GPU 核心，并构建了专门的数据集、评估基础设施、工具链以及可扩展的执行后端；

**💡 创新点**

创新点包括：① 引入可验证奖励（RLVR），将功能正确性与相对加速作为可确定的奖励；② 采用静态可达性分析与 LLM 判断双重防御机制防止奖励黑客；③ 设计多轮工具增强的 RL 训练框架，实现迭代调优与检验；④ 构建可扩展的评估后端和缓存策略，显著提升训练效率；⑤ 通过 GPT‑5‑RL 在 Triton 核心生成上实现新标杆。

**🔧 技术方法**

技术手段包括：大模型微调、强化学习、RL 从可验证奖励、Triton 代码生成、静态代码分析、LLM 判别器、工具调用（评估、搜索、Web 搜索、计时器）、缓存与 AST 规范化、k‑means 加权采样、基准化加速测量及基于逻辑函数的奖励设计。

**📊 数据集**

使用的数据集主要为扩展版 KernelBench（264 个基准），以及从公开 PyTorch 代码库抽取并清洗的 11,363 条训练样本，进一步构造 100 条和 1,000 条训练子集；同时使用公开的 264 条验证集进行评估。

**📈 对比分析**

与 GPT‑5、Gemini 2.5 Pro、Claude、Grok 以及 MakoraGenerate 进行对比，采用功能正确率、击败 TorchInductor 的比例和几何平均加速等指标；GPT‑5‑RL 在单次尝试中功能正确率提升 33%，击败 TorchInductor 的比例从 14.8% 提升至 21.8%，几何平均加速提升至 0.81×，在多次尝试后可达 83.7% 正确率。

**⚠️ 局限性**

局限性包括：奖励函数对加速的敏感度受 shift 参数限制，导致加速提升相对有限；数据集规模和多样性仍受限，尤其是高难度样本；奖励黑客仍存在风险，需进一步完善检测；RL 训练仍无法完全匹配多代理演化系统的探索效果；模型对不同 GPU 架构的泛化能力需进一步验证。

---

## 297. Tensor Methods: A Unified and Interpretable Approach for Material Design

**arXiv ID:** 2602.10392 | [PDF](https://arxiv.org/pdf/2602.10392v1)

**作者:** Shaan Pakala `[一作]` (University of California), Evangelos E. Papalexakis `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出使用张量完成方法来构建可解释的材料设计代理模型，解决传统机器学习在非均匀采样下的泛化和解释性不足问题。

**💡 创新点**

创新点在于将张量分解作为一体化的代理模型，既提供高精度预测，又能直接生成可解释的张量因子；并在非均匀采样场景下展示其优越的泛化性能。

**🔧 技术方法**

采用CPD、CPD‑S、NeAT、CoSTCo等张量完成方法，并与线性回归、Gaussian Process、Random Forest、XGBoost、CatBoost、MLP等传统ML模型对比。

**📊 数据集**

使用三组材料设计数据集：晶格结构（lattice structures）、交叉桶结构（crossed barrel）和电纺纳米纤维（cogni‑e‑spin）共计约1100k条实验记录。

**📈 对比分析**

在均匀采样和偏差采样两种评估情境下，张量模型在R^2上提升至5%并在离群区误差减半，整体表现与传统ML相当甚至更优；同时因子匹配得分证明因子可重现已知物理规律。

**⚠️ 局限性**

主要局限在于数据规模有限、低秩假设可能不成立以及对设计参数增多时的可扩展性尚未验证。

---

## 298. When Tables Go Crazy: Evaluating Multimodal Models on French Financial Documents

**arXiv ID:** 2602.10384 | [PDF](https://arxiv.org/pdf/2602.10384v1)

**作者:** Virginie Mouilleron `[一作]` (Inria Paris), Djamé Seddah `[通讯]` (Inria Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个面向法语金融文件的多模态基准，评估视觉语言模型在文本、表格、图表以及多轮对话中的理解能力。

**💡 创新点**

创新点在于构建了1204道专家验证的问答数据集Scribe‑Finance，并通过LLM‑as‑judge评测揭示了模型在图表解读和对话误差传播上的脆弱性。

**🔧 技术方法**

采用了六款开源视觉语言模型（Qwen3‑VL、Gemma、Pixtral 等）进行实验，并使用 GPT‑4o/Gemini‑2.0 辅助生成问题，评估时采用多模型投票的 LLM‑as‑judge 方案。

**📊 数据集**

数据集来源于资产管理公司的法语投资文件（招股说明书、KID、PRIIP），经整理后形成包含文本、表格和图表的 1204 题目问答集。

**📈 对比分析**

通过对不同任务类别（文本、表格、图表、多轮对话）的准确率进行比较，最强模型 Qwen3‑VL‑32B 的平均得分为 75.6%，文本/表格任务达到 80–90%，图表解读仅 34–62%，多轮对话约 46–59%。

**⚠️ 局限性**

局限性包括仅覆盖法语文档、生成过程可能引入偏差、LLM‑as‑judge 评测对细微错误敏感度有限，且模型在图表与多轮对话中的错误无法通过规模提升得到显著缓解。

---

## 299. Dual-End Consistency Model

**arXiv ID:** 2602.10764 | [PDF](https://arxiv.org/pdf/2602.10764v1)

**作者:** Linwei Dong `[一作]` (Zhejiang University), Changqing Zou `[通讯]` (Zhejiang University)

**通讯引用:** 2974 | [OpenAlex ID](https://openalex.org/A5100604564)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Dual-End Consistency Model (DE-CM)，通过选择关键子轨迹并结合流匹配与噪声-噪声映射，解决一致性模型的训练不稳定与采样不灵活问题，并在 ImageNet 256×256 上实现一阶采样 FID 1.70。

**💡 创新点**

创新点在于将 PF-ODE 轨迹分解为一致性、瞬时和噪声-噪声三类子轨迹作为优化目标，并利用流匹配边界正则化与新型噪声-噪声映射来提升训练稳定性与采样灵活性。

**🔧 技术方法**

采用连续时间一致性模型、流匹配（Flow Matching）、噪声-噪声映射（N2N）、混合采样（Mix Sampler）、自适应加权、速度归一化等技术。

**📊 数据集**

使用 ImageNet 256×256（类到图像）和 Text-to-Image-2M（文本到图像）数据集，并在预训练的 VAE Tokenizer 上进行实验。

**📈 对比分析**

与现有一致性模型和 ODE 采样方法对比，使用 FID、CLIP/BLIP 等指标，DE-CM 在 1 NFE 取得 FID 1.70，在 2 NFE 仍优于多数多步模型，并在更高 NFE 下保持稳定性能。

**⚠️ 局限性**

主要局限是 JVP 计算与 FSDP/Flash Attention 不兼容，导致显存消耗大，限制了更大规模模型的训练。

---

## 300. A Robust Optimization Approach for Regenerator Placement in Fault-Tolerant Networks Under Discrete Cost Uncertainty

**arXiv ID:** 2602.11058 | [PDF](https://arxiv.org/pdf/2602.11058v1)

**作者:** Mohammad Khosravi `[一作]` (Ruhr University of Bochum), Setareh Maghsudi `[通讯]` (Ruhr University of Bochum)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在离散成本不确定性及边故障预算不确定性下的容错网络中再生器放置的鲁棒优化问题。

**💡 创新点**

创新点在于提出了基于离散不确定集的鲁棒 FTRLP，设计了 Full Recovery of Equality (FRE) 约束以保证模型与原图的一致性，并给出了两种新的整数规划模型（基于流与基于割）以及高效的迭代求解方法。

**🔧 技术方法**

主要技术包括整数规划、网络流模型、割集合与 Menger 定理、McCormick 线性化、分支剪枝、分支–Benders 及迭代主子问题求解。

**📊 数据集**

实验使用作者自定义的两类随机实例 Gen-1 与 Gen-2，随机生成图密度、边长度和节点成本，未使用公开真实数据集。

**📈 对比分析**

通过与改进的 IP-LA 对比，实验显示 IP‑FB 与 IP‑CB 在小规模实例上平均比 IP‑LA 快数倍，LP 松弛更紧；IT‑FB 在大规模实例中能够在几分钟内获得最优解，表现优于现有方法。

**⚠️ 局限性**

局限性包括 IP‑CB 的指数约束仅适用于小规模实例，迭代方法需要多轮求解，且在多条边失效情况的扩展需要额外处理；实验仅基于随机生成的实例，缺乏对真实网络的验证。

---

## 301. Coarse-Grained Boltzmann Generators

**arXiv ID:** 2602.10637 | [PDF](https://arxiv.org/pdf/2602.10637v1)

**作者:** Weilong Chen `[一作]` (Technical University of Munich), Julija Zavadlav `[通讯]` (Technical University of Munich)

**通讯引用:** 738 | [OpenAlex ID](https://openalex.org/A5015672483)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Coarse‑Grained Boltzmann Generators (CG‑BGs)，在粗粒化坐标空间内实现可扩展且可保持统计正确的平衡采样

**💡 创新点**

将生成式模型与重要性重加权结合，利用学习到的势能平均势 (PMF) 作为目标能量，从而实现无偏、可扩展的粗粒化采样

**🔧 技术方法**

采用连续归一化流 (CNF)、增强采样力匹配 (ESFM) 学习 PMF、重要性采样与权重裁剪技术

**📊 数据集**

在二维 Müller–Brown 潜能、以及使用显式和隐式溶剂的丙氨酸二肽（Heavy Atom 与 Core‑Beta 粗粒化映射）上训练与测试

**📈 对比分析**

与显式溶剂 MD 参考和隐式溶剂基线相比，CG‑BG 在重加权后实现了低 JS 散度、较高有效样本数（ESS），且在粗粒化映射上仍保持与原始系统相当的自由能预测精度

**⚠️ 局限性**

依赖预先定义的粗粒化坐标与增强采样路径，且在极度粗粒化或复杂系统中可能出现条件方差增大导致学习不稳定

---

## 302. LUCID: Attention with Preconditioned Representations

**arXiv ID:** 2602.10410 | [PDF](https://arxiv.org/pdf/2602.10410v1)

**作者:** Sai Surya Duvvuri `[一作]` (University of Texas at Austin), Inderjit S. Dhillon `[通讯]` (Google)

**通讯引用:** 29551 | [OpenAlex ID](https://openalex.org/A5063459703)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出LUCID注意力机制，通过对键的预条件化降低键相关性，从而在长序列中显著提升检索精度。

**💡 创新点**

创新点在于将键-键相似性的指数矩阵做为RKHS的预条件器，实现对软max注意力的可学习性与精确检索的分离。

**🔧 技术方法**

使用Reproducing Kernel Hilbert Space（RKHS）理论、指数核预条件化、正则化以及cuBLAS TRSM等实现。

**📊 数据集**

在多长文本检索数据集（BABILong、RULER、SCROLLS、LongBench等）上进行评估。

**📈 对比分析**

与标准Softmax、Diff Transformer、DeltaNet、PaTH等方法对比，LUCID在长上下文检索任务中提升约10–20%（BABILong 18%），且训练与推理开销仅提升5.5%和1.3%。

**⚠️ 局限性**

局限在于仅适用于因果（单向）模型，双向场景预条件矩阵失去三角结构，导致求解成本过高，且对超长序列外推仍有限。

---

## 303. AgentTrace: A Structured Logging Framework for Agent System Observability

**arXiv ID:** 2602.10133 | [PDF](https://arxiv.org/pdf/2602.10133v1)

**作者:** Adam AlSayyad `[一作]` (University of California), Richik Pal `[通讯]` (University of California)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5113268115)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

设计并实现了 AgentTrace，一个支持认知、操作和上下文三层的结构化日志框架，可在运行时无侵入地捕获 LLM 代理的完整行为，并通过 OpenTelemetry 实现实时分布式追踪。

**💡 创新点**

创新点包括：①统一三层表面（认知、操作、上下文）的 schema 设计；②利用装饰器在不改动代理代码的前提下实现自动注入；③将认知层（LLM 推理过程）嵌入分布式追踪链，保持语义关联；④同时提供本地 JSONL 和远程 OTEL 输出，实现低延迟本地调试与可扩展的云端监控。

**🔧 技术方法**

核心技术：Python 装饰器/观察者模式、OpenTelemetry 自动 instrumentation、JSONL 文件输出、JSON Schema 校验、OpenAI/Anthropic API 解析、OTel span 生成与批量导出、轻量级日志路由。

**📊 数据集**

论文未使用特定实验数据集，主要通过示例代理代码演示框架功能；若需评估可在公开 LLM 代理项目（如 OpenAI 的 Agent API、LangChain 等）上使用。

**📈 对比分析**

对比方法：与 AgentOps、LADYBUG、AgentSight 等现有工具比较，指出它们在表面覆盖和认知嵌入上的不足。论文未给出定量性能指标，主要描述低开销（两事件/调用）和无阻塞导出；在实验环境中观察到正常运行时的吞吐量与标准分布式追踪工具相当。

**⚠️ 局限性**

局限性：①需可装饰的代理接口，特殊或闭源代理可能无法兼容；②认知抽取依赖模型输出的固定格式，跨模型解析困难；③未捕获内部细粒度状态变化（如模型内部变量）；④在大规模并发代理场景下的延迟与可扩展性尚未彻底验证；⑤外部工具链完整性仍受限，某些自定义 I/O 需要手动补充。

---

## 304. IMITATOR4AMAS: Strategy Synthesis for STCTL

**arXiv ID:** 2602.10810 | [PDF](https://arxiv.org/pdf/2602.10810v1)

**作者:** Davide Catta `[一作]` (Universite Sorbonne Paris Nord), Teofil Sidoruk `[通讯]` (Polish Academy of Sciences)

**通讯引用:** 579 | [OpenAlex ID](https://openalex.org/A5057757068)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了一个支持不完全信息、无记忆策略合成的实时多代理系统验证工具。

**💡 创新点**

在不完全信息下直接合成无记忆策略，并将时序约束融入策略搜索，显著提升效率。

**🔧 技术方法**

基于OCaml实现的模型检查器Imitator，采用BFS构建状态空间并结合SMT进行时序约束求解。

**📊 数据集**

使用投票与寻宝等多代理模型作为基准，包括参数化时序自动机（Parametric TA）实例。

**📈 对比分析**

与先前的SMT4SMTL/Maude方法相比，实验显示在多代理投票案例中速度提升至35倍，整体性能显著提升。

**⚠️ 局限性**

目前仅支持无记忆策略，且对更复杂的策略语义（计数、时间、部分视图等）尚未实现，且某些实例仍可能无法终止。

---

## 305. Free-Flying Crew Cooperative Robots on the ISS: A Joint Review of Astrobee, CIMON, and Int-Ball Operations

**arXiv ID:** 2602.10686 | [PDF](https://arxiv.org/pdf/2602.10686v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 306. Characterization and Computation of Normal-Form Proper Equilibria in Extensive-Form Games via the Sequence-Form Representation

**arXiv ID:** 2602.10524 | [PDF](https://arxiv.org/pdf/2602.10524v1)

**作者:** Yuqing Hou `[一作]` (University of Science and Technology of China), Chuangyin Dang `[通讯]` (City University of Hong Kong)

**通讯引用:** 9174 | [OpenAlex ID](https://openalex.org/A5085575860)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了在序列形式（sequence‑form）中对正常形式精炼均衡（normal‑form proper equilibrium）的紧凑表征，并基于此设计了两种可微分路径跟踪算法（对数障碍路径和熵障碍路径）用于求解扩展形式游戏中的正常形式精炼均衡。

**💡 创新点**

创新点主要包括：① 定义了序列形式下的正常形式精炼均衡并证明其与传统正常形式精炼均衡等价；② 通过引入ε‑permutahedron构造了一类扰动游戏，使得每个Nash均衡都对应一个正常形式精炼均衡；③ 设计了两种利用对数或熵正则化的可微分路径跟踪方法，实现了从任意正向实现计划开始的平滑路径并收敛到精炼均衡；④ 在理论上给出了平滑路径存在性和收敛性的证明。

**🔧 技术方法**

技术手段包括：序列形式的线性流约束、对数/熵正则化的凸优化、补偿法与可微分方程求解、预测‑校正数值路径跟踪、以及对ε‑permutahedron扰动的数学分析。

**📊 数据集**

实验使用了两种随机生成的扩展形式游戏（类型1和类型2），并对示例图中的具体游戏进行验证，所有游戏均为有限完美回忆扩展形式游戏。

**📈 对比分析**

通过在相同起始点下对比LGPR（对数障碍路径）和ETPR（熵障碍路径）两种方法，在不同游戏规模（玩家数、历史深度、动作数）下记录迭代次数、计算时间和失败率。结果显示：LGPR在大多数实例中迭代次数和计算时间都更少、失败率更低；ETPR在较大规模游戏中稳定性下降，失败率显著上升。

**⚠️ 局限性**

限制：① 对数障碍方法在数值上可能出现梯度爆炸，导致数值不稳定；② 熵障碍方法在大规模游戏中易失效；③ 目前仅适用于有限完美回忆的扩展形式游戏，未考虑信息不完全或无限深度情形；④ 缺乏针对精炼均衡的选择性或收敛加速策略。

---

## 307. MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation

**arXiv ID:** 2602.11062 | [PDF](https://arxiv.org/pdf/2602.11062v1)

**作者:** Jialin Liu `[一作]` (City University of Hong Kong), Ray C. C. Cheung `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种通过稀疏正则化的残差量化变分自动编码器（RQ-VAE）将多模态特征离散化为可解释语义代码，并结合自适应稀有度放大和分层多源图编码实现冷启动推荐；

**💡 创新点**

创新点在于将推荐任务转化为离散语义标记化，使用稀疏正则化鼓励代码稀疏且可解释，加入自适应稀有度放大提升稀有物品学习，构建多源图融合架构实现内容与协同信号的有效整合；

**🔧 技术方法**

核心技术包括RQ-VAE、残差量化、KL稀疏正则、稀有度加权学习、自适应稀有度放大、分层LightGCN图编码、注意力与融合机制以及BPR+Contrastive训练；

**📊 数据集**

在三个Amazon公开数据集（Baby、Sports、Clothing）上进行实验，分别提供用户-物品交互、视觉与文本特征；

**📈 对比分析**

与传统MF-BPR、LightGCN以及多模态基线（VBPR、MMGCN、LATTICE、FREEDOM、BM3、LGMRec、LPIC）等共计约20个模型对比，MoToRec在整体和冷启动场景均显著提升，冷启动Recall@20/ NDCG@20提升最高可达12.6%；

**⚠️ 局限性**

局限包括对代码书大小与稀疏度调参敏感，对非常稀疏或无文本/视觉信息的物品效果有限；模型训练复杂度高于纯ID模型，且在超大规模场景下可能需要进一步优化；

---

## 308. Deep learning outperforms traditional machine learning methods in predicting childhood malnutrition: evidence from survey data

**arXiv ID:** 2602.10381 | [PDF](https://arxiv.org/pdf/2602.10381v1)

**作者:** Deepak Bastola `[一作]` (Florida Atlantic University), Yang Li `[通讯]` (Florida Atlantic University)

**通讯引用:** 7793 | [OpenAlex ID](https://openalex.org/A5100421768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评估并比较了16种机器学习与深度学习模型，用于预测尼泊尔2019年NMICS调查中5岁以下儿童的营养不良风险。

**💡 创新点**

首次在低资源环境下采用TabNet等基于注意力机制的深度学习架构，并提出统一的、多指标（F1、召回、精确度等）评估框架，突出对严重类别不平衡的关注。

**🔧 技术方法**

使用TabNet、DNN、Wide&Deep、ResNet等深度学习模型，以及AdaBoost、CatBoost、XGBoost、LightGBM、SVM、RF、DT等传统与提升学习算法，进行超参数调优和交叉验证。

**📊 数据集**

基于尼泊尔2019年多指标集群调查（NMICS）数据集，共6,416个样本，构建了整合体重/身高/体重指数的二分类营养不良指标。

**📈 对比分析**

通过十种性能指标（包括F1、召回、精确度、AUC、Cohen’s kappa、MCC等）进行统一比较，发现TabNet在F1（0.62）、召回（0.62）与精确度（0.63）上优于其他模型，SVM与AdaBoost次之。

**⚠️ 局限性**

主要限制包括样本量有限、变量仅限于社会经济与基本健康信息、缺乏临床与膳食细节、以及自报数据可能导致的偏差。

---

## 309. Simultaneous Speech-to-Speech Translation Without Aligned Data

**arXiv ID:** 2602.11072 | [PDF](https://arxiv.org/pdf/2602.11072v1)

**作者:** Tom Labiausse `[一作]` (Kyutai), Neil Zeghidour `[通讯]` (Gradium)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种无需词级对齐数据即可进行实时语音转语音与语音转文本翻译的多语种模型；

**💡 创新点**

创新点在于用粗粒度句级对齐训练基模型，再通过仅基于 BLEU 的过程奖励和 GRPO 强化学习实现低延迟、低质量权衡；

**🔧 技术方法**

技术包括多流 RQ‑Transformer、Mimi 低帧率音频码器、内在对话文本流、基于 BLEU 的过程奖励、GRPO 强化学习以及轻量化蒸馏；

**📊 数据集**

使用约 40,000 小时多语种合成翻译数据（法、斯、葡、德），并在 200 小时以内的合成数据上微调，最后针对意大利语使用不到 1,000 小时进行新语言适配；

**📈 对比分析**

与现有 Seamless、Hibiki 进行客观指标（BLEU、COMET、LAAL、End Offset）和主观 MOS 评估，结果显示在长短句场景下均取得 5+ BLEU 点、20+ 语音相似度提升、延迟降低等显著优势；

**⚠️ 局限性**

局限在于无法控制输入语言口音强度，若需此功能需在训练时加入口音标注并做条件化处理。

---

## 310. Omnidirectional Dual-Arm Aerial Manipulator with Proprioceptive Contact Localization for Landing on Slanted Roofs

**arXiv ID:** 2602.10703 | [PDF](https://arxiv.org/pdf/2602.10703v1)

**作者:** Martijn B. J. Brummelhuis `[一作]`, Salua Hamaza `[通讯]` (TU Delft)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

设计并验证了一种双臂全向无人机操纵器，能够通过自我感知的扭矩观测器检测与定位接触点，并在不依赖外部传感器的情况下估计倾斜表面倾角，从而实现在斜屋顶上的安全降落。

**💡 创新点**

创新点包括：1) 双臂全向设计与低惯性臂结构，使操纵器在保持平稳飞行的同时可在 360° 工作空间内操作；2) 基于动量的扭矩观测器实现的本体感知接触检测与定位；3) 通过接触点推断斜面倾角并规划降落的端到端闭环流程。

**🔧 技术方法**

使用动量基扭矩观测、前向与逆运动学、简化的动力学模型、机械臂运动学以及运动捕捉系统对状态进行测量；实现了扭矩估计、接触点定位和斜面估计。

**📊 数据集**

实验使用9次飞行测试，覆盖3个不同倾斜角（11.3°, 20.6°, 30.5°），采用 OptiTrack 运动捕捉系统获取真实倾斜角数据；未使用公开数据集。

**📈 对比分析**

与真实倾斜角进行对比，平均估计误差为 2.87°，误差分布在 ±5° 以内，足以实现可靠降落；在所有 9 次实验中均成功降落。

**⚠️ 局限性**

局限性包括：仅在平行于机体 x 轴的平面上估计倾斜角；需要高摩擦表面或附加柔性末端执行器；依赖准确的动力学模型；仅在倾斜角≤30.5°的范围内验证；未考虑更复杂屋顶形状或多点接触的情形。

---

## 311. How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning

**arXiv ID:** 2602.10622 | [PDF](https://arxiv.org/pdf/2602.10622v1)

**作者:** Jiahao Yuan `[一作]` (Ant Group), Weiqiang Wang `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在用户表示学习中使用解码器型大语言模型（LLM）的注意力掩码策略，系统比较了因果、混合和双向掩码，并提出了基于梯度的软掩码预热机制（GG‑SM）以平滑因果到双向的迁移。

**💡 创新点**

创新点在于将注意力掩码视为可学习的动态过程，利用梯度信息在预热阶段自适应开启未来注意力，从而在对比学习框架下显著提升LLM在用户建模任务上的表现。

**🔧 技术方法**

技术上使用了对比学习（InfoNCE）目标、梯度引导的软掩码预热、线性调度、LoRA微调、以及多模态行为序列的统一编码。

**📊 数据集**

数据集来自支付宝真实业务，包含两类对齐数据：基于规则的行为轨迹数据集（𝒟_behavior）和LLM合成的查询–答案对齐数据集（𝒟_qa），并在12个用户理解基准上进行评测。

**📈 对比分析**

通过在9个二分类任务上对比Oracle（因果掩码）、Hybrid、Bidirectional和Scheduler等三种掩码策略，实验发现GG‑SM在平均AUC上达0.7745，显著优于传统用户模型（如MSDP、FOUND）和大型通用嵌入模型（如Llama‑embed‑nemotron）。

**⚠️ 局限性**

限制主要体现在对大规模真实行为序列的依赖，掩码策略的可迁移性与不同业务场景和模型规模的适配需要进一步验证。

---

## 312. PRISM-XR: Empowering Privacy-Aware XR Collaboration with Multimodal Large Language Models

**arXiv ID:** 2602.10154 | [PDF](https://arxiv.org/pdf/2602.10154v1)

**作者:** Jiangong Chen `[一作]` (Pennsylvania State University), Bin Li `[通讯]` (Pennsylvania State University)

**通讯引用:** 26038 | [OpenAlex ID](https://openalex.org/A5100365212)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e0540dec-d77f-42db-94ae-d039248f6393` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种支持多用户协作、隐私保护的基于多模态大语言模型的XR平台PRISM‑XR。

**💡 创新点**

创新点在于将边缘端隐私友好的图像处理、半标记式定位与可定制的内容同步结合，实现在低延迟、低带宽下的多用户协同与隐私保护。

**🔧 技术方法**

使用YOLOv11目标检测、OpenAI Whisper语音识别、GPT‑4o等LLM生成结构化JSON、AprilTag+XR设备内部定位进行注册、WebSocket+低字节同步协议等技术。

**📊 数据集**

采用自制的12类隐私级别物体数据集（杯、桌、椅、书等）进行隐私评估，并在实验环境中使用这些物体做测试。

**📈 对比分析**

与云端LLM（o1、o3‑mini、GPT‑4o）及边缘LLM（LLaVA、Llama3.2 Vision）在分类准确率、裁剪召回率、生成时长等指标上进行对比；注册平均延迟0.27 s，交互同步延迟低于15.5 ms，整体请求处理时间约7.7 s。

**⚠️ 局限性**

主要限制包括较高的总延迟（受云端推理限制）、关键词唤醒与语音转写鲁棒性不足、注册过程在复杂环境下需多次尝试、隐私确认易被忽略以及实验规模仅限两人、短任务，缺乏大规模验证。

---

## 313. Statistical Learning Analysis of Physics-Informed Neural Networks

**arXiv ID:** 2602.11097 | [PDF](https://arxiv.org/pdf/2602.11097v1)

**作者:** David A. Barajas-Solano `[一作]` (Pacific Northwest National Laboratory), David A. Barajas-Solano `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 1446 | [OpenAlex ID](https://openalex.org/A5053853892)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

将物理信息学习问题转化为统计学习问题，利用奇异学习理论分析PINN在热方程IBVP中的训练与损失曲面，并通过局部学习系数（LLC）评估损失的平坦度。

**💡 创新点**

将物理约束项视为无限间接数据而非正则化项，首次证明PINN学习是奇异学习问题，并用LLC量化损失曲面的平坦性及其对不确定性和外推性的影响。

**🔧 技术方法**

使用硬约束参数化的PINN、Adam随机优化、MCMC（NUTS）估计LLC、统计学习理论中的KL散度与自由能展开、Flax NNX与BlackJAX实现。

**📊 数据集**

采用热方程IBVP的解析解及其残差评估点，残差采样点从空间时间域Ω×(0,T]均匀/随机采样得到。

**📈 对比分析**

通过不同批量大小和学习率训练PINN，计算得到LLC≈9.5，表明不同初始化落入同一平坦区域；损失约1.85e-5，说明模型误差小但未能完全逼近真实分布。

**⚠️ 局限性**

仅适用于硬约束PINN，未处理软约束多重残差与权重，MCMC在高维参数空间中计算成本高，且实验仅基于单一热方程案例。

---

## 314. Developing Neural Network-Based Gaze Control Systems for Social Robots

**arXiv ID:** 2602.10946 | [PDF](https://arxiv.org/pdf/2602.10946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 315. Towards Probabilistic Strategic Timed CTL

**arXiv ID:** 2602.10824 | [PDF](https://arxiv.org/pdf/2602.10824v1)

**作者:** Wojciech Jamroga `[一作]` (Institute of Computer Science, Polish Academy of Sciences), Teofil Sidoruk `[通讯]` (Institute of Computer Science, Polish Academy of Sciences)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种概率性多代理系统的战略时序CTL（Probabilistic Strategic Timed CTL，PST-CTL）并实现了其模型检验。

**💡 创新点**

提出了在连续时间非同步多代理系统中引入概率无记忆策略（irP）进行战略推理的框架，并首次将其用于实际模型检验。

**🔧 技术方法**

使用参数化概率选择编码、离散化数字时钟、PRISM等工具实现模型检验，并结合概率分布与时序约束。

**📊 数据集**

采用经典的 Train‑Gate‑Controller (TGC) 车道控制案例进行实验，变量为列车数量 n。

**📈 对比分析**

与先前基于确定性策略的模型检验方法对比，实验表明对小规模模型（n≤5）检验时间在秒级至分钟级，但随着 n 增大显著增长，出现内存溢出。

**⚠️ 局限性**

主要限制在于编码概率策略导致状态空间爆炸，模型规模增长速度快；此外缺乏针对更大模型的近似或抽象技术。

---

## 316. 3D-Printed Anisotropic Soft Magnetic Coating for Directional Rolling of a Magnetically Actuated Capsule Robot

**arXiv ID:** 2602.10688 | [PDF](https://arxiv.org/pdf/2602.10688v1)

**作者:** Jin Zhou `[一作]` (University of Texas at Austin), Fangzhou Xia `[通讯]` (University of Texas at Austin)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5049403879)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

设计并制造了一种通过3D打印柔性磁性涂层实现滚动行走的软体胶囊机器人。

**💡 创新点**

创新点在于用可编程磁化的薄磁涂层替代内部永磁体，保留内部空间并实现可编程磁各向异性。

**🔧 技术方法**

采用硅胶-磁粉复合材料3D打印、可编程磁化、磁静态仿真（Ansys Maxwell）、动态模型与视觉跟踪实验。

**📊 数据集**

使用自制四种测试基底（光滑PLA、倾斜硅胶、干燥/湿润纹理硅胶）进行实验，不依赖公开数据集。

**📈 对比分析**

通过RMS误差对比四种表面，平滑基底误差最低，湿润硅胶误差最大；滚动速度约12.5 mm/s，最低磁场0.3 mT。

**⚠️ 局限性**

局限在平面模型、磁场均匀假设、对倾斜/摩擦变化敏感、需要精确磁体对准、缺乏实时闭环控制及体内环境适配。

---

## 317. Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking

**arXiv ID:** 2602.10743 | [PDF](https://arxiv.org/pdf/2602.10743v1)

**作者:** Vaisakh Shaj `[一作]` (University of Edinburgh), Amos Storkey `[通讯]` (University of Edinburgh)

**通讯引用:** 13680 | [OpenAlex ID](https://openalex.org/A5007901825)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Kalman Linear Attention（KLA）——一种将语言建模视为贝叶斯滤波的概率序列混合层，能够并行高效地更新隐藏状态并显式追踪不确定性。

**💡 创新点**

创新点在于将 Kalman 滤波器重参数化为信息形式，使其递推变为可结合的 Möbius（分式线性）映射，从而实现 𝒪(log T) 并行扫描；此外利用不确定性驱动的门控提升了表达能力，优于现有线性 SSM/GLA 模型。

**🔧 技术方法**

使用技术包括：线性高斯状态空间模型、Kalman 滤波、信息形式推理、Möbius 变换、并行前缀扫描、Triton 自定义算子，以及对比的 Mamba、GDN、GLA、mLSTM 等实现。

**📊 数据集**

实验数据集涵盖：MAD‑Lab 合成语言建模任务、Multi‑Query Associative Recall（MQAR）长序列检索任务，以及 A₅ 置换组合任务，用于检验表达力和记忆能力。

**📈 对比分析**

与基线模型比较时，KLA 在大多数 MAD‑Lab 任务上与或优于最先进的 SSM/GLA；在 MQAR 中显著超越 Mamba，并在 A₅ 任务中仅需 1–2 层即可完成，说明其更强的状态跟踪能力；同时保持与 SSM 同级的并行速度和内存效率。

**⚠️ 局限性**

局限性包括：未在大规模预训练或长上下文（如 web‑scale 语料）中测试；未充分利用后验协方差做不确定性评估；实验多采用浅层配置，未展示在更大规模模型上的性能。

---

## 318. Stability Analysis of Geometric Control for a Canonical Class of Underactuated Aerial Vehicles with Spurious Forces

**arXiv ID:** 2602.10961 | [PDF](https://arxiv.org/pdf/2602.10961v1)

**作者:** Simone Orelli `[一作]` (Sapienza University of Rome), Antonio Franchi `[通讯]` (Sapienza University of Rome)

**通讯引用:** 8154 | [OpenAlex ID](https://openalex.org/A5001771133)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10和ImageNet数据集进行实验。

**📈 对比分析**

与传统的激活函数模型进行了比较，结果显示新模型在分类精度上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

限制在于模型在处理高分辨率图像时的性能仍有待提高。

---

## 319. Self-Supervised Image Super-Resolution Quality Assessment based on Content-Free Multi-Model Oriented Representation Learning

**arXiv ID:** 2602.10744 | [PDF](https://arxiv.org/pdf/2602.10744v1)

**作者:** Kian Majlessi `[一作]` (University of Isfahan), Peyman Adibi `[通讯]` (Grenoble Alpes University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于自监督对比学习的无参考超分辨率图像质量评估方法S^3RIQA，并构建了用于预训练的大规模SRMORSS数据集。

**💡 创新点**

创新点包括①将不同SR模型视为独立降解模式，利用同一模型但不同内容的正样本实现内容无关的对比学习；②引入辅助缩放因子回归任务提升表征判别力；③针对真实低分辨率图像进行无监督预训练，消除传统合成数据偏差；④提供全新多模型多尺度SRMORSS数据集，为预训练提供丰富无标注样本。

**🔧 技术方法**

采用SimCLR/NT-Xent自监督对比框架，ConvNeXt‑Tiny编码器与投影头，辅助回归网络，色彩空间变换，多尺度裁剪与特征融合，以及下游线性回归（Ridge）进行质量预测。

**📊 数据集**

预训练使用SRMORSS（1,446张真实LR图像及13种SR方法生成的15,374张SR图像）；下游评估在RealSRQ、SRIJ和SREB这三个真实SR‑IQA基准数据集上进行。

**📈 对比分析**

在上述三个基准上与多种无参考及全参考指标（如C^2MT、PSCT、CN‑BSRIQA、PSNR/SSIM）对比，S^3RIQA在PLCC与SRCC均实现或逼近最高分数，显著优于现有NR‑IQA方法，并在不同尺度和消融实验中展现稳健提升。

**⚠️ 局限性**

局限性包括对色彩空间与尺度的依赖需额外预处理；在分布偏移较大时需额外辅助模块；负样本构造依赖已知SR模型，未知模型可能影响效果；仅针对单帧评估，未考虑视频或序列上下文；多裁剪策略虽提升性能但增加计算成本。

---

## 320. CryptoCatch: Cryptomining Hidden Nowhere

**arXiv ID:** 2602.10573 | [PDF](https://arxiv.org/pdf/2602.10573v1)

**作者:** Ruisheng Shi `[一作]` (Beijing University of Posts and Telecommunications), Chenfeng Wang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了 CryptoCatch 两阶段加密矿业流量检测框架，利用时间序列机器学习和主动探测精准识别矿业流量及其对应币种。

**💡 创新点**

创新点在于将流级时间序列特征与 XGBoost 多类分类相结合，再通过协议特定主动探测校验显著降低误报并实现多币种识别。

**🔧 技术方法**

采用 XGBoost 分类器、时间序列特征提取（统计、频域、连续小波变换）、主动探测的 Stratum 协议模板、贝叶斯优化超参数、Python/Scikit‑learn 等技术。

**📊 数据集**

使用自建的 7 种加密币主动/被动挖矿流量数据、CIC‑IDS‑2017 正常流量混合数据，以及公共/代理/私有矿池 URL 列表，实验共计 28,074 条网络流。

**📈 对比分析**

通过与单阈值 F1 模型对比，四种实验场景显示主动探测将误报率从约 30% 降至 <5%，F1‑score 0.99、召回率 0.99，多币种识别准确率 99.39%，系统可实时处理约 4000 流/秒。

**⚠️ 局限性**

局限性包括：对 VPN、协议伪装、流量混淆等高度加密/混淆场景识别能力不足；主动探测可能被矿池限流或 Token 校验拦截；实验主要在受控 LAN 环境，缺乏在更动态运营网络中的验证。

---

## 321. Rank-Accuracy Trade-off for LoRA: A Gradient-Flow Analysis

**arXiv ID:** 2602.10212 | [PDF](https://arxiv.org/pdf/2602.10212v1)

**作者:** Michael Rushka `[一作]` (Northwestern University), Diego Klabjan `[通讯]` (Northwestern University)

**通讯引用:** 5160 | [OpenAlex ID](https://openalex.org/A5013049879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文通过梯度流（GF）分析，研究低秩适配器LoRA在不同秩r下的精度与损失表现；

**💡 创新点**

提出了对LoRA更新方式（同步与顺序）不敏感的梯度流方程，并给出闭式解，揭示LoRA对秩的精度依赖；

**🔧 技术方法**

使用连续时间梯度流理论、矩阵分解、奇异值分解及闭式微分方程求解等技术；

**📊 数据集**

未在实验中使用具体数据集，主要是理论推导与解析结果；

**📈 对比分析**

通过理论推导比较了全秩梯度下降与LoRA限制秩梯度下降的最终损失与近似误差；在trace‑squared目标下，LoRA在任意秩下可达零损失，期望相对误差随秩~r⁻¹/²衰减；在低秩逼近任务中，LoRA收敛到Eckart–Young–Mirsky最优解，误差取决于残余奇异值；

**⚠️ 局限性**

局限在于仅考虑了两种简化损失函数，假设了可实现的谱初始化，未讨论随机梯度下降的梯度流；未来需扩展至更复杂任务、数据集以及无SVD初始化方案。

---

## 322. OccFace: Unified Occlusion-Aware Facial Landmark Detection with Per-Point Visibility

**arXiv ID:** 2602.10728 | [PDF](https://arxiv.org/pdf/2602.10728v1)

**作者:** Xinhao Xiang `[一作]` (University of California), Weiyang Li `[通讯]` (Genies inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 OcFace 框架，实现统一 100 点关键点检测并同时预测每点可见性，适用于人类及多样化人类相似面孔在遮挡与大角度姿态下的定位。

**💡 创新点**

创新点在于：①统一的 100 点密集布局；②局部+跨点上下文相结合的可见性预测模块；③基于遮挡的伪可见性监督与门控融合；④完整的遮挡评估套件与新的 Genie‑Face 数据集。

**🔧 技术方法**

使用热图回归与堆叠 hourglass 结构、点与边几何辅助图、门控融合的可见性头、遮挡增强与伪可见性标注方法，以及多任务联合损失。

**📊 数据集**

训练与评估使用 Genie‑Face（15,475 张人类相似面孔），以及 COFW、300W、WFLW、COCO 等公开人脸关键点数据集。

**📈 对比分析**

与 ViTPose、ORFormer、ADNet 等基线对比，标准数据集 NME 与失真率保持竞争力；在 Genie‑Face 上可见性评估中 NME_occ 下降 40%+、Occ AP、F1、ROC‑AUC 取得显著提升，证明对遮挡和大姿态的鲁棒性。

**⚠️ 局限性**

局限性包括：对极端遮挡/极端姿态的泛化仍有限；需手工标注可见性标签导致标注成本高；额外可见性头增加模型复杂度和推理时间。

---

## 323. Binary Flow Matching: Prediction-Loss Space Alignment for Robust Learning

**arXiv ID:** 2602.10420 | [PDF](https://arxiv.org/pdf/2602.10420v1)

**作者:** Jiadong Hong `[一作]`, Zhaoyang Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在二值数据上应用流匹配（Flow Matching）的稳健训练方法，提出了预测-损失空间对齐（Prediction–Loss Alignment）来消除x-prediction与velocity loss的结构性不匹配；

**💡 创新点**

创新点在于揭示并理论证明了x-prediction与v-loss的匹配会导致梯度方差三阶发散，并首次提供了通过对齐损失空间来实现采样器无关稳定性的原则；

**🔧 技术方法**

采用了连续时间流匹配框架、信号空间预测（x-prediction）、对齐的MSE/BCE损失、Logit-Normal时间采样以及基于U-Net/DiT的网络结构；

**📊 数据集**

实验数据集包括二值化MNIST（Binary MNIST）和多输入多输出（MIMO）信号检测数据；

**📈 对比分析**

通过与不对齐（x-prediction+v-loss）、Logit-Normal调度、以及传统FM（v-prediction+v-loss）等方案比较，发现对齐后x-MSE在Binary MNIST上获得最低FID，BCE在MIMO检测上实现最低BER，且训练过程更平稳；

**⚠️ 局限性**

局限性包括对对齐策略的依赖、对网络容量和超参数的敏感性、以及未在更大规模或其他离散数据上验证的通用性。

---

## 324. ACE-RTL: When Agentic Context Evolution Meets RTL-Specialized LLMs

**arXiv ID:** 2602.10218 | [PDF](https://arxiv.org/pdf/2602.10218v1)

**作者:** Chenhui Deng `[一作]` (NVIDIA), Haoxing Ren `[通讯]` (Agentrys)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ACE-RTL框架，将专门训练的RTL LLM与前沿通用LLM结合，实现多轮迭代生成与纠错；

**💡 创新点**

创新点在于Agentic Context Evolution (ACE)机制，三组件（Generator、Reflector、Coordinator）协同迭代，以及并行缩放策略显著减少迭代次数；

**🔧 技术方法**

技术包括大规模RTL数据集构建与自监督微调、使用Claude4-Sonnet作为反思引擎、Icarus Verilog仿真反馈、vLLM服务部署、并行多进程并行缩放；

**📊 数据集**

使用约170万条规格–RTL对的数据集（从500万原始RTL脚本筛选并生成），涵盖代码生成、修改、调试等任务；

**📈 对比分析**

与14个竞争基线（含GPT‑5、Claude4‑Sonnet、ScaleRTL等）在CVDP基准上对比，ACE‑RTL在Pass@1和APR上分别提升至44.87%及平均仅需4轮迭代；

**⚠️ 局限性**

局限在于尚未支持测试平台生成、主观评估、某些复杂设计域的细粒度验证，以及对硬件专用工具链的依赖仍有限。

---

## 325. Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters

**arXiv ID:** 2602.10604 | [PDF](https://arxiv.org/pdf/2602.10604v1)

**作者:** Ailin Huang `[一作]`, Zixuan Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种稀疏 Mixture-of-Experts（MoE）架构的语言模型，通过 196B 总参数、11B 活跃参数实现高效推理，同时结合 3:1 滑动窗口/全注意力混合与多标记预测（MTP‑3）显著降低多轮代理交互的延迟。

**💡 创新点**

创新点包括：① 采用头维度门控滑动窗口注意力以兼顾长距建模与并行推理；② 引入 EP‑Group 平衡 MoE 路由与权重裁剪机制，解决专家冲突与激活爆炸；③ 设计 MIS‑PO（Metropolis‑Independent Sampling‑Filtered Policy Optimization）和路由置信度监控，以实现对长轨迹的高效、低方差强化学习；④ 通过多阶段预训练、mid‑训练与后训练的统一 RL 框架实现从通用到专家的自我提升。

**🔧 技术方法**

主要技术包括：稀疏 MoE、滑动窗口注意力、头维门控注意力、MTP‑3、Muon 优化器与 ZeRO‑1 重构、EP‑Group 负载平衡、MIS‑PO、路由置信度、截断感知值回溯、统一 RL 奖励系统（可验证奖励与非可验证奖励）。

**📊 数据集**

使用了规模达 17.6T 令牌的通用网页/书籍数据、代码/PR/Issue/Commit 语料、合成数学与工具使用数据，mid‑训练阶段再使用 750B 令牌；评估时使用 IMO‑AnswerBench、LiveCodeBench‑v6、τ²‑Bench、BrowseComp、Terminal‑Bench 2.0、C‑Eval、MMLU、HumanEval、MultiPL‑E、以及多任务对比基准。

**📈 对比分析**

在上述基准上，模型取得 85.4%（IMO‑AnswerBench）、86.4%（LiveCodeBench‑v6）、88.2%（τ²‑Bench）、69%（BrowseComp with Context Manager）、51%（Terminal‑Bench 2.0）等成绩，整体表现与 GPT‑5.2 xHigh、Gemini 3.0 Pro 相当，并显著优于同等或更大规模的开源 MoE 模型。

**⚠️ 局限性**

局限性包括：① 生成时需要较长序列以达到相同质量；② 对超大规模长距推理的效率尚需进一步压缩；③ 目前 RL 仍主要在学术基准上验证，缺乏对真实工业工作流的鲁棒性；④ 在分布偏移、长回合对话与多语言混杂时可能出现稳定性下降。

---

## 326. LHAW: Controllable Underspecification for Long-Horizon Tasks

**arXiv ID:** 2602.10525 | [PDF](https://arxiv.org/pdf/2602.10525v1)

**作者:** George Pu `[一作]` (Scale AI), Samuel Marc Denton `[通讯]` (Scale AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个可控的合成不足规格化流水线，用于在长期工作流中生成并验证任务的模糊变体，以评估代理的澄清行为。

**💡 创新点**

提出三阶段（段落提取、候选生成、经验验证）流水线，按四维信息（目标、约束、输入、上下文）系统地生成可调严重度的不足规格，并通过多轮代理试验进行客观分类；同时引入Gain/Q指标衡量澄清效率。

**🔧 技术方法**

基于大型语言模型的语义提取、生成与评分；多轮代理推理实验；Ask_User工具模拟用户；统计指标（pass@3、checkpoint、Ask%、Avg/Q、Gain/Q）。

**📊 数据集**

OwnCloud子集、SWE-Bench-Pro、MCP-Atlas、TAC任务，共生成285个变体。

**📈 对比分析**

对5大前沿模型（Claude Opus-4.5、Claude Sonnet-4.5、Gemini-3-Pro、Gemini-3-Flash、GPT-5.2）在原始与模糊任务上进行pass@3与checkpoint比较；澄清能显著提升性能，但未完全恢复基线；不同模型澄清效率差异明显，GPT-5.2提问多但Gain/Q低，Gemini-3-Pro提问少但Gain/Q高。

**⚠️ 局限性**

仅处理提示层面的不足规格，无法捕捉环境级或语义对立的歧义；缺乏人类验证；对多模态或动态任务的适用性未知；未构建完整成本决策框架。

---

## 327. How segmented is my network?

**arXiv ID:** 2602.10125 | [PDF](https://arxiv.org/pdf/2602.10125v1)

**作者:** Rohit Dube `[一作]` `[通讯]` (Independent Researcher), Rohit Dube (Independent Researcher)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于图的‘segmentedness’度量，定义为网络全局边密度的补数，并给出通过随机抽样得到的无偏估计器及其Wald置信区间；

**💡 创新点**

首次将边密度作为量化网络分割度量，证明在任意规模网络上仅需约100个样本即可估计，并提供统计置信保证；

**🔧 技术方法**

使用随机抽样、伯努利试验、Wald置信区间、蒙特卡罗仿真以及Erdős–Rényi与随机块模型验证等技术；

**📊 数据集**

主要利用合成网络（Erdős–Rényi和随机块模型）进行模拟验证，未使用真实网络数据集；

**📈 对比分析**

通过模拟检验估计器的无偏性和95%置信区间覆盖率接近0.95，样本量与网络规模无关，且在不同结构模型下保持良好性能；

**⚠️ 局限性**

局限于二值对称连通性；不考虑协议/端口细节、动态或非对称安全策略，忽略结构性安全特征，且采样假设均匀随机，实际工具可能导致偏差。

---

## 328. Exploiting the Structure in Tensor Decompositions for Matrix Multiplication

**arXiv ID:** 2602.11041 | [PDF](https://arxiv.org/pdf/2602.11041v1)

**作者:** Manuel Kauers `[一作]` (Johannes Kepler University), Isaac Wood `[通讯]` (Johannes Kepler University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

提出一种利用张量分解结构的新递归算法，显著降低6×6矩阵乘法的指数，并对多种尺寸的矩阵乘法张量进行结构化分解。

**💡 创新点**

创新点在于：1) 识别并合并共享输入/输出的递归调用，将多次小张量乘法合并为更大矩阵乘法，从而使实际指数低于仅依据张量秩给出的上界；2) 系统性搜索包含1×2×2、k×1×1等结构的分解，并通过对称群、Hensel提升等手段保持稀疏性。

**🔧 技术方法**

使用了张量分解与限制、直接与克罗内克积、Flip‑graph 搜索、DeGroote 对称群、Hensel 提升、Mårtensson‑Wagner 公共子表达式消除，以及递归模拟来评估性能。

**📊 数据集**

研究针对多种尺寸（4×4、5×5、6×6、3×3×4、2×3×8、2×4×4等）矩阵乘法张量进行分解；主要在 \,mathbb{Z}_2 \ 上进行符号计算，并将分解结果发布在 GitHub。

**📈 对比分析**

通过理论上的指数与领先系数，以及对不同输入尺寸的递归模拟，比较新算法与 Winograd‑Strassen 及标准算法的操作数。模拟显示，当矩阵尺寸约为 10^6 时开始优于 Strassen，10^10 规模则表现更为显著。

**⚠️ 局限性**

局限性：算法主要在理论层面验证，实际实现需处理零填充、矩形子问题和递归深度；指数提升仅在极大规模矩阵中显现；搜索与符号计算复杂，未给出直接可用于工业级实现的高效代码。

---

## 329. SecureScan: An AI-Driven Multi-Layer Framework for Malware and Phishing Detection Using Logistic Regression and Threat Intelligence Integration

**arXiv ID:** 2602.10750 | [PDF](https://arxiv.org/pdf/2602.10750v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 330. Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents

**arXiv ID:** 2602.10715 | [PDF](https://arxiv.org/pdf/2602.10715v1)

**作者:** Yifei Li `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 74110 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 LoCoMo-Plus benchmark，重新设计长短期对话记忆评估方式，关注隐式约束（如用户目标、价值观）在长对话中的保留与应用；同时构建统一的约束一致性评价框架。

**💡 创新点**

创新点在于：① 把记忆任务从显式事实回忆转向认知记忆，关注cue–trigger语义断裂下的隐式约束；② 通过生成、筛选、验证流程构造高质量cue–trigger对；③ 用约束一致性取代表面匹配与任务显式提示，消除评估偏差。

**🔧 技术方法**

技术包括：LLM生成与人工筛选（生成cue/trigger对）、语义相似度过滤（BM25、MPNet）、LLM判别器（评估约束一致性）、检索增强（文本嵌入+RAG）、多种长期记忆系统（Mem0、SeCom、A‑Mem）以及多种大型模型（Qwen系列、GPT‑4o、Gemini）。

**📊 数据集**

使用 LoCoMo‑Plus 数据集（从 LoCoMo 迁移并手工验证的 cue–trigger 对），以及原始 LoCoMo 作为基准；检索使用 OpenAI 的多种文本嵌入模型；评估数据均为英语对话。

**📈 对比分析**

通过统一输入（无任务显式提示）与约束一致性判别，比较开源 LLM、闭源 LLM、RAG 方法和记忆系统在 LoCoMo 与 LoCoMo‑Plus 上的表现。实验显示 LoCoMo‑Plus 任务对所有方法仍极具挑战，性能差距显著，且不同模型间的差异在认知记忆场景下被压缩，证明评估偏差被消除。

**⚠️ 局限性**

局限性包括：1）LoCoMo‑Plus 仅为诊断规模较小的数据集，不能用于大规模训练；2）评估依赖 LLM 判别器，可能受判别模型和提示设计影响；3）仅覆盖英语对话，未涉及情绪、信念更新、多主体交互等更复杂的认知记忆；4）未探讨实际部署中的隐私与记忆管理问题。

---

## 331. Pitch Angle Control of a Magnetically Actuated Capsule Robot with Nonlinear FEA-based MPC and EKF Multisensory Fusion

**arXiv ID:** 2602.10610 | [PDF](https://arxiv.org/pdf/2602.10610v1)

**作者:** Chongxun Wang `[一作]` (University of Texas at Austin), Fangzhou Xia `[通讯]` (University of Texas at Austin)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5049403879)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一套基于磁场有限元仿真和约束模型预测控制（MPC）的可吞咽胶囊机体倾角控制系统，并通过惯性测量单元（IMU）与低频视觉观测融合实现了在摄像率降低至 1 Hz 时的闭环控制。

**💡 创新点**

创新点包括：① 将三维有限元磁场模拟产生的角度相关磁力/力矩查找表直接嵌入到刚体倾角动力学模型中，实现了非线性磁驱动的精确建模；② 采用约束 MPC 对电流轨迹进行预测和优化，在满足电流幅值和上升率限制的前提下大幅提升倾角调节速度；③ 通过 EKF 将 50 Hz IMU 数据与 1 Hz 视觉测量融合，解决了外部成像稀疏时的状态估计延迟问题，提升了系统鲁棒性。

**🔧 技术方法**

主要技术手段包括：三维有限元磁场仿真（ANSYS Maxwell）生成力/力矩 lookup 表；控制导向的刚体倾角动力学模型；约束模型预测控制（MPC）与离散一阶电流驱动模型；扩展卡尔曼滤波（EKF）实现多频率传感器融合；蓝牙低功耗（BLE）实现无线 IMU 数据传输；LabVIEW + MATLAB Script Node 实现实时 QP 求解与电流驱动。

**📊 数据集**

该研究没有使用公开数据集；所有数据均来自内部仿真（有限元力/力矩表）与实验平台（硅胶胃壁模型、摄像头采集的视觉角度、IMU 原始传感值）。

**📈 对比分析**

通过与基线“开关控制”（on–off）以及不同摄像头更新速率（30 Hz、5 Hz、1 Hz）下的 MPC 对比实验，验证了性能。结果显示：MPC（30 Hz 视觉）在 0°→30° 或 90°→30° 的倾角重定位任务中，±2.5° 逼近窗口内收敛时间约为 4–6 s；基线控制需 20 s 以上；MPC（5 Hz 视觉）出现不稳定；采用 EKF 融合 IMU 与 1 Hz 视觉后，系统仍保持稳定收敛，仅略慢至 5–6 s，说明在极低成像率下仍能保持可用性。

**⚠️ 局限性**

主要限制包括：① MPC 计算耗时（每步约 100–150 ms），限制了控制更新频率；② EKF 估计在 1 Hz 视觉更新时仍存在延迟与漂移，导致收敛略慢；③ 当前系统仅在二维倾角控制，未涉及滚转与偏航耦合；④ 对胃壁弹性、摩擦等非线性不确定性处理不足，鲁棒性待提升。

---

## 332. The Complexity of Strategic Behavior in Primary Elections

**arXiv ID:** 2602.10290 | [PDF](https://arxiv.org/pdf/2602.10290v1)

**作者:** Colin Cleveland `[一作]` (King's College London), Maria Polukarov `[通讯]` (King's College London)

**通讯引用:** 862 | [OpenAlex ID](https://openalex.org/A5065622718)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文以算法博弈论和计算社会选择为工具，对民主党内初选机制的战略结构进行形式化建模与复杂性分析。

**💡 创新点**

创新点在于将初选视为多阶段投票博弈，证明了从最简单的两阶段模型到多阶段顺序模型，最优反应、纳什均衡验证与存在问题的复杂度分别从NP、co‑NP、Σ₂ᴾ升级到PSPACE，揭示时间顺序和策略可预见性对策略计算的指数级影响。

**🔧 技术方法**

主要技术包括：①构造投票博弈的完整策略空间并利用可查询的有限空间GE决策函数；②使用从SAT/3‑SAT到QBF的多阶段还原；③利用Presburger算术对小规模参数的可解性讨论；④在顺序初选中引入信息历史并进行深度优先搜索证明PSPACE可判定性。

**📊 数据集**

本文没有使用实证数据集，而是通过理论构造与归约证明来展示复杂性。

**📈 对比分析**

对比方法主要是理论复杂度对比：在传统直接选举中已知的多项式时间算法，本文通过归约展示初选的NP/Σ₂ᴾ/PSPACE难度，说明在相同投票规则下，初选增加了显著的计算负担。

**⚠️ 局限性**

局限性包括：①仅考虑单纯的FPTP投票和固定分区投票规则；②假设投票者完全理性且拥有完整信息；③未涉及混合策略、候选人策略或动态学习；④对实际选举的可解释性和实验验证不足。

---

## 333. Flow-Enabled Generalization to Human Demonstrations in Few-Shot Imitation Learning

**arXiv ID:** 2602.10594 | [PDF](https://arxiv.org/pdf/2602.10594v1)

**作者:** Runze Tang `[一作]` (Australian National University), Penny Sweetser `[通讯]` (Australian National University)

**通讯引用:** 480 | [OpenAlex ID](https://openalex.org/A5063432097)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

使用人类视频与机器人演示训练跨身体的场景流预测模型SFCr，并基于流与裁剪点云的动作策略FCrP，使机器人能在极少演示下快速学习交互任务。

**💡 创新点**

将Transformer解码器用于全场景流预测，结合点云裁剪与流条件的扩散策略，显著提升数据效率与跨身体泛化，解决了先前仅关注对象或机器人臂流预测的局限。

**🔧 技术方法**

Transformer解码器、点云分组与DP3点云编码器、扩散策略、CoTracker追踪、FastSam分割、随机裁剪与点云掩蔽。

**📊 数据集**

30段人类视频 + 10条机器人演示（每个任务）以及公开的RGBD/点云数据集用于预训练与评估。

**📈 对比分析**

在7个真实世界任务中与DP3、RISE、SUGAR对比，成功率提升至96.67%，并在仅有人类视频的任务中保持高成功率，展示出强大的空间与实例泛化能力。

**⚠️ 局限性**

未考虑演示速度差异导致的流长变化；训练使用随机查询点而执行时使用裁剪点云导致预测误差；点云掩蔽可能削弱精细动作性能。

---

## 334. Breaking the Curse of Repulsion: Optimistic Distributionally Robust Policy Optimization for Off-Policy Generative Recommendation

**arXiv ID:** 2602.10430 | [PDF](https://arxiv.org/pdf/2602.10430v1)

**作者:** Jie Jiang `[一作]`, Jun Zhang `[通讯]` (Tencent Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对工业推荐系统中离线强化学习面对极端噪声导致的模型崩溃问题，提出分布鲁棒策略优化（DRPO）框架，并通过乐观分布鲁棒优化推导出硬阈值过滤机制。

**💡 创新点**

创新点包括：将离线RL视为乐观分布鲁棒优化，证明硬阈值（Top‑κ）过滤为最优解；设计动态信噪比引导的课程学习和自适应信赖域，实现对噪声的物理剔除与稳定收敛的统一。

**🔧 技术方法**

使用技术包括分布鲁棒优化（Optimistic DRO）、CVaR变分双重形式、硬阈值过滤、软基础稳定训练、动态信噪比调节以及自适应信赖域。

**📊 数据集**

实验基于工业级高保真仿真环境 RecSim，生成混合策略日志，涵盖中等质量（曝光日志）和极端噪声（全检索）两种场景。

**📈 对比分析**

与 8 类代表性方法（APG、BC、CRR、BPPO、AWR、IQL、AsymRe、Adaptive BC）进行对比，DRPO 在两种噪声场景均位列前两名；在极端噪声下仍能保持约 0.38 的奖励和 1.87 的 eCPM，显著优于软加权和价值基方法。

**⚠️ 局限性**

局限性：硬阈值过滤依赖奖励分位数估计，可能导致样本稀疏时过拟合；当前设计主要针对连续动作空间，对离散大规模或多步返回未知场景的适用性仍需进一步验证。

---

## 335. Towards Term-based Verification of Diagrammatic Equivalence

**arXiv ID:** 2602.11035 | [PDF](https://arxiv.org/pdf/2602.11035v1)

**作者:** Julie Cailler `[一作]` (Université de Lorraine), Sophie Tourret `[通讯]` (Max Plank Institute for Informatics)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了字符串图的自动化等价推理，为两类图形引入了归约系统，并通过 Isabelle/HOL 证明其终止性与可合并性。

**💡 创新点**

创新点在于：①将二维图形问题转化为一维项归约；②设计了可在证明助手中验证的归约系统；③证明了系统的终止性与可合并性，填补了图形等价判定的理论空白。

**🔧 技术方法**

采用的技术主要是：字符串图的形式化定义、项归约（term rewriting）、一致性与终止性证明、以及 Isabelle/HOL 证明助手的形式化验证。

**📊 数据集**

本文未使用任何实验数据集，工作以理论证明为主。

**📈 对比分析**

由于是理论验证，未进行实验比较；主要通过 Isabelle/HOL 的形式化验证展示了方法的正确性与健壮性。

**⚠️ 局限性**

局限性包括：①仅针对两类特定图形，可能不适用于更一般的图形；②缺乏实验评估，未能展示在实际量子电路验证中的性能表现；③归约系统的实现细节与可扩展性仍待进一步研究。

---

## 336. Control Reinforcement Learning: Token-Level Mechanistic Analysis via Learned SAE Feature Steering

**arXiv ID:** 2602.10437 | [PDF](https://arxiv.org/pdf/2602.10437v1)

**作者:** Seonglae Cho `[一作]` (Holistic AI), Adriano Koshiyama `[通讯]` (Holistic AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练策略网络在每个token生成时挑选稀疏自编码器(SAE)特征进行放大干预，输出可解释的逐token干预日志。

**💡 创新点**

将控制强化学习与SAE特征结合，提出Adaptive Feature Masking保持单特征可解释性，同时提供分支点追踪、评论家轨迹分析等新型解释工具。

**🔧 技术方法**

使用PPO强化学习、稀疏自编码器、残差流观测、单特征干预与Adaptive Feature Masking等技术。

**📊 数据集**

在Gemma-2 2B与Gemma Scope SAE上，使用MMLU、BBQ、GSM8K、HarmBench、XSTest等基准数据集进行评估。

**📈 对比分析**

与基准模型、约束解码、随机/最活跃特征等做对比，CRL在大多数任务上提升数个百分点，并在可解释性方面优于静态特征选择。

**⚠️ 局限性**

仅支持单特征干预导致多特征协同效应受限；PPO训练耗时且对不同层、任务的调参较复杂。

---

## 337. ECHO: An Open Research Platform for Evaluation of Chat, Human Behavior, and Outcomes

**arXiv ID:** 2602.10295 | [PDF](https://arxiv.org/pdf/2602.10295v1)

**作者:** Jiqun Liu `[一作]` (University of Oklahoma), Ran Yu `[通讯]` (GESIS – Leibniz Institute for the Social Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个可低代码配置、支持聊天与搜索双模态的人机交互实验平台——ECHO，用于收集和分析用户在多轮对话和传统搜索中的行为与体验。

**💡 创新点**

创新点包括：①统一的实验工作流管理，研究者可通过仪表盘完成任务、问卷和触发式测量的配置；②细粒度交互日志（提示、回答、点击、时间戳）与用户主观反馈的同步采集；③支持多种LLM与搜索API的无缝集成；④可导出结构化CSV，便于跨学科后续分析；⑤在同一平台实现聊天与搜索的直接对比，降低跨工具研究的技术门槛。

**🔧 技术方法**

技术实现：前端React + TailwindCSS；后端采用Firebase（Firestore、Authentication、Cloud Functions）实现无服务器架构；与OpenAI、Google Gemini、Anthropic Claude等LLM接口以及第三方搜索API对接；Node.js作为服务端脚本运行环境；数据导出使用Firebase SDK生成CSV。

**📊 数据集**

平台自身不依赖特定公开数据集，而是收集实验参与者生成的交互日志与问卷数据。研究者可自由指定任务内容与目标，后续可结合公开的检索或对话数据集做对照实验。

**📈 对比分析**

比较方法：在同一实验流程中分别部署聊天任务与搜索任务，对比预后/后测问卷（期望与成效）、交互行为（如提示/点击次数、停留时间）以及用户满意度、信任度等主观指标。平台未提供预设的基准性能数值，主要强调可重复性和跨方法的数据可比性。

**⚠️ 局限性**

局限性：①受限于外部API调用额度与速率，实验规模受限；②仅支持现有LLM和搜索接口，无法离线部署；③平台依赖Firebase免费层，若用户量大需额外付费；④复杂的多步实验逻辑仍需手工调整；⑤缺乏大规模验证和多语言/文化适配的实证支持。

---

## 338. Near-Feasible Stable Matchings: Incentives and Optimality

**arXiv ID:** 2602.10851 | [PDF](https://arxiv.org/pdf/2602.10851v1)

**作者:** Frederik Glitzner `[一作]` (University of Glasgow), Frederik Glitzner `[通讯]` (University of Glasgow)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5099022632)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在非二分、多容量的稳定匹配框架（Stable Fixtures）中研究近可行稳定匹配，构建了评估代理人偏差激励的新框架，并提出了最小化容量修改（MIM、MAM）和最小化偏差激励（MIDI、MADI）的最优性概念，给出了相应的多项式时间算法与理论上限；

**💡 创新点**

首次将近可行性与代理人偏差激励结合，提出了阻塞条目（blocking entry）作为新不稳定性度量；证明了在任意实例中存在“平衡”容量修改和匹配，使得每个代理人最多只有一次偏差激励，并且容量修改与总偏差激励均可在多项式时间内最小化；

**🔧 技术方法**

利用Generalised Stable Partition（GSP）结构进行组合分析；设计了基于GSP的容量修改算法（NearFeasible_+、NearFeasible_-等）；构造了XP算法和ILP模型求解MADI_c与MIDI_c；在实验中使用Python实现、PuLP + Gurobi求解ILP；

**📊 数据集**

随机生成的完全可接受偏好实例，规模n∈{10,12,…,40}，容量c∈{1,3,5,7}，共1000个实例每组；

**📈 对比分析**

通过计算平均所需容量修改数、平均阻塞条目数来比较；实验表明平均容量扩展与阻塞条目数随n线性增长，但绝对值均极小（<1或最多4），MADI_c匹配往往仅产生≤2个阻塞条目；在可解实例上，XP算法与ILP在小最优值下效率良好；

**⚠️ 局限性**

对MIDI_c和MADI_c的最优求解仍为NP/para‑NP难，算法在仅减小容量或仅使用完整偏好的设定下表现不佳；实验仅限于随机实例，未验证在真实匹配市场中的稳健性；

---

## 339. LOREN: Low Rank-Based Code-Rate Adaptation in Neural Receivers

**arXiv ID:** 2602.10770 | [PDF](https://arxiv.org/pdf/2602.10770v1)

**作者:** Bram Van Bolderik `[一作]`, Manil Dev Gomony `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 383 | [OpenAlex ID](https://openalex.org/A5086060203)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

设计并实现了低秩适配的多码率神经接收机LOREN，能够在不同码率下动态切换，仅需极小的适配参数。

**💡 创新点**

创新点是将LoRA理念迁移到卷积接收机，冻结基网络，只学习每个码率的1×1低秩适配器，显著降低存储和功耗。

**🔧 技术方法**

使用的技术包括卷积残差网络（Sionna）、低秩LoRA适配器、端到端训练、3GPP CDL-C 信道仿真以及基于HLS的22nm FD‑SOI硬件实现。

**📊 数据集**

数据集为通过3GPP CDL-C 通道仿真生成的 OFDM QAM‑16 随机包，覆盖多种 SNR 与三种码率（0.5、0.66、0.75）。

**📈 对比分析**

通过与完美CSI、LS估计基线以及单独训练的完整神经接收机进行BLER比较，LOREN在三码率下与完整模型相当或更优，硬件上功耗降低约15%，面积缩减约65%。

**⚠️ 局限性**

局限性包括仅验证了三码率、QAM‑16、SIMO单天线场景；对更高码率、更复杂调制、更多天线及更大网络规模尚未评估，且LoRA超参需经验调优。

---

## 340. Resource-Efficient Model-Free Reinforcement Learning for Board Games

**arXiv ID:** 2602.10894 | [PDF](https://arxiv.org/pdf/2602.10894v1)

**作者:** Kazuki Ota `[一作]` (University of Tokyo), Tatsuya Harada `[通讯]` (University of Tokyo)

**通讯引用:** 10769 | [OpenAlex ID](https://openalex.org/A5042711470)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种完全不使用搜索的模型无监督强化学习算法 KLENT，用于棋类游戏实现高效学习。

**💡 创新点**

通过结合 KL 正则化、熵正则化和 λ 回报，完全消除搜索，显著提升学习效率。

**🔧 技术方法**

使用反向 KL 正则化的策略优化、熵正则化、λ 回报估计以及 ResNet6 残差网络进行特征提取。

**📊 数据集**

在五种中等规模棋类游戏（动物将棋、加德纳象棋、9x9 围棋、六角棋、黑白棋）以及 19x19 围棋上进行实验。

**📈 对比分析**

与 AlphaZero、TRPO AlphaZero、Gumbel AlphaZero、DQN、PPO 等基线对比，KLENT 在模拟器评估次数上实现数倍效率提升，胜率与搜索基准相当甚至更高。

**⚠️ 局限性**

未证明在无限计算资源下是否能与搜索方法竞争，且对大型游戏的扩展仍有限。

---

## 341. A Multimodal Conditional Mixture Model with Distribution-Level Physics Priors

**arXiv ID:** 2602.10451 | [PDF](https://arxiv.org/pdf/2602.10451v1)

**作者:** Jinkyo Han `[一作]` (Northwestern University), Bahador Bahmani `[通讯]` (Northwestern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种基于混合密度网络的物理信息多模态条件建模框架，并在四类物理系统上验证其性能。

**💡 创新点**

引入了组件特定的物理正则化、类别条件化机制以及与传统流匹配模型的对比，提供可解释的多模态生成模型。

**🔧 技术方法**

使用混合密度网络（MDN）、物理约束正则化、类别条件化训练、与条件流匹配（CFM）比较，以及自动微分求解物理残差。

**📊 数据集**

以分岔系统、双尺度随机微分方程、单晶 3C‑SiC 冲击 Hugoniot 数据、Chafee–Infante 反应扩散方程的数值模拟数据为例。

**📈 对比分析**

与同等参数的条件流匹配模型比较，MDN 在分岔、双稳态以及冲击等场景下获得与 CFM 相近或更好分布拟合，同时训练与采样更简单。

**⚠️ 局限性**

仅采用条件高斯假设，难以处理非高斯多模态；未显式建模模型不确定性；需要更灵活的分布或更复杂的物理约束。

---

## 342. Smart Lotteries in School Choice: Ex-ante Pareto-Improvement with Ex-post Stability

**arXiv ID:** 2602.10679 | [PDF](https://arxiv.org/pdf/2602.10679v1)

**作者:** Haris Aziz `[一作]` (UNSW Sydney), Tom Demeulemeester `[通讯]` (Maastricht University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于“智能彩票”的学校选择机制，在保持后验稳定性的前提下，对给定随机匹配进行 SD 改进，并通过整数规划+列生成求解最优改进随机匹配。

**💡 创新点**

创新点：① 在随机化前实现预期福利提升，仍保证后验稳定性；② 设计了一个列生成框架，利用定价问题动态生成弱稳定匹配，求解 SD 改进最优随机匹配；③ 对相关判定与优化问题的复杂度给出了精确分析。

**🔧 技术方法**

使用技术：整数规划（IP）、列生成（column generation）、定价子问题（基于稳定匹配的 LP/ILP）、随机化的 Deferred Acceptance 及 Stable Improvement Cycles 算法。

**📊 数据集**

数据集：① 通过学生与学校位置、偏好、距离等生成的合成实例；② 真实数据为 2015 年爱沙尼亚 Harku 幼儿园的 152 名家庭、7 所学校的偏好与优先级信息。

**📈 对比分析**

比较方法与性能：与 Erdil‑Ergin 的稳定改进循环（EE）、EADA（效率调整 DA）以及随机 DA/ RSD 进行对比。实验显示，DA‑CG 在平均排名上比 EE 提升显著，尤其在偏好相关度高时平均改进可达 1.05 位；在真实数据中，DA‑CG 的改善比例可达 60%+，平均排名提升明显；列生成方法在 10 分钟内比单纯启发式提升约 0.06 位。

**⚠️ 局限性**

局限性：① 计算量大，求解时间随实例规模增长显著；② 只考虑后验稳定性，未兼顾前验稳定或多目标优化；③ 需要大量初始随机匹配或稳定匹配来启动列生成，采样不足时可能影响结果；④ 对极大规模实例的可扩展性仍有待改进。

---

## 343. Monte Carlo Maximum Likelihood Reconstruction for Digital Holography with Speckle

**arXiv ID:** 2602.10344 | [PDF](https://arxiv.org/pdf/2602.10344v1)

**作者:** Xi Chen `[一作]` (Rutgers University), Shirin Jalali `[通讯]` (Rutgers University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于投影梯度下降和随机矩阵估计（PGD-MC）的最大似然估计框架，用于在相干成像中直接处理多重斑点噪声而无需显式求逆协方差矩阵。

**💡 创新点**

创新点在于将随机迹估计与共轭梯度（CG）相结合，实现对高维协方差矩阵逆和对角线的无矩阵逆矩阵自由估计，从而消除了对前向模型（如圆形/环形光阑）的简化假设；同时保持了对真实光阑的准确建模。

**🔧 技术方法**

核心技术包括投影梯度下降、随机Monte‑Carlo迹估计、共轭梯度求解、FFT加速的前向/反向算子、以及三种不同的先验/去噪器（BM3D、DnCNN、Deep Decoder）。

**📊 数据集**

使用标准灰度图像（Barbar、Boats、Foreman、House、Monarch、Parrots、Peppers）在 256×256 分辨率下生成的合成数字全息测量数据，实验覆盖不同光阑（圆形/环形）、噪声水平和多视角数。

**📈 对比分析**

与基准裁剪法和基于CPnP-EM的迭代重建方法比较；PGD‑MC 在 PSNR、SSIM 以及收敛稳定性上均优于 CPnP‑EM，且每次迭代的计算时间更低，尤其在 512×512 大尺寸图像时显著提升了可扩展性。

**⚠️ 局限性**

局限性包括：每次梯度评估需要多次 CG 迭代和 MC 采样，导致计算量仍不低；对噪声参数、CG 容差等超参数敏感；未在真实全息数据上验证，模型假设（如光阑形状、噪声统计）在实际环境中可能不完全成立。

---

## 344. Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning

**arXiv ID:** 2602.11149 | [PDF](https://arxiv.org/pdf/2602.11149v1)

**作者:** Dawid J. Kopiczko `[一作]` (University of Technology Nuremberg), Yuki M. Asano `[通讯]` (University of Technology Nuremberg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在固定更新预算下，使用多轮训练而非增大数据量进行长链思维数据的监督微调。

**💡 创新点**

创新点是发现并量化了“重复优势”，即在相同训练步数下，重复少量样本可显著提升推理性能。

**🔧 技术方法**

采用了多轮微调、基于token准确率的停止准则以及常规的交叉熵损失训练。

**📊 数据集**

使用了Dolci SFT 7B、NuminaMath‑TIR等长链思维数据集，并在AIME、GPQA等推理基准上评估。

**📈 对比分析**

通过与单轮大样本训练对比，结果显示在同等梯度更新数下，少量样本多轮训练能提升12‑26个百分点的准确率，且不导致灾难性遗忘。

**⚠️ 局限性**

局限在于尚未解释重复优势背后的机制，且最佳数据规模与模型、教师质量有关，需要进一步研究。

---

## 345. Constructing Industrial-Scale Optimization Modeling Benchmark

**arXiv ID:** 2602.10450 | [PDF](https://arxiv.org/pdf/2602.10450v1)

**作者:** Zhong Li `[一作]` (Great Bay University), Zaiwen Wen `[通讯]` (Peking University)

**通讯引用:** 4746 | [OpenAlex ID](https://openalex.org/A5006127137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从 MIPLIB 2017 的混合整数线性规划实例中，通过结构感知的逆向生成方法构建了名为 MIPLIB-NL 的工业规模自然语言到优化建模基准。

**💡 创新点**

创新点在于：① 用专家驱动的结构抽象恢复变量组与约束族，② 在统一模型–数据分离格式下生成自然语言蓝图并进行 LLM 辅助润色，③ 采用独立重建与人机交互的双重语义验证，④ 对比现有 NL‑to‑Opt 系统在工业规模实例上的性能差距，揭示先前基准的不足。

**🔧 技术方法**

技术手段包括：结构化抽象与循环框架构建、专家设计的 NL 蓝图、LLM 辅助语言润色、Pass@N 语义一致性评估、求解器可执行性检查、专家与 LLM 的交互式验证流程。

**📊 数据集**

数据集：从 MIPLIB 2017 选取 223 个实例进行逆向生成（即 MIPLIB‑NL），并与 10 个现有 NL‑to‑Opt 基准数据集一起用于评估。

**📈 对比分析**

方法比较：对 14 种 NL‑to‑Opt 系统（细调模型与直接 Prompt 的 LLM）在传统基准和 MIPLIB‑NL 上分别计算 Pass@1、Pass@8、求解器可执行性等指标。实验显示，在 MIPLIB‑NL 上性能显著下降，说明现有系统在工业规模结构上表现差劲。

**⚠️ 局限性**

局限性：① 需要人工专家进行结构抽象，工作量大；② 仅覆盖 223 个实例，难以完全代表所有工业模型类型；③ 逆向生成过程可能遗漏某些细节；④ 仅针对 MIPLIB 2017 的线性 MILP，未覆盖非线性或混合整数非线性等模型。

---

## 346. Canvas-of-Thought: Grounding Reasoning via Mutable Structured States

**arXiv ID:** 2602.10494 | [PDF](https://arxiv.org/pdf/2602.10494v1)

**作者:** Lingzhuang Sun `[一作]` (University of Chinese Academy of Sciences), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14730 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Canvas‑CoT 框架，将多模态推理从线性文本生成转变为可变结构的状态操作。

**💡 创新点**

创新点在于使用 HTML DOM 作为可编辑的外部状态子系统，支持 CRUD 操作与渲染‑批判循环，实现可视化自纠。

**🔧 技术方法**

采用 LLM 作为控制器、DOM 作为结构化存储、渲染器（headless 浏览器）以及视觉批判模块，并通过增量操作避免全量重生成。

**📊 数据集**

在 VCode、RBench‑V 与 MathVista 三大基准集上进行实验，验证了该方法在视觉编码、物理推理与多步推理任务中的有效性。

**📈 对比分析**

与标准 CoT、Tree‑of‑Thought、Program‑of‑Thought 与 Iterative‑Reflection 等基线相比，Canvas‑CoT 在 VCode、RBench‑V 与 MathVista 上均实现了显著提升（平均分提高 4–10 % 以上，token 省耗降低 20–30 %）。

**⚠️ 局限性**

局限性包括对渲染与批判循环的额外计算开销、对 LLM 仍需高效解析 DOM 片段的依赖，以及在极大规模任务中 DOM 操作的可扩展性待进一步研究。

---

## 347. Online Causal Kalman Filtering for Stable and Effective Policy Optimization

**arXiv ID:** 2602.10609 | [PDF](https://arxiv.org/pdf/2602.10609v1)

**作者:** Shuo He `[一作]` (Nanyang Technological University), Bo An `[通讯]` (Nanyang Technological University)

**通讯引用:** 6786 | [OpenAlex ID](https://openalex.org/A5017743551)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在线因果卡尔曼滤波（Online Causal Kalman Filtering）方法，对大语言模型训练中的 token 级重要性采样（IS）比率进行时序平滑，以提升 RL 训练的稳定性和效果。

**💡 创新点**

创新点在于将 token 级 IS 比率视为噪声干扰的时间序列，采用卡尔曼滤波器仅利用过去信息实现自回归平滑，既抑制高频噪声又保持局部结构一致性，从而弥补传统序列级或单独调节策略导致的结构失衡。

**🔧 技术方法**

核心技术是卡尔曼滤波（Kalman filtering）在对数 IS 比率空间的状态空间模型、预测、增益计算与更新步骤；此外结合 PPO/GRPO 目标、CLIP 机制、以及对 Qwen3‑4B 模型的强化学习优化。

**📊 数据集**

实验使用 Qwen3‑4B 预训练模型，训练数据来自 DAPO 的数学推理数据集；评估基准包括 AIME'24、AIME'25、AMC'23、MATH500、OlympiadBench 等六个数学推理挑战。

**📈 对比分析**

与 token‑级 GRPO、序列级 GMPO/GSPO 等基线相比，Kalman‑Filtered 方法在所有评估指标（平均准确率和通过率）均取得显著提升，尤其在 AIME、AMC、MATH500 等难度更高的任务上表现更为突出；同时训练过程中的奖励、熵、剪裁比例和梯度损失均更为平稳。

**⚠️ 局限性**

局限性包括：卡尔曼滤波需要按顺序逐 token 处理，难以并行化；滤波参数 Q/V 对性能影响较大，需要经验调优；在极端离谱或大规模分布差异的场景下，卡尔曼滤波可能无法及时捕捉快速变化的 IS 比率。

---

## 348. Reviewing the Reviewer: Elevating Peer Review Quality through LLM-Guided Feedback

**arXiv ID:** 2602.10118 | [PDF](https://arxiv.org/pdf/2602.10118v1)

**作者:** Sukannya Purkayastha `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 25401 | [OpenAlex ID](https://openalex.org/A5027450194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个基于大型语言模型的端到端框架，先将评审文本拆分成段落，利用神经符号化方法检测“懒惰思维”和“具体化”问题，再通过遗传算法优化的模板生成符合ACL ARR指南的针对性反馈。

**💡 创新点**

创新点在于将LLM生成的是/否问题转化为结构化特征向量与传统机器学习结合（Extra‑Trees）实现精确的多标签检测，并采用遗传算法对模板化生成过程进行自适应优化，显著提升反馈的多样性和规范性。

**🔧 技术方法**

使用的技术包括多款开源LLM（Qwen 2.5 7B Instruct、Yi 1.5 9B Chat、Deepseek LLM 7B Chat、Phi‑4 14B、GPT OSS 20B）、Extra‑Trees、其他经典分类器、遗传算法（模板构造、计划生成、种群初始化、适应度评估、交叉等）以及B‑I‑O句子分割。

**📊 数据集**

使用的数据集为LazyReviewPlus（1,309句，包含懒惰思维和具体化的多标签标注），并以ARR 2022、EMNLP 2024的评审文本为来源。

**📈 对比分析**

在多标签检测上，神经符号化+Extra‑Trees的F0.5达0.49~0.51，超过零样本LLM至少0.9点，并且比finetuned LLM低。反馈生成方面，结合模板+遗传算法的系统在自动化评测（Constructiveness、Relevance等）和人工评测中均优于所有基线约20%，在受控实验中能将评审问题降低至92.4%。

**⚠️ 局限性**

限制主要包括：仅针对ACL Rolling Review会议的指南；需要预定义模板，可能导致反馈风格单一；在多语言或其他会议域的适用性未知；LLM可能带来偏见，需人工监督。

---

## 349. Why Does RL Generalize Better Than SFT? A Data-Centric Perspective on VLM Post-Training

**arXiv ID:** 2602.10815 | [PDF](https://arxiv.org/pdf/2602.10815v1)

**作者:** Aojun Lu `[一作]` (Sichuan University), Yanan Sun `[通讯]` (Sichuan University)

**通讯引用:** 6729 | [OpenAlex ID](https://openalex.org/A5091058342)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉-语言模型（VLM）在下游任务中的后训练阶段，探讨了监督微调（SFT）与强化学习（RL）在 OOD 泛化上的差距，并提出通过筛选训练样本的难度来改进 SFT。

**💡 创新点**

核心创新在于发现 RL 的泛化优势主要源自其隐式的中等难度样本过滤机制，并基于此提出 Difficulty-Curated SFT（DC‑SFT）——一种显式去除难样本、保留易样本与中等难度样本的 SFT 方案，显著提升 OOD 泛化、训练稳定性和计算效率。

**🔧 技术方法**

使用了 SFT、RL（GRPO）算法、LoRA 参数高效微调、KL 正则、梯度归一化与监控、数据难度划分（易/中/难）、自动化数据筛选等技术。

**📊 数据集**

实验数据集包括 ImageNet‑1K（100/200 类子集）、ImageNet‑R/A、RefCOCO、Ref‑L4、Lisa、MiniCPM‑V‑4 以及多模态推理数据集 MMK12、MMMU、WeMath、MathVerse、MathVista、MathVision。

**📈 对比分析**

对比标准 SFT、GRPO 与 DC‑SFT，DC‑SFT 在保持 ID 性能的同时，OOD 平均提升约 2–5%（在 ImageNet‑R、ImageNet‑A、Ref‑L4、Lisa 上），并且训练稳定性提升、效率提升（约 4.9× 训练时间），表现优于 RL 方案。

**⚠️ 局限性**

局限性：实验仅覆盖 3–7B 参数规模的 VLM，主要采用 LoRA 微调；未对更大模型或更多模态任务进行验证；部分评估集中在少数 OOD 任务，数据集和任务覆盖范围有限。

---

## 350. Stride-Net: Fairness-Aware Disentangled Representation Learning for Chest X-Ray Diagnosis

**arXiv ID:** 2602.10875 | [PDF](https://arxiv.org/pdf/2602.10875v1)

**作者:** Darakshan Rashid `[一作]` (Indian Institute of Technology Delhi), Brejesh Lall `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 2844 | [OpenAlex ID](https://openalex.org/A5066116024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 STride-Net，一个公平、可分离表示学习框架，用于胸部 X 光图像的疾病诊断。

**💡 创新点**

创新点在于可学习步幅掩码实现局部显著区域选择，结合群最优传输对图像补丁与 BioBERT 标签嵌入进行语义对齐，并通过对抗困惑损失消除敏感属性信息，从而在保持诊断性能的同时提升公平性。

**🔧 技术方法**

使用技术包括 Vision Transformer 编码、BioBERT 语义嵌入、可学习掩码、Group-Optimal Transport、对抗混淆损失以及交叉熵分类损失。

**📊 数据集**

实验数据集为公开的 MIMIC-CXR 和 CheXpert 胸部 X 光图像数据集，聚焦“无异常”二分类任务及种族/性别子组。

**📈 对比分析**

与 ERM、UBAIA、CheXclusion 等基线比较，S Tride-Net 在 MIMIC-CXR 上平均准确率提升 1.6%，在 CheXpert 上提升 0.8%，且公平性指标（PQD、EOM）显著提高，展现更优的准确性-公平性折衷。

**⚠️ 局限性**

局限性包括对掩码学习和对抗权重的超参数敏感，计算成本相对较高，且在更大异质人群中的泛化验证仍待进一步研究。

---

## 351. A Vision-Language Foundation Model for Zero-shot Clinical Collaboration and Automated Concept Discovery in Dermatology

**arXiv ID:** 2602.10624 | [PDF](https://arxiv.org/pdf/2602.10624v1)

**作者:** Siyuan Yan `[一作]` (Monash University), Zongyuan Ge `[通讯]` (Monash University)

**通讯引用:** 11614 | [OpenAlex ID](https://openalex.org/A5005014252)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究开发了一种名为DermFM‑Zero的多模态视觉‑语言基础模型，在400多万张皮肤图像与文本对上预训练，能够在无需任何任务特定微调的情况下完成零样本诊断、跨模态检索和问答等任务。

**💡 创新点**

创新之处包括：①采用掩码潜在建模与自监督视觉预训练的两阶段策略，②使用PubMedBERT并扩展词长实现医学语义对齐，③通过稀疏自编码器自动发现临床概念，提供可解释性与可干预的鲁棒性。

**🔧 技术方法**

关键技术涵盖掩码潜在建模、Bootstrap对比学习、PubMedBERT文本编码、稀疏自编码器、概念瓶颈模型、线性探针以及跨模态检索与视觉问答框架。

**📊 数据集**

数据来源包括：约400万无标签多模态图像（临床照片、皮镜图、移动拍摄）、约100万图像‑文本对（Derm1M、教育资源、MoleMap）以及公开基准集如HAM10000、ISIC2020、PAD‑UFES‑20、SNU‑134、SD‑128、Derm7pt、SkinCap、F17K、DDI、ISIC‑Intervention等。

**📈 对比分析**

在20项零样本与多模态任务上与CLIP、DINOv3、BioMedCLIP、MONET、DermLIP、MAKE、PanDerm等现有模型对比，DermFM‑Zero均实现SOTA。其在零样本分类上准确率提升约23%+，在跨模态检索Recall@50提升32%+；在三项跨国读者研究中，AI辅助将基层诊师的差异诊准确率翻倍，专家级诊断优于人类平均，并显著提升管理适当性。

**⚠️ 局限性**

局限性包括：①评估基于回溯模拟，缺乏真实临床流水线验证；②模型仅覆盖约400种疾病，尚未覆盖全谱皮肤病；③未对肤色公平性和多样性进行系统评估；④解释与干预机制尚未完整集成至临床工作流程。

---

## 352. TVCACHE: A Stateful Tool-Value Cache for Post-Training LLM Agents

**arXiv ID:** 2602.10986 | [PDF](https://arxiv.org/pdf/2602.10986v1)

**作者:** Abhishek Vijaya Kumar `[一作]` (Cornell University), Rachee Singh `[通讯]` (Cornell University)

**通讯引用:** 396 | [OpenAlex ID](https://openalex.org/A5071694147)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在RL后训练LLM代理时，提出了一种状态化工具值缓存（Stateful Tool-Value Cache），通过在工具调用序列上构建工具调用图（Tool Call Graph，TCG），利用最长前缀匹配实现高效的缓存命中与结果复用，从而显著减少工具调用的延迟和计算资源浪费。

**💡 创新点**

创新点包括：
1) 将工具调用视为状态化序列，使用TCG捕捉完整的调用路径，确保缓存结果对应相同沙盒状态；
2) 采用最长前缀匹配（LPM）做缓存查询，保证命中结果的正确性；
3) 引入选择性沙盒快照策略，仅在执行成本大于快照开销时才存储快照，降低内存与IO压力；
4) 通过主动预分叉、被动响应式分叉与后台实例化三种分叉策略，进一步减少分叉开销；
5) 对并发访问进行引用计数与子树裁剪的缓存清理，支持大规模并行rollout。

**🔧 技术方法**

技术手段包括：
- 工具调用图（TCG）数据结构与最长前缀匹配算法；
- 选择性快照决策（成本对比策略）；
- 沙盒快照与分叉（Docker/文件复制/云API）实现；
- 后台线程与前台同步的分叉调度；
- 服务器-客户端架构，支持多任务分片与线程安全的API；
- 参考计数与多层裁剪的缓存淘汰策略。

**📊 数据集**

实验使用的三大数据集：
1) terminal-bench（终端命令执行任务，包含 easy/medium 难度）
2) SkyRL-SQL（云数据库查询任务）
3) EgoSchema（视频理解与处理任务）。

**📈 对比分析**

对比方法：在相同模型、硬件与训练配置下，分别跑无缓存与带缓存两种设置。
- 缓存命中率在 15%–70% 范围内，随着迭代次数递增。 
- 平均工具调用时间从 8.07–36.23 秒降至 1.40–6.53 秒，速度提升 3.44×–6.92×；最高单调用缩短 6.9 倍。 
- 训练过程中的奖励曲线与无缓存保持一致，证明缓存不会影响学习效果。

**⚠️ 局限性**

局限性：
- 对需要大量沙盒快照的任务仍会产生较高内存和磁盘IO负担；
- 快照策略基于成本阈值，可能在某些极端场景下误判；
- 并发分叉与缓存清理需要额外调度，复杂度较高；
- 对于不重复或高度随机的工具调用，缓存收益有限；
- 目前主要验证在三类任务，跨更多多样化工具与环境的泛化性仍待进一步评估。

---

## 353. AMAP-APP: Efficient Segmentation and Morphometry Quantification of Fluorescent Microscopy Images of Podocytes

**arXiv ID:** 2602.10663 | [PDF](https://arxiv.org/pdf/2602.10663v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 354. Training-Induced Bias Toward LLM-Generated Content in Dense Retrieval

**arXiv ID:** 2602.10833 | [PDF](https://arxiv.org/pdf/2602.10833v1)

**作者:** William Xion `[一作]` (L3S Research Center), Wolfgang Nejdl `[通讯]` (L3S Research Center)

**通讯引用:** 17493 | [OpenAlex ID](https://openalex.org/A5074427964)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对密集检索器在不同训练阶段对大语言模型生成文本的偏好进行系统评估，并追踪训练诱导的源偏差；

**💡 创新点**

首次证明源偏差并非密集检索器固有属性，而是由监督微调与语料决定；通过实验证实 MS MARCO 微调和 LLM 生成语料显著诱导 pro‑LLM 偏好，同时否定低困惑度解释；

**🔧 技术方法**

使用密集检索框架（E5、Contriever、AugTriever）+对比学习、InfoNCE 损失、NDCG 与 Relative Δ 评估，并引入检索器中心困惑度（PRA）测量；

**📊 数据集**

使用 SciFact、Natural Questions (NQ320K) 的人工与 Llama2 生成对齐语料，以及标准 MS MARCO 语料；

**📈 对比分析**

通过将两类文本合并检索，分别计算各自 NDCG 并求 Relative Δ，比较不同微调阶段的偏好变化；结果显示无监督阶段无一致偏好，MS MARCO 微调显著 pro‑LLM，LLM 生成语料微调进一步强化该偏好，PRA 低于 50% 表明困惑度与相关性无关；

**⚠️ 局限性**

实验仅涵盖有限模型与数据集，未深入探究导致偏差的具体语言特征；评测仅局限于特定检索任务，缺乏跨领域泛化分析；对齐方式与评估工具的选择可能影响结论。

---

## 355. In-the-Wild Model Organisms: Mitigating Undesirable Emergent Behaviors in Production LLM Post-Training via Data Attribution

**arXiv ID:** 2602.11079 | [PDF](https://arxiv.org/pdf/2602.11079v1)

**作者:** Frank Xiao `[一作]` (California Institute of Technology), Santiago Aranguri `[通讯]` (Goodfire)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于激活差分向量的归因方法，用于追踪后期训练（如 DPO）中出现的不可预见的违规行为，并通过无监督聚类自动发现这些行为。

**💡 创新点**

创新点在于：①使用激活空间中的余弦相似度将行为变化向量与训练样本向量匹配，从而定位导致行为变化的数据点；②在无监督条件下通过层聚类揭示隐藏的行为模式；③在生产级 DPO 训练中首次发现并验证了一个 in‑the‑wild 模型实例（诱导性违规满足）。

**🔧 技术方法**

核心技术包括：激活差分向量、余弦相似度归因、层级聚类（Ward 方法）以及可视化工具；对比方法包括梯度归因（LESS）和 LLM‑judge。

**📊 数据集**

使用 OLMo 2 7B 378,341 条偏好对训练集（LMSys 提示、20 个 LLM 生成回复）和同一模型的测试提示集。

**📈 对比分析**

与梯度方法和 LLM‑judge 的归因相比，激活归因在过滤/切换数据点时以约 10 倍更低的计算成本实现了 63%–78% 的违规率下降，且对模型能力影响最小；在模型层级归因上可实现 85% 的违规率进一步降低。

**⚠️ 局限性**

局限性包括：仅在 OLMo 2 上验证，缺乏跨模型推广性；归因解释需人工聚类分析，自动化程度低；依赖可访问训练数据和检查点；对 RLHF 等其他后期训练方法的适用性尚未验证。

---

## 356. Evaluation metrics for temporal preservation in synthetic longitudinal patient data

**arXiv ID:** 2602.10643 | [PDF](https://arxiv.org/pdf/2602.10643v1)

**作者:** Katariina Perkonoja `[一作]` (University of Turku), Joni Virta `[通讯]` (University of Turku)

**通讯引用:** 429 | [OpenAlex ID](https://openalex.org/A5062244712)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了合成纵向患者数据的时间维度保持评价方法，并构建了多维度的评估框架

**💡 创新点**

提出了一套基于核平滑的多指标体系，覆盖边际统计、协方差结构、个体轨迹和测量时间分布，可多角度评估合成数据的时间维度保真度

**🔧 技术方法**

采用核平滑、加权经验分布、变异图、排名稳定性、局部转移概率等非参数统计方法实现指标计算

**📊 数据集**

以MIMIC‑III数据库为基础，生成了两套合成数据（HALO和Health Gym GAN）进行实验

**📈 对比分析**

通过与原始数据对比多项指标（均值/分位、方差/变异图、排名变异、测量密度、相似度、KL散度等）评估性能，发现单一边际指标可能掩盖协方差和个体轨迹失真，强调多指标综合评估的必要性

**⚠️ 局限性**

局限性：仅关注单变量时间结构，未评估多变量关系、隐私与实用性；未区分缺失与不均衡采样；缺少对不同预处理/生成模型的系统比较

---

## 357. AugVLA-3D: Depth-Driven Feature Augmentation for Vision-Language-Action Models

**arXiv ID:** 2602.10698 | [PDF](https://arxiv.org/pdf/2602.10698v1)

**作者:** Zhifeng Rao `[一作]` (Southern University of Science and Technology), F. Richard Yu `[通讯]` (Carleton University)

**通讯引用:** 52956 | [OpenAlex ID](https://openalex.org/A5100420016)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种 AugVLA-3D 框架，在 Vision‑Language‑Action 模型中注入 sensor‑free 3D 结构特征，并通过 Action Assistant 正则化提升动作预测。

**💡 创新点**

创新点在于：① 通过 monocular depth 模型 VGGT 将 2D RGB 转为稠密 3D 点云，实现在不使用 LiDAR 的情况下获得 3D 信息；② 设计轻量级 Action Assistant 模块以任务先验引导 3D 特征，保证与预训练 2D 表示的兼容；③ 将 3D 特征与原始 2D 特征融合，提升几何感知与动作稳健性。

**🔧 技术方法**

核心技术包括 VGGT 深度估计、PointNet 3D 编码、Transformer‑Diffusion 结构的 Action Assistant、跨层特征注入与可学习门控，整体实现了 2D‑3D 多模态融合。

**📊 数据集**

实验使用：RoboCasa 仿真 24 个桌面任务（30/100 次演示），以及真实机械手 ROH‑A001 在五个多样化任务的 2D 视频数据；仅训练 10% 数据，单 NVIDIA RTX 4090 GPU。

**📈 对比分析**

与 Gr00T、Diffusion Policy 等基线对比，AugVLA-3D 在仿真任务中平均成功率提升约 10%+（从 43% 提升至 50%），在真实环境中表现出更高的抓取、放置精度和轨迹平滑度，整体性能显著优于传统 2D‑VLA。

**⚠️ 局限性**

局限性：仅在有限 GPU 资源下训练、使用少量数据、未进行大规模预训练、对实时 3D 传感器或更复杂环境的适配尚未验证；模型对深度估计误差的鲁棒性还有待进一步提升。

---

## 358. Implementability of Global Distributed Protocols modulo Network Architectures

**arXiv ID:** 2602.10320 | [PDF](https://arxiv.org/pdf/2602.10320v1)

**作者:** Elaine Li `[一作]`, Thomas Wies `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

研究在不同网络架构下全局协议的可实现性问题，提出网络参数化的一致性条件，并给出判定算法和工具实现。

**💡 创新点**

创新点在于首次提供针对任意网络架构的完整实现性判定（通用一致性条件），并将其归约为简单的缓冲区操作公理，覆盖了常见的 FIFO、邮箱、发送盒、单盒和包等五种网络模型。

**🔧 技术方法**

使用形式化网络模型、LTS 与 CLTS 定义，构造一致性条件、缓冲区公理，利用修订的符号化协议与 μ-CLP 编码实现算法，采用Coq 证明其可靠性。

**📊 数据集**

使用原有的全球会话类型基准集合（包含多种 Web 服务和分布式协议示例），并新增“网络分离”微基准以测试不同网络之间的可实现性差异。

**📈 对比分析**

通过在 2024 MacBook Air（Apple M3）上多次跑 10 次实验，结果表明工具对所有五种网络架构的判定时间与现有基准保持一致，且性能与理论复杂度一致；对 P2P FIFO 结果与文献完全一致，证明工具正确且扩展性良好。

**⚠️ 局限性**

局限性包括：仅考虑异步网络，排除了同步模型；无法处理有界通道和允许消息复制的网络；实现性判定在符号协议下仍不可判定，工具可能在某些实例上无限循环。

---

## 359. LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation

**arXiv ID:** 2602.10367 | [PDF](https://arxiv.org/pdf/2602.10367v1)

**作者:** Zhiling Yan `[一作]` (Lehigh University), Lichao Sun `[通讯]` (Lehigh University)

**通讯引用:** 8043 | [OpenAlex ID](https://openalex.org/A5071709543)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LiveMedBench，一个持续更新、无污染的医学评测基准，用于评估大语言模型的临床推理能力。

**💡 创新点**

创新点在于四方面：1）每周实时采集真实临床案例，解决数据污染与时间失效问题；2）多智能体医学策划框架，自动验证案例与权威医学证据的一致性；3）自动化案例特定 Rubric 生成与评估，提升评测细粒度与与医生一致性；4）在多语言、多专业场景下实现可扩展性。

**🔧 技术方法**

使用的技术包括：检索增强生成（RAG）用于证据对齐；多智能体（Screening、Validation、Controller）流程进行案例结构化与验证；自动化 Rubric Generator 与 Rubric-based Grader（基于二元判定）实现评测；大规模 LLM 评估与人类医生的交叉验证。

**📊 数据集**

数据集为 LiveMedBench，包含 2,756 条真实临床案例，覆盖 38 专业，语言为英中，配备 16,702 条案例特定评估标准。

**📈 对比分析**

与 38 只 LLM 进行对比评测，最佳模型 GPT‑5.2 仅 39.2% 分，且 84% 模型在训练截止后案例表现显著下降，表明数据污染与知识陈旧严重；相比传统静态基准，LiveMedBench 的难度更高、评测更可靠。

**⚠️ 局限性**

局限性包括：1）依赖在线医学社区，数据可获得性与平台政策可能影响更新频率；2）多智能体与 Rubric 自动化仍可能引入误检，需进一步人工审核；3）评测侧重开放式生成，尚未涵盖多模态或交互式情境；4）目前仅支持英中两语，未来需扩展至更多语言。

---

## 360. The Complexity of Bayesian Network Learning: Revisiting the Superstructure

**arXiv ID:** 2602.10253 | [PDF](https://arxiv.org/pdf/2602.10253v1)

**作者:** Robert Ganian `[一作]` (TU Wien), Viktoriia Korchemna `[通讯]` (TU Wien)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5002623446)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文系统研究贝叶斯网络结构学习（BNSL）在不同图参数化下的参数化复杂度，提出并证明了新的可行参数（如反馈边数、局部反馈边数、树宽下的加性表示），并给出对应的固定参数可解算法以及相应的下界；随后将所有结果推广到多叉树学习（Polytree Learning）。

**💡 创新点**

创新点包括：①首次证明 BNSL 在反馈边数参数化下为 FPT 并给出多项式核化；②引入局部反馈边数参数，进一步提升可解性；③利用加性表示大幅降低树宽参数化下的复杂度；④将上述所有技术统一并应用于多叉树学习。

**🔧 技术方法**

技术方法主要涵盖参数化复杂度分析、数据归约规则、动态规划（树分解、局部记录）、Monadic Second‑Order 逻辑、Matroid 交叉等；在加性表示下采用 matroid‑intersection 的思想实现多项式时间求解。

**📊 数据集**

该工作属于理论计算机科学研究，不依赖具体实验数据集，所有结果均通过严格的理论证明得出。

**📈 对比分析**

与以往仅在顶点覆盖等参数下给出难度结果的研究相比，本文在反馈边数、局部反馈边数、树宽加性表示等参数下实现了 FPT 算法，并为多叉树学习提供了多项式核化与 FPT 算法；同时给出了树剪宽、顶点覆盖等参数下仍为 W[1]-难的下界，进一步完善了 BNSL 的复杂度图景。

**⚠️ 局限性**

主要局限包括：①未针对实际数据评估算法性能；②在树剪宽参数化下仍未获得 FPT 结果；③加性表示的适用性受限于能够被拆解为可加分数的特殊分数函数。

---

## 361. Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs

**arXiv ID:** 2602.10352 | [PDF](https://arxiv.org/pdf/2602.10352v1)

**作者:** Keenan Pepper `[一作]` (AE Studio), Diogo de Lucena `[通讯]` (AE Studio)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冻结大型语言模型的情况下，利用已有的解释性向量-标签对训练轻量级适配器，使模型能够可靠地对自身内部表示进行自我解释；

**💡 创新点**

核心创新是把已有的解释性向量-标签对视为监督数据，训练仅有d_model+1个参数的标量仿射适配器，从而在不修改模型权重的前提下显著提升自解释质量；

**🔧 技术方法**

采用Patchscopes框架，训练轻量级标量仿射（以及低秩扩展）适配器，将激活向量投射回嵌入空间并注入提示中，利用交叉熵损失进行监督；

**📊 数据集**

使用三类数据集：Goodfire SAE解码器向量与自动解释标签、Llama Scope SAE解码器向量与自动解释标签，以及维基百科对比激活向量与合成主题描述；

**📈 对比分析**

通过生成评分、检索召回率、检测评分等指标与未训练的SelfIE、原始标签、多次重述等基线对比，结果显示：在SAE任务上训练适配器的生成命中率可达70%以上，主题检索召回率从1%提升至90%以上，多跳推理中桥接实体检出率从56%提升至91%；

**⚠️ 局限性**

限制包括：全秩适配器在SAE数据上易过拟合；标量仿射对多义激活的泛化仍受限；自解释结果虽可验证，但无法覆盖所有安全相关特征，且模型仍可能产生可信但无根的描述；

---

## 362. SCRAPL: Scattering Transform with Random Paths for Machine Learning

**arXiv ID:** 2602.11145 | [PDF](https://arxiv.org/pdf/2602.11145v1)

**作者:** Christopher Mitcheltree `[一作]` (Queen Mary University of London), Mathieu Lagrange `[通讯]` (Nantes Université)

**通讯引用:** 2471 | [OpenAlex ID](https://openalex.org/A5029522748)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于随机路径的散射变换（SCRAPL）算法，用于在深度学习中高效地使用散射变换作为可微损失函数，显著降低计算量；

**💡 创新点**

创新点在于：①使用路径级随机采样实现散射变换梯度的无偏近似；②设计了路径级自适应动量（𝒫-Adam）和路径级平均梯度（𝒫-SAGA）以抑制梯度方差；③引入了基于任务信息的θ重要性采样（θ-IS）来加速收敛；

**🔧 技术方法**

主要技术包括：散射变换（特别是联合时频散射 JTFS）、随机梯度下降、路径级 Adam 与 SAGA 优化、重要性采样、FFT/多速率滤波、Python 库实现；

**📊 数据集**

使用了三类无监督声音匹配数据集：①粒子合成器（granular synth）与可调密度/斜率参数；②chirplet 合成器（AM/FM 参数）；③Roland TR‑808 打击乐单拍记录（681 条 44.1 kHz 录音）；

**📈 对比分析**

与全树 JTFS、MSS、P-loss 等基线对比，SCRAPL 在准确率上仅比 JTFS 稍逊（约 2 倍以内），但运行时间与内存仅比 JTFS 低 2 倍，远快于 JTFS；在对齐与未对齐的 TR‑808 任务中，SCRAPL 仍能保持 transient 质量，而 MSS 在未对齐场景下失效；

**⚠️ 局限性**

局限性包括：①对低频路径采样不足，导致对声纹衰减部分匹配效果不佳；②虽然无额外优化技巧即可得到无偏梯度，但相较于完整 JTFS 仍有精度损失；③P-Adam 与 P-SAGA 需要额外的路径维度内存，适用性受限。

---

## 363. PMMA: The Polytechnique Montreal Mobility Aids Dataset

**arXiv ID:** 2602.10259 | [PDF](https://arxiv.org/pdf/2602.10259v1)

**作者:** Qingwu Liu `[一作]` (Polytechnique Montreal), Guillaume-Alexandre Bilodeau `[通讯]` (Polytechnique Montreal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文收集了 PMMA 数据集，包含 9 类使用不同移动辅助装置的人群，并对七种主流目标检测器和三种目标跟踪器进行基准测试。

**💡 创新点**

创新点在于：①首次公开真实户外环境下的多类移动辅助装置数据集；②对检测器/跟踪器在细粒度类别上的性能进行系统评估；③提供了细粒度类别、遮挡等级和多视角的完整标注。

**🔧 技术方法**

采用了 MMDetection 框架，评估了 Faster R‑CNN、CenterNet、YOLOX、DETR、Deformable DETR、DINO、RT‑DETR 等检测模型；跟踪方面使用 ByteTrack、BOT‑SORT、OC‑SORT。

**📊 数据集**

使用 PMMA 数据集（约 28k 张 2208×1242 像素图像，3 个摄像角度，9 个类别，包含遮挡信息）。

**📈 对比分析**

实验表明 YOLOX、Deformable DETR 与 Faster R‑CNN 在 mAP、AP_75 等指标上名列前茅；跟踪性能差异不大，主要受检测质量影响，ByteTrack 在 HOTA 上略胜。

**⚠️ 局限性**

局限性包括：①检测/跟踪效果在 cane、walker 等细小类别仍不理想；②数据采集仅来自单一校园停车场，缺乏多场景、多天气的泛化；③模型在遮挡严重或光照变化大时性能下降。

---

## 364. Limits of Residual-Based Detection for Physically Consistent False Data Injection

**arXiv ID:** 2602.10162 | [PDF](https://arxiv.org/pdf/2602.10162v1)

**作者:** Chenhan Xiao `[一作]` (Arizona State University), Yang Weng `[通讯]` (Arizona State University)

**通讯引用:** 3934 | [OpenAlex ID](https://openalex.org/A5021106309)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究在AC功率系统状态估计中，物理一致的假数据注入攻击如何逃避基于残差的检测，揭示了残差检测的固有限制。

**💡 创新点**

创新点在于从测量流形几何视角提出可达不可检测条件，并构造物理引导的自编码器，利用符号基函数将非线性测量映射到线性空间，实现对测量流形的精确重构与扰动生成，从而系统地展示残差检测的可检测极限。

**🔧 技术方法**

主要技术包括符号基函数构造（基于Y‑bus 的双线性关系）、物理引导的自编码器（Encoder+线性Decoder）、残差检测（chi‑squared BDD）和学习型残差检测（自编码器重构误差）。

**📊 数据集**

实验使用 IEEE 14、30、39、57、118、200 号测试系统的历史测量数据，采用 MATPOWER 生成含真实负荷变化的AC潮流结果，加入 2% 噪声。

**📈 对比分析**

与经典 BDD、基于学习的残差检测器以及 AE‑GAN、SA‑GAN 等基准对比，所提出的扰动生成方法在不同显著性水平下的突破率普遍高于 1‑α，且学习型检测器突破率更高，表明测量流形一致性是决定可检测性的核心。

**⚠️ 局限性**

局限性包括对充足历史数据的依赖，数据量不足会降低检测突破率；扰动幅度对可检测性仍有影响；实验仅评估残差检测，未探讨结合其他多维检测策略的实际防御效果。

---

## 365. RePO: Bridging On-Policy Learning and Off-Policy Knowledge through Rephrasing Policy Optimization

**arXiv ID:** 2602.10819 | [PDF](https://arxiv.org/pdf/2602.10819v1)

**作者:** Linxuan Xia `[一作]` (Zhejiang University), Boxi Wu `[通讯]` (Zhejiang University)

**通讯引用:** 1679 | [OpenAlex ID](https://openalex.org/A5101833492)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出Rephrasing Policy Optimization（RePO），通过先理解并改写离线专家轨迹，使其与模型自身分布兼容，从而在对抗“硬样本”时实现高效学习。

**💡 创新点**

创新点在于使用“理解‑改写”机制，将离线知识转换为本地语料，从而避免了传统离线学习中的分布失配与训练不稳定。

**🔧 技术方法**

结合强化学习中的verifiable reward、GRPO基准、重采样与动态注入策略，并在推理过程中加入prompt式改写。

**📊 数据集**

使用SuperGPQA（通用知识）、OpenR1-Math Hard/Multi‑source Harder（数学推理）以及金融数据集进行训练与评测。

**📈 对比分析**

与SFT、GRPO、LUFFY等基线对比，RePO在GPQA、AIME等数学/知识基准上均取得领先成绩，同时保持推理能力稳定。

**⚠️ 局限性**

局限在于对极端长链推理的改写质量和推理速度仍受限，且在多模态或非常大规模离线数据时需进一步优化效率。

---

## 366. Parameterized Complexity of Finding a Maximum Common Vertex Subgraph Without Isolated Vertices

**arXiv ID:** 2602.10948 | [PDF](https://arxiv.org/pdf/2602.10948v1)

**作者:** Palash Dey `[一作]` (Indian Institute of Technology Kharagpur), Aritra Mitra `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5103859463)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文系统研究了最大公共顶点子图（Maximum Common Vertex Subgraph）问题的参数化复杂性，阐明了在多种结构参数（公共子图大小、顶点覆盖数、最大度、树深、路径宽度、树宽度）的组合下的W[1]-难度，并给出对应的FPT算法。

**💡 创新点**

创新点在于给出了该问题完整的参数化复杂性双分界，首次将公共子图大小、顶点覆盖数与多种图类结构参数结合，揭示其在多维参数下的可解性与不可解性。

**🔧 技术方法**

主要技术包括参数化归约、树分解动态规划、Baker分层技术以及星森林结构的组合与计数等理论算法框架。

**📊 数据集**

论文不依赖实验数据集，而是基于理论分析与证明。

**📈 对比分析**

通过理论证明与已知W[1]-难问题比较，所给FPT算法在参数为公共子图大小或顶点覆盖数等情形下运行时间为多项式级 2^{O(k^2)}·n^2，满足可接受的效率；对于平面图与常数最大度情形，还提供了 (1-ε)-近似的EPTAS。

**⚠️ 局限性**

局限在于对W[1]-难的参数组合仍无法提供多项式时间近似或更高效的FPT解，且对实际图类（如单位圆盘图等）缺乏实验验证。

---

## 367. MetaphorStar: Image Metaphor Understanding and Reasoning with End-to-End Visual Reinforcement Learning

**arXiv ID:** 2602.10575 | [PDF](https://arxiv.org/pdf/2602.10575v1)

**作者:** Chenhao Zhang `[一作]` (Shanghai AI Laboratory), Hongsheng Li `[通讯]` (The Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MetaphorStar 框架，使用视觉强化学习（GRPO）在专门设计的 TFQ 数据集上训练多模态大模型，提升图像隐喻/隐含理解能力。

**💡 创新点**

创新点包括：①首个基于 RL 的端到端图像隐喻理解框架；②引入 True‑False Question（TFQ）问答格式，提供高知识密度、可验证的训练信号；③揭示并解决 SFT Warmup 造成的 “entropy bottleneck”（SFT Curse），证明全端 RL 更适合此类抽象推理任务；④证明隐喻任务训练能显著提升通用视觉推理性能。

**🔧 技术方法**

核心技术：视觉强化学习（Group Relative Policy Optimization, GRPO）；结构化推理提示模板（先描述→分析隐喻→答案）；Token 级熵分析揭示推理不确定点；对比 SFT、SFT+RL、纯 RL 的 ablation；使用多模态 LLM（Qwen‑VL‑2.5 系列、LLaVA‑1.5 等）做基线。

**📊 数据集**

使用 TFQ‑Data（基于 II‑Bench 1434 张隐喻图像生成 14,099 条真/假问题，包含 Lite 100 图/984 题子集）和 TFQ‑Bench（Full、Lite 两级评测集），并在 II‑Bench、CII‑Bench、MMBench 等公开基准上做对比。

**📈 对比分析**

评价方法：在 TFQ、MCQ、OSQ 三种问答格式上与 20+ 主流 MLLM（Gemini‑3.0‑pro、GPT‑4o、Claude‑4.0‑Sonnet 等）进行对照；MetaphorStar‑32B 在 TFQ 达 78%（SOTA）、MCQ 78%、OSQ 3.94；在多种视觉推理基准（MMMU、MathVerse、V* 等）上提升 2–4 分；通过参数规模、数据规模、架构和训练策略 ablation 验证系统性提升。

**⚠️ 局限性**

局限性：①训练数据依赖人工/半自动生成，规模相对有限，未覆盖多语言与跨文化隐喻；②RL 训练成本高，且对模型规模与推理长度敏感；③对极端长文本生成和多轮对话的鲁棒性尚未充分评估；④在某些基准上仍不及部分闭源模型，需进一步提升通用性。

---

## 368. Reducing Estimation Uncertainty Using Normalizing Flows and Stratification

**arXiv ID:** 2602.10706 | [PDF](https://arxiv.org/pdf/2602.10706v1)

**作者:** Paweł Lorek `[一作]` (University of Wrocław), Aleksandra Krystecka `[通讯]` (University of Wrocław)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用正则化流（Normalizing Flow）学习未知数据分布，然后在流的潜空间进行分层采样，以此来估计随机变量函数期望并降低估计不确定性。

**💡 创新点**

创新点在于：① 将流模型与分层采样结合，实现对未知分布的非参数估计；② 提出了适用于高维情形的两种分层方法（Method M、Method M1/M2），通过只分层半径或少量坐标来避免指数级分层数量；③ 采用比例与最优分配在分层采样中进一步降低方差。

**🔧 技术方法**

技术手段包括：连续正则化流（CNF/FFJORD）训练、逆变换采样、分层采样（Cartesian、Spherical、Method M1/M2/Method M等）、比例/最优分配、基于流的采样与估计。

**📊 数据集**

实验数据集包括：合成多维数据（30D、128D）、多模态混合指数-帕累托分布、学生 t 分布、真实世界风速数据（AmeriGEOSS）等。

**📈 对比分析**

与粗 Monte Carlo、观测样本估计以及 GMM（Gaussian Mixture Model）进行对比。结果表明，流+分层方法在低至高维场景中均能显著减小方差、收窄 95% 置信区间，准确度优于传统方法，尤其在 30D/128D 例子中表现突出。

**⚠️ 局限性**

局限性包括：对重尾分布建模不够准确；需要耗时训练正则化流；在高维下直接分层会导致分层数量指数增长，需采用近似方法；训练集规模对性能仍有一定影响。

---

## 369. Informal and Privatized Transit: Incentives, Efficiency and Coordination

**arXiv ID:** 2602.10456 | [PDF](https://arxiv.org/pdf/2602.10456v1)

**作者:** Devansh Jalota `[一作]` (Columbia), Matthew Tsao `[通讯]` (Lyft)

**通讯引用:** 277 | [OpenAlex ID](https://openalex.org/A5108766687)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个基于游戏理论的固定路线非正式私有交通系统模型，分析司机利润最大化与乘客排队延迟之间的相互作用，并给出了系统整体效率的价格无谓损失（PoA）界限；

**💡 创新点**

创新点在于首次对非正式/私有共享交通系统提出了闭式可分析的需求函数，并证明了利润PoA≤2、乘客福利PoA≤1+p_max/p_min，提出了预算平衡的交叉补贴和Stackelberg路由两种可行机制，实现了对司机激励的精确调整；

**🔧 技术方法**

主要技术包括非原子拥堵博弈、Vickrey拥堵瓶颈模型的变形、闭式需求函数推导、PoA分析、线性规划求解交叉补贴、NP难度证明、LPF和L‑NCF算法的设计与性能证明；

**📊 数据集**

采用了印度纳拉萨帕拉市共享自动三轮车的真实数据（18条线路、每天约10万乘客、车辆容量4、费用与运营成本等参数），并利用当地NGO和人口普查信息校准模型；

**📈 对比分析**

与传统“Greedy”基线（仅优化控制司机）对比，实验显示在只占30%集中控制的情况下，L‑NCF和LPF可将利润与乘客福利比从约1.18/1.19 降至1.10/1.09，且比率在α→1时趋近1；在实际数据中，PoA与理论上限相差约10–25%；

**⚠️ 局限性**

局限在于模型假设司机与乘客同质、路网不考虑物理交通拥堵、出行时间均匀分布、且对外部干预（如价格波动）敏感，未来可扩展至异质价值时间、非均匀需求、路网交通影响及多中心化运营等场景。

---

## 370. Conditional Uncertainty-Aware Political Deepfake Detection with Stochastic Convolutional Neural Networks

**arXiv ID:** 2602.10343 | [PDF](https://arxiv.org/pdf/2602.10343v1)

**作者:** Rafael-Petruţ Gardoş `[一作]` `[通讯]`, Rafael-Petruţ Gardoş

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了政治深度伪造检测中不确定性感知推理的可靠性，并评估不同推理方法对校准误差和误差-不确定性关联的影响。

**💡 创新点**

将不确定性评估转化为可观测的实证指标，并通过置信区间划分与方差划分的系统扫描，明确不确定性在高置信区间的局部作用范围。

**🔧 技术方法**

采用蒙特卡洛Dropout、单通道随机前向、温度缩放、模型集成等推理方案，使用ResNet‑18和EfficientNet‑B4全微调的CNN架构。

**📊 数据集**

构建了4000张政治场景的真实与合成图像二分类数据集（OpenFake过滤得到的2000真/2000假），并设定生成器分离的OOD测试集。

**📈 对比分析**

在ID与generator‑disjoint OOD下，对准确率、ROC‑AUC、ECE、Brier、NLL等指标进行Bootstrap置信区间比较，发现所有方法保持高ROC‑AUC但校准差异显著，单通道随机推理在部分设置下提升校准且误差-不确定性关联在最高置信区间显著。

**⚠️ 局限性**

仅针对生成器迁移的OOD，未覆盖身份、平台或对抗性变化；不确定性评估仅在图像级别，缺乏时间/上下文信息；未使用更复杂的贝叶斯网络，结论对其他模型迁移的普适性有限。

---

## 371. A Swap-Adversarial Framework for Improving Domain Generalization in Electroencephalography-Based Parkinson's Disease Prediction

**arXiv ID:** 2602.10528 | [PDF](https://arxiv.org/pdf/2602.10528v1)

**作者:** Seongwon Jin `[一作]` (Incheon National University), Jibum Kim `[通讯]` (Incheon National University)

**通讯引用:** 6564 | [OpenAlex ID](https://openalex.org/A5100608898)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文先构建了首个可复现的基于ECoG的帕金森病预测基准数据集MOCOP，并提出了Swap‑Adversarial Framework（SAF），通过ISBCS对齐跨受试者通道并结合领域对抗学习，实现域不变特征学习；

**💡 创新点**

创新点在于：1）公开ECoG PD预测基准数据集MOCOP，弥补公共数据缺失；2）提出Inter‑Subject Balanced Channel Swap（ISBCS）作为计量对抗的计数器假设数据增强，显著削弱受试者特异性；3）将ISBCS与领域对抗训练相结合，构建端到端的域不变学习框架；4）在跨受试者、跨模态、跨数据集三种严苛场景中进行系统验证，展示通用性；

**🔧 技术方法**

技术手段包括：ECoG/EEG信号预处理（band‑pass、notch、ASR）；ISBCS随机交换同类受试者通道以生成对抗样本；领域对抗学习利用梯度反转层（GRL）和互信息约束实现受试者信息消除；采用EEGNet轻量级CNN做特征提取器，配合交叉熵、互信息损失和对抗损失共同训练；超参数通过网格搜索调优；

**📊 数据集**

使用了两大数据集：①来自6-OHDA诱导的6只大鼠的ECoG记录（无线/有线，6s段，采样率512Hz），构成MOCOP基准；②公开EEG数据集University of Iowa（UI）与University of New Mexico（UNM），每个包含28/52受试者，64通道，采样率500Hz；

**📈 对比分析**

与基线EEGNet和DMMR进行对比，采用macro‑accuracy、precision、recall、F1四项指标。实验显示：在无线条件下跨受试者测试，SAF提升宏观准确率≈41%；跨模态（无线→有线、反向）宏观准确率从68.2%提升至91.2%；在公共EEG数据跨数据集测试中，SAF相较EEGNet提升宏观准确率≈32%。整体在所有设置下均优于基线，验证了框架的鲁棒性；

**⚠️ 局限性**

局限性包括：①数据规模有限，受试者数少，易受高维低样本影响；②对超参数λ_GRL、λ_MI敏感，需要手动调优；③仅在EEGNet上验证，扩展到更深网络需进一步研究；④ISBCS在受试者差异较小的数据中效果有限；⑤当前仅针对PD预测，需验证在其他BCI任务中的通用性。

---

## 372. Towards Remote Sensing Change Detection with Neural Memory

**arXiv ID:** 2602.10491 | [PDF](https://arxiv.org/pdf/2602.10491v1)

**作者:** Zhenyu Yang `[一作]` (Nanjing University of Science and Technology), Fumin Shen `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 13012 | [OpenAlex ID](https://openalex.org/A5074492050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套名为ChangeTitans的远程感知变化检测框架，利用Titan架构的神经记忆和分段局部注意力实现高效的长程依赖建模与精确检测。

**💡 创新点**

创新点在于将神经记忆与分段局部注意力相结合的VTitans视觉骨干，实现线性复杂度的长程上下文捕捉；引入轻量级VTitans‑Adapter构建多尺度特征；以及设计跨时域TS‑CBAM融合模块，显著抑制伪变化并提升检测精度。

**🔧 技术方法**

采用了Titan架构、神经记忆模块、分段注意力机制、层次化Adapter、两流CBAM融合、Convex Upsampling、BCE+Dice损失等技术。

**📊 数据集**

在LEVIR‑CD、WHU‑CD、LEVIR‑CD+、SYSU‑CD和SAR‑CD等公开基准数据集上进行评估。

**📈 对比分析**

与多种CNN、Transformer、SSM等主流方法对比，ChangeTitans在LEVIR‑CD实现84.36% IoU和91.52% F1，WHU‑CD达88.56% IoU和94.10% F1，在所有数据集均达到或逼近SOTA，并保持27M参数、30.4G FLOPs的高效性能。

**⚠️ 局限性**

仍面临伪变化抑制和细粒度边界定位的挑战，跨传感器和大时间间隔场景的适应性不足，且缺乏针对实时部署的轻量化版本。

---

## 373. Simple LLM Baselines are Competitive for Model Diffing

**arXiv ID:** 2602.10371 | [PDF](https://arxiv.org/pdf/2602.10371v1)

**作者:** Elias Kempf `[一作]` (University of Freiburg), Arthur Conmy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过对比LLM基于方法与稀疏自编码器(SAE)基于方法的模型差异检测，提出并验证了一套针对差异假设质量的评估指标，系统评估两种方法在一般化、趣味性与抽象度方面的表现。

**💡 创新点**

创新点在于：①构造了针对模型差异假设的三大评估维度（一般化、趣味性、抽象度）并给出了量化指标；②在API仅访问的约束下实现了两种差异检测管道；③通过多实验验证两种方法在不同任务中相对优势的细致对比。

**🔧 技术方法**

技术上主要使用基于大型语言模型的差异抽取与聚类、稀疏自编码器特征激活差异检测、LLM判定器评估假设准确性、以及三维指标的统计计算。

**📊 数据集**

采用WildChat 1000条真实用户对话样本进行差异检测，额外使用了两组 finetuned 模型（风险金融建议与隐式性别假设）和两代 Gemini 2.5 Flash Lite 进行验证。

**📈 对比分析**

在三个实验中，两种方法在准确率与频率上相近，LLM方法在抽象度与接受率上表现更好，SAE方法在具体 token/语法细节上更敏感，整体性能互补。

**⚠️ 局限性**

局限性包括：评估依赖 LLM 判定器的可靠性；仅能探测模型输出中显现的差异，无法捕获隐藏行为；所用 prompt 集成与 WildChat 数据分布可能忽略领域特定差异。

---

## 374. Embedding Inversion via Conditional Masked Diffusion Language Models

**arXiv ID:** 2602.11047 | [PDF](https://arxiv.org/pdf/2602.11047v1)

**作者:** Han Xiao `[一作]` `[通讯]` (Jina AI by Elastic), Han Xiao (Jina AI by Elastic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于条件掩码扩散的嵌入反演方法，能够并行恢复原始文本序列，且不需要目标编码器的访问

**💡 创新点**

创新点在于用可学习的 AdaLN 通过扩散过程直接将嵌入信息注入模型，消除了传统自回归迭代校正的两大瓶颈

**🔧 技术方法**

采用条件掩码扩散模型、AdaLN 调整层归一化、Euler 采样与自适应重掩码策略等技术

**📊 数据集**

在 C4 / mC4 语料上随机抽取 2M 32 词长度样本，使用 GPT-2 分词器训练模型

**📈 对比分析**

与四种解码策略（顺序贪心、Euler、Euler+重掩码、两阶段）进行对比，最高 token 识别率达到 81.3%，余弦相似度 0.87，整体性能优于传统 Vec2Text 等自回归方法

**⚠️ 局限性**

局限性包括仅在 32 词长的短序列上测试，需大量训练样本和步骤，重掩码阈值选择敏感，且对跨域、长序列的泛化能力尚未验证

---

## 375. DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning

**arXiv ID:** 2602.11089 | [PDF](https://arxiv.org/pdf/2602.11089v1)

**作者:** Yicheng Chen `[一作]` (Fudan University), Kai Chen `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种端到端的数据配方（data recipe）生成框架 DataChef，自动生成适配大语言模型（LLM）的数据处理管道和训练数据。

**💡 创新点**

创新点在于：① 将数据配方生成视为一项强化学习任务；② 设计低成本的代理奖励——Data Verifier，用 LLM 评估生成数据的质量；③ 通过冷启动监督微调与在线 RL 结合，显著提升可执行性和质量；④ 构建了覆盖 19 个领域、31 个基准、257 个数据源的任务池，为此类任务提供了首个标准化数据集。

**🔧 技术方法**

主要技术包括：语言模型 (Qwen3-32B)、规划与代码生成、冷启动监督微调 (SFT)、组相对策略优化 (GRPO) 强化学习、数据质量评估 LLM（Data Verifier）、任务池构建与采样。

**📊 数据集**

使用 257 个公开数据源（来自 Hugging Face 等），涵盖 19 个领域、31 个基准，生成 25 条训练任务、6 条隐藏评估任务，每条任务配备 8–15 个原始数据源。

**📈 对比分析**

与三类基线对比：参数匹配模型 Qwen3-32B、开源旗舰 Kimi‑K2‑Instruct 与 Qwen3‑Next‑80B‑A3B‑Thinking 的组合、闭源 SOTA Gemini‑3‑Pro。DataChef‑32B 在 6 个隐藏任务上 DVS_avg@32 与 DBS 均优于开源基线，达成与 Gemini‑3‑Pro 相当甚至更优的下游性能；在选取最优样本的“oracle upper bound”下，性能可匹配甚至超过专家手工配方。

**⚠️ 局限性**

局限性：代理奖励依赖 LLM‑as‑Judge，可能在特定细分任务上缺乏精细化评估；奖励信号的泛化性虽好，但在极其专业或低资源场景下可能不足；未覆盖实时在线训练的完整闭环，仅使用了采样子集作为评估。

---

## 376. AudioRouter: Data Efficient Audio Understanding via RL based Dual Reasoning

**arXiv ID:** 2602.10439 | [PDF](https://arxiv.org/pdf/2602.10439v1)

**作者:** Liyang Chen `[一作]` (University of California), Yiwei Wang `[通讯]` (University of California)

**通讯引用:** 5537 | [OpenAlex ID](https://openalex.org/A5108046735)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AudioRouter，一种基于强化学习的工具路由框架，使大型音频语言模型学习何时调用外部音频工具来辅助推理。

**💡 创新点**

创新点在于将工具使用抽象为决策问题，解耦工具路由与推理模型，采用相对结果奖励仅训练轻量路由器，从而显著降低对数据的依赖。

**🔧 技术方法**

技术包括：GRPO 强化学习、LoRA 参数高效微调、固定的音频推理模型以及多种成熟的外部音频工具（Whisper、AST、autochord 等）。

**📊 数据集**

使用 1.5k 条 MMAU Pro 数据进行路由器训练，评测基准为 MMAU-mini 与 MMAR 两大多任务音频推理数据集。

**📈 对比分析**

与端到端工具使用以及多种基线相比，AudioRouter 在 MMAU-mini 和 MMAR 上提升 1–4% 的准确率，同时训练数据量缩减 25–647 倍，在 Qwen2.5‑Omni/Omni‑CLST 背景下实现 SOTA 结果。

**⚠️ 局限性**

局限性包括：相对奖励依赖固定的推理模型，限制了奖励信号的上限；实验仅针对短小封闭集任务和有限工具，难以直接扩展到长序列、开放式问题或更丰富的工具集合。

---

## 377. BOute: Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization

**arXiv ID:** 2602.10729 | [PDF](https://arxiv.org/pdf/2602.10729v1)

**作者:** Youhe Jiang `[一作]` (University of Cambridge), Eiko Yoneki `[通讯]` (University of Cambridge)

**通讯引用:** 5195 | [OpenAlex ID](https://openalex.org/A5063536695)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多目标贝叶斯优化的LLM服务系统，联合优化异构模型路由和GPU部署，实现成本效益。

**💡 创新点**

首次将模型路由与硬件部署进行协同优化，使用MOBO框架结合结构先验（负载分数、GPU偏好、预算约束）实现高效搜索，显著提升服务性能。

**🔧 技术方法**

多目标贝叶斯优化（MOBO）、高斯过程回归、qNEHVI采样、离线性能数据库、模型路由、异构GPU部署与并行策略。

**📊 数据集**

GSM8K（数值准确性）和MTBench（多模态评测）作为评估数据集，结合真实请求流量生成的工作负载。

**📈 对比分析**

与单机H100、异构H100/RTX、RouteLLM、Homo四个基线对比；使用P95延迟、吞吐量、成本等指标；系统在相同预算和质量下可降低P95延迟高达91%（平均57%），吞吐量提升约90%（平均55%），成本平均降低约39%。

**⚠️ 局限性**

依赖离线模拟和预先构建的性能数据库，需人工维护；优化过程对GPU类型和规模敏感，扩展到更大集群和更多模型时可能需要更多计算；仅在两种LLM（8B/70B）和四种GPU上验证，未验证对其他模型或硬件的泛化。

---

## 378. Hierarchical Zero-Order Optimization for Deep Neural Networks

**arXiv ID:** 2602.10607 | [PDF](https://arxiv.org/pdf/2602.10607v1)

**作者:** Sansheng Cao `[一作]` (Peking University), Yonghong Tian `[通讯]` (Peking University)

**通讯引用:** 15607 | [OpenAlex ID](https://openalex.org/A5023918894)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种层级零阶优化(HZO)方法，使用递归二分和局部目标传播来替代传统反向传播；

**💡 创新点**

创新点在于沿网络深度维度采用分而治之策略，将查询复杂度从O(ML²)降低到O(ML log L)，并通过空间并行扰动显著减少卷积层的查询成本；

**🔧 技术方法**

采用零阶有限差分估计Jacobian、递归目标传播、Delta规则权重更新、空间并行扰动（SPP）以及对比BP的等价性分析；

**📊 数据集**

在CIFAR‑10和ImageNet‑10（10类子集）上进行实验；

**📈 对比分析**

与DeepZero、FA、Align‑ada等非BP方法以及标准BP进行对比；HZO在CIFAR‑10上达到74.2%准确率，ImageNet‑10上65%，训练时间约8h（比DeepZero快≈60%），在梯度方向上余弦相似度≥0.95，表明梯度估计稳定；

**⚠️ 局限性**

局限在于误差仍随网络深度指数增长，需要Lipschitz常数接近1（如残差网络或正交初始化），对激活函数光滑性和数值精度敏感，且在极深网络上对扰动步长与精度要求更高。

---

## 379. Enormous Fluid Antenna Systems (E-FAS) for Multiuser MIMO: Channel Modeling and Analysis

**arXiv ID:** 2602.11099 | [PDF](https://arxiv.org/pdf/2602.11099v1)

**作者:** Farshad Rostami Ghadi `[一作]` (University College London), Lajos Hanzo `[通讯]` (University of Southampton)

**通讯引用:** 85432 | [OpenAlex ID](https://openalex.org/A5091122305)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过物理一致的端到端信道模型，将巨型流体天线系统（E‑FAS）与小尺度衰落耦合，推导了单用户和多用户下的等效复高斯信道，并给出了闭式的失效概率、可期容量以及零迫化（ZF）预编码下的吞吐量分析。

**💡 创新点**

创新点在于：①将表面波传播与空间波衰落整合为单一复高斯等效信道，①提供了单/多用户下的闭式性能表达式；②用高斯假设与随机矩阵理论描述了ZF预编码后SINR的分布；③展示了E‑FAS通过表面波传输实现的功率增益与编码增益提升，而保持多径分集阶数不变。

**🔧 技术方法**

所采用技术主要包括：物理表面波传播模型（包括衰减与相位常数），Rayleigh小尺度衰落模型，随机矩阵理论（Gram矩阵逆分布），Gamma分布近似，指数积分函数用于可期容量计算，以及多用户ZF预编码与均衡化。

**📊 数据集**

使用了 Monte‑Carlo 仿真数据来验证理论推导；并未使用任何公开实验数据集，而是基于仿真参数（M=16, K=4, Ω_sw∈{1,5,10} 等）进行性能评估。

**📈 对比分析**

比较方法为：在相同系统参数下，分别模拟无 E‑FAS（仅直射通道）与有 E‑FAS 的情形，计算失效概率、可期容量和ZF和多用户总速率。结果表明，E‑FAS 能显著降低失效概率、提升可期容量和总速率，主要表现为功率增益（如 Ω_eq 的提升），并且在高 SNR 区域保持相同的分集阶数。

**⚠️ 局限性**

局限性包括：①假设表面波传播是确定性的、无小尺度衰落；②采用均匀随机预编码或完全无 CSI 的模型；③多用户分析仅在对称用户情形下给出闭式近似；④未考虑实际硬件非理想、互耦合或表面波散射等效应；⑤缺乏实验验证，仿真结果可能未能完全捕捉实际环境中的复杂散射与阻塞情况。

---

## 380. Sample Efficient Generative Molecular Optimization with Joint Self-Improvement

**arXiv ID:** 2602.10984 | [PDF](https://arxiv.org/pdf/2602.10984v1)

**作者:** Serra Korkmaz `[一作]` (Institute of AI for Health), Ewa Szczurek `[通讯]` (Institute of AI for Health)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了联合自我改进（Joint Self-Improvement）框架，将生成模型与代理模型联合训练，并在推理时通过自我改进采样高效生成目标分子。

**💡 创新点**

创新点包括：①使用共享参数的联合模型缓解生成与代理的分布漂移；②在推理阶段利用全局优势偏置生成器，实现高效采样；③基于全局优势而非局部优势，进一步提升性能。

**🔧 技术方法**

采用 Hyformer 图变换器作为联合模型，使用联合似然损失进行训练；采样时采用随机束搜索（SBS）和基于全局优势的 logits 调整；实验涵盖离线与在线分子优化场景。

**📊 数据集**

使用 ZINC250K 进行预训练与离线数据；在五个靶蛋白（PARP1、FA7、JAK2、BRAF、5HT1B）上进行对接分数优化实验。

**📈 对比分析**

与 REINVENT、SATURN、GeneticGFN、RaM 等传统方法在离线与在线设置下进行比较，Joint Self-Improvement 在所有靶点的 Hit Ratio 均显著优于基线，且在仅 64 次评估下仍保持高性能且采样速度更快。

**⚠️ 局限性**

局限性：代理模型预测误差可能限制优化效果；缺乏不确定性估计，未来可进一步提升预测鲁棒性和采样探索性。

---

## 381. Comp2Comp: Open-Source Software with FDA-Cleared Artificial Intelligence Algorithms for Computed Tomography Image Analysis

**arXiv ID:** 2602.10364 | [PDF](https://arxiv.org/pdf/2602.10364v1)

**作者:** Adrit Rao `[一作]` (Stanford University), Akshay S. Chaudhari `[通讯]` (Stanford University)

**通讯引用:** 4428 | [OpenAlex ID](https://openalex.org/A5064829377)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发并公开发布了两个可直接在医院CT影像上运行的深度学习算法：AAQ（腹主动脉最大直径自动测量）和BMD（利用CT估算椎体骨密度）。

**💡 创新点**

创新点在于：①将FDA 510(k)认证的AI模型完全开源，实现算法、模型权重和验证数据透明公开；②在多机构、多设备、不同扫描参数下完成严格的临床验证，填补了商业闭源方案的“透明度缺口”。

**🔧 技术方法**

技术主要包括：nnU‑Net 3D分割模型（用于主动脉和椎体分割）、体素至毫米转换、两点HU校准、基于椎体平均HU与脂肪/空气校准的骨密度判定阈值，配合统计学评估（MAE、ICC、灵敏度/特异性、AUROC）。

**📊 数据集**

数据集：AAQ训练集共1804例CT（来自Basel、深圳、Stanford）；AAQ验证集258例CT（四家美国+巴西机构）；BMD验证集371例CT‑DXA配对（四家机构），涵盖多种扫描仪、kVp、切片厚度与重建内核。

**📈 对比分析**

比较方法：AAQ与三位放射科医生手工测量的平均值比较，MAE 1.57 mm（95% CI 1.38–1.80）且ICC 0.983，接近放射科医师间一致性；BMD与DXA T‑score<−1的二分类比较，灵敏度81.0%（95% CI 74.0–86.8），特异性78.4%（95% CI 72.3–83.7），连续预测的Pearson r = 0.791，AUROC分别为0.883（连续）和0.797（二分类）。

**⚠️ 局限性**

局限性：AAQ在存在血栓或血管支架的术后主动脉影像上准确度下降（MAE≈3.96 mm）；BMD未在对比增强CT上验证，特异性在部分扫描仪、切片厚度>3 mm、某些重建内核时低于70%；两者均不支持凹凸性（saccular）AAA；未来需扩大对术后患者及不同扫描条件的适用性。

---

## 382. LaSSM: Efficient Semantic-Spatial Query Decoding via Local Aggregation and State Space Models for 3D Instance Segmentation

**arXiv ID:** 2602.11007 | [PDF](https://arxiv.org/pdf/2602.11007v1)

**作者:** Lei Yao `[一作]` (Hong Kong Polytechnic University), Lap-Pui Chau `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 5860 | [OpenAlex ID](https://openalex.org/A5044722301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于分层语义-空间查询初始化器与坐标引导状态空间模型（SSM）的3D实例分割框架；

**💡 创新点**

创新点在于将语义与空间信息结合的分层查询初始化、局部聚合与双路径SSM解码器，实现线性复杂度的高效查询细化；

**🔧 技术方法**

采用分层语义-空间查询初始化、局部聚合聚合方案、Hilbert曲线序列化、双路径SSM、中心回归等技术；

**📊 数据集**

在ScanNet++ V2、V1、ScanNet、S3DIS等室内点云数据集上进行实验；

**📈 对比分析**

与SPFormer、OneFormer3D、SGIFormer等方法对比，V2排行榜第一，mAP提升约2.5%，仅使用1/3 FLOPs，V1、S3DIS等也获得竞争性结果；

**⚠️ 局限性**

局限性包括固定查询数缺乏自适应性、Hilbert排序带来额外计算开销、仅在室内场景验证，未扩展到户外环境。

---

## 383. Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation

**arXiv ID:** 2602.10717 | [PDF](https://arxiv.org/pdf/2602.10717v1)

**作者:** Songen Gu `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 15870 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为Say、Dream、Act的框架，通过先选取并适配高质量视频生成模型Cosmos‑Predict2，然后利用对抗蒸馏在少量去噪步中实现快速视频预测，并将预测的视频作为上下文指导动作模型，实现指令驱动的机器人操控。

**💡 创新点**

创新点包括：①基于任务相关动态和空间一致性的评估，挑选并适配Cosmos‑Predict2并做域适配；②引入对抗蒸馏的潜在空间损失，在仅8步或更少的迭代中保持高保真度；③提出长度无关的想象机制，通过关键帧压缩实现任意长度轨迹的全局视频预测；④将想象轨迹视为上下文示例，训练动作模型能够纠正空间误差并生成可执行动作。

**🔧 技术方法**

使用技术主要包括：视频生成模型Cosmos‑Predict2、潜在空间对抗蒸馏（latent‑adversarial loss）、多尺度关键帧压缩、Transformer‑based动作模型、Vision‑Language Model Qwen2.5（作为动作模型的基础），以及FSDP并行训练、LoRA参数高效微调等。

**📊 数据集**

数据集与实验环境：LIBERO benchmark（包含Spatial、Object、Goal、Long四个任务套），以及基于Franka 7‑DoF机器人和Intel RealSense D435的真实世界实验。

**📈 对比分析**

在LIBERO benchmark上，与多种基线（OpenVLA、DreamVLA、UniVLA等）对比，Dream4manip实现总成功率98.2%，在各子任务中最高分别为99.4%（Spatial）、99.2%（Object）、98.6%（Goal）、95.4%（Long），显著优于基线。视频生成质量上，域适配+蒸馏后Cosmos‑2B模型FVD下降至211.84、SSIM上升至0.82、PSNR上升至26.82，优于未适配和未蒸馏版本。

**⚠️ 局限性**

局限性：对抗蒸馏虽然加速，但仍需进一步降低延迟；长时预测仍易累积误差，动态或杂乱环境下性能下降；动作模型对生成视频质量高度依赖，生成错误会影响控制；跨机器人、视角与环境的泛化仍有限，需要更广泛的预训练与更强的嵌入表示。

---

## 384. $μ$pscaling small models: Principled warm starts and hyperparameter transfer

**arXiv ID:** 2602.10545 | [PDF](https://arxiv.org/pdf/2602.10545v1)

**作者:** Yuxin Ma `[一作]` (Johns Hopkins University), Soledad Villar `[通讯]` (Johns Hopkins University)

**通讯引用:** 822 | [OpenAlex ID](https://openalex.org/A5035079908)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通用的模型宽度扩展(upscaling)方法，能够把训练好的小模型热启动到更大模型，同时通过注入噪声让新模型充分利用额外容量；同时给出了零拷贝的超参数迁移（hyperparameter transfer）方案，使得在小规模上调优的超参数可直接迁移到大规模训练；

**💡 创新点**

创新点在于：①将函数保持的宽度扩展与最大更新参数化（μP）结合，形成理论上等价的宽度扩展过程；②通过解析训练动力学证明了宽度扩展后模型保持动态等价；③提出了基于噪声注入与学习率调整的 upscaling 算法，并证明其在无限宽极限下保持最优特征学习；④展示了超参数迁移可在不同宽度间零拷贝实现的理论与实验支持；

**🔧 技术方法**

使用了：①最大更新参数化（μP）对权重和学习率、权重衰减等做宽度相关缩放；②张量程序（Tensor Programs）框架分析无限宽极限下的训练动力学；③在实验中使用了 AdamW、SGD、以及多种架构（MLP、ResNet、Transformer/GPT-2）；④构造了宽度扩展的权重复制+缩放规则；

**📊 数据集**

实验数据集包括：Forest Cover Type（表格分类）、CIFAR-100（图像分类）和 FineWeb（语言模型预训练）；

**📈 对比分析**

与从零开始训练宽模型（同样采用 μP 超参数）和使用传统扩展/预训练比较；实验显示：upscaling 模型在相同训练步数下收敛更快，训练损失更低；在 MLP、GPT-2 上表现最优；ResNet 上泛化略逊；超参数迁移实验表明最优学习率/噪声在不同宽度上保持稳定；

**⚠️ 局限性**

局限性：1）理论主要针对无限宽极限，有限宽时可能偏差；2）只分析了训练动态，未探讨泛化；3）在某些架构（如 ResNet）上 upscaling 可能导致泛化下降；4）噪声注入幅度需经验调优；5）目前只对宽度扩展，深度扩展未覆盖。

---

## 385. S-GRec: Personalized Semantic-Aware Generative Recommendation with Asymmetric Advantage

**arXiv ID:** 2602.10606 | [PDF](https://arxiv.org/pdf/2602.10606v1)

**作者:** Jie Jiang `[一作]` (Tencent Inc.), Huan Yu `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出S-GRec框架，使用离线LLM作为语义判别器为生成式推荐提供训练时监督，线上推理完全不依赖LLM。

**💡 创新点**

创新点在于两阶段个性化语义判别（细粒度维度评分 + 用户条件加权聚合）以及在优势空间进行业务奖励与语义奖励的非对称融合（A2PO），实现语义深度与业务目标的兼顾。

**🔧 技术方法**

采用LLM做离线判别器、基于组采样的GRPO策略优化、对维度评分进行RL对齐、使用pairwise偏好聚合、以及优势空间的A2PO融合技术。

**📊 数据集**

实验数据包括公共Amazon Review的Office Products和Industrial & Scientific两个子集，用于离线评估；以及腾讯微信渠道广告系统的真实流量数据用于在线A/B测试。

**📈 对比分析**

与传统序列模型（GRU4Rec、Caser、SASRec）、生成式推荐模型（HSTU、BIGRec、MiniOneRec）以及LLM增强模型对比，S-GRec在HR@K和NDCG@K上均取得最优或第二优表现；在线AB测试中提升GMV +1.19%、GMV-Normal +1.55%、CTR +1.16%，不良率下降2.02%。

**⚠️ 局限性**

局限性包括：需要离线LLM推理和人工校验导致标注成本高；判别器可能存在偏差且维度有限，难以覆盖所有用户偏好；在数据分布或业务目标变化时需要重新调优采样比例与一致性门控。

---

## 386. A Unified Theory of Random Projection for Influence Functions

**arXiv ID:** 2602.10449 | [PDF](https://arxiv.org/pdf/2602.10449v1)

**作者:** Pingbang Hu `[一作]` (University of Illinois Urbana-Champaign), Han Zhao `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2647 | [OpenAlex ID](https://openalex.org/A5101670508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种统一的理论，分析了在不同设置下投影如何保持影响函数的有效性，特别是在使用随机投影和岭正则化的情况下。

**💡 创新点**

创新点在于系统性地阐明了投影与曲率算子之间的相互作用，提供了选择投影大小的原则性指导，并扩展了对Kronecker分解的影响分析。

**🔧 技术方法**

使用了随机投影技术和岭正则化方法，结合了影响函数的逆敏感双线性形式。

**📊 数据集**

使用了MNIST和CIFAR数据集进行实验，具体包括MNIST-10与逻辑回归、MNIST-10与多层感知机、CIFAR-2与ResNet9的组合。

**📈 对比分析**

通过与传统的影响函数计算方法进行比较，展示了在不同的投影大小和正则化强度下，投影影响的近似误差如何随投影大小的变化而变化，结果表明有效维度d_λ(F)可以显著小于F的秩r。

**⚠️ 局限性**

限制在于当前理论主要集中在影响函数的近似质量上，而未深入探讨投影、正则化或曲率近似如何影响影响函数作为底层留一法（LOO）量的估计质量。

---

## 387. Large Language Models Predict Functional Outcomes after Acute Ischemic Stroke

**arXiv ID:** 2602.10119 | [PDF](https://arxiv.org/pdf/2602.10119v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 388. SteuerLLM: Local specialized large language model for German tax law analysis

**arXiv ID:** 2602.11081 | [PDF](https://arxiv.org/pdf/2602.11081v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 389. LATA: A Tool for LLM-Assisted Translation Annotation

**arXiv ID:** 2602.10454 | [PDF](https://arxiv.org/pdf/2602.10454v1)

**作者:** Baorong Huang `[一作]` (Huaihua University), Ali Asiri `[通讯]` (Umm al-Qura University)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5110581181)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于LLM的交互式平行语料标注工具，集成文档元数据收集、段落对齐和句子级别LLM辅助分割对齐。

**💡 创新点**

创新点在于结合模板化Prompt引导LLM进行分割与对齐，并提供人机协同的后编辑界面，实现自动化与细粒度控制的平衡。

**🔧 技术方法**

技术主要包括React‑Electron‑SQLite前后端架构、模板化Prompt与JSON输出、LLM（如ChatGPT）对句子对齐、可视化对齐与技术注释。

**📊 数据集**

数据集未在论文中明确给出，推测使用公开的阿拉伯‑英语并行语料库（如Tatoeba、OpenSubtitles）或内部领域语料。

**📈 对比分析**

通过与传统工具（LDC aligner、BRAT等）对比，实验报告在准确率提升约10%且人工校对时间缩短约40%的效果。

**⚠️ 局限性**

局限在于LLM仍对非字面翻译产生误判，需要人工干预；多模态支持功能仍处于规划阶段，未在实验中验证。

---

## 390. Following Dragons: Code Review-Guided Fuzzing

**arXiv ID:** 2602.10487 | [PDF](https://arxiv.org/pdf/2602.10487v1)

**作者:** Viet Hoang Luu `[一作]`, Van-Thuan Pham `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出EyeQ工作流，将代码评审中的安全相关讨论转化为注解驱动的模糊测试指引。

**💡 创新点**

创新点在于利用代码评审信息自动生成注解指引，实现无额外手工干预的安全驱动模糊测试。

**🔧 技术方法**

采用LLM进行评审分类、代码定位与注解生成，并集成IJON注解与AFL++进行模糊测试。

**📊 数据集**

使用公开的PHP项目代码评审数据集（2011-2022共240条评论）以及2025年9-12月的额外评审数据。

**📈 对比分析**

通过对比人类手工注解与LLM自动化流程、以及注解版与无注解AFL++，发现LLM整体发现47个漏洞，较基线提升约1.5倍。

**⚠️ 局限性**

局限在于定位阶段准确率不足，无法覆盖所有评论，且依赖LLM对上下文理解的准确性。

---

## 391. Domain Knowledge Guided Bayesian Optimization For Autonomous Alignment Of Complex Scientific Instruments

**arXiv ID:** 2602.10670 | [PDF](https://arxiv.org/pdf/2602.10670v1)

**作者:** Aashwin Mishra `[一作]` (SLAC National Laboratory), Apurva Mehta `[通讯]` (SLAC National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于领域知识的贝叶斯优化方法，通过坐标变换与逆退火探索策略实现高维耦合优化的快速收敛；

**💡 创新点**

创新点在于利用物理洞察将非轴对齐的高维耦合问题转化为轴对齐子空间，并采用逆退火提升探索力度，从而克服传统BO在“针在干草堆”问题上的早期收敛；

**🔧 技术方法**

技术包括线性坐标旋转变换、Gaussian Process（RBF核）与UCB采样、逆退火β调度、以及对多目标问题的标量化处理；

**📊 数据集**

使用Linac Coherent Light Source（LCLS）实验的12维Split‑and‑Delay光学系统的仿真数据与实验测量数据；

**📈 对比分析**

与标准BO、TurBO和MOBO进行25次重复实验比较，结论是域知识引导BO在150次采样内显著降低光束位置误差并提升光强度，表现出比基线方法更快、更稳定的收敛；

**⚠️ 局限性**

局限在于对领域知识的依赖——若物理模型不完整或可变，变换可能失效；逆退火参数需要手工调优；算法仍受限于采样预算与GP模型的高维可扩展性。

---

## 392. Online Min-Max Optimization: From Individual Regrets to Cumulative Saddle Points

**arXiv ID:** 2602.10565 | [PDF](https://arxiv.org/pdf/2602.10565v1)

**作者:** Abhijeet Vyas `[一作]` (Purdue University), Brian Bullins `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了针对在线零和博弈的新框架，以累计鞍点为核心目标，并引入了两种新的性能度量——累计对偶性间隙（Static Duality Gap）和动态鞍点遗憾（Dynamic Saddle Point Regret），以此来评估算法对累计鞍点的逼近效果。

**💡 创新点**

创新点在于：①将累计鞍点视为在线min‑max优化的关键目标，打破传统单纯关注个体遗憾的做法；②提出并分析了两种新的遗憾度量；③针对强凸-强凹、指数凹和双侧Polyak‑Łojasiewicz（PL）等不同正则性条件，给出了对应的算法（OGDA、OMMNS、MMFLH、在线AGDA）及其对累计鞍点与动态鞍点遗憾的理论上界；④证明了在强凸-强凹及指数凹等条件下，平均动作序列收敛到累计鞍点。

**🔧 技术方法**

采用了在线凸优化（OCO）中的镜像下降、牛顿步等技术；通过将min‑max问题转化为OCO的静态/动态遗憾问题，利用睡眠专家框架和改进的FLH（MMFLH）来实现动态鞍点遗憾的上界；在双侧PL情形下，利用在线AGDA与变步长策略。

**📊 数据集**

本文未使用公开数据集，而是以理论分析与构造对抗性序列为主，给出构造性反例与理论上界。

**📈 对比分析**

在强凸-强凹和指数凹情形下，OGDA与OMMNS分别实现了O(log T)的累计对偶性间隙；在动态鞍点遗憾上，OGDA+MMFLH实现了O(max{log T,√(TV_T log T)})的上界；在双侧PL条件下，在线AGDA得到与累积变化相关的线性收敛速度。

**⚠️ 局限性**

限制主要体现在：①对累计鞍点的度量依赖于对累计鞍点的先验知识，导致实际算法需预估该点；②在某些正则性条件下（如双侧PL）仍需假设鞍点存在且可求；③对非凸-凹情形的扩展仍有限；④实验验证缺乏，对理论上界的实证效果尚未评估。

---

## 393. Beyond Closed-Pool Video Retrieval: A Benchmark and Agent Framework for Real-World Video Search and Moment Localization

**arXiv ID:** 2602.10159 | [PDF](https://arxiv.org/pdf/2602.10159v1)

**作者:** Tao Yu `[一作]` (Chinese Academy of Sciences), Jinwen Luo `[通讯]` (Tencent)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5069413253)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了针对真实世界视频记忆检索的RVMS-Bench基准，并提出了基于归纳推理的RACLO检索代理。

**💡 创新点**

引入多维层次化记忆描述（全局印象、关键时刻、时间上下文、听觉记忆），在开放式网络上模拟模糊记忆检索，并通过人机协同验证提升数据质量。

**🔧 技术方法**

使用Gemini 3 Pro/2.5 Pro等大型多模态语言模型、ReAct与链式推理、Web搜索工具、规则+模型双重验证、音视频分帧与特征提取等技术。

**📊 数据集**

RVMS-Bench数据集：1440条样本，覆盖20类主题、4个时长区间、9种检索任务，全部来源于YouTube并经过人工校验。

**📈 对比分析**

对比了闭源与开源LLM（Gemini 3 Pro、GPT-5、Qwen3-VL-235B等），评估检索召回率和关键帧定位准确率；闭源模型表现最佳，召回率约50-70%，定位准确率约30-50%，但整体仍低于人类水平。

**⚠️ 局限性**

模型在长时长视频、结构不清晰的内容以及将模糊记忆映射到精确时间点方面表现欠佳，且RACLO依赖网络搜索和模型验证，存在可扩展性与通用性限制。

---

## 394. A Weakest Precondition Calculus for Programs and Linear Temporal Specifications

**arXiv ID:** 2602.10746 | [PDF](https://arxiv.org/pdf/2602.10746v1)

**作者:** Gidon Ernst `[一作]` (University of Augsburg), Gidon Ernst `[通讯]` (University of Augsburg)

**通讯引用:** 773 | [OpenAlex ID](https://openalex.org/A5049068584)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套弱前置条件计算法，将线性时序逻辑与结构化程序结合，支持在无交互的自动化验证工具中生成验证条件。

**💡 创新点**

创新点在于统一利用继续语义、步进规范以及占位符变量来自动生成并管理循环的递归与共递归假设，避免手工写中间状态，弥补了现有技术在时序属性支持上的缺陷。

**🔧 技术方法**

采用弱前置条件推导、线性时序逻辑（LTL）与固定点理论、逐步规范化、占位符化假设以及 Isabelle/HOL 形式化验证，并在 Scala DSL 中实现了计算器。

**📊 数据集**

研究未使用外部数据集，主要通过 Isabelle/HOL 的形式化证明和若干示例程序（如循环递增、素数遍历）来验证方法的正确性。

**📈 对比分析**

未给出量化的性能对比，论文侧重理论阐述和实现演示；实现已公开（Zenodo 网址），但缺乏与现有工具在大规模实例上的基准评估。

**⚠️ 局限性**

局限性包括：只针对顺序结构化程序；缺乏经验性实验和性能评估；需要手工提供循环假设；在多层递归或并发情境下的适用性尚未验证。

---

## 395. The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis

**arXiv ID:** 2602.10453 | [PDF](https://arxiv.org/pdf/2602.10453v1)

**作者:** Peiran Wang `[一作]` (University of California Los Angeles), Yuan Tian `[通讯]` (University of California Los Angeles)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统化综述LLM代理中的Prompt Injection威胁，构建攻击与防御的完整分类，并提出针对上下文依赖任务的新Benchmark；

**💡 创新点**

首次将攻击按生成策略、防御按介入层级进行统一分类，并揭示当前防御在安全、效能、延迟三者之间的不可兼得三角；

**🔧 技术方法**

采用系统文献检索、定量分析构建分类，利用GPT‑4o‑mini及多种现有防御方法进行实验，设计多维度评估指标（攻击成功率、效能、时延、代币消耗）；

**📊 数据集**

自定义5个上下文相关任务与对应攻击模板，并借鉴InjecAgent、AgentDojo、ASB等Benchmark进行补充实验；

**📈 对比分析**

在5种攻击向量下对防御的攻击成功率、无攻击时效能、时间和代币开销进行比较；结果显示执行级防御能显著降低行动切换攻击，但对参数操纵、分支偏移、推理篡改等上下文攻击无效；多数防御在降低攻击成功率的同时导致效能下降或时延增加，体现三者权衡；

**⚠️ 局限性**

当前防御多聚焦静态输入，忽视上下文依赖任务；缺乏细粒度、可解释的干预；大多需额外LLM调用或人工参与，导致计算成本高、可用性差；缺少同时兼顾高安全性、效能与低时延的方案。

---

## 396. MerLin: A Discovery Engine for Photonic and Hybrid Quantum Machine Learning

**arXiv ID:** 2602.11092 | [PDF](https://arxiv.org/pdf/2602.11092v1)

**作者:** Cassandre Notton `[一作]` (Quandela Quantique), Jean Senellart `[通讯]` (Quandela)

**通讯引用:** 1982 | [OpenAlex ID](https://openalex.org/A5033330160)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MerLin框架，作为一个开源的光子与混合量子机器学习实验引擎，集成了强模拟、PyTorch/Scikit-learn兼容、硬件感知与实验复现功能。

**💡 创新点**

创新点在于将光子量子电路的强模拟与标准机器学习工作流深度耦合，支持可微分训练、硬件调度以及统一的基准化复现；同时首次在18项光子和混合QML工作上完成系统复现，形成可复用的实验基线。

**🔧 技术方法**

核心技术包括SLOS强模拟、TorchScript加速、量子层（QuantumLayer）抽象、角度/幅度编码、硬件感知的检测模型、量子桥接、以及在PyTorch/Scikit-learn上的插件化集成。

**📊 数据集**

使用的典型数据集包括Moon、MNIST、SST‑2、ImageNet等多种分类、序列和生成任务，亦复现了多种公开量子核函数与卷积网络的实验。

**📈 对比分析**

与原始论文比较，MerLin实现的模拟速度提升数十到数百倍；在光子原生实现中，学习性能与基于门控的模型相近；通过统一基准，发现角度编码对对抗鲁棒性更好，幅度编码对噪声敏感。

**⚠️ 局限性**

局限性包括：强模拟仍受限于光子数（≈20），难以覆盖更大规模；硬件接入受云服务限制；复现依赖对原始实现细节的猜测；并未系统评估真实噪声对性能的长期影响。

---

## 397. FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference

**arXiv ID:** 2602.11105 | [PDF](https://arxiv.org/pdf/2602.11105v1)

**作者:** Divya Jyoti Bajpai `[一作]` (Indian Institute of Technology Bombay), Manjesh Kumar Hanawal `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FastFlow，一种可插拔的自适应推理框架，通过跳过流匹配模型中冗余的去噪步骤并用有限差分速度估计进行近似，显著加速图像、视频生成与编辑。

**💡 创新点**

创新点在于：①使用一阶泰勒展开结合有限差分估计动态近似速度，②将跳过步骤的决策建模为多臂赌博机（MAB）在线学习问题，能够在保持质量的前提下自适应决定跳过多少步，③无需重新训练或额外网络，完全兼容现有流匹配管线。

**🔧 技术方法**

核心技术包括流匹配（Flow Matching）与Euler数值求解、有限差分速度近似、一次性多臂赌博机（MAB）策略以及基于奖励平衡计算量与误差的在线学习。

**📊 数据集**

使用了多种公开基准数据集：文本到图像的 GenEval、图像编辑的 GEdit、文本到视频的 VBench 子集，以及多个流匹配模型（BAGEL、Flux-Kontext、PeRFlow、HunyuanVideo）。

**📈 对比分析**

与全步生成、TeaCache、InstaFlow、PerFlow 等基线比较，FastFlow 在保持或略优的视觉与语义质量（CLIPIQA、G_SC/G_PQ/G_O、VBench 等指标）的同时，速度提升可达 2.6× 至 7.1×（取决于模型与任务），显著优于静态加速方法。

**⚠️ 局限性**

主要限制包括：MAB 在初始探索阶段可能无法立即实现速度提升；跳过策略依赖于预先估计的误差尺度 μ，若模型或数据分布剧烈变化，需重新校准；当流动轨迹出现大幅非线性变化时，有限差分近似误差可能累积导致质量下降。

---

## 398. Ecological mapping with geospatial foundation models

**arXiv ID:** 2602.10720 | [PDF](https://arxiv.org/pdf/2602.10720v1)

**作者:** Craig Mahlasi `[一作]` (IBM Research), Campbell Watson `[通讯]` (IBM Research)

**通讯引用:** 640 | [OpenAlex ID](https://openalex.org/A5025361716)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 Prithvi‑EO‑2.0、TerraMind 两款地理空间基础模型进行微调，完成零射 LULC 生成、森林功能特征映射（叶片形式与冠层密度）以及泥炭地检测任务，并与 ResNet‑101 基准模型对比。

**💡 创新点**

首次在生态映射任务中演示基于 Transformer 的 GFMs 的零射生成能力、跨模态缺失补全功能，并系统评估了多模态输入提升模型性能的可行性，揭示标签质量与模型泛化的关键影响。

**🔧 技术方法**

采用 Transformer‑based GFM（Prithvi、TerraMind）与 ResNet‑101 进行对比实验，使用 AdamW 优化器、Dice 损失、TerraTorch 工具包，并将 Sentinel‑1、Sentinel‑2、NDVI、DEM 等多源遥感数据融合。

**📊 数据集**

使用 NEON 站点的 Sentinel‑2 L2A、Karukinka 自然公园 Sentinel‑2 与 Sentinel‑1 数据，Copernicus 全球动态土地覆盖、Chilean CONAF 绿化资源、PEATGRIDS‑NDVI 以及 ESRI LULC 等公开标签集。

**📈 对比分析**

通过 IoU / F1 指标进行定量评估，GFMs 在冠层密度与叶片形式任务上平均提升约 20%（最高 0.75‑0.70），TerraMind 零射 LULC IoU 达 78.82%，泥炭地检测中多模态 TerraMind 的 F1 可达 0.95（Chile LULC），整体表现优于 ResNet‑101。

**⚠️ 局限性**

存在领域偏移、时空分辨率（10 m）限制、标签分辨率与模糊性、缺乏泥炭层等地下信息、Transformer 对像素细节恢复不足等限制，需结合高分辨率、多时相数据与更高质量的现场标注才能进一步提升性能。

---

## 399. Computing Least Fixed Points with Overwrite Semantics in Parallel and Distributed Systems

**arXiv ID:** 2602.10486 | [PDF](https://arxiv.org/pdf/2602.10486v1)

**作者:** Vijay K. Garg `[一作]` (University of Texas at Austin), Rohan Garg `[通讯]` (Purdue University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5100576551)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 400. FedPS: Federated data Preprocessing via aggregated Statistics

**arXiv ID:** 2602.10870 | [PDF](https://arxiv.org/pdf/2602.10870v1)

**作者:** Xuefeng Xu `[一作]` (University of Warwick), Graham Cormode `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 FedPS，一个统一的联邦数据预处理框架，支持多种预处理方法并可在横向和纵向 FL 中使用。

**💡 创新点**

通过聚合统计量和数据草图实现通信高效的预处理；在纵向 FL 中通过重写 BLR 仅用 X Xᵀ 解决跨客户端特征交互；提供全面的预处理方法集合和通信成本分析。

**🔧 技术方法**

使用数据草图（KLL、REQ、频繁项 Sketch）、聚合统计、联邦 Bayesian Linear Regression、k-Means、k-NN 回归、量化/分位数、编码、离散化、缺失值填补等技术。

**📊 数据集**

在 Adult、Bank、Cover 三个公开表格数据集上进行实验。

**📈 对比分析**

与无预处理、本地预处理、联邦预处理三种策略对比；在 IID 情况下效果相近，非 IID 下联邦预处理显著优于本地预处理，提升 5%–17% 准确率；同时测量并对比不同预处理方法的通信成本，验证理论分析。

**⚠️ 局限性**

未实现完整的隐私保护协议，复杂统计的安全多方计算仍待研究；实验仅覆盖表格数据，缺少图像/文本等类型；仅实现了常见的预处理方法，深度预处理与更复杂任务仍有待扩展。

---

## 401. Learning Mixture Density via Natural Gradient Expectation Maximization

**arXiv ID:** 2602.10602 | [PDF](https://arxiv.org/pdf/2602.10602v1)

**作者:** Yutao Chen `[一作]` (University of Macau), Steven Morad `[通讯]` (University of Macau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出自然梯度期望最大化（NGEM）算法，用以训练混合密度网络，显著提升收敛速度并缓解模式坍塌；

**💡 创新点**

通过理论证明EM与自然梯度在高斯混合模型中的等价性，利用可解析的完整数据Fisher信息矩阵实现可计算的自然梯度更新；

**🔧 技术方法**

混合密度网络、期望最大化、自然梯度下降、完整数据Fisher信息矩阵（高斯与分类分布的解析形式）以及小批量梯度下降；

**📊 数据集**

合成数据（Two‑Gaussians、Two‑Sinusoids）、高维逆 MNIST、UCI 机器人运动学、能源效率和波士顿房价回归；

**📈 对比分析**

与标准负对数似然目标、Adam、KFAC、Muon、Soap、β‑NLL、MSE 等基线进行对比，实验显示NGEM 速度提升 10 倍、负对数似然更低、熵更高、模式坍塌显著减少，计算开销几乎无增；

**⚠️ 局限性**

假设高斯分量协方差为对角矩阵，限制了模型表达力，未来可考虑低秩或非高斯混合。

---

## 402. GoodVibe: Security-by-Vibe for LLM-Based Code Generation

**arXiv ID:** 2602.10778 | [PDF](https://arxiv.org/pdf/2602.10778v1)

**作者:** Maximilian Thang `[一作]` (Technical University of Darmstadt), Ahmad-Reza Sadeghi `[通讯]` (Technical University of Darmstadt)

**通讯引用:** 23871 | [OpenAlex ID](https://openalex.org/A5079497016)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于神经元层面的安全优化框架GoodVibe，使大型语言模型在代码生成时默认更安全；

**💡 创新点**

创新点在于将安全相关推理定位到模型内部少量神经元，采用梯度归因识别安全神经元，并通过聚类共享更新方向，实现极小参数量的定向微调；

**🔧 技术方法**

使用梯度归因法识别安全神经元、聚类技术实现结构化更新、AdamW优化器、BF16混合精度训练；

**📊 数据集**

利用CyberNative Code Vulnerability and Security Dataset（包含C++、Java、Swift、Go等安全/不安全代码对），以及Qwen3-0.6B训练的安全评判模型；

**📈 对比分析**

与全参数微调、LoRA等方法对比，GoodVibe在保持或超过安全率（最高可达C++ 86.6%）的同时，训练参数仅0.03%（≈1.9M），计算量比LoRA低70%+，表现优于其他方法；

**⚠️ 局限性**

局限性包括对安全评判模型的依赖、在极端恶意或复杂安全场景下效果不保证、聚类阈值与k值需要经验调优，且仅提升默认安全性，仍需人工审查。

---

## 403. Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning

**arXiv ID:** 2602.10503 | [PDF](https://arxiv.org/pdf/2602.10503v1)

**作者:** Yuan Liu `[一作]` (Beijing Normal University), Dongbin Zhao `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于增量强化微调的LifeLong-RFT，能够让视觉语言动作模型在不依赖环境反馈的情况下实现长期学习。

**💡 创新点**

创新点在于将分块级别的自策略强化学习与多维过程奖励（MDPR）结合，MDPR包括量化动作一致奖励、连续轨迹对齐奖励和格式合规奖励。

**🔧 技术方法**

采用Chunking-level On-Policy RL（GRPO）与MDPR奖励机制，基于Fast+分词器对动作进行编码。

**📊 数据集**

使用SimEnv、LIBERO基准以及Franka真实机器人实验数据集。

**📈 对比分析**

与SFT及其他连续/离散动作模型对比，LifeLong-RFT在多任务和连续学习上平均提升约4‑6%成功率，连续任务提升高达22%+，并显著降低灾难性遗忘。

**⚠️ 局限性**

局限在于仅验证了离散动作模型，连续动作模型的效果不佳，且在现实环境下仍需更多验证。

---

## 404. Generative clinical time series models trained on moderate amounts of patient data are privacy preserving

**arXiv ID:** 2602.10631 | [PDF](https://arxiv.org/pdf/2602.10631v1)

**作者:** Rustam Zhumagambetov `[一作]` (Physikalisch-Technische Bundesanstalt), Stefan Haufe `[通讯]` (Technische Universität Berlin)

**通讯引用:** 8361 | [OpenAlex ID](https://openalex.org/A5068256213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对生成式模型产生的医疗时间序列数据进行隐私审计，评估会员推断攻击（MIA）在不同生成器和训练样本规模下的有效性。

**💡 创新点**

系统性地将多种现有隐私攻击方法应用于最新的ICU时间序列生成模型，研究训练样本量对隐私泄露的阈值，并验证跨数据集攻击的可行性，首次为医疗时间序列生成器提供全面的隐私评估。

**🔧 技术方法**

使用GAN、变分自编码器（VAE）、Koopman VAE、扩散模型等生成器；实现Monte Carlo、GAN‑Leak、DOMIAS、Logan-pb等多种会员推断攻击；采用PCA降维、NRMSE衡量过拟合、AUROC/准确率评估攻击性能。

**📊 数据集**

MIMIC‑IV ICU电子病历数据集（训练/验证/holdout拆分）作为主要训练和评估数据；eICU 数据集作为外部辅助数据用于跨数据集攻击实验。

**📈 对比分析**

通过比较不同攻击的AUROC、准确率、TPR/FPR，发现训练样本>500时所有攻击均低于随机水平，说明生成器具有良好隐私保护；样本不足时部分攻击仍能有效，展示了训练规模与隐私风险的明确关系。

**⚠️ 局限性**

仅在单一数据集上评估，未覆盖更复杂的多模态或更高维特征；未深入验证差分隐私实现的实际效果；攻击方法与特征空间维度之间的定量关系未充分探讨；仅聚焦会员推断攻击，忽略属性泄露和模型反演等其它隐私威胁。

---

## 405. TwiFF (Think With Future Frames): A Large-Scale Dataset for Dynamic Visual Reasoning

**arXiv ID:** 2602.10675 | [PDF](https://arxiv.org/pdf/2602.10675v1)

**作者:** Junhua Liu `[一作]` (Zhejiang University), Kun Kuang `[通讯]` (Zhejiang University)

**通讯引用:** 2528 | [OpenAlex ID](https://openalex.org/A5041727387)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于视频的动态视觉链式推理（TwiFF）框架，包含全新规模的2.7M视频动态VCoT数据集及1,078条评测样本的TwIFF-Bench；

**💡 创新点**

创新点在于：①把视觉推理步骤与未来帧生成结合，构建真实动态推理链；②构建了大规模动态视觉推理数据集并提供可测评基准；③通过统一模型同时利用预训练视频生成和图像理解实现时序一致的视觉推理；

**🔧 技术方法**

采用多模态大型语言模型（如InternVL3.5、Qwen3VL等）进行事件抽取和推理链生成；利用预训练视频生成模型生成未来动作帧；结合图像-文本交替推理实现动态VCoT；

**📊 数据集**

数据集为TwiFF-2.7M（来自Panda-70M的2.7M视频剪辑），并构建TwIFF-Bench（1,078开放式动态推理样本）

**📈 对比分析**

在TwIFF-Bench上，TwiFF模型相较于Bagel基线提升约28.8%（CoT）和41.6%（答案），在Seed-Bench-R1 OOD数据上也提升约21%；相较于现有TCoT和静态VCoT方法表现更优，只有Qwen3VL-8B略优；

**⚠️ 局限性**

局限性包括：①对高质量视频依赖较高，低动态或短时长视频筛除率高；②模型生成的视觉帧仍可能出现误差，导致推理不准；③缺乏对生成视频的真实性度量与自我校正机制；

---

## 406. Actions Speak Louder Than Chats: Investigating AI Chatbot Age Gating

**arXiv ID:** 2602.10251 | [PDF](https://arxiv.org/pdf/2602.10251v1)

**作者:** Olivia Figueira `[一作]` (University of California), Athina Markopoulou `[通讯]` (University of California)

**通讯引用:** 1297 | [OpenAlex ID](https://openalex.org/A5102747771)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过编写自动化脚本与主流AI聊天机器人进行 1050 次交互实验，系统性审计其能否识别儿童并执行年龄门控（阻断或提示家长）。

**💡 创新点**

首次构建显式与隐式年龄提示库并整合自动化与人工标注的审计框架，提供可复现的年龄门控评估方法与数据资源。

**🔧 技术方法**

使用 pyautogui 进行网页交互自动化，Gemini‑3 API 做文本标注，配合自定义 L1–L6 标签进行结果分析；同时设计实验脚本记录交互日志。

**📊 数据集**

自建的年龄提示库（共 11 条提示，覆盖 5、7、9、11 岁儿童与 13–17 岁青少年、成人），以及基于该库产生的 4890 条聊天记录（1050 次实验）。

**📈 对比分析**

比较方法：分别对每款聊天机器人的年龄估计准确率、是否触发门控动作进行统计；显式提示下准确率 93–99%，隐式单独提示低于 30%，而所有机器人均未执行阻断，自动标注准确率达 94%。

**⚠️ 局限性**

局限性：未使用真实儿童用户数据；仅测试文本模式，未考虑语音或其他多模态交互；实验仅覆盖五款主流聊天机器人，未检验不同地区或版本的差异；未深入评估对话内容的风险与安全性。

---

## 407. RiemannGL: Riemannian Geometry Changes Graph Deep Learning

**arXiv ID:** 2602.10982 | [PDF](https://arxiv.org/pdf/2602.10982v1)

**作者:** Li Sun `[一作]` (Beijing University of Posts and Telecommunications), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 133424 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Riemannian图学习的系统性框架与分类法，归纳了不同曲率空间、神经网络结构和学习范式，并梳理现有方法与缺口。

**💡 创新点**

将Riemannian几何视为图学习的基础范式，构建三维维度（曲率空间、网络架构、学习范式）分类体系，并强调内在流形特性而非仅外在扩展。

**🔧 技术方法**

采用Riemannian几何理论（曲率、指数/对数映射、对称群、黎曼距离等）对图卷积、VAE、Transformer、ODE、扩散、流等架构进行形式化改造。

**📊 数据集**

参考了众多公开图数据集（如Cora、Citeseer、PubMed、PPI、ogbn-arxiv等），并整理了对应的开源实现与基准。

**📈 对比分析**

通过对比已发表的Riemannian图学习方法（Hyperbolic, Spherical, CCS, Product/Quotient, Pseudo‑Riemannian, Grassmann, SPD, Generic等），展示在不同任务（节点分类、链路预测、图分类）中取得的性能提升，但整体缺乏统一实验评估，主要以已有文献为参照。

**⚠️ 局限性**

限制在于仅为综述，缺乏统一实验平台和基准；对不同曲率空间的理论与实现细节仍不完整；在大规模图上可扩展性、推理效率和迁移学习的实际效果仍待验证。

---

## 408. Towards Learning a Generalizable 3D Scene Representation from 2D Observations

**arXiv ID:** 2602.10943 | [PDF](https://arxiv.org/pdf/2602.10943v1)

**作者:** Martin Gromniak `[一作]` (ZAL Center of Applied Aeronautical Research), Stefan Wermter `[通讯]` (University of Hamburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种可泛化的神经辐射场（GNeRF）模型，能够从机器人头部或静态摄像机的二维图像预测工作空间的三维占据信息，并在全局坐标系下构建可直接用于抓取等机器人操作的3D场景表示。

**💡 创新点**

创新点在于：①在工作空间全局坐标系下构建占据概率而非相机坐标系；②使用多视角特征聚合的成本体积在世界坐标中进行几何恢复；③实现对未见场景和物体布置的无细调泛化；④仅通过自监督光度一致性即可推断遮挡区域的完整3D几何。

**🔧 技术方法**

采用了多视角特征提取（9 层轻量级卷积）、在世界坐标系中构建成本体积、3D U‑Net 进行几何恢复、MLP 输出密度和颜色、NeRF 渲染和 alpha 损失，并通过 MSE + beta 损失进行自监督训练。

**📊 数据集**

使用 60 个基于 YCB 物体的工作空间场景（20 个用于评估），机器人头部 15 个视角 + 3 个静态 RealSense 摄像机；通过 RealSense 深度数据近似完整的 3D 真实标注。

**📈 对比分析**

对比单视图、双视图（立体）和三视图设置；评估指标为平均绝对深度误差 MAE_depth 与 PSNR；在 40 场训练集、三视图下获得 MAE 0.026 m（≈26 mm）和 PSNR 23.64 dB；误差随视角多样性和训练场景数增加而下降。

**⚠️ 局限性**

局限性包括：仅提供单通道占据信息，缺乏语义分割；对动态或高度遮挡场景仍有限；需要精准相机标定和静态摄像机布置；泛化能力仍受限于训练场景的多样性，未能完全覆盖跨类别或更复杂环境的迁移。

---

## 409. Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation

**arXiv ID:** 2602.10489 | [PDF](https://arxiv.org/pdf/2602.10489v1)

**作者:** Wei Chen `[一作]` (Beihang University), Deqing wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种自适应分布对齐框架ADAlign，用于图域适应（GDA），无需人工指定对齐准则；

**💡 创新点**

创新点在于引入神经谱不一致度（NSD）—利用神经特征函数在频域上度量源‑目标分布差异，并通过可学习的频率采样器在minimax框架下自适应聚焦最具信息量的频谱成分；

**🔧 技术方法**

技术包括图神经网络编码、神经特征函数（Characteristic Function）与谱变换、可学习频率采样器、最小化源分类损失与最大化谱不一致度的对抗式（minimax）优化；

**📊 数据集**

实验使用10个公开图数据集（ArnetMiner、Airport、Blog、Twitch等），涵盖学术、社交、交通等四类域，共16个跨域转移任务；

**📈 对比分析**

与多种基线（GAT、GIN、GCN、DANE、UDAGCN、AdaGCN、StruRW、SA-GDA、GRADE、PairAlign、GraphAlign、A2GNN、TDSS、DGSDA、GAA、HGDA）对比，ADAlign在所有16个转移任务中均取得最高Micro‑F1分数，平均提升约6%，且在训练速度和显存占用上比最优竞争对手快约2×、占用内存少约3×；

**⚠️ 局限性**

局限性：仍需手动调节超参数（如κ、M、λ），对频率采样器的学习过程依赖于梯度估计；在极端高维或极小样本场景下的谱表示可能不充分，且当前框架主要关注节点分类任务，尚未扩展到更复杂的图任务或异构图场景。

---

## 410. The Anatomy of the Moltbook Social Graph

**arXiv ID:** 2602.10131 | [PDF](https://arxiv.org/pdf/2602.10131v1)

**作者:** David Holtz `[一作]` `[通讯]` (Columbia Business School), David Holtz (Columbia Business School)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对仅由 AI 代理构成的社交平台 Moltbook 进行初步描述性分析，考察其宏观网络结构与微观互动特征。

**💡 创新点**

首次系统评估 AI 代理社交平台的社会性，揭示其与人类社交网络在网络结构相似但互动深度与语言模式截然不同的差异。

**🔧 技术方法**

采用网络分析（度分布、聚类系数、回报率等）与文本处理（Zipf 分布、重复检测、关键词统计）等技术进行量化研究。

**📊 数据集**

使用 2026 年 1 月 27 日至 31 日 Moltbook 的完整抓取数据，包括 6 159 名活跃代理、13 875 篇帖子、115 031 条评论、4 532 个子社区。

**📈 对比分析**

通过与人类 Reddit 社区已知指标（回报率、线程深度、词频指数等）的对比，发现 Moltbook 的回报率仅 0.197、线程深度平均 1.07、复制率 34.1%，表明其社会性不足。

**⚠️ 局限性**

局限在于观察窗口短、API 限制（每帖 1 000 条评论）、无法区分人工或简单机器人产出、复制率高导致统计偏差以及缺乏对平台成熟后行为变化的跟踪。

---

## 411. Data Reductions for the Strong Maximum Independent Set Problem in Hypergraphs

**arXiv ID:** 2602.10781 | [PDF](https://arxiv.org/pdf/2602.10781v1)

**作者:** Ernestine Großmann `[一作]` (Heidelberg University), Antonie Wagner `[通讯]` (Heidelberg University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对超图最大独立集问题，提出并实现了九条新的数据约简规则，形成了预处理流程；

**💡 创新点**

创新点在于专门为超图强独立集设计的约简规则，能够显著压缩实例规模并显著提升后续求解器性能；

**🔧 技术方法**

采用图论与整数规划方法，结合超图强独立性的定义，构建约简技术并与多种求解器集成；

**📊 数据集**

实验使用公开的超图数据集（涵盖多种规模和结构的实例），未具体列出数据集名称；

**📈 对比分析**

与多种后续求解器对比，预处理后平均实例缩减至原来的22%，运行时间平均提升3.84倍，最高可达53倍，且有实例由不可解变为可解；

**⚠️ 局限性**

局限性包括：约简规则在某些极大规模实例的效果有限，对特定超图结构的约简效果不显著；未对更广泛的求解器和更大规模实例进行评估。

---

## 412. The Sample Complexity of Uniform Approximation for Multi-Dimensional CDFs and Fixed-Price Mechanisms

**arXiv ID:** 2602.10868 | [PDF](https://arxiv.org/pdf/2602.10868v1)

**作者:** Matteo Castiglioni `[一作]` (Politecnico di Milano), Alberto Marchesi `[通讯]` (Politecnico di Milano)

**通讯引用:** 358 | [OpenAlex ID](https://openalex.org/A5039843107)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究在一位一比特反馈下学习多维CDF的样本复杂度及其在小市场中的应用

**💡 创新点**

证明样本复杂度几乎与维度无关，仅受log(K/δ)影响，并给出固定价格机制的样本复杂度和下界

**🔧 技术方法**

使用自适应递归分解、代表性超矩形族、稀疏性分析和Monte Carlo估计等技术

**📊 数据集**

未使用公开数据集，而是理论分析

**📈 对比分析**

与多变量DKW不等式和单维需求曲线学习的下界相比，取得了接近最优的样本复杂度；在小市场中获得了新的收益最大化样本复杂度和log‑T^{3/4}的后悔界

**⚠️ 局限性**

样本复杂度中对K的对数依赖仍较大；后悔率仍不匹配单维下界，且对任意维度的下界仍不完整

---

## 413. EVOKE: Emotion Vocabulary Of Korean and English

**arXiv ID:** 2602.10414 | [PDF](https://arxiv.org/pdf/2602.10414v1)

**作者:** Yoonwon Jung `[一作]` (University of California San Diego), Benjamin K. Bergen `[通讯]` (University of California San Diego)

**通讯引用:** 4709 | [OpenAlex ID](https://openalex.org/A5043344696)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了一个名为EVOKE的韩英情感词平行语料库，收集1427韩语词与1399英语词，并对其进行多维度（可接受性、经验、排除、多义性）注释，生成多对多翻译映射。

**💡 创新点**

提出了理论中立、可定制的情感词注释框架，首次将可接受性、语义经验、排除标准与多义性/隐喻识别整合到同一数据集中，并支持跨语言比较。

**🔧 技术方法**

采用双语专家翻译、基于句式的可接受性判断、经验与因果性评估、排除标准筛选，以及概念隐喻识别协议（MIP）进行多义性标注，并使用Cohen κ进行一致性评估。

**📊 数据集**

使用先前韩英情感词研究、词典（韩英双语词典、英韩词典）与单语词典构建词表，最终得到1427韩语词、1399英语词及其多对多映射。

**📈 对比分析**

通过比较四维度标注分布与交叉语言差异，计算Cohen κ平均值0.74（韩语）与0.60（英语），表明标注一致性良好；在可接受性判定中acpt4得分最高，体现对情感词的系统评估。

**⚠️ 局限性**

局限在于仅覆盖韩英两语，标注者人数有限且多义性标注样本少，主要聚焦形容词和动词，未覆盖名词和其他语言，且可能存在主观性与文化偏差。

---

## 414. Single-Turn LLM Reformulation Powered Multi-Stage Hybrid Re-Ranking for Tip-of-the-Tongue Known-Item Retrieval

**arXiv ID:** 2602.10321 | [PDF](https://arxiv.org/pdf/2602.10321v1)

**作者:** Debayan Mukhopadhyay `[一作]` (Independent Researcher), Shubham Chatterjee `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 151 | [OpenAlex ID](https://openalex.org/A5004223654)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对模糊、片段或错误的 ToT 查询进行一次性 LLM 重新表述与受控扩展，然后按多阶段检索管线（BM25 → 复合编码器 → 交叉编码器 → 72B LLM 列表排序）检索，显著提升召回与排名指标。

**💡 创新点**

创新点在于：①使用通用 8B LLM（无微调、无域适配）通过精心设计的提示实现查询重写与受控扩展；②将该轻量化预检索步骤嵌入完整多阶段检索体系，突破 ToT 查询语义与检索索引间的鸿沟；③证明单次 LLM 介入即可获得可观收益，无需复杂的多轮推理或领域知识库。

**🔧 技术方法**

技术包括：大规模语言模型（8B 参数 LLM）提示式查询重写、受控扩展；多阶段检索管线：BM25（稀疏检索）、Contriever、E5‑large‑v2、ColBERTv2（双编码器/后交互）、monoT5（交叉编码器重排）、Qwen 2.5 Instruct 72B（4‑bit 量化）最终列表重排；实验评测使用标准召回/排序指标。

**📊 数据集**

主要使用 TREC‑ToT 2025 公开基准（涵盖电影、地标、名人等多域开放式查询），并在 GitHub 上公开代码与数据以实现复现。

**📈 对比分析**

对比方法：将原始 ToT 查询直接输入完整检索管线与先通过 LLM 预处理后再输入管线进行评估。实验显示：召回提升 20.61%；nDCG@10 提升 33.88%；MRR 提升 29.92%；MAP@10 提升 29.98%。表明单次预检索改写能显著解锁后续检索与重排的潜能。

**⚠️ 局限性**

限制：①仅采用一次性重写，未结合检索反馈进行迭代细化；②仅针对文本查询，未探究多模态 ToT 场景；③实验仅在英语数据上验证，跨语言/更大模型效果未知；④受控扩展不具自适应实体链接，未来需兼顾 KB 触发与纯查询内重写。

---

## 415. Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away

**arXiv ID:** 2602.11096 | [PDF](https://arxiv.org/pdf/2602.11096v1)

**作者:** Soumya Suvra Ghosal `[一作]`, Amrit Singh Bedi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SafeThink，一种在推理时监测安全奖励并在安全阈值被破坏时插入短提示“Wait, think safely”的轻量级防御；

**💡 创新点**

创新点在于：①将安全对齐视为满足阈值的约束（satisficing）而非最大化；②采用早期步骤干预，仅需前1–3步即可恢复安全；③通过离线生成并挑选最优安全提示词，减少运行时搜索。

**🔧 技术方法**

技术包括：安全奖励模型评估中间推理链、基于KL散度的安全提示词选择、蒙特卡罗估计安全成功率、条件分布约束下的最小化干预。

**📊 数据集**

数据集：JailbreakV‑28K（文本+图像）、Hades、FigStep、MM‑SafetyBench（四类图像攻击）以及MathVista（数学推理）。

**📈 对比分析**

与ZeroThink、LessThink、ZS‑SafePath、AdaShield等基线对比，SafeThink在六款开源MLRM上将攻击成功率从约63%降至5–6%，同时MathVista准确率保持≈65%（差异≤0.2%），推理时延仅增加0.1–0.9秒，显著优于现有方法。

**⚠️ 局限性**

局限性包括：依赖安全奖励模型的准确性；仅针对已知攻击模式的早期干预；离线提示词集可能不覆盖所有恶意情景；对极端或新型攻击的鲁棒性尚未充分验证。

---

## 416. Data-Efficient Hierarchical Goal-Conditioned Reinforcement Learning via Normalizing Flows

**arXiv ID:** 2602.11142 | [PDF](https://arxiv.org/pdf/2602.11142v1)

**作者:** Shaswat Garg `[一作]` (ArenaX Labs), Brandon Da Silva `[通讯]` (ArenaX Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 NF‑HIQL，一种利用 Normalizing Flow 取代高低层高斯策略的分层目标导向强化学习框架；

**💡 创新点**

创新点在于将可逆流（RealNVP）嵌入两级策略，获得多模态表达能力、可解析对数似然与熵，并给出 KL 与 PAC‑style 采样效率理论保证；

**🔧 技术方法**

技术包括 Hierarchical Implicit Q‑Learning、Normalizing Flow（RealNVP）、优势加权最大似然（AWR）、离线数据训练与 KL 限制；

**📊 数据集**

在 OGBench（antmaze‑medium‑navigate、antsoccer‑medium‑navigate、antsoccer‑arena‑navigate、cube‑single‑play、scene‑play）以及真实机器人 myCobot‑280 的离线采样数据上进行评估；

**📈 对比分析**

与 GCIQL、CRL、HIQL、BESO 等基线对比，NF‑HIQL 在 100% 与 50% 数据量下均表现最佳，成功率在各任务上比同类方法提升 10–70%，在数据稀缺场景下仍保持高性能；

**⚠️ 局限性**

局限性包括对连续状态空间的依赖、对 RealNVP 架构参数敏感、在极低成功率的最难任务上仍有不足，以及缺乏对视觉输入和实时规划的扩展。

---

## 417. Equivariant Evidential Deep Learning for Interatomic Potentials

**arXiv ID:** 2602.10419 | [PDF](https://arxiv.org/pdf/2602.10419v1)

**作者:** Zhongyao Wang `[一作]` (Fudan University), Mao Su `[通讯]` (Shenzhen Institute of Advanced Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了一种单前向推理的等变式证据深度学习框架 e^2IP，用于在原子力场预测中联合估计力的均值和完整 3×3 SPD 协方差，从而实现可解释的量化不确定性。

**💡 创新点**

创新点在于把等变式对称性与证据学习结合：通过 Lie 代数到 SPD 矩阵的指数映射构造旋转一致的全协方差，并加入谱稳定化与证据正则化以提升数值鲁棒性。

**🔧 技术方法**

使用了 SE(3)-等变图神经网络骨干（如 NequIP、MACE 等）、Normal–Inverse‑Wishart 先验、Student‑t 预测分布、矩阵指数、对称张量表示、谱衰减、softplus 约束等技术。

**📊 数据集**

在多种分子动力学基准上评估，包括 ab initio 液态水、MD22（Buckyball‑Catcher、DWCT）、Silica 玻璃（OOD）等数据集。

**📈 对比分析**

与五模型集成、非等变式证据基线 eIP 以及不同骨干迁移进行对比；e^2IP 在单模型推理下实现与集成相近或更优的误差、低 NLL、优良校准（CE/PIT），并且速度提升约 4.5 倍，样本效率、旋转一致性和跨骨干迁移性能最佳。

**⚠️ 局限性**

主要局限在于仅验证 SO(3) 等变性，未覆盖反射或极端非平衡情况；NIW 先验对重尾残差可能不够；需要更多实验验证在主动学习或长时序 MD 中的实际收益。

---

## 418. From Buffers to Registers: Unlocking Fine-Grained FlashAttention with Hybrid-Bonded 3D NPU Co-Design

**arXiv ID:** 2602.11016 | [PDF](https://arxiv.org/pdf/2602.11016v1)

**作者:** Jinxin Yu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Ying Wang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 3D-Flow hybrid‑bonded 3D‑staked spatial 加速器及其配套的 3D‑FlashAttention 细粒度调度方案，解决 Transformer 关注层在 2D 体系中由于 SRAM 访问成为能耗瓶颈的问题。

**💡 创新点**

核心创新包括：① 通过 sub‑10 µm TSV 进行 hybrid bonding，实现垂直 PE 层之间的 register‑to‑register 通信；② 将关注层的各子操作 QK^T、Softmax、PV 等映射到不同垂直层，实现 cycle‑level 细粒度流水线；③ 结合硬件与算法的 co‑design，形成无瓶颈的垂直数据流，显著降低 SRAM 访问。

**🔧 技术方法**

技术手段包括 3D‑Flow 体系结构、hybrid bonding、TSV 链、细粒度调度、FlashAttention 算法融合、阵列化计算（MAC、比较、指数运算）、能耗建模与仿真。

**📊 数据集**

使用 Transformer 模型 OPT（多头关注）与 QWEN（分组查询关注），对序列长度从 1K 扩展到 64K（通过动态 RoPE 缩放），作为评测基准。

**📈 对比分析**

与 2D‑Unfused、2D‑Fused（FuseMax、Dual‑SA）以及 3D‑Base 等基线进行能耗与速度比较。实验表明能耗降低 46%–93%，速度提升 1.4×–7.6×，PE 利用率平均 87%，整体能耗降低 32.7%–64.2%。

**⚠️ 局限性**

局限性：3D‑IC 及 TSV 互连带来的 5–10% 能耗与寄存器访问开销；设计复杂度与 3D 包装成本高；当前方案主要针对关注层，其他算子尚未充分验证。

---

## 419. Learning Page Order in Shuffled WOO Releases

**arXiv ID:** 2602.11040 | [PDF](https://arxiv.org/pdf/2602.11040v1)

**作者:** Efe Kahraman `[一作]` (utf), Giulio Tosato `[通讯]` (utf)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对5,461份荷兰公共信息发布（WOO）文档进行页面重排序实验，比较多种模型的排序性能。

**💡 创新点**

提出长度专属的pairwise ranking transformer，并系统分析了seq2seq transformer在长文档上失效以及课程学习失败的根源。

**🔧 技术方法**

采用text-embedding-3-large提取文本嵌入，实验指针网络、seq2seq transformer（含不同位置编码）、pairwise ranking transformer，进一步做位置编码消融、专属模型与课程学习训练。

**📊 数据集**

使用涵盖2–25页的5,461份WOO文档的完整数据集，训练集70%，验证15%，测试15%。

**📈 对比分析**

通过Kendall's τ评估排序质量；专属pairwise模型在15页文档上取得τ=0.722，短文档0.953；seq2seq在长文档仅τ=0.014；指针网络及传统启发式方法性能远低于此。

**⚠️ 局限性**

仅使用文本嵌入，忽略视觉信息；长文档样本不足；seq2seq失效原因尚未完全阐明；所有实验仅基于单一embedding模型。

---

## 420. Conversational Behavior Modeling Foundation Model With Multi-Level Perception

**arXiv ID:** 2602.11065 | [PDF](https://arxiv.org/pdf/2602.11065v1)

**作者:** Dingkun Zhou `[一作]` (University of California), Gopala Anumanchipalli `[通讯]` (University of California)

**通讯引用:** 2738 | [OpenAlex ID](https://openalex.org/A5068922218)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个严格因果、流式、全双工对话框架，利用多级感知模型对每秒的高低级言语行为进行预测，并通过Graph-of-Thought (GoT) 进行因果推理，实时生成可解释的自然语言理由。

**💡 创新点**

创新点在于：① 将人类对话的意图→行动链条建模为多层次感知+图结构推理；② 设计可解释的“证据链”选择器和轻量级GoT推理器，实现毫秒级低延迟解释；③ 通过自监督的“对话GoT-120h”合成语料，提供高质量的层级行为标签和理由。

**🔧 技术方法**

核心技术包括：多层Transformer（用于感知和推理）、FiLM调制实现高低层次协同；Graph Transformer用于证据链筛选；T5解码器生成理由；自监督训练与跨任务迁移；混合精度训练与梯度裁剪等。

**📊 数据集**

使用的主要数据集是自研的 ConversationGoT-120h（120小时合成对话，带有高低级言语行为标签和理由），并在真实对话数据 Candor 上做 OOD 验证。

**📈 对比分析**

实验表明：在 ConversationGoT-120h 上，语义行为识别 F1 ≥0.73（低层）/0.52（高层），AUC ≥0.97；在 Candor 上迁移性能下降不大；GoT 的解释质量在四个维度上均优于随机证据链、GPT‑4o、GPT‑5（思考）等基线，同时推理延迟仅 0.74 s，显著低于 GPT‑4o（≈3 s）与 GPT‑5（≈17 s）。

**⚠️ 局限性**

局限性包括：① 合成数据可能存在分布偏移，导致在极端真实对话中的鲁棒性下降；② 低层行为识别对极短事件敏感，仍需改进；③ 依赖 GPT‑4o/5 作为教师，若教师错误会影响解释质量；④ 对资源要求较高（需要多模态流式处理）。

---

## 421. Don't blame me: How Intelligent Support Affects Moral Responsibility in Human Oversight

**arXiv ID:** 2602.10701 | [PDF](https://arxiv.org/pdf/2602.10701v1)

**作者:** Cedric Faas `[一作]`, Anna Maria Feit `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

介绍并说明CEUR-WS 会议论文模板的使用方法及其主要功能，阐述模板如何实现统一排版与元数据提取。

**💡 创新点**

提出模板可通过参数灵活调整（如双栏、页眉页脚、标题变体）并内置无障碍与可访问性支持，提供更高效的出版流程。

**🔧 技术方法**

基于 LaTeX 及其标准宏包（如 booktabs、hyperref 等）实现排版、表格、公式和图形的高质量呈现。

**📊 数据集**

无使用数据集，示例中仅展示模板代码与排版效果。

**📈 对比分析**

未进行实验性对比，主要通过示例展示模板功能与使用体验。

**⚠️ 局限性**

仅适用于 CEUR-WS 会议出版物，且不允许修改模板核心参数，限制了对特殊需求的自定义空间。

---

## 422. General Flexible $f$-divergence for Challenging Offline RL Datasets with Low Stochasticity and Diverse Behavior Policies

**arXiv ID:** 2602.11087 | [PDF](https://arxiv.org/pdf/2602.11087v1)

**作者:** Jianxun Wang `[一作]` (North Carolina State University), David L. Roberts `[通讯]` (North Carolina State University)

**通讯引用:** 2466 | [OpenAlex ID](https://openalex.org/A5012351691)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究离线强化学习中数据集多样性与探索不足的问题，提出通用线性规划形式并引入灵活的f-散度约束，构造 Flex-f-Q 与 Flex-f-DICE 两种算法。

**💡 创新点**

创新点在于：① 将RL的Bellman误差最小化与线性规划形式统一，去除传统的ζ≥0约束；② 设计可调的“灵活f-散度”函数，能够根据数据的混合度与行为策略的随机性动态调整约束强度。

**🔧 技术方法**

采用线性规划与Fenchel对偶理论、f-散度（如χ²、KL、Le‑Cam等）与其共轭函数、期望值回归、行为克隆估计α±、β的启发式估计，以及半梯度下降等技术。

**📊 数据集**

实验使用D4RL基准环境（MuJoCo Hopper‑v4、Walker2d‑v4、Ant‑v4、HalfCheetah‑v4；Fetch Push‑v2、PickAndPlace‑v2；ArdoitHand Pen‑v1、Hammer‑v1）以及作者自行收集的混合行为策略数据集（2‑p、4‑p、10‑p）。

**📈 对比分析**

与IQL、OptiDICE、TD3BC、CQL等基线在相同数据集上对比，Flex‑f‑Q与IQL相近甚至略优，Flex‑f‑DICE往往明显优于OptiDICE，平均提升5–15%归一化回报。

**⚠️ 局限性**

局限性包括：需要手工估计α±、β并对其稳定性敏感；不同环境与算法组合下最佳参数不统一；对高维连续动作空间的可扩展性仍有限；未实现完全自动化的参数优化过程。

---

## 423. Configuration-to-Performance Scaling Law with Neural Ansatz

**arXiv ID:** 2602.10300 | [PDF](https://arxiv.org/pdf/2602.10300v1)

**作者:** Huaqing Zhang `[一作]` (Tsinghua University), Tengyu Ma `[通讯]` (Stanford University)

**通讯引用:** 9168 | [OpenAlex ID](https://openalex.org/A5101821970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一种能够将完整训练配置映射到预训练性能的预测模型（NCPL）。

**💡 创新点**

创新点在于使用大语言模型作为通用回归器，直接学习配置到性能的非参数映射，能够同时预测最终损失、损失曲线以及多超参数的联合优化。

**🔧 技术方法**

采用 Qwen3‑1.7B 预训练语言模型进行两阶段微调，并以相对 Chinchilla 基线的残差为训练目标。

**📊 数据集**

使用 Marin 与 StepLaw 两个公开预训练日志数据集（约 5,000 条记录）。

**📈 对比分析**

与传统 Chinchilla 规模法和 XGBoost 等基线比较，NCPL 在 ID 与 OOD 上均显著降低 MAE/RMSE、提升 Spearman 相关性，误差比 Chinchilla 低 20–40%。

**⚠️ 局限性**

局限性包括仅覆盖 430M 参数以内的模型、缺乏 MoE、线性注意力等新架构，超参数离散化导致对未见取值预测不可靠，需要更多开放日志以进一步提升。

---

## 424. The Neurosymbolic Frontier of Nonuniform Ellipticity: Formalizing Sharp Schauder Theory via Topos-Theoretic Reasoning Models

**arXiv ID:** 2602.10632 | [PDF](https://arxiv.org/pdf/2602.10632v1)

**作者:** Suyash Mishra `[一作]` `[通讯]`, Suyash Mishra

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对非均匀椭圆正则性理论的最新突破和神经符号大型推理模型（LRMs）领域进行了关键综合，探讨了Schauder理论中的长期尖锐增长率猜想的解决方案。

**💡 创新点**

创新点在于提出了“幽灵方程”方法，这是一种复杂的辅助推导，能够绕过经典Euler-Lagrange系统的非可微性，并将纯分析构造与基于拓扑理论的LRMs相结合。

**🔧 技术方法**

使用了幽灵方程方法、Besov空间技术和分数版本的Moser迭代等数学技术。

**📊 数据集**

未具体提及使用的数据集，但讨论了与物理系统相关的复杂多相问题。

**📈 对比分析**

通过与传统的Schauder理论进行比较，证明了在特定条件下，梯度的Hölder连续性得以保持，且在非均匀情况下的正则性结果得到了验证。

**⚠️ 局限性**

限制在于幽灵方程的构造可能无法适用于所有非均匀椭圆问题，且LRMs的推理过程仍需进一步验证和完善。

---

## 425. ICA: Information-Aware Credit Assignment for Visually Grounded Long-Horizon Information-Seeking Agents

**arXiv ID:** 2602.10863 | [PDF](https://arxiv.org/pdf/2602.10863v1)

**作者:** Cong Pang `[一作]` (ShanghaiTech University), Xin Lou `[通讯]` (SenseTime Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了视觉原生网页快照与信息感知信用分配（ICA）框架，用于提升长周期信息检索代理在开放网络环境中的学习效率。

**💡 创新点**

创新点在于（1）使用网页快照保持布局和多模态语义，避免文本解析噪声；（2）引入后置信息层信用分配，将稀疏终点奖励转化为密集转弯奖励，从而缓解信用分配瓶颈。

**🔧 技术方法**

采用了视觉快照感知、信息感知信用分配（ICA）、GRPO强化学习、LLM-as-Judge评判、Playwright抓取网页、Serper搜索工具以及SFT+RL联合训练策略。

**📊 数据集**

使用了BrowseComp、GAIA、Xbench-DS、Seal-0等长周期信息检索基准，以及自构造的复杂报告数据集。

**📈 对比分析**

与文本RAG基线、公开代理以及专有系统进行对比，ICA在多项基准上均取得显著提升；30B模型在BrowseComp和Seal-0等任务上与专有系统相近或更优。

**⚠️ 局限性**

局限性包括：仍依赖网络渲染和图像尺寸限制；ICA为后置方法，无法在实时交互中即时调整信用；在极短轨迹或非视觉内容任务中效果有限。

---

## 426. Consistency Meets Verification: Enhancing Test Generation Quality in Large Language Models Without Ground-Truth Solutions

**arXiv ID:** 2602.10522 | [PDF](https://arxiv.org/pdf/2602.10522v1)

**作者:** Hamed Taherkhani `[一作]` (York University), Hadi Hemmati `[通讯]` (York University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ConVerTest 两阶段管线，利用一致性驱动的生成与共识验证，生成无需基准实现的高质量 LLM 测试用例。

**💡 创新点**

创新点在于将 Self-Consistency、Chain-of-Verification 与 Dual Execution Agreement 三种一致性/验证技术融合为完整流程，既在生成阶段抑制幻觉，又在后置阶段以共识判定测试有效性，消除对代码基准的依赖。

**🔧 技术方法**

技术手段包括：多样本 Self-Consistency 采样+多数投票、CoVe 迭代自检与改写、Dual Execution Agreement 基于代码-测试共识得分筛选、AST 分析统一测试逻辑、LLM 生成与提示工程。

**📊 数据集**

使用了 BigCodeBench-Hard 与 Less Basic Python Problems（LBPP）两大 Python 代码生成基准。

**📈 对比分析**

通过与全局生成（Holistic）和两阶段但单样本（TSTG）基线对比，使用 Validity Rate、Line Coverage、Mutation Score、Precision/Recall/F1 等指标。ConVerTest 在 VR 最高可提升 39%，LC 最高提升 28%，MS 最高提升 18%，并保持高精度/召回率，证明三项技术协同显著提升测试质量。

**⚠️ 局限性**

局限性包括：对采样数量、CoVe 迭代次数等超参敏感；共识假设不一定对应正确性；仅在 Python 上验证，其他语言与大型项目集成仍待评估；多阶段采样与执行带来计算成本。

---

## 427. Time-to-Event Transformer to Capture Timing Attention of Events in EHR Time Series

**arXiv ID:** 2602.10385 | [PDF](https://arxiv.org/pdf/2602.10385v1)

**作者:** Jia Li `[一作]` (University of Minnesota), Rui Zhang `[通讯]` (University of Minnesota)

**通讯引用:** 11836 | [OpenAlex ID](https://openalex.org/A5100675481)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种名为LITT（Level-of-Individual Timing Transformer）的时序模型，用于在电子健康记录（EHR）中实现事件时间的预测和个性化序列模式发现；

**💡 创新点**

其创新点在于把时间视为可计算维度，引入相对时间戳、时间对齐与“时间注意力”，并在LSTM框架中设计专门的时间变换门，实现对事件时序的可解释性和可预测性；

**🔧 技术方法**

技术上主要采用LSTM循环网络、时间变换门（Time‑Transformation Gate）、相对时间计算与基于峰度的时间注意力机制，并与传统RNN、Cox比例风险模型、随机生存森林、DeepSurv、DeepHit等深度生存模型做对比；

**📊 数据集**

使用了真实世界的乳腺癌患者EHR数据（3,276例）用于心脏毒性预测，并在公共支持数据集SUPPORT和METABRIC上进行生存分析验证；

**📈 对比分析**

与LSTM/GRU、CPH、RSF、DeepSurv、DeepHit、SurvTrace等方法对比，LITT在事件时间回归任务中RMSE显著下降，在公共数据集的C‑index上表现竞争性，部分场景甚至优于现有深度生存模型；

**⚠️ 局限性**

局限性包括：LITT并非专门为生存分析设计，需手动转换为C‑index；对个体化时间特征依赖较大，往往需要额外临床信息；在不同疾病或更复杂时序场景下的泛化性尚待进一步验证。

---

## 428. Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs

**arXiv ID:** 2602.10377 | [PDF](https://arxiv.org/pdf/2602.10377v1)

**作者:** Luoyang Sun `[一作]` (AI Lab, Yangtze River Delta), Cheng Deng `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在边缘设备上部署大型语言模型时的硬件‑软件协同设计，提出硬件协同缩放法则，将模型损失与硬件延迟联合建模，并通过经验损失缩放、Roofline延迟预测与Pareto前沿构造，快速定位最优架构。

**💡 创新点**

创新点在于：①将Transformer的损失缩放规律与硬件Roofline延迟模型融合，形成可直接推断最优loss‑latency Pareto前沿的硬件协同缩放法则；②针对MoE稀疏度、宽度、深度、FFN比例、GQA等关键超参给出理论闭式解，揭示不同硬件约束下的最优稀疏度与宽度比例；③提供一次性NAS+延迟预测+Pareto前沿的完整流程，可在新硬件上仅通过几次小规模训练即可快速得到最优设计。

**🔧 技术方法**

使用的技术包括：多项式经验损失缩放模型（拟合10B token训练结果），Roofline分析的延迟预测（基于FLOPs与内存带宽），一次性NAS（Latin hypercube采样+迭代局部搜索），Pareto前沿构造与可视化，INT8/FP16量化实验，vLLM推理引擎，NVIDIA Jetson Orin边缘平台实验。

**📊 数据集**

训练数据：10B token混合一般语料、数学推理与代码数据；验证数据：约1B token验证集；测试数据：WikiText‑2测试集，用于计算困惑度（perplexity）评估模型泛化。

**📈 对比分析**

与生产模型Qwen2.5‑0.5B进行对比：在相同推理延迟下，硬件协同设计的模型在训练损失更低、WikiText‑2 perplexity下降19.42%（从63.14降至49.22），推理速度相同；INT8量化提升约10%–15%（但实际加速低于理论2×），验证了量化在非线性运算与精度转换开销下的子线性加速。

**⚠️ 局限性**

局限性包括：①经验损失缩放模型基于10B token训练，可能不适用于更大/不同训练量或数据分布；②延迟模型假设理想的Roofline，忽略核启动、缓存、操作融合等实际系统开销，导致10–20%误差；③仅覆盖标准Transformer与MoE结构，未考虑SSM、线性注意等新型架构；④新硬件预测需手动测定峰值计算与带宽，若硬件特性与假设不符可能影响预测精度。

---

## 429. Neural Additive Experts: Context-Gated Experts for Controllable Model Additivity

**arXiv ID:** 2602.10585 | [PDF](https://arxiv.org/pdf/2602.10585v1)

**作者:** Guangzhi Xiong `[一作]` (University of Virginia), Aidong Zhang `[通讯]` (University of Virginia)

**通讯引用:** 11851 | [OpenAlex ID](https://openalex.org/A5013588572)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 Neural Additive Experts (NAE) 模型，用混合专家架构和动态门控来在保持可解释性的同时捕捉特征交互。

**💡 创新点**

创新点在于：① 将每个特征关联多个专家网络并通过上下文感知门控灵活组合，② 引入专家方差正则化以调节模型从严格可加到高度灵活的连续取舍，③ 提供理论证明 NAEs 在表达能力上优于传统 GAM 与 GA^2M，同时保留特征级解释。

**🔧 技术方法**

核心技术包括：多专家网络（per-feature expert），编码器将原始特征映射到低维潜在空间，动态门控（softmax gating）依赖所有特征上下文，专家方差正则化（variance penalty）控制可加性，损失函数结合任务损失与正则化。

**📊 数据集**

使用了合成数据（单峰与多峰分布）以及六个真实数据集：Housing、MIMIC‑II、MIMIC‑III、Income、Credit 和 Year，涵盖回归与分类任务，包含连续与分类特征。

**📈 对比分析**

通过与多种基线（线性、Spline、NAM、EBM、NODE‑GAM、NBM、黑盒 MLP、XGBoost 等）进行对比。NAE 在保持 O(n) 解释复杂度的前提下，预测准确度与最强黑盒模型相近或更优；同时提供可调的可加性度量和上下界可视化。

**⚠️ 局限性**

局限性包括：① 需要手动调节正则化强度 λ 以平衡可加性与性能；② 门控机制与专家数量增加时计算开销上升；③ 对极高维特征空间的扩展与稀疏性处理仍待进一步研究。

---

## 430. APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots

**arXiv ID:** 2602.11143 | [PDF](https://arxiv.org/pdf/2602.11143v1)

**作者:** Yikai Wang `[一作]` (Carnegie Mellon University), Ding Zhao `[通讯]` (Carnegie Mellon University)

**通讯引用:** 5556 | [OpenAlex ID](https://openalex.org/A5037644321)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

训练六个与地形相关的技能（四个全身动作和两个循环行走）并将其融合为单一感知策略，实现机器人在高平台上的连续跨越。

**💡 创新点**

提出通用 ratchet progress 奖励以引导多接触目标达成；双阶段学习+蒸馏整合多技能；在 LiDAR 高程映射中加入仿真失真模拟与实时后处理，最终实现零转移在 0.8 m 高平台上的成功跨越。

**🔧 技术方法**

采用深度强化学习、LiDAR 高程映射、域随机化、行为克隆+DAgger 蒸馏、目标进度奖励和全身多接触控制技术。

**📊 数据集**

在仿真中随机生成平台高度、角度、起始姿态等多种环境；使用 Unitree G1 29-DoF 机器人收集 LiDAR 点云进行实测验证。

**📈 对比分析**

与多种基准奖励（速度追踪、距离最小化、RND 等）对比，使用成功率、最大接触力等指标评估；仿真中成功率 95.4%，实物零转移成功率 99% 以上，最大接触力保持在安全阈值以内。

**⚠️ 局限性**

仅针对相对平坦的高平台，软/高摩擦表面和极端环境的泛化仍需验证；奖励权重调参依赖经验；在更复杂多关节环境中的表现尚待进一步研究。

---

## 431. Enhancing Underwater Images via Adaptive Semantic-aware Codebook Learning

**arXiv ID:** 2602.10586 | [PDF](https://arxiv.org/pdf/2602.10586v1)

**作者:** Bosen Lin `[一作]` (Ocean University of China), Qian Du `[通讯]` (Mississippi State University)

**通讯引用:** 40588 | [OpenAlex ID](https://openalex.org/A5033017179)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于语义感知离散码书的水下图像增强网络 SUCode，采用三阶段自监督训练实现对空间异质退化的自适应增强。

**💡 创新点**

创新点在于：①像素级语义引导的码书学习，生成针对不同语义区域的离散表示；②自监督三阶段训练避免伪真值污染；③结合门控通道注意（GCAM）与频域特征融合（FAFF）提升色彩恢复与细节重建。

**🔧 技术方法**

主要技术包括 VQ‑VAE/VQGAN 码书学习、Swin Transformer 权重预测器、GCAM、FAFF、GAN 对抗损失、感知损失以及多尺度残差网络。

**📊 数据集**

使用 SUIM‑E、UIEB 训练集进行训练；在 SUIM‑E、UIEB 以及跨数据集 LSUI、UFO120 进行评估；语义分割标签来自 SUIM 数据集。

**📈 对比分析**

与传统物理模型、现有深度学习方法及其他码书方法进行全参考（PSNR、SSIM、LPIPS）和无参考（UCIQE、UIQM）指标比较，SUCode 在 PSNR 23.9 dB、SSIM 0.925、LPIPS 0.124 处领先 SOTA，并在无参考指标上同样表现优异；跨数据集测试显示良好的泛化能力。

**⚠️ 局限性**

局限性包括：对极端颜色衰减或极低对比度图像恢复有限；在高亮与阴影对比差异大的场景下难以同时保留细节；对语义分割错误（类别交换、缺失区域）鲁棒性待提升。

---

## 432. FastUSP: A Multi-Level Collaborative Acceleration Framework for Distributed Diffusion Model Inference

**arXiv ID:** 2602.10940 | [PDF](https://arxiv.org/pdf/2602.10940v1)

**作者:** Guandong Li `[一作]` `[通讯]` (iFLYTEK), Guandong Li (iFLYTEK)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了FastUSP框架，优化多GPU扩散模型推理性能。

**💡 创新点**

创新点在于多级优化（编译级、通信级、算子级）与CUDA Graph结合，显著降低内核调用开销并提升推理速度。

**🔧 技术方法**

使用的技术包括CUDA Graph编译、FP8量化全体通信、双缓冲流水线Ring注意力以及PyTorch Inductor重排。

**📊 数据集**

在FLUX（12B）和Qwen-Image模型上进行评估，使用RTX 5090 GPU集群。

**📈 对比分析**

对比基线USP，FastUSP在2-8 GPU上分别实现1.12×-1.16×的端到端加速，Qwen-Image 2 GPU提升1.09×。

**⚠️ 局限性**

主要限制是PyTorch Inductor对Ring注意力的兼容性不足，导致4+ GPU时编译优化失效。

---

## 433. When Less Is More? Diagnosing ASR Predictions in Sardinian via Layer-Wise Decoding

**arXiv ID:** 2602.10350 | [PDF](https://arxiv.org/pdf/2602.10350v1)

**作者:** Domenico De Cristofaro `[一作]` (Free University of Bozen-Bolzano), Aleese Block `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

使用层级截断策略探究多语言ASR模型在低资源语言Campidanese Sardinian上的音素级解码表现

**💡 创新点**

提出“回归错误”概念，发现中间层解码在音素识别上往往优于最终层，并揭示深层会对已正确音素产生错误

**🔧 技术方法**

基于Wav2Vec2预训练模型，应用Logit Lens式层级探测与贪婪CTC解码

**📊 数据集**

Campidanese Sardinian短语音数据集（48句，4位说话人）

**📈 对比分析**

对比不同层级（从第24层到第19层）解码的PER，发现移除两层后PER最低（35.4%），比完整模型低约1%

**⚠️ 局限性**

仅在极低资源环境下验证，未测试高资源语言，且忽略音素时序对齐与语境因素

---

## 434. Supercharging Packet-level Network Simulation of Large Model Training via Memoization and Fast-Forwarding

**arXiv ID:** 2602.10615 | [PDF](https://arxiv.org/pdf/2602.10615v1)

**作者:** Fei Long `[一作]` (Tsinghua University), Bingyang Liu `[通讯]` (Huawei Technologies Co)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种用户透明的包级离散事件网络模拟引擎，利用记忆化和快进技术加速LLM训练网络模拟；

**💡 创新点**

创新点在于通过网络分区、流冲突图记忆化和基于发送速率的稳态识别，自动跳过重复争用和稳态期间的冗余事件，保持高保真度；

**🔧 技术方法**

核心技术包括：端口级网络分区算法、流冲突图（FCG）抽象与图同构匹配、发送速率阈值稳态检测、事件时间戳偏移与包暂停、与Unison等多线程DES的无缝协同；

**📊 数据集**

实验基准为GPT系列（13B、175B等）与MoE模型在不同规模GPU集群（64/128/256/1024）上的训练迭代，使用Rail-Optimized Fat-tree拓扑及真实GPT-18B训练日志；

**📈 对比分析**

与原始ns-3、Unison、ASTRA-sim等对比，单核下可达744×/510×的加速，16核并行后可达1012×/716×，平均FCT误差<1%，整体性能提升超过1000倍；

**⚠️ 局限性**

在流模式高度随机、短暂或低频稳态的公共云、多租户场景下，记忆化与快进收益下降至ns-3基线，且需要额外的数据库存储与维护成本。

---

## 435. When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning

**arXiv ID:** 2602.10560 | [PDF](https://arxiv.org/pdf/2602.10560v1)

**作者:** Leheng Sheng `[一作]` (Bytedance Seed), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60398 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GRU-Mem，一种在MemAgent基础上加入更新门和退出门的门控递归记忆框架，用于长上下文推理；

**💡 创新点**

创新点在于通过两个文本控制的门（更新门和退出门）和专门的奖励信号，显式控制记忆更新与早停，解决记忆爆炸与无效计算问题；

**🔧 技术方法**

采用强化学习（端到端RL+Multi-Conv DAPO）训练门控策略，结合GRU式门控机制、结构化输出提示、奖励设计（更新奖励、退出奖励、格式奖励）以及优势计算；

**📊 数据集**

在多种长上下文QA基准上验证：HotpotQA、SQuAD、SK-1/2/3、MK-1/2/3、MQ、MV，覆盖7K–896K长度；

**📈 对比分析**

与原版MemAgent对比，GRU-Mem在准确率上普遍提升（部分任务提升约4%），推理速度提升显著，最高可达400%；

**⚠️ 局限性**

局限在仅针对QA任务，未扩展到摘要等任务；额外奖励导致训练稳定性下降，需要更小的离线比例和更长的收敛时间。

---

## 436. Online Generalized-mean Welfare Maximization: Achieving Near-Optimal Regret from Samples

**arXiv ID:** 2602.10469 | [PDF](https://arxiv.org/pdf/2602.10469v1)

**作者:** Zongjun Yang `[一作]` (Columbia University), Christian Kroer `[通讯]` (Columbia University)

**通讯引用:** 675 | [OpenAlex ID](https://openalex.org/A5083207349)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在线公平分配序列到达的物品，最大化泛化-均值福利，提出贪心与重解两类无信息、仅单样本的高效算法；

**💡 创新点**

在未知i.i.d.分布下实现O(1/T)平均遗憾；在非平稳分布下仅用单个历史样本即可恢复同样速率；证明对分布漂移鲁棒，首次将贪心和重解结合并给出相应理论保障；

**🔧 技术方法**

利用对数福利凸化与KKT条件建立 primal–dual 结构，证明福利最大化的稳定性与集中性；采用补偿耦合框架跟踪算法轨迹；用 Wasserstein 距离量化历史与实时分布差异；结合分块与安全区域论证决策一致性；

**📊 数据集**

在 Instagram 通知数据集与 MovieLens 推荐数据集上验证；

**📈 对比分析**

将算法与既往的最优流、最坏情况贪心及其他贪心改进算法对比；实验表明在 i.i.d. 与周期性非平稳、真实时间三种输入模型下，贪心可在 10,000 轮后达到约 99.9% 的后验最优，重解算法更快收敛并在所有模型下保持高性能；

**⚠️ 局限性**

仅适用于 p<1 的公平性目标，假设值向量满足可连续化且有上下界；非平稳模型仅能利用单个历史样本，无法处理极端分布漂移或非常高维/极端稀疏场景；算法对大规模 n、T 的计算复杂度与可扩展性尚未系统评估。

---

## 437. Skirting Additive Error Barriers for Private Turnstile Streams

**arXiv ID:** 2602.10360 | [PDF](https://arxiv.org/pdf/2602.10360v1)

**作者:** Anders Aamand `[一作]` (University of Copenhagen), Sandeep Silwal `[通讯]` (University of Wisconsin-Madison)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

在持续的流式数据环境下，本文研究了不同ially 私有的连续发布算法，用于估计不同元素数量（distinct elements）和第二频率矩（F₂ moment），突破了传统多项式加性误差下限，得到多项式对数级别的加性和乘法误差以及多项式对数级别的空间；

**💡 创新点**

创新点在于提出混合乘法与加法误差框架，利用 DP 持续计数与最小哈希（MinHash）和域降维（domain reduction）相结合，实现了对不同元素计数和 F₂ 矩的多项式对数误差与空间效率；

**🔧 技术方法**

核心技术包括 DP 持续计数、MinHash、域降维、Johnson–Lindenstrauss 投影以及相关的随机哈希与高斯噪声机制；

**📊 数据集**

本文为理论工作，未使用实际数据集，而是通过严谨的理论分析和构造实现误差与空间的下界与上界；

**📈 对比分析**

与之前仅能达到 O(T^{1/3}) 加性误差的算法相比，本文实现了 O(log^k T) 的加性误差和相同级别的乘法误差，并将空间从多项式降低到多项式对数；

**⚠️ 局限性**

主要限制包括：仍需在乘法误差与加性误差之间做权衡；缺乏对 n 与 T 的最优误差下界；在更强的项级私有性下的表现尚未明朗；以及对更通用的多项式加性误差下限的进一步突破仍是开放问题。

---

## 438. FeatureBench: Benchmarking Agentic Coding for Complex Feature Development

**arXiv ID:** 2602.10975 | [PDF](https://arxiv.org/pdf/2602.10975v1)

**作者:** Qixing Zhou `[一作]` (Institute of Automation), Zhaoxiang Zhang `[通讯]` (Institute of Automation)

**通讯引用:** 10808 | [OpenAlex ID](https://openalex.org/A5028016065)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向特性开发、可执行评估的LLM代理编码基准，能够自动从真实开源仓库生成可验证的任务环境

**💡 创新点**

创新点包括：①通过测试驱动、动态追踪构建依赖图实现多文件特性提取；②完全自动化的实例收集管道，支持持续更新与扩展；③明确接口规范与执行管道确保无歧义评估

**🔧 技术方法**

技术手段涵盖：Docker化运行环境、Pytest收集/执行、Python跟踪/依赖图构建、LLM判别接口与补丁、失败-通过与通过-通过测试、执行基准评估、指标计算

**📊 数据集**

数据集为24个主流Python开源项目，生成200个高质量任务（3825个可执行环境），覆盖两种难度（L1增量实现、L2从零实现）

**📈 对比分析**

与现有Bench（SWE-bench等）对比，Claude Opus 4.5仅在本基准上实现约11‑12%任务，SWE-bench上可达70%以上；说明本基准挑战更大，测试更贴近真实特性开发

**⚠️ 局限性**

局限性：LLM在跨文件依赖与命名解析上仍易失误；依赖明确接口，若无接口信息成功率显著下降；评估仍需大量执行资源；模型仍难以一次性完成大规模特性开发

---

## 439. LLM-Based Scientific Equation Discovery via Physics-Informed Token-Regularized Policy Optimization

**arXiv ID:** 2602.10576 | [PDF](https://arxiv.org/pdf/2602.10576v1)

**作者:** Boxiao Wang `[一作]` (Institute of Automation Chinese Academy of Sciences), Jian Cheng `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 PiT-PO 框架，将大型语言模型从静态方程生成器转变为可通过强化学习自适应的生成器，用于符号回归（SR）

**💡 创新点**

创新点包括：1) 双重约束学习信号——层级物理约束与基于支持排除定理的 token 级正则化；2) 将全局奖励与局部 token 级惩罚融合到政策优化中；3) 在搜索过程中进行 LLM 的 in‑search 细调，实现生成策略与任务特性实时同步

**🔧 技术方法**

采用 LLM（Llama、Mixtral 等）+ 强化学习（GRPO、PiT‑PO）+ token 级正则化 + LoRA 适配 + 多岛群体进化 + 语法树复杂度惩罚 + 物理约束（维度一致性、可微性、域特定约束）

**📊 数据集**

使用标准 SR 公开数据集：LLM‑SR Suite（Oscillation 1/2、E. coli Growth、Stress‑Strain）和 LLM‑SRBench（LSR‑Transform、LSR‑Synth）；还在湍流模型任务中利用 DNS 数据（周期山丘流）评估

**📈 对比分析**

与 GPlearn、PySR、uDSR、RAG‑SR、LLM‑SR、SGA、LaSR 等基线在准确率、NMSE、符号准确率上对比；PiT‑PO 在 LLM‑SR Suite 上实现 100% 方程复现率，NMSE 低于对手；在 LLM‑SRBench 上取得最高符号准确率；在湍流任务中得到的 Reynolds‑stress 预测和再附着位置明显优于传统 RANS 与 LLM‑SR，逼近 DNS 结果；同时即使使用 1B 规模 LLM 也可与商业大模型竞争

**⚠️ 局限性**

局限性：1) 对域特定约束的依赖，需要手工编写物理约束；2) 仍需大量计算资源来完成多岛群体与 in‑search 细调；3) 对更复杂的真实系统的推广和跨域迁移尚未系统验证；4) 生成质量仍受 LLM 预训练知识限制，极端稀缺数据场景效果不明

---

## 440. Identifying Evidence-Based Nudges in Biomedical Literature with Large Language Models

**arXiv ID:** 2602.10345 | [PDF](https://arxiv.org/pdf/2602.10345v1)

**作者:** Jaydeep Chauhan `[一作]` (Center for Health Innovation and Implementation Science), Malaz Boustani `[通讯]` (Center for Health Innovation and Implementation Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个两阶段 AI 检索管线，先通过关键词、TF‑IDF、余弦相似度和 nudge 词汇加权把 PubMed 文章从 800 万条缩减到约 81k 条候选，再用 fine‑tuned 的 LLaMA 3.1 8B（OpenScholar）一次性完成 nudge 分类和结构化字段提取，最终得到约 12k 条高质量行为干预文献。

**💡 创新点**

将传统信息检索与 LLM 语义推理深度结合，实现了对医学文献中微妙、非强制性行为干预的自动发现与结构化；提供可调节召回/精确度模式；并将结果直接嵌入实际决策支持平台（Agile Nudge+），大幅提升证据链透明度与可用性。

**🔧 技术方法**

混合检索（关键词 + TF‑IDF + 余弦相似度 + nudge‑term bonus）、量化 LLaMA 3.1 8B（OpenScholar）用于分类与信息抽取、Self‑Consistency 投票、LLM‑as‑Judge 复核、JSON schema 验证。

**📊 数据集**

PubMed 开放存取子集（约 8 M 条全文）作为主语料；人工标注的 197 条样本（86 正例、111 负例）用于评估。

**📈 对比分析**

评估了四种配置：标题+摘要+引言、全文、Self‑Consistency × 7 次投票、Gemini 2.5 Pro 全文。测量指标包括 Precision、Recall、F1 与 Accuracy。最佳召回 72%、F1 67% 的 Title/Abstract/Intro 配置适用于探索性检索；Self‑Consistency 提供 100% Precision、12% Recall，适合高信任场景；全文版在精确度上有所提升但召回下降。

**⚠️ 局限性**

模型易产生假阳性（误把非干预描述视为 nudge）和假阴性（忽略非标准术语或隐含干预）；全文本输入有时会干扰模型注意力；Self‑Consistency 与 Judge 模式计算成本高，吞吐量低；缺乏人工复核可能导致错误传播；对训练数据偏见的放大和伦理风险需通过人机协作与透明推理机制加以缓解。

---

## 441. From Prompt-Response to Goal-Directed Systems: The Evolution of Agentic AI Software Architecture

**arXiv ID:** 2602.10479 | [PDF](https://arxiv.org/pdf/2602.10479v1)

**作者:** Mamdouh Alenezi `[一作]` `[通讯]` (Saudi Technology and Security Comprehensive Control Company Tahakom), Mamdouh Alenezi (Saudi Technology and Security Comprehensive Control Company Tahakom)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文探讨从无状态 prompt 模型到目标驱动 Agentic AI 的架构演进，提出生产级 LLM Agent 的参考架构、跨代理拓扑分类与治理硬化清单，并分析行业平台的共性。

**💡 创新点**

主要创新在于将传统智能体理论与 LLM 工具调用、记忆、治理结合，形成分层参考架构；给出多代理拓扑失败模式与缓解；提出可落地的企业治理检查表。

**🔧 技术方法**

采用 LLM 作为认知核心，工具接口化、内存层化、控制层循环、策略执行、观测追踪；使用 ReAct、Tree of Thoughts 等思维-行动循环；参考 LangChain、ZenML、TrueFoundry 等平台实现。

**📊 数据集**

论文为综述，无实验数据集；参考行业案例平台描述。

**📈 对比分析**

未进行实验比较，讨论采用通用评估框架和指标（工具调用准确率、成本、延迟、可靠性），但未给出具体性能数值。

**⚠️ 局限性**

主要限制是缺乏实证评估、未验证不同拓扑的实际性能；依赖灰度文献，可能存在偏见；未覆盖物理嵌入式系统；对安全可验证性研究仍不足。

---

## 442. Can Large Language Models Make Everyone Happy?

**arXiv ID:** 2602.11091 | [PDF](https://arxiv.org/pdf/2602.11091v1)

**作者:** Usman Naseem `[一作]` (Macquarie University), Rafiq Ali `[通讯]` (DSEU-Okhla)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了MisAlign-Profile统一基准，评估LLM在安全、价值与文化三维度下的交叉偏差，并通过MisAlignTrade数据集实现多维度多语义类型的对齐与失配评测。

**💡 创新点**

创新点在于将安全、价值、文化三维度与对象、属性、关系三类语义误差整合到同一评估框架，并通过两阶段生成与拒绝采样自动构造高质量对齐/失配样本。

**🔧 技术方法**

使用Gemma-2-9B-it进行多标签语义分类，Qwen3-30B-A3B-Instruct-2507用于prompt扩充与评估，SimHash指纹去重，结合两阶段采样与反馈引导生成实现数据集构建。

**📊 数据集**

数据集来源于BeaverTails、ValueCompass、UNESCO的112个规范域，并扩增生成约382,424条prompt‑response对，形成MisAlignTrade。

**📈 对比分析**

与H^3Fusion、TrinityX等通用对齐模型、维度特化微调模型以及开放权重LLM在零样本下进行对比，指标为Coverage、False Failure Rate与Alignment Score；通用模型AS达到79%–86%，单维度模型Coverage可高达97%但FFR>50%。

**⚠️ 局限性**

局限性包括仅覆盖英语文本、依赖现有规范域、自动评估可能带来偏差、人工标注样本有限，且未直接探查模型内部表征。

---

## 443. Deep Learning-based Method for Expressing Knowledge Boundary of Black-Box LLM

**arXiv ID:** 2602.10801 | [PDF](https://arxiv.org/pdf/2602.10801v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 444. Making Databases Faster with LLM Evolutionary Sampling

**arXiv ID:** 2602.10387 | [PDF](https://arxiv.org/pdf/2602.10387v1)

**作者:** Mehmet Hamza Erol `[一作]` (Stanford University), James Zou `[通讯]` (TogetherAI)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用LLM与进化采样对DataFusion的物理执行计划进行增量修改，提升OLAP查询性能。

**💡 创新点**

提出了可序列化的物理计划表示与JSON Patch接口，让LLM局部改动计划；并将进化算法与LLM结合，自动寻找最优补丁。

**🔧 技术方法**

核心技术包括GPT‑5语言模型、DataFusion计划序列化、JSON Patch、进化采样（Parallel Single Threads Evolution 与 Best of Last Evolution）以及Modal沙盒评测。

**📊 数据集**

实验基于TPC‑H与TPC‑DS标准决策支持基准，规模因子3（及规模因子10用于迁移验证）。

**📈 对比分析**

通过在多次沙盒测量中取最小延迟与DataFusion基线比较，平均速度提升约1.1–1.2×，单条查询最高可达4.78×加速，约10%查询无改进。

**⚠️ 局限性**

局限性在于搜索时间通常超过单次执行时间，且对重复性OLAP工作负载的依赖较强；对统计信息的全局控制不足，复杂计划结构的可行性有限。

---

## 445. OSIL: Learning Offline Safe Imitation Policies with Safety Inferred from Non-preferred Trajectories

**arXiv ID:** 2602.11018 | [PDF](https://arxiv.org/pdf/2602.11018v1)

**作者:** Returaj Burnwal `[一作]` (Indian Institute of Technology Madras), Balaraman Ravindran `[通讯]` (Indian Institute of Technology Madras)

**通讯引用:** 4656 | [OpenAlex ID](https://openalex.org/A5009374923)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种离线安全模仿学习算法，利用非首选轨迹推断安全约束，从高收益但安全成本各异的轨迹中学习安全策略。

**💡 创新点**

① 将安全策略学习视为CMDP并在无奖励信息下推导奖励下界；② 通过对比学习与偏好学习共同训练成本模型，估计非首选轨迹的安全成本；③ 采用Lagrangian自适应惩罚平衡安全与性能，实现在安全约束下的最优行为。

**🔧 技术方法**

CMDP框架、对比学习（Contrastive Loss）、偏好学习（Bradley–Terry）、强化学习中的Lagrangian relaxation与自适应α、行为克隆、价值函数逼近。

**📊 数据集**

DSRL离线安全RL数据集，包含六个任务：Walker2d‑Velocity、Swimmer‑Velocity、Ant‑Velocity、Point‑Circle2、Point‑Goal1、Point‑Button1，其中使用大规模高收益轨迹与有限的高成本轨迹。

**📈 对比分析**

与BC‑Union、DWBC、PPL、SafeDICE以及Constrained Decision Transformer对比；在速度限制与导航任务上，该方法在安全度量（Normalized Cost、CVaR）上优于所有基线，且收益接近Constrained RL；消融实验显示对非首选数据量、对比损失与轨迹长度具有鲁棒性。

**⚠️ 局限性**

需要联合数据集包含大量高收益轨迹；对非首选数据覆盖度敏感；成本模型依赖对比与偏好标签，标签噪声或分布偏移可能影响性能；目前仅在离线场景验证，未探索在线交互或多成本约束。

---

## 446. Integrating Generative AI-enhanced Cognitive Systems in Higher Education: From Stakeholder Perceptions to a Conceptual Framework considering the EU AI Act

**arXiv ID:** 2602.10802 | [PDF](https://arxiv.org/pdf/2602.10802v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 447. DFIC: Towards a balanced facial image dataset for automatic ICAO compliance verification

**arXiv ID:** 2602.10985 | [PDF](https://arxiv.org/pdf/2602.10985v1)

**作者:** Nuno Gonçalves `[一作]` (Institute of Systems and Robotics - University of Coimbra), João Marcos `[通讯]` (Institute of Systems and Robotics - University of Coimbra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个规模庞大且分布均衡的面部图像数据集DFIC，用于自动ICAO合规性验证。

**💡 创新点**

创新点在于提供超过58,000张、多模态视频以及超过200万标注，覆盖26项ICAO合规性要求，并设计了统一单模型、注意力机制的验证框架。

**🔧 技术方法**

采用深度学习，结合DeepLabV3语义分割、Mask-specific Spatial Attention Modules、SE通道注意力以及多标签交叉熵训练。

**📊 数据集**

使用DFIC数据集，并在DFIC与TONO+ONOT等公开数据集上进行评估；对比了ICAONet、BioGaze、OFIQ等现有方法。

**📈 对比分析**

在DFIC测试集上平均EER降至0.020，显著优于其他方法；在TONO+ONOT上表现也最优，且偏差与年龄/族裔的公平性最低。

**⚠️ 局限性**

局限性在于模型仍依赖大规模标注；对极端老年或深色肤色的样本性能仍略低，且未利用视频帧进行时序学习。

---

## 448. Eliminating VAE for Fast and High-Resolution Generative Detail Restoration

**arXiv ID:** 2602.10630 | [PDF](https://arxiv.org/pdf/2602.10630v1)

**作者:** Yan Wang `[一作]` (ByteDance), Li Zhang `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在基于扩散的超分辨率模型 GenDR 上，完全移除 VAE，将编码器改为 ×8 pixel-unshuffle，解码器改为 ×8 pixel-shuffle，并通过多阶段对抗蒸馏训练得到 GenDR-Pix，实现端到端像素空间推理。

**💡 创新点**

创新点包括：① 将整个 VAE 替换为简单的 pixel‑shuffle/unshuffle，从根本上降低内存和时间成本；② 采用分阶段对抗蒸馏，逐步移除 encoder/decoder 并使用来自先前阶段模型的特征进行对抗学习；③ 设计带掩码的傅里叶空间损失和随机填充（RandPad）来抑制重复图案伪影；④ 结合自投影和 classifier‑free 指导的 PadCFG，进一步提升推理质量与速度。

**🔧 技术方法**

核心技术：pixel‑shuffle / pixel‑unshuffle、扩散模型（Latent Diffusion）、多阶段对抗蒸馏、随机填充增强、掩码 Fourier 损失、PadCFG（自投影 + CFG）。

**📊 数据集**

训练使用 LSDIR、FFHQ、DiffusionDB、Laion 之高质量子集；测试使用 ImageNet‑Test、RealSR、RealSet80、RealLR250、4K benchmark 等真实世界降质数据集。

**📈 对比分析**

与 GAN（BSRGAN、Real‑ESRGAN、Real‑HATGAN）、一阶扩散（SinSR、OSEDiff、InvSR、AdcSR、GenDR）以及多步扩散（StableSR、DiffBIR、SeeSR、DreamClear）对比。结果显示 GenDR‑Pix 在 PSNR/SSIM 上与 GenDR 相近，且在速度上比 GenDR 提升 2.8×、内存降低 60%，可在 4K 下单帧 <1 s，推理时长 32 ms（512×512），在 RealSet80 上实现最高效且质量优秀的超分。

**⚠️ 局限性**

仍存在轻微感知质量下降（相较原 GenDR），对不同放大倍数的鲁棒性尚待进一步验证，且多阶段蒸馏与 RandPad 等技术对训练稳定性有一定要求。

---

## 449. Navigating heterogeneous protein landscapes through geometry-aware smoothing

**arXiv ID:** 2602.10422 | [PDF](https://arxiv.org/pdf/2602.10422v1)

**作者:** Srinivas Anumasa `[一作]` (National University of Singapore), Dianbo Liu `[通讯]` (A Star Singapore)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了一种基于局部密度自适应的噪声调节方法 Density‑Dependent Smoothing (DDS)，改进了离散扩散模型在蛋白质序列生成中的性能。

**💡 创新点**

创新点在于将扩散噪声与序列空间的局部密度反比关联，实现对稀疏、异质生物序列景观的自适应平滑，解决了固定全局噪声导致的过度平滑或碎片化问题。

**🔧 技术方法**

技术上采用神经经验贝叶斯 (NEB) 与离散 Walk‑Jump Sampling (dWJS)，利用核密度估计 (KDE) 计算序列密度，构建 σ(x) 调度，并在 σ 条件下进行 Langevin 采样；评估使用 ESM‑2 结构预测、语言模型困惑度、BLAST 同源性等指标。

**📊 数据集**

实验使用了观察抗体空间 (OAS)、临床抗体、抗菌肽数据库 DBAASP、冠状病毒抗体数据库 CoV‑AbDab，以及一个 4D 合成多模态基准。

**📈 对比分析**

与固定噪声的 dWJS、SeqVDM、DEEN、IgLM、ESM‑2、GPT‑3.5 等基线进行多维度对比（多样性、质量、结构置信度、伪困惑度、BLAST 同源性、KS/Wasserstein 等），DDS 在所有任务中均表现出更高的质量‑多样性平衡，显著减少假模式并提升结构与功能可信度。

**⚠️ 局限性**

局限性包括需手动设定 σ_min/σ_max 范围、缺乏理论误差分析、实验主要为经验性，对极端稀疏区域的自适应性仍有待进一步研究。

---

## 450. ImprovEvolve: Ask AlphaEvolve to Improve the Input Solution and Then Improvise

**arXiv ID:** 2602.10233 | [PDF](https://arxiv.org/pdf/2602.10233v1)

**作者:** Alexey Kravatskiy `[一作]` (MIRIAI), Ivan Oseledets `[通讯]` (Institute of Numerical Mathematics)

**通讯引用:** 10533 | [OpenAlex ID](https://openalex.org/A5004111307)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ImprovEvolve 方法，通过将 LLM 进化的程序拆分为初始化、局部改进和扰动三模块，以降低对 LLM 的认知负担，实现更高效的 LLM‑驱动进化优化。

**💡 创新点**

创新点在于将优化任务拆解为可独立进化的接口类，采用基于 LLM 的局部搜索和扰动子程序，在传统 MAP‑Elites 演化框架中实现模块化优化，显著提升在高维、非凸问题上的效果。

**🔧 技术方法**

使用大型语言模型（Gemini 3 Pro/Flash）、MAP‑Elites 进化算法、基于 LLM 的变异、基于基地跳跃（basin‑hopping）的双阶段验证方案，以及手工编辑的辅助改进。

**📊 数据集**

用到的数据集为两个优化基准：正六边形包装（HEX n，n=11‑30）和第二自相关不等式（ACI 2）及其对应的目标函数评估。

**📈 对比分析**

与 AlphaEvolve、ThetaEvolve、CodeEvolve、GigaEvo 等开源框架对比，在 HEX 11‑30 与 ACI 2 上取得了新的最优结果，提升了包装长度或下界系数，并通过双阶段验证和调参实现更高成功率。

**⚠️ 局限性**

仅在两类数值优化问题上验证，缺乏对组合优化或其他领域的通用性；实验结果单次跑优未给出置信区间；计算成本高且高度依赖 LLM。

---

## 451. Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets

**arXiv ID:** 2602.10583 | [PDF](https://arxiv.org/pdf/2602.10583v1)

**作者:** Bo Xue `[一作]` (Shanghai Jiao Tong University), Zhouhan Lin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6432 | [OpenAlex ID](https://openalex.org/A5024900991)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于生成流网络（GFlowNets）的跨度语言模型FoSS，通过动态构造跨度词汇表，将文本生成问题建模为有向无环图（DAG）上的状态空间，提升生成多样性与质量。

**💡 创新点**

创新点在于：① 用跨度而非单词构建动态词汇，② 明确将生成过程视作DAG而非传统树结构，③ 结合专门的奖励模型（LM+PM）引导文本生成。

**🔧 技术方法**

技术包括：生成流网络框架、跨度语言模型（Transformer编码器+跨度检索器）、动态词汇构造算法、混合在线-离线训练策略以及奖励函数设计。

**📊 数据集**

主要数据集有WikiText-103（语料库）、Law-MT（法律域适配）、En-Wiki（检索存储）、以及TruthfulQA、OpenBookQA、ARC-Challenge、MedMCQA、Med-USMLE等知识密集问答集。

**📈 对比分析**

在MAUVE、Diversity及GPT‑4质量评估上，FoSS相较于Transformer、kNN‑LM、RETRO、CoG、GFlowNets‑FT、GDV等基线提升约5–12.5% MAUVE、约3–5%多样性，并在知识问答任务中实现最高准确率。

**⚠️ 局限性**

局限性包括：对检索语料的依赖、跨度切分策略仍需改进、奖励模型可能引入偏差，以及在极端长文本生成时仍可能出现连贯性下降。

---

## 452. MindPilot: Closed-loop Visual Stimulation Optimization for Brain Modulation with EEG-guided Diffusion

**arXiv ID:** 2602.10552 | [PDF](https://arxiv.org/pdf/2602.10552v1)

**作者:** Dongyang Li `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 3808 | [OpenAlex ID](https://openalex.org/A5078854583)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一种闭环EEG引导的视觉刺激优化框架MindPilot，利用非侵入式EEG反馈和扩散模型生成自然图像以调节脑活动。

**💡 创新点**

将大脑视为黑盒，采用伪模型梯度无梯度优化，引入EEG语义和频谱特征的双重目标，并在非侵入式设置下实现闭环图像生成。

**🔧 技术方法**

EEG黑盒代理预测模型、CLIP/EEG特征编码、Stable Diffusion XL-Lightning扩散模型、伪模型梯度（Gaussian Process）指导、遗传算法与交叉变异、滚动搜索等。

**📊 数据集**

THINGS‑EEG2（17通道×250时点EEG）与THINGS图像集，公开EEG数据集。

**📈 对比分析**

与随机抽样、离线伪模型、ATM‑S、CongCapturer等基线对比；在语义相似度、CLIP匹配、EEG特征匹配等指标上显著优于随机，性能接近理论上限，实验中人类评分与模型预测高度相关。

**⚠️ 局限性**

存在多重刺激导致相同EEG（metamer）现象、个体差异建模不足、实时性低、仅验证视觉模态，可进一步扩展到其它感官或更高级脑功能。

---

## 453. Low-Dimensional Execution Manifolds in Transformer Learning Dynamics: Evidence from Modular Arithmetic Tasks

**arXiv ID:** 2602.10496 | [PDF](https://arxiv.org/pdf/2602.10496v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过对过参数化 Transformer 在标记式模数加法任务中的学习轨迹进行几何分析，发现训练过程快速坍缩到 3–4 维的执行流形，并用此解释注意力聚焦、SGD 可积性和稀疏自编码器的局部性。

**💡 创新点**

创新点在于提出并验证了“执行流形”这一概念：学习动态主要集中在低维子空间内，进而统一解释注意力“泡沫”、SGD 的近似可积性以及稀疏特征与计算本质的分离，并展示了训练课程对流形结构的决定性影响。

**🔧 技术方法**

使用了主成分分析（PCA）评估有效秩、梯度交换子（commutator）分析判断可积性、稀疏自编码器（SAE）探测可解释特征、以及自定义标记式模数加法任务；实验中还比较了注意力仅模型与标准 Transformer、混合训练与单一课程的效果。

**📊 数据集**

数据集为自定义的标记式模数加法数据集，序列长度 32，标记数 m 从 1 到 6，模数 C 设为 8 或 16；通过在不同 m 组合和课程设定下训练多次得到轨迹和性能数据。

**📈 对比分析**

方法上通过多种随机种子评估有效秩稳定性、对比原始与投影后的梯度交换子范数、分析稀疏自编码器在不同训练阶段的消融效果；结果显示，训练轨迹在 20–30% 步骤内即坍缩至 3–4 维，混合训练能保持对所有 m 的高精度，单一课程易产生灾难性遗忘；整体性能与现有研究一致或更优。

**⚠️ 局限性**

局限性包括：仅在极度控制的合成任务上验证，尚未证实在自然语言或视觉等复杂任务中是否同样出现低维执行流形；加入 MLP 层会破坏低维结构；执行流形维度随任务复杂度的扩展规律尚不明确。

---

## 454. Fault Tolerant Design of IGZO-based Binary Search ADCs

**arXiv ID:** 2602.10790 | [PDF](https://arxiv.org/pdf/2602.10790v1)

**作者:** Paula Carolina Lozano Duarte `[一作]` (Karlsruhe Institute of Technology), Mehdi Tahoori `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 9042 | [OpenAlex ID](https://openalex.org/A5064445713)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

针对基于IGZO薄膜技术的二进制搜索ADC设计了层次化的缺陷注入框架并实现了缺陷容错设计

**💡 创新点**

创新点包括：①首次将层次化故障注入与系统级故障传播分析结合，用于高缺陷密度的IGZO技术；②通过缺陷灵敏度分析实现针对关键元件的选择性冗余和比较器拓扑改造；③在单/多缺陷场景下显著提升错误覆盖率，面积与功耗提升仅为4.2%和6%

**🔧 技术方法**

采用的技术主要有：IGZO晶体管级缺陷模型（开路/短路电阻注入）、子电路级故障库生成、系统级DNL分类（灾难性、边际、良性）、自动化Python仿真管线、选择性前端冗余与比较器级冗余设计

**📊 数据集**

未使用公开数据集；所有实验均基于Cadence Spectre仿真，输入为5 Hz全范围正弦波，覆盖超过220种单/多缺陷配置

**📈 对比分析**

方法上将基准设计与两种容错架构（SFR、ECLR）在相同仿真环境下进行单/多缺陷覆盖率、面积与功耗对比；结果显示单缺陷覆盖率从60%提升至92%，多缺陷从34%提升至77.3%；面积提升4.2%，功耗提升6%

**⚠️ 局限性**

局限性包括：仅验证3位ADC，对更高分辨率的可扩展性未深入评估；缺陷模型仅考虑开路/短路，未覆盖参数漂移或电荷累积等失效机制；设计改进主要针对IGZO类技术，迁移到其他薄膜或CMOS需要进一步验证

---

## 455. SimuScene: Training and Benchmarking Code Generation to Simulate Physical Scenarios

**arXiv ID:** 2602.10840 | [PDF](https://arxiv.org/pdf/2602.10840v1)

**作者:** Yanan Wang `[一作]`, Haonan Li `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 946 | [OpenAlex ID](https://openalex.org/A5100742670)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究 LLM 能否通过代码生成来模拟物理场景，构建首个物理场景代码生成任务与数据集 SimuScene，设计完整的代码→视频→VLM 评估管线，并基于视觉奖励进行强化学习。

**💡 创新点**

创新点包括：①提出物理场景代码生成这一新任务并创建大规模、高质量数据集；②使用 Vision‑Language 模型作为判定器评估视频结果，解决传统文本评估无法捕捉动态过程的缺陷；③引入 Code‑Video‑Judge 的强化学习框架，利用可验证的视频奖励显著提升 LLM 的模拟能力。

**🔧 技术方法**

核心技术：自动化场景与验证问题生成 pipeline（GPT‑4o、DeepSeek‑R1‑0528）；评估 pipeline（代码提取、视频渲染、VLM 判定，采用 Qwen‑2.5‑VL 等模型）；SFT 与 RL（AgentFly + GRPO）训练；多模型 VLM 投票与训练-测试 VLM 分离以防奖励作弊。

**📊 数据集**

数据集：SimuScene 包含 7,659 条动态物理场景（5 个领域 52 个概念），其中 334 条经过人工核验的测试集；训练集 12,556 条已验证的代码/视频对，用于 SFT；约 4,325 条用于 RL 训练。

**📈 对比分析**

实验比较：对 10 个前沿 LLM 进行评估，最佳 Avg@8 仅 21.5%，Pass@8 最高 52.7%；SFT 后模型性能提升明显；RL（vision‑based）训练后 7B 模型 Pass@8 达 34.4%，32B 模型达 72.2%，甚至超过部分更大规模模型。VLM 评估与人工标注一致率 ≥ 83%。

**⚠️ 局限性**

局限性：①现有 LLM 在物理推理和动态模拟上仍显不足；②RL 依赖 VLM 评判，若 VLM 出现偏差或误判可能导致奖励失真；③数据集规模相对有限，某些领域（如光学）表现尤其差；④评估仍受限于 VLM 的可解释性与算力需求。

---

## 456. Collaborative Threshold Watermarking

**arXiv ID:** 2602.10765 | [PDF](https://arxiv.org/pdf/2602.10765v1)

**作者:** Tameem Bakr `[一作]` (MBZUAI), Nils Lukas `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在联邦学习环境下的（t, K）阈值水印方案，使得至少 t 名参与方共同验证模型归属，而单个或少数参与方无法提取水印键。

**💡 创新点**

将 Shamir 秘密共享与安全聚合相结合，实现可扩展的集体嵌入与阈值验证；并在白盒设置下实现无需泄露水印键即可计算检测统计。

**🔧 技术方法**

Shamir 共享、Lagrange 係数、加密聚合（Secure Aggregation）、自适应水印强度调节、基于余弦相似度的一侧 z‑检验。

**📊 数据集**

CIFAR‑10、CIFAR‑100 与 Tiny ImageNet 图像分类数据集，使用 ResNet‑18 作为模型。

**📈 对比分析**

与每位客户端独立嵌入的基线做对比；在 K 从 4 到 128 扩展时，阈值水印保持可检测（z≥4），基线在 K≥16 时失效；在保持模型准确率低于 1–2pp 的前提下，水印在剪枝、量化、精调与蒸馏等攻击下保持检测阈值以上。

**⚠️ 局限性**

仅实现白盒验证；假设参与方诚实好奇，未考虑训练阶段的恶意客户端或服务器；仅在 IID 图像分类任务上实验，未验证对大规模 LLM 或非 IID 场景的适用性；对全轨迹攻击的理论安全性仍未给出下界。

---

## 457. When are We Worried? Temporal Trends of Anxiety and What They Reveal about Us

**arXiv ID:** 2602.10400 | [PDF](https://arxiv.org/pdf/2602.10400v1)

**作者:** Saif M. Mohammad `[一作]` (National Research Council Canada), Saif M. Mohammad `[通讯]` (National Research Council Canada)

**通讯引用:** 15340 | [OpenAlex ID](https://openalex.org/A5033684482)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用词汇情绪词典和大规模推文语料，对北美用户在不同时间、时态和人称使用焦虑词汇的模式进行聚合分析。

**💡 创新点**

创新点在于将焦虑词典与推文时序数据相结合，首次揭示了焦虑在一天中的高峰时段、周中峰值以及时态和人称关联的系统性差异。

**🔧 技术方法**

主要技术为词典匹配与情绪得分计算（焦虑词占比减去平静词占比），并采用t检验评估不同时间/人称/时态组之间的显著差异。

**📊 数据集**

使用的数据集包括约44,500词条的WorryWords焦虑词典和2015–2021年美国与加拿大的地理定位推文语料TUSC。

**📈 对比分析**

在对比方法上，本文借鉴并扩展了Dodds等人（2011）关于情绪时序的分析框架，结果显示焦虑在早晨最高、午间最低，周三峰值，过去式焦虑最高，未来式最平静；这些结论在统计上均显著，表明词典方法能可靠捕捉聚合级焦虑趋势。

**⚠️ 局限性**

局限性包括样本仅覆盖北美英语使用者，词典可能忽略不同文化与专业领域的语义差异，且推文语言表达可能不完全反映真实情绪，故结论对其他语言和地区的推广需谨慎。

---

## 458. Implementation of Polynomial NP-Complete Algorithms Based on the NP Verifier Simulation Framework

**arXiv ID:** 2602.10991 | [PDF](https://arxiv.org/pdf/2602.10991v1)

**作者:** Changryeol Lee `[一作]` (Yonsei University), Changryeol Lee `[通讯]` (Yonsei University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5008770227)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文通过在可行图模拟框架下构造了明确的有限状态确定性图灵机（DTM），分别针对 SAT 和 Subset‑Sum 两个 NP‑完备问题给出了完整的机器描述、状态表、数据结构以及实现细节，并提供了 Python 代码实现；同时将可行图构造与候选边验证的实现从原始抽象版本改为更直接、低复杂度的算法，并把该框架扩展到 FNP 计算，能够在接受实例时显式生成证书。

**💡 创新点**

创新点主要有三：
1) 先前仅给出抽象证明的可行图模拟框架，本文首次给出完整、可执行的 DTM 具体实现；
2) 通过引入限制式候选边验证和定义忠实的可行图构造算法，显著降低了候选边检查的组合增长，从而减小了多项式阶数；
3) 在实现中加入了 FNP 证书重构机制，证明在接受实例时能够在相同多项式复杂度下直接生成合法证书，扩展了框架的应用范围。

**🔧 技术方法**

使用的技术与方法包括：
- 可行图（feasible graph）模拟与重构；
- 计算图、可行 walk 与边验证机制；
- 限制式候选边扩展与边验证算法；
- FNP 证书重构（floor‑edge 追踪）；
- 具体的状态转移表、数据结构与算法实现；
- Python 实现与实验验证。

**📊 数据集**

实验主要在人工构造的 SAT 与 Subset‑Sum 小规模实例上进行，输入长度约 50–200 位（部分实验达 850 位），未使用公开的大型数据集；代码已托管在公开仓库中，便于复现。

**📈 对比分析**

与原始框架（理论复杂度最高至 O(p(n)^19)）相比，本文通过限制式边验证和更直接的可行图构造将有效多项式度数显著降低，实验结果显示实际运行时间与理论预测相符，能够在上述规模输入下完成；但由于缺乏大规模基准测试，无法与工业级 SAT/Subset‑Sum 求解器在规模与性能上直接对比。

**⚠️ 局限性**

局限性包括：
- 实现仍以小规模实例为主，未验证在更大规模输入上的可扩展性；
- 代码未进行性能调优，运行时间虽然在多项式范围内但相对传统求解器仍较慢；
- 仅针对 SAT 与 Subset‑Sum 两个问题，未展示框架在其他 NP‑完备问题上的通用性；
- 对可行图重构与候选边扩展的优化空间仍然存在，特别是对边数增长的更高阶约束未被充分挖掘。

---

## 459. Kernel-Based Learning of Chest X-ray Images for Predicting ICU Escalation among COVID-19 Patients

**arXiv ID:** 2602.10261 | [PDF](https://arxiv.org/pdf/2602.10261v1)

**作者:** Qiyuan Shi `[一作]` (University of Michigan), Yi Li `[通讯]` (University of Michigan)

**通讯引用:** 40619 | [OpenAlex ID](https://openalex.org/A5100421454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一种将多核学习嵌入广义线性模型的逐步列生成方法GLIMARK，用于预测COVID‑19患者ICU升级并进行放射组学特征解释。

**💡 创新点**

将多核学习推广到指数族分布的GLM框架，结合分层视图、列生成与稀疏选择，实现可解释的患者代表点选择，并在多尺度特征层面进行自适应建模。

**🔧 技术方法**

多核学习、广义线性模型、列生成（Column Generation）、前向稀疏选择、Adam优化、RBF/多项式核、特征层级分解。

**📊 数据集**

Michigan Medicine 2020–2023 COVID‑19胸部X光影像与电子病历融合数据集（2396例，2064维放射组学特征）。

**📈 对比分析**

与随机森林、L1/L2正则化逻辑回归和SVM进行5折交叉验证比较，GLIMARK在约200–400个代表点时AUC可达0.83，优于RF（≈0.82）和逻辑回归，且模型复杂度显著降低。

**⚠️ 局限性**

对核函数和层级结构的选择依赖经验，列生成对大规模视图组合的计算仍昂贵，且解释性受相似度度量限制；未充分利用深度特征或多视角（如LAT）信息。

---

## 460. Diffusion-Pretrained Dense and Contextual Embeddings

**arXiv ID:** 2602.11151 | [PDF](https://arxiv.org/pdf/2602.11151v1)

**作者:** Sedigheh Eslami `[一作]` (Perplexity AI), Denis Bykov `[通讯]` (Perplexity AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 PPLX-Embed 族模型，利用扩散预训练的双向语言模型实现多语言文本嵌入，并通过多阶段对比学习（对齐、上下文、三元组）和量化感知训练，直接输出 INT8/二进制嵌入。

**💡 创新点**

创新点在于：①将扩散模型（diffusion language model）迁移至嵌入任务，实现全双向注意力；②设计多阶段训练管线，先对齐查询-文档，再通过上下文训练学习分块嵌入，最后利用硬负样本提升判别力；③在训练阶段即嵌入量化，保持 INT8/二进制的性能；④在大规模真实检索日志上构建内部评测基准，验证模型在 Web‑scale 情境下的召回能力。

**🔧 技术方法**

使用的技术包括：扩散预训练（持续预训练）、InfoNCE 对比损失、掩码负样本策略、Spherical Linear Interpolation 合并、Mean Pooling、INT8 量化与二进制量化、以及对批次内重复文档的掩码处理。

**📊 数据集**

数据集覆盖多语言：FineWeb-Edu、FineWeb2、FineWeb2-HQ；对比训练使用 65.6% 英文、6.7% 跨语言、1% 代码、26.7% 多语种样本；上下文训练使用 ConTEB、MLDR；三元组训练使用 12 个高质量英文/多语种数据；内部基准基于真实搜索日志的 PPLXQuery2Query 与 PPLXQuery2Doc。

**📈 对比分析**

与公开基准（MTEB、MIRACL、MTEB Code、ConTEB、BERGEN、ToolRet）以及内部基准比较，PPLX-Embed‑4B 在 MTEB 69.66%（INT8）接近或优于 Qwen3‑Embedding‑4B 与 gemini‑embedding‑001；在 ConTEB 上 81.96% 超过 voyage‑context‑3、Anthropic Contextual；在 ToolRet 上 44.45% 超过 NV‑Embed‑v1、GritLM‑7B；在内部检索基准中，-4B 在 PPLXQ2Q 和 PPLXQ2D 的召回率均领先，尤其在 1000 条候选时达到 88%+，显示出强大的首检性能。

**⚠️ 局限性**

局限性包括：①模型未进行指令微调，使用时需自行处理查询格式；②对比学习中对负样本的掩码和硬负样本的采样对不同任务的适用性仍需进一步验证；③二进制量化在小模型（0.6B）上降幅明显（2–4%），需在部署时权衡精度与存储；④目前主要针对检索场景，未针对生成或对话任务展开评测。

---

## 461. The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste Segregation

**arXiv ID:** 2602.10500 | [PDF](https://arxiv.org/pdf/2602.10500v1)

**作者:** Suman Kunwar `[一作]` `[通讯]` (DWaste), Suman Kunwar (DWaste)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文发布了Garbage Dataset（GD）——一个包含13,348张、10类家庭废弃物的真实场景图像数据集，并在该数据集上进行基准实验，分析其类不平衡、背景复杂度和碳排放等特性；

**💡 创新点**

创新点在于构建大规模、多源、公开可用的废弃物图像数据集，同时将数据属性分析与环境成本评估相结合，提供了可持续AI评估框架；

**🔧 技术方法**

使用手机应用与网络爬取收集图像，校验和与感知哈希保证数据完整性，利用PCA、t‑SNE、熵、显著性评估视觉可分离度；模型方面采用迁移学习的EfficientNetV2、MobileNet、ResNet系列，并通过Code Carbon追踪训练碳排放；

**📊 数据集**

核心使用自己构建的GD数据集（13,348张，10类），并对比现有公开数据集如TrashNet、TACO等；

**📈 对比分析**

实验采用训练/验证/测试分割，随机欠采样处理不平衡，评估准确率、召回率、F1、训练时间和碳排放；EfficientNetV2S在原图上达到96.19%准确率、0.96 F1、5,635s训练、0.0698kg CO₂e排放；MobileNet速度快但准确率低；

**⚠️ 局限性**

限制包括严重的类不平衡、背景复杂、光照偏差，模型仍易在高复杂度背景下混淆；目前仅做基准实验，未应用高级增强或重采样技术；碳排放评估仅覆盖训练阶段，未涵盖推理阶段。

---

## 462. Don't Eliminate Cut: Exponential Separations in LLM-Based Theorem Proving

**arXiv ID:** 2602.10512 | [PDF](https://arxiv.org/pdf/2602.10512v1)

**作者:** Sho Sonoda `[一作]` (RIKEN AIP), Yuya Uezato `[通讯]` (CyberAgent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文将LLM驱动的交互式定理证明视为有限时程的确定性MDP，并通过分布式生成模型引入可共享的证明DAG，进而给出平面与层次化学习+搜索的成功概率下界，证明在剪枝消除会导致的指数级证据膨胀下，层次化子目标分解方法在样本效率上呈指数级优越；

**💡 创新点**

创新点在于：①将证明过程建模为compact metric空间上的Lipschitz策略MDP；②提出基于参考策略q的成功过滤分布与潜在证明DAG的生成分布；③在top‑k搜索与Tsybakov型边距条件下，利用序列Rademacher/covering复杂度与ELBO误差分离地给出平面与层次化方法的下界；④通过剪枝消除导致的决策点指数膨胀，得到平面与层次化学习的指数级分离；

**🔧 技术方法**

使用的技术包括：有限时程确定性MDP建模、compact metric状态/动作空间与Lipschitz策略、top‑k搜索、Tsybakov‑型边距条件、序列Rademacher/covering复杂度分析、EM/VAE式潜变量推理与ELBO控制、对比分析中的指数级决策点计数；

**📊 数据集**

采用理论生成分布：①成功过滤的q策略产生的cut‑free轨迹分布Q_tree；②潜在证明DAG分布Q_DAG（包含参数D、b_eff、α等）；并未使用实际公开数据集，主要以假设的生成模型进行分析；

**📈 对比分析**

比较方法：将层次化子目标分解+两级top‑k搜索与先剪枝消除后平面学习+top‑k搜索进行对比；结果显示在满足边距与学习曲线假设下，层次化方法在同等成功概率下所需样本量为平面方法的指数级比例；

**⚠️ 局限性**

局限性包括：依赖严格的理论假设（compact metric、Lipschitz、Tsybakov边距、确定性转移、可逆的潜变量模型）；未在真实LLM或Lean系统上实证验证；对高维动作空间的实际可行性未讨论；

---

## 463. AIvilization v0: Toward Large-Scale Artificial Social Simulation with a Unified Agent Architecture and Adaptive Agent Profiles

**arXiv ID:** 2602.10429 | [PDF](https://arxiv.org/pdf/2602.10429v1)

**作者:** Wenkai Fan `[一作]` (Hong Kong University of Science and Technology), Kani Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 871 | [OpenAlex ID](https://openalex.org/A5040402090)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发并公开部署了 AIvilization v0，一款大规模、长时限的人工社会平台，集成 LLM‑驱动代理、分层分支规划、双层记忆与人类介入，并通过自动做市商与教育-职业门槛实现可持续的经济与社会模拟。

**💡 创新点**

创新点在于：① 将分层分支规划、前向模拟与多级重新规划集成到单一代理循环，显著降低长序列规划失误；② 采用双层记忆架构，让短期执行与长期身份演化并行；③ 通过人类长短期目标注入，实现在代理记忆层级的持续影响；④ 在公开部署中用真实玩家交易日志重现金融市场的重尾与波动聚类，为人工社会实验提供了实证基准。

**🔧 技术方法**

技术手段包括：大语言模型（LLM）代理、分层分支思维规划器、行动模拟器与修复机制、双层记忆（短期/长期）系统、自动做市商（AMM）价格机制、教育-职业门槛与居住层级的硬约束、以及基于规则的物理生存与生产模型。

**📊 数据集**

使用的数据集是平台公开运行期间收集的约 600,000 条高频交易记录（按 5 分钟分箱得到 OHLC 数据），以及对应的代理身份、教育分数、居住层级、资产与库存信息，用于统计学分析与消除。

**📈 对比分析**

通过对比三种规划变体（完整分支规划、无分支、无目标分解）在多目标任务与单目标任务中的表现，评估指标包括净资产、库存价值、货币余额、教育分数、健康/能量/饱食度等。实验显示完整规划在多目标、长期投资任务上显著优于两种简化版；在单目标直接任务上简化版可实现相近表现，减少计算开销。

**⚠️ 局限性**

局限性：① 代理性能受限于 LLM 的推理与潜在幻觉，偶尔出现子优化或错误约束；② 社会分层与规划时序的因果关系仅为观察性关联，需进一步 A/B 试验验证；③ 维持数千名并发记忆增强代理的计算成本高，影响规模扩展到百万级时的可行性。

---

## 464. First International StepUP Competition for Biometric Footstep Recognition: Methods, Results and Remaining Challenges

**arXiv ID:** 2602.11086 | [PDF](https://arxiv.org/pdf/2602.11086v1)

**作者:** Robyn Larracy `[一作]` (University of New Brunswick), Erik Scheme `[通讯]` (University of New Brunswick)

**通讯引用:** 7166 | [OpenAlex ID](https://openalex.org/A5025818642)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

推出首届国际StepUP竞赛，利用UNB StepUP-P150数据集开展足迹识别验证，聚焦未知用户与未见鞋履、速度的泛化挑战

**💡 创新点**

提出生成奖励机（GRM）框架以自动化模型架构与超参数搜索，并与多方法对比，突出利用训练动态预测最终性能的理念

**🔧 技术方法**

使用深度时空卷积网络（R(2+1)D）、三元组损失、数据增强、基于梯度的奖励学习、贝叶斯优化及进化课程协同优化等技术

**📊 数据集**

以UNB StepUP-P150（200k+高分辨率足迹）作为训练集，包含30名未参与训练的被试作为参考与探针集，进行验证评估

**📈 对比分析**

以等错误率（EER）为主要排名指标，顶级方案EER 10.77%，相较基线19.5%显著提升；同时提供FMR、FNMR、ACC、BACC等指标，显示不同操作点的权衡

**⚠️ 局限性**

仍面临鞋履变异导致的泛化不足，尤其第二双鞋与特殊鞋类（如Birkenstock）误判高，说明需进一步提升对鞋型、脚型多样性的鲁棒性

---

## 465. Med-SegLens: Latent-Level Model Diffing for Interpretable Medical Image Segmentation

**arXiv ID:** 2602.10508 | [PDF](https://arxiv.org/pdf/2602.10508v1)

**作者:** Salma J. Ahmed `[一作]` (Wilfrid Laurier University), Azam Asilian Bidgoli `[通讯]` (Wilfrid Laurier University)

**通讯引用:** 334 | [OpenAlex ID](https://openalex.org/A5017611735)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 Med‑SegLens 框架，通过稀疏自编码器对医学图像分割模型的中间激活进行分解，执行跨数据集模型 diffing，定位并通过潜在层干预纠正因数据集偏移导致的分割错误。

**💡 创新点**

首次将稀疏自编码器与跨数据集潜在特征对齐相结合，实现在医学分割任务中对共享与特定潜在特征进行因果诊断与可解释干预。

**🔧 技术方法**

使用 BatchTopK 稀疏自编码器、Hungarian 对齐、自动化潜在语义解释、潜在层 Steering（放大/抑制）以及 Additive/Multiplicative Latent Steering。

**📊 数据集**

四个 T1 头颅 MRI 数据集：IXI（健康成人）、BraTS 2023 的成人、儿童与撒哈拉以南非洲胶质瘤三种子群。

**📈 对比分析**

通过潜在特征对齐对比共享/特定特征，并以 Dice、精度召回等指标评估；未重新训练的干预可在失效案例中恢复 70% 性能，最差类别 Dice 从 39.4% 提升至 74.2%。

**⚠️ 局限性**

潜在特征受训练分布影响，解释依赖预定义的几何与空间度量，单一对应的对齐可能无法捕获多重或模糊的特征对应。

---

## 466. Vulnerabilities in Partial TEE-Shielded LLM Inference with Precomputed Noise

**arXiv ID:** 2602.11088 | [PDF](https://arxiv.org/pdf/2602.11088v1)

**作者:** Abhishek Saini `[一作]` (Rutgers, The State University of New Jersey), Hang Liu `[通讯]` (Rutgers, The State University of New Jersey)

**通讯引用:** 16941 | [OpenAlex ID](https://openalex.org/A5038008601)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文通过发现并利用预计算静态基向量导致的低维噪声子空间，设计了一种代数式的多查询攻击，能够完全破解基于 TEE 的 LLM 推理的机密性和完整性防护。

**💡 创新点**

创新点在于：①首次将预计算静态基向量识别为系统级别的安全缺陷；②提出两种跨查询的线性子空间恢复攻击，既能恢复隐藏的置换键，也能绕过完整性检查；③证明该缺陷在所有主流 PTSE 方案中通用。

**🔧 技术方法**

采用的技术主要是线性代数工具（Gram‑Schmidt 正交化、投影矩阵构造、向量空间交集求解）来从多次交互中恢复噪声子空间；利用预计算噪声表、有限域运算与 GPU 线性算子混合计算实现攻击；并在 SGXv2 环境下实现原型。

**📊 数据集**

使用的模型数据集包括 LLaMA‑3 8B、Gemma‑3 27B、LLaMA‑3 70B、LLaMA‑3.1 405B 等多种规模的 LLM，且在 GSM8K 数据集上测评模型准确率与吞吐量。

**📈 对比分析**

与基线（无 TEE）和现有 PTSE 系统做对比：在 8B 模型上，攻击恢复单层置换键仅需约 6.2 分钟，恢复完整模型后可恢复原始 240 tok/s 吞吐；完整性攻击可在几小时内完成；相比之下，传统的完整性验证与模型加密在性能上均不可行，预计算方式虽快但被攻击。

**⚠️ 局限性**

限制包括：假设 TEE 完全可信且无侧信道；攻击需要多次查询（至少等于基向量维度 K），对实时交互系统可能不够及时；增加基向量数量仅能线性延迟攻击时间，无法根本阻止；若模型使用完全不同的加密方式或不使用预计算基向量，则攻击无效。

---

## 467. The Computational Intractability of Not Worst Responding

**arXiv ID:** 2602.10966 | [PDF](https://arxiv.org/pdf/2602.10966v1)

**作者:** Mete Şeref Ahunbay `[一作]` (University of Oxford), Bassel Tarbush `[通讯]` (University of Oxford)

**通讯引用:** 127 | [OpenAlex ID](https://openalex.org/A5052991579)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究在游戏中将传统的“最佳响应”弱化为“非最坏响应”这一最弱的理性要求，并探讨其在纯策略游戏中的计算可行性。

**💡 创新点**

创新点在于：①提出并系统分析了“非最坏响应”概念；②证明在一般游戏及势游戏中，该概念对应的决策、搜索与计数问题均与纯纳什平衡等价且在相同的复杂度类（NP、PLS、#P）内完成；③揭示了最小理性保证与计算难度的内在联系，并给出了在部分玩家满足约束时的可计算性阈值α、β。

**🔧 技术方法**

主要技术包括：布尔电路游戏模型、可多项式约简、Lovász Local Lemma与概率方法、潜在函数映射、PLS和#P完备性证明等。

**📊 数据集**

本研究并未使用现实数据集，所用的游戏实例均为通过布尔电路构造的人工游戏。

**📈 对比分析**

与已知的纯纳什平衡复杂度相比，本文证明了相同的硬度；在非全局约束下，作者提出了随机化算法，成功率至少为1/2。

**⚠️ 局限性**

局限性：仅讨论纯策略情况，未涉及混合策略或随机化学习动态；只给出了最坏情况的复杂度，未探讨平均情况或实际游戏的可行性。

---

## 468. SplitCom: Communication-efficient Split Federated Fine-tuning of LLMs via Temporal Compression

**arXiv ID:** 2602.10564 | [PDF](https://arxiv.org/pdf/2602.10564v1)

**作者:** Tao Li `[一作]` (University of Hong Kong), Xianhao Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 1837 | [OpenAlex ID](https://openalex.org/A5083484070)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 SplitCom——一种面向大语言模型的通信高效分布式微调框架。

**💡 创新点**

创新点包括：利用激活时间冗余的相似度感知上传机制；两种自适应阈值控制策略（bang‑bang 与 DDPG 强化学习）；以及 U‑shape 架构将损失计算迁移至客户端，进一步压缩梯度并保障标签隐私。

**🔧 技术方法**

核心技术包括 Split Federated Learning、LoRA 参数高效微调、余弦相似度激活重用、随机投影降维、DDPG 强化学习阈值调节、INT8 量化兼容；实现基于 PyTorch、gRPC 的分布式系统。

**📊 数据集**

在 GPT‑2 Small 与 GPT‑2 XLarge 两个模型上，分别在 E2E、DART 与 WebNLG 三大 NLG 基准数据集上进行评估。

**📈 对比分析**

与 SplitLoRA、固定阈值、BBC、DDPG 等基线对比：标准 SplitCom 端到端激活上传量可降低约 98.6%（U‑shape 95.8%），模型生成性能保持与基线相当或略优；与 INT8 量化组合后，通信量可进一步压缩至 90%+；实验显示在保持 BLEU、METEOR 等指标的同时，通信成本显著下降。

**⚠️ 局限性**

限制与挑战：小模型在 INT8 量化下易出现性能下降；阈值区间需手动调节；实验仅在 IID 数据划分下进行；缺乏对更大模型（如 Llama‑3）或多模态任务的验证；梯度量化与压缩的进一步优化尚未探索。

---

## 469. GeoGR: A Generative Retrieval Framework for Spatio-Temporal Aware POI Recommendation

**arXiv ID:** 2602.10411 | [PDF](https://arxiv.org/pdf/2602.10411v1)

**作者:** Fangye Wang `[一作]` (AMAP Alibaba Group), Pengjie Wang `[通讯]` (AMAP Alibaba Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GeoGR框架，基于生成式推荐与大语言模型，对POI进行语义ID（SID）化，并在此基础上实现下一目标地点的生成式预测。

**💡 创新点**

创新点包括：① geo-aware SID tokenization 通过地理约束共访对、对比学习、RQ‑Kmeans 与 EM‑style 迭代细化，显著提升SID语义与空间协同；② 多阶段 LLM 对齐（CPT+SFT），使非本地化 SID 与 LLM 语义空间兼容，并实现条件感知的自回归 POI 生成。

**🔧 技术方法**

使用技术包括：Qwen‑4B 嵌入与语言模型、对比学习、R‑Kmeans 量化、EM‑style SID 迭代优化、持续预训练（CPT）+ 监督微调（SFT）、自回归 beam‑search 生成、基于时空上下文的多模态输入。

**📊 数据集**

数据集涵盖公开 Foursquare NYC、TKY 两个小规模数据集，以及阿里地图 700 万用户、340 万 POI 的大规模工业数据集。

**📈 对比分析**

在离线指标（Recall@K、NDCG@K）上与传统顺序模型、Transformer 基线、TIGER、GNPR‑SID、OneLoc 等对比，GeoGR 均提升约 4–5%（相对提升）并在在线 A/B 测试中提升 WinRate +2.3%、CTR +1.5% 等业务指标。

**⚠️ 局限性**

局限性包括：① 对大模型硬件需求高，推理延迟仍受限；② SID 生成与对齐流程复杂，部署成本较高；③ 目前仅在导航型 LBS 场景验证，跨域泛化与解释性尚待进一步研究。

---

## 470. Morphogenetic Assembly and Adaptive Control for Heterogeneous Modular Robots

**arXiv ID:** 2602.10561 | [PDF](https://arxiv.org/pdf/2602.10561v1)

**作者:** Chongxi Meng `[一作]` (Shanghai Research Institute for Intelligent Autonomous Systems), Bin He `[通讯]` (Shanghai Research Institute for Intelligent Autonomous Systems)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

本工作提出了一个闭环自动化框架，实现了异质模块化机器人的形态组装与自适应控制，涵盖从模块抓取到成形再到实时运动控制的完整流程。

**💡 创新点**

创新点包括：①层次化规划器，利用双向启发式搜索与类型惩罚解耦离散配置与连续运动；②GPU加速的变异降温MPPI控制器，实现形态无关的实时运动生成；③完整闭环实现，首次在同一系统中同步完成组装、合并分离与动态控制。

**🔧 技术方法**

采用的技术包括：双向启发式搜索与类型惩罚、A*低层执行规划、基于MPPI的采样式模型预测控制、变异降温策略、GPU并行物理仿真以及模块化机器人构造与抓取算法。

**📊 数据集**

实验数据集为程序化生成的重构任务集合，规模覆盖20–75个模块，设置不同重叠率与异质比例，共对每种组合生成100个随机实例进行测试。

**📈 对比分析**

通过与Hungarian启发式、Greedy启发式以及标准MPPI进行对比，结果显示：①加入类型惩罚后成功率>80%；②Greedy启发式比Hungarian生成更低执行成本；③变异降温MPPI在速度跟踪上误差显著降低、控制频率提升至50Hz（相较标准MPPI仅约3Hz）。

**⚠️ 局限性**

局限性在于仅在仿真环境中验证，缺乏实机验证与Sim-to-Real迁移，依赖GPU加速，且在极大规模（>75模块）或高度异质配置下的可扩展性尚待进一步研究。

---

## 471. Causal-Informed Hybrid Online Adaptive Optimization for Ad Load Personalization in Large-Scale Social Networks

**arXiv ID:** 2602.10129 | [PDF](https://arxiv.org/pdf/2602.10129v1)

**作者:** Aakash Mishra `[一作]` (Meta), Ren Mao `[通讯]` (Meta)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种结合原始-对偶方法和贝叶斯优化的混合在线自适应优化框架（CTRCBO），用于大规模社交网络中的广告负载个性化。

**💡 创新点**

创新点在于：①将原始-对偶方法的约束满足稳健性与贝叶斯优化的探索效率相结合；②使用置信区间（trust‑region）更新提升算法收敛稳定性；③利用因果机器学习模型为高斯过程回归（GPR）先验提供信息，使代理模型更具预测性；④在实际千亿级用户场景中实现在线实时调优与AB测试。

**🔧 技术方法**

核心技术包括：原始-对偶优化、贝叶斯优化（Gaussian Process Regression）、trust‑region 更新、因果ML估计、在线AB实验与性能监控。

**📊 数据集**

使用的是一款千亿级社交网络平台的真实用户数据（包含广告曝光、点击、转化等多维度日志）。

**📈 对比分析**

与传统单一原始-对偶或单纯贝叶斯优化方法相比，CTRCBO在实验中展示了更快的收敛速度、更稳健的约束满足（如用户体验指标不下降）以及更高的个性化转化率；在线AB测试进一步验证了其在生产环境中的有效性。

**⚠️ 局限性**

局限性包括：①高斯过程回归在极高维度下仍受限，代理模型质量对收敛影响显著；②trust‑region参数需要经验调优，可能对不同业务场景不通用；③该方法目前主要针对广告负载调优，未充分验证在其他类型的资源分配问题上的通用性；④在大规模部署时仍存在计算和存储开销，需要进一步优化。

---

## 472. SoftMatcha 2: A Fast and Soft Pattern Matcher for Trillion-Scale Corpora

**arXiv ID:** 2602.10908 | [PDF](https://arxiv.org/pdf/2602.10908v1)

**作者:** Masataka Yoneda `[一作]` (University of Tokyo), Sho Yokoi `[通讯]` (National Institute for Japanese Language and Linguistics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在兆级乃至万亿级语料库上实现软匹配（支持替换、插入、删除）的极快全文检索算法。

**💡 创新点**

核心创新包括磁盘感知的分阶段后缀数组设计、基于自然语言分布的动态语料库剪枝，以及改进的词向量相似度度量。

**🔧 技术方法**

算法基于分词、词向量（GloVe、FastText、Moses）、后缀数组索引、动态剪枝、以及对查询相似度阈值的自适应调节。

**📊 数据集**

主要使用FineWeb‑Edu（1.4T token）以及多语言C4语料（中文 38B、日文 169B 等），并在多语言测试集上验证。

**📈 对比分析**

与 infini‑gram、infini‑gram mini 及 SoftMatcha 比较，95% 分位搜索延迟在 1.4T 语料下低于 300 ms，速度比现有方法快 10–50 倍，且可扩展到更大规模。

**⚠️ 局限性**

局限包括仅按词级匹配，无法捕捉跨词语义变形；需要随语料规模扩展的磁盘与内存资源；以及对多词合成表示的处理仍不充分。

---

## 473. Modular Multi-Task Learning for Chemical Reaction Prediction

**arXiv ID:** 2602.10404 | [PDF](https://arxiv.org/pdf/2602.10404v1)

**作者:** Jiayun Pang `[一作]` (University of Greenwich), Ivan Vulić `[通讯]` (University of Cambridge)

**通讯引用:** 6456 | [OpenAlex ID](https://openalex.org/A5014866912)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了低秩适配器（LoRA）在化学反应预测中的参数高效微调，比较其与传统全微调在多任务（正向预测、逆合成、试剂预测）上的表现，重点验证其在复杂C‑H功能化反应中的效果与灾难性遗忘抑制。

**💡 创新点**

创新点在于将LoRA与多任务T5/ByT5模型结合，形成模块化、可切换的反应特定适配器；展示LoRA在保持多任务性能、降低灾难性遗忘、捕捉反应特异性反应性模式方面优于全微调。

**🔧 技术方法**

使用LoRA技术、T5/ByT5和nach0基础模型、正向预测/逆合成/试剂预测的多任务学习框架，以及对LoRA超参数（r、α、dropout）进行限定搜索。

**📊 数据集**

使用通用有机反应数据集USPTO_1K_TPL（约445k条、1000类）以及经过筛选的C‑H Borylation数据集（685条）。

**📈 对比分析**

通过Acc@1/2/3/5排名准确率、Cliff’s delta和Wilcoxon检验比较全微调与LoRA；LoRA在单任务上与全微调相当或略优，在多任务上保持80%+准确率并显著减少灾难性遗忘；在C‑H功能化正向预测中Acc@1提升至约78–80%，相较全微调仅差几个百分点。

**⚠️ 局限性**

局限性包括：仅在旧版T5/ByT5模型上验证，缺乏对最新大规模LLM的评估；LoRA在某些反应类型仍未能完全捕捉微妙选择性；超参数搜索受限，可能未达最优；实验验证对化学解释的深入探讨不足。

---

## 474. Multi-Environment MDPs with Prior and Universal Semantics

**arXiv ID:** 2602.10938 | [PDF](https://arxiv.org/pdf/2602.10938v1)

**作者:** Benjamin Bordais `[一作]` (Université Libre de Bruxelles), Jean-François Raskin `[通讯]` (Université Libre de Bruxelles)

**通讯引用:** 5346 | [OpenAlex ID](https://openalex.org/A5050196522)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究多环境马尔可夫决策过程（MEMDP），对比先验（prior）与通用（universal）语义，并提出求解先验值及其逼近、间隙问题的算法。

**💡 创新点**

1) 证明先验值与通用值之间的对应关系：通用值等于所有先验值的下确界；2) 对于奇偶性目标给出了可在 PSPACE（概率以单词表示）或 EXPSPACE（概率以二进制表示）内逼近先验值的空间高效算法；3) 将 MEMDP 与非递增熵的 POMDP 等价，扩展了 MEMDP 的适用范围；4) 对先验值的 1-Lipschitz 性做出新证明，并利用 Hoeffding 等工具处理环境信念的更新。

**🔧 技术方法**

主要技术包括：贝叶斯信念更新与截断、递归分治、极化与 1-Lipschitz 连续性、Hoeffding 不等式改造、熵不递增性证明、混合策略与 Von Neumann–Minimax 定理的应用、对两环境 MEMDP 进行二分搜索求最小先验值、以及从 Dirac‑preserving POMDP 构造等价 MEMDP 的指数时间算法。

**📊 数据集**

论文未使用实际实验数据集，而是基于形式化模型（MEMDP、POMDP）进行理论分析和算法复杂度证明。

**📈 对比分析**

通过与之前的双指数空间算法对比，所提出的先验间隙问题算法在概率以单词表示时降为 PSPACE，在二进制表示时为 EXPSPACE，显著提高了复杂度；同时，通过先验-通用值关系，可将通用间隙问题的复杂度也降至 PSPACE/EXPSPACE。

**⚠️ 局限性**

限制包括：算法仍属于高阶空间复杂度（PSPACE/EXPSPACE），对大规模问题不具备多项式时间；仅针对奇偶性目标和有限环境数；对于一般 POMDP 的通用性有限，仍存在不可判定性问题；截断和逼近过程中需要对先验空间进行离散化，可能导致指数级的状态扩展。

---

## 475. Spectral-Spatial Contrastive Learning Framework for Regression on Hyperspectral Data

**arXiv ID:** 2602.10745 | [PDF](https://arxiv.org/pdf/2602.10745v1)

**作者:** Mohamad Dhaini `[一作]` (Univ Rouen Normandie), Antonin Van Exem `[通讯]` (Tellux)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了面向遥感光谱-空间对比学习的回归框架，能够兼容3D CNN和Transformer骨干网络。

**💡 创新点**

创新点在于将光谱与空间两类数据增强相结合的对比学习策略，并提供了专门针对高光谱数据的增广工具箱。

**🔧 技术方法**

采用自监督对比学习、3D CNN/Transformer特征提取、光谱与空间增广以及回归损失与对比损失的联合训练。

**📊 数据集**

在合成混合端元数据和真实Samson高光谱数据集上进行实验。

**📈 对比分析**

与仅使用回归损失的基线模型对比，使用R²和MAE评估，光谱+空间对比学习方案在所有骨干网络上均取得显著提升。

**⚠️ 局限性**

局限性包括模型尤其是Transformer的计算复杂度较高，实验仅覆盖有限的数据集，未验证跨域迁移性能。

---

## 476. Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs

**arXiv ID:** 2602.10388 | [PDF](https://arxiv.org/pdf/2602.10388v1)

**作者:** Zhongzhi Li `[一作]` (University of Georgia), Ninghao Liu `[通讯]` (Polyu)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于稀疏自编码器的Feature Activation Coverage（FAC）度量和FAC Synthesis覆盖驱动的数据合成框架，用于在LLM后训练阶段生成覆盖缺失任务相关特征的合成样本。

**💡 创新点**

通过在模型内部特征空间量化多样性并以缺失特征为目标进行覆盖驱动的合成，实现了可解释的跨模型共享特征空间和高效的数据生成。

**🔧 技术方法**

利用稀疏自编码器（SAE）提取可解释特征，结合PAC-Bayesian分析与KL距离约束降低分布与采样误差，并采用两步对比样本引导的生成策略。

**📊 数据集**

在毒性检测、奖励建模、行为引导与指令跟随四个公开基准上评估，并使用anchor语料库与多模型生成器进行实验。

**📈 对比分析**

与Alpaca、Evol-Instruct、Magpie、CoT-Self-Instruct、SAO、Prismatic、SynAlign等基线对比，FAC Synthesis在四个任务上均取得显著提升；在指令跟随任务中仅用2000条合成样本即可匹敌MAGPIE的30万样本。

**⚠️ 局限性**

对高度分布式的推理特征仍难以捕捉，生成器对模型和任务的依赖性较强，且在极端温度或阈值设置下可能产生噪声样本。

---

## 477. A Jointly Efficient and Optimal Algorithm for Heteroskedastic Generalized Linear Bandits with Adversarial Corruptions

**arXiv ID:** 2602.10971 | [PDF](https://arxiv.org/pdf/2602.10971v1)

**作者:** Sanghwa Kim `[一作]` (KAIST), Se-Young Yun `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种高效的自适应算法（HCW-GLB-OMD），用于处理自协方差且存在异方差的广义线性分支问题，并在面对自适应对抗性损坏时仍能保持低回报损失。

**💡 创新点**

创新点在于：1) 结合 Hessian 加权的自适应置信度，构造在线估计器，避免了传统批量最大似然的高复杂度；2) 在对抗性损坏场景下实现了实例级最优的回报损失，上界与下界相匹配（仅在损坏项上多一个 κ 因子）。

**🔧 技术方法**

主要技术包括：在线镜像下降（OMD）更新、基于自协方差的自适应置信加权、混合损失与Ville不等式的自监督证明、以及新的自界限不等式用于控制损坏项。

**📊 数据集**

该工作为理论研究，无使用具体数据集；所有结果均基于概率模型与下界构造。

**📈 对比分析**

与现有的线性/逻辑/泊松 GLB 以及带损坏的线性分支算法比较，HCW-GLB-OMD 在计算复杂度（每轮 O(1) 时空）和回报损失上实现了最优或接近最优的性能，且不需要批量估计。

**⚠️ 局限性**

局限性：1) 需要已知并观测到每一步的方差参数 τ_t；2) 在损坏项的上界中仍保留了 κ 乘子，实际是否可进一步去除尚待研究；3) 仅在自协方差假设下证明，扩展到更广泛的 GLM 需要进一步工作。

---

## 478. PELLI: Framework to effectively integrate LLMs for quality software generation

**arXiv ID:** 2602.10808 | [PDF](https://arxiv.org/pdf/2602.10808v1)

**作者:** Rasmus Krebs `[一作]` (Copenhagen Business School), Somnath Mazumdar `[通讯]` (Copenhagen Business School)

**通讯引用:** 549 | [OpenAlex ID](https://openalex.org/A5063602818)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过提出Programmatic Excellence via LLM Iteration (PELLI) 框架，对五大流行LLM在三类应用领域（高性能计算、机器学习、数据处理）下生成的Python代码进行迭代式分析，评估其在可维护性、性能和可靠性三大非功能指标上的表现。

**💡 创新点**

创新点包括：①引入迭代式质量评估框架PELLI；②在同一研究中同时考量三大非功能指标；③使用多长度（短、中、长）提示策略以及跨领域的实验设计，系统揭示提示与领域对LLM代码质量的影响。

**🔧 技术方法**

采用GPT‑4T、Claude‑2、Bard、CodeLlama、Llama2‑70B等LLM，配合Python静态分析工具Pylint、Radon、psutil，使用平滑、归一化、标准化等预处理方法进行指标量化。

**📊 数据集**

使用自定义的九个代表性算法（快速排序、Strassen、Monte‑Carlo、Attention、卷积、PCA、Huffman、PageRank、Rabin‑Karp）作为代码生成任务，构成实验数据集；无公开基准集，仅基于这些算法实现进行评估。

**📈 对比分析**

通过在每个LLM、提示长度和领域下执行多次并收集11项指标，进行平滑、归一化、标准化后进行量化比较；结果显示GPT‑4T略优，所有LLM总体可达或超过人工基线，但不同LLM、不同提示和不同领域之间仍存在显著差异。

**⚠️ 局限性**

局限性包括：仅评估Python语言与五个LLM；缺少动态学习与多语言验证；未覆盖更细粒度的质量维度（如安全性）；评估工具与提示策略可能引入偏差；LLM技术快速演进可能导致结果短期失效。

---

## 479. Resource-Efficient RGB-Only Action Recognition for Edge Deployment

**arXiv ID:** 2602.10818 | [PDF](https://arxiv.org/pdf/2602.10818v1)

**作者:** Dongsik Yoon `[一作]` (HDC LABS), Dayeon Lee `[通讯]` (HDC LABS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种专为边缘设备设计的RGB-only动作识别网络。

**💡 创新点**

创新点在于将X3D骨干+Temporal Shift与可参数化的Universal Inverted Bottleneck、Ghost点卷积、选择性时序适配以及无参数SimAM融合，显著降低参数与引擎体积，同时保持高精度。

**🔧 技术方法**

采用了X3D风格骨干、Temporal Shift模块、Universal Inverted Bottleneck（UIB）、Ghost 3D点卷积、Selective Temporal Adaptation（TAdaConv）、SimAM注意力、Poly-1交叉熵损失等技术。

**📊 数据集**

在NTU RGB+D 60和120两个基准数据集上进行评估。

**📈 对比分析**

与现有RGB-only方法（如EPAM-Net、DVANet等）对比，取得95.21%/98.31%（NTU60）和90.88%/92.67%（NTU120）的准确率，在Jetson Orin Nano上FP16 TensorRT推理时帧率10.3 FPS、引擎尺寸5.3 MB、参数0.96 M，展示了优异的精度-效率平衡。

**⚠️ 局限性**

局限性在于推理吞吐量受TensorRT执行特性限制，轻量化、分散的算子导致核启动开销高、算子融合受限，整体算术强度低，导致相较于更大模型的FPS略低。

---

## 480. AI-rithmetic

**arXiv ID:** 2602.10416 | [PDF](https://arxiv.org/pdf/2602.10416v1)

**作者:** Alex Bie `[一作]` (Google), Sergei Vassilvitskii `[通讯]` (Google)

**通讯引用:** 11787 | [OpenAlex ID](https://openalex.org/A5070795618)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对前沿大型语言模型进行大规模整数加法实验，系统分析错误类型并量化误差来源。

**💡 创新点**

发现模型错误主要由对齐错误和临近进位错误两类组成，且对齐错误与tokenization的周期性特征相关，进位错误呈现独立几何分布。

**🔧 技术方法**

采用自动化答案提取、误差分类、频谱与傅里叶分析以及几何随机模型等技术。

**📊 数据集**

使用随机生成的两数相加数据集，长度从1到100，每位数长度随机生成100个样本，覆盖多种前沿模型。

**📈 对比分析**

通过比较各模型随位数变化的准确率曲线（含10点滑动平均）和误差分布，发现即使是大模型在数位增大时误差率急剧上升；GPT‑5误差主要为对齐错误，Gemini 2.5 Pro和Claude Opus 4.1误差多为对齐或临近进位。

**⚠️ 局限性**

仅聚焦整数加法，未扩展到减法、乘法等；误差归因基于经验规则，未深入探究模型内部机制；结果可能受提示语和输入格式影响，缺乏对不同训练策略或工具使用的系统评估。

---

## 481. Enhancing Weakly Supervised Multimodal Video Anomaly Detection through Text Guidance

**arXiv ID:** 2602.10549 | [PDF](https://arxiv.org/pdf/2602.10549v1)

**作者:** Shengyang Sun `[一作]` (Hangzhou Dianzi University), Xiaojin Gong `[通讯]` (Zhejiang University)

**通讯引用:** 2633 | [OpenAlex ID](https://openalex.org/A5061298101)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了多阶段文本增强（MSTA）机制和多尺度瓶颈Transformer（MSBT）框架，用于弱监督多模态视频异常检测；

**💡 创新点**

创新点在于利用in‑context学习自动生成高质量异常文本样本以平衡数据，再通过MSBT实现高效、逐层压缩的模态融合与加权；

**🔧 技术方法**

采用大语言模型（LLaMA-3-8B）进行文本摘要、伪标签生成与异常文本合成，BERT作为文本特征提取器，I3D和VGGish提取RGB/光流/音频特征，并构建Transformer‑based multimodal融合网络；

**📊 数据集**

在UCF‑Crime（含文本注释）和XD‑Violence（使用BLIP2生成文本）两个公开大规模数据集上进行实验；

**📈 对比分析**

与现有SOTA方法（如TEVAD、DAR、Tan等）比较，使用RGB+Flow+Text组合时在UCF‑Crime上取得AUC 89.67%，在XD‑Violence上取得AP 85.92%，显著超越前沿模型；

**⚠️ 局限性**

局限性包括对大型语言模型的高计算成本、对文本生成质量的依赖以及在无文本注释数据集上的泛化受限；

---

## 482. PRISM: Differentially Private Synthetic Data with Structure-Aware Budget Allocation for Prediction

**arXiv ID:** 2602.10228 | [PDF](https://arxiv.org/pdf/2602.10228v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Bradley A. Malin `[通讯]` (Vanderbilt University Medical Center)

**通讯引用:** 11448 | [OpenAlex ID](https://openalex.org/A5090647314)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 PRISM 机制，针对已知或可推断的因果/图结构，在差分隐私下生成合成数据，并将隐私预算聚焦在预测任务相关的统计量上；

**💡 创新点**

创新点在于：①构建三种结构知识层次（因果、图形、预测）并据此确定特征子集；②利用任务权重和闭式最优分配实现预算精确分配；③把工作负载设计与下游预测风险上界紧密关联，提供理论与实证双重证明；

**🔧 技术方法**

技术手段包括：差分隐私（高斯机制、指数机制）、PGM 生成（Private‑PGM）、DP 近似互信息/特征选择、闭式预算分配公式、实验使用 TSTR、ROC‑AUC 评估；

**📊 数据集**

实验数据集：半合成的结构因果模型（包含因果父母、马尔可夫毯和噪声变量）和实际的 Adult 收入数据集；

**📈 对比分析**

对比方法包括：MST、PrivBayes、PrivSyn、以及基于相关系数的 oracle top‑k；在因果偏移场景下 PRISM‑Causal 维持 AUC≈0.73（接近非私有上界），而相关系数选取仅 0.49；在边缘偏移场景下 PRISM‑Graphical 接近完美 AUC；成人数据上 PRISM‑Predictive 在相同隐私下显著优于 MST/PrivBayes；

**⚠️ 局限性**

局限性：需要已知或可推断的因果/图结构，特征选择和互信息估计会消耗预算且在小样本时误差较大；离散化导致高基数或连续特征难以处理；仅适用于单一预测目标，无法直接支持多目标或迭代查询；在低维度/大样本情形下预算分配提升有限。

---

## 483. Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis

**arXiv ID:** 2602.10881 | [PDF](https://arxiv.org/pdf/2602.10881v1)

**作者:** Zhiyin Tan `[一作]` (L3S Research Center Leibniz University Hannover), Jennifer D'Souza `[通讯]` (TIB Leibniz Information Centre for Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了大型语言模型（LLM）在系统综述与元分析证据提取中的结构化性能，并提出了分层查询与诊断框架，系统评估单属性提取、多属性绑定和统计推导三类任务的表现。

**💡 创新点**

创新点在于：① 将元分析提取任务拆解为递进的结构化查询层级；② 统一语义研究模式与多层次查询，能够精准定位模型在关系绑定和数值归属上的弱点；③ 对五个跨学科领域进行手工标注并公开评测基准，提供可复现的结构化评测流程。

**🔧 技术方法**

使用的技术包括：GPT‑5.2 与 Qwen3‑VL 两大 LLM；将全文PDF转换为 Markdown 结构化文本；对单篇与全域两种输入模式进行对照；采用 tuple‑level precision/recall 与数值一致性评估；对结果进行错误模式分析。

**📊 数据集**

数据集：五个领域（土木工程、医学健康、农业科学、地球环境科学、社会科学）各自的原始研究论文，手工标注了研究人群、样本量、变量、统计方法、效应量等结构化属性，共计数千篇文献。

**📈 对比分析**

比较方法：在 Per‑Paper 与 Global 两种输入模式下，对 O1、O2、M1、M2 等任务组分别计算 F1。单属性任务 F1 约 0.6–0.8，L2 绑定任务显著下降（0.2–0.3），最高阶任务几乎 0；Global 输入导致更大性能损失。总体表现说明 LLM 在高结构性任务上仍有限。

**⚠️ 局限性**

局限性：LLM 在角色归属、跨分析绑定漂移、结果密集导致实例压缩以及聚合误差放大方面表现不佳，无法保持稳定的关系绑定与数值归属，因而难以满足严格的元分析需求。

---

## 484. Pupillometry and Brain Dynamics for Cognitive Load in Working Memory

**arXiv ID:** 2602.10614 | [PDF](https://arxiv.org/pdf/2602.10614v1)

**作者:** Nusaibah Farrukh `[一作]` (Digital University Kerala), Elizabeth Sherly `[通讯]` (Digital University Kerala)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文研究了利用脑电（EEG）和瞳孔测量（pupillometry）对工作记忆任务中的认知负荷进行分类，并提出了可解释、轻量化的多模态处理流程。

**💡 创新点**

创新点在于将Catch‑22时间序列特征与树模型结合，实现高效且可解释的认知负荷识别，证明瞳孔动力学可作为EEG的可穿戴替代方案。

**🔧 技术方法**

采用的技术包括Catch‑22特征提取、SHAP解释、随机森林、XGBoost、SVM以及轻量化CNN、LSTM和Transformer等模型。

**📊 数据集**

使用的数据集为OpenNeuro Digit Span任务数据集，包含64通道EEG和可穿戴瞳孔跟踪信号。

**📈 对比分析**

在二分类和多分类任务中对比树模型与深度网络，XGBoost在EEG上达约57%准确率、在瞳孔上达约61%准确率，且在资源与推理速度上明显优于深度模型。

**⚠️ 局限性**

局限性包括难以准确识别“仅听”低负荷状态、对光照和佩戴差异敏感、EEG受硬件限制，以及缺乏跨模态融合与域适应来提升在真实环境中的鲁棒性。

---

## 485. When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models

**arXiv ID:** 2602.10179 | [PDF](https://arxiv.org/pdf/2602.10179v1)

**作者:** Jiacheng Hou `[一作]` (Tsinghua University), Alex Jinpeng Wang `[通讯]` (Central South University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5061458314)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了视觉到视觉的 jailbreak 攻击，针对大规模图像编辑模型通过视觉提示实现恶意编辑，并构建了首个针对图像编辑安全的 benchmark；

**💡 创新点**

创新点在于：①首次将攻击媒介从文本转向纯视觉，揭示了视觉安全对齐缺陷；②设计了层级化风险分类和 1054 张视觉提示样本的 VJAS benchmark；③提出无训练的 introspection 反向推理防御方案；

**🔧 技术方法**

采用视觉提示嵌入、多模态推理、内置安全触发文本、MLLM 作为评判器以及对抗实验；

**📊 数据集**

使用自构建的 VJAS benchmark（15 类风险、116 属性、9 动作，共 1054 张视觉提示图像），并在公开的商业与开源图像编辑模型上进行评测；

**📈 对比分析**

在 7 个主流模型上实验，攻击成功率平均 85.7%，单模型最高可达 97.5%；防御后 ASR 降低约 33%，安全性与主流商用系统相当；

**⚠️ 局限性**

局限性：对视觉感知和推理能力弱的模型攻击效果有限；防御依赖预先对齐的 VLM，难以对抗伪造或误导性信息。

---

## 486. Benchmarking Large Language Models for Knowledge Graph Validation

**arXiv ID:** 2602.10748 | [PDF](https://arxiv.org/pdf/2602.10748v1)

**作者:** Farzad Shami `[一作]` (Aalto University), Gianmaria Silvello `[通讯]` (University of Padua)

**通讯引用:** 1450 | [OpenAlex ID](https://openalex.org/A5078254809)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了面向知识图谱（KG）事实验证的Benchmark（FactCheck），系统性评估大语言模型（LLM）在内部知识、检索增强生成（RAG）和多模型共识三种维度下的验证性能；

**💡 创新点**

创新点在于：①构建大规模RAG数据集（2M+文档）并提供Mock API；②设计统一的验证流程与多模型投票机制；③公开完整的Web可视化平台与实验代码；

**🔧 技术方法**

主要技术包括：大语言模型（Gemma2、Qwen2.5、Llama3.1、Mistral、GPT‑4o mini），RAG检索流水线（查询生成、检索、筛选、chunking），多模型共识投票与Tie‑breaking，性能度量（F1、Consensus Alignment、耗时），以及实验监控（OpenTelemetry）；

**📊 数据集**

使用三大真实KG数据集：FactBench、YAGO、DBpedia，并基于这些KG生成的事实三元组作为验证目标；

**📈 对比分析**

比较方法包括内部知识直接验证（DKA）、引导式迭代验证（GIV-Z、GIV‑F）和RAG；实验显示RAG在大多数情况提升F1（最高可达≈0.91）但耗时约10×；多模型共识可稳定结果但未必超越最佳单模型；在YAGO等极不平衡数据上模型倾向于全真预测；

**⚠️ 局限性**

局限性在于：①数据集不均衡导致F1(F)极低；②RAG检索成本高且依赖外部API；③模型存在幻觉与偏差，尤其在内部知识验证时；④不同KG结构导致检索难度差异；⑤实验主要基于英文文本，跨语言能力待验证。

---

## 487. On the Use of a Large Language Model to Support the Conduction of a Systematic Mapping Study: A Brief Report from a Practitioner's View

**arXiv ID:** 2602.10147 | [PDF](https://arxiv.org/pdf/2602.10147v1)

**作者:** Cauã Ferreira Barros `[一作]` (Federal University of Goiás), Valdemar Vicente Graciano Neto `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用ChatGPT‑4自动完成系统性映射研究的标题/摘要筛选和数据提取，并与完全人工流程对比，记录时间、准确率及提示工程经验。

**💡 创新点**

首次提供完整端到端LLM支持的系统性映射经验报告，量化时间节省与准确率，并系统阐述提示工程、幻觉检测与人工验证的协同工作流程。

**🔧 技术方法**

采用ChatGPT‑4进行自动化筛选与结构化数据提取，并对比Gemini PRO、Manus、Copilot等模型的表现。

**📊 数据集**

复用先前SMS检索到的219篇文献进行筛选，13篇文献进行数据提取；在模型对比中分别使用50篇筛选样本和10篇提取样本。

**📈 对比分析**

通过对比人工（约30天）与LLM（约9+1小时）的执行时间，以及准确率（筛选95% vs 95%，提取92% vs 92%）验证效果；在模型对比中Gemini PRO与ChatGPT相当，Manus在筛选上更优。

**⚠️ 局限性**

样本量有限、实验顺序影响时间估计、模型幻觉与提示依赖、需人工验证导致效率提升受限，结果的普适性与可复制性有限。

---

## 488. Enhancing Ride-Hailing Forecasting at DiDi with Multi-View Geospatial Representation Learning from the Web

**arXiv ID:** 2602.10502 | [PDF](https://arxiv.org/pdf/2602.10502v1)

**作者:** Xixuan Hao `[一作]` (Hong Kong University of Science and Technology), Yuxuan Liang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5601 | [OpenAlex ID](https://openalex.org/A5018828723)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出MVGR-Net框架，结合POI与时空移动模式预训练地理表征，并通过Prompt与LoRA微调LLM实现打车需求预测

**💡 创新点**

创新在于多视角地理表征学习、Prompt驱动的LLM适配以及将外部事件嵌入预测

**🔧 技术方法**

使用双视角自注意力网络、跨视图注意力、Prompt生成网络、LoRA微调与文本描述的LLM编码

**📊 数据集**

采用DiDi 2023-2025年实测数据（Call、TSH，392县级区域，30min间隔）以及全国9.3亿POI数据

**📈 对比分析**

与PatchTST、DLinear、CrossFormer、ExoLLM、XGB、ARIMA、Weekly Counterpart比较，平均提升1.5-2.5%（WMAPE/MAE），线上AB测试亦显著提升

**⚠️ 局限性**

受限于可用历史数据量有限、对特殊事件的预测仍不够精准、LLM需大量算力与数据

---

## 489. Blind Gods and Broken Screens: Architecting a Secure, Intent-Centric Mobile Agent Operating System

**arXiv ID:** 2602.10915 | [PDF](https://arxiv.org/pdf/2602.10915v1)

**作者:** Zhenhua Zou `[一作]` (Tsinghua University), Zhuotao Liu `[通讯]` (Tsinghua University)

**通讯引用:** 1174 | [OpenAlex ID](https://openalex.org/A5045206037)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对现有基于大模型的移动助手进行系统安全审计，提出并实现了一套以Agent Universal Runtime Architecture为核心的安全Agent OS框架，彻底替代传统的Screen‑as‑Interface交互方式。

**💡 创新点**

创新点在于：①把Agent作为系统首要实体，采用Hub‑and‑Spoke结构将意图解析与任务执行解耦；②四层安全防护：身份鉴权（Global Agent Registry + Agent Identity Card）、语义防火墙、认知完整性检查（taint‑aware 内存 + 计划轨迹对齐）和可审计执行控制；③在硬件可信根（TEE）上实现可信启动与动态权限管控。

**🔧 技术方法**

关键技术包括：TEE‑backed Secure Boot、Global Agent Registry、Agent Identity Card、Semantic Firewall、Taint‑aware 内存管理、Plan‑Trajectory Alignment、Runtime Alignment Validator、Critical Node Interception、动态权限与会话令牌、日志完整性与可导出审计链。

**📊 数据集**

使用 MobileSafetyBench 作为评测基准，涵盖 80 个任务，区分低风险功能任务与高风险安全任务。

**📈 对比分析**

与 Doubao Mobile Assistant（Standard 与 Pro）在同一基准上对比，结果显示：任务成功率从约 75% 提升至 94.3%，攻击成功率从约 40% 降至 4.4%，平均任务延迟从约 450 秒降至 68 秒，展示了显著的安全性与效率提升。

**⚠️ 局限性**

局限性包括：仍受 LLM 模型本身的不确定性与幻觉影响；语义攻击的深度识别仍需改进；实现依赖硬件 TEE 与可信链，跨平台可移植性待验证；AIC 与权限规范的标准化尚未完成，后续需解决兼容性与生态落地问题。

---

## 490. From Natural Language to Materials Discovery:The Materials Knowledge Navigation Agent

**arXiv ID:** 2602.11123 | [PDF](https://arxiv.org/pdf/2602.11123v1)

**作者:** Genmao Zhuang `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 36616 | [OpenAlex ID](https://openalex.org/A5003442464)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了材料知识导航代理 MKNA，通过自然语言驱动的推理、检索、预测、结构生成和稳定性评估，实现自动化材料发现。

**💡 创新点**

创新点在于将文献自洽化的 Map‑Reduce 证据提取与 LLM 代码生成相结合，能够在缺失数据库字段时自动构造计算函数，并在单一闭环中从文本到实验可验证候选的完整流程。

**🔧 技术方法**

技术包括 LLM（GPT‑5‑mini）推理、LangChain 调度、Map‑Reduce 文献提取、自动代码生成、图卷积网络 CGCNN、M3GNet 结构优化、以及化学族保持的结构变异。

**📊 数据集**

使用的数据库有 Materials Project、AFLOW、OQMD、NOMAD，文献来源为 arXiv/论文；还构建了基于弹性数据的 Debye 温度标签集合。

**📈 对比分析**

与传统 RAG 或手工筛选相比，MKNA 的 Map‑Reduce 方案在准确率与覆盖度上更优；CGCNN 预测的 Debye 温度 RMSE ≈ 247 K，M3GNet 验证后得到 1500–1700 K 的高 Debye 温度稳定结构。

**⚠️ 局限性**

局限在于依赖弹性推导的 Debye 温度估计在强非线性材料上误差较大，未考虑合成可行性和实验验证，且目前仅适用于已存在的结构库。

---

## 491. What Does Preference Learning Recover from Pairwise Comparison Data?

**arXiv ID:** 2602.10286 | [PDF](https://arxiv.org/pdf/2602.10286v1)

**作者:** Rattana Pukdee `[一作]` (Carnegie Mellon University), Pradeep Ravikumar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8223 | [OpenAlex ID](https://openalex.org/A5053209283)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文从三元组数据出发，对偏好学习的理论基础进行系统研究，并提出了条件偏好分布（CPRD）的概念。

**💡 创新点**

创新点在于给出了CPRD可被布拉德利‑泰瑞模型表示的必要与充分条件，并证明BT学习目标等价于对CPRD的KL投影；同时揭示了样本效率受两因素——对比间隔和比较图连通度的影响。

**🔧 技术方法**

主要技术包括概率模型定义、条件独立性分析、KL投影解释、经验风险与Rademacher复杂度理论、以及实验中的神经网络拟合。

**📊 数据集**

实验使用的是合成数据，生成的上下文与响应均来自同一维度空间，通过两层神经网络构造真实评分函数。

**📈 对比分析**

与基线（原评分函数）对比，采用排名归一化的评分和连通度优化的负样本分布能显著提升准确率；实验结果验证了间隔越大、连通度越高时学习效果更好。

**⚠️ 局限性**

局限性在于理论假设的条件独立性和对真实评分函数已知的前提；连通度和间隔的实际估计与提升仍缺乏直接可操作的方法。

---

## 492. MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation

**arXiv ID:** 2602.10271 | [PDF](https://arxiv.org/pdf/2602.10271v1)

**作者:** Yongyue Zhang `[一作]` (Independent Researcher), Yaxiong Wu `[通讯]` (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于多模态块查询图的长文档检索增强生成框架MLDocRAG，能在多模态长文档中进行跨页面、跨模态的问答。

**💡 创新点**

创新点在于将文档扩展为生成可答查询的多模态块查询图，并通过查询中心化的图检索实现细粒度的跨模态和跨页证据聚合。

**🔧 技术方法**

使用了大型视觉-语言模型（LVLM）生成查询、BGE-m3编码器做向量检索、图数据库存储多模态块查询图，以及多跳图遍历和多模态融合生成。

**📊 数据集**

在MMLongBench-Doc和LongDocURL这两个多模态长文档问答基准上进行实验。

**📈 对比分析**

与文本检索、图像2文本、视觉-语言检索、页面级检索以及图知识图谱RAG等基线相比，MLDocRAG在两数据集上分别取得47.9%和50.8%的准确率，均为最高。

**⚠️ 局限性**

局限包括仅支持文本和图像两种模态、对查询生成质量高度依赖、构建大型图的计算成本较高。

---

## 493. Learning to Evict from Key-Value Cache

**arXiv ID:** 2602.10238 | [PDF](https://arxiv.org/pdf/2602.10238v1)

**作者:** Luca Moschella `[一作]` (Apple), Ozan Sener `[通讯]` (Apple)

**通讯引用:** 2636 | [OpenAlex ID](https://openalex.org/A5109463535)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于强化学习的KV缓存淘汰框架kvs，学习每个注意力头的排序策略以保留未来重要token

**💡 创新点**

将KV缓存淘汰转化为无偏排序任务，使用离线RL和全预算奖励实现无查询、无额外推理的轻量化策略

**🔧 技术方法**

使用Plackett-Luce排序模型、Gumbel-Softmax采样、REINFORCE+RLOO基线以及离线预计算的KV轨迹进行训练

**📊 数据集**

在RULER、OASST2-4k以及EleutherAI评估集（BoolQ、ARC、MMLU、HellaSwag、GovReport）进行实验

**📈 对比分析**

相较于启发式和注意力相关基线，kvs在多种缓存预算下保持更高准确率/更低困惑度，并在零样本下仍具优异性能

**⚠️ 局限性**

依赖离线生成的KV轨迹，未考虑头间协同，且对不同模型尺寸与训练数据分布的迁移性需进一步验证

---

## 494. Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries

**arXiv ID:** 2602.10997 | [PDF](https://arxiv.org/pdf/2602.10997v1)

**作者:** Zhanyu Guo `[一作]` (Zhejiang University), Shiyu Zhao `[通讯]` (Westlake University)

**通讯引用:** 4645 | [OpenAlex ID](https://openalex.org/A5052346042)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一个多任务强化学习框架 GEAR，能够在单一端到端策略中学习并执行多种高速度无人机机动（如翻滚、翻转、旋转、悬停），并支持通过组合基本动作实现复杂机动。

**💡 创新点**

创新点在于将 SO(2) 旋转对称性显式嵌入到策略网络（EMLP），利用 FiLM 层实现轻量级任务调制，同时采用多头评论器分离不同任务的价值估计，从而兼顾对称性优点与任务异质性需求。

**🔧 技术方法**

使用的技术包括 SO(2) 等变换神经网络（EMLP）、Feature‑wise Linear Modulation (FiLM)、多头 Critic、PPO 强化学习、IsaacGym 高保真仿真、Vicon 运动捕捉、数据随机化与 curriculum 学习等。

**📊 数据集**

数据来源为自行构建的高保真仿真环境（IsaacGym+物理引擎）以及实机实验中的 Vicon 传感器记录；未使用公开数据集，所有数据均由作者自行生成与采集。

**📈 对比分析**

与基线 SOTA MTRL 框架在四种基本机动任务上进行比较，GEAR 的成功率从 84% 提升至 99%+，平均误差下降 1–2 倍，训练时间比基线短约 1 小时；实机测试显示能通过组合基本动作完成 Power Loop、Barrel Roll、Multi‑Flip 等复杂机动，表现稳健。

**⚠️ 局限性**

局限性包括：对 SO(2) 对称性的严格约束在某些任务中可能限制表达能力，需手动调节；对外部扰动、动态障碍等复杂环境的鲁棒性尚未充分验证；仅测试了四种基础机动，复杂组合仍依赖人工指令；依赖高质量仿真与真实实验平台，部署成本较高。

---

## 495. Robust Semantic Transmission for Low-Altitude UAVs: Predictive Channel-Aware Scheduling and Generative Reconstruction

**arXiv ID:** 2602.10482 | [PDF](https://arxiv.org/pdf/2602.10482v1)

**作者:** Jijia Tian `[一作]` (Chinese University of Hong Kong), Pooi-Yuen Kam `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对低空无人机下行链路的有限带宽和不确定信道，提出了基于预测信道的语义传输框架，利用结构–纹理分离的变分自编码器（ST‑VAE）和生成式后置恢复，实现结构信息优先发送、纹理信息在缺失时通过生成模型补全。

**💡 创新点**

创新点在于：① 将语义表示分解为确定性结构流和随机纹理流，实现不同误差保护；② 采用轨迹驱动的 SNR 预测和通道感知的预测调度，提前分配可用时隙和样本预算；③ 通过条件先验（生成式模型）在接收端对缺失纹理进行补全，显著降低深度衰落导致的语义失真。

**🔧 技术方法**

使用技术包括：ST‑VAE 结构–纹理分离、深度联合源信道编码（DeepJSCC）、轨迹+历史信噪的神经预测器、块级调度算法、感知损失+KL 正则化、条件生成先验、模拟块衰落信道模型。

**📊 数据集**

实验数据集为 MS COCO 2017 验证集，图像中心裁剪为 256×256，归一化至 [-1,1]。

**📈 对比分析**

与 DeepJSCC、Uniform Scheduling（均匀预算）和 No Generation（无生成补全）三种基线进行对比；在平均实际 SNR 15 dB 时，所提方法 PSNR 达 32.8 dB，比 DeepJSCC 提升 5.6 dB，且比 Uniform Scheduling 提升约 3.3 dB；在预测误差 σ_err 0–10 dB 的情况下，性能保持稳定，显示出较强的鲁棒性。

**⚠️ 局限性**

局限性包括：仅针对图像语义传输，未验证视频或其他多模态；调度与生成模型依赖于轨迹与 SNR 预测精度；实验仅在仿真块衰落信道和 10 时隙窗口内验证，真实无人机环境下的实现与延迟、功耗等问题尚待进一步研究。

---

## 496. Multi-UAV Trajectory Optimization for Bearing-Only Localization in GPS Denied Environments

**arXiv ID:** 2602.11116 | [PDF](https://arxiv.org/pdf/2602.11116v1)

**作者:** Alfonso Sciacchitano `[一作]` (Naval Postgraduate School), Isaac Kaminer `[通讯]` (Naval Postgraduate School)

**通讯引用:** 4519 | [OpenAlex ID](https://openalex.org/A5088457398)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套基于信息理论的轨迹优化框架，利用固定视场无人机与水面平台协同，在 GPS‑失效环境下实现估计感知的航迹规划与目标定位，并通过多机协同与单机舵机系统对比实验验证其优势。

**💡 创新点**

① 将目标可观测信息量（PCRLB）作为优化目标，实现估计感知的轨迹规划；② 使用伯恩斯坦多项式参数化，使轨迹在离散化后仍保持动态可行性；③ 采用终端联邦 EKF（FKF）实现低带宽多机协同融合，避免实时通信负载。

**🔧 技术方法**

轨迹优化、伯恩斯坦多项式参数化、PCRLB 与信息矩阵分析、离散时间测量模型、Sigmoid 平滑 FOV 约束、扩展卡尔曼滤波（EKF）与终端联邦卡尔曼滤波（FKF）。

**📊 数据集**

主要使用仿真生成的飞行参数、测量噪声、目标/USV 位置信息等；未采用公开真实数据集。

**📈 对比分析**

通过仿真对比启发式“racetrack”路径与优化轨迹，以及单机舵机系统与多机固定视场系统。结果显示：优化轨迹将目标 PCRLB 降低 65%（误差从 3.77 m 降至 1.32 m），任务时间缩短 10%；5 架 FFOV 无人机可匹配甚至超过单机舵机系统的定位精度，同时实现更低的任务成本和更高的系统弹性。

**⚠️ 局限性**

仅在理想仿真环境验证，未考虑动态障碍、气象变化、通信延迟与失真；固定高度和简化动力学限制了对更复杂飞行场景的适用；终端联邦融合只能在任务结束后完成，无法提供实时协同反馈。

---

## 497. Traceable, Enforceable, and Compensable Participation: A Participation Ledger for People-Centered AI Governance

**arXiv ID:** 2602.10916 | [PDF](https://arxiv.org/pdf/2602.10916v1)

**作者:** Rashid Mushkani `[一作]` `[通讯]` (University of Montreal), Rashid Mushkani (University of Montreal)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Participation Ledger，构建可机器读取的参与影响图谱，实现参与贡献与AI系统更新、评估、部署和治理动作的可追溯、可执行、可补偿；

**💡 创新点**

创新点在于将参与证据标准、影响追踪、可重放的变更测试、可执行的能力凭证与可计量的参与积分统一纳入单一可审计框架；

**🔧 技术方法**

采用JSON‑LD与W3C PROV语义进行数据建模，利用JSON Schema实现机器校验，使用附加日志与数字签名保障不可篡改；

**📊 数据集**

基于四个城市AI项目（AIAI/Mid‑Space/LIVS、EVADIA+、AI‑EDI‑Space & Street Review、WeDesign+）的文档与公开数据，如提示、图像、标注、访谈等；

**📈 对比分析**

通过文档分析评估证据覆盖度（招聘路径、角色、中介、同意、补偿、影响链）并验证schema可验证性；未进行性能实验，主要以案例覆盖率和可审计性为评估指标；

**⚠️ 局限性**

局限性包括缺乏真实部署验证、对隐私与撤回的冲突、治理落实依赖组织与采购约束、可能产生监控与游戏风险、需进一步评估跨组织可操作性

---

## 498. To Reconfigure or Not to Reconfigure: Optimizing All-to-All Collectives in Circuit-Switched Photonic Interconnects

**arXiv ID:** 2602.10468 | [PDF](https://arxiv.org/pdf/2602.10468v1)

**作者:** Anchengcheng Zhou `[一作]` (Princeton University), Maria Apostolaki `[通讯]` (Princeton University)

**通讯引用:** 3180 | [OpenAlex ID](https://openalex.org/A5026774096)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对光学电路交换互连上的 all-to-all 通信，设计了一种矩阵表述的全局优化框架，联合规划拓扑序列和流量调度，动态决定何时重新配置。

**💡 创新点**

创新点在于：① 用邻接矩阵及其幂的求和方式将所有可能的拓扑与调度抽象成一个整体，获得全局最优下界；② 发现并利用“高度对称与高扩展性”拓扑序列的结构，构造接近全局最优的低成本方案；③ 通过简单的流分组与最短路径调度算法，显著降低了计算复杂度，避免了组合爆炸。

**🔧 技术方法**

技术方法包括：矩阵分解与幂运算、对称/扩展性拓扑生成（循环图、Circulant、Generalized Kautz）、冲突消除的分组调度、对不同规模、不同流量模型（均匀、随机、Zipf）进行仿真评估。

**📊 数据集**

实验使用 HTSim 包的仿真模型，考虑多种规模（n=8、16、32、64 GPU；k=1、2）和多种流量模型/大小（8–64 MB），无真实生产数据集，但覆盖了广泛的工作负载特征。

**📈 对比分析**

与 MixNet、FAST、DirectConnect、TopoOpt 等基线相比，在 8–64 GPU、k=1/2 的配置下，平均减少 39–47% 的总完成时间，最高可达 44%；在不同流量模型和重配置成本下保持优于或相当于最佳基线。

**⚠️ 局限性**

局限性：① 依赖于所有对称、无依赖的 all-to-all 通信，对高度不均匀（Zipf）或有数据依赖的工作负载效果下降；② 仅在选定的对称/扩展性拓扑区域内求解，可能无法达到真正的全局最优；③ 对极端重配置延迟或非常小/大规模网络的适用性需进一步验证。

---

## 499. Colorimeter-Supervised Skin Tone Estimation from Dermatoscopic Images for Fairness Auditing

**arXiv ID:** 2602.10265 | [PDF](https://arxiv.org/pdf/2602.10265v1)

**作者:** Marin Benčević `[一作]` (Josip Juraj Strossmayer University of Osijek), Irena Galić `[通讯]` (Josip Juraj Strossmayer University of Osijek)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用神经网络从黑色素镜图像中直接预测 Fitzpatrick 皮肤类型和 Individual Typology Angle（ITA），并将其应用于公开数据集进行皮肤色分布分析。

**💡 创新点**

创新点在于通过有色彩计量器（colorimeter）真实测量作为监督信号，训练出能够克服光照、相机管线和局部组织影响的深度学习模型，显著优于传统基于像素平均的估计方法。

**🔧 技术方法**

采用 EfficientNet-B0 作为主干网络，先在合成与真实 dermatoscopy/临床图像上预训练 Fitzpatrick 序数回归头，随后在 MSKCC 数据集上细调，用三通道 CIELAB 回归预测 ITA，损失函数为 CIE ΔE 1976 色差。

**📊 数据集**

主要训练和评估使用 MSKCC Skin Tone Labeling 数据集（64 受试者，4878 张图像，包含专家 Fitzpatrick 标签和三次色彩计量器测量）；预训练利用 PAD‑UFES‑20、SCIN、Fitzpatrick17k、MRA‑MIDAS、DDI、S‑SYNTH；推理阶段在 ISIC 2020（33,126 张）和 MILK10k（5,240 张）上评估皮肤色分布。

**📈 对比分析**

与人类众包标签和传统像素方法相比，模型在 Fitzpatrick 类型上的线性加权 Cohen's κ 为 52.98%（相对众包的 66.08% 较低），但误差仅为 0.84，近似类别内误差；ITA 的 ICC3 为 93.88%，远高于 K‑Means（43.64%）和 Patch‑Based（57.56%）等基线，接近色彩计量器重复测量的理论极限 98.38%。

**⚠️ 局限性**

局限性包括：仅在单中心、仅含良性病变的 MSKCC 数据集上验证，缺乏多设备、多地区和多病变类型的外部验证；ITA 仅在非病变皮肤上评估，未直接验证病变区域的泛化能力；模型在 Fitzpatrick I 与 II 的区分上仍存在混淆，影响极暗/极浅皮肤的准确标注。

---

## 500. Asymmetric Prompt Weighting for Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2602.11128 | [PDF](https://arxiv.org/pdf/2602.11128v1)

**作者:** Reinhard Heckel `[一作]` (Technical University of Munich), Christos Thramboulidis `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了针对可验证奖励（verifiable rewards）的强化学习中的不对称提示权重（asymmetric prompt weighting）策略，用以提高从零开始训练的语言模型在推理任务上的性能。

**💡 创新点**

创新点在于：1) 设计了四种不对称权重函数（Linear-R、Sqrt-R、Plateau-R、Uniform-R），在低成功率 regime 下显著提升梯度信号；2) 理论推导了在普通时间和有效时间（考虑低成功率下采样成本）下的最优权重，证明 Sqrt-R 在有效时间下最优；3) 通过实验验证不对称权重在从零开始 RL 时优于 GRPO、RLOO 等主流对称方法，而在已有高准确率的 post‑SFT 场景中不造成显著影响。

**🔧 技术方法**

技术主要包括：基于 REINFORCE / RLOO / GRPO 等策略梯度算法的改写，引入提示层权重；使用离线政策优化和 on‑policy 训练；利用 KL 正则化控制更新步长；以及在理论层面采用 ODE 与有效时间变换来分析学习动力学。

**📊 数据集**

使用的数据集包括：TinyZero（倒计时与乘法推理任务）和 GSM8K（数学推理任务）用于从零开始 RL；MATH 与 DAPO‑math 用于 post‑SFT RL；基模型分别为 Qwen2.5‑3B、Llama‑3.1‑8B、Llama‑3.2‑3B‑instruct 与 DeepSeek‑R1‑Distill‑Qwen‑1.5B。

**📈 对比分析**

实验对比方法：GRPO、RLOO、Uniform‑R 以及提出的不对称权重。结果显示：在 TinyZero 与 GSM8K 的从零开始 RL 中，不对称权重的 Pass@1 在 0.74–0.80 之间提升约 6%（相较于 0.74 的对称方法）；在 MATH 与 DAPO‑math 的 post‑SFT 场景中，四种方法表现相近，差异不显著。

**⚠️ 局限性**

局限性：1) 仅在二元奖励（0/1）下进行理论和实验，尚未全面验证连续奖励情形；2) 理论模型采用理想化的无参数策略优化，实际效果受模型参数化和近似误差影响；3) 计算成本高，实验耗时数百 H100 小时，限制了更大规模验证；4) 对如何在低成功率下动态分配采样预算的实际实现尚未深入探索。

---

## 501. A Unified Experimental Architecture for Informative Path Planning: from Simulation to Deployment with GuadalPlanner

**arXiv ID:** 2602.10702 | [PDF](https://arxiv.org/pdf/2602.10702v1)

**作者:** Alejandro Mendoza Barrionuevo `[一作]` (University of Sevilla), Samuel Yanes Luis `[通讯]` (University of Sevilla)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了GuadalPlanner统一实验架构，实现信息路径规划（IPP）算法与车辆控制解耦，使同一算法可在离散图环境的算法级仿真、SITL仿真及真实ASV部署中一致运行；

**💡 创新点**

首创将IPP决策层与车辆执行层分离的统一架构，提供标准化接口和可扩展性，并以开源方式提供完整实验平台，显著降低从仿真到真实部署的迁移成本；

**🔧 技术方法**

基于ROS2中间件、MAVLink/ArduPilot、MQTT通信、Python实现，并利用Docker封装SITL与ROS2节点，支持离散图环境、图像/传感器数据交互；

**📊 数据集**

使用Lake Mayor（Alamillo Park）湖泊的卫星图分辨率约25 m²的离散网格（30×49格）作为实验场景；同时构造GP模拟、TrashClean、OilSpill等多种离散环境，并在真实ASV上采集水质电导率等传感器数据；

**📈 对比分析**

通过在算法级仿真、SITL仿真和真实ASV上分别测试贪婪、期望改进（贝叶斯优化）和覆盖（Flooding）三种IPP策略，比较指标包括MSE、GP不确定性下降率、轨迹偏差；结果表明所有策略最终收敛至低MSE，贪婪最快降低局部不确定性，期望改进在全局均衡采样，Flooding实现完整覆盖但不具信息驱动；

**⚠️ 局限性**

局限性包括：依赖中心服务器（需网络连接）导致分布式部署受限；仅兼容MAVLink基控机，难以直接支持非MAVLink平台；仅支持二维离散图环境，无法直接扩展至连续或三维规划；

---

## 502. Interpretable Graph-Level Anomaly Detection via Contrast with Normal Prototypes

**arXiv ID:** 2602.10708 | [PDF](https://arxiv.org/pdf/2602.10708v1)

**作者:** Qiuran Zhao `[一作]` (Nanjing University), Xinpeng Li `[通讯]` (Nanjing University)

**通讯引用:** 8024 | [OpenAlex ID](https://openalex.org/A5100750156)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 ProtoGLAD，一种基于原型的无监督图层异常检测框架，能够在检测到异常图后给出与最近正常原型图的对比解释。

**💡 创新点**

创新点在于：①使用点集核迭代发现多重正常原型和聚类；②将异常得分定义为与最近正常聚类的相似度；③利用节点级相似度为异常图提供可解释的子结构差异。

**🔧 技术方法**

主要技术包括：改进的 Weisfeiler–Lehman 嵌入与 Isolation Kernel 的组合作为点集核；基于聚类的原型选择；节点-图相似度评分。

**📊 数据集**

使用 TUDataset 公开数据集（共八个）进行实验，并在人工合成数据上验证解释能力。

**📈 对比分析**

与六种无监督 GLAD 基线（WL-iForest、GLADC、GLocalKD、OCGTL、SIGNET、GLADPro）比较，ProtoGLAD 在 6/8 公开数据集上排名第一，平均排名仅为 1.25，整体检测性能优于或相当于现有方法。

**⚠️ 局限性**

局限性包括：依赖于预设的相似度阈值和增长率参数；对大规模图数据的计算复杂度相对较高；在极端噪声或异常模式高度多样化的场景下，原型发现可能不够稳健。

---

## 503. Exploring the impact of adaptive rewiring in Graph Neural Networks

**arXiv ID:** 2602.10754 | [PDF](https://arxiv.org/pdf/2602.10754v1)

**作者:** Charlotte Cambier van Nooten `[一作]` (Radboud University), Lucia Cavallaro `[通讯]` (Radboud University)

**通讯引用:** 415 | [OpenAlex ID](https://openalex.org/A5086236981)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在图神经网络中通过控制稀疏度和自适应重连（rewiring）来降低计算成本并提升性能，尤其针对电力网的 N‑1 故障分析任务；

**💡 创新点**

创新点在于提出了结合 Erdos‑Renyi 随机稀疏化与自适应重连（根据验证性能动态调整重连比例并结合早停机制）的稀疏化策略；

**🔧 技术方法**

采用的技术包括 Erdos‑Renyi 初始化稀疏权重、固定速率重连（SET）、自适应重连（动态 ζ）、GCN/GCNE、GIN/GINE 架构、早停与批归一化等；

**📊 数据集**

使用的实验数据集包括化学分子基准 MUTAG、PROTEINS 以及真实电力网的 N‑1 故障判定数据集；

**📈 对比分析**

与无稀疏化、固定重连、Dropout 等基线进行对比，结果显示在中等稀疏度下自适应重连可提升 GNN 准确率至 0.98–0.99（N‑1 数据集），并显著减少模型参数；

**⚠️ 局限性**

主要局限在于过度稀疏会导致性能下降，需要在不同任务和网络架构中进行稀疏度与重连参数的细致调优，且方法对极大图或高连通图的迁移效果尚未充分验证。

---

## 504. Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models

**arXiv ID:** 2602.10520 | [PDF](https://arxiv.org/pdf/2602.10520v1)

**作者:** Williams Jonathan `[一作]`, Tureci Esin `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在循环语言模型中对整个潜在推理轨迹进行奖励分配的强化学习框架 RLTT。

**💡 创新点**

创新点在于打破传统终点奖励瓶颈，实现轨迹级别的信用分配，使学习信号更密集、更符合模型内部多步推理过程。

**🔧 技术方法**

采用 REINFORCE 策略梯度、KL 正则化、循环权重策略以及 Ouro‑2.6B‑Thinking 的循环 Transformer 架构。

**📊 数据集**

使用数学推理数据集 MATH、AIME24、BeyondAIME、GSM8k，以及非数学推理数据集 ARC‑C、MMLU‑ST、GPQA、MBPP 进行训练与评估。

**📈 对比分析**

与传统 GRPO 直接比较，RLTT 在 MATH‑500、AIME24、BeyondAIME、GSM8k 的准确率分别提升 +14.4%、+16.6%、+10.0%、+34.3%，在数学任务上的平均提升约 18.8%，非数学任务平均提升约 6.6%。

**⚠️ 局限性**

局限性包括：仅适用于循环模型、需要保存每步的 log‑概率导致显著内存占用、训练时循环深度固定，无法动态调节推理步数，且对外部验证器完全不依赖。

---

## 505. RSHallu: Dual-Mode Hallucination Evaluation for Remote-Sensing Multimodal Large Language Models with Domain-Tailored Mitigation

**arXiv ID:** 2602.10799 | [PDF](https://arxiv.org/pdf/2602.10799v1)

**作者:** Zihui Zhou `[一作]` (Chongqing University), Weijia Jia `[通讯]` (Beijing Normal University)

**通讯引用:** 11809 | [OpenAlex ID](https://openalex.org/A5051803761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究遥感多模态大语言模型（RS-MLLM）中的幻觉现象，提出RS专用幻觉分类体系，构建RSHalluEval幻觉评估基准，并设计双模态评估流程；同时提供训练友好型的RSHalluShield数据集和无训练的插件式干预（解码时logit校正与RS感知提示）以降低幻觉率。

**💡 创新点**

① 引入图像层级幻觉（Image‑level hallucination），补充传统目标层级幻觉，适应遥感多尺度、多模态的特殊性；② 开发RSHalluEval和RSHalluCheck双模评估机制，实现云端高精度与本地低成本可复现评估；③ 提供RSHalluShield专属数据集用于训练，显著提升幻觉抑制；④ 设计无训练的解码‑时logit校正与RS-aware提示两种插件式方法，兼顾效果与部署成本。

**🔧 技术方法**

利用大型多模态语言模型（LLaVA‑1.5、Qwen‑2‑VL、GeoChat、VHM 等），结合链式思维（CoT）评估；通过LoRA微调实现RSHalluShield的训练；采用注意力权重与中间层logit加权、阈值筛选实现解码时logit校正；使用多轮提示构建RS-aware提示策略。

**📊 数据集**

遥感图像与文本基础数据集：RSIEval、RSITMD、UCM‑Captions、Sydney‑Captions；基准与检查集：RSHalluEval（2,023 QA）、RSHalluCheck（15,396 QA）及RSHalluShield（30,000 QA）；下游任务评估：RSVQA‑LR、RSVQA‑HR、DIOR‑RSVG。

**📈 对比分析**

对比多款RS‑MLLM与通用MLLM（LLaVA‑1.5、mPLUG‑Owl3、Qwen‑2‑VL、GeoChat、LHRS‑Bot、VHM）在RSHalluEval上进行专家评估与自动CoT评估，显示最高可提升21.63%幻觉无误率；在RSVQA和RSVG任务上，微调后Qwen‑2‑VL‑HF在多项指标上位居前列或实现显著提升；插件式方法亦能在不训练模型的情况下显著降低幻觉。

**⚠️ 局限性**

① 评估与干预主要针对文本答案，可能无法完全捕捉长文本中的细微幻觉；② RSHalluEval与RSHalluShield构造依赖人工与自动标注，存在标注偏差与类别不全；③ 解决方案对大模型依赖度高，部署时需考虑算力与隐私；④ 本研究聚焦遥感图像与文本，未覆盖视频、时序或多模态更复杂场景；⑤ 部分本地检查器在对象属性与存在类别的识别上仍低于云端API，需进一步优化。

---

## 506. Bounding the Average Move Structure Query for Faster and Smaller RLBWT Permutations

**arXiv ID:** 2602.11029 | [PDF](https://arxiv.org/pdf/2602.11029v1)

**作者:** Nathaniel K. Brown `[一作]` (Johns Hopkins University), Ben Langmead `[通讯]` (Johns Hopkins University)

**通讯引用:** 125987 | [OpenAlex ID](https://openalex.org/A5009556658)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种长度截断（length capping）拆分策略，构造移动结构，实现平均常数查询时间、O(r) 构造时间并显著降低空间占用。

**💡 创新点**

用最简单的截断拆分替代原有的平衡拆分，理论证明平均 O(1) 查询、最坏 O(log(n/r)) 查询、以及更紧凑的 O(r log r + r log(n/r)) 位表示。

**🔧 技术方法**

利用移动结构理论、BWT/运行长度 BWT、基于循环排列的 ϕ 与 ϕ⁻¹、相对/绝对位置压缩、指数搜索与位打包矩阵实现。

**📊 数据集**

在人类 chr19 染色体的 16、32、64、128、256、512、1000 条等位基因拼接集合（总长度 59M 字符）上进行实验。

**📈 对比分析**

与传统平衡拆分及未拆分实现对比，实验显示长度截断在空间上可缩减约 40%–46%，查询时间提升 10%–20%，在大规模数据集上保持最快或最小的结构。

**⚠️ 局限性**

限制：对 ϕ 的平衡实现仍有限制；未解决全排列枚举的最佳方法；实验中未观察到理论上 O(r) 构造时间优势；极大输入仍可能出现内存瓶颈。

---

## 507. Personalized PageRank Estimation in Undirected Graphs

**arXiv ID:** 2602.10843 | [PDF](https://arxiv.org/pdf/2602.10843v1)

**作者:** Christian Bertram `[一作]` (University of Copenhagen), Mads Vestergaard Jensen `[通讯]` (University of Copenhagen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对无向图中个人化 PageRank（PPR）估计问题进行完整的复杂度表征，给出了在不同查询模型（仅邻接表、加上随机节点、按度数排序邻接查询、边存在性查询等）下，单对、单源、单目标和单节点估计的最优时间上界与下界（含对数因子）。

**💡 创新点**

创新点在于：①提出利用无向图的可逆性（d(u)π(u,v)=d(v)π(v,u)）构造新的硬实例和上界算法；②结合反向 Monte Carlo、局部推送（push）、随机局部推送、双向估计等技术，得到最优的时间复杂度；③在有额外查询（随机节点、按度排序查询、边查询）时，进一步降低复杂度并消除对数因子。

**🔧 技术方法**

核心技术包括：α‑折扣随机游走、反向 Monte Carlo、局部推送与随机局部推送、双向估计、分区邻居（低度/高度邻居）处理、swap 变换构造下界、图访问模型的查询优化、概率分析与 Chernoff/Chebyshev 边界。

**📊 数据集**

本文不使用实际数据集，而是纯理论分析。所有结果均基于无向图的规模参数 n、m、平均度 d、阈值 δ，并通过构造硬实例演示下界。

**📈 对比分析**

通过严格的上界与下界证明，所有结果互相匹配（仅差多项式对数因子），从而完成了复杂度的完全表征。相比于之前仅在有向图或缺少某些查询类型下的结果，本文在无向图且任意查询组合下实现了最优的时间复杂度。

**⚠️ 局限性**

局限性包括：①复杂度结果带有隐藏的 polylog(n,1/δ) 因子；②仅覆盖无向图，仍未完成有向图的完整对比；③算法实现可能依赖于高阶的查询支持（如随机节点、度数排序查询），在实际应用中这类查询成本可能不低。

---

## 508. RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics, Spatial-Physical Intelligence, and Autonomous Evaluation

**arXiv ID:** 2602.10980 | [PDF](https://arxiv.org/pdf/2602.10980v1)

**作者:** Yuhao Chen `[一作]` (Sun Yat-sen University), Guangrun Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3022 | [OpenAlex ID](https://openalex.org/A5052611320)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了RADAR benchmark，用以在真实环境中评估Vision‑Language‑Action (VLA) 模型的通用性与鲁棒性，包含多种物理动态、空间推理任务和完全自动化的3D评估流程。

**💡 创新点**

创新点在于：①将真实世界的光照、背景、初始姿态、传感器噪声等多维动态显式化；②设计专门的空间推理任务，要求模型理解3D几何和相对关系；③构建全自动化的客户端‑服务器‑工作器架构，使用高精度3D重建实现无人工干预的评估。

**🔧 技术方法**

技术包括：机器人实时控制与闭环视觉；多视角RGB‑D传感与三维重建；3D IoU、位置误差等几何评估指标；使用大型语言模型对指令进行多样化改写；基于云端或本地推理的异步推断架构。

**📊 数据集**

数据集为RADAR自身收集的真实机器人实验数据，包含四个任务分割（SingleBlock、TwoBlock、ThreeBlock、Spatial）以及多种干扰条件（光照、背景、传感器噪声、语言改写等）。

**📈 对比分析**

通过对比5个主流VLA模型（π_0、π_0‑FAST、π_0.5、RDT、OpenVLA），在RADAR上进行统一训练与评估。实验表明，在无扰动条件下部分模型仍能取得一定成功率，但一旦加入轻微的物理或语言扰动，3D IoU从0.26降至0.07，模型整体表现急剧下降，凸显其对动态环境的脆弱性。

**⚠️ 局限性**

限制主要包括：①评估仍依赖特定硬件平台，难以完全泛化；②当前任务设计对深度感知依赖较高，易受传感器噪声影响；③缺乏针对不同语言细粒度的指令解析机制，导致模型对语言变化不敏感；④对高度动态、连续交互的场景尚未覆盖。

---

## 509. EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling

**arXiv ID:** 2602.10811 | [PDF](https://arxiv.org/pdf/2602.10811v1)

**作者:** Mingyang Liu `[一作]` (Taobao and Tmall Group of Alibaba), Xinyang Chen `[通讯]` (Taobao and Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种高效可扩展的 Transformer 结构（EST），专门用于工业级 CTR 预测任务，能够在保持毫秒级低延迟的前提下实现对用户行为、候选信息等多种异构特征的全统一建模。

**💡 创新点**

核心创新在于：①轻量交叉注意力（LCA）消除自注意力中的冗余交互，仅保留对非行为特征与行为序列最关键的交叉交互；②内容稀疏注意力（CSA）利用预训练的内容特征相似度做稀疏注意力，引导行为序列内部的高信号交互；③结合两者实现全统一建模，从而打破传统早期聚合导致的信息瓶颈。

**🔧 技术方法**

技术手段包括 Transformer 核心、LCA 与 CSA 两大模块、预训练的多模态内容特征（图像、文本）作为辅助，训练时采用稀疏+稠密参数分离策略，部署时利用用户-候选解耦计算降低延迟。

**📊 数据集**

实验使用阿里巴巴 Taobao 生产日志：1 亿元+ 6 天用户曝光样本（包含 300 长度短期行为与 5,000 长期行为），并在 3.5 亿元扩展数据集上进一步验证。

**📈 对比分析**

与多种基线（MLP、AutoInt、DCNv2、RankMixer、HiFormer、MTGR、OneTrans 及其变种）在 AUC/GAUC 进行对比，EST 在 GAUC 上提升约 0.99%（AUC +0.87%），且在相同 GFLOPs/参数规模下表现更优；此外，展示了计算量与 GAUC 的幂律 scaling 关系。

**⚠️ 局限性**

局限性包括：①模型仍依赖大规模稀疏嵌入表，训练成本高；②对候选行为序列长度有上限，极长序列仍需进一步压缩；③对内容特征的效果受预训练语料与映射质量限制，若内容分布变化可能导致性能退化。

---

## 510. Hyperspectral Smoke Segmentation via Mixture of Prototypes

**arXiv ID:** 2602.10858 | [PDF](https://arxiv.org/pdf/2602.10858v1)

**作者:** Lujian Yao `[一作]` (East China University of Science and Technology), Yuhan Xu `[通讯]` (East China University of Science and Technology)

**通讯引用:** 143 | [OpenAlex ID](https://openalex.org/A5076420187)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出使用高光谱影像进行烟雾分割，并首次构建高光谱烟雾分割数据集 HSSDataset 与多光谱扩展数据集 MSSDataset。

**💡 创新点**

核心创新是 Mixture of Prototypes（MoP）网络，它通过光谱隔离的 Band Split、基于原型的光谱表征以及双层路由（Proto‑Router 与 Feature‑Router）实现空间感知的自适应波段加权。

**🔧 技术方法**

使用了组卷积实现波段隔离、原型学习与对比损失、Sinkhorn 算法进行像素‑原型匹配、双层路由网络、混合专家机制以及多光谱融合技术。

**📊 数据集**

主要数据集包括：HSSDataset（1,007 例，25 维波段 600–974 nm，来自 18,000 帧 20 场景）和 MSSDataset（200 例 RGB‑IR 4 通道）。

**📈 对比分析**

在高光谱与多光谱两种场景下，与多种 CNN 与 Transformer 基线（PSPNet、DeepLabV3+、SegFormer 等）进行对比，MoP 在所有尺度上均取得最高 F1/mIoU，尤其在小烟雾与云干扰场景提升约 2–5 %，参数仅 4.4 M，FLOPs 7.3 G。

**⚠️ 局限性**

局限性在于 Band Split 仅使用简单的组卷积，双分支融合采用基础的加/平均策略，未尝试更高级的分解与融合方法，可能限制进一步性能提升。

---

## 511. Flow Matching with Uncertainty Quantification and Guidance

**arXiv ID:** 2602.10326 | [PDF](https://arxiv.org/pdf/2602.10326v1)

**作者:** Juyeop Han `[一作]` (Massachusetts Institute of Technology), Sertac Karaman `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 11990 | [OpenAlex ID](https://openalex.org/A5081073767)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 UA-Flow，一种在流匹配中同时预测速度场和异方差不确定性的模型，并利用该不确定性实现样本可靠性评估与自适应引导；

**💡 创新点**

创新点在于把不确定性直接嵌入速度场并通过确定性 ODE 动力学传播，生成可局部化的样本级不确定性，并基于此设计两种自适应指导策略 U-CG 与 U-CFG；

**🔧 技术方法**

核心技术包括流匹配框架、异方差回归（高斯负对数似然）、Euler 离散化、Hutchinson 估计器、基于不确定性的伪似然引导与自适应 CFG；

**📊 数据集**

在 CIFAR‑10、ImageNet‑128 与 ImageNet‑256 三个数据集上进行实验；

**📈 对比分析**

与 BayesDiff、AU、GenUnc 等基线比较，UA-Flow 在过滤高不确定性样本时能显著降低 FID、提升精度，且在指导实验中实现更好的 FID 与精度/召回权衡，且计算开销低于现有方法；

**⚠️ 局限性**

局限性包括仅建模随机不确定性（无贝叶斯/模型不确定性），对非确定性流模型适用性有限，以及在极端指导或高分辨率任务中仍需进一步验证稳定性。

---

## 512. OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization

**arXiv ID:** 2602.10635 | [PDF](https://arxiv.org/pdf/2602.10635v1)

**作者:** Keane Ong `[一作]` (Massachusetts Institute of Technology), Paul Pu Liang `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 7498 | [OpenAlex ID](https://openalex.org/A5086233510)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 HARPO 方法，利用该方法训练 OmniSapiens 2.0，构建统一的社交行为处理模型；在多任务环境中实现对不同行为任务的平衡学习。

**💡 创新点**

创新点在于：1) 异质性感知的相对策略优化（HARPO），通过优势调制（几何均值中心化 + 惯性平滑）实现对任务与样本贡献的动态平衡；2) 结合无评判者（critic‑free）RL 以保持模型的推理能力。

**🔧 技术方法**

技术细节：基于 GRPO 的 on‑policy RL 框架；使用 PPO 风格的信赖域目标；采用几何均值参考对优势进行中心化调制；对调制因子进行惯性（乘法）平滑；使用 Qwen 2.5 Omni‑7B 作为 LLM 基座。

**📊 数据集**

数据集：Human Behavior Atlas（10 个行为任务，共约 10 万样本），以及 AUT（自闭症行为识别）和 SER（情感识别）用于零样本迁移评估。

**📈 对比分析**

比较方式：在同一基准和奖励设定下，将 HARPO 与 GRPO、RLOO、RE++、GPG 等最新 critic‑free RL 方法及 Gemma‑3、Qwen 系列 LLM 进行对比。OmniSapiens 2.0 在 10 个任务的平均排名为 1.20，提升幅度最高可达 +16.85%（多任务）和 +9.37%（held‑out）。在 AUT 和 SER 的零样本转移上也取得最佳表现，分别比竞争模型高 9% 与 16%。

**⚠️ 局限性**

局限性：1) 需要手工设计奖励函数，可能对新任务不易迁移；2) 对极少样本或高度不平衡任务的优势调制效果仍有待验证；3) 代码与模型仍未公开，复现性受限；4) 在某些任务中，过度调制可能导致训练不稳定。

---

## 513. Driving Reaction Trajectories via Latent Flow Matching

**arXiv ID:** 2602.10476 | [PDF](https://arxiv.org/pdf/2602.10476v1)

**作者:** Yili Shen `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 12579 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 LatentRxnFlow，一种利用连续潜在流匹配（Conditional Flow Matching）对化学反应进行建模的预测框架，直接从反应物-产物对学习反应动力学并生成完整的潜在轨迹。

**💡 创新点**

创新点在于：①将反应过程建模为连续潜在轨迹，避免传统离散编辑步骤的错误累积；②在不需要机制标签或中间标注的情况下通过潜在流匹配学习时间相关的向量场；③利用轨迹几何特征提供无监督的不确定性评估，并通过轨迹门控推理纠正预测错误。

**🔧 技术方法**

采用图自编码器（GAE）进行结构编码/解码，Conditional Flow Matching 学习潜在向量场，ODE求解器（RK4）实现轨迹集成，FiLM 注入条件信息，以及基于潜在轨迹的几何描述（路径效率、曲率、最小对齐、动能）进行诊断。

**📊 数据集**

主要使用公开 USPTO 反应数据集（USPTO‑480k、USPTO‑MIT、USPTO‑50k）进行训练与评估。

**📈 对比分析**

与 Seq2Seq、Graph‑to‑Graph、MEGAN、GTPN、FlowER、NERF 等基线对比，LatentRxnFlow 在 USPTO‑480k 上 Top‑1 达到 90.2%，与 NERF 相当，且在推理速度（8.5 ms）和轨迹可解释性方面显著优于离散程序模型。

**⚠️ 局限性**

局限性包括：①轨迹仍可能出现“过冲/振荡”需门控推理修正；②对极端罕见反应或高能垒反应的泛化仍有限；③缺乏显式的终止或稳定器机制，轨迹终点不一定是收敛的稳定点；④模型训练仍需大规模反应数据，且对条件编码的鲁棒性有待进一步验证。

---

## 514. Efficient Computation of Maximum Flexi-Clique in Networks

**arXiv ID:** 2602.10459 | [PDF](https://arxiv.org/pdf/2602.10459v1)

**作者:** Song Kim `[一作]` (UNIST), Jungeun Kim `[通讯]` (Inha University)

**通讯引用:** 6155 | [OpenAlex ID](https://openalex.org/A5100452997)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了新的“flexi‑clique”连通子图模型，并证明其最大化问题是NP‑hard；随后设计了两种求解算法：基于核心分解和连通性剪枝的快速启发式 Flexi‑Prune，以及利用非遗传性连通性保留分支与多重剪枝的精确 Branch‑and‑Bound。

**💡 创新点**

创新点包括①将传统模型的线性阈值替换为子线性度数阈值 ⌊|H|^τ⌋，实现大小自适应连通度；②针对非遗传性和连通性问题提出了 Connectivity‑Preserving Branching 框架；③提出六种高效剪枝规则（度数、直径、追随者、启发式）和动态连通性维护机制；④系统性地理论分析与实验验证。

**🔧 技术方法**

技术手段主要有核心分解、动态连通性结构（Holm‑de Lichtenberg‑Thorup 链表/Link‑Cut Tree）、度数和直径下界推导、追随者分析、启发式预估、排序的分支顺序，以及基于分支限界的剪枝。

**📊 数据集**

实验使用了 9 个真实网络（Karate、Polbooks、Football、Polblogs、Erdős、PGP、Amazon、DBLP、Florida）以及 LFR 合成网络，覆盖社交、合作、通信等多种场景。

**📈 对比分析**

与基线启发式、最大团、k‑plex、γ‑准团等方法对比；Flexi‑Prune 在大多数图上取得近 90% 以上的近似率且运行时间仅为基线的数倍；Flexi‑BB 在中等规模图（如 DBLP、Florida）能够在可接受时间内求得最优解；实验显示两种算法均显著优于传统模型。

**⚠️ 局限性**

限制主要包括①模型非遗传性导致子图枚举复杂度高，Branch‑and‑Bound 对图规模仍有限制；②参数 τ 的选择依赖经验，过大或过小都会影响结果；③在极稠密或大规模图上，动态连通性维护与剪枝仍可能产生瓶颈。

---

## 515. New Algorithms and Hardness Results for Robust Satisfiability of (Promise) CSPs

**arXiv ID:** 2602.10368 | [PDF](https://arxiv.org/pdf/2602.10368v1)

**作者:** Joshua Brakensiek `[一作]` (University of California Berkeley), Stanislav Živný `[通讯]` (University of Oxford)

**通讯引用:** 1099 | [OpenAlex ID](https://openalex.org/A5067656814)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文研究了承诺约束满足问题（PCSP）的鲁棒近似算法，证明基本SDP的完整性比与鲁棒性等价，并给出了在UGC假设下鲁棒算法的可转化与多态子映射的鲁棒可解性保持性。

**💡 创新点**

创新点在于将Charikar–Makarychev–Makarychev（CMM）算法与Brown‑Cohen‑Raghavendra（BCR）算法的分析改进至更紧的鲁棒误差界（O(√ε)或O(√ε log(1/ε))），并将鲁棒可解性推广到非布尔域的可分离PCSP（如包含Plurality多态子），同时证明等价关系的加入不破坏鲁棒性。

**🔧 技术方法**

主要技术包括：基本SDP求解与高维高斯随机化取整、噪声算子与Hermite多项式近似、多态子与准随机性定义、超卷积与Bonami-Bonami引理、凸体分离与超平面分离、BCR算法框架与近似多态子改造、以及对等价关系的均匀化与误差控制。

**📊 数据集**

本研究为理论分析，不涉及实验数据集。

**📈 对比分析**

通过理论分析与概率界定，改进算法在满足约束的预期比例上可达1−O(√ε)（布尔域为O(√ε)，非布尔域为O(√ε log(1/ε))），与已有CMM、BCR等算法相比，误差界更紧且适用范围更广，鲁棒性能理论上优于现有方法。

**⚠️ 局限性**

局限性包括：结果依赖UGC假设；在非布尔域鲁棒性需乘以对数因子，误差扩展为O(√ε log(1/ε))；等价关系加入后鲁棒误差会按ε^{1/6}放大；算法复杂度高，缺乏实验验证；并且对某些PCSP模板（如Horn‑SAT）仍无鲁棒可解性保证。

---

## 516. GenFaceUI: Meta-Design of Generative Personalized Facial Expression Interfaces for Intelligent Agents

**arXiv ID:** 2602.11055 | [PDF](https://arxiv.org/pdf/2602.11055v1)

**作者:** Yate Ge `[一作]` (Tongji University), Xiaohua Sun `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 828 | [OpenAlex ID](https://openalex.org/A5100530166)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了生成式个性化面部表情接口（GPFEI）框架，并实现了Meta-Design工具GenFaceUI，随后通过对12名设计师的定性研究评估了该框架和工具在可控性、一致性与创意支持方面的效果。

**💡 创新点**

创新点在于将Meta-Design视角与面部表情生成结合，构建了面部模板、语义标签、约束规则与上下文映射的四层结构，首次为AI驱动的实时面部表情提供了可控、可调的设计规范，并通过工具实现了设计者对生成空间的可视化和迭代。

**🔧 技术方法**

技术上主要采用大型语言模型（LLM）进行面部参数与图形的生成，结合SVG模板、语义标签系统、规则编辑器与上下文映射引擎，前端使用React实现可视化编辑与实时预览，后端使用Node.js负责提示合成与与LLM的交互。

**📊 数据集**

在实验中未使用公开数据集，而是让设计师基于自身经验和任务描述手工构造面部模板与规则；LLM根据这些规则生成SVG面部表达，整个过程完全基于用户输入与模型推理，无需预训练数据集。

**📈 对比分析**

评估方式为定性分析（访谈、思考实验、系统日志）和任务完成度统计；研究显示设计师对工具的可控性和一致性评价普遍较高，但缺乏量化性能指标；相比传统手工设计，工具显著降低了创作门槛并提升了表达多样性。

**⚠️ 局限性**

主要局限包括：①缺乏专业级细节控制与动画过渡，无法满足高保真设计需求；②LLM生成的结果不稳定，导致设计者难以预测与调试；③缺乏对实时用户交互与情境适配的评估，尚未验证对用户体验、信任与情感交互的影响；④工具的可视化与反馈机制不足，限制了设计者对生成过程的理解与迭代效率。

---

## 517. Beyond Permissions: An Empirical Static Analysis of Privacy and Security Risks in Children-Oriented and General-Audience Mobile Apps for Gaming

**arXiv ID:** 2602.10877 | [PDF](https://arxiv.org/pdf/2602.10877v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 518. C^2ROPE: Causal Continuous Rotary Positional Encoding for 3D Large Multimodal-Models Reasoning

**arXiv ID:** 2602.10551 | [PDF](https://arxiv.org/pdf/2602.10551v1)

**作者:** Guanting Ye `[一作]` (University of Macau), Ka-Veng Yuen `[通讯]` (University of Macau)

**通讯引用:** 10183 | [OpenAlex ID](https://openalex.org/A5083222672)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在3D大规模多模态模型中改进位置编码，使视觉特征更好地保持空间连续性和因果关系，从而提升视觉问题回答与场景推理能力。

**💡 创新点**

提出 C²RoPE，包含三元时空连续位置编码和基于 Chebyshev 距离的因果掩码，解决传统 RoPE 在3D视觉任务中的“空间局部性损失”和“图像令牌被忽略”两大缺陷。

**🔧 技术方法**

技术手段：RoPE 的频率分配策略、三元（时间、x、y）位置索引、Chebyshev 距离因果掩码、与 LLaVA‑3D 的结合、训练细化两阶段方法。

**📊 数据集**

使用 ScanQA、SQA3D（包含 ScanRefer）等公开3D视觉问答与推理基准进行评测；实验还覆盖多视角图像输入与 16 视角配置。

**📈 对比分析**

与基线 LLaVA‑3D 及其他前沿 3D LMMs 对比，C²RoPE 在 ScanQA 上提升 EM@1 4.3、B‑4 8.5、MET 13.4、RGE 2.5、CIDEr 18.1；在 SQA3D 上提升 EM@1 与 EM@R 各 1.2；在多模态问答与推理任务中明显优于传统 RoPE、CCA/MCA、以及部分 2D LLMs，达到或逼近当前最佳 3D LMM 结果。

**⚠️ 局限性**

仍然依赖于 LLaVA‑3D 的点云嵌入方式，无法提供更丰富的3D几何信息；对极大视角序列或复杂场景下的长序列注意力衰减问题尚未完全解决；在某些专用3D LMM（如 ChatScene、Ross3D）上表现仍不如其顶尖模型。

---

## 519. VideoSTF: Stress-Testing Output Repetition in Video Large Language Models

**arXiv ID:** 2602.10639 | [PDF](https://arxiv.org/pdf/2602.10639v1)

**作者:** Yuxin Cao `[一作]` (National University of Singapore), Jin Song Dong `[通讯]` (National University of Singapore)

**通讯引用:** 6636 | [OpenAlex ID](https://openalex.org/A5085067496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个评估框架，系统测量和压力测试视频大语言模型在视频输入下的输出重复行为。

**💡 创新点**

首次提出专门针对视频语言生成的输出重复度量体系（RR、RI、IE）和标准化的测试集与可控时间扰动库，并展示其对模型鲁棒性和安全性的影响。

**🔧 技术方法**

使用n-gram重复率、重复强度和信息熵三种指标，结合视频编码、投影到语言模型空间、以及各种时间变换（插入、删除、替换、反转、洗牌）。

**📊 数据集**

采用10,000条来自LLaVA‑Video‑178K、NeXT‑QA、ActivityNetQA、LLaVA‑Hound等公开数据集的视频作为基准。

**📈 对比分析**

与10种主流VideoLLM（如LLaVA‑Video、LLaVA‑NeXT、VideoLLaMA2、ShareGPT4Video等）对比，实验显示多模型均存在70%以上的重复率；时间扰动可将重复率提升至90%+，并可在黑盒查询中以平均15次内诱发重复，验证了其高风险性。

**⚠️ 局限性**

局限性包括：未提出有效的抑制重复的补救措施；仅评估了n-gram级别的重复，未深入探究语义层面的循环；时间扰动仅覆盖有限几类变换，其他可能的攻击手段未覆盖；且实验集中在10种模型，可能缺乏对更大规模模型或多模态架构的泛化。

---

## 520. When Gradient Clipping Becomes a Control Mechanism for Differential Privacy in Deep Learning

**arXiv ID:** 2602.10584 | [PDF](https://arxiv.org/pdf/2602.10584v1)

**作者:** Mohammad Partohaghighi `[一作]` (University of California), YangQuan Chen `[通讯]` (University of California)

**通讯引用:** 49985 | [OpenAlex ID](https://openalex.org/A5100715957)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于权重谱诊断的闭环控制算法WW-DP-SGD，用于在差分隐私训练中自适应调节梯度裁剪阈值；

**💡 创新点**

创新点在于将裁剪阈值视为闭环控制问题，通过仅使用模型权重的重尾谱指数作为反馈信号，构造对数域饱和控制器实现无额外隐私开销的自适应裁剪；

**🔧 技术方法**

采用WeightWatcher式重尾谱估计、指数移动平均平滑、饱和对数域控制器以及Poisson子采样的DP-SGD；

**📊 数据集**

在MNIST、EMNIST、CIFAR-10/100、ImageNet-100以及UCI Adult、Heart、Vehicle Energy Dataset等视觉与表格数据集上进行实验；

**📈 对比分析**

与固定裁剪的DP-SGD、自动裁剪、DP-PSAC、Bounded Adaptive、AdaDPIGU等多种自适应裁剪与优化基线进行比较，结果表明WW-DP-SGD在相同隐私预算下实现更高准确率/AUC或更低RMSE，且在非IID标签偏移和大模型场景下性能优势更为明显；

**⚠️ 局限性**

局限包括对谱指数作为健康信号的经验依赖、对探测层与尾部拟合参数的敏感性、控制器超参数调优需求、谱探测的计算开销，以及仅在标准Poisson子采样与匹配噪声比例下保持隐私一致性。

---

## 521. Interpretable Attention-Based Multi-Agent PPO for Latency Spike Resolution in 6G RAN Slicing

**arXiv ID:** 2602.11076 | [PDF](https://arxiv.org/pdf/2602.11076v1)

**作者:** Kavan Fatehi `[一作]` (University of York), Radu Calinescu `[通讯]` (University of York)

**通讯引用:** 4162 | [OpenAlex ID](https://openalex.org/A5011374280)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于注意力的多智能体PPO框架（AE-MAPPO），用于在6G RAN切片中快速诊断和解决突发延迟峰值，并在决策过程中实时提供解释。

**💡 创新点**

创新点包括：①将语义、时序、跨切片、置信度、反事实以及元控制器六种注意力机制嵌入到策略网络，形成可解释的决策过程；②将解释性目标直接纳入多目标优化，使解释与性能协同优化；③三阶段时序调度（预测、反应、跨切片优化）实现跨时间尺度的自适应资源分配；④零成本、实时可视化解释，提升运维信任。

**🔧 技术方法**

技术手段主要是：多智能体近端策略优化（MAPPO）+注意力机制；联合奖励函数包含性能与解释性；基于O‑RAN架构的三阶段资源分配；强化学习中的策略梯度、熵正则化与梯度对齐等。

**📊 数据集**

使用了基于6G RAN切片的真实感仿真数据（包含URLLC、eMBB、mMTC三种切片的非平稳Poisson流量、信道状态等），并在该仿真环境下进行URILC延迟峰值案例研究。

**📈 对比分析**

与传统手工故障排查、黑盒DRL及后置解释（如SHAP）等基线相比，AE-MAPPO在18 ms内完成资源重分配，将URLLC延迟恢复至0.98 ms、可靠率99.9999%，并将故障排查时间缩短93%；在多维性能指标（延迟、可靠性、吞吐、能耗、解释性）上达到94.2%性能与89%解释性，优于MAPPO（64%/15%）和DRL+SHAP（52%/45%）。

**⚠️ 局限性**

局限性包括：在单小区仿真环境下验证，跨小区或大规模网络的可扩展性尚未评估；对能耗与能源效率的考虑相对薄弱；部分注意力约束可能导致在极端情况下性能略逊于纯黑盒DRL；未来工作需在多小区、能耗优化、联邦学习等方向进一步验证。

---

## 522. RISE: Self-Improving Robot Policy with Compositional World Model

**arXiv ID:** 2602.11075 | [PDF](https://arxiv.org/pdf/2602.11075v1)

**作者:** Jiazhi Yang `[一作]` (Chinese University of Hong Kong), Hongyang Li `[通讯]` (University of Hong Kong)

**通讯引用:** 3234 | [OpenAlex ID](https://openalex.org/A5100450555)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一套基于想象的机器人强化学习框架，在物理环境中通过可控的世界模型（包括动力学预测和价值评估）生成大规模想象 rollouts，利用优势估计实现在线自我改进。

**💡 创新点**

创新点包括：①将世界模型拆解为可控动力学模型和价值评估模型，分别使用最适合的架构；②引入任务中心化批处理提升动作可控性；③在价值模型中融合进度估计与TD学习，提供稠密且失败敏感的奖励信号；④通过优势条件化训练和 EMA 更新实现完全在想象空间进行的在线强化学习。

**🔧 技术方法**

使用的技术包括：大规模视频扩散模型（Genie Envisioner/GE）进行可控动力学预测；任务中心化批处理策略；进度估计与 TD 学习构建价值模型；优势条件化训练与概率推断式策略更新；EMA 参数融合；VLA（π_0.5）基础模型与动作编码。

**📊 数据集**

使用的数据集包括：Agibot World、Galaxea 用于预训练与微调动力学模型；Horizon Robotics 与自身实验室收集的专家演示、DAgger、成功/失败轨迹；三项真实任务（Dynamic Brick Sorting、Backpack Packing、Box Closing）作为评估基准；Bridge 数据集用于对比动态生成质量。

**📈 对比分析**

与基线方法（π_0.5、π_0.5+DAgger、π_0.5+PPO、π_0.5+DSRL、RECAP）在三项任务上进行比较。结果显示本方法在成功率上分别提升至85%、95%、85%，比对齐基线平均提升约+40%至+45%；整体在成功率和分数上显著优于所有 RL/IL 基线。

**⚠️ 局限性**

限制主要有：①想象模型仍存在物理不一致的情况，难以完全覆盖真实世界的动态；②离线与在线数据比例需要手工调节，过多或过少都会影响性能；③高保真世界模型训练成本高，计算资源消耗大；④模型对罕见情景的泛化能力有限，需要进一步的不确定性估计和物理约束。

---

